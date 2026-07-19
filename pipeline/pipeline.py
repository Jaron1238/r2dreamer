#!/usr/bin/env python3

import argparse
import io
import math
import mmap as mmap_module
import os
import resource

import cv2
cv2.setNumThreads(0)  # let OpenCV manage its own threads on GitHub CI

import contextlib
import gc
import ctypes
import hashlib
import logging
import multiprocessing
import queue
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytesseract
import requests
from PIL import Image
from scipy.interpolate import interp1d
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".MP4", ".MOV"}
DEPTH_PATTERNS = ["*_depth.png", "*_depth.jpg", "*_depth.exr", "*.depth.png", "*depth*.png"]
TESS_FULL_FRAME = "--oem 0 --psm 11"
TESS_SINGLE_LINE = "--oem 0 --psm 7 -c tessedit_char_whitelist=0123456789.,V"

# GitHub Actions 6-hour wall-clock limit; shut down 30 min early so the
# final HF upload can finish before the runner is killed (exit code 137).
GRACEFUL_SHUTDOWN_SECONDS: float = 5.5 * 3600  # 5 h 30 min


@dataclass
class PipelineConfig:
    csv_path: str = "pipeline/fpv_videos_augmented_v2.csv"
    output_root: str = "output"
    tmp_root: str = "tmp_downloads"
    resume_file: str = "processed_urls.txt"
    segment_done_file: str = "segments_done.txt"
    log_file: str = "pipeline.log"
    hf_repo_id: str = "jaron12/fpv-dataset"
    hf_upload_threshold_bytes: int = int(3.0 * 1024 ** 3)  # must be < runner RAM (7 GB)
    upload_workers: int = 1  # 2 upload threads + processing saturates 2 vCPUs
    n_workers: int = 1          # GitHub standard runner: 2 vCPUs / 7 GB RAM
    target_fps: float = 60.0
    out_h: int = 288
    out_w: int = 512  # matches RSSM encoder resolution (img_height=288, img_width=512 → Landscape 512×288)
    # SafetyNet resolution — must match model.safety_img_height/width in configs.yaml
    safety_h: int = 720
    safety_w: int = 1280
    depth_h: int = 720
    depth_w: int = 1280
    pack_chunk_gb: float = 1.0  # keep peak RAM usage within CI runner budget
    takeoff_fps: float = 15.0
    takeoff_res: int = 128
    takeoff_max_seconds: float = 30.0
    takeoff_flow_threshold: float = 2.5
    takeoff_sustain_frames: int = 30
    scene_threshold: float = 24.0
    min_scene_len_frames: int = 256
    osd_scan_frames: int = 5
    osd_scale: float = 0.5
    osd_roi_change_threshold: float = 4.0
    stick_corner_fraction: float = 0.25
    stick_flow_ratio_threshold: float = 0.35
    stick_flow_res: int = 256
    crash_flow_threshold: float = 3.2
    crash_brightness_drop: float = 25.0
    crash_sharpness_drop: float = 60.0
    near_miss_flow_threshold: float = 2.0
    # Crash / safety dataset
    hf_crash_repo_id: str = "jaron12/fpv-crash-dataset"
    crash_context_frames: int = 30   # frames BEFORE crash to save as crash=1.0
    crash_upload_threshold_bytes: int = int(0.5 * 1024 ** 3)  # 500 MB
    drone_num_classes: int = 6
    enable_osd: bool = False          # set True to run Tesseract OCR; False saves significant compute
    enable_drone_clustering: bool = False  # Phase-2 Barlow-Twins training ignores drone_id; disable to skip fingerprint compute
    drone_fingerprint_frames: int = 60
    inpaint_erode_px: int = 5
    max_single_file_gb: float = 4.0   # >14 GB würde den Runner killen (No space left)
    download_max_retries: int = 3
    download_stall_timeout_s: int = 60
    playlist_batch_size: int = 3  # limit concurrent playlist downloads on 14 GB GitHub runners
    max_segment_frames: int = 3600
    mmap_max_frames: int = 3600
    chunk_size_frames: int = 60
    # Frames per Parquet row for frames_rgb/frames_gray/actions/osd/etc.
    # A segment with n_frames > this is written as multiple rows
    # (chunk_id 0..n_chunks-1) instead of one giant blob-per-column row.
    # Keeps np.concatenate + Snappy encoder peaks bounded to ~this many
    # frames instead of the full segment (up to max_segment_frames).
    # 240 frames @ 1280x720x1 (gray) ≈ 221 MB raw — well under CI RAM budget.
    parquet_row_frames: int = 240

CFG = PipelineConfig()


def _estimated_task_hours(row: pd.Series) -> float:
    """Return a deterministic sharding weight for one CSV row."""
    value = pd.to_numeric(row.get("estimated_hours", 0.0), errors="coerce")
    if pd.notna(value) and float(value) > 0:
        return float(value)

    platform = str(row.get("platform", "")).lower()
    if "playlist" in platform:
        return 1.0
    return 0.15


def split_dataframe_balanced_by_estimated_hours(df: pd.DataFrame, shard: int, num_shards: int) -> pd.DataFrame:
    """Assign rows to shards by estimated work instead of raw row count.

    GitHub Actions matrix shards cannot steal work from each other once they
    have started, so simple ``df.iloc[shard::num_shards]`` slicing can leave one
    runner with most of the long playlists.  A largest-processing-time-first
    pass distributes high-cost rows across shards while remaining deterministic.
    """
    if num_shards <= 1:
        return df.reset_index(drop=True)
    if shard < 0 or shard >= num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")

    weighted_rows = [
        (idx, _estimated_task_hours(row))
        for idx, row in df.iterrows()
    ]
    shard_loads = [0.0] * num_shards
    shard_indices: List[List[int]] = [[] for _ in range(num_shards)]

    for idx, hours in sorted(weighted_rows, key=lambda item: (-item[1], item[0])):
        target_shard = min(range(num_shards), key=lambda i: (shard_loads[i], i))
        shard_indices[target_shard].append(idx)
        shard_loads[target_shard] += hours

    selected = sorted(shard_indices[shard])
    return df.loc[selected].reset_index(drop=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(name)-10s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(CFG.log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


@contextlib.contextmanager
def stage_timer(name: str, logger: logging.Logger) -> Generator:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.debug(f"[{name}] Done in {time.perf_counter() - t0:.2f}s")


class SegmentRegistry:
    def __init__(self, path: str, lock: threading.Lock):
        self.path = Path(path)
        self.lock = lock
        self.path.touch(exist_ok=True)
        with self.lock:
            raw = self.path.read_text(encoding="utf-8").splitlines()
        self._done: Set[str] = {l.strip() for l in raw if l.strip()}

    @staticmethod
    def _key(stem: str, seg_id: int) -> str:
        return f"{stem}::{seg_id}"

    def is_done(self, stem: str, seg_id: int) -> bool:
        return self._key(stem, seg_id) in self._done

    def mark_done(self, stem: str, seg_id: int) -> None:
        k = self._key(stem, seg_id)
        if k in self._done:
            return
        self._done.add(k)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(k + "\n")


class URLRegistry:
    def __init__(self, path: str, lock: multiprocessing.Lock):
        self.path = Path(path)
        self.lock = lock
        self.path.touch(exist_ok=True)
        with self.lock:
            raw = self.path.read_text(encoding="utf-8").splitlines()
        self._done: Set[str] = {l.strip() for l in raw if l.strip()}

    def is_done(self, url: str) -> bool:
        return url.strip() in self._done

    def mark_done(self, url: str) -> None:
        clean = url.strip()
        if clean in self._done:
            return
        self._done.add(clean)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(clean + "\n")


def _pack_segments_to_chunks(parquet_dir: Path, chunk_dir: Path, target_bytes: int, logger: logging.Logger) -> List[Path]:
    seg_files = sorted(parquet_dir.glob("*_seg*.parquet"))
    if not seg_files:
        return []
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[Path] = []
    batch: List[pa.Table] = []
    batch_bytes = 0
    chunk_idx = 0
    ts = int(time.time())

    def _flush(batch_tables: List[pa.Table], idx: int) -> Path:
        out = chunk_dir / f"chunk_{ts}_{idx:04d}.parquet"
        writer = None
        try:
            for tbl in batch_tables:
                if writer is None:
                    writer = pq.ParquetWriter(out, tbl.schema, compression="snappy")
                writer.write_table(tbl)
        finally:
            if writer:
                writer.close()
        logger.info(f"[Pack] Chunk {idx:04d} written: {len(batch_tables)} segments, {out.stat().st_size/1e9:.2f} GB -> {out.name}")
        return out

    for fpath in seg_files:
        fsize = fpath.stat().st_size

        # A segment that's already >= target_bytes on its own can't be usefully
        # batched with anything else anyway, so reading it via pq.read_table()
        # just to immediately rewrite it via ParquetWriter would materialize
        # the same multi-GB buffer in RAM a second time for zero benefit.
        # Pass it straight through as its own chunk instead.
        if fsize >= target_bytes:
            if batch:
                chunks.append(_flush(batch, chunk_idx))
                for t in batch:
                    del t
                batch, batch_bytes, chunk_idx = [], 0, chunk_idx + 1
                gc.collect()
            out = chunk_dir / f"chunk_{ts}_{chunk_idx:04d}.parquet"
            shutil.move(str(fpath), str(out))
            logger.info(f"[Pack] Segment already >= target ({fsize/1e9:.2f} GB) - passed through as {out.name}")
            chunks.append(out)
            chunk_idx += 1
            continue

        try:
            tbl = pq.read_table(fpath)
        except Exception as e:
            logger.warning(f"[Pack] Skipping unreadable parquet {fpath.name}: {e}")
            continue
        if batch and batch_bytes + fsize > target_bytes:
            chunks.append(_flush(batch, chunk_idx))
            for t in batch:
                del t
            batch, batch_bytes, chunk_idx = [], 0, chunk_idx + 1
            gc.collect()
        batch.append(tbl)
        batch_bytes += fsize

    if batch:
        chunks.append(_flush(batch, chunk_idx))
        for t in batch:
            del t
        gc.collect()

    for fpath in seg_files:
        if fpath.exists():
            try:
                fpath.unlink()
            except Exception:
                pass
    return chunks


def hf_upload_and_clear(output_dir: Path, repo_id: str, logger: logging.Logger, max_workers: int = 2, pack_chunk_bytes: int = int(2.0 * 1024 ** 3)) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    parquet_dir = output_dir / "parquet"
    chunk_dir = output_dir / "chunks"
    seg_count = len(list(parquet_dir.glob("*_seg*.parquet")))
    if seg_count == 0:
        return
    logger.info(f"[HF-Upload] Packing {seg_count} segments into ~{pack_chunk_bytes/1e9:.1f}GB chunks...")
    chunk_files = _pack_segments_to_chunks(parquet_dir, chunk_dir, pack_chunk_bytes, logger)
    if not chunk_files:
        return
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception:
        pass
    upload_errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: Dict[Any, Path] = {
            pool.submit(
                api.upload_file,
                path_or_fileobj=str(fpath),
                path_in_repo=f"data/{fpath.name}",
                repo_id=repo_id,
                repo_type="dataset",
            ): fpath
            for fpath in chunk_files
        }
        for fut in tqdm(as_completed(futures), total=len(chunk_files), desc="HF-Upload", leave=False):
            fpath = futures[fut]
            try:
                fut.result(timeout=600)  # prevent a stalled upload from blocking until the 6h runner limit
                fpath.unlink()
            except Exception as e:
                logger.error(f"[HF-Upload] Failed {fpath.name}: {e}")
                upload_errors += 1
    logger.info(f"[HF-Upload] Completed {len(chunk_files) - upload_errors}/{len(chunk_files)} chunks successfully.")


def maybe_upload(parquet_exporter: "ParquetExporter", output_dir: Path, repo_id: str, threshold_bytes: int, logger: logging.Logger, upload_workers: int = 2, pack_chunk_bytes: int = int(2.0 * 1024 ** 3)) -> None:
    if parquet_exporter.total_bytes >= threshold_bytes:
        logger.info(f"[Trigger] Size threshold reached ({parquet_exporter.total_bytes/1e9:.2f} GB). Starting upload process.")
        hf_upload_and_clear(output_dir, repo_id, logger, upload_workers, pack_chunk_bytes)
        parquet_exporter.reset_bytes()


def save_progress_to_gist(resume_file: str, logger: logging.Logger) -> None:
    """Push processed_urls.txt to a GitHub Gist for cross-run persistence.

    Requires env vars:
      GH_GIST_ID  – the gist ID to update
      GH_TOKEN    – GitHub token (set as a workflow secret)
    The gh CLI must be available on PATH (pre-installed on GitHub runners).
    """
    gist_id = os.environ.get("GH_GIST_ID", "").strip()
    if not gist_id:
        return
    try:
        subprocess.run(
            ["gh", "gist", "edit", gist_id, resume_file],
            timeout=60,
            check=True,
            capture_output=True,
        )
        logger.info("[Gist] Progress saved to GitHub Gist.")
    except FileNotFoundError:
        logger.debug("[Gist] gh CLI not found – skipping gist save.")
    except Exception as exc:
        logger.warning(f"[Gist] Failed to save progress: {exc}")


# ── Cross-shard YouTube download lock (best-effort, via shared Gist) ────
# All shards must keep using the SAME account cookies (anonymous requests
# get blocked harder). To stop 10 concurrent sessions on that one account
# from tripping YouTube's hour-long rate limit, only one shard is allowed
# to actually be mid-download at any instant. We use a second file in the
# same Gist you already use for processed_urls.txt as a mutex: its content
# is "<holder_id>|<epoch_seconds>", empty/missing/stale means free.
#
# This is NOT a real distributed lock (no compare-and-swap on gh gist
# edit) — two shards can both see "free" and both write "claimed" within
# the same short window. We mitigate that with a random settle delay +
# re-read-to-verify after claiming, which makes true double-claims rare in
# practice with ~10 low-frequency contenders. Worst case on a race is a
# couple of shards briefly downloading at once for a few seconds — not
# the 10x pileup we're trying to avoid, and it self-heals on the next
# poll. If exact single-flight matters more than this pipeline needs it
# to, that requires real infra (e.g. a tiny lock server) instead.
_YT_LOCK_FILENAME = "yt_lock.json"


def _yt_lock_read(gist_id: str) -> Optional[Tuple[str, float]]:
    try:
        out = subprocess.run(
            ["gh", "gist", "view", gist_id, "--filename", _YT_LOCK_FILENAME, "--raw"],
            timeout=15, capture_output=True, text=True,
        )
        content = out.stdout.strip()
        if out.returncode != 0 or not content or "|" not in content:
            return None
        holder, ts = content.split("|", 1)
        return (holder, float(ts))
    except Exception:
        return None


def _yt_lock_write(gist_id: str, holder: str, ts: float) -> bool:
    try:
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / _YT_LOCK_FILENAME
            lock_path.write_text(f"{holder}|{ts}")
            # gh gist edit <id> <file> only UPDATES a file already present in
            # the gist (matched by basename) — it does not create new ones.
            # yt_lock.json won't exist yet on the very first call in a fresh
            # gist, so that attempt fails once; --add creates it. All calls
            # after that succeed via the plain edit path.
            out = subprocess.run(
                ["gh", "gist", "edit", gist_id, str(lock_path)],
                timeout=15, capture_output=True, text=True,
            )
            if out.returncode == 0:
                return True
            out = subprocess.run(
                ["gh", "gist", "edit", gist_id, "--add", str(lock_path)],
                timeout=15, capture_output=True, text=True,
            )
            return out.returncode == 0
    except Exception:
        return False


# Active heartbeat threads keyed by holder_id, so a long-running download
# keeps re-stamping the lock's timestamp instead of relying on a single
# stale_after_s window sized for the worst-case download duration. This lets
# stale_after_s stay tight (fast recovery if a shard crashes/gets killed
# mid-download) without a genuinely still-running download getting reaped by
# another shard that thinks it's stale.
_yt_lock_heartbeats: Dict[str, Tuple[threading.Event, threading.Thread]] = {}


def _yt_lock_heartbeat_loop(gist_id: str, holder_id: str, stop_event: threading.Event,
                             interval_s: float, logger: logging.Logger) -> None:
    while not stop_event.wait(interval_s):
        lock = _yt_lock_read(gist_id)
        if lock and lock[0] == holder_id:
            _yt_lock_write(gist_id, holder_id, time.time())
        else:
            # Lost the lock somehow (e.g. another shard's clock/network hiccup
            # caused a false-stale reap). Nothing to renew anymore; stop.
            logger.warning(f"[yt-lock] Heartbeat for {holder_id} found lock no longer ours; stopping renewal.")
            return


def acquire_yt_download_lock(logger: logging.Logger, max_wait_s: float = 600.0, stale_after_s: float = 120.0,
                              heartbeat_interval_s: float = 30.0) -> Optional[str]:
    """Block (polling the Gist) until this process holds the cross-shard
    YouTube download lock, or max_wait_s elapses. Returns this process's
    holder id on success (pass it to release_yt_download_lock), or None if
    it timed out — callers should treat None as "proceed anyway" rather
    than stalling forever, since this is a throughput throttle, not a
    correctness requirement.

    On success, spawns a background thread that re-stamps the lock every
    heartbeat_interval_s for as long as we hold it, so stale_after_s only
    has to cover "how long until we notice a crashed holder", not "how long
    could the longest possible download take". stale_after_s should stay
    a small multiple of heartbeat_interval_s (default: 4x) to tolerate an
    occasional missed heartbeat without false-reaping an active holder.
    """
    gist_id = os.environ.get("GH_GIST_ID", "").strip()
    if not gist_id:
        return None  # no gist configured -> no coordination possible, don't block
    my_id = f"{os.environ.get('GITHUB_RUN_ID', 'local')}-{uuid.uuid4().hex[:8]}"
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        lock = _yt_lock_read(gist_id)
        now = time.time()
        free = lock is None or (now - lock[1]) > stale_after_s
        if free:
            _yt_lock_write(gist_id, my_id, now)
            time.sleep(random.uniform(0.8, 2.0))  # let any racing claim settle
            check = _yt_lock_read(gist_id)
            if check and check[0] == my_id:
                stop_event = threading.Event()
                t = threading.Thread(
                    target=_yt_lock_heartbeat_loop,
                    args=(gist_id, my_id, stop_event, heartbeat_interval_s, logger),
                    daemon=True,
                )
                t.start()
                _yt_lock_heartbeats[my_id] = (stop_event, t)
                return my_id  # won the race
        time.sleep(random.uniform(2.0, 5.0))
    logger.warning("[yt-lock] Timed out waiting for cross-shard download lock; proceeding without it.")
    return None


def release_yt_download_lock(holder_id: Optional[str], logger: logging.Logger) -> None:
    gist_id = os.environ.get("GH_GIST_ID", "").strip()
    if not gist_id or holder_id is None:
        return
    heartbeat = _yt_lock_heartbeats.pop(holder_id, None)
    if heartbeat is not None:
        stop_event, t = heartbeat
        stop_event.set()
        t.join(timeout=5.0)
    lock = _yt_lock_read(gist_id)
    if lock and lock[0] == holder_id:
        _yt_lock_write(gist_id, "", 0.0)


# ── YouTube session rate-limit cooldown ─────────────────────────────────
# yt-dlp's "current session has been rate-limited by YouTube for up to an
# hour" error (see https://github.com/yt-dlp/yt-dlp/issues/14921) is not a
# transient per-request failure: once YouTube trips it, every further
# request from this process is doomed for up to an hour. Our normal
# per-URL retry/backoff just burns CI minutes and requests against an
# already-tripped limiter, and can make the ban stick longer. We track it
# as a process-wide (per-shard) cooldown and skip all further YouTube/
# AirVuz downloads until it expires.
_YT_RATE_LIMIT_RE = re.compile(r"rate-limited by YouTube", re.IGNORECASE)
_yt_rate_limit_lock = threading.Lock()
_yt_rate_limit_until: float = 0.0  # time.monotonic() timestamp, 0 = not limited


def _yt_rate_limited_now() -> bool:
    with _yt_rate_limit_lock:
        return time.monotonic() < _yt_rate_limit_until


def _note_if_yt_rate_limit(exc: BaseException, logger: logging.Logger, cooldown_s: float = 3600.0) -> None:
    """Check whether `exc` is YouTube's session rate-limit error; if so, arm
    the process-wide cooldown so callers stop issuing further YouTube/AirVuz
    requests instead of retrying into a limit that won't lift for up to an
    hour regardless of retry count."""
    if not _YT_RATE_LIMIT_RE.search(str(exc)):
        return
    global _yt_rate_limit_until
    with _yt_rate_limit_lock:
        _yt_rate_limit_until = time.monotonic() + cooldown_s
    logger.error(
        f"[yt-dlp] YouTube session rate limit hit — pausing all YouTube/AirVuz "
        f"downloads in this shard for {cooldown_s / 60:.0f} min."
    )


def check_content_length_gb(url: str) -> Optional[float]:
    try:
        r = requests.head(url, timeout=15, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        if cl and cl.isdigit():
            return int(cl) / (1024 ** 3)
    except Exception:
        pass
    return None


def download_file_http(url: str, dest: Path, logger: logging.Logger) -> Optional[Path]:
    size_gb = check_content_length_gb(url)
    if size_gb is not None and size_gb > CFG.max_single_file_gb:
        logger.warning(f"[Download] Skipped (Exceeds Limit): {url}")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = CFG.max_single_file_gb * 1024 ** 3
    for attempt in range(1, CFG.download_max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                downloaded = 0
                last_chunk_time = time.monotonic()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        if time.monotonic() - last_chunk_time > CFG.download_stall_timeout_s:
                            raise RuntimeError("Download stall detected.")
                        f.write(chunk)
                        downloaded += len(chunk)
                        last_chunk_time = time.monotonic()
                        if downloaded > max_bytes:
                            dest.unlink(missing_ok=True)
                            logger.warning(f"[Download] Aborted (exceeded {CFG.max_single_file_gb:.0f}GB limit): {url}")
                            return None
            if dest.stat().st_size > max_bytes:
                dest.unlink(missing_ok=True)
                logger.warning(f"[Download] Final size exceeded limit: {url}")
                return None
            return dest
        except Exception as e:
            dest.unlink(missing_ok=True)
            if attempt < CFG.download_max_retries:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"[Download] Failed {url}: {e}")
    return None


def _yt_dlp_options(dest_dir: Path, *, allow_playlist: bool) -> Dict[str, Any]:
    cookies_path = Path(__file__).parent / "cookies.txt"
    ydl_opts: Dict[str, Any] = {
        "format": "bestvideo+bestaudio/best",
        "format_sort": ["vcodec:h264", "acodec:aac", "quality", "res", "fps"],
        "merge_output_format": "mp4",
        "remux_video": "mp4",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "noplaylist": not allow_playlist,
        "verbose": True,
        "extractor_args": {
            "youtube": {"player_client": ["mweb", "web_safari"]},
        },
        "remote_components": ["ejs:github"],
        # Bumped from 1/2/5: with up to 10 parallel CI shards potentially
        # sharing one account's cookies, the old values let combined request
        # rate spike far above what a single YouTube session tolerates
        # before tripping the hour-long session rate limit (see
        # https://github.com/yt-dlp/yt-dlp/issues/14921). These are per-shard
        # values, so effective combined rate is roughly num_shards x higher.
        "sleep_interval_requests": 3,
        "sleep_interval": 5,
        "max_sleep_interval": 15,
    }
    if cookies_path.exists() and cookies_path.stat().st_size > 0:
        ydl_opts["cookiefile"] = str(cookies_path)
    return ydl_opts


def _download_yt_dlp_items(urls: List[str], dest_dir: Path, logger: logging.Logger) -> List[Path]:
    import yt_dlp
    dest_dir.mkdir(parents=True, exist_ok=True)
    cookies_path = Path(__file__).parent / "cookies.txt"
    # Only inject cookies when the file actually exists (written by CI from secret)
    if cookies_path.exists():
        size = cookies_path.stat().st_size
        logger.info(f"cookies.txt gefunden ({size} Bytes).")
    else:
        logger.warning(f"cookies.txt fehlt unter: {cookies_path}")
    downloaded = []
    try:
        ydl_opts = _yt_dlp_options(dest_dir, allow_playlist=False)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for item_url in urls:
                lock_id = acquire_yt_download_lock(logger)
                try:
                    info = ydl.extract_info(item_url, download=True)
                finally:
                    release_yt_download_lock(lock_id, logger)
                if info is None:
                    continue
                entries = info.get("entries") or [info]
                for entry in entries:
                    if entry is None:
                        continue
                    candidates = list(dest_dir.glob(f"{entry.get('id','video')}.*"))
                    for p in candidates:
                        if p.suffix.lower() in VIDEO_EXTS and p.stat().st_size > 0:
                            downloaded.append(p)
                            break
    except Exception as e:
        _note_if_yt_rate_limit(e, logger)
        raise RuntimeError(f"yt-dlp failed for {urls[0] if urls else '<empty>'}: {e}") from e
    return downloaded


def iter_yt_dlp_batched(url: str, dest_dir: Path, logger: logging.Logger, deadline_reached: Optional[Any] = None) -> Iterator[Path]:
    """Download YouTube/AirVuz URLs in small batches and yield local files.

    Playlists are first resolved without downloading.  Each batch is then
    downloaded with playlist expansion disabled so at most
    CFG.playlist_batch_size videos reside in ``dest_dir`` at once.
    """
    import yt_dlp

    batch_size = max(1, int(CFG.playlist_batch_size))
    dest_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = _yt_dlp_options(dest_dir, allow_playlist=True)
    ydl_opts["extract_flat"] = "in_playlist"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        return

    entries = info.get("entries")
    if entries:
        item_urls = []
        for entry in entries:
            if entry is None:
                continue
            item_url = entry.get("webpage_url") or entry.get("url")
            if not item_url:
                continue
            item_urls.append(item_url)
    else:
        item_urls = [url]

    logger.info(f"[yt-dlp] Downloading {len(item_urls)} item(s) in batches of {batch_size}.")
    for start in range(0, len(item_urls), batch_size):
        if deadline_reached and deadline_reached():
            logger.warning("[Shutdown] Deadline reached before next playlist batch; stopping URL without marking done.")
            return
        batch_urls = item_urls[start:start + batch_size]
        batch_dir = dest_dir / f"batch_{start // batch_size:05d}"
        paths = _download_yt_dlp_items(batch_urls, batch_dir, logger)
        try:
            for p in paths:
                yield p
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)


def iter_github_repo_videos(repo_url: str, tmp_dir: Path, logger: logging.Logger) -> Iterator[Tuple[Path, bool]]:
    match = re.search(r"github\.com/([^/]+/[^/\s?#]+)", repo_url)
    if not match:
        return
    repo_slug = match.group(1).rstrip("/")
    api_url = f"https://api.github.com/repos/{repo_slug}/git/trees/HEAD?recursive=1"
    try:
        r = requests.get(api_url, timeout=30, headers={"Accept": "application/vnd.github.v3+json"})
        r.raise_for_status()
        tree = r.json().get("tree", [])
    except Exception:
        return
    video_files = [i for i in tree if i.get("type") == "blob" and Path(i["path"]).suffix.lower() in VIDEO_EXTS]
    depth_paths = {i["path"] for i in tree if "depth" in i["path"].lower()}
    for item in video_files:
        fpath = item["path"]
        raw_url = f"https://raw.githubusercontent.com/{repo_slug}/HEAD/{fpath}"
        dest = tmp_dir / Path(fpath).name
        if item.get("size", 0) > CFG.max_single_file_gb * 1024 ** 3:
            continue
        local = download_file_http(raw_url, dest, logger)
        if local:
            yield local, any(Path(fpath).stem in d for d in depth_paths)


def iter_hf_dataset_videos(repo_id: str, tmp_dir: Path, logger: logging.Logger) -> Iterator[Tuple[Path, bool]]:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    try:
        all_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception:
        return
    video_files = [f for f in all_files if Path(f).suffix.lower() in VIDEO_EXTS]
    depth_files_set = {f for f in all_files if "depth" in f.lower()}
    for vf in video_files:
        has_depth = any(Path(vf).stem in df for df in depth_files_set)
        try:
            local = hf_hub_download(repo_id=repo_id, filename=vf, repo_type="dataset", local_dir=str(tmp_dir))
            yield Path(local), has_depth
        except Exception:
            continue


def classify_url(url: str, platform: str) -> str:
    url_l = url.lower()
    if platform in ("YouTube", "YouTube Playlist"): return "youtube"
    if platform == "AirVuz": return "airvuz"
    if "huggingface.co/datasets/" in url_l: return "hf_dataset"
    if "github.com" in url_l or "gitlab.com" in url_l: return "github"
    if any(url_l.rstrip("/").endswith(ext) for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".zip", ".tar", ".tar.gz"]): return "http_file"
    return "web_dir"


def get_video_info(video_path: Path) -> Tuple[int, int, float]:
    cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate", "-of", "csv=p=0", str(video_path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30).decode().strip()
        parts = out.split(",")
        w, h = int(parts[0]), int(parts[1])
        fps_str = parts[2]
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / max(float(den), 1e-6)
        else:
            fps = float(fps_str)
        return w, h, max(fps, 1.0)
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return w, h, max(fps, 1.0) if fps and fps > 0 else 30.0


def _build_ffmpeg_cmd(video_path: Path, out_hw: Tuple[int, int], target_fps: float, start_sec: float, max_frames: int = 0) -> List[str]:
    out_h, out_w = out_hw
    cmd = ["ffmpeg", "-y", "-threads", "2"]
    if start_sec > 0.01:
        cmd += ["-ss", f"{start_sec:.4f}"]
    cmd += ["-i", str(video_path)]
    if max_frames > 0:
        cmd += ["-vframes", str(max_frames)]
    cmd += ["-vf", f"fps={target_fps:.6f},scale={out_w}:{out_h}:flags=area"]
    cmd += ["-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-an", "-"]
    return cmd


def read_video_frames_ffmpeg(video_path: Path, out_hw: Optional[Tuple[int, int]] = None, target_fps: Optional[float] = None, start_sec: float = 0.0, max_frames: int = 0) -> Tuple[np.ndarray, float]:
    _, _, native_fps = get_video_info(video_path)
    effective_fps = target_fps if target_fps else native_fps
    out_h = out_hw[0] if out_hw else None
    out_w = out_hw[1] if out_hw else None

    cmd = ["ffmpeg", "-y", "-threads", "2"]
    if start_sec > 0.01:
        cmd += ["-ss", f"{start_sec:.4f}"]
    cmd += ["-i", str(video_path)]
    if max_frames > 0:
        cmd += ["-vframes", str(max_frames)]
    vf_parts = []
    if target_fps:
        vf_parts.append(f"fps={target_fps:.6f}")
    if out_hw:
        vf_parts.append(f"scale={out_w}:{out_h}:flags=area")
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-an", "-"]

    frame_bytes = out_w * out_h * 3
    frames: List[np.ndarray] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3).copy())
    finally:
        if not proc.stdout.closed:
            proc.stdout.close()
        proc.wait()

    if not frames:
        raise RuntimeError(f"Decode failed for {video_path}")
    return np.stack(frames, axis=0), effective_fps


def read_video_to_mmap(video_path: Path, out_hw: Tuple[int, int], target_fps: float, start_sec: float, tmp_dir: Path, max_frames: int = 0) -> Tuple[Path, Tuple[int, int, int, int], float]:
    """Decode video into a memory-mapped file using PyAV (no subprocess, no pipe deadlocks).

    PyAV decodes directly into numpy-compatible buffers, skipping the
    subprocess+image2pipe pattern that can deadlock when the OS pipe buffer fills up.
    """
    import av  # PyAV — already used in fly_real.py
    out_h, out_w = out_hw
    frame_bytes = out_w * out_h * 3

    fd, mmap_path_str = tempfile.mkstemp(suffix="_video.mmap", dir=tmp_dir)
    os.close(fd)
    mmap_path = Path(mmap_path_str)

    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        # Seek to start position
        if start_sec > 0.01:
            container.seek(int(start_sec * av.time_base ** -1), any_frame=False, backward=True)

        # Compute frame interval for FPS sub-sampling
        native_fps = float(stream.average_rate or target_fps)
        keep_every = max(1, round(native_fps / target_fps))

        frame_idx = 0
        written = 0
        with open(mmap_path, "wb") as f_out:
            for packet in container.demux(stream):
                for av_frame in packet.decode():
                    if max_frames > 0 and written >= max_frames:
                        break
                    if frame_idx % keep_every != 0:
                        frame_idx += 1
                        continue
                    # Resize + convert BGR via numpy (no subprocess)
                    bgr = av_frame.to_ndarray(format="bgr24")
                    if bgr.shape[:2] != (out_h, out_w):
                        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    f_out.write(bgr.tobytes())
                    written += 1
                    frame_idx += 1
                if max_frames > 0 and written >= max_frames:
                    break
        container.close()

        n_frames = mmap_path.stat().st_size // frame_bytes
        if n_frames == 0:
            mmap_path.unlink(missing_ok=True)
            raise RuntimeError(f"PyAV decode produced 0 frames for {video_path}")
        return mmap_path, (n_frames, out_h, out_w, 3), target_fps
    except Exception as e:
        mmap_path.unlink(missing_ok=True)
        raise e


def open_mmap_video(mmap_path: Path, shape: Tuple) -> np.ndarray:
    return np.memmap(mmap_path, dtype=np.uint8, mode="r", shape=shape)


def release_mmap_range(mm_array: np.memmap, frame_start: int, frame_end: int, frame_bytes: int) -> None:
    """Tell the kernel to drop already-consumed pages of a read-only video mmap.

    Without this, RSS grows ~linearly as we sequentially scan through the mmap
    (each newly-touched page stays resident/page-cached until evicted under
    memory pressure), even though the pages are never re-read. Since the data
    is read-only and unmodified, the kernel can always re-fault it back in
    from disk for free if we're ever wrong about not needing it again.
    """
    try:
        underlying_mm = mm_array._mmap
        offset = frame_start * frame_bytes
        length = (frame_end - frame_start) * frame_bytes
        page_size = mmap_module.ALLOCATIONGRANULARITY
        aligned_offset = (offset // page_size) * page_size
        aligned_length = length + (offset - aligned_offset)
        underlying_mm.madvise(mmap_module.MADV_DONTNEED, aligned_offset, aligned_length)
    except Exception:
        pass  # best-effort only; never let memory-hygiene fail the pipeline


class TakeoffDetector:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @staticmethod
    def _flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return float(np.mean(np.linalg.norm(flow, axis=-1)))

    def find_takeoff_frame(self, video_path: Path) -> Tuple[int, float]:
        _, _, native_fps = get_video_info(video_path)
        native_fps = native_fps or 30.0
        step = max(1, int(round(native_fps / CFG.takeoff_fps)))
        max_sample = int(CFG.takeoff_max_seconds * CFG.takeoff_fps)
        try:
            frames_small, _ = read_video_frames_ffmpeg(video_path, out_hw=(CFG.takeoff_res, CFG.takeoff_res), target_fps=CFG.takeoff_fps, max_frames=max_sample)
        except Exception:
            return 0, 0.0
        frames_small = frames_small[:max_sample]
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_small]
        if len(grays) < 10:
            return 0, 0.0
        magnitudes = np.array([self._flow_magnitude(grays[i-1], grays[i]) for i in range(1, len(grays))], dtype=np.float32)
        smoothed = np.convolve(magnitudes, np.ones(5) / 5, mode="valid")
        sustain = min(CFG.takeoff_sustain_frames, max(1, len(smoothed) // 2))
        if sustain <= 0 or sustain >= len(smoothed):
            return 0, 0.0
        for i in range(len(smoothed) - sustain):
            if np.all(smoothed[i:i+sustain] > CFG.takeoff_flow_threshold):
                sample_pos = min(i + 1, len(grays) - 1)
                start_sec = sample_pos / CFG.takeoff_fps
                native_idx = int(sample_pos * step)
                return native_idx, start_sec
        return 0, 0.0


class OSDAutoDetector:
    _PATTERNS = {
        "battery": re.compile(r"\b(\d{1,2}[\.,]\d{1,2})\s*[Vv]\b|\b(\d{2,3})\s*[Vv]\b"),
        "speed": re.compile(r"\b(\d{1,4})\s*(?:km/h|kmh|kph|mph)\b", re.IGNORECASE),
        "altitude": re.compile(r"\b(\d{1,5})\s*m\b(?!\s*/|\s*ph|\s*s)", re.IGNORECASE),
    }

    def detect_rois(self, frames_bgr: List[np.ndarray]) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
        collected = {"battery": [], "speed": [], "altitude": []}
        scale = CFG.osd_scale
        for frame in frames_bgr[:CFG.osd_scan_frames]:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            try:
                data = pytesseract.image_to_data(small, output_type=pytesseract.Output.DICT, config=TESS_FULL_FRAME)
            except Exception:
                continue
            for i, text in enumerate(data["text"]):
                text = text.strip()
                if not text:
                    continue
                for key, pat in self._PATTERNS.items():
                    if pat.search(text):
                        x, y = int(data["left"][i] / scale), int(data["top"][i] / scale)
                        bw, bh = int(data["width"][i] / scale), int(data["height"][i] / scale)
                        pad = max(8, int(h * 0.01))  # relative to frame height, not hardcoded
                        collected[key].append((max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)))
        result = {}
        for key, boxes in collected.items():
            result[key] = tuple(np.median(np.array(boxes, dtype=float), axis=0).astype(int).tolist()) if boxes else None
        return result

    @staticmethod
    def _ocr_number(crop: np.ndarray) -> Optional[float]:
        if crop is None or crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            txt = pytesseract.image_to_string(bw, config=TESS_SINGLE_LINE)
        except Exception:
            return None
        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", txt)
        return float(m.group(1).replace(",", ".")) if m else None

    def extract_values_adaptive(self, frames_bgr: np.ndarray, rois: Dict[str, Optional[Tuple]]) -> Tuple[Dict[str, np.ndarray], bool]:
        n = len(frames_bgr)
        results = {k: np.full(n, np.nan, dtype=np.float32) for k in rois}
        if not any(v is not None for v in rois.values()):
            return results, False
        last_crops = {k: None for k in rois}
        last_vals = {k: None for k in rois}
        for i, frame in enumerate(frames_bgr):
            for key, roi in rois.items():
                if roi is None:
                    continue
                x0, y0, x1, y1 = roi
                crop_bgr = frame[y0:y1, x0:x1]
                if crop_bgr.size == 0:
                    continue
                crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                run_ocr = True
                if last_crops[key] is not None:
                    mad = float(np.mean(np.abs(crop_gray.astype(np.float32) - last_crops[key].astype(np.float32))))
                    run_ocr = mad > CFG.osd_roi_change_threshold
                if run_ocr:
                    val = self._ocr_number(crop_bgr)
                    last_vals[key] = val
                    last_crops[key] = crop_gray
                else:
                    val = last_vals[key]
                if val is not None:
                    results[key][i] = val
        has_osd = bool(np.mean(np.any([np.isfinite(results[k]) for k in results], axis=0)) >= 0.1)
        return results, has_osd


class StickCamDetector:
    @staticmethod
    def _to_small_gray(frames_bgr: np.ndarray, res: int) -> List[np.ndarray]:
        return [cv2.cvtColor(cv2.resize(f, (res, res), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames_bgr]

    def find_stick_roi(self, frames_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if len(frames_bgr) < 5:
            return None
        H_orig, W_orig = frames_bgr.shape[1:3]
        res = CFG.stick_flow_res
        grays_small = self._to_small_gray(frames_bgr[:min(60, len(frames_bgr))], res)
        n = len(grays_small)
        f = CFG.stick_corner_fraction
        corners_norm = {
            "tl": (0, 0, f, f),
            "tr": (1-f, 0, 1.0, f),
            "bl": (0, 1-f, f, 1.0),
            "br": (1-f, 1-f, 1.0, 1.0),
        }
        def roi_flow_small(nx0, ny0, nx1, ny1, i) -> float:
            x0, y0, x1, y1 = int(nx0*res), int(ny0*res), int(nx1*res), int(ny1*res)
            p, c = grays_small[i-1][y0:y1, x0:x1], grays_small[i][y0:y1, x0:x1]
            if p.size == 0 or c.size == 0: return 0.0
            fl = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            return float(np.mean(np.linalg.norm(fl, axis=-1)))
        center_flows = [roi_flow_small(f, f, 1-f, 1-f, i) for i in range(1, n)]
        mean_center = max(np.mean(center_flows), 1e-6)
        best_key, best_ratio = None, float("inf")
        for key, (nx0, ny0, nx1, ny1) in corners_norm.items():
            ratio = (np.mean([roi_flow_small(nx0, ny0, nx1, ny1, i) for i in range(1, n)]) / mean_center)
            if ratio < best_ratio:
                best_ratio, best_key = ratio, key
        if best_ratio > CFG.stick_flow_ratio_threshold:
            return None
        nx0, ny0, nx1, ny1 = corners_norm[best_key]
        return (int(nx0*W_orig), int(ny0*H_orig), int(nx1*W_orig), int(ny1*H_orig))

    def extract_actions(self, frames_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = roi
        n = len(frames_bgr)
        roi_h, roi_w = y1 - y0, x1 - x0
        roi_grays = [cv2.cvtColor(frames_bgr[i, y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) for i in range(n)]
        bg = np.median(np.stack(roi_grays[:min(10, n)], axis=0), axis=0).astype(np.uint8)
        raw = np.full((n, 4), np.nan, dtype=np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for i, gray in enumerate(roi_grays):
            diff = cv2.absdiff(gray, bg)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 8 < area < 3000:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        centroids.append((M["m10"]/M["m00"], M["m01"]/M["m00"], area))
            if len(centroids) < 2: continue
            centroids.sort(key=lambda p: -p[2])
            top2 = sorted([(c[0], c[1]) for c in centroids[:2]], key=lambda p: p[0])
            left, right = top2[0], top2[1]
            raw[i, 0] = left[0] / max(1, roi_w - 1)
            raw[i, 1] = left[1] / max(1, roi_h - 1)
            raw[i, 2] = right[0] / max(1, roi_w - 1)
            raw[i, 3] = right[1] / max(1, roi_h - 1)
        idx = np.arange(n)
        for j in range(4):
            col = raw[:, j]
            mask = np.isfinite(col)
            col[~mask] = (np.interp(idx[~mask], idx[mask], col[mask]) if np.any(mask) else 0.5)
        return np.stack([
            np.clip((raw[:, 2] - 0.5) * 2.0, -1.0, 1.0),
            np.clip(-(raw[:, 3] - 0.5) * 2.0, -1.0, 1.0),
            np.clip((raw[:, 0] - 0.5) * 2.0, -1.0, 1.0),
            np.clip(-(raw[:, 1] - 0.5) * 2.0, -1.0, 1.0),
        ], axis=-1).astype(np.float32)


class SceneSegmenter:
    def __init__(self, threshold: float, min_scene_len_frames: int):
        self.threshold = threshold / 100.0 * 255.0
        self.min_len = min_scene_len_frames

    def segment_from_frames(self, frames_gray: np.ndarray) -> List[Tuple[int, int]]:
        n = len(frames_gray)
        if n < self.min_len:
            return [(0, n)]
        diffs = np.mean(np.abs(frames_gray[1:].astype(np.float32) - frames_gray[:-1].astype(np.float32)), axis=(1, 2))
        cuts = [0]
        for i, d in enumerate(diffs):
            if d > self.threshold:
                if (i + 1) - cuts[-1] >= self.min_len:
                    cuts.append(i + 1)
        cuts.append(n)
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= self.min_len] or [(0, n)]


class CrashDetector:
    def __init__(self, flow_threshold: float, brightness_drop: float, sharpness_drop: float, near_miss_threshold: float):
        self.flow_threshold = flow_threshold
        self.brightness_drop = brightness_drop
        self.sharpness_drop = sharpness_drop
        self.near_miss_threshold = near_miss_threshold

    def detect(self, frames_gray: np.ndarray, flow_mags: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[int], List[int]]:
        n = len(frames_gray)
        is_terminal = np.zeros(n, dtype=bool)
        crashes, near_misses = [], []
        if n < 3: return is_terminal, crashes, near_misses
        prev = frames_gray[0]
        prev_b = float(np.mean(prev))
        prev_s = float(cv2.Laplacian(prev, cv2.CV_32F).var())
        for i in range(1, n):
            cur = frames_gray[i]
            if flow_mags is not None:
                flow_mag = float(flow_mags[i])
            else:
                flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_mag = float(np.mean(np.linalg.norm(flow, axis=-1)))
            b, s = float(np.mean(cur)), float(cv2.Laplacian(cur, cv2.CV_32F).var())
            if flow_mag > self.flow_threshold and ((prev_b - b) > self.brightness_drop or (prev_s - s) > self.sharpness_drop):
                is_terminal[i] = True
                crashes.append(i)
            elif flow_mag > self.near_miss_threshold:
                near_misses.append(i)
            prev, prev_b, prev_s = cur, b, s
        return is_terminal, crashes, near_misses


class DroneClassifier:
    def __init__(self, num_classes: int):
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        from sklearn.pipeline import Pipeline as SKPipeline
        from sklearn.preprocessing import StandardScaler
        self.num_classes = num_classes
        self._pipe = SKPipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=4))])
        self._gmm = GaussianMixture(n_components=num_classes, covariance_type="full", random_state=42)
        self._fingerprints: List[np.ndarray] = []
        self._fitted = False
        self._lock = threading.Lock()

    def compute_fingerprint(self, frames_gray: np.ndarray, n: int = 60, flow_mags: Optional[np.ndarray] = None, flow_yaws: Optional[np.ndarray] = None, flow_pitches: Optional[np.ndarray] = None) -> np.ndarray:
        n = min(n, len(frames_gray))
        if n < 3: return np.zeros(8, dtype=np.float32)
        sub = frames_gray[:n]
        speeds, yaws, pitches, jerks, last_speed = [], [], [], [], 0.0
        for i in range(1, n):
            if flow_mags is not None and flow_yaws is not None and flow_pitches is not None:
                sp, yaw, pitch = float(flow_mags[i]), float(flow_yaws[i]), float(flow_pitches[i])
            else:
                flow = cv2.calcOpticalFlowFarneback(sub[i-1], sub[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                sp = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                yaw, pitch = float(np.mean(flow[..., 0])), float(np.mean(flow[..., 1]))
            speeds.append(sp); yaws.append(yaw); pitches.append(pitch)
            jerks.append(abs(sp - last_speed)); last_speed = sp
        return np.array([np.mean(speeds), np.std(speeds), np.mean(yaws), np.std(yaws), np.mean(pitches), np.std(pitches), np.mean(jerks), np.std(jerks)], dtype=np.float32)

    def classify(self, fingerprint: np.ndarray) -> int:
        with self._lock:
            self._fingerprints.append(fingerprint)
            should_fit = not self._fitted and len(self._fingerprints) >= max(8, self.num_classes)
            fingerprints_snapshot = list(self._fingerprints) if should_fit else None

        if should_fit and fingerprints_snapshot is not None:
            from sklearn.mixture import GaussianMixture
            X = np.stack(fingerprints_snapshot)
            Z = self._pipe.fit_transform(X)
            gmm = GaussianMixture(n_components=min(self.num_classes, len(Z)), covariance_type="full", random_state=42)
            gmm.fit(Z)
            with self._lock:
                self._gmm = gmm
                self._fitted = True

        with self._lock:
            if self._fitted:
                return int(self._gmm.predict(self._pipe.transform(fingerprint[None, :]))[0])
        return 0


def build_overlay_mask(frame_hw: Tuple[int, int], osd_rois: Optional[Dict[str, Optional[Tuple]]], stick_roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    H, W = frame_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    if stick_roi is not None:
        mask[stick_roi[1]:stick_roi[3], stick_roi[0]:stick_roi[2]] = 255
    if osd_rois:
        for roi in osd_rois.values():
            if roi is not None:
                mask[roi[1]:roi[3], roi[0]:roi[2]] = 255
    return mask


def inpaint_overlays_to_numpy(frames_bgr: np.ndarray, mask: np.ndarray, erode_px: int = 5) -> np.ndarray:
    """
    Masks overlay regions (OSD, stick cam) and returns an RGB uint8 array (T, H, W, 3).

    cv2.inpaint was removed: R2Dreamer receives the cam_overlay_mask and learns that
    masked pixels are irrelevant.  Replacing with the frame mean is a simple numpy
    assignment and is ~10 000× faster on a CPU-only runner.
    """
    n_frames = len(frames_bgr)
    out_frames = np.empty_like(frames_bgr)

    # Pre-compute a boolean mask for direct array indexing
    bool_mask: Optional[np.ndarray] = None
    if np.any(mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        eroded = cv2.erode(mask, k, iterations=1)
        if np.any(eroded):
            bool_mask = eroded > 0  # (H, W) bool

    for i in range(n_frames):
        frame = frames_bgr[i].copy()
        if bool_mask is not None:
            # Fill masked pixels with the per-channel mean of the unmasked area.
            # The mask tells the model these pixels are irrelevant — no need to
            # reconstruct plausible content via the expensive inpaint algorithm.
            fill = frame[~bool_mask].mean(axis=0).astype(np.uint8) if np.any(~bool_mask) else np.zeros(3, dtype=np.uint8)
            frame[bool_mask] = fill
        out_frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        del frame

    return out_frames  # uint8 (T, H, W, 3) RGB


def build_osd_array(speeds: np.ndarray, altitudes: np.ndarray, batteries: np.ndarray, n: int) -> np.ndarray:
    """
    Build the (T, 8) OSD feature array expected by r2dreamer's FPVDataset._extract_osd().
    Slots: [speed, altitude, battery, 0, 0, 0, 0, 0]
    """
    osd = np.zeros((n, 8), dtype=np.float32)
    if len(speeds) == n:
        osd[:, 0] = np.nan_to_num(speeds, nan=0.0)
    if len(altitudes) == n:
        osd[:, 1] = np.nan_to_num(altitudes, nan=0.0)
    if len(batteries) == n:
        osd[:, 2] = np.nan_to_num(batteries, nan=0.0)
    return osd


def _stream_windows(chunks: List[np.ndarray], window: int) -> Iterator[np.ndarray]:
    """Yield fixed-size windows (along axis 0) from a list of arrays, without
    ever materializing more than ~one window's worth of data at a time.

    Replaces a single np.concatenate(chunks, axis=0) over the WHOLE segment
    (peak = 2x full segment size: old chunks + one giant merged copy) with a
    generator that copies into a small pre-allocated `window`-sized buffer as
    it walks the input chunks. Peak here is ~window size, independent of how
    long the segment is (up to max_segment_frames). The last window may be
    shorter than `window` if the total frame count doesn't divide evenly.
    """
    if not chunks:
        return
    tail_shape = chunks[0].shape[1:]
    dtype = chunks[0].dtype
    out = np.empty((window, *tail_shape), dtype=dtype)
    out_n = 0
    for c in chunks:
        pos = 0
        while pos < c.shape[0]:
            take = min(window - out_n, c.shape[0] - pos)
            out[out_n:out_n + take] = c[pos:pos + take]
            out_n += take
            pos += take
            if out_n == window:
                yield out.copy()
                out_n = 0
    if out_n:
        yield out[:out_n].copy()


def _blob(array: np.ndarray) -> bytes:
    """Serialize a small (single row-group) ndarray to contiguous bytes."""
    return np.ascontiguousarray(array).tobytes()


def _table_from_chunked_rows(rows: List[Dict[str, Any]]) -> pa.Table:
    """Build a pyarrow Table from a list of per-chunk row dicts, forcing
    raw-bytes fields to large_binary (64-bit offsets).

    Unlike the old single-row-per-segment layout, each row here only covers
    parquet_row_frames frames, so blob sizes stay small (tens/low-hundreds of
    MB) and pa.array(values, type=large_binary()) can build the whole column
    in one go without needing the zero-copy py_buffer trick — there's nothing
    large enough left to worry about.

    large_binary (not binary) is still required: even a single row-group can
    theoretically approach the 2**31-2 byte binary-type cap at large
    parquet_row_frames / resolutions, and large_binary has no such limit.
    """
    if not rows:
        return pa.Table.from_arrays([], names=[])
    binary_fields = {
        k for k, v in rows[0].items() if isinstance(v, (bytes, bytearray))
    }
    arrays: List[pa.Array] = []
    names: List[str] = []
    for name in rows[0].keys():
        values = [r[name] for r in rows]
        if name in binary_fields:
            arrays.append(pa.array(values, type=pa.large_binary()))
        else:
            arrays.append(pa.array(values))
        names.append(name)
    return pa.Table.from_arrays(arrays, names=names)



# ---------------------------------------------------------------------------
#  CrashDatasetExporter
#  Saves pre-crash and clean frames to a separate Parquet dataset that is
#  consumed by r2dreamer's SafetyRawImageSource in "streaming" mode.
#
#  Each row has two columns:
#    image  — PNG-encoded grayscale frame at safety resolution (1280×720)
#    crash  — float32: 1.0 = crash / pre-crash frame, 0.0 = clean frame
#
#  Compatible with SafetyRawImageSource config:
#    raw_dataset:
#      source: streaming
#      hf_repo: jaron12/fpv-crash-dataset
#      streaming: true
# ---------------------------------------------------------------------------
class CrashDatasetExporter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rows: List[Dict] = []
        self.total_bytes: int = 0
        self._file_idx: int = 0

    def reset_bytes(self) -> None:
        self.total_bytes = 0

    def _encode_frame(self, frame_hw1: np.ndarray) -> bytes:
        """Encode a (H, W, 1) or (H, W) uint8 array as PNG bytes."""
        arr = frame_hw1.squeeze() if frame_hw1.ndim == 3 else frame_hw1
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=1)
        return buf.getvalue()

    def add_frames(
        self,
        safety_frames: np.ndarray,   # (T, safety_h, safety_w, 1) uint8
        crash_mask: np.ndarray,       # (T,) bool  — True = crash/pre-crash frame
    ) -> None:
        """Buffer frames with their crash labels."""
        for i in range(len(safety_frames)):
            self._rows.append({
                "image":  self._encode_frame(safety_frames[i]),
                "crash":  float(crash_mask[i]),
            })

    def flush(self) -> Optional[Path]:
        """Write buffered rows to a Parquet file and clear the buffer."""
        if not self._rows:
            return None
        df = pd.DataFrame(self._rows)
        # Store image bytes as a dict {bytes, path} — matches HF Image feature convention
        df["image"] = df["image"].apply(lambda b: {"bytes": b, "path": None})
        table = pa.Table.from_pandas(df, preserve_index=False)
        out_path = self.out_dir / f"crash_{self._file_idx:06d}.parquet"
        pq.write_table(table, out_path, compression="snappy")
        self.total_bytes += out_path.stat().st_size
        self._file_idx += 1
        self._rows.clear()
        return out_path

    def maybe_flush(self, threshold_bytes: int) -> None:
        approx = sum(len(r["image"]) for r in self._rows)
        if approx >= threshold_bytes:
            self.flush()


def upload_crash_dataset(out_dir: Path, repo_id: str, logger: logging.Logger) -> None:
    """Push all crash Parquet files in out_dir to a HuggingFace dataset repo."""
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        import datasets as hf_datasets

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        parquet_files = sorted(out_dir.glob("crash_*.parquet"))
        if not parquet_files:
            logger.info("[CrashDS] No files to upload.")
            return

        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        operations = [
            CommitOperationAdd(
                path_in_repo=f"data/{f.name}",
                path_or_fileobj=str(f),
            )
            for f in parquet_files
        ]
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add {len(parquet_files)} crash shard(s)",
        )
        logger.info(f"[CrashDS] Uploaded {len(parquet_files)} files → {repo_id}")

        # Remove local files after successful upload
        for f in parquet_files:
            f.unlink(missing_ok=True)

    except Exception as exc:
        logger.warning(f"[CrashDS] Upload failed: {exc}")

class ParquetExporter:
    """
    Exports processed video segments to Parquet in the format expected by r2dreamer's FPVDataset:
      - frames_rgb:     uint8 numpy array (T, H, W, 3)       – RGB, masked (RSSM resolution)
      - frames_gray:    uint8 numpy array (T, safety_h, safety_w, 1) – grayscale (SafetyNet resolution)
      - cam_overlay:    uint8 numpy array (H, W)              – binary overlay mask (0/255)
      - osd:            float32 numpy array (T, 8)      – [speed, alt, battery, 0…0]
      - actions:        float32 numpy array (T, 4)
      - speeds:         float32 numpy array (T,)
      - altitudes:      float32 numpy array (T,)
      - batteries:      float32 numpy array (T,)
      - is_terminal:    bool numpy array (T,)
      - n_frames, drone_id, has_osd, fps, stem, segment_id

    Note: frames_gray is now stored at SafetyNet resolution (safety_h × safety_w, default 1280×720),
    downscaled from the 1080p source. The RSSM trainer still converts frames_rgb → gray on-the-fly
    for its own encoder; frames_gray is exclusively consumed by the SafetyNet.
    """

    def __init__(self, out_dir: Path, row_frames: int = CFG.parquet_row_frames):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total_bytes: int = 0
        self.row_frames = row_frames

    def reset_bytes(self) -> None:
        self.total_bytes = 0

    def export_segment(
        self,
        stem: str,
        segment_id: int,
        frames_rgb_chunks: List[np.ndarray],           # list of (t, H, W, 3) uint8 RGB chunks – already inpainted
        frames_gray_chunks: Optional[List[np.ndarray]],  # list of (t, safety_h, safety_w, 1) uint8 chunks; None/empty = omit
        cam_overlay_mask: np.ndarray,         # (H, W) uint8 binary mask
        actions: np.ndarray,           # (T, 4) float32
        speeds: np.ndarray,            # (T,) float32
        altitudes: np.ndarray,         # (T,) float32
        batteries: np.ndarray,         # (T,) float32
        is_terminal: np.ndarray,       # (T,) bool
        drone_id: int,
        has_osd: bool,
        fps: float,
    ) -> Path:
        """Write one segment as multiple parquet_row_frames-sized ROWS instead
        of one giant blob-per-column row.

        Rationale: the old path did one np.concatenate over the *whole*
        segment (up to max_segment_frames, e.g. 3600 frames) per column, then
        handed the ~GB-sized result to the Snappy encoder, which allocates its
        own internal buffer in raw-data size on top. Combined peak could hit
        ~3x a single column's raw size and blow past the CI runner's
        MemoryMax.

        Here, frames_rgb/frames_gray/actions/osd/speeds/altitudes/batteries/
        is_terminal are all split into aligned windows of `self.row_frames`
        frames via _stream_windows, and each window becomes its own row
        (chunk_id 0..n_chunks-1, chunk_frame_offset = starting frame index).
        Both np.concatenate and the Snappy encoder now only ever see one
        window's worth of data (tens/low-hundreds of MB) at a time, so the
        peak no longer scales with segment length. Compression stays ON
        (snappy) — chunking doesn't change compression ratio, Snappy is
        stateless per call.

        cam_overlay is per-segment, not per-frame; it's small (H, W) and gets
        duplicated into every row so a downstream reader can reconstruct a
        segment from any single row without a separate join.

        Downstream (FPVDataset) must group rows by (stem, segment_id), sort
        by chunk_id, and concatenate the decoded frame windows back into one
        (T, H, W, C) array per segment — that reassembly happens one segment
        at a time at load time, where memory pressure is a non-issue.
        """
        n = sum(c.shape[0] for c in frames_rgb_chunks)
        osd = build_osd_array(speeds, altitudes, batteries, n)
        cam_overlay_bytes = _blob(cam_overlay_mask.astype(np.uint8, copy=False))
        cam_overlay_shape = list(cam_overlay_mask.shape)

        rgb_windows = _stream_windows(frames_rgb_chunks, self.row_frames)
        gray_windows = (
            _stream_windows(frames_gray_chunks, self.row_frames)
            if frames_gray_chunks else None
        )

        rows: List[Dict[str, Any]] = []
        frame_offset = 0
        for rgb_win in rgb_windows:
            t = rgb_win.shape[0]
            sl = slice(frame_offset, frame_offset + t)
            row: Dict[str, Any] = {
                "frames_rgb": _blob(rgb_win),
                "frames_rgb_shape": list(rgb_win.shape),
            }
            if gray_windows is not None:
                gray_win = next(gray_windows)
                row["frames_gray"] = _blob(gray_win)
                row["frames_gray_shape"] = list(gray_win.shape)
            row["cam_overlay"] = cam_overlay_bytes
            row["cam_overlay_shape"] = cam_overlay_shape
            row["osd"] = _blob(osd[sl].astype(np.float32, copy=False))
            row["osd_shape"] = list(osd[sl].shape)
            row["actions"] = _blob(actions[sl].astype(np.float32, copy=False))
            row["actions_shape"] = list(actions[sl].shape)
            row["speeds"] = _blob(speeds[sl].astype(np.float32, copy=False))
            row["speeds_shape"] = list(speeds[sl].shape)
            row["altitudes"] = _blob(altitudes[sl].astype(np.float32, copy=False))
            row["altitudes_shape"] = list(altitudes[sl].shape)
            row["batteries"] = _blob(batteries[sl].astype(np.float32, copy=False))
            row["batteries_shape"] = list(batteries[sl].shape)
            row["is_terminal"] = _blob(is_terminal[sl].astype(np.bool_, copy=False))
            row["is_terminal_shape"] = list(is_terminal[sl].shape)
            # Scalar / segment-level metadata (repeated per row)
            row["n_frames"] = n
            row["chunk_id"] = len(rows)
            row["chunk_frame_offset"] = frame_offset
            row["drone_id"] = drone_id
            row["has_osd"] = has_osd
            row["fps"] = fps
            row["stem"] = stem
            row["segment_id"] = segment_id
            rows.append(row)
            frame_offset += t

        n_chunks = len(rows)
        for row in rows:
            row["n_chunks"] = n_chunks

        table = _table_from_chunked_rows(rows)
        out_path = self.out_dir / f"{stem}_seg{segment_id:04d}.parquet"
        pq.write_table(table, out_path, compression="snappy")
        self.total_bytes += out_path.stat().st_size
        return out_path


def interp_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy().astype(np.float32)
    idx = np.arange(len(arr))
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.zeros_like(arr)
    arr[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
    return arr


def _iter_url_videos(url: str, platform: str, tmp_dir: Path, registry: URLRegistry, logger: logging.Logger, deadline_reached: Optional[Any] = None) -> Iterator[Tuple[Path, bool, bool]]:
    url_type = classify_url(url, platform)
    if url_type in ("youtube", "airvuz"):
        if _yt_rate_limited_now():
            logger.info(f"[{url_type}] Skipping (YouTube session rate limit active): {url}")
            return
        dl_dir = tmp_dir / "yt" / hashlib.md5(url.encode()).hexdigest()[:10]
        completed = False
        try:
            yielded = 0
            for p in iter_yt_dlp_batched(url, dl_dir, logger, deadline_reached):
                if deadline_reached and deadline_reached():
                    logger.warning("[Shutdown] Deadline reached inside playlist; stopping URL without marking done.")
                    return
                yielded += 1
                yield p, True, False
            completed = yielded > 0
        except Exception as e:
            # NICHT mark_done: nur bei echtem Erfolg oben markieren, sonst geht die URL
            # beim naechsten Lauf erneut in den Retry statt fuer immer verloren zu sein.
            _note_if_yt_rate_limit(e, logger)
            logger.warning(f"[{url_type}] Download error: {e}")
        finally:
            shutil.rmtree(dl_dir, ignore_errors=True)
        if completed and not (deadline_reached and deadline_reached()):
            registry.mark_done(url)
        elif not completed:
            # ignoreerrors=True lets yt-dlp swallow failures without raising - treat an
            # empty result the same as a real failure and leave the URL for a retry.
            logger.warning(f"[{url_type}] Download lieferte 0 Dateien, URL bleibt offen: {url}")
    elif url_type == "hf_dataset":
        match = re.search(r"huggingface\.co/datasets/([^/\s?#]+/[^/\s?#]+)", url)
        if match:
            hf_tmp = tmp_dir / "hf" / match.group(1).replace("/", "__")
            for vpath, has_depth in iter_hf_dataset_videos(match.group(1).rstrip("/"), hf_tmp, logger):
                yield vpath, False, has_depth
            registry.mark_done(url)
    elif url_type == "github":
        gh_tmp = tmp_dir / "github" / hashlib.md5(url.encode()).hexdigest()[:10]
        for vpath, has_depth in iter_github_repo_videos(url, gh_tmp, logger):
            yield vpath, False, has_depth
        registry.mark_done(url)
    elif url_type == "http_file":
        fname = Path(url.split("?")[0]).name
        if not fname or "." not in fname:
            fname = f"video_{hashlib.md5(url.encode()).hexdigest()[:8]}.mp4"
        local = download_file_http(url, tmp_dir / "http" / fname, logger)
        if local: yield local, False, False
        registry.mark_done(url)
    else:
        registry.mark_done(url)


def _process_video_cpu(video_path: Path, is_fpv: bool, parquet_exp: ParquetExporter, crash_exp: CrashDatasetExporter, output_dir: Path, cfg: PipelineConfig, logger: logging.Logger, takeoff_det: TakeoffDetector, osd_det: OSDAutoDetector, stick_det: StickCamDetector, scene_seg: SceneSegmenter, crash_det: CrashDetector, drone_cls: DroneClassifier, seg_registry: SegmentRegistry, tmp_dir: Path) -> None:
    stem = video_path.stem
    out_hw_cam = (cfg.out_h, cfg.out_w)
    out_hw_depth = (cfg.depth_h, cfg.depth_w)
    start_sec = 0.0

    if is_fpv:
        with stage_timer(f"{stem}: Takeoff Detect", logger):
            try:
                _, start_sec = takeoff_det.find_takeoff_frame(video_path)
            except Exception:
                pass

    mmap_path_dep = None
    try:
        prev_gray_boundary: Optional[np.ndarray] = None
        seg_id_counter: int = 0
        open_bufs: Dict[str, list] = {
            "frames_rgb": [],   # list of (T_chunk, H, W, 3) uint8 arrays
            "frames_gray": [],  # list of (T_chunk, safety_h, safety_w, 1) uint8 – SafetyNet resolution
            "actions": [],
            "speeds": [], "alts": [], "batts": [], "is_terminal": [],
        }
        open_drone_id: int = 0
        open_has_osd: bool = False
        osd_rois: Optional[Dict] = None
        stick_roi: Optional[Tuple[int, int, int, int]] = None
        overlay_mask_dep: Optional[np.ndarray] = None
        overlay_mask_cam: Optional[np.ndarray] = None
        roi_512: Optional[Tuple[int, int, int, int]] = None
        fps: float = cfg.target_fps
        window_start_frame: int = 0

        def _release_freed_memory() -> None:
            """Force glibc to actually return freed heap memory to the OS.

            Refcounting frees numpy/bytes buffers immediately once out of
            scope, but glibc malloc often keeps the arena mapped for reuse
            instead of returning it — especially with the varying chunk/
            segment sizes here. Over many videos in the same long-lived
            worker process that shows up as RSS creeping up without an
            actual Python-level leak. gc.collect() also clears any
            reference cycles the _flush_scene closure might hold.
            """
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass  # non-glibc platform — nothing to trim

        def _flush_scene(sid: int) -> None:
            n_chunks = len(open_bufs["frames_rgb"])
            if n_chunks == 0:
                return
            n = sum(c.shape[0] for c in open_bufs["frames_rgb"])
            if n < cfg.min_scene_len_frames or seg_registry.is_done(stem, sid):
                for k in open_bufs:
                    open_bufs[k].clear()
                return
            actions_arr = np.concatenate(open_bufs["actions"], axis=0) if open_bufs["actions"] else np.zeros((n, 4), dtype=np.float32)
            speeds_arr  = interp_nan(np.concatenate(open_bufs["speeds"])  if open_bufs["speeds"]  else np.full(n, np.nan, dtype=np.float32))
            alts_arr    = interp_nan(np.concatenate(open_bufs["alts"])    if open_bufs["alts"]    else np.full(n, np.nan, dtype=np.float32))
            batts_arr   = interp_nan(np.concatenate(open_bufs["batts"])   if open_bufs["batts"]   else np.full(n, np.nan, dtype=np.float32))
            terminal_arr = np.concatenate(open_bufs["is_terminal"]) if open_bufs["is_terminal"] else np.zeros(n, dtype=bool)

            # Use cam overlay mask; fall back to empty mask if not yet set
            mask = overlay_mask_cam if overlay_mask_cam is not None else np.zeros(out_hw_cam, dtype=np.uint8)

            # frames_rgb/frames_gray are passed as chunk lists, not concatenated —
            # np.concatenate here would briefly double the ~1.5-3.1GB buffer per
            # field before serialization even touches it. See
            # _array_record_from_chunks for the zero-copy pa.py_buffer wrap that replaces it.
            out_path = parquet_exp.export_segment(
                stem, sid,
                open_bufs["frames_rgb"],
                open_bufs["frames_gray"] if open_bufs["frames_gray"] else None,
                mask,
                actions_arr, speeds_arr, alts_arr, batts_arr,
                terminal_arr,
                open_drone_id, open_has_osd, fps,
            )
            seg_registry.mark_done(stem, sid)
            logger.info(f"[{stem}] Segment {sid:04d} ({n} frames) -> {out_path.name}")
            for k in open_bufs:
                open_bufs[k].clear()
            _release_freed_memory()

        while True:
            window_start_sec = start_sec + window_start_frame / fps
            try:
                with stage_timer(f"{stem}: Decode MMAP 1080p (window={window_start_frame})", logger):
                    mmap_path_dep, mmap_shape_dep, fps = read_video_to_mmap(
                        video_path, out_hw_depth, cfg.target_fps, window_start_sec,
                        tmp_dir, max_frames=cfg.mmap_max_frames,
                    )
            except Exception as e:
                logger.error(f"[{stem}] Decode failed: {e}")
                break

            n_window = mmap_shape_dep[0]
            if n_window < cfg.min_scene_len_frames:
                mmap_path_dep.unlink(missing_ok=True)
                mmap_path_dep = None
                break

            frames_dep_mmap = open_mmap_video(mmap_path_dep, mmap_shape_dep)

            if window_start_frame == 0 and is_fpv:
                scan_end = min(max(cfg.osd_scan_frames + 5, 65), n_window)
                osd_sample = np.array(frames_dep_mmap[:scan_end])
                try:
                    with stage_timer(f"{stem}: Detect ROIs", logger):
                            if cfg.enable_osd:
                                osd_rois = osd_det.detect_rois(list(osd_sample[:cfg.osd_scan_frames]))
                            stick_roi = stick_det.find_stick_roi(osd_sample[:60])
                except Exception:
                    pass
                finally:
                    del osd_sample
                    gc.collect()

            if overlay_mask_dep is None:
                overlay_mask_dep = build_overlay_mask(out_hw_depth, osd_rois, stick_roi)
                overlay_mask_cam = cv2.resize(overlay_mask_dep, (out_hw_cam[1], out_hw_cam[0]), interpolation=cv2.INTER_NEAREST)
                if stick_roi is not None:
                    sx, sy = out_hw_cam[1] / out_hw_depth[1], out_hw_cam[0] / out_hw_depth[0]
                    roi_512 = (int(stick_roi[0] * sx), int(stick_roi[1] * sy), int(stick_roi[2] * sx), int(stick_roi[3] * sy))

            chunk_size = cfg.chunk_size_frames
            frame_bytes_depth = out_hw_depth[0] * out_hw_depth[1] * 3
            for chunk_start in range(0, n_window, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_window)
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                logger.debug(f"[Mem] chunk {chunk_start}-{chunk_end}/{n_window} | RSS peak: {rss_mb:.0f} MB | open segment frames: {sum(len(b) for b in open_bufs['frames_rgb'])}")

                chunk_1080p = frames_dep_mmap[chunk_start:chunk_end]
                chunk_512p = np.stack(
                    [cv2.resize(chunk_1080p[fi], (cfg.out_w, cfg.out_h), interpolation=cv2.INTER_AREA) for fi in range(len(chunk_1080p))],
                    axis=0,
                )
                gray_chunk = np.stack([cv2.cvtColor(chunk_512p[fi], cv2.COLOR_BGR2GRAY) for fi in range(len(chunk_512p))], axis=0)
                # Bug fix: build SafetyNet frames at safety resolution from 1080p source (downscale, not upscale)
                chunk_safety_gray = np.stack(
                    [cv2.cvtColor(cv2.resize(chunk_1080p[fi], (cfg.safety_w, cfg.safety_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)[..., np.newaxis]
                     for fi in range(len(chunk_1080p))],
                    axis=0,
                )  # (T, safety_h, safety_w, 1) uint8

                gray_with_prev = np.concatenate([prev_gray_boundary[np.newaxis], gray_chunk], axis=0) if prev_gray_boundary is not None else gray_chunk
                boundary_offset = 1 if prev_gray_boundary is not None else 0

                local_segments = scene_seg.segment_from_frames(gray_with_prev)

                for local_s, local_e in local_segments:
                    seg_s = min(max(local_s - boundary_offset, 0), len(chunk_512p))
                    seg_e = min(max(local_e - boundary_offset, 0), len(chunk_512p))
                    if seg_e <= seg_s:
                        continue

                    if local_s > boundary_offset and open_bufs["frames_rgb"]:
                        _flush_scene(seg_id_counter)
                        seg_id_counter += 1
                        maybe_upload(parquet_exp, output_dir, cfg.hf_repo_id, cfg.hf_upload_threshold_bytes, logger, cfg.upload_workers, int(cfg.pack_chunk_gb * 1024 ** 3))

                    seg_512 = chunk_512p[seg_s:seg_e]
                    seg_gray = gray_chunk[seg_s:seg_e]
                    seg_1080p = chunk_1080p[seg_s:seg_e]
                    seg_safety_gray = chunk_safety_gray[seg_s:seg_e]  # (T, safety_h, safety_w, 1) uint8
                    n_seg = len(seg_512)

                    if roi_512 is not None:
                        try:
                            actions_seg = stick_det.extract_actions(seg_512, roi_512)
                        except Exception:
                            actions_seg = np.zeros((n_seg, 4), dtype=np.float32)
                    else:
                        actions_seg = np.zeros((n_seg, 4), dtype=np.float32)

                    speeds_seg = np.full(n_seg, np.nan, dtype=np.float32)
                    alts_seg = np.full(n_seg, np.nan, dtype=np.float32)
                    batts_seg = np.full(n_seg, np.nan, dtype=np.float32)
                    has_osd_seg = False
                    if osd_rois and any(v is not None for v in osd_rois.values()):
                        try:
                            osd_vals, has_osd_seg = osd_det.extract_values_adaptive(seg_1080p, osd_rois)
                            speeds_seg = osd_vals.get("speed", speeds_seg)
                            alts_seg = osd_vals.get("altitude", alts_seg)
                            batts_seg = osd_vals.get("battery", batts_seg)
                        except Exception:
                            pass

                    n_seg_flow = len(seg_gray)
                    flow_mags = np.zeros(n_seg_flow, dtype=np.float32)
                    flow_yaws = np.zeros(n_seg_flow, dtype=np.float32)
                    flow_pitches = np.zeros(n_seg_flow, dtype=np.float32)
                    for fi in range(1, n_seg_flow):
                        fl = cv2.calcOpticalFlowFarneback(seg_gray[fi-1], seg_gray[fi], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        flow_mags[fi] = float(np.mean(np.linalg.norm(fl, axis=-1)))
                        flow_yaws[fi] = float(np.mean(fl[..., 0]))
                        flow_pitches[fi] = float(np.mean(fl[..., 1]))

                    is_terminal_seg, crash_indices, _ = crash_det.detect(seg_gray, flow_mags=flow_mags)

                    # ── Save pre-crash frames to the safety dataset ──────────────────────
                    # Clean frames are already in the main FPV dataset (is_terminal=False).
                    # Only crash=1.0 frames go here to avoid duplicating clean data.
                    if crash_indices and seg_safety_gray is not None:
                        ctx = cfg.crash_context_frames
                        n_seg_f = len(seg_safety_gray)
                        crash_mask = np.zeros(n_seg_f, dtype=bool)
                        for ci in crash_indices:
                            lo = max(0, ci - ctx)
                            hi = min(n_seg_f, ci + 1)
                            crash_mask[lo:hi] = True
                        crash_frames = seg_safety_gray[crash_mask]  # only crash=1.0 frames
                        labels = np.ones(len(crash_frames), dtype=bool)
                        crash_exp.add_frames(crash_frames, labels)
                        crash_exp.maybe_flush(cfg.crash_upload_threshold_bytes)
                    # ────────────────────────────────────────────────────────────────────
                    open_drone_id = drone_cls.classify(drone_cls.compute_fingerprint(
                        seg_gray, cfg.drone_fingerprint_frames,
                        flow_mags=flow_mags, flow_yaws=flow_yaws, flow_pitches=flow_pitches,
                    )) if drone_cls is not None else 0  # skip expensive fingerprint when Barlow-Twins is active
                    open_has_osd = open_has_osd or has_osd_seg

                    # Inpaint OSD/stick overlays and convert BGR→RGB for r2dreamer
                    rgb_seg = inpaint_overlays_to_numpy(seg_512, overlay_mask_cam, cfg.inpaint_erode_px)

                    open_bufs["frames_rgb"].append(rgb_seg)
                    open_bufs["frames_gray"].append(seg_safety_gray)
                    open_bufs["actions"].append(actions_seg)
                    open_bufs["speeds"].append(speeds_seg)
                    open_bufs["alts"].append(alts_seg)
                    open_bufs["batts"].append(batts_seg)
                    open_bufs["is_terminal"].append(is_terminal_seg)

                    if sum(len(b) for b in open_bufs["frames_rgb"]) >= cfg.max_segment_frames:
                        _flush_scene(seg_id_counter)
                        seg_id_counter += 1
                        maybe_upload(parquet_exp, output_dir, cfg.hf_repo_id, cfg.hf_upload_threshold_bytes, logger, cfg.upload_workers, int(cfg.pack_chunk_gb * 1024 ** 3))

                    del seg_512, seg_gray, seg_1080p, rgb_seg, seg_safety_gray

                prev_gray_boundary = gray_chunk[-1].copy()
                del chunk_1080p, chunk_512p, gray_chunk, gray_with_prev

                # We scan the mmap strictly forward and never revisit earlier
                # frames, so drop the pages for this chunk now instead of
                # letting them sit resident until the whole ~3600-frame
                # window has been swept (which is what was driving RSS up
                # to ~13GB per segment before this fix).
                release_mmap_range(frames_dep_mmap, chunk_start, chunk_end, frame_bytes_depth)

            del frames_dep_mmap
            mmap_path_dep.unlink(missing_ok=True)
            mmap_path_dep = None
            gc.collect()

            if n_window < cfg.mmap_max_frames:
                break

            window_start_frame += n_window

        _flush_scene(seg_id_counter)
        maybe_upload(parquet_exp, output_dir, cfg.hf_repo_id, cfg.hf_upload_threshold_bytes, logger, cfg.upload_workers, int(cfg.pack_chunk_gb * 1024 ** 3))

    finally:
        if mmap_path_dep:
            mmap_path_dep.unlink(missing_ok=True)
    gc.collect()


def worker(worker_id: int, task_queue: multiprocessing.Queue, lock: multiprocessing.Lock, cfg: PipelineConfig, progress_counter: Optional[Any] = None, progress_lock: Optional[Any] = None) -> None:
    global CFG
    CFG = cfg
    output_dir = Path(cfg.output_root) / f"worker_{worker_id}"
    tmp_dir = Path(cfg.tmp_root) / f"worker_{worker_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"Worker-{worker_id}")
    logger.info(f"Started Worker {worker_id}")

    # ── Graceful-Shutdown timer ────────────────────────────────────────────
    # GitHub Actions kills a job after exactly 6 hours.  We stop accepting
    # new tasks after GRACEFUL_SHUTDOWN_SECONDS so the final HF upload can
    # complete before the runner is forcibly terminated.
    worker_start = time.monotonic()

    def _deadline_reached() -> bool:
        return (time.monotonic() - worker_start) >= GRACEFUL_SHUTDOWN_SECONDS

    takeoff_det = TakeoffDetector(logger)
    osd_det = OSDAutoDetector() if cfg.enable_osd else None  # skip Tesseract init when OSD disabled
    stick_det = StickCamDetector()
    scene_seg = SceneSegmenter(cfg.scene_threshold, cfg.min_scene_len_frames)
    crash_det = CrashDetector(cfg.crash_flow_threshold, cfg.crash_brightness_drop, cfg.crash_sharpness_drop, cfg.near_miss_flow_threshold)
    drone_cls = DroneClassifier(cfg.drone_num_classes) if cfg.enable_drone_clustering else None  # unused in Barlow-Twins phase
    parquet_exp = ParquetExporter(output_dir / "parquet")
    crash_exp = CrashDatasetExporter(output_dir / "crash_parquet")
    registry = URLRegistry(cfg.resume_file, lock)
    seg_registry = SegmentRegistry(cfg.segment_done_file, threading.Lock())

    gist_save_interval = 300  # save progress every 5 minutes
    gist_jitter = random.uniform(0, gist_save_interval)
    last_gist_save = time.monotonic() - gist_jitter
    logger.info(f"[Gist] Periodic save jitter offset: {gist_jitter:.1f}s.")

    while True:
        # ── Deadline check: stop before GitHub kills us ────────────────────
        if _deadline_reached():
            logger.warning("[Shutdown] 5.5 h deadline reached – stopping task loop for final upload.")
            break

        try:
            item = task_queue.get(timeout=15)
        except Exception:
            break
        if item is None:
            break

        url, platform, title = item
        if registry.is_done(url):
            if progress_counter is not None and progress_lock is not None:
                with progress_lock:
                    progress_counter.value += 1
            continue

        logger.info(f"Processing: [{platform}] {title[:50]}...")
        for video_path, is_fpv, _ in _iter_url_videos(url, platform, tmp_dir, registry, logger, _deadline_reached):
            try:
                if _deadline_reached():
                    logger.warning("[Shutdown] Deadline reached before processing next video; deleting current file and stopping.")
                    break
                _process_video_cpu(video_path, is_fpv, parquet_exp, crash_exp, output_dir, cfg, logger, takeoff_det, osd_det, stick_det, scene_seg, crash_det, drone_cls, seg_registry, tmp_dir)
            except Exception as e:
                logger.error(f"[{video_path.stem}] Critical Error: {e}")
                # ── Automated issue tracking ───────────────────────────────
                # Create a GitHub Issue so broken videos are visible without
                # digging through CI logs.  Requires GH_TOKEN in the env.
                try:
                    issue_title = f"[pipeline] Failed: {video_path.stem}"
                    issue_body = (
                        f"**URL:** {url}\n"
                        f"**Platform:** {platform}\n"
                        f"**Title:** {title}\n"
                        f"**Error:** {e}\n"
                        f"**Worker:** {worker_id}"
                    )
                    subprocess.run(
                        ["gh", "issue", "create", "--title", issue_title, "--body", issue_body],
                        timeout=30,
                        check=False,
                        capture_output=True,
                    )
                except Exception as gh_exc:
                    logger.debug(f"[Issue] gh issue create failed: {gh_exc}")
            finally:
                try:
                    video_path.unlink()
                except Exception:
                    pass

            if _deadline_reached():
                logger.warning("[Shutdown] Deadline reached inside URL video loop; stopping task loop for final upload.")
                break

        if progress_counter is not None and progress_lock is not None:
            with progress_lock:
                progress_counter.value += 1

        # ── Periodic gist save ─────────────────────────────────────────────
        now = time.monotonic()
        if now - last_gist_save >= gist_save_interval:
            save_progress_to_gist(cfg.resume_file, logger)
            last_gist_save = now

    logger.info("Triggering final pack and HF-Upload...")
    try:
        hf_upload_and_clear(output_dir, cfg.hf_repo_id, logger, cfg.upload_workers, int(cfg.pack_chunk_gb * 1024 ** 3))
        # NEU: Crash-Dataset hier flushen und hochladen!
        crash_exp.flush()
        if cfg.hf_crash_repo_id:
            upload_crash_dataset(output_dir / "crash_parquet", cfg.hf_crash_repo_id, logger)
    except Exception as e:
        # Covers MemoryError too (Exception subclass) — with ulimit -v set in
        # the workflow, an OOM here raises MemoryError instead of the kernel
        # killing the whole runner, so this still gets a chance to log and
        # clean up instead of dying silently mid-shutdown.
        logger.error(f"[Shutdown] Final pack/upload failed: {e}")
    # Save final progress so the next shard/run knows what was done
    save_progress_to_gist(cfg.resume_file, logger)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Worker gracefully shutting down.")


def main() -> None:
    # ── CLI arguments (used by GitHub Actions matrix strategy) ─────────────
    parser = argparse.ArgumentParser(description="FPV Data Pipeline – GitHub CI Edition")
    parser.add_argument("--shard", type=int, default=0,
                        help="0-based shard index for this runner (default: 0)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards / runners (default: 1)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Override CSV path from PipelineConfig")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.csv:
        cfg.csv_path = args.csv

    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    Path(cfg.tmp_root).mkdir(parents=True, exist_ok=True)

    logger = get_logger("Main")
    logger.info("==========================================================")
    logger.info(" FPV Data Pipeline — GitHub CI Edition")
    logger.info("==========================================================")
    logger.info(f" CSV:        {cfg.csv_path}")
    logger.info(f" Output:     {cfg.output_root}")
    logger.info(f" HF Repo:    {cfg.hf_repo_id}")
    logger.info(f" Workers:    {cfg.n_workers}")
    logger.info(f" Shard:      {args.shard + 1}/{args.num_shards}")
    logger.info(f" Shutdown:   {GRACEFUL_SHUTDOWN_SECONDS/3600:.1f} h graceful limit")

    try:
        df = pd.read_csv(cfg.csv_path)
        logger.info(f"Loaded {len(df)} entries from CSV.")
    except Exception as e:
        logger.critical(f"CSV load error: {e}")
        sys.exit(1)

    # ── Sharding: balance estimated work so long playlists are spread out ──
    if args.num_shards > 1:
        df = split_dataframe_balanced_by_estimated_hours(df, args.shard, args.num_shards)
        shard_hours = sum(_estimated_task_hours(row) for _, row in df.iterrows())
        logger.info(
            f"[Shard {args.shard}/{args.num_shards}] Processing {len(df)} entries "
            f"(estimated {shard_hours:.2f} h)."
        )

    task_queue = multiprocessing.Queue()
    for _, row in df.iterrows():
        url, platform, title = str(row.get("url", "")).strip(), str(row.get("platform", "YouTube")).strip(), str(row.get("title", "Unknown")).strip()
        if url: task_queue.put((url, platform, title))
    for _ in range(cfg.n_workers):
        task_queue.put(None)

    lock = multiprocessing.Lock()
    progress_counter = multiprocessing.Value("i", 0)
    progress_lock = multiprocessing.Lock()
    total_tasks = len(df)

    processes = []
    for wid in range(cfg.n_workers):
        p = multiprocessing.Process(target=worker, args=(wid, task_queue, lock, cfg, progress_counter, progress_lock), name=f"Worker_{wid}")
        p.start()
        processes.append(p)

    with tqdm(total=total_tasks, desc="Pipeline", unit="video") as pbar:
        last = 0
        while any(p.is_alive() for p in processes):
            current = progress_counter.value
            if current > last:
                pbar.update(current - last)
                last = current
            time.sleep(1.0)
        pbar.update(progress_counter.value - last)

    for p in processes:
        p.join(timeout=300)
        if p.is_alive():
            logger.warning(f"Worker {p.name} did not finish in time, terminating.")
            p.terminate()
            p.join(timeout=10)

   
    logger.info("==========================================================")
    logger.info(" Pipeline completed. Dataset is HF-streamable.")
    logger.info("==========================================================")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
