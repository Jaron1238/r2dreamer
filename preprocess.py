import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytesseract
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class PreprocessConfig:
    input_root: str
    output_root: str
    fps: Optional[float] = None
    resize_w: int = 256
    resize_h: int = 256
    scene_threshold: float = 24.0
    min_scene_len_frames: int = 24
    drone_num_classes: int = 6
    drone_fingerprint_frames: int = 60
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Large-hf"
    depth_batch_size: int = 6
    crash_flow_threshold: float = 3.2
    crash_brightness_drop: float = 25.0
    crash_sharpness_drop: float = 60.0
    near_miss_flow_threshold: float = 2.0
    handcam_roi: Tuple[int, int, int, int] = (0, 0, 420, 240)
    osd_speed_roi: Tuple[int, int, int, int] = (20, 20, 200, 70)
    osd_alt_roi: Tuple[int, int, int, int] = (20, 72, 200, 122)
    osd_batt_roi: Tuple[int, int, int, int] = (20, 124, 220, 178)
    stick_hsv_low: Tuple[int, int, int] = (35, 40, 40)
    stick_hsv_high: Tuple[int, int, int] = (95, 255, 255)


class SceneSegmenter:
    def __init__(self, threshold: float, min_scene_len_frames: int):
        self.threshold = threshold
        self.min_scene_len_frames = min_scene_len_frames

    def segment(self, video_path: Path, num_frames: int) -> List[Tuple[int, int]]:
        try:
            from scenedetect import ContentDetector, SceneManager, open_video
        except Exception:
            return [(0, num_frames)]

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len_frames))
        manager.detect_scenes(video)
        scene_list = manager.get_scene_list()
        if not scene_list:
            return [(0, num_frames)]

        segments = []
        for start_tc, end_tc in scene_list:
            s = int(start_tc.get_frames())
            e = int(end_tc.get_frames())
            if e - s >= self.min_scene_len_frames:
                segments.append((s, min(e, num_frames)))
        if not segments:
            segments = [(0, num_frames)]
        return segments


class DroneClassifier:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=4)
        self.gmm = GaussianMixture(n_components=num_classes, covariance_type="full", random_state=42)
        self.pipeline = Pipeline([("scaler", self.scaler), ("pca", self.pca)])
        self._fingerprints: List[np.ndarray] = []
        self._fitted = False

    def compute_fingerprint(self, frames_gray: np.ndarray, n_frames: int = 60) -> np.ndarray:
        n = min(n_frames, len(frames_gray))
        if n < 3:
            return np.zeros(8, dtype=np.float32)

        subset = frames_gray[:n]
        prev = subset[0]
        speeds, yaws, pitches, jerks = [], [], [], []
        last_speed = 0.0

        for idx in range(1, n):
            cur = subset[idx]
            flow = cv2.calcOpticalFlowFarneback(
                prev,
                cur,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            fx = flow[..., 0]
            fy = flow[..., 1]
            speed = float(np.mean(np.sqrt(fx * fx + fy * fy)))
            yaw = float(np.mean(fx))
            pitch = float(np.mean(fy))
            jerk = abs(speed - last_speed)
            speeds.append(speed)
            yaws.append(yaw)
            pitches.append(pitch)
            jerks.append(jerk)
            last_speed = speed
            prev = cur

        feats = np.array(
            [
                np.mean(speeds), np.std(speeds),
                np.mean(yaws), np.std(yaws),
                np.mean(pitches), np.std(pitches),
                np.mean(jerks), np.std(jerks),
            ],
            dtype=np.float32,
        )
        return feats

    def classify(self, fingerprint: np.ndarray) -> int:
        self._fingerprints.append(fingerprint)
        if (not self._fitted) and len(self._fingerprints) >= max(8, self.num_classes):
            X = np.stack(self._fingerprints, axis=0)
            Z = self.pipeline.fit_transform(X)
            n_comp = min(self.num_classes, len(Z))
            if self.gmm.n_components != n_comp:
                self.gmm = GaussianMixture(n_components=n_comp, covariance_type="full", random_state=42)
            self.gmm.fit(Z)
            self._fitted = True

        if self._fitted:
            z_cur = self.pipeline.transform(fingerprint[None, :])
            return int(self.gmm.predict(z_cur)[0])
        return 0


class StickTracker:
    def __init__(self, roi: Tuple[int, int, int, int], hsv_low: Tuple[int, int, int], hsv_high: Tuple[int, int, int]):
        self.roi = roi
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)

    @staticmethod
    def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
        idx = np.arange(len(x))
        mask = np.isfinite(x)
        if not np.any(mask):
            return np.zeros_like(x)
        x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
        return x

    def _track_points(self, frame_bgr: np.ndarray) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.roi
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return (np.nan, np.nan, np.nan, np.nan)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_mask = cv2.inRange(gray, 180, 255)
        mask = cv2.bitwise_and(color_mask, bright_mask)
        ys, xs = np.where(mask > 0)
        if len(xs) < 2:
            return (np.nan, np.nan, np.nan, np.nan)

        vals = gray[ys, xs].astype(np.float32)
        idx = np.argsort(vals)[::-1]
        points = np.stack([xs[idx], ys[idx]], axis=1)
        picked: List[np.ndarray] = []
        min_dist2 = 20.0 ** 2
        for p in points:
            if not picked:
                picked.append(p)
                continue
            d2 = np.sum((picked[0] - p) ** 2)
            if d2 >= min_dist2:
                picked.append(p)
                break
        if len(picked) < 2:
            return (np.nan, np.nan, np.nan, np.nan)

        p0 = picked[0].astype(np.float32)
        p1 = picked[1].astype(np.float32)
        centers = sorted([(p0[0], p0[1]), (p1[0], p1[1])], key=lambda p: p[0])
        (lx, ly), (rx, ry) = centers[0], centers[1]
        w = max(1.0, float(roi.shape[1] - 1))
        h = max(1.0, float(roi.shape[0] - 1))
        return (lx / w, ly / h, rx / w, ry / h)

    def extract_actions(self, frames_bgr: np.ndarray) -> np.ndarray:
        vals = np.array([self._track_points(f) for f in frames_bgr], dtype=np.float32)
        for i in range(vals.shape[1]):
            vals[:, i] = self._interp_nan_1d(vals[:, i])

        left_x, left_y, right_x, right_y = [vals[:, i] for i in range(4)]
        roll = np.clip((right_x - 0.5) * 2.0, -1.0, 1.0)
        pitch = np.clip(-(right_y - 0.5) * 2.0, -1.0, 1.0)
        yaw = np.clip((left_x - 0.5) * 2.0, -1.0, 1.0)
        throttle = np.clip(-(left_y - 0.5) * 2.0, -1.0, 1.0)
        actions = np.stack([roll, pitch, yaw, throttle], axis=-1).astype(np.float32)
        return actions


class OSDExtractor:
    def __init__(self, speed_roi, alt_roi, batt_roi):
        self.speed_roi = speed_roi
        self.alt_roi = alt_roi
        self.batt_roi = batt_roi

    @staticmethod
    def _crop(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = roi
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _ocr_number(img: np.ndarray) -> Optional[float]:
        if img.size == 0:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(bw, config="--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.,")
        m = re.search(r"([0-9]+(?:[\.,][0-9]+)?)", txt)
        if not m:
            return None
        return float(m.group(1).replace(",", "."))

    @staticmethod
    def _interp_series(arr: np.ndarray) -> np.ndarray:
        idx = np.arange(len(arr))
        mask = np.isfinite(arr)
        if not np.any(mask):
            return np.zeros_like(arr, dtype=np.float32)
        arr[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
        return arr.astype(np.float32)

    def extract(self, frames_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        speed = np.full((len(frames_bgr),), np.nan, dtype=np.float32)
        alt = np.full((len(frames_bgr),), np.nan, dtype=np.float32)
        batt = np.full((len(frames_bgr),), np.nan, dtype=np.float32)

        for i, f in enumerate(frames_bgr):
            speed[i] = self._ocr_number(self._crop(f, self.speed_roi)) or np.nan
            alt[i] = self._ocr_number(self._crop(f, self.alt_roi)) or np.nan
            batt[i] = self._ocr_number(self._crop(f, self.batt_roi)) or np.nan

        valid_ratio = np.mean(np.isfinite(speed) | np.isfinite(alt) | np.isfinite(batt))
        has_osd = bool(valid_ratio >= 0.1)

        speed = self._interp_series(speed).reshape(-1, 1)
        alt = self._interp_series(alt).reshape(-1, 1)
        batt = self._interp_series(batt).reshape(-1, 1)
        return speed, alt, batt, has_osd


class CrashDetector:
    def __init__(self, flow_threshold: float, brightness_drop: float, sharpness_drop: float, near_miss_flow_threshold: float):
        self.flow_threshold = flow_threshold
        self.brightness_drop = brightness_drop
        self.sharpness_drop = sharpness_drop
        self.near_miss_flow_threshold = near_miss_flow_threshold
        self._raft = None
        self._raft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

            self._raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(self._raft_device).eval()
        except Exception:
            self._raft = None

    @torch.no_grad()
    def _flow_mag(self, prev: np.ndarray, cur: np.ndarray) -> float:
        if self._raft is None:
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            return float(np.mean(np.linalg.norm(flow, axis=-1)))
        p = torch.from_numpy(prev).float().to(self._raft_device) / 255.0
        c = torch.from_numpy(cur).float().to(self._raft_device) / 255.0
        p = p.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        c = c.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        flow = self._raft(p, c)[-1]
        return float(flow.norm(dim=1).mean().item())

    def detect(self, frames_gray: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
        n = len(frames_gray)
        is_terminal = np.zeros((n,), dtype=bool)
        crashes: List[int] = []
        near_misses: List[int] = []
        if n < 3:
            return is_terminal, crashes, near_misses

        prev = frames_gray[0]
        prev_b = float(np.mean(prev))
        prev_s = float(cv2.Laplacian(prev, cv2.CV_32F).var())

        for i in range(1, n):
            cur = frames_gray[i]
            flow_mag = self._flow_mag(prev, cur)
            b = float(np.mean(cur))
            s = float(cv2.Laplacian(cur, cv2.CV_32F).var())

            brightness_drop = prev_b - b
            sharpness_drop = prev_s - s

            if flow_mag > self.flow_threshold and (brightness_drop > self.brightness_drop or sharpness_drop > self.sharpness_drop):
                is_terminal[i] = True
                crashes.append(i)
            elif flow_mag > self.near_miss_flow_threshold:
                near_misses.append(i)

            prev, prev_b, prev_s = cur, b, s

        return is_terminal, crashes, near_misses


class DepthEstimator:
    def __init__(self, model_id: str, batch_size: int, device: Optional[str] = None):
        from transformers import pipeline

        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_idx = 0 if self.device == "cuda" else -1
        self.pipe = pipeline("depth-estimation", model=model_id, device=device_idx)

    @torch.no_grad()
    def estimate(self, frames_bgr: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
        H, W = out_size
        outputs: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self.batch_size):
            chunk = frames_bgr[i : i + self.batch_size]
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in chunk]
            preds = self.pipe(images=rgb, batch_size=len(rgb))
            for pred in preds:
                d = np.array(pred["depth"], dtype=np.float32)
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
                p2, p98 = np.percentile(d, [2, 98])
                if p98 <= p2:
                    d_norm = np.zeros_like(d, dtype=np.uint8)
                else:
                    d_clip = np.clip((d - p2) / (p98 - p2), 0.0, 1.0)
                    d_norm = (d_clip * 255.0).astype(np.uint8)
                outputs.append(d_norm[..., None])
        return np.stack(outputs, axis=0)


class VideoReader:
    def __init__(self, resize_hw: Tuple[int, int], forced_fps: Optional[float] = None):
        self.resize_h, self.resize_w = resize_hw
        self.forced_fps = forced_fps

    def read(self, path: Path) -> Tuple[np.ndarray, float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = self.forced_fps if self.forced_fps is not None else (native_fps if native_fps > 1e-6 else 30.0)
        frames = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
            frames.append(frame)

        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded: {path}")

        return np.stack(frames, axis=0), float(fps)


def mask_handcam(frames_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    out = frames_bgr.copy()
    out[:, y0:y1, x0:x1] = 0
    return out


class EventClipExporter:
    def __init__(self, root: Path):
        self.root = root
        for sub in [
            "crashes/gray", "crashes/depth", "near_misses/gray", "near_misses/depth"
        ]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def export_events(
        self,
        video_stem: str,
        fps: float,
        crash_indices: Sequence[int],
        near_indices: Sequence[int],
        gray: np.ndarray,
        depth: np.ndarray,
    ) -> None:
        window = max(1, int(round(fps)))

        def dump(indices: Sequence[int], prefix: str):
            for idx in indices:
                s = max(0, idx - window)
                e = min(len(gray), idx + window + 1)
                for j in range(s, e):
                    name = f"{video_stem}_ev{idx:06d}_f{j:06d}.png"
                    cv2.imwrite(str(self.root / f"{prefix}/gray" / name), gray[j])
                    cv2.imwrite(str(self.root / f"{prefix}/depth" / name), depth[j, ..., 0])

        dump(crash_indices, "crashes")
        dump(near_indices, "near_misses")


class ParquetExporter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export_segment(
        self,
        stem: str,
        segment_id: int,
        frames_gray: np.ndarray,
        frames_depth: np.ndarray,
        frames_cam_overlay: np.ndarray,
        actions: np.ndarray,
        speeds: np.ndarray,
        altitudes: np.ndarray,
        batteries: np.ndarray,
        is_terminal: np.ndarray,
        drone_id: int,
        has_osd: bool,
    ) -> Path:
        record = {
            "frames_gray": [frames_gray.astype(np.uint8)],
            "frames_depth": [frames_depth.astype(np.uint8)],
            "frames_cam_overlay": [frames_cam_overlay.astype(np.uint8)],
            "actions": [actions.astype(np.float32)],
            "speeds": [speeds.astype(np.float32)],
            "altitudes": [altitudes.astype(np.float32)],
            "batteries": [batteries.astype(np.float32)],
            "is_terminal": [is_terminal.astype(bool)],
            "drone_id": [int(drone_id)],
            "has_osd": [bool(has_osd)],
        }
        df = pd.DataFrame(record)
        table = pa.Table.from_pandas(df, preserve_index=False)
        out_path = self.out_dir / f"{stem}_seg{segment_id:04d}.parquet"
        pq.write_table(table, out_path, compression="snappy")
        return out_path


class FPVPreprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.reader = VideoReader((cfg.resize_h, cfg.resize_w), cfg.fps)
        self.segmenter = SceneSegmenter(cfg.scene_threshold, cfg.min_scene_len_frames)
        self.stick_tracker = StickTracker(cfg.handcam_roi, cfg.stick_hsv_low, cfg.stick_hsv_high)
        self.osd_extractor = OSDExtractor(cfg.osd_speed_roi, cfg.osd_alt_roi, cfg.osd_batt_roi)
        self.crash_detector = CrashDetector(
            cfg.crash_flow_threshold,
            cfg.crash_brightness_drop,
            cfg.crash_sharpness_drop,
            cfg.near_miss_flow_threshold,
        )
        self.depth_estimator = DepthEstimator(cfg.depth_model_id, cfg.depth_batch_size)
        self.drone_classifier = DroneClassifier(cfg.drone_num_classes)

        root = Path(cfg.output_root)
        self.parquet_exporter = ParquetExporter(root / "parquet")
        self.event_exporter = EventClipExporter(root / "event_clips")

    def _iter_videos(self) -> List[Path]:
        vids = [p for p in Path(self.cfg.input_root).rglob("*") if p.suffix.lower() in VIDEO_EXTS]
        vids.sort()
        return vids

    def _prepare_segment(self, frames_bgr_seg: np.ndarray, fps: float, stem: str, seg_id: int) -> Optional[Dict]:
        if len(frames_bgr_seg) < self.cfg.min_scene_len_frames:
            return None

        actions = self.stick_tracker.extract_actions(frames_bgr_seg)
        speeds, altitudes, batteries, has_osd = self.osd_extractor.extract(frames_bgr_seg)
        masked = mask_handcam(frames_bgr_seg, self.cfg.handcam_roi)
        cam_overlay = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr_seg], axis=0)[..., None]

        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in masked], axis=0)
        gray = gray[..., None]

        depth = self.depth_estimator.estimate(masked, (self.cfg.resize_h, self.cfg.resize_w))

        is_terminal, crash_idx, near_idx = self.crash_detector.detect(gray[..., 0])

        fp = self.drone_classifier.compute_fingerprint(gray[..., 0], n_frames=self.cfg.drone_fingerprint_frames)
        drone_id = self.drone_classifier.classify(fp)

        self.event_exporter.export_events(stem, fps, crash_idx, near_idx, gray[..., 0], depth)

        return {
            "frames_gray": gray,
            "frames_depth": depth,
            "frames_cam_overlay": cam_overlay,
            "actions": actions,
            "speeds": speeds,
            "altitudes": altitudes,
            "batteries": batteries,
            "is_terminal": is_terminal,
            "drone_id": drone_id,
            "has_osd": has_osd,
        }

    def run(self) -> None:
        videos = self._iter_videos()
        if not videos:
            print(f"No videos found under {self.cfg.input_root}")
            return

        print(f"Found {len(videos)} videos")
        for video_path in tqdm(videos, desc="videos"):
            try:
                frames_bgr, fps = self.reader.read(video_path)
            except Exception as e:
                print(f"[WARN] skip {video_path}: {e}")
                continue

            segs = self.segmenter.segment(video_path, len(frames_bgr))
            stem = video_path.stem

            for seg_id, (s, e) in enumerate(segs):
                chunk = frames_bgr[s:e]
                out = self._prepare_segment(chunk, fps, stem, seg_id)
                if out is None:
                    continue
                self.parquet_exporter.export_segment(stem, seg_id, **out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline FPV preprocessing to parquet+event clips")
    p.add_argument("--input_root", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--config_json", type=str, default=None)
    return p.parse_args()


def load_config(args: argparse.Namespace) -> PreprocessConfig:
    cfg = PreprocessConfig(input_root=args.input_root, output_root=args.output_root)
    if args.config_json:
        patch = json.loads(Path(args.config_json).read_text())
        base = asdict(cfg)
        base.update(patch)
        cfg = PreprocessConfig(**base)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    proc = FPVPreprocessor(cfg)
    proc.run()


if __name__ == "__main__":
    main()
