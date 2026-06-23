

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

def _frames_to_pil(frames: np.ndarray):
    

    from PIL import Image
    T, H, W, C = frames.shape
    
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    canvas = np.concatenate(list(frames), axis=1)   
    if C == 1:
        canvas = canvas[..., 0]
    return Image.fromarray(canvas)

class MoondreamLabeler:
    

    _DEFAULT_ID  = "vikhyatk/moondream2"
    _DEFAULT_REV = "2025-01-09"

    _PROMPT = (
        "These frames show a drone flight captured by a forward-facing camera. "
        "Describe the intended flight manoeuvre as a short imperative command "
        "(≤12 words) a pilot might have given *before* the flight began. "
        "Focus on spatial direction, obstacles, and objects in the scene. "
        "Reply with the command only, no explanation."
    )

    def __init__(
        self,
        model_id:       str = _DEFAULT_ID,
        revision:       str = _DEFAULT_REV,
        device:         str = "mps",
        max_new_tokens: int = 32,
    ):
        self._model_id       = model_id
        self._revision       = revision
        self._device         = device
        self._max_new_tokens = max_new_tokens
        self._model          = None
        self._tokenizer      = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers>=4.36 is required for MoondreamLabeler.\n"
                "Install with:  pip install transformers"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, revision=self._revision, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            revision          = self._revision,
            trust_remote_code = True,
            torch_dtype       = torch.float16,
        ).to(self._device).eval()

    def label(self, frames: np.ndarray) -> str:
        
        self._ensure_loaded()
        pil_img = _frames_to_pil(frames)
        enc = self._model.encode_image(pil_img)
        answer = self._model.answer_question(
            enc, self._PROMPT, self._tokenizer,
            num_beams        = 1,
            max_new_tokens   = self._max_new_tokens,
        )
        return answer.strip()

@dataclass
class LabelledSegment:
    
    episode_id: int
    start_step: int
    frames: np.ndarray          
    instruction: str  = ""
    z_text: torch.Tensor | None = None   

class RelabelBuffer:
    

    def __init__(
        self,
        base_buffer,
        labeler:       MoondreamLabeler,
        text_encoder,                       
        clip_len:      int = 16,
        label_every:   int = 500,
        cache_size:    int = 10_000,
        text_dim:      int = 384,
    ):
        self._buf          = base_buffer
        self._labeler      = labeler
        self._text_enc     = text_encoder
        self._clip_len     = clip_len
        self._label_every  = label_every
        self._cache_size   = cache_size
        self._text_dim     = text_dim
        self._step         = 0

        
        self._cache: dict[int, list[LabelledSegment]] = {}
        self._lock  = threading.Lock()

        
        self._queue:  queue.Queue = queue.Queue(maxsize=32)
        self._worker  = threading.Thread(target=self._label_loop, daemon=True)
        self._worker.start()

    

    def sample(self, batch_size: int) -> dict:
        
        batch = self._buf.sample(batch_size)
        self._step += 1
        if self._step % self._label_every == 0:
            self._enqueue_random_clip()
        batch["z_text"] = self._lookup_z_text(batch)
        return batch

    

    def _enqueue_random_clip(self) -> None:
        
        try:
            clip_data = self._buf.sample_clip(self._clip_len)
        except Exception:
            return
        if clip_data is None:
            return
        seg = LabelledSegment(
            episode_id = int(clip_data.get("ep_id", -1)),
            start_step = int(clip_data.get("start_step", 0)),
            frames     = clip_data["frames"],
        )
        try:
            self._queue.put_nowait(seg)
        except queue.Full:
            pass  

    def _label_loop(self) -> None:
        
        while True:
            seg: LabelledSegment = self._queue.get()
            try:
                seg.instruction = self._labeler.label(seg.frames)
                z = self._text_enc(seg.instruction)          
                seg.z_text = z.detach().cpu()
                with self._lock:
                    segs = self._cache.setdefault(seg.episode_id, [])
                    segs.append(seg)
                    
                    total = sum(len(v) for v in self._cache.values())
                    if total > self._cache_size:
                        oldest_ep = min(self._cache)
                        del self._cache[oldest_ep]
            except Exception as exc:
                print(f"[RelabelBuffer] labelling error: {exc}")
            finally:
                self._queue.task_done()

    def _lookup_z_text(self, batch: dict) -> torch.Tensor:
        
        B   = batch["action"].shape[0]
        out = torch.zeros(B, 1, self._text_dim, dtype=torch.float32)

        ep_ids     = batch.get("ep_id")
        step_ids   = batch.get("step")
        if ep_ids is None:
            return out

        with self._lock:
            for i in range(B):
                ep_id = int(ep_ids[i])
                step  = int(step_ids[i]) if step_ids is not None else -1
                segs  = self._cache.get(ep_id, [])
                best  = _best_segment(segs, step, self._clip_len)
                if best is not None and best.z_text is not None:
                    out[i] = best.z_text[0]   

        return out

def _best_segment(
    segs: list[LabelledSegment],
    query_step: int,
    clip_len: int,
) -> LabelledSegment | None:
    
    best, best_dist = None, float("inf")
    for seg in segs:
        if seg.z_text is None:
            continue
        mid  = seg.start_step + clip_len // 2
        dist = abs(mid - query_step)
        if dist < best_dist:
            best, best_dist = seg, dist
    return best

def relabel_dataset(
    clips:        list[np.ndarray],
    labeler:      MoondreamLabeler,
    text_encoder,
    n_workers:    int = 2,
) -> list[tuple[np.ndarray, torch.Tensor]]:
    

    results: list[tuple[np.ndarray, torch.Tensor]] = [None] * len(clips)  
    lock = threading.Lock()

    def worker(idx: int, clip: np.ndarray) -> None:
        try:
            instruction = labeler.label(clip)
            z = text_encoder(instruction).detach().cpu()
            with lock:
                results[idx] = (clip, z)
        except Exception as exc:
            print(f"[relabel_dataset] clip {idx} failed: {exc}")

    threads = []
    sem = threading.Semaphore(n_workers)
    for idx, clip in enumerate(clips):
        sem.acquire()
        t = threading.Thread(target=lambda i=idx, c=clip: (worker(i, c), sem.release()))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    return [r for r in results if r is not None]
