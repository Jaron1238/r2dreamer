import pathlib
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from accelerate import Accelerator
from buffer import Buffer


class DepthPreprocessor:
    def __init__(self, use_depth: bool = True, output_size: int = 256):
        self.use_depth = bool(use_depth)
        self.output_size = int(output_size)
        self._depth_model = None
        self._backend = "disabled"
        if not self.use_depth:
            return
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
            self._depth_model = DepthAnythingV2(encoder="vits")
            self._depth_model.eval()
            self._backend = "depth-anything-v2"
        except Exception:
            self._depth_model = None
            self._backend = "grayscale-fallback"

    @property
    def backend(self):
        return self._backend

    def _resize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = F.interpolate(
            depth[None, None],
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return depth.clamp(0.0, 1.0)

    def _resize_depth_batch(self, depth: torch.Tensor) -> torch.Tensor:
        # depth: (T, H, W)
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return depth.clamp(0.0, 1.0)

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames can be:
        # - RGB:            (T, H, W, 3)
        # - depth only:     (T, H, W) or (T, H, W, 1)
        # - depth + diff:   (T, H, W, 2)
        if not self.use_depth:
            return frames
        if frames.ndim == 3:
            # Already depth maps without channel dimension.
            if frames.max() > 1.5:
                frames = frames / 255.0
            return self._resize_depth_batch(frames)
        if frames.ndim == 4 and frames.shape[-1] in (1, 2):
            # Already depth-like input (depth or depth+diff). Keep only depth channel.
            depth = frames[..., 0]
            if depth.max() > 1.5:
                depth = depth / 255.0
            return self._resize_depth_batch(depth)
        if self._depth_model is None:
            gray = frames.mean(dim=-1)
            return self._resize_depth_batch(gray)

        # Depth Anything path (best effort in offline preprocessing).
        # In constrained environments we gracefully fallback to grayscale.
        try:
            outs = []
            for t in range(frames.shape[0]):
                rgb = (frames[t].cpu().numpy() * 255.0).astype(np.uint8)
                depth = self._depth_model.infer_image(rgb)  # np.ndarray(H, W)
                depth = torch.from_numpy(depth).float()
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                outs.append(self._resize_depth(depth))
            return torch.stack(outs, dim=0)
        except Exception:
            gray = frames.mean(dim=-1)
            return self._resize_depth_batch(gray)

class FPVDataset(IterableDataset):
    def __init__(self, config, batch_length: int = 64, require_osd: bool = False):
        super().__init__()
        self.mode = str(getattr(config.dataset, "mode", "streaming"))
        self.batch_length = batch_length
        self.require_osd  = require_osd
        self.use_depth = bool(getattr(config, "use_depth", False))
        self.action_dim = int(getattr(config.model, "act_dim", 4))
        self.raw_image_mode = str(getattr(config.dataset, "raw_image_mode", "grayscale"))
        self.img_height = int(getattr(config.model, "img_height", 256))
        self.img_width = int(getattr(config.model, "img_width", 256))
        self.burn_in_steps = int(getattr(config.trainer, "burn_in_steps", 5))
        self.phase = int(getattr(config, "phase", 1))
        self.telemetry_jitter_std = float(getattr(config.dataset, "telemetry_jitter_std", 0.02))
        self.telemetry_stale_prob = float(getattr(config.dataset, "telemetry_stale_prob", 0.2))
        self.raw_source = SafetyRawImageSource(
            config.dataset.get("raw_dataset", None),
            output_size=(self.img_height, self.img_width),
            raw_image_mode=self.raw_image_mode,
        )
        self.depth_preprocessor = DepthPreprocessor(
            use_depth=self.use_depth,
            output_size=min(self.img_height, self.img_width),
        )
        from datasets import load_dataset
        self.ds = load_dataset(config.dataset.hf_repo, streaming=True, split="train")
        if self.require_osd:
            self.ds = self.ds.filter(lambda x: x.get("has_osd", False))
        shuffle_buffer = int(getattr(config.dataset, "shuffle_buffer", 1000))
        if shuffle_buffer > 0:
            self.ds = self.ds.shuffle(seed=42, buffer_size=shuffle_buffer)
            
        print(
            f"FPVDataset (Streaming): Repo '{config.dataset.hf_repo}' | "
            f"OSD-Pflicht: {self.require_osd} | use_depth={self.use_depth} "
            f"(backend={self.depth_preprocessor.backend})"
        )

    def __iter__(self):
        worker_info = get_worker_info()
        ds_iter = self.ds
        if worker_info is not None and worker_info.num_workers > 1:
            ds_iter = ds_iter.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        for sample in ds_iter:
            gray_full = torch.from_numpy(np.array(sample["frames_gray"][0])).float() / 255.0
            gray_full = self._ensure_channel_last(gray_full)
            rgb_full = None
            if "frames_rgb" in sample:
                rgb_full = torch.from_numpy(np.array(sample["frames_rgb"][0])).float() / 255.0
                rgb_full = self._ensure_channel_last(rgb_full)
            depth_full = torch.from_numpy(np.array(sample["frames_depth"][0])).float() / 255.0
            depth_full = self._ensure_channel_last(depth_full)
            is_terminal_full = torch.from_numpy(np.array(sample["is_terminal"][0])).bool()

            T = gray_full.shape[0]
            if T < self.batch_length:
                continue

            if T == self.batch_length:
                start = 0
            else:
                start = torch.randint(0, T - self.batch_length + 1, (1,)).item()

            raw_image = gray_full[start : start + self.batch_length]
            raw_image = self._resize_sequence(raw_image)
            raw_image = self._build_raw_image(raw_image)

            if rgb_full is not None:
                rgb = rgb_full[start : start + self.batch_length]
                rgb = self._resize_sequence(rgb)
                rgb = rgb[..., :3]
            else:
                rgb = self._resize_sequence(raw_image)
                if rgb.shape[-1] == 1:
                    rgb = rgb.repeat(1, 1, 1, 3)
                elif rgb.shape[-1] > 3:
                    rgb = rgb[..., :3]

            depth = depth_full[start : start + self.batch_length]
            depth = self._resize_sequence(depth)
            if depth.ndim == 3:
                depth = depth.unsqueeze(-1)
            is_terminal = is_terminal_full[start : start + self.batch_length]
            diff = torch.zeros_like(rgb)
            diff[1:] = rgb[1:] - rgb[:-1]
            image = torch.cat([rgb, diff], dim=-1)

            reward = torch.ones((self.batch_length, 1), dtype=torch.float32)
            if is_terminal.any():
                crash_idx = torch.where(is_terminal)[0]
                reward[crash_idx] = -1.0

            safety = self.raw_source.sample_sequence(self.batch_length)
            is_first = torch.zeros(self.batch_length, dtype=torch.bool)
            is_first[0] = True
            burn_in_mask = torch.ones(self.batch_length, dtype=torch.bool)
            burn_in_mask[: min(self.burn_in_steps, self.batch_length)] = False
            is_last = torch.zeros(self.batch_length, dtype=torch.bool)
            is_last[-1] = True

            if self.require_osd and sample.get("has_osd", False):
                actions_full = torch.from_numpy(np.array(sample["actions"][0])).float()
                actions = actions_full[start : start + self.batch_length]
            else:
                actions = torch.zeros((self.batch_length, self.action_dim), dtype=torch.float32)
            osd, has_osd = self._extract_osd(sample, start, self.batch_length)
            cam_overlay, has_cam_overlay = self._extract_cam_overlay(sample, start, self.batch_length, raw_image)

            speed_full = torch.from_numpy(np.array(sample["speeds"][0])).float() if "speeds" in sample else None
            speed = speed_full[start : start + self.batch_length] if speed_full is not None else torch.zeros(
                self.batch_length, 1, dtype=torch.float32
            )
            altitude_full = torch.from_numpy(np.array(sample["altitudes"][0])).float() if "altitudes" in sample else None
            altitude = altitude_full[start : start + self.batch_length] if altitude_full is not None else torch.zeros(
                self.batch_length, 1, dtype=torch.float32
            )
            if self.phase == 2:
                speed = self._apply_telemetry_noise(speed, self.telemetry_jitter_std, self.telemetry_stale_prob)
                altitude = self._apply_telemetry_noise(altitude, self.telemetry_jitter_std, self.telemetry_stale_prob)
            battery_full = torch.from_numpy(np.array(sample["batteries"][0])).float() if "batteries" in sample else None
            battery = battery_full[start : start + self.batch_length] if battery_full is not None else torch.zeros(
                self.batch_length, 1, dtype=torch.float32
            )

            drone_value = sample.get("drone_id", 0)
            if isinstance(drone_value, (list, tuple, np.ndarray)):
                drone_value = drone_value[0]
            drone_id = int(drone_value)
            drone_id = torch.full((self.batch_length,), drone_id, dtype=torch.long)

            if safety is not None:
                inj_raw_image = safety["raw_image"]
                inj_crash = safety["crash"]
            else:
                inj_raw_image = torch.zeros_like(raw_image)
                inj_crash = torch.zeros_like(is_terminal.float().unsqueeze(-1))

            yield {
                "image": image,
                "raw_image": raw_image,
                "is_first": is_first,
                "burn_in_mask": burn_in_mask,
                "is_last": is_last,
                "is_terminal": is_terminal,
                "reward": reward,
                "action": actions,
                "drone_id": drone_id,
                "speed": speed,
                "altitude": altitude,
                "battery": battery,
                "crash": is_terminal.float().unsqueeze(-1),
                "inj_raw_image": inj_raw_image,
                "inj_crash": inj_crash,
                "osd": osd,
                "has_osd": has_osd,
                "cam_overlay": cam_overlay,
                "has_cam_overlay": has_cam_overlay,
                "depth_target": depth,
            }

    def _build_raw_image(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 3:
            frames = frames[..., None]
        if frames.shape[-1] == 2:
            frames = frames[..., :1]

        if self.raw_image_mode == "rgb":
            if frames.shape[-1] == 1:
                return frames.repeat(1, 1, 1, 3)
            return frames[..., :3]

        if frames.shape[-1] == 1:
            return frames
        return frames[..., :3].mean(dim=-1, keepdim=True)

    def _ensure_channel_last(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 3:
            return frames[..., None]
        return frames

    def _resize_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[1:3] == (self.img_height, self.img_width):
            return frames
        resized = F.interpolate(
            frames.permute(0, 3, 1, 2),
            size=(self.img_height, self.img_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1)

    def _apply_telemetry_noise(self, values: torch.Tensor, jitter_std: float, stale_prob: float) -> torch.Tensor:
        out = values.clone()
        if jitter_std > 0:
            out = out + torch.randn_like(out) * jitter_std
        if stale_prob > 0 and out.shape[0] > 1:
            stale_mask = torch.rand(out.shape[0] - 1, 1) < stale_prob
            out[1:] = torch.where(stale_mask, out[:-1], out[1:])
        return out

    def _extract_osd(self, sample, start: int, length: int):
        has_osd = bool(sample.get("has_osd", False))
        if has_osd and "osd" in sample:
            arr = torch.from_numpy(np.array(sample["osd"][0])).float()
            osd = arr[start : start + length]
        else:
            osd = torch.zeros((length, 8), dtype=torch.float32)
        return osd, torch.full((length, 1), float(has_osd), dtype=torch.float32)

    def _extract_cam_overlay(self, sample, start: int, length: int, fallback_image: torch.Tensor):
        keys = ("frames_cam_overlay", "cam_overlay", "overlay")
        overlay = None
        for key in keys:
            if key in sample:
                raw = torch.from_numpy(np.array(sample[key][0])).float()
                if raw.max() > 1.5:
                    raw = raw / 255.0
                raw = self._ensure_channel_last(raw)
                raw = raw[start : start + length]
                overlay = self._resize_sequence(raw)
                if overlay.shape[-1] > 1:
                    overlay = overlay.mean(dim=-1, keepdim=True)
                break
        has_overlay = overlay is not None
        if overlay is None:
            overlay = torch.zeros((length, fallback_image.shape[1], fallback_image.shape[2], 1), dtype=torch.float32)
        return overlay, torch.full((length, 1), float(has_overlay), dtype=torch.float32)


class SafetyRawImageSource:
    def __init__(self, raw_cfg, output_size, raw_image_mode: str):
        self.output_size = tuple(output_size)
        self.raw_image_mode = raw_image_mode
        self.enabled = bool(raw_cfg and raw_cfg.get("enabled", False))
        self.source = None
        self.good_files = []
        self.bad_files = []
        self.hf_ds = None
        self.hf_iter = None
        if not self.enabled:
            return
        self.source = str(raw_cfg.get("source", "folders"))
        if self.source == "folders":
            root = pathlib.Path(raw_cfg.get("root", ""))
            good_dir = root / "good"
            bad_dir = root / "bad"
            if good_dir.exists() or bad_dir.exists():
                self.good_files = sorted(good_dir.glob("*"))
                self.bad_files = sorted(bad_dir.glob("*"))
            else:
                crash_gray_dir = root / "gray"
                if crash_gray_dir.exists():
                    self.bad_files = sorted(crash_gray_dir.glob("*"))
                else:
                    self.bad_files = sorted(root.glob("*"))
            if not self.good_files and not self.bad_files:
                self.enabled = False
        elif self.source == "streaming":
            from datasets import load_dataset

            repo = raw_cfg.get("hf_repo", None)
            split = raw_cfg.get("split", "train")
            if not repo:
                self.enabled = False
                return
            stream = bool(raw_cfg.get("streaming", True))
            self.hf_ds = load_dataset(repo, split=split, streaming=stream)
            self.hf_iter = iter(self.hf_ds)
        else:
            self.enabled = False

    def sample_sequence(self, length: int):
        if not self.enabled:
            return None
        if self.source == "folders":
            raw_frames = []
            crash = []
            for _ in range(length):
                is_bad = bool(torch.rand(1).item() > 0.5)
                path = self._choose_file(is_bad)
                raw_frames.append(self._load_local_image(path))
                crash.append([1.0 if is_bad else 0.0])
            return {
                "raw_image": torch.stack(raw_frames, dim=0),
                "crash": torch.tensor(crash, dtype=torch.float32),
            }
        if self.source == "streaming":
            raw_frames = []
            crash = []
            for _ in range(length):
                item = self._next_stream_item()
                img = self._stream_item_to_image(item)
                raw_frames.append(img)
                label = float(item.get("crash", item.get("label", 0.0)))
                crash.append([label])
            return {
                "raw_image": torch.stack(raw_frames, dim=0),
                "crash": torch.tensor(crash, dtype=torch.float32),
            }
        return None

    def _choose_file(self, is_bad: bool):
        files = self.bad_files if is_bad and self.bad_files else self.good_files
        if not files:
            files = self.bad_files or self.good_files
        idx = torch.randint(0, len(files), (1,)).item()
        return files[idx]

    def _load_local_image(self, path: pathlib.Path):
        image = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.array(image)).float() / 255.0
        return self._postprocess(arr)

    def _next_stream_item(self):
        assert self.hf_iter is not None
        try:
            return next(self.hf_iter)
        except StopIteration:
            if self.hf_ds is None:
                raise
            self.hf_iter = iter(self.hf_ds)
            return next(self.hf_iter)

    def _stream_item_to_image(self, item):
        if "image" in item:
            image = item["image"]
            if isinstance(image, Image.Image):
                arr = torch.from_numpy(np.array(image.convert("RGB"))).float() / 255.0
            else:
                arr = torch.from_numpy(np.array(image)).float()
        elif "frames" in item:
            frames = np.array(item["frames"])
            arr = torch.from_numpy(frames[0]).float()
        else:
            arr = torch.zeros(*self.output_size, 3, dtype=torch.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        return self._postprocess(arr)

    def _postprocess(self, arr: torch.Tensor):
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim == 3 and arr.shape[-1] not in (1, 3):
            arr = arr[..., :3]
        if self.raw_image_mode == "grayscale" and arr.shape[-1] == 3:
            arr = arr.mean(dim=-1, keepdim=True)
        if self.raw_image_mode == "rgb" and arr.shape[-1] == 1:
            arr = arr.repeat(1, 1, 3)
        if arr.shape[:2] != self.output_size:
            arr = F.interpolate(
                arr.permute(2, 0, 1).unsqueeze(0),
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0)
        return arr.clamp(0.0, 1.0)

class Augmentation:
    def __init__(self, config, phase: int = 1, use_depth: bool = False):
        self.phase = int(phase)
        self.use_depth_data = bool(use_depth)
        self.depth_noise_std = float(getattr(config, "depth_noise_std", config.noise_std))
        self.resolution_scale_min = float(getattr(config, "resolution_scale_min", 0.5))
        self.resolution_scale_max = float(getattr(config, "resolution_scale_max", 1.0))
        self.noise_std      = float(config.noise_std)
        self.use_flip       = bool(config.use_flip) and self.phase != 2
        self.use_brightness = bool(config.use_brightness) and not self.use_depth_data
        self.brightness_std = float(config.brightness_std)
        self.use_cutout     = bool(config.use_cutout)
        self.cutout_size    = int(config.cutout_size)

    def active_transforms(self):
        transforms = []
        if self.noise_std > 0:
            transforms.append(f"noise(std={self.noise_std})")
        if self.depth_noise_std > 0:
            transforms.append(f"depth_noise(std={self.depth_noise_std})")
        transforms.append(
            f"random_resolution_scale([{self.resolution_scale_min}, {self.resolution_scale_max}])"
        )
        if self.use_flip:
            transforms.append("horizontal_flip(p=0.5)")
        if self.use_brightness:
            transforms.append(f"brightness(std={self.brightness_std})")
        if self.use_cutout:
            transforms.append(f"cutout(size={self.cutout_size})")
        return transforms

    def enabled(self):
        return bool(self.active_transforms())

    def _apply_to_rgb(self, x):
        if self.noise_std > 0:
            x = (x + self.noise_std * torch.randn_like(x)).clamp(0, 1)
        if self.use_flip and torch.rand(1, device=x.device).item() > 0.5:
            x = torch.flip(x, dims=[3])
        if self.use_brightness:
            delta = self.brightness_std * torch.randn(x.shape[0], 1, 1, 1, 1, device=x.device)
            x = (x + delta).clamp(0, 1)
        if self.use_cutout:
            _, _, H, W, _ = x.shape
            cutout_h = min(self.cutout_size, H)
            cutout_w = min(self.cutout_size, W)
            y_max = H - cutout_h + 1
            x_max = W - cutout_w + 1
            y = torch.randint(0, y_max, (1,), device=x.device).item() if y_max > 1 else 0
            x0 = torch.randint(0, x_max, (1,), device=x.device).item() if x_max > 1 else 0
            x[:, :, y:y + cutout_h, x0:x0 + cutout_w, :] = 0
        return x

    def _random_resolution_scale(self, x):
        # x: (B, T, H, W, C)
        _, _, h, w, _ = x.shape
        scale = self.resolution_scale_min + torch.rand(1, device=x.device).item() * (
            self.resolution_scale_max - self.resolution_scale_min
        )
        scale = max(0.1, min(1.0, scale))
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        x_nchw = x.permute(0, 1, 4, 2, 3).reshape(-1, x.shape[-1], h, w)
        down = F.interpolate(x_nchw, size=(nh, nw), mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        return up.reshape(x.shape[0], x.shape[1], x.shape[-1], h, w).permute(0, 1, 3, 4, 2)

    def __call__(self, x):
        channels = x.shape[-1]
        if channels not in (2, 3, 6):
            raise ValueError(f"Expected 2, 3 or 6 image channels for augmentation, got {channels}.")

        x = self._random_resolution_scale(x)
        if channels == 2:
            depth = x[..., :1]
            if self.depth_noise_std > 0:
                depth = (depth + self.depth_noise_std * torch.randn_like(depth)).clamp(0, 1)
            diff = torch.zeros_like(depth)
            diff[:, 1:] = depth[:, 1:] - depth[:, :-1]
            return torch.cat([depth, diff], dim=-1)
        if channels == 3:
            return self._apply_to_rgb(x)

        rgb = self._apply_to_rgb(x[..., :3].clone())
        diff = torch.zeros_like(rgb)
        diff[:, 1:] = rgb[:, 1:] - rgb[:, :-1]
        return torch.cat([rgb, diff], dim=-1)


class OfflineTrainer:
    def __init__(self, config, dataset, logger, logdir):
        self.dataset = dataset
        self.logger = logger
        self.logdir = pathlib.Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.steps = int(config.steps)
        self.batch_per_gpu = int(config.batch_per_gpu)
        self.checkpoint_every = int(config.checkpoint_every)
        self.log_every = int(config.log_every)
        self.num_workers = int(config.num_workers)
        self.augmentation = Augmentation(
            config.augmentation,
            phase=int(getattr(config, "phase", 1)),
            use_depth=bool(getattr(config, "use_depth", False)),
        )
        self.accumulation_steps = config.get("accumulation_steps", 8)
        self.eval_batches = int(config.get("eval_batches", 8))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.accumulation_steps,
            mixed_precision="fp16"
        )
        self._augmentation_logged = False
        self._eval_loader = None

    def _save_checkpoint(self, step):
        ckpt_path = self.logdir / f"checkpoint-{step:07d}"
        self.accelerator.save_state(output_dir=str(ckpt_path))
        self.accelerator.print(f"Checkpoint: {ckpt_path.name}")

    def load_checkpoint(self, path):
        self.accelerator.load_state(path)
        self.accelerator.print(f"Loaded: {path}")

    def _compute_loss(self, agent, batch, augment=False):
        image = batch["image"]
        if augment and self.augmentation.enabled():
            image = self.augmentation(image)

        batch_size, time_steps = image.shape[:2]
        data = TensorDict(
            {
                "image": image,
                "raw_image": batch["raw_image"],
                "is_first": batch["is_first"],
                "burn_in_mask": batch["burn_in_mask"],
                "is_last": batch["is_last"],
                "is_terminal": batch["is_terminal"],
                "reward": batch["reward"],
                "action": batch["action"],
                "drone_id": batch["drone_id"],
                "speed": batch["speed"],
                "altitude": batch["altitude"],
                "battery": batch["battery"],
                "crash": batch["crash"],
                "inj_raw_image": batch["inj_raw_image"],
                "inj_crash": batch["inj_crash"],
                "osd": batch.get("osd", torch.zeros(batch_size, time_steps, 8, device=image.device)),
                "has_osd": batch.get("has_osd", torch.zeros(batch_size, time_steps, 1, device=image.device)),
                "cam_overlay": batch.get(
                    "cam_overlay",
                    torch.zeros(batch_size, time_steps, image.shape[2], image.shape[3], 1, device=image.device),
                ),
                "has_cam_overlay": batch.get("has_cam_overlay", torch.zeros(batch_size, time_steps, 1, device=image.device)),
            },
            batch_size=[batch_size, time_steps],
        )
        data = agent.preprocess(data)
        (_, _), total_loss, metrics = agent.compute_losses(data, initial=None)
        return total_loss, metrics

    @torch.no_grad()
    def _eval(self, agent, step):
        agent.eval()
        total_reward = 0.0
        count = 0
        eval_iter = iter(self._eval_loader)
        for _ in range(self.eval_batches):
            batch = next(eval_iter)
            image = batch["image"].to(self.accelerator.device)
            raw_image = batch["raw_image"].to(self.accelerator.device)
            batch_size, time_steps = image.shape[:2]
            data = TensorDict(
                {
                    "image": image,
                    "raw_image": raw_image,
                    "is_first": batch["is_first"].to(self.accelerator.device),
                    "burn_in_mask": batch["burn_in_mask"].to(self.accelerator.device),
                    "is_last": batch["is_last"].to(self.accelerator.device),
                    "is_terminal": batch["is_terminal"].to(self.accelerator.device),
                    "reward": batch["reward"].to(self.accelerator.device),
                    "action": batch["action"].to(self.accelerator.device),
                    "drone_id": batch["drone_id"].to(self.accelerator.device),
                    "speed": batch["speed"].to(self.accelerator.device),
                    "crash": batch["crash"].to(self.accelerator.device),
                    "osd": batch.get("osd", torch.zeros(batch_size, time_steps, 8)).to(self.accelerator.device),
                    "has_osd": batch.get("has_osd", torch.zeros(batch_size, time_steps, 1)).to(self.accelerator.device),
                    "cam_overlay": batch.get(
                        "cam_overlay", torch.zeros(batch_size, time_steps, image.shape[2], image.shape[3], 1)
                    ).to(self.accelerator.device),
                    "has_cam_overlay": batch.get("has_cam_overlay", torch.zeros(batch_size, time_steps, 1)).to(self.accelerator.device),
                },
                batch_size=[batch_size, time_steps],
            )
            (_, _), _, metrics = agent.compute_losses(agent.preprocess(data), initial=None)
            rew = metrics.get("rew", 0.0)
            total_reward += float(rew.item() if hasattr(rew, "item") else rew)
            count += 1
        agent.train()
        if self.accelerator.is_main_process:
            self.logger.scalar("eval/reward", total_reward / max(count, 1))
            self.logger.write(step)

    def begin(self, agent):
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_per_gpu,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self._eval_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_per_gpu,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=True,
            drop_last=True,
        )
        optimizer = agent._optimizer
        scheduler = agent._scheduler
        agent, optimizer, loader, scheduler = self.accelerator.prepare(agent, optimizer, loader, scheduler)
        train_agent = self.accelerator.unwrap_model(agent)
        agent.train()
        
        self.accelerator.print(f"Start Training | Processes: {self.accelerator.num_processes}")
        if not self._augmentation_logged:
            active_augmentations = ", ".join(self.augmentation.active_transforms()) or "none"
            self.accelerator.print(
                f"Training augmentations on image batches (RGB augmented, diff channels recomputed): {active_augmentations}"
            )
            self._augmentation_logged = True

        running_loss = 0.0
        step = 0

        while step < self.steps:
            for batch in loader:
                if step >= self.steps:
                    break

                with self.accelerator.accumulate(agent):
                    loss, metrics = self._compute_loss(agent, batch, augment=True)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(agent.parameters(), max_norm=100.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if self.accelerator.sync_gradients and hasattr(train_agent, "_update_slow_target"):
                        train_agent._update_slow_target()

                    running_loss += loss.detach().item()
                    step += 1

                    if step % self.log_every == 0:
                        loss_tensor = torch.tensor(running_loss, device=self.accelerator.device)
                        gathered_loss = self.accelerator.gather_for_metrics(loss_tensor)
                        avg_loss = gathered_loss.mean().item() / self.log_every
                        
                        self.accelerator.print(f"Step {step:6d} | Loss: {avg_loss:.4f}")

                        if self.accelerator.is_main_process:
                            self.logger.scalar("train/loss", avg_loss)
                            for k, v in metrics.items():
                                val = v.item() if hasattr(v, 'item') else v
                                self.logger.scalar(f"train/{k}", val)
                            self.logger.write(step)

                        running_loss = 0.0

                if step % self.checkpoint_every == 0:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        self._save_checkpoint(step)
                    self._eval(agent, step)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_checkpoint(step)


class OnlineTrainer:
    def __init__(self, config, logger, logdir):
        self.logger = logger
        self.logdir = pathlib.Path(logdir)
        self.steps = int(config.steps)
        self.log_every = int(config.log_every)
        self.checkpoint_every = int(config.checkpoint_every)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=int(config.get("accumulation_steps", 1)),
            mixed_precision="fp16",
        )
        self.num_eps = 0

    def begin(self, agent, env):
        replay_buffer = Buffer(agent.config.buffer)
        optimizer = agent._optimizer
        agent, optimizer = self.accelerator.prepare(agent, optimizer)
        state = agent.get_initial_state(B=1).to(self.accelerator.device)
        obs, _ = env.reset()
        obs = self._to_device_obs(obs, is_first=True)
        step = 0
        train_every = int(getattr(agent.config.trainer, "online_train_every", 1))
        learning_starts = int(getattr(agent.config.trainer, "online_learning_starts", agent.config.buffer.batch_length + 1))
        while step < self.steps:
            action, state = agent.act(obs, state, eval=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = self._to_device_obs(next_obs, is_first=False)

            done = bool(terminated or truncated)
            transition = TensorDict(
                {
                    "image": obs["image"],
                    "raw_image": obs.get("raw_image", obs["image"]),
                    "is_first": obs["is_first"],
                    "is_last": torch.tensor([done], dtype=torch.bool, device=self.accelerator.device),
                    "is_terminal": torch.tensor([bool(terminated)], dtype=torch.bool, device=self.accelerator.device),
                    "reward": torch.tensor([[reward]], dtype=torch.float32, device=self.accelerator.device),
                    "action": action.detach(),
                    "speed": obs.get("speed", torch.zeros(1, 1, device=self.accelerator.device)),
                    "drone_id": obs.get("drone_id", torch.zeros(1, dtype=torch.long, device=self.accelerator.device)),
                    "crash": torch.tensor([[float(terminated)]], dtype=torch.float32, device=self.accelerator.device),
                    "episode": torch.tensor([self.num_eps], dtype=torch.long, device=self.accelerator.device),
                    "stoch": state["stoch"].detach(),
                    "deter": state["deter"].detach(),
                },
                batch_size=[1],
            )
            replay_buffer.add_transition(transition)
            obs = next_obs

            if replay_buffer.count() >= learning_starts and (step + 1) % train_every == 0:
                metrics = agent.update(replay_buffer)
                if step % self.log_every == 0 and self.accelerator.is_main_process:
                    for k, v in metrics.items():
                        val = v.item() if hasattr(v, "item") else v
                        self.logger.scalar(f"online/{k}", val)

            if done:
                self.num_eps += 1
                obs, _ = env.reset()
                obs = self._to_device_obs(obs, is_first=True)
                state = agent.get_initial_state(B=1).to(self.accelerator.device)
            step += 1
            if step % self.log_every == 0 and self.accelerator.is_main_process:
                self.logger.scalar("online/steps", step)
                self.logger.write(step)

    def _to_device_obs(self, obs, *, is_first=False):
        out = {}
        for key, value in obs.items():
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            tensor = tensor.to(self.accelerator.device)
            if key == "drone_id":
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            if key in ("image", "raw_image") and tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            out[key] = tensor
        out["is_first"] = torch.tensor([bool(is_first)], dtype=torch.bool, device=self.accelerator.device)
        if "speed" not in out:
            out["speed"] = torch.zeros(1, 1, dtype=torch.float32, device=self.accelerator.device)
        if "drone_id" not in out:
            out["drone_id"] = torch.zeros(1, dtype=torch.long, device=self.accelerator.device)
        return out
