import pathlib
import torch
import numpy as np
import torch.nn.functional as F
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator


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

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (T, H, W, 3), float32 in [0, 1]
        if not self.use_depth:
            return frames
        if self._depth_model is None:
            gray = frames.mean(dim=-1)
            return torch.stack([self._resize_depth(gray[t]) for t in range(gray.shape[0])], dim=0)

        # Depth Anything path (best effort in offline preprocessing).
        # In constrained environments we gracefully fallback to grayscale.
        try:
            import cv2  # type: ignore

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
            return torch.stack([self._resize_depth(gray[t]) for t in range(gray.shape[0])], dim=0)

class FPVDataset(IterableDataset):
    def __init__(self, config, batch_length: int = 64, require_osd: bool = False):
        super().__init__()
        self.mode = str(getattr(config.dataset, "mode", "streaming"))
        self.batch_length = batch_length
        self.require_osd  = require_osd
        self.use_depth = bool(getattr(config, "use_depth", False))
        self.img_height = int(getattr(config.model, "img_height", 256))
        self.img_width = int(getattr(config.model, "img_width", 256))
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
        for sample in self.ds:
            frames = torch.from_numpy(np.array(sample["frames"])).float() / 255.0
            T = frames.shape[0]
            start = 0
            if T > self.batch_length:
                start = torch.randint(0, T - self.batch_length, (1,)).item()
                frames = frames[start : start + self.batch_length]
            elif T < self.batch_length:
                continue
            if self.use_depth:
                depth = self.depth_preprocessor(frames)  # (T, 256, 256)
                depth = depth[..., None]
                diff = torch.zeros_like(depth)
                diff[1:] = depth[1:] - depth[:-1]
                image = torch.cat([depth, diff], dim=-1)
            else:
                diff = torch.zeros_like(frames)
                diff[1:] = frames[1:] - frames[:-1]
                image = torch.cat([frames, diff], dim=-1)
            is_first = torch.zeros(self.batch_length, dtype=torch.bool)
            is_first[0] = True
            is_last = torch.zeros(self.batch_length, dtype=torch.bool)
            is_last[-1] = True
            is_terminal = torch.zeros(self.batch_length, dtype=torch.bool)
            reward = torch.zeros(self.batch_length, 1, dtype=torch.float32)

            if self.require_osd and sample.get("has_osd", False):
                actions = torch.tensor(sample["actions"], dtype=torch.float32)[start : start + self.batch_length]
                if actions.shape[0] < self.batch_length:
                    action_dim = actions.shape[-1]
                    pad_act = torch.zeros((self.batch_length - actions.shape[0], action_dim), dtype=torch.float32)
                    actions = torch.cat([actions, pad_act], dim=0)
            else:
                actions = torch.zeros((self.batch_length, 4), dtype=torch.float32)

            drone_id = int(sample.get("drone_id", 0))
            drone_id = torch.full((self.batch_length,), drone_id, dtype=torch.long)

            yield {
                "image": image,
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
                "reward": reward,
                "action": actions,
                "drone_id": drone_id,
                "speed": torch.zeros(self.batch_length, 1, dtype=torch.float32),
                "crash": is_terminal.float().unsqueeze(-1),
            }

class Augmentation:
    def __init__(self, config):
        self.depth_noise_std = float(getattr(config, "depth_noise_std", config.noise_std))
        self.resolution_scale_min = float(getattr(config, "resolution_scale_min", 0.5))
        self.resolution_scale_max = float(getattr(config, "resolution_scale_max", 1.0))
        self.noise_std      = float(config.noise_std)
        self.use_flip       = bool(config.use_flip)
        self.use_brightness = bool(config.use_brightness)
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
        self.augmentation = Augmentation(config.augmentation)
        self.accumulation_steps = config.get("accumulation_steps", 8)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.accumulation_steps,
            mixed_precision="fp16"
        )
        self._augmentation_logged = False

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
                "is_first": batch["is_first"],
                "is_last": batch["is_last"],
                "is_terminal": batch["is_terminal"],
                "reward": batch["reward"],
                "action": batch["action"],
                "drone_id": batch["drone_id"],
            },
            batch_size=[batch_size, time_steps],
        )
        post, metrics = agent._cal_grad(data, initial=None)
        loss = metrics.pop("opt/loss")
        return loss, metrics

    def begin(self, agent):
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_per_gpu,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        optimizer = agent.get_optimizer()
        agent, optimizer, loader = self.accelerator.prepare(agent, optimizer, loader)
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
                    optimizer.zero_grad()

                    running_loss += loss.detach().item()
                    step += 1

                    if step % self.log_every == 0:
                        loss_tensor = torch.tensor(running_loss, device=self.accelerator.device)
                        avg_loss = self.accelerator.gather_for_metrics(loss_tensor).sum().item() / self.log_every
                        
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

    def begin(self, agent, env):
        optimizer = agent.get_optimizer()
        agent, optimizer = self.accelerator.prepare(agent, optimizer)
        state = agent.get_initial_state(B=1)
        obs, _ = env.reset()
        step = 0
        while step < self.steps:
            action, state = agent.act(obs, state, eval=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            _ = (reward, terminated, truncated, next_obs)
            step += 1
            if step % self.log_every == 0 and self.accelerator.is_main_process:
                self.logger.scalar("online/steps", step)
                self.logger.write(step)
