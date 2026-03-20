import pathlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FPVDataset(Dataset):
    def __init__(self, segment_dir: str, batch_length: int = 64):
        self.files = sorted(pathlib.Path(segment_dir).glob("*.npz"))
        self.batch_length = batch_length
        assert len(self.files) > 0, f"Keine .npz Dateien in {segment_dir} gefunden"
        print(f"FPVDataset: {len(self.files)} Segmente gefunden")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        frames = data["frames"]
        T = frames.shape[0]
        if T > self.batch_length:
            start = np.random.randint(0, T - self.batch_length)
            frames = frames[start : start + self.batch_length]
        elif T < self.batch_length:
            pad = np.zeros((self.batch_length - T, *frames.shape[1:]), dtype=np.uint8)
            frames = np.concatenate([frames, pad], axis=0)
        frames = torch.from_numpy(frames).float() / 255.0
        return {"image": frames}


class Augmentation:
    def __init__(self, config):
        self.noise_std      = float(config.noise_std)
        self.use_flip       = bool(config.use_flip)
        self.use_brightness = bool(config.use_brightness)
        self.brightness_std = float(config.brightness_std)
        self.use_cutout     = bool(config.use_cutout)
        self.cutout_size    = int(config.cutout_size)

    def __call__(self, x):
        if self.noise_std > 0:
            x = (x + self.noise_std * torch.randn_like(x)).clamp(0, 1)
        if self.use_flip and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
        if self.use_brightness:
            delta = self.brightness_std * torch.randn(x.shape[0], 1, 1, 1, device=x.device)
            x = (x + delta).clamp(0, 1)
        if self.use_cutout:
            B, T, H, W, C = x.shape
            y = torch.randint(0, H - self.cutout_size, (1,)).item()
            xc = torch.randint(0, W - self.cutout_size, (1,)).item()
            x[:, :, y:y + self.cutout_size, xc:xc + self.cutout_size, :] = 0
        return x


class OfflineTrainer:
    def __init__(self, config, dataset, logger, logdir):
        self.dataset          = dataset
        self.logger           = logger
        self.logdir           = pathlib.Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.steps            = int(config.steps)
        self.batch_per_gpu    = int(config.batch_per_gpu)
        self.checkpoint_every = int(config.checkpoint_every)
        self.log_every        = int(config.log_every)
        self.num_workers      = int(config.num_workers)
        self.augmentation     = Augmentation(config.augmentation)

    def _save_checkpoint(self, agent, step):
        path = self.logdir / f"ckpt_{step:07d}.pt"
        state = agent.module.state_dict() if isinstance(agent, nn.DataParallel) else agent.state_dict()
        torch.save({"step": step, "model": state}, path)
        print(f"  → Checkpoint gespeichert: {path.name}")

    def load_checkpoint(self, agent, path: str):
        ckpt = torch.load(path, map_location="cpu")
        target = agent.module if isinstance(agent, nn.DataParallel) else agent
        target.load_state_dict(ckpt["model"])
        print(f"Checkpoint geladen: {path} (Step {ckpt['step']})")
        return ckpt["step"]

    def _setup_parallel(self, agent):
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"DataParallel: {n_gpus} GPUs gefunden")
            agent = nn.DataParallel(agent)
        else:
            print(f"Einzelne GPU gefunden")
        return agent

    def _compute_loss(self, agent, batch, device):
        target = agent.module if isinstance(agent, nn.DataParallel) else agent
        images = batch["image"].to(device, non_blocking=True)
        aug_a = self.augmentation(images)
        aug_b = self.augmentation(images)
        z_a = target.encode(aug_a)
        z_b = target.encode(aug_b)
        loss, metrics = target.barlow_loss(z_a, z_b)
        return loss, metrics

    def begin(self, agent):
        device = next(agent.parameters()).device
        agent = self._setup_parallel(agent)
        agent.train()

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_per_gpu,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

        print(f"\nStarte Offline Training")
        print(f"  Segmente:    {len(self.dataset)}")
        print(f"  Batch/GPU:   {self.batch_per_gpu}")
        print(f"  Ziel-Steps:  {self.steps}")
        print(f"  Checkpoints: alle {self.checkpoint_every} Steps\n")

        target = agent.module if isinstance(agent, nn.DataParallel) else agent
        optimizer = target.get_optimizer()
        running_loss = 0.0
        step = 0

        while step < self.steps:
            for batch in loader:
                if step >= self.steps:
                    break

                optimizer.zero_grad()
                loss, metrics = self._compute_loss(agent, batch, device)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_norm=100.0)
                optimizer.step()

                running_loss += loss.item()
                step += 1

                if step % self.log_every == 0:
                    avg_loss = running_loss / self.log_every
                    print(f"Step {step:6d}/{self.steps} | Loss: {avg_loss:.4f}", end="")
                    for k, v in metrics.items():
                        print(f" | {k}: {v:.4f}", end="")
                    print()
                    self.logger.scalar("train/loss", avg_loss)
                    for k, v in metrics.items():
                        self.logger.scalar(f"train/{k}", v)
                    self.logger.write(step)
                    running_loss = 0.0

                if step % self.checkpoint_every == 0:
                    self._save_checkpoint(agent, step)

        self._save_checkpoint(agent, step)
        print(f"\nTraining abgeschlossen nach {step} Steps")