import pathlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

class FPVDataset(IterableDataset):
    def __init__(self, config, batch_length: int = 64, require_osd: bool = False):
        super().__init__()
        self.batch_length = batch_length
        self.require_osd  = require_osd
        from datasets import load_dataset
        self.ds = load_dataset(config.dataset.hf_repo, streaming=True, split="train")
        if self.require_osd:
            self.ds = self.ds.filter(lambda x: x.get("has_osd", False))
        shuffle_buffer = int(getattr(config.dataset, "shuffle_buffer", 1000))
        if shuffle_buffer > 0:
            self.ds = self.ds.shuffle(seed=42, buffer_size=shuffle_buffer)
            
        print(f"FPVDataset (Streaming): Repo '{config.dataset.hf_repo}' | OSD-Pflicht: {self.require_osd}")

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
            diff = torch.zeros_like(frames)
            diff[1:] = frames[1:] - frames[:-1]
            frames_6ch = torch.cat([frames, diff], dim=-1)
            is_first = torch.zeros(self.batch_length, dtype=torch.bool)
            is_first[0] = True


            if self.require_osd and sample.get("has_osd", False):

                actions = torch.tensor(sample["actions"], dtype=torch.float32)[start : start + self.batch_length]
                

                if actions.shape[0] < self.batch_length:
                    pad_act = torch.zeros((self.batch_length - actions.shape[0], 4), dtype=torch.float32)
                    actions = torch.cat([actions, pad_act], dim=0)
            else:
                actions = torch.zeros((self.batch_length, 4), dtype=torch.float32)

            yield {
                "image": frames_6ch,
                "is_first": is_first,
                "action": actions,
                "drone_id": torch.tensor(sample["drone_id"], dtype=torch.long)
            }

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
            y  = torch.randint(0, H - self.cutout_size, (1,)).item()
            xc = torch.randint(0, W - self.cutout_size, (1,)).item()
            x[:, :, y:y + self.cutout_size, xc:xc + self.cutout_size, :] = 0
        return x


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

    def _save_checkpoint(self, step):
        ckpt_path = self.logdir / f"checkpoint-{step:07d}"
        self.accelerator.save_state(output_dir=str(ckpt_path))
        self.accelerator.print(f"Checkpoint: {ckpt_path.name}")

    def load_checkpoint(self, path):
        self.accelerator.load_state(path)
        self.accelerator.print(f"Loaded: {path}")

    def _compute_loss(self, agent, batch):
        data = {
            "image": batch["image"],
            "is_first": batch["is_first"],
            "action": batch["action"],
            "drone_id": batch["drone_id"]
        }
        post, metrics = agent._cal_grad(data, initial=None)
        loss = metrics.pop("opt/loss")
        return loss, metrics

    def begin(self, agent):
        is_stream = (self.dataset.mode == "streaming")
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
        
        running_loss = 0.0
        step = 0

        while step < self.steps:
            for batch in loader:
                if step >= self.steps:
                    break

                with self.accelerator.accumulate(agent):
                    loss, metrics = self._compute_loss(agent, batch)
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
        

