

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from native.mlx_types import TransitionBatch

def _np2mx(batch: dict) -> dict:
    
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = mx.array(v.astype(np.float32) if v.dtype == np.bool_ else v)
        elif isinstance(v, mx.array):
            out[k] = v
        else:
            out[k] = v
    return out

def _mx_save(path: pathlib.Path, model: nn.Module) -> None:
    
    flat = dict(tree_flatten(model.parameters()))
    sfx  = path.suffix.lower()
    if sfx == ".safetensors":
        mx.save_safetensors(str(path), flat)
    else:
            mx.savez(str(path), **flat)

def _mx_load(path: pathlib.Path, model: nn.Module) -> None:
    
    
    data   = mx.load(str(path))          
    model.load_weights(list(data.items()))

@dataclass
class TrainerConfig:
    lr:                       float = 4e-5
    eps:                      float = 1e-8
    clip_grad_norm:           float = 100.0
    max_active_memory_gb:     float = 24.0
    steps:                    int   = 100_000
    log_every:                int   = 50
    checkpoint_every:         int   = 500
    batch_length:             int   = 64
    batch_per_gpu:            int   = 8
    online_train_every:       int   = 1
    online_learning_starts:   int   = 64
    replay_buffer_capacity:   int   = 100_000

class MLXReplayBuffer:
    

    def __init__(self, capacity: int, obs_shape: tuple, act_dim: int):
        self._cap    = capacity
        self._ptr    = 0
        self._size   = 0
        H, W, C      = obs_shape
        self._image    = np.zeros((capacity, H, W, C), dtype=np.float32)
        self._action   = np.zeros((capacity, act_dim),  dtype=np.float32)
        self._reward   = np.zeros((capacity, 1),         dtype=np.float32)
        self._is_first = np.zeros((capacity, 1),         dtype=np.float32)
        self._is_last  = np.zeros((capacity, 1),         dtype=np.float32)
        self._is_term  = np.zeros((capacity, 1),         dtype=np.float32)
        self._speed    = np.zeros((capacity, 1),         dtype=np.float32)
        self._drone_id = np.zeros((capacity,),           dtype=np.int32)

    def add(self, image, action, reward, is_first, is_last, is_term,
            speed=0.0, drone_id=0):
        i = self._ptr
        self._image[i]    = image
        self._action[i]   = action
        self._reward[i]   = reward
        self._is_first[i] = float(is_first)
        self._is_last[i]  = float(is_last)
        self._is_term[i]  = float(is_term)
        self._speed[i]    = speed
        self._drone_id[i] = drone_id
        self._ptr  = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size: int, seq_len: int) -> dict:
        idxs = np.random.randint(0, self._size - seq_len, size=batch_size)
        s    = np.stack([np.arange(i, i + seq_len) % self._cap for i in idxs])
        return {
            "image":       mx.array(self._image[s]),
            "action":      mx.array(self._action[s]),
            "reward":      mx.array(self._reward[s]),
            "is_first":    mx.array(self._is_first[s]),
            "is_last":     mx.array(self._is_last[s]),
            "is_terminal": mx.array(self._is_term[s]),
            "speed":       mx.array(self._speed[s]),
            "drone_id":    mx.array(self._drone_id[s[:, 0]]),
        }

    def __len__(self) -> int:
        return self._size

class _MLXUpdateEngine:
    

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 clip_grad_norm: float = 100.0):
        self.model     = model
        self.optimizer = optimizer

        _mutable = [
            model.trainable_parameters(),
            optimizer.state,
            mx.random.state,
        ]

        def _step(batch: dict):
            def _loss_fn(m):
                total, losses, _ = m.compute_losses(batch)
                return total, losses

            (total, losses), grads = nn.value_and_grad(model, _loss_fn)(model)

            flat_g = [g for _, g in tree_flatten(grads) if isinstance(g, mx.array)]
            if flat_g:
                gnorm = mx.sqrt(mx.sum(mx.stack([mx.sum(g ** 2) for g in flat_g])))
                scale = mx.minimum(mx.array(clip_grad_norm) / (gnorm + 1e-6), mx.array(1.0))
                grads = tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads)
            else:
                gnorm = mx.array(0.0)

            optimizer.update(model, grads)
            return total, gnorm, losses

        self._compiled = mx.compile(_step, inputs=_mutable, outputs=_mutable)

    def step(self, batch: dict) -> dict[str, Any]:
        total, gnorm, losses = self._compiled(batch)
        mx.eval(self.model.parameters(), self.optimizer.state)
        if hasattr(self.model, "_update_nedreamer_ema_target"):
            self.model._update_nedreamer_ema_target()
            mx.eval(self.model.parameters())
        metrics = {
            "total_loss":    float(total.item()),
            "opt/grad_norm": float(gnorm.item()),
        }
        for k, v in losses.items():
            metrics[f"loss/{k}"] = float(v.item())
        return metrics

class MLXOnlineTrainer:
    

    def __init__(
        self,
        model_or_config,
        optimizer_or_logger=None,
        logdir=None,
        *,
        cfg: TrainerConfig | None = None,
        loss_fn: Callable | None  = None,
    ):
        if hasattr(model_or_config, "steps"):
            
            raw          = model_or_config
            self._tcfg   = TrainerConfig(
                lr                    = float(getattr(raw, "lr",                    4e-5)),
                eps                   = float(getattr(raw, "eps",                   1e-8)),
                clip_grad_norm        = float(getattr(raw, "clip_grad_norm",        100.0)),
                max_active_memory_gb  = float(getattr(raw, "max_active_memory_gb",  24.0)),
                steps                 = int(getattr(raw,   "steps",                 100_000)),
                log_every             = int(getattr(raw,   "log_every",             50)),
                checkpoint_every      = int(getattr(raw,   "checkpoint_every",      500)),
                batch_length          = int(getattr(raw,   "batch_length",          64)),
                batch_per_gpu         = int(getattr(raw,   "batch_per_gpu",         8)),
                online_train_every    = int(getattr(raw,   "online_train_every",    1)),
                online_learning_starts= int(getattr(raw,   "online_learning_starts",64)),
                replay_buffer_capacity= int(getattr(raw,   "replay_buffer_capacity",100_000)),
            )
            self._logger  = optimizer_or_logger
            self._logdir  = pathlib.Path(logdir) if logdir else pathlib.Path("logdir")
            self._logdir.mkdir(parents=True, exist_ok=True)
            self._engine  = None
        else:
            
            self._tcfg   = cfg or TrainerConfig()
            self._logger  = None
            self._logdir  = None
            optimizer     = optimizer_or_logger or optim.AdamW(
                learning_rate = self._tcfg.lr,
                eps           = self._tcfg.eps,
            )
            self._engine  = _MLXUpdateEngine(
                model_or_config, optimizer, self._tcfg.clip_grad_norm
            )

    def _memory_guard(self) -> None:
        try:
            active = mx.metal.get_active_memory()
        except AttributeError:
            return
        limit = int(self._tcfg.max_active_memory_gb * 1024 ** 3)
        if active > limit:
            raise MemoryError(
                f"MLX active memory {active / (1024**3):.2f} GiB "
                f"exceeds guard {self._tcfg.max_active_memory_gb:.2f} GiB"
            )

    def update(self, batch: dict) -> dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("update() requires Signature A.")
        self._memory_guard()
        batch = _np2mx(batch) if any(isinstance(v, np.ndarray) for v in batch.values()) else batch
        return self._engine.step(batch)

    def begin(self, model: nn.Module, env=None) -> None:
        optimizer    = optim.AdamW(learning_rate=self._tcfg.lr, eps=self._tcfg.eps)
        self._engine = _MLXUpdateEngine(model, optimizer, self._tcfg.clip_grad_norm)
        if env is not None:
            self._begin_online(model, env)
        else:
            self._begin_offline(model)

    def _begin_offline(self, model: nn.Module) -> None:
        
        from trainer import FPVDataset

        cfg     = self._tcfg
        dataset = FPVDataset(
            getattr(model, "config", None),
            batch_length = cfg.batch_length,
            require_osd  = False,
        )

        print(f"[MLXTrainer] Offline training | steps={cfg.steps}")
        step    = 0
        running = 0.0
        t0      = time.perf_counter()

        while step < cfg.steps:
            raw   = dataset.get_batch(cfg.batch_per_gpu)
            batch = _np2mx(raw)
            self._memory_guard()
            metrics = self._engine.step(batch)

            running += metrics.get("total_loss", 0.0)
            step    += 1

            if step % cfg.log_every == 0:
                avg = running / cfg.log_every
                sps = cfg.log_every / max(time.perf_counter() - t0, 1e-6)
                print(f"[MLX] step={step:6d} | loss={avg:.4f} | {sps:.1f} steps/s")
                if self._logger:
                    self._logger.scalar("train/loss", avg)
                    for k, v in metrics.items():
                        self._logger.scalar(f"train/{k}", v)
                    self._logger.write(step)
                running = 0.0
                t0      = time.perf_counter()

            if step % cfg.checkpoint_every == 0 and self._logdir:
                ckpt = self._logdir / f"ckpt_{step:07d}.safetensors"
                _mx_save(ckpt, model)
                print(f"[MLX] Checkpoint → {ckpt}")

        print("[MLXTrainer] Offline training complete.")

    def _begin_online(self, model: nn.Module, env) -> None:
        
        cfg       = self._tcfg
        act_dim   = model.rssm.cfg.act_dim
        obs_shape = getattr(model, "obs_shape", (84, 84, 3))
        replay    = MLXReplayBuffer(cfg.replay_buffer_capacity, obs_shape, act_dim)

        rssm_state  = model.rssm.initial(1)
        raw_obs, _  = env.reset()
        obs         = {k: np.array(v) for k, v in raw_obs.items()}
        is_first    = True
        ctx_stoch: list[np.ndarray] = []
        ctx_deter: list[np.ndarray] = []
        ctx_action: list[np.ndarray] = []
        safety_frames: list[np.ndarray] = []
        step        = 0
        num_eps     = 0
        t0          = time.perf_counter()

        print(f"[MLXTrainer] Online training | steps={cfg.steps}")

        while step < cfg.steps:
            img_mx = mx.array(obs["image"][None, None])
            embed  = model.encoder(img_mx)
            d_emb  = mx.zeros((1, model.d_emb_dim))
            if len(ctx_stoch) > 0:
                flat_stoch = mx.array(np.stack(ctx_stoch, axis=1))
                deter = mx.array(np.stack(ctx_deter, axis=1))
                act_hist = mx.array(np.stack(ctx_action, axis=1))
                valid_len = mx.array([flat_stoch.shape[1]], dtype=mx.int32)
                d_emb = model.context_encoder(flat_stoch, deter, act_hist, valid_len=valid_len)
            rssm_state, _ = model.rssm.obs_step(
                rssm_state,
                embed[:, 0],
                mx.array([[float(is_first)]]),
                d_emb,
            )
            feat   = model.rssm.get_feat(rssm_state.stoch, rssm_state.deter)
            pi_inp = mx.concatenate([feat, d_emb], axis=-1)
            action = model.actor(pi_inp, sample=True)

            safety_key = getattr(model, "safety_input_key", "image")
            safety_img = obs.get(safety_key, obs["image"])
            safety_frames.append(np.array(safety_img, dtype=np.float32))
            safety_frames = safety_frames[-int(model.safety_net.frame_stack):]
            if len(safety_frames) == int(model.safety_net.frame_stack):
                stacked = np.concatenate(safety_frames, axis=-1)[None, None]
                speed = mx.array([[[float(np.asarray(obs.get("speed", 0.0)).reshape(-1)[0])]]])
                safety_logit, safe_action = model.safety_net(mx.array(stacked), speed, mx.expand_dims(action, axis=1))
                unsafe = bool((mx.sigmoid(safety_logit)[0, 0, 0] > model.safety_threshold).item())
                if unsafe:
                    action = safe_action[:, 0]
            mx.eval(action)
            action_np = np.array(action[0])
            ctx_stoch.append(np.array(mx.reshape(rssm_state.stoch, (1, -1))[0]))
            ctx_deter.append(np.array(rssm_state.deter[0]))
            ctx_action.append(action_np.astype(np.float32))
            ctx_stoch = ctx_stoch[-int(model.ctx_len):]
            ctx_deter = ctx_deter[-int(model.ctx_len):]
            ctx_action = ctx_action[-int(model.ctx_len):]

            raw_next, reward, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)

            replay.add(
                image    = obs["image"],
                action   = action_np,
                reward   = float(reward),
                is_first = is_first,
                is_last  = done,
                is_term  = bool(terminated),
                speed    = float(obs.get("speed", 0.0)),
                drone_id = int(obs.get("drone_id", 0)),
            )
            obs      = {k: np.array(v) for k, v in raw_next.items()}
            is_first = done

            if (len(replay) >= cfg.online_learning_starts
                    and (step + 1) % cfg.online_train_every == 0):
                batch = replay.sample(cfg.batch_per_gpu, cfg.batch_length)
                self._memory_guard()
                metrics = self._engine.step(batch)
                if step % cfg.log_every == 0 and self._logger:
                    for k, v in metrics.items():
                        self._logger.scalar(f"online/{k}", v)

            if done:
                num_eps   += 1
                raw_obs, _ = env.reset()
                obs        = {k: np.array(v) for k, v in raw_obs.items()}
                rssm_state = model.rssm.initial(1)
                ctx_stoch.clear(); ctx_deter.clear(); ctx_action.clear(); safety_frames.clear()

            if step % cfg.log_every == 0:
                sps = cfg.log_every / max(time.perf_counter() - t0, 1e-6)
                print(
                    f"[MLX] step={step:6d} | buf={len(replay):6d} | "
                    f"eps={num_eps} | {sps:.1f} steps/s"
                )
                if self._logger:
                    self._logger.scalar("online/steps",    step)
                    self._logger.scalar("online/episodes", num_eps)
                    self._logger.write(step)
                t0 = time.perf_counter()

            if step % cfg.checkpoint_every == 0 and self._logdir:
                ckpt = self._logdir / f"ckpt_{step:07d}.safetensors"
                _mx_save(ckpt, model)
                print(f"[MLX] Checkpoint → {ckpt}")

            step += 1

        print("[MLXTrainer] Online training complete.")

    def save(self, path: str | pathlib.Path, model: nn.Module) -> None:
        _mx_save(pathlib.Path(path), model)

    def load(self, path: str | pathlib.Path, model: nn.Module) -> None:
        _mx_load(pathlib.Path(path), model)
