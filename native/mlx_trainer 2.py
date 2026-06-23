

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

from native.mlx_types import TransitionBatch, RSSMState

# ---------------------------------------------------------------------------
# Adaptive Gradient Clipping — Brock et al. 2021, per-parameter version.
# Mirrors trainer.py clip_grad_agc_() behaviour in the MLX functional style.
# ---------------------------------------------------------------------------

def _agc_mlx(
    grads,
    params,
    clip: float = 0.3,
    pmin: float = 1e-3,
):
    """Apply per-parameter AGC to a pytree of gradients.

    For each parameter p with gradient g:
        upper  = max(‖p‖, pmin) × clip
        scale  = min(upper / ‖g‖, 1)
        g_new  = g × scale

    Parameters and gradients must have identical pytree structure
    (both sourced from nn.Module.trainable_parameters()).
    """
    flat_g = tree_flatten(grads)
    flat_p = tree_flatten(params)
    new_g  = []
    for (key, g), (_, p) in zip(flat_g, flat_p):
        if not isinstance(g, mx.array) or g.size == 0:
            new_g.append((key, g))
            continue
        pf     = p.astype(mx.float32)
        gf     = g.astype(mx.float32)
        pnorm  = mx.sqrt(mx.sum(mx.square(pf)) + 1e-12)
        gnorm  = mx.sqrt(mx.sum(mx.square(gf)) + 1e-12)
        upper  = mx.maximum(pnorm, mx.array(pmin, dtype=mx.float32)) * clip
        scale  = mx.minimum(upper / mx.maximum(gnorm, upper), mx.array(1.0, dtype=mx.float32))
        new_g.append((key, (gf * scale).astype(g.dtype)))
    from mlx.utils import tree_unflatten
    return tree_unflatten(new_g)

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
    clip_grad_norm:           float = 100.0   # global-norm fallback (logging only when AGC active)
    agc_clip:                 float = 0.3     # per-parameter AGC clip ratio (Brock et al. 2021)
    agc_pmin:                 float = 1e-3    # AGC minimum parameter norm floor
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
                 clip_grad_norm: float = 100.0,
                 agc_clip: float = 0.3,
                 agc_pmin: float = 1e-3):
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

            # Compute global gnorm for logging (cheap — reuses the already-built
            # grad list before AGC rescales individual parameters).
            flat_g = [g for _, g in tree_flatten(grads) if isinstance(g, mx.array)]
            if flat_g:
                gnorm = mx.sqrt(mx.sum(mx.stack([mx.sum(g ** 2) for g in flat_g])))
                # Per-parameter AGC — mirrors trainer.py clip_grad_agc_()
                grads = _agc_mlx(grads, model.trainable_parameters(),
                                  clip=agc_clip, pmin=agc_pmin)
            else:
                gnorm = mx.array(0.0)

            optimizer.update(model, grads)
            return total, gnorm, losses

        self._compiled = mx.compile(_step, inputs=_mutable, outputs=_mutable)

    def step(self, batch: dict) -> dict[str, Any]:
        total, gnorm, losses = self._compiled(batch)
        mx.eval(self.model.parameters(), self.optimizer.state)
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
                model_or_config, optimizer, self._tcfg.clip_grad_norm,
                agc_clip=self._tcfg.agc_clip, agc_pmin=self._tcfg.agc_pmin,
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
        self._engine = _MLXUpdateEngine(
            model, optimizer, self._tcfg.clip_grad_norm,
            agc_clip=self._tcfg.agc_clip, agc_pmin=self._tcfg.agc_pmin,
        )
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
        step        = 0
        num_eps     = 0
        t0          = time.perf_counter()

        print(f"[MLXTrainer] Online training | steps={cfg.steps}")

        while step < cfg.steps:
            img_mx = mx.array(obs["image"][None, None])         
            embed  = model.encoder(img_mx)
            d_emb  = mx.zeros((1, model.d_emb_dim))
            rssm_state, _ = model.rssm.obs_step(
                rssm_state,
                embed[:, 0],
                mx.array([[float(is_first)]]),
                d_emb,
            )
            feat   = model.rssm.get_feat(rssm_state.stoch, rssm_state.deter)
            pi_inp = mx.concatenate([feat, d_emb], axis=-1)
            action = model.actor(pi_inp, sample=True)
            mx.eval(action)
            action_np = np.array(action[0])

            # Write the newly computed action back into rssm_state.prev_action so
            # the next obs_step sees it through the motor-inertia filter.
            # Without this, the filter always reads the zeroed value from obs_step's
            # return rather than the action that was actually executed.
            rssm_state = RSSMState(
                stoch               = rssm_state.stoch,
                deter               = rssm_state.deter,
                prev_action         = action,          # shape (1, act_dim)
                prev_filtered_action= rssm_state.prev_filtered_action,
            )

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
