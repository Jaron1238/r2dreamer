from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from native.mlx_types import TransitionBatch


@dataclass
class TrainerConfig:
    lr: float = 4e-5
    max_active_memory_gb: float = 24.0


class MLXOnlineTrainer:
    """Online trainer with functional loss and compiled update step."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        *,
        cfg: TrainerConfig | None = None,
        loss_fn: Callable[[nn.Module, TransitionBatch], mx.array] | None = None,
    ):
        self.model = model
        self.cfg = cfg or TrainerConfig()
        self.optimizer = optimizer or optim.Adam(learning_rate=self.cfg.lr)
        self.loss_fn = loss_fn or self._default_loss_fn

        def _step(model: nn.Module, batch: TransitionBatch) -> tuple[mx.array, dict[str, mx.array]]:
            self._memory_guard()
            loss_and_grad = nn.value_and_grad(model, lambda m: self.loss_fn(m, batch))
            loss, grads = loss_and_grad(model)
            self.optimizer.update(model, grads)
            mx.eval(model.parameters(), self.optimizer.state)
            metrics = {
                "loss": loss,
                "grad_abs_mean": tree_map(lambda x: mx.mean(mx.abs(x)), grads),
            }
            return loss, metrics

        self._compiled_step = mx.compile(_step)

    def _memory_guard(self) -> None:
        # Apple Metal backend active memory in bytes.
        active_memory = mx.metal.get_active_memory()
        limit = int(self.cfg.max_active_memory_gb * 1024**3)
        if active_memory > limit:
            raise MemoryError(
                f"MLX active memory {active_memory / (1024**3):.2f} GiB exceeds guard {self.cfg.max_active_memory_gb:.2f} GiB"
            )

    def _default_loss_fn(self, model: nn.Module, batch: TransitionBatch) -> mx.array:
        pred = model(batch["image"])
        target = batch["reward"]
        return mx.mean(mx.square(pred - target))

    def update(self, batch: TransitionBatch) -> dict[str, Any]:
        loss, metrics = self._compiled_step(self.model, batch)
        return {
            "loss": float(loss.item()),
            "metrics": metrics,
        }
