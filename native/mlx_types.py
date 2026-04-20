from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import mlx.core as mx


@dataclass
class RSSMState:
    """Explicit stateless RSSM state container for MLX modules."""

    stoch: mx.array  # (B, stoch, discrete)
    deter: mx.array  # (B, deter)
    prev_action: mx.array  # (B, action_dim)
    prev_filtered_action: mx.array  # (B, action_dim)


@dataclass
class ContextState:
    """Context buffer state for temporal drone embedding in MLX runtime."""

    flat_stoch: mx.array  # (B, ctx_len, flat_stoch)
    deter: mx.array  # (B, ctx_len, deter)
    action: mx.array  # (B, ctx_len, action_dim)
    valid_steps: mx.array  # (B,)


@dataclass
class DreamerMLXState:
    """Top-level state for online inference/training in native MLX."""

    rssm: RSSMState
    context: ContextState


class TransitionBatch(TypedDict):
    image: mx.array
    action: mx.array
    reward: mx.array
    is_first: mx.array
    is_terminal: mx.array
    speed: mx.array


class LoaderReport(TypedDict):
    loaded: int
    skipped: int
    missing: int
    details: dict[str, str]


def zeros_rssm_state(
    batch_size: int,
    stoch: int,
    discrete: int,
    deter: int,
    action_dim: int,
    *,
    dtype: Any = mx.float32,
) -> RSSMState:
    return RSSMState(
        stoch=mx.zeros((batch_size, stoch, discrete), dtype=dtype),
        deter=mx.zeros((batch_size, deter), dtype=dtype),
        prev_action=mx.zeros((batch_size, action_dim), dtype=dtype),
        prev_filtered_action=mx.zeros((batch_size, action_dim), dtype=dtype),
    )
