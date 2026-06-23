from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import mlx.core as mx

@dataclass
class RSSMState:
    

    stoch: mx.array  
    deter: mx.array  
    prev_action: mx.array  
    prev_filtered_action: mx.array  

@dataclass
class ContextState:
    

    flat_stoch: mx.array  
    deter: mx.array  
    action: mx.array  
    valid_steps: mx.array  

@dataclass
class DreamerMLXState:
    

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
