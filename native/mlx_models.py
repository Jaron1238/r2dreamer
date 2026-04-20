from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from native.mlx_types import RSSMState


@dataclass
class RSSMConfig:
    stoch: int
    deter: int
    hidden: int
    discrete: int
    act_dim: int
    embed_size: int
    d_emb_dim: int = 16
    obs_layers: int = 1
    img_layers: int = 2
    dyn_layers: int = 1
    blocks: int = 8
    unimix_ratio: float = 0.01
    motor_inertia_alpha: float = 1.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return x * rms * self.weight


class BlockLinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        scale = (2.0 / max(1, self.in_ch // self.blocks)) ** 0.5
        self.weight = mx.random.normal((self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks)) * scale
        self.bias = mx.zeros((self.out_ch,))

    def __call__(self, x: mx.array) -> mx.array:
        batch_shape = x.shape[:-1]
        x = mx.reshape(x, (*batch_shape, self.blocks, self.in_ch // self.blocks))
        x = mx.einsum("...gi,oig->...go", x, self.weight)
        x = mx.reshape(x, (*batch_shape, self.out_ch))
        return x + self.bias


class DeterNet(nn.Module):
    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.blocks = cfg.blocks
        self.flat_stoch = cfg.stoch * cfg.discrete
        self.in0 = nn.Sequential(nn.Linear(cfg.deter, cfg.hidden), RMSNorm(cfg.hidden), nn.SiLU())
        self.in1 = nn.Sequential(nn.Linear(self.flat_stoch, cfg.hidden), RMSNorm(cfg.hidden), nn.SiLU())
        self.in2 = nn.Sequential(nn.Linear(cfg.act_dim, cfg.hidden), RMSNorm(cfg.hidden), nn.SiLU())
        self.in3 = nn.Sequential(nn.Linear(cfg.d_emb_dim, cfg.hidden), RMSNorm(cfg.hidden), nn.SiLU())

        in_ch = (4 * cfg.hidden + cfg.deter // cfg.blocks) * cfg.blocks
        layers: list[nn.Module] = []
        for _ in range(cfg.dyn_layers):
            layers.append(BlockLinear(in_ch, cfg.deter, cfg.blocks))
            layers.append(RMSNorm(cfg.deter))
            layers.append(nn.SiLU())
            in_ch = cfg.deter
        self.hid = nn.Sequential(*layers)
        self.gru = BlockLinear(in_ch, 3 * cfg.deter, cfg.blocks)
        self.deter = cfg.deter

    def __call__(self, stoch: mx.array, deter: mx.array, action: mx.array, d_emb: mx.array) -> mx.array:
        b = action.shape[0]
        stoch = mx.reshape(stoch, (b, -1))
        action = action / mx.maximum(mx.abs(action), 1.0)

        x = mx.concatenate([self.in0(deter), self.in1(stoch), self.in2(action), self.in3(d_emb)], axis=-1)
        x = mx.expand_dims(x, axis=-2)
        x = mx.broadcast_to(x, (b, self.blocks, x.shape[-1]))

        deter_g = mx.reshape(deter, (b, self.blocks, -1))
        x = mx.concatenate([deter_g, x], axis=-1)
        x = mx.reshape(x, (b, -1))
        x = self.hid(x)
        x = self.gru(x)

        gates = mx.reshape(x, (b, self.blocks, -1))
        reset, cand, update = mx.split(gates, 3, axis=-1)
        reset = mx.sigmoid(mx.reshape(reset, (b, -1)))
        cand = mx.tanh(reset * mx.reshape(cand, (b, -1)))
        update = mx.sigmoid(mx.reshape(update, (b, -1)) - 1.0)
        return update * cand + (1.0 - update) * deter


class MLPStack(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, out_dim: int):
        super().__init__()
        net: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            net.extend([nn.Linear(d, hidden), RMSNorm(hidden), nn.SiLU()])
            d = hidden
        net.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*net)

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class MLXRSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.flat_stoch = cfg.stoch * cfg.discrete
        self.feat_size = self.flat_stoch + cfg.deter
        self.deter_net = DeterNet(cfg)
        self.obs_net = MLPStack(cfg.deter + cfg.embed_size, cfg.hidden, cfg.obs_layers, self.flat_stoch)
        self.img_net = MLPStack(cfg.deter, cfg.hidden, cfg.img_layers, self.flat_stoch)

    def _default_d_emb(self, tensor: mx.array) -> mx.array:
        return mx.zeros((*tensor.shape[:-1], self.cfg.d_emb_dim), dtype=tensor.dtype)

    def _filter_action(self, action: mx.array, prev_filtered_action: mx.array, alpha: float | mx.array | None) -> mx.array:
        if alpha is None:
            alpha = self.cfg.motor_inertia_alpha
        if not isinstance(alpha, mx.array):
            alpha = mx.full((*action.shape[:-1], 1), float(alpha), dtype=action.dtype)
        while len(alpha.shape) < len(action.shape):
            alpha = mx.expand_dims(alpha, axis=-1)
        return alpha * action + (1.0 - alpha) * prev_filtered_action

    def get_dist_logits(self, logit: mx.array) -> mx.array:
        probs = mx.softmax(logit.astype(mx.float32), axis=-1)
        if self.cfg.unimix_ratio > 0:
            k = probs.shape[-1]
            probs = probs * (1.0 - self.cfg.unimix_ratio) + self.cfg.unimix_ratio / k
        return mx.log(mx.maximum(probs, 1e-8))

    def gumbel_onehot(self, logits: mx.array, temperature: float = 1.0) -> mx.array:
        g = mx.random.gumbel(logits.shape)
        y = mx.softmax((logits + g) / temperature, axis=-1)
        idx = mx.argmax(y, axis=-1)
        hard = mx.eye(logits.shape[-1])[idx]
        return hard + y - mx.stop_gradient(y)

    def initial(self, batch_size: int) -> RSSMState:
        return RSSMState(
            stoch=mx.zeros((batch_size, self.cfg.stoch, self.cfg.discrete), dtype=mx.float32),
            deter=mx.zeros((batch_size, self.cfg.deter), dtype=mx.float32),
            prev_action=mx.zeros((batch_size, self.cfg.act_dim), dtype=mx.float32),
            prev_filtered_action=mx.zeros((batch_size, self.cfg.act_dim), dtype=mx.float32),
        )

    def prior(self, deter: mx.array) -> tuple[mx.array, mx.array]:
        logit = mx.reshape(self.img_net(deter), (*deter.shape[:-1], self.cfg.stoch, self.cfg.discrete))
        logits = self.get_dist_logits(logit)
        stoch = self.gumbel_onehot(logits)
        return stoch, logits

    def obs_step(
        self,
        state: RSSMState,
        embed: mx.array,
        reset: mx.array,
        d_emb: mx.array | None = None,
        alpha: float | mx.array | None = None,
    ) -> tuple[RSSMState, mx.array]:
        if d_emb is None:
            d_emb = self._default_d_emb(state.prev_action)
        reset_f = reset.astype(mx.float32).reshape(-1, 1)
        stoch = state.stoch * (1.0 - reset_f[..., None])
        deter = state.deter * (1.0 - reset_f)
        prev_action = state.prev_action * (1.0 - reset_f)
        prev_filtered = state.prev_filtered_action * (1.0 - reset_f)

        filtered_action = self._filter_action(prev_action, prev_filtered, alpha)
        deter = self.deter_net(stoch, deter, filtered_action, d_emb)
        x = mx.concatenate([deter, embed], axis=-1)
        logit = mx.reshape(self.obs_net(x), (*x.shape[:-1], self.cfg.stoch, self.cfg.discrete))
        logits = self.get_dist_logits(logit)
        stoch = self.gumbel_onehot(logits)
        new_state = RSSMState(stoch=stoch, deter=deter, prev_action=filtered_action, prev_filtered_action=filtered_action)
        return new_state, logits

    def get_feat(self, stoch: mx.array, deter: mx.array) -> mx.array:
        flat_stoch = mx.reshape(stoch, (*stoch.shape[:-2], self.flat_stoch))
        return mx.concatenate([flat_stoch, deter], axis=-1)


class MLXConvEncoder(nn.Module):
    """NHWC encoder for MLX runtime."""

    def __init__(self, in_ch: int, depth: int = 48, mults: tuple[int, ...] = (2, 3, 4, 4)):
        super().__init__()
        channels = [depth * m for m in mults]
        self.convs = []
        c_in = in_ch
        for c_out in channels:
            self.convs.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1))
            c_in = c_out
        self.out_dim = channels[-1]

    def __call__(self, obs_nhwc: mx.array) -> mx.array:
        x = obs_nhwc
        for conv in self.convs:
            x = nn.silu(conv(x))
        return mx.mean(x, axis=(1, 2))


class MLXActor(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, hidden: int = 768, layers: int = 3):
        super().__init__()
        self.backbone = MLPStack(in_dim, hidden, layers, action_dim * 2)
        self.action_dim = action_dim

    def __call__(self, x: mx.array, sample: bool = False) -> mx.array:
        mean, std_preact = mx.split(self.backbone(x), 2, axis=-1)
        std = (1.0 - 0.1) * mx.sigmoid(std_preact + 2.0) + 0.1
        if sample:
            eps = mx.random.normal(mean.shape)
            a = mean + std * eps
        else:
            a = mean
        return a / mx.maximum(mx.abs(a), 1.0)


class MLXValue(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 768, layers: int = 3):
        super().__init__()
        self.head = MLPStack(in_dim, hidden, layers, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(x)


def functional_apply(module: nn.Module, fn: Callable[[mx.array], mx.array], x: mx.array) -> mx.array:
    _ = module
    return fn(x)
