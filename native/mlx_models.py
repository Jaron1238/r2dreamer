from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from native.mlx_types import RSSMState


def _cfg_get(obj, name: str, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default

def weight_init_(module: nn.Module) -> None:
    """Compatibility hook for modules that are initialized from PyTorch weights."""
    _ = module


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
        # Match PyTorch's RMSNorm behavior exactly
        # PyTorch: x * weight * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        # We need to compute over the last dimension
        squared = mx.square(x)
        mean_squared = mx.mean(squared, axis=-1, keepdims=True)
        inv_std = mx.rsqrt(mean_squared + self.eps)
        return x * self.weight * inv_std

class BlockLinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)
        scale = (2.0 / max(1, self.in_ch // self.blocks)) ** 0.5 * self.outscale
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
            layers.append(BlockLinear(in_ch, cfg.deter, cfg.blocks, outscale=1.0))
            layers.append(RMSNorm(cfg.deter))
            layers.append(nn.SiLU())
            in_ch = cfg.deter
        self.hid = nn.Sequential(*layers)
        self.gru = BlockLinear(in_ch, 3 * cfg.deter, cfg.blocks, outscale=1.0)
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
        new_state = RSSMState(stoch=stoch, deter=deter, prev_action=prev_action, prev_filtered_action=filtered_action)
        return new_state, logits

    def get_feat(self, stoch: mx.array, deter: mx.array) -> mx.array:
        flat_stoch = mx.reshape(stoch, (*stoch.shape[:-2], self.flat_stoch))
        return mx.concatenate([flat_stoch, deter], axis=-1)

    def observe(
        self,
        embed:         mx.array,
        action:        mx.array,
        initial_state: RSSMState,
        is_first:      mx.array,
        d_emb:         mx.array | None = None,
        alpha:         float | None     = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        
        B, T = embed.shape[:2]
        if d_emb is None:
            d_emb = mx.zeros((B, T, self.cfg.d_emb_dim), dtype=embed.dtype)

        state = initial_state
        stochs, deters, logits = [], [], []
        for t in range(T):
            reset_t = is_first[:, t]
            if reset_t.ndim == 2:
                reset_t = reset_t.squeeze(-1)
            state, logit = self.obs_step(state, embed[:, t], reset_t, d_emb[:, t], alpha)
            stochs.append(state.stoch)
            deters.append(state.deter)
            logits.append(logit)

        return (
            mx.stack(stochs, axis=1),
            mx.stack(deters, axis=1),
            mx.stack(logits, axis=1),
        )

    def img_step(
        self,
        state:  RSSMState,
        action: mx.array,
        d_emb:  mx.array | None = None,
        alpha:  float | None     = None,
    ) -> RSSMState:
        
        if d_emb is None:
            d_emb = self._default_d_emb(action)
        filtered = self._filter_action(action, state.prev_filtered_action, alpha)
        deter    = self.deter_net(state.stoch, state.deter, filtered, d_emb)
        stoch, _ = self.prior(deter)
        return RSSMState(stoch=stoch, deter=deter,
                         prev_action=action, prev_filtered_action=filtered)

    def kl_loss(
        self,
        post_logit:  mx.array,
        prior_logit: mx.array,
        free:        float,
    ) -> tuple[mx.array, mx.array]:
        

        from native.mlx_distributions import kl
        rep = kl(post_logit, mx.stop_gradient(prior_logit)).sum(axis=-1)
        dyn = kl(mx.stop_gradient(post_logit), prior_logit).sum(axis=-1)
        return mx.maximum(dyn, free), mx.maximum(rep, free)

_B0_STAGES: list[tuple[int, int, int, int, int]] = [
    (1, 16,  1, 1, 3),  
    (6, 24,  2, 2, 3),  
    (6, 40,  2, 2, 5),  
    (6, 80,  3, 2, 3),  
    (6, 112, 3, 1, 5),  
    (6, 192, 4, 2, 5),  
    (6, 320, 1, 1, 3),  
]

def _eff_b0_out_spatial(h: int, w: int) -> tuple[int, int]:
    

    for s in [2, 1, 2, 2, 2, 1, 2, 1]:  
        h = math.ceil(h / s)
        w = math.ceil(w / s)
    return h, w

class _MLXSEBlock(nn.Module):
    

    def __init__(self, in_ch: int, reduced_ch: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_ch, reduced_ch, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_ch, in_ch, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        
        s = mx.mean(x, axis=(1, 2), keepdims=True)   
        s = nn.silu(self.fc1(s))
        s = mx.sigmoid(self.fc2(s))
        return x * s

class _MLXMBConvBlock(nn.Module):
    

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.use_skip = (stride == 1 and in_ch == out_ch)
        expanded_ch = in_ch * expand_ratio
        pad = (kernel_size - 1) // 2
        
        
        reduced_ch = max(1, int(in_ch * se_ratio))

        if expand_ratio == 1:
            self.block: list = [
                
                [
                    nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=pad,
                              groups=in_ch, bias=False),
                    nn.BatchNorm(in_ch, eps=1e-3, momentum=0.01),
                ],
                
                _MLXSEBlock(in_ch, reduced_ch),
                
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm(out_ch, eps=1e-3, momentum=0.01),
                ],
            ]
        else:
            self.block = [
                
                [
                    nn.Conv2d(in_ch, expanded_ch, kernel_size=1, bias=False),
                    nn.BatchNorm(expanded_ch, eps=1e-3, momentum=0.01),
                ],
                
                [
                    nn.Conv2d(expanded_ch, expanded_ch, kernel_size, stride=stride, padding=pad,
                              groups=expanded_ch, bias=False),
                    nn.BatchNorm(expanded_ch, eps=1e-3, momentum=0.01),
                ],
                
                _MLXSEBlock(expanded_ch, reduced_ch),
                
                [
                    nn.Conv2d(expanded_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm(out_ch, eps=1e-3, momentum=0.01),
                ],
            ]

    def __call__(self, x: mx.array) -> mx.array:
        inp = x
        last_idx = len(self.block) - 1
        for i, sub in enumerate(self.block):
            if isinstance(sub, _MLXSEBlock):
                x = sub(x)
            else:
                
                x = sub[1](sub[0](x))
                
                if i != last_idx:
                    x = nn.silu(x)
        if self.use_skip:
            x = x + inp
        return x

class MLXConvEncoder(nn.Module):
    

    def __init__(
        self,
        in_ch: int,
        input_h: int,
        input_w: int,
        depth: int = 48,
        mults: tuple[int, ...] = (2, 3, 4, 4),
    ) -> None:
        super().__init__()
        self.input_ch = int(in_ch)

        
        stages: list = []

        
        
        stages.append([
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm(32, eps=1e-3, momentum=0.01),
        ])

        
        in_c = 32
        for exp, out_c, n_blocks, stride, k in _B0_STAGES:
            stage_blocks: list[_MLXMBConvBlock] = []
            for i in range(n_blocks):
                s = stride if i == 0 else 1
                stage_blocks.append(_MLXMBConvBlock(in_c, out_c, exp, k, s))
                in_c = out_c
            stages.append(stage_blocks)

        
        
        stages.append([
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm(1280, eps=1e-3, momentum=0.01),
        ])

        self.backbone: list = stages  

    def __call__(self, x: mx.array) -> mx.array:
        # Stem: Conv2d → BN → SiLU
        stem_conv, stem_bn = self.backbone[0]
        x = nn.silu(stem_bn(stem_conv(x)))

        # MBConv stages
        for stage in self.backbone[1:-1]:
            for block in stage:
                x = block(x)

        # Head: Conv2d → BN → SiLU
        head_conv, head_bn = self.backbone[-1]
        x = nn.silu(head_bn(head_conv(x)))

        # Global average pooling: (B, H, W, C) → (B, C)
        x = x.mean(axis=(1, 2))
        return x

from native.mlx_types import RSSMState

@dataclass
class MLXActor(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, hidden: int = 768, layers: int = 3):
        super().__init__()
        self.backbone = MLPStack(in_dim, hidden, layers, action_dim * 2)
        self.action_dim = action_dim

    def __call__(self, x: mx.array, sample: bool = False) -> mx.array:
        mean, std_preact = mx.split(self.backbone(x), 2, axis=-1)
        std = (1.0 - 0.1) * mx.sigmoid(std_preact + 2.0) + 0.1
        
        
        
        
        
        tanh_mean = mx.tanh(mean)
        if sample:
            eps = mx.random.normal(mean.shape)
            return tanh_mean + std * eps
        return tanh_mean

class MLXValue(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 768, layers: int = 3, bins: int = 255):
        super().__init__()
        
        
        
        
        self.head = MLPStack(in_dim, hidden, layers, bins)

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(x)

def functional_apply(module: nn.Module, fn: Callable[[mx.array], mx.array], x: mx.array) -> mx.array:
    _ = module
    return fn(x)

class _MLXResBlock(nn.Module):
    

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(self.conv(x) + self.skip(x))

class MLXSafetyNet(nn.Module):
    

    def __init__(
        self,
        in_channels: int = 1,
        action_dim: int = 4,
        speed_dim: int = 1,
        hidden: int = 64,
        frame_stack: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden     = hidden
        stacked_channels = in_channels * frame_stack

        
        self.layer1 = _MLXResBlock(stacked_channels, 16,     stride=1)  
        self.layer2 = _MLXResBlock(16,               16,     stride=2)  
        self.layer3 = _MLXResBlock(16,               32,     stride=2)  
        self.layer4 = _MLXResBlock(32,               64,     stride=2)  
        self.layer5 = _MLXResBlock(64,               hidden, stride=2)  

        feat_in = hidden * 25 + action_dim + speed_dim

        self.prob_head = nn.Sequential(
            nn.Linear(feat_in, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        self.safe_action_head = nn.Sequential(
            nn.Linear(feat_in, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def __call__(self, image_nhwc: mx.array, speed: mx.array, action: mx.array) -> tuple[mx.array, mx.array]:
        b, t = image_nhwc.shape[:2]
        
        x = mx.reshape(image_nhwc, (b * t, *image_nhwc.shape[2:]))

        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        
        
        
        
        x = x[:, 1:-1, 1:-1, :]                             
        N, H, W, C = x.shape                                
        x = mx.reshape(x, (N, 5, H // 5, 5, W // 5, C))    
        x = mx.max(x, axis=(2, 4))                          
        feat = mx.reshape(x, (N, 5 * 5 * C))                

        speed_flat  = mx.reshape(speed,  (b * t, -1))
        action_flat = mx.reshape(action, (b * t, -1))
        combined    = mx.concatenate([feat, speed_flat, action_flat], axis=-1)

        prob_logit  = mx.reshape(self.prob_head(combined),        (b, t, 1))
        safe_action = mx.reshape(self.safe_action_head(combined), (b, t, self.action_dim))

        return prob_logit, safe_action

class MLXTransformerEncoderLayer(nn.Module):
    

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int | None = None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        
        self.attention = nn.MultiHeadAttention(d_model, nhead, bias=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        
        nx = self.ln1(x)
        attn_out = self.attention(nx, nx, nx, mask=mask)
        x = x + attn_out

        nx = self.ln2(x)
        ff_out = self.linear2(nn.gelu(self.linear1(nx)))
        x = x + ff_out

        return x


class MLXStackedGRU(nn.Module):
    """Small MLX-native stack that mirrors PyTorch's multi-layer GRU shape contract."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.layers = [
            nn.GRU(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> tuple[mx.array, list[mx.array]]:
        states: list[mx.array] = []
        for layer in self.layers:
            x = layer(x)
            states.append(x[:, -1])
        return x, states


class MLXTransformerEncoder(nn.Module):
    """MLX-native Transformer encoder with PyTorch-compatible pre-norm math."""

    def __init__(self, d_model: int, nhead: int, num_layers: int = 2):
        super().__init__()
        self.layers = [
            MLXTransformerEncoderLayer(d_model, nhead, d_model * 4)
            for _ in range(num_layers)
        ]

    @staticmethod
    def _additive_attention_mask(padding_mask: mx.array, dtype) -> mx.array:
        return mx.where(
            padding_mask[:, None, None, :],
            mx.array(-mx.inf, dtype=dtype),
            mx.array(0.0, dtype=dtype),
        )

    def __call__(self, x: mx.array, padding_mask: mx.array | None = None) -> mx.array:
        mask = None
        if padding_mask is not None:
            mask = self._additive_attention_mask(padding_mask, x.dtype)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class MLXContextEncoder(nn.Module):
    
    def __init__(
        self,
        flat_stoch: int,
        deter: int,
        act_dim: int,
        ctx_len: int = 16,
        bottleneck: int = 256,
        out_dim: int = 16,
        encoder_type: str = "gru",
    ):
        super().__init__()
        self.ctx_len      = int(ctx_len)
        self.bottleneck   = int(bottleneck)
        self.out_dim      = int(out_dim)
        self.encoder_type = str(encoder_type)

        inp_dim = int(flat_stoch + deter + act_dim)

        
        self.proj = nn.Sequential(
            nn.Linear(inp_dim, self.bottleneck, bias=True),
            RMSNorm(self.bottleneck, eps=1e-4),
            nn.SiLU(),
            nn.Linear(self.bottleneck, self.bottleneck, bias=True),
            RMSNorm(self.bottleneck, eps=1e-4),
            nn.SiLU(),
        )

        
        if self.encoder_type == "gru":
            self.rnn = MLXStackedGRU(
                input_size=self.bottleneck,
                hidden_size=self.bottleneck,
                num_layers=2,
            )
        elif self.encoder_type == "transformer":
            
            half_dim = self.bottleneck // 2
            rope_freqs = 1.0 / (10_000 ** (mx.arange(0, half_dim, dtype=mx.float32) / half_dim))
            self.rope_freqs = rope_freqs

            num_heads = max(1, self.bottleneck // 64)  
            self.transformer = MLXTransformerEncoder(
                d_model=self.bottleneck,
                nhead=num_heads,
                num_layers=2,
            )
        else:
            raise ValueError(f"ContextEncoder: unbekannter encoder_type='{encoder_type}'. "
                             f"Erlaubt: 'gru', 'transformer'.")

        
        self.temporal_attn = nn.Linear(self.bottleneck, 1, bias=True)

        
        self.out_head = nn.Linear(self.bottleneck, self.out_dim, bias=True)

        self.apply(weight_init_)

    
    def _apply_rope(self, x: mx.array) -> mx.array:
        

        B, T, D = x.shape
        half = D // 2

        
        positions = mx.arange(T, dtype=x.dtype)
        angles = mx.outer(positions, self.rope_freqs.astype(x.dtype))  
        cos = mx.expand_dims(mx.cos(angles), 0)   
        sin = mx.expand_dims(mx.sin(angles), 0)   

        x1 = x[..., :half]               
        x2 = x[..., half : half * 2]     
        x_rot = mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

        
        if D % 2 != 0:
            x_rot = mx.concatenate([x_rot, x[..., -1:]], axis=-1)
        return x_rot

    def __call__(
        self,
        flat_stoch: mx.array,
        deter: mx.array,
        action: mx.array,
        valid_len: mx.array | None = None,
    ) -> mx.array:
        

        B, T, _ = flat_stoch.shape

        
        
        
        padding_mask: mx.array | None = None
        if valid_len is not None:
            
            positions  = mx.expand_dims(mx.arange(T), axis=0)
            valid_start = mx.expand_dims(T - mx.clip(valid_len, None, T), axis=1)
            padding_mask = positions < valid_start                                 

        
        x = mx.concatenate([flat_stoch, deter, action], axis=-1)   
        x = self.proj(x)                                      

        
        if self.encoder_type == "gru":
            # Bug #15 fix: zero out leading padded positions before the GRU so that
            # padding-contaminated states don't bleed into valid hidden states.
            # The temporal-attention pool below already masks via padding_mask, but
            # zeroing the inputs makes the GRU path consistent with the transformer path.
            if padding_mask is not None:
                x = mx.where(mx.expand_dims(padding_mask, axis=-1), mx.zeros_like(x), x)
            x, _ = self.rnn(x)                               
        else:  
            x = self._apply_rope(x)                          
            
            
            x = self.transformer(x, padding_mask=padding_mask)

        
        
        attn_logits = self.temporal_attn(x)                  
        if padding_mask is not None:
            attn_logits = mx.where(
                mx.expand_dims(padding_mask, axis=-1),
                mx.array(-mx.inf, dtype=attn_logits.dtype),
                attn_logits,
            )
        attn_w = mx.softmax(attn_logits, axis=1)
        ctx    = mx.sum(attn_w * x, axis=1)

        
        return self.out_head(ctx)                             

class MLXCausalTemporalTransformer(nn.Module):
    """Causal transformer used by NeDreamer with PyTorch-compatible parameter names."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        action_proj_dim: int = 32,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, action_proj_dim, bias=True)
        self.in_proj = nn.Linear(state_dim + action_proj_dim, model_dim, bias=True)
        self.pos_embed = mx.zeros((1, max_len, model_dim))
        self.encoder = MLXTransformerEncoder(model_dim, num_heads, num_layers=num_layers)

    @staticmethod
    def _causal_attention_mask(seq_len: int, dtype) -> mx.array:
        idx = mx.arange(seq_len)
        mask = mx.expand_dims(idx, axis=1) < mx.expand_dims(idx, axis=0)
        return mx.where(mask, mx.array(-mx.inf, dtype=dtype), mx.array(0.0, dtype=dtype))

    def __call__(self, state_seq: mx.array, prev_action_seq: mx.array) -> mx.array:
        action_emb = self.action_proj(prev_action_seq)
        x = mx.concatenate([state_seq, action_emb], axis=-1)
        x = self.in_proj(x)
        x = x + self.pos_embed[:, : x.shape[1], :]
        mask = self._causal_attention_mask(x.shape[1], x.dtype)
        for layer in self.encoder.layers:
            x = layer(x, mask=mask)
        return x


class MLXNEPredictorHead(nn.Module):
    """MLX port of networks.NEPredictorHead."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class MLXProjector(nn.Module):
    

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        scale       = (2.0 / in_dim) ** 0.5
        self.weight = mx.random.normal((out_dim, in_dim)) * scale

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T

class MLXInvDynHead(nn.Module):
    

    def __init__(self, feat_dim: int, d_emb_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = feat_dim * 2 + d_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.apply(weight_init_)

    def __call__(
        self,
        feat_t:  mx.array,   
        feat_t1: mx.array,   
        d_emb:   mx.array,   
    ) -> mx.array:
        x = mx.concatenate([feat_t, feat_t1, d_emb], axis=-1)
        return self.net(x)

class MLXMLPHead(nn.Module):
    

    def __init__(self, in_dim: int, hidden: int, layers: int, out_dim: int, dist_fn):
        super().__init__()
        self.mlp     = MLPStack(in_dim, hidden, layers, out_dim)
        self.dist_fn = dist_fn

    def __call__(self, x: mx.array):
        return self.dist_fn(self.mlp(x))

class MLXReturnEMA:
    

    def __init__(self, alpha: float = 0.01):
        self.alpha    = alpha
        self.ema_low  = mx.zeros(())
        self.ema_high = mx.zeros(())

    def __call__(self, x: mx.array):
        flat     = mx.reshape(x, (-1,))
        sorted_x = mx.sort(flat)
        n        = sorted_x.shape[0]
        low_idx  = max(0, int(0.05 * n))
        high_idx = min(n - 1, int(0.95 * n))
        low_val  = sorted_x[low_idx]
        high_val = sorted_x[high_idx]

        self.ema_low  = mx.stop_gradient(
            self.alpha * low_val  + (1 - self.alpha) * self.ema_low
        )
        self.ema_high = mx.stop_gradient(
            self.alpha * high_val + (1 - self.alpha) * self.ema_high
        )
        scale  = mx.maximum(self.ema_high - self.ema_low, mx.array(1.0))
        return self.ema_low, scale

class MLXDreamer(nn.Module):
    

    def __init__(self, cfg):
        super().__init__()
        from functools import partial
        from native.mlx_distributions import binary, symexp_twohot

        self.config = cfg
        root_cfg = cfg
        cfg = _cfg_get(root_cfg, "model", root_cfg)

        self.phase               = int(_cfg_get(root_cfg, "phase", _cfg_get(cfg, "phase", 1)))
        self.kl_free             = float(cfg.kl_free)
        self.act_entropy         = float(cfg.act_entropy)
        self.imag_horizon        = int(cfg.imag_horizon)
        self.horizon             = int(cfg.horizon)
        self.lamb                = float(cfg.lamb)
        self.barlow_lambd        = float(getattr(cfg, "barlow_lambd", 5e-3))
        self.inv_dyn_loss_weight = float(getattr(cfg, "inv_dyn_loss_weight", 1.0))
        self.ctx_len             = int(getattr(cfg, "ctx_len", 16))
        self.ctx_warmup_steps    = int(getattr(cfg, "ctx_warmup_steps", 1000))
        self.ctx_consistency_w   = float(getattr(cfg, "ctx_consistency_weight", 1.0))
        self._ctx_updates        = 0
        self._loss_scales        = dict(cfg.loss_scales)
        self.num_drones          = int(getattr(cfg, "num_drones", 1))
        self.d_emb_dim           = int(getattr(cfg, "drone_embed_dim", 16))
        self.embed_size          = int(cfg.embed_size)
        self.rep_loss            = str(_cfg_get(cfg, "rep_loss", "r2dreamer"))
        self.safety_threshold    = float(_cfg_get(cfg, "safety_threshold", 0.4))
        self.safety_input_key    = str(_cfg_get(cfg, "safety_input_key", "image"))

        rssm_cfg = RSSMConfig(**cfg.rssm)
        self.rssm = MLXRSSM(rssm_cfg)

        self.obs_shape = (int(cfg.encoder.input_h), int(cfg.encoder.input_w), int(cfg.encoder.in_ch))
        self.encoder = MLXConvEncoder(
            in_ch    = int(cfg.encoder.in_ch),
            input_h  = int(cfg.encoder.input_h),
            input_w  = int(cfg.encoder.input_w),
            depth    = int(cfg.encoder.depth),
            mults    = tuple(cfg.encoder.mults),
        )

        self.drone_embed = nn.Embedding(self.num_drones, self.d_emb_dim)

        self.context_encoder = MLXContextEncoder(
            flat_stoch   = rssm_cfg.stoch * rssm_cfg.discrete,
            deter        = rssm_cfg.deter,
            act_dim      = rssm_cfg.act_dim,
            ctx_len      = self.ctx_len,
            bottleneck   = int(getattr(cfg, "ctx_bottleneck", 256)),
            out_dim      = self.d_emb_dim,
            encoder_type = str(getattr(cfg, "ctx_encoder_type", "gru")),
        )

        feat_size = self.rssm.feat_size
        self.inv_dyn_head = MLXInvDynHead(feat_size, self.d_emb_dim, rssm_cfg.act_dim)

        self.projector        = MLXProjector(feat_size, self.embed_size)
        self.projector_target = MLXProjector(self.embed_size, self.embed_size)

        if self.rep_loss == "nedreamer":
            ne_cfg = _cfg_get(cfg, "nedreamer")
            self.barlow_lambd = float(_cfg_get(ne_cfg, "lambd", self.barlow_lambd))
            self.use_ema_target = bool(_cfg_get(ne_cfg, "use_ema_target", False))
            ne_hidden = int(_cfg_get(ne_cfg, "hidden_dim", 256))
            self.nedreamer_transformer = MLXCausalTemporalTransformer(
                state_dim=feat_size,
                action_dim=rssm_cfg.act_dim,
                model_dim=ne_hidden,
                num_layers=int(_cfg_get(ne_cfg, "transformer_layers", 2)),
                num_heads=int(_cfg_get(ne_cfg, "transformer_heads", 4)),
                max_len=int(_cfg_get(root_cfg, "batch_length", _cfg_get(_cfg_get(root_cfg, "trainer"), "batch_length", self.horizon))),
                action_proj_dim=32,
            )
            self.nedreamer_predictor = MLXNEPredictorHead(ne_hidden, self.embed_size)
            self.target_encoder = copy.deepcopy(self.encoder) if self.use_ema_target else None

        rew_bins = int(getattr(cfg.reward.dist, "bin_num", 255))
        self.reward_head = MLXMLPHead(
            in_dim  = feat_size,
            hidden  = int(cfg.reward.units),
            layers  = int(cfg.reward.layers),
            out_dim = rew_bins,
            dist_fn = partial(symexp_twohot, bin_num=rew_bins),
        )
        self.cont_head = MLXMLPHead(
            in_dim  = feat_size,
            hidden  = int(cfg.cont.units),
            layers  = int(cfg.cont.layers),
            out_dim = 1,
            dist_fn = binary,
        )

        actor_in = feat_size + self.d_emb_dim
        self.actor = MLXActor(
            in_dim     = actor_in,
            action_dim = rssm_cfg.act_dim,
            hidden     = int(getattr(cfg.actor, "units", 768)),
            layers     = int(getattr(cfg.actor, "layers", 3)),
        )
        self.value = MLXValue(
            in_dim  = actor_in,
            hidden  = int(getattr(cfg.value, "units", 768)),
            layers  = int(getattr(cfg.value, "layers", 3)),
            bins    = int(getattr(cfg.value.dist, "bin_num", 255)),
        )

        self.safety_net = MLXSafetyNet(
            in_channels = int(getattr(cfg.safety_net, "in_channels", 2)),
            action_dim  = rssm_cfg.act_dim,
            speed_dim   = 1,
            hidden      = int(getattr(cfg.safety_net, "hidden", 64)),
            frame_stack = int(getattr(cfg.safety_net, "frame_stack", 3)),
        )

        self.return_ema = MLXReturnEMA()

    @staticmethod
    def _barlow_loss(z_a: mx.array, z_b: mx.array, lambd: float) -> tuple[mx.array, dict]:
        weights = mx.ones((z_a.shape[0],), dtype=z_a.dtype)
        return MLXDreamer._barlow_loss_masked(z_a, z_b, weights, lambd)

    @staticmethod
    def _barlow_loss_masked(z_a: mx.array, z_b: mx.array, weights: mx.array, lambd: float) -> tuple[mx.array, dict]:
        eps = 1e-8
        weights = weights.astype(z_a.dtype)
        weights = weights / mx.maximum(mx.sum(weights), mx.array(1.0, dtype=z_a.dtype))
        w = mx.expand_dims(weights, axis=-1)
        mean_a = mx.sum(z_a * w, axis=0)
        mean_b = mx.sum(z_b * w, axis=0)
        var_a = mx.sum(((z_a - mean_a) ** 2) * w, axis=0)
        var_b = mx.sum(((z_b - mean_b) ** 2) * w, axis=0)
        z_a_norm = (z_a - mean_a) / mx.sqrt(var_a + eps)
        z_b_norm = (z_b - mean_b) / mx.sqrt(var_b + eps)
        c = mx.matmul((z_a_norm * w).T, z_b_norm)
        diag = mx.diag(c)
        inv_loss = mx.sum((diag - 1.0) ** 2)
        off_diag = c - mx.diag(diag)
        red_loss = mx.sum(off_diag ** 2)
        loss = inv_loss + lambd * red_loss
        metrics = {
            "barlow/invariance": float(inv_loss.item()),
            "barlow/redundancy": float(red_loss.item()),
            "barlow/std_mean": float(mx.sqrt(var_a + eps).mean().item()),
        }
        return loss, metrics

    @staticmethod
    def _lambda_return(
        last:   mx.array,
        term:   mx.array,
        reward: mx.array,
        value:  mx.array,
        boot:   mx.array,
        disc:   float,
        lamb:   float,
    ) -> mx.array:
        
        T       = reward.shape[1]
        ret     = boot
        returns = []
        for t in reversed(range(T)):
            rew_t  = reward[:, t]
            val_t  = value[:, t]
            cont_t = 1.0 - term[:, t]
            ret    = rew_t + disc * cont_t * ((1.0 - lamb) * val_t + lamb * ret)
            returns.append(ret)
        returns.reverse()
        return mx.stack(returns, axis=1)

    def _stack_frames(self, image: mx.array, n: int) -> mx.array:
        
        B, T, H, W, C = image.shape
        frames = [mx.zeros_like(image[:, :1])] * max(0, n - 1) + [image]
        stacked = mx.concatenate(frames[-n:], axis=-1)
        return stacked


    def _update_nedreamer_ema_target(self) -> None:
        if self.rep_loss != "nedreamer" or getattr(self, "target_encoder", None) is None:
            return
        from mlx.utils import tree_flatten

        model_cfg = _cfg_get(self.config, "model", self.config)
        ne_cfg = _cfg_get(model_cfg, "nedreamer")
        ema_rate = float(_cfg_get(ne_cfg, "ema_rate", 0.99))
        online = dict(tree_flatten(self.encoder.parameters()))
        target = dict(tree_flatten(self.target_encoder.parameters()))
        updated = [
            (name, ema_rate * target[name] + (1.0 - ema_rate) * online[name])
            for name in target
            if name in online
        ]
        if updated:
            self.target_encoder.load_weights(updated, strict=False)

    def compute_losses(
        self,
        data:    dict,
        initial: RSSMState | None = None,
    ) -> tuple[mx.array, dict, dict]:
        
        from native.mlx_distributions import focal_bce, masked_mean, symexp_twohot

        losses:  dict = {}
        metrics: dict = {}

        image  = data["image"]         
        action = data["action"]        
        B, T   = action.shape[:2]

        if initial is None:
            initial = self.rssm.initial(B)

        
        drone_id = data.get("drone_id", mx.zeros((B,), dtype=mx.int32))
        if drone_id.ndim == 1:
            drone_id = mx.expand_dims(drone_id, 1)            
        d_emb_static = self.drone_embed(drone_id.reshape(-1)).reshape(B, 1, self.d_emb_dim)
        d_emb_static = mx.broadcast_to(d_emb_static, (B, T, self.d_emb_dim))

        
        embed = self.encoder(image)    

        
        if self.phase == 2:
            post_stoch, post_deter, post_logit = self.rssm.observe(
                embed, action, initial, data["is_first"], d_emb_static
            )
            d_emb = d_emb_static
        else:
            
            post_stoch_w, post_deter_w, _ = self.rssm.observe(
                mx.stop_gradient(embed), action, initial,
                data["is_first"], d_emb_static
            )
            ctx_end      = min(T, self.ctx_len)
            flat_warm    = post_stoch_w.reshape(B, T, -1)
            d_emb_ctx    = self.context_encoder(
                flat_warm[:, :ctx_end],
                post_deter_w[:, :ctx_end],
                action[:, :ctx_end],
            )                                                  
            d_emb_exp    = mx.expand_dims(d_emb_ctx, 1)
            d_emb        = mx.broadcast_to(d_emb_exp, (B, T, self.d_emb_dim))

            
            post_stoch, post_deter, post_logit = self.rssm.observe(
                embed, action, initial, data["is_first"], d_emb
            )

        
        _, prior_logit       = self.rssm.prior(post_deter)
        dyn_loss, rep_loss   = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        burn_in_mask         = data.get("burn_in_mask", None)
        losses["dyn"]        = masked_mean(dyn_loss, burn_in_mask)
        losses["rep"]        = masked_mean(rep_loss, burn_in_mask)
        metrics["dyn_entropy"] = float(mx.mean(
            mx.sum(mx.softmax(prior_logit, axis=-1) * mx.log_softmax(prior_logit, axis=-1), axis=-1)
        ).item())

        
        feat             = self.rssm.get_feat(post_stoch, post_deter)  
        flat_post_stoch  = post_stoch.reshape(B, T, -1)

        
        if self.phase != 2 and T > 1:
            inv_end     = min(T - 1, self.ctx_len - 1)
            feat_t      = mx.stop_gradient(feat[:, :inv_end])
            feat_t1     = mx.stop_gradient(feat[:, 1:inv_end + 1])
            d_emb_inv   = d_emb[:, :inv_end]
            pred_action = self.inv_dyn_head(feat_t, feat_t1, d_emb_inv)
            tgt_action  = action[:, :inv_end]
            inv_mse     = mx.mean((pred_action - tgt_action) ** 2, axis=-1)

            inv_mask = (
                burn_in_mask[:, :inv_end] if burn_in_mask is not None
                else mx.ones((B, inv_end), dtype=mx.float32)
            )
            losses["inv_dyn"] = self.inv_dyn_loss_weight * masked_mean(inv_mse, inv_mask)

        
        if self.rep_loss == "nedreamer":
            if T > 1:
                state_seq = self.rssm.get_feat(post_stoch, post_deter)
                tr_out = self.nedreamer_transformer(state_seq, action)
                target_embed = (
                    self.target_encoder(image)
                    if getattr(self, "target_encoder", None) is not None
                    else embed
                )
                pred = self.nedreamer_predictor(tr_out[:, :-1])
                target = mx.stop_gradient(target_embed[:, 1:])
                valid = (1.0 - data["is_terminal"][:, :-1].astype(mx.float32))
                if burn_in_mask is not None:
                    valid = valid * burn_in_mask[:, :-1].astype(mx.float32)
                flat_pred = mx.reshape(pred, (B * (T - 1), -1))
                flat_target = mx.reshape(target, (B * (T - 1), -1))
                flat_valid = mx.reshape(valid, (B * (T - 1),)).astype(mx.float32)
                losses["nedreamer"], barlow_metrics = self._barlow_loss_masked(
                    flat_pred, flat_target, flat_valid, self.barlow_lambd
                )
            else:
                losses["nedreamer"] = mx.zeros(())
                barlow_metrics = {"barlow/invariance": 0.0, "barlow/redundancy": 0.0, "barlow/std_mean": 0.0}
            metrics.update({f"nedreamer/{k.split('/')[-1]}": v for k, v in barlow_metrics.items()})
        else:
            x1 = self.projector(feat.reshape(B * T, -1))
            x2 = self.projector_target(mx.stop_gradient(embed.reshape(B * T, -1)))
            losses["barlow"], barlow_metrics = self._barlow_loss(x1, x2, self.barlow_lambd)
            metrics.update(barlow_metrics)

        
        losses["rew"] = mx.mean(-self.reward_head(feat).log_prob(
            data["reward"].astype(mx.float32)
        ))
        cont          = (1.0 - data["is_terminal"].astype(mx.float32))
        losses["con"] = mx.mean(-self.cont_head(feat).log_prob(cont))

        
        if self.phase == 2:
            ctx_student = self.context_encoder(
                flat_post_stoch[:, :min(T, self.ctx_len)],
                post_deter[:, :min(T, self.ctx_len)],
                action[:, :min(T, self.ctx_len)],
            )
            ctx_time = mx.broadcast_to(
                mx.expand_dims(ctx_student, 1), (B, T, self.d_emb_dim)
            )
            policy_inp = mx.concatenate([
                mx.stop_gradient(feat), mx.stop_gradient(d_emb)
            ], axis=-1)
            from native.mlx_distributions import bounded_normal
            actor_out  = mx.concatenate(
                mx.split(self.actor.backbone(policy_inp), 2, axis=-1), axis=-1
            )
            actor_dist = bounded_normal(actor_out)
            bc_loss    = -actor_dist.log_prob(action.astype(mx.float32))
            losses["bc"]        = masked_mean(bc_loss, burn_in_mask)
            losses["ctx_align"] = mx.mean((ctx_time - mx.stop_gradient(d_emb)) ** 2)

            speed        = data.get("speed", mx.zeros((B, T, 1)))
            crash_target = data.get("crash", cont[..., None].astype(mx.float32))
            skey         = getattr(self, "safety_input_key", "image")
            safety_img   = data.get(skey, image)
            stacked      = self._stack_frames(safety_img, self.safety_net.frame_stack)
            safety_logit, _ = self.safety_net(stacked, speed, action)
            losses["safety"] = focal_bce(safety_logit, crash_target)
            metrics["safety/score_mean"] = float(mx.mean(mx.sigmoid(safety_logit)).item())

        
        elif self.phase >= 3:
            self._ctx_updates += 1
            ctx_student = self.context_encoder(
                flat_post_stoch[:, :min(T, self.ctx_len)],
                post_deter[:, :min(T, self.ctx_len)],
                action[:, :min(T, self.ctx_len)],
            )
            d_emb = mx.broadcast_to(
                mx.expand_dims(ctx_student, 1), (B, T, self.d_emb_dim)
            )

            
            start_stoch = mx.stop_gradient(post_stoch.reshape(B * T, *post_stoch.shape[2:]))
            start_deter = mx.stop_gradient(post_deter.reshape(B * T, post_deter.shape[-1]))
            flat_d_emb  = mx.stop_gradient(d_emb.reshape(B * T, self.d_emb_dim))
            BT          = B * T

            
            imag_state  = RSSMState(
                stoch=start_stoch, deter=start_deter,
                prev_action=mx.zeros((BT, self.rssm.cfg.act_dim)),
                prev_filtered_action=mx.zeros((BT, self.rssm.cfg.act_dim)),
            )
            imag_feats, imag_acts = [], []
            for _ in range(self.imag_horizon + 1):
                feat_t   = self.rssm.get_feat(imag_state.stoch, imag_state.deter)
                pi_inp   = mx.concatenate([feat_t, flat_d_emb], axis=-1)
                act_t    = self.actor(pi_inp, sample=True)
                imag_feats.append(feat_t)
                imag_acts.append(act_t)
                imag_state = self.rssm.img_step(imag_state, act_t, flat_d_emb)

            imag_feat   = mx.stack(imag_feats, axis=1)     
            imag_action = mx.stack(imag_acts,  axis=1)     
            imag_d_emb  = mx.broadcast_to(
                mx.expand_dims(flat_d_emb, 1),
                (BT, self.imag_horizon + 1, self.d_emb_dim),
            )
            pi_inp_full = mx.concatenate([
                mx.stop_gradient(imag_feat), imag_d_emb
            ], axis=-1)

            imag_rew  = self.reward_head(mx.stop_gradient(imag_feat)).mode()
            imag_cont = self.cont_head(mx.stop_gradient(imag_feat)).mean()
            imag_val  = mx.stop_gradient(
                symexp_twohot(self.value(pi_inp_full), bin_num=255).mode()
            )

            disc   = 1.0 - 1.0 / self.horizon
            weight = mx.cumprod(imag_cont * disc, axis=1)
            last   = mx.zeros_like(imag_cont)
            ret    = self._lambda_return(
                last, 1.0 - imag_cont, imag_rew, imag_val, imag_val, disc, self.lamb
            )
            ret_offset, ret_scale = self.return_ema(ret)
            adv    = (ret - imag_val[:, :-1]) / ret_scale

            policy_inp  = mx.concatenate([imag_feat, imag_d_emb], axis=-1)
            from native.mlx_distributions import bounded_normal
            actor_raw   = mx.concatenate(
                mx.split(self.actor.backbone(policy_inp[:, :-1]), 2, axis=-1), axis=-1
            )
            actor_dist  = bounded_normal(actor_raw)
            logpi       = mx.expand_dims(actor_dist.log_prob(imag_action[:, :-1]), -1)
            entropy     = mx.expand_dims(actor_dist.entropy(), -1)
            losses["policy"] = mx.mean(
                mx.stop_gradient(weight[:, :-1])
                * -(logpi * mx.stop_gradient(adv) + self.act_entropy * entropy)
            )

            val_dist       = symexp_twohot(self.value(pi_inp_full), bin_num=255)
            tar_padded     = mx.concatenate([ret, mx.zeros_like(ret[:, -1:])], axis=1)
            slow_val       = mx.stop_gradient(imag_val)
            losses["value"] = mx.mean(
                mx.stop_gradient(weight[:, :-1])
                * (-val_dist.log_prob(mx.stop_gradient(tar_padded))
                   - val_dist.log_prob(slow_val))[:, :-1, None]
            )

            
            half = min(max(1, T // 2), self.ctx_len)
            z_a  = self.context_encoder(flat_post_stoch[:, :half],
                                        post_deter[:, :half], action[:, :half])
            z_b  = self.context_encoder(flat_post_stoch[:, T - half:],
                                        post_deter[:, T - half:], action[:, T - half:])
            warm = min(1.0, self._ctx_updates / max(1, self.ctx_warmup_steps))
            losses["ctx_consistency"] = (
                self.ctx_consistency_w * warm * mx.mean((z_a - mx.stop_gradient(z_b)) ** 2)
            )

            
            skey       = getattr(self, "safety_input_key", "image")
            safety_img = data.get(skey, image)
            stacked    = self._stack_frames(safety_img, self.safety_net.frame_stack)
            speed      = data.get("speed", mx.zeros((B, T, 1)))
            crash_tgt  = data.get("crash", cont[..., None].astype(mx.float32))
            logit_p3, safe_act = self.safety_net(stacked, speed, action)
            losses["safety"] = focal_bce(logit_p3, crash_tgt)
            metrics["safety/score_mean"] = float(mx.mean(mx.sigmoid(logit_p3)).item())

            if "expert_evasion" in data:
                exp_ev  = data["expert_evasion"]
                exp_act = data.get(
                    "expert_active",
                    (mx.sigmoid(mx.stop_gradient(logit_p3)).squeeze(-1) > 0.3).astype(mx.float32),
                )
                if exp_act.ndim == 3:
                    exp_act = exp_act.squeeze(-1)
                exp_w    = mx.expand_dims(exp_act, -1)
                dist_tgt = exp_w * exp_ev
                dist_loss = (safe_act - dist_tgt) ** 2
                sample_w  = exp_w + (1.0 - exp_w) * 0.1
                losses["safety_distill"] = mx.mean(dist_loss * sample_w)

            metrics.update({
                "ret":              float(mx.mean(ret).item()),
                "adv":              float(mx.mean(adv).item()),
                "val":              float(mx.mean(imag_val).item()),
                "weight":           float(mx.mean(weight).item()),
                "action_entropy":   float(mx.mean(entropy).item()),
            })

        
        total = sum(
            v * self._loss_scales.get(k, 1.0)
            for k, v in losses.items()
        )
        return total, losses, metrics
