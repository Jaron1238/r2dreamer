

import math
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
            # Inject the current timestep's action as prev_action so the
            # motor-inertia filter sees the real action sequence.
            # prev_filtered_action carries momentum from the previous step.
            state = RSSMState(
                stoch=state.stoch,
                deter=state.deter,
                prev_action=action[:, t],
                prev_filtered_action=state.prev_filtered_action,
            )
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

class MLXSpatialAttentionPool(nn.Module):
    """9-Query (3×3) cross-attention pooling — MLX port of SpatialAttentionPool.

    Mirrors networks.py (PyTorch):
    - 2D sinusoidal positional encoding, learnable (warm-start from sinusoids × 0.1)
    - 9 orthogonally initialised queries → no slot collapse
    - Manual MHA (avoids MLX MHA API differences from PyTorch batch_first=True)
    - RMSNorm on output slots
    """

    def __init__(
        self,
        backbone_ch: int,
        embed_dim:   int,
        num_queries: int = 9,
        num_heads:   int = 4,
        feat_h:      int = 9,
        feat_w:      int = 16,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_queries = num_queries
        self.num_heads   = num_heads
        self.head_dim    = embed_dim // num_heads

        # Shared K/V projection
        self.in_proj = nn.Linear(backbone_ch, embed_dim, bias=False)

        # Learnable 2D sinusoidal PE: (1, H*W, embed_dim)
        # Stored as plain mx.array → treated as trainable by MLX.
        self.pos_embed = mx.array(
            self._make_2d_sinusoidal(feat_h, feat_w, embed_dim), dtype=mx.float32
        )

        # 9 orthogonal query vectors — mirrors nn.init.orthogonal_ in PyTorch
        self.queries = mx.array(
            self._orthogonal_init(num_queries, embed_dim), dtype=mx.float32
        )

        # MHA projections (manual: avoids MLX MHA API surface differences)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm = nn.RMSNorm([embed_dim], eps=1e-4)
        # 9 slots → embed_dim (mirrors PyTorch: nn.Linear(num_queries * embed_dim, embed_dim))
        self.out  = nn.Linear(num_queries * embed_dim, embed_dim, bias=False)

    @staticmethod
    def _make_2d_sinusoidal(h: int, w: int, d: int):
        import numpy as _np
        d_half   = d // 2
        div_term = _np.exp(
            _np.arange(0, d_half, 2) * -(_np.log(10_000.0) / d_half)
        )
        pe_h = _np.zeros((h, d_half), dtype=_np.float32)
        rows = _np.arange(h, dtype=_np.float32)[:, None]
        pe_h[:, 0::2] = _np.sin(rows * div_term)
        pe_h[:, 1::2] = _np.cos(rows * div_term)
        pe_w = _np.zeros((w, d_half), dtype=_np.float32)
        cols = _np.arange(w, dtype=_np.float32)[:, None]
        pe_w[:, 0::2] = _np.sin(cols * div_term)
        pe_w[:, 1::2] = _np.cos(cols * div_term)
        # Grid: (h, w, d)
        pe = _np.concatenate([
            _np.broadcast_to(pe_h[:, None, :], (h, w, d_half)),
            _np.broadcast_to(pe_w[None, :, :], (h, w, d_half)),
        ], axis=-1)
        return pe.reshape(1, h * w, d) * 0.1

    @staticmethod
    def _orthogonal_init(n: int, d: int):
        import numpy as _np
        _np.random.seed(42)
        mat = _np.random.randn(max(n, d), min(n, d)).astype(_np.float32)
        U, _, Vt = _np.linalg.svd(mat, full_matrices=False)
        Q = U if n <= d else Vt.T
        return Q[:n, :d]

    def __call__(self, x: mx.array) -> mx.array:
        """x: (N, H, W, C) NHWC"""
        N, H, W, C = x.shape
        n_tokens = H * W
        D, Dh, nH = self.queries.shape[-1], self.head_dim, self.num_heads
        Nq, Ns    = self.num_queries, n_tokens
        scale     = Dh ** -0.5

        # K/V token sequence: (N, HW, backbone_ch) → (N, HW, embed_dim)
        tokens = x.reshape(N, n_tokens, C)
        tokens = self.in_proj(tokens) + self.pos_embed   # broadcast over N

        # Q from learnable query slots: (N, Nq, D)
        q = mx.broadcast_to(self.queries[None], (N, Nq, D))

        # Project Q/K/V then reshape to (N, nH, seq, Dh)
        Q_ = self.q_proj(q).reshape(N, Nq, nH, Dh).transpose(0, 2, 1, 3)
        K_ = self.k_proj(tokens).reshape(N, Ns, nH, Dh).transpose(0, 2, 1, 3)
        V_ = self.v_proj(tokens).reshape(N, Ns, nH, Dh).transpose(0, 2, 1, 3)

        attn = mx.softmax((Q_ @ K_.swapaxes(-2, -1)) * scale, axis=-1)
        out  = (attn @ V_).transpose(0, 2, 1, 3).reshape(N, Nq, D)
        out  = self.o_proj(out)
        out  = self.norm(out)                             # (N, Nq, D)

        return self.out(out.reshape(N, Nq * D))           # (N, embed_dim)


class MLXConvEncoder(nn.Module):
    """EfficientNet-B0-like backbone with FPN-light + SpatialAttentionPool.

    Structural parity with PyTorch ConvEncoder (networks.py lines ~317-429):
      - early stages (features[:3])  → H/8,  40 ch  (geometric edges/cables)
      - late  stages (features[3:])  → H/32, 1280 ch (semantic context)
      - fpn_reduce: RMSNorm + 1×1 Conv on early features
      - SpatialAttentionPool: 9-query 3×3 grid, orthogonal init, 2D PE

    Note: backbone is EfficientNet-B0 approximation (torchvision V2-S unavailable
    in MLX). Structural parity (FPN + AttnPool) is preserved; pretrained weights
    from PyTorch cannot be transferred (mlx_utils.py remapping won't match).

    out_dim = depth × mults[-1]  — mirrors PyTorch ConvEncoder.out_dim.
    """

    def __init__(
        self,
        in_ch:   int,
        input_h: int,
        input_w: int,
        depth:   int             = 48,
        mults:   tuple[int, ...] = (2, 3, 4, 4),
    ) -> None:
        super().__init__()
        self.input_ch = int(in_ch)
        embed_dim     = int(depth) * int(mults[-1])
        self.out_dim  = embed_dim

        # === Stem (Conv2d → BN → SiLU) ===
        self.stem_conv = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm(32, eps=1e-3, momentum=0.01)

        # === MBConv stages split at stage-2 boundary (mirrors V2-S features[:4] / [4:]) ===
        # early: stages 0-2 → 40ch at H/8   (geometric / fine detail)
        # late:  stages 3-6 → 320ch at H/32  (semantic / coarse context)
        early_blocks: list = []
        late_blocks:  list = []
        in_c = 32
        for idx, (exp, out_c, n_blocks, stride, k) in enumerate(_B0_STAGES):
            for j in range(n_blocks):
                s = stride if j == 0 else 1
                blk = _MLXMBConvBlock(in_c, out_c, exp, k, s)
                if idx < 3:
                    early_blocks.append(blk)
                else:
                    late_blocks.append(blk)
                in_c = out_c

        self.early_stages: list = early_blocks   # 40ch at H/8
        self.late_stages:  list = late_blocks    # 320ch at H/32

        # Head: 320 → 1280
        self.head_conv = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.head_bn   = nn.BatchNorm(1280, eps=1e-3, momentum=0.01)

        early_ch = 40
        deep_ch  = 1280
        fused_ch = early_ch + deep_ch           # 1320 (vs 1344 in PyTorch V2-S)

        # FPN-light: RMSNorm over channels (NHWC last dim) + 1×1 Conv + SiLU
        # Mirrors PyTorch fpn_reduce: Conv2d(early_ch, early_ch, 1) + RMSNorm2D + SiLU
        self.fpn_reduce_conv = nn.Conv2d(early_ch, early_ch, kernel_size=1, bias=False)
        self.fpn_reduce_norm = nn.RMSNorm([early_ch], eps=1e-4)

        # feat_h, feat_w = H/32, W/32 for the deep feature map
        self.feat_h = max(1, int(input_h) // 32)
        self.feat_w = max(1, int(input_w) // 32)

        self.pool = MLXSpatialAttentionPool(
            backbone_ch = fused_ch,
            embed_dim   = embed_dim,
            num_queries = 9,
            num_heads   = 4,
            feat_h      = self.feat_h,
            feat_w      = self.feat_w,
        )

    def __call__(self, obs: mx.array) -> mx.array:
        # Preprocessing — mirrors PyTorch ConvEncoder.forward() exactly:
        #   RGB channels:   obs - 0.5
        #   Extra channels: obs / 2.0
        C = obs.shape[-1]
        if C <= 3:
            x = obs - 0.5
        else:
            rgb  = obs[..., :3] - 0.5
            diff = obs[..., 3:] * 0.5
            x = mx.concatenate([rgb, diff], axis=-1)

        x = x.reshape(-1, *x.shape[-3:])           # flatten (B, T, H, W, C) → (N, H, W, C)

        # Stem
        x = nn.silu(self.stem_bn(self.stem_conv(x)))

        # Early stages → (N, H/8, W/8, 40)
        for blk in self.early_stages:
            x = blk(x)
        early = x

        # Late stages → (N, H/32, W/32, 320) → head → 1280
        for blk in self.late_stages:
            x = blk(x)
        x = nn.silu(self.head_bn(self.head_conv(x)))   # (N, H/32, W/32, 1280)
        deep = x

        # FPN-light: reduce early features and avg-pool to deep resolution
        early_proj = nn.silu(self.fpn_reduce_norm(self.fpn_reduce_conv(early)))
        N, H_e, W_e, C_e = early_proj.shape
        fh, fw = self.feat_h, self.feat_w
        sh = max(1, H_e // fh)
        sw = max(1, W_e // fw)
        # Non-overlapping average pool via reshape (MLX has no adaptive_avg_pool2d)
        ep          = early_proj[:, :fh * sh, :fw * sw, :]
        ep          = ep.reshape(N, fh, sh, fw, sw, C_e)
        early_down  = ep.mean(axis=(2, 4))              # (N, fh, fw, C_e=40)

        # Fuse: (N, fh, fw, 1320)
        fused = mx.concatenate([deep, early_down], axis=-1)

        embed = self.pool(fused)                        # (N, embed_dim)
        return embed.reshape(*obs.shape[:-3], self.out_dim)


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
    

    def __init__(self, in_channels: int = 1, action_dim: int = 4, speed_dim: int = 1, hidden: int = 64, frame_stack: int = 3):
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
        self.ctx_len = int(ctx_len)
        self.bottleneck = int(bottleneck)
        self.out_dim = int(out_dim)
        self.encoder_type = str(encoder_type)

        inp_dim = flat_stoch + deter + act_dim

        
        self.proj = nn.Sequential(
            nn.Linear(inp_dim, bottleneck),
            RMSNorm(bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
            RMSNorm(bottleneck),
            nn.SiLU(),
        )

        
        if self.encoder_type == "gru":
            
            
            self.rnn = nn.GRU(bottleneck, bottleneck, num_layers=2)
        elif self.encoder_type == "transformer":
            half_dim = self.bottleneck // 2
            
            freqs = 1.0 / (10_000 ** (mx.arange(0, half_dim, dtype=mx.float32) / half_dim))
            self.rope_freqs = freqs

            num_heads = max(1, self.bottleneck // 64)
            self.transformer_layers =[
                MLXTransformerEncoderLayer(self.bottleneck, num_heads) for _ in range(2)
            ]
        else:
            raise ValueError(f"Unbekannter encoder_type='{encoder_type}'")

        
        self.temporal_attn = nn.Linear(self.bottleneck, 1)
        self.out_head = nn.Linear(self.bottleneck, self.out_dim)

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

        
        padding_mask = None
        if valid_len is not None:
            positions = mx.expand_dims(mx.arange(T, dtype=mx.int32), 0)
            valid_start = mx.expand_dims(T - mx.clip(valid_len, None, T), 1)
            padding_mask = positions < valid_start  

        x = mx.concatenate([flat_stoch, deter, action], axis=-1)
        x = self.proj(x)

        
        if self.encoder_type == "gru":
            # Bug #15 (MLX-Port): zero out leading padded positions before the GRU
            # to prevent padding-contaminated hidden states bleeding into valid positions.
            # The temporal-attention pool below already masks via padding_mask, but
            # zeroing the inputs makes the GRU path consistent with the transformer path.
            if padding_mask is not None:
                x = mx.where(mx.expand_dims(padding_mask, -1), mx.zeros_like(x), x)
            x = self.rnn(x)
        else:
            x = self._apply_rope(x)
            attn_mask = None
            if padding_mask is not None:
                
                attn_mask = mx.expand_dims(mx.expand_dims(padding_mask, 1), 1)
                attn_mask = mx.where(
                    attn_mask,
                    mx.array(-float("inf"), dtype=x.dtype),
                    mx.array(0.0, dtype=x.dtype),
                )

            for layer in self.transformer_layers:
                x = layer(x, mask=attn_mask)

        
        attn_logits = self.temporal_attn(x)  
        if padding_mask is not None:
            attn_logits = mx.where(
                mx.expand_dims(padding_mask, -1),
                mx.array(-float("inf"), dtype=x.dtype),
                attn_logits,
            )

        attn_w = mx.softmax(attn_logits, axis=1)
        ctx = mx.sum(attn_w * x, axis=1)

        return self.out_head(ctx)

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

    def __call__(
        self,
        feat_t:  mx.array,   
        feat_t1: mx.array,   
        d_emb:   mx.array,   
    ) -> mx.array:
        x = mx.concatenate([feat_t, feat_t1, d_emb], axis=-1)
        return self.net(x)


def _upsample2x(x: mx.array) -> mx.array:
    """Nearest-neighbor 2× spatial upsampling: (N, H, W, C) → (N, 2H, 2W, C)."""
    return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


class MLXDepthAuxHead(nn.Module):
    """Depth auxiliary prediction head — MLX port of PyTorch DepthAuxHead.

    Uses nearest-neighbor upsampling + Conv2d instead of ConvTranspose2d
    (NHWC convention, 4× upsampling chain, softplus output).
    """

    def __init__(self, in_dim: int, out_h: int, out_w: int) -> None:
        super().__init__()
        self.out_h  = int(out_h)
        self.out_w  = int(out_w)
        hidden      = max(128, in_dim // 2)
        self.base_h = max(4, self.out_h // 16)
        self.base_w = max(4, self.out_w // 16)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 64 * self.base_h * self.base_w),
            nn.SiLU(),
        )
        # 4× upsampling stages: each doubles H and W via upsample + conv
        self.up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.up3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.up4 = nn.Conv2d(16,  1, kernel_size=3, padding=1)

    def __call__(self, post_feat: mx.array) -> mx.array:
        b, t, _ = post_feat.shape
        x = self.proj(post_feat.reshape(b * t, -1))
        x = x.reshape(b * t, self.base_h, self.base_w, 64)  # NHWC
        x = nn.silu(self.up1(_upsample2x(x)))
        x = nn.silu(self.up2(_upsample2x(x)))
        x = nn.silu(self.up3(_upsample2x(x)))
        x = self.up4(_upsample2x(x))                         # (B*T, H', W', 1)
        x = x[:, :self.out_h, :self.out_w, :]               # exact crop
        x = mx.log1p(mx.exp(x))                             # softplus → positive
        return x.reshape(b, t, self.out_h, self.out_w, 1)


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
        from native.mlx_distributions import (
            symexp_twohot, binary, bounded_normal,
        )

        self.phase               = int(getattr(cfg, "phase", 1))
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

        rssm_cfg = RSSMConfig(**cfg.rssm)
        self.rssm = MLXRSSM(rssm_cfg)

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

        # --- VLA placeholder (mirrors dreamer.py line ~138) ---
        self.vla_enabled = bool(getattr(cfg, "vla_enabled", False))
        if self.vla_enabled:
            import warnings
            warnings.warn(
                "[MLXDreamer] vla_enabled=True: VLA conditioning is not yet implemented "
                "in the MLX simulator backend. VLA inputs will be ignored during training.",
                stacklevel=2,
            )

        # --- Depth-aux head (mirrors dreamer.py lines ~126-133) ---
        self.use_depth_aux      = bool(getattr(cfg, "use_depth_aux", False))
        self.use_depth_aux_prob = float(getattr(cfg, "use_depth_aux_prob", 1.0))
        if self.phase >= 3 and self.use_depth_aux:
            depth_cfg = getattr(cfg, "depth_aux", None)
            _out_h = int(getattr(depth_cfg, "out_h", 64)) if depth_cfg is not None else 64
            _out_w = int(getattr(depth_cfg, "out_w", 64)) if depth_cfg is not None else 64
            self.depth_aux_head = MLXDepthAuxHead(
                in_dim = self.rssm.feat_size,
                out_h  = _out_h,
                out_w  = _out_w,
            )

    @staticmethod
    def _scale_invariant_depth_loss(pred: mx.array, target: mx.array) -> mx.array:
        """Scale-invariant log-depth loss (Eigen et al.)."""
        pred_log   = mx.log(pred   + 1e-6)
        target_log = mx.log(target + 1e-6)
        diff       = pred_log - target_log
        mse        = mx.mean(diff ** 2)
        mean_term  = mx.mean(diff) ** 2
        return mse - 0.5 * mean_term

    @staticmethod
    def _barlow_loss(z_a: mx.array, z_b: mx.array, lambd: float) -> tuple[mx.array, dict]:
        N, D      = z_a.shape
        eps       = 1e-8
        z_a_norm  = (z_a - z_a.mean(axis=0)) / (z_a.std(axis=0) + eps)
        z_b_norm  = (z_b - z_b.mean(axis=0)) / (z_b.std(axis=0) + eps)
        c         = mx.matmul(z_a_norm.T, z_b_norm) / N          
        diag      = mx.diag(c)
        inv_loss  = mx.sum((diag - 1.0) ** 2)
        off_diag  = c - mx.diag(diag)
        red_loss  = mx.sum(off_diag ** 2)
        loss      = inv_loss + lambd * red_loss
        metrics   = {
            "barlow/invariance": float(inv_loss.item()),
            "barlow/redundancy": float(red_loss.item()),
            "barlow/std_mean":   float(z_a_norm.std(axis=0).mean().item()),
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

    def compute_losses(
        self,
        data:    dict,
        initial: RSSMState | None = None,
    ) -> tuple[mx.array, dict, dict]:
        
        from native.mlx_distributions import (
            masked_mean, focal_bce, symexp_twohot, binary,
        )
        from functools import partial

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
            from native.mlx_distributions import BoundedNormalDist, bounded_normal
            actor_out  = mx.concatenate(
                mx.split(self.actor.backbone(policy_inp), 2, axis=-1), axis=-1
            )
            actor_dist = bounded_normal(actor_out)
            bc_loss    = -actor_dist.log_prob(action.astype(mx.float32))
            losses["bc"] = masked_mean(bc_loss, burn_in_mask)

            # Two-window Barlow Twins on context encoder — mirrors dreamer.py ~926-935.
            # Replaces old degenerate ctx_align MSE-against-d_emb.
            ctx_end = min(T, self.ctx_len)
            mid     = ctx_end // 2
            if mid > 0 and ctx_end - mid > 0:
                z_a = self.context_encoder(
                    flat_post_stoch[:, :mid], post_deter[:, :mid], action[:, :mid]
                )
                z_b = self.context_encoder(
                    flat_post_stoch[:, mid:ctx_end],
                    post_deter[:, mid:ctx_end],
                    action[:, mid:ctx_end],
                )
                losses["ctx_barlow"], ctx_barlow_m = self._barlow_loss(
                    z_a, mx.stop_gradient(z_b), self.barlow_lambd
                )
                metrics.update({f"ctx/{k.split('/')[-1]}": v for k, v in ctx_barlow_m.items()})
            else:
                losses["ctx_barlow"] = mx.zeros(())

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

            # --- flat_mask: select only valid (non-burn-in) positions ---
            # Mirrors dreamer.py ~980-990 (Bug #3 fix: prevents shape mismatch
            # in _lambda_return when T_valid < T due to active burn_in_mask).
            if burn_in_mask is not None:
                flat_mask = burn_in_mask.reshape(-1).astype(mx.bool_)
            else:
                flat_mask = mx.ones((B * T,), dtype=mx.bool_)

            start_stoch = mx.stop_gradient(post_stoch.reshape(B * T, *post_stoch.shape[2:]))[flat_mask]
            start_deter = mx.stop_gradient(post_deter.reshape(B * T, post_deter.shape[-1]))[flat_mask]
            flat_d_emb  = mx.stop_gradient(d_emb.reshape(B * T, self.d_emb_dim))[flat_mask]
            BT          = start_stoch.shape[0]

            
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
            from native.mlx_distributions import BoundedNormalDist, bounded_normal
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


            
            replay_pi_inp = mx.concatenate([feat, d_emb], axis=-1)
            last_f  = data["is_last"].astype(mx.float32)
            term_f  = data["is_terminal"].astype(mx.float32)
            rew_f   = data["reward"].astype(mx.float32)

            # Boot-scatter fix (Bug #3 MLX-Port):
            # ret[:, 0] has shape (N, 1) where N = flat_mask.sum() ≤ B*T.
            # The old ret[:, 0].reshape(B,-1,1)[:,:T] silently yields wrong
            # shapes when burn_in_mask is active.  Use a cumsum scatter instead:
            #   cumidx[i] = index into ret[:, 0] for position i in (B*T,)
            boot_vals = ret[:, 0, 0]                                     # (N,)
            n_valid   = boot_vals.shape[0]
            cumidx    = mx.clip(
                mx.cumsum(flat_mask.astype(mx.int32)) - 1, 0, max(n_valid - 1, 0)
            )
            boot_dense = mx.where(
                flat_mask, boot_vals[cumidx], mx.zeros((), dtype=ret.dtype)
            )
            boot = boot_dense.reshape(B, T, 1)                           # always (B, T, 1)

            rval    = mx.stop_gradient(
                symexp_twohot(self.value(replay_pi_inp), bin_num=255).mode()
            )
            slow_rval = mx.stop_gradient(rval)
            ret_r     = self._lambda_return(last_f, term_f, rew_f, rval, boot, disc, self.lamb)
            ret_r_pad = mx.concatenate([ret_r, mx.zeros_like(ret_r[:, -1:])], axis=1)
            rval_dist = symexp_twohot(self.value(replay_pi_inp), bin_num=255)
            weight_r  = (1.0 - last_f)
            losses["repval"] = mx.mean(
                weight_r[:, :-1]
                * (-rval_dist.log_prob(mx.stop_gradient(ret_r_pad))
                   - rval_dist.log_prob(slow_rval))[:, :-1, None]
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
                metrics["safety/distill_danger_frac"] = float(mx.mean(exp_act).item())
                metrics["safety/distill_loss"]        = float(losses["safety_distill"].item())

            # --- Depth-aux loss (mirrors dreamer.py ~1118-1125) ---
            # Key fix: use `use_depth_aux` (not the stale `use_depth`).
            if bool(getattr(self, "use_depth_aux", False)) and "depth_target" in data:
                use_now = (
                    self.use_depth_aux_prob >= 1.0
                    or float(mx.random.uniform()) < self.use_depth_aux_prob
                )
                metrics["depth_aux/active"] = float(use_now)
                if use_now:
                    post_feat_3d = mx.concatenate(
                        [post_stoch.reshape(B, T, -1), post_deter], axis=-1
                    )
                    depth_pred   = self.depth_aux_head(post_feat_3d)
                    depth_target = data["depth_target"].astype(mx.float32)
                    losses["depth_aux"] = self._scale_invariant_depth_loss(
                        depth_pred, depth_target
                    )

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
