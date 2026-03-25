import math
import re
from functools import partial
import torchvision.models as tv_models
import torch
import torch.nn.functional as F
from torch import nn

import distributions as dists
from tools import weight_init_


class LambdaLayer(nn.Module):
    """Wrap an arbitrary callable into an ``nn.Module``."""

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BlockLinear(nn.Module):
    """Block-wise linear layer.

    Weight layout is chosen to cooperate with PyTorch's fan-in/fan-out
    calculation used by initializers.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)

        # Store weight in a layout that works with torch's fan calculation.
        # (O/G, I/G, G)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., I)
        batch_shape = x.shape[:-1]
        # Reshape to expose block dimension.
        # (..., I) -> (..., G, I/G)
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)

        # Block-wise multiplication
        # (..., G, I/G), (O/G, I/G, G) -> (..., G, O/G)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        # Merge block dimension back.
        # (..., G, O/G) -> (..., O)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class Conv2dSamePad(nn.Conv2d):
    """A Conv2d layer that emulates TensorFlow's 'SAME' padding."""

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        i_div_s_ceil = (i + s - 1) // s
        return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over channel-last format applied to 4D tensors."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMSNorm over the channel dimension.
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MultiEncoder(nn.Module):
    def __init__(
        self,
        config,
        shapes,
    ):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded and not k.startswith("log_")}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.out_dim = 0
        self.encoder_specs = []
        self.encoders = []
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self.encoders.append(ConvEncoder(config.cnn, input_shape))
            self.encoder_specs.append(("cnn", tuple(self.cnn_shapes.keys())))
            self.out_dim += self.encoders[-1].out_dim
        if self.mlp_shapes:
            inp_dim = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoders.append(MLP(config.mlp, inp_dim))
            self.encoder_specs.append(("mlp", tuple(self.mlp_shapes.keys())))
            self.out_dim += self.encoders[-1].out_dim
        self.encoders = nn.ModuleList(self.encoders)
        if len(self.encoders) == 0:
            raise NotImplementedError

        self.apply(weight_init_)

    def forward(self, obs):
        """Encode a dict of observations."""
        # dict of (B, T, *)
        outputs = []
        for encoder, (_, keys) in zip(self.encoders, self.encoder_specs):
            selected = torch.cat([obs[k] for k in keys], dim=-1)
            outputs.append(encoder(selected))
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=-1)


class MultiDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shapes):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        self.all_keys = list(self.mlp_shapes.keys()) + list(self.cnn_shapes.keys())

        # Unlike the encoder, each decoder is initialized independently.
        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                config.cnn,
                deter,
                flat_stoch,
                shape,
            )
            self._image_dist = partial(getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist)
        if self.mlp_shapes:
            shape = (sum(sum(x) for x in self.mlp_shapes.values()),)
            config.mlp.shape = shape
            self._mlp = MLPHead(config.mlp, deter + flat_stoch)
            self._mlp_dist = partial(getattr(dists, str(config.mlp_dist.name)), **config.mlp_dist)

    def forward(self, stoch, deter):
        """Decode latent states into observation distributions."""
        # (B, T, S, K), (B, T, D)
        dists = {}
        if self.cnn_shapes:
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            # (B, T, H, W, C_sum)
            outputs = self._cnn(stoch, deter)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._image_dist(output) for key, output in zip(self.cnn_shapes.keys(), outputs)})
        if self.mlp_shapes:
            split_sizes = [v[0] for v in self.mlp_shapes.values()]
            # (B, T, S*K + D)
            feat = torch.cat([stoch.reshape(*deter.shape[:-1], -1), deter], -1)
            outputs = self._mlp(feat)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._mlp_dist(output) for key, output in zip(self.mlp_shapes.keys(), outputs)})
        return dists


class ConvEncoder(nn.Module):
    def __init__(self, config, input_shape):
        super().__init__()
        h, w, input_ch = input_shape
        self.input_ch = int(input_ch)

        # EfficientNet-B0 Backbone laden
        backbone = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)

        # Layer 1 auf variable Channels erweitern (z. B. Depth + Diff = 2)
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels=self.input_ch,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        # Pretrained Gewichte kopieren / wiederholen.
        with torch.no_grad():
            if self.input_ch <= 3:
                backbone.features[0][0].weight[:, :self.input_ch] = old_conv.weight[:, :self.input_ch]
            else:
                repeat = math.ceil(self.input_ch / 3)
                repeated = old_conv.weight.repeat(1, repeat, 1, 1)[:, :self.input_ch]
                backbone.features[0][0].weight.copy_(repeated)

        # Modality-shift (RGB -> Depth/Flow): kompletten Backbone feinjustieren.
        for param in backbone.features.parameters():
            param.requires_grad = True

        # Classifier-Head entfernen
        self.backbone = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Adapter: 1280 (EfficientNet Output) → was RSSM erwartet
        embed_dim = int(config.depth) * int(config.mults[-1])
        self.adapter = nn.Linear(1280, embed_dim)
        self.out_dim = embed_dim

    def forward(self, obs):
        """Encode image-like observations mit EfficientNet-B0."""
        # (B, T, H, W, C)
        channels = obs.shape[-1]
        if channels <= 3:
            obs = obs - 0.5
        else:
            rgb = obs[..., :3] - 0.5
            diff = obs[..., 3:] / 2.0
            obs = torch.cat([rgb, diff], dim=-1)
        x = obs.reshape(-1, *obs.shape[-3:])
        x = x.permute(0, 3, 1, 2)
        # (B*T, 1280)
        x = self.backbone(x)
        # (B*T, embed_dim)
        x = self.adapter(x)
        # (B, T, embed_dim)
        return x.reshape(*obs.shape[:-3], x.shape[-1])


class SafetyNet(nn.Module):
    def __init__(self, in_channels: int = 2, action_dim: int = 4, speed_dim: int = 1, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + action_dim + speed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, speed, action):
        # image: (B, T, H, W, C)
        b, t = image.shape[:2]
        x = image.reshape(b * t, *image.shape[2:]).permute(0, 3, 1, 2)
        feat = self.conv(x)
        speed = speed.reshape(b * t, -1)
        action = action.reshape(b * t, -1)
        return self.head(torch.cat([feat, speed, action], dim=-1)).reshape(b, t, 1)

class ConvDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shape=(3, 64, 64)):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._shape = shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        factor = 2 ** (len(self.depths))
        minres = [int(x // factor) for x in shape[1:]]
        self.min_shape = (*minres, self.depths[-1])
        self.bspace = int(config.bspace)
        self.kernel_size = int(config.kernel_size)
        self.units = int(config.units)
        u, g = math.prod(self.min_shape), self.bspace
        self.sp0 = BlockLinear(deter, u, g)
        self.sp1 = nn.Sequential(
            nn.Linear(flat_stoch, 2 * self.units), nn.RMSNorm(2 * self.units, eps=1e-04, dtype=torch.float32), act()
        )
        self.sp2 = nn.Linear(2 * self.units, math.prod(self.min_shape))
        self.sp_norm = nn.Sequential(nn.RMSNorm(self.depths[-1], eps=1e-04, dtype=torch.float32), act())
        layers = []
        in_dim = self.depths[-1]
        for depth in reversed(self.depths[:-1]):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(Conv2dSamePad(in_dim, depth, self.kernel_size, stride=1, bias=True))
            layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(Conv2dSamePad(in_dim, self._shape[0], self.kernel_size, stride=1, bias=True))
        self.layers = nn.Sequential(*layers)
        self.apply(weight_init_)

    def forward(self, stoch, deter):
        """Decode latent states into images.

        Notes
        -----
        The decoder first constructs a low-resolution spatial feature map from
        the deterministic state (block-linear projection) and from the stochastic
        state (MLP projection), concats them, then upsamples back to the target
        resolution.
        """
        # (B, T, S, K), (B, T, D)
        B_T = deter.shape[:-1]
        # (B*T, D), (B*T, S*K)
        x0, x1 = deter.reshape(B_T.numel(), deter.shape[-1]), stoch.reshape(B_T.numel(), -1)

        # Spatial features from deterministic state
        # (H_feat, W_feat, C_feat)
        H_feat, W_feat, C_feat = self.min_shape
        # (B*T, H_feat*W_feat*C_feat)
        x0 = self.sp0(x0)
        # (B*T, G, H_feat, W_feat, C_feat/G)
        x0 = x0.reshape(-1, self.bspace, H_feat, W_feat, C_feat // self.bspace)
        # (B*T, H_feat, W_feat, C_feat)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, H_feat, W_feat, C_feat)

        # Spatial features from stochastic state
        # (B*T, 2*U)
        x1 = self.sp1(x1)
        # (B*T, H_feat, W_feat, C_feat)
        x1 = self.sp2(x1).reshape(-1, H_feat, W_feat, C_feat)

        # Combine and upsample
        # (B*T, H_feat, W_feat, C_feat)
        x = self.sp_norm(x0 + x1)
        # (B*T, C_feat, H_feat, W_feat)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)  # Upsamples to original H, W
        # (B*T, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = torch.sigmoid(x)
        # (B, T, H, W, C)
        return x.reshape(*B_T, *x.shape[1:])


class MLP(nn.Module):
    def __init__(
        self,
        config,
        inp_dim,
    ):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._symlog_inputs = bool(config.symlog_inputs)
        self._device = torch.device(config.device)
        self.layers = nn.Sequential()
        for i in range(config.layers):
            self.layers.add_module(f"{config.name}_linear{i}", nn.Linear(inp_dim, config.units, bias=True))
            self.layers.add_module(f"{config.name}_norm{i}", nn.RMSNorm(config.units, eps=1e-04, dtype=torch.float32))
            self.layers.add_module(f"{config.name}_act{i}", act())
            inp_dim = config.units
        self.out_dim = config.units

    def forward(self, x):
        # (B, T, I)
        if self._symlog_inputs:
            x = dists.symlog(x)
        # (B, T, U)
        return self.layers(x)


class MLPHead(nn.Module):
    def __init__(self, config, inp_dim):
        super().__init__()
        self.mlp = MLP(config, inp_dim)
        self._dist_name = str(config.dist.name)
        self._outscale = float(config.outscale)
        self._dist = getattr(dists, str(config.dist.name))

        if self._dist_name == "bounded_normal":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0] * 2, bias=True)
            kwargs = {"min_std": float(config.dist.min_std), "max_std": float(config.dist.max_std)}
        elif self._dist_name == "onehot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio)}
        elif self._dist_name == "multi_onehot":
            self.last = nn.Linear(self.mlp.out_dim, sum(config.shape), bias=True)
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio), "shape": tuple(config.shape)}
        elif self._dist_name == "symexp_twohot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {"device": torch.device(config.device), "bin_num": int(config.dist.bin_num)}
        elif self._dist_name in ("binary", "identity"):
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {}
        else:
            raise NotImplementedError

        self._dist = partial(self._dist, **kwargs)

        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        # apply explicit output scaling.
        if self._outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(self._outscale)

    def forward(self, x):
        """Produce a distribution head."""
        # (B, T, F)
        return self._dist(self.last(self.mlp(x)))


class Projector(nn.Module):
    def __init__(self, in_ch1, in_ch2):
        super().__init__()
        self.w = nn.Linear(in_ch1, in_ch2, bias=False)
        self.apply(weight_init_)

    def forward(self, x):
        return self.w(x)


class ReturnEMA(nn.Module):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32, device=self.device))

    def __call__(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        # Using out-of-place update for torch.compile compatibility
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()
