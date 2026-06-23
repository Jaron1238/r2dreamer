import math
import re
from functools import partial
import torchvision.models as tv_models
import torch
import torch.nn.functional as F
from torch import nn

import distributions as dists
from tools import weight_init_

def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    
    if mask is None:
        return torch.mean(values)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=values.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return torch.sum(values * mask) / denom

class LambdaLayer(nn.Module):
    

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BlockLinear(nn.Module):
    

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)

        
        
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_shape = x.shape[:-1]
        
        
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)

        
        
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        
        
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias

class Conv2dSamePad(nn.Conv2d):
    

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
    

    def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
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
        
        
        dists = {}
        if self.cnn_shapes:
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            
            outputs = self._cnn(stoch, deter)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._image_dist(output) for key, output in zip(self.cnn_shapes.keys(), outputs)})
        if self.mlp_shapes:
            split_sizes = [v[0] for v in self.mlp_shapes.values()]
            
            feat = torch.cat([stoch.reshape(*deter.shape[:-1], -1), deter], -1)
            outputs = self._mlp(feat)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._mlp_dist(output) for key, output in zip(self.mlp_shapes.keys(), outputs)})
        return dists

class SpatialAttentionPool(nn.Module):
    """
    FPN-fused cross-attention pooling for 16:9 FPV imagery → (B, embed_dim).

    Fixes vs. the vanilla version
    ──────────────────────────────────────────────────────────────────────────
    Problem 1 – Positionsblindheit:
      Standard-Attention ist permutationsinvariant — das Modell sah das Bild
      als „Sack voller Pixel" ohne Ortskenntnis.
      Lösung: lernbares 2D-Positionsgitter (nn.Parameter), warm-gestartet
      von der sinusoidalen 2D-PE, sodass Geometrie ab Schritt 0 bedeutsam
      ist.  Statische Größe → ONNX-sichere Konstante beim Export.

    Problem 2 – Slot-Collapse (Symmetrie):
      4 Queries nahe Null konvergierten im Training auf denselben Fokuspunkt
      (meist Bildmitte) → effektiv nur 1 aktiver Slot.
      Lösung: orthogonale Initialisierung (nn.init.orthogonal_).  9 Vektoren
      in 192-dim Raum können exakt orthogonal bleiben; kein Auxiliary-Loss
      nötig.

    3 × 3 Query-Grid (9 Queries):
      • Dedizierte Center-Query für den Fluchtpunkt beim Vorwärtsflug.
      • 8 Perimeter-Queries sichern seitliche, obere und untere Grenzen ab.
      Im Gegensatz zu 2 × 2 hat das 3 × 3 Gitter einen echten Mittelpunkt.

    ONNX-Sicherheit:
      • nn.MultiheadAttention (TensorRT fused kernel).
      • Kein dynamisches Slicing [:H] im forward-Pfad.
      • Kein .expand() — stattdessen .repeat() mit statischem B.
      • pos_embed als statische nn.Parameter-Konstante beim Export.
    """

    def __init__(
        self,
        backbone_ch: int,        # Kanäle der FPN-fusionierten Feature-Map (1344)
        embed_dim: int,
        num_queries: int = 9,    # 3×3 Grid
        num_heads: int = 4,
        feat_h: int = 9,         # Höhe der tiefen Feature-Map (H/32)
        feat_w: int = 16,        # Breite der tiefen Feature-Map (W/32)
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_queries = num_queries

        # Shared K/V-Projektion: fused_ch → embed_dim
        self.in_proj = nn.Linear(backbone_ch, embed_dim, bias=False)

        # 2D lernbares Positionsgitter — warm-gestartet von sinusoidaler PE.
        # Als nn.Parameter: wird trainiert, verhält sich beim ONNX-Export wie
        # eine schnelle Konstante (kein dynamischer Shape-Op).
        self.pos_embed = nn.Parameter(
            self._make_2d_sinusoidal(feat_h, feat_w, embed_dim)
        )  # (1, feat_h*feat_w, embed_dim)

        # 9 Slot-Queries, orthogonal initialisiert → kein Slot-Collapse möglich.
        # nn.init.orthogonal_ auf (Q, D) liefert Q orthogonale D-dim Zeilenvektoren.
        self.queries = nn.Parameter(torch.empty(num_queries, embed_dim))
        nn.init.orthogonal_(self.queries)

        # Standard MHA — ONNX/TensorRT fused kernel; need_weights=False
        # überspringt O(T²)-Gewichts-Materialisierung → schneller bei 50 Hz.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=False)

        self.norm = nn.RMSNorm(embed_dim, eps=1e-4, dtype=torch.float32)

        # 9 Slots → einzelner embed_dim-Vektor; Linear lernt Slot-Interaktionen.
        self.out = nn.Linear(num_queries * embed_dim, embed_dim, bias=False)

    @staticmethod
    def _make_2d_sinusoidal(h: int, w: int, d: int) -> torch.Tensor:
        """
        2D-sinusoidale Positions-Enkodierung als Warm-Start für das lernbare Gitter.

        Erste d//2 Dims enkodieren die Zeilenposition (H),
        zweite d//2 Dims enkodieren die Spaltenposition (W).
        Gibt (1, h*w, d) mit Skalierung 0.1 zurück, damit das Training
        die Werte frei anpassen kann.
        """
        d_model  = d // 2
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / d_model)
        )
        # Zeilen-PE: (h, d//2)
        pe_h     = torch.zeros(h, d_model)
        row_pos  = torch.arange(h, dtype=torch.float32).unsqueeze(1)
        pe_h[:, 0::2] = torch.sin(row_pos * div_term)
        pe_h[:, 1::2] = torch.cos(row_pos * div_term)

        # Spalten-PE: (w, d//2)
        pe_w     = torch.zeros(w, d_model)
        col_pos  = torch.arange(w, dtype=torch.float32).unsqueeze(1)
        pe_w[:, 0::2] = torch.sin(col_pos * div_term)
        pe_w[:, 1::2] = torch.cos(col_pos * div_term)

        # Auf 2D-Gitter aufspannen und zusammenfügen: (h, w, d)
        pe = torch.cat(
            [
                pe_h.unsqueeze(1).expand(h, w, d_model),
                pe_w.unsqueeze(0).expand(h, w, d_model),
            ],
            dim=-1,
        )
        return pe.reshape(1, h * w, d).mul_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # (B, C, H', W') → (B, H'W', C) Token-Sequenz
        tokens = x.flatten(2).permute(0, 2, 1)     # (B, n_tokens, backbone_ch)
        tokens = self.in_proj(tokens)               # (B, n_tokens, embed_dim)
        # Positions-Encoding addieren (broadcast über B)
        tokens = tokens + self.pos_embed            # (B, n_tokens, embed_dim)

        # ONNX-sicher: .repeat() statt .expand() — vermeidet dynamische Shape-Nodes
        q = self.queries.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, embed_dim)

        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        out    = self.norm(out)
        return self.out(out.reshape(B, -1))         # (B, embed_dim)


class ConvEncoder(nn.Module):
    """
    EfficientNet-V2-S Backbone mit FPN-light ("Kabel-Killer") und
    SpatialAttentionPool (9-Query 3×3 Grid).

    FPN-light:
      Stage 3  (features[:4])  → (B, 64, H/8, W/8):  geometrische Kanten,
        dünne Äste und Stromleitungen sind hier noch scharf.
      Stage 7  (features[4:])  → (B, 1280, H/32, W/32): semantischer Kontext.
      Fusion:  Stage-3 Merkmale werden auf H/32 × W/32 gepooled und
        entlang der Kanal-Dim konkateniert → 1344 Kanäle.

    Für 512 × 288 (16:9 Landscape):
      feat_h = 9, feat_w = 16 (exakt 16:9 → kein Zuschnitt / Verzerrung).
      Early: (B, 64, 36, 64).  Deep + fused: (B, 1344, 9, 16).
    """

    def __init__(self, config, input_shape):
        super().__init__()
        h, w, input_ch = input_shape
        self.input_ch = int(input_ch)

        # ── Backbone: EfficientNet-V2-S (20M params, Fused-MBConv stages) ────
        backbone = tv_models.efficientnet_v2_s(weights=tv_models.EfficientNet_V2_S_Weights.DEFAULT)

        # Ersten Conv2d für beliebige Eingangskanäle patchen.
        # out_channels dynamisch lesen → bleibt korrekt falls Backbone getauscht wird.
        old_conv = backbone.features[0][0]   # Conv2d(3, 24, 3, stride=2) für V2-S
        out_ch   = old_conv.out_channels     # 24

        backbone.features[0][0] = nn.Conv2d(
            in_channels=self.input_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        # Vortrainierte RGB-Filter kopieren; Extra-Kanäle mit Filter-Mittel initialisieren
        # damit alle Kanäle gleichwertig starten.
        with torch.no_grad():
            if self.input_ch <= 3:
                backbone.features[0][0].weight[:, :self.input_ch].copy_(
                    old_conv.weight[:, :self.input_ch]
                )
            else:
                backbone.features[0][0].weight[:, :3].copy_(old_conv.weight[:, :3])
                mean_filter = old_conv.weight[:, :3].mean(dim=1, keepdim=True)
                for c in range(3, self.input_ch):
                    backbone.features[0][0].weight[:, c : c + 1].copy_(mean_filter)

        for p in backbone.features.parameters():
            p.requires_grad = True

        # ── FPN-light: Backbone NACH dem Patch aufteilen ──────────────────────
        # features[:4]  →  H/8  × W/8  (geometrische Früh-Features, C=64)
        # features[4:]  →  H/32 × W/32 (semantische Tief-Features, C=1280)
        self.backbone_early = backbone.features[:4]
        self.backbone_late  = backbone.features[4:]

        # Shapes per Dummy-Forward ermitteln (device-agnostisch, kein Gewichts-Download)
        with torch.no_grad():
            dummy      = torch.zeros(1, self.input_ch, int(h), int(w))
            early_feat = self.backbone_early(dummy)
            deep_feat  = self.backbone_late(early_feat)

        early_ch    = int(early_feat.shape[1])   # 64
        deep_ch     = int(deep_feat.shape[1])    # 1280
        self.feat_h = int(deep_feat.shape[2])    # H/32
        self.feat_w = int(deep_feat.shape[3])    # W/32
        fused_ch    = early_ch + deep_ch         # 1344

        # Früh-Feature-Pfad: 1×1 Conv + Norm + Aktivierung.
        # RMSNorm2D gleicht die unterschiedlichen Feature-Skalen zwischen
        # frühen (BN-normiert nach Stage 3) und tiefen Merkmalen an.
        self.fpn_reduce = nn.Sequential(
            nn.Conv2d(early_ch, early_ch, kernel_size=1, bias=False),
            RMSNorm2D(early_ch, eps=1e-4),
            nn.SiLU(),
        )

        # ── Attention-Pooling ─────────────────────────────────────────────────
        embed_dim   = int(config.depth) * int(config.mults[-1])
        num_queries = int(getattr(config, "attn_pool_queries", 9))   # neu: 9 statt 4
        num_heads   = int(getattr(config, "attn_pool_heads",   4))

        self.pool    = SpatialAttentionPool(
            fused_ch, embed_dim, num_queries, num_heads,
            self.feat_h, self.feat_w,
        )
        self.out_dim = embed_dim

    def forward(self, obs):
        channels = obs.shape[-1]
        if channels <= 3:
            obs = obs - 0.5
        else:
            rgb  = obs[..., :3] - 0.5
            diff = obs[..., 3:] / 2.0
            obs  = torch.cat([rgb, diff], dim=-1)

        x = obs.reshape(-1, *obs.shape[-3:])
        x = x.permute(0, 3, 1, 2)          # (B, C, H, W)

        # ── FPN-light Forward ──────────────────────────────────────────────────
        early = self.backbone_early(x)      # (B, 64,   H/8,  W/8)
        deep  = self.backbone_late(early)   # (B, 1280, H/32, W/32)

        # Früh-Features komprimieren + auf tiefe Auflösung poolien.
        # (self.feat_h, self.feat_w) sind statische Attribute → ONNX-sicher.
        early_proj = self.fpn_reduce(early)                               # (B, 64, H/8, W/8)
        early_down = F.adaptive_avg_pool2d(
            early_proj, (self.feat_h, self.feat_w)                        # statische Größe
        )                                                                  # (B, 64, H/32, W/32)

        fused = torch.cat([deep, early_down], dim=1)                      # (B, 1344, H/32, W/32)
        x     = self.pool(fused)                                          # (B, embed_dim)
        return x.reshape(*obs.shape[:-3], x.shape[-1])

class CausalTemporalTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_len: int,
        action_proj_dim: int = 32,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, action_proj_dim)
        self.in_proj = nn.Linear(state_dim + action_proj_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, state_seq: torch.Tensor, prev_action_seq: torch.Tensor) -> torch.Tensor:
        
        x = torch.cat([state_seq, self.action_proj(prev_action_seq)], dim=-1)
        x = self.in_proj(x)
        x = x + self.pos_embed[:, : x.shape[1]]
        t = x.shape[1]
        causal_mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        return self.encoder(x, mask=causal_mask)

class NEPredictorHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DepthAuxHead(nn.Module):
    def __init__(self, in_dim: int, out_h: int, out_w: int):
        super().__init__()
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        hidden = max(128, in_dim // 2)
        base_h = max(4, self.out_h // 16)
        base_w = max(4, self.out_w // 16)
        self.base_h = base_h
        self.base_w = base_w
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 64 * base_h * base_w),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, post_feat: torch.Tensor) -> torch.Tensor:
        
        b, t, _ = post_feat.shape
        x = self.proj(post_feat).reshape(b * t, 64, self.base_h, self.base_w)
        pred = self.decoder(x)
        pred = F.interpolate(pred, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        pred = pred.reshape(b, t, self.out_h, self.out_w, 1)
        return F.softplus(pred)

class _ResBlock(nn.Module):
    

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))

class SafetyNet(nn.Module):
    

    def __init__(
        self,
        in_channels: int = 1,
        action_dim: int = 4,
        speed_dim: int = 1,
        hidden: int = 64,
        frame_stack: int = 3,
    ):
        super().__init__()
        self.frame_stack = frame_stack
        self.action_dim  = action_dim
        stacked_channels = in_channels * frame_stack

        
        
        
        self.layer1 = _ResBlock(stacked_channels, 16,     stride=1)  
        self.layer2 = _ResBlock(16,               16,     stride=2)  
        self.layer3 = _ResBlock(16,               32,     stride=2)  
        self.layer4 = _ResBlock(32,               64,     stride=2)  
        self.layer5 = _ResBlock(64,               hidden, stride=2)  

        
        
        self.pool    = nn.AdaptiveMaxPool2d((5, 5))
        self.flatten = nn.Flatten()

        feat_in = hidden * 25 + action_dim + speed_dim

        
        
        
        
        
        self.prob_head = nn.Sequential(
            nn.Linear(feat_in, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            
        )

        
        self.safe_action_head = nn.Sequential(
            nn.Linear(feat_in, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        image: torch.Tensor,   
        speed: torch.Tensor,   
        action: torch.Tensor,  
    ):
        b, t = image.shape[:2]
        x = image.reshape(b * t, *image.shape[2:]).permute(0, 3, 1, 2).contiguous()

        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        
        feat = self.flatten(self.pool(x))

        speed_flat  = speed.reshape(b * t, -1)
        action_flat = action.reshape(b * t, -1)
        combined    = torch.cat([feat, speed_flat, action_flat], dim=-1)

        prob_logit  = self.prob_head(combined).reshape(b, t, 1)
        safe_action = self.safe_action_head(combined).reshape(b, t, self.action_dim)
        return prob_logit, safe_action

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
        

        
        B_T = deter.shape[:-1]
        
        x0, x1 = deter.reshape(B_T.numel(), deter.shape[-1]), stoch.reshape(B_T.numel(), -1)

        
        
        H_feat, W_feat, C_feat = self.min_shape
        
        x0 = self.sp0(x0)
        
        x0 = x0.reshape(-1, self.bspace, H_feat, W_feat, C_feat // self.bspace)
        
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, H_feat, W_feat, C_feat)

        
        
        x1 = self.sp1(x1)
        
        x1 = self.sp2(x1).reshape(-1, H_feat, W_feat, C_feat)

        
        
        x = self.sp_norm(x0 + x1)
        
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)  
        
        x = x.permute(0, 2, 3, 1)
        x = torch.sigmoid(x)
        
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
        
        if self._symlog_inputs:
            x = dists.symlog(x)
        
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
        
        if self._outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(self._outscale)

    def forward(self, x):
        
        
        return self._dist(self.last(self.mlp(x)))

class ContextEncoder(nn.Module):
    

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
            nn.RMSNorm(self.bottleneck, eps=1e-4, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(self.bottleneck, self.bottleneck, bias=True),
            nn.RMSNorm(self.bottleneck, eps=1e-4, dtype=torch.float32),
            nn.SiLU(),
        )

        
        if self.encoder_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.bottleneck,
                hidden_size=self.bottleneck,
                num_layers=2,
                batch_first=True,
                dropout=0.0,
            )
        elif self.encoder_type == "transformer":
            
            half_dim = self.bottleneck // 2
            rope_freqs = 1.0 / (10_000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
            self.register_buffer("rope_freqs", rope_freqs, persistent=False)

            num_heads = max(1, self.bottleneck // 64)  
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.bottleneck,
                nhead=num_heads,
                dim_feedforward=self.bottleneck * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,          
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            raise ValueError(f"ContextEncoder: unbekannter encoder_type='{encoder_type}'. "
                             f"Erlaubt: 'gru', 'transformer'.")

        
        self.temporal_attn = nn.Linear(self.bottleneck, 1, bias=True)

        
        self.out_head = nn.Linear(self.bottleneck, self.out_dim, bias=True)

        self.apply(weight_init_)

    
    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        

        B, T, D = x.shape
        half = D // 2

        
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        angles = torch.outer(positions, self.rope_freqs.to(x.dtype))  
        cos = angles.cos().unsqueeze(0)   
        sin = angles.sin().unsqueeze(0)   

        x1 = x[..., :half]               
        x2 = x[..., half : half * 2]     
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        
        if D % 2 != 0:
            x_rot = torch.cat([x_rot, x[..., -1:]], dim=-1)
        return x_rot

    def forward(
        self,
        flat_stoch: torch.Tensor,            
        deter: torch.Tensor,                 
        action: torch.Tensor,                
        valid_len: torch.Tensor | None = None,  
    ) -> torch.Tensor:
        

        B, T, _ = flat_stoch.shape

        
        
        
        padding_mask: torch.Tensor | None = None
        if valid_len is not None:
            
            positions  = torch.arange(T, device=flat_stoch.device).unsqueeze(0)  
            valid_start = (T - valid_len.clamp(max=T)).unsqueeze(1)               
            padding_mask = positions < valid_start                                 

        
        x = torch.cat([flat_stoch, deter, action], dim=-1)   
        x = self.proj(x)                                      

        
        if self.encoder_type == "gru":
            # Bug #15 fix: zero out leading padded positions before the GRU so that
            # padding-contaminated states don't bleed into valid hidden states.
            # The temporal-attention pool below already masks via padding_mask, but
            # zeroing the inputs makes the GRU path consistent with the transformer path.
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            x, _ = self.rnn(x)                               
        else:  
            x = self._apply_rope(x)                          
            
            
            x = self.transformer(x, src_key_padding_mask=padding_mask)  

        
        
        attn_logits = self.temporal_attn(x)                  
        if padding_mask is not None:
            attn_logits = attn_logits.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        attn_w = torch.softmax(attn_logits, dim=1)
        ctx    = (attn_w * x).sum(dim=1)                     

        
        return self.out_head(ctx)                             

class InverseDynamicsHead(nn.Module):
    

    def __init__(self, feat_dim: int, d_emb_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = int(feat_dim) * 2 + int(d_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, int(act_dim)),
            nn.Tanh(),
        )
        self.apply(weight_init_)

    def forward(
        self,
        feat_t:  torch.Tensor,   
        feat_t1: torch.Tensor,   
        d_emb:   torch.Tensor,   
    ) -> torch.Tensor:
        
        x = torch.cat([feat_t, feat_t1, d_emb], dim=-1)
        return self.net(x)

class Projector(nn.Module):
    def __init__(self, in_ch1, in_ch2):
        super().__init__()
        self.w = nn.Linear(in_ch1, in_ch2, bias=False)
        self.apply(weight_init_)

    def forward(self, x):
        return self.w(x)

class ReturnEMA(nn.Module):
    

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32, device=self.device))

    def __call__(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()
