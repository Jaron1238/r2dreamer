

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    import coremltools as ct
    _CT_AVAILABLE = True
except ImportError:   
    _CT_AVAILABLE = False

@dataclass
class ExportConfig:
    img_height: int          # RSSM encoder input height
    img_width: int           # RSSM encoder input width
    img_channels: int        # channels of ONE RSSM frame (always 6: RGB+diff)
    action_dim: int
    stoch_flat_dim: int      # stoch * discrete
    deter_dim: int           # GRU hidden size
    drone_id: int = 0        # which drone embedding to bake in
    safety_frame_stack: int = 3
    safety_img_height: int = 0  # SafetyNet input height (0 → falls back to img_height)
    safety_img_width: int = 0   # SafetyNet input width  (0 → falls back to img_width)
    # Channels per single safety frame (1 = grayscale, 3 = RGB, 6 = RGB+diff).
    # If 0, falls back to img_channels (RSSM channels) for backward compat.
    safety_channels_per_frame: int = 0

    @property
    def safety_h(self) -> int:
        return self.safety_img_height if self.safety_img_height > 0 else self.img_height

    @property
    def safety_w(self) -> int:
        return self.safety_img_width if self.safety_img_width > 0 else self.img_width

    @property
    def safety_in_channels(self) -> int:
        ch = self.safety_channels_per_frame if self.safety_channels_per_frame > 0 else self.img_channels
        return ch * self.safety_frame_stack

class CoreMLDreamerWrapper(nn.Module):
    

    def __init__(
        self,
        encoder: nn.Module,
        rssm: nn.Module,
        actor: nn.Module,
        safety_net: nn.Module,
        drone_embed: nn.Module,
        drone_id: int = 0,
    ) -> None:
        super().__init__()
        self.encoder    = encoder
        self.rssm       = rssm
        self.actor      = actor
        self.safety_net = safety_net

        with torch.no_grad():
            d_emb = drone_embed(torch.tensor([drone_id], dtype=torch.long))
        self.register_buffer("d_emb", d_emb)   

    def forward(
        self,
        image: torch.Tensor,               
        safety_image: torch.Tensor,        
        prev_stoch_flat: torch.Tensor,     
        prev_deter: torch.Tensor,          
        prev_action: torch.Tensor,         # Bug #16 fix: separate raw action state
        prev_filtered_act: torch.Tensor,   
        speed: torch.Tensor,               
    ):
        
        embed = self.encoder({"image": image})          

        
        stoch = prev_stoch_flat.reshape(1, self.rssm._stoch, self.rssm._discrete)
        reset = torch.zeros(1, dtype=torch.float32, device=prev_deter.device)
        # Bug #16 fix: was passing prev_filtered_act for BOTH prev_action and
        # prev_filtered_action → filter became identity → frozen at zero for entire flight
        next_stoch, next_deter, _, next_filtered_act = self.rssm.obs_step(
            stoch, prev_deter, prev_action,
            embed, reset, self.d_emb, prev_filtered_act,
        )

        
        feat         = self.rssm.get_feat(next_stoch, next_deter)
        policy_input = torch.cat([feat, self.d_emb], dim=-1)
        raw_out      = self.actor.last(self.actor.mlp(policy_input))  
        mean, _      = torch.chunk(raw_out, 2, dim=-1)
        action       = torch.tanh(mean)                               

        
        
        safety_img_t = safety_image.unsqueeze(0)
        safety_prob, safe_action_raw = self.safety_net(
            safety_img_t,
            speed.reshape(1, 1, 1),
            action.reshape(1, 1, -1),
        )
        
        
        
        safety_score = torch.sigmoid(safety_prob).reshape(1)  
        safe_action  = safe_action_raw.reshape(1, -1)         

        next_stoch_flat = next_stoch.reshape(1, -1)    
        # Bug #16 fix: return action as next_action so caller can feed it back as prev_action
        return action, next_stoch_flat, next_deter, next_filtered_act, action, safety_score, safe_action

def export_to_coreml(
    dreamer,
    cfg: ExportConfig,
    out_path: str | pathlib.Path,
) -> pathlib.Path:
    
    if not _CT_AVAILABLE:
        raise ImportError(
            "coremltools is not installed.  Run: pip install coremltools"
        )

    wrapper = CoreMLDreamerWrapper(
        encoder=dreamer.encoder,
        rssm=dreamer.rssm,
        actor=dreamer.actor,
        safety_net=dreamer.safety_net,
        drone_embed=dreamer.drone_embed,
        drone_id=cfg.drone_id,
    ).eval()

    H, W, C  = cfg.img_height, cfg.img_width, cfg.img_channels
    SH, SW   = cfg.safety_h, cfg.safety_w       # SafetyNet resolution (may differ from RSSM)
    SC       = cfg.safety_in_channels   # img_channels * safety_frame_stack
    A, SF, D = cfg.action_dim, cfg.stoch_flat_dim, cfg.deter_dim

    dummy_inputs = (
        torch.zeros(1, H,  W,  C,  dtype=torch.float32),   # image  (RSSM)
        torch.zeros(1, SH, SW, SC, dtype=torch.float32),   # safety_image (SafetyNet, stacked)
        torch.zeros(1, SF,       dtype=torch.float32),   
        torch.zeros(1, D,        dtype=torch.float32),   
        torch.zeros(1, A,        dtype=torch.float32),   # prev_action  (Bug #16 fix)
        torch.zeros(1, A,        dtype=torch.float32),   # prev_filtered_act
        torch.zeros(1, 1,        dtype=torch.float32),   # speed
    )

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_inputs, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="image",             shape=(1, H,  W,  C),  dtype=float),
            ct.TensorType(name="safety_image",       shape=(1, SH, SW, SC), dtype=float),
            ct.TensorType(name="prev_stoch_flat",   shape=(1, SF),        dtype=float),
            ct.TensorType(name="prev_deter",        shape=(1, D),         dtype=float),
            ct.TensorType(name="prev_action",       shape=(1, A),         dtype=float),  # Bug #16 fix
            ct.TensorType(name="prev_filtered_act", shape=(1, A),         dtype=float),
            ct.TensorType(name="speed",             shape=(1, 1),         dtype=float),
        ],
        outputs=[
            ct.TensorType(name="action"),
            ct.TensorType(name="next_stoch_flat"),
            ct.TensorType(name="next_deter"),
            ct.TensorType(name="next_filtered_act"),
            ct.TensorType(name="next_action"),  # Bug #16 fix: fed back as prev_action next call
            ct.TensorType(name="safety_score"),
            ct.TensorType(name="safe_action"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )

    
    op_cfg  = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8"
    )
    opt_cfg = ct.optimize.coreml.OptimizationConfig(global_config=op_cfg)
    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=opt_cfg)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(
        f"[coreml_export] Saved -> {out_path}  "
        f"(safety_frame_stack={cfg.safety_frame_stack}, "
        f"safety_in_channels={SC})"
    )
    return out_path
