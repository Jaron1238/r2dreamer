from __future__ import annotations

import pathlib
from dataclasses import dataclass

import coremltools as ct
import torch
import torch.nn as nn


@dataclass
class ExportConfig:
    img_height: int
    img_width: int
    img_channels: int
    action_dim: int
    state_dim: int


class CoreMLInferenceWrapper(nn.Module):
    """Static forward wrapper for deterministic Core ML tracing/export."""

    def __init__(self, policy_backbone: nn.Module, action_head: nn.Module):
        super().__init__()
        self.policy_backbone = policy_backbone
        self.action_head = action_head

    def forward(self, image: torch.Tensor, speed: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Static graph only: fixed sequence of ops, no data-dependent branches.
        flat_image = image.flatten(start_dim=1)
        flat_speed = speed.flatten(start_dim=1)
        x = torch.cat([flat_image, flat_speed, state], dim=-1)
        feat = self.policy_backbone(x)
        action = self.action_head(feat)
        return torch.tanh(action)


def export_to_coreml(
    wrapper: CoreMLInferenceWrapper,
    cfg: ExportConfig,
    out_path: str | pathlib.Path,
) -> pathlib.Path:
    wrapper = wrapper.eval()
    traced = torch.jit.trace(
        wrapper,
        (
            torch.zeros(1, cfg.img_height, cfg.img_width, cfg.img_channels, dtype=torch.float32),
            torch.zeros(1, 1, dtype=torch.float32),
            torch.zeros(1, cfg.state_dim, dtype=torch.float32),
        ),
        strict=True,
    )

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="image", shape=(1, cfg.img_height, cfg.img_width, cfg.img_channels), dtype=float),
            ct.TensorType(name="speed", shape=(1, 1), dtype=float),
            ct.TensorType(name="state", shape=(1, cfg.state_dim), dtype=float),
        ],
        outputs=[ct.TensorType(name="action")],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )

    op_cfg = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    opt_cfg = ct.optimize.coreml.OptimizationConfig(global_config=op_cfg)
    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=opt_cfg)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    return out_path
