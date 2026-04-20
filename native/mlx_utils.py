from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from native.mlx_types import LoaderReport


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _map_conv2d_weight_torch_to_mlx(weight: np.ndarray) -> np.ndarray:
    # torch: (out_ch, in_ch, k_h, k_w) -> mlx(nn.Conv2d NHWC kernel): (out_ch, k_h, k_w, in_ch)
    return np.transpose(weight, (0, 2, 3, 1))


def _map_conv_transpose2d_weight_torch_to_mlx(weight: np.ndarray) -> np.ndarray:
    # torch conv_transpose2d: (in_ch, out_ch, k_h, k_w) -> mlx expected NHWC kernel-like layout
    return np.transpose(weight, (1, 2, 3, 0))


def _map_linear_weight_torch_to_mlx(weight: np.ndarray) -> np.ndarray:
    # torch linear: (out, in), mlx linear matmul uses (in, out)
    return np.transpose(weight, (1, 0))


def _strip_prefix(name: str, prefixes: tuple[str, ...]) -> str:
    for p in prefixes:
        if name.startswith(p):
            return name[len(p) :]
    return name


def load_pytorch_to_mlx(
    mlx_params: Mapping[str, Any],
    torch_state_dict: Mapping[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...] = ("module.",),
) -> tuple[dict[str, mx.array], LoaderReport]:
    """Convert a PyTorch state_dict into an MLX parameter tree.

    - Conv2d kernels: NCHW -> NHWC kernel layout
    - ConvTranspose2d kernels: torch transpose layout -> NHWC layout
    - Linear kernels: transposed
    - BatchNorm running buffers are included when matching keys exist.
    """

    flat_mlx: dict[str, Any] = dict(mlx_params)
    converted: dict[str, mx.array] = {}
    details: dict[str, str] = {}
    loaded = 0
    skipped = 0

    for raw_name, tensor in torch_state_dict.items():
        name = _strip_prefix(raw_name, prefixes)
        if name not in flat_mlx:
            skipped += 1
            details[name] = "missing_in_mlx"
            continue

        arr = _to_numpy(tensor)
        target_shape = tuple(flat_mlx[name].shape)

        if arr.ndim == 4 and "conv_transpose" in name:
            arr = _map_conv_transpose2d_weight_torch_to_mlx(arr)
        elif arr.ndim == 4:
            arr = _map_conv2d_weight_torch_to_mlx(arr)
        elif arr.ndim == 2 and name.endswith("weight"):
            arr = _map_linear_weight_torch_to_mlx(arr)

        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            # BatchNorm buffers can be float/bool/int in torch; cast to target dtype.
            pass

        if tuple(arr.shape) != target_shape:
            skipped += 1
            details[name] = f"shape_mismatch torch={tuple(arr.shape)} mlx={target_shape}"
            continue

        converted[name] = mx.array(arr)
        loaded += 1

    missing = sum(1 for k in flat_mlx if k not in converted)
    report: LoaderReport = {
        "loaded": loaded,
        "skipped": skipped,
        "missing": missing,
        "details": details,
    }
    return converted, report
