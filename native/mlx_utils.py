from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from native.mlx_types import LoaderReport

def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    
    return tensor.detach().cpu().numpy()

def _map_conv2d_weight_torch_to_mlx(weight: np.ndarray) -> np.ndarray:
    

    return np.transpose(weight, (0, 2, 3, 1))

def _map_conv_transpose2d_weight_torch_to_mlx(weight: np.ndarray) -> np.ndarray:
    
    return np.transpose(weight, (1, 2, 3, 0))

def _map_linear_weight_torch_to_mlx(weight: np.ndarray) -> np.ndarray:
    

    return weight

def _strip_prefix(name: str, prefixes: tuple[str, ...]) -> str:
    
    for p in prefixes:
        if name.startswith(p):
            return name[len(p) :]
    return name

def _remap_pytorch_name_to_mlx(name: str, obs_layers: int = 1, img_layers: int = 2) -> str:
    
    
    name = re.sub(
        r"^_deter_net\._dyn_in(\d+)\.",
        lambda m: f"deter_net.in{m.group(1)}.layers.",
        name,
    )

    
    name = re.sub(
        r"^_deter_net\._dyn_hid\.dyn_hid_(\d+)\.",
        lambda m: f"deter_net.hid.layers.{3 * int(m.group(1))}.",
        name,
    )
    name = re.sub(
        r"^_deter_net\._dyn_hid\.norm_(\d+)\.",
        lambda m: f"deter_net.hid.layers.{3 * int(m.group(1)) + 1}.",
        name,
    )

    
    name = re.sub(r"^_deter_net\._dyn_gru\.", "deter_net.gru.", name)

    
    name = re.sub(
        r"^_obs_net\.obs_net_(\d+)\.",
        lambda m: f"obs_net.net.layers.{3 * int(m.group(1))}.",
        name,
    )
    name = re.sub(
        r"^_obs_net\.obs_net_n_(\d+)\.",
        lambda m: f"obs_net.net.layers.{3 * int(m.group(1)) + 1}.",
        name,
    )
    name = re.sub(
        r"^_obs_net\.obs_net_logit\.",
        f"obs_net.net.layers.{3 * obs_layers}.",
        name,
    )

    
    name = re.sub(
        r"^_img_net\.img_net_(\d+)\.",
        lambda m: f"img_net.net.layers.{3 * int(m.group(1))}.",
        name,
    )
    name = re.sub(
        r"^_img_net\.img_net_n_(\d+)\.",
        lambda m: f"img_net.net.layers.{3 * int(m.group(1)) + 1}.",
        name,
    )
    name = re.sub(
        r"^_img_net\.img_net_logit\.",
        f"img_net.net.layers.{3 * img_layers}.",
        name,
    )
    name = re.sub(
        r"^context_encoder\.rnn\.weight_(ih|hh)_l(\d+)",
        lambda m: f"context_encoder.rnn.layers.{m.group(2)}.weight_{m.group(1)}",
        name,
    )
    name = re.sub(
        r"^context_encoder\.rnn\.bias_(ih|hh)_l(\d+)",
        lambda m: f"context_encoder.rnn.layers.{m.group(2)}.bias_{m.group(1)}",
        name,
    )
    return name

def _remap_efficientnet_encoder_name(name: str) -> str:
    
    name = re.sub(r"^encoder\.encoders\.0\.", "", name)
    return name

def _preprocess_transformer_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    

    new_dict = {}
    for key, tensor in state_dict.items():
        if "context_encoder.transformer.layers" in key:
            
            new_key = key.replace("context_encoder.transformer.layers.", "context_encoder.transformer_layers.")

            
            if "self_attn.in_proj_weight" in new_key:
                dim = tensor.shape[0] // 3
                q, k, v = torch.split(tensor, dim, dim=0)
                base = new_key.replace("self_attn.in_proj_weight", "attention")
                new_dict[f"{base}.query_proj.weight"] = q
                new_dict[f"{base}.key_proj.weight"] = k
                new_dict[f"{base}.value_proj.weight"] = v
                continue
            
            
            if "self_attn.in_proj_bias" in new_key:
                dim = tensor.shape[0] // 3
                q, k, v = torch.split(tensor, dim, dim=0)
                base = new_key.replace("self_attn.in_proj_bias", "attention")
                new_dict[f"{base}.query_proj.bias"] = q
                new_dict[f"{base}.key_proj.bias"] = k
                new_dict[f"{base}.value_proj.bias"] = v
                continue

            
            if "self_attn.out_proj" in new_key:
                new_dict[new_key.replace("self_attn.", "attention.")] = tensor
                continue

            
            if "norm1" in new_key:
                new_dict[new_key.replace("norm1", "ln1")] = tensor
                continue
                
            if "norm2" in new_key:
                new_dict[new_key.replace("norm2", "ln2")] = tensor
                continue
            
            
            new_dict[new_key] = tensor
        else:
            new_dict[key] = tensor

    return new_dict

def load_pytorch_to_mlx(
    mlx_params: Mapping[str, Any],
    torch_state_dict: Mapping[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...] = ("module.",),
) -> tuple[dict[str, mx.array], LoaderReport]:
    

    flat_mlx: dict[str, Any] = dict(mlx_params)
    converted: dict[str, mx.array] = {}
    details: dict[str, str] = {}
    loaded = 0
    skipped = 0

    
    torch_state_dict = _preprocess_transformer_weights(torch_state_dict)

    
    stripped_keys =[_strip_prefix(k, prefixes) for k in torch_state_dict]
    _obs_layers = max(
        (int(m.group(1)) + 1 for k in stripped_keys for m in[re.search(r"_obs_net\.obs_net_(\d+)\.", k)] if m),
        default=1,
    )
    _img_layers = max(
        (int(m.group(1)) + 1 for k in stripped_keys for m in[re.search(r"_img_net\.img_net_(\d+)\.", k)] if m),
        default=2,
    )

    
    for raw_name, tensor in torch_state_dict.items():
        name = _strip_prefix(raw_name, prefixes)
        
        
        name = _remap_pytorch_name_to_mlx(name, obs_layers=_obs_layers, img_layers=_img_layers)
        name = _remap_efficientnet_encoder_name(name)
        
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