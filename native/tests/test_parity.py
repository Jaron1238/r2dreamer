#!/usr/bin/env python3
"""PyTorch ↔ MLX parity checks.

These tests are intentionally dependency-aware: on non-Apple CI or minimal
containers without ``numpy``, ``torch`` or ``mlx`` they are reported as skipped
instead of failing during module collection.  Run on an MLX-capable machine with:

    python -m pytest native/tests/test_parity.py
    # or
    python native/tests/test_parity.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def deps():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    mx = pytest.importorskip("mlx.core")
    from networks import BlockLinear as TorchBlockLinear
    from networks import RMSNorm2D
    from native.mlx_models import BlockLinear as MLXBlockLinear
    from native.mlx_models import MLXDreamer, RMSNorm
    from native.mlx_utils import load_pytorch_to_mlx
    from mlx.utils import tree_flatten

    return NS(
        np=np,
        torch=torch,
        mx=mx,
        TorchBlockLinear=TorchBlockLinear,
        RMSNorm2D=RMSNorm2D,
        MLXBlockLinear=MLXBlockLinear,
        MLXDreamer=MLXDreamer,
        RMSNorm=RMSNorm,
        load_pytorch_to_mlx=load_pytorch_to_mlx,
        tree_flatten=tree_flatten,
    )


def set_seeds(deps, seed: int = 42) -> None:
    deps.np.random.seed(seed)
    deps.torch.manual_seed(seed)
    deps.mx.random.seed(seed)


def _tiny_cfg(rep_loss: str = "nedreamer") -> NS:
    rssm = NS(
        stoch=4,
        discrete=4,
        deter=32,
        hidden=32,
        act_dim=4,
        embed_size=1280,
        d_emb_dim=8,
        obs_layers=1,
        img_layers=1,
        dyn_layers=1,
        blocks=2,
        unimix_ratio=0.01,
        motor_inertia_alpha=1.0,
    )
    return NS(
        phase=3,
        kl_free=0.5,
        act_entropy=3e-4,
        imag_horizon=2,
        horizon=3,
        lamb=0.95,
        embed_size=1280,
        rep_loss=rep_loss,
        num_drones=2,
        drone_embed_dim=8,
        ctx_len=4,
        ctx_warmup_steps=10,
        ctx_consistency_weight=1.0,
        ctx_bottleneck=16,
        ctx_encoder_type="gru",
        barlow_lambd=5e-3,
        inv_dyn_loss_weight=1.0,
        loss_scales={
            "dyn": 1.0,
            "rep": 1.0,
            "nedreamer": 1.0,
            "rew": 1.0,
            "con": 1.0,
            "policy": 1.0,
            "value": 1.0,
            "ctx_consistency": 1.0,
            "safety": 1.0,
        },
        rssm=rssm,
        encoder=NS(in_ch=6, input_h=32, input_w=32, depth=8, mults=(1, 2, 2, 2)),
        reward=NS(units=32, layers=1, dist=NS(bin_num=7)),
        cont=NS(units=32, layers=1),
        actor=NS(units=32, layers=1),
        value=NS(units=32, layers=1, dist=NS(bin_num=7)),
        safety_net=NS(in_channels=6, hidden=16, frame_stack=2),
        nedreamer=NS(
            lambd=5e-3,
            use_ema_target=False,
            hidden_dim=32,
            transformer_layers=1,
            transformer_heads=4,
            transformer_dropout=0.0,
            ema_rate=0.99,
        ),
    )


def test_rmsnorm_parity(deps) -> None:
    set_seeds(deps)
    np, torch, mx = deps.np, deps.torch, deps.mx
    RMSNorm2D, RMSNorm = deps.RMSNorm2D, deps.RMSNorm
    dim = 64
    x_np = np.random.randn(2, 10, dim).astype(np.float32)

    torch_norm = RMSNorm2D(dim, eps=1e-4)
    with torch.no_grad():
        torch_norm.weight.fill_(1.0)
    torch_out = torch_norm(torch.from_numpy(x_np)).detach().cpu().numpy()

    mlx_norm = RMSNorm(dim, eps=1e-4)
    mlx_norm.weight = mx.ones((dim,))
    mlx_out = np.array(mlx_norm(mx.array(x_np)))

    np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-5, atol=1e-5)


def test_blocklinear_parity(deps) -> None:
    set_seeds(deps)
    np, torch, mx = deps.np, deps.torch, deps.mx
    TorchBlockLinear, MLXBlockLinear = deps.TorchBlockLinear, deps.MLXBlockLinear
    in_ch, out_ch, blocks = 32, 64, 8
    x_np = np.random.randn(2, 10, in_ch).astype(np.float32)

    torch_layer = TorchBlockLinear(in_ch, out_ch, blocks)
    with torch.no_grad():
        torch_layer.weight.fill_(0.1)
        torch_layer.bias.fill_(0.2)
    torch_out = torch_layer(torch.from_numpy(x_np)).detach().cpu().numpy()

    mlx_layer = MLXBlockLinear(in_ch, out_ch, blocks)
    mlx_layer.weight = mx.full((out_ch // blocks, in_ch // blocks, blocks), 0.1)
    mlx_layer.bias = mx.full((out_ch,), 0.2)
    mlx_out = np.array(mlx_layer(mx.array(x_np)))

    np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-5, atol=1e-5)


def test_nedreamer_mlx_model_exposes_phase3_modules(deps) -> None:
    model = deps.MLXDreamer(_tiny_cfg("nedreamer"))
    assert hasattr(model, "nedreamer_transformer")
    assert hasattr(model, "nedreamer_predictor")
    assert model.rep_loss == "nedreamer"


def test_nedreamer_transformer_weight_preprocessing_maps_qkv(deps) -> None:
    model = deps.MLXDreamer(_tiny_cfg("nedreamer"))
    mlx_params = dict(deps.tree_flatten(model.parameters()))
    hidden = 32
    torch_state = {
        "nedreamer_transformer.encoder.layers.0.self_attn.in_proj_weight": deps.torch.zeros(3 * hidden, hidden),
        "nedreamer_transformer.encoder.layers.0.self_attn.in_proj_bias": deps.torch.zeros(3 * hidden),
    }

    converted, report = deps.load_pytorch_to_mlx(mlx_params, torch_state)

    assert report["skipped"] == 0, report["details"]
    assert "nedreamer_transformer.encoder.layers.0.attention.query_proj.weight" in converted
    assert "nedreamer_transformer.encoder.layers.0.attention.key_proj.weight" in converted
    assert "nedreamer_transformer.encoder.layers.0.attention.value_proj.weight" in converted
    assert "nedreamer_transformer.encoder.layers.0.attention.query_proj.bias" in converted


def main() -> int:
    return pytest.main([__file__])


if __name__ == "__main__":
    raise SystemExit(main())
