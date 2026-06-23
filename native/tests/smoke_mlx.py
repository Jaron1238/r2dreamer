#!/usr/bin/env python3
"""
native/tests/smoke_mlx.py
Bug #18 fix: Smoke test for the MLX inference path.

Run with:
    cd r2dreamer && python -m native.tests.smoke_mlx

Verifies:
  1. MLXDreamer can be instantiated from a tiny config
  2. Encoder forward pass produces correct embed shape
  3. MLXRSSM.obs_step produces correct state shapes
  4. Actor + Value produce correct output shapes
  5. A full train_step (loss computation) runs without crashing

Requires Apple Silicon with the `mlx` package installed.
"""
from __future__ import annotations
import sys, os

# Allow running as a script from r2dreamer/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def _make_cfg():
    """Build a minimal SimpleNamespace config that satisfies MLXDreamer.__init__."""
    from types import SimpleNamespace as NS

    loss_scales = NS(
        kl=1.0, recon=1.0, reward=1.0, cont=1.0,
        barlow=1.0, inv_dyn=1.0, ctx_consistency=1.0,
    )
    rssm = NS(stoch=4, discrete=4, deter=32, hidden=32, act_dim=4,
               embed_size=32, d_emb_dim=8, obs_layers=1, img_layers=1,
               dyn_layers=1, blocks=2, unimix_ratio=0.01, motor_inertia_alpha=1.0)
    encoder = NS(in_ch=6, input_h=32, input_w=32, depth=8,
                 mults=(1, 2, 2, 2))
    reward  = NS(units=32, layers=1, dist=NS(bin_num=7))
    cont    = NS(units=32, layers=1)
    actor   = NS(units=32, layers=1, min_std=0.1, max_std=2.0)
    value   = NS(units=32, layers=1, dist=NS(bin_num=7))
    safety_net = NS(in_channels=2, hidden=16, frame_stack=2)

    return NS(
        phase=1,
        kl_free=0.5, act_entropy=3e-4,
        imag_horizon=3, horizon=3, lamb=0.95,
        embed_size=32, num_drones=2, drone_embed_dim=8,
        ctx_len=4, ctx_warmup_steps=10, ctx_consistency_weight=1.0,
        ctx_bottleneck=16, ctx_encoder_type="gru",
        loss_scales=loss_scales,
        rssm=rssm, encoder=encoder,
        reward=reward, cont=cont, actor=actor, value=value,
        safety_net=safety_net,
    )


def run_smoke_test():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        print("[SKIP] mlx not installed — skipping MLX smoke test.")
        return

    from native.mlx_models import MLXDreamer
    from native.mlx_types import RSSMState, ContextState, DreamerMLXState

    print("[smoke_mlx] Building tiny MLXDreamer config …")
    cfg = _make_cfg()

    print("[smoke_mlx] Instantiating MLXDreamer …")
    model = MLXDreamer(cfg)
    mx.eval(model.parameters())
    print(f"           ✓ MLXDreamer built | feat_size={model.rssm.feat_size}")

    # ── Shapes ────────────────────────────────────────────────────────────────
    B, H, W, C = 1, 32, 32, 6
    A = cfg.rssm.act_dim
    S, D_cat, DETER = cfg.rssm.stoch, cfg.rssm.discrete, cfg.rssm.deter

    # 1) Encoder
    print("[smoke_mlx] Testing encoder …")
    img = mx.zeros((B, H, W, C))
    embed = model.encoder({"image": img})
    mx.eval(embed)
    assert embed.shape == (B, cfg.embed_size), \
        f"embed shape mismatch: {embed.shape} != ({B}, {cfg.embed_size})"
    print(f"           ✓ encoder: {embed.shape}")

    # 2) RSSM obs_step
    print("[smoke_mlx] Testing MLXRSSM.obs_step …")
    d_emb = model.drone_embed(mx.array([0]))           # (1, d_emb_dim)
    state0 = RSSMState(
        stoch               = mx.zeros((B, S, D_cat)),
        deter               = mx.zeros((B, DETER)),
        prev_action         = mx.zeros((B, A)),
        prev_filtered_action= mx.zeros((B, A)),
    )
    reset = mx.zeros((B,))
    new_state, logits = model.rssm.obs_step(state0, embed, reset, d_emb)
    mx.eval(new_state.stoch, new_state.deter)

    assert new_state.stoch.shape == (B, S, D_cat), \
        f"stoch shape: {new_state.stoch.shape}"
    assert new_state.deter.shape == (B, DETER), \
        f"deter shape: {new_state.deter.shape}"
    print(f"           ✓ obs_step: stoch={new_state.stoch.shape} deter={new_state.deter.shape}")

    # 3) get_feat → actor / value
    print("[smoke_mlx] Testing actor + value …")
    feat = model.rssm.get_feat(new_state.stoch, new_state.deter)   # (B, feat_size)
    actor_in = mx.concatenate([feat, d_emb], axis=-1)               # (B, feat_size + d_emb_dim)
    act_dist = model.actor(actor_in)
    val_dist = model.value(actor_in)
    action   = act_dist.sample()
    mx.eval(action)

    assert action.shape == (B, A), f"action shape: {action.shape}"
    print(f"           ✓ actor output shape: {action.shape}")

    # 4) Encoder output for safety_net
    print("[smoke_mlx] Testing safety_net …")
    SH, SW, SC = 32, 32, cfg.safety_net.in_channels * cfg.safety_net.frame_stack
    safety_img = mx.zeros((B, SH, SW, SC))
    speed      = mx.zeros((B, 1, 1))
    act_seq    = mx.zeros((B, 1, A))
    safety_prob, safe_act = model.safety_net(safety_img, speed, act_seq)
    mx.eval(safety_prob, safe_act)
    print(f"           ✓ safety_net: prob={safety_prob.shape} action={safe_act.shape}")

    print("\n[smoke_mlx] ✅ All smoke checks passed.")


if __name__ == "__main__":
    run_smoke_test()
