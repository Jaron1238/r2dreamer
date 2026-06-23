from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn

from nightly.cross_attn import CrossAttnMLP, KVCache

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import distributions as dists

def _weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class _CrossAttnHead(nn.Module):
    

    def __init__(
        self,
        inp_dim: int,
        units: int,
        n_layers: int,
        text_dim: int,
        out_dim: int,
        dist_fn: Any,
        num_heads: int = 4,
        outscale: float = 1.0,
        symlog_inputs: bool = False,
    ) -> None:
        super().__init__()

        self.mlp = CrossAttnMLP(
            inp_dim=inp_dim,
            units=units,
            n_layers=n_layers,
            text_dim=text_dim,
            num_heads=num_heads,
            symlog_inputs=symlog_inputs,
        )
        self.last = nn.Linear(units, out_dim, bias=True)
        self._dist_fn = dist_fn

        self.mlp.apply(_weight_init)
        self.last.apply(_weight_init)
        if outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(outscale)

    def precompute_kv(
        self,
        z_tokens: torch.Tensor,         
        key_padding_mask: torch.Tensor, 
    ) -> KVCache:
        
        return self.mlp.precompute_kv(z_tokens, key_padding_mask)

    def forward(
        self,
        x: torch.Tensor,                
        z_tokens: torch.Tensor,         
        key_padding_mask: torch.Tensor, 
    ):
        
        h = self.mlp(x, z_tokens, key_padding_mask)
        return self._dist_fn(self.last(h))

    def forward_cached(
        self,
        x: torch.Tensor,  
        cache: KVCache,
    ):
        
        h = self.mlp.forward_cached(x, cache)
        return self._dist_fn(self.last(h))

class VLAActorHead(_CrossAttnHead):
    def __init__(
        self,
        inp_dim: int,
        act_dim: int,
        text_dim: int,
        units: int = 512,
        n_layers: int = 4,
        num_heads: int = 4,
        min_std: float = 0.1,
        max_std: float = 1.0,
        outscale: float = 1.0,
    ) -> None:
        dist_fn = partial(dists.bounded_normal, min_std=min_std, max_std=max_std)
        super().__init__(
            inp_dim=inp_dim,
            units=units,
            n_layers=n_layers,
            text_dim=text_dim,
            out_dim=act_dim * 2,
            dist_fn=dist_fn,
            num_heads=num_heads,
            outscale=outscale,
        )

class VLARewardHead(_CrossAttnHead):
    def __init__(
        self,
        inp_dim: int,
        text_dim: int,
        units: int = 512,
        n_layers: int = 4,
        num_heads: int = 4,
        bin_num: int = 255,
        device: str = "cpu",
        outscale: float = 1.0,
    ) -> None:
        dist_fn = partial(
            dists.symexp_twohot,
            device=torch.device(device),
            bin_num=bin_num,
        )
        super().__init__(
            inp_dim=inp_dim,
            units=units,
            n_layers=n_layers,
            text_dim=text_dim,
            out_dim=bin_num,
            dist_fn=dist_fn,
            num_heads=num_heads,
            outscale=outscale,
            symlog_inputs=True,
        )

class VLAValueHead(_CrossAttnHead):
    def __init__(
        self,
        inp_dim: int,
        text_dim: int,
        units: int = 512,
        n_layers: int = 4,
        num_heads: int = 4,
        bin_num: int = 255,
        device: str = "cpu",
        outscale: float = 0.0,
    ) -> None:
        dist_fn = partial(
            dists.symexp_twohot,
            device=torch.device(device),
            bin_num=bin_num,
        )
        super().__init__(
            inp_dim=inp_dim,
            units=units,
            n_layers=n_layers,
            text_dim=text_dim,
            out_dim=bin_num,
            dist_fn=dist_fn,
            num_heads=num_heads,
            outscale=outscale,
            symlog_inputs=True,
        )

def build_vla_heads(
    cfg,
    actor_input_dim: int,
    reward_input_dim: int,
    value_input_dim: int,
    text_dim: int,
) -> tuple[VLAActorHead, VLARewardHead, VLAValueHead]:

    actor = VLAActorHead(
        inp_dim=actor_input_dim,
        act_dim=cfg.actor.shape[0],
        text_dim=text_dim,
        units=int(cfg.actor.units),
        n_layers=int(cfg.actor.layers),
        num_heads=int(getattr(cfg.actor, "num_heads", 4)),
        min_std=float(cfg.actor.dist.min_std),
        max_std=float(cfg.actor.dist.max_std),
        outscale=float(cfg.actor.outscale),
    )
    reward = VLARewardHead(
        inp_dim=reward_input_dim,
        text_dim=text_dim,
        units=int(cfg.reward.units),
        n_layers=int(cfg.reward.layers),
        num_heads=int(getattr(cfg.reward, "num_heads", 4)),
        bin_num=int(cfg.reward.dist.bin_num),
        device=str(cfg.reward.device),
        outscale=float(cfg.reward.outscale),
    )
    value = VLAValueHead(
        inp_dim=value_input_dim,
        text_dim=text_dim,
        units=int(cfg.critic.units),
        n_layers=int(cfg.critic.layers),
        num_heads=int(getattr(cfg.critic, "num_heads", 4)),
        bin_num=int(cfg.critic.dist.bin_num),
        device=str(cfg.critic.device),
        outscale=float(cfg.critic.outscale),
    )
    return actor, reward, value

class TextEncoder(nn.Module):
    

    _DEFAULT_MODEL = "all-MiniLM-L6-v2"

    _EMBED_DIMS: dict[str, int] = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
    }

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_length: int = 32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._max_length = max_length
        self._device = torch.device(device)
        self._hf_model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._hf_model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the VLA text encoder.\n"
                "Install with:  pip install transformers"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._hf_model = AutoModel.from_pretrained(self._model_name)
        self._hf_model.eval()
        self._hf_model.to(self._device)

    @property
    def embed_dim(self) -> int:
        return self._EMBED_DIMS.get(self._model_name, 384)

    @property
    def max_length(self) -> int:
        return self._max_length

    @torch.no_grad()
    def encode(self, command: str) -> tuple[torch.Tensor, torch.Tensor]:
        

        self._ensure_loaded()

        inputs = self._tokenizer(
            command,
            return_tensors="pt",
            padding="max_length",       
            truncation=True,
            max_length=self._max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        out = self._hf_model(**inputs)
        token_embs = out.last_hidden_state          

        attention_mask = inputs["attention_mask"]   
        key_padding_mask = attention_mask.eq(0)     

        return token_embs.float(), key_padding_mask

    def forward(self, command: str) -> torch.Tensor:
        """Bug #17 fix: nn.Module interface for relabeling.py which calls text_encoder(instruction).
        Mean-pools over valid (non-padded) tokens → (1, embed_dim) tensor.
        """
        token_embs, key_padding_mask = self.encode(command)   # (1, L, D), (1, L) bool
        valid  = ~key_padding_mask                             # (1, L) True where real token
        pooled = (token_embs * valid.unsqueeze(-1)).sum(1)     # (1, D)
        counts = valid.sum(1, keepdim=True).float().clamp(min=1.0)
        return pooled / counts                                 # (1, embed_dim)

    def make_null_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        

        tokens = torch.zeros(1, self._max_length, self.embed_dim, device=self._device)
        mask   = torch.ones(1, self._max_length, dtype=torch.bool, device=self._device)
        return tokens, mask
