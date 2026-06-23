from __future__ import annotations

import torch
import torch.nn as nn

class CrossAttnMLPBlock(nn.Module):
    

    def __init__(
        self,
        units: int,
        text_dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert units % num_heads == 0, (
            f"units ({units}) must be divisible by num_heads ({num_heads})"
        )
        self.units = units

        self.kv_proj = nn.Linear(text_dim, units, bias=False)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=units,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.zero_gate = nn.Parameter(torch.zeros(1))

        self.norm_attn = nn.RMSNorm(units, eps=1e-4)

        mlp_hidden = units * mlp_ratio
        self.mlp_fc   = nn.Linear(units, mlp_hidden, bias=True)
        self.norm_mlp  = nn.RMSNorm(mlp_hidden, eps=1e-4)
        self.act       = nn.SiLU()
        self.mlp_out   = nn.Linear(mlp_hidden, units, bias=True)

        self.norm_out  = nn.RMSNorm(units, eps=1e-4)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.kv_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.mlp_fc.weight)
        nn.init.zeros_(self.mlp_fc.bias)
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    def project_kv(self, z_tokens: torch.Tensor) -> torch.Tensor:
        

        return self.kv_proj(z_tokens)

    def _attn_mlp(
        self,
        x: torch.Tensor,               
        kv: torch.Tensor,              
        key_padding_mask: torch.Tensor, 
    ) -> torch.Tensor:                 
        attn_out, _ = self.cross_attn(
            query=x,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm_attn(x + self.zero_gate * attn_out)
        h = self.act(self.norm_mlp(self.mlp_fc(x)))
        x = self.norm_out(x + self.mlp_out(h))
        return x

    def forward(
        self,
        x: torch.Tensor,                
        z_tokens: torch.Tensor,         
        key_padding_mask: torch.Tensor, 
    ) -> torch.Tensor:                  
        
        kv = self.project_kv(z_tokens)
        return self._attn_mlp(x, kv, key_padding_mask)

    def forward_cached(
        self,
        x: torch.Tensor,                
        kv: torch.Tensor,               
        key_padding_mask: torch.Tensor, 
    ) -> torch.Tensor:                  
        
        return self._attn_mlp(x, kv, key_padding_mask)

class KVCache:
    

    __slots__ = ("kv_list", "key_padding_mask")

    def __init__(
        self,
        kv_list: list[torch.Tensor],
        key_padding_mask: torch.Tensor,
    ) -> None:
        self.kv_list          = kv_list
        self.key_padding_mask = key_padding_mask

class CrossAttnMLP(nn.Module):
    

    def __init__(
        self,
        inp_dim: int,
        units: int,
        n_layers: int,
        text_dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        symlog_inputs: bool = False,
    ) -> None:
        super().__init__()
        self._symlog_inputs = symlog_inputs
        self.out_dim = units

        self.input_proj = nn.Sequential(
            nn.Linear(inp_dim, units, bias=True),
            nn.RMSNorm(units, eps=1e-4),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            CrossAttnMLPBlock(
                units=units,
                text_dim=text_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(n_layers)
        ])

    def _project_x(self, x: torch.Tensor) -> torch.Tensor:
        if self._symlog_inputs:
            x = torch.sign(x) * torch.log1p(x.abs())
        return self.input_proj(x).unsqueeze(1)   

    def precompute_kv(
        self,
        z_tokens: torch.Tensor,         
        key_padding_mask: torch.Tensor, 
    ) -> KVCache:
        

        kv_list: list[torch.Tensor] = []
        block: CrossAttnMLPBlock
        for block in self.blocks:
            kv_list.append(block.project_kv(z_tokens))
        return KVCache(kv_list=kv_list, key_padding_mask=key_padding_mask)

    def forward(
        self,
        x: torch.Tensor,                
        z_tokens: torch.Tensor,         
        key_padding_mask: torch.Tensor, 
    ) -> torch.Tensor:                  
        
        x = self._project_x(x)
        block: CrossAttnMLPBlock
        for block in self.blocks:
            x = block(x, z_tokens, key_padding_mask)
        return x.squeeze(1)

    def forward_cached(
        self,
        x: torch.Tensor,  
        cache: KVCache,
    ) -> torch.Tensor:    
        
        x = self._project_x(x)
        block: CrossAttnMLPBlock
        for i, block in enumerate(self.blocks):
            x = block.forward_cached(x, cache.kv_list[i], cache.key_padding_mask)
        return x.squeeze(1)
