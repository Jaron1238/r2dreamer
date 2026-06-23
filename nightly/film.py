

from __future__ import annotations

import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    

    def __init__(self, units: int):
        super().__init__()
        self.units = units

    def forward(
        self,
        x: torch.Tensor,        
        gamma: torch.Tensor,    
        beta: torch.Tensor,     
    ) -> torch.Tensor:
        return gamma * x + beta

class FiLMConditioner(nn.Module):
    

    def __init__(
        self,
        text_dim: int,
        units: int,
        n_layers: int,
        hidden: int | None = None,
    ):
        super().__init__()
        if hidden is None:
            hidden = 4 * units
        self.units    = units
        self.n_layers = n_layers
        out_dim       = 2 * units * n_layers   

        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self._init_identity()

    
    def _init_identity(self) -> None:
        
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            bias = self.net[-1].bias
            
            
            bias[:self.units * self.n_layers] = 1.0
            bias[self.units * self.n_layers:] = 0.0

    
    def forward(
        self,
        z_text: torch.Tensor,   
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        

        params = self.net(z_text)                               
        gammas, betas = params.chunk(2, dim=-1)                 
        pairs = []
        for l in range(self.n_layers):
            g = gammas[..., l * self.units:(l + 1) * self.units]
            b = betas[...,  l * self.units:(l + 1) * self.units]
            pairs.append((g, b))
        return pairs

class FiLMedMLP(nn.Module):
    

    def __init__(
        self,
        inp_dim: int,
        units: int,
        n_layers: int,
        text_dim: int,
        act: type[nn.Module] = nn.SiLU,
        symlog_inputs: bool = False,
    ):
        super().__init__()
        self._symlog_inputs = symlog_inputs
        self.out_dim        = units
        self.n_layers       = n_layers

        self.linears  = nn.ModuleList()
        self.norms    = nn.ModuleList()
        self.acts     = nn.ModuleList()
        self.films    = nn.ModuleList()

        dim = inp_dim
        for _ in range(n_layers):
            self.linears.append(nn.Linear(dim, units, bias=True))
            self.norms.append(nn.RMSNorm(units, eps=1e-4, dtype=torch.float32))
            self.acts.append(act())
            self.films.append(FiLMLayer(units))
            dim = units

        self.conditioner = FiLMConditioner(
            text_dim  = text_dim,
            units     = units,
            n_layers  = n_layers,
        )

    def forward(
        self,
        x: torch.Tensor,          
        z_text: torch.Tensor,     
    ) -> torch.Tensor:
        if self._symlog_inputs:
            x = torch.sign(x) * torch.log1p(x.abs())

        film_pairs = self.conditioner(z_text)   

        for linear, norm, act, film_layer, (gamma, beta) in zip(
            self.linears, self.norms, self.acts, self.films, film_pairs
        ):
            x = film_layer(act(norm(linear(x))), gamma, beta)

        return x
