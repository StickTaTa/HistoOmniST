from __future__ import annotations

import torch
from torch import nn

from histoomnist.models.mlp import MLP


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SizeFactorRegressor(nn.Module):
    """Image-only log(size_factor) predictor.

    The default mode is a residual MLP intended for engineered HIPT spot features
    (spot mean/std, neighborhood context, coordinate Fourier features). The legacy
    plain MLP path is kept for ablations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.15,
        architecture: str = "residual_mlp",
        width: int = 512,
        depth: int = 4,
    ):
        super().__init__()
        self.architecture = architecture
        if architecture == "plain_mlp":
            if not hidden_dims:
                hidden_dims = [width] * depth
            self.model = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, dropout=dropout)
        elif architecture == "residual_mlp":
            layers: list[nn.Module] = [
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            layers.extend(ResidualMLPBlock(width, dropout) for _ in range(depth))
            layers.extend([nn.LayerNorm(width), nn.Linear(width, 1)])
            self.model = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported SF architecture: {architecture}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    @torch.no_grad()
    def predict_sf(self, features: torch.Tensor, mean_one: bool = True) -> torch.Tensor:
        log_sf = self.forward(features)
        sf = torch.exp(log_sf)
        if mean_one:
            sf = sf / sf.mean().clamp_min(1e-6)
        return sf
