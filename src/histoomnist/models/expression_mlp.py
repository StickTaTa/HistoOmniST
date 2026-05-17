from __future__ import annotations

import torch
from torch import nn

from histoomnist.models.mlp import MLP


class ExpressionRateRegressor(nn.Module):
    """Predict log1p normalized gene-expression rates from image features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.20,
    ):
        super().__init__()
        self.model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def predict_rate(self, features: torch.Tensor) -> torch.Tensor:
        return torch.expm1(self.forward(features)).clamp_min(0.0)
