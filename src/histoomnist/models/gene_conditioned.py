from __future__ import annotations

import torch
from torch import nn

from histoomnist.models.mlp import MLP


class GeneConditionedRateRegressor(nn.Module):
    """Low-rank gene-conditioned rate decoder with trainable gene embeddings.

    This is a lightweight alternative to CellFM. It does not import or initialize any
    single-cell foundation model.
    """

    def __init__(
        self,
        input_dim: int,
        num_genes: int,
        latent_dim: int = 256,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.20,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [1024, 512]
        self.spot_encoder = MLP(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.gene_embedding = nn.Embedding(num_genes, latent_dim)
        self.gene_bias = nn.Parameter(torch.zeros(num_genes))
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor, gene_ids: torch.Tensor | None = None) -> torch.Tensor:
        spot_latent = self.spot_encoder(features)
        if gene_ids is None:
            weight = self.gene_embedding.weight
            bias = self.gene_bias
        else:
            weight = self.gene_embedding(gene_ids)
            bias = self.gene_bias[gene_ids]
        return spot_latent @ weight.T + bias

    def encode_spots(self, features: torch.Tensor) -> torch.Tensor:
        return self.spot_encoder(features)
