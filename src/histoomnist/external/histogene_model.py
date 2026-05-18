from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HisToGeneChunkRegressor(nn.Module):
    """Pure PyTorch HisToGene-style patch transformer for HEST patch-H5 chunks."""

    def __init__(
        self,
        *,
        n_genes: int,
        patch_size: int = 56,
        dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_pos: int = 64,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.n_pos = int(n_pos)
        patch_dim = 3 * self.patch_size * self.patch_size
        self.patch_embedding = nn.Linear(patch_dim, int(dim))
        self.x_embed = nn.Embedding(self.n_pos, int(dim))
        self.y_embed = nn.Embedding(self.n_pos, int(dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(dim),
            nhead=int(n_heads),
            dim_feedforward=2 * int(dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.gene_head = nn.Sequential(nn.LayerNorm(int(dim)), nn.Linear(int(dim), int(n_genes)))

    def forward(self, patches: torch.Tensor, positions: torch.Tensor, spot_mask: torch.Tensor | None = None) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(f"patches must be shaped (batch, spots, channels, height, width), got {tuple(patches.shape)}")
        batch, spots, channels, height, width = patches.shape
        x = patches.reshape(batch * spots, channels, height, width)
        if height != self.patch_size or width != self.patch_size:
            x = F.interpolate(x, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)
        x = x.reshape(batch, spots, -1)
        x = self.patch_embedding(x)
        bins = torch.clamp((positions * float(self.n_pos - 1)).round().long(), min=0, max=self.n_pos - 1)
        x = x + self.x_embed(bins[:, :, 0]) + self.y_embed(bins[:, :, 1])
        padding_mask = None if spot_mask is None else ~spot_mask.bool()
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.gene_head(x)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    spot_mask: torch.Tensor,
    expression_mask: torch.Tensor,
) -> torch.Tensor:
    valid = spot_mask.bool().unsqueeze(-1) & expression_mask.bool().unsqueeze(1)
    if not torch.any(valid):
        raise ValueError("No valid values for masked MSE.")
    return (pred - target).pow(2)[valid].mean()
