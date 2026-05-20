from __future__ import annotations

import torch
from torch import nn


class SCellSTGenePredictor(nn.Module):
    """sCellST-style embedding-to-gene predictor for HEST spot features.

    The reference sCellST `GenePredictor` is an MLP over image/cell embeddings.
    This adapter keeps that supervised prediction head while using the HEST
    HIPT/H&E spot features already prepared by the project.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (512, 512),
        dropout: float = 0.1,
        final_activation: str = "identity",
    ):
        super().__init__()
        dims = [int(input_dim)] + [int(x) for x in hidden_dims] + [int(output_dim)]
        layers: list[nn.Module] = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(dims) - 2
            if is_last:
                if final_activation == "identity":
                    layers.append(nn.Identity())
                elif final_activation == "softplus":
                    layers.append(nn.Softplus(beta=20))
                elif final_activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Unsupported final_activation: {final_activation}")
            else:
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(float(dropout)))
        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, expression_mask: torch.Tensor) -> torch.Tensor:
    valid = expression_mask.bool()
    if not torch.any(valid):
        raise ValueError("No valid values for masked MSE.")
    return (pred - target).pow(2)[valid].mean()
