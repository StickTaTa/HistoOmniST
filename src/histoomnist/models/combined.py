from __future__ import annotations

import torch
from torch import nn

from histoomnist.models.expression_mlp import ExpressionRateRegressor
from histoomnist.models.sf_model import SizeFactorRegressor


class CombinedCountModel(nn.Module):
    def __init__(self, sf_model: SizeFactorRegressor, rate_model: ExpressionRateRegressor):
        super().__init__()
        self.sf_model = sf_model
        self.rate_model = rate_model

    def forward(self, features: torch.Tensor, normalize_sf_mean_one: bool = True) -> dict[str, torch.Tensor]:
        log1p_rate = self.rate_model(features)
        rate = torch.expm1(log1p_rate).clamp_min(0.0)
        log_sf = self.sf_model(features)
        sf = torch.exp(log_sf)
        if normalize_sf_mean_one:
            sf = sf / sf.mean().clamp_min(1e-6)
            log_sf = torch.log(sf.clamp_min(1e-8))
        pred_count = sf * rate
        return {
            "log_sf": log_sf,
            "sf": sf,
            "log1p_rate": log1p_rate,
            "rate": rate,
            "count": pred_count,
        }
