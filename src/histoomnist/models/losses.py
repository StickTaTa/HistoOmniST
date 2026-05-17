from __future__ import annotations

import torch
import torch.nn.functional as F


def negative_binomial_nll(
    counts: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean negative-binomial NLL with mean `mu` and inverse dispersion `theta`."""

    counts = counts.float()
    mu = mu.float().clamp_min(eps)
    theta = theta.float().clamp_min(eps)
    log_prob = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + counts * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )
    return -log_prob.mean()


def log1p_rate_mse(pred_log1p_rate: torch.Tensor, target_log1p_rate: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_log1p_rate, target_log1p_rate)
