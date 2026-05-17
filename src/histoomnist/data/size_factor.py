from __future__ import annotations

import numpy as np


def row_sums(matrix) -> np.ndarray:
    sums = matrix.sum(axis=1)
    if hasattr(sums, "A1"):
        return sums.A1.astype(np.float32)
    return np.asarray(sums).reshape(-1).astype(np.float32)


def compute_size_factor(counts, min_total_counts: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return mean-one size factors and a valid-spot mask."""

    totals = row_sums(counts)
    valid = np.isfinite(totals) & (totals >= float(min_total_counts))
    if not np.any(valid):
        raise ValueError("No valid spots after total-count filtering.")
    mean_total = float(np.mean(totals[valid]))
    if mean_total <= 0:
        raise ValueError("Mean total count must be positive.")
    sf = totals / mean_total
    return sf.astype(np.float32), valid


def log_size_factor(size_factor: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log(np.clip(size_factor.astype(np.float32), eps, None)).astype(np.float32)


def normalize_predicted_sf_mean_one(log_sf: np.ndarray) -> np.ndarray:
    """Normalize predicted factors to mean one without using true counts."""

    sf = np.exp(log_sf.astype(np.float32))
    mean_sf = float(np.mean(sf))
    if mean_sf <= 0 or not np.isfinite(mean_sf):
        raise ValueError("Predicted size factor mean is invalid.")
    return (sf / mean_sf).astype(np.float32)
