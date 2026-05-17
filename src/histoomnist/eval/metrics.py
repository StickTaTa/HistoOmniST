from __future__ import annotations

import numpy as np


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def genewise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}")
    return np.asarray([pearsonr_np(pred[:, i], true[:, i]) for i in range(pred.shape[1])])


def summarize_genewise(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    vals = genewise_pearson(pred, true)
    return {
        "mean_gene_pearson": float(np.nanmean(vals)),
        "median_gene_pearson": float(np.nanmedian(vals)),
        "valid_genes": int(np.isfinite(vals).sum()),
    }


def residualize_on_covariate(x: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(covariate, dtype=np.float64).reshape(-1, 1)
    if x.shape[0] != c.shape[0]:
        raise ValueError("covariate length must match rows in x")
    design = np.concatenate([np.ones_like(c), c], axis=1)
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ beta


def partial_genewise_pearson(pred: np.ndarray, true: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    pred_resid = residualize_on_covariate(pred, covariate)
    true_resid = residualize_on_covariate(true, covariate)
    return genewise_pearson(pred_resid, true_resid)


def sf_metrics(pred_log_sf: np.ndarray, true_log_sf: np.ndarray) -> dict[str, float]:
    pred_log_sf = np.asarray(pred_log_sf).reshape(-1)
    true_log_sf = np.asarray(true_log_sf).reshape(-1)
    pred_sf = np.exp(pred_log_sf)
    true_sf = np.exp(true_log_sf)
    pred_sf = pred_sf / (np.mean(pred_sf) + 1e-8)
    true_sf = true_sf / (np.mean(true_sf) + 1e-8)
    pred_log_sf = np.log(pred_sf + 1e-8)
    true_log_sf = np.log(true_sf + 1e-8)
    tail_mask = true_log_sf >= np.quantile(true_log_sf, 0.90)
    pred_tail_sf = pred_sf[tail_mask]
    true_tail_sf = true_sf[tail_mask]
    return {
        "log_sf_pearson": pearsonr_np(pred_log_sf, true_log_sf),
        "sf_pearson": pearsonr_np(pred_sf, true_sf),
        "log_sf_mae": float(np.mean(np.abs(pred_log_sf - true_log_sf))),
        "log_sf_rmse": float(np.sqrt(np.mean((pred_log_sf - true_log_sf) ** 2))),
        "sf_std_ratio": float(np.std(pred_sf) / (np.std(true_sf) + 1e-8)),
        "sf_top_decile_mean_ratio": float(np.mean(pred_tail_sf) / (np.mean(true_tail_sf) + 1e-8)),
        "log_sf_top_decile_mae": float(np.mean(np.abs(pred_log_sf[tail_mask] - true_log_sf[tail_mask]))),
    }
