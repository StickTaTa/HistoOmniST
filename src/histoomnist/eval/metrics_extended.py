from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from histoomnist.eval.metrics import genewise_pearson, pearsonr_np


def _as_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}")
    return arr


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x[mask], y[mask]).statistic)
    except Exception:
        xr = pd.Series(x[mask]).rank(method="average").to_numpy()
        yr = pd.Series(y[mask]).rank(method="average").to_numpy()
        return pearsonr_np(xr, yr)


def _r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 2:
        return float("nan")
    y = true[mask]
    p = pred[mask]
    denom = np.sum((y - y.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def overall_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = _as_2d_float("pred", pred)
    true = _as_2d_float("true", true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}")
    diff = pred - true
    return {
        "n_spots": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "pearson": pearsonr_np(pred.reshape(-1), true.reshape(-1)),
        "spearman": _safe_spearman(pred.reshape(-1), true.reshape(-1)),
        "mse": float(np.nanmean(diff**2)),
        "rmse": float(np.sqrt(np.nanmean(diff**2))),
        "mae": float(np.nanmean(np.abs(diff))),
        "r2": _r2_score(pred, true),
        "mean_gene_pearson": float(np.nanmean(genewise_pearson(pred, true))),
        "median_gene_pearson": float(np.nanmedian(genewise_pearson(pred, true))),
    }


def gene_wise_metrics(pred: np.ndarray, true: np.ndarray, genes: list[str] | None = None) -> pd.DataFrame:
    pred = _as_2d_float("pred", pred)
    true = _as_2d_float("true", true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}")
    if genes is None:
        genes = [f"gene_{i}" for i in range(pred.shape[1])]
    if len(genes) != pred.shape[1]:
        raise ValueError(f"genes length {len(genes)} != n_genes {pred.shape[1]}")
    rows = []
    for i, gene in enumerate(genes):
        diff = pred[:, i] - true[:, i]
        rows.append(
            {
                "gene": gene,
                "pearson": pearsonr_np(pred[:, i], true[:, i]),
                "spearman": _safe_spearman(pred[:, i], true[:, i]),
                "mse": float(np.nanmean(diff**2)),
                "rmse": float(np.sqrt(np.nanmean(diff**2))),
                "mae": float(np.nanmean(np.abs(diff))),
                "r2": _r2_score(pred[:, i], true[:, i]),
                "true_mean": float(np.nanmean(true[:, i])),
                "pred_mean": float(np.nanmean(pred[:, i])),
                "true_var": float(np.nanvar(true[:, i])),
                "pred_var": float(np.nanvar(pred[:, i])),
            }
        )
    return pd.DataFrame(rows)


def spot_wise_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    spot_ids: list[str] | None = None,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    pred = _as_2d_float("pred", pred)
    true = _as_2d_float("true", true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}")
    if spot_ids is None:
        spot_ids = [f"spot_{i}" for i in range(pred.shape[0])]
    if sample_ids is None:
        sample_ids = ["sample"] * pred.shape[0]
    rows = []
    for i in range(pred.shape[0]):
        diff = pred[i] - true[i]
        rows.append(
            {
                "sample_id": sample_ids[i],
                "spot_id": spot_ids[i],
                "spot_index": i,
                "pearson": pearsonr_np(pred[i], true[i]),
                "spearman": _safe_spearman(pred[i], true[i]),
                "mse": float(np.nanmean(diff**2)),
                "rmse": float(np.sqrt(np.nanmean(diff**2))),
                "mae": float(np.nanmean(np.abs(diff))),
                "r2": _r2_score(pred[i], true[i]),
            }
        )
    return pd.DataFrame(rows)


def sample_wise_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    sample_ids: list[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    pred = _as_2d_float("pred", pred)
    true = _as_2d_float("true", true)
    if sample_ids is None:
        sample_ids = np.asarray(["sample"] * pred.shape[0])
    sample_ids = np.asarray(sample_ids)
    rows = []
    for sample in pd.unique(sample_ids):
        mask = sample_ids == sample
        metrics = overall_metrics(pred[mask], true[mask])
        metrics["sample_id"] = sample
        metrics["n_spots"] = int(mask.sum())
        rows.append(metrics)
    return pd.DataFrame(rows)


def hvg_metrics(gene_df: pd.DataFrame, top_k: int = 1000) -> dict[str, float]:
    if gene_df.empty:
        return {"hvg_top_k": top_k, "hvg_mean_pearson": float("nan"), "hvg_median_pearson": float("nan")}
    ranked = gene_df.sort_values("true_var", ascending=False).head(top_k)
    return {
        "hvg_top_k": int(min(top_k, len(ranked))),
        "hvg_mean_pearson": float(np.nanmean(ranked["pearson"])),
        "hvg_median_pearson": float(np.nanmedian(ranked["pearson"])),
    }


def marker_metrics(gene_df: pd.DataFrame, marker_genes: list[str]) -> pd.DataFrame:
    if not marker_genes:
        return pd.DataFrame(columns=gene_df.columns)
    marker_set = {g.upper() for g in marker_genes}
    return gene_df[gene_df["gene"].astype(str).str.upper().isin(marker_set)].copy()

