from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histoomnist.data.dataset import ExpressionRateDataset, FeatureStandardizer
from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.models.expression_mlp import ExpressionRateRegressor
from histoomnist.models.gene_conditioned import GeneConditionedRateRegressor
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest


def summarize_genewise_masked(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if pred.shape != true.shape or pred.shape != mask.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}, mask={mask.shape}")
    vals = []
    for idx in range(pred.shape[1]):
        keep = mask[:, idx].astype(bool) & np.isfinite(pred[:, idx]) & np.isfinite(true[:, idx])
        if keep.sum() < 3:
            vals.append(np.nan)
            continue
        x = pred[keep, idx].astype(np.float64)
        y = true[keep, idx].astype(np.float64)
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
        vals.append(float(np.sum(x * y) / denom) if denom > 0 else np.nan)
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean_gene_pearson": float(np.nanmean(vals)),
        "median_gene_pearson": float(np.nanmedian(vals)),
        "valid_genes": int(np.isfinite(vals).sum()),
    }


class GenePearsonAccumulator:
    def __init__(self, n_genes: int):
        self.n = np.zeros(n_genes, dtype=np.float64)
        self.sum_x = np.zeros(n_genes, dtype=np.float64)
        self.sum_y = np.zeros(n_genes, dtype=np.float64)
        self.sum_x2 = np.zeros(n_genes, dtype=np.float64)
        self.sum_y2 = np.zeros(n_genes, dtype=np.float64)
        self.sum_xy = np.zeros(n_genes, dtype=np.float64)

    def update(self, pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> None:
        if pred.shape != true.shape or pred.shape != mask.shape:
            raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}, mask={mask.shape}")
        valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(true)
        x = np.where(valid, pred, 0.0).astype(np.float64)
        y = np.where(valid, true, 0.0).astype(np.float64)
        self.n += valid.sum(axis=0)
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.sum_x2 += (x * x).sum(axis=0)
        self.sum_y2 += (y * y).sum(axis=0)
        self.sum_xy += (x * y).sum(axis=0)

    def summarize(self) -> dict[str, float]:
        numerator = self.sum_xy - (self.sum_x * self.sum_y / np.maximum(self.n, 1.0))
        denom_x = self.sum_x2 - (self.sum_x * self.sum_x / np.maximum(self.n, 1.0))
        denom_y = self.sum_y2 - (self.sum_y * self.sum_y / np.maximum(self.n, 1.0))
        denom = np.sqrt(np.maximum(denom_x, 0.0) * np.maximum(denom_y, 0.0))
        vals = np.full_like(numerator, np.nan, dtype=np.float64)
        keep = (self.n >= 3) & (denom > 0)
        vals[keep] = numerator[keep] / denom[keep]
        return {
            "mean_gene_pearson": float(np.nanmean(vals)),
            "median_gene_pearson": float(np.nanmedian(vals)),
            "valid_genes": int(np.isfinite(vals).sum()),
        }


def evaluate(cfg: dict, checkpoint: str | Path, split_names: list[str] | None = None) -> dict[str, float]:
    device = torch.device(get_device_name(cfg.get("device")))
    ckpt = load_checkpoint(checkpoint, map_location=str(device))
    manifest_path = Path(cfg["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    gene_names, gene_indices = selected_genes_from_config(cfg, base_dir=manifest_path.parent)
    gene_key, raw_st_root = gene_key_settings_from_config(cfg)
    ds = ExpressionRateDataset(
        manifest,
        base_dir=manifest_path.parent,
        splits=split_names or list(cfg["data"]["test_splits"]),
        min_total_counts=float(cfg["data"].get("min_total_counts", 1.0)),
        standardizer=FeatureStandardizer(mean=ckpt["feature_mean"], std=ckpt["feature_std"]),
        gene_names=gene_names,
        gene_indices=gene_indices,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    model_name = ckpt.get("model_name", cfg["model"].get("name", "expression_mlp"))
    kwargs = ckpt.get("model_kwargs", {})
    if model_name == "gene_conditioned":
        model = GeneConditionedRateRegressor(
            input_dim=int(ckpt["input_dim"]),
            num_genes=int(ckpt["output_dim"]),
            latent_dim=int(kwargs.get("latent_dim", cfg["model"].get("latent_dim", 256))),
            hidden_dims=list(kwargs.get("hidden_dims", cfg["model"].get("hidden_dims", [1024, 512]))),
            dropout=float(kwargs.get("dropout", cfg["model"].get("dropout", 0.20))),
        ).to(device)
    else:
        model = ExpressionRateRegressor(
            input_dim=int(ckpt["input_dim"]),
            output_dim=int(ckpt["output_dim"]),
            hidden_dims=list(kwargs.get("hidden_dims", cfg["model"]["hidden_dims"])),
            dropout=float(kwargs.get("dropout", cfg["model"].get("dropout", 0.20))),
        ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    accumulator = GenePearsonAccumulator(int(ckpt["output_dim"]))
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["features"].to(device)).cpu().numpy()
            accumulator.update(pred, batch["log1p_rate"].numpy(), batch["expression_mask"].numpy().astype(bool))
    metrics = accumulator.summarize()
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint, split_names=args.splits)


if __name__ == "__main__":
    main()
