from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histoomnist.data.dataset import ExpressionRateDataset, FeatureStandardizer
from histoomnist.eval.metrics import summarize_genewise
from histoomnist.models.expression_mlp import ExpressionRateRegressor
from histoomnist.models.gene_conditioned import GeneConditionedRateRegressor
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest


def evaluate(cfg: dict, checkpoint: str | Path, split_names: list[str] | None = None) -> dict[str, float]:
    device = torch.device(get_device_name(cfg.get("device")))
    ckpt = load_checkpoint(checkpoint, map_location=str(device))
    manifest_path = Path(cfg["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    ds = ExpressionRateDataset(
        manifest,
        base_dir=manifest_path.parent,
        splits=split_names or list(cfg["data"]["test_splits"]),
        min_total_counts=float(cfg["data"].get("min_total_counts", 1.0)),
        standardizer=FeatureStandardizer(mean=ckpt["feature_mean"], std=ckpt["feature_std"]),
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
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["features"].to(device)).cpu().numpy()
            preds.append(pred)
            trues.append(batch["log1p_rate"].numpy())
    metrics = summarize_genewise(np.concatenate(preds), np.concatenate(trues))
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
