from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histoomnist.data.dataset import ExpressionRateDataset, FeatureStandardizer
from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.eval.metrics import sf_metrics
from histoomnist.eval.evaluate_expression import GenePearsonAccumulator
from histoomnist.models.expression_mlp import ExpressionRateRegressor
from histoomnist.models.gene_conditioned import GeneConditionedRateRegressor
from histoomnist.models.sf_model import SizeFactorRegressor
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest


def _load_sf_model(cfg: dict, ckpt: dict, device: torch.device) -> SizeFactorRegressor:
    model = SizeFactorRegressor(
        input_dim=int(ckpt["input_dim"]),
        **ckpt.get(
            "model_kwargs",
            {
                "hidden_dims": list(cfg["model"].get("hidden_dims") or []),
                "dropout": float(cfg["model"].get("dropout", 0.15)),
                "architecture": str(cfg["model"].get("architecture", "residual_mlp")),
                "width": int(cfg["model"].get("width", 512)),
                "depth": int(cfg["model"].get("depth", 4)),
            },
        ),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_rate_model(cfg: dict, ckpt: dict, device: torch.device) -> ExpressionRateRegressor:
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
    return model


def evaluate(
    sf_config: dict,
    expression_config: dict,
    sf_checkpoint: str | Path,
    expression_checkpoint: str | Path,
    split_names: list[str] | None = None,
    out_json: str | Path | None = None,
) -> dict[str, float]:
    device = torch.device(get_device_name(expression_config.get("device")))
    sf_ckpt = load_checkpoint(sf_checkpoint, map_location=str(device))
    rate_ckpt = load_checkpoint(expression_checkpoint, map_location=str(device))

    manifest_path = Path(expression_config["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    splits = split_names or list(expression_config["data"]["test_splits"])
    gene_names, gene_indices = selected_genes_from_config(expression_config, base_dir=manifest_path.parent)
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)

    rate_ds = ExpressionRateDataset(
        manifest,
        base_dir=manifest_path.parent,
        splits=splits,
        min_total_counts=float(expression_config["data"].get("min_total_counts", 1.0)),
        standardizer=FeatureStandardizer(mean=rate_ckpt["feature_mean"], std=rate_ckpt["feature_std"]),
        gene_names=gene_names,
        gene_indices=gene_indices,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    sf_model = _load_sf_model(sf_config, sf_ckpt, device)
    rate_model = _load_rate_model(expression_config, rate_ckpt, device)

    rate_loader = DataLoader(rate_ds, batch_size=int(expression_config["training"]["batch_size"]), shuffle=False)

    pred_sf = np.empty(len(rate_ds), dtype=np.float32)
    true_log_sf = np.empty(len(rate_ds), dtype=np.float32)
    sf_standardizer = FeatureStandardizer(mean=sf_ckpt["feature_mean"], std=sf_ckpt["feature_std"])
    offset = 0
    with torch.no_grad():
        for batch in rate_loader:
            raw_features = batch["raw_features"].numpy()
            sf_features = sf_standardizer.transform(raw_features)
            batch_pred_log_sf = sf_model(torch.from_numpy(sf_features).to(device)).cpu().numpy().reshape(-1)
            batch_true_log_sf = batch["true_log_sf"].numpy().reshape(-1)
            stop = offset + batch_pred_log_sf.shape[0]
            pred_sf[offset:stop] = np.exp(batch_pred_log_sf).astype(np.float32)
            true_log_sf[offset:stop] = batch_true_log_sf.astype(np.float32)
            offset = stop

    for sample_id in np.unique(rate_ds.sample_ids):
        idx = rate_ds.sample_ids == sample_id
        pred_sf[idx] = pred_sf[idx] / (pred_sf[idx].mean() + 1e-8)
    true_log_sf = true_log_sf.reshape(-1)

    metrics: dict[str, float] = {}
    metrics.update({f"sf_{k}": v for k, v in sf_metrics(np.log(pred_sf + 1e-8), true_log_sf).items()})

    n_genes = int(rate_ckpt["output_dim"])
    rate_acc = GenePearsonAccumulator(n_genes)
    count_no_sf_acc = GenePearsonAccumulator(n_genes)
    count_pred_sf_acc = GenePearsonAccumulator(n_genes)
    count_oracle_sf_acc = GenePearsonAccumulator(n_genes)

    offset = 0
    with torch.no_grad():
        for batch in rate_loader:
            pred_log1p_rate = rate_model(batch["features"].to(device)).cpu().numpy()
            pred_rate = np.expm1(pred_log1p_rate).clip(min=0.0)
            true_rate = np.expm1(batch["log1p_rate"].numpy())
            expression_mask = batch["expression_mask"].numpy().astype(bool)
            batch_size = pred_rate.shape[0]
            stop = offset + batch_size
            batch_pred_sf = pred_sf[offset:stop]
            batch_true_sf = np.exp(batch["true_log_sf"].numpy().reshape(-1))
            true_count = true_rate * batch_true_sf[:, None]
            rate_acc.update(pred_rate, true_rate, expression_mask)
            count_no_sf_acc.update(pred_rate, true_count, expression_mask)
            count_pred_sf_acc.update(pred_rate * batch_pred_sf[:, None], true_count, expression_mask)
            count_oracle_sf_acc.update(pred_rate * batch_true_sf[:, None], true_count, expression_mask)
            offset = stop

    metrics.update({f"rate_{k}": v for k, v in rate_acc.summarize().items()})
    metrics.update({f"count_no_sf_{k}": v for k, v in count_no_sf_acc.summarize().items()})
    metrics.update({f"count_pred_sf_{k}": v for k, v in count_pred_sf_acc.summarize().items()})
    metrics.update({f"count_oracle_sf_{k}": v for k, v in count_oracle_sf_acc.summarize().items()})
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf-config", required=True)
    parser.add_argument("--expression-config", required=True)
    parser.add_argument("--sf-checkpoint", required=True)
    parser.add_argument("--expression-checkpoint", required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    evaluate(
        sf_config=load_config(args.sf_config),
        expression_config=load_config(args.expression_config),
        sf_checkpoint=args.sf_checkpoint,
        expression_checkpoint=args.expression_checkpoint,
        split_names=args.splits,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
