from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from histoomnist.data.dataset import ExpressionRateDataset
from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.models.expression_mlp import ExpressionRateRegressor
from histoomnist.models.gene_conditioned import GeneConditionedRateRegressor
from histoomnist.train.common import checkpoint_payload, save_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import ensure_dir, read_manifest
from histoomnist.utils.seed import set_seed


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for batch in tqdm(loader, leave=False, disable=not sys.stderr.isatty()):
        x = batch["features"].to(device)
        y = batch["log1p_rate"].to(device)
        mask = batch.get("expression_mask")
        mask = mask.to(device).bool() if mask is not None else None
        with torch.set_grad_enabled(training):
            pred = model(x)
            if mask is None:
                loss = loss_fn(pred, y)
            else:
                values = (pred - y).pow(2)
                loss = values[mask].mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def build_rate_model(cfg: dict, input_dim: int, output_dim: int):
    model_cfg = cfg["model"]
    name = str(model_cfg.get("name", "gene_conditioned"))
    if name == "expression_mlp":
        return ExpressionRateRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=list(model_cfg["hidden_dims"]),
            dropout=float(model_cfg.get("dropout", 0.20)),
        )
    if name == "gene_conditioned":
        return GeneConditionedRateRegressor(
            input_dim=input_dim,
            num_genes=output_dim,
            latent_dim=int(model_cfg.get("latent_dim", 256)),
            hidden_dims=list(model_cfg.get("hidden_dims", [1024, 512])),
            dropout=float(model_cfg.get("dropout", 0.20)),
        )
    raise ValueError(f"Unsupported expression model: {name}")


def train(cfg: dict) -> Path:
    set_seed(int(cfg.get("seed", 2026)))
    device = torch.device(get_device_name(cfg.get("device")))
    manifest_path = Path(cfg["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    base_dir = manifest_path.parent
    min_total_counts = float(cfg["data"].get("min_total_counts", 1.0))
    gene_names, gene_indices = selected_genes_from_config(cfg, base_dir=base_dir)
    gene_key, raw_st_root = gene_key_settings_from_config(cfg)
    train_ds = ExpressionRateDataset(
        manifest,
        base_dir=base_dir,
        splits=list(cfg["data"]["train_splits"]),
        min_total_counts=min_total_counts,
        fit_standardizer=True,
        gene_names=gene_names,
        gene_indices=gene_indices,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    val_ds = ExpressionRateDataset(
        manifest,
        base_dir=base_dir,
        splits=list(cfg["data"]["val_splits"]),
        min_total_counts=min_total_counts,
        standardizer=train_ds.standardizer,
        gene_names=gene_names,
        gene_indices=gene_indices,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    input_dim = train_ds.x.shape[1]
    output_dim = int(train_ds.output_dim)
    model = build_rate_model(cfg, input_dim=input_dim, output_dim=output_dim).to(device)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()
    out_dir = ensure_dir(cfg["output"]["dir"])
    best_path = out_dir / "best.pt"
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    model,
                    cfg,
                    extra={
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                        "model_name": str(cfg["model"].get("name", "gene_conditioned")),
                        "model_kwargs": {
                            "hidden_dims": list(cfg["model"].get("hidden_dims", [1024, 512])),
                            "dropout": float(cfg["model"].get("dropout", 0.20)),
                            "latent_dim": int(cfg["model"].get("latent_dim", 256)),
                        },
                        "feature_mean": train_ds.standardizer.mean,
                        "feature_std": train_ds.standardizer.std,
                        "best_val_loss": best_val,
                        "genes": train_ds.genes,
                    },
                ),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["training"].get("patience", 12)):
                break
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    best = train(load_config(args.config))
    print(f"saved best checkpoint: {best}")


if __name__ == "__main__":
    main()
