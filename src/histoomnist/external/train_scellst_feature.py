from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from histoomnist.data.dataset import ExpressionRateDataset, FeatureStandardizer
from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.external.scellst_model import SCellSTGenePredictor, masked_mse
from histoomnist.train.common import checkpoint_payload, load_checkpoint, save_checkpoint
from histoomnist.utils.config import get_device_name
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path
from histoomnist.utils.seed import set_seed


def _manifest_for_splits(expression_config: dict[str, Any], splits: list[str], max_slides: int | None):
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression config data.manifest resolved to None")
    manifest = read_manifest(manifest_path)
    rows = manifest[manifest["split"].isin(splits)].copy()
    if max_slides is not None:
        rows = rows.head(int(max_slides)).copy()
    if rows.empty:
        raise ValueError(f"No manifest rows for splits={splits}")
    return rows, manifest_path.parent


def build_dataset(
    expression_config: dict[str, Any],
    *,
    splits: list[str],
    max_slides: int | None,
    standardizer: FeatureStandardizer | None = None,
    fit_standardizer: bool = False,
) -> ExpressionRateDataset:
    rows, base_dir = _manifest_for_splits(expression_config, splits, max_slides)
    target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
    if target_genes is None or gene_indices is not None:
        raise ValueError("sCellST feature adapter requires data.gene_names_path target genes.")
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    return ExpressionRateDataset(
        rows,
        base_dir=base_dir,
        splits=splits,
        min_total_counts=float(expression_config["data"].get("min_total_counts", 1.0)),
        standardizer=standardizer,
        fit_standardizer=fit_standardizer,
        gene_names=target_genes,
        gene_indices=None,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )


def build_scellst_model(model_cfg: dict[str, Any], *, input_dim: int, output_dim: int) -> SCellSTGenePredictor:
    return SCellSTGenePredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=list(model_cfg.get("hidden_dims", [512, 512])),
        dropout=float(model_cfg.get("dropout", 0.1)),
        final_activation=str(model_cfg.get("final_activation", "identity")),
    )


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for batch in loader:
        features = batch["features"].to(device)
        target = batch["log1p_rate"].to(device)
        expression_mask = batch["expression_mask"].to(device)
        with torch.set_grad_enabled(training):
            pred = model(features)
            loss = masked_mse(pred, target, expression_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_scellst_feature(
    *,
    expression_config: dict[str, Any],
    train_splits: list[str],
    val_splits: list[str],
    output_dir: str | Path,
    model_cfg: dict[str, Any],
    epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
    device_name: str | None = None,
    max_train_slides: int | None = None,
    max_val_slides: int | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    set_seed(int(seed))
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    print(
        f"[scellst] resolved_device={device}"
        + (f" cuda_name={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
        flush=True,
    )
    train_ds = build_dataset(
        expression_config,
        splits=train_splits,
        max_slides=max_train_slides,
        fit_standardizer=True,
    )
    val_ds = build_dataset(
        expression_config,
        splits=val_splits,
        max_slides=max_val_slides,
        standardizer=train_ds.standardizer,
    )
    input_dim = int(train_ds.x.shape[1])
    output_dim = int(train_ds.output_dim)
    model = build_scellst_model(model_cfg, input_dim=input_dim, output_dim=output_dim).to(device)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_val = float("inf")
    history = []
    for epoch in range(1, int(epochs) + 1):
        train_loss = run_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)
        val_loss = run_epoch(model=model, loader=val_loader, device=device)
        row = {"epoch": int(epoch), "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    model,
                    {
                        "model": model_cfg,
                        "target_kind": "log1p_rate",
                        "train_splits": train_splits,
                        "val_splits": val_splits,
                    },
                    extra={
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                        "n_genes": output_dim,
                        "genes": train_ds.genes,
                        "feature_mean": train_ds.standardizer.mean,
                        "feature_std": train_ds.standardizer.std,
                        "best_val_loss": best_val,
                        "history": history,
                    },
                ),
            )
    n_train_slides = int(len(np.unique(train_ds.sample_ids)))
    n_val_slides = int(len(np.unique(val_ds.sample_ids)))
    summary = {
        "checkpoint": str(best_path),
        "device": str(device),
        "target_kind": "log1p_rate",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "train_splits": train_splits,
        "val_splits": val_splits,
        "n_train_slides": n_train_slides,
        "n_val_slides": n_val_slides,
        "n_train_spots": int(len(train_ds)),
        "n_val_spots": int(len(val_ds)),
        "n_genes": output_dim,
        "best_val_loss": float(best_val),
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_scellst_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[SCellSTGenePredictor, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
    model = build_scellst_model(
        ckpt["config"]["model"],
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export_scellst_feature_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    out_dir: str | Path,
    splits: list[str],
    batch_size: int = 512,
    device_name: str | None = None,
    max_slides: int | None = None,
    max_spots_per_slide: int | None = None,
) -> dict[str, Any]:
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    model, ckpt = load_scellst_checkpoint(checkpoint_path, device)
    ds = build_dataset(
        expression_config,
        splits=splits,
        max_slides=max_slides,
        standardizer=FeatureStandardizer(mean=ckpt["feature_mean"], std=ckpt["feature_std"]),
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    by_slide: dict[str, list[np.ndarray]] = defaultdict(list)
    offset = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["features"].to(device)).detach().cpu().numpy().astype(np.float32, copy=False)
            batch_sample_ids = ds.sample_ids[offset : offset + pred.shape[0]]
            for idx, sample_id in enumerate(batch_sample_ids):
                chunks = by_slide[str(sample_id)]
                if max_spots_per_slide is not None:
                    current = sum(chunk.shape[0] for chunk in chunks)
                    if current >= int(max_spots_per_slide):
                        continue
                chunks.append(pred[idx : idx + 1])
            offset += pred.shape[0]
    out = Path(out_dir)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    target_kind = str(ckpt["config"].get("target_kind", "log1p_rate"))
    (out / "genes.txt").write_text("\n".join(ckpt["genes"]) + "\n", encoding="utf-8")
    expected_spots = {str(sample_id): int((ds.sample_ids == sample_id).sum()) for sample_id in np.unique(ds.sample_ids)}
    slide_rows = []
    for sample_id, chunks in sorted(by_slide.items()):
        array = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if max_spots_per_slide is not None:
            array = array[: int(max_spots_per_slide)]
        np.save(pred_dir / f"{sample_id}_{target_kind}.npy", array)
        expected = int(expected_spots.get(sample_id, -1))
        slide_rows.append(
            {
                "sample_id": sample_id,
                "n_predicted_spots": int(array.shape[0]),
                "expected_spots": expected,
                "complete_slide_prediction": bool(expected >= 0 and int(array.shape[0]) == expected),
            }
        )
    all_complete = bool(slide_rows) and all(bool(row["complete_slide_prediction"]) for row in slide_rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "prediction_kind": target_kind,
        "splits": splits,
        "n_slides": int(len(slide_rows)),
        "n_genes": int(len(ckpt["genes"])),
        "max_slides": None if max_slides is None else int(max_slides),
        "max_spots_per_slide": None if max_spots_per_slide is None else int(max_spots_per_slide),
        "all_slide_predictions_complete": all_complete,
        "benchmark_evaluable_without_truncation": bool(all_complete and max_spots_per_slide is None),
        "slides": slide_rows,
        "outputs": {
            "prediction_root": str(out),
            "genes": str(out / "genes.txt"),
            "predictions": str(pred_dir),
        },
    }
    (out / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
