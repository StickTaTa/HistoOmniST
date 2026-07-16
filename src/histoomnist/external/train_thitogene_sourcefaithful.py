from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from histoomnist.external.thitogene_sourcefaithful import (
    OfficialTHItoGeneCore,
    THItoGeneHESTSlideDataset,
    thitogene_masked_mse,
    thitogene_source_metadata,
)
from histoomnist.train.common import checkpoint_payload, load_checkpoint, save_checkpoint
from histoomnist.utils.config import get_device_name
from histoomnist.utils.seed import set_seed


OFFICIAL_THITOGENE_MODEL_DEFAULTS = {
    "patch_size": 112,
    "n_layers": 4,
    "dim": 1024,
    "heads": [16, 8],
    "dropout": 0.2,
    "n_pos": 64,
    "learning_rate": 1.0e-5,
    "caps": 20,
    "route_dim": 64,
    "optimizer": "Adam",
    "loss": "MSE",
}


def build_thitogene_sourcefaithful_model(
    *,
    n_genes: int,
    model_cfg: dict[str, Any] | None = None,
    upstream_root: str | Path | None = None,
) -> OfficialTHItoGeneCore:
    cfg = dict(OFFICIAL_THITOGENE_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    heads = [int(x) for x in cfg.get("heads", [16, 8])]
    if heads != [16, 8]:
        raise ValueError("Official THItoGene benchmark must keep heads=[16, 8].")
    return OfficialTHItoGeneCore(
        patch_size=int(cfg["patch_size"]),
        n_layers=int(cfg["n_layers"]),
        n_genes=int(n_genes),
        dim=int(cfg["dim"]),
        learning_rate=float(cfg["learning_rate"]),
        dropout=float(cfg["dropout"]),
        n_pos=int(cfg["n_pos"]),
        heads=heads,
        caps=int(cfg["caps"]),
        route_dim=int(cfg["route_dim"]),
        upstream_root=upstream_root,
    )


def run_thitogene_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_kind: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    used_spots: list[int] = []
    for batch in loader:
        patches = batch["patches"].to(device)
        positions = batch["positions"].to(device)
        adj = batch["adj"].to(device)
        target = batch[target_kind].to(device)
        expression_mask = batch["expression_mask"].to(device)
        with torch.set_grad_enabled(training):
            pred = model(patches, positions, adj)
            loss = thitogene_masked_mse(pred, target, expression_mask=expression_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        used_spots.extend(int(x) for x in batch["n_used_spots"].detach().cpu().numpy())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "n_batches": int(len(losses)),
        "mean_used_spots": float(np.mean(used_spots)) if used_spots else float("nan"),
        "max_used_spots": int(max(used_spots)) if used_spots else 0,
    }


def data_smoke_summary(
    *,
    expression_config: dict[str, Any],
    splits: list[str],
    output_dir: str | Path,
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    target_kind: str = "log1p_rate",
    patch_size: int = 112,
    n_pos: int = 64,
    k_neighbors: int = 4,
    max_spots_per_slide: int | None = None,
    max_slide_spots: int | None = None,
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset = THItoGeneHESTSlideDataset(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_kind=target_kind,
        patch_size=patch_size,
        n_pos=n_pos,
        k_neighbors=k_neighbors,
        max_spots_per_slide=max_spots_per_slide,
        max_slide_spots=max_slide_spots,
        upstream_root=upstream_root,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slide_frame = dataset.slide_summary_frame()
    slide_frame.to_csv(out / "data_smoke_slides.csv", index=False)
    item = dataset[0]
    patches = item["patches"].numpy()
    positions = item["positions"].numpy()
    target = item[target_kind].numpy()
    mask = item["expression_mask"].numpy().astype(bool)
    adj = item["adj"].numpy()
    adj_density = float(np.count_nonzero(adj) / max(adj.size, 1))
    summary = {
        "splits": [str(x) for x in splits],
        "slide_ids": None if slide_ids is None else [str(x) for x in slide_ids],
        "max_slides": None if max_slides is None else int(max_slides),
        "smallest_slides": bool(smallest_slides),
        "target_kind": str(target_kind),
        "patch_size": int(patch_size),
        "n_pos": int(n_pos),
        "k_neighbors": int(k_neighbors),
        "max_spots_per_slide": None if max_spots_per_slide is None else int(max_spots_per_slide),
        "max_slide_spots": None if max_slide_spots is None else int(max_slide_spots),
        "n_slides": int(len(dataset)),
        "n_target_genes": int(len(dataset.target_genes)),
        "first_slide": {
            "sample_id": str(item["sample_id"]),
            "patches_shape": list(patches.shape),
            "positions_shape": list(positions.shape),
            "target_shape": list(target.shape),
            "adj_shape": list(adj.shape),
            "adj_density": adj_density,
            "adj_nonzero": int(np.count_nonzero(adj)),
            "patch_value_min": float(np.min(patches)),
            "patch_value_max": float(np.max(patches)),
            "positions_min": int(np.min(positions)),
            "positions_max": int(np.max(positions)),
            "target_finite_fraction_measured": float(np.mean(np.isfinite(target[:, mask]))),
            "n_spots": int(item["n_spots"]),
            "n_used_spots": int(item["n_used_spots"]),
            "truncated_for_smoke": bool(item["truncated_for_smoke"]),
        },
        "source": thitogene_source_metadata(upstream_root),
        "outputs": {
            "slides": str(out / "data_smoke_slides.csv"),
            "summary": str(out / "data_smoke_summary.json"),
        },
    }
    (out / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train_thitogene_sourcefaithful(
    *,
    expression_config: dict[str, Any],
    train_splits: list[str],
    val_splits: list[str],
    output_dir: str | Path,
    train_slide_ids: list[str] | None = None,
    val_slide_ids: list[str] | None = None,
    max_train_slides: int | None = None,
    max_val_slides: int | None = None,
    smallest_train_slides: bool = False,
    smallest_val_slides: bool = False,
    max_train_spots_per_slide: int | None = None,
    max_val_spots_per_slide: int | None = None,
    max_train_slide_spots: int | None = None,
    max_val_slide_spots: int | None = None,
    target_kind: str = "log1p_rate",
    epochs: int = 1,
    lr: float = 1.0e-5,
    patience: int | None = None,
    min_delta: float = 0.0,
    device_name: str | None = None,
    seed: int = 2026,
    model_cfg: dict[str, Any] | None = None,
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(int(seed))
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    print(
        f"[thitogene-sourcefaithful] resolved_device={device}"
        + (f" cuda_name={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
        flush=True,
    )
    cfg = dict(OFFICIAL_THITOGENE_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    train_ds = THItoGeneHESTSlideDataset(
        expression_config,
        splits=train_splits,
        slide_ids=train_slide_ids,
        max_slides=max_train_slides,
        smallest_slides=smallest_train_slides,
        target_kind=target_kind,
        patch_size=int(cfg["patch_size"]),
        n_pos=int(cfg["n_pos"]),
        k_neighbors=4,
        max_spots_per_slide=max_train_spots_per_slide,
        max_slide_spots=max_train_slide_spots,
        upstream_root=upstream_root,
    )
    val_ds = THItoGeneHESTSlideDataset(
        expression_config,
        splits=val_splits,
        slide_ids=val_slide_ids,
        max_slides=max_val_slides,
        smallest_slides=smallest_val_slides,
        target_kind=target_kind,
        patch_size=int(cfg["patch_size"]),
        n_pos=int(cfg["n_pos"]),
        k_neighbors=4,
        max_spots_per_slide=max_val_spots_per_slide,
        max_slide_spots=max_val_slide_spots,
        upstream_root=upstream_root,
    )
    model = build_thitogene_sourcefaithful_model(
        n_genes=len(train_ds.target_genes),
        model_cfg=cfg,
        upstream_root=upstream_root,
    ).to(device)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_slide_frame = train_ds.slide_summary_frame()
    val_slide_frame = val_ds.slide_summary_frame()
    train_slide_frame.to_csv(out / "train_slides.csv", index=False)
    val_slide_frame.to_csv(out / "val_slides.csv", index=False)
    best_path = out / "best.pt"
    history = []
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    for epoch in range(1, int(epochs) + 1):
        train_stats = run_thitogene_epoch(
            model=model,
            loader=train_loader,
            device=device,
            target_kind=target_kind,
            optimizer=optimizer,
        )
        val_stats = run_thitogene_epoch(
            model=model,
            loader=val_loader,
            device=device,
            target_kind=target_kind,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_stats["loss"]),
            "val_loss": float(val_stats["loss"]),
            "train_max_used_spots": int(train_stats["max_used_spots"]),
            "val_max_used_spots": int(val_stats["max_used_spots"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f}",
            flush=True,
        )
        improved = row["val_loss"] < (best_val - float(min_delta))
        if improved:
            best_val = float(row["val_loss"])
            best_epoch = int(epoch)
            stale_epochs = 0
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    model,
                    {
                        "method": "thitogene_sourcefaithful",
                        "model": cfg,
                        "target_kind": target_kind,
                        "train_splits": [str(x) for x in train_splits],
                        "val_splits": [str(x) for x in val_splits],
                        "source": thitogene_source_metadata(upstream_root),
                    },
                    extra={
                        "n_genes": len(train_ds.target_genes),
                        "genes": train_ds.target_genes,
                        "best_val_loss": best_val,
                        "history": history,
                    },
                ),
            )
        else:
            stale_epochs += 1
        if patience is not None and stale_epochs >= int(patience):
            stopped_early = True
            print(
                f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                f"best_val_loss={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break
    truncated = bool(
        train_slide_frame["truncated_for_smoke"].any()
        or val_slide_frame["truncated_for_smoke"].any()
    )
    limited_slide_scope = bool(
        train_slide_ids
        or val_slide_ids
        or max_train_slides is not None
        or max_val_slides is not None
        or smallest_train_slides
        or smallest_val_slides
        or max_train_slide_spots is not None
        or max_val_slide_spots is not None
    )
    formal_benchmark_candidate = bool(
        not truncated
        and not limited_slide_scope
        and [str(x) for x in train_splits] == ["train"]
        and [str(x) for x in val_splits] == ["val"]
    )
    summary = {
        "checkpoint": str(best_path),
        "device": str(device),
        "method": "thitogene_sourcefaithful",
        "target_kind": target_kind,
        "precision": "fp32",
        "amp": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None,
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else None,
        "source_faithful_core": True,
        "formal_benchmark_candidate": formal_benchmark_candidate,
        "limited_slide_scope": limited_slide_scope,
        "truncated_for_smoke": truncated,
        "model": cfg,
        "epochs": int(len(history)),
        "max_epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "early_stopping_patience": "" if patience is None else int(patience),
        "early_stopping_min_delta": float(min_delta),
        "stopped_early": bool(stopped_early),
        "train_splits": [str(x) for x in train_splits],
        "val_splits": [str(x) for x in val_splits],
        "train_slide_ids": None if train_slide_ids is None else [str(x) for x in train_slide_ids],
        "val_slide_ids": None if val_slide_ids is None else [str(x) for x in val_slide_ids],
        "max_train_slide_spots": None if max_train_slide_spots is None else int(max_train_slide_spots),
        "max_val_slide_spots": None if max_val_slide_spots is None else int(max_val_slide_spots),
        "n_train_slides": int(len(train_ds)),
        "n_val_slides": int(len(val_ds)),
        "n_genes": int(len(train_ds.target_genes)),
        "history": history,
        "source": thitogene_source_metadata(upstream_root),
        "outputs": {
            "checkpoint": str(best_path),
            "train_slides": str(out / "train_slides.csv"),
            "val_slides": str(out / "val_slides.csv"),
            "summary": str(out / "train_summary.json"),
        },
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_thitogene_sourcefaithful_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[OfficialTHItoGeneCore, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
    model = build_thitogene_sourcefaithful_model(
        n_genes=int(ckpt["n_genes"]),
        model_cfg=ckpt["config"]["model"],
        upstream_root=ckpt["config"].get("source", {}).get("upstream_root"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export_thitogene_sourcefaithful_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    out_dir: str | Path,
    splits: list[str],
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_spots_per_slide: int | None = None,
    max_slide_spots: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    model, ckpt = load_thitogene_sourcefaithful_checkpoint(checkpoint_path, device)
    target_kind = str(ckpt["config"].get("target_kind", "log1p_rate"))
    model_cfg = dict(ckpt["config"]["model"])
    ds = THItoGeneHESTSlideDataset(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_kind=target_kind,
        patch_size=int(model_cfg["patch_size"]),
        n_pos=int(model_cfg["n_pos"]),
        k_neighbors=4,
        max_spots_per_slide=max_spots_per_slide,
        max_slide_spots=max_slide_spots,
        upstream_root=ckpt["config"].get("source", {}).get("upstream_root"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    out = Path(out_dir)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "genes.txt").write_text("\n".join(ckpt["genes"]) + "\n", encoding="utf-8")
    slide_rows = []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["patches"].to(device),
                batch["positions"].to(device),
                batch["adj"].to(device),
            )
            values = pred.detach().cpu().numpy().astype(np.float32, copy=False)
            sample_id = str(batch["sample_id"][0])
            np.save(pred_dir / f"{sample_id}_{target_kind}.npy", values)
            n_spots = int(batch["n_spots"].item())
            n_used = int(batch["n_used_spots"].item())
            slide_rows.append(
                {
                    "sample_id": sample_id,
                    "n_predicted_spots": int(values.shape[0]),
                    "expected_spots": n_spots,
                    "complete_slide_prediction": bool(values.shape[0] == n_spots),
                    "truncated_for_smoke": bool(n_used != n_spots),
                }
            )
            print(
                f"[thitogene-sourcefaithful] predicted {sample_id}: "
                f"spots={values.shape[0]} genes={values.shape[1]}",
                flush=True,
            )
    all_complete = bool(slide_rows) and all(bool(row["complete_slide_prediction"]) for row in slide_rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "method": "thitogene_sourcefaithful",
        "prediction_kind": target_kind,
        "splits": [str(x) for x in splits],
        "n_slides": int(len(slide_rows)),
        "n_genes": int(len(ckpt["genes"])),
        "max_slide_spots": None if max_slide_spots is None else int(max_slide_spots),
        "all_slide_predictions_complete": bool(all_complete),
        "benchmark_evaluable_without_truncation": bool(all_complete and max_spots_per_slide is None),
        "slides": slide_rows,
        "outputs": {
            "prediction_root": str(out),
            "genes": str(out / "genes.txt"),
            "predictions": str(pred_dir),
            "summary": str(out / "prediction_summary.json"),
        },
    }
    (out / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def summarize_prediction_slides(prediction_summary_path: str | Path) -> pd.DataFrame:
    summary = json.loads(Path(prediction_summary_path).read_text(encoding="utf-8"))
    by_flag: dict[str, int] = defaultdict(int)
    for row in summary.get("slides", []):
        by_flag[str(bool(row.get("complete_slide_prediction", False)))] += 1
    return pd.DataFrame([{"complete_slide_prediction": key, "n_slides": value} for key, value in by_flag.items()])


def forward_smoke_summary(
    *,
    expression_config: dict[str, Any],
    splits: list[str],
    output_dir: str | Path,
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    target_kind: str = "log1p_rate",
    device_name: str | None = None,
    seed: int = 2026,
    model_cfg: dict[str, Any] | None = None,
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(int(seed))
    cfg = dict(OFFICIAL_THITOGENE_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    dataset = THItoGeneHESTSlideDataset(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_kind=target_kind,
        patch_size=int(cfg["patch_size"]),
        n_pos=int(cfg["n_pos"]),
        k_neighbors=4,
        upstream_root=upstream_root,
    )
    item = dataset[0]
    model = build_thitogene_sourcefaithful_model(
        n_genes=len(dataset.target_genes),
        model_cfg=cfg,
        upstream_root=upstream_root,
    ).to(device)
    model.train()
    patches = item["patches"].unsqueeze(0).to(device)
    positions = item["positions"].unsqueeze(0).to(device)
    adj = item["adj"].to(device)
    target = item[target_kind].unsqueeze(0).to(device)
    expression_mask = item["expression_mask"].unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pred = model(patches, positions, adj)
    loss = thitogene_masked_mse(pred, target, expression_mask=expression_mask)
    loss.backward()
    memory = {}
    if device.type == "cuda":
        memory = {
            "max_allocated_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
            "max_reserved_gb": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
        }
    summary = {
        "status": "ok",
        "sample_id": str(item["sample_id"]),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "precision": "fp32",
        "amp": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None,
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else None,
        "n_used_spots": int(item["n_used_spots"]),
        "n_genes": int(len(dataset.target_genes)),
        "patches_shape": list(item["patches"].shape),
        "positions_shape": list(item["positions"].shape),
        "adj_shape": list(item["adj"].shape),
        "target_shape": list(item[target_kind].shape),
        "prediction_shape": list(pred.shape),
        "loss": float(loss.detach().cpu()),
        "model": cfg,
        "memory": memory,
        "source": thitogene_source_metadata(upstream_root),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forward_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
