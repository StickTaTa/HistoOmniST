from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from histoomnist.external.hist2st_sourcefaithful import (
    Hist2STHESTSlideDataset,
    adjacency_stats,
    hist2st_masked_mse,
    hist2st_source_metadata,
    hist2st_zinb_or_nb_loss,
    load_official_hist2st_components,
)
from histoomnist.train.common import checkpoint_payload, load_checkpoint, save_checkpoint
from histoomnist.utils.config import get_device_name
from histoomnist.utils.seed import set_seed


OFFICIAL_HIST2ST_MODEL_DEFAULTS = {
    "fig_size": 112,
    "kernel_size": 5,
    "patch_size": 7,
    "depth1": 2,
    "depth2": 8,
    "depth3": 4,
    "heads": 16,
    "channel": 32,
    "dropout": 0.2,
    "n_pos": 64,
    "learning_rate": 1.0e-5,
    "zinb": 0.25,
    "nb": False,
    "bake": 5,
    "lamb": 0.5,
    "policy": "mean",
    "optimizer": "Adam",
    "loss": "MSE + zinb * ZINB + lamb * self-distillation",
    "tag": "5-7-2-8-4-16-32",
}


def build_hist2st_sourcefaithful_model(
    *,
    n_genes: int,
    model_cfg: dict[str, Any] | None = None,
    upstream_root: str | Path | None = None,
) -> torch.nn.Module:
    cfg = dict(OFFICIAL_HIST2ST_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    expected_tag = (
        f"{int(cfg['kernel_size'])}-{int(cfg['patch_size'])}-{int(cfg['depth1'])}-"
        f"{int(cfg['depth2'])}-{int(cfg['depth3'])}-{int(cfg['heads'])}-{int(cfg['channel'])}"
    )
    if expected_tag != str(cfg.get("tag", expected_tag)):
        raise ValueError(f"Hist2ST model config does not match official tag: {expected_tag} vs {cfg.get('tag')}")
    components = load_official_hist2st_components(upstream_root)
    cls = components["Hist2ST"]
    return cls(
        learning_rate=float(cfg["learning_rate"]),
        fig_size=int(cfg["fig_size"]),
        label=None,
        dropout=float(cfg["dropout"]),
        n_pos=int(cfg["n_pos"]),
        kernel_size=int(cfg["kernel_size"]),
        patch_size=int(cfg["patch_size"]),
        n_genes=int(n_genes),
        depth1=int(cfg["depth1"]),
        depth2=int(cfg["depth2"]),
        depth3=int(cfg["depth3"]),
        heads=int(cfg["heads"]),
        channel=int(cfg["channel"]),
        zinb=float(cfg["zinb"]),
        nb=bool(cfg["nb"]),
        bake=int(cfg["bake"]),
        lamb=float(cfg["lamb"]),
        policy=str(cfg["policy"]),
    )


def _batch_to_device(batch: dict[str, Any], device: torch.device, target_kind: str) -> dict[str, Any]:
    out = dict(batch)
    for key in ["patches", "positions", "adj", target_kind, "raw_counts", "size_factors", "expression_mask"]:
        out[key] = batch[key].to(device)
    return out


def _hist2st_loss(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    target_kind: str,
    components: dict[str, Any],
    model_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    patches = batch["patches"]
    positions = batch["positions"]
    adj = batch["adj"].squeeze(0)
    target = batch[target_kind]
    expression_mask = batch["expression_mask"]
    pred, extra, _ = model(patches, positions, adj)
    mse_loss = hist2st_masked_mse(pred, target, expression_mask=expression_mask)
    bake_loss = pred.new_tensor(0.0)
    if int(model_cfg["bake"]) > 0:
        bake_pred = model.distillation(model.aug(patches, positions, adj))
        bake_loss = torch.nn.functional.mse_loss(bake_pred, pred)
    zinb_loss = pred.new_tensor(0.0)
    if float(model_cfg["zinb"]) > 0:
        zinb_loss = hist2st_zinb_or_nb_loss(
            extra=extra,
            raw_counts=batch["raw_counts"],
            size_factors=batch["size_factors"],
            expression_mask=expression_mask,
            nb=bool(model_cfg["nb"]),
            components=components,
        )
    loss = mse_loss + float(model_cfg["zinb"]) * zinb_loss + float(model_cfg["lamb"]) * bake_loss
    stats = {
        "loss": float(loss.detach().cpu()),
        "mse_loss": float(mse_loss.detach().cpu()),
        "zinb_loss": float(zinb_loss.detach().cpu()),
        "bake_loss": float(bake_loss.detach().cpu()),
    }
    return loss, stats


def run_hist2st_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_kind: str,
    components: dict[str, Any],
    model_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    mse_losses: list[float] = []
    zinb_losses: list[float] = []
    bake_losses: list[float] = []
    used_spots: list[int] = []
    zero_degree_nodes: list[int] = []
    for batch in loader:
        batch = _batch_to_device(batch, device, target_kind)
        adj_stats = adjacency_stats(batch["adj"].squeeze(0))
        zero_degree_nodes.append(int(adj_stats["adj_zero_degree_nodes"]))
        with torch.set_grad_enabled(training):
            loss, stats = _hist2st_loss(
                model=model,
                batch=batch,
                target_kind=target_kind,
                components=components,
                model_cfg=model_cfg,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(stats["loss"])
        mse_losses.append(stats["mse_loss"])
        zinb_losses.append(stats["zinb_loss"])
        bake_losses.append(stats["bake_loss"])
        used_spots.extend(int(x) for x in batch["n_used_spots"].detach().cpu().numpy())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mse_loss": float(np.mean(mse_losses)) if mse_losses else float("nan"),
        "zinb_loss": float(np.mean(zinb_losses)) if zinb_losses else float("nan"),
        "bake_loss": float(np.mean(bake_losses)) if bake_losses else float("nan"),
        "n_batches": int(len(losses)),
        "mean_used_spots": float(np.mean(used_spots)) if used_spots else float("nan"),
        "max_used_spots": int(max(used_spots)) if used_spots else 0,
        "max_zero_degree_nodes": int(max(zero_degree_nodes)) if zero_degree_nodes else 0,
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
    fig_size: int = 112,
    n_pos: int = 64,
    k_neighbors: int = 4,
    prune: str = "NA",
    graph_coord_source: str = "position_grid",
    max_spots_per_slide: int | None = None,
    max_slide_spots: int | None = None,
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset = Hist2STHESTSlideDataset(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_kind=target_kind,
        fig_size=fig_size,
        n_pos=n_pos,
        k_neighbors=k_neighbors,
        prune=prune,
        graph_coord_source=graph_coord_source,  # type: ignore[arg-type]
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
    summary = {
        "splits": [str(x) for x in splits],
        "slide_ids": None if slide_ids is None else [str(x) for x in slide_ids],
        "max_slides": None if max_slides is None else int(max_slides),
        "smallest_slides": bool(smallest_slides),
        "target_kind": str(target_kind),
        "fig_size": int(fig_size),
        "n_pos": int(n_pos),
        "k_neighbors": int(k_neighbors),
        "prune": str(prune),
        "graph_coord_source": str(graph_coord_source),
        "max_spots_per_slide": None if max_spots_per_slide is None else int(max_spots_per_slide),
        "max_slide_spots": None if max_slide_spots is None else int(max_slide_spots),
        "n_slides": int(len(dataset)),
        "n_target_genes": int(len(dataset.target_genes)),
        "first_slide": {
            "sample_id": str(item["sample_id"]),
            "patches_shape": list(patches.shape),
            "positions_shape": list(positions.shape),
            "target_shape": list(target.shape),
            "raw_counts_shape": list(item["raw_counts"].shape),
            "size_factors_shape": list(item["size_factors"].shape),
            **adjacency_stats(adj),
            "patch_value_min": float(np.min(patches)),
            "patch_value_max": float(np.max(patches)),
            "positions_min": int(np.min(positions)),
            "positions_max": int(np.max(positions)),
            "target_finite_fraction_measured": float(np.mean(np.isfinite(target[:, mask]))),
            "n_spots": int(item["n_spots"]),
            "n_used_spots": int(item["n_used_spots"]),
            "truncated_for_smoke": bool(item["truncated_for_smoke"]),
        },
        "source": hist2st_source_metadata(upstream_root),
        "outputs": {
            "slides": str(out / "data_smoke_slides.csv"),
            "summary": str(out / "data_smoke_summary.json"),
        },
    }
    (out / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train_hist2st_sourcefaithful(
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
    k_neighbors: int = 4,
    prune: str = "NA",
    graph_coord_source: str = "position_grid",
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(int(seed))
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    print(
        f"[hist2st-sourcefaithful] resolved_device={device}"
        + (f" cuda_name={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
        flush=True,
    )
    cfg = dict(OFFICIAL_HIST2ST_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    cfg["learning_rate"] = float(lr)
    components = load_official_hist2st_components(upstream_root)
    train_ds = Hist2STHESTSlideDataset(
        expression_config,
        splits=train_splits,
        slide_ids=train_slide_ids,
        max_slides=max_train_slides,
        smallest_slides=smallest_train_slides,
        target_kind=target_kind,
        fig_size=int(cfg["fig_size"]),
        n_pos=int(cfg["n_pos"]),
        k_neighbors=int(k_neighbors),
        prune=prune,
        graph_coord_source=graph_coord_source,  # type: ignore[arg-type]
        max_spots_per_slide=max_train_spots_per_slide,
        max_slide_spots=max_train_slide_spots,
        upstream_root=upstream_root,
    )
    val_ds = Hist2STHESTSlideDataset(
        expression_config,
        splits=val_splits,
        slide_ids=val_slide_ids,
        max_slides=max_val_slides,
        smallest_slides=smallest_val_slides,
        target_kind=target_kind,
        fig_size=int(cfg["fig_size"]),
        n_pos=int(cfg["n_pos"]),
        k_neighbors=int(k_neighbors),
        prune=prune,
        graph_coord_source=graph_coord_source,  # type: ignore[arg-type]
        max_spots_per_slide=max_val_spots_per_slide,
        max_slide_spots=max_val_slide_spots,
        upstream_root=upstream_root,
    )
    model = build_hist2st_sourcefaithful_model(
        n_genes=len(train_ds.target_genes),
        model_cfg=cfg,
        upstream_root=upstream_root,
    ).to(device)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
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
        train_stats = run_hist2st_epoch(
            model=model,
            loader=train_loader,
            device=device,
            target_kind=target_kind,
            components=components,
            model_cfg=cfg,
            optimizer=optimizer,
        )
        val_stats = run_hist2st_epoch(
            model=model,
            loader=val_loader,
            device=device,
            target_kind=target_kind,
            components=components,
            model_cfg=cfg,
        )
        scheduler.step()
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_stats["loss"]),
            "train_mse_loss": float(train_stats["mse_loss"]),
            "train_zinb_loss": float(train_stats["zinb_loss"]),
            "train_bake_loss": float(train_stats["bake_loss"]),
            "val_loss": float(val_stats["loss"]),
            "val_mse_loss": float(val_stats["mse_loss"]),
            "val_zinb_loss": float(val_stats["zinb_loss"]),
            "val_bake_loss": float(val_stats["bake_loss"]),
            "train_max_used_spots": int(train_stats["max_used_spots"]),
            "val_max_used_spots": int(val_stats["max_used_spots"]),
            "train_max_zero_degree_nodes": int(train_stats["max_zero_degree_nodes"]),
            "val_max_zero_degree_nodes": int(val_stats["max_zero_degree_nodes"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} "
            f"train_zero_deg={row['train_max_zero_degree_nodes']} "
            f"val_zero_deg={row['val_max_zero_degree_nodes']}",
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
                        "method": "hist2st_sourcefaithful",
                        "model": cfg,
                        "target_kind": target_kind,
                        "train_splits": [str(x) for x in train_splits],
                        "val_splits": [str(x) for x in val_splits],
                        "k_neighbors": int(k_neighbors),
                        "prune": str(prune),
                        "graph_coord_source": str(graph_coord_source),
                        "source": hist2st_source_metadata(upstream_root),
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
        "method": "hist2st_sourcefaithful",
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
        "graph": {
            "k_neighbors": int(k_neighbors),
            "prune": str(prune),
            "graph_coord_source": str(graph_coord_source),
        },
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
        "n_train_slides": int(len(train_ds)),
        "n_val_slides": int(len(val_ds)),
        "n_genes": int(len(train_ds.target_genes)),
        "history": history,
        "source": hist2st_source_metadata(upstream_root),
        "outputs": {
            "checkpoint": str(best_path),
            "train_slides": str(out / "train_slides.csv"),
            "val_slides": str(out / "val_slides.csv"),
            "summary": str(out / "train_summary.json"),
        },
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_hist2st_sourcefaithful_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
    model = build_hist2st_sourcefaithful_model(
        n_genes=int(ckpt["n_genes"]),
        model_cfg=ckpt["config"]["model"],
        upstream_root=ckpt["config"].get("source", {}).get("upstream_root"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export_hist2st_sourcefaithful_predictions(
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
    model, ckpt = load_hist2st_sourcefaithful_checkpoint(checkpoint_path, device)
    target_kind = str(ckpt["config"].get("target_kind", "log1p_rate"))
    model_cfg = dict(ckpt["config"]["model"])
    ds = Hist2STHESTSlideDataset(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_kind=target_kind,
        fig_size=int(model_cfg["fig_size"]),
        n_pos=int(model_cfg["n_pos"]),
        k_neighbors=int(ckpt["config"].get("k_neighbors", 4)),
        prune=str(ckpt["config"].get("prune", "NA")),
        graph_coord_source=str(ckpt["config"].get("graph_coord_source", "position_grid")),  # type: ignore[arg-type]
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
            pred, _, _ = model(
                batch["patches"].to(device),
                batch["positions"].to(device),
                batch["adj"].to(device).squeeze(0),
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
                f"[hist2st-sourcefaithful] predicted {sample_id}: "
                f"spots={values.shape[0]} genes={values.shape[1]}",
                flush=True,
            )
    all_complete = bool(slide_rows) and all(bool(row["complete_slide_prediction"]) for row in slide_rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "method": "hist2st_sourcefaithful",
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
    k_neighbors: int = 4,
    prune: str = "NA",
    graph_coord_source: str = "position_grid",
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(int(seed))
    cfg = dict(OFFICIAL_HIST2ST_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    components = load_official_hist2st_components(upstream_root)
    dataset = Hist2STHESTSlideDataset(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_kind=target_kind,
        fig_size=int(cfg["fig_size"]),
        n_pos=int(cfg["n_pos"]),
        k_neighbors=int(k_neighbors),
        prune=prune,
        graph_coord_source=graph_coord_source,  # type: ignore[arg-type]
        upstream_root=upstream_root,
    )
    item = dataset[0]
    model = build_hist2st_sourcefaithful_model(
        n_genes=len(dataset.target_genes),
        model_cfg=cfg,
        upstream_root=upstream_root,
    ).to(device)
    model.train()
    batch = {
        "patches": item["patches"].unsqueeze(0).to(device),
        "positions": item["positions"].unsqueeze(0).to(device),
        "adj": item["adj"].unsqueeze(0).to(device),
        target_kind: item[target_kind].unsqueeze(0).to(device),
        "raw_counts": item["raw_counts"].unsqueeze(0).to(device),
        "size_factors": item["size_factors"].unsqueeze(0).to(device),
        "expression_mask": item["expression_mask"].unsqueeze(0).to(device),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss, loss_stats = _hist2st_loss(
        model=model,
        batch=batch,
        target_kind=target_kind,
        components=components,
        model_cfg=cfg,
    )
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
        "adjacency": adjacency_stats(item["adj"]),
        "target_shape": list(item[target_kind].shape),
        "loss_stats": loss_stats,
        "model": cfg,
        "graph": {
            "k_neighbors": int(k_neighbors),
            "prune": str(prune),
            "graph_coord_source": str(graph_coord_source),
        },
        "memory": memory,
        "source": hist2st_source_metadata(upstream_root),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forward_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def summarize_prediction_slides(prediction_summary_path: str | Path) -> pd.DataFrame:
    summary = json.loads(Path(prediction_summary_path).read_text(encoding="utf-8"))
    by_flag: dict[str, int] = defaultdict(int)
    for row in summary.get("slides", []):
        by_flag[str(bool(row.get("complete_slide_prediction", False)))] += 1
    return pd.DataFrame([{"complete_slide_prediction": key, "n_slides": value} for key, value in by_flag.items()])
