from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from histoomnist.external.istar_sourcefaithful import (
    DEFAULT_ISTAR_PREFIX_ROOT,
    IStarHESTPrefixSlideDataset,
    OfficialIStarCore,
    istar_masked_mse,
    istar_source_metadata,
)
from histoomnist.train.common import checkpoint_payload, load_checkpoint, save_checkpoint
from histoomnist.utils.config import get_device_name
from histoomnist.utils.seed import set_seed


OFFICIAL_ISTAR_MODEL_DEFAULTS = {
    "learning_rate": 1.0e-4,
    "optimizer": "Adam",
    "loss": "MSE",
    "target_normalization": "train_gene_minmax",
    "histology_feature_source": "official_iStar_embeddings_hist",
    "spot_alignment": "official_disk_patch_mean",
}


def _finite_minmax_update(
    *,
    y_min: np.ndarray,
    y_max: np.ndarray,
    target: np.ndarray,
    measured: np.ndarray,
) -> None:
    measured = measured.astype(bool, copy=False)
    if not measured.any():
        return
    values = np.asarray(target[:, measured], dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return
    masked = np.where(finite, values, np.nan)
    local_min = np.nanmin(masked, axis=0)
    local_max = np.nanmax(masked, axis=0)
    idx = np.where(measured)[0]
    keep_min = np.isfinite(local_min)
    keep_max = np.isfinite(local_max)
    y_min[idx[keep_min]] = np.minimum(y_min[idx[keep_min]], local_min[keep_min])
    y_max[idx[keep_max]] = np.maximum(y_max[idx[keep_max]], local_max[keep_max])


def fit_train_gene_minmax(dataset: IStarHESTPrefixSlideDataset) -> dict[str, Any]:
    n_genes = len(dataset.target_genes)
    y_min = np.full(n_genes, np.inf, dtype=np.float32)
    y_max = np.full(n_genes, -np.inf, dtype=np.float32)
    for idx in range(len(dataset)):
        item = dataset.load_target_only(idx)
        _finite_minmax_update(
            y_min=y_min,
            y_max=y_max,
            target=np.asarray(item[dataset.target_kind], dtype=np.float32),
            measured=np.asarray(item["expression_mask"], dtype=bool),
        )
    seen = np.isfinite(y_min) & np.isfinite(y_max)
    y_min[~seen] = 0.0
    y_max[~seen] = 1.0
    span = y_max - y_min
    span[span < 1.0e-6] = 1.0
    return {
        "kind": "train_gene_minmax",
        "min": y_min,
        "max": y_max,
        "span": span.astype(np.float32, copy=False),
        "n_seen_genes": int(seen.sum()),
    }


def _normalise_target(target: torch.Tensor, normalizer: dict[str, Any], device: torch.device) -> torch.Tensor:
    y_min = torch.as_tensor(normalizer["min"], dtype=target.dtype, device=device)
    span = torch.as_tensor(normalizer["span"], dtype=target.dtype, device=device)
    return (target - y_min.view(1, -1)) / span.view(1, -1)


def _denormalise_prediction(pred: torch.Tensor, normalizer: dict[str, Any], device: torch.device) -> torch.Tensor:
    y_min = torch.as_tensor(normalizer["min"], dtype=pred.dtype, device=device)
    span = torch.as_tensor(normalizer["span"], dtype=pred.dtype, device=device)
    return pred * span.view(1, -1) + y_min.view(1, -1)


def _spot_batch_ranges(n_spots: int, spot_batch_size: int | None) -> list[tuple[int, int]]:
    if spot_batch_size is None or int(spot_batch_size) <= 0 or int(spot_batch_size) >= int(n_spots):
        return [(0, int(n_spots))]
    step = int(spot_batch_size)
    return [(start, min(start + step, int(n_spots))) for start in range(0, int(n_spots), step)]


def _expression_mask_1d(expression_mask: torch.Tensor) -> torch.Tensor:
    if expression_mask.ndim == 2:
        expression_mask = expression_mask[0]
    return expression_mask.bool()


def _forward_mean_spot_batches(
    *,
    model: torch.nn.Module,
    features: torch.Tensor,
    spot_batch_size: int | None,
) -> torch.Tensor:
    preds = []
    for start, end in _spot_batch_ranges(int(features.shape[0]), spot_batch_size):
        preds.append(model(features[start:end]).mean(dim=-2))
    return torch.cat(preds, dim=0)


def _chunked_masked_mse_sum(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    expression_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if pred.ndim == 3:
        pred = pred.mean(dim=-2)
    if target.ndim == 3:
        target = target[0]
    mask = _expression_mask_1d(expression_mask).view(1, -1).expand_as(target)
    valid = mask & torch.isfinite(target) & torch.isfinite(pred)
    if not torch.any(valid):
        return pred.sum() * 0.0, 0
    return (pred - target).pow(2)[valid].sum(), int(valid.sum().detach().cpu())


def _run_training_slide(
    *,
    model: torch.nn.Module,
    features: torch.Tensor,
    target_norm: torch.Tensor,
    expression_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    spot_batch_size: int | None,
) -> float:
    mask = _expression_mask_1d(expression_mask).view(1, -1).expand_as(target_norm)
    total_valid = int((mask & torch.isfinite(target_norm)).sum().detach().cpu())
    if total_valid <= 0:
        raise ValueError("No valid expression values for iStar masked MSE.")
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    for start, end in _spot_batch_ranges(int(features.shape[0]), spot_batch_size):
        pred = model(features[start:end]).mean(dim=-2)
        loss_sum, n_valid = _chunked_masked_mse_sum(
            pred=pred,
            target=target_norm[start:end],
            expression_mask=expression_mask,
        )
        if n_valid > 0:
            (loss_sum / float(total_valid)).backward()
            total_loss_sum += float(loss_sum.detach().cpu())
    optimizer.step()
    return total_loss_sum / float(total_valid)


def _run_eval_slide(
    *,
    model: torch.nn.Module,
    features: torch.Tensor,
    target_norm: torch.Tensor,
    expression_mask: torch.Tensor,
    spot_batch_size: int | None,
) -> float:
    total_loss_sum = 0.0
    total_valid = 0
    with torch.no_grad():
        for start, end in _spot_batch_ranges(int(features.shape[0]), spot_batch_size):
            pred = model(features[start:end]).mean(dim=-2)
            loss_sum, n_valid = _chunked_masked_mse_sum(
                pred=pred,
                target=target_norm[start:end],
                expression_mask=expression_mask,
            )
            total_loss_sum += float(loss_sum.detach().cpu())
            total_valid += int(n_valid)
    if total_valid <= 0:
        raise ValueError("No valid expression values for iStar masked MSE.")
    return total_loss_sum / float(total_valid)


def infer_istar_feature_dim(dataset: IStarHESTPrefixSlideDataset) -> int:
    item = dataset[0]
    return int(item["features"].shape[-1])


def build_istar_sourcefaithful_model(
    *,
    n_features: int,
    n_genes: int,
    model_cfg: dict[str, Any] | None = None,
    upstream_root: str | Path | None = None,
) -> OfficialIStarCore:
    cfg = dict(OFFICIAL_ISTAR_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    return OfficialIStarCore(
        n_inp=int(n_features),
        n_out=int(n_genes),
        lr=float(cfg["learning_rate"]),
        upstream_root=upstream_root,
    )


def run_istar_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_kind: str,
    normalizer: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    spot_batch_size: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    used_spots: list[int] = []
    finite_spots: list[int] = []
    for batch in loader:
        features = batch["features"].squeeze(0).to(device)
        target = batch[target_kind].squeeze(0).to(device)
        target_norm = _normalise_target(target, normalizer, device)
        expression_mask = batch["expression_mask"].to(device)
        if training:
            assert optimizer is not None
            loss = _run_training_slide(
                model=model,
                features=features,
                target_norm=target_norm,
                expression_mask=expression_mask,
                optimizer=optimizer,
                spot_batch_size=spot_batch_size,
            )
        else:
            loss = _run_eval_slide(
                model=model,
                features=features,
                target_norm=target_norm,
                expression_mask=expression_mask,
                spot_batch_size=spot_batch_size,
            )
        losses.append(float(loss))
        used_spots.append(int(batch["n_used_spots"].item()))
        finite_spots.append(int(batch["n_finite_spots"].item()))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "n_batches": int(len(losses)),
        "mean_used_spots": float(np.mean(used_spots)) if used_spots else float("nan"),
        "max_used_spots": int(max(used_spots)) if used_spots else 0,
        "min_finite_spots": int(min(finite_spots)) if finite_spots else 0,
    }


def data_smoke_summary(
    *,
    expression_config: dict[str, Any],
    splits: list[str],
    output_dir: str | Path,
    prefix_root: str | Path = DEFAULT_ISTAR_PREFIX_ROOT,
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    target_kind: str = "log1p_rate",
    max_spots_per_slide: int | None = None,
    max_slide_spots: int | None = None,
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset = IStarHESTPrefixSlideDataset(
        expression_config,
        splits=splits,
        prefix_root=prefix_root,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_kind=target_kind,
        max_spots_per_slide=max_spots_per_slide,
        max_slide_spots=max_slide_spots,
        upstream_root=upstream_root,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slide_frame = dataset.slide_summary_frame()
    slide_frame.to_csv(out / "data_smoke_slides.csv", index=False)
    item = dataset[0]
    target = item[target_kind].numpy()
    features = item["features"].numpy()
    summary = {
        "splits": [str(x) for x in splits],
        "slide_ids": None if slide_ids is None else [str(x) for x in slide_ids],
        "prefix_root": str(prefix_root),
        "max_slides": None if max_slides is None else int(max_slides),
        "smallest_slides": bool(smallest_slides),
        "target_kind": str(target_kind),
        "max_spots_per_slide": None if max_spots_per_slide is None else int(max_spots_per_slide),
        "max_slide_spots": None if max_slide_spots is None else int(max_slide_spots),
        "n_slides": int(len(dataset)),
        "n_target_genes": int(len(dataset.target_genes)),
        "first_slide": {
            "sample_id": str(item["sample_id"]),
            "features_shape": list(features.shape),
            "target_shape": list(target.shape),
            "feature_min": float(np.nanmin(features)),
            "feature_max": float(np.nanmax(features)),
            "target_finite_fraction": float(np.mean(np.isfinite(target))),
            "n_spots": int(item["n_spots"]),
            "n_used_spots": int(item["n_used_spots"]),
            "n_finite_spots": int(item["n_finite_spots"]),
            "complete_feature_spots": bool(item["complete_feature_spots"]),
            "truncated_for_smoke": bool(item["truncated_for_smoke"]),
        },
        "source": istar_source_metadata(upstream_root),
        "outputs": {
            "slides": str(out / "data_smoke_slides.csv"),
            "summary": str(out / "data_smoke_summary.json"),
        },
    }
    (out / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train_istar_sourcefaithful(
    *,
    expression_config: dict[str, Any],
    train_splits: list[str],
    val_splits: list[str],
    output_dir: str | Path,
    prefix_root: str | Path = DEFAULT_ISTAR_PREFIX_ROOT,
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
    lr: float = 1.0e-4,
    patience: int | None = None,
    min_delta: float = 0.0,
    spot_batch_size: int | None = None,
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
        f"[istar-sourcefaithful-crossslide] resolved_device={device}"
        + (f" cuda_name={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
        flush=True,
    )
    cfg = dict(OFFICIAL_ISTAR_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    cfg["learning_rate"] = float(lr)
    train_ds = IStarHESTPrefixSlideDataset(
        expression_config,
        splits=train_splits,
        prefix_root=prefix_root,
        slide_ids=train_slide_ids,
        max_slides=max_train_slides,
        smallest_slides=smallest_train_slides,
        target_kind=target_kind,
        max_spots_per_slide=max_train_spots_per_slide,
        max_slide_spots=max_train_slide_spots,
        upstream_root=upstream_root,
    )
    val_ds = IStarHESTPrefixSlideDataset(
        expression_config,
        splits=val_splits,
        prefix_root=prefix_root,
        slide_ids=val_slide_ids,
        max_slides=max_val_slides,
        smallest_slides=smallest_val_slides,
        target_kind=target_kind,
        max_spots_per_slide=max_val_spots_per_slide,
        max_slide_spots=max_val_slide_spots,
        upstream_root=upstream_root,
    )
    normalizer = fit_train_gene_minmax(train_ds)
    n_features = infer_istar_feature_dim(train_ds)
    model = build_istar_sourcefaithful_model(
        n_features=n_features,
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
        train_stats = run_istar_epoch(
            model=model,
            loader=train_loader,
            device=device,
            target_kind=target_kind,
            normalizer=normalizer,
            optimizer=optimizer,
            spot_batch_size=spot_batch_size,
        )
        val_stats = run_istar_epoch(
            model=model,
            loader=val_loader,
            device=device,
            target_kind=target_kind,
            normalizer=normalizer,
            spot_batch_size=spot_batch_size,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_stats["loss"]),
            "val_loss": float(val_stats["loss"]),
            "train_max_used_spots": int(train_stats["max_used_spots"]),
            "val_max_used_spots": int(val_stats["max_used_spots"]),
            "train_min_finite_spots": int(train_stats["min_finite_spots"]),
            "val_min_finite_spots": int(val_stats["min_finite_spots"]),
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
                        "method": "istar_sourcefaithful_crossslide",
                        "model": cfg,
                        "target_kind": target_kind,
                        "prefix_root": str(prefix_root),
                        "train_splits": [str(x) for x in train_splits],
                        "val_splits": [str(x) for x in val_splits],
                        "source": istar_source_metadata(upstream_root),
                    },
                    extra={
                        "n_features": int(n_features),
                        "n_genes": len(train_ds.target_genes),
                        "genes": train_ds.target_genes,
                        "normalizer": {
                            "kind": str(normalizer["kind"]),
                            "min": normalizer["min"],
                            "max": normalizer["max"],
                            "span": normalizer["span"],
                            "n_seen_genes": int(normalizer["n_seen_genes"]),
                        },
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
        "method": "istar_sourcefaithful_crossslide",
        "target_kind": target_kind,
        "precision": "fp32",
        "amp": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None,
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else None,
        "source_faithful_core": True,
        "paper_faithful_cross_slide_adapter": True,
        "formal_benchmark_candidate": formal_benchmark_candidate,
        "limited_slide_scope": limited_slide_scope,
        "truncated_for_smoke": truncated,
        "model": cfg,
        "n_features": int(n_features),
        "epochs": int(len(history)),
        "max_epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "early_stopping_patience": "" if patience is None else int(patience),
        "early_stopping_min_delta": float(min_delta),
        "spot_batch_size": None if spot_batch_size is None else int(spot_batch_size),
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
        "normalizer": {
            "kind": str(normalizer["kind"]),
            "n_seen_genes": int(normalizer["n_seen_genes"]),
        },
        "history": history,
        "source": istar_source_metadata(upstream_root),
        "outputs": {
            "checkpoint": str(best_path),
            "train_slides": str(out / "train_slides.csv"),
            "val_slides": str(out / "val_slides.csv"),
            "summary": str(out / "train_summary.json"),
        },
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_istar_sourcefaithful_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[OfficialIStarCore, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
    model = build_istar_sourcefaithful_model(
        n_features=int(ckpt["n_features"]),
        n_genes=int(ckpt["n_genes"]),
        model_cfg=ckpt["config"]["model"],
        upstream_root=ckpt["config"].get("source", {}).get("upstream_root"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export_istar_sourcefaithful_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    out_dir: str | Path,
    splits: list[str],
    prefix_root: str | Path = DEFAULT_ISTAR_PREFIX_ROOT,
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_spots_per_slide: int | None = None,
    max_slide_spots: int | None = None,
    spot_batch_size: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    model, ckpt = load_istar_sourcefaithful_checkpoint(checkpoint_path, device)
    target_kind = str(ckpt["config"].get("target_kind", "log1p_rate"))
    normalizer = ckpt["normalizer"]
    ds = IStarHESTPrefixSlideDataset(
        expression_config,
        splits=splits,
        prefix_root=prefix_root,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_kind=target_kind,
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
            features = batch["features"].squeeze(0).to(device)
            pred_norm = _forward_mean_spot_batches(
                model=model,
                features=features,
                spot_batch_size=spot_batch_size,
            )
            pred = _denormalise_prediction(pred_norm, normalizer, device)
            values = pred.detach().cpu().numpy().astype(np.float32, copy=False)
            sample_id = str(batch["sample_id"][0])
            np.save(pred_dir / f"{sample_id}_{target_kind}.npy", values)
            n_spots = int(batch["n_spots"].item())
            n_used = int(batch["n_used_spots"].item())
            n_finite = int(batch["n_finite_spots"].item())
            slide_rows.append(
                {
                    "sample_id": sample_id,
                    "n_predicted_spots": int(values.shape[0]),
                    "expected_spots": n_spots,
                    "n_used_spots": n_used,
                    "n_finite_spots": n_finite,
                    "complete_slide_prediction": bool(values.shape[0] == n_spots),
                    "complete_feature_spots": bool(batch["complete_feature_spots"].item()),
                    "truncated_for_smoke": bool(n_used != n_spots),
                }
            )
            print(
                f"[istar-sourcefaithful-crossslide] predicted {sample_id}: "
                f"spots={values.shape[0]} genes={values.shape[1]}",
                flush=True,
            )
    all_complete = bool(slide_rows) and all(bool(row["complete_slide_prediction"]) for row in slide_rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "method": "istar_sourcefaithful_crossslide",
        "prediction_kind": target_kind,
        "splits": [str(x) for x in splits],
        "prefix_root": str(prefix_root),
        "n_slides": int(len(slide_rows)),
        "n_genes": int(len(ckpt["genes"])),
        "max_slide_spots": None if max_slide_spots is None else int(max_slide_spots),
        "spot_batch_size": None if spot_batch_size is None else int(spot_batch_size),
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
