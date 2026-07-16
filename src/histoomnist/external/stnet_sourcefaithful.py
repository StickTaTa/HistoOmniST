from __future__ import annotations

import bisect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from numpy.lib.format import open_memmap
from torch import nn
from torch.utils.data import DataLoader, Dataset

from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.eval.benchmark_predictions import evaluate_prediction_bundle
from histoomnist.external.histogene_patch_h5 import (
    HistogenePatchSlide,
    load_histogene_patch_slide,
    target_matrix_from_counts,
    target_values_from_counts,
)
from histoomnist.utils.config import get_device_name
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path
from histoomnist.utils.seed import set_seed


TargetKind = Literal["log1p_rate"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STNET_UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "benchmarks" / "ST-Net"


@dataclass(frozen=True)
class STNetSourceInfo:
    upstream_root: Path
    commit: str | None
    model_reference: str
    classifier_reference: str
    training_reference: str


class STNetSpotDataset(Dataset):
    def __init__(
        self,
        *,
        slides: list[HistogenePatchSlide],
        target_genes: list[str],
        target_kind: TargetKind = "log1p_rate",
        transform: Any | None = None,
    ):
        if target_kind != "log1p_rate":
            raise ValueError("ST-Net formal adapter currently supports log1p_rate only.")
        if not slides:
            raise ValueError("STNetSpotDataset received no slides.")
        self.slides = list(slides)
        self.target_genes = list(target_genes)
        self.target_kind = target_kind
        self.transform = transform
        lengths = [slide.n_spots for slide in self.slides]
        self.cumlen = np.cumsum(lengths).astype(np.int64)

    def __len__(self) -> int:
        return int(self.cumlen[-1]) if len(self.cumlen) else 0

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slide_idx = bisect.bisect_right(self.cumlen, index)
        prev = 0 if slide_idx == 0 else int(self.cumlen[slide_idx - 1])
        return slide_idx, int(index - prev)

    def __getitem__(self, index: int) -> dict[str, Any]:
        slide_idx, local_idx = self._locate(index)
        slide = self.slides[slide_idx]
        patch_index = int(slide.patch_indices[local_idx])
        with h5py.File(slide.patch_h5_path, "r") as handle:
            patch = np.asarray(handle["img"][patch_index], dtype=np.float32)
        if patch.ndim != 3 or patch.shape[-1] != 3:
            raise ValueError(f"Patch image must be HWC RGB, got {patch.shape} for {slide.sample_id}")
        patch = np.transpose(patch / 255.0, (2, 0, 1)).astype(np.float32, copy=False)
        image = torch.from_numpy(patch)
        if self.transform is not None:
            image = self.transform(image)
        counts = slide.counts.getrow(local_idx).toarray().reshape(-1).astype(np.float32, copy=False)
        target = target_values_from_counts(counts, float(slide.size_factor[local_idx]), self.target_kind)
        return {
            "image": image,
            self.target_kind: torch.from_numpy(target),
            "expression_mask": torch.from_numpy(slide.measured_genes),
            "sample_id": slide.sample_id,
            "spot_id": slide.spot_ids[local_idx],
            "local_index": int(local_idx),
            "patch_index": patch_index,
        }

    def slide_summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sample_id": slide.sample_id,
                    "split": slide.split,
                    "organ": slide.organ,
                    "cohort": slide.cohort,
                    "disease_state": slide.disease_state,
                    "n_spots": int(slide.n_spots),
                    "n_measured_target_genes": int(slide.measured_genes.sum()),
                    "patch_h5_path": str(slide.patch_h5_path),
                }
                for slide in self.slides
            ]
        )


class STNetTrainTransform:
    """Official ST-Net-style patch augmentation plus normalization."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.reshape(3, 1, 1)
        self.std = torch.clamp(std.reshape(3, 1, 1), min=1.0e-6)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        out = image
        if torch.rand(()) < 0.5:
            out = torch.flip(out, dims=[2])
        if torch.rand(()) < 0.5:
            out = torch.flip(out, dims=[1])
        if torch.rand(()) < 0.5:
            out = torch.rot90(out, k=1, dims=[1, 2])
        return (out - self.mean) / self.std


class STNetEvalTransform:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.reshape(3, 1, 1)
        self.std = torch.clamp(std.reshape(3, 1, 1), min=1.0e-6)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


def stnet_source_metadata(upstream_root: str | Path = DEFAULT_STNET_UPSTREAM_ROOT) -> dict[str, Any]:
    root = Path(upstream_root)
    provenance_path = root / "source_provenance.json"
    provenance: dict[str, Any] = {}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {
        "upstream_root": str(root),
        "provenance": provenance,
        "source_info": {
            "model_reference": "torchvision.models.densenet121(pretrained=True)",
            "classifier_reference": "third_party/benchmarks/ST-Net/stnet/utils/nn.py::set_out_features",
            "training_reference": "third_party/benchmarks/ST-Net/stnet/cmd/run_spatial.py",
            "official_main_command": "densenet121 window=224 pretrain average batch=32 lr=1e-6 gene_n=250 norm 50 epochs",
            "hest_adapter_boundary": "HEST split/data loading, coverage95 log1p_rate target, export, evaluation, and spatial maps only",
        },
    }


def select_manifest_rows(
    expression_config: dict,
    *,
    splits: list[str],
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_slide_spots: int | None = None,
) -> pd.DataFrame:
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression config data.manifest resolved to None")
    manifest = read_manifest(manifest_path)
    rows = manifest[manifest["split"].astype(str).isin([str(x) for x in splits])].copy()
    if slide_ids is not None:
        wanted = [str(x) for x in slide_ids]
        rows = rows[rows["sample_id"].astype(str).isin(wanted)].copy()
        rows["_requested_order"] = pd.Categorical(rows["sample_id"].astype(str), categories=wanted, ordered=True)
        rows = rows.sort_values("_requested_order").drop(columns=["_requested_order"])
    if max_slide_spots is not None and "n_spots" in rows.columns:
        rows = rows[pd.to_numeric(rows["n_spots"], errors="coerce").le(int(max_slide_spots))].copy()
    if smallest_slides and "n_spots" in rows.columns:
        rows = rows.assign(_n_spots=pd.to_numeric(rows["n_spots"], errors="coerce")).sort_values("_n_spots")
        rows = rows.drop(columns=["_n_spots"])
    if max_slides is not None:
        rows = rows.head(int(max_slides)).copy()
    if rows.empty:
        raise ValueError(
            f"No manifest rows selected for splits={splits}, slide_ids={slide_ids}, max_slides={max_slides}."
        )
    return rows


def load_stnet_slides(
    expression_config: dict,
    rows: pd.DataFrame,
    *,
    target_kind: TargetKind = "log1p_rate",
) -> tuple[list[HistogenePatchSlide], list[str]]:
    if target_kind != "log1p_rate":
        raise ValueError("ST-Net formal adapter currently supports log1p_rate only.")
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression config data.manifest resolved to None")
    base_dir = manifest_path.parent
    target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
    if target_genes is None or gene_indices is not None:
        raise ValueError("ST-Net adapter requires data.gene_names_path target genes.")
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
    raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
    if raw_root is None:
        raise ValueError("Expression config paths.raw_root resolved to None")
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    slides = [
        load_histogene_patch_slide(
            row=row,
            base_dir=base_dir,
            raw_root=raw_root,
            target_genes=list(target_genes),
            gene_key=gene_key,
            raw_st_root=raw_st_root,
            min_total_counts=min_total_counts,
        )
        for row in rows.itertuples(index=False)
    ]
    return slides, list(target_genes)


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("module.", "", 1) if key.startswith("module.") else key: value for key, value in state.items()}


def build_stnet_densenet121(
    *,
    n_outputs: int,
    pretrained: bool = True,
    source_root: str | Path = DEFAULT_STNET_UPSTREAM_ROOT,
) -> nn.Module:
    if not Path(source_root).exists():
        raise FileNotFoundError(source_root)
    try:
        weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = torchvision.models.densenet121(weights=weights)
    except AttributeError:
        model = torchvision.models.densenet121(pretrained=bool(pretrained))
    inputs = int(model.classifier.in_features)
    model.classifier = nn.Linear(inputs, int(n_outputs), bias=True)
    model.classifier.weight.data.zero_()
    model.classifier.bias.data.zero_()
    return model


def estimate_patch_mean_std(
    dataset: STNetSpotDataset,
    *,
    batch_size: int,
    num_workers: int,
    max_batches: int = 11,
) -> tuple[torch.Tensor, torch.Tensor]:
    old_transform = dataset.transform
    dataset.transform = None
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers), pin_memory=False)
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    n_values = 0
    try:
        for batch_idx, batch in enumerate(loader):
            image = batch["image"].to(torch.float64)
            flat = image.transpose(0, 1).contiguous().view(3, -1)
            total += flat.sum(dim=1)
            total_sq += (flat * flat).sum(dim=1)
            n_values += int(flat.shape[1])
            if batch_idx + 1 >= int(max_batches):
                break
    finally:
        dataset.transform = old_transform
    if n_values <= 0:
        raise RuntimeError("Could not estimate ST-Net patch mean/std from an empty loader.")
    mean = (total / n_values).to(torch.float32)
    var = total_sq / n_values - total * total / (n_values * n_values)
    std = torch.sqrt(torch.clamp(var, min=1.0e-12)).to(torch.float32)
    return mean, std


def compute_train_target_mean(slides: list[HistogenePatchSlide], *, target_kind: TargetKind = "log1p_rate") -> np.ndarray:
    if not slides:
        raise ValueError("No training slides.")
    n_genes = int(slides[0].measured_genes.shape[0])
    total = np.zeros(n_genes, dtype=np.float64)
    denom = np.zeros(n_genes, dtype=np.float64)
    for slide in slides:
        mask = np.asarray(slide.measured_genes, dtype=bool)
        if not mask.any():
            continue
        denom[mask] += float(slide.n_spots)
        for start in range(0, slide.n_spots, 256):
            stop = min(start + 256, slide.n_spots)
            counts = slide.counts[start:stop].toarray().astype(np.float32, copy=False)
            target = target_matrix_from_counts(counts, slide.size_factor[start:stop], target_kind)
            total[mask] += target[:, mask].sum(axis=0, dtype=np.float64)
    mean = np.divide(total, np.maximum(denom, 1.0), out=np.zeros_like(total), where=denom > 0)
    return mean.astype(np.float32)


def masked_official_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=pred.dtype)
    err = (pred - target) * mask_f
    denom = torch.clamp(mask_f.sum(dim=1).float().mean(), min=1.0)
    return torch.sum(err * err) / denom


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_kind: TargetKind,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_mask = 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch[target_kind].to(device, non_blocking=True)
            mask = batch["expression_mask"].to(device, non_blocking=True)
            pred = model(image)
            loss = masked_official_mse_loss(pred, target, mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            mask_f = mask.to(dtype=pred.dtype)
            err = (pred.detach() - target) * mask_f
            total_loss += float(loss.detach().cpu()) * int(image.shape[0])
            total_abs += float(torch.sum(torch.abs(err)).detach().cpu())
            total_sq += float(torch.sum(err * err).detach().cpu())
            total_mask += float(torch.sum(mask_f).detach().cpu())
    denom = max(total_mask, 1.0)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "mae": total_abs / denom,
        "rmse": math.sqrt(total_sq / denom),
    }


def _predict_batch(model: nn.Module, image: torch.Tensor, *, average: bool) -> torch.Tensor:
    if not average:
        return model(image)
    variants = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot90 in [False, True]:
                x = image
                if rot90:
                    x = torch.rot90(x, k=1, dims=[2, 3])
                if vflip:
                    x = torch.flip(x, dims=[2])
                if hflip:
                    x = torch.flip(x, dims=[3])
                variants.append(x)
    stacked = torch.cat(variants, dim=0)
    pred = model(stacked)
    return pred.view(8, image.shape[0], -1).mean(dim=0)


def train_stnet_sourcefaithful(
    *,
    expression_config: dict,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    output_dir: str | Path,
    upstream_root: str | Path = DEFAULT_STNET_UPSTREAM_ROOT,
    target_kind: TargetKind = "log1p_rate",
    epochs: int = 50,
    batch_size: int = 32,
    test_batch_size: int | None = None,
    lr: float = 1.0e-6,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    patience: int | None = 6,
    min_delta: float = 0.0,
    pretrained: bool = True,
    average: bool = True,
    num_workers: int = 4,
    seed: int = 2026,
    device_name: str | None = None,
    data_parallel: bool = False,
) -> dict[str, Any]:
    set_seed(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(get_device_name(device_name))
    train_slides, target_genes = load_stnet_slides(expression_config, train_rows, target_kind=target_kind)
    val_slides, val_genes = load_stnet_slides(expression_config, val_rows, target_kind=target_kind)
    if target_genes != val_genes:
        raise ValueError("Train and validation target gene panels differ.")
    train_dataset = STNetSpotDataset(slides=train_slides, target_genes=target_genes, target_kind=target_kind)
    patch_mean, patch_std = estimate_patch_mean_std(
        train_dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )
    train_dataset.transform = STNetTrainTransform(patch_mean, patch_std)
    eval_transform = STNetEvalTransform(patch_mean, patch_std)
    val_dataset = STNetSpotDataset(
        slides=val_slides,
        target_genes=target_genes,
        target_kind=target_kind,
        transform=eval_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(test_batch_size or batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    model = build_stnet_densenet121(
        n_outputs=len(target_genes),
        pretrained=bool(pretrained),
        source_root=upstream_root,
    )
    train_mean = compute_train_target_mean(train_slides, target_kind=target_kind)
    model.classifier.bias.data = torch.from_numpy(train_mean).to(model.classifier.bias.data.dtype)
    if bool(data_parallel) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch: int | None = None
    no_improve = 0
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    source_meta = stnet_source_metadata(upstream_root)
    config = {
        "target_kind": target_kind,
        "n_genes": len(target_genes),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "test_batch_size": int(test_batch_size or batch_size),
        "lr": float(lr),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "patience": patience,
        "min_delta": float(min_delta),
        "pretrained": bool(pretrained),
        "average": bool(average),
        "num_workers": int(num_workers),
        "seed": int(seed),
        "data_parallel": bool(data_parallel),
        "source_info": source_meta["source_info"],
        "provenance": source_meta["provenance"],
    }

    def save(path: Path, epoch: int, val_metrics: dict[str, float]) -> None:
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(
            {
                "model_state": _strip_module_prefix(state),
                "optimizer_state": optimizer.state_dict(),
                "epoch": int(epoch),
                "config": config,
                "target_genes": target_genes,
                "train_mean_log1p_rate": train_mean,
                "patch_mean": patch_mean.cpu().numpy(),
                "patch_std": patch_std.cpu().numpy(),
                "val_metrics": val_metrics,
                "history": history,
            },
            path,
        )

    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            target_kind=target_kind,
            optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            target_kind=target_kind,
            optimizer=None,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "train_log.csv", index=False)
        save(last_path, epoch, val_metrics)
        if val_metrics["loss"] < best_val_loss - float(min_delta):
            best_val_loss = float(val_metrics["loss"])
            best_epoch = int(epoch)
            no_improve = 0
            save(best_path, epoch, val_metrics)
        else:
            no_improve += 1
        if patience is not None and no_improve >= int(patience):
            break
    stopped_early = best_epoch is not None and len(history) < int(epochs)
    summary = {
        "method": "stnet_sourcefaithful",
        "output_dir": str(output_dir),
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "target_kind": target_kind,
        "n_train_slides": int(len(train_slides)),
        "n_val_slides": int(len(val_slides)),
        "n_train_spots": int(sum(slide.n_spots for slide in train_slides)),
        "n_val_spots": int(sum(slide.n_spots for slide in val_slides)),
        "n_genes": int(len(target_genes)),
        "epochs": int(len(history)),
        "max_epochs": int(epochs),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": float(min_delta),
        "stopped_early": bool(stopped_early),
        "patch_mean": [float(x) for x in patch_mean.tolist()],
        "patch_std": [float(x) for x in patch_std.tolist()],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else str(device),
        "source_info": source_meta,
        "outputs": {
            "train_log": str(output_dir / "train_log.csv"),
            "best_model": str(best_path),
            "last_model": str(last_path),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    train_dataset.slide_summary_frame().to_csv(output_dir / "train_slides.csv", index=False)
    val_dataset.slide_summary_frame().to_csv(output_dir / "val_slides.csv", index=False)
    (output_dir / "genes.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    return summary


def load_stnet_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    target_genes = [str(x) for x in ckpt["target_genes"]]
    model = build_stnet_densenet121(
        n_outputs=len(target_genes),
        pretrained=False,
        source_root=DEFAULT_STNET_UPSTREAM_ROOT,
    )
    model.load_state_dict(_strip_module_prefix(ckpt["model_state"]), strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


def export_stnet_sourcefaithful_predictions(
    *,
    expression_config: dict,
    test_rows: pd.DataFrame,
    checkpoint_path: str | Path,
    prediction_root: str | Path,
    target_kind: TargetKind = "log1p_rate",
    batch_size: int = 32,
    num_workers: int = 4,
    average: bool | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    prediction_root = Path(prediction_root)
    pred_dir = prediction_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(get_device_name(device_name))
    model, ckpt = load_stnet_checkpoint(checkpoint_path, device)
    target_genes = [str(x) for x in ckpt["target_genes"]]
    if target_kind != str(ckpt["config"].get("target_kind", "log1p_rate")):
        raise ValueError("Requested target kind differs from checkpoint target kind.")
    patch_mean = torch.from_numpy(np.asarray(ckpt["patch_mean"], dtype=np.float32))
    patch_std = torch.from_numpy(np.asarray(ckpt["patch_std"], dtype=np.float32))
    eval_transform = STNetEvalTransform(patch_mean, patch_std)
    slides, loaded_genes = load_stnet_slides(expression_config, test_rows, target_kind=target_kind)
    if target_genes != loaded_genes:
        raise ValueError("Checkpoint and test target genes differ.")
    use_average = bool(ckpt["config"].get("average", True) if average is None else average)
    rows: list[dict[str, Any]] = []
    for slide in slides:
        dataset = STNetSpotDataset(
            slides=[slide],
            target_genes=target_genes,
            target_kind=target_kind,
            transform=eval_transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=device.type == "cuda",
        )
        pred_path = pred_dir / f"{slide.sample_id}_{target_kind}.npy"
        predictions = open_memmap(pred_path, mode="w+", dtype="float32", shape=(slide.n_spots, len(target_genes)))
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                pred = _predict_batch(model, image, average=use_average)
                values = pred.detach().cpu().numpy().astype(np.float32, copy=False)
                stop = offset + values.shape[0]
                predictions[offset:stop] = values
                offset = stop
        predictions.flush()
        rows.append(
            {
                "sample_id": slide.sample_id,
                "split": slide.split,
                "organ": slide.organ,
                "cohort": slide.cohort,
                "disease_state": slide.disease_state,
                "expected_spots": int(slide.n_spots),
                "n_predicted_spots": int(offset),
                "n_genes": int(len(target_genes)),
                "complete_slide_prediction": bool(offset == slide.n_spots),
                "truncated_for_smoke": False,
                "prediction_path": str(pred_path),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = prediction_root / "prediction_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    genes_path = prediction_root / "genes.txt"
    genes_path.write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    summary = {
        "method": "stnet_sourcefaithful",
        "checkpoint": str(checkpoint_path),
        "prediction_root": str(prediction_root),
        "prediction_kind": target_kind,
        "n_slides": int(len(rows)),
        "n_genes": int(len(target_genes)),
        "complete_prediction_arrays": bool(manifest["complete_slide_prediction"].all()) if not manifest.empty else False,
        "all_slide_predictions_complete": bool(manifest["complete_slide_prediction"].all()) if not manifest.empty else False,
        "benchmark_evaluable_without_truncation": bool(manifest["complete_slide_prediction"].all()) if not manifest.empty else False,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "average": bool(use_average),
        "outputs": {
            "prediction_manifest": str(manifest_path),
            "genes": str(genes_path),
            "predictions": str(pred_dir),
            "summary": str(prediction_root / "prediction_summary.json"),
        },
    }
    (prediction_root / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_stnet_predictions(
    *,
    prediction_root: str | Path,
    benchmark_out_dir: str | Path,
    expression_config: dict,
    splits: list[str] | None = None,
    slide_ids: list[str] | None = None,
    target_kind: TargetKind = "log1p_rate",
) -> dict[str, Any]:
    return evaluate_prediction_bundle(
        expression_config=expression_config,
        prediction_root=prediction_root,
        prediction_genes_path=Path(prediction_root) / "genes.txt",
        out_dir=benchmark_out_dir,
        method_name="stnet_sourcefaithful",
        prediction_kind=target_kind,
        splits=splits or ["test"],
        slide_ids=slide_ids,
        prediction_pattern=f"predictions/{{sample_id}}_{target_kind}.npy",
        oracle_smoke_test=False,
    )


def data_smoke_summary(
    *,
    expression_config: dict,
    rows: pd.DataFrame,
    output_dir: str | Path,
    upstream_root: str | Path = DEFAULT_STNET_UPSTREAM_ROOT,
    target_kind: TargetKind = "log1p_rate",
    batch_size: int = 8,
    num_workers: int = 0,
    pretrained: bool = False,
    device_name: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(get_device_name(device_name))
    slides, target_genes = load_stnet_slides(expression_config, rows, target_kind=target_kind)
    dataset = STNetSpotDataset(slides=slides, target_genes=target_genes, target_kind=target_kind)
    patch_mean, patch_std = estimate_patch_mean_std(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        max_batches=2,
    )
    dataset.transform = STNetEvalTransform(patch_mean, patch_std)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    batch = next(iter(loader))
    model = build_stnet_densenet121(n_outputs=len(target_genes), pretrained=bool(pretrained), source_root=upstream_root)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        pred = model(batch["image"].to(device))
    summary = {
        "status": "passed",
        "sample_ids": [slide.sample_id for slide in slides],
        "n_slides": int(len(slides)),
        "n_spots": int(sum(slide.n_spots for slide in slides)),
        "n_genes": int(len(target_genes)),
        "batch_shape": list(batch["image"].shape),
        "target_shape": list(batch[target_kind].shape),
        "prediction_shape": list(pred.shape),
        "patch_mean": [float(x) for x in patch_mean.tolist()],
        "patch_std": [float(x) for x in patch_std.tolist()],
        "device": str(device),
        "pretrained": bool(pretrained),
        "source_info": stnet_source_metadata(upstream_root),
    }
    (output_dir / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    STNetSpotDataset(slides=slides, target_genes=target_genes, target_kind=target_kind).slide_summary_frame().to_csv(
        output_dir / "slides.csv", index=False
    )
    return summary
