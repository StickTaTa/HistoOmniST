from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from histoomnist.data.dataset import SizeFactorDataset
from histoomnist.models.sf_model import SizeFactorRegressor
from histoomnist.train.common import checkpoint_payload, save_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import ensure_dir, read_manifest
from histoomnist.utils.seed import set_seed


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm()
    if denom.detach().item() <= 1e-8:
        return pred.new_tensor(1.0)
    return 1.0 - (pred * target).sum() / denom.clamp_min(1e-8)


def pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.05,
    max_pairs: int = 4096,
) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    n = pred.numel()
    if n < 2:
        return pred.new_tensor(0.0)
    pair_count = min(max_pairs, n * 2)
    i = torch.randint(0, n, (pair_count,), device=pred.device)
    j = torch.randint(0, n, (pair_count,), device=pred.device)
    direction = torch.sign(target[i] - target[j])
    keep = direction != 0
    if not torch.any(keep):
        return pred.new_tensor(0.0)
    violation = margin - direction[keep] * (pred[i][keep] - pred[j][keep])
    return torch.relu(violation).mean()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def upper_tail_distribution_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    top_fraction: float = 0.25,
    top_weight: float = 3.0,
) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    n = pred.numel()
    if n < 2:
        return pred.new_tensor(0.0)
    k = max(2, min(n, int(round(n * top_fraction))))
    pred_top = torch.sort(pred).values[-k:]
    target_top = torch.sort(target).values[-k:]
    if top_weight <= 1.0:
        return F.mse_loss(pred_top, target_top)
    weights = torch.linspace(1.0, float(top_weight), k, device=pred.device, dtype=pred.dtype)
    return _weighted_mean((pred_top - target_top).pow(2), weights)


def sorted_distribution_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    return F.mse_loss(torch.sort(pred).values, torch.sort(target).values)


def log_std_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    pred_std = torch.std(pred, unbiased=False).clamp_min(1e-6)
    target_std = torch.std(target, unbiased=False).clamp_min(1e-6)
    return torch.log(pred_std / target_std).pow(2)


class CompositeSFLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        training_cfg = cfg["training"]
        self.base = str(training_cfg.get("loss", "huber"))
        if self.base not in {"mse", "huber"}:
            raise ValueError(f"Unsupported loss: {self.base}")
        self.huber_delta = float(training_cfg.get("huber_delta", 0.5))
        self.pearson_weight = float(training_cfg.get("pearson_weight", 0.0))
        self.rank_weight = float(training_cfg.get("rank_weight", 0.0))
        self.rank_margin = float(training_cfg.get("rank_margin", 0.05))
        self.rank_max_pairs = int(training_cfg.get("rank_max_pairs", 4096))
        self.tail_weight = float(training_cfg.get("tail_weight", 0.0))
        self.tail_quantile = float(training_cfg.get("tail_quantile", 0.80))
        self.tail_threshold = training_cfg.get("tail_threshold")
        self.tail_scale = float(training_cfg.get("tail_scale", 0.35))
        self.tail_under_weight = float(training_cfg.get("tail_under_weight", 0.0))
        self.tail_distribution_weight = float(training_cfg.get("tail_distribution_weight", 0.0))
        self.tail_distribution_fraction = float(training_cfg.get("tail_distribution_fraction", 0.25))
        self.tail_distribution_top_weight = float(training_cfg.get("tail_distribution_top_weight", 3.0))
        self.distribution_weight = float(training_cfg.get("distribution_weight", 0.0))
        self.std_weight = float(training_cfg.get("std_weight", 0.0))
        self.sf_mse_weight = float(training_cfg.get("sf_mse_weight", 0.0))
        self.sf_tail_under_weight = float(training_cfg.get("sf_tail_under_weight", 0.0))
        self.tail_classification_weight = float(training_cfg.get("tail_classification_weight", 0.0))

    def _tail_threshold_tensor(self, target: torch.Tensor) -> torch.Tensor:
        if self.tail_threshold is None:
            return torch.quantile(target.detach().reshape(-1), self.tail_quantile)
        return target.new_tensor(float(self.tail_threshold))

    def _tail_gate(self, target: torch.Tensor) -> torch.Tensor:
        threshold = self._tail_threshold_tensor(target)
        return torch.sigmoid((target - threshold) / max(self.tail_scale, 1e-6))

    def _base_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.base == "mse":
            values = (pred - target).pow(2)
        else:
            values = F.huber_loss(pred, target, delta=self.huber_delta, reduction="none")
        if not self.tail_weight:
            return values.mean()
        weights = 1.0 + self.tail_weight * self._tail_gate(target)
        return _weighted_mean(values, weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self._base_loss(pred, target)
        if self.tail_under_weight:
            gate = self._tail_gate(target)
            under_prediction = torch.relu(target - pred).pow(2)
            loss = loss + self.tail_under_weight * _weighted_mean(under_prediction, gate)
        if self.tail_distribution_weight:
            loss = loss + self.tail_distribution_weight * upper_tail_distribution_loss(
                pred,
                target,
                top_fraction=self.tail_distribution_fraction,
                top_weight=self.tail_distribution_top_weight,
            )
        if self.distribution_weight:
            loss = loss + self.distribution_weight * sorted_distribution_loss(pred, target)
        if self.std_weight:
            loss = loss + self.std_weight * log_std_loss(pred, target)
        if self.sf_mse_weight or self.sf_tail_under_weight:
            gate = self._tail_gate(target)
            pred_sf = torch.exp(pred.clamp(min=-8.0, max=4.0))
            target_sf = torch.exp(target.clamp(min=-8.0, max=4.0))
            if self.sf_mse_weight:
                loss = loss + self.sf_mse_weight * _weighted_mean((pred_sf - target_sf).pow(2), 1.0 + gate)
            if self.sf_tail_under_weight:
                sf_under_prediction = torch.relu(target_sf - pred_sf).pow(2)
                loss = loss + self.sf_tail_under_weight * _weighted_mean(sf_under_prediction, gate)
        if self.tail_classification_weight:
            threshold = self._tail_threshold_tensor(target)
            labels = (target >= threshold).float()
            logits = (pred - threshold) / max(self.tail_scale, 1e-6)
            positives = labels.sum()
            negatives = labels.numel() - positives
            pos_weight = (negatives / positives.clamp_min(1.0)).clamp(min=1.0, max=10.0)
            class_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
            )
            loss = loss + self.tail_classification_weight * class_loss
        if self.pearson_weight:
            loss = loss + self.pearson_weight * pearson_loss(pred, target)
        if self.rank_weight:
            loss = loss + self.rank_weight * pairwise_rank_loss(
                pred,
                target,
                margin=self.rank_margin,
                max_pairs=self.rank_max_pairs,
            )
        return loss


def build_loss(cfg: dict) -> nn.Module:
    return CompositeSFLoss(cfg)


def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 1e-8:
        return 0.0
    return float(np.sum(x * y) / denom)


def validation_selection_score(pred_log_sf: np.ndarray, true_log_sf: np.ndarray, cfg: dict) -> float:
    training_cfg = cfg["training"]
    metric = str(training_cfg.get("selection_metric", "loss"))
    if metric == "loss":
        raise ValueError("validation_selection_score should not be called for selection_metric='loss'")
    if metric != "tail_score":
        raise ValueError(f"Unsupported selection_metric: {metric}")

    pred_sf = np.exp(np.asarray(pred_log_sf, dtype=np.float64).reshape(-1))
    true_sf = np.exp(np.asarray(true_log_sf, dtype=np.float64).reshape(-1))
    pred_sf = pred_sf / (pred_sf.mean() + 1e-8)
    true_sf = true_sf / (true_sf.mean() + 1e-8)
    pred_log = np.log(pred_sf + 1e-8)
    true_log = np.log(true_sf + 1e-8)

    q = float(training_cfg.get("selection_tail_quantile", 0.90))
    tail_mask = true_log >= np.quantile(true_log, q)
    tail_under = np.maximum(true_log[tail_mask] - pred_log[tail_mask], 0.0)
    tail_under_mse = float(np.mean(tail_under**2))
    tail_mae = float(np.mean(np.abs(pred_log[tail_mask] - true_log[tail_mask])))
    std_ratio = float(np.std(pred_sf) / (np.std(true_sf) + 1e-8))
    top_ratio = float(np.mean(pred_sf[tail_mask]) / (np.mean(true_sf[tail_mask]) + 1e-8))
    corr = _pearson_np(pred_log, true_log)

    return (
        float(training_cfg.get("selection_under_weight", 1.0)) * tail_under_mse
        + float(training_cfg.get("selection_tail_mae_weight", 0.25)) * tail_mae
        + float(training_cfg.get("selection_std_weight", 0.50)) * (np.log(std_ratio + 1e-8) ** 2)
        + float(training_cfg.get("selection_top_ratio_weight", 0.75)) * (np.log(top_ratio + 1e-8) ** 2)
        + float(training_cfg.get("selection_corr_weight", 0.20)) * (1.0 - corr)
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    show_progress = os.environ.get("HISTOOMNIST_TQDM", "").strip() not in {"", "0", "false", "False"}
    for batch in tqdm(loader, leave=False, disable=not show_progress):
        x = batch["features"].to(device)
        y = batch["log_sf"].to(device)
        with torch.set_grad_enabled(training):
            pred = model(x)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def build_train_loader(train_ds: SizeFactorDataset, cfg: dict) -> DataLoader:
    training_cfg = cfg["training"]
    sampler_name = str(training_cfg.get("sampler", "uniform"))
    batch_size = int(training_cfg["batch_size"])
    if sampler_name == "uniform":
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    if sampler_name in {"slide_balanced", "sample_balanced"}:
        sample_ids = np.asarray(train_ds.sample_ids).astype(str)
        _, inverse, counts = np.unique(sample_ids, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse].astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(training_cfg.get("samples_per_epoch", len(train_ds))),
            replacement=True,
        )
        return DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    raise ValueError(f"Unsupported training sampler: {sampler_name}")


def train(cfg: dict) -> Path:
    cfg = copy.deepcopy(cfg)
    set_seed(int(cfg.get("seed", 2026)))
    device = torch.device(get_device_name(cfg.get("device")))
    manifest_path = Path(cfg["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    base_dir = manifest_path.parent
    min_total_counts = float(cfg["data"].get("min_total_counts", 1.0))

    train_ds = SizeFactorDataset(
        manifest,
        base_dir=base_dir,
        splits=list(cfg["data"]["train_splits"]),
        min_total_counts=min_total_counts,
        fit_standardizer=True,
    )
    val_ds = SizeFactorDataset(
        manifest,
        base_dir=base_dir,
        splits=list(cfg["data"]["val_splits"]),
        min_total_counts=min_total_counts,
        standardizer=train_ds.standardizer,
    )
    training_cfg = cfg["training"]
    uses_tail_loss = any(
        float(training_cfg.get(name, 0.0))
        for name in (
            "tail_weight",
            "tail_under_weight",
            "tail_distribution_weight",
            "sf_mse_weight",
            "sf_tail_under_weight",
            "tail_classification_weight",
            "distribution_weight",
            "std_weight",
        )
    )
    if uses_tail_loss and training_cfg.get("tail_threshold") is None:
        q = float(training_cfg.get("tail_quantile", 0.80))
        training_cfg["tail_threshold"] = float(np.quantile(train_ds.y.reshape(-1), q))
        print(f"tail_threshold[{q:.2f}]={training_cfg['tail_threshold']:.6f}")
    input_dim = train_ds.x.shape[1]
    model = SizeFactorRegressor(
        input_dim=input_dim,
        hidden_dims=list(cfg["model"].get("hidden_dims") or []),
        dropout=float(cfg["model"].get("dropout", 0.15)),
        architecture=str(cfg["model"].get("architecture", "residual_mlp")),
        width=int(cfg["model"].get("width", 512)),
        depth=int(cfg["model"].get("depth", 4)),
    ).to(device)
    train_loader = build_train_loader(train_ds, cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    loss_fn = build_loss(cfg)
    out_dir = ensure_dir(cfg["output"]["dir"])
    best_path = out_dir / "best.pt"
    best_val = float("inf")
    bad_epochs = 0
    selection_metric = str(cfg["training"].get("selection_metric", "loss"))
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_pred, val_true = run_epoch(model, val_loader, loss_fn, device)
        val_score = (
            val_loss
            if selection_metric == "loss"
            else validation_selection_score(val_pred, val_true, cfg)
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_score={val_score:.6f}"
        )
        if val_score < best_val:
            best_val = val_score
            bad_epochs = 0
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    model,
                    cfg,
                    extra={
                        "input_dim": input_dim,
                        "model_kwargs": {
                            "hidden_dims": list(cfg["model"].get("hidden_dims") or []),
                            "dropout": float(cfg["model"].get("dropout", 0.15)),
                            "architecture": str(cfg["model"].get("architecture", "residual_mlp")),
                            "width": int(cfg["model"].get("width", 512)),
                            "depth": int(cfg["model"].get("depth", 4)),
                        },
                        "feature_mean": train_ds.standardizer.mean,
                        "feature_std": train_ds.standardizer.std,
                        "best_val_loss": val_loss,
                        "best_val_score": best_val,
                        "selection_metric": selection_metric,
                    },
                ),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["training"].get("patience", 15)):
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
