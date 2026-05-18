from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from histoomnist.external.histogene_model import HisToGeneChunkRegressor, masked_mse
from histoomnist.external.histogene_patch_h5 import HistogenePatchH5ChunkDataset
from histoomnist.train.common import checkpoint_payload, load_checkpoint, save_checkpoint
from histoomnist.utils.config import get_device_name


def build_histogene_model(model_cfg: dict[str, Any], *, n_genes: int) -> HisToGeneChunkRegressor:
    return HisToGeneChunkRegressor(
        n_genes=n_genes,
        patch_size=int(model_cfg.get("patch_size", 56)),
        dim=int(model_cfg.get("dim", 128)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        n_pos=int(model_cfg.get("n_pos", 64)),
    )


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_kind: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for batch in loader:
        patches = batch["patches"].to(device)
        positions = batch["positions"].to(device)
        target = batch[target_kind].to(device)
        spot_mask = batch["spot_mask"].to(device)
        expression_mask = batch["expression_mask"].to(device)
        with torch.set_grad_enabled(training):
            pred = model(patches, positions, spot_mask=spot_mask)
            loss = masked_mse(pred, target, spot_mask=spot_mask, expression_mask=expression_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_histogene_patch(
    *,
    expression_config: dict[str, Any],
    train_splits: list[str],
    val_splits: list[str],
    output_dir: str | Path,
    model_cfg: dict[str, Any],
    target_kind: str = "log1p_rate",
    epochs: int = 1,
    batch_size: int = 1,
    chunk_size: int = 32,
    lr: float = 1.0e-4,
    weight_decay: float = 0.0,
    device_name: str | None = None,
    max_train_slides: int | None = None,
    max_val_slides: int | None = None,
    max_train_chunks_per_slide: int | None = None,
    max_val_chunks_per_slide: int | None = None,
) -> dict[str, Any]:
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    train_ds = HistogenePatchH5ChunkDataset(
        expression_config,
        splits=train_splits,
        chunk_size=chunk_size,
        max_slides=max_train_slides,
        max_chunks_per_slide=max_train_chunks_per_slide,
        target_kind=target_kind,
    )
    val_ds = HistogenePatchH5ChunkDataset(
        expression_config,
        splits=val_splits,
        chunk_size=chunk_size,
        max_slides=max_val_slides,
        max_chunks_per_slide=max_val_chunks_per_slide,
        target_kind=target_kind,
    )
    model = build_histogene_model(model_cfg, n_genes=len(train_ds.target_genes)).to(device)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history = []
    best_val = float("inf")
    for epoch in range(1, int(epochs) + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            target_kind=target_kind,
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            target_kind=target_kind,
        )
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
                        "target_kind": target_kind,
                        "chunk_size": int(chunk_size),
                        "train_splits": train_splits,
                        "val_splits": val_splits,
                    },
                    extra={
                        "n_genes": len(train_ds.target_genes),
                        "genes": train_ds.target_genes,
                        "best_val_loss": best_val,
                        "history": history,
                    },
                ),
            )
    summary = {
        "checkpoint": str(best_path),
        "device": str(device),
        "target_kind": target_kind,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "chunk_size": int(chunk_size),
        "train_splits": train_splits,
        "val_splits": val_splits,
        "n_train_slides": int(len(train_ds.slides)),
        "n_val_slides": int(len(val_ds.slides)),
        "n_train_chunks": int(len(train_ds)),
        "n_val_chunks": int(len(val_ds)),
        "n_genes": int(len(train_ds.target_genes)),
        "best_val_loss": float(best_val),
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_histogene_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[HisToGeneChunkRegressor, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
    model = build_histogene_model(ckpt["config"]["model"], n_genes=int(ckpt["n_genes"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export_histogene_patch_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    out_dir: str | Path,
    splits: list[str],
    batch_size: int = 1,
    chunk_size: int = 32,
    device_name: str | None = None,
    max_slides: int | None = None,
    max_chunks_per_slide: int | None = None,
    max_spots_per_slide: int | None = None,
) -> dict[str, Any]:
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    model, ckpt = load_histogene_checkpoint(checkpoint_path, device)
    target_kind = str(ckpt["config"].get("target_kind", "log1p_rate"))
    ds = HistogenePatchH5ChunkDataset(
        expression_config,
        splits=splits,
        chunk_size=chunk_size,
        max_slides=max_slides,
        max_chunks_per_slide=max_chunks_per_slide,
        target_kind=target_kind,
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    by_slide: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["patches"].to(device),
                batch["positions"].to(device),
                spot_mask=batch["spot_mask"].to(device),
            ).detach().cpu().numpy()
            spot_mask = batch["spot_mask"].numpy().astype(bool)
            sample_ids = list(batch["sample_id"])
            starts = batch["start"].numpy().astype(np.int64)
            for idx, sample_id in enumerate(sample_ids):
                values = pred[idx, spot_mask[idx]]
                if max_spots_per_slide is not None:
                    current = sum(chunk.shape[0] for _, chunk in by_slide[str(sample_id)])
                    remaining = int(max_spots_per_slide) - current
                    if remaining <= 0:
                        continue
                    values = values[:remaining]
                if values.size:
                    by_slide[str(sample_id)].append((int(starts[idx]), values.astype(np.float32, copy=False)))
    out = Path(out_dir)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "genes.txt").write_text("\n".join(ckpt["genes"]) + "\n", encoding="utf-8")
    slide_rows = []
    expected_spots = {slide.sample_id: slide.n_spots for slide in ds.slides}
    for sample_id, chunks in sorted(by_slide.items()):
        ordered = [chunk for _, chunk in sorted(chunks, key=lambda item: item[0])]
        array = np.concatenate(ordered, axis=0).astype(np.float32, copy=False)
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
        "max_chunks_per_slide": None if max_chunks_per_slide is None else int(max_chunks_per_slide),
        "max_spots_per_slide": None if max_spots_per_slide is None else int(max_spots_per_slide),
        "all_slide_predictions_complete": all_complete,
        "benchmark_evaluable_without_truncation": bool(
            all_complete and max_chunks_per_slide is None and max_spots_per_slide is None
        ),
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
