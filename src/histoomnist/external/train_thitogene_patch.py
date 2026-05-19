from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from histoomnist.external.histogene_patch_h5 import (
    HistogenePatchH5Dataset,
    target_matrix_from_counts,
)
from histoomnist.external.thitogene_model import THItoGenePatchRegressor, masked_mse
from histoomnist.train.common import checkpoint_payload, load_checkpoint, save_checkpoint
from histoomnist.utils.config import get_device_name
from histoomnist.utils.seed import set_seed


def _knn_adj(coords: np.ndarray, *, k_neighbors: int) -> np.ndarray:
    n = int(coords.shape[0])
    adj = np.eye(n, dtype=np.float32)
    if n <= 1:
        return adj
    k = min(int(k_neighbors), n - 1)
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    order = np.argsort(distances, axis=1)[:, 1 : k + 1]
    rows = np.arange(n)[:, None]
    adj[rows, order] = 1.0
    return adj


class THItoGenePatchH5ChunkDataset(Dataset):
    def __init__(
        self,
        expression_config: dict[str, Any],
        *,
        splits: list[str],
        chunk_size: int = 64,
        max_slides: int | None = None,
        max_chunks_per_slide: int | None = None,
        target_kind: str = "log1p_rate",
        n_pos: int = 64,
        k_neighbors: int = 4,
    ):
        if int(chunk_size) <= 1:
            raise ValueError("chunk_size must be greater than 1 for THItoGene graph chunks.")
        self.spot_dataset = HistogenePatchH5Dataset(
            expression_config,
            splits=splits,
            max_slides=max_slides,
            target_kind=target_kind,
        )
        self.slides = self.spot_dataset.slides
        self.target_genes = self.spot_dataset.target_genes
        self.target_kind = str(target_kind)
        self.chunk_size = int(chunk_size)
        self.n_pos = int(n_pos)
        self.k_neighbors = int(k_neighbors)
        chunks: list[tuple[int, int, int]] = []
        for slide_idx, slide in enumerate(self.slides):
            slide_chunks = [
                (slide_idx, start, min(start + self.chunk_size, slide.n_spots))
                for start in range(0, slide.n_spots, self.chunk_size)
            ]
            if max_chunks_per_slide is not None:
                slide_chunks = slide_chunks[: int(max_chunks_per_slide)]
            chunks.extend(slide_chunks)
        if not chunks:
            raise ValueError("No chunks were created for THItoGene patch-H5 dataset.")
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, object]:
        slide_idx, start, stop = self.chunks[index]
        slide = self.slides[slide_idx]
        local_indices = np.arange(start, stop, dtype=np.int64)
        patches = np.zeros((len(local_indices), 3, 224, 224), dtype=np.float32)
        with h5py.File(slide.patch_h5_path, "r") as handle:
            img = handle["img"]
            for out_idx, local_idx in enumerate(local_indices):
                patch_index = int(slide.patch_indices[int(local_idx)])
                patch = np.asarray(img[patch_index], dtype=np.float32)
                if patch.ndim != 3 or patch.shape[-1] != 3:
                    raise ValueError(f"Patch image must be HWC RGB, got {patch.shape} for {slide.sample_id}")
                patches[out_idx] = np.transpose(patch / 255.0, (2, 0, 1))
        counts = slide.counts[start:stop].toarray().astype(np.float32, copy=False)
        target = target_matrix_from_counts(
            counts,
            slide.size_factor[start:stop],
            self.target_kind,
        )
        pos_norm = slide.position_norm[start:stop]
        position_bins = np.rint(pos_norm * float(self.n_pos - 1)).clip(0, self.n_pos - 1).astype(np.int64)
        coords = slide.spatial_coords[start:stop] if slide.spatial_coords is not None else pos_norm
        adj = _knn_adj(np.asarray(coords, dtype=np.float32), k_neighbors=self.k_neighbors)
        return {
            "patches": torch.from_numpy(patches),
            "positions": torch.from_numpy(position_bins),
            self.target_kind: torch.from_numpy(target),
            "expression_mask": torch.from_numpy(slide.measured_genes),
            "adj": torch.from_numpy(adj),
            "sample_id": slide.sample_id,
            "start": int(start),
            "stop": int(stop),
            "local_indices": torch.from_numpy(local_indices),
        }


def build_thitogene_model(model_cfg: dict[str, Any], *, n_genes: int) -> THItoGenePatchRegressor:
    return THItoGenePatchRegressor(
        n_genes=n_genes,
        patch_size=int(model_cfg.get("patch_size", 112)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        transformer_heads=int(model_cfg.get("transformer_heads", 4)),
        gat_heads=int(model_cfg.get("gat_heads", 2)),
        dim=int(model_cfg.get("dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        n_pos=int(model_cfg.get("n_pos", 64)),
        caps=int(model_cfg.get("caps", 20)),
        route_dim=int(model_cfg.get("route_dim", 64)),
        gat_hidden=int(model_cfg.get("gat_hidden", 128)),
        gat_out=int(model_cfg.get("gat_out", 256)),
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
        expression_mask = batch["expression_mask"].to(device)
        adj = batch["adj"].to(device)
        with torch.set_grad_enabled(training):
            pred = model(patches, positions, adj)
            loss = masked_mse(pred, target, expression_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_thitogene_patch(
    *,
    expression_config: dict[str, Any],
    train_splits: list[str],
    val_splits: list[str],
    output_dir: str | Path,
    model_cfg: dict[str, Any],
    target_kind: str = "log1p_rate",
    epochs: int = 1,
    batch_size: int = 1,
    chunk_size: int = 64,
    k_neighbors: int = 4,
    lr: float = 1.0e-4,
    weight_decay: float = 0.0,
    device_name: str | None = None,
    max_train_slides: int | None = None,
    max_val_slides: int | None = None,
    max_train_chunks_per_slide: int | None = None,
    max_val_chunks_per_slide: int | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    if int(batch_size) != 1:
        raise ValueError("THItoGene patch adapter currently requires batch_size=1.")
    set_seed(int(seed))
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    n_pos = int(model_cfg.get("n_pos", 64))
    train_ds = THItoGenePatchH5ChunkDataset(
        expression_config,
        splits=train_splits,
        chunk_size=chunk_size,
        max_slides=max_train_slides,
        max_chunks_per_slide=max_train_chunks_per_slide,
        target_kind=target_kind,
        n_pos=n_pos,
        k_neighbors=k_neighbors,
    )
    val_ds = THItoGenePatchH5ChunkDataset(
        expression_config,
        splits=val_splits,
        chunk_size=chunk_size,
        max_slides=max_val_slides,
        max_chunks_per_slide=max_val_chunks_per_slide,
        target_kind=target_kind,
        n_pos=n_pos,
        k_neighbors=k_neighbors,
    )
    model = build_thitogene_model(model_cfg, n_genes=len(train_ds.target_genes)).to(device)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
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
                        "k_neighbors": int(k_neighbors),
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
        "batch_size": 1,
        "chunk_size": int(chunk_size),
        "k_neighbors": int(k_neighbors),
        "seed": int(seed),
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


def load_thitogene_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[THItoGenePatchRegressor, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
    model = build_thitogene_model(ckpt["config"]["model"], n_genes=int(ckpt["n_genes"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export_thitogene_patch_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    out_dir: str | Path,
    splits: list[str],
    batch_size: int = 1,
    chunk_size: int = 64,
    device_name: str | None = None,
    max_slides: int | None = None,
    max_chunks_per_slide: int | None = None,
    max_spots_per_slide: int | None = None,
) -> dict[str, Any]:
    if int(batch_size) != 1:
        raise ValueError("THItoGene patch adapter currently requires batch_size=1.")
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    model, ckpt = load_thitogene_checkpoint(checkpoint_path, device)
    target_kind = str(ckpt["config"].get("target_kind", "log1p_rate"))
    model_cfg = ckpt["config"]["model"]
    ds = THItoGenePatchH5ChunkDataset(
        expression_config,
        splits=splits,
        chunk_size=chunk_size,
        max_slides=max_slides,
        max_chunks_per_slide=max_chunks_per_slide,
        target_kind=target_kind,
        n_pos=int(model_cfg.get("n_pos", 64)),
        k_neighbors=int(ckpt["config"].get("k_neighbors", 4)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    by_slide: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["patches"].to(device),
                batch["positions"].to(device),
                batch["adj"].to(device),
            ).detach().cpu().numpy()
            sample_id = str(batch["sample_id"][0])
            values = pred.astype(np.float32, copy=False)
            if max_spots_per_slide is not None:
                current = sum(chunk.shape[0] for _, chunk in by_slide[sample_id])
                remaining = int(max_spots_per_slide) - current
                if remaining <= 0:
                    continue
                values = values[:remaining]
            if values.size:
                by_slide[sample_id].append((int(batch["start"].numpy()[0]), values))
    out = Path(out_dir)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "genes.txt").write_text("\n".join(ckpt["genes"]) + "\n", encoding="utf-8")
    expected_spots = {slide.sample_id: slide.n_spots for slide in ds.slides}
    slide_rows = []
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
