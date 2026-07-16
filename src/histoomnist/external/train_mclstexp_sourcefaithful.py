from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from histoomnist.eval.benchmark_predictions import evaluate_prediction_bundle
from histoomnist.external.mclstexp_sourcefaithful import (
    DEFAULT_MCLSTEXP_UPSTREAM_ROOT,
    MCLSTExpHESTSpotDataset,
    build_official_mclstexp_model,
    mclstexp_contrastive_loss,
    mclstexp_embeddings,
    mclstexp_source_info,
)
from histoomnist.train.common import checkpoint_payload, load_checkpoint, save_checkpoint
from histoomnist.utils.config import get_device_name
from histoomnist.utils.seed import set_seed


OFFICIAL_MCLSTEXP_MODEL_DEFAULTS = {
    "encoder_name": "densenet121",
    "temperature": 1.0,
    "image_dim": 1024,
    "projection_dim": 256,
    "heads_num": 8,
    "heads_dim": 64,
    "head_layers": 2,
    "dropout": 0.0,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-3,
    "optimizer": "Adam",
    "loss": "bidirectional_contrastive_cross_entropy",
    "prediction": "topk_weighted_retrieval",
    "top_k": 200,
}


def _jsonable_source_info(upstream_root: str | Path | None = None) -> dict[str, str]:
    info = mclstexp_source_info(upstream_root)
    return {
        "upstream_root": str(info.upstream_root),
        "model_path": str(info.model_path),
        "imported_component": info.imported_component,
        "commit": info.commit,
    }


def estimate_mclstexp_trainable_parameters(*, spot_dim: int, cfg: dict[str, Any]) -> dict[str, float | int]:
    heads_num = int(cfg["heads_num"])
    heads_dim = int(cfg["heads_dim"])
    head_layers = int(cfg["head_layers"])
    projection_dim = int(cfg["projection_dim"])
    image_dim = int(cfg["image_dim"])
    inner_dim = heads_num * heads_dim
    position = 2 * 65536 * int(spot_dim)
    attention_per_layer = int(spot_dim) * inner_dim * 3 + inner_dim * int(spot_dim) + int(spot_dim)
    feedforward_per_layer = 2 * int(spot_dim) * int(spot_dim) + 2 * int(spot_dim)
    layernorm_per_layer = 4 * int(spot_dim)
    spot_encoder = head_layers * (attention_per_layer + feedforward_per_layer + layernorm_per_layer)
    spot_projection = int(spot_dim) * projection_dim + projection_dim + projection_dim * projection_dim + projection_dim
    image_projection = image_dim * projection_dim + projection_dim + projection_dim * projection_dim + projection_dim
    densenet121_approx = 7_978_856
    total = position + spot_encoder + spot_projection + image_projection + densenet121_approx
    return {
        "spot_dim": int(spot_dim),
        "position_embedding_params": int(position),
        "spot_encoder_params_approx": int(spot_encoder),
        "spot_projection_params_approx": int(spot_projection),
        "image_projection_params_approx": int(image_projection),
        "densenet121_params_approx": int(densenet121_approx),
        "total_trainable_params_approx": int(total),
        "fp32_parameter_gb_approx": float(total * 4 / 1024**3),
        "adam_parameter_plus_state_gb_approx": float(total * 12 / 1024**3),
    }


def build_mclstexp_sourcefaithful_model(
    *,
    n_genes: int,
    model_cfg: dict[str, Any] | None = None,
    upstream_root: str | Path | None = None,
) -> torch.nn.Module:
    cfg = dict(OFFICIAL_MCLSTEXP_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    return build_official_mclstexp_model(
        spot_dim=int(n_genes),
        encoder_name=str(cfg["encoder_name"]),
        temperature=float(cfg["temperature"]),
        image_dim=int(cfg["image_dim"]),
        projection_dim=int(cfg["projection_dim"]),
        heads_num=int(cfg["heads_num"]),
        heads_dim=int(cfg["heads_dim"]),
        head_layers=int(cfg["head_layers"]),
        dropout=float(cfg["dropout"]),
        upstream_root=upstream_root,
    )


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key in {"image", "position", "expression", "expression_mask", "spatial_coords", "local_index"}:
            out[key] = value.to(device) if hasattr(value, "to") else value
        else:
            out[key] = value
    return out


def run_mclstexp_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    spot_counts: list[int] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            loss = mclstexp_contrastive_loss(model, batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        count = int(batch["image"].shape[0])
        losses.append(float(loss.detach().cpu()))
        spot_counts.append(count)
    return {
        "loss": float(np.average(losses, weights=spot_counts)) if losses else float("nan"),
        "n_batches": int(len(losses)),
        "n_spots": int(sum(spot_counts)),
        "batch_size_max": int(max(spot_counts)) if spot_counts else 0,
    }


def data_smoke_summary(
    *,
    expression_config: dict[str, Any],
    splits: list[str],
    output_dir: str | Path,
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    target_gene_limit: int | None = None,
    max_spots_per_slide: int | None = None,
    max_slide_spots: int | None = None,
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset = MCLSTExpHESTSpotDataset(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_gene_limit=target_gene_limit,
        max_spots_per_slide=max_spots_per_slide,
        max_slide_spots=max_slide_spots,
        train=True,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slides = dataset.slide_summary_frame()
    slides.to_csv(out / "data_smoke_slides.csv", index=False)
    item = dataset[0]
    expr = item["expression"].numpy()
    image = item["image"].numpy()
    summary = {
        "splits": [str(x) for x in splits],
        "slide_ids": None if slide_ids is None else [str(x) for x in slide_ids],
        "max_slides": None if max_slides is None else int(max_slides),
        "smallest_slides": bool(smallest_slides),
        "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
        "max_spots_per_slide": None if max_spots_per_slide is None else int(max_spots_per_slide),
        "max_slide_spots": None if max_slide_spots is None else int(max_slide_spots),
        "n_slides": int(len(dataset.slides)),
        "n_spots": int(len(dataset)),
        "n_target_genes": int(len(dataset.target_genes)),
        "first_item": {
            "sample_id": str(item["sample_id"]),
            "spot_id": str(item["spot_id"]),
            "image_shape": list(image.shape),
            "image_min": float(np.min(image)),
            "image_max": float(np.max(image)),
            "position": [int(x) for x in item["position"].numpy().tolist()],
            "expression_shape": list(expr.shape),
            "expression_min": float(np.nanmin(expr)),
            "expression_max": float(np.nanmax(expr)),
            "expression_finite_fraction": float(np.mean(np.isfinite(expr))),
        },
        "model_parameter_estimate": estimate_mclstexp_trainable_parameters(
            spot_dim=len(dataset.target_genes),
            cfg=dict(OFFICIAL_MCLSTEXP_MODEL_DEFAULTS),
        ),
        "source": _jsonable_source_info(upstream_root),
        "outputs": {
            "slides": str(out / "data_smoke_slides.csv"),
            "summary": str(out / "data_smoke_summary.json"),
        },
    }
    (out / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train_mclstexp_sourcefaithful(
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
    target_gene_limit: int | None = None,
    max_train_spots_per_slide: int | None = None,
    max_val_spots_per_slide: int | None = None,
    max_train_slide_spots: int | None = None,
    max_val_slide_spots: int | None = None,
    epochs: int = 1,
    batch_size: int = 128,
    num_workers: int = 0,
    lr: float = 1.0e-4,
    weight_decay: float = 1.0e-3,
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
        f"[mclstexp-sourcefaithful] resolved_device={device}"
        + (f" cuda_name={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
        flush=True,
    )
    cfg = dict(OFFICIAL_MCLSTEXP_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    cfg["learning_rate"] = float(lr)
    cfg["weight_decay"] = float(weight_decay)

    train_ds = MCLSTExpHESTSpotDataset(
        expression_config,
        splits=train_splits,
        slide_ids=train_slide_ids,
        max_slides=max_train_slides,
        smallest_slides=smallest_train_slides,
        target_gene_limit=target_gene_limit,
        max_spots_per_slide=max_train_spots_per_slide,
        max_slide_spots=max_train_slide_spots,
        train=True,
    )
    val_ds = MCLSTExpHESTSpotDataset(
        expression_config,
        splits=val_splits,
        slide_ids=val_slide_ids,
        max_slides=max_val_slides,
        smallest_slides=smallest_val_slides,
        target_gene_limit=target_gene_limit,
        max_spots_per_slide=max_val_spots_per_slide,
        max_slide_spots=max_val_slide_spots,
        train=False,
    )
    model = build_mclstexp_sourcefaithful_model(
        n_genes=len(train_ds.target_genes),
        model_cfg=cfg,
        upstream_root=upstream_root,
    ).to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_ds.slide_summary_frame().to_csv(out / "train_slides.csv", index=False)
    val_ds.slide_summary_frame().to_csv(out / "val_slides.csv", index=False)
    best_path = out / "best.pt"
    history = []
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    for epoch in range(1, int(epochs) + 1):
        train_stats = run_mclstexp_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)
        val_stats = run_mclstexp_epoch(model=model, loader=val_loader, device=device)
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_stats["loss"]),
            "val_loss": float(val_stats["loss"]),
            "train_batches": int(train_stats["n_batches"]),
            "val_batches": int(val_stats["n_batches"]),
            "train_spots": int(train_stats["n_spots"]),
            "val_spots": int(val_stats["n_spots"]),
        }
        history.append(row)
        print(
            f"[mclstexp-sourcefaithful] epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f}",
            flush=True,
        )
        improved = float(val_stats["loss"]) < (best_val - float(min_delta))
        if improved:
            best_val = float(val_stats["loss"])
            best_epoch = int(epoch)
            stale_epochs = 0
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    model,
                    {
                        "method": "mclstexp_sourcefaithful",
                        "target_kind": "log1p_rate",
                        "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
                        "n_genes": int(len(train_ds.target_genes)),
                        "model_cfg": cfg,
                        "train_splits": [str(x) for x in train_splits],
                        "val_splits": [str(x) for x in val_splits],
                        "train_slide_ids": None if train_slide_ids is None else [str(x) for x in train_slide_ids],
                        "val_slide_ids": None if val_slide_ids is None else [str(x) for x in val_slide_ids],
                        "max_train_slides": None if max_train_slides is None else int(max_train_slides),
                        "max_val_slides": None if max_val_slides is None else int(max_val_slides),
                        "smallest_train_slides": bool(smallest_train_slides),
                        "smallest_val_slides": bool(smallest_val_slides),
                        "max_train_slide_spots": None if max_train_slide_spots is None else int(max_train_slide_spots),
                        "max_val_slide_spots": None if max_val_slide_spots is None else int(max_val_slide_spots),
                        "batch_size": int(batch_size),
                    },
                    extra={
                        "target_genes": train_ds.target_genes,
                        "source": _jsonable_source_info(upstream_root),
                    },
                ),
            )
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= int(patience):
                stopped_early = True
                print(f"[mclstexp-sourcefaithful] early stop at epoch={epoch}", flush=True)
                break

    if best_epoch == 0:
        save_checkpoint(
            best_path,
            checkpoint_payload(
                model,
                {
                    "method": "mclstexp_sourcefaithful",
                    "target_kind": "log1p_rate",
                    "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
                    "n_genes": int(len(train_ds.target_genes)),
                    "model_cfg": cfg,
                    "train_splits": [str(x) for x in train_splits],
                    "val_splits": [str(x) for x in val_splits],
                    "train_slide_ids": None if train_slide_ids is None else [str(x) for x in train_slide_ids],
                    "val_slide_ids": None if val_slide_ids is None else [str(x) for x in val_slide_ids],
                    "max_train_slides": None if max_train_slides is None else int(max_train_slides),
                    "max_val_slides": None if max_val_slides is None else int(max_val_slides),
                    "smallest_train_slides": bool(smallest_train_slides),
                    "smallest_val_slides": bool(smallest_val_slides),
                    "max_train_slide_spots": None if max_train_slide_spots is None else int(max_train_slide_spots),
                    "max_val_slide_spots": None if max_val_slide_spots is None else int(max_val_slide_spots),
                    "batch_size": int(batch_size),
                },
                extra={
                    "target_genes": train_ds.target_genes,
                    "source": _jsonable_source_info(upstream_root),
                },
            ),
        )
        best_epoch = len(history)
        best_val = float(history[-1]["val_loss"]) if history else float("nan")

    summary = {
        "checkpoint": str(best_path),
        "device": str(device),
        "method": "mclstexp_sourcefaithful",
        "target_kind": "log1p_rate",
        "precision": "fp32",
        "amp": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "source_faithful_core": True,
        "prediction_protocol": "topk_weighted_retrieval_over_train_spot_embeddings",
        "formal_benchmark_candidate": target_gene_limit is None,
        "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
        "model": cfg,
        "parameter_estimate": estimate_mclstexp_trainable_parameters(spot_dim=len(train_ds.target_genes), cfg=cfg),
        "epochs": int(len(history)),
        "max_epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "early_stopping_patience": None if patience is None else int(patience),
        "early_stopping_min_delta": float(min_delta),
        "stopped_early": bool(stopped_early),
        "n_train_slides": int(len(train_ds.slides)),
        "n_val_slides": int(len(val_ds.slides)),
        "n_train_spots": int(len(train_ds)),
        "n_val_spots": int(len(val_ds)),
        "n_genes": int(len(train_ds.target_genes)),
        "history": history,
        "source": _jsonable_source_info(upstream_root),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def collect_mclstexp_key_embeddings(
    *,
    model: torch.nn.Module,
    dataset: MCLSTExpHESTSpotDataset,
    out_dir: str | Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    n = len(dataset)
    n_genes = len(dataset.target_genes)
    projection_dim = int(model.image_projection.layer_norm.normalized_shape[0])
    emb_path = out / "train_spot_embeddings.npy"
    expr_path = out / "train_expression_key.npy"
    spot_emb = open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, projection_dim))
    expr_key = open_memmap(expr_path, mode="w+", dtype=np.float32, shape=(n, n_genes))
    rows = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_device = _batch_to_device(batch, device)
            _, emb = mclstexp_embeddings(model, batch_device, include_image=False, include_spot=True)
            assert emb is not None
            b = int(emb.shape[0])
            spot_emb[offset : offset + b] = emb.detach().cpu().numpy().astype(np.float32, copy=False)
            expr_key[offset : offset + b] = batch["expression"].numpy().astype(np.float32, copy=False)
            for i in range(b):
                rows.append(
                    {
                        "row_index": int(offset + i),
                        "sample_id": str(batch["sample_id"][i]),
                        "spot_id": str(batch["spot_id"][i]),
                        "local_index": int(batch["local_index"][i]),
                    }
                )
            offset += b
            print(f"[mclstexp-export] key embeddings {offset}/{n}", flush=True)
    spot_emb.flush()
    expr_key.flush()
    manifest_path = out / "train_key_index.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return {
        "spot_embeddings": str(emb_path),
        "expression_key": str(expr_path),
        "index": str(manifest_path),
        "n_key_spots": int(n),
        "n_genes": int(n_genes),
        "projection_dim": int(projection_dim),
    }


def _predict_retrieval_batch(
    *,
    query: torch.Tensor,
    train_embeddings: torch.Tensor,
    train_embeddings_norm: torch.Tensor,
    expression_key: np.ndarray,
    top_k: int,
    device: torch.device,
) -> np.ndarray:
    query_norm = F.normalize(query, p=2, dim=-1)
    sim = query_norm @ train_embeddings_norm.T
    k = min(int(top_k), int(train_embeddings_norm.shape[0]))
    _, indices = torch.topk(sim, k=k, dim=1)
    selected_embeddings = train_embeddings[indices]
    distances = torch.linalg.vector_norm(selected_embeddings - query[:, None, :], ord=2, dim=2)
    weights = torch.reciprocal(distances.square() + 1.0e-12)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    idx_np = indices.detach().cpu().numpy()
    expr = np.asarray(expression_key[idx_np.reshape(-1), :], dtype=np.float32).reshape(
        idx_np.shape[0],
        idx_np.shape[1],
        expression_key.shape[1],
    )
    expr_t = torch.from_numpy(expr).to(device)
    pred = (weights[:, :, None] * expr_t).sum(dim=1)
    return pred.detach().cpu().numpy().astype(np.float32, copy=False)


def export_mclstexp_sourcefaithful_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    out_dir: str | Path,
    splits: list[str],
    train_splits: list[str] | None = None,
    train_slide_ids: list[str] | None = None,
    max_train_slides: int | None = None,
    smallest_train_slides: bool | None = None,
    max_train_slide_spots: int | None = None,
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_train_spots_per_slide: int | None = None,
    max_predict_spots_per_slide: int | None = None,
    max_test_slide_spots: int | None = None,
    embedding_batch_size: int = 64,
    retrieval_batch_size: int = 8,
    top_k: int = 200,
    num_workers: int = 0,
    device_name: str | None = None,
    upstream_root: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    target_genes = list(ckpt.get("target_genes", []))
    if not target_genes:
        raise ValueError("Checkpoint lacks target_genes.")
    model = build_mclstexp_sourcefaithful_model(
        n_genes=len(target_genes),
        model_cfg=cfg.get("model_cfg", {}),
        upstream_root=upstream_root,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    out = Path(out_dir)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "genes.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")

    train_ds = MCLSTExpHESTSpotDataset(
        expression_config,
        splits=train_splits or list(cfg.get("train_splits", ["train"])),
        slide_ids=train_slide_ids if train_slide_ids is not None else cfg.get("train_slide_ids"),
        max_slides=max_train_slides if max_train_slides is not None else cfg.get("max_train_slides"),
        smallest_slides=(
            bool(smallest_train_slides)
            if smallest_train_slides is not None
            else bool(cfg.get("smallest_train_slides", False))
        ),
        target_gene_limit=cfg.get("target_gene_limit"),
        max_spots_per_slide=max_train_spots_per_slide,
        max_slide_spots=max_train_slide_spots
        if max_train_slide_spots is not None
        else cfg.get("max_train_slide_spots"),
        train=False,
    )
    key_dir = out / "retrieval_key"
    key_summary = collect_mclstexp_key_embeddings(
        model=model,
        dataset=train_ds,
        out_dir=key_dir,
        device=device,
        batch_size=int(embedding_batch_size),
        num_workers=int(num_workers),
    )
    train_embeddings_np = np.load(key_summary["spot_embeddings"], mmap_mode="r")
    expression_key = np.load(key_summary["expression_key"], mmap_mode="r")
    train_embeddings = torch.from_numpy(np.array(train_embeddings_np, dtype=np.float32, copy=True)).to(device)
    train_embeddings_norm = F.normalize(train_embeddings, p=2, dim=-1)

    test_index_ds = MCLSTExpHESTSpotDataset(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        target_gene_limit=cfg.get("target_gene_limit"),
        max_spots_per_slide=max_predict_spots_per_slide,
        max_slide_spots=max_test_slide_spots,
        train=False,
    )
    slide_summaries = []
    complete = True
    for slide in test_index_ds.slides:
        slide_ds = MCLSTExpHESTSpotDataset(
            expression_config,
            splits=[slide.split],
            slide_ids=[slide.sample_id],
            target_gene_limit=cfg.get("target_gene_limit"),
            max_spots_per_slide=max_predict_spots_per_slide,
            max_slide_spots=max_test_slide_spots,
            train=False,
        )
        loader = DataLoader(
            slide_ds,
            batch_size=int(retrieval_batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=device.type == "cuda",
        )
        chunks = []
        with torch.no_grad():
            for batch in loader:
                batch_device = _batch_to_device(batch, device)
                query, _ = mclstexp_embeddings(model, batch_device, include_image=True, include_spot=False)
                assert query is not None
                pred = _predict_retrieval_batch(
                    query=query,
                    train_embeddings=train_embeddings,
                    train_embeddings_norm=train_embeddings_norm,
                    expression_key=expression_key,
                    top_k=int(top_k),
                    device=device,
                )
                chunks.append(pred)
        array = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        np.save(pred_dir / f"{slide.sample_id}_log1p_rate.npy", array)
        expected = int(slide.n_spots)
        used = int(array.shape[0])
        slide_complete = used == expected and max_predict_spots_per_slide is None and max_test_slide_spots is None
        complete = complete and slide_complete
        row = {
            "sample_id": slide.sample_id,
            "n_predicted_spots": used,
            "expected_spots": expected,
            "n_genes": int(array.shape[1]),
            "complete_slide_prediction": bool(slide_complete),
            "truncated_for_smoke": not bool(slide_complete),
        }
        slide_summaries.append(row)
        print(f"[mclstexp-export] {slide.sample_id}: {used}/{expected}", flush=True)

    summary = {
        "checkpoint": str(checkpoint_path),
        "method": "mclstexp_sourcefaithful",
        "prediction_kind": "log1p_rate",
        "prediction_protocol": "topk_weighted_retrieval_over_train_spot_embeddings",
        "top_k": int(top_k),
        "n_slides": int(len(slide_summaries)),
        "n_genes": int(len(target_genes)),
        "benchmark_evaluable_without_truncation": bool(complete),
        "target_gene_limit": cfg.get("target_gene_limit"),
        "retrieval_key": key_summary,
        "slides": slide_summaries,
        "outputs": {
            "prediction_root": str(out),
            "genes": str(out / "genes.txt"),
            "predictions": str(pred_dir),
            "summary": str(out / "prediction_summary.json"),
        },
    }
    (out / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_mclstexp_predictions(
    *,
    expression_config: dict[str, Any],
    prediction_root: str | Path,
    out_dir: str | Path,
    splits: list[str],
    method_name: str = "mclstexp_sourcefaithful",
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    max_slide_spots: int | None = None,
) -> dict[str, Any]:
    return evaluate_prediction_bundle(
        expression_config=expression_config,
        prediction_root=prediction_root,
        method_name=method_name,
        prediction_kind="log1p_rate",
        out_dir=out_dir,
        splits=splits,
        prediction_genes_path=Path(prediction_root) / "genes.txt",
        slide_ids=slide_ids,
        max_slides=max_slides,
        max_slide_spots=max_slide_spots,
    )
