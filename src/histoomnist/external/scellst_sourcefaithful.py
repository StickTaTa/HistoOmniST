from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import ConcatDataset, DataLoader

from histoomnist.data.gene_selection import (
    gene_key_settings_from_config,
    selected_genes_from_config,
)
from histoomnist.eval.benchmark_predictions import evaluate_prediction_bundle, load_slide_target
from histoomnist.utils.config import get_device_name
from histoomnist.utils.io import read_manifest
from histoomnist.utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCELLST_UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "benchmarks" / "sCellST"
DEFAULT_SCELLST_EMBEDDING_TAG = "imagenet-rn50_train"
DEFAULT_SCELLST_SHAPE_NAME = "cellvit"


@dataclass(frozen=True)
class SCellSTSourceInfo:
    upstream_root: Path
    commit: str
    dataset_path: Path
    model_path: Path
    predictor_path: Path
    training_core: str


def ensure_scellst_on_path(upstream_root: str | Path | None = None) -> Path:
    root = Path(upstream_root or DEFAULT_SCELLST_UPSTREAM_ROOT).resolve()
    if not root.exists():
        raise FileNotFoundError(f"sCellST upstream root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def scellst_source_info(upstream_root: str | Path | None = None) -> SCellSTSourceInfo:
    root = ensure_scellst_on_path(upstream_root)
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    return SCellSTSourceInfo(
        upstream_root=root,
        commit=commit,
        dataset_path=root / "scellst" / "dataset" / "torch_dataset.py",
        model_path=root / "scellst" / "model" / "instance_mil_model.py",
        predictor_path=root / "scellst" / "module" / "gene_predictor.py",
        training_core="official GenePredictor + InstanceMilModel + EmbeddedMilDataset; HistoOmniST adapter handles HEST target/split/export only",
    )


def _jsonable_source_info(upstream_root: str | Path | None = None) -> dict[str, str]:
    info = scellst_source_info(upstream_root)
    return {
        "upstream_root": str(info.upstream_root),
        "commit": info.commit,
        "dataset_path": str(info.dataset_path),
        "model_path": str(info.model_path),
        "predictor_path": str(info.predictor_path),
        "training_core": info.training_core,
    }


def _manifest_path(expression_config: dict[str, Any]) -> Path:
    return Path(expression_config["data"]["manifest"])


def target_genes_from_config(
    expression_config: dict[str, Any],
    *,
    target_gene_limit: int | None = None,
) -> list[str]:
    manifest_path = _manifest_path(expression_config)
    genes, gene_indices = selected_genes_from_config(expression_config, base_dir=manifest_path.parent)
    if genes is None or gene_indices is not None:
        raise ValueError("sCellST source-faithful benchmark requires data.gene_names_path.")
    if target_gene_limit is not None:
        genes = genes[: int(target_gene_limit)]
    return [str(g) for g in genes]


def select_manifest_rows(
    expression_config: dict[str, Any],
    *,
    splits: Iterable[str],
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_slide_spots: int | None = None,
) -> pd.DataFrame:
    manifest = read_manifest(_manifest_path(expression_config))
    selected = manifest[manifest["split"].astype(str).isin([str(x) for x in splits])].copy()
    if slide_ids:
        wanted = {str(x) for x in slide_ids}
        selected = selected[selected["sample_id"].astype(str).isin(wanted)].copy()
    if max_slide_spots is not None:
        selected = selected[selected["n_spots"].astype(int) <= int(max_slide_spots)].copy()
    if smallest_slides:
        selected = selected.sort_values(["n_spots", "sample_id"], ascending=[True, True]).copy()
    if max_slides is not None:
        selected = selected.head(int(max_slides)).copy()
    if selected.empty:
        raise ValueError(f"No slides selected for splits={list(splits)} slide_ids={slide_ids}")
    return selected.reset_index(drop=True)


def _slide_ids(rows: pd.DataFrame) -> list[str]:
    return [str(x) for x in rows["sample_id"].tolist()]


def download_hest_segmentation_assets(
    *,
    raw_root: str | Path,
    slide_ids: list[str],
    output_dir: str | Path,
    repo_id: str = "MahmoodLab/hest",
    strict: bool = True,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    root = Path(raw_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for sid in slide_ids:
        for asset_type, rel in [
            ("cellvit_seg", f"cellvit_seg/{sid}_cellvit_seg.parquet"),
            ("tissue_contours", f"tissue_seg/{sid}_contours.geojson"),
        ]:
            local_path = root / rel
            status = "exists" if local_path.exists() else "downloaded"
            try:
                if not local_path.exists():
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=rel,
                        local_dir=str(root),
                    )
                rows.append(
                    {
                        "sample_id": sid,
                        "asset_type": asset_type,
                        "relative_path": rel,
                        "local_path": str(local_path),
                        "status": status,
                        "size_bytes": int(local_path.stat().st_size) if local_path.exists() else 0,
                    }
                )
            except Exception as exc:
                msg = f"{sid} {rel}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                rows.append(
                    {
                        "sample_id": sid,
                        "asset_type": asset_type,
                        "relative_path": rel,
                        "local_path": str(local_path),
                        "status": "failed",
                        "error": msg,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "segmentation_asset_downloads.csv", index=False)
    summary = {
        "repo_id": repo_id,
        "raw_root": str(root),
        "n_slides": int(len(slide_ids)),
        "n_assets": int(len(rows)),
        "n_failed": int(len(errors)),
        "errors": errors[:20],
        "outputs": {"asset_table": str(out / "segmentation_asset_downloads.csv")},
    }
    (out / "segmentation_asset_downloads_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if strict and errors:
        raise RuntimeError("Failed to download required sCellST segmentation assets:\n" + "\n".join(errors[:20]))
    return summary


def _prepare_single_scellst_h5ad(
    *,
    expression_config: dict[str, Any],
    row: Any,
    target_genes: list[str],
    data_dir: Path,
) -> dict[str, Any]:
    import anndata as ad
    import scanpy as sc

    manifest_path = _manifest_path(expression_config)
    base_dir = manifest_path.parent
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    sample_id = str(row.sample_id)
    out_path = data_dir / "st" / f"{sample_id}.h5ad"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = sc.read_h5ad(out_path, backed="r")
        try:
            existing_genes = [str(x) for x in existing.var_names.tolist()]
            if existing_genes == target_genes:
                return {
                    "sample_id": sample_id,
                    "status": "exists",
                    "path": str(out_path),
                    "n_spots": int(existing.n_obs),
                    "n_genes": int(existing.n_vars),
                }
        finally:
            existing.file.close()

    target = load_slide_target(
        row=row,
        base_dir=base_dir,
        target_genes=target_genes,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
        min_total_counts=float(expression_config["data"].get("min_total_counts", 1.0)),
    )
    counts = target.counts.astype(np.float32).tocsr()
    inv_sf = np.reciprocal(np.clip(target.size_factor.astype(np.float32), 1.0e-6, None))
    log1p_rate = counts.multiply(inv_sf[:, None]).tocsr()
    log1p_rate.data = np.log1p(log1p_rate.data).astype(np.float32, copy=False)

    if raw_st_root is None:
        raw_st_root = Path(expression_config.get("paths", {}).get("raw_root", "data/HEST-1k/raw")) / "st"
    raw_path = Path(raw_st_root) / f"{sample_id}.h5ad"
    raw = sc.read_h5ad(raw_path, backed="r")
    try:
        raw_index = pd.Index(raw.obs_names.astype(str))
        indexer = raw_index.get_indexer(target.spot_ids)
        if np.any(indexer < 0):
            missing = [target.spot_ids[i] for i in np.where(indexer < 0)[0][:5]]
            raise ValueError(f"{sample_id}: processed spots missing from raw H5AD, examples={missing}")
        obs = raw.obs.iloc[indexer].copy()
        obs.index = pd.Index(target.spot_ids, name=raw.obs.index.name)
        obs["size_factor"] = target.size_factor.astype(np.float32)
        obs["size_factor_mean_one"] = target.size_factor.astype(np.float32)
        if "spatial" not in raw.obsm:
            raise ValueError(f"{sample_id}: raw H5AD lacks obsm['spatial']")
        spatial = np.asarray(raw.obsm["spatial"][indexer], dtype=np.float32)
        uns = {}
        if "spatial" in raw.uns:
            uns["spatial"] = raw.uns["spatial"]
    finally:
        raw.file.close()

    var = pd.DataFrame(index=pd.Index(target_genes, name="gene"))
    var["measured_in_slide"] = target.measured_genes.astype(bool)
    adata = ad.AnnData(
        X=log1p_rate,
        obs=obs,
        var=var,
        obsm={"spatial": spatial},
        uns=uns,
    )
    adata.uns["hest_id"] = sample_id
    adata.uns["histoomnist_target"] = "log1p_rate"
    adata.uns["histoomnist_size_factor"] = "mean-one slide-normalized"
    adata.write_h5ad(out_path, compression="gzip")
    return {
        "sample_id": sample_id,
        "status": "written",
        "path": str(out_path),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_measured_genes": int(target.measured_genes.sum()),
    }


def prepare_scellst_log1p_rate_h5ads(
    *,
    expression_config: dict[str, Any],
    rows: pd.DataFrame,
    data_dir: str | Path,
    target_gene_limit: int | None = None,
) -> dict[str, Any]:
    out_dir = Path(data_dir)
    target_genes = target_genes_from_config(expression_config, target_gene_limit=target_gene_limit)
    records = [
        _prepare_single_scellst_h5ad(
            expression_config=expression_config,
            row=row,
            target_genes=target_genes,
            data_dir=out_dir,
        )
        for row in rows.itertuples(index=False)
    ]
    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "prepared_log1p_rate_h5ads.csv", index=False)
    (out_dir / "genes.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    summary = {
        "data_dir": str(out_dir),
        "n_slides": int(len(records)),
        "n_genes": int(len(target_genes)),
        "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
        "outputs": {
            "prepared_table": str(out_dir / "prepared_log1p_rate_h5ads.csv"),
            "genes": str(out_dir / "genes.txt"),
        },
    }
    (out_dir / "prepared_log1p_rate_h5ads_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def prepare_scellst_cell_assets(
    *,
    raw_root: str | Path,
    data_dir: str | Path,
    slide_ids: list[str],
    upstream_root: str | Path | None = None,
    shape_name: str = DEFAULT_SCELLST_SHAPE_NAME,
    embedding_tag: str = DEFAULT_SCELLST_EMBEDDING_TAG,
    model_name: str = "resnet50",
    weights_path: str = "imagenet",
    normalisation_type: str = "train",
    write_in_tmp_dir: bool = True,
) -> dict[str, Any]:
    ensure_scellst_on_path(upstream_root)
    from hest import iter_hest
    from scellst.cellhest_adapter.cell_hest_data import CellHESTData

    raw = Path(raw_root)
    out = Path(data_dir)
    cell_img_dir = out / "cell_images"
    cell_stat_dir = out / "cell_image_stats"
    cell_emb_dir = out / "cell_embeddings"
    records: list[dict[str, Any]] = []
    for sid in slide_ids:
        emb_path = cell_emb_dir / f"{embedding_tag}_{sid}_{shape_name}.h5"
        img_path = cell_img_dir / f"{sid}_{shape_name}.h5"
        stat_path = cell_stat_dir / f"{sid}.json"
        if emb_path.exists():
            records.append(
                {
                    "sample_id": sid,
                    "status": "embedding_exists",
                    "cell_image_path": str(img_path),
                    "cell_stat_path": str(stat_path),
                    "cell_embedding_path": str(emb_path),
                    "embedding_size_bytes": int(emb_path.stat().st_size),
                }
            )
            continue
        st = next(iter_hest(hest_dir=str(raw), id_list=[sid], load_transcripts=False))
        cst = CellHESTData.from_HESTData(st)
        if not img_path.exists():
            cst.dump_cell_images(
                save_dir=str(cell_img_dir),
                name=sid,
                shape_name=shape_name,
                write_in_tmp_dir=write_in_tmp_dir,
            )
        if not stat_path.exists():
            cst.dump_cell_image_stats(
                cell_img_save_dir=str(cell_img_dir),
                save_dir=str(cell_stat_dir),
                shape_name=shape_name,
                name=sid,
            )
        cst.dump_cell_embeddings(
            cell_img_save_dir=str(cell_img_dir),
            cell_stat_img_save_dir=str(cell_stat_dir),
            normalisation_type=normalisation_type,
            save_dir=str(cell_emb_dir),
            shape_name=shape_name,
            name=sid,
            model_name=model_name,
            weights_path=weights_path,
            tag=embedding_tag,
            write_in_tmp_dir=write_in_tmp_dir,
        )
        records.append(
            {
                "sample_id": sid,
                "status": "written",
                "cell_image_path": str(img_path),
                "cell_stat_path": str(stat_path),
                "cell_embedding_path": str(emb_path),
                "embedding_size_bytes": int(emb_path.stat().st_size) if emb_path.exists() else 0,
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "prepared_cell_assets.csv", index=False)
    summary = {
        "raw_root": str(raw),
        "data_dir": str(out),
        "shape_name": shape_name,
        "embedding_tag": embedding_tag,
        "model_name": model_name,
        "weights_path": weights_path,
        "normalisation_type": normalisation_type,
        "n_slides": int(len(slide_ids)),
        "outputs": {"prepared_cell_assets": str(out / "prepared_cell_assets.csv")},
    }
    (out / "prepared_cell_assets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_scellst_mil_dataset(
    *,
    data_dir: str | Path,
    sample_id: str,
    upstream_root: str | Path | None = None,
    embedding_tag: str = DEFAULT_SCELLST_EMBEDDING_TAG,
    shape_name: str = DEFAULT_SCELLST_SHAPE_NAME,
) -> Any:
    ensure_scellst_on_path(upstream_root)
    import scanpy as sc
    from scellst.cellhest_adapter.cell_utils import create_spot_cell_map
    from scellst.dataset.torch_dataset import EmbeddedMilDataset

    data = Path(data_dir)
    adata = sc.read_h5ad(data / "st" / f"{sample_id}.h5ad")
    adata.obs_names = [f"{name}_{sample_id}" for name in adata.obs_names.astype(str)]
    adata.uns["hest_id"] = sample_id
    embedding_path = data / "cell_embeddings" / f"{embedding_tag}_{sample_id}_{shape_name}.h5"
    if not embedding_path.exists():
        raise FileNotFoundError(f"sCellST cell embedding file not found: {embedding_path}")
    adata.uns["cell_embedding_path"] = str(embedding_path)
    ser_map = create_spot_cell_map(str(embedding_path))
    ser_map.index = [f"{idx}_{sample_id}" for idx in ser_map.index.astype(str)]
    available = set(ser_map.index.astype(str))
    spot_names = [str(name) for name in adata.obs_names if str(name) in available]
    if not spot_names:
        raise ValueError(f"{sample_id}: no prepared spots have associated sCellST cells.")
    adata = adata[spot_names].copy()
    adata.uns["spot_cell_map"] = ser_map.loc[spot_names]
    return EmbeddedMilDataset(adata, data)


def scellst_collate():
    ensure_scellst_on_path()
    from scellst.dataset.dataset_utils import custom_collate

    return custom_collate


def build_scellst_model(
    *,
    target_genes: list[str],
    upstream_root: str | Path | None = None,
    input_dim: int = 2048,
    hidden_dim: list[int] | None = None,
    final_activation: str = "softplus",
    dropout_rate: float = 0.1,
    lr: float = 1.0e-4,
):
    ensure_scellst_on_path(upstream_root)
    from omegaconf import OmegaConf
    from scellst.lightning_model.gene_lightning_model import GeneLightningModel

    predictor_config = OmegaConf.create(
        {
            "input_dim": int(input_dim),
            "hidden_dim": [256, 256, 256] if hidden_dim is None else [int(x) for x in hidden_dim],
            "output_dim": int(len(target_genes)),
            "final_activation": str(final_activation),
            "dropout_rate": float(dropout_rate),
        }
    )
    return GeneLightningModel(
        task_type="regression",
        predictor_config=predictor_config,
        lr=float(lr),
        gene_names=list(target_genes),
        criterion="mse",
    )


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def run_scellst_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | int]:
    ensure_scellst_on_path()
    from scellst.constant import REGISTRY_KEYS

    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    spot_counts: list[int] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            bag_dict, _ = model.model(batch)
            loss = model.model.loss(bag_dict, batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        count = int(batch[REGISTRY_KEYS.Y_BAG_KEY].shape[0])
        losses.append(float(loss.detach().cpu()))
        spot_counts.append(count)
    return {
        "loss": float(np.average(losses, weights=spot_counts)) if losses else float("nan"),
        "n_batches": int(len(losses)),
        "n_spots": int(sum(spot_counts)),
        "batch_size_max": int(max(spot_counts)) if spot_counts else 0,
    }


def compute_train_mean_log1p_rate(
    *,
    data_dir: str | Path,
    slide_ids: list[str],
) -> np.ndarray:
    import scanpy as sc

    total = None
    n_spots = 0
    for sid in slide_ids:
        adata = sc.read_h5ad(Path(data_dir) / "st" / f"{sid}.h5ad")
        values = adata.X
        sums = np.asarray(values.sum(axis=0)).reshape(-1).astype(np.float64)
        total = sums if total is None else total + sums
        n_spots += int(adata.n_obs)
    if total is None or n_spots == 0:
        raise ValueError("Cannot compute train mean without training spots.")
    return (total / float(n_spots)).astype(np.float32)


def estimate_scellst_parameters(
    *,
    n_genes: int,
    input_dim: int = 2048,
    hidden_dim: list[int] | None = None,
) -> dict[str, float | int]:
    dims = [int(input_dim)] + ([256, 256, 256] if hidden_dim is None else [int(x) for x in hidden_dim]) + [int(n_genes)]
    params = 0
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        params += in_dim * out_dim + out_dim
    return {
        "input_dim": int(input_dim),
        "hidden_dim": dims[1:-1],
        "n_genes": int(n_genes),
        "mlp_trainable_params": int(params),
        "fp32_parameter_gb": float(params * 4 / 1024**3),
        "adamw_parameter_plus_state_gb": float(params * 12 / 1024**3),
    }


def train_scellst_sourcefaithful(
    *,
    expression_config: dict[str, Any],
    data_dir: str | Path,
    output_dir: str | Path,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    upstream_root: str | Path | None = None,
    target_gene_limit: int | None = None,
    embedding_tag: str = DEFAULT_SCELLST_EMBEDDING_TAG,
    shape_name: str = DEFAULT_SCELLST_SHAPE_NAME,
    epochs: int = 100,
    batch_size: int = 32,
    num_workers: int = 0,
    lr: float = 1.0e-4,
    weight_decay: float = 0.0,
    patience: int | None = 20,
    min_delta: float = 0.0,
    seed: int = 2026,
    device_name: str | None = None,
    hidden_dim: list[int] | None = None,
    final_activation: str = "softplus",
    dropout_rate: float = 0.1,
) -> dict[str, Any]:
    set_seed(int(seed))
    ensure_scellst_on_path(upstream_root)
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    target_genes = target_genes_from_config(expression_config, target_gene_limit=target_gene_limit)
    train_ids = _slide_ids(train_rows)
    val_ids = _slide_ids(val_rows)
    collate_fn = scellst_collate()
    train_ds = ConcatDataset(
        [
            load_scellst_mil_dataset(
                data_dir=data_dir,
                sample_id=sid,
                upstream_root=upstream_root,
                embedding_tag=embedding_tag,
                shape_name=shape_name,
            )
            for sid in train_ids
        ]
    )
    val_ds = ConcatDataset(
        [
            load_scellst_mil_dataset(
                data_dir=data_dir,
                sample_id=sid,
                upstream_root=upstream_root,
                embedding_tag=embedding_tag,
                shape_name=shape_name,
            )
            for sid in val_ids
        ]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = build_scellst_model(
        target_genes=target_genes,
        upstream_root=upstream_root,
        hidden_dim=hidden_dim,
        final_activation=final_activation,
        dropout_rate=dropout_rate,
        lr=lr,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows.to_csv(out / "train_slides.csv", index=False)
    val_rows.to_csv(out / "val_slides.csv", index=False)
    best_path = out / "best.pt"
    train_mean = compute_train_mean_log1p_rate(data_dir=data_dir, slide_ids=train_ids)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    for epoch in range(1, int(epochs) + 1):
        train_stats = run_scellst_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)
        val_stats = run_scellst_epoch(model=model, loader=val_loader, device=device)
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_stats["loss"]),
            "val_loss": float(val_stats["loss"]),
            "train_batches": int(train_stats["n_batches"]),
            "val_batches": int(val_stats["n_batches"]),
            "train_spots_with_cells": int(train_stats["n_spots"]),
            "val_spots_with_cells": int(val_stats["n_spots"]),
        }
        history.append(row)
        print(
            f"[scellst-sourcefaithful] epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f}",
            flush=True,
        )
        improved = float(val_stats["loss"]) < (best_val - float(min_delta))
        if improved:
            best_val = float(val_stats["loss"])
            best_epoch = int(epoch)
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_genes": target_genes,
                    "train_mean_log1p_rate": train_mean,
                    "metadata": {
                        "method": "scellst_sourcefaithful",
                        "target_kind": "log1p_rate",
                        "precision": "fp32",
                        "amp": False,
                        "tf32_matmul": False,
                        "tf32_cudnn": False,
                        "embedding_tag": embedding_tag,
                        "shape_name": shape_name,
                        "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
                        "train_slide_ids": train_ids,
                        "val_slide_ids": val_ids,
                        "batch_size": int(batch_size),
                        "lr": float(lr),
                        "weight_decay": float(weight_decay),
                        "hidden_dim": [256, 256, 256] if hidden_dim is None else [int(x) for x in hidden_dim],
                        "final_activation": final_activation,
                        "dropout_rate": float(dropout_rate),
                        "source": _jsonable_source_info(upstream_root),
                    },
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= int(patience):
                stopped_early = True
                print(f"[scellst-sourcefaithful] early stop at epoch={epoch}", flush=True)
                break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(out / "history.csv", index=False)
    summary = {
        "checkpoint": str(best_path),
        "device": str(device),
        "method": "scellst_sourcefaithful",
        "target_kind": "log1p_rate",
        "precision": "fp32",
        "amp": False,
        "source_faithful_core": True,
        "source": _jsonable_source_info(upstream_root),
        "embedding_tag": embedding_tag,
        "shape_name": shape_name,
        "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
        "n_genes": int(len(target_genes)),
        "train_slides": int(len(train_ids)),
        "val_slides": int(len(val_ids)),
        "train_spots_with_cells": int(history[-1]["train_spots_with_cells"]) if history else 0,
        "val_spots_with_cells": int(history[-1]["val_spots_with_cells"]) if history else 0,
        "epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "stopped_early": bool(stopped_early),
        "model_parameter_estimate": estimate_scellst_parameters(n_genes=len(target_genes), hidden_dim=hidden_dim),
        "outputs": {
            "checkpoint": str(best_path),
            "history": str(out / "history.csv"),
            "train_slides": str(out / "train_slides.csv"),
            "val_slides": str(out / "val_slides.csv"),
        },
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def export_scellst_sourcefaithful_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    test_rows: pd.DataFrame,
    upstream_root: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device_name: str | None = None,
) -> dict[str, Any]:
    ensure_scellst_on_path(upstream_root)
    from scellst.constant import REGISTRY_KEYS

    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    target_genes = [str(x) for x in ckpt["target_genes"]]
    metadata = dict(ckpt.get("metadata", {}))
    embedding_tag = str(metadata.get("embedding_tag", DEFAULT_SCELLST_EMBEDDING_TAG))
    shape_name = str(metadata.get("shape_name", DEFAULT_SCELLST_SHAPE_NAME))
    model = build_scellst_model(
        target_genes=target_genes,
        upstream_root=upstream_root,
        hidden_dim=list(metadata.get("hidden_dim", [256, 256, 256])),
        final_activation=str(metadata.get("final_activation", "softplus")),
        dropout_rate=float(metadata.get("dropout_rate", 0.1)),
        lr=float(metadata.get("lr", 1.0e-4)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    train_mean = np.asarray(ckpt["train_mean_log1p_rate"], dtype=np.float32)
    collate_fn = scellst_collate()
    out = Path(out_dir)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "genes.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for row in test_rows.itertuples(index=False):
        sid = str(row.sample_id)
        import scanpy as sc

        full = sc.read_h5ad(Path(data_dir) / "st" / f"{sid}.h5ad", backed="r")
        try:
            full_obs = [f"{name}_{sid}" for name in full.obs_names.astype(str)]
            n_full = int(full.n_obs)
        finally:
            full.file.close()
        prediction = np.empty((n_full, len(target_genes)), dtype=np.float32)
        prediction[:] = train_mean[None, :]
        dataset = load_scellst_mil_dataset(
            data_dir=data_dir,
            sample_id=sid,
            upstream_root=upstream_root,
            embedding_tag=embedding_tag,
            shape_name=shape_name,
        )
        obs_to_row = {name: idx for idx, name in enumerate(full_obs)}
        predicted_rows = np.asarray([obs_to_row[str(name)] for name in dataset.obs_names], dtype=np.int64)
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                batch = _batch_to_device(batch, device)
                bag_dict, _ = model.model(batch)
                pred = bag_dict[REGISTRY_KEYS.OUTPUT_PREDICTION].detach().cpu().numpy().astype(np.float32)
                stop = offset + pred.shape[0]
                prediction[predicted_rows[offset:stop], :] = pred
                offset = stop
        np.save(pred_dir / f"{sid}_log1p_rate.npy", prediction, allow_pickle=False)
        rows.append(
            {
                "sample_id": sid,
                "split": str(getattr(row, "split", "")),
                "organ": str(getattr(row, "organ", "")),
                "n_spots": int(n_full),
                "n_spots_with_cells": int(len(predicted_rows)),
                "n_spots_filled_with_train_mean": int(n_full - len(predicted_rows)),
                "prediction_path": str(pred_dir / f"{sid}_log1p_rate.npy"),
            }
        )
        print(
            f"[scellst-sourcefaithful] predicted {sid}: spots={n_full} "
            f"with_cells={len(predicted_rows)} filled={n_full - len(predicted_rows)}",
            flush=True,
        )
    slide_frame = pd.DataFrame(rows)
    slide_frame.to_csv(out / "prediction_slides.csv", index=False)
    complete = bool((slide_frame["n_spots"].astype(int) > 0).all())
    summary = {
        "method": "scellst_sourcefaithful",
        "prediction_kind": "log1p_rate",
        "checkpoint": str(checkpoint_path),
        "prediction_root": str(out),
        "n_slides": int(len(rows)),
        "n_genes": int(len(target_genes)),
        "complete_prediction_arrays": complete,
        "fill_strategy_for_spots_without_cells": "training-set mean log1p_rate vector",
        "outputs": {
            "prediction_slides": str(out / "prediction_slides.csv"),
            "genes": str(out / "genes.txt"),
            "predictions": str(pred_dir),
        },
    }
    (out / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def evaluate_scellst_predictions(
    *,
    expression_config: dict[str, Any],
    prediction_root: str | Path,
    out_dir: str | Path,
    test_rows: pd.DataFrame,
) -> dict[str, Any]:
    return evaluate_prediction_bundle(
        expression_config=expression_config,
        prediction_root=prediction_root,
        method_name="scellst_sourcefaithful_full_fp32",
        prediction_kind="log1p_rate",
        out_dir=out_dir,
        splits=sorted(set(str(x) for x in test_rows["split"].tolist())),
        slide_ids=_slide_ids(test_rows),
        prediction_pattern="predictions/{sample_id}_{kind}.npy",
        prediction_genes_path="genes.txt",
    )


def data_smoke_summary(
    *,
    expression_config: dict[str, Any],
    data_dir: str | Path,
    rows: pd.DataFrame,
    output_dir: str | Path,
    upstream_root: str | Path | None = None,
    target_gene_limit: int | None = None,
    embedding_tag: str = DEFAULT_SCELLST_EMBEDDING_TAG,
    shape_name: str = DEFAULT_SCELLST_SHAPE_NAME,
) -> dict[str, Any]:
    ensure_scellst_on_path(upstream_root)
    target_genes = target_genes_from_config(expression_config, target_gene_limit=target_gene_limit)
    sample_id = str(rows.iloc[0]["sample_id"])
    ds = load_scellst_mil_dataset(
        data_dir=data_dir,
        sample_id=sample_id,
        upstream_root=upstream_root,
        embedding_tag=embedding_tag,
        shape_name=shape_name,
    )
    item = ds[0]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "sample_id": sample_id,
        "n_selected_slides": int(len(rows)),
        "n_target_genes": int(len(target_genes)),
        "dataset_len_spots_with_cells": int(len(ds)),
        "first_item": {
            "x_shape": list(item["X"].shape),
            "y_bag_shape": list(item["Y_bag"].shape),
            "y_bag_min": float(item["Y_bag"].min()),
            "y_bag_max": float(item["Y_bag"].max()),
            "n_instance_labels": int(item["Y_ins"].shape[0]),
        },
        "source": _jsonable_source_info(upstream_root),
    }
    (out / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
