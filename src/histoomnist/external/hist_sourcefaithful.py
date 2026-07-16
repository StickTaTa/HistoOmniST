from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.eval.benchmark_predictions import evaluate_prediction_bundle, load_slide_target
from histoomnist.utils.config import get_device_name
from histoomnist.utils.io import read_manifest
from histoomnist.utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HIST_UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "benchmarks" / "HiST"
HIST_GRID_ROWS = 80
HIST_GRID_COLS = 64
HIST_NATIVE_VISIUM_ROWS = 78
HIST_FEATURE_DIM = 768


@dataclass(frozen=True)
class HiSTSourceInfo:
    upstream_root: Path
    commit: str
    model_path: Path
    feature_model_path: Path
    training_core: str


def ensure_hist_on_path(upstream_root: str | Path | None = None) -> Path:
    root = Path(upstream_root or DEFAULT_HIST_UPSTREAM_ROOT).resolve()
    if not root.exists():
        raise FileNotFoundError(f"HiST upstream root not found: {root}")
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(f"HiST src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def hist_source_info(upstream_root: str | Path | None = None) -> HiSTSourceInfo:
    root = ensure_hist_on_path(upstream_root)
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        provenance = root / "source_provenance.json"
        if provenance.exists():
            commit = json.loads(provenance.read_text(encoding="utf-8")).get("commit", "unknown")
    return HiSTSourceInfo(
        upstream_root=root,
        commit=commit,
        model_path=root / "src" / "PredictionModule" / "model.py",
        feature_model_path=root / "src" / "FeatureExtraction" / "model.py",
        training_core=(
            "official PredictionModule.model.CMUNet with HiST-style 80x64 spatial tensors; "
            "HistoOmniST adapter handles HEST split, tensor construction, export, and evaluation"
        ),
    )


def _jsonable_source_info(upstream_root: str | Path | None = None) -> dict[str, str]:
    info = hist_source_info(upstream_root)
    return {
        "upstream_root": str(info.upstream_root),
        "commit": info.commit,
        "model_path": str(info.model_path),
        "feature_model_path": str(info.feature_model_path),
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
        raise ValueError("HiST source-faithful benchmark requires data.gene_names_path.")
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


def _decode_h5_strings(values: np.ndarray) -> list[str]:
    out: list[str] = []
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    for value in array:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _raw_st_path(expression_config: dict[str, Any], sample_id: str) -> Path:
    raw_root = expression_config["data"].get(
        "raw_st_root",
        expression_config.get("paths", {}).get("raw_root", "data/HEST-1k/raw/st"),
    )
    return Path(raw_root) / f"{sample_id}.h5ad"


def _patch_h5_path(raw_root: str | Path, sample_id: str) -> Path:
    return Path(raw_root) / "patches" / f"{sample_id}.h5"


def _read_raw_visium_positions(raw_st_path: Path) -> pd.DataFrame:
    if not raw_st_path.exists():
        raise FileNotFoundError(f"Raw H5AD not found: {raw_st_path}")
    with h5py.File(raw_st_path, "r") as handle:
        obs = handle["obs"]
        required = ["_index", "array_row", "array_col"]
        missing = [name for name in required if name not in obs]
        if missing:
            raise ValueError(f"{raw_st_path} lacks obs fields required by HiST grid mapping: {missing}")
        frame = pd.DataFrame(
            {
                "spot_id": _decode_h5_strings(obs["_index"][()]),
                "array_row": np.asarray(obs["array_row"][()], dtype=np.int64),
                "array_col": np.asarray(obs["array_col"][()], dtype=np.int64),
            }
        )
        if "pxl_row_in_fullres" in obs:
            frame["pxl_row_in_fullres"] = np.asarray(obs["pxl_row_in_fullres"][()], dtype=np.float64)
        if "pxl_col_in_fullres" in obs:
            frame["pxl_col_in_fullres"] = np.asarray(obs["pxl_col_in_fullres"][()], dtype=np.float64)
    return frame


def _half_col_from_local_staggered(rel_row: np.ndarray, rel_col: np.ndarray) -> np.ndarray:
    return np.where(rel_row % 2 == 0, rel_col // 2, (rel_col - 1) // 2).astype(np.int64)


def _scale_to_limit(values: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if max_value <= 0:
        return np.zeros(values.shape[0], dtype=np.int64)
    scaled = np.rint(values * float(limit) / max_value).astype(np.int64)
    return np.clip(scaled, 0, limit)


def build_hist_grid_mapping(
    *,
    expression_config: dict[str, Any],
    row: Any,
    target_spot_ids: list[str],
    grid_policy: str = "scale_overflow",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_id = str(row.sample_id)
    raw_positions = _read_raw_visium_positions(_raw_st_path(expression_config, sample_id))
    position_lookup = raw_positions.set_index("spot_id", drop=False)
    missing = [spot for spot in target_spot_ids if spot not in position_lookup.index]
    if missing:
        raise ValueError(f"{sample_id}: {len(missing)} target spots missing from raw H5AD positions; examples={missing[:5]}")
    aligned = position_lookup.loc[target_spot_ids].reset_index(drop=True).copy()
    rel_row = aligned["array_row"].to_numpy(dtype=np.int64) - int(aligned["array_row"].min())
    rel_col = aligned["array_col"].to_numpy(dtype=np.int64) - int(aligned["array_col"].min())
    half_col = _half_col_from_local_staggered(rel_row, rel_col)
    half_col = np.clip(half_col, 0, None)
    raw_rel_rows = int(rel_row.max() + 1) if rel_row.size else 0
    raw_rel_cols = int(half_col.max() + 1) if half_col.size else 0
    overflow = raw_rel_rows > HIST_NATIVE_VISIUM_ROWS or raw_rel_cols > HIST_GRID_COLS
    if overflow and grid_policy != "scale_overflow":
        raise ValueError(
            f"{sample_id}: local Visium grid exceeds HiST 80x64 capacity "
            f"(rows={raw_rel_rows}, cols={raw_rel_cols}) and grid_policy={grid_policy!r}"
        )
    if overflow:
        grid_row = _scale_to_limit(rel_row, HIST_NATIVE_VISIUM_ROWS - 1) + 1
        grid_col = _scale_to_limit(half_col, HIST_GRID_COLS - 1)
        applied_policy = "scale_overflow"
    else:
        grid_row = rel_row + 1
        grid_col = half_col
        applied_policy = "native_local_staggered"
    grid_row = np.clip(grid_row, 0, HIST_GRID_ROWS - 1).astype(np.int64)
    grid_col = np.clip(grid_col, 0, HIST_GRID_COLS - 1).astype(np.int64)
    mapping = aligned[["spot_id", "array_row", "array_col"]].copy()
    for optional in ["pxl_row_in_fullres", "pxl_col_in_fullres"]:
        if optional in aligned.columns:
            mapping[optional] = aligned[optional].to_numpy()
    mapping.insert(0, "sample_id", sample_id)
    mapping["grid_row"] = grid_row
    mapping["grid_col"] = grid_col
    mapping["grid_linear"] = grid_row * HIST_GRID_COLS + grid_col
    collision_sizes = mapping.groupby("grid_linear")["spot_id"].transform("size").astype(int)
    mapping["collision_size"] = collision_sizes
    mapping["grid_policy"] = applied_policy
    summary = {
        "sample_id": sample_id,
        "n_spots": int(len(mapping)),
        "raw_rel_rows": raw_rel_rows,
        "raw_rel_cols": raw_rel_cols,
        "grid_rows": HIST_GRID_ROWS,
        "grid_cols": HIST_GRID_COLS,
        "grid_policy": applied_policy,
        "overflow": bool(overflow),
        "n_occupied_cells": int(mapping["grid_linear"].nunique()),
        "n_colliding_spots": int((mapping["collision_size"] > 1).sum()),
        "max_collision_size": int(mapping["collision_size"].max()) if len(mapping) else 0,
    }
    return mapping, summary


def _target_grid_from_slide_target(target, n_genes: int, mapping: pd.DataFrame) -> np.ndarray:
    counts = target.counts.astype(np.float32).tocsr()
    inv_sf = np.reciprocal(np.clip(target.size_factor.astype(np.float32), 1.0e-6, None))
    log1p_rate = counts.multiply(inv_sf[:, None]).tocsr()
    log1p_rate.data = np.log1p(log1p_rate.data).astype(np.float32, copy=False)
    grid = np.zeros((n_genes, HIST_GRID_ROWS, HIST_GRID_COLS), dtype=np.float32)
    occupancy = np.zeros((HIST_GRID_ROWS, HIST_GRID_COLS), dtype=np.int32)
    grid_rows = mapping["grid_row"].to_numpy(dtype=np.int64)
    grid_cols = mapping["grid_col"].to_numpy(dtype=np.int64)
    indptr = log1p_rate.indptr
    indices = log1p_rate.indices
    data = log1p_rate.data
    for i, (grid_row, grid_col) in enumerate(zip(grid_rows, grid_cols, strict=True)):
        start = indptr[i]
        stop = indptr[i + 1]
        if stop > start:
            grid[indices[start:stop], grid_row, grid_col] += data[start:stop]
        occupancy[grid_row, grid_col] += 1
    multi = occupancy > 1
    if np.any(multi):
        grid[:, multi] /= occupancy[multi][None, :]
    return grid


def prepare_hist_sourcefaithful_data(
    *,
    expression_config: dict[str, Any],
    rows: pd.DataFrame,
    data_dir: str | Path,
    target_gene_limit: int | None = None,
    grid_policy: str = "scale_overflow",
    write_targets: bool = True,
) -> dict[str, Any]:
    out = Path(data_dir)
    mapping_dir = out / "grid_mapping"
    target_dir = out / "targets"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    if write_targets:
        target_dir.mkdir(parents=True, exist_ok=True)
    target_genes = target_genes_from_config(expression_config, target_gene_limit=target_gene_limit)
    (out / "genes.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    manifest_path = _manifest_path(expression_config)
    base_dir = manifest_path.parent
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    records: list[dict[str, Any]] = []
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    for manifest_row in rows.itertuples(index=False):
        target = load_slide_target(
            row=manifest_row,
            base_dir=base_dir,
            target_genes=target_genes,
            gene_key=gene_key,
            raw_st_root=raw_st_root,
            min_total_counts=min_total_counts,
        )
        mapping, grid_summary = build_hist_grid_mapping(
            expression_config=expression_config,
            row=manifest_row,
            target_spot_ids=target.spot_ids,
            grid_policy=grid_policy,
        )
        mapping_path = mapping_dir / f"{target.sample_id}.csv"
        mapping.to_csv(mapping_path, index=False)
        target_path = None
        if write_targets:
            target_path = target_dir / f"{target.sample_id}.npy"
            reuse_target = False
            if target_path.exists():
                existing = np.load(target_path, mmap_mode="r", allow_pickle=False)
                reuse_target = existing.shape == (len(target_genes), HIST_GRID_ROWS, HIST_GRID_COLS)
            if not reuse_target:
                target_grid = _target_grid_from_slide_target(target, len(target_genes), mapping)
                np.save(target_path, target_grid, allow_pickle=False)
        records.append(
            {
                **grid_summary,
                "split": target.split,
                "organ": target.organ,
                "cohort": target.cohort,
                "n_genes": int(len(target_genes)),
                "n_measured_genes": int(target.measured_genes.sum()),
                "mapping_path": str(mapping_path),
                "target_path": None if target_path is None else str(target_path),
            }
        )
        print(
            f"[HiST prepare] {target.sample_id}: spots={len(target.spot_ids)} genes={len(target_genes)} "
            f"policy={grid_summary['grid_policy']} occupied={grid_summary['n_occupied_cells']}",
            flush=True,
        )
    frame = pd.DataFrame(records)
    frame.to_csv(out / "prepared_slides.csv", index=False)
    summary = {
        "data_dir": str(out),
        "n_slides": int(len(frame)),
        "n_genes": int(len(target_genes)),
        "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
        "grid_policy_requested": grid_policy,
        "n_overflow_slides": int(frame["overflow"].sum()) if not frame.empty else 0,
        "n_colliding_spots": int(frame["n_colliding_spots"].sum()) if not frame.empty else 0,
        "write_targets": bool(write_targets),
        "outputs": {
            "prepared_slides": str(out / "prepared_slides.csv"),
            "genes": str(out / "genes.txt"),
        },
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_ctranspath_model(
    *,
    upstream_root: str | Path | None,
    model_weight_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    import torch

    ensure_hist_on_path(upstream_root)
    from FeatureExtraction.model import ctranspath

    weight_path = Path(model_weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"HiST CTransPath weight not found: {weight_path}")
    model = ctranspath()
    model.head = torch.nn.Identity()
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def _transform_patch_batch(images: list[np.ndarray]) -> torch.Tensor:
    import torch
    from PIL import Image
    from torchvision import transforms

    trnsfrms_valid = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensors = [trnsfrms_valid(Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")) for image in images]
    return torch.stack(tensors, dim=0)


def extract_hist_ctranspath_features_from_patch_h5(
    *,
    expression_config: dict[str, Any],
    rows: pd.DataFrame,
    data_dir: str | Path,
    raw_root: str | Path,
    upstream_root: str | Path | None = None,
    model_weight_path: str | Path | None = None,
    batch_size: int = 80,
    device_name: str | None = None,
) -> dict[str, Any]:
    import torch

    out = Path(data_dir)
    feature_dir = out / "features"
    mapping_dir = out / "grid_mapping"
    feature_dir.mkdir(parents=True, exist_ok=True)
    weight = Path(model_weight_path) if model_weight_path else ensure_hist_on_path(upstream_root) / "resource" / "ctranspath.pth"
    device = torch.device(device_name or get_device_name(expression_config.get("device")))
    model = _load_ctranspath_model(upstream_root=upstream_root, model_weight_path=weight, device=device)
    with torch.no_grad():
        blank_tensor = _transform_patch_batch([np.full((224, 224, 3), 255, dtype=np.uint8)]).to(device)
        blank_feature = model(blank_tensor).detach().cpu().numpy()[0].astype(np.float32)
    records: list[dict[str, Any]] = []
    for manifest_row in rows.itertuples(index=False):
        sample_id = str(manifest_row.sample_id)
        mapping_path = mapping_dir / f"{sample_id}.csv"
        if not mapping_path.exists():
            raise FileNotFoundError(f"HiST grid mapping not found for {sample_id}: {mapping_path}")
        patch_path = _patch_h5_path(raw_root, sample_id)
        if not patch_path.exists():
            raise FileNotFoundError(f"HEST patch H5 not found for {sample_id}: {patch_path}")
        mapping = pd.read_csv(mapping_path)
        feature_path = feature_dir / f"{sample_id}.npy"
        if feature_path.exists():
            existing = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            if existing.shape == (HIST_GRID_ROWS, HIST_GRID_COLS, HIST_FEATURE_DIM):
                records.append(
                    {
                        "sample_id": sample_id,
                        "feature_path": str(feature_path),
                        "patch_h5_path": str(patch_path),
                        "n_patch_barcodes": None,
                        "n_used_spots": None,
                        "n_occupied_cells": int(mapping["grid_linear"].nunique()),
                        "n_collision_cells": int((mapping["collision_size"] > 1).sum()),
                        "feature_source": "hest_patch_h5_ctranspath",
                        "status": "exists",
                    }
                )
                print(f"[HiST features] {sample_id}: existing feature grid reused", flush=True)
                continue
        spot_to_cell = {
            str(row.spot_id): (int(row.grid_row), int(row.grid_col))
            for row in mapping.itertuples(index=False)
        }
        feature_sum = np.zeros((HIST_GRID_ROWS, HIST_GRID_COLS, HIST_FEATURE_DIM), dtype=np.float32)
        occupancy = np.zeros((HIST_GRID_ROWS, HIST_GRID_COLS), dtype=np.int32)
        n_seen = 0
        n_used = 0
        with h5py.File(patch_path, "r") as handle:
            barcodes = _decode_h5_strings(handle["barcode"][()])
            images = handle["img"]
            batch_images: list[np.ndarray] = []
            batch_cells: list[tuple[int, int]] = []
            for idx, barcode in enumerate(barcodes):
                n_seen += 1
                cell = spot_to_cell.get(str(barcode))
                if cell is None:
                    continue
                batch_images.append(images[idx])
                batch_cells.append(cell)
                if len(batch_images) >= int(batch_size):
                    patch_tensor = _transform_patch_batch(batch_images).to(device)
                    with torch.no_grad():
                        features = model(patch_tensor).detach().cpu().numpy().astype(np.float32)
                    for feature, (grid_row, grid_col) in zip(features, batch_cells, strict=True):
                        feature_sum[grid_row, grid_col] += feature
                        occupancy[grid_row, grid_col] += 1
                        n_used += 1
                    batch_images.clear()
                    batch_cells.clear()
            if batch_images:
                patch_tensor = _transform_patch_batch(batch_images).to(device)
                with torch.no_grad():
                    features = model(patch_tensor).detach().cpu().numpy().astype(np.float32)
                for feature, (grid_row, grid_col) in zip(features, batch_cells, strict=True):
                    feature_sum[grid_row, grid_col] += feature
                    occupancy[grid_row, grid_col] += 1
                    n_used += 1
        feature_grid = np.empty_like(feature_sum)
        feature_grid[:] = blank_feature.reshape(1, 1, HIST_FEATURE_DIM)
        occupied = occupancy > 0
        if np.any(occupied):
            feature_grid[occupied] = feature_sum[occupied] / occupancy[occupied, None]
        np.save(feature_path, feature_grid, allow_pickle=False)
        records.append(
            {
                "sample_id": sample_id,
                "feature_path": str(feature_path),
                "patch_h5_path": str(patch_path),
                "n_patch_barcodes": int(n_seen),
                "n_used_spots": int(n_used),
                "n_occupied_cells": int(occupied.sum()),
                "n_collision_cells": int((occupancy > 1).sum()),
                "feature_source": "hest_patch_h5_ctranspath",
            }
        )
        print(
            f"[HiST features] {sample_id}: patches={n_seen} used={n_used} occupied={int(occupied.sum())}",
            flush=True,
        )
    frame = pd.DataFrame(records)
    frame.to_csv(out / "extracted_ctranspath_features.csv", index=False)
    summary = {
        "data_dir": str(out),
        "raw_root": str(raw_root),
        "n_slides": int(len(frame)),
        "model_weight_path": str(weight),
        "feature_source": "hest_patch_h5_ctranspath",
        "source_info": _jsonable_source_info(upstream_root),
        "outputs": {"feature_table": str(out / "extracted_ctranspath_features.csv")},
    }
    (out / "feature_extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class HiSTGridDataset:
    def __init__(self, rows: pd.DataFrame, data_dir: str | Path, *, require_target: bool = True):
        self.rows = rows.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.require_target = require_target

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        sample_id = str(self.rows.iloc[idx]["sample_id"])
        feature = np.load(self.data_dir / "features" / f"{sample_id}.npy", allow_pickle=False)
        if feature.shape != (HIST_GRID_ROWS, HIST_GRID_COLS, HIST_FEATURE_DIM):
            raise ValueError(f"{sample_id}: unexpected HiST feature shape {feature.shape}")
        item: dict[str, Any] = {
            "sample_id": sample_id,
            "feature": torch.from_numpy(feature.transpose(2, 0, 1).astype(np.float32, copy=False)),
        }
        if self.require_target:
            target = np.load(self.data_dir / "targets" / f"{sample_id}.npy", allow_pickle=False)
            if target.ndim != 3 or target.shape[1:] != (HIST_GRID_ROWS, HIST_GRID_COLS):
                raise ValueError(f"{sample_id}: unexpected HiST target shape {target.shape}")
            item["target"] = torch.from_numpy(target.astype(np.float32, copy=False))
        return item


def _load_hist_cmunet(upstream_root: str | Path | None, *, n_genes: int, device: torch.device) -> torch.nn.Module:
    import torch

    ensure_hist_on_path(upstream_root)
    from PredictionModule.model import CMUNet

    model = CMUNet(img_ch=HIST_FEATURE_DIM, output_ch=int(n_genes), l=7, k=7)
    return model.to(device)


def _epoch_loop(
    *,
    loader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    import torch

    train = optimizer is not None
    model.train(mode=train)
    total = 0
    loss_sum = 0.0
    mae_sum = 0.0
    rmse_sum = 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["feature"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n = int(x.shape[0])
            err = pred.detach() - y
            loss_sum += float(loss.detach().cpu()) * n
            mae_sum += float(torch.mean(torch.abs(err)).detach().cpu()) * n
            rmse_sum += float(torch.sqrt(torch.mean(torch.square(err))).detach().cpu()) * n
            total += n
    denom = max(total, 1)
    return {"loss": loss_sum / denom, "mae": mae_sum / denom, "rmse": rmse_sum / denom}


def train_hist_sourcefaithful(
    *,
    expression_config: dict[str, Any],
    data_dir: str | Path,
    output_dir: str | Path,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    upstream_root: str | Path | None = None,
    target_gene_limit: int | None = None,
    epochs: int = 200,
    batch_size: int = 1,
    num_workers: int = 0,
    lr: float = 0.001,
    weight_decay: float = 1.0e-4,
    seed: int = 42,
    device_name: str | None = None,
    best_warmup_epochs: int = 20,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    genes = target_genes_from_config(expression_config, target_gene_limit=target_gene_limit)
    device = torch.device(device_name or get_device_name(expression_config.get("device")))
    train_loader = DataLoader(
        HiSTGridDataset(train_rows, data_dir, require_target=True),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        HiSTGridDataset(val_rows, data_dir, require_target=True),
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model = _load_hist_cmunet(upstream_root, n_genes=len(genes), device=device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1.0e-5)
    criterion = torch.nn.MSELoss().to(device)
    log_rows: list[dict[str, Any]] = []
    best_mae = float("inf")
    best_epoch = 0
    best_path = out / "best_model.pt"
    last_path = out / "last_model.pt"
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _epoch_loop(
            loader=train_loader,
            model=model,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = _epoch_loop(
            loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(out / "train_log.csv", index=False)
        if epoch > int(best_warmup_epochs) and val_metrics["mae"] < best_mae:
            best_mae = float(val_metrics["mae"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "best_val_mae": best_mae,
                    "n_genes": int(len(genes)),
                    "genes": genes,
                    "source_info": _jsonable_source_info(upstream_root),
                    "train_slide_ids": [str(x) for x in train_rows["sample_id"].tolist()],
                    "val_slide_ids": [str(x) for x in val_rows["sample_id"].tolist()],
                },
                best_path,
            )
        print(
            f"[HiST train] epoch={epoch}/{epochs} loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_mae={val_metrics['mae']:.6f}",
            flush=True,
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epochs),
            "n_genes": int(len(genes)),
            "genes": genes,
            "source_info": _jsonable_source_info(upstream_root),
        },
        last_path,
    )
    if not best_path.exists():
        best_path.write_bytes(last_path.read_bytes())
        best_epoch = int(epochs)
        best_mae = float(log_rows[-1]["val_mae"]) if log_rows else float("nan")
    summary = {
        "output_dir": str(out),
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "n_train_slides": int(len(train_rows)),
        "n_val_slides": int(len(val_rows)),
        "n_genes": int(len(genes)),
        "target_gene_limit": None if target_gene_limit is None else int(target_gene_limit),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "best_warmup_epochs": int(best_warmup_epochs),
        "source_info": _jsonable_source_info(upstream_root),
        "outputs": {"train_log": str(out / "train_log.csv")},
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def export_hist_sourcefaithful_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    test_rows: pd.DataFrame,
    upstream_root: str | Path | None = None,
    target_gene_limit: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    genes = target_genes_from_config(expression_config, target_gene_limit=target_gene_limit)
    out = Path(out_dir)
    prediction_dir = out / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (out / "genes.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
    device = torch.device(device_name or get_device_name(expression_config.get("device")))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _load_hist_cmunet(upstream_root, n_genes=len(genes), device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = HiSTGridDataset(test_rows, data_dir, require_target=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            sample_id = str(batch["sample_id"][0])
            pred_grid = model(batch["feature"].to(device)).detach().cpu().numpy()[0]
            mapping = pd.read_csv(Path(data_dir) / "grid_mapping" / f"{sample_id}.csv")
            grid_rows = mapping["grid_row"].to_numpy(dtype=np.int64)
            grid_cols = mapping["grid_col"].to_numpy(dtype=np.int64)
            spot_predictions = pred_grid[:, grid_rows, grid_cols].T.astype(np.float32, copy=False)
            pred_path = prediction_dir / f"{sample_id}_log1p_rate.npy"
            np.save(pred_path, spot_predictions, allow_pickle=False)
            records.append(
                {
                    "sample_id": sample_id,
                    "prediction_path": str(pred_path),
                    "n_spots": int(spot_predictions.shape[0]),
                    "n_genes": int(spot_predictions.shape[1]),
                }
            )
            print(f"[HiST export] {sample_id}: {spot_predictions.shape}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "prediction_manifest.csv", index=False)
    summary = {
        "prediction_root": str(out),
        "n_slides": int(len(frame)),
        "n_genes": int(len(genes)),
        "checkpoint": str(checkpoint_path),
        "prediction_kind": "log1p_rate",
        "outputs": {"prediction_manifest": str(out / "prediction_manifest.csv"), "genes": str(out / "genes.txt")},
    }
    (out / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_hist_predictions(
    *,
    expression_config: dict[str, Any],
    prediction_root: str | Path,
    out_dir: str | Path,
    splits: list[str] | None = None,
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    max_slide_spots: int | None = None,
) -> dict[str, Any]:
    return evaluate_prediction_bundle(
        expression_config=expression_config,
        prediction_root=prediction_root,
        method_name="hist_sourcefaithful",
        prediction_kind="log1p_rate",
        out_dir=out_dir,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        max_slide_spots=max_slide_spots,
        prediction_pattern="predictions/{sample_id}_log1p_rate.npy",
        prediction_genes_path="genes.txt",
    )
