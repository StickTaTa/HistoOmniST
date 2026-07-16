from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from scipy import sparse
from torch import nn

from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.eval.benchmark_predictions import (
    evaluate_prediction_bundle,
    load_slide_target,
)
from histoomnist.hest.raw_assets import read_h5_string_vector
from histoomnist.train.common import checkpoint_payload, load_checkpoint, save_checkpoint
from histoomnist.utils.config import get_device_name
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path
from histoomnist.utils.seed import set_seed


TargetKind = Literal["log1p_rate"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH2SPACE_UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "benchmarks" / "Path2Space"
DEFAULT_CTRANSPATH_WEIGHT_PATH = (
    PROJECT_ROOT / "third_party" / "benchmarks" / "HiST" / "resource" / "ctranspath.pth"
)


@dataclass
class Path2SpaceSourceInfo:
    upstream_root: Path
    feature_extraction_root: Path
    regression_script_path: Path
    ctranspath_weight_path: Path
    imported_component: str
    commit: str | None = None


@dataclass
class Path2SpaceSlide:
    sample_id: str
    split: str
    organ: str
    cohort: str
    disease_state: str
    patch_h5_path: Path
    spot_ids: list[str]
    patch_indices: np.ndarray
    counts: sparse.csr_matrix
    size_factor: np.ndarray
    measured_genes: np.ndarray
    feature_path: Path | None = None

    @property
    def n_spots(self) -> int:
        return int(self.patch_indices.shape[0])


class Path2SpaceSlideCollection:
    def __init__(
        self,
        *,
        slides: list[Path2SpaceSlide],
        target_genes: list[str],
        feature_dir: str | Path | None = None,
        target_kind: TargetKind = "log1p_rate",
    ):
        self.slides = list(slides)
        self.target_genes = list(target_genes)
        self.target_kind = target_kind
        self.feature_dir = None if feature_dir is None else Path(feature_dir)
        if self.feature_dir is not None:
            self.feature_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.slides)

    def feature_path_for(self, sample_id: str) -> Path:
        if self.feature_dir is None:
            raise ValueError("feature_dir is not set on this collection")
        return self.feature_dir / f"{sample_id}.npy"

    def slide_by_id(self, sample_id: str) -> Path2SpaceSlide:
        sample_id = str(sample_id)
        for slide in self.slides:
            if slide.sample_id == sample_id:
                return slide
        raise KeyError(sample_id)

    def slide_summary_frame(self) -> pd.DataFrame:
        rows = []
        for slide in self.slides:
            feature_path = slide.feature_path
            if feature_path is None and self.feature_dir is not None:
                feature_path = self.feature_path_for(slide.sample_id)
            rows.append(
                {
                    "sample_id": slide.sample_id,
                    "split": slide.split,
                    "organ": slide.organ,
                    "cohort": slide.cohort,
                    "disease_state": slide.disease_state,
                    "n_spots": int(slide.n_spots),
                    "n_measured_target_genes": int(slide.measured_genes.sum()),
                    "patch_h5_path": str(slide.patch_h5_path),
                    "feature_path": "" if feature_path is None else str(feature_path),
                    "feature_exists": bool(feature_path is not None and Path(feature_path).exists()),
                }
            )
        return pd.DataFrame(rows)


OFFICIAL_PATH2SPACE_MODEL_DEFAULTS = {
    "n_inputs": 768,
    "n_hiddens": 768,
    "dropout": 0.2,
    "learning_rate": 1.0e-4,
    "optimizer": "Adam",
    "loss": "MSE",
    "feature_extractor": "CTransPath",
    "regression_head": "MLP_regression_relu_two",
}


def _resolve_root(path: str | Path | None, fallback: Path) -> Path:
    if path in (None, ""):
        return fallback.resolve()
    resolved = resolve_project_path(path)
    if resolved is None:
        return fallback.resolve()
    return Path(resolved).resolve()


def _git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return (
            subprocess.check_output(
                ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def ensure_path2space_on_path(upstream_root: str | Path | None = None) -> Path:
    root = _resolve_root(upstream_root, DEFAULT_PATH2SPACE_UPSTREAM_ROOT)
    feature_root = root / "scripts" / "1.ST_prediction" / "1.1.Feature_extraction"
    if not feature_root.exists():
        raise FileNotFoundError(f"Path2Space feature extraction root not found: {feature_root}")
    if str(feature_root) not in sys.path:
        sys.path.insert(0, str(feature_root))
    return feature_root


def path2space_source_info(
    upstream_root: str | Path | None = None,
    *,
    ctranspath_weight_path: str | Path | None = None,
) -> Path2SpaceSourceInfo:
    root = _resolve_root(upstream_root, DEFAULT_PATH2SPACE_UPSTREAM_ROOT)
    feature_root = ensure_path2space_on_path(root)
    regression_script_path = root / "scripts" / "1.ST_prediction" / "1.2.Regression" / "1main_regression.py"
    if not regression_script_path.exists():
        raise FileNotFoundError(f"Path2Space regression script not found: {regression_script_path}")
    weight_path = _resolve_root(ctranspath_weight_path, DEFAULT_CTRANSPATH_WEIGHT_PATH)
    if not weight_path.exists():
        raise FileNotFoundError(f"CTransPath weight file not found: {weight_path}")
    return Path2SpaceSourceInfo(
        upstream_root=root,
        feature_extraction_root=feature_root,
        regression_script_path=regression_script_path,
        ctranspath_weight_path=weight_path,
        imported_component="official func.ctrans_model.CTransPath + official regression MLP",
        commit=_git_commit(root),
    )


def _jsonable_source_info(
    upstream_root: str | Path | None = None,
    *,
    ctranspath_weight_path: str | Path | None = None,
) -> dict[str, str]:
    info = path2space_source_info(upstream_root, ctranspath_weight_path=ctranspath_weight_path)
    return {
        "upstream_root": str(info.upstream_root),
        "feature_extraction_root": str(info.feature_extraction_root),
        "regression_script_path": str(info.regression_script_path),
        "ctranspath_weight_path": str(info.ctranspath_weight_path),
        "imported_component": info.imported_component,
        "commit": "" if info.commit is None else info.commit,
    }


def _enforce_fp32_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def select_manifest_rows(
    expression_config: dict[str, Any],
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
    if slide_ids:
        wanted = {str(x) for x in slide_ids}
        rows = rows[rows["sample_id"].astype(str).isin(wanted)].copy()
    if max_slide_spots is not None:
        if "n_spots" not in rows.columns:
            raise ValueError("Manifest lacks n_spots; cannot apply max_slide_spots.")
        rows = rows[rows["n_spots"].astype(int) <= int(max_slide_spots)].copy()
    if smallest_slides:
        rows = rows.sort_values(["n_spots", "sample_id"], ascending=[True, True]).copy()
    if max_slides is not None:
        rows = rows.head(int(max_slides)).copy()
    if rows.empty:
        raise ValueError(f"No manifest rows for splits={splits}, slide_ids={slide_ids}")
    return rows.reset_index(drop=True)


def _build_slide(
    *,
    row: Any,
    base_dir: Path,
    raw_root: Path,
    target_genes: list[str],
    gene_key: str,
    raw_st_root: Path | None,
    min_total_counts: float,
) -> Path2SpaceSlide:
    target = load_slide_target(
        row=row,
        base_dir=base_dir,
        target_genes=target_genes,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
        min_total_counts=min_total_counts,
    )
    sample_id = str(target.sample_id)
    patch_h5_path = raw_root / "patches" / f"{sample_id}.h5"
    if not patch_h5_path.exists():
        raise FileNotFoundError(f"Patch H5 not found for {sample_id}: {patch_h5_path}")
    with h5py.File(patch_h5_path, "r") as handle:
        barcodes = read_h5_string_vector(handle["barcode"])
    patch_index = {barcode: idx for idx, barcode in enumerate(barcodes)}
    missing = [barcode for barcode in target.spot_ids if barcode not in patch_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{sample_id} has {len(missing)} processed spots missing from patch H5 barcode: {preview}"
        )
    patch_indices = np.asarray([patch_index[barcode] for barcode in target.spot_ids], dtype=np.int64)
    return Path2SpaceSlide(
        sample_id=sample_id,
        split=str(target.split),
        organ=str(target.organ),
        cohort=str(target.cohort),
        disease_state=str(target.disease_state),
        patch_h5_path=patch_h5_path,
        spot_ids=list(target.spot_ids),
        patch_indices=patch_indices,
        counts=target.counts,
        size_factor=np.array(target.size_factor, dtype=np.float32, copy=False),
        measured_genes=np.array(target.measured_genes, dtype=bool, copy=False),
    )


def build_path2space_collection(
    expression_config: dict[str, Any],
    *,
    splits: list[str],
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_slide_spots: int | None = None,
    feature_dir: str | Path | None = None,
    upstream_root: str | Path | None = None,
) -> Path2SpaceSlideCollection:
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression config data.manifest resolved to None")
    base_dir = manifest_path.parent
    target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
    if target_genes is None or gene_indices is not None:
        raise ValueError("Path2Space benchmark requires data.gene_names_path canonical genes.")
    raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
    if raw_root is None:
        raise ValueError("Expression config paths.raw_root resolved to None")
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    rows = select_manifest_rows(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        max_slide_spots=max_slide_spots,
    )
    slides = [
        _build_slide(
            row=row,
            base_dir=base_dir,
            raw_root=raw_root,
            target_genes=[str(x) for x in target_genes],
            gene_key=gene_key,
            raw_st_root=raw_st_root,
            min_total_counts=min_total_counts,
        )
        for row in rows.itertuples(index=False)
    ]
    return Path2SpaceSlideCollection(
        slides=slides,
        target_genes=[str(x) for x in target_genes],
        feature_dir=feature_dir,
        target_kind="log1p_rate",
    )


def _ctranspath_import(feature_root: Path):
    if str(feature_root) not in sys.path:
        sys.path.insert(0, str(feature_root))
    module = importlib.import_module("func.ctrans_model")
    if not hasattr(module, "CTransPath"):
        raise AttributeError("Path2Space func.ctrans_model lacks CTransPath")
    return module.CTransPath


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {str(key).removeprefix("module."): value for key, value in state_dict.items()}


def build_official_ctranspath_model(
    *,
    upstream_root: str | Path | None = None,
    ctranspath_weight_path: str | Path | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    info = path2space_source_info(upstream_root, ctranspath_weight_path=ctranspath_weight_path)
    ctranspath_cls = _ctranspath_import(info.feature_extraction_root)
    model = ctranspath_cls(num_classes=0)
    if device is not None:
        model = model.to(device)
    payload = torch.load(info.ctranspath_weight_path, map_location="cpu")
    if isinstance(payload, dict):
        if "model" in payload:
            payload = payload["model"]
        elif "state_dict" in payload:
            payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected CTransPath checkpoint format: {type(payload)!r}")
    payload = _strip_module_prefix(payload)
    model.load_state_dict(payload, strict=True)
    model.eval()
    return model


def _feature_transform(use_macenko: bool):
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    color_normalizer = None
    if use_macenko:
        try:
            from func.utils_color_norm import macenko_normalizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Macenko normalization requested but Path2Space color normalization stack is unavailable."
            ) from exc
        color_normalizer = macenko_normalizer()
    return Image, transform, color_normalizer


def _read_patch_batch(
    *,
    h5_handle: h5py.File,
    patch_indices: np.ndarray,
    start: int,
    end: int,
) -> list[np.ndarray]:
    imgs = h5_handle["img"]
    batch = [np.asarray(imgs[int(idx)], dtype=np.uint8) for idx in patch_indices[start:end]]
    return batch


def _batch_to_tensor(
    batch: list[np.ndarray],
    *,
    image_cls,
    transform,
    color_normalizer,
) -> torch.Tensor:
    images = []
    for patch in batch:
        arr = patch
        if color_normalizer is not None:
            arr = color_normalizer.transform(arr)
        images.append(transform(image_cls.fromarray(arr, mode="RGB")))
    return torch.stack(images, dim=0)


def extract_path2space_slide_features(
    *,
    slide: Path2SpaceSlide,
    model: nn.Module,
    device: torch.device,
    feature_batch_size: int = 128,
    use_macenko: bool = False,
) -> np.ndarray:
    image_cls, transform, color_normalizer = _feature_transform(use_macenko)
    chunks: list[np.ndarray] = []
    model.eval()
    with h5py.File(slide.patch_h5_path, "r") as handle, torch.inference_mode():
        for start in range(0, slide.n_spots, int(feature_batch_size)):
            end = min(start + int(feature_batch_size), slide.n_spots)
            batch = _read_patch_batch(
                h5_handle=handle,
                patch_indices=slide.patch_indices,
                start=start,
                end=end,
            )
            tensor = _batch_to_tensor(
                batch,
                image_cls=image_cls,
                transform=transform,
                color_normalizer=color_normalizer,
            ).to(device)
            features = model(tensor)
            if features.ndim != 2:
                features = features.reshape(features.shape[0], -1)
            chunks.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
    if not chunks:
        raise ValueError(f"No features extracted for {slide.sample_id}")
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def cache_path2space_slide_features(
    *,
    slide: Path2SpaceSlide,
    feature_dir: str | Path,
    model: nn.Module,
    device: torch.device,
    feature_batch_size: int = 128,
    use_macenko: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / f"{slide.sample_id}.npy"
    reused = False
    feature_dim = None
    if feature_path.exists() and not overwrite:
        existing = np.load(feature_path, mmap_mode="r")
        if existing.ndim == 2 and existing.shape[0] == slide.n_spots and existing.shape[1] == 768:
            slide.feature_path = feature_path
            reused = True
            feature_dim = int(existing.shape[1])
            return {
                "sample_id": slide.sample_id,
                "feature_path": str(feature_path),
                "feature_dim": feature_dim,
                "n_spots": int(slide.n_spots),
                "reused": True,
                "feature_bytes": int(existing.nbytes),
            }
    writer = None
    offset = 0
    image_cls, transform, color_normalizer = _feature_transform(use_macenko)
    with h5py.File(slide.patch_h5_path, "r") as handle, torch.inference_mode():
        for start in range(0, slide.n_spots, int(feature_batch_size)):
            end = min(start + int(feature_batch_size), slide.n_spots)
            batch = _read_patch_batch(
                h5_handle=handle,
                patch_indices=slide.patch_indices,
                start=start,
                end=end,
            )
            tensor = _batch_to_tensor(
                batch,
                image_cls=image_cls,
                transform=transform,
                color_normalizer=color_normalizer,
            ).to(device)
            features = model(tensor)
            if features.ndim != 2:
                features = features.reshape(features.shape[0], -1)
            features_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
            if writer is None:
                feature_dim = int(features_np.shape[1])
                writer = open_memmap(
                    feature_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(slide.n_spots, feature_dim),
                )
            writer[offset : offset + features_np.shape[0]] = features_np
            offset += int(features_np.shape[0])
    if writer is None or feature_dim is None:
        raise ValueError(f"Failed to cache features for {slide.sample_id}")
    writer.flush()
    slide.feature_path = feature_path
    return {
        "sample_id": slide.sample_id,
        "feature_path": str(feature_path),
        "feature_dim": int(feature_dim),
        "n_spots": int(slide.n_spots),
        "reused": reused,
        "feature_bytes": int(feature_path.stat().st_size),
    }


def prepare_path2space_feature_cache(
    *,
    expression_config: dict[str, Any],
    rows: pd.DataFrame,
    data_dir: str | Path,
    upstream_root: str | Path | None = None,
    ctranspath_weight_path: str | Path | None = None,
    feature_batch_size: int = 128,
    device_name: str | None = None,
    use_macenko: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    feature_dir = data_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    _enforce_fp32_cuda(device)
    model = build_official_ctranspath_model(
        upstream_root=upstream_root,
        ctranspath_weight_path=ctranspath_weight_path,
        device=device,
    )
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression config data.manifest resolved to None")
    base_dir = manifest_path.parent
    target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
    if target_genes is None or gene_indices is not None:
        raise ValueError("Path2Space benchmark requires data.gene_names_path canonical genes.")
    raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
    if raw_root is None:
        raise ValueError("Expression config paths.raw_root resolved to None")
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    slides = [
        _build_slide(
            row=row,
            base_dir=base_dir,
            raw_root=raw_root,
            target_genes=[str(x) for x in target_genes],
            gene_key=gene_key,
            raw_st_root=raw_st_root,
            min_total_counts=min_total_counts,
        )
        for row in rows.itertuples(index=False)
    ]
    cache_rows = []
    for slide in slides:
        summary = cache_path2space_slide_features(
            slide=slide,
            feature_dir=feature_dir,
            model=model,
            device=device,
            feature_batch_size=feature_batch_size,
            use_macenko=use_macenko,
            overwrite=overwrite,
        )
        cache_rows.append(summary)
        print(
            f"[path2space-cache] {slide.sample_id}: "
            f"spots={slide.n_spots} feature_dim={summary['feature_dim']} reused={summary['reused']}",
            flush=True,
        )
    frame = pd.DataFrame(cache_rows)
    frame.to_csv(feature_dir / "feature_manifest.csv", index=False)
    summary = {
        "feature_dir": str(feature_dir),
        "n_slides": int(len(slides)),
        "n_spots": int(sum(slide.n_spots for slide in slides)),
        "feature_batch_size": int(feature_batch_size),
        "use_macenko": bool(use_macenko),
        "overwrite": bool(overwrite),
        "source": _jsonable_source_info(
            upstream_root,
            ctranspath_weight_path=ctranspath_weight_path,
        ),
        "outputs": {
            "feature_manifest": str(feature_dir / "feature_manifest.csv"),
        },
    }
    (feature_dir / "feature_cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _target_matrix_for_slide_chunk(
    slide: Path2SpaceSlide,
    *,
    start: int,
    end: int,
    target_kind: TargetKind = "log1p_rate",
) -> np.ndarray:
    counts = slide.counts[start:end].toarray().astype(np.float32, copy=False)
    sf = np.asarray(slide.size_factor[start:end], dtype=np.float32).reshape(-1, 1)
    if target_kind != "log1p_rate":
        raise ValueError(f"Unsupported target_kind: {target_kind}")
    return np.log1p(counts / np.clip(sf, 1.0e-6, None)).astype(np.float32, copy=False)


def _masked_mse_sum(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    measured_genes: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected 2D pred/target, got pred={pred.shape}, target={target.shape}")
    mask = measured_genes.bool()
    if mask.ndim == 1:
        mask = mask.view(1, -1).expand_as(target)
    elif mask.shape != target.shape:
        raise ValueError(f"Mask shape must be 1D or target-shaped, got mask={mask.shape}, target={target.shape}")
    valid = mask & torch.isfinite(pred) & torch.isfinite(target)
    if not torch.any(valid):
        return pred.sum() * 0.0, 0
    return (pred - target).pow(2)[valid].sum(), int(valid.sum().detach().cpu())


def _forward_slide_chunk(model: nn.Module, features_chunk: torch.Tensor) -> torch.Tensor:
    if features_chunk.ndim != 2:
        raise ValueError(f"Expected a 2D feature chunk, got {features_chunk.shape}")
    return _forward_spot_batch(model, features_chunk)


def _train_target_mean(slides: list[Path2SpaceSlide], *, target_kind: TargetKind = "log1p_rate") -> np.ndarray:
    if target_kind != "log1p_rate":
        raise ValueError(f"Unsupported target_kind: {target_kind}")
    if not slides:
        raise ValueError("Cannot compute target mean from an empty slide list.")
    n_genes = int(slides[0].measured_genes.shape[0])
    sum_y = np.zeros(n_genes, dtype=np.float64)
    n_y = np.zeros(n_genes, dtype=np.float64)
    for slide in slides:
        measured = np.asarray(slide.measured_genes, dtype=bool)
        n_y[measured] += float(slide.n_spots)
        counts = slide.counts.tocsr()
        indptr = counts.indptr
        indices = counts.indices
        data = counts.data
        for spot_idx in range(slide.n_spots):
            start = int(indptr[spot_idx])
            end = int(indptr[spot_idx + 1])
            if start == end:
                continue
            sf = max(float(slide.size_factor[spot_idx]), 1.0e-6)
            values = np.log1p(data[start:end].astype(np.float64, copy=False) / sf)
            sum_y[indices[start:end]] += values
    mean = np.zeros(n_genes, dtype=np.float32)
    seen = n_y > 0
    mean[seen] = (sum_y[seen] / n_y[seen]).astype(np.float32)
    return mean


class Path2SpaceRegressionMLP(nn.Module):
    def __init__(
        self,
        *,
        n_inputs: int,
        n_hiddens: int,
        n_outputs: int,
        dropout: float,
        bias_init: torch.Tensor | None = None,
    ):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(int(n_inputs), int(n_hiddens)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(int(n_hiddens), int(n_outputs)),
            nn.ReLU(),
        )
        if bias_init is not None:
            with torch.no_grad():
                self.layer1[0].bias.copy_(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = torch.mean(x, dim=0)
        return x

    def predict_spots(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(self.layer0(x))


def build_path2space_sourcefaithful_model(
    *,
    n_genes: int,
    model_cfg: dict[str, Any] | None = None,
) -> Path2SpaceRegressionMLP:
    cfg = dict(OFFICIAL_PATH2SPACE_MODEL_DEFAULTS)
    if model_cfg:
        cfg.update(model_cfg)
    return Path2SpaceRegressionMLP(
        n_inputs=int(cfg["n_inputs"]),
        n_hiddens=int(cfg["n_hiddens"]),
        n_outputs=int(n_genes),
        dropout=float(cfg["dropout"]),
    )


def _load_feature_array(feature_path: Path) -> np.ndarray:
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {feature_path}")
    return np.load(feature_path, mmap_mode="r")


def _target_matrix_for_local_indices(
    slide: Path2SpaceSlide,
    local_indices: np.ndarray,
    *,
    target_kind: TargetKind = "log1p_rate",
) -> np.ndarray:
    counts = slide.counts[np.asarray(local_indices, dtype=np.int64)].toarray().astype(np.float32, copy=False)
    sf = np.asarray(slide.size_factor[np.asarray(local_indices, dtype=np.int64)], dtype=np.float32).reshape(-1, 1)
    if target_kind != "log1p_rate":
        raise ValueError(f"Unsupported target_kind: {target_kind}")
    return np.log1p(counts / np.clip(sf, 1.0e-6, None)).astype(np.float32, copy=False)


class _MaskedVectorMetricAccumulator:
    def __init__(self, n_features: int):
        self.n = np.zeros(n_features, dtype=np.float64)
        self.sum_pred = np.zeros(n_features, dtype=np.float64)
        self.sum_true = np.zeros(n_features, dtype=np.float64)
        self.sum_pred2 = np.zeros(n_features, dtype=np.float64)
        self.sum_true2 = np.zeros(n_features, dtype=np.float64)
        self.sum_pred_true = np.zeros(n_features, dtype=np.float64)

    def update(self, pred: np.ndarray, true: np.ndarray, valid_mask: np.ndarray) -> None:
        valid = np.isfinite(pred) & np.isfinite(true) & valid_mask.astype(bool)
        x = np.where(valid, pred, 0.0).astype(np.float64)
        y = np.where(valid, true, 0.0).astype(np.float64)
        self.n += valid.sum(axis=0)
        self.sum_pred += x.sum(axis=0)
        self.sum_true += y.sum(axis=0)
        self.sum_pred2 += (x * x).sum(axis=0)
        self.sum_true2 += (y * y).sum(axis=0)
        self.sum_pred_true += (x * y).sum(axis=0)

    def summary(self) -> dict[str, float | int]:
        denom_n = np.maximum(self.n, 1.0)
        numerator = self.sum_pred_true - (self.sum_pred * self.sum_true / denom_n)
        pred_var = self.sum_pred2 - (self.sum_pred * self.sum_pred / denom_n)
        true_var = self.sum_true2 - (self.sum_true * self.sum_true / denom_n)
        denom = np.sqrt(np.maximum(pred_var, 0.0) * np.maximum(true_var, 0.0))
        pearson = np.full(self.n.shape[0], np.nan, dtype=np.float64)
        keep = (self.n >= 3) & (denom > 0)
        pearson[keep] = numerator[keep] / denom[keep]
        return {
            "mean_gene_pearson": float(np.nanmean(pearson)),
            "median_gene_pearson": float(np.nanmedian(pearson)),
            "valid_genes": int(np.isfinite(pearson).sum()),
        }


def _forward_spot_batch(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "predict_spots"):
        return model.predict_spots(features)  # type: ignore[attr-defined]
    if hasattr(model, "layer0") and hasattr(model, "layer1"):
        return model.layer1(model.layer0(features))  # type: ignore[attr-defined]
    pred = model(features.unsqueeze(1))
    if pred.ndim == 2:
        return pred
    raise ValueError(f"Unexpected Path2Space prediction shape: {pred.shape}")


def _open_feature_maps(slides: list[Path2SpaceSlide], feature_dir: str | Path) -> dict[int, np.ndarray]:
    feature_maps: dict[int, np.ndarray] = {}
    for idx, slide in enumerate(slides):
        feature_path = Path(feature_dir) / f"{slide.sample_id}.npy"
        features = _load_feature_array(feature_path)
        if features.ndim != 2:
            raise ValueError(f"Feature array must be 2D for {slide.sample_id}: {features.shape}")
        if features.shape[0] != slide.n_spots:
            raise ValueError(
                f"Feature spot mismatch for {slide.sample_id}: features={features.shape[0]} slide={slide.n_spots}"
            )
        feature_maps[idx] = features
    return feature_maps


def _spot_index_arrays(slides: list[Path2SpaceSlide]) -> tuple[np.ndarray, np.ndarray]:
    slide_parts = []
    local_parts = []
    for slide_idx, slide in enumerate(slides):
        slide_parts.append(np.full(slide.n_spots, slide_idx, dtype=np.int32))
        local_parts.append(np.arange(slide.n_spots, dtype=np.int32))
    if not slide_parts:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    return np.concatenate(slide_parts), np.concatenate(local_parts)


def _build_spot_batch(
    *,
    slides: list[Path2SpaceSlide],
    feature_maps: dict[int, np.ndarray],
    batch_slide_idx: np.ndarray,
    batch_local_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = int(batch_slide_idx.shape[0])
    feature_dim = int(next(iter(feature_maps.values())).shape[1])
    n_genes = int(slides[0].measured_genes.shape[0])
    features_np = np.empty((batch_size, feature_dim), dtype=np.float32)
    target_np = np.empty((batch_size, n_genes), dtype=np.float32)
    mask_np = np.zeros((batch_size, n_genes), dtype=bool)
    for slide_idx in np.unique(batch_slide_idx):
        pos = np.where(batch_slide_idx == slide_idx)[0]
        locals_for_slide = batch_local_idx[pos].astype(np.int64, copy=False)
        slide = slides[int(slide_idx)]
        features_np[pos] = np.asarray(feature_maps[int(slide_idx)][locals_for_slide], dtype=np.float32)
        target_np[pos] = _target_matrix_for_local_indices(slide, locals_for_slide)
        mask_np[pos] = np.asarray(slide.measured_genes, dtype=bool).reshape(1, -1)
    return features_np, target_np, mask_np


def run_path2space_epoch(
    *,
    model: nn.Module,
    slides: list[Path2SpaceSlide],
    feature_dir: str | Path,
    device: torch.device,
    spot_batch_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    shuffle_slides: bool = False,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    gene_acc = _MaskedVectorMetricAccumulator(int(slides[0].measured_genes.shape[0])) if slides else None
    total_loss_sum = 0.0
    total_valid = 0
    if int(spot_batch_size) <= 0:
        raise ValueError("spot_batch_size must be positive.")
    feature_maps = _open_feature_maps(slides, feature_dir)
    slide_idx_all, local_idx_all = _spot_index_arrays(slides)
    order = np.arange(slide_idx_all.shape[0], dtype=np.int64)
    if shuffle_slides:
        np.random.shuffle(order)
    for start in range(0, order.shape[0], int(spot_batch_size)):
        batch_order = order[start : start + int(spot_batch_size)]
        batch_slide_idx = slide_idx_all[batch_order]
        batch_local_idx = local_idx_all[batch_order]
        features_np, target_np, mask_np = _build_spot_batch(
            slides=slides,
            feature_maps=feature_maps,
            batch_slide_idx=batch_slide_idx,
            batch_local_idx=batch_local_idx,
        )
        features = torch.from_numpy(features_np).to(device)
        target = torch.from_numpy(target_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        with torch.set_grad_enabled(training):
            pred = _forward_spot_batch(model, features)
            loss_sum, n_valid = _masked_mse_sum(pred=pred, target=target, measured_genes=mask)
            loss = loss_sum / float(max(n_valid, 1))
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss_sum += float(loss_sum.detach().cpu())
        total_valid += int(n_valid)
        if gene_acc is not None:
            gene_acc.update(
                pred.detach().cpu().numpy().astype(np.float32, copy=False),
                target_np,
                mask_np,
            )
    used_spots = [int(slide.n_spots) for slide in slides]
    return {
        "loss": float(total_loss_sum / max(total_valid, 1)),
        "mean_gene_pearson": float(gene_acc.summary()["mean_gene_pearson"]) if gene_acc is not None else float("nan"),
        "median_gene_pearson": float(gene_acc.summary()["median_gene_pearson"]) if gene_acc is not None else float("nan"),
        "valid_genes": int(gene_acc.summary()["valid_genes"]) if gene_acc is not None else 0,
        "n_slides": int(len(slides)),
        "n_spots": int(sum(used_spots)),
        "max_used_spots": int(max(used_spots)) if used_spots else 0,
    }


def _save_history_csv(history: list[dict[str, Any]], out_dir: Path) -> None:
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)


def train_path2space_sourcefaithful(
    *,
    expression_config: dict[str, Any],
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    output_dir: str | Path,
    data_dir: str | Path,
    upstream_root: str | Path | None = None,
    ctranspath_weight_path: str | Path | None = None,
    feature_batch_size: int = 128,
    spot_batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1.0e-4,
    patience: int | None = None,
    min_delta: float = 0.0,
    seed: int = 2026,
    device_name: str | None = None,
    model_cfg: dict[str, Any] | None = None,
    use_macenko: bool = False,
    overwrite_features: bool = False,
) -> dict[str, Any]:
    set_seed(int(seed))
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    _enforce_fp32_cuda(device)
    print(
        f"[path2space-sourcefaithful] resolved_device={device}"
        + (f" cuda_name={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
        flush=True,
    )
    data_dir = Path(data_dir)
    feature_dir = data_dir / "features"
    train_collection = build_path2space_collection(
        expression_config,
        splits=[str(x) for x in train_rows["split"].astype(str).unique().tolist()],
        slide_ids=[str(x) for x in train_rows["sample_id"].tolist()],
        feature_dir=feature_dir,
        upstream_root=upstream_root,
    )
    val_collection = build_path2space_collection(
        expression_config,
        splits=[str(x) for x in val_rows["split"].astype(str).unique().tolist()],
        slide_ids=[str(x) for x in val_rows["sample_id"].tolist()],
        feature_dir=feature_dir,
        upstream_root=upstream_root,
    )
    prepare_path2space_feature_cache(
        expression_config=expression_config,
        rows=pd.concat([train_rows, val_rows], ignore_index=True).drop_duplicates("sample_id"),
        data_dir=data_dir,
        upstream_root=upstream_root,
        ctranspath_weight_path=ctranspath_weight_path,
        feature_batch_size=feature_batch_size,
        device_name=device_name,
        use_macenko=use_macenko,
        overwrite=overwrite_features,
    )
    model = build_path2space_sourcefaithful_model(
        n_genes=len(train_collection.target_genes),
        model_cfg=model_cfg,
    ).to(device)
    bias_init = torch.from_numpy(_train_target_mean(train_collection.slides)).to(device)
    with torch.no_grad():
        model.layer1[0].bias.copy_(bias_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_collection.slide_summary_frame().to_csv(out / "train_slides.csv", index=False)
    val_collection.slide_summary_frame().to_csv(out / "val_slides.csv", index=False)
    best_path = out / "best.pt"
    history: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    patience = int(max(1, int(epochs) // 10)) if patience is None else int(patience)
    for epoch in range(1, int(epochs) + 1):
        train_stats = run_path2space_epoch(
            model=model,
            slides=train_collection.slides,
            feature_dir=feature_dir,
            device=device,
            spot_batch_size=spot_batch_size,
            optimizer=optimizer,
            shuffle_slides=True,
        )
        val_stats = run_path2space_epoch(
            model=model,
            slides=val_collection.slides,
            feature_dir=feature_dir,
            device=device,
            spot_batch_size=spot_batch_size,
            optimizer=None,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_stats["loss"]),
            "train_mean_gene_pearson": float(train_stats["mean_gene_pearson"]),
            "train_median_gene_pearson": float(train_stats["median_gene_pearson"]),
            "val_loss": float(val_stats["loss"]),
            "val_mean_gene_pearson": float(val_stats["mean_gene_pearson"]),
            "val_median_gene_pearson": float(val_stats["median_gene_pearson"]),
            "train_max_used_spots": int(train_stats["max_used_spots"]),
            "val_max_used_spots": int(val_stats["max_used_spots"]),
        }
        history.append(row)
        print(
            f"[path2space-sourcefaithful] epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"val_mean_gene_pearson={row['val_mean_gene_pearson']:.6f}",
            flush=True,
        )
        improved = row["val_mean_gene_pearson"] > (best_val_score + float(min_delta))
        if improved:
            best_val_score = float(row["val_mean_gene_pearson"])
            best_val_loss = float(row["val_loss"])
            best_epoch = int(epoch)
            stale_epochs = 0
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    model,
                    {
                        "method": "path2space_sourcefaithful_full_fp32",
                        "model": dict(OFFICIAL_PATH2SPACE_MODEL_DEFAULTS, **(model_cfg or {})),
                        "target_kind": "log1p_rate",
                        "train_splits": [str(x) for x in train_rows["split"].astype(str).unique().tolist()],
                        "val_splits": [str(x) for x in val_rows["split"].astype(str).unique().tolist()],
                        "feature_cache_dir": str(feature_dir),
                        "source": _jsonable_source_info(
                            upstream_root,
                            ctranspath_weight_path=ctranspath_weight_path,
                        ),
                    },
                    extra={
                        "n_genes": int(len(train_collection.target_genes)),
                        "genes": train_collection.target_genes,
                        "feature_dim": int(OFFICIAL_PATH2SPACE_MODEL_DEFAULTS["n_inputs"]),
                        "best_val_mean_gene_pearson": best_val_score,
                        "best_val_loss": best_val_loss,
                        "history": history,
                        "spot_batch_size": int(spot_batch_size),
                        "feature_batch_size": int(feature_batch_size),
                    },
                ),
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(patience):
            stopped_early = True
            print(
                f"[path2space-sourcefaithful] early_stop epoch={epoch:03d} "
                f"best_epoch={best_epoch:03d} best_val_pearson={best_val_score:.6f} "
                f"patience={int(patience)}",
                flush=True,
            )
            break
    _save_history_csv(history, out)
    formal_benchmark_candidate = bool(
        [str(x) for x in train_rows["split"].astype(str).unique().tolist()] == ["train"]
        and [str(x) for x in val_rows["split"].astype(str).unique().tolist()] == ["val"]
    )
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is not None:
        manifest = read_manifest(manifest_path)
        formal_benchmark_candidate = bool(
            formal_benchmark_candidate
            and len(train_rows) == int(manifest["split"].astype(str).eq("train").sum())
            and len(val_rows) == int(manifest["split"].astype(str).eq("val").sum())
        )
    summary = {
        "checkpoint": str(best_path),
        "device": str(device),
        "method": "path2space_sourcefaithful_full_fp32",
        "target_kind": "log1p_rate",
        "precision": "fp32",
        "amp": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None,
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else None,
        "source_faithful_core": True,
        "feature_extraction_stage": "official CTransPath",
        "regression_stage": "official two-layer MLP with ReLU",
        "formal_benchmark_candidate": formal_benchmark_candidate,
        "model": dict(OFFICIAL_PATH2SPACE_MODEL_DEFAULTS, **(model_cfg or {})),
        "epochs": int(len(history)),
        "max_epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_mean_gene_pearson": float(best_val_score),
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "stopped_early": bool(stopped_early),
        "spot_batch_size": int(spot_batch_size),
        "feature_batch_size": int(feature_batch_size),
        "train_splits": [str(x) for x in train_rows["split"].astype(str).unique().tolist()],
        "val_splits": [str(x) for x in val_rows["split"].astype(str).unique().tolist()],
        "n_train_slides": int(len(train_collection.slides)),
        "n_val_slides": int(len(val_collection.slides)),
        "n_genes": int(len(train_collection.target_genes)),
        "history": history,
        "source": _jsonable_source_info(
            upstream_root,
            ctranspath_weight_path=ctranspath_weight_path,
        ),
        "outputs": {
            "checkpoint": str(best_path),
            "train_slides": str(out / "train_slides.csv"),
            "val_slides": str(out / "val_slides.csv"),
            "training_history": str(out / "training_history.csv"),
            "summary": str(out / "train_summary.json"),
        },
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_path2space_sourcefaithful_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Path2SpaceRegressionMLP, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = dict(OFFICIAL_PATH2SPACE_MODEL_DEFAULTS)
    cfg.update(ckpt.get("config", {}).get("model", {}))
    model = build_path2space_sourcefaithful_model(
        n_genes=int(ckpt["n_genes"]),
        model_cfg=cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def export_path2space_sourcefaithful_predictions(
    *,
    expression_config: dict[str, Any],
    checkpoint_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    splits: list[str],
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_slide_spots: int | None = None,
    feature_batch_size: int = 128,
    spot_batch_size: int = 128,
    device_name: str | None = None,
    upstream_root: str | Path | None = None,
    ctranspath_weight_path: str | Path | None = None,
    use_macenko: bool = False,
    overwrite_features: bool = False,
) -> dict[str, Any]:
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    _enforce_fp32_cuda(device)
    model, ckpt = load_path2space_sourcefaithful_checkpoint(checkpoint_path, device)
    target_kind = str(ckpt["config"].get("target_kind", "log1p_rate"))
    data_dir = Path(data_dir)
    feature_dir = data_dir / "features"
    collection = build_path2space_collection(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        max_slide_spots=max_slide_spots,
        feature_dir=feature_dir,
        upstream_root=upstream_root,
    )
    rows = select_manifest_rows(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        max_slide_spots=max_slide_spots,
    )
    prepare_path2space_feature_cache(
        expression_config=expression_config,
        rows=rows,
        data_dir=data_dir,
        upstream_root=upstream_root,
        ctranspath_weight_path=ctranspath_weight_path,
        feature_batch_size=feature_batch_size,
        device_name=device_name,
        use_macenko=use_macenko,
        overwrite=overwrite_features,
    )
    out = Path(out_dir)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "genes.txt").write_text("\n".join(ckpt["genes"]) + "\n", encoding="utf-8")
    slide_rows = []
    all_complete = True
    for slide in collection.slides:
        feature_path = feature_dir / f"{slide.sample_id}.npy"
        features = _load_feature_array(feature_path)
        pred_path = pred_dir / f"{slide.sample_id}_{target_kind}.npy"
        writer = open_memmap(
            pred_path,
            mode="w+",
            dtype=np.float32,
            shape=(slide.n_spots, int(ckpt["n_genes"])),
        )
        offset = 0
        with torch.inference_mode():
            for start in range(0, slide.n_spots, int(spot_batch_size)):
                end = min(start + int(spot_batch_size), slide.n_spots)
                feat_chunk = torch.from_numpy(np.array(features[start:end], dtype=np.float32, copy=True)).to(device)
                pred_chunk = _forward_slide_chunk(model, feat_chunk)
                pred_np = pred_chunk.detach().cpu().numpy().astype(np.float32, copy=False)
                writer[offset : offset + pred_np.shape[0]] = pred_np
                offset += int(pred_np.shape[0])
        writer.flush()
        slide_complete = int(offset) == int(slide.n_spots)
        all_complete = all_complete and slide_complete
        slide_rows.append(
            {
                "sample_id": slide.sample_id,
                "expected_spots": int(slide.n_spots),
                "n_predicted_spots": int(offset),
                "n_genes": int(ckpt["n_genes"]),
                "complete_slide_prediction": bool(slide_complete),
                "truncated_for_smoke": False,
                "feature_path": str(feature_path),
                "prediction_path": str(pred_path),
            }
        )
        print(
            f"[path2space-export] {slide.sample_id}: spots={offset}/{slide.n_spots} genes={int(ckpt['n_genes'])}",
            flush=True,
        )
    summary = {
        "checkpoint": str(checkpoint_path),
        "method": "path2space_sourcefaithful_full_fp32",
        "prediction_kind": target_kind,
        "splits": [str(x) for x in splits],
        "slide_ids": None if slide_ids is None else [str(x) for x in slide_ids],
        "n_slides": int(len(slide_rows)),
        "n_genes": int(ckpt["n_genes"]),
        "all_slide_predictions_complete": bool(all_complete),
        "benchmark_evaluable_without_truncation": bool(all_complete and max_slide_spots is None),
        "feature_batch_size": int(feature_batch_size),
        "spot_batch_size": int(spot_batch_size),
        "slides": slide_rows,
        "outputs": {
            "prediction_root": str(out),
            "genes": str(out / "genes.txt"),
            "predictions": str(pred_dir),
            "summary": str(out / "prediction_summary.json"),
        },
    }
    (out / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_path2space_predictions(
    *,
    expression_config: dict[str, Any],
    prediction_root: str | Path,
    out_dir: str | Path,
    splits: list[str],
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    max_slide_spots: int | None = None,
    method_name: str = "path2space_sourcefaithful_full_fp32",
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


def data_smoke_summary(
    *,
    expression_config: dict[str, Any],
    splits: list[str],
    output_dir: str | Path,
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_slide_spots: int | None = None,
    feature_batch_size: int = 128,
    spot_batch_size: int = 128,
    upstream_root: str | Path | None = None,
    ctranspath_weight_path: str | Path | None = None,
    device_name: str | None = None,
    use_macenko: bool = False,
    overwrite_features: bool = False,
) -> dict[str, Any]:
    rows = select_manifest_rows(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        max_slide_spots=max_slide_spots,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_summary = prepare_path2space_feature_cache(
        expression_config=expression_config,
        rows=rows,
        data_dir=out / "cache",
        upstream_root=upstream_root,
        ctranspath_weight_path=ctranspath_weight_path,
        feature_batch_size=feature_batch_size,
        device_name=device_name,
        use_macenko=use_macenko,
        overwrite=overwrite_features,
    )
    collection = build_path2space_collection(
        expression_config,
        splits=splits,
        slide_ids=slide_ids,
        max_slides=max_slides,
        smallest_slides=smallest_slides,
        max_slide_spots=max_slide_spots,
        feature_dir=out / "cache" / "features",
        upstream_root=upstream_root,
    )
    first_slide = collection.slides[0]
    feature_path = out / "cache" / "features" / f"{first_slide.sample_id}.npy"
    features = np.load(feature_path, mmap_mode="r")
    target = _target_matrix_for_slide_chunk(first_slide, start=0, end=first_slide.n_spots)
    device = torch.device(get_device_name(device_name or expression_config.get("device")))
    _enforce_fp32_cuda(device)
    model = build_path2space_sourcefaithful_model(n_genes=len(collection.target_genes)).to(device)
    with torch.inference_mode():
        chunk = torch.from_numpy(
            np.array(features[: min(first_slide.n_spots, spot_batch_size)], dtype=np.float32, copy=True)
        ).to(device)
        pred_chunk = _forward_slide_chunk(model, chunk)
    summary = {
        "splits": [str(x) for x in splits],
        "slide_ids": None if slide_ids is None else [str(x) for x in slide_ids],
        "max_slides": None if max_slides is None else int(max_slides),
        "smallest_slides": bool(smallest_slides),
        "max_slide_spots": None if max_slide_spots is None else int(max_slide_spots),
        "feature_batch_size": int(feature_batch_size),
        "spot_batch_size": int(spot_batch_size),
        "overwrite_features": bool(overwrite_features),
        "n_slides": int(len(collection.slides)),
        "n_genes": int(len(collection.target_genes)),
        "cache_summary": cache_summary,
        "first_slide": {
            "sample_id": first_slide.sample_id,
            "patch_h5_path": str(first_slide.patch_h5_path),
            "feature_path": str(feature_path),
            "feature_shape": list(features.shape),
            "feature_min": float(np.nanmin(np.asarray(features))),
            "feature_max": float(np.nanmax(np.asarray(features))),
            "target_shape": list(target.shape),
            "target_min": float(np.nanmin(target)),
            "target_max": float(np.nanmax(target)),
            "n_spots": int(first_slide.n_spots),
            "n_measured_target_genes": int(first_slide.measured_genes.sum()),
            "spot_batch_probe_shape": list(pred_chunk.shape),
            "spot_batch_probe_finite_fraction": float(np.mean(np.isfinite(pred_chunk.detach().cpu().numpy()))),
        },
        "source": _jsonable_source_info(
            upstream_root,
            ctranspath_weight_path=ctranspath_weight_path,
        ),
        "outputs": {
            "slides": str(out / "cache" / "features" / "feature_manifest.csv"),
            "summary": str(out / "data_smoke_summary.json"),
        },
    }
    (out / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "data_smoke_slides.csv").write_text(collection.slide_summary_frame().to_csv(index=False), encoding="utf-8")
    return summary
