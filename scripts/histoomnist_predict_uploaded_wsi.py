from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from histoomnist.inference.count_scale import (  # noqa: E402
    COUNT_LOG1P_PREFIX,
    add_count_scale_columns as add_reconstructed_count_columns,
    add_count_scale_programs,
    mean_one_sf_from_log,
)


DEFAULT_EXPRESSION_CHECKPOINT = ROOT / "checkpoints/hest1k_human_visium_expression/highconf_symbol95_rate/best.pt"
DEFAULT_SF_CHECKPOINT = ROOT / "checkpoints/hest1k_human_visium_sf/context_distribution_light_hipt256_leave_slide_out/best.pt"
DEFAULT_SF_CONFIG = ROOT / "configs/hest1k_human_visium_sf_context_distribution_light.yaml"
DEFAULT_HIPT_SOURCE = ROOT / "third_party/HIPT/1-Hierarchical-Pretraining"
DEFAULT_HIPT_WEIGHTS = ROOT / "third_party/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth"
DEFAULT_TARGET_CONFIG = ROOT / "configs" / "manuscript_release_28_gene_panel.json"

DEFAULT_GENES = [
    "EPCAM",
    "KRT8",
    "KRT18",
    "KRT19",
    "CD3D",
    "CD3E",
    "CD8A",
    "CD4",
    "LST1",
    "AIF1",
    "C1QA",
    "C1QB",
    "COL1A1",
    "COL1A2",
    "DCN",
    "LUM",
    "ACTA2",
    "MKI67",
    "TOP2A",
    "PCNA",
    "VEGFA",
    "CA9",
    "SLC2A1",
    "VIM",
    "TGFB1",
    "ZEB1",
    "SNAI2",
    "AXIN2",
    "LGR5",
    "MYC",
    "ASCL2",
]

PROGRAMS = {
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19"],
    "t_cell": ["CD3D", "CD3E", "CD8A", "CD4"],
    "myeloid": ["LST1", "AIF1", "C1QA", "C1QB"],
    "stromal": ["COL1A1", "COL1A2", "DCN", "LUM", "ACTA2"],
    "proliferation": ["MKI67", "TOP2A", "PCNA"],
    "hypoxia": ["VEGFA", "CA9", "SLC2A1"],
    "emt_tgfb": ["VIM", "TGFB1", "ZEB1", "SNAI2"],
    "wnt_crc": ["AXIN2", "LGR5", "MYC", "ASCL2"],
}


def parse_gene_list(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_GENES)
    genes: list[str] = []
    for value in values:
        for item in str(value).split(","):
            gene = item.strip()
            if gene and gene not in genes:
                genes.append(gene)
    return genes or list(DEFAULT_GENES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HistoOmniST selected-gene virtual spatial transcriptomics on user supplied H&E WSI/image files."
        )
    )
    parser.add_argument("--input", dest="inputs", action="append", default=None, help="Input WSI/image path. Repeat for multiple files.")
    parser.add_argument("--input-list", type=Path, default=None, help="Text or CSV file containing input image paths.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expression-checkpoint", type=Path, default=DEFAULT_EXPRESSION_CHECKPOINT)
    parser.add_argument("--sf-checkpoint", type=Path, default=DEFAULT_SF_CHECKPOINT)
    parser.add_argument("--sf-config", type=Path, default=DEFAULT_SF_CONFIG)
    parser.add_argument("--hipt-source", "--hipt-source-dir", dest="hipt_source_dir", type=Path, default=DEFAULT_HIPT_SOURCE)
    parser.add_argument("--hipt-weights", type=Path, default=DEFAULT_HIPT_WEIGHTS)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    parser.add_argument("--selected-genes", nargs="+", default=None)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--target-tiles-per-slide", type=int, default=0, help="If positive, adapt stride per slide to approach this many tissue tiles.")
    parser.add_argument("--min-stride", type=int, default=224, help="Smallest stride allowed when --target-tiles-per-slide is enabled.")
    parser.add_argument("--thumbnail-max-dim", type=int, default=2048)
    parser.add_argument("--tissue-threshold", type=float, default=0.25)
    parser.add_argument("--max-tiles-per-slide", type=int, default=5000, help="0 means no cap.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--write-zip", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def safe_stem(path: Path, fallback_index: int) -> str:
    stem = path.stem or f"input_{fallback_index}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("._") or f"input_{fallback_index}"


def load_target_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_input_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        column = "image_path" if "image_path" in frame.columns else frame.columns[0]
        return [str(value) for value in frame[column].dropna().tolist()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rgb_tissue_mask(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    rgb_max = x.max(axis=2)
    rgb_min = x.min(axis=2)
    gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    saturation = (rgb_max - rgb_min) / np.clip(rgb_max, 1.0e-6, None)
    return (gray < 0.92) & (saturation > 0.05)


def tile_tissue_fraction(mask: np.ndarray, x: int, y: int, tile_size: int, scale_x: float, scale_y: float) -> float:
    x0 = max(0, int(math.floor(x * scale_x)))
    y0 = max(0, int(math.floor(y * scale_y)))
    x1 = min(mask.shape[1], int(math.ceil((x + tile_size) * scale_x)))
    y1 = min(mask.shape[0], int(math.ceil((y + tile_size) * scale_y)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def select_tile_grid(
    *,
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    mask: np.ndarray,
    tissue_threshold: float,
    max_tiles: int,
) -> pd.DataFrame:
    scale_x = mask.shape[1] / max(float(width), 1.0)
    scale_y = mask.shape[0] / max(float(height), 1.0)
    rows: list[dict[str, object]] = []
    for y in range(0, max(1, height - tile_size + 1), stride):
        for x in range(0, max(1, width - tile_size + 1), stride):
            frac = tile_tissue_fraction(mask, x, y, tile_size, scale_x, scale_y)
            if frac >= tissue_threshold:
                rows.append({"x": int(x), "y": int(y), "tile_size": int(tile_size), "stride": int(stride), "tissue_fraction": float(frac)})
    if not rows:
        return pd.DataFrame(columns=["tile_index", "x", "y", "tile_size", "stride", "tissue_fraction"])
    frame = pd.DataFrame(rows).reset_index(drop=True)
    if max_tiles > 0 and len(frame) > max_tiles:
        frame = frame.sort_values("tissue_fraction", ascending=False).head(max_tiles).sort_index().reset_index(drop=True)
    frame.insert(0, "tile_index", np.arange(len(frame), dtype=int))
    return frame


def stride_candidates(base_stride: int, min_stride: int, tile_size: int) -> list[int]:
    base = max(1, int(base_stride))
    lower = max(1, min(int(min_stride), base))
    anchors = [2048, 1536, 1024, 768, 512, 384, 320, 256, tile_size, 192, 160, 128, 112, 96, 80, 64, 48, 32]
    values = {base, lower}
    values.update(int(value) for value in anchors)
    return sorted([value for value in values if lower <= value <= base], reverse=True)


def choose_adaptive_stride(
    *,
    width: int,
    height: int,
    tile_size: int,
    requested_stride: int,
    mask: np.ndarray,
    tissue_threshold: float,
    max_tiles: int,
    target_tiles: int,
    min_stride: int,
) -> tuple[int, dict[str, Any]]:
    if int(target_tiles) <= 0:
        return int(requested_stride), {
            "enabled": False,
            "requested_stride": int(requested_stride),
            "effective_stride": int(requested_stride),
        }

    estimates: list[dict[str, Any]] = []
    for candidate in stride_candidates(int(requested_stride), int(min_stride), int(tile_size)):
        tiles = select_tile_grid(
            width=int(width),
            height=int(height),
            tile_size=int(tile_size),
            stride=int(candidate),
            mask=mask,
            tissue_threshold=float(tissue_threshold),
            max_tiles=0,
        )
        estimates.append({"stride": int(candidate), "estimated_tissue_tiles": int(len(tiles))})

    if not estimates:
        return int(requested_stride), {
            "enabled": True,
            "decision": "no_candidates",
            "requested_stride": int(requested_stride),
            "effective_stride": int(requested_stride),
            "target_tiles": int(target_tiles),
        }

    cap = int(max_tiles)
    target = int(target_tiles)
    within_cap = [row for row in estimates if cap <= 0 or int(row["estimated_tissue_tiles"]) <= cap]
    target_reached = [row for row in within_cap if int(row["estimated_tissue_tiles"]) >= target]
    if target_reached:
        selected = target_reached[0]
        decision = "target_reached_within_cap"
    elif within_cap:
        selected = max(within_cap, key=lambda row: int(row["estimated_tissue_tiles"]))
        decision = "densest_within_cap_below_target"
    else:
        selected = min(estimates, key=lambda row: int(row["estimated_tissue_tiles"]))
        decision = "all_candidates_exceed_cap"

    return int(selected["stride"]), {
        "enabled": True,
        "decision": decision,
        "requested_stride": int(requested_stride),
        "effective_stride": int(selected["stride"]),
        "target_tiles": target,
        "max_tiles": cap,
        "min_stride": int(min_stride),
        "candidate_estimates": estimates,
    }


class PILSlide:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.image = Image.open(path).convert("RGB")
        self.dimensions = self.image.size
        self.level_count = 1
        self.level_dimensions = [self.image.size]
        self.properties: dict[str, str] = {}

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        thumb = self.image.copy()
        thumb.thumbnail(size)
        return thumb

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        del level
        x, y = location
        w, h = size
        crop = self.image.crop((int(x), int(y), int(x) + int(w), int(y) + int(h)))
        if crop.size != (w, h):
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            canvas.paste(crop, (0, 0))
            crop = canvas
        return crop.convert("RGB")

    def close(self) -> None:
        self.image.close()


class TiffFileSlide:
    """Open OME-TIF / tiled TIFF through tifffile's zarr interface."""

    def __init__(self, path: Path):
        self.path = Path(path)
        import tifffile
        import zarr

        self._zarr = zarr
        self._tif = tifffile.TiffFile(path)
        if not self._tif.series:
            self._tif.close()
            raise ValueError(f"No TIFF image series found in {path}")
        self._series = next((series for series in self._tif.series if "Y" in series.axes and "X" in series.axes), self._tif.series[0])
        self._levels = list(getattr(self._series, "levels", []) or [self._series])
        if not self._levels:
            self._tif.close()
            raise ValueError(f"No TIFF pyramid levels found in {path}")
        self._stores: dict[int, Any] = {}
        self._arrays: dict[int, Any] = {}
        self.dimensions = self._level_dimensions(0)
        self.level_count = len(self._levels)
        self.level_dimensions = [self._level_dimensions(level) for level in range(self.level_count)]
        self.properties: dict[str, str] = {
            "histoomnist.reader": "tifffile_zarr",
            "tiff.is_ome": str(bool(getattr(self._tif, "is_ome", False))),
            "tiff.is_bigtiff": str(bool(getattr(self._tif, "is_bigtiff", False))),
        }

    def _level_dimensions(self, level: int) -> tuple[int, int]:
        axes = self._levels[level].axes
        shape = self._levels[level].shape
        if "Y" not in axes or "X" not in axes:
            raise ValueError(f"Unsupported TIFF axes {axes!r}; expected axes containing Y and X.")
        return int(shape[axes.index("X")]), int(shape[axes.index("Y")])

    def _array_for_level(self, level: int):
        level = int(level)
        if level < 0 or level >= self.level_count:
            raise ValueError(f"TIFF level {level} is outside available range 0..{self.level_count - 1}.")
        if level not in self._arrays:
            store = self._series.aszarr(level=level)
            self._stores[level] = store
            self._arrays[level] = self._zarr.open(store, mode="r")
        return self._arrays[level]

    @staticmethod
    def _to_rgb(array: np.ndarray, axes: str) -> np.ndarray:
        data = np.asarray(array)
        local_axes = axes
        for axis in reversed(range(len(local_axes))):
            if local_axes[axis] not in {"Y", "X", "S"}:
                if data.shape[axis] != 1:
                    raise ValueError(f"Unsupported non-singleton TIFF axis {local_axes[axis]!r} in axes {axes!r}.")
                data = np.take(data, 0, axis=axis)
                local_axes = local_axes[:axis] + local_axes[axis + 1 :]
        if "Y" in local_axes and "X" in local_axes:
            order = [local_axes.index("Y"), local_axes.index("X")]
            if "S" in local_axes:
                order.append(local_axes.index("S"))
            data = np.transpose(data, order)
        if data.ndim == 2:
            data = np.repeat(data[:, :, None], 3, axis=2)
        elif data.ndim == 3:
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
            elif data.shape[2] >= 3:
                data = data[:, :, :3]
            else:
                raise ValueError(f"Unsupported TIFF sample dimension {data.shape}.")
        else:
            raise ValueError(f"Unsupported TIFF array shape {data.shape}.")
        if data.dtype != np.uint8:
            if np.issubdtype(data.dtype, np.integer):
                info = np.iinfo(data.dtype)
                data = (data.astype(np.float32) / max(float(info.max), 1.0) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                data = np.asarray(data, dtype=np.float32)
                data = (data * 255.0 if data.max(initial=0.0) <= 1.5 else data).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(data)

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        max_w, max_h = int(size[0]), int(size[1])
        chosen_level = self.level_count - 1
        for level, (level_w, level_h) in enumerate(self.level_dimensions):
            if level_w <= max_w and level_h <= max_h:
                chosen_level = level
                break
        data = np.asarray(self._array_for_level(chosen_level)[:])
        image = Image.fromarray(self._to_rgb(data, self._levels[chosen_level].axes), mode="RGB")
        image.thumbnail(size)
        return image

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        x, y = int(location[0]), int(location[1])
        w, h = int(size[0]), int(size[1])
        level = int(level)
        level_w, level_h = self._level_dimensions(level)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(level_w, x + w)
        y1 = min(level_h, y + h)
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        if x1 > x0 and y1 > y0:
            data = np.asarray(self._array_for_level(level)[y0:y1, x0:x1, ...])
            rgb = self._to_rgb(data, self._levels[level].axes)
            paste_x = x0 - x
            paste_y = y0 - y
            canvas[paste_y : paste_y + rgb.shape[0], paste_x : paste_x + rgb.shape[1], :] = rgb
        return Image.fromarray(canvas, mode="RGB")

    def close(self) -> None:
        for store in self._stores.values():
            close = getattr(store, "close", None)
            if close is not None:
                close()
        self._stores.clear()
        self._arrays.clear()
        self._tif.close()


def open_slide(path: Path):
    try:
        import openslide

        return openslide.OpenSlide(str(path)), "openslide"
    except Exception as openslide_error:
        suffix = path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            try:
                return TiffFileSlide(path), "tifffile_zarr"
            except Exception as tifffile_error:
                try:
                    return PILSlide(path), "pil"
                except Exception as pil_error:
                    raise RuntimeError(
                        "Failed to open TIFF image with OpenSlide, tifffile/zarr, and PIL. "
                        f"OpenSlide error: {openslide_error}; tifffile error: {tifffile_error}; PIL error: {pil_error}"
                    ) from pil_error
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise openslide_error
        return PILSlide(path), "pil"


def read_tile_batch(slide, tile_frame: pd.DataFrame, tile_size: int, start: int, end: int) -> np.ndarray:
    images: list[np.ndarray] = []
    for row in tile_frame.iloc[start:end].itertuples(index=False):
        image = slide.read_region((int(row.x), int(row.y)), 0, (tile_size, tile_size)).convert("RGB")
        if image.size != (tile_size, tile_size):
            image = image.resize((tile_size, tile_size), resample=Image.BILINEAR)
        images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images, axis=0)


def load_rate_model(checkpoint_path: Path, device: str):
    import torch
    from histoomnist.models.expression_mlp import ExpressionRateRegressor
    from histoomnist.models.gene_conditioned import GeneConditionedRateRegressor

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = int(ckpt.get("input_dim", len(ckpt.get("feature_mean", []))))
    genes = list(ckpt.get("genes", []))
    output_dim = int(ckpt.get("output_dim", len(genes)))
    model_name = ckpt.get("model_name", "gene_conditioned")
    kwargs = dict(ckpt.get("model_kwargs", {}))
    if model_name == "gene_conditioned":
        model = GeneConditionedRateRegressor(
            input_dim=input_dim,
            num_genes=output_dim,
            latent_dim=int(kwargs.get("latent_dim", 256)),
            hidden_dims=list(kwargs.get("hidden_dims", [768, 384])),
            dropout=float(kwargs.get("dropout", 0.15)),
        )
    else:
        model = ExpressionRateRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=list(kwargs.get("hidden_dims", [768, 384])),
            dropout=float(kwargs.get("dropout", 0.20)),
        )
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model.to(torch.device(device))
    model.eval()
    mean = np.asarray(ckpt.get("feature_mean"), dtype=np.float32)
    std = np.asarray(ckpt.get("feature_std"), dtype=np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return model, genes, mean, std


def load_sf_model(sf_config_path: Path, checkpoint_path: Path, device: str):
    import torch
    from histoomnist.eval.evaluate_combined import _load_sf_model
    from histoomnist.train.common import load_checkpoint
    from histoomnist.utils.config import load_config

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        cfg = load_config(sf_config_path) if sf_config_path.exists() else {"model": {}}
    model = _load_sf_model(cfg, ckpt, torch.device(device))
    mean = np.asarray(ckpt.get("feature_mean"), dtype=np.float32)
    std = np.asarray(ckpt.get("feature_std"), dtype=np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return model, mean, std


def features_for_checkpoint(raw_features: np.ndarray, tile_frame: pd.DataFrame, expected_dim: int) -> np.ndarray:
    if raw_features.shape[1] == expected_dim:
        return raw_features.astype(np.float32, copy=False)
    from hest_make_context_features import build_context_features

    coords = np.stack(
        [
            tile_frame["x"].to_numpy(np.float32) + tile_frame["tile_size"].to_numpy(np.float32) / 2.0,
            tile_frame["y"].to_numpy(np.float32) + tile_frame["tile_size"].to_numpy(np.float32) / 2.0,
        ],
        axis=1,
    )
    context_features, _ = build_context_features(
        features=raw_features.astype(np.float32, copy=False),
        coords=coords.astype(np.float32, copy=False),
        ks=[8, 24],
        chunk_size=4096,
    )
    if context_features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch after context construction: raw={raw_features.shape[1]} "
            f"context={context_features.shape[1]} checkpoint_expected={expected_dim}"
        )
    return context_features.astype(np.float32, copy=False)


def predict_rate_selected(
    *,
    features: np.ndarray,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    selected_indices: list[int],
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    x = ((features.astype(np.float32) - mean[None, :]) / std[None, :]).astype(np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        tensor = torch.from_numpy(x[start : start + batch_size]).to(torch.device(device))
        with torch.inference_mode():
            pred = model(tensor).detach().cpu().numpy().astype(np.float32)
        chunks.append(pred[:, selected_indices])
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(selected_indices)), dtype=np.float32)


def predict_sf(
    *,
    features: np.ndarray,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x = ((features.astype(np.float32) - mean[None, :]) / std[None, :]).astype(np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        tensor = torch.from_numpy(x[start : start + batch_size]).to(torch.device(device))
        with torch.inference_mode():
            pred_log_sf = model(tensor).detach().cpu().numpy().reshape(-1).astype(np.float32)
        chunks.append(pred_log_sf)
    raw_log_sf = np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)
    return mean_one_sf_from_log(raw_log_sf)


def compute_programs(
    pred_df: pd.DataFrame,
    available_genes: list[str],
    programs: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    return add_count_scale_programs(pred_df, available_genes, programs)


def add_count_columns(pred_df: pd.DataFrame, selected_genes: list[str]) -> pd.DataFrame:
    return add_reconstructed_count_columns(
        pred_df,
        selected_genes,
        source_rate_prefix="gene_",
        keep_explicit_rate=True,
    )


def promote_count_log1p_gene_aliases(
    pred_df: pd.DataFrame,
    selected_genes: list[str],
) -> pd.DataFrame:
    """Expose reconstructed count-log1p values through the website gene_* interface."""
    for gene in selected_genes:
        count_log1p_column = f"{COUNT_LOG1P_PREFIX}{gene}"
        if count_log1p_column not in pred_df.columns:
            raise KeyError(f"Missing canonical count-scale column: {count_log1p_column}")
        pred_df[f"gene_{gene}"] = pred_df[count_log1p_column].to_numpy(np.float32)
    return pred_df


def plot_spatial_maps(
    *,
    thumbnail: Image.Image,
    tile_df: pd.DataFrame,
    slide_width: int,
    slide_height: int,
    value_columns: list[str],
    out_path: Path,
    title: str,
) -> None:
    thumb = np.asarray(thumbnail.convert("RGB"))
    thumb_h, thumb_w = thumb.shape[:2]
    tx = (tile_df["x"].to_numpy(float) + tile_df["tile_size"].to_numpy(float) / 2.0) / slide_width * thumb_w
    ty = (tile_df["y"].to_numpy(float) + tile_df["tile_size"].to_numpy(float) / 2.0) / slide_height * thumb_h
    panels = ["thumbnail"] + value_columns[:5]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        ax.imshow(thumb)
        ax.set_xticks([])
        ax.set_yticks([])
        if panel == "thumbnail":
            ax.set_title("H&E thumbnail")
            continue
        values = tile_df[panel].to_numpy(float)
        if np.isfinite(values).any():
            lo, hi = np.nanpercentile(values, [2, 98])
        else:
            lo, hi = 0.0, 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(values)) if values.size else 0.0
            hi = float(np.nanmax(values) + 1.0e-6) if values.size else 1.0
        sc = ax.scatter(tx, ty, c=values, s=14, cmap="magma", vmin=lo, vmax=hi, linewidths=0, alpha=0.86)
        ax.set_title(panel.replace("program_", "").replace("gene_", "").replace("count_", "count "))
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_state_map(
    *,
    thumbnail: Image.Image,
    tile_df: pd.DataFrame,
    slide_width: int,
    slide_height: int,
    out_path: Path,
    title: str,
) -> str | None:
    program_cols = [col for col in tile_df.columns if col.startswith("program_")]
    if not program_cols:
        return None
    values = tile_df[program_cols].to_numpy(float)
    state_ids = np.nanargmax(values, axis=1)
    state_names = [col.replace("program_", "") for col in program_cols]
    thumb = np.asarray(thumbnail.convert("RGB"))
    thumb_h, thumb_w = thumb.shape[:2]
    tx = (tile_df["x"].to_numpy(float) + tile_df["tile_size"].to_numpy(float) / 2.0) / slide_width * thumb_w
    ty = (tile_df["y"].to_numpy(float) + tile_df["tile_size"].to_numpy(float) / 2.0) / slide_height * thumb_h
    cmap = plt.get_cmap("tab10", len(program_cols))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(thumb)
    ax.set_xticks([])
    ax.set_yticks([])
    scatter = ax.scatter(tx, ty, c=state_ids, cmap=cmap, s=16, linewidths=0, alpha=0.88, vmin=-0.5, vmax=len(program_cols) - 0.5)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02, ticks=range(len(program_cols)))
    cbar.ax.set_yticklabels(state_names)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def write_report(out_dir: Path, summary: dict[str, Any]) -> Path:
    figure_rows = []
    for slide in summary.get("slide_summaries", []):
        for key in ("spatial_maps", "state_map"):
            value = slide.get(key)
            if value:
                rel = Path(value).relative_to(out_dir)
                figure_rows.append(
                    f"<figure><img src='{html.escape(str(rel).replace(chr(92), '/'))}' alt='{html.escape(key)}'>"
                    f"<figcaption>{html.escape(slide.get('input_name', 'input'))} - {html.escape(key)}</figcaption></figure>"
                )
    figures = "\n".join(figure_rows) or "<p>No spatial figures were written.</p>"
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HistoOmniST user WSI prediction report</title>
  <style>
    body {{ margin: 0; padding: 34px; background: #f6f1e8; color: #171a1f; font-family: Arial, sans-serif; }}
    main {{ max-width: 1080px; margin: 0 auto; background: #fffdf8; border: 1px solid #d8d0c2; border-radius: 8px; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p, li {{ line-height: 1.7; color: #4f585b; }}
    code {{ background: #eef4ef; padding: 2px 5px; border-radius: 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e4ded3; padding: 9px; text-align: left; vertical-align: top; }}
    th {{ color: #0a6657; }}
    img {{ max-width: 100%; border: 1px solid #d8d0c2; border-radius: 8px; }}
    figure {{ margin: 18px 0; }}
    figcaption {{ color: #656d70; font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <main>
    <h1>HistoOmniST user WSI prediction report</h1>
    <p>This report was generated by a command-line HistoOmniST inference entrypoint for user supplied H&E WSI/image files.</p>
    <table>
      <tbody>
        <tr><th>Status</th><td>{html.escape(str(summary.get("status")))}</td></tr>
        <tr><th>Generated at</th><td>{html.escape(str(summary.get("generated_at")))}</td></tr>
        <tr><th>Device</th><td>{html.escape(str(summary.get("device")))}</td></tr>
        <tr><th>Inputs</th><td>{html.escape(str(summary.get("n_inputs")))}</td></tr>
        <tr><th>Predicted slides</th><td>{html.escape(str(summary.get("n_slides_predicted")))}</td></tr>
        <tr><th>Tiles</th><td>{html.escape(str(summary.get("n_tiles")))}</td></tr>
      </tbody>
    </table>
    <p><strong>Prediction scale:</strong> columns named <code>gene_*</code> are website aliases of the canonical <code>count_log1p_*</code> values, reconstructed as log1p(max(expm1(rate_log1p), 0) multiplied by mean-one predicted SF). Columns named <code>rate_log1p_*</code> retain the frozen rate-branch outputs, and <code>count_*</code> retain the reconstructed values before log1p transformation. Program columns are means of count-log1p genes and should be treated as model-derived count-scale virtual signals, not measured ST.</p>
    <h2>Figures</h2>
    {figures}
  </main>
</body>
</html>
"""
    path = out_dir / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def make_zip(out_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_path.resolve():
                archive.write(path, path.relative_to(out_dir))


def main() -> None:
    args = parse_args()
    started = time.time()
    target_config = load_target_config(resolve(args.target_config))
    programs = dict(target_config.get("programs") or PROGRAMS)
    if args.selected_genes is None and target_config.get("selected_genes"):
        selected_gene_request = list(target_config["selected_genes"])
    else:
        selected_gene_request = parse_gene_list(args.selected_genes)
    input_values: list[str] = []
    if args.inputs:
        input_values.extend(args.inputs)
    if args.input_list is not None:
        input_values.extend(read_input_list(resolve(args.input_list)))
    if not input_values:
        raise ValueError("Provide at least one --input or an --input-list file.")
    inputs = [resolve(Path(value)) for value in input_values]
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "predictions"
    figure_dir = out_dir / "figures"
    thumbnail_dir = out_dir / "thumbnails"
    for path in (prediction_dir, figure_dir, thumbnail_dir):
        path.mkdir(parents=True, exist_ok=True)

    import torch
    from histoomnist.features.patch_features import hipt256_features, load_hipt256_model

    if int(args.tile_size) != 224:
        raise ValueError("HIPT ViT-256 feature extraction requires --tile-size 224.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    expression_checkpoint = resolve(args.expression_checkpoint)
    sf_checkpoint = resolve(args.sf_checkpoint)
    rate_model, rate_genes, rate_mean, rate_std = load_rate_model(expression_checkpoint, device=device)
    sf_model, sf_mean, sf_std = load_sf_model(resolve(args.sf_config), sf_checkpoint, device=device)
    gene_to_idx = {gene: idx for idx, gene in enumerate(rate_genes)}
    selected_genes = [gene for gene in selected_gene_request if gene in gene_to_idx]
    missing_genes = [gene for gene in selected_gene_request if gene not in gene_to_idx]
    if not selected_genes:
        raise ValueError("None of the requested genes are available in the expression checkpoint.")
    selected_indices = [gene_to_idx[gene] for gene in selected_genes]

    hipt_model = load_hipt256_model(
        hipt_source_dir=resolve(args.hipt_source_dir),
        weights_path=resolve(args.hipt_weights),
        device=device,
    )

    all_predictions: list[pd.DataFrame] = []
    slide_summaries: list[dict[str, Any]] = []
    skipped_inputs: list[dict[str, Any]] = []

    for input_index, input_path in enumerate(inputs, start=1):
        slide_started = time.time()
        if not input_path.exists():
            skipped_inputs.append({"input": str(input_path), "status": "missing"})
            continue
        output_stem = safe_stem(input_path, input_index)
        slide, reader = open_slide(input_path)
        try:
            width, height = slide.dimensions
            thumbnail = slide.get_thumbnail((int(args.thumbnail_max_dim), int(args.thumbnail_max_dim))).convert("RGB")
            thumbnail_path = thumbnail_dir / f"{output_stem}_thumbnail.png"
            thumbnail.save(thumbnail_path)
            mask = rgb_tissue_mask(np.asarray(thumbnail))
            effective_stride, stride_diagnostics = choose_adaptive_stride(
                width=int(width),
                height=int(height),
                tile_size=int(args.tile_size),
                requested_stride=int(args.stride),
                mask=mask,
                tissue_threshold=float(args.tissue_threshold),
                max_tiles=int(args.max_tiles_per_slide),
                target_tiles=int(args.target_tiles_per_slide),
                min_stride=int(args.min_stride),
            )
            tiles = select_tile_grid(
                width=int(width),
                height=int(height),
                tile_size=int(args.tile_size),
                stride=int(effective_stride),
                mask=mask,
                tissue_threshold=float(args.tissue_threshold),
                max_tiles=int(args.max_tiles_per_slide),
            )
            if tiles.empty:
                skipped_inputs.append({"input": str(input_path), "status": "no_tissue_tiles"})
                continue
            tiles.insert(0, "input_index", input_index)
            tiles.insert(1, "input_name", input_path.name)
            tiles.insert(2, "input_path", str(input_path))
            tiles.insert(3, "slide_id", output_stem)

            feature_chunks: list[np.ndarray] = []
            for start in range(0, len(tiles), int(args.batch_size)):
                images = read_tile_batch(slide, tiles, int(args.tile_size), start, min(start + int(args.batch_size), len(tiles)))
                feature_chunks.append(hipt256_features(images, hipt_model, device=device))
            raw_features = np.concatenate(feature_chunks, axis=0).astype(np.float32)
            rate_features = features_for_checkpoint(raw_features, tiles, expected_dim=int(rate_mean.shape[0]))
            if int(sf_mean.shape[0]) == int(rate_features.shape[1]):
                sf_features = rate_features
            else:
                sf_features = features_for_checkpoint(raw_features, tiles, expected_dim=int(sf_mean.shape[0]))

            pred_log1p_rate = predict_rate_selected(
                features=rate_features,
                model=rate_model,
                mean=rate_mean,
                std=rate_std,
                selected_indices=selected_indices,
                device=device,
                batch_size=int(args.batch_size),
            )
            pred_log_sf, pred_sf = predict_sf(
                features=sf_features,
                model=sf_model,
                mean=sf_mean,
                std=sf_std,
                device=device,
                batch_size=int(args.batch_size),
            )

            pred_df = tiles.copy()
            pred_df["pred_log_sf"] = pred_log_sf
            pred_df["pred_sf"] = pred_sf
            for idx, gene in enumerate(selected_genes):
                pred_df[f"gene_{gene}"] = pred_log1p_rate[:, idx]
            pred_df = add_count_columns(pred_df, selected_genes)
            pred_df, used_programs = compute_programs(pred_df, selected_genes, programs)
            pred_df = promote_count_log1p_gene_aliases(pred_df, selected_genes)

            pred_path = prediction_dir / f"{output_stem}_tile_predictions.csv"
            pred_df.to_csv(pred_path, index=False)
            all_predictions.append(pred_df)

            spatial_path = figure_dir / f"{output_stem}_spatial_gene_program_maps.png"
            state_path_str = ""
            if not args.skip_figures:
                value_columns = [
                    col
                    for col in [
                        f"{COUNT_LOG1P_PREFIX}EPCAM",
                        f"{COUNT_LOG1P_PREFIX}COL1A1",
                        f"{COUNT_LOG1P_PREFIX}CD3D",
                        f"{COUNT_LOG1P_PREFIX}MKI67",
                        "pred_log_sf",
                        "program_stromal",
                        "program_t_cell",
                        "program_proliferation",
                        "program_stromal_ecm",
                        "program_t_cell_immune",
                        "program_epithelial_luminal_tumour",
                    ]
                    if col in pred_df.columns
                ]
                plot_spatial_maps(
                    thumbnail=thumbnail,
                    tile_df=pred_df,
                    slide_width=int(width),
                    slide_height=int(height),
                    value_columns=value_columns,
                    out_path=spatial_path,
                    title=f"{output_stem} HistoOmniST virtual-ST maps",
                )
                state_path = figure_dir / f"{output_stem}_state_map.png"
                state_path_str = plot_state_map(
                    thumbnail=thumbnail,
                    tile_df=pred_df,
                    slide_width=int(width),
                    slide_height=int(height),
                    out_path=state_path,
                    title=f"{output_stem} predicted program state map",
                ) or ""
            slide_summaries.append(
                {
                    "input_name": input_path.name,
                    "input_path": str(input_path),
                    "slide_id": output_stem,
                    "reader": reader,
                    "wsi_width": int(width),
                    "wsi_height": int(height),
                    "n_tiles": int(len(pred_df)),
                    "requested_stride": int(args.stride),
                    "effective_stride": int(effective_stride),
                    "target_tiles_per_slide": int(args.target_tiles_per_slide),
                    "max_tiles_per_slide": int(args.max_tiles_per_slide),
                    "stride_diagnostics": stride_diagnostics,
                    "mean_tissue_fraction": float(pred_df["tissue_fraction"].mean()),
                    "pred_mean_sf": float(pred_df["pred_sf"].mean()),
                    "prediction_csv": str(pred_path),
                    "thumbnail": str(thumbnail_path),
                    "spatial_maps": "" if args.skip_figures else str(spatial_path),
                    "state_map": state_path_str or "",
                    "programs": used_programs,
                    "seconds": round(time.time() - slide_started, 4),
                }
            )
            print(
                json.dumps(
                    {
                        "input": str(input_path),
                        "n_tiles": int(len(pred_df)),
                        "prediction_csv": str(pred_path),
                        "seconds": round(time.time() - slide_started, 3),
                    }
                ),
                flush=True,
            )
        finally:
            slide.close()

    if not all_predictions:
        raise RuntimeError("No input produced predictions.")

    combined = pd.concat(all_predictions, ignore_index=True)
    combined_path = out_dir / "tile_predictions.csv"
    combined.to_csv(combined_path, index=False)
    slide_features = []
    feature_cols = [col for col in combined.columns if col.startswith(("gene_", "count_", "program_")) or col in {"pred_sf", "pred_log_sf"}]
    for slide_id, sub in combined.groupby("slide_id", sort=True):
        row: dict[str, Any] = {"slide_id": slide_id, "input_name": sub["input_name"].iloc[0], "n_tiles": int(len(sub))}
        for col in feature_cols:
            values = sub[col].to_numpy(float)
            row[f"mean_{col}"] = float(np.nanmean(values))
            row[f"p90_{col}"] = float(np.nanpercentile(values, 90))
            row[f"std_{col}"] = float(np.nanstd(values))
        slide_features.append(row)
    slide_features_path = out_dir / "slide_features.csv"
    pd.DataFrame(slide_features).to_csv(slide_features_path, index=False)

    summary = {
        "status": "ok",
        "generated_at": time.strftime("%F %T %Z"),
        "device": device,
        "n_inputs": int(len(inputs)),
        "n_slides_predicted": int(combined["slide_id"].nunique()),
        "n_tiles": int(len(combined)),
        "tile_size": int(args.tile_size),
        "stride": int(args.stride),
        "target_tiles_per_slide": int(args.target_tiles_per_slide),
        "min_stride": int(args.min_stride),
        "tissue_threshold": float(args.tissue_threshold),
        "max_tiles_per_slide": int(args.max_tiles_per_slide),
        "skip_figures": bool(args.skip_figures),
        "expression_checkpoint": str(expression_checkpoint),
        "sf_checkpoint": str(sf_checkpoint),
        "sf_config": str(resolve(args.sf_config)),
        "hipt_source_dir": str(resolve(args.hipt_source_dir)),
        "hipt_weights": str(resolve(args.hipt_weights)),
        "target_config": str(resolve(args.target_config)),
        "selected_genes_present": selected_genes,
        "selected_genes_missing": missing_genes,
        "program_definitions": programs,
        "prediction_scale_note": (
            "gene_* columns are website aliases of canonical count_log1p_* values; "
            "rate_log1p_* columns retain frozen rate-branch outputs; count_* columns are "
            "max(expm1(rate_log1p), 0) multiplied by mean-one predicted SF; count_log1p_* and "
            "program_* columns are the canonical count-scale analysis values."
        ),
        "combined_tile_predictions": str(combined_path),
        "slide_features": str(slide_features_path),
        "slide_summaries": slide_summaries,
        "skipped_inputs": skipped_inputs,
        "seconds": round(time.time() - started, 4),
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_safe), encoding="utf-8")

    if args.write_report:
        report_path = write_report(out_dir, summary)
        summary["report_html"] = str(report_path)
        summary_path.write_text(json.dumps(summary, indent=2, default=json_safe), encoding="utf-8")
    if args.write_zip:
        zip_path = out_dir / "prediction_outputs.zip"
        make_zip(out_dir, zip_path)
        summary["prediction_zip"] = str(zip_path)
        summary_path.write_text(json.dumps(summary, indent=2, default=json_safe), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=json_safe), flush=True)


if __name__ == "__main__":
    main()
