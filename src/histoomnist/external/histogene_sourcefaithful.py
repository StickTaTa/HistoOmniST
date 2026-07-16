from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from histoomnist.data.gene_selection import (
    gene_key_settings_from_config,
    selected_genes_from_config,
)
from histoomnist.external.histogene_patch_h5 import (
    TargetKind,
    load_histogene_patch_slide,
    target_matrix_from_counts,
)
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path


HISTOGENE_UPSTREAM_REL = Path("third_party") / "benchmarks" / "HisToGene"


@dataclass(frozen=True)
class HisToGeneSourceInfo:
    upstream_root: Path
    transformer_path: Path
    vis_model_path: Path
    imported_component: str
    upstream_commit: str | None = None


def _load_module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def histogene_source_info(upstream_root: str | Path | None = None) -> HisToGeneSourceInfo:
    root = resolve_project_path(upstream_root or HISTOGENE_UPSTREAM_REL)
    if root is None:
        raise ValueError("HisToGene upstream root resolved to None")
    transformer_path = root / "transformer.py"
    vis_model_path = root / "vis_model.py"
    if not transformer_path.exists():
        raise FileNotFoundError(f"Official HisToGene transformer.py not found: {transformer_path}")
    if not vis_model_path.exists():
        raise FileNotFoundError(f"Official HisToGene vis_model.py not found: {vis_model_path}")
    return HisToGeneSourceInfo(
        upstream_root=root,
        transformer_path=transformer_path,
        vis_model_path=vis_model_path,
        imported_component="official transformer.ViT",
    )


def load_official_vit(upstream_root: str | Path | None = None):
    info = histogene_source_info(upstream_root)
    module = _load_module_from_file("_histogene_official_transformer", info.transformer_path)
    if not hasattr(module, "ViT"):
        raise AttributeError(f"Official transformer.py lacks ViT: {info.transformer_path}")
    return module.ViT


class OfficialHisToGeneCore(nn.Module):
    """HisToGene core architecture, matching upstream ``vis_model.HisToGene``.

    The local benchmark environment cannot import upstream ``vis_model.py`` because that
    file imports torchvision at module import time. The HisToGene class itself only needs
    PyTorch and upstream ``transformer.ViT``, so this module keeps the same layers,
    defaults, fixed 16 attention heads, Adam-compatible learning rate, and forward path.
    """

    def __init__(
        self,
        *,
        patch_size: int = 112,
        n_layers: int = 4,
        n_genes: int = 1000,
        dim: int = 1024,
        learning_rate: float = 1.0e-4,
        dropout: float = 0.1,
        n_pos: int = 64,
        upstream_root: str | Path | None = None,
    ):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.patch_size = int(patch_size)
        self.n_layers = int(n_layers)
        self.n_genes = int(n_genes)
        self.dim = int(dim)
        self.dropout = float(dropout)
        self.n_pos = int(n_pos)
        patch_dim = 3 * self.patch_size * self.patch_size
        self.patch_embedding = nn.Linear(patch_dim, self.dim)
        self.x_embed = nn.Embedding(self.n_pos, self.dim)
        self.y_embed = nn.Embedding(self.n_pos, self.dim)
        vit_cls = load_official_vit(upstream_root)
        self.vit = vit_cls(
            dim=self.dim,
            depth=self.n_layers,
            heads=16,
            mlp_dim=2 * self.dim,
            dropout=self.dropout,
            emb_dropout=self.dropout,
        )
        self.gene_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.n_genes))

    def forward(self, patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(patches)
        centers = centers.long()
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        return self.gene_head(h)


def _center_crop_or_resize_hwc(patch: np.ndarray, patch_size: int) -> np.ndarray:
    arr = np.asarray(patch)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB patch, got {arr.shape}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if height >= patch_size and width >= patch_size:
        y0 = (height - patch_size) // 2
        x0 = (width - patch_size) // 2
        return np.asarray(arr[y0 : y0 + patch_size, x0 : x0 + patch_size, :], dtype=np.float32)
    tensor = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)


def _position_bins(position_norm: np.ndarray, *, n_pos: int) -> np.ndarray:
    values = np.asarray(position_norm, dtype=np.float32)
    bins = np.rint(values * float(int(n_pos) - 1)).astype(np.int64)
    return np.clip(bins, 0, int(n_pos) - 1)


class HisToGeneHESTSlideDataset(Dataset):
    """HEST full-slide dataset for source-faithful HisToGene training.

    Each item is one slide, matching upstream ``ViT_HER2ST.__getitem__``:
    ``patches`` has shape ``[n_spots, 3*112*112]``, ``positions`` is an integer
    embedding grid, and ``target`` has shape ``[n_spots, n_genes]``.
    """

    def __init__(
        self,
        expression_config: dict[str, Any],
        *,
        splits: list[str],
        slide_ids: list[str] | None = None,
        max_slides: int | None = None,
        smallest_slides: bool = False,
        target_kind: TargetKind = "log1p_rate",
        patch_size: int = 112,
        n_pos: int = 64,
        max_spots_per_slide: int | None = None,
        max_slide_spots: int | None = None,
    ):
        manifest_path = resolve_project_path(expression_config["data"]["manifest"])
        if manifest_path is None:
            raise ValueError("Expression config data.manifest resolved to None")
        manifest = read_manifest(manifest_path)
        rows = manifest[manifest["split"].isin([str(x) for x in splits])].copy()
        if slide_ids:
            wanted = {str(x) for x in slide_ids}
            rows = rows[rows["sample_id"].astype(str).isin(wanted)].copy()
        if max_slide_spots is not None:
            if "n_spots" not in rows.columns:
                raise ValueError("Manifest lacks n_spots; cannot apply max_slide_spots.")
            rows = rows[rows["n_spots"].astype(int) <= int(max_slide_spots)].copy()
        if smallest_slides:
            rows = rows.sort_values(["n_spots", "sample_id"]).copy()
        if max_slides is not None:
            rows = rows.head(int(max_slides)).copy()
        if rows.empty:
            raise ValueError(
                f"No manifest rows for splits={splits}, slide_ids={slide_ids}, "
                f"max_slide_spots={max_slide_spots}"
            )

        base_dir = manifest_path.parent
        target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
        if target_genes is None or gene_indices is not None:
            raise ValueError("HisToGene source-faithful adapter requires data.gene_names_path target genes.")
        gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
        raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
        raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
        if raw_root is None:
            raise ValueError("Expression config paths.raw_root resolved to None")
        min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))

        self.target_kind = target_kind
        self.target_genes = list(target_genes)
        self.patch_size = int(patch_size)
        self.n_pos = int(n_pos)
        self.max_spots_per_slide = None if max_spots_per_slide is None else int(max_spots_per_slide)
        self.max_slide_spots = None if max_slide_spots is None else int(max_slide_spots)
        self.slides = [
            load_histogene_patch_slide(
                row=row,
                base_dir=base_dir,
                raw_root=raw_root,
                target_genes=self.target_genes,
                gene_key=gene_key,
                raw_st_root=raw_st_root,
                min_total_counts=min_total_counts,
            )
            for row in rows.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.slides)

    def _used_spots(self, n_spots: int) -> int:
        if self.max_spots_per_slide is None:
            return int(n_spots)
        return min(int(n_spots), int(self.max_spots_per_slide))

    def slide_summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sample_id": slide.sample_id,
                    "split": slide.split,
                    "organ": slide.organ,
                    "cohort": slide.cohort,
                    "n_spots": slide.n_spots,
                    "n_used_spots": self._used_spots(slide.n_spots),
                    "truncated_for_smoke": bool(self._used_spots(slide.n_spots) != slide.n_spots),
                    "n_measured_target_genes": int(slide.measured_genes.sum()),
                    "patch_h5_path": str(slide.patch_h5_path),
                }
                for slide in self.slides
            ]
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        slide = self.slides[index]
        n_spots = self._used_spots(slide.n_spots)
        patch_dim = 3 * self.patch_size * self.patch_size
        patches = np.zeros((n_spots, patch_dim), dtype=np.float32)
        with h5py.File(slide.patch_h5_path, "r") as handle:
            img = handle["img"]
            for out_idx, patch_index in enumerate(slide.patch_indices[:n_spots]):
                patch = np.asarray(img[int(patch_index)])
                patches[out_idx] = _center_crop_or_resize_hwc(patch, self.patch_size).reshape(-1)
        counts = slide.counts[:n_spots].toarray().astype(np.float32, copy=False)
        target = target_matrix_from_counts(counts, slide.size_factor[:n_spots], self.target_kind)
        positions = _position_bins(slide.position_norm[:n_spots], n_pos=self.n_pos)
        return {
            "patches": torch.from_numpy(patches),
            "positions": torch.from_numpy(positions),
            self.target_kind: torch.from_numpy(target),
            "expression_mask": torch.from_numpy(slide.measured_genes.astype(bool, copy=False)),
            "sample_id": slide.sample_id,
            "spot_ids": slide.spot_ids[:n_spots],
            "n_spots": int(slide.n_spots),
            "n_used_spots": int(n_spots),
            "truncated_for_smoke": bool(n_spots != slide.n_spots),
        }


def histogene_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    expression_mask: torch.Tensor,
) -> torch.Tensor:
    valid = expression_mask.bool().view(expression_mask.shape[0], 1, expression_mask.shape[1])
    valid = valid.expand_as(pred)
    if not torch.any(valid):
        raise ValueError("No valid expression values for HisToGene masked MSE.")
    return (pred - target).pow(2)[valid].mean()
