from __future__ import annotations

import importlib.util
import json
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

from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.external.histogene_patch_h5 import (
    TargetKind,
    load_histogene_patch_slide,
    target_matrix_from_counts,
)
from histoomnist.external.histogene_sourcefaithful import _center_crop_or_resize_hwc, _position_bins
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path


THITOGENE_UPSTREAM_REL = Path("third_party") / "benchmarks" / "THItoGene"


@dataclass(frozen=True)
class THItoGeneSourceInfo:
    upstream_root: Path
    vis_model_path: Path
    odconv_path: Path
    capsnet_path: Path
    transformer_path: Path
    gat_path: Path
    graph_path: Path
    imported_components: tuple[str, ...]


def _load_module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def thitogene_source_info(upstream_root: str | Path | None = None) -> THItoGeneSourceInfo:
    root = resolve_project_path(upstream_root or THITOGENE_UPSTREAM_REL)
    if root is None:
        raise ValueError("THItoGene upstream root resolved to None")
    paths = {
        "vis_model_path": root / "vis_model.py",
        "odconv_path": root / "ODConv.py",
        "capsnet_path": root / "efficient_capsnet.py",
        "transformer_path": root / "transformer.py",
        "gat_path": root / "GATLayer.py",
        "graph_path": root / "graph_construction.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing official THItoGene source files: " + ", ".join(missing))
    return THItoGeneSourceInfo(
        upstream_root=root,
        imported_components=(
            "official ODConv.ODConv2d",
            "official efficient_capsnet.EfficientCapsNet",
            "official transformer.ViT",
            "official GATLayer.MultiHeadGAT",
            "official graph_construction.calcADJ",
        ),
        **paths,
    )


def load_official_thitogene_components(upstream_root: str | Path | None = None) -> dict[str, Any]:
    info = thitogene_source_info(upstream_root)
    odconv = _load_module_from_file("_thitogene_official_odconv", info.odconv_path)
    caps = _load_module_from_file("_thitogene_official_capsnet", info.capsnet_path)
    transformer = _load_module_from_file("_thitogene_official_transformer", info.transformer_path)
    gat = _load_module_from_file("_thitogene_official_gat", info.gat_path)
    graph = _load_module_from_file("_thitogene_official_graph", info.graph_path)
    _patch_capsnet_routing_device(caps)
    return {
        "ODConv2d": odconv.ODConv2d,
        "EfficientCapsNet": caps.EfficientCapsNet,
        "ViT": transformer.ViT,
        "MultiHeadGAT": gat.MultiHeadGAT,
        "calcADJ": graph.calcADJ,
    }


def _patch_capsnet_routing_device(caps_module: ModuleType) -> None:
    """Keep upstream capsule math while fixing a CPU scalar in CUDA runs."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        u = torch.einsum("...ji,kjiz->...kjz", input, self.W)
        c = torch.einsum("...ij,...kj->...i", u, u)[..., None]
        scale = torch.sqrt(
            torch.tensor([self.dim_capsules], device=input.device, dtype=input.dtype)
        )
        c = c / scale
        c = torch.softmax(c, axis=1)
        c = c + self.b
        s = torch.sum(torch.mul(u, c), dim=-2)
        return caps_module.squash(s)

    caps_module.RoutingLayer.forward = forward


class OfficialTHItoGeneCore(nn.Module):
    """THItoGene core matching upstream ``vis_model.THItoGene``.

    The upstream file imports torchvision-backed helper classes at module import time.
    To avoid that unrelated local DLL failure, this class imports the official core
    modules directly and reconstructs the upstream THItoGene block sequence.
    """

    def __init__(
        self,
        *,
        patch_size: int = 112,
        n_layers: int = 4,
        n_genes: int = 1000,
        dim: int = 1024,
        learning_rate: float = 1.0e-5,
        dropout: float = 0.2,
        n_pos: int = 64,
        heads: tuple[int, int] | list[int] = (16, 8),
        caps: int = 20,
        route_dim: int = 64,
        upstream_root: str | Path | None = None,
    ):
        super().__init__()
        if len(heads) != 2:
            raise ValueError("Official THItoGene expects heads=[vit_heads, gat_heads].")
        self.learning_rate = float(learning_rate)
        self.patch_size = int(patch_size)
        self.n_layers = int(n_layers)
        self.n_genes = int(n_genes)
        self.dim = int(dim)
        self.dropout = float(dropout)
        self.n_pos = int(n_pos)
        self.heads = (int(heads[0]), int(heads[1]))
        self.caps = int(caps)
        self.route_dim = int(route_dim)
        components = load_official_thitogene_components(upstream_root)
        caps_out = (self.caps + 2) * self.route_dim

        self.relu = nn.ReLU()
        self.odconv2d = components["ODConv2d"](in_planes=3, out_planes=16, kernel_size=4, stride=4)
        self.caps_layer = components["EfficientCapsNet"](rout_capsules=self.caps, route_dim=self.route_dim)
        self.x_embed = nn.Embedding(self.n_pos, self.route_dim)
        self.y_embed = nn.Embedding(self.n_pos, self.route_dim)
        self.vit = components["ViT"](
            dim=caps_out,
            depth=self.n_layers,
            heads=self.heads[0],
            mlp_dim=2 * self.dim,
            dropout=self.dropout,
            emb_dropout=self.dropout,
        )
        self.gat = components["MultiHeadGAT"](
            in_features=caps_out,
            nhid=1024,
            out_features=512,
            heads=self.heads[1],
            dropout=self.dropout,
            alpha=0.01,
        )
        self.gene_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, self.n_genes),
        )

    def forward(self, patches: torch.Tensor, centers: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(f"patches must be shaped (batch, spots, channels, height, width), got {tuple(patches.shape)}")
        if patches.shape[0] != 1:
            raise ValueError("Official THItoGene forward assumes batch_size=1 full slide item.")
        if adj.ndim == 3:
            adj = adj[0]
        batch, n_spots, channels, height, width = patches.shape
        x = patches.reshape(batch * n_spots, channels, height, width)
        if height != self.patch_size or width != self.patch_size:
            x = F.interpolate(x, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)
        x = self.relu(self.odconv2d(x))
        x = self.caps_layer(x)
        x = x.reshape(-1, self.caps, self.route_dim)

        centers = torch.clamp(centers.long(), min=0, max=self.n_pos - 1)
        centers_x = self.x_embed(centers[:, :, 0]).permute(1, 0, 2)
        centers_y = self.y_embed(centers[:, :, 1]).permute(1, 0, 2)
        x = torch.concat((x, centers_x, centers_y), dim=1)
        x = x.reshape(1, x.shape[0], -1)
        x = self.vit(x)
        x = x.reshape(x.shape[1], -1)
        x = self.gat(x, adj.to(device=x.device, dtype=x.dtype))
        return self.gene_head(x)


def thitogene_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    expression_mask: torch.Tensor,
) -> torch.Tensor:
    if target.ndim == 3:
        target = target[0]
    if expression_mask.ndim == 2:
        expression_mask = expression_mask[0]
    valid = expression_mask.bool().unsqueeze(0).expand_as(target)
    if not torch.any(valid):
        raise ValueError("No valid expression values for THItoGene masked MSE.")
    return (pred - target).pow(2)[valid].mean()


class THItoGeneHESTSlideDataset(Dataset):
    """HEST full-slide dataset matching upstream THItoGene data surface."""

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
        k_neighbors: int = 4,
        max_spots_per_slide: int | None = None,
        max_slide_spots: int | None = None,
        upstream_root: str | Path | None = None,
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
            raise ValueError(f"No manifest rows for splits={splits}, slide_ids={slide_ids}.")

        base_dir = manifest_path.parent
        target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
        if target_genes is None or gene_indices is not None:
            raise ValueError("THItoGene source-faithful adapter requires data.gene_names_path target genes.")
        gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
        raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
        raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
        if raw_root is None:
            raise ValueError("Expression config paths.raw_root resolved to None")
        min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))

        components = load_official_thitogene_components(upstream_root)
        self.calc_adj = components["calcADJ"]
        self.target_kind = target_kind
        self.target_genes = list(target_genes)
        self.patch_size = int(patch_size)
        self.n_pos = int(n_pos)
        self.k_neighbors = int(k_neighbors)
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
        patches = np.zeros((n_spots, 3, self.patch_size, self.patch_size), dtype=np.float32)
        with h5py.File(slide.patch_h5_path, "r") as handle:
            img = handle["img"]
            for out_idx, patch_index in enumerate(slide.patch_indices[:n_spots]):
                patch = np.asarray(img[int(patch_index)])
                crop = _center_crop_or_resize_hwc(patch, self.patch_size)
                patches[out_idx] = np.transpose(crop, (2, 0, 1)).astype(np.float32, copy=False)
        counts = slide.counts[:n_spots].toarray().astype(np.float32, copy=False)
        target = target_matrix_from_counts(counts, slide.size_factor[:n_spots], self.target_kind)
        positions = _position_bins(slide.position_norm[:n_spots], n_pos=self.n_pos)
        coords = slide.spatial_coords[:n_spots] if slide.spatial_coords is not None else positions
        adj = self.calc_adj(coord=np.asarray(coords, dtype=np.float32), k=self.k_neighbors, pruneTag="NA")
        if isinstance(adj, torch.Tensor):
            adj_array = adj.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            adj_array = np.asarray(adj, dtype=np.float32)
        return {
            "patches": torch.from_numpy(patches),
            "positions": torch.from_numpy(positions),
            self.target_kind: torch.from_numpy(target),
            "expression_mask": torch.from_numpy(slide.measured_genes.astype(bool, copy=False)),
            "adj": torch.from_numpy(adj_array),
            "sample_id": slide.sample_id,
            "spot_ids": slide.spot_ids[:n_spots],
            "n_spots": int(slide.n_spots),
            "n_used_spots": int(n_spots),
            "truncated_for_smoke": bool(n_spots != slide.n_spots),
        }


def thitogene_source_metadata(upstream_root: str | Path | None = None) -> dict[str, Any]:
    info = thitogene_source_info(upstream_root)
    metadata = {
        "upstream_root": str(info.upstream_root),
        "vis_model_path": str(info.vis_model_path),
        "odconv_path": str(info.odconv_path),
        "capsnet_path": str(info.capsnet_path),
        "transformer_path": str(info.transformer_path),
        "gat_path": str(info.gat_path),
        "graph_path": str(info.graph_path),
        "imported_components": list(info.imported_components),
    }
    provenance = info.upstream_root / "source_provenance.json"
    if provenance.exists():
        metadata["provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    return metadata
