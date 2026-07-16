from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


HIST2ST_UPSTREAM_REL = Path("third_party") / "benchmarks" / "Hist2ST"
GraphCoordSource = Literal["position_grid", "spatial"]


@dataclass(frozen=True)
class Hist2STSourceInfo:
    upstream_root: Path
    hist2st_path: Path
    transformer_path: Path
    gcn_path: Path
    nb_module_path: Path
    graph_path: Path
    imported_components: tuple[str, ...]


def _install_optional_import_stubs() -> None:
    """Patch unused optional imports so the official architecture can be imported.

    The upstream files import some notebook / validation dependencies at module import
    time. The benchmark training loop does not use those code paths, so missing
    optional modules are stubbed only to let the official model classes load.
    """

    if "easydl" not in sys.modules and importlib.util.find_spec("easydl") is None:
        sys.modules["easydl"] = types.ModuleType("easydl")
    if "scanpy" not in sys.modules and importlib.util.find_spec("scanpy") is None:
        sys.modules["scanpy"] = types.ModuleType("scanpy")
    if "anndata" not in sys.modules and importlib.util.find_spec("anndata") is None:
        anndata_stub = types.ModuleType("anndata")

        class AnnData:  # pragma: no cover - import shim only
            def __init__(self, *args, **kwargs):
                raise ImportError("anndata is required only for unused Hist2ST validation utilities.")

        anndata_stub.AnnData = AnnData
        sys.modules["anndata"] = anndata_stub
    if "pytorch_lightning" not in sys.modules and importlib.util.find_spec("pytorch_lightning") is None:
        lightning_stub = types.ModuleType("pytorch_lightning")
        lightning_stub.LightningModule = torch.nn.Module
        sys.modules["pytorch_lightning"] = lightning_stub


def _load_module_from_file(module_name: str, path: Path, *, upstream_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def hist2st_source_info(upstream_root: str | Path | None = None) -> Hist2STSourceInfo:
    root = resolve_project_path(upstream_root or HIST2ST_UPSTREAM_REL)
    if root is None:
        raise ValueError("Hist2ST upstream root resolved to None")
    paths = {
        "hist2st_path": root / "HIST2ST.py",
        "transformer_path": root / "transformer.py",
        "gcn_path": root / "gcn.py",
        "nb_module_path": root / "NB_module.py",
        "graph_path": root / "graph_construction.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing official Hist2ST source files: " + ", ".join(missing))
    return Hist2STSourceInfo(
        upstream_root=root,
        imported_components=(
            "official HIST2ST.Hist2ST",
            "official graph_construction.calcADJ",
            "official NB_module.NB_loss",
            "official NB_module.ZINB_loss",
        ),
        **paths,
    )


def load_official_hist2st_components(upstream_root: str | Path | None = None) -> dict[str, Any]:
    info = hist2st_source_info(upstream_root)
    _install_optional_import_stubs()
    hist2st = _load_module_from_file("_hist2st_official_model", info.hist2st_path, upstream_root=info.upstream_root)
    graph = _load_module_from_file("_hist2st_official_graph", info.graph_path, upstream_root=info.upstream_root)
    nb_module = _load_module_from_file("_hist2st_official_nb", info.nb_module_path, upstream_root=info.upstream_root)
    return {
        "Hist2ST": hist2st.Hist2ST,
        "calcADJ": graph.calcADJ,
        "NB_loss": nb_module.NB_loss,
        "ZINB_loss": nb_module.ZINB_loss,
    }


def hist2st_source_metadata(upstream_root: str | Path | None = None) -> dict[str, Any]:
    info = hist2st_source_info(upstream_root)
    metadata = {
        "upstream_root": str(info.upstream_root),
        "hist2st_path": str(info.hist2st_path),
        "transformer_path": str(info.transformer_path),
        "gcn_path": str(info.gcn_path),
        "nb_module_path": str(info.nb_module_path),
        "graph_path": str(info.graph_path),
        "imported_components": list(info.imported_components),
    }
    provenance = info.upstream_root / "source_provenance.json"
    if provenance.exists():
        metadata["provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    return metadata


def hist2st_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    expression_mask: torch.Tensor,
) -> torch.Tensor:
    if target.ndim == 3:
        target = target[0]
    if expression_mask.ndim == 2:
        expression_mask = expression_mask[0]
    valid = expression_mask.bool().view(1, -1).expand_as(pred)
    if not torch.any(valid):
        raise ValueError("No valid expression values for Hist2ST masked MSE.")
    return (pred - target).pow(2)[valid].mean()


def hist2st_zinb_or_nb_loss(
    *,
    extra: tuple[torch.Tensor, ...] | None,
    raw_counts: torch.Tensor,
    size_factors: torch.Tensor,
    expression_mask: torch.Tensor,
    nb: bool,
    components: dict[str, Any],
) -> torch.Tensor:
    if extra is None:
        raise ValueError("Hist2ST ZINB/NB loss requested but model did not return extra distribution tensors.")
    if raw_counts.ndim == 3:
        raw_counts = raw_counts[0]
    if size_factors.ndim == 2:
        size_factors = size_factors[0]
    if expression_mask.ndim == 2:
        expression_mask = expression_mask[0]
    mask = expression_mask.bool()
    if not torch.any(mask):
        raise ValueError("No measured genes available for Hist2ST ZINB/NB loss.")
    counts = raw_counts[:, mask]
    if nb:
        r, p = extra
        return components["NB_loss"](counts, r[:, mask], p[:, mask])
    mean, disp, pi = extra
    return components["ZINB_loss"](counts, mean[:, mask], disp[:, mask], pi[:, mask], size_factors)


class Hist2STHESTSlideDataset(Dataset):
    """HEST full-slide dataset matching the official Hist2ST data surface."""

    def __init__(
        self,
        expression_config: dict[str, Any],
        *,
        splits: list[str],
        slide_ids: list[str] | None = None,
        max_slides: int | None = None,
        smallest_slides: bool = False,
        target_kind: TargetKind = "log1p_rate",
        fig_size: int = 112,
        n_pos: int = 64,
        k_neighbors: int = 4,
        prune: str = "NA",
        graph_coord_source: GraphCoordSource = "position_grid",
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
            raise ValueError("Hist2ST source-faithful adapter requires data.gene_names_path target genes.")
        gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
        raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
        raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
        if raw_root is None:
            raise ValueError("Expression config paths.raw_root resolved to None")
        min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
        components = load_official_hist2st_components(upstream_root)

        self.calc_adj = components["calcADJ"]
        self.target_kind = target_kind
        self.target_genes = list(target_genes)
        self.fig_size = int(fig_size)
        self.n_pos = int(n_pos)
        self.k_neighbors = int(k_neighbors)
        self.prune = str(prune)
        self.graph_coord_source = graph_coord_source
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

    def _graph_coords(self, positions: np.ndarray, spatial_coords: np.ndarray | None) -> np.ndarray:
        if self.graph_coord_source == "position_grid":
            return positions.astype(np.float32, copy=False)
        if spatial_coords is None:
            return positions.astype(np.float32, copy=False)
        return np.asarray(spatial_coords, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        slide = self.slides[index]
        n_spots = self._used_spots(slide.n_spots)
        patches = np.zeros((n_spots, 3, self.fig_size, self.fig_size), dtype=np.float32)
        with h5py.File(slide.patch_h5_path, "r") as handle:
            img = handle["img"]
            for out_idx, patch_index in enumerate(slide.patch_indices[:n_spots]):
                patch = np.asarray(img[int(patch_index)])
                crop = _center_crop_or_resize_hwc(patch, self.fig_size)
                patches[out_idx] = np.transpose(crop, (2, 0, 1)).astype(np.float32, copy=False)
        counts = slide.counts[:n_spots].toarray().astype(np.float32, copy=False)
        target = target_matrix_from_counts(counts, slide.size_factor[:n_spots], self.target_kind)
        positions = _position_bins(slide.position_norm[:n_spots], n_pos=self.n_pos)
        spatial_coords = None if slide.spatial_coords is None else slide.spatial_coords[:n_spots]
        graph_coords = self._graph_coords(positions, spatial_coords)
        adj = self.calc_adj(coord=np.asarray(graph_coords, dtype=np.float32), k=self.k_neighbors, pruneTag=self.prune)
        if isinstance(adj, torch.Tensor):
            adj_array = adj.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            adj_array = np.asarray(adj, dtype=np.float32)
        return {
            "patches": torch.from_numpy(patches),
            "positions": torch.from_numpy(positions),
            self.target_kind: torch.from_numpy(target),
            "raw_counts": torch.from_numpy(counts),
            "size_factors": torch.from_numpy(slide.size_factor[:n_spots].astype(np.float32, copy=False)),
            "expression_mask": torch.from_numpy(slide.measured_genes.astype(bool, copy=False)),
            "adj": torch.from_numpy(adj_array),
            "sample_id": slide.sample_id,
            "spot_ids": slide.spot_ids[:n_spots],
            "n_spots": int(slide.n_spots),
            "n_used_spots": int(n_spots),
            "truncated_for_smoke": bool(n_spots != slide.n_spots),
        }


def adjacency_stats(adj: torch.Tensor | np.ndarray) -> dict[str, Any]:
    arr = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    degree = arr.sum(axis=1)
    return {
        "adj_shape": [int(x) for x in arr.shape],
        "adj_nonzero": int(np.count_nonzero(arr)),
        "adj_density": float(np.count_nonzero(arr) / max(arr.size, 1)),
        "adj_zero_degree_nodes": int(np.count_nonzero(degree == 0)),
        "adj_min_degree": float(np.min(degree)) if degree.size else 0.0,
        "adj_mean_degree": float(np.mean(degree)) if degree.size else 0.0,
        "adj_max_degree": float(np.max(degree)) if degree.size else 0.0,
    }
