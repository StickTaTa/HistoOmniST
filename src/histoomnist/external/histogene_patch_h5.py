from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset

from histoomnist.data.gene_selection import (
    gene_key_settings_from_config,
    load_gene_keys_for_slide,
    selected_genes_from_config,
)
from histoomnist.data.spot_table import load_spot_table
from histoomnist.hest.raw_assets import read_h5_string_vector
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path


TargetKind = Literal["log1p_rate", "rate", "count", "log1p_count"]


@dataclass(frozen=True)
class HistogenePatchSlide:
    sample_id: str
    split: str
    organ: str
    cohort: str
    disease_state: str
    patch_h5_path: Path
    spot_ids: list[str]
    patch_indices: np.ndarray
    spatial_coords: np.ndarray | None
    position_norm: np.ndarray
    counts: sparse.csr_matrix
    size_factor: np.ndarray
    measured_genes: np.ndarray

    @property
    def n_spots(self) -> int:
        return int(self.patch_indices.shape[0])


def _optional_path(row, name: str):
    if not hasattr(row, name):
        return None
    value = getattr(row, name)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if str(value).strip() == "":
        return None
    return value


def _read_spot_ids(base_dir: Path, row, n_spots: int) -> list[str]:
    spots_path = _optional_path(row, "spots_path")
    candidates: list[Path] = []
    if spots_path is not None:
        candidates.append(base_dir / str(spots_path))
    candidates.append((base_dir / str(row.counts_path)).parent / "spots.txt")
    for path in candidates:
        if path.exists():
            values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(values) == n_spots:
                return values
            raise ValueError(f"Spot id count mismatch for {row.sample_id}: {path} has {len(values)}, expected {n_spots}")
    raise FileNotFoundError(f"Could not find processed spot ids for {row.sample_id}")


def _read_patch_barcodes(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        if "barcode" not in handle:
            raise KeyError(f"Patch H5 lacks barcode dataset: {path}")
        return read_h5_string_vector(handle["barcode"])


def _normalise_positions(coords: np.ndarray | None, n_spots: int) -> np.ndarray:
    if coords is None:
        return np.zeros((n_spots, 2), dtype=np.float32)
    values = np.asarray(coords, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Coordinates must be shaped (n, >=2), got {values.shape}")
    values = values[:, :2]
    lo = np.nanmin(values, axis=0)
    hi = np.nanmax(values, axis=0)
    span = hi - lo
    span[span < 1.0e-6] = 1.0
    out = (values - lo) / span
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _select_counts_for_target_genes(
    *,
    counts,
    slide_genes: list[str | None],
    target_genes: list[str],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    counts_csr = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
    target_index = {gene: idx for idx, gene in enumerate(target_genes)}
    source_indices: list[int] = []
    target_indices: list[int] = []
    for source_idx, gene in enumerate(slide_genes):
        if gene is None:
            continue
        target_idx = target_index.get(gene)
        if target_idx is None:
            continue
        source_indices.append(source_idx)
        target_indices.append(target_idx)
    if not source_indices:
        raise ValueError("No target genes were found in slide genes.")
    source_array = np.asarray(source_indices, dtype=np.int64)
    target_array = np.asarray(target_indices, dtype=np.int64)
    selected_source = counts_csr[:, source_array].astype(np.float32).tocsr()
    mapper = sparse.csr_matrix(
        (
            np.ones(target_array.shape[0], dtype=np.float32),
            (np.arange(target_array.shape[0]), target_array),
        ),
        shape=(target_array.shape[0], len(target_genes)),
    )
    selected_counts = (selected_source @ mapper).tocsr()
    measured = np.zeros(len(target_genes), dtype=bool)
    measured[np.unique(target_array)] = True
    return selected_counts, measured


def load_histogene_patch_slide(
    *,
    row,
    base_dir: Path,
    raw_root: Path,
    target_genes: list[str],
    gene_key: str,
    raw_st_root: Path | None,
    min_total_counts: float,
) -> HistogenePatchSlide:
    sample_id = str(row.sample_id)
    patch_h5_path = raw_root / "patches" / f"{sample_id}.h5"
    if not patch_h5_path.exists():
        raise FileNotFoundError(f"Patch H5 not found for {sample_id}: {patch_h5_path}")
    table = load_spot_table(
        sample_id=sample_id,
        features_path=base_dir / str(row.features_path),
        counts_path=base_dir / str(row.counts_path),
        coords_path=base_dir / str(_optional_path(row, "coords_path"))
        if _optional_path(row, "coords_path") is not None
        else None,
        size_factor_path=base_dir / str(_optional_path(row, "size_factor_path"))
        if _optional_path(row, "size_factor_path") is not None
        else None,
        min_total_counts=min_total_counts,
    )
    spot_ids_all = _read_spot_ids(base_dir, row, table.features.shape[0])
    patch_index = {barcode: idx for idx, barcode in enumerate(_read_patch_barcodes(patch_h5_path))}
    missing = [barcode for barcode in spot_ids_all if barcode not in patch_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{sample_id} has {len(missing)} processed spots missing from patch H5 barcode: {preview}")
    valid = table.valid_mask.astype(bool)
    patch_indices_all = np.asarray([patch_index[barcode] for barcode in spot_ids_all], dtype=np.int64)
    slide_genes = load_gene_keys_for_slide(
        sample_id=sample_id,
        processed_gene_path=base_dir / str(row.genes_path),
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    selected_counts, measured = _select_counts_for_target_genes(
        counts=table.counts[valid],
        slide_genes=slide_genes,
        target_genes=target_genes,
    )
    coords = None if table.coords is None else np.asarray(table.coords[valid], dtype=np.float32)
    return HistogenePatchSlide(
        sample_id=sample_id,
        split=str(row.split),
        organ=str(getattr(row, "organ", "")),
        cohort=str(getattr(row, "cohort", "")),
        disease_state=str(getattr(row, "disease_state", "")),
        patch_h5_path=patch_h5_path,
        spot_ids=[str(x) for x, keep in zip(spot_ids_all, valid) if keep],
        patch_indices=patch_indices_all[valid],
        spatial_coords=coords,
        position_norm=_normalise_positions(coords, int(np.sum(valid))),
        counts=selected_counts,
        size_factor=table.size_factor[valid].astype(np.float32, copy=False),
        measured_genes=measured,
    )


def target_values_from_counts(counts: np.ndarray, size_factor: float, target_kind: TargetKind) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float32)
    if target_kind == "count":
        return values
    if target_kind == "log1p_count":
        return np.log1p(values).astype(np.float32, copy=False)
    rate = values / max(float(size_factor), 1.0e-6)
    if target_kind == "rate":
        return rate.astype(np.float32, copy=False)
    if target_kind == "log1p_rate":
        return np.log1p(rate).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported target_kind: {target_kind}")


def target_matrix_from_counts(counts: np.ndarray, size_factor: np.ndarray, target_kind: TargetKind) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float32)
    if target_kind == "count":
        return values
    if target_kind == "log1p_count":
        return np.log1p(values).astype(np.float32, copy=False)
    sf = np.asarray(size_factor, dtype=np.float32).reshape(-1, 1)
    rate = values / np.clip(sf, 1.0e-6, None)
    if target_kind == "rate":
        return rate.astype(np.float32, copy=False)
    if target_kind == "log1p_rate":
        return np.log1p(rate).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported target_kind: {target_kind}")


class HistogenePatchH5Dataset(Dataset):
    """HEST patch-H5 dataset with HisToGene-style items.

    Items expose the data surface needed by HisToGene-like supervised baselines:
    image patch, normalized position, expression target, and measured-gene mask.
    """

    def __init__(
        self,
        expression_config: dict,
        *,
        splits: list[str],
        max_slides: int | None = None,
        target_kind: TargetKind = "log1p_rate",
    ):
        manifest_path = resolve_project_path(expression_config["data"]["manifest"])
        if manifest_path is None:
            raise ValueError("Expression config data.manifest resolved to None")
        manifest = read_manifest(manifest_path)
        rows = manifest[manifest["split"].isin(splits)].copy()
        if max_slides is not None:
            rows = rows.head(int(max_slides)).copy()
        if rows.empty:
            raise ValueError(f"No manifest rows for splits={splits}")
        base_dir = manifest_path.parent
        target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
        if target_genes is None or gene_indices is not None:
            raise ValueError("Histogene patch adapter requires data.gene_names_path target genes.")
        gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
        raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
        raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
        if raw_root is None:
            raise ValueError("Expression config paths.raw_root resolved to None")
        min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
        self.target_kind = target_kind
        self.target_genes = list(target_genes)
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
        lengths = [slide.n_spots for slide in self.slides]
        self.cumlen = np.cumsum(lengths).astype(np.int64)

    def __len__(self) -> int:
        return int(self.cumlen[-1]) if len(self.cumlen) else 0

    def slide_summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sample_id": slide.sample_id,
                    "split": slide.split,
                    "organ": slide.organ,
                    "cohort": slide.cohort,
                    "n_spots": slide.n_spots,
                    "n_measured_target_genes": int(slide.measured_genes.sum()),
                    "patch_h5_path": str(slide.patch_h5_path),
                }
                for slide in self.slides
            ]
        )

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slide_idx = bisect.bisect_right(self.cumlen, index)
        prev = 0 if slide_idx == 0 else int(self.cumlen[slide_idx - 1])
        return slide_idx, int(index - prev)

    def __getitem__(self, index: int) -> dict[str, object]:
        slide_idx, local_idx = self._locate(index)
        slide = self.slides[slide_idx]
        patch_index = int(slide.patch_indices[local_idx])
        with h5py.File(slide.patch_h5_path, "r") as handle:
            patch = np.asarray(handle["img"][patch_index], dtype=np.float32)
        if patch.ndim != 3 or patch.shape[-1] != 3:
            raise ValueError(f"Patch image must be HWC RGB, got {patch.shape} for {slide.sample_id}")
        patch = np.transpose(patch / 255.0, (2, 0, 1)).astype(np.float32, copy=False)
        counts = slide.counts.getrow(local_idx).toarray().reshape(-1).astype(np.float32, copy=False)
        target = target_values_from_counts(counts, float(slide.size_factor[local_idx]), self.target_kind)
        item: dict[str, object] = {
            "patch": torch.from_numpy(patch),
            "position": torch.from_numpy(slide.position_norm[local_idx]),
            self.target_kind: torch.from_numpy(target),
            "expression_mask": torch.from_numpy(slide.measured_genes),
            "sample_id": slide.sample_id,
            "spot_id": slide.spot_ids[local_idx],
            "patch_index": patch_index,
            "local_index": local_idx,
        }
        if slide.spatial_coords is not None:
            item["spatial_coords"] = torch.from_numpy(slide.spatial_coords[local_idx])
        return item


class HistogenePatchH5ChunkDataset(Dataset):
    """Fixed-size slide chunks for HisToGene-style transformer training."""

    def __init__(
        self,
        expression_config: dict,
        *,
        splits: list[str],
        chunk_size: int = 64,
        max_slides: int | None = None,
        max_chunks_per_slide: int | None = None,
        target_kind: TargetKind = "log1p_rate",
    ):
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        self.spot_dataset = HistogenePatchH5Dataset(
            expression_config,
            splits=splits,
            max_slides=max_slides,
            target_kind=target_kind,
        )
        self.slides = self.spot_dataset.slides
        self.target_genes = self.spot_dataset.target_genes
        self.target_kind = target_kind
        self.chunk_size = int(chunk_size)
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
            raise ValueError("No chunks were created for HisToGene patch-H5 dataset.")
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def slide_summary_frame(self) -> pd.DataFrame:
        return self.spot_dataset.slide_summary_frame()

    def __getitem__(self, index: int) -> dict[str, object]:
        slide_idx, start, stop = self.chunks[index]
        slide = self.slides[slide_idx]
        length = int(stop - start)
        patches = np.zeros((self.chunk_size, 3, 224, 224), dtype=np.float32)
        positions = np.zeros((self.chunk_size, 2), dtype=np.float32)
        targets = np.zeros((self.chunk_size, len(self.target_genes)), dtype=np.float32)
        spot_mask = np.zeros(self.chunk_size, dtype=bool)
        local_indices = np.arange(start, stop, dtype=np.int64)
        padded_local_indices = np.full(self.chunk_size, -1, dtype=np.int64)
        padded_local_indices[:length] = local_indices
        with h5py.File(slide.patch_h5_path, "r") as handle:
            img = handle["img"]
            for out_idx, local_idx in enumerate(local_indices):
                patch_index = int(slide.patch_indices[int(local_idx)])
                patch = np.asarray(img[patch_index], dtype=np.float32)
                if patch.ndim != 3 or patch.shape[-1] != 3:
                    raise ValueError(f"Patch image must be HWC RGB, got {patch.shape} for {slide.sample_id}")
                patches[out_idx] = np.transpose(patch / 255.0, (2, 0, 1))
        counts = slide.counts[start:stop].toarray().astype(np.float32, copy=False)
        targets[:length] = target_matrix_from_counts(
            counts,
            slide.size_factor[start:stop],
            self.target_kind,
        )
        positions[:length] = slide.position_norm[start:stop]
        spot_mask[:length] = True
        return {
            "patches": torch.from_numpy(patches),
            "positions": torch.from_numpy(positions),
            self.target_kind: torch.from_numpy(targets),
            "spot_mask": torch.from_numpy(spot_mask),
            "expression_mask": torch.from_numpy(slide.measured_genes),
            "sample_id": slide.sample_id,
            "start": int(start),
            "stop": int(stop),
            "local_indices": torch.from_numpy(padded_local_indices),
        }
