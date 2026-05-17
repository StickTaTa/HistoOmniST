from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_h5_string(value.item())
        if value.size == 1:
            return decode_h5_string(value.reshape(-1)[0])
    return str(value)


def read_h5_string_vector(dataset: h5py.Dataset) -> list[str]:
    values = dataset[:]
    return [decode_h5_string(value) for value in values.reshape(-1)]


@dataclass(frozen=True)
class RawSlidePaths:
    slide_id: str
    metadata: Path
    st: Path
    patches: Path
    thumbnail: Path
    wsi: Path | None


def find_wsi_path(raw_root: Path, slide_id: str) -> Path | None:
    wsis_dir = raw_root / "wsis"
    if not wsis_dir.exists():
        return None
    for suffix in (".tif", ".tiff", ".svs"):
        path = wsis_dir / f"{slide_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(wsis_dir.glob(f"{slide_id}.*"))
    return matches[0] if matches else None


def raw_slide_paths(raw_root: Path, slide_id: str) -> RawSlidePaths:
    return RawSlidePaths(
        slide_id=slide_id,
        metadata=raw_root / "metadata" / f"{slide_id}.json",
        st=raw_root / "st" / f"{slide_id}.h5ad",
        patches=raw_root / "patches" / f"{slide_id}.h5",
        thumbnail=raw_root / "thumbnails" / f"{slide_id}_downscaled_fullres.jpeg",
        wsi=find_wsi_path(raw_root, slide_id),
    )


def read_h5ad_barcodes(h5ad_path: Path) -> list[str]:
    with h5py.File(h5ad_path, "r") as handle:
        return read_h5_string_vector(handle["obs"]["_index"])


def read_h5ad_genes(h5ad_path: Path) -> list[str]:
    with h5py.File(h5ad_path, "r") as handle:
        return read_h5_string_vector(handle["var"]["_index"])


def read_h5ad_spatial(h5ad_path: Path) -> np.ndarray:
    with h5py.File(h5ad_path, "r") as handle:
        return np.asarray(handle["obsm"]["spatial"][:], dtype=np.float32)


def read_h5ad_total_counts(h5ad_path: Path) -> np.ndarray:
    with h5py.File(h5ad_path, "r") as handle:
        obs = handle["obs"]
        if "total_counts" in obs:
            return np.asarray(obs["total_counts"][:], dtype=np.float64)
        matrix = read_h5ad_csr_matrix_from_handle(handle)
    return np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)


def read_patch_barcodes(patch_path: Path) -> list[str]:
    with h5py.File(patch_path, "r") as handle:
        return read_h5_string_vector(handle["barcode"])


def read_patch_coords(patch_path: Path) -> np.ndarray:
    with h5py.File(patch_path, "r") as handle:
        return np.asarray(handle["coords"][:], dtype=np.float32)


def read_h5ad_csr_matrix_from_handle(handle: h5py.File) -> sparse.csr_matrix:
    x = handle["X"]
    if isinstance(x, h5py.Dataset):
        return sparse.csr_matrix(np.asarray(x[:]))
    data = np.asarray(x["data"][:])
    indices = np.asarray(x["indices"][:])
    indptr = np.asarray(x["indptr"][:])
    n_obs = len(indptr) - 1
    n_vars = len(handle["var"]["_index"])
    return sparse.csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))


def read_h5ad_gene_vector(h5ad_path: Path, gene: str) -> tuple[np.ndarray, bool]:
    with h5py.File(h5ad_path, "r") as handle:
        genes = read_h5_string_vector(handle["var"]["_index"])
        if gene not in genes:
            return np.zeros(len(handle["obs"]["_index"]), dtype=np.float32), False
        gene_idx = genes.index(gene)
        matrix = read_h5ad_csr_matrix_from_handle(handle)
    values = np.asarray(matrix[:, gene_idx].toarray()).ravel()
    return values.astype(np.float32), True


@dataclass(frozen=True)
class AlignmentResult:
    st_barcodes: list[str]
    patch_barcodes: list[str]
    kept_barcodes: list[str]
    st_indices: np.ndarray
    patch_indices: np.ndarray
    patch_not_in_st: list[str]
    st_not_in_patch: list[str]
    barcode_order_monotonic: bool


def align_patch_to_h5ad_barcodes(st_barcodes: list[str], patch_barcodes: list[str]) -> AlignmentResult:
    st_index = {barcode: idx for idx, barcode in enumerate(st_barcodes)}
    patch_index = {barcode: idx for idx, barcode in enumerate(patch_barcodes)}
    kept = [barcode for barcode in patch_barcodes if barcode in st_index]
    st_indices = np.asarray([st_index[barcode] for barcode in kept], dtype=np.int64)
    patch_indices = np.asarray([patch_index[barcode] for barcode in kept], dtype=np.int64)
    patch_not_in_st = sorted(set(patch_barcodes) - set(st_barcodes))
    st_not_in_patch = sorted(set(st_barcodes) - set(patch_barcodes))
    monotonic = bool(np.all(np.diff(st_indices) > 0)) if len(st_indices) > 1 else True
    return AlignmentResult(
        st_barcodes=st_barcodes,
        patch_barcodes=patch_barcodes,
        kept_barcodes=kept,
        st_indices=st_indices,
        patch_indices=patch_indices,
        patch_not_in_st=patch_not_in_st,
        st_not_in_patch=st_not_in_patch,
        barcode_order_monotonic=monotonic,
    )


def mean_one_log_size_factor(total_counts: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    counts = np.asarray(total_counts, dtype=np.float64)
    valid = np.isfinite(counts) & (counts > 0)
    if not np.any(valid):
        raise ValueError("Cannot compute size factor: no positive finite total counts.")
    sf = counts / float(np.mean(counts[valid]))
    return np.log(sf + eps).astype(np.float32)
