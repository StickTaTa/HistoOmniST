from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from histoomnist.data.size_factor import compute_size_factor, log_size_factor, row_sums


@dataclass
class SpotTable:
    sample_id: str
    features: np.ndarray
    counts: object
    coords: np.ndarray | None
    size_factor: np.ndarray
    log_size_factor: np.ndarray
    valid_mask: np.ndarray


def load_array(path: str | Path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if p.suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.numpy()
        if isinstance(obj, dict) and "features" in obj:
            value = obj["features"]
            return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        raise ValueError(f"Unsupported torch feature object in {p}")
    if p.suffix == ".npz":
        try:
            return sparse.load_npz(p)
        except Exception:
            obj = np.load(p, allow_pickle=False)
            if len(obj.files) == 1:
                return obj[obj.files[0]]
            return {k: obj[k] for k in obj.files}
    raise ValueError(f"Unsupported array file type: {p}")


def load_spot_table(
    sample_id: str,
    features_path: str | Path,
    counts_path: str | Path,
    coords_path: str | Path | None = None,
    size_factor_path: str | Path | None = None,
    min_total_counts: float = 1.0,
) -> SpotTable:
    features = np.asarray(load_array(features_path), dtype=np.float32)
    counts = load_array(counts_path)
    coords = None if coords_path in (None, "") else np.asarray(load_array(coords_path), dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D for {sample_id}, got {features.shape}")
    if counts.shape[0] != features.shape[0]:
        raise ValueError(
            f"spot count mismatch for {sample_id}: features={features.shape[0]}, counts={counts.shape[0]}"
        )
    if coords is not None and coords.shape[0] != features.shape[0]:
        raise ValueError(
            f"coord count mismatch for {sample_id}: coords={coords.shape[0]}, features={features.shape[0]}"
        )
    if size_factor_path in (None, ""):
        sf, valid = compute_size_factor(counts, min_total_counts=min_total_counts)
    else:
        sf = np.asarray(load_array(size_factor_path), dtype=np.float32).reshape(-1)
        if sf.shape[0] != features.shape[0]:
            raise ValueError(
                f"size factor count mismatch for {sample_id}: sf={sf.shape[0]}, features={features.shape[0]}"
            )
        totals = row_sums(counts)
        valid = np.isfinite(sf) & (sf > 0) & np.isfinite(totals) & (totals >= float(min_total_counts))
    return SpotTable(
        sample_id=sample_id,
        features=features,
        counts=counts,
        coords=coords,
        size_factor=sf,
        log_size_factor=log_size_factor(sf),
        valid_mask=valid,
    )
