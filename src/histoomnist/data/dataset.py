from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset

from histoomnist.data.spot_table import load_spot_table


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


class FeatureStandardizer:
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, x: np.ndarray) -> "FeatureStandardizer":
        self.mean = np.nanmean(x, axis=0).astype(np.float32)
        self.std = np.nanstd(x, axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)


class SizeFactorDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        base_dir: str | Path,
        splits: list[str],
        min_total_counts: float = 1.0,
        standardizer: FeatureStandardizer | None = None,
        fit_standardizer: bool = False,
    ):
        rows = manifest[manifest["split"].isin(splits)].copy()
        if rows.empty:
            raise ValueError(f"No manifest rows for splits={splits}")
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        sample_ids: list[str] = []
        base = Path(base_dir)
        for row in rows.itertuples(index=False):
            table = load_spot_table(
                sample_id=str(row.sample_id),
                features_path=base / str(row.features_path),
                counts_path=base / str(row.counts_path),
                coords_path=base / str(_optional_path(row, "coords_path"))
                if _optional_path(row, "coords_path") is not None
                else None,
                size_factor_path=base / str(_optional_path(row, "size_factor_path"))
                if _optional_path(row, "size_factor_path") is not None
                else None,
                min_total_counts=min_total_counts,
            )
            mask = table.valid_mask
            xs.append(table.features[mask])
            ys.append(table.log_size_factor[mask])
            sample_ids.extend([table.sample_id] * int(mask.sum()))
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.float32)
        self.standardizer = standardizer or FeatureStandardizer()
        if fit_standardizer:
            self.standardizer.fit(x)
        self.x = self.standardizer.transform(x)
        self.y = y[:, None]
        self.sample_ids = np.asarray(sample_ids)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.x[idx]),
            "log_sf": torch.from_numpy(self.y[idx]),
        }


class ExpressionRateDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        base_dir: str | Path,
        splits: list[str],
        min_total_counts: float = 1.0,
        standardizer: FeatureStandardizer | None = None,
        fit_standardizer: bool = False,
        gene_indices: np.ndarray | None = None,
    ):
        rows = manifest[manifest["split"].isin(splits)].copy()
        if rows.empty:
            raise ValueError(f"No manifest rows for splits={splits}")
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        log_sfs: list[np.ndarray] = []
        base = Path(base_dir)
        for row in rows.itertuples(index=False):
            table = load_spot_table(
                sample_id=str(row.sample_id),
                features_path=base / str(row.features_path),
                counts_path=base / str(row.counts_path),
                coords_path=base / str(_optional_path(row, "coords_path"))
                if _optional_path(row, "coords_path") is not None
                else None,
                size_factor_path=base / str(_optional_path(row, "size_factor_path"))
                if _optional_path(row, "size_factor_path") is not None
                else None,
                min_total_counts=min_total_counts,
            )
            mask = table.valid_mask
            counts = table.counts[mask]
            if gene_indices is not None:
                counts = counts[:, gene_indices]
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts = np.asarray(counts, dtype=np.float32)
            rate = counts / np.clip(table.size_factor[mask, None], 1e-6, None)
            xs.append(table.features[mask])
            ys.append(np.log1p(rate).astype(np.float32))
            log_sfs.append(table.log_size_factor[mask].astype(np.float32)[:, None])
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.float32)
        self.true_log_sf = np.concatenate(log_sfs, axis=0).astype(np.float32)
        self.standardizer = standardizer or FeatureStandardizer()
        if fit_standardizer:
            self.standardizer.fit(x)
        self.x = self.standardizer.transform(x)
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.x[idx]),
            "log1p_rate": torch.from_numpy(self.y[idx]),
            "true_log_sf": torch.from_numpy(self.true_log_sf[idx]),
        }
