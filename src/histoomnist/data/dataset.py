from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset

from histoomnist.data.gene_selection import load_gene_keys_for_slide
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


def _read_gene_names(base: Path, row) -> list[str]:
    genes_path = _optional_path(row, "genes_path")
    if genes_path is None:
        raise ValueError(f"Manifest row for {row.sample_id} does not include genes_path.")
    path = (base / str(genes_path)).resolve(strict=False)
    return path.read_text(encoding="utf-8").splitlines()


def _read_gene_keys(base: Path, row, gene_key: str, raw_st_root: str | Path | None) -> list[str | None]:
    genes_path = _optional_path(row, "genes_path")
    if genes_path is None:
        raise ValueError(f"Manifest row for {row.sample_id} does not include genes_path.")
    return load_gene_keys_for_slide(
        sample_id=str(row.sample_id),
        processed_gene_path=(base / str(genes_path)).resolve(strict=False),
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )


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
        gene_names: list[str] | None = None,
        gene_key: str = "var_names",
        raw_st_root: str | Path | None = None,
        lazy_expression_threshold: int = 2048,
    ):
        rows = manifest[manifest["split"].isin(splits)].copy()
        if rows.empty:
            raise ValueError(f"No manifest rows for splits={splits}")
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        log_sfs: list[np.ndarray] = []
        base = Path(base_dir)
        self.genes = list(gene_names) if gene_names is not None else None
        self.output_dim = len(gene_names) if gene_names is not None else None
        self.lazy_expression = gene_names is not None and len(gene_names) > int(lazy_expression_threshold)
        self._slide_counts: list[sparse.csr_matrix] = []
        self._slide_measured: list[np.ndarray] = []
        self._row_slide: list[int] = []
        self._row_local: list[int] = []
        self._slide_size_factor: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        sample_ids: list[str] = []
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
            if gene_names is not None:
                slide_genes = _read_gene_keys(base, row, gene_key=gene_key, raw_st_root=raw_st_root)
                gene_to_target = {gene: idx for idx, gene in enumerate(gene_names)}
                source_indices: list[int] = []
                target_indices: list[int] = []
                for source_idx, gene in enumerate(slide_genes):
                    if gene is None:
                        continue
                    target_idx = gene_to_target.get(gene)
                    if target_idx is None:
                        continue
                    source_indices.append(source_idx)
                    target_indices.append(target_idx)
                if not source_indices:
                    continue
                source_indices_array = np.asarray(source_indices, dtype=np.int64)
                target_indices_array = np.asarray(target_indices, dtype=np.int64)
                measured = np.zeros(len(gene_names), dtype=bool)
                measured[np.unique(target_indices_array)] = True
                counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
                selected_source_counts = counts[:, source_indices_array].astype(np.float32).tocsr()
                mapper = sparse.csr_matrix(
                    (
                        np.ones(target_indices_array.shape[0], dtype=np.float32),
                        (np.arange(target_indices_array.shape[0]), target_indices_array),
                    ),
                    shape=(target_indices_array.shape[0], len(gene_names)),
                )
                selected_counts = (selected_source_counts @ mapper).tocsr()
                if self.lazy_expression:
                    slide_idx = len(self._slide_counts)
                    n_rows = selected_counts.shape[0]
                    self._slide_counts.append(selected_counts)
                    self._slide_measured.append(measured)
                    self._slide_size_factor.append(table.size_factor[mask].astype(np.float32))
                    self._row_slide.extend([slide_idx] * n_rows)
                    self._row_local.extend(range(n_rows))
                    xs.append(table.features[mask])
                    log_sfs.append(table.log_size_factor[mask].astype(np.float32)[:, None])
                    sample_ids.extend([table.sample_id] * int(mask.sum()))
                    continue
                else:
                    dense_counts = selected_counts.toarray().astype(np.float32)
                    counts = dense_counts
                    masks.append(np.broadcast_to(measured, dense_counts.shape).copy())
            elif gene_indices is not None:
                counts = counts[:, gene_indices]
                masks.append(np.ones((counts.shape[0], len(gene_indices)), dtype=bool))
            else:
                masks.append(np.ones((counts.shape[0], counts.shape[1]), dtype=bool))
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts = np.asarray(counts, dtype=np.float32)
            rate = counts / np.clip(table.size_factor[mask, None], 1e-6, None)
            xs.append(table.features[mask])
            if not self.lazy_expression:
                ys.append(np.log1p(rate).astype(np.float32))
            log_sfs.append(table.log_size_factor[mask].astype(np.float32)[:, None])
            sample_ids.extend([table.sample_id] * int(mask.sum()))
        if not xs:
            raise ValueError(f"No usable expression rows for splits={splits}")
        x = np.concatenate(xs, axis=0).astype(np.float32)
        self.raw_x = x
        self.true_log_sf = np.concatenate(log_sfs, axis=0).astype(np.float32)
        self.expression_mask = None if self.lazy_expression else np.concatenate(masks, axis=0).astype(bool)
        self.sample_ids = np.asarray(sample_ids)
        self._row_slide_array = np.asarray(self._row_slide, dtype=np.int32) if self.lazy_expression else None
        self._row_local_array = np.asarray(self._row_local, dtype=np.int32) if self.lazy_expression else None
        self.standardizer = standardizer or FeatureStandardizer()
        if fit_standardizer:
            self.standardizer.fit(x)
        self.x = self.standardizer.transform(x)
        if self.lazy_expression:
            self.y = None
            self.output_dim = len(gene_names)
        else:
            y = np.concatenate(ys, axis=0).astype(np.float32)
            self.y = y
            self.output_dim = y.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.lazy_expression:
            if self._row_slide_array is None or self._row_local_array is None:
                raise RuntimeError("Lazy expression index arrays were not initialized.")
            slide_idx = int(self._row_slide_array[idx])
            local_idx = int(self._row_local_array[idx])
            counts = self._slide_counts[slide_idx].getrow(local_idx).toarray().reshape(-1).astype(np.float32)
            sf = float(max(self._slide_size_factor[slide_idx][local_idx], 1e-6))
            y = np.log1p(counts / sf).astype(np.float32)
            expression_mask = self._slide_measured[slide_idx]
        else:
            y = self.y[idx]
            expression_mask = self.expression_mask[idx]
        return {
            "features": torch.from_numpy(self.x[idx]),
            "raw_features": torch.from_numpy(self.raw_x[idx]),
            "log1p_rate": torch.from_numpy(y),
            "true_log_sf": torch.from_numpy(self.true_log_sf[idx]),
            "expression_mask": torch.from_numpy(expression_mask),
        }
