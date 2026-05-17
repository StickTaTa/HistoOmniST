from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from histoomnist.data.dataset import SizeFactorDataset
from histoomnist.eval.metrics import sf_metrics
from histoomnist.utils.io import read_manifest


def _collect_dataset(manifest: pd.DataFrame, base_dir: Path, splits: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ds = SizeFactorDataset(manifest, base_dir=base_dir, splits=splits, min_total_counts=1.0)
    return ds.x, ds.y.reshape(-1)


def run_constant_baseline(y_test: np.ndarray) -> dict[str, float]:
    pred = np.zeros_like(y_test, dtype=np.float32)
    metrics = sf_metrics(pred, y_test)
    return {"baseline": "constant_sf", **metrics}


def run_hipt_ridge_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float = 10.0,
) -> dict[str, float]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    model = Ridge(alpha=alpha)
    model.fit(x_train_s, y_train)
    pred = model.predict(x_test_s).astype(np.float32)
    metrics = sf_metrics(pred, y_test)
    return {"baseline": "hipt_ridge", "alpha": alpha, **metrics}


def run_available_baselines(
    *,
    manifest_path: str | Path,
    train_splits: list[str],
    test_splits: list[str],
    output_csv: str | Path,
) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    base_dir = manifest_path.parent
    x_train, y_train = _collect_dataset(manifest, base_dir, train_splits)
    x_test, y_test = _collect_dataset(manifest, base_dir, test_splits)
    rows = [
        run_constant_baseline(y_test),
        run_hipt_ridge_baseline(x_train, y_train, x_test, y_test),
    ]
    out = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
