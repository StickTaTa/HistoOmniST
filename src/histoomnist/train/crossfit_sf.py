from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class CrossFitFold:
    fold: int
    train_samples: list[str]
    oof_samples: list[str]


def make_group_folds(manifest: pd.DataFrame, group_col: str, n_splits: int) -> list[CrossFitFold]:
    if group_col not in manifest.columns:
        raise KeyError(f"group_col not found: {group_col}")
    groups = manifest[group_col].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    folds: list[CrossFitFold] = []
    for fold, (train_idx, oof_idx) in enumerate(splitter.split(np.zeros(len(manifest)), groups=groups)):
        train_samples = sorted(manifest.iloc[train_idx]["sample_id"].astype(str).unique())
        oof_samples = sorted(manifest.iloc[oof_idx]["sample_id"].astype(str).unique())
        folds.append(CrossFitFold(fold=fold, train_samples=train_samples, oof_samples=oof_samples))
    return folds


def explain_crossfit() -> str:
    return (
        "Train SF models on K-1 training folds, predict out-of-fold log_sf for the held-out "
        "training fold, then train the expression model on these noisy out-of-fold SF "
        "predictions. This avoids training the expression model with perfect true SF while "
        "deploying with predicted SF."
    )
