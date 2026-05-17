from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign_group_splits(
    df: pd.DataFrame,
    group_col: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 2026,
) -> pd.DataFrame:
    if group_col not in df.columns:
        raise KeyError(f"group_col not found: {group_col}")
    if not 0 < train_size < 1 or not 0 < val_size < 1 or train_size + val_size >= 1:
        raise ValueError("train_size and val_size must be positive and sum to < 1")
    out = df.copy()
    groups = out[group_col].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, rest_idx = next(splitter.split(out, groups=groups))
    out["split"] = "unset"
    out.loc[out.index[train_idx], "split"] = "train"
    rest = out.iloc[rest_idx]
    rest_groups = rest[group_col].astype(str).to_numpy()
    val_fraction_of_rest = val_size / (1.0 - train_size)
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=val_fraction_of_rest, random_state=seed + 1)
    val_rel_idx, test_rel_idx = next(splitter2.split(rest, groups=rest_groups))
    out.loc[rest.index[val_rel_idx], "split"] = "val"
    out.loc[rest.index[test_rel_idx], "split"] = "test"
    return out
