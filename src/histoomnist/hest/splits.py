from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def assign_leave_slide_out(
    df: pd.DataFrame,
    *,
    seed: int = 2026,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    """Assign slide-level train/val/test splits, stratifying coarsely by organ."""

    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.")
    rng = np.random.default_rng(seed)
    rows = []
    for organ, group in df.groupby("organ", dropna=False):
        ids = group["sample_id"].astype(str).to_numpy() if "sample_id" in group.columns else group["id"].astype(str).to_numpy()
        ids = ids.copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))
        if n_train + n_val >= n and n >= 3:
            n_train = max(1, n - 2)
            n_val = 1
        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train : n_train + n_val])
        for sid in ids:
            split = "train" if sid in train_ids else "val" if sid in val_ids else "test"
            rows.append({"sample_id": sid, "split": split, "split_type": "leave_slide_out", "heldout": ""})
    return pd.DataFrame(rows)


def make_leave_organ_out(df: pd.DataFrame, organs: list[str]) -> pd.DataFrame:
    id_col = "sample_id" if "sample_id" in df.columns else "id"
    rows = []
    for organ in organs:
        if organ not in set(df["organ"].dropna().astype(str)):
            continue
        for row in df.itertuples(index=False):
            sid = str(getattr(row, id_col))
            row_organ = str(getattr(row, "organ"))
            rows.append(
                {
                    "sample_id": sid,
                    "split": "test" if row_organ == organ else "train",
                    "split_type": "leave_organ_out",
                    "heldout": organ,
                }
            )
    return pd.DataFrame(rows)


def make_leave_cohort_out(df: pd.DataFrame, *, cohort_column: str, min_test_slides: int = 5) -> pd.DataFrame:
    id_col = "sample_id" if "sample_id" in df.columns else "id"
    rows = []
    counts = df[cohort_column].value_counts(dropna=False)
    cohorts = [cohort for cohort, count in counts.items() if count >= min_test_slides]
    for cohort in cohorts:
        for row in df.itertuples(index=False):
            sid = str(getattr(row, id_col))
            value = getattr(row, cohort_column)
            rows.append(
                {
                    "sample_id": sid,
                    "split": "test" if value == cohort else "train",
                    "split_type": "leave_cohort_out",
                    "heldout": str(cohort),
                }
            )
    return pd.DataFrame(rows)


def apply_split_to_manifest(manifest: pd.DataFrame, split_table: pd.DataFrame, split_type: str = "leave_slide_out") -> pd.DataFrame:
    split_rows = split_table[split_table["split_type"].eq(split_type)][["sample_id", "split"]].drop_duplicates()
    out = manifest.drop(columns=["split"], errors="ignore").merge(split_rows, on="sample_id", how="left")
    out["split"] = out["split"].fillna("unassigned")
    return out


def write_split_tables(split_dir: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    out = Path(split_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
