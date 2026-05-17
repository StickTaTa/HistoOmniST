from __future__ import annotations

import pandas as pd


def assert_disjoint_groups(manifest: pd.DataFrame, group_cols: list[str]) -> None:
    for group_col in group_cols:
        if group_col not in manifest.columns:
            continue
        seen: dict[str, set[str]] = {}
        for split, group in manifest.groupby("split"):
            for value in group[group_col].dropna().astype(str).unique():
                seen.setdefault(value, set()).add(str(split))
        leaked = {value: splits for value, splits in seen.items() if len(splits) > 1}
        if leaked:
            examples = list(leaked.items())[:10]
            raise ValueError(f"Group leakage in {group_col}: {examples}")


def assert_no_test_calibration(calibration_manifest: pd.DataFrame, test_manifest: pd.DataFrame) -> None:
    if "sample_id" not in calibration_manifest.columns or "sample_id" not in test_manifest.columns:
        raise KeyError("Both manifests need sample_id for calibration leakage checks.")
    overlap = set(calibration_manifest["sample_id"].astype(str)) & set(test_manifest["sample_id"].astype(str))
    if overlap:
        raise ValueError(f"Calibration samples overlap final test samples: {sorted(overlap)[:10]}")
