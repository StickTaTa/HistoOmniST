from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from histoomnist.utils.project_paths import project_root, resolve_project_path


DEFAULT_RUNS_DIR = "results/hest1k_human_visium_expression/generalization_runs"
DEFAULT_READINESS_TABLE = "results/hest1k_human_visium_expression/generalization_readiness/task_summary.csv"
DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/generalization_task_inspection"


PRIMARY_METRICS = {
    "sf": ("metric_log_sf_pearson", "log_sf_pearson"),
    "expression": ("metric_mean_gene_pearson", "mean_gene_pearson"),
    "combined": ("metric_count_pred_sf_mean_gene_pearson", "count_pred_sf_mean_gene_pearson"),
}


LOW_METRIC_THRESHOLDS = {
    "sf": 0.10,
    "expression": 0.05,
    "combined": 0.10,
}


def rel_project_path(path: str | Path | None) -> str:
    if path in (None, ""):
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_formal_generalization_rows(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(runs_dir.glob("formal_*/summary.csv")):
        summary = read_csv_if_exists(summary_path)
        if summary.empty:
            continue
        summary = summary.copy()
        summary["run_name"] = summary_path.parent.name
        summary["source_path"] = rel_project_path(summary_path)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def readiness_metadata(readiness_table: Path) -> pd.DataFrame:
    readiness = read_csv_if_exists(readiness_table)
    if readiness.empty:
        return pd.DataFrame()
    keep_cols = [
        "split_type",
        "heldout",
        "task_slug",
        "n_slides",
        "n_train_slides",
        "n_val_slides",
        "n_test_slides",
        "train_organs",
        "test_organs",
        "train_cohorts",
        "test_cohorts",
        "ready_for_split_specific_training",
        "passes_min_test_slides",
    ]
    return readiness[[col for col in keep_cols if col in readiness.columns]].drop_duplicates(
        ["split_type", "task_slug"]
    )


def numeric_value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return float("nan")
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def build_task_metrics(formal: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    if formal.empty:
        return pd.DataFrame()
    if not readiness.empty:
        formal = formal.merge(readiness, on=["split_type", "task_slug"], how="left", suffixes=("", "_ready"))
        if "heldout_ready" in formal.columns:
            formal["heldout"] = formal["heldout"].fillna(formal["heldout_ready"])
            formal = formal.drop(columns=["heldout_ready"])
    rows: list[dict[str, Any]] = []
    for _, row in formal.iterrows():
        stage = str(row.get("stage", ""))
        metric_col, metric_name = PRIMARY_METRICS.get(stage, ("", ""))
        primary = numeric_value(row, metric_col) if metric_col else float("nan")
        out: dict[str, Any] = {
            "split_type": str(row.get("split_type", "")),
            "heldout": str(row.get("heldout", "")),
            "task_slug": str(row.get("task_slug", "")),
            "stage": stage,
            "status": str(row.get("status", "")),
            "run_name": str(row.get("run_name", "")),
            "primary_metric_name": metric_name,
            "primary_metric_value": primary,
            "low_metric_threshold": LOW_METRIC_THRESHOLDS.get(stage, float("nan")),
            "is_low_metric": bool(
                np.isfinite(primary)
                and stage in LOW_METRIC_THRESHOLDS
                and primary < float(LOW_METRIC_THRESHOLDS[stage])
            ),
            "source_path": str(row.get("source_path", "")),
            "task_dir": str(row.get("task_dir", "")),
        }
        for col in [
            "metric_log_sf_pearson",
            "metric_sf_pearson",
            "metric_mean_gene_pearson",
            "metric_median_gene_pearson",
            "metric_valid_genes",
            "metric_rate_mean_gene_pearson",
            "metric_count_no_sf_mean_gene_pearson",
            "metric_count_pred_sf_mean_gene_pearson",
            "metric_count_oracle_sf_mean_gene_pearson",
            "metric_count_pred_sf_valid_genes",
            "metric_sf_log_sf_pearson",
        ]:
            if col in row.index:
                out[col] = numeric_value(row, col)
        for col in [
            "n_slides",
            "n_train_slides",
            "n_val_slides",
            "n_test_slides",
            "train_organs",
            "test_organs",
            "train_cohorts",
            "test_cohorts",
            "ready_for_split_specific_training",
            "passes_min_test_slides",
        ]:
            if col in row.index:
                out[col] = row[col]
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["split_type", "stage", "primary_metric_value", "heldout"]).reset_index(
        drop=True
    )


def summarize_task_metrics(task_metrics: pd.DataFrame) -> pd.DataFrame:
    if task_metrics.empty:
        return pd.DataFrame()
    rows = []
    ok = task_metrics[task_metrics["status"].astype(str).eq("ok")].copy()
    for (split_type, stage), group in ok.groupby(["split_type", "stage"], sort=True):
        values = pd.to_numeric(group["primary_metric_value"], errors="coerce").dropna()
        if values.empty:
            continue
        sorted_group = group.sort_values("primary_metric_value")
        worst = sorted_group.iloc[0]
        best = sorted_group.iloc[-1]
        low_count = int(pd.to_numeric(group["is_low_metric"], errors="coerce").fillna(0).astype(bool).sum())
        rows.append(
            {
                "split_type": split_type,
                "stage": stage,
                "primary_metric_name": str(group["primary_metric_name"].iloc[0]),
                "n_tasks": int(len(group)),
                "n_ok_tasks": int(len(group)),
                "mean_primary_metric": float(values.mean()),
                "median_primary_metric": float(values.median()),
                "min_primary_metric": float(values.min()),
                "q25_primary_metric": float(values.quantile(0.25)),
                "q75_primary_metric": float(values.quantile(0.75)),
                "max_primary_metric": float(values.max()),
                "low_metric_threshold": float(group["low_metric_threshold"].iloc[0]),
                "n_low_metric_tasks": low_count,
                "worst_heldout": str(worst["heldout"]),
                "worst_task_slug": str(worst["task_slug"]),
                "worst_primary_metric": float(worst["primary_metric_value"]),
                "best_heldout": str(best["heldout"]),
                "best_task_slug": str(best["task_slug"]),
                "best_primary_metric": float(best["primary_metric_value"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split_type", "stage"]).reset_index(drop=True)


def evaluate_generalization_inspection(*, runs_dir: Path, readiness_table: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    formal = load_formal_generalization_rows(runs_dir)
    readiness = readiness_metadata(readiness_table)
    task_metrics = build_task_metrics(formal, readiness)
    summary = summarize_task_metrics(task_metrics)

    task_metrics_path = out_dir / "generalization_task_metrics.csv"
    summary_path = out_dir / "generalization_task_summary.csv"
    task_metrics.to_csv(task_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)

    run_summary = {
        "runs_dir": rel_project_path(runs_dir),
        "readiness_table": rel_project_path(readiness_table),
        "n_task_stage_rows": int(len(task_metrics)),
        "n_summary_rows": int(len(summary)),
        "split_types": sorted(task_metrics["split_type"].dropna().astype(str).unique().tolist())
        if not task_metrics.empty
        else [],
        "stages": sorted(task_metrics["stage"].dropna().astype(str).unique().tolist()) if not task_metrics.empty else [],
        "outputs": {
            "generalization_task_metrics": rel_project_path(task_metrics_path),
            "generalization_task_summary": rel_project_path(summary_path),
        },
    }
    write_json(out_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect formal leave-organ/cohort generalization task metrics.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--readiness-table", default=DEFAULT_READINESS_TABLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = resolve_project_path(args.runs_dir)
    readiness_table = resolve_project_path(args.readiness_table)
    out_dir = resolve_project_path(args.out_dir)
    if None in (runs_dir, readiness_table, out_dir):
        raise ValueError("Required paths did not resolve.")
    evaluate_generalization_inspection(runs_dir=runs_dir, readiness_table=readiness_table, out_dir=out_dir)


if __name__ == "__main__":
    main()
