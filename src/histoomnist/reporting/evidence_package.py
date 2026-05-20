from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from histoomnist.reporting.markdown import write_markdown_report
from histoomnist.utils.project_paths import project_root, resolve_project_path


DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/evidence_package"
EXPR_ROOT = "results/hest1k_human_visium_expression"


def rel_project_path(path: str | Path | None) -> str:
    if path in (None, ""):
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def bool_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    mapped = values.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.fillna(default).astype(bool)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def metric_from_method(df: pd.DataFrame, method: str, metric: str = "mean_gene_pearson") -> float | None:
    if df.empty or "method" not in df.columns or metric not in df.columns:
        return None
    rows = df[df["method"].astype(str).eq(method)]
    if rows.empty:
        return None
    return float(rows.iloc[0][metric])


def generalization_runs(root: Path) -> list[tuple[Path, dict[str, Any], pd.DataFrame]]:
    runs_dir = root / EXPR_ROOT / "generalization_runs"
    if not runs_dir.exists():
        return []
    runs = []
    for summary_path in sorted(runs_dir.glob("*/summary.csv")):
        summary = read_csv_if_exists(summary_path)
        if summary.empty:
            continue
        manifest_path = summary_path.parent / "run_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {"run_name": summary_path.parent.name}
        runs.append((summary_path, manifest, summary))
    return runs


def is_smoke_generalization_run(manifest: dict[str, Any], run_name: str, stage: str) -> bool:
    name = run_name.lower()
    if "smoke" in name or "pilot" in name:
        return True
    if stage == "sf":
        epochs = manifest.get("sf_epochs")
        return epochs == 1
    if stage == "expression":
        epochs = manifest.get("expression_epochs")
        return epochs == 1
    if stage == "combined":
        return manifest.get("sf_epochs") == 1 or manifest.get("expression_epochs") == 1
    return False


def generalization_metric_values(group: pd.DataFrame, stage: str) -> tuple[str, list[float]]:
    candidates = {
        "sf": [("metric_log_sf_pearson", "mean_log_sf_pearson"), ("metric_sf_pearson", "mean_sf_pearson")],
        "expression": [("metric_mean_gene_pearson", "mean_gene_pearson")],
        "combined": [
            ("metric_count_pred_sf_mean_gene_pearson", "mean_count_pred_sf_gene_pearson"),
            ("metric_rate_mean_gene_pearson", "mean_rate_gene_pearson"),
        ],
    }
    for column, label in candidates.get(stage, []):
        if column in group.columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if not values.empty:
                return label, [float(value) for value in values]
    return "", []


def generalization_metric(group: pd.DataFrame, stage: str) -> tuple[str, str]:
    label, values = generalization_metric_values(group, stage)
    if not values:
        return "", ""
    return label, f"{float(pd.Series(values).mean()):.4f}"


def external_run_provenance(summary: dict[str, Any]) -> dict[str, Any]:
    prediction_path = ""
    train_path = ""
    prediction_summary: dict[str, Any] = {}
    train_summary: dict[str, Any] = {}
    prediction_root = summary.get("prediction_root")
    if prediction_root not in (None, ""):
        candidate = Path(str(prediction_root)) / "prediction_summary.json"
        if candidate.exists():
            prediction_path = rel_project_path(candidate)
            prediction_summary = read_json(candidate)
    checkpoint = prediction_summary.get("checkpoint")
    if checkpoint not in (None, ""):
        candidate = Path(str(checkpoint)).parent / "train_summary.json"
        if candidate.exists():
            train_path = rel_project_path(candidate)
            train_summary = read_json(candidate)

    return {
        "prediction_summary_path": prediction_path,
        "train_summary_path": train_path,
        "prediction_complete": prediction_summary.get("benchmark_evaluable_without_truncation", ""),
        "n_train_slides": train_summary.get("n_train_slides", ""),
        "n_val_slides": train_summary.get("n_val_slides", ""),
        "n_train_chunks": train_summary.get("n_train_chunks", ""),
        "n_val_chunks": train_summary.get("n_val_chunks", ""),
        "n_train_spots": train_summary.get("n_train_spots", ""),
        "n_val_spots": train_summary.get("n_val_spots", ""),
        "train_epochs": train_summary.get("epochs", ""),
    }


def external_training_is_broad(provenance: dict[str, Any]) -> bool:
    try:
        n_train_slides = int(provenance.get("n_train_slides", 0))
        n_train_chunks = int(provenance.get("n_train_chunks") or 0)
        n_train_spots = int(provenance.get("n_train_spots") or 0)
    except (TypeError, ValueError):
        return False
    return n_train_slides >= 100 and (n_train_chunks >= 1000 or n_train_spots >= 10000)


def build_benchmark_table(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    histo_path = root / EXPR_ROOT / "benchmark_results" / "histoomnist_coverage95" / "summary.csv"
    stat_path = root / EXPR_ROOT / "statistical_baselines" / "summary.csv"
    benchmark_root = root / EXPR_ROOT / "benchmark_results"

    histo = read_csv_if_exists(histo_path)
    for _, row in histo.iterrows():
        rows.append(
            {
                "family": "HistoOmniST",
                "method": row["method"],
                "prediction_kind": row["prediction_kind"],
                "mean_gene_pearson": float(row["mean_gene_pearson"]),
                "median_gene_pearson": float(row["median_gene_pearson"]),
                "valid_genes": int(row["valid_genes"]),
                "scope": "HEST coverage95 held-out test split",
                "evidence_level": "formal_internal",
                "caveat": "",
                "source_path": rel_project_path(histo_path),
            }
        )

    stat = read_csv_if_exists(stat_path)
    for _, row in stat.iterrows():
        rows.append(
            {
                "family": "Statistical/SF-only baseline",
                "method": row["method"],
                "prediction_kind": row["prediction_kind"],
                "mean_gene_pearson": float(row["mean_gene_pearson"]),
                "median_gene_pearson": float(row["median_gene_pearson"]),
                "valid_genes": int(row["valid_genes"]),
                "scope": "HEST coverage95 held-out test split",
                "evidence_level": "formal_internal_baseline",
                "caveat": "",
                "source_path": rel_project_path(stat_path),
            }
        )

    for summary_path in sorted(benchmark_root.glob("*/run_summary.json")):
        run_name = summary_path.parent.name
        if run_name == "histoomnist_coverage95":
            continue
        summary = read_json(summary_path)
        if "gene_metrics" not in summary:
            continue
        metrics = summary.get("gene_metrics", {})
        n_slides = int(summary.get("n_slides", 0))
        provenance = external_run_provenance(summary)
        prediction_complete = provenance.get("prediction_complete")
        complete_text = (
            "complete predictions"
            if prediction_complete is True
            else "prediction completeness unknown"
            if prediction_complete == ""
            else "incomplete/truncated predictions"
        )
        train_text = ""
        if provenance.get("n_train_slides") not in ("", None):
            train_unit = (
                f"{int(provenance['n_train_chunks'])} chunks"
                if provenance.get("n_train_chunks") not in ("", None)
                else f"{int(provenance['n_train_spots'])} spots"
                if provenance.get("n_train_spots") not in ("", None)
                else "unknown training units"
            )
            train_text = (
                f"; training used {int(provenance['n_train_slides'])} train slides/"
                f"{train_unit}"
            )
        is_smoke = bool(summary.get("oracle_smoke_test", False)) or "smoke" in run_name.lower() or n_slides <= 1
        if is_smoke:
            family = "External baseline smoke"
            evidence_level = "smoke_only"
            scope = f"{n_slides} slide engineering smoke"
            caveat = "Do not report as formal external benchmark performance."
        elif n_slides >= 48 and external_training_is_broad(provenance):
            family = "External baseline"
            evidence_level = "formal_external_pilot"
            scope = f"{n_slides} held-out test slides"
            caveat = f"Full test split with {complete_text}{train_text}; one epoch and not a tuned SOTA external benchmark."
        elif n_slides >= 48:
            family = "External baseline full-test limited-training"
            evidence_level = "full_test_limited_external"
            scope = f"{n_slides} held-out test slides"
            caveat = f"Full test split with {complete_text}{train_text}; training is limited, so do not use as final external comparison."
        else:
            family = "External baseline partial"
            evidence_level = "partial_external"
            scope = f"{n_slides} held-out test slides"
            caveat = f"Partial external benchmark with {complete_text}{train_text}; do not use as the final external comparison."
        rows.append(
            {
                "family": family,
                "method": run_name,
                "prediction_kind": summary.get("prediction_kind", "log1p_rate"),
                "mean_gene_pearson": float(metrics.get("mean_gene_pearson", float("nan"))),
                "median_gene_pearson": float(metrics.get("median_gene_pearson", float("nan"))),
                "valid_genes": int(metrics.get("valid_genes", 0)),
                "scope": scope,
                "evidence_level": evidence_level,
                "caveat": caveat,
                "source_path": rel_project_path(summary_path),
                **provenance,
            }
        )
    return pd.DataFrame(rows)


def build_generalization_table(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    readiness_path = root / EXPR_ROOT / "generalization_readiness" / "run_summary.json"
    task_path = root / EXPR_ROOT / "generalization_readiness" / "task_summary.csv"
    ready_task_slugs_by_split: dict[str, set[str]] = {}
    formal_completed_by_stage: dict[str, dict[str, set[str]]] = {
        "expression": {"leave_organ_out": set(), "leave_cohort_out": set()},
        "combined": {"leave_organ_out": set(), "leave_cohort_out": set()},
    }
    formal_sources_by_stage: dict[str, dict[str, set[str]]] = {
        "expression": {"leave_organ_out": set(), "leave_cohort_out": set()},
        "combined": {"leave_organ_out": set(), "leave_cohort_out": set()},
    }
    formal_metric_values_by_stage: dict[str, dict[str, list[float]]] = {
        "expression": {"leave_organ_out": [], "leave_cohort_out": []},
        "combined": {"leave_organ_out": [], "leave_cohort_out": []},
    }
    if readiness_path.exists():
        readiness = read_json(readiness_path)
        rows.append(
            {
                "item": "generalization_task_readiness",
                "status": "ready_for_selected_tasks",
                "scope": "|".join(readiness.get("split_types", [])),
                "n_tasks": int(readiness.get("n_tasks", 0)),
                "n_ready_tasks": int(readiness.get("n_ready_tasks", 0)),
                "n_generated_task_files": int(readiness.get("n_generated_task_files", 0)),
                "metric": "",
                "value": "",
                "caveat": "Readiness only; no trained expression generalization result is implied.",
                "source_path": rel_project_path(readiness_path),
            }
        )
    tasks = read_csv_if_exists(task_path)
    if not tasks.empty and {"split_type", "ready_for_split_specific_training"}.issubset(tasks.columns):
        ready_mask = bool_series(tasks, "ready_for_split_specific_training", False) & bool_series(
            tasks,
            "passes_min_test_slides",
            True,
        )
        for split_type, group in tasks.groupby("split_type", sort=True):
            ready_group = group[ready_mask.loc[group.index]]
            if "task_slug" in group.columns:
                ready_task_slugs_by_split[str(split_type)] = set(ready_group["task_slug"].astype(str))
            rows.append(
                {
                    "item": f"{split_type}_ready_task_count",
                    "status": "readiness_audit",
                    "scope": split_type,
                    "n_tasks": int(len(group)),
                    "n_ready_tasks": int(len(ready_group)),
                    "n_generated_task_files": "",
                    "metric": "missing_asset_paths",
                    "value": int(group.get("n_missing_asset_paths", pd.Series([0])).sum()),
                    "caveat": "Counts reflect manifest/split/assets readiness, not final model performance.",
                    "source_path": rel_project_path(task_path),
                }
            )

    for summary_path, manifest, summary in generalization_runs(root):
        required = {"stage", "split_type", "status"}
        if not required.issubset(summary.columns):
            continue
        run_name = str(manifest.get("run_name") or summary_path.parent.name)
        for (stage_value, split_value), group in summary.groupby(["stage", "split_type"], sort=True):
            stage = str(stage_value)
            split_type = str(split_value)
            ok_mask = group["status"].astype(str).eq("ok")
            ok_count = int(ok_mask.sum())
            if ok_count == len(group) and ok_count > 0:
                status = "ok"
            elif ok_count > 0:
                status = "partial"
            else:
                status = str(group["status"].iloc[0]) if len(group) else "error"
            smoke = is_smoke_generalization_run(manifest, run_name, stage)
            metric, value = generalization_metric(group.loc[ok_mask] if ok_count else group, stage)
            if not smoke and stage in formal_completed_by_stage and "task_slug" in group.columns:
                if split_type in formal_completed_by_stage[stage]:
                    formal_completed_by_stage[stage][split_type].update(group.loc[ok_mask, "task_slug"].astype(str))
                    if ok_count:
                        formal_sources_by_stage[stage][split_type].add(rel_project_path(summary_path))
                        _, task_values = generalization_metric_values(group.loc[ok_mask], stage)
                        formal_metric_values_by_stage[stage][split_type].extend(task_values)
            if smoke:
                caveat = "Smoke/short-run aggregate; checks task plumbing only and is not formal generalization performance."
            elif stage == "sf":
                caveat = "SF generalization run; does not establish coverage95 expression/count generalization."
            else:
                caveat = "Run aggregate; manuscript claims require full ready task coverage and per-task inspection."
            rows.append(
                {
                    "item": f"generalization_run_{run_name}_{stage}_{split_type}",
                    "status": status,
                    "scope": split_type,
                    "n_tasks": int(len(group)),
                    "n_ready_tasks": ok_count,
                    "n_generated_task_files": "",
                    "metric": metric,
                    "value": value,
                    "caveat": caveat,
                    "source_path": rel_project_path(summary_path),
                }
            )

    formal_stage_scopes = {
        "expression": "coverage95 expression-rate",
        "combined": "coverage95 count-scale",
    }
    for split_type in ["leave_organ_out", "leave_cohort_out"]:
        ready = ready_task_slugs_by_split.get(split_type, set())
        for stage, scope in formal_stage_scopes.items():
            completed = formal_completed_by_stage[stage].get(split_type, set())
            sources = formal_sources_by_stage[stage].get(split_type, set())
            metric_values = formal_metric_values_by_stage[stage].get(split_type, [])
            if metric_values:
                metric = {
                    "expression": "task_mean_gene_pearson",
                    "combined": "task_mean_count_pred_sf_gene_pearson",
                }[stage]
                value = f"{float(pd.Series(metric_values).mean()):.4f}"
            else:
                metric = ""
                value = ""
            if not ready:
                status = "not_ready"
                caveat = "No ready task set is available for this split type."
            elif ready.issubset(completed):
                status = "run"
                caveat = f"Full ready task set has formal {scope} outputs; inspect per-task metrics before manuscript claims."
            elif completed:
                status = "partial_not_final"
                caveat = (
                    f"{len(completed)} of {len(ready)} ready tasks have formal {scope} outputs; "
                    "not sufficient for a final generalization claim."
                )
            else:
                status = "not_run"
                caveat = f"Runner/task files exist, but full {scope} training/evaluation has not been run."
            rows.append(
                {
                    "item": f"formal_{split_type}_{stage}",
                    "status": status,
                    "scope": scope,
                    "n_tasks": int(len(ready)),
                    "n_ready_tasks": int(len(completed)),
                    "n_generated_task_files": "",
                    "metric": metric,
                    "value": value,
                    "caveat": caveat,
                    "source_path": "|".join(sorted(sources)),
                }
            )
    return pd.DataFrame(rows)


def build_generalization_inspection_table(root: Path) -> pd.DataFrame:
    inspection_root = root / EXPR_ROOT / "generalization_task_inspection"
    summary_path = inspection_root / "generalization_task_summary.csv"
    summary = read_csv_if_exists(summary_path)
    if summary.empty:
        return pd.DataFrame()
    out = summary.copy().sort_values(["split_type", "stage"]).reset_index(drop=True)
    out["source_path"] = rel_project_path(summary_path)
    return out


def build_biology_table(root: Path) -> pd.DataFrame:
    summary_path = root / EXPR_ROOT / "biological_signatures" / "signature_summary.csv"
    run_path = root / EXPR_ROOT / "biological_signatures" / "run_summary.json"
    summary = read_csv_if_exists(summary_path)
    if summary.empty:
        return pd.DataFrame()
    rate = summary[summary["metric_kind"].astype(str).eq("rate")][
        ["signature", "rate_n_spots", "rate_pearson", "rate_mae", "n_present_genes", "missing_genes"]
    ].rename(columns={"rate_n_spots": "n_spots", "rate_pearson": "rate_signature_pearson", "rate_mae": "rate_mae"})
    count = summary[summary["metric_kind"].astype(str).eq("count_pred_sf")][
        ["signature", "count_pred_sf_pearson", "count_pred_sf_mae"]
    ].rename(columns={"count_pred_sf_mae": "count_mae"})
    out = rate.merge(count, on="signature", how="inner").sort_values("rate_signature_pearson", ascending=False)
    out["missing_genes"] = out["missing_genes"].fillna("")
    out["source_path"] = rel_project_path(summary_path)
    if run_path.exists():
        run_summary = read_json(run_path)
        out["n_spots_evaluated"] = int(run_summary.get("n_spots_evaluated", 0))
        out["is_truncated"] = bool(run_summary.get("is_truncated", True))
    return out.reset_index(drop=True)


def build_pathway_module_table(root: Path) -> pd.DataFrame:
    summary_path = root / EXPR_ROOT / "pathway_modules" / "signature_summary.csv"
    run_path = root / EXPR_ROOT / "pathway_modules" / "run_summary.json"
    summary = read_csv_if_exists(summary_path)
    if summary.empty:
        return pd.DataFrame()
    rate = summary[summary["metric_kind"].astype(str).eq("rate")][
        ["signature", "rate_n_spots", "rate_pearson", "rate_mae", "n_present_genes", "missing_genes"]
    ].rename(
        columns={
            "signature": "module",
            "rate_n_spots": "n_spots",
            "rate_pearson": "rate_module_pearson",
            "rate_mae": "rate_mae",
        }
    )
    count = summary[summary["metric_kind"].astype(str).eq("count_pred_sf")][
        ["signature", "count_pred_sf_pearson", "count_pred_sf_mae"]
    ].rename(
        columns={
            "signature": "module",
            "count_pred_sf_pearson": "count_pred_sf_pearson",
            "count_pred_sf_mae": "count_mae",
        }
    )
    out = rate.merge(count, on="module", how="inner").sort_values("rate_module_pearson", ascending=False)
    out["missing_genes"] = out["missing_genes"].fillna("")
    out["source_path"] = rel_project_path(summary_path)
    if run_path.exists():
        run_summary = read_json(run_path)
        out["n_spots_evaluated"] = int(run_summary.get("n_spots_evaluated", 0))
        out["is_truncated"] = bool(run_summary.get("is_truncated", True))
        out["min_module_genes"] = int(run_summary.get("min_signature_genes", 0))
    return out.reset_index(drop=True)


def build_cell_type_composition_table(root: Path) -> pd.DataFrame:
    composition_root = root / EXPR_ROOT / "cell_type_composition"
    summary_path = composition_root / "cell_type_fraction_summary.csv"
    overall_path = composition_root / "cell_type_composition_overall.csv"
    run_path = composition_root / "run_summary.json"
    summary = read_csv_if_exists(summary_path)
    if summary.empty:
        return pd.DataFrame()
    out = summary.copy().sort_values("spot_fraction_pearson", ascending=False).reset_index(drop=True)
    out["source_path"] = rel_project_path(summary_path)
    overall = read_csv_if_exists(overall_path)
    if not overall.empty:
        row = overall.iloc[0]
        out["overall_n_spots"] = int(row.get("n_spots", 0))
        out["overall_mean_spot_fraction_pearson"] = float(row.get("mean_spot_fraction_pearson", float("nan")))
        out["overall_mean_slide_fraction_pearson"] = float(row.get("mean_slide_fraction_pearson", float("nan")))
        out["dominant_cell_type_accuracy"] = float(row.get("dominant_cell_type_accuracy", float("nan")))
        out["dominant_cell_type_macro_f1"] = float(row.get("dominant_cell_type_macro_f1", float("nan")))
        out["dominant_cell_type_adjusted_rand"] = float(
            row.get("dominant_cell_type_adjusted_rand", float("nan"))
        )
        out["dominant_cell_type_nmi"] = float(row.get("dominant_cell_type_nmi", float("nan")))
    if run_path.exists():
        run_summary = read_json(run_path)
        out["score_kind"] = run_summary.get("score_kind", "")
        out["is_marker_reference_proxy"] = bool(run_summary.get("is_marker_reference_proxy", True))
    return out


def build_state_table(root: Path) -> pd.DataFrame:
    state_root = root / EXPR_ROOT / "biological_signature_states"
    summary_path = state_root / "run_summary.json"
    overall_path = state_root / "state_fidelity_overall.csv"
    organ_path = state_root / "state_fidelity_by_organ.csv"
    annotations_path = state_root / "state_annotations.csv"
    if not summary_path.exists() or not overall_path.exists():
        return pd.DataFrame()
    summary = read_json(summary_path)
    overall = read_csv_if_exists(overall_path)
    if overall.empty:
        return pd.DataFrame()
    overall_row = overall.iloc[0].to_dict()
    rows = [
        {
            "level": "overall",
            "group": "all",
            "score_kind": summary.get("score_kind", overall_row.get("score_kind", "")),
            "n_spots": int(overall_row.get("n_spots", summary.get("n_spots", 0))),
            "n_states": int(overall_row.get("n_states", summary.get("n_states", 0))),
            "adjusted_rand": float(overall_row.get("adjusted_rand", float("nan"))),
            "normalized_mutual_info": float(overall_row.get("normalized_mutual_info", float("nan"))),
            "best_match_accuracy": float(overall_row.get("best_match_accuracy", float("nan"))),
            "dominant_state_signatures": "",
            "source_path": rel_project_path(overall_path),
        }
    ]
    organ = read_csv_if_exists(organ_path)
    if not organ.empty:
        for _, row in organ.iterrows():
            rows.append(
                {
                    "level": "organ",
                    "group": row["organ"],
                    "score_kind": summary.get("score_kind", overall_row.get("score_kind", "")),
                    "n_spots": int(row["n_spots"]),
                    "n_states": int(overall_row.get("n_states", summary.get("n_states", 0))),
                    "adjusted_rand": float(row["adjusted_rand"]),
                    "normalized_mutual_info": float(row["normalized_mutual_info"]),
                    "best_match_accuracy": float(row["best_match_accuracy"]),
                    "dominant_state_signatures": "",
                    "source_path": rel_project_path(organ_path),
                }
            )
    annotations = read_csv_if_exists(annotations_path)
    if not annotations.empty:
        top_states = annotations.sort_values("n_true_spots", ascending=False).head(3)
        rows.append(
            {
                "level": "state_annotation",
                "group": "largest_true_states",
                "score_kind": summary.get("score_kind", overall_row.get("score_kind", "")),
                "n_spots": int(top_states["n_true_spots"].sum()),
                "n_states": int(len(top_states)),
                "adjusted_rand": "",
                "normalized_mutual_info": "",
                "best_match_accuracy": "",
                "dominant_state_signatures": "; ".join(
                    f"state{int(row['state'])}:{row['top_true_signatures']}" for _, row in top_states.iterrows()
                ),
                "source_path": rel_project_path(annotations_path),
            }
        )
    return pd.DataFrame(rows)


def build_spatial_signature_table(root: Path) -> pd.DataFrame:
    spatial_root = root / EXPR_ROOT / "spatial_signature_fidelity"
    summary_path = spatial_root / "spatial_signature_summary.csv"
    overall_path = spatial_root / "spatial_signature_overall.csv"
    run_path = spatial_root / "run_summary.json"
    summary = read_csv_if_exists(summary_path)
    if summary.empty:
        return pd.DataFrame()
    out = summary.copy().sort_values("mean_spatial_lag_pearson", ascending=False).reset_index(drop=True)
    out["source_path"] = rel_project_path(summary_path)
    overall = read_csv_if_exists(overall_path)
    if not overall.empty:
        overall_row = overall.iloc[0]
        out["overall_mean_spatial_lag_pearson"] = float(overall_row.get("mean_spatial_lag_pearson", float("nan")))
        out["overall_mean_hotspot_jaccard"] = float(overall_row.get("mean_hotspot_jaccard", float("nan")))
    if run_path.exists():
        run_summary = read_json(run_path)
        out["score_kind"] = run_summary.get("score_kind", "")
        out["k_neighbors"] = int(run_summary.get("k_neighbors", 0))
        out["hotspot_fraction"] = float(run_summary.get("hotspot_fraction", float("nan")))
    return out


def build_claim_table(
    *,
    benchmark: pd.DataFrame,
    generalization: pd.DataFrame,
    generalization_inspection: pd.DataFrame,
    biology: pd.DataFrame,
    pathways: pd.DataFrame,
    cell_types: pd.DataFrame,
    states: pd.DataFrame,
    spatial: pd.DataFrame,
    diagnostics_summary: dict[str, Any],
) -> pd.DataFrame:
    claims: list[dict[str, Any]] = []
    count_pred = metric_from_method(benchmark, "histoomnist_count_pred_sf")
    count_no = metric_from_method(benchmark, "histoomnist_count_no_sf")
    count_oracle = metric_from_method(benchmark, "histoomnist_count_oracle_sf")
    organ_sf = metric_from_method(benchmark, "organ_sf_only_count_pred_sf")
    if count_pred is not None and count_no is not None:
        delta = count_pred - count_no
        claims.append(
            {
                "claim": "Predicted mean-one SF improves coverage95 count-scale prediction over no-SF counts.",
                "status": "supported",
                "evidence": f"count_pred_sf mean gene Pearson {count_pred:.4f} vs count_no_sf {count_no:.4f}; delta {delta:.4f}.",
                "limitation": f"Oracle SF upper bound remains {count_oracle:.4f}." if count_oracle is not None else "",
                "source_path": "results/hest1k_human_visium_expression/benchmark_results/histoomnist_coverage95/summary.csv",
            }
        )
    if count_pred is not None and organ_sf is not None:
        claims.append(
            {
                "claim": "HistoOmniST exceeds SF-only/statistical count baselines on the same coverage95 evaluator.",
                "status": "supported",
                "evidence": f"HistoOmniST count_pred_sf {count_pred:.4f} vs organ SF-only count_pred_sf {organ_sf:.4f}.",
                "limitation": "External deep-learning method suite and tuning are not yet complete.",
                "source_path": "results/hest1k_human_visium_expression/statistical_baselines/summary.csv",
            }
        )
    expr_overall = diagnostics_summary.get("expression_overall", {})
    if expr_overall:
        n_genes = int(diagnostics_summary.get("n_genes", 0))
        n_spots = int(diagnostics_summary.get("n_spots", 0))
        rate_mean = expr_overall.get("rate", {}).get("mean_gene_pearson")
        claims.append(
            {
                "claim": "The current formal HEST target is a broad coverage95 canonical-symbol panel.",
                "status": "supported",
                "evidence": f"{n_genes} genes evaluated over {n_spots} held-out spots; rate mean gene Pearson {float(rate_mean):.4f}.",
                "limitation": "Broad target lowers average rate correlation relative to smaller top-gene panels.",
                "source_path": "results/hest1k_human_visium_expression/coverage95_diagnostics/run_summary.json",
            }
        )
    if not biology.empty:
        best = biology.iloc[0]
        claims.append(
            {
                "claim": "Predicted expression preserves interpretable marker/signature structure for several biological programs.",
                "status": "supported_first_pass",
                "evidence": (
                    f"Best rate-level signature is {best['signature']} with Pearson "
                    f"{float(best['rate_signature_pearson']):.4f}; {len(biology)} signatures evaluated."
                ),
                "limitation": "This is marker/signature fidelity, not pathway enrichment, cell deconvolution, or clinical validation.",
                "source_path": "results/hest1k_human_visium_expression/biological_signatures/signature_summary.csv",
            }
        )
    if not pathways.empty:
        best = pathways.iloc[0]
        median_rate = float(pd.to_numeric(pathways["rate_module_pearson"], errors="coerce").median())
        n_spots = int(best.get("n_spots_evaluated", best.get("n_spots", 0)))
        claims.append(
            {
                "claim": "Predicted expression preserves pathway-module-level biological programs.",
                "status": "supported_first_pass_pathway",
                "evidence": (
                    f"{len(pathways)} pathway modules evaluated over {n_spots} held-out spots; "
                    f"best module {best['module']} has rate Pearson {float(best['rate_module_pearson']):.4f}; "
                    f"median module Pearson {median_rate:.4f}."
                ),
                "limitation": "Pathway modules are gene-set score fidelity analyses, not formal enrichment, deconvolution, or pathology-region validation.",
                "source_path": "results/hest1k_human_visium_expression/pathway_modules/signature_summary.csv",
            }
        )
    if not cell_types.empty:
        best = cell_types.iloc[0]
        overall_spot = float(cell_types["overall_mean_spot_fraction_pearson"].iloc[0])
        overall_slide = float(cell_types["overall_mean_slide_fraction_pearson"].iloc[0])
        dominant_acc = float(cell_types["dominant_cell_type_accuracy"].iloc[0])
        dominant_f1 = float(cell_types["dominant_cell_type_macro_f1"].iloc[0])
        n_spots = int(cell_types["overall_n_spots"].iloc[0])
        claims.append(
            {
                "claim": "Marker-reference cell-type composition signals are partially recovered from predicted expression.",
                "status": "supported_first_pass_cell_composition",
                "evidence": (
                    f"{len(cell_types)} cell-type signatures over {n_spots} valid held-out spots; "
                    f"mean spot-fraction Pearson {overall_spot:.3f}, mean slide-fraction Pearson {overall_slide:.3f}; "
                    f"dominant cell-type accuracy {dominant_acc:.3f}, macro-F1 {dominant_f1:.3f}; "
                    f"best cell type {best['cell_type']} spot-fraction Pearson "
                    f"{float(best['spot_fraction_pearson']):.3f}."
                ),
                "limitation": "This is a marker-reference composition proxy, not scRNA reference deconvolution or pathology-annotated cell typing.",
                "source_path": "results/hest1k_human_visium_expression/cell_type_composition/cell_type_fraction_summary.csv",
            }
        )
    state_overall = states[states["level"].astype(str).eq("overall")] if not states.empty else pd.DataFrame()
    if not state_overall.empty:
        row = state_overall.iloc[0]
        claims.append(
            {
                "claim": "Signature-derived local tissue states are partially recovered from predicted expression.",
                "status": "supported_first_pass_partial",
                "evidence": (
                    f"{int(row['n_states'])} measured signature-state centroids; predicted assignments reach "
                    f"ARI {float(row['adjusted_rand']):.3f}, NMI {float(row['normalized_mutual_info']):.3f}, "
                    f"best-match accuracy {float(row['best_match_accuracy']):.3f}."
                ),
                "limitation": "States are derived from marker/signature scores, not independent pathology annotations or cell-type deconvolution.",
                "source_path": "results/hest1k_human_visium_expression/biological_signature_states/state_fidelity_overall.csv",
            }
        )
    if not spatial.empty:
        best = spatial.sort_values("mean_spatial_lag_pearson", ascending=False).iloc[0]
        n_pairs = int(pd.to_numeric(spatial["n_slide_signature_pairs"], errors="coerce").fillna(0).sum())
        overall_lag = (
            float(spatial["overall_mean_spatial_lag_pearson"].iloc[0])
            if "overall_mean_spatial_lag_pearson" in spatial.columns
            else float(spatial["mean_spatial_lag_pearson"].mean())
        )
        overall_hotspot = (
            float(spatial["overall_mean_hotspot_jaccard"].iloc[0])
            if "overall_mean_hotspot_jaccard" in spatial.columns
            else float(spatial["mean_hotspot_jaccard"].mean())
        )
        overall_moran_delta = float(spatial["mean_abs_moran_delta"].mean())
        claims.append(
            {
                "claim": "Predicted signature maps retain part of the measured spatial organization.",
                "status": "supported_first_pass_spatial",
                "evidence": (
                    f"{len(spatial)} signatures over {n_pairs} slide-signature pairs; "
                    f"overall mean spatial-lag Pearson {overall_lag:.3f}, mean hotspot Jaccard {overall_hotspot:.3f}; "
                    f"mean absolute Moran's I delta {overall_moran_delta:.3f}; "
                    f"best signature {best['signature']} has mean spatial-lag Pearson "
                    f"{float(best['mean_spatial_lag_pearson']):.3f}."
                ),
                "limitation": "This kNN spatial-signature analysis is marker-derived and shows over-smoothing; it is not independent pathology-region validation.",
                "source_path": "results/hest1k_human_visium_expression/spatial_signature_fidelity/spatial_signature_summary.csv",
            }
        )
    ready_row = generalization[generalization["item"].astype(str).eq("generalization_task_readiness")]
    if not ready_row.empty:
        row = ready_row.iloc[0]
        formal_rows = generalization[generalization["item"].astype(str).str.startswith("formal_leave_")]
        formal_status = (
            ", ".join(f"{item}:{status}" for item, status in formal_rows[["item", "status"]].itertuples(index=False))
            if not formal_rows.empty
            else "no formal expression/count summary rows"
        )
        formal_complete = (
            not formal_rows.empty
            and set(formal_rows["item"].astype(str))
            == {
                "formal_leave_organ_out_expression",
                "formal_leave_organ_out_combined",
                "formal_leave_cohort_out_expression",
                "formal_leave_cohort_out_combined",
            }
            and formal_rows["status"].astype(str).eq("run").all()
        )
        if formal_complete:
            claim = "Leave-organ-out and leave-cohort-out coverage95 generalization runs are complete for the ready task set."
            status = "supported_ready_set"
            if not generalization_inspection.empty:
                limitation = (
                    "The ready set excludes tasks failing minimum test-slide or asset-readiness gates; "
                    "per-task inspection is available and shows heterogeneous performance."
                )
            else:
                limitation = "The ready set excludes tasks failing minimum test-slide or asset-readiness gates; inspect per-task metrics before broad generalization claims."
        else:
            claim = "Leave-slide-out, leave-organ-out, and leave-cohort-out task files are prepared for formal generalization runs."
            status = "prepared_not_final"
            limitation = f"Formal coverage95 expression/count generalization remains incomplete ({formal_status})."
        claims.append(
            {
                "claim": claim,
                "status": status,
                "evidence": (
                    f"{int(row['n_ready_tasks'])}/{int(row['n_tasks'])} tasks ready; "
                    f"{int(row['n_generated_task_files'])} task-file sets generated; {formal_status}."
                ),
                "limitation": limitation,
                "source_path": "results/hest1k_human_visium_expression/generalization_readiness/run_summary.json",
            }
        )
    if not generalization_inspection.empty:
        def inspection_row(split_type: str, stage: str) -> pd.Series | None:
            rows = generalization_inspection[
                generalization_inspection["split_type"].astype(str).eq(split_type)
                & generalization_inspection["stage"].astype(str).eq(stage)
            ]
            if rows.empty:
                return None
            return rows.iloc[0]

        cohort_expr = inspection_row("leave_cohort_out", "expression")
        cohort_count = inspection_row("leave_cohort_out", "combined")
        organ_expr = inspection_row("leave_organ_out", "expression")
        organ_count = inspection_row("leave_organ_out", "combined")
        if all(row is not None for row in [cohort_expr, cohort_count, organ_expr, organ_count]):
            claims.append(
                {
                    "claim": "Ready-set leave-organ-out and leave-cohort-out performance has been inspected per task and is heterogeneous.",
                    "status": "supported_inspected_heterogeneous",
                    "evidence": (
                        f"leave-organ expression/count means {float(organ_expr['mean_primary_metric']):.4f}/"
                        f"{float(organ_count['mean_primary_metric']):.4f}, worst heldout "
                        f"{organ_expr['worst_heldout']} ({float(organ_expr['worst_primary_metric']):.4f})/"
                        f"{organ_count['worst_heldout']} ({float(organ_count['worst_primary_metric']):.4f}); "
                        f"leave-cohort expression/count means {float(cohort_expr['mean_primary_metric']):.4f}/"
                        f"{float(cohort_count['mean_primary_metric']):.4f}, low-task counts "
                        f"{int(cohort_expr['n_low_metric_tasks'])}/{int(cohort_expr['n_tasks'])} and "
                        f"{int(cohort_count['n_low_metric_tasks'])}/{int(cohort_count['n_tasks'])}."
                    ),
                    "limitation": "Supports bounded ready-set reporting; broad cross-tissue or cross-cohort claims need task-level caveats and follow-up on weak heldouts.",
                    "source_path": "results/hest1k_human_visium_expression/generalization_task_inspection/generalization_task_summary.csv",
                }
            )
    external_formal = (
        benchmark[benchmark["evidence_level"].astype(str).isin(["formal_external", "formal_external_pilot"])]
        if not benchmark.empty and "evidence_level" in benchmark.columns
        else pd.DataFrame()
    )
    if not external_formal.empty:
        external_formal = external_formal.sort_values("mean_gene_pearson", ascending=False)
        best_external = external_formal.iloc[0]
        formal_methods = ", ".join(
            (
                f"{row.method} ({float(row.mean_gene_pearson):.4f})"
                for row in external_formal.itertuples(index=False)
            )
        )
        external_limited = (
            benchmark[benchmark["evidence_level"].astype(str).eq("full_test_limited_external")]
            if "evidence_level" in benchmark.columns
            else pd.DataFrame()
        )
        if not external_limited.empty:
            limited_methods = ", ".join(external_limited["method"].astype(str).tolist())
            limitation = (
                "Broad-training full-test external evidence is still limited to formal pilot rows; "
                f"{limited_methods} have full test coverage but limited training, and no external method is tuned."
            )
        else:
            limitation = "External method suite and tuning are not complete; this run used one epoch rather than full tuning."
        claims.append(
            {
                "claim": "Full-split external deep-learning benchmark pilot rows have been run.",
                "status": "supported_pilot",
                "evidence": (
                    f"{len(external_formal)} formal pilot row(s) evaluated on {best_external['scope']}; "
                    f"mean gene Pearson by method: {formal_methods}."
                ),
                "limitation": limitation,
                "source_path": "|".join(external_formal["source_path"].astype(str).tolist()),
            }
        )
    else:
        claims.append(
            {
                "claim": "External deep-learning benchmark performance can be claimed.",
                "status": "not_supported_yet",
                "evidence": "HisToGene patch-H5 path has adapter/training/complete single-slide smoke checks only.",
                "limitation": "Do not compare external method performance in the manuscript until full-slide formal benchmark runs are complete.",
                "source_path": "results/hest1k_human_visium_expression/benchmark_results/histogene_patch_h5_single_slide_smoke/run_summary.json",
            }
        )
    return pd.DataFrame(claims)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows available._"
    view = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).copy()

    def cell_text(value: Any) -> str:
        if isinstance(value, float):
            if pd.isna(value):
                return ""
            text = f"{value:.4f}"
        elif pd.isna(value):
            text = ""
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        values = [cell_text(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown_report(
    *,
    out_path: Path,
    benchmark: pd.DataFrame,
    generalization: pd.DataFrame,
    generalization_inspection: pd.DataFrame,
    biology: pd.DataFrame,
    pathways: pd.DataFrame,
    cell_types: pd.DataFrame,
    states: pd.DataFrame,
    spatial: pd.DataFrame,
    claims: pd.DataFrame,
) -> Path:
    sections = [
        (
            "Evidence Status",
            "\n".join(
                [
                    "This report is generated only from existing result files under `results/`.",
                    "It separates formal results, engineering smoke checks, prepared-but-not-run tasks, and unsupported manuscript claims.",
                ]
            ),
        ),
        (
            "Claim Audit",
            markdown_table(claims, ["status", "claim", "evidence", "limitation", "source_path"], max_rows=20),
        ),
        (
            "Coverage95 Benchmark",
            markdown_table(
                benchmark,
                [
                    "family",
                    "method",
                    "prediction_kind",
                    "mean_gene_pearson",
                    "median_gene_pearson",
                    "valid_genes",
                    "evidence_level",
                    "n_train_slides",
                    "n_train_chunks",
                    "n_train_spots",
                    "prediction_complete",
                    "caveat",
                ],
                max_rows=20,
            ),
        ),
        (
            "Generalization",
            markdown_table(
                generalization,
                ["item", "status", "scope", "metric", "value", "caveat", "source_path"],
                max_rows=24,
            ),
        ),
        (
            "Generalization Task Inspection",
            markdown_table(
                generalization_inspection,
                [
                    "split_type",
                    "stage",
                    "primary_metric_name",
                    "n_tasks",
                    "mean_primary_metric",
                    "min_primary_metric",
                    "n_low_metric_tasks",
                    "worst_heldout",
                    "worst_primary_metric",
                    "best_heldout",
                    "best_primary_metric",
                ],
                max_rows=12,
            ),
        ),
        (
            "Biological Signatures",
            markdown_table(
                biology,
                [
                    "signature",
                    "n_spots",
                    "rate_signature_pearson",
                    "count_pred_sf_pearson",
                    "n_present_genes",
                    "missing_genes",
                ],
                max_rows=20,
            ),
        ),
        (
            "Pathway Module Fidelity",
            markdown_table(
                pathways,
                [
                    "module",
                    "n_spots",
                    "rate_module_pearson",
                    "count_pred_sf_pearson",
                    "n_present_genes",
                    "missing_genes",
                ],
                max_rows=20,
            ),
        ),
        (
            "Cell-Type Composition Fidelity",
            markdown_table(
                cell_types,
                [
                    "cell_type",
                    "spot_fraction_pearson",
                    "slide_fraction_pearson",
                    "spot_fraction_mae",
                    "slide_fraction_mae",
                    "dominant_cell_type_accuracy",
                    "score_kind",
                    "is_marker_reference_proxy",
                ],
                max_rows=20,
            ),
        ),
        (
            "Signature-Derived States",
            markdown_table(
                states,
                [
                    "level",
                    "group",
                    "score_kind",
                    "n_spots",
                    "adjusted_rand",
                    "normalized_mutual_info",
                    "best_match_accuracy",
                    "dominant_state_signatures",
                ],
                max_rows=18,
            ),
        ),
        (
            "Spatial Signature Fidelity",
            markdown_table(
                spatial,
                [
                    "signature",
                    "n_slide_signature_pairs",
                    "mean_spatial_lag_pearson",
                    "median_spatial_lag_pearson",
                    "mean_abs_moran_delta",
                    "mean_hotspot_jaccard",
                    "score_kind",
                    "k_neighbors",
                ],
                max_rows=20,
            ),
        ),
        (
            "Next Required Evidence",
            "\n".join(
                [
                    "1. Expand external deep-learning baselines beyond the one-epoch HisToGene patch-H5 full-split pilot, with fixed preprocessing and tuned/reportable checkpoints.",
                    "2. Extend biological validation beyond marker/signature, pathway-module, kNN spatial-signature, and marker-reference composition fidelity to scRNA-reference deconvolution and pathology-anchored regions.",
                    "3. Follow up weak leave-cohort and leave-organ heldouts identified by per-task inspection before making broad cross-tissue or cross-cohort claims.",
                    "4. Generate final source-data tables for manuscript figures after the formal comparison set is stable.",
                ]
            ),
        ),
    ]
    return write_markdown_report(out_path, "HistoOmniST Coverage95 Evidence Package", sections)


def build_evidence_package(out_dir: Path) -> dict[str, Any]:
    root = project_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = root / EXPR_ROOT / "coverage95_diagnostics" / "run_summary.json"
    diagnostics_summary = read_json(diagnostics_path) if diagnostics_path.exists() else {}
    benchmark = build_benchmark_table(root)
    generalization = build_generalization_table(root)
    generalization_inspection = build_generalization_inspection_table(root)
    biology = build_biology_table(root)
    pathways = build_pathway_module_table(root)
    cell_types = build_cell_type_composition_table(root)
    states = build_state_table(root)
    spatial = build_spatial_signature_table(root)
    claims = build_claim_table(
        benchmark=benchmark,
        generalization=generalization,
        generalization_inspection=generalization_inspection,
        biology=biology,
        pathways=pathways,
        cell_types=cell_types,
        states=states,
        spatial=spatial,
        diagnostics_summary=diagnostics_summary,
    )

    outputs = {
        "benchmark_table": out_dir / "benchmark_evidence_table.csv",
        "generalization_status": out_dir / "generalization_status.csv",
        "generalization_inspection_table": out_dir / "generalization_inspection_evidence_table.csv",
        "biological_signature_table": out_dir / "biological_signature_evidence_table.csv",
        "pathway_module_table": out_dir / "pathway_module_evidence_table.csv",
        "cell_type_composition_table": out_dir / "cell_type_composition_evidence_table.csv",
        "signature_state_table": out_dir / "signature_state_evidence_table.csv",
        "spatial_signature_table": out_dir / "spatial_signature_evidence_table.csv",
        "claim_audit": out_dir / "claim_audit.csv",
        "report": out_dir / "histo_omnist_coverage95_evidence_report.md",
        "manifest": out_dir / "evidence_manifest.json",
    }
    benchmark.to_csv(outputs["benchmark_table"], index=False)
    generalization.to_csv(outputs["generalization_status"], index=False)
    generalization_inspection.to_csv(outputs["generalization_inspection_table"], index=False)
    biology.to_csv(outputs["biological_signature_table"], index=False)
    pathways.to_csv(outputs["pathway_module_table"], index=False)
    cell_types.to_csv(outputs["cell_type_composition_table"], index=False)
    states.to_csv(outputs["signature_state_table"], index=False)
    spatial.to_csv(outputs["spatial_signature_table"], index=False)
    claims.to_csv(outputs["claim_audit"], index=False)
    build_markdown_report(
        out_path=outputs["report"],
        benchmark=benchmark,
        generalization=generalization,
        generalization_inspection=generalization_inspection,
        biology=biology,
        pathways=pathways,
        cell_types=cell_types,
        states=states,
        spatial=spatial,
        claims=claims,
    )
    manifest = {
        "status": "built",
        "inputs": {
            "coverage95_diagnostics": rel_project_path(diagnostics_path),
            "histoomnist_benchmark": f"{EXPR_ROOT}/benchmark_results/histoomnist_coverage95/summary.csv",
            "statistical_baselines": f"{EXPR_ROOT}/statistical_baselines/summary.csv",
            "generalization_readiness": f"{EXPR_ROOT}/generalization_readiness/run_summary.json",
            "generalization_runs": f"{EXPR_ROOT}/generalization_runs/*/summary.csv",
            "generalization_task_inspection": f"{EXPR_ROOT}/generalization_task_inspection/run_summary.json",
            "biological_signatures": f"{EXPR_ROOT}/biological_signatures/run_summary.json",
            "pathway_modules": f"{EXPR_ROOT}/pathway_modules/run_summary.json",
            "cell_type_composition": f"{EXPR_ROOT}/cell_type_composition/run_summary.json",
            "biological_signature_states": f"{EXPR_ROOT}/biological_signature_states/run_summary.json",
            "spatial_signature_fidelity": f"{EXPR_ROOT}/spatial_signature_fidelity/run_summary.json",
        },
        "rows": {
            "benchmark_table": int(len(benchmark)),
            "generalization_status": int(len(generalization)),
            "generalization_inspection_table": int(len(generalization_inspection)),
            "biological_signature_table": int(len(biology)),
            "pathway_module_table": int(len(pathways)),
            "cell_type_composition_table": int(len(cell_types)),
            "signature_state_table": int(len(states)),
            "spatial_signature_table": int(len(spatial)),
            "claim_audit": int(len(claims)),
        },
        "claim_status_counts": claims["status"].value_counts().to_dict() if not claims.empty else {},
        "outputs": {key: rel_project_path(path) for key, path in outputs.items() if key != "manifest"},
    }
    write_json(outputs["manifest"], manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an evidence package from actual HistoOmniST coverage95 results.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    if out_dir is None:
        raise ValueError("out-dir resolved to None")
    build_evidence_package(out_dir)


if __name__ == "__main__":
    main()
