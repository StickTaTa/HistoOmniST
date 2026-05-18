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


def generalization_metric(group: pd.DataFrame, stage: str) -> tuple[str, str]:
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
                return label, f"{float(values.mean()):.4f}"
    return "", ""


def build_benchmark_table(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    histo_path = root / EXPR_ROOT / "benchmark_results" / "histoomnist_coverage95" / "summary.csv"
    stat_path = root / EXPR_ROOT / "statistical_baselines" / "summary.csv"
    histogene_path = root / EXPR_ROOT / "benchmark_results" / "histogene_patch_h5_single_slide_smoke" / "run_summary.json"

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

    if histogene_path.exists():
        summary = read_json(histogene_path)
        metrics = summary.get("gene_metrics", {})
        rows.append(
            {
                "family": "External baseline smoke",
                "method": summary.get("method", "histogene_patch_h5"),
                "prediction_kind": summary.get("prediction_kind", "log1p_rate"),
                "mean_gene_pearson": float(metrics.get("mean_gene_pearson", float("nan"))),
                "median_gene_pearson": float(metrics.get("median_gene_pearson", float("nan"))),
                "valid_genes": int(metrics.get("valid_genes", 0)),
                "scope": f"{int(summary.get('n_slides', 0))} slide engineering smoke",
                "evidence_level": "smoke_only",
                "caveat": "Do not report as formal external benchmark performance.",
                "source_path": rel_project_path(histogene_path),
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
            if not smoke and stage in formal_completed_by_stage and "task_slug" in group.columns:
                if split_type in formal_completed_by_stage[stage]:
                    formal_completed_by_stage[stage][split_type].update(group.loc[ok_mask, "task_slug"].astype(str))
                    if ok_count:
                        formal_sources_by_stage[stage][split_type].add(rel_project_path(summary_path))
            metric, value = generalization_metric(group.loc[ok_mask] if ok_count else group, stage)
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
                    "metric": "",
                    "value": "",
                    "caveat": caveat,
                    "source_path": "|".join(sorted(sources)),
                }
            )
    return pd.DataFrame(rows)


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


def build_claim_table(
    *,
    benchmark: pd.DataFrame,
    generalization: pd.DataFrame,
    biology: pd.DataFrame,
    states: pd.DataFrame,
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
                "limitation": "External deep-learning baselines are not yet formally complete.",
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
    ready_row = generalization[generalization["item"].astype(str).eq("generalization_task_readiness")]
    if not ready_row.empty:
        row = ready_row.iloc[0]
        formal_rows = generalization[generalization["item"].astype(str).str.startswith("formal_leave_")]
        formal_status = (
            ", ".join(f"{item}:{status}" for item, status in formal_rows[["item", "status"]].itertuples(index=False))
            if not formal_rows.empty
            else "no formal expression/count summary rows"
        )
        claims.append(
            {
                "claim": "Leave-slide-out, leave-organ-out, and leave-cohort-out task files are prepared for formal generalization runs.",
                "status": "prepared_not_final",
                "evidence": (
                    f"{int(row['n_ready_tasks'])}/{int(row['n_tasks'])} tasks ready; "
                    f"{int(row['n_generated_task_files'])} task-file sets generated."
                ),
                "limitation": f"Formal coverage95 expression/count generalization remains incomplete ({formal_status}).",
                "source_path": "results/hest1k_human_visium_expression/generalization_readiness/run_summary.json",
            }
        )
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
    biology: pd.DataFrame,
    states: pd.DataFrame,
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
            "Next Required Evidence",
            "\n".join(
                [
                    "1. Run full leave-organ-out and leave-cohort-out coverage95 expression/count evaluations.",
                    "2. Run at least one full external baseline under the unified coverage95 benchmark harness.",
                    "3. Extend biological fidelity from marker/signature states to pathway modules, cell-type deconvolution, and pathology-anchored validation.",
                    "4. Generate final source-data tables for any manuscript figures after the formal runs complete.",
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
    biology = build_biology_table(root)
    states = build_state_table(root)
    claims = build_claim_table(
        benchmark=benchmark,
        generalization=generalization,
        biology=biology,
        states=states,
        diagnostics_summary=diagnostics_summary,
    )

    outputs = {
        "benchmark_table": out_dir / "benchmark_evidence_table.csv",
        "generalization_status": out_dir / "generalization_status.csv",
        "biological_signature_table": out_dir / "biological_signature_evidence_table.csv",
        "signature_state_table": out_dir / "signature_state_evidence_table.csv",
        "claim_audit": out_dir / "claim_audit.csv",
        "report": out_dir / "histo_omnist_coverage95_evidence_report.md",
        "manifest": out_dir / "evidence_manifest.json",
    }
    benchmark.to_csv(outputs["benchmark_table"], index=False)
    generalization.to_csv(outputs["generalization_status"], index=False)
    biology.to_csv(outputs["biological_signature_table"], index=False)
    states.to_csv(outputs["signature_state_table"], index=False)
    claims.to_csv(outputs["claim_audit"], index=False)
    build_markdown_report(
        out_path=outputs["report"],
        benchmark=benchmark,
        generalization=generalization,
        biology=biology,
        states=states,
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
            "biological_signatures": f"{EXPR_ROOT}/biological_signatures/run_summary.json",
            "biological_signature_states": f"{EXPR_ROOT}/biological_signature_states/run_summary.json",
        },
        "rows": {
            "benchmark_table": int(len(benchmark)),
            "generalization_status": int(len(generalization)),
            "biological_signature_table": int(len(biology)),
            "signature_state_table": int(len(states)),
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
