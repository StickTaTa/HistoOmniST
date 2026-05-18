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
    smoke_path = root / EXPR_ROOT / "generalization_runs" / "smoke_sf_generated_tasks_epoch1_runner" / "summary.csv"
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
        for split_type, group in tasks.groupby("split_type", sort=True):
            rows.append(
                {
                    "item": f"{split_type}_ready_task_count",
                    "status": "readiness_audit",
                    "scope": split_type,
                    "n_tasks": int(len(group)),
                    "n_ready_tasks": int(group["ready_for_split_specific_training"].sum()),
                    "n_generated_task_files": "",
                    "metric": "missing_asset_paths",
                    "value": int(group.get("n_missing_asset_paths", pd.Series([0])).sum()),
                    "caveat": "Counts reflect manifest/split/assets readiness, not final model performance.",
                    "source_path": rel_project_path(task_path),
                }
            )
    smoke = read_csv_if_exists(smoke_path)
    if not smoke.empty:
        for _, row in smoke.iterrows():
            rows.append(
                {
                    "item": f"sf_smoke_{row['split_type']}_{row['task_slug']}",
                    "status": row["status"],
                    "scope": f"{row['split_type']}:{row['heldout']}",
                    "n_tasks": "",
                    "n_ready_tasks": "",
                    "n_generated_task_files": "",
                    "metric": "log_sf_pearson",
                    "value": float(row["metric_log_sf_pearson"]),
                    "caveat": "One-epoch SF smoke; not formal generalization performance.",
                    "source_path": rel_project_path(smoke_path),
                }
            )
    rows.append(
        {
            "item": "formal_leave_organ_out_expression",
            "status": "not_run",
            "scope": "coverage95 expression/count",
            "n_tasks": "",
            "n_ready_tasks": "",
            "n_generated_task_files": "",
            "metric": "",
            "value": "",
            "caveat": "Runner and task files exist, but full expression generalization training/evaluation has not been run.",
            "source_path": "",
        }
    )
    rows.append(
        {
            "item": "formal_leave_cohort_out_expression",
            "status": "not_run",
            "scope": "coverage95 expression/count",
            "n_tasks": "",
            "n_ready_tasks": "",
            "n_generated_task_files": "",
            "metric": "",
            "value": "",
            "caveat": "Runner and task files exist, but full expression generalization training/evaluation has not been run.",
            "source_path": "",
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


def build_claim_table(
    *,
    benchmark: pd.DataFrame,
    generalization: pd.DataFrame,
    biology: pd.DataFrame,
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
    ready_row = generalization[generalization["item"].astype(str).eq("generalization_task_readiness")]
    if not ready_row.empty:
        row = ready_row.iloc[0]
        claims.append(
            {
                "claim": "Leave-slide-out, leave-organ-out, and leave-cohort-out task files are prepared for formal generalization runs.",
                "status": "prepared_not_final",
                "evidence": (
                    f"{int(row['n_ready_tasks'])}/{int(row['n_tasks'])} tasks ready; "
                    f"{int(row['n_generated_task_files'])} task-file sets generated."
                ),
                "limitation": "Only SF smoke runs exist; full coverage95 expression generalization metrics are not yet run.",
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
            "Next Required Evidence",
            "\n".join(
                [
                    "1. Run full leave-organ-out and leave-cohort-out coverage95 expression/count evaluations.",
                    "2. Run at least one full external baseline under the unified coverage95 benchmark harness.",
                    "3. Extend biological fidelity from marker signatures to pathway/module and cell-state analyses.",
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
    claims = build_claim_table(
        benchmark=benchmark,
        generalization=generalization,
        biology=biology,
        diagnostics_summary=diagnostics_summary,
    )

    outputs = {
        "benchmark_table": out_dir / "benchmark_evidence_table.csv",
        "generalization_status": out_dir / "generalization_status.csv",
        "biological_signature_table": out_dir / "biological_signature_evidence_table.csv",
        "claim_audit": out_dir / "claim_audit.csv",
        "report": out_dir / "histo_omnist_coverage95_evidence_report.md",
        "manifest": out_dir / "evidence_manifest.json",
    }
    benchmark.to_csv(outputs["benchmark_table"], index=False)
    generalization.to_csv(outputs["generalization_status"], index=False)
    biology.to_csv(outputs["biological_signature_table"], index=False)
    claims.to_csv(outputs["claim_audit"], index=False)
    build_markdown_report(
        out_path=outputs["report"],
        benchmark=benchmark,
        generalization=generalization,
        biology=biology,
        claims=claims,
    )
    manifest = {
        "status": "built",
        "inputs": {
            "coverage95_diagnostics": rel_project_path(diagnostics_path),
            "histoomnist_benchmark": f"{EXPR_ROOT}/benchmark_results/histoomnist_coverage95/summary.csv",
            "statistical_baselines": f"{EXPR_ROOT}/statistical_baselines/summary.csv",
            "generalization_readiness": f"{EXPR_ROOT}/generalization_readiness/run_summary.json",
            "generalization_smoke": f"{EXPR_ROOT}/generalization_runs/smoke_sf_generated_tasks_epoch1_runner/summary.csv",
            "biological_signatures": f"{EXPR_ROOT}/biological_signatures/run_summary.json",
        },
        "rows": {
            "benchmark_table": int(len(benchmark)),
            "generalization_status": int(len(generalization)),
            "biological_signature_table": int(len(biology)),
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
