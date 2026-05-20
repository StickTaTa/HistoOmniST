from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from histoomnist.eval.biological_signatures import rel_project_path, write_json
from histoomnist.utils.project_paths import resolve_project_path


DEFAULT_SIGNATURE_DIR = "results/hest1k_human_visium_expression/biological_signatures"
DEFAULT_CELL_TYPE_MAP = "configs/hest1k_cell_type_composition_signatures.csv"
DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/cell_type_composition"


@dataclass
class FractionMetrics:
    n: int
    pearson: float
    mae: float
    rmse: float
    true_mean: float
    pred_mean: float


def read_cell_type_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"cell_type", "signature"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cell-type map missing columns: {missing}")
    frame = frame.copy()
    frame["cell_type"] = frame["cell_type"].astype(str).str.strip()
    frame["signature"] = frame["signature"].astype(str).str.strip()
    frame = frame[(frame["cell_type"] != "") & (frame["signature"] != "")]
    if frame.empty:
        raise ValueError(f"No usable cell-type mapping rows in {path}")
    return frame.drop_duplicates(["cell_type", "signature"]).reset_index(drop=True)


def score_columns(score_kind: str) -> tuple[str, str, str]:
    if score_kind == "rate":
        return "rate_true", "rate_pred", "rate_valid"
    if score_kind == "count_pred_sf":
        return "count_pred_sf_true", "count_pred_sf_pred", "count_pred_sf_valid"
    raise ValueError(f"Unsupported score kind: {score_kind}")


def load_signature_score_matrix(
    signature_dir: Path,
    cell_type_map: pd.DataFrame,
    *,
    score_kind: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    path = signature_dir / "spot_signature_scores.csv"
    scores = pd.read_csv(path)
    true_col, pred_col, valid_col = score_columns(score_kind)
    required = {"row_index", "sample_id", "signature", true_col, pred_col, valid_col}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Spot signature score table missing columns: {missing}")

    available = set(scores["signature"].dropna().astype(str))
    mapped = cell_type_map[cell_type_map["signature"].isin(available)].copy()
    if mapped.empty:
        raise ValueError("None of the requested cell-type signatures are present in spot_signature_scores.csv")
    missing_rows = cell_type_map[~cell_type_map["signature"].isin(available)].copy()

    use_signatures = mapped["signature"].tolist()
    cell_types = mapped["cell_type"].tolist()
    subset = scores[scores["signature"].isin(use_signatures)].copy()
    meta = subset[["row_index", "sample_id"]].drop_duplicates("row_index").sort_values("row_index")

    true = subset.pivot(index="row_index", columns="signature", values=true_col).reindex(meta["row_index"])
    pred = subset.pivot(index="row_index", columns="signature", values=pred_col).reindex(meta["row_index"])
    valid = subset.pivot(index="row_index", columns="signature", values=valid_col).reindex(meta["row_index"])
    true = true.reindex(columns=use_signatures)
    pred = pred.reindex(columns=use_signatures)
    valid = valid.reindex(columns=use_signatures)
    keep = (
        valid.fillna(False).astype(bool).all(axis=1).to_numpy()
        & np.isfinite(true.to_numpy(dtype=np.float32)).all(axis=1)
        & np.isfinite(pred.to_numpy(dtype=np.float32)).all(axis=1)
    )
    meta = meta.loc[keep].reset_index(drop=True)
    true_matrix = true.to_numpy(dtype=np.float32)[keep]
    pred_matrix = pred.to_numpy(dtype=np.float32)[keep]
    coverage_rows = []
    for row in cell_type_map.itertuples(index=False):
        coverage_rows.append(
            {
                "cell_type": row.cell_type,
                "signature": row.signature,
                "used": bool(row.signature in set(use_signatures)),
                "reason": "" if row.signature in set(use_signatures) else "signature_score_missing",
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    if not missing_rows.empty:
        coverage = coverage.sort_values(["used", "cell_type"], ascending=[False, True]).reset_index(drop=True)
    return meta, true_matrix, pred_matrix, cell_types, coverage


def zscore_from_true(true_matrix: np.ndarray, pred_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = true_matrix.mean(axis=0, keepdims=True)
    std = true_matrix.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return (true_matrix - mean) / std, (pred_matrix - mean) / std, mean.reshape(-1), std.reshape(-1)


def softmax_fraction(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1.0e-6)
    scaled = scores / temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    denom = np.clip(exp.sum(axis=1, keepdims=True), 1.0e-8, None)
    return (exp / denom).astype(np.float32, copy=False)


def fraction_metrics(true: np.ndarray, pred: np.ndarray) -> FractionMetrics:
    keep = np.isfinite(true) & np.isfinite(pred)
    if keep.sum() < 3:
        return FractionMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    x = pred[keep].astype(np.float64)
    y = true[keep].astype(np.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    pearson = float(np.sum(x_centered * y_centered) / denom) if denom > 0 else float("nan")
    err = x - y
    return FractionMetrics(
        n=int(keep.sum()),
        pearson=pearson,
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err * err))),
        true_mean=float(y.mean()),
        pred_mean=float(x.mean()),
    )


def per_cell_type_metrics(
    *,
    true_fraction: np.ndarray,
    pred_fraction: np.ndarray,
    cell_types: list[str],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    slide_true = []
    slide_pred = []
    slide_ids = []
    for sample_id, indices in meta.groupby("sample_id", sort=True).indices.items():
        slide_ids.append(str(sample_id))
        slide_true.append(true_fraction[indices].mean(axis=0))
        slide_pred.append(pred_fraction[indices].mean(axis=0))
    slide_true_matrix = np.vstack(slide_true) if slide_true else np.zeros((0, len(cell_types)), dtype=np.float32)
    slide_pred_matrix = np.vstack(slide_pred) if slide_pred else np.zeros((0, len(cell_types)), dtype=np.float32)

    for idx, cell_type in enumerate(cell_types):
        spot = fraction_metrics(true_fraction[:, idx], pred_fraction[:, idx])
        slide = fraction_metrics(slide_true_matrix[:, idx], slide_pred_matrix[:, idx])
        rows.append(
            {
                "cell_type": cell_type,
                "spot_n": spot.n,
                "spot_fraction_pearson": spot.pearson,
                "spot_fraction_mae": spot.mae,
                "spot_fraction_rmse": spot.rmse,
                "spot_true_mean_fraction": spot.true_mean,
                "spot_pred_mean_fraction": spot.pred_mean,
                "slide_n": slide.n,
                "slide_fraction_pearson": slide.pearson,
                "slide_fraction_mae": slide.mae,
                "slide_fraction_rmse": slide.rmse,
                "slide_true_mean_fraction": slide.true_mean,
                "slide_pred_mean_fraction": slide.pred_mean,
            }
        )
    return pd.DataFrame(rows)


def slide_fraction_table(
    *,
    true_fraction: np.ndarray,
    pred_fraction: np.ndarray,
    cell_types: list[str],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for sample_id, indices in meta.groupby("sample_id", sort=True).indices.items():
        true_mean = true_fraction[indices].mean(axis=0)
        pred_mean = pred_fraction[indices].mean(axis=0)
        row: dict[str, Any] = {"sample_id": str(sample_id), "n_spots": int(len(indices))}
        for idx, cell_type in enumerate(cell_types):
            row[f"true_{cell_type}"] = float(true_mean[idx])
            row[f"pred_{cell_type}"] = float(pred_mean[idx])
            row[f"delta_{cell_type}"] = float(pred_mean[idx] - true_mean[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def wide_fraction_table(
    *,
    meta: pd.DataFrame,
    true_fraction: np.ndarray,
    pred_fraction: np.ndarray,
    cell_types: list[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> pd.DataFrame:
    out = meta[["row_index", "sample_id"]].copy()
    out["true_dominant_cell_type"] = [cell_types[int(i)] for i in true_labels]
    out["pred_dominant_cell_type"] = [cell_types[int(i)] for i in pred_labels]
    for idx, cell_type in enumerate(cell_types):
        out[f"true_fraction_{cell_type}"] = true_fraction[:, idx]
        out[f"pred_fraction_{cell_type}"] = pred_fraction[:, idx]
    return out


def confusion_table(true_labels: np.ndarray, pred_labels: np.ndarray, cell_types: list[str]) -> pd.DataFrame:
    confusion = np.zeros((len(cell_types), len(cell_types)), dtype=np.int64)
    for true, pred in zip(true_labels, pred_labels):
        confusion[int(true), int(pred)] += 1
    rows = []
    for true_idx, true_type in enumerate(cell_types):
        for pred_idx, pred_type in enumerate(cell_types):
            rows.append(
                {
                    "true_cell_type": true_type,
                    "pred_cell_type": pred_type,
                    "n_spots": int(confusion[true_idx, pred_idx]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_cell_type_composition(
    *,
    signature_dir: Path,
    cell_type_map_path: Path,
    out_dir: Path,
    score_kind: str,
    temperature: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_type_map = read_cell_type_map(cell_type_map_path)
    meta, true_scores, pred_scores, cell_types, coverage = load_signature_score_matrix(
        signature_dir,
        cell_type_map,
        score_kind=score_kind,
    )
    if len(cell_types) < 2:
        raise ValueError("Cell-type composition analysis requires at least two usable cell-type signatures.")
    true_scaled, pred_scaled, score_mean, score_std = zscore_from_true(true_scores, pred_scores)
    true_fraction = softmax_fraction(true_scaled, temperature=temperature)
    pred_fraction = softmax_fraction(pred_scaled, temperature=temperature)
    true_labels = np.argmax(true_fraction, axis=1)
    pred_labels = np.argmax(pred_fraction, axis=1)

    metrics = per_cell_type_metrics(
        true_fraction=true_fraction,
        pred_fraction=pred_fraction,
        cell_types=cell_types,
        meta=meta,
    )
    slide = slide_fraction_table(
        true_fraction=true_fraction,
        pred_fraction=pred_fraction,
        cell_types=cell_types,
        meta=meta,
    )
    fractions = wide_fraction_table(
        meta=meta,
        true_fraction=true_fraction,
        pred_fraction=pred_fraction,
        cell_types=cell_types,
        true_labels=true_labels,
        pred_labels=pred_labels,
    )
    confusion = confusion_table(true_labels, pred_labels, cell_types)

    overall = {
        "score_kind": score_kind,
        "n_spots": int(len(meta)),
        "n_cell_types": int(len(cell_types)),
        "temperature": float(temperature),
        "mean_spot_fraction_pearson": float(metrics["spot_fraction_pearson"].mean()),
        "median_spot_fraction_pearson": float(metrics["spot_fraction_pearson"].median()),
        "mean_slide_fraction_pearson": float(metrics["slide_fraction_pearson"].mean()),
        "median_slide_fraction_pearson": float(metrics["slide_fraction_pearson"].median()),
        "dominant_cell_type_accuracy": float(np.mean(true_labels == pred_labels)),
        "dominant_cell_type_macro_f1": float(f1_score(true_labels, pred_labels, average="macro")),
        "dominant_cell_type_adjusted_rand": float(adjusted_rand_score(true_labels, pred_labels)),
        "dominant_cell_type_nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
    }
    score_scaling = pd.DataFrame(
        {
            "cell_type": cell_types,
            "true_score_mean": score_mean,
            "true_score_std": score_std,
        }
    )

    coverage_path = out_dir / "cell_type_signature_coverage.csv"
    metrics_path = out_dir / "cell_type_fraction_summary.csv"
    overall_path = out_dir / "cell_type_composition_overall.csv"
    slide_path = out_dir / "slide_cell_type_fractions.csv"
    fractions_path = out_dir / "spot_cell_type_fractions.csv"
    confusion_path = out_dir / "dominant_cell_type_confusion.csv"
    score_scaling_path = out_dir / "cell_type_score_scaling.csv"
    coverage.to_csv(coverage_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame([overall]).to_csv(overall_path, index=False)
    slide.to_csv(slide_path, index=False)
    fractions.to_csv(fractions_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    score_scaling.to_csv(score_scaling_path, index=False)

    summary = {
        "signature_dir": rel_project_path(signature_dir),
        "cell_type_map": rel_project_path(cell_type_map_path),
        "score_kind": score_kind,
        "n_spots": int(len(meta)),
        "n_cell_types": int(len(cell_types)),
        "cell_types": cell_types,
        "temperature": float(temperature),
        "overall": overall,
        "is_marker_reference_proxy": True,
        "outputs": {
            "cell_type_signature_coverage": rel_project_path(coverage_path),
            "cell_type_fraction_summary": rel_project_path(metrics_path),
            "cell_type_composition_overall": rel_project_path(overall_path),
            "slide_cell_type_fractions": rel_project_path(slide_path),
            "spot_cell_type_fractions": rel_project_path(fractions_path),
            "dominant_cell_type_confusion": rel_project_path(confusion_path),
            "cell_type_score_scaling": rel_project_path(score_scaling_path),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate marker-reference cell-type composition fidelity.")
    parser.add_argument("--signature-dir", default=DEFAULT_SIGNATURE_DIR)
    parser.add_argument("--cell-type-map", default=DEFAULT_CELL_TYPE_MAP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-kind", choices=["rate", "count_pred_sf"], default="rate")
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signature_dir = resolve_project_path(args.signature_dir)
    cell_type_map = resolve_project_path(args.cell_type_map)
    out_dir = resolve_project_path(args.out_dir)
    if None in (signature_dir, cell_type_map, out_dir):
        raise ValueError("Required paths did not resolve.")
    evaluate_cell_type_composition(
        signature_dir=signature_dir,
        cell_type_map_path=cell_type_map,
        out_dir=out_dir,
        score_kind=str(args.score_kind),
        temperature=float(args.temperature),
    )


if __name__ == "__main__":
    main()
