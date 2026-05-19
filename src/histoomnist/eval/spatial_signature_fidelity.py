from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from histoomnist.eval.biological_signatures import build_coords_for_dataset_order
from histoomnist.utils.config import load_config
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import project_root, resolve_project_path


DEFAULT_SIGNATURE_DIR = "results/hest1k_human_visium_expression/biological_signatures"
DEFAULT_EXPRESSION_CONFIG = "configs/hest1k_human_visium_expression_highconf_symbol95.yaml"
DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/spatial_signature_fidelity"


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


def score_columns(score_kind: str) -> tuple[str, str, str]:
    if score_kind == "rate":
        return "rate_true", "rate_pred", "rate_valid"
    if score_kind == "count_pred_sf":
        return "count_pred_sf_true", "count_pred_sf_pred", "count_pred_sf_valid"
    raise ValueError(f"Unsupported score kind: {score_kind}")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return float("nan")
    xx = x[keep].astype(np.float64, copy=False)
    yy = y[keep].astype(np.float64, copy=False)
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    denom = math.sqrt(float(np.sum(xx * xx) * np.sum(yy * yy)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(xx * yy) / denom)


def zscore(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=False)
    out = out - float(np.mean(out))
    std = float(np.std(out))
    if std <= 1e-12:
        return np.zeros_like(out, dtype=np.float64)
    return out / std


def moran_i(values: np.ndarray, neighbor_idx: np.ndarray) -> float:
    if values.size < 3 or neighbor_idx.size == 0:
        return float("nan")
    z = zscore(values)
    denom = float(np.sum(z * z))
    if denom <= 0:
        return float("nan")
    lag = z[neighbor_idx].mean(axis=1)
    return float(np.sum(z * lag) / denom)


def hotspot_jaccard(true_values: np.ndarray, pred_values: np.ndarray, fraction: float) -> float:
    n = int(true_values.size)
    if n < 3:
        return float("nan")
    n_hot = max(1, int(round(n * float(fraction))))
    true_top = np.argpartition(true_values, -n_hot)[-n_hot:]
    pred_top = np.argpartition(pred_values, -n_hot)[-n_hot:]
    true_set = set(int(i) for i in true_top)
    pred_set = set(int(i) for i in pred_top)
    union = len(true_set | pred_set)
    if union == 0:
        return float("nan")
    return float(len(true_set & pred_set) / union)


def load_scores_with_metadata(
    *,
    signature_dir: Path,
    expression_config: dict[str, Any],
    score_kind: str,
) -> pd.DataFrame:
    score_path = signature_dir / "spot_signature_scores.csv"
    scores = pd.read_csv(score_path)
    true_col, pred_col, valid_col = score_columns(score_kind)
    required = {"row_index", "sample_id", "signature", true_col, pred_col, valid_col}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Spot signature score table missing columns: {missing}")

    manifest_path = Path(expression_config["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    split_names = list(expression_config["data"].get("test_splits", ["test"]))
    coords = build_coords_for_dataset_order(
        manifest=manifest,
        manifest_base=manifest_path.parent,
        splits=split_names,
        min_total_counts=float(expression_config["data"].get("min_total_counts", 1.0)),
    )
    row_index = scores["row_index"].to_numpy(dtype=np.int64)
    if row_index.max(initial=-1) >= coords.shape[0]:
        raise ValueError("Spot score row_index exceeds coordinate table length.")
    scores["x"] = coords[row_index, 0]
    scores["y"] = coords[row_index, 1]

    sample_info = manifest.drop_duplicates("sample_id").set_index("sample_id")
    for column in ["organ", "cohort", "split"]:
        if column in sample_info.columns:
            scores[column] = scores["sample_id"].map(sample_info[column].astype(str).to_dict()).fillna("unknown")
        else:
            scores[column] = "unknown"
    scores = scores.rename(columns={true_col: "true_score", pred_col: "pred_score", valid_col: "valid_score"})
    scores["valid_score"] = scores["valid_score"].astype(bool)
    return scores


def neighbor_indices(coords: np.ndarray, k_neighbors: int) -> np.ndarray:
    n = int(coords.shape[0])
    if n <= 2:
        return np.zeros((0, 0), dtype=np.int64)
    k = min(int(k_neighbors), n - 1)
    model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    model.fit(coords)
    indices = model.kneighbors(coords, return_distance=False)
    return indices[:, 1:].astype(np.int64, copy=False)


def evaluate_group(
    group: pd.DataFrame,
    *,
    k_neighbors: int,
    hotspot_fraction: float,
    min_spots: int,
) -> dict[str, Any] | None:
    keep = (
        group["valid_score"].to_numpy(dtype=bool)
        & np.isfinite(group["true_score"].to_numpy(dtype=np.float64))
        & np.isfinite(group["pred_score"].to_numpy(dtype=np.float64))
        & np.isfinite(group["x"].to_numpy(dtype=np.float64))
        & np.isfinite(group["y"].to_numpy(dtype=np.float64))
    )
    if int(keep.sum()) < int(min_spots):
        return None
    valid = group.loc[keep].copy()
    coords = valid[["x", "y"]].to_numpy(dtype=np.float64)
    true_values = valid["true_score"].to_numpy(dtype=np.float64)
    pred_values = valid["pred_score"].to_numpy(dtype=np.float64)
    neighbors = neighbor_indices(coords, k_neighbors=k_neighbors)
    if neighbors.size == 0:
        return None

    true_z = zscore(true_values)
    pred_z = zscore(pred_values)
    true_lag = true_z[neighbors].mean(axis=1)
    pred_lag = pred_z[neighbors].mean(axis=1)
    true_i = moran_i(true_values, neighbors)
    pred_i = moran_i(pred_values, neighbors)
    return {
        "n_spots": int(len(valid)),
        "k_neighbors": int(neighbors.shape[1]),
        "spot_pearson": pearson(true_values, pred_values),
        "true_moran_i": true_i,
        "pred_moran_i": pred_i,
        "moran_abs_delta": float(abs(pred_i - true_i)) if np.isfinite(true_i) and np.isfinite(pred_i) else float("nan"),
        "spatial_lag_pearson": pearson(true_lag, pred_lag),
        "hotspot_jaccard": hotspot_jaccard(true_values, pred_values, hotspot_fraction),
    }


def summary_by_signature(by_slide: pd.DataFrame) -> pd.DataFrame:
    if by_slide.empty:
        return pd.DataFrame()
    grouped = by_slide.groupby("signature", sort=True)
    rows = []
    for signature, group in grouped:
        rows.append(
            {
                "signature": signature,
                "n_slide_signature_pairs": int(len(group)),
                "n_slides": int(group["sample_id"].nunique()),
                "n_spots": int(group["n_spots"].sum()),
                "mean_spot_pearson": float(group["spot_pearson"].mean()),
                "mean_true_moran_i": float(group["true_moran_i"].mean()),
                "mean_pred_moran_i": float(group["pred_moran_i"].mean()),
                "mean_abs_moran_delta": float(group["moran_abs_delta"].mean()),
                "mean_spatial_lag_pearson": float(group["spatial_lag_pearson"].mean()),
                "median_spatial_lag_pearson": float(group["spatial_lag_pearson"].median()),
                "mean_hotspot_jaccard": float(group["hotspot_jaccard"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_spatial_lag_pearson", ascending=False).reset_index(drop=True)


def overall_summary(summary: pd.DataFrame, by_slide: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or by_slide.empty:
        return pd.DataFrame()
    row = {
        "score_kind": str(by_slide["score_kind"].iloc[0]),
        "n_signatures": int(summary["signature"].nunique()),
        "n_slides": int(by_slide["sample_id"].nunique()),
        "n_slide_signature_pairs": int(len(by_slide)),
        "n_spots": int(by_slide["n_spots"].sum()),
        "mean_abs_moran_delta": float(by_slide["moran_abs_delta"].mean()),
        "mean_spatial_lag_pearson": float(by_slide["spatial_lag_pearson"].mean()),
        "median_spatial_lag_pearson": float(by_slide["spatial_lag_pearson"].median()),
        "mean_hotspot_jaccard": float(by_slide["hotspot_jaccard"].mean()),
        "best_signature": str(summary.iloc[0]["signature"]),
        "best_signature_spatial_lag_pearson": float(summary.iloc[0]["mean_spatial_lag_pearson"]),
    }
    return pd.DataFrame([row])


def evaluate_spatial_signature_fidelity(
    *,
    signature_dir: Path,
    expression_config: dict[str, Any],
    expression_config_path: Path,
    out_dir: Path,
    score_kind: str,
    k_neighbors: int,
    hotspot_fraction: float,
    min_spots: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = load_scores_with_metadata(
        signature_dir=signature_dir,
        expression_config=expression_config,
        score_kind=score_kind,
    )

    rows = []
    for (sample_id, signature), group in scores.groupby(["sample_id", "signature"], sort=True):
        metrics = evaluate_group(
            group,
            k_neighbors=k_neighbors,
            hotspot_fraction=hotspot_fraction,
            min_spots=min_spots,
        )
        if metrics is None:
            continue
        first = group.iloc[0]
        rows.append(
            {
                "sample_id": str(sample_id),
                "signature": str(signature),
                "organ": str(first.get("organ", "unknown")),
                "cohort": str(first.get("cohort", "unknown")),
                "score_kind": score_kind,
                "hotspot_fraction": float(hotspot_fraction),
                **metrics,
            }
        )

    by_slide_columns = [
        "sample_id",
        "signature",
        "organ",
        "cohort",
        "score_kind",
        "hotspot_fraction",
        "n_spots",
        "k_neighbors",
        "spot_pearson",
        "true_moran_i",
        "pred_moran_i",
        "moran_abs_delta",
        "spatial_lag_pearson",
        "hotspot_jaccard",
    ]
    by_slide = pd.DataFrame(rows, columns=by_slide_columns)
    if not by_slide.empty:
        by_slide = by_slide.sort_values(["signature", "sample_id"]).reset_index(drop=True)
    signature_summary = summary_by_signature(by_slide)
    overall = overall_summary(signature_summary, by_slide)

    by_slide_path = out_dir / "spatial_signature_by_slide.csv"
    signature_summary_path = out_dir / "spatial_signature_summary.csv"
    overall_path = out_dir / "spatial_signature_overall.csv"
    by_slide.to_csv(by_slide_path, index=False)
    signature_summary.to_csv(signature_summary_path, index=False)
    overall.to_csv(overall_path, index=False)

    run_summary = {
        "signature_dir": rel_project_path(signature_dir),
        "expression_config": rel_project_path(expression_config_path),
        "score_kind": score_kind,
        "k_neighbors": int(k_neighbors),
        "hotspot_fraction": float(hotspot_fraction),
        "min_spots": int(min_spots),
        "n_slide_signature_pairs": int(len(by_slide)),
        "n_signatures": int(signature_summary["signature"].nunique()) if not signature_summary.empty else 0,
        "n_slides": int(by_slide["sample_id"].nunique()) if not by_slide.empty else 0,
        "overall_mean_spatial_lag_pearson": (
            float(overall.iloc[0]["mean_spatial_lag_pearson"]) if not overall.empty else float("nan")
        ),
        "overall_mean_hotspot_jaccard": float(overall.iloc[0]["mean_hotspot_jaccard"]) if not overall.empty else float("nan"),
        "outputs": {
            "spatial_signature_by_slide": rel_project_path(by_slide_path),
            "spatial_signature_summary": rel_project_path(signature_summary_path),
            "spatial_signature_overall": rel_project_path(overall_path),
        },
    }
    write_json(out_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate spatial fidelity of biological signature maps.")
    parser.add_argument("--signature-dir", default=DEFAULT_SIGNATURE_DIR)
    parser.add_argument("--expression-config", default=DEFAULT_EXPRESSION_CONFIG)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-kind", choices=["rate", "count_pred_sf"], default="rate")
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--hotspot-fraction", type=float, default=0.2)
    parser.add_argument("--min-spots", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signature_dir = resolve_project_path(args.signature_dir)
    expression_config_path = resolve_project_path(args.expression_config)
    out_dir = resolve_project_path(args.out_dir)
    if signature_dir is None or expression_config_path is None or out_dir is None:
        raise ValueError("signature-dir, expression-config, and out-dir must resolve to paths.")
    evaluate_spatial_signature_fidelity(
        signature_dir=signature_dir,
        expression_config=load_config(expression_config_path),
        expression_config_path=expression_config_path,
        out_dir=out_dir,
        score_kind=str(args.score_kind),
        k_neighbors=int(args.k_neighbors),
        hotspot_fraction=float(args.hotspot_fraction),
        min_spots=int(args.min_spots),
    )


if __name__ == "__main__":
    main()
