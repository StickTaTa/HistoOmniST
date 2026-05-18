from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from histoomnist.eval.biological_signatures import build_coords_for_dataset_order
from histoomnist.utils.config import load_config
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import project_root, resolve_project_path


DEFAULT_SIGNATURE_DIR = "results/hest1k_human_visium_expression/biological_signatures"
DEFAULT_EXPRESSION_CONFIG = "configs/hest1k_human_visium_expression_highconf_symbol95.yaml"
DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/biological_signature_states"


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


def load_score_matrices(signature_dir: Path, *, score_kind: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    path = signature_dir / "spot_signature_scores.csv"
    scores = pd.read_csv(path)
    true_col, pred_col, valid_col = score_columns(score_kind)
    required = {"row_index", "sample_id", "signature", true_col, pred_col, valid_col}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Spot signature score table missing columns: {missing}")

    meta = scores[["row_index", "sample_id"]].drop_duplicates("row_index").sort_values("row_index")
    signatures = sorted(scores["signature"].dropna().astype(str).unique())
    true = scores.pivot(index="row_index", columns="signature", values=true_col).reindex(meta["row_index"])
    pred = scores.pivot(index="row_index", columns="signature", values=pred_col).reindex(meta["row_index"])
    valid = scores.pivot(index="row_index", columns="signature", values=valid_col).reindex(meta["row_index"])
    true = true.reindex(columns=signatures)
    pred = pred.reindex(columns=signatures)
    valid = valid.reindex(columns=signatures)
    keep = (
        valid.fillna(False).astype(bool).all(axis=1).to_numpy()
        & np.isfinite(true.to_numpy(dtype=np.float32)).all(axis=1)
        & np.isfinite(pred.to_numpy(dtype=np.float32)).all(axis=1)
    )
    meta = meta.loc[keep].reset_index(drop=True)
    true_matrix = true.to_numpy(dtype=np.float32)[keep]
    pred_matrix = pred.to_numpy(dtype=np.float32)[keep]
    return meta, true_matrix, pred_matrix, signatures


def zscore_pair(true_matrix: np.ndarray, pred_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = true_matrix.mean(axis=0, keepdims=True)
    std = true_matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (true_matrix - mean) / std, (pred_matrix - mean) / std, mean.reshape(-1), std.reshape(-1)


def assign_to_centers(x: np.ndarray, centers: np.ndarray, *, chunk_size: int = 16384) -> np.ndarray:
    labels = np.empty(x.shape[0], dtype=np.int32)
    center_norm = np.sum(centers * centers, axis=1)[None, :]
    for start in range(0, x.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), x.shape[0])
        chunk = x[start:stop]
        distances = np.sum(chunk * chunk, axis=1)[:, None] + center_norm - 2.0 * (chunk @ centers.T)
        labels[start:stop] = np.argmin(distances, axis=1).astype(np.int32)
    return labels


def fit_kmeans_numpy(
    x: np.ndarray,
    *,
    n_states: int,
    seed: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] < int(n_states):
        raise ValueError(f"Need at least {n_states} rows to fit states, got {x.shape[0]}")
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(x.shape[0], size=int(n_states), replace=False)].astype(np.float32, copy=True)
    labels = np.zeros(x.shape[0], dtype=np.int32)
    for _ in range(int(max_iter)):
        labels = assign_to_centers(x, centers)
        new_centers = centers.copy()
        for state in range(int(n_states)):
            idx = labels == state
            if np.any(idx):
                new_centers[state] = x[idx].mean(axis=0)
            else:
                new_centers[state] = x[int(rng.integers(0, x.shape[0]))]
        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        if shift < float(tol):
            break
    labels = assign_to_centers(x, centers)
    return centers, labels


def best_match_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray, n_states: int) -> tuple[float, np.ndarray, dict[int, int]]:
    confusion = np.zeros((n_states, n_states), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels):
        if 0 <= int(true_label) < n_states and 0 <= int(pred_label) < n_states:
            confusion[int(true_label), int(pred_label)] += 1
    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    matched = int(confusion[row_ind, col_ind].sum())
    mapping = {int(pred): int(true) for true, pred in zip(row_ind, col_ind)}
    accuracy = float(matched / max(len(true_labels), 1))
    return accuracy, confusion, mapping


def cluster_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, n_states: int) -> dict[str, float | int]:
    if len(true_labels) == 0:
        return {
            "n_spots": 0,
            "adjusted_rand": float("nan"),
            "normalized_mutual_info": float("nan"),
            "best_match_accuracy": float("nan"),
        }
    acc, _, _ = best_match_accuracy(true_labels, pred_labels, n_states=n_states)
    return {
        "n_spots": int(len(true_labels)),
        "adjusted_rand": float(adjusted_rand_score(true_labels, pred_labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "best_match_accuracy": acc,
    }


def group_metrics(meta: pd.DataFrame, true_labels: np.ndarray, pred_labels: np.ndarray, *, group_col: str, n_states: int) -> pd.DataFrame:
    rows = []
    for group, idx in meta.groupby(group_col, sort=True).indices.items():
        metrics = cluster_metrics(true_labels[idx], pred_labels[idx], n_states=n_states)
        rows.append({group_col: group, **metrics})
    return pd.DataFrame(rows)


def state_annotation(
    true_scaled: np.ndarray,
    pred_scaled: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    signatures: list[str],
    *,
    n_states: int,
) -> pd.DataFrame:
    rows = []
    for state in range(n_states):
        true_mask = true_labels == state
        pred_mask = pred_labels == state
        centroid = true_scaled[true_mask].mean(axis=0) if np.any(true_mask) else np.full(len(signatures), np.nan)
        pred_centroid = pred_scaled[pred_mask].mean(axis=0) if np.any(pred_mask) else np.full(len(signatures), np.nan)
        order = np.argsort(np.nan_to_num(centroid, nan=-np.inf))[::-1]
        rows.append(
            {
                "state": int(state),
                "n_true_spots": int(true_mask.sum()),
                "n_pred_spots": int(pred_mask.sum()),
                "top_true_signatures": "|".join(signatures[i] for i in order[:3]),
                "top_true_z": "|".join(f"{float(centroid[i]):.3f}" for i in order[:3]),
                "true_centroid_json": json.dumps({sig: float(centroid[i]) for i, sig in enumerate(signatures)}),
                "pred_centroid_json": json.dumps({sig: float(pred_centroid[i]) for i, sig in enumerate(signatures)}),
            }
        )
    return pd.DataFrame(rows)


def add_sample_metadata(meta: pd.DataFrame, expression_config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    manifest_path = Path(expression_config["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    split_names = list(expression_config["data"].get("test_splits", ["test"]))
    sample_info = manifest.drop_duplicates("sample_id").set_index("sample_id")
    out = meta.copy()
    for column in ["organ", "cohort", "split"]:
        if column in sample_info.columns:
            out[column] = out["sample_id"].map(sample_info[column].astype(str).to_dict()).fillna("unknown")
        else:
            out[column] = "unknown"
    coords = build_coords_for_dataset_order(
        manifest=manifest,
        manifest_base=manifest_path.parent,
        splits=split_names,
        min_total_counts=float(expression_config["data"].get("min_total_counts", 1.0)),
    )
    coords = coords[meta["row_index"].to_numpy(dtype=np.int64)]
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    return out, coords


def plot_state_map(
    *,
    sample_id: str,
    coords: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    out_path: Path,
    n_states: int,
) -> bool:
    keep = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if keep.sum() < 3:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20", n_states)
    x = coords[keep, 0]
    y = coords[keep, 1]
    true = true_labels[keep]
    pred = pred_labels[keep]
    match = (true == pred).astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=160)
    for ax, values, title, cm, vmin, vmax in [
        (axes[0], true, "measured state", cmap, -0.5, n_states - 0.5),
        (axes[1], pred, "predicted state", cmap, -0.5, n_states - 0.5),
        (axes[2], match, "same label", "coolwarm", 0.0, 1.0),
    ]:
        sc = ax.scatter(x, y, c=values, s=7, cmap=cm, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{sample_id} signature-derived state map", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def evaluate_signature_states(
    *,
    signature_dir: Path,
    expression_config: dict[str, Any],
    out_dir: Path,
    score_kind: str,
    n_states: int,
    max_fit_spots: int,
    seed: int,
    max_overlay_slides: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, true_matrix, pred_matrix, signatures = load_score_matrices(signature_dir, score_kind=score_kind)
    meta, coords = add_sample_metadata(meta, expression_config)
    true_scaled, pred_scaled, mean, std = zscore_pair(true_matrix, pred_matrix)

    rng = np.random.default_rng(seed)
    if len(true_scaled) > int(max_fit_spots):
        fit_idx = rng.choice(len(true_scaled), size=int(max_fit_spots), replace=False)
    else:
        fit_idx = np.arange(len(true_scaled))
    centers, _ = fit_kmeans_numpy(
        true_scaled[fit_idx].astype(np.float32, copy=False),
        n_states=int(n_states),
        seed=int(seed),
    )
    true_labels = assign_to_centers(true_scaled.astype(np.float32, copy=False), centers)
    pred_labels = assign_to_centers(pred_scaled.astype(np.float32, copy=False), centers)
    true_label_column = "true_state"
    pred_label_column = "pred_state"
    meta[true_label_column] = true_labels
    meta[pred_label_column] = pred_labels
    meta["same_state_label"] = true_labels == pred_labels

    overall = cluster_metrics(true_labels, pred_labels, n_states=int(n_states))
    overall["score_kind"] = score_kind
    overall["n_states"] = int(n_states)
    overall["n_signatures"] = int(len(signatures))
    overall["n_fit_spots"] = int(len(fit_idx))
    overall_frame = pd.DataFrame([overall])
    by_slide = group_metrics(meta, true_labels, pred_labels, group_col="sample_id", n_states=int(n_states))
    by_slide = by_slide.merge(meta[["sample_id", "organ", "cohort"]].drop_duplicates("sample_id"), on="sample_id", how="left")
    by_organ = group_metrics(meta, true_labels, pred_labels, group_col="organ", n_states=int(n_states))
    annotations = state_annotation(
        true_scaled,
        pred_scaled,
        true_labels,
        pred_labels,
        signatures,
        n_states=int(n_states),
    )
    accuracy, confusion, mapping = best_match_accuracy(true_labels, pred_labels, n_states=int(n_states))
    confusion_frame = pd.DataFrame(confusion)
    confusion_frame.insert(0, "true_state", np.arange(int(n_states), dtype=int))

    spot_assignments_path = out_dir / "spot_state_assignments.csv"
    meta.to_csv(spot_assignments_path, index=False)
    overall_frame.to_csv(out_dir / "state_fidelity_overall.csv", index=False)
    by_slide.to_csv(out_dir / "state_fidelity_by_slide.csv", index=False)
    by_organ.to_csv(out_dir / "state_fidelity_by_organ.csv", index=False)
    annotations.to_csv(out_dir / "state_annotations.csv", index=False)
    confusion_frame.to_csv(out_dir / "state_confusion_matrix.csv", index=False)
    pd.DataFrame(
        {
            "signature": signatures,
            "true_mean": mean.astype(float),
            "true_std": std.astype(float),
        }
    ).to_csv(out_dir / "signature_scaling.csv", index=False)

    overlay_rows = []
    for sample_id in list(dict.fromkeys(meta["sample_id"].astype(str).tolist()))[: int(max_overlay_slides)]:
        sample_mask = meta["sample_id"].astype(str).eq(sample_id).to_numpy()
        out_path = out_dir / "spatial_state_maps" / f"{sample_id}_{score_kind}_states.png"
        ok = plot_state_map(
            sample_id=sample_id,
            coords=coords[sample_mask],
            true_labels=true_labels[sample_mask],
            pred_labels=pred_labels[sample_mask],
            out_path=out_path,
            n_states=int(n_states),
        )
        overlay_rows.append({"sample_id": sample_id, "path": rel_project_path(out_path), "written": bool(ok)})
    overlay_manifest = pd.DataFrame(overlay_rows)
    overlay_manifest.to_csv(out_dir / "spatial_state_map_manifest.csv", index=False)

    summary = {
        "signature_dir": rel_project_path(signature_dir),
        "score_kind": score_kind,
        "n_spots": int(len(meta)),
        "n_signatures": int(len(signatures)),
        "n_states": int(n_states),
        "n_fit_spots": int(len(fit_idx)),
        "adjusted_rand": float(overall["adjusted_rand"]),
        "normalized_mutual_info": float(overall["normalized_mutual_info"]),
        "best_match_accuracy": float(accuracy),
        "state_mapping_pred_to_true": {str(k): int(v) for k, v in mapping.items()},
        "outputs": {
            "state_fidelity_overall": rel_project_path(out_dir / "state_fidelity_overall.csv"),
            "state_fidelity_by_slide": rel_project_path(out_dir / "state_fidelity_by_slide.csv"),
            "state_fidelity_by_organ": rel_project_path(out_dir / "state_fidelity_by_organ.csv"),
            "state_annotations": rel_project_path(out_dir / "state_annotations.csv"),
            "state_confusion_matrix": rel_project_path(out_dir / "state_confusion_matrix.csv"),
            "spot_state_assignments": rel_project_path(spot_assignments_path),
            "spatial_state_map_manifest": rel_project_path(out_dir / "spatial_state_map_manifest.csv"),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate signature-derived spatial state fidelity.")
    parser.add_argument("--signature-dir", default=DEFAULT_SIGNATURE_DIR)
    parser.add_argument("--expression-config", default=DEFAULT_EXPRESSION_CONFIG)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-kind", choices=["rate", "count_pred_sf"], default="rate")
    parser.add_argument("--n-states", type=int, default=6)
    parser.add_argument("--max-fit-spots", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-overlay-slides", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signature_dir = resolve_project_path(args.signature_dir)
    expression_config_path = resolve_project_path(args.expression_config)
    out_dir = resolve_project_path(args.out_dir)
    if signature_dir is None or expression_config_path is None or out_dir is None:
        raise ValueError("signature-dir, expression-config, and out-dir must resolve to paths.")
    evaluate_signature_states(
        signature_dir=signature_dir,
        expression_config=load_config(expression_config_path),
        out_dir=out_dir,
        score_kind=str(args.score_kind),
        n_states=int(args.n_states),
        max_fit_spots=int(args.max_fit_spots),
        seed=int(args.seed),
        max_overlay_slides=int(args.max_overlay_slides),
    )


if __name__ == "__main__":
    main()
