from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from histoomnist.eval.biological_signatures import DEFAULT_EXPRESSION_CONFIG, rel_project_path
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


DEFAULT_SIGNATURE_SCORE_PATH = "results/hest1k_human_visium_expression/biological_signatures/spot_signature_scores.csv"
DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/metadata_stratification"
DEFAULT_LABEL_FIELDS = ["organ", "disease_state", "tissue", "oncotree_code"]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clean_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "not provided"}:
        return ""
    return text


def finite_pearson(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return float("nan")
    xk = x[keep].astype(np.float64)
    yk = y[keep].astype(np.float64)
    if float(np.std(xk)) == 0.0 or float(np.std(yk)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xk, yk)[0, 1])


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    keep = np.isfinite(values)
    values = values[keep].astype(np.float64)
    labels = labels[keep]
    if values.size < 3 or len(np.unique(labels)) < 2:
        return float("nan")
    grand = float(values.mean())
    total_ss = float(np.sum((values - grand) ** 2))
    if total_ss <= 0:
        return float("nan")
    between = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        if group.size:
            between += float(group.size) * float((group.mean() - grand) ** 2)
    return float(between / total_ss)


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    group_a = group_a[np.isfinite(group_a)].astype(np.float64)
    group_b = group_b[np.isfinite(group_b)].astype(np.float64)
    if group_a.size < 2 or group_b.size < 2:
        return float("nan")
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled = math.sqrt(((group_a.size - 1) * var_a + (group_b.size - 1) * var_b) / (group_a.size + group_b.size - 2))
    if pooled <= 0 or not math.isfinite(pooled):
        return float("nan")
    return float((group_a.mean() - group_b.mean()) / pooled)


def topk_jaccard(true_values: pd.Series, pred_values: pd.Series, k: int) -> float:
    aligned = pd.concat([true_values, pred_values], axis=1, keys=["true", "pred"]).dropna()
    if aligned.empty:
        return float("nan")
    k = min(int(k), int(len(aligned)))
    if k <= 0:
        return float("nan")
    true_top = set(aligned.sort_values("true", ascending=False).head(k).index)
    pred_top = set(aligned.sort_values("pred", ascending=False).head(k).index)
    union = true_top | pred_top
    return float(len(true_top & pred_top) / len(union)) if union else float("nan")


def load_slide_signature_table(
    *,
    signature_score_path: Path,
    metadata_csv: Path,
    label_fields: list[str],
    score_kind: str,
) -> pd.DataFrame:
    scores = pd.read_csv(signature_score_path)
    required = {"sample_id", "signature", f"{score_kind}_true", f"{score_kind}_pred", f"{score_kind}_valid"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Signature score table missing columns for score_kind={score_kind}: {missing}")
    valid = scores[f"{score_kind}_valid"].astype(str).str.lower().isin(["true", "1"])
    scores = scores.loc[valid, ["sample_id", "signature", f"{score_kind}_true", f"{score_kind}_pred"]].copy()
    slide = (
        scores.groupby(["sample_id", "signature"], sort=True)
        .agg(
            true_mean=(f"{score_kind}_true", "mean"),
            pred_mean=(f"{score_kind}_pred", "mean"),
            n_spots=(f"{score_kind}_true", "size"),
        )
        .reset_index()
    )
    metadata = pd.read_csv(metadata_csv)
    if "id" not in metadata.columns:
        raise ValueError(f"HEST metadata is missing id column: {metadata_csv}")
    keep_columns = ["id"] + [field for field in label_fields if field in metadata.columns]
    metadata = metadata.loc[:, keep_columns].rename(columns={"id": "sample_id"})
    for field in label_fields:
        if field in metadata.columns:
            metadata[field] = metadata[field].map(clean_label)
    return slide.merge(metadata, on="sample_id", how="left")


def valid_groups(frame: pd.DataFrame, field: str, min_group_slides: int) -> tuple[pd.DataFrame, list[str]]:
    if field not in frame.columns:
        return frame.iloc[0:0].copy(), []
    labels = frame[["sample_id", field]].drop_duplicates()
    labels = labels[labels[field].map(clean_label).ne("")]
    counts = labels.groupby(field)["sample_id"].nunique()
    groups = counts[counts >= int(min_group_slides)].sort_index().index.astype(str).tolist()
    if len(groups) < 2:
        return frame.iloc[0:0].copy(), []
    return frame[frame[field].astype(str).isin(groups)].copy(), groups


def group_profile_rows(frame: pd.DataFrame, fields: list[str], min_group_slides: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in fields:
        field_frame, groups = valid_groups(frame, field, min_group_slides)
        if not groups:
            continue
        for (group, signature), sub in field_frame.groupby([field, "signature"], sort=True):
            rows.append(
                {
                    "label_field": field,
                    "label_value": str(group),
                    "signature": str(signature),
                    "n_slides": int(sub["sample_id"].nunique()),
                    "n_spots": int(sub["n_spots"].sum()),
                    "true_mean": float(sub["true_mean"].mean()),
                    "pred_mean": float(sub["pred_mean"].mean()),
                    "true_std": float(sub["true_mean"].std(ddof=0)),
                    "pred_std": float(sub["pred_mean"].std(ddof=0)),
                }
            )
    return pd.DataFrame(rows)


def signature_effect_rows(frame: pd.DataFrame, fields: list[str], min_group_slides: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in fields:
        field_frame, groups = valid_groups(frame, field, min_group_slides)
        if not groups:
            continue
        for signature, sub in field_frame.groupby("signature", sort=True):
            labels = sub[field].astype(str).to_numpy()
            true = sub["true_mean"].to_numpy(dtype=np.float64)
            pred = sub["pred_mean"].to_numpy(dtype=np.float64)
            true_group_means = sub.groupby(field)["true_mean"].mean()
            pred_group_means = sub.groupby(field)["pred_mean"].mean()
            true_top = str(true_group_means.sort_values(ascending=False).index[0])
            pred_top = str(pred_group_means.sort_values(ascending=False).index[0])
            rows.append(
                {
                    "label_field": field,
                    "signature": str(signature),
                    "n_slides": int(sub["sample_id"].nunique()),
                    "n_groups": int(len(groups)),
                    "groups": "|".join(groups),
                    "true_eta_squared": eta_squared(true, labels),
                    "pred_eta_squared": eta_squared(pred, labels),
                    "abs_eta_error": abs(eta_squared(pred, labels) - eta_squared(true, labels)),
                    "true_top_group": true_top,
                    "pred_top_group": pred_top,
                    "top_group_match": bool(true_top == pred_top),
                    "true_group_range": float(true_group_means.max() - true_group_means.min()),
                    "pred_group_range": float(pred_group_means.max() - pred_group_means.min()),
                }
            )
    return pd.DataFrame(rows)


def contrast_rows(frame: pd.DataFrame, fields: list[str], min_group_slides: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in fields:
        field_frame, groups = valid_groups(frame, field, min_group_slides)
        if not groups:
            continue
        for group_a, group_b in combinations(groups, 2):
            pair_frame = field_frame[field_frame[field].astype(str).isin([group_a, group_b])]
            for signature, sub in pair_frame.groupby("signature", sort=True):
                a = sub[sub[field].astype(str).eq(group_a)]
                b = sub[sub[field].astype(str).eq(group_b)]
                true_d = cohen_d(a["true_mean"].to_numpy(dtype=np.float64), b["true_mean"].to_numpy(dtype=np.float64))
                pred_d = cohen_d(a["pred_mean"].to_numpy(dtype=np.float64), b["pred_mean"].to_numpy(dtype=np.float64))
                rows.append(
                    {
                        "label_field": field,
                        "group_a": group_a,
                        "group_b": group_b,
                        "signature": str(signature),
                        "n_a": int(a["sample_id"].nunique()),
                        "n_b": int(b["sample_id"].nunique()),
                        "true_cohen_d": true_d,
                        "pred_cohen_d": pred_d,
                        "abs_contrast_error": abs(pred_d - true_d) if np.isfinite(true_d) and np.isfinite(pred_d) else float("nan"),
                        "same_direction": bool(np.isfinite(true_d) and np.isfinite(pred_d) and np.sign(true_d) == np.sign(pred_d)),
                    }
                )
    return pd.DataFrame(rows)


def summary_rows(effects: pd.DataFrame, contrasts: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if effects.empty:
        return pd.DataFrame()
    for field, group in effects.groupby("label_field", sort=True):
        true_eta = group.set_index("signature")["true_eta_squared"]
        pred_eta = group.set_index("signature")["pred_eta_squared"]
        field_contrasts = contrasts[contrasts["label_field"].astype(str).eq(str(field))] if not contrasts.empty else pd.DataFrame()
        rows.append(
            {
                "label_field": str(field),
                "n_signatures": int(len(group)),
                "n_slides": int(group["n_slides"].max()),
                "n_groups": int(group["n_groups"].max()),
                "groups": str(group["groups"].iloc[0]),
                "eta_pearson": finite_pearson(true_eta.to_numpy(dtype=np.float64), pred_eta.to_numpy(dtype=np.float64)),
                "mean_abs_eta_error": float(group["abs_eta_error"].mean()),
                "top_group_match_rate": float(group["top_group_match"].mean()),
                "top_signature_eta_jaccard": topk_jaccard(true_eta, pred_eta, top_k),
                "contrast_pearson": finite_pearson(
                    field_contrasts["true_cohen_d"].to_numpy(dtype=np.float64)
                    if not field_contrasts.empty
                    else np.asarray([], dtype=np.float64),
                    field_contrasts["pred_cohen_d"].to_numpy(dtype=np.float64)
                    if not field_contrasts.empty
                    else np.asarray([], dtype=np.float64),
                ),
                "contrast_same_direction_rate": float(field_contrasts["same_direction"].mean())
                if not field_contrasts.empty
                else float("nan"),
                "n_contrasts": int(len(field_contrasts)),
            }
        )
    return pd.DataFrame(rows).sort_values("eta_pearson", ascending=False, na_position="last").reset_index(drop=True)


def evaluate_metadata_stratification(
    *,
    expression_config: dict[str, Any],
    expression_config_path: Path | None,
    signature_score_path: Path,
    out_dir: Path,
    label_fields: list[str],
    score_kind: str,
    min_group_slides: int,
    top_k: int,
) -> dict[str, Any]:
    metadata_csv = resolve_project_path(expression_config["paths"]["metadata_csv"])
    if metadata_csv is None:
        raise ValueError("paths.metadata_csv did not resolve")
    out_dir.mkdir(parents=True, exist_ok=True)
    slide = load_slide_signature_table(
        signature_score_path=signature_score_path,
        metadata_csv=metadata_csv,
        label_fields=label_fields,
        score_kind=score_kind,
    )
    slide.to_csv(out_dir / "slide_signature_metadata.csv", index=False)
    profiles = group_profile_rows(slide, label_fields, min_group_slides)
    effects = signature_effect_rows(slide, label_fields, min_group_slides)
    contrasts = contrast_rows(slide, label_fields, min_group_slides)
    summary = summary_rows(effects, contrasts, top_k=top_k)
    profiles.to_csv(out_dir / "metadata_group_profiles.csv", index=False)
    effects.to_csv(out_dir / "metadata_signature_effects.csv", index=False)
    contrasts.to_csv(out_dir / "metadata_group_contrasts.csv", index=False)
    summary.to_csv(out_dir / "metadata_stratification_summary.csv", index=False)
    run_summary = {
        "expression_config": rel_project_path(expression_config_path or DEFAULT_EXPRESSION_CONFIG),
        "signature_score_path": rel_project_path(signature_score_path),
        "metadata_csv": rel_project_path(metadata_csv),
        "score_kind": score_kind,
        "label_fields_requested": label_fields,
        "label_fields_evaluated": summary["label_field"].astype(str).tolist() if not summary.empty else [],
        "min_group_slides": int(min_group_slides),
        "top_k": int(top_k),
        "n_slide_signature_rows": int(len(slide)),
        "n_unique_slides": int(slide["sample_id"].nunique()),
        "n_signatures": int(slide["signature"].nunique()),
        "is_metadata_anchored": True,
        "has_spot_level_pathology_annotation": False,
        "outputs": {
            "slide_signature_metadata": rel_project_path(out_dir / "slide_signature_metadata.csv"),
            "metadata_group_profiles": rel_project_path(out_dir / "metadata_group_profiles.csv"),
            "metadata_signature_effects": rel_project_path(out_dir / "metadata_signature_effects.csv"),
            "metadata_group_contrasts": rel_project_path(out_dir / "metadata_group_contrasts.csv"),
            "metadata_stratification_summary": rel_project_path(out_dir / "metadata_stratification_summary.csv"),
        },
    }
    write_json(out_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate metadata-anchored slide-level biological stratification.")
    parser.add_argument("--expression-config", default=DEFAULT_EXPRESSION_CONFIG)
    parser.add_argument("--signature-score-path", default=DEFAULT_SIGNATURE_SCORE_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--label-fields", nargs="*", default=DEFAULT_LABEL_FIELDS)
    parser.add_argument("--score-kind", choices=["rate", "count_pred_sf"], default="rate")
    parser.add_argument("--min-group-slides", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_config_path = resolve_project_path(args.expression_config)
    signature_score_path = resolve_project_path(args.signature_score_path)
    out_dir = resolve_project_path(args.out_dir)
    if None in (expression_config_path, signature_score_path, out_dir):
        raise ValueError("Required paths did not resolve.")
    evaluate_metadata_stratification(
        expression_config=load_config(expression_config_path),
        expression_config_path=expression_config_path,
        signature_score_path=signature_score_path,
        out_dir=out_dir,
        label_fields=[str(field) for field in args.label_fields],
        score_kind=str(args.score_kind),
        min_group_slides=int(args.min_group_slides),
        top_k=int(args.top_k),
    )


if __name__ == "__main__":
    main()
