from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.data.gene_selection import gene_key_settings_from_config  # noqa: E402
from histoomnist.eval.benchmark_predictions import load_slide_target  # noqa: E402
from histoomnist.external.histogene_patch_h5 import (  # noqa: E402
    HistogenePatchH5Dataset,
    target_values_from_counts,
)
from histoomnist.utils.config import load_config  # noqa: E402
from histoomnist.utils.io import read_manifest  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the HEST patch-H5 adapter for HisToGene-style baselines.")
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--max-slides", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--target-kind", choices=["log1p_rate", "rate", "count", "log1p_count"], default="log1p_rate")
    parser.add_argument(
        "--out-dir",
        default="results/hest1k_human_visium_expression/external_baselines/histogene_patch_h5_adapter_smoke",
    )
    return parser.parse_args()


def tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def first_slide_consistency_check(cfg: dict, dataset: HistogenePatchH5Dataset, n_rows: int) -> dict[str, object]:
    manifest_path = resolve_project_path(cfg["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Manifest path resolved to None")
    manifest = read_manifest(manifest_path)
    first_slide = dataset.slides[0]
    row = manifest[manifest["sample_id"].astype(str).eq(first_slide.sample_id)].iloc[0]
    gene_key, raw_st_root = gene_key_settings_from_config(cfg)
    raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
    target = load_slide_target(
        row=row,
        base_dir=manifest_path.parent,
        target_genes=dataset.target_genes,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
        min_total_counts=float(cfg["data"].get("min_total_counts", 1.0)),
    )
    keep = min(int(n_rows), first_slide.n_spots, len(target.spot_ids))
    max_abs_diff = 0.0
    for local_idx in range(keep):
        adapter = target_values_from_counts(
            first_slide.counts.getrow(local_idx).toarray().reshape(-1),
            float(first_slide.size_factor[local_idx]),
            dataset.target_kind,
        )
        benchmark = target_values_from_counts(
            target.counts.getrow(local_idx).toarray().reshape(-1),
            float(target.size_factor[local_idx]),
            dataset.target_kind,
        )
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(adapter - benchmark))))
    return {
        "sample_id": first_slide.sample_id,
        "checked_rows": int(keep),
        "spot_ids_match": bool(first_slide.spot_ids[:keep] == target.spot_ids[:keep]),
        "measured_gene_mask_match": bool(np.array_equal(first_slide.measured_genes, target.measured_genes)),
        "max_abs_target_diff": float(max_abs_diff),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.expression_config))
    dataset = HistogenePatchH5Dataset(
        cfg,
        splits=[str(x) for x in args.splits],
        max_slides=args.max_slides,
        target_kind=args.target_kind,
    )
    out_dir = resolve_project_path(args.out_dir)
    if out_dir is None:
        raise ValueError("Output dir resolved to None")
    out_dir.mkdir(parents=True, exist_ok=True)

    item_rows = []
    patch_min = []
    patch_max = []
    target_finite = []
    n_items = min(int(args.max_items), len(dataset))
    for idx in range(n_items):
        item = dataset[idx]
        patch = tensor_to_numpy(item["patch"])
        target = tensor_to_numpy(item[args.target_kind])
        position = tensor_to_numpy(item["position"])
        mask = tensor_to_numpy(item["expression_mask"]).astype(bool)
        patch_min.append(float(np.min(patch)))
        patch_max.append(float(np.max(patch)))
        target_finite.append(float(np.mean(np.isfinite(target[mask]))))
        item_rows.append(
            {
                "index": int(idx),
                "sample_id": str(item["sample_id"]),
                "spot_id": str(item["spot_id"]),
                "patch_index": int(item["patch_index"]),
                "patch_shape": "x".join(str(x) for x in patch.shape),
                "patch_min": float(np.min(patch)),
                "patch_max": float(np.max(patch)),
                "position_min": float(np.min(position)),
                "position_max": float(np.max(position)),
                "target_dim": int(target.shape[0]),
                "measured_genes": int(mask.sum()),
                "target_finite_fraction_measured": float(np.mean(np.isfinite(target[mask]))),
                "target_mean_measured": float(np.mean(target[mask])),
            }
        )
    item_frame = pd.DataFrame(item_rows)
    slide_frame = dataset.slide_summary_frame()
    item_frame.to_csv(out_dir / "sample_items.csv", index=False)
    slide_frame.to_csv(out_dir / "slides.csv", index=False)
    consistency = first_slide_consistency_check(cfg, dataset, n_rows=max(n_items, 1))
    summary = {
        "expression_config": str(args.expression_config),
        "splits": [str(x) for x in args.splits],
        "max_slides": None if args.max_slides is None else int(args.max_slides),
        "target_kind": str(args.target_kind),
        "n_slides": int(len(dataset.slides)),
        "n_spots": int(len(dataset)),
        "n_target_genes": int(len(dataset.target_genes)),
        "n_items_checked": int(n_items),
        "patch_value_min": float(np.min(patch_min)) if patch_min else float("nan"),
        "patch_value_max": float(np.max(patch_max)) if patch_max else float("nan"),
        "all_checked_patch_values_in_0_1": bool(patch_min and min(patch_min) >= 0.0 and max(patch_max) <= 1.0),
        "all_checked_targets_finite_on_measured_genes": bool(target_finite and min(target_finite) == 1.0),
        "first_slide_consistency": consistency,
        "outputs": {
            "sample_items": str(out_dir / "sample_items.csv"),
            "slides": str(out_dir / "slides.csv"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
