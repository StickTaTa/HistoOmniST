from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.external.hist_sourcefaithful import (  # noqa: E402
    DEFAULT_HIST_UPSTREAM_ROOT,
    evaluate_hist_predictions,
    export_hist_sourcefaithful_predictions,
    extract_hist_ctranspath_features_from_patch_h5,
    prepare_hist_sourcefaithful_data,
    select_manifest_rows,
    train_hist_sourcefaithful,
)
from histoomnist.utils.config import load_config  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def _none_if_empty(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    values = [str(x) for x in values if str(x).strip()]
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-faithful HiST CMUNet benchmark on HEST coverage95 log1p_rate targets."
    )
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--upstream-root", default=str(DEFAULT_HIST_UPSTREAM_ROOT).replace("\\", "/"))
    parser.add_argument("--raw-root", default="data/HEST-1k/raw")
    parser.add_argument(
        "--data-dir",
        default="results/hest1k_human_visium_expression/benchmark_inputs/hist_sourcefaithful_full_fp32",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/hest1k_human_visium_expression_external/hist_sourcefaithful_full_fp32",
    )
    parser.add_argument(
        "--prediction-root",
        default="results/hest1k_human_visium_expression/external_baselines/hist_sourcefaithful_full_fp32_predictions",
    )
    parser.add_argument(
        "--benchmark-out-dir",
        default="results/hest1k_human_visium_expression/benchmark_results/hist_sourcefaithful_full_fp32",
    )
    parser.add_argument("--train-splits", nargs="*", default=["train"])
    parser.add_argument("--val-splits", nargs="*", default=["val"])
    parser.add_argument("--test-splits", nargs="*", default=["test"])
    parser.add_argument("--train-slide-ids", nargs="*", default=None)
    parser.add_argument("--val-slide-ids", nargs="*", default=None)
    parser.add_argument("--test-slide-ids", nargs="*", default=None)
    parser.add_argument("--max-train-slides", type=int, default=None)
    parser.add_argument("--max-val-slides", type=int, default=None)
    parser.add_argument("--max-test-slides", type=int, default=None)
    parser.add_argument("--smallest-train-slides", action="store_true")
    parser.add_argument("--smallest-val-slides", action="store_true")
    parser.add_argument("--smallest-test-slides", action="store_true")
    parser.add_argument("--max-train-slide-spots", type=int, default=None)
    parser.add_argument("--max-val-slide-spots", type=int, default=None)
    parser.add_argument("--max-test-slide-spots", type=int, default=None)
    parser.add_argument("--target-gene-limit", type=int, default=None)
    parser.add_argument("--grid-policy", default="scale_overflow", choices=["scale_overflow", "native_only"])
    parser.add_argument("--model-weight-path", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ctranspath-batch-size", type=int, default=80)
    parser.add_argument("--best-warmup-epochs", type=int, default=20)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--skip-targets", action="store_true")
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--evaluate-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_project_path(args.expression_config)
    if cfg_path is None:
        raise ValueError("Expression config resolved to None")
    cfg = load_config(cfg_path)
    upstream_root = resolve_project_path(args.upstream_root)
    raw_root = resolve_project_path(args.raw_root)
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    prediction_root = resolve_project_path(args.prediction_root)
    benchmark_out_dir = resolve_project_path(args.benchmark_out_dir)
    if any(x is None for x in [upstream_root, raw_root, data_dir, output_dir, prediction_root, benchmark_out_dir]):
        raise ValueError("One or more paths resolved to None")

    train_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.train_splits],
        slide_ids=_none_if_empty(args.train_slide_ids),
        max_slides=args.max_train_slides,
        smallest_slides=bool(args.smallest_train_slides),
        max_slide_spots=args.max_train_slide_spots,
    )
    val_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.val_splits],
        slide_ids=_none_if_empty(args.val_slide_ids),
        max_slides=args.max_val_slides,
        smallest_slides=bool(args.smallest_val_slides),
        max_slide_spots=args.max_val_slide_spots,
    )
    test_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.test_splits],
        slide_ids=_none_if_empty(args.test_slide_ids),
        max_slides=args.max_test_slides,
        smallest_slides=bool(args.smallest_test_slides),
        max_slide_spots=args.max_test_slide_spots,
    )
    prepare_rows = pd.concat([train_rows, val_rows], ignore_index=True)
    if args.prepare_data or args.export_predictions or args.evaluate_predictions or args.extract_features:
        prepare_rows = pd.concat([prepare_rows, test_rows], ignore_index=True)
    prepare_rows = prepare_rows.drop_duplicates("sample_id").reset_index(drop=True)

    summaries: dict[str, object] = {
        "selected": {
            "train_slides": [str(x) for x in train_rows["sample_id"].tolist()],
            "val_slides": [str(x) for x in val_rows["sample_id"].tolist()],
            "test_slides": [str(x) for x in test_rows["sample_id"].tolist()],
        }
    }
    if args.prepare_data:
        summaries["prepare"] = prepare_hist_sourcefaithful_data(
            expression_config=cfg,
            rows=prepare_rows,
            data_dir=data_dir,
            target_gene_limit=args.target_gene_limit,
            grid_policy=args.grid_policy,
            write_targets=not bool(args.skip_targets),
        )
    if args.extract_features:
        summaries["features"] = extract_hist_ctranspath_features_from_patch_h5(
            expression_config=cfg,
            rows=prepare_rows,
            data_dir=data_dir,
            raw_root=raw_root,
            upstream_root=upstream_root,
            model_weight_path=args.model_weight_path,
            batch_size=int(args.ctranspath_batch_size),
            device_name=args.device,
        )
    train_summary = None
    if args.train:
        train_summary = train_hist_sourcefaithful(
            expression_config=cfg,
            data_dir=data_dir,
            output_dir=output_dir,
            train_rows=train_rows,
            val_rows=val_rows,
            upstream_root=upstream_root,
            target_gene_limit=args.target_gene_limit,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed),
            device_name=args.device,
            best_warmup_epochs=int(args.best_warmup_epochs),
        )
        summaries["train"] = train_summary
    if args.export_predictions:
        checkpoint = train_summary["checkpoint"] if train_summary is not None else str(Path(output_dir) / "best_model.pt")
        summaries["predictions"] = export_hist_sourcefaithful_predictions(
            expression_config=cfg,
            checkpoint_path=checkpoint,
            data_dir=data_dir,
            out_dir=prediction_root,
            test_rows=test_rows,
            upstream_root=upstream_root,
            target_gene_limit=args.target_gene_limit,
            device_name=args.device,
        )
    if args.evaluate_predictions:
        summaries["benchmark"] = evaluate_hist_predictions(
            expression_config=cfg,
            prediction_root=prediction_root,
            out_dir=benchmark_out_dir,
            splits=[str(x) for x in args.test_splits],
            slide_ids=_none_if_empty(args.test_slide_ids),
            max_slides=args.max_test_slides,
            max_slide_spots=args.max_test_slide_spots,
        )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
