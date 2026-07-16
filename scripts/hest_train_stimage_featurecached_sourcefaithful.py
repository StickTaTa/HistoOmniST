from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.external.stimage_featurecached import (  # noqa: E402
    evaluate_featurecached_predictions,
    export_featurecached_predictions,
    select_manifest_rows,
    train_featurecached_stimage,
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
        description="Run source-equivalent feature-cached STimage benchmark on HEST coverage95 log1p_rate targets."
    )
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/hest1k_human_visium_expression_external/stimage_sourcefaithful_featurecached_full_fp32",
    )
    parser.add_argument(
        "--feature-root",
        default="results/hest1k_human_visium_expression/benchmark_inputs/stimage_sourcefaithful_featurecached_full_fp32",
    )
    parser.add_argument(
        "--prediction-root",
        default="results/hest1k_human_visium_expression/external_baselines/stimage_sourcefaithful_featurecached_full_fp32_predictions",
    )
    parser.add_argument(
        "--benchmark-out-dir",
        default="results/hest1k_human_visium_expression/benchmark_results/stimage_sourcefaithful_featurecached_full_fp32",
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument("--tile-size", type=int, default=299)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--evaluate-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_project_path(args.expression_config)
    if cfg_path is None:
        raise ValueError("Expression config resolved to None")
    cfg = load_config(cfg_path)
    output_dir = resolve_project_path(args.output_dir)
    feature_root = resolve_project_path(args.feature_root)
    prediction_root = resolve_project_path(args.prediction_root)
    benchmark_out_dir = resolve_project_path(args.benchmark_out_dir)
    if any(x is None for x in [output_dir, feature_root, prediction_root, benchmark_out_dir]):
        raise ValueError("One or more paths resolved to None")

    train_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.train_splits],
        slide_ids=_none_if_empty(args.train_slide_ids),
        max_slides=args.max_train_slides,
    )
    val_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.val_splits],
        slide_ids=_none_if_empty(args.val_slide_ids),
        max_slides=args.max_val_slides,
    )
    test_slide_ids = _none_if_empty(args.test_slide_ids)
    test_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.test_splits],
        slide_ids=test_slide_ids,
        max_slides=args.max_test_slides,
    )

    checkpoint_path = resolve_project_path(args.checkpoint_path) if args.checkpoint_path else None
    train_summary = None
    if checkpoint_path is None:
        train_summary = train_featurecached_stimage(
            expression_config=cfg,
            train_rows=train_rows,
            val_rows=val_rows,
            output_dir=output_dir,
            feature_root=feature_root,
            epochs=int(args.epochs),
            patience=int(args.patience),
            feature_batch_size=int(args.feature_batch_size),
            train_batch_size=int(args.train_batch_size),
            tile_size=int(args.tile_size),
            seed=int(args.seed),
        )
        checkpoint_path = Path(train_summary["outputs"]["best_head"])
        print(json.dumps({"train_summary": train_summary}, indent=2), flush=True)
    if args.export_predictions or args.evaluate_predictions:
        prediction_summary = export_featurecached_predictions(
            expression_config=cfg,
            test_rows=test_rows,
            checkpoint_path=checkpoint_path,
            prediction_root=prediction_root,
            feature_root=Path(feature_root) / "test",
            feature_batch_size=int(args.feature_batch_size),
            predict_batch_size=int(args.predict_batch_size),
            tile_size=int(args.tile_size),
        )
        print(json.dumps({"prediction_summary": prediction_summary}, indent=2), flush=True)
    if args.evaluate_predictions:
        benchmark_summary = evaluate_featurecached_predictions(
            prediction_root=prediction_root,
            benchmark_out_dir=benchmark_out_dir,
            expression_config=cfg,
            splits=[str(x) for x in args.test_splits],
            slide_ids=test_slide_ids,
        )
        print(json.dumps({"benchmark_summary": benchmark_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
