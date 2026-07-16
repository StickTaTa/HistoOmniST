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

from histoomnist.external.path2space_sourcefaithful import (  # noqa: E402
    DEFAULT_CTRANSPATH_WEIGHT_PATH,
    DEFAULT_PATH2SPACE_UPSTREAM_ROOT,
    data_smoke_summary,
    evaluate_path2space_predictions,
    export_path2space_sourcefaithful_predictions,
    prepare_path2space_feature_cache,
    select_manifest_rows,
    train_path2space_sourcefaithful,
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
        description="Run source-faithful Path2Space CTransPath+MLP benchmark on HEST coverage95 log1p_rate targets."
    )
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--upstream-root", default=str(DEFAULT_PATH2SPACE_UPSTREAM_ROOT).replace("\\", "/"))
    parser.add_argument("--ctranspath-weight", default=str(DEFAULT_CTRANSPATH_WEIGHT_PATH).replace("\\", "/"))
    parser.add_argument(
        "--data-dir",
        default="results/hest1k_human_visium_expression/benchmark_inputs/path2space_sourcefaithful_full_fp32",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/hest1k_human_visium_expression_external/path2space_sourcefaithful_full_fp32",
    )
    parser.add_argument(
        "--prediction-root",
        default="results/hest1k_human_visium_expression/external_baselines/path2space_sourcefaithful_full_fp32_predictions",
    )
    parser.add_argument(
        "--benchmark-out-dir",
        default="results/hest1k_human_visium_expression/benchmark_results/path2space_sourcefaithful_full_fp32",
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
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--spot-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-macenko", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--prepare-features-only", action="store_true")
    parser.add_argument("--data-smoke-only", action="store_true")
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
    ctranspath_weight = resolve_project_path(args.ctranspath_weight)
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    prediction_root = resolve_project_path(args.prediction_root)
    benchmark_out_dir = resolve_project_path(args.benchmark_out_dir)
    if any(x is None for x in [upstream_root, ctranspath_weight, data_dir, output_dir, prediction_root, benchmark_out_dir]):
        raise ValueError("One or more paths resolved to None")

    train_slide_ids = _none_if_empty(args.train_slide_ids)
    val_slide_ids = _none_if_empty(args.val_slide_ids)
    test_slide_ids = _none_if_empty(args.test_slide_ids)

    train_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.train_splits],
        slide_ids=train_slide_ids,
        max_slides=args.max_train_slides,
        smallest_slides=bool(args.smallest_train_slides),
        max_slide_spots=args.max_train_slide_spots,
    )
    val_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.val_splits],
        slide_ids=val_slide_ids,
        max_slides=args.max_val_slides,
        smallest_slides=bool(args.smallest_val_slides),
        max_slide_spots=args.max_val_slide_spots,
    )
    test_rows = select_manifest_rows(
        cfg,
        splits=[str(x) for x in args.test_splits],
        slide_ids=test_slide_ids,
        max_slides=args.max_test_slides,
        smallest_slides=bool(args.smallest_test_slides),
        max_slide_spots=args.max_test_slide_spots,
    )

    if args.data_smoke_only:
        summary = data_smoke_summary(
            expression_config=cfg,
            splits=[str(x) for x in args.train_splits],
            slide_ids=train_slide_ids,
            max_slides=args.max_train_slides,
            smallest_slides=bool(args.smallest_train_slides),
            max_slide_spots=args.max_train_slide_spots,
            output_dir=Path(output_dir) / "data_smoke",
            feature_batch_size=int(args.feature_batch_size),
            spot_batch_size=int(args.spot_batch_size),
            upstream_root=upstream_root,
            ctranspath_weight_path=ctranspath_weight,
            device_name=args.device,
            use_macenko=bool(args.use_macenko),
            overwrite_features=bool(args.overwrite_features),
        )
        print(json.dumps({"data_smoke": summary}, indent=2), flush=True)
        return

    if args.prepare_features_only:
        prepare_rows = pd.concat([train_rows, val_rows, test_rows], ignore_index=True).drop_duplicates("sample_id")
        summary = prepare_path2space_feature_cache(
            expression_config=cfg,
            rows=prepare_rows,
            data_dir=data_dir,
            upstream_root=upstream_root,
            ctranspath_weight_path=ctranspath_weight,
            feature_batch_size=int(args.feature_batch_size),
            device_name=args.device,
            use_macenko=bool(args.use_macenko),
            overwrite=bool(args.overwrite_features),
        )
        print(json.dumps({"feature_cache": summary}, indent=2), flush=True)
        return

    train_summary = train_path2space_sourcefaithful(
        expression_config=cfg,
        train_rows=train_rows,
        val_rows=val_rows,
        output_dir=output_dir,
        data_dir=data_dir,
        upstream_root=upstream_root,
        ctranspath_weight_path=ctranspath_weight,
        feature_batch_size=int(args.feature_batch_size),
        spot_batch_size=int(args.spot_batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        patience=args.patience,
        min_delta=float(args.min_delta),
        seed=int(args.seed),
        device_name=args.device,
        use_macenko=bool(args.use_macenko),
        overwrite_features=bool(args.overwrite_features),
    )

    prediction_summary = None
    benchmark_summary = None
    if args.export_predictions or args.evaluate_predictions:
        prediction_summary = export_path2space_sourcefaithful_predictions(
            expression_config=cfg,
            checkpoint_path=train_summary["checkpoint"],
            data_dir=data_dir,
            out_dir=prediction_root,
            splits=[str(x) for x in args.test_splits],
            slide_ids=test_slide_ids,
            max_slides=args.max_test_slides,
            smallest_slides=bool(args.smallest_test_slides),
            max_slide_spots=args.max_test_slide_spots,
            feature_batch_size=int(args.feature_batch_size),
            spot_batch_size=int(args.spot_batch_size),
            device_name=args.device,
            upstream_root=upstream_root,
            ctranspath_weight_path=ctranspath_weight,
            use_macenko=bool(args.use_macenko),
            overwrite_features=bool(args.overwrite_features),
        )
    if args.evaluate_predictions:
        if prediction_summary is None or not bool(prediction_summary["benchmark_evaluable_without_truncation"]):
            raise ValueError("Refusing benchmark evaluation because Path2Space predictions are incomplete.")
        benchmark_summary = evaluate_path2space_predictions(
            expression_config=cfg,
            prediction_root=prediction_root,
            out_dir=benchmark_out_dir,
            splits=[str(x) for x in args.test_splits],
            slide_ids=test_slide_ids,
            max_slides=args.max_test_slides,
            max_slide_spots=args.max_test_slide_spots,
        )

    print(
        json.dumps(
            {
                "train": train_summary,
                "prediction": prediction_summary,
                "benchmark": benchmark_summary,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
