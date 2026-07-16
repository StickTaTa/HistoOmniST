from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.external.stnet_sourcefaithful import (  # noqa: E402
    DEFAULT_STNET_UPSTREAM_ROOT,
    data_smoke_summary,
    evaluate_stnet_predictions,
    export_stnet_sourcefaithful_predictions,
    select_manifest_rows,
    train_stnet_sourcefaithful,
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
        description="Run source-faithful ST-Net DenseNet-121 benchmark on HEST coverage95 log1p_rate targets."
    )
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--upstream-root", default=str(DEFAULT_STNET_UPSTREAM_ROOT).replace("\\", "/"))
    parser.add_argument(
        "--output-dir",
        default="checkpoints/hest1k_human_visium_expression_external/stnet_sourcefaithful_full_fp32",
    )
    parser.add_argument(
        "--prediction-root",
        default="results/hest1k_human_visium_expression/external_baselines/stnet_sourcefaithful_full_fp32_predictions",
    )
    parser.add_argument(
        "--benchmark-out-dir",
        default="results/hest1k_human_visium_expression/benchmark_results/stnet_sourcefaithful_full_fp32",
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-average", action="store_true")
    parser.add_argument("--data-smoke-only", action="store_true")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--evaluate-predictions", action="store_true")
    parser.add_argument("--checkpoint-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_project_path(args.expression_config)
    if cfg_path is None:
        raise ValueError("Expression config resolved to None")
    cfg = load_config(cfg_path)
    upstream_root = resolve_project_path(args.upstream_root)
    output_dir = resolve_project_path(args.output_dir)
    prediction_root = resolve_project_path(args.prediction_root)
    benchmark_out_dir = resolve_project_path(args.benchmark_out_dir)
    if any(x is None for x in [upstream_root, output_dir, prediction_root, benchmark_out_dir]):
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
            rows=train_rows,
            output_dir=Path(output_dir) / "data_smoke",
            upstream_root=upstream_root,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pretrained=not bool(args.no_pretrained),
            device_name=args.device,
        )
        print(json.dumps({"data_smoke": summary}, indent=2), flush=True)
        return

    checkpoint_path = resolve_project_path(args.checkpoint_path) if args.checkpoint_path else None
    train_summary = None
    if checkpoint_path is None:
        train_summary = train_stnet_sourcefaithful(
            expression_config=cfg,
            train_rows=train_rows,
            val_rows=val_rows,
            output_dir=output_dir,
            upstream_root=upstream_root,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            test_batch_size=args.test_batch_size,
            lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            patience=args.patience,
            min_delta=float(args.min_delta),
            pretrained=not bool(args.no_pretrained),
            average=not bool(args.no_average),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            device_name=args.device,
            data_parallel=bool(args.data_parallel),
        )
        checkpoint_path = Path(train_summary["checkpoint"])
    if args.export_predictions or args.evaluate_predictions:
        prediction_summary = export_stnet_sourcefaithful_predictions(
            expression_config=cfg,
            test_rows=test_rows,
            checkpoint_path=checkpoint_path,
            prediction_root=prediction_root,
            batch_size=int(args.test_batch_size or args.batch_size),
            num_workers=int(args.num_workers),
            average=None if not args.no_average else False,
            device_name=args.device,
        )
        print(json.dumps({"prediction_summary": prediction_summary}, indent=2), flush=True)
    if args.evaluate_predictions:
        benchmark_summary = evaluate_stnet_predictions(
            prediction_root=prediction_root,
            benchmark_out_dir=benchmark_out_dir,
            expression_config=cfg,
            splits=[str(x) for x in args.test_splits],
            slide_ids=test_slide_ids,
        )
        print(json.dumps({"benchmark_summary": benchmark_summary}, indent=2), flush=True)
    if train_summary is not None and not (args.export_predictions or args.evaluate_predictions):
        print(json.dumps({"train_summary": train_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
