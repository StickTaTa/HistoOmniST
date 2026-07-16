from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.external.mclstexp_sourcefaithful import DEFAULT_MCLSTEXP_UPSTREAM_ROOT  # noqa: E402
from histoomnist.external.train_mclstexp_sourcefaithful import (  # noqa: E402
    data_smoke_summary,
    evaluate_mclstexp_predictions,
    export_mclstexp_sourcefaithful_predictions,
    train_mclstexp_sourcefaithful,
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
        description="Train source-faithful mclSTExp on HEST RGB patch-H5 data and export top-k retrieval predictions."
    )
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--upstream-root", default=str(DEFAULT_MCLSTEXP_UPSTREAM_ROOT).replace("\\", "/"))
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
    parser.add_argument("--target-gene-limit", type=int, default=None)
    parser.add_argument("--max-train-spots-per-slide", type=int, default=None)
    parser.add_argument("--max-val-spots-per-slide", type=int, default=None)
    parser.add_argument("--max-predict-spots-per-slide", type=int, default=None)
    parser.add_argument("--max-train-slide-spots", type=int, default=None)
    parser.add_argument("--max-val-slide-spots", type=int, default=None)
    parser.add_argument("--max-test-slide-spots", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="checkpoints/hest1k_human_visium_expression_external/mclstexp_sourcefaithful_full_fp32",
    )
    parser.add_argument(
        "--prediction-root",
        default="results/hest1k_human_visium_expression/external_baselines/mclstexp_sourcefaithful_full_fp32_predictions",
    )
    parser.add_argument(
        "--benchmark-out-dir",
        default="results/hest1k_human_visium_expression/benchmark_results/mclstexp_sourcefaithful_full_fp32",
    )
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
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
    output_dir = resolve_project_path(args.output_dir)
    if output_dir is None:
        raise ValueError("Output dir resolved to None")
    upstream_root = resolve_project_path(args.upstream_root)
    if upstream_root is None:
        raise ValueError("mclSTExp upstream root resolved to None")

    train_slide_ids = _none_if_empty(args.train_slide_ids)
    val_slide_ids = _none_if_empty(args.val_slide_ids)
    test_slide_ids = _none_if_empty(args.test_slide_ids)

    if args.data_smoke_only:
        summary = data_smoke_summary(
            expression_config=cfg,
            splits=[str(x) for x in args.train_splits],
            slide_ids=train_slide_ids,
            max_slides=args.max_train_slides,
            smallest_slides=bool(args.smallest_train_slides),
            target_gene_limit=args.target_gene_limit,
            max_spots_per_slide=args.max_train_spots_per_slide,
            max_slide_spots=args.max_train_slide_spots,
            output_dir=output_dir / "data_smoke",
            upstream_root=upstream_root,
        )
        print(json.dumps({"data_smoke": summary}, indent=2), flush=True)
        return

    train_summary = train_mclstexp_sourcefaithful(
        expression_config=cfg,
        train_splits=[str(x) for x in args.train_splits],
        val_splits=[str(x) for x in args.val_splits],
        train_slide_ids=train_slide_ids,
        val_slide_ids=val_slide_ids,
        max_train_slides=args.max_train_slides,
        max_val_slides=args.max_val_slides,
        smallest_train_slides=bool(args.smallest_train_slides),
        smallest_val_slides=bool(args.smallest_val_slides),
        target_gene_limit=args.target_gene_limit,
        max_train_spots_per_slide=args.max_train_spots_per_slide,
        max_val_spots_per_slide=args.max_val_spots_per_slide,
        max_train_slide_spots=args.max_train_slide_spots,
        max_val_slide_spots=args.max_val_slide_spots,
        output_dir=output_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        patience=args.patience,
        min_delta=float(args.min_delta),
        seed=int(args.seed),
        device_name=args.device,
        upstream_root=upstream_root,
    )

    prediction_summary = None
    benchmark_summary = None
    if args.export_predictions or args.evaluate_predictions:
        prediction_root = resolve_project_path(args.prediction_root)
        if prediction_root is None:
            raise ValueError("Prediction root resolved to None")
        prediction_summary = export_mclstexp_sourcefaithful_predictions(
            expression_config=cfg,
            checkpoint_path=Path(train_summary["checkpoint"]),
            out_dir=prediction_root,
            splits=[str(x) for x in args.test_splits],
            train_splits=[str(x) for x in args.train_splits],
            train_slide_ids=train_slide_ids,
            max_train_slides=args.max_train_slides,
            smallest_train_slides=bool(args.smallest_train_slides),
            max_train_slide_spots=args.max_train_slide_spots,
            slide_ids=test_slide_ids,
            max_slides=args.max_test_slides,
            smallest_slides=bool(args.smallest_test_slides),
            max_train_spots_per_slide=args.max_train_spots_per_slide,
            max_predict_spots_per_slide=args.max_predict_spots_per_slide,
            max_test_slide_spots=args.max_test_slide_spots,
            embedding_batch_size=int(args.embedding_batch_size),
            retrieval_batch_size=int(args.retrieval_batch_size),
            top_k=int(args.top_k),
            num_workers=int(args.num_workers),
            device_name=args.device,
            upstream_root=upstream_root,
        )
        if args.evaluate_predictions:
            if not bool(prediction_summary["benchmark_evaluable_without_truncation"]):
                raise ValueError(
                    "Refusing benchmark evaluation because exported predictions are truncated or incomplete."
                )
            benchmark_out_dir = resolve_project_path(args.benchmark_out_dir)
            if benchmark_out_dir is None:
                raise ValueError("Benchmark output dir resolved to None")
            benchmark_summary = evaluate_mclstexp_predictions(
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
            {"train": train_summary, "prediction": prediction_summary, "benchmark": benchmark_summary},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
