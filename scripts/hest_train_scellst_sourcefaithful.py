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

from histoomnist.external.scellst_sourcefaithful import (  # noqa: E402
    DEFAULT_SCELLST_EMBEDDING_TAG,
    DEFAULT_SCELLST_SHAPE_NAME,
    DEFAULT_SCELLST_UPSTREAM_ROOT,
    data_smoke_summary,
    download_hest_segmentation_assets,
    evaluate_scellst_predictions,
    export_scellst_sourcefaithful_predictions,
    prepare_scellst_cell_assets,
    prepare_scellst_log1p_rate_h5ads,
    select_manifest_rows,
    train_scellst_sourcefaithful,
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
        description="Run source-faithful sCellST MIL benchmark on HEST coverage95 log1p_rate targets."
    )
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--upstream-root", default=str(DEFAULT_SCELLST_UPSTREAM_ROOT).replace("\\", "/"))
    parser.add_argument("--raw-root", default="data/HEST-1k/raw")
    parser.add_argument(
        "--data-dir",
        default="results/hest1k_human_visium_expression/benchmark_inputs/scellst_sourcefaithful_full_fp32",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/hest1k_human_visium_expression_external/scellst_sourcefaithful_full_fp32",
    )
    parser.add_argument(
        "--prediction-root",
        default="results/hest1k_human_visium_expression/external_baselines/scellst_sourcefaithful_full_fp32_predictions",
    )
    parser.add_argument(
        "--benchmark-out-dir",
        default="results/hest1k_human_visium_expression/benchmark_results/scellst_sourcefaithful_full_fp32",
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
    parser.add_argument("--embedding-tag", default=DEFAULT_SCELLST_EMBEDDING_TAG)
    parser.add_argument("--shape-name", default=DEFAULT_SCELLST_SHAPE_NAME)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-dim", nargs="*", type=int, default=[256, 256, 256])
    parser.add_argument("--final-activation", default="softplus")
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--download-segmentation", action="store_true")
    parser.add_argument("--skip-prepare-h5ad", action="store_true")
    parser.add_argument("--skip-cell-assets", action="store_true")
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
    if upstream_root is None:
        raise ValueError("sCellST upstream root resolved to None")
    raw_root = resolve_project_path(args.raw_root)
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    prediction_root = resolve_project_path(args.prediction_root)
    benchmark_out_dir = resolve_project_path(args.benchmark_out_dir)
    if any(x is None for x in [raw_root, data_dir, output_dir, prediction_root, benchmark_out_dir]):
        raise ValueError("One or more output paths resolved to None")

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
    if args.export_predictions or args.evaluate_predictions:
        prepare_rows = pd.concat([prepare_rows, test_rows], ignore_index=True)
    prepare_rows = prepare_rows.drop_duplicates("sample_id").reset_index(drop=True)

    prep_summary = None
    cell_summary = None
    if args.download_segmentation:
        download_hest_segmentation_assets(
            raw_root=raw_root,
            slide_ids=[str(x) for x in prepare_rows["sample_id"].tolist()],
            output_dir=Path(data_dir) / "asset_downloads",
        )
    if not args.skip_prepare_h5ad:
        prep_summary = prepare_scellst_log1p_rate_h5ads(
            expression_config=cfg,
            rows=prepare_rows,
            data_dir=data_dir,
            target_gene_limit=args.target_gene_limit,
        )
    if not args.skip_cell_assets:
        cell_summary = prepare_scellst_cell_assets(
            raw_root=raw_root,
            data_dir=data_dir,
            slide_ids=[str(x) for x in prepare_rows["sample_id"].tolist()],
            upstream_root=upstream_root,
            shape_name=str(args.shape_name),
            embedding_tag=str(args.embedding_tag),
        )
    if args.data_smoke_only:
        smoke = data_smoke_summary(
            expression_config=cfg,
            data_dir=data_dir,
            rows=train_rows,
            output_dir=Path(output_dir) / "data_smoke",
            upstream_root=upstream_root,
            target_gene_limit=args.target_gene_limit,
            embedding_tag=str(args.embedding_tag),
            shape_name=str(args.shape_name),
        )
        print(json.dumps({"prepared": prep_summary, "cell_assets": cell_summary, "data_smoke": smoke}, indent=2), flush=True)
        return

    train_summary = train_scellst_sourcefaithful(
        expression_config=cfg,
        data_dir=data_dir,
        output_dir=output_dir,
        train_rows=train_rows,
        val_rows=val_rows,
        upstream_root=upstream_root,
        target_gene_limit=args.target_gene_limit,
        embedding_tag=str(args.embedding_tag),
        shape_name=str(args.shape_name),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        patience=args.patience,
        min_delta=float(args.min_delta),
        seed=int(args.seed),
        device_name=args.device,
        hidden_dim=[int(x) for x in args.hidden_dim],
        final_activation=str(args.final_activation),
        dropout_rate=float(args.dropout_rate),
    )

    prediction_summary = None
    benchmark_summary = None
    if args.export_predictions or args.evaluate_predictions:
        prediction_summary = export_scellst_sourcefaithful_predictions(
            expression_config=cfg,
            checkpoint_path=train_summary["checkpoint"],
            data_dir=data_dir,
            out_dir=prediction_root,
            test_rows=test_rows,
            upstream_root=upstream_root,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device_name=args.device,
        )
    if args.evaluate_predictions:
        benchmark_summary = evaluate_scellst_predictions(
            expression_config=cfg,
            prediction_root=prediction_root,
            out_dir=benchmark_out_dir,
            test_rows=test_rows,
        )
    print(
        json.dumps(
            {
                "prepared": prep_summary,
                "cell_assets": cell_summary,
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
