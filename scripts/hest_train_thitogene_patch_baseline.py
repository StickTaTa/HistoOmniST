from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")

from histoomnist.eval.benchmark_predictions import evaluate_prediction_bundle  # noqa: E402
from histoomnist.external.train_thitogene_patch import (  # noqa: E402
    export_thitogene_patch_predictions,
    train_thitogene_patch,
)
from histoomnist.utils.config import load_config  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a THItoGene-style HEST patch-H5 external baseline.")
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--train-splits", nargs="*", default=["train"])
    parser.add_argument("--val-splits", nargs="*", default=["val"])
    parser.add_argument("--test-splits", nargs="*", default=["test"])
    parser.add_argument("--output-dir", default="checkpoints/hest1k_human_visium_expression_external/thitogene_patch_h5")
    parser.add_argument("--prediction-root", default=None)
    parser.add_argument("--target-kind", choices=["log1p_rate"], default="log1p_rate")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=112)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-pos", type=int, default=64)
    parser.add_argument("--caps", type=int, default=20)
    parser.add_argument("--route-dim", type=int, default=64)
    parser.add_argument("--gat-hidden", type=int, default=128)
    parser.add_argument("--gat-out", type=int, default=256)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-slides", type=int, default=None)
    parser.add_argument("--max-val-slides", type=int, default=None)
    parser.add_argument("--max-test-slides", type=int, default=None)
    parser.add_argument("--max-train-chunks-per-slide", type=int, default=None)
    parser.add_argument("--max-val-chunks-per-slide", type=int, default=None)
    parser.add_argument("--max-predict-chunks-per-slide", type=int, default=None)
    parser.add_argument("--max-predict-spots-per-slide", type=int, default=None)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--evaluate-predictions", action="store_true")
    parser.add_argument(
        "--benchmark-out-dir",
        default="results/hest1k_human_visium_expression/benchmark_results/thitogene_patch_h5",
    )
    parser.add_argument("--benchmark-max-slides", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.expression_config))
    output_dir = resolve_project_path(args.output_dir)
    if output_dir is None:
        raise ValueError("Output dir resolved to None")
    model_cfg = {
        "patch_size": int(args.patch_size),
        "n_layers": int(args.layers),
        "transformer_heads": int(args.transformer_heads),
        "gat_heads": int(args.gat_heads),
        "dim": int(args.dim),
        "dropout": float(args.dropout),
        "n_pos": int(args.n_pos),
        "caps": int(args.caps),
        "route_dim": int(args.route_dim),
        "gat_hidden": int(args.gat_hidden),
        "gat_out": int(args.gat_out),
    }
    train_summary = train_thitogene_patch(
        expression_config=cfg,
        train_splits=[str(x) for x in args.train_splits],
        val_splits=[str(x) for x in args.val_splits],
        output_dir=output_dir,
        model_cfg=model_cfg,
        target_kind=str(args.target_kind),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        k_neighbors=int(args.k_neighbors),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device_name=args.device,
        max_train_slides=args.max_train_slides,
        max_val_slides=args.max_val_slides,
        max_train_chunks_per_slide=args.max_train_chunks_per_slide,
        max_val_chunks_per_slide=args.max_val_chunks_per_slide,
        seed=int(args.seed),
    )
    prediction_summary = None
    benchmark_summary = None
    if args.export_predictions or args.evaluate_predictions:
        prediction_root = resolve_project_path(
            args.prediction_root
            or "results/hest1k_human_visium_expression/external_baselines/thitogene_patch_h5_predictions"
        )
        if prediction_root is None:
            raise ValueError("Prediction root resolved to None")
        prediction_summary = export_thitogene_patch_predictions(
            expression_config=cfg,
            checkpoint_path=Path(train_summary["checkpoint"]),
            out_dir=prediction_root,
            splits=[str(x) for x in args.test_splits],
            batch_size=1,
            chunk_size=int(args.chunk_size),
            device_name=args.device,
            max_slides=args.max_test_slides,
            max_chunks_per_slide=args.max_predict_chunks_per_slide,
            max_spots_per_slide=args.max_predict_spots_per_slide,
        )
        if args.evaluate_predictions:
            if not bool(prediction_summary["benchmark_evaluable_without_truncation"]):
                raise ValueError(
                    "Refusing benchmark evaluation because exported predictions are truncated. "
                    "Use full-slide prediction export with no max-predict truncation options."
                )
            benchmark_out_dir = resolve_project_path(args.benchmark_out_dir)
            if benchmark_out_dir is None:
                raise ValueError("Benchmark output dir resolved to None")
            benchmark_summary = evaluate_prediction_bundle(
                expression_config=cfg,
                prediction_root=prediction_root,
                method_name="thitogene_patch_h5",
                prediction_kind=str(prediction_summary["prediction_kind"]),
                out_dir=benchmark_out_dir,
                splits=[str(x) for x in args.test_splits],
                prediction_genes_path=prediction_root / "genes.txt",
                max_slides=args.benchmark_max_slides if args.benchmark_max_slides is not None else args.max_test_slides,
            )
    summary = {"train": train_summary, "prediction": prediction_summary, "benchmark": benchmark_summary}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
