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

from histoomnist.external.train_histogene_patch import (  # noqa: E402
    export_histogene_patch_predictions,
    train_histogene_patch,
)
from histoomnist.utils.config import load_config  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HisToGene-style HEST patch-H5 external baseline.")
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--train-splits", nargs="*", default=["train"])
    parser.add_argument("--val-splits", nargs="*", default=["val"])
    parser.add_argument("--test-splits", nargs="*", default=["test"])
    parser.add_argument("--output-dir", default="checkpoints/hest1k_human_visium_expression_external/histogene_patch_h5")
    parser.add_argument("--prediction-root", default=None)
    parser.add_argument("--target-kind", choices=["log1p_rate"], default="log1p_rate")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=56)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-pos", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-slides", type=int, default=None)
    parser.add_argument("--max-val-slides", type=int, default=None)
    parser.add_argument("--max-test-slides", type=int, default=None)
    parser.add_argument("--max-train-chunks-per-slide", type=int, default=None)
    parser.add_argument("--max-val-chunks-per-slide", type=int, default=None)
    parser.add_argument("--max-predict-chunks-per-slide", type=int, default=None)
    parser.add_argument("--max-predict-spots-per-slide", type=int, default=None)
    parser.add_argument("--export-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.expression_config))
    output_dir = resolve_project_path(args.output_dir)
    if output_dir is None:
        raise ValueError("Output dir resolved to None")
    model_cfg = {
        "patch_size": int(args.patch_size),
        "dim": int(args.dim),
        "n_layers": int(args.layers),
        "n_heads": int(args.heads),
        "dropout": float(args.dropout),
        "n_pos": int(args.n_pos),
    }
    train_summary = train_histogene_patch(
        expression_config=cfg,
        train_splits=[str(x) for x in args.train_splits],
        val_splits=[str(x) for x in args.val_splits],
        output_dir=output_dir,
        model_cfg=model_cfg,
        target_kind=str(args.target_kind),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device_name=args.device,
        max_train_slides=args.max_train_slides,
        max_val_slides=args.max_val_slides,
        max_train_chunks_per_slide=args.max_train_chunks_per_slide,
        max_val_chunks_per_slide=args.max_val_chunks_per_slide,
    )
    prediction_summary = None
    if args.export_predictions:
        prediction_root = resolve_project_path(
            args.prediction_root
            or "results/hest1k_human_visium_expression/external_baselines/histogene_patch_h5_predictions"
        )
        if prediction_root is None:
            raise ValueError("Prediction root resolved to None")
        prediction_summary = export_histogene_patch_predictions(
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
    summary = {"train": train_summary, "prediction": prediction_summary}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
