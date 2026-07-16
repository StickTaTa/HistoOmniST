from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.eval.benchmark_predictions import evaluate_prediction_bundle  # noqa: E402
from histoomnist.external.train_hist2st_sourcefaithful import (  # noqa: E402
    data_smoke_summary,
    export_hist2st_sourcefaithful_predictions,
    forward_smoke_summary,
    train_hist2st_sourcefaithful,
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
        description="Train source-faithful Hist2ST on HEST full-slide patch H5 data."
    )
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--splits", nargs="*", default=["train"], help="Legacy smoke split selector.")
    parser.add_argument("--slide-ids", nargs="*", default=None, help="Legacy smoke slide selector.")
    parser.add_argument("--max-slides", type=int, default=None, help="Legacy smoke max slide count.")
    parser.add_argument("--train-splits", nargs="*", default=["train"])
    parser.add_argument("--val-splits", nargs="*", default=["val"])
    parser.add_argument("--test-splits", nargs="*", default=["test"])
    parser.add_argument("--train-slide-ids", nargs="*", default=None)
    parser.add_argument("--val-slide-ids", nargs="*", default=None)
    parser.add_argument("--test-slide-ids", nargs="*", default=None)
    parser.add_argument("--max-train-slides", type=int, default=None)
    parser.add_argument("--max-val-slides", type=int, default=None)
    parser.add_argument("--max-test-slides", type=int, default=None)
    parser.add_argument("--smallest-slides", action="store_true")
    parser.add_argument("--smallest-train-slides", action="store_true")
    parser.add_argument("--smallest-val-slides", action="store_true")
    parser.add_argument("--smallest-test-slides", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints/hest1k_human_visium_expression_external/hist2st_sourcefaithful")
    parser.add_argument("--prediction-root", default="results/hest1k_human_visium_expression/external_baselines/hist2st_sourcefaithful_predictions")
    parser.add_argument("--benchmark-out-dir", default="results/hest1k_human_visium_expression/benchmark_results/hist2st_sourcefaithful")
    parser.add_argument("--target-kind", choices=["log1p_rate"], default="log1p_rate")
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--prune", default="NA", choices=["Grid", "NA", "STD"])
    parser.add_argument("--graph-coord-source", default="position_grid", choices=["position_grid", "spatial"])
    parser.add_argument("--max-spots-per-slide", type=int, default=None)
    parser.add_argument("--max-slide-spots", type=int, default=None)
    parser.add_argument("--max-train-spots-per-slide", type=int, default=None)
    parser.add_argument("--max-val-spots-per-slide", type=int, default=None)
    parser.add_argument("--max-predict-spots-per-slide", type=int, default=None)
    parser.add_argument("--max-train-slide-spots", type=int, default=None)
    parser.add_argument("--max-val-slide-spots", type=int, default=None)
    parser.add_argument("--max-test-slide-spots", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--zinb", type=float, default=0.25)
    parser.add_argument("--nb", action="store_true")
    parser.add_argument("--bake", type=int, default=5)
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--data-smoke-only", action="store_true")
    parser.add_argument("--forward-smoke", action="store_true")
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
    slide_ids = _none_if_empty(args.slide_ids)
    train_slide_ids = _none_if_empty(args.train_slide_ids)
    val_slide_ids = _none_if_empty(args.val_slide_ids)
    test_slide_ids = _none_if_empty(args.test_slide_ids)
    model_cfg = {"zinb": float(args.zinb), "nb": bool(args.nb), "bake": int(args.bake), "lamb": float(args.lamb)}

    if args.forward_smoke and (args.max_spots_per_slide is not None or args.max_slide_spots is not None):
        raise ValueError("Forward smoke for source-faithful Hist2ST must not use spot truncation.")

    if args.data_smoke_only or args.forward_smoke:
        data_summary = data_smoke_summary(
            expression_config=cfg,
            splits=[str(x) for x in args.splits],
            slide_ids=slide_ids,
            max_slides=args.max_slides,
            smallest_slides=bool(args.smallest_slides),
            output_dir=output_dir / "data_smoke",
            target_kind=str(args.target_kind),
            k_neighbors=int(args.k_neighbors),
            prune=str(args.prune),
            graph_coord_source=str(args.graph_coord_source),
            max_spots_per_slide=args.max_spots_per_slide,
            max_slide_spots=args.max_slide_spots,
        )
        forward_summary = None
        if args.forward_smoke:
            forward_summary = forward_smoke_summary(
                expression_config=cfg,
                splits=[str(x) for x in args.splits],
                slide_ids=slide_ids,
                max_slides=args.max_slides,
                smallest_slides=bool(args.smallest_slides),
                output_dir=output_dir / "forward_smoke",
                target_kind=str(args.target_kind),
                device_name=args.device,
                seed=int(args.seed),
                model_cfg=model_cfg,
                k_neighbors=int(args.k_neighbors),
                prune=str(args.prune),
                graph_coord_source=str(args.graph_coord_source),
            )
        print(json.dumps({"data_smoke": data_summary, "forward_smoke": forward_summary}, indent=2), flush=True)
        return

    train_summary = train_hist2st_sourcefaithful(
        expression_config=cfg,
        train_splits=[str(x) for x in args.train_splits],
        val_splits=[str(x) for x in args.val_splits],
        train_slide_ids=train_slide_ids,
        val_slide_ids=val_slide_ids,
        max_train_slides=args.max_train_slides,
        max_val_slides=args.max_val_slides,
        smallest_train_slides=bool(args.smallest_train_slides),
        smallest_val_slides=bool(args.smallest_val_slides),
        max_train_spots_per_slide=args.max_train_spots_per_slide,
        max_val_spots_per_slide=args.max_val_spots_per_slide,
        max_train_slide_spots=args.max_train_slide_spots,
        max_val_slide_spots=args.max_val_slide_spots,
        output_dir=output_dir,
        target_kind=str(args.target_kind),
        epochs=int(args.epochs),
        lr=float(args.lr),
        patience=args.patience,
        min_delta=float(args.min_delta),
        seed=int(args.seed),
        device_name=args.device,
        model_cfg=model_cfg,
        k_neighbors=int(args.k_neighbors),
        prune=str(args.prune),
        graph_coord_source=str(args.graph_coord_source),
    )

    prediction_summary = None
    benchmark_summary = None
    if args.export_predictions or args.evaluate_predictions:
        prediction_root = resolve_project_path(args.prediction_root)
        if prediction_root is None:
            raise ValueError("Prediction root resolved to None")
        prediction_summary = export_hist2st_sourcefaithful_predictions(
            expression_config=cfg,
            checkpoint_path=Path(train_summary["checkpoint"]),
            out_dir=prediction_root,
            splits=[str(x) for x in args.test_splits],
            slide_ids=test_slide_ids,
            max_slides=args.max_test_slides,
            smallest_slides=bool(args.smallest_test_slides),
            max_spots_per_slide=args.max_predict_spots_per_slide,
            max_slide_spots=args.max_test_slide_spots,
            device_name=args.device,
        )
        if args.evaluate_predictions:
            if not bool(prediction_summary["benchmark_evaluable_without_truncation"]):
                raise ValueError(
                    "Refusing benchmark evaluation because exported predictions are truncated. "
                    "Use full-slide prediction export with no max-predict-spots-per-slide."
                )
            benchmark_out_dir = resolve_project_path(args.benchmark_out_dir)
            if benchmark_out_dir is None:
                raise ValueError("Benchmark output dir resolved to None")
            benchmark_summary = evaluate_prediction_bundle(
                expression_config=cfg,
                prediction_root=prediction_root,
                method_name="hist2st_sourcefaithful",
                prediction_kind=str(prediction_summary["prediction_kind"]),
                out_dir=benchmark_out_dir,
                splits=[str(x) for x in args.test_splits],
                prediction_genes_path=prediction_root / "genes.txt",
                max_slides=args.max_test_slides,
                slide_ids=test_slide_ids,
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
