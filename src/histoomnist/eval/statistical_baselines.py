from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from histoomnist.data.dataset import FeatureStandardizer
from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.data.spot_table import load_spot_table
from histoomnist.eval.benchmark_predictions import (
    ScalarMetricAccumulator,
    VectorMetricAccumulator,
    group_frames,
    load_slide_target,
    update_group_accumulators,
)
from histoomnist.eval.evaluate_combined import _load_sf_model
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path


def _optional_path(row, name: str):
    if not hasattr(row, name):
        return None
    value = getattr(row, name)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if str(value).strip() == "":
        return None
    return value


def load_sf_training_config_and_checkpoint(path: str | Path, checkpoint: str | Path | None) -> tuple[dict, Path]:
    cfg_path = resolve_project_path(path)
    if cfg_path is None:
        raise ValueError("SF config path resolved to None")
    pointer_or_cfg = load_config(cfg_path)
    model_cfg = pointer_or_cfg.get("model", {})
    training_config = model_cfg.get("training_config")
    if training_config:
        sf_config_path = resolve_project_path(training_config)
        if sf_config_path is None:
            raise ValueError("SF training config resolved to None")
        sf_config = load_config(sf_config_path)
    else:
        sf_config = pointer_or_cfg
    ckpt = checkpoint or model_cfg.get("checkpoint")
    if ckpt is None:
        raise ValueError("Missing SF checkpoint.")
    ckpt_path = resolve_project_path(ckpt)
    if ckpt_path is None:
        raise ValueError("SF checkpoint resolved to None")
    return sf_config, ckpt_path


def target_setup(expression_config: dict) -> tuple[Path, pd.DataFrame, list[str], str, Path | None]:
    manifest_path = Path(expression_config["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    genes, gene_indices = selected_genes_from_config(expression_config, base_dir=manifest_path.parent)
    if genes is None or gene_indices is not None:
        raise ValueError("Statistical baselines require data.gene_names_path coverage95 target genes.")
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    return manifest_path, manifest, genes, gene_key, raw_st_root


def mean_rate_table(
    *,
    expression_config: dict,
    splits: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    manifest_path, manifest, genes, gene_key, raw_st_root = target_setup(expression_config)
    train_manifest = manifest[manifest["split"].isin(splits)].copy()
    if train_manifest.empty:
        raise ValueError(f"No training rows for splits={splits}")
    base_dir = manifest_path.parent
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    n_genes = len(genes)
    global_sum = np.zeros(n_genes, dtype=np.float64)
    global_n = np.zeros(n_genes, dtype=np.float64)
    organ_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_genes, dtype=np.float64))
    organ_n: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_genes, dtype=np.float64))
    rows = []

    for row in train_manifest.itertuples(index=False):
        target = load_slide_target(
            row=row,
            base_dir=base_dir,
            target_genes=genes,
            gene_key=gene_key,
            raw_st_root=raw_st_root,
            min_total_counts=min_total_counts,
        )
        inv_sf = 1.0 / np.clip(target.size_factor.astype(np.float64), 1.0e-6, None)
        rate = target.counts.multiply(inv_sf[:, None]).tocsr()
        sums = np.asarray(rate.sum(axis=0)).reshape(-1).astype(np.float64)
        measured = target.measured_genes.astype(bool)
        n_spots = float(target.counts.shape[0])
        global_sum[measured] += sums[measured]
        global_n[measured] += n_spots
        organ_sum[target.organ][measured] += sums[measured]
        organ_n[target.organ][measured] += n_spots
        rows.append(
            {
                "sample_id": target.sample_id,
                "split": target.split,
                "organ": target.organ,
                "n_spots": int(n_spots),
                "n_measured_genes": int(measured.sum()),
            }
        )
        print(
            f"[stat-baseline] train {target.sample_id}: spots={int(n_spots)} measured_genes={int(measured.sum())}",
            flush=True,
        )

    global_mean = np.divide(global_sum, np.maximum(global_n, 1.0)).astype(np.float32)
    organ_means: dict[str, np.ndarray] = {}
    for organ, sums in organ_sum.items():
        means = np.divide(sums, np.maximum(organ_n[organ], 1.0)).astype(np.float32)
        missing = organ_n[organ] <= 0
        means[missing] = global_mean[missing]
        organ_means[organ] = means
    return global_mean, organ_means, pd.DataFrame(rows)


def predict_slide_sf(
    *,
    row,
    base_dir: Path,
    expression_config: dict,
    sf_model: torch.nn.Module,
    sf_standardizer: FeatureStandardizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    table = load_spot_table(
        sample_id=str(row.sample_id),
        features_path=base_dir / str(row.features_path),
        counts_path=base_dir / str(row.counts_path),
        coords_path=base_dir / str(_optional_path(row, "coords_path"))
        if _optional_path(row, "coords_path") is not None
        else None,
        size_factor_path=base_dir / str(_optional_path(row, "size_factor_path"))
        if _optional_path(row, "size_factor_path") is not None
        else None,
        min_total_counts=float(expression_config["data"].get("min_total_counts", 1.0)),
    )
    features = table.features[table.valid_mask].astype(np.float32, copy=False)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            x = sf_standardizer.transform(features[start:stop])
            pred_log_sf = sf_model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
            chunks.append(np.exp(pred_log_sf).astype(np.float32, copy=False))
    sf = np.concatenate(chunks, axis=0)
    return (sf / (float(sf.mean()) + 1.0e-8)).astype(np.float32, copy=False)


def update_method(
    *,
    method_name: str,
    prediction_kind: str,
    pred: np.ndarray,
    true: np.ndarray,
    target,
    measured: np.ndarray,
    gene_accumulators: dict[str, VectorMetricAccumulator],
    group_accumulators: dict[str, dict[tuple[str, str], ScalarMetricAccumulator]],
) -> None:
    gene_accumulators[method_name].update(pred, true, measured)
    update_group_accumulators(
        accumulators=group_accumulators[method_name],
        prediction=pred,
        truth=true,
        target=target,
        measured_genes=measured,
    )


def evaluate_statistical_baselines(
    *,
    expression_config: dict,
    sf_config: dict,
    sf_checkpoint: str | Path,
    out_dir: str | Path,
    train_splits: list[str] | None = None,
    test_splits: list[str] | None = None,
    batch_size: int = 512,
    chunk_size: int = 512,
) -> dict[str, object]:
    train_splits = train_splits or list(expression_config["data"]["train_splits"])
    test_splits = test_splits or list(expression_config["data"]["test_splits"])
    device = torch.device(get_device_name(expression_config.get("device")))
    sf_ckpt = load_checkpoint(sf_checkpoint, map_location=str(device))
    sf_model = _load_sf_model(sf_config, sf_ckpt, device)
    sf_standardizer = FeatureStandardizer(mean=sf_ckpt["feature_mean"], std=sf_ckpt["feature_std"])

    manifest_path, manifest, genes, gene_key, raw_st_root = target_setup(expression_config)
    base_dir = manifest_path.parent
    test_manifest = manifest[manifest["split"].isin(test_splits)].copy()
    if test_manifest.empty:
        raise ValueError(f"No test rows for splits={test_splits}")
    global_mean, organ_means, train_rows = mean_rate_table(
        expression_config=expression_config,
        splits=train_splits,
    )

    method_kinds = {
        "global_mean_rate": "rate",
        "global_sf_only_count_pred_sf": "count",
        "global_sf_only_count_oracle_sf": "count",
        "organ_sf_only_count_pred_sf": "count",
        "organ_sf_only_count_oracle_sf": "count",
    }
    gene_accumulators = {method: VectorMetricAccumulator(len(genes)) for method in method_kinds}
    group_accumulators = {
        method: defaultdict(ScalarMetricAccumulator) for method in method_kinds
    }
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    slide_rows = []

    for row in test_manifest.itertuples(index=False):
        target = load_slide_target(
            row=row,
            base_dir=base_dir,
            target_genes=genes,
            gene_key=gene_key,
            raw_st_root=raw_st_root,
            min_total_counts=min_total_counts,
        )
        pred_sf = predict_slide_sf(
            row=row,
            base_dir=base_dir,
            expression_config=expression_config,
            sf_model=sf_model,
            sf_standardizer=sf_standardizer,
            device=device,
            batch_size=batch_size,
        )
        if pred_sf.shape[0] != target.counts.shape[0]:
            raise ValueError(
                f"SF spot count mismatch for {target.sample_id}: pred={pred_sf.shape[0]}, target={target.counts.shape[0]}"
            )
        organ_mean = organ_means.get(target.organ, global_mean)
        measured = target.measured_genes.astype(bool)
        for start in range(0, target.counts.shape[0], chunk_size):
            stop = min(start + chunk_size, target.counts.shape[0])
            true_count = target.counts[start:stop].astype(np.float32).toarray()
            true_rate = true_count / np.clip(target.size_factor[start:stop, None], 1.0e-6, None)
            n = stop - start
            global_rate = np.broadcast_to(global_mean[None, :], (n, len(genes))).astype(np.float32, copy=False)
            organ_rate = np.broadcast_to(organ_mean[None, :], (n, len(genes))).astype(np.float32, copy=False)
            update_method(
                method_name="global_mean_rate",
                prediction_kind="rate",
                pred=global_rate,
                true=true_rate,
                target=target,
                measured=measured,
                gene_accumulators=gene_accumulators,
                group_accumulators=group_accumulators,
            )
            for method, rate, sf in [
                ("global_sf_only_count_pred_sf", global_rate, pred_sf[start:stop]),
                ("global_sf_only_count_oracle_sf", global_rate, target.size_factor[start:stop]),
                ("organ_sf_only_count_pred_sf", organ_rate, pred_sf[start:stop]),
                ("organ_sf_only_count_oracle_sf", organ_rate, target.size_factor[start:stop]),
            ]:
                pred_count = (rate * sf[:, None]).astype(np.float32, copy=False)
                update_method(
                    method_name=method,
                    prediction_kind="count",
                    pred=pred_count,
                    true=true_count,
                    target=target,
                    measured=measured,
                    gene_accumulators=gene_accumulators,
                    group_accumulators=group_accumulators,
                )
        slide_rows.append(
            {
                "sample_id": target.sample_id,
                "split": target.split,
                "organ": target.organ,
                "cohort": target.cohort,
                "n_spots": int(target.counts.shape[0]),
                "n_measured_genes": int(measured.sum()),
                "pred_sf_mean": float(pred_sf.mean()),
                "pred_sf_std": float(pred_sf.std()),
                "true_sf_std": float(target.size_factor.std()),
            }
        )
        print(
            f"[stat-baseline] test {target.sample_id}: spots={target.counts.shape[0]} measured_genes={int(measured.sum())}",
            flush=True,
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows.to_csv(out / "train_slides_used.csv", index=False)
    pd.DataFrame(slide_rows).to_csv(out / "test_slides_evaluated.csv", index=False)
    np.savez_compressed(out / "train_mean_rates.npz", global_mean=global_mean, **organ_means)

    summary_rows = []
    for method, kind in method_kinds.items():
        method_dir = out / method
        method_dir.mkdir(parents=True, exist_ok=True)
        per_gene = gene_accumulators[method].to_frame(genes)
        per_gene.insert(0, "method", method)
        per_gene.insert(1, "prediction_kind", kind)
        per_gene.to_csv(method_dir / "per_gene_metrics.csv", index=False)
        overall, per_organ, per_slide = group_frames(
            group_accumulators[method],
            method_name=method,
            prediction_kind=kind,
        )
        overall.to_csv(method_dir / "overall_metrics.csv", index=False)
        per_organ.to_csv(method_dir / "per_organ_metrics.csv", index=False)
        per_slide.to_csv(method_dir / "per_slide_metrics.csv", index=False)
        metrics = gene_accumulators[method].summary()
        summary_rows.append({"method": method, "prediction_kind": kind, **metrics})
        (method_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "prediction_kind": kind,
                    "train_splits": train_splits,
                    "test_splits": test_splits,
                    "n_train_slides": int(len(train_rows)),
                    "n_test_slides": int(len(slide_rows)),
                    "n_genes": int(len(genes)),
                    "gene_metrics": metrics,
                    "sf_checkpoint": str(sf_checkpoint),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    summary = {
        "train_splits": train_splits,
        "test_splits": test_splits,
        "n_train_slides": int(len(train_rows)),
        "n_test_slides": int(len(slide_rows)),
        "n_genes": int(len(genes)),
        "methods": summary_rows,
        "sf_checkpoint": str(sf_checkpoint),
        "outputs": {
            "summary_csv": str(out / "summary.csv"),
            "train_slides_used": str(out / "train_slides_used.csv"),
            "test_slides_evaluated": str(out / "test_slides_evaluated.csv"),
        },
    }
    pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HEST coverage95 statistical benchmark baselines.")
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--sf-config", default="configs/hest1k_human_visium_sf_current.yaml")
    parser.add_argument("--sf-checkpoint", default=None)
    parser.add_argument("--train-splits", nargs="*", default=None)
    parser.add_argument("--test-splits", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--out-dir", default="results/hest1k_human_visium_expression/statistical_baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_config = load_config(args.expression_config)
    sf_config, sf_checkpoint = load_sf_training_config_and_checkpoint(args.sf_config, args.sf_checkpoint)
    evaluate_statistical_baselines(
        expression_config=expression_config,
        sf_config=sf_config,
        sf_checkpoint=sf_checkpoint,
        out_dir=args.out_dir,
        train_splits=args.train_splits,
        test_splits=args.test_splits,
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
    )


if __name__ == "__main__":
    main()
