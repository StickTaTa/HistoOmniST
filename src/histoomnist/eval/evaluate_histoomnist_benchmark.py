from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from histoomnist.data.dataset import ExpressionRateDataset, FeatureStandardizer
from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.eval.benchmark_predictions import (
    ScalarMetricAccumulator,
    VectorMetricAccumulator,
    group_frames,
    update_group_accumulators,
)
from histoomnist.eval.evaluate_combined import _load_rate_model, _load_sf_model
from histoomnist.eval.metrics import sf_metrics
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path


@dataclass(frozen=True)
class SampleLabels:
    sample_id: str
    split: str
    organ: str
    cohort: str
    disease_state: str


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


def default_expression_checkpoint(cfg: dict) -> Path:
    output_dir = cfg.get("output", {}).get("dir")
    if output_dir:
        return Path(output_dir) / "best.pt"
    checkpoint_root = cfg.get("paths", {}).get("checkpoint_root", "checkpoints")
    return Path(checkpoint_root) / str(cfg["project"]["name"]) / "best.pt"


def sample_labels(manifest: pd.DataFrame) -> dict[str, SampleLabels]:
    out: dict[str, SampleLabels] = {}
    for row in manifest.itertuples(index=False):
        out[str(row.sample_id)] = SampleLabels(
            sample_id=str(row.sample_id),
            split=str(getattr(row, "split", "")),
            organ=str(getattr(row, "organ", "")),
            cohort=str(getattr(row, "cohort", "")),
            disease_state=str(getattr(row, "disease_state", "")),
        )
    return out


def predict_slide_normalized_sf(
    *,
    ds: ExpressionRateDataset,
    loader: DataLoader,
    sf_model: torch.nn.Module,
    sf_ckpt: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pred_sf = np.empty(len(ds), dtype=np.float32)
    true_log_sf = np.empty(len(ds), dtype=np.float32)
    sf_standardizer = FeatureStandardizer(mean=sf_ckpt["feature_mean"], std=sf_ckpt["feature_std"])
    offset = 0
    with torch.no_grad():
        for batch in loader:
            raw_features = batch["raw_features"].numpy()
            sf_features = sf_standardizer.transform(raw_features)
            batch_pred_log_sf = sf_model(torch.from_numpy(sf_features).to(device)).cpu().numpy().reshape(-1)
            batch_true_log_sf = batch["true_log_sf"].numpy().reshape(-1)
            stop = offset + batch_pred_log_sf.shape[0]
            pred_sf[offset:stop] = np.exp(batch_pred_log_sf).astype(np.float32)
            true_log_sf[offset:stop] = batch_true_log_sf.astype(np.float32)
            offset = stop
    rows = []
    for sample_id in np.unique(ds.sample_ids):
        idx = ds.sample_ids == sample_id
        pred_sf[idx] = pred_sf[idx] / (float(pred_sf[idx].mean()) + 1.0e-8)
        metrics = sf_metrics(np.log(pred_sf[idx] + 1.0e-8), true_log_sf[idx])
        rows.append({"sample_id": str(sample_id), "n_spots": int(idx.sum()), **metrics})
    return pred_sf, true_log_sf.reshape(-1), pd.DataFrame(rows)


def update_method_for_batch(
    *,
    method_name: str,
    prediction_kind: str,
    pred: np.ndarray,
    true: np.ndarray,
    expression_mask: np.ndarray,
    sample_ids: np.ndarray,
    label_lookup: dict[str, SampleLabels],
    gene_accumulators: dict[str, VectorMetricAccumulator],
    group_accumulators: dict[str, dict[tuple[str, str], ScalarMetricAccumulator]],
) -> None:
    for sample_id in np.unique(sample_ids):
        spot_mask = sample_ids == sample_id
        measured = expression_mask[spot_mask][0].astype(bool)
        target = label_lookup[str(sample_id)]
        gene_accumulators[method_name].update(pred[spot_mask], true[spot_mask], measured)
        update_group_accumulators(
            accumulators=group_accumulators[method_name],
            prediction=pred[spot_mask],
            truth=true[spot_mask],
            target=target,
            measured_genes=measured,
        )


def write_method_outputs(
    *,
    out_dir: Path,
    genes: list[str],
    method_kinds: dict[str, str],
    gene_accumulators: dict[str, VectorMetricAccumulator],
    group_accumulators: dict[str, dict[tuple[str, str], ScalarMetricAccumulator]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, kind in method_kinds.items():
        method_dir = out_dir / method
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
        rows.append({"method": method, "prediction_kind": kind, **metrics})
        (method_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "prediction_kind": kind,
                    "n_genes": len(genes),
                    "gene_metrics": metrics,
                    "outputs": {
                        "per_gene_metrics": str(method_dir / "per_gene_metrics.csv"),
                        "overall_metrics": str(method_dir / "overall_metrics.csv"),
                        "per_organ_metrics": str(method_dir / "per_organ_metrics.csv"),
                        "per_slide_metrics": str(method_dir / "per_slide_metrics.csv"),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return rows


def evaluate_histoomnist_benchmark(
    *,
    expression_config: dict,
    sf_config: dict,
    expression_checkpoint: str | Path,
    sf_checkpoint: str | Path,
    out_dir: str | Path,
    splits: list[str] | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    device = torch.device(get_device_name(expression_config.get("device")))
    expression_ckpt = load_checkpoint(expression_checkpoint, map_location=str(device))
    sf_ckpt = load_checkpoint(sf_checkpoint, map_location=str(device))
    manifest_path = Path(expression_config["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    selected_splits = splits or list(expression_config["data"]["test_splits"])
    manifest = manifest[manifest["split"].isin(selected_splits)].copy()
    if manifest.empty:
        raise ValueError(f"No manifest rows for splits={selected_splits}")
    base_dir = manifest_path.parent
    genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
    if genes is None or gene_indices is not None:
        raise ValueError("HistoOmniST benchmark requires data.gene_names_path coverage95 target genes.")
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    ds = ExpressionRateDataset(
        manifest,
        base_dir=base_dir,
        splits=selected_splits,
        min_total_counts=float(expression_config["data"].get("min_total_counts", 1.0)),
        standardizer=FeatureStandardizer(mean=expression_ckpt["feature_mean"], std=expression_ckpt["feature_std"]),
        gene_names=genes,
        gene_indices=gene_indices,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    loader = DataLoader(
        ds,
        batch_size=int(batch_size or expression_config["training"]["batch_size"]),
        shuffle=False,
    )
    sf_model = _load_sf_model(sf_config, sf_ckpt, device)
    rate_model = _load_rate_model(expression_config, expression_ckpt, device)
    pred_sf, true_log_sf, sf_slide = predict_slide_normalized_sf(
        ds=ds,
        loader=loader,
        sf_model=sf_model,
        sf_ckpt=sf_ckpt,
        device=device,
    )
    label_lookup = sample_labels(manifest)
    method_kinds = {
        "histoomnist_rate": "rate",
        "histoomnist_count_no_sf": "count",
        "histoomnist_count_pred_sf": "count",
        "histoomnist_count_oracle_sf": "count",
    }
    gene_accumulators = {method: VectorMetricAccumulator(len(genes)) for method in method_kinds}
    group_accumulators = {
        method: defaultdict(ScalarMetricAccumulator) for method in method_kinds
    }
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            pred_log1p_rate = rate_model(batch["features"].to(device)).cpu().numpy()
            pred_rate = np.expm1(pred_log1p_rate).clip(min=0.0)
            true_rate = np.expm1(batch["log1p_rate"].numpy())
            expression_mask = batch["expression_mask"].numpy().astype(bool)
            batch_size_actual = pred_rate.shape[0]
            stop = offset + batch_size_actual
            batch_pred_sf = pred_sf[offset:stop]
            batch_true_sf = np.exp(batch["true_log_sf"].numpy().reshape(-1))
            batch_sample_ids = ds.sample_ids[offset:stop].astype(str)
            true_count = true_rate * batch_true_sf[:, None]
            predictions = {
                "histoomnist_rate": (pred_rate, true_rate),
                "histoomnist_count_no_sf": (pred_rate, true_count),
                "histoomnist_count_pred_sf": (pred_rate * batch_pred_sf[:, None], true_count),
                "histoomnist_count_oracle_sf": (pred_rate * batch_true_sf[:, None], true_count),
            }
            for method, (pred, true) in predictions.items():
                update_method_for_batch(
                    method_name=method,
                    prediction_kind=method_kinds[method],
                    pred=pred,
                    true=true,
                    expression_mask=expression_mask,
                    sample_ids=batch_sample_ids,
                    label_lookup=label_lookup,
                    gene_accumulators=gene_accumulators,
                    group_accumulators=group_accumulators,
                )
            offset = stop
            if batch_idx % 20 == 0:
                print(f"[histoomnist-benchmark] processed batches={batch_idx} spots={offset}", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    method_rows = write_method_outputs(
        out_dir=out,
        genes=genes,
        method_kinds=method_kinds,
        gene_accumulators=gene_accumulators,
        group_accumulators=group_accumulators,
    )
    sf_overall = sf_metrics(np.log(pred_sf + 1.0e-8), true_log_sf)
    sf_slide = sf_slide.merge(
        pd.DataFrame.from_dict({k: v.__dict__ for k, v in label_lookup.items()}, orient="index")
        .reset_index(drop=True),
        on="sample_id",
        how="left",
    )
    pd.DataFrame([{"scope": "all", "n_spots": int(len(ds)), **sf_overall}]).to_csv(
        out / "sf_overall_metrics.csv",
        index=False,
    )
    sf_slide.to_csv(out / "sf_slide_metrics.csv", index=False)
    pd.DataFrame(method_rows).to_csv(out / "summary.csv", index=False)
    summary = {
        "splits": list(selected_splits),
        "n_spots": int(len(ds)),
        "n_slides": int(len(np.unique(ds.sample_ids))),
        "n_genes": int(len(genes)),
        "methods": method_rows,
        "sf_overall": sf_overall,
        "expression_checkpoint": str(expression_checkpoint),
        "sf_checkpoint": str(sf_checkpoint),
        "outputs": {
            "summary_csv": str(out / "summary.csv"),
            "sf_overall_metrics": str(out / "sf_overall_metrics.csv"),
            "sf_slide_metrics": str(out / "sf_slide_metrics.csv"),
        },
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HistoOmniST in the formal coverage95 benchmark format.")
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--sf-config", default="configs/hest1k_human_visium_sf_current.yaml")
    parser.add_argument("--expression-checkpoint", default=None)
    parser.add_argument("--sf-checkpoint", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out-dir", default="results/hest1k_human_visium_expression/benchmark_results/histoomnist_coverage95")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_config = load_config(args.expression_config)
    sf_config, sf_checkpoint = load_sf_training_config_and_checkpoint(args.sf_config, args.sf_checkpoint)
    expression_checkpoint = resolve_project_path(
        args.expression_checkpoint or default_expression_checkpoint(expression_config)
    )
    if expression_checkpoint is None:
        raise ValueError("Expression checkpoint resolved to None")
    evaluate_histoomnist_benchmark(
        expression_config=expression_config,
        sf_config=sf_config,
        expression_checkpoint=expression_checkpoint,
        sf_checkpoint=sf_checkpoint,
        out_dir=args.out_dir,
        splits=args.splits,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
