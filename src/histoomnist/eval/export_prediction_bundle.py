from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from histoomnist.data.dataset import FeatureStandardizer
from histoomnist.data.spot_table import load_array, load_spot_table
from histoomnist.eval.evaluate_combined import _load_rate_model, _load_sf_model
from histoomnist.eval.metrics import genewise_pearson, sf_metrics, summarize_genewise
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest


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


def _load_genes(base_dir: Path, manifest: pd.DataFrame, cfg: dict) -> list[str]:
    gene_list = cfg.get("data", {}).get("gene_list")
    candidates: list[Path] = []
    if gene_list:
        candidates.append(Path(gene_list))
    first = manifest.iloc[0]
    candidates.append(base_dir / str(first["sample_id"]) / "genes.txt")
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8").splitlines()
    return [f"gene_{i}" for i in range(int(manifest.iloc[0]["n_genes"]))]


def _read_spot_ids(base_dir: Path, row, n: int) -> list[str]:
    explicit = _optional_path(row, "spots_path")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(base_dir / str(explicit))
    candidates.append((base_dir / str(row.features_path)).parent / "spots.txt")
    for path in candidates:
        if path.exists():
            ids = path.read_text(encoding="utf-8").splitlines()
            if len(ids) >= n:
                return ids[:n]
    return [f"spot_{i}" for i in range(n)]


def _as_dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _batched_model_outputs(
    rate_model,
    sf_model,
    features: np.ndarray,
    rate_standardizer: FeatureStandardizer,
    sf_standardizer: FeatureStandardizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_log1p_rate_chunks: list[np.ndarray] = []
    pred_log_sf_chunks: list[np.ndarray] = []
    latent_chunks: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        x_rate = torch.from_numpy(rate_standardizer.transform(features[start:stop])).to(device)
        x_sf = torch.from_numpy(sf_standardizer.transform(features[start:stop])).to(device)
        with torch.no_grad():
            pred_log1p_rate = rate_model(x_rate).detach().cpu().numpy()
            pred_log_sf = sf_model(x_sf).detach().cpu().numpy().reshape(-1)
            if hasattr(rate_model, "encode_spots"):
                latent = rate_model.encode_spots(x_rate).detach().cpu().numpy()
            elif hasattr(rate_model, "hidden"):
                latent = rate_model.hidden(x_rate).detach().cpu().numpy()
            else:
                latent = np.empty((stop - start, 0), dtype=np.float32)
        pred_log1p_rate_chunks.append(pred_log1p_rate.astype(np.float32, copy=False))
        pred_log_sf_chunks.append(pred_log_sf.astype(np.float32, copy=False))
        latent_chunks.append(latent.astype(np.float32, copy=False))
    return (
        np.concatenate(pred_log1p_rate_chunks, axis=0),
        np.concatenate(pred_log_sf_chunks, axis=0),
        np.concatenate(latent_chunks, axis=0),
    )


def export_prediction_bundle(
    sf_config: dict,
    expression_config: dict,
    sf_checkpoint: str | Path,
    expression_checkpoint: str | Path,
    out_dir: str | Path,
    experiment_name: str,
    split_names: list[str] | None = None,
) -> Path:
    device = torch.device(get_device_name(expression_config.get("device")))
    sf_ckpt = load_checkpoint(sf_checkpoint, map_location=str(device))
    rate_ckpt = load_checkpoint(expression_checkpoint, map_location=str(device))
    sf_model = _load_sf_model(sf_config, sf_ckpt, device)
    rate_model = _load_rate_model(expression_config, rate_ckpt, device)
    rate_standardizer = FeatureStandardizer(mean=rate_ckpt["feature_mean"], std=rate_ckpt["feature_std"])
    sf_standardizer = FeatureStandardizer(mean=sf_ckpt["feature_mean"], std=sf_ckpt["feature_std"])

    manifest_path = Path(expression_config["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    splits = split_names or list(expression_config["data"]["test_splits"])
    manifest = manifest[manifest["split"].isin(splits)].copy()
    if manifest.empty:
        raise ValueError(f"No manifest rows for splits={splits}")
    base_dir = manifest_path.parent
    genes = _load_genes(base_dir, manifest, expression_config)

    features_all: list[np.ndarray] = []
    counts_all: list[np.ndarray] = []
    coords_all: list[np.ndarray] = []
    true_sf_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    spot_ids: list[str] = []

    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    for row in manifest.itertuples(index=False):
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
            min_total_counts=min_total_counts,
        )
        mask = table.valid_mask
        features_all.append(table.features[mask].astype(np.float32, copy=False))
        counts_all.append(_as_dense_float32(table.counts[mask]))
        if table.coords is None:
            coords_all.append(np.full((int(mask.sum()), 2), np.nan, dtype=np.float32))
        else:
            coords_all.append(table.coords[mask].astype(np.float32, copy=False))
        true_sf_all.append(table.size_factor[mask].astype(np.float32, copy=False))
        ids = _read_spot_ids(base_dir, row, table.features.shape[0])
        kept = np.asarray(ids, dtype=object)[mask]
        spot_ids.extend([str(x) for x in kept])
        sample_ids.extend([str(row.sample_id)] * int(mask.sum()))

    features = np.concatenate(features_all, axis=0)
    true_count = np.concatenate(counts_all, axis=0).astype(np.float32, copy=False)
    coords = np.concatenate(coords_all, axis=0).astype(np.float32, copy=False)
    true_sf = np.concatenate(true_sf_all, axis=0).astype(np.float32, copy=False)
    true_sf = true_sf / (float(true_sf.mean()) + 1e-8)
    true_rate = true_count / np.clip(true_sf[:, None], 1e-6, None)

    batch_size = int(expression_config["training"].get("batch_size", 128))
    pred_log1p_rate, pred_log_sf_raw, spot_latent = _batched_model_outputs(
        rate_model=rate_model,
        sf_model=sf_model,
        features=features,
        rate_standardizer=rate_standardizer,
        sf_standardizer=sf_standardizer,
        device=device,
        batch_size=batch_size,
    )
    pred_rate = np.expm1(pred_log1p_rate).clip(min=0.0).astype(np.float32, copy=False)
    pred_sf = np.exp(pred_log_sf_raw).astype(np.float32, copy=False)
    pred_sf = pred_sf / (float(pred_sf.mean()) + 1e-8)
    pred_count = (pred_rate * pred_sf[:, None]).astype(np.float32, copy=False)
    pred_count_no_sf = pred_rate.astype(np.float32, copy=False)
    pred_count_oracle_sf = (pred_rate * true_sf[:, None]).astype(np.float32, copy=False)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_rate.npy", pred_rate)
    np.save(out / "pred_sf.npy", pred_sf)
    np.save(out / "pred_log_sf.npy", np.log(pred_sf + 1e-8).astype(np.float32))
    np.save(out / "pred_count.npy", pred_count)
    np.save(out / "pred_count_no_sf.npy", pred_count_no_sf)
    np.save(out / "pred_count_oracle_sf.npy", pred_count_oracle_sf)
    np.save(out / "true_count.npy", true_count)
    np.save(out / "true_rate.npy", true_rate.astype(np.float32, copy=False))
    np.save(out / "true_sf.npy", true_sf)
    np.save(out / "coords.npy", coords)
    np.save(out / "spot_latent.npy", spot_latent)
    (out / "genes.txt").write_text("\n".join(genes[: true_count.shape[1]]) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "spot_id": spot_ids,
            "y": coords[:, 0],
            "x": coords[:, 1],
            "pred_sf": pred_sf,
            "true_sf": true_sf,
        }
    ).to_csv(out / "spots.csv", index=False)

    gene_pearson = pd.DataFrame(
        {
            "gene": genes[: true_count.shape[1]],
            "rate_pearson": genewise_pearson(pred_rate, true_rate),
            "count_no_sf_pearson": genewise_pearson(pred_count_no_sf, true_count),
            "count_pred_sf_pearson": genewise_pearson(pred_count, true_count),
            "count_oracle_sf_pearson": genewise_pearson(pred_count_oracle_sf, true_count),
            "true_mean_count": true_count.mean(axis=0),
            "pred_mean_count": pred_count.mean(axis=0),
            "true_var_count": true_count.var(axis=0),
            "pred_var_count": pred_count.var(axis=0),
        }
    )
    gene_pearson.to_csv(out / "gene_metrics.csv", index=False)
    metrics = {
        "experiment": experiment_name,
        "n_spots": int(true_count.shape[0]),
        "n_genes": int(true_count.shape[1]),
        "sf": sf_metrics(np.log(pred_sf + 1e-8), np.log(true_sf + 1e-8)),
        "rate": summarize_genewise(pred_rate, true_rate),
        "count_no_sf": summarize_genewise(pred_count_no_sf, true_count),
        "count_pred_sf": summarize_genewise(pred_count, true_count),
        "count_oracle_sf": summarize_genewise(pred_count_oracle_sf, true_count),
        "sf_checkpoint": str(sf_checkpoint),
        "expression_checkpoint": str(expression_checkpoint),
        "expression_config": str(expression_config.get("_path", "")),
        "sf_config": str(sf_config.get("_path", "")),
        "splits": list(splits),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[bundle] wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf-config", required=True)
    parser.add_argument("--expression-config", required=True)
    parser.add_argument("--sf-checkpoint", required=True)
    parser.add_argument("--expression-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    args = parser.parse_args()
    sf_cfg = load_config(args.sf_config)
    sf_cfg["_path"] = args.sf_config
    expr_cfg = load_config(args.expression_config)
    expr_cfg["_path"] = args.expression_config
    export_prediction_bundle(
        sf_config=sf_cfg,
        expression_config=expr_cfg,
        sf_checkpoint=args.sf_checkpoint,
        expression_checkpoint=args.expression_checkpoint,
        out_dir=args.out_dir,
        experiment_name=args.experiment_name,
        split_names=args.splits,
    )


if __name__ == "__main__":
    main()
