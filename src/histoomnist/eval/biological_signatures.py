from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from histoomnist.data.dataset import ExpressionRateDataset, FeatureStandardizer
from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.data.spot_table import load_spot_table
from histoomnist.eval.evaluate_combined import _load_rate_model, _load_sf_model
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import project_root, resolve_project_path


DEFAULT_EXPRESSION_CONFIG = "configs/hest1k_human_visium_expression_highconf_symbol95.yaml"
DEFAULT_SF_CONFIG = "configs/hest1k_human_visium_sf_highconf_context_distribution_light.yaml"
DEFAULT_EXPRESSION_CHECKPOINT = "checkpoints/hest1k_human_visium_expression/highconf_symbol95_rate/best.pt"
DEFAULT_SF_CHECKPOINT = "checkpoints/hest1k_human_visium_sf/context_distribution_light_hipt256_leave_slide_out/best.pt"
DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/biological_signatures"


DEFAULT_SIGNATURES: dict[str, list[str]] = {
    "epithelial_tumor": ["EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "MUC1", "ERBB2", "MSLN"],
    "pan_immune": ["PTPRC", "LST1", "LYZ", "CD68", "CD3D", "CD3E", "CD8A", "MS4A1"],
    "t_cell": ["CD3D", "CD3E", "CD2", "CD247", "TRAC", "CD8A", "CD4"],
    "b_cell": ["MS4A1", "CD79A", "CD79B", "CD74", "BANK1"],
    "myeloid": ["LYZ", "LST1", "CD68", "FCGR3A", "S100A8", "S100A9"],
    "stromal_ecm": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "VIM", "ACTA2", "TAGLN"],
    "endothelial": ["PECAM1", "VWF", "KDR", "ENG", "PLVAP", "RAMP2"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "UBE2C", "CENPF"],
    "hypoxia_stress": ["VEGFA", "CA9", "LDHA", "ENO1", "HIF1A", "BNIP3", "NDRG1"],
    "interferon_response": ["ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "STAT1", "CXCL10"],
}


@dataclass
class PairAccumulator:
    n: int = 0
    sum_pred: float = 0.0
    sum_true: float = 0.0
    sum_pred2: float = 0.0
    sum_true2: float = 0.0
    sum_pred_true: float = 0.0
    sum_abs_error: float = 0.0
    sum_sq_error: float = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray, valid: np.ndarray) -> None:
        keep = valid.astype(bool) & np.isfinite(pred) & np.isfinite(true)
        if not np.any(keep):
            return
        x = np.asarray(pred[keep], dtype=np.float64)
        y = np.asarray(true[keep], dtype=np.float64)
        err = x - y
        self.n += int(x.size)
        self.sum_pred += float(x.sum())
        self.sum_true += float(y.sum())
        self.sum_pred2 += float(np.sum(x * x))
        self.sum_true2 += float(np.sum(y * y))
        self.sum_pred_true += float(np.sum(x * y))
        self.sum_abs_error += float(np.sum(np.abs(err)))
        self.sum_sq_error += float(np.sum(err * err))

    def metrics(self, prefix: str) -> dict[str, float | int]:
        if self.n <= 0:
            return {
                f"{prefix}_n_spots": 0,
                f"{prefix}_pearson": float("nan"),
                f"{prefix}_mae": float("nan"),
                f"{prefix}_rmse": float("nan"),
                f"{prefix}_true_mean": float("nan"),
                f"{prefix}_pred_mean": float("nan"),
                f"{prefix}_true_std": float("nan"),
                f"{prefix}_pred_std": float("nan"),
            }
        n = float(self.n)
        numerator = self.sum_pred_true - (self.sum_pred * self.sum_true / n)
        pred_var = self.sum_pred2 - (self.sum_pred * self.sum_pred / n)
        true_var = self.sum_true2 - (self.sum_true * self.sum_true / n)
        denom = math.sqrt(max(pred_var, 0.0) * max(true_var, 0.0))
        pearson = float(numerator / denom) if self.n >= 3 and denom > 0 else float("nan")
        return {
            f"{prefix}_n_spots": int(self.n),
            f"{prefix}_pearson": pearson,
            f"{prefix}_mae": float(self.sum_abs_error / n),
            f"{prefix}_rmse": float(math.sqrt(self.sum_sq_error / n)),
            f"{prefix}_true_mean": float(self.sum_true / n),
            f"{prefix}_pred_mean": float(self.sum_pred / n),
            f"{prefix}_true_std": float(math.sqrt(max(true_var / n, 0.0))),
            f"{prefix}_pred_std": float(math.sqrt(max(pred_var / n, 0.0))),
        }


def rel_project_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_signature_table(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {name: list(genes) for name, genes in DEFAULT_SIGNATURES.items()}
    frame = pd.read_csv(path)
    required = {"signature", "gene"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Signature table missing columns: {missing}")
    signatures: dict[str, list[str]] = {}
    for signature, group in frame.groupby("signature", sort=True):
        genes = [str(g).strip() for g in group["gene"] if str(g).strip()]
        if genes:
            signatures[str(signature)] = genes
    if not signatures:
        raise ValueError(f"No signatures found in {path}")
    return signatures


def signature_index_rows(signatures: dict[str, list[str]], gene_names: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    rows = []
    indices = {}
    for name, genes in signatures.items():
        present = [gene for gene in genes if gene in gene_to_idx]
        missing = [gene for gene in genes if gene not in gene_to_idx]
        idx = np.asarray([gene_to_idx[gene] for gene in present], dtype=np.int64)
        indices[name] = idx
        rows.append(
            {
                "signature": name,
                "n_target_genes": len(genes),
                "n_present_genes": len(present),
                "present_genes": "|".join(present),
                "missing_genes": "|".join(missing),
            }
        )
    return pd.DataFrame(rows), indices


def signature_scores(
    values: np.ndarray,
    measured: np.ndarray,
    indices: np.ndarray,
    *,
    min_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        n_rows = values.shape[0]
        return np.full(n_rows, np.nan, dtype=np.float32), np.zeros(n_rows, dtype=bool)
    sub_values = values[:, indices]
    sub_measured = measured[:, indices].astype(bool)
    n_measured = sub_measured.sum(axis=1)
    valid = n_measured >= int(min_genes)
    score = np.full(values.shape[0], np.nan, dtype=np.float32)
    denom = np.maximum(n_measured, 1)
    score[valid] = (np.where(sub_measured, sub_values, 0.0).sum(axis=1) / denom)[valid]
    return score, valid


def build_coords_for_dataset_order(
    *,
    manifest: pd.DataFrame,
    manifest_base: Path,
    splits: list[str],
    min_total_counts: float,
) -> np.ndarray:
    coords: list[np.ndarray] = []
    rows = manifest[manifest["split"].isin(splits)].copy()
    for row in rows.itertuples(index=False):
        table = load_spot_table(
            sample_id=str(row.sample_id),
            features_path=manifest_base / str(row.features_path),
            counts_path=manifest_base / str(row.counts_path),
            coords_path=manifest_base / str(row.coords_path) if hasattr(row, "coords_path") else None,
            size_factor_path=manifest_base / str(row.size_factor_path) if hasattr(row, "size_factor_path") else None,
            min_total_counts=min_total_counts,
        )
        mask = table.valid_mask
        if table.coords is None:
            coords.append(np.full((int(mask.sum()), 2), np.nan, dtype=np.float32))
        else:
            coords.append(table.coords[mask].astype(np.float32))
    if not coords:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(coords, axis=0)


def normalise_predicted_sf(pred_log_sf: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    pred_sf = np.exp(pred_log_sf.astype(np.float32, copy=False)).reshape(-1)
    for sample_id in np.unique(sample_ids):
        idx = sample_ids == sample_id
        mean = float(pred_sf[idx].mean())
        if mean > 0 and np.isfinite(mean):
            pred_sf[idx] = pred_sf[idx] / mean
    return pred_sf.astype(np.float32)


def metric_rows(
    accumulators: dict[tuple[str, str, str], PairAccumulator],
    *,
    level: str,
    key_names: list[str],
) -> pd.DataFrame:
    rows = []
    for key, acc in sorted(accumulators.items()):
        metric_kind, signature, *group_values = key
        row: dict[str, Any] = {
            "level": level,
            "metric_kind": metric_kind,
            "signature": signature,
        }
        for name, value in zip(key_names, group_values):
            row[name] = value
        row.update(acc.metrics(metric_kind))
        rows.append(row)
    return pd.DataFrame(rows)


def update_overall(
    accumulators: dict[tuple[str, str, str], PairAccumulator],
    *,
    prefix: str,
    signature: str,
    pred: np.ndarray,
    true: np.ndarray,
    valid: np.ndarray,
) -> None:
    accumulators[(prefix, signature, "overall")].update(pred, true, valid)


def update_by_slide(
    accumulators: dict[tuple[str, str, str], PairAccumulator],
    *,
    prefix: str,
    signature: str,
    pred: np.ndarray,
    true: np.ndarray,
    valid: np.ndarray,
    sample_ids: np.ndarray,
) -> None:
    for sample_id in np.unique(sample_ids):
        idx = sample_ids == sample_id
        accumulators[(prefix, signature, str(sample_id))].update(pred[idx], true[idx], valid[idx])


def update_by_organ(
    accumulators: dict[tuple[str, str, str], PairAccumulator],
    *,
    prefix: str,
    signature: str,
    pred: np.ndarray,
    true: np.ndarray,
    valid: np.ndarray,
    sample_ids: np.ndarray,
    organ_by_sample: dict[str, str],
) -> None:
    organs = np.asarray([organ_by_sample.get(str(sample_id), "unknown") for sample_id in sample_ids])
    for organ in np.unique(organs):
        idx = organs == organ
        accumulators[(prefix, signature, str(organ))].update(pred[idx], true[idx], valid[idx])


def plot_signature_map(
    *,
    sample_id: str,
    signature: str,
    coords: np.ndarray,
    true_score: np.ndarray,
    pred_score: np.ndarray,
    valid: np.ndarray,
    out_path: Path,
) -> bool:
    keep = valid & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if keep.sum() < 3:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = coords[keep, 0]
    y = coords[keep, 1]
    true = true_score[keep]
    pred = pred_score[keep]
    resid = pred - true
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), dpi=160)
    specs = [
        (true, "true", "viridis"),
        (pred, "predicted", "viridis"),
        (resid, "pred - true", "coolwarm"),
    ]
    for ax, (values, title, cmap) in zip(axes, specs):
        sc = ax.scatter(x, y, c=values, s=7, cmap=cmap, linewidths=0)
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{sample_id} {signature} signature score", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def evaluate_biological_signatures(
    *,
    expression_config: dict[str, Any],
    sf_config: dict[str, Any],
    expression_config_path: Path | None,
    sf_config_path: Path | None,
    expression_checkpoint: Path,
    sf_checkpoint: Path,
    signatures: dict[str, list[str]],
    out_dir: Path,
    splits: list[str],
    min_signature_genes: int,
    batch_size: int | None,
    max_batches: int | None,
    max_overlay_slides: int,
    max_overlay_signatures: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(get_device_name(expression_config.get("device")))
    expression_ckpt = load_checkpoint(expression_checkpoint, map_location=str(device))
    sf_ckpt = load_checkpoint(sf_checkpoint, map_location=str(device))
    manifest_path = Path(expression_config["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    gene_names, gene_indices = selected_genes_from_config(expression_config, base_dir=manifest_path.parent)
    if gene_names is None:
        raise ValueError("Biological signature analysis requires data.gene_names_path.")
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    signature_table, signature_indices = signature_index_rows(signatures, gene_names)
    signature_table.to_csv(out_dir / "signature_gene_coverage.csv", index=False)

    ds = ExpressionRateDataset(
        manifest,
        base_dir=manifest_path.parent,
        splits=splits,
        min_total_counts=min_total_counts,
        standardizer=FeatureStandardizer(mean=expression_ckpt["feature_mean"], std=expression_ckpt["feature_std"]),
        gene_names=gene_names,
        gene_indices=gene_indices,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    coords = build_coords_for_dataset_order(
        manifest=manifest,
        manifest_base=manifest_path.parent,
        splits=splits,
        min_total_counts=min_total_counts,
    )
    if coords.shape[0] != len(ds):
        raise ValueError(f"Coordinate row mismatch: coords={coords.shape[0]}, dataset={len(ds)}")

    rate_model = _load_rate_model(expression_config, expression_ckpt, device)
    sf_model = _load_sf_model(sf_config, sf_ckpt, device)
    sf_standardizer = FeatureStandardizer(mean=sf_ckpt["feature_mean"], std=sf_ckpt["feature_std"])
    loader = DataLoader(ds, batch_size=batch_size or int(expression_config["training"]["batch_size"]), shuffle=False)

    sample_ids = ds.sample_ids
    organ_by_sample = (
        manifest.drop_duplicates("sample_id").set_index("sample_id")["organ"].fillna("unknown").astype(str).to_dict()
        if "organ" in manifest.columns
        else {}
    )

    overall_acc: dict[tuple[str, str, str], PairAccumulator] = defaultdict(PairAccumulator)
    slide_acc: dict[tuple[str, str, str], PairAccumulator] = defaultdict(PairAccumulator)
    organ_acc: dict[tuple[str, str, str], PairAccumulator] = defaultdict(PairAccumulator)
    spot_rows: list[pd.DataFrame] = []

    pred_log_sf_full = np.empty(len(ds), dtype=np.float32)
    sf_offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > int(max_batches):
                break
            raw_features = batch["raw_features"].numpy()
            pred_log_sf = sf_model(torch.from_numpy(sf_standardizer.transform(raw_features)).to(device)).cpu().numpy()
            stop = sf_offset + pred_log_sf.shape[0]
            pred_log_sf_full[sf_offset:stop] = pred_log_sf.reshape(-1).astype(np.float32)
            sf_offset = stop
    pred_sf_full = normalise_predicted_sf(pred_log_sf_full[:sf_offset], sample_ids[:sf_offset])

    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > int(max_batches):
                break
            features = batch["features"].to(device)
            pred_log1p_rate = rate_model(features).cpu().numpy().astype(np.float32)
            true_log1p_rate = batch["log1p_rate"].numpy().astype(np.float32)
            measured = batch["expression_mask"].numpy().astype(bool)
            batch_size_actual = pred_log1p_rate.shape[0]
            stop = offset + batch_size_actual
            batch_sample_ids = sample_ids[offset:stop]
            pred_sf = pred_sf_full[offset:stop]
            true_sf = np.exp(batch["true_log_sf"].numpy().reshape(-1)).astype(np.float32)
            pred_rate = np.expm1(pred_log1p_rate).clip(min=0.0)
            true_rate = np.expm1(true_log1p_rate).clip(min=0.0)

            for signature, indices in signature_indices.items():
                true_rate_score, valid = signature_scores(
                    true_log1p_rate,
                    measured,
                    indices,
                    min_genes=min_signature_genes,
                )
                pred_rate_score, _ = signature_scores(
                    pred_log1p_rate,
                    measured,
                    indices,
                    min_genes=min_signature_genes,
                )
                true_count_log = np.log1p(true_rate[:, indices] * true_sf[:, None]) if indices.size else true_rate[:, []]
                pred_count_log = np.log1p(pred_rate[:, indices] * pred_sf[:, None]) if indices.size else pred_rate[:, []]
                true_count_score, count_valid = signature_scores(
                    true_count_log,
                    measured[:, indices] if indices.size else measured[:, []],
                    np.arange(indices.size, dtype=np.int64),
                    min_genes=min_signature_genes,
                )
                pred_count_score, _ = signature_scores(
                    pred_count_log,
                    measured[:, indices] if indices.size else measured[:, []],
                    np.arange(indices.size, dtype=np.int64),
                    min_genes=min_signature_genes,
                )
                update_overall(
                    overall_acc,
                    prefix="rate",
                    signature=signature,
                    pred=pred_rate_score,
                    true=true_rate_score,
                    valid=valid,
                )
                update_by_slide(
                    slide_acc,
                    prefix="rate",
                    signature=signature,
                    pred=pred_rate_score,
                    true=true_rate_score,
                    valid=valid,
                    sample_ids=batch_sample_ids,
                )
                update_by_organ(
                    organ_acc,
                    prefix="rate",
                    signature=signature,
                    pred=pred_rate_score,
                    true=true_rate_score,
                    valid=valid,
                    sample_ids=batch_sample_ids,
                    organ_by_sample=organ_by_sample,
                )
                update_overall(
                    overall_acc,
                    prefix="count_pred_sf",
                    signature=signature,
                    pred=pred_count_score,
                    true=true_count_score,
                    valid=count_valid,
                )
                update_by_slide(
                    slide_acc,
                    prefix="count_pred_sf",
                    signature=signature,
                    pred=pred_count_score,
                    true=true_count_score,
                    valid=count_valid,
                    sample_ids=batch_sample_ids,
                )
                update_by_organ(
                    organ_acc,
                    prefix="count_pred_sf",
                    signature=signature,
                    pred=pred_count_score,
                    true=true_count_score,
                    valid=count_valid,
                    sample_ids=batch_sample_ids,
                    organ_by_sample=organ_by_sample,
                )
                spot_rows.append(
                    pd.DataFrame(
                        {
                            "row_index": np.arange(offset, stop, dtype=np.int64),
                            "sample_id": batch_sample_ids,
                            "signature": signature,
                            "rate_true": true_rate_score,
                            "rate_pred": pred_rate_score,
                            "rate_valid": valid,
                            "count_pred_sf_true": true_count_score,
                            "count_pred_sf_pred": pred_count_score,
                            "count_pred_sf_valid": count_valid,
                        }
                    )
                )
            offset = stop

    spot_scores = pd.concat(spot_rows, ignore_index=True) if spot_rows else pd.DataFrame()
    spot_scores_path = out_dir / "spot_signature_scores.csv"
    spot_scores.to_csv(spot_scores_path, index=False)

    overall = metric_rows(overall_acc, level="overall", key_names=["group"])
    by_slide = metric_rows(slide_acc, level="slide", key_names=["sample_id"])
    by_organ = metric_rows(organ_acc, level="organ", key_names=["organ"])
    overall = overall.merge(signature_table, on="signature", how="left")
    by_slide = by_slide.merge(signature_table, on="signature", how="left")
    by_organ = by_organ.merge(signature_table, on="signature", how="left")
    overall.to_csv(out_dir / "signature_summary.csv", index=False)
    by_slide.to_csv(out_dir / "signature_by_slide.csv", index=False)
    by_organ.to_csv(out_dir / "signature_by_organ.csv", index=False)

    overlay_rows = []
    plotted = 0
    rate_overall = overall[overall["metric_kind"].eq("rate")].sort_values("rate_pearson", ascending=False)
    overlay_signatures = rate_overall["signature"].head(int(max_overlay_signatures)).tolist()
    evaluated_sample_ids = sample_ids[:offset]
    evaluated_coords = coords[:offset]
    overlay_sample_ids = list(dict.fromkeys(evaluated_sample_ids.tolist()))[: int(max_overlay_slides)]
    for sample_id in overlay_sample_ids:
        sample_mask = evaluated_sample_ids == sample_id
        sample_coords = evaluated_coords[sample_mask]
        for signature in overlay_signatures:
            rows = spot_scores[spot_scores["sample_id"].eq(sample_id) & spot_scores["signature"].eq(signature)]
            if rows.empty:
                continue
            out_path = out_dir / "spatial_signature_maps" / f"{sample_id}_{signature}.png"
            ok = plot_signature_map(
                sample_id=str(sample_id),
                signature=str(signature),
                coords=sample_coords,
                true_score=rows["rate_true"].to_numpy(dtype=np.float32),
                pred_score=rows["rate_pred"].to_numpy(dtype=np.float32),
                valid=rows["rate_valid"].to_numpy(dtype=bool),
                out_path=out_path,
            )
            overlay_rows.append(
                {
                    "sample_id": str(sample_id),
                    "signature": str(signature),
                    "path": rel_project_path(out_path),
                    "written": bool(ok),
                }
            )
            plotted += int(ok)
    overlay_manifest = pd.DataFrame(overlay_rows)
    overlay_manifest.to_csv(out_dir / "spatial_signature_map_manifest.csv", index=False)

    run_summary = {
        "expression_config": rel_project_path(expression_config_path or DEFAULT_EXPRESSION_CONFIG),
        "sf_config": rel_project_path(sf_config_path or DEFAULT_SF_CONFIG),
        "expression_checkpoint": rel_project_path(expression_checkpoint),
        "sf_checkpoint": rel_project_path(sf_checkpoint),
        "splits": splits,
        "n_spots_evaluated": int(offset),
        "n_dataset_spots": int(len(ds)),
        "n_signatures": int(len(signatures)),
        "min_signature_genes": int(min_signature_genes),
        "max_batches": None if max_batches is None else int(max_batches),
        "is_truncated": bool(max_batches is not None and offset < len(ds)),
        "n_spatial_maps_written": int(plotted),
        "outputs": {
            "signature_gene_coverage": rel_project_path(out_dir / "signature_gene_coverage.csv"),
            "signature_summary": rel_project_path(out_dir / "signature_summary.csv"),
            "signature_by_slide": rel_project_path(out_dir / "signature_by_slide.csv"),
            "signature_by_organ": rel_project_path(out_dir / "signature_by_organ.csv"),
            "spot_signature_scores": rel_project_path(spot_scores_path),
            "spatial_signature_map_manifest": rel_project_path(out_dir / "spatial_signature_map_manifest.csv"),
        },
    }
    write_json(out_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2), flush=True)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coverage95 biological marker/signature fidelity.")
    parser.add_argument("--expression-config", default=DEFAULT_EXPRESSION_CONFIG)
    parser.add_argument("--sf-config", default=DEFAULT_SF_CONFIG)
    parser.add_argument("--expression-checkpoint", default=DEFAULT_EXPRESSION_CHECKPOINT)
    parser.add_argument("--sf-checkpoint", default=DEFAULT_SF_CHECKPOINT)
    parser.add_argument("--signature-table", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--min-signature-genes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-overlay-slides", type=int, default=3)
    parser.add_argument("--max-overlay-signatures", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_config_path = resolve_project_path(args.expression_config)
    sf_config_path = resolve_project_path(args.sf_config)
    expression_checkpoint = resolve_project_path(args.expression_checkpoint)
    sf_checkpoint = resolve_project_path(args.sf_checkpoint)
    signature_table = resolve_project_path(args.signature_table)
    out_dir = resolve_project_path(args.out_dir)
    if None in (expression_config_path, sf_config_path, expression_checkpoint, sf_checkpoint, out_dir):
        raise ValueError("Required paths did not resolve.")
    evaluate_biological_signatures(
        expression_config=load_config(expression_config_path),
        sf_config=load_config(sf_config_path),
        expression_config_path=expression_config_path,
        sf_config_path=sf_config_path,
        expression_checkpoint=expression_checkpoint,
        sf_checkpoint=sf_checkpoint,
        signatures=load_signature_table(signature_table),
        out_dir=out_dir,
        splits=[str(x) for x in args.splits],
        min_signature_genes=int(args.min_signature_genes),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_overlay_slides=int(args.max_overlay_slides),
        max_overlay_signatures=int(args.max_overlay_signatures),
    )


if __name__ == "__main__":
    main()
