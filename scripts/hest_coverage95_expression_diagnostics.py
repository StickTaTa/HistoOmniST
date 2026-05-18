from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.data.dataset import ExpressionRateDataset, FeatureStandardizer  # noqa: E402
from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config  # noqa: E402
from histoomnist.data.spot_table import load_spot_table  # noqa: E402
from histoomnist.eval.evaluate_combined import _load_rate_model, _load_sf_model  # noqa: E402
from histoomnist.eval.metrics import sf_metrics  # noqa: E402
from histoomnist.hest.raw_assets import raw_slide_paths  # noqa: E402
from histoomnist.train.common import load_checkpoint  # noqa: E402
from histoomnist.utils.config import get_device_name, load_config  # noqa: E402
from histoomnist.utils.io import read_manifest  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


GENE_CLASS_MARKERS = {
    "epithelial_tumor": {
        "EPCAM",
        "KRT7",
        "KRT8",
        "KRT18",
        "KRT19",
        "MUC1",
        "ERBB2",
        "MSLN",
    },
    "immune": {
        "PTPRC",
        "CD3D",
        "CD3E",
        "CD4",
        "CD8A",
        "CD79A",
        "MS4A1",
        "LST1",
        "LYZ",
        "CD68",
    },
    "stromal_ecm": {
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "ACTA2",
        "TAGLN",
        "VIM",
    },
    "endothelial": {
        "PECAM1",
        "VWF",
        "KDR",
        "ENG",
        "PLVAP",
        "RAMP2",
    },
    "proliferation": {
        "MKI67",
        "TOP2A",
        "PCNA",
        "UBE2C",
        "HMGB2",
        "CENPF",
    },
    "housekeeping_stress": {
        "ACTB",
        "GAPDH",
        "B2M",
        "FTH1",
        "FTL",
        "HSPB1",
        "HSPA1A",
        "JUN",
        "FOS",
    },
}


@dataclass
class DiagnosticsContext:
    expression_config: dict
    sf_config: dict
    expression_checkpoint: Path
    sf_checkpoint: Path
    out_dir: Path
    splits: list[str]
    batch_size: int
    device: torch.device
    overlay_sample_id: str | None
    max_overlay_genes: int


class VectorMetricAccumulator:
    def __init__(self, n_features: int):
        self.n = np.zeros(n_features, dtype=np.float64)
        self.sum_pred = np.zeros(n_features, dtype=np.float64)
        self.sum_true = np.zeros(n_features, dtype=np.float64)
        self.sum_pred2 = np.zeros(n_features, dtype=np.float64)
        self.sum_true2 = np.zeros(n_features, dtype=np.float64)
        self.sum_pred_true = np.zeros(n_features, dtype=np.float64)
        self.sum_abs_error = np.zeros(n_features, dtype=np.float64)
        self.sum_sq_error = np.zeros(n_features, dtype=np.float64)
        self.nonzero_true = np.zeros(n_features, dtype=np.float64)

    def update(self, pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> None:
        if pred.shape != true.shape or pred.shape != mask.shape:
            raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}, mask={mask.shape}")
        valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(true)
        pred64 = np.where(valid, pred, 0.0).astype(np.float64)
        true64 = np.where(valid, true, 0.0).astype(np.float64)
        err64 = pred64 - true64
        self.n += valid.sum(axis=0)
        self.sum_pred += pred64.sum(axis=0)
        self.sum_true += true64.sum(axis=0)
        self.sum_pred2 += (pred64 * pred64).sum(axis=0)
        self.sum_true2 += (true64 * true64).sum(axis=0)
        self.sum_pred_true += (pred64 * true64).sum(axis=0)
        self.sum_abs_error += np.where(valid, np.abs(err64), 0.0).sum(axis=0)
        self.sum_sq_error += np.where(valid, err64 * err64, 0.0).sum(axis=0)
        self.nonzero_true += (valid & (true > 0)).sum(axis=0)

    def to_frame(self, prefix: str, genes: list[str]) -> pd.DataFrame:
        denom_n = np.maximum(self.n, 1.0)
        numerator = self.sum_pred_true - (self.sum_pred * self.sum_true / denom_n)
        pred_var = self.sum_pred2 - (self.sum_pred * self.sum_pred / denom_n)
        true_var = self.sum_true2 - (self.sum_true * self.sum_true / denom_n)
        denom = np.sqrt(np.maximum(pred_var, 0.0) * np.maximum(true_var, 0.0))
        pearson = np.full(len(genes), np.nan, dtype=np.float64)
        keep = (self.n >= 3) & (denom > 0)
        pearson[keep] = numerator[keep] / denom[keep]
        return pd.DataFrame(
            {
                "gene": genes,
                f"{prefix}_n_obs": self.n.astype(np.int64),
                f"{prefix}_pearson": pearson,
                f"{prefix}_mae": self.sum_abs_error / denom_n,
                f"{prefix}_rmse": np.sqrt(self.sum_sq_error / denom_n),
                f"{prefix}_true_mean": self.sum_true / denom_n,
                f"{prefix}_pred_mean": self.sum_pred / denom_n,
                f"{prefix}_true_std": np.sqrt(np.maximum(true_var / denom_n, 0.0)),
                f"{prefix}_pred_std": np.sqrt(np.maximum(pred_var / denom_n, 0.0)),
                f"{prefix}_detected_fraction": self.nonzero_true / denom_n,
            }
        )

    def summary(self) -> dict[str, float]:
        frame = self.to_frame("metric", [str(i) for i in range(len(self.n))])
        vals = frame["metric_pearson"].to_numpy(dtype=np.float64)
        return {
            "mean_gene_pearson": float(np.nanmean(vals)),
            "median_gene_pearson": float(np.nanmedian(vals)),
            "valid_genes": int(np.isfinite(vals).sum()),
        }


class ScalarMetricAccumulator:
    def __init__(self):
        self.n = 0
        self.sum_pred = 0.0
        self.sum_true = 0.0
        self.sum_pred2 = 0.0
        self.sum_true2 = 0.0
        self.sum_pred_true = 0.0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> None:
        valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(true)
        if not np.any(valid):
            return
        x = pred[valid].astype(np.float64, copy=False)
        y = true[valid].astype(np.float64, copy=False)
        err = x - y
        self.n += int(x.size)
        self.sum_pred += float(x.sum())
        self.sum_true += float(y.sum())
        self.sum_pred2 += float(np.sum(x * x))
        self.sum_true2 += float(np.sum(y * y))
        self.sum_pred_true += float(np.sum(x * y))
        self.sum_abs_error += float(np.sum(np.abs(err)))
        self.sum_sq_error += float(np.sum(err * err))

    def row(self) -> dict[str, float | int]:
        if self.n <= 0:
            return {
                "n_values": 0,
                "pearson": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "true_mean": float("nan"),
                "pred_mean": float("nan"),
                "true_std": float("nan"),
                "pred_std": float("nan"),
            }
        n = float(self.n)
        numerator = self.sum_pred_true - (self.sum_pred * self.sum_true / n)
        pred_var = self.sum_pred2 - (self.sum_pred * self.sum_pred / n)
        true_var = self.sum_true2 - (self.sum_true * self.sum_true / n)
        denom = np.sqrt(max(pred_var, 0.0) * max(true_var, 0.0))
        pearson = float(numerator / denom) if self.n >= 3 and denom > 0 else float("nan")
        return {
            "n_values": int(self.n),
            "pearson": pearson,
            "mae": float(self.sum_abs_error / n),
            "rmse": float(np.sqrt(self.sum_sq_error / n)),
            "true_mean": float(self.sum_true / n),
            "pred_mean": float(self.sum_pred / n),
            "true_std": float(np.sqrt(max(true_var / n, 0.0))),
            "pred_std": float(np.sqrt(max(pred_var / n, 0.0))),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate coverage95 HEST expression diagnostics for the fixed H&E-to-ST model."
    )
    parser.add_argument("--expression-config", type=Path, default=Path("configs/hest1k_human_visium_expression_highconf_symbol95.yaml"))
    parser.add_argument("--sf-config", type=Path, default=Path("configs/hest1k_human_visium_sf_current.yaml"))
    parser.add_argument("--expression-checkpoint", type=Path, default=None)
    parser.add_argument("--sf-checkpoint", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overlay-sample-id", default="MISC33")
    parser.add_argument("--max-overlay-genes", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("results/hest1k_human_visium_expression/coverage95_diagnostics"))
    return parser.parse_args()


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


def load_sf_training_config_and_checkpoint(args: argparse.Namespace) -> tuple[dict, Path]:
    pointer_or_cfg_path = resolve_project_path(args.sf_config)
    if pointer_or_cfg_path is None:
        raise ValueError("--sf-config cannot be empty")
    pointer_or_cfg = load_config(pointer_or_cfg_path)
    model_cfg = pointer_or_cfg.get("model", {})
    training_config = model_cfg.get("training_config")
    if training_config:
        sf_config_path = resolve_project_path(training_config)
        if sf_config_path is None:
            raise ValueError("SF training_config resolved to None")
        sf_config = load_config(sf_config_path)
    else:
        sf_config = pointer_or_cfg
    checkpoint = args.sf_checkpoint or model_cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Missing SF checkpoint. Pass --sf-checkpoint or use a pointer config with model.checkpoint.")
    sf_checkpoint = resolve_project_path(checkpoint)
    if sf_checkpoint is None:
        raise ValueError("SF checkpoint resolved to None")
    return sf_config, sf_checkpoint


def default_expression_checkpoint(cfg: dict) -> Path:
    output_dir = cfg.get("output", {}).get("dir")
    if not output_dir:
        checkpoint_root = cfg.get("paths", {}).get("checkpoint_root", "checkpoints")
        output_dir = Path(checkpoint_root) / str(cfg["project"]["name"])
    return Path(output_dir) / "best.pt"


def load_context(args: argparse.Namespace) -> DiagnosticsContext:
    expression_config_path = resolve_project_path(args.expression_config)
    if expression_config_path is None:
        raise ValueError("--expression-config cannot be empty")
    expression_config = load_config(expression_config_path)
    sf_config, sf_checkpoint = load_sf_training_config_and_checkpoint(args)
    expression_checkpoint = resolve_project_path(args.expression_checkpoint or default_expression_checkpoint(expression_config))
    if expression_checkpoint is None:
        raise ValueError("Expression checkpoint resolved to None")
    splits = args.splits or list(expression_config["data"]["test_splits"])
    batch_size = int(args.batch_size or expression_config["training"]["batch_size"])
    device_name = args.device or expression_config.get("device")
    device = torch.device(get_device_name(device_name))
    return DiagnosticsContext(
        expression_config=expression_config,
        sf_config=sf_config,
        expression_checkpoint=expression_checkpoint,
        sf_checkpoint=sf_checkpoint,
        out_dir=resolve_project_path(args.out_dir) or args.out_dir,
        splits=[str(split) for split in splits],
        batch_size=batch_size,
        device=device,
        overlay_sample_id=None if args.overlay_sample_id in (None, "") else str(args.overlay_sample_id),
        max_overlay_genes=int(args.max_overlay_genes),
    )


def marker_class_for_gene(gene: str) -> str:
    for gene_class, markers in GENE_CLASS_MARKERS.items():
        if gene in markers:
            return gene_class
    return "other"


def sample_metadata(manifest: pd.DataFrame) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for row in manifest.itertuples(index=False):
        meta[str(row.sample_id)] = {
            "split": str(getattr(row, "split", "")),
            "organ": str(getattr(row, "organ", "")),
            "cohort": str(getattr(row, "cohort", "")),
            "disease_state": str(getattr(row, "disease_state", "")),
        }
    return meta


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
    for sample_id in np.unique(ds.sample_ids):
        idx = ds.sample_ids == sample_id
        pred_sf[idx] = pred_sf[idx] / (float(pred_sf[idx].mean()) + 1.0e-8)
    rows = []
    for sample_id in np.unique(ds.sample_ids):
        idx = ds.sample_ids == sample_id
        metrics = sf_metrics(np.log(pred_sf[idx] + 1.0e-8), true_log_sf[idx])
        rows.append({"sample_id": str(sample_id), "n_spots": int(idx.sum()), **metrics})
    return pred_sf, true_log_sf.reshape(-1), pd.DataFrame(rows)


def update_group_accumulators(
    *,
    accumulators: dict[tuple[str, str], ScalarMetricAccumulator],
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    sample_ids: np.ndarray,
    sample_meta: dict[str, dict[str, str]],
    metric_name: str,
) -> None:
    for sample_id in np.unique(sample_ids):
        spot_mask = sample_ids == sample_id
        labels = sample_meta[str(sample_id)]
        group_keys = {
            f"overall|{labels['split']}": "overall",
            f"organ|{labels['split']}|{labels['organ']}": "organ",
            f"slide|{labels['split']}|{sample_id}": "slide",
        }
        for group_key in group_keys:
            acc = accumulators[(metric_name, group_key)]
            acc.update(pred[spot_mask], true[spot_mask], mask[spot_mask])


def group_rows(
    accumulators: dict[tuple[str, str], ScalarMetricAccumulator],
    sample_meta: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slide_lookup = {sample_id: meta for sample_id, meta in sample_meta.items()}
    overall_rows = []
    organ_rows = []
    slide_rows = []
    for (metric_name, group_key), acc in sorted(accumulators.items()):
        parts = group_key.split("|")
        row = {"metric": metric_name, **acc.row()}
        if parts[0] == "overall":
            row.update({"split": parts[1], "level": "overall"})
            overall_rows.append(row)
        elif parts[0] == "organ":
            row.update({"split": parts[1], "organ": parts[2], "level": "organ"})
            organ_rows.append(row)
        elif parts[0] == "slide":
            sample_id = parts[2]
            meta = slide_lookup[sample_id]
            row.update(
                {
                    "split": parts[1],
                    "sample_id": sample_id,
                    "organ": meta["organ"],
                    "cohort": meta["cohort"],
                    "disease_state": meta["disease_state"],
                    "level": "slide",
                }
            )
            slide_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(organ_rows), pd.DataFrame(slide_rows)


def build_per_gene_frame(
    *,
    genes: list[str],
    accumulators: dict[str, VectorMetricAccumulator],
    selection_report: Path | None,
) -> pd.DataFrame:
    frames = []
    for name, acc in accumulators.items():
        frame = acc.to_frame(name, genes).drop(columns=["gene"])
        frames.append(frame)
    out = pd.DataFrame({"gene": genes, "gene_index": np.arange(len(genes), dtype=np.int64)})
    for frame in frames:
        out = pd.concat([out, frame], axis=1)
    out["gene_class"] = [marker_class_for_gene(gene) for gene in genes]
    if selection_report is not None and selection_report.exists():
        selection = pd.read_csv(selection_report)
        rename = {col: f"selection_{col}" for col in selection.columns if col != "gene"}
        selection = selection.rename(columns=rename)
        out = out.merge(selection, on="gene", how="left")
    return out


def summarize_gene_classes(per_gene: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "rate_pearson",
        "count_no_sf_pearson",
        "count_pred_sf_pearson",
        "count_oracle_sf_pearson",
        "count_pred_sf_mae",
        "count_pred_sf_rmse",
        "count_pred_sf_detected_fraction",
    ]
    for gene_class, group in per_gene.groupby("gene_class", dropna=False):
        row: dict[str, object] = {
            "gene_class": gene_class,
            "n_genes": int(len(group)),
            "n_valid_count_pred_sf": int(np.isfinite(group["count_pred_sf_pearson"]).sum()),
        }
        for col in metric_cols:
            vals = group[col].to_numpy(dtype=np.float64)
            row[f"mean_{col}"] = float(np.nanmean(vals))
            row[f"median_{col}"] = float(np.nanmedian(vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["gene_class"]).reset_index(drop=True)


def select_overlay_genes(per_gene: pd.DataFrame, max_genes: int) -> pd.DataFrame:
    candidates = per_gene[
        per_gene["count_pred_sf_pearson"].notna()
        & (per_gene["count_pred_sf_n_obs"] >= 100)
        & (per_gene["count_pred_sf_true_std"] > 0)
    ].copy()
    marker_candidates = candidates[candidates["gene_class"] != "other"].copy()
    if len(marker_candidates) >= 3:
        candidates = marker_candidates
    if candidates.empty:
        return pd.DataFrame(columns=["gene", "gene_index", "selection_reason", "gene_class", "count_pred_sf_pearson"])
    sorted_candidates = candidates.sort_values("count_pred_sf_pearson")
    picks = []
    for reason, frame in [
        ("poor_count_pred_sf", sorted_candidates.head(2)),
        ("median_count_pred_sf", sorted_candidates.iloc[[len(sorted_candidates) // 2]]),
        ("good_count_pred_sf", sorted_candidates.tail(3).sort_values("count_pred_sf_pearson", ascending=False)),
    ]:
        for _, row in frame.iterrows():
            if row["gene"] not in {pick["gene"] for pick in picks}:
                picks.append(
                    {
                        "gene": row["gene"],
                        "gene_index": int(row["gene_index"]),
                        "selection_reason": reason,
                        "gene_class": row["gene_class"],
                        "count_pred_sf_pearson": float(row["count_pred_sf_pearson"]),
                    }
                )
            if len(picks) >= max_genes:
                break
        if len(picks) >= max_genes:
            break
    return pd.DataFrame(picks)


def load_thumbnail(raw_root: Path, sample_id: str) -> Image.Image | None:
    paths = raw_slide_paths(raw_root, sample_id)
    if not paths.thumbnail.exists():
        return None
    with Image.open(paths.thumbnail) as image:
        return image.convert("RGB")


def fullres_size(metadata: pd.DataFrame, sample_id: str) -> tuple[float, float]:
    row = metadata.loc[metadata["id"].astype(str) == str(sample_id)]
    if row.empty:
        raise ValueError(f"sample_id not found in metadata: {sample_id}")
    item = row.iloc[0]
    return float(item["fullres_px_width"]), float(item["fullres_px_height"])


def scaled_xy(coords: np.ndarray, thumb: Image.Image, metadata: pd.DataFrame, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    full_w, full_h = fullres_size(metadata, sample_id)
    x = coords[:, 0].astype(np.float32) * (thumb.width / full_w)
    y = coords[:, 1].astype(np.float32) * (thumb.height / full_h)
    return x, y


def load_sample_coords(
    *,
    manifest: pd.DataFrame,
    manifest_base: Path,
    sample_id: str,
    min_total_counts: float,
) -> np.ndarray:
    row = manifest.loc[manifest["sample_id"].astype(str) == str(sample_id)]
    if row.empty:
        raise ValueError(f"sample_id not found in manifest: {sample_id}")
    item = row.iloc[0]
    table = load_spot_table(
        sample_id=str(item["sample_id"]),
        features_path=manifest_base / str(item["features_path"]),
        counts_path=manifest_base / str(item["counts_path"]),
        coords_path=manifest_base / str(item["coords_path"]) if str(item.get("coords_path", "")).strip() else None,
        size_factor_path=manifest_base / str(item["size_factor_path"]) if str(item.get("size_factor_path", "")).strip() else None,
        min_total_counts=min_total_counts,
    )
    if table.coords is None:
        raise ValueError(f"sample_id has no coords: {sample_id}")
    return table.coords[table.valid_mask].astype(np.float32, copy=False)


def collect_overlay_values(
    *,
    ds: ExpressionRateDataset,
    loader: DataLoader,
    rate_model: torch.nn.Module,
    pred_sf: np.ndarray,
    overlay_sample_id: str,
    overlay_genes: pd.DataFrame,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    wanted = {int(row.gene_index): str(row.gene) for row in overlay_genes.itertuples(index=False)}
    chunks: dict[str, dict[str, list[np.ndarray]]] = {
        gene: defaultdict(list) for gene in wanted.values()
    }
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["features"].shape[0]
            stop = offset + batch_size
            sample_ids = ds.sample_ids[offset:stop]
            keep_spot = sample_ids == overlay_sample_id
            if np.any(keep_spot):
                pred_log1p_rate = rate_model(batch["features"].to(device)).cpu().numpy()
                pred_rate = np.expm1(pred_log1p_rate).clip(min=0.0)
                true_rate = np.expm1(batch["log1p_rate"].numpy())
                expression_mask = batch["expression_mask"].numpy().astype(bool)
                true_sf = np.exp(batch["true_log_sf"].numpy().reshape(-1))
                batch_pred_sf = pred_sf[offset:stop]
                for gene_idx, gene in wanted.items():
                    measured = expression_mask[keep_spot, gene_idx]
                    chunks[gene]["measured"].append(measured)
                    chunks[gene]["true_count"].append((true_rate[keep_spot, gene_idx] * true_sf[keep_spot]).astype(np.float32))
                    chunks[gene]["pred_count"].append((pred_rate[keep_spot, gene_idx] * batch_pred_sf[keep_spot]).astype(np.float32))
                    chunks[gene]["pred_count_no_sf"].append(pred_rate[keep_spot, gene_idx].astype(np.float32))
                    chunks[gene]["pred_count_oracle_sf"].append((pred_rate[keep_spot, gene_idx] * true_sf[keep_spot]).astype(np.float32))
                    chunks[gene]["pred_sf"].append(batch_pred_sf[keep_spot].astype(np.float32))
                    chunks[gene]["true_sf"].append(true_sf[keep_spot].astype(np.float32))
            offset = stop
    out: dict[str, dict[str, np.ndarray]] = {}
    for gene, values in chunks.items():
        out[gene] = {key: np.concatenate(parts, axis=0) for key, parts in values.items()}
    return out


def plot_expression_overlay(
    *,
    sample_id: str,
    gene: str,
    coords: np.ndarray,
    values: dict[str, np.ndarray],
    raw_root: Path,
    metadata: pd.DataFrame,
    out_path: Path,
) -> bool:
    thumb = load_thumbnail(raw_root, sample_id)
    if thumb is None:
        return False
    measured = values["measured"].astype(bool)
    if measured.sum() < 3:
        return False
    xy = coords[measured]
    x, y = scaled_xy(xy, thumb, metadata, sample_id)
    true_log = np.log1p(values["true_count"][measured])
    pred_log = np.log1p(values["pred_count"][measured])
    residual = pred_log - true_log
    sf_residual = np.log(values["pred_sf"][measured] + 1.0e-8) - np.log(values["true_sf"][measured] + 1.0e-8)
    finite_tp = np.isfinite(true_log) & np.isfinite(pred_log)
    if finite_tp.any():
        lo = float(np.nanpercentile(np.concatenate([true_log[finite_tp], pred_log[finite_tp]]), 2))
        hi = float(np.nanpercentile(np.concatenate([true_log[finite_tp], pred_log[finite_tp]]), 98))
    else:
        lo, hi = 0.0, 1.0
    residual_abs = float(np.nanpercentile(np.abs(residual[np.isfinite(residual)]), 98)) if np.isfinite(residual).any() else 1.0
    sf_abs = float(np.nanpercentile(np.abs(sf_residual[np.isfinite(sf_residual)]), 98)) if np.isfinite(sf_residual).any() else 1.0
    panels = [
        ("true log1p count", true_log, "viridis", lo, hi),
        ("pred log1p count", pred_log, "viridis", lo, hi),
        ("pred - true", residual, "RdBu_r", -residual_abs, residual_abs),
        ("log SF residual", sf_residual, "RdBu_r", -sf_abs, sf_abs),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=180)
    for ax, (title, panel_values, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(thumb)
        scatter = ax.scatter(
            x,
            y,
            c=panel_values,
            s=5,
            cmap=cmap,
            alpha=0.72,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()
        fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.01)
    fig.suptitle(f"{sample_id} coverage95 expression diagnostics: {gene}", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def write_overlay_plots(
    *,
    ctx: DiagnosticsContext,
    ds: ExpressionRateDataset,
    loader: DataLoader,
    rate_model: torch.nn.Module,
    pred_sf: np.ndarray,
    per_gene: pd.DataFrame,
    manifest: pd.DataFrame,
    manifest_base: Path,
) -> pd.DataFrame:
    overlay_genes = select_overlay_genes(per_gene, ctx.max_overlay_genes)
    if overlay_genes.empty:
        return overlay_genes
    sample_id = ctx.overlay_sample_id
    if sample_id is None or sample_id not in set(ds.sample_ids.astype(str)):
        sample_id = str(ds.sample_ids[0])
    values_by_gene = collect_overlay_values(
        ds=ds,
        loader=loader,
        rate_model=rate_model,
        pred_sf=pred_sf,
        overlay_sample_id=sample_id,
        overlay_genes=overlay_genes,
        device=ctx.device,
    )
    metadata_csv = resolve_project_path(ctx.expression_config["paths"]["metadata_csv"])
    raw_root = resolve_project_path(ctx.expression_config["paths"]["raw_root"])
    if metadata_csv is None or raw_root is None:
        raise ValueError("metadata_csv/raw_root resolved to None")
    metadata = pd.read_csv(metadata_csv)
    coords = load_sample_coords(
        manifest=manifest,
        manifest_base=manifest_base,
        sample_id=sample_id,
        min_total_counts=float(ctx.expression_config["data"].get("min_total_counts", 1.0)),
    )
    if coords.shape[0] != int((ds.sample_ids == sample_id).sum()):
        raise ValueError(
            f"Coordinate count mismatch for {sample_id}: coords={coords.shape[0]}, dataset={(ds.sample_ids == sample_id).sum()}"
        )
    rows = []
    for row in overlay_genes.itertuples(index=False):
        gene = str(row.gene)
        out_path = ctx.out_dir / "spatial_overlays" / f"{sample_id}_{gene}.png"
        ok = plot_expression_overlay(
            sample_id=sample_id,
            gene=gene,
            coords=coords,
            values=values_by_gene[gene],
            raw_root=raw_root,
            metadata=metadata,
            out_path=out_path,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "gene": gene,
                "gene_index": int(row.gene_index),
                "gene_class": str(row.gene_class),
                "selection_reason": str(row.selection_reason),
                "count_pred_sf_pearson": float(row.count_pred_sf_pearson),
                "path": str(out_path),
                "written": bool(ok),
            }
        )
    return pd.DataFrame(rows)


def run(ctx: DiagnosticsContext) -> dict[str, object]:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_project_path(ctx.expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression manifest resolved to None")
    manifest = read_manifest(manifest_path)
    manifest = manifest[manifest["split"].isin(ctx.splits)].copy()
    if manifest.empty:
        raise ValueError(f"No manifest rows for splits={ctx.splits}")
    manifest_base = manifest_path.parent
    genes, gene_indices = selected_genes_from_config(ctx.expression_config, base_dir=manifest_base)
    if genes is None:
        raise ValueError("coverage95 diagnostics requires data.gene_names_path.")
    gene_key, raw_st_root = gene_key_settings_from_config(ctx.expression_config)
    expression_ckpt = load_checkpoint(ctx.expression_checkpoint, map_location=str(ctx.device))
    sf_ckpt = load_checkpoint(ctx.sf_checkpoint, map_location=str(ctx.device))
    ds = ExpressionRateDataset(
        manifest,
        base_dir=manifest_base,
        splits=ctx.splits,
        min_total_counts=float(ctx.expression_config["data"].get("min_total_counts", 1.0)),
        standardizer=FeatureStandardizer(mean=expression_ckpt["feature_mean"], std=expression_ckpt["feature_std"]),
        gene_names=genes,
        gene_indices=gene_indices,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    loader = DataLoader(ds, batch_size=ctx.batch_size, shuffle=False)
    sf_model = _load_sf_model(ctx.sf_config, sf_ckpt, ctx.device)
    rate_model = _load_rate_model(ctx.expression_config, expression_ckpt, ctx.device)

    print(f"loaded dataset: spots={len(ds)} slides={len(np.unique(ds.sample_ids))} genes={len(genes)}", flush=True)
    pred_sf, true_log_sf, sf_by_slide = predict_slide_normalized_sf(
        ds=ds,
        loader=loader,
        sf_model=sf_model,
        sf_ckpt=sf_ckpt,
        device=ctx.device,
    )
    sf_overall = sf_metrics(np.log(pred_sf + 1.0e-8), true_log_sf)
    sf_by_slide = sf_by_slide.merge(
        pd.DataFrame.from_dict(sample_metadata(manifest), orient="index")
        .reset_index()
        .rename(columns={"index": "sample_id"}),
        on="sample_id",
        how="left",
    )
    sf_by_organ = []
    sample_meta = sample_metadata(manifest)
    for organ in sorted({meta["organ"] for meta in sample_meta.values()}):
        organ_ids = [sample_id for sample_id, meta in sample_meta.items() if meta["organ"] == organ]
        idx = np.isin(ds.sample_ids.astype(str), organ_ids)
        if idx.sum() < 3:
            continue
        sf_by_organ.append({"organ": organ, "n_spots": int(idx.sum()), **sf_metrics(np.log(pred_sf[idx] + 1.0e-8), true_log_sf[idx])})
    pd.DataFrame([{"scope": "all", "n_spots": int(len(ds)), **sf_overall}]).to_csv(ctx.out_dir / "sf_overall_metrics.csv", index=False)
    sf_by_slide.to_csv(ctx.out_dir / "sf_slide_metrics.csv", index=False)
    pd.DataFrame(sf_by_organ).to_csv(ctx.out_dir / "sf_organ_metrics.csv", index=False)

    n_genes = len(genes)
    gene_accumulators = {
        "rate": VectorMetricAccumulator(n_genes),
        "count_no_sf": VectorMetricAccumulator(n_genes),
        "count_pred_sf": VectorMetricAccumulator(n_genes),
        "count_oracle_sf": VectorMetricAccumulator(n_genes),
    }
    group_accumulators: dict[tuple[str, str], ScalarMetricAccumulator] = defaultdict(ScalarMetricAccumulator)
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            pred_log1p_rate = rate_model(batch["features"].to(ctx.device)).cpu().numpy()
            pred_rate = np.expm1(pred_log1p_rate).clip(min=0.0)
            true_rate = np.expm1(batch["log1p_rate"].numpy())
            expression_mask = batch["expression_mask"].numpy().astype(bool)
            batch_size = pred_rate.shape[0]
            stop = offset + batch_size
            batch_pred_sf = pred_sf[offset:stop]
            batch_true_sf = np.exp(batch["true_log_sf"].numpy().reshape(-1))
            batch_sample_ids = ds.sample_ids[offset:stop].astype(str)
            true_count = true_rate * batch_true_sf[:, None]
            predictions = {
                "rate": (pred_rate, true_rate),
                "count_no_sf": (pred_rate, true_count),
                "count_pred_sf": (pred_rate * batch_pred_sf[:, None], true_count),
                "count_oracle_sf": (pred_rate * batch_true_sf[:, None], true_count),
            }
            for metric_name, (pred, true) in predictions.items():
                gene_accumulators[metric_name].update(pred, true, expression_mask)
                update_group_accumulators(
                    accumulators=group_accumulators,
                    pred=pred,
                    true=true,
                    mask=expression_mask,
                    sample_ids=batch_sample_ids,
                    sample_meta=sample_meta,
                    metric_name=metric_name,
                )
            offset = stop
            if batch_idx % 20 == 0:
                print(f"processed expression batches={batch_idx} spots={offset}", flush=True)

    selection_report = ctx.out_dir.parent / "highconf_symbol_coverage95_gene_selection.csv"
    per_gene = build_per_gene_frame(genes=genes, accumulators=gene_accumulators, selection_report=selection_report)
    per_gene.to_csv(ctx.out_dir / "per_gene_metrics.csv", index=False)
    gene_class_summary = summarize_gene_classes(per_gene)
    gene_class_summary.to_csv(ctx.out_dir / "gene_class_summary.csv", index=False)
    overall_metrics, per_organ, per_slide = group_rows(group_accumulators, sample_meta)
    overall_metrics.to_csv(ctx.out_dir / "overall_expression_metrics.csv", index=False)
    per_organ.to_csv(ctx.out_dir / "per_organ_metrics.csv", index=False)
    per_slide.to_csv(ctx.out_dir / "per_slide_metrics.csv", index=False)
    overlay_manifest = write_overlay_plots(
        ctx=ctx,
        ds=ds,
        loader=loader,
        rate_model=rate_model,
        pred_sf=pred_sf,
        per_gene=per_gene,
        manifest=manifest,
        manifest_base=manifest_base,
    )
    overlay_manifest.to_csv(ctx.out_dir / "spatial_overlay_manifest.csv", index=False)

    summary = {
        "expression_config": str(resolve_project_path(Path("configs/hest1k_human_visium_expression_highconf_symbol95.yaml"))),
        "sf_checkpoint": str(ctx.sf_checkpoint),
        "expression_checkpoint": str(ctx.expression_checkpoint),
        "splits": ctx.splits,
        "n_spots": int(len(ds)),
        "n_slides": int(len(np.unique(ds.sample_ids))),
        "n_genes": int(n_genes),
        "sf_overall": sf_overall,
        "expression_overall": {
            metric_name: acc.summary() for metric_name, acc in gene_accumulators.items()
        },
        "outputs": {
            "per_gene_metrics": str(ctx.out_dir / "per_gene_metrics.csv"),
            "per_slide_metrics": str(ctx.out_dir / "per_slide_metrics.csv"),
            "per_organ_metrics": str(ctx.out_dir / "per_organ_metrics.csv"),
            "gene_class_summary": str(ctx.out_dir / "gene_class_summary.csv"),
            "spatial_overlay_manifest": str(ctx.out_dir / "spatial_overlay_manifest.csv"),
        },
    }
    (ctx.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    ctx = load_context(args)
    run(ctx)


if __name__ == "__main__":
    main()
