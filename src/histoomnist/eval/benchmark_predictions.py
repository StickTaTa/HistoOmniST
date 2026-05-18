from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from histoomnist.data.gene_selection import (
    gene_key_settings_from_config,
    load_gene_keys_for_slide,
    selected_genes_from_config,
)
from histoomnist.data.spot_table import load_spot_table
from histoomnist.utils.config import load_config
from histoomnist.utils.io import read_manifest


PREDICTION_KINDS = {"count", "rate", "log1p_count", "log1p_rate"}


@dataclass(frozen=True)
class SlideTarget:
    sample_id: str
    split: str
    organ: str
    cohort: str
    disease_state: str
    spot_ids: list[str]
    counts: sparse.csr_matrix
    size_factor: np.ndarray
    measured_genes: np.ndarray


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

    def update(self, pred: np.ndarray, true: np.ndarray, measured_genes: np.ndarray) -> None:
        if pred.shape != true.shape:
            raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}")
        if pred.shape[1] != measured_genes.shape[0]:
            raise ValueError(
                f"measured gene length mismatch: pred_genes={pred.shape[1]}, measured={measured_genes.shape[0]}"
            )
        valid = np.isfinite(pred) & np.isfinite(true)
        valid[:, ~measured_genes.astype(bool)] = False
        x = np.where(valid, pred, 0.0).astype(np.float64)
        y = np.where(valid, true, 0.0).astype(np.float64)
        err = x - y
        self.n += valid.sum(axis=0)
        self.sum_pred += x.sum(axis=0)
        self.sum_true += y.sum(axis=0)
        self.sum_pred2 += (x * x).sum(axis=0)
        self.sum_true2 += (y * y).sum(axis=0)
        self.sum_pred_true += (x * y).sum(axis=0)
        self.sum_abs_error += np.where(valid, np.abs(err), 0.0).sum(axis=0)
        self.sum_sq_error += np.where(valid, err * err, 0.0).sum(axis=0)
        self.nonzero_true += (valid & (true > 0)).sum(axis=0)

    def to_frame(self, genes: list[str]) -> pd.DataFrame:
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
                "gene_index": np.arange(len(genes), dtype=np.int64),
                "n_obs": self.n.astype(np.int64),
                "pearson": pearson,
                "mae": self.sum_abs_error / denom_n,
                "rmse": np.sqrt(self.sum_sq_error / denom_n),
                "true_mean": self.sum_true / denom_n,
                "pred_mean": self.sum_pred / denom_n,
                "true_std": np.sqrt(np.maximum(true_var / denom_n, 0.0)),
                "pred_std": np.sqrt(np.maximum(pred_var / denom_n, 0.0)),
                "detected_fraction": self.nonzero_true / denom_n,
            }
        )

    def summary(self) -> dict[str, float]:
        frame = self.to_frame([str(i) for i in range(len(self.n))])
        vals = frame["pearson"].to_numpy(dtype=np.float64)
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

    def update(self, pred: np.ndarray, true: np.ndarray, measured_genes: np.ndarray) -> None:
        if pred.shape != true.shape:
            raise ValueError(f"shape mismatch: pred={pred.shape}, true={true.shape}")
        valid = np.isfinite(pred) & np.isfinite(true)
        valid[:, ~measured_genes.astype(bool)] = False
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
        return {
            "n_values": int(self.n),
            "pearson": float(numerator / denom) if self.n >= 3 and denom > 0 else float("nan"),
            "mae": float(self.sum_abs_error / n),
            "rmse": float(np.sqrt(self.sum_sq_error / n)),
            "true_mean": float(self.sum_true / n),
            "pred_mean": float(self.sum_pred / n),
            "true_std": float(np.sqrt(max(true_var / n, 0.0))),
            "pred_std": float(np.sqrt(max(pred_var / n, 0.0))),
        }


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


def _load_prediction_genes(path: str | Path | None, target_genes: list[str]) -> list[str]:
    if path in (None, ""):
        return list(target_genes)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prediction genes file not found: {p}")
    genes = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not genes:
        raise ValueError(f"Prediction genes file is empty: {p}")
    return genes


def align_genes(target_genes: list[str], prediction_genes: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    pred_index = {gene: idx for idx, gene in enumerate(prediction_genes)}
    common_genes = [gene for gene in target_genes if gene in pred_index]
    if not common_genes:
        raise ValueError("No shared genes between target gene list and prediction gene list.")
    target_idx = np.asarray([target_genes.index(gene) for gene in common_genes], dtype=np.int64)
    pred_idx = np.asarray([pred_index[gene] for gene in common_genes], dtype=np.int64)
    return common_genes, target_idx, pred_idx


def load_slide_target(
    *,
    row,
    base_dir: Path,
    target_genes: list[str],
    gene_key: str,
    raw_st_root: str | Path | None,
    min_total_counts: float,
) -> SlideTarget:
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
    counts = table.counts[mask]
    counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
    slide_genes = load_gene_keys_for_slide(
        sample_id=str(row.sample_id),
        processed_gene_path=base_dir / str(row.genes_path),
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    target_index = {gene: idx for idx, gene in enumerate(target_genes)}
    source_indices: list[int] = []
    target_indices: list[int] = []
    for source_idx, gene in enumerate(slide_genes):
        if gene is None:
            continue
        target_idx = target_index.get(gene)
        if target_idx is None:
            continue
        source_indices.append(source_idx)
        target_indices.append(target_idx)
    if not source_indices:
        raise ValueError(f"No target genes found for slide {row.sample_id}")
    source_array = np.asarray(source_indices, dtype=np.int64)
    target_array = np.asarray(target_indices, dtype=np.int64)
    selected_source = counts[:, source_array].astype(np.float32).tocsr()
    mapper = sparse.csr_matrix(
        (
            np.ones(target_array.shape[0], dtype=np.float32),
            (np.arange(target_array.shape[0]), target_array),
        ),
        shape=(target_array.shape[0], len(target_genes)),
    )
    selected_counts = (selected_source @ mapper).tocsr()
    measured = np.zeros(len(target_genes), dtype=bool)
    measured[np.unique(target_array)] = True
    spot_ids_all = np.asarray(_read_spot_ids(base_dir, row, table.features.shape[0]), dtype=object)
    return SlideTarget(
        sample_id=str(row.sample_id),
        split=str(row.split),
        organ=str(getattr(row, "organ", "")),
        cohort=str(getattr(row, "cohort", "")),
        disease_state=str(getattr(row, "disease_state", "")),
        spot_ids=[str(x) for x in spot_ids_all[mask]],
        counts=selected_counts,
        size_factor=table.size_factor[mask].astype(np.float32, copy=False),
        measured_genes=measured,
    )


def prediction_candidates(root: Path, sample_id: str, kind: str, pattern: str | None) -> list[Path]:
    if pattern:
        return [root / pattern.format(sample_id=sample_id, kind=kind)]
    return [
        root / "predictions" / f"{sample_id}_{kind}.npy",
        root / "predictions" / f"{sample_id}.npy",
        root / sample_id / f"{kind}.npy",
        root / sample_id / "pred.npy",
        root / f"{sample_id}_{kind}.npy",
        root / f"{sample_id}.npy",
    ]


def load_prediction_array(root: Path, sample_id: str, kind: str, pattern: str | None) -> np.ndarray:
    for path in prediction_candidates(root, sample_id, kind, pattern):
        if path.exists():
            array = np.load(path, allow_pickle=False)
            if array.ndim != 2:
                raise ValueError(f"Prediction array must be 2D: {path}, got {array.shape}")
            return np.asarray(array, dtype=np.float32)
    searched = ", ".join(str(path) for path in prediction_candidates(root, sample_id, kind, pattern))
    raise FileNotFoundError(f"Prediction array not found for {sample_id}; searched {searched}")


def true_values_for_kind(target: SlideTarget, kind: str, gene_indices: np.ndarray) -> np.ndarray:
    counts = target.counts[:, gene_indices].astype(np.float32).toarray()
    if kind == "count":
        return counts
    if kind == "log1p_count":
        return np.log1p(counts).astype(np.float32, copy=False)
    rate = counts / np.clip(target.size_factor[:, None], 1.0e-6, None)
    if kind == "rate":
        return rate.astype(np.float32, copy=False)
    if kind == "log1p_rate":
        return np.log1p(rate).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported prediction kind: {kind}")


def prediction_values_for_kind(prediction: np.ndarray, kind: str, gene_indices: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction[:, gene_indices], dtype=np.float32)
    if kind in {"count", "rate"}:
        return np.clip(pred, 0.0, None)
    if kind in {"log1p_count", "log1p_rate"}:
        return pred
    raise ValueError(f"Unsupported prediction kind: {kind}")


def update_group_accumulators(
    *,
    accumulators: dict[tuple[str, str], ScalarMetricAccumulator],
    prediction: np.ndarray,
    truth: np.ndarray,
    target: SlideTarget,
    measured_genes: np.ndarray,
) -> None:
    groups = {
        f"overall|{target.split}": {"level": "overall", "split": target.split},
        f"organ|{target.split}|{target.organ}": {
            "level": "organ",
            "split": target.split,
            "organ": target.organ,
        },
        f"slide|{target.split}|{target.sample_id}": {
            "level": "slide",
            "split": target.split,
            "sample_id": target.sample_id,
            "organ": target.organ,
            "cohort": target.cohort,
            "disease_state": target.disease_state,
        },
    }
    for key in groups:
        accumulators[(key, json.dumps(groups[key], sort_keys=True))].update(prediction, truth, measured_genes)


def group_frames(
    accumulators: dict[tuple[str, str], ScalarMetricAccumulator],
    *,
    method_name: str,
    prediction_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (_, labels_json), acc in sorted(accumulators.items()):
        labels = json.loads(labels_json)
        rows.append(
            {
                "method": method_name,
                "prediction_kind": prediction_kind,
                **labels,
                **acc.row(),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, frame, frame
    overall = frame[frame["level"].eq("overall")].copy()
    organ = frame[frame["level"].eq("organ")].copy()
    slide = frame[frame["level"].eq("slide")].copy()
    return overall, organ, slide


def evaluate_prediction_bundle(
    *,
    expression_config: dict,
    prediction_root: str | Path | None,
    method_name: str,
    prediction_kind: str,
    out_dir: str | Path,
    splits: list[str] | None = None,
    prediction_pattern: str | None = None,
    prediction_genes_path: str | Path | None = None,
    oracle_smoke_test: bool = False,
    max_slides: int | None = None,
) -> dict[str, object]:
    if prediction_kind not in PREDICTION_KINDS:
        raise ValueError(f"prediction_kind must be one of {sorted(PREDICTION_KINDS)}")
    if not oracle_smoke_test and prediction_root in (None, ""):
        raise ValueError("prediction_root is required unless oracle_smoke_test=True")

    manifest_path = Path(expression_config["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    selected_splits = splits or list(expression_config["data"]["test_splits"])
    manifest = manifest[manifest["split"].isin(selected_splits)].copy()
    if max_slides is not None:
        manifest = manifest.head(int(max_slides)).copy()
    if manifest.empty:
        raise ValueError(f"No manifest rows for splits={selected_splits}")
    base_dir = manifest_path.parent
    target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
    if gene_indices is not None or target_genes is None:
        raise ValueError("Benchmark evaluation requires data.gene_names_path with canonical target genes.")
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    prediction_root_path = None if prediction_root in (None, "") else Path(prediction_root)
    pred_genes_path = None if prediction_genes_path in (None, "") else Path(prediction_genes_path)
    if pred_genes_path is not None and not pred_genes_path.is_absolute() and prediction_root_path is not None:
        pred_genes_path = prediction_root_path / pred_genes_path
    prediction_genes = _load_prediction_genes(pred_genes_path, target_genes)
    common_genes, target_gene_indices, prediction_gene_indices = align_genes(target_genes, prediction_genes)

    gene_acc = VectorMetricAccumulator(len(common_genes))
    group_acc: dict[tuple[str, str], ScalarMetricAccumulator] = defaultdict(ScalarMetricAccumulator)
    slide_rows = []
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    for row in manifest.itertuples(index=False):
        target = load_slide_target(
            row=row,
            base_dir=base_dir,
            target_genes=target_genes,
            gene_key=gene_key,
            raw_st_root=raw_st_root,
            min_total_counts=min_total_counts,
        )
        truth = true_values_for_kind(target, prediction_kind, target_gene_indices)
        measured_common = target.measured_genes[target_gene_indices]
        if oracle_smoke_test:
            prediction_values = truth.copy()
        else:
            assert prediction_root_path is not None
            prediction_array = load_prediction_array(
                prediction_root_path,
                target.sample_id,
                prediction_kind,
                prediction_pattern,
            )
            if prediction_array.shape[0] != truth.shape[0]:
                raise ValueError(
                    f"spot count mismatch for {target.sample_id}: "
                    f"prediction={prediction_array.shape[0]}, target={truth.shape[0]}"
                )
            if prediction_array.shape[1] != len(prediction_genes):
                raise ValueError(
                    f"gene count mismatch for {target.sample_id}: "
                    f"prediction={prediction_array.shape[1]}, prediction_genes={len(prediction_genes)}"
                )
            prediction_values = prediction_values_for_kind(
                prediction_array,
                prediction_kind,
                prediction_gene_indices,
            )
        gene_acc.update(prediction_values, truth, measured_common)
        update_group_accumulators(
            accumulators=group_acc,
            prediction=prediction_values,
            truth=truth,
            target=target,
            measured_genes=measured_common,
        )
        slide_rows.append(
            {
                "sample_id": target.sample_id,
                "split": target.split,
                "organ": target.organ,
                "cohort": target.cohort,
                "n_spots": int(truth.shape[0]),
                "n_common_genes": int(len(common_genes)),
                "n_measured_common_genes": int(measured_common.sum()),
            }
        )
        print(
            f"[benchmark] {method_name} {target.sample_id}: "
            f"spots={truth.shape[0]} genes={len(common_genes)} measured={int(measured_common.sum())}",
            flush=True,
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_gene = gene_acc.to_frame(common_genes)
    per_gene.insert(0, "method", method_name)
    per_gene.insert(1, "prediction_kind", prediction_kind)
    per_gene.to_csv(out / "per_gene_metrics.csv", index=False)
    overall, per_organ, per_slide = group_frames(
        group_acc,
        method_name=method_name,
        prediction_kind=prediction_kind,
    )
    overall.to_csv(out / "overall_metrics.csv", index=False)
    per_organ.to_csv(out / "per_organ_metrics.csv", index=False)
    per_slide.to_csv(out / "per_slide_metrics.csv", index=False)
    pd.DataFrame(slide_rows).to_csv(out / "slides_evaluated.csv", index=False)

    summary = {
        "method": method_name,
        "prediction_kind": prediction_kind,
        "oracle_smoke_test": bool(oracle_smoke_test),
        "splits": list(selected_splits),
        "n_slides": int(len(slide_rows)),
        "n_target_genes": int(len(target_genes)),
        "n_prediction_genes": int(len(prediction_genes)),
        "n_common_genes": int(len(common_genes)),
        "gene_metrics": gene_acc.summary(),
        "prediction_root": None if prediction_root_path is None else str(prediction_root_path),
        "prediction_pattern": prediction_pattern,
        "prediction_genes_path": None if pred_genes_path is None else str(pred_genes_path),
        "outputs": {
            "per_gene_metrics": str(out / "per_gene_metrics.csv"),
            "overall_metrics": str(out / "overall_metrics.csv"),
            "per_organ_metrics": str(out / "per_organ_metrics.csv"),
            "per_slide_metrics": str(out / "per_slide_metrics.csv"),
            "slides_evaluated": str(out / "slides_evaluated.csv"),
        },
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sharded benchmark predictions against the HEST coverage95 target."
    )
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--prediction-root", default=None)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--prediction-kind", choices=sorted(PREDICTION_KINDS), default="count")
    parser.add_argument("--prediction-pattern", default=None)
    parser.add_argument("--prediction-genes-path", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--oracle-smoke-test", action="store_true")
    parser.add_argument("--max-slides", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.expression_config)
    evaluate_prediction_bundle(
        expression_config=cfg,
        prediction_root=args.prediction_root,
        method_name=args.method_name,
        prediction_kind=args.prediction_kind,
        out_dir=args.out_dir,
        splits=args.splits,
        prediction_pattern=args.prediction_pattern,
        prediction_genes_path=args.prediction_genes_path,
        oracle_smoke_test=bool(args.oracle_smoke_test),
        max_slides=args.max_slides,
    )


if __name__ == "__main__":
    main()
