from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.data.dataset import FeatureStandardizer  # noqa: E402
from histoomnist.data.spot_table import load_spot_table  # noqa: E402
from histoomnist.eval.metrics import sf_metrics  # noqa: E402
from histoomnist.hest.raw_assets import raw_slide_paths  # noqa: E402
from histoomnist.models.calibration import AffineLogSFCalibrator  # noqa: E402
from histoomnist.models.sf_model import SizeFactorRegressor  # noqa: E402
from histoomnist.train.common import load_checkpoint  # noqa: E402
from histoomnist.utils.config import get_device_name, load_config  # noqa: E402
from histoomnist.utils.io import read_manifest  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


METRIC_COLUMNS = [
    "log_sf_pearson",
    "sf_pearson",
    "log_sf_mae",
    "log_sf_rmse",
    "sf_std_ratio",
    "sf_top_decile_mean_ratio",
    "log_sf_top_decile_mae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose HEST SF predictions by slide, organ, and spatial maps.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/hest1k_human_visium_sf/main_hipt256_leave_slide_out/best.pt"),
    )
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-spatial-slides", type=int, default=12)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/hest1k_human_visium_sf/diagnostics/sf_main_hipt256_leave_slide_out"),
    )
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


def _read_spot_ids(base_dir: Path, row, n: int) -> list[str]:
    candidates: list[Path] = []
    explicit = _optional_path(row, "spots_path")
    if explicit is not None:
        candidates.append(base_dir / str(explicit))
    candidates.append((base_dir / str(row.features_path)).parent / "spots.txt")
    for path in candidates:
        if path.exists():
            ids = path.read_text(encoding="utf-8").splitlines()
            if len(ids) >= n:
                return ids[:n]
    return [f"spot_{i}" for i in range(n)]


def build_model(cfg: dict, ckpt: dict, device: torch.device) -> SizeFactorRegressor:
    model = SizeFactorRegressor(
        input_dim=int(ckpt["input_dim"]),
        **ckpt.get(
            "model_kwargs",
            {
                "hidden_dims": list(cfg["model"].get("hidden_dims") or []),
                "dropout": float(cfg["model"].get("dropout", 0.15)),
                "architecture": str(cfg["model"].get("architecture", "residual_mlp")),
                "width": int(cfg["model"].get("width", 512)),
                "depth": int(cfg["model"].get("depth", 4)),
            },
        ),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_array(
    *,
    features: np.ndarray,
    model: SizeFactorRegressor,
    standardizer: FeatureStandardizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        x = standardizer.transform(features[start:stop])
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().reshape(-1)
        chunks.append(pred.astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def mean_one_log_sf(log_sf: np.ndarray) -> np.ndarray:
    sf = np.exp(np.asarray(log_sf, dtype=np.float64))
    sf = sf / (float(np.mean(sf)) + 1.0e-8)
    return np.log(sf + 1.0e-8).astype(np.float32)


def metric_row(
    *,
    pred_log_sf: np.ndarray,
    true_log_sf: np.ndarray,
    label_values: dict[str, object],
) -> dict[str, object]:
    row = dict(label_values)
    row.update(sf_metrics(pred_log_sf, true_log_sf))
    row["n_spots"] = int(len(true_log_sf))
    row["true_log_sf_std"] = float(np.std(true_log_sf))
    row["pred_log_sf_std"] = float(np.std(pred_log_sf))
    return row


def grouped_metrics(
    spots: pd.DataFrame,
    *,
    group_cols: list[str],
    pred_col: str,
    label: str,
) -> pd.DataFrame:
    rows = []
    for values, group in spots.groupby(group_cols, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        labels = dict(zip(group_cols, values))
        labels["prediction"] = label
        rows.append(
            metric_row(
                pred_log_sf=group[pred_col].to_numpy(),
                true_log_sf=group["true_log_sf"].to_numpy(),
                label_values=labels,
            )
        )
    return pd.DataFrame(rows)


def overall_metrics(spots: pd.DataFrame, *, pred_col: str, label: str) -> pd.DataFrame:
    rows = []
    for split, group in spots.groupby("split", dropna=False):
        rows.append(
            metric_row(
                pred_log_sf=group[pred_col].to_numpy(),
                true_log_sf=group["true_log_sf"].to_numpy(),
                label_values={"split": split, "prediction": label},
            )
        )
    return pd.DataFrame(rows)


def fit_val_calibrator(spots: pd.DataFrame) -> AffineLogSFCalibrator:
    val = spots[spots["split"] == "val"].copy()
    if val.empty:
        raise ValueError("Need validation split predictions to fit calibration.")
    return AffineLogSFCalibrator().fit(val["pred_log_sf"].to_numpy(), val["true_log_sf"].to_numpy())


def apply_calibration(spots: pd.DataFrame, calibrator: AffineLogSFCalibrator) -> pd.DataFrame:
    out = spots.copy()
    out["pred_log_sf_calibrated_raw"] = calibrator.transform(out["pred_log_sf"].to_numpy())
    out["pred_log_sf_calibrated"] = np.nan
    for sample_id, group_idx in out.groupby("sample_id").groups.items():
        values = out.loc[group_idx, "pred_log_sf_calibrated_raw"].to_numpy()
        out.loc[group_idx, "pred_log_sf_calibrated"] = mean_one_log_sf(values)
    out["pred_sf_calibrated"] = np.exp(out["pred_log_sf_calibrated"].to_numpy())
    out["residual_log_sf_calibrated"] = out["pred_log_sf_calibrated"] - out["true_log_sf"]
    return out


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


def scaled_xy(group: pd.DataFrame, thumb: Image.Image, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    full_w, full_h = fullres_size(metadata, str(group["sample_id"].iloc[0]))
    x = group["x"].to_numpy(dtype=np.float32) * (thumb.width / full_w)
    y = group["y"].to_numpy(dtype=np.float32) * (thumb.height / full_h)
    return x, y


def plot_spatial_slide(
    *,
    group: pd.DataFrame,
    raw_root: Path,
    metadata: pd.DataFrame,
    out_path: Path,
) -> bool:
    sample_id = str(group["sample_id"].iloc[0])
    thumb = load_thumbnail(raw_root, sample_id)
    if thumb is None:
        return False
    x, y = scaled_xy(group, thumb, metadata)
    panels = [
        ("true log SF", group["true_log_sf"].to_numpy(dtype=np.float32), "coolwarm"),
        ("pred log SF", group["pred_log_sf"].to_numpy(dtype=np.float32), "coolwarm"),
        ("calibrated pred log SF", group["pred_log_sf_calibrated"].to_numpy(dtype=np.float32), "coolwarm"),
        ("residual pred - true", group["residual_log_sf"].to_numpy(dtype=np.float32), "RdBu_r"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=180)
    for ax, (title, values, cmap) in zip(axes, panels):
        ax.imshow(thumb)
        scatter = ax.scatter(x, y, c=values, s=5, cmap=cmap, alpha=0.72, linewidths=0)
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()
        fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.01)
    fig.suptitle(f"{sample_id} SF spatial diagnostics", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return True


def plot_scatter(spots: pd.DataFrame, out_path: Path, *, pred_col: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
    x = spots["true_log_sf"].to_numpy()
    y = spots[pred_col].to_numpy()
    ax.hexbin(x, y, gridsize=70, mincnt=1, cmap="viridis")
    lo = float(np.nanpercentile(np.concatenate([x, y]), 0.5))
    hi = float(np.nanpercentile(np.concatenate([x, y]), 99.5))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("true log SF")
    ax.set_ylabel(pred_col)
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def select_spatial_slides(slide_metrics: pd.DataFrame, max_slides: int) -> list[str]:
    test = slide_metrics[
        (slide_metrics["split"] == "test") & (slide_metrics["prediction"] == "uncalibrated")
    ].copy()
    if test.empty:
        return []
    picked: list[str] = []
    for frame in (
        test.sort_values("log_sf_pearson", ascending=True).head(4),
        test.sort_values("log_sf_pearson", ascending=False).head(4),
        test.sort_values("sf_top_decile_mean_ratio", ascending=True).head(4),
    ):
        for sample_id in frame["sample_id"].astype(str):
            if sample_id not in picked:
                picked.append(sample_id)
            if len(picked) >= max_slides:
                return picked
    return picked[:max_slides]


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.config))
    if args.device is not None:
        cfg["device"] = args.device
    device = torch.device(get_device_name(cfg.get("device")))
    ckpt = load_checkpoint(resolve_project_path(args.checkpoint), map_location=str(device))
    model = build_model(cfg, ckpt, device)
    standardizer = FeatureStandardizer(mean=ckpt["feature_mean"], std=ckpt["feature_std"])

    manifest_path = resolve_project_path(cfg["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    manifest = manifest[manifest["split"].isin(args.splits)].copy()
    base_dir = manifest_path.parent
    raw_root = resolve_project_path(cfg["paths"]["raw_root"])
    metadata = pd.read_csv(resolve_project_path(cfg["paths"]["metadata_csv"]))

    rows: list[dict[str, object]] = []
    min_total_counts = float(cfg["data"].get("min_total_counts", 1.0))
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
        features = table.features[mask].astype(np.float32, copy=False)
        pred_log_raw = predict_array(
            features=features,
            model=model,
            standardizer=standardizer,
            device=device,
            batch_size=args.batch_size,
        )
        pred_log = mean_one_log_sf(pred_log_raw)
        true_sf = table.size_factor[mask].astype(np.float32, copy=False)
        true_log = np.log(true_sf + 1.0e-8).astype(np.float32)
        coords = table.coords[mask] if table.coords is not None else np.full((int(mask.sum()), 2), np.nan)
        spot_ids_all = np.asarray(_read_spot_ids(base_dir, row, table.features.shape[0]), dtype=object)
        spot_ids = spot_ids_all[mask]
        for i in range(features.shape[0]):
            rows.append(
                {
                    "sample_id": str(row.sample_id),
                    "spot_id": str(spot_ids[i]),
                    "spot_index": int(i),
                    "split": str(row.split),
                    "organ": str(getattr(row, "organ", "")),
                    "cohort": str(getattr(row, "cohort", "")),
                    "disease_state": str(getattr(row, "disease_state", "")),
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "true_sf": float(true_sf[i]),
                    "true_log_sf": float(true_log[i]),
                    "pred_log_sf_raw": float(pred_log_raw[i]),
                    "pred_log_sf": float(pred_log[i]),
                    "pred_sf": float(np.exp(pred_log[i])),
                    "residual_log_sf": float(pred_log[i] - true_log[i]),
                }
            )
        print(f"predicted {row.sample_id} n_valid={features.shape[0]}", flush=True)

    spots = pd.DataFrame(rows)
    calibrator = fit_val_calibrator(spots)
    spots = apply_calibration(spots, calibrator)

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spots.to_csv(out_dir / "spot_predictions.csv", index=False)

    metric_tables = []
    for pred_col, label in [
        ("pred_log_sf", "uncalibrated"),
        ("pred_log_sf_calibrated", "val_affine_calibrated"),
    ]:
        overall = overall_metrics(spots, pred_col=pred_col, label=label)
        slide = grouped_metrics(spots, group_cols=["split", "sample_id", "organ", "cohort"], pred_col=pred_col, label=label)
        organ = grouped_metrics(spots, group_cols=["split", "organ"], pred_col=pred_col, label=label)
        cohort = grouped_metrics(spots, group_cols=["split", "cohort"], pred_col=pred_col, label=label)
        overall.to_csv(out_dir / f"metrics_overall_{label}.csv", index=False)
        slide.to_csv(out_dir / f"metrics_by_slide_{label}.csv", index=False)
        organ.to_csv(out_dir / f"metrics_by_organ_{label}.csv", index=False)
        cohort.to_csv(out_dir / f"metrics_by_cohort_{label}.csv", index=False)
        metric_tables.append(overall.assign(level="overall"))
        metric_tables.append(organ.assign(level="organ"))

    pd.concat(metric_tables, ignore_index=True).to_csv(out_dir / "metrics_summary_long.csv", index=False)
    (out_dir / "calibration.json").write_text(json.dumps(calibrator.to_dict(), indent=2), encoding="utf-8")

    test_spots = spots[spots["split"] == "test"].copy()
    plot_scatter(test_spots, out_dir / "plots" / "test_true_vs_pred_log_sf.png", pred_col="pred_log_sf", title="Test true vs predicted log SF")
    plot_scatter(
        test_spots,
        out_dir / "plots" / "test_true_vs_calibrated_pred_log_sf.png",
        pred_col="pred_log_sf_calibrated",
        title="Test true vs calibrated predicted log SF",
    )

    slide_metrics = pd.read_csv(out_dir / "metrics_by_slide_uncalibrated.csv")
    selected = select_spatial_slides(slide_metrics, args.max_spatial_slides)
    spatial_rows = []
    for sample_id in selected:
        group = spots[spots["sample_id"] == sample_id].copy()
        out_path = out_dir / "plots" / "spatial" / f"{sample_id}_sf_spatial_diagnostics.png"
        ok = plot_spatial_slide(group=group, raw_root=raw_root, metadata=metadata, out_path=out_path)
        spatial_rows.append({"sample_id": sample_id, "path": str(out_path), "written": ok})
    pd.DataFrame(spatial_rows).to_csv(out_dir / "spatial_plot_manifest.csv", index=False)

    run_summary = {
        "checkpoint": str(resolve_project_path(args.checkpoint)),
        "config": str(resolve_project_path(args.config)),
        "splits": list(args.splits),
        "n_spots": int(len(spots)),
        "n_slides": int(spots["sample_id"].nunique()),
        "calibration": calibrator.to_dict(),
        "outputs": {
            "spot_predictions": str(out_dir / "spot_predictions.csv"),
            "metrics_summary_long": str(out_dir / "metrics_summary_long.csv"),
            "plots": str(out_dir / "plots"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

