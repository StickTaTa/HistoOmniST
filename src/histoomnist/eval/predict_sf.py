from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from histoomnist.data.dataset import FeatureStandardizer
from histoomnist.data.spot_table import load_array
from histoomnist.models.sf_model import SizeFactorRegressor
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest


def _optional_value(row, name: str):
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
    explicit = _optional_value(row, "spots_path")
    candidates = []
    if explicit is not None:
        candidates.append(base_dir / str(explicit))
    candidates.append((base_dir / str(row.features_path)).parent / "spots.txt")
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8").splitlines()
    return [f"spot_{i}" for i in range(n)]


def _build_model(cfg: dict, ckpt: dict, device: torch.device) -> SizeFactorRegressor:
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


def predict_sf(
    cfg: dict,
    checkpoint: str | Path,
    out_csv: str | Path,
    split_names: list[str] | None = None,
) -> Path:
    device = torch.device(get_device_name(cfg.get("device")))
    ckpt = load_checkpoint(checkpoint, map_location=str(device))
    model = _build_model(cfg, ckpt, device)
    standardizer = FeatureStandardizer(mean=ckpt["feature_mean"], std=ckpt["feature_std"])

    manifest_path = Path(cfg["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    if split_names:
        manifest = manifest[manifest["split"].isin(split_names)].copy()
    base_dir = manifest_path.parent

    rows = []
    with torch.no_grad():
        for row in manifest.itertuples(index=False):
            sample_id = str(row.sample_id)
            features = np.asarray(load_array(base_dir / str(row.features_path)), dtype=np.float32)
            x = standardizer.transform(features)
            pred_log_sf = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
            pred_sf = np.exp(pred_log_sf)
            pred_sf = pred_sf / (float(pred_sf.mean()) + 1e-8)
            coords = None
            coords_path = _optional_value(row, "coords_path")
            if coords_path is not None and (base_dir / str(coords_path)).exists():
                coords = np.asarray(load_array(base_dir / str(coords_path)), dtype=np.float32)
            true_sf = None
            sf_path = _optional_value(row, "size_factor_path")
            if sf_path is not None and (base_dir / str(sf_path)).exists():
                true_sf = np.asarray(load_array(base_dir / str(sf_path)), dtype=np.float32).reshape(-1)
            spot_ids = _read_spot_ids(base_dir, row, features.shape[0])
            for i in range(features.shape[0]):
                item = {
                    "sample_id": sample_id,
                    "spot_index": i,
                    "spot_id": spot_ids[i] if i < len(spot_ids) else f"spot_{i}",
                    "pred_log_sf": float(np.log(pred_sf[i] + 1e-8)),
                    "pred_sf": float(pred_sf[i]),
                }
                if coords is not None:
                    item["y"] = float(coords[i, 0])
                    item["x"] = float(coords[i, 1])
                if true_sf is not None:
                    item["true_sf_for_eval"] = float(true_sf[i])
                    item["true_log_sf_for_eval"] = float(np.log(true_sf[i] + 1e-8))
                rows.append(item)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    args = parser.parse_args()
    predict_sf(
        cfg=load_config(args.config),
        checkpoint=args.checkpoint,
        out_csv=args.out_csv,
        split_names=args.splits,
    )


if __name__ == "__main__":
    main()
