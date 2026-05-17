from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histoomnist.data.dataset import FeatureStandardizer, SizeFactorDataset
from histoomnist.eval.metrics import sf_metrics
from histoomnist.models.sf_model import SizeFactorRegressor
from histoomnist.train.common import load_checkpoint
from histoomnist.utils.config import get_device_name, load_config
from histoomnist.utils.io import read_manifest


def evaluate(
    cfg: dict,
    checkpoint: str | Path,
    split_names: list[str] | None = None,
    out_json: str | Path | None = None,
) -> dict[str, float]:
    device = torch.device(get_device_name(cfg.get("device")))
    ckpt = load_checkpoint(checkpoint, map_location=str(device))
    manifest_path = Path(cfg["data"]["manifest"])
    manifest = read_manifest(manifest_path)
    standardizer = FeatureStandardizer(mean=ckpt["feature_mean"], std=ckpt["feature_std"])
    ds = SizeFactorDataset(
        manifest,
        base_dir=manifest_path.parent,
        splits=split_names or list(cfg["data"]["test_splits"]),
        min_total_counts=float(cfg["data"].get("min_total_counts", 1.0)),
        standardizer=standardizer,
    )
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
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["features"].to(device)).cpu().numpy()
            preds.append(pred)
            trues.append(batch["log_sf"].numpy())
    metrics = sf_metrics(np.concatenate(preds), np.concatenate(trues))
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint, split_names=args.splits, out_json=args.out_json)


if __name__ == "__main__":
    main()
