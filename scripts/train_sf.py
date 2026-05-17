from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.train.train_sf import train
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HEST-1k mean-one SF predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_project_path(args.config))
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.output_dir is not None:
        cfg["output"]["dir"] = str(resolve_project_path(args.output_dir))
    manifest_path = resolve_project_path(cfg["data"]["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty; prepare processed HEST arrays first: {manifest_path}")
    start = time.time()
    best_path = train(cfg)
    elapsed = time.time() - start
    log_path = resolve_project_path("results/hest1k_human_visium_sf/train_sf_run.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "config": str(resolve_project_path(args.config)),
                "manifest": str(manifest_path),
                "best_checkpoint": str(best_path),
                "elapsed_seconds": elapsed,
                "sf_normalization": "mean",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best_checkpoint={best_path}")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
