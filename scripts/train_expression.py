from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.train.train_expression import train  # noqa: E402
from histoomnist.utils.config import load_config  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HEST expression-rate predictor.")
    parser.add_argument("--config", type=Path, required=True)
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
    start = time.time()
    best = train(cfg)
    print(f"best_checkpoint={best}")
    print(f"elapsed_seconds={time.time() - start:.2f}")


if __name__ == "__main__":
    main()
