from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.eval.evaluate_sf import evaluate
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HEST-1k mean-one SF predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--out-json", type=Path, default=Path("results/hest1k_human_visium_sf/sf_metrics.json"))
    args = parser.parse_args()
    cfg = load_config(resolve_project_path(args.config))
    evaluate(
        cfg,
        checkpoint=resolve_project_path(args.checkpoint),
        split_names=args.splits,
        out_json=resolve_project_path(args.out_json),
    )


if __name__ == "__main__":
    main()
