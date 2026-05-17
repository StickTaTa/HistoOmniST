from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.hest.baselines import run_available_baselines
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run available SF baselines on prepared HEST manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/hest1k_human_visium_sf/baselines/sf_baselines.csv"))
    args = parser.parse_args()
    cfg = load_config(resolve_project_path(args.config))
    table = run_available_baselines(
        manifest_path=resolve_project_path(cfg["data"]["manifest"]),
        train_splits=list(cfg["data"]["train_splits"]),
        test_splits=list(cfg["data"]["test_splits"]),
        output_csv=resolve_project_path(args.output_csv),
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
