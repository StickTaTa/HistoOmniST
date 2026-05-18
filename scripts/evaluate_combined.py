from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.eval.evaluate_combined import evaluate  # noqa: E402
from histoomnist.utils.config import load_config  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate combined HEST rate + SF count-scale prediction.")
    parser.add_argument("--sf-config", type=Path, required=True)
    parser.add_argument("--expression-config", type=Path, required=True)
    parser.add_argument("--sf-checkpoint", type=Path, required=True)
    parser.add_argument("--expression-checkpoint", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    evaluate(
        sf_config=load_config(resolve_project_path(args.sf_config)),
        expression_config=load_config(resolve_project_path(args.expression_config)),
        sf_checkpoint=resolve_project_path(args.sf_checkpoint),
        expression_checkpoint=resolve_project_path(args.expression_checkpoint),
        split_names=args.splits,
        out_json=resolve_project_path(args.out_json) if args.out_json else None,
    )


if __name__ == "__main__":
    main()
