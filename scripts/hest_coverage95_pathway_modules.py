from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.eval.biological_signatures import (  # noqa: E402
    DEFAULT_EXPRESSION_CHECKPOINT,
    DEFAULT_EXPRESSION_CONFIG,
    DEFAULT_SF_CHECKPOINT,
    DEFAULT_SF_CONFIG,
    evaluate_biological_signatures,
    load_signature_table,
)
from histoomnist.utils.config import load_config  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


DEFAULT_MODULE_TABLE = "configs/hest1k_pathway_modules.csv"
DEFAULT_OUT_DIR = "results/hest1k_human_visium_expression/pathway_modules"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coverage95 pathway-module fidelity.")
    parser.add_argument("--expression-config", default=DEFAULT_EXPRESSION_CONFIG)
    parser.add_argument("--sf-config", default=DEFAULT_SF_CONFIG)
    parser.add_argument("--expression-checkpoint", default=DEFAULT_EXPRESSION_CHECKPOINT)
    parser.add_argument("--sf-checkpoint", default=DEFAULT_SF_CHECKPOINT)
    parser.add_argument("--module-table", default=DEFAULT_MODULE_TABLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--min-module-genes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-overlay-slides", type=int, default=0)
    parser.add_argument("--max-overlay-modules", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_config_path = resolve_project_path(args.expression_config)
    sf_config_path = resolve_project_path(args.sf_config)
    expression_checkpoint = resolve_project_path(args.expression_checkpoint)
    sf_checkpoint = resolve_project_path(args.sf_checkpoint)
    module_table = resolve_project_path(args.module_table)
    out_dir = resolve_project_path(args.out_dir)
    if None in (expression_config_path, sf_config_path, expression_checkpoint, sf_checkpoint, module_table, out_dir):
        raise ValueError("Required paths did not resolve.")

    evaluate_biological_signatures(
        expression_config=load_config(expression_config_path),
        sf_config=load_config(sf_config_path),
        expression_config_path=expression_config_path,
        sf_config_path=sf_config_path,
        expression_checkpoint=expression_checkpoint,
        sf_checkpoint=sf_checkpoint,
        signatures=load_signature_table(module_table),
        out_dir=out_dir,
        splits=[str(x) for x in args.splits],
        min_signature_genes=int(args.min_module_genes),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_overlay_slides=int(args.max_overlay_slides),
        max_overlay_signatures=int(args.max_overlay_modules),
    )


if __name__ == "__main__":
    main()
