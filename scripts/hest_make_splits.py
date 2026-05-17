from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.hest.metadata import filter_hest_metadata, load_hest_metadata
from histoomnist.hest.splits import (
    apply_split_to_manifest,
    assign_leave_slide_out,
    make_leave_cohort_out,
    make_leave_organ_out,
    write_split_tables,
)
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


def _load_source_dataframe(cfg: dict, mode: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    manifest_path = resolve_project_path(cfg["input"]["manifest"])
    manifest = None
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    if mode == "manifest" and (manifest is None or manifest.empty):
        raise ValueError(f"Manifest is missing or empty: {manifest_path}")
    if mode == "auto" and manifest is not None and not manifest.empty:
        return manifest.copy(), manifest.copy()
    metadata = load_hest_metadata(resolve_project_path(cfg["input"]["metadata_csv"]))
    filters = cfg["filters"]
    filtered = filter_hest_metadata(
        metadata,
        species=str(filters.get("species", "Homo sapiens")),
        st_technology=str(filters.get("st_technology", "Visium")),
        min_spots_under_tissue=int(filters.get("min_spots_under_tissue", 200)),
    )
    source = filtered.rename(columns={"id": "sample_id"})
    return source, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create HEST-1k slide-level split tables.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_splits.yaml"))
    parser.add_argument("--source", choices=["auto", "metadata", "manifest"], default="auto")
    parser.add_argument("--write-split-manifest", action="store_true")
    args = parser.parse_args()

    cfg = load_config(resolve_project_path(args.config))
    source, manifest = _load_source_dataframe(cfg, args.source)
    split_dir = resolve_project_path(cfg["output"]["split_dir"])
    lso_cfg = cfg["leave_slide_out"]
    leave_slide = assign_leave_slide_out(
        source,
        seed=int(cfg.get("seed", 2026)),
        train_fraction=float(lso_cfg.get("train_fraction", 0.70)),
        val_fraction=float(lso_cfg.get("val_fraction", 0.10)),
        test_fraction=float(lso_cfg.get("test_fraction", 0.20)),
    )
    leave_organ = make_leave_organ_out(source, organs=list(cfg["leave_organ_out"].get("organs", [])))
    leave_cohort = make_leave_cohort_out(
        source,
        cohort_column=str(cfg["leave_cohort_out"].get("cohort_column", "dataset_title")),
        min_test_slides=int(cfg["leave_cohort_out"].get("min_test_slides", 5)),
    )
    write_split_tables(
        split_dir,
        {
            "leave_slide_out": leave_slide,
            "leave_organ_out": leave_organ,
            "leave_cohort_out": leave_cohort,
        },
    )
    if args.write_split_manifest and manifest is not None and not manifest.empty:
        split_manifest = apply_split_to_manifest(manifest, leave_slide, split_type="leave_slide_out")
        out_path = resolve_project_path(cfg["input"]["manifest"])
        split_manifest.to_csv(out_path, index=False)
        print(f"updated manifest splits: {out_path}")
    print(f"source_slides={len(source)}")
    print(f"wrote split tables to {split_dir}")


if __name__ == "__main__":
    main()
