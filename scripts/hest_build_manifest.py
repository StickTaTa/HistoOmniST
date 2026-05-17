from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.hest.manifest import build_hest_manifest, write_manifest_outputs
from histoomnist.hest.metadata import filter_hest_metadata, load_hest_metadata
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a HEST-1k processed-data manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_project_path(args.config))
    metadata_csv = resolve_project_path(args.metadata_csv or cfg["paths"]["metadata_csv"])
    processed_root = resolve_project_path(args.processed_root or cfg["paths"]["processed_root"])
    manifest_path = resolve_project_path(args.manifest or cfg["paths"]["manifest"])
    candidate_path = manifest_path.parent / "human_visium_asset_candidates.csv"
    filters = cfg["filters"]
    sf_cfg = cfg["size_factor"]

    df = load_hest_metadata(metadata_csv)
    filtered = filter_hest_metadata(
        df,
        species=str(filters.get("species", "Homo sapiens")),
        st_technology=str(filters.get("st_technology", "Visium")),
        min_spots_under_tissue=int(filters.get("min_spots_under_tissue", 200)),
    )
    manifest, candidates = build_hest_manifest(
        filtered,
        processed_root=processed_root,
        manifest_path=manifest_path,
        sf_normalization=str(sf_cfg.get("normalization", "mean")),
    )
    write_manifest_outputs(
        manifest=manifest,
        candidates=candidates,
        manifest_path=manifest_path,
        candidate_path=candidate_path,
    )
    ready = int(candidates["ready_for_training"].sum()) if not candidates.empty else 0
    print(f"candidate_slides={len(candidates)}")
    print(f"training_ready_slides={ready}")
    print(f"wrote {manifest_path}")
    print(f"wrote {candidate_path}")
    if manifest.empty:
        print("manifest is empty because processed features/counts/coords are not present yet.")


if __name__ == "__main__":
    main()
