from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.hest.metadata import (
    filter_hest_metadata,
    load_hest_metadata,
    summarize_hest_metadata,
    write_metadata_report,
)
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HEST-1k metadata for HistoOmniST SF training.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/hest1k_human_visium_sf/metadata_audit"))
    args = parser.parse_args()

    cfg = load_config(resolve_project_path(args.config))
    metadata_csv = resolve_project_path(cfg["paths"]["metadata_csv"])
    filters = cfg["filters"]
    df = load_hest_metadata(metadata_csv)
    filtered = filter_hest_metadata(
        df,
        species=str(filters.get("species", "Homo sapiens")),
        st_technology=str(filters.get("st_technology", "Visium")),
        min_spots_under_tissue=int(filters.get("min_spots_under_tissue", 200)),
    )
    summaries = summarize_hest_metadata(df, filtered)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_dir / "human_visium_candidates.csv", index=False)
    for name, table in summaries.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    write_metadata_report(
        report_path=out_dir / "hest1k_metadata_audit.md",
        metadata_path=metadata_csv,
        df=df,
        filtered=filtered,
        summaries=summaries,
        species=str(filters.get("species", "Homo sapiens")),
        st_technology=str(filters.get("st_technology", "Visium")),
        min_spots_under_tissue=int(filters.get("min_spots_under_tissue", 200)),
    )
    print(f"total_slides={len(df)}")
    print(f"selected_slides={len(filtered)}")
    print(f"wrote {out_dir / 'hest1k_metadata_audit.md'}")


if __name__ == "__main__":
    main()
