from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.hest.metadata import filter_hest_metadata, load_hest_metadata
from histoomnist.data.size_factor import compute_size_factor
from histoomnist.hest.raw_assets import (
    align_patch_to_h5ad_barcodes,
    raw_slide_paths,
    read_h5_string_vector,
    read_h5ad_csr_matrix_from_handle,
)
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw HEST h5ad/patch assets into barcode-aligned processed arrays."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--min-patch-coverage", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/hest1k_human_visium_sf/processed_array_conversion.csv"),
    )
    return parser.parse_args()


def select_slides(cfg: dict, sample_ids: list[str] | None, max_slides: int | None) -> pd.DataFrame:
    metadata = load_hest_metadata(resolve_project_path(cfg["paths"]["metadata_csv"]))
    filters = cfg["filters"]
    selected = filter_hest_metadata(
        metadata,
        species=str(filters.get("species", "Homo sapiens")),
        st_technology=str(filters.get("st_technology", "Visium")),
        min_spots_under_tissue=int(filters.get("min_spots_under_tissue", 200)),
    )
    if sample_ids:
        wanted = set(sample_ids)
        selected = selected[selected["id"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(selected["id"].astype(str)))
        if missing:
            raise ValueError(f"sample_id values do not match selected metadata filters: {missing}")
    if max_slides:
        selected = selected.head(max_slides).copy()
    return selected.reset_index(drop=True)


def json_safe(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def convert_slide(
    *,
    raw_root: Path,
    processed_root: Path,
    metadata_row: pd.Series,
    min_patch_coverage: float,
    overwrite: bool,
) -> dict[str, object]:
    slide_id = str(metadata_row["id"])
    paths = raw_slide_paths(raw_root, slide_id)
    out_dir = processed_root / slide_id
    required_outputs = [
        out_dir / "counts.npz",
        out_dir / "coords.npy",
        out_dir / "size_factor.npy",
        out_dir / "spots.txt",
        out_dir / "genes.txt",
        out_dir / "metadata.json",
        out_dir / "patch_indices.npy",
        out_dir / "valid_mask.npy",
    ]
    row: dict[str, object] = {
        "slide_id": slide_id,
        "organ": metadata_row.get("organ", ""),
        "dataset_title": metadata_row.get("dataset_title", ""),
        "processed_dir": str(out_dir),
    }
    if not overwrite and all(path.exists() for path in required_outputs):
        row["status"] = "exists"
        return row
    if not paths.st.exists() or not paths.patches.exists():
        row["status"] = "missing_required_raw_asset"
        return row

    try:
        with h5py.File(paths.st, "r") as st, h5py.File(paths.patches, "r") as patches:
            st_barcodes = read_h5_string_vector(st["obs"]["_index"])
            patch_barcodes = read_h5_string_vector(patches["barcode"])
            alignment = align_patch_to_h5ad_barcodes(st_barcodes, patch_barcodes)
            patch_coverage = len(alignment.kept_barcodes) / len(st_barcodes) if st_barcodes else 0.0
            if patch_coverage < min_patch_coverage:
                row.update(
                    {
                        "status": "below_min_patch_coverage",
                        "st_n": len(st_barcodes),
                        "patch_n": len(patch_barcodes),
                        "kept_n": len(alignment.kept_barcodes),
                        "patch_coverage_of_st": patch_coverage,
                    }
                )
                return row

            matrix = read_h5ad_csr_matrix_from_handle(st)
            counts = matrix[alignment.st_indices].tocsr()
            coords = np.asarray(st["obsm"]["spatial"][:], dtype=np.float32)[alignment.st_indices]
            genes = read_h5_string_vector(st["var"]["_index"])
            sf, valid = compute_size_factor(counts, min_total_counts=1.0)

        out_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(out_dir / "counts.npz", counts)
        np.save(out_dir / "coords.npy", coords.astype(np.float32))
        np.save(out_dir / "size_factor.npy", sf.astype(np.float32))
        np.save(out_dir / "valid_mask.npy", valid.astype(bool))
        np.save(out_dir / "patch_indices.npy", alignment.patch_indices.astype(np.int64))
        write_lines(out_dir / "spots.txt", alignment.kept_barcodes)
        write_lines(out_dir / "genes.txt", genes)

        metadata = {key: json_safe(value) for key, value in metadata_row.to_dict().items()}
        metadata.update(
            {
                "source_st": str(paths.st),
                "source_patches": str(paths.patches),
                "source_thumbnail": str(paths.thumbnail) if paths.thumbnail.exists() else None,
                "source_wsi": str(paths.wsi) if paths.wsi else None,
                "alignment_rule": "patch_barcodes_intersect_h5ad_obs_names_ordered_by_patch_barcodes",
                "sf_normalization": "mean",
                "sf_scope": "retained_barcode_aligned_spots_with_min_total_counts_valid_mask",
                "min_total_counts": 1.0,
            }
        )
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        row.update(
            {
                "status": "ok",
                "st_n": len(st_barcodes),
                "patch_n": len(patch_barcodes),
                "kept_n": len(alignment.kept_barcodes),
                "n_genes": len(genes),
                "counts_nnz": int(counts.nnz),
                "valid_n": int(np.sum(valid)),
                "invalid_n": int(len(valid) - np.sum(valid)),
                "patch_coverage_of_st": patch_coverage,
                "st_coverage_of_patch": len(alignment.kept_barcodes) / len(patch_barcodes) if patch_barcodes else np.nan,
                "sf_min": float(sf.min()),
                "sf_mean_all_spots": float(sf.mean()),
                "sf_mean_valid_spots": float(sf[valid].mean()),
                "sf_max": float(sf.max()),
            }
        )
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}:{exc}"
    return row


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.config))
    raw_root = resolve_project_path(args.raw_root or cfg["paths"]["raw_root"])
    processed_root = resolve_project_path(args.processed_root or cfg["paths"]["processed_root"])
    slides = select_slides(cfg, args.sample_id, args.max_slides)
    rows = []
    for idx, metadata_row in slides.iterrows():
        rows.append(
            convert_slide(
                raw_root=raw_root,
                processed_root=processed_root,
                metadata_row=metadata_row,
                min_patch_coverage=args.min_patch_coverage,
                overwrite=args.overwrite,
            )
        )
        if (idx + 1) % 25 == 0:
            print(f"converted_slides={idx + 1}", flush=True)
    report = pd.DataFrame(rows)
    out_csv = resolve_project_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)
    print(f"slides={len(report)}")
    print(report["status"].value_counts().to_string())
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
