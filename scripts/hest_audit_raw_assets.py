from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.hest.metadata import filter_hest_metadata, load_hest_metadata
from histoomnist.hest.raw_assets import (
    align_patch_to_h5ad_barcodes,
    raw_slide_paths,
    read_h5_string_vector,
)
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw HEST assets and barcode alignment.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/hest1k_human_visium_sf/raw_asset_audit/raw_asset_audit.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/hest1k_human_visium_sf/raw_asset_audit/raw_asset_audit.md"),
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


def read_thumbnail_size(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def audit_slide(raw_root: Path, row: pd.Series) -> dict[str, object]:
    slide_id = str(row["id"])
    paths = raw_slide_paths(raw_root, slide_id)
    out: dict[str, object] = {
        "slide_id": slide_id,
        "organ": row.get("organ", ""),
        "disease_state": row.get("disease_state", ""),
        "oncotree_code": row.get("oncotree_code", ""),
        "dataset_title": row.get("dataset_title", ""),
        "patient": row.get("patient", ""),
        "spots_under_tissue_metadata": row.get("spots_under_tissue", np.nan),
        "metadata_exists": paths.metadata.exists(),
        "st_exists": paths.st.exists(),
        "patches_exists": paths.patches.exists(),
        "thumbnail_exists": paths.thumbnail.exists(),
        "wsi_exists": paths.wsi is not None and paths.wsi.exists(),
        "wsi_path": str(paths.wsi) if paths.wsi else "",
    }
    thumb_w, thumb_h = read_thumbnail_size(paths.thumbnail)
    out["thumbnail_width"] = thumb_w
    out["thumbnail_height"] = thumb_h

    if not paths.st.exists() or not paths.patches.exists():
        out["status"] = "missing_required_asset"
        return out

    try:
        with h5py.File(paths.st, "r") as st, h5py.File(paths.patches, "r") as patches:
            st_barcodes = read_h5_string_vector(st["obs"]["_index"])
            patch_barcodes = read_h5_string_vector(patches["barcode"])
            alignment = align_patch_to_h5ad_barcodes(st_barcodes, patch_barcodes)
            st_coords = np.asarray(st["obsm"]["spatial"][:], dtype=np.float32)
            patch_coords = np.asarray(patches["coords"][:], dtype=np.float32)
            var_n = len(st["var"]["_index"])
            x = st["X"]
            x_nnz = int(x["data"].shape[0]) if isinstance(x, h5py.Group) and "data" in x else np.nan
            total_counts = np.asarray(st["obs"]["total_counts"][:], dtype=np.float64) if "total_counts" in st["obs"] else None
            in_tissue = np.asarray(st["obs"]["in_tissue"][:]) if "in_tissue" in st["obs"] else None

            out.update(
                {
                    "st_n": len(st_barcodes),
                    "st_unique": len(set(st_barcodes)),
                    "n_genes": var_n,
                    "x_nnz": x_nnz,
                    "patch_n": len(patch_barcodes),
                    "patch_unique": len(set(patch_barcodes)),
                    "patch_img_shape": "x".join(map(str, patches["img"].shape)) if "img" in patches else "",
                    "intersection": len(alignment.kept_barcodes),
                    "patch_not_in_st": len(alignment.patch_not_in_st),
                    "st_not_in_patch": len(alignment.st_not_in_patch),
                    "st_coverage_of_patch": len(alignment.kept_barcodes) / len(patch_barcodes) if patch_barcodes else np.nan,
                    "patch_coverage_of_st": len(alignment.kept_barcodes) / len(st_barcodes) if st_barcodes else np.nan,
                    "barcode_order_monotonic": alignment.barcode_order_monotonic,
                    "n_in_tissue": int(np.sum(in_tissue == 1)) if in_tissue is not None else np.nan,
                    "total_counts_min": float(np.nanmin(total_counts)) if total_counts is not None else np.nan,
                    "total_counts_median": float(np.nanmedian(total_counts)) if total_counts is not None else np.nan,
                    "total_counts_mean": float(np.nanmean(total_counts)) if total_counts is not None else np.nan,
                    "total_counts_max": float(np.nanmax(total_counts)) if total_counts is not None else np.nan,
                }
            )

            if len(alignment.kept_barcodes):
                st_common = st_coords[alignment.st_indices]
                patch_common = patch_coords[alignment.patch_indices]
                diffs = patch_common - st_common
                out.update(
                    {
                        "mean_dx_patch_minus_st": float(np.mean(diffs[:, 0])),
                        "mean_dy_patch_minus_st": float(np.mean(diffs[:, 1])),
                        "std_dx_patch_minus_st": float(np.std(diffs[:, 0])),
                        "std_dy_patch_minus_st": float(np.std(diffs[:, 1])),
                        "max_abs_dx_patch_minus_st": float(np.max(np.abs(diffs[:, 0]))),
                        "max_abs_dy_patch_minus_st": float(np.max(np.abs(diffs[:, 1]))),
                    }
                )
            out["status"] = "ok"
    except Exception as exc:
        out["status"] = f"error:{type(exc).__name__}:{exc}"
    return out


def write_markdown_report(audit: pd.DataFrame, out_md: Path) -> None:
    ok = audit[audit["status"] == "ok"].copy()
    lines = [
        "# HEST Raw Asset Audit",
        "",
        "This report is generated from raw HEST assets. It should be regenerated after new assets are downloaded.",
        "",
        "## Summary",
        "",
        f"- Slides audited: {len(audit)}",
        f"- OK slides: {len(ok)}",
        f"- Slides with thumbnails: {int(audit['thumbnail_exists'].sum())}",
        f"- Slides with WSI: {int(audit['wsi_exists'].sum())}",
    ]
    if not ok.empty:
        lines.extend(
            [
                f"- Total ST spots: {int(ok['st_n'].sum())}",
                f"- Total patch spots: {int(ok['patch_n'].sum())}",
                f"- Slides with patch barcode not in ST: {int((ok['patch_not_in_st'] > 0).sum())}",
                f"- Mean patch coverage of ST spots: {ok['patch_coverage_of_st'].mean():.4f}",
                f"- Minimum patch coverage of ST spots: {ok['patch_coverage_of_st'].min():.4f}",
                f"- Slides with monotonic patch/ST barcode order: {int(ok['barcode_order_monotonic'].sum())}",
                "",
                "## Worst Patch Coverage",
                "",
                ok.sort_values("patch_coverage_of_st")[
                    ["slide_id", "organ", "patch_n", "st_n", "intersection", "patch_coverage_of_st", "st_not_in_patch"]
                ]
                .head(20)
                .to_markdown(index=False),
                "",
                "## Asset Presence By Status",
                "",
                audit["status"].value_counts().rename_axis("status").reset_index(name="n").to_markdown(index=False),
            ]
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.config))
    raw_root = resolve_project_path(args.raw_root or cfg["paths"]["raw_root"])
    slides = select_slides(cfg, args.sample_id, args.max_slides)
    rows = []
    for idx, row in slides.iterrows():
        rows.append(audit_slide(raw_root, row))
        if (idx + 1) % 50 == 0:
            print(f"audited_slides={idx + 1}", flush=True)
    audit = pd.DataFrame(rows)
    out_csv = resolve_project_path(args.out_csv)
    out_md = resolve_project_path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_csv, index=False)
    write_markdown_report(audit, out_md)
    print(f"slides={len(audit)}")
    print(f"ok={(audit['status'] == 'ok').sum()}")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
