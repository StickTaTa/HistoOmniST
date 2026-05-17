from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
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
    mean_one_log_size_factor,
    raw_slide_paths,
    read_h5_string_vector,
    read_h5ad_csr_matrix_from_handle,
)
from histoomnist.utils.config import load_config
from histoomnist.utils.project_paths import resolve_project_path


DEFAULT_GENES = ["EPCAM", "KRT8", "KRT18", "COL1A1", "PTPRC", "CD3D", "MS4A1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate H&E thumbnail overlay QC plots for HEST slides.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--slide-id", action="append", default=None)
    parser.add_argument("--gene", action="append", default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--spot-size", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/hest1k_human_visium_sf/raw_asset_audit/overlay_qc"),
    )
    return parser.parse_args()


def select_slide_ids(cfg: dict, requested: list[str] | None, max_slides: int | None) -> list[str]:
    if requested:
        return requested[:max_slides] if max_slides else requested
    metadata = load_hest_metadata(resolve_project_path(cfg["paths"]["metadata_csv"]))
    filters = cfg["filters"]
    selected = filter_hest_metadata(
        metadata,
        species=str(filters.get("species", "Homo sapiens")),
        st_technology=str(filters.get("st_technology", "Visium")),
        min_spots_under_tissue=int(filters.get("min_spots_under_tissue", 200)),
    )
    ids = selected["id"].astype(str).tolist()
    return ids[: max_slides or 5]


def slide_fullres_size(cfg: dict, slide_id: str) -> tuple[float, float]:
    metadata = pd.read_csv(resolve_project_path(cfg["paths"]["metadata_csv"]))
    row = metadata.loc[metadata["id"].astype(str) == slide_id]
    if row.empty:
        raise ValueError(f"Slide {slide_id} not found in metadata")
    return float(row.iloc[0]["fullres_px_width"]), float(row.iloc[0]["fullres_px_height"])


def load_slide_arrays(raw_root: Path, slide_id: str, genes: list[str]) -> dict[str, object]:
    paths = raw_slide_paths(raw_root, slide_id)
    if not paths.thumbnail.exists():
        raise FileNotFoundError(f"Missing thumbnail: {paths.thumbnail}")
    with Image.open(paths.thumbnail) as image:
        thumbnail = image.convert("RGB")

    with h5py.File(paths.st, "r") as st, h5py.File(paths.patches, "r") as patches:
        st_barcodes = read_h5_string_vector(st["obs"]["_index"])
        patch_barcodes = read_h5_string_vector(patches["barcode"])
        alignment = align_patch_to_h5ad_barcodes(st_barcodes, patch_barcodes)
        coords = np.asarray(st["obsm"]["spatial"][:], dtype=np.float32)[alignment.st_indices]
        total_counts_all = (
            np.asarray(st["obs"]["total_counts"][:], dtype=np.float64)
            if "total_counts" in st["obs"]
            else np.asarray(read_h5ad_csr_matrix_from_handle(st).sum(axis=1)).ravel()
        )
        total_counts = total_counts_all[alignment.st_indices]
        log_sf = mean_one_log_size_factor(total_counts)
        gene_names = read_h5_string_vector(st["var"]["_index"])
        gene_index = {gene: idx for idx, gene in enumerate(gene_names)}
        matrix = read_h5ad_csr_matrix_from_handle(st)
        gene_values: dict[str, np.ndarray] = {}
        for gene in genes:
            idx = gene_index.get(gene)
            if idx is not None:
                gene_values[gene] = np.asarray(matrix[alignment.st_indices, idx].toarray()).ravel().astype(np.float32)

    return {
        "thumbnail": thumbnail,
        "coords": coords,
        "total_counts": total_counts,
        "log_sf": log_sf,
        "gene_values": gene_values,
        "kept_spots": len(coords),
    }


def scaled_coords(coords: np.ndarray, fullres_size: tuple[float, float], thumb_size: tuple[int, int]) -> np.ndarray:
    full_w, full_h = fullres_size
    thumb_w, thumb_h = thumb_size
    scaled = coords.astype(np.float32).copy()
    scaled[:, 0] *= thumb_w / full_w
    scaled[:, 1] *= thumb_h / full_h
    return scaled


def plot_overlay(
    *,
    thumbnail: Image.Image,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: Path,
    spot_size: float,
    alpha: float,
    dpi: int,
    cmap: str = "magma",
) -> None:
    width, height = thumbnail.size
    figsize = (max(5.0, width / 360), max(4.0, height / 360))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(thumbnail)
    finite = np.isfinite(values)
    scatter = ax.scatter(
        coords[finite, 0],
        coords[finite, 1],
        c=values[finite],
        s=spot_size,
        alpha=alpha,
        cmap=cmap,
        linewidths=0,
    )
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.026, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.config))
    raw_root = resolve_project_path(args.raw_root or cfg["paths"]["raw_root"])
    out_dir = resolve_project_path(args.out_dir)
    genes = args.gene or DEFAULT_GENES
    slide_ids = select_slide_ids(cfg, args.slide_id, args.max_slides)

    rows = []
    for slide_id in slide_ids:
        arrays = load_slide_arrays(raw_root, slide_id, genes)
        thumbnail: Image.Image = arrays["thumbnail"]  # type: ignore[assignment]
        fullres = slide_fullres_size(cfg, slide_id)
        coords = scaled_coords(arrays["coords"], fullres, thumbnail.size)  # type: ignore[arg-type]
        slide_dir = out_dir / slide_id

        total_counts = np.log1p(np.asarray(arrays["total_counts"], dtype=np.float64))
        plot_overlay(
            thumbnail=thumbnail,
            coords=coords,
            values=total_counts,
            title=f"{slide_id} log1p(total counts)",
            out_path=slide_dir / f"{slide_id}_log1p_total_counts.png",
            spot_size=args.spot_size,
            alpha=args.alpha,
            dpi=args.dpi,
            cmap="viridis",
        )
        plot_overlay(
            thumbnail=thumbnail,
            coords=coords,
            values=np.asarray(arrays["log_sf"], dtype=np.float64),
            title=f"{slide_id} true log mean-one SF",
            out_path=slide_dir / f"{slide_id}_true_log_sf.png",
            spot_size=args.spot_size,
            alpha=args.alpha,
            dpi=args.dpi,
            cmap="coolwarm",
        )
        rows.append({"slide_id": slide_id, "layer": "log1p_total_counts", "path": str(slide_dir / f"{slide_id}_log1p_total_counts.png")})
        rows.append({"slide_id": slide_id, "layer": "true_log_sf", "path": str(slide_dir / f"{slide_id}_true_log_sf.png")})

        gene_values: dict[str, np.ndarray] = arrays["gene_values"]  # type: ignore[assignment]
        for gene, values in gene_values.items():
            plot_overlay(
                thumbnail=thumbnail,
                coords=coords,
                values=np.log1p(values.astype(np.float64)),
                title=f"{slide_id} {gene} log1p(count)",
                out_path=slide_dir / f"{slide_id}_{gene}_log1p_count.png",
                spot_size=args.spot_size,
                alpha=args.alpha,
                dpi=args.dpi,
                cmap="magma",
            )
            rows.append({"slide_id": slide_id, "layer": f"{gene}_log1p_count", "path": str(slide_dir / f"{slide_id}_{gene}_log1p_count.png")})

        missing = sorted(set(genes) - set(gene_values))
        if missing:
            print(f"{slide_id}: missing genes skipped: {','.join(missing)}")
        print(f"{slide_id}: wrote {len(rows)} total overlay entries")

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "overlay_qc_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
