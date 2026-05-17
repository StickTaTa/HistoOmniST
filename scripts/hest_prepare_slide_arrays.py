from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.hest.sf_targets import save_mean_one_size_factor
from histoomnist.utils.project_paths import resolve_project_path


def validate_slide(slide_dir: Path, *, min_total_counts: float) -> dict[str, object]:
    features = slide_dir / "features.npy"
    counts = slide_dir / "counts.npz"
    coords = slide_dir / "coords.npy"
    missing = [path.name for path in [features, counts, coords] if not path.exists()]
    if missing:
        return {"slide_id": slide_dir.name, "status": "missing_required", "missing": ";".join(missing)}
    x = np.load(features, mmap_mode="r")
    c = sparse.load_npz(counts)
    xy = np.load(coords, mmap_mode="r")
    if x.shape[0] != c.shape[0] or x.shape[0] != xy.shape[0]:
        return {
            "slide_id": slide_dir.name,
            "status": "shape_mismatch",
            "features_rows": int(x.shape[0]),
            "counts_rows": int(c.shape[0]),
            "coords_rows": int(xy.shape[0]),
        }
    save_mean_one_size_factor(counts, slide_dir / "size_factor.npy", min_total_counts=min_total_counts)
    return {
        "slide_id": slide_dir.name,
        "status": "ok",
        "n_spots": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_genes": int(c.shape[1]),
        "sf_normalization": "mean",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate prepared HEST slide arrays and compute mean-one size_factor.npy."
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/HEST-1k/processed"))
    parser.add_argument("--slide-id", action="append", default=None)
    parser.add_argument("--min-total-counts", type=float, default=1.0)
    parser.add_argument("--out-csv", type=Path, default=Path("results/hest1k_human_visium_sf/processed_slide_audit.csv"))
    args = parser.parse_args()

    import pandas as pd

    processed_root = resolve_project_path(args.processed_root)
    if args.slide_id:
        slide_dirs = [processed_root / sid for sid in args.slide_id]
    else:
        slide_dirs = sorted([path for path in processed_root.iterdir() if path.is_dir()]) if processed_root.exists() else []
    rows = [validate_slide(path, min_total_counts=args.min_total_counts) for path in slide_dirs]
    out = pd.DataFrame(rows)
    out_path = resolve_project_path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"checked_slides={len(out)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
