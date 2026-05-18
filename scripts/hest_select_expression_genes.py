from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.utils.io import read_manifest  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select common expressed genes for HEST expression-rate modeling.")
    parser.add_argument("--manifest", type=Path, default=Path("data/HEST-1k/manifests/human_visium_sf_manifest_highconf_context.csv"))
    parser.add_argument("--splits", nargs="*", default=["train"])
    parser.add_argument("--top-n", type=int, default=256)
    parser.add_argument("--min-detected-spots", type=int, default=100)
    parser.add_argument("--min-slides-present", type=int, default=30)
    parser.add_argument("--out-genes", type=Path, default=Path("data/HEST-1k/manifests/highconf_top256_genes.txt"))
    parser.add_argument("--out-report", type=Path, default=Path("results/hest1k_human_visium_sf/highconf_top256_gene_selection.csv"))
    return parser.parse_args()


def _optional_path(row, name: str):
    if not hasattr(row, name):
        return None
    value = getattr(row, name)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if str(value).strip() == "":
        return None
    return value


def read_genes(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def valid_mask_for_counts(counts) -> np.ndarray:
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    return np.isfinite(totals) & (totals >= 1.0)


def main() -> None:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest)
    manifest = read_manifest(manifest_path)
    manifest = manifest[manifest["split"].isin(args.splits)].copy()
    if manifest.empty:
        raise ValueError(f"No manifest rows for splits={args.splits}")
    base_dir = manifest_path.parent

    slide_gene_lists: dict[str, list[str]] = {}
    present_slides: dict[str, int] = {}
    for row in manifest.itertuples(index=False):
        genes_path = _optional_path(row, "genes_path")
        if genes_path is None:
            raise ValueError(f"Missing genes_path for {row.sample_id}")
        genes = read_genes(base_dir / str(genes_path))
        slide_gene_lists[str(row.sample_id)] = genes
        for gene in set(genes):
            present_slides[gene] = present_slides.get(gene, 0) + 1
    candidate_genes = sorted(
        gene for gene, n_slides in present_slides.items() if n_slides >= int(args.min_slides_present)
    )
    if not candidate_genes:
        raise ValueError("No candidate genes passed the min-slides-present filter.")
    candidate_index = {gene: idx for idx, gene in enumerate(candidate_genes)}

    total_counts = np.zeros(len(candidate_genes), dtype=np.float64)
    detected_spots = np.zeros(len(candidate_genes), dtype=np.int64)
    n_valid_spots = 0

    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        counts = sparse.load_npz(base_dir / str(row.counts_path)).tocsr()
        mask = valid_mask_for_counts(counts)
        counts = counts[mask]
        n_valid_spots += int(mask.sum())
        genes = slide_gene_lists[sample_id]
        slide_gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
        pairs = [
            (candidate_index[gene], slide_gene_to_index[gene])
            for gene in genes
            if gene in candidate_index
        ]
        if not pairs:
            continue
        target_indices = np.asarray([item[0] for item in pairs], dtype=np.int64)
        indices = np.asarray([item[1] for item in pairs], dtype=np.int64)
        sub = counts[:, indices].tocsr()
        total_counts[target_indices] += np.asarray(sub.sum(axis=0)).reshape(-1)
        detected_spots[target_indices] += np.asarray((sub > 0).sum(axis=0)).reshape(-1)
        print(f"processed {sample_id} valid_spots={int(mask.sum())}", flush=True)

    mean_count = total_counts / max(n_valid_spots, 1)
    report = pd.DataFrame(
        {
            "gene": candidate_genes,
            "slides_present": [present_slides[gene] for gene in candidate_genes],
            "total_count": total_counts,
            "mean_count": mean_count,
            "detected_spots": detected_spots,
            "detected_fraction": detected_spots / max(n_valid_spots, 1),
        }
    )
    report = report[report["detected_spots"] >= int(args.min_detected_spots)].copy()
    report = report.sort_values(["detected_spots", "total_count"], ascending=False).reset_index(drop=True)
    selected = report.head(int(args.top_n)).copy()
    if selected.empty:
        raise ValueError("No genes passed selection filters.")

    out_genes = resolve_project_path(args.out_genes)
    out_report = resolve_project_path(args.out_report)
    out_genes.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_genes.write_text("\n".join(selected["gene"].astype(str).tolist()) + "\n", encoding="utf-8")
    report.to_csv(out_report, index=False)
    print(f"candidate_genes={len(candidate_genes)} valid_spots={n_valid_spots}")
    print(f"selected_genes={len(selected)}")
    print(f"wrote {out_genes}")
    print(f"wrote {out_report}")


if __name__ == "__main__":
    main()
