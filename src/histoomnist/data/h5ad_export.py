from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse


def export_h5ad_counts(
    h5ad_path: str | Path,
    out_dir: str | Path,
    count_layer: str | None = None,
    spatial_key: str = "spatial",
) -> dict[str, str]:
    """Export counts and coordinates from one AnnData file.

    This function deliberately exports raw count arrays only. HIPT/pathology features
    should be generated separately and then linked through the manifest.
    """

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install with `pip install -e .[h5ad]` to use h5ad export.") from exc

    h5ad_path = Path(h5ad_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(h5ad_path)
    counts = adata.layers[count_layer] if count_layer else adata.X
    counts_path = out_dir / "counts.npz"
    if sparse.issparse(counts):
        sparse.save_npz(counts_path, counts.tocsr())
    else:
        sparse.save_npz(counts_path, sparse.csr_matrix(np.asarray(counts)))
    coords = None
    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    elif {"array_row", "array_col"}.issubset(adata.obs.columns):
        coords = adata.obs[["array_row", "array_col"]].to_numpy(dtype=np.float32)
    if coords is not None:
        np.save(out_dir / "coords.npy", coords)
    genes = np.asarray(adata.var_names.astype(str))
    np.savetxt(out_dir / "genes.txt", genes, fmt="%s")
    spots = np.asarray(adata.obs_names.astype(str))
    np.savetxt(out_dir / "spots.txt", spots, fmt="%s")
    return {
        "counts_path": str(counts_path),
        "coords_path": str(out_dir / "coords.npy") if coords is not None else "",
        "genes_path": str(out_dir / "genes.txt"),
        "spots_path": str(out_dir / "spots.txt"),
    }
