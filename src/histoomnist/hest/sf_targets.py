from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from histoomnist.data.size_factor import compute_size_factor, log_size_factor


def save_mean_one_size_factor(
    counts_path: str | Path,
    output_path: str | Path,
    *,
    min_total_counts: float = 1.0,
) -> np.ndarray:
    """Compute and save the committed mean-one SF target from a count matrix."""

    counts = sparse.load_npz(counts_path)
    sf, _ = compute_size_factor(counts, min_total_counts=min_total_counts)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, sf.astype(np.float32))
    return log_size_factor(sf)
