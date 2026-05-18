from __future__ import annotations

from pathlib import Path

import numpy as np


def load_gene_names(path: str | Path | None) -> list[str] | None:
    if path in (None, ""):
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Selected gene list not found: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_gene_indices(path: str | Path | None) -> np.ndarray | None:
    if path in (None, ""):
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gene index file not found: {p}")
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False).astype(np.int64)
    values = [int(line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray(values, dtype=np.int64)


def selected_genes_from_config(cfg: dict, *, base_dir: str | Path | None = None) -> tuple[list[str] | None, np.ndarray | None]:
    data_cfg = cfg.get("data", {})
    base = Path(base_dir) if base_dir is not None else Path(".")
    gene_names_path = data_cfg.get("gene_names_path")
    gene_indices_path = data_cfg.get("gene_indices_path")
    gene_names = load_gene_names(base / gene_names_path if gene_names_path else None)
    gene_indices = load_gene_indices(base / gene_indices_path if gene_indices_path else None)
    if gene_names is not None and gene_indices is not None:
        raise ValueError("Use either data.gene_names_path or data.gene_indices_path, not both.")
    return gene_names, gene_indices
