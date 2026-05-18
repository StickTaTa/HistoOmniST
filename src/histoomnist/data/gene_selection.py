from __future__ import annotations

from pathlib import Path

import numpy as np

INVALID_GENE_KEYS = {"", "nan", "none", "null", "unspecified_gene_id"}


def normalize_gene_key(value: object) -> str | None:
    key = str(value).strip()
    if key.lower() in INVALID_GENE_KEYS:
        return None
    return key


def load_gene_names(path: str | Path | None) -> list[str] | None:
    if path in (None, ""):
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Selected gene list not found: {p}")
    genes = []
    for line in p.read_text(encoding="utf-8").splitlines():
        key = normalize_gene_key(line)
        if key is not None:
            genes.append(key)
    return genes


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


def gene_key_settings_from_config(cfg: dict) -> tuple[str, Path | None]:
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    gene_key = str(data_cfg.get("gene_key", "var_names"))
    raw_st_root = data_cfg.get("raw_st_root")
    if raw_st_root in (None, ""):
        raw_root = paths_cfg.get("raw_root")
        raw_st_root = Path(raw_root) / "st" if raw_root not in (None, "") else None
    return gene_key, None if raw_st_root in (None, "") else Path(raw_st_root)


def load_h5ad_gene_symbols(h5ad_path: str | Path) -> list[str | None]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install anndata to read canonical H5AD gene symbols.") from exc

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if "SYMBOL" in adata.var.columns:
            values = adata.var["SYMBOL"].to_numpy()
        else:
            values = adata.var_names
        return [normalize_gene_key(value) for value in values]
    finally:
        adata.file.close()


def load_gene_keys_for_slide(
    *,
    sample_id: str,
    processed_gene_path: str | Path,
    gene_key: str = "var_names",
    raw_st_root: str | Path | None = None,
) -> list[str | None]:
    if gene_key in ("var_names", "genes_path"):
        return [normalize_gene_key(line) for line in Path(processed_gene_path).read_text(encoding="utf-8").splitlines()]
    if gene_key in ("symbol", "canonical_symbol"):
        if raw_st_root is None:
            raise ValueError("raw_st_root is required when data.gene_key is 'symbol'.")
        return load_h5ad_gene_symbols(Path(raw_st_root) / f"{sample_id}.h5ad")
    raise ValueError(f"Unsupported gene_key: {gene_key}")
