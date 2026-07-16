from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

INVALID_GENE_KEYS = {"", "nan", "none", "null", "unspecified_gene_id"}
ENSEMBL_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$")
DEFAULT_ENSEMBL_SYMBOL_MAP = "hest_local_ensembl_to_symbol_map.csv"


def normalize_gene_key(value: object) -> str | None:
    key = str(value).strip()
    if key.lower() in INVALID_GENE_KEYS:
        return None
    return key


def ensembl_stable_id(value: object) -> str | None:
    key = normalize_gene_key(value)
    if key is None or ENSEMBL_RE.match(key) is None:
        return None
    return key.split(".", 1)[0]


def load_gene_names(path: str | Path | None) -> list[str] | None:
    if path in (None, ""):
        return None
    p = Path(path).resolve()
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
    p = Path(path).resolve()
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


def default_ensembl_symbol_map_path(h5ad_path: str | Path) -> Path:
    path = Path(h5ad_path).resolve()
    try:
        return path.parents[2] / "manifests" / DEFAULT_ENSEMBL_SYMBOL_MAP
    except IndexError:
        return path.parent / DEFAULT_ENSEMBL_SYMBOL_MAP


def load_ensembl_symbol_map(path: str | Path) -> dict[str, str]:
    p = Path(path).resolve()
    if not p.exists():
        return {}
    frame = pd.read_csv(p)
    required = {"ensembl_id", "symbol"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Ensembl symbol map must contain columns {sorted(required)}: {p}")
    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        ensembl_id = ensembl_stable_id(getattr(row, "ensembl_id"))
        symbol = normalize_gene_key(getattr(row, "symbol"))
        if ensembl_id is None or symbol is None:
            continue
        mapping[ensembl_id] = symbol
    return mapping


def map_ensembl_keys_to_symbols(keys: list[str | None], map_path: str | Path) -> list[str | None]:
    mapping = load_ensembl_symbol_map(map_path)
    if not mapping:
        return keys
    mapped: list[str | None] = []
    for key in keys:
        ensembl_id = ensembl_stable_id(key)
        mapped.append(mapping.get(ensembl_id, key) if ensembl_id is not None else key)
    return mapped


def load_h5ad_gene_symbols(h5ad_path: str | Path) -> list[str | None]:
    try:
        import anndata as ad
    except ImportError as exc:
        try:
            import h5py
        except ImportError:
            raise ImportError("Install anndata or h5py to read canonical H5AD gene symbols.") from exc

        def _decode_values(values) -> list[str | None]:
            keys: list[str | None] = []
            for value in values:
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                keys.append(normalize_gene_key(value))
            return keys

        with h5py.File(h5ad_path, "r") as handle:
            var = handle["var"]
            has_symbol = "SYMBOL" in var
            if has_symbol:
                keys = _decode_values(var["SYMBOL"][()])
            elif "_index" in var:
                keys = _decode_values(var["_index"][()])
            else:
                raise ValueError(f"H5AD lacks var/SYMBOL and var/_index: {h5ad_path}")
        if not has_symbol:
            keys = map_ensembl_keys_to_symbols(keys, default_ensembl_symbol_map_path(h5ad_path))
        return keys

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        has_symbol = "SYMBOL" in adata.var.columns
        if has_symbol:
            values = adata.var["SYMBOL"].to_numpy()
        else:
            values = adata.var_names
        keys = [normalize_gene_key(value) for value in values]
    finally:
        adata.file.close()
    if not has_symbol:
        keys = map_ensembl_keys_to_symbols(keys, default_ensembl_symbol_map_path(h5ad_path))
    return keys


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
