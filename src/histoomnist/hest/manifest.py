from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


MANIFEST_COLUMNS = [
    "sample_id",
    "patient_id",
    "cohort",
    "organ",
    "disease_state",
    "platform",
    "dataset_title",
    "study_link",
    "split",
    "features_path",
    "counts_path",
    "coords_path",
    "size_factor_path",
    "spots_path",
    "genes_path",
    "n_spots",
    "n_genes",
    "sf_normalization",
]


def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, start=base).replace("\\", "/")


def processed_slide_paths(processed_root: str | Path, sample_id: str) -> dict[str, Path]:
    slide_dir = Path(processed_root) / str(sample_id)
    return {
        "features_path": slide_dir / "features.npy",
        "counts_path": slide_dir / "counts.npz",
        "coords_path": slide_dir / "coords.npy",
        "size_factor_path": slide_dir / "size_factor.npy",
        "spots_path": slide_dir / "spots.txt",
        "genes_path": slide_dir / "genes.txt",
    }


def build_hest_manifest(
    metadata: pd.DataFrame,
    *,
    processed_root: str | Path,
    manifest_path: str | Path,
    sf_normalization: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a training manifest and a full candidate asset-status table."""

    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    processed_root = Path(processed_root)
    candidates: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        sample_id = str(row.id)
        paths = processed_slide_paths(processed_root, sample_id)
        exists = {name: path.exists() for name, path in paths.items()}
        required_ok = exists["features_path"] and exists["counts_path"] and exists["coords_path"]
        status = {
            "sample_id": sample_id,
            "organ": getattr(row, "organ", ""),
            "platform": getattr(row, "st_technology", ""),
            "dataset_title": getattr(row, "dataset_title", ""),
            "n_spots_metadata": getattr(row, "spots_under_tissue", None),
            "n_genes_metadata": getattr(row, "nb_genes", None),
            "has_features": exists["features_path"],
            "has_counts": exists["counts_path"],
            "has_coords": exists["coords_path"],
            "has_size_factor": exists["size_factor_path"],
            "ready_for_training": required_ok,
        }
        candidates.append(status)
        if not required_ok:
            continue
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": getattr(row, "patient", sample_id),
                "cohort": getattr(row, "dataset_title", ""),
                "organ": getattr(row, "organ", ""),
                "disease_state": getattr(row, "disease_state", ""),
                "platform": getattr(row, "st_technology", ""),
                "dataset_title": getattr(row, "dataset_title", ""),
                "study_link": getattr(row, "study_link", ""),
                "split": "unsplit",
                "features_path": _rel(paths["features_path"], manifest_dir),
                "counts_path": _rel(paths["counts_path"], manifest_dir),
                "coords_path": _rel(paths["coords_path"], manifest_dir),
                "size_factor_path": _rel(paths["size_factor_path"], manifest_dir) if exists["size_factor_path"] else "",
                "spots_path": _rel(paths["spots_path"], manifest_dir) if exists["spots_path"] else "",
                "genes_path": _rel(paths["genes_path"], manifest_dir) if exists["genes_path"] else "",
                "n_spots": getattr(row, "spots_under_tissue", None),
                "n_genes": getattr(row, "nb_genes", None),
                "sf_normalization": sf_normalization,
            }
        )
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    candidates_df = pd.DataFrame(candidates)
    return manifest, candidates_df


def write_manifest_outputs(
    *,
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    manifest_path: str | Path,
    candidate_path: str | Path,
) -> None:
    manifest_path = Path(manifest_path)
    candidate_path = Path(candidate_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    candidates.to_csv(candidate_path, index=False)
