from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def resolve_project_path(path: str | Path | None, base: str | Path | None = None) -> Path | None:
    if path in (None, ""):
        return None
    p = Path(str(path))
    if p.is_absolute():
        return p
    return (Path(base) if base is not None else project_root()) / p


@dataclass(frozen=True)
class LocalPaths:
    old_project_root: Path
    new_project_root: Path
    manuscript_root: Path
    data_root: Path | None
    output_root: Path
    figure_root: Path
    checkpoint_root: Path
    run_root: Path
    existing_output_root: Path
    paper_automation_root: Path
    hest1k_root: Path
    hest1k_metadata_csv: Path
    hest1k_raw_root: Path
    hest1k_processed_root: Path


def load_local_paths(config_path: str | Path | None = None) -> LocalPaths:
    cfg_path = resolve_project_path(config_path or "configs/local_paths.yaml")
    root = project_root()
    if cfg_path is None or not cfg_path.exists():
        data = {
            "old_project_root": "E:/Morpho-FM",
            "new_project_root": str(root),
            "manuscript_root": "C:/Users/Administrator/Desktop/人生的第4篇论文/manuscript",
            "data_root": str(root / "data"),
            "output_root": str(root / "results"),
            "figure_root": str(root / "figures"),
            "checkpoint_root": str(root / "checkpoints"),
            "run_root": str(root / "runs"),
            "existing_output_root": str(root / "outputs"),
            "paper_automation_root": str(root / "results" / "paper_automation"),
            "hest1k_root": str(root / "data" / "HEST-1k"),
            "hest1k_metadata_csv": str(root / "data" / "HEST-1k" / "HEST_v1_3_0.csv"),
            "hest1k_raw_root": str(root / "data" / "HEST-1k" / "raw"),
            "hest1k_processed_root": str(root / "data" / "HEST-1k" / "processed"),
        }
    else:
        data = load_yaml(cfg_path)
    project = Path(data["new_project_root"])
    hest_root = Path(data.get("hest1k_root", project / "data" / "HEST-1k"))
    return LocalPaths(
        old_project_root=Path(data["old_project_root"]),
        new_project_root=project,
        manuscript_root=Path(data["manuscript_root"]),
        data_root=None if data.get("data_root") in (None, "") else Path(data["data_root"]),
        output_root=Path(data["output_root"]),
        figure_root=Path(data["figure_root"]),
        checkpoint_root=Path(data["checkpoint_root"]),
        run_root=Path(data.get("run_root", project / "runs")),
        existing_output_root=Path(data.get("existing_output_root", project / "outputs")),
        paper_automation_root=Path(data.get("paper_automation_root", project / "results" / "paper_automation")),
        hest1k_root=hest_root,
        hest1k_metadata_csv=Path(data.get("hest1k_metadata_csv", hest_root / "HEST_v1_3_0.csv")),
        hest1k_raw_root=Path(data.get("hest1k_raw_root", hest_root / "raw")),
        hest1k_processed_root=Path(data.get("hest1k_processed_root", hest_root / "processed")),
    )


def ensure_standard_dirs(root: str | Path | None = None) -> list[Path]:
    base = Path(root) if root is not None else project_root()
    rels = [
        "configs",
        "docs",
        "scripts",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/visualization",
        "src/utils",
        "data/HEST-1k/raw",
        "data/HEST-1k/processed",
        "data/HEST-1k/manifests",
        "data/HEST-1k/splits",
        "results/hest1k_human_visium_sf/metadata_audit",
        "results/hest1k_human_visium_sf/baselines",
        "results/hest1k_human_visium_sf/logs",
        "figures",
        "checkpoints/hest1k_human_visium_sf",
        "runs/hest1k_human_visium_sf",
    ]
    paths = [base / rel for rel in rels]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths
