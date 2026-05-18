from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from histoomnist.utils.config import load_config
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import project_root, resolve_project_path


MANIFEST_PATH_COLUMNS = (
    "features_path",
    "counts_path",
    "coords_path",
    "size_factor_path",
    "spots_path",
    "genes_path",
)


def slugify(value: object, *, max_len: int = 80) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "all"
    return text[:max_len].strip("_") or "all"


def rel_project_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def stable_score(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def add_validation_split(
    task_manifest: pd.DataFrame,
    *,
    seed: int,
    heldout: str,
    val_fraction: float,
) -> pd.DataFrame:
    out = task_manifest.copy()
    if "val" in set(out["split"].astype(str)):
        return out
    train_ids = out.loc[out["split"].eq("train"), "sample_id"].astype(str).tolist()
    if len(train_ids) < 2 or val_fraction <= 0:
        return out
    n_val = max(1, int(round(len(train_ids) * float(val_fraction))))
    n_val = min(n_val, len(train_ids) - 1)
    ranked = sorted(train_ids, key=lambda sid: stable_score(seed, heldout, sid))
    val_ids = set(ranked[:n_val])
    out.loc[out["sample_id"].astype(str).isin(val_ids), "split"] = "val"
    return out


def split_tasks(split_table: pd.DataFrame, split_type: str) -> list[tuple[str, pd.DataFrame]]:
    rows = split_table[split_table["split_type"].astype(str).eq(split_type)].copy()
    if rows.empty:
        return []
    if split_type == "leave_slide_out":
        return [("leave_slide_out", rows)]
    tasks = []
    for heldout, group in rows.groupby("heldout", dropna=False, sort=True):
        tasks.append((str(heldout), group.copy()))
    return tasks


def build_task_manifest(
    manifest: pd.DataFrame,
    split_rows: pd.DataFrame,
    *,
    split_type: str,
    heldout: str,
    seed: int,
    val_fraction: float,
) -> pd.DataFrame:
    splits = split_rows[["sample_id", "split"]].drop_duplicates("sample_id").copy()
    out = manifest.drop(columns=["split"], errors="ignore").merge(splits, on="sample_id", how="left")
    out = out[out["split"].notna()].copy()
    out["split"] = out["split"].astype(str)
    out = add_validation_split(out, seed=seed, heldout=f"{split_type}:{heldout}", val_fraction=val_fraction)
    return out.reset_index(drop=True)


def summarise_task(
    task_manifest: pd.DataFrame,
    *,
    split_type: str,
    heldout: str,
    task_slug: str,
) -> dict[str, Any]:
    split_counts = task_manifest["split"].value_counts().to_dict()
    train = task_manifest[task_manifest["split"].eq("train")]
    val = task_manifest[task_manifest["split"].eq("val")]
    test = task_manifest[task_manifest["split"].eq("test")]
    row: dict[str, Any] = {
        "split_type": split_type,
        "heldout": heldout,
        "task_slug": task_slug,
        "n_slides": int(len(task_manifest)),
        "n_train_slides": int(split_counts.get("train", 0)),
        "n_val_slides": int(split_counts.get("val", 0)),
        "n_test_slides": int(split_counts.get("test", 0)),
        "has_train_val_test": bool(len(train) > 0 and len(val) > 0 and len(test) > 0),
        "train_organs": "|".join(sorted(train["organ"].dropna().astype(str).unique())),
        "test_organs": "|".join(sorted(test["organ"].dropna().astype(str).unique())),
        "train_cohorts": "|".join(sorted(train["cohort"].dropna().astype(str).unique())),
        "test_cohorts": "|".join(sorted(test["cohort"].dropna().astype(str).unique())),
    }
    return row


def summarise_manifest_assets(task_manifest: pd.DataFrame, *, source_base: Path) -> dict[str, Any]:
    checked = 0
    missing = 0
    examples: list[str] = []
    for row in task_manifest.itertuples(index=False):
        sample_id = str(getattr(row, "sample_id", "unknown"))
        for column in MANIFEST_PATH_COLUMNS:
            if not hasattr(row, column):
                continue
            value = getattr(row, column)
            if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
                continue
            checked += 1
            path = Path(str(value))
            source_path = path if path.is_absolute() else source_base / path
            if not source_path.resolve(strict=False).exists():
                missing += 1
                if len(examples) < 5:
                    examples.append(f"{sample_id}:{column}:{str(source_path).replace(chr(92), '/')}")
    return {
        "n_asset_paths_checked": int(checked),
        "n_missing_asset_paths": int(missing),
        "assets_ready": bool(missing == 0),
        "missing_asset_examples": "|".join(examples),
    }


def make_expression_task_config(
    base_cfg: dict[str, Any],
    *,
    task_name: str,
    manifest_path: Path,
    output_dir: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("project", {})["name"] = f"hest1k_expression_{task_name}"
    cfg["data"]["manifest"] = rel_project_path(manifest_path)
    if "paths" in cfg:
        cfg["paths"]["manifest"] = rel_project_path(manifest_path)
    cfg["data"]["train_splits"] = ["train"]
    cfg["data"]["val_splits"] = ["val"]
    cfg["data"]["test_splits"] = ["test"]
    cfg.setdefault("output", {})["dir"] = output_dir
    return cfg


def make_sf_task_config(
    base_cfg: dict[str, Any],
    *,
    task_name: str,
    manifest_path: Path,
    output_dir: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("project", {})["name"] = f"hest1k_sf_{task_name}"
    cfg["data"]["manifest"] = rel_project_path(manifest_path)
    if "paths" in cfg:
        cfg["paths"]["manifest"] = rel_project_path(manifest_path)
    cfg["data"]["train_splits"] = ["train"]
    cfg["data"]["val_splits"] = ["val"]
    cfg["data"]["test_splits"] = ["test"]
    cfg.setdefault("output", {})["dir"] = output_dir
    return cfg


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def rebase_manifest_paths(task_manifest: pd.DataFrame, *, source_base: Path, target_base: Path) -> pd.DataFrame:
    out = task_manifest.copy()
    for column in MANIFEST_PATH_COLUMNS:
        if column not in out.columns:
            continue
        values = []
        for value in out[column]:
            if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
                values.append(value)
                continue
            path = Path(str(value))
            source_path = path if path.is_absolute() else source_base / path
            values.append(os.path.relpath(source_path, start=target_base).replace("\\", "/"))
        out[column] = values
    return out


def write_task_files(
    *,
    task_manifest: pd.DataFrame,
    source_manifest_base: Path,
    source_gene_names_path: Path | None,
    split_type: str,
    heldout: str,
    task_slug: str,
    out_dir: Path,
    expression_config: dict[str, Any],
    sf_config: dict[str, Any],
) -> dict[str, str]:
    task_name = f"{split_type}_{task_slug}"
    task_dir = out_dir / "task_files" / split_type / task_slug
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "manifest.csv"
    expression_cfg_path = task_dir / "expression_config.yaml"
    sf_cfg_path = task_dir / "sf_config.yaml"
    rebased_manifest = rebase_manifest_paths(
        task_manifest,
        source_base=source_manifest_base,
        target_base=task_dir,
    )
    rebased_manifest.to_csv(manifest_path, index=False)
    expression_output = f"checkpoints/hest1k_human_visium_expression/generalization/{split_type}/{task_slug}"
    sf_output = f"checkpoints/hest1k_human_visium_sf/generalization/{split_type}/{task_slug}"
    expression_task_cfg = make_expression_task_config(
        expression_config,
        task_name=task_name,
        manifest_path=manifest_path,
        output_dir=expression_output,
    )
    if source_gene_names_path is not None:
        expression_task_cfg["data"]["gene_names_path"] = os.path.relpath(
            source_gene_names_path,
            start=task_dir,
        ).replace("\\", "/")
    write_yaml(expression_cfg_path, expression_task_cfg)
    write_yaml(
        sf_cfg_path,
        make_sf_task_config(
            sf_config,
            task_name=task_name,
            manifest_path=manifest_path,
            output_dir=sf_output,
        ),
    )
    return {
        "task_manifest": rel_project_path(manifest_path),
        "expression_config": rel_project_path(expression_cfg_path),
        "sf_config": rel_project_path(sf_cfg_path),
        "train_expression_command": f"python scripts/train_expression.py --config {rel_project_path(expression_cfg_path)}",
        "train_sf_command": f"python scripts/train_sf.py --config {rel_project_path(sf_cfg_path)}",
    }


def audit_generalization_tasks(
    *,
    expression_config: dict[str, Any],
    sf_config: dict[str, Any],
    split_dir: Path,
    split_types: list[str],
    out_dir: Path,
    seed: int,
    val_fraction: float,
    min_test_slides: int,
    write_files: bool,
    max_task_files_per_type: int | None,
) -> dict[str, Any]:
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression manifest path resolved to None")
    manifest = read_manifest(manifest_path)
    source_gene_names_path = None
    gene_names_path = expression_config.get("data", {}).get("gene_names_path")
    if gene_names_path not in (None, ""):
        source_gene_names_path = manifest_path.parent / str(gene_names_path)
    rows = []
    file_rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_type in split_types:
        split_path = split_dir / f"{split_type}.csv"
        if not split_path.exists():
            continue
        split_table = pd.read_csv(split_path)
        written_for_type = 0
        for heldout, split_rows in split_tasks(split_table, split_type):
            task_slug = slugify(heldout if split_type != "leave_slide_out" else split_type)
            task_manifest = build_task_manifest(
                manifest,
                split_rows,
                split_type=split_type,
                heldout=heldout,
                seed=seed,
                val_fraction=val_fraction,
            )
            summary = summarise_task(
                task_manifest,
                split_type=split_type,
                heldout=heldout,
                task_slug=task_slug,
            )
            summary.update(summarise_manifest_assets(task_manifest, source_base=manifest_path.parent))
            summary["ready_for_split_specific_training"] = bool(summary["has_train_val_test"] and summary["assets_ready"])
            summary["passes_min_test_slides"] = bool(summary["n_test_slides"] >= int(min_test_slides))
            rows.append(summary)
            can_write = write_files and summary["ready_for_split_specific_training"] and summary["passes_min_test_slides"]
            if can_write and max_task_files_per_type is not None and written_for_type >= int(max_task_files_per_type):
                can_write = False
            if can_write:
                file_info = write_task_files(
                    task_manifest=task_manifest,
                    source_manifest_base=manifest_path.parent,
                    source_gene_names_path=source_gene_names_path,
                    split_type=split_type,
                    heldout=heldout,
                    task_slug=task_slug,
                    out_dir=out_dir,
                    expression_config=expression_config,
                    sf_config=sf_config,
                )
                file_rows.append({**summary, **file_info})
                written_for_type += 1
    summary_frame = pd.DataFrame(rows).sort_values(
        ["split_type", "n_test_slides", "heldout"],
        ascending=[True, False, True],
    )
    file_frame = pd.DataFrame(file_rows)
    summary_frame.to_csv(out_dir / "task_summary.csv", index=False)
    file_frame.to_csv(out_dir / "generated_task_files.csv", index=False)
    run_summary = {
        "expression_manifest": rel_project_path(manifest_path),
        "split_types": split_types,
        "n_tasks": int(len(summary_frame)),
        "n_ready_tasks": int(summary_frame["ready_for_split_specific_training"].sum()) if not summary_frame.empty else 0,
        "n_generated_task_files": int(len(file_frame)),
        "min_test_slides": int(min_test_slides),
        "val_fraction": float(val_fraction),
        "outputs": {
            "task_summary": rel_project_path(out_dir / "task_summary.csv"),
            "generated_task_files": rel_project_path(out_dir / "generated_task_files.csv"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2), flush=True)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and materialize HEST coverage95 generalization tasks.")
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--sf-config", default="configs/hest1k_human_visium_sf_highconf_context_distribution_light.yaml")
    parser.add_argument("--split-dir", default="data/HEST-1k/splits")
    parser.add_argument("--split-types", nargs="*", default=["leave_slide_out", "leave_organ_out", "leave_cohort_out"])
    parser.add_argument("--out-dir", default="results/hest1k_human_visium_expression/generalization_readiness")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-test-slides", type=int, default=5)
    parser.add_argument("--write-task-files", action="store_true")
    parser.add_argument("--max-task-files-per-type", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_config = load_config(resolve_project_path(args.expression_config))
    sf_config = load_config(resolve_project_path(args.sf_config))
    split_dir = resolve_project_path(args.split_dir)
    out_dir = resolve_project_path(args.out_dir)
    if split_dir is None or out_dir is None:
        raise ValueError("split-dir or out-dir resolved to None")
    audit_generalization_tasks(
        expression_config=expression_config,
        sf_config=sf_config,
        split_dir=split_dir,
        split_types=[str(x) for x in args.split_types],
        out_dir=out_dir,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
        min_test_slides=int(args.min_test_slides),
        write_files=bool(args.write_task_files),
        max_task_files_per_type=args.max_task_files_per_type,
    )


if __name__ == "__main__":
    main()
