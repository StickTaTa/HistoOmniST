from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from histoomnist.data.gene_selection import gene_key_settings_from_config, selected_genes_from_config
from histoomnist.data.gene_selection import load_gene_keys_for_slide
from histoomnist.utils.config import load_config
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import load_local_paths, resolve_project_path


@dataclass(frozen=True)
class MethodSpec:
    method: str
    source_rel: str
    key_files: tuple[str, ...]
    dependency_modules: tuple[str, ...]
    original_requires_wsi: bool
    patch_h5_adapter_possible: bool
    pretrained_files: tuple[str, ...] = ()
    adapter_priority: int = 99
    first_adapter_reason: str = ""


METHOD_SPECS = (
    MethodSpec(
        method="HisToGene",
        source_rel="HisToGene",
        key_files=("dataset.py", "vis_model.py", "transformer.py", "predict.py"),
        dependency_modules=("torch", "torchvision", "pytorch_lightning", "anndata", "PIL"),
        original_requires_wsi=True,
        patch_h5_adapter_possible=True,
        adapter_priority=1,
        first_adapter_reason="Simplest image-patch supervised baseline; original WSI crop can be replaced by HEST patch H5 img.",
    ),
    MethodSpec(
        method="THItoGene",
        source_rel="THItoGene",
        key_files=("dataset.py", "vis_model.py", "train.py", "predict.py"),
        dependency_modules=("torch", "torchvision", "pytorch_lightning", "anndata", "PIL"),
        original_requires_wsi=True,
        patch_h5_adapter_possible=True,
        adapter_priority=2,
        first_adapter_reason="Patch-H5 adapter is feasible, but whole-slide graph batches are heavier than HisToGene.",
    ),
    MethodSpec(
        method="mclSTExp",
        source_rel="mclSTExp",
        key_files=("dataset.py", "model.py", "train.py", "utils.py"),
        dependency_modules=("torch", "torchvision", "timm", "anndata", "sklearn"),
        original_requires_wsi=False,
        patch_h5_adapter_possible=True,
        adapter_priority=3,
        first_adapter_reason="Can use HEST patch images, but contrastive train/query alignment needs a larger adapter.",
    ),
    MethodSpec(
        method="HiST",
        source_rel="HiST",
        key_files=("src/PredictionModule/model.py", "src/PredictionModule/solver.py", "src/PredictionModule/dataset.py"),
        dependency_modules=("torch", "torchvision", "timm", "anndata", "sklearn"),
        original_requires_wsi=True,
        patch_h5_adapter_possible=False,
        pretrained_files=("resource/ctranspath.pth",),
        adapter_priority=4,
        first_adapter_reason="Pretrained cTransPath file is present, but original workflow expects WSI tiling/preprocessing.",
    ),
    MethodSpec(
        method="iStar",
        source_rel="istar",
        key_files=("train.py", "impute.py", "extract_features.py", "utils.py"),
        dependency_modules=("torch", "torchvision", "anndata", "sklearn", "PIL"),
        original_requires_wsi=True,
        patch_h5_adapter_possible=False,
        pretrained_files=("checkpoints/vit256_small_dino.pth", "checkpoints/vit4k_xs_dino.pth"),
        adapter_priority=5,
        first_adapter_reason="DINO checkpoints are present, but the method depends on WSI-scale feature extraction and imputation grids.",
    ),
    MethodSpec(
        method="sCellST",
        source_rel="sCellST",
        key_files=("full_pipeline_script.py", "scellst"),
        dependency_modules=("torch", "pytorch_lightning", "anndata", "sklearn"),
        original_requires_wsi=False,
        patch_h5_adapter_possible=True,
        adapter_priority=6,
        first_adapter_reason="Separate framework and configs; useful after direct patch-H5 supervised baselines are in place.",
    ),
)


def module_available(name: str) -> bool:
    if name == "PIL":
        return importlib.util.find_spec("PIL") is not None
    if name == "sklearn":
        return importlib.util.find_spec("sklearn") is not None
    return importlib.util.find_spec(name) is not None


def find_wsi(raw_root: Path, sample_id: str) -> Path | None:
    wsis = raw_root / "wsis"
    if not wsis.exists():
        return None
    for suffix in (".tif", ".tiff", ".svs", ".ndpi", ".mrxs"):
        path = wsis / f"{sample_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(wsis.glob(f"{sample_id}.*"))
    return matches[0] if matches else None


def optional_manifest_path(base_dir: Path, row, column: str) -> Path | None:
    value = getattr(row, column, "")
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if str(value).strip() == "":
        return None
    return base_dir / str(value)


def inspect_patch_h5(path: Path) -> dict[str, object]:
    out = {
        "patch_h5_has_img": False,
        "patch_h5_has_barcode": False,
        "patch_h5_has_coords": False,
        "patch_h5_n_img": 0,
        "patch_h5_img_shape": "",
        "patch_h5_img_dtype": "",
    }
    if not path.exists():
        return out
    with h5py.File(path, "r") as handle:
        out["patch_h5_has_img"] = "img" in handle
        out["patch_h5_has_barcode"] = "barcode" in handle
        out["patch_h5_has_coords"] = "coords" in handle
        if "img" in handle:
            img = handle["img"]
            out["patch_h5_n_img"] = int(img.shape[0])
            out["patch_h5_img_shape"] = "x".join(str(x) for x in img.shape[1:])
            out["patch_h5_img_dtype"] = str(img.dtype)
    return out


def count_target_genes_for_slide(
    *,
    sample_id: str,
    processed_gene_path: Path,
    target_genes: set[str],
    gene_key: str,
    raw_st_root: Path | None,
) -> int:
    keys = load_gene_keys_for_slide(
        sample_id=sample_id,
        processed_gene_path=processed_gene_path,
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    return len({gene for gene in keys if gene in target_genes})


def build_slide_tables(
    *,
    cfg: dict,
    splits: list[str],
    raw_root: Path,
    out_dir: Path,
    skip_gene_audit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = resolve_project_path(cfg["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Manifest path resolved to None")
    manifest = read_manifest(manifest_path)
    if splits:
        manifest = manifest[manifest["split"].isin(splits)].copy()
    if manifest.empty:
        raise ValueError(f"No manifest rows for splits={splits}")
    base_dir = manifest_path.parent
    target_genes, gene_indices = selected_genes_from_config(cfg, base_dir=base_dir)
    if target_genes is None or gene_indices is not None:
        raise ValueError("External benchmark readiness requires data.gene_names_path coverage95 target genes.")
    gene_key, raw_st_root = gene_key_settings_from_config(cfg)
    target_set = set(target_genes)

    slide_rows = []
    manifest_rows = []
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        raw_st = raw_root / "st" / f"{sample_id}.h5ad"
        raw_patches = raw_root / "patches" / f"{sample_id}.h5"
        raw_thumbnail = raw_root / "thumbnails" / f"{sample_id}_downscaled_fullres.jpeg"
        raw_wsi = find_wsi(raw_root, sample_id)
        processed_features = optional_manifest_path(base_dir, row, "features_path")
        processed_counts = optional_manifest_path(base_dir, row, "counts_path")
        processed_coords = optional_manifest_path(base_dir, row, "coords_path")
        processed_spots = optional_manifest_path(base_dir, row, "spots_path")
        processed_genes = optional_manifest_path(base_dir, row, "genes_path")
        patch_info = inspect_patch_h5(raw_patches)
        measured_target_genes = np.nan
        if not skip_gene_audit and processed_genes is not None and processed_genes.exists():
            measured_target_genes = count_target_genes_for_slide(
                sample_id=sample_id,
                processed_gene_path=processed_genes,
                target_genes=target_set,
                gene_key=gene_key,
                raw_st_root=raw_st_root,
            )
        common = {
            "sample_id": sample_id,
            "split": str(row.split),
            "organ": str(getattr(row, "organ", "")),
            "cohort": str(getattr(row, "cohort", "")),
            "disease_state": str(getattr(row, "disease_state", "")),
            "n_spots_manifest": int(getattr(row, "n_spots", 0)),
            "n_genes_manifest": int(getattr(row, "n_genes", 0)),
            "target_gene_count": int(len(target_genes)),
            "measured_target_genes": measured_target_genes,
            "raw_st_exists": raw_st.exists(),
            "raw_patches_exists": raw_patches.exists(),
            "raw_thumbnail_exists": raw_thumbnail.exists(),
            "raw_wsi_exists": raw_wsi is not None,
            "processed_features_exists": processed_features.exists() if processed_features else False,
            "processed_counts_exists": processed_counts.exists() if processed_counts else False,
            "processed_coords_exists": processed_coords.exists() if processed_coords else False,
            "processed_spots_exists": processed_spots.exists() if processed_spots else False,
            "processed_genes_exists": processed_genes.exists() if processed_genes else False,
            **patch_info,
        }
        slide_rows.append(common)
        manifest_rows.append(
            {
                **common,
                "raw_st_path": str(raw_st),
                "raw_patches_path": str(raw_patches),
                "raw_thumbnail_path": str(raw_thumbnail),
                "raw_wsi_path": "" if raw_wsi is None else str(raw_wsi),
                "processed_features_path": "" if processed_features is None else str(processed_features),
                "processed_counts_path": "" if processed_counts is None else str(processed_counts),
                "processed_coords_path": "" if processed_coords is None else str(processed_coords),
                "processed_spots_path": "" if processed_spots is None else str(processed_spots),
                "processed_genes_path": "" if processed_genes is None else str(processed_genes),
                "target_genes_path": str(base_dir / str(cfg["data"]["gene_names_path"])),
                "prediction_count_path": str(out_dir / "predictions" / f"{sample_id}_count.npy"),
                "prediction_rate_path": str(out_dir / "predictions" / f"{sample_id}_rate.npy"),
            }
        )
        print(
            f"[external-readiness] {sample_id}: patches={raw_patches.exists()} "
            f"img={patch_info['patch_h5_has_img']} wsi={raw_wsi is not None}",
            flush=True,
        )
    return pd.DataFrame(slide_rows), pd.DataFrame(manifest_rows)


def method_readiness(
    *,
    benchmark_root: Path,
    slide_assets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_have_patch_h5_img = bool(
        slide_assets["raw_patches_exists"].all()
        and slide_assets["patch_h5_has_img"].all()
        and slide_assets["patch_h5_has_barcode"].all()
        and slide_assets["patch_h5_has_coords"].all()
    )
    all_have_wsi = bool(slide_assets["raw_wsi_exists"].all())
    all_have_st = bool(slide_assets["raw_st_exists"].all())
    for spec in METHOD_SPECS:
        source_dir = benchmark_root / spec.source_rel
        key_present = [(source_dir / rel).exists() for rel in spec.key_files]
        dependencies = {name: module_available(name) for name in spec.dependency_modules}
        pretrained = [(source_dir / rel).exists() for rel in spec.pretrained_files]
        source_blockers = []
        original_workflow_blockers = []
        adapter_blockers = []
        environment_blockers = []
        if not source_dir.exists():
            source_blockers.append("missing_source_dir")
        if not all(key_present):
            source_blockers.append("missing_key_files")
        if not all_have_st:
            adapter_blockers.append("missing_raw_h5ad")
        if spec.original_requires_wsi and not all_have_wsi:
            original_workflow_blockers.append("original_workflow_missing_wsi")
        if spec.patch_h5_adapter_possible and not all_have_patch_h5_img:
            adapter_blockers.append("missing_patch_h5_img")
        if spec.pretrained_files and not all(pretrained):
            environment_blockers.append("missing_pretrained_files")
        missing_deps = sorted(name for name, ok in dependencies.items() if not ok)
        if missing_deps:
            environment_blockers.append("missing_python_modules:" + "|".join(missing_deps))
        patch_adapter_data_ready = (
            source_dir.exists()
            and all(key_present)
            and all_have_st
            and spec.patch_h5_adapter_possible
            and all_have_patch_h5_img
        )
        environment_ready = (not missing_deps) and (not spec.pretrained_files or all(pretrained))
        if patch_adapter_data_ready and environment_ready:
            adapter_status = "patch_h5_adapter_ready"
        elif patch_adapter_data_ready:
            adapter_status = "patch_h5_adapter_ready_missing_environment"
        elif (
            source_dir.exists()
            and all(key_present)
            and (not spec.patch_h5_adapter_possible)
            and (not spec.original_requires_wsi or all_have_wsi)
        ):
            adapter_status = "source_ready_original_assets"
        else:
            adapter_status = "blocked_or_needs_nontrivial_adapter"
        blockers = source_blockers + adapter_blockers + environment_blockers + original_workflow_blockers
        rows.append(
            {
                "method": spec.method,
                "source_dir": str(source_dir),
                "source_dir_exists": source_dir.exists(),
                "key_files_present": int(sum(key_present)),
                "key_files_total": len(spec.key_files),
                "pretrained_files_present": int(sum(pretrained)),
                "pretrained_files_total": len(spec.pretrained_files),
                "original_requires_wsi": spec.original_requires_wsi,
                "patch_h5_adapter_possible": spec.patch_h5_adapter_possible,
                "all_selected_slides_have_wsi": all_have_wsi,
                "all_selected_slides_have_patch_h5_img": all_have_patch_h5_img,
                "all_selected_slides_have_raw_h5ad": all_have_st,
                "missing_python_modules": "|".join(missing_deps),
                "source_blockers": "|".join(source_blockers),
                "adapter_blockers": "|".join(adapter_blockers),
                "environment_blockers": "|".join(environment_blockers),
                "original_workflow_blockers": "|".join(original_workflow_blockers),
                "blockers": "|".join(blockers),
                "adapter_status": adapter_status,
                "adapter_priority": spec.adapter_priority,
                "first_adapter_reason": spec.first_adapter_reason,
            }
        )
    readiness = pd.DataFrame(rows).sort_values(["adapter_priority", "method"]).reset_index(drop=True)
    priority = readiness[
        [
            "method",
            "adapter_priority",
            "adapter_status",
            "patch_h5_adapter_possible",
            "original_requires_wsi",
            "blockers",
            "first_adapter_reason",
        ]
    ].copy()
    return readiness, priority


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit assets and method readiness for HEST coverage95 external benchmarks.")
    parser.add_argument("--expression-config", default="configs/hest1k_human_visium_expression_highconf_symbol95.yaml")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--out-dir", default="results/hest1k_human_visium_expression/external_benchmark_readiness")
    parser.add_argument("--skip-gene-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.expression_config)
    local_paths = load_local_paths()
    benchmark_root = Path(args.benchmark_root) if args.benchmark_root else local_paths.old_project_root / "benchmark"
    raw_root = resolve_project_path(cfg["paths"]["raw_root"])
    if raw_root is None:
        raise ValueError("Raw root resolved to None")
    out_dir = resolve_project_path(args.out_dir)
    if out_dir is None:
        raise ValueError("Output dir resolved to None")
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_assets, input_manifest = build_slide_tables(
        cfg=cfg,
        splits=[str(x) for x in args.splits],
        raw_root=raw_root,
        out_dir=out_dir,
        skip_gene_audit=bool(args.skip_gene_audit),
    )
    readiness, priority = method_readiness(
        benchmark_root=benchmark_root,
        slide_assets=slide_assets,
    )
    slide_assets.to_csv(out_dir / "slide_assets.csv", index=False)
    input_manifest.to_csv(out_dir / "input_manifest.csv", index=False)
    readiness.to_csv(out_dir / "method_readiness.csv", index=False)
    priority.to_csv(out_dir / "adapter_priority.csv", index=False)
    summary = {
        "expression_config": str(args.expression_config),
        "benchmark_root": str(benchmark_root),
        "splits": [str(x) for x in args.splits],
        "n_slides": int(len(slide_assets)),
        "slide_asset_summary": {
            "raw_h5ad_all": bool(slide_assets["raw_st_exists"].all()),
            "patch_h5_all": bool(slide_assets["raw_patches_exists"].all()),
            "patch_h5_img_all": bool(slide_assets["patch_h5_has_img"].all()),
            "thumbnail_all": bool(slide_assets["raw_thumbnail_exists"].all()),
            "wsi_all": bool(slide_assets["raw_wsi_exists"].all()),
        },
        "ready_patch_h5_methods": readiness[
            readiness["adapter_status"].eq("patch_h5_adapter_ready")
        ]["method"].tolist(),
        "patch_h5_ready_missing_environment_methods": readiness[
            readiness["adapter_status"].eq("patch_h5_adapter_ready_missing_environment")
        ]["method"].tolist(),
        "first_priority_method": str(priority.iloc[0]["method"]) if not priority.empty else None,
        "outputs": {
            "slide_assets": str(out_dir / "slide_assets.csv"),
            "input_manifest": str(out_dir / "input_manifest.csv"),
            "method_readiness": str(out_dir / "method_readiness.csv"),
            "adapter_priority": str(out_dir / "adapter_priority.csv"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
