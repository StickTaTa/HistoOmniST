from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.features.patch_features import (  # noqa: E402
    RGB_FEATURE_NAMES,
    batched,
    hipt256_feature_names,
    hipt256_features,
    load_hipt256_model,
    rgb_stats_features,
)
from histoomnist.hest.metadata import filter_hest_metadata, load_hest_metadata  # noqa: E402
from histoomnist.hest.raw_assets import raw_slide_paths  # noqa: E402
from histoomnist.utils.config import get_device_name, load_config  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract spot-level H&E features from HEST patch H5 files.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--local-paths", type=Path, default=Path("configs/local_paths.yaml"))
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--model", choices=["rgb_stats", "hipt256"], default="rgb_stats")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hipt-source-dir", type=Path, default=None)
    parser.add_argument("--hipt-weights", type=Path, default=None)
    parser.add_argument("--include-rgb-stats", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/hest1k_human_visium_sf/patch_feature_extraction.csv"),
    )
    return parser.parse_args()


def read_local_paths(path: Path) -> dict:
    path = resolve_project_path(path)
    if not path.exists():
        return {}
    return load_config(path)


def select_slides(cfg: dict, sample_ids: list[str] | None, max_slides: int | None) -> pd.DataFrame:
    metadata = load_hest_metadata(resolve_project_path(cfg["paths"]["metadata_csv"]))
    filters = cfg["filters"]
    selected = filter_hest_metadata(
        metadata,
        species=str(filters.get("species", "Homo sapiens")),
        st_technology=str(filters.get("st_technology", "Visium")),
        min_spots_under_tissue=int(filters.get("min_spots_under_tissue", 200)),
    )
    if sample_ids:
        wanted = set(sample_ids)
        selected = selected[selected["id"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(selected["id"].astype(str)))
        if missing:
            raise ValueError(f"sample_id values do not match selected metadata filters: {missing}")
    if max_slides:
        selected = selected.head(max_slides).copy()
    return selected.reset_index(drop=True)


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_patch_batch(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)
    if len(indices) == 1 or bool(np.all(np.diff(indices) > 0)):
        return np.asarray(dataset[indices])
    order = np.argsort(indices)
    sorted_images = np.asarray(dataset[indices[order]])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return sorted_images[inverse]


def resolve_hipt_paths(
    *,
    local_paths: dict,
    hipt_source_dir: Path | None,
    hipt_weights: Path | None,
) -> tuple[Path, Path]:
    source = hipt_source_dir
    weights = hipt_weights
    old_project_root = local_paths.get("old_project_root")
    if source is None:
        source_value = local_paths.get("hipt_source_dir")
        if source_value:
            source = Path(source_value)
        elif old_project_root:
            source = Path(old_project_root) / "src" / "st_pipeline" / "superres" / "hipt"
    if weights is None:
        weights_value = local_paths.get("hipt_vit256_weights")
        if weights_value:
            weights = Path(weights_value)
        elif old_project_root:
            weights = Path(old_project_root) / "checkpoints" / "hipt_backbone" / "vit256_small_dino.pth"
    if source is None or weights is None:
        raise ValueError(
            "HIPT extraction requires --hipt-source-dir/--hipt-weights or matching keys in configs/local_paths.yaml."
        )
    return source, weights


def extract_slide_features(
    *,
    slide_id: str,
    raw_root: Path,
    processed_root: Path,
    model_name: str,
    batch_size: int,
    overwrite: bool,
    hipt_model=None,
    device: str = "cpu",
    include_rgb_stats: bool = False,
    feature_names: list[str] | None = None,
) -> dict[str, object]:
    out_dir = processed_root / slide_id
    out_path = out_dir / "features.npy"
    patch_indices_path = out_dir / "patch_indices.npy"
    row: dict[str, object] = {
        "slide_id": slide_id,
        "model": model_name,
        "features_path": str(out_path),
    }
    if out_path.exists() and not overwrite:
        features = np.load(out_path, mmap_mode="r")
        row.update({"status": "exists", "n_spots": int(features.shape[0]), "n_features": int(features.shape[1])})
        return row
    if not patch_indices_path.exists():
        row["status"] = "missing_patch_indices"
        return row
    paths = raw_slide_paths(raw_root, slide_id)
    if not paths.patches.exists():
        row["status"] = "missing_patch_h5"
        return row

    patch_indices = np.load(patch_indices_path)
    features_chunks: list[np.ndarray] = []
    try:
        with h5py.File(paths.patches, "r") as patches:
            images = patches["img"]
            for batch_indices in batched(patch_indices, batch_size):
                batch_images = _read_patch_batch(images, batch_indices)
                if model_name == "rgb_stats":
                    batch_features = rgb_stats_features(batch_images)
                elif model_name == "hipt256":
                    if hipt_model is None:
                        raise ValueError("hipt_model is required for model='hipt256'")
                    batch_features = hipt256_features(batch_images, hipt_model, device=device)
                    if include_rgb_stats:
                        batch_features = np.concatenate(
                            [batch_features, rgb_stats_features(batch_images)],
                            axis=1,
                        )
                else:
                    raise ValueError(f"Unsupported feature model: {model_name}")
                features_chunks.append(batch_features)
        features = np.concatenate(features_chunks, axis=0).astype(np.float32)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, features)
        names = feature_names or [f"feature_{idx:04d}" for idx in range(features.shape[1])]
        (out_dir / "feature_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        metadata = {
            "feature_model": model_name,
            "n_spots": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "source_patches": str(paths.patches),
            "patch_indices": str(patch_indices_path),
            "include_rgb_stats": bool(include_rgb_stats),
        }
        (out_dir / "feature_metadata.json").write_text(
            json.dumps({k: _json_safe(v) for k, v in metadata.items()}, indent=2),
            encoding="utf-8",
        )
        row.update(
            {
                "status": "ok",
                "n_spots": int(features.shape[0]),
                "n_features": int(features.shape[1]),
                "feature_mean_abs": float(np.mean(np.abs(features))),
            }
        )
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}:{exc}"
    return row


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.config))
    raw_root = resolve_project_path(args.raw_root or cfg["paths"]["raw_root"])
    processed_root = resolve_project_path(args.processed_root or cfg["paths"]["processed_root"])
    slides = select_slides(cfg, args.sample_id, args.max_slides)

    local_paths = read_local_paths(args.local_paths)
    device = get_device_name(args.device)
    hipt_model = None
    feature_names = RGB_FEATURE_NAMES
    if args.model == "hipt256":
        hipt_source, hipt_weights = resolve_hipt_paths(
            local_paths=local_paths,
            hipt_source_dir=args.hipt_source_dir,
            hipt_weights=args.hipt_weights,
        )
        print(f"loading HIPT ViT-256 from {hipt_weights}")
        hipt_model = load_hipt256_model(
            hipt_source_dir=hipt_source,
            weights_path=hipt_weights,
            device=device,
        )
        feature_names = hipt256_feature_names()
        if args.include_rgb_stats:
            feature_names = feature_names + RGB_FEATURE_NAMES

    rows = []
    for idx, metadata_row in slides.iterrows():
        slide_id = str(metadata_row["id"])
        row = extract_slide_features(
            slide_id=slide_id,
            raw_root=raw_root,
            processed_root=processed_root,
            model_name=args.model,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            hipt_model=hipt_model,
            device=device,
            include_rgb_stats=args.include_rgb_stats,
            feature_names=feature_names,
        )
        rows.append(row)
        if (idx + 1) % 10 == 0 or row["status"] != "ok":
            print(f"processed_slides={idx + 1} latest={slide_id} status={row['status']}", flush=True)

    report = pd.DataFrame(rows)
    out_csv = resolve_project_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)
    print(f"slides={len(report)}")
    print(report["status"].value_counts().to_string())
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

