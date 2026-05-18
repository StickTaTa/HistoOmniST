from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from histoomnist.data.spot_table import load_array  # noqa: E402
from histoomnist.utils.config import load_config  # noqa: E402
from histoomnist.utils.io import ensure_dir, read_manifest  # noqa: E402
from histoomnist.utils.project_paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HEST spot features with spatial neighborhood context.")
    parser.add_argument("--config", type=Path, default=Path("configs/hest1k_human_visium_sf.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/HEST-1k/manifests/human_visium_sf_manifest_context.csv"),
    )
    parser.add_argument("--feature-name", default="features_context.npy")
    parser.add_argument("--k", type=int, action="append", default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/hest1k_human_visium_sf/context_feature_generation.csv"),
    )
    return parser.parse_args()


def _optional_path(row, name: str):
    if not hasattr(row, name):
        return None
    value = getattr(row, name)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if str(value).strip() == "":
        return None
    return value


def _as_relative(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, start=base_dir)).as_posix()


def _chunked_neighbor_mean(features: np.ndarray, neighbor_idx: np.ndarray, chunk_size: int) -> np.ndarray:
    out = np.empty((features.shape[0], features.shape[1]), dtype=np.float32)
    for start in range(0, features.shape[0], chunk_size):
        stop = min(start + chunk_size, features.shape[0])
        out[start:stop] = features[neighbor_idx[start:stop]].mean(axis=1, dtype=np.float32)
    return out


def _query_neighbors(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_spots = coords.shape[0]
    if n_spots < 2:
        idx = np.zeros((n_spots, 1), dtype=np.int64)
        dist = np.zeros((n_spots, 1), dtype=np.float32)
        return idx, dist
    query_k = min(k + 1, n_spots)
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=query_k, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]
    idx = idx[:, 1:]
    dist = dist[:, 1:]
    return idx.astype(np.int64, copy=False), dist.astype(np.float32, copy=False)


def _coord_context(coords: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, list[str]]:
    coords64 = coords.astype(np.float64, copy=False)
    min_xy = coords64.min(axis=0)
    max_xy = coords64.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    xy01 = (coords64 - min_xy) / span
    xy_centered = xy01 - 0.5
    radius = np.sqrt((xy_centered**2).sum(axis=1, keepdims=True))
    nonzero = dist[dist > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    scale = max(scale, 1.0)
    dist_mean = np.mean(dist, axis=1, keepdims=True) / scale
    dist_std = np.std(dist, axis=1, keepdims=True) / scale
    dist_max = np.max(dist, axis=1, keepdims=True) / scale
    local_density = 1.0 / (dist_mean + 1.0e-6)
    values = np.concatenate(
        [
            xy01,
            xy_centered,
            radius,
            np.log1p(dist_mean),
            np.log1p(dist_std),
            np.log1p(dist_max),
            np.log1p(local_density),
        ],
        axis=1,
    ).astype(np.float32)
    names = [
        "x_norm",
        "y_norm",
        "x_centered",
        "y_centered",
        "radius_norm",
        "log1p_neighbor_dist_mean",
        "log1p_neighbor_dist_std",
        "log1p_neighbor_dist_max",
        "log1p_local_density",
    ]
    return values, names


def build_context_features(
    *,
    features: np.ndarray,
    coords: np.ndarray,
    ks: list[int],
    chunk_size: int,
) -> tuple[np.ndarray, list[str]]:
    parts = [features.astype(np.float32, copy=False)]
    names = [f"hipt_{idx:04d}" for idx in range(features.shape[1])]
    coord_part: np.ndarray | None = None
    coord_names: list[str] | None = None

    for k in ks:
        neighbor_idx, dist = _query_neighbors(coords, k=k)
        neighbor_mean = _chunked_neighbor_mean(features, neighbor_idx, chunk_size=chunk_size)
        parts.append(neighbor_mean)
        names.extend(f"neighbor{k}_mean_hipt_{idx:04d}" for idx in range(features.shape[1]))
        if coord_part is None:
            coord_part, coord_names = _coord_context(coords, dist)

    if coord_part is not None and coord_names is not None:
        parts.append(coord_part)
        names.extend(coord_names)

    return np.concatenate(parts, axis=1).astype(np.float32, copy=False), names


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_project_path(args.config))
    manifest_path = resolve_project_path(args.manifest or cfg["data"]["manifest"])
    out_manifest_path = resolve_project_path(args.output_manifest)
    manifest = read_manifest(manifest_path)
    if args.max_slides is not None:
        manifest = manifest.head(int(args.max_slides)).copy()
    base_dir = manifest_path.parent
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = resolve_project_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ks = sorted(set(int(k) for k in (args.k or [8, 24])))
    if any(k < 1 for k in ks):
        raise ValueError(f"All k values must be >= 1, got {ks}")

    rows: list[dict[str, object]] = []
    updated = manifest.copy()
    feature_names: list[str] | None = None
    start_all = time.time()

    for idx, row in enumerate(manifest.itertuples(index=False), start=1):
        sample_id = str(row.sample_id)
        start = time.time()
        feature_path = base_dir / str(row.features_path)
        coords_value = _optional_path(row, "coords_path")
        if coords_value is None:
            raise ValueError(f"Missing coords_path for {sample_id}")
        coords_path = base_dir / str(coords_value)
        out_path = feature_path.parent / args.feature_name
        metadata_path = out_path.with_name("features_context_metadata.json")

        if out_path.exists() and not args.force:
            context = np.load(out_path, mmap_mode="r")
            status = "exists"
            n_spots = int(context.shape[0])
            n_features = int(context.shape[1])
        else:
            features = np.asarray(load_array(feature_path), dtype=np.float32)
            coords = np.asarray(load_array(coords_path), dtype=np.float32)
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(f"coords must be n x 2 for {sample_id}, got {coords.shape}")
            if features.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"spot count mismatch for {sample_id}: features={features.shape[0]}, coords={coords.shape[0]}"
                )
            context, names = build_context_features(
                features=features,
                coords=coords[:, :2],
                ks=ks,
                chunk_size=int(args.chunk_size),
            )
            np.save(out_path, context)
            if feature_names is None:
                feature_names = names
            (out_path.parent / "feature_context_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
            metadata_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "source_features": str(feature_path),
                        "source_coords": str(coords_path),
                        "feature_name": args.feature_name,
                        "k": ks,
                        "n_spots": int(context.shape[0]),
                        "n_features": int(context.shape[1]),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            status = "written"
            n_spots = int(context.shape[0])
            n_features = int(context.shape[1])

        updated.loc[updated["sample_id"].astype(str) == sample_id, "features_path"] = _as_relative(
            out_path,
            out_manifest_path.parent,
        )
        elapsed = time.time() - start
        rows.append(
            {
                "sample_id": sample_id,
                "status": status,
                "features_path": str(out_path),
                "n_spots": n_spots,
                "n_features": n_features,
                "elapsed_seconds": elapsed,
            }
        )
        print(f"[{idx}/{len(manifest)}] {sample_id} {status} n={n_spots} d={n_features} sec={elapsed:.2f}", flush=True)

    updated.to_csv(out_manifest_path, index=False)
    pd.DataFrame(rows).to_csv(report_path, index=False)
    summary_path = report_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "output_manifest": str(out_manifest_path),
                "report": str(report_path),
                "k": ks,
                "n_slides": int(len(updated)),
                "elapsed_seconds": time.time() - start_all,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_manifest_path}")
    print(f"wrote {report_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
