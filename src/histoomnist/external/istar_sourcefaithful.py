from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image

from histoomnist.data.gene_selection import (
    gene_key_settings_from_config,
    selected_genes_from_config,
)
from histoomnist.external.histogene_patch_h5 import (
    TargetKind,
    load_histogene_patch_slide,
    target_matrix_from_counts,
)
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path


ISTAR_UPSTREAM_REL = Path("third_party") / "benchmarks" / "iStar"
DEFAULT_ISTAR_PREFIX_ROOT = (
    Path("results")
    / "hest1k_human_visium_expression"
    / "external_baselines"
    / "istar_sourcefaithful_inputs"
)


@dataclass(frozen=True)
class IStarSourceInfo:
    upstream_root: Path
    impute_path: Path
    impute_by_basic_path: Path
    image_path: Path
    imported_components: tuple[str, ...]


def istar_source_info(upstream_root: str | Path | None = None) -> IStarSourceInfo:
    root = resolve_project_path(upstream_root or ISTAR_UPSTREAM_REL)
    if root is None:
        raise ValueError("iStar upstream root resolved to None")
    paths = {
        "impute_path": root / "impute.py",
        "impute_by_basic_path": root / "impute_by_basic.py",
        "image_path": root / "image.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing official iStar source files: " + ", ".join(missing))
    return IStarSourceInfo(
        upstream_root=root,
        imported_components=(
            "source-mirrored impute.ForwardSumModel",
            "source-mirrored impute.get_patches_flat",
            "source-mirrored impute_by_basic.get_embeddings",
            "source-mirrored impute_by_basic.get_locs",
            "source-mirrored image.get_disk_mask",
        ),
        **paths,
    )


class FeedForward(nn.Module):
    def __init__(self, n_inp: int, n_out: int, activation: nn.Module | None = None, residual: bool = False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        self.activation = activation if activation is not None else nn.LeakyReLU(0.1, inplace=True)
        self.residual = residual

    def forward(self, x: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x) + self.beta


class ForwardSumModel(nn.Module):
    """Source-mirrored iStar ``impute.py::ForwardSumModel`` without Lightning."""

    def __init__(self, lr: float, n_inp: int, n_out: int):
        super().__init__()
        self.lr = float(lr)
        self.net_lat = nn.Sequential(
            FeedForward(n_inp, 256),
            FeedForward(256, 256),
            FeedForward(256, 256),
            FeedForward(256, 256),
        )
        self.net_out = FeedForward(256, n_out, activation=ELU(alpha=0.01, beta=0.01))

    def inp_to_lat(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_lat.forward(x)

    def lat_to_out(self, x: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        return self.net_out.forward(x, indices)

    def forward(self, x: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        x = self.inp_to_lat(x)
        return self.lat_to_out(x, indices)


def get_disk_mask(radius: float, boundary_width: float | None = None) -> np.ndarray:
    radius_ceil = np.ceil(radius).astype(int)
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius - boundary_width) ** 2
    return isin


def get_patches_flat(img: np.ndarray, locs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shape = np.array(mask.shape)
    center = shape // 2
    radius = np.stack([-center, shape - center], -1)
    x_list = []
    for spot in locs:
        patch = img[
            spot[0] + radius[0][0] : spot[0] + radius[0][1],
            spot[1] + radius[1][0] : spot[1] + radius[1][1],
        ]
        x = patch if mask.all() else patch[mask]
        x_list.append(x)
    return np.stack(x_list)


def _load_pickle(path: Path) -> Any:
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)


def get_embeddings(prefix: Path) -> np.ndarray:
    embs = _load_pickle(prefix / "embeddings-hist.pickle")
    values = np.concatenate([embs["cls"], embs["sub"], embs["rgb"]])
    return values.transpose(1, 2, 0)


def get_locs(prefix: Path, target_shape: tuple[int, int]) -> np.ndarray:
    locs = pd.read_csv(prefix / "locs.tsv", sep="\t", index_col=0)
    values = np.stack([locs["y"].to_numpy(), locs["x"].to_numpy()], -1).astype(np.float32)
    image_path = prefix / "he.jpg"
    if image_path.exists():
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as image:
            current_shape = np.array([image.height, image.width])
    else:
        # Oversized WSI compatibility path: feature extraction used the official
        # in-memory preprocessed image without writing JPEG, so the embedding
        # grid itself is the authoritative source of the 16-pixel ViT stride.
        current_shape = np.asarray(target_shape[:2]) * 16
    rescale_factor = current_shape // np.asarray(target_shape[:2])
    rescale_factor[rescale_factor < 1] = 1
    values /= rescale_factor
    return values.round().astype(int)


def load_official_istar_components(upstream_root: str | Path | None = None) -> dict[str, Any]:
    istar_source_info(upstream_root)
    return {
        "ForwardSumModel": ForwardSumModel,
        "get_patches_flat": get_patches_flat,
        "get_embeddings": get_embeddings,
        "get_locs": get_locs,
        "get_disk_mask": get_disk_mask,
    }


def istar_source_metadata(upstream_root: str | Path | None = None) -> dict[str, Any]:
    info = istar_source_info(upstream_root)
    metadata: dict[str, Any] = {
        "upstream_root": str(info.upstream_root),
        "impute_path": str(info.impute_path),
        "impute_by_basic_path": str(info.impute_by_basic_path),
        "image_path": str(info.image_path),
        "imported_components": list(info.imported_components),
        "paper_faithful_cross_slide_note": (
            "This adapter keeps the official iStar ForwardSumModel and disk-patch "
            "histology-feature aggregation. Small official functions are mirrored from "
            "the source files to avoid importing optional single-script dependencies "
            "such as PyTorch Lightning, OpenCV, and scikit-image on HPC."
        ),
    }
    provenance = info.upstream_root / "source_provenance.json"
    if provenance.exists():
        metadata["provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    return metadata


def _read_prefix_spots(prefix: Path) -> list[str]:
    locs_path = prefix / "locs.tsv"
    if not locs_path.exists():
        locs_path = prefix / "locs-raw.tsv"
    frame = pd.read_csv(locs_path, sep="\t", index_col=0)
    return [str(x) for x in frame.index.tolist()]


class OfficialIStarCore(nn.Module):
    def __init__(
        self,
        *,
        n_inp: int,
        n_out: int,
        lr: float = 1.0e-4,
        upstream_root: str | Path | None = None,
    ):
        super().__init__()
        components = load_official_istar_components(upstream_root)
        self.model = components["ForwardSumModel"](lr=float(lr), n_inp=int(n_inp), n_out=int(n_out))
        self.lr = float(lr)
        self.n_inp = int(n_inp)
        self.n_out = int(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class IStarHESTPrefixSlideDataset(Dataset):
    """HEST slide dataset using official iStar prefix embeddings.

    Each item is one slide. ``features`` has shape
    ``[n_spots, disk_pixels, n_istar_features]`` and is passed directly to the
    official ``ForwardSumModel``. The model predicts per-disk-pixel expression;
    the training loss follows upstream iStar by averaging over the disk axis
    before comparing to spot-level expression.
    """

    def __init__(
        self,
        expression_config: dict[str, Any],
        *,
        splits: list[str],
        prefix_root: str | Path = DEFAULT_ISTAR_PREFIX_ROOT,
        slide_ids: list[str] | None = None,
        max_slides: int | None = None,
        smallest_slides: bool = False,
        target_kind: TargetKind = "log1p_rate",
        max_spots_per_slide: int | None = None,
        max_slide_spots: int | None = None,
        upstream_root: str | Path | None = None,
        verify_spot_order: bool = True,
    ):
        manifest_path = resolve_project_path(expression_config["data"]["manifest"])
        if manifest_path is None:
            raise ValueError("Expression config data.manifest resolved to None")
        manifest = read_manifest(manifest_path)
        rows = manifest[manifest["split"].isin([str(x) for x in splits])].copy()
        if slide_ids:
            wanted = {str(x) for x in slide_ids}
            rows = rows[rows["sample_id"].astype(str).isin(wanted)].copy()
        if max_slide_spots is not None:
            if "n_spots" not in rows.columns:
                raise ValueError("Manifest lacks n_spots; cannot apply max_slide_spots.")
            rows = rows[rows["n_spots"].astype(int) <= int(max_slide_spots)].copy()
        if smallest_slides:
            rows = rows.sort_values(["n_spots", "sample_id"]).copy()
        if max_slides is not None:
            rows = rows.head(int(max_slides)).copy()
        if rows.empty:
            raise ValueError(f"No manifest rows for splits={splits}, slide_ids={slide_ids}.")

        base_dir = manifest_path.parent
        target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
        if target_genes is None or gene_indices is not None:
            raise ValueError("iStar source-faithful adapter requires data.gene_names_path target genes.")
        gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
        raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
        raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
        if raw_root is None:
            raise ValueError("Expression config paths.raw_root resolved to None")
        resolved_prefix_root = resolve_project_path(prefix_root)
        if resolved_prefix_root is None:
            raise ValueError("iStar prefix root resolved to None")
        min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))

        self.target_kind = target_kind
        self.target_genes = list(target_genes)
        self.prefix_root = resolved_prefix_root
        self.max_spots_per_slide = None if max_spots_per_slide is None else int(max_spots_per_slide)
        self.max_slide_spots = None if max_slide_spots is None else int(max_slide_spots)
        self.verify_spot_order = bool(verify_spot_order)
        self.components = load_official_istar_components(upstream_root)
        self.upstream_root = upstream_root
        self.slides = []
        for row in rows.itertuples(index=False):
            sample_id = str(row.sample_id)
            prefix = resolved_prefix_root / sample_id
            required = [prefix / "embeddings-hist.pickle", prefix / "radius.txt", prefix / "locs.tsv"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing iStar prepared files for {sample_id}: " + ", ".join(missing))
            slide = load_histogene_patch_slide(
                row=row,
                base_dir=base_dir,
                raw_root=raw_root,
                target_genes=self.target_genes,
                gene_key=gene_key,
                raw_st_root=raw_st_root,
                min_total_counts=min_total_counts,
            )
            if self.verify_spot_order:
                prefix_spots = _read_prefix_spots(prefix)
                if prefix_spots[: len(slide.spot_ids)] != slide.spot_ids:
                    raise ValueError(f"iStar prefix spot order does not match processed HEST spots for {sample_id}")
            self.slides.append((slide, prefix))

    def __len__(self) -> int:
        return len(self.slides)

    def _used_spots(self, n_spots: int) -> int:
        if self.max_spots_per_slide is None:
            return int(n_spots)
        return min(int(n_spots), int(self.max_spots_per_slide))

    def slide_summary_frame(self) -> pd.DataFrame:
        rows = []
        for slide, prefix in self.slides:
            rows.append(
                {
                    "sample_id": slide.sample_id,
                    "split": slide.split,
                    "organ": slide.organ,
                    "cohort": slide.cohort,
                    "n_spots": slide.n_spots,
                    "n_used_spots": self._used_spots(slide.n_spots),
                    "truncated_for_smoke": bool(self._used_spots(slide.n_spots) != slide.n_spots),
                    "n_measured_target_genes": int(slide.measured_genes.sum()),
                    "prefix": str(prefix),
                    "embeddings_path": str(prefix / "embeddings-hist.pickle"),
                }
            )
        return pd.DataFrame(rows)

    def _features_for_prefix(self, prefix: Path, n_spots: int) -> tuple[np.ndarray, np.ndarray]:
        embs = self.components["get_embeddings"](prefix)
        locs = self.components["get_locs"](prefix, target_shape=embs.shape[:2])
        radius_raw = float((prefix / "radius.txt").read_text(encoding="utf-8").strip())
        radius = radius_raw / 16.0
        mask = self.components["get_disk_mask"](radius)
        center = np.asarray(mask.shape, dtype=np.int64) // 2
        pad_after = np.asarray(mask.shape, dtype=np.int64) - center - 1
        embs_padded = np.pad(
            embs,
            ((int(center[0]), int(pad_after[0])), (int(center[1]), int(pad_after[1])), (0, 0)),
            mode="edge",
        )
        locs_padded = locs[:n_spots] + center.reshape(1, 2)
        features = self.components["get_patches_flat"](embs_padded, locs_padded, mask).astype(
            np.float32,
            copy=False,
        )
        finite = np.isfinite(features).all(axis=(-1, -2))
        return features, finite

    def load_target_only(self, index: int) -> dict[str, object]:
        slide, _prefix = self.slides[index]
        n_spots = self._used_spots(slide.n_spots)
        counts = slide.counts[:n_spots].toarray().astype(np.float32, copy=False)
        target = target_matrix_from_counts(counts, slide.size_factor[:n_spots], self.target_kind)
        return {
            self.target_kind: target,
            "expression_mask": slide.measured_genes.astype(bool, copy=False),
            "sample_id": slide.sample_id,
            "n_spots": int(slide.n_spots),
            "n_used_spots": int(n_spots),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        slide, prefix = self.slides[index]
        n_spots = self._used_spots(slide.n_spots)
        features, finite = self._features_for_prefix(prefix, n_spots)
        target_info = self.load_target_only(index)
        target = np.asarray(target_info[self.target_kind], dtype=np.float32)
        if finite.shape[0] != target.shape[0]:
            raise ValueError(f"Feature/target spot mismatch for {slide.sample_id}: {finite.shape[0]} vs {target.shape[0]}")
        if not bool(finite.all()):
            features = features[finite]
            target = target[finite]
        return {
            "features": torch.from_numpy(features),
            self.target_kind: torch.from_numpy(target),
            "expression_mask": torch.from_numpy(slide.measured_genes.astype(bool, copy=False)),
            "sample_id": slide.sample_id,
            "n_spots": int(slide.n_spots),
            "n_used_spots": int(n_spots),
            "n_finite_spots": int(finite.sum()),
            "complete_feature_spots": bool(finite.all()),
            "truncated_for_smoke": bool(n_spots != slide.n_spots),
        }


def istar_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    expression_mask: torch.Tensor,
) -> torch.Tensor:
    if pred.ndim == 3:
        pred = pred.mean(dim=-2)
    if target.ndim == 3:
        target = target[0]
    if expression_mask.ndim == 2:
        expression_mask = expression_mask[0]
    valid = expression_mask.bool().unsqueeze(0).expand_as(target)
    valid = valid & torch.isfinite(target) & torch.isfinite(pred)
    if not torch.any(valid):
        raise ValueError("No valid expression values for iStar masked MSE.")
    return (pred - target).pow(2)[valid].mean()
