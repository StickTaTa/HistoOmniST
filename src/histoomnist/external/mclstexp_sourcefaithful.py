from __future__ import annotations

import bisect
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import sparse
from torch.utils.data import Dataset

from histoomnist.data.gene_selection import (
    gene_key_settings_from_config,
    load_gene_keys_for_slide,
    selected_genes_from_config,
)
from histoomnist.data.spot_table import load_spot_table
from histoomnist.external.histogene_patch_h5 import (
    _read_patch_barcodes,
    _read_spot_ids,
    target_matrix_from_counts,
    target_values_from_counts,
)
from histoomnist.utils.io import read_manifest
from histoomnist.utils.project_paths import resolve_project_path


TargetKind = Literal["log1p_rate"]

DEFAULT_MCLSTEXP_UPSTREAM_ROOT = Path("third_party/benchmarks/mclSTExp")


@dataclass(frozen=True)
class MCLSTExpSourceInfo:
    upstream_root: Path
    model_path: Path
    imported_component: str
    commit: str


@dataclass(frozen=True)
class MCLSTExpPatchSlide:
    sample_id: str
    split: str
    organ: str
    cohort: str
    disease_state: str
    patch_h5_path: Path
    spot_ids: list[str]
    patch_indices: np.ndarray
    spatial_coords: np.ndarray
    position_indices: np.ndarray
    counts: sparse.csr_matrix
    size_factor: np.ndarray
    measured_genes: np.ndarray

    @property
    def n_spots(self) -> int:
        return int(self.patch_indices.shape[0])


def _resolve_upstream_root(upstream_root: str | Path | None = None) -> Path:
    root = Path(upstream_root) if upstream_root is not None else DEFAULT_MCLSTEXP_UPSTREAM_ROOT
    if not root.is_absolute():
        resolved = resolve_project_path(root)
        root = resolved if resolved is not None else root
    return Path(root).resolve()


def mclstexp_source_info(upstream_root: str | Path | None = None) -> MCLSTExpSourceInfo:
    root = _resolve_upstream_root(upstream_root)
    provenance = root / "source_provenance.json"
    commit = ""
    if provenance.exists():
        import json

        commit = str(json.loads(provenance.read_text(encoding="utf-8")).get("commit", ""))
    return MCLSTExpSourceInfo(
        upstream_root=root,
        model_path=root / "model.py",
        imported_component="model.py::mclSTExp_Attention",
        commit=commit,
    )


def import_official_mclstexp_attention(upstream_root: str | Path | None = None):
    info = mclstexp_source_info(upstream_root)
    if not info.model_path.exists():
        raise FileNotFoundError(info.model_path)
    module_name = "_histoomnist_official_mclstexp_model"
    if module_name in sys.modules:
        return sys.modules[module_name].mclSTExp_Attention
    spec = importlib.util.spec_from_file_location(module_name, info.model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load mclSTExp model from {info.model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.mclSTExp_Attention


def build_official_mclstexp_model(
    *,
    spot_dim: int,
    encoder_name: str = "densenet121",
    temperature: float = 1.0,
    image_dim: int = 1024,
    projection_dim: int = 256,
    heads_num: int = 8,
    heads_dim: int = 64,
    head_layers: int = 2,
    dropout: float = 0.0,
    upstream_root: str | Path | None = None,
) -> torch.nn.Module:
    cls = import_official_mclstexp_attention(upstream_root)
    return cls(
        encoder_name=encoder_name,
        temperature=float(temperature),
        image_dim=int(image_dim),
        spot_dim=int(spot_dim),
        projection_dim=int(projection_dim),
        heads_num=int(heads_num),
        heads_dim=int(heads_dim),
        head_layers=int(head_layers),
        dropout=float(dropout),
    )


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


def _select_counts_for_target_genes(
    *,
    counts,
    slide_genes: list[str | None],
    target_genes: list[str],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    counts_csr = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
    target_index = {gene: idx for idx, gene in enumerate(target_genes)}
    source_indices: list[int] = []
    target_indices: list[int] = []
    for source_idx, gene in enumerate(slide_genes):
        if gene is None:
            continue
        target_idx = target_index.get(gene)
        if target_idx is None:
            continue
        source_indices.append(source_idx)
        target_indices.append(target_idx)
    if not source_indices:
        raise ValueError("No target genes were found in slide genes.")
    source_array = np.asarray(source_indices, dtype=np.int64)
    target_array = np.asarray(target_indices, dtype=np.int64)
    selected_source = counts_csr[:, source_array].astype(np.float32).tocsr()
    mapper = sparse.csr_matrix(
        (
            np.ones(target_array.shape[0], dtype=np.float32),
            (np.arange(target_array.shape[0]), target_array),
        ),
        shape=(target_array.shape[0], len(target_genes)),
    )
    selected_counts = (selected_source @ mapper).tocsr()
    measured = np.zeros(len(target_genes), dtype=bool)
    measured[np.unique(target_array)] = True
    return selected_counts, measured


def _position_indices(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Coordinates must be shaped (n, >=2), got {values.shape}")
    rounded = np.rint(values[:, :2])
    return np.clip(rounded, 0, 65535).astype(np.int64)


def load_mclstexp_patch_slide(
    *,
    row,
    base_dir: Path,
    raw_root: Path,
    target_genes: list[str],
    gene_key: str,
    raw_st_root: Path | None,
    min_total_counts: float,
) -> MCLSTExpPatchSlide:
    sample_id = str(row.sample_id)
    patch_h5_path = raw_root / "patches" / f"{sample_id}.h5"
    if not patch_h5_path.exists():
        raise FileNotFoundError(f"Patch H5 not found for {sample_id}: {patch_h5_path}")
    table = load_spot_table(
        sample_id=sample_id,
        features_path=base_dir / str(row.features_path),
        counts_path=base_dir / str(row.counts_path),
        coords_path=base_dir / str(_optional_path(row, "coords_path"))
        if _optional_path(row, "coords_path") is not None
        else None,
        size_factor_path=base_dir / str(_optional_path(row, "size_factor_path"))
        if _optional_path(row, "size_factor_path") is not None
        else None,
        min_total_counts=min_total_counts,
    )
    spot_ids_all = _read_spot_ids(base_dir, row, table.features.shape[0])
    patch_index = {barcode: idx for idx, barcode in enumerate(_read_patch_barcodes(patch_h5_path))}
    missing = [barcode for barcode in spot_ids_all if barcode not in patch_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{sample_id} has {len(missing)} processed spots missing from patch H5 barcode: {preview}")
    valid = table.valid_mask.astype(bool)
    patch_indices_all = np.asarray([patch_index[barcode] for barcode in spot_ids_all], dtype=np.int64)
    slide_genes = load_gene_keys_for_slide(
        sample_id=sample_id,
        processed_gene_path=base_dir / str(row.genes_path),
        gene_key=gene_key,
        raw_st_root=raw_st_root,
    )
    selected_counts, measured = _select_counts_for_target_genes(
        counts=table.counts[valid],
        slide_genes=slide_genes,
        target_genes=target_genes,
    )
    if table.coords is None:
        with h5py.File(patch_h5_path, "r") as handle:
            coords_all = np.asarray(handle["coords"], dtype=np.float32)
        coords = coords_all[patch_indices_all[valid]]
    else:
        coords = np.asarray(table.coords[valid], dtype=np.float32)
    return MCLSTExpPatchSlide(
        sample_id=sample_id,
        split=str(row.split),
        organ=str(getattr(row, "organ", "")),
        cohort=str(getattr(row, "cohort", "")),
        disease_state=str(getattr(row, "disease_state", "")),
        patch_h5_path=patch_h5_path,
        spot_ids=[str(x) for x, keep in zip(spot_ids_all, valid) if keep],
        patch_indices=patch_indices_all[valid],
        spatial_coords=coords,
        position_indices=_position_indices(coords),
        counts=selected_counts,
        size_factor=table.size_factor[valid].astype(np.float32, copy=False),
        measured_genes=measured,
    )


class MCLSTExpHESTSpotDataset(Dataset):
    """HEST spot-level RGB patch dataset matching mclSTExp's data surface."""

    def __init__(
        self,
        expression_config: dict,
        *,
        splits: list[str],
        slide_ids: list[str] | None = None,
        max_slides: int | None = None,
        smallest_slides: bool = False,
        target_kind: TargetKind = "log1p_rate",
        target_gene_limit: int | None = None,
        max_spots_per_slide: int | None = None,
        max_slide_spots: int | None = None,
        train: bool = True,
    ):
        if target_kind != "log1p_rate":
            raise ValueError("mclSTExp HEST adapter currently exports log1p_rate only.")
        manifest_path = resolve_project_path(expression_config["data"]["manifest"])
        if manifest_path is None:
            raise ValueError("Expression config data.manifest resolved to None")
        manifest = read_manifest(manifest_path)
        rows = manifest[manifest["split"].isin(splits)].copy()
        if slide_ids:
            wanted = {str(x) for x in slide_ids}
            rows = rows[rows["sample_id"].astype(str).isin(wanted)].copy()
        if max_slide_spots is not None:
            rows = rows[rows["n_spots"].astype(int) <= int(max_slide_spots)].copy()
        if smallest_slides:
            rows = rows.sort_values(["n_spots", "sample_id"], ascending=[True, True]).copy()
        if max_slides is not None:
            rows = rows.head(int(max_slides)).copy()
        if rows.empty:
            raise ValueError(f"No manifest rows for splits={splits}")
        base_dir = manifest_path.parent
        target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
        if target_genes is None or gene_indices is not None:
            raise ValueError("mclSTExp source-faithful adapter requires data.gene_names_path target genes.")
        if target_gene_limit is not None:
            target_genes = target_genes[: int(target_gene_limit)]
        gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
        raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
        raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
        if raw_root is None:
            raise ValueError("Expression config paths.raw_root resolved to None")
        min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
        self.target_kind = target_kind
        self.target_genes = list(target_genes)
        self.train = bool(train)
        self.target_gene_limit = None if target_gene_limit is None else int(target_gene_limit)
        self.max_spots_per_slide = None if max_spots_per_slide is None else int(max_spots_per_slide)
        self.slides = [
            load_mclstexp_patch_slide(
                row=row,
                base_dir=base_dir,
                raw_root=raw_root,
                target_genes=self.target_genes,
                gene_key=gene_key,
                raw_st_root=raw_st_root,
                min_total_counts=min_total_counts,
            )
            for row in rows.itertuples(index=False)
        ]
        self.index: list[tuple[int, int]] = []
        for slide_idx, slide in enumerate(self.slides):
            n = slide.n_spots if self.max_spots_per_slide is None else min(slide.n_spots, self.max_spots_per_slide)
            self.index.extend((slide_idx, local_idx) for local_idx in range(n))
        if not self.index:
            raise ValueError("No mclSTExp spot items were created.")

        try:
            import torchvision.transforms as transforms
        except ImportError as exc:
            raise ImportError("mclSTExp source-faithful adapter requires torchvision transforms.") from exc
        self.train_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
            ]
        )
        self.eval_transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.index)

    def slide_summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sample_id": slide.sample_id,
                    "split": slide.split,
                    "organ": slide.organ,
                    "cohort": slide.cohort,
                    "n_spots": slide.n_spots,
                    "n_used_spots": min(slide.n_spots, self.max_spots_per_slide)
                    if self.max_spots_per_slide is not None
                    else slide.n_spots,
                    "n_measured_target_genes": int(slide.measured_genes.sum()),
                    "patch_h5_path": str(slide.patch_h5_path),
                    "position_min": int(np.min(slide.position_indices)),
                    "position_max": int(np.max(slide.position_indices)),
                }
                for slide in self.slides
            ]
        )

    def slide_local_indices(self, sample_id: str) -> list[int]:
        sample_id = str(sample_id)
        for slide in self.slides:
            if slide.sample_id == sample_id:
                n = slide.n_spots if self.max_spots_per_slide is None else min(slide.n_spots, self.max_spots_per_slide)
                return list(range(n))
        raise KeyError(sample_id)

    def _read_patch(self, slide: MCLSTExpPatchSlide, local_idx: int) -> torch.Tensor:
        patch_index = int(slide.patch_indices[local_idx])
        with h5py.File(slide.patch_h5_path, "r") as handle:
            patch = np.asarray(handle["img"][patch_index], dtype=np.uint8)
        if patch.ndim != 3 or patch.shape[-1] != 3:
            raise ValueError(f"Patch image must be HWC RGB, got {patch.shape} for {slide.sample_id}")
        image = Image.fromarray(patch, mode="RGB")
        transform = self.train_transform if self.train else self.eval_transform
        return transform(image)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index = len(self.index) + index
        if index < 0 or index >= len(self.index):
            raise IndexError(index)
        slide_idx, local_idx = self.index[index]
        slide = self.slides[slide_idx]
        counts = slide.counts.getrow(local_idx).toarray().reshape(-1).astype(np.float32, copy=False)
        target = target_values_from_counts(counts, float(slide.size_factor[local_idx]), self.target_kind)
        return {
            "image": self._read_patch(slide, local_idx),
            "position": torch.from_numpy(slide.position_indices[local_idx]).long(),
            "expression": torch.from_numpy(target),
            "expression_mask": torch.from_numpy(slide.measured_genes),
            "sample_id": slide.sample_id,
            "spot_id": slide.spot_ids[local_idx],
            "local_index": int(local_idx),
            "spatial_coords": torch.from_numpy(slide.spatial_coords[local_idx].astype(np.float32, copy=False)),
        }

    def target_matrix_for_slide(self, sample_id: str) -> np.ndarray:
        for slide in self.slides:
            if slide.sample_id == str(sample_id):
                n = slide.n_spots if self.max_spots_per_slide is None else min(slide.n_spots, self.max_spots_per_slide)
                counts = slide.counts[:n].toarray().astype(np.float32, copy=False)
                return target_matrix_from_counts(counts, slide.size_factor[:n], self.target_kind)
        raise KeyError(sample_id)


def mclstexp_embeddings(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    include_image: bool = True,
    include_spot: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image_embeddings = None
    spot_embeddings = None
    if include_image:
        image_features = model.image_encoder(batch["image"])
        image_embeddings = model.image_projection(image_features)
    if include_spot:
        spot_feature = batch["expression"]
        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = model.x_embed(x)
        centers_y = model.y_embed(y)
        spot_feature = spot_feature + centers_x + centers_y
        spot_features = spot_feature.unsqueeze(dim=0)
        spot_embeddings = model.spot_encoder(spot_features)
        spot_embeddings = model.spot_projection(spot_embeddings).squeeze(dim=0)
    return image_embeddings, spot_embeddings


def mclstexp_contrastive_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    image_embeddings, spot_embeddings = mclstexp_embeddings(model, batch, include_image=True, include_spot=True)
    assert image_embeddings is not None
    assert spot_embeddings is not None
    cos_smi = (spot_embeddings @ image_embeddings.T) / model.temperature
    labels = torch.eye(cos_smi.shape[0], cos_smi.shape[1], dtype=cos_smi.dtype, device=cos_smi.device)
    spots_loss = F.cross_entropy(cos_smi, labels)
    images_loss = F.cross_entropy(cos_smi.T, labels.T)
    return (images_loss + spots_loss) / 2.0
