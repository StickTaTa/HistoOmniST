from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from PIL import Image
from scipy import sparse

from histoomnist.data.gene_selection import (
    gene_key_settings_from_config,
    load_gene_keys_for_slide,
    selected_genes_from_config,
)
from histoomnist.data.spot_table import load_spot_table
from histoomnist.eval.benchmark_predictions import evaluate_prediction_bundle
from histoomnist.hest.raw_assets import read_h5_string_vector
from histoomnist.utils.project_paths import resolve_project_path
from histoomnist.utils.io import read_manifest
from histoomnist.utils.seed import set_seed

try:
    from tensorflow.keras.utils import Sequence as KerasSequence
except Exception:  # pragma: no cover - keeps non-TensorFlow static checks importable.
    class KerasSequence:  # type: ignore[no-redef]
        pass


TargetKind = Literal["log1p_rate"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STIMAGE_UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "benchmarks" / "STimage"


@dataclass(frozen=True)
class STImagePatchSlide:
    sample_id: str
    split: str
    organ: str
    cohort: str
    disease_state: str
    patch_h5_path: Path
    spot_ids: list[str]
    patch_indices: np.ndarray
    spatial_coords: np.ndarray | None
    position_norm: np.ndarray
    counts: sparse.csr_matrix
    size_factor: np.ndarray
    measured_genes: np.ndarray

    @property
    def n_spots(self) -> int:
        return int(self.patch_indices.shape[0])


@dataclass(frozen=True)
class STImageSourceInfo:
    upstream_root: Path
    commit: str | None
    model_reference: str
    training_reference: str
    prediction_reference: str


def _add_stimage_to_path(upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT) -> Path:
    root = Path(upstream_root)
    if not root.exists():
        raise FileNotFoundError(root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def import_stimage_model(upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT):
    _add_stimage_to_path(upstream_root)
    from stimage._model import CNN_NB_multiple_genes, negative_binomial_layer, negative_binomial_loss

    return CNN_NB_multiple_genes, negative_binomial_layer, negative_binomial_loss


def recompile_stimage_model(model, *, optimizer_mode: str = "default"):
    if optimizer_mode == "default":
        return model
    import tensorflow as tf

    _, _, negative_binomial_loss = import_stimage_model(DEFAULT_STIMAGE_UPSTREAM_ROOT)
    if optimizer_mode == "legacy_adam":
        model.compile(loss=negative_binomial_loss, optimizer=tf.keras.optimizers.legacy.Adam(1.0e-5))
        return model
    raise ValueError(f"Unsupported STimage optimizer_mode: {optimizer_mode}")


def stimage_source_metadata(upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT) -> dict[str, Any]:
    root = Path(upstream_root)
    provenance_path = root / "source_provenance.json"
    provenance: dict[str, Any] = {}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {
        "upstream_root": str(root),
        "provenance": provenance,
        "source_info": {
            "model_reference": "third_party/benchmarks/STimage/stimage/_model.py::CNN_NB_multiple_genes",
            "loss_reference": "third_party/benchmarks/STimage/stimage/_model.py::negative_binomial_loss",
            "training_reference": "third_party/benchmarks/STimage/stimage/02_Training.py",
            "prediction_reference": "third_party/benchmarks/STimage/stimage/03_Prediction.py",
            "official_benchmark_reference": "third_party/benchmarks/STimage/Figure_scripts/Figure2/benchmarking_her2st/stimage_her2st.py",
            "hest_adapter_boundary": "HEST split/data loading, coverage95 log1p_rate target, sample-weight masking, export, evaluation, and spatial maps only",
        },
    }


def _resize_hwc_uint8(patch: np.ndarray, tile_size: int) -> np.ndarray:
    arr = np.asarray(patch)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB patch, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] == tile_size and arr.shape[1] == tile_size:
        return arr
    image = Image.fromarray(arr, mode="RGB")
    image = image.resize((int(tile_size), int(tile_size)), resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


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


def _read_spot_ids(base_dir: Path, row, n_spots: int) -> list[str]:
    spots_path = _optional_path(row, "spots_path")
    candidates: list[Path] = []
    if spots_path is not None:
        candidates.append(base_dir / str(spots_path))
    candidates.append((base_dir / str(row.counts_path)).parent / "spots.txt")
    for path in candidates:
        if path.exists():
            values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(values) == n_spots:
                return values
            raise ValueError(f"Spot id count mismatch for {row.sample_id}: {path} has {len(values)}, expected {n_spots}")
    raise FileNotFoundError(f"Could not find processed spot ids for {row.sample_id}")


def _read_patch_barcodes(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        if "barcode" not in handle:
            raise KeyError(f"Patch H5 lacks barcode dataset: {path}")
        return read_h5_string_vector(handle["barcode"])


def _normalise_positions(coords: np.ndarray | None, n_spots: int) -> np.ndarray:
    if coords is None:
        return np.zeros((n_spots, 2), dtype=np.float32)
    values = np.asarray(coords, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Coordinates must be shaped (n, >=2), got {values.shape}")
    values = values[:, :2]
    lo = np.nanmin(values, axis=0)
    hi = np.nanmax(values, axis=0)
    span = hi - lo
    span[span < 1.0e-6] = 1.0
    out = (values - lo) / span
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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


def _load_stimage_patch_slide(
    *,
    row,
    base_dir: Path,
    raw_root: Path,
    target_genes: list[str],
    gene_key: str,
    raw_st_root: Path | None,
    min_total_counts: float,
) -> STImagePatchSlide:
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
    coords = None if table.coords is None else np.asarray(table.coords[valid], dtype=np.float32)
    return STImagePatchSlide(
        sample_id=sample_id,
        split=str(row.split),
        organ=str(getattr(row, "organ", "")),
        cohort=str(getattr(row, "cohort", "")),
        disease_state=str(getattr(row, "disease_state", "")),
        patch_h5_path=patch_h5_path,
        spot_ids=[str(x) for x, keep in zip(spot_ids_all, valid) if keep],
        patch_indices=patch_indices_all[valid],
        spatial_coords=coords,
        position_norm=_normalise_positions(coords, int(np.sum(valid))),
        counts=selected_counts,
        size_factor=table.size_factor[valid].astype(np.float32, copy=False),
        measured_genes=measured,
    )


def target_values_from_counts(counts: np.ndarray, size_factor: float, target_kind: TargetKind) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float32)
    rate = values / max(float(size_factor), 1.0e-6)
    if target_kind == "log1p_rate":
        return np.log1p(rate).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported target_kind: {target_kind}")


class STImageSpotSequence(KerasSequence):
    """Keras Sequence-compatible HEST loader for the official STimage NB model."""

    def __init__(
        self,
        *,
        slides: list[STImagePatchSlide],
        target_genes: list[str],
        batch_size: int,
        tile_size: int = 299,
        target_kind: TargetKind = "log1p_rate",
        shuffle: bool = False,
        seed: int = 2026,
        include_targets: bool = True,
        gene_indices: np.ndarray | None = None,
    ):
        if target_kind != "log1p_rate":
            raise ValueError("STimage formal adapter currently supports log1p_rate only.")
        if not slides:
            raise ValueError("STImageSpotSequence received no slides.")
        self.slides = list(slides)
        self.target_genes = list(target_genes)
        self.batch_size = int(batch_size)
        self.tile_size = int(tile_size)
        self.target_kind = target_kind
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.include_targets = bool(include_targets)
        self.gene_indices = (
            np.arange(len(self.target_genes), dtype=np.int64)
            if gene_indices is None
            else np.asarray(gene_indices, dtype=np.int64)
        )
        if len(self.gene_indices) != len(self.target_genes):
            raise ValueError("gene_indices must have the same length as target_genes.")
        self.lengths = np.asarray([slide.n_spots for slide in self.slides], dtype=np.int64)
        self.cumlen = np.cumsum(self.lengths)
        self.indices = np.arange(int(self.cumlen[-1]), dtype=np.int64)
        self.rng = np.random.default_rng(self.seed)
        self._handles: dict[str, h5py.File] = {}
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / max(self.batch_size, 1)))

    @property
    def n_spots(self) -> int:
        return int(self.cumlen[-1])

    @property
    def n_genes(self) -> int:
        return len(self.target_genes)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _locate(self, global_index: int) -> tuple[int, int]:
        slide_idx = int(np.searchsorted(self.cumlen, int(global_index), side="right"))
        prev = 0 if slide_idx == 0 else int(self.cumlen[slide_idx - 1])
        return slide_idx, int(global_index - prev)

    def _handle(self, slide: STImagePatchSlide) -> h5py.File:
        key = str(slide.patch_h5_path)
        handle = self._handles.get(key)
        if handle is None:
            handle = h5py.File(slide.patch_h5_path, "r")
            self._handles[key] = handle
        return handle

    def _batch_arrays(self, batch_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.zeros((len(batch_indices), self.tile_size, self.tile_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_indices), self.n_genes), dtype=np.float32)
        w = np.zeros((len(batch_indices), self.n_genes), dtype=np.float32)
        for out_idx, global_idx in enumerate(batch_indices):
            slide_idx, local_idx = self._locate(int(global_idx))
            slide = self.slides[slide_idx]
            patch_index = int(slide.patch_indices[local_idx])
            patch = np.asarray(self._handle(slide)["img"][patch_index])
            x[out_idx] = _resize_hwc_uint8(patch, self.tile_size).astype(np.float32)
            if self.include_targets:
                counts = slide.counts.getrow(local_idx).toarray().reshape(-1).astype(np.float32, copy=False)
                target_full = target_values_from_counts(counts, float(slide.size_factor[local_idx]), self.target_kind)
                y[out_idx] = target_full[self.gene_indices]
                w[out_idx] = slide.measured_genes[self.gene_indices].astype(np.float32, copy=False)
        return x, y, w

    def __getitem__(self, batch_index: int):
        start = int(batch_index) * self.batch_size
        stop = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:stop]
        x, y, w = self._batch_arrays(batch_indices)
        if not self.include_targets:
            return x
        y_tuple = tuple(y[:, i : i + 1] for i in range(self.n_genes))
        w_tuple = tuple(w[:, i] for i in range(self.n_genes))
        return x, y_tuple, w_tuple

    def slide_summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sample_id": slide.sample_id,
                    "split": slide.split,
                    "organ": slide.organ,
                    "cohort": slide.cohort,
                    "disease_state": slide.disease_state,
                    "n_spots": int(slide.n_spots),
                    "n_measured_target_genes": int(slide.measured_genes.sum()),
                    "patch_h5_path": str(slide.patch_h5_path),
                }
                for slide in self.slides
            ]
        )


def build_stimage_model(
    *,
    n_genes: int,
    tile_size: int = 299,
    cnn_base: str = "resnet50",
    fine_tuning: bool = False,
    upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT,
    optimizer_mode: str = "default",
    run_eagerly: bool = False,
):
    CNN_NB_multiple_genes, _, _ = import_stimage_model(upstream_root)
    model = CNN_NB_multiple_genes((int(tile_size), int(tile_size), 3), int(n_genes), cnnbase=cnn_base, ft=fine_tuning)
    if optimizer_mode == "default" and not run_eagerly:
        return model
    _, _, negative_binomial_loss = import_stimage_model(upstream_root)
    if optimizer_mode == "default":
        import tensorflow as tf

        model.compile(
            loss=negative_binomial_loss,
            optimizer=tf.keras.optimizers.Adam(1.0e-5),
            run_eagerly=bool(run_eagerly),
        )
        return model
    if optimizer_mode == "adam_nojit":
        import tensorflow as tf

        model.compile(
            loss=negative_binomial_loss,
            optimizer=tf.keras.optimizers.Adam(1.0e-5, jit_compile=False),
            run_eagerly=bool(run_eagerly),
        )
        return model
    if optimizer_mode == "legacy_adam":
        import tensorflow as tf

        model.compile(
            loss=negative_binomial_loss,
            optimizer=tf.keras.optimizers.legacy.Adam(1.0e-5),
            run_eagerly=bool(run_eagerly),
        )
        return model
    raise ValueError(f"Unsupported STimage optimizer_mode: {optimizer_mode}")


def train_step_probe_summary(
    *,
    expression_config: dict,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    output_dir: str | Path,
    upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT,
    optimizer_mode: str = "default",
    target_kind: TargetKind = "log1p_rate",
    batch_size: int = 32,
    tile_size: int = 299,
    cnn_base: str = "resnet50",
    fine_tuning: bool = False,
    gene_limit: int | None = None,
    seed: int = 2026,
    run_eagerly: bool = False,
) -> dict[str, Any]:
    import tensorflow as tf

    set_seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_slides, target_genes = load_stimage_slides(
        expression_config,
        train_rows,
        target_kind=target_kind,
        gene_limit=gene_limit,
    )
    val_slides, val_genes = load_stimage_slides(
        expression_config,
        val_rows,
        target_kind=target_kind,
        gene_limit=gene_limit,
    )
    if val_genes != target_genes:
        raise ValueError("Train/validation gene lists differ.")
    train_seq = STImageSpotSequence(
        slides=train_slides,
        target_genes=target_genes,
        batch_size=batch_size,
        tile_size=tile_size,
        target_kind=target_kind,
        shuffle=False,
        seed=seed,
        include_targets=True,
    )
    val_seq = STImageSpotSequence(
        slides=val_slides,
        target_genes=target_genes,
        batch_size=batch_size,
        tile_size=tile_size,
        target_kind=target_kind,
        shuffle=False,
        seed=seed,
        include_targets=True,
    )
    started = time.perf_counter()
    model = build_stimage_model(
        n_genes=len(target_genes),
        tile_size=tile_size,
        cnn_base=cnn_base,
        fine_tuning=fine_tuning,
        upstream_root=upstream_root,
        optimizer_mode=optimizer_mode,
        run_eagerly=run_eagerly,
    )
    build_seconds = time.perf_counter() - started
    fit_started = time.perf_counter()
    try:
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=1,
            steps_per_epoch=1,
            validation_steps=1,
            verbose=2,
        )
    finally:
        train_seq.close()
        val_seq.close()
    fit_seconds = time.perf_counter() - fit_started
    summary = {
        "method": "stimage_sourcefaithful",
        "probe_type": "single_train_step",
        "optimizer_mode": str(optimizer_mode),
        "target_kind": target_kind,
        "n_train_slides": int(len(train_slides)),
        "n_val_slides": int(len(val_slides)),
        "n_train_spots": int(sum(slide.n_spots for slide in train_slides)),
        "n_val_spots": int(sum(slide.n_spots for slide in val_slides)),
        "n_genes": int(len(target_genes)),
        "gene_limit": int(gene_limit) if gene_limit is not None else None,
        "batch_size": int(batch_size),
        "tile_size": int(tile_size),
        "cnn_base": str(cnn_base),
        "fine_tuning": bool(fine_tuning),
        "run_eagerly": bool(run_eagerly),
        "build_seconds": float(build_seconds),
        "fit_one_step_seconds": float(fit_seconds),
        "loss": float(history.history.get("loss", [np.nan])[0]),
        "val_loss": float(history.history.get("val_loss", [np.nan])[0]),
        "tensorflow": tf.__version__,
        "tensorflow_built_cuda": bool(tf.test.is_built_with_cuda()),
        "tensorflow_physical_gpus": [gpu.name for gpu in tf.config.list_physical_devices("GPU")],
        "source_info": stimage_source_metadata(upstream_root),
    }
    (output_dir / "train_step_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def custom_train_step_probe_summary(
    *,
    expression_config: dict,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    output_dir: str | Path,
    upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT,
    target_kind: TargetKind = "log1p_rate",
    batch_size: int = 32,
    tile_size: int = 299,
    cnn_base: str = "resnet50",
    fine_tuning: bool = False,
    gene_limit: int | None = None,
    seed: int = 2026,
    probe_steps: int = 1,
) -> dict[str, Any]:
    import tensorflow as tf

    _, _, negative_binomial_loss = import_stimage_model(upstream_root)
    set_seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_slides, target_genes = load_stimage_slides(
        expression_config,
        train_rows,
        target_kind=target_kind,
        gene_limit=gene_limit,
    )
    val_slides, val_genes = load_stimage_slides(
        expression_config,
        val_rows,
        target_kind=target_kind,
        gene_limit=gene_limit,
    )
    if val_genes != target_genes:
        raise ValueError("Train/validation gene lists differ.")
    train_seq = STImageSpotSequence(
        slides=train_slides,
        target_genes=target_genes,
        batch_size=batch_size,
        tile_size=tile_size,
        target_kind=target_kind,
        shuffle=False,
        seed=seed,
        include_targets=True,
    )
    val_seq = STImageSpotSequence(
        slides=val_slides,
        target_genes=target_genes,
        batch_size=batch_size,
        tile_size=tile_size,
        target_kind=target_kind,
        shuffle=False,
        seed=seed,
        include_targets=True,
    )

    def weighted_stimage_loss(y_np: np.ndarray, w_np: np.ndarray, output_list: list[Any]):
        y_tensor = tf.convert_to_tensor(y_np, dtype=tf.float32)
        w_tensor = tf.convert_to_tensor(w_np, dtype=tf.float32)
        pred = tf.stack([tf.convert_to_tensor(out, dtype=tf.float32) for out in output_list], axis=1)
        raw_loss = tf.squeeze(negative_binomial_loss(tf.expand_dims(y_tensor, axis=-1), pred), axis=-1)
        numerator = tf.reduce_sum(raw_loss * w_tensor, axis=0)
        denominator = tf.reduce_sum(w_tensor, axis=0)
        per_gene = tf.where(denominator > 0.0, numerator / tf.maximum(denominator, 1.0e-6), 0.0)
        return tf.reduce_sum(per_gene)

    build_started = time.perf_counter()
    model = build_stimage_model(
        n_genes=len(target_genes),
        tile_size=tile_size,
        cnn_base=cnn_base,
        fine_tuning=fine_tuning,
        upstream_root=upstream_root,
    )
    optimizer = tf.keras.optimizers.Adam(1.0e-5, jit_compile=False)
    build_seconds = time.perf_counter() - build_started

    train_step_seconds_list: list[float] = []
    train_losses: list[float] = []
    non_null_gradients: list[int] = []
    n_steps = min(max(int(probe_steps), 1), len(train_seq))
    try:
        for step_idx in range(n_steps):
            start = step_idx * int(batch_size)
            stop = min(start + int(batch_size), train_seq.n_spots)
            train_batch_indices = train_seq.indices[start:stop]
            x_train, y_train, w_train = train_seq._batch_arrays(train_batch_indices)
            train_started = time.perf_counter()
            with tf.GradientTape() as tape:
                train_outputs = model(tf.convert_to_tensor(x_train, dtype=tf.float32), training=True)
                train_loss = weighted_stimage_loss(y_train, w_train, list(train_outputs))
            grads = tape.gradient(train_loss, model.trainable_variables)
            grads_and_vars = [(grad, var) for grad, var in zip(grads, model.trainable_variables) if grad is not None]
            optimizer.apply_gradients(grads_and_vars)
            train_step_seconds_list.append(float(time.perf_counter() - train_started))
            train_losses.append(float(train_loss.numpy()))
            non_null_gradients.append(int(len(grads_and_vars)))
        val_batch_indices = val_seq.indices[: int(batch_size)]
        x_val, y_val, w_val = val_seq._batch_arrays(val_batch_indices)
    finally:
        train_seq.close()
        val_seq.close()

    val_started = time.perf_counter()
    val_outputs = model(tf.convert_to_tensor(x_val, dtype=tf.float32), training=False)
    val_loss = weighted_stimage_loss(y_val, w_val, list(val_outputs))
    val_step_seconds = time.perf_counter() - val_started

    summary = {
        "method": "stimage_sourcefaithful",
        "probe_type": "custom_single_train_step",
        "target_kind": target_kind,
        "n_train_slides": int(len(train_slides)),
        "n_val_slides": int(len(val_slides)),
        "n_train_spots": int(sum(slide.n_spots for slide in train_slides)),
        "n_val_spots": int(sum(slide.n_spots for slide in val_slides)),
        "n_genes": int(len(target_genes)),
        "gene_limit": int(gene_limit) if gene_limit is not None else None,
        "batch_size": int(batch_size),
        "probe_steps": int(n_steps),
        "tile_size": int(tile_size),
        "cnn_base": str(cnn_base),
        "fine_tuning": bool(fine_tuning),
        "loss_formula": "sum over genes of masked mean official negative_binomial_loss",
        "build_seconds": float(build_seconds),
        "train_step_seconds": train_step_seconds_list,
        "mean_train_step_seconds": float(np.mean(train_step_seconds_list)),
        "median_train_step_seconds": float(np.median(train_step_seconds_list)),
        "val_step_seconds": float(val_step_seconds),
        "loss": float(train_losses[-1]),
        "train_losses": train_losses,
        "val_loss": float(val_loss.numpy()),
        "n_trainable_variables": int(len(model.trainable_variables)),
        "n_non_null_gradients": int(non_null_gradients[-1]),
        "non_null_gradients_per_step": non_null_gradients,
        "tensorflow": tf.__version__,
        "tensorflow_built_cuda": bool(tf.test.is_built_with_cuda()),
        "tensorflow_physical_gpus": [gpu.name for gpu in tf.config.list_physical_devices("GPU")],
        "source_info": stimage_source_metadata(upstream_root),
    }
    (output_dir / "custom_train_step_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary



def model_smoke_summary(
    *,
    output_dir: str | Path,
    upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT,
    n_genes: int = 8,
    tile_size: int = 299,
    cnn_base: str = "resnet50",
    fine_tuning: bool = False,
) -> dict[str, Any]:
    import tensorflow as tf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = build_stimage_model(
        n_genes=int(n_genes),
        tile_size=int(tile_size),
        cnn_base=str(cnn_base),
        fine_tuning=bool(fine_tuning),
        upstream_root=upstream_root,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "method": "stimage_sourcefaithful",
        "smoke_type": "model_build",
        "n_genes": int(n_genes),
        "tile_size": int(tile_size),
        "cnn_base": str(cnn_base),
        "fine_tuning": bool(fine_tuning),
        "model_outputs": int(len(model.outputs)),
        "first_output_shape": model.outputs[0].shape.as_list() if model.outputs else [],
        "trainable_params": int(np.sum([np.prod(v.shape) for v in model.trainable_weights])),
        "non_trainable_params": int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights])),
        "total_params": int(model.count_params()),
        "build_seconds": float(elapsed),
        "tensorflow": tf.__version__,
        "tensorflow_built_cuda": bool(tf.test.is_built_with_cuda()),
        "tensorflow_physical_gpus": [gpu.name for gpu in tf.config.list_physical_devices("GPU")],
        "source_info": stimage_source_metadata(upstream_root),
    }
    (output_dir / "model_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def select_manifest_rows(
    expression_config: dict,
    *,
    splits: list[str],
    slide_ids: list[str] | None = None,
    max_slides: int | None = None,
    smallest_slides: bool = False,
    max_slide_spots: int | None = None,
) -> pd.DataFrame:
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression config data.manifest resolved to None")
    manifest = read_manifest(manifest_path)
    rows = manifest[manifest["split"].astype(str).isin([str(x) for x in splits])].copy()
    if slide_ids is not None:
        wanted = [str(x) for x in slide_ids]
        rows = rows[rows["sample_id"].astype(str).isin(wanted)].copy()
        rows["_requested_order"] = pd.Categorical(rows["sample_id"].astype(str), categories=wanted, ordered=True)
        rows = rows.sort_values("_requested_order").drop(columns=["_requested_order"])
    if max_slide_spots is not None and "n_spots" in rows.columns:
        rows = rows[pd.to_numeric(rows["n_spots"], errors="coerce").le(int(max_slide_spots))].copy()
    if smallest_slides and "n_spots" in rows.columns:
        rows = rows.assign(_n_spots=pd.to_numeric(rows["n_spots"], errors="coerce")).sort_values("_n_spots")
        rows = rows.drop(columns=["_n_spots"])
    if max_slides is not None:
        rows = rows.head(int(max_slides)).copy()
    if rows.empty:
        raise ValueError(
            f"No manifest rows selected for splits={splits}, slide_ids={slide_ids}, max_slides={max_slides}."
        )
    return rows


def load_stimage_slides(
    expression_config: dict,
    rows: pd.DataFrame,
    *,
    target_kind: TargetKind = "log1p_rate",
    gene_limit: int | None = None,
) -> tuple[list[STImagePatchSlide], list[str]]:
    if target_kind != "log1p_rate":
        raise ValueError("STimage formal adapter currently supports log1p_rate only.")
    manifest_path = resolve_project_path(expression_config["data"]["manifest"])
    if manifest_path is None:
        raise ValueError("Expression config data.manifest resolved to None")
    base_dir = manifest_path.parent
    target_genes, gene_indices = selected_genes_from_config(expression_config, base_dir=base_dir)
    if target_genes is None or gene_indices is not None:
        raise ValueError("STimage adapter requires data.gene_names_path target genes.")
    if gene_limit is not None:
        if int(gene_limit) <= 0:
            raise ValueError("gene_limit must be positive when provided.")
        target_genes = list(target_genes)[: int(gene_limit)]
    gene_key, raw_st_root = gene_key_settings_from_config(expression_config)
    raw_st_root = resolve_project_path(raw_st_root) if raw_st_root is not None else None
    raw_root = resolve_project_path(expression_config["paths"]["raw_root"])
    if raw_root is None:
        raise ValueError("Expression config paths.raw_root resolved to None")
    min_total_counts = float(expression_config["data"].get("min_total_counts", 1.0))
    slides = [
        _load_stimage_patch_slide(
            row=row,
            base_dir=base_dir,
            raw_root=raw_root,
            target_genes=list(target_genes),
            gene_key=gene_key,
            raw_st_root=raw_st_root,
            min_total_counts=min_total_counts,
        )
        for row in rows.itertuples(index=False)
    ]
    return slides, list(target_genes)


def tensorflow_cuda_library_paths() -> list[str]:
    """Return pip-installed NVIDIA library dirs needed by TensorFlow GPU wheels."""
    candidates = [
        "cublas",
        "cuda_cupti",
        "cuda_nvcc",
        "cuda_nvrtc",
        "cuda_runtime",
        "cudnn",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nccl",
        "nvjitlink",
    ]
    paths: list[str] = []
    for entry in sys.path:
        root = Path(entry) / "nvidia"
        if not root.exists():
            continue
        for name in candidates:
            lib_dir = root / name / "lib"
            if lib_dir.exists():
                paths.append(str(lib_dir))
    return paths



def train_stimage_sourcefaithful(
    *,
    expression_config: dict,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    output_dir: str | Path,
    upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT,
    target_kind: TargetKind = "log1p_rate",
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 20,
    tile_size: int = 299,
    cnn_base: str = "resnet50",
    fine_tuning: bool = False,
    seed: int = 2026,
    gene_limit: int | None = None,
    optimizer_mode: str = "default",
    run_eagerly: bool = False,
) -> dict[str, Any]:
    import tensorflow as tf

    set_seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_slides, target_genes = load_stimage_slides(
        expression_config,
        train_rows,
        target_kind=target_kind,
        gene_limit=gene_limit,
    )
    val_slides, val_genes = load_stimage_slides(
        expression_config,
        val_rows,
        target_kind=target_kind,
        gene_limit=gene_limit,
    )
    if val_genes != target_genes:
        raise ValueError("Train/validation gene lists differ.")
    train_seq = STImageSpotSequence(
        slides=train_slides,
        target_genes=target_genes,
        batch_size=batch_size,
        tile_size=tile_size,
        target_kind=target_kind,
        shuffle=True,
        seed=seed,
        include_targets=True,
    )
    val_seq = STImageSpotSequence(
        slides=val_slides,
        target_genes=target_genes,
        batch_size=batch_size,
        tile_size=tile_size,
        target_kind=target_kind,
        shuffle=False,
        seed=seed,
        include_targets=True,
    )
    model = build_stimage_model(
        n_genes=len(target_genes),
        tile_size=tile_size,
        cnn_base=cnn_base,
        fine_tuning=fine_tuning,
        upstream_root=upstream_root,
        optimizer_mode=optimizer_mode,
        run_eagerly=run_eagerly,
    )
    best_weights = output_dir / "best_model.weights.h5"
    last_weights = output_dir / "last_model.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(patience),
            restore_best_weights=True,
            mode="min",
        ),
    ]
    try:
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=int(epochs),
            callbacks=callbacks,
            verbose=2,
        )
        model.save_weights(last_weights)
    finally:
        train_seq.close()
        val_seq.close()
    hist = history.history
    rows = []
    n_epochs = len(hist.get("loss", []))
    for idx in range(n_epochs):
        rows.append(
            {
                "epoch": idx + 1,
                "train_loss": float(hist["loss"][idx]) if "loss" in hist else np.nan,
                "val_loss": float(hist["val_loss"][idx]) if "val_loss" in hist else np.nan,
            }
        )
    train_log = output_dir / "train_log.csv"
    pd.DataFrame(rows).to_csv(train_log, index=False)
    val_losses = [row["val_loss"] for row in rows if np.isfinite(row["val_loss"])]
    best_val = float(np.min(val_losses)) if val_losses else float("nan")
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else None
    summary = {
        "method": "stimage_sourcefaithful",
        "output_dir": str(output_dir),
        "checkpoint": str(best_weights),
        "last_checkpoint": str(last_weights),
        "target_kind": target_kind,
        "n_train_slides": int(len(train_slides)),
        "n_val_slides": int(len(val_slides)),
        "n_train_spots": int(sum(slide.n_spots for slide in train_slides)),
        "n_val_spots": int(sum(slide.n_spots for slide in val_slides)),
        "n_genes": int(len(target_genes)),
        "gene_limit": int(gene_limit) if gene_limit is not None else None,
        "is_reduced_gene_run": bool(gene_limit is not None),
        "epochs": int(n_epochs),
        "max_epochs": int(epochs),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "early_stopping_patience": int(patience),
        "stopped_early": bool(n_epochs < int(epochs)),
        "tile_size": int(tile_size),
        "cnn_base": str(cnn_base),
        "fine_tuning": bool(fine_tuning),
        "optimizer_mode": str(optimizer_mode),
        "run_eagerly": bool(run_eagerly),
        "source_info": stimage_source_metadata(upstream_root),
        "outputs": {
            "train_log": str(train_log),
            "best_model": str(best_weights),
            "last_model": str(last_weights),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    train_seq.slide_summary_frame().to_csv(output_dir / "train_slides.csv", index=False)
    val_seq.slide_summary_frame().to_csv(output_dir / "val_slides.csv", index=False)
    return summary


def _nb_mean_from_model_outputs(outputs: list[np.ndarray]) -> np.ndarray:
    preds = np.zeros((outputs[0].shape[0], len(outputs)), dtype=np.float32)
    for i, arr in enumerate(outputs):
        n = np.asarray(arr[:, 0], dtype=np.float32)
        p = np.clip(np.asarray(arr[:, 1], dtype=np.float32), 1.0e-6, 1.0 - 1.0e-6)
        preds[:, i] = n * (1.0 - p) / p
    return preds


def export_stimage_sourcefaithful_predictions(
    *,
    expression_config: dict,
    test_rows: pd.DataFrame,
    checkpoint_path: str | Path,
    prediction_root: str | Path,
    upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT,
    target_kind: TargetKind = "log1p_rate",
    batch_size: int = 32,
    tile_size: int = 299,
    cnn_base: str = "resnet50",
    fine_tuning: bool = False,
    gene_limit: int | None = None,
) -> dict[str, Any]:
    prediction_root = Path(prediction_root)
    pred_dir = prediction_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    slides, target_genes = load_stimage_slides(
        expression_config,
        test_rows,
        target_kind=target_kind,
        gene_limit=gene_limit,
    )
    model = build_stimage_model(
        n_genes=len(target_genes),
        tile_size=tile_size,
        cnn_base=cnn_base,
        fine_tuning=fine_tuning,
        upstream_root=upstream_root,
    )
    model.load_weights(str(checkpoint_path))
    manifest_rows: list[dict[str, Any]] = []
    for slide in slides:
        seq = STImageSpotSequence(
            slides=[slide],
            target_genes=target_genes,
            batch_size=batch_size,
            tile_size=tile_size,
            target_kind=target_kind,
            shuffle=False,
            include_targets=False,
        )
        try:
            outputs = model.predict(seq, verbose=0)
        finally:
            seq.close()
        if isinstance(outputs, np.ndarray):
            raise ValueError("STimage model returned a single output; expected one output per gene.")
        pred = _nb_mean_from_model_outputs(list(outputs))
        pred_path = pred_dir / f"{slide.sample_id}_{target_kind}.npy"
        mem = open_memmap(pred_path, mode="w+", dtype=np.float32, shape=pred.shape)
        mem[:] = pred
        mem.flush()
        del mem
        manifest_rows.append(
            {
                "sample_id": slide.sample_id,
                "split": slide.split,
                "organ": slide.organ,
                "cohort": slide.cohort,
                "disease_state": slide.disease_state,
                "n_spots": int(slide.n_spots),
                "n_genes": int(len(target_genes)),
                "prediction_path": str(pred_path),
                "complete": bool(pred.shape == (slide.n_spots, len(target_genes))),
            }
        )
    (prediction_root / "genes.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(prediction_root / "prediction_manifest.csv", index=False)
    summary = {
        "method": "stimage_sourcefaithful",
        "checkpoint": str(checkpoint_path),
        "prediction_root": str(prediction_root),
        "prediction_kind": target_kind,
        "n_slides": int(len(slides)),
        "n_genes": int(len(target_genes)),
        "gene_limit": int(gene_limit) if gene_limit is not None else None,
        "is_reduced_gene_run": bool(gene_limit is not None),
        "complete_prediction_arrays": bool(manifest["complete"].all()) if not manifest.empty else False,
        "all_slide_predictions_complete": bool(manifest["complete"].all()) if not manifest.empty else False,
        "benchmark_evaluable_without_truncation": bool(manifest["complete"].all()) if not manifest.empty else False,
        "batch_size": int(batch_size),
        "tile_size": int(tile_size),
        "cnn_base": str(cnn_base),
        "fine_tuning": bool(fine_tuning),
        "outputs": {
            "prediction_manifest": str(prediction_root / "prediction_manifest.csv"),
            "genes": str(prediction_root / "genes.txt"),
            "predictions": str(pred_dir),
            "summary": str(prediction_root / "prediction_summary.json"),
        },
    }
    (prediction_root / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_stimage_predictions(
    *,
    prediction_root: str | Path,
    benchmark_out_dir: str | Path,
    expression_config: dict,
    splits: list[str] | None = None,
    slide_ids: list[str] | None = None,
    target_kind: TargetKind = "log1p_rate",
) -> dict[str, Any]:
    return evaluate_prediction_bundle(
        expression_config=expression_config,
        prediction_root=prediction_root,
        out_dir=benchmark_out_dir,
        method_name="stimage_sourcefaithful",
        prediction_kind=target_kind,
        splits=splits or ["test"],
        slide_ids=slide_ids,
        prediction_pattern=f"predictions/{{sample_id}}_{target_kind}.npy",
        prediction_genes_path="genes.txt",
    )


def data_smoke_summary(
    *,
    expression_config: dict,
    rows: pd.DataFrame,
    output_dir: str | Path,
    upstream_root: str | Path = DEFAULT_STIMAGE_UPSTREAM_ROOT,
    batch_size: int = 2,
    tile_size: int = 299,
    model_genes: int = 8,
    target_kind: TargetKind = "log1p_rate",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slides, target_genes = load_stimage_slides(expression_config, rows, target_kind=target_kind)
    target_genes = target_genes[: int(model_genes)]
    gene_indices = np.arange(len(target_genes), dtype=np.int64)
    seq = STImageSpotSequence(
        slides=slides,
        target_genes=target_genes,
        batch_size=batch_size,
        tile_size=tile_size,
        target_kind=target_kind,
        shuffle=False,
        include_targets=True,
        gene_indices=gene_indices,
    )
    try:
        batch = seq[0]
    finally:
        seq.close()
    x, y_tuple, w_tuple = batch
    summary = {
        "method": "stimage_sourcefaithful",
        "n_slides": int(len(slides)),
        "n_spots": int(sum(slide.n_spots for slide in slides)),
        "n_genes": int(len(target_genes)),
        "batch_x_shape": list(x.shape),
        "batch_y_outputs": int(len(y_tuple)),
        "first_y_shape": list(y_tuple[0].shape) if y_tuple else [],
        "first_weight_shape": list(w_tuple[0].shape) if w_tuple else [],
        "tile_size": int(tile_size),
        "source_info": stimage_source_metadata(upstream_root),
    }
    (output_dir / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    seq.slide_summary_frame().to_csv(output_dir / "slides.csv", index=False)
    return summary
