from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.format import open_memmap

from histoomnist.eval.benchmark_predictions import evaluate_prediction_bundle
from histoomnist.external.stimage_sourcefaithful import (
    DEFAULT_STIMAGE_UPSTREAM_ROOT,
    STImagePatchSlide,
    STImageSpotSequence,
    _resize_hwc_uint8,
    load_stimage_slides,
    select_manifest_rows,
    stimage_source_metadata,
)
from histoomnist.utils.project_paths import resolve_project_path
from histoomnist.utils.seed import set_seed


@dataclass(frozen=True)
class STImageFeatureSlide:
    slide: STImagePatchSlide
    feature_path: Path

    @property
    def n_spots(self) -> int:
        return self.slide.n_spots


class VectorizedSTImageNBHead(tf.Module):
    """Vectorized form of STimage's one Dense(2) NB head per gene."""

    def __init__(self, *, feature_dim: int, n_genes: int, seed: int = 2026):
        super().__init__()
        limit = math.sqrt(6.0 / float(feature_dim + 2))
        rng = tf.random.Generator.from_seed(int(seed))
        init = rng.uniform(
            shape=(int(feature_dim), int(n_genes), 2),
            minval=-limit,
            maxval=limit,
            dtype=tf.float32,
        )
        self.kernel = tf.Variable(init, name="kernel")
        self.bias = tf.Variable(tf.zeros((int(n_genes), 2), dtype=tf.float32), name="bias")

    @property
    def trainable_variables(self):
        return [self.kernel, self.bias]

    def __call__(self, features: tf.Tensor) -> tf.Tensor:
        logits = tf.einsum("bf,fgp->bgp", features, self.kernel) + self.bias
        n = tf.nn.softplus(logits[..., 0])
        p = tf.nn.sigmoid(logits[..., 1])
        return tf.stack([n, p], axis=-1)


def build_stimage_resnet50_gap(tile_size: int = 299) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(int(tile_size), int(tile_size), 3), name="tile_input")
    base = tf.keras.applications.ResNet50(input_tensor=inputs, weights="imagenet", include_top=False)
    base.trainable = False
    outputs = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _slide_image_batch(slide: STImagePatchSlide, local_indices: np.ndarray, tile_size: int) -> np.ndarray:
    x = np.zeros((len(local_indices), int(tile_size), int(tile_size), 3), dtype=np.float32)
    with h5py.File(slide.patch_h5_path, "r") as handle:
        images = handle["img"]
        for out_idx, local_idx in enumerate(local_indices):
            patch_index = int(slide.patch_indices[int(local_idx)])
            patch = np.asarray(images[patch_index])
            x[out_idx] = _resize_hwc_uint8(patch, int(tile_size)).astype(np.float32)
    return x


def extract_or_load_features(
    *,
    slides: list[STImagePatchSlide],
    target_genes: list[str],
    feature_root: str | Path,
    batch_size: int = 256,
    tile_size: int = 299,
) -> tuple[list[STImageFeatureSlide], dict[str, Any]]:
    feature_root = Path(feature_root)
    feature_dir = feature_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    extractor = build_stimage_resnet50_gap(tile_size=tile_size)
    rows: list[dict[str, Any]] = []
    out: list[STImageFeatureSlide] = []
    for slide in slides:
        feature_path = feature_dir / f"{slide.sample_id}_resnet50_gap.npy"
        if feature_path.exists():
            arr = np.load(feature_path, mmap_mode="r")
            if arr.shape != (slide.n_spots, 2048):
                raise ValueError(f"Feature shape mismatch for {slide.sample_id}: {arr.shape}")
            status = "loaded"
        else:
            mem = open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(slide.n_spots, 2048))
            for start in range(0, slide.n_spots, int(batch_size)):
                stop = min(start + int(batch_size), slide.n_spots)
                x = _slide_image_batch(slide, np.arange(start, stop, dtype=np.int64), tile_size)
                feats = extractor.predict_on_batch(x)
                mem[start:stop] = np.asarray(feats, dtype=np.float32)
            mem.flush()
            del mem
            status = "extracted"
        rows.append(
            {
                "sample_id": slide.sample_id,
                "split": slide.split,
                "organ": slide.organ,
                "n_spots": int(slide.n_spots),
                "feature_path": str(feature_path),
                "status": status,
            }
        )
        out.append(STImageFeatureSlide(slide=slide, feature_path=feature_path))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(feature_root / "feature_manifest.csv", index=False)
    summary = {
        "method": "stimage_sourcefaithful_featurecached",
        "feature_root": str(feature_root),
        "n_slides": int(len(slides)),
        "n_spots": int(sum(slide.n_spots for slide in slides)),
        "n_genes": int(len(target_genes)),
        "feature_dim": 2048,
        "tile_size": int(tile_size),
        "batch_size": int(batch_size),
        "source_equivalence": "ResNet50 include_top=False plus GlobalAveragePooling2D matches the frozen CNN trunk in official CNN_NB_multiple_genes.",
        "outputs": {"feature_manifest": str(feature_root / "feature_manifest.csv")},
    }
    (feature_root / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out, summary


def _target_matrix(slide: STImagePatchSlide, local_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = slide.counts[local_indices].toarray().astype(np.float32, copy=False)
    sf = np.asarray(slide.size_factor[local_indices], dtype=np.float32).reshape(-1, 1)
    y = np.log1p(counts / np.clip(sf, 1.0e-6, None)).astype(np.float32, copy=False)
    w = np.broadcast_to(slide.measured_genes.astype(np.float32), y.shape).astype(np.float32, copy=False)
    return y, w


def _feature_batch(
    slides: list[STImageFeatureSlide],
    cumlen: np.ndarray,
    global_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pieces_x: list[np.ndarray] = []
    pieces_y: list[np.ndarray] = []
    pieces_w: list[np.ndarray] = []
    for slide_idx, wrapped in enumerate(slides):
        start = 0 if slide_idx == 0 else int(cumlen[slide_idx - 1])
        stop = int(cumlen[slide_idx])
        mask = (global_indices >= start) & (global_indices < stop)
        if not np.any(mask):
            continue
        local = (global_indices[mask] - start).astype(np.int64)
        feats = np.load(wrapped.feature_path, mmap_mode="r")
        pieces_x.append(np.asarray(feats[local], dtype=np.float32))
        y, w = _target_matrix(wrapped.slide, local)
        pieces_y.append(y)
        pieces_w.append(w)
    return np.concatenate(pieces_x, axis=0), np.concatenate(pieces_y, axis=0), np.concatenate(pieces_w, axis=0)


def _nb_loss(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    n = y_pred[..., 0]
    p = tf.clip_by_value(y_pred[..., 1], 1.0e-6, 1.0 - 1.0e-6)
    loss = (
        tf.math.lgamma(n)
        + tf.math.lgamma(y_true + 1.0)
        - tf.math.lgamma(n + y_true)
        - n * tf.math.log(p)
        - y_true * tf.math.log(1.0 - p)
    )
    numerator = tf.reduce_sum(loss * weights, axis=0)
    denominator = tf.reduce_sum(weights, axis=0)
    per_gene = tf.where(denominator > 0.0, numerator / tf.maximum(denominator, 1.0e-6), 0.0)
    return tf.reduce_sum(per_gene)


def _run_head_epoch(
    *,
    head: VectorizedSTImageNBHead,
    optimizer: tf.keras.optimizers.Optimizer | None,
    slides: list[STImageFeatureSlide],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> float:
    lengths = np.asarray([slide.n_spots for slide in slides], dtype=np.int64)
    cumlen = np.cumsum(lengths)
    indices = np.arange(int(cumlen[-1]), dtype=np.int64)
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(indices)
    losses: list[float] = []
    for start in range(0, len(indices), int(batch_size)):
        batch_idx = indices[start : start + int(batch_size)]
        x, y, w = _feature_batch(slides, cumlen, batch_idx)
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
        w_tf = tf.convert_to_tensor(w, dtype=tf.float32)
        if optimizer is None:
            pred = head(x_tf)
            loss = _nb_loss(y_tf, pred, w_tf)
        else:
            with tf.GradientTape() as tape:
                pred = head(x_tf)
                loss = _nb_loss(y_tf, pred, w_tf)
            grads = tape.gradient(loss, head.trainable_variables)
            optimizer.apply_gradients(zip(grads, head.trainable_variables))
        losses.append(float(loss.numpy()))
    return float(np.mean(losses)) if losses else float("nan")


def save_head(head: VectorizedSTImageNBHead, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, kernel=head.kernel.numpy(), bias=head.bias.numpy())


def load_head(path: str | Path) -> VectorizedSTImageNBHead:
    data = np.load(path)
    kernel = np.asarray(data["kernel"], dtype=np.float32)
    bias = np.asarray(data["bias"], dtype=np.float32)
    head = VectorizedSTImageNBHead(feature_dim=kernel.shape[0], n_genes=kernel.shape[1])
    head.kernel.assign(kernel)
    head.bias.assign(bias)
    return head


def train_featurecached_stimage(
    *,
    expression_config: dict,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    output_dir: str | Path,
    feature_root: str | Path,
    epochs: int = 100,
    patience: int = 20,
    feature_batch_size: int = 256,
    train_batch_size: int = 4096,
    tile_size: int = 299,
    seed: int = 2026,
) -> dict[str, Any]:
    set_seed(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_slides, target_genes = load_stimage_slides(expression_config, train_rows)
    val_slides, val_genes = load_stimage_slides(expression_config, val_rows)
    if val_genes != target_genes:
        raise ValueError("Train/validation gene lists differ.")
    train_features, train_feature_summary = extract_or_load_features(
        slides=train_slides,
        target_genes=target_genes,
        feature_root=Path(feature_root) / "train",
        batch_size=feature_batch_size,
        tile_size=tile_size,
    )
    val_features, val_feature_summary = extract_or_load_features(
        slides=val_slides,
        target_genes=target_genes,
        feature_root=Path(feature_root) / "val",
        batch_size=feature_batch_size,
        tile_size=tile_size,
    )
    head = VectorizedSTImageNBHead(feature_dim=2048, n_genes=len(target_genes), seed=seed)
    optimizer = tf.keras.optimizers.Adam(1.0e-5, jit_compile=False)
    best_val = float("inf")
    best_epoch = None
    bad_epochs = 0
    rows: list[dict[str, Any]] = []
    best_path = output_dir / "best_head.npz"
    last_path = output_dir / "last_head.npz"
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        train_loss = _run_head_epoch(
            head=head,
            optimizer=optimizer,
            slides=train_features,
            batch_size=train_batch_size,
            shuffle=True,
            seed=seed + epoch,
        )
        val_loss = _run_head_epoch(
            head=head,
            optimizer=None,
            slides=val_features,
            batch_size=train_batch_size,
            shuffle=False,
            seed=seed,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            bad_epochs = 0
            save_head(head, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    save_head(head, last_path)
    train_log = output_dir / "train_log.csv"
    pd.DataFrame(rows).to_csv(train_log, index=False)
    summary = {
        "method": "stimage_sourcefaithful_featurecached",
        "source_core": "official STimage frozen ResNet50+GlobalAveragePooling2D plus vectorized equivalent of one Dense(2) NB head per gene",
        "source_equivalence": "The official default STimage CNN trunk is frozen, so cached ResNet50 GAP features and a vectorized Dense(2)-per-gene head preserve the same function class while avoiding Keras 16,942-output tracing.",
        "target_kind": "log1p_rate",
        "n_train_slides": int(len(train_slides)),
        "n_val_slides": int(len(val_slides)),
        "n_train_spots": int(sum(slide.n_spots for slide in train_slides)),
        "n_val_spots": int(sum(slide.n_spots for slide in val_slides)),
        "n_genes": int(len(target_genes)),
        "epochs": int(len(rows)),
        "max_epochs": int(epochs),
        "patience": int(patience),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "feature_batch_size": int(feature_batch_size),
        "train_batch_size": int(train_batch_size),
        "tile_size": int(tile_size),
        "elapsed_seconds": float(time.perf_counter() - started),
        "source_info": stimage_source_metadata(DEFAULT_STIMAGE_UPSTREAM_ROOT),
        "feature_summaries": {"train": train_feature_summary, "val": val_feature_summary},
        "outputs": {"best_head": str(best_path), "last_head": str(last_path), "train_log": str(train_log)},
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "genes.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    return summary


def export_featurecached_predictions(
    *,
    expression_config: dict,
    test_rows: pd.DataFrame,
    checkpoint_path: str | Path,
    prediction_root: str | Path,
    feature_root: str | Path,
    feature_batch_size: int = 256,
    predict_batch_size: int = 4096,
    tile_size: int = 299,
) -> dict[str, Any]:
    prediction_root = Path(prediction_root)
    pred_dir = prediction_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    test_slides, target_genes = load_stimage_slides(expression_config, test_rows)
    test_features, feature_summary = extract_or_load_features(
        slides=test_slides,
        target_genes=target_genes,
        feature_root=feature_root,
        batch_size=feature_batch_size,
        tile_size=tile_size,
    )
    head = load_head(checkpoint_path)
    rows: list[dict[str, Any]] = []
    for wrapped in test_features:
        slide = wrapped.slide
        feats = np.load(wrapped.feature_path, mmap_mode="r")
        pred_path = pred_dir / f"{slide.sample_id}_log1p_rate.npy"
        mem = open_memmap(pred_path, mode="w+", dtype=np.float32, shape=(slide.n_spots, len(target_genes)))
        for start in range(0, slide.n_spots, int(predict_batch_size)):
            stop = min(start + int(predict_batch_size), slide.n_spots)
            params = head(tf.convert_to_tensor(np.asarray(feats[start:stop], dtype=np.float32), dtype=tf.float32))
            n = params[..., 0].numpy()
            p = np.clip(params[..., 1].numpy(), 1.0e-6, 1.0 - 1.0e-6)
            mem[start:stop] = n * (1.0 - p) / p
        mem.flush()
        del mem
        rows.append(
            {
                "sample_id": slide.sample_id,
                "split": slide.split,
                "organ": slide.organ,
                "cohort": slide.cohort,
                "disease_state": slide.disease_state,
                "n_spots": int(slide.n_spots),
                "n_genes": int(len(target_genes)),
                "prediction_path": str(pred_path),
                "complete": True,
            }
        )
    pd.DataFrame(rows).to_csv(prediction_root / "prediction_manifest.csv", index=False)
    (prediction_root / "genes.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    summary = {
        "method": "stimage_sourcefaithful_featurecached",
        "checkpoint": str(checkpoint_path),
        "prediction_root": str(prediction_root),
        "prediction_kind": "log1p_rate",
        "n_slides": int(len(test_slides)),
        "n_genes": int(len(target_genes)),
        "complete_prediction_arrays": True,
        "all_slide_predictions_complete": True,
        "benchmark_evaluable_without_truncation": True,
        "feature_summary": feature_summary,
        "outputs": {
            "prediction_manifest": str(prediction_root / "prediction_manifest.csv"),
            "genes": str(prediction_root / "genes.txt"),
            "predictions": str(pred_dir),
            "summary": str(prediction_root / "prediction_summary.json"),
        },
    }
    (prediction_root / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_featurecached_predictions(
    *,
    prediction_root: str | Path,
    benchmark_out_dir: str | Path,
    expression_config: dict,
    splits: list[str] | None = None,
    slide_ids: list[str] | None = None,
) -> dict[str, Any]:
    return evaluate_prediction_bundle(
        expression_config=expression_config,
        prediction_root=prediction_root,
        out_dir=benchmark_out_dir,
        method_name="stimage_sourcefaithful_featurecached",
        prediction_kind="log1p_rate",
        splits=splits or ["test"],
        slide_ids=slide_ids,
        prediction_pattern="predictions/{sample_id}_log1p_rate.npy",
        prediction_genes_path="genes.txt",
    )
