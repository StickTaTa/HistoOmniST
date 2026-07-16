from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


RATE_LOG1P_PREFIX = "rate_log1p_"
COUNT_PREFIX = "count_"
COUNT_LOG1P_PREFIX = "count_log1p_"


def load_sf_model(
    checkpoint_path: str | Path,
    *,
    device: str,
    fallback_config_path: str | Path | None = None,
):
    """Load an SF model, preferring the configuration embedded in its checkpoint."""
    import torch

    from histoomnist.eval.evaluate_combined import _load_sf_model
    from histoomnist.train.common import load_checkpoint
    from histoomnist.utils.config import load_config

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        if fallback_config_path is None:
            config = {"model": {}}
        else:
            config = load_config(fallback_config_path)
    model = _load_sf_model(config, checkpoint, torch.device(device))
    mean = np.asarray(checkpoint.get("feature_mean"), dtype=np.float32)
    std = np.asarray(checkpoint.get("feature_std"), dtype=np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return model, mean, std, config


def mean_one_sf_from_log(raw_log_sf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw log-SF predictions to mean-one SF within one slide."""
    values = np.asarray(raw_log_sf, dtype=np.float64).reshape(-1)
    if values.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty
    if not np.isfinite(values).all():
        raise ValueError("Raw log-SF predictions contain non-finite values.")

    # Subtracting the maximum preserves the normalized SF while avoiding overflow.
    shifted = np.exp(values - float(values.max()))
    shifted_mean = float(shifted.mean())
    if not np.isfinite(shifted_mean) or shifted_mean <= 0.0:
        raise ValueError("Cannot normalize predicted SF to mean one.")
    sf = shifted / shifted_mean
    normalized_log_sf = np.log(sf)
    return normalized_log_sf.astype(np.float32), sf.astype(np.float32)


def predict_mean_one_sf(
    features: np.ndarray,
    *,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict raw log-SF and return slide-normalized log-SF and SF."""
    import torch

    x = ((features.astype(np.float32) - mean[None, :]) / std[None, :]).astype(np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        tensor = torch.from_numpy(x[start : start + batch_size]).to(torch.device(device))
        with torch.inference_mode():
            prediction = model(tensor).detach().cpu().numpy().reshape(-1).astype(np.float32)
        chunks.append(prediction)
    raw_log_sf = np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)
    normalized_log_sf, sf = mean_one_sf_from_log(raw_log_sf)
    return raw_log_sf, normalized_log_sf, sf


def reconstruct_count_scale(
    rate_log1p: np.ndarray,
    sf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct count after projecting transformed rate predictions to nonnegative values."""
    rate_log1p_values = np.asarray(rate_log1p, dtype=np.float64)
    sf_values = np.asarray(sf, dtype=np.float64).reshape(-1)
    if rate_log1p_values.ndim not in {1, 2}:
        raise ValueError("rate_log1p must be a one- or two-dimensional array.")
    if rate_log1p_values.shape[0] != sf_values.shape[0]:
        raise ValueError(
            f"Rate/SF row mismatch: rate={rate_log1p_values.shape[0]} sf={sf_values.shape[0]}"
        )
    if not np.isfinite(rate_log1p_values).all() or not np.isfinite(sf_values).all():
        raise ValueError("Rate or SF predictions contain non-finite values.")
    if np.any(sf_values <= 0.0):
        raise ValueError("Predicted SF values must be positive.")

    rate = np.clip(np.expm1(rate_log1p_values), a_min=0.0, a_max=None)
    multiplier = sf_values if rate_log1p_values.ndim == 1 else sf_values[:, None]
    count = rate * multiplier
    count_log1p = np.log1p(count)
    if not np.isfinite(count).all() or not np.isfinite(count_log1p).all():
        raise ValueError("Count-scale reconstruction produced non-finite values.")
    return count.astype(np.float32), count_log1p.astype(np.float32)


def add_count_scale_columns(
    frame: pd.DataFrame,
    genes: Sequence[str],
    *,
    source_rate_prefix: str,
    sf_column: str = "pred_sf",
    keep_explicit_rate: bool = True,
) -> pd.DataFrame:
    """Add explicit rate, count and log1p-count columns for selected genes."""
    if sf_column not in frame.columns:
        raise KeyError(f"Missing SF column: {sf_column}")
    sf = frame[sf_column].to_numpy(dtype=np.float32)
    generated: dict[str, np.ndarray] = {}
    for gene in genes:
        source_column = f"{source_rate_prefix}{gene}"
        if source_column not in frame.columns:
            continue
        rate_log1p = frame[source_column].to_numpy(dtype=np.float32)
        count, count_log1p = reconstruct_count_scale(rate_log1p, sf)
        if keep_explicit_rate:
            generated[f"{RATE_LOG1P_PREFIX}{gene}"] = rate_log1p
        generated[f"{COUNT_PREFIX}{gene}"] = count
        generated[f"{COUNT_LOG1P_PREFIX}{gene}"] = count_log1p
    if not generated:
        return frame

    generated_frame = pd.DataFrame(generated, index=frame.index)
    overlapping = [column for column in generated_frame if column in frame.columns]
    if overlapping:
        frame = frame.copy()
        frame.loc[:, overlapping] = generated_frame[overlapping]
        generated_frame = generated_frame.drop(columns=overlapping)
    return pd.concat([frame, generated_frame], axis=1)


def add_count_scale_programs(
    frame: pd.DataFrame,
    genes: Sequence[str],
    programs: Mapping[str, Sequence[str]],
    *,
    program_prefix: str = "program_",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Calculate programs from reconstructed log1p count-scale gene values."""
    available = set(genes)
    used: dict[str, list[str]] = {}
    generated: dict[str, pd.Series] = {}
    for name, members in programs.items():
        present = [
            gene
            for gene in members
            if gene in available and f"{COUNT_LOG1P_PREFIX}{gene}" in frame.columns
        ]
        if not present:
            continue
        used[name] = present
        columns = [f"{COUNT_LOG1P_PREFIX}{gene}" for gene in present]
        generated[f"{program_prefix}{name}"] = frame[columns].mean(axis=1)
    if not generated:
        return frame, used

    generated_frame = pd.DataFrame(generated, index=frame.index)
    overlapping = [column for column in generated_frame if column in frame.columns]
    if overlapping:
        frame = frame.copy()
        frame.loc[:, overlapping] = generated_frame[overlapping]
        generated_frame = generated_frame.drop(columns=overlapping)
    return pd.concat([frame, generated_frame], axis=1), used
