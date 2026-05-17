from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np


RGB_FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "rgb_p10_r",
    "rgb_p10_g",
    "rgb_p10_b",
    "rgb_p50_r",
    "rgb_p50_g",
    "rgb_p50_b",
    "rgb_p90_r",
    "rgb_p90_g",
    "rgb_p90_b",
    "gray_mean",
    "gray_std",
    "gray_p10",
    "gray_p50",
    "gray_p90",
    "tissue_fraction",
    "saturation_mean",
    "saturation_std",
]


def batched(values: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def rgb_stats_features(images: np.ndarray) -> np.ndarray:
    """Compute lightweight color/tissue descriptors from uint8 RGB spot patches."""
    x = np.asarray(images, dtype=np.float32) / 255.0
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected images with shape [N,H,W,3], got {x.shape}")
    flat = x.reshape(x.shape[0], -1, 3)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    percentiles = np.percentile(flat, [10, 50, 90], axis=1).transpose(1, 0, 2).reshape(x.shape[0], -1)

    gray = 0.299 * flat[..., 0] + 0.587 * flat[..., 1] + 0.114 * flat[..., 2]
    gray_stats = np.stack(
        [
            gray.mean(axis=1),
            gray.std(axis=1),
            np.percentile(gray, 10, axis=1),
            np.percentile(gray, 50, axis=1),
            np.percentile(gray, 90, axis=1),
        ],
        axis=1,
    )

    rgb_max = flat.max(axis=2)
    rgb_min = flat.min(axis=2)
    saturation = (rgb_max - rgb_min) / np.clip(rgb_max, 1.0e-6, None)
    tissue = (gray < 0.92) & (saturation > 0.05)
    tissue_fraction = tissue.mean(axis=1, keepdims=True)
    saturation_stats = np.stack([saturation.mean(axis=1), saturation.std(axis=1)], axis=1)

    features = np.concatenate(
        [mean, std, percentiles, gray_stats, tissue_fraction, saturation_stats],
        axis=1,
    )
    return features.astype(np.float32)


def hipt256_feature_names(dim: int = 384) -> list[str]:
    return [f"hipt256_cls_{i:03d}" for i in range(dim)]


def _import_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hipt256_model(
    *,
    hipt_source_dir: Path,
    weights_path: Path,
    device: str,
):
    """Load the HIPT ViT-256 encoder from a local HIPT source checkout."""
    import torch

    source = Path(hipt_source_dir)
    weights = Path(weights_path)
    if not source.exists():
        raise FileNotFoundError(f"HIPT source directory not found: {source}")
    if not weights.exists():
        raise FileNotFoundError(f"HIPT ViT-256 weights not found: {weights}")

    vits = _import_module_from_path("histoomnist_external_hipt_vits", source / "vision_transformer.py")
    model = vits.vit_small(patch_size=16, num_classes=0)
    state_dict = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    state_dict = {
        str(k).replace("module.", "").replace("backbone.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(torch.device(device))
    for param in model.parameters():
        param.requires_grad = False
    return model


def hipt256_features(images: np.ndarray, model, *, device: str) -> np.ndarray:
    """Extract HIPT ViT-256 CLS features from uint8 224x224 RGB spot patches."""
    import torch

    x = np.asarray(images)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected images with shape [N,H,W,3], got {x.shape}")
    if x.shape[1] != 224 or x.shape[2] != 224:
        raise ValueError(f"HIPT ViT-256 expects 224x224 patches, got {x.shape[1:3]}")
    x = x.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    tensor = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(torch.device(device))
    with torch.inference_mode():
        features = model(tensor).detach().cpu().numpy()
    return features.astype(np.float32)

