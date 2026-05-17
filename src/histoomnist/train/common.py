from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_payload(model: torch.nn.Module, cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> dict:
    payload = {
        "model_state": model.state_dict(),
        "config": cfg,
    }
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)
