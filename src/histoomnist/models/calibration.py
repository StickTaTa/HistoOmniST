from __future__ import annotations

import numpy as np


class AffineLogSFCalibrator:
    """Few-shot calibration for frozen SF predictions.

    Fit only on calibration slides:

    calibrated_log_sf = scale * pred_log_sf + bias
    """

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = float(scale)
        self.bias = float(bias)

    def fit(self, pred_log_sf: np.ndarray, true_log_sf: np.ndarray) -> "AffineLogSFCalibrator":
        x = np.asarray(pred_log_sf, dtype=np.float64).reshape(-1)
        y = np.asarray(true_log_sf, dtype=np.float64).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            raise ValueError("Need at least three valid calibration spots.")
        x = x[mask]
        y = y[mask]
        var_x = float(np.var(x))
        if var_x < 1e-12:
            self.scale = 1.0
            self.bias = float(y.mean() - x.mean())
        else:
            self.scale = float(np.cov(x, y, bias=True)[0, 1] / var_x)
            self.bias = float(y.mean() - self.scale * x.mean())
        return self

    def transform(self, pred_log_sf: np.ndarray) -> np.ndarray:
        return (self.scale * np.asarray(pred_log_sf) + self.bias).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "bias": self.bias}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "AffineLogSFCalibrator":
        return cls(scale=float(data["scale"]), bias=float(data["bias"]))
