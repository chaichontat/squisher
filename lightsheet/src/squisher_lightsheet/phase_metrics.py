from __future__ import annotations

import numpy as np


def robust_normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    out = np.zeros(image.shape, dtype=np.float32)
    if positive.size == 0:
        return out
    low, high = np.percentile(positive, [1.0, 99.5])
    clipped = np.clip(image, low, high)
    valid = np.isfinite(clipped)
    centered = clipped - float(np.median(clipped[valid]))
    denom = max(float(np.percentile(np.abs(centered[valid]), 95.0)), 1.0)
    out[valid] = centered[valid] / denom
    return out


def corrcoef_on_mask(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 8:
        return None
    a = fixed[mask].astype(np.float64)
    b = moving[mask].astype(np.float64)
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
