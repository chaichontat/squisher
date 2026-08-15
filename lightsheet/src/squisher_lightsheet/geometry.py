from __future__ import annotations

import numpy as np


def signed_bounds(
    translation: np.ndarray, scale: np.ndarray, shape: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical low/high bounds for an index-zero translation and signed scale."""
    start = np.asarray(translation, dtype=np.float64)
    stop = start + np.asarray(shape, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    return np.minimum(start, stop), np.maximum(start, stop)


def orient_plane_yx(plane: np.ndarray, scale_yx: np.ndarray | tuple[float, float]) -> np.ndarray:
    oriented = plane
    for axis, scale in enumerate(np.asarray(scale_yx, dtype=np.float64)):
        if scale < 0:
            oriented = np.flip(oriented, axis=axis)
    return oriented
