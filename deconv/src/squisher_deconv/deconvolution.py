from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import tifffile


class Deconvolver(Protocol):
    @property
    def halo(self) -> int: ...

    def deconvolve(self, volume: np.ndarray) -> np.ndarray: ...


def infer_psf_halo(path: Path) -> int:
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3-D PSF at {path}, got shape {shape}")
    z_size = int(shape[0])
    if z_size < 1:
        raise ValueError(f"PSF {path} must have at least one z plane, got shape {shape}")
    return int((z_size - 1) * 2)


def infer_psf_halo_many(paths: Sequence[Path]) -> int:
    if not paths:
        raise ValueError("At least one PSF path is required.")
    return max(infer_psf_halo(path) for path in paths)


@dataclass(frozen=True, slots=True)
class IdentityDeconvolver:
    """Contract-test deconvolver that preserves slab values exactly."""

    halo: int = 0

    def deconvolve(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim != 4:
            raise ValueError(f"Expected (Z, C, Y, X) volume, got {volume.shape}")
        return volume.astype(np.float32, copy=True)
