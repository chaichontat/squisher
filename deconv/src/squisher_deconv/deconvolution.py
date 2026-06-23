from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import tifffile
from scipy.ndimage import convolve


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


@dataclass(frozen=True, slots=True)
class IdentityDeconvolver:
    """Contract-test deconvolver that preserves slab values exactly."""

    halo: int = 0

    def deconvolve(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim != 4:
            raise ValueError(f"Expected (Z, C, Y, X) volume, got {volume.shape}")
        return volume.astype(np.float32, copy=True)


@dataclass(frozen=True, slots=True)
class ScipyRichardsonLucyDeconvolver:
    """Small CPU implementation of the Guo-style single-iteration LR update.

    This keeps the package runnable without a full-dataset staging pass. The
    deconvolver is injected into the streaming code, so a CuPy implementation can
    use the same source, planning, scaling, and OME sink contracts.
    """

    forward_projector: np.ndarray
    backward_projector: np.ndarray

    @classmethod
    def from_psf(cls, path: Path) -> "ScipyRichardsonLucyDeconvolver":
        psf = tifffile.imread(path).astype(np.float32, copy=False)
        if psf.ndim != 3:
            raise ValueError(f"Expected 3-D PSF at {path}, got shape {psf.shape}")
        psf = psf[::-1]
        psf_sum = float(psf.sum())
        if psf_sum <= 0:
            raise ValueError(f"PSF {path} has non-positive sum.")
        psf = psf / np.float32(psf_sum)
        return cls(forward_projector=psf[:, None, :, :], backward_projector=psf[::-1, ::-1, ::-1][:, None, :, :])

    @property
    def halo(self) -> int:
        return int((self.forward_projector.shape[0] - 1) + (self.backward_projector.shape[0] - 1))

    def deconvolve(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim != 4:
            raise ValueError(f"Expected (Z, C, Y, X) volume, got {volume.shape}")
        img = volume.astype(np.float32, copy=True)
        np.clip(img, np.float32(1e-9), None, out=img)
        estimate = img.copy()
        filtered = convolve(estimate, self.forward_projector, mode="reflect")
        np.clip(filtered, np.float32(1e-9), np.float32(65535.0), out=filtered)
        ratio = img / filtered
        correction = convolve(ratio, self.backward_projector, mode="reflect")
        estimate *= correction
        return estimate.astype(np.float32, copy=False)
