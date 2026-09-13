"""Smooth residual gain in normalized original camera Z/Y/X coordinates."""

from __future__ import annotations

import numpy as np


def validate_coefficients(value: object) -> np.ndarray:
    coefficient = np.asarray(value, dtype=np.float64)
    if (
        coefficient.ndim != 2
        or coefficient.shape[1] != 5
        or not 1 <= coefficient.shape[0] <= 4
        or not np.all(np.isfinite(coefficient))
    ):
        raise ValueError("Residual coefficient must be a finite (Z degree + 1, 5) array with Z degree 0–3")
    return coefficient


def cosine_basis(yx: np.ndarray, *, z: np.ndarray | None = None, z_degree: int = 0) -> np.ndarray:
    """Use five shared XY modes and shrink higher Z modes by 1 / (1 + degree²)."""
    if not 0 <= z_degree <= 3:
        raise ValueError("Z degree must be between 0 and 3")
    spatial = np.stack(
        [
            np.cos(np.pi * y_degree * yx[:, 0]) * np.cos(np.pi * x_degree * yx[:, 1])
            for y_degree in range(3)
            for x_degree in range(3 - y_degree)
            if y_degree + x_degree
        ],
        axis=1,
    )
    if z_degree == 0:
        return spatial
    if z is None or np.asarray(z).shape != (len(yx),) or not np.all(np.isfinite(z)):
        raise ValueError("Z coordinates must be finite and aligned with XY samples")
    return np.concatenate(
        [spatial * np.cos(np.pi * degree * z)[:, None] / (1 + degree**2) for degree in range(z_degree + 1)],
        axis=1,
    )


def spatial_log_fields(
    coefficient: np.ndarray,
    shape_yx: tuple[int, int],
    *,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> np.ndarray:
    coefficient = validate_coefficients(coefficient)
    height, width = shape_yx
    if height < 2 or width < 2:
        raise ValueError("Residual fields require at least two pixels on each camera XY axis")
    yy, xx = np.meshgrid(np.linspace(0, 1, height)[y_slice], np.linspace(0, 1, width)[x_slice], indexing="ij")
    basis = cosine_basis(np.column_stack([yy.ravel(), xx.ravel()]))
    return (coefficient @ basis.T).reshape(len(coefficient), *yy.shape)


def residual_plane(
    coefficient: np.ndarray,
    *,
    z: int,
    shape_zyx: tuple[int, int, int],
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> np.ndarray:
    """Evaluate a crop using original raw coordinates, never crop-local coordinates."""
    if not 0 <= z < shape_zyx[0]:
        raise ValueError(f"Invalid raw Z {z} for source shape {shape_zyx}")
    return residual_block(
        coefficient,
        z_slice=slice(z, z + 1),
        shape_zyx=shape_zyx,
        y_slice=y_slice,
        x_slice=x_slice,
    )[0]


def residual_block(
    coefficient: np.ndarray,
    *,
    z_slice: slice,
    shape_zyx: tuple[int, int, int],
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> np.ndarray:
    """Evaluate a requested slab in original source coordinates."""
    coefficient = validate_coefficients(coefficient)
    z_indices = np.arange(shape_zyx[0])[z_slice]
    if z_indices.size == 0:
        raise ValueError(f"Residual block has an empty Z slice for source shape {shape_zyx}")
    maps = spatial_log_fields(coefficient, shape_zyx[1:], y_slice=y_slice, x_slice=x_slice)
    degree = np.arange(len(maps))
    weights = np.cos(
        np.pi * z_indices[:, None] * degree[None, :] / max(shape_zyx[0] - 1, 1)
    ) / (1 + degree[None, :] ** 2)
    field = np.exp(np.einsum("zd,dyx->zyx", weights, maps)).astype(np.float32)
    if not np.all(np.isfinite(field)) or np.any(field <= 0):
        raise ValueError("Residual gain is not positive and finite")
    return field
