"""Load and evaluate a correction field defined on the registered output grid."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates

from squisher_lightsheet.post_basic import _fixed_grid, _sha256


@dataclass(frozen=True)
class RegisteredViewCorrection:
    moving_view: str
    origin_zyx_um: tuple[float, float, float]
    spacing_zyx_um: tuple[float, float, float]
    intercept_log_ratio: float
    deviation_log_ratio: np.ndarray
    common_scale: float
    registration_sha256: str | None = None

    def log_ratio_block(
        self,
        *,
        origin_zyx_um: tuple[float, float, float],
        spacing_zyx_um: tuple[float, float, float],
        shape_zyx: tuple[int, int, int],
        xp=np,
    ):
        matrix = np.diag(np.asarray(spacing_zyx_um) / np.asarray(self.spacing_zyx_um))
        offset = (np.asarray(origin_zyx_um) - np.asarray(self.origin_zyx_um)) / np.asarray(
            self.spacing_zyx_um
        )
        if xp is np:
            deviation = affine_transform(
                self.deviation_log_ratio,
                matrix=matrix,
                offset=offset,
                output_shape=shape_zyx,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        else:
            from cupyx.scipy.ndimage import affine_transform as cupy_affine_transform

            deviation = cupy_affine_transform(
                xp.asarray(self.deviation_log_ratio),
                matrix=xp.asarray(matrix),
                offset=xp.asarray(offset),
                output_shape=shape_zyx,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        return deviation + self.intercept_log_ratio


@lru_cache(maxsize=8)
def load_registered_view_correction(path: Path) -> RegisteredViewCorrection:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if (
        payload.get("schema_version") not in (1, 2)
        or payload.get("kind") != "squisher.registered-view-correction"
        or payload.get("coordinate") != "registered-physical-zyx-um"
    ):
        raise ValueError(f"Invalid registered-view correction artifact: {path}")
    moving_view = payload.get("moving_view")
    if not isinstance(moving_view, str) or not moving_view.strip():
        raise ValueError("Registered-view correction requires a moving_view")
    origin = _finite_triplet(payload.get("origin_zyx_um"), "origin_zyx_um")
    spacing = _finite_triplet(payload.get("spacing_zyx_um"), "spacing_zyx_um")
    if any(value <= 0 for value in spacing):
        raise ValueError("Registered-view correction spacing must be positive")
    intercept = payload.get("intercept_log_ratio")
    common_scale = payload.get("common_scale")
    if type(intercept) not in (int, float) or not np.isfinite(intercept):
        raise ValueError("Registered-view correction intercept must be finite")
    if type(common_scale) not in (int, float) or not 0 < common_scale <= 1:
        raise ValueError("Registered-view correction common_scale must be within (0, 1]")
    deviation = np.asarray(payload.get("deviation_log_ratio"), dtype=np.float32)
    if deviation.ndim != 3 or min(deviation.shape) < 2 or not np.isfinite(deviation).all():
        raise ValueError("Registered-view correction deviation must be a finite 3D grid")
    declared_shape = payload.get("shape_zyx")
    if declared_shape != list(deviation.shape):
        raise ValueError("Registered-view correction shape does not match its deviation grid")
    registration_sha256 = payload.get("registration_sha256")
    if payload["schema_version"] == 2 and (
        not isinstance(registration_sha256, str) or len(registration_sha256) != 64
    ):
        raise ValueError("Registered-view correction requires its registration SHA-256")
    return RegisteredViewCorrection(
        moving_view=moving_view,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
        intercept_log_ratio=float(intercept),
        deviation_log_ratio=deviation,
        common_scale=float(common_scale),
        registration_sha256=registration_sha256,
    )


def _finite_triplet(value, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"Registered-view correction {name} must contain three finite values")
    return tuple(float(item) for item in array)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0, 1)
    return clipped * clipped * (3 - 2 * clipped)


def _fit_grid(
    points: np.ndarray,
    ratios: np.ndarray,
    pair_ids: np.ndarray,
    *,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: tuple[int, int, int],
    intercept: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.rint((points - origin) / spacing).astype(np.int64)
    inside = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    indices, ratios, pair_ids = indices[inside], ratios[inside], pair_ids[inside]
    groups: dict[tuple[int, int, int], list[int]] = {}
    for row, index in enumerate(indices):
        groups.setdefault(tuple(int(value) for value in index), []).append(row)
    values = np.zeros(shape, dtype=np.float32)
    confidence = np.zeros(shape, dtype=np.float32)
    for index, rows in groups.items():
        selected = ratios[rows]
        independent_pairs = len(set(pair_ids[rows]))
        if len(rows) < 3 or independent_pairs < 2:
            continue
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        sign_agreement = max(float(np.mean(selected >= 0)), float(np.mean(selected <= 0)))
        values[index] = median - intercept
        confidence[index] = (
            min(1.0, len(rows) / 8)
            * min(1.0, independent_pairs / 3)
            * _smoothstep(np.asarray((sign_agreement - 0.5) / 0.35)).item()
            * float(np.exp(-((mad / 0.18) ** 2)))
        )
    sigma = np.asarray((140.0, 260.0, 260.0)) / spacing
    smoothed_weight = gaussian_filter(confidence, sigma=sigma, mode="constant")
    smoothed_value = gaussian_filter(values * confidence, sigma=sigma, mode="constant")
    fitted = np.divide(
        smoothed_value,
        smoothed_weight,
        out=np.zeros_like(smoothed_value),
        where=smoothed_weight > 1e-6,
    )
    positive = smoothed_weight[smoothed_weight > 0]
    support_scale = float(np.percentile(positive, 45)) if positive.size else 1.0
    support = _smoothstep(smoothed_weight / max(support_scale, 1e-6)).astype(np.float32)
    return (fitted * support).astype(np.float32), support


def _evaluate_grid_at_points(
    deviation: np.ndarray,
    points: np.ndarray,
    *,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    coordinates = ((points - origin) / spacing).T
    return map_coordinates(
        deviation,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _spatial_cross_validation(
    points: np.ndarray,
    ratios: np.ndarray,
    pair_ids: np.ndarray,
    *,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: tuple[int, int, int],
    folds: int = 5,
) -> dict[str, object]:
    block = np.floor((points - origin) / (2 * spacing)).astype(np.int64)
    unique_blocks, assignment = np.unique(block, axis=0, return_inverse=True)
    folds = min(folds, len(unique_blocks))
    before = ratios / np.log(2)
    if folds < 2:
        return {
            "folds": 0,
            "block_size_zyx_um": (2 * spacing).tolist(),
            "reason": "fewer than two occupied spatial blocks",
            "before_median_abs_log2": float(np.median(np.abs(before))),
            "after_median_abs_log2": None,
            "before_p95_abs_log2": float(np.percentile(np.abs(before), 95)),
            "after_p95_abs_log2": None,
        }
    assignment %= folds
    residuals = np.empty_like(ratios)
    for fold in range(folds):
        train = assignment != fold
        validate = ~train
        intercept = float(np.median(ratios[train]))
        deviation, _support = _fit_grid(
            points[train],
            ratios[train],
            pair_ids[train],
            origin=origin,
            spacing=spacing,
            shape=shape,
            intercept=intercept,
        )
        predicted = intercept + _evaluate_grid_at_points(
            deviation, points[validate], origin=origin, spacing=spacing
        )
        residuals[validate] = ratios[validate] - predicted
    after = residuals / np.log(2)
    return {
        "folds": folds,
        "block_size_zyx_um": (2 * spacing).tolist(),
        "before_median_abs_log2": float(np.median(np.abs(before))),
        "after_median_abs_log2": float(np.median(np.abs(after))),
        "before_p95_abs_log2": float(np.percentile(np.abs(before), 95)),
        "after_p95_abs_log2": float(np.percentile(np.abs(after), 95)),
    }


def build_registered_view_correction(
    *,
    rows: Sequence[Mapping[str, object]],
    moving_view: str,
    fixed_fused: Path,
    registration: Path,
    grid_spacing_zyx_um: tuple[float, float, float] = (100.0, 200.0, 200.0),
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Build a registered correction payload from cross-view patch measurements."""
    spacing = np.asarray(grid_spacing_zyx_um, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Registered-view correction grid spacing must contain three positive values")
    if not rows or not moving_view.strip():
        raise ValueError("Spatial measurements and a moving view are required")

    points, ratios, pair_ids = [], [], []
    for row in rows:
        center = row.get("center_zyx_um")
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError("Match artifact predates registered spatial patch measurements")
        ratio = float(row["log_ratio_second_over_first"])
        if row["second_view"] != moving_view:
            ratio = -ratio
        points.append(center)
        ratios.append(ratio)
        pair_ids.append(f"{row['first_source']}|{row['second_source']}")
    points_array = np.asarray(points, dtype=np.float64)
    ratios_array = np.asarray(ratios, dtype=np.float64)
    if not np.isfinite(points_array).all() or not np.isfinite(ratios_array).all():
        raise ValueError("Spatial measurements must be finite")
    pair_array = np.asarray(pair_ids)
    intercept = float(np.median(ratios_array))

    fixed_fused, registration = fixed_fused.resolve(), registration.resolve()
    fixed_grid = _fixed_grid(fixed_fused, fixed_z=None, stride=1)
    origin = fixed_grid.origin.astype(np.float64)
    extent = (fixed_grid.shape - 1) * fixed_grid.scale
    shape = tuple(int(value) for value in np.ceil(extent / spacing).astype(int) + 1)
    cross_validation = _spatial_cross_validation(
        points_array,
        ratios_array,
        pair_array,
        origin=origin,
        spacing=spacing,
        shape=shape,
    )
    deviation, support = _fit_grid(
        points_array,
        ratios_array,
        pair_array,
        origin=origin,
        spacing=spacing,
        shape=shape,
        intercept=intercept,
    )
    fitted = intercept + _evaluate_grid_at_points(deviation, points_array, origin=origin, spacing=spacing)
    before = ratios_array / np.log(2)
    after = (ratios_array - fitted) / np.log(2)
    minimum_field = float(min(intercept, np.min(intercept + deviation)))
    common_scale = float(min(1.0, np.exp(minimum_field)))
    payload = {
        "schema_version": 2,
        "kind": "squisher.registered-view-correction",
        "coordinate": "registered-physical-zyx-um",
        "moving_view": moving_view,
        "registration": str(registration),
        "registration_sha256": _sha256(registration),
        "fixed_fused": str(fixed_fused),
        "origin_zyx_um": origin.tolist(),
        "spacing_zyx_um": spacing.tolist(),
        "shape_zyx": list(shape),
        "intercept_log_ratio": intercept,
        "deviation_log_ratio": deviation.tolist(),
        "support_weight": support.tolist(),
        "common_scale": common_scale,
        "fit": {
            "method": "confidence-weighted robust spatial bins with normalized Gaussian smoothing",
            "measurements": len(rows),
            "outside_support": "intercept_log_ratio",
            "spatial_cross_validation": cross_validation,
            "before_median_abs_log2": float(np.median(np.abs(before))),
            "after_median_abs_log2": float(np.median(np.abs(after))),
            "before_p95_abs_log2": float(np.percentile(np.abs(before), 95)),
            "after_p95_abs_log2": float(np.percentile(np.abs(after), 95)),
        },
    }
    return payload, deviation, support
