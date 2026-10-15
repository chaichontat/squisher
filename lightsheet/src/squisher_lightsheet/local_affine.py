from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Literal, TypedDict

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import affine_transform

from squisher_lightsheet.channel_affine import (
    _block_mean_downsample_zyx_cupy,
    masked_structural_translation_gpu,
    output_to_input_from_model,
    output_to_input_to_model,
)


class PatchMeasurement(TypedDict):
    start_zyx: list[int]
    center_zyx: list[float]
    support_center_zyx: list[float] | None
    support_count: int
    support_fraction: float
    role: Literal["fit", "validation"]
    confident: bool
    shift_zyx: list[float] | None
    peak_ncc: float | None
    second_peak_ncc: float | None
    at_search_edge: bool | None


@dataclass(frozen=True)
class LocalAffineResult:
    matrix_zyx: np.ndarray
    translation_zyx: np.ndarray
    accepted: bool
    rejection_reason: str | None
    before: list[PatchMeasurement]
    after: list[PatchMeasurement]
    capture_spacing_um_zyx: np.ndarray
    estimator: Literal["least-squares", "restrained"] = "least-squares"
    diagnostics: dict[str, object] = field(default_factory=dict)


def affine_from_local_shifts(
    *,
    matrix_zyx: np.ndarray,
    translation_zyx: np.ndarray,
    shape_zyx: tuple[int, int, int],
    centers_zyx: np.ndarray,
    shifts_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an affine from equally weighted 3D image correspondences.

    A positive shift moves the registered image in the positive fixed direction.
    A fixed center therefore corresponds to the previous pull at center - shift.
    """
    centers = np.asarray(centers_zyx, dtype=np.float64)
    shifts = np.asarray(shifts_zyx, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or shifts.shape != centers.shape:
        raise ValueError("local centers and shifts must have matching (N,3) shapes")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(shifts)):
        raise ValueError("local image correspondences must be finite")
    design = np.column_stack((centers, np.ones(len(centers))))
    if np.linalg.matrix_rank(design) != 4:
        raise ValueError("local affine refinement requires noncoplanar 3D correspondences")
    pull_matrix, pull_offset = output_to_input_from_model(matrix_zyx, translation_zyx, shape_zyx)
    source_points = (centers - shifts) @ pull_matrix.T + pull_offset
    coefficients = np.linalg.lstsq(design, source_points, rcond=None)[0].T
    return output_to_input_to_model(coefficients[:, :3], coefficients[:, 3], shape_zyx)


def local_affine_cache_config(estimator: str) -> dict[str, object]:
    if estimator == "least-squares":
        return {}
    if estimator != "restrained":
        raise ValueError(f"unknown local affine estimator: {estimator}")
    return {
        "estimator": "restrained-v1",
        "local_affine_estimator": "restrained",
        "local_affine_policy_version": "restrained-v1",
        "strain_prior_sd": 0.01,
        "strain_penalty_factor": 0.5,
        "working_error_scale": "capture_spacing_native_z_2xy_not_calibrated_uncertainty",
        "huber_radius": 2.0,
        "minimum_fit_points": 6,
        "maximum_design_condition": 100.0,
        "maximum_irls_iterations": 30,
        "irls_tolerance": 1e-7,
        "center_policy": "actual_fixed_mask_support_centroid",
        "validation_policy": "opposite_parity_internal_consistency_unchanged",
    }


def affine_from_restrained_local_shifts(
    *,
    matrix_zyx: np.ndarray,
    translation_zyx: np.ndarray,
    shape_zyx: tuple[int, int, int],
    centers_zyx: np.ndarray,
    shifts_zyx: np.ndarray,
    spacing_um_zyx: np.ndarray,
    capture_spacing_um_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit a Huber shift field with a physical strain prior and free translation.

    Capture spacing is a working residual scale, not calibrated uncertainty.
    Huber rho(r) is r²/2 inside radius 2; the penalty is 0.5 ||G/0.01||².
    """
    centers = np.asarray(centers_zyx, dtype=np.float64)
    shifts = np.asarray(shifts_zyx, dtype=np.float64)
    spacing = np.asarray(spacing_um_zyx, dtype=np.float64)
    error_scale = np.asarray(capture_spacing_um_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or shifts.shape != centers.shape:
        raise ValueError("local centers and shifts must have matching (N,3) shapes")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(shifts)):
        raise ValueError("local image correspondences must be finite")
    for name, value in (("spacing", spacing), ("capture spacing", error_scale), ("shape", shape)):
        if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(f"restrained local affine requires positive finite {name}")
    if np.any(shape <= 1):
        raise ValueError("restrained local affine requires nonzero window spans")
    center = (shape - 1) / 2
    spans = (shape - 1) * spacing / 2
    design = np.column_stack(((centers - center) * spacing / spans, np.ones(len(centers))))
    if len(centers) < 6 or np.linalg.matrix_rank(design) != 4:
        raise ValueError("restrained local affine requires at least six noncoplanar points")
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > 100:
        raise ValueError("restrained local affine support exceeds design condition limit 100")
    target = shifts * spacing / error_scale
    # Coefficients predict standardized shifts. Converting back to physical
    # strain preserves the prior despite coordinate and residual scaling.
    penalties = [
        np.column_stack((np.diag(error_scale[axis] / (spans * 0.01)), np.zeros(3))) for axis in range(3)
    ]
    coefficients = np.zeros((4, 3))
    coefficients[3] = np.median(target, axis=0)
    for iteration in range(1, 31):
        residual_norm = np.linalg.norm(design @ coefficients - target, axis=1)
        weights = np.ones(len(centers))
        outside = residual_norm > 2
        weights[outside] = 2 / residual_norm[outside]
        root_weights = np.sqrt(weights)
        updated = np.column_stack(
            [
                np.linalg.lstsq(
                    np.vstack((design * root_weights[:, None], penalties[axis])),
                    np.concatenate((target[:, axis] * root_weights, np.zeros(3))),
                    rcond=None,
                )[0]
                for axis in range(3)
            ]
        )
        if not np.all(np.isfinite(updated)):
            raise ValueError("restrained local affine produced nonfinite coefficients")
        change = float(np.max(np.abs(design @ (updated - coefficients))))
        coefficients = updated
        if change <= 1e-7:
            break
    else:
        raise ValueError("restrained local affine IRLS did not converge in 30 iterations")
    residual_norm = np.linalg.norm(design @ coefficients - target, axis=1)
    weights = np.ones(len(centers))
    outside = residual_norm > 2
    weights[outside] = 2 / residual_norm[outside]
    strain = coefficients[:3].T * error_scale[:, None] / spans[None, :]
    physical_translation = coefficients[3] * error_scale
    shift_gradient = strain * spacing[None, :] / spacing[:, None]
    shift_translation = physical_translation / spacing
    pull, offset = output_to_input_from_model(matrix_zyx, translation_zyx, shape_zyx)
    if not np.all(np.isfinite(pull)) or not np.all(np.isfinite(offset)):
        raise ValueError("restrained local affine requires a finite initializer")
    new_pull = pull @ (np.eye(3) - shift_gradient)
    new_offset = offset + pull @ (shift_gradient @ center - shift_translation)
    matrix, translation = output_to_input_to_model(new_pull, new_offset, shape_zyx)
    return (
        matrix,
        translation,
        {
            "policy": local_affine_cache_config("restrained"),
            "normalized_design_condition": condition,
            "iterations": iteration,
            "converged": True,
            "maximum_standardized_prediction_change": change,
            "robust_weights": weights.tolist(),
            "standardized_residual_norms": residual_norm.tolist(),
            "strain_physical": strain.tolist(),
            "translation_physical": physical_translation.tolist(),
            "field_center_zyx": center.tolist(),
            "working_error_scale_um_zyx": error_scale.tolist(),
            "objective": float(
                np.sum(np.where(outside, 2 * residual_norm - 2, residual_norm**2 / 2))
                + 0.5 * np.sum((strain / 0.01) ** 2)
            ),
        },
    )


def refine_local_affine_gpu(
    fixed: cp.ndarray,
    moving: cp.ndarray,
    *,
    fixed_mask: cp.ndarray,
    initial_matrix: np.ndarray,
    initial_translation: np.ndarray,
    spacing_um_zyx: np.ndarray,
    patch_span_um: float = 40.0,
    max_shift_um: float = 6.0,
    estimator: Literal["least-squares", "restrained"] = "least-squares",
) -> LocalAffineResult:
    """Refine from a fixed 3x3x3 patch grid; reserve the opposite parity for QC.

    Geometry chooses patches before image matching. Every confident fit patch
    contributes one correspondence, independently of brightness or voxel count.
    Internal consistency uses opposite parity patch centers and requires residuals within one
    capture voxel on every axis. Unverifiable patches cannot certify a fit.
    """
    policy = local_affine_cache_config(estimator)
    spacing = np.asarray(spacing_um_zyx, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("local refinement requires positive finite ZYX spacing")
    if fixed.shape != moving.shape or fixed.shape != fixed_mask.shape or fixed.ndim != 3:
        raise ValueError("local refinement requires matching fixed/moving/mask ZYX shapes")
    if (
        not np.isfinite(patch_span_um)
        or patch_span_um <= 0
        or not np.isfinite(max_shift_um)
        or max_shift_um <= 0
    ):
        raise ValueError("local patch span and search range must be positive and finite")
    block = (1, 2, 2)
    patch_shape = np.maximum(np.rint(patch_span_um / spacing).astype(int), 16)
    patch_shape -= patch_shape % np.asarray(block)
    if np.any(patch_shape >= np.asarray(fixed.shape)):
        raise ValueError("registration window is too small for the local 3D patch grid")
    starts = [
        np.rint(np.linspace(0, size - patch, 3)).astype(int)
        for size, patch in zip(fixed.shape, patch_shape, strict=True)
    ]
    capture_spacing = spacing * np.asarray(block)
    bounds = tuple(int(np.ceil(max_shift_um / value)) for value in capture_spacing)
    if np.any(np.asarray(bounds) >= patch_shape / np.asarray(block) - 1):
        raise ValueError("local search range exceeds the image patch support")
    moving_support = cp.ones(moving.shape, dtype=cp.float32)
    plan = [
        (tuple(int(starts[a][index[a]]) for a in range(3)), "fit" if sum(index) % 2 == 0 else "validation")
        for index in product(range(3), repeat=3)
    ]

    def measure(matrix: np.ndarray, translation: np.ndarray) -> list[PatchMeasurement]:
        pull, offset = output_to_input_from_model(matrix, translation, fixed.shape)
        records: list[PatchMeasurement] = []
        for start, role in plan:
            region = tuple(
                slice(value, value + int(length)) for value, length in zip(start, patch_shape, strict=True)
            )
            mask = _block_mean_downsample_zyx_cupy(fixed_mask[region].astype(cp.float32), block) > 0.5
            native_mask = fixed_mask[region].astype(cp.bool_)
            support_count = int(cp.count_nonzero(native_mask))
            support_center = None
            if support_count:
                support_center = [
                    float(
                        start[axis]
                        + cp.sum(
                            native_mask.sum(axis=tuple(a for a in range(3) if a != axis))
                            * cp.arange(int(patch_shape[axis]))
                        )
                        / support_count
                    )
                    for axis in range(3)
                ]
            record: PatchMeasurement = {
                "start_zyx": list(start),
                "center_zyx": (np.asarray(start) + (patch_shape - 1) / 2).tolist(),
                "support_center_zyx": support_center,
                "support_count": support_count,
                "support_fraction": support_count / int(np.prod(patch_shape)),
                "role": role,
                "confident": False,
                "shift_zyx": None,
                "peak_ncc": None,
                "second_peak_ncc": None,
                "at_search_edge": None,
            }
            if int(cp.count_nonzero(mask)) < 256:
                records.append(record)
                continue
            sample_offset = pull @ np.asarray(start) + offset
            sample_kwargs = {
                "matrix": cp.asarray(pull, dtype=cp.float32),
                "offset": cp.asarray(sample_offset, dtype=cp.float32),
                "output_shape": tuple(patch_shape),
                "order": 1,
                "mode": "constant",
                "cval": 0.0,
                "prefilter": False,
            }
            sample = affine_transform(moving, **sample_kwargs)
            support = affine_transform(moving_support, **sample_kwargs)
            result = masked_structural_translation_gpu(
                _block_mean_downsample_zyx_cupy(fixed[region], block),
                _block_mean_downsample_zyx_cupy(sample, block),
                fixed_mask=mask,
                moving_support=_block_mean_downsample_zyx_cupy(support, block) > 0.99,
                max_shift_zyx=bounds,
                highpass_sigma_zyx=tuple(float(4.6 / value) for value in capture_spacing),
                exclusion_radius_zyx=tuple(int(np.ceil(3.5 / value)) for value in capture_spacing),
                overlap_ratio=0.3,
                subpixel=True,
            )
            record.update(
                confident=not result.at_search_edge
                and result.peak_ncc >= 0.20
                and result.peak_ncc - result.second_peak_ncc >= 0.10,
                shift_zyx=(np.asarray(result.shift_zyx) * np.asarray(block)).tolist(),
                peak_ncc=float(result.peak_ncc),
                second_peak_ncc=float(result.second_peak_ncc),
                at_search_edge=bool(result.at_search_edge),
            )
            records.append(record)
        return records

    before = measure(initial_matrix, initial_translation)
    fitting = [row for row in before if row["role"] == "fit" and row["confident"]]
    validation = [i for i, row in enumerate(before) if row["role"] == "validation" and row["confident"]]
    center_key = "support_center_zyx" if estimator == "restrained" else "center_zyx"
    fit_centers = np.asarray([row[center_key] for row in fitting], dtype=np.float64)
    fit_design = np.column_stack((fit_centers, np.ones(len(fitting)))) if fitting else np.empty((0, 4))
    validation_design = np.asarray([before[i]["center_zyx"] + [1.0] for i in validation])
    if (
        len(fitting) < (6 if estimator == "restrained" else 4)
        or len(validation) < 4
        or np.linalg.matrix_rank(fit_design) < 4
        or np.linalg.matrix_rank(validation_design) < 4
    ):
        return LocalAffineResult(
            initial_matrix,
            initial_translation,
            False,
            "insufficient_local_3d_support",
            before,
            [],
            capture_spacing,
            estimator,
            {"policy": policy},
        )
    diagnostics: dict[str, object] = {"policy": policy}
    fit_kwargs = dict(
        matrix_zyx=initial_matrix,
        translation_zyx=initial_translation,
        shape_zyx=fixed.shape,
        centers_zyx=fit_centers,
        shifts_zyx=np.asarray([row["shift_zyx"] for row in fitting], dtype=np.float64),
    )
    if estimator == "restrained":
        # Unsupported geometry is an unresolved window; solver failures fail fast.
        normalized = np.column_stack(
            (
                (fit_centers - (np.asarray(fixed.shape) - 1) / 2) / ((np.asarray(fixed.shape) - 1) / 2),
                np.ones(len(fitting)),
            )
        )
        condition = float(np.linalg.cond(normalized))
        if not np.isfinite(condition) or condition > 100:
            return LocalAffineResult(
                initial_matrix,
                initial_translation,
                False,
                "ill_conditioned_local_3d_support",
                before,
                [],
                capture_spacing,
                estimator,
                {"policy": policy, "normalized_design_condition": condition},
            )
        matrix, translation, diagnostics = affine_from_restrained_local_shifts(
            **fit_kwargs, spacing_um_zyx=spacing, capture_spacing_um_zyx=capture_spacing
        )
    else:
        matrix, translation = affine_from_local_shifts(**fit_kwargs)
    after = measure(matrix, translation)
    verified = all(
        after[i]["confident"] and np.all(np.abs(np.asarray(after[i]["shift_zyx"])) <= np.asarray(block))
        for i in validation
    )
    return LocalAffineResult(
        matrix,
        translation,
        verified,
        None if verified else "local_3d_validation_residual",
        before,
        after,
        capture_spacing,
        estimator,
        diagnostics,
    )
