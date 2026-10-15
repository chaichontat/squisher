"""Recover fused-fixed registration windows from spatial affine consensus."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
from scipy.linalg import polar
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


from squisher_lightsheet import ngff
from squisher_lightsheet._legacy.rough_align_tltr_center_z_phase import tile_bounds_zyx_um
from squisher_lightsheet.fused_fixed import DEFAULT_SWEEP_RUNNER, SUMMARY_NAME
from squisher_lightsheet.tile_phase import tile_record_from_position_record


SUPPORTED_NATIVE_METHODS = frozenset(("method6", "method8", "method10-mattes", "method11-mattes"))
MASK_REJECTION_REASONS = {
    "fixed_threshold_mask_empty",
    "fixed_threshold_fit_mask_empty",
    "fixed_threshold_mask_too_masked",
    "fixed_threshold_fit_mask_too_masked",
    "empty_fixed_mask",
    "empty_fixed_crop",
    "empty_moving_crop",
}
DEFAULT_OUTLIER_MAD = 4.0
DEFAULT_MAXIMUM_OUTLIER_PX = 5.0
DEFAULT_MAXIMUM_REFIT_DISPLACEMENT_PX = 10.0
DEFAULT_MINIMUM_LINEAR_SAMPLES = 8
DEFAULT_MINIMUM_SMOOTHING_SAMPLES = 4
DEFAULT_LINEAR_LOSS_SCALE_UM = 2.0
DEFAULT_MINIMUM_LINEAR_OUTLIER_UM = 3.0
DEFAULT_MAXIMUM_LINEAR_OUTLIER_UM = 8.0
DEFAULT_MAXIMUM_LINEAR_MEDIAN_RESIDUAL_UM = 4.0
DEFAULT_REPLACEMENT_IMAGE_NCC_MARGIN = 0.02
DEFAULT_NATIVE_DEVICES = (0, 1)
DEFAULT_NATIVE_DEVICE_SPEC = ",".join(map(str, DEFAULT_NATIVE_DEVICES))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _tile_number(value: str) -> str:
    for part in reversed(str(value).split(".")):
        if part.isdigit():
            return part.zfill(3)
    raise ValueError(f"Could not parse tile number from {value!r}")


def _native_method_from_summary(summary: dict[str, Any]) -> str:
    method = summary["cache_config"]["native_method"]
    if method not in SUPPORTED_NATIVE_METHODS:
        raise ValueError(f"Recovery summary has unsupported native method {method!r}")
    return method


def _native_attempt_prefix(native_method: str) -> str:
    if native_method not in SUPPORTED_NATIVE_METHODS:
        raise ValueError(f"Unsupported native recovery method {native_method!r}")
    return f"native_{native_method.replace('-', '_')}_from_"


def _is_local_3d_native_fit(row: dict[str, Any], native_method: str) -> bool:
    cache = row.get("cache_config", {})
    return (
        row.get("selected_attempt") == "local_3d_affine_refinement"
        and cache.get("native_method") == native_method
        and cache.get("local_3d_refinement") is True
    )


def _native_recovery_attempt(native_method: str) -> str:
    return f"{_native_attempt_prefix(native_method)}recovery_initializer"


def _interpolated_recovery_attempt(native_method: str) -> str:
    return f"interpolated_{native_method.replace('-', '_')}_spatial_recovery"


def _linear_outlier_replaced_attempt(native_method: str) -> str:
    return f"linear_interpolated_{native_method.replace('-', '_')}_outlier_replacement"


def decomposed_affine_mean(matrices: np.ndarray) -> np.ndarray:
    """Average affine rotation on SO(3) and symmetric stretch in polar space."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not len(matrices):
        raise ValueError("matrices must have shape (n, 3, 3) with n >= 1")
    rotations = []
    stretches = []
    for matrix in matrices:
        if not np.all(np.isfinite(matrix)) or np.linalg.det(matrix) <= 0:
            raise ValueError("affine mean requires finite orientation-preserving matrices")
        rotation, stretch = polar(matrix)
        if np.linalg.det(rotation) <= 0:
            raise ValueError("affine polar decomposition produced a reflection")
        rotations.append(rotation)
        stretches.append(stretch)
    mean_rotation = Rotation.from_matrix(np.stack(rotations)).mean().as_matrix()
    mean_stretch = np.mean(np.stack(stretches), axis=0)
    result = mean_rotation @ mean_stretch
    if not np.all(np.isfinite(result)) or np.linalg.det(result) <= 0:
        raise ValueError("decomposed affine mean is not orientation preserving")
    return result


def _transform(row: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(row["selected_local_matrix_zyx"], dtype=np.float64)
    translation = np.asarray(row["selected_local_translation_zyx"], dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or translation.shape != (3,)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(translation))
    ):
        raise ValueError(
            f"Invalid selected transform for {row.get('moving_tile')} {row.get('moving_start_l0_zyx')}"
        )
    return np.concatenate([matrix.ravel(), translation])


def _point(row: dict[str, Any]) -> np.ndarray:
    point = np.asarray(row["moving_start_l0_zyx"], dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"Invalid moving_start_l0_zyx for {row.get('moving_tile')}")
    return point


def _normalized_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.min(points, axis=0)
    spacing = np.ones(3, dtype=np.float64)
    for axis in range(3):
        values = np.unique(points[:, axis])
        differences = np.diff(values)
        if differences.size:
            spacing[axis] = float(np.median(differences[differences > 0]))
    return (points - origin) / spacing, origin, spacing


def _corner_displacement_px(first: np.ndarray, second: np.ndarray, shape_zyx: np.ndarray) -> float:
    centered_extent = (np.asarray(shape_zyx, dtype=np.float64) - 1.0) / 2.0
    points = np.asarray(
        [
            [z, y, x]
            for z in (-centered_extent[0], centered_extent[0])
            for y in (-centered_extent[1], centered_extent[1])
            for x in (-centered_extent[2], centered_extent[2])
        ]
        + [[0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    matrix_delta = first[:9].reshape(3, 3) - second[:9].reshape(3, 3)
    translation_delta = first[9:] - second[9:]
    displacement = points @ matrix_delta.T + translation_delta
    return float(np.max(np.linalg.norm(displacement, axis=1)))


def _score_summary(scores: np.ndarray) -> dict[str, float | None]:
    finite = scores[np.isfinite(scores)]
    if not len(finite):
        return {"median": None, "maximum": None}
    return {"median": float(np.median(finite)), "maximum": float(np.max(finite))}


def _aggregate_with_status_counts(summary: dict[str, Any]) -> dict[str, Any]:
    windows = summary.get("windows", [])
    accepted = sum(row.get("status") == "accepted" for row in windows)
    rejected = sum(row.get("status") == "rejected" for row in windows)
    error = sum(row.get("status") == "error" for row in windows)
    return {
        **summary.get("aggregate", {}),
        "window_count": len(windows),
        "accepted_window_count": accepted,
        "rejected_window_count": rejected,
        "error_window_count": error,
        "accepted": accepted,
        "rejected": rejected,
        "error": error,
    }


def _compact_window_record(
    compact: dict[str, Any], row: dict[str, Any], path: Path
) -> dict[str, Any]:
    result = deepcopy(compact)
    result.update(
        {
            "level0_json": str(path.resolve()),
            "status": row.get("status"),
            "rejection_reason": row.get("rejection_reason"),
            "selected_attempt": row.get("selected_attempt"),
            "selected_corr": row.get("selected_corr_refined"),
            "selected_grad_ncc": row.get("selected_gradient_component_ncc_mean"),
        }
    )
    return result


def _write_stage_artifacts(
    *,
    source_summary: dict[str, Any],
    windows: list[dict[str, Any]],
    input_summary: Path,
    output_dir: Path,
    report_key: str,
    report_name: str,
    report: dict[str, Any],
    write_json: Callable[[Path, dict[str, Any]], None] = _write_json,
) -> Path:
    output_summary = deepcopy(source_summary)
    output_summary.update(
        {
            "output_dir": str(output_dir.resolve()),
            "input_summary": str(input_summary.resolve()),
            "windows": windows,
            report_key: report,
        }
    )
    output_summary["aggregate"] = _aggregate_with_status_counts(output_summary)
    write_json(output_dir / report_name, report)
    summary_output = output_dir / SUMMARY_NAME
    write_json(summary_output, output_summary)
    return summary_output


def _varying_coordinate_axes(points: np.ndarray) -> np.ndarray:
    """Return axes with coordinate variation beyond floating-point roundoff."""
    spans = np.ptp(points, axis=0)
    scale = np.maximum(np.max(np.abs(points), axis=0), 1.0)
    tolerance = 16.0 * np.finfo(np.float64).eps * scale
    return spans > tolerance


def detect_linear_displacement_outliers(
    points: np.ndarray,
    displacements: np.ndarray,
    *,
    outlier_mad: float,
    minimum_outlier_um: float,
    loss_scale_um: float,
    maximum_outlier_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a robust physical displacement plane and flag windows that depart from it."""
    points = np.asarray(points, dtype=np.float64)
    displacements = np.asarray(displacements, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or displacements.shape != points.shape:
        raise ValueError("points and displacements must both have shape (n, 3)")
    center = np.mean(points, axis=0)
    spread = np.std(points, axis=0)
    active_axes = _varying_coordinate_axes(points)
    design = np.column_stack(
        [np.ones(len(points)), (points[:, active_axes] - center[active_axes]) / spread[active_axes]]
    )
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("linear displacement fit does not have full spatial rank")

    initial = np.zeros((design.shape[1], 3), dtype=np.float64)
    initial[0] = np.median(displacements, axis=0)
    fit = least_squares(
        lambda coefficients: (design @ coefficients.reshape(design.shape[1], 3) - displacements).ravel(),
        initial.ravel(),
        loss="cauchy",
        f_scale=float(loss_scale_um),
    )
    predicted = design @ fit.x.reshape(design.shape[1], 3)
    scores = np.linalg.norm(displacements - predicted, axis=1)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    threshold = max(float(minimum_outlier_um), median + float(outlier_mad) * 1.4826 * mad)
    if maximum_outlier_um is not None:
        threshold = min(threshold, float(maximum_outlier_um))
    return scores > threshold, scores, threshold, median


def fit_linear_field(
    points: np.ndarray,
    values: np.ndarray,
    *,
    target_points: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit each value as a global affine function of the spatial coordinates."""
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or values.ndim != 2 or len(values) != len(points):
        raise ValueError("points must have shape (n, 3) and values must have shape (n, m)")
    targets = points if target_points is None else np.asarray(target_points, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target_points must have shape (n, 3)")
    if not all(np.all(np.isfinite(array)) for array in (points, values, targets)):
        raise ValueError("linear field inputs must be finite")

    center = np.mean(points, axis=0)
    spread = np.std(points, axis=0)
    varying_axes = np.flatnonzero(_varying_coordinate_axes(points))
    normalized = np.zeros_like(points)
    normalized[:, varying_axes] = (points[:, varying_axes] - center[varying_axes]) / spread[varying_axes]
    active_axes: list[int] = []
    rank = 1
    for axis in varying_axes:
        candidate = np.column_stack([np.ones(len(points)), normalized[:, [*active_axes, int(axis)]]])
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank > rank:
            active_axes.append(int(axis))
            rank = candidate_rank
    design = np.column_stack(
        [np.ones(len(points)), (points[:, active_axes] - center[active_axes]) / spread[active_axes]]
    )
    coefficients, _residuals, fitted_rank, singular_values = np.linalg.lstsq(design, values, rcond=None)
    if fitted_rank != design.shape[1]:
        raise ValueError(f"linear field requires rank {design.shape[1]}, got {fitted_rank}")
    target_design = np.column_stack(
        [np.ones(len(targets)), (targets[:, active_axes] - center[active_axes]) / spread[active_axes]]
    )
    return target_design @ coefficients, {
        "coordinate_center_zyx": center.tolist(),
        "coordinate_spread_zyx": spread.tolist(),
        "modeled_spatial_axes": ["zyx"[axis] for axis in active_axes],
        "constant_spatial_axes": ["zyx"[axis] for axis in range(3) if axis not in varying_axes],
        "dependent_spatial_axes": ["zyx"[axis] for axis in varying_axes if axis not in active_axes],
        "design_rank": int(fitted_rank),
        "design_singular_values": singular_values.tolist(),
        "coefficients": coefficients.tolist(),
    }


def predict_decomposed_affine_field(
    points: np.ndarray,
    transforms: np.ndarray,
    *,
    target_points: np.ndarray,
) -> np.ndarray:
    """Interpolate translation while averaging affine geometry in polar space."""
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 2 or transforms.shape[1] != 12 or not len(transforms):
        raise ValueError("transforms must have shape (n, 12) with n >= 1")
    translations = fit_linear_field(
        points,
        transforms[:, 9:],
        target_points=target_points,
    )[0]
    matrix = decomposed_affine_mean(transforms[:, :9].reshape(-1, 3, 3))
    matrices = np.repeat(matrix.reshape(1, 9), len(translations), axis=0)
    return np.concatenate([matrices, translations], axis=1)


def outlier_only_linear_predictions(
    *,
    points: np.ndarray,
    transforms: np.ndarray,
    values: np.ndarray,
    shape_zyx: np.ndarray,
    maximum_outlier_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None, np.ndarray, float]:
    """Predict field values only where the measured affine exceeds the pixel tolerance."""
    points = np.asarray(points, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(points) != len(transforms) or len(points) != len(values):
        raise ValueError("points, transforms, and values must have equal length")

    outliers, scores, threshold = detect_outliers(
        points,
        transforms,
        np.asarray(shape_zyx, dtype=np.int64),
        maximum_px=maximum_outlier_px,
    )
    predictions = values.copy()
    if not np.any(outliers):
        return outliers, predictions, None, scores, threshold

    inliers = ~outliers
    predicted, model = fit_linear_field(
        points[inliers],
        values[inliers],
        target_points=points[outliers],
    )
    predictions[outliers] = predicted
    return outliers, predictions, model, scores, threshold


def _vector_zyx(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([float(values[axis]) for axis in "zyx"], dtype=np.float64)


def _oriented_origin_spacing_um(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Window indices are oriented: voxel zero sits at the tile's low physical bound."""
    scale = _vector_zyx(record, "scale_um")
    if np.all(scale > 0):
        return _vector_zyx(record, "translation_um"), scale
    return tile_bounds_zyx_um(tile_record_from_position_record(record))[0], np.abs(scale)


def _window_physical_displacement(
    row: dict[str, Any],
    moving_record: dict[str, Any],
    *,
    fixed_scale_um_zyx: np.ndarray,
    fixed_translation_um_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    moving_start = np.asarray(row["moving_start_l0_zyx"], dtype=np.float64)
    moving_stop = np.asarray(row["moving_stop_l0_zyx"], dtype=np.float64)
    fixed_start = np.asarray(row["fixed_start_zyx"], dtype=np.float64)
    local_translation = np.asarray(row["selected_local_translation_zyx"], dtype=np.float64)
    center = ((moving_stop - moving_start) - 1.0) / 2.0
    moving_origin, moving_scale = _oriented_origin_spacing_um(moving_record)
    local_center_um = (moving_start + center) * moving_scale
    moving_center_um = moving_origin + local_center_um
    mapped_center_um = (
        fixed_translation_um_zyx + (fixed_start + center + local_translation) * fixed_scale_um_zyx
    )
    return local_center_um, mapped_center_um - moving_center_um


def _final_linear_input_exclusion(row: dict[str, Any]) -> dict[str, Any] | None:
    attempt = row.get("selected_attempt")
    if (
        row.get("status") == "accepted"
        and isinstance(attempt, str)
        and attempt.startswith("interpolated_")
        and attempt.endswith("_spatial_recovery")
    ):
        return {"reason": "unvalidated_interpolated_recovery", "score_um": None}
    return None


def filter_final_linear_outliers(args: argparse.Namespace) -> Path:
    """Reject measured outliers only when a stable per-tile linear field can be fitted."""
    import zarr

    summary_path = Path(args.summary).resolve()
    summary = _read_json(summary_path)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Linear-filter output already exists: {output_dir}")

    moving_position_path = Path(summary["moving_position"])
    moving_position = _read_json(moving_position_path)
    moving_by_tile = {str(record["tile"]): record for record in moving_position["tiles"]}
    fixed_path = Path(summary["fixed_fused"])
    fixed_group = zarr.open_group(str(fixed_path), mode="r")
    dims, scale, translation, _has_scale, _has_translation = ngff.scale_translation(fixed_group)
    if dims != ["z", "y", "x"]:
        raise ValueError(f"fixed fused axes must be z/y/x, got {dims}")
    fixed_scale = np.asarray(scale, dtype=np.float64)
    fixed_translation = np.asarray(translation, dtype=np.float64)

    records = [
        (Path(compact["level0_json"]), _read_json(Path(compact["level0_json"])))
        for compact in summary["windows"]
    ]
    exclusions: dict[Path, dict[str, Any]] = {}
    accepted_by_tile: dict[str, list[tuple[Path, dict[str, Any], np.ndarray, np.ndarray]]] = {}
    for path, row in records:
        if row.get("status") != "accepted":
            continue
        input_exclusion = _final_linear_input_exclusion(row)
        if input_exclusion is not None:
            exclusions[path] = input_exclusion
            continue
        tile = str(row["moving_tile"])
        point, displacement = _window_physical_displacement(
            row,
            moving_by_tile[tile],
            fixed_scale_um_zyx=fixed_scale,
            fixed_translation_um_zyx=fixed_translation,
        )
        accepted_by_tile.setdefault(tile, []).append((path, row, point, displacement))

    tile_reports = []
    for tile, items in sorted(accepted_by_tile.items()):
        sample_count = len(items)
        if sample_count < int(args.minimum_linear_samples):
            tile_reports.append(
                {
                    "tile": tile,
                    "sample_count": sample_count,
                    "status": "skipped_insufficient_support",
                    "excluded_count": 0,
                }
            )
            continue

        points = np.stack([item[2] for item in items])
        displacements = np.stack([item[3] for item in items])
        try:
            outliers, scores, threshold, median_residual = detect_linear_displacement_outliers(
                points,
                displacements,
                outlier_mad=float(args.outlier_mad),
                minimum_outlier_um=float(args.minimum_linear_outlier_um),
                loss_scale_um=float(args.linear_loss_scale_um),
                maximum_outlier_um=float(args.maximum_linear_outlier_um),
            )
        except ValueError as exc:
            tile_reports.append(
                {
                    "tile": tile,
                    "sample_count": sample_count,
                    "status": "skipped_insufficient_spatial_rank",
                    "error": str(exc),
                    "excluded_count": 0,
                }
            )
            continue

        if median_residual > float(args.maximum_linear_median_residual_um):
            reason = "linear_mapping_no_consensus"
            outliers[:] = True
        else:
            reason = "linear_mapping_outlier"
        for index in np.flatnonzero(outliers):
            exclusions[items[index][0]] = {
                "reason": reason,
                "score_um": float(scores[index]),
                "threshold_um": float(threshold),
            }
        tile_reports.append(
            {
                "tile": tile,
                "sample_count": sample_count,
                "status": "accepted" if reason == "linear_mapping_outlier" else reason,
                "median_residual_um": median_residual,
                "outlier_threshold_um": float(threshold),
                "excluded_count": int(np.count_nonzero(outliers)),
            }
        )

    output_window_dir = output_dir / "window_json"
    output_window_dir.mkdir(parents=True)
    output_paths: dict[Path, Path] = {}
    for original_path, row in records:
        output_path = output_window_dir / original_path.name
        output_paths[original_path] = output_path
        exclusion = exclusions.get(original_path)
        if exclusion is not None:
            row["status"] = "rejected"
            row["rejection_reason"] = exclusion["reason"]
            row["linear_mapping_quality"] = exclusion
        _write_json(output_path, row)

    report = {
        "artifact_type": "lightsheet.fused_fixed_method10_final_linear_filter.v1",
        "input_summary": str(summary_path),
        "settings": {
            "minimum_linear_samples": int(args.minimum_linear_samples),
            "linear_loss": "cauchy",
            "linear_loss_scale_um": float(args.linear_loss_scale_um),
            "outlier_mad": float(args.outlier_mad),
            "minimum_linear_outlier_um": float(args.minimum_linear_outlier_um),
            "maximum_linear_outlier_um": float(args.maximum_linear_outlier_um),
            "maximum_linear_median_residual_um": float(args.maximum_linear_median_residual_um),
        },
        "accepted_input_count": sum(len(items) for items in accepted_by_tile.values()),
        "unvalidated_interpolated_recovery_count": sum(
            exclusion["reason"] == "unvalidated_interpolated_recovery"
            for exclusion in exclusions.values()
        ),
        "excluded_count": len(exclusions),
        "tiles": tile_reports,
    }
    compact_rows = []
    for compact in summary["windows"]:
        original_path = Path(compact["level0_json"])
        updated = deepcopy(compact)
        updated["level0_json"] = str(output_paths[original_path])
        output_row = _read_json(output_paths[original_path])
        updated["status"] = output_row.get("status")
        updated["rejection_reason"] = output_row.get("rejection_reason")
        compact_rows.append(updated)
    return _write_stage_artifacts(
        source_summary=summary,
        windows=compact_rows,
        input_summary=summary_path,
        output_dir=output_dir,
        report_key="final_linear_filter",
        report_name="method10_final_linear_filter.json",
        report=report,
    )


MINIMUM_IMAGE_EVIDENCE_PLANES = 2
RESIDUAL_ALIGNED_MAX_PX = 1.5
RESIDUAL_CONTRADICTED_MIN_PX = 3.0
RESIDUAL_MAX_PLANE_SPREAD_PX = 2.0


def window_image_scorer(summary: dict[str, Any]) -> Any:
    from squisher_lightsheet.fused_fixed_qc import WindowImageScorer

    return WindowImageScorer(
        Path(summary["fixed_fused"]),
        float(summary["quality_settings"]["fixed_mask_threshold"]),
        str(summary["cache_config"]["fit_intensity_transform"]),
        window_shape_zyx=tuple(summary["window_shape_zyx"]),
    )


def image_prefers_replacement(
    fitted_planes: Sequence[float], replacement_planes: Sequence[float], margin: float
) -> bool:
    """Replace a fitted transform only when every informative plane favors the replacement.

    At least two shared planes are required, the replacement must score higher on
    each, and its mean advantage must exceed ``margin``.
    """
    differences = np.asarray(replacement_planes, dtype=np.float64) - np.asarray(
        fitted_planes, dtype=np.float64
    )
    return (
        len(differences) >= MINIMUM_IMAGE_EVIDENCE_PLANES
        and bool(np.all(differences > 0.0))
        and float(np.mean(differences)) > margin
    )


def residual_verdict(plane_shifts: Sequence[Any]) -> dict[str, Any]:
    """Classify a transform from confident, mutually consistent in-plane residual shifts."""
    confident = [shift for shift in plane_shifts if shift.confident]
    report: dict[str, Any] = {
        "measured_plane_count": len(plane_shifts),
        "confident_plane_count": len(confident),
        "residual_yx_px": None,
        "plane_spread_px": None,
    }
    if not confident:
        return {**report, "verdict": "unverified"}
    vectors = np.asarray([shift.shift_yx for shift in confident], dtype=np.float64)
    median = np.median(vectors, axis=0)
    spread = float(np.max(np.linalg.norm(vectors - median, axis=1)))
    magnitude = float(np.linalg.norm(median))
    report.update(residual_yx_px=median.tolist(), plane_spread_px=spread)
    if spread > RESIDUAL_MAX_PLANE_SPREAD_PX:
        verdict = "unverified"
    elif magnitude <= RESIDUAL_ALIGNED_MAX_PX:
        verdict = "verified"
    elif magnitude > RESIDUAL_CONTRADICTED_MIN_PX and len(confident) >= MINIMUM_IMAGE_EVIDENCE_PLANES:
        verdict = "contradicted"
    else:
        verdict = "unverified"
    return {**report, "verdict": verdict}


def aggregate_residual_verdict(shift: Any | None) -> dict[str, Any]:
    """Classify a transform from its aggregate multi-plane residual shift."""
    if shift is None or not shift.confident:
        return {"residual_yx_px": None, "verdict": "unverified"}
    magnitude = float(np.linalg.norm(shift.shift_yx))
    if magnitude <= RESIDUAL_ALIGNED_MAX_PX:
        verdict = "verified"
    elif magnitude > RESIDUAL_CONTRADICTED_MIN_PX:
        verdict = "contradicted"
    else:
        verdict = "unverified"
    return {"residual_yx_px": list(shift.shift_yx), "verdict": verdict}


def smooth_final_linear_mapping(args: argparse.Namespace) -> Path:
    """Replace retained affine outliers with inlier-field predictions the image favors."""
    import zarr

    summary_path = Path(args.summary).resolve()
    summary = _read_json(summary_path)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Linear-smoothing output already exists: {output_dir}")

    moving_position = _read_json(Path(summary["moving_position"]))
    moving_by_tile = {str(record["tile"]): record for record in moving_position["tiles"]}
    fixed_group = zarr.open_group(str(summary["fixed_fused"]), mode="r")
    dims, scale, translation, _has_scale, _has_translation = ngff.scale_translation(fixed_group)
    if dims != ["z", "y", "x"]:
        raise ValueError(f"fixed fused axes must be z/y/x, got {dims}")
    fixed_scale = np.asarray(scale, dtype=np.float64)
    fixed_translation = np.asarray(translation, dtype=np.float64)

    records = [
        (Path(compact["level0_json"]), _read_json(Path(compact["level0_json"])))
        for compact in summary["windows"]
    ]
    native_method = _native_method_from_summary(summary)
    accepted_by_tile: dict[str, list[tuple[Path, dict[str, Any], np.ndarray, np.ndarray]]] = {}
    for path, row in records:
        if row.get("status") != "accepted":
            continue
        tile = str(row["moving_tile"])
        point, displacement = _window_physical_displacement(
            row,
            moving_by_tile[tile],
            fixed_scale_um_zyx=fixed_scale,
            fixed_translation_um_zyx=fixed_translation,
        )
        accepted_by_tile.setdefault(tile, []).append((path, row, point, displacement))

    replacements: dict[Path, dict[str, Any]] = {}
    image_retained: dict[Path, dict[str, Any]] = {}
    scorer: Any | None = None
    margin = float(args.replacement_image_ncc_margin)
    tile_reports = []
    for tile, items in sorted(accepted_by_tile.items()):
        if len(items) < int(args.minimum_smoothing_samples):
            tile_reports.append(
                {
                    "tile": tile,
                    "window_count": len(items),
                    "outlier_count": 0,
                    "preserved_count": len(items),
                    "status": "skipped_insufficient_support",
                }
            )
            continue
        points = np.stack([item[2] for item in items])
        displacements = np.stack([item[3] for item in items])
        transforms = np.stack([_transform(item[1]) for item in items])
        shape = np.asarray(items[0][1]["fixed_stop_zyx"], dtype=np.int64) - np.asarray(
            items[0][1]["fixed_start_zyx"], dtype=np.int64
        )
        outliers, predictions, model, scores, threshold = outlier_only_linear_predictions(
            points=points,
            transforms=transforms,
            values=displacements,
            shape_zyx=shape,
            maximum_outlier_px=float(args.maximum_outlier_px),
        )
        mean_matrix = decomposed_affine_mean(transforms[~outliers, :9].reshape(-1, 3, 3))
        for index, (path, row, _point, _displacement) in enumerate(items):
            if not outliers[index]:
                continue
            moving_record = moving_by_tile[tile]
            moving_start = np.asarray(row["moving_start_l0_zyx"], dtype=np.float64)
            moving_stop = np.asarray(row["moving_stop_l0_zyx"], dtype=np.float64)
            fixed_start = np.asarray(row["fixed_start_zyx"], dtype=np.float64)
            center = ((moving_stop - moving_start) - 1.0) / 2.0
            moving_origin, moving_scale = _oriented_origin_spacing_um(moving_record)
            moving_center_um = moving_origin + (moving_start + center) * moving_scale
            predicted_displacement = predictions[index]
            local_translation = (
                (predicted_displacement + moving_center_um - fixed_translation) / fixed_scale
                - fixed_start
                - center
            )
            transform = np.concatenate([mean_matrix.ravel(), local_translation])
            if not _affine_is_plausible(transform):
                raise ValueError(f"linear prediction produced an implausible affine for {path}")
            if scorer is None:
                scorer = window_image_scorer(summary)
            fitted_planes, replacement_planes = scorer.plane_scores(row, [transforms[index], transform])
            image_check = {
                "fitted_image_ncc_planes": fitted_planes,
                "replacement_image_ncc_planes": replacement_planes,
                "replacement_image_ncc_margin": margin,
            }
            if not image_prefers_replacement(fitted_planes, replacement_planes, margin):
                image_retained[path] = {
                    **image_check,
                    "corner_displacement_score_px": float(scores[index]),
                    "corner_displacement_threshold_px": float(threshold),
                }
                continue
            replacements[path] = {
                "matrix": mean_matrix,
                "translation": local_translation,
                "predicted_displacement_um": predicted_displacement,
                "corner_displacement_score_px": float(scores[index]),
                "corner_displacement_threshold_px": float(threshold),
                **image_check,
            }
        tile_reports.append(
            {
                "tile": tile,
                "window_count": len(items),
                "outlier_count": int(np.count_nonzero(outliers)),
                "image_retained_outlier_count": sum(item[0] in image_retained for item in items),
                "preserved_count": sum(item[0] not in replacements for item in items),
                "corner_displacement_threshold_px": float(threshold),
                "corner_displacement_score_px": _score_summary(scores),
                "matrix_policy": "SO(3) rotation mean and arithmetic mean of polar stretch tensors",
                "mean_matrix_zyx": mean_matrix.tolist(),
                "linear_model": model,
            }
        )

    output_window_dir = output_dir / "window_json"
    output_window_dir.mkdir(parents=True)
    output_paths: dict[Path, Path] = {}
    for original_path, row in records:
        output_path = output_window_dir / original_path.name
        output_paths[original_path] = output_path
        prediction = replacements.get(original_path)
        if prediction is not None:
            row["final_linear_interpolation"] = {
                "source_summary": str(summary_path),
                "original_selected_attempt": row.get("selected_attempt"),
                "original_selected_local_matrix_zyx": row.get("selected_local_matrix_zyx"),
                "original_selected_local_translation_zyx": row.get("selected_local_translation_zyx"),
                "predicted_displacement_um_zyx": prediction["predicted_displacement_um"].tolist(),
                "predicted_matrix_zyx": prediction["matrix"].tolist(),
                "corner_displacement_score_px": prediction["corner_displacement_score_px"],
                "corner_displacement_threshold_px": prediction["corner_displacement_threshold_px"],
                "fitted_image_ncc_planes": prediction["fitted_image_ncc_planes"],
                "replacement_image_ncc_planes": prediction["replacement_image_ncc_planes"],
                "replacement_image_ncc_margin": prediction["replacement_image_ncc_margin"],
            }
            _set_selected_transform(
                row,
                matrix=prediction["matrix"],
                translation=prediction["translation"],
                attempt=_linear_outlier_replaced_attempt(native_method),
            )
        retained = image_retained.get(original_path)
        if retained is not None:
            row["final_linear_image_retained"] = {
                "source_summary": str(summary_path),
                **retained,
            }
        _write_json(output_path, row)

    report = {
        "artifact_type": "lightsheet.fused_fixed_method10_final_linear_outlier_replacement.v2",
        "input_summary": str(summary_path),
        "interpolated_window_count": len(replacements),
        "image_retained_outlier_count": len(image_retained),
        "preserved_window_count": sum(len(items) for items in accepted_by_tile.values()) - len(replacements),
        "tile_count": len(tile_reports),
        "minimum_smoothing_samples": int(args.minimum_smoothing_samples),
        "maximum_outlier_px": float(args.maximum_outlier_px),
        "replacement_image_ncc_margin": margin,
        "interpolated_parameters": [
            "physical_displacement_z",
            "physical_displacement_y",
            "physical_displacement_x",
        ],
        "matrix_policy": "SO(3) rotation mean and arithmetic mean of polar stretch tensors",
        "tiles": tile_reports,
    }
    compact_rows = []
    for compact in summary["windows"]:
        original_path = Path(compact["level0_json"])
        output_path = output_paths[original_path]
        compact_rows.append(
            _compact_window_record(compact, _read_json(output_path), output_path)
        )
    return _write_stage_artifacts(
        source_summary=summary,
        windows=compact_rows,
        input_summary=summary_path,
        output_dir=output_dir,
        report_key="final_linear_interpolation",
        report_name="method10_final_linear_interpolation.json",
        report=report,
    )


def detect_outliers(
    points: np.ndarray,
    transforms: np.ndarray,
    shape_zyx: np.ndarray,
    *,
    maximum_px: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points) < 3:
        return np.zeros(len(points), dtype=bool), np.full(len(points), np.nan), float(maximum_px)
    normalized, _origin, _spacing = _normalized_points(points)
    active = np.ones(len(points), dtype=bool)
    all_scores = np.full(len(points), np.nan, dtype=np.float64)
    while True:
        active_indices = np.flatnonzero(active)
        if len(active_indices) < 3:
            break
        round_scores = np.empty(len(active_indices), dtype=np.float64)
        for score_index, target_index in enumerate(active_indices):
            source_indices = active_indices[active_indices != target_index]
            predicted = predict_decomposed_affine_field(
                normalized[source_indices],
                transforms[source_indices],
                target_points=normalized[target_index : target_index + 1],
            )[0]
            round_scores[score_index] = _corner_displacement_px(
                transforms[target_index],
                predicted,
                shape_zyx,
            )
        all_scores[active_indices] = round_scores
        worst_score_index = int(np.argmax(round_scores))
        if round_scores[worst_score_index] <= maximum_px:
            break
        active[active_indices[worst_score_index]] = False
    return ~active, all_scores, float(maximum_px)


def final_transform_departures(
    rows: Sequence[tuple[Path, dict[str, Any]]], *, maximum_px: float = DEFAULT_MAXIMUM_OUTLIER_PX
) -> dict[Path, dict[str, Any]]:
    """Score each accepted window's final selected transform against the other accepted windows of its tile.

    Read-only use of the recovery outlier rule (``detect_outliers``): the leave-one-out decomposed-affine prediction
    and its maximum corner displacement in level-0 pixels, excluding the worst window above ``maximum_px`` until none
    remains. A tile with fewer than three accepted windows has no field to compare against, and windows left when
    exclusion exhausts a tile's support have too little; both report ``insufficient_support`` instead of a score.
    """
    by_tile: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, row in rows:
        if row.get("status") == "accepted":
            by_tile.setdefault(_tile_number(row["moving_tile"]), []).append((path, row))
    results: dict[Path, dict[str, Any]] = {}
    for _tile, items in sorted(by_tile.items()):
        support = len(items) - 1
        if len(items) < 3:
            for path, _row in items:
                results[path] = {
                    "status": "insufficient_support",
                    "support_count": support,
                    "score_px": None,
                    "threshold_px": float(maximum_px),
                    "departs": None,
                }
            continue
        shape = np.asarray(items[0][1]["fixed_stop_zyx"], dtype=np.int64) - np.asarray(
            items[0][1]["fixed_start_zyx"], dtype=np.int64
        )
        outliers, scores, threshold = detect_outliers(
            np.stack([_point(row) for _path, row in items]),
            np.stack([_transform(row) for _path, row in items]),
            shape,
            maximum_px=maximum_px,
        )
        # Iterative exclusion stops below three windows; their last scores then compare against too little support.
        exhausted = int(np.count_nonzero(~outliers)) < 3
        for index, (path, _row) in enumerate(items):
            if exhausted and not outliers[index]:
                results[path] = {
                    "status": "insufficient_support",
                    "support_count": int(np.count_nonzero(~outliers)) - 1,
                    "score_px": None,
                    "threshold_px": threshold,
                    "departs": None,
                }
                continue
            results[path] = {
                "status": "scored",
                "support_count": support,
                "score_px": float(scores[index]),
                "threshold_px": threshold,
                "departs": bool(outliers[index]),
            }
    return results


def _load_tile_adjacency(path: Path) -> dict[str, set[str]]:
    pair_summaries = _read_json(path).get("pair_summaries")
    if not isinstance(pair_summaries, dict):
        raise ValueError(f"{path} must contain a pair_summaries object")
    adjacency: dict[str, set[str]] = {}
    for pair in pair_summaries:
        parts = str(pair).split("-")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Invalid adjacent tile pair {pair!r} in {path}")
        first, second = (part.zfill(3) for part in parts)
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    return adjacency


def adjacent_translation_initializer(
    points: np.ndarray,
    transforms: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Fit translation at a target-local window coordinate from adjacent tiles."""
    unique_points, inverse = np.unique(np.asarray(points, dtype=np.float64), axis=0, return_inverse=True)
    values = np.asarray(transforms, dtype=np.float64)
    collapsed = np.stack([np.median(values[inverse == index], axis=0) for index in range(len(unique_points))])
    if len(unique_points) < 2:
        raise ValueError("Adjacent translation model requires at least two distinct window coordinates")
    normalized, origin, spacing = _normalized_points(unique_points)
    translation = fit_linear_field(
        normalized,
        collapsed[:, 9:],
        target_points=((np.asarray(target, dtype=np.float64) - origin) / spacing)[np.newaxis, :],
    )[0][0]
    matrix = decomposed_affine_mean(collapsed[:, :9].reshape(-1, 3, 3))
    return np.concatenate([matrix.ravel(), translation])


def _affine_is_plausible(transform: np.ndarray) -> bool:
    matrix = transform[:9].reshape(3, 3)
    determinant = float(np.linalg.det(matrix))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    condition = float(singular_values.max() / singular_values.min())
    return (
        np.all(np.isfinite(transform))
        and 1.0 / 1.1 <= determinant <= 1.1
        and condition <= 1.1
        and float(singular_values.min()) >= 1.0 / 1.1
        and float(singular_values.max()) <= 1.1
    )


def _global_forward_from_local(
    row: dict[str, Any], matrix: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fixed_start = np.asarray(row["fixed_start_zyx"], dtype=np.float64)
    moving_start = np.asarray(row["moving_start_l0_zyx"], dtype=np.float64)
    shape = np.asarray(row["fixed_stop_zyx"], dtype=np.float64) - fixed_start
    center = (shape - 1.0) / 2.0
    local_offset = center + translation - matrix @ center
    return matrix, local_offset - matrix @ moving_start + fixed_start


def _set_selected_transform(
    row: dict[str, Any],
    *,
    matrix: np.ndarray,
    translation: np.ndarray,
    attempt: str,
) -> None:
    global_matrix, global_offset = _global_forward_from_local(row, matrix, translation)
    pull_matrix = np.linalg.inv(global_matrix)
    pull_offset = -pull_matrix @ global_offset
    row.update(
        {
            "selected_attempt": attempt,
            "selected_local_matrix_zyx": matrix.tolist(),
            "selected_local_translation_zyx": translation.tolist(),
            "local_matrix_zyx": matrix.tolist(),
            "local_translation_zyx": translation.tolist(),
            "selected_fixed_fused_l0_to_moving_l0_pull_matrix_zyx": pull_matrix.tolist(),
            "selected_fixed_fused_l0_to_moving_l0_pull_offset_zyx": pull_offset.tolist(),
            "selected_moving_l0_to_fixed_fused_l0_matrix_zyx": global_matrix.tolist(),
            "selected_moving_l0_to_fixed_fused_l0_offset_zyx": global_offset.tolist(),
            "selected_corr_refined": None,
            "selected_gradient_component_ncc_mean": None,
            "selected_gradient_component_ncc_refined": None,
        }
    )


def _record_interpolated_fallback(
    row: dict[str, Any],
    transform: np.ndarray,
    *,
    provenance: dict[str, Any],
    fallback_reason: str,
    native_method: str,
) -> dict[str, Any]:
    recovered = deepcopy(row)
    matrix = transform[:9].reshape(3, 3)
    translation = transform[9:]
    for key in tuple(recovered):
        if key.startswith("selected_"):
            recovered.pop(key)
    recovered.setdefault("method8_attempts", []).append(
        {
            "name": _interpolated_recovery_attempt(native_method),
            "status": "rejected",
            "rejection_reason": fallback_reason,
            "native_return_code": None,
            "local_matrix_zyx": matrix.tolist(),
            "local_translation_zyx": translation.tolist(),
            "recovery_fallback_reason": fallback_reason,
        }
    )
    recovered.update(
        {
            "status": "rejected",
            "rejection_reason": fallback_reason,
            "selected_attempt": None,
            "outlier_recovery": {
                **provenance,
                "interpolated_initializer_local_matrix_zyx": matrix.tolist(),
                "interpolated_initializer_local_translation_zyx": translation.tolist(),
                "interpolated_fallback_reason": fallback_reason,
            },
        }
    )
    return recovered


def _retain_accepted_original(
    row: dict[str, Any], *, provenance: dict[str, Any], reason: str
) -> dict[str, Any]:
    retained = deepcopy(row)
    retained["outlier_recovery"] = {
        **provenance,
        "native_rerun_selected": False,
        "retained_original_reason": reason,
    }
    return retained


def _recovery_provenance(
    row: dict[str, Any],
    *,
    original_path: Path,
    reason: str,
    score_px: float | None,
    threshold_px: float,
    maximum_refit_displacement_px: float,
    source_paths: list[Path],
    native_method: str,
) -> dict[str, Any]:
    return {
        "method": f"per-tile decomposed affine model from spatially inlier {native_method} windows",
        "matrix_policy": "SO(3) rotation mean and arithmetic mean of polar stretch tensors",
        "translation_policy": "ordinary least-squares spatial field",
        "reason": reason,
        "source_window_json": str(original_path.resolve()),
        "source_good_window_json": [str(path.resolve()) for path in source_paths],
        "corner_displacement_score_px": score_px,
        "corner_displacement_threshold_px": float(threshold_px),
        "maximum_refit_displacement_px": float(maximum_refit_displacement_px),
        "original_status": row.get("status"),
        "original_rejection_reason": row.get("rejection_reason"),
        "original_selected_attempt": row.get("selected_attempt"),
        "original_selected_local_matrix_zyx": row.get("selected_local_matrix_zyx"),
        "original_selected_local_translation_zyx": row.get("selected_local_translation_zyx"),
    }


MANUAL_INITIALIZATIONS_ARTIFACT = "lightsheet.manual_window_initializations.v1"
MANUAL_INITIALIZATION_KEYS = (
    "moving_tile",
    "moving_start_l0_zyx",
    "local_matrix_zyx",
    "local_translation_zyx",
    "seed_origin",
    "evidence",
)


def load_manual_initializations(path: Path, *, channel: str) -> dict[str, dict[str, Any]]:
    """Load one channel's curated window initializations from a dataset master file.

    Entries are keyed by window JSON name and hold absolute window-local transforms
    together with the placement they refer to.
    """
    payload = _read_json(path)
    if payload.get("artifact_type") != MANUAL_INITIALIZATIONS_ARTIFACT:
        raise ValueError(f"{path} is not a {MANUAL_INITIALIZATIONS_ARTIFACT} artifact")
    entries = payload["channels"].get(channel, {})
    for window, entry in entries.items():
        missing = [key for key in MANUAL_INITIALIZATION_KEYS if key not in entry]
        if missing:
            raise ValueError(f"Manual initialization {channel}/{window} is missing {missing}")
        if entry["seed_origin"] not in ("manual", "measured"):
            raise ValueError(f"Manual initialization {channel}/{window} has unknown seed_origin")
        transform = np.concatenate(
            [
                np.asarray(entry["local_matrix_zyx"], dtype=np.float64).ravel(),
                np.asarray(entry["local_translation_zyx"], dtype=np.float64),
            ]
        )
        if transform.shape != (12,) or not _affine_is_plausible(transform):
            raise ValueError(f"Manual initialization {channel}/{window} is not a plausible affine")
    return entries


def _apply_manual_initializations(
    *,
    entries: dict[str, dict[str, Any]],
    records: list[tuple[Path, dict[str, Any]]],
    recovery_by_path: dict[Path, dict[str, Any]],
    provenance_base: Callable[[Path, dict[str, Any]], dict[str, Any]],
    source: dict[str, Any],
) -> None:
    """Replace or add recovery initializers for curated windows; fail on stale entries."""
    by_name = {path.name: (path, row) for path, row in records}
    unknown = sorted(set(entries) - set(by_name))
    if unknown:
        raise ValueError(f"Manual initializations name windows absent from the summary: {unknown}")
    for window, entry in sorted(entries.items()):
        path, row = by_name[window]
        for key in ("moving_tile", "moving_start_l0_zyx"):
            if row.get(key) != entry[key]:
                raise ValueError(
                    f"Manual initialization {window} {key}={entry[key]} does not match {row.get(key)}"
                )
        if row.get("rejection_reason") in MASK_REJECTION_REASONS:
            raise ValueError(f"Manual initialization {window} targets a mask-rejected window")
        previous = recovery_by_path.get(path)
        provenance = previous["provenance"] if previous is not None else provenance_base(path, row)
        recovery_by_path[path] = {
            "row": row,
            "transform": np.concatenate(
                [
                    np.asarray(entry["local_matrix_zyx"], dtype=np.float64).ravel(),
                    np.asarray(entry["local_translation_zyx"], dtype=np.float64),
                ]
            ),
            "provenance": {
                **provenance,
                "initializer_source": "manual",
                "manual_initialization": {
                    **source,
                    "seed_origin": entry["seed_origin"],
                    "evidence": entry["evidence"],
                    "replaced_initializer_local_translation_zyx": None
                    if previous is None
                    else previous["transform"][9:].tolist(),
                },
            },
        }


SEED_ATTEMPTS = ("level2_initializer_phase_primed", "native_method6_from_recovery_initializer")
DEFAULT_SEED_MAXIMUM_MASKED_FRACTION = 0.6
DEFAULT_SEED_MAXIMUM_TRANSLATION_PX = 60.0
DEFAULT_SEED_MINIMUM_GRADIENT_NCC = 0.10


def propose_manual_initializations(
    *,
    complete_summary: Path,
    master: Path,
    channel: str,
    maximum_masked_fraction: float = DEFAULT_SEED_MAXIMUM_MASKED_FRACTION,
    maximum_translation_px: float = DEFAULT_SEED_MAXIMUM_TRANSLATION_PX,
    minimum_gradient_ncc: float = DEFAULT_SEED_MINIMUM_GRADIENT_NCC,
) -> dict[str, dict[str, Any]]:
    """Write measured seeds for withheld tissue windows into one channel of the dataset master file.

    Candidates are complete-recovery rows that are not accepted, not mask-rejected, and mostly tissue. A window whose
    spatial placement the image contradicted is seeded with the selected interpolation or native rerun shifted
    by the residual measured for that exact candidate. Every other seed is the window's best native-recovery attempt among the phase-primed
    level-2 initializer and the native refit from the recovery initializer, by gradient NCC, skipping implausible
    affines, translations beyond the neighbourhood, and attempts below the gradient NCC floor. Entries curated by
    hand (seed_origin "manual") and seeds of windows now accepted are kept as they are.
    """
    from squisher_lightsheet.complete_recovery import (
        CONTRADICTED_INTERPOLATION_REASON,
        _quality_gate_rejected_native_rerun,
    )
    from squisher_lightsheet.fused_fixed_qc import MOVING_XY_FACTOR

    summary = _read_json(complete_summary)
    native_windows = Path(summary["input_summary"]).parent / "window_json"
    payload = (
        _read_json(master)
        if master.is_file()
        else {"artifact_type": MANUAL_INITIALIZATIONS_ARTIFACT, "channels": {}}
    )
    if payload.get("artifact_type") != MANUAL_INITIALIZATIONS_ARTIFACT:
        raise ValueError(f"{master} is not a {MANUAL_INITIALIZATIONS_ARTIFACT} artifact")
    existing = payload["channels"].get(channel, {})
    accepted = {Path(row["level0_json"]).name for row in summary["windows"] if row["status"] == "accepted"}
    entries = {
        window: entry
        for window, entry in existing.items()
        if entry["seed_origin"] == "manual" or window in accepted
    }
    for row in summary["windows"]:
        masked = row.get("fixed_masked_fraction")
        if (
            row["status"] == "accepted"
            or row.get("rejection_reason") in MASK_REJECTION_REASONS
            or masked is None
            or masked >= maximum_masked_fraction
        ):
            continue
        name = Path(row["level0_json"]).name
        if name in entries:
            continue
        record = _read_json(native_windows / name)
        if row.get("rejection_reason") == CONTRADICTED_INTERPOLATION_REASON:
            completed = _read_json(Path(row["level0_json"]))
            promotion = completed["interpolated_fallback_promotion"]
            residual = promotion["residual"]
            provenance = completed["outlier_recovery"]
            matrix = np.asarray(provenance["interpolated_initializer_local_matrix_zyx"], dtype=np.float64)
            translation = np.asarray(provenance["interpolated_initializer_local_translation_zyx"], dtype=np.float64)
            placement_source = promotion["initializer_source"]
            candidate = promotion["selected_candidate"]
            if candidate == "native_rerun":
                if "native_method" not in completed:
                    raise ValueError(f"{row['level0_json']} contradicted a native_rerun candidate without method provenance")
                rerun = _quality_gate_rejected_native_rerun(
                    completed, native_method=completed["native_method"],
                    interpolated=np.concatenate([matrix.ravel(), translation]),
                )
                if rerun is None:
                    raise ValueError(f"{row['level0_json']} has no quality-gate-rejected native rerun for its residual")
                matrix = rerun["transform"][:9].reshape(3, 3)
                translation = rerun["transform"][9:].copy()
                placement_source = candidate
            elif candidate != "interpolated":
                raise ValueError(f"{row['level0_json']} contradicted an unsupported {candidate} candidate")
            # The scorer samples its fixed level-2 planes at MOVING_XY_FACTOR level-0 pixels per pixel.
            translation[1:] += np.asarray(residual["residual_yx_px"], dtype=np.float64) * MOVING_XY_FACTOR
            if float(np.max(np.abs(translation))) > maximum_translation_px:
                continue
            entries[name] = {
                "moving_tile": record["moving_tile"],
                "moving_start_l0_zyx": record["moving_start_l0_zyx"],
                "local_matrix_zyx": matrix.tolist(),
                "local_translation_zyx": translation.tolist(),
                "seed_origin": "measured",
                "evidence": (
                    f"{row['level0_json']}: the image contradicted the {placement_source} placement "
                    f"with a residual of {residual['residual_yx_px']} level-2 YX px on "
                    f"{residual['confident_plane_count']} confident planes (spread {residual['plane_spread_px']:.2f} "
                    "px); the seed is that placement shifted by the residual."
                ),
            }
            continue
        candidates = []
        for attempt in record["method8_attempts"]:
            if attempt["name"] not in SEED_ATTEMPTS or attempt.get("local_matrix_zyx") is None:
                continue
            translation = np.asarray(attempt["local_translation_zyx"], dtype=np.float64)
            transform = np.concatenate(
                [np.asarray(attempt["local_matrix_zyx"], dtype=np.float64).ravel(), translation]
            )
            gradient = attempt.get("gradient_component_ncc_mean")
            if (
                gradient is None
                or gradient < minimum_gradient_ncc
                or float(np.max(np.abs(translation))) > maximum_translation_px
                or not _affine_is_plausible(transform)
            ):
                continue
            candidates.append((float(gradient), attempt))
        if not candidates:
            continue
        gradient, attempt = max(candidates, key=lambda item: item[0])
        entries[name] = {
            "moving_tile": record["moving_tile"],
            "moving_start_l0_zyx": record["moving_start_l0_zyx"],
            "local_matrix_zyx": attempt["local_matrix_zyx"],
            "local_translation_zyx": attempt["local_translation_zyx"],
            "seed_origin": "measured",
            "evidence": (
                f"{native_windows / name} attempt {attempt['name']}: gradient NCC {gradient:.3f}, corr "
                f"{attempt.get('corr_refined')}, translation {attempt['local_translation_zyx']} l0 voxels; "
                f"complete-recovery withheld the window ({row.get('rejection_reason')})."
            ),
        }
    payload["channels"][channel] = entries
    _write_json(master, payload)
    return entries


def _initializer_payload(
    row: dict[str, Any], transform: np.ndarray, provenance: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "lightsheet.fused_fixed_native_spatial_recovery_initializer.v1",
        "status": "accepted",
        "rejection_reason": None,
        "moving_tile": row["moving_tile"],
        "moving_start_l0_zyx": row["moving_start_l0_zyx"],
        "level_factor_zyx": [1, 1, 1],
        "local_matrix_zyx": transform[:9].reshape(3, 3).tolist(),
        "local_translation_zyx": transform[9:].tolist(),
        "recovery": provenance,
    }


def _load_sweep_module(sweep_runner: Path = DEFAULT_SWEEP_RUNNER) -> ModuleType:
    sweep_runner = sweep_runner.resolve()
    if sweep_runner == DEFAULT_SWEEP_RUNNER.resolve():
        from squisher_lightsheet._backends import fused_fixed_sweep

        return fused_fixed_sweep
    module_name = "_squisher_custom_sweep_" + str(sweep_runner).encode().hex()
    spec = importlib.util.spec_from_file_location(module_name, sweep_runner)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sweep runner: {sweep_runner}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _persistent_sweep_args(
    *,
    summary: dict[str, Any],
    cache: dict[str, Any],
    output_dir: Path,
    initializer_dir: Path,
    devices: tuple[int, ...],
    target_count: int,
    resume: bool,
    source_cache_max_gib: float = 0.0,
    sweep_runner: Path = DEFAULT_SWEEP_RUNNER,
) -> argparse.Namespace:
    sweep = _load_sweep_module(sweep_runner)
    arguments = [
        "--fixed-position",
        str(summary["moving_position"]),
        "--moving-position",
        str(summary["moving_position"]),
        "--moving-source-position",
        str(summary["moving_source_position"]),
        "--fixed-fused",
        str(summary["fixed_fused"]),
        "--output-dir",
        str(output_dir),
        "--core-shape-zyx",
        ",".join(str(int(value)) for value in summary["core_shape_zyx"]),
        "--window-shape-zyx",
        ",".join(str(int(value)) for value in summary["window_shape_zyx"]),
        "--fit-downsample-zyx",
        ",".join(str(int(value)) for value in cache["fit_downsample_zyx"]),
        "--moving-channel",
        str(int(cache["moving_channel"])),
        "--source-cache-max-gib",
        str(source_cache_max_gib),
        "--native-lib-dir",
        str(cache["native_lib_dir"]),
        "--native-method",
        str(cache["native_method"]),
        "--fit-intensity-transform",
        str(cache["fit_intensity_transform"]),
        "--mattes-bins",
        str(int(cache["mattes_bins"])),
        "--mattes-samples",
        str(int(cache["mattes_samples"])),
        "--ftol",
        str(float(cache["ftol"])),
        "--max-iterations",
        str(int(cache["max_iterations"])),
        "--phase-upsample-factor",
        str(int(cache["phase_upsample_factor"])),
        "--min-corr",
        str(float(cache["min_corr"])),
        "--min-grad-ncc",
        str(float(cache["min_grad_ncc"])),
        "--fixed-mask-threshold",
        str(float(cache["fixed_mask_threshold"])),
        "--fixed-mask-level",
        str(int(cache["fixed_mask_level"])),
        "--fixed-mask-min-voxels",
        str(int(cache["fixed_mask_min_voxels"])),
        "--fixed-mask-max-masked-fraction",
        str(float(cache["fixed_mask_max_masked_fraction"])),
        "--workers",
        str(len(devices)),
        "--max-tasks-per-worker",
        str(max(1, target_count)),
        "--devices",
        ",".join(str(device) for device in devices),
        "--level0-initializer",
        "window-interpolated",
        "--window-initializer-dir",
        str(initializer_dir),
        "--resume" if resume else "--no-resume",
    ]
    if cache.get("local_3d_refinement") is True:
        arguments.extend(["--local-3d-refinement", "--local-affine-estimator", str(cache.get("local_affine_estimator", "least-squares"))])
    if bool(cache["mattes_fixed_shear"]):
        arguments.append("--mattes-fixed-shear")
    return sweep.build_parser().parse_args(arguments)


def _run_persistent_sweep(
    *,
    summary: dict[str, Any],
    cache: dict[str, Any],
    output_dir: Path,
    initializer_dir: Path,
    devices: tuple[int, ...],
    target_count: int,
    resume: bool,
    source_cache_max_gib: float = 0.0,
    sweep_runner: Path = DEFAULT_SWEEP_RUNNER,
) -> None:
    sweep = _load_sweep_module(sweep_runner)
    sweep.run(
        _persistent_sweep_args(
            summary=summary,
            cache=cache,
            output_dir=output_dir,
            initializer_dir=initializer_dir,
            devices=devices,
            target_count=target_count,
            resume=resume,
            source_cache_max_gib=source_cache_max_gib,
            sweep_runner=sweep_runner,
        )
    )


def _transform_log_values(transform: np.ndarray | None) -> dict[str, list[float] | None]:
    if transform is None:
        return {"translation_zyx": None, "matrix_upper_offdiagonal_zy_zx_yx": None, "matrix_zyx": None}
    matrix = transform[:9].reshape(3, 3)
    return {
        "translation_zyx": transform[9:].astype(float).tolist(),
        "matrix_upper_offdiagonal_zy_zx_yx": [
            float(matrix[0, 1]),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
        ],
        "matrix_zyx": matrix.astype(float).tolist(),
    }


def _finalize_native_rerun(
    task: dict[str, Any], *, output_dir: Path, native_method: str
) -> dict[str, Any]:
    row = task["row"]
    original_path = Path(task["original_path"])
    output_path = output_dir / "window_json" / original_path.name
    initializer_path = output_dir / "recovery_initializers" / original_path.name
    shape = np.asarray(row["fixed_stop_zyx"], dtype=np.int64) - np.asarray(
        row["fixed_start_zyx"], dtype=np.int64
    )
    rerun_provenance = {
        **task["provenance"],
        "initializer_json": str(initializer_path),
        "rerun_worker": str(DEFAULT_SWEEP_RUNNER),
        "rerun_native_method": native_method,
        "rerun_phase_priming_skipped": True,
    }
    initial_values = _transform_log_values(task["transform"])
    if output_path.is_file():
        existing = _read_json(output_path)
        existing_recovery = existing.get("outlier_recovery")
        if (
            isinstance(existing_recovery, dict)
            and existing_recovery.get("initializer_json") == str(initializer_path)
            and existing_recovery.get("rerun_native_method") == native_method
            and existing_recovery.get("source_window_json")
            == rerun_provenance.get("source_window_json")
            and existing_recovery.get("initial_round") == initial_values
        ):
            refined_values = existing_recovery.get("refinement_round") or {
                "translation_zyx": None,
                "matrix_upper_offdiagonal_zy_zx_yx": None,
            }
            return {
                "window_json": str(output_path),
                "status": existing.get("status"),
                "rejection_reason": existing.get("rejection_reason"),
                "selected_attempt": existing.get("selected_attempt"),
                "native_rerun_selected": existing_recovery.get(
                    "native_rerun_selected", False
                ),
                "refined_translation_zyx": refined_values.get("translation_zyx"),
                "refined_matrix_upper_offdiagonal_zy_zx_yx": refined_values.get(
                    "matrix_upper_offdiagonal_zy_zx_yx"
                ),
            }
    rerun_transform = None
    if not output_path.is_file():
        fallback_reason = "native_worker_output_missing"
        if row.get("status") == "accepted":
            result = _retain_accepted_original(
                row, provenance=rerun_provenance, reason=fallback_reason
            )
        else:
            result = _record_interpolated_fallback(
                row,
                task["transform"],
                provenance=rerun_provenance,
                fallback_reason=fallback_reason,
                native_method=native_method,
            )
    else:
        rerun = _read_json(output_path)
        if rerun.get("status") == "error":
            raise RuntimeError(f"Native recovery worker failed for {output_path}: {rerun.get('error')}")
        if "selected_local_matrix_zyx" in rerun and "selected_local_translation_zyx" in rerun:
            rerun_transform = _transform(rerun)
        spatial_delta = (
            None
            if rerun_transform is None
            else _corner_displacement_px(rerun_transform, task["transform"], shape)
        )
        rerun_provenance.update(
            {
                "native_rerun_status": rerun.get("status"),
                "native_rerun_rejection_reason": rerun.get("rejection_reason"),
                "native_rerun_corner_displacement_from_initializer_px": spatial_delta,
            }
        )
        original_grad_ncc = row.get("selected_gradient_component_ncc_mean")
        rerun_grad_ncc = rerun.get("selected_gradient_component_ncc_mean")
        require_grad_ncc_improvement = task["provenance"].get("reason") == "spatial_transform_outlier"
        grad_ncc_improved = (
            isinstance(original_grad_ncc, (int, float))
            and np.isfinite(original_grad_ncc)
            and isinstance(rerun_grad_ncc, (int, float))
            and np.isfinite(rerun_grad_ncc)
            and float(rerun_grad_ncc) > float(original_grad_ncc)
        )
        rerun_provenance.update(
            {
                "original_gradient_component_ncc_mean": original_grad_ncc,
                "native_rerun_gradient_component_ncc_mean": rerun_grad_ncc,
                "native_rerun_gradient_component_ncc_improved": bool(grad_ncc_improved),
            }
        )
        if (
            rerun.get("status") == "accepted"
            and (rerun.get("selected_attempt") == _native_recovery_attempt(native_method)
                 or _is_local_3d_native_fit(rerun, native_method))
            and (not require_grad_ncc_improvement or grad_ncc_improved)
        ):
            result = rerun
            result["outlier_recovery"] = {**rerun_provenance, "native_rerun_selected": True}
        else:
            if rerun.get("status") != "accepted":
                fallback_reason = "native_rerun_rejected"
            elif require_grad_ncc_improvement and not grad_ncc_improved:
                fallback_reason = "native_rerun_grad_ncc_not_improved"
            else:
                fallback_reason = "native_rerun_attempt_mismatch"
            if row.get("status") == "accepted":
                result = _retain_accepted_original(
                    row, provenance=rerun_provenance, reason=fallback_reason
                )
            else:
                result = _record_interpolated_fallback(
                    rerun,
                    task["transform"],
                    provenance={**rerun_provenance, "native_rerun_selected": False},
                    fallback_reason=fallback_reason,
                    native_method=native_method,
                )

    refined_values = _transform_log_values(rerun_transform)
    result.setdefault("outlier_recovery", {}).update(
        {
            "initial_round": initial_values,
            "refinement_round": refined_values,
        }
    )
    _write_json(output_path, result)
    return {
        "window_json": str(output_path),
        "status": result.get("status"),
        "rejection_reason": result.get("rejection_reason"),
        "selected_attempt": result.get("selected_attempt"),
        "native_rerun_selected": result.get("outlier_recovery", {}).get("native_rerun_selected", False),
        "refined_translation_zyx": refined_values["translation_zyx"],
        "refined_matrix_upper_offdiagonal_zy_zx_yx": refined_values[
            "matrix_upper_offdiagonal_zy_zx_yx"
        ],
    }


def run(args: argparse.Namespace) -> Path | None:
    if args.exclude_nonlinear_only and args.smooth_retained_linear:
        raise ValueError("--exclude-nonlinear-only and --smooth-retained-linear are mutually exclusive")
    if args.exclude_nonlinear_only:
        if args.dry_run:
            raise ValueError("--exclude-nonlinear-only does not support --dry-run")
        return filter_final_linear_outliers(args)
    if args.smooth_retained_linear:
        if args.dry_run:
            raise ValueError("--smooth-retained-linear does not support --dry-run")
        return smooth_final_linear_mapping(args)
    summary_path = Path(args.summary).resolve()
    summary = _read_json(summary_path)
    native_method = _native_method_from_summary(summary)
    tile_filter = (
        None
        if not args.tile_filter
        else {value.strip().zfill(3) for value in args.tile_filter.split(",") if value.strip()}
    )
    records: list[tuple[Path, dict[str, Any]]] = []
    for summary_row in summary.get("windows", []):
        if tile_filter is not None and _tile_number(summary_row["tile"]) not in tile_filter:
            continue
        path = Path(summary_row["level0_json"])
        row = _read_json(path)
        records.append((path, row))

    by_tile: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, row in records:
        by_tile.setdefault(_tile_number(row["moving_tile"]), []).append((path, row))

    candidates_by_tile = {
        tile: [
            (path, row)
            for path, row in items
            if row.get("status") == "accepted"
            and isinstance(row.get("selected_attempt"), str)
            and (row["selected_attempt"].startswith(_native_attempt_prefix(native_method))
                 or _is_local_3d_native_fit(row, native_method))
            and row["selected_attempt"] != _native_recovery_attempt(native_method)
            and "selected_local_matrix_zyx" in row
            and "selected_local_translation_zyx" in row
        ]
        for tile, items in by_tile.items()
    }

    recovery_by_path: dict[Path, dict[str, Any]] = {}
    inlier_candidates_by_tile: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    tile_reports = []
    for tile, items in sorted(by_tile.items()):
        candidates = candidates_by_tile[tile]
        if len(candidates) < 2:
            inlier_candidates_by_tile[tile] = candidates
            candidate_paths = {path for path, _row in candidates}
            unsupported = [
                path
                for path, row in items
                if row.get("rejection_reason") not in MASK_REJECTION_REASONS
                and (args.rerun_all_eligible or path not in candidate_paths)
            ]
            tile_reports.append(
                {
                    "tile": tile,
                    "candidate_native_count": len(candidates),
                    "good_count": len(candidates),
                    "outlier_count": 0,
                    "outliers": [],
                    "recovery_target_count": len(unsupported),
                    "interpolation_ready_count": 0,
                    "interpolation_failures": [
                        {"window_json": str(path), "reason": "insufficient_within_tile_good_support"}
                        for path in unsupported
                    ],
                }
            )
            continue
        points = np.stack([_point(row) for _path, row in candidates])
        transforms = np.stack([_transform(row) for _path, row in candidates])
        shape = np.asarray(candidates[0][1]["fixed_stop_zyx"], dtype=np.int64) - np.asarray(
            candidates[0][1]["fixed_start_zyx"], dtype=np.int64
        )
        outliers, scores, threshold = detect_outliers(
            points,
            transforms,
            shape,
            maximum_px=float(args.maximum_outlier_px),
        )
        good_indices = np.flatnonzero(~outliers)
        inlier_candidates_by_tile[tile] = [candidates[index] for index in good_indices]
        normalized, origin, spacing = _normalized_points(points)
        candidate_index_by_path = {path: index for index, (path, _row) in enumerate(candidates)}
        candidate_paths = {path for path, _row in candidates}
        if args.rerun_all_eligible:
            targets = [
                (
                    path,
                    row,
                    "spatial_transform_outlier"
                    if path in candidate_index_by_path
                    and outliers[candidate_index_by_path[path]]
                    else "all_chunk_leave_one_out_refinement",
                    None
                    if path not in candidate_index_by_path
                    else float(scores[candidate_index_by_path[path]]),
                )
                for path, row in items
                if row.get("rejection_reason") not in MASK_REJECTION_REASONS
            ]
        else:
            targets = [
                (
                    candidates[index][0],
                    candidates[index][1],
                    "spatial_transform_outlier",
                    float(scores[index]),
                )
                for index in np.flatnonzero(outliers)
            ]
            targets.extend(
                (path, row, "missing_usable_native_transform", None)
                for path, row in items
                if path not in candidate_paths and row.get("rejection_reason") not in MASK_REJECTION_REASONS
            )
        interpolation_failures = []
        for path, row, reason, score in targets:
            target = (_point(row) - origin) / spacing
            source_indices = good_indices
            candidate_index = candidate_index_by_path.get(path)
            if candidate_index is not None and candidate_index in good_indices:
                source_indices = good_indices[good_indices != candidate_index]
            if len(source_indices) < 2:
                interpolation_failures.append(
                    {"window_json": str(path), "reason": "insufficient_leave_one_out_support"}
                )
                continue
            transform = predict_decomposed_affine_field(
                normalized[source_indices],
                transforms[source_indices],
                target_points=target[np.newaxis, :],
            )[0]
            if not _affine_is_plausible(transform):
                interpolation_failures.append(
                    {"window_json": str(path), "reason": "interpolated_affine_not_plausible"}
                )
                continue
            provenance = _recovery_provenance(
                row,
                original_path=path,
                reason=reason,
                score_px=score,
                threshold_px=threshold,
                maximum_refit_displacement_px=float(args.maximum_refit_displacement_px),
                source_paths=[candidates[index][0] for index in source_indices],
                native_method=native_method,
            )
            recovery_by_path[path] = {"row": row, "transform": transform, "provenance": provenance}
        tile_reports.append(
            {
                "tile": tile,
                "candidate_native_count": len(candidates),
                "good_count": len(good_indices),
                "outlier_count": int(np.count_nonzero(outliers)),
                "outliers": [
                    {
                        "window_json": str(candidates[index][0]),
                        "moving_start_l0_zyx": candidates[index][1]["moving_start_l0_zyx"],
                        "corner_displacement_score_px": float(scores[index]),
                    }
                    for index in np.flatnonzero(outliers)
                ],
                "recovery_target_count": len(targets),
                "interpolation_ready_count": sum(
                    path in recovery_by_path for path, _row, _reason, _score in targets
                ),
                "interpolation_failures": interpolation_failures,
                "corner_displacement_threshold_px": float(threshold),
                "corner_displacement_score_px": _score_summary(scores),
            }
        )

    if args.adjacency_json is not None:
        adjacency = _load_tile_adjacency(Path(args.adjacency_json).resolve())
        reports_by_tile = {report["tile"]: report for report in tile_reports}
        for tile, candidates in inlier_candidates_by_tile.items():
            report = reports_by_tile[tile]
            if len(candidates) >= 2 or report["recovery_target_count"] == 0:
                continue
            adjacent_good_tiles = sorted(
                neighbor for neighbor in adjacency.get(tile, set()) if inlier_candidates_by_tile.get(neighbor)
            )
            report["adjacent_good_tiles"] = adjacent_good_tiles
            if not adjacent_good_tiles:
                continue
            sources = [*candidates]
            for neighbor in adjacent_good_tiles:
                sources.extend(inlier_candidates_by_tile[neighbor])
            source_points = np.stack([_point(row) for _path, row in sources])
            source_transforms = np.stack([_transform(row) for _path, row in sources])
            candidate_paths = {path for path, _row in candidates}
            targets = [
                (path, row)
                for path, row in by_tile[tile]
                if path not in candidate_paths and row.get("rejection_reason") not in MASK_REJECTION_REASONS
            ]
            failures = []
            for path, row in targets:
                try:
                    transform = adjacent_translation_initializer(
                        source_points,
                        source_transforms,
                        _point(row),
                    )
                    if not _affine_is_plausible(transform):
                        raise ValueError("adjacent-neighbor affine baseline is not plausible")
                except ValueError as exc:
                    failures.append({"window_json": str(path), "reason": str(exc)})
                    continue
                provenance = _recovery_provenance(
                    row,
                    original_path=path,
                    reason="missing_within_tile_support_with_adjacent_good_neighbor",
                    score_px=None,
                    threshold_px=float(args.maximum_outlier_px),
                    maximum_refit_displacement_px=float(args.maximum_refit_displacement_px),
                    source_paths=[source_path for source_path, _source_row in sources],
                    native_method=native_method,
                )
                provenance.update(
                    {
                        "method": f"translation interpolation from registration-graph-adjacent good {native_method} tiles",
                        "adjacent_good_tiles": adjacent_good_tiles,
                        "matrix_policy": "SO(3) rotation mean and arithmetic mean of polar stretch tensors",
                    }
                )
                recovery_by_path[path] = {"row": row, "transform": transform, "provenance": provenance}
            report["interpolation_ready_count"] = sum(path in recovery_by_path for path, _row in targets)
            report["interpolation_failures"] = failures
            report["adjacent_recovery_ready_count"] = report["interpolation_ready_count"]

    manual_path = getattr(args, "manual_initializations", None)
    manual_channel = getattr(args, "manual_initializations_channel", None)
    if (manual_path is None) != (manual_channel is None):
        raise ValueError(
            "--manual-initializations and --manual-initializations-channel must be given together"
        )
    if manual_path is not None:
        manual_path = Path(manual_path).resolve()
        entries = load_manual_initializations(manual_path, channel=str(manual_channel))
        if tile_filter is not None:
            entries = {
                window: entry
                for window, entry in entries.items()
                if _tile_number(entry["moving_tile"]) in tile_filter
            }
        _apply_manual_initializations(
            entries=entries,
            records=records,
            recovery_by_path=recovery_by_path,
            provenance_base=lambda path, row: _recovery_provenance(
                row,
                original_path=path,
                reason="manual_initialization",
                score_px=None,
                threshold_px=float(args.maximum_outlier_px),
                maximum_refit_displacement_px=float(args.maximum_refit_displacement_px),
                source_paths=[],
                native_method=native_method,
            ),
            source={"path": str(manual_path), "channel": str(manual_channel)},
        )
    report = {
        "artifact_type": "lightsheet.fused_fixed_native_spatial_outlier_recovery.v1",
        "input_summary": str(summary_path),
        "settings": {
            "native_method": native_method,
            "method": "polar decomposition, scipy Rotation.mean, and numpy.linalg.lstsq",
            "model": "constant decomposed affine geometry plus spatially linear translation",
            "maximum_outlier_px": float(args.maximum_outlier_px),
            "maximum_refit_displacement_px": float(args.maximum_refit_displacement_px),
            "tile_filter": None if tile_filter is None else sorted(tile_filter),
            "rerun_all_eligible": bool(args.rerun_all_eligible),
            "adjacency_json": None
            if args.adjacency_json is None
            else str(Path(args.adjacency_json).resolve()),
            "manual_initializations": None
            if manual_path is None
            else {"path": str(manual_path), "channel": str(manual_channel)},
        },
        "tile_count": len(by_tile),
        "outlier_count": sum(int(item.get("outlier_count", 0)) for item in tile_reports),
        "recovery_target_count": sum(int(item.get("recovery_target_count", 0)) for item in tile_reports),
        "interpolation_ready_count": len(recovery_by_path),
        "manual_initialization_count": sum(
            recovery["provenance"].get("initializer_source") == "manual"
            for recovery in recovery_by_path.values()
        ),
        "interpolation_failure_count": sum(
            len(item.get("interpolation_failures", [])) for item in tile_reports
        ),
        "tiles": tile_reports,
    }
    if args.dry_run:
        print(json.dumps(_json_safe(report), indent=2, allow_nan=False), flush=True)
        return None

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Recovery output already exists: {output_dir}")
    output_window_dir = output_dir / "window_json"
    output_window_dir.mkdir(parents=True, exist_ok=args.resume)
    output_paths: dict[Path, Path] = {}
    for original_path, row in records:
        output_path = output_window_dir / original_path.name
        output_paths[original_path] = output_path
        recovery = recovery_by_path.get(original_path)
        if recovery is None:
            _write_json(output_path, row)
            continue
        initializer_path = output_dir / "recovery_initializers" / original_path.name
        _write_json(
            initializer_path, _initializer_payload(row, recovery["transform"], recovery["provenance"])
        )

    devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device")
    if recovery_by_path:
        first_recovery = next(iter(recovery_by_path.values()))
        _run_persistent_sweep(
            summary=summary,
            cache=first_recovery["row"]["cache_config"],
            output_dir=output_dir,
            initializer_dir=output_dir / "recovery_initializers",
            devices=devices,
            target_count=len(recovery_by_path),
            resume=bool(args.resume),
            source_cache_max_gib=float(getattr(args, "source_cache_max_gib", 0.0)),
            sweep_runner=Path(args.sweep_runner),
        )
    rerun_results = [
        _finalize_native_rerun(
            {"original_path": path, **recovery},
            output_dir=output_dir,
            native_method=native_method,
        )
        for path, recovery in sorted(recovery_by_path.items(), key=lambda item: str(item[0]))
    ]
    report["native_rerun"] = {
        "devices": list(devices),
        "count": len(rerun_results),
        "status_counts": {
            status: sum(result.get("status") == status for result in rerun_results)
            for status in sorted({str(result.get("status")) for result in rerun_results})
        },
        "selected_attempt_counts": {
            attempt: sum(result.get("selected_attempt") == attempt for result in rerun_results)
            for attempt in sorted({str(result.get("selected_attempt")) for result in rerun_results})
        },
        "native_rerun_selected_count": sum(
            bool(result.get("native_rerun_selected")) for result in rerun_results
        ),
        "results": rerun_results,
    }

    compact_rows = []
    for compact in summary.get("windows", []):
        original_path = Path(compact["level0_json"])
        if original_path not in output_paths:
            compact_rows.append(deepcopy(compact))
            continue
        output_path = output_paths[original_path]
        compact_rows.append(
            _compact_window_record(compact, _read_json(output_path), output_path)
        )
    return _write_stage_artifacts(
        source_summary=summary,
        windows=compact_rows,
        input_summary=summary_path,
        output_dir=output_dir,
        report_key="outlier_recovery",
        report_name="native_outlier_recovery.json",
        report=report,
    )


def recover_native(
    *,
    summary: Path,
    output_dir: Path | None,
    maximum_outlier_px: float = DEFAULT_MAXIMUM_OUTLIER_PX,
    maximum_refit_displacement_px: float = DEFAULT_MAXIMUM_REFIT_DISPLACEMENT_PX,
    devices: tuple[int, ...] = DEFAULT_NATIVE_DEVICES,
    adjacency_json: Path | None = None,
    tile_filter: str | None = None,
    rerun_all_eligible: bool = False,
    source_cache_max_gib: float = 0.0,
    sweep_runner: Path = DEFAULT_SWEEP_RUNNER,
    manual_initializations: Path | None = None,
    manual_initializations_channel: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> Path | None:
    """Run native spatial recovery without exposing completion-policy stages."""
    if not devices:
        raise ValueError("devices must contain at least one CUDA device")
    if not np.isfinite(source_cache_max_gib) or source_cache_max_gib < 0:
        raise ValueError("source_cache_max_gib must be finite and nonnegative")
    if not dry_run and output_dir is None:
        raise ValueError("output_dir is required unless dry_run is enabled")
    return run(
        argparse.Namespace(
            summary=summary,
            output_dir=output_dir,
            tile_filter=tile_filter,
            maximum_outlier_px=maximum_outlier_px,
            maximum_refit_displacement_px=maximum_refit_displacement_px,
            devices=",".join(map(str, devices)),
            adjacency_json=adjacency_json,
            rerun_all_eligible=rerun_all_eligible,
            source_cache_max_gib=source_cache_max_gib,
            sweep_runner=sweep_runner,
            manual_initializations=manual_initializations,
            manual_initializations_channel=manual_initializations_channel,
            dry_run=dry_run,
            resume=resume,
            exclude_nonlinear_only=False,
            smooth_retained_linear=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect native affine outliers and recover them from decomposed spatial models."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tile-filter", help="Comma-separated three-digit tile numbers.")
    parser.add_argument("--outlier-mad", type=float, default=DEFAULT_OUTLIER_MAD)
    parser.add_argument(
        "--maximum-outlier-px", type=float, default=DEFAULT_MAXIMUM_OUTLIER_PX
    )
    parser.add_argument(
        "--maximum-refit-displacement-px",
        type=float,
        default=DEFAULT_MAXIMUM_REFIT_DISPLACEMENT_PX,
    )
    parser.add_argument(
        "--exclude-nonlinear-only",
        action="store_true",
        help="Do not rerun registration; copy the final result while rejecting physical mappings that violate a per-tile linear field.",
    )
    parser.add_argument(
        "--smooth-retained-linear",
        action="store_true",
        help="Do not rerun registration; replace only retained affine outliers with per-tile linear predictions.",
    )
    parser.add_argument(
        "--minimum-linear-samples", type=int, default=DEFAULT_MINIMUM_LINEAR_SAMPLES
    )
    parser.add_argument(
        "--minimum-smoothing-samples",
        type=int,
        default=DEFAULT_MINIMUM_SMOOTHING_SAMPLES,
    )
    parser.add_argument(
        "--replacement-image-ncc-margin",
        type=float,
        default=DEFAULT_REPLACEMENT_IMAGE_NCC_MARGIN,
    )
    parser.add_argument(
        "--linear-loss-scale-um", type=float, default=DEFAULT_LINEAR_LOSS_SCALE_UM
    )
    parser.add_argument(
        "--minimum-linear-outlier-um",
        type=float,
        default=DEFAULT_MINIMUM_LINEAR_OUTLIER_UM,
    )
    parser.add_argument(
        "--maximum-linear-outlier-um",
        type=float,
        default=DEFAULT_MAXIMUM_LINEAR_OUTLIER_UM,
    )
    parser.add_argument(
        "--maximum-linear-median-residual-um",
        type=float,
        default=DEFAULT_MAXIMUM_LINEAR_MEDIAN_RESIDUAL_UM,
    )
    parser.add_argument(
        "--devices",
        default=DEFAULT_NATIVE_DEVICE_SPEC,
        help=(
            "Comma-separated local CUDA indices when CUDA_VISIBLE_DEVICES is set; "
            "physical indices otherwise."
        ),
    )
    parser.add_argument(
        "--adjacency-json",
        type=Path,
        help="Registration measurements JSON; only graph-adjacent good tiles may seed unsupported tiles.",
    )
    parser.add_argument("--sweep-runner", type=Path, default=DEFAULT_SWEEP_RUNNER)
    parser.add_argument("--source-cache-max-gib", type=float, default=0.0)
    parser.add_argument(
        "--rerun-all-eligible",
        action="store_true",
        help="Rerun every non-masked chunk from a leave-one-out per-tile consensus transform.",
    )
    parser.add_argument("--manual-initializations", type=Path)
    parser.add_argument("--manual-initializations-channel")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.dry_run and args.output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run is used")
    output = run(args)
    if output is not None:
        print(output, flush=True)


if __name__ == "__main__":
    main()
