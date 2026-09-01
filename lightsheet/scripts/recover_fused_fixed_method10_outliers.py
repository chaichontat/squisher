#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.linalg import polar
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


from squisher_lightsheet import ngff
from squisher_lightsheet.channel_affine import output_to_input_from_model


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
SWEEP_SCRIPT = Path("/home/chaichontat/nvme/lightsheet/scripts/run_fused_fixed_method8_sweep.py")


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
    moving_scale = _vector_zyx(moving_record, "scale_um")
    moving_stage = _vector_zyx(moving_record, "translation_um")
    local_center_um = (moving_start + center) * moving_scale
    moving_center_um = moving_stage + local_center_um
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
    filtered_summary = deepcopy(summary)
    filtered_summary["output_dir"] = str(output_dir)
    filtered_summary["input_summary"] = str(summary_path)
    filtered_summary["final_linear_filter"] = report
    for compact in filtered_summary["windows"]:
        original_path = Path(compact["level0_json"])
        compact["level0_json"] = str(output_paths[original_path])
        output_row = _read_json(output_paths[original_path])
        compact["status"] = output_row.get("status")
        compact["rejection_reason"] = output_row.get("rejection_reason")
    filtered_summary["aggregate"] = {
        **filtered_summary.get("aggregate", {}),
        "accepted": sum(row.get("status") == "accepted" for row in filtered_summary["windows"]),
        "rejected": sum(row.get("status") == "rejected" for row in filtered_summary["windows"]),
        "error": sum(row.get("status") == "error" for row in filtered_summary["windows"]),
    }
    report_path = output_dir / "method10_final_linear_filter.json"
    summary_output = output_dir / "fused_fixed_method8_summary.json"
    _write_json(report_path, report)
    _write_json(summary_output, filtered_summary)
    return summary_output


def smooth_final_linear_mapping(args: argparse.Namespace) -> Path:
    """Replace only retained affine outliers with predictions from the inlier field."""
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
            moving_scale = _vector_zyx(moving_record, "scale_um")
            moving_stage = _vector_zyx(moving_record, "translation_um")
            moving_center_um = moving_stage + (moving_start + center) * moving_scale
            predicted_displacement = predictions[index]
            local_translation = (
                (predicted_displacement + moving_center_um - fixed_translation) / fixed_scale
                - fixed_start
                - center
            )
            transform = np.concatenate([mean_matrix.ravel(), local_translation])
            if not _affine_is_plausible(transform):
                raise ValueError(f"linear prediction produced an implausible affine for {path}")
            replacements[path] = {
                "matrix": mean_matrix,
                "translation": local_translation,
                "predicted_displacement_um": predicted_displacement,
                "corner_displacement_score_px": float(scores[index]),
                "corner_displacement_threshold_px": float(threshold),
            }
        tile_reports.append(
            {
                "tile": tile,
                "window_count": len(items),
                "outlier_count": int(np.count_nonzero(outliers)),
                "preserved_count": int(np.count_nonzero(~outliers)),
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
            }
            _set_selected_transform(
                row,
                matrix=prediction["matrix"],
                translation=prediction["translation"],
                attempt=_linear_outlier_replaced_attempt(native_method),
            )
        _write_json(output_path, row)

    report = {
        "artifact_type": "lightsheet.fused_fixed_method10_final_linear_outlier_replacement.v2",
        "input_summary": str(summary_path),
        "interpolated_window_count": len(replacements),
        "preserved_window_count": sum(len(items) for items in accepted_by_tile.values()) - len(replacements),
        "tile_count": len(tile_reports),
        "minimum_smoothing_samples": int(args.minimum_smoothing_samples),
        "maximum_outlier_px": float(args.maximum_outlier_px),
        "interpolated_parameters": [
            "physical_displacement_z",
            "physical_displacement_y",
            "physical_displacement_x",
        ],
        "matrix_policy": "SO(3) rotation mean and arithmetic mean of polar stretch tensors",
        "tiles": tile_reports,
    }
    smoothed_summary = deepcopy(summary)
    smoothed_summary["output_dir"] = str(output_dir)
    smoothed_summary["input_summary"] = str(summary_path)
    smoothed_summary["final_linear_interpolation"] = report
    for compact in smoothed_summary["windows"]:
        original_path = Path(compact["level0_json"])
        compact["level0_json"] = str(output_paths[original_path])
        output_row = _read_json(output_paths[original_path])
        compact["selected_attempt"] = output_row.get("selected_attempt")
    report_path = output_dir / "method10_final_linear_interpolation.json"
    summary_output = output_dir / "fused_fixed_method8_summary.json"
    _write_json(report_path, report)
    _write_json(summary_output, smoothed_summary)
    return summary_output


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
    shape = np.asarray(row["fixed_stop_zyx"], dtype=np.int64) - np.asarray(
        row["fixed_start_zyx"], dtype=np.int64
    )
    pull_matrix, pull_offset = output_to_input_from_model(
        matrix, translation, tuple(int(value) for value in shape)
    )
    global_matrix, global_offset = _global_forward_from_local(row, matrix, translation)
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


def _load_sweep_module() -> Any:
    script_dir = str(SWEEP_SCRIPT.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    return importlib.import_module(SWEEP_SCRIPT.stem)


def _persistent_sweep_args(
    *,
    summary: dict[str, Any],
    cache: dict[str, Any],
    output_dir: Path,
    initializer_dir: Path,
    devices: tuple[int, ...],
    target_count: int,
    resume: bool,
) -> argparse.Namespace:
    sweep = _load_sweep_module()
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
) -> None:
    sweep = _load_sweep_module()
    sweep.run(
        _persistent_sweep_args(
            summary=summary,
            cache=cache,
            output_dir=output_dir,
            initializer_dir=initializer_dir,
            devices=devices,
            target_count=target_count,
            resume=resume,
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
        "rerun_worker": str(SWEEP_SCRIPT),
        "rerun_native_method": native_method,
        "rerun_phase_priming_skipped": True,
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
            and rerun.get("selected_attempt") == _native_recovery_attempt(native_method)
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

    initial_values = _transform_log_values(task["transform"])
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
    native_attempt_prefix = _native_attempt_prefix(native_method)
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
            and row["selected_attempt"].startswith(native_attempt_prefix)
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
        },
        "tile_count": len(by_tile),
        "outlier_count": sum(int(item.get("outlier_count", 0)) for item in tile_reports),
        "recovery_target_count": sum(int(item.get("recovery_target_count", 0)) for item in tile_reports),
        "interpolation_ready_count": len(recovery_by_path),
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

    recovered_summary = deepcopy(summary)
    recovered_summary["output_dir"] = str(output_dir)
    recovered_summary["input_summary"] = str(summary_path)
    recovered_summary["outlier_recovery"] = report
    for row in recovered_summary.get("windows", []):
        original_path = Path(row["level0_json"])
        if original_path not in output_paths:
            continue
        row["level0_json"] = str(output_paths[original_path])
        output_row = _read_json(output_paths[original_path])
        row["status"] = output_row.get("status")
        row["rejection_reason"] = output_row.get("rejection_reason")
        row["selected_attempt"] = output_row.get("selected_attempt")
        row["selected_corr"] = output_row.get("selected_corr_refined")
        row["selected_grad_ncc"] = output_row.get("selected_gradient_component_ncc_mean")
    recovered_summary["aggregate"] = {
        **recovered_summary.get("aggregate", {}),
        "accepted": sum(row.get("status") == "accepted" for row in recovered_summary.get("windows", [])),
        "rejected": sum(row.get("status") == "rejected" for row in recovered_summary.get("windows", [])),
        "error": sum(row.get("status") == "error" for row in recovered_summary.get("windows", [])),
    }
    report_path = output_dir / "native_outlier_recovery.json"
    summary_output = output_dir / "fused_fixed_method8_summary.json"
    _write_json(report_path, report)
    _write_json(summary_output, recovered_summary)
    return summary_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect native affine outliers and recover them from decomposed spatial models."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tile-filter", help="Comma-separated three-digit tile numbers.")
    parser.add_argument("--outlier-mad", type=float, default=4.0)
    parser.add_argument("--maximum-outlier-px", type=float, default=5.0)
    parser.add_argument("--maximum-refit-displacement-px", type=float, default=10.0)
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
    parser.add_argument("--minimum-linear-samples", type=int, default=8)
    parser.add_argument("--minimum-smoothing-samples", type=int, default=4)
    parser.add_argument("--linear-loss-scale-um", type=float, default=2.0)
    parser.add_argument("--minimum-linear-outlier-um", type=float, default=3.0)
    parser.add_argument("--maximum-linear-outlier-um", type=float, default=8.0)
    parser.add_argument("--maximum-linear-median-residual-um", type=float, default=4.0)
    parser.add_argument(
        "--devices", default="0,1", help="Comma-separated physical CUDA devices for native reruns."
    )
    parser.add_argument(
        "--adjacency-json",
        type=Path,
        help="Registration measurements JSON; only graph-adjacent good tiles may seed unsupported tiles.",
    )
    parser.add_argument(
        "--rerun-all-eligible",
        action="store_true",
        help="Rerun every non-masked chunk from a leave-one-out per-tile consensus transform.",
    )
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
