"""Estimate a two-axis planar tilt from a mask and OME-Zarr volumes."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from squisher_lightsheet import ngff


def _read_mask(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"Mask {path} must be 3D, got dimension {image.GetDimension()}")
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    if not np.allclose(direction, np.eye(3), atol=1e-6):
        raise ValueError(f"Mask {path} must have identity direction; got {direction.tolist()}")
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Mask {path} has invalid spacing {spacing.tolist()}")
    origin = np.asarray(image.GetOrigin(), dtype=float)
    if np.any(~np.isfinite(origin)):
        raise ValueError(f"Mask {path} has invalid origin {origin.tolist()}")
    array = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    return array, spacing[::-1] * 1000.0, origin[::-1] * 1000.0


def _validate_window(window: Sequence[float], index: int) -> tuple[float, float]:
    if len(window) != 2:
        raise ValueError(f"Source {index} window must contain exactly lower and upper bounds")
    lower, upper = (float(value) for value in window)
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError(f"Source {index} window must be finite with lower < upper")
    return lower, upper


def _minor_angle(covariance: np.ndarray, axes: tuple[int, int]) -> float:
    projected = covariance[np.ix_(axes, axes)]
    vector = np.linalg.eigh(projected)[1][:, 0]
    if vector[1] < 0:
        vector = -vector
    return math.degrees(math.atan2(float(vector[0]), float(vector[1])))


def _weighted_covariance(
    weights: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute XYZ moments without materializing a dense XYZ coordinate field."""
    x_weights = weights.sum(axis=(0, 1), dtype=np.float64)
    y_weights = weights.sum(axis=(0, 2), dtype=np.float64)
    z_weights = weights.sum(axis=(1, 2), dtype=np.float64)
    total = float(x_weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("no finite positive normalized mask-weighted samples")
    center = np.asarray(
        [
            np.dot(x_weights, x_um) / total,
            np.dot(y_weights, y_um) / total,
            np.dot(z_weights, z_um) / total,
        ]
    )
    centered_x = x_um - center[0]
    centered_y = y_um - center[1]
    centered_z = z_um - center[2]
    xx = np.dot(x_weights, centered_x**2)
    yy = np.dot(y_weights, centered_y**2)
    zz = np.dot(z_weights, centered_z**2)
    xy = np.einsum("zyx,y,x->", weights, centered_y, centered_x, optimize=True)
    xz = np.einsum("zyx,z,x->", weights, centered_z, centered_x, optimize=True)
    yz = np.einsum("zyx,z,y->", weights, centered_z, centered_y, optimize=True)
    covariance = np.asarray([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) / total
    return covariance, center, total


def write_planar_tilt_fit(
    *,
    mask_path: Path,
    source_paths: Sequence[Path],
    windows: Sequence[Sequence[float]],
    output_path: Path,
    level: int,
    z_depth_um: float,
    xy_step_um: float,
) -> Path:
    """Sample centered source slabs and atomically write their pooled tilt fit."""
    if not source_paths:
        raise ValueError("At least one OME-Zarr source is required")
    if len(source_paths) != len(windows):
        raise ValueError(
            f"Expected one window per source, got {len(source_paths)} sources and {len(windows)} windows"
        )
    if (
        level < 0
        or not math.isfinite(z_depth_um)
        or z_depth_um <= 0
        or not math.isfinite(xy_step_um)
        or xy_step_um <= 0
    ):
        raise ValueError("level must be non-negative; z_depth_um and xy_step_um must be positive")

    mask, mask_spacing, mask_origin = _read_mask(mask_path)
    covariances: list[np.ndarray] = []
    reference_shape: tuple[int, ...] | None = None
    reference_scale: np.ndarray | None = None
    reference_translation: np.ndarray | None = None
    fitted_z_planes: int | None = None
    fitted_xy_stride_yx: tuple[int, int] | None = None
    source_records: list[dict[str, Any]] = []
    for index, (path, window) in enumerate(zip(source_paths, windows, strict=True)):
        lower, upper = _validate_window(window, index)
        import zarr

        root = zarr.open_group(str(path), mode="r")
        array = ngff.level_array(root, level=level, context=path)
        axes = ngff.axes(root, array)
        if axes != "ZYX":
            raise ValueError(f"Source {path} must have spatial axes exactly ZYX, got {axes}")
        names, scale, translation, has_scale, _ = ngff.scale_translation(root, dataset_index=level)
        if not has_scale:
            raise ValueError(f"Source {path} is missing physical scale metadata")
        if [str(name).lower() for name in names] != list("zyx"):
            raise ValueError(f"Source {path} multiscale axes must be exactly z,y,x")
        raw_axes = ngff.multiscales(root)[0].get("axes")
        if not isinstance(raw_axes, list):
            raise ValueError(f"Source {path} has malformed multiscale axes metadata")
        axis_units = {
            str(axis.get("name")).lower(): axis.get("unit") for axis in raw_axes if isinstance(axis, dict)
        }
        if any(axis_units.get(axis) not in {"micrometer", "micron", "um", "µm"} for axis in "zyx"):
            raise ValueError(f"Source {path} spatial axes must use micrometer units")
        spatial_scale = np.asarray(scale, dtype=float)
        if np.any(~np.isfinite(spatial_scale)) or np.any(spatial_scale <= 0):
            raise ValueError(f"Source {path} must have positive finite spatial scales")
        current_shape = tuple(int(value) for value in array.shape)
        current_translation = np.asarray(translation, dtype=float)
        if np.any(~np.isfinite(current_translation)):
            raise ValueError(f"Source {path} must have a finite physical translation")
        if reference_shape is None:
            reference_shape = current_shape
            reference_scale = spatial_scale
            reference_translation = current_translation
        elif (
            current_shape != reference_shape
            or not np.allclose(spatial_scale, reference_scale)
            or not np.allclose(current_translation, reference_translation)
        ):
            raise ValueError(f"Source {path} does not match the first source shape, scale, and translation")
        source_z_planes = max(1, round(z_depth_um / spatial_scale[0]))
        if source_z_planes > current_shape[0]:
            raise ValueError(
                f"Source {path} has depth {current_shape[0]}, fewer than the "
                f"{source_z_planes} planes required for z_depth_um={z_depth_um}"
            )
        fitted_z_planes = source_z_planes
        y_stride = max(1, round(xy_step_um / spatial_scale[1]))
        x_stride = max(1, round(xy_step_um / spatial_scale[2]))
        fitted_xy_stride_yx = (y_stride, x_stride)
        z0 = (current_shape[0] - source_z_planes) // 2
        z_indices = np.arange(z0, z0 + source_z_planes)
        values = np.asarray(array[z_indices, ::y_stride, ::x_stride], dtype=np.float32)
        normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
        z_um = z_indices * spatial_scale[0] + current_translation[0]
        y_um = np.arange(0, current_shape[1], y_stride) * spatial_scale[1] + current_translation[1]
        x_um = np.arange(0, current_shape[2], x_stride) * spatial_scale[2] + current_translation[2]
        mask_axes = [
            ((axis_um - origin_um) / spacing_um).astype(np.float32)
            for axis_um, origin_um, spacing_um in zip(
                (z_um, y_um, x_um), mask_origin, mask_spacing, strict=True
            )
        ]
        mask_coords = np.meshgrid(*mask_axes, indexing="ij")
        mask_values = np.clip(
            map_coordinates(mask, mask_coords, order=3, mode="constant", cval=0.0),
            0.0,
            1.0,
        )
        weights = normalized * mask_values
        try:
            covariance, center, total = _weighted_covariance(weights, x_um, y_um, z_um)
        except ValueError as error:
            raise ValueError(f"Source {path} has {error}") from error
        covariances.append(covariance)
        source_records.append(
            {
                "path": str(path),
                "window": [lower, upper],
                "shape_zyx": list(current_shape),
                "scale_zyx_um": spatial_scale.tolist(),
                "translation_zyx_um": current_translation.tolist(),
                "z_index_range": [int(z_indices[0]), int(z_indices[-1])],
                "weight_sum": total,
                "center_xyz_um": center.tolist(),
            }
        )

    pooled = np.mean(covariances, axis=0)
    lateral = _minor_angle(pooled, (0, 2))
    pitch = _minor_angle(pooled, (1, 2))
    normal = np.asarray([math.tan(math.radians(lateral)), math.tan(math.radians(pitch)), 1.0])
    normal /= np.linalg.norm(normal)
    if fitted_z_planes is None or fitted_xy_stride_yx is None or reference_scale is None:
        raise RuntimeError("Planar tilt fit completed without source sampling metadata")
    payload = {
        "coronal_lateral_tilt_deg": lateral,
        "coronal_pitch_deg": pitch,
        "normal_xyz": normal.tolist(),
        "level": level,
        "requested_z_depth_um": z_depth_um,
        "fitted_z_planes": fitted_z_planes,
        "fitted_z_depth_um": fitted_z_planes * float(reference_scale[0]),
        "requested_xy_step_um": xy_step_um,
        "fitted_xy_stride_yx": list(fitted_xy_stride_yx),
        "fitted_xy_step_yx_um": (np.asarray(fitted_xy_stride_yx) * reference_scale[1:]).tolist(),
        "sources": source_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path
