#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet._legacy.rough_align_tltr_center_z_phase import TileRecord
from squisher_lightsheet.channel_affine import (
    _block_mean_downsample_zyx_cupy,
    _corr_gpu,
    _gradient_component_ncc_mean,
    _model_to_fit_downsample,
    _native_pull_from_fit_downsample,
    _native_window_low_content_reason,
    _read_level0_crop,
    _robust_norm_and_content_stats_cupy,
    estimate_translation_gpu,
    full_model_to_local,
    gradient_component_ncc_3d_gpu,
    local_model_to_full,
    moving_crop_start_for_fixed_crop,
    output_to_input_from_model,
    output_to_input_to_model,
)
from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR, register_method8_device, zyx_to_xyz_3x4
from squisher_lightsheet.tile_phase import (
    _open_tile_level_array,
    flattened_channel_count,
    raw_axis_slice_for_oriented_slice,
    tile_record_from_position_record,
)


README_PRESEED_MATRIX_ZYX = np.asarray(
    [
        [1.0000000000, 0.0000000000, 0.0000000000],
        [0.0056567129, 0.9999939077, -0.0034906514],
        [-0.0047305605, 0.0034906514, 0.9999939077],
    ],
    dtype=np.float64,
)
EMPTY_PRECHECK_VERSION = "level2_uniform_gpu_precheck_fixed_mask_v10"
RESUMABLE_EMPTY_PRECHECK_VERSIONS = {EMPTY_PRECHECK_VERSION}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=True) + "\n")
    tmp.replace(path)


def _parse_devices(value: str) -> tuple[int, ...]:
    devices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not devices:
        raise argparse.ArgumentTypeError("expected at least one CUDA device id")
    return devices


def _parse_matrix_zyx(value: str | None) -> np.ndarray:
    if value is None:
        return README_PRESEED_MATRIX_ZYX.copy()
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(values) != 9:
        raise argparse.ArgumentTypeError("--preseed-matrix-zyx must contain 9 row-major values")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).removesuffix(".ome.zarr").removesuffix(".ome.tif")


def _tile_number(tile_name: str) -> str:
    match = re.search(r"\.(\d+)\.ome\.zarr$", tile_name)
    if match is None:
        raise ValueError(f"cannot extract tile number from {tile_name!r}")
    return match.group(1)


def _axis_starts(size: int, *, window: int, step: int) -> list[int]:
    if window > size:
        raise ValueError(f"window={window} exceeds axis size={size}")
    starts = list(range(0, size - window + 1, step))
    tail = size - window
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return starts


def quadrant_z_windows(
    shape_zyx: tuple[int, int, int],
    *,
    core_shape_zyx: tuple[int, int, int] = (480, 480, 480),
    window_shape_zyx: tuple[int, int, int] = (528, 528, 528),
) -> list[dict[str, Any]]:
    shape = tuple(int(value) for value in shape_zyx)
    core = tuple(int(value) for value in core_shape_zyx)
    window = tuple(int(value) for value in window_shape_zyx)
    z_starts = _axis_starts(shape[0], window=window[0], step=core[0])
    y_starts = _axis_starts(shape[1], window=window[1], step=core[1])
    x_starts = _axis_starts(shape[2], window=window[2], step=core[2])
    if len(y_starts) != 2 or len(x_starts) != 2:
        raise ValueError(
            f"expected exactly four xy quadrant windows, got {len(y_starts)}x{len(x_starts)} for shape={shape}"
        )
    windows: list[dict[str, Any]] = []
    for qy, y_start in enumerate(y_starts):
        for qx, x_start in enumerate(x_starts):
            quadrant = f"qy{qy}_qx{qx}"
            for z_start in z_starts:
                start = np.asarray([z_start, y_start, x_start], dtype=np.int64)
                stop = start + np.asarray(window, dtype=np.int64)
                windows.append(
                    {
                        "quadrant": quadrant,
                        "fixed_start_zyx": start.tolist(),
                        "fixed_stop_zyx": stop.tolist(),
                        "window_shape_zyx": list(window),
                    }
                )
    return windows


def preseeded_level0_model(
    *,
    fixed_tile: TileRecord,
    moving_tile: TileRecord,
    preseed_matrix_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if tuple(fixed_tile.shape_zyx.tolist()) != tuple(moving_tile.shape_zyx.tolist()):
        raise ValueError(
            f"fixed/moving logical tile shapes differ for {fixed_tile.tile}: "
            f"{fixed_tile.shape_zyx.tolist()} vs {moving_tile.shape_zyx.tolist()}"
        )
    fixed_origin_um = np.asarray(fixed_tile.translation_zyx_um, dtype=np.float64)
    moving_origin_um = np.asarray(moving_tile.translation_zyx_um, dtype=np.float64)
    fixed_scale_um = np.abs(np.asarray(fixed_tile.scale_zyx_um, dtype=np.float64))
    moving_scale_um = np.abs(np.asarray(moving_tile.scale_zyx_um, dtype=np.float64))
    fixed_center = (np.asarray(fixed_tile.shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moving_center = (np.asarray(moving_tile.shape_zyx, dtype=np.float64) - 1.0) / 2.0
    translation = (moving_origin_um - fixed_origin_um + moving_scale_um * moving_center) / fixed_scale_um - fixed_center
    matrix = np.asarray(preseed_matrix_zyx, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"preseed_matrix_zyx must have shape (3, 3), got {matrix.shape}")
    return matrix.astype(np.float32), translation.astype(np.float32)


def scaled_local_initial_model(
    *,
    full_matrix_zyx: np.ndarray,
    full_translation_zyx: np.ndarray,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    full_shape_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
    fit_downsample_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_matrix, local_translation = full_model_to_local(
        full_matrix=full_matrix_zyx,
        full_translation=full_translation_zyx,
        fixed_start_zyx=fixed_start_zyx,
        moving_start_zyx=moving_start_zyx,
        full_shape_zyx=full_shape_zyx,
        crop_shape_zyx=crop_shape_zyx,
    )
    fit_matrix, fit_translation = _model_to_fit_downsample(local_matrix, local_translation, fit_downsample_zyx)
    return local_matrix, local_translation, fit_matrix.astype(np.float32), fit_translation.astype(np.float32)


def _warp_fit_preseed_cupy(moving_fit_gpu: Any, fit_matrix: np.ndarray, fit_translation: np.ndarray) -> Any:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform

    pull_matrix, pull_offset = output_to_input_from_model(
        np.asarray(fit_matrix, dtype=np.float32),
        np.asarray(fit_translation, dtype=np.float32),
        tuple(int(value) for value in moving_fit_gpu.shape),
    )
    return gpu_affine_transform(
        moving_fit_gpu,
        cp.asarray(pull_matrix, dtype=cp.float32),
        cp.asarray(pull_offset, dtype=cp.float32),
        output_shape=moving_fit_gpu.shape,
        order=1,
        mode="constant",
        cval=0.0,
    )


def _box_corners_zyx(min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [z, y, x]
            for z in (float(min_corner[0]), float(max_corner[0]))
            for y in (float(min_corner[1]), float(max_corner[1]))
            for x in (float(min_corner[2]), float(max_corner[2]))
        ],
        dtype=np.float64,
    )


def _moving_crop_start_unclipped(
    *,
    fixed_start_zyx: np.ndarray,
    crop_shape_zyx: tuple[int, int, int],
    full_matrix: np.ndarray,
    full_translation: np.ndarray,
    fixed_shape_zyx: np.ndarray,
    moving_shape_zyx: np.ndarray,
) -> np.ndarray:
    crop = np.asarray(crop_shape_zyx, dtype=np.float64)
    fixed_center = np.asarray(fixed_start_zyx, dtype=np.float64) + (crop - 1.0) / 2.0
    fixed_full_center = (np.asarray(fixed_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moving_full_center = (np.asarray(moving_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moving_center = moving_full_center + np.linalg.inv(np.asarray(full_matrix, dtype=np.float64)) @ (
        fixed_center - fixed_full_center - np.asarray(full_translation, dtype=np.float64)
    )
    return np.rint(moving_center - (crop - 1.0) / 2.0).astype(np.int64)


def _clip_fixed_crop_to_prior_overlap(
    *,
    fixed_start_zyx: np.ndarray,
    fixed_stop_zyx: np.ndarray,
    fixed_shape_zyx: np.ndarray,
    moving_shape_zyx: np.ndarray,
    full_matrix_zyx: np.ndarray,
    full_translation_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pull_matrix, pull_offset = output_to_input_from_model(
        np.asarray(full_matrix_zyx, dtype=np.float64),
        np.asarray(full_translation_zyx, dtype=np.float64),
        tuple(int(value) for value in fixed_shape_zyx),
    )
    matrix = np.asarray(pull_matrix, dtype=np.float64)
    offset = np.asarray(pull_offset, dtype=np.float64)
    fixed_min = np.maximum(np.asarray(fixed_start_zyx, dtype=np.int64), 0).astype(np.float64)
    fixed_max = np.minimum(
        np.asarray(fixed_stop_zyx, dtype=np.int64),
        np.asarray(fixed_shape_zyx, dtype=np.int64),
    ).astype(np.float64) - 1.0
    if np.any(fixed_max < fixed_min):
        raise ValueError(f"empty fixed crop after fixed-domain clipping: start={fixed_start_zyx}, stop={fixed_stop_zyx}")

    original_mapped = (matrix @ _box_corners_zyx(fixed_min, fixed_max).T).T + offset
    mapped_min = np.min(original_mapped, axis=0)
    mapped_max = np.max(original_mapped, axis=0)
    moving_min = np.maximum(np.floor(mapped_min), 0.0)
    moving_max = np.minimum(np.ceil(mapped_max), np.asarray(moving_shape_zyx, dtype=np.float64) - 1.0)

    inverse = np.linalg.inv(matrix)
    backprojected = (inverse @ (_box_corners_zyx(moving_min, moving_max) - offset).T).T
    fixed_min = np.maximum(fixed_min, np.ceil(np.min(backprojected, axis=0)))
    fixed_max = np.minimum(fixed_max, np.floor(np.max(backprojected, axis=0)))

    moving_lower = np.zeros(3, dtype=np.float64)
    moving_upper = np.asarray(moving_shape_zyx, dtype=np.float64) - 1.0
    shrink_steps: list[dict[str, Any]] = []
    for _iteration in range(32):
        mapped = (matrix @ _box_corners_zyx(fixed_min, fixed_max).T).T + offset
        low_violation = moving_lower - np.min(mapped, axis=0)
        high_violation = np.max(mapped, axis=0) - moving_upper
        if np.all(low_violation <= 1e-6) and np.all(high_violation <= 1e-6):
            break
        changed = False
        for moving_axis in range(3):
            if low_violation[moving_axis] > 1e-6:
                coeffs = matrix[moving_axis]
                fixed_axis = int(np.argmax(np.abs(coeffs)))
                coeff = float(coeffs[fixed_axis])
                delta = int(np.ceil(float(low_violation[moving_axis]) / max(abs(coeff), 1e-6)))
                if coeff >= 0.0:
                    fixed_min[fixed_axis] += delta
                    side = "min"
                else:
                    fixed_max[fixed_axis] -= delta
                    side = "max"
                shrink_steps.append({"moving_axis": int(moving_axis), "violation": "low", "fixed_axis": fixed_axis, "side": side, "delta": delta})
                changed = True
            if high_violation[moving_axis] > 1e-6:
                coeffs = matrix[moving_axis]
                fixed_axis = int(np.argmax(np.abs(coeffs)))
                coeff = float(coeffs[fixed_axis])
                delta = int(np.ceil(float(high_violation[moving_axis]) / max(abs(coeff), 1e-6)))
                if coeff >= 0.0:
                    fixed_max[fixed_axis] -= delta
                    side = "max"
                else:
                    fixed_min[fixed_axis] += delta
                    side = "min"
                shrink_steps.append({"moving_axis": int(moving_axis), "violation": "high", "fixed_axis": fixed_axis, "side": side, "delta": delta})
                changed = True
        if not changed:
            break
        if np.any(fixed_max < fixed_min):
            raise ValueError("fixed overlap support became empty while shrinking to moving domain")

    clipped_start = fixed_min.astype(np.int64)
    clipped_stop = (fixed_max + 1.0).astype(np.int64)
    final_mapped = (matrix @ _box_corners_zyx(clipped_start.astype(np.float64), clipped_stop.astype(np.float64) - 1.0).T).T + offset
    final_min = np.min(final_mapped, axis=0)
    final_max = np.max(final_mapped, axis=0)
    if np.any(final_min < -1e-5) or np.any(final_max > np.asarray(moving_shape_zyx, dtype=np.float64) - 1.0 + 1e-5):
        raise ValueError(f"invalid clipped support: mapped_min={final_min.tolist()} mapped_max={final_max.tolist()}")
    metadata = {
        "method": "fixed_crop_clipped_to_prior_pull_overlap_support",
        "original_fixed_start_zyx": np.asarray(fixed_start_zyx, dtype=np.int64).tolist(),
        "original_fixed_stop_zyx": np.asarray(fixed_stop_zyx, dtype=np.int64).tolist(),
        "mapped_original_moving_min_zyx": mapped_min.tolist(),
        "mapped_original_moving_max_zyx": mapped_max.tolist(),
        "intersected_moving_min_zyx": moving_min.tolist(),
        "intersected_moving_max_zyx": moving_max.tolist(),
        "clipped_fixed_start_zyx": clipped_start.tolist(),
        "clipped_fixed_stop_zyx": clipped_stop.tolist(),
        "clipped_shape_zyx": (clipped_stop - clipped_start).tolist(),
        "mapped_clipped_moving_min_zyx": final_min.tolist(),
        "mapped_clipped_moving_max_zyx": final_max.tolist(),
        "shrink_steps": shrink_steps,
    }
    return clipped_start, clipped_stop, metadata


def _window_path(output_dir: Path, tile_name: str, quadrant: str, z_start: int) -> Path:
    return output_dir / "window_json" / f"{_safe_stem(tile_name)}.{quadrant}.z{int(z_start):05d}.json"


def _cache_config_from_args(args: argparse.Namespace, *, preseed_matrix_zyx: np.ndarray) -> dict[str, Any]:
    return {
        "fixed_position": str(Path(args.fixed_position).resolve()),
        "moving_position": str(Path(args.moving_position).resolve()),
        "fixed_channel": int(args.fixed_channel),
        "moving_channel": int(args.moving_channel),
        "core_shape_zyx": [int(value) for value in args.core_shape_zyx],
        "window_shape_zyx": [int(value) for value in args.window_shape_zyx],
        "fit_downsample_zyx": [int(value) for value in args.fit_downsample_zyx],
        "preseed_matrix_zyx": np.asarray(preseed_matrix_zyx, dtype=np.float64).tolist(),
        "native_lib_dir": str(Path(args.native_lib_dir).resolve()),
        "ftol": float(args.ftol),
        "max_iterations": int(args.max_iterations),
        "min_corr": float(args.min_corr),
        "min_grad_ncc": float(args.min_grad_ncc),
        "empty_precheck_version": EMPTY_PRECHECK_VERSION,
        "empty_precheck_level": int(args.empty_precheck_level),
        "empty_precheck_min_dynamic_range": float(args.empty_precheck_min_dynamic_range),
        "empty_precheck_min_std": float(args.empty_precheck_min_std),
        "fixed_mask_threshold": None if args.fixed_mask_threshold is None else float(args.fixed_mask_threshold),
        "fixed_mask_level": int(args.fixed_mask_level),
        "fixed_mask_min_voxels": int(args.fixed_mask_min_voxels),
        "fixed_mask_max_masked_fraction": float(args.fixed_mask_max_masked_fraction),
    }


def _is_resumable_row(row: dict[str, Any], *, cache_config: dict[str, Any]) -> bool:
    precheck = row.get("empty_precheck")
    if row.get("status") == "error" or not isinstance(precheck, dict):
        return False
    if precheck.get("version") not in RESUMABLE_EMPTY_PRECHECK_VERSIONS:
        return False
    if row.get("cache_config") != cache_config:
        return False
    if row.get("status") == "accepted" and row.get("native_return_code") is None:
        return False
    if cache_config.get("fixed_mask_threshold") is None:
        return row.get("quality_mask") is None and row.get("fixed_threshold_mask") is None
    return row.get("quality_mask") == "fixed_threshold_mask" or row.get("rejection_reason") in {
        "fixed_threshold_mask_empty",
        "fixed_threshold_fit_mask_empty",
        "fixed_threshold_mask_too_masked",
        "fixed_threshold_fit_mask_too_masked",
    }


def _level_bounds_from_level0(
    *,
    start_zyx: np.ndarray,
    stop_zyx: np.ndarray,
    level_shape_zyx: np.ndarray,
    level0_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = np.maximum(
        1,
        np.rint(np.asarray(level0_shape_zyx, dtype=np.float64) / np.asarray(level_shape_zyx, dtype=np.float64)).astype(np.int64),
    )
    start = np.floor(np.asarray(start_zyx, dtype=np.float64) / factor.astype(np.float64)).astype(np.int64)
    stop = np.ceil(np.asarray(stop_zyx, dtype=np.float64) / factor.astype(np.float64)).astype(np.int64)
    start = np.clip(start, 0, np.asarray(level_shape_zyx, dtype=np.int64) - 1)
    stop = np.clip(stop, start + 1, np.asarray(level_shape_zyx, dtype=np.int64))
    return start, stop, factor


def _read_level_crop_from_level0_bounds(
    tile: Any,
    *,
    channel: int,
    level: int,
    start_zyx: np.ndarray,
    stop_zyx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    array, source_level, available_levels, store = _open_tile_level_array(tile.path, source_level=int(level))
    try:
        if tile.axes == "CZYX":
            source_shape_zyx = np.asarray(array.shape[1:4], dtype=np.int64)
        elif tile.axes == "ZYX":
            channels = flattened_channel_count(tile.path)
            if int(array.shape[0]) % channels:
                raise ValueError(f"{tile.path} has {array.shape[0]} planes, not divisible by channels={channels}")
            source_shape_zyx = np.asarray(array.shape, dtype=np.int64)
            source_shape_zyx[0] = int(source_shape_zyx[0]) // channels
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        level_start, level_stop, factor = _level_bounds_from_level0(
            start_zyx=start_zyx,
            stop_zyx=stop_zyx,
            level_shape_zyx=source_shape_zyx,
            level0_shape_zyx=np.asarray(tile.shape_zyx, dtype=np.int64),
        )
        raw_slices: list[slice] = []
        reverse_axes: list[int] = []
        for axis, (start, stop) in enumerate(zip(level_start, level_stop, strict=True)):
            raw_slice, reverse = raw_axis_slice_for_oriented_slice(
                slice(int(start), int(stop)),
                axis_size=int(source_shape_zyx[axis]),
                flipped=bool(tile.scale_zyx_um[axis] < 0),
            )
            raw_slices.append(raw_slice)
            if reverse:
                reverse_axes.append(axis)
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            patch = array[(channel, raw_slices[0], raw_slices[1], raw_slices[2])]
        else:
            channels = flattened_channel_count(tile.path)
            if channel < 0 or channel >= channels:
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {channels}")
            if channels == 1:
                patch = array[(raw_slices[0], raw_slices[1], raw_slices[2])]
            else:
                z_indices = np.arange(raw_slices[0].start, raw_slices[0].stop, dtype=np.int64)
                patch = array[(z_indices * channels + channel, raw_slices[1], raw_slices[2])]
        result = np.asarray(patch.compute(), dtype=np.float32)
        for axis in reverse_axes:
            result = np.flip(result, axis=axis)
        return result, {
            "requested_level": int(level),
            "source_level": int(source_level),
            "available_levels": int(available_levels),
            "level_shape_zyx": source_shape_zyx.tolist(),
            "effective_factor_zyx": factor.tolist(),
            "level_start_zyx": level_start.tolist(),
            "level_stop_zyx": level_stop.tolist(),
            "shape_zyx": list(result.shape),
        }
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _fixed_threshold_mask_stats(mask_gpu: Any) -> dict[str, float | int]:
    import cupy as cp

    unmasked_count = int(cp.asnumpy(cp.sum(mask_gpu)))
    total = int(mask_gpu.size)
    unmasked_fraction = float(unmasked_count / total) if total else 0.0
    return {
        "voxel_count": unmasked_count,
        "total_voxels": total,
        "unmasked_fraction": unmasked_fraction,
        "masked_fraction": float(1.0 - unmasked_fraction) if total else 0.0,
    }


def _block_any_downsample_zyx_cupy(mask_gpu: Any, factors_zyx: tuple[int, int, int]) -> Any:
    import cupy as cp
    from cucim.skimage.measure import block_reduce

    values = cp.ascontiguousarray(cp.asarray(mask_gpu, dtype=cp.bool_))
    factors = tuple(int(value) for value in factors_zyx)
    if values.ndim != 3:
        raise ValueError(f"Expected 3D zyx mask, got shape {values.shape}")
    if factors == (1, 1, 1):
        return values
    if any(size % factor != 0 for size, factor in zip(values.shape, factors, strict=True)):
        raise ValueError(f"Mask shape {values.shape} must be divisible by fit_downsample_zyx={factors}")
    return cp.ascontiguousarray(block_reduce(values, block_size=factors, func=cp.max).astype(cp.bool_, copy=False))


def _resize_bool_mask_nearest_cupy(mask_gpu: Any, target_shape_zyx: tuple[int, int, int]) -> Any:
    import cupy as cp

    values = cp.asarray(mask_gpu, dtype=cp.bool_)
    target = tuple(int(value) for value in target_shape_zyx)
    if values.ndim != 3:
        raise ValueError(f"Expected 3D zyx mask, got shape {values.shape}")
    if values.shape == target:
        return cp.ascontiguousarray(values)
    indices = [
        cp.minimum(
            cp.floor(cp.arange(size, dtype=cp.float32) * (int(source) / float(size))).astype(cp.int64),
            int(source) - 1,
        )
        for size, source in zip(target, values.shape, strict=True)
    ]
    resized = cp.take(values, indices[0], axis=0)
    resized = cp.take(resized, indices[1], axis=1)
    resized = cp.take(resized, indices[2], axis=2)
    return cp.ascontiguousarray(resized.astype(cp.bool_, copy=False))


def _raw_empty_stats(volume: np.ndarray) -> dict[str, float]:
    values = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return {
            "finite_fraction": 0.0,
            "p01": 0.0,
            "p50": 0.0,
            "p99": 0.0,
            "dynamic_range_p99_p01": 0.0,
            "std": 0.0,
        }
    finite_values = values[finite]
    p01, p50, p99 = np.percentile(finite_values, [1.0, 50.0, 99.0])
    return {
        "finite_fraction": float(np.count_nonzero(finite) / values.size),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "dynamic_range_p99_p01": float(p99 - p01),
        "std": float(np.std(finite_values)),
    }


def _raw_empty_stats_gpu(volume: np.ndarray) -> dict[str, float]:
    import cupy as cp

    values = cp.asarray(volume, dtype=cp.float32)
    finite = cp.isfinite(values)
    finite_count = int(cp.asnumpy(cp.count_nonzero(finite)))
    if finite_count == 0:
        return {
            "finite_fraction": 0.0,
            "p01": 0.0,
            "p50": 0.0,
            "p99": 0.0,
            "dynamic_range_p99_p01": 0.0,
            "std": 0.0,
        }
    finite_values = values[finite]
    percentiles = cp.percentile(finite_values, cp.asarray([1.0, 50.0, 99.0], dtype=cp.float32))
    std = cp.std(finite_values)
    p01, p50, p99 = [float(value) for value in cp.asnumpy(percentiles)]
    output = {
        "finite_fraction": float(finite_count / values.size),
        "p01": p01,
        "p50": p50,
        "p99": p99,
        "dynamic_range_p99_p01": float(p99 - p01),
        "std": float(cp.asnumpy(std)),
    }
    return output


def _obviously_empty(stats: dict[str, float], *, min_dynamic_range: float, min_std: float) -> bool:
    return bool(
        stats["finite_fraction"] <= 0.0
        or (stats["dynamic_range_p99_p01"] < min_dynamic_range and stats["std"] < min_std)
    )


def _build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed_payload = json.loads(Path(args.fixed_position).read_text())
    moving_payload = json.loads(Path(args.moving_position).read_text())
    moving_by_number = {_tile_number(str(record["tile"])): record for record in moving_payload["tiles"]}
    tile_filter = None
    if args.tile_filter:
        tile_filter = {part.strip().zfill(3) for part in args.tile_filter.split(",") if part.strip()}

    tasks: list[dict[str, Any]] = []
    cached: list[dict[str, Any]] = []
    devices = _parse_devices(args.devices)
    preseed_matrix = _parse_matrix_zyx(args.preseed_matrix_zyx)
    cache_config = _cache_config_from_args(args, preseed_matrix_zyx=preseed_matrix)
    task_index = 0
    for fixed_record in fixed_payload["tiles"]:
        tile_no = _tile_number(str(fixed_record["tile"]))
        if tile_filter is not None and tile_no not in tile_filter:
            continue
        moving_record = moving_by_number.get(tile_no)
        if moving_record is None:
            raise ValueError(f"moving position file is missing tile number {tile_no}")
        fixed_tile = tile_record_from_position_record(fixed_record)
        moving_tile = tile_record_from_position_record(moving_record)
        if tuple(fixed_tile.shape_zyx.tolist()) != tuple(moving_tile.shape_zyx.tolist()):
            raise ValueError(
                f"fixed/moving logical tile shapes differ for {fixed_record['tile']}: "
                f"{fixed_tile.shape_zyx.tolist()} vs {moving_tile.shape_zyx.tolist()}"
            )
        windows = quadrant_z_windows(
            tuple(int(value) for value in fixed_tile.shape_zyx),
            core_shape_zyx=args.core_shape_zyx,
            window_shape_zyx=args.window_shape_zyx,
        )
        for window in windows:
            path = _window_path(args.output_dir, str(fixed_record["tile"]), window["quadrant"], window["fixed_start_zyx"][0])
            if args.resume and path.exists():
                row = json.loads(path.read_text())
                if _is_resumable_row(row, cache_config=cache_config):
                    cached.append(row)
                    continue
            tasks.append(
                {
                    "task_index": task_index,
                    "fixed_tile": fixed_tile,
                    "moving_tile": moving_tile,
                    "fixed_channel": int(args.fixed_channel),
                    "moving_channel": int(args.moving_channel),
                    "preseed_matrix_zyx": preseed_matrix.tolist(),
                    "window": window,
                    "fit_downsample_zyx": list(args.fit_downsample_zyx),
                    "native_lib_dir": str(args.native_lib_dir),
                    "ftol": float(args.ftol),
                    "max_iterations": int(args.max_iterations),
                    "min_corr": float(args.min_corr),
                    "min_grad_ncc": float(args.min_grad_ncc),
                    "empty_precheck_level": int(args.empty_precheck_level),
                    "empty_precheck_min_dynamic_range": float(args.empty_precheck_min_dynamic_range),
                    "empty_precheck_min_std": float(args.empty_precheck_min_std),
                    "fixed_mask_threshold": None if args.fixed_mask_threshold is None else float(args.fixed_mask_threshold),
                    "fixed_mask_level": int(args.fixed_mask_level),
                    "fixed_mask_min_voxels": int(args.fixed_mask_min_voxels),
                    "fixed_mask_max_masked_fraction": float(args.fixed_mask_max_masked_fraction),
                    "device": int(devices[task_index % len(devices)]),
                    "output_path": str(path),
                    "cache_config": cache_config,
                }
            )
            task_index += 1
    if args.max_windows is not None:
        tasks = tasks[: int(args.max_windows)]
    return tasks, cached


def _measure_window(task: dict[str, Any]) -> dict[str, Any]:
    physical_device = int(task["device"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    started = time.perf_counter()
    fixed_tile = task["fixed_tile"]
    moving_tile = task["moving_tile"]
    window = task["window"]
    original_fixed_start = np.asarray(window["fixed_start_zyx"], dtype=np.int64)
    original_fixed_stop = np.asarray(window["fixed_stop_zyx"], dtype=np.int64)
    full_matrix, full_translation = preseeded_level0_model(
        fixed_tile=fixed_tile,
        moving_tile=moving_tile,
        preseed_matrix_zyx=np.asarray(task["preseed_matrix_zyx"], dtype=np.float64),
    )
    initialization_source = str(task.get("initialization_source", "preseed"))
    if task.get("override_full_matrix_zyx") is not None or task.get("override_full_translation_zyx") is not None:
        full_matrix = np.asarray(task["override_full_matrix_zyx"], dtype=np.float32)
        full_translation = np.asarray(task["override_full_translation_zyx"], dtype=np.float32)
    fixed_start, fixed_stop, overlap_support = _clip_fixed_crop_to_prior_overlap(
        fixed_start_zyx=original_fixed_start,
        fixed_stop_zyx=original_fixed_stop,
        fixed_shape_zyx=fixed_tile.shape_zyx,
        moving_shape_zyx=moving_tile.shape_zyx,
        full_matrix_zyx=full_matrix,
        full_translation_zyx=full_translation,
    )
    crop = fixed_stop - fixed_start
    if np.any(crop < np.asarray([16, 64, 64], dtype=np.int64)):
        raise ValueError(f"clipped overlap crop is too small for method8: shape={crop.tolist()}")
    crop_shape = tuple(int(value) for value in crop)
    moving_start_unclipped = _moving_crop_start_unclipped(
        fixed_start_zyx=fixed_start,
        crop_shape_zyx=crop_shape,
        full_matrix=full_matrix,
        full_translation=full_translation,
        fixed_shape_zyx=fixed_tile.shape_zyx,
        moving_shape_zyx=moving_tile.shape_zyx,
    )
    moving_start = moving_crop_start_for_fixed_crop(
        fixed_start_zyx=fixed_start,
        crop_shape_zyx=crop_shape,
        full_matrix=full_matrix,
        full_translation=full_translation,
        fixed_shape_zyx=fixed_tile.shape_zyx,
        moving_shape_zyx=moving_tile.shape_zyx,
    )
    overlap_support["moving_start_unclipped_zyx"] = moving_start_unclipped.tolist()
    overlap_support["moving_start_zyx"] = moving_start.tolist()
    overlap_support["moving_start_was_clipped"] = bool(not np.array_equal(moving_start_unclipped, moving_start))
    local_matrix, local_translation, fit_matrix, fit_translation = scaled_local_initial_model(
        full_matrix_zyx=full_matrix,
        full_translation_zyx=full_translation,
        fixed_start_zyx=fixed_start,
        moving_start_zyx=moving_start,
        full_shape_zyx=fixed_tile.shape_zyx,
        crop_shape_zyx=crop,
        fit_downsample_zyx=tuple(int(value) for value in task["fit_downsample_zyx"]),
    )
    row: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "lightsheet.image10_image14_tile_quadrant_method8_window.v1",
        "fixed_tile": fixed_tile.tile,
        "moving_tile": moving_tile.tile,
        "fixed_path": str(fixed_tile.path),
        "moving_path": str(moving_tile.path),
        "fixed_channel": int(task["fixed_channel"]),
        "moving_channel": int(task["moving_channel"]),
        "quadrant": window["quadrant"],
        "fixed_start_zyx": fixed_start.tolist(),
        "fixed_stop_zyx": (fixed_start + crop).tolist(),
        "moving_start_zyx": moving_start.tolist(),
        "moving_stop_zyx": (moving_start + crop).tolist(),
        "window_shape_zyx": list(crop_shape),
        "requested_fixed_start_zyx": original_fixed_start.tolist(),
        "requested_fixed_stop_zyx": original_fixed_stop.tolist(),
        "requested_window_shape_zyx": list(window["window_shape_zyx"]),
        "overlap_support": overlap_support,
        "fit_downsample_zyx": [int(value) for value in task["fit_downsample_zyx"]],
        "full_preseed_matrix_zyx": np.asarray(full_matrix, dtype=np.float64).tolist(),
        "full_preseed_translation_zyx": np.asarray(full_translation, dtype=np.float64).tolist(),
        "local_preseed_matrix_zyx": np.asarray(local_matrix, dtype=np.float64).tolist(),
        "local_preseed_translation_zyx": np.asarray(local_translation, dtype=np.float64).tolist(),
        "fit_preseed_matrix_zyx": np.asarray(fit_matrix, dtype=np.float64).tolist(),
        "fit_preseed_translation_zyx": np.asarray(fit_translation, dtype=np.float64).tolist(),
        "native_initial_matrix_xyz_3x4": zyx_to_xyz_3x4(fit_matrix, fit_translation).tolist(),
        "initialization_source": initialization_source,
        "interpolation_prior": task.get("interpolation_prior"),
        "device": physical_device,
        "cache_config": task["cache_config"],
        "empty_precheck": {
            "version": EMPTY_PRECHECK_VERSION,
            "level": int(task["empty_precheck_level"]),
            "min_dynamic_range": float(task["empty_precheck_min_dynamic_range"]),
            "min_std": float(task["empty_precheck_min_std"]),
            "fixed_mask_threshold": task.get("fixed_mask_threshold"),
            "fixed_mask_level": int(task.get("fixed_mask_level", 2)),
            "fixed_mask_min_voxels": int(task.get("fixed_mask_min_voxels", 256)),
            "fixed_mask_max_masked_fraction": float(task.get("fixed_mask_max_masked_fraction", 0.95)),
            "status": "disabled" if int(task["empty_precheck_level"]) < 0 else "not_run",
        },
    }
    try:
        if int(task["empty_precheck_level"]) >= 0:
            precheck_started = time.perf_counter()
            precheck_read_started = time.perf_counter()
            fixed_precheck, fixed_precheck_source = _read_level_crop_from_level0_bounds(
                fixed_tile,
                channel=int(task["fixed_channel"]),
                level=int(task["empty_precheck_level"]),
                start_zyx=fixed_start,
                stop_zyx=fixed_start + crop,
            )
            moving_precheck, moving_precheck_source = _read_level_crop_from_level0_bounds(
                moving_tile,
                channel=int(task["moving_channel"]),
                level=int(task["empty_precheck_level"]),
                start_zyx=moving_start,
                stop_zyx=moving_start + crop,
            )
            precheck_read_seconds = time.perf_counter() - precheck_read_started
            precheck_stats_started = time.perf_counter()
            fixed_precheck_stats = _raw_empty_stats_gpu(fixed_precheck)
            moving_precheck_stats = _raw_empty_stats_gpu(moving_precheck)
            precheck_stats_seconds = time.perf_counter() - precheck_stats_started
            row["empty_precheck"].update(
                {
                    "status": "checked",
                    "fixed_source": fixed_precheck_source,
                    "moving_source": moving_precheck_source,
                    "fixed_stats": fixed_precheck_stats,
                    "moving_stats": moving_precheck_stats,
                    "read_seconds": precheck_read_seconds,
                    "stats_seconds": precheck_stats_seconds,
                    "timing_seconds": time.perf_counter() - precheck_started,
                }
            )
            min_dynamic_range = float(task["empty_precheck_min_dynamic_range"])
            min_std = float(task["empty_precheck_min_std"])
            if _obviously_empty(fixed_precheck_stats, min_dynamic_range=min_dynamic_range, min_std=min_std):
                row.update({"status": "rejected", "rejection_reason": "fixed_level2_flat_or_empty"})
                return row
            if _obviously_empty(moving_precheck_stats, min_dynamic_range=min_dynamic_range, min_std=min_std):
                row.update({"status": "rejected", "rejection_reason": "moving_level2_flat_or_empty"})
                return row

        fit_downsample = tuple(int(value) for value in task["fit_downsample_zyx"])
        fixed_fit_mask_gpu = None
        fixed_mask_threshold = task.get("fixed_mask_threshold")
        if fixed_mask_threshold is not None:
            import cupy as cp

            mask_started = time.perf_counter()
            fixed_mask_source, fixed_mask_source_info = _read_level_crop_from_level0_bounds(
                fixed_tile,
                channel=int(task["fixed_channel"]),
                level=int(task.get("fixed_mask_level", 2)),
                start_zyx=fixed_start,
                stop_zyx=fixed_start + crop,
            )
            fixed_mask_source_gpu = cp.asarray(fixed_mask_source, dtype=cp.float32)
            fixed_mask_level_gpu = cp.ascontiguousarray(fixed_mask_source_gpu > cp.float32(float(fixed_mask_threshold)))
            fixed_mask_stats = _fixed_threshold_mask_stats(fixed_mask_level_gpu)
            fit_shape = tuple(int(crop_axis // factor) for crop_axis, factor in zip(crop_shape, fit_downsample, strict=True))
            fixed_fit_mask_gpu = _resize_bool_mask_nearest_cupy(fixed_mask_level_gpu, fit_shape)
            fit_mask_stats = _fixed_threshold_mask_stats(fixed_fit_mask_gpu)
            row["fixed_threshold_mask"] = {
                "source": f"fixed_level{int(fixed_mask_source_info['source_level'])}",
                "threshold": float(fixed_mask_threshold),
                "source_info": fixed_mask_source_info,
                **fixed_mask_stats,
                "fit_downsample_zyx": [int(value) for value in fit_downsample],
                "fit_voxel_count": int(fit_mask_stats["voxel_count"]),
                "fit_total_voxels": int(fit_mask_stats["total_voxels"]),
                "fit_masked_fraction": float(fit_mask_stats["masked_fraction"]),
                "fit_unmasked_fraction": float(fit_mask_stats["unmasked_fraction"]),
                "timing_seconds": time.perf_counter() - mask_started,
            }
            if float(fixed_mask_stats["masked_fraction"]) > float(task.get("fixed_mask_max_masked_fraction", 0.95)):
                row.update({"status": "rejected", "rejection_reason": "fixed_threshold_mask_too_masked"})
                return row
            if float(fit_mask_stats["masked_fraction"]) > float(task.get("fixed_mask_max_masked_fraction", 0.95)):
                row.update({"status": "rejected", "rejection_reason": "fixed_threshold_fit_mask_too_masked"})
                return row
            if int(fit_mask_stats["voxel_count"]) < int(task.get("fixed_mask_min_voxels", 256)):
                row.update({"status": "rejected", "rejection_reason": "fixed_threshold_fit_mask_empty"})
                return row

        read_started = time.perf_counter()
        fixed_raw, fixed_slices = _read_level0_crop(
            fixed_tile,
            channel=int(task["fixed_channel"]),
            start_zyx=fixed_start,
            crop_shape_zyx=crop_shape,
        )
        moving_raw, moving_slices = _read_level0_crop(
            moving_tile,
            channel=int(task["moving_channel"]),
            start_zyx=moving_start,
            crop_shape_zyx=crop_shape,
        )
        row["fixed_slices_zyx"] = fixed_slices
        row["moving_slices_zyx"] = moving_slices
        row["timing_seconds"] = {"read": time.perf_counter() - read_started}

        normalize_started = time.perf_counter()
        fixed_gpu, fixed_stats = _robust_norm_and_content_stats_cupy(fixed_raw)
        moving_gpu, moving_stats = _robust_norm_and_content_stats_cupy(moving_raw)
        row["fixed_content"] = fixed_stats
        row["moving_content"] = moving_stats
        row["timing_seconds"]["normalize_and_content"] = time.perf_counter() - normalize_started
        reason = _native_window_low_content_reason(fixed_stats, prefix="fixed")
        reason = reason or _native_window_low_content_reason(moving_stats, prefix="moving")
        if reason is not None:
            row.update({"status": "rejected", "rejection_reason": reason})
            return row

        fit_started = time.perf_counter()
        fixed_fit_gpu = _block_mean_downsample_zyx_cupy(fixed_gpu, fit_downsample)
        moving_fit_gpu = _block_mean_downsample_zyx_cupy(moving_gpu, fit_downsample)
        attempts: list[dict[str, Any]] = []

        def run_method8_attempt(attempt_name: str, attempt_translation: np.ndarray) -> dict[str, Any]:
            attempt_started = time.perf_counter()
            native = register_method8_device(
                fixed_fit_gpu,
                moving_fit_gpu,
                lib_dir=Path(task["native_lib_dir"]),
                ftol=float(task["ftol"]),
                max_iterations=int(task["max_iterations"]),
                device=0,
                initial_matrix_xyz_3x4=zyx_to_xyz_3x4(fit_matrix, attempt_translation),
            )
            native_full_matrix, native_full_offset = _native_pull_from_fit_downsample(
                native.matrix_zyx,
                native.offset_zyx,
                fit_downsample,
            )
            refined_local_matrix, refined_local_translation = output_to_input_to_model(
                native_full_matrix,
                native_full_offset,
                tuple(int(value) for value in fixed_gpu.shape),
            )
            refined_full_matrix, refined_full_translation = local_model_to_full(
                local_matrix=refined_local_matrix,
                local_translation=refined_local_translation,
                fixed_start_zyx=fixed_start,
                moving_start_zyx=moving_start,
                full_shape_zyx=fixed_tile.shape_zyx,
                crop_shape_zyx=crop,
            )
            registered = native.registered_zyx.astype(np.float32, copy=False)
            corr = _corr_gpu(fixed_fit_gpu, registered, fixed_mask=fixed_fit_mask_gpu)
            gradient = gradient_component_ncc_3d_gpu(fixed_fit_gpu, registered, fixed_mask=fixed_fit_mask_gpu)
            grad_mean = _gradient_component_ncc_mean(gradient)
            accepted = (
                int(native.return_code) == 0
                and corr is not None
                and np.isfinite(float(corr))
                and float(corr) >= float(task["min_corr"])
                and np.isfinite(grad_mean)
                and grad_mean >= float(task["min_grad_ncc"])
            )
            return {
                "name": attempt_name,
                "initial_fit_translation_zyx": np.asarray(attempt_translation, dtype=np.float64).tolist(),
                "status": "accepted" if accepted else "rejected",
                "rejection_reason": None if accepted else "quality_gate",
                "native_return_code": int(native.return_code),
                "corr_refined": None if corr is None else float(corr),
                "gradient_component_ncc_refined": gradient,
                "gradient_component_ncc_mean": float(grad_mean) if np.isfinite(grad_mean) else float("nan"),
                "native_output_to_input_matrix_zyx": np.asarray(native_full_matrix, dtype=np.float64).tolist(),
                "native_output_to_input_offset_zyx": np.asarray(native_full_offset, dtype=np.float64).tolist(),
                "native_fit_output_to_input_matrix_zyx": np.asarray(native.matrix_zyx, dtype=np.float64).tolist(),
                "native_fit_output_to_input_offset_zyx": np.asarray(native.offset_zyx, dtype=np.float64).tolist(),
                "local_matrix_zyx": np.asarray(refined_local_matrix, dtype=np.float64).tolist(),
                "local_translation_zyx": np.asarray(refined_local_translation, dtype=np.float64).tolist(),
                "full_matrix_zyx": np.asarray(refined_full_matrix, dtype=np.float64).tolist(),
                "full_translation_zyx": np.asarray(refined_full_translation, dtype=np.float64).tolist(),
                "registered_zyx": registered,
                "timing_seconds": time.perf_counter() - attempt_started,
            }

        baseline_attempt = run_method8_attempt("preseed", fit_translation)
        attempts.append({key: value for key, value in baseline_attempt.items() if key != "registered_zyx"})
        selected_attempt = baseline_attempt
        if baseline_attempt["status"] != "accepted" and baseline_attempt["rejection_reason"] == "quality_gate":
            phase_started = time.perf_counter()
            initial_registered = _warp_fit_preseed_cupy(moving_fit_gpu, fit_matrix, fit_translation)
            phase_translation = np.asarray(
                estimate_translation_gpu(
                    fixed_fit_gpu.copy(),
                    initial_registered.copy(),
                    upsample_factor=int(task.get("phase_priming_upsample_factor", 10)),
                ),
                dtype=np.float32,
            )
            phase_fit_translation = np.asarray(fit_translation, dtype=np.float32) + phase_translation
            row["phase_priming"] = {
                "source": "phase_cross_correlation_fixed_vs_initial_preseed_registered",
                "translation_delta_zyx": phase_translation.astype(np.float64).tolist(),
                "initial_fit_translation_zyx": np.asarray(fit_translation, dtype=np.float64).tolist(),
                "primed_fit_translation_zyx": phase_fit_translation.astype(np.float64).tolist(),
                "upsample_factor": int(task.get("phase_priming_upsample_factor", 10)),
                "timing_seconds": time.perf_counter() - phase_started,
            }
            phase_attempt = run_method8_attempt("phase_primed_preseed", phase_fit_translation)
            attempts.append({key: value for key, value in phase_attempt.items() if key != "registered_zyx"})
            if phase_attempt["status"] == "accepted" or float(phase_attempt["gradient_component_ncc_mean"]) > float(
                selected_attempt["gradient_component_ncc_mean"]
            ):
                selected_attempt = phase_attempt
        row["timing_seconds"]["fit"] = time.perf_counter() - fit_started
        row["method8_attempts"] = attempts
        row.update(
            {
                "status": selected_attempt["status"],
                "rejection_reason": selected_attempt["rejection_reason"],
                "native_return_code": selected_attempt["native_return_code"],
                "corr_refined": selected_attempt["corr_refined"],
                "quality_mask": "fixed_threshold_mask" if fixed_fit_mask_gpu is not None else None,
                "gradient_component_ncc_refined": selected_attempt["gradient_component_ncc_refined"],
                "native_output_to_input_matrix_zyx": selected_attempt["native_output_to_input_matrix_zyx"],
                "native_output_to_input_offset_zyx": selected_attempt["native_output_to_input_offset_zyx"],
                "native_fit_output_to_input_matrix_zyx": selected_attempt["native_fit_output_to_input_matrix_zyx"],
                "native_fit_output_to_input_offset_zyx": selected_attempt["native_fit_output_to_input_offset_zyx"],
                "local_matrix_zyx": selected_attempt["local_matrix_zyx"],
                "local_translation_zyx": selected_attempt["local_translation_zyx"],
                "full_matrix_zyx": selected_attempt["full_matrix_zyx"],
                "full_translation_zyx": selected_attempt["full_translation_zyx"],
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        row.setdefault("timing_seconds", {})["total"] = time.perf_counter() - started
    return row


def _measure_window_to_path(task: dict[str, Any]) -> None:
    row = _measure_window(task)
    _write_json(Path(task["output_path"]), row)


def _native_process_error_row(task: dict[str, Any], exitcode: int | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "lightsheet.image10_image14_tile_quadrant_method8_window.v1",
        "fixed_tile": task["fixed_tile"].tile,
        "moving_tile": task["moving_tile"].tile,
        "quadrant": task["window"]["quadrant"],
        "fixed_start_zyx": task["window"]["fixed_start_zyx"],
        "status": "error",
        "error": f"native_process_terminated: exitcode={exitcode}",
        "device": int(task["device"]),
        "cache_config": task.get("cache_config"),
        "empty_precheck": {
            "version": EMPTY_PRECHECK_VERSION,
            "level": int(task["empty_precheck_level"]),
            "min_dynamic_range": float(task["empty_precheck_min_dynamic_range"]),
            "min_std": float(task["empty_precheck_min_std"]),
            "fixed_mask_threshold": task.get("fixed_mask_threshold"),
            "fixed_mask_level": int(task.get("fixed_mask_level", 2)),
            "fixed_mask_min_voxels": int(task.get("fixed_mask_min_voxels", 256)),
            "fixed_mask_max_masked_fraction": float(task.get("fixed_mask_max_masked_fraction", 0.95)),
            "status": "unknown_process_terminated",
        },
    }


def _progress(row: dict[str, Any], *, cached: bool = False) -> str:
    gradient = row.get("gradient_component_ncc_refined")
    grad_mean = gradient.get("mean") if isinstance(gradient, dict) else None
    suffix = " cached=true" if cached else ""
    reason = row.get("rejection_reason") or row.get("error")
    reason_text = "" if reason is None else f" reason={reason}"
    corr = row.get("corr_refined")
    corr_text = "nan" if corr is None else f"{float(corr):.4f}"
    grad_text = "nan" if grad_mean is None else f"{float(grad_mean):.4f}"
    mask = row.get("fixed_threshold_mask")
    mask_text = ""
    if isinstance(mask, dict):
        masked = mask.get("masked_fraction")
        fit_masked = mask.get("fit_masked_fraction")
        if masked is not None:
            mask_text += f" masked={100.0 * float(masked):.4f}%"
        if fit_masked is not None:
            mask_text += f" fit_masked={100.0 * float(fit_masked):.4f}%"
    return (
        f"tile-quadrant-method8 {row.get('fixed_tile')} {row.get('quadrant')} "
        f"z={row.get('fixed_start_zyx', ['?'])[0]} status={row.get('status')} "
        f"device={row.get('device')} corr={corr_text} grad_ncc={grad_text}{mask_text}{reason_text}{suffix}"
    )


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    summary_path = output_dir / "tile_quadrant_method8_summary.json"
    grad_values = np.asarray(
        [row["gradient_component_ncc_refined"]["mean"] for row in accepted if isinstance(row.get("gradient_component_ncc_refined"), dict)],
        dtype=np.float64,
    )
    corr_values = np.asarray([row["corr_refined"] for row in accepted if row.get("corr_refined") is not None], dtype=np.float64)
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "artifact_type": "lightsheet.image10_image14_tile_quadrant_method8_summary.v1",
            "fixed_position": str(Path(args.fixed_position).resolve()),
            "moving_position": str(Path(args.moving_position).resolve()),
            "output_dir": str(output_dir.resolve()),
            "fixed_channel": int(args.fixed_channel),
            "moving_channel": int(args.moving_channel),
            "core_shape_zyx": [int(value) for value in args.core_shape_zyx],
            "window_shape_zyx": [int(value) for value in args.window_shape_zyx],
            "fit_downsample_zyx": [int(value) for value in args.fit_downsample_zyx],
            "preseed_matrix_zyx": _parse_matrix_zyx(args.preseed_matrix_zyx).tolist(),
            "native_lib_dir": str(Path(args.native_lib_dir).resolve()),
            "max_iterations": int(args.max_iterations),
            "ftol": float(args.ftol),
            "min_corr": float(args.min_corr),
            "min_grad_ncc": float(args.min_grad_ncc),
            "empty_precheck": {
                "version": EMPTY_PRECHECK_VERSION,
                "level": int(args.empty_precheck_level),
                "min_dynamic_range": float(args.empty_precheck_min_dynamic_range),
                "min_std": float(args.empty_precheck_min_std),
                "fixed_mask_threshold": args.fixed_mask_threshold,
                "fixed_mask_level": int(args.fixed_mask_level),
                "fixed_mask_min_voxels": int(args.fixed_mask_min_voxels),
                "fixed_mask_max_masked_fraction": float(args.fixed_mask_max_masked_fraction),
            },
            "workers": int(args.workers),
            "devices": list(_parse_devices(args.devices)),
            "resume": bool(args.resume),
            "aggregate": {
                "window_count": len(rows),
                "accepted_window_count": len(accepted),
                "rejected_window_count": sum(1 for row in rows if row.get("status") == "rejected"),
                "error_window_count": sum(1 for row in rows if row.get("status") == "error"),
                "corr_refined": None
                if corr_values.size == 0
                else {"median": float(np.median(corr_values)), "mean": float(np.mean(corr_values))},
                "gradient_component_ncc_mean": None
                if grad_values.size == 0
                else {"median": float(np.median(grad_values)), "mean": float(np.mean(grad_values))},
            },
            "windows": sorted(
                rows,
                key=lambda row: (
                    str(row.get("fixed_tile")),
                    str(row.get("quadrant")),
                    int(row.get("fixed_start_zyx", [0])[0]),
                ),
            ),
        },
    )
    return summary_path

def _require_gpu_env() -> None:
    missing = [name for name in ("CUDA_PATH", "LD_LIBRARY_PATH") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "missing required GPU/native environment variable(s): "
            + ", ".join(missing)
            + "; source conda, activate multi, and export CUDA_PATH/LD_LIBRARY_PATH as documented in lightsheet/README.md"
        )


def run(args: argparse.Namespace) -> Path:
    _require_gpu_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, cached_rows = _build_tasks(args)
    rows: list[dict[str, Any]] = list(cached_rows)
    for row in cached_rows:
        print(_progress(row, cached=True), flush=True)
    _write_summary(args.output_dir, rows, args)
    if args.workers == 1:
        for task in tasks:
            row = _measure_window(task)
            _write_json(Path(task["output_path"]), row)
            rows.append(row)
            _write_summary(args.output_dir, rows, args)
            print(_progress(row), flush=True)
    else:
        context = mp.get_context("spawn")
        pending = list(tasks)
        running: list[tuple[mp.Process, dict[str, Any]]] = []
        while pending or running:
            while pending and len(running) < int(args.workers):
                task = pending.pop(0)
                process = context.Process(target=_measure_window_to_path, args=(task,))
                process.start()
                running.append((process, task))
            time.sleep(0.2)
            still_running: list[tuple[mp.Process, dict[str, Any]]] = []
            for process, task in running:
                if process.is_alive():
                    still_running.append((process, task))
                    continue
                process.join()
                output_path = Path(task["output_path"])
                if process.exitcode == 0 and output_path.exists():
                    row = json.loads(output_path.read_text())
                else:
                    row = _native_process_error_row(task, process.exitcode)
                    _write_json(output_path, row)
                rows.append(row)
                _write_summary(args.output_dir, rows, args)
                print(_progress(row), flush=True)
            running = still_running
    return _write_summary(args.output_dir, rows, args).resolve()


def run_tile_quadrant_method8(
    *,
    fixed_position: Path,
    moving_position: Path,
    output_dir: Path,
    fixed_channel: int = 0,
    moving_channel: int = 0,
    core_shape_zyx: tuple[int, int, int] = (480, 480, 480),
    window_shape_zyx: tuple[int, int, int] = (528, 528, 528),
    fit_downsample_zyx: tuple[int, int, int] = (1, 1, 1),
    preseed_matrix_zyx: str | None = None,
    native_lib_dir: Path = DEFAULT_LIB_DIR,
    ftol: float = 1e-4,
    max_iterations: int = 300,
    min_corr: float = 0.15,
    min_grad_ncc: float = 0.24,
    empty_precheck_level: int = -1,
    empty_precheck_min_dynamic_range: float = 1.0,
    empty_precheck_min_std: float = 0.25,
    fixed_mask_threshold: float | None = 3000.0,
    fixed_mask_level: int = 2,
    fixed_mask_min_voxels: int = 256,
    fixed_mask_max_masked_fraction: float = 0.95,
    workers: int = 1,
    devices: str = "0",
    tile_filter: str | None = None,
    max_windows: int | None = None,
    resume: bool = True,
) -> Path:
    """Run quadrant/z Method8 registration without exposing argparse as the package API."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    args = argparse.Namespace(
        fixed_position=fixed_position,
        moving_position=moving_position,
        output_dir=output_dir,
        fixed_channel=fixed_channel,
        moving_channel=moving_channel,
        core_shape_zyx=core_shape_zyx,
        window_shape_zyx=window_shape_zyx,
        fit_downsample_zyx=fit_downsample_zyx,
        preseed_matrix_zyx=preseed_matrix_zyx,
        native_lib_dir=native_lib_dir,
        ftol=ftol,
        max_iterations=max_iterations,
        min_corr=min_corr,
        min_grad_ncc=min_grad_ncc,
        empty_precheck_level=empty_precheck_level,
        empty_precheck_min_dynamic_range=empty_precheck_min_dynamic_range,
        empty_precheck_min_std=empty_precheck_min_std,
        fixed_mask_threshold=fixed_mask_threshold,
        fixed_mask_level=fixed_mask_level,
        fixed_mask_min_voxels=fixed_mask_min_voxels,
        fixed_mask_max_masked_fraction=fixed_mask_max_masked_fraction,
        workers=workers,
        devices=devices,
        tile_filter=tile_filter,
        max_windows=max_windows,
        resume=resume,
    )
    return run(args)
