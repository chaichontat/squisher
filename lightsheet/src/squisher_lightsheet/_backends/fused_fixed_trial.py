#!/usr/bin/env python
from __future__ import annotations

import argparse
from functools import cache, lru_cache
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.local_affine import local_affine_cache_config
from squisher.jpegxr_zarr import register_jpegxr_codec
from squisher_lightsheet._legacy.stitch_20x_tl_multiview import SourceChunkCache


import zarr

from squisher_lightsheet.channel_affine import (
    _block_mean_downsample_zyx_cupy,
    _corr_gpu,
    _gradient_component_ncc_mean,
    _native_pull_from_fit_downsample,
    _native_window_low_content_reason,
    _read_level0_crop,
    _robust_norm_and_content_stats_cupy,
    gradient_component_ncc_3d_gpu,
    masked_structural_translation_gpu,
    moving_xy_center_slab_z_pivot_in_fit_zyx,
    output_to_input_from_model,
    output_to_input_to_model,
)
from squisher_lightsheet.native_reg3dgpu import (
    DEFAULT_LIB_DIR,
    register_method6_device,
    register_method10_mattes_device,
    register_method11_mattes_device,
    register_method8_device,
    zyx_to_xyz_3x4,
)
from squisher_lightsheet._legacy.rough_align_tltr_center_z_phase import tile_bounds_zyx_um
from squisher_lightsheet.tile_phase import _open_tile_level_array, raw_axis_slice_for_oriented_slice, tile_record_from_position_record


register_jpegxr_codec()


DEFAULT_FIXED_POSITION = Path(
    "/home/chaichontat/nvme/lightsheet/working/eduseg/Image_14/"
    "image10-ch0-to-image14-zarr-level2-tile-phase/Image_14.zarr.reference.positions.json"
)
DEFAULT_MOVING_POSITION = Path(
    "/home/chaichontat/nvme/lightsheet/working/eduseg/Image_14/"
    "image10-ch0-to-image14-zarr-level2-tile-phase/Image_10.ch0.aligned-to-Image_14.zarr.positions.json"
)
DEFAULT_FIXED_FUSED = Path(
    "/home/chaichontat/nvme/lightsheet/working/eduseg/Image_14/"
    "image10-ch0-to-image14-fixedaffine-phi0p2-shear-mean-phasecorr-z10-c128-tilecenter/"
    "Image_14.fused.samegrid.level0.ch0.ome.zarr"
)
DEFAULT_OUTPUT = Path(
    "/home/chaichontat/nvme/lightsheet/working/eduseg/Image_14/"
    "image10-ch0-to-image14-fused-fixed-method8-trial/tile042_qy0_qx0_z00480.json"
)
FUSED_FIXED_CACHE_VERSION = "fused_fixed_method8_v16_masked_structural_capture"
# Median polar rotation vector and median symmetric stretch from the 17 accepted
# slab-Z/full-XY Method 6 windows in the tile-061 and tile-081 trials.
DEFAULT_STARTING_AFFINE_MATRIX_ZYX = (
    0.9999090828,
    -0.0083501824,
    -0.0029757332,
    0.0141554887,
    0.9999284579,
    -0.0096124636,
    0.0026881526,
    0.0099598448,
    0.9999495898,
)
MAX_AFFINE_AXIS_SCALE = 1.1
MAX_AFFINE_CONDITION = 1.1
MIN_AFFINE_DETERMINANT = 1.0 / 1.1
MAX_AFFINE_DETERMINANT = 1.1
MATTES_NATIVE_METHODS = frozenset(("method10-mattes", "method11-mattes"))
NCC_AFFINE_NATIVE_METHODS = frozenset(("method6", "method8"))


@lru_cache(maxsize=1)
def _moving_source_cache(max_gib: float) -> SourceChunkCache | None:
    if not np.isfinite(max_gib) or max_gib < 0:
        raise ValueError("--source-cache-max-gib must be finite and nonnegative")
    if max_gib == 0:
        return None
    return SourceChunkCache(max_bytes=int(max_gib * 1024**3))


def _apply_native_method_defaults(args: argparse.Namespace) -> None:
    if str(getattr(args, "native_method", "")) == "method6" and getattr(args, "starting_affine_matrix_zyx", None) is None:
        args.starting_affine_matrix_zyx = DEFAULT_STARTING_AFFINE_MATRIX_ZYX


def _register_native_device_quiet(
    *args: Any,
    native_method: str,
    mattes_bins: int,
    mattes_samples: int,
    mattes_fixed_shear: bool,
    aff_pivot_zyx: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[Any, str | None]:
    """Capture native stderr; callers decide whether it affected the row."""
    saved_stderr = os.dup(2)
    read_fd, write_fd = os.pipe()
    try:
        os.dup2(write_fd, 2)
        os.close(write_fd)
        write_fd = -1
        if native_method == "method6":
            native = register_method6_device(*args, aff_pivot_zyx=aff_pivot_zyx, **kwargs)
        elif native_method == "method8":
            native = register_method8_device(*args, **kwargs)
        elif native_method in MATTES_NATIVE_METHODS:
            register_mattes = (
                register_method10_mattes_device
                if native_method == "method10-mattes"
                else register_method11_mattes_device
            )
            native = register_mattes(
                *args,
                histogram_bins=mattes_bins,
                sample_count=mattes_samples,
                translation_only=mattes_fixed_shear,
                **kwargs,
            )
        else:
            raise ValueError(f"unsupported native method {native_method!r}")
    finally:
        if write_fd != -1:
            os.close(write_fd)
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)

    chunks: list[bytes] = []
    while True:
        chunk = os.read(read_fd, 8192)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    stderr = b"".join(chunks).decode(errors="replace").strip()
    return native, stderr or None


def _native_initial_xyz_3x4(
    matrix_zyx: np.ndarray,
    translation_zyx: np.ndarray,
    shape_zyx: tuple[int, int, int],
    *,
    native_method: str = "method8",
    mattes_fixed_shear: bool,
) -> np.ndarray:
    """Convert the centered moving-to-fixed model into the native initializer convention."""
    if native_method == "method6":
        matrix_zyx, translation_zyx = output_to_input_from_model(
            matrix_zyx,
            translation_zyx,
            shape_zyx,
        )
    elif mattes_fixed_shear:
        exact_pull, _ = output_to_input_from_model(
            matrix_zyx,
            translation_zyx,
            shape_zyx,
        )
        projected_pull = np.eye(3, dtype=np.float64)
        projected_pull[0, 1] = projected_pull[1, 0] = (exact_pull[0, 1] + exact_pull[1, 0]) / 2.0
        projected_pull[0, 2] = projected_pull[2, 0] = (exact_pull[0, 2] + exact_pull[2, 0]) / 2.0
        projected_pull[1, 2] = projected_pull[2, 1] = (exact_pull[1, 2] + exact_pull[2, 1]) / 2.0
        center = (np.asarray(shape_zyx, dtype=np.float64) - 1.0) / 2.0
        translation_zyx = center - projected_pull @ (center + np.asarray(translation_zyx, dtype=np.float64))
        matrix_zyx = projected_pull
    return zyx_to_xyz_3x4(matrix_zyx, translation_zyx)


def _parse_starting_affine_matrix_zyx(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("starting affine matrix must contain numeric values") from exc
    if len(values) != 9:
        raise argparse.ArgumentTypeError("starting affine matrix must contain 9 row-major values")
    if not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("starting affine matrix values must be finite")
    return values


def _resolve_starting_affine_matrix(
    configured: tuple[float, ...] | None,
    fallback: np.ndarray,
    *,
    already_applied: bool = False,
) -> tuple[np.ndarray, str]:
    if configured is None or already_applied:
        source = "upstream_initializer_refined_from_cli_starting_affine" if already_applied else "upstream_initializer"
        return np.asarray(fallback, dtype=np.float32), source
    matrix = np.asarray(configured, dtype=np.float32).reshape(3, 3)
    rejection = _affine_quality_rejection(_affine_quality(matrix))
    if rejection is not None:
        raise ValueError(f"--starting-affine-matrix-zyx is not a plausible affine initializer: {rejection}")
    return matrix, "cli_starting_affine_matrix_zyx"


def _initializer_refines_starting_affine(artifact_type: str | None) -> bool:
    return artifact_type in {
        "lightsheet.fused_fixed_spatial_initializer.v1",
        "lightsheet.fused_fixed_native_spatial_recovery_initializer.v1",
    }


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


def _tile_number(tile_name: str) -> str:
    return tile_name.split(".")[-3]


@cache
def _records_by_tile_number(position: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(position.read_text())
    return {_tile_number(str(record["tile"])): record for record in payload["tiles"]}


def _record_by_tile_number(position: Path, tile_no: str) -> dict[str, Any]:
    records = _records_by_tile_number(position.resolve())
    if tile_no not in records:
        raise ValueError(f"{position} has no tile {tile_no}")
    return records[tile_no]


def source_record_with_geometry(
    source_record: dict[str, Any],
    geometry_record: dict[str, Any],
) -> dict[str, Any]:
    """Combine source IO metadata with the accepted coarse placement."""
    return source_record | {
        "tile": geometry_record["tile"],
        "translation_um": geometry_record["translation_um"],
        "scale_um": geometry_record["scale_um"],
    }


def _oriented_origin_spacing_um(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tile crops are read in oriented voxel order: voxel zero sits at the low physical bound."""
    tile = tile_record_from_position_record(record)
    return tile_bounds_zyx_um(tile)[0], np.abs(np.asarray(tile.scale_zyx_um, dtype=np.float64))


@cache
def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


@cache
def _fixed_fused_root(path: Path) -> Any:
    return zarr.open_group(str(path), mode="r")


def _fused_scale_translation(root: Any, *, level: int) -> tuple[np.ndarray, np.ndarray]:
    attrs = root.attrs.get("ome", root.attrs)
    multiscales = attrs["multiscales"]
    dataset = multiscales[0]["datasets"][level]
    scale = np.ones(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for transform in dataset.get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            scale = np.asarray(transform["scale"], dtype=np.float64)
        elif transform.get("type") == "translation":
            translation = np.asarray(transform["translation"], dtype=np.float64)
    return scale, translation


def _index_affine_to_center_model(
    *,
    matrix: np.ndarray,
    offset: np.ndarray,
    fixed_shape_zyx: np.ndarray,
    moving_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_center = (np.asarray(fixed_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moving_center = (np.asarray(moving_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    translation = np.asarray(offset, dtype=np.float64) + np.asarray(matrix, dtype=np.float64) @ moving_center - fixed_center
    return np.asarray(matrix, dtype=np.float32), translation.astype(np.float32)


def _center_model_to_index_affine(
    *,
    matrix: np.ndarray,
    translation: np.ndarray,
    fixed_shape_zyx: np.ndarray,
    moving_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_center = (np.asarray(fixed_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moving_center = (np.asarray(moving_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    model_matrix = np.asarray(matrix, dtype=np.float64)
    offset = fixed_center + np.asarray(translation, dtype=np.float64) - model_matrix @ moving_center
    return model_matrix, offset


def _local_model_to_global_index_affine(
    *,
    matrix: np.ndarray,
    translation: np.ndarray,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_matrix, local_offset = _center_model_to_index_affine(
        matrix=matrix,
        translation=translation,
        fixed_shape_zyx=crop_shape_zyx,
        moving_shape_zyx=crop_shape_zyx,
    )
    global_offset = (
        local_offset
        - local_matrix @ np.asarray(moving_start_zyx, dtype=np.float64)
        + np.asarray(fixed_start_zyx, dtype=np.float64)
    )
    return local_matrix, global_offset


def _global_index_affine_to_local_model(
    *,
    matrix: np.ndarray,
    offset: np.ndarray,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_offset = (
        np.asarray(offset, dtype=np.float64)
        + np.asarray(matrix, dtype=np.float64) @ np.asarray(moving_start_zyx, dtype=np.float64)
        - np.asarray(fixed_start_zyx, dtype=np.float64)
    )
    return _index_affine_to_center_model(
        matrix=matrix,
        offset=local_offset,
        fixed_shape_zyx=crop_shape_zyx,
        moving_shape_zyx=crop_shape_zyx,
    )


def _pretranslated_overlap_bounds(
    *,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
    phase_translation_zyx: np.ndarray,
    fixed_array_shape_zyx: np.ndarray,
    moving_array_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shift = np.rint(np.asarray(phase_translation_zyx, dtype=np.float64)).astype(np.int64)
    fixed_start = np.asarray(fixed_start_zyx, dtype=np.int64) + shift
    moving_start = np.asarray(moving_start_zyx, dtype=np.int64).copy()
    shape = np.asarray(crop_shape_zyx, dtype=np.int64).copy()

    if np.any(moving_start < 0) or np.any(moving_start + shape > np.asarray(moving_array_shape_zyx, dtype=np.int64)):
        raise ValueError(
            "phase-pretranslated moving crop is outside bounds: "
            f"moving_start={moving_start.tolist()} moving_stop={(moving_start + shape).tolist()} "
            f"moving_shape={np.asarray(moving_array_shape_zyx, dtype=np.int64).tolist()}"
        )
    return fixed_start, moving_start, shape, shift


def _read_fused_crop(array: Any, start_zyx: np.ndarray, shape_zyx: np.ndarray) -> np.ndarray:
    start = np.asarray(start_zyx, dtype=np.int64)
    shape = np.asarray(shape_zyx, dtype=np.int64)
    if np.any(shape <= 0):
        raise ValueError(f"shape_zyx must be positive, got {shape.tolist()}")
    stop = start + shape
    array_shape = np.asarray(array.shape, dtype=np.int64)
    source_start = np.maximum(start, 0)
    source_stop = np.minimum(stop, array_shape)
    output = np.zeros(tuple(int(value) for value in shape), dtype=np.float32)
    if np.any(source_stop <= source_start):
        return output
    dest_start = source_start - start
    dest_stop = dest_start + (source_stop - source_start)
    output_slices = tuple(slice(int(a), int(b)) for a, b in zip(dest_start, dest_stop, strict=True))
    source_slices = tuple(slice(int(a), int(b)) for a, b in zip(source_start, source_stop, strict=True))
    output[output_slices] = np.asarray(array[source_slices], dtype=np.float32)
    return output


def _crop_padding_metadata(start_zyx: np.ndarray, shape_zyx: np.ndarray, array_shape_zyx: np.ndarray) -> dict[str, Any]:
    start = np.asarray(start_zyx, dtype=np.int64)
    shape = np.asarray(shape_zyx, dtype=np.int64)
    stop = start + shape
    array_shape = np.asarray(array_shape_zyx, dtype=np.int64)
    source_start = np.maximum(start, 0)
    source_stop = np.minimum(stop, array_shape)
    valid_shape = np.maximum(source_stop - source_start, 0)
    valid_voxels = int(np.prod(valid_shape, dtype=np.int64)) if np.all(valid_shape > 0) else 0
    total_voxels = int(np.prod(shape, dtype=np.int64))
    return {
        "requested_start_zyx": start.tolist(),
        "requested_stop_zyx": stop.tolist(),
        "requested_shape_zyx": shape.tolist(),
        "source_start_zyx": source_start.tolist(),
        "source_stop_zyx": source_stop.tolist(),
        "source_shape_zyx": valid_shape.tolist(),
        "padded": bool(valid_voxels < total_voxels),
        "padded_fraction": float(1.0 - (valid_voxels / total_voxels)) if total_voxels else 1.0,
    }


def _cache_version(args: argparse.Namespace) -> str:
    version = FUSED_FIXED_CACHE_VERSION + ("_local_3d_subpixel" if args.local_3d_refinement else "")
    if getattr(args, "level0_initializer", None) == "window-parent":
        version += "_selected_parent_affine_v1"
    return version


def _parent_reference_cache_fields(args: argparse.Namespace) -> dict[str, str]:
    import hashlib

    reference = getattr(args, "parent_reference_fused", None)
    if reference is None:
        return {}
    reference = Path(reference).resolve()
    return {"parent_reference_fused": str(reference),
            "parent_reference_root_metadata_sha256": hashlib.sha256((reference / "zarr.json").read_bytes()).hexdigest()}


def _cache_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": _cache_version(args),
        **_parent_reference_cache_fields(args),
        "level0_initializer": getattr(args, "level0_initializer", None),
        **({"split_parent_windows": True} if getattr(args, "split_parent_windows", False) else {}),
        **({"parent_window_dir": str(Path(args.parent_window_dir).resolve())}
           if getattr(args, "parent_window_dir", None) is not None else {}),
        "level2_phaseprimed_dir": getattr(args, "level2_phaseprimed_dir", None),
        "window_initializer_dir": None
        if getattr(args, "window_initializer_dir", None) is None
        else str(Path(args.window_initializer_dir).resolve()),
        "initializer_sha256": getattr(args, "initializer_sha256", None),
        "fixed_fused": str(Path(args.fixed_fused).resolve()),
        "moving_position": str(Path(args.moving_position).resolve()),
        "moving_source_position": str(Path(args.moving_source_position).resolve()),
        "moving_channel": int(args.moving_channel),
        "fit_intensity_transform": str(args.fit_intensity_transform),
        "fit_downsample_zyx": [int(value) for value in args.fit_downsample_zyx],
        "native_lib_dir": str(Path(args.native_lib_dir).resolve()),
        "native_method": str(args.native_method),
        "starting_affine_matrix_zyx": None
        if getattr(args, "starting_affine_matrix_zyx", None) is None
        else [float(value) for value in args.starting_affine_matrix_zyx],
        "mattes_bins": int(args.mattes_bins),
        "mattes_samples": int(args.mattes_samples),
        "mattes_fixed_shear": bool(args.mattes_fixed_shear),
        "ftol": float(args.ftol),
        "max_iterations": int(args.max_iterations),
        **({"local_3d_refinement": True} if args.local_3d_refinement else {}),
        **(local_affine_cache_config(args.local_affine_estimator) if args.local_3d_refinement else {}),
        "phase_upsample_factor": int(args.phase_upsample_factor),
        "min_corr": float(args.min_corr),
        "min_grad_ncc": float(args.min_grad_ncc),
        "fixed_mask_threshold": None if args.fixed_mask_threshold is None else float(args.fixed_mask_threshold),
        "fixed_mask_level": int(args.fixed_mask_level),
        "fixed_mask_min_voxels": int(args.fixed_mask_min_voxels),
        "fixed_mask_max_masked_fraction": float(args.fixed_mask_max_masked_fraction),
    }


def _prepare_fit_volume_cupy(volume: Any, *, intensity_transform: str) -> tuple[Any, dict[str, float]]:
    """Apply the declared intensity transform before robust fit normalization."""
    import cupy as cp

    values = cp.asarray(volume, dtype=cp.float32)
    if intensity_transform == "log1p":
        values = cp.log1p(cp.maximum(values, cp.float32(0.0)))
    elif intensity_transform != "linear":
        raise ValueError(f"Unsupported fit intensity transform: {intensity_transform!r}")
    return _robust_norm_and_content_stats_cupy(values)


CAPTURE_GRID_UM_ZYX = np.asarray([2.4, 1.15, 1.15], dtype=np.float64)
CAPTURE_MAX_SHIFT_UM = 24.0
CAPTURE_HIGHPASS_SIGMA_UM = 4.6
CAPTURE_PEAK_EXCLUSION_UM = 3.5
CAPTURE_MIN_NCC = 0.20
CAPTURE_MIN_PEAK_MARGIN = 0.10
CAPTURE_OVERLAP_RATIO = 0.3


def _structural_capture(
    *,
    register_script: Any,
    fixed_fit_gpu: Any,
    moving_fit_gpu: Any,
    preseed_registered: Any,
    fit_matrix: np.ndarray,
    fit_translation: np.ndarray,
    fixed_fit_mask_gpu: Any | None,
    fit_voxel_um_zyx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bounded masked structural translation capture ahead of native affine refinement.

    Searches in-mask high-passed tissue structure on a ~level-2 grid. The returned
    fit-space delta is zero unless the peak is interior, strong, and separated.
    """
    import cupy as cp

    block = tuple(
        max(1, int(round(target / voxel)))
        for target, voxel in zip(CAPTURE_GRID_UM_ZYX, np.asarray(fit_voxel_um_zyx, dtype=np.float64), strict=True)
    )
    grid_um = np.asarray(fit_voxel_um_zyx, dtype=np.float64) * np.asarray(block, dtype=np.float64)
    support = register_script._warp_fit_preseed_cupy(
        cp.ones_like(moving_fit_gpu, dtype=cp.float32), fit_matrix, fit_translation
    )
    fixed_valid = (
        cp.ones(fixed_fit_gpu.shape, dtype=cp.float32)
        if fixed_fit_mask_gpu is None
        else cp.asarray(fixed_fit_mask_gpu, dtype=cp.float32)
    )
    result = masked_structural_translation_gpu(
        _block_mean_downsample_zyx_cupy(cp.asarray(fixed_fit_gpu, dtype=cp.float32), block),
        _block_mean_downsample_zyx_cupy(cp.asarray(preseed_registered, dtype=cp.float32), block),
        fixed_mask=_block_mean_downsample_zyx_cupy(fixed_valid, block) > 0.5,
        moving_support=_block_mean_downsample_zyx_cupy(support, block) > 0.5,
        max_shift_zyx=tuple(int(np.ceil(CAPTURE_MAX_SHIFT_UM / value)) for value in grid_um),
        highpass_sigma_zyx=tuple(float(CAPTURE_HIGHPASS_SIGMA_UM / value) for value in grid_um),
        exclusion_radius_zyx=tuple(int(np.ceil(CAPTURE_PEAK_EXCLUSION_UM / value)) for value in grid_um),
        overlap_ratio=CAPTURE_OVERLAP_RATIO,
    )
    confident = (
        not result.at_search_edge
        and result.peak_ncc >= CAPTURE_MIN_NCC
        and result.peak_ncc - result.second_peak_ncc >= CAPTURE_MIN_PEAK_MARGIN
    )
    delta = (
        np.asarray(result.shift_zyx, dtype=np.float32) * np.asarray(block, dtype=np.float32)
        if confident
        else np.zeros(3, dtype=np.float32)
    )
    return delta, {
        "method": "bounded masked high-pass NCC translation capture",
        "confident": bool(confident),
        "capture_grid_um_zyx": grid_um.tolist(),
        "capture_block_zyx": list(block),
        "shift_capture_grid_zyx": list(result.shift_zyx),
        "peak_ncc": result.peak_ncc,
        "second_peak_ncc": result.second_peak_ncc,
        "at_search_edge": result.at_search_edge,
        "max_shift_um": CAPTURE_MAX_SHIFT_UM,
        "min_ncc": CAPTURE_MIN_NCC,
        "min_peak_margin": CAPTURE_MIN_PEAK_MARGIN,
        "translation_delta_fit_zyx": delta.astype(np.float64).tolist(),
    }


def _native_objective_mask(args: argparse.Namespace, fixed_fit_mask_gpu: Any | None) -> Any | None:
    """Restrict the NCC-affine objective to fixed tissue voxels."""
    return fixed_fit_mask_gpu if str(args.native_method) in NCC_AFFINE_NATIVE_METHODS else None


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


def _resize_bool_mask_nearest_cupy(mask_gpu: Any, target_shape_zyx: tuple[int, int, int]) -> Any:
    import cupy as cp

    values = cp.asarray(mask_gpu, dtype=cp.bool_)
    target = tuple(int(value) for value in target_shape_zyx)
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


def _fused_level_crop_from_level0_bounds(
    root: Any,
    *,
    level: int,
    level0_start_zyx: np.ndarray,
    level0_stop_zyx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    level0_scale, level0_translation = _fused_scale_translation(root, level=0)
    target_scale, target_translation = _fused_scale_translation(root, level=int(level))
    world_start = level0_translation + level0_scale * np.asarray(level0_start_zyx, dtype=np.float64)
    world_stop = level0_translation + level0_scale * np.asarray(level0_stop_zyx, dtype=np.float64)
    target_start = np.floor((world_start - target_translation) / target_scale).astype(np.int64)
    target_stop = np.ceil((world_stop - target_translation) / target_scale).astype(np.int64)
    array = root[str(level)]
    target_shape = np.asarray(array.shape, dtype=np.int64)
    crop = _read_fused_crop(array, target_start, target_stop - target_start)
    return crop, {
        "requested_level": int(level),
        "source_level": int(level),
        "level_shape_zyx": target_shape.tolist(),
        "level_start_zyx": target_start.tolist(),
        "level_stop_zyx": target_stop.tolist(),
        "shape_zyx": list(crop.shape),
        "padding": _crop_padding_metadata(target_start, target_stop - target_start, target_shape),
    }


def _prepare_fixed_mask(
    root: Any,
    *,
    args: argparse.Namespace,
    fixed_start_l0: np.ndarray,
    fixed_stop_l0: np.ndarray,
    fit_shape_zyx: tuple[int, int, int],
) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    fixed_mask_threshold = args.fixed_mask_threshold
    if fixed_mask_threshold is None:
        return None, None, None
    mask_source, source_info = _fused_level_crop_from_level0_bounds(
        root,
        level=int(args.fixed_mask_level),
        level0_start_zyx=fixed_start_l0,
        level0_stop_zyx=fixed_stop_l0,
    )
    return _prepare_fixed_mask_from_crop(
        mask_source,
        source_info=source_info,
        args=args,
        fit_shape_zyx=fit_shape_zyx,
    )


def _prepare_fixed_mask_from_crop(
    mask_source: np.ndarray,
    *,
    source_info: dict[str, Any],
    args: argparse.Namespace,
    fit_shape_zyx: tuple[int, int, int],
) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    """Build a fit mask from an already-read pyramid crop without repeating I/O."""
    import cupy as cp

    threshold = float(args.fixed_mask_threshold)
    mask_source_gpu = cp.asarray(mask_source, dtype=cp.float32)
    mask_level_gpu = cp.ascontiguousarray(mask_source_gpu > cp.float32(threshold))
    mask_stats = _fixed_threshold_mask_stats(mask_level_gpu)
    fit_mask_gpu = _resize_bool_mask_nearest_cupy(mask_level_gpu, fit_shape_zyx)
    fit_mask_stats = _fixed_threshold_mask_stats(fit_mask_gpu)
    metadata = {
        "source": f"fixed_fused_level{int(args.fixed_mask_level)}",
        "threshold": threshold,
        "source_info": source_info,
        **mask_stats,
        "fit_shape_zyx": [int(value) for value in fit_shape_zyx],
        "fit_voxel_count": int(fit_mask_stats["voxel_count"]),
        "fit_total_voxels": int(fit_mask_stats["total_voxels"]),
        "fit_masked_fraction": float(fit_mask_stats["masked_fraction"]),
        "fit_unmasked_fraction": float(fit_mask_stats["unmasked_fraction"]),
    }
    if float(mask_stats["masked_fraction"]) > float(args.fixed_mask_max_masked_fraction):
        return fit_mask_gpu, metadata, "fixed_threshold_mask_too_masked"
    if float(fit_mask_stats["masked_fraction"]) > float(args.fixed_mask_max_masked_fraction):
        return fit_mask_gpu, metadata, "fixed_threshold_fit_mask_too_masked"
    if int(fit_mask_stats["voxel_count"]) < int(args.fixed_mask_min_voxels):
        return fit_mask_gpu, metadata, "fixed_threshold_fit_mask_empty"
    return fit_mask_gpu, metadata, None


def _fixed_threshold_mask_metadata_from_crop(
    fixed_raw: np.ndarray,
    *,
    args: argparse.Namespace,
    source: str,
    source_info: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if args.fixed_mask_threshold is None:
        return None, None
    import cupy as cp

    mask_gpu = cp.asarray(fixed_raw, dtype=cp.float32) > cp.float32(float(args.fixed_mask_threshold))
    mask_stats = _fixed_threshold_mask_stats(mask_gpu)
    metadata = {
        "source": source,
        "threshold": float(args.fixed_mask_threshold),
        "source_info": source_info,
        **mask_stats,
    }
    if float(mask_stats["masked_fraction"]) > float(args.fixed_mask_max_masked_fraction):
        return metadata, "fixed_threshold_mask_too_masked"
    if int(mask_stats["voxel_count"]) < int(args.fixed_mask_min_voxels):
        return metadata, "fixed_threshold_mask_empty"
    return metadata, None


def _attempt_status(
    *,
    corr: float | None,
    gradient: dict[str, Any],
    min_corr: float,
    min_grad_ncc: float,
    native_return_code: int | None = None,
    affine_quality: dict[str, Any] | None = None,
    require_corr: bool = True,
    require_gradient_ncc: bool = True,
) -> tuple[str, str | None]:
    grad_mean = _gradient_component_ncc_mean(gradient)
    if native_return_code is not None and int(native_return_code) != 0:
        return "rejected", "native_return_code"
    if affine_quality is not None:
        rejection = _affine_quality_rejection(affine_quality)
        if rejection is not None:
            return "rejected", rejection
    if require_corr and (corr is None or not np.isfinite(float(corr)) or float(corr) < float(min_corr)):
        return "rejected", "quality_gate_corr"
    if require_gradient_ncc and (not np.isfinite(grad_mean) or grad_mean < float(min_grad_ncc)):
        return "rejected", "quality_gate_grad_ncc"
    return "accepted", None


def _initializer_attempt_status(
    *,
    corr: float | None,
    gradient: dict[str, Any],
    min_corr: float,
    min_grad_ncc: float,
    affine_quality: dict[str, Any] | None,
    native_method: str,
) -> tuple[str, str | None]:
    """Keep a finite phase/placement initializer available as the Mattes fallback."""
    return _attempt_status(
        corr=corr,
        gradient=gradient,
        min_corr=min_corr,
        min_grad_ncc=min_grad_ncc,
        affine_quality=affine_quality,
        require_gradient_ncc=native_method not in MATTES_NATIVE_METHODS,
    )


def _select_native_initializer(
    *,
    initial_matrix: np.ndarray,
    initial_translation: np.ndarray,
    phase_matrix: np.ndarray,
    phase_translation: np.ndarray,
    capture_confident: bool,
    recovery_initializer: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Seed native refinement from a confident structural capture, else the incoming initializer.

    Gate metrics of the seeds are diagnostic: they do not decide whether refinement runs.
    """
    if capture_confident:
        name = "recovery_initializer" if recovery_initializer else "phase_primed_initializer"
        return phase_matrix, phase_translation, name
    return initial_matrix, initial_translation, "recovery_initializer" if recovery_initializer else "level2_initializer"


def _affine_quality(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)):
        return {
            "finite": False,
            "determinant": None,
            "condition": None,
            "singular_values": None,
            "max_axis_scale": None,
            "min_axis_scale": None,
            "max_axis_scale_allowed": MAX_AFFINE_AXIS_SCALE,
            "min_axis_scale_allowed": 1.0 / MAX_AFFINE_AXIS_SCALE,
            "max_condition_allowed": MAX_AFFINE_CONDITION,
            "min_determinant_allowed": MIN_AFFINE_DETERMINANT,
            "max_determinant_allowed": MAX_AFFINE_DETERMINANT,
        }
    singular_values = np.linalg.svd(values, compute_uv=False)
    determinant = float(np.linalg.det(values))
    min_singular = float(np.min(singular_values))
    condition = float("inf") if min_singular <= 0 else float(np.max(singular_values) / min_singular)
    return {
        "finite": True,
        "determinant": determinant,
        "condition": condition,
        "singular_values": [float(value) for value in singular_values],
        "max_axis_scale": float(np.max(singular_values)),
        "min_axis_scale": min_singular,
        "max_axis_scale_allowed": MAX_AFFINE_AXIS_SCALE,
        "min_axis_scale_allowed": 1.0 / MAX_AFFINE_AXIS_SCALE,
        "max_condition_allowed": MAX_AFFINE_CONDITION,
        "min_determinant_allowed": MIN_AFFINE_DETERMINANT,
        "max_determinant_allowed": MAX_AFFINE_DETERMINANT,
    }


def _affine_quality_rejection(quality: dict[str, Any]) -> str | None:
    if not quality.get("finite"):
        return "affine_nonfinite"
    determinant = quality.get("determinant")
    condition = quality.get("condition")
    max_scale = quality.get("max_axis_scale")
    min_scale = quality.get("min_axis_scale")
    if determinant is None or not np.isfinite(float(determinant)) or float(determinant) <= 0:
        return "affine_det_nonpositive"
    if max_scale is None or not np.isfinite(float(max_scale)) or float(max_scale) > MAX_AFFINE_AXIS_SCALE:
        return "affine_scale_out_of_bounds"
    if min_scale is None or not np.isfinite(float(min_scale)) or float(min_scale) < (1.0 / MAX_AFFINE_AXIS_SCALE):
        return "affine_scale_out_of_bounds"
    if condition is None or not np.isfinite(float(condition)) or float(condition) > MAX_AFFINE_CONDITION:
        return "affine_condition_out_of_bounds"
    if float(determinant) < MIN_AFFINE_DETERMINANT or float(determinant) > MAX_AFFINE_DETERMINANT:
        return "affine_det_out_of_bounds"
    return None


def _selected_global_transforms(
    *,
    local_matrix: np.ndarray,
    local_translation: np.ndarray,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
) -> dict[str, Any]:
    local_forward_matrix, local_forward_offset = _center_model_to_index_affine(
        matrix=np.asarray(local_matrix, dtype=np.float64),
        translation=np.asarray(local_translation, dtype=np.float64),
        fixed_shape_zyx=crop_shape_zyx,
        moving_shape_zyx=crop_shape_zyx,
    )
    forward_offset = local_forward_offset - local_forward_matrix @ np.asarray(moving_start_zyx, dtype=np.float64) + np.asarray(
        fixed_start_zyx,
        dtype=np.float64,
    )
    pull_matrix, pull_offset = output_to_input_from_model(
        np.asarray(local_matrix, dtype=np.float64),
        np.asarray(local_translation, dtype=np.float64),
        tuple(int(value) for value in crop_shape_zyx),
    )
    global_pull_offset = np.asarray(pull_offset, dtype=np.float64) - np.asarray(pull_matrix, dtype=np.float64) @ np.asarray(
        fixed_start_zyx,
        dtype=np.float64,
    ) + np.asarray(moving_start_zyx, dtype=np.float64)
    return {
        "selected_moving_l0_to_fixed_fused_l0_matrix_zyx": np.asarray(local_forward_matrix, dtype=np.float64).tolist(),
        "selected_moving_l0_to_fixed_fused_l0_offset_zyx": np.asarray(forward_offset, dtype=np.float64).tolist(),
        "selected_fixed_fused_l0_to_moving_l0_pull_matrix_zyx": np.asarray(pull_matrix, dtype=np.float64).tolist(),
        "selected_fixed_fused_l0_to_moving_l0_pull_offset_zyx": np.asarray(global_pull_offset, dtype=np.float64).tolist(),
    }


def _read_tile_level_crop(
    tile: Any,
    *,
    channel: int,
    level: int,
    start_zyx: np.ndarray,
    stop_zyx: np.ndarray,
) -> np.ndarray:
    array, _source_level, _available_levels, store = _open_tile_level_array(tile.path, source_level=int(level))
    try:
        level_shape = np.asarray(array.shape[1:4] if tile.axes == "CZYX" else array.shape, dtype=np.int64)
        raw_slices: list[slice] = []
        reverse_axes: list[int] = []
        for axis, (start, stop) in enumerate(zip(start_zyx, stop_zyx, strict=True)):
            raw_slice, reverse = raw_axis_slice_for_oriented_slice(
                slice(int(start), int(stop)),
                axis_size=int(level_shape[axis]),
                flipped=bool(tile.scale_zyx_um[axis] < 0),
            )
            raw_slices.append(raw_slice)
            if reverse:
                reverse_axes.append(axis)
        if tile.axes == "CZYX":
            patch = array[(int(channel), raw_slices[0], raw_slices[1], raw_slices[2])]
        else:
            if int(channel) != 0:
                raise ValueError(f"{tile.path} has no channel axis; requested channel {channel}")
            patch = array[tuple(raw_slices)]
        result = np.asarray(patch.compute(), dtype=np.float32)
        for axis in reverse_axes:
            result = np.flip(result, axis=axis)
        return result
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _run_level2_metadata(args: argparse.Namespace, moving_record: dict[str, Any], moving_tile: Any) -> Path:
    from squisher_lightsheet import cross_register_method8 as register_script
    root = zarr.open_group(str(args.fixed_fused), mode="r")
    fixed_array = root[str(args.fixed_level)]
    fused_scale, fused_translation = _fused_scale_translation(root, level=int(args.fixed_level))
    moving_translation = tile_bounds_zyx_um(moving_tile)[0]
    moving_scale = np.abs(np.asarray(moving_tile.scale_zyx_um, dtype=np.float64))
    moving_start_l0 = np.asarray(args.moving_start_zyx, dtype=np.int64)
    shape_l0 = np.asarray(args.shape_zyx, dtype=np.int64)
    moving_level = int(args.moving_level)
    started = time.perf_counter()
    level_factor = np.maximum(1, np.rint(fused_scale / moving_scale).astype(np.int64))
    if tuple(level_factor.tolist()) != (4, 4, 4):
        raise ValueError(f"expected fused level2 factor [4,4,4], got {level_factor.tolist()}")
    fixed_start, fixed_stop, _factor, _scale, _origin = _metadata_fixed_bounds(
        fixed_fused_root=root, fixed_level=int(args.fixed_level), moving_record=moving_record,
        moving_start_l0=moving_start_l0, shape_l0=shape_l0,
    )
    fixed_array_shape = np.asarray(fixed_array.shape, dtype=np.int64)
    fixed_padding = _crop_padding_metadata(fixed_start, fixed_stop - fixed_start, fixed_array_shape)
    fixed_raw = _read_fused_crop(fixed_array, fixed_start, fixed_stop - fixed_start)
    import cupy as cp

    with cp.cuda.Device(int(args.device)):
        fixed_mask_metadata, fixed_mask_rejection = _fixed_threshold_mask_metadata_from_crop(
            fixed_raw,
            args=args,
            source=f"fixed_fused_level{int(args.fixed_level)}",
            source_info={
                "level_shape_zyx": fixed_array_shape.tolist(),
                "level_start_zyx": fixed_start.tolist(),
                "level_stop_zyx": fixed_stop.tolist(),
                "shape_zyx": list(fixed_raw.shape),
                "padding": fixed_padding,
            },
        )
    if fixed_mask_rejection is not None:
        row = {
            "schema_version": 1,
            "artifact_type": "lightsheet.image10_to_fused_image14_method8_level2_metadata_trial.v1",
            "cache_config": _cache_config(args),
            "status": "rejected",
            "rejection_reason": fixed_mask_rejection,
            "fixed_source": "Image_14 fused OME-Zarr",
            "fixed_fused": str(args.fixed_fused),
            "fixed_level": int(args.fixed_level),
            "moving_tile": moving_tile.tile,
            "moving_path": str(moving_tile.path),
            "moving_level": moving_level,
            "moving_channel": int(args.moving_channel),
            "placement_source": "moving_position_translation_um_in_fused_image14_grid",
            "level_factor_zyx": level_factor.tolist(),
            "moving_start_l0_zyx": moving_start_l0.tolist(),
            "moving_stop_l0_zyx": (moving_start_l0 + shape_l0).tolist(),
            "fixed_start_zyx": fixed_start.tolist(),
            "fixed_stop_zyx": fixed_stop.tolist(),
            "fixed_crop_padding": fixed_padding,
            "fixed_shape_zyx": list(fixed_raw.shape),
            "fixed_threshold_mask": fixed_mask_metadata,
            "quality_mask": "fixed_threshold_mask",
            "timing_seconds": {"total": time.perf_counter() - started},
        }
        _write_json(args.output, row)
        return args.output.resolve()

    moving_level_xy_factor = 4 if moving_level == 2 else 1
    moving_level_start = np.asarray(
        [
            moving_start_l0[0],
            int(np.floor(moving_start_l0[1] / moving_level_xy_factor)),
            int(np.floor(moving_start_l0[2] / moving_level_xy_factor)),
        ],
        dtype=np.int64,
    )
    moving_level_stop = np.asarray(
        [
            moving_start_l0[0] + shape_l0[0],
            int(np.ceil((moving_start_l0[1] + shape_l0[1]) / moving_level_xy_factor)),
            int(np.ceil((moving_start_l0[2] + shape_l0[2]) / moving_level_xy_factor)),
        ],
        dtype=np.int64,
    )
    moving_raw_zfull = _read_tile_level_crop(
        moving_tile,
        channel=int(args.moving_channel),
        level=moving_level,
        start_zyx=moving_level_start,
        stop_zyx=moving_level_stop,
    )

    with cp.cuda.Device(int(args.device)):
        moving_gpu = cp.asarray(moving_raw_zfull, dtype=cp.float32)
        moving_gpu = _block_mean_downsample_zyx_cupy(moving_gpu, (int(level_factor[0]), 1, 1))
        moving_raw = cp.asnumpy(moving_gpu)
        fixed_gpu, fixed_stats = _prepare_fit_volume_cupy(
            fixed_raw,
            intensity_transform=str(args.fit_intensity_transform),
        )
        moving_gpu, moving_stats = _prepare_fit_volume_cupy(
            moving_raw,
            intensity_transform=str(args.fit_intensity_transform),
        )
        initial_matrix, initial_matrix_source = _resolve_starting_affine_matrix(
            getattr(args, "starting_affine_matrix_zyx", None),
            np.eye(3, dtype=np.float32),
        )
        start_offset = (moving_translation + moving_scale * moving_start_l0.astype(np.float64) - fused_translation) / fused_scale - fixed_start
        initial_translation = start_offset.astype(np.float32)
        if "placement_affine_zyx_um" in moving_record:
            initial_matrix, initial_translation = _placement_crop_initializer(
                moving_record, moving_start_l0=moving_start_l0, fit_shape=np.asarray(moving_raw.shape),
                level_factor=level_factor, fused_scale=fused_scale, fused_translation=fused_translation,
                fixed_start=fixed_start, channel_matrix=initial_matrix,
            )
            initial_matrix_source += "_composed_with_reviewed_world_placement"
        preseed_registered = register_script._warp_fit_preseed_cupy(moving_gpu, initial_matrix, initial_translation)
        fixed_mask_gpu = (
            None
            if args.fixed_mask_threshold is None
            else cp.asarray(fixed_raw > float(args.fixed_mask_threshold), dtype=cp.bool_)
        )
        preseed_corr = _corr_gpu(fixed_gpu, preseed_registered, fixed_mask=fixed_mask_gpu)
        preseed_gradient = gradient_component_ncc_3d_gpu(
            fixed_gpu,
            preseed_registered,
            fixed_mask=fixed_mask_gpu,
        )
        preseed_affine_quality = _affine_quality(initial_matrix)
        preseed_status, preseed_rejection = _initializer_attempt_status(
            corr=preseed_corr,
            gradient=preseed_gradient,
            min_corr=float(args.min_corr),
            min_grad_ncc=float(args.min_grad_ncc),
            affine_quality=preseed_affine_quality,
            native_method=str(args.native_method),
        )
        phase_started = time.perf_counter()
        phase_delta, capture = _structural_capture(
            register_script=register_script,
            fixed_fit_gpu=fixed_gpu,
            moving_fit_gpu=moving_gpu,
            preseed_registered=preseed_registered,
            fit_matrix=initial_matrix,
            fit_translation=initial_translation,
            fixed_fit_mask_gpu=fixed_mask_gpu,
            fit_voxel_um_zyx=fused_scale,
        )
        phase_translation = initial_translation + phase_delta
        phase_registered = register_script._warp_fit_preseed_cupy(
            moving_gpu,
            initial_matrix,
            phase_translation,
        )
        phase_corr = _corr_gpu(fixed_gpu, phase_registered, fixed_mask=fixed_mask_gpu)
        phase_gradient = gradient_component_ncc_3d_gpu(
            fixed_gpu,
            phase_registered,
            fixed_mask=fixed_mask_gpu,
        )
        phase_status, phase_rejection = _initializer_attempt_status(
            corr=phase_corr,
            gradient=phase_gradient,
            min_corr=float(args.min_corr),
            min_grad_ncc=float(args.min_grad_ncc),
            affine_quality=preseed_affine_quality,
            native_method=str(args.native_method),
        )
        phase_seconds = time.perf_counter() - phase_started
        initializer_rejection = _affine_quality_rejection(preseed_affine_quality)
        if initializer_rejection is not None:
            row = {
                "schema_version": 1,
                "artifact_type": "lightsheet.image10_to_fused_image14_method8_level2_metadata_trial.v1",
                "cache_config": _cache_config(args),
                "status": "rejected",
                "rejection_reason": initializer_rejection,
                "fixed_source": "Image_14 fused OME-Zarr",
                "fixed_fused": str(args.fixed_fused),
                "fixed_level": int(args.fixed_level),
                "moving_tile": moving_tile.tile,
                "moving_path": str(moving_tile.path),
                "moving_level": moving_level,
                "moving_channel": int(args.moving_channel),
                "placement_source": "moving_position_translation_um_in_fused_image14_grid",
                "level_factor_zyx": level_factor.tolist(),
                "moving_start_l0_zyx": moving_start_l0.tolist(),
                "moving_stop_l0_zyx": (moving_start_l0 + shape_l0).tolist(),
                "fixed_start_zyx": fixed_start.tolist(),
                "fixed_stop_zyx": fixed_stop.tolist(),
                "fixed_crop_padding": fixed_padding,
                "moving_level_start_zyx": moving_level_start.tolist(),
                "moving_level_stop_zyx": moving_level_stop.tolist(),
                "fixed_shape_zyx": list(fixed_raw.shape),
                "moving_shape_before_z_downsample_zyx": list(moving_raw_zfull.shape),
                "moving_shape_zyx": list(moving_raw.shape),
                "initial_matrix_zyx": initial_matrix.tolist(),
                "initial_matrix_source": initial_matrix_source,
                "initial_translation_zyx": initial_translation.tolist(),
                "preseed_only_corr": None if preseed_corr is None else float(preseed_corr),
                "preseed_only_gradient_component_ncc": preseed_gradient,
                "preseed_only_gradient_component_ncc_mean": _gradient_component_ncc_mean(preseed_gradient),
                "preseed_only_affine_quality": preseed_affine_quality,
                "phase_priming": {
                    "status": phase_status,
                    "rejection_reason": phase_rejection,
                    "masked": True,
                    "capture": capture,
                    "quality_mask": "fixed_threshold_mask" if fixed_mask_gpu is not None else None,
                    "translation_delta_level_zyx": phase_delta.astype(np.float64).tolist(),
                    "local_translation_zyx": phase_translation.astype(np.float64).tolist(),
                    "corr": None if phase_corr is None else float(phase_corr),
                    "gradient_component_ncc": phase_gradient,
                    "gradient_component_ncc_mean": _gradient_component_ncc_mean(phase_gradient),
                    "timing_seconds": phase_seconds,
                },
                "fixed_threshold_mask": fixed_mask_metadata,
                "quality_mask": "fixed_threshold_mask" if fixed_mask_metadata is not None else None,
                "fixed_content": fixed_stats,
                "moving_content": moving_stats,
                "local_matrix_zyx": np.asarray(initial_matrix, dtype=np.float64).tolist(),
                "local_translation_zyx": np.asarray(initial_translation, dtype=np.float64).tolist(),
                "timing_seconds": {"total": time.perf_counter() - started},
            }
            _write_json(args.output, row)
            return args.output.resolve()
        _native_matrix, native_initial_translation, selected_initializer = _select_native_initializer(
            initial_matrix=initial_matrix,
            initial_translation=initial_translation,
            phase_matrix=initial_matrix,
            phase_translation=phase_translation,
            capture_confident=bool(capture["confident"]),
        )
        native_initializer = {
            "level2_initializer": "placement_preseed",
            "phase_primed_initializer": "coarse_phase_primed",
        }[selected_initializer]
        native, native_stderr = _register_native_device_quiet(
            fixed_gpu,
            moving_gpu,
            fixed_mask_zyx=_native_objective_mask(args, fixed_mask_gpu),
            native_method=str(args.native_method),
            mattes_bins=int(args.mattes_bins),
            mattes_samples=int(args.mattes_samples),
            mattes_fixed_shear=bool(args.mattes_fixed_shear),
            lib_dir=args.native_lib_dir,
            ftol=float(args.ftol),
            max_iterations=int(args.max_iterations),
            device=int(args.device),
            aff_pivot_zyx=(
                moving_xy_center_slab_z_pivot_in_fit_zyx(
                    moving_full_shape_zyx=np.asarray(moving_tile.shape_zyx, dtype=np.float64) / level_factor,
                    moving_crop_start_zyx=moving_start_l0.astype(np.float64) / level_factor,
                    crop_shape_zyx=np.asarray(fixed_gpu.shape, dtype=np.int64),
                    local_matrix_zyx=initial_matrix,
                    local_translation_zyx=native_initial_translation,
                    fit_downsample_zyx=(1, 1, 1),
                )
                if str(args.native_method) == "method6"
                else None
            ),
            initial_matrix_xyz_3x4=_native_initial_xyz_3x4(
                initial_matrix,
                native_initial_translation,
                tuple(int(value) for value in fixed_gpu.shape),
                native_method=str(args.native_method),
                mattes_fixed_shear=bool(args.mattes_fixed_shear),
            ),
        )
        refined_matrix, refined_translation = output_to_input_to_model(
            native.matrix_zyx,
            native.offset_zyx,
            tuple(int(v) for v in fixed_gpu.shape),
        )
        corr = _corr_gpu(
            fixed_gpu,
            native.registered_zyx.astype(np.float32, copy=False),
            fixed_mask=fixed_mask_gpu,
        )
        gradient = gradient_component_ncc_3d_gpu(
            fixed_gpu,
            native.registered_zyx.astype(np.float32, copy=False),
            fixed_mask=fixed_mask_gpu,
        )
        affine_quality = _affine_quality(refined_matrix)

    row = {
        "schema_version": 1,
        "artifact_type": "lightsheet.image10_to_fused_image14_method8_level2_metadata_trial.v1",
        "cache_config": _cache_config(args),
        "fixed_source": "Image_14 fused OME-Zarr",
        "fixed_fused": str(args.fixed_fused),
        "fixed_level": int(args.fixed_level),
        "moving_tile": moving_tile.tile,
        "moving_path": str(moving_tile.path),
        "moving_level": moving_level,
        "moving_channel": int(args.moving_channel),
        "placement_source": "moving_position_translation_um_in_fused_image14_grid",
        "level_factor_zyx": level_factor.tolist(),
        "moving_start_l0_zyx": moving_start_l0.tolist(),
        "moving_stop_l0_zyx": (moving_start_l0 + shape_l0).tolist(),
        "fixed_start_zyx": fixed_start.tolist(),
        "fixed_stop_zyx": fixed_stop.tolist(),
        "fixed_crop_padding": fixed_padding,
        "moving_level_start_zyx": moving_level_start.tolist(),
        "moving_level_stop_zyx": moving_level_stop.tolist(),
        "fixed_shape_zyx": list(fixed_raw.shape),
        "moving_shape_before_z_downsample_zyx": list(moving_raw_zfull.shape),
        "moving_shape_zyx": list(moving_raw.shape),
        "initial_matrix_zyx": initial_matrix.tolist(),
        "initial_matrix_source": initial_matrix_source,
        "initial_translation_zyx": initial_translation.tolist(),
        "preseed_only_corr": None if preseed_corr is None else float(preseed_corr),
        "preseed_only_gradient_component_ncc": preseed_gradient,
        "preseed_only_gradient_component_ncc_mean": _gradient_component_ncc_mean(preseed_gradient),
        "preseed_only_affine_quality": preseed_affine_quality,
        "phase_priming": {
            "status": phase_status,
            "rejection_reason": phase_rejection,
            "masked": True,
            "capture": capture,
            "quality_mask": "fixed_threshold_mask" if fixed_mask_gpu is not None else None,
            "translation_delta_level_zyx": phase_delta.astype(np.float64).tolist(),
            "local_translation_zyx": phase_translation.astype(np.float64).tolist(),
            "corr": None if phase_corr is None else float(phase_corr),
            "gradient_component_ncc": phase_gradient,
            "gradient_component_ncc_mean": _gradient_component_ncc_mean(phase_gradient),
            "timing_seconds": phase_seconds,
        },
        "native_initializer": native_initializer,
        "selected_coarse_initializer": {
            "source": native_initializer,
            "local_matrix_zyx": np.asarray(initial_matrix, dtype=np.float64).tolist(),
            "local_translation_zyx": np.asarray(
                native_initial_translation,
                dtype=np.float64,
            ).tolist(),
        },
        "fixed_threshold_mask": fixed_mask_metadata,
        "quality_mask": "fixed_threshold_mask" if fixed_mask_metadata is not None else None,
        "fixed_content": fixed_stats,
        "moving_content": moving_stats,
        "native_return_code": int(native.return_code),
        "corr_refined": None if corr is None else float(corr),
        "gradient_component_ncc_refined": gradient,
        "gradient_component_ncc_mean": _gradient_component_ncc_mean(gradient),
        "affine_quality": affine_quality,
        "native_output_to_input_matrix_zyx": np.asarray(native.matrix_zyx, dtype=np.float64).tolist(),
        "native_output_to_input_offset_zyx": np.asarray(native.offset_zyx, dtype=np.float64).tolist(),
        "local_matrix_zyx": np.asarray(refined_matrix, dtype=np.float64).tolist(),
        "local_translation_zyx": np.asarray(refined_translation, dtype=np.float64).tolist(),
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    status, rejection = _attempt_status(
        corr=corr,
        gradient=gradient,
        min_corr=float(args.min_corr),
        min_grad_ncc=float(args.min_grad_ncc),
        native_return_code=int(native.return_code),
        affine_quality=affine_quality,
        require_corr=str(args.native_method) not in MATTES_NATIVE_METHODS,
        require_gradient_ncc=str(args.native_method) not in MATTES_NATIVE_METHODS,
    )
    row["status"] = status
    row["rejection_reason"] = rejection
    if int(native.return_code) != 0 and native_stderr:
        row["native_stderr"] = native_stderr
    _write_json(args.output, row)
    return args.output.resolve()


def _placed_level2_to_level0_initializer(
    level2: dict[str, Any], *, crop_shape: np.ndarray, fixed_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve a fitted canonical map when changing voxel spacing and crop origin."""
    factor = np.asarray(level2["level_factor_zyx"], dtype=np.float64)
    fit_shape = np.asarray(level2["fixed_shape_zyx"], dtype=np.float64)
    matrix, offset = _center_model_to_index_affine(
        matrix=np.asarray(level2["local_matrix_zyx"]),
        translation=np.asarray(level2["local_translation_zyx"]),
        fixed_shape_zyx=fit_shape, moving_shape_zyx=fit_shape,
    )
    matrix0 = np.diag(factor) @ matrix @ np.diag(1 / factor)
    offset0 = factor * (offset + np.asarray(level2["fixed_start_zyx"])) - fixed_start
    return _index_affine_to_center_model(
        matrix=matrix0, offset=offset0, fixed_shape_zyx=crop_shape, moving_shape_zyx=crop_shape,
    )


def _placement_crop_initializer(
    record: dict[str, Any], *, moving_start_l0: np.ndarray, fit_shape: np.ndarray,
    level_factor: np.ndarray, fused_scale: np.ndarray, fused_translation: np.ndarray,
    fixed_start: np.ndarray, channel_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a reviewed physical placement with the channel's crop-centred voxel initializer."""
    affine = np.asarray(record["placement_affine_zyx_um"], dtype=np.float64)
    origin, spacing = _oriented_origin_spacing_um(record)
    center = (fit_shape - 1) / 2
    world_center = origin + spacing * (moving_start_l0 + level_factor * center)
    placed_center = affine[:3, :3] @ world_center + affine[:3, 3]
    index_matrix = np.diag(1 / fused_scale) @ affine[:3, :3] @ np.diag(spacing * level_factor)
    matrix = index_matrix @ channel_matrix
    translation = (placed_center - fused_translation) / fused_scale - fixed_start - center
    return matrix.astype(np.float32), translation.astype(np.float32)


def _metadata_fixed_bounds(
    *,
    fixed_fused_root: Any,
    fixed_level: int,
    moving_record: dict[str, Any],
    moving_start_l0: np.ndarray,
    shape_l0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fused_scale, fused_translation = _fused_scale_translation(fixed_fused_root, level=int(fixed_level))
    moving_translation, moving_scale = _oriented_origin_spacing_um(moving_record)
    factor =np.maximum(1, np.rint(fused_scale / moving_scale).astype(np.int64))
    fixed_world_start = moving_translation + moving_scale * moving_start_l0.astype(np.float64)
    fixed_start = np.rint((fixed_world_start - fused_translation) / fused_scale).astype(np.int64)
    fixed_shape = np.ceil(shape_l0.astype(np.float64) / factor.astype(np.float64)).astype(np.int64)
    if "placement_affine_zyx_um" in moving_record:
        affine = np.asarray(moving_record["placement_affine_zyx_um"], dtype=np.float64)
        center = (fixed_shape - 1) / 2
        world_center = moving_translation + moving_scale * (moving_start_l0 + factor * center)
        placed_center = affine[:3, :3] @ world_center + affine[:3, 3]
        fixed_start = np.rint((placed_center - fused_translation) / fused_scale - center).astype(np.int64)
    return fixed_start, fixed_start + fixed_shape, factor, fused_scale, fused_translation


def _selected_slab_shape_zyx(level2: dict[str, Any]) -> np.ndarray:
    selected = level2.get("selected_slab_index")
    if selected is not None:
        for slab in level2.get("slabs", []):
            if int(slab.get("slab_index", -1)) == int(selected) and isinstance(slab.get("shape_zyx"), list):
                return np.asarray(slab["shape_zyx"], dtype=np.int64)
    fixed_start = level2.get("fixed_phaseprimed_start_level_zyx")
    fixed_stop = level2.get("fixed_phaseprimed_stop_level_zyx")
    if isinstance(fixed_start, list) and isinstance(fixed_stop, list):
        return np.asarray(fixed_stop, dtype=np.int64) - np.asarray(fixed_start, dtype=np.int64)
    raise ValueError("level2 phase-primed tile row is missing selected slab shape")


def _selected_slab_row(level2: dict[str, Any]) -> dict[str, Any] | None:
    selected = level2.get("selected_slab_index")
    if selected is None:
        return None
    for slab in level2.get("slabs", []):
        if int(slab.get("slab_index", -1)) == int(selected):
            return slab
    return None


def _metadata_fixed_start_level_zyx(
    *,
    fixed_record: dict[str, Any],
    moving_record: dict[str, Any],
    moving_start_level_zyx: np.ndarray,
    fixed_factor_zyx: np.ndarray,
    moving_factor_zyx: np.ndarray,
) -> np.ndarray:
    fixed_translation_um, fixed_scale_l0 = _oriented_origin_spacing_um(fixed_record)
    moving_translation_um, moving_scale_l0 = _oriented_origin_spacing_um(moving_record)
    fixed_level_scale_um = fixed_scale_l0 * np.asarray(fixed_factor_zyx, dtype=np.float64)
    moving_level_scale_um = moving_scale_l0 * np.asarray(moving_factor_zyx, dtype=np.float64)
    world_um = moving_translation_um + moving_level_scale_um * np.asarray(moving_start_level_zyx, dtype=np.float64)
    return np.rint((world_um - fixed_translation_um) / fixed_level_scale_um).astype(np.int64)


def _phaseprimed_tile_level0_initializer(
    *,
    level2: dict[str, Any],
    fixed_record: dict[str, Any],
    moving_record: dict[str, Any],
    root: Any,
    moving_start_l0: np.ndarray,
    crop_shape_l0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    required = (
        "fixed_phaseprimed_start_level_zyx",
        "moving_start_level_zyx",
        "fixed_level_factor_zyx",
        "moving_level_factor_zyx",
        "local_matrix_zyx",
        "local_translation_zyx",
    )
    missing = [key for key in required if key not in level2]
    if missing:
        raise ValueError(f"level2 phase-primed tile row is missing required keys: {missing}")

    level2_matrix = np.asarray(level2["local_matrix_zyx"], dtype=np.float64)
    level2_translation = np.asarray(level2["local_translation_zyx"], dtype=np.float64)
    fixed_start_l2 = np.asarray(level2["fixed_phaseprimed_start_level_zyx"], dtype=np.float64)
    moving_start_l2 = np.asarray(level2["moving_start_level_zyx"], dtype=np.float64)
    fixed_factor = np.asarray(level2["fixed_level_factor_zyx"], dtype=np.float64)
    moving_factor = np.asarray(level2["moving_level_factor_zyx"], dtype=np.float64)
    slab_shape_l2 = _selected_slab_shape_zyx(level2).astype(np.float64)
    if np.any(fixed_factor <= 0) or np.any(moving_factor <= 0):
        raise ValueError(f"invalid level2 factors fixed={fixed_factor.tolist()} moving={moving_factor.tolist()}")

    selected_slab = _selected_slab_row(level2)
    fallback_source = level2.get("nearest_level2_phaseprimed_tile_fallback")
    source_fixed_start_l2 = fixed_start_l2
    source_fixed_metadata_start_l2 = None
    fixed_metadata_start_l2 = None
    fixed_phase_residual_l2 = None
    if selected_slab is not None and isinstance(selected_slab.get("fixed_metadata_start_level_zyx"), list):
        source_fixed_metadata_start_l2 = np.asarray(selected_slab["fixed_metadata_start_level_zyx"], dtype=np.float64)
        fixed_phase_residual_l2 = fixed_start_l2 - source_fixed_metadata_start_l2
        fixed_metadata_start_l2 = _metadata_fixed_start_level_zyx(
            fixed_record=fixed_record,
            moving_record=moving_record,
            moving_start_level_zyx=moving_start_l2,
            fixed_factor_zyx=fixed_factor,
            moving_factor_zyx=moving_factor,
        ).astype(np.float64)
        fixed_start_l2 = fixed_metadata_start_l2 + fixed_phase_residual_l2

    level2_global_matrix, level2_global_offset = _local_model_to_global_index_affine(
        matrix=level2_matrix,
        translation=level2_translation,
        fixed_start_zyx=fixed_start_l2,
        moving_start_zyx=moving_start_l2,
        crop_shape_zyx=slab_shape_l2,
    )

    fused_scale_l0, fused_translation_l0 = _fused_scale_translation(root, level=0)
    fixed_translation_um, fixed_scale_l0 = _oriented_origin_spacing_um(fixed_record)
    fixed_level2_scale_um = fixed_scale_l0 * fixed_factor

    level2_to_fused0_scale = fixed_level2_scale_um / fused_scale_l0
    level0_matrix = (
        np.diag(level2_to_fused0_scale)
        @ np.asarray(level2_global_matrix, dtype=np.float64)
        @ np.diag(1.0 / moving_factor)
    )
    level0_offset = (
        (fixed_translation_um - fused_translation_l0) / fused_scale_l0
        + level2_to_fused0_scale * np.asarray(level2_global_offset, dtype=np.float64)
    )

    crop_shape = np.asarray(crop_shape_l0, dtype=np.int64)
    moving_start = np.asarray(moving_start_l0, dtype=np.int64)
    moving_center = moving_start.astype(np.float64) + (crop_shape.astype(np.float64) - 1.0) / 2.0
    fixed_center = level0_matrix @ moving_center + level0_offset
    fixed_start = np.rint(fixed_center - (crop_shape.astype(np.float64) - 1.0) / 2.0).astype(np.int64)
    fixed_stop = fixed_start + crop_shape
    local_offset = level0_offset + level0_matrix @ moving_start.astype(np.float64) - fixed_start.astype(np.float64)
    local_matrix, local_translation = _index_affine_to_center_model(
        matrix=level0_matrix,
        offset=local_offset,
        fixed_shape_zyx=crop_shape,
        moving_shape_zyx=crop_shape,
    )
    metadata = {
        "source": "level2_dense_z_phaseprimed_method8_tile",
        "level2_tile": level2.get("moving_tile"),
        "level2_status": level2.get("status"),
        "selected_slab_index": level2.get("selected_slab_index"),
        "fallback": fallback_source,
        "fixed_level_factor_zyx": fixed_factor.tolist(),
        "moving_level_factor_zyx": moving_factor.tolist(),
        "fixed_level2_scale_um_zyx": fixed_level2_scale_um.tolist(),
        "fused_level0_scale_um_zyx": fused_scale_l0.tolist(),
        "source_fixed_phaseprimed_start_level2_zyx": source_fixed_start_l2.tolist(),
        "source_fixed_metadata_start_level2_zyx": None
        if source_fixed_metadata_start_l2 is None
        else source_fixed_metadata_start_l2.tolist(),
        "fixed_metadata_start_level2_zyx": None if fixed_metadata_start_l2 is None else fixed_metadata_start_l2.tolist(),
        "fixed_phase_residual_level2_zyx": None if fixed_phase_residual_l2 is None else fixed_phase_residual_l2.tolist(),
        "fixed_phaseprimed_start_level2_zyx": fixed_start_l2.tolist(),
        "moving_slab_start_level2_zyx": moving_start_l2.tolist(),
        "selected_slab_shape_level2_zyx": slab_shape_l2.astype(int).tolist(),
        "level2_global_matrix_zyx": np.asarray(level2_global_matrix, dtype=np.float64).tolist(),
        "level2_global_offset_zyx": np.asarray(level2_global_offset, dtype=np.float64).tolist(),
        "moving_l0_to_fixed_fused_l0_matrix_zyx": level0_matrix.tolist(),
        "moving_l0_to_fixed_fused_l0_offset_zyx": level0_offset.tolist(),
        "fixed_center_l0_zyx": fixed_center.tolist(),
        "scaling_note": "moving l0 coordinates divide by moving_level_factor_zyx; fixed level2 coordinates multiply by fixed_level2_scale_um/fused_level0_scale_um",
    }
    return fixed_start, fixed_stop, local_matrix, local_translation, metadata


def _run_level0_from_level2(
    args: argparse.Namespace,
    moving_record: dict[str, Any],
    moving_tile: Any,
) -> Path:
    if args.level2_result is None:
        raise ValueError("--level2-result is required for --mode level0-from-level2")
    started = time.perf_counter()
    timing_seconds: dict[str, float] = {}
    io_read_counts = {"fixed_mask": 0, "fixed_level0": 0, "moving_level0": 0}
    from squisher_lightsheet import cross_register_method8 as register_script
    level2 = _read_json(Path(args.level2_result).resolve())
    recovery_initializer = (
        level2.get("artifact_type")
        == "lightsheet.fused_fixed_native_spatial_recovery_initializer.v1"
    )
    root = _fixed_fused_root(Path(args.fixed_fused).resolve())
    fixed_array = root["0"]
    moving_start = np.asarray(args.moving_start_zyx, dtype=np.int64)
    crop_shape = np.asarray(args.shape_zyx, dtype=np.int64)
    if level2.get("artifact_type") == "lightsheet.image10_image14_level2_dense_z_phaseprimed_method8.tile.v1":
        fixed_record = _record_by_tile_number(args.fixed_position, str(args.tile).zfill(3))
        fixed_start, fixed_stop, initial_matrix, initial_translation, initializer_metadata = _phaseprimed_tile_level0_initializer(
            level2=level2,
            fixed_record=fixed_record,
            moving_record=moving_record,
            root=root,
            moving_start_l0=moving_start,
            crop_shape_l0=crop_shape,
        )
        fused_scale, fused_translation = _fused_scale_translation(root, level=0)
    else:
        fixed_start, fixed_stop, factor, fused_scale, fused_translation = _metadata_fixed_bounds(
            fixed_fused_root=root,
            fixed_level=0,
            moving_record=moving_record,
            moving_start_l0=moving_start,
            shape_l0=crop_shape,
        )
        if tuple(factor.tolist()) != (1, 1, 1):
            raise ValueError(f"level0 fused factor should be [1,1,1], got {factor.tolist()}")
        if args.level0_initializer == "window-parent":
            global_matrix, global_offset = _local_model_to_global_index_affine(
                matrix=np.asarray(level2["selected_local_matrix_zyx"]),
                translation=np.asarray(level2["selected_local_translation_zyx"]),
                fixed_start_zyx=np.asarray(level2["fixed_start_zyx"]),
                moving_start_zyx=np.asarray(level2["moving_start_l0_zyx"]),
                crop_shape_zyx=np.asarray(level2["moving_shape_zyx"]),
            )
            initial_matrix, initial_translation = _global_index_affine_to_local_model(
                matrix=global_matrix, offset=global_offset, fixed_start_zyx=fixed_start,
                moving_start_zyx=moving_start, crop_shape_zyx=crop_shape,
            )
            initializer_metadata = {"source": "accepted_parent_window", "conversion": "parent global index affine rebased into child fixed and moving crop frames"}
        else:
            level2_factor = np.asarray(level2["level_factor_zyx"], dtype=np.float64)
            initial_matrix = np.asarray(level2["local_matrix_zyx"], dtype=np.float32)
            initial_translation = (np.asarray(level2["local_translation_zyx"], dtype=np.float64) * level2_factor).astype(np.float32)
            if "placement_affine_zyx_um" in moving_record and not recovery_initializer:
                initial_matrix, initial_translation = _placed_level2_to_level0_initializer(
                    level2, crop_shape=crop_shape, fixed_start=fixed_start,
                )
            initializer_metadata = {
                "source": "per_window_level2_metadata_method8",
                "level2_factor_zyx": level2_factor.tolist(),
                "conversion": "level2 local matrix unchanged; level2 local translation multiplied by level_factor_zyx",
            }
    configured_start = getattr(args, "starting_affine_matrix_zyx", None)
    initial_matrix, initial_matrix_source = _resolve_starting_affine_matrix(
        configured_start,
        initial_matrix,
        already_applied=(getattr(args, "level0_initializer", None) == "window-parent"
                         or "placement_affine_zyx_um" in moving_record or _initializer_refines_starting_affine(
            level2.get("artifact_type")
        )),
    )
    initializer_metadata["initial_matrix_source"] = initial_matrix_source
    fixed_array_shape = np.asarray(fixed_array.shape, dtype=np.int64)
    fixed_padding = _crop_padding_metadata(fixed_start, fixed_stop - fixed_start, fixed_array_shape)
    mask_source = None
    mask_source_info = None
    if args.fixed_mask_threshold is not None:
        read_started = time.perf_counter()
        mask_source, mask_source_info = _fused_level_crop_from_level0_bounds(
            root,
            level=int(args.fixed_mask_level),
            level0_start_zyx=fixed_start,
            level0_stop_zyx=fixed_stop,
        )
        io_read_counts["fixed_mask"] += 1
        timing_seconds["fixed_mask_initial_read"] = time.perf_counter() - read_started
        import cupy as cp

        with cp.cuda.Device(int(args.device)):
            fixed_mask_metadata, fixed_mask_rejection = _fixed_threshold_mask_metadata_from_crop(
                mask_source,
                args=args,
                source=f"fixed_fused_level{int(args.fixed_mask_level)}",
                source_info=mask_source_info,
            )
        if fixed_mask_rejection is not None:
            row = {
                "schema_version": 1,
                "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
                "status": "rejected",
                "rejection_reason": fixed_mask_rejection,
                "fixed_source": "Image_14 fused OME-Zarr",
                "fixed_fused": str(args.fixed_fused),
                "fixed_level": 0,
                "moving_tile": moving_tile.tile,
                "moving_path": str(moving_tile.path),
                "moving_channel": int(args.moving_channel),
                "placement_source": "moving_position_translation_um_in_fused_image14_grid",
                "initializer_source": str(Path(args.level2_result).resolve()),
                "initializer_conversion": initializer_metadata,
                "cache_config": _cache_config(args),
                "moving_start_l0_zyx": moving_start.tolist(),
                "moving_stop_l0_zyx": (moving_start + crop_shape).tolist(),
                "fixed_start_zyx": fixed_start.tolist(),
                "fixed_stop_zyx": fixed_stop.tolist(),
                "fixed_crop_padding": fixed_padding,
                "fixed_threshold_mask": fixed_mask_metadata,
                "quality_mask": "fixed_threshold_mask",
                "io_read_counts": io_read_counts,
                "timing_seconds": {**timing_seconds, "total": time.perf_counter() - started},
            }
            _write_json(args.output, row)
            return args.output.resolve()
    read_started = time.perf_counter()
    fixed_raw = _read_fused_crop(fixed_array, fixed_start, crop_shape)
    io_read_counts["fixed_level0"] += 1
    timing_seconds["fixed_initial_read"] = time.perf_counter() - read_started
    read_started = time.perf_counter()
    moving_raw, moving_slices = _read_level0_crop(
        moving_tile,
        channel=int(args.moving_channel),
        start_zyx=moving_start,
        crop_shape_zyx=tuple(int(v) for v in crop_shape),
        source_cache=_moving_source_cache(float(getattr(args, "source_cache_max_gib", 0.0))),
    )
    io_read_counts["moving_level0"] += 1
    timing_seconds["moving_read"] = time.perf_counter() - read_started
    import cupy as cp

    with cp.cuda.Device(int(args.device)):
        fixed_gpu, fixed_stats = _prepare_fit_volume_cupy(
            fixed_raw,
            intensity_transform=str(args.fit_intensity_transform),
        )
        moving_gpu, moving_stats = _prepare_fit_volume_cupy(
            moving_raw,
            intensity_transform=str(args.fit_intensity_transform),
        )
        fit_downsample = tuple(int(v) for v in args.fit_downsample_zyx)
        fixed_fit_gpu = _block_mean_downsample_zyx_cupy(fixed_gpu, fit_downsample)
        moving_fit_gpu = _block_mean_downsample_zyx_cupy(moving_gpu, fit_downsample)
        fit_matrix, fit_translation = register_script._model_to_fit_downsample(initial_matrix, initial_translation, fit_downsample)
        if args.fixed_mask_threshold is None:
            fixed_fit_mask_gpu, fixed_mask_metadata, fixed_mask_rejection = None, None, None
        else:
            assert mask_source is not None and mask_source_info is not None
            fixed_fit_mask_gpu, fixed_mask_metadata, fixed_mask_rejection = _prepare_fixed_mask_from_crop(
                mask_source,
                source_info=mask_source_info,
                args=args,
                fit_shape_zyx=tuple(int(value) for value in fixed_fit_gpu.shape),
            )
        content_rejection = _native_window_low_content_reason(fixed_stats, prefix="fixed")
        content_rejection = content_rejection or _native_window_low_content_reason(moving_stats, prefix="moving")
        if fixed_mask_rejection is not None or content_rejection is not None:
            rejection = fixed_mask_rejection or content_rejection
            row = {
                "schema_version": 1,
                "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
                "status": "rejected",
                "rejection_reason": rejection,
                "fixed_source": "Image_14 fused OME-Zarr",
                "fixed_fused": str(args.fixed_fused),
                "fixed_level": 0,
                "moving_tile": moving_tile.tile,
                "moving_path": str(moving_tile.path),
                "moving_channel": int(args.moving_channel),
                "placement_source": "moving_position_translation_um_in_fused_image14_grid",
                "initializer_source": str(Path(args.level2_result).resolve()),
                "initializer_conversion": initializer_metadata,
                "cache_config": _cache_config(args),
                "moving_start_l0_zyx": moving_start.tolist(),
                "moving_stop_l0_zyx": (moving_start + crop_shape).tolist(),
                "fixed_start_zyx": fixed_start.tolist(),
                "fixed_stop_zyx": fixed_stop.tolist(),
                "fixed_crop_padding": fixed_padding,
                "fixed_shape_zyx": list(fixed_raw.shape),
                "moving_shape_zyx": list(moving_raw.shape),
                "fixed_threshold_mask": fixed_mask_metadata,
                "quality_mask": "fixed_threshold_mask" if fixed_mask_metadata is not None else None,
                "fixed_content": fixed_stats,
                "moving_content": moving_stats,
                "io_read_counts": io_read_counts,
                "timing_seconds": {**timing_seconds, "total": time.perf_counter() - started},
            }
            _write_json(args.output, row)
            return args.output.resolve()
        preseed_registered = register_script._warp_fit_preseed_cupy(moving_fit_gpu, fit_matrix, fit_translation)
        preseed_corr = _corr_gpu(fixed_fit_gpu, preseed_registered, fixed_mask=fixed_fit_mask_gpu)
        preseed_gradient = gradient_component_ncc_3d_gpu(fixed_fit_gpu, preseed_registered, fixed_mask=fixed_fit_mask_gpu)
        preseed_affine_quality = _affine_quality(initial_matrix)
        preseed_status, preseed_rejection = _initializer_attempt_status(
            corr=preseed_corr,
            gradient=preseed_gradient,
            min_corr=float(args.min_corr),
            min_grad_ncc=float(args.min_grad_ncc),
            affine_quality=preseed_affine_quality,
            native_method=str(args.native_method),
        )
        phase_delta, capture = _structural_capture(
            register_script=register_script,
            fixed_fit_gpu=fixed_fit_gpu,
            moving_fit_gpu=moving_fit_gpu,
            preseed_registered=preseed_registered,
            fit_matrix=fit_matrix,
            fit_translation=fit_translation,
            fixed_fit_mask_gpu=fixed_fit_mask_gpu,
            fit_voxel_um_zyx=np.asarray(fused_scale, dtype=np.float64) * np.asarray(fit_downsample, dtype=np.float64),
        )
        phase_fit_translation = np.asarray(fit_translation, dtype=np.float32) + phase_delta
        phase_registered = register_script._warp_fit_preseed_cupy(moving_fit_gpu, fit_matrix, phase_fit_translation)
        phase_corr = _corr_gpu(fixed_fit_gpu, phase_registered, fixed_mask=fixed_fit_mask_gpu)
        phase_gradient = gradient_component_ncc_3d_gpu(fixed_fit_gpu, phase_registered, fixed_mask=fixed_fit_mask_gpu)
        phase_affine_quality = _affine_quality(initial_matrix)
        phase_status, phase_rejection = _initializer_attempt_status(
            corr=phase_corr,
            gradient=phase_gradient,
            min_corr=float(args.min_corr),
            min_grad_ncc=float(args.min_grad_ncc),
            affine_quality=phase_affine_quality,
            native_method=str(args.native_method),
        )
        phase_fit_pull_matrix, phase_fit_pull_offset = output_to_input_from_model(
            fit_matrix,
            phase_fit_translation,
            tuple(int(value) for value in fixed_fit_gpu.shape),
        )
        phase_full_matrix, phase_full_offset = _native_pull_from_fit_downsample(
            phase_fit_pull_matrix,
            phase_fit_pull_offset,
            fit_downsample,
        )
        phase_local_matrix, phase_local_translation = output_to_input_to_model(
            phase_full_matrix,
            phase_full_offset,
            tuple(int(value) for value in fixed_gpu.shape),
        )
        native_seed_matrix, native_seed_translation, native_initializer_name = _select_native_initializer(
            initial_matrix=initial_matrix,
            initial_translation=initial_translation,
            phase_matrix=phase_local_matrix,
            phase_translation=phase_local_translation,
            capture_confident=bool(capture["confident"]),
            recovery_initializer=recovery_initializer,
        )
        native_seed_global_matrix, native_seed_global_offset = _local_model_to_global_index_affine(
            matrix=native_seed_matrix,
            translation=native_seed_translation,
            fixed_start_zyx=fixed_start,
            moving_start_zyx=moving_start,
            crop_shape_zyx=crop_shape,
        )
        native_fixed_start, native_moving_start, native_crop_shape, native_fixed_shift = _pretranslated_overlap_bounds(
            fixed_start_zyx=fixed_start,
            moving_start_zyx=moving_start,
            crop_shape_zyx=crop_shape,
            phase_translation_zyx=native_seed_translation,
            fixed_array_shape_zyx=fixed_array_shape,
            moving_array_shape_zyx=np.asarray(moving_tile.shape_zyx, dtype=np.int64),
        )
        native_initial_matrix, native_initial_translation = _global_index_affine_to_local_model(
            matrix=native_seed_global_matrix,
            offset=native_seed_global_offset,
            fixed_start_zyx=native_fixed_start,
            moving_start_zyx=native_moving_start,
            crop_shape_zyx=native_crop_shape,
        )
        if not np.array_equal(native_moving_start, moving_start) or not np.array_equal(native_crop_shape, crop_shape):
            raise RuntimeError(
                "fixed pretranslation changed the moving crop; refusing to reuse moving data: "
                f"initial_start={moving_start.tolist()} native_start={native_moving_start.tolist()} "
                f"initial_shape={crop_shape.tolist()} native_shape={native_crop_shape.tolist()}"
            )
        if np.array_equal(native_fixed_start, fixed_start):
            native_fixed_gpu, native_fixed_stats = fixed_gpu, fixed_stats
            native_fixed_fit_gpu = fixed_fit_gpu
            timing_seconds["fixed_native_read"] = 0.0
        else:
            read_started = time.perf_counter()
            native_fixed_raw = _read_fused_crop(fixed_array, native_fixed_start, native_crop_shape)
            io_read_counts["fixed_level0"] += 1
            timing_seconds["fixed_native_read"] = time.perf_counter() - read_started
            native_fixed_gpu, native_fixed_stats = _prepare_fit_volume_cupy(
                native_fixed_raw,
                intensity_transform=str(args.fit_intensity_transform),
            )
            native_fixed_fit_gpu = _block_mean_downsample_zyx_cupy(native_fixed_gpu, fit_downsample)
        native_fit_matrix, native_fit_translation = register_script._model_to_fit_downsample(
            native_initial_matrix,
            native_initial_translation,
            fit_downsample,
        )
        if args.fixed_mask_threshold is None:
            native_fixed_fit_mask_gpu, native_fixed_mask_metadata, native_fixed_mask_rejection = None, None, None
        elif np.array_equal(native_fixed_start, fixed_start):
            assert mask_source is not None and mask_source_info is not None
            native_fixed_fit_mask_gpu, native_fixed_mask_metadata, native_fixed_mask_rejection = _prepare_fixed_mask_from_crop(
                mask_source,
                source_info=mask_source_info,
                args=args,
                fit_shape_zyx=tuple(int(value) for value in native_fixed_fit_gpu.shape),
            )
            timing_seconds["fixed_mask_native_read"] = 0.0
        else:
            read_started = time.perf_counter()
            native_fixed_fit_mask_gpu, native_fixed_mask_metadata, native_fixed_mask_rejection = _prepare_fixed_mask(
                root,
                args=args,
                fixed_start_l0=native_fixed_start,
                fixed_stop_l0=native_fixed_start + native_crop_shape,
                fit_shape_zyx=tuple(int(value) for value in native_fixed_fit_gpu.shape),
            )
            io_read_counts["fixed_mask"] += 1
            timing_seconds["fixed_mask_native_read"] = time.perf_counter() - read_started
        if native_fixed_mask_rejection is not None:
            # The native library exits the process on an empty objective mask, so the pretranslated crop must pass
            # the same mask gate as the initial crop before the native call.
            row = {
                "schema_version": 1,
                "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
                "status": "rejected",
                "rejection_reason": f"native_{native_fixed_mask_rejection}",
                "fixed_source": "Image_14 fused OME-Zarr",
                "fixed_fused": str(args.fixed_fused),
                "fixed_level": 0,
                "moving_tile": moving_tile.tile,
                "moving_path": str(moving_tile.path),
                "moving_channel": int(args.moving_channel),
                "placement_source": "moving_position_translation_um_in_fused_image14_grid",
                "initializer_source": str(Path(args.level2_result).resolve()),
                "initializer_conversion": initializer_metadata,
                "cache_config": _cache_config(args),
                "moving_start_l0_zyx": moving_start.tolist(),
                "moving_stop_l0_zyx": (moving_start + crop_shape).tolist(),
                "fixed_start_zyx": fixed_start.tolist(),
                "fixed_stop_zyx": fixed_stop.tolist(),
                "fixed_crop_padding": fixed_padding,
                "fixed_shape_zyx": list(fixed_raw.shape),
                "moving_shape_zyx": list(moving_raw.shape),
                # recover-native can promote this row to an interpolated fallback, whose placement needs the grid.
                "fused_scale_zyx": np.asarray(fused_scale, dtype=np.float64).tolist(),
                "fused_translation_zyx": np.asarray(fused_translation, dtype=np.float64).tolist(),
                "fixed_threshold_mask": fixed_mask_metadata,
                "quality_mask": "fixed_threshold_mask" if fixed_mask_metadata is not None else None,
                "native_pretranslated_fixed_start_zyx": native_fixed_start.tolist(),
                "native_pretranslated_fixed_stop_zyx": (native_fixed_start + native_crop_shape).tolist(),
                "native_fixed_threshold_mask": native_fixed_mask_metadata,
                "fixed_content": fixed_stats,
                "moving_content": moving_stats,
                "io_read_counts": io_read_counts,
                "timing_seconds": {**timing_seconds, "total": time.perf_counter() - started},
            }
            _write_json(args.output, row)
            return args.output.resolve()
        native_call_translation = (
            np.zeros(3, dtype=np.float32)
            if str(args.native_method) in MATTES_NATIVE_METHODS
            else np.asarray(native_fit_translation, dtype=np.float32)
        )
        native_started = time.perf_counter()
        phase_native, phase_native_stderr = _register_native_device_quiet(
            native_fixed_fit_gpu,
            moving_fit_gpu,
            fixed_mask_zyx=_native_objective_mask(args, native_fixed_fit_mask_gpu),
            native_method=str(args.native_method),
            mattes_bins=int(args.mattes_bins),
            mattes_samples=int(args.mattes_samples),
            mattes_fixed_shear=bool(args.mattes_fixed_shear),
            lib_dir=args.native_lib_dir,
            ftol=float(args.ftol),
            max_iterations=int(args.max_iterations),
            device=int(args.device),
            aff_pivot_zyx=(
                moving_xy_center_slab_z_pivot_in_fit_zyx(
                    moving_full_shape_zyx=np.asarray(moving_tile.shape_zyx, dtype=np.int64),
                    moving_crop_start_zyx=native_moving_start,
                    crop_shape_zyx=native_crop_shape,
                    local_matrix_zyx=native_initial_matrix,
                    local_translation_zyx=native_initial_translation,
                    fit_downsample_zyx=fit_downsample,
                )
                if str(args.native_method) == "method6"
                else None
            ),
            initial_matrix_xyz_3x4=_native_initial_xyz_3x4(
                native_fit_matrix,
                native_call_translation,
                tuple(int(value) for value in native_fixed_fit_gpu.shape),
                native_method=str(args.native_method),
                mattes_fixed_shear=bool(args.mattes_fixed_shear),
            ),
        )
        timing_seconds["native_fit"] = time.perf_counter() - native_started
        phase_native_full_matrix, phase_native_full_offset = _native_pull_from_fit_downsample(
            phase_native.matrix_zyx,
            phase_native.offset_zyx,
            fit_downsample,
        )
        phase_native_crop_local_matrix, phase_native_crop_local_translation = output_to_input_to_model(
            phase_native_full_matrix,
            phase_native_full_offset,
            tuple(int(value) for value in native_fixed_gpu.shape),
        )
        phase_native_global_matrix, phase_native_global_offset = _local_model_to_global_index_affine(
            matrix=phase_native_crop_local_matrix,
            translation=phase_native_crop_local_translation,
            fixed_start_zyx=native_fixed_start,
            moving_start_zyx=native_moving_start,
            crop_shape_zyx=native_crop_shape,
        )
        phase_native_local_matrix, phase_native_local_translation = _global_index_affine_to_local_model(
            matrix=phase_native_global_matrix,
            offset=phase_native_global_offset,
            fixed_start_zyx=fixed_start,
            moving_start_zyx=moving_start,
            crop_shape_zyx=crop_shape,
        )
        phase_native_corr = _corr_gpu(
            native_fixed_fit_gpu,
            phase_native.registered_zyx.astype(np.float32, copy=False),
            fixed_mask=native_fixed_fit_mask_gpu,
        )
        phase_native_gradient = gradient_component_ncc_3d_gpu(
            native_fixed_fit_gpu,
            phase_native.registered_zyx.astype(np.float32, copy=False),
            fixed_mask=native_fixed_fit_mask_gpu,
        )
        phase_native_affine_quality = _affine_quality(phase_native_local_matrix)
        phase_native_status, phase_native_rejection = _attempt_status(
            corr=phase_native_corr,
            gradient=phase_native_gradient,
            min_corr=float(args.min_corr),
            min_grad_ncc=float(args.min_grad_ncc),
            native_return_code=int(phase_native.return_code),
            affine_quality=phase_native_affine_quality,
            require_corr=str(args.native_method) not in MATTES_NATIVE_METHODS,
            require_gradient_ncc=str(args.native_method) not in MATTES_NATIVE_METHODS,
        )
        method8_attempts = [
            {
                "name": "level2_initializer_only",
                "status": preseed_status,
                "rejection_reason": preseed_rejection,
                "native_return_code": None,
                "corr_refined": None if preseed_corr is None else float(preseed_corr),
                "gradient_component_ncc_refined": preseed_gradient,
                "gradient_component_ncc_mean": _gradient_component_ncc_mean(preseed_gradient),
                "affine_quality": preseed_affine_quality,
                "local_matrix_zyx": np.asarray(initial_matrix, dtype=np.float64).tolist(),
                "local_translation_zyx": np.asarray(initial_translation, dtype=np.float64).tolist(),
            },
            {
                "name": "level2_initializer_phase_primed",
                "status": phase_status,
                "rejection_reason": phase_rejection,
                "native_return_code": None,
                "corr_refined": None if phase_corr is None else float(phase_corr),
                "gradient_component_ncc_refined": phase_gradient,
                "gradient_component_ncc_mean": _gradient_component_ncc_mean(phase_gradient),
                "affine_quality": phase_affine_quality,
                "local_matrix_zyx": np.asarray(phase_local_matrix, dtype=np.float64).tolist(),
                "local_translation_zyx": np.asarray(phase_local_translation, dtype=np.float64).tolist(),
                "translation_delta_fit_zyx": phase_delta.astype(np.float64).tolist(),
            },
        ]
        method8_attempts.append(
            {
                "name": (
                    f"native_{str(args.native_method).replace('-', '_')}"
                    f"{'_fixed_shear' if bool(args.mattes_fixed_shear) else ''}"
                    f"_from_{native_initializer_name}"
                ),
                "status": phase_native_status,
                "rejection_reason": phase_native_rejection,
                "native_return_code": int(phase_native.return_code),
                "native_records": phase_native.records.astype(np.float64).tolist(),
                "corr_refined": None if phase_native_corr is None else float(phase_native_corr),
                "gradient_component_ncc_refined": phase_native_gradient,
                "gradient_component_ncc_mean": _gradient_component_ncc_mean(phase_native_gradient),
                "affine_quality": phase_native_affine_quality,
                "local_matrix_zyx": np.asarray(phase_native_local_matrix, dtype=np.float64).tolist(),
                "local_translation_zyx": np.asarray(phase_native_local_translation, dtype=np.float64).tolist(),
                "native_pretranslated_fixed_start_zyx": native_fixed_start.tolist(),
                "native_pretranslated_fixed_stop_zyx": (native_fixed_start + native_crop_shape).tolist(),
                "native_pretranslated_moving_start_zyx": native_moving_start.tolist(),
                "native_pretranslated_moving_stop_zyx": (native_moving_start + native_crop_shape).tolist(),
                "native_pretranslated_crop_shape_zyx": native_crop_shape.tolist(),
                "native_absorbed_fixed_shift_zyx": native_fixed_shift.tolist(),
                "native_initial_matrix_zyx": np.asarray(native_initial_matrix, dtype=np.float64).tolist(),
                "native_initial_translation_zyx": np.asarray(native_initial_translation, dtype=np.float64).tolist(),
                "native_fit_initial_matrix_zyx": np.asarray(native_fit_matrix, dtype=np.float64).tolist(),
                "native_fit_initial_translation_zyx": np.asarray(native_fit_translation, dtype=np.float64).tolist(),
                "native_call_initial_translation_zyx": native_call_translation.astype(np.float64).tolist(),
                "native_method": str(args.native_method),
                "native_mattes_fixed_shear": bool(args.mattes_fixed_shear),
                "quality_policy": (
                    "native success and affine plausibility; correlation and gradient NCC diagnostic only"
                    if str(args.native_method) in MATTES_NATIVE_METHODS
                    else "native success, minimum correlation, gradient NCC, and affine plausibility"
                ),
                "native_cropped_local_matrix_zyx": np.asarray(phase_native_crop_local_matrix, dtype=np.float64).tolist(),
                "native_cropped_local_translation_zyx": np.asarray(phase_native_crop_local_translation, dtype=np.float64).tolist(),
                "native_fixed_threshold_mask": native_fixed_mask_metadata,
                "native_fixed_threshold_mask_rejection": native_fixed_mask_rejection,
                "native_fixed_content": native_fixed_stats,
                "native_moving_content": moving_stats,
                "native_moving_slices_zyx": moving_slices,
            }
        )
        if int(phase_native.return_code) != 0 and phase_native_stderr:
            method8_attempts[-1]["native_stderr"] = phase_native_stderr
        method8_attempts[2]["native_optimizer"] = {
            "evaluations": int(phase_native.records[5]),
            "evaluation_limit": int(args.max_iterations),
            "budget_exhausted": bool(phase_native.records[5] >= args.max_iterations),
            "convergence_reported": False,
        }
        if args.local_3d_refinement:
            from squisher_lightsheet.local_affine import refine_local_affine_gpu

            local_started = time.perf_counter()
            local_fit_matrix, local_fit_translation = register_script._model_to_fit_downsample(
                phase_native_local_matrix, phase_native_local_translation, fit_downsample,
            )
            local = refine_local_affine_gpu(
                fixed_fit_gpu,
                moving_fit_gpu,
                fixed_mask=fixed_fit_mask_gpu if fixed_fit_mask_gpu is not None else cp.ones_like(fixed_fit_gpu, dtype=cp.bool_),
                initial_matrix=local_fit_matrix,
                initial_translation=local_fit_translation,
                spacing_um_zyx=np.asarray(fused_scale) * np.asarray(fit_downsample),
                estimator=args.local_affine_estimator,
            )
            local_pull, local_offset = output_to_input_from_model(
                local.matrix_zyx, local.translation_zyx, fixed_fit_gpu.shape,
            )
            local_full_pull, local_full_offset = _native_pull_from_fit_downsample(local_pull, local_offset, fit_downsample)
            local_matrix, local_translation = output_to_input_to_model(local_full_pull, local_full_offset, fixed_gpu.shape)
            local_quality = _affine_quality(local_matrix)
            local_registered = register_script._warp_fit_preseed_cupy(moving_fit_gpu, local.matrix_zyx, local.translation_zyx)
            local_corr = _corr_gpu(fixed_fit_gpu, local_registered, fixed_mask=fixed_fit_mask_gpu)
            local_gradient = gradient_component_ncc_3d_gpu(fixed_fit_gpu, local_registered, fixed_mask=fixed_fit_mask_gpu)
            local_gate_status, local_gate_rejection = _attempt_status(
                corr=local_corr, gradient=local_gradient, min_corr=float(args.min_corr),
                min_grad_ncc=float(args.min_grad_ncc), native_return_code=int(phase_native.return_code),
                affine_quality=local_quality,
            )
            local_accepted = local.accepted and local_gate_status == "accepted"
            selected_attempt = {
                "name": "local_3d_affine_refinement",
                "status": "accepted" if local_accepted else "rejected",
                "rejection_reason": None if local_accepted else local.rejection_reason or local_gate_rejection,
                "local_matrix_zyx": local_matrix.tolist(),
                "local_translation_zyx": local_translation.tolist(),
                "corr_refined": local_corr,
                "gradient_component_ncc_refined": local_gradient,
                "affine_quality": local_quality,
                "local_3d_validation": {
                    "before": local.before,
                    "after": local.after,
                    "capture_spacing_um_zyx": local.capture_spacing_um_zyx.tolist(),
                    "estimator": local.estimator,
                    "diagnostics": local.diagnostics,
                    "criterion": "internal consistency of confident 3D patches within one capture voxel per axis",
                },
            }
            method8_attempts.append(selected_attempt)
            timing_seconds["local_3d_refinement"] = time.perf_counter() - local_started
        else:
            selected_attempt = next(
                (attempt for attempt in (method8_attempts[2], method8_attempts[1]) if attempt.get("status") == "accepted"),
                method8_attempts[2],
            )
        selected_name = str(selected_attempt["name"])
        selected_corr = selected_attempt.get("corr_refined")
        selected_gradient = selected_attempt["gradient_component_ncc_refined"]
        selected_status = str(selected_attempt["status"])
        selected_rejection = selected_attempt.get("rejection_reason")
        selected_matrix = np.asarray(selected_attempt["local_matrix_zyx"], dtype=np.float64)
        selected_translation = np.asarray(selected_attempt["local_translation_zyx"], dtype=np.float64)

    row = {
        "schema_version": 1,
        "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
        "fixed_source": "Image_14 fused OME-Zarr",
        "status": selected_status,
        "rejection_reason": selected_rejection,
        "fixed_fused": str(args.fixed_fused),
        "fixed_level": 0,
        "moving_tile": moving_tile.tile,
        "moving_path": str(moving_tile.path),
        "moving_channel": int(args.moving_channel),
        "placement_source": "moving_position_translation_um_in_fused_image14_grid",
        "initializer_source": str(Path(args.level2_result).resolve()),
        "initializer_conversion": initializer_metadata,
        "cache_config": _cache_config(args),
        "moving_start_l0_zyx": moving_start.tolist(),
        "moving_stop_l0_zyx": (moving_start + crop_shape).tolist(),
        "fixed_start_zyx": fixed_start.tolist(),
        "fixed_stop_zyx": fixed_stop.tolist(),
        "fixed_crop_padding": fixed_padding,
        "fixed_shape_zyx": list(fixed_raw.shape),
        "moving_shape_zyx": list(moving_raw.shape),
        "fused_scale_zyx": fused_scale.tolist(),
        "fused_translation_zyx": fused_translation.tolist(),
        "initial_matrix_zyx": initial_matrix.tolist(),
        "initial_translation_zyx": initial_translation.tolist(),
        "fit_downsample_zyx": [int(value) for value in fit_downsample],
        "fit_initial_matrix_zyx": np.asarray(fit_matrix, dtype=np.float64).tolist(),
        "fit_initial_translation_zyx": np.asarray(fit_translation, dtype=np.float64).tolist(),
        "preseed_only_corr": None if preseed_corr is None else float(preseed_corr),
        "preseed_only_gradient_component_ncc": preseed_gradient,
        "preseed_only_gradient_component_ncc_mean": _gradient_component_ncc_mean(preseed_gradient),
        "method8_attempts": method8_attempts,
        "phase_priming": {
            "translation_delta_fit_zyx": phase_delta.astype(np.float64).tolist(),
            "capture": capture,
            "corr_refined": None if phase_corr is None else float(phase_corr),
            "gradient_component_ncc_refined": phase_gradient,
            "gradient_component_ncc_mean": _gradient_component_ncc_mean(phase_gradient),
            "local_matrix_zyx": np.asarray(phase_local_matrix, dtype=np.float64).tolist(),
            "local_translation_zyx": np.asarray(phase_local_translation, dtype=np.float64).tolist(),
        },
        "selected_attempt": selected_name,
        "selected_corr_refined": None if selected_corr is None else float(selected_corr),
        "selected_gradient_component_ncc_refined": selected_gradient,
        "selected_gradient_component_ncc_mean": _gradient_component_ncc_mean(selected_gradient),
        "selected_local_matrix_zyx": selected_matrix.tolist(),
        "selected_local_translation_zyx": selected_translation.tolist(),
        **_selected_global_transforms(
            local_matrix=selected_matrix,
            local_translation=selected_translation,
            fixed_start_zyx=fixed_start,
            moving_start_zyx=moving_start,
            crop_shape_zyx=crop_shape,
        ),
        "fixed_slices_zyx": [[int(start), int(stop)] for start, stop in zip(fixed_start, fixed_stop, strict=True)],
        "moving_slices_zyx": moving_slices,
        "fixed_threshold_mask": fixed_mask_metadata,
        "quality_mask": "fixed_threshold_mask" if fixed_mask_metadata is not None else None,
        "fixed_content": fixed_stats,
        "moving_content": moving_stats,
        "native_return_code": int(phase_native.return_code),
        "native_method": str(args.native_method),
        "quality_policy": (
            "native success, minimum correlation, gradient NCC, affine plausibility, and local 3D consistency checks"
            if args.local_3d_refinement else
            "native success and affine plausibility; correlation and gradient NCC diagnostic only"
            if str(args.native_method) in MATTES_NATIVE_METHODS
            else "native success, minimum correlation, gradient NCC, and affine plausibility"
        ),
        "native_records": phase_native.records.astype(np.float64).tolist(),
        "corr_refined": None if phase_native_corr is None else float(phase_native_corr),
        "gradient_component_ncc_refined": phase_native_gradient,
        "gradient_component_ncc_mean": _gradient_component_ncc_mean(phase_native_gradient),
        "native_fit_output_to_input_matrix_zyx": np.asarray(phase_native.matrix_zyx, dtype=np.float64).tolist(),
        "native_fit_output_to_input_offset_zyx": np.asarray(phase_native.offset_zyx, dtype=np.float64).tolist(),
        "native_output_to_input_matrix_zyx": np.asarray(phase_native_full_matrix, dtype=np.float64).tolist(),
        "native_output_to_input_offset_zyx": np.asarray(phase_native_full_offset, dtype=np.float64).tolist(),
        "local_matrix_zyx": np.asarray(phase_native_local_matrix, dtype=np.float64).tolist(),
        "local_translation_zyx": np.asarray(phase_native_local_translation, dtype=np.float64).tolist(),
        "io_read_counts": io_read_counts,
        "timing_seconds": {**timing_seconds, "total": time.perf_counter() - started},
    }
    if int(phase_native.return_code) != 0 and phase_native_stderr:
        row["native_stderr"] = phase_native_stderr
    _write_json(args.output, row)
    return args.output.resolve()


def run(args: argparse.Namespace) -> Path:
    _apply_native_method_defaults(args)
    if args.local_affine_estimator != "least-squares" and not args.local_3d_refinement:
        raise ValueError("restrained estimator requires --local-3d-refinement")
    if args.local_3d_refinement and str(args.native_method) != "method6":
        raise ValueError("local 3D refinement requires Method 6")
    missing = [name for name in ("CUDA_PATH", "LD_LIBRARY_PATH") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required native/GPU env vars: {', '.join(missing)}")

    from squisher_lightsheet import cross_register_method8 as register_script
    tile_no = str(args.tile).zfill(3)
    moving_record = _record_by_tile_number(args.moving_position, tile_no)
    moving_source_record = _record_by_tile_number(args.moving_source_position, tile_no)
    moving_tile = tile_record_from_position_record(
        source_record_with_geometry(moving_source_record, moving_record)
    )
    if args.mode == "level2-metadata":
        return _run_level2_metadata(args, moving_record, moving_tile)
    if args.mode == "level0-from-level2":
        return _run_level0_from_level2(args, moving_record, moving_tile)
    fixed_record = _record_by_tile_number(args.fixed_position, tile_no)

    fixed_tile = tile_record_from_position_record(fixed_record)
    root = zarr.open_group(str(args.fixed_fused), mode="r")
    fixed_array = root[str(args.fixed_level)]
    fused_scale, fused_translation = _fused_scale_translation(root, level=int(args.fixed_level))
    fixed_tile_start_in_fused = (tile_bounds_zyx_um(fixed_tile)[0] - fused_translation) / fused_scale

    preseed_matrix, preseed_translation = register_script.preseeded_level0_model(
        fixed_tile=fixed_tile,
        moving_tile=moving_tile,
        preseed_matrix_zyx=register_script.README_PRESEED_MATRIX_ZYX,
    )
    fixed_local_forward_matrix, fixed_local_forward_offset = _center_model_to_index_affine(
        matrix=preseed_matrix,
        translation=preseed_translation,
        fixed_shape_zyx=np.asarray(fixed_tile.shape_zyx, dtype=np.int64),
        moving_shape_zyx=np.asarray(moving_tile.shape_zyx, dtype=np.int64),
    )
    fused_forward_matrix = np.asarray(fixed_local_forward_matrix, dtype=np.float64)
    fused_forward_offset = np.asarray(fixed_local_forward_offset, dtype=np.float64) + fixed_tile_start_in_fused

    moving_start = np.asarray(args.moving_start_zyx, dtype=np.int64)
    crop_shape = np.asarray(args.shape_zyx, dtype=np.int64)
    moving_center = moving_start.astype(np.float64) + (crop_shape.astype(np.float64) - 1.0) / 2.0
    fixed_center = fused_forward_matrix @ moving_center + fused_forward_offset
    fixed_start = np.rint(fixed_center - (crop_shape.astype(np.float64) - 1.0) / 2.0).astype(np.int64)
    local_offset = fused_forward_offset + fused_forward_matrix @ moving_start.astype(np.float64) - fixed_start.astype(np.float64)
    local_matrix, local_translation = _index_affine_to_center_model(
        matrix=fused_forward_matrix,
        offset=local_offset,
        fixed_shape_zyx=crop_shape,
        moving_shape_zyx=crop_shape,
    )
    local_matrix, initial_matrix_source = _resolve_starting_affine_matrix(
        getattr(args, "starting_affine_matrix_zyx", None),
        local_matrix,
    )

    started = time.perf_counter()
    fixed_array_shape = np.asarray(fixed_array.shape, dtype=np.int64)
    fixed_stop = fixed_start + crop_shape
    fixed_padding = _crop_padding_metadata(fixed_start, crop_shape, fixed_array_shape)
    if args.fixed_mask_threshold is not None:
        import cupy as cp

        mask_source, mask_source_info = _fused_level_crop_from_level0_bounds(
            root,
            level=int(args.fixed_mask_level),
            level0_start_zyx=fixed_start,
            level0_stop_zyx=fixed_stop,
        )
        with cp.cuda.Device(int(args.device)):
            early_fixed_mask_metadata, early_fixed_mask_rejection = _fixed_threshold_mask_metadata_from_crop(
                mask_source,
                args=args,
                source=f"fixed_fused_level{int(args.fixed_mask_level)}_early_gate",
                source_info=mask_source_info,
            )
        if early_fixed_mask_rejection is not None:
            row = {
                "schema_version": 1,
                "artifact_type": "lightsheet.image10_to_fused_image14_method8_trial.v1",
                "cache_config": _cache_config(args),
                "status": "rejected",
                "rejection_reason": early_fixed_mask_rejection,
                "fixed_source": "Image_14 fused OME-Zarr",
                "fixed_fused": str(args.fixed_fused),
                "fixed_level": int(args.fixed_level),
                "fixed_level_shape_zyx": fixed_array_shape.tolist(),
                "moving_tile": moving_tile.tile,
                "moving_path": str(moving_tile.path),
                "moving_channel": int(args.moving_channel),
                "moving_start_zyx": moving_start.tolist(),
                "moving_stop_zyx": (moving_start + crop_shape).tolist(),
                "fixed_start_zyx": fixed_start.tolist(),
                "fixed_stop_zyx": fixed_stop.tolist(),
                "shape_zyx": crop_shape.tolist(),
                "fixed_crop_padding": fixed_padding,
                "fixed_threshold_mask": early_fixed_mask_metadata,
                "quality_mask": "fixed_threshold_mask",
                "fixed_tile_start_in_fused_zyx": fixed_tile_start_in_fused.tolist(),
                "fused_forward_matrix_zyx": fused_forward_matrix.tolist(),
                "fused_forward_offset_zyx": fused_forward_offset.tolist(),
                "local_preseed_matrix_zyx": np.asarray(local_matrix, dtype=np.float64).tolist(),
                "initial_matrix_source": initial_matrix_source,
                "local_preseed_translation_zyx": np.asarray(local_translation, dtype=np.float64).tolist(),
                "timing_seconds": {"early_mask": time.perf_counter() - started, "total": time.perf_counter() - started},
            }
            _write_json(args.output, row)
            return args.output.resolve()
    fixed_raw = _read_fused_crop(fixed_array, fixed_start, crop_shape)
    moving_raw, moving_slices = _read_level0_crop(
        moving_tile,
        channel=int(args.moving_channel),
        start_zyx=moving_start,
        crop_shape_zyx=tuple(int(v) for v in crop_shape),
        source_cache=_moving_source_cache(float(getattr(args, "source_cache_max_gib", 0.0))),
    )
    read_seconds = time.perf_counter() - started

    import cupy as cp

    with cp.cuda.Device(int(args.device)):
        fixed_gpu, fixed_stats = _prepare_fit_volume_cupy(
            fixed_raw,
            intensity_transform=str(args.fit_intensity_transform),
        )
        moving_gpu, moving_stats = _prepare_fit_volume_cupy(
            moving_raw,
            intensity_transform=str(args.fit_intensity_transform),
        )
        fit_downsample = tuple(int(v) for v in args.fit_downsample_zyx)
        fixed_fit_gpu = _block_mean_downsample_zyx_cupy(fixed_gpu, fit_downsample)
        moving_fit_gpu = _block_mean_downsample_zyx_cupy(moving_gpu, fit_downsample)
        fixed_fit_mask_gpu, fixed_mask_metadata, fixed_mask_rejection = _prepare_fixed_mask(
            root,
            args=args,
            fixed_start_l0=fixed_start,
            fixed_stop_l0=fixed_start + crop_shape,
            fit_shape_zyx=tuple(int(value) for value in fixed_fit_gpu.shape),
        )
        content_rejection = _native_window_low_content_reason(fixed_stats, prefix="fixed")
        content_rejection = content_rejection or _native_window_low_content_reason(moving_stats, prefix="moving")
        if fixed_mask_rejection is not None or content_rejection is not None:
            rejection = fixed_mask_rejection or content_rejection
            row = {
                "schema_version": 1,
                "artifact_type": "lightsheet.image10_to_fused_image14_method8_trial.v1",
                "cache_config": _cache_config(args),
                "status": "rejected",
                "rejection_reason": rejection,
                "fixed_source": "Image_14 fused OME-Zarr",
                "fixed_fused": str(args.fixed_fused),
                "fixed_level": int(args.fixed_level),
                "moving_tile": moving_tile.tile,
                "moving_path": str(moving_tile.path),
                "moving_channel": int(args.moving_channel),
                "moving_start_zyx": moving_start.tolist(),
                "moving_stop_zyx": (moving_start + crop_shape).tolist(),
                "fixed_start_zyx": fixed_start.tolist(),
                "fixed_stop_zyx": (fixed_start + crop_shape).tolist(),
                "shape_zyx": crop_shape.tolist(),
                "fixed_threshold_mask": fixed_mask_metadata,
                "quality_mask": "fixed_threshold_mask" if fixed_mask_metadata is not None else None,
                "fixed_content": fixed_stats,
                "moving_content": moving_stats,
                "timing_seconds": {"read": read_seconds, "total": time.perf_counter() - started},
            }
            _write_json(args.output, row)
            return args.output.resolve()
        fit_matrix, fit_translation = register_script._model_to_fit_downsample(local_matrix, local_translation, fit_downsample)
        preseed_registered = register_script._warp_fit_preseed_cupy(moving_fit_gpu, fit_matrix, fit_translation)
        preseed_corr = _corr_gpu(fixed_fit_gpu, preseed_registered, fixed_mask=fixed_fit_mask_gpu)
        preseed_gradient = gradient_component_ncc_3d_gpu(fixed_fit_gpu, preseed_registered, fixed_mask=fixed_fit_mask_gpu)
        native, native_stderr = _register_native_device_quiet(
            fixed_fit_gpu,
            moving_fit_gpu,
            fixed_mask_zyx=_native_objective_mask(args, fixed_fit_mask_gpu),
            native_method=str(args.native_method),
            mattes_bins=int(args.mattes_bins),
            mattes_samples=int(args.mattes_samples),
            mattes_fixed_shear=bool(args.mattes_fixed_shear),
            lib_dir=args.native_lib_dir,
            ftol=float(args.ftol),
            max_iterations=int(args.max_iterations),
            device=int(args.device),
            aff_pivot_zyx=(
                moving_xy_center_slab_z_pivot_in_fit_zyx(
                    moving_full_shape_zyx=np.asarray(moving_tile.shape_zyx, dtype=np.int64),
                    moving_crop_start_zyx=moving_start,
                    crop_shape_zyx=crop_shape,
                    local_matrix_zyx=local_matrix,
                    local_translation_zyx=local_translation,
                    fit_downsample_zyx=fit_downsample,
                )
                if str(args.native_method) == "method6"
                else None
            ),
            initial_matrix_xyz_3x4=_native_initial_xyz_3x4(
                fit_matrix,
                fit_translation,
                tuple(int(value) for value in fixed_fit_gpu.shape),
                native_method=str(args.native_method),
                mattes_fixed_shear=bool(args.mattes_fixed_shear),
            ),
        )
        native_full_matrix, native_full_offset = _native_pull_from_fit_downsample(
            native.matrix_zyx,
            native.offset_zyx,
            fit_downsample,
        )
        refined_local_matrix, refined_local_translation = output_to_input_to_model(
            native_full_matrix,
            native_full_offset,
            tuple(int(v) for v in fixed_gpu.shape),
        )
        corr = _corr_gpu(fixed_fit_gpu, native.registered_zyx.astype(np.float32, copy=False), fixed_mask=fixed_fit_mask_gpu)
        gradient = gradient_component_ncc_3d_gpu(fixed_fit_gpu, native.registered_zyx.astype(np.float32, copy=False), fixed_mask=fixed_fit_mask_gpu)
        affine_quality = _affine_quality(refined_local_matrix)

    row = {
        "schema_version": 1,
        "artifact_type": "lightsheet.image10_to_fused_image14_method8_trial.v1",
        "fixed_source": "Image_14 fused OME-Zarr",
        "fixed_fused": str(args.fixed_fused),
        "fixed_level": int(args.fixed_level),
        "moving_tile": moving_tile.tile,
        "moving_path": str(moving_tile.path),
        "moving_channel": int(args.moving_channel),
        "moving_start_zyx": moving_start.tolist(),
        "moving_stop_zyx": (moving_start + crop_shape).tolist(),
        "fixed_start_zyx": fixed_start.tolist(),
        "fixed_stop_zyx": (fixed_start + crop_shape).tolist(),
        "shape_zyx": crop_shape.tolist(),
        "fixed_crop_padding": fixed_padding,
        "fixed_tile_start_in_fused_zyx": fixed_tile_start_in_fused.tolist(),
        "preseed_matrix_zyx": np.asarray(preseed_matrix, dtype=np.float64).tolist(),
        "preseed_translation_tile_local_zyx": np.asarray(preseed_translation, dtype=np.float64).tolist(),
        "fused_forward_matrix_zyx": fused_forward_matrix.tolist(),
        "fused_forward_offset_zyx": fused_forward_offset.tolist(),
        "local_preseed_matrix_zyx": np.asarray(local_matrix, dtype=np.float64).tolist(),
        "initial_matrix_source": initial_matrix_source,
        "local_preseed_translation_zyx": np.asarray(local_translation, dtype=np.float64).tolist(),
        "fit_preseed_matrix_zyx": np.asarray(fit_matrix, dtype=np.float64).tolist(),
        "fit_preseed_translation_zyx": np.asarray(fit_translation, dtype=np.float64).tolist(),
        "native_initial_matrix_xyz_3x4": zyx_to_xyz_3x4(fit_matrix, fit_translation).tolist(),
        "preseed_only_corr": None if preseed_corr is None else float(preseed_corr),
        "preseed_only_gradient_component_ncc": preseed_gradient,
        "preseed_only_gradient_component_ncc_mean": _gradient_component_ncc_mean(preseed_gradient),
        "fixed_slices_zyx": [[int(v), int(v + s)] for v, s in zip(fixed_start, crop_shape, strict=True)],
        "moving_slices_zyx": moving_slices,
        "fixed_threshold_mask": fixed_mask_metadata,
        "quality_mask": "fixed_threshold_mask" if fixed_mask_metadata is not None else None,
        "fixed_content": fixed_stats,
        "moving_content": moving_stats,
        "native_return_code": int(native.return_code),
        "native_method": str(args.native_method),
        "quality_policy": (
            "native success and affine plausibility; correlation and gradient NCC diagnostic only"
            if str(args.native_method) in MATTES_NATIVE_METHODS
            else "native success, minimum correlation, gradient NCC, and affine plausibility"
        ),
        "corr_refined": None if corr is None else float(corr),
        "gradient_component_ncc_refined": gradient,
        "gradient_component_ncc_mean": _gradient_component_ncc_mean(gradient),
        "native_fit_output_to_input_matrix_zyx": np.asarray(native.matrix_zyx, dtype=np.float64).tolist(),
        "native_fit_output_to_input_offset_zyx": np.asarray(native.offset_zyx, dtype=np.float64).tolist(),
        "native_output_to_input_matrix_zyx": np.asarray(native_full_matrix, dtype=np.float64).tolist(),
        "native_output_to_input_offset_zyx": np.asarray(native_full_offset, dtype=np.float64).tolist(),
        "local_matrix_zyx": np.asarray(refined_local_matrix, dtype=np.float64).tolist(),
        "local_translation_zyx": np.asarray(refined_local_translation, dtype=np.float64).tolist(),
        "affine_quality": affine_quality,
        "timing_seconds": {"read": read_seconds, "total": time.perf_counter() - started},
    }
    status, rejection = _attempt_status(
        corr=corr,
        gradient=gradient,
        min_corr=float(args.min_corr),
        min_grad_ncc=float(args.min_grad_ncc),
        native_return_code=int(native.return_code),
        affine_quality=affine_quality,
        require_corr=str(args.native_method) not in MATTES_NATIVE_METHODS,
        require_gradient_ncc=str(args.native_method) not in MATTES_NATIVE_METHODS,
    )
    row["status"] = status
    row["rejection_reason"] = rejection
    if int(native.return_code) != 0 and native_stderr:
        row["native_stderr"] = native_stderr
    _write_json(args.output, row)
    return args.output.resolve()


def _parse_zyx(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"expected z,y,x tuple, got {text!r}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trial Image10 tile/window method8 against fused Image14 fixed zarr.")
    parser.add_argument("--fixed-position", type=Path, default=DEFAULT_FIXED_POSITION)
    parser.add_argument("--moving-position", type=Path, default=DEFAULT_MOVING_POSITION)
    parser.add_argument("--moving-source-position", type=Path, default=DEFAULT_MOVING_POSITION)
    parser.add_argument("--fixed-fused", type=Path, default=DEFAULT_FIXED_FUSED)
    parser.add_argument("--fixed-level", type=int, default=0)
    parser.add_argument("--moving-level", type=int, default=2)
    parser.add_argument("--mode", choices=("level0-preseed", "level2-metadata", "level0-from-level2"), default="level0-preseed")
    parser.add_argument("--level2-result", type=Path, default=None)
    parser.add_argument("--tile", default="042")
    parser.add_argument("--moving-channel", type=int, default=0)
    parser.add_argument("--source-cache-max-gib", type=float, default=0.0)
    parser.add_argument(
        "--fit-intensity-transform",
        choices=("linear", "log1p"),
        default="linear",
        help="Transform both fixed and moving intensities before robust normalization, phase priming, and native fitting.",
    )
    parser.add_argument("--moving-start-zyx", type=_parse_zyx, default=(480, 0, 0))
    parser.add_argument("--shape-zyx", type=_parse_zyx, default=(528, 528, 528))
    parser.add_argument("--fit-downsample-zyx", type=_parse_zyx, default=(1, 1, 1))
    parser.add_argument("--native-lib-dir", type=Path, default=DEFAULT_LIB_DIR)
    parser.add_argument(
        "--native-method",
        choices=("method6", "method8", "method10-mattes", "method11-mattes"),
        default="method8",
    )
    parser.add_argument(
        "--starting-affine-matrix-zyx",
        type=_parse_starting_affine_matrix_zyx,
        default=None,
        help=(
            "Optional 3x3 row-major moving-to-fixed matrix in the local centered ZYX model. "
            "Defaults to the accepted slab-pivot trial median and replaces only the starting linear block; "
            "placement/phase continues to own translation."
        ),
    )
    parser.add_argument("--mattes-bins", type=int, default=50)
    parser.add_argument("--mattes-samples", type=int, default=100000)
    parser.add_argument(
        "--mattes-fixed-shear",
        action="store_true",
        help="Preserve the supplied symmetric shear and optimize translation only in native Mattes.",
    )
    parser.add_argument("--ftol", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--local-3d-refinement", action="store_true", help="Refine Method 6 from local 3D correspondences and require internal spatial consistency.")
    parser.add_argument("--local-affine-estimator", choices=("least-squares", "restrained"), default="least-squares")
    parser.add_argument("--phase-upsample-factor", type=int, default=10)
    parser.add_argument("--window-initializer-dir", type=Path, default=None)
    parser.add_argument("--parent-window-dir", type=Path, default=None)
    parser.add_argument("--min-corr", type=float, default=0.15)
    parser.add_argument("--min-grad-ncc", type=float, default=0.24)
    parser.add_argument("--fixed-mask-threshold", type=float, default=3000.0)
    parser.add_argument("--fixed-mask-level", type=int, default=2)
    parser.add_argument("--fixed-mask-min-voxels", type=int, default=256)
    parser.add_argument("--fixed-mask-max-masked-fraction", type=float, default=0.95)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mattes_bins < 5:
        raise ValueError("--mattes-bins must be >= 5")
    if args.mattes_samples < args.mattes_bins:
        raise ValueError("--mattes-samples must be >= --mattes-bins")
    if args.mattes_fixed_shear and args.native_method not in MATTES_NATIVE_METHODS:
        raise ValueError("--mattes-fixed-shear requires --native-method method10-mattes or method11-mattes")
    _resolve_starting_affine_matrix(args.starting_affine_matrix_zyx, np.eye(3, dtype=np.float32))
    output = run(args)
    print(output, flush=True)


if __name__ == "__main__":
    main()
