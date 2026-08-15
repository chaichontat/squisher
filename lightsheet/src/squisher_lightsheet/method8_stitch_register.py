from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as stitch_legacy
from squisher_lightsheet.artifact_io import (
    registration_input_fingerprint,
    sha256_file,
    write_text_set_atomic,
)
from squisher_lightsheet.channel_affine import (
    _block_mean_downsample_zyx_cupy,
    _corr_gpu,
    _gradient_component_ncc_mean,
    _native_pull_from_fit_downsample,
    _robust_norm_and_content_stats_cupy,
    gradient_component_ncc_3d_gpu,
    output_to_input_to_model,
)
from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR, register_method8_device, zyx_to_xyz_3x4
from squisher_lightsheet.ngff import axes as ngff_axes
from squisher_lightsheet.ngff import open_level_array
from squisher_lightsheet.seams import (
    BoundaryConstraint,
    RobustBoundarySettings,
    center_z_content_prefilter_reason,
)

DEFAULT_FIXED_MASK_THRESHOLD = 3000.0
DIMENSIONS = ("z", "y", "x")
CHUNK_SHIFT_VARIANCE_FLOOR_PX2 = 0.03
SPARSE_VARIANCE_PRIOR_PERCENTILE = 90.0
PHASE_INPUT_REJECTION_REASONS = frozenset(
    {
        "fixed_threshold_fit_mask_too_few_voxels",
        "fixed_threshold_fit_mask_too_sparse",
        "level2_low_content",
        "low_center_z_p99",
        "low_center_z_std",
    }
)
PHASE_RECOVERABLE_REJECTION_REASONS = frozenset(
    {
        "phase_shift_wrap_risk",
        "phase_gradient_component_ncc_too_low",
        "phase_corr_too_low",
    }
)
PhaseRecoveryDisposition = Literal["source_accepted", "recoverable", "terminal"]


@dataclass(frozen=True)
class TileInfo:
    tile_id: str
    tile_name: str
    path: Path
    start_um_zyx: np.ndarray
    spacing_um_zyx: np.ndarray
    shape_zyx: np.ndarray
    channel: int


@dataclass(frozen=True)
class Method8RegistrationOutputs:
    method8_summary: Path
    optimized_positions: Path
    diagnostics: Path
    constraints_jsonl: Path
    tile_corrections: Path


def _threshold_slug(threshold: float | None) -> str:
    if threshold is None:
        return "unmasked"
    threshold_float = float(threshold)
    if threshold_float.is_integer():
        return f"mask{int(threshold_float)}"
    return f"mask{str(threshold_float).replace('.', 'p')}"


def _optimized_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "diagnostics": output_dir / "registration.optimization.diagnostics.json",
        "positions": output_dir / "registration.optimized.positions.json",
        "constraints": output_dir / "registration.constraints.jsonl",
        "corrections": output_dir / "registration.tile-corrections.json",
    }


def _require_optimized_outputs_absent(output_dir: Path) -> None:
    existing = [path for path in _optimized_output_paths(output_dir).values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing registration output(s): {existing}")


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _load_array(path: Path) -> Any:
    return open_level_array(path)


def _load_axes(path: Path, array: Any) -> str:
    import zarr

    group = zarr.open_group(str(path), mode="r")
    return ngff_axes(group, array)


def _record_vector_zyx(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([float(values[dim]) for dim in DIMENSIONS], dtype=np.float64)


def _set_record_vector_zyx(record: dict[str, Any], key: str, values: np.ndarray) -> None:
    record[key] = {dim: float(value) for dim, value in zip(DIMENSIONS, values, strict=True)}


def _tile_id(tile_name: str) -> str:
    name = Path(tile_name).name
    for suffix in (".ome.zarr", ".ome.tif", ".ome.tiff"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.rsplit(".", 1)[-1]


def _zarr_name(tile_name: str) -> str:
    name = Path(tile_name).name
    for suffix in (".ome.tif", ".ome.tiff"):
        if name.endswith(suffix):
            return f"{name[: -len(suffix)]}.ome.zarr"
    return name


def _load_tiles(position_json: Path, zarr_dir: Path, *, channel: int = 0) -> dict[str, TileInfo]:
    payload = json.loads(position_json.read_text())
    if payload.get("units") != "micrometer":
        raise ValueError(f"{position_json} must declare units='micrometer'")
    tiles: dict[str, TileInfo] = {}
    for record in payload["tiles"]:
        tile_name = _zarr_name(str(record["tile"]))
        tile_id = _tile_id(tile_name)
        if not tile_id.isdigit():
            raise ValueError(f"tile {tile_name!r} must end in a numeric tile ID before .ome.zarr")
        if tile_id in tiles:
            raise ValueError(f"duplicate tile ID {tile_id!r} in {position_json}")
        path = zarr_dir / tile_name
        if not path.exists():
            raise FileNotFoundError(f"Missing tile OME-Zarr for {record['tile']}: {path}")
        array = _load_array(path)
        axes = _load_axes(path, array)
        if axes == "CZYX":
            if not 0 <= int(channel) < int(array.shape[0]):
                raise ValueError(f"channel {channel} is outside CZYX shape {array.shape} in {path}")
            shape_zyx = array.shape[1:]
        elif axes == "ZYX":
            if int(channel) != 0:
                raise ValueError(f"channel {channel} requested from ZYX array in {path}")
            shape_zyx = array.shape
        else:
            raise ValueError(f"expected ZYX or CZYX level 0 in {path}, found axes {axes!r}")
        tiles[tile_id] = TileInfo(
            tile_id=tile_id,
            tile_name=tile_name,
            path=path,
            start_um_zyx=_record_vector_zyx(record, "translation_um"),
            spacing_um_zyx=_record_vector_zyx(record, "scale_um"),
            shape_zyx=np.asarray(shape_zyx, dtype=np.int64),
            channel=int(channel),
        )
    return tiles


def _read_tile_crop(tile: TileInfo, slices_zyx: tuple[slice, slice, slice]) -> np.ndarray:
    array = _load_array(tile.path)
    selection = (tile.channel, *slices_zyx) if array.ndim == 4 else slices_zyx
    return np.asarray(array[selection], dtype=np.float32)


def _zarr_shape(path: Path, *, channel: int = 0) -> tuple[int, ...]:
    array = _load_array(path)
    axes = _load_axes(path, array)
    shape = tuple(int(value) for value in array.shape)
    if axes == "CZYX":
        if not 0 <= int(channel) < shape[0]:
            raise ValueError(f"channel {channel} is outside CZYX shape {shape} in {path}")
        return shape[1:]
    if axes != "ZYX":
        raise ValueError(f"expected ZYX or CZYX level 0 at {path}, found axes {axes!r}")
    if int(channel) != 0:
        raise ValueError(f"channel {channel} requested from ZYX array in {path}")
    return shape


def _load_position_tiles(
    position_json: Path,
    zarr_dir: Path,
    *,
    channel: int = 0,
) -> tuple[dict[str, Any], list[str], dict[str, int], np.ndarray, list[stitch_legacy.TileMetadata]]:
    payload = json.loads(position_json.read_text())
    if payload.get("units") != "micrometer":
        raise ValueError(f"{position_json} must declare units='micrometer'")
    records = payload["tiles"]
    tile_names = [_zarr_name(str(record["tile"])) for record in records]
    tile_ids = [_tile_id(name) for name in tile_names]
    if any(not tile_id.isdigit() for tile_id in tile_ids):
        raise ValueError("position tile names must end in numeric IDs before .ome.zarr")
    if len(set(tile_ids)) != len(tile_ids):
        raise ValueError(f"duplicate tile IDs in {position_json}")
    tile_index = {tile_id: index for index, tile_id in enumerate(tile_ids)}
    spacing = _record_vector_zyx(records[0], "scale_um")
    track = stitch_legacy.TrackMetadata(slug="track0", track_id="track0", channels=(0,), channel_names=("0",))
    tiles = [
        stitch_legacy.TileMetadata(
            path=zarr_dir / tile_name,
            shape=_zarr_shape(zarr_dir / tile_name, channel=channel),
            axes="ZYX",
            spacing={dim: float(value) for dim, value in zip(DIMENSIONS, spacing, strict=True)},
            translation={
                dim: float(_record_vector_zyx(record, "translation_um")[index])
                for index, dim in enumerate(DIMENSIONS)
            },
            channels=("0",),
            tracks=(track,),
        )
        for record, tile_name in zip(records, tile_names, strict=True)
    ]
    return payload, tile_names, tile_index, spacing, tiles


def _slice_json(slices: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(s.start), int(s.stop)] for s in slices]


def _slice_tuple(values: list[list[int]]) -> tuple[slice, slice, slice]:
    if len(values) != 3:
        raise ValueError(f"Expected 3 slices, got {values}")
    return tuple(slice(int(start), int(stop)) for start, stop in values)


def _crop_bounds_for_pair(
    fixed: TileInfo,
    moving: TileInfo,
    *,
    z_start: int,
    z_depth: int,
) -> tuple[str, tuple[slice, slice, slice], tuple[slice, slice, slice], np.ndarray]:
    if not np.allclose(fixed.spacing_um_zyx, moving.spacing_um_zyx, rtol=0.0, atol=1e-9):
        raise ValueError(f"spacing differs for {fixed.tile_name} and {moving.tile_name}")
    if not np.array_equal(fixed.shape_zyx, moving.shape_zyx):
        raise ValueError(f"shape differs for {fixed.tile_name} and {moving.tile_name}")

    spacing_abs = np.abs(fixed.spacing_um_zyx)
    fixed_stop_um = fixed.start_um_zyx + fixed.shape_zyx * fixed.spacing_um_zyx
    moving_stop_um = moving.start_um_zyx + moving.shape_zyx * moving.spacing_um_zyx
    fixed_min_um = np.minimum(fixed.start_um_zyx, fixed_stop_um)
    fixed_max_um = np.maximum(fixed.start_um_zyx, fixed_stop_um)
    moving_min_um = np.minimum(moving.start_um_zyx, moving_stop_um)
    moving_max_um = np.maximum(moving.start_um_zyx, moving_stop_um)
    overlap_start_um = np.maximum(fixed_min_um, moving_min_um)
    overlap_stop_um = np.minimum(fixed_max_um, moving_max_um)
    overlap_px = np.floor((overlap_stop_um - overlap_start_um) / spacing_abs + 1e-6).astype(np.int64)
    if np.any(overlap_px <= 0):
        raise ValueError(f"{fixed.tile_name} and {moving.tile_name} do not overlap")

    lateral = overlap_px[1:3]
    seam_axis_local = int(np.argmin(lateral)) + 1
    seam_axis = DIMENSIONS[seam_axis_local]
    across_axis = 2 if seam_axis_local == 1 else 1
    shape = fixed.shape_zyx.copy()
    crop_shape = np.minimum(overlap_px, shape)
    crop_shape[0] = min(int(z_depth), int(shape[0]), int(overlap_px[0]))
    crop_shape[seam_axis_local] = int(overlap_px[seam_axis_local])
    crop_shape[across_axis] = int(overlap_px[across_axis])

    if crop_shape[0] < 64 or crop_shape[1] < 64 or crop_shape[2] < 64:
        raise ValueError(f"crop too small for method8: {crop_shape.tolist()} overlap={overlap_px.tolist()}")

    crop_size_um = crop_shape * spacing_abs
    start_um = overlap_start_um.copy()
    fixed_z_start = min(max(int(z_start), 0), int(shape[0] - crop_shape[0]))
    fixed_z_a = fixed.start_um_zyx[0] + fixed_z_start * fixed.spacing_um_zyx[0]
    fixed_z_b = fixed.start_um_zyx[0] + (fixed_z_start + crop_shape[0]) * fixed.spacing_um_zyx[0]
    start_um[0] = np.clip(
        min(float(fixed_z_a), float(fixed_z_b)),
        float(overlap_start_um[0]),
        float(overlap_stop_um[0] - crop_size_um[0]),
    )

    def _slice_start(tile_start_um: np.ndarray, spacing_um: np.ndarray) -> np.ndarray:
        crop_high_um = start_um + crop_size_um
        coordinate_um = np.where(spacing_um >= 0, start_um, crop_high_um)
        return np.rint((coordinate_um - tile_start_um) / spacing_um).astype(np.int64)

    fixed_start = _slice_start(fixed.start_um_zyx, fixed.spacing_um_zyx)
    moving_start = _slice_start(moving.start_um_zyx, moving.spacing_um_zyx)

    fixed_start[across_axis] = int(
        np.clip(fixed_start[across_axis], 0, shape[across_axis] - crop_shape[across_axis])
    )
    moving_start[across_axis] = int(
        np.clip(moving_start[across_axis], 0, shape[across_axis] - crop_shape[across_axis])
    )
    fixed_start[0] = int(np.clip(fixed_start[0], 0, shape[0] - crop_shape[0]))
    moving_start[0] = int(np.clip(moving_start[0], 0, shape[0] - crop_shape[0]))

    fixed_slices = tuple(
        slice(int(start), int(start + size)) for start, size in zip(fixed_start, crop_shape, strict=True)
    )
    moving_slices = tuple(
        slice(int(start), int(start + size)) for start, size in zip(moving_start, crop_shape, strict=True)
    )
    return seam_axis, fixed_slices, moving_slices, crop_shape


def _fit_downsample_for_shape(shape_zyx: np.ndarray, seam_axis: str) -> tuple[int, int, int]:
    factors = [2, 2, 2]
    thin_axis = 1 if seam_axis == "y" else 2
    factors[thin_axis] = 1
    for axis, size in enumerate(shape_zyx):
        while size % factors[axis] != 0 and factors[axis] > 1:
            factors[axis] -= 1
    return tuple(factors)


def _z_chunks(z_size: int, count: int) -> list[tuple[int, int]]:
    if count < 1:
        raise ValueError(f"chunk count must be positive, got {count}")
    edges = np.linspace(0, int(z_size), int(count) + 1)
    starts = np.floor(edges[:-1]).astype(np.int64)
    stops = np.floor(edges[1:]).astype(np.int64)
    stops[-1] = int(z_size)
    return [
        (int(start), int(stop)) for start, stop in zip(starts, stops, strict=True) if int(stop) > int(start)
    ]


def _shifted_moving_slices_for_prior(
    base_slices: tuple[slice, slice, slice],
    shape_zyx: np.ndarray,
    prior_zyx: np.ndarray,
) -> tuple[tuple[slice, slice, slice], np.ndarray, np.ndarray]:
    crop_shape = np.asarray([slc.stop - slc.start for slc in base_slices], dtype=np.int64)
    base_start = np.asarray([slc.start for slc in base_slices], dtype=np.int64)
    requested_offset = np.rint(-np.asarray(prior_zyx, dtype=np.float32)).astype(np.int64)
    shifted_start = np.clip(base_start + requested_offset, 0, shape_zyx - crop_shape)
    actual_offset = shifted_start - base_start
    residual_prior = np.asarray(prior_zyx, dtype=np.float32) + actual_offset.astype(np.float32)
    shifted_slices = tuple(
        slice(int(start), int(start + size)) for start, size in zip(shifted_start, crop_shape, strict=True)
    )
    return shifted_slices, actual_offset.astype(np.int64), residual_prior.astype(np.float32)


def _row_edge_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _tile_id(str(row["fixed_tile"])),
        _tile_id(str(row["moving_tile"])),
        str(row["seam_axis"]),
    )


def _axis_priors_from_phase_rows(
    rows: list[dict[str, Any]],
    *,
    min_phase_grad: float,
    min_phase_corr: float,
    min_edges_per_axis: int,
) -> tuple[dict[str, np.ndarray], set[tuple[str, str, str]], dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _phase_gate(row, min_phase_grad=min_phase_grad, min_phase_corr=min_phase_corr) is None:
            grouped[_row_edge_key(row)].append(row)

    edge_medians_by_axis: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for _edge_key, edge_rows in grouped.items():
        axis = str(edge_rows[0]["seam_axis"])
        shifts = np.asarray([row["phase_shift_zyx"] for row in edge_rows], dtype=np.float64)
        edge_medians_by_axis[axis].append(np.median(shifts, axis=0))

    priors: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for axis, shifts in edge_medians_by_axis.items():
        values = np.asarray(shifts, dtype=np.float64)
        diagnostics[axis] = {
            "edge_count": int(len(values)),
            "median_shift_zyx": np.median(values, axis=0).tolist(),
            "mean_shift_zyx": np.mean(values, axis=0).tolist(),
        }
        if len(values) >= int(min_edges_per_axis):
            priors[axis] = np.median(values, axis=0).astype(np.float32)
    return priors, set(grouped), diagnostics


def _measure_prior_shifted_phase_z_chunk(
    *,
    fixed: TileInfo,
    moving: TileInfo,
    z_start: int,
    z_stop: int,
    device: int,
    axis_prior_zyx: np.ndarray,
    fixed_mask_threshold: float | None,
    fixed_mask_min_voxels: int,
    fixed_mask_max_masked_fraction: float,
) -> dict[str, Any]:
    import cupy as cp
    from cucim.skimage.registration import phase_cross_correlation
    from cupyx.scipy.ndimage import shift as gpu_shift

    started = time.perf_counter()
    seam_axis, fixed_slices, moving_slices_base, crop_shape = _crop_bounds_for_pair(
        fixed,
        moving,
        z_start=int(z_start),
        z_depth=int(z_stop - z_start),
    )
    moving_slices, crop_offset_zyx, residual_prior_zyx = _shifted_moving_slices_for_prior(
        moving_slices_base,
        moving.shape_zyx,
        np.asarray(axis_prior_zyx, dtype=np.float32),
    )
    fit_downsample = _fit_downsample_for_shape(crop_shape, seam_axis)
    fixed_raw = _read_tile_crop(fixed, fixed_slices)
    moving_raw = _read_tile_crop(moving, moving_slices)
    mask_rejection_reason, fixed_center_z_content, moving_center_z_content = (
        center_z_content_prefilter_reason(fixed_raw, moving_raw, RobustBoundarySettings())
    )

    cp.cuda.Device(int(device)).use()
    fixed_gpu, fixed_content = _robust_norm_and_content_stats_cupy(fixed_raw)
    moving_gpu, moving_content = _robust_norm_and_content_stats_cupy(moving_raw)
    fixed_fit = _block_mean_downsample_zyx_cupy(fixed_gpu, fit_downsample)
    moving_fit = _block_mean_downsample_zyx_cupy(moving_gpu, fit_downsample)
    fixed_fit_mask = None
    fixed_threshold_mask = None
    if fixed_mask_threshold is not None:
        raw_mask = cp.asarray(fixed_raw > float(fixed_mask_threshold), dtype=cp.bool_)
        fixed_fit_mask = _block_any_downsample_zyx_cupy(raw_mask, fit_downsample)
        source_stats = _mask_stats(raw_mask)
        fit_stats = _mask_stats(fixed_fit_mask)
        fixed_threshold_mask = {
            "threshold": float(fixed_mask_threshold),
            "source": source_stats,
            "fit": fit_stats,
        }
        if mask_rejection_reason is None and int(fit_stats["voxel_count"]) < int(
            fixed_mask_min_voxels
        ):
            mask_rejection_reason = "fixed_threshold_fit_mask_too_few_voxels"
        elif mask_rejection_reason is None and float(fit_stats["masked_fraction"]) > float(
            fixed_mask_max_masked_fraction
        ):
            mask_rejection_reason = "fixed_threshold_fit_mask_too_sparse"

    if mask_rejection_reason is not None:
        return {
            "fixed_tile": fixed.tile_name,
            "moving_tile": moving.tile_name,
            "z_start": int(z_start),
            "z_stop": int(z_stop),
            "seam_axis": seam_axis,
            "fixed_slices_zyx": _slice_json(fixed_slices),
            "moving_slices_zyx": _slice_json(moving_slices),
            "moving_slices_base_zyx": _slice_json(moving_slices_base),
            "window_shape_zyx": [int(value) for value in crop_shape],
            "fit_downsample_zyx": [int(value) for value in fit_downsample],
            "fit_shape_zyx": [int(value) for value in fixed_fit.shape],
            "fixed_content": fixed_content,
            "moving_content": moving_content,
            "fixed_center_z_content": fixed_center_z_content,
            "moving_center_z_content": moving_center_z_content,
            "fixed_threshold_mask": fixed_threshold_mask,
            "quality_mask": "fixed_threshold_mask",
            "measurement_mode": "phase_recovery_prior_shifted_crop",
            "axis_prior_zyx": np.asarray(axis_prior_zyx, dtype=np.float64).tolist(),
            "moving_crop_offset_zyx": crop_offset_zyx.astype(np.int64).tolist(),
            "residual_prior_zyx": residual_prior_zyx.astype(np.float64).tolist(),
            "phase_shift_fit_zyx": None,
            "phase_shift_shifted_crop_zyx": None,
            "phase_shift_zyx": None,
            "phase_shift_wrap_risk_axes": [],
            "phase_shift_wrap_risk": False,
            "phase_error": None,
            "phase_diff": None,
            "corr_initial": None,
            "corr_phase": None,
            "gradient_component_ncc_initial": None,
            "gradient_component_ncc_phase": None,
            "gradient_component_ncc_initial_mean": None,
            "gradient_component_ncc_phase_mean": None,
            "method8_zero_z_shear": True,
            "native_return_code": None,
            "corr_method8": None,
            "gradient_component_ncc_method8": None,
            "gradient_component_ncc_method8_mean": None,
            "native_fit_output_to_input_matrix_zyx": None,
            "native_fit_output_to_input_offset_zyx": None,
            "native_output_to_input_matrix_zyx": None,
            "native_output_to_input_offset_zyx": None,
            "local_matrix_zyx": None,
            "local_translation_zyx": None,
            "status": "rejected",
            "rejection_reason": mask_rejection_reason,
            "timing_seconds": {
                "phase": 0.0,
                "method8": 0.0,
                "total": time.perf_counter() - started,
            },
        }

    corr_initial = _corr_gpu(fixed_fit, moving_fit)
    grad_initial = gradient_component_ncc_3d_gpu(fixed_fit, moving_fit)
    phase_started = time.perf_counter()
    phase_shift_fit, phase_error, phase_diff = phase_cross_correlation(
        fixed_fit,
        moving_fit,
        upsample_factor=10,
    )
    phase_shift_fit_np = cp.asnumpy(phase_shift_fit).astype(np.float32)
    phase_shift_shifted_crop = phase_shift_fit_np.astype(np.float64) * np.asarray(fit_downsample)
    phase_shift_effective = phase_shift_shifted_crop - crop_offset_zyx.astype(np.float64)
    wrap_risk_axes = [
        DIMENSIONS[index]
        for index, (shift_value, extent) in enumerate(zip(phase_shift_shifted_crop, crop_shape, strict=True))
        if abs(float(shift_value)) > 0.25 * float(extent)
    ]
    phase_registered = gpu_shift(
        moving_fit,
        shift=tuple(float(value) for value in phase_shift_fit_np),
        order=1,
        mode="constant",
        cval=0.0,
    )
    corr_phase = _corr_gpu(fixed_fit, phase_registered)
    grad_phase = gradient_component_ncc_3d_gpu(fixed_fit, phase_registered, fixed_mask=fixed_fit_mask)

    return {
        "fixed_tile": fixed.tile_name,
        "moving_tile": moving.tile_name,
        "z_start": int(z_start),
        "z_stop": int(z_stop),
        "seam_axis": seam_axis,
        "fixed_slices_zyx": _slice_json(fixed_slices),
        "moving_slices_zyx": _slice_json(moving_slices),
        "moving_slices_base_zyx": _slice_json(moving_slices_base),
        "window_shape_zyx": [int(value) for value in crop_shape],
        "fit_downsample_zyx": [int(value) for value in fit_downsample],
        "fit_shape_zyx": [int(value) for value in fixed_fit.shape],
        "fixed_content": fixed_content,
        "moving_content": moving_content,
        "fixed_center_z_content": fixed_center_z_content,
        "moving_center_z_content": moving_center_z_content,
        "fixed_threshold_mask": fixed_threshold_mask,
        "quality_mask": "fixed_threshold_mask" if fixed_fit_mask is not None else None,
        "measurement_mode": "phase_recovery_prior_shifted_crop",
        "axis_prior_zyx": np.asarray(axis_prior_zyx, dtype=np.float64).tolist(),
        "moving_crop_offset_zyx": crop_offset_zyx.astype(np.int64).tolist(),
        "residual_prior_zyx": residual_prior_zyx.astype(np.float64).tolist(),
        "phase_shift_fit_zyx": phase_shift_fit_np.astype(np.float64).tolist(),
        "phase_shift_shifted_crop_zyx": phase_shift_shifted_crop.tolist(),
        "phase_shift_zyx": phase_shift_effective.tolist(),
        "phase_shift_wrap_risk_axes": wrap_risk_axes,
        "phase_shift_wrap_risk": bool(wrap_risk_axes),
        "phase_error": _phase_value(phase_error),
        "phase_diff": _phase_value(phase_diff),
        "corr_initial": None if corr_initial is None else float(corr_initial),
        "corr_phase": None if corr_phase is None else float(corr_phase),
        "gradient_component_ncc_initial": grad_initial,
        "gradient_component_ncc_phase": grad_phase,
        "gradient_component_ncc_initial_mean": _gradient_component_ncc_mean(grad_initial),
        "gradient_component_ncc_phase_mean": _gradient_component_ncc_mean(grad_phase),
        "method8_zero_z_shear": True,
        "native_return_code": None,
        "corr_method8": None,
        "gradient_component_ncc_method8": None,
        "gradient_component_ncc_method8_mean": None,
        "native_fit_output_to_input_matrix_zyx": None,
        "native_fit_output_to_input_offset_zyx": None,
        "native_output_to_input_matrix_zyx": None,
        "native_output_to_input_offset_zyx": None,
        "local_matrix_zyx": None,
        "local_translation_zyx": None,
        "status": "rejected",
        "rejection_reason": "phase_recovery_no_method8",
        "timing_seconds": {
            "phase": time.perf_counter() - phase_started,
            "method8": 0.0,
            "total": time.perf_counter() - started,
        },
    }


def _phase_value(value: Any) -> float:
    import cupy as cp

    if hasattr(value, "shape"):
        return float(cp.asnumpy(value))
    return float(value)


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
    return cp.ascontiguousarray(
        block_reduce(values, block_size=factors, func=cp.max).astype(cp.bool_, copy=False)
    )


def _mask_stats(mask_gpu: Any) -> dict[str, float | int]:
    import cupy as cp

    voxel_count = int(cp.asnumpy(cp.count_nonzero(mask_gpu)))
    total_voxels = int(mask_gpu.size)
    unmasked_fraction = float(voxel_count / total_voxels) if total_voxels else 0.0
    return {
        "voxel_count": voxel_count,
        "total_voxels": total_voxels,
        "unmasked_fraction": unmasked_fraction,
        "masked_fraction": float(1.0 - unmasked_fraction) if total_voxels else 0.0,
    }


def _all_adjacent_pairs(tiles: dict[str, TileInfo]) -> list[str]:
    records = sorted(tiles.values(), key=lambda tile: int(tile.tile_id))
    pairs: list[tuple[int, str]] = []
    for index, fixed in enumerate(records):
        spacing_abs = np.abs(fixed.spacing_um_zyx)
        fixed_stop_um = fixed.start_um_zyx + fixed.shape_zyx * fixed.spacing_um_zyx
        fixed_min_um = np.minimum(fixed.start_um_zyx, fixed_stop_um)
        fixed_max_um = np.maximum(fixed.start_um_zyx, fixed_stop_um)
        for moving in records[index + 1 :]:
            if not np.allclose(fixed.spacing_um_zyx, moving.spacing_um_zyx, rtol=0.0, atol=1e-9):
                continue
            moving_stop_um = moving.start_um_zyx + moving.shape_zyx * moving.spacing_um_zyx
            moving_min_um = np.minimum(moving.start_um_zyx, moving_stop_um)
            moving_max_um = np.maximum(moving.start_um_zyx, moving_stop_um)
            overlap_start_um = np.maximum(fixed_min_um, moving_min_um)
            overlap_stop_um = np.minimum(fixed_max_um, moving_max_um)
            overlap_px = np.floor((overlap_stop_um - overlap_start_um) / spacing_abs + 1e-6).astype(np.int64)
            if np.any(overlap_px <= 0):
                continue
            lateral = overlap_px[1:3]
            broad_overlap = int(np.max(lateral))
            thin_overlap = int(np.min(lateral))
            if broad_overlap < int(min(fixed.shape_zyx[1], fixed.shape_zyx[2]) * 0.5):
                continue
            if thin_overlap > int(max(fixed.shape_zyx[1], fixed.shape_zyx[2]) * 0.25):
                continue
            sort_key = int(fixed.tile_id) * 1000 + int(moving.tile_id)
            pairs.append((sort_key, f"{fixed.tile_id}-{moving.tile_id}"))
    return [pair for _sort_key, pair in sorted(pairs)]


def _measure_z_chunk(
    *,
    fixed: TileInfo,
    moving: TileInfo,
    z_start: int,
    z_stop: int,
    device: int,
    method8: bool,
    max_iterations: int,
    ftol: float,
    min_corr: float,
    min_grad_ncc: float,
    fixed_mask_threshold: float | None,
    fixed_mask_min_voxels: int,
    fixed_mask_max_masked_fraction: float,
    min_phase_grad: float,
    min_phase_corr: float,
    native_lib_dir: Path,
) -> dict[str, Any]:
    import cupy as cp
    from cucim.skimage.registration import phase_cross_correlation
    from cupyx.scipy.ndimage import shift as gpu_shift

    started = time.perf_counter()
    seam_axis, fixed_slices, moving_slices, crop_shape = _crop_bounds_for_pair(
        fixed,
        moving,
        z_start=int(z_start),
        z_depth=int(z_stop - z_start),
    )
    fit_downsample = _fit_downsample_for_shape(crop_shape, seam_axis)
    fixed_raw = _read_tile_crop(fixed, fixed_slices)
    moving_raw = _read_tile_crop(moving, moving_slices)
    mask_rejection_reason, fixed_center_z_content, moving_center_z_content = (
        center_z_content_prefilter_reason(fixed_raw, moving_raw, RobustBoundarySettings())
    )

    cp.cuda.Device(int(device)).use()
    fixed_gpu, fixed_content = _robust_norm_and_content_stats_cupy(fixed_raw)
    moving_gpu, moving_content = _robust_norm_and_content_stats_cupy(moving_raw)
    fixed_fit = _block_mean_downsample_zyx_cupy(fixed_gpu, fit_downsample)
    moving_fit = _block_mean_downsample_zyx_cupy(moving_gpu, fit_downsample)
    fixed_fit_mask = None
    fixed_threshold_mask = None
    if fixed_mask_threshold is not None:
        raw_mask = cp.asarray(fixed_raw > float(fixed_mask_threshold), dtype=cp.bool_)
        fixed_fit_mask = _block_any_downsample_zyx_cupy(raw_mask, fit_downsample)
        source_stats = _mask_stats(raw_mask)
        fit_stats = _mask_stats(fixed_fit_mask)
        fixed_threshold_mask = {
            "threshold": float(fixed_mask_threshold),
            "source": source_stats,
            "fit": fit_stats,
        }
        if mask_rejection_reason is None and int(fit_stats["voxel_count"]) < int(
            fixed_mask_min_voxels
        ):
            mask_rejection_reason = "fixed_threshold_fit_mask_too_few_voxels"
        elif mask_rejection_reason is None and float(fit_stats["masked_fraction"]) > float(
            fixed_mask_max_masked_fraction
        ):
            mask_rejection_reason = "fixed_threshold_fit_mask_too_sparse"

    if mask_rejection_reason is not None:
        return {
            "fixed_tile": fixed.tile_name,
            "moving_tile": moving.tile_name,
            "z_start": int(z_start),
            "z_stop": int(z_stop),
            "seam_axis": seam_axis,
            "fixed_slices_zyx": _slice_json(fixed_slices),
            "moving_slices_zyx": _slice_json(moving_slices),
            "window_shape_zyx": [int(value) for value in crop_shape],
            "fit_downsample_zyx": [int(value) for value in fit_downsample],
            "fit_shape_zyx": [int(value) for value in fixed_fit.shape],
            "fixed_content": fixed_content,
            "moving_content": moving_content,
            "fixed_center_z_content": fixed_center_z_content,
            "moving_center_z_content": moving_center_z_content,
            "fixed_threshold_mask": fixed_threshold_mask,
            "measurement_mode": "level0_phase_correlation",
            "phase_shift_fit_zyx": None,
            "phase_shift_zyx": None,
            "phase_shift_wrap_risk_axes": [],
            "phase_shift_wrap_risk": False,
            "phase_error": None,
            "phase_diff": None,
            "corr_initial": None,
            "corr_phase": None,
            "gradient_component_ncc_initial": None,
            "gradient_component_ncc_phase": None,
            "gradient_component_ncc_initial_mean": None,
            "gradient_component_ncc_phase_mean": None,
            "method8_zero_z_shear": True,
            "quality_mask": "fixed_threshold_mask",
            "native_return_code": None,
            "corr_method8": None,
            "gradient_component_ncc_method8": None,
            "gradient_component_ncc_method8_mean": None,
            "native_fit_output_to_input_matrix_zyx": None,
            "native_fit_output_to_input_offset_zyx": None,
            "native_output_to_input_matrix_zyx": None,
            "native_output_to_input_offset_zyx": None,
            "local_matrix_zyx": None,
            "local_translation_zyx": None,
            "status": "rejected",
            "rejection_reason": mask_rejection_reason,
            "timing_seconds": {
                "phase": 0.0,
                "method8": 0.0,
                "total": time.perf_counter() - started,
            },
        }

    corr_initial = _corr_gpu(fixed_fit, moving_fit)
    grad_initial = gradient_component_ncc_3d_gpu(fixed_fit, moving_fit)

    phase_started = time.perf_counter()
    phase_shift_fit, phase_error, phase_diff = phase_cross_correlation(
        fixed_fit,
        moving_fit,
        upsample_factor=10,
    )
    phase_shift_fit_np = cp.asnumpy(phase_shift_fit).astype(np.float32)
    phase_shift_zyx = phase_shift_fit_np.astype(np.float64) * np.asarray(fit_downsample)
    wrap_risk_axes = [
        DIMENSIONS[index]
        for index, (shift_value, extent) in enumerate(zip(phase_shift_zyx, crop_shape, strict=True))
        if abs(float(shift_value)) > 0.25 * float(extent)
    ]
    phase_registered = gpu_shift(
        moving_fit,
        shift=tuple(float(value) for value in phase_shift_fit_np),
        order=1,
        mode="constant",
        cval=0.0,
    )
    corr_phase = _corr_gpu(fixed_fit, phase_registered)
    grad_phase = gradient_component_ncc_3d_gpu(fixed_fit, phase_registered, fixed_mask=fixed_fit_mask)

    phase_row = {
        "fixed_tile": fixed.tile_name,
        "moving_tile": moving.tile_name,
        "z_start": int(z_start),
        "z_stop": int(z_stop),
        "seam_axis": seam_axis,
        "fixed_slices_zyx": _slice_json(fixed_slices),
        "moving_slices_zyx": _slice_json(moving_slices),
        "window_shape_zyx": [int(value) for value in crop_shape],
        "fit_downsample_zyx": [int(value) for value in fit_downsample],
        "fit_shape_zyx": [int(value) for value in fixed_fit.shape],
        "fixed_content": fixed_content,
        "moving_content": moving_content,
        "fixed_center_z_content": fixed_center_z_content,
        "moving_center_z_content": moving_center_z_content,
        "fixed_threshold_mask": fixed_threshold_mask,
        "measurement_mode": "level0_phase_correlation",
        "phase_shift_fit_zyx": phase_shift_fit_np.astype(np.float64).tolist(),
        "phase_shift_zyx": phase_shift_zyx.tolist(),
        "phase_shift_wrap_risk_axes": wrap_risk_axes,
        "phase_shift_wrap_risk": bool(wrap_risk_axes),
        "phase_error": _phase_value(phase_error),
        "phase_diff": _phase_value(phase_diff),
        "corr_initial": None if corr_initial is None else float(corr_initial),
        "corr_phase": None if corr_phase is None else float(corr_phase),
        "gradient_component_ncc_initial": grad_initial,
        "gradient_component_ncc_phase": grad_phase,
        "gradient_component_ncc_initial_mean": _gradient_component_ncc_mean(grad_initial),
        "gradient_component_ncc_phase_mean": _gradient_component_ncc_mean(grad_phase),
        "method8_zero_z_shear": True,
        "quality_mask": "fixed_threshold_mask" if fixed_fit_mask is not None else None,
    }
    if not method8:
        rejection_reason = mask_rejection_reason or _phase_gate(
            phase_row,
            min_phase_grad=float(min_phase_grad),
            min_phase_corr=float(min_phase_corr),
        )
        return {
            **phase_row,
            "native_return_code": None,
            "corr_method8": None,
            "gradient_component_ncc_method8": None,
            "gradient_component_ncc_method8_mean": None,
            "native_fit_output_to_input_matrix_zyx": None,
            "native_fit_output_to_input_offset_zyx": None,
            "native_output_to_input_matrix_zyx": None,
            "native_output_to_input_offset_zyx": None,
            "local_matrix_zyx": None,
            "local_translation_zyx": None,
            "status": "accepted" if rejection_reason is None else "rejected",
            "rejection_reason": rejection_reason,
            "timing_seconds": {
                "phase": time.perf_counter() - phase_started,
                "method8": 0.0,
                "total": time.perf_counter() - started,
            },
        }
    method8_started = time.perf_counter()
    native = register_method8_device(
        fixed_fit,
        moving_fit,
        fixed_mask_zyx=fixed_fit_mask,
        lib_dir=native_lib_dir,
        ftol=float(ftol),
        max_iterations=int(max_iterations),
        device=int(device),
        initial_matrix_xyz_3x4=zyx_to_xyz_3x4(np.eye(3, dtype=np.float32), phase_shift_fit_np),
        method8_zero_z_shear=True,
    )
    native_full_matrix, native_full_offset = _native_pull_from_fit_downsample(
        native.matrix_zyx,
        native.offset_zyx,
        fit_downsample,
    )
    local_matrix, local_translation = output_to_input_to_model(
        native_full_matrix,
        native_full_offset,
        tuple(int(value) for value in crop_shape),
    )
    corr_method8 = _corr_gpu(fixed_fit, native.registered_zyx)
    grad_method8 = gradient_component_ncc_3d_gpu(fixed_fit, native.registered_zyx, fixed_mask=fixed_fit_mask)
    grad_method8_mean = _gradient_component_ncc_mean(grad_method8)
    rejection_reason = None
    if int(native.return_code) != 0:
        rejection_reason = "native_return_code"
    elif (
        corr_method8 is None or not np.isfinite(float(corr_method8)) or float(corr_method8) < float(min_corr)
    ):
        rejection_reason = "low_corr_method8"
    elif not np.isfinite(grad_method8_mean) or grad_method8_mean < float(min_grad_ncc):
        rejection_reason = "low_gradient_component_ncc_method8"

    return {
        **phase_row,
        "native_return_code": int(native.return_code),
        "corr_method8": None if corr_method8 is None else float(corr_method8),
        "gradient_component_ncc_method8": grad_method8,
        "gradient_component_ncc_method8_mean": grad_method8_mean,
        "native_fit_output_to_input_matrix_zyx": np.asarray(native.matrix_zyx, dtype=np.float64).tolist(),
        "native_fit_output_to_input_offset_zyx": np.asarray(native.offset_zyx, dtype=np.float64).tolist(),
        "native_output_to_input_matrix_zyx": np.asarray(native_full_matrix, dtype=np.float64).tolist(),
        "native_output_to_input_offset_zyx": np.asarray(native_full_offset, dtype=np.float64).tolist(),
        "local_matrix_zyx": np.asarray(local_matrix, dtype=np.float64).tolist(),
        "local_translation_zyx": np.asarray(local_translation, dtype=np.float64).tolist(),
        "status": "accepted" if rejection_reason is None else "rejected",
        "rejection_reason": rejection_reason,
        "timing_seconds": {
            "phase": time.perf_counter() - phase_started,
            "method8": time.perf_counter() - method8_started,
            "total": time.perf_counter() - started,
        },
    }


def _pair_summary(rows: list[dict[str, Any]], *, method8: bool) -> dict[str, Any]:
    accepted = [row for row in rows if row["status"] == "accepted"]
    phase = np.asarray([row["phase_shift_zyx"] for row in accepted], dtype=np.float64)
    if phase.size == 0:
        return {"accepted_count": 0, "rejected_count": len(rows)}
    summary = {
        "accepted_count": len(accepted),
        "rejected_count": len(rows) - len(accepted),
        "phase_shift_median_zyx": np.median(phase, axis=0).tolist(),
        "phase_shift_mean_zyx": np.mean(phase, axis=0).tolist(),
        "phase_shift_std_zyx": np.std(phase, axis=0).tolist(),
    }
    if method8:
        translations = np.asarray([row["local_translation_zyx"] for row in accepted], dtype=np.float64)
        summary.update(
            {
                "translation_median_zyx": np.median(translations, axis=0).tolist(),
                "translation_mean_zyx": np.mean(translations, axis=0).tolist(),
                "translation_std_zyx": np.std(translations, axis=0).tolist(),
                "translation_min_zyx": np.min(translations, axis=0).tolist(),
                "translation_max_zyx": np.max(translations, axis=0).tolist(),
                "translation_range_zyx": np.ptp(translations, axis=0).tolist(),
            }
        )
    return summary


def _level2_screen_rejection_row(
    *,
    fixed: TileInfo,
    moving: TileInfo,
    z_start: int,
    z_stop: int,
    screen_sample: dict[str, Any],
) -> dict[str, Any]:
    """Represent a coarse rejection without reading either level-0 crop."""
    seam_axis, fixed_slices, moving_slices, crop_shape = _crop_bounds_for_pair(
        fixed,
        moving,
        z_start=z_start,
        z_depth=z_stop - z_start,
    )
    fit_downsample = _fit_downsample_for_shape(crop_shape, seam_axis)
    fit_shape = [
        int(size // factor) for size, factor in zip(crop_shape, fit_downsample, strict=True)
    ]
    return {
        "fixed_tile": fixed.tile_name,
        "moving_tile": moving.tile_name,
        "z_start": z_start,
        "z_stop": z_stop,
        "seam_axis": seam_axis,
        "fixed_slices_zyx": _slice_json(fixed_slices),
        "moving_slices_zyx": _slice_json(moving_slices),
        "window_shape_zyx": [int(value) for value in crop_shape],
        "fit_downsample_zyx": [int(value) for value in fit_downsample],
        "fit_shape_zyx": fit_shape,
        "fixed_content": None,
        "moving_content": None,
        "fixed_center_z_content": None,
        "moving_center_z_content": None,
        "fixed_threshold_mask": None,
        "level2_screen_sample": screen_sample,
        "measurement_mode": "level2_overlap_screen",
        "phase_shift_fit_zyx": None,
        "phase_shift_zyx": None,
        "phase_shift_wrap_risk_axes": [],
        "phase_shift_wrap_risk": False,
        "phase_error": None,
        "phase_diff": None,
        "corr_initial": None,
        "corr_phase": None,
        "gradient_component_ncc_initial": None,
        "gradient_component_ncc_phase": None,
        "gradient_component_ncc_initial_mean": None,
        "gradient_component_ncc_phase_mean": None,
        "method8_zero_z_shear": True,
        "quality_mask": "level2_overlap_screen",
        "native_return_code": None,
        "corr_method8": None,
        "gradient_component_ncc_method8": None,
        "gradient_component_ncc_method8_mean": None,
        "native_fit_output_to_input_matrix_zyx": None,
        "native_fit_output_to_input_offset_zyx": None,
        "native_output_to_input_matrix_zyx": None,
        "native_output_to_input_offset_zyx": None,
        "local_matrix_zyx": None,
        "local_translation_zyx": None,
        "status": "rejected",
        "rejection_reason": "level2_low_content",
        "timing_seconds": {"phase": 0.0, "method8": 0.0, "total": 0.0},
    }


def measure_method8_zcoverage(
    *,
    position_json: Path,
    zarr_dir: Path,
    output: Path,
    level2_screen: Path | None = None,
    pairs: tuple[str, ...] | None = None,
    all_adjacent: bool = True,
    z_chunks: int = 6,
    device: int = 0,
    method8: bool = False,
    channel: int = 0,
    max_iterations: int = 300,
    ftol: float = 1e-4,
    min_corr: float = 0.15,
    min_grad_ncc: float = 0.24,
    fixed_mask_threshold: float | None = DEFAULT_FIXED_MASK_THRESHOLD,
    fixed_mask_min_voxels: int = 256,
    fixed_mask_max_masked_fraction: float = 0.95,
    phase_recovery_shifted_crop: bool = True,
    phase_recovery_min_prior_edges_per_axis: int = 3,
    phase_recovery_min_phase_grad: float = 0.24,
    phase_recovery_min_phase_corr: float = 0.15,
    native_lib_dir: Path = DEFAULT_LIB_DIR,
    progress: Callable[[str], None] | None = None,
) -> Path:
    progress = progress or (lambda _message: None)
    tiles = _load_tiles(position_json, zarr_dir, channel=channel)
    resolved_pairs = _all_adjacent_pairs(tiles) if all_adjacent else list(pairs or ())
    if not resolved_pairs:
        raise ValueError("No adjacent tile pairs were found for registration")

    screen_decisions: dict[tuple[str, int], dict[str, Any]] | None = None
    if level2_screen is not None:
        from squisher_lightsheet.overlap_screen import load_level2_screen

        screen_decisions = load_level2_screen(
            level2_screen,
            position_json=position_json,
            zarr_dir=zarr_dir,
            expected_pairs=resolved_pairs,
            z_chunks=z_chunks,
            channel=channel,
            threshold=fixed_mask_threshold,
        )

    rows: list[dict[str, Any]] = []
    pair_rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    for pair in resolved_pairs:
        fixed_id, moving_id = pair.split("-", 1)
        fixed = tiles[fixed_id]
        moving = tiles[moving_id]
        pair_rows = []
        chunks = _z_chunks(int(fixed.shape_zyx[0]), int(z_chunks))
        for chunk_index, (z_start, z_stop) in enumerate(chunks):
            mode = "method8" if method8 else "phase"
            progress(f"{mode} pair={pair} chunk={chunk_index + 1}/{len(chunks)} z={z_start}:{z_stop}")
            screen_sample = (
                None if screen_decisions is None else screen_decisions[(pair, chunk_index)]
            )
            if screen_sample is not None and (
                screen_sample.get("z_start") != z_start or screen_sample.get("z_stop") != z_stop
            ):
                raise ValueError(
                    f"{level2_screen} unit {pair} chunk {chunk_index} z range differs from registration"
                )
            if screen_sample is not None and screen_sample["status"] == "low_content":
                row = _level2_screen_rejection_row(
                    fixed=fixed,
                    moving=moving,
                    z_start=z_start,
                    z_stop=z_stop,
                    screen_sample=screen_sample,
                )
            else:
                row = _measure_z_chunk(
                    fixed=fixed,
                    moving=moving,
                    z_start=z_start,
                    z_stop=z_stop,
                    device=int(device),
                    method8=bool(method8),
                    max_iterations=int(max_iterations),
                    ftol=float(ftol),
                    min_corr=float(min_corr),
                    min_grad_ncc=float(min_grad_ncc),
                    fixed_mask_threshold=fixed_mask_threshold,
                    fixed_mask_min_voxels=int(fixed_mask_min_voxels),
                    fixed_mask_max_masked_fraction=float(fixed_mask_max_masked_fraction),
                    min_phase_grad=float(phase_recovery_min_phase_grad),
                    min_phase_corr=float(phase_recovery_min_phase_corr),
                    native_lib_dir=native_lib_dir,
                )
            pair_rows.append(row)
            rows.append(row)
            progress(
                " ".join(
                    [
                        f"status={row['status']}",
                        f"reason={row['rejection_reason']}",
                        f"phase={row['phase_shift_zyx']}",
                        f"method8={row['local_translation_zyx']}" if method8 else "method8=disabled",
                        (
                            f"grad={row['gradient_component_ncc_phase_mean']}"
                            if not method8
                            else (
                                f"grad={row['gradient_component_ncc_phase_mean']}"
                                f"->{row['gradient_component_ncc_method8_mean']}"
                            )
                        ),
                    ]
                )
            )
        pair_rows_by_pair[pair] = pair_rows

    phase_recovery_diagnostics: dict[str, Any] = {"enabled": bool(phase_recovery_shifted_crop)}
    if phase_recovery_shifted_crop:
        recovery_decision_counts: Counter[str] = Counter()
        axis_priors, covered_edges, axis_prior_candidates = _axis_priors_from_phase_rows(
            rows,
            min_phase_grad=float(phase_recovery_min_phase_grad),
            min_phase_corr=float(phase_recovery_min_phase_corr),
            min_edges_per_axis=int(phase_recovery_min_prior_edges_per_axis),
        )
        phase_recovery_diagnostics.update(
            {
                "axis_prior_candidates": axis_prior_candidates,
                "axis_priors_zyx": {
                    axis: prior.astype(np.float64).tolist() for axis, prior in axis_priors.items()
                },
                "covered_edge_count": len(covered_edges),
                "recovery_attempted_row_count": 0,
                "skipped_no_axis_prior": [],
                "skipped_terminal_rows": [],
            }
        )
        for pair in resolved_pairs:
            fixed_id, moving_id = pair.split("-", 1)
            fixed = tiles[fixed_id]
            moving = tiles[moving_id]
            seam_axis, _fixed_slices, _moving_slices, _crop_shape = _crop_bounds_for_pair(
                fixed,
                moving,
                z_start=0,
                z_depth=max(64, int(fixed.shape_zyx[0]) // max(int(z_chunks), 1)),
            )
            axis_prior = axis_priors.get(seam_axis)
            original_rows = [
                row
                for row in pair_rows_by_pair.get(pair, [])
                if row.get("measurement_mode") != "phase_recovery_prior_shifted_crop"
            ]
            for chunk_index, original_row in enumerate(original_rows):
                z_start = int(original_row["z_start"])
                z_stop = int(original_row["z_stop"])
                disposition, decision_reason = _classify_phase_recovery(
                    original_row,
                    min_phase_grad=float(phase_recovery_min_phase_grad),
                    min_phase_corr=float(phase_recovery_min_phase_corr),
                )
                recovery_decision_counts[f"{disposition}:{decision_reason}"] += 1
                if disposition != "recoverable":
                    progress(
                        " ".join(
                            [
                                f"phase-recovery skip pair={pair}",
                                f"axis={seam_axis}",
                                f"chunk={chunk_index + 1}/{len(original_rows)}",
                                f"z={z_start}:{z_stop}",
                                f"reason={decision_reason}",
                            ]
                        )
                    )
                    if disposition == "terminal":
                        phase_recovery_diagnostics["skipped_terminal_rows"].append(
                            {
                                "pair": pair,
                                "z_start": z_start,
                                "z_stop": z_stop,
                                "reason": decision_reason,
                            }
                        )
                    continue
                if axis_prior is None:
                    progress(
                        " ".join(
                            [
                                f"phase-recovery skip pair={pair}",
                                f"axis={seam_axis}",
                                f"chunk={chunk_index + 1}/{len(original_rows)}",
                                f"z={z_start}:{z_stop}",
                                "reason=no_axis_prior",
                            ]
                        )
                    )
                    phase_recovery_diagnostics["skipped_no_axis_prior"].append(
                        {
                            "pair": pair,
                            "z_start": z_start,
                            "z_stop": z_stop,
                            "phase_failure_reason": decision_reason,
                        }
                    )
                    continue
                progress(
                    " ".join(
                        [
                            f"phase-recovery pair={pair}",
                            f"axis={seam_axis}",
                            f"chunk={chunk_index + 1}/{len(original_rows)}",
                            f"z={z_start}:{z_stop}",
                            f"prior={axis_prior.astype(np.float64).tolist()}",
                        ]
                    )
                )
                row = _measure_prior_shifted_phase_z_chunk(
                    fixed=fixed,
                    moving=moving,
                    z_start=z_start,
                    z_stop=z_stop,
                    device=int(device),
                    axis_prior_zyx=axis_prior,
                    fixed_mask_threshold=fixed_mask_threshold,
                    fixed_mask_min_voxels=int(fixed_mask_min_voxels),
                    fixed_mask_max_masked_fraction=float(fixed_mask_max_masked_fraction),
                )
                if not method8:
                    rejection_reason = _phase_gate(
                        row,
                        min_phase_grad=float(phase_recovery_min_phase_grad),
                        min_phase_corr=float(phase_recovery_min_phase_corr),
                    )
                    row["status"] = "accepted" if rejection_reason is None else "rejected"
                    row["rejection_reason"] = rejection_reason
                row["original_crop_phase_shift_zyx"] = original_row.get("phase_shift_zyx")
                row["original_crop_corr_phase"] = original_row.get("corr_phase")
                row["original_crop_gradient_component_ncc_phase_mean"] = original_row.get(
                    "gradient_component_ncc_phase_mean"
                )
                row["recovery_corr_phase"] = row.get("corr_phase")
                row["recovery_gradient_component_ncc_phase_mean"] = row.get(
                    "gradient_component_ncc_phase_mean"
                )
                rows.append(row)
                pair_rows_by_pair.setdefault(pair, []).append(row)
                phase_recovery_diagnostics["recovery_attempted_row_count"] += 1
                progress(
                    " ".join(
                        [
                            f"mode={row['measurement_mode']}",
                            f"phase={row['phase_shift_zyx']}",
                            f"crop_offset={row['moving_crop_offset_zyx']}",
                            f"corr_phase_original={row.get('original_crop_corr_phase')}",
                            f"corr_phase_recovery={row.get('recovery_corr_phase')}",
                            f"grad={row['gradient_component_ncc_initial_mean']}->{row['gradient_component_ncc_phase_mean']}",
                        ]
                    )
                )
        phase_recovery_diagnostics["recovery_decision_counts"] = dict(
            sorted(recovery_decision_counts.items())
        )
        recovered_rows = [
            row for row in rows if row.get("measurement_mode") == "phase_recovery_prior_shifted_crop"
        ]
        phase_recovery_diagnostics["recovery_accepted_row_count"] = sum(
            1
            for row in recovered_rows
            if (
                _phase_gate(
                    row,
                    min_phase_grad=phase_recovery_min_phase_grad,
                    min_phase_corr=phase_recovery_min_phase_corr,
                )
                is None
            )
        )
        phase_recovery_diagnostics["wrap_risk_recovered_row_count"] = sum(
            1 for row in recovered_rows if row.get("phase_shift_wrap_risk")
        )

    pair_summaries = {
        pair: _pair_summary(pair_rows, method8=method8)
        for pair, pair_rows in pair_rows_by_pair.items()
    }

    payload = {
        "schema_version": 1,
        "artifact_type": "lightsheet.level0_phase_recovery_measurements.v1",
        "position_json": str(position_json.resolve()),
        "zarr_dir": str(zarr_dir.resolve()),
        "input_fingerprint": registration_input_fingerprint(position_json, zarr_dir),
        "settings": {
            "pairs": list(pairs or ()),
            "resolved_pairs": resolved_pairs,
            "all_adjacent": bool(all_adjacent),
            "z_chunks": int(z_chunks),
            "device": int(device),
            "method8": bool(method8),
            "channel": int(channel),
            "max_iterations": int(max_iterations),
            "ftol": float(ftol),
            "min_corr": float(min_corr),
            "min_grad_ncc": float(min_grad_ncc),
            "phase_primed": True,
            "face_span": "full_overlap",
            "fixed_mask_threshold": fixed_mask_threshold,
            "fixed_mask_min_voxels": int(fixed_mask_min_voxels),
            "fixed_mask_max_masked_fraction": float(fixed_mask_max_masked_fraction),
            "phase_recovery_shifted_crop": bool(phase_recovery_shifted_crop),
            "phase_recovery_min_prior_edges_per_axis": int(phase_recovery_min_prior_edges_per_axis),
            "phase_recovery_min_phase_grad": float(phase_recovery_min_phase_grad),
            "phase_recovery_min_phase_corr": float(phase_recovery_min_phase_corr),
            "native_lib_dir": str(native_lib_dir.resolve()),
            "level2_screen": str(level2_screen.resolve()) if level2_screen is not None else None,
            "level2_screen_sha256": sha256_file(level2_screen) if level2_screen is not None else None,
        },
        "phase_recovery": phase_recovery_diagnostics,
        "pair_summaries": pair_summaries,
        "rows": rows,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(output, json.dumps(payload, indent=2) + "\n")
    progress(f"wrote {output}")
    return output


def _quality_gate(
    row: dict[str, Any], *, max_grad_regression: float, max_corr_regression: float
) -> str | None:
    rejection_reason = row.get("rejection_reason")
    if rejection_reason in PHASE_INPUT_REJECTION_REASONS:
        return str(rejection_reason)
    if row.get("phase_shift_wrap_risk"):
        return "phase_shift_wrap_risk"
    if row["status"] != "accepted":
        return str(row["rejection_reason"])
    initial_grad = row["gradient_component_ncc_initial_mean"]
    initial_corr = row["corr_initial"]
    if initial_grad is None or not np.isfinite(float(initial_grad)):
        return "initial_gradient_component_ncc_not_finite"
    if initial_corr is None or not np.isfinite(float(initial_corr)):
        return "initial_corr_not_finite"
    method8_grad = row["gradient_component_ncc_method8_mean"]
    phase_grad = row["gradient_component_ncc_phase_mean"]
    method8_corr = row["corr_method8"]
    phase_corr = row["corr_phase"]
    if method8_grad is None or not np.isfinite(float(method8_grad)):
        return "method8_gradient_component_ncc_not_finite"
    if phase_grad is not None and np.isfinite(float(phase_grad)):
        if float(method8_grad) < float(phase_grad) - float(max_grad_regression):
            return "method8_gradient_component_ncc_regressed_from_phase"
    if method8_corr is None or not np.isfinite(float(method8_corr)):
        return "method8_corr_not_finite"
    if phase_corr is not None and np.isfinite(float(phase_corr)):
        if float(method8_corr) < float(phase_corr) - float(max_corr_regression):
            return "method8_corr_regressed_from_phase"
    return None


def _phase_gate(row: dict[str, Any], *, min_phase_grad: float, min_phase_corr: float) -> str | None:
    rejection_reason = row.get("rejection_reason")
    if rejection_reason in PHASE_INPUT_REJECTION_REASONS:
        return str(rejection_reason)
    initial_grad = row["gradient_component_ncc_initial_mean"]
    initial_corr = row["corr_initial"]
    if initial_grad is None or not np.isfinite(float(initial_grad)):
        return "initial_gradient_component_ncc_not_finite"
    if initial_corr is None or not np.isfinite(float(initial_corr)):
        return "initial_corr_not_finite"
    phase_grad = row["gradient_component_ncc_phase_mean"]
    phase_corr = row["corr_phase"]
    phase_shift = row["phase_shift_zyx"]
    if phase_shift is None or len(phase_shift) != 3:
        return "phase_shift_missing"
    if not np.all(np.isfinite(np.asarray(phase_shift, dtype=np.float64))):
        return "phase_shift_not_finite"
    if row.get("phase_shift_wrap_risk"):
        return "phase_shift_wrap_risk"
    if phase_grad is None or not np.isfinite(float(phase_grad)):
        return "phase_gradient_component_ncc_not_finite"
    if phase_corr is None or not np.isfinite(float(phase_corr)):
        return "phase_corr_not_finite"
    if float(phase_grad) < float(min_phase_grad):
        return "phase_gradient_component_ncc_too_low"
    if float(phase_corr) < float(min_phase_corr):
        return "phase_corr_too_low"
    return None


def _classify_phase_recovery(
    row: dict[str, Any],
    *,
    min_phase_grad: float,
    min_phase_corr: float,
) -> tuple[PhaseRecoveryDisposition, str]:
    phase_reason = _phase_gate(
        row,
        min_phase_grad=min_phase_grad,
        min_phase_corr=min_phase_corr,
    )
    if phase_reason is None:
        return "source_accepted", "phase_accepted"
    if phase_reason in PHASE_RECOVERABLE_REJECTION_REASONS:
        return "recoverable", phase_reason
    return "terminal", phase_reason


def constraints_from_method8_summary(
    summary: dict[str, Any],
    *,
    tile_index: dict[str, int],
    max_grad_regression: float = 0.02,
    max_corr_regression: float = 0.01,
    use_phase_fallback: bool = True,
    min_phase_grad: float = 0.24,
    min_phase_corr: float = 0.15,
    phase_fallback_weight_scale: float = 0.1,
) -> tuple[list[BoundaryConstraint], Counter[str], dict[str, int], dict[str, Any]]:
    threshold_slug = _threshold_slug(summary["settings"]["fixed_mask_threshold"])
    method8_enabled = bool(summary["settings"].get("method8", False))
    constraints: list[BoundaryConstraint] = []
    rejected: Counter[str] = Counter()
    method8_grouped: defaultdict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    phase_grouped: defaultdict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    phase_attempt_by_chunk: dict[tuple[str, str, str, int, int], str] = {}
    all_edges: set[tuple[int, int, str]] = set()
    for row in summary["rows"]:
        fixed_id = _tile_id(str(row["fixed_tile"]))
        moving_id = _tile_id(str(row["moving_tile"]))
        if fixed_id not in tile_index or moving_id not in tile_index:
            raise ValueError(f"Summary row references tile outside position input: {fixed_id}-{moving_id}")
        fixed_index = tile_index[fixed_id]
        moving_index = tile_index[moving_id]
        edge_key = (fixed_index, moving_index, str(row["seam_axis"]))
        all_edges.add(edge_key)
        gate_reason = None
        if method8_enabled:
            gate_reason = _quality_gate(
                row,
                max_grad_regression=max_grad_regression,
                max_corr_regression=max_corr_regression,
            )
            if gate_reason is not None:
                rejected[gate_reason] += 1
            else:
                method8_grouped[edge_key].append(row)
        phase_gate_reason = _phase_gate(
            row,
            min_phase_grad=min_phase_grad,
            min_phase_corr=min_phase_corr,
        )
        if phase_gate_reason is not None:
            if phase_gate_reason != gate_reason:
                rejected[phase_gate_reason] += 1
        elif not method8_enabled or use_phase_fallback:
            z_start = row.get("z_start")
            z_stop = row.get("z_stop")
            if isinstance(z_start, int) and isinstance(z_stop, int):
                chunk_key = (
                    str(row["fixed_tile"]),
                    str(row["moving_tile"]),
                    str(row["seam_axis"]),
                    z_start,
                    z_stop,
                )
                measurement_mode = str(row.get("measurement_mode", "unknown"))
                previous_mode = phase_attempt_by_chunk.get(chunk_key)
                if previous_mode is not None:
                    raise ValueError(
                        "Multiple phase-valid attempts for one chunk: "
                        f"key={chunk_key}, modes={previous_mode!r},{measurement_mode!r}"
                    )
                phase_attempt_by_chunk[chunk_key] = measurement_mode
            phase_grouped[edge_key].append(row)

    source_counts: Counter[str] = Counter()
    fallback_missed = 0
    selected_edges = []
    for fixed_index, moving_index, axis in sorted(all_edges):
        source = "method8" if method8_enabled else "phase"
        edge_rows = (
            method8_grouped.get((fixed_index, moving_index, axis), [])
            if method8_enabled
            else phase_grouped.get((fixed_index, moving_index, axis), [])
        )
        if method8_enabled and not edge_rows and use_phase_fallback:
            edge_rows = phase_grouped.get((fixed_index, moving_index, axis), [])
            source = "phase_fallback"
        if not edge_rows:
            fallback_missed += 1
            continue
        source_counts[source] += 1
        shift_key = "local_translation_zyx" if source == "method8" else "phase_shift_zyx"
        shifts = np.asarray([row[shift_key] for row in edge_rows], dtype=np.float64)
        variance_zyx = np.var(shifts, axis=0, ddof=1) if len(edge_rows) >= 2 else None
        selected_edges.append((fixed_index, moving_index, axis, source, edge_rows, shifts, variance_zyx))

    observed_total_variances = [
        float(np.sum(variance_zyx))
        for *_edge, variance_zyx in selected_edges
        if variance_zyx is not None
    ]
    if selected_edges and not observed_total_variances:
        raise ValueError(
            "Cannot compute inverse-variance seam weights: no retained seam has at least two accepted chunks"
        )
    sparse_variance_prior = (
        float(np.percentile(observed_total_variances, SPARSE_VARIANCE_PRIOR_PERCENTILE))
        if observed_total_variances
        else None
    )
    effective_variances = [
        max(
            float(np.sum(variance_zyx)) if variance_zyx is not None else float(sparse_variance_prior),
            CHUNK_SHIFT_VARIANCE_FLOOR_PX2,
        )
        for *_edge, variance_zyx in selected_edges
    ]
    raw_precisions = [1.0 / variance for variance in effective_variances]
    maximum_raw_precision = max(raw_precisions, default=None)
    seam_weighting = []

    for edge, effective_variance, raw_precision in zip(
        selected_edges, effective_variances, raw_precisions, strict=True
    ):
        fixed_index, moving_index, axis, source, edge_rows, shifts, variance_zyx = edge
        after_grad_key = (
            "gradient_component_ncc_method8_mean"
            if source == "method8"
            else "gradient_component_ncc_phase_mean"
        )
        after_corr_key = "corr_method8" if source == "method8" else "corr_phase"
        grad_before_values = np.asarray(
            [float(row["gradient_component_ncc_initial_mean"]) for row in edge_rows]
        )
        grad_after_values = np.asarray([float(row[after_grad_key]) for row in edge_rows])
        corr_before_values = np.asarray([float(row["corr_initial"]) for row in edge_rows])
        corr_after_values = np.asarray([float(row[after_corr_key]) for row in edge_rows])
        fixed_nonzero_fractions = []
        fixed_content_fractions = []
        for row in edge_rows:
            fixed_threshold_mask = row["fixed_threshold_mask"]
            if fixed_threshold_mask is None:
                fixed_nonzero_fractions.append(0.0)
                fixed_content_fractions.append(0.0)
            else:
                fixed_nonzero_fractions.append(float(fixed_threshold_mask["fit"]["unmasked_fraction"]))
                fixed_content_fractions.append(float(fixed_threshold_mask["source"]["unmasked_fraction"]))
        fixed_nonzero_values = np.asarray(fixed_nonzero_fractions, dtype=np.float64)
        fixed_content_values = np.asarray(fixed_content_fractions, dtype=np.float64)
        fixed_std_values = np.asarray([float(row["fixed_content"]["std"]) for row in edge_rows])
        moving_std_values = np.asarray([float(row["moving_content"]["std"]) for row in edge_rows])

        shift = tuple(float(value) for value in np.median(shifts, axis=0))
        grad_before = float(np.median(grad_before_values))
        grad_after = float(np.median(grad_after_values))
        corr_before = float(np.median(corr_before_values))
        corr_after = float(np.median(corr_after_values))
        improvement = grad_after - grad_before
        base_quality = max(grad_after - 0.15, 1e-3)
        assert maximum_raw_precision is not None
        normalized_precision = raw_precision / maximum_raw_precision
        fallback_scale = float(phase_fallback_weight_scale) if source == "phase_fallback" else 1.0
        weight = base_quality * normalized_precision * fallback_scale
        pair = (fixed_index, moving_index)
        representative = edge_rows[len(edge_rows) // 2]
        measurement_mode = representative.get("measurement_mode")
        source_label = f"level0_{source}_{threshold_slug}_phase_gated_median_n{len(edge_rows)}"
        if measurement_mode is not None:
            source_label = (
                f"level0_{source}_{measurement_mode}_{threshold_slug}_phase_gated_median_n{len(edge_rows)}"
            )
        constraints.append(
            BoundaryConstraint(
                fixed=fixed_index,
                moving=moving_index,
                pair=pair,
                axis=axis,
                patch_index=0,
                shift_zyx=shift,
                weight=weight,
                correlation_before=corr_before,
                correlation_after=corr_after,
                improvement=improvement,
                fixed_nonzero_fraction=float(np.median(fixed_nonzero_values)),
                moving_nonzero_fraction=0.0,
                fixed_std=float(np.median(fixed_std_values)),
                moving_std=float(np.median(moving_std_values)),
                accepted=True,
                fixed_content_fraction=float(np.median(fixed_content_values)),
                moving_content_fraction=0.0,
                gradient_component_ncc_before=grad_before,
                gradient_component_ncc_after=grad_after,
                gradient_component_ncc_improvement=improvement,
                fixed_slices=_slice_tuple(representative["fixed_slices_zyx"]),
                moving_slices=_slice_tuple(representative["moving_slices_zyx"]),
                source_label=source_label,
            )
        )
        seam_weighting.append(
            {
                "fixed": fixed_index,
                "moving": moving_index,
                "axis": axis,
                "source": source,
                "accepted_chunk_count": len(edge_rows),
                "shift_variance_zyx_px2": (
                    None if variance_zyx is None else [float(value) for value in variance_zyx]
                ),
                "total_variance_px2": None if variance_zyx is None else float(np.sum(variance_zyx)),
                "variance_source": (
                    "observed" if variance_zyx is not None else "observed_90th_percentile_prior"
                ),
                "effective_total_variance_px2": effective_variance,
                "raw_precision_per_px2": raw_precision,
                "normalized_precision": normalized_precision,
                "base_gradient_quality": base_quality,
                "phase_fallback_weight_scale": fallback_scale,
                "final_weight": weight,
            }
        )
    source_counts["missing_edges_after_fallback"] = fallback_missed
    weighting = {
        "formula": "gradient_quality * normalized_inverse_chunk_shift_variance * phase_fallback_scale",
        "variance_axes": "zyx",
        "sample_variance_ddof": 1,
        "variance_floor_px2": CHUNK_SHIFT_VARIANCE_FLOOR_PX2,
        "sparse_variance_prior_percentile": SPARSE_VARIANCE_PRIOR_PERCENTILE,
        "sparse_variance_prior_px2": sparse_variance_prior,
        "precision_normalization": "maximum_raw_precision",
        "maximum_raw_precision_per_px2": maximum_raw_precision,
        "seams": seam_weighting,
    }
    return constraints, rejected, dict(source_counts), weighting


def _constraint_payload(constraint: BoundaryConstraint) -> dict[str, Any]:
    def slices_payload(slices: tuple[slice, slice, slice] | None) -> list[list[int | None]] | None:
        if slices is None:
            return None
        return [[slc.start, slc.stop, slc.step] for slc in slices]

    return {
        "fixed": constraint.fixed,
        "moving": constraint.moving,
        "pair": list(constraint.pair),
        "axis": constraint.axis,
        "patch_index": constraint.patch_index,
        "shift_zyx": list(constraint.shift_zyx),
        "weight": float(constraint.weight),
        "correlation_before": float(constraint.correlation_before),
        "correlation_after": float(constraint.correlation_after),
        "improvement": float(constraint.improvement),
        "gradient_component_ncc_before": constraint.gradient_component_ncc_before,
        "gradient_component_ncc_after": constraint.gradient_component_ncc_after,
        "gradient_component_ncc_improvement": constraint.gradient_component_ncc_improvement,
        "fixed_nonzero_fraction": constraint.fixed_nonzero_fraction,
        "fixed_content_fraction": constraint.fixed_content_fraction,
        "accepted": bool(constraint.accepted),
        "reject_reason": constraint.reject_reason,
        "final_residual_zyx": None
        if constraint.final_residual_zyx is None
        else list(constraint.final_residual_zyx),
        "fixed_slices_zyx": slices_payload(constraint.fixed_slices),
        "moving_slices_zyx": slices_payload(constraint.moving_slices),
        "source_label": constraint.source_label,
    }


def _residual_stats(constraints: list[BoundaryConstraint]) -> dict[str, Any]:
    residuals = np.asarray(
        [
            constraint.final_residual_zyx
            for constraint in constraints
            if constraint.accepted and constraint.final_residual_zyx is not None
        ],
        dtype=np.float64,
    )
    if residuals.size == 0:
        return {"count": 0}
    norms = np.linalg.norm(residuals, axis=1)
    return {
        "count": int(len(residuals)),
        "median_norm_px": float(np.median(norms)),
        "p95_norm_px": float(np.percentile(norms, 95)),
        "max_norm_px": float(np.max(norms)),
        "median_abs_zyx_px": np.median(np.abs(residuals), axis=0).tolist(),
        "p95_abs_zyx_px": np.percentile(np.abs(residuals), 95, axis=0).tolist(),
    }


def optimize_positions_from_method8_summary(
    *,
    method8_summary: Path,
    position_json: Path,
    zarr_dir: Path,
    output_dir: Path,
    channel: int = 0,
    max_grad_regression: float = 0.02,
    max_corr_regression: float = 0.01,
    phase_fallback: bool = True,
    min_phase_grad: float = 0.24,
    min_phase_corr: float = 0.15,
    phase_fallback_weight_scale: float = 0.1,
) -> Method8RegistrationOutputs:
    _require_optimized_outputs_absent(output_dir)
    summary = json.loads(method8_summary.read_text())
    if summary.get("artifact_type") != "lightsheet.level0_phase_recovery_measurements.v1":
        raise ValueError(f"{method8_summary} is not a level-0 phase recovery measurement summary")
    recorded_position = summary.get("position_json")
    recorded_zarr_dir = summary.get("zarr_dir")
    if not isinstance(recorded_position, str) or Path(recorded_position).resolve() != position_json.resolve():
        raise ValueError(f"{method8_summary} belongs to a different position file: {recorded_position}")
    if not isinstance(recorded_zarr_dir, str) or Path(recorded_zarr_dir).resolve() != zarr_dir.resolve():
        raise ValueError(f"{method8_summary} belongs to a different Zarr directory: {recorded_zarr_dir}")
    if summary.get("settings", {}).get("channel") != int(channel):
        raise ValueError(f"{method8_summary} belongs to a different channel")
    position_payload, tile_names, tile_index, spacing, tiles = _load_position_tiles(
        position_json, zarr_dir, channel=channel
    )
    constraints, gate_rejections, constraint_sources, constraint_weighting = constraints_from_method8_summary(
        summary,
        tile_index=tile_index,
        max_grad_regression=float(max_grad_regression),
        max_corr_regression=float(max_corr_regression),
        use_phase_fallback=bool(phase_fallback),
        min_phase_grad=float(min_phase_grad),
        min_phase_corr=float(min_phase_corr),
        phase_fallback_weight_scale=float(phase_fallback_weight_scale),
    )
    if not constraints:
        raise ValueError("No registration constraints survived the phase quality gate")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _optimized_output_paths(output_dir)
    diagnostics_path = output_paths["diagnostics"]
    positions_path = output_paths["positions"]
    residuals_path = output_paths["constraints"]
    corrections_path = output_paths["corrections"]

    settings = RobustBoundarySettings()
    corrections, annotated, anchor_tile = stitch_legacy.solve_tile_corrections_with_multiview_stitcher(
        tiles,
        constraints,
        settings,
    )
    connected = stitch_legacy.anchor_connected_tiles(len(tile_names), annotated, anchor_tile)
    corrections_array = np.asarray(corrections, dtype=np.float64)

    residual_lines = "".join(json.dumps(_constraint_payload(constraint)) + "\n" for constraint in annotated)

    correction_norms = np.linalg.norm(corrections_array, axis=1)
    diagnostics = {
        "schema_version": 1,
        "artifact_type": "lightsheet.level0_phase_recovery_tile_optimization.v1",
        "method8_summary": str(method8_summary.resolve()),
        "method8_summary_sha256": sha256_file(method8_summary),
        "position_json": str(position_json.resolve()),
        "settings": {
            "max_grad_regression": float(max_grad_regression),
            "max_corr_regression": float(max_corr_regression),
            "constraint_aggregation": "median_shift_per_fixed_moving_axis_edge",
            "phase_fallback": bool(phase_fallback),
            "min_phase_grad": float(min_phase_grad),
            "min_phase_corr": float(min_phase_corr),
            "phase_fallback_weight_scale": float(phase_fallback_weight_scale),
            "solver": "multiview_stitcher.global_optimization.translation",
            "zarr_dir": str(zarr_dir.resolve()),
            "robust_boundary_settings": asdict(settings),
        },
        "tile_count": len(tile_names),
        "anchor_tile_index": int(anchor_tile),
        "anchor_tile": tile_names[anchor_tile],
        "connected_tile_count": len(connected),
        "input_rows": len(summary["rows"]),
        "phase_recovery": summary.get("phase_recovery"),
        "constraint_count": len(constraints),
        "constraint_sources": constraint_sources,
        "constraint_weighting": constraint_weighting,
        "accepted_after_residual_rejection": sum(1 for constraint in annotated if constraint.accepted),
        "gate_rejections": dict(gate_rejections),
        "residual_rejections": dict(
            Counter(constraint.reject_reason for constraint in annotated if constraint.reject_reason)
        ),
        "residual_stats": _residual_stats(annotated),
        "correction_stats_px": {
            "max_norm": float(np.max(correction_norms)),
            "median_norm": float(np.median(correction_norms)),
            "p95_norm": float(np.percentile(correction_norms, 95)),
            "min_zyx": np.min(corrections_array, axis=0).tolist(),
            "max_zyx": np.max(corrections_array, axis=0).tolist(),
        },
        "outputs": {
            "positions": str(positions_path.resolve()),
            "constraints_jsonl": str(residuals_path.resolve()),
            "corrections": str(corrections_path.resolve()),
        },
    }
    corrections_text = json.dumps(
            {
                "tile_names": tile_names,
                "spacing_zyx_um": spacing.tolist(),
                "corrections_zyx_px": corrections_array.tolist(),
                "corrections_zyx_um": (corrections_array * spacing).tolist(),
            },
            indent=2,
        ) + "\n"
    optimized_positions = json.loads(json.dumps(position_payload))
    for index, record in enumerate(optimized_positions["tiles"]):
        translation = _record_vector_zyx(record, "translation_um") + corrections_array[index] * spacing
        _set_record_vector_zyx(record, "translation_um", translation)
    optimized_positions["source"] = (
        f"{optimized_positions.get('source', 'position file')} + level-0 seam optimization"
    )
    optimized_positions["derived_by"] = "lightsheet-stitch register"
    optimized_positions["optimization_diagnostics"] = str(diagnostics_path.resolve())
    write_text_set_atomic(
        {
            residuals_path: residual_lines,
            diagnostics_path: json.dumps(diagnostics, indent=2) + "\n",
            corrections_path: corrections_text,
            positions_path: json.dumps(optimized_positions, indent=2) + "\n",
        }
    )
    return Method8RegistrationOutputs(
        method8_summary=method8_summary,
        optimized_positions=positions_path,
        diagnostics=diagnostics_path,
        constraints_jsonl=residuals_path,
        tile_corrections=corrections_path,
    )


def register_level0_phase_recovery(
    *,
    position_json: Path,
    zarr_dir: Path,
    output_dir: Path,
    level2_screen: Path | None = None,
    method8_summary: Path | None = None,
    method8_output: Path | None = None,
    pairs: tuple[str, ...] | None = None,
    all_adjacent: bool = True,
    z_chunks: int = 6,
    device: int = 0,
    method8: bool = False,
    channel: int = 0,
    max_iterations: int = 300,
    ftol: float = 1e-4,
    min_corr: float = 0.15,
    min_grad_ncc: float = 0.24,
    fixed_mask_threshold: float | None = DEFAULT_FIXED_MASK_THRESHOLD,
    fixed_mask_min_voxels: int = 256,
    fixed_mask_max_masked_fraction: float = 0.95,
    phase_recovery_shifted_crop: bool = True,
    phase_recovery_min_prior_edges_per_axis: int = 3,
    max_grad_regression: float = 0.02,
    max_corr_regression: float = 0.01,
    phase_fallback: bool = True,
    min_phase_grad: float = 0.24,
    min_phase_corr: float = 0.15,
    phase_fallback_weight_scale: float = 0.1,
    native_lib_dir: Path = DEFAULT_LIB_DIR,
    progress: Callable[[str], None] | None = None,
) -> Method8RegistrationOutputs:
    resolved_method8_summary = method8_summary
    if resolved_method8_summary is None:
        resolved_output_dir = output_dir
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        summary_name = "registration.measurements.json"
        resolved_method8_summary = method8_output or (resolved_output_dir / summary_name)
        measure_method8_zcoverage(
            position_json=position_json,
            zarr_dir=zarr_dir,
            output=resolved_method8_summary,
            level2_screen=level2_screen,
            pairs=pairs,
            all_adjacent=all_adjacent,
            z_chunks=z_chunks,
            device=device,
            method8=method8,
            channel=channel,
            max_iterations=max_iterations,
            ftol=ftol,
            min_corr=min_corr,
            min_grad_ncc=min_grad_ncc,
            fixed_mask_threshold=fixed_mask_threshold,
            fixed_mask_min_voxels=fixed_mask_min_voxels,
            fixed_mask_max_masked_fraction=fixed_mask_max_masked_fraction,
            phase_recovery_shifted_crop=phase_recovery_shifted_crop,
            phase_recovery_min_prior_edges_per_axis=phase_recovery_min_prior_edges_per_axis,
            phase_recovery_min_phase_grad=min_phase_grad,
            phase_recovery_min_phase_corr=min_phase_corr,
            native_lib_dir=native_lib_dir,
            progress=progress,
        )
    else:
        resolved_output_dir = output_dir
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"optimizing positions from {resolved_method8_summary}")
    return optimize_positions_from_method8_summary(
        method8_summary=resolved_method8_summary,
        position_json=position_json,
        zarr_dir=zarr_dir,
        output_dir=resolved_output_dir,
        channel=channel,
        max_grad_regression=max_grad_regression,
        max_corr_regression=max_corr_regression,
        phase_fallback=phase_fallback,
        min_phase_grad=min_phase_grad,
        min_phase_corr=min_phase_corr,
        phase_fallback_weight_scale=phase_fallback_weight_scale,
    )
