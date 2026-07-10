from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as stitch_legacy
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
from squisher_lightsheet.seams import BoundaryConstraint, RobustBoundarySettings

DEFAULT_IMAGE14_BASE_DIR = Path("/home/chaichontat/nvme/lightsheet/working/eduseg/Image_14")
DEFAULT_IMAGE14_POSITION_JSON = DEFAULT_IMAGE14_BASE_DIR / "Image_14.metadata.positions.json"
DEFAULT_IMAGE14_ZARR_DIR = DEFAULT_IMAGE14_BASE_DIR / "squisher-deconv-run-u16" / "rechunked-ome-zarr"
DEFAULT_OUTPUT_PARENT_DIR = Path("/tmp/image14-method8-seams-conda")
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_PARENT_DIR / "optimized-mask3000-phase-gated"
DEFAULT_FIXED_MASK_THRESHOLD = 3000.0
DIMENSIONS = ("z", "y", "x")


@dataclass(frozen=True)
class TileInfo:
    tile_id: str
    tile_name: str
    path: Path
    start_um_zyx: np.ndarray
    spacing_um_zyx: np.ndarray
    shape_zyx: np.ndarray


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


def _default_output_dir(threshold: float | None) -> Path:
    return DEFAULT_OUTPUT_PARENT_DIR / f"optimized-{_threshold_slug(threshold)}-phase-gated"


def _summary_threshold(method8_summary: Path) -> float | None:
    return json.loads(method8_summary.read_text())["settings"]["fixed_mask_threshold"]


def _optimized_output_paths(output_dir: Path, threshold_slug: str) -> dict[str, Path]:
    return {
        "diagnostics": output_dir / "image14_method8_phase_gated_optimization.diagnostics.json",
        "positions": output_dir / f"Image_14.method8-{threshold_slug}.phase-gated.optimized.positions.json",
        "constraints": output_dir / "image14_method8_phase_gated_constraints.jsonl",
        "corrections": output_dir / "image14_method8_phase_gated_tile_corrections.json",
    }


def _remove_optimized_outputs(output_dir: Path, threshold_slug: str) -> None:
    for path in _optimized_output_paths(output_dir, threshold_slug).values():
        if path.exists():
            path.unlink()


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _load_array(path: Path) -> Any:
    return zarr.open_group(str(path), mode="r")["0"]


def _record_vector_zyx(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([float(values[dim]) for dim in DIMENSIONS], dtype=np.float64)


def _set_record_vector_zyx(record: dict[str, Any], key: str, values: np.ndarray) -> None:
    record[key] = {dim: float(value) for dim, value in zip(DIMENSIONS, values, strict=True)}


def _tile_id(tile_name: str) -> str:
    parts = tile_name.split(".")
    if len(parts) >= 2 and parts[0] == "Image_14":
        return parts[1]
    return tile_name


def _load_tiles(position_json: Path, zarr_dir: Path) -> dict[str, TileInfo]:
    payload = json.loads(position_json.read_text())
    tiles: dict[str, TileInfo] = {}
    for record in payload["tiles"]:
        tile_id = str(record["tile"]).split(".")[1]
        path = zarr_dir / f"Image_14.{tile_id}.ome.zarr"
        if not path.exists():
            raise FileNotFoundError(f"Missing tile OME-Zarr for {record['tile']}: {path}")
        array = _load_array(path)
        tiles[tile_id] = TileInfo(
            tile_id=tile_id,
            tile_name=f"Image_14.{tile_id}.ome.zarr",
            path=path,
            start_um_zyx=_record_vector_zyx(record, "translation_um"),
            spacing_um_zyx=_record_vector_zyx(record, "scale_um"),
            shape_zyx=np.asarray(array.shape, dtype=np.int64),
        )
    return tiles


def _zarr_shape(path: Path) -> tuple[int, ...]:
    root = zarr.open(str(path), mode="r")
    if hasattr(root, "shape"):
        return tuple(int(value) for value in root.shape)
    if "0" not in root:
        raise ValueError(f"Expected OME-Zarr level 0 at {path}")
    return tuple(int(value) for value in root["0"].shape)


def _load_position_tiles(
    position_json: Path,
    zarr_dir: Path,
) -> tuple[dict[str, Any], list[str], dict[str, int], np.ndarray, list[stitch_legacy.TileMetadata]]:
    payload = json.loads(position_json.read_text())
    records = payload["tiles"]
    tile_names = [str(record["tile"]).replace(".ome.tif", ".ome.zarr") for record in records]
    tile_ids = [name.split(".")[1] for name in tile_names]
    tile_index = {tile_id: index for index, tile_id in enumerate(tile_ids)}
    spacing = _record_vector_zyx(records[0], "scale_um")
    track = stitch_legacy.TrackMetadata(slug="track0", track_id="track0", channels=(0,), channel_names=("0",))
    tiles = [
        stitch_legacy.TileMetadata(
            path=zarr_dir / tile_name,
            shape=_zarr_shape(zarr_dir / tile_name),
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

    fixed_stop_um = fixed.start_um_zyx + fixed.shape_zyx * fixed.spacing_um_zyx
    moving_stop_um = moving.start_um_zyx + moving.shape_zyx * moving.spacing_um_zyx
    overlap_start_um = np.maximum(fixed.start_um_zyx, moving.start_um_zyx)
    overlap_stop_um = np.minimum(fixed_stop_um, moving_stop_um)
    overlap_px = np.floor((overlap_stop_um - overlap_start_um) / fixed.spacing_um_zyx + 1e-6).astype(np.int64)
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

    start_um = overlap_start_um.copy()
    start_um[0] = (
        fixed.start_um_zyx[0]
        + min(max(int(z_start), 0), int(shape[0] - crop_shape[0])) * fixed.spacing_um_zyx[0]
    )
    fixed_start = np.rint((start_um - fixed.start_um_zyx) / fixed.spacing_um_zyx).astype(np.int64)
    moving_start = np.rint((start_um - moving.start_um_zyx) / moving.spacing_um_zyx).astype(np.int64)

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
        fixed_stop_um = fixed.start_um_zyx + fixed.shape_zyx * fixed.spacing_um_zyx
        for moving in records[index + 1 :]:
            if not np.allclose(fixed.spacing_um_zyx, moving.spacing_um_zyx, rtol=0.0, atol=1e-9):
                continue
            moving_stop_um = moving.start_um_zyx + moving.shape_zyx * moving.spacing_um_zyx
            overlap_start_um = np.maximum(fixed.start_um_zyx, moving.start_um_zyx)
            overlap_stop_um = np.minimum(fixed_stop_um, moving_stop_um)
            overlap_px = np.floor((overlap_stop_um - overlap_start_um) / fixed.spacing_um_zyx + 1e-6).astype(
                np.int64
            )
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
    max_iterations: int,
    ftol: float,
    min_corr: float,
    min_grad_ncc: float,
    fixed_mask_threshold: float | None,
    fixed_mask_min_voxels: int,
    fixed_mask_max_masked_fraction: float,
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
    fixed_raw = np.asarray(_load_array(fixed.path)[fixed_slices], dtype=np.float32)
    moving_raw = np.asarray(_load_array(moving.path)[moving_slices], dtype=np.float32)

    cp.cuda.Device(int(device)).use()
    fixed_gpu, fixed_content = _robust_norm_and_content_stats_cupy(fixed_raw)
    moving_gpu, moving_content = _robust_norm_and_content_stats_cupy(moving_raw)
    fixed_fit = _block_mean_downsample_zyx_cupy(fixed_gpu, fit_downsample)
    moving_fit = _block_mean_downsample_zyx_cupy(moving_gpu, fit_downsample)
    fixed_fit_mask = None
    fixed_threshold_mask = None
    mask_rejection_reason = None
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
        if int(fit_stats["voxel_count"]) < int(fixed_mask_min_voxels):
            mask_rejection_reason = "fixed_threshold_fit_mask_too_few_voxels"
        elif float(fit_stats["masked_fraction"]) > float(fixed_mask_max_masked_fraction):
            mask_rejection_reason = "fixed_threshold_fit_mask_too_sparse"

    corr_initial = _corr_gpu(fixed_fit, moving_fit)
    grad_initial = gradient_component_ncc_3d_gpu(fixed_fit, moving_fit)

    phase_started = time.perf_counter()
    phase_shift_fit, phase_error, phase_diff = phase_cross_correlation(
        fixed_fit,
        moving_fit,
        upsample_factor=10,
    )
    phase_shift_fit_np = cp.asnumpy(phase_shift_fit).astype(np.float32)
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
        "fixed_threshold_mask": fixed_threshold_mask,
        "phase_shift_fit_zyx": phase_shift_fit_np.astype(np.float64).tolist(),
        "phase_shift_zyx": (phase_shift_fit_np.astype(np.float64) * np.asarray(fit_downsample)).tolist(),
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
    if mask_rejection_reason is not None:
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
            "status": "rejected",
            "rejection_reason": mask_rejection_reason,
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
        lib_dir=DEFAULT_LIB_DIR,
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


def _pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["status"] == "accepted"]
    translations = np.asarray([row["local_translation_zyx"] for row in accepted], dtype=np.float64)
    phase = np.asarray([row["phase_shift_zyx"] for row in accepted], dtype=np.float64)
    if translations.size == 0:
        return {"accepted_count": 0, "rejected_count": len(rows)}
    return {
        "accepted_count": len(accepted),
        "rejected_count": len(rows) - len(accepted),
        "translation_median_zyx": np.median(translations, axis=0).tolist(),
        "translation_mean_zyx": np.mean(translations, axis=0).tolist(),
        "translation_std_zyx": np.std(translations, axis=0).tolist(),
        "translation_min_zyx": np.min(translations, axis=0).tolist(),
        "translation_max_zyx": np.max(translations, axis=0).tolist(),
        "translation_range_zyx": np.ptp(translations, axis=0).tolist(),
        "phase_shift_median_zyx": np.median(phase, axis=0).tolist(),
        "phase_shift_mean_zyx": np.mean(phase, axis=0).tolist(),
        "phase_shift_std_zyx": np.std(phase, axis=0).tolist(),
    }


def measure_method8_zcoverage(
    *,
    position_json: Path,
    zarr_dir: Path,
    output: Path,
    pairs: tuple[str, ...] | None = None,
    all_adjacent: bool = True,
    z_chunks: int = 6,
    device: int = 0,
    max_iterations: int = 300,
    ftol: float = 1e-4,
    min_corr: float = 0.15,
    min_grad_ncc: float = 0.24,
    fixed_mask_threshold: float | None = DEFAULT_FIXED_MASK_THRESHOLD,
    fixed_mask_min_voxels: int = 256,
    fixed_mask_max_masked_fraction: float = 0.95,
    progress: Callable[[str], None] | None = None,
) -> Path:
    progress = progress or (lambda _message: None)
    tiles = _load_tiles(position_json, zarr_dir)
    resolved_pairs = _all_adjacent_pairs(tiles) if all_adjacent else list(pairs or ())
    if not resolved_pairs:
        raise ValueError("No adjacent tile pairs were found for Method8 registration")

    rows: list[dict[str, Any]] = []
    pair_summaries: dict[str, Any] = {}
    for pair in resolved_pairs:
        fixed_id, moving_id = pair.split("-", 1)
        fixed = tiles[fixed_id]
        moving = tiles[moving_id]
        pair_rows = []
        chunks = _z_chunks(int(fixed.shape_zyx[0]), int(z_chunks))
        for chunk_index, (z_start, z_stop) in enumerate(chunks):
            progress(f"method8 pair={pair} chunk={chunk_index + 1}/{len(chunks)} z={z_start}:{z_stop}")
            row = _measure_z_chunk(
                fixed=fixed,
                moving=moving,
                z_start=z_start,
                z_stop=z_stop,
                device=int(device),
                max_iterations=int(max_iterations),
                ftol=float(ftol),
                min_corr=float(min_corr),
                min_grad_ncc=float(min_grad_ncc),
                fixed_mask_threshold=fixed_mask_threshold,
                fixed_mask_min_voxels=int(fixed_mask_min_voxels),
                fixed_mask_max_masked_fraction=float(fixed_mask_max_masked_fraction),
            )
            pair_rows.append(row)
            rows.append(row)
            progress(
                " ".join(
                    [
                        f"status={row['status']}",
                        f"reason={row['rejection_reason']}",
                        f"phase={row['phase_shift_zyx']}",
                        f"method8={row['local_translation_zyx']}",
                        f"grad={row['gradient_component_ncc_phase_mean']}->{row['gradient_component_ncc_method8_mean']}",
                    ]
                )
            )
        pair_summaries[pair] = _pair_summary(pair_rows)

    payload = {
        "schema_version": 1,
        "artifact_type": "lightsheet.image14_method8_zcoverage_trial.v1",
        "position_json": str(position_json),
        "zarr_dir": str(zarr_dir),
        "settings": {
            "pairs": list(pairs or ()),
            "resolved_pairs": resolved_pairs,
            "all_adjacent": bool(all_adjacent),
            "z_chunks": int(z_chunks),
            "device": int(device),
            "max_iterations": int(max_iterations),
            "ftol": float(ftol),
            "min_corr": float(min_corr),
            "min_grad_ncc": float(min_grad_ncc),
            "phase_primed": True,
            "face_span": "full_overlap",
            "fixed_mask_threshold": fixed_mask_threshold,
            "fixed_mask_min_voxels": int(fixed_mask_min_voxels),
            "fixed_mask_max_masked_fraction": float(fixed_mask_max_masked_fraction),
        },
        "pair_summaries": pair_summaries,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    progress(f"wrote {output}")
    return output


def _quality_gate(
    row: dict[str, Any], *, max_grad_regression: float, max_corr_regression: float
) -> str | None:
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
    if phase_grad is None or not np.isfinite(float(phase_grad)):
        return "phase_gradient_component_ncc_not_finite"
    if float(phase_grad) < float(min_phase_grad):
        return "phase_gradient_component_ncc_too_low"
    if phase_corr is None or not np.isfinite(float(phase_corr)):
        return "phase_corr_not_finite"
    if float(phase_corr) < float(min_phase_corr):
        return "phase_corr_too_low"
    return None


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
) -> tuple[list[BoundaryConstraint], Counter[str], dict[str, int]]:
    constraints: list[BoundaryConstraint] = []
    rejected: Counter[str] = Counter()
    method8_grouped: defaultdict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    phase_grouped: defaultdict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    all_edges: set[tuple[int, int, str]] = set()
    threshold_slug = _threshold_slug(summary["settings"]["fixed_mask_threshold"])
    for row in summary["rows"]:
        fixed_id = _tile_id(str(row["fixed_tile"]))
        moving_id = _tile_id(str(row["moving_tile"]))
        if fixed_id not in tile_index or moving_id not in tile_index:
            raise ValueError(f"Summary row references tile outside position input: {fixed_id}-{moving_id}")
        fixed_index = tile_index[fixed_id]
        moving_index = tile_index[moving_id]
        edge_key = (fixed_index, moving_index, str(row["seam_axis"]))
        all_edges.add(edge_key)
        gate_reason = _quality_gate(
            row,
            max_grad_regression=max_grad_regression,
            max_corr_regression=max_corr_regression,
        )
        if gate_reason is not None:
            rejected[gate_reason] += 1
        else:
            method8_grouped[edge_key].append(row)
        if (
            use_phase_fallback
            and _phase_gate(row, min_phase_grad=min_phase_grad, min_phase_corr=min_phase_corr) is None
        ):
            phase_grouped[edge_key].append(row)

    source_counts: Counter[str] = Counter()
    fallback_missed = 0
    for fixed_index, moving_index, axis in sorted(all_edges):
        source = "method8"
        edge_rows = method8_grouped.get((fixed_index, moving_index, axis), [])
        if not edge_rows and use_phase_fallback:
            edge_rows = phase_grouped.get((fixed_index, moving_index, axis), [])
            source = "phase_fallback"
        if not edge_rows:
            fallback_missed += 1
            continue
        source_counts[source] += 1
        shift_key = "local_translation_zyx" if source == "method8" else "phase_shift_zyx"
        after_grad_key = (
            "gradient_component_ncc_method8_mean"
            if source == "method8"
            else "gradient_component_ncc_phase_mean"
        )
        after_corr_key = "corr_method8" if source == "method8" else "corr_phase"
        shifts = np.asarray([row[shift_key] for row in edge_rows], dtype=np.float64)
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
        weight = max(grad_after - 0.15, 1e-3)
        if source == "phase_fallback":
            weight *= float(phase_fallback_weight_scale)
        pair = (fixed_index, moving_index)
        representative = edge_rows[len(edge_rows) // 2]
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
                source_label=f"image14_{source}_{threshold_slug}_phase_gated_median_n{len(edge_rows)}",
            )
        )
    source_counts["missing_edges_after_fallback"] = fallback_missed
    return constraints, rejected, dict(source_counts)


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


def _write_optimized_positions(
    *,
    position_payload: dict[str, Any],
    corrections_px: np.ndarray,
    spacing_zyx: np.ndarray,
    diagnostics_path: Path,
    output_path: Path,
) -> None:
    updated = json.loads(json.dumps(position_payload))
    for index, record in enumerate(updated["tiles"]):
        translation = _record_vector_zyx(record, "translation_um") + corrections_px[index] * spacing_zyx
        _set_record_vector_zyx(record, "translation_um", translation)
    updated["source"] = f"{updated.get('source', 'position file')} + Image_14 method8 seam optimization"
    updated["derived_by"] = "lightsheet-stitch register"
    updated["optimization_diagnostics"] = str(diagnostics_path.resolve())
    _write_text_atomic(output_path, json.dumps(updated, indent=2) + "\n")


def optimize_positions_from_method8_summary(
    *,
    method8_summary: Path,
    position_json: Path,
    zarr_dir: Path,
    output_dir: Path,
    max_grad_regression: float = 0.02,
    max_corr_regression: float = 0.01,
    phase_fallback: bool = True,
    min_phase_grad: float = 0.24,
    min_phase_corr: float = 0.15,
    phase_fallback_weight_scale: float = 0.1,
) -> Method8RegistrationOutputs:
    summary = json.loads(method8_summary.read_text())
    threshold_slug = _threshold_slug(summary["settings"]["fixed_mask_threshold"])
    position_payload, tile_names, tile_index, spacing, tiles = _load_position_tiles(position_json, zarr_dir)
    constraints, gate_rejections, constraint_sources = constraints_from_method8_summary(
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
        raise ValueError("No Method8 constraints survived the phase-regression gate")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _optimized_output_paths(output_dir, threshold_slug)
    diagnostics_path = output_paths["diagnostics"]
    positions_path = output_paths["positions"]
    residuals_path = output_paths["constraints"]
    corrections_path = output_paths["corrections"]
    _remove_optimized_outputs(output_dir, threshold_slug)

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
        "artifact_type": "lightsheet.image14_method8_phase_gated_tile_optimization.v1",
        "method8_summary": str(method8_summary),
        "position_json": str(position_json),
        "settings": {
            "max_grad_regression": float(max_grad_regression),
            "max_corr_regression": float(max_corr_regression),
            "constraint_aggregation": "median_shift_per_fixed_moving_axis_edge",
            "phase_fallback": bool(phase_fallback),
            "min_phase_grad": float(min_phase_grad),
            "min_phase_corr": float(min_phase_corr),
            "phase_fallback_weight_scale": float(phase_fallback_weight_scale),
            "solver": "multiview_stitcher.global_optimization.translation",
            "zarr_dir": str(zarr_dir),
            "robust_boundary_settings": asdict(settings),
        },
        "tile_count": len(tile_names),
        "anchor_tile_index": int(anchor_tile),
        "anchor_tile": tile_names[anchor_tile],
        "connected_tile_count": len(connected),
        "input_rows": len(summary["rows"]),
        "constraint_count": len(constraints),
        "constraint_sources": constraint_sources,
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
            "positions": str(positions_path),
            "constraints_jsonl": str(residuals_path),
            "corrections": str(corrections_path),
        },
    }
    _write_text_atomic(residuals_path, residual_lines)
    _write_text_atomic(diagnostics_path, json.dumps(diagnostics, indent=2) + "\n")
    _write_text_atomic(
        corrections_path,
        json.dumps(
            {
                "tile_names": tile_names,
                "spacing_zyx_um": spacing.tolist(),
                "corrections_zyx_px": corrections_array.tolist(),
                "corrections_zyx_um": (corrections_array * spacing).tolist(),
            },
            indent=2,
        )
        + "\n",
    )
    _write_optimized_positions(
        position_payload=position_payload,
        corrections_px=corrections_array,
        spacing_zyx=spacing,
        diagnostics_path=diagnostics_path,
        output_path=positions_path,
    )
    return Method8RegistrationOutputs(
        method8_summary=method8_summary,
        optimized_positions=positions_path,
        diagnostics=diagnostics_path,
        constraints_jsonl=residuals_path,
        tile_corrections=corrections_path,
    )


def register_image14_method8(
    *,
    position_json: Path = DEFAULT_IMAGE14_POSITION_JSON,
    zarr_dir: Path = DEFAULT_IMAGE14_ZARR_DIR,
    output_dir: Path | None = None,
    method8_summary: Path | None = None,
    method8_output: Path | None = None,
    pairs: tuple[str, ...] | None = None,
    all_adjacent: bool = True,
    z_chunks: int = 6,
    device: int = 0,
    max_iterations: int = 300,
    ftol: float = 1e-4,
    min_corr: float = 0.15,
    min_grad_ncc: float = 0.24,
    fixed_mask_threshold: float | None = DEFAULT_FIXED_MASK_THRESHOLD,
    fixed_mask_min_voxels: int = 256,
    fixed_mask_max_masked_fraction: float = 0.95,
    max_grad_regression: float = 0.02,
    max_corr_regression: float = 0.01,
    phase_fallback: bool = True,
    min_phase_grad: float = 0.24,
    min_phase_corr: float = 0.15,
    phase_fallback_weight_scale: float = 0.1,
    progress: Callable[[str], None] | None = None,
) -> Method8RegistrationOutputs:
    resolved_method8_summary = method8_summary
    if resolved_method8_summary is None:
        resolved_output_dir = output_dir or _default_output_dir(fixed_mask_threshold)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        summary_name = (
            f"image14_method8_all_adjacent_zcoverage_{_threshold_slug(fixed_mask_threshold)}_summary.json"
        )
        resolved_method8_summary = method8_output or (resolved_output_dir / summary_name)
        measure_method8_zcoverage(
            position_json=position_json,
            zarr_dir=zarr_dir,
            output=resolved_method8_summary,
            pairs=pairs,
            all_adjacent=all_adjacent,
            z_chunks=z_chunks,
            device=device,
            max_iterations=max_iterations,
            ftol=ftol,
            min_corr=min_corr,
            min_grad_ncc=min_grad_ncc,
            fixed_mask_threshold=fixed_mask_threshold,
            fixed_mask_min_voxels=fixed_mask_min_voxels,
            fixed_mask_max_masked_fraction=fixed_mask_max_masked_fraction,
            progress=progress,
        )
    else:
        resolved_output_dir = output_dir or _default_output_dir(_summary_threshold(resolved_method8_summary))
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"optimizing positions from {resolved_method8_summary}")
    return optimize_positions_from_method8_summary(
        method8_summary=resolved_method8_summary,
        position_json=position_json,
        zarr_dir=zarr_dir,
        output_dir=resolved_output_dir,
        max_grad_regression=max_grad_regression,
        max_corr_regression=max_corr_regression,
        phase_fallback=phase_fallback,
        min_phase_grad=min_phase_grad,
        min_phase_corr=min_phase_corr,
        phase_fallback_weight_scale=phase_fallback_weight_scale,
    )
