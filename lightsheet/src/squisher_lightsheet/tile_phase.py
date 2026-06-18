from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.artifacts import stamp_artifact
from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as rough_legacy
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as stitch_legacy


DIMENSIONS = ("z", "y", "x")
PATCH_INLIER_THRESHOLDS_ZYX = np.asarray([3.0, 12.0, 12.0], dtype=np.float64)


def normalize_volume_for_phase(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    positive = finite[finite > 0]
    out = np.zeros(volume.shape, dtype=np.float32)
    if positive.size == 0:
        return out
    low, high = np.percentile(positive, [1.0, 99.5])
    clipped = np.clip(volume, low, high)
    valid = np.isfinite(clipped)
    centered = clipped - float(np.median(clipped[valid]))
    denom = max(float(np.percentile(np.abs(centered[valid]), 95.0)), 1.0)
    out[valid] = centered[valid] / denom
    return out


def estimate_tile_shift_zyx_px(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    upsample_factor: int = 10,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation

    if fixed.shape != moving.shape:
        min_shape = tuple(min(a, b) for a, b in zip(fixed.shape, moving.shape, strict=True))
        fixed = fixed[tuple(slice(0, size) for size in min_shape)]
        moving = moving[tuple(slice(0, size) for size in min_shape)]
    if fixed.ndim != 3:
        raise ValueError(f"Expected 3D z/y/x arrays, got shape {fixed.shape}")
    fixed_norm = normalize_volume_for_phase(fixed)
    moving_norm = normalize_volume_for_phase(moving)
    shift, error, phase = phase_cross_correlation(
        fixed_norm,
        moving_norm,
        upsample_factor=upsample_factor,
        normalization="phase",
    )
    shift = np.asarray(shift, dtype=np.float64)
    shifted = ndimage.shift(moving_norm, shift=shift, order=1, mode="constant", cval=0.0, prefilter=False)
    mask_before = np.isfinite(fixed_norm) & np.isfinite(moving_norm)
    mask_after = np.isfinite(fixed_norm) & np.isfinite(shifted)
    return shift, {
        "shape_zyx": [int(value) for value in fixed_norm.shape],
        "upsample_factor": int(upsample_factor),
        "phase_error": float(error),
        "phase": float(phase),
        "corr_before": corrcoef_on_mask(fixed_norm, moving_norm, mask_before),
        "corr_after": corrcoef_on_mask(fixed_norm, shifted, mask_after),
    }


def estimate_tile_shift_zyx_px_gpu(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    if fixed.shape != moving.shape:
        min_shape = tuple(min(a, b) for a, b in zip(fixed.shape, moving.shape, strict=True))
        fixed = fixed[tuple(slice(0, size) for size in min_shape)]
        moving = moving[tuple(slice(0, size) for size in min_shape)]
    if fixed.ndim != 3:
        raise ValueError(f"Expected 3D z/y/x arrays, got shape {fixed.shape}")
    fixed_norm = normalize_volume_for_phase(fixed)
    moving_norm = normalize_volume_for_phase(moving)
    try:
        shift, peak = stitch_legacy.phase_correlation_shift_gpu(fixed_norm, moving_norm)
    except Exception as exc:
        raise RuntimeError("CuPy phase correlation failed; patch mode requires GPU FFT support") from exc
    shift_array = np.asarray(shift, dtype=np.float64)
    shifted = ndimage.shift(moving_norm, shift=shift_array, order=1, mode="constant", cval=0.0, prefilter=False)
    mask_before = np.isfinite(fixed_norm) & np.isfinite(moving_norm)
    mask_after = np.isfinite(fixed_norm) & np.isfinite(shifted)
    return shift_array, {
        "shape_zyx": [int(value) for value in fixed_norm.shape],
        "peak": float(peak),
        "corr_before": corrcoef_on_mask(fixed_norm, moving_norm, mask_before),
        "corr_after": corrcoef_on_mask(fixed_norm, shifted, mask_after),
    }


def corrcoef_on_mask(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 8:
        return None
    a = fixed[mask].astype(np.float64)
    b = moving[mask].astype(np.float64)
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def corresponding_moving_path(reference_path: Path, *, reference_token: str, moving_token: str) -> Path:
    text = str(reference_path)
    if reference_token not in text:
        raise ValueError(f"{reference_path} does not contain reference token {reference_token!r}")
    path = Path(text.replace(reference_token, moving_token))
    if not path.exists():
        raise FileNotFoundError(f"Corresponding moving tile does not exist: {path}")
    return path


def make_moving_tile_name(reference_tile: str, *, reference_token: str, moving_token: str) -> str:
    if reference_token not in reference_tile:
        raise ValueError(f"Tile name {reference_tile!r} does not contain reference token {reference_token!r}")
    return reference_tile.replace(reference_token, moving_token)


def parse_shape_zyx(text: str) -> tuple[int, int, int]:
    values = tuple(int(part) for part in text.split(","))
    if len(values) != 3 or any(value < 1 for value in values):
        raise ValueError(f"Expected comma-separated positive z,y,x shape, got {text!r}")
    return values


def slice_shape_zyx(slices_zyx: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    return tuple(int(slc.stop - slc.start) for slc in slices_zyx)


def slices_to_json(slices_zyx: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(slc.start), int(slc.stop)] for slc in slices_zyx]


def shifted_slices_zyx(
    fixed_slices: tuple[slice, slice, slice],
    *,
    shift_zyx_px: np.ndarray,
) -> tuple[slice, slice, slice]:
    offsets = np.rint(-shift_zyx_px).astype(int)
    return tuple(
        slice(int(slc.start + offset), int(slc.stop + offset))
        for slc, offset in zip(fixed_slices, offsets, strict=True)
    )


def slices_within_shape(slices_zyx: tuple[slice, slice, slice], shape_zyx: np.ndarray) -> bool:
    return all(0 <= slc.start < slc.stop <= int(size) for slc, size in zip(slices_zyx, shape_zyx, strict=True))


def raw_axis_slice_for_oriented_slice(slc: slice, *, axis_size: int, flipped: bool) -> tuple[slice, bool]:
    if slc.step not in (None, 1):
        raise ValueError(f"Patch slices must use unit steps, got {slc}")
    start = int(slc.start)
    stop = int(slc.stop)
    if not 0 <= start < stop <= int(axis_size):
        raise ValueError(f"Slice {slc} is outside axis size {axis_size}")
    if not flipped:
        return slice(start, stop), False
    return slice(int(axis_size) - stop, int(axis_size) - start), True


def read_tile_patch(
    tile: rough_legacy.TileRecord,
    *,
    channel: int,
    slices_zyx: tuple[slice, slice, slice],
) -> np.ndarray:
    import dask.array as da
    import tifffile
    import zarr

    if not slices_within_shape(slices_zyx, tile.shape_zyx):
        raise ValueError(f"Patch slices {slices_zyx} are outside tile shape {tile.shape_zyx.tolist()}")

    store = tifffile.imread(tile.path, aszarr=True)
    try:
        zarray = rough_legacy.base_zarr_array(zarr.open(store, mode="r"))
        array = da.from_zarr(zarray)
        raw_slices = []
        reverse_axes = []
        for axis, slc in enumerate(slices_zyx):
            raw_slice, reverse_axis = raw_axis_slice_for_oriented_slice(
                slc,
                axis_size=int(tile.shape_zyx[axis]),
                flipped=bool(tile.scale_zyx_um[axis] < 0),
            )
            raw_slices.append(raw_slice)
            reverse_axes.append(reverse_axis)
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            patch = array[(channel, *raw_slices)]
        elif tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
            patch = array[tuple(raw_slices)]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        result = np.asarray(patch.compute(), dtype=np.float32)
        for axis, reverse_axis in enumerate(reverse_axes):
            if reverse_axis:
                result = np.flip(result, axis=axis)
        return result
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def tiff_series_level_count(path: Path) -> int:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        return len(tif.series[0].levels)


def source_shape_zyx_from_array_shape(shape: tuple[int, ...], axes: str) -> np.ndarray:
    if axes == "CZYX":
        return np.asarray(shape[1:4], dtype=np.int64)
    if axes == "ZYX":
        return np.asarray(shape, dtype=np.int64)
    raise ValueError(f"Unsupported axes {axes!r}")


def sampled_tile_volume_from_subifd(
    tile: rough_legacy.TileRecord,
    *,
    channel: int,
    requested_level: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    import dask.array as da
    import tifffile
    import zarr

    if requested_level < 0:
        raise ValueError("requested_level must be non-negative")
    available_levels = tiff_series_level_count(tile.path)
    source_level = min(int(requested_level), max(0, available_levels - 1))
    desired_factor = 2**int(requested_level)
    store = tifffile.imread(tile.path, aszarr=True, level=source_level)
    try:
        zarray = rough_legacy.base_zarr_array(zarr.open(store, mode="r"))
        array = da.from_zarr(zarray)
        source_shape_zyx = source_shape_zyx_from_array_shape(tuple(int(value) for value in array.shape), tile.axes)
        source_factor_zyx = np.maximum(
            1,
            np.rint(tile.shape_zyx.astype(np.float64) / source_shape_zyx.astype(np.float64)).astype(np.int64),
        )
        remaining_step_zyx = np.maximum(
            1,
            np.ceil(desired_factor / source_factor_zyx.astype(np.float64)).astype(np.int64),
        )
        z_step = int(-remaining_step_zyx[0] if tile.scale_zyx_um[0] < 0 else remaining_step_zyx[0])
        y_step = int(-remaining_step_zyx[1] if tile.scale_zyx_um[1] < 0 else remaining_step_zyx[1])
        x_step = int(-remaining_step_zyx[2] if tile.scale_zyx_um[2] < 0 else remaining_step_zyx[2])
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            volume = array[channel, ::z_step, ::y_step, ::x_step]
        elif tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
            volume = array[::z_step, ::y_step, ::x_step]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        effective_factor_zyx = source_factor_zyx * remaining_step_zyx
        return np.asarray(volume.compute(), dtype=np.float32), effective_factor_zyx.astype(np.float64), source_level, available_levels
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def candidate_patch_slices(
    scout_volume: np.ndarray,
    *,
    tile_shape_zyx: np.ndarray,
    patch_shape_zyx: tuple[int, int, int],
    scout_scale_zyx: np.ndarray,
    max_candidates: int,
    moving_shape_zyx: np.ndarray | None = None,
    shift_zyx_px: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if any(size > int(tile_size) for size, tile_size in zip(patch_shape_zyx, tile_shape_zyx, strict=True)):
        raise ValueError(
            f"Patch shape {list(patch_shape_zyx)} is larger than tile shape {tile_shape_zyx.tolist()}"
        )

    starts_by_axis = []
    for patch_size, tile_size in zip(patch_shape_zyx, tile_shape_zyx, strict=True):
        tile_size = int(tile_size)
        if tile_size == patch_size:
            starts_by_axis.append([0])
            continue
        step = max(1, patch_size // 2)
        starts = list(range(0, tile_size - patch_size + 1, step))
        if starts[-1] != tile_size - patch_size:
            starts.append(tile_size - patch_size)
        starts_by_axis.append(starts)

    ranked = []
    scout_norm = normalize_volume_for_phase(scout_volume)
    for z0 in starts_by_axis[0]:
        for y0 in starts_by_axis[1]:
            for x0 in starts_by_axis[2]:
                fixed_slices = (slice(z0, z0 + patch_shape_zyx[0]), slice(y0, y0 + patch_shape_zyx[1]), slice(x0, x0 + patch_shape_zyx[2]))
                if moving_shape_zyx is not None and shift_zyx_px is not None:
                    moving_slices = shifted_slices_zyx(fixed_slices, shift_zyx_px=shift_zyx_px)
                    if not slices_within_shape(moving_slices, moving_shape_zyx):
                        continue
                scout_slices = tuple(
                    slice(
                        max(0, int(np.floor(slc.start / scout_scale_zyx[axis]))),
                        min(int(scout_volume.shape[axis]), int(np.ceil(slc.stop / scout_scale_zyx[axis]))),
                    )
                    for axis, slc in enumerate(fixed_slices)
                )
                scout_patch = scout_norm[scout_slices]
                positive_fraction = float(np.count_nonzero(scout_patch > 0) / scout_patch.size) if scout_patch.size else 0.0
                score = float(np.std(scout_patch)) * max(positive_fraction, 1e-6)
                ranked.append(
                    {
                        "fixed_slices": fixed_slices,
                        "scout_slices": scout_slices,
                        "content_score": score,
                        "positive_fraction": positive_fraction,
                    }
                )
    ranked.sort(key=lambda item: item["content_score"], reverse=True)
    return ranked[:max_candidates]


def estimate_patch_shift_zyx_px(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    if fixed.shape != moving.shape:
        raise ValueError(f"Patch phase correlation requires matching shapes, got {fixed.shape} and {moving.shape}")
    fixed_norm = normalize_volume_for_phase(fixed)
    moving_norm = normalize_volume_for_phase(moving)
    try:
        shift, peak = stitch_legacy.phase_correlation_shift_gpu(fixed_norm, moving_norm)
    except Exception as exc:
        raise RuntimeError("CuPy patch phase correlation failed; patch mode requires GPU FFT support") from exc
    shift_array = np.asarray(shift, dtype=np.float64)
    shifted = ndimage.shift(moving_norm, shift=shift_array, order=1, mode="constant", cval=0.0, prefilter=False)
    finite = np.isfinite(fixed_norm) & np.isfinite(moving_norm)
    shifted_finite = np.isfinite(fixed_norm) & np.isfinite(shifted)
    return shift_array, {
        "shape_zyx": [int(value) for value in fixed_norm.shape],
        "peak": float(peak),
        "corr_before": corrcoef_on_mask(fixed_norm, moving_norm, finite),
        "corr_after": corrcoef_on_mask(fixed_norm, shifted, shifted_finite),
    }


def select_inlier_patch_measurements(
    total_shifts: np.ndarray,
    *,
    thresholds_zyx: np.ndarray = PATCH_INLIER_THRESHOLDS_ZYX,
    min_inliers: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if total_shifts.ndim != 2 or total_shifts.shape[1] != 3:
        raise ValueError(f"Expected n x 3 total shifts, got {total_shifts.shape}")
    if total_shifts.shape[0] == 0:
        raise ValueError("No accepted patch shifts are available for inlier selection")
    neighbor_masks = np.all(np.abs(total_shifts[:, None, :] - total_shifts[None, :, :]) <= thresholds_zyx, axis=2)
    counts = neighbor_masks.sum(axis=1)
    best_count = int(counts.max())
    best_indices = np.flatnonzero(counts == best_count)
    if best_indices.size > 1:
        medians = np.asarray([np.median(total_shifts[neighbor_masks[index]], axis=0) for index in best_indices])
        distances = np.asarray(
            [np.median(np.linalg.norm(total_shifts[neighbor_masks[index]] - median, axis=1)) for index, median in zip(best_indices, medians, strict=True)]
        )
        best_index = int(best_indices[int(np.argmin(distances))])
    else:
        best_index = int(best_indices[0])
    inliers = neighbor_masks[best_index]
    if int(np.count_nonzero(inliers)) < min_inliers:
        raise ValueError(f"Only {int(np.count_nonzero(inliers))} inlier patch shifts found; require {min_inliers}")
    return inliers, np.median(total_shifts[inliers], axis=0)


def position_records_by_tile(position_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["tile"]: record for record in position_payload["tiles"]}


def tile_phase_cache_key(
    *,
    reference_position: Path,
    reference_channel: int,
    reference_token: str,
    moving_token: str,
    level: int,
    upsample_factor: int,
    patch_shape_zyx: tuple[int, int, int] | None,
    min_inliers: int,
    max_candidate_patches: int,
    coarse_level: int,
) -> dict[str, Any]:
    return {
        "cache_version": "tile_phase_patch_inbounds_v1",
        "reference_position": str(reference_position.resolve()),
        "reference_channel": int(reference_channel),
        "reference_token": reference_token,
        "moving_token": moving_token,
        "level": int(level),
        "upsample_factor": int(upsample_factor),
        "patch_shape_zyx": None if patch_shape_zyx is None else [int(value) for value in patch_shape_zyx],
        "min_inliers": int(min_inliers),
        "max_candidate_patches": int(max_candidate_patches),
        "coarse_level": int(coarse_level),
    }


def load_tile_phase_cache(cache_path: Path, cache_key: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text())
    if payload.get("cache_key") != cache_key:
        return {}
    return {row["tile"]: row for row in payload.get("measurements", [])}


def write_tile_phase_cache(cache_path: Path, cache_key: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "lightsheet.tile_phase_measurement_cache.v1",
        "cache_key": cache_key,
        "measurements": rows,
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(cache_path)


def apply_shift_row_to_position_record(record: dict[str, Any], row: dict[str, Any], *, moving_path: Path) -> np.ndarray:
    record["tile"] = row["tile"]
    record["path"] = str(moving_path)
    shift_um = np.asarray(row["shift_um_zyx"], dtype=np.float64)
    for axis, value in zip(DIMENSIONS, shift_um, strict=True):
        record["translation_um"][axis] = float(record["translation_um"][axis] + value)
    return shift_um


def adapt_registration_from_reference(
    *,
    reference_registration_input: Path,
    output_registration: Path,
    adapted_position_payload: dict[str, Any],
    reference_token: str,
    moving_token: str,
    adapted_to_position: Path,
    tile_phase_summary: dict[str, Any],
) -> Path:
    reference_registration = json.loads(reference_registration_input.read_text())
    adapted = json.loads(json.dumps(reference_registration))
    position_by_tile = position_records_by_tile(adapted_position_payload)
    adapted_tiles = []
    missing = []
    for record in adapted["tiles"]:
        moving_tile = make_moving_tile_name(record["tile"], reference_token=reference_token, moving_token=moving_token)
        position_record = position_by_tile.get(moving_tile)
        if position_record is None:
            missing.append(moving_tile)
            continue
        adapted_record = json.loads(json.dumps(record))
        adapted_record["tile"] = moving_tile
        adapted_record["stage_translation_um"] = position_record["translation_um"]
        adapted_record["stage_scale_um"] = position_record["scale_um"]
        if "path" in adapted_record or position_record.get("path") is not None:
            adapted_record["path"] = position_record["path"]
        adapted_tiles.append(adapted_record)
    if missing:
        raise ValueError(f"Adapted position file is missing tiles required by registration: {missing}")
    adapted["tiles"] = adapted_tiles
    if adapted_tiles:
        first_path = Path(position_by_tile[adapted_tiles[0]["tile"]]["path"])
        adapted["input_dir"] = str(first_path.parent)
    adapted["adapted_from"] = str(reference_registration_input.resolve())
    adapted["adapted_to_position"] = str(adapted_to_position.resolve())
    adapted["adaptation_method"] = "copy_registered_affine_from_reference_replace_405_stage_from_tile_phase"
    adapted["tile_phase_summary"] = {
        "output_position": tile_phase_summary["output_position"],
        "summary_path": tile_phase_summary.get("summary_path"),
        "tile_count": len(tile_phase_summary["measurements"]),
        "measurements": [
            {
                "tile": item["tile"],
                "reference_tile": item["reference_tile"],
                "final_shift_px_zyx": item["shift_px_zyx"],
                "final_shift_um_zyx": item["shift_um_zyx"],
                "n_inliers": item.get("n_inliers"),
            }
            for item in tile_phase_summary["measurements"]
        ],
    }
    output_registration.parent.mkdir(parents=True, exist_ok=True)
    output_registration.write_text(json.dumps(adapted, indent=2) + "\n")
    return output_registration.resolve()


def make_moving_tile_record(reference_tile: rough_legacy.TileRecord, moving_path: Path) -> rough_legacy.TileRecord:
    moving_shape_zyx, moving_axes = rough_legacy.tile_shape_and_axes(moving_path)
    return rough_legacy.TileRecord(
        tile=moving_path.name,
        side=reference_tile.side,
        path=moving_path,
        translation_zyx_um=reference_tile.translation_zyx_um.copy(),
        scale_zyx_um=reference_tile.scale_zyx_um.copy(),
        shape_zyx=np.asarray(moving_shape_zyx, dtype=np.int64),
        axes=moving_axes,
    )


def shifted_tile_record(tile: rough_legacy.TileRecord, shift_um_zyx: np.ndarray) -> rough_legacy.TileRecord:
    return rough_legacy.TileRecord(
        tile=tile.tile,
        side=tile.side,
        path=tile.path,
        translation_zyx_um=tile.translation_zyx_um + shift_um_zyx,
        scale_zyx_um=tile.scale_zyx_um,
        shape_zyx=tile.shape_zyx,
        axes=tile.axes,
    )


def local_slices_for_world_overlap(
    tile: rough_legacy.TileRecord,
    *,
    overlap_start_um: np.ndarray,
    patch_shape_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    tile_start_um, _tile_stop_um = rough_legacy.tile_bounds_zyx_um(tile)
    starts = np.rint((overlap_start_um - tile_start_um) / np.abs(tile.scale_zyx_um)).astype(int)
    slices = []
    for axis, start in enumerate(starts):
        start = max(0, min(int(start), int(tile.shape_zyx[axis])))
        stop = max(start, min(start + int(patch_shape_zyx[axis]), int(tile.shape_zyx[axis])))
        slices.append(slice(start, stop))
    return tuple(slices)


def overlap_patch_slices(
    fixed_tile: rough_legacy.TileRecord,
    moving_tile: rough_legacy.TileRecord,
    *,
    patch_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | None:
    fixed_start_um, fixed_stop_um = rough_legacy.tile_bounds_zyx_um(fixed_tile)
    moving_start_um, moving_stop_um = rough_legacy.tile_bounds_zyx_um(moving_tile)
    overlap_start_um = np.maximum(fixed_start_um, moving_start_um)
    overlap_stop_um = np.minimum(fixed_stop_um, moving_stop_um)
    overlap_um = overlap_stop_um - overlap_start_um
    if np.any(overlap_um <= 0):
        return None
    fixed_overlap_px = np.floor(overlap_um / np.abs(fixed_tile.scale_zyx_um)).astype(int)
    moving_overlap_px = np.floor(overlap_um / np.abs(moving_tile.scale_zyx_um)).astype(int)
    usable_shape = tuple(
        int(min(patch_shape_zyx[axis], fixed_overlap_px[axis], moving_overlap_px[axis]))
        for axis in range(3)
    )
    if usable_shape[0] < 8 or usable_shape[1] < 64 or usable_shape[2] < 64:
        return None
    patch_um = np.asarray(usable_shape, dtype=np.float64) * np.minimum(
        np.abs(fixed_tile.scale_zyx_um),
        np.abs(moving_tile.scale_zyx_um),
    )
    centered_start_um = overlap_start_um + np.maximum(0.0, (overlap_um - patch_um) / 2.0)
    fixed_slices = local_slices_for_world_overlap(
        fixed_tile,
        overlap_start_um=centered_start_um,
        patch_shape_zyx=usable_shape,
    )
    moving_slices = local_slices_for_world_overlap(
        moving_tile,
        overlap_start_um=centered_start_um,
        patch_shape_zyx=usable_shape,
    )
    shared_shape = tuple(
        min(slice_shape_zyx(fixed_slices)[axis], slice_shape_zyx(moving_slices)[axis])
        for axis in range(3)
    )
    if shared_shape[0] < 8 or shared_shape[1] < 64 or shared_shape[2] < 64:
        return None
    fixed_slices = tuple(slice(fixed_slices[axis].start, fixed_slices[axis].start + shared_shape[axis]) for axis in range(3))
    moving_slices = tuple(slice(moving_slices[axis].start, moving_slices[axis].start + shared_shape[axis]) for axis in range(3))
    return fixed_slices, moving_slices


def infer_shift_from_adjacent_tiles(
    *,
    failed_tile: rough_legacy.TileRecord,
    successful_tiles: list[tuple[rough_legacy.TileRecord, np.ndarray]],
    patch_shape_zyx: tuple[int, int, int],
    min_inliers: int,
    max_neighbors: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    fallback_min_inliers = min_inliers
    failed_center = failed_tile.translation_zyx_um + failed_tile.shape_zyx.astype(np.float64) * failed_tile.scale_zyx_um / 2.0
    neighbor_rows = []
    candidates = sorted(
        [
            (float(np.linalg.norm((tile.translation_zyx_um - failed_tile.translation_zyx_um)[1:3])), tile, shift_um)
            for tile, shift_um in successful_tiles
            if tile.side == failed_tile.side
        ],
        key=lambda item: item[0],
    )[:max_neighbors]
    inferred_shift_rows = []
    for _distance, neighbor_tile, neighbor_shift_um in candidates:
        adjusted_neighbor = shifted_tile_record(neighbor_tile, neighbor_shift_um)
        slices = overlap_patch_slices(adjusted_neighbor, failed_tile, patch_shape_zyx=patch_shape_zyx)
        row = {
            "neighbor_tile": neighbor_tile.tile,
            "neighbor_shift_um_zyx": [float(value) for value in neighbor_shift_um],
        }
        if slices is None:
            row.update(status="rejected", reason="no_usable_adjusted_overlap")
            neighbor_rows.append(row)
            continue
        fixed_slices, moving_slices = slices
        fixed_patch = read_tile_patch(adjusted_neighbor, channel=0, slices_zyx=fixed_slices)
        moving_patch = read_tile_patch(failed_tile, channel=0, slices_zyx=moving_slices)
        if fixed_patch.shape != moving_patch.shape:
            row.update(
                status="rejected",
                reason="patch_shape_mismatch",
                fixed_shape_zyx=[int(value) for value in fixed_patch.shape],
                moving_shape_zyx=[int(value) for value in moving_patch.shape],
            )
            neighbor_rows.append(row)
            continue
        residual_shift_px, details = estimate_patch_shift_zyx_px(fixed_patch, moving_patch)
        inferred_shift_um = residual_shift_px * np.abs(failed_tile.scale_zyx_um)
        inferred_shift_rows.append(inferred_shift_um / np.abs(failed_tile.scale_zyx_um))
        row.update(
            status="accepted",
            reason="measured_adjacent_overlap",
            fixed_slices_zyx=slices_to_json(fixed_slices),
            moving_slices_zyx=slices_to_json(moving_slices),
            inferred_shift_px_zyx=[float(value) for value in inferred_shift_um / np.abs(failed_tile.scale_zyx_um)],
            inferred_shift_um_zyx=[float(value) for value in inferred_shift_um],
            peak=details["peak"],
            corr_before=details["corr_before"],
            corr_after=details["corr_after"],
        )
        neighbor_rows.append(row)
        if len(inferred_shift_rows) >= min_inliers:
            try:
                inlier_mask, final_shift_px = select_inlier_patch_measurements(
                    np.vstack(inferred_shift_rows).astype(np.float64),
                    min_inliers=min_inliers,
                )
            except ValueError:
                continue
            break
    else:
        if len(inferred_shift_rows) < min_inliers:
            raise ValueError(
                f"Only {len(inferred_shift_rows)} adjacent fallback shifts found; require {min_inliers}"
            )
        inlier_mask, final_shift_px = select_inlier_patch_measurements(
            np.vstack(inferred_shift_rows).astype(np.float64),
            min_inliers=min_inliers,
        )

    accepted_index = 0
    for row in neighbor_rows:
        if row.get("status") != "accepted":
            continue
        row["inlier"] = bool(inlier_mask[accepted_index])
        if not row["inlier"]:
            row["reason"] = "outlier_adjacent_shift_cluster"
        accepted_index += 1
        if accepted_index >= len(inlier_mask):
            break

    final_shift_um = final_shift_px * np.abs(failed_tile.scale_zyx_um)
    return final_shift_px, {
        "mode": "adjacent_tile_phase_fallback",
        "min_inliers": int(fallback_min_inliers),
        "failed_tile_center_um_zyx": [float(value) for value in failed_center],
        "n_neighbors_considered": len(candidates),
        "n_measured": len(inferred_shift_rows),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "shift_um_zyx": [float(value) for value in final_shift_um],
        "neighbors": neighbor_rows,
        "corr_before": None,
        "corr_after": None,
    }


def measure_patch_tile_shift(
    *,
    reference_tile: rough_legacy.TileRecord,
    moving_tile: rough_legacy.TileRecord,
    reference_channel: int,
    patch_shape_zyx: tuple[int, int, int],
    coarse_level: int,
    upsample_factor: int,
    max_candidate_patches: int,
    min_inliers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fixed_coarse, fixed_coarse_scale_zyx, fixed_source_level, fixed_available_levels = sampled_tile_volume_from_subifd(
        reference_tile,
        channel=reference_channel,
        requested_level=coarse_level,
    )
    moving_coarse, moving_coarse_scale_zyx, moving_source_level, moving_available_levels = sampled_tile_volume_from_subifd(
        moving_tile,
        channel=0,
        requested_level=coarse_level,
    )
    if not np.array_equal(fixed_coarse_scale_zyx, moving_coarse_scale_zyx):
        raise ValueError(
            "Fixed and moving coarse scout scales differ: "
            f"{fixed_coarse_scale_zyx.tolist()} vs {moving_coarse_scale_zyx.tolist()}"
        )
    coarse_shift_coarse_px, coarse_details = estimate_tile_shift_zyx_px_gpu(fixed_coarse, moving_coarse)
    coarse_shift_l0_px = coarse_shift_coarse_px * fixed_coarse_scale_zyx
    candidates = candidate_patch_slices(
        fixed_coarse,
        tile_shape_zyx=reference_tile.shape_zyx,
        patch_shape_zyx=patch_shape_zyx,
        scout_scale_zyx=fixed_coarse_scale_zyx,
        max_candidates=max_candidate_patches,
        moving_shape_zyx=moving_tile.shape_zyx,
        shift_zyx_px=coarse_shift_l0_px,
    )
    patch_rows = []
    accepted_shift_rows = []
    accepted_patch_indices = []
    final_shift_px: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    early_stop_after_patch: int | None = None
    for patch_index, candidate in enumerate(candidates):
        fixed_slices = candidate["fixed_slices"]
        moving_slices = shifted_slices_zyx(fixed_slices, shift_zyx_px=coarse_shift_l0_px)
        row = {
            "patch_index": int(patch_index),
            "fixed_slices_zyx": slices_to_json(fixed_slices),
            "moving_slices_zyx": slices_to_json(moving_slices),
            "coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
            "content_score": float(candidate["content_score"]),
            "positive_fraction": float(candidate["positive_fraction"]),
        }
        if not slices_within_shape(moving_slices, moving_tile.shape_zyx):
            row.update(status="rejected", reason="moving_patch_out_of_bounds")
            patch_rows.append(row)
            continue
        fixed_patch = read_tile_patch(reference_tile, channel=reference_channel, slices_zyx=fixed_slices)
        moving_patch = read_tile_patch(moving_tile, channel=0, slices_zyx=moving_slices)
        if fixed_patch.shape != patch_shape_zyx or moving_patch.shape != patch_shape_zyx:
            row.update(
                status="rejected",
                reason="patch_shape_mismatch",
                fixed_shape_zyx=[int(value) for value in fixed_patch.shape],
                moving_shape_zyx=[int(value) for value in moving_patch.shape],
            )
            patch_rows.append(row)
            continue
        residual_shift_px, details = estimate_patch_shift_zyx_px(fixed_patch, moving_patch)
        total_shift_px = coarse_shift_l0_px + residual_shift_px
        row.update(
            status="accepted",
            reason="measured",
            residual_shift_px_zyx=[float(value) for value in residual_shift_px],
            total_shift_px_zyx=[float(value) for value in total_shift_px],
            peak=details["peak"],
            corr_before=details["corr_before"],
            corr_after=details["corr_after"],
            fixed_stats={
                "min": float(np.nanmin(fixed_patch)),
                "max": float(np.nanmax(fixed_patch)),
                "mean": float(np.nanmean(fixed_patch)),
                "std": float(np.nanstd(fixed_patch)),
            },
            moving_stats={
                "min": float(np.nanmin(moving_patch)),
                "max": float(np.nanmax(moving_patch)),
                "mean": float(np.nanmean(moving_patch)),
                "std": float(np.nanstd(moving_patch)),
            },
        )
        accepted_patch_indices.append(patch_index)
        accepted_shift_rows.append(total_shift_px)
        patch_rows.append(row)
        if len(accepted_shift_rows) >= min_inliers:
            try:
                inlier_mask, final_shift_px = select_inlier_patch_measurements(
                    np.vstack(accepted_shift_rows).astype(np.float64),
                    min_inliers=min_inliers,
                )
            except ValueError:
                final_shift_px = None
                inlier_mask = None
            else:
                early_stop_after_patch = patch_index
                break

    if early_stop_after_patch is not None:
        for skipped_index, candidate in enumerate(candidates[early_stop_after_patch + 1 :], start=early_stop_after_patch + 1):
            fixed_slices = candidate["fixed_slices"]
            moving_slices = shifted_slices_zyx(fixed_slices, shift_zyx_px=coarse_shift_l0_px)
            patch_rows.append(
                {
                    "patch_index": int(skipped_index),
                    "fixed_slices_zyx": slices_to_json(fixed_slices),
                    "moving_slices_zyx": slices_to_json(moving_slices),
                    "coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
                    "content_score": float(candidate["content_score"]),
                    "positive_fraction": float(candidate["positive_fraction"]),
                    "status": "skipped",
                    "reason": "skipped_after_enough_inliers",
                }
            )

    if len(accepted_shift_rows) < min_inliers:
        raise ValueError(
            f"{reference_tile.tile} produced {len(accepted_shift_rows)} accepted patch shifts; require {min_inliers}"
        )
    if final_shift_px is None or inlier_mask is None:
        total_shifts = np.vstack(accepted_shift_rows).astype(np.float64)
        inlier_mask, final_shift_px = select_inlier_patch_measurements(total_shifts, min_inliers=min_inliers)
    inlier_patch_indices = {
        int(patch_index)
        for patch_index, is_inlier in zip(accepted_patch_indices, inlier_mask, strict=True)
        if bool(is_inlier)
    }
    for row in patch_rows:
        if row.get("status") != "accepted":
            continue
        if int(row["patch_index"]) in inlier_patch_indices:
            row["inlier"] = True
            row["reason"] = "inlier"
        else:
            row["inlier"] = False
            row["reason"] = "outlier_shift_cluster"

    return final_shift_px, {
        "mode": "l0_patch_phase",
        "patch_shape_zyx": [int(value) for value in patch_shape_zyx],
        "coarse_level": int(coarse_level),
        "coarse_scale_zyx": [float(value) for value in fixed_coarse_scale_zyx],
        "fixed_source_level": int(fixed_source_level),
        "moving_source_level": int(moving_source_level),
        "fixed_available_levels": int(fixed_available_levels),
        "moving_available_levels": int(moving_available_levels),
        "coarse_shift_level_px_zyx": [float(value) for value in coarse_shift_coarse_px],
        "coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
        "corr_before": coarse_details["corr_before"],
        "corr_after": coarse_details["corr_after"],
        "coarse_corr_before": coarse_details["corr_before"],
        "coarse_corr_after": coarse_details["corr_after"],
        "n_candidates": len(candidates),
        "n_measured": len(accepted_shift_rows),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "early_stop_after_patch": early_stop_after_patch,
        "patches": patch_rows,
    }


def align_tiles_to_reference(
    *,
    reference_position: Path,
    output_position: Path,
    output_dir: Path,
    reference_channel: int = 3,
    reference_token: str = "488514561638",
    moving_token: str = "405",
    level: int = 4,
    upsample_factor: int = 10,
    patch_shape_zyx: tuple[int, int, int] | None = None,
    min_inliers: int = 2,
    max_candidate_patches: int = 24,
    coarse_level: int = 4,
    reference_registration_input: Path | None = None,
    output_registration: Path | None = None,
) -> Path:
    if level < 0:
        raise ValueError("level must be non-negative")
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")
    if min_inliers < 1:
        raise ValueError("min_inliers must be >= 1")
    if max_candidate_patches < 1:
        raise ValueError("max_candidate_patches must be >= 1")
    if patch_shape_zyx is None and output_registration is not None:
        raise ValueError("--output-registration requires --patch-shape-zyx")
    if output_registration is not None and reference_registration_input is None:
        raise ValueError("--output-registration requires --reference-registration-input")
    payload = json.loads(reference_position.read_text())
    reference_tiles = rough_legacy.load_tiles(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "tile_phase_measurement_cache.json"
    cache_key = tile_phase_cache_key(
        reference_position=reference_position,
        reference_channel=reference_channel,
        reference_token=reference_token,
        moving_token=moving_token,
        level=level,
        upsample_factor=upsample_factor,
        patch_shape_zyx=patch_shape_zyx,
        min_inliers=min_inliers,
        max_candidate_patches=max_candidate_patches,
        coarse_level=coarse_level,
    )
    cached_rows_by_tile = load_tile_phase_cache(cache_path, cache_key)
    level_factor = 2**level
    rows = []
    updated = json.loads(json.dumps(payload))
    updated["source"] = f"{payload.get('source', 'position file')} + tile phase alignment {moving_token} to {reference_token}"
    updated["derived_by"] = "lightsheet.tile_phase.v1"
    diagnostics = updated.setdefault("diagnostics", {})
    diagnostics["tile_phase_alignment"] = {
        "reference_position": str(reference_position.resolve()),
        "reference_channel": int(reference_channel),
        "reference_token": reference_token,
        "moving_token": moving_token,
        "level": int(level),
        "level_factor": int(level_factor),
        "upsample_factor": int(upsample_factor),
        "patch_shape_zyx": None if patch_shape_zyx is None else [int(value) for value in patch_shape_zyx],
        "min_inliers": int(min_inliers),
        "max_candidate_patches": int(max_candidate_patches),
        "coarse_level": int(coarse_level),
        "description": "Per-tile moving z/y/x translations shifted after 3D phase correlation to reference channel tiles.",
    }
    failed_tiles = []
    successful_moving_tiles: list[tuple[rough_legacy.TileRecord, np.ndarray]] = []

    for record, reference_tile in zip(updated["tiles"], reference_tiles, strict=True):
        moving_path = corresponding_moving_path(reference_tile.path, reference_token=reference_token, moving_token=moving_token)
        moving_tile = make_moving_tile_record(reference_tile, moving_path)
        moving_tile_name = make_moving_tile_name(record["tile"], reference_token=reference_token, moving_token=moving_token)
        cached_row = cached_rows_by_tile.get(moving_tile_name)
        if cached_row is not None:
            shift_um = apply_shift_row_to_position_record(record, cached_row, moving_path=moving_path)
            successful_moving_tiles.append((moving_tile, shift_um))
            rows.append(cached_row)
            print(f"tile-phase-cache {reference_tile.tile} -> {moving_path.name}", flush=True)
            continue
        try:
            if patch_shape_zyx is None:
                fixed = rough_legacy.sampled_tile_volume(
                    reference_tile,
                    channel=reference_channel,
                    level_factor=level_factor,
                )
                moving = rough_legacy.sampled_tile_volume(moving_tile, channel=0, level_factor=level_factor)
                shift_px, details = estimate_tile_shift_zyx_px(fixed, moving, upsample_factor=upsample_factor)
                shift_um = shift_px * np.abs(reference_tile.scale_zyx_um) * level_factor
            else:
                shift_px, details = measure_patch_tile_shift(
                    reference_tile=reference_tile,
                    moving_tile=moving_tile,
                    reference_channel=reference_channel,
                    patch_shape_zyx=patch_shape_zyx,
                    coarse_level=coarse_level,
                    upsample_factor=upsample_factor,
                    max_candidate_patches=max_candidate_patches,
                    min_inliers=min_inliers,
                )
                shift_um = shift_px * np.abs(reference_tile.scale_zyx_um)
        except Exception as exc:
            failed_tiles.append(
                {
                    "record": record,
                    "reference_tile": reference_tile,
                    "moving_tile": moving_tile,
                    "moving_path": moving_path,
                    "error": str(exc),
                }
            )
            continue
        record["tile"] = moving_tile_name
        record["path"] = str(moving_path)
        for axis, value in zip(DIMENSIONS, shift_um, strict=True):
            record["translation_um"][axis] = float(record["translation_um"][axis] + value)
        successful_moving_tiles.append((moving_tile, shift_um))
        row = {
            "tile": record["tile"],
            "side": record.get("side"),
            "reference_tile": reference_tile.tile,
            "reference_path": str(reference_tile.path),
            "moving_path": str(moving_path),
            "shift_px_zyx": [float(value) for value in shift_px],
            "shift_um_zyx": [float(value) for value in shift_um],
            "corr_before": details["corr_before"],
            "corr_after": details["corr_after"],
            "phase_error": details.get("phase_error"),
            "shape_zyx": details.get("shape_zyx"),
            "n_inliers": details.get("n_inliers"),
            "patch_details": details if patch_shape_zyx is not None else None,
        }
        rows.append(row)
        write_tile_phase_cache(cache_path, cache_key, rows)
        print(
            f"tile-phase {reference_tile.tile} -> {moving_path.name} "
            f"shift_px={shift_px.tolist()} corr_after={details['corr_after']}",
            flush=True,
        )

    remaining_failed_tiles = []
    for failure in failed_tiles:
        if patch_shape_zyx is None:
            remaining_failed_tiles.append(failure)
            continue
        record = failure["record"]
        reference_tile = failure["reference_tile"]
        moving_tile = failure["moving_tile"]
        moving_path = failure["moving_path"]
        try:
            shift_px, details = infer_shift_from_adjacent_tiles(
                failed_tile=moving_tile,
                successful_tiles=successful_moving_tiles,
                patch_shape_zyx=patch_shape_zyx,
                min_inliers=1,
            )
        except Exception as exc:
            failure["fallback_error"] = str(exc)
            remaining_failed_tiles.append(failure)
            continue
        shift_um = shift_px * np.abs(reference_tile.scale_zyx_um)
        record["tile"] = make_moving_tile_name(record["tile"], reference_token=reference_token, moving_token=moving_token)
        record["path"] = str(moving_path)
        for axis, value in zip(DIMENSIONS, shift_um, strict=True):
            record["translation_um"][axis] = float(record["translation_um"][axis] + value)
        successful_moving_tiles.append((moving_tile, shift_um))
        row = {
            "tile": record["tile"],
            "side": record.get("side"),
            "reference_tile": reference_tile.tile,
            "reference_path": str(reference_tile.path),
            "moving_path": str(moving_path),
            "shift_px_zyx": [float(value) for value in shift_px],
            "shift_um_zyx": [float(value) for value in shift_um],
            "corr_before": details["corr_before"],
            "corr_after": details["corr_after"],
            "phase_error": None,
            "shape_zyx": None,
            "n_inliers": details.get("n_inliers"),
            "patch_details": details,
            "direct_error": failure["error"],
            "fallback": True,
        }
        rows.append(row)
        write_tile_phase_cache(cache_path, cache_key, rows)
        print(
            f"tile-phase-fallback {reference_tile.tile} -> {moving_path.name} "
            f"shift_px={shift_px.tolist()} n_inliers={details.get('n_inliers')}",
            flush=True,
        )

    if remaining_failed_tiles:
        details = "; ".join(
            f"{item['reference_tile'].tile}: direct={item['error']} fallback={item.get('fallback_error')}"
            for item in remaining_failed_tiles
        )
        raise RuntimeError(f"Tile phase alignment failed for {len(remaining_failed_tiles)} tile(s): {details}")

    diagnostics["tile_phase_alignment"]["measurements"] = rows
    updated = stamp_artifact(updated, "lightsheet.position.v1")
    output_position.parent.mkdir(parents=True, exist_ok=True)
    output_position.write_text(json.dumps(updated, indent=2) + "\n")
    summary = {
        "schema_version": 1,
        "artifact_type": "lightsheet.tile_phase_summary.v1",
        "reference_position": str(reference_position.resolve()),
        "output_position": str(output_position.resolve()),
        "reference_channel": int(reference_channel),
        "reference_token": reference_token,
        "moving_token": moving_token,
        "level": int(level),
        "level_factor": int(level_factor),
        "patch_shape_zyx": None if patch_shape_zyx is None else [int(value) for value in patch_shape_zyx],
        "min_inliers": int(min_inliers),
        "max_candidate_patches": int(max_candidate_patches),
        "coarse_level": int(coarse_level),
        "measurements": rows,
    }
    summary_path = output_dir / "tile_phase_alignment.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if output_registration is not None and reference_registration_input is not None:
        adapt_registration_from_reference(
            reference_registration_input=reference_registration_input,
            output_registration=output_registration,
            adapted_position_payload=updated,
            reference_token=reference_token,
            moving_token=moving_token,
            adapted_to_position=output_position,
            tile_phase_summary=summary,
        )
    return output_position.resolve()
