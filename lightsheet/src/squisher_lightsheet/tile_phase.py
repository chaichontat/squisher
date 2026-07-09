from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict, cast

from loguru import logger
import numpy as np

from squisher_lightsheet.artifacts import stamp_artifact
from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as rough_legacy
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as stitch_legacy
from squisher_lightsheet import phase_metrics
from squisher_lightsheet.tiff import tiff_series_level_count


DIMENSIONS = ("z", "y", "x")
PATCH_INLIER_THRESHOLDS_ZYX = np.asarray([3.0, 12.0, 12.0], dtype=np.float64)
PATCH_MIN_CORR_AFTER = 0.15
PATCH_MIN_CORR_IMPROVEMENT = 0.0
PATCH_MAX_RESIDUAL_FRACTION = 0.45
PATCH_MIN_QUALITY_ACCEPTED_FOR_EARLY_STOP = 3
PATCH_MIN_GRADIENT_COMPONENT_NCC_AFTER_FOR_ZERO_RESIDUAL = 0.15
PATCH_ZERO_RESIDUAL_NORM_PX = 0.5
TILE_PHASE_MEASUREMENT_STATUSES = ("direct_accepted", "direct_failed", "fallback_accepted")


class TilePhaseMeasurementError(ValueError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details


MeasurementStatus = Literal["direct_accepted", "direct_failed", "fallback_accepted"]


class TilePhaseBaseRow(TypedDict):
    tile: str
    measurement_status: MeasurementStatus


class TilePhaseAcceptedRow(TilePhaseBaseRow, total=False):
    side: Any
    reference_tile: str
    reference_path: str
    moving_path: str
    shift_px_zyx: list[float]
    shift_um_zyx: list[float]
    corr_before: Any
    corr_after: Any
    phase_error: Any
    shape_zyx: Any
    n_inliers: Any
    patch_details: dict[str, Any] | None
    cache_source: str
    fallback: bool
    direct_error: str


class TilePhaseFailedAttemptRow(TilePhaseBaseRow, total=False):
    side: Any
    reference_tile: str
    reference_path: str
    moving_path: str
    direct_error: str
    patch_details: dict[str, Any] | None
    corr_before: Any
    corr_after: Any
    n_inliers: Any
    fallback_error: str


TilePhaseRow = TilePhaseAcceptedRow | TilePhaseFailedAttemptRow


def deconvolution_sidecar(path: Path) -> Path:
    name = path.name
    if name.endswith(".ome.tif"):
        return path.with_name(name.removesuffix(".ome.tif") + ".ome.deconv.json")
    if name.endswith(".ome.tiff"):
        return path.with_name(name.removesuffix(".ome.tiff") + ".ome.deconv.json")
    return path.with_suffix(path.suffix + ".deconv.json")


def flattened_channel_count(path: Path) -> int:
    sidecar = deconvolution_sidecar(path)
    if not sidecar.exists():
        return 1
    payload = json.loads(sidecar.read_text())
    channels = int(payload.get("provenance", {}).get("channels", 1))
    if channels < 1:
        raise ValueError(f"{sidecar} records invalid channel count {channels}")
    return channels


def _open_tile_level_array(path: Path, *, source_level: int = 0) -> tuple[Any, int, int, Any | None]:
    import dask.array as da
    import tifffile
    import zarr

    if stitch_legacy.is_ome_zarr_path(path):
        available_levels = stitch_legacy._ome_zarr_level_count(path)
        resolved_level = min(int(source_level), max(0, available_levels - 1))
        zarray = stitch_legacy._open_ome_zarr_level_array(path, source_level=resolved_level)
        return da.from_zarr(zarray), resolved_level, available_levels, None

    available_levels = tiff_series_level_count(path)
    resolved_level = min(int(source_level), max(0, available_levels - 1))
    store = tifffile.imread(path, aszarr=True, level=resolved_level)
    zarray = rough_legacy.base_zarr_array(zarr.open(store, mode="r"))
    return da.from_zarr(zarray), resolved_level, available_levels, store


def logical_tile_record(tile: rough_legacy.TileRecord) -> rough_legacy.TileRecord:
    if tile.axes != "ZYX":
        return tile
    channels = flattened_channel_count(tile.path)
    if channels == 1:
        return tile
    if int(tile.shape_zyx[0]) % channels:
        raise ValueError(f"{tile.path} has {tile.shape_zyx[0]} planes, not divisible by channels={channels}")
    shape = tile.shape_zyx.copy()
    shape[0] = int(shape[0]) // channels
    return rough_legacy.TileRecord(
        tile=tile.tile,
        side=tile.side,
        path=tile.path,
        translation_zyx_um=tile.translation_zyx_um,
        scale_zyx_um=tile.scale_zyx_um,
        shape_zyx=shape,
        axes=tile.axes,
    )


def tile_record_from_position_record(record: Mapping[str, Any]) -> rough_legacy.TileRecord:
    path = Path(str(record["path"]))
    axes = str(record.get("axes", ""))
    shape = record.get("shape")
    if shape is None or not axes:
        shape_zyx, axes = rough_legacy.tile_shape_and_axes(path)
    else:
        raw_shape = tuple(int(value) for value in shape)
        if axes == "CZYX" and len(raw_shape) == 4:
            shape_zyx = np.asarray(raw_shape[1:4], dtype=np.int64)
        elif axes == "ZYX" and len(raw_shape) == 3:
            shape_zyx = np.asarray(raw_shape, dtype=np.int64)
        elif len(raw_shape) == 3:
            shape_zyx = np.asarray(raw_shape, dtype=np.int64)
        else:
            raise ValueError(f"{path} has unsupported position shape/axes: shape={raw_shape!r}, axes={axes!r}")
    side = record.get("side")
    if side not in rough_legacy.SIDES:
        raise ValueError(f"{path} has side={side!r}; expected one of {rough_legacy.SIDES}")
    return logical_tile_record(
        rough_legacy.TileRecord(
            tile=str(record["tile"]),
            side=str(side),
            path=path,
            translation_zyx_um=np.asarray([record["translation_um"][dim] for dim in DIMENSIONS], dtype=np.float64),
            scale_zyx_um=np.asarray([record["scale_um"][dim] for dim in DIMENSIONS], dtype=np.float64),
            shape_zyx=shape_zyx,
            axes=axes,
        )
    )


def normalize_volume_for_phase(volume: np.ndarray) -> np.ndarray:
    return phase_metrics.robust_normalize(volume)


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
    return phase_metrics.corrcoef_on_mask(fixed, moving, mask)


def shifted_overlap_mask(shape: tuple[int, ...], shift: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    valid = np.ones(shape, dtype=np.float32)
    shifted_valid = ndimage.shift(valid, shift=shift, order=0, mode="constant", cval=0.0, prefilter=False)
    return shifted_valid > 0.5


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


def shifted_slices_with_realized_shift(
    fixed_slices: tuple[slice, slice, slice],
    *,
    requested_shift_zyx_px: np.ndarray,
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    moving_slices = shifted_slices_zyx(fixed_slices, shift_zyx_px=requested_shift_zyx_px)
    realized_shift = np.asarray(
        [fixed.start - moving.start for fixed, moving in zip(fixed_slices, moving_slices, strict=True)],
        dtype=np.float64,
    )
    return moving_slices, realized_shift


def slices_within_shape(slices_zyx: tuple[slice, slice, slice], shape_zyx: np.ndarray) -> bool:
    return all(0 <= slc.start < slc.stop <= int(size) for slc, size in zip(slices_zyx, shape_zyx, strict=True))


def finite_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def patch_quality_rejection_reasons(
    *,
    details: dict[str, Any],
    residual_shift_px: np.ndarray,
    patch_shape_zyx: tuple[int, int, int],
) -> list[str]:
    corr_after = finite_float(details.get("corr_after"))
    corr_before = finite_float(details.get("corr_before"))
    reasons = []
    if corr_after < PATCH_MIN_CORR_AFTER:
        reasons.append("corr_after_below_threshold")

    gradient_before = finite_float(details.get("gradient_component_ncc_before"))
    gradient_after = finite_float(details.get("gradient_component_ncc_after"))
    if np.isfinite(gradient_before) and np.isfinite(gradient_after):
        if (
            gradient_after < PATCH_MIN_GRADIENT_COMPONENT_NCC_AFTER_FOR_ZERO_RESIDUAL
            and float(np.linalg.norm(residual_shift_px)) < PATCH_ZERO_RESIDUAL_NORM_PX
        ):
            reasons.append("weak_gradient_component_ncc_zero_residual")
        if gradient_after - gradient_before < PATCH_MIN_CORR_IMPROVEMENT - 1e-6:
            reasons.append("gradient_component_ncc_improvement_below_threshold")
    elif corr_after - corr_before < PATCH_MIN_CORR_IMPROVEMENT:
        reasons.append("corr_improvement_below_threshold")

    if np.any(np.abs(residual_shift_px) >= PATCH_MAX_RESIDUAL_FRACTION * np.asarray(patch_shape_zyx, dtype=np.float64)):
        reasons.append("residual_near_periodic_wrap")
    return reasons


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
    if not slices_within_shape(slices_zyx, tile.shape_zyx):
        raise ValueError(f"Patch slices {slices_zyx} are outside tile shape {tile.shape_zyx.tolist()}")

    array, _source_level, _available_levels, store = _open_tile_level_array(tile.path, source_level=0)
    try:
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
            channels = flattened_channel_count(tile.path)
            if channel < 0 or channel >= channels:
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {channels}")
            if channels == 1:
                patch = array[tuple(raw_slices)]
            else:
                z_slice = raw_slices[0]
                z_indices = np.arange(int(z_slice.start), int(z_slice.stop), dtype=np.int64)
                patch = array[(z_indices * channels + channel, *raw_slices[1:])]
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


def _slice_from_json(value: list[int]) -> slice:
    if len(value) != 2:
        raise ValueError(f"Expected [start, stop] slice record, got {value!r}")
    return slice(int(value[0]), int(value[1]))


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).removesuffix(".ome.tif")


def _robust_uint8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = float(np.max(finite))
        low = float(np.min(finite))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _center_plane_uint8(volume: np.ndarray) -> np.ndarray:
    center_z = int(volume.shape[0] // 2)
    return _robust_uint8(np.asarray(volume[center_z], dtype=np.float32))


def _rgb_overlay(fixed_plane: np.ndarray, moving_plane: np.ndarray) -> np.ndarray:
    red = _robust_uint8(moving_plane)
    green = _robust_uint8(fixed_plane)
    blue = np.zeros_like(red)
    return np.stack([red, green, blue], axis=-1)


def _slice_axis_from_json(value: list[int], *, axis_size: int, axis_name: str) -> slice:
    slc = _slice_from_json(value)
    if not 0 <= slc.start < slc.stop <= int(axis_size):
        raise ValueError(f"{axis_name} slice {value!r} is outside axis size {int(axis_size)}")
    return slc


def _z_indices_from_patch(patch: Mapping[str, Any], key: str, *, fallback_slice: slice, axis_size: int) -> np.ndarray:
    raw_indices = patch.get(key)
    if raw_indices is None:
        if not 0 <= fallback_slice.start < fallback_slice.stop <= int(axis_size):
            raise ValueError(f"Patch z slice {fallback_slice} is outside axis size {int(axis_size)}")
        return np.arange(fallback_slice.start, fallback_slice.stop, dtype=np.int64)
    indices = np.asarray(raw_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{key} must be a non-empty 1D z-index list")
    if np.any(indices < 0) or np.any(indices >= int(axis_size)):
        raise ValueError(f"{key} contains indices outside axis size {int(axis_size)}")
    return indices


def raw_axis_indices_for_oriented_indices(indices: np.ndarray, *, axis_size: int, flipped: bool) -> np.ndarray:
    if np.any(indices < 0) or np.any(indices >= int(axis_size)):
        raise ValueError(f"Indices are outside axis size {axis_size}")
    if not flipped:
        return indices.astype(np.int64, copy=False)
    return (int(axis_size) - 1 - indices).astype(np.int64, copy=False)


def read_tile_indexed_z_patch(
    tile: rough_legacy.TileRecord,
    *,
    channel: int,
    z_indices: np.ndarray,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    z_indices = np.asarray(z_indices, dtype=np.int64)
    if z_indices.ndim != 1 or z_indices.size == 0:
        raise ValueError("z_indices must be a non-empty 1D array")
    if np.any(z_indices < 0) or np.any(z_indices >= int(tile.shape_zyx[0])):
        raise ValueError(f"z_indices are outside tile z shape {int(tile.shape_zyx[0])}")
    if not slices_within_shape((slice(0, 1), y_slice, x_slice), np.asarray([1, tile.shape_zyx[1], tile.shape_zyx[2]])):
        raise ValueError(f"Y/X patch slices {(y_slice, x_slice)} are outside tile shape {tile.shape_zyx.tolist()}")

    array, _source_level, _available_levels, store = _open_tile_level_array(tile.path, source_level=0)
    try:
        raw_z_indices = raw_axis_indices_for_oriented_indices(
            z_indices,
            axis_size=int(tile.shape_zyx[0]),
            flipped=bool(tile.scale_zyx_um[0] < 0),
        )
        raw_y_slice, reverse_y = raw_axis_slice_for_oriented_slice(
            y_slice,
            axis_size=int(tile.shape_zyx[1]),
            flipped=bool(tile.scale_zyx_um[1] < 0),
        )
        raw_x_slice, reverse_x = raw_axis_slice_for_oriented_slice(
            x_slice,
            axis_size=int(tile.shape_zyx[2]),
            flipped=bool(tile.scale_zyx_um[2] < 0),
        )
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            patch = array[(channel, raw_z_indices, raw_y_slice, raw_x_slice)]
        elif tile.axes == "ZYX":
            channels = flattened_channel_count(tile.path)
            if channel < 0 or channel >= channels:
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {channels}")
            if channels == 1:
                patch = array[(raw_z_indices, raw_y_slice, raw_x_slice)]
            else:
                patch = array[(raw_z_indices * channels + channel, raw_y_slice, raw_x_slice)]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        result = np.asarray(patch.compute(), dtype=np.float32)
        if reverse_y:
            result = np.flip(result, axis=1)
        if reverse_x:
            result = np.flip(result, axis=2)
        return result
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _display_residual_shift(patch: Mapping[str, Any], residual_shift: np.ndarray) -> np.ndarray:
    if patch.get("patch_source") != "sparse_subifd_scout":
        return residual_shift
    scale = np.asarray(patch.get("coarse_scale_zyx", [1.0, 1.0, 1.0]), dtype=np.float32)
    display_shift = residual_shift.astype(np.float32, copy=True)
    if scale.shape == (3,) and float(scale[0]) > 0:
        display_shift[0] = display_shift[0] / scale[0]
    return display_shift


def write_tile_phase_contact_sheets(
    *,
    rows: list[TilePhaseRow],
    reference_tiles: list[rough_legacy.TileRecord],
    output_dir: Path,
    reference_channel: int,
    moving_channel: int,
    max_panel_size: int = 192,
) -> list[Path]:
    """Render per-tile QC sheets from recorded tile-phase patch measurements."""
    from PIL import Image, ImageDraw
    from scipy import ndimage

    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    reference_tiles_by_name = {tile.tile: tile for tile in reference_tiles}
    for row in rows:
        patch_details = row.get("patch_details")
        if not isinstance(patch_details, dict):
            continue
        patches = [
            patch
            for patch in patch_details.get("patches", [])
            if patch.get("fixed_slices_zyx")
            and patch.get("moving_slices_zyx")
            and patch.get("residual_shift_px_zyx") is not None
        ]
        if not patches:
            continue
        reference_path = row.get("reference_path")
        moving_path = row.get("moving_path")
        if not isinstance(reference_path, str) or not isinstance(moving_path, str):
            continue
        reference_name = row.get("reference_tile")
        reference_tile = reference_tiles_by_name.get(str(reference_name))
        if reference_tile is None:
            continue
        moving_tile = make_moving_tile_record(reference_tile, Path(moving_path))

        row_images: list[Image.Image] = []
        for patch in patches:
            fixed_slices = (
                _slice_axis_from_json(patch["fixed_slices_zyx"][0], axis_size=int(reference_tile.shape_zyx[0]), axis_name="fixed z"),
                _slice_axis_from_json(patch["fixed_slices_zyx"][1], axis_size=int(reference_tile.shape_zyx[1]), axis_name="fixed y"),
                _slice_axis_from_json(patch["fixed_slices_zyx"][2], axis_size=int(reference_tile.shape_zyx[2]), axis_name="fixed x"),
            )
            moving_slices = (
                _slice_axis_from_json(patch["moving_slices_zyx"][0], axis_size=int(moving_tile.shape_zyx[0]), axis_name="moving z"),
                _slice_axis_from_json(patch["moving_slices_zyx"][1], axis_size=int(moving_tile.shape_zyx[1]), axis_name="moving y"),
                _slice_axis_from_json(patch["moving_slices_zyx"][2], axis_size=int(moving_tile.shape_zyx[2]), axis_name="moving x"),
            )
            residual_shift = np.asarray(patch["residual_shift_px_zyx"], dtype=np.float32)
            fixed_z_indices = _z_indices_from_patch(
                patch,
                "fixed_z_indices_l0",
                fallback_slice=fixed_slices[0],
                axis_size=int(reference_tile.shape_zyx[0]),
            )
            moving_z_indices = _z_indices_from_patch(
                patch,
                "moving_z_indices_l0",
                fallback_slice=moving_slices[0],
                axis_size=int(moving_tile.shape_zyx[0]),
            )
            fixed_patch = read_tile_indexed_z_patch(
                reference_tile,
                channel=reference_channel,
                z_indices=fixed_z_indices,
                y_slice=fixed_slices[1],
                x_slice=fixed_slices[2],
            )
            moving_patch = read_tile_indexed_z_patch(
                moving_tile,
                channel=moving_channel,
                z_indices=moving_z_indices,
                y_slice=moving_slices[1],
                x_slice=moving_slices[2],
            )
            display_shift = _display_residual_shift(patch, residual_shift)
            shifted_moving = ndimage.shift(
                moving_patch,
                shift=display_shift,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            fixed_plane = np.asarray(fixed_patch[fixed_patch.shape[0] // 2], dtype=np.float32)
            shifted_plane = np.asarray(shifted_moving[shifted_moving.shape[0] // 2], dtype=np.float32)
            panels = [
                Image.fromarray(_center_plane_uint8(fixed_patch)).convert("RGB"),
                Image.fromarray(_center_plane_uint8(moving_patch)).convert("RGB"),
                Image.fromarray(_center_plane_uint8(shifted_moving)).convert("RGB"),
                Image.fromarray(_rgb_overlay(fixed_plane, shifted_plane)).convert("RGB"),
            ]
            if max_panel_size > 0:
                panels = [
                    panel.resize(
                        (
                            max(1, int(round(panel.width * min(max_panel_size / panel.width, max_panel_size / panel.height, 1.0)))),
                            max(1, int(round(panel.height * min(max_panel_size / panel.width, max_panel_size / panel.height, 1.0)))),
                        ),
                        Image.Resampling.BILINEAR,
                    )
                    for panel in panels
                ]
            label_height = 40
            gap = 6
            width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
            height = max(panel.height for panel in panels) + label_height
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            status = patch.get("reason") or patch.get("status") or "patch"
            label = (
                f"p{int(patch.get('patch_index', -1)):03d} {status} "
                f"corr={patch.get('corr_after')} shift={np.round(residual_shift, 2).tolist()}"
            )
            draw.text((4, 4), label, fill=(0, 0, 0))
            draw.text(
                (4, 20),
                f"fixed ch{reference_channel} | moving ch{moving_channel} seed | moving shifted | overlay moving red / fixed green",
                fill=(0, 0, 0),
            )
            x = 0
            for panel in panels:
                canvas.paste(panel, (x, label_height))
                x += panel.width + gap
            row_images.append(canvas)

        if not row_images:
            continue
        gap = 8
        width = max(image.width for image in row_images)
        height = sum(image.height for image in row_images) + gap * (len(row_images) - 1)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for image in row_images:
            sheet.paste(image, (0, y))
            y += image.height + gap
        output_path = contact_dir / f"{_safe_stem(str(row['tile']))}_cross_channel_patch_contact_sheet.png"
        sheet.save(output_path)
        written.append(output_path)
    index_path = contact_dir / "contact_sheets.json"
    index_path.write_text(json.dumps([str(path.resolve()) for path in written], indent=2) + "\n")
    return written


def sampled_tile_volume_from_subifd(
    tile: rough_legacy.TileRecord,
    *,
    channel: int,
    requested_level: int,
    z_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    if requested_level < 0:
        raise ValueError("requested_level must be non-negative")
    desired_factor = 2**int(requested_level)
    array, source_level, available_levels, store = _open_tile_level_array(tile.path, source_level=int(requested_level))
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
        raw_z_indices: np.ndarray | None = None
        if z_samples is not None:
            if z_samples < 1:
                raise ValueError("z_samples must be >= 1")
            sampled_z_count = int(np.ceil(int(source_shape_zyx[0]) / abs(z_step)))
            if z_samples < sampled_z_count:
                raw_z_indices = np.unique(
                    np.rint(np.linspace(0, int(source_shape_zyx[0]) - 1, z_samples)).astype(np.int64)
                )
                if tile.scale_zyx_um[0] < 0:
                    raw_z_indices = raw_z_indices[::-1]
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            z_selection = slice(None, None, z_step) if raw_z_indices is None else raw_z_indices
            volume = array[channel, z_selection, ::y_step, ::x_step]
        elif tile.axes == "ZYX":
            channels = flattened_channel_count(tile.path)
            if channel < 0 or channel >= channels:
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {channels}")
            if channels == 1:
                z_selection = slice(None, None, z_step) if raw_z_indices is None else raw_z_indices
                volume = array[z_selection, ::y_step, ::x_step]
            else:
                if raw_z_indices is not None:
                    z_indices = raw_z_indices
                elif z_step > 0:
                    z_indices = np.arange(0, int(source_shape_zyx[0]), z_step, dtype=np.int64)
                else:
                    z_indices = np.arange(int(source_shape_zyx[0]) - 1, -1, z_step, dtype=np.int64)
                volume = array[z_indices * channels + channel, ::y_step, ::x_step]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        effective_factor_zyx = source_factor_zyx * remaining_step_zyx
        if raw_z_indices is None:
            if z_step > 0:
                selected_raw_z_indices = np.arange(0, int(source_shape_zyx[0]), z_step, dtype=np.int64)
            else:
                selected_raw_z_indices = np.arange(int(source_shape_zyx[0]) - 1, -1, z_step, dtype=np.int64)
        else:
            selected_raw_z_indices = raw_z_indices
        if tile.scale_zyx_um[0] < 0:
            selected_logical_source_z = int(source_shape_zyx[0]) - 1 - selected_raw_z_indices
        else:
            selected_logical_source_z = selected_raw_z_indices
        z_indices_l0 = np.rint(selected_logical_source_z.astype(np.float64) * float(source_factor_zyx[0])).astype(np.int64)
        z_indices_l0 = np.clip(z_indices_l0, 0, int(tile.shape_zyx[0]) - 1)
        if raw_z_indices is not None and len(raw_z_indices) > 1:
            z_span = (int(source_shape_zyx[0]) - 1) * float(source_factor_zyx[0])
            effective_factor_zyx = effective_factor_zyx.astype(np.float64)
            effective_factor_zyx[0] = z_span / float(len(raw_z_indices) - 1)
        return (
            np.asarray(volume.compute(), dtype=np.float32),
            effective_factor_zyx.astype(np.float64),
            source_level,
            available_levels,
            z_indices_l0,
        )
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
    moving_shape_zyx: np.ndarray,
    shift_zyx_px: np.ndarray,
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


def candidate_sparse_scout_chunk_slices(
    scout_volume: np.ndarray,
    *,
    patch_shape_zyx: tuple[int, int, int],
    scout_scale_zyx: np.ndarray,
    max_candidates: int,
    moving_shape_zyx: np.ndarray,
    shift_zyx_px: np.ndarray,
) -> list[dict[str, Any]]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    scout_shape = np.asarray(scout_volume.shape, dtype=np.int64)
    patch_shape_coarse = np.maximum(
        1,
        np.ceil(np.asarray(patch_shape_zyx, dtype=np.float64) / scout_scale_zyx.astype(np.float64)).astype(np.int64),
    )
    patch_shape_coarse[0] = int(scout_shape[0])
    patch_shape_coarse = np.minimum(patch_shape_coarse, scout_shape)
    starts_by_axis: list[list[int]] = [[0]]
    for axis in (1, 2):
        size = int(scout_shape[axis])
        patch_size = int(patch_shape_coarse[axis])
        if size == patch_size:
            starts_by_axis.append([0])
            continue
        step = max(1, patch_size // 2)
        starts = list(range(0, size - patch_size + 1, step))
        if starts[-1] != size - patch_size:
            starts.append(size - patch_size)
        starts_by_axis.append(starts)

    ranked = []
    scout_norm = normalize_volume_for_phase(scout_volume)
    for z0 in starts_by_axis[0]:
        for y0 in starts_by_axis[1]:
            for x0 in starts_by_axis[2]:
                fixed_slices = (
                    slice(z0, z0 + int(patch_shape_coarse[0])),
                    slice(y0, y0 + int(patch_shape_coarse[1])),
                    slice(x0, x0 + int(patch_shape_coarse[2])),
                )
                moving_slices = shifted_slices_zyx(fixed_slices, shift_zyx_px=shift_zyx_px)
                if not slices_within_shape(moving_slices, moving_shape_zyx):
                    continue
                patch = scout_norm[fixed_slices]
                finite = patch[np.isfinite(patch)]
                if finite.size == 0:
                    continue
                positive_fraction = float(np.count_nonzero(finite > 0.0) / finite.size)
                score = float(np.nanstd(finite)) * max(positive_fraction, 1e-6)
                ranked.append(
                    {
                        "fixed_slices": fixed_slices,
                        "content_score": score,
                        "positive_fraction": positive_fraction,
                    }
                )
    ranked.sort(key=lambda item: item["content_score"], reverse=True)
    return ranked[:max_candidates]


def scale_slices_to_l0_json(slices_zyx: tuple[slice, slice, slice], scale_zyx: np.ndarray) -> list[list[int]]:
    return [
        [int(np.floor(slc.start * scale)), int(np.ceil(slc.stop * scale))]
        for slc, scale in zip(slices_zyx, scale_zyx, strict=True)
    ]


def sparse_scout_slices_to_l0_json(
    slices_zyx: tuple[slice, slice, slice],
    *,
    scale_zyx: np.ndarray,
    z_indices_l0: np.ndarray,
) -> list[list[int]]:
    patch_z = np.asarray(z_indices_l0[slices_zyx[0]], dtype=np.int64)
    if patch_z.size == 0:
        raise ValueError(f"Sparse scout z slice {slices_zyx[0]} selected no z planes")
    return [
        [int(np.min(patch_z)), int(np.max(patch_z)) + 1],
        [int(np.floor(slices_zyx[1].start * scale_zyx[1])), int(np.ceil(slices_zyx[1].stop * scale_zyx[1]))],
        [int(np.floor(slices_zyx[2].start * scale_zyx[2])), int(np.ceil(slices_zyx[2].stop * scale_zyx[2]))],
    ]


def sparse_scout_z_indices_json(slices_zyx: tuple[slice, slice, slice], z_indices_l0: np.ndarray) -> list[int]:
    return [int(value) for value in np.asarray(z_indices_l0[slices_zyx[0]], dtype=np.int64)]


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
    overlap = shifted_overlap_mask(fixed_norm.shape, shift_array)
    overlap_finite = finite & np.isfinite(shifted) & overlap
    return shift_array, {
        "shape_zyx": [int(value) for value in fixed_norm.shape],
        "peak": float(peak),
        "corr_before": corrcoef_on_mask(fixed_norm, moving_norm, overlap_finite),
        "corr_after": corrcoef_on_mask(fixed_norm, shifted, overlap_finite),
        "corr_before_full_support": corrcoef_on_mask(fixed_norm, moving_norm, finite),
        "corr_valid_overlap_fraction": float(np.count_nonzero(overlap) / overlap.size),
    }


def maximum_compatible_clique(compatibility: np.ndarray) -> list[int]:
    if compatibility.ndim != 2 or compatibility.shape[0] != compatibility.shape[1]:
        raise ValueError(f"Expected square compatibility matrix, got {compatibility.shape}")
    n_items = int(compatibility.shape[0])
    best: list[int] = []

    def search(candidates: list[int], clique: list[int]) -> None:
        nonlocal best
        if len(clique) + len(candidates) < len(best):
            return
        if len(clique) > len(best):
            best = clique.copy()
        for offset, candidate in enumerate(candidates):
            if len(clique) + len(candidates) - offset < len(best):
                break
            next_clique = [*clique, candidate]
            next_candidates = [
                item
                for item in candidates[offset + 1 :]
                if all(bool(compatibility[item, member]) for member in next_clique)
            ]
            search(next_candidates, next_clique)

    search(list(range(n_items)), [])
    return best


def select_inlier_patch_measurements(
    total_shifts: np.ndarray,
    *,
    thresholds_zyx: np.ndarray = PATCH_INLIER_THRESHOLDS_ZYX,
    min_inliers: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    total_shifts = np.asarray(total_shifts, dtype=np.float64)
    thresholds_zyx = np.asarray(thresholds_zyx, dtype=np.float64)
    if total_shifts.ndim != 2 or total_shifts.shape[1] != 3:
        raise ValueError(f"Expected n x 3 total shifts, got {total_shifts.shape}")
    if total_shifts.shape[0] == 0:
        raise ValueError("No accepted patch shifts are available for inlier selection")
    if thresholds_zyx.shape != (3,) or np.any(thresholds_zyx <= 0):
        raise ValueError("thresholds_zyx must contain three positive values")
    if min_inliers < 1:
        raise ValueError("min_inliers must be >= 1")
    compatibility = np.all(np.abs(total_shifts[:, None, :] - total_shifts[None, :, :]) <= thresholds_zyx, axis=2)
    clique = maximum_compatible_clique(compatibility)
    inliers = np.zeros(total_shifts.shape[0], dtype=bool)
    inliers[clique] = True
    if int(np.count_nonzero(inliers)) < min_inliers:
        raise ValueError(f"Only {int(np.count_nonzero(inliers))} mutually compatible patch shifts found; require {min_inliers}")
    return inliers, np.median(total_shifts[inliers], axis=0)


def _patch_xy_key(row: Mapping[str, Any]) -> tuple[int, int] | None:
    fixed_slices = row.get("fixed_slices_zyx")
    if not isinstance(fixed_slices, list) or len(fixed_slices) != 3:
        return None
    try:
        return int(fixed_slices[1][0]), int(fixed_slices[2][0])
    except (TypeError, ValueError, IndexError):
        return None


def _inlier_patch_indices(accepted_patch_indices: list[int], inlier_mask: np.ndarray) -> set[int]:
    return {
        int(patch_index)
        for patch_index, is_inlier in zip(accepted_patch_indices, inlier_mask, strict=True)
        if bool(is_inlier)
    }


def patch_inliers_have_xy_diversity(
    patch_rows: list[dict[str, Any]],
    accepted_patch_indices: list[int],
    inlier_mask: np.ndarray,
    *,
    min_unique_xy: int = 2,
) -> bool:
    inlier_indices = _inlier_patch_indices(accepted_patch_indices, inlier_mask)
    keys = {
        key
        for row in patch_rows
        if int(row.get("patch_index", -1)) in inlier_indices
        for key in [_patch_xy_key(row)]
        if key is not None
    }
    return len(keys) >= min_unique_xy


def position_records_by_tile(position_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["tile"]: record for record in position_payload["tiles"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat_fingerprint(path: Path, *, role: str, tile: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    row: dict[str, Any] = {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stitch_legacy.is_ome_zarr_path(path):
        import zarr

        group = zarr.open_group(str(path), mode="r")
        level0 = group["0"]
        row["ome_zarr_level0_shape"] = [int(value) for value in level0.shape]
        row["ome_zarr_level0_chunks"] = [int(value) for value in level0.chunks]
        row["ome_zarr_axes"] = [
            str(axis.get("name", axis)) if isinstance(axis, Mapping) else str(axis)
            for axis in group.attrs.asdict().get("multiscales", [{}])[0].get("axes", [])
        ]
    if tile is not None:
        row["tile"] = tile
    return row


def tile_phase_source_fingerprints(
    *,
    reference_position: Path,
    reference_token: str,
    moving_token: str,
) -> list[dict[str, Any]]:
    payload = json.loads(reference_position.read_text())
    fingerprints = []
    for record in payload.get("tiles", []):
        reference_path = Path(record["path"])
        moving_path = corresponding_moving_path(
            reference_path,
            reference_token=reference_token,
            moving_token=moving_token,
        )
        fingerprints.append(file_stat_fingerprint(reference_path, role="reference", tile=record.get("tile")))
        fingerprints.append(file_stat_fingerprint(moving_path, role="moving", tile=moving_path.name))
    return fingerprints


def tile_phase_cache_key(
    *,
    reference_position: Path,
    reference_channel: int,
    moving_channel: int,
    reference_token: str,
    moving_token: str,
    level: int,
    upsample_factor: int,
    patch_shape_zyx: tuple[int, int, int] | None,
    min_inliers: int,
    max_candidate_patches: int,
    coarse_level: int,
    scout_z_samples: int | None,
) -> dict[str, Any]:
    return {
        "cache_version": "tile_phase_robust_v4",
        "reference_position": str(reference_position.resolve()),
        "reference_position_sha256": sha256_file(reference_position),
        "source_tile_fingerprints": tile_phase_source_fingerprints(
            reference_position=reference_position,
            reference_token=reference_token,
            moving_token=moving_token,
        ),
        "reference_channel": int(reference_channel),
        "moving_channel": int(moving_channel),
        "reference_token": reference_token,
        "moving_token": moving_token,
        "level": int(level),
        "upsample_factor": int(upsample_factor),
        "patch_shape_zyx": None if patch_shape_zyx is None else [int(value) for value in patch_shape_zyx],
        "min_inliers": int(min_inliers),
        "max_candidate_patches": int(max_candidate_patches),
        "coarse_level": int(coarse_level),
        "scout_z_samples": None if scout_z_samples is None else int(scout_z_samples),
        "inlier_thresholds_zyx": [float(value) for value in PATCH_INLIER_THRESHOLDS_ZYX],
        "quality_thresholds": {
            "min_corr_after": PATCH_MIN_CORR_AFTER,
            "min_corr_improvement": PATCH_MIN_CORR_IMPROVEMENT,
            "corr_improvement_metric": "gradient_component_ncc_when_available_else_same_shifted_overlap_support",
            "min_gradient_component_ncc_after_for_zero_residual": PATCH_MIN_GRADIENT_COMPONENT_NCC_AFTER_FOR_ZERO_RESIDUAL,
            "zero_residual_norm_px": PATCH_ZERO_RESIDUAL_NORM_PX,
            "max_residual_fraction": PATCH_MAX_RESIDUAL_FRACTION,
            "min_quality_accepted_for_early_stop": PATCH_MIN_QUALITY_ACCEPTED_FOR_EARLY_STOP,
            "summary_corr_source": "inlier_patch_median",
        },
    }


def tile_phase_row_status(row: Mapping[str, Any]) -> MeasurementStatus:
    status = row.get("measurement_status", row.get("status"))
    if status not in TILE_PHASE_MEASUREMENT_STATUSES:
        raise ValueError(f"Invalid tile-phase measurement status for {row.get('tile', '<unknown>')}: {status!r}")
    return cast(MeasurementStatus, status)


def _validate_tile_phase_row(row: dict[str, Any]) -> TilePhaseRow:
    if not isinstance(row.get("tile"), str) or not row["tile"]:
        raise ValueError("Tile-phase row is missing a string tile")
    tile_phase_row_status(row)
    return cast(TilePhaseRow, row)


def _validate_tile_phase_accepted_row(row: dict[str, Any]) -> TilePhaseAcceptedRow:
    status = tile_phase_row_status(row)
    if status not in {"direct_accepted", "fallback_accepted"}:
        raise ValueError(f"Cached tile-phase measurement for {row.get('tile')} is not accepted: status={status}")
    if "shift_um_zyx" not in row:
        raise ValueError(f"Cached tile-phase measurement for {row.get('tile')} is missing shift_um_zyx")
    return cast(TilePhaseAcceptedRow, row)


def _direct_accepted_status(details: Mapping[str, Any]) -> Literal["direct_accepted"]:
    status = details.get("measurement_status", "direct_accepted")
    if status != "direct_accepted":
        raise ValueError(f"Direct tile-phase measurement must be accepted, got status={status!r}")
    return "direct_accepted"


def load_tile_phase_cache(cache_path: Path, cache_key: dict[str, Any]) -> dict[str, TilePhaseAcceptedRow]:
    return _load_tile_phase_cache_rows(
        cache_path,
        cache_key,
        row_key="measurements",
        validate_row=_validate_tile_phase_accepted_row,
    )


def load_tile_phase_attempt_cache(cache_path: Path, cache_key: dict[str, Any]) -> dict[str, TilePhaseRow]:
    return _load_tile_phase_cache_rows(
        cache_path,
        cache_key,
        row_key="attempts",
        validate_row=_validate_tile_phase_row,
    )


def _load_tile_phase_cache_rows(
    cache_path: Path,
    cache_key: dict[str, Any],
    *,
    row_key: str,
    validate_row,
):
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text())
    if payload.get("cache_key") != cache_key:
        return {}
    return {row["tile"]: row for row in (validate_row(row) for row in payload.get(row_key, []))}


def phase_cache_keys_match(stored_key: dict[str, Any], current_key: dict[str, Any]) -> bool:
    ignored = {"quality_thresholds"}
    return {key: value for key, value in stored_key.items() if key not in ignored} == {
        key: value for key, value in current_key.items() if key not in ignored
    }


def load_compatible_tile_phase_attempt_cache(cache_path: Path, cache_key: dict[str, Any]) -> dict[str, TilePhaseRow]:
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text())
    stored_key = payload.get("cache_key")
    if not isinstance(stored_key, dict) or not phase_cache_keys_match(stored_key, cache_key):
        return {}
    rows_by_tile: dict[str, TilePhaseRow] = {}
    for row in payload.get("measurements", []):
        if row.get("patch_details"):
            validated = _validate_tile_phase_row(row)
            rows_by_tile[validated["tile"]] = validated
    for row in payload.get("attempts", []):
        if row.get("patch_details"):
            validated = _validate_tile_phase_row(row)
            rows_by_tile[validated["tile"]] = validated
    return rows_by_tile


def write_tile_phase_cache(
    cache_path: Path,
    cache_key: dict[str, Any],
    rows: list[TilePhaseAcceptedRow],
    attempts: list[TilePhaseRow] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "lightsheet.tile_phase_measurement_cache.v1",
        "cache_key": cache_key,
        "measurements": rows,
        "attempts": rows if attempts is None else attempts,
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(cache_path)


def _persist_tile_phase_cache(
    cache_path: Path,
    cache_key: dict[str, Any],
    rows: list[TilePhaseAcceptedRow],
    cached_attempts_by_tile: dict[str, TilePhaseRow],
) -> None:
    write_tile_phase_cache(cache_path, cache_key, rows, list(cached_attempts_by_tile.values()))


def _apply_shift_to_position_record(
    record: dict[str, Any],
    *,
    moving_tile_name: str,
    moving_path: Path,
    shift_um: np.ndarray,
) -> None:
    record["tile"] = moving_tile_name
    record["path"] = str(moving_path)
    if stitch_legacy.is_ome_zarr_path(moving_path):
        metadata = stitch_legacy.parse_ome_zarr_metadata(moving_path)
        record["axes"] = metadata.axes
        record["shape"] = [int(value) for value in metadata.shape]
        record["channels"] = list(metadata.channels)
        record["tracks"] = [
            {
                "slug": track.slug,
                "track_id": track.track_id,
                "channels": [int(channel) for channel in track.channels],
                "channel_names": list(track.channel_names),
            }
            for track in metadata.tracks
        ]
    for axis, value in zip(DIMENSIONS, shift_um, strict=True):
        record["translation_um"][axis] = float(record["translation_um"][axis] + value)


def apply_shift_row_to_position_record(
    record: dict[str, Any],
    row: TilePhaseAcceptedRow,
    *,
    moving_path: Path,
) -> np.ndarray:
    shift_um = np.asarray(row["shift_um_zyx"], dtype=np.float64)
    _apply_shift_to_position_record(record, moving_tile_name=row["tile"], moving_path=moving_path, shift_um=shift_um)
    return shift_um


def _build_failed_attempt_row(
    *,
    record: dict[str, Any],
    reference_tile: rough_legacy.TileRecord,
    moving_tile_name: str,
    moving_path: Path,
    error: Exception,
) -> TilePhaseFailedAttemptRow:
    direct_details = error.details if isinstance(error, TilePhaseMeasurementError) else None
    return {
        "tile": moving_tile_name,
        "side": record.get("side"),
        "reference_tile": reference_tile.tile,
        "reference_path": str(reference_tile.path),
        "moving_path": str(moving_path),
        "direct_error": str(error),
        "measurement_status": "direct_failed",
        "patch_details": direct_details,
        "corr_before": None if direct_details is None else direct_details.get("coarse_corr_before"),
        "corr_after": None if direct_details is None else direct_details.get("coarse_corr_after"),
        "n_inliers": None if direct_details is None else direct_details.get("n_inliers"),
    }


def _build_direct_measurement_row(
    *,
    record: dict[str, Any],
    reference_tile: rough_legacy.TileRecord,
    moving_path: Path,
    shift_px: np.ndarray,
    shift_um: np.ndarray,
    details: dict[str, Any],
    patch_details: dict[str, Any] | None,
    phase_error: float | None,
    cache_source: str | None = None,
) -> TilePhaseAcceptedRow:
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
        "phase_error": phase_error,
        "shape_zyx": details.get("shape_zyx"),
        "n_inliers": details.get("n_inliers"),
        "patch_details": patch_details,
        "measurement_status": _direct_accepted_status(details),
    }
    if cache_source is not None:
        row["cache_source"] = cache_source
    return row


def _build_fallback_measurement_row(
    *,
    record: dict[str, Any],
    reference_tile: rough_legacy.TileRecord,
    moving_path: Path,
    shift_px: np.ndarray,
    shift_um: np.ndarray,
    details: dict[str, Any],
    direct_error: str,
) -> TilePhaseAcceptedRow:
    return {
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
        "direct_error": direct_error,
        "fallback": True,
        "measurement_status": "fallback_accepted",
    }


def _append_failed_tile(
    failed_tiles: list[dict[str, Any]],
    *,
    record: dict[str, Any],
    reference_tile: rough_legacy.TileRecord,
    moving_tile: rough_legacy.TileRecord,
    moving_path: Path,
    error: str,
    direct_attempt: TilePhaseRow,
) -> None:
    failed_tiles.append(
        {
            "record": record,
            "reference_tile": reference_tile,
            "moving_tile": moving_tile,
            "moving_path": moving_path,
            "error": error,
            "direct_attempt": direct_attempt,
        }
    )


def _measure_tile_phase_attempt(
    *,
    record: dict[str, Any],
    reference_tile: rough_legacy.TileRecord,
    moving_tile: rough_legacy.TileRecord,
    moving_tile_name: str,
    moving_path: Path,
    reference_channel: int,
    moving_channel: int,
    level_factor: int,
    upsample_factor: int,
    patch_shape_zyx: tuple[int, int, int] | None,
    min_inliers: int,
    max_candidate_patches: int,
    coarse_level: int,
    scout_z_samples: int | None = None,
    cached_phase_attempt: TilePhaseRow | None,
) -> TilePhaseRow:
    try:
        if patch_shape_zyx is not None and cached_phase_attempt is not None:
            shift_px, details = rescore_cached_patch_attempt(
                reference_tile=reference_tile,
                moving_tile=moving_tile,
                reference_channel=reference_channel,
                moving_channel=moving_channel,
                cached_attempt=cached_phase_attempt,
                min_inliers=min_inliers,
            )
            shift_um = shift_px * np.abs(reference_tile.scale_zyx_um)
            row_record = {**record, "tile": moving_tile_name}
            return _build_direct_measurement_row(
                record=row_record,
                reference_tile=reference_tile,
                moving_path=moving_path,
                shift_px=shift_px,
                shift_um=shift_um,
                details=details,
                patch_details=details,
                phase_error=None,
                cache_source="cached_patch_residual_rescore",
            )

        if patch_shape_zyx is None:
            fixed = rough_legacy.sampled_tile_volume(
                reference_tile,
                channel=reference_channel,
                level_factor=level_factor,
            )
            moving = rough_legacy.sampled_tile_volume(
                moving_tile,
                channel=moving_channel,
                level_factor=level_factor,
            )
            shift_px, details = estimate_tile_shift_zyx_px(fixed, moving, upsample_factor=upsample_factor)
            shift_um = shift_px * np.abs(reference_tile.scale_zyx_um) * level_factor
        else:
            shift_px, details = measure_patch_tile_shift(
                reference_tile=reference_tile,
                moving_tile=moving_tile,
                reference_channel=reference_channel,
                moving_channel=moving_channel,
                patch_shape_zyx=patch_shape_zyx,
                coarse_level=coarse_level,
                upsample_factor=upsample_factor,
                max_candidate_patches=max_candidate_patches,
                min_inliers=min_inliers,
                scout_z_samples=scout_z_samples,
            )
            shift_um = shift_px * np.abs(reference_tile.scale_zyx_um)
    except Exception as exc:
        return _build_failed_attempt_row(
            record=record,
            reference_tile=reference_tile,
            moving_tile_name=moving_tile_name,
            moving_path=moving_path,
            error=exc,
        )

    row_record = {**record, "tile": moving_tile_name}
    return _build_direct_measurement_row(
        record=row_record,
        reference_tile=reference_tile,
        moving_path=moving_path,
        shift_px=shift_px,
        shift_um=shift_um,
        details=details,
        patch_details=details if patch_shape_zyx is not None else None,
        phase_error=details.get("phase_error"),
    )


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
    measurement_by_tile = {
        item["tile"]: item for item in tile_phase_summary.get("measurements", []) if "tile" in item
    }
    adapted_tiles = []
    missing = []
    for record in adapted["tiles"]:
        moving_tile = make_moving_tile_name(record["tile"], reference_token=reference_token, moving_token=moving_token)
        position_record = position_by_tile.get(moving_tile)
        if position_record is None:
            missing.append(moving_tile)
            continue
        measurement = measurement_by_tile.get(moving_tile)
        if measurement is None:
            raise ValueError(f"Refusing to adapt registration without tile-phase measurement for {moving_tile}")
        try:
            measurement_status = tile_phase_row_status(measurement)
        except ValueError:
            measurement_status = measurement.get("measurement_status", measurement.get("status"))
            raise ValueError(
                f"Refusing to adapt registration for rejected tile {moving_tile}: status={measurement_status}"
            ) from None
        if measurement_status not in {"direct_accepted", "fallback_accepted"}:
            raise ValueError(f"Refusing to adapt registration for rejected tile {moving_tile}: status={measurement_status}")
        adapted_record = json.loads(json.dumps(record))
        adapted_record["tile"] = moving_tile
        adapted_record["stage_translation_um"] = position_record["translation_um"]
        adapted_record["stage_scale_um"] = position_record["scale_um"]
        if "path" in adapted_record or position_record.get("path") is not None:
            adapted_record["path"] = position_record["path"]
        if adapted_record.get("registered_affine") != record.get("registered_affine"):
            raise ValueError(f"registered_affine changed while adapting {moving_tile}")
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
    adapted["transform_contract"] = {
        "registered_affine_copied_exactly": True,
        "stage_translation_is_phase_adjusted": True,
        "loader_must_compose_stage_and_registered_affine": True,
    }
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
        "measurement_status": tile_phase_row_status(item),
                "fallback": bool(item.get("fallback", False)),
            }
            for item in tile_phase_summary["measurements"]
        ],
    }
    output_registration.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_registration.with_name(f"{output_registration.name}.tmp")
    tmp_path.write_text(json.dumps(adapted, indent=2) + "\n")
    tmp_path.replace(output_registration)
    return output_registration.resolve()


def make_moving_tile_record(reference_tile: rough_legacy.TileRecord, moving_path: Path) -> rough_legacy.TileRecord:
    if reference_tile.axes == "ZYX" and flattened_channel_count(moving_path) > 1:
        return rough_legacy.TileRecord(
            tile=moving_path.name,
            side=reference_tile.side,
            path=moving_path,
            translation_zyx_um=reference_tile.translation_zyx_um.copy(),
            scale_zyx_um=reference_tile.scale_zyx_um.copy(),
            shape_zyx=reference_tile.shape_zyx.copy(),
            axes="ZYX",
        )
    elif stitch_legacy.is_ome_zarr_path(moving_path):
        metadata = stitch_legacy.parse_ome_zarr_metadata(moving_path)
        moving_shape_zyx = np.asarray(stitch_legacy.normalized_czyx_shape(metadata.axes, metadata.shape)[1:], dtype=np.int64)
        moving_axes = metadata.axes
    else:
        moving_shape_zyx, moving_axes = rough_legacy.tile_shape_and_axes(moving_path)
    return logical_tile_record(
        rough_legacy.TileRecord(
            tile=moving_path.name,
            side=reference_tile.side,
            path=moving_path,
            translation_zyx_um=reference_tile.translation_zyx_um.copy(),
            scale_zyx_um=reference_tile.scale_zyx_um.copy(),
            shape_zyx=np.asarray(moving_shape_zyx, dtype=np.int64),
            axes=moving_axes,
        )
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
    max_neighbors: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    fallback_min_inliers = min_inliers
    failed_center = failed_tile.translation_zyx_um + failed_tile.shape_zyx.astype(np.float64) * failed_tile.scale_zyx_um / 2.0
    neighbor_rows = []
    all_candidates = sorted(
        [
            (float(np.linalg.norm((tile.translation_zyx_um - failed_tile.translation_zyx_um)[1:3])), tile, shift_um)
            for tile, shift_um in successful_tiles
            if tile.side == failed_tile.side
        ],
        key=lambda item: item[0],
    )
    candidates = all_candidates if max_neighbors is None else all_candidates[:max_neighbors]
    inferred_shift_rows = []
    final_shift_px: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    for distance_um_yx, neighbor_tile, neighbor_shift_um in candidates:
        inferred_shift_px = neighbor_shift_um / np.abs(failed_tile.scale_zyx_um)
        row = {
            "neighbor_tile": neighbor_tile.tile,
            "distance_um_yx": float(distance_um_yx),
            "status": "accepted",
            "reason": "same_side_shift_field_sample",
            "inferred_shift_px_zyx": [float(value) for value in inferred_shift_px],
            "neighbor_shift_um_zyx": [float(value) for value in neighbor_shift_um],
        }
        inferred_shift_rows.append(inferred_shift_px)
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

    if len(inferred_shift_rows) < min_inliers:
        raise ValueError(f"Only {len(inferred_shift_rows)} same-side fallback shifts found; require {min_inliers}")
    if final_shift_px is None or inlier_mask is None:
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
            row["reason"] = "outlier_shift_field_cluster"
        accepted_index += 1
        if accepted_index >= len(inlier_mask):
            break

    final_shift_um = final_shift_px * np.abs(failed_tile.scale_zyx_um)
    return final_shift_px, {
        "mode": "same_side_shift_field_fallback",
        "min_inliers": int(fallback_min_inliers),
        "failed_tile_center_um_zyx": [float(value) for value in failed_center],
        "patch_shape_zyx": [int(value) for value in patch_shape_zyx],
        "source": "nearest_non_adjacent_successful_tile_shifts",
        "n_neighbors_considered": len(candidates),
        "n_measured": len(inferred_shift_rows),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "shift_um_zyx": [float(value) for value in final_shift_um],
        "neighbors": neighbor_rows,
        "corr_before": None,
        "corr_after": None,
    }


def measure_sparse_scout_patch_shift(
    *,
    fixed_coarse: np.ndarray,
    moving_coarse: np.ndarray,
    fixed_coarse_scale_zyx: np.ndarray,
    fixed_z_indices_l0: np.ndarray,
    moving_z_indices_l0: np.ndarray,
    coarse_shift_coarse_px: np.ndarray,
    coarse_details: dict[str, Any],
    patch_shape_zyx: tuple[int, int, int],
    max_candidate_patches: int,
    min_inliers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    coarse_shift_l0_px = coarse_shift_coarse_px * fixed_coarse_scale_zyx
    candidates = candidate_sparse_scout_chunk_slices(
        fixed_coarse,
        patch_shape_zyx=patch_shape_zyx,
        scout_scale_zyx=fixed_coarse_scale_zyx,
        max_candidates=max_candidate_patches,
        moving_shape_zyx=np.asarray(moving_coarse.shape, dtype=np.int64),
        shift_zyx_px=coarse_shift_coarse_px,
    )
    patch_rows: list[dict[str, Any]] = []
    accepted_shift_rows: list[np.ndarray] = []
    accepted_patch_indices: list[int] = []
    final_shift_l0_px: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    early_stop_after_patch: int | None = None

    for patch_index, candidate in enumerate(candidates):
        fixed_slices = candidate["fixed_slices"]
        moving_slices, realized_seed_shift_coarse_px = shifted_slices_with_realized_shift(
            fixed_slices,
            requested_shift_zyx_px=coarse_shift_coarse_px,
        )
        row = {
            "patch_index": int(patch_index),
            "patch_source": "sparse_subifd_scout",
            "fixed_slices_zyx": sparse_scout_slices_to_l0_json(
                fixed_slices,
                scale_zyx=fixed_coarse_scale_zyx,
                z_indices_l0=fixed_z_indices_l0,
            ),
            "moving_slices_zyx": sparse_scout_slices_to_l0_json(
                moving_slices,
                scale_zyx=fixed_coarse_scale_zyx,
                z_indices_l0=moving_z_indices_l0,
            ),
            "fixed_z_indices_l0": sparse_scout_z_indices_json(fixed_slices, fixed_z_indices_l0),
            "moving_z_indices_l0": sparse_scout_z_indices_json(moving_slices, moving_z_indices_l0),
            "fixed_slices_coarse_zyx": slices_to_json(fixed_slices),
            "moving_slices_coarse_zyx": slices_to_json(moving_slices),
            "requested_coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
            "realized_seed_shift_px_zyx": [float(value) for value in realized_seed_shift_coarse_px * fixed_coarse_scale_zyx],
            "coarse_scale_zyx": [float(value) for value in fixed_coarse_scale_zyx],
            "content_score": float(candidate["content_score"]),
            "positive_fraction": float(candidate["positive_fraction"]),
        }
        if not slices_within_shape(moving_slices, np.asarray(moving_coarse.shape, dtype=np.int64)):
            row.update(status="rejected", reason="moving_patch_out_of_bounds")
            patch_rows.append(row)
            continue
        fixed_patch = fixed_coarse[fixed_slices]
        moving_patch = moving_coarse[moving_slices]
        residual_shift_coarse_px, details = estimate_tile_shift_zyx_px_gpu(fixed_patch, moving_patch)
        total_shift_coarse_px = realized_seed_shift_coarse_px + residual_shift_coarse_px
        total_shift_l0_px = total_shift_coarse_px * fixed_coarse_scale_zyx
        residual_shift_l0_px = residual_shift_coarse_px * fixed_coarse_scale_zyx
        row.update(
            status="accepted",
            reason="measured",
            residual_shift_px_zyx=[float(value) for value in residual_shift_l0_px],
            residual_shift_coarse_px_zyx=[float(value) for value in residual_shift_coarse_px],
            total_shift_px_zyx=[float(value) for value in total_shift_l0_px],
            total_shift_coarse_px_zyx=[float(value) for value in total_shift_coarse_px],
            peak=details["peak"],
            corr_before=details["corr_before"],
            corr_after=details["corr_after"],
        )
        for detail_key in (
            "corr_before_full_support",
            "corr_valid_overlap_fraction",
        ):
            if detail_key in details:
                row[detail_key] = details[detail_key]
        rejection_reasons = []
        if finite_float(details.get("corr_after")) < PATCH_MIN_CORR_AFTER:
            rejection_reasons.append("corr_after_below_threshold")
        if np.any(
            np.abs(residual_shift_coarse_px)
            >= PATCH_MAX_RESIDUAL_FRACTION * np.asarray(fixed_patch.shape, dtype=np.float64)
        ):
            rejection_reasons.append("residual_near_periodic_wrap")
        if rejection_reasons:
            row.update(status="rejected", reason="quality_threshold", rejection_reasons=rejection_reasons)
            patch_rows.append(row)
            continue
        accepted_patch_indices.append(patch_index)
        accepted_shift_rows.append(total_shift_l0_px)
        patch_rows.append(row)
        if len(accepted_shift_rows) >= max(PATCH_MIN_QUALITY_ACCEPTED_FOR_EARLY_STOP, min_inliers):
            try:
                inlier_mask, final_shift_l0_px = select_inlier_patch_measurements(
                    np.vstack(accepted_shift_rows).astype(np.float64),
                    min_inliers=min_inliers,
                )
            except ValueError:
                final_shift_l0_px = None
                inlier_mask = None
            else:
                if patch_inliers_have_xy_diversity(patch_rows, accepted_patch_indices, inlier_mask):
                    early_stop_after_patch = patch_index
                    break
                final_shift_l0_px = None
                inlier_mask = None

    base_details = {
        "mode": "sparse_subifd_patch_phase_robust_v1",
        "patch_source": "sparse_subifd_scout",
        "patch_shape_zyx": [int(value) for value in patch_shape_zyx],
        "coarse_scale_zyx": [float(value) for value in fixed_coarse_scale_zyx],
        "coarse_shift_level_px_zyx": [float(value) for value in coarse_shift_coarse_px],
        "requested_coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
        "coarse_corr_before": coarse_details["corr_before"],
        "coarse_corr_after": coarse_details["corr_after"],
        "n_candidates": len(candidates),
        "n_measured": len(accepted_shift_rows),
        "n_inliers": 0 if inlier_mask is None else int(np.count_nonzero(inlier_mask)),
        "patches": patch_rows,
    }
    if len(accepted_shift_rows) < min_inliers:
        details = {**base_details, "measurement_status": "direct_failed"}
        raise TilePhaseMeasurementError(
            f"sparse scout produced {len(accepted_shift_rows)} accepted patch shifts; require {min_inliers}",
            details=details,
        )
    if final_shift_l0_px is None or inlier_mask is None:
        try:
            inlier_mask, final_shift_l0_px = select_inlier_patch_measurements(
                np.vstack(accepted_shift_rows).astype(np.float64),
                min_inliers=min_inliers,
            )
        except ValueError as exc:
            details = {**base_details, "measurement_status": "direct_failed"}
            raise TilePhaseMeasurementError(str(exc), details=details) from exc
    if not patch_inliers_have_xy_diversity(patch_rows, accepted_patch_indices, inlier_mask):
        details = {
            **base_details,
            "measurement_status": "direct_failed",
            "spatial_diversity_error": "inlier sparse scout shifts came from fewer than 2 XY windows",
        }
        raise TilePhaseMeasurementError("Inlier sparse scout shifts came from fewer than 2 XY windows", details=details)

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

    inlier_rows = [row for row in patch_rows if row.get("inlier")]
    inlier_corr_before = np.asarray([row["corr_before"] for row in inlier_rows], dtype=np.float64)
    inlier_corr_after = np.asarray([row["corr_after"] for row in inlier_rows], dtype=np.float64)
    return final_shift_l0_px, {
        **base_details,
        "measurement_status": "direct_accepted",
        "corr_before": float(np.median(inlier_corr_before)),
        "corr_after": float(np.median(inlier_corr_after)),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "early_stop_after_patch": early_stop_after_patch,
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
    moving_channel: int = 0,
    scout_z_samples: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if min_inliers < 3:
        raise ValueError("Patch-mode registration requires min_inliers >= 3")
    (
        fixed_coarse,
        fixed_coarse_scale_zyx,
        fixed_source_level,
        fixed_available_levels,
        fixed_z_indices_l0,
    ) = sampled_tile_volume_from_subifd(
        reference_tile,
        channel=reference_channel,
        requested_level=coarse_level,
        z_samples=scout_z_samples,
    )
    (
        moving_coarse,
        moving_coarse_scale_zyx,
        moving_source_level,
        moving_available_levels,
        moving_z_indices_l0,
    ) = sampled_tile_volume_from_subifd(
        moving_tile,
        channel=moving_channel,
        requested_level=coarse_level,
        z_samples=scout_z_samples,
    )
    if not np.array_equal(fixed_coarse_scale_zyx, moving_coarse_scale_zyx):
        raise ValueError(
            "Fixed and moving coarse scout scales differ: "
            f"{fixed_coarse_scale_zyx.tolist()} vs {moving_coarse_scale_zyx.tolist()}"
        )
    coarse_shift_coarse_px, coarse_details = estimate_tile_shift_zyx_px_gpu(fixed_coarse, moving_coarse)
    if scout_z_samples is not None:
        shift_px, details = measure_sparse_scout_patch_shift(
            fixed_coarse=fixed_coarse,
            moving_coarse=moving_coarse,
            fixed_coarse_scale_zyx=fixed_coarse_scale_zyx,
            fixed_z_indices_l0=fixed_z_indices_l0,
            moving_z_indices_l0=moving_z_indices_l0,
            coarse_shift_coarse_px=coarse_shift_coarse_px,
            coarse_details=coarse_details,
            patch_shape_zyx=patch_shape_zyx,
            max_candidate_patches=max_candidate_patches,
            min_inliers=min_inliers,
        )
        details.update(
            coarse_level=int(coarse_level),
            scout_z_samples=int(scout_z_samples),
            fixed_source_level=int(fixed_source_level),
            moving_source_level=int(moving_source_level),
            fixed_available_levels=int(fixed_available_levels),
            moving_available_levels=int(moving_available_levels),
        )
        return shift_px, details
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
        moving_slices, realized_seed_shift_px = shifted_slices_with_realized_shift(
            fixed_slices,
            requested_shift_zyx_px=coarse_shift_l0_px,
        )
        row = {
            "patch_index": int(patch_index),
            "fixed_slices_zyx": slices_to_json(fixed_slices),
            "moving_slices_zyx": slices_to_json(moving_slices),
            "requested_coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
            "realized_seed_shift_px_zyx": [float(value) for value in realized_seed_shift_px],
            "content_score": float(candidate["content_score"]),
            "positive_fraction": float(candidate["positive_fraction"]),
        }
        if not slices_within_shape(moving_slices, moving_tile.shape_zyx):
            row.update(status="rejected", reason="moving_patch_out_of_bounds")
            patch_rows.append(row)
            continue
        fixed_patch = read_tile_patch(reference_tile, channel=reference_channel, slices_zyx=fixed_slices)
        moving_patch = read_tile_patch(moving_tile, channel=moving_channel, slices_zyx=moving_slices)
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
        total_shift_px = realized_seed_shift_px + residual_shift_px
        rejection_reasons = patch_quality_rejection_reasons(
            details=details,
            residual_shift_px=residual_shift_px,
            patch_shape_zyx=patch_shape_zyx,
        )
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
        for detail_key in (
            "patch_optimizer",
            "phasecorr_method",
            "phasecorr_fft_highpass_sigma_zyx",
            "phasecorr_spatial_highpass_sigma",
            "phase_candidate_shift_px_zyx",
            "phase_candidate_corr_after",
            "gradient_component_ncc_before",
            "gradient_component_ncc_phase_after",
            "gradient_component_ncc_after",
            "selected_residual",
            "simpleitk_mattes_status",
            "simpleitk_mattes_error",
            "simpleitk_mattes_optimizer",
            "simpleitk_mattes_bins",
            "simpleitk_mattes_sampling_fraction",
            "simpleitk_mattes_metric_value",
            "simpleitk_mattes_stop",
            "simpleitk_mattes_iterations",
            "simpleitk_mattes_residual_transform_parameters_xyz_px",
            "mattes_residual_shift_px_zyx",
            "simpleitk_mattes_mip_gradient_component_ncc",
            "corr_before_full_support",
            "corr_valid_overlap_fraction",
        ):
            if detail_key in details:
                row[detail_key] = details[detail_key]
        if rejection_reasons:
            row.update(status="rejected", reason="quality_threshold", rejection_reasons=rejection_reasons)
            patch_rows.append(row)
            continue
        accepted_patch_indices.append(patch_index)
        accepted_shift_rows.append(total_shift_px)
        patch_rows.append(row)
        if len(accepted_shift_rows) >= max(PATCH_MIN_QUALITY_ACCEPTED_FOR_EARLY_STOP, min_inliers):
            try:
                inlier_mask, final_shift_px = select_inlier_patch_measurements(
                    np.vstack(accepted_shift_rows).astype(np.float64),
                    min_inliers=min_inliers,
                )
            except ValueError:
                final_shift_px = None
                inlier_mask = None
            else:
                if patch_inliers_have_xy_diversity(patch_rows, accepted_patch_indices, inlier_mask):
                    early_stop_after_patch = patch_index
                    break
                final_shift_px = None
                inlier_mask = None

    if early_stop_after_patch is not None:
        for skipped_index, candidate in enumerate(candidates[early_stop_after_patch + 1 :], start=early_stop_after_patch + 1):
            fixed_slices = candidate["fixed_slices"]
            moving_slices, realized_seed_shift_px = shifted_slices_with_realized_shift(
                fixed_slices,
                requested_shift_zyx_px=coarse_shift_l0_px,
            )
            patch_rows.append(
                {
                    "patch_index": int(skipped_index),
                    "fixed_slices_zyx": slices_to_json(fixed_slices),
                    "moving_slices_zyx": slices_to_json(moving_slices),
                    "requested_coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
                    "realized_seed_shift_px_zyx": [float(value) for value in realized_seed_shift_px],
                    "content_score": float(candidate["content_score"]),
                    "positive_fraction": float(candidate["positive_fraction"]),
                    "status": "skipped",
                    "reason": "skipped_after_enough_inliers",
                }
            )

    def failed_direct_details() -> dict[str, Any]:
        return {
            "mode": "l0_patch_phase_robust_v2",
            "measurement_status": "direct_failed",
            "patch_shape_zyx": [int(value) for value in patch_shape_zyx],
            "coarse_level": int(coarse_level),
            "scout_z_samples": None if scout_z_samples is None else int(scout_z_samples),
            "coarse_scale_zyx": [float(value) for value in fixed_coarse_scale_zyx],
            "fixed_source_level": int(fixed_source_level),
            "moving_source_level": int(moving_source_level),
            "fixed_available_levels": int(fixed_available_levels),
            "moving_available_levels": int(moving_available_levels),
            "coarse_shift_level_px_zyx": [float(value) for value in coarse_shift_coarse_px],
            "requested_coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
            "coarse_corr_before": coarse_details["corr_before"],
            "coarse_corr_after": coarse_details["corr_after"],
            "n_candidates": len(candidates),
            "n_measured": len(accepted_shift_rows),
            "n_inliers": 0 if inlier_mask is None else int(np.count_nonzero(inlier_mask)),
            "early_stop_after_patch": early_stop_after_patch,
            "quality_thresholds": {
                "min_corr_after": PATCH_MIN_CORR_AFTER,
                "min_corr_improvement": PATCH_MIN_CORR_IMPROVEMENT,
                "corr_improvement_metric": "same_shifted_overlap_support",
                "max_residual_fraction": PATCH_MAX_RESIDUAL_FRACTION,
                "min_quality_accepted_for_early_stop": PATCH_MIN_QUALITY_ACCEPTED_FOR_EARLY_STOP,
                "summary_corr_source": "inlier_patch_median",
            },
            "patches": patch_rows,
        }

    if len(accepted_shift_rows) < min_inliers:
        message = f"{reference_tile.tile} produced {len(accepted_shift_rows)} accepted patch shifts; require {min_inliers}"
        raise TilePhaseMeasurementError(message, details=failed_direct_details())
    if final_shift_px is None or inlier_mask is None:
        total_shifts = np.vstack(accepted_shift_rows).astype(np.float64)
        try:
            inlier_mask, final_shift_px = select_inlier_patch_measurements(total_shifts, min_inliers=min_inliers)
        except ValueError as exc:
            raise TilePhaseMeasurementError(str(exc), details=failed_direct_details()) from exc
    if not patch_inliers_have_xy_diversity(patch_rows, accepted_patch_indices, inlier_mask):
        details = failed_direct_details()
        details["spatial_diversity_error"] = "inlier patch shifts came from fewer than 2 XY windows"
        raise TilePhaseMeasurementError("Inlier patch shifts came from fewer than 2 XY windows", details=details)
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

    inlier_rows = [row for row in patch_rows if row.get("inlier")]
    inlier_corr_before = np.asarray([row["corr_before"] for row in inlier_rows], dtype=np.float64)
    inlier_corr_after = np.asarray([row["corr_after"] for row in inlier_rows], dtype=np.float64)

    return final_shift_px, {
        "mode": "l0_patch_phase_robust_v2",
        "measurement_status": "direct_accepted",
        "patch_shape_zyx": [int(value) for value in patch_shape_zyx],
        "coarse_level": int(coarse_level),
        "scout_z_samples": None if scout_z_samples is None else int(scout_z_samples),
        "coarse_scale_zyx": [float(value) for value in fixed_coarse_scale_zyx],
        "fixed_source_level": int(fixed_source_level),
        "moving_source_level": int(moving_source_level),
        "fixed_available_levels": int(fixed_available_levels),
        "moving_available_levels": int(moving_available_levels),
        "coarse_shift_level_px_zyx": [float(value) for value in coarse_shift_coarse_px],
        "requested_coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
        "corr_before": float(np.median(inlier_corr_before)),
        "corr_after": float(np.median(inlier_corr_after)),
        "coarse_corr_before": coarse_details["corr_before"],
        "coarse_corr_after": coarse_details["corr_after"],
        "n_candidates": len(candidates),
        "n_measured": len(accepted_shift_rows),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "inlier_corr_after_min": float(np.min(inlier_corr_after)),
        "inlier_corr_after_median": float(np.median(inlier_corr_after)),
        "early_stop_after_patch": early_stop_after_patch,
        "quality_thresholds": {
            "min_corr_after": PATCH_MIN_CORR_AFTER,
            "min_corr_improvement": PATCH_MIN_CORR_IMPROVEMENT,
            "corr_improvement_metric": "same_shifted_overlap_support",
            "max_residual_fraction": PATCH_MAX_RESIDUAL_FRACTION,
            "min_quality_accepted_for_early_stop": PATCH_MIN_QUALITY_ACCEPTED_FOR_EARLY_STOP,
            "summary_corr_source": "inlier_patch_median",
        },
        "patches": patch_rows,
    }


def rescore_cached_patch_attempt(
    *,
    reference_tile: rough_legacy.TileRecord,
    moving_tile: rough_legacy.TileRecord,
    reference_channel: int,
    cached_attempt: dict[str, Any],
    min_inliers: int,
    moving_channel: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    patch_details = cached_attempt.get("patch_details")
    if not isinstance(patch_details, dict):
        raise TilePhaseMeasurementError("Cached attempt has no patch details", details=None)
    patch_shape_zyx = tuple(int(value) for value in patch_details.get("patch_shape_zyx", []))
    if len(patch_shape_zyx) != 3:
        raise TilePhaseMeasurementError("Cached attempt has no patch_shape_zyx", details=patch_details)

    patch_rows = []
    accepted_shift_rows = []
    accepted_patch_indices = []
    cached_patches = patch_details.get("patches", [])
    final_shift_px: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    early_stop_after_patch: int | None = None
    for patch in cached_patches:
        row = json.loads(json.dumps(patch))
        residual_values = row.get("residual_shift_px_zyx")
        total_values = row.get("total_shift_px_zyx")
        if residual_values is None or total_values is None:
            patch_rows.append(row)
            continue
        fixed_slices = tuple(slice(int(start), int(stop)) for start, stop in row["fixed_slices_zyx"])
        moving_slices = tuple(slice(int(start), int(stop)) for start, stop in row["moving_slices_zyx"])
        fixed_patch = read_tile_patch(reference_tile, channel=reference_channel, slices_zyx=fixed_slices)
        moving_patch = read_tile_patch(moving_tile, channel=moving_channel, slices_zyx=moving_slices)
        if fixed_patch.shape != patch_shape_zyx or moving_patch.shape != patch_shape_zyx:
            row.update(
                status="rejected",
                reason="patch_shape_mismatch",
                fixed_shape_zyx=[int(value) for value in fixed_patch.shape],
                moving_shape_zyx=[int(value) for value in moving_patch.shape],
            )
            patch_rows.append(row)
            continue
        residual_shift_px = np.asarray(residual_values, dtype=np.float64)
        fixed_norm = normalize_volume_for_phase(fixed_patch)
        moving_norm = normalize_volume_for_phase(moving_patch)
        shifted = ndimage.shift(moving_norm, shift=residual_shift_px, order=1, mode="constant", cval=0.0, prefilter=False)
        finite = np.isfinite(fixed_norm) & np.isfinite(moving_norm)
        overlap = shifted_overlap_mask(fixed_norm.shape, residual_shift_px)
        overlap_finite = finite & np.isfinite(shifted) & overlap
        details = {
            "corr_before": corrcoef_on_mask(fixed_norm, moving_norm, overlap_finite),
            "corr_after": corrcoef_on_mask(fixed_norm, shifted, overlap_finite),
        }
        rejection_reasons = patch_quality_rejection_reasons(
            details=details,
            residual_shift_px=residual_shift_px,
            patch_shape_zyx=patch_shape_zyx,
        )
        row.update(
            status="accepted",
            reason="measured",
            corr_before=details["corr_before"],
            corr_after=details["corr_after"],
            corr_before_full_support=corrcoef_on_mask(fixed_norm, moving_norm, finite),
            corr_valid_overlap_fraction=float(np.count_nonzero(overlap_finite) / overlap_finite.size),
            quality_rejection_reasons=None,
            rejection_reasons=None,
            rescored_from_cached_residual=True,
        )
        if rejection_reasons:
            row.update(status="rejected", reason="quality_threshold", rejection_reasons=rejection_reasons)
            patch_rows.append(row)
            continue
        accepted_patch_indices.append(int(row["patch_index"]))
        accepted_shift_rows.append(np.asarray(total_values, dtype=np.float64))
        patch_rows.append(row)
        if len(accepted_shift_rows) >= max(PATCH_MIN_QUALITY_ACCEPTED_FOR_EARLY_STOP, min_inliers):
            try:
                inlier_mask, final_shift_px = select_inlier_patch_measurements(
                    np.vstack(accepted_shift_rows).astype(np.float64),
                    min_inliers=min_inliers,
                )
            except ValueError:
                final_shift_px = None
                inlier_mask = None
            else:
                if patch_inliers_have_xy_diversity(patch_rows, accepted_patch_indices, inlier_mask):
                    early_stop_after_patch = int(row["patch_index"])
                    break
                final_shift_px = None
                inlier_mask = None

    if early_stop_after_patch is not None:
        for patch in cached_patches:
            if int(patch.get("patch_index", -1)) <= early_stop_after_patch:
                continue
            row = json.loads(json.dumps(patch))
            row.update(status="skipped", reason="skipped_after_enough_inliers")
            patch_rows.append(row)

    details = json.loads(json.dumps(patch_details))
    details.update(
        cache_source="cached_patch_residual_rescore",
        quality_thresholds={
            "min_corr_after": PATCH_MIN_CORR_AFTER,
            "min_corr_improvement": PATCH_MIN_CORR_IMPROVEMENT,
            "corr_improvement_metric": "same_shifted_overlap_support",
            "max_residual_fraction": PATCH_MAX_RESIDUAL_FRACTION,
            "min_quality_accepted_for_early_stop": PATCH_MIN_QUALITY_ACCEPTED_FOR_EARLY_STOP,
            "summary_corr_source": "inlier_patch_median",
        },
        patches=patch_rows,
        n_measured=len(accepted_shift_rows),
        n_inliers=0,
        early_stop_after_patch=early_stop_after_patch,
    )
    if len(accepted_shift_rows) < min_inliers:
        details["measurement_status"] = "direct_failed"
        raise TilePhaseMeasurementError(
            f"{reference_tile.tile} produced {len(accepted_shift_rows)} accepted cached patch shifts; require {min_inliers}",
            details=details,
        )
    if final_shift_px is None or inlier_mask is None:
        try:
            inlier_mask, final_shift_px = select_inlier_patch_measurements(
                np.vstack(accepted_shift_rows).astype(np.float64),
                min_inliers=min_inliers,
            )
        except ValueError as exc:
            details["measurement_status"] = "direct_failed"
            raise TilePhaseMeasurementError(str(exc), details=details) from exc
    if not patch_inliers_have_xy_diversity(patch_rows, accepted_patch_indices, inlier_mask):
        details["measurement_status"] = "direct_failed"
        details["n_inliers"] = int(np.count_nonzero(inlier_mask))
        details["spatial_diversity_error"] = "inlier patch shifts came from fewer than 2 XY windows"
        raise TilePhaseMeasurementError("Inlier patch shifts came from fewer than 2 XY windows", details=details)

    inlier_patch_indices = {
        patch_index
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

    inlier_rows = [row for row in patch_rows if row.get("inlier")]
    inlier_corr_before = np.asarray([row["corr_before"] for row in inlier_rows], dtype=np.float64)
    inlier_corr_after = np.asarray([row["corr_after"] for row in inlier_rows], dtype=np.float64)
    details.update(
        measurement_status="direct_accepted",
        corr_before=float(np.median(inlier_corr_before)),
        corr_after=float(np.median(inlier_corr_after)),
        n_inliers=int(np.count_nonzero(inlier_mask)),
        inlier_corr_after_min=float(np.min(inlier_corr_after)),
        inlier_corr_after_median=float(np.median(inlier_corr_after)),
    )
    return final_shift_px, details


def align_tiles_to_reference(
    *,
    reference_position: Path,
    output_position: Path,
    output_dir: Path,
    reference_channel: int = 3,
    moving_channel: int = 0,
    reference_token: str = "488514561638",
    moving_token: str = "405",
    level: int = 4,
    upsample_factor: int = 10,
    patch_shape_zyx: tuple[int, int, int] | None = None,
    min_inliers: int = 3,
    max_candidate_patches: int = 24,
    coarse_level: int = 4,
    scout_z_samples: int | None = 32,
    workers: int = 1,
    reference_registration_input: Path | None = None,
    output_registration: Path | None = None,
) -> Path:
    if level < 0:
        raise ValueError("level must be non-negative")
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")
    if min_inliers < 1:
        raise ValueError("min_inliers must be >= 1")
    if patch_shape_zyx is not None and min_inliers < 3:
        raise ValueError("patch-mode min_inliers must be >= 3")
    if max_candidate_patches < 1:
        raise ValueError("max_candidate_patches must be >= 1")
    if scout_z_samples is not None and scout_z_samples < 1:
        raise ValueError("scout_z_samples must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if patch_shape_zyx is None and output_registration is not None:
        raise ValueError("--output-registration requires --patch-shape-zyx")
    if output_registration is not None and reference_registration_input is None:
        raise ValueError("--output-registration requires --reference-registration-input")
    payload = json.loads(reference_position.read_text())
    if all(record.get("shape") is not None and record.get("axes") for record in payload["tiles"]):
        reference_tiles = [tile_record_from_position_record(record) for record in payload["tiles"]]
    else:
        reference_tiles = [logical_tile_record(tile) for tile in rough_legacy.load_tiles(payload)]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "tile_phase_measurement_cache.json"
    cache_key = tile_phase_cache_key(
        reference_position=reference_position,
        reference_channel=reference_channel,
        moving_channel=moving_channel,
        reference_token=reference_token,
        moving_token=moving_token,
        level=level,
        upsample_factor=upsample_factor,
        patch_shape_zyx=patch_shape_zyx,
        min_inliers=min_inliers,
        max_candidate_patches=max_candidate_patches,
        coarse_level=coarse_level,
        scout_z_samples=scout_z_samples if patch_shape_zyx is not None else None,
    )
    cached_rows_by_tile = load_tile_phase_cache(cache_path, cache_key)
    strict_cached_attempts_by_tile = load_tile_phase_attempt_cache(cache_path, cache_key)
    compatible_phase_attempts_by_tile = load_compatible_tile_phase_attempt_cache(cache_path, cache_key)
    cached_attempts_by_tile: dict[str, TilePhaseRow] = {
        **compatible_phase_attempts_by_tile,
        **strict_cached_attempts_by_tile,
    }
    level_factor = 2**level
    rows: list[TilePhaseAcceptedRow] = []
    updated = json.loads(json.dumps(payload))
    updated["source"] = f"{payload.get('source', 'position file')} + tile phase alignment {moving_token} to {reference_token}"
    updated["derived_by"] = "lightsheet.tile_phase.v1"
    diagnostics = updated.setdefault("diagnostics", {})
    diagnostics["tile_phase_alignment"] = {
        "reference_position": str(reference_position.resolve()),
        "reference_channel": int(reference_channel),
        "moving_channel": int(moving_channel),
        "reference_token": reference_token,
        "moving_token": moving_token,
        "level": int(level),
        "level_factor": int(level_factor),
        "upsample_factor": int(upsample_factor),
        "patch_shape_zyx": None if patch_shape_zyx is None else [int(value) for value in patch_shape_zyx],
        "min_inliers": int(min_inliers),
        "max_candidate_patches": int(max_candidate_patches),
        "coarse_level": int(coarse_level),
        "scout_z_samples": None if scout_z_samples is None else int(scout_z_samples),
        "workers": int(workers),
        "description": "Per-tile moving z/y/x translations shifted after 3D phase correlation to reference channel tiles.",
    }
    failed_tiles: list[dict[str, Any]] = []
    successful_moving_tiles: list[tuple[rough_legacy.TileRecord, np.ndarray]] = []
    cached_successful_tiles: list[tuple[rough_legacy.TileRecord, Path, np.ndarray]] = []
    cached_failed_tiles: list[tuple[dict[str, Any], rough_legacy.TileRecord, Path, TilePhaseRow]] = []
    pending: dict[
        Future[TilePhaseRow],
        tuple[dict[str, Any], rough_legacy.TileRecord, rough_legacy.TileRecord, Path, str],
    ] = {}

    def handle_attempt_row(
        row: TilePhaseRow,
        *,
        record: dict[str, Any],
        reference_tile: rough_legacy.TileRecord,
        moving_tile: rough_legacy.TileRecord,
        moving_path: Path,
        moving_tile_name: str,
    ) -> None:
        cached_attempts_by_tile[moving_tile_name] = row
        if tile_phase_row_status(row) == "direct_accepted":
            accepted_row = _validate_tile_phase_accepted_row(dict(row))
            shift_um = apply_shift_row_to_position_record(record, accepted_row, moving_path=moving_path)
            successful_moving_tiles.append((moving_tile, shift_um))
            rows.append(accepted_row)
            _persist_tile_phase_cache(cache_path, cache_key, rows, cached_attempts_by_tile)
            if accepted_row.get("cache_source") == "cached_patch_residual_rescore":
                logger.info(
                    "tile-phase-cache-rescore {} -> {} shift_px={} n_inliers={}",
                    reference_tile.tile,
                    moving_path.name,
                    accepted_row["shift_px_zyx"],
                    accepted_row.get("n_inliers"),
                )
            else:
                logger.info(
                    "tile-phase {} -> {} shift_px={} corr_after={}",
                    reference_tile.tile,
                    moving_path.name,
                    accepted_row["shift_px_zyx"],
                    accepted_row["corr_after"],
                )
            return

        _persist_tile_phase_cache(cache_path, cache_key, rows, cached_attempts_by_tile)
        _append_failed_tile(
            failed_tiles,
            record=record,
            reference_tile=reference_tile,
            moving_tile=moving_tile,
            moving_path=moving_path,
            error=str(row.get("direct_error", "tile phase measurement failed")),
            direct_attempt=row,
        )

    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for record, reference_tile in zip(updated["tiles"], reference_tiles, strict=True):
            moving_path = corresponding_moving_path(reference_tile.path, reference_token=reference_token, moving_token=moving_token)
            moving_tile_name = make_moving_tile_name(record["tile"], reference_token=reference_token, moving_token=moving_token)
            cached_row = cached_rows_by_tile.get(moving_tile_name)
            if cached_row is not None:
                shift_um = apply_shift_row_to_position_record(record, cached_row, moving_path=moving_path)
                cached_successful_tiles.append((reference_tile, moving_path, shift_um))
                rows.append(cached_row)
                logger.info("tile-phase-cache {} -> {}", reference_tile.tile, moving_path.name)
                continue
            cached_attempt = strict_cached_attempts_by_tile.get(moving_tile_name)
            if cached_attempt is not None and cached_attempt.get("measurement_status") == "direct_failed":
                cached_failed_tiles.append((record, reference_tile, moving_path, cached_attempt))
                logger.info("tile-phase-direct-cache-failed {} -> {}", reference_tile.tile, moving_path.name)
                continue
            cached_phase_attempt = compatible_phase_attempts_by_tile.get(moving_tile_name)
            moving_tile = make_moving_tile_record(reference_tile, moving_path)
            if executor is None:
                row = _measure_tile_phase_attempt(
                    record=record,
                    reference_tile=reference_tile,
                    moving_tile=moving_tile,
                    moving_tile_name=moving_tile_name,
                    moving_path=moving_path,
                    reference_channel=reference_channel,
                    moving_channel=moving_channel,
                    level_factor=level_factor,
                    upsample_factor=upsample_factor,
                    patch_shape_zyx=patch_shape_zyx,
                    min_inliers=min_inliers,
                    max_candidate_patches=max_candidate_patches,
                    coarse_level=coarse_level,
                    cached_phase_attempt=cached_phase_attempt,
                    scout_z_samples=scout_z_samples if patch_shape_zyx is not None else None,
                )
                handle_attempt_row(
                    row,
                    record=record,
                    reference_tile=reference_tile,
                    moving_tile=moving_tile,
                    moving_path=moving_path,
                    moving_tile_name=moving_tile_name,
                )
                continue
            future = executor.submit(
                _measure_tile_phase_attempt,
                record=record,
                reference_tile=reference_tile,
                moving_tile=moving_tile,
                moving_tile_name=moving_tile_name,
                moving_path=moving_path,
                reference_channel=reference_channel,
                moving_channel=moving_channel,
                level_factor=level_factor,
                upsample_factor=upsample_factor,
                patch_shape_zyx=patch_shape_zyx,
                min_inliers=min_inliers,
                max_candidate_patches=max_candidate_patches,
                coarse_level=coarse_level,
                cached_phase_attempt=cached_phase_attempt,
                scout_z_samples=scout_z_samples if patch_shape_zyx is not None else None,
            )
            pending[future] = (record, reference_tile, moving_tile, moving_path, moving_tile_name)

        for future in as_completed(pending):
            record, reference_tile, moving_tile, moving_path, moving_tile_name = pending[future]
            handle_attempt_row(
                future.result(),
                record=record,
                reference_tile=reference_tile,
                moving_tile=moving_tile,
                moving_path=moving_path,
                moving_tile_name=moving_tile_name,
            )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    if cached_failed_tiles:
        for record, reference_tile, moving_path, cached_attempt in cached_failed_tiles:
            moving_tile = make_moving_tile_record(reference_tile, moving_path)
            _append_failed_tile(
                failed_tiles,
                record=record,
                reference_tile=reference_tile,
                moving_tile=moving_tile,
                moving_path=moving_path,
                error=str(cached_attempt.get("direct_error", "cached direct failure")),
                direct_attempt=cached_attempt,
            )
    if failed_tiles:
        for reference_tile, moving_path, shift_um in cached_successful_tiles:
            successful_moving_tiles.append((make_moving_tile_record(reference_tile, moving_path), shift_um))

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
                min_inliers=3,
            )
        except Exception as exc:
            failure["fallback_error"] = str(exc)
            direct_attempt = failure.get("direct_attempt")
            if direct_attempt is not None:
                direct_attempt["fallback_error"] = str(exc)
                cached_attempts_by_tile[str(direct_attempt["tile"])] = direct_attempt
                _persist_tile_phase_cache(cache_path, cache_key, rows, cached_attempts_by_tile)
            remaining_failed_tiles.append(failure)
            continue
        shift_um = shift_px * np.abs(reference_tile.scale_zyx_um)
        moving_tile_name = make_moving_tile_name(record["tile"], reference_token=reference_token, moving_token=moving_token)
        _apply_shift_to_position_record(
            record,
            moving_tile_name=moving_tile_name,
            moving_path=moving_path,
            shift_um=shift_um,
        )
        successful_moving_tiles.append((moving_tile, shift_um))
        row = _build_fallback_measurement_row(
            record=record,
            reference_tile=reference_tile,
            moving_path=moving_path,
            shift_px=shift_px,
            shift_um=shift_um,
            details=details,
            direct_error=failure["error"],
        )
        rows.append(row)
        cached_attempts_by_tile[record["tile"]] = row
        _persist_tile_phase_cache(cache_path, cache_key, rows, cached_attempts_by_tile)
        logger.info(
            "tile-phase-fallback {} -> {} shift_px={} n_inliers={}",
            reference_tile.tile,
            moving_path.name,
            shift_px.tolist(),
            details.get("n_inliers"),
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
        "moving_channel": int(moving_channel),
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
    if patch_shape_zyx is not None:
        contact_sheet_paths = write_tile_phase_contact_sheets(
            rows=rows,
            reference_tiles=reference_tiles,
            output_dir=output_dir,
            reference_channel=reference_channel,
            moving_channel=moving_channel,
        )
        summary["contact_sheets"] = [str(path.resolve()) for path in contact_sheet_paths]
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("wrote {} tile-phase contact sheets to {}", len(contact_sheet_paths), output_dir / "contact_sheets")
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
