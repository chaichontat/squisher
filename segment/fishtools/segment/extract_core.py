from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import click
import numpy as np
import tifffile
import zarr
from loguru import logger
from scipy.ndimage import zoom

from fishtools.segment.extract_support import unsharp_all
from fishtools.segment.extract_helpers import (
    DEFAULT_CROP_SIZE,
    MAX_WIDTH_AFTER_UPSCALE,
    ZARR_TILE_SIZE,
    ZERO_PIXEL_SKIP_THRESHOLD,
    ArrayLike,
    ExtractionConfig,
    MaskLike,
    SliceJob,
    TileJob,
    TIFF_KWARGS,
    Volume,
    _compute_perpendicular_slice,
    _compute_tile_origins,
    _expand_positions_with_context,
    _format_size,
    _format_tile_maxproj_filename,
    _format_tile_filename,
    _mask_filename,
    _normalize_reporter,
    _path_size_cached,
    _prefix_with_roi,
    _resize_mask,
    _resolve_output_names,
    _sample_positions,
    _score_and_select_tiles,
    _select_high_diversity_positions,
    _squeeze_mask,
    _validate_max_from_path,
    _write_mask_tiff,
    _write_tiff,
    load_roi_points,
)
from fishtools.segment.extract_support import (
    ProgressReporter,
    TaskCancelledException,
    get_cancel_event,
    progress_bar_threadpool,
    progress_reporter,
)

"""Extract utilities for the segment CLI.

This module provides the core extraction logic. The top-level `segment` CLI
in `__init__.py` exposes these functions as Click commands.
"""


@dataclass
class ExtractionContext:
    """Precomputed per-file state for extraction operations."""

    file: Path
    roi: str
    vol: Volume
    channel_names: list[str] | None
    selected_indices: list[int]
    out_names: list[str]
    other_vol: Volume | None
    mask_vol: MaskLike | None
    enrich_mask_vol: MaskLike | None
    upscale: float
    anisotropy: int
    out_dir: Path


@dataclass(frozen=True)
class OrthoSliceRequest:
    axis: Literal["y", "x"]
    position: int
    perpendicular_slice: slice
    out_file: Path
    axes: str
    z_slice: slice | None = None


@dataclass
class OrthoStripCache:
    request: OrthoSliceRequest
    channel_strips: list[np.ndarray]
    mask_strip: np.ndarray | None = None


@dataclass(frozen=True)
class RandomContentCrop:
    z_index: int
    y0: int
    x0: int


@dataclass(frozen=True)
class ContentEstimationVolume:
    vol: Volume
    scale_zyx: tuple[float, float, float]


def resolve_file_mask_path(file: Path, explicit_mask: Path | None) -> Path | None:
    """Resolve mask path for a single input file."""

    if explicit_mask is not None:
        return explicit_mask
    return _resolve_mask_path(file)


def open_and_validate_mask(mask_path: Path | None, vol: Volume, *, label: str) -> MaskLike | None:
    """Open a mask volume and validate that its shape matches the input volume.

    For TIFF/Zarr inputs we expect:
    - volume: (Z, Y, X, C)
    - mask:   (Z, Y, X)  or (Z, 1, Y, X) which is squeezed to (Z, Y, X)

    Any mismatch in Z, Y, or X dimensions is treated as an error so that
    downstream slices/tiles are guaranteed to be spatially aligned.
    """

    if mask_path is None:
        return None

    mask_vol = _open_mask_volume(mask_path)
    if mask_vol.shape[0] != vol.shape[0]:
        raise ValueError(
            f"Mask volume {mask_path} Z-dimension ({mask_vol.shape[0]}) does not match volume ({vol.shape[0]}) for {label}"
        )
    # Spatial YX dimensions must also agree; we do not attempt to resample.
    if mask_vol.shape[1] != vol.shape[1] or mask_vol.shape[2] != vol.shape[2]:
        raise ValueError(
            f"Mask volume {mask_path} spatial dimensions {mask_vol.shape[1:]} "
            f"do not match volume ({vol.shape[1]}, {vol.shape[2]}) for {label}"
        )

    logger.info(f"[{label}] Using mask: {mask_path}")
    return mask_vol


def build_extraction_context(
    file: Path,
    roi: str,
    *,
    channels: str | None,
    max_from_path: Path | None,
    mask_path: Path | None,
    enrich_path: Path | None,
    upscale: float,
    anisotropy: int,
    out_dir: Path,
) -> ExtractionContext:
    vol, channel_names = _open_volume(file)
    selected_indices = _parse_channels(channels, channel_names, vol.shape[-1])
    _ensure_channel_bounds(selected_indices, vol.shape[-1], label=file.name)
    base_names = _resolve_output_names(selected_indices, channel_names, channels)
    other_vol = _resolve_other_volume(file, max_from_path)
    out_names = [*base_names, "max_from"] if other_vol is not None else base_names
    mask_vol = open_and_validate_mask(mask_path, vol, label=file.name)
    enrich_mask_vol = open_and_validate_mask(enrich_path, vol, label=f"{file.name} (enrich)")

    return ExtractionContext(
        file=file,
        roi=roi,
        vol=vol,
        channel_names=channel_names,
        selected_indices=selected_indices,
        out_names=out_names,
        other_vol=other_vol,
        mask_vol=mask_vol,
        enrich_mask_vol=enrich_mask_vol,
        upscale=upscale,
        anisotropy=anisotropy,
        out_dir=out_dir,
    )


def write_slab_with_mask(
    *,
    ctx: ExtractionContext,
    slab: np.ndarray,
    other_max: np.ndarray | None,
    mask_slice: np.ndarray | None,
    out_file: Path,
    axes: str,
    resize_factors: tuple[float, ...],
    channels_arg: str | None,
) -> None:
    """Shared slab processing and writing logic for extracted slices."""

    processed = _prep_slab(
        slab,
        ch_idx=ctx.selected_indices,
        channel_axis=slab.ndim - 1,
        crop_slices=None,
        filter_before=False,
        append_max=other_max,
        apply_filter=False,
    )
    processed = _resize_uint16(processed, resize_factors)
    _write_tiff(
        out_file,
        processed,
        axes=axes,
        names=ctx.out_names,
        channels_arg=channels_arg,
        upscale=ctx.upscale,
    )

    if mask_slice is not None:
        if mask_slice.ndim != 2:
            raise ValueError("Mask extraction expected 2D data.")
        mask_factors = resize_factors[1:] if len(resize_factors) > 2 else resize_factors
        resized_mask = _resize_mask(mask_slice, mask_factors)
        _write_mask_tiff(out_file.parent / _mask_filename(out_file.name), resized_mask, axes=axes[1:])


def _read_channel_names(file: Path) -> list[str] | None:
    """Extract channel names from TIFF metadata, returning None if unavailable."""
    try:
        with tifffile.TiffFile(file) as tif:
            shaped = getattr(tif, "shaped_metadata", None)
            if isinstance(shaped, (list, tuple)) and shaped:
                md = shaped[0]
                if isinstance(md, dict):
                    names = md.get("key")
                    if isinstance(names, (list, tuple)):
                        return [str(x) for x in names]
    except (OSError, ValueError, KeyError, tifffile.TiffFileError) as exc:
        logger.debug(f"Failed to read channel names from {file}: {exc}")
    return None


def _normalize_channel_names(names: object) -> list[str] | None:
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return None


def _read_channel_names_from_zarr(store: Path) -> list[str] | None:
    try:
        arr = _open_zarr_array(store)
    except (OSError, ValueError, Exception) as exc:  # pragma: no cover - defensive
        logger.debug(f"Failed to open Zarr store {store} for channel names: {exc}")
        return None
    raw_key = arr.attrs.get("key")
    return _normalize_channel_names(raw_key)


def _is_zarr_path(file: Path) -> bool:
    return file.suffix == ".zarr" or (file.is_dir() and file.name.endswith(".zarr"))


def _open_zarr_array(file: Path) -> zarr.Array:
    if _is_ome_zarr_path(file):
        return _open_ome_zarr_level_array(file)
    return zarr.open_array(file, mode="r")


def _is_ome_zarr_path(file: Path) -> bool:
    return file.name.endswith(".ome.zarr")


def _attrs_dict(attrs: Any) -> dict[str, Any]:
    if hasattr(attrs, "asdict"):
        return dict(attrs.asdict())
    return dict(attrs)


def _ome_zarr_dataset_paths(file: Path) -> list[str]:
    group = zarr.open_group(file, mode="r")
    attrs = _attrs_dict(group.attrs)
    multiscales = attrs.get("multiscales")
    if multiscales is None and isinstance(attrs.get("ome"), dict):
        multiscales = attrs["ome"].get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError(f"OME-Zarr store {file} is missing multiscales metadata.")

    datasets = multiscales[0].get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"OME-Zarr store {file} does not list any multiscale datasets.")

    paths: list[str] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError(f"OME-Zarr store {file} has an invalid dataset entry.")
        level_path = dataset.get("path")
        if not isinstance(level_path, str) or not level_path:
            raise ValueError(f"OME-Zarr store {file} has an invalid dataset path.")
        paths.append(level_path)
    return paths


def _open_ome_zarr_level_array(file: Path, *, level_index: int = 0) -> zarr.Array:
    paths = _ome_zarr_dataset_paths(file)
    try:
        level_path = paths[level_index]
    except IndexError as error:
        raise ValueError(f"OME-Zarr store {file} does not have level index {level_index}.") from error
    if not isinstance(level_path, str) or not level_path:
        raise ValueError(f"OME-Zarr store {file} has an invalid first dataset path.")
    return zarr.open_array(file / level_path, mode="r")


def _open_ome_zarr_highest_level_array(file: Path) -> zarr.Array:
    return _open_ome_zarr_level_array(file, level_index=-1)


def _zarr_array_axes(array: zarr.Array) -> str | None:
    dimensions = array.attrs.get("_ARRAY_DIMENSIONS")
    if isinstance(dimensions, list):
        return "".join(str(axis).upper() for axis in dimensions)
    return None


class SingleChannelZarrVolume:
    """Expose a ZYX Zarr array as the extraction ZYXC contract without changing read windows."""

    def __init__(self, array: zarr.Array) -> None:
        self.array = array
        self.dtype = array.dtype
        self.ndim = 4
        self.shape = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]), 1)

    def __getitem__(self, key: Any) -> Any:
        z_selector, y_selector, x_selector, c_selector = _normalize_array_key(key, self.ndim)
        _c_indices, c_block_selector = _selector_indices(c_selector, 1)
        block = np.asarray(self.array[z_selector, y_selector, x_selector])
        return np.expand_dims(block, axis=-1)[..., c_block_selector]


def _normalize_array_key(key: object, ndim: int) -> tuple[object, ...]:
    if not isinstance(key, tuple):
        key = (key,)

    parts: list[object] = []
    ellipsis_seen = False
    for item in key:
        if item is Ellipsis:
            if ellipsis_seen:
                raise IndexError("an index can only have a single ellipsis")
            ellipsis_seen = True
            missing = ndim - (len(key) - 1)
            parts.extend([slice(None)] * missing)
        else:
            if item is None:
                raise IndexError("newaxis indexing is not supported for lazy TIFF volumes")
            parts.append(item)

    if len(parts) > ndim:
        raise IndexError(f"too many indices for lazy TIFF volume: {len(parts)} for ndim {ndim}")
    parts.extend([slice(None)] * (ndim - len(parts)))
    return tuple(parts)


def _selector_indices(selector: object, length: int) -> tuple[list[int], object]:
    if isinstance(selector, slice):
        return list(range(*selector.indices(length))), slice(None)
    if isinstance(selector, np.integer | int):
        index = int(selector)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"index {selector} is out of bounds for axis with size {length}")
        return [index], 0
    if isinstance(selector, list | tuple | np.ndarray):
        indices: list[int] = []
        for raw_index in selector:
            index = int(raw_index)
            if index < 0:
                index += length
            if index < 0 or index >= length:
                raise IndexError(f"index {raw_index} is out of bounds for axis with size {length}")
            indices.append(index)
        return indices, list(range(len(indices)))
    raise IndexError(f"unsupported lazy TIFF index selector {selector!r}")


class SingleChannelMaskArray:
    """Expose a (Z,1,Y,X) array-like mask as (Z,Y,X)."""

    def __init__(self, array: ArrayLike) -> None:
        if array.ndim != 4 or array.shape[1] != 1:
            raise ValueError("Mask array with 4D shape must be (Z,1,Y,X).")
        self.array = array
        self.shape = (int(array.shape[0]), int(array.shape[2]), int(array.shape[3]))
        self.ndim = 3

    def __getitem__(self, key: Any) -> Any:
        z_selector, y_selector, x_selector = _normalize_array_key(key, self.ndim)
        return self.array[z_selector, 0, y_selector, x_selector]


class LazyTiffArray:
    """Array-like TIFF reader that normalizes explicit TIFF axes to ZYXC or ZYX."""

    def __init__(self, file: Path, *, kind: Literal["volume", "mask"]) -> None:
        self.file = file
        self.kind = kind
        self._tif = tifffile.TiffFile(file)
        self._series = self._tif.series[0]
        self._axes = self._series.axes
        self.dtype = self._series.dtype
        self._source_shape = tuple(int(axis) for axis in self._series.shape)
        self._page_axes = self._axes[:-2]
        self._page_shape = self._source_shape[:-2]
        self._z_axis: int | None = None
        self._c_axis: int | None = None
        self.shape = self._normalized_shape()
        self.ndim = len(self.shape)

    def _fail_axes(self, message: str) -> None:
        self.close()
        raise ValueError(f"{message} Got axes={self._axes!r}, shape={self._source_shape}.")

    def _validate_spatial_axes(self) -> tuple[int, int]:
        if len(self._source_shape) < 2 or self._axes[-2:] != "YX":
            self._fail_axes("TIFF series must end in YX spatial axes.")
        return self._source_shape[-2], self._source_shape[-1]

    def _configure_nonspatial_axes(self, *, require_z: bool, allow_singleton_c: bool) -> tuple[int, int]:
        z_axes = [i for i, axis in enumerate(self._page_axes) if axis in {"Z", "Q"}]
        c_axes = [i for i, axis in enumerate(self._page_axes) if axis == "C"]
        if len(z_axes) > 1 or len(c_axes) > 1:
            self._fail_axes("TIFF series has duplicate Z/Q or C axes.")
        if require_z and not z_axes:
            self._fail_axes("Registered TIFF volume requires a Z axis.")

        self._z_axis = z_axes[0] if z_axes else None
        self._c_axis = c_axes[0] if c_axes else None
        for axis_index, (axis, size) in enumerate(zip(self._page_axes, self._page_shape)):
            if axis_index == self._z_axis:
                continue
            if axis_index == self._c_axis:
                if allow_singleton_c and size != 1:
                    self._fail_axes("Mask TIFF with a C axis must have a single channel.")
                continue
            if size != 1:
                self._fail_axes(f"Unsupported non-singleton TIFF axis {axis!r}.")

        z_len = self._page_shape[self._z_axis] if self._z_axis is not None else 1
        c_len = self._page_shape[self._c_axis] if self._c_axis is not None else 1
        return int(z_len), int(c_len)

    def _normalized_shape(self) -> tuple[int, ...]:
        y_len, x_len = self._validate_spatial_axes()
        if self.kind == "volume":
            z_len, c_len = self._configure_nonspatial_axes(require_z=True, allow_singleton_c=False)
            return (z_len, y_len, x_len, c_len)

        z_len, c_len = self._configure_nonspatial_axes(require_z=False, allow_singleton_c=True)
        if c_len != 1:
            self._fail_axes("Mask TIFF with a C axis must have a single channel.")
        return (z_len, y_len, x_len)

    def close(self) -> None:
        self._tif.close()

    def __del__(self) -> None:
        self.close()

    def _page_index(self, *, z_index: int, c_index: int) -> int:
        if not self._page_shape:
            return 0
        indices: list[int] = []
        for axis_index, _axis in enumerate(self._page_axes):
            if axis_index == self._z_axis:
                indices.append(z_index)
            elif axis_index == self._c_axis:
                indices.append(c_index)
            else:
                indices.append(0)
        return int(np.ravel_multi_index(tuple(indices), self._page_shape))

    def _read_page(self, page_index: int) -> np.ndarray:
        page = np.asarray(self._series.asarray(key=page_index))
        expected_shape = (self.shape[1], self.shape[2])
        if page.size != math.prod(expected_shape):
            raise ValueError(f"Expected TIFF page to contain YX shape {expected_shape}, got shape {page.shape}.")
        return page.reshape(expected_shape)

    def _read_volume_block(self, z_indices: list[int], c_indices: list[int]) -> np.ndarray:
        block = np.empty((len(z_indices), self.shape[1], self.shape[2], len(c_indices)), dtype=self.dtype)
        for out_z, z_index in enumerate(z_indices):
            for out_c, c_index in enumerate(c_indices):
                block[out_z, :, :, out_c] = self._read_page(
                    self._page_index(z_index=z_index, c_index=c_index)
                )
        return block

    def _read_mask_block(self, z_indices: list[int]) -> np.ndarray:
        block = np.empty((len(z_indices), self.shape[1], self.shape[2]), dtype=self.dtype)
        for out_z, z_index in enumerate(z_indices):
            block[out_z] = self._read_page(self._page_index(z_index=z_index, c_index=0))
        return block

    def __getitem__(self, key: Any) -> np.ndarray:
        key_tuple = _normalize_array_key(key, self.ndim)
        z_indices, z_block_selector = _selector_indices(key_tuple[0], self.shape[0])

        if self.kind == "volume":
            y_selector, x_selector, c_selector = key_tuple[1], key_tuple[2], key_tuple[3]
            c_indices, c_block_selector = _selector_indices(c_selector, self.shape[3])
            block = self._read_volume_block(z_indices, c_indices)
            return block[z_block_selector, y_selector, x_selector, c_block_selector]

        y_selector, x_selector = key_tuple[1], key_tuple[2]
        block = self._read_mask_block(z_indices)
        return block[z_block_selector, y_selector, x_selector]


class ZarrBackedTiffVolume:
    """Expose a TIFF Zarr view as the extraction ZYXC contract with windowed reads."""

    def __init__(self, file: Path) -> None:
        self.file = file
        with tifffile.TiffFile(file) as tif:
            series = tif.series[0]
            self._axes = series.axes
            self.dtype = series.dtype
            self._source_shape = tuple(int(axis) for axis in series.shape)

        self._store = tifffile.imread(file, aszarr=True, level=0)
        source_array = zarr.open(self._store, mode="r")
        if hasattr(source_array, "keys") and "0" in source_array:
            source_array = source_array["0"]
        self.array = source_array
        self._z_axis: int | None = None
        self._c_axis: int | None = None
        self.shape = self._normalized_shape()
        self.ndim = 4

    def _fail_axes(self, message: str) -> None:
        self.close()
        raise ValueError(f"{message} Got axes={self._axes!r}, shape={self._source_shape}.")

    def _normalized_shape(self) -> tuple[int, int, int, int]:
        if len(self._source_shape) < 2 or self._axes[-2:] != "YX":
            self._fail_axes("TIFF series must end in YX spatial axes.")
        if tuple(self.array.shape) != self._source_shape:
            self._fail_axes(f"TIFF Zarr view shape {tuple(self.array.shape)} does not match series metadata.")

        page_axes = self._axes[:-2]
        page_shape = self._source_shape[:-2]
        z_axes = [i for i, axis in enumerate(page_axes) if axis in {"Z", "Q"}]
        c_axes = [i for i, axis in enumerate(page_axes) if axis == "C"]
        if len(z_axes) > 1 or len(c_axes) > 1:
            self._fail_axes("TIFF series has duplicate Z/Q or C axes.")
        if not z_axes:
            self._fail_axes("Registered TIFF volume requires a Z axis.")
        self._z_axis = z_axes[0]
        self._c_axis = c_axes[0] if c_axes else None

        for axis_index, (axis, size) in enumerate(zip(page_axes, page_shape)):
            if axis_index in {self._z_axis, self._c_axis}:
                continue
            if size != 1:
                self._fail_axes(f"Unsupported non-singleton TIFF axis {axis!r}.")

        z_len = int(page_shape[self._z_axis])
        c_len = int(page_shape[self._c_axis]) if self._c_axis is not None else 1
        return (z_len, int(self._source_shape[-2]), int(self._source_shape[-1]), c_len)

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            close()

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, key: Any) -> np.ndarray:
        z_selector, y_selector, x_selector, c_selector = _normalize_array_key(key, self.ndim)
        _selector_indices(c_selector, self.shape[3])

        selectors_by_normalized_axis = {
            "Z": z_selector,
            "Y": y_selector,
            "X": x_selector,
            "C": c_selector,
        }
        source_key: list[object] = []
        remaining_source_axes: list[str] = []
        has_source_c = False

        for source_axis, axis_size in zip(self._axes, self._source_shape):
            normalized_axis = "Z" if source_axis == "Q" else source_axis
            if normalized_axis in selectors_by_normalized_axis:
                selector = selectors_by_normalized_axis[normalized_axis]
                source_key.append(selector)
                if normalized_axis == "C":
                    has_source_c = True
                if not isinstance(selector, np.integer | int):
                    remaining_source_axes.append(normalized_axis)
            else:
                if axis_size != 1:
                    self._fail_axes(f"Unsupported non-singleton TIFF axis {source_axis!r}.")
                source_key.append(0)

        block = np.asarray(self.array[tuple(source_key)])
        remaining_normalized_axes = [
            axis
            for axis, selector in (("Z", z_selector), ("Y", y_selector), ("X", x_selector), ("C", c_selector))
            if not isinstance(selector, np.integer | int) and (axis != "C" or has_source_c)
        ]
        if remaining_source_axes != remaining_normalized_axes:
            transpose_order = [remaining_source_axes.index(axis) for axis in remaining_normalized_axes]
            block = np.transpose(block, transpose_order)

        if not has_source_c and not isinstance(c_selector, np.integer | int):
            block = np.expand_dims(block, axis=-1)
        return block


def _open_volume(file: Path) -> tuple[Volume, list[str] | None]:
    """Open a registered volume as an array-like object with shape (Z,Y,X,C).

    - TIFF: opened lazily by Z/C page; explicit axes are normalized to ZYXC.
    - Zarr: opened directly without materialization.
    Returns: (volume, channel_names)
    """
    if _is_zarr_path(file):
        arr = _open_zarr_array(file)
        axes = _zarr_array_axes(arr)
        if arr.ndim == 3 and axes in {None, "ZYX"}:
            return SingleChannelZarrVolume(arr), _read_channel_names_from_zarr(file)
        if arr.ndim != 4:
            raise ValueError("Expected fused Zarr to be 4D (Z,Y,X,C) or OME-Zarr ZYX.")
        return arr, _read_channel_names_from_zarr(file)
    return LazyTiffArray(file, kind="volume"), _read_channel_names(file)


def _open_aux_channel_volumes(aux_channel_stack: Path | None, primary_vol: Volume, *, label: str) -> tuple[list[Volume], list[str]]:
    if aux_channel_stack is None:
        return [], []

    aux_vol, aux_names = _open_aux_channel_volume(aux_channel_stack)
    if aux_vol.shape[:3] != primary_vol.shape[:3]:
        raise ValueError(
            f"Auxiliary channel stack {aux_channel_stack} Z/Y/X shape {aux_vol.shape[:3]} "
            f"does not match primary volume shape {primary_vol.shape[:3]} for {label}."
        )

    channel_count = int(aux_vol.shape[-1])
    names = _resolve_aux_channel_names(aux_names, channel_count, aux_channel_stack)
    logger.info(f"[{label}] Appending {channel_count} channel(s) from auxiliary stack: {aux_channel_stack}")
    return [aux_vol], names


def _open_aux_channel_volume(file: Path) -> tuple[Volume, list[str] | None]:
    if _is_zarr_path(file):
        return _open_volume(file)
    if file.suffix.lower() in {".tif", ".tiff"}:
        return ZarrBackedTiffVolume(file), _read_channel_names(file)
    return _open_volume(file)


def _resolve_aux_channel_names(names: list[str] | None, channel_count: int, aux_channel_stack: Path) -> list[str]:
    if names and len(names) >= channel_count:
        return [str(name) for name in names[:channel_count]]
    return [f"{aux_channel_stack.stem}:{channel}" for channel in range(channel_count)]


def _prep_aux_channel_slab(slab: np.ndarray, *, channel_axis: int) -> np.ndarray:
    channel_count = int(slab.shape[channel_axis])
    return _prep_slab(
        slab,
        ch_idx=list(range(channel_count)),
        channel_axis=channel_axis,
        crop_slices=None,
        filter_before=False,
        append_max=None,
        apply_filter=False,
    )


def _append_aux_channel_slabs(
    base_c_first: np.ndarray,
    aux_slabs: list[np.ndarray],
    *,
    channel_axis: int,
) -> np.ndarray:
    if not aux_slabs:
        return base_c_first
    aux_channels = [_prep_aux_channel_slab(slab, channel_axis=channel_axis) for slab in aux_slabs]
    return np.concatenate([base_c_first, *aux_channels], axis=0)


def _resolve_mask_path(file: Path) -> Path | None:
    """Return the expected mask path for a given registered input, if it exists."""
    candidate: Path | None = None
    if _is_zarr_path(file):
        candidate = file.parent / f"{file.stem}_masks.zarr"
    elif file.suffix.lower() in {".tif", ".tiff"}:
        candidate = file.with_name(f"{file.stem}_masks{file.suffix}")
    if candidate is not None and candidate.exists():
        return candidate
    return None


def _open_mask_volume(mask_path: Path) -> MaskLike:
    """Load a mask volume produced alongside a registered stack."""
    if _is_zarr_path(mask_path):
        arr = _open_zarr_array(mask_path)
        if arr.ndim == 3:
            return arr
        if arr.ndim == 4:
            return SingleChannelMaskArray(arr)
        raise ValueError("Mask Zarr is expected to be 3D (Z,Y,X) or 4D (Z,1,Y,X).")
    return LazyTiffArray(mask_path, kind="mask")


def _ortho_resize_shape(shape: tuple[int, int], factors: tuple[float, float]) -> tuple[int, int]:
    return tuple(int(round(size * factor)) for size, factor in zip(shape, factors))


def _ortho_output_request(
    *,
    file_stem: str,
    roi: str,
    out_dir: Path,
    axis: Literal["y", "x"],
    position: int,
    perpendicular_slice: slice,
    z_slice: slice | None = None,
) -> OrthoSliceRequest:
    perp_start = perpendicular_slice.start or 0
    z_prefix = "" if z_slice is None else f"-z{z_slice.start or 0}"
    if axis == "y":
        out_name = _prefix_with_roi(f"{file_stem}_orthozx{z_prefix}-x{perp_start}-y{position}.tif", roi)
        return OrthoSliceRequest(
            axis=axis,
            position=position,
            perpendicular_slice=perpendicular_slice,
            out_file=out_dir / out_name,
            axes="CZX",
            z_slice=z_slice,
        )

    out_name = _prefix_with_roi(f"{file_stem}_orthozy{z_prefix}-y{perp_start}-x{position}.tif", roi)
    return OrthoSliceRequest(
        axis=axis,
        position=position,
        perpendicular_slice=perpendicular_slice,
        out_file=out_dir / out_name,
        axes="CZY",
        z_slice=z_slice,
    )


def _resize_ortho_strip_uint16(strip: np.ndarray, resize_factors: tuple[float, float]) -> np.ndarray:
    resized = zoom(strip, resize_factors, order=1)
    return np.clip(np.rint(resized), 0, 65530).astype(np.uint16)


def _write_cached_ortho_tiff(
    *,
    cache: OrthoStripCache,
    out_names: list[str],
    channels_arg: str | None,
    upscale: float,
    resize_factors: tuple[float, float],
) -> None:
    strip_shape = cache.channel_strips[0].shape
    resized_shape = _ortho_resize_shape(strip_shape, resize_factors)

    def iter_channels():
        for strip in cache.channel_strips:
            yield _resize_ortho_strip_uint16(strip, resize_factors)

    tiff_kwargs = dict(TIFF_KWARGS)
    tiff_kwargs.pop("planarconfig", None)
    with tifffile.TiffWriter(cache.request.out_file) as writer:
        writer.write(
            iter_channels(),
            shape=(len(cache.channel_strips), *resized_shape),
            dtype=np.uint16,
            metadata={
                "axes": cache.request.axes,
                "channel_names": out_names,
                "channels_arg": channels_arg,
                "upscale": upscale,
            },
            **tiff_kwargs,
        )

    if cache.mask_strip is not None:
        resized_mask = zoom(cache.mask_strip, resize_factors, order=0)
        _write_mask_tiff(
            cache.request.out_file.parent / _mask_filename(cache.request.out_file.name),
            resized_mask.astype(cache.mask_strip.dtype, copy=False),
            axes=cache.request.axes[1:],
        )


def _fill_requested_ortho_strips(
    *,
    vol: Volume,
    mask_vol: MaskLike | None,
    other_vol: Volume | None,
    requests: list[OrthoSliceRequest],
    selected_indices: list[int],
    aux_channel_vols: list[Volume] | None = None,
) -> list[OrthoStripCache]:
    z_len = int(vol.shape[0])
    channel_sources: list[tuple[Volume, int | None]] = [(vol, channel) for channel in selected_indices]
    for aux_vol in aux_channel_vols or []:
        channel_sources.extend((aux_vol, channel) for channel in range(int(aux_vol.shape[-1])))
    if other_vol is not None:
        channel_sources.append((other_vol, None))

    z_step = _ortho_z_batch_size([source for source, _channel in channel_sources], mask_vol)
    caches: list[OrthoStripCache] = []
    for request in requests:
        z_start, z_stop = _ortho_request_z_bounds(request, z_len)
        width = int((request.perpendicular_slice.stop or 0) - (request.perpendicular_slice.start or 0))
        caches.append(
            OrthoStripCache(
                request=request,
                channel_strips=[np.empty((z_stop - z_start, width), dtype=np.float32) for _ in channel_sources],
            )
        )

    cancel_event = get_cancel_event()
    for cache in caches:
        request = cache.request
        request_z_start, request_z_stop = _ortho_request_z_bounds(request, z_len)
        for z_start in range(request_z_start, request_z_stop, z_step):
            if cancel_event.is_set():
                raise TaskCancelledException("Cancelled by user")
            z_stop = min(z_start + z_step, request_z_stop)
            z_slice = slice(z_start, z_stop)
            out_z_slice = slice(z_start - request_z_start, z_stop - request_z_start)

            for source_idx, (volume, channel_index) in enumerate(channel_sources):
                if request.axis == "y":
                    if channel_index is None:
                        slab = np.asarray(volume[z_slice, request.position, request.perpendicular_slice, :])
                        cache.channel_strips[source_idx][out_z_slice] = slab.max(axis=-1)
                    else:
                        cache.channel_strips[source_idx][out_z_slice] = np.asarray(
                            volume[z_slice, request.position, request.perpendicular_slice, channel_index]
                        )
                else:
                    if channel_index is None:
                        slab = np.asarray(volume[z_slice, request.perpendicular_slice, request.position, :])
                        cache.channel_strips[source_idx][out_z_slice] = slab.max(axis=-1)
                    else:
                        cache.channel_strips[source_idx][out_z_slice] = np.asarray(
                            volume[z_slice, request.perpendicular_slice, request.position, channel_index]
                        )

            if mask_vol is not None:
                if cache.mask_strip is None:
                    cache.mask_strip = np.empty(
                        cache.channel_strips[0].shape,
                        dtype=np.dtype(getattr(mask_vol, "dtype", np.uint16)),
                    )
                if request.axis == "y":
                    cache.mask_strip[out_z_slice] = _squeeze_mask(
                        mask_vol[z_slice, request.position, request.perpendicular_slice]
                    )
                else:
                    cache.mask_strip[out_z_slice] = _squeeze_mask(
                        mask_vol[z_slice, request.perpendicular_slice, request.position]
                    )
    return caches


def _ortho_request_z_bounds(request: OrthoSliceRequest, z_len: int) -> tuple[int, int]:
    if request.z_slice is None:
        return 0, z_len
    start, stop, step = request.z_slice.indices(z_len)
    if step != 1:
        raise ValueError("Ortho Z crop slice must have step 1.")
    if stop <= start:
        raise ValueError("Ortho Z crop slice is empty.")
    return start, stop


Z_CONTEXT_PAIRS = 25


def _z_crop_slice_around(center: int, *, z_len: int, context_pairs: int = Z_CONTEXT_PAIRS) -> slice:
    start = max(0, center - context_pairs)
    stop = min(z_len, center + context_pairs + 1)
    return slice(start, stop)


def _z_candidates_around(
    center: int,
    *,
    z_len: int,
    dz: int,
    context_pairs: int = Z_CONTEXT_PAIRS,
) -> list[int]:
    if dz < 1:
        raise ValueError("--dz must be a positive integer.")
    candidates = [center] if 0 <= center < z_len else []
    for offset in range(dz, context_pairs + 1, dz):
        for z_index in (center - offset, center + offset):
            if 0 <= z_index < z_len:
                candidates.append(z_index)
    candidates.sort()
    if not candidates:
        raise ValueError("Z crop candidate window is empty.")
    return candidates


def _has_too_many_zero_pixels(tile: np.ndarray) -> bool:
    return float(np.mean(tile == 0)) > ZERO_PIXEL_SKIP_THRESHOLD


def _contentful_z_candidates_around(
    *,
    vol: Volume,
    center: int,
    y0: int,
    x0: int,
    z_len: int,
    dz: int,
) -> list[int]:
    target_count = len(_z_candidates_around(center, z_len=z_len, dz=dz))
    y_slice = slice(y0, y0 + ZARR_TILE_SIZE)
    x_slice = slice(x0, x0 + ZARR_TILE_SIZE)
    z_candidates: list[int] = []
    seen: set[int] = set()
    max_offset = max(center, z_len - 1 - center)

    for offset in [0, *range(dz, max_offset + 1, dz)]:
        candidates = [center] if offset == 0 else [center - offset, center + offset]
        for z_index in candidates:
            if z_index in seen or not 0 <= z_index < z_len:
                continue
            seen.add(z_index)
            if _has_too_many_zero_pixels(np.asarray(vol[z_index, y_slice, x_slice, :])):
                continue
            z_candidates.append(z_index)
            if len(z_candidates) >= target_count:
                return sorted(z_candidates)

    return sorted(z_candidates)


def _ortho_z_batch_size(volumes: list[Volume], mask_vol: MaskLike | None) -> int:
    chunk_depths: list[int] = []
    for volume in [*volumes, *([mask_vol] if mask_vol is not None else [])]:
        source = getattr(volume, "array", volume)
        chunks = getattr(source, "chunks", None)
        if isinstance(chunks, tuple | list) and chunks:
            chunk_depths.append(max(1, int(chunks[0])))
    return min(chunk_depths) if chunk_depths else 1


def _write_requested_ortho_slices(
    *,
    vol: Volume,
    mask_vol: MaskLike | None,
    other_vol: Volume | None,
    requests: list[OrthoSliceRequest],
    selected_indices: list[int],
    out_names: list[str],
    anisotropy: int,
    upscale: float,
    channels_arg: str | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
    aux_channel_vols: list[Volume] | None = None,
) -> None:
    if not requests:
        return

    out_dir = requests[0].out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    resize_factors = (anisotropy * upscale, upscale)
    reporter = _normalize_reporter(progress)
    caches = _fill_requested_ortho_strips(
        vol=vol,
        mask_vol=mask_vol,
        other_vol=other_vol,
        aux_channel_vols=aux_channel_vols,
        requests=requests,
        selected_indices=selected_indices,
    )
    for cache in caches:
        _write_cached_ortho_tiff(
            cache=cache,
            out_names=out_names,
            channels_arg=channels_arg,
            upscale=upscale,
            resize_factors=resize_factors,
        )
        if reporter is not None:
            reporter.advance()


def _execute_extraction(
    *,
    label: str,
    files: list[Path],
    mode: str,
    out_dir: Path,
    channels: str | None,
    crop: int,
    dz: int,
    n: int,
    z_crops_per_file: int = 1,
    anisotropy: int,
    threads: int,
    upscale: float,
    seed: int | None,
    max_from_path: Path | None,
    aux_channel_stack: Path | None = None,
    explicit_mask_path: Path | None = None,
    enrich_boundaries: Path | None = None,
    roi_points: Path | None = None,
) -> None:
    config = ExtractionConfig(
        mode=mode,
        channels=channels,
        crop=crop,
        dz=dz,
        n=n,
        anisotropy=anisotropy,
        upscale=upscale,
        seed=seed,
        threads=threads,
    )

    files = files[:config.n]
    logger.info(f"[{label}] Using {len(files)} registered {'file' if len(files) == 1 else 'files'}")
    if roi_points is not None:
        logger.info(f"[{label}] Using ROI points from: {roi_points}")

    all_zarr = bool(files and all(_is_zarr_path(f) for f in files))
    if aux_channel_stack is not None and not (all_zarr and config.mode in {"ortho", "z"}):
        raise click.BadParameter("--aux-channel-stack is currently supported for z/ortho extraction from a Zarr input.")

    if all_zarr and config.mode == "z":
        _execute_zarr_z_extraction(
            label=label,
            files=files,
            config=config,
            out_dir=out_dir,
            max_from_path=max_from_path,
            explicit_mask_path=explicit_mask_path,
            enrich_boundaries=enrich_boundaries,
            roi_points=roi_points,
            aux_channel_stack=aux_channel_stack,
        )
        return

    if all_zarr and config.mode == "maxproj":
        _execute_zarr_maxproj_extraction(
            label=label,
            files=files,
            config=config,
            out_dir=out_dir,
            max_from_path=max_from_path,
            explicit_mask_path=explicit_mask_path,
            enrich_boundaries=enrich_boundaries,
            roi_points=roi_points,
        )
        return

    if all_zarr and config.mode == "ortho":
        _execute_zarr_ortho_extraction(
            label=label,
            files=files,
            config=config,
            out_dir=out_dir,
            max_from_path=max_from_path,
            aux_channel_stack=aux_channel_stack,
            explicit_mask_path=explicit_mask_path,
            enrich_boundaries=enrich_boundaries,
            roi_points=roi_points,
        )
        return

    _execute_tiff_extraction(
        label=label,
        files=files,
        config=config,
        out_dir=out_dir,
        max_from_path=max_from_path,
        explicit_mask_path=explicit_mask_path,
        enrich_boundaries=enrich_boundaries,
        roi_points=roi_points,
        z_crops_per_file=z_crops_per_file,
    )


def _execute_zarr_z_extraction(
    *,
    label: str,
    files: list[Path],
    config: ExtractionConfig,
    out_dir: Path,
    max_from_path: Path | None,
    explicit_mask_path: Path | None,
    enrich_boundaries: Path | None,
    roi_points: Path | None = None,
    aux_channel_stack: Path | None = None,
) -> None:
    enrich_mask_vol: MaskLike | None = None
    if enrich_boundaries is not None:
        enrich_mask_vol = _open_mask_volume(enrich_boundaries)
        logger.info(f"[{label}] Using enrichment mask for tile selection: {enrich_boundaries}")

    # Load ROI points if provided (coordinates are upscaled 2x inside load_roi_points)
    point_coords: list[tuple[int, int]] | None = None
    if roi_points is not None:
        point_coords = load_roi_points(roi_points)
        logger.info(f"[{label}] Loaded {len(point_coords)} points from ROI file for Z extraction")

    rng = np.random.default_rng(config.seed)
    tile_jobs: list[TileJob] = []
    total_outputs = 0

    for f in files:
        vol, names_all = _open_volume(f)
        aux_channel_vols, aux_channel_names = _open_aux_channel_volumes(
            aux_channel_stack,
            vol,
            label=f"{label}:{f.name}",
        )
        mask_path = resolve_file_mask_path(f, explicit_mask_path)
        mask_vol = open_and_validate_mask(mask_path, vol, label=f"{label}:{f.name}")

        if point_coords is None and enrich_mask_vol is None:
            content_estimation = _content_estimation_volume(f, vol)
            content_crops = _sample_random_content_z_crops(
                vol=vol,
                content_estimation=content_estimation,
                crop=config.crop,
                count=config.n,
                rng=rng,
                label=f"{label}:{f.name}",
            )
            for crop in content_crops:
                z_candidates = _contentful_z_candidates_around(
                    vol=vol,
                    center=crop.z_index,
                    y0=crop.y0,
                    x0=crop.x0,
                    z_len=vol.shape[0],
                    dz=config.dz,
                )
                if not z_candidates:
                    logger.debug(
                        f"[{label}:{f.name}] Skipped empty content crop at "
                        f"z={crop.z_index}, y={crop.y0}, x={crop.x0}."
                    )
                    continue
                total_outputs += len(z_candidates)
                tile_jobs.append(
                    TileJob(
                        file=f,
                        vol=vol,
                        channel_names=names_all,
                        mask_vol=mask_vol,
                        mask_path=mask_path,
                        tile_origins=[(crop.y0, crop.x0)],
                        z_candidates=z_candidates,
                        aux_channel_vols=aux_channel_vols,
                        aux_channel_names=aux_channel_names,
                    )
                )
            continue

        if enrich_mask_vol is not None:
            all_tiles = _compute_tile_origins(
                vol.shape,
                tile_size=ZARR_TILE_SIZE,
                n_tiles=config.n * 10,
                crop=config.crop,
            )
            logger.info(f"[{label}] Scoring {len(all_tiles)} tile candidates by mask coverage...")
            tile_origins = _score_and_select_tiles(
                all_tiles,
                enrich_mask_vol,
                tile_size=ZARR_TILE_SIZE,
                count=config.n,
                score_fn=lambda tile: int(np.sum(tile > 0)),
            )
            logger.info(f"[{label}] Selected {len(tile_origins)} tiles by diversity scoring")
        else:
            tile_half = ZARR_TILE_SIZE // 2
            tile_origins = []
            for px, py in point_coords:
                y0 = max(config.crop, py - tile_half)
                x0 = max(config.crop, px - tile_half)
                y0 = min(y0, vol.shape[1] - config.crop - ZARR_TILE_SIZE)
                x0 = min(x0, vol.shape[2] - config.crop - ZARR_TILE_SIZE)
                tile_origins.append((max(0, y0), max(0, x0)))
            logger.info(f"[{label}] Using {len(tile_origins)} tile origins from ROI points")

        z_candidates = list(range(0, vol.shape[0], config.dz))
        if not z_candidates:
            raise ValueError(f"[{label}] No Z indices available after applying dz to fused Zarr volume.")
        total_outputs += len(tile_origins) * len(z_candidates)
        tile_jobs.append(
            TileJob(
                file=f,
                vol=vol,
                channel_names=names_all,
                mask_vol=mask_vol,
                mask_path=mask_path,
                tile_origins=tile_origins,
                z_candidates=z_candidates,
                aux_channel_vols=aux_channel_vols,
                aux_channel_names=aux_channel_names,
            )
        )

    logger.info(f"[{label}] Tracking {total_outputs} output files from Zarr input(s)")

    with progress_reporter(total_outputs) as progress_update:
        for job in tile_jobs:
            _extract_tiles_from_zarr(
                job=job,
                roi=label,
                out_dir=out_dir,
                channels=config.channels,
                dz=config.dz,
                upscale=config.upscale,
                max_from_path=max_from_path,
                progress=progress_update,
            )


def _execute_zarr_maxproj_extraction(
    *,
    label: str,
    files: list[Path],
    config: ExtractionConfig,
    out_dir: Path,
    max_from_path: Path | None,
    explicit_mask_path: Path | None,
    enrich_boundaries: Path | None,
    roi_points: Path | None = None,
) -> None:
    enrich_mask_vol: MaskLike | None = None
    if enrich_boundaries is not None:
        enrich_mask_vol = _open_mask_volume(enrich_boundaries)
        logger.info(f"[{label}] Using enrichment mask for tile selection: {enrich_boundaries}")

    point_coords: list[tuple[int, int]] | None = None
    if roi_points is not None:
        point_coords = load_roi_points(roi_points)
        logger.info(f"[{label}] Loaded {len(point_coords)} points from ROI file for max projection")

    tile_jobs: list[TileJob] = []
    total_outputs = 0

    for f in files:
        vol, names_all = _open_volume(f)
        mask_path = resolve_file_mask_path(f, explicit_mask_path)
        mask_vol = open_and_validate_mask(mask_path, vol, label=f"{label}:{f.name}")

        if point_coords is not None:
            tile_half = ZARR_TILE_SIZE // 2
            tile_origins: list[tuple[int, int]] = []
            for px, py in point_coords:
                y0 = max(config.crop, py - tile_half)
                x0 = max(config.crop, px - tile_half)
                y0 = min(y0, vol.shape[1] - config.crop - ZARR_TILE_SIZE)
                x0 = min(x0, vol.shape[2] - config.crop - ZARR_TILE_SIZE)
                y0 = max(0, y0)
                x0 = max(0, x0)
                tile_origins.append((y0, x0))
            logger.info(f"[{label}] Using {len(tile_origins)} tile origins from ROI points")
        elif enrich_mask_vol is not None:
            all_tiles = _compute_tile_origins(
                vol.shape,
                tile_size=ZARR_TILE_SIZE,
                n_tiles=config.n * 10,
                crop=config.crop,
            )
            logger.info(f"[{label}] Scoring {len(all_tiles)} tile candidates by mask coverage...")
            tile_origins = _score_and_select_tiles(
                all_tiles,
                enrich_mask_vol,
                tile_size=ZARR_TILE_SIZE,
                count=config.n,
                score_fn=lambda tile: int(np.sum(tile > 0)),
            )
            logger.info(f"[{label}] Selected {len(tile_origins)} tiles by diversity scoring")
        else:
            tile_origins = _compute_tile_origins(
                vol.shape,
                tile_size=ZARR_TILE_SIZE,
                n_tiles=config.n,
                crop=config.crop,
            )

        z_candidates = list(range(0, vol.shape[0], config.dz))
        if not z_candidates:
            raise ValueError(f"[{label}] No Z indices available after applying dz to fused Zarr volume.")
        total_outputs += len(tile_origins)
        tile_jobs.append(
            TileJob(
                file=f,
                vol=vol,
                channel_names=names_all,
                mask_vol=mask_vol,
                mask_path=mask_path,
                tile_origins=tile_origins,
                z_candidates=z_candidates,
            )
        )

    logger.info(f"[{label}] Tracking {total_outputs} max-projection output files from Zarr input(s)")

    with progress_reporter(total_outputs) as progress_update:
        for job in tile_jobs:
            _extract_maxproj_tiles_from_zarr(
                job=job,
                roi=label,
                out_dir=out_dir,
                channels=config.channels,
                dz=config.dz,
                upscale=config.upscale,
                max_from_path=max_from_path,
                progress=progress_update,
            )


def _execute_zarr_ortho_extraction(
    *,
    label: str,
    files: list[Path],
    config: ExtractionConfig,
    out_dir: Path,
    max_from_path: Path | None,
    aux_channel_stack: Path | None,
    explicit_mask_path: Path | None,
    enrich_boundaries: Path | None,
    roi_points: Path | None = None,
) -> None:
    rng = np.random.default_rng(config.seed)
    logger.info(f"[{label}] Random seed: {config.seed if config.seed is not None else 'system entropy'}")

    slice_jobs: list[SliceJob] = []
    content_crops_by_file: dict[Path, list[RandomContentCrop]] = {}
    enrich_mask_vol: MaskLike | None = None
    if enrich_boundaries is not None:
        enrich_mask_vol = _open_mask_volume(enrich_boundaries)
        logger.info(f"[{label}] Using enrichment mask for diversity scoring: {enrich_boundaries}")

    # Load ROI points if provided (coordinates are upscaled 2x inside load_roi_points)
    point_coords: list[tuple[int, int]] | None = None
    if roi_points is not None:
        point_coords = load_roi_points(roi_points)
        logger.info(f"[{label}] Loaded {len(point_coords)} points from ROI file")

    for f in files:
        vol, names_all = _open_volume(f)
        mask_path = resolve_file_mask_path(f, explicit_mask_path)
        mask_vol = open_and_validate_mask(mask_path, vol, label=f"{label}:{f.name}")

        other_vol = _resolve_other_volume(f, max_from_path)
        aux_channel_vols, aux_channel_names = _open_aux_channel_volumes(
            aux_channel_stack,
            vol,
            label=f"{label}:{f.name}",
        )

        selected_indices = _parse_channels(config.channels, names_all, vol.shape[-1])
        _ensure_channel_bounds(selected_indices, vol.shape[-1], label=f.name)
        base_names = _resolve_output_names(selected_indices, names_all, config.channels)
        out_names = [*base_names, *aux_channel_names]
        if other_vol is not None:
            out_names.append("max_from")

        max_width_pre_upscale = int(MAX_WIDTH_AFTER_UPSCALE / config.upscale)
        slab_half_width = 256  # 512px slab width / 2
        content_estimation = _content_estimation_volume(f, vol)

        if point_coords is not None:
            # Use ROI points: for each point, create both Y and X slabs with context
            for px, py in point_coords:
                # Clamp to valid range
                base_py = max(config.crop, min(py, vol.shape[1] - config.crop - 1))
                base_px = max(config.crop, min(px, vol.shape[2] - config.crop - 1))

                # ZX slabs at position py (with context), x_slice centered at px
                x_start = max(config.crop, base_px - slab_half_width)
                x_end = min(vol.shape[2] - config.crop, base_px + slab_half_width)
                x_slice = slice(x_start, x_end)
                expanded_y = _expand_positions_with_context([base_py], crop=config.crop, axis_len=vol.shape[1])
                for yi in expanded_y:
                    slice_jobs.append(
                        SliceJob(
                            file=f,
                            vol=vol,
                            channel_names=names_all,
                            mask_vol=mask_vol,
                            other_vol=other_vol,
                            aux_channel_vols=aux_channel_vols,
                            position=yi,
                            axis="y",
                            perpendicular_slice=x_slice,
                            selected_indices=selected_indices,
                            out_names=out_names,
                        )
                    )

                # ZY slabs at position px (with context), y_slice centered at py
                y_start = max(config.crop, base_py - slab_half_width)
                y_end = min(vol.shape[1] - config.crop, base_py + slab_half_width)
                y_slice = slice(y_start, y_end)
                expanded_x = _expand_positions_with_context([base_px], crop=config.crop, axis_len=vol.shape[2])
                for xi in expanded_x:
                    slice_jobs.append(
                        SliceJob(
                            file=f,
                            vol=vol,
                            channel_names=names_all,
                            mask_vol=mask_vol,
                            other_vol=other_vol,
                            aux_channel_vols=aux_channel_vols,
                            position=xi,
                            axis="x",
                            perpendicular_slice=y_slice,
                            selected_indices=selected_indices,
                            out_names=out_names,
                        )
                    )
        else:
            if enrich_mask_vol is not None:
                y_candidates = _sample_positions(vol.shape[1], crop=config.crop, count=config.n * 10, rng=rng)
                x_candidates = _sample_positions(vol.shape[2], crop=config.crop, count=config.n * 10, rng=rng)
                y_candidates = _select_high_diversity_positions(y_candidates, enrich_mask_vol, "y", config.n)
                x_candidates = _select_high_diversity_positions(x_candidates, enrich_mask_vol, "x", config.n)
                for base_y in y_candidates:
                    x_slice = _compute_perpendicular_slice(
                        axis_len=vol.shape[2], crop=config.crop, max_width=max_width_pre_upscale, rng=rng
                    )
                    expanded_y = _expand_positions_with_context([base_y], crop=config.crop, axis_len=vol.shape[1])
                    for yi in expanded_y:
                        slice_jobs.append(
                            SliceJob(
                                file=f,
                                vol=vol,
                                channel_names=names_all,
                                mask_vol=mask_vol,
                                other_vol=other_vol,
                                aux_channel_vols=aux_channel_vols,
                                position=yi,
                                axis="y",
                                perpendicular_slice=x_slice,
                                selected_indices=selected_indices,
                                out_names=out_names,
                            )
                        )
                for base_x in x_candidates:
                    y_slice = _compute_perpendicular_slice(
                        axis_len=vol.shape[1], crop=config.crop, max_width=max_width_pre_upscale, rng=rng
                    )
                    expanded_x = _expand_positions_with_context([base_x], crop=config.crop, axis_len=vol.shape[2])
                    for xi in expanded_x:
                        slice_jobs.append(
                            SliceJob(
                                file=f,
                                vol=vol,
                                channel_names=names_all,
                                mask_vol=mask_vol,
                                other_vol=other_vol,
                                aux_channel_vols=aux_channel_vols,
                                position=xi,
                                axis="x",
                                perpendicular_slice=y_slice,
                                selected_indices=selected_indices,
                                out_names=out_names,
                            )
                        )
            else:
                content_crops = _sample_random_content_z_crops(
                    vol=vol,
                    content_estimation=content_estimation,
                    crop=config.crop,
                    count=config.n,
                    rng=rng,
                    label=f"{label}:{f.name}",
                )
                content_crops_by_file[f] = content_crops
                for sample in content_crops:
                    base_y = sample.y0 + ZARR_TILE_SIZE // 2
                    base_x = sample.x0 + ZARR_TILE_SIZE // 2
                    x_slice = slice(sample.x0, sample.x0 + ZARR_TILE_SIZE)
                    y_slice = slice(sample.y0, sample.y0 + ZARR_TILE_SIZE)
                    z_slice = _z_crop_slice_around(sample.z_index, z_len=vol.shape[0])

                    expanded_y = _expand_positions_with_context(
                        [base_y],
                        crop=config.crop,
                        axis_len=vol.shape[1],
                        step=2,
                        context_pairs=25,
                    )
                    for yi in expanded_y:
                        slice_jobs.append(
                            SliceJob(
                                file=f,
                                vol=vol,
                                channel_names=names_all,
                                mask_vol=mask_vol,
                                other_vol=other_vol,
                                aux_channel_vols=aux_channel_vols,
                                position=yi,
                                axis="y",
                                perpendicular_slice=x_slice,
                                selected_indices=selected_indices,
                                out_names=out_names,
                                z_slice=z_slice,
                            )
                        )

                    expanded_x = _expand_positions_with_context(
                        [base_x],
                        crop=config.crop,
                        axis_len=vol.shape[2],
                        step=2,
                        context_pairs=25,
                    )
                    for xi in expanded_x:
                        slice_jobs.append(
                            SliceJob(
                                file=f,
                                vol=vol,
                                channel_names=names_all,
                                mask_vol=mask_vol,
                                other_vol=other_vol,
                                aux_channel_vols=aux_channel_vols,
                                position=xi,
                                axis="x",
                                perpendicular_slice=y_slice,
                                selected_indices=selected_indices,
                                out_names=out_names,
                                z_slice=z_slice,
                            )
                        )

    logger.info(f"[{label}] Processing {len(slice_jobs)} ortho slices")

    jobs_by_file: dict[Path, list[SliceJob]] = {}
    for job in slice_jobs:
        jobs_by_file.setdefault(job.file, []).append(job)

    with progress_reporter(len(slice_jobs)) as progress_update:
        for file_jobs in jobs_by_file.values():
            first_job = file_jobs[0]
            requests = [
                _ortho_output_request(
                    file_stem=job.file.stem,
                    roi=label,
                    out_dir=out_dir,
                    axis=job.axis,
                    position=job.position,
                    perpendicular_slice=job.perpendicular_slice,
                    z_slice=job.z_slice,
                )
                for job in file_jobs
            ]
            _write_requested_ortho_slices(
                vol=first_job.vol,
                mask_vol=first_job.mask_vol,
                other_vol=first_job.other_vol,
                requests=requests,
                selected_indices=first_job.selected_indices,
                out_names=first_job.out_names,
                anisotropy=config.anisotropy,
                upscale=config.upscale,
                channels_arg=config.channels,
                progress=progress_update,
                aux_channel_vols=first_job.aux_channel_vols,
            )

    for file_jobs in jobs_by_file.values():
        first_job = file_jobs[0]
        content_crops = content_crops_by_file.get(first_job.file)
        if content_crops is None:
            content_crops = _sample_random_content_z_crops(
                vol=first_job.vol,
                content_estimation=_content_estimation_volume(first_job.file, first_job.vol),
                crop=config.crop,
                count=config.n,
                rng=rng,
                label=f"{label}:{first_job.file.name}",
            )
        _write_random_content_z_crops(
            file=first_job.file,
            roi=label,
            vol=first_job.vol,
            mask_vol=first_job.mask_vol,
            aux_channel_vols=first_job.aux_channel_vols,
            selected_indices=first_job.selected_indices,
            out_names=first_job.out_names,
            out_dir=out_dir,
            channels=config.channels,
            upscale=config.upscale,
            content_crops=content_crops,
        )


def _sample_random_content_z_crops(
    *,
    vol: Volume,
    content_estimation: ContentEstimationVolume,
    crop: int,
    count: int,
    rng: np.random.Generator,
    label: str,
) -> list[RandomContentCrop]:
    z_len, y_len, x_len, _channel_count = vol.shape
    tile_size = ZARR_TILE_SIZE
    if y_len < tile_size + 2 * crop or x_len < tile_size + 2 * crop:
        raise ValueError(
            f"Zarr volume spatial dimensions are smaller than the requested {tile_size}x{tile_size} crop size."
        )

    max_y0 = y_len - crop - tile_size
    max_x0 = x_len - crop - tile_size
    content_vol = content_estimation.vol
    z_scale, y_scale, x_scale = content_estimation.scale_zyx
    content_z_len, content_y_len, content_x_len = content_vol.shape[:3]
    content_tile_y = max(1, int(math.ceil(tile_size / y_scale)))
    content_tile_x = max(1, int(math.ceil(tile_size / x_scale)))
    if content_y_len < content_tile_y or content_x_len < content_tile_x:
        raise ValueError("Content-estimation volume is smaller than the requested crop footprint.")

    content_max_y0 = content_y_len - content_tile_y
    content_max_x0 = content_x_len - content_tile_x
    accepted = 0
    attempts = 0
    max_attempts = max(count * 50, count)
    cancel_event = get_cancel_event()
    content_crops: list[RandomContentCrop] = []

    while accepted < count and attempts < max_attempts:
        if cancel_event.is_set():
            raise TaskCancelledException("Cancelled by user")
        attempts += 1

        content_z = int(rng.integers(0, content_z_len))
        content_y0 = int(rng.integers(0, content_max_y0 + 1))
        content_x0 = int(rng.integers(0, content_max_x0 + 1))
        content_y_slice = slice(content_y0, content_y0 + content_tile_y)
        content_x_slice = slice(content_x0, content_x0 + content_tile_x)
        plane = content_vol[content_z, content_y_slice, content_x_slice, :]
        if _has_too_many_zero_pixels(np.asarray(plane)):
            continue

        z_index = min(z_len - 1, int(round(content_z * z_scale)))
        y0 = max(crop, min(max_y0, int(round(content_y0 * y_scale))))
        x0 = max(crop, min(max_x0, int(round(content_x0 * x_scale))))
        content_crops.append(RandomContentCrop(z_index=z_index, y0=y0, x0=x0))
        accepted += 1

    if accepted < count:
        logger.warning(
            f"[{label}] Sampled {accepted}/{count} random content crop(s); "
            f"remaining candidates exceeded the zero-pixel threshold."
        )
    return content_crops


def _content_estimation_volume(file: Path, level0_vol: Volume) -> ContentEstimationVolume:
    if not _is_ome_zarr_path(file):
        return ContentEstimationVolume(vol=level0_vol, scale_zyx=(1.0, 1.0, 1.0))

    level_array = _open_ome_zarr_highest_level_array(file)
    level_vol: Volume
    axes = _zarr_array_axes(level_array)
    if level_array.ndim == 3 and axes in {None, "ZYX"}:
        level_vol = SingleChannelZarrVolume(level_array)
    elif level_array.ndim == 4:
        level_vol = level_array
    else:
        raise ValueError(f"Expected OME-Zarr highest level to be ZYX or ZYXC, got shape={level_array.shape}.")

    scale_zyx = tuple(
        float(level0_size) / float(level_size)
        for level0_size, level_size in zip(level0_vol.shape[:3], level_vol.shape[:3], strict=True)
    )
    return ContentEstimationVolume(vol=level_vol, scale_zyx=scale_zyx)


def _write_random_content_z_crops(
    *,
    file: Path,
    roi: str,
    vol: Volume,
    mask_vol: MaskLike | None,
    selected_indices: list[int],
    out_names: list[str],
    out_dir: Path,
    channels: str | None,
    upscale: float,
    content_crops: list[RandomContentCrop],
    aux_channel_vols: list[Volume] | None = None,
) -> None:
    _z_len, y_len, x_len, _channel_count = vol.shape
    tile_size = ZARR_TILE_SIZE
    coord_width = len(str(max(y_len, x_len)))

    for sample in content_crops:
        z_index = sample.z_index
        y0 = sample.y0
        x0 = sample.x0
        y_slice = slice(y0, y0 + tile_size)
        x_slice = slice(x0, x0 + tile_size)
        plane = vol[z_index, y_slice, x_slice, :]
        aux_planes = [
            np.asarray(aux_vol[z_index, y_slice, x_slice, :])
            for aux_vol in aux_channel_vols or []
        ]

        cyx_u16 = _prep_slab(
            plane,
            ch_idx=selected_indices,
            channel_axis=2,
            crop_slices=None,
            filter_before=True,
            append_max=None,
            apply_filter=False,
        )
        cyx_u16 = _append_aux_channel_slabs(cyx_u16, aux_planes, channel_axis=2)
        cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))
        out_name = _format_tile_filename(
            file.stem,
            roi,
            z_index,
            y0,
            x0,
            coord_width=coord_width,
        )
        _write_tiff(
            out_dir / out_name,
            cyx_u16,
            axes="CYX",
            names=out_names,
            channels_arg=channels,
            upscale=upscale,
        )
        if mask_vol is not None:
            mask_tile = _squeeze_mask(mask_vol[z_index, y_slice, x_slice])
            if mask_tile.ndim != 2:
                raise ValueError("Mask crop extraction expected 2D data.")
            resized_mask = _resize_mask(mask_tile, (upscale, upscale))
            _write_mask_tiff(out_dir / _mask_filename(out_name), resized_mask, axes="YX")


def _execute_tiff_extraction(
    *,
    label: str,
    files: list[Path],
    config: ExtractionConfig,
    out_dir: Path,
    max_from_path: Path | None,
    explicit_mask_path: Path | None,
    enrich_boundaries: Path | None,
    roi_points: Path | None = None,
    z_crops_per_file: int = 1,
) -> None:
    if roi_points is not None:
        logger.warning(f"[{label}] --roi-points is only supported for Zarr inputs; ignoring for TIFF extraction")

    with progress_bar_threadpool(len(files), threads=config.threads, stop_on_exception=True) as submit:
        if config.mode == "z":
            for idx, f in enumerate(files):
                mask_path = explicit_mask_path if explicit_mask_path is not None else _resolve_mask_path(f)
                if mask_path is not None:
                    logger.info(f"[{label}] Found mask stack: {mask_path}")
                file_seed = (config.seed + idx) if config.seed is not None else None
                submit(
                    _extract_z_slices,
                    file=f,
                    roi=label,
                    out_dir=out_dir,
                    channels=config.channels,
                    dz=config.dz,
                    n_crops=z_crops_per_file,
                    upscale=config.upscale,
                    max_from_path=max_from_path,
                    mask_path=mask_path,
                    enrich_boundaries=enrich_boundaries,
                    seed=file_seed,
                    progress=None,
                )
        elif config.mode == "maxproj":
            for idx, f in enumerate(files):
                mask_path = explicit_mask_path if explicit_mask_path is not None else _resolve_mask_path(f)
                if mask_path is not None:
                    logger.info(f"[{label}] Found mask stack: {mask_path}")
                file_seed = (config.seed + idx) if config.seed is not None else None
                submit(
                    _extract_maxproj_slices,
                    file=f,
                    roi=label,
                    out_dir=out_dir,
                    channels=config.channels,
                    dz=config.dz,
                    n_crops=z_crops_per_file,
                    upscale=config.upscale,
                    max_from_path=max_from_path,
                    mask_path=mask_path,
                    enrich_boundaries=enrich_boundaries,
                    seed=file_seed,
                    progress=None,
                )
        elif config.mode == "ortho":
            for idx, f in enumerate(files):
                mask_path = explicit_mask_path if explicit_mask_path is not None else _resolve_mask_path(f)
                if mask_path is not None:
                    logger.info(f"[{label}] Found mask stack: {mask_path}")
                file_seed = (config.seed + idx) if config.seed is not None else None
                submit(
                    _extract_ortho_slices,
                    file=f,
                    roi=label,
                    out_dir=out_dir,
                    channels=config.channels,
                    crop=config.crop,
                    n=config.n,
                    anisotropy=config.anisotropy,
                    upscale=config.upscale,
                    max_from_path=max_from_path,
                    mask_path=mask_path,
                    enrich_boundaries=enrich_boundaries,
                    seed=file_seed,
                    progress=None,
                )
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")


def _resize_uint16(data: np.ndarray, factors: tuple[float, ...]) -> np.ndarray:
    """Resize data with scipy.ndimage.zoom, preserving uint16 output."""

    if all(math.isclose(f, 1.0, abs_tol=1e-9, rel_tol=1e-9) for f in factors):
        # Ensure dtype consistency without extra work when no scaling requested.
        return data.astype(np.uint16, copy=False)

    resized = zoom(data.astype(np.float32, copy=False), factors, order=1)
    return np.clip(np.rint(resized), 0, 65530).astype(np.uint16)


def _prep_slab(
    slab: np.ndarray,
    *,
    ch_idx: list[int],
    channel_axis: int,
    crop_slices: tuple[slice, ...] | None,
    filter_before: bool,
    append_max: np.ndarray | None,
    apply_filter: bool = True,
) -> np.ndarray:
    """Select channels, optional filtering and max-append; return (C, ... spatial ...) uint16.

    - slab: spatial slab with channels on `channel_axis` (e.g., (Y,X,C), (Z,X,C), (Z,Y,C)).
    - ch_idx: selected channel indices (0-based).
    - crop_slices: optional slices on spatial axes (must not include channel axis).
    - filter_before: if True, filter before selection; else filter after selection.
    - append_max: optional array broadcastable to spatial shape, added as an extra channel.
    - apply_filter: enable sharpening filter (skipped for Zarr streaming to avoid extra IO).
    """
    arr = slab
    if apply_filter and filter_before:
        arr = unsharp_all(arr, channel_axis=channel_axis)

    # Select channels (channel axis last in arr)
    sel = np.take(arr, ch_idx, axis=channel_axis)

    if apply_filter and not filter_before:
        sel = unsharp_all(sel, channel_axis=sel.ndim - 1)

    # Append max channel if provided
    if append_max is not None:
        # Ensure append_max matches spatial shape
        while append_max.ndim < sel.ndim:
            append_max = append_max[..., None]
        sel = np.concatenate([sel, append_max], axis=sel.ndim - 1)

    # Move channels to first axis
    sel_c_first = np.moveaxis(sel, -1, 0)

    # Apply cropping on spatial axes if requested
    if crop_slices is not None and any(slc != slice(None) for slc in crop_slices):
        # Build slices with channel dim first
        idx = (slice(None),) + crop_slices
        sel_c_first = sel_c_first[idx]

    # Final dtype
    return np.clip(sel_c_first, 0, 65530).astype(np.uint16)


def _parse_channels(ch_arg: str | None, names: list[str] | None, channel_count: int | None) -> list[int]:
    """Parse channel specification; requires explicit indices or metadata-backed names."""
    if ch_arg is None or ch_arg.strip().lower() == "auto":
        if names:
            default_count = min(2, len(names))
            if default_count == 0:
                raise click.BadParameter("Channel metadata is empty; supply --channels explicitly.")
            return list(range(default_count))
        if channel_count is not None:
            return list(range(min(2, channel_count)))
        return [0, 1]

    parts = [p.strip() for p in ch_arg.split(",") if p.strip()]
    if not parts:
        raise click.BadParameter("Empty --channels specification.")

    try:
        return [int(p) for p in parts]
    except ValueError:
        if not names:
            raise click.BadParameter("Channel names not in metadata; pass numeric indices.")
        name_to_idx = {n: i for i, n in enumerate(names)}
        indices: list[int] = []
        for part in parts:
            try:
                indices.append(name_to_idx[part])
            except KeyError as error:
                raise click.BadParameter(f"Unknown channel name: {error.args[0]}") from error
        return indices


def _ensure_channel_bounds(indices: list[int], channel_count: int, *, label: str) -> None:
    if not indices:
        raise click.BadParameter("At least one channel index must be selected.")
    if min(indices) < 0:
        raise click.BadParameter("Channel indices must be non-negative.")
    if max(indices) >= channel_count:
        raise click.BadParameter(
            f"{label}: requested channel index {max(indices)} exceeds available channels ({channel_count})."
        )


def _materialize_zyxc_by_z(volume: Volume) -> np.ndarray:
    return np.stack([np.asarray(volume[z_index, :, :, :]) for z_index in range(volume.shape[0])], axis=0)


def _load_registered_stack(file: Path) -> tuple[np.ndarray, list[str] | None, str]:
    """Load a registered stack from TIFF or Zarr and normalise to (Z,C,Y,X).

    This reuses `_open_volume` to enforce the common shape contract (Z,Y,X,C)
    and then materialises a NumPy array with channels moved to axis 1.
    """
    vol_zyxc, names = _open_volume(file)
    img = vol_zyxc if isinstance(vol_zyxc, np.ndarray) else _materialize_zyxc_by_z(vol_zyxc)
    if img.ndim != 4:
        raise ValueError("Expected registered volume to be 4D (Z,Y,X,C).")
    img_zcyx = np.moveaxis(img, -1, 1)  # -> (Z,C,Y,X)
    return img_zcyx, names, file.name


def normalize_numeric_options(
    *,
    mode: str,
    dz: int,
    anisotropy: int,
    upscale: float | None,
    use_zarr: bool,
    has_max_from: bool,
    ortho_anisotropy_default: int,
) -> float:
    """Validate dz/anisotropy/upscale and mode-specific constraints.

    Returns the normalized ``upscale`` value while preserving existing CLI semantics.
    """
    if mode in {"z", "maxproj"} and anisotropy != ortho_anisotropy_default:
        raise click.BadParameter("--anisotropy parameter is only valid for 'ortho' mode.")
    if mode == "ortho" and dz != 1:
        raise click.BadParameter("--dz parameter is only valid for 'z' mode.")

    if use_zarr and has_max_from:
        raise click.BadParameter("--max-from cannot be used together with --zarr inputs.")

    if use_zarr:
        upscale_value = 2.0 if upscale is None else upscale
    elif upscale is None:
        upscale_value = 1.0
    else:
        upscale_value = upscale

    if upscale_value <= 0:
        raise click.BadParameter("--upscale must be positive.")

    return upscale_value


def run_single_file_extract(
    *,
    mode: str,
    registered: Path,
    out: Path,
    dz: int,
    n: int,
    z_crops_per_file: int,
    anisotropy: int,
    channels: str | None,
    crop: int,
    threads: int,
    upscale: float,
    seed: int | None,
    max_from_path: Path | None,
    aux_channel_stack: Path | None,
    label: str,
    masks: Path | None,
    enrich_boundaries: Path | None,
) -> None:
    files = [registered]

    if max_from_path is not None:
        _validate_max_from_path(max_from_path, files, label=label)

    _execute_extraction(
        label=label,
        files=files,
        mode=mode,
        out_dir=out,
        channels=channels,
        crop=crop,
        dz=dz,
        n=n,
        z_crops_per_file=z_crops_per_file,
        anisotropy=anisotropy,
        threads=threads,
        upscale=upscale,
        seed=seed,
        max_from_path=max_from_path,
        aux_channel_stack=aux_channel_stack,
        explicit_mask_path=masks,
        enrich_boundaries=enrich_boundaries,
    )


# ---------- Logging helpers ----------


def _resolve_other_volume(file: Path, max_from_path: Path | None) -> Volume | None:
    """Resolve and open the comparison volume for --max-from if it exists."""
    if max_from_path is None:
        return None

    target = (
        max_from_path
        if _is_zarr_path(max_from_path)
        else (max_from_path / file.name if max_from_path.is_dir() else max_from_path)
    )
    if not target.exists():
        return None
    other_vol, _ = _open_volume(target)
    return other_vol


def _compute_z_crop_positions(
    *,
    y_len: int,
    x_len: int,
    crop_size: int,
    n_crops: int,
    enrich_mask_vol: MaskLike | None,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    max_y = max(0, y_len - crop_size)
    max_x = max(0, x_len - crop_size)

    if enrich_mask_vol is not None and (max_y > 0 or max_x > 0):
        n_candidates = n_crops * 10
        positions: list[tuple[int, int]] = []
        for _ in range(n_candidates):
            cy = rng.integers(0, max_y + 1) if max_y > 0 else 0
            cx = rng.integers(0, max_x + 1) if max_x > 0 else 0
            positions.append((cy, cx))

        return _score_and_select_tiles(
            positions,
            enrich_mask_vol,
            tile_size=crop_size,
            count=n_crops,
            score_fn=lambda tile: len(np.unique(tile)),
        )

    return [
        (
            rng.integers(0, max_y + 1) if max_y > 0 else 0,
            rng.integers(0, max_x + 1) if max_x > 0 else 0,
        )
        for _ in range(n_crops)
    ]


def _iterative_max_projection(
    volume: Volume | MaskLike,
    z_indices: list[int],
    *,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    if not z_indices:
        raise ValueError("At least one Z index is required for max projection.")

    projection: np.ndarray | None = None
    for z_index in z_indices:
        plane = np.asarray(volume[z_index, y_slice, x_slice, ...])
        if projection is None:
            projection = plane.copy()
            continue
        np.maximum(projection, plane, out=projection)

    if projection is None:
        raise ValueError("Failed to compute max projection.")
    return projection


def _extract_z_slices(
    *,
    file: Path,
    roi: str,
    out_dir: Path,
    channels: str | None,
    dz: int,
    n_crops: int,
    upscale: float,
    max_from_path: Path | None,
    mask_path: Path | None,
    enrich_boundaries: Path | None,
    seed: int | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> None:
    reporter = _normalize_reporter(progress)

    size_str = _format_size(_path_size_cached(str(file.resolve())))
    logger.info(f"3D→Z: {file.name} [{size_str}] (dz={dz})")

    ctx = build_extraction_context(
        file,
        roi,
        channels=channels,
        max_from_path=max_from_path,
        mask_path=mask_path,
        enrich_path=enrich_boundaries,
        upscale=upscale,
        anisotropy=1,
        out_dir=out_dir,
    )

    vol = ctx.vol
    selected_indices = ctx.selected_indices
    out_names = ctx.out_names
    other_vol = ctx.other_vol
    mask_vol = ctx.mask_vol
    enrich_mask_vol = ctx.enrich_mask_vol

    z_len, y_len, x_len, _ = vol.shape

    crop_size = DEFAULT_CROP_SIZE

    n_crops = max(1, n_crops)
    rng = np.random.default_rng(seed)

    crop_positions = _compute_z_crop_positions(
        y_len=y_len,
        x_len=x_len,
        crop_size=crop_size,
        n_crops=n_crops,
        enrich_mask_vol=enrich_mask_vol,
        rng=rng,
    )
    if enrich_mask_vol is not None:
        logger.info(f"[{roi}] Selected {len(crop_positions)} crops by diversity scoring")

    z_idxs = list(range(0, z_len, dz))
    cancel_event = get_cancel_event()

    for crop_idx, (y_start, x_start) in enumerate(crop_positions):
        y_slice = slice(y_start, min(y_start + crop_size, y_len))
        x_slice = slice(x_start, min(x_start + crop_size, x_len))

        for index in z_idxs:
            if cancel_event.is_set():
                raise TaskCancelledException("Cancelled by user")
            t0 = perf_counter()
            plane = vol[index, y_slice, x_slice, :]
            other_max = (
                other_vol[index, y_slice, x_slice, :].max(axis=2) if other_vol is not None else None
            )
            t_read = perf_counter() - t0

            t0 = perf_counter()
            cyx_u16 = _prep_slab(
                plane,
                ch_idx=selected_indices,
                channel_axis=2,
                crop_slices=None,
                filter_before=False,
                append_max=other_max,
            )
            t_prep = perf_counter() - t0

            t0 = perf_counter()
            cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))
            t_resize = perf_counter() - t0

            out_name = (
                _prefix_with_roi(f"{file.stem}_crop{crop_idx:02d}_z{index:02d}.tif", roi)
                if n_crops > 1
                else _prefix_with_roi(f"{file.stem}_z{index:02d}.tif", roi)
            )
            out_file = out_dir / out_name

            t0 = perf_counter()
            _write_tiff(
                out_file,
                cyx_u16,
                axes="CYX",
                names=out_names,
                channels_arg=channels,
                upscale=upscale,
            )
            t_write = perf_counter() - t0

            t_mask = 0.0
            if mask_vol is not None:
                t0 = perf_counter()
                mask_plane = _squeeze_mask(mask_vol[index, ...])
                if mask_plane.ndim != 2:
                    raise ValueError("Mask plane extraction expected 2D data.")
                mask_cropped = mask_plane[y_slice, x_slice]
                resized_mask = _resize_mask(mask_cropped, (upscale, upscale))
                mask_out_name = _mask_filename(out_name)
                _write_mask_tiff(out_dir / mask_out_name, resized_mask, axes="YX")
                t_mask = perf_counter() - t0

            logger.debug(
                f"z{index:02d} read={t_read:.3f}s prep={t_prep:.3f}s resize={t_resize:.3f}s write={t_write:.3f}s mask={t_mask:.3f}s"
            )
            if reporter is not None:
                reporter.advance()


def _extract_maxproj_slices(
    *,
    file: Path,
    roi: str,
    out_dir: Path,
    channels: str | None,
    dz: int,
    n_crops: int,
    upscale: float,
    max_from_path: Path | None,
    mask_path: Path | None,
    enrich_boundaries: Path | None,
    seed: int | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> None:
    reporter = _normalize_reporter(progress)

    size_str = _format_size(_path_size_cached(str(file.resolve())))
    logger.info(f"3D→MaxProj: {file.name} [{size_str}] (dz={dz})")

    ctx = build_extraction_context(
        file,
        roi,
        channels=channels,
        max_from_path=max_from_path,
        mask_path=mask_path,
        enrich_path=enrich_boundaries,
        upscale=upscale,
        anisotropy=1,
        out_dir=out_dir,
    )

    vol = ctx.vol
    selected_indices = ctx.selected_indices
    out_names = ctx.out_names
    other_vol = ctx.other_vol
    mask_vol = ctx.mask_vol
    enrich_mask_vol = ctx.enrich_mask_vol

    z_len, y_len, x_len, _ = vol.shape
    z_idxs = list(range(0, z_len, dz))
    crop_size = DEFAULT_CROP_SIZE

    n_crops = max(1, n_crops)
    rng = np.random.default_rng(seed)
    crop_positions = _compute_z_crop_positions(
        y_len=y_len,
        x_len=x_len,
        crop_size=crop_size,
        n_crops=n_crops,
        enrich_mask_vol=enrich_mask_vol,
        rng=rng,
    )
    if enrich_mask_vol is not None:
        logger.info(f"[{roi}] Selected {len(crop_positions)} crops by diversity scoring")

    cancel_event = get_cancel_event()
    for crop_idx, (y_start, x_start) in enumerate(crop_positions):
        if cancel_event.is_set():
            raise TaskCancelledException("Cancelled by user")

        y_slice = slice(y_start, min(y_start + crop_size, y_len))
        x_slice = slice(x_start, min(x_start + crop_size, x_len))

        plane = _iterative_max_projection(vol, z_idxs, y_slice=y_slice, x_slice=x_slice)
        other_max = None
        if other_vol is not None:
            other_plane = _iterative_max_projection(other_vol, z_idxs, y_slice=y_slice, x_slice=x_slice)
            other_max = other_plane.max(axis=2)

        cyx_u16 = _prep_slab(
            plane,
            ch_idx=selected_indices,
            channel_axis=2,
            crop_slices=None,
            filter_before=False,
            append_max=other_max,
        )
        cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))

        out_name = (
            _prefix_with_roi(f"{file.stem}_crop{crop_idx:02d}_maxproj.tif", roi)
            if n_crops > 1
            else _prefix_with_roi(f"{file.stem}_maxproj.tif", roi)
        )
        out_file = out_dir / out_name
        _write_tiff(
            out_file,
            cyx_u16,
            axes="CYX",
            names=out_names,
            channels_arg=channels,
            upscale=upscale,
        )

        if mask_vol is not None:
            mask_plane = _iterative_max_projection(mask_vol, z_idxs, y_slice=y_slice, x_slice=x_slice)
            mask_plane = _squeeze_mask(mask_plane)
            if mask_plane.ndim != 2:
                raise ValueError("Mask max-projection expected 2D data.")
            resized_mask = _resize_mask(mask_plane, (upscale, upscale))
            _write_mask_tiff(out_dir / _mask_filename(out_name), resized_mask, axes="YX")

        if reporter is not None:
            reporter.advance()


def _extract_tiles_from_zarr(
    *,
    job: TileJob,
    roi: str,
    out_dir: Path,
    channels: str | None,
    dz: int,
    upscale: float,
    max_from_path: Path | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> None:
    reporter = _normalize_reporter(progress)
    size_str = _format_size(_path_size_cached(str(job.file.resolve())))
    logger.info(f"3D→Z tiles: {job.file.name} [{size_str}]")

    selected_indices = _parse_channels(channels, job.channel_names, job.vol.shape[-1])
    _ensure_channel_bounds(selected_indices, job.vol.shape[-1], label=job.file.name)
    base_names = _resolve_output_names(selected_indices, job.channel_names, channels)
    other_vol = _resolve_other_volume(job.file, max_from_path)
    out_names = [*base_names, *job.aux_channel_names]
    if other_vol is not None:
        out_names.append("max_from")

    z_len = job.vol.shape[0]
    if job.mask_vol is not None and job.mask_vol.shape[0] != z_len:
        raise ValueError(f"Mask volume {_resolve_mask_path(job.file)} does not match Z dimension of {job.file}.")

    coord_width = len(str(max(job.vol.shape[1], job.vol.shape[2])))
    z_candidates = job.z_candidates
    if not z_candidates:
        raise ValueError(f"[{roi}] No Z indices available after applying dz to fused Zarr volume.")

    cancel_event = get_cancel_event()
    for y0, x0 in job.tile_origins:
        y_slice_tile = slice(y0, y0 + ZARR_TILE_SIZE)
        x_slice_tile = slice(x0, x0 + ZARR_TILE_SIZE)
        skipped_for_tile = 0
        for z_index in z_candidates:
            if cancel_event.is_set():
                raise TaskCancelledException("Cancelled by user")
            plane = job.vol[z_index, y_slice_tile, x_slice_tile, :]
            aux_planes = [
                np.asarray(aux_vol[z_index, y_slice_tile, x_slice_tile, :])
                for aux_vol in job.aux_channel_vols
            ]
            skip_tile = _has_too_many_zero_pixels(np.asarray(plane))
            other_max = None
            if other_vol is not None:
                other_tile = other_vol[z_index, y_slice_tile, x_slice_tile, :]
                if not skip_tile:
                    skip_tile = _has_too_many_zero_pixels(np.asarray(other_tile))
                if not skip_tile:
                    other_max = other_tile.max(axis=2)

            if skip_tile:
                skipped_for_tile += 1
                if reporter is not None:
                    reporter.advance()
                continue

            cyx_u16 = _prep_slab(
                plane,
                ch_idx=selected_indices,
                channel_axis=2,
                crop_slices=None,
                filter_before=True,
                append_max=other_max,
                apply_filter=False,
            )
            cyx_u16 = _append_aux_channel_slabs(cyx_u16, aux_planes, channel_axis=2)
            cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))
            out_name = _format_tile_filename(
                job.file.stem,
                roi,
                z_index,
                y0,
                x0,
                coord_width=coord_width,
            )
            out_file = out_dir / out_name
            _write_tiff(
                out_file,
                cyx_u16,
                axes="CYX",
                names=out_names,
                channels_arg=channels,
                upscale=upscale,
            )
            if job.mask_vol is not None:
                mask_tile = _squeeze_mask(job.mask_vol[z_index, y_slice_tile, x_slice_tile])
                if mask_tile.ndim != 2:
                    raise ValueError("Mask tile extraction expected 2D data.")
                resized_mask = _resize_mask(mask_tile, (upscale, upscale))
                mask_out = out_dir / _mask_filename(out_name)
                _write_mask_tiff(mask_out, resized_mask, axes="YX")
            if reporter is not None:
                reporter.advance()
        if skipped_for_tile:
            logger.debug(
                f"Skipped {skipped_for_tile} z-slice(s) in tile ({y0},{x0}) of {job.file.name} due to zeros."
            )


def _extract_maxproj_tiles_from_zarr(
    *,
    job: TileJob,
    roi: str,
    out_dir: Path,
    channels: str | None,
    dz: int,
    upscale: float,
    max_from_path: Path | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> None:
    reporter = _normalize_reporter(progress)
    size_str = _format_size(_path_size_cached(str(job.file.resolve())))
    logger.info(f"3D→MaxProj tiles: {job.file.name} [{size_str}]")

    selected_indices = _parse_channels(channels, job.channel_names, job.vol.shape[-1])
    _ensure_channel_bounds(selected_indices, job.vol.shape[-1], label=job.file.name)
    base_names = _resolve_output_names(selected_indices, job.channel_names, channels)
    other_vol = _resolve_other_volume(job.file, max_from_path)
    out_names = [*base_names, "max_from"] if other_vol is not None else base_names

    z_len = job.vol.shape[0]
    if job.mask_vol is not None and job.mask_vol.shape[0] != z_len:
        raise ValueError(f"Mask volume {_resolve_mask_path(job.file)} does not match Z dimension of {job.file}.")

    coord_width = len(str(max(job.vol.shape[1], job.vol.shape[2])))
    z_candidates = job.z_candidates
    if not z_candidates:
        raise ValueError(f"[{roi}] No Z indices available after applying dz to fused Zarr volume.")

    cancel_event = get_cancel_event()
    for y0, x0 in job.tile_origins:
        if cancel_event.is_set():
            raise TaskCancelledException("Cancelled by user")

        y_slice_tile = slice(y0, y0 + ZARR_TILE_SIZE)
        x_slice_tile = slice(x0, x0 + ZARR_TILE_SIZE)

        plane = _iterative_max_projection(job.vol, z_candidates, y_slice=y_slice_tile, x_slice=x_slice_tile)
        zero_fraction = np.mean(plane == 0)
        skip_tile = zero_fraction > ZERO_PIXEL_SKIP_THRESHOLD

        other_max = None
        if other_vol is not None:
            other_plane = _iterative_max_projection(other_vol, z_candidates, y_slice=y_slice_tile, x_slice=x_slice_tile)
            if not skip_tile:
                other_zero_fraction = np.mean(other_plane == 0)
                skip_tile = other_zero_fraction > ZERO_PIXEL_SKIP_THRESHOLD
            if not skip_tile:
                other_max = other_plane.max(axis=2)

        if skip_tile:
            if reporter is not None:
                reporter.advance()
            continue

        cyx_u16 = _prep_slab(
            plane,
            ch_idx=selected_indices,
            channel_axis=2,
            crop_slices=None,
            filter_before=True,
            append_max=other_max,
            apply_filter=False,
        )
        cyx_u16 = _resize_uint16(cyx_u16, (1.0, upscale, upscale))
        out_name = _format_tile_maxproj_filename(job.file.stem, roi, y0, x0, coord_width=coord_width)
        out_file = out_dir / out_name
        _write_tiff(
            out_file,
            cyx_u16,
            axes="CYX",
            names=out_names,
            channels_arg=channels,
            upscale=upscale,
        )
        if job.mask_vol is not None:
            mask_tile = _iterative_max_projection(job.mask_vol, z_candidates, y_slice=y_slice_tile, x_slice=x_slice_tile)
            mask_tile = _squeeze_mask(mask_tile)
            if mask_tile.ndim != 2:
                raise ValueError("Mask tile max-projection expected 2D data.")
            resized_mask = _resize_mask(mask_tile, (upscale, upscale))
            mask_out = out_dir / _mask_filename(out_name)
            _write_mask_tiff(mask_out, resized_mask, axes="YX")
        if reporter is not None:
            reporter.advance()


def _extract_ortho_slices(
    *,
    file: Path,
    roi: str,
    out_dir: Path,
    channels: str | None,
    crop: int,
    n: int,
    anisotropy: int,
    upscale: float,
    max_from_path: Path | None,
    mask_path: Path | None,
    enrich_boundaries: Path | None,
    seed: int | None,
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> None:
    reporter = _normalize_reporter(progress)
    size_str = _format_size(_path_size_cached(str(file.resolve())))
    logger.info(f"Ortho: {file.name} [{size_str}]")

    ctx = build_extraction_context(
        file,
        roi,
        channels=channels,
        max_from_path=max_from_path,
        mask_path=mask_path,
        enrich_path=enrich_boundaries,
        upscale=upscale,
        anisotropy=anisotropy,
        out_dir=out_dir,
    )

    vol = ctx.vol
    selected_indices = ctx.selected_indices
    out_names = ctx.out_names
    other_vol = ctx.other_vol
    mask_vol = ctx.mask_vol
    enrich_mask_vol = ctx.enrich_mask_vol

    _, y_len, x_len, _ = vol.shape

    y_eff = y_len - 2 * crop
    x_eff = x_len - 2 * crop
    if y_eff <= 0 or x_eff <= 0:
        raise ValueError("Image after cropping is empty.")

    oversample = 10 if enrich_mask_vol is not None else 1
    base_y = (
        np.linspace(int(0.1 * y_eff), int(0.9 * y_eff), n * oversample).astype(int) + crop
    ).tolist()
    base_x = (
        np.linspace(int(0.1 * x_eff), int(0.9 * x_eff), n * oversample).astype(int) + crop
    ).tolist()
    if enrich_mask_vol is not None:
        base_y = _select_high_diversity_positions(base_y, enrich_mask_vol, "y", n)
        base_x = _select_high_diversity_positions(base_x, enrich_mask_vol, "x", n)
    y_bases = base_y
    x_bases = base_x

    max_width_pre_upscale = int(MAX_WIDTH_AFTER_UPSCALE / upscale)

    rng = np.random.default_rng(seed)
    cancel_event = get_cancel_event()

    requests: list[OrthoSliceRequest] = []
    for base_yi in y_bases:
        if cancel_event.is_set():
            raise TaskCancelledException("Cancelled by user")
        x_slice = _compute_perpendicular_slice(
            axis_len=x_len, crop=crop, max_width=max_width_pre_upscale, rng=rng
        )
        expanded_y = _expand_positions_with_context([base_yi], crop=crop, axis_len=y_len)
        for yi in expanded_y:
            requests.append(
                _ortho_output_request(
                    file_stem=file.stem,
                    roi=roi,
                    out_dir=out_dir,
                    axis="y",
                    position=yi,
                    perpendicular_slice=x_slice,
                )
            )

    for base_xi in x_bases:
        if cancel_event.is_set():
            raise TaskCancelledException("Cancelled by user")
        y_slice = _compute_perpendicular_slice(
            axis_len=y_len, crop=crop, max_width=max_width_pre_upscale, rng=rng
        )
        expanded_x = _expand_positions_with_context([base_xi], crop=crop, axis_len=x_len)
        for xi in expanded_x:
            requests.append(
                _ortho_output_request(
                    file_stem=file.stem,
                    roi=roi,
                    out_dir=out_dir,
                    axis="x",
                    position=xi,
                    perpendicular_slice=y_slice,
                )
            )

    _write_requested_ortho_slices(
        vol=vol,
        mask_vol=mask_vol,
        other_vol=other_vol,
        requests=requests,
        selected_indices=selected_indices,
        out_names=out_names,
        anisotropy=anisotropy,
        upscale=upscale,
        channels_arg=channels,
        progress=reporter,
    )
