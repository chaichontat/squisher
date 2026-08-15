#!/usr/bin/env python
"""Roughly align paired light-sheet acquisitions by low-res phase correlation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from squisher_lightsheet import ngff
from squisher_lightsheet import phase_metrics
from squisher_lightsheet import qc as qc_core
from squisher_lightsheet.tiff import choose_tiff_source_level, spatial_shape_array_zyx_from_axes


DIMENSIONS = ("z", "y", "x")
SIDES = ("L", "R")
PHASE_PLANES = {
    "xy": ("y", "x"),
    "xz": ("z", "x"),
    "zyx": ("z", "y", "x"),
}


@dataclass(frozen=True)
class TileRecord:
    tile: str
    side: str
    path: Path
    translation_zyx_um: np.ndarray
    scale_zyx_um: np.ndarray
    shape_zyx: np.ndarray
    axes: str


@dataclass(frozen=True)
class CanvasGeometry:
    level_factor: int
    level_spacing_zyx_um: np.ndarray
    global_min_zyx_um: np.ndarray
    global_max_zyx_um: np.ndarray
    shape_zyx: np.ndarray
    level_spacing_yx_um: np.ndarray
    global_min_yx_um: np.ndarray
    global_max_yx_um: np.ndarray
    shape_yx: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-input", type=Path, required=True)
    parser.add_argument("--output-position", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument(
        "--search-margin-px",
        type=int,
        default=64,
        help="Pixels to include around the metadata L/R overlap before phase correlation.",
    )
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument(
        "--z-slab-planes",
        type=int,
        default=1,
        help="For x-join rough phase, use this many level-sampled Z planes around the mosaic center.",
    )
    parser.add_argument(
        "--seam-fraction",
        type=float,
        default=0.10,
        help="For z-join XZ phase correlation, use only this seam-adjacent fraction of the tile depth.",
    )
    return parser.parse_args()


def tile_shape_and_axes(path: Path) -> tuple[np.ndarray, str]:
    if path.suffix == ".zarr":
        import zarr

        root = zarr.open(str(path), mode="r")
        zarray = ngff.level_array(root, context=path)
        shape = tuple(int(value) for value in zarray.shape)
        axes = ngff.axes(root, zarray)
        if axes == "CZYX":
            return np.asarray(shape[1:4], dtype=np.int64), axes
        if axes == "ZYX":
            return np.asarray(shape, dtype=np.int64), axes
        raise ValueError(f"{path} has unsupported axes {axes!r}")

    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        axes = str(series.axes)
    if axes == "CZYX":
        return np.asarray(shape[1:4], dtype=np.int64), axes
    if axes == "ZYX":
        return np.asarray(shape, dtype=np.int64), axes
    raise ValueError(f"{path} has unsupported axes {axes!r}")


def base_zarr_array(zarr_obj: Any) -> Any:
    """Return the base-resolution array from array or grouped OME-TIFF stores."""
    return ngff.level_array(zarr_obj)


def zarr_axes(zarray: Any) -> str:
    attrs = dict(getattr(zarray, "attrs", {}) or {})
    dimensions = attrs.get("_ARRAY_DIMENSIONS")
    if dimensions:
        return "".join(str(dim)[0].upper() for dim in dimensions)
    return "CZYX" if len(zarray.shape) == 4 else "ZYX"


def sampled_source_array(tile: TileRecord, desired_factor: np.ndarray) -> tuple[Any, np.ndarray, Any | None]:
    import dask.array as da
    import zarr

    if tile.path.suffix == ".zarr":
        zarray = base_zarr_array(zarr.open(str(tile.path), mode="r"))
        return da.from_zarr(zarray), np.ones(3, dtype=np.int64), None

    import tifffile

    source_level, source_factor = choose_tiff_source_level(tile.path, desired_factor)
    store = tifffile.imread(tile.path, aszarr=True, level=source_level)
    zarray = base_zarr_array(zarr.open(store, mode="r"))
    return da.from_zarr(zarray), source_factor, store


def source_slices_for_factor(
    tile: TileRecord,
    *,
    source_factor_zyx: np.ndarray,
    desired_factor_zyx: np.ndarray,
) -> tuple[slice, slice, slice]:
    remaining_step = np.maximum(
        1,
        np.ceil(np.asarray(desired_factor_zyx, dtype=np.float64) / source_factor_zyx.astype(np.float64)).astype(np.int64),
    )
    z_step = int(-remaining_step[0] if tile.scale_zyx_um[0] < 0 else remaining_step[0])
    y_step = int(-remaining_step[1] if tile.scale_zyx_um[1] < 0 else remaining_step[1])
    x_step = int(-remaining_step[2] if tile.scale_zyx_um[2] < 0 else remaining_step[2])
    return slice(None, None, z_step), slice(None, None, y_step), slice(None, None, x_step)


def load_tiles(position_payload: dict[str, Any]) -> list[TileRecord]:
    tiles = []
    for record in position_payload["tiles"]:
        path = Path(record["path"])
        shape_zyx, axes = tile_shape_and_axes(path)
        side = record.get("side")
        if side not in SIDES:
            raise ValueError(f"{path} has side={side!r}; expected one of {SIDES}")
        tiles.append(
            TileRecord(
                tile=record["tile"],
                side=side,
                path=path,
                translation_zyx_um=np.asarray([record["translation_um"][dim] for dim in DIMENSIONS], dtype=np.float64),
                scale_zyx_um=np.asarray([record["scale_um"][dim] for dim in DIMENSIONS], dtype=np.float64),
                shape_zyx=shape_zyx,
                axes=axes,
            )
        )
    return tiles


def tile_bounds_zyx_um(tile: TileRecord) -> tuple[np.ndarray, np.ndarray]:
    start = tile.translation_zyx_um
    stop = start + tile.shape_zyx.astype(np.float64) * tile.scale_zyx_um
    return np.minimum(start, stop), np.maximum(start, stop)


def build_geometry(tiles: list[TileRecord], *, level: int) -> CanvasGeometry:
    if level < 0:
        raise ValueError("--level must be non-negative")
    level_factor = 2**level
    bounds = [tile_bounds_zyx_um(tile) for tile in tiles]
    global_min = np.min([item[0] for item in bounds], axis=0)
    global_max = np.max([item[1] for item in bounds], axis=0)
    base_spacing_yx = np.abs(tiles[0].scale_zyx_um[1:3])
    for tile in tiles[1:]:
        if not np.allclose(np.abs(tile.scale_zyx_um[1:3]), base_spacing_yx):
            raise ValueError("All tiles must have the same absolute y/x scale for rough center-z fusion")
    level_spacing_yx = base_spacing_yx * level_factor
    shape_yx = np.ceil((global_max[1:3] - global_min[1:3]) / level_spacing_yx).astype(np.int64)
    level_spacing_zyx = np.asarray(
        [np.abs(tiles[0].scale_zyx_um[0]), base_spacing_yx[0], base_spacing_yx[1]],
        dtype=np.float64,
    ) * level_factor
    shape_zyx = np.ceil((global_max - global_min) / level_spacing_zyx).astype(np.int64)
    return CanvasGeometry(
        level_factor=level_factor,
        level_spacing_zyx_um=level_spacing_zyx,
        global_min_zyx_um=global_min,
        global_max_zyx_um=global_max,
        shape_zyx=shape_zyx,
        level_spacing_yx_um=level_spacing_yx,
        global_min_yx_um=global_min[1:3],
        global_max_yx_um=global_max[1:3],
        shape_yx=shape_yx,
    )


def sampled_center_z_plane(tile: TileRecord, *, channel: int, level_factor: int) -> np.ndarray:
    desired_factor = np.asarray([1, level_factor, level_factor], dtype=np.int64)
    array, source_factor, store = sampled_source_array(tile, desired_factor)
    try:
        source_shape_zyx = spatial_shape_array_zyx_from_axes(tuple(array.shape), tile.axes)
        z_index = int(round((tile.shape_zyx[0] // 2) / max(int(source_factor[0]), 1)))
        z_index = min(max(z_index, 0), int(source_shape_zyx[0]) - 1)
        if tile.scale_zyx_um[0] < 0:
            z_index = int(source_shape_zyx[0] - 1 - z_index)
        _z_slice, y_slice, x_slice = source_slices_for_factor(
            tile,
            source_factor_zyx=source_factor,
            desired_factor_zyx=desired_factor,
        )
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            plane = array[channel, z_index, y_slice, x_slice]
        elif tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
            plane = array[z_index, y_slice, x_slice]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        return np.asarray(plane.compute(), dtype=np.float32)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def sampled_tile_volume(tile: TileRecord, *, channel: int, level_factor: int) -> np.ndarray:
    desired_factor = np.asarray([level_factor, level_factor, level_factor], dtype=np.int64)
    array, source_factor, store = sampled_source_array(tile, desired_factor)
    try:
        z_slice, y_slice, x_slice = source_slices_for_factor(
            tile,
            source_factor_zyx=source_factor,
            desired_factor_zyx=desired_factor,
        )
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            volume = array[channel, z_slice, y_slice, x_slice]
        elif tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
            volume = array[z_slice, y_slice, x_slice]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        return np.asarray(volume.compute(), dtype=np.float32)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def sampled_tile_volume_z_range(
    tile: TileRecord,
    *,
    channel: int,
    level_factor: int,
    sampled_z_start: int,
    sampled_z_stop: int,
) -> np.ndarray:
    if sampled_z_start < 0 or sampled_z_stop <= sampled_z_start:
        raise ValueError(f"Invalid sampled z range [{sampled_z_start}, {sampled_z_stop}) for tile {tile.tile}")
    desired_factor = np.asarray([level_factor, level_factor, level_factor], dtype=np.int64)
    array, source_factor, store = sampled_source_array(tile, desired_factor)
    try:
        z_slice, y_slice, x_slice = source_slices_for_factor(
            tile,
            source_factor_zyx=source_factor,
            desired_factor_zyx=desired_factor,
        )
        source_shape_zyx = spatial_shape_array_zyx_from_axes(tuple(array.shape), tile.axes)
        sampled_z_size = len(range(*z_slice.indices(int(source_shape_zyx[0]))))
        sampled_z_stop = min(int(sampled_z_stop), sampled_z_size)
        if sampled_z_stop <= sampled_z_start:
            return np.zeros((0, 0, 0), dtype=np.float32)
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            volume = array[channel, z_slice, y_slice, x_slice][sampled_z_start:sampled_z_stop]
        elif tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
            volume = array[z_slice, y_slice, x_slice][sampled_z_start:sampled_z_stop]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        return np.asarray(volume.compute(), dtype=np.float32)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def sampled_tile_center_z_slab(
    tile: TileRecord,
    *,
    channel: int,
    source_z_start: int,
    source_z_stop: int,
    yx_level_factor: int,
) -> np.ndarray:
    if source_z_start < 0 or source_z_stop > int(tile.shape_zyx[0]) or source_z_stop <= source_z_start:
        raise ValueError(
            f"Invalid source z slab [{source_z_start}, {source_z_stop}) for tile {tile.tile} "
            f"with z size {int(tile.shape_zyx[0])}"
        )

    desired_factor = np.asarray([1, yx_level_factor, yx_level_factor], dtype=np.int64)
    array, source_factor, store = sampled_source_array(tile, desired_factor)
    try:
        source_shape_zyx = spatial_shape_array_zyx_from_axes(tuple(array.shape), tile.axes)
        source_z_start = int(np.floor(source_z_start / max(int(source_factor[0]), 1)))
        source_z_stop = int(np.ceil(source_z_stop / max(int(source_factor[0]), 1)))
        source_z_start = min(max(source_z_start, 0), int(source_shape_zyx[0]))
        source_z_stop = min(max(source_z_stop, source_z_start + 1), int(source_shape_zyx[0]))
        if tile.scale_zyx_um[0] < 0:
            z_start = int(source_shape_zyx[0] - 1 - source_z_start)
            z_stop = int(source_shape_zyx[0] - source_z_stop - 1)
            z_slice = slice(z_start, None if z_stop < 0 else z_stop, -1)
        else:
            z_slice = slice(source_z_start, source_z_stop)
        _z_slice, y_slice, x_slice = source_slices_for_factor(
            tile,
            source_factor_zyx=source_factor,
            desired_factor_zyx=desired_factor,
        )
        if tile.axes == "CZYX":
            if channel < 0 or channel >= int(array.shape[0]):
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
            volume = array[channel, z_slice, y_slice, x_slice]
        elif tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
            volume = array[z_slice, y_slice, x_slice]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        return np.asarray(volume.compute(), dtype=np.float32)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def place_max(canvas: np.ndarray, image: np.ndarray, start_yx: tuple[int, int]) -> None:
    qc_core.place_max(canvas, image, start_yx)


def place_max_nd(canvas: np.ndarray, image: np.ndarray, start: tuple[int, ...]) -> None:
    if canvas.ndim != image.ndim or canvas.ndim != len(start):
        raise ValueError("canvas, image, and start must have matching dimensionality")
    src_slices = []
    dst_slices = []
    for axis, start_index in enumerate(start):
        if start_index >= canvas.shape[axis]:
            return
        src0 = max(0, -start_index)
        dst0 = max(0, start_index)
        size = min(image.shape[axis] - src0, canvas.shape[axis] - dst0)
        if size <= 0:
            return
        src_slices.append(slice(src0, src0 + size))
        dst_slices.append(slice(dst0, dst0 + size))
    canvas[tuple(dst_slices)] = np.maximum(canvas[tuple(dst_slices)], image[tuple(src_slices)])


def render_center_z_canvases(
    tiles: list[TileRecord],
    *,
    geometry: CanvasGeometry,
    channel: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    images = {side: np.zeros(tuple(geometry.shape_yx), dtype=np.float32) for side in SIDES}
    coverage = {side: np.zeros(tuple(geometry.shape_yx), dtype=bool) for side in SIDES}
    rows = []
    for tile in tiles:
        start_um, _stop_um = tile_bounds_zyx_um(tile)
        start_yx = np.rint((start_um[1:3] - geometry.global_min_yx_um) / geometry.level_spacing_yx_um).astype(int)
        plane = sampled_center_z_plane(tile, channel=channel, level_factor=geometry.level_factor)
        place_max(images[tile.side], plane, (int(start_yx[0]), int(start_yx[1])))
        place_max(coverage[tile.side], np.ones(plane.shape, dtype=bool), (int(start_yx[0]), int(start_yx[1])))
        rows.append(
            {
                "tile": tile.tile,
                "side": tile.side,
                "center_z_raw_index": int(tile.shape_zyx[0] // 2),
                "level_start_yx": [int(start_yx[0]), int(start_yx[1])],
                "sampled_shape_yx": [int(plane.shape[0]), int(plane.shape[1])],
            }
        )
        print(f"placed {tile.tile} side={tile.side} start_yx={start_yx.tolist()} shape={list(plane.shape)}", flush=True)
    return images, coverage, rows


def render_xz_projection_canvases(
    tiles: list[TileRecord],
    *,
    geometry: CanvasGeometry,
    channel: int,
    z_range_px: tuple[int, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    if z_range_px is None:
        z_start_px = 0
        z_stop_px = int(geometry.shape_zyx[0])
    else:
        z_start_px = max(0, min(int(geometry.shape_zyx[0]), int(z_range_px[0])))
        z_stop_px = max(z_start_px, min(int(geometry.shape_zyx[0]), int(z_range_px[1])))
    images = {side: np.zeros((z_stop_px - z_start_px, int(geometry.shape_zyx[2])), dtype=np.float32) for side in SIDES}
    coverage = {side: np.zeros((z_stop_px - z_start_px, int(geometry.shape_zyx[2])), dtype=bool) for side in SIDES}
    rows = []
    for tile in tiles:
        start_um, _stop_um = tile_bounds_zyx_um(tile)
        start_zyx = np.rint((start_um - geometry.global_min_zyx_um) / geometry.level_spacing_zyx_um).astype(int)
        sampled_z_size = int(np.ceil(int(tile.shape_zyx[0]) / int(geometry.level_factor)))
        tile_z_start = int(start_zyx[0])
        tile_z_stop = tile_z_start + sampled_z_size
        read_z_start = max(tile_z_start, z_start_px)
        read_z_stop = min(tile_z_stop, z_stop_px)
        if read_z_stop <= read_z_start:
            continue
        volume = sampled_tile_volume_z_range(
            tile,
            channel=channel,
            level_factor=geometry.level_factor,
            sampled_z_start=read_z_start - tile_z_start,
            sampled_z_stop=read_z_stop - tile_z_start,
        )
        projection = volume.max(axis=1)
        start_zx = (int(read_z_start - z_start_px), int(start_zyx[2]))
        place_max(images[tile.side], projection, start_zx)
        place_max(coverage[tile.side], np.ones(projection.shape, dtype=bool), start_zx)
        rows.append(
            {
                "tile": tile.tile,
                "side": tile.side,
                "level_start_zx": [int(read_z_start), int(start_zyx[2])],
                "phase_canvas_start_zx": [start_zx[0], start_zx[1]],
                "sampled_z_range": [int(read_z_start - tile_z_start), int(read_z_stop - tile_z_start)],
                "sampled_shape_zx": [int(projection.shape[0]), int(projection.shape[1])],
            }
        )
        print(
            f"placed-xz {tile.tile} side={tile.side} level_start_zx={[int(read_z_start), int(start_zyx[2])]} "
            f"phase_start_zx={list(start_zx)} sampled_z_range={[int(read_z_start - tile_z_start), int(read_z_stop - tile_z_start)]} "
            f"shape={list(projection.shape)}",
            flush=True,
        )
    return images, coverage, rows


def center_slab_range(shape_z: int, slab_planes: int) -> tuple[int, int]:
    if slab_planes < 1:
        raise ValueError("slab_planes must be >= 1")
    slab_planes = min(int(slab_planes), int(shape_z))
    center = int(shape_z // 2)
    start = max(0, center - slab_planes // 2)
    stop = min(int(shape_z), start + slab_planes)
    start = max(0, stop - slab_planes)
    return start, stop


def render_center_z_slab_canvases(
    tiles: list[TileRecord],
    *,
    geometry: CanvasGeometry,
    channel: int,
    slab_planes: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    native_z_spacing_um = abs(float(tiles[0].scale_zyx_um[0]))
    native_global_shape_z = int(
        np.ceil((geometry.global_max_zyx_um[0] - geometry.global_min_zyx_um[0]) / native_z_spacing_um)
    )
    slab_start, slab_stop = center_slab_range(native_global_shape_z, slab_planes)
    slab_shape = (slab_stop - slab_start, int(geometry.shape_zyx[1]), int(geometry.shape_zyx[2]))
    images = {side: np.zeros(slab_shape, dtype=np.float32) for side in SIDES}
    coverage = {side: np.zeros(slab_shape, dtype=bool) for side in SIDES}
    rows = []
    for tile in tiles:
        start_um, _stop_um = tile_bounds_zyx_um(tile)
        start_z = int(round((start_um[0] - geometry.global_min_zyx_um[0]) / native_z_spacing_um))
        start_yx = np.rint((start_um[1:3] - geometry.global_min_zyx_um[1:3]) / geometry.level_spacing_zyx_um[1:3]).astype(int)
        global_z_start = max(slab_start, start_z)
        global_z_stop = min(slab_stop, start_z + int(tile.shape_zyx[0]))
        if global_z_stop <= global_z_start:
            continue
        source_z_start = global_z_start - start_z
        source_z_stop = global_z_stop - start_z
        volume = sampled_tile_center_z_slab(
            tile,
            channel=channel,
            source_z_start=source_z_start,
            source_z_stop=source_z_stop,
            yx_level_factor=geometry.level_factor,
        )
        slab_start_zyx = (int(global_z_start) - slab_start, int(start_yx[0]), int(start_yx[1]))
        place_max_nd(images[tile.side], volume, slab_start_zyx)
        place_max_nd(coverage[tile.side], np.ones(volume.shape, dtype=bool), slab_start_zyx)
        rows.append(
            {
                "tile": tile.tile,
                "side": tile.side,
                "level_start_zyx": [int(start_z), int(start_yx[0]), int(start_yx[1])],
                "source_z_range": [int(source_z_start), int(source_z_stop)],
                "slab_start_zyx": [int(value) for value in slab_start_zyx],
                "sampled_shape_zyx": [int(value) for value in volume.shape],
            }
        )
        print(
            f"placed-slab {tile.tile} side={tile.side} start_zyx={[int(start_z), int(start_yx[0]), int(start_yx[1])]} "
            f"slab_start_zyx={list(slab_start_zyx)} shape={list(volume.shape)}",
            flush=True,
        )
    details = {
        "slab_range_z_px": [int(slab_start), int(slab_stop)],
        "slab_planes": int(slab_shape[0]),
        "z_sampling": "native_center_z_slab",
        "native_z_spacing_um": native_z_spacing_um,
        "global_shape_zyx": [int(native_global_shape_z), int(geometry.shape_zyx[1]), int(geometry.shape_zyx[2])],
        "yx_downsample_factor": int(geometry.level_factor),
    }
    return images, coverage, rows, details


def side_axis_bounds_um(tiles: list[TileRecord], *, axis_index: int) -> dict[str, tuple[float, float]]:
    bounds = {side: [] for side in SIDES}
    for tile in tiles:
        start_um, stop_um = tile_bounds_zyx_um(tile)
        bounds[tile.side].append((float(start_um[axis_index]), float(stop_um[axis_index])))
    return {
        side: (
            min(item[0] for item in side_bounds),
            max(item[1] for item in side_bounds),
        )
        for side, side_bounds in bounds.items()
        if side_bounds
    }


def side_axis_centers_um(tiles: list[TileRecord], *, axis_index: int) -> dict[str, float]:
    bounds = side_axis_bounds_um(tiles, axis_index=axis_index)
    return {side: (start + stop) / 2.0 for side, (start, stop) in bounds.items()}


def seam_band_mask(
    tiles: list[TileRecord],
    *,
    geometry: CanvasGeometry,
    axes: tuple[str, str],
    seam_fraction: float,
    overlap_um: float | None = None,
    overlap_fraction: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 < seam_fraction <= 1.0:
        raise ValueError("seam_fraction must be in (0, 1]")
    axis_to_index = {"z": 0, "y": 1, "x": 2}
    seam_axis = axes[0]
    seam_axis_index = axis_to_index[seam_axis]
    canvas_axis_index = 0
    side_bounds = side_axis_bounds_um(tiles, axis_index=seam_axis_index)
    if set(SIDES) - set(side_bounds):
        raise ValueError("Both L and R tiles are required to build a seam mask")
    overlap_start = max(side_bounds["L"][0], side_bounds["R"][0])
    overlap_stop = min(side_bounds["L"][1], side_bounds["R"][1])
    if overlap_stop <= overlap_start:
        raise ValueError(f"L/R do not overlap along {seam_axis}; cannot build seam mask")

    centers = side_axis_centers_um(tiles, axis_index=seam_axis_index)
    left_start, left_stop = side_bounds["L"]
    seam_um = left_start if centers["R"] < centers["L"] else left_stop
    if overlap_fraction and overlap_um:
        requested_band_um = abs(float(overlap_um)) * seam_fraction / abs(float(overlap_fraction))
    else:
        requested_band_um = min(left_stop - left_start, side_bounds["R"][1] - side_bounds["R"][0]) * seam_fraction
    band_um = min(float(requested_band_um), overlap_stop - overlap_start)

    if abs(seam_um - overlap_start) <= abs(seam_um - overlap_stop):
        band_start_um = overlap_start
        band_stop_um = min(overlap_stop, overlap_start + band_um)
    else:
        band_start_um = max(overlap_start, overlap_stop - band_um)
        band_stop_um = overlap_stop

    axis_min_um = geometry.global_min_zyx_um[seam_axis_index]
    axis_spacing_um = geometry.level_spacing_zyx_um[seam_axis_index]
    start_px = int(np.floor((band_start_um - axis_min_um) / axis_spacing_um))
    stop_px = int(np.ceil((band_stop_um - axis_min_um) / axis_spacing_um))
    mask = np.zeros(tuple(int(value) for value in (geometry.shape_zyx[0], geometry.shape_zyx[2])), dtype=bool)
    start_px = max(0, min(mask.shape[canvas_axis_index], start_px))
    stop_px = max(0, min(mask.shape[canvas_axis_index], stop_px))
    mask[start_px:stop_px, :] = True
    details = {
        "seam_axis": seam_axis,
        "seam_um": float(seam_um),
        "seam_fraction": float(seam_fraction),
        "seam_band_um": float(band_stop_um - band_start_um),
        "seam_band_um_requested": float(requested_band_um),
        "overlap_um": float(overlap_stop - overlap_start),
        "seam_band_range_um": [float(band_start_um), float(band_stop_um)],
        "seam_band_range_px": [int(start_px), int(stop_px)],
    }
    return mask, details


scale_u8 = qc_core.scale_u8
write_overlay = qc_core.write_overlay
write_overlay_scaled = qc_core.write_overlay_scaled
write_contact_sheet = qc_core.write_contact_sheet
empty_projection_canvases = qc_core.empty_projection_canvases
place_global_projections = qc_core.place_global_projections


def render_global_projection_canvases(
    tiles: list[TileRecord],
    *,
    geometry: CanvasGeometry,
    channel: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    projections = empty_projection_canvases(geometry.shape_zyx)
    rows = []
    for tile in tiles:
        start_um, _stop_um = tile_bounds_zyx_um(tile)
        start_zyx = np.rint((start_um - geometry.global_min_zyx_um) / geometry.level_spacing_zyx_um).astype(int)
        volume = sampled_tile_volume(tile, channel=channel, level_factor=geometry.level_factor)
        place_global_projections(projections, side=tile.side, volume=volume, start_zyx=start_zyx)
        rows.append(
            {
                "tile": tile.tile,
                "side": tile.side,
                "level_start_zyx": [int(value) for value in start_zyx],
                "sampled_shape_zyx": [int(value) for value in volume.shape],
            }
        )
        print(f"projected {tile.tile} side={tile.side} start_zyx={start_zyx.tolist()} shape={list(volume.shape)}", flush=True)
    return projections, rows


def write_global_projection_outputs(
    output_dir: Path,
    *,
    projections: dict[str, dict[str, np.ndarray]],
    level: int,
    channel: int,
    z_display_scale: float,
) -> tuple[list[tuple[str, Path]], Path]:
    outputs = []
    projection_y_scales = {"xy": 1.0, "xz": z_display_scale, "yz": z_display_scale}
    for name in ("xy", "xz", "yz"):
        path = output_dir / f"level{level}_phase_corrected_lr_{name}_isoZ_yellowOverlay_ch{channel}.png"
        write_overlay_scaled(
            path,
            left=projections["L"][name],
            right=projections["R"][name],
            y_scale=projection_y_scales[name],
        )
        outputs.append((name.upper(), path))
        print(path, flush=True)
    contact_sheet = output_dir / f"level{level}_phase_corrected_lr_isoZ_yellowOverlay_ch{channel}.png"
    write_contact_sheet(contact_sheet, outputs)
    print(contact_sheet, flush=True)
    return outputs, contact_sheet


def normalize_for_phase(image: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    valid = coverage & np.isfinite(image)
    positive = image[valid & (image > 0)]
    out = np.zeros(image.shape, dtype=np.float32)
    if positive.size == 0:
        return out
    low, high = np.percentile(positive, [1.0, 99.5])
    clipped = np.clip(image, low, high)
    values = clipped[valid]
    centered = clipped - float(np.median(values))
    denom = max(float(np.percentile(np.abs(centered[valid]), 95.0)), 1.0)
    out[valid] = centered[valid] / denom
    return out


def expanded_bbox(mask: np.ndarray, *, margin: int) -> tuple[slice, ...]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("The metadata L/R overlap is empty; increase initial overlap before phase correlation")
    starts = coords.min(axis=0)
    stops = coords.max(axis=0) + 1
    slices = []
    for axis, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        axis_start = max(0, int(start) - margin)
        axis_stop = min(mask.shape[axis], int(stop) + margin)
        slices.append(slice(axis_start, axis_stop))
    return tuple(slices)


def estimate_shift_px(
    images: dict[str, np.ndarray],
    coverage: dict[str, np.ndarray],
    *,
    axes: tuple[str, str],
    phase_mask: np.ndarray | None = None,
    crop_to_overlap: bool = True,
    search_margin_px: int,
    upsample_factor: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation

    phase_coverage = coverage
    if phase_mask is not None:
        if phase_mask.shape != coverage["L"].shape:
            raise ValueError(f"phase_mask shape {phase_mask.shape} does not match coverage shape {coverage['L'].shape}")
        phase_coverage = {side: side_coverage & phase_mask for side, side_coverage in coverage.items()}
    overlap = phase_coverage["L"] & phase_coverage["R"]
    if crop_to_overlap:
        crop_basis = phase_coverage["L"] | phase_coverage["R"] if phase_mask is not None else overlap
        crop_slices = expanded_bbox(crop_basis, margin=0 if phase_mask is not None else search_margin_px)
    else:
        crop_slices = tuple(slice(0, size) for size in images["L"].shape)
    fixed = normalize_for_phase(images["L"][crop_slices], phase_coverage["L"][crop_slices])
    moving = normalize_for_phase(images["R"][crop_slices], phase_coverage["R"][crop_slices])
    fixed_mask = phase_coverage["L"][crop_slices]
    moving_mask = phase_coverage["R"][crop_slices]
    if phase_mask is not None:
        shift, error, phase = phase_cross_correlation(
            fixed,
            moving,
            reference_mask=fixed_mask,
            moving_mask=moving_mask,
            overlap_ratio=0.05,
        )
        phase_mode = "masked"
    else:
        shift, error, phase = phase_cross_correlation(
            fixed,
            moving,
            upsample_factor=upsample_factor,
            normalization="phase",
        )
        phase_mode = "unmasked"
    shift = np.asarray(shift, dtype=np.float64)
    shifted = ndimage.shift(moving, shift=shift, order=1, mode="constant", cval=0.0, prefilter=False)
    common = fixed_mask & ndimage.shift(
        moving_mask.astype(np.float32),
        shift=shift,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(bool)
    before_corr = corrcoef_on_mask(fixed, moving, fixed_mask & moving_mask)
    after_corr = corrcoef_on_mask(fixed, shifted, common)
    axis_suffix = "".join(axes)
    details = {
        "phase_axes": list(axes),
        f"crop_{axis_suffix}": [[axis_slice.start, axis_slice.stop] for axis_slice in crop_slices],
        "overlap_pixels": int(overlap.sum()),
        "phase_mask_pixels": None if phase_mask is None else int(phase_mask.sum()),
        "phase_valid_pixels": None if phase_mask is None else {side: int(phase_coverage[side].sum()) for side in SIDES},
        "crop_to_overlap": bool(crop_to_overlap),
        "search_margin_px": int(search_margin_px),
        "upsample_factor": int(upsample_factor),
        "phase_mode": phase_mode,
        "masked_overlap_ratio": 0.05 if phase_mask is not None else None,
        "phase_error": float(error),
        "phase": float(phase),
        "corr_before": before_corr,
        "corr_after": after_corr,
    }
    return shift, details


def estimate_shift_yx_px(
    images: dict[str, np.ndarray],
    coverage: dict[str, np.ndarray],
    *,
    search_margin_px: int,
    upsample_factor: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    return estimate_shift_px(
        images,
        coverage,
        axes=("y", "x"),
        phase_mask=None,
        crop_to_overlap=True,
        search_margin_px=search_margin_px,
        upsample_factor=upsample_factor,
    )


def corrcoef_on_mask(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float | None:
    return phase_metrics.corrcoef_on_mask(fixed, moving, mask)


def shifted_payload_by_axes(
    payload: dict[str, Any],
    *,
    shift_um: np.ndarray,
    axes: tuple[str, str],
    phase_plane: str,
    phase_details: dict[str, Any],
    output_position: Path,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(payload))
    for record in updated["tiles"]:
        if record.get("side") == "R":
            for axis, value in zip(axes, shift_um, strict=True):
                record["translation_um"][axis] = float(record["translation_um"][axis] + value)
    diagnostics = updated.setdefault("diagnostics", {})
    axis_suffix = "".join(axes)
    diagnostics[f"level{phase_details['level']}_{phase_plane}_phase_alignment"] = {
        **phase_details,
        f"shift_to_apply_R_um_{axis_suffix}": [float(value) for value in shift_um],
        "output_position": str(output_position.resolve()),
        "description": f"R {'/'.join(axes)} translations shifted after phase correlating level-{phase_details['level']} {phase_plane.upper()} side mosaics.",
    }
    updated["source"] = f"{updated.get('source', 'position file')} + level-{phase_details['level']} {phase_plane.upper()} phase rough alignment"
    return updated


def shifted_payload(
    payload: dict[str, Any],
    *,
    shift_yx_um: np.ndarray,
    phase_details: dict[str, Any],
    output_position: Path,
) -> dict[str, Any]:
    return shifted_payload_by_axes(
        payload,
        shift_um=shift_yx_um,
        axes=("y", "x"),
        phase_plane="xy",
        phase_details=phase_details,
        output_position=output_position,
    )


def main() -> int:
    args = parse_args()
    from squisher_lightsheet.rough_phase import rough_phase_align

    payload = json.loads(args.position_input.read_text())
    phase_plane = "xz" if payload.get("diagnostics", {}).get("join_axis") == "z" else ("zyx" if args.z_slab_planes > 1 else "xy")
    iso_tag = "_isoZ" if phase_plane == "xz" else ""
    rough_phase_align(
        position_input=args.position_input,
        output_position=args.output_position,
        output_dir=args.output_dir,
        channel=args.channel,
        level=args.level,
        search_margin_px=args.search_margin_px,
        upsample_factor=args.upsample_factor,
        seam_fraction=args.seam_fraction,
        z_slab_planes=args.z_slab_planes,
    )
    print(args.output_position.resolve(), flush=True)
    print((args.output_dir / f"level{args.level}_metadata_initial_{phase_plane}{iso_tag}_yellowOverlay_ch{args.channel}.png").resolve(), flush=True)
    print((args.output_dir / f"level{args.level}_phase_corrected_{phase_plane}{iso_tag}_yellowOverlay_ch{args.channel}.png").resolve(), flush=True)
    print((args.output_dir / f"level{args.level}_{phase_plane}_phase_alignment_ch{args.channel}.json").resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
