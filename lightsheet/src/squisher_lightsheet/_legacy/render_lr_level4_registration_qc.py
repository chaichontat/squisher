#!/usr/bin/env python
"""Render level-N L/R registration QC overlays colored by acquisition side."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from squisher_lightsheet import qc as qc_core


DIMENSIONS = ("z", "y", "x")
SIDES = ("L", "R")


@dataclass(frozen=True)
class OverlayLabel:
    text: str
    yx: tuple[float, float]


@dataclass(frozen=True)
class RenderGeometry:
    level_factor: int
    spacing_zyx_um: np.ndarray
    level_spacing_zyx_um: np.ndarray
    global_min_zyx_um: np.ndarray
    global_max_zyx_um: np.ndarray
    shape_zyx: np.ndarray
    bounds_zyx_um: list[tuple[np.ndarray, np.ndarray]]


def parse_args(
    *,
    default_position_input: Path | None = None,
    default_registration_input: Path | None = None,
    default_output_dir: Path | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--position-input",
        type=Path,
        default=default_position_input,
        required=default_position_input is None,
    )
    parser.add_argument(
        "--registration-input",
        type=Path,
        default=default_registration_input,
        required=default_registration_input is None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        required=default_output_dir is None,
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--center-y-xz", action="store_true", help="Also render an XZ slab at the center y position.")
    parser.add_argument("--center-y-um", type=float, default=None, help="Center-y position for --center-y-xz.")
    parser.add_argument("--center-z-um", type=float, default=None, help="Center-z position for full-affine XY plane QC.")
    parser.add_argument("--slab-half-px", type=int, default=4, help="Half-width of the center-y XZ slab at the sampled level.")
    parser.add_argument("--skip-global-projections", action="store_true")
    parser.add_argument("--full-affine-planes", action="store_true", help="Render QC planes by resampling through the full affine.")
    parser.add_argument("--center-z-only", action="store_true", help="For --full-affine-planes, render only the center-z XY PNG.")
    return parser.parse_args()


def load_stitch_module() -> Any:
    script = Path(__file__).resolve().parent / "stitch_20x_tl_multiview.py"
    spec = importlib.util.spec_from_file_location("stitch_20x_tl_multiview_for_qc", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def side_by_tile(position_payload: dict[str, Any]) -> dict[str, str]:
    return {str(Path(record["path"]).resolve()): record["side"] for record in position_payload["tiles"]}


def registered_bounds_um(tile: Any, param: Any, stitch: Any) -> tuple[np.ndarray, np.ndarray]:
    correction_um = np.asarray(stitch.affine_translation_zyx(param), dtype=np.float64)
    origin_um = np.asarray([stitch.tile_sim_translation(tile)[dim] for dim in DIMENSIONS], dtype=np.float64)
    scale_um = np.asarray([stitch.tile_sim_scale(tile)[dim] for dim in DIMENSIONS], dtype=np.float64)
    shape_zyx = np.asarray(stitch.tile_shape_zyx(tile), dtype=np.float64)
    start_um = origin_um + correction_um
    stop_um = start_um + shape_zyx * scale_um
    return np.minimum(start_um, stop_um), np.maximum(start_um, stop_um)


def affine_matrix_zyx(param: Any) -> np.ndarray:
    data = np.asarray(param.data if hasattr(param, "data") else param, dtype=np.float64)
    while data.ndim > 2:
        data = data[0]
    if data.shape[0] < 4 or data.shape[1] < 4:
        raise ValueError(f"Expected 4x4 affine matrix, got {data.shape}")
    return np.asarray(data[:4, :4], dtype=np.float64)


def stage_origin_zyx_um(tile: Any, stitch: Any) -> np.ndarray:
    return np.asarray([stitch.tile_sim_translation(tile)[dim] for dim in DIMENSIONS], dtype=np.float64)


def tile_extent_zyx_um(tile: Any, stitch: Any) -> np.ndarray:
    scale_um = np.asarray([stitch.tile_sim_scale(tile)[dim] for dim in DIMENSIONS], dtype=np.float64)
    shape_zyx = np.asarray(stitch.tile_shape_zyx(tile), dtype=np.float64)
    return shape_zyx * scale_um


def full_affine_from_local_um(tile: Any, param: Any, stitch: Any) -> np.ndarray:
    stage = np.eye(4, dtype=np.float64)
    stage[:3, 3] = stage_origin_zyx_um(tile, stitch)
    return affine_matrix_zyx(param) @ stage


def full_affine_registered_bounds_um(tile: Any, param: Any, stitch: Any) -> tuple[np.ndarray, np.ndarray]:
    extent = tile_extent_zyx_um(tile, stitch)
    corners = np.array(
        [
            [z, y, x]
            for z in (0.0, extent[0])
            for y in (0.0, extent[1])
            for x in (0.0, extent[2])
        ],
        dtype=np.float64,
    )
    full_affine = full_affine_from_local_um(tile, param, stitch)
    homogeneous = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = (full_affine @ homogeneous.T).T[:, :3]
    return transformed.min(axis=0), transformed.max(axis=0)


def geometry_from_bounds(
    tiles: list[Any],
    bounds: list[tuple[np.ndarray, np.ndarray]],
    *,
    level: int,
) -> RenderGeometry:
    if level < 0:
        raise ValueError("--level must be non-negative")
    level_factor = 2**level
    spacing = np.asarray([tiles[0].spacing[dim] for dim in DIMENSIONS], dtype=np.float64)
    level_spacing = spacing * level_factor
    global_min = np.min([item[0] for item in bounds], axis=0)
    global_max = np.max([item[1] for item in bounds], axis=0)
    shape_zyx = np.ceil((global_max - global_min) / level_spacing).astype(int)
    return RenderGeometry(
        level_factor=level_factor,
        spacing_zyx_um=spacing,
        level_spacing_zyx_um=level_spacing,
        global_min_zyx_um=global_min,
        global_max_zyx_um=global_max,
        shape_zyx=shape_zyx,
        bounds_zyx_um=bounds,
    )


def build_geometry(tiles: list[Any], params: list[Any], *, level: int, stitch: Any) -> RenderGeometry:
    return geometry_from_bounds(
        tiles,
        [registered_bounds_um(tile, param, stitch) for tile, param in zip(tiles, params, strict=True)],
        level=level,
    )


def build_full_affine_geometry(tiles: list[Any], params: list[Any], *, level: int, stitch: Any) -> RenderGeometry:
    return geometry_from_bounds(
        tiles,
        [full_affine_registered_bounds_um(tile, param, stitch) for tile, param in zip(tiles, params, strict=True)],
        level=level,
    )


def source_shape_zyx(tile: Any, source_shape: tuple[int, ...]) -> tuple[int, int, int]:
    if tile.axes == "CZYX":
        return int(source_shape[1]), int(source_shape[2]), int(source_shape[3])
    if tile.axes == "ZYX":
        return int(source_shape[0]), int(source_shape[1]), int(source_shape[2])
    raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")


def remaining_sample_steps(tile: Any, *, source_shape: tuple[int, ...], target_level_factor: int) -> tuple[int, int, int]:
    full_shape_zyx = np.asarray(stitch_shape_zyx(tile), dtype=np.float64)
    source_shape_array = np.asarray(source_shape_zyx(tile, source_shape), dtype=np.float64)
    source_factors = np.maximum(np.rint(full_shape_zyx / source_shape_array).astype(int), 1)
    steps = []
    for source_factor in source_factors:
        steps.append(max(1, int(round(target_level_factor / max(int(source_factor), 1)))))
    return tuple(steps)


def stitch_shape_zyx(tile: Any) -> tuple[int, int, int]:
    if tile.axes == "CZYX":
        return int(tile.shape[1]), int(tile.shape[2]), int(tile.shape[3])
    if tile.axes == "ZYX":
        return int(tile.shape[0]), int(tile.shape[1]), int(tile.shape[2])
    raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")


def sampled_tile(tile: Any, *, channel: int, level: int, stitch: Any) -> np.ndarray:
    import dask.array as da

    source_level, _available_levels = stitch.fusion_source_level_for_tile(tile, int(level))
    store = None
    try:
        zarray, store = stitch.open_tile_array(tile, source_level=source_level)
        source_tile = stitch.fusion_tile_for_source_array(
            tile,
            tuple(int(value) for value in zarray.shape),
            source_level=source_level,
        )
        steps_zyx = remaining_sample_steps(
            tile,
            source_shape=tuple(int(value) for value in zarray.shape),
            target_level_factor=2**int(level),
        )
        array = da.from_zarr(zarray)
        z_flip, y_flip, x_flip = stitch.tile_flip_axes_zyx(source_tile)
        z_slice = slice(None, None, -steps_zyx[0] if z_flip else steps_zyx[0])
        y_slice = slice(None, None, -steps_zyx[1] if y_flip else steps_zyx[1])
        x_slice = slice(None, None, -steps_zyx[2] if x_flip else steps_zyx[2])
        if source_tile.axes == "CZYX":
            if channel < 0 or channel >= source_tile.shape[0]:
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {tile.shape[0]}")
            sampled = array[channel, z_slice, y_slice, x_slice]
        elif source_tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
            sampled = array[z_slice, y_slice, x_slice]
        else:
            raise ValueError(f"Unsupported axes {source_tile.axes!r} in {tile.path}")
        return np.asarray(sampled.compute())
    finally:
        if store is not None:
            close = getattr(store, "close", None)
            if callable(close):
                close()


def place_max(canvas: np.ndarray, image: np.ndarray, start: tuple[int, int]) -> None:
    qc_core.place_max(canvas, image, start)


scale_u8 = qc_core.scale_u8
rgb_overlay_image = qc_core.rgb_overlay_image


def compact_tile_label(tile_name: str) -> str:
    stem = tile_name.removesuffix(".ome.tif").removesuffix(".tif")
    match = re.search(r"(C[LR]-[^.]+)\.(\d+)$", stem)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return stem


def draw_overlay_labels(image: Any, labels: list[OverlayLabel], *, y_scale: float = 1.0) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for label in labels:
        y, x = label.yx
        draw.text(
            (int(round(x)) + 3, int(round(y * y_scale)) + 3),
            label.text,
            fill=(255, 255, 255),
            stroke_fill=(0, 0, 0),
            stroke_width=2,
        )


def write_rgb(
    path: Path,
    *,
    left: np.ndarray,
    right: np.ndarray,
    y_scale: float = 1.0,
    labels: list[OverlayLabel] | None = None,
) -> None:
    from PIL import Image

    image = rgb_overlay_image(left=left, right=right)
    if not np.isclose(y_scale, 1.0):
        image = image.resize(
            (image.width, max(1, int(round(image.height * y_scale)))),
            Image.Resampling.BILINEAR,
        )
    if labels:
        draw_overlay_labels(image, labels, y_scale=y_scale)
    image.save(path)


write_contact_sheet = qc_core.write_contact_sheet
empty_projection_canvases = qc_core.empty_projection_canvases


def place_global_projections(
    projections: dict[str, dict[str, np.ndarray]],
    *,
    side: str,
    block: np.ndarray,
    start_zyx: np.ndarray,
) -> None:
    qc_core.place_global_projections(
        projections,
        side=side,
        volume=block,
        start_zyx=start_zyx,
    )


def global_projection_labels(tile_name: str, start_zyx: np.ndarray, block_shape: tuple[int, ...]) -> dict[str, OverlayLabel]:
    label = compact_tile_label(tile_name)
    z_center = float(start_zyx[0]) + float(block_shape[0]) / 2.0
    y_center = float(start_zyx[1]) + float(block_shape[1]) / 2.0
    x_center = float(start_zyx[2]) + float(block_shape[2]) / 2.0
    return {
        "xy": OverlayLabel(label, (y_center, x_center)),
        "xz": OverlayLabel(label, (z_center, x_center)),
        "yz": OverlayLabel(label, (z_center, y_center)),
    }


def plane_label(tile_name: str, start_yx: tuple[int, int], image_shape_yx: tuple[int, int]) -> OverlayLabel:
    return OverlayLabel(
        compact_tile_label(tile_name),
        (
            float(start_yx[0]) + float(image_shape_yx[0]) / 2.0,
            float(start_yx[1]) + float(image_shape_yx[1]) / 2.0,
        ),
    )


def center_y_xz_projection(
    center_canvases: dict[str, np.ndarray],
    *,
    side: str,
    block: np.ndarray,
    start_zyx: np.ndarray,
    center_y_px: int,
    slab_half_px: int,
) -> bool:
    slab_start = center_y_px - slab_half_px
    slab_stop = center_y_px + slab_half_px + 1
    local_start = max(0, slab_start - int(start_zyx[1]))
    local_stop = min(block.shape[1], slab_stop - int(start_zyx[1]))
    if local_stop <= local_start:
        return False
    xz = block[:, local_start:local_stop, :].max(axis=1)
    place_max(center_canvases[side], xz, (int(start_zyx[0]), int(start_zyx[2])))
    return True


def write_global_projection_outputs(
    output_dir: Path,
    *,
    projections: dict[str, dict[str, np.ndarray]],
    labels: dict[str, list[OverlayLabel]],
    level: int,
    channel: int,
    z_display_scale: float,
) -> tuple[list[tuple[str, Path]], Path]:
    outputs = []
    projection_y_scales = {"xy": 1.0, "xz": z_display_scale, "yz": z_display_scale}
    for name in ("xy", "xz", "yz"):
        path = output_dir / f"level{level}_registered_lr_{name}_isoZ_yellowOverlay_ch{channel}.png"
        write_rgb(
            path,
            left=projections["L"][name],
            right=projections["R"][name],
            y_scale=projection_y_scales[name],
            labels=labels[name],
        )
        outputs.append((name.upper(), path))
        print(path, flush=True)
    contact_sheet = output_dir / f"level{level}_registered_lr_isoZ_yellowOverlay_ch{channel}.png"
    write_contact_sheet(contact_sheet, outputs)
    print(contact_sheet, flush=True)
    return outputs, contact_sheet


def write_center_y_outputs(
    output_dir: Path,
    *,
    center_canvases: dict[str, np.ndarray],
    labels: list[OverlayLabel],
    level: int,
    channel: int,
    z_display_scale: float,
) -> dict[str, str]:
    iso_path = output_dir / f"level{level}_registered_lr_xz_centerY_isoZ_yellowOverlay_ch{channel}.png"
    raw_path = output_dir / f"level{level}_registered_lr_xz_centerY_rawZ_yellowOverlay_ch{channel}.png"
    write_rgb(iso_path, left=center_canvases["L"], right=center_canvases["R"], y_scale=z_display_scale, labels=labels)
    write_rgb(raw_path, left=center_canvases["L"], right=center_canvases["R"], y_scale=1.0, labels=labels)
    print(iso_path, flush=True)
    print(raw_path, flush=True)
    return {"center_y_xz_iso_z": str(iso_path.resolve()), "center_y_xz_raw_z": str(raw_path.resolve())}


def output_range_for_bounds(
    *,
    bounds_min_um: np.ndarray,
    bounds_max_um: np.ndarray,
    geometry: RenderGeometry,
    axis: int,
) -> tuple[int, int]:
    start = int(np.floor((bounds_min_um[axis] - geometry.global_min_zyx_um[axis]) / geometry.level_spacing_zyx_um[axis])) - 1
    stop = int(np.ceil((bounds_max_um[axis] - geometry.global_min_zyx_um[axis]) / geometry.level_spacing_zyx_um[axis])) + 1
    return max(0, start), min(int(geometry.shape_zyx[axis]), stop)


def sample_full_affine_plane(
    *,
    block: np.ndarray,
    inverse_affine: np.ndarray,
    geometry: RenderGeometry,
    plane: str,
    fixed_um: float,
    bounds_min_um: np.ndarray,
    bounds_max_um: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], bool]:
    from scipy import ndimage

    if plane == "xy":
        if fixed_um < bounds_min_um[0] or fixed_um > bounds_max_um[0]:
            return np.zeros((0, 0), dtype=np.float32), (0, 0), False
        y0, y1 = output_range_for_bounds(bounds_min_um=bounds_min_um, bounds_max_um=bounds_max_um, geometry=geometry, axis=1)
        x0, x1 = output_range_for_bounds(bounds_min_um=bounds_min_um, bounds_max_um=bounds_max_um, geometry=geometry, axis=2)
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0), dtype=np.float32), (0, 0), False
        y_um = geometry.global_min_zyx_um[1] + np.arange(y0, y1, dtype=np.float64) * geometry.level_spacing_zyx_um[1]
        x_um = geometry.global_min_zyx_um[2] + np.arange(x0, x1, dtype=np.float64) * geometry.level_spacing_zyx_um[2]
        yy, xx = np.meshgrid(y_um, x_um, indexing="ij")
        out_zyx = np.stack([np.full_like(yy, fixed_um), yy, xx], axis=0)
        start = (y0, x0)
    elif plane == "xz":
        if fixed_um < bounds_min_um[1] or fixed_um > bounds_max_um[1]:
            return np.zeros((0, 0), dtype=np.float32), (0, 0), False
        z0, z1 = output_range_for_bounds(bounds_min_um=bounds_min_um, bounds_max_um=bounds_max_um, geometry=geometry, axis=0)
        x0, x1 = output_range_for_bounds(bounds_min_um=bounds_min_um, bounds_max_um=bounds_max_um, geometry=geometry, axis=2)
        if z1 <= z0 or x1 <= x0:
            return np.zeros((0, 0), dtype=np.float32), (0, 0), False
        z_um = geometry.global_min_zyx_um[0] + np.arange(z0, z1, dtype=np.float64) * geometry.level_spacing_zyx_um[0]
        x_um = geometry.global_min_zyx_um[2] + np.arange(x0, x1, dtype=np.float64) * geometry.level_spacing_zyx_um[2]
        zz, xx = np.meshgrid(z_um, x_um, indexing="ij")
        out_zyx = np.stack([zz, np.full_like(zz, fixed_um), xx], axis=0)
        start = (z0, x0)
    else:
        raise ValueError(f"Unsupported plane {plane!r}")

    flat = out_zyx.reshape(3, -1)
    homogeneous = np.vstack([flat, np.ones(flat.shape[1], dtype=np.float64)])
    local_um = (inverse_affine @ homogeneous)[:3]
    sample_coords = local_um / geometry.level_spacing_zyx_um[:, None]
    inside = (
        (sample_coords[0] >= 0.0)
        & (sample_coords[1] >= 0.0)
        & (sample_coords[2] >= 0.0)
        & (sample_coords[0] <= block.shape[0] - 1)
        & (sample_coords[1] <= block.shape[1] - 1)
        & (sample_coords[2] <= block.shape[2] - 1)
    )
    sampled = ndimage.map_coordinates(
        block,
        sample_coords,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(out_zyx.shape[1:]).astype(np.float32, copy=False)
    sampled.reshape(-1)[~inside] = 0.0
    return sampled, start, bool(np.any(inside))


def render_full_affine_planes(
    *,
    tiles: list[Any],
    params: list[Any],
    side_lookup: dict[str, str],
    stitch: Any,
    geometry: RenderGeometry,
    channel: int,
    level: int,
    output_dir: Path,
    center_z_um: float,
    center_y_um: float,
    z_display_scale: float,
    center_z_only: bool = False,
) -> tuple[list[str], Path, list[dict[str, Any]]]:
    xy_canvases = {side: np.zeros((geometry.shape_zyx[1], geometry.shape_zyx[2]), dtype=np.float32) for side in SIDES}
    xz_canvases = (
        None
        if center_z_only
        else {side: np.zeros((geometry.shape_zyx[0], geometry.shape_zyx[2]), dtype=np.float32) for side in SIDES}
    )
    xy_labels: list[OverlayLabel] = []
    xz_labels: list[OverlayLabel] = []
    metadata_rows = []
    for tile, param, (bounds_min_um, bounds_max_um) in zip(tiles, params, geometry.bounds_zyx_um, strict=True):
        side = side_lookup[str(tile.path.resolve())]
        block = sampled_tile(tile, channel=channel, level=level, stitch=stitch).astype(np.float32)
        full_affine = full_affine_from_local_um(tile, param, stitch)
        inverse_affine = np.linalg.inv(full_affine)
        xy, xy_start, in_xy = sample_full_affine_plane(
            block=block,
            inverse_affine=inverse_affine,
            geometry=geometry,
            plane="xy",
            fixed_um=center_z_um,
            bounds_min_um=bounds_min_um,
            bounds_max_um=bounds_max_um,
        )
        if in_xy:
            place_max(xy_canvases[side], xy, xy_start)
            xy_labels.append(plane_label(tile.path.name, xy_start, xy.shape))
        in_xz = False
        if xz_canvases is not None:
            xz, xz_start, in_xz = sample_full_affine_plane(
                block=block,
                inverse_affine=inverse_affine,
                geometry=geometry,
                plane="xz",
                fixed_um=center_y_um,
                bounds_min_um=bounds_min_um,
                bounds_max_um=bounds_max_um,
            )
            if in_xz:
                place_max(xz_canvases[side], xz, xz_start)
                xz_labels.append(plane_label(tile.path.name, xz_start, xz.shape))
        metadata_rows.append(
            {
                "tile": tile.path.name,
                "label": compact_tile_label(tile.path.name),
                "side": side,
                "registered_bounds_min_zyx_um": bounds_min_um.tolist(),
                "registered_bounds_max_zyx_um": bounds_max_um.tolist(),
                "sampled_shape_zyx": [int(value) for value in block.shape],
                "in_center_z_xy": in_xy,
                "in_center_y_xz": in_xz,
            }
        )
        print(
            f"full-affine sampled {tile.path.name} side={side} "
            f"bounds_min={bounds_min_um.tolist()} bounds_max={bounds_max_um.tolist()} "
            f"shape={list(block.shape)}",
            flush=True,
        )

    xy_path = output_dir / f"level{level}_registered_lr_fullAffine_centerZ_xy_yellowOverlay_ch{channel}.png"
    write_rgb(xy_path, left=xy_canvases["L"], right=xy_canvases["R"], labels=xy_labels)
    if center_z_only:
        print(xy_path, flush=True)
        return [str(xy_path.resolve())], xy_path, metadata_rows

    if xz_canvases is None:
        raise RuntimeError("XZ canvases were unexpectedly missing")
    xz_iso_path = output_dir / f"level{level}_registered_lr_fullAffine_centerY_xz_isoZ_yellowOverlay_ch{channel}.png"
    xz_raw_path = output_dir / f"level{level}_registered_lr_fullAffine_centerY_xz_rawZ_yellowOverlay_ch{channel}.png"
    write_rgb(xz_iso_path, left=xz_canvases["L"], right=xz_canvases["R"], y_scale=z_display_scale, labels=xz_labels)
    write_rgb(xz_raw_path, left=xz_canvases["L"], right=xz_canvases["R"], y_scale=1.0, labels=xz_labels)
    contact_sheet = output_dir / f"level{level}_registered_lr_fullAffine_planes_yellowOverlay_ch{channel}.png"
    write_contact_sheet(contact_sheet, [("XY center Z", xy_path), ("XZ center Y iso Z", xz_iso_path), ("XZ center Y raw Z", xz_raw_path)])
    for path in (xy_path, xz_iso_path, xz_raw_path, contact_sheet):
        print(path, flush=True)
    return [str(xy_path.resolve()), str(xz_iso_path.resolve()), str(xz_raw_path.resolve())], contact_sheet, metadata_rows


def main(
    *,
    default_position_input: Path | None = None,
    default_registration_input: Path | None = None,
    default_output_dir: Path | None = None,
) -> int:
    args = parse_args(
        default_position_input=default_position_input,
        default_registration_input=default_registration_input,
        default_output_dir=default_output_dir,
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    if args.skip_global_projections and not args.center_y_xz and not (args.full_affine_planes and args.center_z_only):
        raise ValueError("Nothing to render: remove --skip-global-projections or pass --center-y-xz")
    if args.slab_half_px < 0:
        raise ValueError("--slab-half-px must be non-negative")

    stitch = load_stitch_module()
    position_payload = json.loads(args.position_input.read_text())
    side_lookup = side_by_tile(position_payload)
    tiles = stitch.read_position_input_tiles(args.position_input.resolve())
    params = stitch.load_registration_params(args.registration_input.resolve(), tiles)
    geometry = (
        build_full_affine_geometry(tiles, params, level=args.level, stitch=stitch)
        if args.full_affine_planes
        else build_geometry(tiles, params, level=args.level, stitch=stitch)
    )
    z_display_scale = float(geometry.level_spacing_zyx_um[0] / geometry.level_spacing_zyx_um[1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    center_y_um = (
        float(args.center_y_um)
        if args.center_y_um is not None
        else float((geometry.global_min_zyx_um[1] + geometry.global_max_zyx_um[1]) / 2.0)
    )
    center_y_px = int(round((center_y_um - geometry.global_min_zyx_um[1]) / geometry.level_spacing_zyx_um[1]))
    center_z_um = (
        float(args.center_z_um)
        if args.center_z_um is not None
        else float((geometry.global_min_zyx_um[0] + geometry.global_max_zyx_um[0]) / 2.0)
    )
    center_z_px = int(round((center_z_um - geometry.global_min_zyx_um[0]) / geometry.level_spacing_zyx_um[0]))

    if args.full_affine_planes:
        outputs, contact_sheet, metadata_rows = render_full_affine_planes(
            tiles=tiles,
            params=params,
            side_lookup=side_lookup,
            stitch=stitch,
            geometry=geometry,
            channel=args.channel,
            level=args.level,
            output_dir=args.output_dir,
            center_z_um=center_z_um,
            center_y_um=center_y_um,
            z_display_scale=z_display_scale,
            center_z_only=bool(args.center_z_only),
        )
        summary = {
            "position_input": str(args.position_input.resolve()),
            "registration_input": str(args.registration_input.resolve()),
            "channel": args.channel,
            "level": args.level,
            "level_factor": geometry.level_factor,
            "mode": "full_affine_center_z" if args.center_z_only else "full_affine_planes",
            "global_min_zyx_um": geometry.global_min_zyx_um.tolist(),
            "global_max_zyx_um": geometry.global_max_zyx_um.tolist(),
            "level_spacing_zyx_um": geometry.level_spacing_zyx_um.tolist(),
            "level_shape_zyx": [int(value) for value in geometry.shape_zyx],
            "z_display_scale_for_xz_yz": z_display_scale,
            "center_z_um": center_z_um,
            "center_z_px": center_z_px,
            "center_y_um": center_y_um,
            "center_y_px": center_y_px,
            "color_mapping": {"L": "green", "R": "red", "overlap": "yellow"},
            "tiles": metadata_rows,
            "outputs": outputs,
            "contact_sheet": str(contact_sheet.resolve()),
        }
        summary_path = args.output_dir / f"level{args.level}_registered_lr_fullAffine_planes_yellowOverlay_ch{args.channel}.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(summary_path, flush=True)
        return 0

    projections = empty_projection_canvases(geometry.shape_zyx) if not args.skip_global_projections else None
    projection_labels = {name: [] for name in ("xy", "xz", "yz")}
    center_canvases = (
        {side: np.zeros((geometry.shape_zyx[0], geometry.shape_zyx[2]), dtype=np.float32) for side in SIDES}
        if args.center_y_xz
        else None
    )
    center_y_labels: list[OverlayLabel] = []

    metadata_rows = []
    for tile, (start_um, _stop_um) in zip(tiles, geometry.bounds_zyx_um, strict=True):
        side = side_lookup[str(tile.path.resolve())]
        block = sampled_tile(tile, channel=args.channel, level=args.level, stitch=stitch).astype(np.float32)
        start_zyx = np.rint((start_um - geometry.global_min_zyx_um) / geometry.level_spacing_zyx_um).astype(int)
        if projections is not None:
            place_global_projections(projections, side=side, block=block, start_zyx=start_zyx)
            labels = global_projection_labels(tile.path.name, start_zyx, block.shape)
            for name, label in labels.items():
                projection_labels[name].append(label)
        in_center_y_slab = False
        if center_canvases is not None:
            in_center_y_slab = center_y_xz_projection(
                center_canvases,
                side=side,
                block=block,
                start_zyx=start_zyx,
                center_y_px=center_y_px,
                slab_half_px=int(args.slab_half_px),
            )
            if in_center_y_slab:
                center_y_labels.append(
                    OverlayLabel(
                        compact_tile_label(tile.path.name),
                        (float(start_zyx[0]) + float(block.shape[0]) / 2.0, float(start_zyx[2]) + float(block.shape[2]) / 2.0),
                    )
                )
        metadata_rows.append(
            {
                "tile": tile.path.name,
                "label": compact_tile_label(tile.path.name),
                "side": side,
                "level_start_zyx": [int(value) for value in start_zyx],
                "sampled_shape_zyx": [int(value) for value in block.shape],
                "in_center_y_slab": in_center_y_slab,
            }
        )
        print(f"placed {tile.path.name} side={side} start_zyx={start_zyx.tolist()} shape={list(block.shape)}", flush=True)

    outputs: list[str] = []
    contact_sheet = None
    if projections is not None:
        projection_outputs, contact_sheet = write_global_projection_outputs(
            args.output_dir,
            projections=projections,
            labels=projection_labels,
            level=args.level,
            channel=args.channel,
            z_display_scale=z_display_scale,
        )
        outputs.extend(str(path.resolve()) for _, path in projection_outputs)
    if center_canvases is not None:
        outputs.extend(
            write_center_y_outputs(
                args.output_dir,
                center_canvases=center_canvases,
                labels=center_y_labels,
                level=args.level,
                channel=args.channel,
                z_display_scale=z_display_scale,
            ).values()
        )

    summary = {
        "position_input": str(args.position_input.resolve()),
        "registration_input": str(args.registration_input.resolve()),
        "channel": args.channel,
        "level": args.level,
        "level_factor": geometry.level_factor,
        "global_min_zyx_um": geometry.global_min_zyx_um.tolist(),
        "global_max_zyx_um": geometry.global_max_zyx_um.tolist(),
        "level_spacing_zyx_um": geometry.level_spacing_zyx_um.tolist(),
        "level_shape_zyx": [int(value) for value in geometry.shape_zyx],
        "z_display_scale_for_xz_yz": z_display_scale,
        "center_z_um": center_z_um if args.full_affine_planes else None,
        "center_z_px": center_z_px if args.full_affine_planes else None,
        "center_y_um": center_y_um if args.center_y_xz else None,
        "center_y_px": center_y_px if args.center_y_xz else None,
        "slab_half_px": int(args.slab_half_px) if args.center_y_xz else None,
        "color_mapping": {"L": "green", "R": "red", "overlap": "yellow"},
        "tiles": metadata_rows,
        "outputs": outputs,
        "contact_sheet": None if contact_sheet is None else str(contact_sheet.resolve()),
    }
    summary_path = args.output_dir / f"level{args.level}_registered_lr_isoZ_yellowOverlay_ch{args.channel}.json"
    if args.center_y_xz and args.skip_global_projections:
        summary_path = args.output_dir / f"level{args.level}_registered_lr_xz_centerY_isoZ_yellowOverlay_ch{args.channel}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
