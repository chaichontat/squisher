#!/usr/bin/env python
"""Render level-N L/R registration QC overlays colored by acquisition side."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


DIMENSIONS = ("z", "y", "x")
SIDES = ("L", "R")


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
    parser.add_argument("--slab-half-px", type=int, default=4, help="Half-width of the center-y XZ slab at the sampled level.")
    parser.add_argument("--skip-global-projections", action="store_true")
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


def build_geometry(tiles: list[Any], params: list[Any], *, level: int, stitch: Any) -> RenderGeometry:
    if level < 0:
        raise ValueError("--level must be non-negative")
    level_factor = 2**level
    spacing = np.asarray([tiles[0].spacing[dim] for dim in DIMENSIONS], dtype=np.float64)
    level_spacing = spacing * level_factor
    bounds = [registered_bounds_um(tile, param, stitch) for tile, param in zip(tiles, params, strict=True)]
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


def tiff_series_level_count(path: Path) -> int:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        return len(tif.series[0].levels)


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
    import tifffile
    import zarr

    source_level = min(int(level), max(0, tiff_series_level_count(tile.path) - 1))
    store = tifffile.imread(tile.path, aszarr=True, level=source_level)
    try:
        zarray = zarr.open(store, mode="r")
        steps_zyx = remaining_sample_steps(
            tile,
            source_shape=tuple(int(value) for value in zarray.shape),
            target_level_factor=2**int(level),
        )
        array = da.from_zarr(zarray)
        z_flip, y_flip, x_flip = stitch.tile_flip_axes_zyx(tile)
        z_slice = slice(None, None, -steps_zyx[0] if z_flip else steps_zyx[0])
        y_slice = slice(None, None, -steps_zyx[1] if y_flip else steps_zyx[1])
        x_slice = slice(None, None, -steps_zyx[2] if x_flip else steps_zyx[2])
        if tile.axes == "CZYX":
            if channel < 0 or channel >= tile.shape[0]:
                raise ValueError(f"Channel {channel} is outside {tile.path} channel count {tile.shape[0]}")
            sampled = array[channel, z_slice, y_slice, x_slice]
        elif tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
            sampled = array[z_slice, y_slice, x_slice]
        else:
            raise ValueError(f"Unsupported axes {tile.axes!r} in {tile.path}")
        return np.asarray(sampled.compute())
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def place_max(canvas: np.ndarray, image: np.ndarray, start: tuple[int, int]) -> None:
    y0, x0 = start
    if y0 >= canvas.shape[0] or x0 >= canvas.shape[1]:
        return
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    y_size = min(image.shape[0] - src_y0, canvas.shape[0] - dst_y0)
    x_size = min(image.shape[1] - src_x0, canvas.shape[1] - dst_x0)
    if y_size <= 0 or x_size <= 0:
        return
    canvas[dst_y0 : dst_y0 + y_size, dst_x0 : dst_x0 + x_size] = np.maximum(
        canvas[dst_y0 : dst_y0 + y_size, dst_x0 : dst_x0 + x_size],
        image[src_y0 : src_y0 + y_size, src_x0 : src_x0 + x_size],
    )


def scale_u8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(positive, [1.0, 99.8])
    scaled = np.clip((image - low) / max(float(high - low), 1.0), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def rgb_overlay_image(*, left: np.ndarray, right: np.ndarray):
    from PIL import Image

    rgb = np.zeros((*left.shape, 3), dtype=np.uint8)
    rgb[..., 0] = scale_u8(right)
    rgb[..., 1] = scale_u8(left)
    return Image.fromarray(rgb)


def write_rgb(path: Path, *, left: np.ndarray, right: np.ndarray, y_scale: float = 1.0) -> None:
    from PIL import Image

    image = rgb_overlay_image(left=left, right=right)
    if not np.isclose(y_scale, 1.0):
        image = image.resize(
            (image.width, max(1, int(round(image.height * y_scale)))),
            Image.Resampling.BILINEAR,
        )
    image.save(path)


def write_contact_sheet(output: Path, images: list[tuple[str, Path]]) -> None:
    from PIL import Image, ImageDraw

    opened = [(title, Image.open(path).convert("RGB")) for title, path in images]
    target_width = max(image.width for _, image in opened)
    resized = []
    for title, image in opened:
        if image.width != target_width:
            height = max(1, int(round(image.height * target_width / image.width)))
            image = image.resize((target_width, height), Image.Resampling.LANCZOS)
        resized.append((title, image))
    title_height = 28
    sheet = Image.new("RGB", (target_width, sum(image.height + title_height for _, image in resized)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for title, image in resized:
        draw.text((8, y + 7), title, fill=(0, 0, 0))
        y += title_height
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output)


def empty_projection_canvases(shape_zyx: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return {
        side: {
            "xy": np.zeros((shape_zyx[1], shape_zyx[2]), dtype=np.float32),
            "xz": np.zeros((shape_zyx[0], shape_zyx[2]), dtype=np.float32),
            "yz": np.zeros((shape_zyx[0], shape_zyx[1]), dtype=np.float32),
        }
        for side in SIDES
    }


def place_global_projections(
    projections: dict[str, dict[str, np.ndarray]],
    *,
    side: str,
    block: np.ndarray,
    start_zyx: np.ndarray,
) -> None:
    place_max(projections[side]["xy"], block.max(axis=0), (int(start_zyx[1]), int(start_zyx[2])))
    place_max(projections[side]["xz"], block.max(axis=1), (int(start_zyx[0]), int(start_zyx[2])))
    place_max(projections[side]["yz"], block.max(axis=2), (int(start_zyx[0]), int(start_zyx[1])))


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
    level: int,
    channel: int,
    z_display_scale: float,
) -> dict[str, str]:
    iso_path = output_dir / f"level{level}_registered_lr_xz_centerY_isoZ_yellowOverlay_ch{channel}.png"
    raw_path = output_dir / f"level{level}_registered_lr_xz_centerY_rawZ_yellowOverlay_ch{channel}.png"
    write_rgb(iso_path, left=center_canvases["L"], right=center_canvases["R"], y_scale=z_display_scale)
    write_rgb(raw_path, left=center_canvases["L"], right=center_canvases["R"], y_scale=1.0)
    print(iso_path, flush=True)
    print(raw_path, flush=True)
    return {"center_y_xz_iso_z": str(iso_path.resolve()), "center_y_xz_raw_z": str(raw_path.resolve())}


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
    if args.skip_global_projections and not args.center_y_xz:
        raise ValueError("Nothing to render: remove --skip-global-projections or pass --center-y-xz")
    if args.slab_half_px < 0:
        raise ValueError("--slab-half-px must be non-negative")

    stitch = load_stitch_module()
    position_payload = json.loads(args.position_input.read_text())
    side_lookup = side_by_tile(position_payload)
    tiles = stitch.read_position_input_tiles(args.position_input.resolve())
    params = stitch.load_registration_params(args.registration_input.resolve(), tiles)
    geometry = build_geometry(tiles, params, level=args.level, stitch=stitch)
    z_display_scale = float(geometry.level_spacing_zyx_um[0] / geometry.level_spacing_zyx_um[1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    projections = empty_projection_canvases(geometry.shape_zyx) if not args.skip_global_projections else None
    center_y_um = (
        float(args.center_y_um)
        if args.center_y_um is not None
        else float((geometry.global_min_zyx_um[1] + geometry.global_max_zyx_um[1]) / 2.0)
    )
    center_y_px = int(round((center_y_um - geometry.global_min_zyx_um[1]) / geometry.level_spacing_zyx_um[1]))
    center_canvases = (
        {side: np.zeros((geometry.shape_zyx[0], geometry.shape_zyx[2]), dtype=np.float32) for side in SIDES}
        if args.center_y_xz
        else None
    )

    metadata_rows = []
    for tile, (start_um, _stop_um) in zip(tiles, geometry.bounds_zyx_um, strict=True):
        side = side_lookup[str(tile.path.resolve())]
        block = sampled_tile(tile, channel=args.channel, level=args.level, stitch=stitch).astype(np.float32)
        start_zyx = np.rint((start_um - geometry.global_min_zyx_um) / geometry.level_spacing_zyx_um).astype(int)
        if projections is not None:
            place_global_projections(projections, side=side, block=block, start_zyx=start_zyx)
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
        metadata_rows.append(
            {
                "tile": tile.path.name,
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
