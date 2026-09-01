#!/usr/bin/env python
"""Render center-Z BaSiC dumb-stitch QC and overlap correlations.

The same corrected center-Z tile planes are used for both outputs:
1. no-blend dumb-stitch PNGs
2. registered-overlap Pearson correlations on sampled tile edges
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from benchmark_basicpy_overlap_correlation import (
    RAW_CASE,
    ImageReaderCache,
    Tile,
    apply_basic_to_crop,
    build_basic_cases,
    candidate_pairs,
    finite_float,
    load_tiles,
    parse_case,
    parse_case_source_view,
    parse_source_view_root,
    pearson,
    read_json,
    select_basic_profile,
    spatial_shape_zyx,
    summarize,
)


@dataclass(frozen=True)
class PlaneRecord:
    tile: Tile
    plane: np.ndarray
    y0: int
    x0: int


def parse_channels(value: str) -> list[int]:
    channels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not channels or any(channel < 0 for channel in channels):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative channel indexes")
    return channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channels", type=parse_channels, default=[0, 1])
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--center-z-index", type=int, help="Source-level Z index. Defaults to the center Z plane.")
    parser.add_argument("--image-path-key", choices=("path", "source_path"), default="path")
    parser.add_argument(
        "--image-source-view-root",
        type=parse_source_view_root,
        action="append",
        default=[],
        metavar="VIEW=DIR",
        help="Map tile source_view to a raw image directory. Tile names ending in .ome.zarr map to .ome.tif.",
    )
    parser.add_argument("--case", type=parse_case, action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--case-source-view", type=parse_case_source_view, action="append", default=[])
    parser.add_argument("--edge-sample-count", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--min-overlap-px-yx", default="32,32")
    parser.add_argument("--min-valid-pixels", type=int, default=4096)
    parser.add_argument("--min-positive-fraction", type=float, default=1e-4)
    parser.add_argument("--draw-tile-labels", action="store_true")
    args = parser.parse_args()
    args.flatfield_dir = None
    args.flatfield_dir_by_source_view = []
    args.min_overlap_px_yx = parse_yx(args.min_overlap_px_yx)
    if args.level < 0:
        raise ValueError("--level must be non-negative")
    if args.center_z_index is not None and args.center_z_index < 0:
        raise ValueError("--center-z-index must be non-negative")
    if args.edge_sample_count <= 0:
        raise ValueError("--edge-sample-count must be positive")
    if args.min_valid_pixels <= 1:
        raise ValueError("--min-valid-pixels must be greater than 1")
    if not 0 <= args.min_positive_fraction <= 1:
        raise ValueError("--min-positive-fraction must be in [0, 1]")
    return args


def parse_yx(value: str) -> tuple[int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 2 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("expected positive Y,X shape, for example 32,32")
    return parts


def tile_number(name: str) -> str:
    stem = name.removesuffix(".ome.zarr").removesuffix(".ome.tif")
    return stem.rsplit(".", 1)[-1]


def stretch_uint8(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = np.concatenate(
        [
            image[np.isfinite(image) & (image > 0)].ravel()
            for image in images.values()
            if np.any(np.isfinite(image) & (image > 0))
        ]
    )
    if values.size == 0:
        values = np.concatenate([image[np.isfinite(image)].ravel() for image in images.values()])
    low, high = np.percentile(values, [0.5, 99.8]) if values.size else (0.0, 1.0)
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return {
        name: np.clip((image - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)
        for name, image in images.items()
    }


def contact_sheet(images: list[tuple[str, np.ndarray]], *, columns: int) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet")
    thumbs: list[tuple[str, Image.Image]] = []
    max_width = 900
    label_height = 24
    for label, image in images:
        pil = Image.fromarray(image)
        scale = min(1.0, max_width / max(1, pil.width))
        if scale < 1.0:
            pil = pil.resize((int(round(pil.width * scale)), int(round(pil.height * scale))), Image.Resampling.BILINEAR)
        thumbs.append((label, pil))
    cell_width = max(pil.width for _, pil in thumbs)
    cell_height = max(pil.height for _, pil in thumbs) + label_height
    rows = int(math.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, pil) in enumerate(thumbs):
        row, col = divmod(index, columns)
        x = col * cell_width
        y = row * cell_height
        draw.text((x + 6, y + 4), label, fill=(0, 0, 0))
        sheet.paste(pil.convert("RGB"), (x, y + label_height))
    return sheet


def compute_canvas(
    tiles: list[Tile],
    *,
    level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    pixel_um_yx: np.ndarray | None = None
    bounds_min = np.full(2, np.inf, dtype=np.float64)
    bounds_max = np.full(2, -np.inf, dtype=np.float64)
    for tile in tiles:
        current_pixel_um_yx = tile.spacing_um_zyx[1:] * (2**level)
        if pixel_um_yx is None:
            pixel_um_yx = current_pixel_um_yx
        elif not np.allclose(pixel_um_yx, current_pixel_um_yx, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Inconsistent level {level} pixel spacing: {pixel_um_yx} versus {current_pixel_um_yx}")
        bounds_min = np.minimum(bounds_min, tile.bbox_min_um_zyx[1:])
        bounds_max = np.maximum(bounds_max, tile.bbox_max_um_zyx[1:])
    if pixel_um_yx is None:
        raise ValueError("No tiles loaded")
    shape_yx = tuple(np.ceil((bounds_max - bounds_min) / pixel_um_yx).astype(int).tolist())
    return pixel_um_yx, bounds_min, bounds_max, shape_yx


def load_case_planes(
    *,
    tiles: list[Tile],
    channel: int,
    level: int,
    center_z_index: int | None,
    basic_cases: dict[str, dict[str, Any]],
    reader_cache: ImageReaderCache,
    bounds_min_yx_um: np.ndarray,
    pixel_um_yx: np.ndarray,
) -> tuple[dict[str, list[PlaneRecord]], int]:
    planes_by_case: dict[str, list[PlaneRecord]] = {RAW_CASE: []}
    planes_by_case.update({name: [] for name in sorted(basic_cases)})
    resolved_center_z: int | None = None

    for tile in tiles:
        axes, source_level, source_shape = reader_cache.metadata(tile.path, level=level)
        source_shape_zyx = spatial_shape_zyx(axes=axes, shape=source_shape)
        z_index = int(source_shape_zyx[0] // 2 if center_z_index is None else center_z_index)
        if z_index >= int(source_shape_zyx[0]):
            raise ValueError(f"Center Z {z_index} is outside {tile.path} level {source_level} shape {tuple(source_shape_zyx)}")
        if resolved_center_z is None:
            resolved_center_z = z_index
        elif resolved_center_z != z_index:
            raise ValueError(f"Inconsistent center Z: {resolved_center_z} versus {z_index}")

        slices = (slice(z_index, z_index + 1), slice(0, int(source_shape_zyx[1])), slice(0, int(source_shape_zyx[2])))
        raw_stack = reader_cache.read_crop(tile.path, level=source_level, channel=channel, slices_zyx=slices)
        y0 = int(round((tile.bbox_min_um_zyx[1] - bounds_min_yx_um[0]) / pixel_um_yx[0]))
        x0 = int(round((tile.bbox_min_um_zyx[2] - bounds_min_yx_um[1]) / pixel_um_yx[1]))
        planes_by_case[RAW_CASE].append(PlaneRecord(tile=tile, plane=raw_stack[0], y0=y0, x0=x0))
        for case_name, profiles in sorted(basic_cases.items()):
            profile = select_basic_profile(profiles, tile)
            corrected = apply_basic_to_crop(
                raw_stack,
                profile=profile,
                crop_slices_zyx=slices,
                level_shape_yx=(int(source_shape_zyx[1]), int(source_shape_zyx[2])),
            )
            planes_by_case[case_name].append(PlaneRecord(tile=tile, plane=corrected[0], y0=y0, x0=x0))

    if resolved_center_z is None:
        raise ValueError("No planes loaded")
    return planes_by_case, resolved_center_z


def render_mosaics(
    planes_by_case: dict[str, list[PlaneRecord]],
    *,
    shape_yx: tuple[int, int],
) -> dict[str, np.ndarray]:
    mosaics = {}
    for case_name, planes in planes_by_case.items():
        mosaic = np.zeros(shape_yx, dtype=np.float32)
        for record in planes:
            y1 = min(shape_yx[0], record.y0 + record.plane.shape[0])
            x1 = min(shape_yx[1], record.x0 + record.plane.shape[1])
            if y1 > record.y0 and x1 > record.x0:
                mosaic[record.y0:y1, record.x0:x1] = record.plane[: y1 - record.y0, : x1 - record.x0]
        mosaics[case_name] = mosaic
    return mosaics


def annotate_tiles(image: np.ndarray, planes: list[PlaneRecord]) -> np.ndarray:
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for record in planes:
        label = tile_number(record.tile.name)
        draw.text((record.x0 + 4, record.y0 + 4), label, fill=(255, 255, 255))
        draw.rectangle(
            [record.x0, record.y0, record.x0 + record.plane.shape[1] - 1, record.y0 + record.plane.shape[0] - 1],
            outline=(255, 255, 255),
            width=1,
        )
    return np.asarray(pil)


def correlate_case(
    planes: list[PlaneRecord],
    sampled_pairs: list[tuple[Tile, Tile, np.ndarray, np.ndarray, np.ndarray]],
    *,
    min_valid_pixels: int,
    min_positive_fraction: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    by_index = {record.tile.index: record for record in planes}
    correlations: list[float] = []
    rows = []
    for pair_index, (tile_a, tile_b, _start, _stop, overlap_px) in enumerate(sampled_pairs):
        record_a = by_index[tile_a.index]
        record_b = by_index[tile_b.index]
        y0 = max(record_a.y0, record_b.y0)
        x0 = max(record_a.x0, record_b.x0)
        y1 = min(record_a.y0 + record_a.plane.shape[0], record_b.y0 + record_b.plane.shape[0])
        x1 = min(record_a.x0 + record_a.plane.shape[1], record_b.x0 + record_b.plane.shape[1])
        if y1 <= y0 or x1 <= x0:
            corr = float("nan")
            valid_pixels = 0
            positive_fraction = 0.0
        else:
            crop_a = record_a.plane[y0 - record_a.y0 : y1 - record_a.y0, x0 - record_a.x0 : x1 - record_a.x0]
            crop_b = record_b.plane[y0 - record_b.y0 : y1 - record_b.y0, x0 - record_b.x0 : x1 - record_b.x0]
            mask = np.isfinite(crop_a) & np.isfinite(crop_b)
            valid_pixels = int(np.count_nonzero(mask))
            positive_fraction = float(
                np.count_nonzero((crop_a > 0) & (crop_b > 0) & mask) / max(1, valid_pixels)
            )
            corr = pearson(crop_a, crop_b, mask)
        accepted = (
            valid_pixels >= min_valid_pixels
            and positive_fraction >= min_positive_fraction
            and math.isfinite(corr)
        )
        if accepted:
            correlations.append(corr)
        rows.append(
            {
                "pair_index": pair_index,
                "tile_a": tile_a.name,
                "tile_b": tile_b.name,
                "source_view_a": tile_a.source_view,
                "source_view_b": tile_b.source_view,
                "overlap_shape_px_zyx": overlap_px.astype(int).tolist(),
                "canvas_overlap_yx_px": [int(y1 - y0), int(x1 - x0)] if y1 > y0 and x1 > x0 else [0, 0],
                "pearson": finite_float(corr),
                "accepted": accepted,
                "valid_pixels": valid_pixels,
                "positive_fraction": positive_fraction,
            }
        )
    return correlations, rows


def run_channel(
    args: argparse.Namespace,
    *,
    tiles: list[Tile],
    sampled_pairs: list[tuple[Tile, Tile, np.ndarray, np.ndarray, np.ndarray]],
    channel: int,
    pixel_um_yx: np.ndarray,
    bounds_min_yx_um: np.ndarray,
    shape_yx: tuple[int, int],
) -> dict[str, Any]:
    basic_cases = build_basic_cases(args, channel)
    reader_cache = ImageReaderCache()
    try:
        planes_by_case, center_z = load_case_planes(
            tiles=tiles,
            channel=channel,
            level=args.level,
            center_z_index=args.center_z_index,
            basic_cases=basic_cases,
            reader_cache=reader_cache,
            bounds_min_yx_um=bounds_min_yx_um,
            pixel_um_yx=pixel_um_yx,
        )
    finally:
        reader_cache.close()

    mosaics = render_mosaics(planes_by_case, shape_yx=shape_yx)
    stretched = stretch_uint8(mosaics)
    output_paths = {}
    contact_inputs = []
    for case_name in [RAW_CASE, *sorted(basic_cases)]:
        image = stretched[case_name]
        if args.draw_tile_labels:
            image = annotate_tiles(image, planes_by_case[case_name])
        path = args.output_dir / f"centerZ{center_z}_ch{channel}_{case_name}_level{args.level}_dumb_stitch.png"
        Image.fromarray(image).save(path)
        output_paths[case_name] = str(path)
        contact_inputs.append((f"ch{channel} {case_name}", image))

    correlations = {}
    patch_rows = {}
    for case_name in [RAW_CASE, *sorted(basic_cases)]:
        values, rows = correlate_case(
            planes_by_case[case_name],
            sampled_pairs,
            min_valid_pixels=args.min_valid_pixels,
            min_positive_fraction=args.min_positive_fraction,
        )
        correlations[case_name] = values
        patch_rows[case_name] = rows

    return {
        "channel": channel,
        "center_z_index": center_z,
        "outputs": output_paths,
        "case_summaries": {case_name: summarize(values) for case_name, values in correlations.items()},
        "edges": patch_rows,
        "contact_inputs": contact_inputs,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registration_payload = read_json(args.registration_json)
    source_view_roots = {view: path for view, path in args.image_source_view_root}
    tiles = load_tiles(
        registration_payload,
        image_path_key=args.image_path_key,
        source_view_roots=source_view_roots,
    )
    pixel_um_yx, bounds_min_yx_um, bounds_max_yx_um, shape_yx = compute_canvas(tiles, level=args.level)
    rng = np.random.default_rng(args.sample_seed)
    sampled_pairs = candidate_pairs(
        tiles,
        min_overlap_px_zyx=(1, *args.min_overlap_px_yx),
        max_pairs=None,
        edge_sample_count=args.edge_sample_count,
        rng=rng,
    )
    if not sampled_pairs:
        raise ValueError("No overlapping tile pairs passed the overlap filters")

    channels = []
    contact_inputs = []
    for channel in args.channels:
        result = run_channel(
            args,
            tiles=tiles,
            sampled_pairs=sampled_pairs,
            channel=channel,
            pixel_um_yx=pixel_um_yx,
            bounds_min_yx_um=bounds_min_yx_um,
            shape_yx=shape_yx,
        )
        contact_inputs.extend(result.pop("contact_inputs"))
        channels.append(result)

    contact_path = args.output_dir / f"centerZ_level{args.level}_basic_cases_contact_sheet.png"
    contact_sheet(contact_inputs, columns=3).save(contact_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": "basicpy_center_z_dumb_stitch_overlap_qc.v1",
        "created_at_unix": time.time(),
        "inputs": {
            "registration_json": str(args.registration_json),
            "channels": args.channels,
            "level": args.level,
            "center_z_index": args.center_z_index,
            "resolved_center_z_index": channels[0]["center_z_index"] if channels else None,
            "edge_sample_count": args.edge_sample_count,
            "sample_seed": args.sample_seed,
            "image_path_key": args.image_path_key,
            "image_source_view_roots": {view: str(path) for view, path in source_view_roots.items()},
            "min_overlap_px_yx": list(args.min_overlap_px_yx),
            "min_valid_pixels": args.min_valid_pixels,
            "min_positive_fraction": args.min_positive_fraction,
        },
        "tile_count": len(tiles),
        "sampled_edge_count": len(sampled_pairs),
        "pixel_um_yx": pixel_um_yx.astype(float).tolist(),
        "bounds_yx_um": {"min": bounds_min_yx_um.astype(float).tolist(), "max": bounds_max_yx_um.astype(float).tolist()},
        "shape_yx_px": list(shape_yx),
        "contact_sheet": str(contact_path),
        "channels": channels,
    }
    manifest_path = args.output_dir / "center_z_basic_dumb_stitch_overlap_qc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "contact_sheet": str(contact_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
