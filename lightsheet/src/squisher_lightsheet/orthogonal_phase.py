from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import ndimage
import tifffile
import zarr

from squisher.jpegxr_zarr import register_jpegxr_codec
from squisher_lightsheet import ngff, phase_metrics, qc
from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as legacy
from squisher_lightsheet.channel_mattes_anchors import phasecorr_shift_gpu


IntensityTransform = Literal["identity", "log1p"]
DEFAULT_LATERAL_FACTOR = 4


@dataclass(frozen=True)
class OrthogonalPhaseResult:
    z_residual_um: float
    summary: Path
    contact_sheet: Path


def position_tiles(
    payload: dict[str, Any],
    *,
    tile_dir: Path | None,
) -> list[legacy.TileRecord]:
    rendered = deepcopy(payload)
    for record in rendered.get("tiles", []):
        if record.get("side") is None:
            record["side"] = "L"
        if tile_dir is None:
            continue
        name = Path(str(record["tile"])).name
        for suffix in (".ome.tif", ".ome.tiff"):
            if name.endswith(suffix):
                name = f"{name[: -len(suffix)]}.ome.zarr"
                break
        tile_path = tile_dir / name
        if not tile_path.exists():
            raise FileNotFoundError(
                f"Tile directory override could not resolve {record['tile']!r} at {tile_path}"
            )
        record["path"] = str(tile_path)
    return legacy.load_tiles(rendered)


def apply_intensity_transform(image: np.ndarray, transform: IntensityTransform) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if transform == "identity":
        return values
    if transform == "log1p":
        if np.any(values < 0):
            raise ValueError("log1p intensity transform requires nonnegative values")
        return np.log1p(values).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported intensity transform {transform!r}")


def _bounds(tile: legacy.TileRecord) -> tuple[np.ndarray, np.ndarray]:
    return legacy.tile_bounds_zyx_um(tile)


def _center(tile: legacy.TileRecord) -> np.ndarray:
    start, stop = _bounds(tile)
    return (start + stop) / 2.0


def select_center_neighborhood(
    tiles: list[legacy.TileRecord],
) -> tuple[list[legacy.TileRecord], legacy.TileRecord]:
    if not tiles:
        raise ValueError("Orthogonal phase requires fixed tiles")
    tile_bounds = [_bounds(tile) for tile in tiles]
    mosaic_center = (
        np.min([start for start, _stop in tile_bounds], axis=0)
        + np.max([stop for _start, stop in tile_bounds], axis=0)
    ) / 2.0
    center_tile = min(
        tiles,
        key=lambda tile: np.linalg.norm((_center(tile) - mosaic_center)[1:]),
    )
    tile_size_yx = np.abs(center_tile.scale_zyx_um[1:]) * center_tile.shape_zyx[1:]
    neighborhood = [
        tile
        for tile in tiles
        if np.all(np.abs((_center(tile)[1:] - _center(center_tile)[1:]) / tile_size_yx) <= 1.1)
    ]
    if len(neighborhood) < 5:
        raise ValueError(
            "Orthogonal phase requires a center tile with adjacent coverage; "
            f"found {len(neighborhood)} neighborhood tiles"
        )
    return neighborhood, center_tile


def _channel_array(tile: legacy.TileRecord, channel: int):
    group = zarr.open(str(tile.path), mode="r")
    array = ngff.level_array(group, context=tile.path)
    axes = ngff.axes(group, array)
    if axes == "CZYX":
        if channel < 0 or channel >= int(array.shape[0]):
            raise ValueError(f"Channel {channel} is outside {tile.path} channel count {array.shape[0]}")
        return array[channel]
    if axes == "ZYX":
        if channel != 0:
            raise ValueError(f"Channel {channel} is outside single-channel tile {tile.path}")
        return array
    raise ValueError(f"Unsupported axes {axes!r} in {tile.path}")


def _sample_plane(
    tile: legacy.TileRecord,
    *,
    channel: int,
    plane: Literal["zx", "zy"],
    z_world: np.ndarray,
    lateral_world: np.ndarray,
    cross_world: float,
) -> np.ndarray:
    array = _channel_array(tile, channel)
    z_coord = (z_world - tile.translation_zyx_um[0]) / tile.scale_zyx_um[0]
    if plane == "zx":
        y_coord = np.asarray([(cross_world - tile.translation_zyx_um[1]) / tile.scale_zyx_um[1]])
        x_coord = (lateral_world - tile.translation_zyx_um[2]) / tile.scale_zyx_um[2]
    else:
        y_coord = (lateral_world - tile.translation_zyx_um[1]) / tile.scale_zyx_um[1]
        x_coord = np.asarray([(cross_world - tile.translation_zyx_um[2]) / tile.scale_zyx_um[2]])
    coordinates = (z_coord, y_coord, x_coord)
    starts = [max(0, int(np.floor(values.min())) - 1) for values in coordinates]
    stops = [
        min(int(array.shape[axis]), int(np.ceil(values.max())) + 2) for axis, values in enumerate(coordinates)
    ]
    block = np.asarray(
        array[tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))],
        dtype=np.float32,
    )
    if plane == "zx":
        shape = (z_coord.size, x_coord.size)
        grid = np.asarray(
            [
                np.broadcast_to((z_coord - starts[0])[:, None], shape),
                np.full(shape, y_coord[0] - starts[1]),
                np.broadcast_to((x_coord - starts[2])[None, :], shape),
            ]
        )
    else:
        shape = (z_coord.size, y_coord.size)
        grid = np.asarray(
            [
                np.broadcast_to((z_coord - starts[0])[:, None], shape),
                np.broadcast_to((y_coord - starts[1])[None, :], shape),
                np.full(shape, x_coord[0] - starts[2]),
            ]
        )
    return ndimage.map_coordinates(
        block,
        grid,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)


def _intersects_line(
    tile: legacy.TileRecord,
    *,
    plane: Literal["zx", "zy"],
    cross_world: float,
) -> bool:
    start, stop = _bounds(tile)
    cross_axis = 1 if plane == "zx" else 2
    return bool(start[cross_axis] <= cross_world < stop[cross_axis])


def render_dumb_plane(
    tiles: list[legacy.TileRecord],
    *,
    channel: int,
    plane: Literal["zx", "zy"],
    z_world: np.ndarray,
    lateral_world: np.ndarray,
    cross_world: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lateral_axis = 2 if plane == "zx" else 1
    image = np.zeros((z_world.size, lateral_world.size), dtype=np.float32)
    coverage = np.zeros(image.shape, dtype=bool)
    contributors = []
    for tile in tiles:
        start, stop = _bounds(tile)
        if not _intersects_line(tile, plane=plane, cross_world=cross_world):
            continue
        if stop[lateral_axis] <= lateral_world[0] or start[lateral_axis] >= lateral_world[-1]:
            continue
        sampled = _sample_plane(
            tile,
            channel=channel,
            plane=plane,
            z_world=z_world,
            lateral_world=lateral_world,
            cross_world=cross_world,
        )
        tile_coverage = (
            (z_world >= start[0])[:, None]
            & (z_world < stop[0])[:, None]
            & (lateral_world >= start[lateral_axis])[None, :]
            & (lateral_world < stop[lateral_axis])[None, :]
        )
        image = np.maximum(image, np.where(tile_coverage, sampled, 0.0))
        coverage |= tile_coverage
        contributors.append(tile.tile)
    if len(contributors) < 2:
        raise ValueError(f"{plane.upper()} dumb stitch requires adjacent tiles; found {len(contributors)}")
    return image, coverage, contributors


def _expanded_shifted_pair(
    fixed: np.ndarray,
    moving: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.floor(np.minimum(0.0, shift)).astype(np.int64)
    stops = np.ceil(
        np.maximum(
            np.asarray(fixed.shape, dtype=np.float64),
            np.asarray(moving.shape, dtype=np.float64) + shift,
        )
    ).astype(np.int64)
    output_shape = tuple(int(value) for value in stops - starts)
    fixed_output = np.zeros(output_shape, dtype=fixed.dtype)
    moving_output = np.zeros(output_shape, dtype=moving.dtype)
    insert = tuple(
        slice(int(-start), int(-start + size)) for start, size in zip(starts, fixed.shape, strict=True)
    )
    fixed_output[insert] = fixed
    moving_output[insert] = moving
    order = 0 if moving.dtype == np.bool_ else 1
    shifted = ndimage.shift(
        moving_output,
        shift=shift,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(moving.dtype, copy=False)
    return fixed_output, shifted


def _scale_u16(image: np.ndarray) -> np.ndarray:
    positive = image[np.isfinite(image) & (image > 0)]
    if positive.size == 0:
        return np.zeros(image.shape, dtype=np.uint16)
    low, high = np.percentile(positive, [1.0, 99.8])
    scaled = np.clip((image - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return np.rint(scaled * np.iinfo(np.uint16).max).astype(np.uint16)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _analyze_plane(
    fixed_tiles: list[legacy.TileRecord],
    moving_tiles: list[legacy.TileRecord],
    *,
    fixed_channel: int,
    moving_channel: int,
    plane: Literal["zx", "zy"],
    cross_world: float,
    region_start: np.ndarray,
    region_stop: np.ndarray,
    z_spacing: float,
    lateral_spacing: float,
    max_shift_um: float,
    fixed_transform: IntensityTransform,
    moving_transform: IntensityTransform,
    output_dir: Path,
) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    lateral_axis = 2 if plane == "zx" else 1
    fixed_line_tiles = [
        tile for tile in fixed_tiles if _intersects_line(tile, plane=plane, cross_world=cross_world)
    ]
    moving_line_tiles = [
        tile
        for tile in moving_tiles
        if _intersects_line(tile, plane=plane, cross_world=cross_world)
        and _bounds(tile)[1][lateral_axis] > region_start[lateral_axis]
        and _bounds(tile)[0][lateral_axis] < region_stop[lateral_axis]
    ]
    if not fixed_line_tiles or not moving_line_tiles:
        raise ValueError(f"No fixed/moving tiles intersect the {plane.upper()} center line")
    z_world = np.arange(
        max(
            min(_bounds(tile)[0][0] for tile in fixed_line_tiles),
            min(_bounds(tile)[0][0] for tile in moving_line_tiles),
        ),
        min(
            max(_bounds(tile)[1][0] for tile in fixed_line_tiles),
            max(_bounds(tile)[1][0] for tile in moving_line_tiles),
        ),
        z_spacing,
    )
    lateral_world = np.arange(region_start[lateral_axis], region_stop[lateral_axis], lateral_spacing)
    if z_world.size < 2 or lateral_world.size < 2:
        raise ValueError(f"{plane.upper()} fixed/moving physical overlap is empty")
    fixed_raw, fixed_coverage, fixed_names = render_dumb_plane(
        fixed_line_tiles,
        channel=fixed_channel,
        plane=plane,
        z_world=z_world,
        lateral_world=lateral_world,
        cross_world=cross_world,
    )
    moving_raw, moving_coverage, moving_names = render_dumb_plane(
        moving_line_tiles,
        channel=moving_channel,
        plane=plane,
        z_world=z_world,
        lateral_world=lateral_world,
        cross_world=cross_world,
    )
    fixed = apply_intensity_transform(fixed_raw, fixed_transform)
    moving = apply_intensity_transform(moving_raw, moving_transform)
    if plane == "zx":
        fixed_phase = fixed[:, None, :]
        moving_phase = moving[:, None, :]
        sigma = (1.0, 0.0, 4.0)
        radius = np.asarray([max_shift_um / z_spacing, 0.0, max_shift_um / lateral_spacing])
        lateral_index = 2
    else:
        fixed_phase = fixed[:, :, None]
        moving_phase = moving[:, :, None]
        sigma = (1.0, 4.0, 0.0)
        radius = np.asarray([max_shift_um / z_spacing, max_shift_um / lateral_spacing, 0.0])
        lateral_index = 1
    shift_3d, metadata = phasecorr_shift_gpu(
        fixed_phase,
        moving_phase,
        fft_highpass_sigma_zyx=sigma,
        spatial_highpass_sigma=None,
        search_center_zyx=np.zeros(3),
        max_shift_from_center_zyx=radius,
    )
    shift_2d = np.asarray([shift_3d[0], shift_3d[lateral_index]], dtype=np.float64)
    fixed_before, moving_before = _expanded_shifted_pair(fixed, moving, np.zeros(2))
    fixed_before_coverage, moving_before_coverage = _expanded_shifted_pair(
        fixed_coverage, moving_coverage, np.zeros(2)
    )
    fixed_after, moving_after = _expanded_shifted_pair(fixed, moving, shift_2d)
    fixed_after_coverage, moving_after_coverage = _expanded_shifted_pair(
        fixed_coverage, moving_coverage, shift_2d
    )
    before_mask = fixed_before_coverage & moving_before_coverage
    after_mask = fixed_after_coverage & moving_after_coverage
    corr_before = phase_metrics.corrcoef_on_mask(fixed_before, moving_before, before_mask)
    corr_after = phase_metrics.corrcoef_on_mask(fixed_after, moving_after, after_mask)
    overlap_before = int(before_mask.sum())
    overlap_after = int(after_mask.sum())
    retention = overlap_after / overlap_before if overlap_before else 0.0

    fixed_tif = output_dir / f"{plane}.fixed.tif"
    moving_tif = output_dir / f"{plane}.moving.tif"
    before_png = output_dir / f"{plane}.before.png"
    after_png = output_dir / f"{plane}.after.png"
    tifffile.imwrite(fixed_tif, _scale_u16(fixed), photometric="minisblack")
    tifffile.imwrite(moving_tif, _scale_u16(moving), photometric="minisblack")
    display_scale = z_spacing / lateral_spacing
    qc.write_overlay_scaled(before_png, left=fixed_before, right=moving_before, y_scale=display_scale)
    qc.write_overlay_scaled(after_png, left=fixed_after, right=moving_after, y_scale=display_scale)
    shift_um = shift_2d * np.asarray([z_spacing, lateral_spacing])
    result = {
        "cross_section_world_um": cross_world,
        "shape": list(fixed.shape),
        "spacing_um": [z_spacing, lateral_spacing],
        "fixed_contributors": fixed_names,
        "moving_contributors": moving_names,
        "shift_to_apply_moving_px": shift_2d.tolist(),
        "shift_to_apply_moving_um": shift_um.tolist(),
        "final_applied_axes": ["z"],
        "diagnostic_only_lateral_shift_um": float(shift_um[1]),
        "corr_before": corr_before,
        "corr_after": corr_after,
        "overlap_before": overlap_before,
        "overlap_after": overlap_after,
        "overlap_retention": retention,
        "phase_metadata": metadata,
        "fixed_tif": str(fixed_tif.resolve()),
        "moving_tif": str(moving_tif.resolve()),
        "before_overlay": str(before_png.resolve()),
        "after_overlay": str(after_png.resolve()),
    }
    return result, [(f"{plane.upper()} before", before_png), (f"{plane.upper()} after", after_png)]


def run_orthogonal_dumb_phase(
    *,
    fixed_payload: dict[str, Any],
    moving_payload: dict[str, Any],
    fixed_tile_dir: Path | None,
    moving_tile_dir: Path | None,
    fixed_channel: int,
    moving_channel: int,
    fixed_transform: IntensityTransform,
    moving_transform: IntensityTransform,
    output_dir: Path,
    max_shift_um: float,
    lateral_factor: int = DEFAULT_LATERAL_FACTOR,
) -> OrthogonalPhaseResult:
    """Refine Z from composite ZX/ZY dumb stitches after global XY placement."""
    if lateral_factor < 1:
        raise ValueError("orthogonal lateral_factor must be positive")
    register_jpegxr_codec()
    fixed_tiles = position_tiles(fixed_payload, tile_dir=fixed_tile_dir)
    moving_tiles = position_tiles(moving_payload, tile_dir=moving_tile_dir)
    fixed_neighborhood, center_tile = select_center_neighborhood(fixed_tiles)
    center = _center(center_tile)
    starts = np.asarray([_bounds(tile)[0] for tile in fixed_neighborhood])
    stops = np.asarray([_bounds(tile)[1] for tile in fixed_neighborhood])
    region_start = starts.min(axis=0)
    region_stop = stops.max(axis=0)
    z_spacing = abs(float(center_tile.scale_zyx_um[0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    panels = []
    for plane in ("zx", "zy"):
        lateral_axis = 2 if plane == "zx" else 1
        cross_world = float(center[1] if plane == "zx" else center[2])
        results[plane], plane_panels = _analyze_plane(
            fixed_neighborhood,
            moving_tiles,
            fixed_channel=fixed_channel,
            moving_channel=moving_channel,
            plane=plane,
            cross_world=cross_world,
            region_start=region_start,
            region_stop=region_stop,
            z_spacing=z_spacing,
            lateral_spacing=(abs(float(center_tile.scale_zyx_um[lateral_axis])) * lateral_factor),
            max_shift_um=max_shift_um,
            fixed_transform=fixed_transform,
            moving_transform=moving_transform,
            output_dir=output_dir,
        )
        panels.extend(plane_panels)
    zx_z = float(results["zx"]["shift_to_apply_moving_um"][0])
    zy_z = float(results["zy"]["shift_to_apply_moving_um"][0])
    disagreement = abs(zx_z - zy_z)
    z_residual = float((zx_z + zy_z) / 2.0)
    contact_sheet = output_dir / "orthogonal.png"
    qc.write_contact_sheet(contact_sheet, panels)
    summary_path = output_dir / "orthogonal.summary.json"
    summary = {
        "schema_version": 1,
        "artifact_type": "squisher_lightsheet.orthogonal_dumb_phase.v1",
        "contract": "XY-corrected composite ZX/ZY dumb stitches refine Z only",
        "fixed_center_tile": center_tile.tile,
        "fixed_center_zyx_um": center.tolist(),
        "fixed_neighborhood": [tile.tile for tile in fixed_neighborhood],
        "lateral_factor": lateral_factor,
        "max_shift_um": max_shift_um,
        "zx_z_residual_um": zx_z,
        "zy_z_residual_um": zy_z,
        "z_disagreement_um": disagreement,
        "applied_z_residual_um": z_residual,
        "lateral_components_applied": False,
        "human_review_required": True,
        "results": results,
        "contact_sheet": str(contact_sheet.resolve()),
    }
    _write_json_atomic(summary_path, summary)
    return OrthogonalPhaseResult(
        z_residual_um=z_residual,
        summary=summary_path.resolve(),
        contact_sheet=contact_sheet.resolve(),
    )
