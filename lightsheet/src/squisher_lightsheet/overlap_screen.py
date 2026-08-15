from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.artifact_io import registration_input_fingerprint
from squisher_lightsheet.method8_stitch_register import (
    TileInfo,
    _all_adjacent_pairs,
    _crop_bounds_for_pair,
    _load_tiles,
    _tile_id,
    _z_chunks,
    _zarr_name,
)
from squisher_lightsheet.ngff import axes as ngff_axes
from squisher_lightsheet.ngff import open_level_array
from squisher_lightsheet.ome_metadata_dumb_stitch import read_tile_metadata

ARTIFACT_TYPE = "lightsheet.level2_overlap_screen.v1"
LOW_CONTENT_REASON = "level2_low_content"


@dataclass(frozen=True)
class SampleResult:
    chunk: int
    z_start: int
    z_stop: int
    fixed_z: int
    moving_z: int
    pixel_count: int
    minimum_foreground_pixels: int
    fixed_foreground_pixels: int
    moving_foreground_pixels: int
    fixed_foreground_fraction: float
    moving_foreground_fraction: float
    status: str
    reason: str | None


def _level_tiles(
    position_json: Path,
    zarr_dir: Path,
    *,
    level: int,
    channel: int,
) -> dict[str, TileInfo]:
    payload = json.loads(position_json.read_text())
    tiles: dict[str, TileInfo] = {}
    for record in payload["tiles"]:
        tile_name = _zarr_name(str(record["tile"]))
        tile_id = _tile_id(tile_name)
        path = zarr_dir / tile_name
        metadata = read_tile_metadata(path, level=level)
        if metadata.axes == "CZYX":
            if not 0 <= channel < metadata.shape[0]:
                raise ValueError(f"channel {channel} is outside {metadata.shape} in {path}")
            shape_zyx = metadata.shape[1:]
        elif metadata.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"channel {channel} requested from ZYX tile {path}")
            shape_zyx = metadata.shape
        else:
            raise ValueError(f"expected CZYX or ZYX in {path}, found {metadata.axes}")
        tiles[tile_id] = TileInfo(
            tile_id=tile_id,
            tile_name=tile_name,
            path=path,
            start_um_zyx=np.asarray(metadata.translation_um_zyx, dtype=np.float64),
            spacing_um_zyx=np.asarray(metadata.spacing_um_zyx, dtype=np.float64),
            shape_zyx=np.asarray(shape_zyx, dtype=np.int64),
            channel=channel,
        )
    return tiles


def _overlap_slices(
    fixed: TileInfo,
    moving: TileInfo,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    """Return physically aligned overlap slices without level-0 size constraints."""
    if not np.allclose(fixed.spacing_um_zyx, moving.spacing_um_zyx, rtol=0.0, atol=1e-9):
        raise ValueError(f"spacing differs for {fixed.tile_name} and {moving.tile_name}")

    spacing_abs = np.abs(fixed.spacing_um_zyx)
    fixed_stop_um = fixed.start_um_zyx + fixed.shape_zyx * fixed.spacing_um_zyx
    moving_stop_um = moving.start_um_zyx + moving.shape_zyx * moving.spacing_um_zyx
    overlap_start_um = np.maximum(
        np.minimum(fixed.start_um_zyx, fixed_stop_um),
        np.minimum(moving.start_um_zyx, moving_stop_um),
    )
    overlap_stop_um = np.minimum(
        np.maximum(fixed.start_um_zyx, fixed_stop_um),
        np.maximum(moving.start_um_zyx, moving_stop_um),
    )
    overlap_shape = np.floor((overlap_stop_um - overlap_start_um) / spacing_abs + 1e-6).astype(
        np.int64
    )
    if np.any(overlap_shape <= 0):
        raise ValueError(f"{fixed.tile_name} and {moving.tile_name} do not overlap")

    crop_size_um = overlap_shape * spacing_abs

    def slices_for(tile: TileInfo) -> tuple[slice, slice, slice]:
        crop_high_um = overlap_start_um + crop_size_um
        coordinate_um = np.where(tile.spacing_um_zyx >= 0, overlap_start_um, crop_high_um)
        start = np.rint((coordinate_um - tile.start_um_zyx) / tile.spacing_um_zyx).astype(np.int64)
        start = np.clip(start, 0, tile.shape_zyx - overlap_shape)
        return tuple(
            slice(int(axis_start), int(axis_start + size))
            for axis_start, size in zip(start, overlap_shape, strict=True)
        )

    return slices_for(fixed), slices_for(moving)


def _read_plane(
    array: Any,
    axes: str,
    channel: int,
    z_index: int,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    selection = (
        (channel, z_index, y_slice, x_slice)
        if axes == "CZYX"
        else (z_index, y_slice, x_slice)
    )
    return np.asarray(array[selection])


def screen_level2_overlaps(
    *,
    position_json: Path,
    zarr_dir: Path,
    output: Path,
    threshold: float,
    level: int = 2,
    channel: int = 0,
    z_chunks: int = 6,
    min_foreground_pixels: int = 256,
    min_foreground_fraction: float = 0.05,
    progress: Any | None = None,
) -> Path:
    """Write a fail-closed content decision for every adjacent pair and z chunk."""
    progress = progress or (lambda _message: None)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite overlap screen: {output}")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if not 0.0 <= min_foreground_fraction <= 1.0:
        raise ValueError("min foreground fraction must be between zero and one")

    level0_tiles = _load_tiles(position_json, zarr_dir, channel=channel)
    level_tiles = _level_tiles(position_json, zarr_dir, level=level, channel=channel)
    pairs = _all_adjacent_pairs(level0_tiles)
    if not pairs:
        raise ValueError("No adjacent tile pairs were found for screening")

    arrays: dict[str, Any] = {}
    array_axes: dict[str, str] = {}
    import zarr

    for tile_id, tile in level_tiles.items():
        array = open_level_array(tile.path, level=level)
        group = zarr.open_group(str(tile.path), mode="r")
        axes = ngff_axes(group, array)
        if axes not in {"CZYX", "ZYX"}:
            raise ValueError(f"expected CZYX or ZYX in {tile.path}, found {axes}")
        arrays[tile_id] = array
        array_axes[tile_id] = axes

    pair_results: list[dict[str, Any]] = []
    accepted_unit_count = 0
    for pair_index, pair in enumerate(pairs, start=1):
        fixed_id, moving_id = pair.split("-", 1)
        fixed0 = level0_tiles[fixed_id]
        moving0 = level0_tiles[moving_id]
        fixed = level_tiles[fixed_id]
        moving = level_tiles[moving_id]
        fixed_slices, moving_slices = _overlap_slices(fixed, moving)
        _fixed_z_slice, fixed_y_slice, fixed_x_slice = fixed_slices
        _moving_z_slice, moving_y_slice, moving_x_slice = moving_slices
        samples: list[SampleResult] = []
        for chunk_index, (z_start, z_stop) in enumerate(_z_chunks(int(fixed0.shape_zyx[0]), z_chunks)):
            _axis, fixed0_slices, moving0_slices, _shape = _crop_bounds_for_pair(
                fixed0,
                moving0,
                z_start=z_start,
                z_depth=z_stop - z_start,
            )
            fixed0_z = (fixed0_slices[0].start + fixed0_slices[0].stop - 1) // 2
            moving0_z = (moving0_slices[0].start + moving0_slices[0].stop - 1) // 2
            fixed_z_um = fixed0.start_um_zyx[0] + fixed0_z * fixed0.spacing_um_zyx[0]
            moving_z_um = moving0.start_um_zyx[0] + moving0_z * moving0.spacing_um_zyx[0]
            fixed_z = int(round((fixed_z_um - fixed.start_um_zyx[0]) / fixed.spacing_um_zyx[0]))
            moving_z = int(round((moving_z_um - moving.start_um_zyx[0]) / moving.spacing_um_zyx[0]))
            if not 0 <= fixed_z < fixed.shape_zyx[0] or not 0 <= moving_z < moving.shape_zyx[0]:
                raise ValueError(f"planned z sample for {pair} chunk {chunk_index} is outside level {level}")
            fixed_plane = _read_plane(
                arrays[fixed_id], array_axes[fixed_id], channel, fixed_z, fixed_y_slice, fixed_x_slice
            )
            moving_plane = _read_plane(
                arrays[moving_id], array_axes[moving_id], channel, moving_z, moving_y_slice, moving_x_slice
            )
            if fixed_plane.shape != moving_plane.shape:
                raise ValueError(
                    f"level-{level} overlap shapes differ for {pair}: "
                    f"{fixed_plane.shape} vs {moving_plane.shape}"
                )
            pixel_count = int(fixed_plane.size)
            required_pixels = max(
                min_foreground_pixels,
                math.ceil(min_foreground_fraction * pixel_count),
            )
            fixed_pixels = int(np.count_nonzero(fixed_plane > threshold))
            moving_pixels = int(np.count_nonzero(moving_plane > threshold))
            accepted = fixed_pixels >= required_pixels and moving_pixels >= required_pixels
            accepted_unit_count += int(accepted)
            samples.append(
                SampleResult(
                    chunk=chunk_index,
                    z_start=z_start,
                    z_stop=z_stop,
                    fixed_z=fixed_z,
                    moving_z=moving_z,
                    pixel_count=pixel_count,
                    minimum_foreground_pixels=required_pixels,
                    fixed_foreground_pixels=fixed_pixels,
                    moving_foreground_pixels=moving_pixels,
                    fixed_foreground_fraction=fixed_pixels / pixel_count,
                    moving_foreground_fraction=moving_pixels / pixel_count,
                    status="accepted" if accepted else "low_content",
                    reason=None if accepted else LOW_CONTENT_REASON,
                )
            )
        pair_results.append(
            {
                "pair": pair,
                "fixed_tile": fixed.tile_name,
                "moving_tile": moving.tile_name,
                "status": "accepted" if any(sample.status == "accepted" for sample in samples) else "low_content",
                "samples": [asdict(sample) for sample in samples],
            }
        )
        progress(f"screen pair={pair_index}/{len(pairs)} id={pair}")

    unit_count = len(pairs) * z_chunks
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "position_json": str(position_json.resolve()),
        "zarr_dir": str(zarr_dir.resolve()),
        "input_fingerprint": registration_input_fingerprint(position_json, zarr_dir),
        "settings": {
            "level": level,
            "channel": channel,
            "threshold": float(threshold),
            "threshold_source": "human_reviewed_threshold",
            "z_chunks": z_chunks,
            "sample": "center_plane_per_registration_z_chunk",
            "min_foreground_pixels": min_foreground_pixels,
            "min_foreground_fraction": min_foreground_fraction,
            "unit_acceptance": "foreground_requirement_met_in_both_tiles",
        },
        "tile_count": len(level_tiles),
        "pair_count": len(pairs),
        "unit_count": unit_count,
        "accepted_unit_count": accepted_unit_count,
        "low_content_unit_count": unit_count - accepted_unit_count,
        "pairs": pair_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    progress(
        f"wrote {output} accepted_units={accepted_unit_count} "
        f"low_content_units={unit_count - accepted_unit_count}"
    )
    return output


def load_level2_screen(
    path: Path,
    *,
    position_json: Path,
    zarr_dir: Path,
    expected_pairs: list[str],
    z_chunks: int,
    channel: int,
    threshold: float | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate a complete screen manifest and return decisions by pair and chunk."""
    payload = json.loads(path.read_text())
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"{path} is not a level-2 overlap screen")
    for key, expected in (("position_json", position_json), ("zarr_dir", zarr_dir)):
        value = payload.get(key)
        if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
            raise ValueError(f"{path} belongs to a different {key}: {value}")
    if payload.get("input_fingerprint") != registration_input_fingerprint(position_json, zarr_dir):
        raise ValueError(f"{path} input fingerprint differs from current registration inputs")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"{path} is missing settings")
    expected_settings = {"level": 2, "channel": channel, "threshold": threshold, "z_chunks": z_chunks}
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            raise ValueError(f"{path} setting {key}={settings.get(key)!r} differs from {expected!r}")

    records = payload.get("pairs")
    if not isinstance(records, list):
        raise ValueError(f"{path} is missing pair records")
    by_pair: dict[str, dict[str, Any]] = {}
    for record in records:
        pair = record.get("pair") if isinstance(record, dict) else None
        if not isinstance(pair, str) or pair in by_pair:
            raise ValueError(f"{path} contains an invalid or duplicate pair: {pair!r}")
        by_pair[pair] = record
    if set(by_pair) != set(expected_pairs):
        missing = sorted(set(expected_pairs) - set(by_pair))
        extra = sorted(set(by_pair) - set(expected_pairs))
        raise ValueError(f"{path} pair set differs from registration; missing={missing}, extra={extra}")

    decisions: dict[tuple[str, int], dict[str, Any]] = {}
    for pair in expected_pairs:
        samples = by_pair[pair].get("samples")
        if not isinstance(samples, list) or len(samples) != z_chunks:
            raise ValueError(f"{path} pair {pair} does not contain {z_chunks} samples")
        for sample in samples:
            chunk = sample.get("chunk") if isinstance(sample, dict) else None
            status = sample.get("status") if isinstance(sample, dict) else None
            reason = sample.get("reason") if isinstance(sample, dict) else None
            if not isinstance(chunk, int) or not 0 <= chunk < z_chunks:
                raise ValueError(f"{path} pair {pair} has invalid chunk {chunk!r}")
            if status not in {"accepted", "low_content"}:
                raise ValueError(f"{path} pair {pair} chunk {chunk} has invalid status {status!r}")
            if status == "low_content" and reason != LOW_CONTENT_REASON:
                raise ValueError(f"{path} pair {pair} chunk {chunk} has invalid reason {reason!r}")
            key = (pair, chunk)
            if key in decisions:
                raise ValueError(f"{path} contains duplicate unit {pair} chunk {chunk}")
            decisions[key] = sample
    expected_unit_count = len(expected_pairs) * z_chunks
    if len(decisions) != expected_unit_count:
        raise ValueError(f"{path} contains {len(decisions)}/{expected_unit_count} required units")
    if not any(sample["status"] == "accepted" for sample in decisions.values()):
        raise ValueError(f"{path} accepted no registration units")
    return decisions
