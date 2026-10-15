from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import math
import json
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.artifacts import stamp_artifact
from squisher_lightsheet._legacy import create_lr_position_file as legacy
from squisher_lightsheet.modes import ModeName, POSITION_MODES
from squisher_lightsheet.workflow_runner import sha256_file


TileInfo = legacy.TileInfo
JoinedTile = legacy.JoinedTile
LEFT_SIDE = legacy.LEFT_SIDE
RIGHT_SIDE = legacy.RIGHT_SIDE
DIMENSIONS = legacy.DIMENSIONS


def compute_joined_tiles(
    left_tiles: list[TileInfo],
    right_tiles: list[TileInfo],
    *,
    mode: ModeName = "tltr_x_join_center_z_phase",
    overlap_fraction: float | None = None,
) -> tuple[list[JoinedTile], dict[str, Any]]:
    selected = POSITION_MODES[mode]
    joined, diagnostics = legacy.compute_joined_tiles(
        left_tiles,
        right_tiles,
        join_axis=selected["join_axis"],
        overlap_fraction=selected["overlap_fraction"] if overlap_fraction is None else overlap_fraction,
        right_flip_axes=selected["right_flip_axes"],
    )
    diagnostics = {
        **diagnostics,
        "mode": mode,
        "artifact_type": "lightsheet.position.v1",
    }
    return joined, diagnostics


def position_payload(joined: list[JoinedTile], diagnostics: dict[str, Any]) -> dict[str, Any]:
    right_scale_sign = [-1.0 if dim in diagnostics["right_flip_axes"] else 1.0 for dim in DIMENSIONS]
    return stamp_artifact(
        {
            "units": "micrometer",
            "source": "metadata-only OME Plane Position and PhysicalSize",
            "transform": {
                "left": {"scale_zyx": [1.0, 1.0, 1.0]},
                "right": {
                    "scale_zyx": right_scale_sign,
                    "description": "scale selected axes, then offset",
                },
            },
            "diagnostics": diagnostics,
            "tiles": [
                {
                    "tile": tile.info.tile,
                    "side": tile.info.side,
                    "path": str(tile.info.path),
                    "translation_um": legacy.dict_zyx(tile.translation_zyx),
                    "scale_um": legacy.dict_zyx(tile.scale_zyx),
                }
                for tile in joined
            ],
        },
        "lightsheet.position.v1",
    )


def create_position_file(
    *,
    left_dir: Path,
    right_dir: Path,
    output: Path,
    mode: ModeName = "tltr_x_join_center_z_phase",
    overlap_fraction: float | None = None,
    plot_title: str = "metadata joined tile positions",
) -> Path:
    output = output.resolve()
    joined, diagnostics = compute_joined_tiles(
        legacy.read_tiles(left_dir.resolve(), side=LEFT_SIDE),
        legacy.read_tiles(right_dir.resolve(), side=RIGHT_SIDE),
        mode=mode,
        overlap_fraction=overlap_fraction,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(position_payload(joined, diagnostics), indent=2) + "\n")
    legacy.write_positions_csv(output, joined)
    legacy.render_positions(output, joined, title=plot_title)
    return output


def create_single_position_file(
    *,
    input_dir: Path,
    output: Path,
    side: str,
    plot_title: str = "metadata tile positions",
    progress: Callable[[str], None] | None = None,
    position_overrides: Path | None = None,
) -> Path:
    """Write acquisition-local tile positions without joining another source view."""
    input_dir = input_dir.resolve()
    paths = sorted(input_dir.glob("*.ome.tif"))
    if not paths:
        raise FileNotFoundError(f"No *.ome.tif files found in {input_dir}")
    if progress is not None:
        progress(f"single-position start side={side} tiles={len(paths)} input={input_dir}")
    tiles: list[TileInfo] = []
    for index, path in enumerate(paths, start=1):
        if progress is not None:
            progress(f"single-position metadata tile={index}/{len(paths)} file={path.name}")
        tiles.append(legacy.read_tile_info(path, side=side))
    if position_overrides is not None:
        tiles = _apply_position_overrides(tiles, position_overrides)
    joined: list[JoinedTile] = []
    for tile in tiles:
        tile_bounds = legacy.bounds_zyx(tile.translation_zyx, tile.spacing_zyx, tile.shape_zyx)
        joined.append(
            JoinedTile(
                info=tile,
                translation_zyx=tile.translation_zyx,
                scale_zyx=tile.spacing_zyx,
                raw_bounds_zyx=tile_bounds,
                joined_bounds_zyx=tile_bounds,
            )
        )
    bounds = [tile.joined_bounds_zyx for tile in joined]
    payload = stamp_artifact(
        {
            "units": "micrometer",
            "source": "metadata-only OME Plane Position and PhysicalSize",
            "transform": {side: {"scale_zyx": [1.0, 1.0, 1.0]}},
            "diagnostics": {
                "mode": "single_view_metadata",
                "side": side,
                "tile_count": len(joined),
                "joined_min_zyx_um": np.min([value[0] for value in bounds], axis=0).tolist(),
                "joined_max_zyx_um": np.max([value[1] for value in bounds], axis=0).tolist(),
            },
            "tiles": [
                {
                    "tile": tile.info.tile,
                    "side": side,
                    "path": str(tile.info.path),
                    "translation_um": legacy.dict_zyx(tile.translation_zyx),
                    "scale_um": legacy.dict_zyx(tile.scale_zyx),
                }
                for tile in joined
            ],
        },
        "lightsheet.position.v1",
    )
    if position_overrides is not None:
        payload["position_overrides"] = {
            "path": str(position_overrides.resolve()),
            "sha256": sha256_file(position_overrides),
        }
        additional_pairs = json.loads(position_overrides.read_text()).get("additional_registration_pairs", [])
        if additional_pairs:
            payload["additional_registration_pairs"] = additional_pairs
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    legacy.write_positions_csv(output, joined)
    legacy.render_positions(output, joined, title=plot_title)
    if progress is not None:
        progress(f"single-position complete output={output}")
    return output


def _apply_position_overrides(tiles: list[TileInfo], path: Path) -> list[TileInfo]:
    """Apply only reviewed translations bound to the exact source tile and original pose."""
    payload = json.loads(path.read_text())
    if (
        payload.get("artifact_type") != "lightsheet.position_overrides.v1"
        or payload.get("approved") is not True
    ):
        raise ValueError(f"Unapproved or invalid position overrides: {path}")
    records = payload.get("tiles")
    if not isinstance(records, dict) or not records or not set(records) <= {tile.tile for tile in tiles}:
        raise ValueError(f"Position overrides contain unknown or missing tiles: {path}")
    result = []
    for tile in tiles:
        if tile.tile not in records:
            result.append(tile)
            continue
        record = records[tile.tile]
        stat = tile.path.stat()
        if (
            not isinstance(record, dict)
            or record.get("source_identity") != {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            or record.get("original_translation_um_zyx") != list(tile.translation_zyx)
        ):
            raise ValueError(f"Position override no longer matches source or original pose: {tile.path}")
        values = record.get("translation_um_zyx")
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in values
            )
        ):
            raise ValueError(f"Expected three finite ZYX position values for {tile.path}")
        result.append(replace(tile, translation_zyx=tuple(float(value) for value in values)))
    return result
