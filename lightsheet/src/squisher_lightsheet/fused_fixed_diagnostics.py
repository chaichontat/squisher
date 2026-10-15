"""Numerical agreement of accepted maps on their shared native source domains."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.artifact_io import read_json_object
from squisher_lightsheet.tile_phase import flattened_channel_count, tile_record_from_position_record


@dataclass(frozen=True)
class _SourceMap:
    identity: dict[str, Any]
    source_path: Path
    channel: int
    start: np.ndarray
    stop: np.ndarray
    matrix_um: np.ndarray
    offset_um: np.ndarray


def _path(value: Any, relative_to: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{relative_to} requires a nonempty path, got {value!r}")
    path = Path(value)
    return (path if path.is_absolute() else relative_to.parent / path).resolve()


def _array(row: dict[str, Any], key: str, shape: tuple[int, ...], *, integer: bool = False) -> np.ndarray:
    values = np.asarray(row.get(key))
    if values.shape != shape or values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"requires finite {key} with shape {shape}")
    if integer and values.dtype.kind not in "iu":
        raise ValueError(f"requires integer {key}")
    return values.astype(np.int64 if integer else np.float64)


def _position_records(path: Path, key: str) -> dict[str, dict[str, Any]]:
    payload = read_json_object(path)
    if not isinstance(payload.get("tiles"), list):
        raise ValueError(f"{path} requires a tiles list")
    result = {}
    for row in payload["tiles"]:
        if not isinstance(row, dict):
            raise ValueError(f"{path} requires object tile records")
        identity = str(_path(row.get(key), path)) if key == "path" else row.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"{path} requires unique {key} identities")
        result[identity] = row
    return result


def _source_map(
    path: Path,
    row: dict[str, Any],
    geometry: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    fixed_path: Path,
) -> _SourceMap:
    source_path = _path(row.get("moving_path"), path)
    tile_name = row.get("moving_tile")
    if tile_name not in geometry or str(source_path) not in sources:
        raise ValueError(f"{path} has no matching moving geometry or raw source record")
    placement = geometry[tile_name]
    source = sources[str(source_path)]
    axes = source.get("axes")
    if axes not in {"CZYX", "ZYX"}:
        raise ValueError(f"{path} requires recorded native CZYX or ZYX source axes")
    source_dimensions = _array(source, "shape", (4,) if axes == "CZYX" else (3,), integer=True)
    if np.any(source_dimensions <= 0):
        raise ValueError(f"{path} requires positive recorded native source dimensions")
    tile = tile_record_from_position_record(source | {
        "tile": tile_name,
        "path": str(source_path),
        "translation_um": placement["translation_um"],
        "scale_um": placement["scale_um"],
    })
    source_shape = np.asarray(tile.shape_zyx, dtype=np.int64)
    spacing = np.asarray(tile.scale_zyx_um, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing == 0):
        raise ValueError(f"{path} has invalid moving scale")
    channel = row.get("moving_channel")
    if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
        raise ValueError(f"{path} requires a nonnegative integer moving_channel")
    channel_count = int(source_dimensions[0]) if axes == "CZYX" else flattened_channel_count(source_path)
    if channel >= channel_count:
        raise ValueError(f"{path} moving_channel is outside source channels")
    if _path(row.get("fixed_fused"), path) != fixed_path or row.get("fixed_level") != 0:
        raise ValueError(f"{path} requires the summary's native fixed grid")
    start = _array(row, "moving_start_l0_zyx", (3,), integer=True)
    stop = _array(row, "moving_stop_l0_zyx", (3,), integer=True)
    shape = _array(row, "moving_shape_zyx", (3,), integer=True)
    if (
        np.any(start < 0)
        or np.any(stop > source_shape)
        or np.any(stop <= start)
        or not np.array_equal(stop - start, shape)
    ):
        raise ValueError(f"{path} has inconsistent or out-of-bounds native source domain")
    matrix = _array(row, "selected_moving_l0_to_fixed_fused_l0_matrix_zyx", (3, 3))
    offset = _array(row, "selected_moving_l0_to_fixed_fused_l0_offset_zyx", (3,))
    if np.linalg.det(matrix) == 0:
        raise ValueError(f"{path} requires a nonsingular selected global affine")
    fixed_scale = _array(row, "fused_scale_zyx", (3,))
    fixed_origin = _array(row, "fused_translation_zyx", (3,))
    if np.any(fixed_scale <= 0):
        raise ValueError(f"{path} requires positive native fixed spacing")
    flipped = spacing < 0
    raw_start = np.where(flipped, source_shape - stop, start)
    raw_stop = np.where(flipped, source_shape - start, stop)
    raw_to_oriented_offset = np.where(flipped, source_shape - 1, 0)
    raw_to_oriented_scale = np.where(flipped, -1, 1)
    physical_matrix = fixed_scale[:, None] * matrix
    physical_offset = fixed_origin + fixed_scale * offset + physical_matrix @ raw_to_oriented_offset
    identity = {
        "window_json": str(path),
        "moving_tile": tile_name,
        "source_domain_start_raw_l0_zyx": raw_start.tolist(),
        "source_domain_stop_raw_l0_zyx": raw_stop.tolist(),
        "fixed_scale_zyx_um": fixed_scale.tolist(),
        "fixed_origin_zyx_um": fixed_origin.tolist(),
    }
    for key, value in row.items():
        if key == "selected_attempt" or "recovery" in key or key in {"residual_verdict", "transform_origin"}:
            identity[key] = value
    return _SourceMap(
        identity,
        source_path,
        channel,
        raw_start,
        raw_stop,
        physical_matrix * raw_to_oriented_scale[None, :],
        physical_offset,
    )


def _pair(first: _SourceMap, second: _SourceMap, start: np.ndarray, stop: np.ndarray) -> dict[str, Any]:
    last = stop - 1
    corners = np.asarray(list(product(*zip(start, last, strict=True))), dtype=np.float64)
    points = np.vstack((corners, (start + last) / 2))
    first_target = points @ first.matrix_um.T + first.offset_um
    second_target = points @ second.matrix_um.T + second.offset_um
    delta = second_target - first_target
    return {
        "source_path": str(first.source_path),
        "acquisition_channel": first.channel,
        "first": first.identity,
        "second": second.identity,
        "overlap_start_raw_l0_zyx": start.tolist(),
        "overlap_stop_raw_l0_zyx": stop.tolist(),
        "samples": [
            {
                "location": "center" if index == 8 else f"corner_{index}",
                "source_raw_l0_zyx": point.tolist(),
                "first_target_zyx_um": first_target[index].tolist(),
                "second_target_zyx_um": second_target[index].tolist(),
                "second_minus_first_zyx_um": delta[index].tolist(),
                "disagreement_um": float(np.linalg.norm(delta[index])),
            }
            for index, point in enumerate(points)
        ],
    }


def write_overlap_disagreement(summary_path: Path, output: Path) -> Path:
    """Compare accepted maps at overlap corners and center without reading source pixels.

    Source domains are half-open native raw index bounds. Corner samples lie at
    the first and last voxel centers; displacement is second target minus first.
    Maps from distinct resolved source paths or acquisition channels never pair.
    """
    summary_path = summary_path.resolve()
    summary = read_json_object(summary_path)
    windows = summary.get("windows")
    if not isinstance(windows, list):
        raise ValueError(f"{summary_path} requires a windows list")
    accepted: dict[Path, dict[str, Any]] = {}
    for compact in windows:
        if not isinstance(compact, dict) or compact.get("status") not in {"accepted", "rejected"}:
            raise ValueError(f"{summary_path} requires completed window statuses")
        if compact["status"] == "rejected":
            continue
        path = _path(compact.get("level0_json"), summary_path)
        row = read_json_object(path)
        if row.get("status") != "accepted":
            raise ValueError(f"{path} accepted status disagrees with summary")
        if path in accepted:
            raise ValueError(f"{summary_path} repeats accepted window {path}")
        accepted[path] = row
    groups: dict[tuple[Path, int], list[_SourceMap]] = defaultdict(list)
    input_paths = {summary_path, *accepted}
    if accepted:
        moving_position = _path(summary.get("moving_position"), summary_path)
        source_position = _path(summary.get("moving_source_position"), summary_path)
        geometry = _position_records(moving_position, "tile")
        sources = _position_records(source_position, "path")
        fixed_path = _path(summary.get("fixed_fused"), summary_path)
        input_paths.update((moving_position, source_position, fixed_path))
        fixed_grid = None
        for path, row in accepted.items():
            item = _source_map(path, row, geometry, sources, fixed_path)
            grid = (item.identity["fixed_scale_zyx_um"], item.identity["fixed_origin_zyx_um"])
            if fixed_grid is not None and grid != fixed_grid:
                raise ValueError(f"{path} fixed geometry disagrees with other accepted maps")
            fixed_grid = grid
            input_paths.add(item.source_path)
            groups[(item.source_path, item.channel)].append(item)
    pairs = []
    for key in sorted(groups):
        active: list[_SourceMap] = []
        for current in sorted(groups[key], key=lambda item: (item.start[0], item.identity["window_json"])):
            active = [item for item in active if item.stop[0] > current.start[0]]
            for previous in active:
                start = np.maximum(previous.start, current.start)
                stop = np.minimum(previous.stop, current.stop)
                if np.all(stop > start):
                    pairs.append(_pair(previous, current, start, stop))
            active.append(current)
    payload = {
        "artifact_type": "squisher_lightsheet.fused_fixed_overlap_disagreement.v1",
        "summary": str(summary_path),
        "source_coordinates": "native raw ZYX voxel centers",
        "domain_bounds": "half-open",
        "displacement": "second target minus first target, fixed micrometers",
        "accepted_window_count": len(accepted),
        "same_source_channel_group_count": len(groups),
        "overlap_pair_count": len(pairs),
        "pairs": pairs,
    }
    output = output.resolve()
    if output in input_paths:
        raise ValueError("diagnostic output must differ from its input artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return output
