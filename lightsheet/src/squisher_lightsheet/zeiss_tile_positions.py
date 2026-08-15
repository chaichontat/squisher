from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any
from xml.etree import ElementTree

from squisher_lightsheet.artifacts import stamp_artifact
from squisher_lightsheet.zeiss_positions import ZeissPosition, read_zeiss_positions


CACHE_ARTIFACT_TYPE = "squisher_lightsheet.ome_tile_metadata_cache.v1"


@dataclass(frozen=True, slots=True)
class OmeTilePosition:
    path: Path
    size_bytes: int
    mtime_ns: int
    output_index: int
    mosaic_index: int
    expected_tile_count: int
    shape_zyx: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    translation_zyx: tuple[float, float, float]
    czi_box_xywh: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ComputedTilePosition:
    output_index: int
    mosaic_index: int
    row: int
    column: int
    translation_zyx: tuple[float, float, float]


def read_ome_tile_positions(input_dir: Path, cache_path: Path) -> list[OmeTilePosition]:
    """Read OME tile placement metadata, reusing records for unchanged TIFFs."""
    paths = sorted(input_dir.glob("*.ome.tif"))
    if not paths:
        raise FileNotFoundError(f"No *.ome.tif files found in {input_dir}")

    cached = _read_metadata_cache(cache_path, input_dir)
    stats = {path.resolve(): path.stat() for path in paths}
    records = {
        path: record
        for path, record in cached.items()
        if path in stats
        and record.size_bytes == stats[path].st_size
        and record.mtime_ns == stats[path].st_mtime_ns
    }

    for path, initial_stat in stats.items():
        if path in records:
            continue
        record = _read_ome_tile_position(path, initial_stat.st_size, initial_stat.st_mtime_ns)
        final_stat = path.stat()
        if (final_stat.st_size, final_stat.st_mtime_ns) != (
            initial_stat.st_size,
            initial_stat.st_mtime_ns,
        ):
            raise ValueError(f"{path} changed while its OME metadata was being read; retry later")
        records[path] = record
        _write_metadata_cache(cache_path, input_dir, records.values())

    _write_metadata_cache(cache_path, input_dir, records.values())
    tiles = sorted(records.values(), key=lambda record: record.output_index)
    _validate_tile_set(tiles)
    return tiles


def derive_tile_positions_from_pos(
    positions: list[ZeissPosition],
    *,
    overlap_fraction: float = 0.2,
    min_hull_overlap_fraction: float = 0.02,
) -> tuple[list[ComputedTilePosition], dict[str, Any]]:
    """Generate a centered, convex-hull-clipped tile lattice from Zeiss rectangles."""
    if len(positions) % 4:
        raise ValueError(f"Expected Zeiss positions in groups of four, found {len(positions)}")
    if not 0 <= overlap_fraction < 1:
        raise ValueError(f"overlap_fraction must be in [0, 1), got {overlap_fraction}")
    if not 0 <= min_hull_overlap_fraction < 1:
        raise ValueError(
            "min_hull_overlap_fraction must be in [0, 1), "
            f"got {min_hull_overlap_fraction}"
        )

    groups: list[tuple[int, list[ZeissPosition], float, float]] = []
    for start in range(0, len(positions), 4):
        group = positions[start : start + 4]
        if (
            len({position.x for position in group}) != 2
            or len({position.y for position in group}) != 2
            or len({(position.x, position.y) for position in group}) != 4
        ):
            raise ValueError(f"Positions {start + 1}-{start + 4} do not form a rectangle")
        width = max(position.x for position in group) - min(position.x for position in group)
        height = max(position.y for position in group) - min(position.y for position in group)
        groups.append((start // 4 + 1, group, width, height))

    size_counts: dict[tuple[float, float], int] = {}
    for _, _, width, height in groups:
        size = (round(width, 3), round(height, 3))
        size_counts[size] = size_counts.get(size, 0) + 1
    modal_size = max(size_counts, key=lambda size: size_counts[size])
    matching_groups = [
        (index, group, width, height)
        for index, group, width, height in groups
        if (round(width, 3), round(height, 3)) == modal_size
    ]
    footprint_x = median(width for _, _, width, _ in matching_groups)
    footprint_y = median(height for _, _, _, height in matching_groups)

    matched_positions = [position for _, group, _, _ in matching_groups for position in group]
    z_values = [position.z for position in matched_positions]
    if max(z_values) - min(z_values) > 0.001:
        raise ValueError("Matching Zeiss position groups do not share one Z position")
    region_min_x = min(position.x for position in matched_positions)
    region_max_x = max(position.x for position in matched_positions)
    region_min_y = min(position.y for position in matched_positions)
    region_max_y = max(position.y for position in matched_positions)
    region_z = sum(z_values) / len(z_values)
    step_x = footprint_x * (1 - overlap_fraction)
    step_y = footprint_y * (1 - overlap_fraction)
    column_count = 1 + max(0, math.ceil((region_max_x - region_min_x - footprint_x) / step_x))
    row_count = 1 + max(0, math.ceil((region_max_y - region_min_y - footprint_y) / step_y))
    grid_width = footprint_x + (column_count - 1) * step_x
    grid_height = footprint_y + (row_count - 1) * step_y
    grid_min_x = (region_min_x + region_max_x - grid_width) / 2
    grid_min_y = (region_min_y + region_max_y - grid_height) / 2

    from scipy.spatial import ConvexHull

    points = [(position.x, position.y) for position in matched_positions]
    hull = ConvexHull(points)
    hull_polygon = [points[index] for index in hull.vertices]
    candidates = []
    for row in range(row_count):
        y = grid_min_y + row * step_y
        for column in range(column_count):
            x = grid_min_x + column * step_x
            overlap_area = _rectangle_polygon_intersection_area(
                hull_polygon,
                min_x=x,
                max_x=x + footprint_x,
                min_y=y,
                max_y=y + footprint_y,
            )
            if overlap_area / (footprint_x * footprint_y) > min_hull_overlap_fraction:
                candidates.append((row, column, x, y))

    mosaic_indices = {}
    for acquisition_row, row in enumerate(range(row_count - 1, -1, -1)):
        row_tiles = [candidate for candidate in candidates if candidate[0] == row]
        row_tiles.sort(key=lambda candidate: candidate[2], reverse=acquisition_row % 2 == 0)
        for candidate in row_tiles:
            mosaic_indices[(candidate[0], candidate[1])] = len(mosaic_indices)

    candidates.sort(key=lambda candidate: (-candidate[3], -candidate[2]))
    derived = [
        ComputedTilePosition(
            output_index=output_index,
            mosaic_index=mosaic_indices[(row, column)],
            row=row,
            column=column,
            translation_zyx=(region_z, y, x),
        )
        for output_index, (row, column, x, y) in enumerate(candidates)
    ]
    diagnostics = {
        "matching_position_groups": [index for index, _, _, _ in matching_groups],
        "tile_count": len(derived),
        "tile_footprint_yx_um": [footprint_y, footprint_x],
        "overlap_fraction": overlap_fraction,
        "min_hull_overlap_fraction": min_hull_overlap_fraction,
        "grid_shape_yx": [row_count, column_count],
        "grid_step_yx_um": [step_y, step_x],
        "grid_padding_yx_um": [
            (grid_height - (region_max_y - region_min_y)) / 2,
            (grid_width - (region_max_x - region_min_x)) / 2,
        ],
        "row_tile_counts_bottom_to_top": [
            sum(tile.row == row for tile in derived) for row in range(row_count)
        ],
    }
    return derived, diagnostics


def create_zeiss_tile_position_file(
    *,
    pos_input: Path,
    output: Path,
    side: str,
    overlap_fraction: float = 0.2,
    min_hull_overlap_fraction: float = 0.02,
) -> Path:
    tiles, diagnostics = derive_tile_positions_from_pos(
        read_zeiss_positions(pos_input),
        overlap_fraction=overlap_fraction,
        min_hull_overlap_fraction=min_hull_overlap_fraction,
    )
    payload = stamp_artifact(
        {
            "units": "micrometer",
            "source": "Zeiss position rectangles",
            "position_input": str(pos_input.resolve()),
            "diagnostics": diagnostics,
            "tiles": [
                {
                    "tile": f"tile.{tile.output_index:03d}",
                    "output_index": tile.output_index,
                    "mosaic_index": tile.mosaic_index,
                    "grid_index_yx": [tile.row, tile.column],
                    "side": side,
                    "translation_um": dict(zip(("z", "y", "x"), tile.translation_zyx, strict=True)),
                }
                for tile in tiles
            ],
        },
        "squisher_lightsheet.zeiss_tile_grid.v1",
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def _rectangle_polygon_intersection_area(
    polygon: list[tuple[float, float]],
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> float:
    clipped = polygon
    for axis, bound, keep_greater in (
        (0, min_x, True),
        (0, max_x, False),
        (1, min_y, True),
        (1, max_y, False),
    ):
        clipped = _clip_polygon(clipped, axis=axis, bound=bound, keep_greater=keep_greater)
        if not clipped:
            return 0.0
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(clipped, [*clipped[1:], clipped[0]], strict=True)
        )
    ) / 2


def _clip_polygon(
    polygon: list[tuple[float, float]],
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    clipped = []
    previous = polygon[-1]
    previous_inside = (previous[axis] >= bound) == keep_greater
    for current in polygon:
        current_inside = (current[axis] >= bound) == keep_greater
        if current_inside != previous_inside:
            fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
            intersection = list(previous)
            intersection[axis] = bound
            intersection[1 - axis] += fraction * (current[1 - axis] - previous[1 - axis])
            clipped.append((intersection[0], intersection[1]))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return clipped


def _read_ome_tile_position(path: Path, size_bytes: int, mtime_ns: int) -> OmeTilePosition:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        ome_xml = tif.ome_metadata
    if ome_xml is None:
        raise ValueError(f"{path} has no OME metadata")
    root = ElementTree.fromstring(ome_xml)
    pixels = next((element for element in root.iter() if _local_name(element.tag) == "Pixels"), None)
    plane = next((element for element in root.iter() if _local_name(element.tag) == "Plane"), None)
    if pixels is None or plane is None:
        raise ValueError(f"{path} OME metadata lacks Pixels or Plane metadata")
    map_values = {
        element.attrib["K"]: element.text or ""
        for element in root.iter()
        if _local_name(element.tag) == "M" and "K" in element.attrib
    }

    def required_int(key: str) -> int:
        if key not in map_values:
            raise ValueError(f"{path} OME metadata lacks {key}")
        return int(map_values[key])

    return OmeTilePosition(
        path=path.resolve(),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        output_index=required_int("squisher.output_tile_index"),
        mosaic_index=required_int("czi.mosaic_index"),
        expected_tile_count=required_int("squisher.tile_count"),
        shape_zyx=(int(pixels.attrib["SizeZ"]), int(pixels.attrib["SizeY"]), int(pixels.attrib["SizeX"])),
        spacing_zyx=(
            float(pixels.attrib["PhysicalSizeZ"]),
            float(pixels.attrib["PhysicalSizeY"]),
            float(pixels.attrib["PhysicalSizeX"]),
        ),
        translation_zyx=(
            float(plane.attrib.get("PositionZ", 0.0)),
            float(plane.attrib["PositionY"]),
            float(plane.attrib["PositionX"]),
        ),
        czi_box_xywh=(
            required_int("czi.tile_x"),
            required_int("czi.tile_y"),
            required_int("czi.tile_width"),
            required_int("czi.tile_height"),
        ),
    )


def _validate_tile_set(tiles: list[OmeTilePosition]) -> None:
    if not tiles:
        raise ValueError("OME tile metadata is empty")
    expected_counts = {tile.expected_tile_count for tile in tiles}
    if len(expected_counts) != 1:
        raise ValueError(f"OME tiles disagree on expected tile count: {sorted(expected_counts)}")
    expected_count = next(iter(expected_counts))
    if len(tiles) != expected_count:
        raise ValueError(f"Expected {expected_count} OME tiles, found {len(tiles)}")
    indices = sorted(tile.output_index for tile in tiles)
    if indices != list(range(expected_count)):
        raise ValueError(f"OME output tile indices must be 0 through {expected_count - 1}; found {indices}")
    shapes = {tile.shape_zyx for tile in tiles}
    spacings = {tile.spacing_zyx for tile in tiles}
    if len(shapes) != 1 or len(spacings) != 1:
        raise ValueError("OME tiles do not share one shape and physical spacing")
    reference = tiles[0]
    for tile in tiles:
        czi_x, czi_y, czi_width, czi_height = tile.czi_box_xywh
        if (czi_width, czi_height) != (tile.shape_zyx[2], tile.shape_zyx[1]):
            raise ValueError(f"{tile.path} CZI box does not match its OME XY shape")
        expected_x = reference.translation_zyx[2] + (
            czi_x - reference.czi_box_xywh[0]
        ) * tile.spacing_zyx[2]
        expected_y = reference.translation_zyx[1] + (
            czi_y - reference.czi_box_xywh[1]
        ) * tile.spacing_zyx[1]
        if not math.isclose(tile.translation_zyx[2], expected_x, abs_tol=1e-6) or not math.isclose(
            tile.translation_zyx[1], expected_y, abs_tol=1e-6
        ):
            raise ValueError(f"{tile.path} OME position disagrees with its CZI mosaic box")


def _read_metadata_cache(cache_path: Path, input_dir: Path) -> dict[Path, OmeTilePosition]:
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text())
    if payload.get("artifact_type") != CACHE_ARTIFACT_TYPE:
        raise ValueError(f"{cache_path} is not a {CACHE_ARTIFACT_TYPE} cache")
    if payload.get("input_dir") != str(input_dir.resolve()):
        raise ValueError(f"{cache_path} belongs to a different OME-TIFF directory")
    records = {}
    for item in payload["tiles"]:
        record = OmeTilePosition(
            path=Path(item["path"]),
            size_bytes=int(item["size_bytes"]),
            mtime_ns=int(item["mtime_ns"]),
            output_index=int(item["output_index"]),
            mosaic_index=int(item["mosaic_index"]),
            expected_tile_count=int(item["expected_tile_count"]),
            shape_zyx=tuple(item["shape_zyx"]),
            spacing_zyx=tuple(item["spacing_zyx"]),
            translation_zyx=tuple(item["translation_zyx"]),
            czi_box_xywh=tuple(item["czi_box_xywh"]),
        )
        records[record.path] = record
    return records


def _write_metadata_cache(
    cache_path: Path,
    input_dir: Path,
    records: Iterable[OmeTilePosition],
) -> None:
    cache_path = cache_path.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": CACHE_ARTIFACT_TYPE,
        "input_dir": str(input_dir.resolve()),
        "tiles": [
            {**asdict(record), "path": str(record.path)}
            for record in sorted(records, key=lambda item: item.output_index)
        ],
    }
    temporary = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(cache_path)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
