#!/usr/bin/env python
"""Create a metadata-only joined position file for two 180-degree L/R views."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


DIMENSIONS = ("z", "y", "x")
DIMENSION_INDEX = {"z": 0, "y": 1, "x": 2}
LEFT_SIDE = "L"
RIGHT_SIDE = "R"


@dataclass(frozen=True)
class TileInfo:
    tile: str
    side: str
    path: Path
    shape_zyx: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    translation_zyx: tuple[float, float, float]


@dataclass(frozen=True)
class JoinedTile:
    info: TileInfo
    translation_zyx: tuple[float, float, float]
    scale_zyx: tuple[float, float, float]
    raw_bounds_zyx: tuple[tuple[float, float, float], tuple[float, float, float]]
    joined_bounds_zyx: tuple[tuple[float, float, float], tuple[float, float, float]]


def parse_args(
    *,
    default_left_dir: Path | None = None,
    default_right_dir: Path | None = None,
    default_output: Path | None = None,
    default_plot_title: str = "metadata joined tile positions",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", type=Path, default=default_left_dir, required=default_left_dir is None)
    parser.add_argument("--right-dir", type=Path, default=default_right_dir, required=default_right_dir is None)
    parser.add_argument("--output", type=Path, default=default_output, required=default_output is None)
    parser.add_argument("--plot-title", default=default_plot_title)
    parser.add_argument(
        "--join-axis",
        choices=("z", "x"),
        default="z",
        help="Axis used to place R next to L. z places R on the -z side; x places R on the +x side.",
    )
    parser.add_argument(
        "--overlap-fraction",
        "--z-overlap-fraction",
        dest="overlap_fraction",
        type=float,
        default=0.10,
        help="Initial L/R overlap as a fraction of the smaller view depth along --join-axis.",
    )
    parser.add_argument(
        "--right-flip-axes",
        nargs="*",
        choices=DIMENSIONS,
        help="Spatial axes to flip for R. Defaults to z x for --join-axis z, and no flips for --join-axis x.",
    )
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def dict_zyx(values: tuple[float, float, float]) -> dict[str, float]:
    return {dim: float(value) for dim, value in zip(DIMENSIONS, values, strict=True)}


def read_tile_info(path: Path, *, side: str) -> TileInfo:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        axes = str(series.axes)
        ome_xml = tif.ome_metadata
    if ome_xml is None:
        raise ValueError(f"{path} has no OME metadata")
    if axes == "CZYX":
        shape_zyx = (shape[1], shape[2], shape[3])
    elif axes == "ZYX":
        shape_zyx = shape
    else:
        raise ValueError(f"{path} has unsupported axes {axes!r}")

    pixels_attrs: dict[str, str] | None = None
    plane_attrs: dict[str, str] | None = None
    for _, elem in ET.iterparse(io.StringIO(ome_xml), events=("start",)):
        name = local_name(elem.tag)
        if name == "Pixels" and pixels_attrs is None:
            pixels_attrs = dict(elem.attrib)
        elif name == "Plane":
            plane_attrs = dict(elem.attrib)
            break
    if pixels_attrs is None or plane_attrs is None:
        raise ValueError(f"{path} OME metadata lacks Pixels or Plane metadata")

    return TileInfo(
        tile=path.name,
        side=side,
        path=path.resolve(),
        shape_zyx=shape_zyx,
        spacing_zyx=(
            float(pixels_attrs["PhysicalSizeZ"]),
            float(pixels_attrs["PhysicalSizeY"]),
            float(pixels_attrs["PhysicalSizeX"]),
        ),
        translation_zyx=(
            float(plane_attrs.get("PositionZ", 0.0)),
            float(plane_attrs["PositionY"]),
            float(plane_attrs["PositionX"]),
        ),
    )


def read_tiles(directory: Path, *, side: str) -> list[TileInfo]:
    paths = sorted(directory.glob("*.ome.tif"))
    if not paths:
        raise FileNotFoundError(f"No *.ome.tif files found in {directory}")
    return [read_tile_info(path, side=side) for path in paths]


def bounds_zyx(
    translation_zyx: tuple[float, float, float],
    scale_zyx: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    edge_a = np.asarray(translation_zyx, dtype=np.float64)
    edge_b = edge_a + np.asarray(shape_zyx, dtype=np.float64) * np.asarray(scale_zyx, dtype=np.float64)
    return tuple(np.minimum(edge_a, edge_b)), tuple(np.maximum(edge_a, edge_b))


def bounds_center(bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> np.ndarray:
    return (np.asarray(bounds[0], dtype=np.float64) + np.asarray(bounds[1], dtype=np.float64)) / 2.0


def compute_joined_tiles(
    left_tiles: list[TileInfo],
    right_tiles: list[TileInfo],
    *,
    z_overlap_fraction: float | None = 0.10,
    join_axis: str = "z",
    overlap_fraction: float | None = None,
    right_flip_axes: tuple[str, ...] | None = None,
) -> tuple[list[JoinedTile], dict[str, Any]]:
    if not left_tiles or not right_tiles:
        raise ValueError("Both left and right tile lists must be non-empty")
    if join_axis not in {"z", "x"}:
        raise ValueError("join_axis must be 'z' or 'x'")
    if right_flip_axes is None:
        right_flip_axes = ("z", "x") if join_axis == "z" else ()
    invalid_flip_axes = sorted(set(right_flip_axes) - set(DIMENSIONS))
    if invalid_flip_axes:
        raise ValueError(f"right_flip_axes contains invalid dimensions: {invalid_flip_axes}")
    if overlap_fraction is None:
        if z_overlap_fraction is None:
            raise ValueError("overlap_fraction must be provided")
        overlap_fraction = z_overlap_fraction
    if overlap_fraction < 0.0 or overlap_fraction >= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")

    left_raw_bounds = [bounds_zyx(tile.translation_zyx, tile.spacing_zyx, tile.shape_zyx) for tile in left_tiles]
    right_raw_bounds = [bounds_zyx(tile.translation_zyx, tile.spacing_zyx, tile.shape_zyx) for tile in right_tiles]
    left_centers = np.asarray([bounds_center(bounds) for bounds in left_raw_bounds])
    right_centers = np.asarray([bounds_center(bounds) for bounds in right_raw_bounds])

    left_min = np.min([bounds[0] for bounds in left_raw_bounds], axis=0)
    left_max = np.max([bounds[1] for bounds in left_raw_bounds], axis=0)
    right_min = np.min([bounds[0] for bounds in right_raw_bounds], axis=0)
    right_max = np.max([bounds[1] for bounds in right_raw_bounds], axis=0)
    left_centroid = left_centers.mean(axis=0)
    right_centroid = right_centers.mean(axis=0)
    join_index = DIMENSION_INDEX[join_axis]
    overlap_um = overlap_fraction * min(
        left_max[join_index] - left_min[join_index],
        right_max[join_index] - right_min[join_index],
    )

    right_scale_sign = np.asarray([-1.0 if dim in right_flip_axes else 1.0 for dim in DIMENSIONS])
    transformed_right_centroid = right_scale_sign * right_centroid
    transformed_right_min = np.minimum(right_scale_sign * right_min, right_scale_sign * right_max)
    transformed_right_max = np.maximum(right_scale_sign * right_min, right_scale_sign * right_max)

    right_offset_zyx = np.zeros(3, dtype=np.float64)
    for index, dim in enumerate(DIMENSIONS):
        if dim != join_axis:
            right_offset_zyx[index] = left_centroid[index] - transformed_right_centroid[index]
    if join_axis == "z":
        right_offset_zyx[0] = left_min[0] + overlap_um - transformed_right_max[0]
    else:
        right_offset_zyx[2] = left_max[2] - overlap_um - transformed_right_min[2]

    joined: list[JoinedTile] = []
    for tile, raw_bounds in zip(left_tiles, left_raw_bounds, strict=True):
        joined.append(
            JoinedTile(
                info=tile,
                translation_zyx=tile.translation_zyx,
                scale_zyx=tile.spacing_zyx,
                raw_bounds_zyx=raw_bounds,
                joined_bounds_zyx=raw_bounds,
            )
        )

    for tile, raw_bounds in zip(right_tiles, right_raw_bounds, strict=True):
        raw_translation = np.asarray(tile.translation_zyx, dtype=np.float64)
        spacing = np.asarray(tile.spacing_zyx, dtype=np.float64)
        translation_zyx = tuple(right_offset_zyx + right_scale_sign * raw_translation)
        scale_zyx = tuple(right_scale_sign * spacing)
        joined.append(
            JoinedTile(
                info=tile,
                translation_zyx=translation_zyx,
                scale_zyx=scale_zyx,
                raw_bounds_zyx=raw_bounds,
                joined_bounds_zyx=bounds_zyx(translation_zyx, scale_zyx, tile.shape_zyx),
            )
        )

    diagnostics = joined_diagnostics(
        joined,
        left_tiles=left_tiles,
        right_tiles=right_tiles,
        left_centroid=left_centroid,
        right_centroid=right_centroid,
        right_offset_zyx=right_offset_zyx,
        join_axis=join_axis,
        right_flip_axes=right_flip_axes,
        overlap_fraction=overlap_fraction,
        overlap_um=overlap_um,
    )
    return joined, diagnostics


def joined_diagnostics(
    joined: list[JoinedTile],
    *,
    left_tiles: list[TileInfo],
    right_tiles: list[TileInfo],
    left_centroid: np.ndarray,
    right_centroid: np.ndarray,
    right_offset_zyx: np.ndarray,
    join_axis: str,
    right_flip_axes: tuple[str, ...],
    overlap_fraction: float,
    overlap_um: float,
) -> dict[str, Any]:
    left_joined = [tile for tile in joined if tile.info.side == LEFT_SIDE]
    right_joined = [tile for tile in joined if tile.info.side == RIGHT_SIDE]
    left_centers = np.asarray([bounds_center(tile.joined_bounds_zyx) for tile in left_joined])
    right_centers = np.asarray([bounds_center(tile.joined_bounds_zyx) for tile in right_joined])
    left_min_z = float(min(tile.joined_bounds_zyx[0][0] for tile in left_joined))
    right_max_z = float(max(tile.joined_bounds_zyx[1][0] for tile in right_joined))
    left_max_x = float(max(tile.joined_bounds_zyx[1][2] for tile in left_joined))
    right_min_x = float(min(tile.joined_bounds_zyx[0][2] for tile in right_joined))
    centroid_axes = [dim for dim in DIMENSIONS if dim != join_axis]
    centroid_indices = [DIMENSION_INDEX[dim] for dim in centroid_axes]

    return {
        "left_tile_count": len(left_tiles),
        "right_tile_count": len(right_tiles),
        "join_axis": join_axis,
        "right_flip_axes": list(right_flip_axes),
        "centroid_alignment_axes": centroid_axes,
        "left_centroid_um": left_centroid.tolist(),
        "right_raw_centroid_um": right_centroid.tolist(),
        "right_joined_centroid_um": right_centers.mean(axis=0).tolist(),
        "left_alignment_centroid_um": left_centroid[centroid_indices].tolist(),
        "right_raw_alignment_centroid_um": right_centroid[centroid_indices].tolist(),
        "right_joined_alignment_centroid_um": right_centers[:, centroid_indices].mean(axis=0).tolist(),
        "left_xy_centroid_um": left_centroid[1:3].tolist(),
        "right_raw_xy_centroid_um": right_centroid[1:3].tolist(),
        "right_joined_xy_centroid_um": right_centers[:, 1:3].mean(axis=0).tolist(),
        "right_to_left_offset_um": right_offset_zyx.tolist(),
        "overlap_fraction": float(overlap_fraction),
        "overlap_um": float(overlap_um),
        "z_overlap_fraction": float(overlap_fraction) if join_axis == "z" else None,
        "z_overlap_um": float(overlap_um) if join_axis == "z" else None,
        "x_overlap_fraction": float(overlap_fraction) if join_axis == "x" else None,
        "x_overlap_um": float(overlap_um) if join_axis == "x" else None,
        "left_min_z_um": left_min_z,
        "right_joined_max_z_um": right_max_z,
        "z_adjacency_error_um": right_max_z - left_min_z,
        "left_max_x_um": left_max_x,
        "right_joined_min_x_um": right_min_x,
        "x_adjacency_error_um": left_max_x - right_min_x,
        "joined_min_zyx_um": np.min([tile.joined_bounds_zyx[0] for tile in joined], axis=0).tolist(),
        "joined_max_zyx_um": np.max([tile.joined_bounds_zyx[1] for tile in joined], axis=0).tolist(),
        "left_joined_xy_centroid_um": left_centers[:, 1:3].mean(axis=0).tolist(),
    }


def write_position_json(path: Path, joined: list[JoinedTile], diagnostics: dict[str, Any]) -> None:
    right_scale_sign = [-1.0 if dim in diagnostics["right_flip_axes"] else 1.0 for dim in DIMENSIONS]
    payload = {
        "schema_version": 1,
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
                "translation_um": dict_zyx(tile.translation_zyx),
                "scale_um": dict_zyx(tile.scale_zyx),
            }
            for tile in joined
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_positions_csv(path: Path, joined: list[JoinedTile]) -> None:
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tile",
                "side",
                "path",
                "translation_z_um",
                "translation_y_um",
                "translation_x_um",
                "scale_z_um",
                "scale_y_um",
                "scale_x_um",
                "joined_min_z_um",
                "joined_min_y_um",
                "joined_min_x_um",
                "joined_max_z_um",
                "joined_max_y_um",
                "joined_max_x_um",
            ]
        )
        for tile in joined:
            writer.writerow(
                [
                    tile.info.tile,
                    tile.info.side,
                    str(tile.info.path),
                    *tile.translation_zyx,
                    *tile.scale_zyx,
                    *tile.joined_bounds_zyx[0],
                    *tile.joined_bounds_zyx[1],
                ]
            )


def box_edges(bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> list[tuple[np.ndarray, np.ndarray]]:
    z0, y0, x0 = bounds[0]
    z1, y1, x1 = bounds[1]
    corners = [
        np.asarray(point, dtype=np.float64)
        for point in (
            (z0, y0, x0),
            (z0, y0, x1),
            (z0, y1, x0),
            (z0, y1, x1),
            (z1, y0, x0),
            (z1, y0, x1),
            (z1, y1, x0),
            (z1, y1, x1),
        )
    ]
    edge_indices = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7))
    return [(corners[start], corners[stop]) for start, stop in edge_indices]


def render_positions(output: Path, joined: list[JoinedTile], *, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {LEFT_SIDE: "#2f6fbb", RIGHT_SIDE: "#c44e52"}
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for tile in joined:
        color = colors[tile.info.side]
        for start, stop in box_edges(tile.joined_bounds_zyx):
            ax.plot([start[2], stop[2]], [start[1], stop[1]], [start[0], stop[0]], color=color, alpha=0.35)
        center = bounds_center(tile.joined_bounds_zyx)
        ax.scatter(center[2], center[1], center[0], color=color, s=12)
        ax.text(center[2], center[1], center[0], tile.info.tile.rsplit(".", 2)[-2], fontsize=6)
    ax.set_xlabel("x um")
    ax.set_ylabel("y um")
    ax.set_zlabel("z um")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".3d.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    projection_specs = [
        ("XY", 2, 1, "x um", "y um"),
        ("XZ", 2, 0, "x um", "z um"),
        ("YZ", 1, 0, "y um", "z um"),
    ]
    for ax2, (projection_title, axis_a, axis_b, label_a, label_b) in zip(axes, projection_specs, strict=True):
        for tile in joined:
            color = colors[tile.info.side]
            mins = tile.joined_bounds_zyx[0]
            maxs = tile.joined_bounds_zyx[1]
            rect_x = [mins[axis_a], maxs[axis_a], maxs[axis_a], mins[axis_a], mins[axis_a]]
            rect_y = [mins[axis_b], mins[axis_b], maxs[axis_b], maxs[axis_b], mins[axis_b]]
            ax2.plot(rect_x, rect_y, color=color, alpha=0.45)
            center = bounds_center(tile.joined_bounds_zyx)
            ax2.scatter(center[axis_a], center[axis_b], color=color, s=10)
        ax2.set_title(projection_title)
        ax2.set_xlabel(label_a)
        ax2.set_ylabel(label_b)
        ax2.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".projections.png"), dpi=200)
    plt.close(fig)


def main(
    *,
    default_left_dir: Path | None = None,
    default_right_dir: Path | None = None,
    default_output: Path | None = None,
    default_plot_title: str = "metadata joined tile positions",
) -> int:
    args = parse_args(
        default_left_dir=default_left_dir,
        default_right_dir=default_right_dir,
        default_output=default_output,
        default_plot_title=default_plot_title,
    )
    joined, diagnostics = compute_joined_tiles(
        read_tiles(args.left_dir.resolve(), side=LEFT_SIDE),
        read_tiles(args.right_dir.resolve(), side=RIGHT_SIDE),
        join_axis=args.join_axis,
        overlap_fraction=float(args.overlap_fraction),
        right_flip_axes=tuple(args.right_flip_axes) if args.right_flip_axes is not None else None,
    )
    output = args.output.resolve()
    write_position_json(output, joined, diagnostics)
    write_positions_csv(output, joined)
    render_positions(output, joined, title=args.plot_title)
    print(output)
    print(output.with_suffix(".csv"))
    print(output.with_suffix(".3d.png"))
    print(output.with_suffix(".projections.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
