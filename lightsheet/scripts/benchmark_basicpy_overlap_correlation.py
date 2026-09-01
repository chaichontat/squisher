#!/usr/bin/env python
"""Benchmark BaSiC profiles by registered tile-overlap correlation.

The score follows the metric used in the BaSiC paper: after tile registration,
overlapping regions from neighboring tiles should have higher correlation when
illumination correction is better. This script reads this project's registration
JSON format, samples bounded registered-space patches in pairwise tile overlaps,
and reports Pearson correlations for raw images and optional BaSiC profile cases.
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
from scipy import ndimage as scipy_ndimage
from squisher_lightsheet import ngff


DIMENSIONS = ("z", "y", "x")
DEFAULT_PATCH_SHAPE_ZYX = (16, 256, 256)
RAW_CASE = "raw"


@dataclass(frozen=True)
class Tile:
    index: int
    name: str
    record: dict[str, Any]
    path: Path
    shape_zyx: np.ndarray
    spacing_um_zyx: np.ndarray
    stage_translation_um_zyx: np.ndarray
    registered_affine: np.ndarray
    bbox_min_um_zyx: np.ndarray
    bbox_max_um_zyx: np.ndarray
    source_view: str | None


@dataclass(frozen=True)
class BasicProfile:
    flatfield: np.ndarray
    darkfield: np.ndarray | None
    pre_scale: float
    flatfield_path: Path
    darkfield_path: Path | None
    scale_path: Path | None


@dataclass
class ImageLevelAccessor:
    axes: str
    level: int
    shape: tuple[int, ...]
    array: Any
    store: Any | None = None

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()


class ImageReaderCache:
    def __init__(self) -> None:
        self._accessors: dict[tuple[Path, int], ImageLevelAccessor] = {}

    def close(self) -> None:
        seen: set[int] = set()
        for accessor in self._accessors.values():
            if id(accessor) in seen:
                continue
            seen.add(id(accessor))
            accessor.close()
        self._accessors.clear()

    def metadata(self, path: Path, *, level: int) -> tuple[str, int, tuple[int, ...]]:
        accessor = self._accessor(path, level=level)
        return accessor.axes, accessor.level, accessor.shape

    def read_crop(
        self,
        path: Path,
        *,
        level: int,
        channel: int,
        slices_zyx: tuple[slice, slice, slice],
    ) -> np.ndarray:
        accessor = self._accessor(path, level=level)
        if accessor.axes == "CZYX":
            crop = accessor.array[(channel, *slices_zyx)]
        elif accessor.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} requested for single-channel tile {path}")
            crop = accessor.array[slices_zyx]
        else:
            raise ValueError(f"Expected CZYX or ZYX axes for {path}, got {accessor.axes!r}")
        return np.asarray(crop, dtype=np.float32)

    def _accessor(self, path: Path, *, level: int) -> ImageLevelAccessor:
        key = (path.resolve(), int(level))
        if key not in self._accessors:
            accessor = open_image_level_accessor(path, level=level)
            self._accessors[key] = accessor
            self._accessors.setdefault((path.resolve(), accessor.level), accessor)
        return self._accessors[key]


def parse_shape_zyx(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("expected positive Z,Y,X shape, for example 16,256,256")
    return parts


def parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=DIR")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("case name must be non-empty")
    return name, Path(path).expanduser()


def parse_case_source_view(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected NAME:SOURCE_VIEW=DIR")
    label_view, path = value.split("=", 1)
    label, source_view = (part.strip() for part in label_view.split(":", 1))
    if not label or not source_view:
        raise argparse.ArgumentTypeError("case name and source_view must be non-empty")
    return label, source_view, Path(path).expanduser()


def parse_source_view_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected VIEW=DIR")
    source_view, path = (part.strip() for part in value.split("=", 1))
    if not source_view:
        raise argparse.ArgumentTypeError("source_view must be non-empty")
    return source_view, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration-json", type=Path, required=True, help="Registration JSON in this repo's format.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output benchmark JSON.")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--level", type=int, default=2, help="Input pyramid level used for sampling.")
    parser.add_argument("--patch-shape-zyx", type=parse_shape_zyx, default=DEFAULT_PATCH_SHAPE_ZYX)
    parser.add_argument("--patches-per-pair", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, help="Optional cap after overlap-neighbor filtering.")
    parser.add_argument(
        "--edge-sample-count",
        type=int,
        help="Randomly sample this many candidate overlap edges with --sample-seed before patch sampling.",
    )
    parser.add_argument("--min-overlap-px-zyx", type=parse_shape_zyx, default=(1, 32, 32))
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--image-path-key",
        choices=("path", "source_path"),
        default="path",
        help="Tile-record path key used for image reads. Use source_path to benchmark raw TIFFs.",
    )
    parser.add_argument(
        "--image-source-view-root",
        type=parse_source_view_root,
        action="append",
        default=[],
        metavar="VIEW=DIR",
        help="Map tile source_view to a raw image directory. Tile names ending in .ome.zarr map to .ome.tif.",
    )
    parser.add_argument(
        "--case",
        type=parse_case,
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Named pooled BaSiC profile directory containing *-chN-flatfield.tif.",
    )
    parser.add_argument(
        "--case-source-view",
        type=parse_case_source_view,
        action="append",
        default=[],
        metavar="NAME:VIEW=DIR",
        help="Named source-view-specific BaSiC profile directory. Repeat for each VIEW in a case.",
    )
    parser.add_argument("--flatfield-dir", type=Path, help="Alias for --case basic=DIR.")
    parser.add_argument(
        "--flatfield-dir-by-source-view",
        type=parse_case_source_view,
        action="append",
        default=[],
        metavar="basic:VIEW=DIR",
        help="Alias for --case-source-view; include the case label, usually basic.",
    )
    parser.add_argument("--min-valid-voxels", type=int, default=4096)
    parser.add_argument("--min-positive-fraction", type=float, default=1e-4)
    args = parser.parse_args()
    if args.level < 0:
        raise ValueError("--level must be non-negative")
    if args.channel < 0:
        raise ValueError("--channel must be non-negative")
    if args.patches_per_pair <= 0:
        raise ValueError("--patches-per-pair must be positive")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive when provided")
    if args.edge_sample_count is not None and args.edge_sample_count <= 0:
        raise ValueError("--edge-sample-count must be positive when provided")
    if args.min_valid_voxels <= 1:
        raise ValueError("--min-valid-voxels must be greater than 1")
    if not 0 <= args.min_positive_fraction <= 1:
        raise ValueError("--min-positive-fraction must be in [0, 1]")
    return args


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def zyx(record: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        values = record.get(key)
        if isinstance(values, dict):
            return np.asarray([float(values[dim]) for dim in DIMENSIONS], dtype=np.float64)
    raise ValueError(f"Tile {record.get('tile')!r} is missing one of {keys}")


def tile_shape_zyx(record: dict[str, Any]) -> np.ndarray:
    shape = record.get("shape")
    if not isinstance(shape, list | tuple) or len(shape) < 3:
        raise ValueError(f"Tile {record.get('tile')!r} is missing shape")
    return np.asarray(shape[-3:], dtype=np.int64)


def registered_affine(record: dict[str, Any]) -> np.ndarray:
    affine = record.get("registered_affine")
    if affine is None:
        return np.eye(4, dtype=np.float64)
    matrix = affine.get("matrix") if isinstance(affine, dict) else affine
    array = np.asarray(matrix, dtype=np.float64)
    while array.ndim > 2:
        array = array[0]
    if array.shape[0] < 4 or array.shape[1] < 4:
        raise ValueError(f"Tile {record.get('tile')!r} registered_affine must be at least 4x4")
    return array[:4, :4]


def source_view_root_path(record: dict[str, Any], *, source_view_roots: dict[str, Path]) -> Path | None:
    source_view = record.get("source_view")
    if source_view is None:
        return None
    root = source_view_roots.get(str(source_view))
    if root is None:
        return None
    tile_name = Path(str(record.get("tile") or record.get("path") or "")).name
    if tile_name.endswith(".ome.zarr"):
        tile_name = f"{tile_name[:-len('.ome.zarr')]}.ome.tif"
    candidate = root / tile_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Mapped source_view {source_view!r} tile {record.get('tile')!r} to {candidate}, but it does not exist"
    )


def resolve_tile_path(
    registration_payload: dict[str, Any],
    record: dict[str, Any],
    *,
    image_path_key: str,
    source_view_roots: dict[str, Path],
) -> Path:
    source_path = source_view_root_path(record, source_view_roots=source_view_roots)
    if source_path is not None:
        return source_path

    keys = (image_path_key, "path", "tile") if image_path_key != "path" else ("path", "tile")
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        path = Path(str(value)).expanduser()
        if path.is_absolute() and path.exists():
            return path

    input_dir = Path(str(registration_payload.get("input_dir", "."))).expanduser()
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        candidate = input_dir / str(value)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path for tile {record.get('tile')!r}")


def tile_registered_bbox(
    *,
    shape_zyx: np.ndarray,
    spacing_um_zyx: np.ndarray,
    stage_translation_um_zyx: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corners_px = np.array(
        [
            [z, y, x]
            for z in (0.0, float(shape_zyx[0]))
            for y in (0.0, float(shape_zyx[1]))
            for x in (0.0, float(shape_zyx[2]))
        ],
        dtype=np.float64,
    )
    stage_um = stage_translation_um_zyx[None, :] + corners_px * spacing_um_zyx[None, :]
    world_um = (affine @ np.column_stack([stage_um, np.ones(stage_um.shape[0])]).T)[:3].T
    return np.min(world_um, axis=0), np.max(world_um, axis=0)


def load_tiles(
    registration_payload: dict[str, Any],
    *,
    image_path_key: str,
    source_view_roots: dict[str, Path],
) -> list[Tile]:
    tiles = []
    for index, record in enumerate(registration_payload.get("tiles", [])):
        shape = tile_shape_zyx(record)
        spacing = np.abs(zyx(record, "spacing_um", "stage_scale_um", "scale_um"))
        stage = zyx(record, "stage_translation_um", "translation_um")
        affine = registered_affine(record)
        bbox_min, bbox_max = tile_registered_bbox(
            shape_zyx=shape,
            spacing_um_zyx=spacing,
            stage_translation_um_zyx=stage,
            affine=affine,
        )
        tiles.append(
            Tile(
                index=index,
                name=str(record.get("tile", index)),
                record=record,
                path=resolve_tile_path(
                    registration_payload,
                    record,
                    image_path_key=image_path_key,
                    source_view_roots=source_view_roots,
                ),
                shape_zyx=shape,
                spacing_um_zyx=spacing,
                stage_translation_um_zyx=stage,
                registered_affine=affine,
                bbox_min_um_zyx=bbox_min,
                bbox_max_um_zyx=bbox_max,
                source_view=None if record.get("source_view") is None else str(record["source_view"]),
            )
        )
    if not tiles:
        raise ValueError("registration JSON contains no tiles")
    return tiles


def zarr_level_path(root: Any, *, level: int, path: Path) -> str:
    return ngff.level_path(root, level=level, context=path)


def zarr_level_metadata(path: Path, *, level: int) -> tuple[str, int, tuple[int, ...]]:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    array = root[zarr_level_path(root, level=level, path=path)]
    return ngff.axes(root, array), int(level), tuple(int(value) for value in array.shape)


def tiff_level_metadata(path: Path, *, level: int) -> tuple[str, int, tuple[int, ...]]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if level >= len(series.levels):
            raise ValueError(f"{path} has {len(series.levels)} TIFF pyramid level(s); requested level {level}")
        page_series = series.levels[level]
        return str(page_series.axes), int(level), tuple(int(value) for value in page_series.shape)


def open_image_level_accessor(path: Path, *, level: int) -> ImageLevelAccessor:
    if path.name.endswith(".zarr"):
        import zarr

        axes, source_level, shape = zarr_level_metadata(path, level=level)
        root = zarr.open_group(str(path), mode="r")
        array = root[zarr_level_path(root, level=source_level, path=path)]
        return ImageLevelAccessor(axes=axes, level=source_level, shape=shape, array=array)

    import tifffile
    import zarr

    axes, source_level, shape = tiff_level_metadata(path, level=level)
    store = tifffile.imread(path, aszarr=True, level=source_level)
    array = zarr.open(store, mode="r")
    if hasattr(array, "keys") and "0" in array:
        array = array["0"]
    return ImageLevelAccessor(axes=axes, level=source_level, shape=shape, array=array, store=store)


def spatial_shape_zyx(*, axes: str, shape: tuple[int, ...]) -> np.ndarray:
    if axes == "CZYX":
        return np.asarray(shape[1:4], dtype=np.int64)
    if axes == "ZYX":
        return np.asarray(shape, dtype=np.int64)
    raise ValueError(f"Expected CZYX or ZYX axes, got {axes!r}")


def find_profile_path(flatfield_dir: Path, channel: int, suffix: str) -> Path | None:
    matches = sorted(flatfield_dir.glob(f"*-ch{channel}-{suffix}.tif"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one ch{channel} {suffix} TIFF in {flatfield_dir}, found {matches}")
    return matches[0]


def find_scale_path(flatfield_dir: Path, channel: int) -> Path | None:
    matches = sorted(flatfield_dir.glob(f"*-ch{channel}-*-corrected-max.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one ch{channel} corrected-max JSON in {flatfield_dir}, found {matches}")
    return matches[0]


def load_basic_profile(flatfield_dir: Path, channel: int) -> BasicProfile:
    import tifffile

    flatfield_path = find_profile_path(flatfield_dir, channel, "flatfield")
    if flatfield_path is None:
        raise FileNotFoundError(f"No *-ch{channel}-flatfield.tif found in {flatfield_dir}")
    darkfield_path = find_profile_path(flatfield_dir, channel, "darkfield")
    scale_path = find_scale_path(flatfield_dir, channel)
    pre_scale = 1.0
    if scale_path is not None:
        pre_scale = float(json.loads(scale_path.read_text())["pre_scale_for_uint16"])
        if not math.isfinite(pre_scale) or pre_scale <= 0:
            raise ValueError(f"{scale_path} has invalid pre_scale_for_uint16={pre_scale}")
    flatfield = np.asarray(tifffile.imread(flatfield_path), dtype=np.float32)
    darkfield = None if darkfield_path is None else np.asarray(tifffile.imread(darkfield_path), dtype=np.float32)
    if flatfield.ndim != 2:
        raise ValueError(f"{flatfield_path} must be 2D, got shape {flatfield.shape}")
    if darkfield is not None and darkfield.shape != flatfield.shape:
        raise ValueError(f"{darkfield_path} shape {darkfield.shape} does not match flatfield {flatfield.shape}")
    if not np.all(np.isfinite(flatfield)) or np.any(flatfield <= 0):
        raise ValueError(f"{flatfield_path} must contain positive finite values")
    return BasicProfile(
        flatfield=flatfield,
        darkfield=darkfield,
        pre_scale=pre_scale,
        flatfield_path=flatfield_path,
        darkfield_path=darkfield_path,
        scale_path=scale_path,
    )


def block_mean_resize_2d(image: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    if image.shape == expected_shape:
        return image.astype(np.float32, copy=False)
    src_y, src_x = image.shape
    dst_y, dst_x = expected_shape
    if dst_y <= 0 or dst_x <= 0 or dst_y > src_y or dst_x > src_x:
        raise ValueError(f"Cannot resize profile shape {image.shape} to {expected_shape}")
    factor_y = math.ceil(src_y / dst_y)
    factor_x = math.ceil(src_x / dst_x)
    if (src_y + factor_y - 1) // factor_y != dst_y or (src_x + factor_x - 1) // factor_x != dst_x:
        raise ValueError(f"Cannot block-downsample profile shape {image.shape} to {expected_shape}")
    y_starts = np.arange(0, src_y, factor_y)
    x_starts = np.arange(0, src_x, factor_x)
    y_counts = np.diff(np.append(y_starts, src_y)).astype(np.float32)
    x_counts = np.diff(np.append(x_starts, src_x)).astype(np.float32)
    y_sums = np.add.reduceat(image.astype(np.float32, copy=False), y_starts, axis=0)
    xy_sums = np.add.reduceat(y_sums, x_starts, axis=1)
    return xy_sums / y_counts[:, None] / x_counts[None, :]


def build_basic_cases(args: argparse.Namespace, channel: int) -> dict[str, dict[str, BasicProfile]]:
    case_dirs: dict[str, dict[str, Path]] = {}
    if args.flatfield_dir is not None:
        case_dirs.setdefault("basic", {})["__default__"] = args.flatfield_dir.expanduser()
    for name, path in args.case:
        case_dirs.setdefault(name, {})["__default__"] = path
    for name, view, path in [*args.case_source_view, *args.flatfield_dir_by_source_view]:
        case_dirs.setdefault(name, {})[view] = path
    return {
        name: {view: load_basic_profile(path, channel) for view, path in by_view.items()}
        for name, by_view in case_dirs.items()
    }


def select_basic_profile(case_profiles: dict[str, BasicProfile], tile: Tile) -> BasicProfile:
    if tile.source_view is not None and tile.source_view in case_profiles:
        return case_profiles[tile.source_view]
    if "__default__" in case_profiles:
        return case_profiles["__default__"]
    raise ValueError(f"No BaSiC profile for tile {tile.name} source_view={tile.source_view!r}")


def apply_basic_to_crop(
    crop: np.ndarray,
    *,
    profile: BasicProfile,
    crop_slices_zyx: tuple[slice, slice, slice],
    level_shape_yx: tuple[int, int],
) -> np.ndarray:
    flatfield = block_mean_resize_2d(profile.flatfield, level_shape_yx)
    y_slice, x_slice = crop_slices_zyx[1], crop_slices_zyx[2]
    flat_crop = flatfield[y_slice, x_slice]
    corrected = crop.astype(np.float32, copy=False)
    if profile.darkfield is not None:
        darkfield = block_mean_resize_2d(profile.darkfield, level_shape_yx)
        corrected = corrected - darkfield[y_slice, x_slice][None, :, :]
    return corrected / (profile.pre_scale * flat_crop[None, :, :])


def local_coords_for_world(tile: Tile, world_um_zyx: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([world_um_zyx, np.ones(world_um_zyx.shape[0], dtype=np.float64)]).T
    local_input_um = (np.linalg.inv(tile.registered_affine) @ homogeneous)[:3].T
    local_um = local_input_um - tile.stage_translation_um_zyx[None, :]
    return local_um / tile.spacing_um_zyx[None, :]


def sample_tile_patch(
    *,
    registration_payload: dict[str, Any],
    tile: Tile,
    world_points_um_zyx: np.ndarray,
    output_shape_zyx: tuple[int, int, int],
    level: int,
    channel: int,
    reader_cache: ImageReaderCache,
    profile: BasicProfile | None,
) -> tuple[np.ndarray, np.ndarray]:
    del registration_payload
    axes, source_level, source_shape = reader_cache.metadata(tile.path, level=level)
    source_shape_zyx = spatial_shape_zyx(axes=axes, shape=source_shape)
    level_spacing = tile.spacing_um_zyx.copy()
    level_spacing[1:] *= 2**int(source_level)

    coords = local_coords_for_world(tile, world_points_um_zyx)
    source_coords = coords.copy()
    source_coords[:, 1] = coords[:, 1] / (2**int(source_level))
    source_coords[:, 2] = coords[:, 2] / (2**int(source_level))
    inside = np.all((source_coords >= 0.0) & (source_coords <= (source_shape_zyx - 1)[None, :]), axis=1)
    if not np.any(inside):
        return np.zeros(output_shape_zyx, dtype=np.float32), inside.reshape(output_shape_zyx)

    lo = np.floor(np.nanmin(source_coords[inside], axis=0)).astype(int) - 2
    hi = np.ceil(np.nanmax(source_coords[inside], axis=0)).astype(int) + 3
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, source_shape_zyx)
    if np.any(hi <= lo):
        return np.zeros(output_shape_zyx, dtype=np.float32), np.zeros(output_shape_zyx, dtype=bool)

    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lo, hi, strict=True))
    source = reader_cache.read_crop(tile.path, level=source_level, channel=channel, slices_zyx=slices)
    if profile is not None:
        source = apply_basic_to_crop(
            source,
            profile=profile,
            crop_slices_zyx=slices,
            level_shape_yx=(int(source_shape_zyx[1]), int(source_shape_zyx[2])),
        )

    crop_coords = source_coords - lo[None, :]
    sampled = scipy_ndimage.map_coordinates(
        source,
        [crop_coords[:, 0], crop_coords[:, 1], crop_coords[:, 2]],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    sampled[~inside] = 0.0
    return sampled.reshape(output_shape_zyx).astype(np.float32, copy=False), inside.reshape(output_shape_zyx)


def overlap_bounds(tile_a: Tile, tile_b: Tile) -> tuple[np.ndarray, np.ndarray] | None:
    start = np.maximum(tile_a.bbox_min_um_zyx, tile_b.bbox_min_um_zyx)
    stop = np.minimum(tile_a.bbox_max_um_zyx, tile_b.bbox_max_um_zyx)
    if np.any(stop <= start):
        return None
    return start, stop


def candidate_pairs(
    tiles: list[Tile],
    *,
    min_overlap_px_zyx: tuple[int, int, int],
    max_pairs: int | None,
    edge_sample_count: int | None,
    rng: np.random.Generator,
) -> list[tuple[Tile, Tile, np.ndarray, np.ndarray, np.ndarray]]:
    pairs = []
    for i, tile_a in enumerate(tiles):
        for tile_b in tiles[i + 1 :]:
            bounds = overlap_bounds(tile_a, tile_b)
            if bounds is None:
                continue
            start, stop = bounds
            overlap_um = stop - start
            overlap_px = np.floor(overlap_um / tile_a.spacing_um_zyx + 1e-6).astype(np.int64)
            if np.any(overlap_px < np.asarray(min_overlap_px_zyx, dtype=np.int64)):
                continue
            pairs.append((tile_a, tile_b, start, stop, overlap_px))
    pairs.sort(key=lambda item: (item[0].name, item[1].name))
    if edge_sample_count is not None and len(pairs) > edge_sample_count:
        selected = rng.choice(len(pairs), size=edge_sample_count, replace=False)
        pairs = [pairs[int(index)] for index in selected]
    return pairs if max_pairs is None else pairs[:max_pairs]


def patch_centers(
    start_um_zyx: np.ndarray,
    stop_um_zyx: np.ndarray,
    *,
    spacing_um_zyx: np.ndarray,
    patch_shape_zyx: tuple[int, int, int],
    count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    patch_um = np.asarray(patch_shape_zyx, dtype=np.float64) * spacing_um_zyx
    low = start_um_zyx + patch_um / 2.0
    high = stop_um_zyx - patch_um / 2.0
    centers = [(start_um_zyx + stop_um_zyx) / 2.0]
    if count == 1:
        return centers
    for _ in range(count - 1):
        center = np.empty(3, dtype=np.float64)
        for axis in range(3):
            if high[axis] <= low[axis]:
                center[axis] = (start_um_zyx[axis] + stop_um_zyx[axis]) / 2.0
            else:
                center[axis] = rng.uniform(low[axis], high[axis])
        centers.append(center)
    return centers


def world_grid_for_patch(
    center_um_zyx: np.ndarray,
    *,
    spacing_um_zyx: np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    axes = [
        center_um_zyx[axis] + (np.arange(shape_zyx[axis], dtype=np.float64) - shape_zyx[axis] // 2) * spacing_um_zyx[axis]
        for axis in range(3)
    ]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])


def pearson(values_a: np.ndarray, values_b: np.ndarray, mask: np.ndarray) -> float:
    a = values_a[mask].astype(np.float64, copy=False)
    b = values_b[mask].astype(np.float64, copy=False)
    if a.size < 2:
        return float("nan")
    a -= np.mean(a)
    b -= np.mean(b)
    denom = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def summarize(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    registration_payload = read_json(args.registration_json)
    rng = np.random.default_rng(args.sample_seed)
    source_view_roots = {view: path for view, path in args.image_source_view_root}
    tiles = load_tiles(
        registration_payload,
        image_path_key=args.image_path_key,
        source_view_roots=source_view_roots,
    )
    pairs = candidate_pairs(
        tiles,
        min_overlap_px_zyx=args.min_overlap_px_zyx,
        max_pairs=args.max_pairs,
        edge_sample_count=args.edge_sample_count,
        rng=rng,
    )
    if not pairs:
        raise ValueError("No overlapping tile pairs passed the overlap filters")

    basic_cases = build_basic_cases(args, args.channel)
    case_names = [RAW_CASE, *sorted(basic_cases)]
    reader_cache = ImageReaderCache()
    patch_rows: list[dict[str, Any]] = []
    correlations: dict[str, list[float]] = {name: [] for name in case_names}
    try:
        for pair_index, (tile_a, tile_b, start, stop, overlap_px) in enumerate(pairs):
            centers = patch_centers(
                start,
                stop,
                spacing_um_zyx=tile_a.spacing_um_zyx,
                patch_shape_zyx=args.patch_shape_zyx,
                count=args.patches_per_pair,
                rng=rng,
            )
            for patch_index, center in enumerate(centers):
                world_points = world_grid_for_patch(
                    center,
                    spacing_um_zyx=tile_a.spacing_um_zyx,
                    shape_zyx=args.patch_shape_zyx,
                )
                row: dict[str, Any] = {
                    "pair_index": pair_index,
                    "patch_index": patch_index,
                    "tile_a": tile_a.name,
                    "tile_b": tile_b.name,
                    "source_view_a": tile_a.source_view,
                    "source_view_b": tile_b.source_view,
                    "overlap_start_um_zyx": start.astype(float).tolist(),
                    "overlap_stop_um_zyx": stop.astype(float).tolist(),
                    "overlap_shape_px_zyx": overlap_px.astype(int).tolist(),
                    "center_um_zyx": center.astype(float).tolist(),
                    "cases": {},
                }
                for case_name in case_names:
                    profiles = None if case_name == RAW_CASE else basic_cases[case_name]
                    profile_a = None if profiles is None else select_basic_profile(profiles, tile_a)
                    profile_b = None if profiles is None else select_basic_profile(profiles, tile_b)
                    sample_a, mask_a = sample_tile_patch(
                        registration_payload=registration_payload,
                        tile=tile_a,
                        world_points_um_zyx=world_points,
                        output_shape_zyx=args.patch_shape_zyx,
                        level=args.level,
                        channel=args.channel,
                        reader_cache=reader_cache,
                        profile=profile_a,
                    )
                    sample_b, mask_b = sample_tile_patch(
                        registration_payload=registration_payload,
                        tile=tile_b,
                        world_points_um_zyx=world_points,
                        output_shape_zyx=args.patch_shape_zyx,
                        level=args.level,
                        channel=args.channel,
                        reader_cache=reader_cache,
                        profile=profile_b,
                    )
                    mask = mask_a & mask_b & np.isfinite(sample_a) & np.isfinite(sample_b)
                    positive_fraction = float(np.count_nonzero((sample_a > 0) & (sample_b > 0) & mask) / max(1, np.count_nonzero(mask)))
                    valid_voxels = int(np.count_nonzero(mask))
                    corr = pearson(sample_a, sample_b, mask)
                    accepted = (
                        valid_voxels >= args.min_valid_voxels
                        and positive_fraction >= args.min_positive_fraction
                        and math.isfinite(corr)
                    )
                    if accepted:
                        correlations[case_name].append(corr)
                    row["cases"][case_name] = {
                        "pearson": finite_float(corr),
                        "accepted": accepted,
                        "valid_voxels": valid_voxels,
                        "positive_fraction": positive_fraction,
                    }
                patch_rows.append(row)
    finally:
        reader_cache.close()

    profile_inputs = {}
    for case_name, profiles in basic_cases.items():
        profile_inputs[case_name] = {
            view: {
                "flatfield": str(profile.flatfield_path),
                "darkfield": None if profile.darkfield_path is None else str(profile.darkfield_path),
                "pre_scale": profile.pre_scale,
                "scale_json": None if profile.scale_path is None else str(profile.scale_path),
            }
            for view, profile in profiles.items()
        }

    return {
        "schema_version": 1,
        "artifact_type": "basicpy_registered_overlap_correlation_benchmark.v1",
        "created_at_unix": time.time(),
        "inputs": {
            "registration_json": str(args.registration_json),
            "channel": int(args.channel),
            "level": int(args.level),
            "patch_shape_zyx": list(args.patch_shape_zyx),
            "patches_per_pair": int(args.patches_per_pair),
            "max_pairs": args.max_pairs,
            "edge_sample_count": args.edge_sample_count,
            "min_overlap_px_zyx": list(args.min_overlap_px_zyx),
            "sample_seed": int(args.sample_seed),
            "image_path_key": args.image_path_key,
            "image_source_view_roots": {view: str(path) for view, path in source_view_roots.items()},
            "min_valid_voxels": int(args.min_valid_voxels),
            "min_positive_fraction": float(args.min_positive_fraction),
            "basic_profiles": profile_inputs,
        },
        "tile_count": len(tiles),
        "candidate_pair_count": len(pairs),
        "case_summaries": {name: summarize(values) for name, values in correlations.items()},
        "patches": patch_rows,
    }


def main() -> None:
    args = parse_args()
    payload = benchmark(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_json": str(args.output_json), "case_summaries": payload["case_summaries"]}, indent=2))


if __name__ == "__main__":
    main()
