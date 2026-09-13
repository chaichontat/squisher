"""Fit residual illumination fields from registered raw-tile overlaps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.csgraph import connected_components
from scipy.special import pseudo_huber
from skimage.filters import threshold_otsu
import tifffile

from squisher_deconv.basic_profiles import compose_basic_profile, compose_z_profile, load_basic_profile_arrays
from squisher_deconv.residual_field import cosine_basis as _basis
from squisher_deconv.tile_gains import write_tile_gains
from squisher_lightsheet import ngff
from squisher_lightsheet.post_basic_qc import write_post_basic_qc
from squisher_lightsheet.ome_metadata_dumb_stitch import (
    BasicProfile,
    TileMetadata,
    apply_basic,
    load_basic_profile,
    read_tile_metadata,
)


DEGREE = 2
PENALTIES = (0.003, 0.03, 0.3)
FOLDS = 5
SEAM_RADIUS = 12
SEAM_GRID = 4
MIN_PAIR_SAMPLES = 12


@dataclass(frozen=True)
class PostBasicChannel:
    label: str
    index: int
    registration: Path


@dataclass(frozen=True)
class _Grid:
    shape: np.ndarray
    scale: np.ndarray
    origin: np.ndarray
    z: int
    stride: int

    @property
    def output_shape(self) -> tuple[int, int]:
        values = (self.shape[1:] + self.stride - 1) // self.stride
        return int(values[0]), int(values[1])


@dataclass(frozen=True)
class _Plan:
    source: Path
    source_shape: tuple[int, int, int]
    source_start: np.ndarray
    source_stop: np.ndarray
    matrix: np.ndarray
    offset: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


@dataclass(frozen=True)
class _SampledTile:
    source: Path
    lo: np.ndarray
    hi: np.ndarray
    data: np.ndarray
    yx: np.ndarray
    valid: np.ndarray
    camera_z: np.ndarray | None = None


@dataclass(frozen=True)
class _SampledPlane:
    grid: _Grid
    tiles: list[_SampledTile]
    before: np.ndarray
    owner: np.ndarray
    cutoff: float
    sampling: dict[str, object]


def parse_channel_spec(value: str) -> PostBasicChannel:
    """Parse LABEL=INDEX=REGISTRATION without restricting characters in the path."""
    parts = value.split("=", 2)
    if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
        raise ValueError(f"Expected LABEL=INDEX=REGISTRATION, got {value!r}")
    try:
        index = int(parts[1])
    except ValueError as error:
        raise ValueError(f"Channel index must be an integer in {value!r}") from error
    if index < 0:
        raise ValueError(f"Channel index must be non-negative in {value!r}")
    return PostBasicChannel(parts[0].strip(), index, Path(parts[2]).expanduser())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, *, digest: bool) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if digest:
        result["sha256"] = _sha256(path)
    return result


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _vector_zyx(record: Mapping[str, object], *keys: str) -> np.ndarray:
    for key in keys:
        value = record.get(key)
        if isinstance(value, Mapping) and all(axis in value for axis in "zyx"):
            result = np.asarray([value[axis] for axis in "zyx"], dtype=np.float64)
            if result.shape == (3,) and np.all(np.isfinite(result)):
                return result
    raise ValueError(f"Tile record {record.get('tile')!r} is missing finite z/y/x values for {keys}")


def _fixed_to_source_pull(
    record: Mapping[str, object],
    *,
    fixed_scale: np.ndarray,
    fixed_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed-grid-index to original-raw-index affine.

    Registered affines map source stage microns into fixed stage microns. A
    materialized crop uses the same stage frame, so its original source-index
    offset is added only after applying the inverse physical affine.
    """
    stage_translation = _vector_zyx(record, "stage_translation_um", "translation_um")
    stage_scale = _vector_zyx(record, "stage_scale_um", "scale_um", "spacing_um")
    if np.any(stage_scale == 0):
        raise ValueError(f"Tile record {record.get('tile')!r} has a zero stage scale")
    registered = record.get("registered_affine")
    if registered is None:
        affine = np.eye(4, dtype=np.float64)
    else:
        matrix = registered.get("matrix") if isinstance(registered, Mapping) else registered
        affine = np.asarray(matrix, dtype=np.float64)
        while affine.ndim > 2:
            affine = affine[0]
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError(
                f"Tile record {record.get('tile')!r} registered_affine must be a finite 4x4 matrix"
            )
    try:
        inverse = np.linalg.inv(affine)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"Tile record {record.get('tile')!r} registered_affine is singular") from error
    crop_start = np.asarray(record.get("materialized_source_start_zyx", [0, 0, 0]), dtype=np.float64)
    if crop_start.shape != (3,) or not np.all(np.isfinite(crop_start)):
        raise ValueError(f"Tile record {record.get('tile')!r} has an invalid materialized crop start")
    matrix = inverse[:3, :3] * fixed_scale[None, :] / stage_scale[:, None]
    offset = crop_start + (inverse[:3, :3] @ fixed_origin + inverse[:3, 3] - stage_translation) / stage_scale
    return matrix, offset


def _tile_identity(name: str) -> str:
    basename = Path(name).name
    for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr"):
        if basename.endswith(suffix):
            return basename.removesuffix(suffix)
    return basename


def _raw_source_index(raw_dir: Path) -> dict[str, Path]:
    paths = sorted((*raw_dir.glob("*.ome.tif"), *raw_dir.glob("*.ome.tiff")))
    if not paths:
        raise FileNotFoundError(f"No raw *.ome.tif or *.ome.tiff sources found in {raw_dir}")
    result: dict[str, Path] = {}
    resolved_seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        identity = _tile_identity(path.name)
        if identity in result:
            raise ValueError(f"Duplicate raw tile identity {identity!r}: {result[identity]} and {path}")
        if resolved in resolved_seen:
            raise ValueError(f"Raw tile paths must resolve uniquely; repeated source: {resolved}")
        result[identity] = resolved
        resolved_seen.add(resolved)
    return result


def _fixed_grid(path: Path, *, fixed_z: int | None, stride: int) -> _Grid:
    import zarr

    root = zarr.open_group(path, mode="r")
    array = ngff.level_array(root, level=0, context=path)
    axes = ngff.axes(root, array)
    names, scale, origin, has_scale, has_origin = ngff.scale_translation(root, dataset_index=0)
    if axes != "ZYX" or [name.lower() for name in names] != list("zyx"):
        raise ValueError(f"Fixed grid must have ZYX axes, got array={axes!r}, metadata={names}")
    if not has_scale or not has_origin:
        raise ValueError(f"Fixed grid {path} must define level-0 scale and translation")
    shape = np.asarray(array.shape, dtype=np.int64)
    scale_array = np.asarray(scale, dtype=np.float64)
    origin_array = np.asarray(origin, dtype=np.float64)
    if np.any(shape <= 0) or np.any(scale_array == 0):
        raise ValueError(f"Fixed grid {path} has invalid shape or scale")
    z = int((shape[0] - 1) // 2 if fixed_z is None else fixed_z)
    if not 0 <= z < shape[0]:
        raise ValueError(f"Fixed Z {z} is outside [0, {shape[0]})")
    return _Grid(shape=shape, scale=scale_array, origin=origin_array, z=z, stride=stride)


def _registration_records(path: Path) -> list[Mapping[str, object]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("tiles"), list):
        raise ValueError(f"Registration {path} must contain a tiles list")
    records = []
    for index, record in enumerate(payload["tiles"]):
        if not isinstance(record, Mapping):
            raise ValueError(f"Registration {path} tile {index} must be an object")
        if "status" in record and record["status"] != "accepted":
            continue
        records.append(record)
    if not records:
        raise ValueError(f"Registration {path} has no accepted tile records")
    return records


def _record_source(record: Mapping[str, object], sources: Mapping[str, Path]) -> Path:
    name = record.get("moving_tile", record.get("tile"))
    if not isinstance(name, str):
        raise ValueError("Registration tile record requires tile or moving_tile identity")
    identity = _tile_identity(name)
    if identity not in sources:
        raise ValueError(f"Registration source {name!r} has no unique raw TIFF in the raw directory")
    return sources[identity]


def _record_bounds(
    record: Mapping[str, object], source_shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(record.get("materialized_source_start_zyx", [0, 0, 0]), dtype=np.int64)
    stop = np.asarray(record.get("materialized_source_stop_zyx", source_shape), dtype=np.int64)
    shape = np.asarray(source_shape, dtype=np.int64)
    if start.shape != (3,) or stop.shape != (3,) or np.any(start < 0) or np.any(stop <= start):
        raise ValueError(f"Tile record {record.get('tile')!r} has invalid source crop bounds")
    if np.any(stop > shape):
        raise ValueError(
            f"Tile record {record.get('tile')!r} crop {start.tolist()}:{stop.tolist()} "
            f"exceeds raw shape {shape.tolist()}"
        )
    return start, stop


def _record_shape_zyx(record: Mapping[str, object]) -> np.ndarray:
    shape = np.asarray(record.get("shape"), dtype=np.int64)
    axes = str(record.get("axes", "ZYX")).upper()
    if shape.ndim != 1 or len(axes) != len(shape) or not set("ZYX") <= set(axes):
        raise ValueError(f"Tile record {record.get('tile')!r} has invalid shape or axes")
    spatial = shape[[axes.index(axis) for axis in "ZYX"]]
    if np.any(spatial < 1):
        raise ValueError(f"Tile record {record.get('tile')!r} has an invalid spatial shape")
    return spatial


def _plan_record(
    *,
    record: Mapping[str, object],
    source: Path,
    source_shape: tuple[int, int, int],
    grid: _Grid,
) -> _Plan | None:
    matrix, offset = _fixed_to_source_pull(record, fixed_scale=grid.scale, fixed_origin=grid.origin)
    source_start, source_stop = _record_bounds(record, source_shape)
    explicit_fixed_extent = "materialized_fixed_origin_um" in record
    fused_fixed_extent = "fixed_origin_um" in record
    if "materialized_source_start_zyx" in record and (explicit_fixed_extent or fused_fixed_extent):
        if explicit_fixed_extent:
            fixed_origin = _vector_zyx(record, "materialized_fixed_origin_um")
            fixed_spacing = _vector_zyx(record, "materialized_fixed_spacing_um")
            fixed_shape = np.asarray(record.get("materialized_fixed_shape_zyx"), dtype=np.int64)
            if fixed_shape.shape != (3,) or np.any(fixed_shape < 1):
                raise ValueError(
                    f"Tile record {record.get('tile')!r} has an invalid materialized fixed shape"
                )
        else:
            # Fused-fixed registrations use their materialized stage grid as the
            # selected fixed window. Its translation includes any overlap expansion.
            fixed_origin = _vector_zyx(record, "stage_translation_um")
            fixed_spacing = _vector_zyx(record, "stage_scale_um")
            fixed_shape = _record_shape_zyx(record)
        fixed_start = (fixed_origin - grid.origin) / grid.scale
        fixed_end = fixed_start + (fixed_spacing / grid.scale) * (fixed_shape - 1)
        fixed_corners = np.stack(
            [fixed_start, fixed_end],
            axis=1,
        )
    else:
        corners = np.asarray(
            list(itertools.product(*zip(source_start, source_stop - 1, strict=True))),
            dtype=np.float64,
        ).T
        try:
            fixed_corners = np.linalg.solve(matrix, corners - offset[:, None])
        except np.linalg.LinAlgError as error:
            raise ValueError(f"Tile record {record.get('tile')!r} has a singular pull transform") from error
    low, high = fixed_corners.min(axis=1), fixed_corners.max(axis=1)
    if not low[0] <= grid.z <= high[0]:
        return None
    lo = np.maximum(0, np.ceil(low[1:] / grid.stride).astype(np.int64))
    hi = np.minimum(
        np.asarray(grid.output_shape),
        np.floor(high[1:] / grid.stride).astype(np.int64) + 1,
    )
    if np.any(hi <= lo):
        return None
    return _Plan(
        source=source,
        source_shape=source_shape,
        source_start=source_start,
        source_stop=source_stop,
        matrix=matrix,
        offset=offset,
        lo=lo,
        hi=hi,
    )


def _pull_grid(plan: _Plan, grid: _Grid) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.arange(plan.lo[0], plan.hi[0]) * grid.stride,
        np.arange(plan.lo[1], plan.hi[1]) * grid.stride,
        indexing="ij",
    )
    fixed = np.stack([np.full_like(yy, grid.z), yy, xx])
    return np.einsum("ij,jhw->ihw", plan.matrix, fixed) + plan.offset[:, None, None]


def _read_corrected_crop(
    tif: tifffile.TiffFile,
    *,
    channel: int,
    source_shape: tuple[int, int, int],
    lo: np.ndarray,
    hi: np.ndarray,
    profile: BasicProfile,
) -> np.ndarray:
    z_count, height, width = source_shape
    if profile.flatfield.shape != (height, width):
        raise ValueError(
            f"BaSiC profile shape {profile.flatfield.shape} does not match raw tile YX {(height, width)}"
        )
    y_slice = slice(int(lo[1]), int(hi[1]))
    x_slice = slice(int(lo[2]), int(hi[2]))
    cropped_profile = BasicProfile(
        flatfield=profile.flatfield[y_slice, x_slice],
        residual_coefficient=profile.residual_coefficient,
        darkfield=None if profile.darkfield is None else profile.darkfield[y_slice, x_slice],
        flatfield_path=profile.flatfield_path,
        darkfield_path=profile.darkfield_path,
    )
    return np.stack(
        [
            apply_basic(
                np.asarray(tif.pages[channel * z_count + z].asarray()[y_slice, x_slice]),
                cropped_profile,
                raw_z=z,
                raw_shape_zyx=source_shape,
                y_slice=y_slice,
                x_slice=x_slice,
            )
            for z in range(int(lo[0]), int(hi[0]))
        ]
    )


def _sample_plan(
    tif: tifffile.TiffFile,
    *,
    plan: _Plan,
    grid: _Grid,
    channel: int,
    profile: BasicProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    coordinates = _pull_grid(plan, grid)
    valid = np.all(
        (coordinates >= plan.source_start[:, None, None])
        & (coordinates <= (plan.source_stop - 1)[:, None, None]),
        axis=0,
    )
    if not np.any(valid):
        return None
    lo = np.maximum(plan.source_start, np.floor(coordinates[:, valid].min(axis=1)).astype(np.int64))
    hi = np.minimum(
        plan.source_stop,
        np.ceil(coordinates[:, valid].max(axis=1)).astype(np.int64) + 1,
    )
    corrected = _read_corrected_crop(
        tif,
        channel=channel,
        source_shape=plan.source_shape,
        lo=lo,
        hi=hi,
        profile=profile,
    )
    data = map_coordinates(
        corrected,
        coordinates - lo[:, None, None],
        order=1,
        prefilter=False,
        mode="constant",
    ).astype(np.float32, copy=False)
    data[~valid] = 0
    weight = np.minimum(distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1], 16).astype(np.float32)
    yx = np.moveaxis(
        np.stack(
            [
                coordinates[1] / (plan.source_shape[1] - 1),
                coordinates[2] / (plan.source_shape[2] - 1),
            ]
        ),
        0,
        -1,
    ).astype(np.float32)
    camera_z = (coordinates[0] / max(plan.source_shape[0] - 1, 1)).astype(np.float32)
    return data, yx, camera_z, weight


def _sample_source(
    source: Path,
    plans: Sequence[_Plan],
    *,
    grid: _Grid,
    channel: int,
    profile: BasicProfile,
) -> _SampledTile | None:
    low = np.min([plan.lo for plan in plans], axis=0)
    high = np.max([plan.hi for plan in plans], axis=0)
    shape = tuple(int(value) for value in high - low)
    data = np.zeros(shape, dtype=np.float32)
    yx = np.zeros((*shape, 2), dtype=np.float64)
    camera_z = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    with tifffile.TiffFile(source) as tif:
        for plan in plans:
            sampled = _sample_plan(tif, plan=plan, grid=grid, channel=channel, profile=profile)
            if sampled is None:
                continue
            patch, patch_yx, patch_z, patch_weight = sampled
            slices = tuple(
                slice(int(start - origin), int(stop - origin))
                for start, stop, origin in zip(plan.lo, plan.hi, low, strict=True)
            )
            target_weight = weight[slices]
            inside = patch_weight > target_weight
            data[slices][inside] = patch[inside]
            yx[slices][inside] = patch_yx[inside]
            camera_z[slices][inside] = patch_z[inside]
            target_weight[inside] = patch_weight[inside]
    valid = weight > 0
    if not np.any(valid):
        return None
    return _SampledTile(source=source, lo=low, hi=high, data=data, yx=yx, valid=valid, camera_z=camera_z)


def _channel_plans(
    spec: PostBasicChannel,
    *,
    sources: Mapping[str, Path],
    grid: _Grid,
) -> list[_Plan]:
    records = _registration_records(spec.registration)
    metadata: dict[Path, TileMetadata] = {}
    plans: list[_Plan] = []
    for record_index, record in enumerate(records):
        source = _record_source(record, sources)
        if source not in metadata:
            metadata[source] = read_tile_metadata(source)
        tile = metadata[source]
        if spec.index >= tile.channel_count:
            raise ValueError(
                f"Channel {spec.index} is outside raw tile {source} with {tile.channel_count} channel(s)"
            )
        declared_channels = record.get("channels")
        # Full multichannel and single-channel records store OME channel labels;
        # materialized ZYX records from multichannel sources declare source indices.
        axes = str(record.get("axes", "ZYX")).upper()
        if "C" in axes:
            shape = record.get("shape")
            if not isinstance(shape, list) or len(shape) != len(axes):
                raise ValueError(f"Registration record {record_index} has invalid shape or axes")
            contains_channel = spec.index < int(shape[axes.index("C")])
        elif tile.channel_count == 1:
            contains_channel = True
        else:
            contains_channel = not isinstance(declared_channels, list) or str(spec.index) in {
                str(value) for value in declared_channels
            }
        if not contains_channel:
            raise ValueError(
                f"Registration {spec.registration} record {record_index} does not contain channel {spec.index}"
            )
        stage_scale = _vector_zyx(record, "stage_scale_um", "scale_um", "spacing_um")
        if not np.allclose(
            np.abs(stage_scale),
            np.asarray(tile.spacing_um_zyx),
            rtol=1e-6,
            atol=1e-9,
        ):
            raise ValueError(
                f"Registration scale {stage_scale.tolist()} for {source} differs from raw TIFF spacing "
                f"{tile.spacing_um_zyx}"
            )
        plan = _plan_record(
            record=record,
            source=source,
            source_shape=tile.shape_zyx,
            grid=grid,
        )
        if plan is None:
            continue
        plans.append(plan)
    if not plans:
        raise ValueError(f"Registration {spec.registration} has no records intersecting fixed Z {grid.z}")
    return plans


def _sample_channel(
    spec: PostBasicChannel,
    *,
    sources: Mapping[str, Path],
    grid: _Grid,
    profile: BasicProfile,
    workers: int,
) -> tuple[list[_SampledTile], dict[str, object]]:
    plans = _channel_plans(spec, sources=sources, grid=grid)
    grouped: dict[Path, list[_Plan]] = defaultdict(list)
    for plan in plans:
        grouped[plan.source].append(plan)
    items = sorted(grouped.items(), key=lambda item: str(item[0]))

    def sample(item: tuple[Path, list[_Plan]]) -> _SampledTile | None:
        source, source_plans = item
        return _sample_source(
            source,
            source_plans,
            grid=grid,
            channel=spec.index,
            profile=profile,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        tiles = [tile for tile in pool.map(sample, items) if tile is not None]
    if len(tiles) < 2:
        raise ValueError(f"Channel {spec.index} has fewer than two sampled source tiles")
    return tiles, {
        "records_intersecting": len(plans),
        "sampled_sources": [str(tile.source) for tile in tiles],
    }


def _mosaic(shape: tuple[int, int], tiles: Sequence[_SampledTile]) -> tuple[np.ndarray, np.ndarray]:
    data = np.zeros(shape, dtype=np.float32)
    owner = np.full(shape, -1, dtype=np.int32)
    for index, tile in enumerate(tiles):
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(tile.lo, tile.hi, strict=True))
        target = data[slices]
        target_owner = owner[slices]
        target[tile.valid] = tile.data[tile.valid]
        target_owner[tile.valid] = index
    return data, owner


def _seam_seed(owner: np.ndarray, *, first: int, second: int) -> np.ndarray:
    seeds = np.zeros(owner.shape, dtype=bool)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left_slice, right_slice = tuple(left), tuple(right)
        edge = ((owner[left_slice] == first) & (owner[right_slice] == second)) | (
            (owner[left_slice] == second) & (owner[right_slice] == first)
        )
        seeds[left_slice] |= edge
        seeds[right_slice] |= edge
    return seeds


def _seam_samples(tiles: Sequence[_SampledTile], owner: np.ndarray, cutoff: float) -> list[dict[str, object]]:
    smooth = []
    for tile in tiles:
        weight = gaussian_filter(tile.valid.astype(np.float32), 1)
        values = gaussian_filter(tile.data, 1)
        smooth.append(np.divide(values, weight, out=np.zeros_like(values), where=weight > 0))
    rows = []
    for first, a in enumerate(tiles):
        for second in range(first + 1, len(tiles)):
            b = tiles[second]
            low, high = np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi)
            if np.any(high - low < 8):
                continue
            a_slice = tuple(
                slice(int(start - origin), int(stop - origin))
                for start, stop, origin in zip(low, high, a.lo, strict=True)
            )
            b_slice = tuple(
                slice(int(start - origin), int(stop - origin))
                for start, stop, origin in zip(low, high, b.lo, strict=True)
            )
            # Include one neighboring pixel so overlap-edge seams use the same
            # adjacency rule as interior seams, then crop back to the overlap.
            halo_low, halo_high = np.maximum(low - 1, 0), np.minimum(high + 1, owner.shape)
            halo = tuple(
                slice(int(start), int(stop)) for start, stop in zip(halo_low, halo_high, strict=True)
            )
            interior = tuple(
                slice(int(start - origin), int(stop - origin))
                for start, stop, origin in zip(low, high, halo_low, strict=True)
            )
            seeds = _seam_seed(owner[halo], first=first, second=second)[interior]
            if not np.any(seeds):
                continue
            a_values, b_values = smooth[first][a_slice], smooth[second][b_slice]
            mask = (
                (distance_transform_edt(~seeds) <= SEAM_RADIUS)
                & a.valid[a_slice]
                & b.valid[b_slice]
                & (a.data[a_slice] > cutoff)
                & (b.data[b_slice] > cutoff)
                & (a_values > 0)
                & (b_values > 0)
            )
            grid_mask = np.zeros(mask.shape, dtype=bool)
            grid_mask[::SEAM_GRID, ::SEAM_GRID] = True
            mask &= grid_mask
            if np.count_nonzero(mask) < MIN_PAIR_SAMPLES:
                continue
            source_a, source_b = str(a.source), str(b.source)
            pair_id = json.dumps(
                [source_a, source_b, low.tolist(), high.tolist()],
                separators=(",", ":"),
            )
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "pair_id": pair_id,
                    "low": low,
                    "high": high,
                    "ya": a.yx[a_slice][mask],
                    "yb": b.yx[b_slice][mask],
                    "za": None if a.camera_z is None else a.camera_z[a_slice][mask],
                    "zb": None if b.camera_z is None else b.camera_z[b_slice][mask],
                    "target": np.log(b_values[mask] / a_values[mask]),
                }
            )
    return sorted(rows, key=lambda row: str(row["pair_id"]))


def _row_design(row: Mapping[str, object], z_degree: int) -> np.ndarray:
    return _basis(np.asarray(row["ya"]), z=row.get("za"), z_degree=z_degree) - _basis(
        np.asarray(row["yb"]), z=row.get("zb"), z_degree=z_degree
    )


def _fit_field(
    design: np.ndarray | spmatrix,
    target: np.ndarray,
    penalty: float | np.ndarray,
) -> np.ndarray:
    if design.shape[0] == 0 or design.shape[0] != target.shape[0]:
        raise ValueError("Residual fit requires non-empty aligned design and target rows")
    penalty_by_coefficient = np.broadcast_to(np.asarray(penalty, dtype=np.float64), (design.shape[1],))
    if not np.all(np.isfinite(penalty_by_coefficient)) or np.any(penalty_by_coefficient <= 0):
        raise ValueError("Residual fit penalties must be positive and finite")

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        residual = np.asarray(design @ coefficient).ravel() - target
        value = np.mean(pseudo_huber(0.1, residual)) + 0.5 * np.sum(penalty_by_coefficient * coefficient**2)
        gradient = (
            np.asarray(design.T @ (residual / np.sqrt(1 + (residual / 0.1) ** 2))).ravel() / design.shape[0]
            + penalty_by_coefficient * coefficient
        )
        return float(value), gradient

    fit = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 1000, "maxls": 100},
    )
    if not fit.success:
        gradient_norm = float(np.linalg.norm(np.asarray(fit.jac), ord=np.inf))
        raise RuntimeError(
            f"Residual fit failed: {fit.message}; iterations={fit.nit}, "
            f"objective={fit.fun:.12g}, gradient_inf_norm={gradient_norm:.12g}"
        )
    return np.asarray(fit.x, dtype=np.float64)


def _source_field_design(
    rows: Sequence[Mapping[str, object]],
    *,
    source_count: int,
    z_degree: int,
    active: np.ndarray,
) -> csr_matrix:
    """Build sparse per-source field differences without expanding pixel-by-source arrays."""
    values = []
    row_indexes = []
    column_indexes = []
    offset = 0
    mode_count = len(active)
    for row in rows:
        first_basis = _basis(np.asarray(row["ya"]), z=row.get("za"), z_degree=z_degree)[:, active]
        second_basis = _basis(np.asarray(row["yb"]), z=row.get("zb"), z_degree=z_degree)[:, active]
        if first_basis.shape != second_basis.shape:
            raise ValueError("Source field samples must have aligned camera coordinates")
        pixels = np.arange(offset, offset + len(first_basis))
        repeated_pixels = np.repeat(pixels, mode_count)
        row_indexes.extend((repeated_pixels, repeated_pixels))
        column_indexes.extend(
            (
                int(row["first"]) * mode_count + np.tile(np.arange(mode_count), len(pixels)),
                int(row["second"]) * mode_count + np.tile(np.arange(mode_count), len(pixels)),
            )
        )
        values.extend((first_basis.ravel(), -second_basis.ravel()))
        offset += len(first_basis)
    return csr_matrix(
        (
            np.concatenate(values),
            (np.concatenate(row_indexes), np.concatenate(column_indexes)),
        ),
        shape=(offset, source_count * mode_count),
    )


def _expanded_source_fields(
    compact: np.ndarray,
    *,
    source_count: int,
    z_degree: int,
    active: np.ndarray,
) -> np.ndarray:
    coefficient = np.zeros((source_count, 5 * (z_degree + 1)))
    coefficient[:, active] = compact.reshape(source_count, len(active))
    return coefficient


def _pair_metrics(residual: np.ndarray, pair: np.ndarray) -> dict[str, float | int]:
    jumps = [abs(float(np.median(residual[pair == index]))) for index in np.unique(pair)]
    if not jumps:
        raise ValueError("Cannot compute seam metrics without overlap pairs")
    return {
        "pairs": len(jumps),
        "median": float(np.median(jumps)),
        "p90": float(np.percentile(jumps, 90)),
        "pixel_abs_median": float(np.median(np.abs(residual))),
    }


def _pair_folds(rows: Sequence[Mapping[str, object]], *, seed: int) -> np.ndarray:
    groups = sorted({(int(row["first"]), int(row["second"])) for row in rows})
    if len(groups) < FOLDS:
        raise ValueError(
            f"Residual fitting requires at least {FOLDS} eligible source pairs; got {len(groups)}"
        )
    order = np.random.default_rng(seed).permutation(len(groups))
    group_folds = np.empty(len(groups), dtype=np.int64)
    group_folds[order] = np.arange(len(groups)) % FOLDS
    folds = dict(zip(groups, group_folds, strict=True))
    return np.asarray([folds[(int(row["first"]), int(row["second"]))] for row in rows])


def _fit_seams(
    *,
    rows: Sequence[Mapping[str, object]],
    sources: Sequence[Path],
    seed: int,
    fit_tile_gains: bool,
    fit_source_fields: bool = False,
    z_degree: int = 0,
    xy_degree: int = 2,
    field_penalty: float | None = None,
    source_field_penalty: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if xy_degree not in (1, 2):
        raise ValueError("XY degree must be 1 or 2")
    if field_penalty is not None and (not np.isfinite(field_penalty) or field_penalty <= 0):
        raise ValueError("Field penalty must be positive and finite")
    if source_field_penalty is not None and (
        not np.isfinite(source_field_penalty) or source_field_penalty <= 0
    ):
        raise ValueError("Source-field penalty must be positive and finite")
    penalties = PENALTIES if field_penalty is None else (field_penalty,)
    # Fit the requested subspace; inactive modes retain exact zeros in the common profile encoding.
    active = np.tile([True, xy_degree == 2, True, xy_degree == 2, xy_degree == 2], z_degree + 1)
    folds = _pair_folds(rows, seed=seed)
    target = np.concatenate([np.asarray(row["target"]) for row in rows])
    pair = np.concatenate(
        [np.full(len(np.asarray(row["target"])), index, dtype=np.int64) for index, row in enumerate(rows)]
    )
    pixel_fold = folds[pair]
    train, validation, test = pixel_fold >= 2, pixel_fold == 1, pixel_fold == 0
    design = np.concatenate([_row_design(row, z_degree) for row in rows])
    trials = []
    for penalty in penalties:
        train_coefficient = np.zeros(design.shape[1])
        train_coefficient[active] = _fit_field(design[train][:, active], target[train], penalty)
        error = (design @ train_coefficient - target) / np.log(2)
        trials.append(
            (
                penalty,
                _pair_metrics(error[validation], pair[validation]),
                train_coefficient,
            )
        )
    selected = min(
        trials,
        key=lambda trial: (
            trial[1]["median"] + 0.25 * trial[1]["p90"],
            trial[0],
        ),
    )
    coefficient = np.zeros(design.shape[1])
    coefficient[active] = _fit_field(design[~test][:, active], target[~test], selected[0])
    shared_error = design @ coefficient - target
    result: dict[str, object] = {
        "degree": xy_degree,
        "z_degree": z_degree,
        "field_coordinate": "normalized-original-raw-zyx",
        "penalties": list(penalties),
        "metric_unit": "absolute log2 fold-change",
        "selected_penalty": selected[0],
        "coefficient": coefficient.tolist(),
        "heldout_before": _pair_metrics(-target[test] / np.log(2), pair[test]),
        "heldout_after": _pair_metrics(shared_error[test] / np.log(2), pair[test]),
        "pairs": [
            {
                "id": hashlib.sha256(f"{row['z']}:{row['pair_id']}".encode()).hexdigest()[:16],
                "sources": [
                    sources[int(row["first"])].name,
                    sources[int(row["second"])].name,
                ],
                "z": int(row["z"]),
                "samples": len(np.asarray(row["target"])),
                "fold": int(folds[index]),
            }
            for index, row in enumerate(rows)
        ],
        "trials": [{"penalty": trial[0], "validation": trial[1]} for trial in trials],
    }
    incidence = np.zeros((len(rows), len(sources)), dtype=np.float64)
    for index, row in enumerate(rows):
        incidence[index, int(row["first"])] = 1
        incidence[index, int(row["second"])] = -1

    train_log_gain = np.zeros(len(sources), dtype=np.float64)
    log_gain = np.zeros(len(sources), dtype=np.float64)
    tile_error = shared_error
    if fit_tile_gains:
        train_shared_error = design @ selected[2] - target
        train_pair_target = np.asarray(
            [-np.median(train_shared_error[pair == index]) for index in range(len(rows))],
            dtype=np.float64,
        )
        gain_trials = []
        for penalty in PENALTIES:
            candidate = _fit_field(incidence[folds >= 2], train_pair_target[folds >= 2], penalty)
            error = (incidence @ candidate - train_pair_target) / np.log(2)
            gain_trials.append(
                (
                    penalty,
                    _pair_metrics(error[folds == 1], np.flatnonzero(folds == 1)),
                    candidate,
                )
            )
        selected_gain = min(
            gain_trials,
            key=lambda trial: (
                trial[1]["median"] + 0.25 * trial[1]["p90"],
                trial[0],
            ),
        )
        train_log_gain = selected_gain[2]
        final_pair_target = np.asarray(
            [-np.median(shared_error[pair == index]) for index in range(len(rows))],
            dtype=np.float64,
        )
        log_gain = _fit_field(incidence[folds != 0], final_pair_target[folds != 0], selected_gain[0])
        support = np.count_nonzero(incidence[folds != 0], axis=0)
        log_gain[support == 0] = 0
        unsupported_sources = [str(source) for index, source in enumerate(sources) if support[index] == 0]
        tile_error = shared_error + incidence[pair] @ log_gain
        result.update(
            {
                "tile_gain_selected_penalty": selected_gain[0],
                "heldout_shared": _pair_metrics(shared_error[test] / np.log(2), pair[test]),
                "heldout_with_tile_gain": _pair_metrics(tile_error[test] / np.log(2), pair[test]),
                "tile_gain_trials": [{"penalty": trial[0], "validation": trial[1]} for trial in gain_trials],
                "tile_gain_zero_support_policy": "identity",
                "tile_gain_unestimated_sources": unsupported_sources,
                "tile_gains": [
                    {
                        "source": str(source),
                        "gain": float(np.exp(log_gain[index])),
                        "training_pairs": int(support[index]),
                    }
                    for index, source in enumerate(sources)
                ],
            }
        )

    if fit_source_fields:
        source_penalties = PENALTIES if source_field_penalty is None else (source_field_penalty,)
        active_indexes = np.flatnonzero(active)
        source_design = _source_field_design(
            rows,
            source_count=len(sources),
            z_degree=z_degree,
            active=active_indexes,
        )
        train_base_error = design @ selected[2] - target + incidence[pair] @ train_log_gain
        source_trials = []
        for penalty in source_penalties:
            candidate = _fit_field(source_design[train], -train_base_error[train], penalty)
            error = (train_base_error + np.asarray(source_design @ candidate).ravel()) / np.log(2)
            source_trials.append(
                (
                    penalty,
                    _pair_metrics(error[validation], pair[validation]),
                )
            )
        selected_source = min(
            source_trials,
            key=lambda trial: (
                trial[1]["median"] + 0.25 * trial[1]["p90"],
                trial[0],
            ),
        )
        source_compact = _fit_field(source_design[~test], -tile_error[~test], selected_source[0])
        source_coefficient = _expanded_source_fields(
            source_compact,
            source_count=len(sources),
            z_degree=z_degree,
            active=active_indexes,
        )
        source_error = tile_error + np.asarray(source_design @ source_compact).ravel()
        source_support = np.count_nonzero(incidence[folds != 0], axis=0)
        result.update(
            {
                "source_fields_applied": True,
                "source_field_selected_penalty": selected_source[0],
                "source_field_trials": [
                    {"penalty": trial[0], "validation": trial[1]} for trial in source_trials
                ],
                "heldout_with_source_field": _pair_metrics(source_error[test] / np.log(2), pair[test]),
                "source_fields": [
                    {
                        "source": str(source),
                        "coefficient": source_coefficient[index].tolist(),
                        "training_pairs": int(source_support[index]),
                    }
                    for index, source in enumerate(sources)
                ],
            }
        )
    else:
        result["source_fields_applied"] = False
    return coefficient, np.exp(log_gain), result


def _refit_seams(
    *,
    rows: Sequence[Mapping[str, object]],
    sources: Sequence[Path],
    selection: Mapping[str, object],
    fit_tile_gains: bool,
    fit_source_fields: bool,
    z_degree: int,
    xy_degree: int,
    require_full_support: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
    """Refit selected penalties on all supplied pairs."""
    design = np.concatenate([_row_design(row, z_degree) for row in rows])
    target = np.concatenate([np.asarray(row["target"]) for row in rows])
    active = np.tile(
        [True, xy_degree == 2, True, xy_degree == 2, xy_degree == 2],
        z_degree + 1,
    )
    coefficient = np.zeros(design.shape[1])
    coefficient[active] = _fit_field(design[:, active], target, float(selection["selected_penalty"]))

    incidence = np.zeros((len(rows), len(sources)), dtype=np.float64)
    pair_target = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        first, second = int(row["first"]), int(row["second"])
        incidence[index, first] = 1
        incidence[index, second] = -1
        pair_target[index] = float(
            np.median(np.asarray(row["target"]) - _row_design(row, z_degree) @ coefficient)
        )
    support = np.count_nonzero(incidence, axis=0)
    if require_full_support and (fit_tile_gains or fit_source_fields):
        if np.any(support == 0):
            unsupported = [str(source) for source, count in zip(sources, support, strict=True) if count == 0]
            raise ValueError(f"No overlap support for sources: {unsupported}")
        adjacency = np.abs(incidence).T @ np.abs(incidence)
        if connected_components(adjacency, directed=False, return_labels=False) != 1:
            raise ValueError("Residual correction requires a connected overlap graph across all sources")

    gains = np.ones(len(sources), dtype=np.float64)
    if fit_tile_gains:
        gains = np.exp(
            _fit_field(
                incidence,
                pair_target,
                float(selection["tile_gain_selected_penalty"]),
            )
        )

    source_fields: list[dict[str, object]] = []
    if fit_source_fields:
        active_indexes = np.flatnonzero(active)
        source_design = _source_field_design(
            rows,
            source_count=len(sources),
            z_degree=z_degree,
            active=active_indexes,
        )
        pair = np.concatenate(
            [np.full(len(np.asarray(row["target"])), index, dtype=np.int64) for index, row in enumerate(rows)]
        )
        base_error = design @ coefficient - target + incidence[pair] @ np.log(gains)
        compact = _fit_field(
            source_design,
            -base_error,
            float(selection["source_field_selected_penalty"]),
        )
        expanded = _expanded_source_fields(
            compact,
            source_count=len(sources),
            z_degree=z_degree,
            active=active_indexes,
        )
        source_fields = [
            {
                "source": str(source),
                "coefficient": expanded[index].tolist(),
                "training_pairs": int(support[index]),
            }
            for index, source in enumerate(sources)
        ]
    return coefficient, gains, source_fields, support


def fit_residual_model(
    *,
    rows: Sequence[Mapping[str, object]],
    sources: Sequence[Path],
    seed: int,
    fit_tile_gains: bool,
    fit_source_fields: bool = False,
    z_degree: int = 0,
    xy_degree: int = 2,
    field_penalty: float | None = None,
    source_field_penalty: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Evaluate on held-out pairs, then refit the deployment model on every pair."""
    _, _, result = _fit_seams(
        rows=rows,
        sources=sources,
        seed=seed,
        fit_tile_gains=fit_tile_gains,
        fit_source_fields=fit_source_fields,
        z_degree=z_degree,
        xy_degree=xy_degree,
        field_penalty=field_penalty,
        source_field_penalty=source_field_penalty,
    )
    coefficient, gains, source_fields, support = _refit_seams(
        rows=rows,
        sources=sources,
        selection=result,
        fit_tile_gains=fit_tile_gains,
        fit_source_fields=fit_source_fields,
        z_degree=z_degree,
        xy_degree=xy_degree,
        require_full_support=True,
    )
    result["evaluation_coefficient"] = result["coefficient"]
    result["coefficient"] = coefficient.tolist()
    result["deployment_training"] = "all eligible source pairs"
    if fit_tile_gains:
        result["tile_gain_unestimated_sources"] = []
        result["tile_gains"] = [
            {
                "source": str(source),
                "gain": float(gain),
                "training_pairs": int(count),
            }
            for source, gain, count in zip(sources, gains, support, strict=True)
        ]

    if fit_source_fields:
        result["evaluation_source_fields"] = result["source_fields"]
        result["source_fields"] = source_fields
    return coefficient, gains, result


def _validate_z_models(
    rows: Sequence[Mapping[str, object]],
    *,
    sources: Sequence[Path],
    reference_z: int,
    seed: int,
    fit_tile_gains: bool,
    fit_source_fields: bool = False,
    z_degree: int = 0,
    xy_degree: int = 2,
    field_penalty: float | None = None,
    source_field_penalty: float | None = None,
) -> list[dict[str, object]]:
    """Compare pooled-plane and single-plane fits on wholly excluded Z planes.

    Penalty selection and source-pair splits run only within the training planes.
    Source indices remain fixed across planes so each tile has one shared gain.
    """
    planes = sorted({int(row["z"]) for row in rows})
    evaluations = []
    for heldout_z in planes:
        training_z = [z for z in planes if z != heldout_z]
        training = [row for row in rows if row["z"] != heldout_z]
        test_rows = [row for row in rows if row["z"] == heldout_z]
        single_z = min(training_z, key=lambda z: (abs(z - reference_z), z))
        shared_evaluation = _fit_seams(
            rows=training,
            sources=sources,
            seed=seed,
            fit_tile_gains=fit_tile_gains,
            fit_source_fields=fit_source_fields,
            xy_degree=xy_degree,
            field_penalty=field_penalty,
            source_field_penalty=source_field_penalty,
            z_degree=z_degree,
        )
        shared_coefficient, shared_gains, shared_fields, _ = _refit_seams(
            rows=training,
            sources=sources,
            selection=shared_evaluation[2],
            fit_tile_gains=fit_tile_gains,
            fit_source_fields=fit_source_fields,
            z_degree=z_degree,
            xy_degree=xy_degree,
            require_full_support=False,
        )
        shared_result = dict(shared_evaluation[2])
        shared_result["source_fields"] = shared_fields
        shared = (shared_coefficient, shared_gains, shared_result)
        shared_2d = shared
        if z_degree:
            shared_2d_evaluation = _fit_seams(
                rows=training,
                sources=sources,
                seed=seed,
                fit_tile_gains=fit_tile_gains,
                fit_source_fields=fit_source_fields,
                xy_degree=xy_degree,
                field_penalty=field_penalty,
                source_field_penalty=source_field_penalty,
            )
            shared_2d_coefficient, shared_2d_gains, shared_2d_fields, _ = _refit_seams(
                rows=training,
                sources=sources,
                selection=shared_2d_evaluation[2],
                fit_tile_gains=fit_tile_gains,
                fit_source_fields=fit_source_fields,
                z_degree=0,
                xy_degree=xy_degree,
                require_full_support=False,
            )
            shared_2d_result = dict(shared_2d_evaluation[2])
            shared_2d_result["source_fields"] = shared_2d_fields
            shared_2d = (shared_2d_coefficient, shared_2d_gains, shared_2d_result)
        single_rows = [row for row in rows if row["z"] == single_z]
        single_coefficient, single_gains, single_fields, _ = _refit_seams(
            rows=single_rows,
            sources=sources,
            selection=shared_2d[2],
            fit_tile_gains=fit_tile_gains,
            fit_source_fields=fit_source_fields,
            z_degree=0,
            xy_degree=xy_degree,
            require_full_support=False,
        )
        single = (single_coefficient, single_gains, {"source_fields": single_fields})
        target = np.concatenate([np.asarray(row["target"]) for row in test_rows])
        pair = np.concatenate(
            [np.full(len(np.asarray(row["target"])), index) for index, row in enumerate(test_rows)]
        )
        evaluation = {
            "heldout_z": heldout_z,
            "training_z": training_z,
            "single_plane_z": single_z,
            "before": _pair_metrics(-target / np.log(2), pair),
        }
        models = [("single_plane", single), ("shared", shared)]
        if z_degree:
            models.append(("shared_2d", shared_2d))
        for name, (coefficient, gains, model_result) in models:
            model_z_degree = len(coefficient) // 5 - 1
            fitted_source_fields = model_result.get("source_fields", [])
            if fitted_source_fields and [Path(row["source"]) for row in fitted_source_fields] != list(
                sources
            ):
                raise RuntimeError("Source fields do not align with validation sources")
            source_fields = (
                np.asarray([row["coefficient"] for row in fitted_source_fields])
                if fitted_source_fields
                else np.zeros((len(sources), len(coefficient)))
            )
            residual = np.concatenate(
                [
                    _row_design(row, model_z_degree) @ coefficient
                    - np.asarray(row["target"])
                    + np.log(gains[int(row["first"])])
                    - np.log(gains[int(row["second"])])
                    + _basis(
                        np.asarray(row["ya"]),
                        z=row.get("za"),
                        z_degree=model_z_degree,
                    )
                    @ source_fields[int(row["first"])]
                    - _basis(
                        np.asarray(row["yb"]),
                        z=row.get("zb"),
                        z_degree=model_z_degree,
                    )
                    @ source_fields[int(row["second"])]
                    for row in test_rows
                ]
            )
            evaluation[name] = _pair_metrics(residual / np.log(2), pair)
        evaluations.append(evaluation)
        logger.info(
            "Residual model validated held-out Z {} ({}/{})", heldout_z, len(evaluations), len(planes)
        )
    return evaluations


def _corrected_tiles(
    tiles: Sequence[_SampledTile],
    coefficient: np.ndarray,
    gains: np.ndarray,
    source_coefficients: np.ndarray | None = None,
) -> list[_SampledTile]:
    corrected = []
    if source_coefficients is None:
        source_coefficients = np.zeros((len(tiles), len(coefficient)))
    if source_coefficients.shape != (len(tiles), len(coefficient)):
        raise ValueError("Source fields must align with sampled tiles and shared coefficients")
    for tile, gain, source_coefficient in zip(tiles, gains, source_coefficients, strict=True):
        field = np.exp(
            (
                _basis(
                    tile.yx.reshape(-1, 2),
                    z=None if tile.camera_z is None else tile.camera_z.ravel(),
                    z_degree=len(coefficient) // 5 - 1,
                )
                @ (coefficient + source_coefficient)
            ).reshape(tile.data.shape)
        )
        corrected.append(replace(tile, data=(tile.data * field * gain).astype(np.float32)))
    return corrected


def _boundary_metrics(
    before: np.ndarray, after: np.ndarray, owner: np.ndarray, cutoff: float
) -> dict[str, object]:
    records = []
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first_slice, second_slice = tuple(first), tuple(second)
        a, b = owner[first_slice], owner[second_slice]
        valid = (
            (a >= 0) & (b >= 0) & (a != b) & (before[first_slice] > cutoff) & (before[second_slice] > cutoff)
        )
        for tile_a, tile_b in sorted(set(zip(a[valid].tolist(), b[valid].tolist(), strict=True))):
            mask = valid & (a == tile_a) & (b == tile_b)
            if np.count_nonzero(mask) < 24:
                continue
            jumps = []
            for data in (before, after):
                jumps.append(
                    abs(float(np.median(np.log2(data[first_slice][mask] / data[second_slice][mask]))))
                )
            records.append(
                {
                    "axis": axis,
                    "pair": [int(tile_a), int(tile_b)],
                    "pixels": int(np.count_nonzero(mask)),
                    "before": jumps[0],
                    "after": jumps[1],
                }
            )
    summary: dict[str, object] = {}
    for axis in (0, 1):
        selected = [record for record in records if record["axis"] == axis]
        if selected:
            summary[str(axis)] = {
                name: {
                    "median": float(np.median([record[name] for record in selected])),
                    "p90": float(np.percentile([record[name] for record in selected], 90)),
                }
                for name in ("before", "after")
            }
    return {"summary": summary, "records": records}


def _display(image: np.ndarray, limits: Sequence[float]) -> Image.Image:
    low, high = limits
    scaled = (255 * np.clip((image - low) / (high - low), 0, 1)).astype(np.uint8)
    return Image.fromarray(scaled)


def _write_qc(
    output: Path,
    *,
    root: Path,
    channel: int,
    before: np.ndarray,
    after: np.ndarray,
    spacing_yx: np.ndarray,
    labels: tuple[str, str] = ("BaSiC", "BaSiC + residual"),
) -> dict[str, dict[str, object]]:
    positive = before[before > 0]
    if positive.size == 0:
        raise ValueError(f"Channel {channel} sampled mosaic has no positive pixels")
    limits = np.percentile(positive, [1, 99.5]).tolist()
    images = []
    artifacts: dict[str, dict[str, object]] = {}
    for name, data in (("basic", before), ("corrected", after)):
        tiff_path = output / f"{name}-ch{channel}.ome.tif"
        tifffile.imwrite(
            tiff_path,
            data.astype(np.float32),
            ome=True,
            tile=tuple(max(16, min(256, (size // 16) * 16)) for size in data.shape),
            compression="zstd",
            metadata={
                "axes": "YX",
                "PhysicalSizeY": float(spacing_yx[0]),
                "PhysicalSizeX": float(spacing_yx[1]),
            },
        )
        image = _display(data, limits)
        png_path = output / f"{name}-ch{channel}.png"
        image.save(png_path)
        images.append(image)
        artifacts[f"{name}_tiff"] = _artifact(tiff_path, root)
        artifacts[f"{name}_png"] = _artifact(png_path, root)
    comparison = Image.new("RGB", (before.shape[1] * 2, before.shape[0] + 28), "white")
    draw = ImageDraw.Draw(comparison)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        comparison.paste(image, (index * before.shape[1], 28))
        draw.text((index * before.shape[1] + 8, 7), label, fill="black")
    comparison_path = output / f"compare-ch{channel}.png"
    comparison.save(comparison_path)
    artifacts["comparison"] = _artifact(comparison_path, root)
    artifacts["display_range"] = {"values": limits}
    return artifacts


def _profile_files(basic_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(basic_dir.glob("*-ch*.pkl")):
        match = re.search(r"-ch(\d+)\.pkl$", path.name)
        if match is None:
            continue
        channel = int(match.group(1))
        if channel in result:
            raise ValueError(f"Multiple BaSiC pickle profiles found for channel {channel} in {basic_dir}")
        result[channel] = path.resolve()
    if not result:
        raise FileNotFoundError(f"No *-chN.pkl BaSiC profiles found in {basic_dir}")
    expected = set(range(max(result) + 1))
    if set(result) != expected:
        raise ValueError(f"BaSiC pickle channel indexes must be contiguous; found {sorted(result)}")
    return result


def _profile_input_identities(basic_dir: Path, profiles: Mapping[int, Path]) -> dict[str, dict[str, object]]:
    result = {}
    for channel, profile in profiles.items():
        sidecar = load_basic_profile(basic_dir, channel)
        files = [profile, sidecar.flatfield_path]
        if sidecar.darkfield_path is not None:
            files.append(sidecar.darkfield_path)
        result[str(channel)] = {path.name: _file_identity(path, digest=True) for path in files}
    return result


def _validated_basic_profile(basic_dir: Path, profile_path: Path, channel: int) -> BasicProfile:
    sidecar = load_basic_profile(basic_dir, channel)
    serialized = load_basic_profile_arrays(profile_path)
    if serialized.residual_coefficient is not None:
        raise ValueError(
            f"Post-BaSiC fitting requires a base profile without a Z-dependent residual: {profile_path}"
        )
    if not np.array_equal(sidecar.flatfield, serialized.flatfield):
        raise ValueError(
            f"Channel {channel} BaSiC flatfield TIFF does not match serialized profile {profile_path}"
        )
    if (sidecar.darkfield is None) != (serialized.darkfield is None) or (
        sidecar.darkfield is not None and not np.array_equal(sidecar.darkfield, serialized.darkfield)
    ):
        raise ValueError(
            f"Channel {channel} BaSiC darkfield TIFF does not match serialized profile {profile_path}"
        )
    return sidecar


def _copy_or_compose_profiles(
    *,
    profiles: Mapping[int, Path],
    basic_profiles: Mapping[int, BasicProfile],
    masks: Mapping[int, Path],
    results: Mapping[int, Mapping[str, object]],
    output: Path,
) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for channel, source in profiles.items():
        target = output / source.name
        profile = basic_profiles[channel]
        if channel not in masks:
            shutil.copy2(source, target)
            shutil.copy2(profile.flatfield_path, output / profile.flatfield_path.name)
            if profile.darkfield_path is not None:
                shutil.copy2(profile.darkfield_path, output / profile.darkfield_path.name)
        else:
            mask = np.asarray(tifffile.imread(masks[channel]), dtype=np.float32)
            compose = compose_z_profile if results[channel]["z_degree"] else compose_basic_profile
            correction = (
                {"coefficient": np.asarray(results[channel]["coefficient"]).reshape(-1, 5)}
                if results[channel]["z_degree"]
                else {"mask": mask}
            )
            composed = compose(
                source,
                target,
                **correction,
                provenance={
                    "operation": "BaSiC multiplied by raw-Z residual"
                    if results[channel]["z_degree"]
                    else "flatfield / shared_mask; darkfield unchanged",
                    "model": "regularized cosine log-gain",
                    "mask": masks[channel].name,
                    "mask_sha256": _sha256(masks[channel]),
                    "source_basic": str(source),
                    "source_basic_sha256": _sha256(source),
                    "post_basic_manifest": "manifest.json",
                    "channel": channel,
                    "selected_penalty": results[channel]["selected_penalty"],
                },
            )
            tifffile.imwrite(
                output / profile.flatfield_path.name,
                composed.flatfield,
                compression="zstd",
            )
            if composed.darkfield is not None and profile.darkfield_path is not None:
                tifffile.imwrite(
                    output / profile.darkfield_path.name,
                    composed.darkfield,
                    compression="zstd",
                )
        artifacts[str(channel)] = {
            path.name: _artifact(path, output)
            for path in output.iterdir()
            if path.name == source.name
            or path.name == profile.flatfield_path.name
            or (profile.darkfield_path is not None and path.name == profile.darkfield_path.name)
        }
    return artifacts


def _snapshot_inputs(
    *,
    fixed_fused: Path,
    raw_sources: Sequence[Path],
    specs: Sequence[PostBasicChannel],
    profile_identities: Mapping[str, object],
) -> dict[str, object]:
    grid_files = [fixed_fused / "zarr.json", fixed_fused / "0" / "zarr.json"]
    return {
        "fixed_grid": [_file_identity(path, digest=True) for path in grid_files],
        "registrations": {str(spec.index): _file_identity(spec.registration, digest=True) for spec in specs},
        "raw_sources": [_file_identity(path, digest=False) for path in raw_sources],
        "basic_profiles": profile_identities,
    }


def run_post_basic(
    *,
    fixed_fused: Path,
    raw_dir: Path,
    basic_dir: Path,
    output_dir: Path,
    channels: Sequence[PostBasicChannel],
    tile_gain_channels: frozenset[int] = frozenset(),
    fixed_z: int | None = None,
    z_percentiles: Sequence[float] = (),
    z_degree: int = 0,
    xy_degree: int = 2,
    field_penalty: float | None = None,
    stride: int = 4,
    workers: int = 8,
    seed: int = 20260907,
) -> Path:
    """Fit and compose post-BaSiC corrections using channel-correct saved geometry."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite post-BaSiC output: {output_dir}")
    if not channels:
        raise ValueError("At least one post-BaSiC channel specification is required")
    if stride < 1 or workers < 1:
        raise ValueError("Post-BaSiC stride and workers must be positive")
    indexes = [spec.index for spec in channels]
    labels = [spec.label for spec in channels]
    if len(set(indexes)) != len(indexes) or len(set(labels)) != len(labels):
        raise ValueError("Post-BaSiC channel indexes and labels must be unique")
    if not tile_gain_channels <= set(indexes):
        raise ValueError("Tile-gain channels must be included in the channel specifications")
    for spec in channels:
        if not spec.registration.is_file():
            raise FileNotFoundError(f"Missing channel registration: {spec.registration}")

    fixed_fused = fixed_fused.resolve()
    raw_dir = raw_dir.resolve()
    basic_dir = basic_dir.resolve()
    output_dir = output_dir.resolve()
    specs = tuple(sorted(channels, key=lambda spec: spec.index))
    sources = _raw_source_index(raw_dir)
    raw_sources = [sources[key] for key in sorted(sources)]
    profiles = _profile_files(basic_dir)
    if any(spec.index not in profiles for spec in specs):
        raise ValueError(
            f"Channel specifications {indexes} are not covered by BaSiC profiles {sorted(profiles)}"
        )
    basic_profiles = {
        channel: _validated_basic_profile(basic_dir, profile, channel)
        for channel, profile in profiles.items()
    }
    profile_identities = _profile_input_identities(basic_dir, profiles)
    input_snapshot = _snapshot_inputs(
        fixed_fused=fixed_fused,
        raw_sources=raw_sources,
        specs=specs,
        profile_identities=profile_identities,
    )
    if xy_degree not in (1, 2):
        raise ValueError("XY degree must be 1 or 2")
    if field_penalty is not None and (not np.isfinite(field_penalty) or field_penalty <= 0):
        raise ValueError("Field penalty must be positive and finite")
    if not 0 <= z_degree <= 3:
        raise ValueError("Z degree must be between 0 and 3")
    if z_degree and len(z_percentiles) < z_degree + 2:
        raise ValueError("Z-dependent fitting requires at least Z degree + 2 sampled planes")
    if z_percentiles and fixed_z is not None:
        raise ValueError("Pass either fixed_z or z_percentiles, not both")
    if any(not np.isfinite(p) or not 0 <= p <= 100 for p in z_percentiles):
        raise ValueError("Z percentiles must be finite values in [0, 100]")
    grid = _fixed_grid(fixed_fused, fixed_z=fixed_z, stride=stride)
    sample_z = [int(np.floor((grid.shape[0] - 1) * p / 100)) for p in sorted(z_percentiles)]
    if z_percentiles and (len(sample_z) < 2 or len(set(sample_z)) != len(sample_z)):
        raise ValueError("Z percentiles must select at least two distinct planes")
    if sample_z:
        grid = replace(grid, z=min(sample_z, key=lambda z: (abs(z - grid.z), z)))
    else:
        sample_z = [grid.z]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".post-basic-", dir=output_dir.parent))
    try:
        channel_results: dict[int, dict[str, object]] = {}
        masks: dict[int, Path] = {}
        fitted_gains: dict[int, dict[Path, float]] = {}
        excluded_sources: dict[int, set[Path]] = {}
        for spec in specs:
            logger.info(
                "Post-BaSiC channel {} ({}) sampling from {}",
                spec.index,
                spec.label,
                spec.registration,
            )
            registration_payload = json.loads(spec.registration.read_text())
            excluded_names = (
                registration_payload.get("metrics", {})
                .get("registration_run", {})
                .get("connectivity", {})
                .get("excluded_tiles", [])
            )
            excluded = {_record_source({"tile": name}, sources) for name in excluded_names}
            registered = {
                _record_source(record, sources) for record in _registration_records(spec.registration)
            }
            if excluded & registered:
                raise ValueError(f"Channel {spec.index} registration includes explicitly excluded sources")
            excluded_sources[spec.index] = excluded
            profile = basic_profiles[spec.index]
            planes = []
            for z in sample_z:
                plane_grid = replace(grid, z=z)
                tiles, sampling = _sample_channel(
                    spec,
                    sources=sources,
                    grid=plane_grid,
                    profile=profile,
                    workers=workers,
                )
                before, owner = _mosaic(grid.output_shape, tiles)
                positive = before[before > 0]
                if positive.size < 2:
                    raise ValueError(f"Channel {spec.index} Z {z} has insufficient positive sampled pixels")
                cutoff = float(np.expm1(threshold_otsu(np.log1p(positive))))
                planes.append(_SampledPlane(plane_grid, tiles, before, owner, cutoff, sampling))
                logger.info("Post-BaSiC channel {} sampled Z {} from {} sources", spec.index, z, len(tiles))
            sampled_sources = sorted({tile.source for plane in planes for tile in plane.tiles})
            source_indices = {source: index for index, source in enumerate(sampled_sources)}
            rows = []
            for plane in planes:
                plane_rows = _seam_samples(plane.tiles, plane.owner, plane.cutoff)
                if not plane_rows:
                    raise ValueError(f"Channel {spec.index} Z {plane.grid.z} has no eligible seams")
                for row in plane_rows:
                    row["first"] = source_indices[plane.tiles[int(row["first"])].source]
                    row["second"] = source_indices[plane.tiles[int(row["second"])].source]
                    row["z"] = plane.grid.z
                rows.extend(plane_rows)
            coefficient, gains, result = fit_residual_model(
                rows=rows,
                sources=sampled_sources,
                seed=seed,
                fit_tile_gains=spec.index in tile_gain_channels,
                z_degree=z_degree,
                xy_degree=xy_degree,
                field_penalty=field_penalty,
            )
            if len(planes) > 1:
                result["z_validation"] = _validate_z_models(
                    rows,
                    sources=sampled_sources,
                    reference_z=grid.z,
                    seed=seed,
                    fit_tile_gains=spec.index in tile_gain_channels,
                    z_degree=z_degree,
                    xy_degree=xy_degree,
                    field_penalty=field_penalty,
                )
            logger.info(
                "Post-BaSiC channel {} fitted {} sources and {} plane/seam pairs",
                spec.index,
                len(sampled_sources),
                len(result["pairs"]),
            )
            preview = next(plane for plane in planes if plane.grid.z == grid.z)
            height, width = profile.flatfield.shape
            yy, xx = np.meshgrid(
                np.linspace(0, 1, height),
                np.linspace(0, 1, width),
                indexing="ij",
            )
            field = np.exp(
                (
                    _basis(
                        np.stack([yy.ravel(), xx.ravel()], axis=1), z=np.full(yy.size, 0.5), z_degree=z_degree
                    )
                    @ coefficient
                ).reshape(height, width)
            ).astype(np.float32)
            if not np.all(np.isfinite(field)) or np.any(field <= 0):
                raise ValueError(f"Channel {spec.index} fitted residual mask is not positive and finite")
            mask_path = stage / f"mask-ch{spec.index}.tif"
            tifffile.imwrite(mask_path, field, compression="zstd")
            masks[spec.index] = mask_path
            if spec.index in tile_gain_channels:
                fitted_gains[spec.index] = {
                    source: float(gain) for source, gain in zip(sampled_sources, gains, strict=True)
                }
                unsupported = result["tile_gain_unestimated_sources"]
                if unsupported:
                    logger.warning(
                        "Post-BaSiC channel {} uses identity gain for {} source(s) without "
                        "non-test seam-pair support",
                        spec.index,
                        len(unsupported),
                    )
            result.update(
                {
                    "label": spec.label,
                    "channel": spec.index,
                    "registration": str(spec.registration.resolve()),
                    "registration_sha256": _sha256(spec.registration),
                    "cutoff": preview.cutoff,
                    "field_range": [float(field.min()), float(field.max())],
                    "mask_raw_z_fraction": 0.5 if z_degree else None,
                    "tile_gains_applied": spec.index in tile_gain_channels,
                    "sampling": preview.sampling,
                    "excluded_sources": sorted(str(path) for path in excluded),
                    "excluded_source_gain_policy": "identity",
                }
            )
            result["planes"] = []
            for plane in planes:
                plane_dir = stage if plane is preview else stage / "planes" / f"z{plane.grid.z}"
                plane_dir.mkdir(parents=True, exist_ok=True)
                corrected = _corrected_tiles(
                    plane.tiles,
                    coefficient,
                    np.asarray([gains[source_indices[tile.source]] for tile in plane.tiles]),
                )
                after, owner = _mosaic(grid.output_shape, corrected)
                if not np.array_equal(plane.owner, owner):
                    raise RuntimeError("Post-BaSiC QC owner map changed during correction")
                artifacts = _write_qc(
                    plane_dir,
                    root=stage,
                    channel=spec.index,
                    before=plane.before,
                    after=after,
                    spacing_yx=np.abs(grid.scale[1:]) * grid.stride,
                )
                if plane is preview:
                    owner_path = stage / f"owner-ch{spec.index}.tif"
                    tifffile.imwrite(owner_path, owner, compression="zstd")
                    artifacts["owner"] = _artifact(owner_path, stage)
                    artifacts["mask"] = _artifact(mask_path, stage)
                    result["artifacts"] = artifacts
                    result["boundaries"] = _boundary_metrics(plane.before, after, owner, plane.cutoff)
                result["planes"].append(
                    {
                        "z": plane.grid.z,
                        "z_um": float(grid.origin[0] + grid.scale[0] * plane.grid.z),
                        "cutoff": plane.cutoff,
                        "sampling": plane.sampling,
                        "artifacts": artifacts,
                    }
                )
            channel_results[spec.index] = result

        all_sources = set(raw_sources)
        for channel in tile_gain_channels:
            required_sources = all_sources - excluded_sources[channel]
            if set(fitted_gains[channel]) != required_sources:
                raise ValueError(
                    f"Tile-gain channel {channel} did not sample every raw source: "
                    f"missing={sorted(str(path) for path in required_sources - set(fitted_gains[channel]))}, "
                    f"unexpected={sorted(str(path) for path in set(fitted_gains[channel]) - required_sources)}"
                )
        gain_matrix = np.ones((len(raw_sources), len(profiles)), dtype=np.float32)
        for channel in tile_gain_channels:
            for row, source in enumerate(raw_sources):
                if source not in excluded_sources[channel]:
                    gain_matrix[row, channel] = fitted_gains[channel][source]
        gains_path = write_tile_gains(stage / "tile-gains.json", sources=raw_sources, gains=gain_matrix)
        logger.info("Post-BaSiC composing {} BaSiC profile(s)", len(profiles))
        profile_artifacts = _copy_or_compose_profiles(
            profiles=profiles,
            basic_profiles=basic_profiles,
            masks=masks,
            results=channel_results,
            output=stage,
        )

        manifest = {
            "schema_version": 1,
            "artifact_type": "squisher_lightsheet.post_basic.v1",
            "status": "complete",
            "fixed_fused": str(fixed_fused),
            "fixed_z": grid.z,
            "sampled_z": sample_z,
            "z_percentiles": sorted(z_percentiles),
            "z_degree": z_degree,
            "xy_degree": xy_degree,
            "field_penalty": field_penalty,
            "fixed_z_um": float(grid.origin[0] + grid.scale[0] * grid.z),
            "fixed_spacing_um": grid.scale.tolist(),
            "fixed_origin_um": grid.origin.tolist(),
            "stride": stride,
            "output_shape_yx": list(grid.output_shape),
            "seed": seed,
            "algorithm": {
                "degree": xy_degree,
                "penalties": list(PENALTIES),
                "folds": FOLDS,
                "seam_radius": SEAM_RADIUS,
                "seam_grid": SEAM_GRID,
                "basic_order": "(raw - darkfield) / flatfield before affine interpolation",
            },
            "inputs": input_snapshot,
            "channel_results": {str(key): value for key, value in channel_results.items()},
            "profiles": profile_artifacts,
            "tile_gains": _artifact(gains_path, stage),
            "correction": "clip((raw - darkfield) / flatfield, 0, inf) * residual_field(raw_zyx) * tile_gain"
            if z_degree
            else "clip((raw - darkfield) / composed_flatfield, 0, inf) * tile_gain",
        }
        logger.info("Post-BaSiC rendering consolidated channel QC")
        write_post_basic_qc(
            stage,
            manifest=manifest,
            flatfields={spec.index: basic_profiles[spec.index].flatfield for spec in specs},
        )
        manifest["qc"] = {
            "artifacts": {path.name: _artifact(path, stage) for path in sorted((stage / "qc").iterdir())},
        }
        if (
            _snapshot_inputs(
                fixed_fused=fixed_fused,
                raw_sources=raw_sources,
                specs=specs,
                profile_identities=_profile_input_identities(basic_dir, profiles),
            )
            != input_snapshot
        ):
            raise RuntimeError("Post-BaSiC inputs changed while the workflow was running")
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        for result in channel_results.values():
            for artifact in result["artifacts"].values():
                if "path" not in artifact:
                    continue
                path = stage / str(artifact["path"])
                if not path.is_file() or _sha256(path) != artifact["sha256"]:
                    raise RuntimeError(f"Post-BaSiC artifact validation failed: {path}")
        stage.replace(output_dir)
        logger.info("Post-BaSiC output complete: {}", output_dir)
        return output_dir / "manifest.json"
    except BaseException:
        shutil.rmtree(stage, ignore_errors=False)
        raise
