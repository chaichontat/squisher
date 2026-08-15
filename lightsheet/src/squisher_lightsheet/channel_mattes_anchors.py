from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
import itertools
import json
from pathlib import Path
import re
from typing import Any, Literal

from loguru import logger
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as rough_legacy
from squisher_lightsheet.tiff import choose_tiff_source_level


DIMENSIONS = ("z", "y", "x")
DEFAULT_LEVEL = 3
DEFAULT_PATCH_SHAPE_ZYX = (12, 160, 160)
DEFAULT_REFERENCE_CHANNEL = 3
DEFAULT_MOVING_CHANNEL = 0
DEFAULT_MAX_ATTEMPTED_CHUNKS = 24
DEFAULT_MIN_INLIER_CHUNKS = 3
DEFAULT_MAX_INLIER_CHUNKS = 5
DEFAULT_MIN_GOOD_MIP_GRADIENT_NCC = 0.15
DEFAULT_HIGH_FREQUENCY_CONTENT_SIGMA_ZYX = (0.0, 3.0, 3.0)
DEFAULT_MIN_HIGH_FREQUENCY_CONTENT_SCORE = 0.001
DEFAULT_CANDIDATE_POOL_SIZE = 36
DEFAULT_EDGE_INSET_FRACTION = 0.0
DEFAULT_INLIER_THRESHOLD_LEVEL_PX_ZYX = (3.0, 12.0, 12.0)
PHASECORR_FFT_HIGHPASS_SIGMA_ZYX = (1.0, 4.0, 1.0)
CandidateSource = Literal["seed_rows", "structure_tensor"]


@dataclass(frozen=True)
class RoundInput:
    label: str
    position_path: Path


@dataclass(frozen=True)
class TilePose:
    round_label: str
    site: str
    tile: rough_legacy.TileRecord
    record: dict[str, Any]

    @property
    def translation_um(self) -> np.ndarray:
        return self.tile.translation_zyx_um

    @property
    def scale_um(self) -> np.ndarray:
        return self.tile.scale_zyx_um

    @property
    def spacing_um(self) -> np.ndarray:
        return np.abs(self.tile.scale_zyx_um)

    @property
    def shape_zyx(self) -> np.ndarray:
        return self.tile.shape_zyx


@dataclass(frozen=True)
class RoundTiles:
    label: str
    position_path: Path
    payload: dict[str, Any]
    poses: dict[str, TilePose]


@dataclass(frozen=True)
class MattesAnchorParameters:
    candidate_source: CandidateSource = "seed_rows"
    level: int = DEFAULT_LEVEL
    patch_shape_zyx: tuple[int, int, int] = DEFAULT_PATCH_SHAPE_ZYX
    reference_channel: int = DEFAULT_REFERENCE_CHANNEL
    moving_channel: int = DEFAULT_MOVING_CHANNEL
    max_attempted_chunks: int = DEFAULT_MAX_ATTEMPTED_CHUNKS
    min_inlier_chunks: int = DEFAULT_MIN_INLIER_CHUNKS
    max_inlier_chunks: int = DEFAULT_MAX_INLIER_CHUNKS
    min_good_mip_gradient_ncc: float = DEFAULT_MIN_GOOD_MIP_GRADIENT_NCC
    high_frequency_content_sigma_zyx: tuple[float, float, float] = DEFAULT_HIGH_FREQUENCY_CONTENT_SIGMA_ZYX
    min_high_frequency_content_score: float = DEFAULT_MIN_HIGH_FREQUENCY_CONTENT_SCORE
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE
    edge_inset_fraction: float = DEFAULT_EDGE_INSET_FRACTION
    inlier_threshold_level_px_zyx: tuple[float, float, float] = DEFAULT_INLIER_THRESHOLD_LEVEL_PX_ZYX


def log(message: str) -> None:
    logger.info(message)


def require_gpu_modules() -> tuple[Any, Any, Any]:
    try:
        import cupy as cp
        import cucim.skimage.feature as cu_feature
        from cupyx.scipy import ndimage as cpx_ndimage
    except ImportError as exc:
        raise RuntimeError("Stage-3 Mattes anchor candidate generation requires CuPy and cuCIM") from exc
    return cp, cu_feature, cpx_ndimage


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def content_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "measured"
        and float(row.get("fixed_content_score") or 0.0) > 0.0
        and float(row.get("moving_content_score") or 0.0) > 0.0
    ]


def load_position_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def tile_site(record: dict[str, Any]) -> str:
    side = str(record.get("side") or "")
    if side not in {"L", "R"}:
        name = str(record.get("tile") or Path(str(record.get("path", ""))).name)
        if "-CL-" in name:
            side = "L"
        elif "-CR-" in name:
            side = "R"
        else:
            side = "?"
    name = str(record.get("tile") or Path(str(record.get("path", ""))).name)
    matches = re.findall(r"\.(\d+)\.ome\.tif$", name)
    if matches:
        index = int(matches[-1])
    else:
        numeric = re.findall(r"(\d+)", Path(name).stem)
        index = int(numeric[-1]) if numeric else 0
    return f"{side}:{index:04d}"


def load_round(round_input: RoundInput) -> RoundTiles:
    path = round_input.position_path.resolve()
    payload = load_position_payload(path)
    tiles = rough_legacy.load_tiles(payload)
    records_by_name = {str(record["tile"]): record for record in payload["tiles"]}
    records_by_path_name = {Path(str(record["path"])).name: record for record in payload["tiles"]}
    poses: dict[str, TilePose] = {}
    for tile in tiles:
        record = records_by_name.get(tile.tile) or records_by_path_name.get(tile.path.name)
        if record is None:
            raise ValueError(f"{path} has no position record for {tile.path.name}")
        site = tile_site(record)
        if site in poses:
            raise ValueError(f"{path} has duplicate physical site {site}")
        poses[site] = TilePose(round_label=round_input.label, site=site, tile=tile, record=record)
    return RoundTiles(label=round_input.label, position_path=path, payload=payload, poses=poses)


def dict_zyx(values: np.ndarray | list[float] | tuple[float, float, float]) -> dict[str, float]:
    return {dim: float(value) for dim, value in zip(DIMENSIONS, values, strict=True)}


def canonicalize_round_geometry(reference: RoundTiles, moving: RoundTiles) -> RoundTiles:
    poses: dict[str, TilePose] = {}
    for site, pose in moving.poses.items():
        if site not in reference.poses:
            poses[site] = pose
            continue
        ref_translation = dict_zyx(reference.poses[site].translation_um)
        tile = replace(pose.tile, translation_zyx_um=np.asarray([ref_translation[dim] for dim in DIMENSIONS]))
        record = dict(pose.record)
        record["translation_um"] = ref_translation
        poses[site] = TilePose(round_label=pose.round_label, site=site, tile=tile, record=record)
    return RoundTiles(label=moving.label, position_path=moving.position_path, payload=moving.payload, poses=poses)


def tile_bounds_um(pose: TilePose) -> tuple[np.ndarray, np.ndarray]:
    start = pose.translation_um
    stop = start + pose.shape_zyx.astype(np.float64) * pose.scale_um
    return np.minimum(start, stop), np.maximum(start, stop)


def close_store(store: Any) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        close()


_TILE_ARRAY_CACHE: dict[tuple[Path, int, int], tuple[Any, Any]] = {}


def close_cached_tile_arrays() -> None:
    for _array, store in _TILE_ARRAY_CACHE.values():
        close_store(store)
    _TILE_ARRAY_CACHE.clear()


def tile_channel_array(path: Path, channel: int, *, source_level: int = 0) -> tuple[Any, Any]:
    import tifffile
    import zarr

    store = tifffile.imread(path, aszarr=True, level=source_level)
    zarray = rough_legacy.base_zarr_array(zarr.open(store, mode="r"))
    if zarray.ndim == 4:
        if channel < 0 or channel >= int(zarray.shape[0]):
            close_store(store)
            raise ValueError(f"{path} channel {channel} out of range for shape {zarray.shape}")
        return zarray[channel], store
    if zarray.ndim == 3:
        if channel != 0:
            close_store(store)
            raise ValueError(f"{path} is single-channel but channel {channel} was requested")
        return zarray, store
    close_store(store)
    raise ValueError(f"Expected 3D or 4D TIFF array for {path}, got {zarray.shape}")


def cached_tile_channel_array(path: Path, channel: int, *, source_level: int) -> Any:
    key = (Path(path).resolve(), int(channel), int(source_level))
    if key not in _TILE_ARRAY_CACHE:
        _TILE_ARRAY_CACHE[key] = tile_channel_array(key[0], key[1], source_level=key[2])
    return _TILE_ARRAY_CACHE[key][0]


def content_score(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size < 128:
        return 0.0
    p50, p995 = np.percentile(sample, [50.0, 99.5])
    denom = max(float(np.percentile(sample, 99.9)), 1.0)
    return float(max(p995 - p50, 0.0) / denom)


def sample_world_patch(
    pose: TilePose,
    *,
    channel: int,
    world_origin_um: np.ndarray,
    output_shape: np.ndarray,
    output_spacing_um: np.ndarray,
    extra_translation_um: np.ndarray | None = None,
) -> np.ndarray:
    desired_factors = np.maximum(np.rint(np.abs(output_spacing_um / pose.spacing_um)).astype(np.int64), 1)
    source_level, source_factors = choose_tiff_source_level(pose.tile.path, desired_factors)
    array = cached_tile_channel_array(pose.tile.path, channel, source_level=source_level)
    z_world = world_origin_um[0] + np.arange(int(output_shape[0]), dtype=np.float64) * output_spacing_um[0]
    y_world = world_origin_um[1] + np.arange(int(output_shape[1]), dtype=np.float64) * output_spacing_um[1]
    x_world = world_origin_um[2] + np.arange(int(output_shape[2]), dtype=np.float64) * output_spacing_um[2]
    extra = np.zeros(3, dtype=np.float64) if extra_translation_um is None else np.asarray(extra_translation_um, dtype=np.float64)
    z_coord = (z_world - extra[0] - pose.translation_um[0]) / (pose.scale_um[0] * source_factors[0])
    y_coord = (y_world - extra[1] - pose.translation_um[1]) / (pose.scale_um[1] * source_factors[1])
    x_coord = (x_world - extra[2] - pose.translation_um[2]) / (pose.scale_um[2] * source_factors[2])

    y_min = int(max(np.floor(y_coord.min() - 2), 0))
    y_max = int(min(np.ceil(y_coord.max() + 3), array.shape[1]))
    x_min = int(max(np.floor(x_coord.min() - 2), 0))
    x_max = int(min(np.ceil(x_coord.max() + 3), array.shape[2]))
    if y_max <= y_min or x_max <= x_min:
        return np.zeros(tuple(int(v) for v in output_shape), dtype=np.float32)

    z_planes = sorted(
        {
            plane
            for value in z_coord
            for plane in (int(np.floor(value)), int(np.floor(value)) + 1)
            if 0 <= plane < array.shape[0]
        }
    )
    if not z_planes:
        return np.zeros(tuple(int(v) for v in output_shape), dtype=np.float32)
    source_by_plane = {
        plane: np.asarray(array[plane, y_min:y_max, x_min:x_max], dtype=np.float32) for plane in z_planes
    }
    yy, xx = np.meshgrid(y_coord - y_min, x_coord - x_min, indexing="ij")
    coords_yx = np.asarray([yy, xx])
    out = np.zeros(tuple(int(v) for v in output_shape), dtype=np.float32)
    for out_z, src_z in enumerate(z_coord):
        z0 = int(np.floor(src_z))
        z1 = z0 + 1
        wz = float(src_z - z0)
        plane0 = source_by_plane.get(z0)
        plane1 = source_by_plane.get(z1)
        if plane0 is None and plane1 is None:
            continue
        if plane0 is None:
            interpolated = plane1 * wz
        elif plane1 is None:
            interpolated = plane0 * (1.0 - wz)
        else:
            interpolated = plane0 * (1.0 - wz) + plane1 * wz
        out[out_z] = ndimage.map_coordinates(interpolated, coords_yx, order=1, mode="constant", cval=0.0)
    return out


def normalized_gpu(values: np.ndarray) -> Any:
    cp, _cu_feature, _cpx_ndimage = require_gpu_modules()
    finite = values[np.isfinite(values)]
    if finite.size < 128:
        return cp.zeros(values.shape, dtype=cp.float32)
    low, high = np.percentile(finite, [1.0, 99.8])
    gpu = cp.asarray(values, dtype=cp.float32)
    gpu = cp.clip((gpu - cp.float32(low)) / cp.float32(max(float(high - low), 1.0)), 0.0, 1.0)
    return cp.sqrt(gpu)


def highpass_gpu(values: np.ndarray, sigma_zyx: tuple[float, float, float]) -> Any:
    _cp, _cu_feature, cpx_ndimage = require_gpu_modules()
    gpu = normalized_gpu(values)
    if any(float(value) > 0.0 for value in sigma_zyx):
        gpu = gpu - cpx_ndimage.gaussian_filter(gpu, sigma=tuple(float(value) for value in sigma_zyx))
    return gpu


def high_frequency_content_score(
    values: np.ndarray,
    *,
    high_frequency_content_sigma_zyx: tuple[float, float, float] = DEFAULT_HIGH_FREQUENCY_CONTENT_SIGMA_ZYX,
) -> float:
    cp, _cu_feature, _cpx_ndimage = require_gpu_modules()
    highpass = highpass_gpu(values.astype(np.float32, copy=False), high_frequency_content_sigma_zyx)
    score = float(cp.mean(highpass * highpass).get())
    del highpass
    cp.get_default_memory_pool().free_all_blocks()
    return score


def structure_tensor_response_gpu(highpass: Any) -> Any:
    _cp, cu_feature, _cpx_ndimage = require_gpu_modules()
    tensor = cu_feature.structure_tensor(highpass, sigma=1.0)
    return cu_feature.structure_tensor_eigenvalues(tensor)[-1]


def highpass_candidate_centers_for_site(
    site: str,
    reference: RoundTiles,
    *,
    parameters: MattesAnchorParameters,
) -> list[dict[str, Any]]:
    cp, cu_feature, _cpx_ndimage = require_gpu_modules()
    pose = reference.poses[site]
    patch_shape_zyx = np.asarray(parameters.patch_shape_zyx, dtype=np.int64)
    desired_factor = np.repeat(2**parameters.level, 3)
    source_level, source_factors = choose_tiff_source_level(pose.tile.path, desired_factor)
    source_factors = np.asarray(source_factors, dtype=np.float64)
    source_spacing = np.abs(pose.scale_um * source_factors)
    level_spacing = pose.spacing_um * (2**parameters.level)
    patch_size_um = patch_shape_zyx.astype(np.float64) * level_spacing
    patch_shape_source = np.maximum(np.rint(patch_size_um / source_spacing).astype(np.int64), 1)

    start, stop = tile_bounds_um(pose)
    inset_um = parameters.edge_inset_fraction * (stop - start)
    usable_start = start + inset_um + patch_size_um / 2.0
    usable_stop = stop - inset_um - patch_size_um / 2.0
    usable_stop = np.maximum(usable_start, usable_stop)
    coord_a = (usable_start - pose.translation_um) / (pose.scale_um * source_factors)
    coord_b = (usable_stop - pose.translation_um) / (pose.scale_um * source_factors)
    coord_min = np.maximum(np.floor(np.minimum(coord_a, coord_b)).astype(np.int64), 0)
    reference_array = cached_tile_channel_array(
        pose.tile.path,
        parameters.reference_channel,
        source_level=source_level,
    )
    coord_max = np.minimum(
        np.ceil(np.maximum(coord_a, coord_b)).astype(np.int64) + 1,
        np.asarray(reference_array.shape),
    )
    if np.any(coord_max <= coord_min):
        return []

    crop = np.asarray(
        reference_array[
            coord_min[0] : coord_max[0],
            coord_min[1] : coord_max[1],
            coord_min[2] : coord_max[2],
        ],
        dtype=np.float32,
    )
    sigma_source = tuple(
        float(sigma * level_spacing[index] / source_spacing[index])
        for index, sigma in enumerate(parameters.high_frequency_content_sigma_zyx)
    )
    highpass = highpass_gpu(crop, sigma_source)
    score_map_gpu = structure_tensor_response_gpu(highpass)
    footprint = cp.ones(tuple(int(max(value // 4, 3)) for value in patch_shape_source), dtype=cp.bool_)
    coords_gpu = cu_feature.corner_peaks(
        score_map_gpu,
        min_distance=1,
        threshold_rel=0.0,
        exclude_border=False,
        footprint=footprint,
        num_peaks=parameters.candidate_pool_size,
    )
    score_map = cp.asnumpy(score_map_gpu)
    coords = cp.asnumpy(coords_gpu)
    del highpass, score_map_gpu, coords_gpu, footprint
    cp.get_default_memory_pool().free_all_blocks()

    records = []
    for local_coord in coords:
        coord = np.asarray(local_coord, dtype=np.int64) + coord_min
        center_um = pose.translation_um + coord.astype(np.float64) * pose.scale_um * source_factors
        origin = np.clip(center_um - patch_size_um / 2.0, start + inset_um, stop - inset_um - patch_size_um)
        patch = sample_world_patch(
            pose,
            channel=parameters.reference_channel,
            world_origin_um=origin,
            output_shape=patch_shape_zyx,
            output_spacing_um=level_spacing,
        )
        records.append(
            {
                "center_um_zyx": (origin + patch_size_um / 2.0).astype(float).tolist(),
                "structure_tensor_score": float(score_map[tuple(int(value) for value in local_coord)]),
                "high_frequency_content_score": high_frequency_content_score(
                    patch,
                    high_frequency_content_sigma_zyx=parameters.high_frequency_content_sigma_zyx,
                ),
                "content_score": float(content_score(patch)),
            }
        )
    records.sort(
        key=lambda record: (
            float(record["structure_tensor_score"]),
            float(record["high_frequency_content_score"]),
        ),
        reverse=True,
    )
    return records[: parameters.candidate_pool_size]


def row_center_um(row: dict[str, Any]) -> np.ndarray:
    origin = np.asarray(row["world_origin_um_zyx"], dtype=np.float64)
    shape = np.asarray(row["patch_shape_zyx"], dtype=np.float64)
    spacing = np.asarray(row["level_spacing_um_zyx"], dtype=np.float64)
    return origin + shape * spacing / 2.0


def candidate_rows_for_site(
    site: str,
    existing_rows: list[dict[str, Any]],
    *,
    reference: RoundTiles,
    parameters: MattesAnchorParameters,
    candidate_metrics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    spacing = reference.poses[site].spacing_um * (2**parameters.level)
    patch_shape_zyx = np.asarray(parameters.patch_shape_zyx, dtype=np.int64)
    patch_size_um = patch_shape_zyx.astype(np.float64) * spacing
    site_bootstrap = site_bootstrap_translation(existing_rows)
    if candidate_metrics is None and parameters.candidate_source == "seed_rows":
        rows = []
        for index, source_row in enumerate(
            sorted(existing_rows, key=lambda row: int(row.get("block_index", row.get("candidate_index", 0))))[
                : parameters.max_attempted_chunks
            ]
        ):
            row = dict(source_row)
            row.setdefault("chunk_index", int(row.get("block_index", index)))
            row.setdefault("candidate_index", int(row.get("block_index", index)))
            row.setdefault("reference_channel", parameters.reference_channel)
            row.setdefault("moving_channel", parameters.moving_channel)
            row.setdefault("level", parameters.level)
            row.setdefault("patch_shape_zyx", patch_shape_zyx.astype(int).tolist())
            row.setdefault("level_spacing_um_zyx", spacing.astype(float).tolist())
            row.setdefault("bootstrap_translation_um_zyx", site_bootstrap.astype(float).tolist())
            row.setdefault("anchor_prior_offset_um_zyx", [0.0, 0.0, 0.0])
            if row.get("applied_translation_um_zyx") is None:
                bootstrap = np.asarray(row["bootstrap_translation_um_zyx"], dtype=np.float64)
                prior = np.asarray(row["anchor_prior_offset_um_zyx"], dtype=np.float64)
                row["applied_translation_um_zyx"] = (bootstrap + prior).astype(float).tolist()
            row.setdefault("source", "seed_records_jsonl")
            rows.append(row)
        return rows

    if candidate_metrics is None:
        candidate_metrics = highpass_candidate_centers_for_site(site, reference, parameters=parameters)
    existing_by_center = [
        (row_center_um(row), row) for row in existing_rows if row.get("world_origin_um_zyx") is not None
    ]
    rows = []
    for index, record in enumerate(candidate_metrics[: parameters.max_attempted_chunks]):
        center = np.asarray(record["center_um_zyx"], dtype=np.float64)
        origin = center - patch_size_um / 2.0
        row: dict[str, Any] = {
            "tile_site": site,
            "block_index": int(index),
            "chunk_index": int(index),
            "candidate_index": int(index),
            "reference_channel": parameters.reference_channel,
            "moving_channel": parameters.moving_channel,
            "level": parameters.level,
            "patch_shape_zyx": patch_shape_zyx.astype(int).tolist(),
            "level_spacing_um_zyx": spacing.astype(float).tolist(),
            "world_origin_um_zyx": origin.astype(float).tolist(),
            "reference_content_score": float(record["content_score"]),
            "reference_structure_tensor_score": float(record["structure_tensor_score"]),
            "reference_high_frequency_content_score": float(record["high_frequency_content_score"]),
            "bootstrap_translation_um_zyx": site_bootstrap.astype(float).tolist(),
            "anchor_prior_offset_um_zyx": [0.0, 0.0, 0.0],
            "applied_translation_um_zyx": site_bootstrap.astype(float).tolist(),
            "source": "generated_gpu_highpass_structure_tensor_candidate",
        }
        for existing_center, existing in existing_by_center:
            if np.allclose(existing_center, center, atol=float(np.max(spacing)) * 0.5):
                source_anchor_block_index = existing.get("block_index")
                row.update(existing)
                row["block_index"] = int(index)
                row["chunk_index"] = int(index)
                row["candidate_index"] = int(index)
                row["source_anchor_block_index"] = source_anchor_block_index
                row["source"] = "generated_gpu_highpass_structure_tensor_candidate_matched_existing_anchor"
                break
        rows.append(row)
    return rows


def site_bootstrap_translation(existing_rows: list[dict[str, Any]]) -> np.ndarray:
    return next(
        (
            np.asarray(row["bootstrap_translation_um_zyx"], dtype=np.float64)
            for row in existing_rows
            if row.get("bootstrap_translation_um_zyx") is not None
        ),
        np.zeros(3, dtype=np.float64),
    )


def start_shift_from_row(row: dict[str, Any]) -> np.ndarray | None:
    for key in ("phasecorr_direct_shift_level_vox_zyx", "phasecorr_init_shift_level_vox_zyx"):
        if row.get(key) is not None:
            return np.asarray(row[key], dtype=np.float64)
    return None


def starts_for_row(row: dict[str, Any], known_good: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = []
    phase = start_shift_from_row(row)
    if phase is not None:
        starts.append({"name": "block_phasecorr", "shift_to_apply_moving_zyx_level3_px": phase.tolist()})
    residual = row.get("residual_shift_level_vox_zyx")
    if residual is not None:
        starts.append({"name": "block_previous_residual", "shift_to_apply_moving_zyx_level3_px": residual})
    neighbor_shifts = []
    for other in known_good:
        shift = np.asarray(other["best_by_selection_metric"]["shift_to_apply_moving_zyx_level3_px"], dtype=np.float64)
        neighbor_shifts.append(shift)
        starts.append(
            {
                "name": f"good_chunk{int(other['block_index'])}_best_shift",
                "shift_to_apply_moving_zyx_level3_px": shift.astype(float).tolist(),
            }
        )
    if neighbor_shifts:
        starts.append(
            {
                "name": "good_chunk_median",
                "shift_to_apply_moving_zyx_level3_px": np.median(np.asarray(neighbor_shifts), axis=0).tolist(),
            }
        )
    unique: list[dict[str, Any]] = []
    seen = set()
    for start in starts:
        shift = np.asarray(start["shift_to_apply_moving_zyx_level3_px"], dtype=np.float64)
        rounded = tuple(np.round(shift, 3))
        if rounded in seen:
            continue
        seen.add(rounded)
        unique.append(start)
    return unique


def complete_linkage_inliers(records: list[dict[str, Any]], *, parameters: MattesAnchorParameters) -> list[int]:
    good = [
        record
        for record in records
        if record.get("best_by_selection_metric") is not None
        and record["best_by_selection_metric"].get("mip_gradient_component_ncc_moving") is not None
        and float(record["best_by_selection_metric"]["mip_gradient_component_ncc_moving"])
        >= parameters.min_good_mip_gradient_ncc
    ]
    if len(good) < parameters.min_inlier_chunks:
        return []
    shifts = [
        np.asarray(record["best_by_selection_metric"]["shift_to_apply_moving_zyx_level3_px"], dtype=np.float64)
        for record in good
    ]
    thresholds = np.asarray(parameters.inlier_threshold_level_px_zyx, dtype=np.float64)
    max_size = max(parameters.min_inlier_chunks, parameters.max_inlier_chunks)
    for size in range(min(len(good), max_size), parameters.min_inlier_chunks - 1, -1):
        for combo in itertools.combinations(range(len(good)), size):
            if all(
                np.all(np.abs(shifts[a] - shifts[b]) <= thresholds)
                for a, b in itertools.combinations(combo, 2)
            ):
                return [int(good[index]["block_index"]) for index in combo]
    return []


def median_anchor_from_inlier_blocks(
    records: list[dict[str, Any]],
    inlier_blocks: list[int],
) -> dict[str, Any] | None:
    inlier_block_set = {int(value) for value in inlier_blocks}
    chunks = []
    shifts = []
    for record in records:
        block_index = int(record.get("block_index", -1))
        if block_index not in inlier_block_set:
            continue
        best = record.get("best_by_selection_metric") or {}
        shift = best.get("shift_to_apply_moving_zyx_level3_px")
        if shift is None:
            continue
        shift_zyx = np.asarray(shift, dtype=np.float64)
        if shift_zyx.shape != (3,) or not np.all(np.isfinite(shift_zyx)):
            continue
        shifts.append(shift_zyx)
        chunks.append(
            {
                "block_index": block_index,
                "shift_to_apply_moving_zyx_level3_px": shift_zyx.astype(float).tolist(),
                "mip_gradient_component_ncc_moving": best.get("mip_gradient_component_ncc_moving"),
                "metric": best.get("metric"),
                "optimizer": best.get("optimizer"),
            }
        )
    if not shifts:
        return None
    return {
        "shift_to_apply_moving_zyx_level3_px": np.median(np.asarray(shifts, dtype=np.float64), axis=0)
        .astype(float)
        .tolist(),
        "inlier_chunk_count": len(chunks),
        "inlier_chunks": chunks,
        "aggregation": "median",
    }


def robust_norm(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size == 0:
        return np.zeros(volume.shape, dtype=np.float32)
    low, high = np.percentile(sample, [1.0, 99.7])
    return np.clip((volume - low) / max(float(high - low), 1.0), 0.0, 1.0).astype(np.float32)


def signal_mask(volume: np.ndarray) -> np.ndarray:
    norm = robust_norm(volume)
    positive = norm[norm > 0]
    if positive.size == 0:
        return np.zeros(norm.shape, dtype=bool)
    mask = norm > max(0.03, float(np.percentile(positive, 45.0)))
    mask = ndimage.binary_dilation(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)


def ncc_values(a: np.ndarray, b: np.ndarray, mask: np.ndarray, *, min_pixels: int = 256) -> float:
    if int(np.count_nonzero(mask)) < min_pixels:
        return float("nan")
    aa = a[mask].astype(np.float64, copy=False)
    bb = b[mask].astype(np.float64, copy=False)
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom else float("nan")


def mip_gradient_component_ncc_moving(fixed: np.ndarray, moving: np.ndarray) -> float:
    fixed_mip = robust_norm(np.max(fixed, axis=0))
    moving_mip = robust_norm(np.max(moving, axis=0))
    moving_mask = ndimage.binary_dilation(signal_mask(moving_mip), iterations=2)
    values = []
    for axis in range(2):
        value = ncc_values(
            ndimage.sobel(fixed_mip, axis=axis),
            ndimage.sobel(moving_mip, axis=axis),
            moving_mask,
        )
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def make_image(volume: np.ndarray) -> Any:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def make_mask(mask: np.ndarray, reference: Any) -> Any:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
    image.CopyInformation(reference)
    return image


def configure_metric(method: Any, name: str) -> None:
    if name == "mattes_mi":
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    else:
        raise ValueError(name)


def configure_optimizer(method: Any, name: str) -> None:
    if name == "regular_step":
        method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=6.0,
            minStep=0.05,
            numberOfIterations=50,
            relaxationFactor=0.5,
            gradientMagnitudeTolerance=1e-6,
        )
    else:
        raise ValueError(name)
    method.SetOptimizerScalesFromPhysicalShift()


def resample(moving_image: Any, fixed_image: Any, transform: Any) -> np.ndarray:
    import SimpleITK as sitk

    out = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out).astype(np.float32, copy=False)


def fit_translation(
    fixed: np.ndarray,
    moving: np.ndarray,
    spacing_zyx_um: np.ndarray,
    *,
    metric_name: str,
    optimizer_name: str,
    start: dict[str, Any],
) -> dict[str, Any]:
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    fixed_norm = robust_norm(fixed)
    moving_norm = robust_norm(moving)
    start_shift_zyx_px = np.asarray(start["shift_to_apply_moving_zyx_level2_px"], dtype=np.float64)
    moving_start_norm = ndimage.shift(moving_norm, shift=start_shift_zyx_px, order=1, mode="constant", cval=0.0)
    fixed_image = make_image(fixed_norm)
    moving_image = make_image(moving_start_norm)
    fixed_mask = make_mask(signal_mask(fixed), fixed_image)
    method = sitk.ImageRegistrationMethod()
    configure_metric(method, metric_name)
    method.SetMetricFixedMask(fixed_mask)
    method.SetMetricSamplingStrategy(method.REGULAR)
    method.SetMetricSamplingPercentage(0.35)
    method.SetInterpolator(sitk.sitkLinear)
    configure_optimizer(method, optimizer_name)
    method.SetShrinkFactorsPerLevel([2, 1])
    method.SetSmoothingSigmasPerLevel([1.0, 0.0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(sitk.TranslationTransform(3), inPlace=False)
    try:
        transform = method.Execute(fixed_image, moving_image)
        params_xyz_px = np.asarray(transform.GetParameters(), dtype=np.float64)
        residual_shift_to_apply_moving_zyx_px = -params_xyz_px[::-1]
        shift_to_apply_moving_zyx_px = start_shift_zyx_px + residual_shift_to_apply_moving_zyx_px
        moved = resample(moving_image, fixed_image, transform)
        status = "measured"
        error = None
    except RuntimeError as exc:
        params_xyz_px = np.zeros(3, dtype=np.float64)
        residual_shift_to_apply_moving_zyx_px = np.zeros(3, dtype=np.float64)
        shift_to_apply_moving_zyx_px = start_shift_zyx_px
        moved = moving_start_norm
        status = "failed"
        error = str(exc)
    qc_score = mip_gradient_component_ncc_moving(fixed_norm, moved) if status == "measured" else float("nan")
    return {
        "metric": metric_name,
        "optimizer": optimizer_name,
        "start": start["name"],
        "status": status,
        "error": error,
        "metric_value": float(method.GetMetricValue()) if status == "measured" else None,
        "mip_gradient_component_ncc_moving": None if not np.isfinite(qc_score) else float(qc_score),
        "stop": method.GetOptimizerStopConditionDescription() if status == "measured" else None,
        "iterations": int(method.GetOptimizerIteration()) if status == "measured" else None,
        "initial_shift_to_apply_moving_zyx_level2_px": start_shift_zyx_px.tolist(),
        "initial_shift_to_apply_moving_um_zyx": (start_shift_zyx_px * spacing_zyx_um).astype(float).tolist(),
        "residual_transform_parameters_xyz_level2_px": params_xyz_px.tolist(),
        "residual_shift_to_apply_moving_zyx_level2_px": residual_shift_to_apply_moving_zyx_px.tolist(),
        "residual_shift_to_apply_moving_um_zyx": (
            residual_shift_to_apply_moving_zyx_px * spacing_zyx_um
        ).astype(float).tolist(),
        "transform_parameters_xyz_level2_px": (-shift_to_apply_moving_zyx_px[::-1]).astype(float).tolist(),
        "shift_to_apply_moving_zyx_level2_px": shift_to_apply_moving_zyx_px.tolist(),
        "shift_to_apply_moving_um_zyx": (shift_to_apply_moving_zyx_px * spacing_zyx_um).astype(float).tolist(),
        "equivalent_fixed_green_shift_zyx_level2_px": (-shift_to_apply_moving_zyx_px).tolist(),
        "_moved": moved,
    }


def preprocess_gpu(volume: np.ndarray, *, spatial_highpass_sigma: float | None = 4.0) -> Any:
    cp, _cu_feature, cpx_ndimage = require_gpu_modules()
    gpu = cp.asarray(volume, dtype=cp.float32)
    finite = volume[np.isfinite(volume)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size == 0:
        return cp.zeros_like(gpu)
    low, high = np.percentile(sample, [1.0, 99.8])
    gpu = cp.clip((gpu - cp.float32(low)) / cp.float32(max(float(high - low), 1.0)), 0.0, 1.0)
    gpu = cp.sqrt(gpu)
    if spatial_highpass_sigma is not None and spatial_highpass_sigma > 0.0:
        gpu = gpu - cpx_ndimage.gaussian_filter(gpu, sigma=float(spatial_highpass_sigma))
    return gpu


def subpixel_peak(corr: Any, peak: tuple[int, int, int]) -> np.ndarray:
    values = []
    shape = corr.shape
    for axis, center in enumerate(peak):
        if center == 0 or center == shape[axis] - 1:
            values.append(0.0)
            continue
        selector = list(peak)
        selector[axis] = center - 1
        left = float(corr[tuple(selector)].get())
        selector[axis] = center
        middle = float(corr[tuple(selector)].get())
        selector[axis] = center + 1
        right = float(corr[tuple(selector)].get())
        denom = left - 2.0 * middle + right
        values.append(0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom)
    return np.asarray(values, dtype=np.float64)


def fft_gaussian_highpass_weight(shape: tuple[int, ...], sigma_zyx: tuple[float, float, float]) -> Any:
    cp, _cu_feature, _cpx_ndimage = require_gpu_modules()
    exponent = None
    for axis, (size, sigma) in enumerate(zip(shape, sigma_zyx, strict=True)):
        axis_shape = [1] * len(shape)
        axis_shape[axis] = int(size)
        freq = cp.fft.fftfreq(int(size)).reshape(axis_shape)
        term = (2.0 * np.pi**2 * float(sigma) ** 2) * (freq * freq)
        exponent = term if exponent is None else exponent + term
    if exponent is None:
        return cp.float32(1.0)
    return (1.0 - cp.exp(-exponent)).astype(cp.float32, copy=False)


def phasecorr_shift_gpu(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    fft_highpass_sigma_zyx: tuple[float, float, float] | None = None,
    spatial_highpass_sigma: float | None = 4.0,
    search_center_zyx: np.ndarray | None = None,
    max_shift_from_center_zyx: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cp, _cu_feature, _cpx_ndimage = require_gpu_modules()
    fixed_gpu = preprocess_gpu(fixed, spatial_highpass_sigma=spatial_highpass_sigma)
    moving_gpu = preprocess_gpu(moving, spatial_highpass_sigma=spatial_highpass_sigma)
    fixed_gpu = fixed_gpu - cp.mean(fixed_gpu)
    moving_gpu = moving_gpu - cp.mean(moving_gpu)
    product = cp.fft.fftn(fixed_gpu) * cp.conj(cp.fft.fftn(moving_gpu))
    product /= cp.maximum(cp.abs(product), cp.float32(1e-7))
    if fft_highpass_sigma_zyx is not None and any(float(value) > 0.0 for value in fft_highpass_sigma_zyx):
        product *= fft_gaussian_highpass_weight(product.shape, fft_highpass_sigma_zyx)
    corr = cp.real(cp.fft.ifftn(product))
    if (search_center_zyx is None) != (max_shift_from_center_zyx is None):
        raise ValueError(
            "search_center_zyx and max_shift_from_center_zyx must be provided together"
        )
    bounded_search = search_center_zyx is not None
    search_corr = corr
    if bounded_search:
        center = np.asarray(search_center_zyx, dtype=np.float64)
        radius = np.asarray(max_shift_from_center_zyx, dtype=np.float64)
        if (
            center.shape != (3,)
            or radius.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(radius))
            or np.any(radius < 0)
        ):
            raise ValueError(
                "Phase-correlation search center must be a finite 3-vector and radius "
                "must be a finite nonnegative 3-vector"
            )
        valid = cp.ones(corr.shape, dtype=cp.bool_)
        for axis, size in enumerate(corr.shape):
            indices = cp.arange(size, dtype=cp.float64)
            signed = cp.where(indices <= np.fix(size / 2), indices, indices - size)
            axis_valid = cp.abs(signed - float(center[axis])) <= float(radius[axis])
            axis_shape = [1] * corr.ndim
            axis_shape[axis] = size
            valid &= axis_valid.reshape(axis_shape)
        if not bool(cp.any(valid).get()):
            raise ValueError("Phase-correlation search window contains no candidate shifts")
        search_corr = cp.where(valid, corr, -cp.inf)
        del valid
    peak_flat = int(cp.argmax(search_corr).get())
    peak = np.asarray(np.unravel_index(peak_flat, search_corr.shape), dtype=np.int64)
    subpixel = subpixel_peak(corr, tuple(int(v) for v in peak))
    if bounded_search:
        for axis, (index, size) in enumerate(zip(peak, corr.shape, strict=True)):
            neighbor_indices = ((int(index) - 1) % size, (int(index) + 1) % size)
            neighbor_shifts = np.asarray(
                [value if value <= np.fix(size / 2) else value - size for value in neighbor_indices],
                dtype=np.float64,
            )
            if np.any(
                np.abs(neighbor_shifts - center[axis]) > radius[axis]
            ):
                subpixel[axis] = 0.0
    shifts = peak.astype(np.float64) + subpixel
    midpoint = np.asarray([np.fix(axis_size / 2) for axis_size in corr.shape], dtype=np.float64)
    shape = np.asarray(corr.shape, dtype=np.float64)
    shifts[shifts > midpoint] -= shape[shifts > midpoint]
    peak_value = float(corr[tuple(int(v) for v in peak)].get())
    metadata: dict[str, Any] = {
        "peak_value": peak_value,
        "phasecorr_fft_highpass_sigma_zyx": list(fft_highpass_sigma_zyx) if fft_highpass_sigma_zyx is not None else None,
        "phasecorr_spatial_highpass_sigma": spatial_highpass_sigma,
        "search_center_zyx": None if search_center_zyx is None else center.tolist(),
        "max_shift_from_center_zyx": (
            None if max_shift_from_center_zyx is None else radius.tolist()
        ),
    }
    del fixed_gpu, moving_gpu, product, corr, search_corr
    cp.get_default_memory_pool().free_all_blocks()
    return shifts, metadata


def live_phasecorr_start(
    fixed: np.ndarray,
    moving: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shift, metadata = phasecorr_shift_gpu(
        fixed,
        moving,
        fft_highpass_sigma_zyx=PHASECORR_FFT_HIGHPASS_SIGMA_ZYX,
        spatial_highpass_sigma=None,
    )
    shift = np.asarray(shift, dtype=np.float64)
    fixed_norm = robust_norm(fixed)
    moving_norm = robust_norm(moving)
    shifted_norm = ndimage.shift(moving_norm, shift=shift, order=1, mode="constant", cval=0.0)
    score = mip_gradient_component_ncc_moving(fixed_norm, shifted_norm)
    start = {
        "name": "live_phasecorr_fft_highpass_1_4_1",
        "shift_to_apply_moving_zyx_level2_px": shift.tolist(),
        "phasecorr_metadata": metadata,
    }
    direct_result = {
        "metric": "phasecorr_direct",
        "optimizer": "none",
        "start": "live_phasecorr_fft_highpass_1_4_1",
        "status": "measured",
        "error": None,
        "metric_value": None,
        "mip_gradient_component_ncc_moving": None if not np.isfinite(score) else float(score),
        "stop": "unrefined phase-correlation candidate",
        "iterations": 0,
        "initial_shift_to_apply_moving_zyx_level2_px": shift.tolist(),
        "initial_shift_to_apply_moving_um_zyx": (shift * spacing_zyx_um).astype(float).tolist(),
        "transform_parameters_xyz_level2_px": (-shift[::-1]).astype(float).tolist(),
        "shift_to_apply_moving_zyx_level2_px": shift.tolist(),
        "shift_to_apply_moving_um_zyx": (shift * spacing_zyx_um).astype(float).tolist(),
        "equivalent_fixed_green_shift_zyx_level2_px": (-shift).astype(float).tolist(),
        "phasecorr_metadata": metadata,
        "_moved": shifted_norm,
    }
    return start, direct_result


def to_fit_start(start: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": start["name"],
        "shift_to_apply_moving_zyx_level2_px": start["shift_to_apply_moving_zyx_level3_px"],
    }


def phasecorr_direct_result(
    fixed: np.ndarray,
    moving: np.ndarray,
    spacing_zyx_um: np.ndarray,
    shift_zyx: np.ndarray,
) -> dict[str, Any]:
    fixed_norm = robust_norm(fixed)
    moving_norm = robust_norm(moving)
    shifted_norm = ndimage.shift(moving_norm, shift=shift_zyx, order=1, mode="constant", cval=0.0)
    score = mip_gradient_component_ncc_moving(fixed_norm, shifted_norm)
    return {
        "metric": "phasecorr_direct",
        "optimizer": "none",
        "start": "block_phasecorr",
        "status": "measured",
        "error": None,
        "mip_gradient_component_ncc_moving": None if not np.isfinite(score) else float(score),
        "shift_to_apply_moving_zyx_level3_px": shift_zyx.tolist(),
        "shift_to_apply_moving_um_zyx": (shift_zyx * spacing_zyx_um).astype(float).tolist(),
        "_moved": shifted_norm,
    }


def rename_fit_result(result: dict[str, Any]) -> dict[str, Any]:
    renamed = dict(result)
    for old_key in (
        "initial_shift_to_apply_moving_zyx_level2_px",
        "residual_shift_to_apply_moving_zyx_level2_px",
        "shift_to_apply_moving_zyx_level2_px",
    ):
        if old_key in renamed:
            renamed[old_key.replace("level2", "level3")] = renamed.pop(old_key)
    return renamed


def sample_row(
    row: dict[str, Any],
    *,
    reference: RoundTiles,
    moving: RoundTiles,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    site = row["tile_site"]
    origin = np.asarray(row["world_origin_um_zyx"], dtype=np.float64)
    shape = np.asarray(row["patch_shape_zyx"], dtype=np.int64)
    spacing = np.asarray(row["level_spacing_um_zyx"], dtype=np.float64)
    fixed = sample_world_patch(
        reference.poses[site],
        channel=int(row["reference_channel"]),
        world_origin_um=origin,
        output_shape=shape,
        output_spacing_um=spacing,
    )
    extra_translation_um = np.asarray(row.get("applied_translation_um_zyx", [0.0, 0.0, 0.0]), dtype=np.float64)
    moving_patch = sample_world_patch(
        moving.poses[site],
        channel=int(row["moving_channel"]),
        world_origin_um=origin,
        output_shape=shape,
        output_spacing_um=spacing,
        extra_translation_um=extra_translation_um,
    )
    return fixed, moving_patch, spacing


def add_total_shift_fields(result: dict[str, Any], row: dict[str, Any]) -> None:
    residual_um = result.get("shift_to_apply_moving_um_zyx")
    if residual_um is None:
        return
    applied = np.asarray(row.get("applied_translation_um_zyx", [0.0, 0.0, 0.0]), dtype=np.float64)
    total = applied + np.asarray(residual_um, dtype=np.float64)
    result["applied_translation_um_zyx"] = applied.astype(float).tolist()
    result["total_shift_to_apply_moving_um_zyx"] = total.astype(float).tolist()


def overlay(fixed: np.ndarray, moving: np.ndarray) -> Image.Image:
    green = (robust_norm(np.max(fixed, axis=0)) * 255).astype(np.uint8)
    red = (robust_norm(np.max(moving, axis=0)) * 255).astype(np.uint8)
    rgb = np.zeros((*green.shape, 3), dtype=np.uint8)
    rgb[..., 0] = red
    rgb[..., 1] = green
    return Image.fromarray(rgb)


def write_site_sheet(path: Path, site: str, panels: list[dict[str, Any]]) -> None:
    panel_w = 220
    panel_h = 220
    title_h = 58
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (columns * panel_w, rows * (panel_h + title_h) + 24), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), site, fill=(255, 255, 255))
    for index, panel in enumerate(panels):
        row = index // columns
        column = index % columns
        x0 = column * panel_w
        y0 = 24 + row * (panel_h + title_h)
        image = overlay(panel["fixed"], panel["_moved"])
        image.thumbnail((panel_w, panel_h), Image.Resampling.BILINEAR)
        shift = np.asarray(panel["shift_to_apply_moving_zyx_level3_px"], dtype=np.float64)
        score = panel.get("mip_gradient_component_ncc_moving")
        draw.text((x0 + 4, y0 + 4), f"b{panel['block_index']} {panel['metric']}", fill=(255, 255, 255))
        draw.text((x0 + 4, y0 + 22), f"{panel.get('start', '')}", fill=(215, 215, 215))
        draw.text((x0 + 4, y0 + 40), f"s={np.round(shift, 1).tolist()} g={score}", fill=(215, 215, 215))
        canvas.paste(image, (x0, y0 + title_h))
    canvas.save(path)


def summarize_site(
    site: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    reference: RoundTiles,
    moving_round: RoundTiles,
    parameters: MattesAnchorParameters,
) -> dict[str, Any]:
    site_records = []
    panels = []
    known_good: list[dict[str, Any]] = []
    candidate_rows = candidate_rows_for_site(site, rows, reference=reference, parameters=parameters)
    inlier_blocks: list[int] = []
    for row in candidate_rows:
        fixed, moving, spacing = sample_row(row, reference=reference, moving=moving_round)
        fixed_score = content_score(fixed)
        moving_score = content_score(moving)
        fixed_high_frequency_score = high_frequency_content_score(
            fixed,
            high_frequency_content_sigma_zyx=parameters.high_frequency_content_sigma_zyx,
        )
        moving_high_frequency_score = high_frequency_content_score(
            moving,
            high_frequency_content_sigma_zyx=parameters.high_frequency_content_sigma_zyx,
        )
        row["fixed_content_score"] = float(fixed_score)
        row["moving_content_score"] = float(moving_score)
        row["fixed_high_frequency_content_score"] = float(fixed_high_frequency_score)
        row["moving_high_frequency_content_score"] = float(moving_high_frequency_score)
        row["high_frequency_content_sigma_zyx"] = [float(value) for value in parameters.high_frequency_content_sigma_zyx]
        if fixed_score <= 0.0 or moving_score <= 0.0:
            site_records.append(
                {
                    "block_index": int(row["block_index"]),
                    "source_record": row,
                    "candidate_starts": [],
                    "best_by_selection_metric": None,
                    "results": [],
                    "status": "rejected",
                    "reason": "low_content",
                }
            )
            continue
        if (
            fixed_high_frequency_score < parameters.min_high_frequency_content_score
            or moving_high_frequency_score < parameters.min_high_frequency_content_score
        ):
            site_records.append(
                {
                    "block_index": int(row["block_index"]),
                    "source_record": row,
                    "candidate_starts": [],
                    "best_by_selection_metric": None,
                    "results": [],
                    "status": "rejected",
                    "reason": "low_high_frequency_content",
                }
            )
            continue
        if start_shift_from_row(row) is None:
            phase_start, _phase_direct = live_phasecorr_start(fixed, moving, spacing)
            row["phasecorr_direct_shift_level_vox_zyx"] = phase_start["shift_to_apply_moving_zyx_level2_px"]
            row["phasecorr_metadata"] = phase_start.get("phasecorr_metadata")
        starts = starts_for_row(row, known_good)
        results = []
        phase = start_shift_from_row(row)
        if phase is not None:
            direct = phasecorr_direct_result(fixed, moving, spacing, phase)
            add_total_shift_fields(direct, row)
            results.append(direct)
        for start in starts:
            logger.info("running site={} block={} start={}", site, row["block_index"], start["name"])
            result = fit_translation(
                fixed,
                moving,
                spacing,
                metric_name="mattes_mi",
                optimizer_name="regular_step",
                start=to_fit_start(start),
            )
            renamed = rename_fit_result(result)
            add_total_shift_fields(renamed, row)
            results.append(renamed)
        ranked = ranked_measured_results(results)
        best = None if not ranked else ranked[0]
        if best is not None:
            panels.append({"fixed": fixed, "block_index": int(row["block_index"]), **best})
        for result in results:
            result.pop("_moved", None)
        site_records.append(
            {
                "block_index": int(row["block_index"]),
                "source_record": row,
                "candidate_starts": starts,
                "best_by_selection_metric": best,
                "results": results,
                "status": "measured" if best is not None else "rejected",
                "reason": None if best is not None else "no_finite_candidate",
            }
        )
        latest = site_records[-1]
        if (
            best is not None
            and best.get("mip_gradient_component_ncc_moving") is not None
            and float(best["mip_gradient_component_ncc_moving"]) >= parameters.min_good_mip_gradient_ncc
        ):
            known_good.append(latest)
        inlier_blocks = complete_linkage_inliers(site_records, parameters=parameters)
        if len(inlier_blocks) >= max(parameters.min_inlier_chunks, parameters.max_inlier_chunks):
            break
    sheet = output_dir / "site_sheets" / f"{site.replace(':', '-')}_mattes_phasecorr_starts.png"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    write_site_sheet(sheet, site, panels)
    attempted = len(site_records)
    accepted = len(inlier_blocks) >= parameters.min_inlier_chunks
    summary = {
        "tile_site": site,
        "status": "accepted" if accepted else "rejected",
        "reason": None if accepted else "insufficient_inlier_chunks",
        "attempted_chunks": int(attempted),
        "max_attempted_chunks": parameters.max_attempted_chunks,
        "min_inlier_chunks": parameters.min_inlier_chunks,
        "max_inlier_chunks": parameters.max_inlier_chunks,
        "inlier_chunk_blocks": inlier_blocks,
        "accepted_anchor": median_anchor_from_inlier_blocks(site_records, inlier_blocks) if accepted else None,
        "good_chunk_count": int(len(known_good)),
        "blocks": site_records,
        "contact_sheet": str(sheet),
    }
    site_json = output_dir / "site_json" / f"{site.replace(':', '-')}_mattes_phasecorr_starts.json"
    site_json.parent.mkdir(parents=True, exist_ok=True)
    site_json.write_text(json.dumps(summary, indent=2) + "\n")
    summary["site_json"] = str(site_json)
    return summary


def ranked_measured_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            result
            for result in results
            if result.get("status") == "measured" and result.get("mip_gradient_component_ncc_moving") is not None
        ],
        key=lambda item: item["mip_gradient_component_ncc_moving"],
        reverse=True,
    )


def measure_405_to_488_mattes_anchors(
    *,
    records_jsonl: Path,
    reference_position: Path,
    moving_position: Path,
    output_dir: Path,
    reference_label: str = "488",
    moving_label: str = "405",
    parameters: MattesAnchorParameters = MattesAnchorParameters(),
    preserve_moving_geometry: bool = True,
) -> Path:
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(records_jsonl)
    rows_by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_site[str(row["tile_site"])].append(row)
    sites = sorted(rows_by_site)
    reference = load_round(RoundInput(reference_label, reference_position))
    moving_loaded = load_round(RoundInput(moving_label, moving_position))
    moving_round = moving_loaded if preserve_moving_geometry else canonicalize_round_geometry(reference, moving_loaded)
    try:
        summaries = [
            summarize_site(
                site,
                rows_by_site[site],
                output_dir,
                reference=reference,
                moving_round=moving_round,
                parameters=parameters,
            )
            for site in sites
        ]
        payload = {
            "created_at": datetime.now().isoformat(),
            "artifact_type": "lightsheet.405_to_488.stage3_mattes_anchors.v1",
            "derived_by": "squisher_lightsheet.channel_mattes_anchors.measure_405_to_488_mattes_anchors",
            "records": str(records_jsonl),
            "reference_position": str(reference_position),
            "moving_position": str(moving_position),
            "preserve_moving_geometry": bool(preserve_moving_geometry),
            "output_dir": str(output_dir),
            "included_sites": sites,
            "included_rows": len(rows),
            "skipped_rows": [
                {
                    "tile_site": row.get("tile_site"),
                    "block_index": row.get("block_index"),
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                }
                for row in rows
                if row not in content_rows(rows)
            ],
            "metric": "mattes_mi",
            "optimizer": "regular_step",
            "selection_metric": "mip_gradient_component_ncc_moving",
            "start_policy": (
                "no zero start; phasecorr direct plus residual refinement from all same-site "
                "block phasecorr shifts, all same-site previous residual shifts, and same-site median"
            ),
            "attempt_policy": (
                "accept once min_inlier_chunks complete-linkage inlier measurements are found, "
                "then continue until max_inlier_chunks inliers or the maximum chunk-attempt budget"
            ),
            "min_inlier_chunks": parameters.min_inlier_chunks,
            "max_inlier_chunks": parameters.max_inlier_chunks,
            "max_attempted_chunks": parameters.max_attempted_chunks,
            "candidate_pool_size": parameters.candidate_pool_size,
            "candidate_source": parameters.candidate_source,
            "edge_inset_fraction": parameters.edge_inset_fraction,
            "candidate_detector": "cucim.structure_tensor_eigenvalues_smallest_after_gpu_highpass",
            "min_good_mip_gradient_component_ncc_moving": parameters.min_good_mip_gradient_ncc,
            "high_frequency_content_sigma_zyx": [
                float(value) for value in parameters.high_frequency_content_sigma_zyx
            ],
            "min_high_frequency_content_score": parameters.min_high_frequency_content_score,
            "inlier_threshold_level_px_zyx": [
                float(value) for value in parameters.inlier_threshold_level_px_zyx
            ],
            "sites": summaries,
        }
        output = output_dir / "all_tiles_mattes_phasecorr_starts.json"
        output.write_text(json.dumps(payload, indent=2) + "\n")
        return output
    finally:
        close_cached_tile_arrays()
