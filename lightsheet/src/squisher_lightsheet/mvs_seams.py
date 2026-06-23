from __future__ import annotations

import json
import ast
from collections import defaultdict, deque
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


DEFAULT_MVS_SEAM_QUALITY_THRESHOLD = 0.25
DIMENSIONS = ("z", "y", "x")


def normalize_mvs_edge(edge: Any) -> tuple[int, int]:
    if isinstance(edge, str):
        edge = ast.literal_eval(edge)
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        return tuple(sorted((int(edge[0]), int(edge[1]))))
    raise ValueError(f"Invalid MVS edge identifier: {edge!r}")


def mvs_data_scalar(value: Any) -> float:
    if isinstance(value, dict) and "data" in value:
        data = value["data"]
        while isinstance(data, list):
            if not data:
                return float("nan")
            data = data[0]
        return float(data)
    return float(value)


def mvs_transform_matrix(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        data = np.asarray(value["data"], dtype=float)
        if data.ndim == 3:
            data = data[0]
        return data
    return np.asarray(value, dtype=float)


def _zyx(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([values[dim] for dim in DIMENSIONS], dtype=np.float64)


def tile_image_path(registration_payload: dict[str, Any], tile_record: dict[str, Any]) -> Path:
    tile_path = Path(str(tile_record["tile"]))
    if tile_path.is_absolute() and tile_path.exists():
        return tile_path

    input_dir = Path(str(registration_payload["input_dir"]))
    candidate = input_dir / tile_path
    if candidate.exists():
        return candidate

    source_view = str(tile_record.get("source_view", ""))
    if source_view == "R":
        right_dir = Path(str(input_dir).replace("-CL-", "-CR-").replace("/CL-", "/CR-"))
        candidate = right_dir / tile_path
        if candidate.exists():
            return candidate
    if source_view == "L":
        left_dir = Path(str(input_dir).replace("-CR-", "-CL-").replace("/CR-", "/CL-"))
        candidate = left_dir / tile_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path for {tile_record['tile']} from input_dir={input_dir}")


def _tiff_level_metadata(path: Path, *, level: int) -> tuple[str, int, tuple[int, ...]]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        source_level = min(int(level), max(0, len(series.levels) - 1))
        page_series = series.levels[source_level]
        return str(page_series.axes), source_level, tuple(int(value) for value in page_series.shape)


def _spatial_shape_zyx(*, axes: str, shape: tuple[int, ...]) -> np.ndarray:
    if axes == "CZYX":
        return np.asarray(shape[1:4], dtype=np.int64)
    if axes == "ZYX":
        return np.asarray(shape, dtype=np.int64)
    raise ValueError(f"Expected CZYX or ZYX axes, got {axes!r}")


def _read_tiff_level_crop(
    path: Path,
    *,
    level: int,
    axes: str,
    channel: int,
    slices_zyx: tuple[slice, slice, slice],
) -> np.ndarray:
    import tifffile
    import zarr

    store = tifffile.imread(path, aszarr=True, level=level)
    try:
        zarray = zarr.open(store, mode="r")
        if hasattr(zarray, "keys") and "0" in zarray:
            zarray = zarray["0"]
        if axes == "CZYX":
            crop = zarray[(channel, *slices_zyx)]
        elif axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} requested for single-channel tile {path}")
            crop = zarray[slices_zyx]
        else:
            raise ValueError(f"Expected CZYX or ZYX axes, got {axes!r} for {path}")
        return np.asarray(crop, dtype=np.float32)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _stage_translation_um(tile_record: dict[str, Any]) -> np.ndarray:
    if "stage_translation_um" in tile_record:
        return _zyx(tile_record, "stage_translation_um")
    if "translation_um" in tile_record:
        return _zyx(tile_record, "translation_um")
    return np.zeros(3, dtype=np.float64)


def _registered_affine(tile_record: dict[str, Any]) -> np.ndarray:
    return np.asarray(tile_record["registered_affine"]["matrix"], dtype=np.float64)


def _edge_bbox_center_um(edge: dict[str, Any]) -> np.ndarray | None:
    bbox = edge.get("attrs", {}).get("bbox")
    if not isinstance(bbox, dict) or "data" not in bbox:
        return None
    data = np.asarray(bbox["data"], dtype=np.float64)
    if data.ndim == 3:
        data = data[0]
    if data.shape != (2, 3):
        return None
    return np.mean(data, axis=0)


def _sample_registered_center_patch(
    *,
    registration_payload: dict[str, Any],
    tile_record: dict[str, Any],
    center_zyx_um: np.ndarray,
    spacing_um_zyx: np.ndarray,
    level: int,
    channel: int,
    patch_shape_yx: tuple[int, int],
) -> np.ndarray:
    from scipy import ndimage as scipy_ndimage

    path = tile_image_path(registration_payload, tile_record)
    axes, source_level, source_shape = _tiff_level_metadata(path, level=level)
    shape_zyx = _spatial_shape_zyx(axes=axes, shape=source_shape)
    level_spacing = np.array(spacing_um_zyx, dtype=np.float64, copy=True)
    level_spacing[1:] *= 2**int(source_level)
    height, width = patch_shape_yx
    y_um = center_zyx_um[1] + (np.arange(height, dtype=np.float64) - height // 2) * level_spacing[1]
    x_um = center_zyx_um[2] + (np.arange(width, dtype=np.float64) - width // 2) * level_spacing[2]
    yy, xx = np.meshgrid(y_um, x_um, indexing="ij")
    zz = np.full_like(yy, center_zyx_um[0])
    homogeneous = np.stack([zz, yy, xx, np.ones_like(zz)], axis=0).reshape(4, -1)

    local_input_um = (np.linalg.inv(_registered_affine(tile_record)) @ homogeneous)[:3].T
    local_um = local_input_um - _stage_translation_um(tile_record)
    coords = np.empty_like(local_um)
    coords[:, 0] = local_um[:, 0] / spacing_um_zyx[0]
    coords[:, 1] = local_um[:, 1] / level_spacing[1]
    coords[:, 2] = local_um[:, 2] / level_spacing[2]
    if not np.all(np.isfinite(coords)):
        return np.zeros((height, width), dtype=np.float32)

    lo = np.floor(np.nanmin(coords, axis=0)).astype(int) - 2
    hi = np.ceil(np.nanmax(coords, axis=0)).astype(int) + 3
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape_zyx)
    if np.any(hi <= lo):
        return np.zeros((height, width), dtype=np.float32)

    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lo, hi, strict=True))
    source = _read_tiff_level_crop(path, level=source_level, axes=axes, channel=channel, slices_zyx=slices)
    crop_coords = coords - lo[None, :]
    inside = np.all((coords >= 0.0) & (coords <= (shape_zyx - 1)[None, :]), axis=1)
    sampled = scipy_ndimage.map_coordinates(
        source,
        [crop_coords[:, 0], crop_coords[:, 1], crop_coords[:, 2]],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    sampled[~inside] = 0.0
    return sampled.reshape(height, width).astype(np.float32, copy=False)


def _sample_registered_patch(
    *,
    registration_payload: dict[str, Any],
    tile_record: dict[str, Any],
    center_zyx_um: np.ndarray,
    spacing_um_zyx: np.ndarray,
    level: int,
    channel: int,
    patch_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    from scipy import ndimage as scipy_ndimage

    path = tile_image_path(registration_payload, tile_record)
    axes, source_level, source_shape = _tiff_level_metadata(path, level=level)
    shape_zyx = _spatial_shape_zyx(axes=axes, shape=source_shape)
    level_spacing = np.array(spacing_um_zyx, dtype=np.float64, copy=True)
    level_spacing[1:] *= 2**int(source_level)
    depth, height, width = patch_shape_zyx
    z_um = center_zyx_um[0] + (np.arange(depth, dtype=np.float64) - depth // 2) * level_spacing[0]
    y_um = center_zyx_um[1] + (np.arange(height, dtype=np.float64) - height // 2) * level_spacing[1]
    x_um = center_zyx_um[2] + (np.arange(width, dtype=np.float64) - width // 2) * level_spacing[2]
    zz, yy, xx = np.meshgrid(z_um, y_um, x_um, indexing="ij")
    homogeneous = np.stack([zz, yy, xx, np.ones_like(zz)], axis=0).reshape(4, -1)

    local_input_um = (np.linalg.inv(_registered_affine(tile_record)) @ homogeneous)[:3].T
    local_um = local_input_um - _stage_translation_um(tile_record)
    coords = np.empty_like(local_um)
    coords[:, 0] = local_um[:, 0] / spacing_um_zyx[0]
    coords[:, 1] = local_um[:, 1] / level_spacing[1]
    coords[:, 2] = local_um[:, 2] / level_spacing[2]
    if not np.all(np.isfinite(coords)):
        return np.zeros((depth, height, width), dtype=np.float32)

    lo = np.floor(np.min(coords, axis=0)).astype(int) - 2
    hi = np.ceil(np.max(coords, axis=0)).astype(int) + 3
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape_zyx)
    if np.any(hi <= lo):
        return np.zeros((depth, height, width), dtype=np.float32)

    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lo, hi, strict=True))
    source = _read_tiff_level_crop(path, level=source_level, axes=axes, channel=channel, slices_zyx=slices)
    crop_coords = coords - lo[None, :]
    inside = np.all((coords >= 0.0) & (coords <= (shape_zyx - 1)[None, :]), axis=1)
    sampled = scipy_ndimage.map_coordinates(
        source,
        [crop_coords[:, 0], crop_coords[:, 1], crop_coords[:, 2]],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    sampled[~inside] = 0.0
    return sampled.reshape(depth, height, width).astype(np.float32, copy=False)


def _phase_refine_shift(
    fixed_patch: np.ndarray,
    moving_patch: np.ndarray,
) -> tuple[tuple[float, float, float], float, float, float]:
    import cupy as cp

    from squisher_lightsheet.seams import (
        RobustBoundarySettings,
        center_z_gradient_component_ncc_after_shift,
        content_mask_gpu_array,
        mask_fraction_gpu_array,
        phase_correlation_shift_gpu_arrays,
    )

    settings = RobustBoundarySettings(
        patch_shape_zyx=tuple(int(value) for value in fixed_patch.shape),
        min_content_voxels=1024,
        min_content_fraction=0.001,
    )
    fixed_gpu = cp.asarray(np.asarray(fixed_patch, dtype=np.float32))
    moving_gpu = cp.asarray(np.asarray(moving_patch, dtype=np.float32))
    fixed_mask = content_mask_gpu_array(fixed_gpu, settings)
    moving_mask = content_mask_gpu_array(moving_gpu, settings)
    fixed_content = mask_fraction_gpu_array(fixed_mask)
    moving_content = mask_fraction_gpu_array(moving_mask)
    shift, peak = phase_correlation_shift_gpu_arrays(
        fixed_gpu,
        moving_gpu,
        fixed_mask,
        moving_mask,
        min_mask_voxels=settings.min_content_voxels,
    )
    _gradient_before, gradient_after = center_z_gradient_component_ncc_after_shift(fixed_patch, moving_patch, shift)
    return shift, float(peak), float(gradient_after), float(min(fixed_content, moving_content))


def score_mvs_edges_with_gradient_ncc(
    registration_payload: dict[str, Any],
    *,
    level: int = 2,
    channel: int = 0,
    patch_shape_yx: tuple[int, int] = (256, 256),
    phase_patch_shape_zyx: tuple[int, int, int] = (32, 256, 256),
    min_gradient_ncc: float = 0.15,
    max_phase_shift_native_zyx: tuple[float, float, float] = (16.0, 96.0, 96.0),
    phase_refine_bad_gradients: bool = False,
    used_edges_only: bool = True,
    max_cached_tiles: int = 2,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from squisher_lightsheet.seams import center_z_gradient_component_ncc

    payload = deepcopy(registration_payload)
    tiles = payload["tiles"]
    spacing_um_zyx = np.asarray(
        [
            payload["spacing_um"]["z"],
            payload["spacing_um"]["y"],
            payload["spacing_um"]["x"],
        ],
        dtype=np.float64,
    )
    used_edges = mvs_used_edge_set(payload)

    scored = []
    skipped = []
    for edge in payload["metrics"]["pairwise_registration"]["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        pair = normalize_mvs_edge((source, target))
        if used_edges_only and used_edges and pair not in used_edges:
            continue
        center_um = _edge_bbox_center_um(edge)
        if center_um is None:
            skipped.append({"pair": list(pair), "reason": "missing_bbox"})
            continue
        source_patch = _sample_registered_center_patch(
            registration_payload=payload,
            tile_record=tiles[source],
            center_zyx_um=center_um,
            spacing_um_zyx=spacing_um_zyx,
            level=level,
            channel=channel,
            patch_shape_yx=patch_shape_yx,
        )
        target_patch = _sample_registered_center_patch(
            registration_payload=payload,
            tile_record=tiles[target],
            center_zyx_um=center_um,
            spacing_um_zyx=spacing_um_zyx,
            level=level,
            channel=channel,
            patch_shape_yx=patch_shape_yx,
        )
        score = center_z_gradient_component_ncc(source_patch, target_patch)
        attrs = edge.setdefault("attrs", {})
        attrs["gradient_component_ncc_before_phase"] = None if not np.isfinite(score) else float(score)
        attrs["gradient_component_ncc_source"] = "registered_center_z_patch"
        phase_shift = None
        phase_peak = None
        phase_content = None
        phase_refined = False
        phase_reject_reason = None
        if phase_refine_bad_gradients and (not np.isfinite(score) or float(score) < float(min_gradient_ncc)):
            original_score = score
            fixed_volume = _sample_registered_patch(
                registration_payload=payload,
                tile_record=tiles[source],
                center_zyx_um=center_um,
                spacing_um_zyx=spacing_um_zyx,
                level=level,
                channel=channel,
                patch_shape_zyx=phase_patch_shape_zyx,
            )
            moving_volume = _sample_registered_patch(
                registration_payload=payload,
                tile_record=tiles[target],
                center_zyx_um=center_um,
                spacing_um_zyx=spacing_um_zyx,
                level=level,
                channel=channel,
                patch_shape_zyx=phase_patch_shape_zyx,
            )
            phase_shift, phase_peak, refined_score, phase_content = _phase_refine_shift(fixed_volume, moving_volume)
            phase_refined = True
            if np.isfinite(refined_score):
                phase_shift_native = np.asarray(phase_shift, dtype=np.float64)
                phase_shift_native[1:] *= 2**int(level)
                attrs["phase_refined_shift_level_px_zyx"] = [float(value) for value in phase_shift]
                attrs["phase_refined_shift_native_px_zyx"] = [float(value) for value in phase_shift_native]
                if np.any(np.abs(phase_shift_native) > np.asarray(max_phase_shift_native_zyx, dtype=np.float64)):
                    phase_reject_reason = "phase_shift_out_of_bounds"
                    score = original_score
                else:
                    score = float(refined_score)
                    base_target_delta_px = -mvs_transform_matrix(attrs["transform"])[:3, 3] / spacing_um_zyx
                    attrs["target_correction_delta_px_zyx"] = [
                        float(value) for value in (base_target_delta_px + phase_shift_native)
                    ]
        attrs["gradient_component_ncc_after"] = None if not np.isfinite(score) else float(score)
        attrs["phase_refined_bad_gradient"] = bool(phase_refined)
        attrs["phase_refined_peak"] = phase_peak
        attrs["phase_refined_min_content_fraction"] = phase_content
        attrs["phase_refined_reject_reason"] = phase_reject_reason
        scored.append(
            {
                "source": source,
                "target": target,
                "pair": list(pair),
                "gradient_component_ncc_before_phase": attrs["gradient_component_ncc_before_phase"],
                "gradient_component_ncc_after": None if not np.isfinite(score) else float(score),
                "phase_refined_bad_gradient": bool(phase_refined),
                "phase_refined_shift_level_px_zyx": None if phase_shift is None else [float(value) for value in phase_shift],
                "phase_refined_peak": phase_peak,
                "phase_refined_min_content_fraction": phase_content,
                "phase_refined_reject_reason": phase_reject_reason,
                "mvs_quality": mvs_data_scalar(edge.get("quality", edge.get("attrs", {}).get("quality", float("nan")))),
            }
        )

    summary = {
        "artifact_type": "lightsheet.mvs_seam_gradient_ncc_scoring.v1",
        "level": int(level),
        "channel": int(channel),
        "patch_shape_yx": [int(patch_shape_yx[0]), int(patch_shape_yx[1])],
        "phase_patch_shape_zyx": [int(value) for value in phase_patch_shape_zyx],
        "min_gradient_ncc": float(min_gradient_ncc),
        "max_phase_shift_native_zyx": [float(value) for value in max_phase_shift_native_zyx],
        "phase_refine_bad_gradients": bool(phase_refine_bad_gradients),
        "used_edges_only": bool(used_edges_only),
        "max_cached_tiles": int(max_cached_tiles),
        "scored_edge_count": len(scored),
        "skipped_edge_count": len(skipped),
        "scored_edges": scored,
        "skipped_edges": skipped,
    }
    payload.setdefault("metrics", {})["gradient_ncc_edge_scoring"] = summary
    return payload, summary


def mvs_edge_score(edge: dict[str, Any]) -> tuple[float, str]:
    """Return raw MVS pairwise quality for translation-only seam stitching."""

    return mvs_data_scalar(edge.get("quality", edge.get("attrs", {}).get("quality", float("nan")))), "mvs_quality"


def mvs_used_edge_set(registration_payload: dict[str, Any]) -> set[tuple[int, int]]:
    used_edges = (
        registration_payload.get("metrics", {})
        .get("groupwise_resolution", {})
        .get("metrics", {})
        .get("used_edges", {})
    )
    if isinstance(used_edges, dict):
        used_edges = used_edges.get("0") or used_edges.get(0) or []
    return {normalize_mvs_edge(edge) for edge in used_edges}


def mvs_edge_residuals(registration_payload: dict[str, Any]) -> dict[tuple[int, int], float]:
    residuals = (
        registration_payload.get("metrics", {})
        .get("groupwise_resolution", {})
        .get("metrics", {})
        .get("edge_residuals", {})
    )
    if isinstance(residuals, dict) and ("0" in residuals or 0 in residuals):
        residuals = residuals.get("0") or residuals.get(0) or {}
    if not isinstance(residuals, dict):
        return {}
    return {normalize_mvs_edge(edge): float(value) for edge, value in residuals.items()}


def mvs_measured_edge_records(registration_payload: dict[str, Any]) -> list[dict[str, Any]]:
    edges = (
        registration_payload.get("metrics", {})
        .get("pairwise_registration", {})
        .get("edges", [])
    )
    records = []
    residuals = mvs_edge_residuals(registration_payload)
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        pair = normalize_mvs_edge((source, target))
        quality = mvs_data_scalar(edge.get("quality", edge.get("attrs", {}).get("quality", float("nan"))))
        score, score_source = mvs_edge_score(edge)
        records.append(
            {
                "source": source,
                "target": target,
                "pair": list(pair),
                "quality": quality,
                "score": score,
                "score_source": score_source,
                "residual_um": residuals.get(pair),
            }
        )
    return records


def mvs_used_edge_audit(registration_payload: dict[str, Any]) -> dict[str, Any]:
    measured_records = mvs_measured_edge_records(registration_payload)
    measured = {tuple(record["pair"]) for record in measured_records}
    used = mvs_used_edge_set(registration_payload)
    dropped = measured - used if used else set()
    residuals = mvs_edge_residuals(registration_payload)
    record_by_pair = {tuple(record["pair"]): record for record in measured_records}

    def edge_summary(pair: tuple[int, int]) -> dict[str, Any]:
        record = dict(record_by_pair.get(pair, {"pair": list(pair)}))
        record["pair"] = list(pair)
        record["used"] = pair in used if used else None
        record["residual_um"] = residuals.get(pair)
        return record

    measured_residual_pairs = [pair for pair in measured if pair in residuals]
    used_residual_pairs = [pair for pair in used if pair in residuals]
    max_measured_pair = max(measured_residual_pairs, key=lambda pair: residuals[pair], default=None)
    max_used_pair = max(used_residual_pairs, key=lambda pair: residuals[pair], default=None)

    return {
        "measured_edge_count": len(measured),
        "used_edge_count": len(used),
        "dropped_edge_count": len(dropped),
        "used_edges_present": bool(used),
        "dropped_edges": [edge_summary(pair) for pair in sorted(dropped)],
        "max_measured_residual_edge": None if max_measured_pair is None else edge_summary(max_measured_pair),
        "max_used_residual_edge": None if max_used_pair is None else edge_summary(max_used_pair),
        "measured_edges": [edge_summary(pair) for pair in sorted(measured)],
        "used_edges": [edge_summary(pair) for pair in sorted(used)],
    }


def mvs_pairwise_edges(
    registration_payload: dict[str, Any],
    *,
    min_quality: float = DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    used_edges_only: bool = True,
) -> list[dict[str, Any]]:
    edges = (
        registration_payload.get("metrics", {})
        .get("pairwise_registration", {})
        .get("edges", [])
    )
    used_edges = mvs_used_edge_set(registration_payload)
    filtered = []
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        pair = tuple(sorted((source, target)))
        if used_edges_only and used_edges and pair not in used_edges:
            continue
        quality = mvs_data_scalar(edge.get("quality", edge.get("attrs", {}).get("quality", float("nan"))))
        score, score_source = mvs_edge_score(edge)
        if not np.isfinite(score) or score < min_quality:
            continue
        matrix = mvs_transform_matrix(edge["attrs"]["transform"])
        if matrix.shape[0] < 3 or matrix.shape[1] < 4:
            raise ValueError(f"MVS edge {pair} has invalid transform shape {matrix.shape}")
        filtered.append(
            {
                "source": source,
                "target": target,
                "pair": [source, target],
                "quality": quality,
                "score": score,
                "score_source": score_source,
                "translation_um_zyx": matrix[:3, 3].astype(float),
                "attrs": edge.get("attrs", {}),
            }
        )
    return filtered


def mvs_seam_constraints(
    registration_payload: dict[str, Any],
    *,
    tile_names: list[str],
    spacing_um_zyx: np.ndarray,
    min_quality: float = DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    used_edges_only: bool = True,
) -> list[dict[str, Any]]:
    registration_tiles = [str(record["tile"]) for record in registration_payload["tiles"]]
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    constraints = []
    for edge in mvs_pairwise_edges(
        registration_payload,
        min_quality=min_quality,
        used_edges_only=used_edges_only,
    ):
        fixed_tile = registration_tiles[edge["source"]]
        moving_tile = registration_tiles[edge["target"]]
        if fixed_tile not in tile_index or moving_tile not in tile_index:
            continue
        target_delta_px = -np.asarray(edge["translation_um_zyx"], dtype=float) / spacing_um_zyx
        quality = float(edge["score"])
        constraints.append(
            {
                "fixed": fixed_tile,
                "moving": moving_tile,
                "fixed_index": tile_index[fixed_tile],
                "moving_index": tile_index[moving_tile],
                "pair": edge["pair"],
                "axis": "mvs_pairwise",
                "patch_index": -1,
                "target_correction_delta_px": target_delta_px,
                "corr_after": quality,
                "corr_before": None,
                "weight": max(quality - 0.15, 1e-3),
                "source": "mvs_pairwise_registration",
                "score_source": edge["score_source"],
                "mvs_quality": float(edge["quality"]),
                "mvs_translation_um_zyx": edge["translation_um_zyx"].tolist(),
            }
        )
    if not constraints:
        raise ValueError("No MVS seam constraints passed the quality/tile filters")
    return constraints


def load_mvs_registration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "metrics" not in payload or "pairwise_registration" not in payload.get("metrics", {}):
        raise ValueError(f"{path} is not an MVS registration JSON with pairwise metrics")
    return payload


def recover_anchor_shifts_from_mvs_seams(
    *,
    direct_anchor_shift_um_by_tile: dict[str, np.ndarray],
    mvs_registration: dict[str, Any],
    spacing_um_zyx: np.ndarray,
    min_quality: float = DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    used_edges_only: bool = True,
    max_residual_px_zyx: tuple[float, float, float] = (np.inf, np.inf, np.inf),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tile_names = [str(record["tile"]) for record in mvs_registration["tiles"]]
    direct_tiles = set(direct_anchor_shift_um_by_tile)
    unknown_tiles = [tile for tile in tile_names if tile not in direct_tiles]
    spacing = np.asarray(spacing_um_zyx, dtype=float)
    direct_px = {tile: shift_um / spacing for tile, shift_um in direct_anchor_shift_um_by_tile.items()}
    edge_constraints = mvs_seam_constraints(
        mvs_registration,
        tile_names=tile_names,
        spacing_um_zyx=spacing,
        min_quality=min_quality,
        used_edges_only=used_edges_only,
    )

    adjacency: dict[str, set[str]] = defaultdict(set)
    for constraint in edge_constraints:
        adjacency[constraint["fixed"]].add(constraint["moving"])
        adjacency[constraint["moving"]].add(constraint["fixed"])

    anchored_component_tiles = set(direct_tiles)
    queue = deque(direct_tiles)
    while queue:
        tile = queue.popleft()
        for neighbor in adjacency.get(tile, set()):
            if neighbor not in anchored_component_tiles:
                anchored_component_tiles.add(neighbor)
                queue.append(neighbor)

    solved_unknowns = [tile for tile in unknown_tiles if tile in anchored_component_tiles]
    solved_unknown_index = {tile: index for index, tile in enumerate(solved_unknowns)}
    active_constraints = [
        constraint
        for constraint in edge_constraints
        if constraint["fixed"] in anchored_component_tiles and constraint["moving"] in anchored_component_tiles
    ]

    def tile_shift(tile: str, values: np.ndarray) -> np.ndarray:
        if tile in direct_px:
            return direct_px[tile]
        index = solved_unknown_index[tile]
        return values.reshape(len(solved_unknowns), 3)[index]

    def residual_vector(flat: np.ndarray) -> np.ndarray:
        residuals = []
        for constraint in active_constraints:
            quality = np.sqrt(float(constraint["weight"]))
            residuals.append(
                quality
                * (
                    tile_shift(constraint["moving"], flat)
                    - tile_shift(constraint["fixed"], flat)
                    - constraint["target_correction_delta_px"]
                )
            )
        return np.concatenate(residuals) if residuals else np.zeros(0, dtype=float)

    if solved_unknowns:
        result = least_squares(
            residual_vector,
            np.zeros(len(solved_unknowns) * 3, dtype=float),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=500,
        )
        solved = result.x.reshape(len(solved_unknowns), 3)
    else:
        result = None
        solved = np.zeros((0, 3), dtype=float)

    recovered = {
        tile: solved[index] * spacing
        for tile, index in solved_unknown_index.items()
    }
    max_residual = np.asarray(max_residual_px_zyx, dtype=float)
    residual_records = []
    incident_inlier_count: dict[str, int] = {tile: 0 for tile in solved_unknowns}
    for constraint in active_constraints:
        fixed = constraint["fixed"]
        moving = constraint["moving"]
        residual = (
            tile_shift(moving, solved.reshape(-1))
            - tile_shift(fixed, solved.reshape(-1))
            - constraint["target_correction_delta_px"]
        )
        inlier = bool(np.all(np.abs(residual) <= max_residual))
        for tile in (fixed, moving):
            if tile in incident_inlier_count and inlier:
                incident_inlier_count[tile] += 1
        residual_records.append(
            {
                "fixed": fixed,
                "moving": moving,
                "pair": constraint["pair"],
                "quality": constraint["corr_after"],
                "target_delta_px_zyx": constraint["target_correction_delta_px"].tolist(),
                "residual_px_zyx": residual.tolist(),
                "residual_abs_within_bound_zyx": inlier,
            }
        )

    recovered = {
        tile: shift
        for tile, shift in recovered.items()
        if incident_inlier_count.get(tile, 0) > 0
    }
    diagnostics = {
        "mvs_pairwise_edge_count": len(edge_constraints),
        "active_edge_count": len(active_constraints),
        "direct_anchor_count": len(direct_tiles),
        "unknown_tile_count": len(unknown_tiles),
        "recovered_tile_count": len(recovered),
        "unrecovered_tiles": [tile for tile in unknown_tiles if tile not in recovered],
        "min_quality": min_quality,
        "used_edges_only": used_edges_only,
        "optimizer": None
        if result is None
        else {
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
        },
        "residuals": residual_records,
        "incident_inlier_count_by_tile": incident_inlier_count,
    }
    return recovered, diagnostics
