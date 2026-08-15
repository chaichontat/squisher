from __future__ import annotations

import json
import re
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from squisher_lightsheet import phase_metrics
from squisher_lightsheet.artifacts import stamp_artifact
from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as rough_legacy
from squisher_lightsheet.tile_phase import (
    DIMENSIONS,
    _rgb_overlay,
    corrcoef_on_mask,
    corresponding_moving_path,
    make_moving_tile_name,
    make_moving_tile_record,
    read_tile_indexed_z_patch,
    sampled_tile_volume_from_subifd,
    tile_record_from_position_record,
)


AffineMode = Literal["affine-12dof"]
AffineFitMode = Literal["rigid", "affine-12dof"]
AffineTileOrder = Literal["input", "grid-fanout"]
NATIVE_PIVOT_MIN_WINDOW_STD = 1e-6
NATIVE_PIVOT_MIN_ACTIVE_FRACTION = 1e-5
CHANNEL_PHASE_UPSAMPLE_FACTOR = 10
METHOD8_JOINT_LOW_CORR_THRESHOLD = 0.15
METHOD8_JOINT_LOW_GRADIENT_NCC_THRESHOLD = 0.05
METHOD8_JOINT_LOW_DIRECT_PRIOR_WEIGHT = 0.1


class AffineMeasurementRejected(RuntimeError):
    """Expected per-tile rejection caused by insufficient image evidence or quality gates."""


class RegistrationTransformContract(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_: Literal["squisher_lightsheet.registration_transform_contract.v1"] = Field(
        "squisher_lightsheet.registration_transform_contract.v1",
        alias="schema",
    )
    registered_affine_semantics: Literal["moving_tile_stage_um_to_reference_registered_um"]
    source_space: str
    target_space: str
    composition_order: tuple[
        Literal[
            "reference_registered_affine",
            "reference_stage_translation_um",
            "moving_to_reference_channel_affine_um",
            "inverse_moving_stage_translation_um",
        ],
        ...,
    ]
    registered_affine_contains_full_channel_affine: bool
    stage_translation_source: Literal["moving_registration_input", "reference_registration_input"]


def _translation_matrix_zyx_um(translation_zyx_um: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation_zyx_um, dtype=np.float64)
    return matrix


def _record_vector_zyx(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([values[dim] for dim in DIMENSIONS], dtype=np.float64)


def _record_stage_translation_um(record: dict[str, Any]) -> np.ndarray:
    if "stage_translation_um" in record:
        return _record_vector_zyx(record, "stage_translation_um")
    if "translation_um" in record:
        return _record_vector_zyx(record, "translation_um")
    return np.zeros(3, dtype=np.float64)


def _record_registered_affine_um(record: dict[str, Any]) -> np.ndarray:
    registered = record.get("registered_affine", {}).get("matrix")
    if registered is None:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(registered, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"registered_affine.matrix must be 4x4, got {matrix.shape}")
    return matrix


def _robust_norm(volume: np.ndarray) -> np.ndarray:
    return phase_metrics.robust_normalize(volume).astype(np.float32, copy=False)


def _corr(fixed: np.ndarray, moving: np.ndarray) -> float | None:
    return corrcoef_on_mask(fixed, moving, np.isfinite(fixed) & np.isfinite(moving) & (moving != 0))


def _fused_masked_ncc_module() -> Any:
    import cupy as cp

    if not hasattr(_fused_masked_ncc_module, "module"):
        _fused_masked_ncc_module.module = cp.RawModule(
            code=r'''
extern "C" __global__ void masked_ncc_partials(
    const float* fixed,
    const float* moving,
    const bool* fixed_mask,
    const bool use_fixed_mask,
    const long total,
    double* partials
) {
    extern __shared__ double shared[];
    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;
    const int lanes = 6;
    for (int i = tid; i < lanes; i += block_threads) {
        shared[i] = 0.0;
    }
    __syncthreads();

    double local[6];
    for (int i = 0; i < lanes; ++i) {
        local[i] = 0.0;
    }

    for (long idx = (long)blockIdx.x * block_threads + tid; idx < total; idx += (long)gridDim.x * block_threads) {
        if (use_fixed_mask && !fixed_mask[idx]) {
            continue;
        }
        const float f = fixed[idx];
        const float m = moving[idx];
        if (!isfinite(f) || !isfinite(m) || m == 0.0f) {
            continue;
        }
        const double fd = (double)f;
        const double md = (double)m;
        local[0] += 1.0;
        local[1] += fd;
        local[2] += md;
        local[3] += fd * fd;
        local[4] += md * md;
        local[5] += fd * md;
    }

    for (int i = 0; i < lanes; ++i) {
        atomicAdd(&shared[i], local[i]);
    }
    __syncthreads();
    for (int i = tid; i < lanes; i += block_threads) {
        partials[(long)blockIdx.x * lanes + i] = shared[i];
    }
}
''',
        )
    return _fused_masked_ncc_module.module


def _ncc_from_sums(values: np.ndarray, *, min_count: int) -> float | None:
    count, fixed_sum, moving_sum, fixed_sq_sum, moving_sq_sum, cross_sum = values
    if count < min_count:
        return None
    count64 = float(count)
    fixed_centered_sq = fixed_sq_sum - fixed_sum * fixed_sum / count64
    moving_centered_sq = moving_sq_sum - moving_sum * moving_sum / count64
    if fixed_centered_sq <= 0.0 or moving_centered_sq <= 0.0:
        return None
    cross_centered = cross_sum - fixed_sum * moving_sum / count64
    denominator = np.sqrt(fixed_centered_sq * moving_centered_sq)
    return float(cross_centered / denominator)


def _corr_gpu(fixed: Any, moving: Any, fixed_mask: Any | None = None) -> float | None:
    import cupy as cp

    fixed_gpu = cp.ascontiguousarray(cp.asarray(fixed, dtype=cp.float32))
    moving_gpu = cp.ascontiguousarray(cp.asarray(moving, dtype=cp.float32))
    if fixed_gpu.shape != moving_gpu.shape:
        raise ValueError(f"fixed and moving shapes differ: {fixed_gpu.shape} vs {moving_gpu.shape}")
    use_fixed_mask = fixed_mask is not None
    if fixed_mask is None:
        fixed_mask_gpu = cp.ones((1,), dtype=cp.bool_)
    else:
        fixed_mask_gpu = cp.ascontiguousarray(cp.asarray(fixed_mask, dtype=cp.bool_))
        if fixed_mask_gpu.shape != fixed_gpu.shape:
            raise ValueError(f"fixed_mask shape differs: {fixed_mask_gpu.shape} vs {fixed_gpu.shape}")
    total = int(fixed_gpu.size)
    threads = 256
    blocks = max(1, min(4096, (total + threads - 1) // threads))
    partials = cp.zeros((blocks, 6), dtype=cp.float64)
    kernel = _fused_masked_ncc_module().get_function("masked_ncc_partials")
    kernel(
        (blocks,),
        (threads,),
        (fixed_gpu, moving_gpu, fixed_mask_gpu, np.bool_(use_fixed_mask), np.int64(total), partials),
        shared_mem=6 * 8,
    )
    return _ncc_from_sums(cp.asnumpy(cp.sum(partials, axis=0)), min_count=8)


def _finite_metric(value: float | None) -> float:
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _gradient_component_ncc_mean(metric: dict[str, Any] | None) -> float:
    if not isinstance(metric, dict):
        return float("nan")
    value = metric.get("mean")
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _safe_json_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).removesuffix(".ome.zarr").removesuffix(".ome.tif")


def _canonical_tile_filter_entry(token: str, *, reference_token: str) -> str:
    value = token.strip()
    if value.isdigit():
        return f"{reference_token}.{int(value):03d}.ome.zarr"
    if value.endswith(".ome.zarr"):
        return value
    parts = value.split(".")
    if len(parts) >= 2 and parts[-1].isdigit():
        return f"{'.'.join(parts[:-1])}.{int(parts[-1]):03d}.ome.zarr"
    raise ValueError(f"Cannot parse tile filter entry {token!r}")


def _native_window_content_stats(volume: Any) -> dict[str, float]:
    try:
        import cupy as cp
    except ImportError:
        cp = None
    if cp is not None and isinstance(volume, cp.ndarray):
        finite = cp.isfinite(volume)
        if not bool(cp.asnumpy(cp.any(finite))):
            return {"std": 0.0, "active_fraction": 0.0}
        finite_volume = volume[finite].astype(cp.float32, copy=False)
        return {
            "std": float(cp.asnumpy(cp.std(finite_volume))),
            "active_fraction": float(cp.asnumpy(cp.count_nonzero(cp.abs(finite_volume) > 1e-6) / finite_volume.size)),
        }

    finite = np.isfinite(volume)
    if not bool(np.any(finite)):
        return {"std": 0.0, "active_fraction": 0.0}
    finite_volume = np.asarray(volume[finite], dtype=np.float32)
    return {
        "std": float(np.std(finite_volume)),
        "active_fraction": float(np.count_nonzero(np.abs(finite_volume) > 1e-6) / finite_volume.size),
    }


def _robust_norm_and_content_stats_cupy(volume: Any) -> tuple[Any, dict[str, float]]:
    import cupy as cp

    image = cp.asarray(volume, dtype=cp.float32)
    finite = cp.isfinite(image)
    positive = image[finite & (image > 0)]
    if positive.size == 0:
        return cp.zeros(image.shape, dtype=cp.float32), {"std": 0.0, "active_fraction": 0.0}

    low, high = cp.percentile(positive, cp.asarray([1.0, 99.5], dtype=cp.float32))
    clipped = cp.clip(image, low, high)
    valid = cp.isfinite(clipped)
    centered = clipped - cp.median(clipped[valid])
    denom = cp.maximum(cp.percentile(cp.abs(centered[valid]), 95.0), cp.float32(1.0))
    normalized = cp.where(valid, centered / denom, cp.float32(0.0))

    finite_normalized = normalized[valid]
    stats = {
        "std": float(cp.asnumpy(cp.std(finite_normalized))),
        "active_fraction": float(cp.asnumpy(cp.count_nonzero(cp.abs(finite_normalized) > 1e-6) / finite_normalized.size)),
    }
    return cp.ascontiguousarray(normalized.astype(cp.float32, copy=False)), stats


def _robust_norm_and_content_stats_gpu(volume: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    import cupy as cp

    normalized, stats = _robust_norm_and_content_stats_cupy(volume)
    return cp.asnumpy(normalized).astype(np.float32, copy=False), stats


def _native_window_low_content_reason(stats: dict[str, float], *, prefix: str) -> str | None:
    if stats["std"] <= NATIVE_PIVOT_MIN_WINDOW_STD:
        return f"{prefix}_constant"
    if stats["active_fraction"] < NATIVE_PIVOT_MIN_ACTIVE_FRACTION:
        return f"{prefix}_low_active_fraction"
    return None


def affine_quality_gates(
    *,
    matrix: np.ndarray,
    corr_initial: float | None,
    corr_refined: float | None,
    gradient_initial: dict[str, Any],
    gradient_refined: dict[str, Any],
) -> dict[str, Any]:
    determinant = float(np.linalg.det(np.asarray(matrix, dtype=np.float64)))
    condition = float(np.linalg.cond(np.asarray(matrix, dtype=np.float64)))
    corr_improvement = _finite_metric(corr_refined) - _finite_metric(corr_initial)
    gradient_improvement = _finite_metric(gradient_refined["mean"]) - _finite_metric(gradient_initial["mean"])
    reasons: list[str] = []
    if not 0.75 <= determinant <= 1.35:
        reasons.append("determinant_out_of_bounds")
    if not np.isfinite(condition) or condition > 4.0:
        reasons.append("condition_number_out_of_bounds")
    if not np.isfinite(corr_improvement) or corr_improvement < -0.02:
        reasons.append("corr_regressed")
    if not np.isfinite(gradient_improvement) or gradient_improvement < -0.02:
        reasons.append("gradient_ncc_regressed")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "determinant": determinant,
        "condition_number": condition,
        "corr_improvement": corr_improvement,
        "gradient_ncc_improvement": gradient_improvement,
    }


def output_to_input_from_model(
    moving_to_fixed_matrix: np.ndarray,
    moving_to_fixed_translation: np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    center = (np.asarray(shape_zyx, dtype=np.float64) - 1.0) / 2.0
    inverse = np.linalg.inv(np.asarray(moving_to_fixed_matrix, dtype=np.float64))
    offset = center - inverse @ (center + np.asarray(moving_to_fixed_translation, dtype=np.float64))
    return inverse, offset


def model_to_level0(
    *,
    model_matrix: np.ndarray,
    model_translation: np.ndarray,
    sampled_factor_zyx: np.ndarray | None = None,
    fixed_sampled_factor_zyx: np.ndarray | None = None,
    moving_sampled_factor_zyx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if sampled_factor_zyx is not None:
        fixed_sampled_factor_zyx = sampled_factor_zyx
        moving_sampled_factor_zyx = sampled_factor_zyx
    if fixed_sampled_factor_zyx is None or moving_sampled_factor_zyx is None:
        raise ValueError("model_to_level0 requires fixed and moving sampled factors")
    fixed_scale = np.diag(np.asarray(fixed_sampled_factor_zyx, dtype=np.float64))
    moving_scale = np.diag(np.asarray(moving_sampled_factor_zyx, dtype=np.float64))
    matrix = fixed_scale @ np.asarray(model_matrix, dtype=np.float64) @ np.linalg.inv(moving_scale)
    translation = fixed_scale @ np.asarray(model_translation, dtype=np.float64)
    return matrix.astype(np.float32), translation.astype(np.float32)


def local_model_to_full(
    *,
    local_matrix: np.ndarray,
    local_translation: np.ndarray,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    full_shape_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(local_matrix, dtype=np.float64)
    full_center = (np.asarray(full_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    crop_center = (np.asarray(crop_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    translation = (
        np.asarray(local_translation, dtype=np.float64)
        - matrix @ (np.asarray(moving_start_zyx, dtype=np.float64) + crop_center - full_center)
        - full_center
        + np.asarray(fixed_start_zyx, dtype=np.float64)
        + crop_center
    )
    return matrix.astype(np.float32), translation.astype(np.float32)


def full_model_to_local(
    *,
    full_matrix: np.ndarray,
    full_translation: np.ndarray,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    full_shape_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(full_matrix, dtype=np.float64)
    full_center = (np.asarray(full_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    crop_center = (np.asarray(crop_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    translation = (
        matrix @ (np.asarray(moving_start_zyx, dtype=np.float64) + crop_center - full_center)
        + full_center
        + np.asarray(full_translation, dtype=np.float64)
        - np.asarray(fixed_start_zyx, dtype=np.float64)
        - crop_center
    )
    return matrix.astype(np.float32), translation.astype(np.float32)


def output_to_input_to_model(
    matrix: np.ndarray,
    offset: np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    center = (np.asarray(shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moving_to_fixed = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    translation = moving_to_fixed @ (center - np.asarray(offset, dtype=np.float64)) - center
    return moving_to_fixed, translation


def center_model_to_homogeneous_um(
    *,
    matrix_px: np.ndarray,
    translation_px: np.ndarray,
    shape_zyx: np.ndarray,
    fixed_scale_um_zyx: np.ndarray,
    moving_scale_um_zyx: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(matrix_px, dtype=np.float64)
    translation = np.asarray(translation_px, dtype=np.float64)
    center = (np.asarray(shape_zyx, dtype=np.float64) - 1.0) / 2.0
    fixed_scale = np.diag(np.abs(np.asarray(fixed_scale_um_zyx, dtype=np.float64)))
    moving_scale = np.diag(np.abs(np.asarray(moving_scale_um_zyx, dtype=np.float64)))
    origin_translation_px = center + translation - matrix @ center
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :3] = fixed_scale @ matrix @ np.linalg.inv(moving_scale)
    homogeneous[:3, 3] = fixed_scale @ origin_translation_px
    return homogeneous


def homogeneous_um_to_center_model(
    *,
    homogeneous_um: np.ndarray,
    shape_zyx: np.ndarray,
    fixed_scale_um_zyx: np.ndarray,
    moving_scale_um_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.asarray(homogeneous_um, dtype=np.float64)
    if homogeneous.shape != (4, 4):
        raise ValueError(f"Expected 4x4 homogeneous affine, got {homogeneous.shape}")
    fixed_scale = np.diag(np.abs(np.asarray(fixed_scale_um_zyx, dtype=np.float64)))
    moving_scale = np.diag(np.abs(np.asarray(moving_scale_um_zyx, dtype=np.float64)))
    matrix = np.linalg.inv(fixed_scale) @ homogeneous[:3, :3] @ moving_scale
    center = (np.asarray(shape_zyx, dtype=np.float64) - 1.0) / 2.0
    origin_translation_px = np.linalg.inv(fixed_scale) @ homogeneous[:3, 3]
    translation = origin_translation_px - center + matrix @ center
    return matrix.astype(np.float32), translation.astype(np.float32)


def level0_model_to_sampled(
    *,
    level0_matrix: np.ndarray,
    level0_translation: np.ndarray,
    sampled_factor_zyx: np.ndarray | None = None,
    fixed_sampled_factor_zyx: np.ndarray | None = None,
    moving_sampled_factor_zyx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if sampled_factor_zyx is not None:
        fixed_sampled_factor_zyx = sampled_factor_zyx
        moving_sampled_factor_zyx = sampled_factor_zyx
    if fixed_sampled_factor_zyx is None or moving_sampled_factor_zyx is None:
        raise ValueError("level0_model_to_sampled requires fixed and moving sampled factors")
    fixed_scale = np.diag(np.asarray(fixed_sampled_factor_zyx, dtype=np.float64))
    moving_scale = np.diag(np.asarray(moving_sampled_factor_zyx, dtype=np.float64))
    matrix = np.linalg.inv(fixed_scale) @ np.asarray(level0_matrix, dtype=np.float64) @ moving_scale
    translation = np.linalg.inv(fixed_scale) @ np.asarray(level0_translation, dtype=np.float64)
    return matrix.astype(np.float32), translation.astype(np.float32)


def compose_registration_affine(
    *,
    reference_affine: dict[str, Any],
    channel_affine_um: np.ndarray,
    stage_translation_um_zyx: np.ndarray | None = None,
    moving_stage_translation_um_zyx: np.ndarray | None = None,
) -> dict[str, Any]:
    output = json.loads(json.dumps(reference_affine))
    matrix = np.asarray(output["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"registered_affine.matrix must be 4x4, got {matrix.shape}")
    fixed_stage = (
        np.zeros(3, dtype=np.float64)
        if stage_translation_um_zyx is None
        else np.asarray(stage_translation_um_zyx, dtype=np.float64)
    )
    moving_stage = (
        fixed_stage
        if moving_stage_translation_um_zyx is None
        else np.asarray(moving_stage_translation_um_zyx, dtype=np.float64)
    )
    output["matrix"] = (
        matrix
        @ _translation_matrix_zyx_um(fixed_stage)
        @ np.asarray(channel_affine_um, dtype=np.float64)
        @ np.linalg.inv(_translation_matrix_zyx_um(moving_stage))
    ).tolist()
    return output


def _tile_record_placement_um(record: dict[str, Any]) -> np.ndarray:
    return _record_registered_affine_um(record) @ _translation_matrix_zyx_um(_record_stage_translation_um(record))


def _tile_record_center_yx_um(record: dict[str, Any], tile: rough_legacy.TileRecord) -> np.ndarray:
    local_center_um = tile.shape_zyx.astype(np.float64) * tile.scale_zyx_um.astype(np.float64) / 2.0
    center_registered = _tile_record_placement_um(record) @ np.r_[local_center_um, 1.0]
    return center_registered[1:3]


def _cluster_axis(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    diffs = np.diff(sorted_values)
    positive = diffs[diffs > 1e-6]
    tolerance = float(np.min(positive) * 0.25) if positive.size else 1e-3
    clusters: list[list[int]] = []
    centers: list[float] = []
    for index in order:
        value = float(values[index])
        if not clusters or abs(value - centers[-1]) > tolerance:
            clusters.append([int(index)])
            centers.append(value)
        else:
            clusters[-1].append(int(index))
            centers[-1] = float(np.mean(values[clusters[-1]]))
    labels = [0] * values.size
    for label, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = label
    return np.asarray(centers, dtype=np.float64), labels


def grid_fanout_order(
    records: list[dict[str, Any]],
    tiles: list[rough_legacy.TileRecord],
) -> tuple[list[int], list[dict[str, Any]]]:
    if len(records) != len(tiles):
        raise ValueError("records and tiles must have the same length")
    if not records:
        return [], []
    centers = np.asarray([_tile_record_center_yx_um(record, tile) for record, tile in zip(records, tiles, strict=True)])
    row_centers, rows = _cluster_axis(centers[:, 0])
    col_centers, cols = _cluster_axis(centers[:, 1])
    occupied: dict[tuple[int, int], int] = {}
    for index, cell in enumerate(zip(rows, cols, strict=True)):
        if cell in occupied:
            raise ValueError(
                f"Ambiguous tile grid: {records[occupied[cell]]['tile']!r} and {records[index]['tile']!r} share cell {cell}"
            )
        occupied[cell] = index
    grid_center = np.asarray([(len(row_centers) - 1) / 2.0, (len(col_centers) - 1) / 2.0], dtype=np.float64)
    start_cell = min(
        occupied,
        key=lambda cell: (
            abs(cell[0] - grid_center[0]) + abs(cell[1] - grid_center[1]),
            np.linalg.norm(np.asarray(cell, dtype=np.float64) - grid_center),
            records[occupied[cell]]["tile"],
        ),
    )
    order: list[int] = []
    visited: set[tuple[int, int]] = set()
    queue = [start_cell]
    while queue:
        cell = queue.pop(0)
        if cell in visited or cell not in occupied:
            continue
        visited.add(cell)
        order.append(occupied[cell])
        neighbors = [(cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)]
        queue.extend(
            sorted(
                (neighbor for neighbor in neighbors if neighbor in occupied and neighbor not in visited),
                key=lambda neighbor: (
                    abs(neighbor[0] - start_cell[0]) + abs(neighbor[1] - start_cell[1]),
                    neighbor,
                ),
            )
        )
    connected_indices = set(order)
    if len(order) != len(records):
        remaining = [index for index in range(len(records)) if index not in order]
        order.extend(
            sorted(
                remaining,
                key=lambda index: (
                    abs(rows[index] - start_cell[0]) + abs(cols[index] - start_cell[1]),
                    np.linalg.norm(centers[index] - centers[occupied[start_cell]]),
                    records[index]["tile"],
                ),
            )
        )
    metadata_by_index = {
        index: {
            "tile": records[index]["tile"],
            "grid_row": int(rows[index]),
            "grid_col": int(cols[index]),
            "grid_ring": int(abs(rows[index] - start_cell[0]) + abs(cols[index] - start_cell[1])),
            "grid_center_yx_um": [float(value) for value in centers[index]],
            "order_index": position,
            "start_tile": records[occupied[start_cell]]["tile"],
            "disconnected_from_start": index not in connected_indices,
        }
        for position, index in enumerate(order)
    }
    return order, [metadata_by_index[index] for index in order]


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _singular_values, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def rigid_group_mean(matrices: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    if not matrices:
        raise ValueError("rigid_group_mean requires at least one matrix")
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for index, matrix in enumerate(matrices):
        affine = np.asarray(matrix, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError(f"Rigid matrix {index} must be 4x4, got {affine.shape}")
        determinant = float(np.linalg.det(affine[:3, :3]))
        if not np.isfinite(determinant) or determinant <= 0.0:
            raise ValueError(f"Rigid matrix {index} has invalid determinant={determinant}")
        rotations.append(_project_rotation(affine[:3, :3]))
        translations.append(affine[:3, 3])

    quaternions = Rotation.from_matrix(np.stack(rotations, axis=0)).as_quat()
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quaternion in quaternions:
        accumulator += np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    mean_quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if mean_quaternion[3] < 0.0:
        mean_quaternion *= -1.0
    mean_rotation = Rotation.from_quat(mean_quaternion).as_matrix()
    mean = np.eye(4, dtype=np.float64)
    mean[:3, :3] = mean_rotation
    mean[:3, 3] = np.mean(np.stack(translations, axis=0), axis=0)
    determinant = float(np.linalg.det(mean[:3, :3]))
    return mean, {
        "count": len(matrices),
        "method": "rigid_quaternion_rotation_mean",
        "quaternion_xyzw": [float(value) for value in mean_quaternion],
        "determinant": determinant,
        "condition_number": float(np.linalg.cond(mean[:3, :3])),
        "translation_mean_um_zyx": [float(value) for value in mean[:3, 3]],
    }


class RunningRigidPrior:
    def __init__(self, *, min_inliers: int, fit_mode: AffineFitMode) -> None:
        self.min_inliers = int(min_inliers)
        self.fit_mode = fit_mode
        self._world_affines: list[np.ndarray] = []
        self._tile_names: list[str] = []

    @property
    def count(self) -> int:
        return len(self._world_affines)

    @property
    def tile_names(self) -> list[str]:
        return list(self._tile_names)

    @property
    def latest_mean(self) -> dict[str, Any] | None:
        mean = self._mean()
        return None if mean is None else mean[1]

    def _mean(self) -> tuple[np.ndarray, dict[str, Any]] | None:
        if self.count < self.min_inliers:
            return None
        mean_world_affine, stats = rigid_group_mean(self._world_affines)
        return mean_world_affine, {
            **stats,
            "world_affine_um_zyx_homogeneous": mean_world_affine.tolist(),
            "inlier_tiles": self.tile_names,
            "fit_mode": self.fit_mode,
            "prior_transform_family": "rigid",
        }

    def prior_for(self, *, placement_um: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        mean = self._mean()
        if mean is None:
            return None, None
        mean_world_affine, details = mean
        return np.linalg.inv(placement_um) @ mean_world_affine @ placement_um, details

    def add(
        self,
        *,
        tile_name: str,
        channel_affine_um: np.ndarray,
        placement_um: np.ndarray,
    ) -> None:
        placement = np.asarray(placement_um, dtype=np.float64)
        self._world_affines.append(
            placement @ np.asarray(channel_affine_um, dtype=np.float64) @ np.linalg.inv(placement)
        )
        self._tile_names.append(str(tile_name))


def _rotation_zyx(angles: np.ndarray) -> np.ndarray:
    rz, ry, rx = [float(value) for value in angles]
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    rot_z = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_x = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _initial_params(mode: AffineMode | Literal["translation", "rigid", "scale-9dof"], shift_zyx: np.ndarray) -> np.ndarray:
    if mode == "translation":
        return shift_zyx.astype(np.float32, copy=True)
    if mode == "rigid":
        return np.r_[np.zeros(3, dtype=np.float32), shift_zyx]
    if mode == "scale-9dof":
        return np.r_[np.zeros(6, dtype=np.float32), shift_zyx]
    return np.r_[np.zeros(9, dtype=np.float32), shift_zyx]


def _model_from_params(
    mode: AffineMode | Literal["translation", "rigid", "scale-9dof"],
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float32)
    if mode == "translation":
        return np.eye(3, dtype=np.float32), params[0:3]
    if mode == "rigid":
        return _rotation_zyx(params[0:3]), params[3:6]
    if mode == "scale-9dof":
        scales = np.diag(np.exp(params[3:6]).astype(np.float32))
        return _rotation_zyx(params[0:3]) @ scales, params[6:9]
    return (np.eye(3, dtype=np.float32) + params[0:9].reshape(3, 3)).astype(np.float32), params[9:12]


def _compose_delta(
    *,
    mode: AffineMode | Literal["translation", "rigid", "scale-9dof"],
    params: np.ndarray,
    base_matrix: np.ndarray,
    base_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta_matrix, delta_translation = _model_from_params(mode, params)
    return (
        (delta_matrix @ base_matrix).astype(np.float32),
        (delta_matrix @ base_translation + delta_translation).astype(np.float32),
    )


def estimate_translation_gpu(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    *,
    upsample_factor: int = CHANNEL_PHASE_UPSAMPLE_FACTOR,
) -> tuple[float, float, float]:
    import cupy as cp
    from cucim.skimage.registration import phase_cross_correlation

    if fixed_zyx.shape != moving_zyx.shape:
        raise ValueError(f"Expected matching shapes, got fixed={fixed_zyx.shape}, moving={moving_zyx.shape}")
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")
    fixed = cp.asarray(fixed_zyx, dtype=cp.float32)
    moving = cp.asarray(moving_zyx, dtype=cp.float32)
    fixed -= cp.mean(fixed)
    moving -= cp.mean(moving)
    shift, _error, _phase = phase_cross_correlation(fixed, moving, upsample_factor=upsample_factor)
    shift_cpu = cp.asnumpy(shift)
    return float(shift_cpu[0]), float(shift_cpu[1]), float(shift_cpu[2])


def fit_affine_gpu(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    initial_matrix: np.ndarray,
    initial_translation: np.ndarray,
    max_iterations: int,
    ftol: float = 1e-4,
    stage_modes: tuple[AffineMode | Literal["translation", "rigid", "scale-9dof"], ...] = ("affine-12dof",),
) -> tuple[np.ndarray, np.ndarray, float]:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform

    fixed_gpu = cp.asarray(fixed, dtype=cp.float32)
    moving_gpu = cp.asarray(moving, dtype=cp.float32)
    fixed_centered = fixed_gpu - cp.mean(fixed_gpu)
    fixed_norm = cp.linalg.norm(fixed_centered)
    if float(cp.asnumpy(fixed_norm)) == 0.0:
        raise AffineMeasurementRejected("fixed volume is constant; affine registration requires image contrast")

    def objective(model_matrix: np.ndarray, model_translation: np.ndarray) -> float:
        matrix, offset = output_to_input_from_model(model_matrix, model_translation, fixed.shape)
        registered = gpu_affine_transform(
            moving_gpu,
            cp.asarray(matrix, dtype=cp.float32),
            cp.asarray(offset, dtype=cp.float32),
            output_shape=fixed.shape,
            order=1,
            mode="constant",
            cval=0.0,
        )
        registered_centered = registered - cp.mean(registered)
        denom = fixed_norm * cp.linalg.norm(registered_centered)
        if float(cp.asnumpy(denom)) == 0.0:
            return 1.0
        return -float(cp.asnumpy(cp.sum(fixed_centered * registered_centered) / denom))

    model_matrix = np.asarray(initial_matrix, dtype=np.float32)
    model_translation = np.asarray(initial_translation, dtype=np.float32)
    for stage_mode in stage_modes:
        start = _initial_params(stage_mode, np.zeros(3, dtype=np.float32))
        result = minimize(
            lambda params: objective(*_compose_delta(
                mode=stage_mode,
                params=np.asarray(params, dtype=np.float32),
                base_matrix=model_matrix,
                base_translation=model_translation,
            )),
            start,
            method="Powell",
            options={"maxiter": max_iterations, "ftol": ftol, "xtol": ftol, "disp": False},
        )
        model_matrix, model_translation = _compose_delta(
            mode=stage_mode,
            params=result.x.astype(np.float32, copy=False),
            base_matrix=model_matrix,
            base_translation=model_translation,
        )
    return model_matrix, model_translation, -objective(model_matrix, model_translation)


def _transform_volume_cupy(volume: Any, matrix: np.ndarray, translation: np.ndarray) -> Any:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform

    output_to_input, offset = output_to_input_from_model(matrix, translation, volume.shape)
    return gpu_affine_transform(
        cp.asarray(volume, dtype=cp.float32),
        cp.asarray(output_to_input, dtype=cp.float32),
        cp.asarray(offset, dtype=cp.float32),
        output_shape=volume.shape,
        order=1,
        mode="constant",
        cval=0.0,
    )


def transform_volume_gpu(volume: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    import cupy as cp

    return cp.asnumpy(_transform_volume_cupy(volume, matrix, translation))


def transformed_corr_gpu(
    fixed: np.ndarray,
    moving: np.ndarray,
    matrix: np.ndarray,
    translation: np.ndarray,
) -> float | None:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform

    output_to_input, offset = output_to_input_from_model(matrix, translation, fixed.shape)
    fixed_gpu = cp.asarray(fixed, dtype=cp.float32)
    moved = gpu_affine_transform(
        cp.asarray(moving, dtype=cp.float32),
        cp.asarray(output_to_input, dtype=cp.float32),
        cp.asarray(offset, dtype=cp.float32),
        output_shape=fixed.shape,
        order=1,
        mode="constant",
        cval=0.0,
    )
    mask = cp.isfinite(fixed_gpu) & cp.isfinite(moved) & (moved != 0)
    count = int(cp.asnumpy(cp.count_nonzero(mask)))
    if count < 8:
        return None
    fixed_values = fixed_gpu[mask].astype(cp.float64)
    moved_values = moved[mask].astype(cp.float64)
    fixed_std = cp.std(fixed_values)
    moved_std = cp.std(moved_values)
    if float(cp.asnumpy(fixed_std)) == 0.0 or float(cp.asnumpy(moved_std)) == 0.0:
        return None
    fixed_centered = fixed_values - cp.mean(fixed_values)
    moved_centered = moved_values - cp.mean(moved_values)
    denominator = cp.sqrt(cp.sum(fixed_centered * fixed_centered) * cp.sum(moved_centered * moved_centered))
    if float(cp.asnumpy(denominator)) == 0.0:
        return None
    return float(cp.asnumpy(cp.sum(fixed_centered * moved_centered) / denominator))


def gradient_component_ncc_3d_gpu(fixed: np.ndarray, moving: np.ndarray, fixed_mask: Any | None = None) -> dict[str, Any]:
    return gradient_component_ncc_3d_gpu_fused(fixed, moving, fixed_mask=fixed_mask)


def _gradient_component_ncc_3d_gpu_sobel_arrays(fixed: np.ndarray, moving: np.ndarray) -> dict[str, Any]:
    import cupy as cp
    from cupyx.scipy import ndimage as gpu_ndimage

    fixed_gpu = cp.asarray(fixed, dtype=cp.float32)
    moving_gpu = cp.asarray(moving, dtype=cp.float32)
    mask = cp.isfinite(fixed_gpu) & cp.isfinite(moving_gpu) & (moving_gpu != 0)
    values: list[float] = []
    for axis in range(3):
        fixed_gradient = gpu_ndimage.sobel(fixed_gpu, axis=axis)
        moving_gradient = gpu_ndimage.sobel(moving_gpu, axis=axis)
        axis_mask = mask & cp.isfinite(fixed_gradient) & cp.isfinite(moving_gradient)
        count = int(cp.asnumpy(cp.count_nonzero(axis_mask)))
        if count < 256:
            values.append(float("nan"))
            continue
        fixed_values = cp.where(axis_mask, fixed_gradient, 0.0)
        moving_values = cp.where(axis_mask, moving_gradient, 0.0)
        fixed_mean = cp.sum(fixed_values) / cp.float32(count)
        moving_mean = cp.sum(moving_values) / cp.float32(count)
        fixed_centered = cp.where(axis_mask, fixed_gradient - fixed_mean, 0.0)
        moving_centered = cp.where(axis_mask, moving_gradient - moving_mean, 0.0)
        denominator = cp.sqrt(cp.sum(fixed_centered * fixed_centered) * cp.sum(moving_centered * moving_centered))
        values.append(
            float("nan")
            if float(cp.asnumpy(denominator)) == 0.0
            else float(cp.asnumpy(cp.sum(fixed_centered * moving_centered) / denominator))
        )
    finite = [value for value in values if np.isfinite(value)]
    return {"mean": float(np.mean(finite)) if finite else float("nan"), "zyx_components": values, "backend": "cupy"}


def _fused_gradient_ncc_module() -> Any:
    import cupy as cp

    if not hasattr(_fused_gradient_ncc_module, "module"):
        _fused_gradient_ncc_module.module = cp.RawModule(
            code=r'''
extern "C" __global__ void gradient_ncc_sobel3d_partials(
    const float* fixed,
    const float* moving,
    const bool* fixed_mask,
    const bool use_fixed_mask,
    const long total_inner,
    const int z_size,
    const int y_size,
    const int x_size,
    double* partials
) {
    extern __shared__ double shared[];
    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;
    const int lanes = 18;
    for (int i = tid; i < lanes; i += block_threads) {
        shared[i] = 0.0;
    }
    __syncthreads();

    double local[18];
    for (int i = 0; i < 18; ++i) {
        local[i] = 0.0;
    }

    const long yz_inner = (long)(y_size - 2) * (long)(x_size - 2);
    const long plane = (long)y_size * (long)x_size;
    const int sobel[3] = {1, 2, 1};
    const int deriv[3] = {-1, 0, 1};

    for (long inner = (long)blockIdx.x * block_threads + tid; inner < total_inner; inner += (long)gridDim.x * block_threads) {
        const int z = (int)(inner / yz_inner) + 1;
        const long rem = inner - (long)(z - 1) * yz_inner;
        const int y = (int)(rem / (x_size - 2)) + 1;
        const int x = (int)(rem - (long)(y - 1) * (x_size - 2)) + 1;
        const long center = (long)z * plane + (long)y * x_size + (long)x;
        if (use_fixed_mask && !fixed_mask[center]) {
            continue;
        }
        const float moving_center = moving[center];
        if (!isfinite(fixed[center]) || !isfinite(moving_center) || moving_center == 0.0f) {
            continue;
        }

        bool finite_axis[3] = {true, true, true};
        double fixed_grad[3] = {0.0, 0.0, 0.0};
        double moving_grad[3] = {0.0, 0.0, 0.0};
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const long idx = center + (long)dz * plane + (long)dy * x_size + (long)dx;
                    const float f = fixed[idx];
                    const float m = moving[idx];
                    const int wz = sobel[dz + 1];
                    const int wy = sobel[dy + 1];
                    const int wx = sobel[dx + 1];
                    const int weights[3] = {
                        deriv[dz + 1] * wy * wx,
                        wz * deriv[dy + 1] * wx,
                        wz * wy * deriv[dx + 1],
                    };
                    for (int axis = 0; axis < 3; ++axis) {
                        if (weights[axis] == 0) {
                            continue;
                        }
                        if (!isfinite(f) || !isfinite(m)) {
                            finite_axis[axis] = false;
                            continue;
                        }
                        fixed_grad[axis] += (double)weights[axis] * (double)f;
                        moving_grad[axis] += (double)weights[axis] * (double)m;
                    }
                }
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!finite_axis[axis]) {
                continue;
            }
            const double f = fixed_grad[axis];
            const double m = moving_grad[axis];
            const int base = axis * 6;
            local[base] += 1.0;
            local[base + 1] += f;
            local[base + 2] += m;
            local[base + 3] += f * f;
            local[base + 4] += m * m;
            local[base + 5] += f * m;
        }
    }

    for (int i = 0; i < lanes; ++i) {
        atomicAdd(&shared[i], local[i]);
    }
    __syncthreads();
    for (int i = tid; i < lanes; i += block_threads) {
        partials[(long)blockIdx.x * lanes + i] = shared[i];
    }
}
''',
        )
    return _fused_gradient_ncc_module.module


def _ncc_from_axis_sums(values: np.ndarray) -> list[float]:
    components: list[float] = []
    for axis in range(3):
        count, fixed_sum, moving_sum, fixed_sq_sum, moving_sq_sum, cross_sum = values[axis * 6 : (axis + 1) * 6]
        if count < 256:
            components.append(float("nan"))
            continue
        fixed_centered_sq = fixed_sq_sum - fixed_sum * fixed_sum / count
        moving_centered_sq = moving_sq_sum - moving_sum * moving_sum / count
        cross_centered = cross_sum - fixed_sum * moving_sum / count
        denominator = np.sqrt(fixed_centered_sq * moving_centered_sq)
        components.append(float("nan") if denominator <= 0.0 else float(cross_centered / denominator))
    return components


def gradient_component_ncc_3d_gpu_fused(fixed: np.ndarray, moving: np.ndarray, fixed_mask: Any | None = None) -> dict[str, Any]:
    """Compute Sobel-component NCC in one CUDA pass over the interior voxels.

    The fused kernel intentionally excludes the 1-voxel border instead of using
    reflect boundary conditions. This preserves the Sobel metric on the large
    registration crops while avoiding three full gradient arrays per image.
    """
    import cupy as cp

    if fixed.shape != moving.shape:
        raise ValueError(f"fixed and moving shapes differ: {fixed.shape} vs {moving.shape}")
    if fixed.ndim != 3:
        raise ValueError(f"gradient_component_ncc_3d_gpu_fused expects a 3D zyx volume, got {fixed.ndim}D")
    if min(fixed.shape) < 3:
        return {"mean": float("nan"), "zyx_components": [float("nan")] * 3, "backend": "cupy_fused_sobel_interior"}

    fixed_gpu = cp.asarray(fixed, dtype=cp.float32)
    moving_gpu = cp.asarray(moving, dtype=cp.float32)
    use_fixed_mask = fixed_mask is not None
    if fixed_mask is None:
        fixed_mask_gpu = cp.ones((1,), dtype=cp.bool_)
    else:
        fixed_mask_gpu = cp.ascontiguousarray(cp.asarray(fixed_mask, dtype=cp.bool_))
        if fixed_mask_gpu.shape != fixed_gpu.shape:
            raise ValueError(f"fixed_mask shape differs: {fixed_mask_gpu.shape} vs {fixed_gpu.shape}")
    z_size, y_size, x_size = (int(value) for value in fixed.shape)
    total_inner = int((z_size - 2) * (y_size - 2) * (x_size - 2))
    threads = 256
    blocks = max(1, min(4096, (total_inner + threads - 1) // threads))
    partials = cp.zeros((blocks, 18), dtype=cp.float64)
    kernel = _fused_gradient_ncc_module().get_function("gradient_ncc_sobel3d_partials")
    kernel(
        (blocks,),
        (threads,),
        (
            fixed_gpu,
            moving_gpu,
            fixed_mask_gpu,
            np.bool_(use_fixed_mask),
            np.int64(total_inner),
            np.int32(z_size),
            np.int32(y_size),
            np.int32(x_size),
            partials,
        ),
        shared_mem=18 * 8,
    )
    sums = cp.asnumpy(cp.sum(partials, axis=0))
    values = _ncc_from_axis_sums(sums)
    finite = [value for value in values if np.isfinite(value)]
    return {
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "zyx_components": values,
        "backend": "cupy_fused_sobel_interior",
        "fixed_masked": fixed_mask is not None,
    }


def _read_level0_crop(
    tile: rough_legacy.TileRecord,
    *,
    channel: int,
    start_zyx: np.ndarray,
    crop_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, list[list[int]]]:
    starts = [int(value) for value in start_zyx]
    stops = [start + int(crop) for start, crop in zip(starts, crop_shape_zyx, strict=True)]
    z_indices = np.arange(starts[0], stops[0], dtype=np.int64)
    volume = read_tile_indexed_z_patch(
        tile,
        channel=channel,
        z_indices=z_indices,
        y_slice=slice(starts[1], stops[1]),
        x_slice=slice(starts[2], stops[2]),
    )
    return volume, [[int(start), int(stop)] for start, stop in zip(starts, stops, strict=True)]


def _clipped_crop_start(start_zyx: np.ndarray, *, shape_zyx: np.ndarray, crop_shape_zyx: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(np.clip(round(float(start)), 0, int(size) - int(crop)))
            for start, size, crop in zip(start_zyx, shape_zyx, crop_shape_zyx, strict=True)
        ],
        dtype=np.int64,
    )


def _candidate_starts(size: int, window: int) -> list[int]:
    if window > size:
        raise ValueError(f"Content window {window} exceeds sampled axis size {size}")
    step = max(1, window // 4)
    starts = list(range(0, size - window + 1, step))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def _content_selection_maps_gpu(
    fixed_sampled: np.ndarray,
    moving_registered_sampled: np.ndarray,
) -> dict[str, np.ndarray | float]:
    import cupy as cp
    from cupyx.scipy import ndimage as gpu_ndimage

    fixed = cp.asarray(fixed_sampled, dtype=cp.float32)
    moving = cp.asarray(moving_registered_sampled, dtype=cp.float32)
    finite = cp.isfinite(fixed) & cp.isfinite(moving)
    moving_support = finite & (moving != 0)
    if int(cp.asnumpy(cp.count_nonzero(moving_support))) == 0:
        raise AffineMeasurementRejected("No finite moving support for Sobel content selection")

    def robust_unit(volume: cp.ndarray, support: cp.ndarray) -> cp.ndarray:
        values = volume[support]
        if int(values.size) == 0:
            return cp.zeros(volume.shape, dtype=cp.float32)
        low, high = cp.percentile(values, cp.asarray([1.0, 99.8], dtype=cp.float32))
        span = cp.maximum(high - low, cp.float32(1e-6))
        scaled = cp.clip((volume - low) / span, 0.0, 1.0)
        return cp.where(support, cp.sqrt(scaled), 0.0).astype(cp.float32, copy=False)

    fixed = robust_unit(fixed, finite)
    moving = robust_unit(moving, moving_support)
    fixed_smooth = gpu_ndimage.gaussian_filter(cp.where(finite, fixed, 0.0), sigma=(1.0, 2.0, 2.0))
    moving_smooth = gpu_ndimage.gaussian_filter(cp.where(moving_support, moving, 0.0), sigma=(1.0, 2.0, 2.0))

    def foreground_mask(volume: cp.ndarray, support: cp.ndarray) -> tuple[cp.ndarray, float]:
        positive = volume[support & (volume > 0)]
        if int(positive.size) == 0:
            return cp.zeros(volume.shape, dtype=cp.bool_), float("inf")
        threshold = cp.maximum(cp.percentile(positive, 60.0), cp.float32(0.05))
        mask = support & (volume >= threshold)
        structure = cp.ones((3, 9, 9), dtype=cp.bool_)
        return gpu_ndimage.binary_dilation(mask, structure=structure), float(cp.asnumpy(threshold))

    fixed_foreground, fixed_threshold = foreground_mask(fixed_smooth, finite)
    moving_foreground, moving_threshold = foreground_mask(moving_smooth, moving_support)
    mutual_foreground = fixed_foreground & moving_foreground & moving_support

    def sobel_energy(volume: cp.ndarray, support: cp.ndarray) -> cp.ndarray:
        highpass = cp.where(
            support,
            volume - gpu_ndimage.gaussian_filter(cp.where(support, volume, 0.0), sigma=(1.0, 4.0, 4.0)),
            0.0,
        )
        energy = cp.zeros(volume.shape, dtype=cp.float32)
        for axis in range(3):
            gradient = gpu_ndimage.sobel(highpass, axis=axis)
            energy += gradient * gradient
        return cp.sqrt(energy)

    fixed_edge = sobel_energy(fixed, finite)
    moving_edge = sobel_energy(moving, moving_support)
    maps = {
        "finite": cp.asnumpy(finite),
        "moving_support": cp.asnumpy(moving_support),
        "fixed_foreground": cp.asnumpy(fixed_foreground),
        "moving_foreground": cp.asnumpy(moving_foreground),
        "mutual_foreground": cp.asnumpy(mutual_foreground),
        "fixed_edge": cp.asnumpy(fixed_edge),
        "moving_edge": cp.asnumpy(moving_edge),
        "fixed_foreground_threshold": fixed_threshold,
        "moving_foreground_threshold": moving_threshold,
    }
    return maps


def select_content_fixed_crop_candidates_l0(
    *,
    fixed_sampled: np.ndarray,
    moving_registered_sampled: np.ndarray,
    sampled_factor_zyx: np.ndarray,
    sampled_z_l0: np.ndarray,
    tile_shape_zyx: np.ndarray,
    crop_shape_zyx: tuple[int, int, int],
    max_candidates: int = 1,
    min_center_separation_fraction_yx: float = 0.5,
    selection_mode: Literal["score", "corners"] = "score",
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    if max_candidates < 1:
        raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")
    if min_center_separation_fraction_yx < 0.0:
        raise ValueError(
            "min_center_separation_fraction_yx must be non-negative, "
            f"got {min_center_separation_fraction_yx}"
        )
    if selection_mode not in {"score", "corners"}:
        raise ValueError(f"selection_mode must be 'score' or 'corners', got {selection_mode!r}")
    crop = np.asarray(crop_shape_zyx, dtype=np.int64)
    factor = np.asarray(sampled_factor_zyx, dtype=np.float64)
    sampled_window = np.maximum(3, np.rint(crop.astype(np.float64) / factor).astype(np.int64))
    sampled_window = np.minimum(sampled_window, np.asarray(fixed_sampled.shape, dtype=np.int64))
    maps = _content_selection_maps_gpu(fixed_sampled, moving_registered_sampled)
    finite = np.asarray(maps["finite"], dtype=bool)
    moving_support = np.asarray(maps["moving_support"], dtype=bool)
    fixed_foreground = np.asarray(maps["fixed_foreground"], dtype=bool)
    moving_foreground = np.asarray(maps["moving_foreground"], dtype=bool)
    mutual_foreground = np.asarray(maps["mutual_foreground"], dtype=bool)
    fixed_edge = np.asarray(maps["fixed_edge"], dtype=np.float32)
    moving_edge = np.asarray(maps["moving_edge"], dtype=np.float32)
    candidates: list[dict[str, Any]] = []
    rejected_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    for z_start in _candidate_starts(int(fixed_sampled.shape[0]), int(sampled_window[0])):
        z_slice = slice(z_start, z_start + int(sampled_window[0]))
        for y_start in _candidate_starts(int(fixed_sampled.shape[1]), int(sampled_window[1])):
            y_slice = slice(y_start, y_start + int(sampled_window[1]))
            for x_start in _candidate_starts(int(fixed_sampled.shape[2]), int(sampled_window[2])):
                x_slice = slice(x_start, x_start + int(sampled_window[2]))
                window = (z_slice, y_slice, x_slice)
                window_support = moving_support[window]
                support_fraction = float(np.count_nonzero(window_support) / window_support.size)
                if support_fraction < 0.9:
                    reject("low_moving_support")
                    continue
                fixed_mask = fixed_foreground[window]
                moving_mask = moving_foreground[window]
                mutual_mask = mutual_foreground[window]
                fixed_fraction = float(np.count_nonzero(fixed_mask) / fixed_mask.size)
                moving_fraction = float(np.count_nonzero(moving_mask) / moving_mask.size)
                if not 0.005 <= fixed_fraction <= 0.75:
                    reject("fixed_foreground_fraction_out_of_bounds")
                    continue
                if not 0.005 <= moving_fraction <= 0.75:
                    reject("moving_foreground_fraction_out_of_bounds")
                    continue
                mutual_count = int(np.count_nonzero(mutual_mask))
                min_foreground = min(int(np.count_nonzero(fixed_mask)), int(np.count_nonzero(moving_mask)))
                mutual_fraction = float(mutual_count / mutual_mask.size)
                mutual_overlap = 0.0 if min_foreground == 0 else float(mutual_count / min_foreground)
                if mutual_fraction < 0.002 or mutual_overlap < 0.15:
                    reject("low_mutual_foreground_overlap")
                    continue
                edge_mask = mutual_mask & finite[window]
                if int(np.count_nonzero(edge_mask)) < 256:
                    reject("low_edge_support")
                    continue
                fixed_edge_values = fixed_edge[window][edge_mask]
                moving_edge_values = moving_edge[window][edge_mask]
                fixed_edge_p95 = float(np.percentile(fixed_edge_values, 95.0))
                moving_edge_p95 = float(np.percentile(moving_edge_values, 95.0))
                fixed_edge_median = float(np.median(fixed_edge_values))
                moving_edge_median = float(np.median(moving_edge_values))
                fixed_edge_mad = float(np.median(np.abs(fixed_edge_values - fixed_edge_median)))
                moving_edge_mad = float(np.median(np.abs(moving_edge_values - moving_edge_median)))
                fixed_edge_snr = fixed_edge_p95 / max(1e-6, 1.4826 * fixed_edge_mad)
                moving_edge_snr = moving_edge_p95 / max(1e-6, 1.4826 * moving_edge_mad)
                score = (
                    mutual_overlap
                    * min(fixed_edge_snr, moving_edge_snr)
                    * float(np.sqrt(max(0.0, fixed_edge_p95) * max(0.0, moving_edge_p95)))
                    * support_fraction
                )
                candidates.append(
                    {
                        "sampled_start": np.asarray([z_start, y_start, x_start], dtype=np.int64),
                        "score": float(score),
                        "score_terms": {
                            "moving_support_fraction": support_fraction,
                            "fixed_foreground_fraction": fixed_fraction,
                            "moving_foreground_fraction": moving_fraction,
                            "mutual_foreground_fraction": mutual_fraction,
                            "mutual_foreground_overlap": mutual_overlap,
                            "fixed_edge_p95": fixed_edge_p95,
                            "moving_edge_p95": moving_edge_p95,
                            "fixed_edge_snr": fixed_edge_snr,
                            "moving_edge_snr": moving_edge_snr,
                        },
                    }
                )
    if not candidates:
        raise AffineMeasurementRejected(
            f"No Sobel foreground-overlap content crop candidate was found; rejected_counts={rejected_counts}"
        )

    min_separation_yx = crop[1:3].astype(np.float64) * float(min_center_separation_fraction_yx)
    min_separation_distance = float(np.linalg.norm(min_separation_yx))
    selected: list[tuple[np.ndarray, dict[str, Any]]] = []
    ranked_candidates: list[dict[str, Any]] = []
    for rank, candidate in enumerate(sorted(candidates, key=lambda item: float(item["score"]), reverse=True)):
        sampled_start = np.asarray(candidate["sampled_start"], dtype=np.int64)
        sampled_center = sampled_start.astype(np.float64) + (sampled_window.astype(np.float64) - 1.0) / 2.0
        z_center_l0 = float(sampled_z_l0[int(round(float(sampled_center[0])))])
        center_l0 = np.asarray(
            [z_center_l0, sampled_center[1] * factor[1], sampled_center[2] * factor[2]],
            dtype=np.float64,
        )
        fixed_start = _clipped_crop_start(
            center_l0 - (crop.astype(np.float64) - 1.0) / 2.0,
            shape_zyx=np.asarray(tile_shape_zyx, dtype=np.int64),
            crop_shape_zyx=crop,
        )
        ranked_candidates.append(
            {
                **candidate,
                "score_rank": int(rank),
                "sampled_center": sampled_center,
                "center_l0": center_l0,
                "fixed_start": fixed_start,
            }
        )

    def is_too_close(center_l0: np.ndarray) -> bool:
        if min_separation_distance > 0.0:
            return any(
                float(np.linalg.norm(center_l0[1:3] - np.asarray(details["selected_center_l0_zyx"], dtype=np.float64)[1:3]))
                < min_separation_distance
                for _start, details in selected
            )
        return False

    def add_candidate(candidate: dict[str, Any], *, corner_target_l0_yx: np.ndarray | None = None) -> None:
        sampled_start = np.asarray(candidate["sampled_start"], dtype=np.int64)
        sampled_center = np.asarray(candidate["sampled_center"], dtype=np.float64)
        center_l0 = np.asarray(candidate["center_l0"], dtype=np.float64)
        fixed_start = np.asarray(candidate["fixed_start"], dtype=np.int64)
        details: dict[str, Any] = {
            "method": "level2_cupy_foreground_overlap_sobel_edge_energy",
            "selection_mode": selection_mode,
            "candidate_rank": int(candidate["score_rank"]),
            "candidate_count_before_nms": int(len(candidates)),
            "selected_candidate_count": None,
            "nms_min_center_separation_l0_yx": [float(value) for value in min_separation_yx],
            "sampled_window_zyx": [int(value) for value in sampled_window],
            "sampled_start_zyx": [int(value) for value in sampled_start],
            "sampled_center_zyx": [float(value) for value in sampled_center],
            "selected_center_l0_zyx": [float(value) for value in center_l0],
            "score": float(candidate["score"]),
            "score_terms": candidate["score_terms"],
            "rejected_candidate_counts": rejected_counts,
            "fixed_foreground_threshold": float(maps["fixed_foreground_threshold"]),
            "moving_foreground_threshold": float(maps["moving_foreground_threshold"]),
        }
        if corner_target_l0_yx is not None:
            details["corner_target_l0_yx"] = [float(value) for value in corner_target_l0_yx]
            details["corner_distance_l0_yx"] = [float(value) for value in center_l0[1:3] - corner_target_l0_yx]
            details["corner_distance_l0"] = float(np.linalg.norm(center_l0[1:3] - corner_target_l0_yx))
        selected.append(
            (
                fixed_start,
                details,
            )
        )

    if selection_mode == "corners":
        tile_shape = np.asarray(tile_shape_zyx, dtype=np.float64)
        center_low_yx = (crop[1:3].astype(np.float64) - 1.0) / 2.0
        center_high_yx = tile_shape[1:3] - (crop[1:3].astype(np.float64) + 1.0) / 2.0
        corner_targets = [
            np.asarray([center_low_yx[0], center_low_yx[1]], dtype=np.float64),
            np.asarray([center_low_yx[0], center_high_yx[1]], dtype=np.float64),
            np.asarray([center_high_yx[0], center_low_yx[1]], dtype=np.float64),
            np.asarray([center_high_yx[0], center_high_yx[1]], dtype=np.float64),
        ]
        for target_yx in corner_targets:
            if len(selected) >= max_candidates:
                break
            eligible = [
                candidate
                for candidate in ranked_candidates
                if not is_too_close(np.asarray(candidate["center_l0"], dtype=np.float64))
            ]
            if not eligible:
                break
            best = min(
                eligible,
                key=lambda candidate: (
                    float(np.linalg.norm(np.asarray(candidate["center_l0"], dtype=np.float64)[1:3] - target_yx)),
                    -float(candidate["score"]),
                ),
            )
            add_candidate(best, corner_target_l0_yx=target_yx)

    for candidate in ranked_candidates:
        if len(selected) >= max_candidates:
            break
        if is_too_close(np.asarray(candidate["center_l0"], dtype=np.float64)):
            continue
        add_candidate(candidate)
    if not selected:
        raise AffineMeasurementRejected(
            "No spatially separated Sobel foreground-overlap content crop candidate was found after NMS; "
            f"candidate_count_before_nms={len(candidates)}"
        )
    for _start, details in selected:
        details["selected_candidate_count"] = int(len(selected))
    return selected


def select_content_fixed_crop_start_l0(
    *,
    fixed_sampled: np.ndarray,
    moving_registered_sampled: np.ndarray,
    sampled_factor_zyx: np.ndarray,
    sampled_z_l0: np.ndarray,
    tile_shape_zyx: np.ndarray,
    crop_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = select_content_fixed_crop_candidates_l0(
        fixed_sampled=fixed_sampled,
        moving_registered_sampled=moving_registered_sampled,
        sampled_factor_zyx=sampled_factor_zyx,
        sampled_z_l0=sampled_z_l0,
        tile_shape_zyx=tile_shape_zyx,
        crop_shape_zyx=crop_shape_zyx,
        max_candidates=1,
    )
    return candidates[0]


def moving_crop_start_for_fixed_crop(
    *,
    fixed_start_zyx: np.ndarray,
    crop_shape_zyx: tuple[int, int, int],
    full_matrix: np.ndarray,
    full_translation: np.ndarray,
    fixed_shape_zyx: np.ndarray,
    moving_shape_zyx: np.ndarray,
) -> np.ndarray:
    crop = np.asarray(crop_shape_zyx, dtype=np.int64)
    fixed_center = crop_center_zyx(fixed_start_zyx, crop)
    fixed_full_center = (np.asarray(fixed_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moving_full_center = (np.asarray(moving_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    moving_center = moving_full_center + np.linalg.inv(np.asarray(full_matrix, dtype=np.float64)) @ (
        fixed_center - fixed_full_center - np.asarray(full_translation, dtype=np.float64)
    )
    return _clipped_crop_start(
        moving_center - (crop.astype(np.float64) - 1.0) / 2.0,
        shape_zyx=np.asarray(moving_shape_zyx, dtype=np.int64),
        crop_shape_zyx=crop,
    )


def crop_center_zyx(start_zyx: np.ndarray, crop_shape_zyx: np.ndarray | tuple[int, int, int]) -> np.ndarray:
    crop = np.asarray(crop_shape_zyx, dtype=np.float64)
    return np.asarray(start_zyx, dtype=np.float64) + (crop - 1.0) / 2.0


def _write_affine_contact_sheet(
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    initial_moved: np.ndarray,
    refined_moved: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    rows: list[Image.Image] = []
    for index in np.unique(np.rint(np.linspace(0, fixed.shape[0] - 1, 6)).astype(np.int64)):
        panels = [
            _labelled_overlay("unshifted", fixed[index], moving[index]),
            _labelled_overlay("level2 init", fixed[index], initial_moved[index]),
            _labelled_overlay("level0 refined", fixed[index], refined_moved[index]),
        ]
        row = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height + 28), (20, 20, 20))
        ImageDraw.Draw(row).text((6, 8), f"{title} z={int(index)}", fill=(255, 255, 255))
        x = 0
        for panel in panels:
            row.paste(panel, (x, 28))
            x += panel.width
        rows.append(row)
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), (20, 20, 20))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _labelled_overlay(name: str, fixed_plane: np.ndarray, moving_plane: np.ndarray) -> Image.Image:
    image = Image.fromarray(_rgb_overlay(fixed_plane, moving_plane))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((6, 7), name, fill=(255, 255, 255))
    return image


def _measure_tile_affine(
    *,
    reference_tile: rough_legacy.TileRecord,
    moving_tile: rough_legacy.TileRecord,
    reference_channel: int,
    moving_channel: int,
    init_level: int,
    init_z_samples: int,
    refine_crop_shape_zyx: tuple[int, int, int],
    max_iterations: int,
    contact_sheet_path: Path | None,
    fit_mode: AffineFitMode = "rigid",
    prior_channel_affine_um: np.ndarray | None = None,
    accepted_inliers_before_tile: int = 0,
    tile_order_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fixed_init_raw, init_factor, init_source_level, fixed_available_levels, fixed_z_l0 = sampled_tile_volume_from_subifd(
        reference_tile,
        channel=reference_channel,
        requested_level=init_level,
        z_samples=init_z_samples,
    )
    moving_init_raw, moving_init_factor, moving_source_level, moving_available_levels, _moving_z_l0 = (
        sampled_tile_volume_from_subifd(
            moving_tile,
            channel=moving_channel,
            requested_level=init_level,
            z_samples=init_z_samples,
        )
    )
    if fixed_init_raw.shape != moving_init_raw.shape:
        raise ValueError(
            f"Initial affine volumes differ for {reference_tile.tile}: "
            f"fixed={fixed_init_raw.shape}, moving={moving_init_raw.shape}"
        )
    if not np.array_equal(fixed_z_l0, _moving_z_l0):
        raise ValueError(f"Initial affine z samples differ for {reference_tile.tile}")
    fixed_init = _robust_norm(fixed_init_raw)
    moving_init = _robust_norm(moving_init_raw)
    prior_details: dict[str, Any] = {
        "used": False,
        "accepted_inliers_before_tile": int(accepted_inliers_before_tile),
        "method": "phase_only",
    }
    if prior_channel_affine_um is None:
        phase_shift = np.asarray(estimate_translation_gpu(fixed_init, moving_init), dtype=np.float32)
        initial_matrix = np.eye(3, dtype=np.float32)
        initial_translation = phase_shift
    else:
        prior_level0_matrix, prior_level0_translation = homogeneous_um_to_center_model(
            homogeneous_um=prior_channel_affine_um,
            shape_zyx=reference_tile.shape_zyx,
            fixed_scale_um_zyx=reference_tile.scale_zyx_um,
            moving_scale_um_zyx=moving_tile.scale_zyx_um,
        )
        prior_sampled_matrix, prior_sampled_translation = level0_model_to_sampled(
            level0_matrix=prior_level0_matrix,
            level0_translation=prior_level0_translation,
            fixed_sampled_factor_zyx=init_factor,
            moving_sampled_factor_zyx=moving_init_factor,
        )
        moving_prior_registered = transform_volume_gpu(moving_init, prior_sampled_matrix, prior_sampled_translation)
        phase_shift = np.asarray(estimate_translation_gpu(fixed_init, moving_prior_registered), dtype=np.float32)
        initial_matrix = prior_sampled_matrix
        initial_translation = (prior_sampled_translation + phase_shift).astype(np.float32)
        prior_details = {
            "used": True,
            "accepted_inliers_before_tile": int(accepted_inliers_before_tile),
            "method": "running_rigid_quaternion_mean_then_residual_phase",
            "prior_channel_affine_um_zyx_homogeneous": np.asarray(prior_channel_affine_um, dtype=np.float64).tolist(),
            "prior_level0_moving_to_fixed_matrix_zyx": prior_level0_matrix.tolist(),
            "prior_level0_moving_to_fixed_translation_px_zyx": prior_level0_translation.tolist(),
            "prior_sampled_moving_to_fixed_matrix_zyx": prior_sampled_matrix.tolist(),
            "prior_sampled_moving_to_fixed_translation_px_zyx": prior_sampled_translation.tolist(),
            "residual_phase_shift_px_zyx": [float(value) for value in phase_shift],
        }
    if fit_mode == "rigid":
        init_stage_modes = ("translation", "rigid")
        refine_stage_modes = ("rigid",)
    else:
        init_stage_modes = ("translation", "rigid", "scale-9dof", "affine-12dof")
        refine_stage_modes = ("affine-12dof",)
    init_matrix, init_translation, init_corr = fit_affine_gpu(
        fixed_init,
        moving_init,
        initial_matrix=initial_matrix,
        initial_translation=initial_translation,
        max_iterations=max_iterations,
        stage_modes=init_stage_modes,
    )
    level0_matrix, level0_translation = model_to_level0(
        model_matrix=init_matrix,
        model_translation=init_translation,
        fixed_sampled_factor_zyx=init_factor,
        moving_sampled_factor_zyx=moving_init_factor,
    )
    moving_init_registered = transform_volume_gpu(moving_init, init_matrix, init_translation)
    fixed_start_l0, crop_selection = select_content_fixed_crop_start_l0(
        fixed_sampled=fixed_init,
        moving_registered_sampled=moving_init_registered,
        sampled_factor_zyx=init_factor,
        sampled_z_l0=fixed_z_l0,
        tile_shape_zyx=reference_tile.shape_zyx,
        crop_shape_zyx=refine_crop_shape_zyx,
    )
    moving_start_l0 = moving_crop_start_for_fixed_crop(
        fixed_start_zyx=fixed_start_l0,
        crop_shape_zyx=refine_crop_shape_zyx,
        full_matrix=level0_matrix,
        full_translation=level0_translation,
        fixed_shape_zyx=reference_tile.shape_zyx,
        moving_shape_zyx=moving_tile.shape_zyx,
    )
    local_init_matrix, local_init_translation = full_model_to_local(
        full_matrix=level0_matrix,
        full_translation=level0_translation,
        fixed_start_zyx=fixed_start_l0,
        moving_start_zyx=moving_start_l0,
        full_shape_zyx=reference_tile.shape_zyx,
        crop_shape_zyx=np.asarray(refine_crop_shape_zyx, dtype=np.int64),
    )
    fixed_refine_raw, fixed_crop_slices = _read_level0_crop(
        reference_tile,
        channel=reference_channel,
        start_zyx=fixed_start_l0,
        crop_shape_zyx=refine_crop_shape_zyx,
    )
    moving_refine_raw, moving_crop_slices = _read_level0_crop(
        moving_tile,
        channel=moving_channel,
        start_zyx=moving_start_l0,
        crop_shape_zyx=refine_crop_shape_zyx,
    )
    fixed_refine = _robust_norm(fixed_refine_raw)
    moving_refine = _robust_norm(moving_refine_raw)
    initial_moved = transform_volume_gpu(moving_refine, local_init_matrix, local_init_translation)
    local_refined_matrix, local_refined_translation, refined_corr = fit_affine_gpu(
        fixed_refine,
        moving_refine,
        initial_matrix=local_init_matrix,
        initial_translation=local_init_translation,
        max_iterations=max_iterations,
        stage_modes=refine_stage_modes,
    )
    refined_moved = transform_volume_gpu(moving_refine, local_refined_matrix, local_refined_translation)
    refined_matrix, refined_translation = local_model_to_full(
        local_matrix=local_refined_matrix,
        local_translation=local_refined_translation,
        fixed_start_zyx=fixed_start_l0,
        moving_start_zyx=moving_start_l0,
        full_shape_zyx=reference_tile.shape_zyx,
        crop_shape_zyx=np.asarray(refine_crop_shape_zyx, dtype=np.int64),
    )
    channel_affine_um = center_model_to_homogeneous_um(
        matrix_px=refined_matrix,
        translation_px=refined_translation,
        shape_zyx=reference_tile.shape_zyx,
        fixed_scale_um_zyx=reference_tile.scale_zyx_um,
        moving_scale_um_zyx=moving_tile.scale_zyx_um,
    )
    corr_unshifted = _corr(fixed_refine, moving_refine)
    corr_initial = _corr(fixed_refine, initial_moved)
    corr_refined = _corr(fixed_refine, refined_moved)
    gradient_unshifted = gradient_component_ncc_3d_gpu(fixed_refine, moving_refine)
    gradient_initial = gradient_component_ncc_3d_gpu(fixed_refine, initial_moved)
    gradient_refined = gradient_component_ncc_3d_gpu(fixed_refine, refined_moved)
    quality = affine_quality_gates(
        matrix=refined_matrix,
        corr_initial=corr_initial,
        corr_refined=corr_refined,
        gradient_initial=gradient_initial,
        gradient_refined=gradient_refined,
    )
    if not quality["accepted"]:
        raise AffineMeasurementRejected(f"Affine measurement for {moving_tile.tile} failed quality gates: {quality['reasons']}")
    if contact_sheet_path is not None:
        _write_affine_contact_sheet(
            fixed=fixed_refine,
            moving=moving_refine,
            initial_moved=initial_moved,
            refined_moved=refined_moved,
            output_path=contact_sheet_path,
            title=moving_tile.tile,
        )
    return {
        "reference_tile": reference_tile.tile,
        "moving_tile": moving_tile.tile,
        "reference_path": str(reference_tile.path),
        "moving_path": str(moving_tile.path),
        "status": "accepted",
        "affine_fit_mode": fit_mode,
        "tile_order": tile_order_metadata or {},
        "running_affine_prior": prior_details,
        "quality_gates": quality,
        "init_source_level": int(init_source_level),
        "moving_init_source_level": int(moving_source_level),
        "available_levels": {"fixed": int(fixed_available_levels), "moving": int(moving_available_levels)},
        "init_z_samples": int(init_z_samples),
        "init_volume_shape_zyx": [int(value) for value in fixed_init.shape],
        "init_effective_factor_zyx": [float(value) for value in init_factor],
        "moving_init_effective_factor_zyx": [float(value) for value in moving_init_factor],
        "init_phase_shift_px_zyx": [float(value) for value in phase_shift],
        "init_affine_corr": float(init_corr),
        "refine_crop_shape_zyx": [int(value) for value in refine_crop_shape_zyx],
        "refine_crop_selection": crop_selection,
        "fixed_refine_crop_slices_zyx": fixed_crop_slices,
        "moving_refine_crop_slices_zyx": moving_crop_slices,
        "initial_level0_moving_to_fixed_matrix_zyx": level0_matrix.tolist(),
        "initial_level0_moving_to_fixed_translation_px_zyx": level0_translation.tolist(),
        "initial_local_crop_moving_to_fixed_matrix_zyx": local_init_matrix.tolist(),
        "initial_local_crop_moving_to_fixed_translation_px_zyx": local_init_translation.tolist(),
        "refined_local_crop_moving_to_fixed_matrix_zyx": local_refined_matrix.tolist(),
        "refined_local_crop_moving_to_fixed_translation_px_zyx": local_refined_translation.tolist(),
        "refined_level0_moving_to_fixed_matrix_zyx": refined_matrix.tolist(),
        "refined_level0_moving_to_fixed_translation_px_zyx": refined_translation.tolist(),
        "refined_translation_um_zyx": (refined_translation * np.abs(reference_tile.scale_zyx_um)).tolist(),
        "channel_affine_um_zyx_homogeneous": channel_affine_um.tolist(),
        "corr_unshifted": corr_unshifted,
        "corr_initial_affine": corr_initial,
        "corr_refined_affine": corr_refined,
        "refined_objective_corr": float(refined_corr),
        "gradient_component_ncc_unshifted": gradient_unshifted,
        "gradient_component_ncc_initial_affine": gradient_initial,
        "gradient_component_ncc_refined_affine": gradient_refined,
        "contact_sheet": None if contact_sheet_path is None else str(contact_sheet_path.resolve()),
    }


def _validate_fit_downsample_zyx(fit_downsample_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    factors = tuple(int(value) for value in fit_downsample_zyx)
    if len(factors) != 3 or any(value < 1 for value in factors):
        raise ValueError(f"fit_downsample_zyx values must be >= 1, got {fit_downsample_zyx}")
    return factors


def _block_mean_downsample_zyx_cupy(array: Any, factors_zyx: tuple[int, int, int]) -> Any:
    factors = _validate_fit_downsample_zyx(factors_zyx)
    import cupy as cp
    from cucim.skimage.measure import block_reduce

    values = cp.ascontiguousarray(cp.asarray(array, dtype=cp.float32))
    if values.ndim != 3:
        raise ValueError(f"Expected 3D zyx array, got shape {values.shape}")
    if factors == (1, 1, 1):
        return values
    if any(size % factor != 0 for size, factor in zip(values.shape, factors, strict=True)):
        raise ValueError(f"Array shape {values.shape} must be divisible by fit_downsample_zyx={factors}")
    reduced = block_reduce(values, block_size=factors, func=cp.mean)
    return cp.ascontiguousarray(reduced.astype(cp.float32, copy=False))


def _model_to_fit_downsample(
    matrix_zyx: np.ndarray,
    translation_zyx: np.ndarray,
    factors_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    factors = np.asarray(_validate_fit_downsample_zyx(factors_zyx), dtype=np.float64)
    scale = np.diag(1.0 / factors)
    inverse_scale = np.diag(factors)
    matrix = np.asarray(matrix_zyx, dtype=np.float64)
    translation = np.asarray(translation_zyx, dtype=np.float64)
    return scale @ matrix @ inverse_scale, scale @ translation


def moving_xy_center_slab_z_pivot_in_fit_zyx(
    *,
    moving_full_shape_zyx: np.ndarray,
    moving_crop_start_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
    local_matrix_zyx: np.ndarray,
    local_translation_zyx: np.ndarray,
    fit_downsample_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Map the moving-image XY center and current slab Z center into the fit crop.

    The local affine is the centered moving-to-fixed model for the
    undownsampled crops, so it already incorporates the fixed crop start.
    Block means represent input block centers; the final conversion therefore
    includes the half-block offset rather than only dividing by the factor.
    """
    full_shape = np.asarray(moving_full_shape_zyx, dtype=np.float64)
    moving_start = np.asarray(moving_crop_start_zyx, dtype=np.float64)
    crop_shape = np.asarray(crop_shape_zyx, dtype=np.int64)
    matrix = np.asarray(local_matrix_zyx, dtype=np.float64)
    translation = np.asarray(local_translation_zyx, dtype=np.float64)
    factors = np.asarray(_validate_fit_downsample_zyx(fit_downsample_zyx), dtype=np.float64)
    if full_shape.shape != (3,) or moving_start.shape != (3,) or crop_shape.shape != (3,):
        raise ValueError("moving shape, crop start, and crop shape must be zyx vectors")
    if matrix.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("local affine must contain a 3x3 matrix and a zyx translation")
    if np.any(full_shape < 1) or np.any(crop_shape < 1):
        raise ValueError("moving and crop shapes must be positive")
    if np.any(crop_shape % factors.astype(np.int64)):
        raise ValueError(
            f"crop_shape_zyx={crop_shape.tolist()} must be divisible by fit_downsample_zyx={factors.astype(int).tolist()}"
        )

    crop_center = (crop_shape.astype(np.float64) - 1.0) / 2.0
    moving_pivot_local = (full_shape - 1.0) / 2.0 - moving_start
    moving_pivot_local[0] = crop_center[0]
    fixed_center_local = matrix @ (moving_pivot_local - crop_center) + crop_center + translation
    block_center_offset = (factors - 1.0) / 2.0
    return ((fixed_center_local - block_center_offset) / factors).astype(np.float32)


def _native_pull_from_fit_downsample(
    matrix_zyx: np.ndarray,
    offset_zyx: np.ndarray,
    factors_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    factors = np.asarray(_validate_fit_downsample_zyx(factors_zyx), dtype=np.float64)
    scale = np.diag(1.0 / factors)
    inverse_scale = np.diag(factors)
    matrix = np.asarray(matrix_zyx, dtype=np.float64)
    offset = np.asarray(offset_zyx, dtype=np.float64)
    return inverse_scale @ matrix @ scale, inverse_scale @ offset


def _set_record_stage_translation_um(record: dict[str, Any], translation_zyx_um: np.ndarray) -> None:
    translation = np.asarray(translation_zyx_um, dtype=np.float64)
    record.setdefault("translation_um", {})
    for axis, value in zip(DIMENSIONS, translation, strict=True):
        record["translation_um"][axis] = float(value)


def _tile_record_from_registration_record(record: dict[str, Any], *, fallback_side: str = "L") -> rough_legacy.TileRecord:
    if (
        record.get("side") in rough_legacy.SIDES
        and record.get("translation_um") is not None
        and record.get("scale_um") is not None
    ):
        return tile_record_from_position_record(record)
    patched = json.loads(json.dumps(record))
    patched["side"] = fallback_side
    if patched.get("translation_um") is None and patched.get("stage_translation_um") is not None:
        patched["translation_um"] = patched["stage_translation_um"]
    if patched.get("scale_um") is None and patched.get("stage_scale_um") is not None:
        patched["scale_um"] = patched["stage_scale_um"]
    if patched.get("scale_um") is None and patched.get("spacing_um") is not None:
        patched["scale_um"] = patched["spacing_um"]
    return tile_record_from_position_record(patched)


def _tile_center_registered_um(record: dict[str, Any]) -> np.ndarray:
    tile = _tile_record_from_registration_record(record)
    center_um = tile.shape_zyx.astype(np.float64) * tile.scale_zyx_um.astype(np.float64) / 2.0
    placed = _tile_record_placement_um(record) @ np.r_[center_um, 1.0]
    return placed[:3].astype(np.float64)


def _registration_edge_pairs(payload: dict[str, Any]) -> list[tuple[int, int]]:
    metrics = payload.get("metrics", {})
    used_edges = metrics.get("groupwise_resolution", {}).get("metrics", {}).get("used_edges", {})
    pairs = [
        tuple(int(value) for value in pair)
        for group in used_edges.values()
        for pair in group
        if len(pair) == 2
    ]
    if not pairs:
        pairs = [
            (int(edge["source"]), int(edge["target"]))
            for edge in metrics.get("pairwise_registration", {}).get("edges", [])
            if "source" in edge and "target" in edge
        ]
    seen: set[tuple[int, int]] = set()
    output: list[tuple[int, int]] = []
    for source, target in pairs:
        pair = (min(source, target), max(source, target))
        if pair not in seen:
            seen.add(pair)
            output.append(pair)
    return output


def _reference_tile_graph(payload: dict[str, Any]) -> dict[str, list[str]]:
    tiles = [str(record["tile"]) for record in payload.get("tiles", [])]
    graph: dict[str, list[str]] = {tile: [] for tile in tiles}
    for source, target in _registration_edge_pairs(payload):
        if source >= len(tiles) or target >= len(tiles):
            continue
        source_tile = tiles[source]
        target_tile = tiles[target]
        graph[source_tile].append(target_tile)
        graph[target_tile].append(source_tile)
    return graph


def _nearest_accepted_reference_tile(
    reference_tile: str,
    *,
    graph: dict[str, list[str]],
    accepted_rows: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    queue: list[tuple[str, list[str]]] = [(reference_tile, [reference_tile])]
    seen = {reference_tile}
    while queue:
        tile, path = queue.pop(0)
        if tile in accepted_rows:
            return {
                "donor_reference_tile": tile,
                "graph_path_reference_tiles": path,
                "graph_distance_edges": len(path) - 1,
                "method": "nearest_accepted_channel_affine_over_reference_registration_graph",
            }
        for neighbor in graph.get(tile, []):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, [*path, neighbor]))
    return None


def _nearest_accepted_reference_origin(
    reference_tile: str,
    *,
    reference_records_by_tile: dict[str, Any],
    accepted_rows: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    record = reference_records_by_tile.get(reference_tile)
    if record is None:
        return None
    center = (_tile_record_placement_um(record) @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64))[:3]
    candidates = []
    for tile in accepted_rows:
        accepted_record = reference_records_by_tile.get(tile)
        if accepted_record is None:
            continue
        accepted_center = (
            _tile_record_placement_um(accepted_record)
            @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )[:3]
        distance = float(np.linalg.norm(accepted_center - center))
        candidates.append((distance, tile))
    if not candidates:
        return None
    distance, donor_tile = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "donor_reference_tile": donor_tile,
        "graph_path_reference_tiles": [reference_tile, donor_tile],
        "graph_distance_edges": None,
        "nearest_registered_origin_distance_um": distance,
        "method": "nearest_accepted_channel_affine_by_reference_registered_origin",
    }


def image10_internal_geometry_qc(
    *,
    old_moving_registration: dict[str, Any],
    composed_moving_registration: dict[str, Any],
) -> dict[str, Any]:
    old_tiles = old_moving_registration.get("tiles", [])
    new_by_tile = {str(record["tile"]): record for record in composed_moving_registration.get("tiles", [])}
    records: list[dict[str, Any]] = []
    for source, target in _registration_edge_pairs(old_moving_registration):
        if source >= len(old_tiles) or target >= len(old_tiles):
            continue
        old_source = old_tiles[source]
        old_target = old_tiles[target]
        source_name = str(old_source["tile"])
        target_name = str(old_target["tile"])
        new_source = new_by_tile.get(source_name)
        new_target = new_by_tile.get(target_name)
        if new_source is None or new_target is None:
            records.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "status": "missing_composed_tile",
                }
            )
            continue
        old_delta = _tile_center_registered_um(old_target) - _tile_center_registered_um(old_source)
        new_delta = _tile_center_registered_um(new_target) - _tile_center_registered_um(new_source)
        residual = new_delta - old_delta
        records.append(
            {
                "source": source_name,
                "target": target_name,
                "status": "measured",
                "old_center_delta_um_zyx": old_delta.tolist(),
                "composed_center_delta_um_zyx": new_delta.tolist(),
                "residual_um_zyx": residual.tolist(),
                "residual_norm_um": float(np.linalg.norm(residual)),
            }
        )

    measured = [record for record in records if record["status"] == "measured"]
    residuals = (
        np.asarray([record["residual_um_zyx"] for record in measured], dtype=np.float64)
        if measured
        else np.empty((0, 3), dtype=np.float64)
    )
    norms = (
        np.asarray([record["residual_norm_um"] for record in measured], dtype=np.float64)
        if measured
        else np.empty((0,), dtype=np.float64)
    )
    return {
        "artifact_type": "lightsheet.image10_internal_geometry_qc.v1",
        "motivation": (
            "The old moving-channel registration is an internal-consistency reference: after cross-channel "
            "composition into Image_14 space, adjacent Image_10 tile deltas should remain close to the old "
            "Image_10-only seam geometry."
        ),
        "edge_count": len(records),
        "measured_edge_count": len(measured),
        "missing_edge_count": sum(1 for record in records if record["status"] != "measured"),
        "residual_um_zyx": None
        if residuals.size == 0
        else {
            "median": np.median(residuals, axis=0).tolist(),
            "mean": np.mean(residuals, axis=0).tolist(),
            "std": np.std(residuals, axis=0).tolist(),
            "max_abs": np.max(np.abs(residuals), axis=0).tolist(),
        },
        "residual_norm_um": None
        if norms.size == 0
        else {
            "median": float(np.median(norms)),
            "mean": float(np.mean(norms)),
            "max": float(np.max(norms)),
        },
        "edges": records,
    }


def _registration_spacing_um(payload: dict[str, Any]) -> np.ndarray:
    spacing = payload.get("spacing_um")
    if isinstance(spacing, dict):
        return np.asarray([spacing[dim] for dim in DIMENSIONS], dtype=np.float64)
    tiles = payload.get("tiles", [])
    if tiles:
        record = tiles[0]
        for key in ("stage_scale_um", "scale_um"):
            if key in record:
                return _record_vector_zyx(record, key)
    return np.ones(3, dtype=np.float64)


def write_affine_registration_from_reference(
    *,
    reference_registration_input: Path,
    output_registration: Path,
    reference_token: str,
    moving_token: str,
    rows: list[dict[str, Any]],
    moving_registration_input: Path | None = None,
    internal_geometry_qc_output: Path | None = None,
    require_all_tiles: bool = True,
) -> Path:
    payload = json.loads(reference_registration_input.read_text())
    moving_payload = json.loads(moving_registration_input.read_text()) if moving_registration_input is not None else None
    moving_records_by_tile = (
        {str(record["tile"]): record for record in moving_payload.get("tiles", [])}
        if moving_payload is not None
        else {}
    )
    rows_by_reference = {str(row["reference_tile"]): row for row in rows}
    adapted = json.loads(json.dumps(payload))
    adapted_tiles: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in payload["tiles"]:
        reference_tile = str(record["tile"])
        row = rows_by_reference.get(reference_tile)
        if row is None or row.get("status") != "accepted" or row.get("channel_affine_um_zyx_homogeneous") is None:
            missing.append(reference_tile)
            continue
        moving_tile = make_moving_tile_name(
            reference_tile,
            reference_token=reference_token,
            moving_token=moving_token,
        )
        moving_record = moving_records_by_tile.get(moving_tile)
        moving_stage = _record_stage_translation_um(moving_record) if moving_record is not None else _record_stage_translation_um(record)
        adapted_record = json.loads(json.dumps(moving_record if moving_record is not None else record))
        adapted_record["tile"] = moving_tile
        adapted_record["path"] = row["moving_path"]
        adapted_record["registered_affine"] = compose_registration_affine(
            reference_affine=record["registered_affine"],
            channel_affine_um=np.asarray(row["channel_affine_um_zyx_homogeneous"], dtype=np.float64),
            stage_translation_um_zyx=_record_stage_translation_um(record),
            moving_stage_translation_um_zyx=moving_stage,
        )
        adapted_tiles.append(adapted_record)
    if missing and require_all_tiles:
        raise ValueError(f"Missing affine measurements for registration tiles: {missing}")
    adapted["tiles"] = adapted_tiles
    if adapted_tiles:
        adapted["input_dir"] = str(Path(adapted_tiles[0]["path"]).parent)
    adapted["adapted_from"] = str(reference_registration_input.resolve())
    adapted["adaptation_method"] = "compose_reference_registered_affine_with_channel_affine"
    stage_translation_source: Literal["moving_registration_input", "reference_registration_input"] = (
        "moving_registration_input" if moving_registration_input is not None else "reference_registration_input"
    )
    adapted["transform_contract"] = RegistrationTransformContract(
        registered_affine_semantics="moving_tile_stage_um_to_reference_registered_um",
        source_space=f"{moving_token}_stage_um",
        target_space=f"{reference_token}_registered_um",
        composition_order=(
            "reference_registered_affine",
            "reference_stage_translation_um",
            "moving_to_reference_channel_affine_um",
            "inverse_moving_stage_translation_um",
        ),
        registered_affine_contains_full_channel_affine=True,
        stage_translation_source=stage_translation_source,
    ).model_dump(mode="json", by_alias=True)
    adapted.setdefault("diagnostics", {})["tile_affine_registration"] = {
        "reference_registration_input": str(reference_registration_input.resolve()),
        "moving_registration_input": None if moving_registration_input is None else str(moving_registration_input.resolve()),
        "reference_token": reference_token,
        "moving_token": moving_token,
        "tile_count": len(adapted_tiles),
        "missing_tile_count": len(missing),
        "missing_reference_tiles": missing,
        "registered_affine_contains_full_channel_affine": True,
        "stage_translation_source": stage_translation_source,
        "require_all_tiles": bool(require_all_tiles),
    }
    if moving_payload is not None:
        qc = image10_internal_geometry_qc(
            old_moving_registration=moving_payload,
            composed_moving_registration=adapted,
        )
        adapted["diagnostics"]["moving_internal_geometry_qc"] = qc
        if internal_geometry_qc_output is not None:
            internal_geometry_qc_output.parent.mkdir(parents=True, exist_ok=True)
            internal_geometry_qc_output.write_text(json.dumps(qc, indent=2) + "\n")
    output_registration.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_registration.with_name(f"{output_registration.name}.tmp")
    tmp_path.write_text(json.dumps(adapted, indent=2) + "\n")
    tmp_path.replace(output_registration)
    return output_registration.resolve()


def align_tiles_to_reference_affine(
    *,
    reference_position: Path,
    output_position: Path,
    output_dir: Path,
    reference_channel: int = 3,
    moving_channel: int = 0,
    reference_token: str = "488514561638",
    moving_token: str = "405",
    init_level: int = 2,
    init_z_samples: int = 200,
    refine_crop_shape_zyx: tuple[int, int, int] = (200, 480, 480),
    max_iterations: int = 20,
    render_contact_sheet: bool = True,
    fit_mode: AffineFitMode = "rigid",
    tile_order: AffineTileOrder = "grid-fanout",
    running_average_min_inliers: int = 5,
    reference_registration_input: Path | None = None,
    output_registration: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    if init_level < 0:
        raise ValueError("init_level must be non-negative")
    if init_z_samples < 1:
        raise ValueError("init_z_samples must be >= 1")
    if any(value < 1 for value in refine_crop_shape_zyx):
        raise ValueError("refine_crop_shape_zyx values must be >= 1")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if fit_mode not in {"rigid", "affine-12dof"}:
        raise ValueError(f"Unsupported affine fit mode {fit_mode!r}")
    if running_average_min_inliers < 1:
        raise ValueError("running_average_min_inliers must be >= 1")
    if (reference_registration_input is None) != (output_registration is None):
        raise ValueError("Affine registration output requires both reference_registration_input and output_registration")
    payload = json.loads(reference_position.read_text())
    reference_tiles = [tile_record_from_position_record(record) for record in payload["tiles"]]
    updated = json.loads(json.dumps(payload))
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "affine_contact_sheets"
    summary_path = output_dir / "tile_affine_alignment.json"
    registration_payload = json.loads(reference_registration_input.read_text()) if reference_registration_input is not None else None
    placement_records_by_tile = (
        {str(record["tile"]): record for record in registration_payload["tiles"]} if registration_payload is not None else {}
    )
    if tile_order == "grid-fanout":
        ordered_indices, order_metadata = grid_fanout_order(
            registration_payload["tiles"] if registration_payload is not None else payload["tiles"],
            reference_tiles,
        )
    elif tile_order == "input":
        ordered_indices = list(range(len(reference_tiles)))
        order_metadata = [
            {"tile": payload["tiles"][index]["tile"], "order_index": index, "method": "input"}
            for index in ordered_indices
        ]
    else:
        raise ValueError(f"Unknown affine tile order {tile_order!r}")
    order_metadata_by_index = dict(zip(ordered_indices, order_metadata, strict=True))
    rows: list[dict[str, Any]] = []
    running_prior = RunningRigidPrior(min_inliers=running_average_min_inliers, fit_mode=fit_mode)

    def write_summary() -> None:
        summary = {
            "schema_version": 1,
            "artifact_type": "lightsheet.tile_affine_summary.v1",
            "reference_position": str(reference_position.resolve()),
            "output_position": str(output_position.resolve()),
            "reference_channel": int(reference_channel),
            "moving_channel": int(moving_channel),
            "reference_token": reference_token,
            "moving_token": moving_token,
            "init_level": int(init_level),
            "init_z_samples": int(init_z_samples),
            "refine_level": 0,
            "refine_crop_shape_zyx": [int(value) for value in refine_crop_shape_zyx],
            "max_iterations": int(max_iterations),
            "fit_mode": fit_mode,
            "tile_order": tile_order,
            "tile_order_metadata": order_metadata,
            "running_average_min_inliers": int(running_average_min_inliers),
            "running_affine_mean": running_prior.latest_mean,
            "accepted_inlier_tiles": running_prior.tile_names,
            "reference_registration_input": None
            if reference_registration_input is None
            else str(reference_registration_input.resolve()),
            "output_registration": None if output_registration is None else str(output_registration.resolve()),
            "measurements": rows,
        }
        summary["summary_path"] = str(summary_path.resolve())
        tmp_path = summary_path.with_name(f"{summary_path.name}.tmp")
        tmp_path.write_text(json.dumps(summary, indent=2) + "\n")
        tmp_path.replace(summary_path)

    for index in ordered_indices:
        record = updated["tiles"][index]
        reference_tile = reference_tiles[index]
        placement_record = placement_records_by_tile.get(str(reference_tile.tile), payload["tiles"][index])
        placement_um = _tile_record_placement_um(placement_record)
        prior_channel_affine_um, mean_details = running_prior.prior_for(placement_um=placement_um)
        moving_path = corresponding_moving_path(reference_tile.path, reference_token=reference_token, moving_token=moving_token)
        moving_tile = make_moving_tile_record(reference_tile, moving_path)
        moving_tile_name = make_moving_tile_name(record["tile"], reference_token=reference_token, moving_token=moving_token)
        sheet_path = contact_dir / f"{moving_tile_name}_affine_contact_sheet.png" if render_contact_sheet else None
        try:
            row = _measure_tile_affine(
                reference_tile=reference_tile,
                moving_tile=moving_tile,
                reference_channel=reference_channel,
                moving_channel=moving_channel,
                init_level=init_level,
                init_z_samples=init_z_samples,
                refine_crop_shape_zyx=refine_crop_shape_zyx,
                max_iterations=max_iterations,
                contact_sheet_path=sheet_path,
                fit_mode=fit_mode,
                prior_channel_affine_um=prior_channel_affine_um,
                accepted_inliers_before_tile=running_prior.count,
                tile_order_metadata=order_metadata_by_index[index],
            )
        except AffineMeasurementRejected as exc:
            row = {
                "reference_tile": reference_tile.tile,
                "moving_tile": moving_tile_name,
                "reference_path": str(reference_tile.path),
                "moving_path": str(moving_path),
                "status": "rejected",
                "rejection_reason": f"{type(exc).__name__}: {exc}",
                "tile_order": order_metadata_by_index[index],
                "running_affine_prior": {
                    "used": prior_channel_affine_um is not None,
                    "accepted_inliers_before_tile": running_prior.count,
                    "mean": mean_details,
                },
                "contact_sheet": None if sheet_path is None else str(sheet_path.resolve()),
            }
            rows.append(row)
            write_summary()
            if progress is not None:
                progress(f"tile-affine {reference_tile.tile} -> {moving_tile_name} rejected: {row['rejection_reason']}")
            continue
        row["running_affine_prior"]["mean"] = mean_details
        shift_um = np.asarray(row["refined_translation_um_zyx"], dtype=np.float64)
        record["tile"] = moving_tile_name
        record["path"] = str(moving_path)
        for axis, value in zip(DIMENSIONS, shift_um, strict=True):
            record["translation_um"][axis] = float(record["translation_um"][axis] + value)
        rows.append(row)
        channel_affine_um = np.asarray(row["channel_affine_um_zyx_homogeneous"], dtype=np.float64)
        running_prior.add(
            tile_name=reference_tile.tile,
            channel_affine_um=channel_affine_um,
            placement_um=placement_um,
        )
        write_summary()
        if progress is not None:
            progress(
                f"tile-affine {reference_tile.tile} -> {moving_tile_name} "
                f"corr={row['corr_refined_affine']} grad={row['gradient_component_ncc_refined_affine']['mean']} "
                f"prior={row['running_affine_prior']['used']} inliers={running_prior.count}"
            )
    rejected = [row for row in rows if row.get("status") != "accepted"]
    if rejected:
        write_summary()
        raise ValueError(
            f"{len(rejected)} affine tile measurements were rejected; summary written to {summary_path.resolve()}"
        )
    diagnostics = updated.setdefault("diagnostics", {})
    diagnostics["tile_affine_alignment"] = {
        "reference_position": str(reference_position.resolve()),
        "reference_channel": int(reference_channel),
        "moving_channel": int(moving_channel),
        "reference_token": reference_token,
        "moving_token": moving_token,
        "init_level": int(init_level),
        "init_z_samples": int(init_z_samples),
        "refine_level": 0,
        "refine_crop_shape_zyx": [int(value) for value in refine_crop_shape_zyx],
        "max_iterations": int(max_iterations),
        "fit_mode": fit_mode,
        "tile_order": tile_order,
        "tile_order_metadata": order_metadata,
        "running_average_min_inliers": int(running_average_min_inliers),
        "running_affine_mean": running_prior.latest_mean,
        "position_file_translation": "compatibility translation component only",
        "full_affine_available_in_summary": True,
        "output_registration": None if output_registration is None else str(output_registration.resolve()),
        "measurements": rows,
    }
    updated["source"] = f"{payload.get('source', 'position file')} + tile affine alignment {moving_token} to {reference_token}"
    updated["derived_by"] = "lightsheet.tile_affine.v1"
    updated = stamp_artifact(updated, "lightsheet.position.v1")
    output_position.parent.mkdir(parents=True, exist_ok=True)
    output_position.write_text(json.dumps(updated, indent=2) + "\n")
    write_summary()
    if reference_registration_input is not None and output_registration is not None:
        write_affine_registration_from_reference(
            reference_registration_input=reference_registration_input,
            output_registration=output_registration,
            reference_token=reference_token,
            moving_token=moving_token,
            rows=rows,
        )
    return output_position.resolve()
