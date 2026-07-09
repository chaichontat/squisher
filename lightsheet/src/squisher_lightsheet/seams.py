from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import numpy as np


PHASE_CORRELATION_UPSAMPLE_FACTOR = 10


def cupy_argmax_index(values_gpu: Any) -> tuple[int, ...]:
    shape = tuple(int(value) for value in values_gpu.shape)
    if values_gpu.size == 0:
        raise ValueError("Cannot find argmax of an empty array")
    chunk_len = min(shape[0], 64) if shape else 1
    best_value = -np.inf
    best_flat = 0
    trailing_size = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    chunk_start = 0
    while chunk_start < shape[0]:
        chunk_stop = min(chunk_start + chunk_len, shape[0])
        values = values_gpu[chunk_start:chunk_stop].get()
        chunk_flat = int(np.argmax(values))
        chunk_value = float(values.ravel()[chunk_flat])
        if chunk_value > best_value:
            best_value = chunk_value
            best_flat = chunk_start * trailing_size + chunk_flat
        chunk_start = chunk_stop
    return tuple(int(value) for value in np.unravel_index(best_flat, shape))


@dataclass(frozen=True)
class RobustBoundarySettings:
    patch_shape_zyx: tuple[int, int, int] = (64, 512, 512)
    max_patches_per_edge: int = 128
    overlap_margin_zyx: tuple[int, int, int] = (0, 128, 128)
    min_inlier_patches_per_edge: int = 3
    min_center_z_p99: float = 180.0
    min_center_z_std: float = 8.0
    min_nonzero_fraction: float = 0.05
    min_std: float = 1e-3
    content_mask_percentile: float = 60.0
    min_content_fraction: float = 0.002
    min_content_voxels: int = 4096
    min_correlation: float = 0.60
    min_improvement: float = 0.02
    min_gradient_component_ncc: float = 0.15
    min_gradient_component_ncc_improvement: float = 0.0
    weak_edge_weight_factor: float = 0.1
    boundary_edge_workers: int = 2
    min_stable_correlation: float = 0.75
    max_stable_shift_zyx: tuple[float, float, float] = (1.0, 2.0, 2.0)
    max_correction_zyx: tuple[float, float, float] = (16.0, 96.0, 96.0)
    max_final_residual_zyx: tuple[float, float, float] = (4.0, 8.0, 8.0)
    huber_delta: float = 4.0
    irls_iterations: int = 5
    reference_xy_prior_weight: float = 0.01


@dataclass(frozen=True)
class TileBounds:
    start_zyx: tuple[float, float, float]
    stop_zyx: tuple[float, float, float]


@dataclass(frozen=True)
class BoundaryPatchSpec:
    pair: tuple[int, int]
    axis: str
    patch_index: int
    fixed_slices: tuple[slice, slice, slice]
    moving_slices: tuple[slice, slice, slice]
    overlap_start_zyx: tuple[int, int, int]
    overlap_shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class BoundaryConstraint:
    fixed: int
    moving: int
    pair: tuple[int, int]
    axis: str
    patch_index: int
    shift_zyx: tuple[float, float, float]
    weight: float
    correlation_before: float
    correlation_after: float
    improvement: float
    fixed_nonzero_fraction: float
    moving_nonzero_fraction: float
    fixed_std: float
    moving_std: float
    accepted: bool
    fixed_content_fraction: float = 0.0
    moving_content_fraction: float = 0.0
    gradient_component_ncc_before: float | None = None
    gradient_component_ncc_after: float | None = None
    gradient_component_ncc_improvement: float | None = None
    fixed_center_z_p99: float | None = None
    moving_center_z_p99: float | None = None
    fixed_center_z_std: float | None = None
    moving_center_z_std: float | None = None
    edge_status: str | None = None
    reject_reason: str | None = None
    final_residual_zyx: tuple[float, float, float] | None = None
    fixed_slices: tuple[slice, slice, slice] | None = None
    moving_slices: tuple[slice, slice, slice] | None = None
    source_label: str | None = None


def robust_boundary_settings() -> RobustBoundarySettings:
    return RobustBoundarySettings(
        patch_shape_zyx=(64, 512, 512),
        max_patches_per_edge=128,
        overlap_margin_zyx=(0, 128, 128),
    )


def overlap_recovery_settings() -> RobustBoundarySettings:
    return RobustBoundarySettings(
        patch_shape_zyx=(64, 256, 256),
        max_patches_per_edge=128,
        overlap_margin_zyx=(0, 96, 96),
        min_correlation=0.15,
        min_improvement=0.0,
    )


def boundary_axis_for_pair(bounds_a: TileBounds, bounds_b: TileBounds) -> str:
    centers_a = tuple((bounds_a.start_zyx[index] + bounds_a.stop_zyx[index]) / 2 for index in range(3))
    centers_b = tuple((bounds_b.start_zyx[index] + bounds_b.stop_zyx[index]) / 2 for index in range(3))
    deltas = [abs(centers_a[index] - centers_b[index]) for index in range(3)]
    return ("z", "y", "x")[int(max(range(3), key=lambda index: deltas[index]))]


def clipped_slice(start: int, size: int, limit: int) -> slice:
    start = max(0, min(start, limit))
    stop = max(start, min(start + size, limit))
    return slice(start, stop)


def evenly_spaced_starts(start: int, stop: int, size: int, count: int) -> list[int]:
    if stop <= start:
        return []
    span = stop - start
    if span <= size or count <= 1:
        return [start]
    last = stop - size
    return [int(round(start + (last - start) * index / (count - 1))) for index in range(count)]


def slice_shape(slices: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    return tuple(int(slc.stop or 0) - int(slc.start or 0) for slc in slices)


def local_slices_for_world_patch(
    *,
    tile_shape_zyx: tuple[int, int, int],
    bounds: TileBounds,
    world_start_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    starts = [int(round(world_start_zyx[index] - bounds.start_zyx[index])) for index in range(3)]
    return tuple(clipped_slice(starts[index], patch_shape_zyx[index], tile_shape_zyx[index]) for index in range(3))


def overlap_plus_margin_support(
    fixed_bounds: TileBounds,
    moving_bounds: TileBounds,
    settings: RobustBoundarySettings,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None:
    overlap_start = tuple(
        int(math.ceil(max(fixed_bounds.start_zyx[index], moving_bounds.start_zyx[index])))
        for index in range(3)
    )
    overlap_stop = tuple(
        int(math.floor(min(fixed_bounds.stop_zyx[index], moving_bounds.stop_zyx[index])))
        for index in range(3)
    )
    overlap_shape = tuple(overlap_stop[index] - overlap_start[index] for index in range(3))
    if any(size <= 0 for size in overlap_shape):
        return None

    union_start = tuple(
        int(math.floor(min(fixed_bounds.start_zyx[index], moving_bounds.start_zyx[index])))
        for index in range(3)
    )
    union_stop = tuple(
        int(math.ceil(max(fixed_bounds.stop_zyx[index], moving_bounds.stop_zyx[index])))
        for index in range(3)
    )
    support_start = tuple(
        max(union_start[index], overlap_start[index] - int(settings.overlap_margin_zyx[index]))
        for index in range(3)
    )
    support_stop = tuple(
        min(union_stop[index], overlap_stop[index] + int(settings.overlap_margin_zyx[index]))
        for index in range(3)
    )
    support_shape = tuple(support_stop[index] - support_start[index] for index in range(3))
    if any(size <= 0 for size in support_shape):
        return None
    return support_start, support_stop, overlap_shape


def sample_boundary_patches_from_bounds(
    *,
    tile_shapes_zyx: Sequence[tuple[int, int, int]],
    bounds: Sequence[TileBounds],
    pairs: Sequence[tuple[int, int]],
    settings: RobustBoundarySettings,
) -> list[BoundaryPatchSpec]:
    patch_specs: list[BoundaryPatchSpec] = []

    for fixed, moving in pairs:
        fixed_bounds = bounds[fixed]
        moving_bounds = bounds[moving]
        support = overlap_plus_margin_support(fixed_bounds, moving_bounds, settings)
        if support is None:
            continue
        support_start, support_stop, overlap_shape = support
        support_shape = tuple(support_stop[index] - support_start[index] for index in range(3))
        patch_shape = tuple(min(settings.patch_shape_zyx[index], support_shape[index]) for index in range(3))
        axis = boundary_axis_for_pair(fixed_bounds, moving_bounds)
        varying_axes = [index for index, dim in enumerate(("z", "y", "x")) if dim != axis]
        counts = {index: 1 for index in range(3)}
        max_patches = max(1, int(settings.max_patches_per_edge))
        if max_patches > 1:
            first_axis = varying_axes[0]
            second_axis = varying_axes[1]
            counts[first_axis] = max(1, int(math.floor(math.sqrt(max_patches))))
            counts[second_axis] = max(1, int(math.ceil(max_patches / counts[first_axis])))

        starts_by_axis = [
            evenly_spaced_starts(
                support_start[index],
                support_stop[index],
                patch_shape[index],
                counts[index],
            )
            for index in range(3)
        ]
        import itertools

        for patch_index, patch_start in enumerate(itertools.product(*starts_by_axis)):
            if patch_index >= max_patches:
                break
            fixed_slices = local_slices_for_world_patch(
                tile_shape_zyx=tile_shapes_zyx[fixed],
                bounds=fixed_bounds,
                world_start_zyx=tuple(int(value) for value in patch_start),
                patch_shape_zyx=patch_shape,
            )
            moving_slices = local_slices_for_world_patch(
                tile_shape_zyx=tile_shapes_zyx[moving],
                bounds=moving_bounds,
                world_start_zyx=tuple(int(value) for value in patch_start),
                patch_shape_zyx=patch_shape,
            )
            shared_shape = tuple(
                min(slice_shape(fixed_slices)[index], slice_shape(moving_slices)[index])
                for index in range(3)
            )
            if any(size <= 1 for size in shared_shape):
                continue
            fixed_slices = tuple(
                slice(fixed_slices[index].start, fixed_slices[index].start + shared_shape[index])
                for index in range(3)
            )
            moving_slices = tuple(
                slice(moving_slices[index].start, moving_slices[index].start + shared_shape[index])
                for index in range(3)
            )
            patch_specs.append(
                BoundaryPatchSpec(
                    pair=(fixed, moving),
                    axis=axis,
                    patch_index=patch_index,
                    fixed_slices=fixed_slices,
                    moving_slices=moving_slices,
                    overlap_start_zyx=tuple(int(value) for value in patch_start),
                    overlap_shape_zyx=shared_shape,
                )
            )
    return patch_specs


def quantize_phase_shift(
    shift: float,
    *,
    upsample_factor: int = PHASE_CORRELATION_UPSAMPLE_FACTOR,
) -> float:
    if upsample_factor <= 1:
        quantized = float(round(shift))
    else:
        quantized = round(float(shift) * upsample_factor) / float(upsample_factor)
    return 0.0 if quantized == 0.0 else quantized


def quadratic_subpixel_peak_offset(left: float, center: float, right: float) -> float:
    if not all(math.isfinite(value) for value in (left, center, right)):
        return 0.0
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    if not math.isfinite(offset):
        return 0.0
    return min(0.5, max(-0.5, float(offset)))


def refined_phase_shift_from_samples(
    peak_index: int,
    size: int,
    left: float,
    center: float,
    right: float,
    *,
    upsample_factor: int = PHASE_CORRELATION_UPSAMPLE_FACTOR,
) -> float:
    refined = float(peak_index) + quadratic_subpixel_peak_offset(left, center, right)
    if refined > float(size) / 2.0:
        refined -= float(size)
    return quantize_phase_shift(refined, upsample_factor=upsample_factor)


def upsampled_dft_gpu(
    data: Any,
    upsampled_region_size: int,
    upsample_factor: int,
    axis_offsets: list[float],
) -> Any:
    import cupy as cp

    if len(axis_offsets) != data.ndim:
        raise ValueError("axis_offsets must match data dimensionality")
    float_dtype = data.real.dtype
    im2pi = 1j * 2.0 * math.pi
    for n_items, axis_offset in zip(data.shape[::-1], axis_offsets[::-1], strict=True):
        sample_positions = cp.arange(upsampled_region_size, dtype=float_dtype) - float(axis_offset)
        frequencies = cp.fft.fftfreq(n_items, d=float(upsample_factor)).astype(float_dtype, copy=False)
        kernel = cp.exp((-im2pi * sample_positions[:, None]) * frequencies[None, :])
        kernel = kernel.astype(data.dtype, copy=False)
        data = cp.tensordot(kernel, data, axes=(1, -1))
    return data


def refined_phase_shifts_gpu(
    image_product: Any,
    peak_index: tuple[int, ...],
    *,
    upsample_factor: int,
) -> tuple[float, ...]:
    import cupy as cp

    shape = tuple(int(value) for value in image_product.shape)
    shifts = []
    for index, size in zip(peak_index, shape, strict=True):
        shift = float(index)
        if index > size // 2:
            shift -= float(size)
        shifts.append(shift)

    if upsample_factor <= 1:
        return tuple(quantize_phase_shift(shift, upsample_factor=1) for shift in shifts)

    shifts = [quantize_phase_shift(shift, upsample_factor=upsample_factor) for shift in shifts]
    upsampled_region_size = int(math.ceil(float(upsample_factor) * 1.5))
    dftshift = int(math.trunc(upsampled_region_size / 2.0))
    sample_region_offset = [float(dftshift) - shift * float(upsample_factor) for shift in shifts]
    refined = upsampled_dft_gpu(
        image_product.conj(),
        upsampled_region_size,
        upsample_factor,
        sample_region_offset,
    ).conj()
    refined_peak = cupy_argmax_index(cp.abs(refined))
    output = []
    for shift, maximum, size in zip(shifts, refined_peak, shape, strict=True):
        refined_shift = shift + (float(maximum) - float(dftshift)) / float(upsample_factor)
        if size == 1:
            refined_shift = 0.0
        output.append(quantize_phase_shift(refined_shift, upsample_factor=upsample_factor))
    return tuple(output)


def content_mask_gpu_array(values_gpu: Any, settings: RobustBoundarySettings) -> Any:
    import cupy as cp

    finite_mask = cp.isfinite(values_gpu)
    positive = values_gpu[finite_mask & (values_gpu > 0)]
    if int(positive.size) < settings.min_content_voxels:
        return cp.zeros(values_gpu.shape, dtype=cp.bool_)
    threshold = cp.percentile(positive, settings.content_mask_percentile)
    return finite_mask & (values_gpu > threshold)


def mask_fraction_gpu_array(mask_gpu: Any) -> float:
    import cupy as cp

    if mask_gpu.size == 0:
        return 0.0
    return float((cp.count_nonzero(mask_gpu) / mask_gpu.size).get())


def masked_centered_array(values_gpu: Any, mask_gpu: Any, min_voxels: int) -> Any:
    import cupy as cp

    centered = cp.zeros_like(values_gpu, dtype=cp.float32)
    if int(cp.count_nonzero(mask_gpu).get()) < min_voxels:
        return centered
    masked_values = values_gpu[mask_gpu]
    centered[mask_gpu] = masked_values - cp.mean(masked_values)
    return centered


def phase_correlation_shift_gpu_arrays(
    fixed_gpu: Any,
    moving_gpu: Any,
    fixed_mask_gpu: Any | None = None,
    moving_mask_gpu: Any | None = None,
    *,
    min_mask_voxels: int = 2,
    upsample_factor: int = PHASE_CORRELATION_UPSAMPLE_FACTOR,
) -> tuple[tuple[float, float, float], float]:
    import cupy as cp

    if fixed_mask_gpu is not None and moving_mask_gpu is not None:
        fixed_gpu = masked_centered_array(fixed_gpu, fixed_mask_gpu, min_mask_voxels)
        moving_gpu = masked_centered_array(moving_gpu, moving_mask_gpu, min_mask_voxels)
    else:
        fixed_gpu = fixed_gpu - cp.mean(fixed_gpu)
        moving_gpu = moving_gpu - cp.mean(moving_gpu)
    cross_power = cp.fft.fftn(fixed_gpu) * cp.conj(cp.fft.fftn(moving_gpu))
    magnitude = cp.abs(cross_power)
    cp.maximum(magnitude, cp.finfo(cp.float32).eps, out=magnitude)
    cross_power /= magnitude
    del magnitude
    corr = cp.real(cp.fft.ifftn(cross_power))
    peak_index = cupy_argmax_index(corr)
    peak = float(corr[peak_index].get())
    shifts = refined_phase_shifts_gpu(cross_power, peak_index, upsample_factor=upsample_factor)
    return (shifts[0], shifts[1], shifts[2]), peak


def integer_shift_slices(
    shape: tuple[int, ...],
    shift_zyx: tuple[float, float, float],
) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    source_slices = []
    destination_slices = []
    for size, shift in zip(shape, shift_zyx, strict=True):
        rounded = int(round(shift))
        if abs(rounded) >= size:
            return None
        if rounded > 0:
            source_slices.append(slice(0, size - rounded))
            destination_slices.append(slice(rounded, size))
        elif rounded < 0:
            source_slices.append(slice(-rounded, size))
            destination_slices.append(slice(0, size + rounded))
        else:
            source_slices.append(slice(None))
            destination_slices.append(slice(None))
    return tuple(source_slices), tuple(destination_slices)


def is_integer_shift(shift_zyx: tuple[float, float, float]) -> bool:
    return all(abs(float(shift) - round(float(shift))) < 1e-6 for shift in shift_zyx)


def shift_array_gpu_array(source: Any, shift_zyx: tuple[float, float, float]) -> Any:
    import cupy as cp

    if not is_integer_shift(shift_zyx):
        import cupyx.scipy.ndimage as cpx_ndimage

        is_bool = source.dtype == cp.bool_
        return cpx_ndimage.shift(
            source,
            shift=shift_zyx,
            order=0 if is_bool else 1,
            mode="constant",
            cval=0 if is_bool else 0.0,
            prefilter=False,
        )

    shifted = cp.zeros_like(source)
    slices = integer_shift_slices(tuple(source.shape), shift_zyx)
    if slices is None:
        return shifted
    source_slices, destination_slices = slices
    shifted[destination_slices] = source[source_slices]
    return shifted


def shift_array_cpu(array: Any, shift_zyx: tuple[float, float, float]) -> Any:
    source = np.asarray(array)
    if not is_integer_shift(shift_zyx):
        import scipy.ndimage as scipy_ndimage

        is_bool = source.dtype == np.bool_
        return scipy_ndimage.shift(
            source,
            shift=shift_zyx,
            order=0 if is_bool else 1,
            mode="constant",
            cval=0 if is_bool else 0.0,
            prefilter=False,
        )

    shifted = np.zeros_like(source)
    slices = integer_shift_slices(tuple(source.shape), shift_zyx)
    if slices is None:
        return shifted
    source_slices, destination_slices = slices
    shifted[destination_slices] = source[source_slices]
    return shifted


def patch_support_stats_gpu(values_gpu: Any) -> tuple[float, float]:
    import cupy as cp

    finite_mask = cp.isfinite(values_gpu)
    finite_count = int(cp.count_nonzero(finite_mask).get())
    if finite_count == 0:
        return 0.0, 0.0
    finite = values_gpu[finite_mask]
    nonzero_fraction = float((cp.count_nonzero(finite) / finite_count).get())
    return nonzero_fraction, float(cp.std(finite).get())


def normalized_cross_correlation_gpu_arrays(
    fixed_gpu: Any,
    moving_gpu: Any,
    mask_gpu: Any | None = None,
    *,
    min_voxels: int = 2,
) -> float:
    import cupy as cp

    finite_mask = cp.isfinite(fixed_gpu) & cp.isfinite(moving_gpu)
    if mask_gpu is not None:
        finite_mask &= mask_gpu
    if int(cp.count_nonzero(finite_mask).get()) < min_voxels:
        return float("nan")
    fixed_values = fixed_gpu[finite_mask]
    moving_values = moving_gpu[finite_mask]
    fixed_values = fixed_values - cp.mean(fixed_values)
    moving_values = moving_values - cp.mean(moving_values)
    denominator = cp.sqrt(cp.sum(fixed_values * fixed_values) * cp.sum(moving_values * moving_values))
    if float(denominator.get()) == 0.0:
        return float("nan")
    return float((cp.sum(fixed_values * moving_values) / denominator).get())


def evaluate_boundary_patch_gpu(
    fixed_patch: Any,
    moving_patch: Any,
    settings: RobustBoundarySettings,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
    tuple[float, float, float],
    float,
    float,
]:
    import cupy as cp

    fixed_gpu = cp.asarray(np.asarray(fixed_patch, dtype=np.float32))
    moving_gpu = cp.asarray(np.asarray(moving_patch, dtype=np.float32))
    fixed_support = patch_support_stats_gpu(fixed_gpu)
    moving_support = patch_support_stats_gpu(moving_gpu)
    fixed_content_mask = content_mask_gpu_array(fixed_gpu, settings)
    moving_content_mask = content_mask_gpu_array(moving_gpu, settings)
    fixed_content = mask_fraction_gpu_array(fixed_content_mask)
    moving_content = mask_fraction_gpu_array(moving_content_mask)
    content_overlap_mask = fixed_content_mask & moving_content_mask
    corr_before = normalized_cross_correlation_gpu_arrays(
        fixed_gpu,
        moving_gpu,
        content_overlap_mask,
        min_voxels=settings.min_content_voxels,
    )
    if (
        fixed_support[0] < settings.min_nonzero_fraction
        or moving_support[0] < settings.min_nonzero_fraction
        or fixed_support[1] < settings.min_std
        or moving_support[1] < settings.min_std
        or fixed_content < settings.min_content_fraction
        or moving_content < settings.min_content_fraction
    ):
        return fixed_support, moving_support, (fixed_content, moving_content), corr_before, (0.0, 0.0, 0.0), float("nan"), corr_before
    shift, peak = phase_correlation_shift_gpu_arrays(
        fixed_gpu,
        moving_gpu,
        fixed_content_mask,
        moving_content_mask,
        min_mask_voxels=settings.min_content_voxels,
    )
    shifted = shift_array_gpu_array(moving_gpu, shift)
    shifted_moving_content_mask = shift_array_gpu_array(moving_content_mask, shift)
    shifted_content_overlap_mask = fixed_content_mask & shifted_moving_content_mask
    corr_after = normalized_cross_correlation_gpu_arrays(
        fixed_gpu,
        shifted,
        shifted_content_overlap_mask,
        min_voxels=settings.min_content_voxels,
    )
    return fixed_support, moving_support, (fixed_content, moving_content), corr_before, shift, peak, corr_after


def read_native_patch(array: Any, axes: str, channel: int, slices_zyx: tuple[slice, slice, slice]) -> Any:
    if axes == "CZYX":
        return np.asarray(array[(channel, *slices_zyx)])
    if axes == "ZYX":
        if channel != 0:
            raise ValueError(f"Channel {channel} out of range for single-channel ZYX tile")
        return np.asarray(array[slices_zyx])
    raise ValueError(f"Expected CZYX or ZYX axes, got {axes}")


def is_stable_aligned_patch(
    corr_before: float,
    shift_zyx: tuple[float, float, float],
    settings: RobustBoundarySettings,
) -> bool:
    return (
        math.isfinite(corr_before)
        and corr_before >= settings.min_stable_correlation
        and all(abs(shift_zyx[index]) <= settings.max_stable_shift_zyx[index] for index in range(3))
    )


def accepted_boundary_weight(
    *,
    improvement: float,
    fixed_content: float,
    moving_content: float,
    peak: float,
    stable_alignment: bool,
    settings: RobustBoundarySettings,
) -> float:
    support = min(fixed_content, moving_content)
    peak_weight = max(0.0, peak) if math.isfinite(peak) else 1.0
    evidence = settings.min_improvement if stable_alignment else max(settings.min_improvement, improvement)
    return evidence * support * peak_weight


def boundary_constraint_from_evaluation(
    *,
    spec: BoundaryPatchSpec,
    fixed_index: int,
    moving_index: int,
    fixed_slices: tuple[slice, slice, slice],
    moving_slices: tuple[slice, slice, slice],
    fixed_support: tuple[float, float],
    moving_support: tuple[float, float],
    fixed_content: float,
    moving_content: float,
    corr_before: float,
    shift: tuple[float, float, float],
    peak: float,
    corr_after: float,
    gradient_before: float,
    gradient_after: float,
    settings: RobustBoundarySettings,
    source_label: str | None = None,
) -> BoundaryConstraint:
    fixed_nonzero, fixed_std = fixed_support
    moving_nonzero, moving_std = moving_support
    reject_reason = None
    if fixed_nonzero < settings.min_nonzero_fraction or moving_nonzero < settings.min_nonzero_fraction:
        reject_reason = "low_nonzero_fraction"
    elif fixed_std < settings.min_std or moving_std < settings.min_std:
        reject_reason = "low_texture"
    elif fixed_content < settings.min_content_fraction or moving_content < settings.min_content_fraction:
        reject_reason = "low_content"

    stable_alignment = False
    if reject_reason is None:
        improvement = (
            gradient_after - gradient_before
            if math.isfinite(gradient_before) and math.isfinite(gradient_after)
            else float("nan")
        )
        stable_alignment = is_stable_aligned_patch(corr_before, shift, settings)
        if stable_alignment and (
            not math.isfinite(improvement) or improvement < settings.min_gradient_component_ncc_improvement
        ):
            shift = (0.0, 0.0, 0.0)
            corr_after = corr_before
            gradient_after = gradient_before
            improvement = 0.0
        elif not math.isfinite(gradient_after) or gradient_after < settings.min_gradient_component_ncc:
            reject_reason = "weak_gradient_component_ncc"
        elif not math.isfinite(improvement) or improvement < settings.min_gradient_component_ncc_improvement:
            reject_reason = "weak_gradient_component_ncc_improvement"
    else:
        improvement = 0.0

    if reject_reason is None:
        weight = accepted_boundary_weight(
            improvement=improvement,
            fixed_content=fixed_content,
            moving_content=moving_content,
            peak=peak,
            stable_alignment=stable_alignment,
            settings=settings,
        )
        if weight <= 0:
            reject_reason = "zero_weight"
    else:
        weight = 0.0

    return BoundaryConstraint(
        fixed=fixed_index,
        moving=moving_index,
        pair=spec.pair,
        axis=spec.axis,
        patch_index=spec.patch_index,
        shift_zyx=shift,
        weight=weight,
        correlation_before=corr_before,
        correlation_after=corr_after,
        improvement=improvement,
        fixed_nonzero_fraction=fixed_nonzero,
        moving_nonzero_fraction=moving_nonzero,
        fixed_std=fixed_std,
        moving_std=moving_std,
        accepted=reject_reason is None,
        fixed_content_fraction=fixed_content,
        moving_content_fraction=moving_content,
        gradient_component_ncc_before=None if not math.isfinite(gradient_before) else gradient_before,
        gradient_component_ncc_after=None if not math.isfinite(gradient_after) else gradient_after,
        gradient_component_ncc_improvement=None if not math.isfinite(improvement) else improvement,
        reject_reason=reject_reason,
        fixed_slices=fixed_slices,
        moving_slices=moving_slices,
        source_label=source_label,
    )


def center_plane(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        return np.asarray(array[int(array.shape[0] // 2)])
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected 2D or 3D patch, got shape {array.shape}")


def center_z_content_stats(image: Any) -> dict[str, float]:
    plane = np.asarray(center_plane(image), dtype=np.float32)
    finite = plane[np.isfinite(plane)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size == 0:
        return {"p99": 0.0, "std": 0.0}
    return {
        "p99": float(np.percentile(sample, 99.0)),
        "std": float(np.std(sample)),
    }


def center_z_content_prefilter_reason(
    fixed_patch: Any,
    moving_patch: Any,
    settings: RobustBoundarySettings,
) -> tuple[str | None, dict[str, float], dict[str, float]]:
    fixed_stats = center_z_content_stats(fixed_patch)
    moving_stats = center_z_content_stats(moving_patch)
    if min(fixed_stats["p99"], moving_stats["p99"]) < settings.min_center_z_p99:
        return "low_center_z_p99", fixed_stats, moving_stats
    if min(fixed_stats["std"], moving_stats["std"]) < settings.min_center_z_std:
        return "low_center_z_std", fixed_stats, moving_stats
    return None, fixed_stats, moving_stats


def boundary_constraint_from_prefilter_rejection(
    *,
    spec: BoundaryPatchSpec,
    fixed_index: int,
    moving_index: int,
    fixed_slices: tuple[slice, slice, slice],
    moving_slices: tuple[slice, slice, slice],
    fixed_stats: dict[str, float],
    moving_stats: dict[str, float],
    reject_reason: str,
    source_label: str | None = None,
) -> BoundaryConstraint:
    return BoundaryConstraint(
        fixed=fixed_index,
        moving=moving_index,
        pair=spec.pair,
        axis=spec.axis,
        patch_index=spec.patch_index,
        shift_zyx=(0.0, 0.0, 0.0),
        weight=0.0,
        correlation_before=float("nan"),
        correlation_after=float("nan"),
        improvement=0.0,
        fixed_nonzero_fraction=0.0,
        moving_nonzero_fraction=0.0,
        fixed_std=float(fixed_stats["std"]),
        moving_std=float(moving_stats["std"]),
        accepted=False,
        fixed_center_z_p99=float(fixed_stats["p99"]),
        moving_center_z_p99=float(moving_stats["p99"]),
        fixed_center_z_std=float(fixed_stats["std"]),
        moving_center_z_std=float(moving_stats["std"]),
        reject_reason=reject_reason,
        fixed_slices=fixed_slices,
        moving_slices=moving_slices,
        source_label=source_label,
    )


def robust_norm_2d(image: Any) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(sample, [1.0, 99.7])
    return np.clip((image - low) / max(float(high - low), 1.0), 0.0, 1.0).astype(np.float32)


def signal_mask_2d(image: Any) -> np.ndarray:
    from scipy import ndimage as scipy_ndimage

    norm = robust_norm_2d(image)
    positive = norm[norm > 0]
    if positive.size == 0:
        return np.zeros(norm.shape, dtype=bool)
    mask = norm > max(0.03, float(np.percentile(positive, 45.0)))
    mask = scipy_ndimage.binary_dilation(mask, iterations=1)
    mask = scipy_ndimage.binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)


def ncc_values_2d(a: Any, b: Any, mask: np.ndarray, *, min_pixels: int = 256) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if int(np.count_nonzero(mask)) < min_pixels:
        return float("nan")
    aa = a[mask].astype(np.float64, copy=False)
    bb = b[mask].astype(np.float64, copy=False)
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator else float("nan")


def center_z_gradient_component_ncc(fixed: Any, moving: Any) -> float:
    from scipy import ndimage as scipy_ndimage

    fixed_array = np.asarray(fixed)
    moving_array = np.asarray(moving)
    if fixed_array.ndim == 3:
        fixed_plane = fixed_array[int(fixed_array.shape[0] // 2)]
    elif fixed_array.ndim == 2:
        fixed_plane = fixed_array
    else:
        raise ValueError(f"Expected 2D or 3D fixed patch, got shape {fixed_array.shape}")
    if moving_array.ndim == 3:
        moving_plane = moving_array[int(moving_array.shape[0] // 2)]
    elif moving_array.ndim == 2:
        moving_plane = moving_array
    else:
        raise ValueError(f"Expected 2D or 3D moving patch, got shape {moving_array.shape}")

    fixed_norm = robust_norm_2d(fixed_plane)
    moving_norm = robust_norm_2d(moving_plane)
    moving_mask = scipy_ndimage.binary_dilation(signal_mask_2d(moving_norm), iterations=2)
    values = []
    for axis in range(2):
        value = ncc_values_2d(
            scipy_ndimage.sobel(fixed_norm, axis=axis),
            scipy_ndimage.sobel(moving_norm, axis=axis),
            moving_mask,
        )
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def center_z_gradient_component_ncc_after_shift(
    fixed_patch: Any,
    moving_patch: Any,
    shift_zyx: tuple[float, float, float],
) -> tuple[float, float]:
    """Score seam evidence on the center z plane after applying the 3D patch shift."""

    shifted = shifted_center_plane_cpu(moving_patch, shift_zyx)
    return (
        center_z_gradient_component_ncc(np.asarray(fixed_patch), np.asarray(moving_patch)),
        center_z_gradient_component_ncc(center_plane(fixed_patch), shifted),
    )


def shifted_center_plane_cpu(
    source: Any,
    shift_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Return only the center-z plane of a 3D patch after applying a z/y/x shift."""

    from scipy import ndimage as scipy_ndimage

    array = np.asarray(source)
    if array.ndim == 2:
        return shift_array_cpu(array, (shift_zyx[1], shift_zyx[2]))
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D patch, got shape {array.shape}")

    z_center = int(array.shape[0] // 2)
    y_coords, x_coords = np.meshgrid(
        np.arange(array.shape[1], dtype=np.float32) - np.float32(shift_zyx[1]),
        np.arange(array.shape[2], dtype=np.float32) - np.float32(shift_zyx[2]),
        indexing="ij",
    )
    z_coords = np.full_like(y_coords, np.float32(z_center) - np.float32(shift_zyx[0]))
    coords = np.asarray([z_coords, y_coords, x_coords], dtype=np.float32)
    return scipy_ndimage.map_coordinates(
        array.astype(np.float32, copy=False),
        coords,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def mark_boundary_inliers_by_edge(
    constraints: list[BoundaryConstraint],
    settings: RobustBoundarySettings,
) -> list[BoundaryConstraint]:
    """Keep only mutually compatible accepted patch shifts on each seam edge."""
    if settings.min_inlier_patches_per_edge <= 1:
        return constraints

    from squisher_lightsheet.tile_phase import select_inlier_patch_measurements

    grouped: dict[tuple[tuple[int, int], str], list[int]] = {}
    for index, constraint in enumerate(constraints):
        grouped.setdefault((constraint.pair, constraint.axis), []).append(index)

    updated = list(constraints)
    for indices in grouped.values():
        accepted_indices = [index for index in indices if constraints[index].accepted]
        if len(accepted_indices) < settings.min_inlier_patches_per_edge:
            for index in accepted_indices:
                updated[index] = replace(
                    updated[index],
                    weight=max(updated[index].weight, 0.0) * settings.weak_edge_weight_factor,
                    edge_status="downweighted_no_inlier_cluster",
                )
            continue

        shifts = np.asarray([constraints[index].shift_zyx for index in accepted_indices], dtype=np.float64)
        try:
            inlier_mask, _ = select_inlier_patch_measurements(
                shifts,
                min_inliers=settings.min_inlier_patches_per_edge,
            )
        except ValueError:
            inlier_mask = np.zeros(len(accepted_indices), dtype=bool)

        if not np.any(inlier_mask):
            for index in accepted_indices:
                updated[index] = replace(
                    updated[index],
                    weight=max(updated[index].weight, 0.0) * settings.weak_edge_weight_factor,
                    edge_status="downweighted_no_inlier_cluster",
                )
            continue

        for index, inlier in zip(accepted_indices, inlier_mask, strict=True):
            if bool(inlier):
                updated[index] = replace(updated[index], edge_status="inlier_cluster")
                continue
            updated[index] = replace(
                updated[index],
                accepted=False,
                weight=0.0,
                reject_reason="outlier_shift_cluster",
            )
    return updated
