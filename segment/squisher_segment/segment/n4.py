"""N4 bias-field correction for channel-separated Lightsheet OME-Zarr stores."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import SimpleITK as sitk

from squisher.jpegxr_zarr import register_jpegxr_codec


logger = logging.getLogger(__name__)

DEFAULT_ITERATIONS = (50, 50, 30, 20)
DEFAULT_SPLINE_LOWRES_PX_ZYX = (24.0, 48.0, 48.0)
QUANT_MAX_SAMPLES = 50_000
QUANT_LOWER_PERCENTILE = 0.01
QUANT_UPPER_PERCENTILE = 99.99
QUANT_FALLBACK_UPPER_PERCENTILE = 99.9
QUANT_MIN_POPULATION = 200_000
QUANT_MIN_RANGE = 1e-6
UNSHARP_HALO = 16
MAX_FIELD_VOXELS = 64_000_000
N4_FIELD_FILENAME = "n4-field.npy"
EXPECTED_MULTI_PYTHON = Path("/home/chaichontat/miniforge3/envs/multi/bin/python")


@dataclass(frozen=True, slots=True)
class N4Config:
    field_level: int | None = None
    shrink: int = 4
    spline_lowres_px_zyx: tuple[float, float, float] = DEFAULT_SPLINE_LOWRES_PX_ZYX
    iterations: tuple[int, ...] = DEFAULT_ITERATIONS
    threshold: float | None = None
    unsharp: bool = False

    def __post_init__(self) -> None:
        if self.field_level is not None and self.field_level < 0:
            raise ValueError("field_level must be non-negative")
        if self.shrink < 1:
            raise ValueError("shrink must be at least 1")
        if len(self.spline_lowres_px_zyx) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in self.spline_lowres_px_zyx
        ):
            raise ValueError("spline_lowres_px_zyx must contain three finite positive values")
        if not self.iterations or any(value < 1 for value in self.iterations):
            raise ValueError("iterations must contain positive integers")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    scale_zyx: tuple[float, float, float]
    translation_zyx: tuple[float, float, float]


def _spline_spacing_zyx(
    config: N4Config,
    level0_geometry: ImageGeometry,
) -> tuple[float, float, float]:
    """Convert the level-0 voxel-grid default into physical ZYX spacing."""
    return tuple(
        lowres_px * config.shrink * spacing
        for lowres_px, spacing in zip(
            config.spline_lowres_px_zyx,
            level0_geometry.scale_zyx,
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class QuantizationParams:
    lower: float
    upper: float
    scale: float
    lower_percentile: float
    upper_percentile: float
    population_count: int


def _default_n4_output_path(source: Path) -> Path:
    """Choose a compact channel-specific sibling name for a Lightsheet output."""
    match = re.search(r"\.ch(\d+)\.ome\.zarr$", source.name)
    suffix = f".ch{match.group(1)}" if match else ""
    return source.parent / f"fused-n4{suffix}.ome.zarr"


def _attrs_dict(node: Any) -> dict[str, Any]:
    attrs = node.attrs
    return copy.deepcopy(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _multiscale_record(root: Any, source: Path) -> dict[str, Any]:
    attrs = root.attrs
    ome = attrs.get("ome")
    records = ome.get("multiscales") if isinstance(ome, dict) else attrs.get("multiscales")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"{source} must contain exactly one OME-Zarr multiscale image")
    return records[0]


def _dataset_paths(root: Any, source: Path) -> list[str]:
    datasets = _multiscale_record(root, source).get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{source} OME-Zarr multiscale metadata has no datasets")
    paths: list[str] = []
    for index, dataset in enumerate(datasets):
        path = dataset.get("path") if isinstance(dataset, dict) else None
        if not isinstance(path, str) or not path:
            raise ValueError(f"{source} OME-Zarr dataset {index} has no path")
        if path in paths:
            raise ValueError(f"{source} OME-Zarr dataset path {path!r} is duplicated")
        paths.append(path)
    return paths


def _dataset_geometry(root: Any, source: Path, path: str) -> ImageGeometry:
    multiscale = _multiscale_record(root, source)
    datasets = multiscale["datasets"]
    dataset = next(item for item in datasets if item.get("path") == path)

    scale = np.ones(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    saw_scale = False
    for owner, transforms in (
        (f"{source}/{path}", dataset.get("coordinateTransformations")),
        (f"{source} multiscale", multiscale.get("coordinateTransformations", [])),
    ):
        if not isinstance(transforms, list):
            raise ValueError(f"{owner} has no NGFF coordinate transformations")
        for transform in transforms:
            if not isinstance(transform, dict):
                raise ValueError(f"{owner} has an invalid NGFF coordinate transformation")
            transform_type = transform.get("type")
            values = transform.get(transform_type, "")
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f"{owner} has an invalid {transform_type!r} transformation")
            vector = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"{owner} has a non-finite coordinate transformation")
            if transform_type == "scale":
                if np.any(vector <= 0):
                    raise ValueError(f"{owner} has a non-positive coordinate scale")
                translation *= vector
                scale *= vector
                saw_scale = True
            elif transform_type == "translation":
                translation += vector
            else:
                raise ValueError(f"{owner} uses unsupported {transform_type!r} coordinate transformation")
    if not saw_scale:
        raise ValueError(f"{source}/{path} has no NGFF scale transformation")
    return ImageGeometry(tuple(scale.tolist()), tuple(translation.tolist()))


def _dimension_names(array: Any) -> tuple[str, ...]:
    dimensions = getattr(array.metadata, "dimension_names", None)
    if dimensions is None:
        dimensions = array.attrs.get("_ARRAY_DIMENSIONS")
    if dimensions is None:
        return ()
    return tuple(str(value).lower() for value in dimensions)


def _validate_source(root: Any, source: Path, config: N4Config) -> list[str]:
    if getattr(root.metadata, "zarr_format", None) != 3:
        raise ValueError(f"{source} must be a Zarr v3 OME-Zarr group")
    attrs = _attrs_dict(root)
    if attrs.get("squisher_complete") is not True:
        raise ValueError(f"{source} is not marked complete by the Lightsheet writer")
    if "squisher_n4" in attrs:
        raise ValueError(f"{source} is already marked as N4-corrected")

    axes = _multiscale_record(root, source).get("axes")
    axis_names = tuple(
        str(axis.get("name") if isinstance(axis, dict) else axis).lower() for axis in axes or []
    )
    if axis_names != ("z", "y", "x"):
        raise ValueError(f"{source} must be a channel-separated ZYX OME-Zarr image; found axes {axis_names}")
    axis_units = tuple(axis.get("unit") if isinstance(axis, dict) else None for axis in axes or [])
    if axis_units != ("micrometer", "micrometer", "micrometer"):
        raise ValueError(f"{source} spatial axes must use micrometer units; found {axis_units}")

    paths = _dataset_paths(root, source)
    previous_shape: tuple[int, int, int] | None = None
    previous_geometry: ImageGeometry | None = None
    for path in paths:
        try:
            array = root[path]
        except KeyError as exc:
            raise ValueError(f"{source} is missing OME-Zarr dataset {path!r}") from exc
        shape = tuple(int(value) for value in array.shape)
        dimension_names = _dimension_names(array)
        if len(shape) != 3 or dimension_names != ("z", "y", "x"):
            raise ValueError(
                f"{source}/{path} must have ZYX dimensions; found shape={shape}, "
                f"dimensions={dimension_names}"
            )
        if np.dtype(array.dtype) != np.dtype(np.uint16):
            raise ValueError(f"{source}/{path} must have uint16 data; found {array.dtype}")
        geometry = _dataset_geometry(root, source, path)
        if previous_shape is not None and previous_geometry is not None:
            factors = _pyramid_factors(previous_shape, shape, context=f"{source}/{path}")
            expected_scale = np.asarray(previous_geometry.scale_zyx) * np.asarray(factors)
            if not np.allclose(geometry.scale_zyx, expected_scale, rtol=1e-6, atol=1e-9):
                raise ValueError(
                    f"{source}/{path} scale does not match pyramid factors {factors}: "
                    f"expected {tuple(expected_scale)}, found {geometry.scale_zyx}"
                )
            if not np.allclose(
                geometry.translation_zyx,
                previous_geometry.translation_zyx,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(f"{source}/{path} translation does not match the preceding pyramid level")
        previous_shape = shape
        previous_geometry = geometry

    if config.field_level is not None and config.field_level >= len(paths):
        raise ValueError(
            f"field level {config.field_level} is outside the {len(paths)}-level OME-Zarr pyramid"
        )
    return paths


def _parse_threshold(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    token = value.strip()
    try:
        numeric = float(token)
    except ValueError as exc:
        raise ValueError("threshold must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError("threshold must be finite")
    return numeric


def _threshold_mask(image: Any, threshold: float | None) -> Any:
    import cupy as cp

    data = cp.asarray(image, dtype=cp.float32)
    finite_data = cp.where(cp.isfinite(data), data, 0.0)
    if threshold is None:
        mask = finite_data > 0.0
    else:
        mask = finite_data > threshold
    if not bool(cp.any(mask)):
        raise ValueError("N4 foreground mask is empty for the selected field volume")
    return mask


def _threshold_summary(threshold: float | None) -> dict[str, Any]:
    if threshold is None:
        return {"kind": "default_gt_zero", "value": 0.0}
    return {"kind": "numeric", "value": float(threshold)}


def _estimate_n4_field(
    image_zyx: Any,
    *,
    spacing_zyx: tuple[float, float, float],
    spline_spacing_zyx: tuple[float, float, float],
    shrink: int,
    iterations: Iterable[int] = DEFAULT_ITERATIONS,
    threshold: float | None,
) -> Any:
    """Estimate a positive 3D multiplicative field normalized on foreground."""
    import cupy as cp

    image_gpu = cp.asarray(image_zyx, dtype=cp.float32)
    if image_gpu.ndim != 3:
        raise ValueError(f"N4 expects a ZYX volume; found shape {image_gpu.shape}")
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"N4 spacing must contain three finite positive values; found {spacing_zyx}")
    spline_spacing = np.asarray(spline_spacing_zyx, dtype=np.float64)
    if spline_spacing.shape != (3,) or not np.all(np.isfinite(spline_spacing)) or np.any(spline_spacing <= 0):
        raise ValueError(
            f"N4 spline spacing must contain three finite positive values; found {spline_spacing_zyx}"
        )
    if shrink < 1:
        raise ValueError("shrink must be at least 1")
    shrunk_shape = tuple(size // shrink for size in image_gpu.shape)
    if any(size < 2 for size in shrunk_shape):
        raise ValueError(
            f"N4 shrink={shrink} leaves shape {shrunk_shape}; at least two voxels per axis are required"
        )

    full = sitk.GetImageFromArray(cp.asnumpy(image_gpu))
    full.SetSpacing(tuple(reversed(spacing.tolist())))
    small = sitk.Shrink(full, [int(shrink)] * full.GetDimension())
    small_array = sitk.GetArrayFromImage(small)
    small_mask_gpu = _threshold_mask(small_array, threshold)
    small_mask = sitk.GetImageFromArray(cp.asnumpy(small_mask_gpu).astype(np.uint8, copy=False))
    small_mask.CopyInformation(small)

    correction = sitk.N4BiasFieldCorrectionImageFilter()
    correction.SetMaximumNumberOfIterations([int(value) for value in iterations])
    spline_order = int(correction.GetSplineOrder())
    control_points = []
    for dimension, desired_spacing in enumerate(reversed(spline_spacing.tolist())):
        physical_length = float(small.GetSpacing()[dimension]) * int(small.GetSize()[dimension])
        mesh_size = max(1, int(round(physical_length / desired_spacing)))
        control_points.append(mesh_size + spline_order)
    correction.SetNumberOfControlPoints(control_points)
    correction.Execute(small, small_mask)

    log_bias_small = sitk.Cast(correction.GetLogBiasFieldAsImage(small), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(full)
    resampler.SetInterpolator(sitk.sitkBSpline)
    log_bias_full = resampler.Execute(log_bias_small)
    field_gpu = cp.asarray(sitk.GetArrayFromImage(sitk.Exp(log_bias_full)), dtype=cp.float32)

    foreground_gpu = _threshold_mask(image_gpu, threshold)
    scale = float(cp.median(field_gpu[foreground_gpu]).item())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"N4 produced an invalid foreground normalization scale {scale!r}")
    field_gpu /= scale
    field_gpu[~foreground_gpu] = 1.0
    if not bool(cp.all(cp.isfinite(field_gpu))) or bool(cp.any(field_gpu <= 0)):
        raise ValueError("N4 produced a non-finite or non-positive correction field")
    return field_gpu


def _ensure_gpu() -> None:
    import cupy as cp

    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("N4 GPU processing requested but no CUDA device is available")


def _validate_runtime() -> None:
    interpreter = Path(sys.executable).resolve()
    expected = EXPECTED_MULTI_PYTHON.resolve()
    if interpreter != expected:
        raise RuntimeError(
            f"N4 OME-Zarr writes require the multi environment interpreter {expected}; found {interpreter}"
        )
    logger.info("N4 runtime interpreter: %s", interpreter)


def _correct_gpu(block_zyx: np.ndarray, field_gpu: Any, *, unsharp: bool) -> tuple[Any, Any]:
    import cupy as cp
    from cucim.skimage import filters as cucim_filters

    block_gpu = cp.asarray(block_zyx, dtype=cp.float32)
    if block_gpu.shape != field_gpu.shape:
        raise ValueError(f"Correction block {block_gpu.shape} and field {field_gpu.shape} must match")
    foreground_gpu = block_gpu > 0
    block_gpu /= field_gpu
    if unsharp:
        for index in range(block_gpu.shape[0]):
            sharpened = cucim_filters.unsharp_mask(block_gpu[index], radius=3, preserve_range=True)
            cp.copyto(block_gpu[index], sharpened, where=foreground_gpu[index])
    return block_gpu, foreground_gpu


def _resample_field_gpu(
    field_gpu: Any,
    field_geometry: ImageGeometry,
    target_geometry: ImageGeometry,
    selection: tuple[slice, slice, slice],
) -> Any:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates

    shape = tuple(part.stop - part.start for part in selection)
    result = cp.empty(shape, dtype=cp.float32)
    coordinate_vectors = []
    for axis, part in enumerate(selection):
        target_indices = cp.arange(part.start, part.stop, dtype=cp.float32)
        world = target_indices * target_geometry.scale_zyx[axis] + target_geometry.translation_zyx[axis]
        coordinate_vectors.append(
            (world - field_geometry.translation_zyx[axis]) / field_geometry.scale_zyx[axis]
        )

    y_coordinates, x_coordinates = cp.meshgrid(coordinate_vectors[1], coordinate_vectors[2], indexing="ij")
    coordinates = cp.empty((3, shape[1], shape[2]), dtype=cp.float32)
    coordinates[1] = y_coordinates
    coordinates[2] = x_coordinates
    for output_z, field_z in enumerate(coordinate_vectors[0]):
        coordinates[0].fill(field_z)
        result[output_z] = map_coordinates(field_gpu, coordinates, order=1, mode="nearest")
    return result


def _quantization_params_from_sample(sample: Any, population: int) -> QuantizationParams:
    import cupy as cp

    values = cp.asarray(sample, dtype=cp.float32)
    if values.ndim != 1 or values.size == 0 or population < values.size:
        raise ValueError("Quantization requires a non-empty 1D sample and valid population count")
    upper_percentile = (
        QUANT_UPPER_PERCENTILE if population >= QUANT_MIN_POPULATION else QUANT_FALLBACK_UPPER_PERCENTILE
    )
    lower = float(cp.percentile(values, QUANT_LOWER_PERCENTILE).item())
    upper = float(cp.percentile(values, upper_percentile).item())
    if not math.isfinite(lower):
        lower = 0.0
    if not math.isfinite(upper):
        upper = lower + QUANT_MIN_RANGE
    if upper - lower < QUANT_MIN_RANGE:
        center = (upper + lower) * 0.5
        lower = center - QUANT_MIN_RANGE * 0.5
        upper = center + QUANT_MIN_RANGE * 0.5
    return QuantizationParams(
        lower=lower,
        upper=upper,
        scale=float(np.iinfo(np.uint16).max - 1) / (upper - lower),
        lower_percentile=QUANT_LOWER_PERCENTILE,
        upper_percentile=upper_percentile,
        population_count=population,
    )


def _quantize_gpu(corrected_gpu: Any, foreground_gpu: Any, params: QuantizationParams) -> Any:
    import cupy as cp

    scaled = (corrected_gpu - params.lower) * params.scale
    max_u16 = int(np.iinfo(np.uint16).max)
    cp.clip(scaled, 0.0, float(max_u16 - 1), out=scaled)
    scaled = cp.where(corrected_gpu > params.upper, float(max_u16), scaled)
    result = cp.rint(scaled).astype(cp.uint16)
    result[~foreground_gpu] = 0
    return result


def _chunk_slices(shape: tuple[int, ...], chunks: tuple[int, ...]):
    starts = [range(0, size, chunk) for size, chunk in zip(shape, chunks, strict=True)]
    for indices in itertools.product(*starts):
        yield tuple(
            slice(start, min(start + chunk, size))
            for start, chunk, size in zip(indices, chunks, shape, strict=True)
        )


def _storage_chunks(array: Any) -> tuple[int, ...]:
    shards = getattr(array.metadata, "shards", None)
    return tuple(int(value) for value in (shards or array.chunks))


def _foreground_population_gpu(source: Any) -> int:
    import cupy as cp

    shape = tuple(int(value) for value in source.shape)
    population = 0
    for selection in _chunk_slices(shape, _storage_chunks(source)):
        population += int(cp.count_nonzero(cp.asarray(source[selection]) > 0).item())
    return population


def _create_array_like(destination_root: Any, path: str, source_array: Any) -> Any:
    destination = destination_root.create_array(
        path,
        shape=source_array.shape,
        chunks=source_array.chunks,
        shards=getattr(source_array.metadata, "shards", None),
        dtype=source_array.dtype,
        filters=source_array.filters,
        serializer=source_array.serializer,
        compressors=source_array.compressors,
        fill_value=source_array.fill_value,
        dimension_names=source_array.metadata.dimension_names,
        chunk_key_encoding=source_array.metadata.chunk_key_encoding,
        config={"write_empty_chunks": False},
    )
    destination.attrs.update(_attrs_dict(source_array))
    return destination


def _quantization_windows(
    shape: tuple[int, int, int], chunks: tuple[int, int, int]
) -> list[tuple[slice, slice, slice]]:
    """Choose deterministic level-0 windows totaling roughly ``QUANT_MAX_SAMPLES`` voxels."""
    bin_counts = (min(4, shape[0]), min(2, shape[1]), min(2, shape[2]))
    window_count = math.prod(bin_counts)
    side = max(1, int(math.sqrt(QUANT_MAX_SAMPLES / window_count)))
    window_shape = (1, min(side, shape[1], chunks[1]), min(side, shape[2], chunks[2]))
    selections: list[tuple[slice, slice, slice]] = []
    for indices in itertools.product(*(range(count) for count in bin_counts)):
        bounds = []
        for size, chunk, count, index, width in zip(
            shape, chunks, bin_counts, indices, window_shape, strict=True
        ):
            bin_start = (index * size) // count
            bin_stop = ((index + 1) * size) // count
            center = ((2 * index + 1) * size) // (2 * count)
            chunk_start = (center // chunk) * chunk
            chunk_stop = min(chunk_start + chunk, size)
            valid_start = max(bin_start, chunk_start)
            valid_stop = min(bin_stop, chunk_stop)
            local_width = min(width, valid_stop - valid_start)
            start = min(max(valid_start, center - local_width // 2), valid_stop - local_width)
            bounds.append((start, start + local_width))
        selections.append(tuple(slice(start, stop) for start, stop in bounds))
    return selections


def _correct_selection_gpu(
    source: Any,
    field_gpu: Any,
    field_geometry: ImageGeometry,
    target_geometry: ImageGeometry,
    selection: tuple[slice, slice, slice],
    *,
    unsharp: bool,
) -> tuple[Any, Any]:
    shape = tuple(int(value) for value in source.shape)
    z_slice, y_slice, x_slice = selection
    halo = UNSHARP_HALO if unsharp else 0
    read_y = slice(max(0, y_slice.start - halo), min(shape[1], y_slice.stop + halo))
    read_x = slice(max(0, x_slice.start - halo), min(shape[2], x_slice.stop + halo))
    read_selection = (z_slice, read_y, read_x)
    source_block = np.asarray(source[read_selection])
    local_field_gpu = _resample_field_gpu(
        field_gpu,
        field_geometry,
        target_geometry,
        read_selection,
    )
    corrected_gpu, foreground_gpu = _correct_gpu(source_block, local_field_gpu, unsharp=unsharp)
    core_y = slice(y_slice.start - read_y.start, y_slice.stop - read_y.start)
    core_x = slice(x_slice.start - read_x.start, x_slice.stop - read_x.start)
    return corrected_gpu[:, core_y, core_x], foreground_gpu[:, core_y, core_x]


def _level0_quantization_params(
    source: Any,
    field_gpu: Any,
    field_geometry: ImageGeometry,
    target_geometry: ImageGeometry,
    config: N4Config,
) -> QuantizationParams:
    import cupy as cp

    shape = tuple(int(value) for value in source.shape)
    chunks = tuple(int(value) for value in source.chunks)
    population = _foreground_population_gpu(source)
    if population == 0:
        raise ValueError("Cannot quantize level 0 because it has no foreground")
    sample_parts = []
    for selection in _quantization_windows(shape, chunks):
        corrected_gpu, foreground_gpu = _correct_selection_gpu(
            source,
            field_gpu,
            field_geometry,
            target_geometry,
            selection,
            unsharp=config.unsharp,
        )
        values = corrected_gpu[foreground_gpu]
        if values.size:
            sample_parts.append(values)
    if not sample_parts:
        raise ValueError("Cannot quantize level 0 because sampled windows contain no foreground")
    sample = cp.concatenate(sample_parts)
    return _quantization_params_from_sample(sample, population)


def _write_corrected_level0(
    source: Any,
    destination: Any,
    field_gpu: Any,
    field_geometry: ImageGeometry,
    target_geometry: ImageGeometry,
    params: QuantizationParams,
    config: N4Config,
) -> None:
    import cupy as cp

    shape = tuple(int(value) for value in source.shape)
    storage_chunks = _storage_chunks(destination)
    total = math.prod(math.ceil(size / chunk) for size, chunk in zip(shape, storage_chunks, strict=True))
    for index, selection in enumerate(_chunk_slices(shape, storage_chunks), start=1):
        corrected_gpu, foreground_gpu = _correct_selection_gpu(
            source,
            field_gpu,
            field_geometry,
            target_geometry,
            selection,
            unsharp=config.unsharp,
        )
        destination[selection] = cp.asnumpy(_quantize_gpu(corrected_gpu, foreground_gpu, params))
        if index <= 3 or index % 100 == 0 or index == total:
            logger.info("N4 level 0: wrote storage block %d/%d", index, total)


def _pyramid_factors(
    source_shape: tuple[int, ...], destination_shape: tuple[int, ...], *, context: str
) -> tuple[int, ...]:
    factors = []
    for source_size, destination_size in zip(source_shape, destination_shape, strict=True):
        if destination_size < 1 or destination_size > source_size:
            raise ValueError(
                f"{context} has invalid pyramid transition {source_shape} -> {destination_shape}"
            )
        factor = max(1, source_size // destination_size)
        if source_size // factor != destination_size:
            raise ValueError(
                f"{context} pyramid shape {destination_shape} is not an integer downsample of {source_shape}"
            )
        factors.append(factor)
    return tuple(factors)


def _downsample_mean(
    block: np.ndarray,
    *,
    factors: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Mean-reduce an exactly divisible block and preserve the requested dtype."""
    if block.ndim != len(factors) or any(
        size % factor for size, factor in zip(block.shape, factors, strict=True)
    ):
        raise ValueError(f"Block shape {block.shape} is not divisible by factors {factors}")
    reshape: list[int] = []
    mean_axes = []
    for axis, (size, factor) in enumerate(zip(block.shape, factors, strict=True)):
        reshape.extend((size // factor, factor))
        mean_axes.append(2 * axis + 1)

    import cupy as cp

    reduced = cp.asarray(block).reshape(tuple(reshape)).mean(axis=tuple(mean_axes), dtype=cp.float32)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        reduced = cp.clip(cp.rint(reduced), info.min, info.max)
    return cp.asnumpy(reduced.astype(dtype, copy=False))


def _write_downsampled_level(
    source: Any,
    destination: Any,
    *,
    factors: tuple[int, ...],
    level_path: str,
) -> None:
    shape = tuple(int(value) for value in destination.shape)
    storage_chunks = _storage_chunks(destination)
    total = math.prod(math.ceil(size / chunk) for size, chunk in zip(shape, storage_chunks, strict=True))
    for index, selection in enumerate(_chunk_slices(shape, storage_chunks), start=1):
        source_selection = tuple(
            slice(part.start * factor, part.stop * factor)
            for part, factor in zip(selection, factors, strict=True)
        )
        destination[selection] = _downsample_mean(
            np.asarray(source[source_selection]),
            factors=factors,
            dtype=np.dtype(destination.dtype),
        )
        if index <= 3 or index % 100 == 0 or index == total:
            logger.info("N4 pyramid %s: wrote storage block %d/%d", level_path, index, total)


def _metadata_digest(root: Any, paths: list[str]) -> str:
    payload = {
        "root_attrs": _attrs_dict(root),
        "arrays": {
            path: {
                "metadata": root[path].metadata.to_dict(),
                "attrs": _attrs_dict(root[path]),
            }
            for path in paths
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns


def _validate_overwrite_target(path: Path) -> None:
    """Require overwrite targets to be completed outputs owned by this workflow."""
    import zarr

    if path.is_symlink() or not path.is_dir() or not (path / "zarr.json").is_file():
        raise ValueError(f"Refusing to overwrite non-N4 output: {path}")
    root = zarr.open_group(path, mode="r")
    attrs = _attrs_dict(root)
    provenance = attrs.get("squisher_n4")
    field_path = provenance.get("field_path") if isinstance(provenance, dict) else None
    if attrs.get("squisher_complete") is not True or field_path != N4_FIELD_FILENAME:
        raise ValueError(f"Refusing to overwrite incomplete or foreign output: {path}")
    field_file = path / N4_FIELD_FILENAME
    if field_file.is_symlink() or not field_file.is_file():
        raise ValueError(f"Refusing to overwrite N4 output with missing field artifact: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = []
    for current, child_directories, files in os.walk(root):
        current_path = Path(current)
        directories.append(current_path)
        for name in files:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"Refusing to publish symlink inside staged output: {path}")
            _fsync_file(path)
        for name in child_directories:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"Refusing to publish symlink inside staged output: {path}")
    for directory in reversed(directories):
        _fsync_directory(directory)


def _renameat2(source: Path, destination: Path, *, flags: int) -> None:
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic N4 publication requires Linux renameat2 support")
    result = renameat2(
        ctypes.c_int(-100),
        ctypes.c_char_p(os.fsencode(source)),
        ctypes.c_int(-100),
        ctypes.c_char_p(os.fsencode(destination)),
        ctypes.c_uint(flags),
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def _publish_staged(
    staged: Path,
    output: Path,
    *,
    overwrite: bool,
    initial_identity: tuple[int, int, int] | None,
) -> None:
    """Publish without clobbering a destination that changed during processing."""
    current_identity = _path_identity(output)
    if current_identity != initial_identity:
        shutil.rmtree(staged)
        if initial_identity is None:
            raise FileExistsError(f"Output {output} was created while N4 was running")
        raise RuntimeError(f"Output {output} changed while N4 was running")

    if current_identity is None:
        try:
            _renameat2(staged, output, flags=1)  # RENAME_NOREPLACE
        except FileExistsError as exc:
            shutil.rmtree(staged)
            raise FileExistsError(f"Output {output} was created while N4 was running") from exc
        try:
            _fsync_directory(output.parent)
        except OSError as exc:
            raise RuntimeError(f"Published {output}, but failed to sync its parent directory") from exc
        return

    if not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")

    _validate_overwrite_target(output)
    _renameat2(staged, output, flags=2)  # RENAME_EXCHANGE
    if _path_identity(staged) != initial_identity:
        _renameat2(staged, output, flags=2)
        _fsync_directory(output.parent)
        raise RuntimeError(f"Output {output} was replaced while N4 was publishing")
    try:
        _fsync_directory(output.parent)
    except OSError as exc:
        raise RuntimeError(
            f"Published {output}, but failed to sync its directory; prior output remains at {staged}"
        ) from exc
    shutil.rmtree(staged)
    try:
        _fsync_directory(output.parent)
    except OSError as exc:
        raise RuntimeError(f"Published {output}, but failed to sync removal of the prior output") from exc


def _relative_store_path(value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must stay inside the OME-Zarr store; found {value!r}")
    return path


def _copy_source_sidecars(
    source: Path,
    destination: Path,
    *,
    dataset_paths: list[str],
    root_attrs: dict[str, Any],
) -> dict[str, str]:
    """Copy the validated fusion provenance tree and reject unsupported store nodes."""
    dataset_roots = {Path(path).parts[0] for path in dataset_paths}
    allowed_roots = {"zarr.json", *dataset_roots}
    fusion = root_attrs.get("squisher_fusion")
    copied_hashes: dict[str, str] = {}
    if fusion is not None:
        if not isinstance(fusion, dict):
            raise ValueError("squisher_fusion must be an object")
        manifest_relative = _relative_store_path(fusion.get("manifest"), context="squisher_fusion.manifest")
        sidecar_root = manifest_relative.parts[0]
        if sidecar_root in dataset_roots or sidecar_root == "zarr.json":
            raise ValueError("squisher_fusion.manifest overlaps an image dataset")
        source_sidecars = source / sidecar_root
        source_manifest = source / manifest_relative
        if not source_manifest.is_file():
            raise FileNotFoundError(f"Fusion provenance manifest is missing: {source_manifest}")
        for path in source_sidecars.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Fusion provenance bundle must not contain symlinks: {path}")
        manifest = json.loads(source_manifest.read_text())
        if not isinstance(manifest, dict):
            raise ValueError(f"Fusion provenance manifest must contain an object: {source_manifest}")
        artifacts = manifest.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError(f"Fusion provenance artifacts must be a list: {source_manifest}")
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                raise ValueError(f"Fusion provenance artifact {index} must be an object")
            artifact_path = _relative_store_path(
                artifact.get("bundled_path"),
                context=f"fusion provenance artifact {index}.bundled_path",
            )
            source_artifact = source / artifact_path
            if artifact_path.parts[0] != sidecar_root or not source_artifact.is_file():
                raise FileNotFoundError(f"Fusion provenance artifact is missing: {source_artifact}")
        destination_sidecars = destination / sidecar_root
        shutil.copytree(source_sidecars, destination_sidecars)
        allowed_roots.add(sidecar_root)
        for path in sorted(destination_sidecars.rglob("*")):
            if path.is_file():
                copied_hashes[path.relative_to(destination).as_posix()] = _file_sha256(path)

    unsupported = sorted(path.name for path in source.iterdir() if path.name not in allowed_roots)
    if unsupported:
        raise ValueError(
            f"{source} contains unsupported auxiliary store nodes: {unsupported}; "
            "only the fusion provenance bundle is supported"
        )
    return copied_hashes


def _provenance(
    *,
    source: Path,
    source_digest: str,
    level0_path: str,
    field_dataset: str,
    field_shape: tuple[int, int, int],
    field_geometry: ImageGeometry,
    spline_spacing_zyx: tuple[float, float, float],
    field_sha256: str,
    source_sidecars_sha256: dict[str, str],
    config: N4Config,
    params: QuantizationParams,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "algorithm": "SimpleITK.N4BiasFieldCorrectionImageFilter",
        "simpleitk_version": sitk.Version_VersionString(),
        "source": str(source.resolve()),
        "source_metadata_sha256": source_digest,
        "target_dataset": level0_path,
        "field_dataset": field_dataset,
        "field_shape": list(field_shape),
        "field_scale_zyx": list(field_geometry.scale_zyx),
        "field_translation_zyx": list(field_geometry.translation_zyx),
        "shrink": config.shrink,
        "spline_lowres_px_zyx": list(config.spline_lowres_px_zyx),
        "spline_spacing_zyx": list(spline_spacing_zyx),
        "iterations": list(config.iterations),
        "threshold": _threshold_summary(config.threshold),
        "unsharp": config.unsharp,
        "field_estimation_backend": "SimpleITK_CPU",
        "correction_backend": "CUDA_CuPy_cuCIM",
        "field_path": N4_FIELD_FILENAME,
        "field_sha256": field_sha256,
        "source_sidecars_sha256": source_sidecars_sha256,
        "field_interpolation": "trilinear_ngff_physical_coordinates",
        "quantization": {
            "dtype": "uint16",
            "lower": params.lower,
            "upper": params.upper,
            "scale": params.scale,
            "lower_percentile": params.lower_percentile,
            "upper_percentile": params.upper_percentile,
            "population_count": params.population_count,
            "population_method": "exact_level0_nonzero_gpu_prepass",
            "reference_dataset": level0_path,
            "sampling": "deterministic_level0_stratified_windows",
        },
        "pyramid_strategy": "mean_downsample_from_corrected_previous_level",
    }


def run_n4_ome_zarr(
    source: Path,
    output: Path,
    config: N4Config,
    *,
    overwrite: bool = False,
) -> Path:
    """Correct one channel-separated Lightsheet OME-Zarr into a staged destination."""
    import zarr

    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("N4 output must differ from the source OME-Zarr")
    initial_output_identity = _path_identity(output)
    if initial_output_identity is not None and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")
    if initial_output_identity is not None:
        _validate_overwrite_target(output)
    _validate_runtime()
    register_jpegxr_codec()
    source_root = zarr.open_group(source, mode="r")
    paths = _validate_source(source_root, source, config)
    _ensure_gpu()
    import cupy as cp

    source_digest = _metadata_digest(source_root, paths)
    level0 = source_root[paths[0]]
    field_level = config.field_level if config.field_level is not None else len(paths) - 1
    field_dataset = paths[field_level]
    field_source = source_root[field_dataset]
    field_shape = tuple(int(value) for value in field_source.shape)
    field_voxels = math.prod(field_shape)
    shrunk_shape = tuple(size // config.shrink for size in field_shape)
    if any(size < 2 for size in shrunk_shape):
        raise ValueError(
            f"N4 field level {field_dataset!r} shrink={config.shrink} leaves shape {shrunk_shape}; "
            "at least two voxels per axis are required"
        )
    if field_voxels > MAX_FIELD_VOXELS:
        raise ValueError(
            f"N4 field level {field_dataset!r} has {field_voxels:,} voxels, above the "
            f"{MAX_FIELD_VOXELS:,}-voxel safety limit; add a coarser pyramid level"
        )
    field_geometry = _dataset_geometry(source_root, source, field_dataset)
    level0_geometry = _dataset_geometry(source_root, source, paths[0])
    spline_spacing_zyx = _spline_spacing_zyx(config, level0_geometry)
    field_input_gpu = cp.asarray(field_source[:], dtype=cp.float32)
    if not bool(cp.any(field_input_gpu > 0)):
        raise ValueError(f"N4 field dataset {field_dataset!r} has no positive voxels")

    if config.unsharp:
        corrected_gpu, _ = _correct_gpu(
            field_input_gpu,
            cp.ones(field_shape, dtype=cp.float32),
            unsharp=True,
        )
        n4_input_gpu = corrected_gpu
    else:
        n4_input_gpu = field_input_gpu
    logger.info(
        "Estimating 3D N4 field from pyramid dataset %s with shape %s",
        field_dataset,
        field_shape,
    )
    field_gpu = _estimate_n4_field(
        n4_input_gpu,
        spacing_zyx=field_geometry.scale_zyx,
        spline_spacing_zyx=spline_spacing_zyx,
        shrink=config.shrink,
        iterations=config.iterations,
        threshold=config.threshold,
    )
    logger.info("Sampling corrected level 0 for quantization")
    params = _level0_quantization_params(
        level0,
        field_gpu,
        field_geometry,
        level0_geometry,
        config,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="n4-", suffix=".tmp", dir=output.parent))
    try:
        destination_root = zarr.open_group(temporary, mode="w", zarr_format=3)
        root_attrs = _attrs_dict(source_root)
        root_attrs["squisher_complete"] = False
        destination_root.attrs.update(root_attrs)
        source_sidecars_sha256 = _copy_source_sidecars(
            source,
            temporary,
            dataset_paths=paths,
            root_attrs=root_attrs,
        )

        destination_level0 = _create_array_like(destination_root, paths[0], level0)
        _write_corrected_level0(
            level0,
            destination_level0,
            field_gpu,
            field_geometry,
            level0_geometry,
            params,
            config,
        )

        previous = destination_level0
        for path in paths[1:]:
            source_level = source_root[path]
            destination_level = _create_array_like(destination_root, path, source_level)
            factors = _pyramid_factors(
                tuple(int(value) for value in previous.shape),
                tuple(int(value) for value in destination_level.shape),
                context=f"{source}/{path}",
            )
            _write_downsampled_level(
                previous,
                destination_level,
                factors=factors,
                level_path=path,
            )
            previous = destination_level

        field_file = temporary / N4_FIELD_FILENAME
        np.save(field_file, cp.asnumpy(field_gpu), allow_pickle=False)
        field_sha256 = _file_sha256(field_file)
        destination_root.attrs["squisher_n4"] = _provenance(
            source=source,
            source_digest=source_digest,
            level0_path=paths[0],
            field_dataset=field_dataset,
            field_shape=field_shape,
            field_geometry=field_geometry,
            spline_spacing_zyx=spline_spacing_zyx,
            field_sha256=field_sha256,
            source_sidecars_sha256=source_sidecars_sha256,
            config=config,
            params=params,
        )
        logger.info("Syncing staged N4 output before marking it complete")
        _fsync_tree(temporary)
        destination_root.attrs["squisher_complete"] = True
        _fsync_file(temporary / "zarr.json")
        _fsync_directory(temporary)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    del field_gpu
    _publish_staged(
        temporary,
        output,
        overwrite=overwrite,
        initial_identity=initial_output_identity,
    )

    return output


__all__ = [
    "N4Config",
    "run_n4_ome_zarr",
]
