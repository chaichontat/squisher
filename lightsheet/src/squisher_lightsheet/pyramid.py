from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from squisher.jpegxr_zarr import DEFAULT_JPEGXR_LEVEL

from squisher_lightsheet.legacy_runner import run_legacy_script


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def chunk_count(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    count = 1
    for size, chunk_size in zip(shape, chunks, strict=True):
        count *= ceil_div(int(size), int(chunk_size))
    return count


def chunk_slices(shape: tuple[int, ...], chunks: tuple[int, ...]):
    ranges = [range(0, int(size), int(chunk_size)) for size, chunk_size in zip(shape, chunks, strict=True)]
    for starts in itertools.product(*ranges):
        yield tuple(
            slice(start, min(start + chunk_size, size))
            for start, size, chunk_size in zip(starts, shape, chunks, strict=True)
        )


def downsampled_chunks(
    source_chunks: tuple[int, ...],
    shape: tuple[int, ...],
    factors: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        min(max(int(chunk) // int(factor), 1), int(size))
        for chunk, factor, size in zip(source_chunks, factors, shape, strict=True)
    )


def pyramid_shard_chunks(
    source_storage_chunks: tuple[int, ...],
    shape: tuple[int, ...],
    inner_chunks: tuple[int, ...],
) -> tuple[int, ...]:
    chunks = []
    for source_chunk, size, inner_chunk in zip(source_storage_chunks, shape, inner_chunks, strict=True):
        capped = min(max(int(source_chunk), int(inner_chunk)), int(size))
        if capped == int(size):
            capped = (capped // int(inner_chunk)) * int(inner_chunk)
        if capped < int(inner_chunk):
            capped = int(inner_chunk)
        chunks.append(capped)
    return tuple(chunks)


def pyramid_relative_factors(shape: tuple[int, ...], dims: tuple[str, ...]) -> dict[str, int]:
    factors = {}
    for dim, size in zip(dims, shape, strict=True):
        if dim in {"z", "y", "x"} and int(size) // 2 > 100:
            factors[dim] = 2
        else:
            factors[dim] = 1
    return factors


def level_coordinate_transformations(
    base_transforms: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    abs_factors: dict[str, int],
) -> list[dict[str, Any]]:
    transforms = copy.deepcopy(base_transforms)
    for transform in transforms:
        if transform.get("type") != "scale":
            continue
        transform["scale"] = [
            float(value) * abs_factors.get(axis["name"], 1)
            for value, axis in zip(transform["scale"], axes, strict=True)
        ]
    return transforms


def axis_slices(*, size: int, step: int):
    size = int(size)
    step = int(step)
    for start in range(0, size, step):
        yield slice(start, min(start + step, size))


def spatial_slices(*, shape_yx: tuple[int, int], chunks_yx: tuple[int, int]):
    for y_slice in axis_slices(size=shape_yx[0], step=chunks_yx[0]):
        for x_slice in axis_slices(size=shape_yx[1], step=chunks_yx[1]):
            yield y_slice, x_slice


def xy_downsampled_shape(*, axes: str, shape: tuple[int, ...], factor: int) -> tuple[int, ...]:
    if axes == "ZYX":
        return (shape[0], max(1, shape[1] // factor), max(1, shape[2] // factor))
    if axes == "CZYX":
        return (shape[0], shape[1], max(1, shape[2] // factor), max(1, shape[3] // factor))
    raise ValueError(f"Expected axes ZYX or CZYX, got {axes!r}")


def downsample_xy_mean(block: np.ndarray, *, factor: int, dtype: np.dtype) -> np.ndarray:
    usable_y = (block.shape[-2] // factor) * factor
    usable_x = (block.shape[-1] // factor) * factor
    if usable_y == 0 or usable_x == 0:
        raise ValueError(f"Cannot downsample block shape {block.shape} by factor {factor}")
    trimmed = block[..., :usable_y, :usable_x]
    reduced = trimmed.reshape(*trimmed.shape[:-2], usable_y // factor, factor, usable_x // factor, factor).mean(
        axis=(-3, -1)
    )
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        reduced = np.rint(reduced)
        reduced = np.clip(reduced, info.min, info.max)
    return reduced.astype(dtype, copy=False)


def copy_z_slabs(
    *,
    source_array: Any,
    output_array: Any,
    axes: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> None:
    if axes == "ZYX":
        for z_slice in axis_slices(size=shape[0], step=chunks[0]):
            slab = np.asarray(source_array[z_slice, :, :])
            for y_slice, x_slice in spatial_slices(shape_yx=shape[1:], chunks_yx=chunks[1:]):
                output_array[z_slice, y_slice, x_slice] = slab[:, y_slice, x_slice]
        return

    if axes == "CZYX":
        for channel in range(shape[0]):
            for z_slice in axis_slices(size=shape[1], step=chunks[1]):
                slab = np.asarray(source_array[channel, z_slice, :, :])
                for y_slice, x_slice in spatial_slices(shape_yx=shape[2:], chunks_yx=chunks[2:]):
                    output_array[channel, z_slice, y_slice, x_slice] = slab[:, y_slice, x_slice]
        return

    raise ValueError(f"Expected axes ZYX or CZYX, got {axes!r}")


def copy_xy_downsampled(
    *,
    source_array: Any,
    output_array: Any,
    axes: str,
    factor: int,
    chunks: tuple[int, ...],
) -> None:
    shape = tuple(int(value) for value in output_array.shape)
    if axes == "ZYX":
        for z_slice in axis_slices(size=shape[0], step=chunks[0]):
            for y_slice, x_slice in spatial_slices(shape_yx=shape[1:], chunks_yx=chunks[1:]):
                source_selection = (
                    z_slice,
                    slice(y_slice.start * factor, y_slice.stop * factor),
                    slice(x_slice.start * factor, x_slice.stop * factor),
                )
                output_array[z_slice, y_slice, x_slice] = downsample_xy_mean(
                    np.asarray(source_array[source_selection]),
                    factor=factor,
                    dtype=np.dtype(output_array.dtype),
                )
        return

    if axes == "CZYX":
        for channel in range(shape[0]):
            for z_slice in axis_slices(size=shape[1], step=chunks[1]):
                for y_slice, x_slice in spatial_slices(shape_yx=shape[2:], chunks_yx=chunks[2:]):
                    source_selection = (
                        channel,
                        z_slice,
                        slice(y_slice.start * factor, y_slice.stop * factor),
                        slice(x_slice.start * factor, x_slice.stop * factor),
                    )
                    output_array[channel, z_slice, y_slice, x_slice] = downsample_xy_mean(
                        np.asarray(source_array[source_selection]),
                        factor=factor,
                        dtype=np.dtype(output_array.dtype),
                    )
        return

    raise ValueError(f"Expected axes ZYX or CZYX, got {axes!r}")


def add_pyramids(
    *,
    ome_zarrs: list[Path],
    template: Path | None = None,
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
    dry_run: bool = False,
) -> str:
    args = ["--jpegxr-level", str(jpegxr_level)]
    if template is not None:
        args.extend(["--template", str(template)])
    args.extend(str(path) for path in ome_zarrs)
    return run_legacy_script("add_ome_zarr_pyramid.py", args, dry_run=dry_run)
