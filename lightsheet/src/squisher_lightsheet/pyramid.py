from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

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


def add_pyramids(*, ome_zarrs: list[Path], template: Path | None = None, dry_run: bool = False) -> str:
    args = []
    if template is not None:
        args.extend(["--template", str(template)])
    args.extend(str(path) for path in ome_zarrs)
    return run_legacy_script("add_ome_zarr_pyramid.py", args, dry_run=dry_run)
