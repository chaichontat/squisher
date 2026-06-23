from __future__ import annotations

import functools
from pathlib import Path

import numpy as np


def tiff_series_level_count(path: Path) -> int:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        return len(tif.series[0].levels)


def spatial_shape_zyx_from_axes(shape: tuple[int, ...], axes: str) -> tuple[int, int, int]:
    if axes == "CZYX":
        return int(shape[1]), int(shape[2]), int(shape[3])
    if axes == "ZYX":
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError(f"Expected CZYX or ZYX axes, got {axes!r}")


def spatial_shape_array_zyx_from_axes(shape: tuple[int, ...], axes: str) -> np.ndarray:
    return np.asarray(spatial_shape_zyx_from_axes(shape, axes), dtype=np.int64)


@functools.lru_cache(maxsize=512)
def tiff_level_factors_zyx(path: Path) -> tuple[tuple[int, int, int], ...]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        base_shape = np.asarray(spatial_shape_zyx_from_axes(tuple(series.shape), str(series.axes)), dtype=np.float64)
        factors = []
        for level in series.levels:
            level_shape = np.asarray(spatial_shape_zyx_from_axes(tuple(level.shape), str(level.axes)), dtype=np.float64)
            factors.append(tuple(int(v) for v in np.maximum(np.rint(base_shape / level_shape), 1)))
    return tuple(factors)


def choose_tiff_source_level(path: Path, desired_factor_zyx: np.ndarray) -> tuple[int, np.ndarray]:
    factors = tuple(np.asarray(factor, dtype=np.int64) for factor in tiff_level_factors_zyx(path))
    desired = np.asarray(desired_factor_zyx, dtype=np.int64)
    valid = [(index, factor) for index, factor in enumerate(factors) if np.all(factor <= desired)]
    if not valid:
        return 0, factors[0]
    return max(valid, key=lambda item: int(np.prod(item[1])))
