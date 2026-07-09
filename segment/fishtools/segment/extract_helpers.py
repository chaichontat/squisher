from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import zarr
from loguru import logger
from numpy.random import Generator
from scipy.ndimage import zoom
from tifffile import imwrite

from fishtools.segment.extract_support import ProgressReporter, progress_reporter, wrap_progress

"""Shared helpers for the `segment.extract` module."""

ZARR_TILE_SIZE = 512
DEFAULT_CROP_SIZE = 1024
MAX_WIDTH_AFTER_UPSCALE = 1024  # Prevent excessively wide ortho slices

# Quality thresholds
ZERO_PIXEL_SKIP_THRESHOLD = 0.1  # Skip tiles with >10% zeros (edge regions)


TIFF_KWARGS = {
    "compression": 22610,
    "photometric": "minisblack",
    "planarconfig": "separate",
    "compressionargs": {"level": 0.8},
}


@runtime_checkable
class ArrayLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    def __getitem__(self, key: Any) -> Any: ...


@runtime_checkable
class MaskLike(ArrayLike, Protocol):
    pass


Volume: TypeAlias = np.ndarray | zarr.Array | ArrayLike


@dataclass(frozen=True)
class ExtractionConfig:
    mode: Literal["z", "ortho", "maxproj"]
    channels: str | None
    crop: int
    dz: int
    n: int
    anisotropy: int
    upscale: float
    seed: int | None
    threads: int


@dataclass
class TileJob:
    file: Path
    vol: Volume
    channel_names: list[str] | None
    mask_vol: MaskLike | None
    mask_path: Path | None
    tile_origins: list[tuple[int, int]]
    z_candidates: list[int]
    aux_channel_vols: list[Volume] = field(default_factory=list)
    aux_channel_names: list[str] = field(default_factory=list)


@dataclass
class SliceJob:
    file: Path
    vol: Volume
    channel_names: list[str] | None
    mask_vol: MaskLike | None
    other_vol: Volume | None
    aux_channel_vols: list[Volume]
    position: int
    axis: Literal["y", "x"]
    perpendicular_slice: slice
    selected_indices: list[int]
    out_names: list[str]
    z_slice: slice | None = None


def _squeeze_mask(data: Any) -> np.ndarray:
    """Convert arbitrary mask slices to a numpy array with redundant axes removed."""
    arr = np.asarray(data)
    return np.squeeze(arr)


def _resize_mask(array: np.ndarray, factors: tuple[float, ...]) -> np.ndarray:
    """Resize mask data using nearest-neighbour interpolation to preserve labels."""
    if all(math.isclose(f, 1.0, abs_tol=1e-9, rel_tol=1e-9) for f in factors):
        return array.astype(array.dtype, copy=True)
    resized = zoom(array, factors, order=0)
    return resized.astype(array.dtype, copy=False)


def _resolve_output_names(
    indices: list[int],
    available_names: list[str] | None,
    channels_arg: str | None,
) -> list[str]:
    """Compute output channel names after selection; prefer explicit --channels names."""
    arg_names = [p.strip() for p in channels_arg.split(",") if p.strip()] if channels_arg else None
    if arg_names:
        return arg_names
    if available_names:
        try:
            return [available_names[i] for i in indices]
        except IndexError:
            return [str(i) for i in indices]
    return [str(i) for i in indices]


def _write_tiff(
    path: Path,
    data: np.ndarray,
    axes: str,
    names: list[str] | None,
    channels_arg: str | None,
    *,
    upscale: float,
) -> None:
    tiff_kwargs = dict(TIFF_KWARGS)
    if "C" in axes:
        ch_axis = axes.index("C")
        if data.shape[ch_axis] <= 1:
            tiff_kwargs.pop("planarconfig", None)
    imwrite(
        path,
        data,
        metadata={
            "axes": axes,
            "channel_names": names,
            "channels_arg": channels_arg,
            "upscale": upscale,
        },
        **tiff_kwargs,  # type: ignore[arg-type]
    )


def _write_mask_tiff(path: Path, data: np.ndarray, axes: str) -> None:
    imwrite(
        path,
        data,
        metadata={"axes": axes},
        compression="zstd",
    )


def _mask_filename(base: str) -> str:
    return f"{base[:-4]}_masks.tif" if base.endswith(".tif") else f"{base}_masks.tif"



def _compute_perpendicular_slice(
    *,
    axis_len: int,
    crop: int,
    max_width: int,
    rng: Generator,
) -> slice:
    """Compute a random slice along the perpendicular axis for ortho sampling.

    If the cropped axis fits within max_width, return the full cropped range.
    Otherwise, randomly sample a center position and return a slice of max_width.
    """
    start = crop
    end = axis_len - crop
    width = end - start

    if width <= max_width:
        return slice(start, end)

    # Randomly sample center position within valid range
    min_center = start + max_width // 2
    max_center = end - max_width // 2
    center = rng.integers(min_center, max_center + 1)
    return slice(center - max_width // 2, center - max_width // 2 + max_width)


def _validate_max_from_path(source: Path, files: list[Path], *, label: str) -> None:
    """Validate that a --max-from path is usable for the provided inputs."""
    if not source.exists():
        raise FileNotFoundError(f"[{label}] --max-from path not found: {source}")

    if source.is_dir():
        # Treat fused Zarr directories specially; they do not mirror reg-*.tif names.
        if source.suffix == ".zarr" or source.name.endswith(".zarr"):
            return
        missing = [str((source / f.name).resolve()) for f in files if not (source / f.name).exists()]
        if missing:
            raise FileNotFoundError(
                f"[{label}] --max-from is missing registered file(s): {', '.join(sorted(missing))}"
            )


@lru_cache(maxsize=1024)
def _path_size_cached(path_str: str) -> int:
    """Return on-disk size in bytes for a path.

    Directories (e.g., fused .zarr) are summed recursively. Returns 0 if the
    path does not exist or cannot be read.
    """
    path = Path(path_str)
    try:
        if path.is_file():
            return path.stat().st_size
        if not path.is_dir():
            return 0

        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                file_path = Path(root) / name
                try:
                    total += file_path.stat().st_size
                except OSError as exc:  # pragma: no cover - best-effort logging
                    logger.debug(f"Skipping file during size calculation: {file_path} ({exc})")
        return total
    except OSError as exc:  # pragma: no cover - best-effort logging
        logger.debug(f"Failed to compute size for {path}: {exc}")
        return 0


def _format_size(num_bytes: int) -> str:
    """Format byte size into human-friendly string using binary units."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(num_bytes)
    for u in units:
        if n < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(n)} {u}"
            return f"{n:.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def _prefix_with_roi(name: str, roi: str) -> str:
    """Prefix output filename with the ROI identifier."""
    prefix = f"{roi}--"
    return name if name.startswith(prefix) else f"{prefix}{name}"


def _normalize_reporter(
    progress: ProgressReporter | Callable[[], int | None] | None,
) -> ProgressReporter | None:
    if progress is None:
        return None
    if isinstance(progress, ProgressReporter):
        return progress
    return wrap_progress(progress)


def _sample_positions(length: int, *, crop: int, count: int, rng: Generator) -> list[int]:
    start = crop
    end = length - crop
    if end <= start:
        raise ValueError("Image after cropping is empty.")

    population = np.arange(start, end)

    if count >= population.size:
        return population.tolist()

    choices = rng.choice(population, size=count, replace=False)
    return sorted(int(v) for v in choices)


def _select_high_diversity_positions(
    positions: list[int],
    mask: MaskLike,
    axis: Literal["z", "y", "x"],
    count: int,
) -> list[int]:
    """From candidate positions, return top ``count`` by label diversity.

    Computes the unique label count for each slice position and returns
    the positions with the highest diversity, re-sorted by coordinate.
    """
    if len(positions) <= count:
        return positions

    # Compute unique label count for each position with progress bar
    logger.info(f"Scoring {len(positions)} {axis} positions for diversity...")
    scores: list[int] = []
    with progress_reporter(len(positions)) as progress:
        for pos in positions:
            if axis == "z":
                scores.append(len(np.unique(mask[pos, :, :])))
            elif axis == "y":
                scores.append(len(np.unique(mask[:, pos, :])))
            else:
                scores.append(len(np.unique(mask[:, :, pos])))
            progress.advance()

    # Sort by score descending, take top count
    sorted_positions = [p for _, p in sorted(zip(scores, positions), reverse=True)]
    return sorted(sorted_positions[: count])


def _score_and_select_tiles(
    candidates: list[tuple[int, int]],
    mask: MaskLike,
    *,
    tile_size: int,
    count: int,
    score_fn: Callable[[np.ndarray], int],
) -> list[tuple[int, int]]:
    """Score tile candidates by diversity/coverage and return the top N.

    Args:
        candidates: List of (y, x) tile origins.
        mask: Mask volume used for scoring (Z,Y,X).
        tile_size: Spatial tile size used when slicing the mask.
        count: Number of tiles to return.
        score_fn: Callable that maps a mask tile to an integer score.
    """
    scored: list[tuple[int, int, int]] = []
    with progress_reporter(len(candidates)) as progress:
        for y0, x0 in candidates:
            tile_mask = mask[:, y0 : y0 + tile_size, x0 : x0 + tile_size]
            score = score_fn(tile_mask)
            scored.append((score, y0, x0))
            progress.advance()

    scored.sort(reverse=True)
    return [(y, x) for _, y, x in scored[:count]]


def _expand_positions_with_context(
    base_positions: list[int],
    *,
    crop: int,
    axis_len: int,
    step: int = 2,
    context_pairs: int = 10,
) -> list[int]:
    """Return ordered unique positions including surrounding context for each base index.

    For every position in ``base_positions`` this yields the base index followed by
    ``context_pairs`` offsets in both positive and negative directions, stepping by ``step``.
    Positions are clamped to the valid range implied by ``crop`` and ``axis_len`` and
    deduplicated while preserving the first-seen order.
    """

    min_idx = crop
    max_idx = axis_len - crop - 1 if crop > 0 else axis_len - 1
    if min_idx > max_idx:
        raise ValueError("Cropping removes all available positions.")

    seen: set[int] = set()
    ordered: list[int] = []

    for base in base_positions:
        offsets = [0]
        for k in range(1, context_pairs + 1):
            offsets.append(step * k)
            offsets.append(-step * k)
        for offset in offsets:
            candidate = base + offset
            if candidate < min_idx or candidate > max_idx:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)

    return ordered


FILE_NAME = "fused_n4.zarr"


def _distribute_file_budget(rois: list[str], counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate a global quota across ROIs respecting per-ROI availability."""
    if total <= 0:
        return {roi: 0 for roi in rois}

    availability = {roi: max(0, counts.get(roi, 0)) for roi in rois}
    total_available = sum(availability.values())
    if total_available <= total:
        return availability

    # Largest remainder method: allocate floor of proportional share, then distribute leftovers.
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for roi in rois:
        share = total * availability[roi] / total_available if total_available else 0.0
        base = min(availability[roi], int(math.floor(share)))
        allocations[roi] = base
        remainders.append((share - base, roi))

    remaining = total - sum(allocations.values())
    if remaining > 0:
        for _, roi in sorted(remainders, key=lambda item: item[0], reverse=True):
            if remaining == 0:
                break
            if allocations[roi] < availability[roi]:
                allocations[roi] += 1
                remaining -= 1

    return allocations


def _compute_tile_origins(
    shape: tuple[int, int, int, int],
    *,
    tile_size: int,
    n_tiles: int | None,
    crop: int,
) -> list[tuple[int, int]]:
    _z, y_len, x_len, _ = shape
    if y_len < tile_size + 2 * crop or x_len < tile_size + 2 * crop:
        raise ValueError(
            f"Zarr volume spatial dimensions are smaller than the requested {tile_size}×{tile_size} tile size."
        )

    start_y = crop
    start_x = crop
    max_y = y_len - crop - tile_size
    max_x = x_len - crop - tile_size
    if max_y < start_y or max_x < start_x:
        raise ValueError("Tile cropping leaves no valid region to sample.")

    y_range = max_y - start_y + 1
    x_range = max_x - start_x + 1
    total_positions = y_range * x_range
    desired = n_tiles or 50
    desired = min(desired, total_positions)
    idxs = np.linspace(0, total_positions - 1, desired, dtype=int)
    idxs = np.unique(idxs)
    if idxs.size < desired:
        extras = np.setdiff1d(np.arange(total_positions), idxs, assume_unique=False)[: desired - idxs.size]
        idxs = np.sort(np.concatenate([idxs, extras]))

    origins: list[tuple[int, int]] = []
    for idx in idxs:
        offset_y = idx // x_range
        offset_x = idx % x_range
        origins.append((start_y + int(offset_y), start_x + int(offset_x)))
    return origins


def _format_tile_filename(
    stem: str,
    roi: str,
    z_index: int,
    y0: int,
    x0: int,
    *,
    coord_width: int,
) -> str:
    name = f"{stem}_y{y0:0{coord_width}d}_x{x0:0{coord_width}d}_z{z_index:02d}.tif"
    return _prefix_with_roi(name, roi)


def _format_tile_maxproj_filename(
    stem: str,
    roi: str,
    y0: int,
    x0: int,
    *,
    coord_width: int,
) -> str:
    name = f"{stem}_maxproj-y{y0:0{coord_width}d}-x{x0:0{coord_width}d}.tif"
    return _prefix_with_roi(name, roi)


def load_roi_points(roi_path: Path, *, upscale: int = 2) -> list[tuple[int, int]]:
    """Load point coordinates from ImageJ ROI file (.roi or .zip).

    Parameters
    ----------
    roi_path
        Path to .roi file or .zip containing multiple ROIs.
    upscale
        Factor to multiply coordinates by (default 2, since ROIs are
        typically drawn on 2x downsampled images).

    Returns
    -------
    list[tuple[int, int]]
        List of (x, y) tuples in full-resolution pixel coordinates.
    """
    import roifile

    points: list[tuple[int, int]] = []
    rois = roifile.roiread(roi_path)
    if not isinstance(rois, list):
        rois = [rois]

    for roi in rois:
        if roi.roitype == roifile.ROI_TYPE.POINT:
            for coord in roi.coordinates():
                x = int(coord[0]) * upscale
                y = int(coord[1]) * upscale
                points.append((x, y))

    if not points:
        logger.warning(f"No point ROIs found in {roi_path}")

    return points


__all__ = [
    "ExtractionConfig",
    "MaskLike",
    "SliceJob",
    "TileJob",
    "Volume",
    "FILE_NAME",
    "DEFAULT_CROP_SIZE",
    "MAX_WIDTH_AFTER_UPSCALE",
    "TIFF_KWARGS",
    "ZERO_PIXEL_SKIP_THRESHOLD",
    "ZARR_TILE_SIZE",
    "_compute_perpendicular_slice",
    "_compute_tile_origins",
    "_distribute_file_budget",
    "_expand_positions_with_context",
    "_format_size",
    "_format_tile_filename",
    "_mask_filename",
    "_normalize_reporter",
    "_path_size_cached",
    "_prefix_with_roi",
    "_resolve_output_names",
    "_score_and_select_tiles",
    "_sample_positions",
    "_select_high_diversity_positions",
    "_squeeze_mask",
    "_validate_max_from_path",
    "_write_mask_tiff",
    "_write_tiff",
    "_resize_mask",
    "load_roi_points",
]
