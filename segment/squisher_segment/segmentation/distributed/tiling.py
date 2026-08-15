"""
Cellpose internal tiling geometry calculations.

These functions compute how raw image dimensions map to Cellpose's internal
tile layout after padding and rescaling.
"""

import numpy as np
from cellpose import transforms as cp_transforms


def _padded_from_raw_y(Ly_raw: int, *, bsize: int) -> int:
    """
    Compute padded Y size seen by Cellpose tiler for a given raw Ly.

    Uses the same padding rule as core.run_net via transforms.get_pad_yx.
    """
    ypad1, ypad2, _, _ = cp_transforms.get_pad_yx(Ly_raw, 0, min_size=(bsize, bsize))
    return Ly_raw + ypad1 + ypad2


def _padded_from_raw_x(Lx_raw: int, *, bsize: int) -> int:
    """
    Compute padded X size seen by Cellpose tiler for a given raw Lx.

    Uses the same padding rule as core.run_net via transforms.get_pad_yx.
    """
    _, _, xpad1, xpad2 = cp_transforms.get_pad_yx(0, Lx_raw, min_size=(bsize, bsize))
    return Lx_raw + xpad1 + xpad2


def _find_raw_for_target_tiles(
    n_target: int,
    *,
    bsize: int = 256,
    tile_overlap: float = 0.1,
    axis: str,
    search_margin: int = 512,
) -> tuple[int, int]:
    """
    Find the largest raw size whose padded size yields the desired tile count.

    Returns (raw_size, padded_size) along the requested axis.
    """
    if n_target <= 0:
        raise ValueError("n_target must be positive")

    alpha = 1.0 + 2.0 * tile_overlap

    if n_target == 1:
        padded_min = 0
        padded_max = bsize
    else:
        padded_min = int(np.floor((n_target - 1) * bsize / alpha)) + 1
        padded_max = int(np.floor(n_target * bsize / alpha))
        if padded_min <= bsize:
            padded_min = bsize + 1

    if padded_max < padded_min:
        raise ValueError(f"No padded size range for n_target={n_target}")

    if axis == "y":
        mapper = _padded_from_raw_y
    elif axis == "x":
        mapper = _padded_from_raw_x
    else:
        raise ValueError("axis must be 'y' or 'x'")

    raw_upper = max(bsize, padded_max + search_margin)
    best_raw: int | None = None
    best_padded: int | None = None

    for raw in range(1, raw_upper + 1):
        padded = mapper(raw, bsize=bsize)
        if padded_min <= padded <= padded_max:
            if best_raw is None or raw > best_raw:
                best_raw = raw
                best_padded = padded

    if best_raw is None or best_padded is None:
        raise ValueError(f"Could not find raw size for n_target={n_target} along axis={axis}")

    return best_raw, best_padded


def solve_internal_xy_for_tiles(
    ny_target: int,
    nx_target: int,
    *,
    bsize: int = 256,
    tile_overlap: float = 0.1,
) -> tuple[int, int]:
    """
    Solve for effective internal (Ly, Lx) that yield desired (ny, nx) tiles.

    The returned Ly/Lx correspond to the dimensions seen by Cellpose's tiler
    after padding and any diameter rescaling. This is exactly the geometry
    that run_net operates on.
    """
    if ny_target <= 0 or nx_target <= 0:
        raise ValueError("ny_target and nx_target must be positive")

    Ly_raw, _ = _find_raw_for_target_tiles(
        ny_target,
        bsize=bsize,
        tile_overlap=tile_overlap,
        axis="y",
    )
    Lx_raw, _ = _find_raw_for_target_tiles(
        nx_target,
        bsize=bsize,
        tile_overlap=tile_overlap,
        axis="x",
    )
    return Ly_raw, Lx_raw
