from __future__ import annotations

import numpy as np
import numpy.typing as npt
from loguru import logger


def _sample_z_indices(z_size: int, z_samples: int, *, seed: int) -> np.ndarray:
    if z_size < 1:
        raise ValueError("Image Z dimension must be positive.")
    if z_samples < 1:
        raise ValueError("z_samples must be positive.")
    count = min(z_size, z_samples)
    if count == z_size:
        return np.arange(z_size, dtype=np.intp)
    edges = np.linspace(0, z_size, num=count + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.asarray(
        [rng.integers(edges[index], edges[index + 1]) for index in range(count)],
        dtype=np.intp,
    )


def _gpu_unsharp_planes(sampled: np.ndarray, *, radius: float) -> np.ndarray:
    """Sharpen planes independently while keeping only one plane on the GPU."""
    import cupy as cp
    from cucim.skimage import filters as cucim_filters

    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("CUDA is required for normalization unsharp filtering.")

    pool = cp.get_default_memory_pool()
    pinned_pool = cp.get_default_pinned_memory_pool()
    filtered = np.empty(sampled.shape, dtype=np.float32)
    try:
        for z_index in range(sampled.shape[0]):
            plane_gpu = cp.asarray(sampled[z_index], dtype=cp.float32)
            try:
                sharpened = cucim_filters.unsharp_mask(
                    plane_gpu,
                    radius=radius,
                    preserve_range=True,
                    channel_axis=2,
                )
                try:
                    cp.cuda.get_current_stream().synchronize()
                    filtered[z_index] = cp.asnumpy(sharpened)
                finally:
                    del sharpened
            finally:
                del plane_gpu
                pool.free_all_blocks()
                pinned_pool.free_all_blocks()
        return filtered
    finally:
        pool.free_all_blocks()
        pinned_pool.free_all_blocks()


def sample_percentiles(
    img: npt.ArrayLike,
    channels: list[int],
    block: tuple[int, int] = (512, 512),
    *,
    n: int = 25,
    low: float = 1,
    high: float = 99,
    seed: int = 0,
    z_samples: int = 32,
    unsharp: bool = True,
    unsharp_radius: float = 3.0,
    foreground_channel: int | None = None,
    foreground_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample percentile ranges from a large 4D stack.

    Assumes input is shaped (Z, Y, X, C) and returns per-channel
    low/high percentiles computed from up to ``n`` randomly sampled
    spatial crops of size ``block`` across deterministic representative Z
    planes after plane-wise GPU unsharp filtering. Source reads are limited to
    the candidate spatial crop, and only one cropped plane is held on the GPU.
    One Z plane is sampled from each equal-depth stratum to avoid alignment with
    regularly spaced empty bands.

    Behavior matches the historical implementation used by the
    distributed segmentation scripts and preserves the 1-based
    channel indexing convention those scripts expect.

    Parameters
    - img: Array-like of shape (Z, Y, X, C)
    - channels: 1-based channel indices to include (e.g., [1, 2])
    - block: (height, width) of sampled crops
    - n: max number of crops to sample
    - low, high: percentile bounds to compute (0-100)
    - seed: RNG seed for reproducibility
    - z_samples: maximum number of stratified source Z planes to load
    - foreground_channel: optional 1-based channel defining valid voxels
    - foreground_threshold: strict lower bound for valid foreground voxels

    Returns
    - mean_perc: array (n_channels, 2) with [[low, high], ...]
    - all_samples: array (n_samples, 2, n_channels) of individual crop percentiles
    """
    # Accept numpy, zarr, or array-likes that support shape and slicing
    arr = img  # type: ignore[assignment]
    try:
        ndim = arr.ndim  # type: ignore[attr-defined]
        shape = arr.shape  # type: ignore[attr-defined]
    except Exception:
        arr = np.asarray(img)
        ndim = arr.ndim
        shape = arr.shape

    if ndim != 4:
        raise ValueError("Expected image with shape (Z, Y, X, C).")
    if shape[1] < block[0] or shape[2] < block[1]:
        raise ValueError("Block size larger than image spatial dimensions.")
    if (foreground_channel is None) != (foreground_threshold is None):
        raise ValueError("foreground_channel and foreground_threshold must be provided together.")
    if foreground_channel is not None and (
        isinstance(foreground_channel, bool)
        or foreground_channel < 1
        or foreground_channel > shape[3]
    ):
        raise ValueError("foreground_channel must be a valid 1-based channel index.")

    z_indices = _sample_z_indices(int(shape[0]), z_samples, seed=seed)
    logger.info(
        f"Selected {len(z_indices)} representative Z planes from {shape[0]} "
        f"for bounded {block[0]}x{block[1]} normalization crops"
    )

    rng = np.random.default_rng(seed)
    # Historical convention: incoming channels are 1-based
    ch_idx = [c - 1 for c in channels]
    foreground_idx = foreground_channel - 1 if foreground_channel is not None else None

    # Over-sample starts to allow skipping zero-padded/stitch-edge crops
    y_starts = rng.integers(0, shape[1] - block[0] + 1, n * 2)
    x_starts = rng.integers(0, shape[2] - block[1] + 1, n * 2)

    samples: list[np.ndarray] = []
    taken = 0
    for y_start, x_start in zip(y_starts, x_starts):
        if taken >= n:
            break
        y_slice = slice(y_start, y_start + block[0])
        x_slice = slice(x_start, x_start + block[1])
        raw_crop = np.stack(
            [
                np.asarray(arr[int(z_index), y_slice, x_slice, :])
                for z_index in z_indices
            ],
            axis=0,
        )
        logger.info(f"Sampled crop {taken} at ({y_start}, {x_start})")
        if foreground_idx is not None:
            foreground = raw_crop[..., foreground_idx] > foreground_threshold
            if not np.any(foreground):
                logger.info("Crop rejected: no foreground voxels")
                continue
        else:
            # Preserve the generic sampler's historical stitched-edge rejection.
            sel = raw_crop[..., ch_idx]
            total = sel.size
            zero_ratio = float(np.count_nonzero(sel == 0)) / max(total, 1)
            one_ratio = float(np.count_nonzero(sel == 1)) / max(total, 1)
            if zero_ratio > 0.10 or one_ratio > 0.10:
                logger.info(
                    f"Crop rejected: zero_ratio={zero_ratio:.3f}, "
                    f"one_ratio={one_ratio:.3f} (>0.10 threshold)"
                )
                continue

        filtered_crop = (
            _gpu_unsharp_planes(raw_crop, radius=unsharp_radius) if unsharp else raw_crop
        )
        if foreground_idx is not None:
            values = filtered_crop[..., ch_idx][foreground]
            samples.append(np.percentile(values, [low, high], axis=0))
        else:
            samples.append(
                np.percentile(filtered_crop[..., ch_idx], [low, high], axis=(0, 1, 2))
            )
        taken += 1

    if not samples:
        raise RuntimeError("No valid crops sampled; image may be mostly zeros or block too large.")

    all_samples = np.asarray(samples)  # (n_samples, 2, n_channels)
    mean_perc = all_samples.mean(axis=0).T  # -> (n_channels, 2)
    return mean_perc, all_samples


def sample_percentile(
    img: npt.ArrayLike,
    channels: list[int],
    block: tuple[int, int] = (512, 512),
    *,
    n: int = 25,
    low: float = 1,
    high: float = 99,
    seed: int = 0,
    z_samples: int = 32,
    unsharp: bool = True,
    unsharp_radius: float = 3.0,
    foreground_channel: int | None = None,
    foreground_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper returning the same result as sample_percentiles."""

    return sample_percentiles(
        img,
        channels,
        block,
        n=n,
        low=low,
        high=high,
        seed=seed,
        z_samples=z_samples,
        unsharp=unsharp,
        unsharp_radius=unsharp_radius,
        foreground_channel=foreground_channel,
        foreground_threshold=foreground_threshold,
    )
