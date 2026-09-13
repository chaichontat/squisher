"""Select local sharpness winners and feather only their interfaces."""

from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
from multiview_stitcher.misc_utils import requires_overlap
from multiview_stitcher.weights import nan_gaussian_filter
from scipy.ndimage import gaussian_filter


def feather_ownership(ownership, valid, radius: tuple[int, ...]):
    """Finite-support Gaussian feathering cannot mix sources beyond the interface radius."""
    xp = cp if isinstance(valid, cp.ndarray) else np
    filter_func = gpu_gaussian_filter if xp is cp else gaussian_filter
    weights = xp.zeros(valid.shape, dtype=xp.float32)
    for index in range(len(valid)):
        weights[index] = filter_func(
            (ownership == index).astype(xp.float32),
            sigma=tuple(max(r / 2, 0.5) for r in radius),
            radius=radius,
            mode="nearest",
        )
    weights *= valid
    total = weights.sum(axis=0)
    weights /= xp.where(total > 0, total, 1)
    return weights


@requires_overlap(
    lambda kwargs: {
        dim: int(4 * s1 + 0.5) + int(4 * s2 + 0.5) + radius
        for dim, s1, s2, radius in zip(
            ("z", "y", "x")[-len(kwargs["sigma_1"]) :],
            kwargs["sigma_1"],
            kwargs["sigma_2"],
            kwargs["feather_radius"],
        )
    }
)
def sharpness_seam_fusion(
    transformed_views,
    blending_weights,
    *,
    sigma_1: tuple[float, ...],
    sigma_2: tuple[float, ...],
    feather_radius: tuple[int, ...],
    intensity_threshold: float | None = None,
):
    """Use normalized local high-pass energy, then blend only where winners change.

    Normalization makes the score invariant to multiplicative brightness.
    Sharpness is measured at native resolution. Halos cover both score
    convolutions and feathering, preserving the result across chunk boundaries.
    """
    xp = cp if isinstance(transformed_views, cp.ndarray) else np
    valid = (blending_weights > 1e-7) & xp.isfinite(transformed_views)
    if intensity_threshold is not None:
        valid &= transformed_views > intensity_threshold
    best_score = xp.full(transformed_views.shape[1:], -xp.inf, dtype=xp.float32)
    ownership = xp.full(transformed_views.shape[1:], -1, dtype=xp.int32)
    for index, view in enumerate(transformed_views):
        if not bool(valid[index].any()):
            continue
        masked = xp.where(valid[index], view, xp.nan)
        # Center to keep a constant view's high-pass response exactly zero.
        reference = view.ravel()[valid[index].ravel().argmax()]
        centered = masked - reference
        low = nan_gaussian_filter(centered, sigma=sigma_1, mode="nearest")
        energy = nan_gaussian_filter((centered - low) ** 2, sigma=sigma_2, mode="nearest")
        mean = nan_gaussian_filter(masked, sigma=sigma_2, mode="nearest")
        score = xp.where(mean > 0, energy / xp.where(mean > 0, mean**2, 1), 0)
        better = valid[index] & (score > best_score)
        xp.copyto(best_score, score, where=better)
        xp.copyto(ownership, index, where=better)
    del best_score
    weights = feather_ownership(ownership, valid, feather_radius)
    values = xp.where(valid, transformed_views, 0)
    return (values * weights).sum(axis=0).astype(transformed_views.dtype)
