from __future__ import annotations

from typing import Literal

import numpy as np

ShearMode = Literal["sum_y", "central_y", "max_y"]


def weighted_linear_fit(z_px: np.ndarray, x_px: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(z_px) & np.isfinite(x_px) & np.isfinite(weights) & (weights > 0)
    if int(finite.sum()) < 3:
        return float("nan"), float("nan")
    coeff = np.polyfit(z_px[finite], x_px[finite], deg=1, w=np.sqrt(weights[finite]))
    return float(coeff[0]), float(coeff[1])


def x_centers_by_z(
    volume: np.ndarray,
    *,
    z_fraction_threshold: float,
    mode: ShearMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 <= z_fraction_threshold <= 1:
        raise ValueError("z_fraction_threshold must be between 0 and 1.")

    arr = np.asarray(volume, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"volume must be 3D ZYX, got shape {arr.shape}.")

    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)

    if mode == "sum_y":
        xz = arr.sum(axis=1)
    elif mode == "central_y":
        _, yc, _ = np.unravel_index(int(np.argmax(arr)), arr.shape)
        xz = arr[:, yc, :]
    elif mode == "max_y":
        xz = arr.max(axis=1)
    else:
        raise ValueError(f"Unknown shear mode: {mode!r}")

    z_mass = xz.sum(axis=1)
    threshold = z_fraction_threshold * float(z_mass.max())
    keep = z_mass >= threshold

    xs = np.arange(arr.shape[2], dtype=np.float64)
    x_centers = np.full(arr.shape[0], np.nan, dtype=np.float64)
    for z in np.where(keep)[0]:
        row = xz[z]
        total = float(row.sum())
        if total > 0:
            x_centers[z] = float(np.sum(xs * row) / total)

    z_px = np.arange(arr.shape[0], dtype=np.float64)
    return z_px, x_centers, z_mass, keep


def shear_for_volume(
    volume: np.ndarray,
    *,
    spacing_zyx_um: tuple[float, float, float],
    z_fraction_threshold: float,
    mode: ShearMode,
) -> dict[str, object]:
    spacing_z_um, _, spacing_x_um = spacing_zyx_um
    if min(spacing_zyx_um) <= 0:
        raise ValueError("spacing_zyx_um values must be positive.")

    z_px, x_centers, z_mass, keep = x_centers_by_z(
        volume,
        z_fraction_threshold=z_fraction_threshold,
        mode=mode,
    )
    slope_px_per_zpx, intercept_px = weighted_linear_fit(z_px[keep], x_centers[keep], z_mass[keep])
    slope_um_per_um = slope_px_per_zpx * spacing_x_um / spacing_z_um
    angle_deg = float(np.degrees(np.arctan(slope_um_per_um)))
    kept = np.where(keep & np.isfinite(x_centers))[0]
    return {
        "mode": mode,
        "z_fraction_threshold": z_fraction_threshold,
        "slope_x_px_per_z_plane": slope_px_per_zpx,
        "intercept_x_px": intercept_px,
        "slope_x_um_per_z_um": float(slope_um_per_um),
        "angle_deg_from_z_axis": angle_deg,
        "kept_z_indices": [int(v) for v in kept],
        "x_centers_px": [None if not np.isfinite(v) else float(v) for v in x_centers],
        "z_mass": [float(v) for v in z_mass],
    }


def summarize_distribution(values: np.ndarray) -> dict[str, float | int]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "mad": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
        }
    return {
        "n": int(vals.size),
        "median": float(np.median(vals)),
        "mad": float(np.median(np.abs(vals - np.median(vals)))),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "p16": float(np.percentile(vals, 16)),
        "p84": float(np.percentile(vals, 84)),
    }


def crop_shear_distribution(
    crops: np.ndarray,
    *,
    spacing_zyx_um: tuple[float, float, float],
    z_fraction_threshold: float,
    mode: ShearMode,
) -> dict[str, object]:
    arr = np.asarray(crops, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"crops must be 4D NZYX, got shape {arr.shape}.")

    slopes_px = []
    slopes_um = []
    angles = []
    for crop in arr:
        metrics = shear_for_volume(
            crop,
            spacing_zyx_um=spacing_zyx_um,
            z_fraction_threshold=z_fraction_threshold,
            mode=mode,
        )
        slopes_px.append(float(metrics["slope_x_px_per_z_plane"]))
        slopes_um.append(float(metrics["slope_x_um_per_z_um"]))
        angles.append(float(metrics["angle_deg_from_z_axis"]))
    return {
        "mode": mode,
        "z_fraction_threshold": z_fraction_threshold,
        "slope_x_px_per_z_plane": summarize_distribution(np.asarray(slopes_px)),
        "slope_x_um_per_z_um": summarize_distribution(np.asarray(slopes_um)),
        "angle_deg_from_z_axis": summarize_distribution(np.asarray(angles)),
    }
