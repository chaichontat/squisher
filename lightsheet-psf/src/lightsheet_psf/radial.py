from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates, shift


def peak_normalize(image: np.ndarray) -> np.ndarray:
    peak = float(np.nanmax(image))
    if peak <= 0:
        raise ValueError("Cannot peak-normalize a non-positive PSF.")
    return image / peak


def sum_normalize(volume: np.ndarray) -> np.ndarray:
    total = float(np.sum(volume))
    if total <= 0:
        raise ValueError("Cannot sum-normalize a non-positive PSF.")
    return volume / total


def fwhm_1d(profile: np.ndarray, spacing_um: float) -> float | None:
    if spacing_um <= 0:
        raise ValueError("spacing_um must be positive.")
    profile = np.asarray(profile, dtype=np.float64)
    if float(np.nanmax(profile)) <= 0:
        return None
    peak_idx = int(np.nanargmax(profile))
    half = float(profile[peak_idx]) / 2.0

    left_idx = np.where(profile[: peak_idx + 1] < half)[0]
    right_idx = np.where(profile[peak_idx:] < half)[0]
    if len(left_idx) == 0 or len(right_idx) == 0:
        return None

    li = int(left_idx[-1])
    ri = int(peak_idx + right_idx[0])

    def interp_cross(i0: int, i1: int) -> float:
        y0 = float(profile[i0])
        y1 = float(profile[i1])
        if y1 == y0:
            return float(i0)
        return i0 + (half - y0) / (y1 - y0)

    left = interp_cross(li, li + 1)
    right = interp_cross(ri - 1, ri)
    return float((right - left) * spacing_um)


def central_fwhm(image: np.ndarray, *, spacing_zyx_um: tuple[float, float, float]) -> dict[str, float | None]:
    if min(spacing_zyx_um) <= 0:
        raise ValueError("spacing_zyx_um values must be positive.")
    zc, yc, xc = np.unravel_index(int(np.nanargmax(image)), image.shape)
    return {
        "x_um": fwhm_1d(image[zc, yc, :], spacing_zyx_um[2]),
        "y_um": fwhm_1d(image[zc, :, xc], spacing_zyx_um[1]),
        "z_um": fwhm_1d(image[:, yc, xc], spacing_zyx_um[0]),
    }


def z_fwhm_along_xz_shear(
    image: np.ndarray,
    slope_x_px_per_z: float,
    *,
    spacing_z_um: float,
) -> float | None:
    zc, yc, xc = np.unravel_index(int(np.nanargmax(image)), image.shape)
    z = np.arange(image.shape[0], dtype=np.float64)
    y = np.full_like(z, float(yc))
    x = float(xc) + slope_x_px_per_z * (z - float(zc))
    profile = map_coordinates(
        np.asarray(image, dtype=np.float64),
        np.vstack([z, y, x]),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return fwhm_1d(profile, spacing_z_um)


def shear_volume_x(volume: np.ndarray, slope_x_px_per_z: float, *, inverse: bool = False) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"volume must be 3D ZYX, got shape {arr.shape}.")
    z_ref = (arr.shape[0] - 1) / 2.0
    out = np.zeros_like(arr, dtype=np.float64)
    sign = 1.0 if inverse else -1.0
    for z in range(arr.shape[0]):
        x_shift = sign * slope_x_px_per_z * (z - z_ref)
        out[z] = shift(
            arr[z],
            shift=(0.0, x_shift),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return np.clip(out, 0, None)


def radial_average_xy(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"volume must be 3D ZYX, got shape {arr.shape}.")
    _, yc, xc = np.unravel_index(int(np.nanargmax(arr)), arr.shape)
    yy, xx = np.indices(arr.shape[1:])
    radius = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
    radius_key = np.round(radius, 8)
    radial_samples = np.unique(radius_key)
    out = np.zeros_like(arr, dtype=np.float64)

    for z in range(arr.shape[0]):
        radial_profile = np.array(
            [float(np.mean(arr[z, radius_key == r])) for r in radial_samples],
            dtype=np.float64,
        )
        radial_profile = np.minimum.accumulate(radial_profile)
        interpolator = PchipInterpolator(radial_samples, radial_profile, extrapolate=True)
        out[z] = np.clip(interpolator(radius), 0, None)
    return out
