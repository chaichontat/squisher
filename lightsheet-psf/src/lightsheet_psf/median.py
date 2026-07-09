from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import shift
from scipy.spatial import cKDTree


def fwhm_1d_px(profile: np.ndarray) -> float | None:
    profile = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(profile)
    if not finite.any() or np.nanmax(profile) <= 0:
        return None

    peak_idx = int(np.nanargmax(profile))
    half = float(profile[peak_idx]) / 2.0
    left = np.where(np.isfinite(profile[: peak_idx + 1]) & (profile[: peak_idx + 1] < half))[0]
    right = np.where(np.isfinite(profile[peak_idx:]) & (profile[peak_idx:] < half))[0]
    if len(left) == 0 or len(right) == 0:
        return None

    li = int(left[-1])
    ri = int(peak_idx + right[0])

    def interp_cross(i0: int, i1: int) -> float:
        y0 = float(profile[i0])
        y1 = float(profile[i1])
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
            return float(i0)
        return i0 + (half - y0) / (y1 - y0)

    return float(interp_cross(ri - 1, ri) - interp_cross(li, li + 1))


def crop_fwhm_px(crop: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if not np.isfinite(crop).any():
        return None, None, None
    zc, yc, xc = np.unravel_index(int(np.nanargmax(crop)), crop.shape)
    return (
        fwhm_1d_px(crop[zc, yc, :]),
        fwhm_1d_px(crop[zc, :, xc]),
        fwhm_1d_px(crop[:, yc, xc]),
    )


def z_profile_plus_to_minus_ratio(crop: np.ndarray) -> float | None:
    if not np.isfinite(crop).any():
        return None
    zc, yc, xc = np.unravel_index(int(np.nanargmax(crop)), crop.shape)
    profile = np.asarray(crop[:, yc, xc], dtype=np.float64)
    radius = min(zc, len(profile) - 1 - zc)
    if radius < 2:
        return None

    minus = profile[zc - radius : zc][::-1]
    plus = profile[zc + 1 : zc + 1 + radius]
    finite = np.isfinite(minus) & np.isfinite(plus)
    if not finite.any():
        return None

    minus_sum = float(np.sum(np.clip(minus[finite], a_min=0, a_max=None)))
    plus_sum = float(np.sum(np.clip(plus[finite], a_min=0, a_max=None)))
    if minus_sum <= 0 or plus_sum < 0:
        return None
    return plus_sum / minus_sum


def z_near_peak_min_ratio(crop: np.ndarray, start_offset: int = 2, stop_offset: int = 4) -> float | None:
    if not np.isfinite(crop).any():
        return None
    zc, yc, xc = np.unravel_index(int(np.nanargmax(crop)), crop.shape)
    profile = np.asarray(crop[:, yc, xc], dtype=np.float64)
    radius = min(zc, len(profile) - 1 - zc)
    if radius < stop_offset:
        return None

    minus = profile[zc - radius : zc][::-1]
    plus = profile[zc + 1 : zc + 1 + radius]
    ratios: list[float] = []
    for offset in range(start_offset, stop_offset + 1):
        minus_val = float(minus[offset - 1])
        plus_val = float(plus[offset - 1])
        if not np.isfinite(minus_val) or not np.isfinite(plus_val) or minus_val <= 0:
            continue
        ratios.append(max(plus_val, 0.0) / minus_val)
    if not ratios:
        return None
    return float(min(ratios))


def centered_z_shape_metrics(
    crop: np.ndarray, support_fraction: float = 0.2, pre_tail_offset: int = 4
) -> tuple[float, float, float]:
    rz, ry, rx = (size // 2 for size in crop.shape)
    profile = np.asarray(crop[:, ry, rx], dtype=np.float64)
    finite = np.isfinite(profile)
    if not finite.any() or np.nanmax(profile) <= 0:
        return float("nan"), float("nan"), float("nan")

    peak = float(np.nanmax(profile))
    peak_z = int(np.nanargmax(profile))
    above = np.where(np.isfinite(profile) & (profile >= support_fraction * peak))[0]
    support_span = float(above[-1] - above[0] + 1) if above.size else 0.0
    pre_idx = rz - pre_tail_offset
    pre_tail_fraction = (
        float("nan")
        if pre_idx < 0 or not np.isfinite(profile[pre_idx])
        else float(max(profile[pre_idx], 0.0) / peak)
    )
    return float(peak_z - rz), support_span, pre_tail_fraction


def robust_keep_mask(
    values: np.ndarray, mad_mult: float, min_abs_tol: float
) -> tuple[np.ndarray, float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    keep = np.zeros(vals.shape, dtype=bool)
    if not finite.any():
        return keep, float("nan"), float("nan"), float("nan")
    median = float(np.median(vals[finite]))
    mad = float(np.median(np.abs(vals[finite] - median)))
    tol = max(mad_mult * 1.4826 * mad, min_abs_tol)
    keep[finite] = np.abs(vals[finite] - median) <= tol
    return keep, median, mad, tol


def add_quality_flags(
    beads: pd.DataFrame,
    stack_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    min_xy_distance: float,
) -> pd.DataFrame:
    out = beads.copy()
    rz, ry, rx = (size // 2 for size in crop_shape)
    z_size, y_size, x_size = stack_shape
    out["z_round"] = np.rint(out["z"]).astype(int)
    out["y_round"] = np.rint(out["y"]).astype(int)
    out["x_round"] = np.rint(out["x"]).astype(int)
    xy = out[["x", "y"]].to_numpy(dtype=np.float64)
    out["nearest_xy_px"] = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(out) > 1 else np.inf
    out["full_crop"] = (
        (out["z_round"] >= rz)
        & (out["z_round"] < z_size - rz)
        & (out["y_round"] >= ry)
        & (out["y_round"] < y_size - ry)
        & (out["x_round"] >= rx)
        & (out["x_round"] < x_size - rx)
    )
    out["center_in_bounds"] = (
        (out["z_round"] >= 0)
        & (out["z_round"] < z_size)
        & (out["y_round"] >= 0)
        & (out["y_round"] < y_size)
        & (out["x_round"] >= 0)
        & (out["x_round"] < x_size)
    )
    out["unsaturated"] = out["peak_intensity"] < np.iinfo(np.uint16).max
    out["isolated_xy"] = out["nearest_xy_px"] >= min_xy_distance
    out["basic_quality"] = out["center_in_bounds"] & out["unsaturated"] & out["isolated_xy"]
    out["good_quality"] = out["basic_quality"]
    return out


def crop_and_align(stack: np.ndarray, bead: pd.Series, crop_shape: tuple[int, int, int]) -> np.ndarray:
    rz, ry, rx = (size // 2 for size in crop_shape)
    zc = int(bead["z_round"])
    yc = int(bead["y_round"])
    xc = int(bead["x_round"])
    crop = np.full(crop_shape, np.nan, dtype=np.float32)
    z0_src = max(0, zc - rz)
    z1_src = min(stack.shape[0], zc + rz + 1)
    y0_src = max(0, yc - ry)
    y1_src = min(stack.shape[1], yc + ry + 1)
    x0_src = max(0, xc - rx)
    x1_src = min(stack.shape[2], xc + rx + 1)
    z0_dst = z0_src - (zc - rz)
    y0_dst = y0_src - (yc - ry)
    x0_dst = x0_src - (xc - rx)
    crop[
        z0_dst : z0_dst + (z1_src - z0_src),
        y0_dst : y0_dst + (y1_src - y0_src),
        x0_dst : x0_dst + (x1_src - x0_src),
    ] = stack[z0_src:z1_src, y0_src:y1_src, x0_src:x1_src]
    aligned = shift(
        crop,
        shift=(zc - float(bead["z"]), yc - float(bead["y"]), xc - float(bead["x"])),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return aligned.astype(np.float32, copy=False)


def subtract_border_background(crop: np.ndarray) -> np.ndarray:
    border = np.zeros(crop.shape, dtype=bool)
    border[[0, -1], :, :] = True
    border[:, [0, -1], :] = True
    border[:, :, [0, -1]] = True
    bg = float(np.nanmedian(crop[border]))
    return np.clip(crop - bg, a_min=0, a_max=None)


def build_medians(
    stack: np.ndarray,
    quality: pd.DataFrame,
    crop_shape: tuple[int, int, int],
    size_mad_mult: float,
    z_asym_mad_mult: float,
    z_near_ratio_floor: float,
    require_full_crop: bool,
    min_z_fwhm_px: float | None,
    max_z_fwhm_px: float | None,
    min_z_support_span_px: float | None,
    max_z_support_span_px: float | None,
    min_z_pre_tail_fraction: float | None,
    max_central_z_peak_offset_px: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_mask = quality["basic_quality"].to_numpy(dtype=bool)
    crop_store: dict[int, np.ndarray] = {}
    peaks = np.full(len(quality), np.nan, dtype=np.float64)
    fwhm_x = np.full(len(quality), np.nan, dtype=np.float64)
    fwhm_y = np.full(len(quality), np.nan, dtype=np.float64)
    fwhm_z = np.full(len(quality), np.nan, dtype=np.float64)
    z_plus_to_minus = np.full(len(quality), np.nan, dtype=np.float64)
    z_near_min_ratio = np.full(len(quality), np.nan, dtype=np.float64)
    central_z_peak_offset = np.full(len(quality), np.nan, dtype=np.float64)
    z_support_span_02 = np.full(len(quality), np.nan, dtype=np.float64)
    z_pre_tail_fraction_offset4 = np.full(len(quality), np.nan, dtype=np.float64)

    for idx, bead in quality.loc[candidate_mask].iterrows():
        crop = subtract_border_background(crop_and_align(stack, bead, crop_shape))
        peak = float(np.nanmax(crop))
        if not np.isfinite(peak) or peak <= 0:
            continue
        fx, fy, fz = crop_fwhm_px(crop)
        z_ratio = z_profile_plus_to_minus_ratio(crop)
        z_min_ratio = z_near_peak_min_ratio(crop)
        z_peak_offset, z_support_span, z_pre_tail_fraction = centered_z_shape_metrics(crop)
        crop_store[int(idx)] = crop
        peaks[int(idx)] = peak
        fwhm_x[int(idx)] = np.nan if fx is None else fx
        fwhm_y[int(idx)] = np.nan if fy is None else fy
        fwhm_z[int(idx)] = np.nan if fz is None else fz
        z_plus_to_minus[int(idx)] = np.nan if z_ratio is None else z_ratio
        z_near_min_ratio[int(idx)] = np.nan if z_min_ratio is None else z_min_ratio
        central_z_peak_offset[int(idx)] = z_peak_offset
        z_support_span_02[int(idx)] = z_support_span
        z_pre_tail_fraction_offset4[int(idx)] = z_pre_tail_fraction

    quality["crop_peak"] = peaks
    quality["fwhm_x_px"] = fwhm_x
    quality["fwhm_y_px"] = fwhm_y
    quality["fwhm_z_px"] = fwhm_z
    quality["z_plus_to_minus_ratio"] = z_plus_to_minus
    quality["z_near_peak_min_ratio"] = z_near_min_ratio
    quality["central_z_peak_offset_px"] = central_z_peak_offset
    quality["z_support_span_02_px"] = z_support_span_02
    quality["z_pre_tail_fraction_offset4"] = z_pre_tail_fraction_offset4
    z_log_ratio = np.full(len(quality), np.nan, dtype=np.float64)
    positive_ratio = z_near_min_ratio > 0
    z_log_ratio[positive_ratio] = np.log(z_near_min_ratio[positive_ratio])
    quality["z_near_peak_min_log_ratio"] = z_log_ratio
    xy_values = np.column_stack([fwhm_x, fwhm_y])
    xy_sum = np.nansum(xy_values, axis=1)
    xy_count = np.sum(np.isfinite(xy_values), axis=1)
    quality["fwhm_xy_mean_px"] = np.divide(
        xy_sum, xy_count, out=np.full(len(quality), np.nan), where=xy_count > 0
    )

    calib_mask = (
        quality["basic_quality"]
        & quality["full_crop"]
        & np.isfinite(quality["fwhm_xy_mean_px"])
        & np.isfinite(quality["fwhm_z_px"])
    )
    if not bool(calib_mask.any()):
        calib_mask = (
            quality["basic_quality"]
            & np.isfinite(quality["fwhm_xy_mean_px"])
            & np.isfinite(quality["fwhm_z_px"])
        )
    xy_keep, xy_median, xy_mad, xy_tol = robust_keep_mask(
        quality.loc[calib_mask, "fwhm_xy_mean_px"].to_numpy(), size_mad_mult, 0.35
    )
    z_keep, z_median, z_mad, z_tol = robust_keep_mask(
        quality.loc[calib_mask, "fwhm_z_px"].to_numpy(), size_mad_mult, 0.75
    )
    quality["size_xy_ok"] = False
    quality["size_z_ok"] = False
    quality.loc[quality.index[calib_mask], "size_xy_ok"] = xy_keep
    quality.loc[quality.index[calib_mask], "size_z_ok"] = z_keep
    measured_xy = quality["basic_quality"] & np.isfinite(quality["fwhm_xy_mean_px"])
    measured_z = quality["basic_quality"] & np.isfinite(quality["fwhm_z_px"])
    quality.loc[measured_xy, "size_xy_ok"] = (
        np.abs(quality.loc[measured_xy, "fwhm_xy_mean_px"] - xy_median) <= xy_tol
    )
    quality.loc[measured_z, "size_z_ok"] = np.abs(quality.loc[measured_z, "fwhm_z_px"] - z_median) <= z_tol
    quality["size_consistent"] = quality["size_xy_ok"] & quality["size_z_ok"]

    asym_calib_mask = (
        quality["basic_quality"]
        & quality["size_consistent"]
        & quality["full_crop"]
        & np.isfinite(quality["z_near_peak_min_log_ratio"])
    )
    if not bool(asym_calib_mask.any()):
        asym_calib_mask = (
            quality["basic_quality"]
            & quality["size_consistent"]
            & np.isfinite(quality["z_near_peak_min_log_ratio"])
        )
    asym_keep, asym_median, asym_mad, asym_tol = robust_keep_mask(
        quality.loc[asym_calib_mask, "z_near_peak_min_log_ratio"].to_numpy(),
        z_asym_mad_mult,
        0.25,
    )
    quality["z_asym_ok"] = False
    quality.loc[quality.index[asym_calib_mask], "z_asym_ok"] = asym_keep
    measured_asym = (
        quality["basic_quality"]
        & quality["size_consistent"]
        & np.isfinite(quality["z_near_peak_min_log_ratio"])
    )
    quality.loc[measured_asym, "z_asym_ok"] = (
        quality.loc[measured_asym, "z_near_peak_min_log_ratio"] >= asym_median - asym_tol
    )
    quality["z_asym_ok"] = quality["z_asym_ok"] & (quality["z_near_peak_min_ratio"] >= z_near_ratio_floor)

    quality["z_fwhm_min_ok"] = True if min_z_fwhm_px is None else quality["fwhm_z_px"] >= min_z_fwhm_px
    quality["z_fwhm_max_ok"] = True if max_z_fwhm_px is None else quality["fwhm_z_px"] <= max_z_fwhm_px
    quality["z_support_span_ok"] = (
        True if min_z_support_span_px is None else quality["z_support_span_02_px"] >= min_z_support_span_px
    )
    quality["z_support_span_max_ok"] = (
        True if max_z_support_span_px is None else quality["z_support_span_02_px"] <= max_z_support_span_px
    )
    quality["z_pre_tail_ok"] = (
        True
        if min_z_pre_tail_fraction is None
        else quality["z_pre_tail_fraction_offset4"] >= min_z_pre_tail_fraction
    )
    quality["central_z_peak_offset_ok"] = (
        True
        if max_central_z_peak_offset_px is None
        else quality["central_z_peak_offset_px"].abs() <= max_central_z_peak_offset_px
    )
    quality["full_crop_required_ok"] = True if not require_full_crop else quality["full_crop"]
    quality["good_quality"] = (
        quality["basic_quality"]
        & quality["full_crop_required_ok"]
        & quality["size_consistent"]
        & quality["z_asym_ok"]
        & quality["z_fwhm_min_ok"]
        & quality["z_fwhm_max_ok"]
        & quality["z_support_span_ok"]
        & quality["z_support_span_max_ok"]
        & quality["z_pre_tail_ok"]
        & quality["central_z_peak_offset_ok"]
        & np.isfinite(quality["crop_peak"])
    )
    quality["size_reference_xy_median_px"] = xy_median
    quality["size_reference_xy_mad_px"] = xy_mad
    quality["size_reference_xy_tol_px"] = xy_tol
    quality["size_reference_z_median_px"] = z_median
    quality["size_reference_z_mad_px"] = z_mad
    quality["size_reference_z_tol_px"] = z_tol
    quality["z_asym_reference_log_ratio_median"] = asym_median
    quality["z_asym_reference_log_ratio_mad"] = asym_mad
    quality["z_asym_reference_log_ratio_tol"] = asym_tol
    quality["z_asym_reference_ratio_lower_bound"] = (
        np.exp(asym_median - asym_tol) if np.isfinite(asym_median) else np.nan
    )
    quality["z_asym_reference_ratio_floor"] = z_near_ratio_floor
    quality["z_fwhm_min_px"] = np.nan if min_z_fwhm_px is None else min_z_fwhm_px
    quality["z_fwhm_max_px"] = np.nan if max_z_fwhm_px is None else max_z_fwhm_px
    quality["z_support_span_min_px"] = np.nan if min_z_support_span_px is None else min_z_support_span_px
    quality["z_support_span_max_px"] = np.nan if max_z_support_span_px is None else max_z_support_span_px
    quality["z_pre_tail_fraction_offset4_min"] = (
        np.nan if min_z_pre_tail_fraction is None else min_z_pre_tail_fraction
    )
    quality["central_z_peak_offset_max_px"] = (
        np.nan if max_central_z_peak_offset_px is None else max_central_z_peak_offset_px
    )

    crops = []
    norm_crops = []
    for idx, _bead in quality[quality["good_quality"]].iterrows():
        crop = crop_store.get(int(idx))
        peak = float(quality.at[idx, "crop_peak"]) if crop is not None else float("nan")
        if crop is None or not np.isfinite(peak) or peak <= 0:
            continue
        crops.append(crop)
        norm_crops.append(crop / peak)
    if not crops:
        raise RuntimeError("No good-quality crops remained after filtering.")
    crop_stack = np.stack(crops, axis=0)
    norm_crop_stack = np.stack(norm_crops, axis=0)
    return (
        crop_stack,
        np.nanmedian(crop_stack, axis=0).astype(np.float32),
        np.nanmedian(norm_crop_stack, axis=0).astype(np.float32),
    )


def write_median_qc_png(
    quality: pd.DataFrame, raw_median: np.ndarray, norm_median: np.ndarray, output: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=180)
    good = quality["good_quality"]
    axes[0, 0].hist(quality["z"], bins=np.arange(-0.5, 50.5, 1), color="0.25")
    axes[0, 0].hist(quality.loc[good, "z"], bins=np.arange(-0.5, 50.5, 1), color="tab:green")
    axes[0, 0].set_title(f"Z centers: {int(good.sum())}/{len(quality)} kept")
    axes[0, 0].set_xlabel("z")
    axes[0, 0].set_ylabel("count")
    axes[0, 1].hist(quality["nearest_xy_px"], bins=60, color="0.25")
    axes[0, 1].hist(quality.loc[good, "nearest_xy_px"], bins=60, color="tab:green")
    axes[0, 1].set_title("Nearest XY distance")
    axes[0, 1].set_xlabel("px")
    axes[0, 2].scatter(quality.loc[~good, "x"], quality.loc[~good, "y"], s=1, c="tab:red", alpha=0.35)
    axes[0, 2].scatter(quality.loc[good, "x"], quality.loc[good, "y"], s=1, c="tab:green", alpha=0.65)
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set_title("Good/rejected XY")
    xy_proj = np.nanmax(norm_median, axis=0)
    yz_proj = np.nanmax(norm_median, axis=2)
    xz_proj = np.nanmax(norm_median, axis=1)
    axes[1, 0].imshow(xy_proj, cmap="magma", vmax=float(np.nanpercentile(xy_proj, 99.5)))
    axes[1, 0].set_title("Median max XY")
    axes[1, 1].imshow(yz_proj, cmap="magma", aspect="auto", vmax=float(np.nanpercentile(yz_proj, 99.5)))
    axes[1, 1].set_title("Median max ZY")
    axes[1, 2].imshow(xz_proj, cmap="magma", aspect="auto", vmax=float(np.nanpercentile(xz_proj, 99.5)))
    axes[1, 2].set_title("Median max ZX")
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"raw_peak={float(np.nanmax(raw_median)):.1f}, norm_peak={float(np.nanmax(norm_median)):.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
