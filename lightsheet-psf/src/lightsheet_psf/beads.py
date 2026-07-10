from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def local_z_center(
    stack: np.ndarray,
    y: float,
    x: float,
    *,
    xy_radius: int,
    z_radius: int,
) -> tuple[float, int, float]:
    yi = int(round(y))
    xi = int(round(x))
    y0 = max(0, yi - xy_radius)
    y1 = min(stack.shape[1], yi + xy_radius + 1)
    x0 = max(0, xi - xy_radius)
    x1 = min(stack.shape[2], xi + xy_radius + 1)

    profile = stack[:, y0:y1, x0:x1].max(axis=(1, 2)).astype(np.float64)
    z_peak = int(np.argmax(profile))
    z0 = max(0, z_peak - z_radius)
    z1 = min(stack.shape[0], z_peak + z_radius + 1)
    local = profile[z0:z1]
    baseline = float(np.percentile(local, 20))
    weights = np.clip(local - baseline, a_min=0, a_max=None)
    if weights.sum() <= 0:
        return float(z_peak), z_peak, float(profile[z_peak])
    zs = np.arange(z0, z1, dtype=np.float64)
    return float(np.sum(zs * weights) / np.sum(weights)), z_peak, float(profile[z_peak])


def detect_beads(
    stack: np.ndarray,
    *,
    fwhm: float,
    threshold_sigma: float,
    brightest: int | None,
    xy_radius: int,
    z_radius: int,
) -> pd.DataFrame:
    projection = stack.max(axis=0).astype(np.float32, copy=False)
    mean, median, std = sigma_clipped_stats(projection, sigma=3.0)
    finder = DAOStarFinder(
        threshold=threshold_sigma * float(std),
        fwhm=fwhm,
        brightest=brightest,
        exclude_border=False,
    )
    spots = finder(np.clip(projection, a_min=median, a_max=None) - median)
    if spots is None:
        return pd.DataFrame()

    df = spots.to_pandas()
    x_col = "xcentroid" if "xcentroid" in df.columns else "x_centroid"
    y_col = "ycentroid" if "ycentroid" in df.columns else "y_centroid"
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"DAOStarFinder output is missing centroid columns; got {list(df.columns)}")
    z_info = [
        local_z_center(stack, float(row[y_col]), float(row[x_col]), xy_radius=xy_radius, z_radius=z_radius)
        for _, row in df.iterrows()
    ]
    z_center, z_peak, peak_intensity = zip(*z_info, strict=True)
    df.insert(0, "z", z_center)
    df.insert(1, "y", df[y_col].astype(float))
    df.insert(2, "x", df[x_col].astype(float))
    df["z_peak"] = z_peak
    df["peak_intensity"] = peak_intensity
    df["projection_bg_mean"] = float(mean)
    df["projection_bg_median"] = float(median)
    df["projection_bg_std"] = float(std)
    df["fwhm"] = float(fwhm)
    df["threshold_sigma"] = float(threshold_sigma)
    return df


def write_bead_qc_png(stack: np.ndarray, beads: pd.DataFrame, output: Path) -> None:
    projection = stack.max(axis=0)
    vmax = float(np.percentile(projection, 99.9))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.imshow(projection, cmap="gray", vmax=vmax)
    if not beads.empty:
        ax.scatter(beads["x"], beads["y"], s=8, facecolors="none", edgecolors="lime", linewidths=0.4)
    ax.set_axis_off()
    ax.set_title(f"{len(beads)} beads")
    fig.tight_layout(pad=0)
    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
