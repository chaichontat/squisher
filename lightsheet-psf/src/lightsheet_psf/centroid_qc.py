from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_padded_crop(
    stack: np.ndarray, zc: int, yc: int, xc: int, crop_shape: tuple[int, int, int]
) -> np.ndarray:
    rz, ry, rx = (s // 2 for s in crop_shape)
    z_size, y_size, x_size = stack.shape
    crop = np.full(crop_shape, np.nan, dtype=np.float32)
    z0s, z1s = max(0, zc - rz), min(z_size, zc + rz + 1)
    y0s, y1s = max(0, yc - ry), min(y_size, yc + ry + 1)
    x0s, x1s = max(0, xc - rx), min(x_size, xc + rx + 1)
    z0d = z0s - (zc - rz)
    y0d = y0s - (yc - ry)
    x0d = x0s - (xc - rx)
    crop[z0d : z0d + (z1s - z0s), y0d : y0d + (y1s - y0s), x0d : x0d + (x1s - x0s)] = stack[
        z0s:z1s,
        y0s:y1s,
        x0s:x1s,
    ]
    return crop


def add_pick(selected: list[pd.Series], used_ids: set[int], label: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    row = frame.iloc[0].copy()
    rid = int(row["id"])
    if rid in used_ids:
        return
    row["label"] = label
    used_ids.add(rid)
    selected.append(row)


def choose_spots(
    df: pd.DataFrame, crop_shape: tuple[int, int, int], stack_shape: tuple[int, int, int]
) -> pd.DataFrame:
    rz, ry, rx = (s // 2 for s in crop_shape)
    z_size, y_size, x_size = stack_shape
    good = df[df["good_quality"]].copy()
    full = good[good["full_crop"]].copy()
    edge = good[~good["full_crop"]].copy()
    selected: list[pd.Series] = []
    used_ids: set[int] = set()
    add_pick(selected, used_ids, "full brightest", full.sort_values("peak_intensity", ascending=False))
    median_z = full.loc[(full["z"] - full["z"].median()).abs().sort_values().index]
    add_pick(selected, used_ids, "full median-z", median_z.sort_values("peak_intensity", ascending=False))
    add_pick(selected, used_ids, "full dimmer", full.sort_values("peak_intensity", ascending=True))
    add_pick(
        selected,
        used_ids,
        "edge z-low",
        edge[edge["z_round"] < rz].sort_values("peak_intensity", ascending=False),
    )
    add_pick(
        selected,
        used_ids,
        "edge z-high",
        edge[edge["z_round"] >= z_size - rz].sort_values("peak_intensity", ascending=False),
    )
    add_pick(
        selected,
        used_ids,
        "edge y-low",
        edge[edge["y_round"] < ry].sort_values("peak_intensity", ascending=False),
    )
    add_pick(
        selected,
        used_ids,
        "edge x-low",
        edge[edge["x_round"] < rx].sort_values("peak_intensity", ascending=False),
    )
    add_pick(
        selected,
        used_ids,
        "edge x-high",
        edge[edge["x_round"] >= x_size - rx].sort_values("peak_intensity", ascending=False),
    )
    if len(selected) < 8:
        for _, row in good.sort_values("peak_intensity", ascending=False).iterrows():
            rid = int(row["id"])
            if rid in used_ids:
                continue
            row = row.copy()
            row["label"] = "fallback"
            used_ids.add(rid)
            selected.append(row)
            if len(selected) >= 8:
                break
    return pd.DataFrame(selected)


def render_sheet(
    stack: np.ndarray, selected: pd.DataFrame, crop_shape: tuple[int, int, int], output_png: Path
) -> None:
    rz, ry, rx = (s // 2 for s in crop_shape)
    fig, axes = plt.subplots(
        len(selected), 3, figsize=(10.5, 2.8 * len(selected)), dpi=200, constrained_layout=True
    )
    if len(selected) == 1:
        axes = np.array([axes])
    for i, row in enumerate(selected.itertuples(index=False)):
        crop = extract_padded_crop(stack, int(row.z_round), int(row.y_round), int(row.x_round), crop_shape)
        xy = np.nanmax(crop, axis=0)
        zy = np.nanmax(crop, axis=2)
        zx = np.nanmax(crop, axis=1)
        finite = crop[np.isfinite(crop)]
        vmin = float(np.percentile(finite, 5)) if finite.size else 0.0
        vmax = float(np.percentile(finite, 99.7)) if finite.size else 1.0
        cx = rx + (float(row.x) - float(row.x_round))
        cy = ry + (float(row.y) - float(row.y_round))
        cz = rz + (float(row.z) - float(row.z_round))
        for ax, img, (mx, my), title in (
            (axes[i, 0], xy, (cx, cy), "XY max"),
            (axes[i, 1], zy, (cy, cz), "ZY max"),
            (axes[i, 2], zx, (cx, cz), "ZX max"),
        ):
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
            ax.axvline(mx, color="cyan", linewidth=0.8)
            ax.axhline(my, color="cyan", linewidth=0.8)
            ax.plot(mx, my, marker="+", color="red", markersize=6, markeredgewidth=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
        dz = float(row.z) - float(row.z_round)
        dy = float(row.y) - float(row.y_round)
        dx = float(row.x) - float(row.x_round)
        axes[i, 0].set_ylabel(
            f"id={int(row.id)} {row.label} full={bool(row.full_crop)} peak={float(row.peak_intensity):.0f}\n"
            f"zyx=({float(row.z):.2f},{float(row.y):.2f},{float(row.x):.2f}) "
            f"d=({dz:+.2f},{dy:+.2f},{dx:+.2f})",
            fontsize=7,
            rotation=0,
            labelpad=75,
            va="center",
        )
    fig.suptitle("Centroid QC, cyan lines/red plus = fitted centroid", fontsize=11)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
