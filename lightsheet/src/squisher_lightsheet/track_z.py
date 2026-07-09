from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np

from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as rough_legacy
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as stitch_legacy
from squisher_lightsheet import phase_metrics


DIMENSIONS = ("z", "y", "x")


@dataclass(frozen=True)
class ChannelMeasurement:
    tile: str
    side: str
    path: str
    tile_center_y_um: float
    tile_center_x_um: float
    moving_track: str
    moving_channel: int
    reference_channel: int
    dz_px: float | None
    dz_um: float | None
    corr_before: float | None
    corr_after: float | None
    n_voxels: int
    valid: bool
    reason: str | None


@dataclass(frozen=True)
class TrackMeasurement:
    tile: str
    side: str
    path: str
    tile_center_y_um: float
    tile_center_x_um: float
    moving_track: str
    moving_channels: tuple[int, ...]
    dz_px: float | None
    dz_um: float | None
    smoothed_dz_um: float | None
    residual_dz_um: float | None
    score: float | None
    valid: bool
    outlier: bool
    reason: str | None


def robust_normalize(image: np.ndarray) -> np.ndarray:
    return phase_metrics.robust_normalize(image)


def fixed_content_mask(fixed: np.ndarray, *, min_voxels: int) -> np.ndarray:
    positive = fixed[np.isfinite(fixed) & (fixed > 0)]
    if positive.size == 0:
        return np.zeros(fixed.shape, dtype=bool)
    threshold = float(np.percentile(positive, 70.0))
    mask = np.isfinite(fixed) & (fixed >= threshold)
    if int(mask.sum()) < min_voxels:
        threshold = float(np.percentile(positive, 50.0))
        mask = np.isfinite(fixed) & (fixed >= threshold)
    return mask


def z_shift_slices(z_size: int, shift_px: int) -> tuple[slice, slice]:
    if shift_px > 0:
        return slice(shift_px, z_size), slice(0, z_size - shift_px)
    if shift_px < 0:
        return slice(0, z_size + shift_px), slice(-shift_px, z_size)
    return slice(0, z_size), slice(0, z_size)


def corrcoef_masked(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float | None:
    return phase_metrics.corrcoef_on_mask(fixed, moving, mask)


def estimate_z_shift_px(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    max_shift_px: int = 8,
    min_voxels: int = 2048,
) -> tuple[float | None, dict[str, Any]]:
    if fixed.shape != moving.shape:
        raise ValueError(f"fixed shape {fixed.shape} does not match moving shape {moving.shape}")
    if fixed.ndim != 3:
        raise ValueError(f"Expected 3D z/y/x volumes, got {fixed.ndim}D")
    if max_shift_px < 0:
        raise ValueError("max_shift_px must be non-negative")

    fixed_norm = robust_normalize(fixed)
    moving_norm = robust_normalize(moving)
    fixed_mask = fixed_content_mask(fixed_norm, min_voxels=min_voxels)
    if int(fixed_mask.sum()) < min_voxels:
        return None, {
            "corr_before": None,
            "corr_after": None,
            "n_voxels": int(fixed_mask.sum()),
            "reason": "insufficient_fixed_content",
        }

    z_size = int(fixed.shape[0])
    candidates = list(range(-min(max_shift_px, z_size - 1), min(max_shift_px, z_size - 1) + 1))
    scores: dict[int, float | None] = {}
    voxel_counts: dict[int, int] = {}
    for shift in candidates:
        fixed_slice, moving_slice = z_shift_slices(z_size, shift)
        mask = fixed_mask[fixed_slice]
        score = corrcoef_masked(fixed_norm[fixed_slice], moving_norm[moving_slice], mask)
        scores[shift] = score
        voxel_counts[shift] = int(mask.sum())

    finite_scores = {shift: score for shift, score in scores.items() if score is not None and np.isfinite(score)}
    if not finite_scores:
        return None, {
            "corr_before": scores.get(0),
            "corr_after": None,
            "n_voxels": max(voxel_counts.values(), default=0),
            "reason": "no_finite_correlation",
        }

    best_shift = max(finite_scores, key=lambda shift: finite_scores[shift])
    best_score = float(finite_scores[best_shift])
    subpixel_shift = float(best_shift)
    left = scores.get(best_shift - 1)
    center = scores.get(best_shift)
    right = scores.get(best_shift + 1)
    if left is not None and center is not None and right is not None:
        denom = float(left - 2.0 * center + right)
        if denom != 0.0:
            delta = 0.5 * float(left - right) / denom
            if np.isfinite(delta):
                subpixel_shift += float(np.clip(delta, -1.0, 1.0))

    return subpixel_shift, {
        "corr_before": scores.get(0),
        "corr_after": best_score,
        "n_voxels": int(voxel_counts.get(best_shift, 0)),
        "reason": None,
        "integer_shift_px": int(best_shift),
    }


def tile_center_yx_um(tile: rough_legacy.TileRecord) -> tuple[float, float]:
    start, stop = rough_legacy.tile_bounds_zyx_um(tile)
    center = (start + stop) / 2.0
    return float(center[1]), float(center[2])


def channel_groups(position_input: Path, reference_channel: int) -> tuple[dict[str, tuple[int, ...]], str]:
    tiles = stitch_legacy.read_position_input_tiles(position_input)
    groups: dict[str, tuple[int, ...]] = {}
    reference_track = None
    for track in tiles[0].tracks:
        groups[track.slug] = tuple(int(channel) for channel in track.channels)
        if reference_channel in track.channels:
            reference_track = track.slug
    if reference_track is None:
        raise ValueError(f"Reference channel {reference_channel} was not found in detected tracks")
    return groups, reference_track


def aggregate_track_measurements(
    measurements: list[ChannelMeasurement],
    *,
    track_channels: dict[str, tuple[int, ...]],
    level_spacing_z_um: float,
    min_score: float,
) -> list[TrackMeasurement]:
    by_tile_track: dict[tuple[str, str], list[ChannelMeasurement]] = {}
    for measurement in measurements:
        by_tile_track.setdefault((measurement.tile, measurement.moving_track), []).append(measurement)

    track_measurements = []
    for (tile, track), items in sorted(by_tile_track.items()):
        first = items[0]
        valid = [
            item
            for item in items
            if item.valid
            and item.dz_px is not None
            and item.dz_um is not None
            and item.corr_after is not None
            and item.corr_after >= min_score
        ]
        if valid:
            dz_px = float(np.median([item.dz_px for item in valid if item.dz_px is not None]))
            dz_um = dz_px * level_spacing_z_um
            score = float(np.median([item.corr_after for item in valid if item.corr_after is not None]))
            reason = None
            is_valid = True
        else:
            dz_px = None
            dz_um = None
            score = None
            reason = "no_valid_channel_measurements"
            is_valid = False
        track_measurements.append(
            TrackMeasurement(
                tile=tile,
                side=first.side,
                path=first.path,
                tile_center_y_um=first.tile_center_y_um,
                tile_center_x_um=first.tile_center_x_um,
                moving_track=track,
                moving_channels=track_channels[track],
                dz_px=dz_px,
                dz_um=dz_um,
                smoothed_dz_um=None,
                residual_dz_um=None,
                score=score,
                valid=is_valid,
                outlier=False,
                reason=reason,
            )
        )
    return track_measurements


def median_nearest_neighbor_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    distances = []
    for index, point in enumerate(points):
        delta = points[np.arange(len(points)) != index] - point
        distances.append(float(np.sqrt(np.sum(delta * delta, axis=1)).min()))
    return max(float(np.median(distances)), 1.0)


def smooth_track_measurements(
    measurements: list[TrackMeasurement],
    *,
    smooth_sigma_tiles: float,
    outlier_mad: float,
) -> list[TrackMeasurement]:
    by_track: dict[str, list[int]] = {}
    for index, measurement in enumerate(measurements):
        by_track.setdefault(measurement.moving_track, []).append(index)

    updated = list(measurements)
    for indices in by_track.values():
        valid_indices = [index for index in indices if measurements[index].valid and measurements[index].dz_um is not None]
        if not valid_indices:
            continue
        values = np.asarray([measurements[index].dz_um for index in valid_indices], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = outlier_mad * max(1.4826 * mad, 1e-6)
        inlier_indices = [
            index
            for index in valid_indices
            if abs(float(measurements[index].dz_um) - median) <= threshold
        ]
        if not inlier_indices:
            inlier_indices = valid_indices
        points = np.asarray(
            [[measurements[index].tile_center_y_um, measurements[index].tile_center_x_um] for index in inlier_indices],
            dtype=np.float64,
        )
        inlier_values = np.asarray([measurements[index].dz_um for index in inlier_indices], dtype=np.float64)
        scores = np.asarray([measurements[index].score or 0.0 for index in inlier_indices], dtype=np.float64)
        sigma_um = smooth_sigma_tiles * median_nearest_neighbor_distance(points)
        for index in indices:
            measurement = measurements[index]
            target = np.asarray([measurement.tile_center_y_um, measurement.tile_center_x_um], dtype=np.float64)
            distances2 = np.sum((points - target) ** 2, axis=1)
            weights = np.exp(-0.5 * distances2 / max(sigma_um * sigma_um, 1.0)) * np.maximum(scores, 0.05)
            if float(weights.sum()) == 0.0:
                smoothed = float(np.mean(inlier_values))
            else:
                smoothed = float(np.sum(weights * inlier_values) / np.sum(weights))
            raw = measurement.dz_um
            outlier = index in valid_indices and index not in inlier_indices
            valid = measurement.valid and not outlier
            reason = "outlier" if outlier else measurement.reason
            residual = None if raw is None else float(raw - smoothed)
            updated[index] = TrackMeasurement(
                **{
                    **measurement.__dict__,
                    "smoothed_dz_um": smoothed,
                    "residual_dz_um": residual,
                    "valid": valid,
                    "outlier": outlier,
                    "reason": reason,
                }
            )
    return updated


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def measurement_to_row(measurement: ChannelMeasurement | TrackMeasurement) -> dict[str, Any]:
    row = dict(measurement.__dict__)
    if isinstance(measurement, TrackMeasurement):
        row["moving_channels"] = " ".join(str(channel) for channel in measurement.moving_channels)
    return row


def write_track_heatmaps(output_dir: Path, measurements: list[TrackMeasurement]) -> dict[str, dict[str, str]]:
    import matplotlib.pyplot as plt

    outputs: dict[str, dict[str, str]] = {}
    by_track: dict[str, list[TrackMeasurement]] = {}
    for measurement in measurements:
        by_track.setdefault(measurement.moving_track, []).append(measurement)

    for track, track_measurements in sorted(by_track.items()):
        outputs[track] = {}
        fields = {
            "raw_dz_um": "dz_um",
            "smoothed_dz_um": "smoothed_dz_um",
            "residual_dz_um": "residual_dz_um",
            "score": "score",
        }
        for label, field in fields.items():
            xs = np.asarray([item.tile_center_x_um for item in track_measurements], dtype=np.float64)
            ys = np.asarray([item.tile_center_y_um for item in track_measurements], dtype=np.float64)
            values = np.asarray(
                [np.nan if getattr(item, field) is None else getattr(item, field) for item in track_measurements],
                dtype=np.float64,
            )
            fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
            scatter = ax.scatter(xs, ys, c=values, s=80, cmap="coolwarm")
            ax.set_title(f"{track} {label}")
            ax.set_xlabel("x um")
            ax.set_ylabel("y um")
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(scatter, ax=ax, label=label)
            path = output_dir / f"{track}_{label}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs[track][label] = str(path.resolve())
    return outputs


def write_representative_overlays(
    output_dir: Path,
    *,
    tiles: list[rough_legacy.TileRecord],
    channel_measurements: list[ChannelMeasurement],
    reference_channel: int,
    level_factor: int,
) -> dict[str, str]:
    from PIL import Image, ImageDraw
    from scipy import ndimage

    by_track: dict[str, list[ChannelMeasurement]] = {}
    for measurement in channel_measurements:
        if measurement.valid and measurement.dz_px is not None and measurement.corr_after is not None:
            by_track.setdefault(measurement.moving_track, []).append(measurement)
    tile_by_name = {tile.tile: tile for tile in tiles}
    outputs = {}
    for track, measurements in sorted(by_track.items()):
        best = max(measurements, key=lambda item: item.corr_after or -np.inf)
        tile = tile_by_name[best.tile]
        fixed = rough_legacy.sampled_tile_volume(tile, channel=reference_channel, level_factor=level_factor)
        moving = rough_legacy.sampled_tile_volume(tile, channel=best.moving_channel, level_factor=level_factor)
        shifted = ndimage.shift(
            moving,
            shift=(float(best.dz_px), 0.0, 0.0),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        z_index = int(fixed.shape[0] // 2)
        before = output_dir / f"{track}_representative_before.png"
        after = output_dir / f"{track}_representative_after.png"
        rough_legacy.write_overlay_scaled(before, left=fixed[z_index], right=moving[z_index])
        rough_legacy.write_overlay_scaled(after, left=fixed[z_index], right=shifted[z_index])

        before_image = Image.open(before).convert("RGB")
        after_image = Image.open(after).convert("RGB")
        title_h = 24
        sheet = Image.new("RGB", (before_image.width + after_image.width, before_image.height + title_h), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), "before", fill=(0, 0, 0))
        draw.text((before_image.width + 8, 6), "after z-only", fill=(0, 0, 0))
        sheet.paste(before_image, (0, title_h))
        sheet.paste(after_image, (before_image.width, title_h))
        path = output_dir / f"{track}_representative_overlay.png"
        sheet.save(path)
        outputs[track] = str(path.resolve())
    return outputs


def run_track_z_diagnostics(
    *,
    position_input: Path,
    output_dir: Path,
    reference_channel: int = 3,
    level: int = 4,
    max_z_shift_px: int = 8,
    min_score: float = 0.05,
    min_voxels: int = 2048,
    smooth_sigma_tiles: float = 1.5,
    outlier_mad: float = 4.0,
) -> Path:
    if max_z_shift_px < 0:
        raise ValueError("max_z_shift_px must be non-negative")
    if min_voxels < 1:
        raise ValueError("min_voxels must be positive")
    if smooth_sigma_tiles <= 0.0:
        raise ValueError("smooth_sigma_tiles must be positive")
    if outlier_mad <= 0.0:
        raise ValueError("outlier_mad must be positive")

    payload = json.loads(position_input.read_text())
    tiles = rough_legacy.load_tiles(payload)
    if not tiles:
        raise ValueError("position input contains no tiles")
    track_channels, reference_track = channel_groups(position_input, reference_channel)
    moving_tracks = {
        track: channels
        for track, channels in track_channels.items()
        if reference_channel not in channels
    }
    if not moving_tracks:
        raise ValueError("No non-reference tracks found")

    geometry = rough_legacy.build_geometry(tiles, level=level)
    output_dir.mkdir(parents=True, exist_ok=True)
    level_factor = int(geometry.level_factor)
    channel_measurements: list[ChannelMeasurement] = []
    for tile in tiles:
        fixed = rough_legacy.sampled_tile_volume(tile, channel=reference_channel, level_factor=level_factor)
        center_y_um, center_x_um = tile_center_yx_um(tile)
        for track, channels in sorted(moving_tracks.items()):
            for channel in channels:
                moving = rough_legacy.sampled_tile_volume(tile, channel=channel, level_factor=level_factor)
                dz_px, details = estimate_z_shift_px(
                    fixed,
                    moving,
                    max_shift_px=max_z_shift_px,
                    min_voxels=min_voxels,
                )
                valid = dz_px is not None and details["corr_after"] is not None
                reason = details["reason"]
                dz_um = None if dz_px is None else float(dz_px * geometry.level_spacing_zyx_um[0])
                channel_measurements.append(
                    ChannelMeasurement(
                        tile=tile.tile,
                        side=tile.side,
                        path=str(tile.path),
                        tile_center_y_um=center_y_um,
                        tile_center_x_um=center_x_um,
                        moving_track=track,
                        moving_channel=int(channel),
                        reference_channel=int(reference_channel),
                        dz_px=None if dz_px is None else float(dz_px),
                        dz_um=dz_um,
                        corr_before=details["corr_before"],
                        corr_after=details["corr_after"],
                        n_voxels=int(details["n_voxels"]),
                        valid=bool(valid),
                        reason=reason,
                    )
                )
                logger.info(
                    "track-z {} track={} channel={} dz_px={} score={}",
                    tile.tile,
                    track,
                    channel,
                    dz_px,
                    details["corr_after"],
                )

    track_measurements = aggregate_track_measurements(
        channel_measurements,
        track_channels=track_channels,
        level_spacing_z_um=float(geometry.level_spacing_zyx_um[0]),
        min_score=min_score,
    )
    track_measurements = smooth_track_measurements(
        track_measurements,
        smooth_sigma_tiles=smooth_sigma_tiles,
        outlier_mad=outlier_mad,
    )

    channel_csv = output_dir / "track_z_channel_measurements.csv"
    track_csv = output_dir / "track_z_measurements.csv"
    write_csv(
        channel_csv,
        [measurement_to_row(item) for item in channel_measurements],
        list(ChannelMeasurement.__dataclass_fields__),
    )
    write_csv(
        track_csv,
        [measurement_to_row(item) for item in track_measurements],
        list(TrackMeasurement.__dataclass_fields__),
    )
    heatmaps = write_track_heatmaps(output_dir, track_measurements)
    overlays = write_representative_overlays(
        output_dir,
        tiles=tiles,
        channel_measurements=channel_measurements,
        reference_channel=reference_channel,
        level_factor=level_factor,
    )

    summary_path = output_dir / "track_z_diagnostics.json"
    summary = {
        "schema_version": 1,
        "artifact_type": "lightsheet.track_z_diagnostics.v1",
        "position_input": str(position_input.resolve()),
        "reference_channel": int(reference_channel),
        "reference_track": reference_track,
        "track_channels": {track: list(channels) for track, channels in track_channels.items()},
        "moving_tracks": {track: list(channels) for track, channels in moving_tracks.items()},
        "level": int(level),
        "level_factor": level_factor,
        "level_spacing_zyx_um": geometry.level_spacing_zyx_um.tolist(),
        "settings": {
            "max_z_shift_px": int(max_z_shift_px),
            "min_score": float(min_score),
            "min_voxels": int(min_voxels),
            "smooth_sigma_tiles": float(smooth_sigma_tiles),
            "outlier_mad": float(outlier_mad),
        },
        "outputs": {
            "channel_measurements_csv": str(channel_csv.resolve()),
            "track_measurements_csv": str(track_csv.resolve()),
            "heatmaps": heatmaps,
            "representative_overlays": overlays,
        },
        "track_summary": summarize_tracks(track_measurements),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path.resolve()


def summarize_tracks(measurements: list[TrackMeasurement]) -> dict[str, dict[str, Any]]:
    by_track: dict[str, list[TrackMeasurement]] = {}
    for measurement in measurements:
        by_track.setdefault(measurement.moving_track, []).append(measurement)
    summary = {}
    for track, items in sorted(by_track.items()):
        valid = [item for item in items if item.valid and item.dz_um is not None]
        smoothed = [item.smoothed_dz_um for item in items if item.smoothed_dz_um is not None]
        summary[track] = {
            "tiles": len(items),
            "valid_tiles": len(valid),
            "outliers": sum(1 for item in items if item.outlier),
            "raw_dz_um_median": None if not valid else float(np.median([item.dz_um for item in valid])),
            "smoothed_dz_um_min": None if not smoothed else float(np.min(smoothed)),
            "smoothed_dz_um_max": None if not smoothed else float(np.max(smoothed)),
        }
    return summary
