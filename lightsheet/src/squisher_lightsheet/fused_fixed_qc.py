"""PNG quality-control renders for completed fused-fixed recovery artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, map_coordinates
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
import zarr

from squisher_lightsheet.artifact_io import read_json_object
from squisher_lightsheet.fused_fixed import (
    LINEAR_MAPPING_RECOVERY_ATTEMPT,
    SPATIAL_RECOVERY_ATTEMPT,
)
from squisher_lightsheet.global_phase import masked_highpass
from squisher_lightsheet.qc import load_qc_font


LEVEL = 2
FIXED_LEVEL_FACTOR = 4
MOVING_XY_FACTOR = 4
WINDOW = 528
INTERPOLATED_ATTEMPTS_BY_METHOD = {
    "method6": (SPATIAL_RECOVERY_ATTEMPT, LINEAR_MAPPING_RECOVERY_ATTEMPT),
    "method10-mattes": (
        "interpolated_method10_mattes_spatial_recovery",
        "interpolated_method10_mattes_linear_mapping_recovery",
    ),
}
IMAGE_SCORE_PLANE_OFFSETS_L0 = (-WINDOW // 4, 0, WINDOW // 4)
IMAGE_SCORE_HIGHPASS_SIGMA_PX = 4.0
IMAGE_SCORE_MINIMUM_PIXELS = 1000
# Residual-shift confidence, calibrated in the prod3-2 window inspection against a
# 180-degree-rotated null (0.15% pass) and native controls (97.9% pass).
RESIDUAL_MAX_SHIFT_PX = 16
RESIDUAL_PEAK_EXCLUSION_PX = 3
RESIDUAL_MIN_NCC = 0.20
RESIDUAL_MIN_PEAK_MARGIN = 0.10
DEFAULT_INTERPOLATED_QC_COLUMNS = 14
THUMBNAIL_SIZE = 184
CELL_WIDTH = 220
CELL_HEIGHT = 238
HEADER_HEIGHT = 62
CUBE_CORNERS = np.asarray(
    [
        [z, y, x]
        for z in (0.0, WINDOW - 1.0)
        for y in (0.0, WINDOW - 1.0)
        for x in (0.0, WINDOW - 1.0)
    ]
)
CUBE_EDGES = tuple(
    (first, second)
    for first in range(8)
    for second in range(first + 1, 8)
    if np.count_nonzero(CUBE_CORNERS[first] != CUBE_CORNERS[second]) == 1
)


def _stretch(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0]
    if not positive.size:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(positive, (1.0, 99.7))
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(
        np.uint8
    )


def _validate_summary_geometry(summary: dict[str, Any], source: Path) -> None:
    shape = summary.get("window_shape_zyx")
    if not isinstance(shape, list | tuple) or list(shape) != [WINDOW, WINDOW, WINDOW]:
        raise ValueError(
            f"{source} uses unsupported window_shape_zyx {shape}; expected "
            f"{[WINDOW, WINDOW, WINDOW]}"
        )


def plane_support_bounds(
    fixed_start: np.ndarray,
    matrix: np.ndarray,
    translation: np.ndarray,
    fixed_z_l0: int,
    canvas_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    center = (WINDOW - 1.0) / 2.0
    transformed = (
        (matrix @ (CUBE_CORNERS - center).T).T
        + center
        + translation
        + fixed_start
    )
    points: list[np.ndarray] = []
    for first, second in CUBE_EDGES:
        start = transformed[first]
        stop = transformed[second]
        if start[0] == stop[0]:
            if np.isclose(start[0], fixed_z_l0):
                points.extend((start, stop))
            continue
        fraction = (fixed_z_l0 - start[0]) / (stop[0] - start[0])
        if 0.0 <= fraction <= 1.0:
            points.append(start + fraction * (stop - start))
    if not points:
        return None

    section = np.asarray(points)
    y0 = max(0, int(np.floor(np.min(section[:, 1]) / MOVING_XY_FACTOR)))
    x0 = max(0, int(np.floor(np.min(section[:, 2]) / MOVING_XY_FACTOR)))
    y1 = min(
        canvas_shape[0],
        int(np.floor(np.max(section[:, 1]) / MOVING_XY_FACTOR)) + 1,
    )
    x1 = min(
        canvas_shape[1],
        int(np.floor(np.max(section[:, 2]) / MOVING_XY_FACTOR)) + 1,
    )
    return None if y1 <= y0 or x1 <= x0 else (y0, y1, x0, x1)


def _selected_transform(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(row["selected_local_matrix_zyx"], dtype=np.float64),
        np.asarray(row["selected_local_translation_zyx"], dtype=np.float64),
    )


def _sample_moving_plane(
    *,
    row: dict[str, Any],
    matrix: np.ndarray,
    translation: np.ndarray,
    fixed_z_l0: int,
    bounds: tuple[int, int, int, int],
    arrays: dict[str, zarr.Array],
) -> tuple[np.ndarray, np.ndarray]:
    fixed_start = np.asarray(row["fixed_start_zyx"], dtype=np.float64)
    y0, y1, x0, x1 = bounds
    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64) * MOVING_XY_FACTOR,
        np.arange(x0, x1, dtype=np.float64) * MOVING_XY_FACTOR,
        indexing="ij",
    )
    fixed_local = np.stack(
        [
            np.full(yy.shape, fixed_z_l0 - fixed_start[0]),
            yy - fixed_start[1],
            xx - fixed_start[2],
        ]
    )
    center = (WINDOW - 1.0) / 2.0
    moving_local = (
        np.linalg.inv(matrix)
        @ (fixed_local.reshape(3, -1) - center - translation[:, None])
        + center
    )
    moving_global = moving_local + np.asarray(
        row["moving_start_l0_zyx"], dtype=np.float64
    )[:, None]
    path = str(row["moving_path"])
    if path not in arrays:
        arrays[path] = zarr.open_group(path, mode="r")[str(LEVEL)]
    source = arrays[path]
    coords = np.vstack(
        [
            moving_global[0],
            moving_global[1] / MOVING_XY_FACTOR,
            moving_global[2] / MOVING_XY_FACTOR,
        ]
    )
    valid = np.all(moving_local >= 0.0, axis=0) & np.all(
        moving_local <= WINDOW - 1.0, axis=0
    )
    valid &= np.all(coords >= 0.0, axis=0) & np.all(
        coords <= np.asarray(source.shape[-3:], dtype=np.float64)[:, None] - 1.0,
        axis=0,
    )
    sampled = np.zeros(coords.shape[1], dtype=np.float32)
    if np.any(valid):
        lo = np.maximum(
            np.floor(np.min(coords[:, valid], axis=1)).astype(int) - 1, 0
        )
        hi = np.minimum(
            np.ceil(np.max(coords[:, valid], axis=1)).astype(int) + 2,
            source.shape[-3:],
        )
        block = np.asarray(
            source[
                int(row.get("moving_channel", 0)),
                lo[0] : hi[0],
                lo[1] : hi[1],
                lo[2] : hi[2],
            ],
            dtype=np.float32,
        )
        sampled[valid] = map_coordinates(
            block,
            (coords - lo[:, None])[:, valid],
            order=1,
            mode="constant",
            cval=0.0,
        )
    return sampled.reshape(yy.shape), valid.reshape(yy.shape)


@dataclass(frozen=True)
class PlaneShift:
    """Best in-plane shift to apply to the moving plane to match fixed, in level-2 pixels."""

    shift_yx: tuple[float, float]
    peak_ncc: float
    second_peak_ncc: float
    at_search_edge: bool

    @property
    def confident(self) -> bool:
        return (
            not self.at_search_edge
            and self.peak_ncc >= RESIDUAL_MIN_NCC
            and self.peak_ncc - self.second_peak_ncc >= RESIDUAL_MIN_PEAK_MARGIN
        )


class WindowImageScorer:
    """Compare window transforms against the fixed volume on shared level-2 planes.

    Planes are placed at the window centre implied by the mean candidate z
    translation, so results do not depend on candidate order. Each plane uses
    pixels where the fixed channel shows tissue and every candidate has moving
    support, and scores zero-shift NCC of in-mask high-passed signal. Both planes
    are scored in the channel's Method 6 fit intensity space (``linear`` or
    ``log1p``); the tissue mask stays on raw fixed intensity.
    """

    def __init__(self, fixed_fused: Path, fixed_threshold: float, intensity_transform: str) -> None:
        if intensity_transform not in ("linear", "log1p"):
            raise ValueError(f"Unsupported fit intensity transform: {intensity_transform!r}")
        self._fixed = zarr.open_group(str(fixed_fused), mode="r")[str(LEVEL)]
        self._fixed_threshold = float(fixed_threshold)
        self._log1p = intensity_transform == "log1p"
        self._arrays: dict[str, zarr.Array] = {}

    def _planes(
        self, row: dict[str, Any], transforms: Sequence[np.ndarray]
    ) -> list[tuple[np.ndarray, list[np.ndarray], np.ndarray]]:
        fixed_start = np.asarray(row["fixed_start_zyx"], dtype=np.float64)
        candidates = [
            (transform[:9].reshape(3, 3), transform[9:]) for transform in transforms
        ]
        center_z = (
            fixed_start[0]
            + (WINDOW - 1.0) / 2.0
            + float(np.mean([translation[0] for _matrix, translation in candidates]))
        )
        planes = []
        for offset in IMAGE_SCORE_PLANE_OFFSETS_L0:
            fixed_z = int(np.rint((center_z + offset) / FIXED_LEVEL_FACTOR))
            if not 0 <= fixed_z < self._fixed.shape[-3]:
                continue
            fixed_z_l0 = fixed_z * FIXED_LEVEL_FACTOR
            candidate_bounds = [
                bounds
                for matrix, translation in candidates
                if (
                    bounds := plane_support_bounds(
                        fixed_start, matrix, translation, fixed_z_l0, self._fixed.shape[-2:]
                    )
                )
                is not None
            ]
            if len(candidate_bounds) != len(candidates):
                continue
            y0 = min(bounds[0] for bounds in candidate_bounds)
            y1 = max(bounds[1] for bounds in candidate_bounds)
            x0 = min(bounds[2] for bounds in candidate_bounds)
            x1 = max(bounds[3] for bounds in candidate_bounds)
            fixed_patch = np.asarray(self._fixed[fixed_z, y0:y1, x0:x1], dtype=np.float32)
            mask = fixed_patch > self._fixed_threshold
            moving_patches = []
            for matrix, translation in candidates:
                sampled, valid = _sample_moving_plane(
                    row=row,
                    matrix=matrix,
                    translation=translation,
                    fixed_z_l0=fixed_z_l0,
                    bounds=(y0, y1, x0, x1),
                    arrays=self._arrays,
                )
                moving_patches.append(sampled)
                mask &= valid
            if np.count_nonzero(mask) < IMAGE_SCORE_MINIMUM_PIXELS:
                continue
            if self._log1p:
                fixed_patch = np.log1p(np.maximum(fixed_patch, 0.0))
                moving_patches = [np.log1p(np.maximum(patch, 0.0)) for patch in moving_patches]
            planes.append((fixed_patch, moving_patches, mask))
        return planes

    def plane_scores(
        self, row: dict[str, Any], transforms: Sequence[np.ndarray]
    ) -> list[list[float]]:
        """Per-candidate zero-shift NCC on the same informative planes."""
        scores: list[list[float]] = [[] for _ in transforms]
        for fixed_patch, moving_patches, mask in self._planes(row, transforms):
            fixed_signal = masked_highpass(fixed_patch, mask, IMAGE_SCORE_HIGHPASS_SIGMA_PX)[mask]
            moving_signals = [
                masked_highpass(patch, mask, IMAGE_SCORE_HIGHPASS_SIGMA_PX)[mask]
                for patch in moving_patches
            ]
            if np.std(fixed_signal) == 0.0 or any(np.std(signal) == 0.0 for signal in moving_signals):
                continue
            for candidate_scores, signal in zip(scores, moving_signals, strict=True):
                candidate_scores.append(float(np.corrcoef(fixed_signal, signal)[0, 1]))
        return scores

    def residual_shifts(self, row: dict[str, Any], transform: np.ndarray) -> list[PlaneShift]:
        """Bounded masked-NCC search for residual in-plane misalignment of one transform."""
        shifts = []
        for fixed_patch, (moving_patch,), mask in self._planes(row, [transform]):
            fixed_signal = masked_highpass(fixed_patch, mask, IMAGE_SCORE_HIGHPASS_SIGMA_PX)
            moving_signal = masked_highpass(moving_patch, mask, IMAGE_SCORE_HIGHPASS_SIGMA_PX)
            xcorr = cross_correlate_masked(
                moving_signal, fixed_signal, mask, mask, mode="full", overlap_ratio=0.3
            )
            origin = np.asarray(mask.shape) - 1
            bound = RESIDUAL_MAX_SHIFT_PX
            window = np.nan_to_num(
                xcorr[
                    origin[0] - bound : origin[0] + bound + 1,
                    origin[1] - bound : origin[1] + bound + 1,
                ],
                nan=-1.0,
            )
            peak_y, peak_x = np.unravel_index(int(np.argmax(window)), window.shape)
            peak = float(window[peak_y, peak_x])
            excluded = window.copy()
            radius = RESIDUAL_PEAK_EXCLUSION_PX
            excluded[
                max(0, peak_y - radius) : peak_y + radius + 1,
                max(0, peak_x - radius) : peak_x + radius + 1,
            ] = -1.0
            shift = (float(bound - peak_y), float(bound - peak_x))
            shifts.append(
                PlaneShift(
                    shift_yx=shift,
                    peak_ncc=peak,
                    second_peak_ncc=float(np.max(excluded)),
                    at_search_edge=bool(max(abs(shift[0]), abs(shift[1])) == bound),
                )
            )
        return shifts


def _render_plane(
    rows: list[dict[str, Any]], fixed_z_l0: int, canvas_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, int]:
    canvas = np.zeros(canvas_shape, dtype=np.float32)
    coverage = np.zeros(canvas_shape, dtype=np.uint16)
    contributors = 0
    arrays: dict[str, zarr.Array] = {}
    for row in rows:
        matrix, translation = _selected_transform(row)
        bounds = plane_support_bounds(
            np.asarray(row["fixed_start_zyx"], dtype=np.float64),
            matrix,
            translation,
            fixed_z_l0,
            canvas_shape,
        )
        if bounds is None:
            continue
        y0, y1, x0, x1 = bounds
        sampled, valid = _sample_moving_plane(
            row=row,
            matrix=matrix,
            translation=translation,
            fixed_z_l0=fixed_z_l0,
            bounds=bounds,
            arrays=arrays,
        )
        np.maximum(
            canvas[y0:y1, x0:x1], sampled, out=canvas[y0:y1, x0:x1]
        )
        coverage[y0:y1, x0:x1] += valid
        contributors += 1
    return canvas, coverage, contributors


def _annotated(image: Image.Image, title: str) -> Image.Image:
    banner = 42
    output = Image.new("RGB", (image.width, image.height + banner), "black")
    output.paste(image.convert("RGB"), (0, banner))
    ImageDraw.Draw(output).text((12, 12), title, fill="white")
    return output


def _hole_overlay(
    fixed_plane: np.ndarray, tissue: np.ndarray, coverage: np.ndarray
) -> Image.Image:
    gray = _stretch(fixed_plane)
    rgb = np.repeat((gray // 3)[..., None], 3, axis=2)
    covered = tissue & (coverage > 0)
    missing = tissue & (coverage == 0)
    rgb[covered] = np.stack(
        [gray[covered] // 4, np.maximum(gray[covered], 100), gray[covered] // 4],
        axis=1,
    )
    rgb[missing] = np.stack(
        [
            np.full(np.count_nonzero(missing), 255, dtype=np.uint8),
            gray[missing] // 5,
            gray[missing] // 5,
        ],
        axis=1,
    )
    return Image.fromarray(rgb)


def render_prefusion_qc(
    *,
    summaries: Sequence[tuple[str, Path]],
    fixed_volume: Path,
    fixed_threshold: float,
    output_dir: Path,
) -> Path:
    """Render middle-plane dumb stitches and geometric tissue-hole maps."""
    if not summaries:
        raise ValueError("at least one recovery summary is required")
    labels = [label_name for label_name, _path in summaries]
    if len(labels) != len(set(labels)):
        raise ValueError(f"summary labels must be unique: {labels}")
    fixed_group = zarr.open_group(fixed_volume, mode="r")
    fixed = fixed_group[str(LEVEL)]
    fixed_z = fixed.shape[-3] // 2
    fixed_z_l0 = fixed_z * FIXED_LEVEL_FACTOR
    fixed_plane = np.asarray(fixed[fixed_z], dtype=np.float32)
    tissue = fixed_plane >= fixed_threshold
    tissue_pixels = int(np.count_nonzero(tissue))
    if not tissue_pixels:
        raise ValueError("fixed-plane tissue mask is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    report_rows = []
    combined_coverage = np.zeros(fixed_plane.shape, dtype=np.uint32)
    combined_contributors = 0
    for label_name, summary_path in summaries:
        summary = read_json_object(summary_path)
        _validate_summary_geometry(summary, summary_path)
        if Path(summary["fixed_fused"]).resolve() != fixed_volume.resolve():
            raise ValueError(f"fixed volume mismatch in {summary_path}")
        rows = [
            read_json_object(Path(window["level0_json"]))
            for window in summary["windows"]
            if window.get("status") == "accepted"
        ]
        started = time.perf_counter()
        canvas, coverage, contributors = _render_plane(
            rows, fixed_z_l0, fixed_plane.shape
        )
        combined_coverage += coverage
        combined_contributors += contributors
        missing = tissue & (coverage == 0)
        missing_pixels = int(np.count_nonzero(missing))
        components, count = label(missing)
        sizes = (
            np.bincount(components.ravel())[1:]
            if count
            else np.asarray([], dtype=int)
        )
        largest = int(sizes.max(initial=0))
        missing_percent = 100.0 * missing_pixels / tissue_pixels
        intensity_panel = _annotated(
            Image.fromarray(_stretch(canvas)),
            f"{label_name} dumb stitch | fixed L2 z={fixed_z} | "
            f"windows={contributors}/{len(rows)}",
        )
        coverage_panel = _annotated(
            _hole_overlay(fixed_plane, tissue, coverage),
            f"{label_name} coverage | uncovered tissue={missing_percent:.3f}% | "
            f"largest hole={largest:,} px (red)",
        )
        intensity_panel.save(output_dir / f"prefusion-middle-z-{label_name}.png")
        coverage_panel.save(output_dir / f"prefusion-hole-map-{label_name}.png")
        panels.extend((intensity_panel, coverage_panel))
        report_rows.append(
            {
                "label": label_name,
                "summary": str(summary_path.resolve()),
                "uncovered_tissue_percent": missing_percent,
                "uncovered_tissue_pixels": missing_pixels,
                "tissue_pixels": tissue_pixels,
                "largest_hole_pixels": largest,
                "contributors": contributors,
                "accepted_windows": len(rows),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    combined_report = None
    if len(summaries) > 1:
        missing = tissue & (combined_coverage == 0)
        missing_pixels = int(np.count_nonzero(missing))
        components, count = label(missing)
        sizes = (
            np.bincount(components.ravel())[1:]
            if count
            else np.asarray([], dtype=int)
        )
        largest = int(sizes.max(initial=0))
        missing_percent = 100.0 * missing_pixels / tissue_pixels
        combined_panel = _annotated(
            _hole_overlay(fixed_plane, tissue, combined_coverage),
            f"combined coverage | uncovered tissue={missing_percent:.3f}% | "
            f"largest hole={largest:,} px (red)",
        )
        combined_panel.save(output_dir / "prefusion-hole-map-combined.png")
        panels.append(combined_panel)
        combined_report = {
            "labels": labels,
            "uncovered_tissue_percent": missing_percent,
            "uncovered_tissue_pixels": missing_pixels,
            "tissue_pixels": tissue_pixels,
            "largest_hole_pixels": largest,
            "contributors": combined_contributors,
        }

    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    thumb_width = width // 2
    thumb_height = height // 2
    sheet = Image.new(
        "RGB", (thumb_width * 2, thumb_height * ((len(panels) + 1) // 2)), "black"
    )
    for index, panel in enumerate(panels):
        panel.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(
            panel,
            ((index % 2) * thumb_width, (index // 2) * thumb_height),
        )
    output = output_dir / "prefusion-hole-review.png"
    sheet.save(output)
    (output_dir / "prefusion-hole-review.json").write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.fused_fixed_prefusion_qc.v1",
                "fixed_volume": str(fixed_volume.resolve()),
                "fixed_threshold": fixed_threshold,
                "level": LEVEL,
                "fixed_z_index": fixed_z,
                "channels": report_rows,
                "combined": combined_report,
                "contact_sheet": str(output.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    return output.resolve()


def _render_interpolated_window(
    row: dict[str, Any],
    fixed: zarr.Array,
    moving_arrays: dict[str, zarr.Array],
) -> tuple[Image.Image, int]:
    fixed_start = np.asarray(row["fixed_start_zyx"], dtype=np.float64)
    matrix, translation = _selected_transform(row)
    center = (WINDOW - 1.0) / 2.0
    fixed_z = int(
        np.clip(
            np.rint(
                (fixed_start[0] + center + translation[0]) / FIXED_LEVEL_FACTOR
            ),
            0,
            fixed.shape[-3] - 1,
        )
    )
    fixed_z_l0 = fixed_z * FIXED_LEVEL_FACTOR
    bounds = plane_support_bounds(
        fixed_start, matrix, translation, fixed_z_l0, fixed.shape[-2:]
    )
    if bounds is None:
        raise ValueError(
            "interpolated window has no support at its center plane: "
            f"{row['moving_tile']}"
        )
    y0, y1, x0, x1 = bounds
    moving_patch, _valid = _sample_moving_plane(
        row=row,
        matrix=matrix,
        translation=translation,
        fixed_z_l0=fixed_z_l0,
        bounds=bounds,
        arrays=moving_arrays,
    )
    fixed_patch = np.asarray(
        fixed[fixed_z, y0:y1, x0:x1], dtype=np.float32
    )
    fixed_gray = _stretch(fixed_patch)
    moving_gray = _stretch(moving_patch)
    overlay = np.zeros((*fixed_gray.shape, 3), dtype=np.uint8)
    overlay[..., 0] = moving_gray
    overlay[..., 1] = fixed_gray
    overlay[..., 2] = np.minimum(fixed_gray, moving_gray) // 3
    image = Image.fromarray(overlay)
    image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), "black")
    padded.paste(
        image,
        (
            (THUMBNAIL_SIZE - image.width) // 2,
            (THUMBNAIL_SIZE - image.height) // 2,
        ),
    )
    return padded, fixed_z


def _support_count(row: dict[str, Any], spatial_attempt: str) -> int:
    if row["selected_attempt"] == spatial_attempt:
        return int(
            row["interpolated_fallback_promotion"]["source_good_window_count"]
        )
    return int(row["linear_mapping_recovery"]["source_window_count"])


def render_interpolated_window_qc(
    *,
    summary_path: Path,
    output: Path,
    moving_label: str,
    fixed_label: str,
    dataset_label: str = "",
    columns: int = DEFAULT_INTERPOLATED_QC_COLUMNS,
) -> Path:
    """Render fixed/moving overlays for interpolation-accepted windows."""
    if columns < 1:
        raise ValueError("columns must be positive")
    summary = read_json_object(summary_path)
    _validate_summary_geometry(summary, summary_path)
    native_method = summary.get("cache_config", {}).get("native_method")
    if not summary["windows"]:
        native_method = "method6"
    elif native_method not in INTERPOLATED_ATTEMPTS_BY_METHOD:
        raise ValueError(f"Unsupported native method for interpolated QC: {native_method!r}")
    spatial_attempt, linear_attempt = INTERPOLATED_ATTEMPTS_BY_METHOD[native_method]
    records = [
        read_json_object(Path(item["level0_json"]))
        for item in summary["windows"]
        if item.get("selected_attempt") in (spatial_attempt, linear_attempt)
    ]
    title_prefix = f"{dataset_label} " if dataset_label else ""
    if not records:
        sheet = Image.new("RGB", (900, 80), "#111111")
        ImageDraw.Draw(sheet).text(
            (12, 24),
            f"{title_prefix}{moving_label}: no interpolated recovery windows",
            font=load_qc_font(18),
            fill="white",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, optimize=True)
        return output.resolve()

    records.sort(
        key=lambda row: (
            row["selected_attempt"] != spatial_attempt,
            row["moving_tile"],
            *row["moving_start_l0_zyx"],
        )
    )
    fixed = zarr.open_group(summary["fixed_fused"], mode="r")[str(LEVEL)]
    moving_arrays: dict[str, zarr.Array] = {}
    row_count = (len(records) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * CELL_WIDTH, HEADER_HEIGHT + row_count * CELL_HEIGHT),
        "#111111",
    )
    draw = ImageDraw.Draw(sheet)
    spatial_count = sum(
        row["selected_attempt"] == spatial_attempt for row in records
    )
    linear_count = len(records) - spatial_count
    draw.text(
        (12, 8),
        f"{title_prefix}interpolated-window QC: moving {moving_label} red, "
        f"fixed {fixed_label} green; aligned signal is yellow",
        font=load_qc_font(18),
        fill="white",
    )
    draw.text(
        (12, 35),
        f"{spatial_count} spatial fallbacks (orange) + "
        f"{linear_count} linear-field recoveries (cyan)",
        font=load_qc_font(13),
        fill="#d0d0d0",
    )

    for index, row in enumerate(records):
        thumbnail, fixed_z = _render_interpolated_window(
            row, fixed, moving_arrays
        )
        column = index % columns
        sheet_row = index // columns
        left = column * CELL_WIDTH
        top = HEADER_HEIGHT + sheet_row * CELL_HEIGHT
        color = (
            "#ff9d3d"
            if row["selected_attempt"] == spatial_attempt
            else "#42d9f5"
        )
        draw.rectangle(
            (left + 4, top + 4, left + CELL_WIDTH - 5, top + CELL_HEIGHT - 5),
            outline=color,
            width=2,
        )
        sheet.paste(thumbnail, (left + 18, top + 9))
        start_z, start_y, start_x = row["moving_start_l0_zyx"]
        kind = (
            "spatial"
            if row["selected_attempt"] == spatial_attempt
            else "linear"
        )
        draw.text(
            (left + 9, top + 197),
            f"{Path(row['moving_tile']).name}  z/y/x {start_z}/{start_y}/{start_x}",
            font=load_qc_font(13),
            fill="white",
        )
        draw.text(
            (left + 9, top + 216),
            f"{kind}  support={_support_count(row, spatial_attempt)}  fixed-z={fixed_z}",
            font=load_qc_font(13),
            fill=color,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    return output.resolve()
