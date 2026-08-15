from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from squisher.jpegxr_zarr import register_jpegxr_codec
from squisher_lightsheet import phase_metrics, qc
from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as legacy
from squisher_lightsheet.channel_mattes_anchors import phasecorr_shift_gpu
from squisher_lightsheet.orthogonal_phase import (
    DEFAULT_LATERAL_FACTOR as DEFAULT_ORTHOGONAL_LATERAL_FACTOR,
    IntensityTransform,
    apply_intensity_transform,
    position_tiles,
    run_orthogonal_dumb_phase,
)


DIMENSIONS = ("z", "y", "x")
DEFAULT_LEVEL = 4
DEFAULT_Z_SLAB_PLANES = 32
DEFAULT_FFT_HIGHPASS_SIGMA_ZYX = (1.0, 4.0, 1.0)
DEFAULT_MAX_RESIDUAL_SHIFT_UM = 100.0


@dataclass(frozen=True)
class GlobalPhaseCanvas:
    image: np.ndarray
    coverage: np.ndarray
    spacing_zyx_um: np.ndarray
    global_min_zyx_um: np.ndarray
    slab: dict[str, Any]
    tile_count: int


@dataclass(frozen=True)
class GlobalPhaseResult:
    output_position: Path
    summary: Path
    fixed_mip: Path
    moving_mip: Path
    before_overlay: Path
    after_overlay: Path
    orthogonal_summary: Path
    orthogonal_contact_sheet: Path


def render_position_canvas(
    position: Path,
    *,
    tile_dir: Path | None,
    channel: int,
    level: int,
    z_slab_planes: int,
) -> GlobalPhaseCanvas:
    """Render a complete registered mosaic without blending for global translation estimation."""
    register_jpegxr_codec()
    payload = json.loads(position.read_text())
    tiles = position_tiles(payload, tile_dir=tile_dir)
    if not tiles:
        raise ValueError(f"{position} contains no tiles")
    geometry = legacy.build_geometry(tiles, level=level)
    images, coverage, _rows, slab = legacy.render_center_z_slab_canvases(
        tiles,
        geometry=geometry,
        channel=channel,
        slab_planes=z_slab_planes,
    )
    combined_coverage = coverage["L"] | coverage["R"]
    if not np.any(combined_coverage):
        raise ValueError(f"{position} rendered no covered voxels at level {level}, channel {channel}")
    combined_image = np.maximum(images["L"], images["R"])
    combined_image = np.where(combined_coverage, combined_image, 0.0).astype(np.float32, copy=False)
    spacing = np.asarray(geometry.level_spacing_zyx_um, dtype=np.float64).copy()
    spacing[0] = float(slab["native_z_spacing_um"])
    global_min = np.asarray(geometry.global_min_zyx_um, dtype=np.float64).copy()
    global_min[0] += float(slab["slab_range_z_px"][0]) * spacing[0]
    return GlobalPhaseCanvas(
        image=combined_image,
        coverage=combined_coverage,
        spacing_zyx_um=spacing,
        global_min_zyx_um=global_min,
        slab=slab,
        tile_count=len(tiles),
    )


def pad_to_shape(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Global phase canvases must be 3D, got shape {array.shape}")
    if any(source > target for source, target in zip(array.shape, shape, strict=True)):
        raise ValueError(f"Cannot pad shape {array.shape} into smaller shape {shape}")
    if array.shape == shape:
        return array
    output = np.zeros(shape, dtype=array.dtype)
    output[tuple(slice(0, size) for size in array.shape)] = array
    return output


def expanded_shifted_pair(
    fixed: np.ndarray,
    moving: np.ndarray,
    shift_zyx_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if fixed.shape != moving.shape:
        raise ValueError(f"Expected matching padded shapes, got {fixed.shape} and {moving.shape}")
    shift = np.asarray(shift_zyx_px, dtype=np.float64)
    starts = np.floor(np.minimum(0.0, shift)).astype(np.int64)
    stops = np.ceil(
        np.maximum(
            np.asarray(fixed.shape, dtype=np.float64),
            np.asarray(moving.shape, dtype=np.float64) + shift,
        )
    ).astype(np.int64)
    output_shape = tuple(int(value) for value in stops - starts)
    fixed_output = np.zeros(output_shape, dtype=fixed.dtype)
    moving_output = np.zeros(output_shape, dtype=moving.dtype)
    insert = tuple(
        slice(int(-start), int(-start + size)) for start, size in zip(starts, fixed.shape, strict=True)
    )
    fixed_output[insert] = fixed
    moving_output[insert] = moving
    interpolation_order = 0 if moving.dtype == np.bool_ else 1
    shifted_moving = ndimage.shift(
        moving_output,
        shift=shift_zyx_px,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(moving.dtype, copy=False)
    return fixed_output, shifted_moving


def shifted_position_payload(
    payload: dict[str, Any],
    *,
    total_shift_zyx_um: np.ndarray,
    summary: dict[str, Any],
) -> dict[str, Any]:
    output = deepcopy(payload)
    tiles = output.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("Moving position artifact must contain a nonempty tiles list")
    for record in tiles:
        translation = record.get("translation_um")
        if not isinstance(translation, dict):
            raise ValueError(f"Moving tile {record.get('tile')!r} lacks translation_um")
        for axis, value in zip(DIMENSIONS, total_shift_zyx_um, strict=True):
            translation[axis] = float(translation[axis] + value)
    output["source"] = f"{output.get('source', 'position artifact')} + global phase translation"
    output["derived_by"] = "lightsheet cross-register global-phase"
    output.setdefault("diagnostics", {})["global_phase"] = summary
    return output


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _write_mip(path: Path, image: np.ndarray) -> None:
    import tifffile

    mip = np.max(image, axis=0)
    finite = mip[np.isfinite(mip)]
    positive = finite[finite > 0]
    if positive.size == 0:
        rendered = np.zeros(mip.shape, dtype=np.uint16)
    else:
        low, high = np.percentile(positive, [1.0, 99.8])
        scaled = np.clip((mip - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        rendered = np.rint(scaled * np.iinfo(np.uint16).max).astype(np.uint16)
    tifffile.imwrite(path, rendered, photometric="minisblack")


def run_global_phase(
    *,
    fixed_position: Path,
    moving_position: Path,
    output_dir: Path,
    output_position: Path,
    fixed_tile_dir: Path | None = None,
    moving_tile_dir: Path | None = None,
    fixed_channel: int = 0,
    moving_channel: int = 0,
    level: int = DEFAULT_LEVEL,
    z_slab_planes: int = DEFAULT_Z_SLAB_PLANES,
    fixed_intensity_transform: IntensityTransform = "log1p",
    moving_intensity_transform: IntensityTransform = "identity",
    fft_highpass_sigma_zyx: tuple[float, float, float] | None = DEFAULT_FFT_HIGHPASS_SIGMA_ZYX,
    spatial_highpass_sigma: float | None = None,
    max_residual_shift_um: float = DEFAULT_MAX_RESIDUAL_SHIFT_UM,
    orthogonal_lateral_factor: int = DEFAULT_ORTHOGONAL_LATERAL_FACTOR,
) -> GlobalPhaseResult:
    """Estimate and persist one global translation that moves a registered mosaic onto another."""
    if not np.isfinite(max_residual_shift_um) or max_residual_shift_um < 0:
        raise ValueError("max_residual_shift_um must be finite and nonnegative")
    fixed_canvas = render_position_canvas(
        fixed_position,
        tile_dir=fixed_tile_dir,
        channel=fixed_channel,
        level=level,
        z_slab_planes=z_slab_planes,
    )
    moving_canvas = render_position_canvas(
        moving_position,
        tile_dir=moving_tile_dir,
        channel=moving_channel,
        level=level,
        z_slab_planes=z_slab_planes,
    )
    if not np.allclose(
        fixed_canvas.spacing_zyx_um,
        moving_canvas.spacing_zyx_um,
        rtol=1e-5,
        atol=1e-8,
    ):
        raise ValueError(
            "Fixed and moving global-phase canvases must have matching physical spacing; "
            f"got {fixed_canvas.spacing_zyx_um.tolist()} and "
            f"{moving_canvas.spacing_zyx_um.tolist()}"
        )

    shape = tuple(
        max(fixed_size, moving_size)
        for fixed_size, moving_size in zip(fixed_canvas.image.shape, moving_canvas.image.shape, strict=True)
    )
    fixed_volume = pad_to_shape(
        apply_intensity_transform(fixed_canvas.image, fixed_intensity_transform), shape
    )
    moving_volume = pad_to_shape(
        apply_intensity_transform(moving_canvas.image, moving_intensity_transform), shape
    )
    fixed = np.max(fixed_volume, axis=0, keepdims=True)
    moving = np.max(moving_volume, axis=0, keepdims=True)
    fixed_coverage = np.any(pad_to_shape(fixed_canvas.coverage, shape), axis=0, keepdims=True)
    moving_coverage = np.any(pad_to_shape(moving_canvas.coverage, shape), axis=0, keepdims=True)

    before_shift_zyx_px = np.zeros(3, dtype=np.float64)
    before_shift_zyx_px[1:] = (
        moving_canvas.global_min_zyx_um[1:] - fixed_canvas.global_min_zyx_um[1:]
    ) / fixed_canvas.spacing_zyx_um[1:]
    max_residual_shift_zyx_px = max_residual_shift_um / fixed_canvas.spacing_zyx_um
    max_residual_shift_zyx_px[0] = 0.0
    shift_zyx_px, phase_metadata = phasecorr_shift_gpu(
        fixed,
        moving,
        fft_highpass_sigma_zyx=fft_highpass_sigma_zyx,
        spatial_highpass_sigma=spatial_highpass_sigma,
        search_center_zyx=before_shift_zyx_px,
        max_shift_from_center_zyx=max_residual_shift_zyx_px,
    )
    shift_zyx_px = np.asarray(shift_zyx_px, dtype=np.float64)
    residual_shift_zyx_um = (shift_zyx_px - before_shift_zyx_px) * fixed_canvas.spacing_zyx_um
    max_residual_shift_zyx_um = max_residual_shift_zyx_px * fixed_canvas.spacing_zyx_um
    if not np.all(np.isfinite(shift_zyx_px)) or np.any(
        np.abs(residual_shift_zyx_um) > max_residual_shift_zyx_um + 1e-6
    ):
        raise ValueError(
            "Global phase returned a nonfinite or out-of-window shift: "
            f"residual_zyx_um={residual_shift_zyx_um.tolist()}"
        )
    fixed_before, moving_before = expanded_shifted_pair(fixed, moving, before_shift_zyx_px)
    fixed_before_coverage, moving_before_coverage = expanded_shifted_pair(
        fixed_coverage, moving_coverage, before_shift_zyx_px
    )
    fixed_after, moving_after = expanded_shifted_pair(fixed, moving, shift_zyx_px)
    fixed_after_coverage, moving_after_coverage = expanded_shifted_pair(
        fixed_coverage, moving_coverage, shift_zyx_px
    )
    before_mask = fixed_before_coverage & moving_before_coverage
    after_mask = fixed_after_coverage & moving_after_coverage
    if not np.any(after_mask):
        raise ValueError("Global phase translation produced no fixed/moving coverage overlap")

    origin_delta_zyx_um = np.zeros(3, dtype=np.float64)
    origin_delta_zyx_um[1:] = fixed_canvas.global_min_zyx_um[1:] - moving_canvas.global_min_zyx_um[1:]
    phase_shift_zyx_um = shift_zyx_px * fixed_canvas.spacing_zyx_um
    xy_total_shift_zyx_um = origin_delta_zyx_um + phase_shift_zyx_um
    xy_total_shift_zyx_um[0] = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "global-phase.summary.json"
    fixed_mip = output_dir / "fixed.phase.tif"
    moving_mip = output_dir / "moving.phase.tif"
    before_overlay = output_dir / "before.png"
    after_overlay = output_dir / "after.png"
    _write_mip(fixed_mip, fixed)
    _write_mip(moving_mip, moving)
    qc.write_overlay(
        before_overlay,
        left=np.max(fixed_before, axis=0),
        right=np.max(moving_before, axis=0),
    )
    qc.write_overlay(
        after_overlay,
        left=np.max(fixed_after, axis=0),
        right=np.max(moving_after, axis=0),
    )
    corr_before = phase_metrics.corrcoef_on_mask(fixed_before, moving_before, before_mask)
    corr_after = phase_metrics.corrcoef_on_mask(fixed_after, moving_after, after_mask)
    if (
        corr_before is None
        or corr_after is None
        or not np.isfinite(corr_before)
        or not np.isfinite(corr_after)
        or corr_after <= corr_before
    ):
        raise ValueError(
            f"Bounded global phase did not improve correlation: before={corr_before!r}, after={corr_after!r}"
        )
    fixed_payload = json.loads(fixed_position.read_text())
    moving_payload = json.loads(moving_position.read_text())
    xy_payload = shifted_position_payload(
        moving_payload,
        total_shift_zyx_um=xy_total_shift_zyx_um,
        summary={"stage": "whole-mosaic XY placement"},
    )
    orthogonal = run_orthogonal_dumb_phase(
        fixed_payload=fixed_payload,
        moving_payload=xy_payload,
        fixed_tile_dir=fixed_tile_dir,
        moving_tile_dir=moving_tile_dir,
        fixed_channel=fixed_channel,
        moving_channel=moving_channel,
        fixed_transform=fixed_intensity_transform,
        moving_transform=moving_intensity_transform,
        output_dir=output_dir / "orthogonal",
        max_shift_um=max_residual_shift_um,
        lateral_factor=orthogonal_lateral_factor,
    )
    orthogonal_z_shift_zyx_um = np.asarray([orthogonal.z_residual_um, 0.0, 0.0], dtype=np.float64)
    total_shift_zyx_um = xy_total_shift_zyx_um + orthogonal_z_shift_zyx_um
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "squisher_lightsheet.global_phase.v1",
        "fixed_position": str(fixed_position.resolve()),
        "moving_position": str(moving_position.resolve()),
        "fixed_tile_dir": None if fixed_tile_dir is None else str(fixed_tile_dir.resolve()),
        "moving_tile_dir": None if moving_tile_dir is None else str(moving_tile_dir.resolve()),
        "output_position": str(output_position.resolve()),
        "fixed_channel": fixed_channel,
        "moving_channel": moving_channel,
        "fixed_tile_count": fixed_canvas.tile_count,
        "moving_tile_count": moving_canvas.tile_count,
        "fixed_intensity_transform": fixed_intensity_transform,
        "moving_intensity_transform": moving_intensity_transform,
        "level": level,
        "z_slab_planes": z_slab_planes,
        "source_slab_shape_zyx": list(shape),
        "phase_canvas_shape_zyx": list(fixed.shape),
        "effective_spacing_zyx_um": fixed_canvas.spacing_zyx_um.tolist(),
        "fixed_global_min_zyx_um": fixed_canvas.global_min_zyx_um.tolist(),
        "moving_global_min_zyx_um": moving_canvas.global_min_zyx_um.tolist(),
        "canvas_origin_delta_zyx_um": origin_delta_zyx_um.tolist(),
        "before_shift_to_place_moving_zyx_px": before_shift_zyx_px.tolist(),
        "max_residual_shift_um": max_residual_shift_um,
        "max_residual_shift_zyx_px": max_residual_shift_zyx_px.tolist(),
        "global_phase_applied_axes": ["y", "x"],
        "phase_shift_to_apply_moving_level_px_zyx": shift_zyx_px.tolist(),
        "phase_shift_to_apply_moving_zyx_um": phase_shift_zyx_um.tolist(),
        "residual_shift_from_coarse_zyx_um": residual_shift_zyx_um.tolist(),
        "xy_total_shift_to_apply_moving_zyx_um": xy_total_shift_zyx_um.tolist(),
        "orthogonal_z_residual_um": orthogonal.z_residual_um,
        "orthogonal_lateral_components_applied": False,
        "orthogonal_summary": str(orthogonal.summary),
        "orthogonal_contact_sheet": str(orthogonal.contact_sheet),
        "total_shift_to_apply_moving_zyx_um": total_shift_zyx_um.tolist(),
        "phase_metadata": phase_metadata,
        "corr_before": corr_before,
        "corr_after": corr_after,
        "before_overlap_voxels": int(before_mask.sum()),
        "after_overlap_voxels": int(after_mask.sum()),
        "fixed_slab": fixed_canvas.slab,
        "moving_slab": moving_canvas.slab,
        "fixed_mip": str(fixed_mip.resolve()),
        "moving_mip": str(moving_mip.resolve()),
        "before_overlay": str(before_overlay.resolve()),
        "after_overlay": str(after_overlay.resolve()),
    }
    output_payload = shifted_position_payload(
        moving_payload,
        total_shift_zyx_um=total_shift_zyx_um,
        summary=summary,
    )
    _write_json_atomic(output_position, output_payload)
    _write_json_atomic(summary_path, summary)
    return GlobalPhaseResult(
        output_position=output_position.resolve(),
        summary=summary_path.resolve(),
        fixed_mip=fixed_mip.resolve(),
        moving_mip=moving_mip.resolve(),
        before_overlay=before_overlay.resolve(),
        after_overlay=after_overlay.resolve(),
        orthogonal_summary=orthogonal.summary,
        orthogonal_contact_sheet=orthogonal.contact_sheet,
    )
