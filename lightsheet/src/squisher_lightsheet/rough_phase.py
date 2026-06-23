from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.artifacts import stamp_artifact
from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as legacy

estimate_shift_yx_px = legacy.estimate_shift_yx_px
write_overlay = legacy.write_overlay

DEFAULT_Z_SLAB_PLANES = 8
DEFAULT_PHASE_DOWNSAMPLE_ZYX = (4, 4, 4)


def phase_plane_for_payload(payload: dict, *, z_slab_planes: int = 1) -> str:
    join_axis = payload.get("diagnostics", {}).get("join_axis")
    if join_axis == "z":
        return "xz"
    return "zyx" if z_slab_planes > 1 else "xy"


def render_phase_canvases(
    tiles: list,
    *,
    geometry,
    channel: int,
    phase_plane: str,
    z_slab_planes: int,
):
    if phase_plane == "xz":
        images, coverage, rows = legacy.render_xz_projection_canvases(tiles, geometry=geometry, channel=channel)
        return images, coverage, rows, None
    if phase_plane == "zyx":
        return legacy.render_center_z_slab_canvases(
            tiles,
            geometry=geometry,
            channel=channel,
            slab_planes=z_slab_planes,
        )
    if phase_plane == "xy":
        images, coverage, rows = legacy.render_center_z_canvases(tiles, geometry=geometry, channel=channel)
        return images, coverage, rows, None
    raise ValueError(f"Unsupported phase plane {phase_plane!r}; expected 'xy', 'xz', or 'zyx'")


def phase_spacing_um(geometry, phase_axes: tuple[str, str]):
    axis_to_index = {"z": 0, "y": 1, "x": 2}
    return geometry.level_spacing_zyx_um[[axis_to_index[axis] for axis in phase_axes]]


def phase_shift_spacing_um(
    geometry,
    phase_axes: tuple[str, ...],
    *,
    slab_details: dict[str, Any] | None,
) -> np.ndarray:
    spacing = np.asarray(phase_spacing_um(geometry, phase_axes), dtype=np.float64).copy()
    if slab_details is not None and slab_details.get("z_sampling") == "native_center_z_slab":
        native_z_spacing_um = slab_details.get("native_z_spacing_um")
        if native_z_spacing_um is None:
            raise ValueError("native_center_z_slab phase data must report native_z_spacing_um")
        for index, axis in enumerate(phase_axes):
            if axis == "z":
                spacing[index] = float(native_z_spacing_um)
    return spacing


def overlay_y_scale(geometry, phase_plane: str) -> float:
    if phase_plane == "xz":
        return float(geometry.level_spacing_zyx_um[0] / geometry.level_spacing_zyx_um[1])
    return 1.0


def downsample_axis_blocks(array: np.ndarray, factors: tuple[int, ...], *, reducer: str) -> np.ndarray:
    if array.ndim != len(factors):
        raise ValueError(f"Expected {array.ndim} downsample factors, got {len(factors)}")
    trimmed = array
    shape_parts = []
    reduce_axes = []
    for axis, factor in enumerate(factors):
        if factor < 1:
            raise ValueError("phase downsample factors must be >= 1")
        stop = trimmed.shape[axis] - trimmed.shape[axis] % factor
        selector = [slice(None)] * trimmed.ndim
        selector[axis] = slice(0, stop)
        trimmed = trimmed[tuple(selector)]
    for size, factor in zip(trimmed.shape, factors, strict=True):
        shape_parts.extend([size // factor, factor])
        reduce_axes.append(len(shape_parts) - 1)
    blocks = trimmed.reshape(tuple(shape_parts))
    if reducer == "mean":
        return blocks.mean(axis=tuple(reduce_axes)).astype(np.float32, copy=False)
    if reducer == "any":
        return blocks.any(axis=tuple(reduce_axes))
    raise ValueError(f"Unsupported reducer {reducer!r}")


def downsample_phase_zyx(
    images: dict[str, np.ndarray],
    coverage: dict[str, np.ndarray],
    *,
    factors_zyx: tuple[int, int, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if factors_zyx == (1, 1, 1):
        return images, coverage
    if min(factors_zyx) < 1:
        raise ValueError("phase_downsample_zyx values must be >= 1")

    downsampled_images = {}
    downsampled_coverage = {}
    for side in images:
        image = images[side]
        mask = coverage[side]
        if image.ndim == 2:
            factors = factors_zyx[1:3]
        elif image.ndim == 3:
            factors = factors_zyx
        else:
            raise ValueError(f"Expected 2D or 3D phase image, got ndim={image.ndim}")
        downsampled_images[side] = downsample_axis_blocks(image, factors, reducer="mean")
        downsampled_coverage[side] = downsample_axis_blocks(mask, factors, reducer="any")
    return downsampled_images, downsampled_coverage


def expand_downsampled_shift(
    shift_px: np.ndarray,
    *,
    phase_axes: tuple[str, ...],
    factors_zyx: tuple[int, int, int],
) -> np.ndarray:
    expanded = np.asarray(shift_px, dtype=np.float64).copy()
    axis_factors = {"z": factors_zyx[0], "y": factors_zyx[1], "x": factors_zyx[2]}
    for index, axis in enumerate(phase_axes):
        expanded[index] *= axis_factors[axis]
    return expanded


def shifted_phase_images(
    images: dict[str, Any],
    *,
    shift_px,
) -> dict[str, Any]:
    from scipy import ndimage

    return {
        "L": images["L"],
        "R": ndimage.shift(
            images["R"],
            shift=shift_px,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ),
    }


def shifted_coverage(coverage: dict[str, Any], *, shift_px) -> dict[str, Any]:
    from scipy import ndimage

    return {
        "L": coverage["L"],
        "R": ndimage.shift(
            coverage["R"].astype(np.float32),
            shift=shift_px,
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(bool),
    }


def center_plane_images(images: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int | None]]:
    planes = {}
    z_indices: dict[str, int | None] = {}
    for side, image in images.items():
        if image.ndim == 3:
            z_index = int(image.shape[0] // 2)
            planes[side] = image[z_index]
            z_indices[side] = z_index
        else:
            planes[side] = image
            z_indices[side] = None
    return planes, z_indices


def expanded_shifted_phase_images(
    images: dict[str, np.ndarray],
    *,
    shift_px: np.ndarray,
) -> dict[str, np.ndarray]:
    from scipy import ndimage

    moving = images["R"]
    fixed = images["L"]
    shift = np.asarray(shift_px, dtype=np.float64)
    if fixed.shape != moving.shape:
        raise ValueError(f"Expected matching phase image shapes, got {fixed.shape} and {moving.shape}")
    starts = np.floor(np.minimum(0.0, shift)).astype(int)
    stops = np.ceil(np.maximum(np.asarray(fixed.shape, dtype=np.float64), np.asarray(moving.shape, dtype=np.float64) + shift)).astype(int)
    output_shape = tuple(int(value) for value in stops - starts)
    fixed_output = np.zeros(output_shape, dtype=fixed.dtype)
    moving_output = np.zeros(output_shape, dtype=moving.dtype)
    insert = tuple(slice(int(-start), int(-start + size)) for start, size in zip(starts, fixed.shape, strict=True))
    fixed_output[insert] = fixed
    moving_output[insert] = moving
    moving_output = ndimage.shift(
        moving_output,
        shift=shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    return {"L": fixed_output.astype(np.float32, copy=False), "R": moving_output}


def overlap_center_plane(
    images: dict[str, np.ndarray],
    coverage: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    common = coverage["L"] & coverage["R"]
    coords = np.argwhere(common)
    if coords.size == 0:
        raise ValueError("Corrected L/R overlap is empty; cannot render overlap center plane")
    z_start, y_start, x_start = coords.min(axis=0)
    z_stop, y_stop, x_stop = coords.max(axis=0) + 1
    z_index = int((z_start + z_stop - 1) // 2)
    plane_common = common[z_index]
    plane_coords = np.argwhere(plane_common)
    if plane_coords.size == 0:
        z_index = int(coords[len(coords) // 2, 0])
        plane_common = common[z_index]
        plane_coords = np.argwhere(plane_common)
    y_start, x_start = plane_coords.min(axis=0)
    y_stop, x_stop = plane_coords.max(axis=0) + 1
    crop = (slice(int(y_start), int(y_stop)), slice(int(x_start), int(x_stop)))
    mask = plane_common[crop]
    planes = {
        "L": np.where(mask, images["L"][z_index][crop], 0.0),
        "R": np.where(mask, images["R"][z_index][crop], 0.0),
    }
    details = {
        "overlap_center_z_index": int(z_index),
        "overlap_crop_yx": [[int(y_start), int(y_stop)], [int(x_start), int(x_stop)]],
        "overlap_voxels": int(common.sum()),
        "overlap_plane_pixels": int(mask.sum()),
    }
    return planes, details


def rough_phase_align(
    *,
    position_input: Path,
    output_position: Path,
    output_dir: Path,
    channel: int = 0,
    level: int = 4,
    search_margin_px: int = 64,
    upsample_factor: int = 10,
    seam_fraction: float = 0.10,
    crop_overlap: bool = True,
    z_slab_planes: int = DEFAULT_Z_SLAB_PLANES,
    phase_downsample_yx: int = 1,
    phase_downsample_zyx: tuple[int, int, int] | None = None,
) -> Path:
    if search_margin_px < 0:
        raise ValueError("search_margin_px must be non-negative")
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")
    if not 0.0 < seam_fraction <= 1.0:
        raise ValueError("seam_fraction must be in (0, 1]")
    if z_slab_planes < 1:
        raise ValueError("z_slab_planes must be >= 1")
    if phase_downsample_yx < 1:
        raise ValueError("phase_downsample_yx must be >= 1")
    if phase_downsample_zyx is None:
        phase_downsample_zyx = (1, phase_downsample_yx, phase_downsample_yx) if phase_downsample_yx != 1 else DEFAULT_PHASE_DOWNSAMPLE_ZYX
    if len(phase_downsample_zyx) != 3 or min(phase_downsample_zyx) < 1:
        raise ValueError("phase_downsample_zyx must contain three values >= 1")

    payload = json.loads(position_input.read_text())
    phase_plane = phase_plane_for_payload(payload, z_slab_planes=z_slab_planes)
    phase_axes = legacy.PHASE_PLANES[phase_plane]
    tiles = legacy.load_tiles(payload)
    geometry = legacy.build_geometry(tiles, level=level)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, coverage, tile_rows, slab_details = render_phase_canvases(
        tiles,
        geometry=geometry,
        channel=channel,
        phase_plane=phase_plane,
        z_slab_planes=z_slab_planes,
    )
    phase_images, phase_coverage = downsample_phase_zyx(images, coverage, factors_zyx=phase_downsample_zyx)
    iso_tag = "_isoZ" if phase_plane == "xz" else ""
    initial_overlay = output_dir / f"level{level}_metadata_initial_{phase_plane}{iso_tag}_yellowOverlay_ch{channel}.png"
    if phase_plane == "zyx":
        initial_images, initial_overlap_details = overlap_center_plane(phase_images, phase_coverage)
    else:
        initial_images = phase_images
        initial_overlap_details = None
    legacy.write_overlay_scaled(
        initial_overlay,
        left=initial_images["L"],
        right=initial_images["R"],
        y_scale=overlay_y_scale(geometry, phase_plane),
    )
    initial_full_center_plane_overlay = output_dir / (
        f"level{level}_metadata_initial_{phase_plane}{iso_tag}_full_centerZ_yellowOverlay_ch{channel}.png"
    )
    initial_full_center_plane_images, initial_full_center_plane_indices = center_plane_images(phase_images)
    legacy.write_overlay_scaled(
        initial_full_center_plane_overlay,
        left=initial_full_center_plane_images["L"],
        right=initial_full_center_plane_images["R"],
        y_scale=overlay_y_scale(geometry, phase_plane),
    )

    phase_mask = None
    seam_details = None
    if phase_plane == "xz":
        diagnostics = payload.get("diagnostics", {})
        seam_details_overlap_um = diagnostics.get("overlap_um") or diagnostics.get("z_overlap_um")
        seam_details_overlap_fraction = diagnostics.get("overlap_fraction") or diagnostics.get("z_overlap_fraction")
        phase_mask, seam_details = legacy.seam_band_mask(
            tiles,
            geometry=geometry,
            axes=phase_axes,
            seam_fraction=seam_fraction,
            overlap_um=seam_details_overlap_um,
            overlap_fraction=seam_details_overlap_fraction,
        )

    shift_px, phase_details = legacy.estimate_shift_px(
        phase_images,
        phase_coverage,
        axes=phase_axes,
        phase_mask=phase_mask,
        crop_to_overlap=crop_overlap,
        search_margin_px=search_margin_px,
        upsample_factor=upsample_factor,
    )
    shift_px_level = expand_downsampled_shift(shift_px, phase_axes=phase_axes, factors_zyx=phase_downsample_zyx)
    shift_spacing_um = phase_shift_spacing_um(geometry, phase_axes, slab_details=slab_details)
    shift_um = shift_px_level * shift_spacing_um
    axis_suffix = "".join(phase_axes)
    phase_details = {
        **phase_details,
        "level": int(level),
        "level_factor": int(geometry.level_factor),
        "phase_plane": phase_plane,
        "seam_restricted": phase_mask is not None,
        "seam_band": seam_details,
        "phase_downsample_yx": int(phase_downsample_yx),
        "phase_downsample_zyx": [int(value) for value in phase_downsample_zyx],
        f"shift_to_apply_R_px_downsampled_{axis_suffix}": [float(value) for value in shift_px],
        "slab": slab_details,
        "level_spacing_zyx_um": geometry.level_spacing_zyx_um.tolist(),
        f"level_spacing_{axis_suffix}_um": phase_spacing_um(geometry, phase_axes).tolist(),
        f"shift_spacing_{axis_suffix}_um": shift_spacing_um.tolist(),
        f"shift_to_apply_R_px_{axis_suffix}": [float(value) for value in shift_px_level],
        f"shift_to_apply_R_um_{axis_suffix}": [float(value) for value in shift_um],
    }

    updated = legacy.shifted_payload_by_axes(
        payload,
        shift_um=shift_um,
        axes=phase_axes,
        phase_plane=phase_plane,
        phase_details=phase_details,
        output_position=output_position,
    )
    updated["derived_by"] = "lightsheet.rough_phase.v1"
    updated = stamp_artifact(updated, "lightsheet.position.v1")
    output_position.parent.mkdir(parents=True, exist_ok=True)
    output_position.write_text(json.dumps(updated, indent=2) + "\n")

    if phase_plane == "xz":
        shifted_tiles = legacy.load_tiles(updated)
        shifted_geometry = legacy.build_geometry(shifted_tiles, level=level)
        shifted_projections, projection_tile_rows = legacy.render_global_projection_canvases(
            shifted_tiles,
            geometry=shifted_geometry,
            channel=channel,
        )
        shifted_tile_rows = projection_tile_rows
        shifted_images = {
            "L": shifted_projections["L"]["xz"],
            "R": shifted_projections["R"]["xz"],
        }
        z_display_scale = float(shifted_geometry.level_spacing_zyx_um[0] / shifted_geometry.level_spacing_zyx_um[1])
        projection_outputs, projection_contact_sheet = legacy.write_global_projection_outputs(
            output_dir,
            projections=shifted_projections,
            level=level,
            channel=channel,
            z_display_scale=z_display_scale,
        )
        projection_overlay_paths = {name.lower(): str(path.resolve()) for name, path in projection_outputs}
        projection_contact_sheet_path = str(projection_contact_sheet.resolve())
        corrected_overlap_details = None
    elif phase_plane == "zyx":
        shifted_geometry = geometry
        shifted_volume_images = shifted_phase_images(phase_images, shift_px=shift_px)
        shifted_volume_coverage = shifted_coverage(phase_coverage, shift_px=shift_px)
        shifted_images, corrected_overlap_details = overlap_center_plane(shifted_volume_images, shifted_volume_coverage)
        shifted_tile_rows = tile_rows
        projection_tile_rows = tile_rows
        z_display_scale = None
        projection_overlay_paths = {
            "xy_overlap_center": str((output_dir / f"level{level}_phase_corrected_{phase_plane}{iso_tag}_yellowOverlay_ch{channel}.png").resolve())
        }
        projection_contact_sheet_path = None
    else:
        shifted_geometry = geometry
        shifted_images = shifted_phase_images(images, shift_px=shift_px)
        shifted_tile_rows = tile_rows
        projection_tile_rows = tile_rows
        z_display_scale = None
        projection_overlay_paths = {"xy": str((output_dir / f"level{level}_phase_corrected_{phase_plane}{iso_tag}_yellowOverlay_ch{channel}.png").resolve())}
        projection_contact_sheet_path = None
        corrected_overlap_details = None
    corrected_overlay = output_dir / f"level{level}_phase_corrected_{phase_plane}{iso_tag}_yellowOverlay_ch{channel}.png"
    legacy.write_overlay_scaled(
        corrected_overlay,
        left=shifted_images["L"],
        right=shifted_images["R"],
        y_scale=overlay_y_scale(shifted_geometry, phase_plane),
    )
    corrected_full_center_plane_overlay = output_dir / (
        f"level{level}_phase_corrected_{phase_plane}{iso_tag}_full_centerZ_yellowOverlay_ch{channel}.png"
    )
    if phase_plane == "zyx":
        corrected_full_source_images = expanded_shifted_phase_images(phase_images, shift_px=shift_px)
    else:
        corrected_full_source_images = shifted_images
    corrected_full_center_plane_images, corrected_full_center_plane_indices = center_plane_images(corrected_full_source_images)
    legacy.write_overlay_scaled(
        corrected_full_center_plane_overlay,
        left=corrected_full_center_plane_images["L"],
        right=corrected_full_center_plane_images["R"],
        y_scale=overlay_y_scale(shifted_geometry, phase_plane),
    )

    summary = {
        "schema_version": 1,
        "artifact_type": "lightsheet.rough_phase_summary.v1",
        "position_input": str(position_input.resolve()),
        "output_position": str(output_position.resolve()),
        "channel": int(channel),
        "level": int(level),
        "phase_plane": phase_plane,
        "phase_axes": list(phase_axes),
        "initial_phase_canvas_shape": list(images["L"].shape),
        "corrected_phase_canvas_shape": list(shifted_images["L"].shape),
        "corrected_projection_shape_zyx": shifted_geometry.shape_zyx.astype(int).tolist(),
        "level_spacing_zyx_um": shifted_geometry.level_spacing_zyx_um.tolist(),
        "z_display_scale_for_xz_yz": z_display_scale,
        "phase_alignment": phase_details,
        "initial_overlap_center_plane": initial_overlap_details,
        "corrected_overlap_center_plane": corrected_overlap_details,
        "color_mapping": {"L": "green", "R": "red", "overlap": "yellow"},
        "initial_overlay": str(initial_overlay.resolve()),
        "corrected_overlay": str(corrected_overlay.resolve()),
        "initial_full_center_plane_overlay": str(initial_full_center_plane_overlay.resolve()),
        "corrected_full_center_plane_overlay": str(corrected_full_center_plane_overlay.resolve()),
        "initial_full_center_plane_z_indices": initial_full_center_plane_indices,
        "corrected_full_center_plane_z_indices": corrected_full_center_plane_indices,
        "corrected_projection_overlays": projection_overlay_paths,
        "corrected_projection_contact_sheet": projection_contact_sheet_path,
        "initial_tiles": tile_rows,
        "corrected_tiles": shifted_tile_rows,
        "corrected_projection_tiles": projection_tile_rows,
    }
    summary_path = output_dir / f"level{level}_{phase_plane}_phase_alignment_ch{channel}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return output_position.resolve()
