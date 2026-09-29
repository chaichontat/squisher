from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.registration import phase_cross_correlation
import tifffile
import zarr

from squisher_lightsheet.geometry import signed_bounds
from squisher_lightsheet.ngff import axes, level_array, scale_translation


DIMENSIONS = ("z", "y", "x")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _tile_key(value: str) -> str:
    name = Path(value).name
    for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _vector(record: dict[str, Any], field: str) -> np.ndarray:
    values = record.get(field)
    if not isinstance(values, dict) or any(axis not in values for axis in DIMENSIONS):
        raise ValueError(f"tile {record.get('tile')!r} is missing {field} z/y/x")
    return np.asarray([float(values[axis]) for axis in DIMENSIONS], dtype=np.float64)


def _shape_zyx(record: dict[str, Any]) -> np.ndarray:
    shape = record.get("shape")
    axes_value = record.get("axes")
    if not isinstance(shape, list):
        raise ValueError(f"tile {record.get('tile')!r} is missing shape")
    if axes_value == "CZYX" and len(shape) == 4:
        return np.asarray(shape[1:], dtype=np.float64)
    if axes_value == "ZYX" and len(shape) == 3:
        return np.asarray(shape, dtype=np.float64)
    raise ValueError(f"tile {record.get('tile')!r} has unsupported axes/shape {axes_value!r}/{shape!r}")


def _record_centers(records: list[dict[str, Any]]) -> np.ndarray:
    centers = []
    for record in records:
        low, high = signed_bounds(
            _vector(record, "translation_um"),
            _vector(record, "scale_um"),
            _shape_zyx(record),
        )
        centers.append((low + high) / 2.0)
    return np.asarray(centers)


def _metadata_center(
    side_records: list[dict[str, Any]], metadata_by_key: dict[str, dict[str, Any]]
) -> np.ndarray:
    matched = []
    for record in side_records:
        key = _tile_key(str(record.get("tile", "")))
        if key not in metadata_by_key:
            raise ValueError(f"side tile {record.get('tile')!r} is absent from metadata positions")
        metadata_record = deepcopy(metadata_by_key[key])
        metadata_record["shape"] = record["shape"]
        metadata_record["axes"] = record["axes"]
        matched.append(metadata_record)
    return _record_centers(matched).mean(axis=0)


def _orientation_signs(
    side_records: list[dict[str, Any]], metadata_by_key: dict[str, dict[str, Any]]
) -> np.ndarray:
    signs = []
    for record in side_records:
        metadata = metadata_by_key[_tile_key(str(record["tile"]))]
        signs.append(np.sign(_vector(metadata, "scale_um")))
    first = signs[0]
    if np.any(first == 0.0) or any(not np.array_equal(sign, first) for sign in signs[1:]):
        raise ValueError("metadata positions have inconsistent or zero orientation signs within a side")
    return first


def _orient_records(
    records: list[dict[str, Any]], *, signs_zyx: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray]:
    side_center = _record_centers(records).mean(axis=0)
    oriented = []
    for source in records:
        record = deepcopy(source)
        shape = _shape_zyx(record)
        center = _record_centers([record])[0]
        oriented_center = side_center + signs_zyx * (center - side_center)
        scale = np.abs(_vector(record, "scale_um")) * signs_zyx
        translation = oriented_center - scale * shape / 2.0
        record["scale_um"] = {
            axis: float(value) for axis, value in zip(DIMENSIONS, scale, strict=True)
        }
        record["translation_um"] = {
            axis: float(value) for axis, value in zip(DIMENSIONS, translation, strict=True)
        }
        oriented.append(record)
    return oriented, side_center


def _fused_zmedian(
    path: Path,
    *,
    level: int,
    signs_zyx: np.ndarray,
    side_center_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group = zarr.open_group(str(path), mode="r")
    array = level_array(group, level=level, context=path)
    if axes(group, array) != "ZYX":
        raise ValueError(f"{path} level {level} must have ZYX axes")
    names, scale, translation, _, _ = scale_translation(group, dataset_index=level)
    if [name.lower() for name in names] != list(DIMENSIONS):
        raise ValueError(f"{path} has unsupported axes {names}")
    volume = np.asarray(array)
    for volume_axis, sign in enumerate(signs_zyx):
        if sign < 0:
            volume = np.flip(volume, axis=volume_axis)
    xy_image = np.median(volume, axis=0).astype(np.uint16)
    xz_image = np.max(volume, axis=1).astype(np.uint16)
    scale_array = np.asarray(scale, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64)
    end = translation_array + scale_array * np.asarray(array.shape, dtype=np.float64)
    oriented_endpoints = np.stack(
        [
            side_center_zyx + signs_zyx * (translation_array - side_center_zyx),
            side_center_zyx + signs_zyx * (end - side_center_zyx),
        ]
    )
    oriented_origin = np.min(oriented_endpoints, axis=0)
    return xy_image, xz_image, np.abs(scale_array), oriented_origin


def _fused_position_record(
    path: Path,
    *,
    side: str,
    signs_zyx: np.ndarray,
    side_center_zyx: np.ndarray,
    offset_zyx: np.ndarray,
) -> dict[str, Any]:
    group = zarr.open_group(str(path), mode="r")
    array = level_array(group, level=0, context=path)
    if axes(group, array) != "ZYX":
        raise ValueError(f"{path} level 0 must have ZYX axes")
    names, scale, translation, _, _ = scale_translation(group, dataset_index=0)
    if [name.lower() for name in names] != list(DIMENSIONS):
        raise ValueError(f"{path} has unsupported axes {names}")
    scale_array = np.asarray(scale, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64)
    oriented_translation = (
        side_center_zyx
        + signs_zyx * (translation_array - side_center_zyx)
        + offset_zyx
    )
    signed_scale = np.abs(scale_array) * signs_zyx
    return {
        "tile": path.name,
        "side": side,
        "path": str(path.resolve()),
        "translation_um": {
            axis: float(value)
            for axis, value in zip(DIMENSIONS, oriented_translation, strict=True)
        },
        "scale_um": {
            axis: float(value) for axis, value in zip(DIMENSIONS, signed_scale, strict=True)
        },
        "shape": [int(value) for value in array.shape],
        "axes": "ZYX",
    }


def _normalize(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image) & (image > 0)]
    low, high = np.percentile(values, [1.0, 99.8]) if values.size else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled - ndimage.gaussian_filter(scaled, sigma=8.0)).astype(np.float32)


def _stretch(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image) & (image > 0)]
    low, high = np.percentile(values, [0.5, 99.8]) if values.size else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    return np.clip((image - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)


def _correlation(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float:
    if np.count_nonzero(mask) < 2:
        return 0.0
    left = fixed[mask].astype(np.float64)
    right = moving[mask].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def align_registered_lr_fusions(
    *,
    metadata_position: Path,
    left_position: Path,
    right_position: Path,
    left_fusion: Path,
    right_fusion: Path,
    output_dir: Path,
    fusion_level: int = 0,
    manual_cr_shift_zyx_um: tuple[float, float, float] | None = None,
) -> Path:
    """Place registered side mosaics in metadata space and phase-align CR to CL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_payload = _read_json(metadata_position)
    left_payload = _read_json(left_position)
    right_payload = _read_json(right_position)
    metadata_records = metadata_payload.get("tiles")
    left_records = left_payload.get("tiles")
    right_records = right_payload.get("tiles")
    if not all(isinstance(records, list) and records for records in (metadata_records, left_records, right_records)):
        raise ValueError("metadata, left, and right position inputs must contain tiles")
    metadata_by_key = {_tile_key(str(record["tile"])): record for record in metadata_records}
    source_records = {"CL": left_records, "CR": right_records}
    orientation_signs = {
        side: _orientation_signs(records, metadata_by_key)
        for side, records in source_records.items()
    }
    oriented = {
        side: _orient_records(records, signs_zyx=orientation_signs[side])
        for side, records in source_records.items()
    }
    side_records = {side: result[0] for side, result in oriented.items()}
    side_centers = {side: result[1] for side, result in oriented.items()}
    metadata_centers = {
        side: _metadata_center(records, metadata_by_key) for side, records in side_records.items()
    }
    offsets = {
        side: metadata_centers[side] - _record_centers(records).mean(axis=0)
        for side, records in side_records.items()
    }

    fused: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for side, path in (("CL", left_fusion), ("CR", right_fusion)):
        fused[side] = _fused_zmedian(
            path,
            level=fusion_level,
            signs_zyx=orientation_signs[side],
            side_center_zyx=side_centers[side],
        )
    pixel_um = fused["CL"][2][1:]
    if not np.allclose(pixel_um, fused["CR"][2][1:], rtol=1e-6, atol=1e-6):
        raise ValueError("CL and CR fused XY scales differ")
    if manual_cr_shift_zyx_um is None:
        manual_shift_zyx = None
    else:
        manual_shift_zyx = np.asarray(manual_cr_shift_zyx_um, dtype=np.float64)
        if manual_shift_zyx.shape != (3,) or not np.all(np.isfinite(manual_shift_zyx)):
            raise ValueError("manual CR shift must contain three finite z,y,x values")
    initial_shift_zyx = (
        manual_shift_zyx if manual_shift_zyx is not None else np.zeros(3, dtype=np.float64)
    )
    placed_origins = {
        side: fused[side][3][1:]
        + offsets[side][1:]
        + (initial_shift_zyx[1:] if side == "CR" else 0.0)
        for side in fused
    }
    canvas_min = np.min(np.stack(list(placed_origins.values())), axis=0)
    canvas_max = np.max(
        np.stack(
            [
                placed_origins[side] + pixel_um * np.asarray(fused[side][0].shape)
                for side in fused
            ]
        ),
        axis=0,
    )
    canvas_shape = np.ceil((canvas_max - canvas_min) / pixel_um).astype(int)
    canvases: dict[str, np.ndarray] = {}
    for side in ("CL", "CR"):
        image = fused[side][0]
        start = np.rint((placed_origins[side] - canvas_min) / pixel_um).astype(int)
        stop = start + np.asarray(image.shape)
        canvas = np.zeros(tuple(canvas_shape), dtype=np.uint16)
        canvas[start[0] : stop[0], start[1] : stop[1]] = image
        canvases[side] = canvas
        tifffile.imwrite(output_dir / f"{side.lower()}-zmedian.tif", canvas, compression="zstd")

    fixed_mask = canvases["CL"] > 0
    moving_mask = canvases["CR"] > 0
    initial_overlap = fixed_mask & moving_mask
    if np.count_nonzero(initial_overlap) < 1024:
        raise ValueError("metadata placement produced fewer than 1024 overlapping pixels")
    correlation_before = _correlation(canvases["CL"], canvases["CR"], initial_overlap)
    if manual_shift_zyx is None:
        candidate_shift_yx, phase_error, phase_diff = phase_cross_correlation(
            _normalize(canvases["CL"]),
            _normalize(canvases["CR"]),
            reference_mask=fixed_mask,
            moving_mask=moving_mask,
            overlap_ratio=0.1,
        )
        candidate_shift_yx = np.asarray(candidate_shift_yx, dtype=np.float64)
        candidate_cr = ndimage.shift(
            canvases["CR"], candidate_shift_yx, order=1, mode="constant", cval=0.0
        )
        candidate_mask = (
            ndimage.shift(moving_mask, candidate_shift_yx, order=0, mode="constant", cval=0) > 0
        )
        candidate_overlap = fixed_mask & candidate_mask
        correlation_candidate = _correlation(canvases["CL"], candidate_cr, candidate_overlap)
        minimum_retained_overlap = 0.5 * np.count_nonzero(initial_overlap)
        phase_accepted = bool(
            np.count_nonzero(candidate_overlap) >= minimum_retained_overlap
            and correlation_candidate > correlation_before
        )
        residual_shift_yx_px = (
            candidate_shift_yx if phase_accepted else np.zeros(2, dtype=np.float64)
        )
        residual_shift_yx_um = residual_shift_yx_px * pixel_um
        total_shift_zyx_um = np.asarray([0.0, *residual_shift_yx_um], dtype=np.float64)
        shifted_cr = candidate_cr if phase_accepted else canvases["CR"]
        final_overlap = candidate_overlap if phase_accepted else initial_overlap
    else:
        candidate_shift_yx = np.zeros(2, dtype=np.float64)
        candidate_overlap = initial_overlap
        correlation_candidate = correlation_before
        phase_error = None
        phase_diff = None
        phase_accepted = False
        residual_shift_yx_px = np.zeros(2, dtype=np.float64)
        residual_shift_yx_um = np.zeros(2, dtype=np.float64)
        total_shift_zyx_um = manual_shift_zyx
        shifted_cr = canvases["CR"]
        final_overlap = initial_overlap
    overlay = np.zeros((*canvases["CL"].shape, 3), dtype=np.uint8)
    overlay[..., 1] = _stretch(canvases["CL"])
    overlay[..., 0] = _stretch(shifted_cr)
    Image.fromarray(overlay).save(output_dir / "phase-overlay.png")

    xz_scale = fused["CL"][2][[0, 2]]
    if not np.allclose(xz_scale, fused["CR"][2][[0, 2]], rtol=1e-6, atol=1e-6):
        raise ValueError("CL and CR fused XZ scales differ")
    xz_origins = {
        side: fused[side][3][[0, 2]] + offsets[side][[0, 2]] for side in fused
    }
    xz_origins["CR"] = xz_origins["CR"] + total_shift_zyx_um[[0, 2]]
    xz_min = np.min(np.stack(list(xz_origins.values())), axis=0)
    xz_max = np.max(
        np.stack(
            [
                xz_origins[side] + xz_scale * np.asarray(fused[side][1].shape)
                for side in fused
            ]
        ),
        axis=0,
    )
    xz_shape = np.ceil((xz_max - xz_min) / xz_scale).astype(int)
    xz_canvases = {
        side: np.zeros(tuple(xz_shape), dtype=np.uint16) for side in ("CL", "CR")
    }
    for side in ("CL", "CR"):
        image = fused[side][1]
        start = np.rint((xz_origins[side] - xz_min) / xz_scale).astype(int)
        stop = start + np.asarray(image.shape)
        xz_canvases[side][start[0] : stop[0], start[1] : stop[1]] = image
    xz_overlay = np.zeros((*xz_canvases["CL"].shape, 3), dtype=np.uint8)
    xz_overlay[..., 1] = _stretch(xz_canvases["CL"])
    xz_overlay[..., 0] = _stretch(xz_canvases["CR"])
    Image.fromarray(_stretch(xz_canvases["CL"])).save(output_dir / "cl-xz.png")
    Image.fromarray(_stretch(xz_canvases["CR"])).save(output_dir / "cr-xz.png")
    Image.fromarray(xz_overlay).save(output_dir / "phase-overlay-xz.png")

    phase_path = output_dir / "phase.json"
    phase = {
        "artifact_type": "squisher_lightsheet.registered_lr_phasecorr.v1",
        "fixed": "CL",
        "moving": "CR",
        "alignment_mode": "manual" if manual_shift_zyx is not None else "phase",
        "pixel_um_yx": pixel_um.tolist(),
        "phase_shift_to_apply_to_cr_yx_px": residual_shift_yx_px.tolist(),
        "phase_shift_to_apply_to_cr_yx_um": residual_shift_yx_um.tolist(),
        "total_shift_to_apply_to_cr_zyx_um": total_shift_zyx_um.tolist(),
        "phase_candidate_shift_yx_px": candidate_shift_yx.tolist(),
        "phase_candidate_accepted": phase_accepted,
        "overlap_pixels_before": int(np.count_nonzero(initial_overlap)),
        "overlap_pixels_after": int(np.count_nonzero(final_overlap)),
        "phase_candidate_overlap_pixels": int(np.count_nonzero(candidate_overlap)),
        "corr_before_on_initial_overlap": correlation_before,
        "corr_after_on_shifted_overlap": _correlation(canvases["CL"], shifted_cr, final_overlap),
        "phase_candidate_correlation": correlation_candidate,
        "phase_error": None if phase_error is None else float(phase_error),
        "phase_diff": None if phase_diff is None else float(phase_diff),
        "output_overlay": str((output_dir / "phase-overlay.png").resolve()),
    }
    _write_json(phase_path, phase)

    side_outputs = {}
    for side, source, records in (
        ("CL", left_position, left_records),
        ("CR", right_position, right_records),
    ):
        payload = deepcopy(left_payload if side == "CL" else right_payload)
        payload["tiles"] = deepcopy(side_records[side])
        offset = offsets[side].copy()
        if side == "CR":
            offset += total_shift_zyx_um
        for record in payload["tiles"]:
            record["translation_um"] = {
                axis: float(value)
                for axis, value in zip(
                    DIMENSIONS, _vector(record, "translation_um") + offset, strict=True
                )
            }
        payload["lr_coarse_alignment"] = {
            "source_position": str(source.resolve()),
            "metadata_position": str(metadata_position.resolve()),
            "metadata_center_um_zyx": metadata_centers[side].tolist(),
            "centroid_offset_um_zyx": offsets[side].tolist(),
            "alignment_mode": "manual" if manual_shift_zyx is not None else "phase",
            "shift_um_zyx": total_shift_zyx_um.tolist()
            if side == "CR"
            else [0.0, 0.0, 0.0],
        }
        side_output = output_dir / f"{side.lower()}.positions.json"
        _write_json(side_output, payload)
        side_outputs[side] = str(side_output.resolve())

    fused_records = []
    for side, path in (("CL", left_fusion), ("CR", right_fusion)):
        fused_offset = offsets[side].copy()
        if side == "CR":
            fused_offset += total_shift_zyx_um
        fused_records.append(
            _fused_position_record(
                path,
                side="L" if side == "CL" else "R",
                signs_zyx=orientation_signs[side],
                side_center_zyx=side_centers[side],
                offset_zyx=fused_offset,
            )
        )
    fused_bounds = [
        signed_bounds(
            _vector(record, "translation_um"),
            _vector(record, "scale_um"),
            _shape_zyx(record),
        )
        for record in fused_records
    ]
    z_overlap_um = float(
        min(bounds[1][0] for bounds in fused_bounds)
        - max(bounds[0][0] for bounds in fused_bounds)
    )
    if z_overlap_um <= 0:
        raise ValueError("placed CL/CR fused volumes have no Z overlap")
    fused_position_path = output_dir / "fused.positions.json"
    _write_json(
        fused_position_path,
        {
            "units": "micrometer",
            "source": "registered side fusions placed by align-registered-lr",
            "diagnostics": {
                "join_axis": "z",
                "right_flip_axes": ["z", "x"],
                "overlap_um": z_overlap_um,
                "z_overlap_um": z_overlap_um,
                "overlap_fraction": z_overlap_um
                / min(bounds[1][0] - bounds[0][0] for bounds in fused_bounds),
                "manual_cr_shift_zyx_um": (
                    None if manual_shift_zyx is None else manual_shift_zyx.tolist()
                ),
            },
            "tiles": fused_records,
        },
    )

    manifest_path = output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "artifact_type": "squisher_lightsheet.registered_lr_zmedian.v1",
            "fusion_level": fusion_level,
            "metadata_position_json": str(metadata_position.resolve()),
            "metadata_centers_um_zyx": {
                side: center.tolist() for side, center in metadata_centers.items()
            },
            "pixel_um_yx": pixel_um.tolist(),
            "canvas_origin_um_yx": canvas_min.tolist(),
            "shape_yx_px": canvas_shape.tolist(),
            "outputs": {
                "CL": str((output_dir / "cl-zmedian.tif").resolve()),
                "CR": str((output_dir / "cr-zmedian.tif").resolve()),
                "overlay": str((output_dir / "phase-overlay.png").resolve()),
                "overlay_xz": str((output_dir / "phase-overlay-xz.png").resolve()),
                "cl_xz": str((output_dir / "cl-xz.png").resolve()),
                "cr_xz": str((output_dir / "cr-xz.png").resolve()),
            },
            "side_positions": side_outputs,
            "fused_position": str(fused_position_path.resolve()),
            "side_fusions": {"CL": str(left_fusion.resolve()), "CR": str(right_fusion.resolve())},
            "phasecorr": str(phase_path.resolve()),
            "orientation_signs_zyx": {
                side: signs.tolist() for side, signs in orientation_signs.items()
            },
        },
    )
    return manifest_path
