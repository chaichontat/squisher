"""Verified matched-tile anchors measured on one registered world grid."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from squisher_lightsheet.artifact_io import sha256_file
from squisher_lightsheet.channel_mattes_anchors import mip_gradient_component_ncc_moving
from squisher_lightsheet.mvs_seams import _sample_registered_patch, tile_image_path
from squisher_lightsheet.phase_metrics import robust_normalize
from squisher_lightsheet.tile_input import level_scale_ratio


ARTIFACT_TYPE = "squisher_lightsheet.registered_grid_channel_anchors.v1"
DEFAULT_LEVEL = 2
DEFAULT_PATCH_SHAPE_ZYX = (32, 240, 240)
DEFAULT_NATIVE_VERIFY_SHAPE_ZYX = (16, 192, 192)
DEFAULT_Z_FRACTIONS = (0.25, 0.5, 0.75)
DEFAULT_YX_FRACTIONS = (0.25, 0.75)


def _finite_list_or_none(values: np.ndarray | list[float]) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    return array.tolist() if np.all(np.isfinite(array)) else None


def _records_by_tile(payload: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("tiles")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must contain a nonempty tiles list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        tile = row.get("tile") if isinstance(row, dict) else None
        if not isinstance(tile, str) or not tile or tile in result:
            raise ValueError(f"{label} contains an invalid or duplicate tile identity {tile!r}")
        result[tile] = row
    return result


def _vector_zyx(row: dict[str, Any], key: str) -> np.ndarray:
    try:
        values = np.asarray([row[key][axis] for axis in "zyx"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Tile {row.get('tile')!r} has invalid {key}") from error
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Tile {row.get('tile')!r} has invalid {key}")
    return values


def _registered_affine(row: dict[str, Any]) -> np.ndarray:
    try:
        affine = np.asarray(row["registered_affine"]["matrix"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Tile {row.get('tile')!r} has invalid registered affine") from error
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError(f"Tile {row.get('tile')!r} has invalid registered affine")
    return affine


def _registered_center_um(row: dict[str, Any], local_center_px_zyx: np.ndarray) -> np.ndarray:
    input_world = (
        _vector_zyx(row, "stage_translation_um")
        + local_center_px_zyx * np.abs(_vector_zyx(row, "stage_scale_um"))
    )
    return (_registered_affine(row) @ np.r_[input_world, 1.0])[:3]


def _highpass(volume: np.ndarray, *, sigma: tuple[float, float, float]) -> np.ndarray:
    normalized = robust_normalize(volume)
    return normalized - ndimage.gaussian_filter(normalized, sigma=sigma)


def _phase_shift(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    mask: np.ndarray,
    sigma: tuple[float, float, float],
) -> tuple[np.ndarray, float, float]:
    fixed_high = _highpass(fixed, sigma=sigma)
    moving_high = _highpass(moving, sigma=sigma)
    tapered = np.ones(fixed.shape, dtype=np.float32)
    for axis, size in enumerate(fixed.shape):
        window = np.hanning(size).astype(np.float32)
        reshape = [1, 1, 1]
        reshape[axis] = size
        tapered *= window.reshape(reshape)
    support = mask.astype(np.float32) * tapered
    fixed_fit = fixed_high * support
    moving_fit = moving_high * support
    shift, error, _phase = phase_cross_correlation(
        fixed_fit,
        moving_fit,
        upsample_factor=10,
        normalization=None,
    )

    product = np.fft.fftn(fixed_fit) * np.conj(np.fft.fftn(moving_fit))
    product /= np.maximum(np.abs(product), 1e-7)
    correlation = np.abs(np.fft.ifftn(product))
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    peak_value = float(correlation[peak])
    competing = correlation.copy()
    neighborhood = tuple(
        np.asarray([(index + delta) % size for delta in range(-2, 3)], dtype=np.int64)
        for index, size in zip(peak, correlation.shape, strict=True)
    )
    competing[np.ix_(*neighborhood)] = 0
    second_peak = float(np.max(competing))
    ratio = peak_value / max(second_peak, 1e-12)
    return np.asarray(shift, dtype=np.float64), float(error), float(ratio)


def _shifted_gradient_score(
    fixed: np.ndarray,
    moving: np.ndarray,
    shift: np.ndarray,
) -> tuple[float, float]:
    before = mip_gradient_component_ncc_moving(fixed, moving)
    shifted = ndimage.shift(
        moving,
        shift=tuple(float(value) for value in shift),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    after = mip_gradient_component_ncc_moving(fixed, shifted)
    return float(before), float(after)


def _subblock_shifts(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    fixed_threshold: float,
) -> list[np.ndarray]:
    shifts: list[np.ndarray] = []
    y_mid = fixed.shape[1] // 2
    x_mid = fixed.shape[2] // 2
    for y_slice in (slice(0, y_mid), slice(y_mid, fixed.shape[1])):
        for x_slice in (slice(0, x_mid), slice(x_mid, fixed.shape[2])):
            selection = (slice(None), y_slice, x_slice)
            fixed_block = fixed[selection]
            moving_block = moving[selection]
            mask = (fixed_block >= fixed_threshold) & (moving_block > 0)
            if int(np.count_nonzero(mask)) < 4096 or float(np.mean(mask)) < 0.05:
                continue
            if float(np.std(fixed_block[mask])) <= 1 or float(np.std(moving_block[mask])) <= 1:
                continue
            shift, _error, _ratio = _phase_shift(
                fixed_block,
                moving_block,
                mask=mask,
                sigma=(1.0, 3.0, 3.0),
            )
            shifts.append(shift)
    return shifts


def _measure_anchor(
    *,
    fixed_payload: dict[str, Any],
    moving_payload: dict[str, Any],
    fixed_record: dict[str, Any],
    moving_record: dict[str, Any],
    local_center_px_zyx: np.ndarray,
    base_spacing_um_zyx: np.ndarray,
    grid_spacing_um_zyx: np.ndarray,
    level: int,
    patch_shape_zyx: tuple[int, int, int],
    native_verify_shape_zyx: tuple[int, int, int],
    fixed_threshold: float,
) -> dict[str, Any]:
    tile = str(moving_record["tile"])
    center = _registered_center_um(fixed_record, local_center_px_zyx)
    fixed = _sample_registered_patch(
        registration_payload=fixed_payload,
        tile_record=fixed_record,
        center_zyx_um=center,
        spacing_um_zyx=base_spacing_um_zyx,
        level=level,
        channel=0,
        patch_shape_zyx=patch_shape_zyx,
    )
    moving = _sample_registered_patch(
        registration_payload=moving_payload,
        tile_record=moving_record,
        center_zyx_um=center,
        spacing_um_zyx=base_spacing_um_zyx,
        level=level,
        channel=0,
        patch_shape_zyx=patch_shape_zyx,
    )
    mask = (fixed >= fixed_threshold) & (moving > 0)
    support_voxels = int(np.count_nonzero(mask))
    support_fraction = float(np.mean(mask))
    reasons: list[str] = []
    if support_voxels < 16384 or support_fraction < 0.05:
        reasons.append("insufficient_threshold_reviewed_support")
    if support_voxels and (
        float(np.std(fixed[mask])) <= 1 or float(np.std(moving[mask])) <= 1
    ):
        reasons.append("insufficient_intensity_variation")

    shift = np.zeros(3, dtype=np.float64)
    phase_error = float("nan")
    peak_ratio = float("nan")
    subblocks: list[np.ndarray] = []
    gradient_before = float("nan")
    gradient_after = float("nan")
    native_shift = np.full(3, np.nan, dtype=np.float64)
    native_agreement = float("nan")
    if not reasons:
        shift, phase_error, peak_ratio = _phase_shift(
            fixed,
            moving,
            mask=mask,
            sigma=(1.0, 3.0, 3.0),
        )
        if abs(float(shift[0])) >= 4 or np.any(np.abs(shift[1:]) >= 16):
            reasons.append("phase_peak_at_or_beyond_search_limit")
        if peak_ratio < 1.01:
            reasons.append("ambiguous_competing_phase_peak")
        subblocks = _subblock_shifts(
            fixed,
            moving,
            fixed_threshold=fixed_threshold,
        )
        if len(subblocks) < 2:
            reasons.append("insufficient_independent_subblocks")
        else:
            subblock_yx = np.asarray(subblocks)[:, 1:3]
            if float(np.max(np.linalg.norm(subblock_yx - shift[None, 1:3], axis=1))) > 2.0:
                reasons.append("independent_subblocks_disagree")
        lateral_shift = np.asarray([0.0, shift[1], shift[2]])
        gradient_before, gradient_after = _shifted_gradient_score(fixed, moving, lateral_shift)
        if not np.isfinite(gradient_after) or gradient_after < gradient_before - 0.01:
            reasons.append("lateral_shift_worsens_gradient_alignment")
        elif np.linalg.norm(lateral_shift[1:]) > 0.5 and gradient_after < gradient_before + 0.005:
            reasons.append("lateral_shift_has_no_verified_improvement")

        fixed_native = _sample_registered_patch(
            registration_payload=fixed_payload,
            tile_record=fixed_record,
            center_zyx_um=center,
            spacing_um_zyx=base_spacing_um_zyx,
            level=0,
            channel=0,
            patch_shape_zyx=native_verify_shape_zyx,
        )
        moving_native = _sample_registered_patch(
            registration_payload=moving_payload,
            tile_record=moving_record,
            center_zyx_um=center,
            spacing_um_zyx=base_spacing_um_zyx,
            level=0,
            channel=0,
            patch_shape_zyx=native_verify_shape_zyx,
        )
        native_mask = (fixed_native >= fixed_threshold) & (moving_native > 0)
        if int(np.count_nonzero(native_mask)) < 4096 or float(np.mean(native_mask)) < 0.05:
            reasons.append("insufficient_native_verification_support")
        else:
            native_shift, _native_error, _native_ratio = _phase_shift(
                fixed_native,
                moving_native,
                mask=native_mask,
                sigma=(1.0, 4.0, 4.0),
            )
            expected_native_yx = shift[1:3] * (
                grid_spacing_um_zyx[1:3] / base_spacing_um_zyx[1:3]
            )
            native_agreement = float(np.linalg.norm(native_shift[1:3] - expected_native_yx))
            if native_agreement > 3.0:
                reasons.append("native_verification_disagrees")

    uncertainty: list[float] | None = None
    if len(subblocks) >= 2:
        spread = np.std(np.asarray(subblocks)[:, 1:3], axis=0, ddof=1)
        uncertainty = np.maximum(spread / np.sqrt(len(subblocks)), 0.25).tolist()
    return {
        "status": "accepted" if not reasons else "rejected",
        "rejection_reasons": reasons,
        "fixed_tile": str(fixed_record["tile"]),
        "moving_tile": tile,
        "fixed_path": str(tile_image_path(fixed_payload, fixed_record).resolve()),
        "moving_path": str(tile_image_path(moving_payload, moving_record).resolve()),
        "moving_local_center_px_zyx": local_center_px_zyx.tolist(),
        "registered_center_um_zyx": center.tolist(),
        "grid_spacing_um_zyx": grid_spacing_um_zyx.tolist(),
        "patch_shape_zyx": list(patch_shape_zyx),
        "fixed_threshold": float(fixed_threshold),
        "threshold_support_voxels": support_voxels,
        "threshold_support_fraction": support_fraction,
        "shift_to_apply_moving_grid_px_zyx": shift.tolist(),
        "lateral_shift_to_apply_moving_um_yx": (
            shift[1:3] * grid_spacing_um_zyx[1:3]
        ).tolist(),
        "uncertainty_grid_px_yx": uncertainty,
        "phase_error": None if not np.isfinite(phase_error) else phase_error,
        "peak_to_competing_peak_ratio": None if not np.isfinite(peak_ratio) else peak_ratio,
        "independent_subblock_shifts_grid_px_zyx": [row.tolist() for row in subblocks],
        "gradient_component_ncc_before": (
            None if not np.isfinite(gradient_before) else gradient_before
        ),
        "gradient_component_ncc_after_lateral_shift": (
            None if not np.isfinite(gradient_after) else gradient_after
        ),
        "native_verification_shift_zyx_px": _finite_list_or_none(native_shift),
        "native_verification_yx_disagreement_px": (
            None if not np.isfinite(native_agreement) else native_agreement
        ),
        "z_shift_role": "diagnostic_only_fixed_world_z_is_locked",
    }


def write_registered_grid_channel_anchors(
    *,
    fixed_registration: Path,
    moving_baseline_registration: Path,
    output: Path,
    fixed_threshold: float,
    level: int = DEFAULT_LEVEL,
    patch_shape_zyx: tuple[int, int, int] = DEFAULT_PATCH_SHAPE_ZYX,
    native_verify_shape_zyx: tuple[int, int, int] = DEFAULT_NATIVE_VERIFY_SHAPE_ZYX,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Measure verified lateral anchors between matched registered tile sources."""
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(fixed_threshold) or fixed_threshold < 0:
        raise ValueError("fixed_threshold must be finite and nonnegative")
    if level < 0 or workers < 1:
        raise ValueError("level must be nonnegative and workers must be positive")
    fixed = json.loads(fixed_registration.read_text())
    moving = json.loads(moving_baseline_registration.read_text())
    fixed_by_tile = _records_by_tile(fixed, label="fixed registration")
    moving_by_tile = _records_by_tile(moving, label="moving baseline registration")
    if fixed_by_tile.keys() != moving_by_tile.keys():
        raise ValueError("Fixed and moving registrations must contain identical tile identities")
    base_spacing = np.asarray([fixed["spacing_um"][axis] for axis in "zyx"], dtype=np.float64)
    first_path = tile_image_path(fixed, next(iter(fixed_by_tile.values())))
    ratio = np.asarray(level_scale_ratio(first_path, level), dtype=np.float64)
    grid_spacing = base_spacing * ratio
    jobs: list[tuple[dict[str, Any], dict[str, Any], np.ndarray]] = []
    for tile in fixed_by_tile:
        fixed_record = fixed_by_tile[tile]
        moving_record = moving_by_tile[tile]
        shape = np.asarray(fixed_record["shape"][-3:], dtype=np.float64)
        for z_fraction in DEFAULT_Z_FRACTIONS:
            z = float(round(z_fraction * (shape[0] - 1)))
            for y_fraction in DEFAULT_YX_FRACTIONS:
                for x_fraction in DEFAULT_YX_FRACTIONS:
                    local_center = np.asarray(
                        [z, y_fraction * (shape[1] - 1), x_fraction * (shape[2] - 1)],
                        dtype=np.float64,
                    )
                    jobs.append((fixed_record, moving_record, local_center))

    def run(job: tuple[dict[str, Any], dict[str, Any], np.ndarray]) -> dict[str, Any]:
        fixed_record, moving_record, local_center = job
        return _measure_anchor(
            fixed_payload=fixed,
            moving_payload=moving,
            fixed_record=fixed_record,
            moving_record=moving_record,
            local_center_px_zyx=local_center,
            base_spacing_um_zyx=base_spacing,
            grid_spacing_um_zyx=grid_spacing,
            level=level,
            patch_shape_zyx=patch_shape_zyx,
            native_verify_shape_zyx=native_verify_shape_zyx,
            fixed_threshold=fixed_threshold,
        )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run, job): index for index, job in enumerate(jobs)}
        completed = 0
        indexed_rows: dict[int, dict[str, Any]] = {}
        for future in as_completed(futures):
            index = futures[future]
            indexed_rows[index] = future.result()
            completed += 1
            if progress is not None:
                accepted = sum(row["status"] == "accepted" for row in indexed_rows.values())
                progress(f"registered-grid anchors {completed}/{len(jobs)} accepted={accepted}")
        rows = [indexed_rows[index] for index in range(len(jobs))]

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "fixed_registration": str(fixed_registration.resolve()),
        "fixed_registration_sha256": sha256_file(fixed_registration),
        "moving_baseline_registration": str(moving_baseline_registration.resolve()),
        "moving_baseline_sha256": sha256_file(moving_baseline_registration),
        "fixed_reference_contract": "matched_registered_tile_source_used_by_tilewise_subtraction",
        "fixed_threshold": float(fixed_threshold),
        "threshold_method": "human_reviewed_threshold",
        "level": int(level),
        "base_spacing_um_zyx": base_spacing.tolist(),
        "grid_spacing_um_zyx": grid_spacing.tolist(),
        "patch_shape_zyx": list(patch_shape_zyx),
        "native_verify_shape_zyx": list(native_verify_shape_zyx),
        "z_fractions": list(DEFAULT_Z_FRACTIONS),
        "yx_fractions": list(DEFAULT_YX_FRACTIONS),
        "candidate_count": len(rows),
        "accepted_count": sum(row["status"] == "accepted" for row in rows),
        "rejected_count": sum(row["status"] != "accepted" for row in rows),
        "rows": rows,
    }
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return output.resolve()
