"""Place a separately acquired series against a reference acquisition on the same stage lattice.

A separate acquisition (an HCR 405 series) shares the reference acquisition's tile lattice, so its tile N images the
same block as reference tile N, but its stacks start at an unrecorded depth. For central tiles this measures the
depth lag and XY shift that register highpassed moving planes to the reference tile's planes: every lag at a coarse
pyramid level, then a fine scan around the best one. Moving planes are first resampled onto the reference XY pixel
spacing, since a separate acquisition may record a slightly different one, so a feature at moving physical position
p * moving_scale sits at reference local index p * moving_scale / reference_scale + o, and in world coordinates
T_moving = T_reference + o * reference_scale. A lag counts whole planes, so the recorded depth spacings may differ
only by less than one plane across the stack (HCR steps a nominal 1.8 um; recorded values such as 1.807 and
1.8065 um differ by metadata noise). Level factors are read from each store's multiscale metadata; the light-sheet
pyramids downsample XY only, so a depth lag usually counts level-0 planes.

The median per-tile correction shifts the moving positions. Validation uses the shifted positions alone, on tiles
held out of the median and spread away from the measured ones: at several world depths of each held-out tile pair,
a depth-lag scan around the predicted planes must peak at zero, with the XY shift the positions imply. Each check
also records how far its peak stands above the next-best lag.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
import zarr

from squisher_lightsheet.ngff import scale_translation

DEFAULT_TILE_COUNT = 9
DEFAULT_COARSE_LEVEL = 2
DEFAULT_FINE_LEVEL = 1
DEFAULT_COARSE_STEP = 4
DEFAULT_MINIMUM_OVERLAP_PLANES = 64
HIGHPASS_SIGMA_PX = 10.0
CHECK_LAGS = tuple(range(-8, 9, 2))
CHECK_XY_TOLERANCE_PX = 1.0
DEFAULT_VALIDATION_TILE_COUNT = 3
VALIDATION_DEPTH_FRACTIONS = (0.25, 0.5, 0.75)
TILE_PATTERN = re.compile(r"\.(\d+)\.ome\.(?:tiff?|zarr)$")
SUFFIX_PATTERN = re.compile(r"\.ome\.(?:tiff?|zarr)$")


def tile_number(name: str) -> str:
    match = TILE_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Tile name {name!r} has no numeric .NNN.ome.<ext> suffix")
    return match.group(1)


def _vector(record: dict[str, Any], field: str) -> np.ndarray:
    return np.asarray([record[field][axis] for axis in "zyx"], dtype=np.float64)


def _store(tile_dir: Path, record: dict[str, Any]) -> zarr.Group:
    path = tile_dir / f"{SUFFIX_PATTERN.sub('', record['tile'])}.ome.zarr"
    if not path.is_dir():
        raise FileNotFoundError(f"Tile store for {record['tile']!r} is missing: {path}")
    return zarr.open_group(str(path), mode="r")


def _level_factor_zyx(group: zarr.Group, level: int) -> np.ndarray:
    _names, base, *_ = scale_translation(group, dataset_index=0)
    _names, scaled, *_ = scale_translation(group, dataset_index=level)
    return np.asarray(scaled[-3:], dtype=np.float64) / np.asarray(base[-3:], dtype=np.float64)


def _channel_volume(group: zarr.Group, level: int, channel: int) -> zarr.Array:
    array = group[str(level)]
    if array.ndim == 4:
        return array[channel]
    if array.ndim == 3 and channel == 0:
        return array
    raise ValueError(f"Level {level} of {group.store} has shape {array.shape}; cannot select channel {channel}")


def _prepare(plane: np.ndarray) -> np.ndarray:
    image = np.log1p(np.asarray(plane, dtype=np.float32))
    image = image - ndi.gaussian_filter(image, HIGHPASS_SIGMA_PX)
    return (image - image.mean()) / max(float(image.std()), 1e-6)


def _register_planes(
    moving: list[np.ndarray], reference: list[np.ndarray], moving_to_reference_yx: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean phase-correlation error and median YX shift in reference pixels (reference = resampled moving shifted by it).

    moving_to_reference_yx is the moving XY pixel spacing over the reference's: moving pixel p lies at reference
    pixel p * ratio from the shared origin.
    """
    shifts, errors = [], []
    for moving_plane, reference_plane in zip(moving, reference, strict=True):
        resampled = ndi.affine_transform(
            np.asarray(moving_plane, dtype=np.float32),
            1.0 / moving_to_reference_yx,
            output_shape=reference_plane.shape,
            order=1,
            mode="nearest",
        )
        shift, error, _ = phase_cross_correlation(
            _prepare(reference_plane), _prepare(resampled), upsample_factor=4, normalization=None
        )
        shifts.append(shift)
        errors.append(error)
    return float(np.mean(errors)), np.median(np.asarray(shifts), axis=0)


def _depths(moving_planes: int, reference_planes: int, lag: int, minimum_overlap: int) -> np.ndarray | None:
    low, high = max(0, -lag), min(moving_planes, reference_planes - lag)
    if high - low < minimum_overlap:
        return None
    return np.linspace(low, high - 1, 5)[1:4].astype(int)


def _scan(
    moving: Any, reference: Any, lags: range, minimum_overlap: int, moving_to_reference_yx: np.ndarray
) -> dict[int, tuple[float, np.ndarray]]:
    results = {}
    for lag in lags:
        depths = _depths(moving.shape[0], reference.shape[0], lag, minimum_overlap)
        if depths is None:
            continue
        results[lag] = _register_planes(
            [np.asarray(moving[z]) for z in depths],
            [np.asarray(reference[z + lag]) for z in depths],
            moving_to_reference_yx,
        )
    if not results:
        raise ValueError(f"No depth lag in {lags} leaves {minimum_overlap} overlapping planes")
    return results


def measure_tile_offset(
    moving_group: zarr.Group,
    reference_group: zarr.Group,
    *,
    moving_channel: int,
    reference_channel: int,
    moving_to_reference_yx: np.ndarray,
    coarse_level: int = DEFAULT_COARSE_LEVEL,
    fine_level: int = DEFAULT_FINE_LEVEL,
    coarse_step: int = DEFAULT_COARSE_STEP,
    minimum_overlap_planes: int = DEFAULT_MINIMUM_OVERLAP_PLANES,
) -> dict[str, Any]:
    """Return the moving-to-reference offset of one tile pair in reference level-0 voxels."""
    coarse_factor = _level_factor_zyx(moving_group, coarse_level)
    fine_factor = _level_factor_zyx(moving_group, fine_level)
    for level, factor in ((coarse_level, coarse_factor), (fine_level, fine_factor)):
        if not np.allclose(_level_factor_zyx(reference_group, level), factor):
            raise ValueError(f"Moving and reference pyramids differ at level {level}")
    moving_coarse = np.asarray(_channel_volume(moving_group, coarse_level, moving_channel))
    reference_coarse = np.asarray(_channel_volume(reference_group, coarse_level, reference_channel))
    coarse = _scan(
        moving_coarse,
        reference_coarse,
        range(-moving_coarse.shape[0] + 1, reference_coarse.shape[0], coarse_step),
        minimum_overlap_planes,
        moving_to_reference_yx,
    )
    coarse_lag = min(coarse, key=lambda lag: coarse[lag][0])
    center = int(round(coarse_lag * coarse_factor[0] / fine_factor[0]))
    margin = int(np.ceil(coarse_step * coarse_factor[0] / fine_factor[0])) + 2
    fine = _scan(
        _channel_volume(moving_group, fine_level, moving_channel),
        _channel_volume(reference_group, fine_level, reference_channel),
        range(center - margin, center + margin + 1),
        minimum_overlap_planes,
        moving_to_reference_yx,
    )
    lag = min(fine, key=lambda value: fine[value][0])
    error, shift_yx = fine[lag]
    offset = np.array([lag, *shift_yx], dtype=np.float64) * fine_factor
    return {
        "coarse_lag_planes": int(coarse_lag),
        "coarse_error": coarse[coarse_lag][0],
        "coarse_error_median": float(np.median([result[0] for result in coarse.values()])),
        "fine_lag_planes": int(lag),
        "fine_error": error,
        "fine_shift_yx_px": shift_yx.tolist(),
        "coarse_level_factor_zyx": coarse_factor.tolist(),
        "fine_level_factor_zyx": fine_factor.tolist(),
        "offset_level0_zyx": offset.tolist(),
    }


def _central_tiles(records: dict[str, dict[str, Any]], count: int) -> list[str]:
    centers = {}
    for key, record in records.items():
        shape = np.asarray(record["shape"][-3:], dtype=np.float64)
        centers[key] = _vector(record, "translation_um") + _vector(record, "scale_um") * shape / 2.0
    centroid = np.mean(list(centers.values()), axis=0)
    return sorted(records, key=lambda key: float(np.linalg.norm((centers[key] - centroid)[1:])))[:count]


def _validation_tiles(records: dict[str, dict[str, Any]], measured: list[str], count: int) -> list[str]:
    """Held-out tiles spread away from the measured ones: repeatedly the tile farthest in XY from those chosen."""
    centers = {}
    for key, record in records.items():
        shape = np.asarray(record["shape"][-3:], dtype=np.float64)
        centers[key] = (_vector(record, "translation_um") + _vector(record, "scale_um") * shape / 2.0)[1:]
    chosen = list(measured)
    held_out: list[str] = []
    for _ in range(min(count, len(records) - len(measured))):
        candidates = sorted(set(records) - set(chosen))
        farthest = max(
            candidates, key=lambda key: min(float(np.linalg.norm(centers[key] - centers[other])) for other in chosen)
        )
        chosen.append(farthest)
        held_out.append(farthest)
    return held_out


def _require_compatible_scales(key: str, moving: dict[str, Any], reference: dict[str, Any]) -> np.ndarray:
    """The moving-to-reference XY spacing ratio, after checking orientation and depth spacing agree."""
    moving_scale, reference_scale = _vector(moving, "scale_um"), _vector(reference, "scale_um")
    if np.any(np.sign(moving_scale) != np.sign(reference_scale)):
        raise ValueError(f"Tile {key} axis orientation differs between the moving and reference acquisitions")
    planes = max(int(moving["shape"][-3]), int(reference["shape"][-3]))
    if planes * abs(moving_scale[0] - reference_scale[0]) >= abs(reference_scale[0]):
        raise ValueError(
            f"Tile {key} depth spacings {moving_scale[0]} and {reference_scale[0]} um drift apart by a plane or more "
            f"across {planes} planes"
        )
    return moving_scale[1:] / reference_scale[1:]


def _check_pair(
    moving_record: dict[str, Any],
    reference_record: dict[str, Any],
    moving_group: zarr.Group,
    reference_group: zarr.Group,
    *,
    moving_channel: int,
    reference_channel: int,
    fine_level: int,
    moving_to_reference_yx: np.ndarray,
    depth_fraction: float,
) -> dict[str, Any]:
    factor = _level_factor_zyx(moving_group, fine_level)
    scale = _vector(reference_record, "scale_um")
    moving_t = _vector(moving_record, "translation_um")
    reference_t = _vector(reference_record, "translation_um")
    moving = _channel_volume(moving_group, fine_level, moving_channel)
    reference = _channel_volume(reference_group, fine_level, reference_channel)
    step_z = scale[0] * factor[0]
    moving_z = (np.arange(moving.shape[0]) * step_z) + moving_t[0]
    reference_z = (np.arange(reference.shape[0]) * step_z) + reference_t[0]
    low = max(min(moving_z), min(reference_z))
    high = min(max(moving_z), max(reference_z))
    if high <= low:
        raise ValueError(f"Shifted tile {moving_record['tile']} does not overlap its reference in depth")
    # Snap the depth to a moving plane so both indices name the same world plane, not two rounded halves of one.
    moving_index = int(round((low + depth_fraction * (high - low) - moving_t[0]) / step_z))
    world_z = moving_t[0] + moving_index * step_z
    reference_index = int(round((world_z - reference_t[0]) / step_z))
    expected_yx = (moving_t[1:] - reference_t[1:]) / (scale[1:] * factor[1:])
    scores = {}
    for lag in CHECK_LAGS:
        index = reference_index + lag
        if 0 <= index < reference.shape[0]:
            scores[lag] = _register_planes(
                [np.asarray(moving[moving_index])], [np.asarray(reference[index])], moving_to_reference_yx
            )
    ranked = sorted(scores, key=lambda lag: scores[lag][0])
    best = ranked[0]
    shift = scores[best][1]
    return {
        "tile": moving_record["tile"],
        "world_z_um": float(world_z),
        "best_lag_planes": int(best),
        "peak_margin": float(scores[ranked[1]][0] - scores[best][0]) if len(ranked) > 1 else None,
        "shift_yx_px": shift.tolist(),
        "expected_shift_yx_px": expected_yx.tolist(),
        "errors_by_lag": {str(lag): scores[lag][0] for lag in scores},
        "passed": bool(best == 0 and np.all(np.abs(shift - expected_yx) <= CHECK_XY_TOLERANCE_PX)),
    }


def shifted_acquisition_positions(
    moving_payload: dict[str, Any], correction_zyx_um: list[float] | np.ndarray, *, summary_path: Path,
) -> dict[str, Any]:
    """Apply a measured physical translation once, preserving independent native tile geometry."""
    correction = np.asarray(correction_zyx_um, dtype=np.float64)
    if correction.shape != (3,) or not np.all(np.isfinite(correction)):
        raise ValueError(f"Acquisition correction must be three finite ZYX values: {correction_zyx_um}")
    shifted = deepcopy(moving_payload)
    for record in shifted["tiles"]:
        record["translation_um"] = {
            axis: float(value)
            for axis, value in zip("zyx", _vector(record, "translation_um") + correction, strict=True)
        }
    shifted["source"] = f"{moving_payload.get('source', 'position artifact')} + median per-tile acquisition offset"
    shifted["derived_by"] = "lightsheet cross-register acquisition-offset"
    shifted.setdefault("diagnostics", {})["acquisition_offset"] = str(summary_path.resolve())
    return shifted


def run_acquisition_offset(
    *,
    moving_position: Path,
    reference_position: Path,
    moving_tile_dir: Path,
    reference_tile_dir: Path,
    output_dir: Path,
    output_position: Path,
    moving_channel: int = 0,
    reference_channel: int,
    tiles: list[str] | None = None,
    tile_count: int = DEFAULT_TILE_COUNT,
    validation_tile_count: int = DEFAULT_VALIDATION_TILE_COUNT,
    coarse_level: int = DEFAULT_COARSE_LEVEL,
    fine_level: int = DEFAULT_FINE_LEVEL,
    coarse_step: int = DEFAULT_COARSE_STEP,
    minimum_overlap_planes: int = DEFAULT_MINIMUM_OVERLAP_PLANES,
) -> dict[str, Any]:
    """Measure per-tile offsets and validate them on held-out tile pairs; publish the shifted positions if they pass."""
    moving_payload = json.loads(moving_position.read_text())
    reference_payload = json.loads(reference_position.read_text())
    moving_by_tile = {tile_number(record["tile"]): record for record in moving_payload["tiles"]}
    reference_by_tile = {tile_number(record["tile"]): record for record in reference_payload["tiles"]}
    shared = {key: reference_by_tile[key] for key in moving_by_tile.keys() & reference_by_tile.keys()}
    if tiles is None:
        tiles = _central_tiles(shared, tile_count)
    if not tiles:
        raise ValueError("No tiles to measure: the position artifacts share no tile numbers")
    missing = sorted(set(tiles) - set(shared))
    if missing:
        raise ValueError(f"Tiles {missing} are not in both position artifacts")
    held_out = _validation_tiles(shared, tiles, validation_tile_count)
    if not held_out:
        raise ValueError(f"No shared tile is left to validate against: all {len(shared)} were measured")
    ratios = {key: _require_compatible_scales(key, moving_by_tile[key], shared[key]) for key in [*tiles, *held_out]}

    rows = []
    for key in tiles:
        moving_record, reference_record = moving_by_tile[key], shared[key]
        result = measure_tile_offset(
            _store(moving_tile_dir, moving_record),
            _store(reference_tile_dir, reference_record),
            moving_channel=moving_channel,
            reference_channel=reference_channel,
            moving_to_reference_yx=ratios[key],
            coarse_level=coarse_level,
            fine_level=fine_level,
            coarse_step=coarse_step,
            minimum_overlap_planes=minimum_overlap_planes,
        )
        correction = (
            _vector(reference_record, "translation_um")
            + np.asarray(result["offset_level0_zyx"]) * _vector(reference_record, "scale_um")
            - _vector(moving_record, "translation_um")
        )
        rows.append(
            {
                "tile": key,
                **result,
                "moving_to_reference_yx": ratios[key].tolist(),
                "correction_zyx_um": correction.tolist(),
            }
        )
        print(
            f"acquisition-offset tile {key}: lag {result['fine_lag_planes']} "
            f"shift_yx {np.round(result['fine_shift_yx_px'], 2).tolist()} "
            f"correction_zyx_um {np.round(correction, 2).tolist()}",
            flush=True,
        )
    corrections = np.asarray([row["correction_zyx_um"] for row in rows])
    median = np.median(corrections, axis=0)
    mad = np.median(np.abs(corrections - median), axis=0)

    shifted = shifted_acquisition_positions(
        moving_payload, median, summary_path=output_dir / "acquisition-offset.json",
    )
    shifted_by_tile = {tile_number(record["tile"]): record for record in shifted["tiles"]}
    checks = [
        _check_pair(
            shifted_by_tile[key],
            shared[key],
            _store(moving_tile_dir, moving_by_tile[key]),
            _store(reference_tile_dir, shared[key]),
            moving_channel=moving_channel,
            reference_channel=reference_channel,
            fine_level=fine_level,
            moving_to_reference_yx=ratios[key],
            depth_fraction=fraction,
        )
        for key in held_out
        for fraction in VALIDATION_DEPTH_FRACTIONS
    ]
    validation = {"held_out_tiles": held_out, "checks": checks, "passed": all(check["passed"] for check in checks)}
    summary = {
        "artifact_type": "squisher_lightsheet.acquisition_offset.v2",
        "moving_position": str(moving_position.resolve()),
        "reference_position": str(reference_position.resolve()),
        "moving_channel": moving_channel,
        "reference_channel": reference_channel,
        "tiles": rows,
        "median_correction_zyx_um": median.tolist(),
        "correction_mad_zyx_um": mad.tolist(),
        "validation": validation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "acquisition-offset.json", summary)
    # A failed validation is a measured result (a comparison of candidate references needs it); only passing
    # positions are published.
    if not validation["passed"]:
        return summary
    _write_json(output_position, shifted)
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
