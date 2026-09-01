#!/usr/bin/env python
"""Canonicalize CL/CR right-to-left placement artifacts for fusion.

This script starts after side-internal registration and CL/CR Method8 sampling
have already been measured. It materializes the stable contract we used for
230Tnc:

1. keep each side's own Li/phase-correlation tile geometry;
2. place CR into CL space by metadata centroid + z-median phase correlation +
   median accepted Method8 local translation;
3. write deconvolved CZYX position JSON plus canonical registration.json with
   unique CL/CR zarr basenames for the fuser;
4. optionally render a no-blend z-median dumb stitch and print fusion/movie
   commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.geometry import orient_plane_yx, signed_bounds
from squisher_lightsheet.ngff import open_level_array


DIMENSIONS = ("z", "y", "x")
IDENTITY_AFFINE = {
    "dims": ["x_in", "x_out"],
    "coords": {"x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]},
    "matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.exists()
    payload: dict[str, Any] = {"path": str(path), "resolved": str(resolved), "exists": exists}
    if not exists:
        return payload
    stat = resolved.stat()
    payload.update({"size_bytes": stat.st_size, "mtime_unix": stat.st_mtime, "is_dir": resolved.is_dir()})
    if resolved.is_file():
        payload["sha256"] = _sha256_file(resolved)
    return payload


def _git_state(path: Path) -> dict[str, Any]:
    try:
        root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", root, "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}
    return {"available": True, "root": root, "commit": commit, "dirty": bool(status.strip()), "status_short": status.splitlines()}


def _run_provenance(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "script": _path_fingerprint(Path(__file__)),
        "git": _git_state(Path(__file__).resolve().parent),
        "inputs": {name: _path_fingerprint(path) for name, path in paths.items()},
    }


def _vector(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([float(values[dim]) for dim in DIMENSIONS], dtype=np.float64)


def _dict_zyx(values: np.ndarray) -> dict[str, float]:
    return {dim: float(value) for dim, value in zip(DIMENSIONS, values, strict=True)}


def _parse_shape(value: str) -> list[int]:
    shape = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(shape) != 4:
        raise argparse.ArgumentTypeError("expected C,Z,Y,X shape, for example 2,1361,1920,1920")
    if any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError(f"shape values must be positive: {shape}")
    return shape


def _parse_channels(value: str) -> list[str]:
    channels = [part.strip() for part in value.split(",") if part.strip()]
    if not channels:
        raise argparse.ArgumentTypeError("expected at least one channel")
    return channels


def _source_zarr_name(record: dict[str, Any]) -> str:
    source_tile = str(record.get("source_tile") or record["tile"])
    return source_tile.removesuffix(".ome.tif").removesuffix(".ome.zarr") + ".ome.zarr"


def _side_from_tile(tile_name: str) -> str:
    if "-CL-" in tile_name:
        return "L"
    if "-CR-" in tile_name:
        return "R"
    raise ValueError(f"cannot infer CL/CR side from {tile_name!r}")


def _side_token(side_label: str) -> str:
    if side_label == "CL":
        return "-CL-"
    if side_label == "CR":
        return "-CR-"
    raise ValueError(f"unsupported side label {side_label!r}; expected CL or CR")


def _validate_side_records(records: list[dict[str, Any]], side_label: str, path: Path) -> None:
    token = _side_token(side_label)
    bad = [record.get("tile", record.get("source_tile", "")) for record in records if token not in _source_zarr_name(record)]
    if bad:
        preview = ", ".join(map(str, bad[:5]))
        raise ValueError(f"{path} was labeled {side_label}, but {len(bad)} records do not contain {token}: {preview}")


def _load_side_records(path: Path, side_label: str) -> list[dict[str, Any]]:
    payload = _read_json(path)
    records = [dict(record) for record in payload["tiles"]]
    if not records:
        raise ValueError(f"{path} contains no tile records")
    _validate_side_records(records, side_label, path)
    for record in records:
        record.setdefault("source_side", side_label)
    return records


def _validate_phasecorr(phasecorr: dict[str, Any], path: Path) -> None:
    fixed = phasecorr.get("fixed")
    moving = phasecorr.get("moving")
    if fixed != "CL" or moving != "CR":
        raise ValueError(f"{path} must be CL-fixed/CR-moving phasecorr; found fixed={fixed!r}, moving={moving!r}")


def _validate_method8(summary: dict[str, Any], path: Path) -> None:
    bad: list[str] = []
    for row in summary.get("windows", []):
        if row.get("status") != "accepted":
            continue
        fixed_tile = str(row.get("fixed_tile", ""))
        moving_tile = str(row.get("moving_tile", ""))
        if "-CL-" not in fixed_tile or "-CR-" not in moving_tile:
            bad.append(f"fixed={fixed_tile!r} moving={moving_tile!r} quadrant={row.get('quadrant')!r}")
    if bad:
        raise ValueError(f"{path} contains accepted Method8 rows that are not CL-fixed/CR-moving: {bad[:5]}")


def _metadata_centers_from_manifest(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    centers = manifest.get("metadata_centers_um_zyx")
    if not isinstance(centers, dict):
        raise ValueError("dumb-stitch manifest is missing metadata_centers_um_zyx")
    return {side: np.asarray(value, dtype=np.float64) for side, value in centers.items()}


def _shape_zyx(record: dict[str, Any], default_shape_czyx: list[int]) -> np.ndarray:
    shape = record.get("shape")
    axes = str(record.get("axes", ""))
    if isinstance(shape, list) and axes == "ZYX":
        return np.asarray(shape, dtype=np.float64)
    if isinstance(shape, list) and axes == "CZYX":
        return np.asarray(shape[1:], dtype=np.float64)
    return np.asarray(default_shape_czyx[1:], dtype=np.float64)


def _record_centers(records: list[dict[str, Any]], default_shape_czyx: list[int]) -> np.ndarray:
    centers = []
    for record in records:
        low, high = signed_bounds(
            _vector(record, "translation_um"),
            _vector(record, "scale_um"),
            _shape_zyx(record, default_shape_czyx),
        )
        centers.append((low + high) / 2.0)
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)


def _common_scale(records: list[dict[str, Any]], side: str) -> np.ndarray:
    first = _vector(records[0], "scale_um")
    for record in records[1:]:
        current = _vector(record, "scale_um")
        if not np.allclose(current, first, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{side} records have inconsistent signed scales: {first} versus {current}")
    return first


def _method8_accepted_local_translations(summary: dict[str, Any]) -> np.ndarray:
    rows = [row for row in summary.get("windows", []) if row.get("status") == "accepted"]
    translations = [row.get("local_translation_zyx") for row in rows if row.get("local_translation_zyx") is not None]
    if not translations:
        raise ValueError("Method8 summary contains no accepted local_translation_zyx rows")
    return np.asarray(translations, dtype=np.float64)


def _translation_stats(translations: np.ndarray) -> dict[str, Any]:
    if translations.size == 0:
        return {"count": 0}
    q25, q75 = np.percentile(translations, [25.0, 75.0], axis=0)
    median = np.median(translations, axis=0)
    mad = np.median(np.abs(translations - median), axis=0)
    return {
        "count": int(translations.shape[0]),
        "median_zyx": median.astype(float).tolist(),
        "mean_zyx": np.mean(translations, axis=0).astype(float).tolist(),
        "std_zyx": np.std(translations, axis=0).astype(float).tolist(),
        "iqr_zyx": (q75 - q25).astype(float).tolist(),
        "mad_zyx": mad.astype(float).tolist(),
        "min_zyx": np.min(translations, axis=0).astype(float).tolist(),
        "max_zyx": np.max(translations, axis=0).astype(float).tolist(),
    }


def _method8_provenance(summary: dict[str, Any]) -> dict[str, Any]:
    windows = summary.get("windows", [])
    accepted = [row for row in windows if row.get("status") == "accepted" and row.get("local_translation_zyx") is not None]
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    accepted_windows = []
    for index, row in enumerate(windows):
        if row.get("status") != "accepted" or row.get("local_translation_zyx") is None:
            continue
        translation = np.asarray(row["local_translation_zyx"], dtype=np.float64)
        fixed_tile = str(row.get("fixed_tile", ""))
        moving_tile = str(row.get("moving_tile", ""))
        grouped[(fixed_tile, moving_tile)].append(translation)
        accepted_windows.append(
            {
                "index": index,
                "fixed_tile": fixed_tile,
                "moving_tile": moving_tile,
                "quadrant": row.get("quadrant"),
                "local_translation_zyx": translation.astype(float).tolist(),
                "full_preseed_translation_zyx": row.get("full_preseed_translation_zyx"),
                "fixed_start_zyx": row.get("fixed_start_zyx"),
                "moving_start_zyx": row.get("moving_start_zyx"),
            }
        )

    per_pair = []
    for (fixed_tile, moving_tile), values in sorted(grouped.items()):
        per_pair.append(
            {
                "fixed_tile": fixed_tile,
                "moving_tile": moving_tile,
                "stats": _translation_stats(np.asarray(values, dtype=np.float64)),
            }
        )

    return {
        "artifact_type": summary.get("artifact_type"),
        "coordinate_frame": (
            "local_translation_zyx is the Method8 moving-to-fixed correction in level-0 voxel units; "
            "the canonical R-to-L workflow multiplies the median accepted value by scale_um_zyx and adds "
            "it to CR after metadata-centroid placement and z-median phase correlation."
        ),
        "cache_config": summary.get("cache_config"),
        "status_counts": summary.get("status_counts", dict(Counter(row.get("status") for row in windows))),
        "reason_counts": summary.get("reason_counts", dict(Counter(row.get("rejection_reason") for row in windows))),
        "task_count": summary.get("task_count"),
        "completed_window_count": summary.get("completed_window_count", len(windows)),
        "accepted_window_count": len(accepted),
        "global_local_translation_stats_px_zyx": _translation_stats(_method8_accepted_local_translations(summary)),
        "pair_count": summary.get("pair_count"),
        "pairs": summary.get("pairs"),
        "per_pair_local_translation_stats_px_zyx": per_pair,
        "accepted_windows": accepted_windows,
    }


def _zarr_level_shape(path: Path, level: int = 0) -> tuple[int, ...] | None:
    if not path.exists():
        return None
    return tuple(int(value) for value in open_level_array(path, level=level).shape)


def _infer_shape_czyx(records: list[dict[str, Any]], deconv_root: Path, channels: list[str]) -> list[int] | None:
    inferred: set[tuple[int, ...]] = set()
    for record in records:
        shape = record.get("shape")
        axes = str(record.get("axes", ""))
        if isinstance(shape, list) and axes == "CZYX":
            inferred.add(tuple(int(value) for value in shape))
            continue
        if isinstance(shape, list) and axes == "ZYX":
            inferred.add((len(channels), *(int(value) for value in shape)))
            continue
        zarr_shape = _zarr_level_shape(deconv_root / _source_zarr_name(record), level=0)
        if zarr_shape is not None:
            inferred.add(zarr_shape)
    if not inferred:
        return None
    if len(inferred) != 1:
        raise ValueError(f"inconsistent inferred CZYX shapes: {sorted(inferred)}")
    return list(next(iter(inferred)))


def _resolve_shape_czyx(
    cli_shape_czyx: list[int] | None,
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    deconv_root: Path,
    channels: list[str],
) -> list[int]:
    inferred = _infer_shape_czyx(left_records + right_records, deconv_root, channels)
    if inferred is None:
        if cli_shape_czyx is None:
            raise ValueError("could not infer CZYX shape from records or deconvolved OME-Zarrs; pass --shape-czyx")
        inferred = cli_shape_czyx
    elif cli_shape_czyx is not None and cli_shape_czyx != inferred:
        raise ValueError(f"--shape-czyx {cli_shape_czyx} disagrees with inferred deconvolved CZYX shape {inferred}")
    if inferred[0] != len(channels):
        raise ValueError(f"channel list {channels} has {len(channels)} entries but inferred C dimension is {inferred[0]}")
    return inferred


def _tile_sort_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("source_side", "")), _source_zarr_name(record))


def build_clspace_position_payload(
    *,
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    manifest: dict[str, Any],
    phasecorr: dict[str, Any],
    method8_summary: dict[str, Any],
    left_position: Path,
    right_position: Path,
    manifest_path: Path,
    phasecorr_path: Path,
    method8_summary_path: Path,
    shape_czyx: list[int],
    channels: list[str],
    deconv_root: Path | None,
    artifact_stem: str,
    run_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_phasecorr(phasecorr, phasecorr_path)
    _validate_method8(method8_summary, method8_summary_path)
    metadata_centers = _metadata_centers_from_manifest(manifest)
    if "CL" not in metadata_centers or "CR" not in metadata_centers:
        raise ValueError("metadata centers must contain CL and CR")

    left_scale_um = _common_scale(left_records, "CL")
    right_scale_um = _common_scale(right_records, "CR")
    if not np.allclose(np.abs(left_scale_um), np.abs(right_scale_um), rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"CL/CR absolute scales differ: {np.abs(left_scale_um)} versus {np.abs(right_scale_um)}"
        )
    phase_yx_um = np.asarray(phasecorr["phase_shift_to_apply_to_cr_yx_um"], dtype=np.float64)
    phase_shift_um = np.asarray([0.0, phase_yx_um[0], phase_yx_um[1]], dtype=np.float64)
    accepted_translations = _method8_accepted_local_translations(method8_summary)
    method8_median_px = np.median(accepted_translations, axis=0)
    method8_mean_px = np.mean(accepted_translations, axis=0)
    method8_median_um = method8_median_px * right_scale_um
    method8_mean_um = method8_mean_px * right_scale_um

    left_offset = metadata_centers["CL"] - _record_centers(left_records, shape_czyx)
    right_offset = metadata_centers["CR"] - _record_centers(right_records, shape_czyx)
    net_right_shift = phase_shift_um + method8_median_um
    right_total_offset = right_offset + net_right_shift

    tiles: list[dict[str, Any]] = []
    channel_indexes = list(range(len(channels)))
    for side_name, records, offset in (
        ("CL", left_records, left_offset),
        ("CR", right_records, right_total_offset),
    ):
        for record in records:
            tile_name = _source_zarr_name(record)
            updated = {
                "tile": tile_name,
                "path": str(deconv_root / tile_name) if deconv_root is not None else str(record["path"]),
                "side": _side_from_tile(tile_name),
                "source_side": side_name,
                "translation_um": _dict_zyx(_vector(record, "translation_um") + offset),
                "scale_um": _dict_zyx(_vector(record, "scale_um")),
                "spacing_um": _dict_zyx(np.abs(_vector(record, "scale_um"))),
                "shape": shape_czyx,
                "axes": "CZYX",
                "channels": channels,
                "tracks": [{"slug": "track0", "track_id": "all", "channels": channel_indexes, "channel_names": channels}],
                "source_tile": str(record.get("source_tile", record["tile"])),
                "source_path": str(record.get("source_path", record.get("path", ""))),
            }
            tiles.append(updated)
    tiles.sort(key=_tile_sort_key)

    return {
        "schema_version": 1,
        "artifact_type": f"{artifact_stem}.clspace_phasecorr_method8_median_deconv_czyx_positions.v1",
        "units": "micrometer",
        "created_at_unix": time.time(),
        "source": (
            "CL/CR side-internal positions; CR placed in CL space by metadata centroid, "
            "z-median phase correlation, and median accepted Method8 local translation"
        ),
        "run_provenance": run_provenance,
        "inputs": {
            "manifest": str(manifest_path),
            "phasecorr": str(phasecorr_path),
            "method8_summary": str(method8_summary_path),
            "optimized_positions": {"CL": str(left_position), "CR": str(right_position)},
            "deconvolved_root": None if deconv_root is None else str(deconv_root),
        },
        "alignment": {
            "fixed_side": "CL",
            "moving_side": "CR",
            "method8_translation_source": "median accepted local_translation_zyx from Method8 summary",
            "method8_coordinate_frame": (
                "Method8 accepted local_translation_zyx is interpreted as a moving-CR-to-fixed-CL "
                "correction in level-0 voxel units."
            ),
            "accepted_method8_count": int(accepted_translations.shape[0]),
            "scale_um_zyx": right_scale_um.astype(float).tolist(),
            "initial_phasecorr_shift_px_yx": [float(v) for v in phasecorr["phase_shift_to_apply_to_cr_yx_px"]],
            "initial_phasecorr_shift_um_zyx": phase_shift_um.astype(float).tolist(),
            "method8_median_shift_px_zyx": method8_median_px.astype(float).tolist(),
            "method8_median_shift_um_zyx": method8_median_um.astype(float).tolist(),
            "method8_mean_shift_px_zyx": method8_mean_px.astype(float).tolist(),
            "method8_mean_shift_um_zyx": method8_mean_um.astype(float).tolist(),
            "net_cr_shift_after_centroid_um_zyx": net_right_shift.astype(float).tolist(),
            "cl_global_centroid_offset_um_zyx": left_offset.astype(float).tolist(),
            "cr_global_centroid_plus_alignment_offset_um_zyx": right_total_offset.astype(float).tolist(),
        },
        "method8_summary": _method8_provenance(method8_summary),
        "tiles": tiles,
    }


def build_identity_registration_payload(
    position_payload: dict[str, Any],
    *,
    deconv_root: Path,
    artifact_stem: str,
) -> dict[str, Any]:
    spacing = deepcopy(position_payload["tiles"][0]["spacing_um"])
    tiles = []
    for record in position_payload["tiles"]:
        tiles.append(
            {
                "tile": record["tile"],
                "path": str(deconv_root / record["tile"]),
                "source_view": "C" + str(record["side"]),
                "stage_translation_um": deepcopy(record["translation_um"]),
                "stage_scale_um": deepcopy(record["scale_um"]),
                "spacing_um": deepcopy(record["spacing_um"]),
                "shape": deepcopy(record["shape"]),
                "axes": deepcopy(record["axes"]),
                "channels": deepcopy(record["channels"]),
                "tracks": deepcopy(record["tracks"]),
                "registered_affine": deepcopy(IDENTITY_AFFINE),
            }
        )
    return {
        "input_dir": str(deconv_root),
        "metadata_transform_key": "stage_metadata",
        "registered_transform_key": "registered_affine",
        "spacing_um": spacing,
        "metrics": {
            "artifact_type": f"{artifact_stem}.identity_registration_for_deconv_czyx.v1",
            "registered_affine_note": (
                "Identity affine; CL/CR placement is already baked into each tile's stage_translation_um."
            ),
        },
        "tiles": tiles,
    }


def _stretch_uint8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    values = finite[finite > 0]
    if values.size == 0:
        values = finite
    low, high = np.percentile(values, [0.5, 99.8]) if values.size else (0.0, 1.0)
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return np.clip((image - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)


def render_zmedian_dumb_stitch(
    position_payload: dict[str, Any],
    *,
    output_dir: Path,
    channel: int,
    level: int,
) -> dict[str, Any]:
    import zarr
    from PIL import Image

    level_name = str(level)
    pixel_um_yx: np.ndarray | None = None
    prepared = []
    bounds_min = np.full(2, np.inf, dtype=np.float64)
    bounds_max = np.full(2, -np.inf, dtype=np.float64)
    for record in position_payload["tiles"]:
        path = Path(record["path"])
        level_path = path / level_name
        if not level_path.exists():
            raise ValueError(f"{path} does not contain OME-Zarr level {level}; render with an existing pyramid level")
        array = zarr.open(str(level_path), mode="r")
        if len(array.shape) != 4:
            raise ValueError(f"{level_path} must be CZYX, found shape {array.shape}")
        if channel < 0 or channel >= int(array.shape[0]):
            raise ValueError(f"channel {channel} is outside {level_path} C dimension {array.shape[0]}")
        full_shape_zyx = np.asarray(record["shape"][1:], dtype=np.float64)
        level_shape_yx = np.asarray(array.shape[-2:], dtype=np.float64)
        level_factor_yx = full_shape_zyx[-2:] / level_shape_yx
        scale = np.asarray([record["scale_um"][dim] for dim in DIMENSIONS], dtype=np.float64)
        current_pixel_um_yx = np.abs(scale[1:]) * level_factor_yx
        if pixel_um_yx is None:
            pixel_um_yx = current_pixel_um_yx
        elif not np.allclose(pixel_um_yx, current_pixel_um_yx, rtol=1e-6, atol=1e-6):
            raise ValueError(f"inconsistent level {level} pixel spacing: {pixel_um_yx} versus {current_pixel_um_yx}")
        start = np.asarray([record["translation_um"][dim] for dim in DIMENSIONS], dtype=np.float64)
        tile_min, tile_max = signed_bounds(start, scale, full_shape_zyx)
        bounds_min = np.minimum(bounds_min, tile_min[1:])
        bounds_max = np.maximum(bounds_max, tile_max[1:])
        prepared.append((record, array, tile_min, scale))

    if pixel_um_yx is None:
        raise ValueError("position payload contains no tiles")
    height = int(math.ceil((bounds_max[0] - bounds_min[0]) / pixel_um_yx[0]))
    width = int(math.ceil((bounds_max[1] - bounds_min[1]) / pixel_um_yx[1]))
    mosaics = {"L": np.zeros((height, width), dtype=np.float32), "R": np.zeros((height, width), dtype=np.float32)}
    tiles = []
    for record, array, tile_min, scale in prepared:
        side = str(record["side"])
        y0 = int(round((tile_min[1] - bounds_min[0]) / pixel_um_yx[0]))
        x0 = int(round((tile_min[2] - bounds_min[1]) / pixel_um_yx[1]))
        zmed = orient_plane_yx(
            np.median(np.asarray(array[channel], dtype=np.float32), axis=0),
            scale[1:],
        )
        y1 = min(height, y0 + zmed.shape[0])
        x1 = min(width, x0 + zmed.shape[1])
        if y1 > y0 and x1 > x0:
            mosaics[side][y0:y1, x0:x1] = zmed[: y1 - y0, : x1 - x0]
        tiles.append(
            {
                "tile": record["tile"],
                "side": side,
                "origin_yx_px": [y0, x0],
                "shape_yx_px": [int(zmed.shape[0]), int(zmed.shape[1])],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    left = _stretch_uint8(mosaics["L"])
    right = _stretch_uint8(mosaics["R"])
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[..., 1] = left
    overlay[..., 0] = right
    left_path = output_dir / f"cl_channel{channel}_level{level}_zmedian.png"
    right_path = output_dir / f"cr_channel{channel}_level{level}_zmedian.png"
    overlay_path = output_dir / f"clcr_channel{channel}_level{level}_zmedian_CLgreen_CRred.png"
    Image.fromarray(left).save(left_path)
    Image.fromarray(right).save(right_path)
    Image.fromarray(overlay).save(overlay_path)
    manifest = {
        "artifact_type": "clcr_r_to_l_zmedian_dumb_stitch.v1",
        "channel": channel,
        "level": level,
        "pixel_um_yx": pixel_um_yx.astype(float).tolist(),
        "shape_yx_px": [height, width],
        "bounds_yx_um": {"min": bounds_min.astype(float).tolist(), "max": bounds_max.astype(float).tolist()},
        "outputs": {"CL": str(left_path), "CR": str(right_path), "overlay": str(overlay_path)},
        "tiles": tiles,
    }
    _write_json(output_dir / f"clcr_channel{channel}_level{level}_zmedian_manifest.json", manifest)
    return manifest


def _artifact_stem_from_position_path(position_json: Path) -> str:
    name = position_json.name
    for suffix in (".deconvCZYX.positions.json", ".positions.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return position_json.stem


def fusion_commands(
    *,
    dataset_dir: Path,
    position_json: Path,
    registration_json: Path,
    output_root: Path,
    channels: list[str],
) -> list[str]:
    commands = []
    for channel in channels:
        commands.append(
            "PYTHONPATH=/home/chaichontat/squisher/lightsheet/src "
            "/home/chaichontat/miniforge3/envs/multi/bin/python -m squisher_lightsheet.cli fuse "
            f"{dataset_dir} --position-input {position_json} --registration-input {registration_json} "
            f"--output {output_root} --channel {channel} --fusion-level 0 --fusion-weight-mode content-preibisch-coarse "
            "--batch-size 1 --output-chunksize-zyx 8,1024,1024"
        )
    return commands


def movie_commands(*, fusion_output_root: Path, position_json: Path, channels: list[str]) -> list[str]:
    commands = []
    for channel in channels:
        zarr_path = fusion_output_root / f"fused.ch{channel}.ome.zarr"
        color = "green" if channel == "0" else "magenta"
        out = zarr_path.with_name(f"{zarr_path.name.removesuffix('.ome.zarr')}.1080p.z_h265.high99999_sigma5.hdr.mp4")
        commands.append(
            "/home/chaichontat/miniforge3/envs/multi/bin/python -u "
            f"/home/chaichontat/nvme/lightsheet/20260523-fullHCR/render_omezarr_z_mp4.py {zarr_path} "
            f"--out {out} --size 1080 --fps 20 --colors {color} --high 99.999 --hdr "
            "--unsharp-sigma 5 --unsharp-amount 0.5 --frame-batch-size 16 --encoder hevc_nvenc --preset p4"
        )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Combined CL/CR dataset directory.")
    parser.add_argument("--left-position", type=Path, required=True, help="Side-internal CL optimized positions JSON.")
    parser.add_argument("--right-position", type=Path, required=True, help="Side-internal CR optimized positions JSON.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Centroid-positioned z-median dumb-stitch manifest.",
    )
    parser.add_argument("--phasecorr", type=Path, required=True, help="CL fixed / CR moving z-median phase-corr JSON.")
    parser.add_argument("--method8-summary", type=Path, required=True, help="Method8 CL/CR window summary JSON.")
    parser.add_argument(
        "--deconv-root",
        type=Path,
        required=True,
        help="Directory containing unique CL/CR deconvolved OME-Zarr roots.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to --dataset-dir.")
    parser.add_argument("--artifact-stem", default="clcr_r_to_l", help="Output filename/artifact stem.")
    parser.add_argument(
        "--shape-czyx",
        type=_parse_shape,
        help="Optional C,Z,Y,X override; if omitted, inferred from deconvolved OME-Zarr level 0.",
    )
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        default=["0", "1"],
        help="Comma-separated channels for commands.",
    )
    parser.add_argument("--render-qc", action="store_true", help="Render a no-blend z-median CL/CR dumb-stitch QC PNG.")
    parser.add_argument("--qc-channel", type=int, default=0)
    parser.add_argument("--qc-level", type=int, default=0)
    parser.add_argument(
        "--fusion-output-root",
        type=Path,
        help="Final fused output directory. Defaults to --output-dir, or --dataset-dir when --output-dir is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.dataset_dir
    fusion_output_root = args.fusion_output_root or output_dir
    left_records = _load_side_records(args.left_position, "CL")
    right_records = _load_side_records(args.right_position, "CR")
    manifest = _read_json(args.manifest)
    phasecorr = _read_json(args.phasecorr)
    method8_summary = _read_json(args.method8_summary)
    shape_czyx = _resolve_shape_czyx(
        args.shape_czyx,
        left_records,
        right_records,
        args.deconv_root,
        args.channels,
    )
    provenance = _run_provenance(
        {
            "dataset_dir": args.dataset_dir,
            "left_position": args.left_position,
            "right_position": args.right_position,
            "manifest": args.manifest,
            "phasecorr": args.phasecorr,
            "method8_summary": args.method8_summary,
            "deconv_root": args.deconv_root,
        }
    )

    position_payload = build_clspace_position_payload(
        left_records=left_records,
        right_records=right_records,
        manifest=manifest,
        phasecorr=phasecorr,
        method8_summary=method8_summary,
        left_position=args.left_position,
        right_position=args.right_position,
        manifest_path=args.manifest,
        phasecorr_path=args.phasecorr,
        method8_summary_path=args.method8_summary,
        shape_czyx=shape_czyx,
        channels=args.channels,
        deconv_root=args.deconv_root,
        artifact_stem=args.artifact_stem,
        run_provenance=provenance,
    )
    position_json = output_dir / f"{args.artifact_stem}.deconvCZYX.positions.json"
    registration_json = output_dir / "registration.json"
    _write_json(position_json, position_payload)
    registration_payload = build_identity_registration_payload(
        position_payload,
        deconv_root=args.deconv_root,
        artifact_stem=args.artifact_stem,
    )
    _write_json(registration_json, registration_payload)

    qc_manifest = None
    if args.render_qc:
        qc_manifest = render_zmedian_dumb_stitch(
            position_payload,
            output_dir=output_dir / f"{args.artifact_stem}.qc_zmedian",
            channel=args.qc_channel,
            level=args.qc_level,
        )

    commands = {
        "fusion": fusion_commands(
            dataset_dir=args.dataset_dir,
            position_json=position_json,
            registration_json=registration_json,
            output_root=fusion_output_root,
            channels=args.channels,
        ),
        "movie": movie_commands(
            fusion_output_root=fusion_output_root,
            position_json=position_json,
            channels=args.channels,
        ),
    }
    run_manifest = {
        "position": str(position_json),
        "registration": str(registration_json),
        "qc": qc_manifest,
        "commands": commands,
        "run_provenance": provenance,
    }
    _write_json(output_dir / f"{args.artifact_stem}.workflow.json", run_manifest)
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
