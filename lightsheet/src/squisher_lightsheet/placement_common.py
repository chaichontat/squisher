"""Layout-independent pieces of cross-dataset placement publication, shared by CL/CR and TL/TR.

Both layouts keep each side's own tile geometry and publish one position file over a directory of both sides'
deconvolved tiles, an identity-affine registration whose placement lives in each tile's stage translation, the median
accepted Method8 spatial-overlap translation between the sides, and a no-blend z-median overlay for review.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.geometry import orient_plane_yx, signed_bounds

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
RECORD_KEYS = ("shape", "axes", "channels", "tracks")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def vector(record: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray([float(record[key][dim]) for dim in DIMENSIONS], dtype=np.float64)


def dict_zyx(values: np.ndarray) -> dict[str, float]:
    return {dim: float(value) for dim, value in zip(DIMENSIONS, values, strict=True)}


def source_zarr_name(record: dict[str, Any]) -> str:
    source_tile = str(record.get("source_tile") or record["tile"])
    return source_tile.removesuffix(".ome.tif").removesuffix(".ome.zarr") + ".ome.zarr"


def shape_zyx(record: dict[str, Any]) -> np.ndarray:
    shape = record["shape"]
    if record["axes"] == "CZYX":
        return np.asarray(shape[1:], dtype=np.float64)
    if record["axes"] == "ZYX":
        return np.asarray(shape, dtype=np.float64)
    raise ValueError(f"{record['tile']} has unsupported axes {record['axes']!r}")


def load_side_records(path: Path, side: str) -> list[dict[str, Any]]:
    records = [dict(record) for record in read_json(path)["tiles"]]
    if not records:
        raise ValueError(f"{path} contains no tile records")
    for record in records:
        missing = [key for key in RECORD_KEYS if key not in record]
        if missing:
            raise ValueError(f"{path} record {record.get('tile')} lacks {missing}")
        record["source_side"] = side
    return records


def record_centers(records: list[dict[str, Any]]) -> np.ndarray:
    centers = []
    for record in records:
        low, high = signed_bounds(vector(record, "translation_um"), vector(record, "scale_um"), shape_zyx(record))
        centers.append((low + high) / 2.0)
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)


def common_scale(records: list[dict[str, Any]], side: str) -> np.ndarray:
    first = vector(records[0], "scale_um")
    for record in records[1:]:
        current = vector(record, "scale_um")
        if not np.allclose(current, first, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{side} records have inconsistent signed scales: {first} versus {current}")
    return first


def accepted_method8_translations(
    summary: dict[str, Any], left: set[str], right: set[str], *, sides: tuple[str, str]
) -> np.ndarray:
    """Accepted local translations, after checking every accepted row pairs a fixed-side tile with a moving-side one."""
    rows = [row for row in summary.get("windows", []) if row.get("status") == "accepted"]
    stems = {name.removesuffix(".ome.zarr") for name in left | right}

    def side_of(tile: str) -> str | None:
        stem = Path(tile).name.removesuffix(".ome.zarr").removesuffix(".ome.tif")
        if stem not in stems:
            return None
        return sides[0] if f"{stem}.ome.zarr" in left else sides[1]

    bad = [
        row
        for row in rows
        if side_of(str(row.get("fixed_tile", ""))) != sides[0] or side_of(str(row.get("moving_tile", ""))) != sides[1]
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} accepted Method8 rows do not pair a {sides[0]} fixed tile with a {sides[1]} moving tile"
        )
    translations = [row["local_translation_zyx"] for row in rows if row.get("local_translation_zyx") is not None]
    if not translations:
        raise ValueError("Method8 summary contains no accepted local_translation_zyx rows")
    return np.asarray(translations, dtype=np.float64)


def translation_stats(translations: np.ndarray) -> dict[str, Any]:
    q25, q75 = np.percentile(translations, [25.0, 75.0], axis=0)
    median = np.median(translations, axis=0)
    return {
        "count": int(translations.shape[0]),
        "median_zyx": median.astype(float).tolist(),
        "mean_zyx": np.mean(translations, axis=0).astype(float).tolist(),
        "iqr_zyx": (q75 - q25).astype(float).tolist(),
        "mad_zyx": np.median(np.abs(translations - median), axis=0).astype(float).tolist(),
    }


def build_identity_registration_payload(
    position_payload: dict[str, Any], *, deconv_root: Path, artifact_type: str
) -> dict[str, Any]:
    tiles = [
        {
            "tile": record["tile"],
            "path": str(deconv_root / record["tile"]),
            "source_view": record["side"],
            "stage_translation_um": deepcopy(record["translation_um"]),
            "stage_scale_um": deepcopy(record["scale_um"]),
            "spacing_um": deepcopy(record["spacing_um"]),
            **{key: deepcopy(record[key]) for key in RECORD_KEYS},
            "registered_affine": deepcopy(IDENTITY_AFFINE),
        }
        for record in position_payload["tiles"]
    ]
    return {
        "input_dir": str(deconv_root),
        "metadata_transform_key": "stage_metadata",
        "registered_transform_key": "registered_affine",
        "spacing_um": deepcopy(position_payload["tiles"][0]["spacing_um"]),
        "metrics": {
            "artifact_type": artifact_type,
            "registered_affine_note": "Identity affine; the cross-dataset placement is baked into each tile's stage_translation_um.",
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


def render_zmedian_overlay(
    position_payload: dict[str, Any], *, output_dir: Path, channel: int, level: int, sides: tuple[str, str], artifact_type: str
) -> dict[str, Any]:
    """No-blend z-median mosaics of each side and their fixed green / moving red overlay at one pyramid level."""
    import zarr
    from PIL import Image

    pixel_um_yx: np.ndarray | None = None
    prepared = []
    bounds_min = np.full(2, np.inf, dtype=np.float64)
    bounds_max = np.full(2, -np.inf, dtype=np.float64)
    for record in position_payload["tiles"]:
        level_path = Path(record["path"]) / str(level)
        if not level_path.exists():
            raise ValueError(f"{record['path']} does not contain OME-Zarr level {level}")
        array = zarr.open(str(level_path), mode="r")
        stack = array[channel] if record["axes"] == "CZYX" else array
        full_shape_zyx = shape_zyx(record)
        scale = vector(record, "scale_um")
        current = np.abs(scale[1:]) * full_shape_zyx[-2:] / np.asarray(stack.shape[-2:], dtype=np.float64)
        if pixel_um_yx is None:
            pixel_um_yx = current
        elif not np.allclose(pixel_um_yx, current, rtol=1e-6, atol=1e-6):
            raise ValueError(f"inconsistent level {level} pixel spacing: {pixel_um_yx} versus {current}")
        tile_min, tile_max = signed_bounds(vector(record, "translation_um"), scale, full_shape_zyx)
        bounds_min = np.minimum(bounds_min, tile_min[1:])
        bounds_max = np.maximum(bounds_max, tile_max[1:])
        prepared.append((record, stack, tile_min, scale))
    if pixel_um_yx is None:
        raise ValueError("position payload contains no tiles")
    height = int(math.ceil((bounds_max[0] - bounds_min[0]) / pixel_um_yx[0]))
    width = int(math.ceil((bounds_max[1] - bounds_min[1]) / pixel_um_yx[1]))
    mosaics = {"L": np.zeros((height, width), dtype=np.float32), "R": np.zeros((height, width), dtype=np.float32)}
    for record, stack, tile_min, scale in prepared:
        y0 = int(round((tile_min[1] - bounds_min[0]) / pixel_um_yx[0]))
        x0 = int(round((tile_min[2] - bounds_min[1]) / pixel_um_yx[1]))
        zmed = orient_plane_yx(np.median(np.asarray(stack, dtype=np.float32), axis=0), scale[1:])
        y1 = min(height, y0 + zmed.shape[0])
        x1 = min(width, x0 + zmed.shape[1])
        if y1 > y0 and x1 > x0:
            mosaics[record["side"]][y0:y1, x0:x1] = zmed[: y1 - y0, : x1 - x0]
    output_dir.mkdir(parents=True, exist_ok=True)
    left = _stretch_uint8(mosaics["L"])
    right = _stretch_uint8(mosaics["R"])
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[..., 1] = left
    overlay[..., 0] = right
    fixed, moving = sides
    outputs = {
        fixed: output_dir / f"{fixed.lower()}_channel{channel}_level{level}_zmedian.png",
        moving: output_dir / f"{moving.lower()}_channel{channel}_level{level}_zmedian.png",
        "overlay": output_dir
        / f"{fixed.lower()}{moving.lower()}_channel{channel}_level{level}_zmedian_{fixed}green_{moving}red.png",
    }
    Image.fromarray(left).save(outputs[fixed])
    Image.fromarray(right).save(outputs[moving])
    Image.fromarray(overlay).save(outputs["overlay"])
    manifest = {
        "artifact_type": artifact_type,
        "channel": channel,
        "level": level,
        "pixel_um_yx": pixel_um_yx.astype(float).tolist(),
        "shape_yx_px": [height, width],
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(output_dir / "zmedian_overlay.json", manifest)
    return manifest
