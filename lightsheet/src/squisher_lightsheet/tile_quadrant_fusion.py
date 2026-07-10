from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DIMENSIONS = ("z", "y", "x")
AFFINE_DIMS = ["x_in", "x_out"]
AFFINE_COORDS = {"x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]}
MATERIALIZED_ZSTD_LEVEL = 3


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def _vector_zyx(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([float(values[dim]) for dim in DIMENSIONS], dtype=np.float64)


def _dict_zyx(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {dim: float(array[index]) for index, dim in enumerate(DIMENSIONS)}


def _translation_matrix_zyx(values_um: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(values_um, dtype=np.float64)
    return matrix


def _scale_matrix_zyx(values: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.diag(np.asarray(values, dtype=np.float64))
    return matrix


def _affine_matrix(record: dict[str, Any]) -> np.ndarray:
    affine = record.get("registered_affine")
    if not isinstance(affine, dict) or "matrix" not in affine:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(affine["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"tile {record.get('tile')!r} registered_affine.matrix must be 4x4, got {matrix.shape}")
    return matrix


def _stage_translation_zyx(record: dict[str, Any]) -> np.ndarray:
    if "stage_translation_um" in record:
        return _vector_zyx(record, "stage_translation_um")
    return _vector_zyx(record, "translation_um")


def _stage_scale_zyx(record: dict[str, Any]) -> np.ndarray:
    if "stage_scale_um" in record:
        return _vector_zyx(record, "stage_scale_um")
    return _vector_zyx(record, "scale_um")


def _records_by_tile(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    records = payload.get("tiles")
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a tiles list")
    return {str(record["tile"]): record for record in records}


def _shape_zyx_from_record(record: dict[str, Any]) -> np.ndarray:
    shape = record.get("shape")
    axes = record.get("axes")
    if not isinstance(shape, list) or not isinstance(axes, str):
        raise ValueError(f"tile {record.get('tile')!r} is missing shape/axes")
    if axes == "CZYX":
        return np.asarray(shape[1:4], dtype=np.int64)
    if axes == "ZYX":
        return np.asarray(shape, dtype=np.int64)
    raise ValueError(f"tile {record.get('tile')!r} has unsupported axes {axes!r}")


def _materialized_tile_name(moving_tile: str, quadrant: str, z_start: int) -> str:
    return f"{moving_tile.removesuffix('.ome.zarr')}.{quadrant}.z{int(z_start):05d}.ome.zarr"


def _ome_axis(axis: str) -> dict[str, str]:
    if axis == "C":
        return {"name": "c", "type": "channel"}
    return {"name": axis.lower(), "type": "space", "unit": "micrometer"}


def _ome_scale_transform(*, axes: str, spacing_zyx: np.ndarray) -> dict[str, Any]:
    scale = []
    for axis in axes:
        if axis == "C":
            scale.append(1.0)
        else:
            scale.append(float(abs(spacing_zyx[DIMENSIONS.index(axis.lower())])))
    return {"type": "scale", "scale": scale}


def _ome_level0_array(path: Path) -> Any:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    datasets = root.attrs.get("multiscales", [{}])[0].get("datasets", [])
    dataset_path = str(datasets[0]["path"]) if datasets else "0"
    return root[dataset_path]


def _materialized_compressor() -> Any:
    from zarr.codecs import BloscCodec, BloscShuffle

    return BloscCodec(cname="zstd", clevel=MATERIALIZED_ZSTD_LEVEL, shuffle=BloscShuffle.bitshuffle)


def _materialize_crop_ome_zarr(
    *,
    source_path: Path,
    output_path: Path,
    source_record: dict[str, Any],
    start_zyx: np.ndarray,
    stop_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
) -> list[int]:
    import zarr

    axes = str(source_record.get("axes"))
    if axes not in {"CZYX", "ZYX"}:
        raise ValueError(f"source tile {source_record.get('tile')!r} has unsupported axes {axes!r}")

    source = _ome_level0_array(source_path)
    crop_shape_zyx = (stop_zyx - start_zyx).astype(np.int64)
    if np.any(crop_shape_zyx <= 0):
        raise ValueError(f"crop shape must be positive, got {crop_shape_zyx.tolist()}")
    shape = (
        [int(source.shape[0]), *(int(value) for value in crop_shape_zyx)]
        if axes == "CZYX"
        else [int(value) for value in crop_shape_zyx]
    )
    chunks = [min(int(chunk), int(size)) for chunk, size in zip(source.chunks, shape, strict=True)]

    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    output = root.create_array(
        "0",
        shape=tuple(shape),
        chunks=tuple(chunks),
        dtype=source.dtype,
        dimension_names=tuple(axis.lower() for axis in axes),
        compressors=[_materialized_compressor()],
    )
    output.attrs["_ARRAY_DIMENSIONS"] = [axis.lower() for axis in axes]
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": output_path.name,
            "axes": [_ome_axis(axis) for axis in axes],
            "datasets": [{"path": "0", "coordinateTransformations": [_ome_scale_transform(axes=axes, spacing_zyx=spacing_zyx)]}],
        }
    ]

    z_chunk = int(chunks[1] if axes == "CZYX" else chunks[0])
    for z0 in range(int(start_zyx[0]), int(stop_zyx[0]), z_chunk):
        z1 = min(z0 + z_chunk, int(stop_zyx[0]))
        if axes == "CZYX":
            source_sel = (
                slice(None),
                slice(z0, z1),
                slice(int(start_zyx[1]), int(stop_zyx[1])),
                slice(int(start_zyx[2]), int(stop_zyx[2])),
            )
            target_sel = (slice(None), slice(z0 - int(start_zyx[0]), z1 - int(start_zyx[0])), slice(None), slice(None))
        else:
            source_sel = (
                slice(z0, z1),
                slice(int(start_zyx[1]), int(stop_zyx[1])),
                slice(int(start_zyx[2]), int(stop_zyx[2])),
            )
            target_sel = (slice(z0 - int(start_zyx[0]), z1 - int(start_zyx[0])), slice(None), slice(None))
        output[target_sel] = source[source_sel]
    return shape


def _requested_source_window_zyx(row: dict[str, Any], moving_shape_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray(row.get("requested_fixed_start_zyx", row["fixed_start_zyx"]), dtype=np.int64)
    shape_value = row.get("requested_window_shape_zyx")
    if shape_value is None:
        shape_value = row["window_shape_zyx"]
    shape = np.asarray(shape_value, dtype=np.int64)
    stop = start + shape
    if np.any(shape <= 0):
        raise ValueError(f"requested window shape must be positive, got {shape.tolist()}")
    if np.any(start < 0) or np.any(stop > moving_shape_zyx):
        raise ValueError(
            "requested Image10-local materialization window is outside the moving tile: "
            f"start={start.tolist()} stop={stop.tolist()} moving_shape={moving_shape_zyx.tolist()}"
        )
    return start, stop, shape


def _method8_model_matrix(row: dict[str, Any], *, fixed_shape_zyx: np.ndarray, moving_shape_zyx: np.ndarray) -> np.ndarray:
    if "full_matrix_zyx" not in row or "full_translation_zyx" not in row:
        raise ValueError(f"{row.get('fixed_tile')}->{row.get('moving_tile')} window is missing method8 full transform fields")
    if not np.array_equal(fixed_shape_zyx, moving_shape_zyx):
        raise ValueError(
            f"method8 materialized transform requires equal fixed/moving tile shapes, "
            f"got fixed={fixed_shape_zyx.tolist()} moving={moving_shape_zyx.tolist()}"
        )
    linear = np.asarray(row["full_matrix_zyx"], dtype=np.float64)
    translation = np.asarray(row["full_translation_zyx"], dtype=np.float64)
    if linear.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(
            f"{row.get('fixed_tile')}->{row.get('moving_tile')} method8 transform has invalid shape: "
            f"matrix={linear.shape} translation={translation.shape}"
        )
    center = (fixed_shape_zyx.astype(np.float64) - 1.0) / 2.0
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = center - linear @ center + translation
    return matrix


def _method8_registered_affine_um(
    *,
    row: dict[str, Any],
    fixed_record: dict[str, Any],
    moving_record: dict[str, Any],
    channel_shift_um: np.ndarray,
) -> np.ndarray:
    """Map Image10 source-stage microns into Image14 registered microns.

    Method8 stores a full moving-pixel to fixed-pixel model for each accepted
    quadrant/z window. The materialized chunk keeps coarse Image10 stage
    metadata, so the relative fuser affine must convert those source-stage
    microns through method8 into the fixed Image14 registered coordinate system.
    """
    fixed_shape = _shape_zyx_from_record(fixed_record)
    moving_shape = _shape_zyx_from_record(moving_record)
    fixed_stage = _stage_translation_zyx(fixed_record)
    moving_stage = _stage_translation_zyx(moving_record)
    fixed_scale = _stage_scale_zyx(fixed_record)
    moving_scale = _stage_scale_zyx(moving_record)
    if np.any(moving_scale == 0):
        raise ValueError(f"moving tile {moving_record.get('tile')!r} has zero scale component")

    method8_px = _method8_model_matrix(row, fixed_shape_zyx=fixed_shape, moving_shape_zyx=moving_shape)
    fixed_stage_from_pixel = _translation_matrix_zyx(fixed_stage) @ _scale_matrix_zyx(fixed_scale)
    moving_pixel_from_stage = _scale_matrix_zyx(1.0 / moving_scale) @ _translation_matrix_zyx(-moving_stage)
    affine = _affine_matrix(fixed_record) @ fixed_stage_from_pixel @ method8_px @ moving_pixel_from_stage
    if np.any(channel_shift_um):
        affine = affine @ _translation_matrix_zyx(channel_shift_um)
    return affine


def _accepted_or_quality_gate(row: dict[str, Any]) -> bool:
    if row.get("status") == "accepted":
        return True
    return row.get("rejection_reason") == "quality_gate"


def export_tile_quadrant_materialized_chunks(
    *,
    window_json_dir: Path,
    moving_position_input: Path,
    fixed_registration_input: Path,
    output_dir: Path,
    channel_source_shift_px_zyx: tuple[float, float, float] | None = None,
    include_quality_gate_rejected: bool = False,
) -> dict[str, Path]:
    moving_by_tile = _records_by_tile(moving_position_input)
    fixed_by_tile = _records_by_tile(fixed_registration_input)
    fixed_payload = _read_json(fixed_registration_input)
    moving_payload = _read_json(moving_position_input)
    channel_shift_px = (
        np.zeros(3, dtype=np.float64)
        if channel_source_shift_px_zyx is None
        else np.asarray(channel_source_shift_px_zyx, dtype=np.float64)
    )

    position_records: list[dict[str, Any]] = []
    registration_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    channel_shift_um_summary: list[float] | None = None

    for path in sorted(window_json_dir.glob("*.json")):
        row = _read_json(path)
        if not _accepted_or_quality_gate(row):
            skipped.append({"window_json": str(path), "status": row.get("status"), "rejection_reason": row.get("rejection_reason")})
            continue
        if row.get("rejection_reason") == "quality_gate" and not include_quality_gate_rejected:
            skipped.append({"window_json": str(path), "status": row.get("status"), "rejection_reason": row.get("rejection_reason")})
            continue

        fixed_tile = str(row["fixed_tile"])
        moving_tile = str(row["moving_tile"])
        moving_record = moving_by_tile[moving_tile]
        fixed_record = fixed_by_tile[fixed_tile]
        moving_scale = _vector_zyx(moving_record, "scale_um")
        moving_spacing = np.abs(moving_scale)
        moving_stage = _vector_zyx(moving_record, "translation_um")
        moving_shape = _shape_zyx_from_record(moving_record)
        source_start, source_stop, crop_shape = _requested_source_window_zyx(row, moving_shape)
        if np.any(channel_shift_px):
            channel_shift_um = channel_shift_px * moving_spacing
        else:
            channel_shift_um = np.zeros(3, dtype=np.float64)
        if channel_shift_um_summary is None:
            channel_shift_um_summary = channel_shift_um.tolist()

        moving_chunk_origin_um = moving_stage + source_start.astype(np.float64) * moving_scale
        chunk_affine = _method8_registered_affine_um(
            row=row,
            fixed_record=fixed_record,
            moving_record=moving_record,
            channel_shift_um=channel_shift_um,
        )

        z_start = int(source_start[0])
        source_path = Path(str(moving_record["path"]))
        if not source_path.is_absolute():
            source_path = moving_position_input.parent / source_path
        chunk_name = _materialized_tile_name(moving_tile, str(row["quadrant"]), z_start)
        chunk_path = output_dir / "materialized_tiles" / chunk_name
        shape = _materialize_crop_ome_zarr(
            source_path=source_path,
            output_path=chunk_path,
            source_record=moving_record,
            start_zyx=source_start,
            stop_zyx=source_stop,
            spacing_zyx=moving_spacing,
        )
        position_records.append(
            {
                "tile": chunk_name,
                "path": str(chunk_path),
                "source_tile": moving_tile,
                "materialized_source_path": str(source_path),
                "materialized_source_start_zyx": source_start.tolist(),
                "materialized_source_stop_zyx": source_stop.tolist(),
                "translation_um": _dict_zyx(moving_chunk_origin_um),
                "scale_um": _dict_zyx(moving_scale),
                "shape": shape,
                "axes": moving_record.get("axes"),
                "spacing_um": moving_record.get("spacing_um", moving_record.get("scale_um")),
                "channels": moving_record.get("channels", ["0"]),
                "tracks": moving_record.get("tracks", []),
                "source_window_json": str(path),
                "status": row.get("status"),
                "rejection_reason": row.get("rejection_reason"),
            }
        )
        registration_records.append(
            {
                "tile": chunk_name,
                "path": str(chunk_path),
                "source_tile": moving_tile,
                "materialized_source_path": str(source_path),
                "materialized_source_start_zyx": source_start.tolist(),
                "materialized_source_stop_zyx": source_stop.tolist(),
                "shape": shape,
                "axes": moving_record.get("axes"),
                "spacing_um": moving_record.get("spacing_um", moving_record.get("scale_um")),
                "channels": moving_record.get("channels", ["0"]),
                "tracks": moving_record.get("tracks", []),
                "source_view": moving_record.get("source_view", moving_record.get("side")),
                "stage_translation_um": _dict_zyx(moving_chunk_origin_um),
                "stage_scale_um": _dict_zyx(moving_scale),
                "registered_affine": {
                    "dims": AFFINE_DIMS,
                    "coords": AFFINE_COORDS,
                    "matrix": chunk_affine.tolist(),
                },
                "source_window_json": str(path),
                "source_window_status": row.get("status"),
                "source_window_rejection_reason": row.get("rejection_reason"),
                "method8_full_matrix_zyx": row.get("full_matrix_zyx"),
                "method8_full_translation_zyx": row.get("full_translation_zyx"),
            }
        )
        diagnostics.append(
            {
                "tile": chunk_name,
                "source_window_json": str(path),
                "fixed_tile": fixed_tile,
                "moving_tile": moving_tile,
                "quadrant": row.get("quadrant"),
                "z_start": z_start,
                "status": row.get("status"),
                "rejection_reason": row.get("rejection_reason"),
                "geometry_source": "moving_position_input_parent_tile_translation_plus_image10_local_window",
                "registration_affine_source": "method8_full_transform_composed_with_fixed_registration_and_source_channel_shift",
            }
        )

    position_path = output_dir / "tile_quadrant_materialized_chunks.positions.json"
    registration_path = output_dir / "tile_quadrant_materialized_chunks.registration.json"
    summary_path = output_dir / "tile_quadrant_materialized_chunks.summary.json"
    _write_json(
        position_path,
        {
            "units": "micrometer",
            "artifact_type": "squisher_lightsheet.tile_quadrant_materialized_chunk_positions.v1",
            "window_json_dir": str(window_json_dir.resolve()),
            "moving_position_input": str(moving_position_input.resolve()),
            "fixed_registration_input": str(fixed_registration_input.resolve()),
            "channel_source_shift_px_zyx": channel_shift_px.tolist(),
            "channel_source_shift_um_zyx": channel_shift_um_summary,
            "tiles": position_records,
        },
    )
    _write_json(
        registration_path,
        {
            "input_dir": str((output_dir / "materialized_tiles").resolve()),
            "metadata_transform_key": "stage_metadata",
            "registered_transform_key": "registered_affine",
            "spacing_um": moving_payload.get("spacing_um"),
            "artifact_type": "squisher_lightsheet.tile_quadrant_materialized_chunk_registration.v1",
            "window_json_dir": str(window_json_dir.resolve()),
            "moving_position_input": str(moving_position_input.resolve()),
            "fixed_registration_input": str(fixed_registration_input.resolve()),
            "fixed_registration_metadata": {
                "input_dir": fixed_payload.get("input_dir"),
                "registered_transform_key": fixed_payload.get("registered_transform_key"),
            },
            "channel_correction": {
                "schema": "squisher_lightsheet.source_space_channel_translation.v1",
                "shift_direction": "Image10 ch1 chunks keep source bytes; registered_affine applies the ch1-to-ch0 source-space translation",
                "shift_ch1_to_ch0_px_zyx": channel_shift_px.tolist(),
                "shift_ch1_to_ch0_um_zyx": channel_shift_um_summary,
                "composition": "registered_affine = method8_source_stage_to_image14_registered @ T(channel_shift_um_zyx)",
            },
            "method8_transform_usage": (
                "registered_affine maps materialized Image10 source-stage microns through each window's "
                "method8 full moving-pixel-to-fixed-pixel transform into Image14 registered microns"
            ),
            "tiles": registration_records,
        },
    )
    _write_json(
        summary_path,
        {
            "artifact_type": "squisher_lightsheet.tile_quadrant_materialized_chunk_export_summary.v1",
            "position_output": str(position_path.resolve()),
            "registration_output": str(registration_path.resolve()),
            "window_count": len(position_records),
            "skipped_window_count": len(skipped),
            "included_quality_gate_rejected": sum(1 for row in diagnostics if row.get("rejection_reason") == "quality_gate"),
            "geometry_source": "moving_position_input_parent_tile_translation_plus_image10_local_window",
            "method8_usage": "accepted/skipped window mask plus per-window registered_affine transform for fusion",
            "diagnostics": diagnostics,
            "skipped": skipped,
        },
    )
    return {"position": position_path.resolve(), "registration": registration_path.resolve(), "summary": summary_path.resolve()}
