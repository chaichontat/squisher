from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np

from squisher.jpegxr_zarr import (
    DEFAULT_JPEGXR_LEVEL,
    jpegxr_sharding_codec,
    register_jpegxr_codec,
)

from squisher_lightsheet.cross_register_method8 import _axis_starts
from squisher_lightsheet.ngff import axes as ngff_axes
from squisher_lightsheet.ngff import dataset_paths
from squisher_lightsheet.ngff import level_array

DIMENSIONS = ("z", "y", "x")
AFFINE_DIMS = ["x_in", "x_out"]
AFFINE_COORDS = {"x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]}
MATERIALIZED_SHARD_Z = 48


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def _materialization_payload(path: Path) -> list[dict[str, Any]]:
    array_path = path / "0"
    return [
        {"path": str(file.relative_to(path)), "size": file.stat().st_size}
        for file in sorted(array_path.rglob("*"))
        if file.is_file() and file.name != "zarr.json"
    ]


def _write_materialization_completion(path: Path) -> None:
    _write_json(
        path / "squisher.complete.json",
        {
            "artifact_type": "squisher_lightsheet.materialization_completion.v1",
            "payload": _materialization_payload(path),
        },
    )


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
        raise ValueError(
            f"tile {record.get('tile')!r} registered_affine.matrix must be 4x4, got {matrix.shape}"
        )
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


def _tile_identity(name: str) -> str:
    for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def _merge_moving_pixel_sources(
    moving_by_tile: dict[str, dict[str, Any]],
    source_input: Path | None,
) -> dict[str, dict[str, Any]]:
    if source_input is None:
        return moving_by_tile
    source_by_identity: dict[str, dict[str, Any]] = {}
    for record in _records_by_tile(source_input).values():
        identity = _tile_identity(str(record["tile"]))
        if identity in source_by_identity:
            raise ValueError(f"{source_input} contains duplicate tile identity {identity!r}")
        source_by_identity[identity] = record
    merged = {}
    for tile, moving_record in moving_by_tile.items():
        identity = _tile_identity(tile)
        if identity not in source_by_identity:
            raise ValueError(f"{source_input} is missing pixel source for moving tile {tile!r}")
        source_record = source_by_identity[identity]
        merged[tile] = moving_record | {
            key: source_record[key]
            for key in ("path", "shape", "axes", "channels", "tracks")
            if key in source_record
        }
    return merged


def _shape_zyx_from_record(record: dict[str, Any]) -> np.ndarray:
    shape = record.get("shape")
    axes = record.get("axes")
    if not isinstance(shape, list) or not isinstance(axes, str):
        source_path = Path(str(record["path"]))
        source, axes = _ome_level0_array_and_axes(source_path)
        shape = list(source.shape)
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


def _ome_level0_array_and_axes(path: Path) -> tuple[Any, str]:
    import zarr

    register_jpegxr_codec()
    root = zarr.open_group(str(path), mode="r")
    source = level_array(root, context=path)
    return source, ngff_axes(root, source)


def _ome_level0_array(path: Path) -> Any:
    return _ome_level0_array_and_axes(path)[0]


def _ome_downsample_source(path: Path, desired_factors_zyx: np.ndarray) -> tuple[Any, str, int, np.ndarray]:
    """Open the deepest pyramid level compatible with an exact block reduction."""
    import zarr

    register_jpegxr_codec()
    root = zarr.open_group(str(path), mode="r")
    paths = dataset_paths(root)
    base = root[paths[0]]
    axes = ngff_axes(root, base)
    if axes not in {"CZYX", "ZYX"}:
        raise ValueError(f"{path} has unsupported axes {axes!r}")
    spatial_indices = np.asarray([axes.index(axis.upper()) for axis in DIMENSIONS], dtype=np.int64)
    base_shape_zyx = np.asarray(base.shape, dtype=np.int64)[spatial_indices]
    desired = np.asarray(desired_factors_zyx, dtype=np.int64)

    candidates: list[tuple[int, int, Any, np.ndarray]] = []
    for level, dataset_path in enumerate(paths):
        array = root[dataset_path]
        if ngff_axes(root, array) != axes:
            continue
        level_shape_zyx = np.asarray(array.shape, dtype=np.int64)[spatial_indices]
        if np.any(level_shape_zyx <= 0) or np.any(base_shape_zyx % level_shape_zyx):
            continue
        factors = base_shape_zyx // level_shape_zyx
        if np.any(desired % factors):
            continue
        candidates.append((int(np.prod(factors)), level, array, factors))
    if not candidates:
        raise ValueError(f"{path} has no pyramid level compatible with factors {desired.tolist()}")
    _score, level, source, factors = max(candidates, key=lambda candidate: candidate[:2])
    return source, axes, level, factors


def _materialized_sharded_codecs(inner_chunks: tuple[int, ...], *, jpegxr_level: float) -> list[Any]:
    return [jpegxr_sharding_codec(inner_chunks, level=jpegxr_level)]


def _materialized_inner_chunks(*, axes: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    if axes == "CZYX":
        return (1, 1, int(shape[2]), int(shape[3]))
    if axes == "ZYX":
        return (1, int(shape[1]), int(shape[2]))
    raise ValueError(f"unsupported axes {axes!r}")


def _materialized_shard_chunks(
    *,
    axes: str,
    shape: tuple[int, ...],
    inner_chunks: tuple[int, ...],
) -> tuple[int, ...]:
    z_axis = 1 if axes == "CZYX" else 0
    z_size = int(shape[z_axis])
    inner_z = int(inner_chunks[z_axis])
    shard_z = min(MATERIALIZED_SHARD_Z, z_size)
    shard_z = max(inner_z, (shard_z // inner_z) * inner_z)
    if axes == "CZYX":
        return (1, shard_z, int(shape[2]), int(shape[3]))
    if axes == "ZYX":
        return (shard_z, int(shape[1]), int(shape[2]))
    raise ValueError(f"unsupported axes {axes!r}")


def _materialize_crop_ome_zarr(
    *,
    source_path: Path,
    output_path: Path,
    source_record: dict[str, Any],
    start_zyx: np.ndarray,
    stop_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
) -> list[int]:
    import zarr

    source, inferred_axes = _ome_level0_array_and_axes(source_path)
    axes_value = source_record.get("axes")
    axes = inferred_axes if not isinstance(axes_value, str) or not axes_value else axes_value
    if axes != inferred_axes:
        raise ValueError(
            f"source tile {source_record.get('tile')!r} records axes {axes!r}, "
            f"but {source_path} stores {inferred_axes!r}"
        )
    if axes not in {"CZYX", "ZYX"}:
        raise ValueError(f"source tile {source_record.get('tile')!r} has unsupported axes {axes!r}")

    crop_shape_zyx = (stop_zyx - start_zyx).astype(np.int64)
    if np.any(crop_shape_zyx <= 0):
        raise ValueError(f"crop shape must be positive, got {crop_shape_zyx.tolist()}")
    shape = tuple(
        [int(source.shape[0]), *(int(value) for value in crop_shape_zyx)]
        if axes == "CZYX"
        else [int(value) for value in crop_shape_zyx]
    )
    inner_chunks = _materialized_inner_chunks(axes=axes, shape=shape)
    shard_chunks = _materialized_shard_chunks(axes=axes, shape=shape, inner_chunks=inner_chunks)

    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    output = zarr.open(
        str(output_path / "0"),
        mode="w",
        shape=shape,
        chunks=shard_chunks,
        dtype=source.dtype,
        zarr_format=3,
        dimension_names=tuple(axis.lower() for axis in axes),
        codecs=_materialized_sharded_codecs(inner_chunks, jpegxr_level=jpegxr_level),
    )
    output.attrs["_ARRAY_DIMENSIONS"] = [axis.lower() for axis in axes]
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": output_path.name,
            "axes": [_ome_axis(axis) for axis in axes],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [_ome_scale_transform(axes=axes, spacing_zyx=spacing_zyx)],
                }
            ],
        }
    ]

    z_chunk = int(shard_chunks[1] if axes == "CZYX" else shard_chunks[0])
    for z0 in range(int(start_zyx[0]), int(stop_zyx[0]), z_chunk):
        z1 = min(z0 + z_chunk, int(stop_zyx[0]))
        if axes == "CZYX":
            source_sel = (
                slice(None),
                slice(z0, z1),
                slice(int(start_zyx[1]), int(stop_zyx[1])),
                slice(int(start_zyx[2]), int(stop_zyx[2])),
            )
            target_sel = (
                slice(None),
                slice(z0 - int(start_zyx[0]), z1 - int(start_zyx[0])),
                slice(None),
                slice(None),
            )
        else:
            source_sel = (
                slice(z0, z1),
                slice(int(start_zyx[1]), int(stop_zyx[1])),
                slice(int(start_zyx[2]), int(stop_zyx[2])),
            )
            target_sel = (slice(z0 - int(start_zyx[0]), z1 - int(start_zyx[0])), slice(None), slice(None))
        output[target_sel] = source[source_sel]
    return list(shape)


def _requested_source_window_zyx(
    row: dict[str, Any], moving_shape_zyx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _fused_fixed_overlapping_window_zyx(
    *,
    core_start_zyx: np.ndarray,
    source_shape_zyx: np.ndarray,
    core_shape_zyx: np.ndarray,
    window_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a core-grid location to the same-index overlapping window grid."""
    overlap_start = []
    for axis, (core_start, size, core, window) in enumerate(
        zip(core_start_zyx, source_shape_zyx, core_shape_zyx, window_shape_zyx, strict=True)
    ):
        core_starts = _axis_starts(int(size), window=int(core), step=int(core))
        window_starts = _axis_starts(int(size), window=int(window), step=int(core))
        if len(core_starts) != len(window_starts):
            raise ValueError(
                f"axis {axis} core grid has {len(core_starts)} windows but overlap grid has "
                f"{len(window_starts)} for size={int(size)}, core={int(core)}, window={int(window)}"
            )
        try:
            index = core_starts.index(int(core_start))
        except ValueError as error:
            raise ValueError(
                f"axis {axis} core start {int(core_start)} is not on expected grid {core_starts}"
            ) from error
        overlap_start.append(window_starts[index])
    start = np.asarray(overlap_start, dtype=np.int64)
    return start, start + np.asarray(window_shape_zyx, dtype=np.int64)


def _fused_fixed_registered_affine_um(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Compose a selected crop-local affine into the fused physical stage frame."""
    fixed_start = np.asarray(row["fixed_start_zyx"], dtype=np.float64)
    shape = np.asarray(row["moving_shape_zyx"], dtype=np.float64)
    fixed_scale = np.asarray(row["fused_scale_zyx"], dtype=np.float64)
    fixed_translation = np.asarray(row["fused_translation_zyx"], dtype=np.float64)
    matrix = np.asarray(row["selected_local_matrix_zyx"], dtype=np.float64)
    translation = np.asarray(row["selected_local_translation_zyx"], dtype=np.float64)
    if matrix.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"{row.get('moving_tile')!r} selected local affine has invalid shape")
    if np.any(fixed_scale == 0):
        raise ValueError(f"{row.get('moving_tile')!r} fused scale contains zero")

    center = (shape - 1.0) / 2.0
    fixed_origin = fixed_translation + fixed_start * fixed_scale
    scale_matrix = np.diag(fixed_scale)
    physical_linear = scale_matrix @ matrix @ np.diag(1.0 / fixed_scale)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = physical_linear
    affine[:3, 3] = (
        fixed_origin + fixed_scale * (center + translation - matrix @ center) - physical_linear @ fixed_origin
    )
    return affine, fixed_origin


def _fused_fixed_registration_from_summary(
    *,
    source_summary_input: Path,
    moving_by_tile: dict[str, dict[str, Any]],
    level_factor_zyx: np.ndarray,
) -> dict[str, Any]:
    summary = _read_json(source_summary_input)
    compact_rows = summary.get("windows")
    if not isinstance(compact_rows, list):
        raise ValueError(f"{source_summary_input} must contain a windows list")

    records: list[dict[str, Any]] = []
    for compact in compact_rows:
        window_json = Path(str(compact["level0_json"]))
        row = _read_json(window_json)
        if row.get("status") != "accepted":
            continue
        moving_tile = str(row["moving_tile"])
        moving_record = moving_by_tile[moving_tile]
        raw_scale = _vector_zyx(moving_record, "scale_um")
        fixed_scale = np.asarray(row["fused_scale_zyx"], dtype=np.float64)
        if not np.allclose(raw_scale, fixed_scale, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{window_json} moving scale {raw_scale.tolist()} does not match "
                f"fused scale {fixed_scale.tolist()}"
            )
        affine, fixed_origin = _fused_fixed_registered_affine_um(row)
        core_start = np.asarray(row["moving_start_l0_zyx"], dtype=np.int64)
        core_stop = np.asarray(row["moving_stop_l0_zyx"], dtype=np.int64)
        source_origin = _stage_translation_zyx(moving_record) + core_start * raw_scale
        quadrant_value = row.get("quadrant")
        if quadrant_value is None:
            quadrant_value = compact["quadrant"]
        quadrant = str(quadrant_value)
        tile_name = _materialized_tile_name(moving_tile, quadrant, int(core_start[0]))
        records.append(
            {
                "tile": tile_name,
                "moving_tile": moving_tile,
                "method8_window_json": str(window_json),
                "status": "accepted",
                "rejection_reason": None,
                "transform_source": row.get("selected_attempt"),
                "shape": ((core_stop - core_start) // level_factor_zyx).tolist(),
                "axes": "ZYX",
                "spacing_um": _dict_zyx(np.abs(raw_scale * level_factor_zyx)),
                "channels": ["0"],
                "tracks": [],
                "source_view": moving_record.get("source_view", moving_record.get("side")),
                "source_origin_um": _dict_zyx(source_origin),
                "fixed_origin_um": _dict_zyx(fixed_origin),
                "stage_translation_um": _dict_zyx(fixed_origin),
                "stage_scale_um": _dict_zyx(raw_scale * level_factor_zyx),
                "registered_affine": {
                    "dims": AFFINE_DIMS,
                    "coords": AFFINE_COORDS,
                    "matrix": affine.tolist(),
                },
            }
        )
    return {
        "artifact_type": "squisher_lightsheet.fused_fixed_selected_transform_registration.v1",
        "source_registration_summary": str(source_summary_input.resolve()),
        "fixed_fused": summary.get("fixed_fused"),
        "tiles": records,
    }


def _materialize_downsampled_channel_crop_ome_zarr(
    *,
    source_path: Path,
    output_path: Path,
    source_record: dict[str, Any],
    source_channel: int,
    start_zyx: np.ndarray,
    stop_zyx: np.ndarray,
    level_factor_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    output_codec: Literal["zstd", "jpegxr"],
    zstd_level: int = 3,
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
) -> list[int]:
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    from squisher_lightsheet.rough_phase import downsample_axis_blocks

    source, inferred_axes, source_level, source_factors = _ome_downsample_source(
        source_path, level_factor_zyx
    )
    axes_value = source_record.get("axes")
    axes = inferred_axes if not isinstance(axes_value, str) or not axes_value else axes_value
    if axes != inferred_axes:
        raise ValueError(
            f"source tile {source_record.get('tile')!r} records axes {axes!r}, "
            f"but {source_path} stores {inferred_axes!r}"
        )
    if axes not in {"CZYX", "ZYX"}:
        raise ValueError(f"source tile {source_record.get('tile')!r} has unsupported axes {axes!r}")
    if axes == "ZYX" and source_channel != 0:
        raise ValueError(f"ZYX source only supports channel 0, got {source_channel}")
    if axes == "CZYX" and not 0 <= source_channel < int(source.shape[0]):
        raise ValueError(f"source channel {source_channel} is outside CZYX shape {source.shape}")

    if np.any(start_zyx % source_factors) or np.any(stop_zyx % source_factors):
        raise ValueError(
            f"crop bounds {start_zyx.tolist()}:{stop_zyx.tolist()} are not aligned to "
            f"source pyramid factors {source_factors.tolist()}"
        )
    source_start = np.asarray(start_zyx, dtype=np.int64) // source_factors
    source_stop = np.asarray(stop_zyx, dtype=np.int64) // source_factors
    spatial_indices = tuple(axes.index(axis.upper()) for axis in DIMENSIONS)
    source_shape_zyx = np.asarray(source.shape, dtype=np.int64)[list(spatial_indices)]
    if np.any(source_start < 0) or np.any(source_stop > source_shape_zyx):
        raise ValueError(
            f"crop {start_zyx.tolist()}:{stop_zyx.tolist()} is outside source shape "
            f"{(source_shape_zyx * source_factors).tolist()}"
        )
    remaining_factors = np.asarray(level_factor_zyx, dtype=np.int64) // source_factors
    crop_shape = np.asarray(stop_zyx, dtype=np.int64) - np.asarray(start_zyx, dtype=np.int64)
    factors = np.asarray(level_factor_zyx, dtype=np.int64)
    if np.any(crop_shape <= 0) or np.any(factors <= 0) or np.any(crop_shape % factors):
        raise ValueError(
            f"crop shape {crop_shape.tolist()} must be positive and divisible by factors {factors.tolist()}"
        )
    output_shape = tuple(int(value) for value in crop_shape // factors)
    inner_chunks = _materialized_inner_chunks(axes="ZYX", shape=output_shape)
    if output_codec == "jpegxr":
        chunks = _materialized_shard_chunks(axes="ZYX", shape=output_shape, inner_chunks=inner_chunks)
        codecs = _materialized_sharded_codecs(inner_chunks, jpegxr_level=jpegxr_level)
    else:
        chunks = (min(12, output_shape[0]), output_shape[1], output_shape[2])
        codecs = [BytesCodec(), ZstdCodec(level=zstd_level)]
    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    output = zarr.open(
        str(output_path / "0"),
        mode="w",
        shape=output_shape,
        chunks=chunks,
        dtype=source.dtype,
        zarr_format=3,
        dimension_names=DIMENSIONS,
        codecs=codecs,
    )
    output.attrs["_ARRAY_DIMENSIONS"] = list(DIMENSIONS)
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": output_path.name,
            "axes": [_ome_axis(axis.upper()) for axis in DIMENSIONS],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [_ome_scale_transform(axes="ZYX", spacing_zyx=spacing_zyx)],
                }
            ],
        }
    ]
    root.attrs["squisher_materialization"] = {
        "source_path": str(source_path.resolve()),
        "source_channel": int(source_channel),
        "source_start_zyx": np.asarray(start_zyx, dtype=np.int64).tolist(),
        "source_stop_zyx": np.asarray(stop_zyx, dtype=np.int64).tolist(),
        "source_level": int(source_level),
        "source_factor_zyx": source_factors.tolist(),
        "remaining_factor_zyx": remaining_factors.tolist(),
        "output_codec": output_codec,
    }

    dtype = np.dtype(source.dtype)
    for output_z0 in range(0, output_shape[0], chunks[0]):
        output_z1 = min(output_z0 + chunks[0], output_shape[0])
        source_z0 = int(source_start[0]) + output_z0 * int(remaining_factors[0])
        source_z1 = int(source_start[0]) + output_z1 * int(remaining_factors[0])
        spatial_selection = (
            slice(source_z0, source_z1),
            slice(int(source_start[1]), int(source_stop[1])),
            slice(int(source_start[2]), int(source_stop[2])),
        )
        selection = (source_channel, *spatial_selection) if axes == "CZYX" else spatial_selection
        source_block = np.asarray(source[selection])
        if np.all(remaining_factors == 1):
            output[output_z0:output_z1] = source_block
        else:
            reduced = downsample_axis_blocks(
                source_block,
                tuple(int(value) for value in remaining_factors),
                reducer="mean",
            )
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                reduced = np.clip(reduced, info.min, info.max)
            output[output_z0:output_z1] = reduced.astype(dtype, copy=False)
    root.attrs["squisher_complete"] = True
    _write_materialization_completion(output_path)
    return list(output_shape)


def _materialize_fused_fixed_task(task: dict[str, Any]) -> list[int]:
    return _materialize_downsampled_channel_crop_ome_zarr(**task)


def _completed_materialization_shape(task: dict[str, Any]) -> list[int] | None:
    """Return the shape only when an existing store exactly matches this task."""
    import zarr

    output_path = Path(task["output_path"])
    if not output_path.exists():
        return None
    try:
        register_jpegxr_codec()
        root = zarr.open_group(str(output_path), mode="r")
        if root.attrs.get("squisher_complete") is not True:
            return None
        output = root["0"]
        metadata = root.attrs.get("squisher_materialization", {})
        completion = _read_json(output_path / "squisher.complete.json")
    except (KeyError, OSError, ValueError):
        return None
    expected_shape = (
        (np.asarray(task["stop_zyx"]) - np.asarray(task["start_zyx"]))
        // np.asarray(task["level_factor_zyx"])
    ).tolist()
    expected = {
        "source_path": str(Path(task["source_path"]).resolve()),
        "source_channel": int(task["source_channel"]),
        "source_start_zyx": np.asarray(task["start_zyx"], dtype=np.int64).tolist(),
        "source_stop_zyx": np.asarray(task["stop_zyx"], dtype=np.int64).tolist(),
        "output_codec": str(task["output_codec"]),
    }
    codec_names = [codec["name"] for codec in output.metadata.to_dict()["codecs"]]
    expected_codecs = (
        ["bytes", "zstd"] if task["output_codec"] == "zstd" else ["sharding_indexed"]
    )
    if (
        list(output.shape) != expected_shape
        or codec_names != expected_codecs
        or completion.get("payload") != _materialization_payload(output_path)
        or any(metadata.get(key) != value for key, value in expected.items())
    ):
        return None
    return expected_shape


def _materialize_native_source_group(tasks: list[dict[str, Any]]) -> list[list[int]]:
    """Decode each full native source slab once and distribute it to overlapping windows."""
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    if not tasks:
        return []
    source_path = Path(tasks[0]["source_path"])
    if any(Path(task["source_path"]) != source_path for task in tasks):
        raise ValueError("native materialization group contains multiple source paths")
    source, axes, source_level, source_factors = _ome_downsample_source(
        source_path, np.ones(3, dtype=np.int64)
    )
    if source_level != 0 or np.any(source_factors != 1):
        raise ValueError(f"native materialization unexpectedly selected source level {source_level}")
    source_channel = int(tasks[0]["source_channel"])
    if any(int(task["source_channel"]) != source_channel for task in tasks):
        raise ValueError("native materialization group contains multiple source channels")
    if axes == "ZYX" and source_channel != 0:
        raise ValueError(f"ZYX source only supports channel 0, got {source_channel}")
    if axes == "CZYX" and not 0 <= source_channel < int(source.shape[0]):
        raise ValueError(f"source channel {source_channel} is outside CZYX shape {source.shape}")
    spatial_indices = tuple(axes.index(axis) for axis in "ZYX")
    source_shape_zyx = np.asarray(source.shape, dtype=np.int64)[list(spatial_indices)]

    active: list[tuple[dict[str, Any], Any, Any]] = []
    for task in tasks:
        start = np.asarray(task["start_zyx"], dtype=np.int64)
        stop = np.asarray(task["stop_zyx"], dtype=np.int64)
        shape = tuple(int(value) for value in stop - start)
        if np.any(stop <= start):
            raise ValueError(f"invalid native crop {start.tolist()}:{stop.tolist()}")
        if np.any(start < 0) or np.any(stop > source_shape_zyx):
            raise ValueError(
                f"native crop {start.tolist()}:{stop.tolist()} is outside source shape "
                f"{source_shape_zyx.tolist()}"
            )
        recorded_axes = task["source_record"].get("axes")
        if isinstance(recorded_axes, str) and recorded_axes and recorded_axes != axes:
            raise ValueError(
                f"source tile {task['source_record'].get('tile')!r} records axes {recorded_axes!r}, "
                f"but {source_path} stores {axes!r}"
            )
        output_codec = str(task["output_codec"])
        if output_codec == "jpegxr":
            inner_chunks = _materialized_inner_chunks(axes="ZYX", shape=shape)
            chunks = _materialized_shard_chunks(axes="ZYX", shape=shape, inner_chunks=inner_chunks)
            codecs = _materialized_sharded_codecs(
                inner_chunks, jpegxr_level=float(task["jpegxr_level"])
            )
        elif output_codec == "zstd":
            chunks = (min(12, shape[0]), shape[1], shape[2])
            codecs = [BytesCodec(), ZstdCodec(level=int(task["zstd_level"]))]
        else:
            raise ValueError(f"unsupported materialization codec {output_codec!r}")
        output_path = Path(task["output_path"])
        root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
        output = zarr.open(
            str(output_path / "0"),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=source.dtype,
            zarr_format=3,
            dimension_names=DIMENSIONS,
            codecs=codecs,
        )
        output.attrs["_ARRAY_DIMENSIONS"] = list(DIMENSIONS)
        spacing = np.asarray(task["spacing_zyx"], dtype=np.float64)
        root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": output_path.name,
                "axes": [_ome_axis(axis.upper()) for axis in DIMENSIONS],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            _ome_scale_transform(axes="ZYX", spacing_zyx=spacing)
                        ],
                    }
                ],
            }
        ]
        root.attrs["squisher_materialization"] = {
            "source_path": str(source_path.resolve()),
            "source_channel": source_channel,
            "source_start_zyx": start.tolist(),
            "source_stop_zyx": stop.tolist(),
            "source_level": 0,
            "source_factor_zyx": [1, 1, 1],
            "remaining_factor_zyx": [1, 1, 1],
            "output_codec": output_codec,
        }
        active.append((task, root, output))

    z_axis = axes.index("Z")
    source_z_chunk = int(source.chunks[z_axis])
    first_z = min(int(np.asarray(task["start_zyx"])[0]) for task, _, _ in active)
    last_z = max(int(np.asarray(task["stop_zyx"])[0]) for task, _, _ in active)
    for source_z0 in range((first_z // source_z_chunk) * source_z_chunk, last_z, source_z_chunk):
        source_z1 = min(source_z0 + source_z_chunk, int(source.shape[z_axis]))
        intersecting = [
            item
            for item in active
            if int(np.asarray(item[0]["start_zyx"])[0]) < source_z1
            and int(np.asarray(item[0]["stop_zyx"])[0]) > source_z0
        ]
        if not intersecting:
            continue
        spatial_selection = (slice(source_z0, source_z1), slice(None), slice(None))
        selection = (source_channel, *spatial_selection) if axes == "CZYX" else spatial_selection
        slab = np.asarray(source[selection])
        for task, _root, output in intersecting:
            start = np.asarray(task["start_zyx"], dtype=np.int64)
            stop = np.asarray(task["stop_zyx"], dtype=np.int64)
            copy_z0 = max(source_z0, int(start[0]))
            copy_z1 = min(source_z1, int(stop[0]))
            output[
                copy_z0 - int(start[0]) : copy_z1 - int(start[0]),
                :,
                :,
            ] = slab[
                copy_z0 - source_z0 : copy_z1 - source_z0,
                int(start[1]) : int(stop[1]),
                int(start[2]) : int(stop[2]),
            ]

    shapes: list[list[int]] = []
    for task, root, output in active:
        root.attrs["squisher_complete"] = True
        _write_materialization_completion(Path(task["output_path"]))
        shapes.append(list(output.shape))
    return shapes


def export_fused_fixed_overlapping_materialized_chunks(
    *,
    source_registration_input: Path | None = None,
    source_summary_input: Path | None = None,
    moving_position_input: Path,
    moving_source_input: Path | None = None,
    output_dir: Path,
    source_channel: int = 0,
    core_shape_zyx: tuple[int, int, int] = (480, 480, 480),
    window_shape_zyx: tuple[int, int, int] = (528, 528, 528),
    level_factor_zyx: tuple[int, int, int] = (4, 4, 4),
    output_codec: Literal["zstd", "jpegxr"],
    zstd_level: int = 3,
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
    workers: int = 1,
    max_tiles: int | None = None,
    resume: bool = False,
) -> dict[str, Path]:
    """Rematerialize registered fused-fixed cores with cross-registration overlap."""
    moving_payload = _read_json(moving_position_input)
    moving_by_tile = _merge_moving_pixel_sources(
        _records_by_tile(moving_position_input),
        moving_source_input,
    )
    core_shape = np.asarray(core_shape_zyx, dtype=np.int64)
    window_shape = np.asarray(window_shape_zyx, dtype=np.int64)
    level_factor = np.asarray(level_factor_zyx, dtype=np.int64)
    if np.any(window_shape < core_shape) or np.all(window_shape == core_shape):
        raise ValueError(
            f"overlap window {window_shape.tolist()} must not be smaller than core "
            f"{core_shape.tolist()} on any axis and must be larger on at least one axis"
        )
    if np.any(window_shape % level_factor):
        raise ValueError(
            f"window shape {window_shape.tolist()} must be divisible by level factors {level_factor.tolist()}"
        )
    if (source_registration_input is None) == (source_summary_input is None):
        raise ValueError("exactly one of source_registration_input or source_summary_input is required")
    if source_summary_input is not None:
        source_payload = _fused_fixed_registration_from_summary(
            source_summary_input=source_summary_input,
            moving_by_tile=moving_by_tile,
            level_factor_zyx=level_factor,
        )
        source_metadata = {"source_summary_input": str(source_summary_input.resolve())}
    else:
        assert source_registration_input is not None
        source_payload = _read_json(source_registration_input)
        source_metadata = {"source_registration_input": str(source_registration_input.resolve())}
    source_records = source_payload.get("tiles")
    if not isinstance(source_records, list):
        raise ValueError("fused-fixed materialization source must contain a tiles list")
    selected_records = source_records if max_tiles is None else source_records[:max_tiles]

    output_records: list[dict[str, Any]] = []
    position_records: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for record in selected_records:
        moving_tile = str(record["moving_tile"])
        moving_record = moving_by_tile[moving_tile]
        moving_shape = _shape_zyx_from_record(moving_record)
        window_json = Path(str(record["method8_window_json"]))
        row = _read_json(window_json)
        core_start = np.asarray(row["moving_start_l0_zyx"], dtype=np.int64)
        core_stop = np.asarray(row["moving_stop_l0_zyx"], dtype=np.int64)
        recorded_shape = core_stop - core_start
        if np.array_equal(recorded_shape, core_shape):
            overlap_start, overlap_stop = _fused_fixed_overlapping_window_zyx(
                core_start_zyx=core_start,
                source_shape_zyx=moving_shape,
                core_shape_zyx=core_shape,
                window_shape_zyx=window_shape,
            )
        elif np.array_equal(recorded_shape, window_shape):
            overlap_start, overlap_stop = core_start, core_stop
        else:
            raise ValueError(
                f"{window_json} recorded shape {recorded_shape.tolist()} matches neither core "
                f"{core_shape.tolist()} nor overlap window {window_shape.tolist()}"
            )
        raw_scale = _vector_zyx(moving_record, "scale_um")
        stage_scale = _stage_scale_zyx(record)
        expected_stage_scale = raw_scale * level_factor.astype(np.float64)
        if not np.allclose(stage_scale, expected_stage_scale, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"tile {record.get('tile')!r} stage scale {stage_scale.tolist()} does not match "
                f"source scale times level factor {expected_stage_scale.tolist()}"
            )
        stage_translation = _stage_translation_zyx(record) + raw_scale * (overlap_start - core_start)
        source_path = Path(str(moving_record["path"]))
        if not source_path.is_absolute():
            source_path = moving_position_input.parent / source_path
        tile_name = str(record["tile"])
        output_path = output_dir / "materialized_tiles" / tile_name
        output_record = record | {
            "path": str(output_path),
            "shape": (window_shape // level_factor).tolist(),
            "axes": "ZYX",
            "spacing_um": _dict_zyx(np.abs(expected_stage_scale)),
            "channels": [str(source_channel)],
            "stage_translation_um": _dict_zyx(stage_translation),
            "stage_scale_um": _dict_zyx(stage_scale),
            "materialized_source_path": str(source_path),
            "materialized_source_start_zyx": overlap_start.tolist(),
            "materialized_source_stop_zyx": overlap_stop.tolist(),
            "registered_core_start_l0_zyx": core_start.tolist(),
            "registered_core_stop_l0_zyx": core_stop.tolist(),
        }
        output_records.append(output_record)
        position_record = {
            key: value
            for key, value in output_record.items()
            if key not in {"registered_affine", "stage_translation_um", "stage_scale_um"}
        }
        position_record["translation_um"] = _dict_zyx(stage_translation)
        position_record["scale_um"] = _dict_zyx(stage_scale)
        position_records.append(position_record)
        tasks.append(
            {
                "source_path": source_path,
                "output_path": output_path,
                "source_record": moving_record,
                "source_channel": source_channel,
                "start_zyx": overlap_start,
                "stop_zyx": overlap_stop,
                "level_factor_zyx": level_factor,
                "spacing_zyx": np.abs(expected_stage_scale),
                "output_codec": output_codec,
                "zstd_level": zstd_level,
                "jpegxr_level": jpegxr_level,
            }
        )

    expected_shape = (window_shape // level_factor).tolist()
    position_path = output_dir / "fused_fixed_materialized_chunks.positions.json"
    registration_path = output_dir / "fused_fixed_materialized_chunks.registration.json"
    summary_path = output_dir / "fused_fixed_materialized_chunks.summary.json"
    grid = {
        "policy": "core_shape_zyx steps with same-index window_shape_zyx context",
        "core_shape_zyx": core_shape.tolist(),
        "window_shape_zyx": window_shape.tolist(),
        "level_factor_zyx": level_factor.tolist(),
        "materialized_shape_zyx": expected_shape,
    }
    summary_payload = {
        "artifact_type": "squisher_lightsheet.fused_fixed_overlapping_materialization_summary.v1",
        **source_metadata,
        "moving_position_input": str(moving_position_input.resolve()),
        "moving_source_input": None if moving_source_input is None else str(moving_source_input.resolve()),
        "source_position_artifact_type": moving_payload.get("artifact_type"),
        "source_channel": source_channel,
        "output_codec": output_codec,
        "zstd_level": zstd_level,
        "jpegxr_level": jpegxr_level,
        "workers": workers,
        "source_level_policy": "deepest OME-Zarr pyramid level exactly dividing level_factor_zyx",
        "materialization_grid": grid,
        "tile_count": len(output_records),
    }
    if resume and summary_path.exists():
        previous = _read_json(summary_path)
        contract_keys = (
            "source_registration_input",
            "source_summary_input",
            "moving_position_input",
            "moving_source_input",
            "source_channel",
            "output_codec",
            "zstd_level",
            "jpegxr_level",
            "materialization_grid",
            "tile_count",
        )
        mismatches = [
            key for key in contract_keys if previous.get(key) != summary_payload.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"cannot resume {output_dir}: materialization plan differs for {mismatches}"
            )
    _write_json(
        position_path,
        {
            "artifact_type": "squisher_lightsheet.fused_fixed_overlapping_materialized_positions.v1",
            "units": "micrometer",
            **source_metadata,
            "moving_position_input": str(moving_position_input.resolve()),
            "materialization_grid": grid,
            "tiles": position_records,
        },
    )
    _write_json(
        registration_path,
        source_payload
        | {
            "artifact_type": "squisher_lightsheet.fused_fixed_overlapping_materialized_registration.v1",
            "input_dir": str((output_dir / "materialized_tiles").resolve()),
            **source_metadata,
            "moving_position_input": str(moving_position_input.resolve()),
            "materialization_grid": grid,
            "tiles": output_records,
        },
    )
    _write_json(summary_path, summary_payload | {"status": "in_progress"})

    written_shapes: list[list[int]] = []
    pending_tasks: list[dict[str, Any]] = []
    for task in tasks:
        completed_shape = _completed_materialization_shape(task) if resume else None
        if completed_shape is None:
            pending_tasks.append(task)
        else:
            written_shapes.append(completed_shape)
    completed_count = len(written_shapes)
    if completed_count:
        print(
            f"resuming with {completed_count}/{len(tasks)} overlapping fused-fixed views complete",
            flush=True,
        )

    if np.all(level_factor == 1):
        grouped: dict[tuple[Path, int], list[dict[str, Any]]] = {}
        for task in pending_tasks:
            key = (Path(task["source_path"]), int(task["source_channel"]))
            grouped.setdefault(key, []).append(task)
        task_groups = list(grouped.values())
        if workers == 1:
            group_results = map(_materialize_native_source_group, task_groups)
            for shapes in group_results:
                written_shapes.extend(shapes)
                completed_count += len(shapes)
                print(
                    f"materialized {completed_count}/{len(tasks)} overlapping fused-fixed views",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for shapes in pool.map(_materialize_native_source_group, task_groups):
                    written_shapes.extend(shapes)
                    completed_count += len(shapes)
                    print(
                        f"materialized {completed_count}/{len(tasks)} overlapping fused-fixed views",
                        flush=True,
                    )
    elif workers == 1:
        for shape in map(_materialize_fused_fixed_task, pending_tasks):
            written_shapes.append(shape)
            completed_count += 1
            if completed_count % 25 == 0 or completed_count == len(tasks):
                print(
                    f"materialized {completed_count}/{len(tasks)} overlapping fused-fixed views",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for shape in pool.map(_materialize_fused_fixed_task, pending_tasks):
                written_shapes.append(shape)
                completed_count += 1
                if completed_count % 25 == 0 or completed_count == len(tasks):
                    print(
                        f"materialized {completed_count}/{len(tasks)} overlapping fused-fixed views",
                        flush=True,
                    )
    if any(shape != expected_shape for shape in written_shapes) or len(written_shapes) != len(tasks):
        raise ValueError(
            f"materialized outputs do not all match expected {expected_shape}: "
            f"validated {len(written_shapes)}/{len(tasks)}"
        )

    _write_json(
        summary_path,
        summary_payload | {"status": "complete"},
    )
    return {
        "position": position_path.resolve(),
        "registration": registration_path.resolve(),
        "summary": summary_path.resolve(),
    }


def _method8_model_matrix(
    row: dict[str, Any], *, fixed_shape_zyx: np.ndarray, moving_shape_zyx: np.ndarray
) -> np.ndarray:
    if "full_matrix_zyx" not in row or "full_translation_zyx" not in row:
        raise ValueError(
            f"{row.get('fixed_tile')}->{row.get('moving_tile')} window is missing method8 full transform fields"
        )
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
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
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
            skipped.append(
                {
                    "window_json": str(path),
                    "status": row.get("status"),
                    "rejection_reason": row.get("rejection_reason"),
                }
            )
            continue
        if row.get("rejection_reason") == "quality_gate" and not include_quality_gate_rejected:
            skipped.append(
                {
                    "window_json": str(path),
                    "status": row.get("status"),
                    "rejection_reason": row.get("rejection_reason"),
                }
            )
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
            channel_shift_um = channel_shift_px * moving_scale
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
            jpegxr_level=jpegxr_level,
        )
        materialized_inner_chunks = _materialized_inner_chunks(
            axes=str(moving_record.get("axes")), shape=tuple(shape)
        )
        materialized_shard_chunks = _materialized_shard_chunks(
            axes=str(moving_record.get("axes")),
            shape=tuple(shape),
            inner_chunks=materialized_inner_chunks,
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
                "materialized_inner_chunks": list(materialized_inner_chunks),
                "materialized_shard_chunks": list(materialized_shard_chunks),
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
            "included_quality_gate_rejected": sum(
                1 for row in diagnostics if row.get("rejection_reason") == "quality_gate"
            ),
            "geometry_source": "moving_position_input_parent_tile_translation_plus_image10_local_window",
            "method8_usage": "accepted/skipped window mask plus per-window registered_affine transform for fusion",
            "diagnostics": diagnostics,
            "skipped": skipped,
        },
    )
    return {
        "position": position_path.resolve(),
        "registration": registration_path.resolve(),
        "summary": summary_path.resolve(),
    }
