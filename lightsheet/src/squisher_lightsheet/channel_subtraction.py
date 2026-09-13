"""Produce pre-fusion spillover-subtracted tiles.

Intended workflow for 514-minus-488 cleanup:

1. Register the physical 488/514/561/638 acquisition once at the track level,
   using the accepted 488 geometry as the target frame.
2. Measure the fine same-tile 514-to-488 offset at level 0 and compose that
   local shift onto the 488 rigid registration for the 514 channel.
3. Before BaSiC correction or multiview fusion, run this producer on the raw
   tiles: load target 514 and reference 488 from the same source tile, shift the
   reference on GPU into the target local coordinates, subtract the fitted
   spillover model, clip to non-negative values, crop the y/x border, and write
   single-channel corrected OME-Zarr tiles.
4. Emit a corrected position JSON, and when a registration JSON is supplied,
   emit an adapted registration JSON with the same registered transforms but the
   corrected tile paths and crop-adjusted stage transforms.
5. Build/use separate L/R intensity-sorted BaSiC profiles on these corrected
   cropped tiles, then fuse them as a normal single-channel acquisition. Fusion
   must not know about the reference channel or spillover model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet import tile_input
from squisher_lightsheet.residual_correction import (
    ResidualCorrection,
    load_residual_corrections,
)


DIMENSIONS = ("z", "y", "x")
DEFAULT_CHUNK_SHAPE_ZYX = (12, 480, 480)
DEFAULT_ZSTD_LEVEL = 3


@dataclass(frozen=True)
class SubtractedTileRecord:
    tile: str
    source_path: str
    output_path: str
    source_level: int
    source_shape_zyx: tuple[int, int, int]
    output_shape_zyx: tuple[int, int, int]
    translation_um: dict[str, float]
    scale_um: dict[str, float]
    side: str | None


@dataclass(frozen=True)
class ChannelSubtractionResult:
    position_output: Path
    summary_output: Path
    tile_count: int
    records: tuple[SubtractedTileRecord, ...]
    registration_output: Path | None = None


def _crop_translation_um(
    tile: legacy.TileMetadata,
    *,
    crop_yx_px: int,
) -> dict[str, float]:
    stage_scale = legacy.tile_stage_scale(tile)
    return {
        "z": float(tile.translation["z"]),
        "y": float(tile.translation["y"] + crop_yx_px * stage_scale["y"]),
        "x": float(tile.translation["x"] + crop_yx_px * stage_scale["x"]),
    }


def _write_ome_zarr_czyx(
    output_path: Path,
    data: np.ndarray,
    *,
    translation_um: dict[str, float],
    scale_um: dict[str, float],
    channel_name: str,
    zstd_level: int,
    chunk_shape_zyx: tuple[int, int, int],
) -> None:
    """Write one CZYX scale with physical placement and a final completion marker."""
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (1, *(int(value) for value in data.shape))
    chunks = (1, *(min(size, chunk) for size, chunk in zip(data.shape, chunk_shape_zyx, strict=True)))
    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": output_path.name.removesuffix(".ome.zarr"),
                        "axes": [
                            {"name": "c", "type": "channel"},
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [
                                            1.0,
                                            abs(float(scale_um["z"])),
                                            abs(float(scale_um["y"])),
                                            abs(float(scale_um["x"])),
                                        ],
                                    },
                                    {
                                        "type": "translation",
                                        "translation": [
                                            0.0,
                                            float(translation_um["z"]),
                                            float(translation_um["y"]),
                                            float(translation_um["x"]),
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
            "omero": {"channels": [{"label": channel_name}]},
            "squisher_complete": False,
        }
    )
    output = root.create_array(
        "0",
        shape=shape,
        chunks=chunks,
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("c", "z", "y", "x"),
        serializer=BytesCodec(),
        compressors=[ZstdCodec(level=zstd_level)],
    )
    output[0] = data
    root.attrs["squisher_complete"] = True


def _subtract_slab_gpu(
    target_slab: np.ndarray,
    reference_halo: np.ndarray,
    *,
    halo_before: int,
    halo_after: int,
    reference_shift_zyx_px: tuple[float, float, float],
    alpha: float,
    beta: float,
    target_background: float,
    reference_background: float,
    crop_yx_px: int,
    output_dtype: np.dtype,
) -> np.ndarray:
    import cupy as cp
    from cupyx.scipy import ndimage as cndi

    target_gpu = cp.asarray(target_slab, dtype=cp.float32)
    reference_gpu = cp.asarray(reference_halo, dtype=cp.float32)
    axis_shift = tuple(-float(value) for value in reference_shift_zyx_px)
    if any(abs(value) > 1e-9 for value in axis_shift):
        reference_gpu = cndi.shift(
            reference_gpu,
            shift=axis_shift,
            order=1,
            mode="nearest",
            prefilter=False,
        )
    z_stop = reference_gpu.shape[0] - int(halo_after)
    reference_gpu = reference_gpu[int(halo_before) : z_stop]
    corrected = (target_gpu - float(target_background)) - float(alpha) * cp.maximum(
        reference_gpu - float(reference_background),
        0.0,
    ) - float(beta)
    cp.maximum(corrected, 0.0, out=corrected)
    if crop_yx_px > 0:
        corrected = corrected[:, crop_yx_px:-crop_yx_px, crop_yx_px:-crop_yx_px]
    if np.issubdtype(output_dtype, np.integer):
        cp.rint(corrected, out=corrected)
        cp.clip(corrected, 0, np.iinfo(output_dtype).max, out=corrected)
        corrected = corrected.astype(output_dtype, copy=False)
    else:
        corrected = corrected.astype(np.float32, copy=False)
    return cp.asnumpy(corrected)


def subtract_spillover_array_gpu(
    target: Any,
    reference: Any,
    *,
    reference_shift_zyx_px: tuple[float, float, float],
    alpha: float,
    beta: float,
    target_background: float,
    reference_background: float,
    crop_yx_px: int,
    z_chunk: int,
    output_dtype: np.dtype,
    target_correction: ResidualCorrection | None = None,
    reference_correction: ResidualCorrection | None = None,
) -> np.ndarray:
    source_shape = tuple(int(value) for value in target.shape)
    if len(source_shape) != 3:
        raise ValueError(f"Expected target ZYX data, got shape={source_shape}")
    if tuple(int(value) for value in reference.shape) != source_shape:
        raise ValueError(
            f"Target and reference shapes must match; got target={source_shape}, "
            f"reference={tuple(int(value) for value in reference.shape)}"
        )
    if crop_yx_px < 0:
        raise ValueError("crop_yx_px must be non-negative")
    if crop_yx_px * 2 >= min(source_shape[1], source_shape[2]):
        raise ValueError(f"crop_yx_px={crop_yx_px} removes the whole y/x tile from shape={source_shape}")
    if z_chunk <= 0:
        raise ValueError("z_chunk must be positive")

    output_shape = (
        source_shape[0],
        source_shape[1] - 2 * crop_yx_px,
        source_shape[2] - 2 * crop_yx_px,
    )
    output = np.empty(output_shape, dtype=output_dtype)
    z_halo = int(math.ceil(abs(float(reference_shift_zyx_px[0])))) + 1
    for z0 in range(0, source_shape[0], z_chunk):
        z1 = min(source_shape[0], z0 + z_chunk)
        halo_z0 = max(0, z0 - z_halo)
        halo_z1 = min(source_shape[0], z1 + z_halo)
        target_slab = np.asarray(target[z0:z1], dtype=output_dtype)
        reference_halo = np.asarray(reference[halo_z0:halo_z1], dtype=output_dtype)
        if target_correction is not None:
            target_slab = target_slab * target_correction.block(
                z_slice=slice(z0, z1),
                y_slice=slice(0, source_shape[1]),
                x_slice=slice(0, source_shape[2]),
            )
        if reference_correction is not None:
            reference_halo = reference_halo * reference_correction.block(
                z_slice=slice(halo_z0, halo_z1),
                y_slice=slice(0, source_shape[1]),
                x_slice=slice(0, source_shape[2]),
            )
        corrected = _subtract_slab_gpu(
            target_slab,
            reference_halo,
            halo_before=z0 - halo_z0,
            halo_after=halo_z1 - z1,
            reference_shift_zyx_px=reference_shift_zyx_px,
            alpha=alpha,
            beta=beta,
            target_background=target_background,
            reference_background=reference_background,
            crop_yx_px=crop_yx_px,
            output_dtype=output.dtype,
        )
        output[z0:z1] = corrected
    return output


def _output_tile_path(source_path: Path, output_tile_dir: Path) -> Path:
    identity = hashlib.sha256(str(source_path).encode()).hexdigest()[:8]
    return output_tile_dir / f"tile-{identity}.ome.zarr"


def _adapt_registration_payload(
    *,
    registration_input: Path,
    output_registration: Path,
    output_tile_dir: Path,
    output_records: list[dict[str, Any]],
) -> Path:
    payload = json.loads(registration_input.read_text())
    registration_records = payload.get("tiles")
    if not isinstance(registration_records, list) or not registration_records:
        raise ValueError(f"{registration_input} must contain a non-empty tiles list")

    outputs_by_source_name = {
        Path(str(record["source_path"])).name: record
        for record in output_records
    }
    outputs_by_source_tile = {
        str(record["source_tile"]): record
        for record in output_records
    }
    adapted_tiles: list[dict[str, Any]] = []
    for record in registration_records:
        if not isinstance(record, dict):
            raise ValueError(f"{registration_input} tile records must be objects")
        raw_tile = str(record.get("tile") or Path(str(record.get("path", ""))).name)
        output_record = outputs_by_source_tile.get(raw_tile) or outputs_by_source_name.get(raw_tile)
        if output_record is None:
            continue
        adapted = dict(record)
        adapted["tile"] = output_record["tile"]
        adapted["path"] = output_record["path"]
        adapted["source_tile"] = output_record["source_tile"]
        adapted["source_path"] = output_record["source_path"]
        adapted["source_view"] = output_record.get("side")
        adapted["stage_translation_um"] = output_record["translation_um"]
        adapted["stage_scale_um"] = output_record["scale_um"]
        adapted["axes"] = output_record["axes"]
        adapted["shape"] = output_record["shape"]
        adapted["channels"] = output_record["channels"]
        adapted_tiles.append(adapted)

    if len(adapted_tiles) != len(output_records):
        raise ValueError(
            f"{registration_input} matched {len(adapted_tiles)} of {len(output_records)} corrected tiles"
        )

    adapted_payload = dict(payload)
    adapted_payload["input_dir"] = str(output_tile_dir)
    adapted_payload["derived_from"] = {
        "registration_input": str(registration_input.resolve()),
        "previous": payload.get("derived_from"),
        "operation": "channel_subtraction_tile_path_adaptation",
    }
    adapted_payload["tiles"] = adapted_tiles
    output_registration.parent.mkdir(parents=True, exist_ok=True)
    output_registration.write_text(json.dumps(adapted_payload, indent=2) + "\n")
    return output_registration


def subtract_channel_tiles(
    *,
    position_input: Path,
    output_dir: Path,
    output_position: Path | None = None,
    registration_input: Path | None = None,
    output_registration: Path | None = None,
    target_channel: int,
    reference_channel: int,
    source_level: int,
    reference_shift_zyx_px: tuple[float, float, float],
    alpha: float,
    beta: float,
    target_background: float,
    reference_background: float,
    target_residual_correction: Path | None = None,
    reference_residual_correction: Path | None = None,
    crop_yx_px: int = 20,
    z_chunk: int = 64,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    chunk_shape_zyx: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE_ZYX,
    overwrite: bool = False,
    limit_tiles: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> ChannelSubtractionResult:
    if target_channel == reference_channel:
        raise ValueError("target_channel and reference_channel must differ")
    if source_level < 0:
        raise ValueError("source_level must be non-negative")
    if limit_tiles is not None and limit_tiles <= 0:
        raise ValueError("limit_tiles must be positive when provided")
    if not 1 <= zstd_level <= 22:
        raise ValueError("zstd_level must be between 1 and 22")
    if len(chunk_shape_zyx) != 3 or any(value <= 0 for value in chunk_shape_zyx):
        raise ValueError("chunk_shape_zyx must contain three positive values")

    output_dir = output_dir.resolve()
    output_tile_dir = output_dir / "tiles"
    output_position = (output_position or output_dir / "subtracted.positions.json").resolve()
    if registration_input is not None:
        output_registration = (output_registration or output_dir / "subtracted.registration.json").resolve()
    summary_output = output_dir / "channel_subtraction_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tile_dir.mkdir(parents=True, exist_ok=True)

    input_payload = json.loads(position_input.read_text())
    records = input_payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{position_input} must contain a non-empty tiles list")
    tiles = legacy.read_position_input_tiles(position_input.resolve())
    selected = list(zip(tiles, records, strict=True))
    if limit_tiles is not None:
        selected = selected[:limit_tiles]

    if (target_residual_correction is None) != (reference_residual_correction is None):
        raise ValueError("Target and reference residual corrections must be supplied together")
    target_corrections: dict[str, ResidualCorrection] = {}
    reference_corrections: dict[str, ResidualCorrection] = {}
    if target_residual_correction is not None and reference_residual_correction is not None:
        correction_sources = [tile.path for tile, _ in selected]
        correction_shapes = [tile_input.spatial_shape_zyx(tile.shape, tile.axes) for tile, _ in selected]
        target_corrections = load_residual_corrections(
            target_residual_correction,
            sources=correction_sources,
            shapes_zyx=correction_shapes,
            channel=target_channel,
        )
        reference_corrections = load_residual_corrections(
            reference_residual_correction,
            sources=correction_sources,
            shapes_zyx=correction_shapes,
            channel=reference_channel,
        )

    output_records: list[dict[str, Any]] = []
    summary_records: list[SubtractedTileRecord] = []
    for tile_index, (tile, source_record) in enumerate(selected):
        array, store = legacy.open_tile_array(tile, source_level=source_level)
        try:
            source_shape = tuple(int(value) for value in array.shape)
            source_tile = legacy.fusion_tile_for_source_array(tile, source_shape, source_level=source_level)
            source_shape_zyx = tile_input.spatial_shape_zyx(source_shape, source_tile.axes)
            target = tile_input.channel_view(array, source_tile.axes, target_channel, path=tile.path)
            reference = tile_input.channel_view(array, source_tile.axes, reference_channel, path=tile.path)
            resolved_tile_path = tile.path.resolve()
            output_path = _output_tile_path(resolved_tile_path, output_tile_dir)
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")
            translation_um = _crop_translation_um(source_tile, crop_yx_px=crop_yx_px)
            scale_um = {dim: float(legacy.tile_stage_scale(source_tile)[dim]) for dim in DIMENSIONS}
            if progress is not None:
                progress(
                    f"Subtracting tile {tile_index + 1}/{len(selected)} "
                    f"{tile.path.name} ch{target_channel}-alpha*ch{reference_channel} "
                    f"source_level={source_level} shape_zyx={source_shape_zyx}"
                )
            corrected = subtract_spillover_array_gpu(
                target,
                reference,
                reference_shift_zyx_px=reference_shift_zyx_px,
                alpha=alpha,
                beta=beta,
                target_background=target_background,
                reference_background=reference_background,
                crop_yx_px=crop_yx_px,
                z_chunk=z_chunk,
                output_dtype=np.dtype(target.dtype),
                target_correction=target_corrections.get(str(resolved_tile_path)),
                reference_correction=reference_corrections.get(str(resolved_tile_path)),
            )
            _write_ome_zarr_czyx(
                output_path,
                corrected,
                translation_um=translation_um,
                scale_um=scale_um,
                channel_name=source_tile.channels[target_channel],
                zstd_level=zstd_level,
                chunk_shape_zyx=chunk_shape_zyx,
            )
            source_tile_name = str(source_record.get("tile") or tile.path.name)
            side = source_record.get("side") if isinstance(source_record, dict) else None
            output_shape_zyx = tuple(int(value) for value in corrected.shape)
            output_record = {
                "tile": output_path.name,
                "source_tile": source_tile_name,
                "source_path": str(tile.path),
                "side": side,
                "path": str(output_path),
                "axes": "CZYX",
                "shape": [1, *output_shape_zyx],
                "channels": [source_tile.channels[target_channel]],
                "translation_um": translation_um,
                "scale_um": scale_um,
            }
            output_records.append(output_record)
            summary_records.append(
                SubtractedTileRecord(
                    tile=source_tile_name,
                    source_path=str(tile.path),
                    output_path=str(output_path),
                    source_level=source_level,
                    source_shape_zyx=source_shape_zyx,
                    output_shape_zyx=output_shape_zyx,
                    translation_um=translation_um,
                    scale_um=scale_um,
                    side=side if isinstance(side, str) else None,
                )
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()

    output_payload = {
        "artifact_type": "lightsheet.position.v1",
        "units": "micrometer",
        "source_position_input": str(position_input.resolve()),
        "subtraction": {
            "target_channel": int(target_channel),
            "reference_channel": int(reference_channel),
            "source_level": int(source_level),
            "reference_shift_zyx_px": [float(value) for value in reference_shift_zyx_px],
            "alpha": float(alpha),
            "beta": float(beta),
            "target_background": float(target_background),
            "reference_background": float(reference_background),
            "target_residual_correction": (
                None if target_residual_correction is None else str(target_residual_correction.resolve())
            ),
            "reference_residual_correction": (
                None
                if reference_residual_correction is None
                else str(reference_residual_correction.resolve())
            ),
            "crop_yx_px": int(crop_yx_px),
            "z_chunk": int(z_chunk),
            "zstd_level": int(zstd_level),
            "chunk_shape_zyx": [int(value) for value in chunk_shape_zyx],
        },
        "tiles": output_records,
    }
    output_position.parent.mkdir(parents=True, exist_ok=True)
    output_position.write_text(json.dumps(output_payload, indent=2) + "\n")
    written_registration = None
    if registration_input is not None:
        if output_registration is None:
            raise ValueError("output_registration was not resolved")
        written_registration = _adapt_registration_payload(
            registration_input=registration_input.resolve(),
            output_registration=output_registration,
            output_tile_dir=output_tile_dir,
            output_records=output_records,
        )

    summary_payload = {
        "position_output": str(output_position),
        "registration_output": None if written_registration is None else str(written_registration),
        "tile_count": len(summary_records),
        "tiles": [
            {
                "tile": record.tile,
                "source_path": record.source_path,
                "output_path": record.output_path,
                "source_level": record.source_level,
                "source_shape_zyx": list(record.source_shape_zyx),
                "output_shape_zyx": list(record.output_shape_zyx),
                "translation_um": record.translation_um,
                "scale_um": record.scale_um,
                "side": record.side,
            }
            for record in summary_records
        ],
    }
    summary_output.write_text(json.dumps(summary_payload, indent=2) + "\n")
    return ChannelSubtractionResult(
        position_output=output_position,
        summary_output=summary_output,
        tile_count=len(summary_records),
        records=tuple(summary_records),
        registration_output=written_registration,
    )
