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
   single-channel corrected OME-TIFF tiles.
4. Emit a corrected position JSON, and when a registration JSON is supplied,
   emit an adapted registration JSON with the same registered transforms but the
   corrected tile paths and crop-adjusted stage transforms.
5. Build/use separate L/R intensity-sorted BaSiC profiles on these corrected
   cropped tiles, then fuse them as a normal single-channel acquisition. Fusion
   must not know about the reference channel or spillover model.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


DIMENSIONS = ("z", "y", "x")
DEFAULT_COMPRESSION = 22610
DEFAULT_COMPRESSION_LEVEL = 0.7


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


def _shape_zyx(tile: legacy.TileMetadata, source_shape: tuple[int, ...]) -> tuple[int, int, int]:
    if tile.axes == "CZYX":
        return int(source_shape[1]), int(source_shape[2]), int(source_shape[3])
    if tile.axes == "ZYX":
        return int(source_shape[0]), int(source_shape[1]), int(source_shape[2])
    raise ValueError(f"Expected CZYX or ZYX tile axes, got {tile.axes!r}")


def _channel_view(array: Any, tile: legacy.TileMetadata, channel: int) -> Any:
    if tile.axes == "CZYX":
        if channel < 0 or channel >= int(array.shape[0]):
            raise ValueError(f"{tile.path} does not have channel {channel}; shape={tuple(array.shape)}")
        return array[channel]
    if tile.axes == "ZYX":
        if channel != 0:
            raise ValueError(f"{tile.path} is single-channel ZYX; requested channel {channel}")
        return array
    raise ValueError(f"Expected CZYX or ZYX tile axes, got {tile.axes!r}")


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


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_with_local_name(parent: ET.Element, name: str) -> ET.Element:
    for child in parent:
        if _local_name(child.tag) == name:
            return child
    raise ValueError(f"OME element {parent.tag!r} does not contain child {name!r}")


def _direct_children_with_local_name(parent: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in parent if _local_name(child.tag) == name]


def _selected_channel_element(pixels: ET.Element, target_channel: int) -> ET.Element:
    channels = _direct_children_with_local_name(pixels, "Channel")
    if not (0 <= target_channel < len(channels)):
        raise ValueError(f"Source OME metadata has {len(channels)} channels; requested {target_channel}")
    return channels[target_channel]


def _ome_xml_for_corrected_tile(
    *,
    source_path: Path,
    output_path: Path,
    output_shape_zyx: tuple[int, int, int],
    translation_um: dict[str, float],
    scale_um: dict[str, float],
    target_channel: int,
) -> str:
    import tifffile

    with tifffile.TiffFile(source_path) as tif:
        source_xml = tif.ome_metadata
    if source_xml is None:
        raise ValueError(f"{source_path} does not contain OME metadata")

    root = ET.fromstring(source_xml)
    image = _child_with_local_name(root, "Image")
    image.attrib["Name"] = output_path.name
    pixels = _child_with_local_name(image, "Pixels")
    selected_channel = _selected_channel_element(pixels, target_channel)

    for child in list(pixels):
        if _local_name(child.tag) in {"Channel", "TiffData", "Plane"}:
            pixels.remove(child)

    pixels.attrib.update(
        {
            "DimensionOrder": "XYCZT",
            "Type": "uint16",
            "SizeX": str(int(output_shape_zyx[2])),
            "SizeY": str(int(output_shape_zyx[1])),
            "SizeZ": str(int(output_shape_zyx[0])),
            "SizeC": "1",
            "SizeT": "1",
            "PhysicalSizeX": f"{abs(float(scale_um['x'])):.17g}",
            "PhysicalSizeY": f"{abs(float(scale_um['y'])):.17g}",
            "PhysicalSizeZ": f"{abs(float(scale_um['z'])):.17g}",
        }
    )

    selected_channel.attrib = dict(selected_channel.attrib)
    pixels.append(selected_channel)

    tiff_data = ET.Element(f"{pixels.tag.rsplit('}', 1)[0]}}}TiffData" if "}" in pixels.tag else "TiffData")
    tiff_data.attrib.update({"IFD": "0", "PlaneCount": str(int(output_shape_zyx[0]))})
    pixels.append(tiff_data)

    plane_tag = f"{pixels.tag.rsplit('}', 1)[0]}}}Plane" if "}" in pixels.tag else "Plane"
    for z_index in range(int(output_shape_zyx[0])):
        plane = ET.Element(plane_tag)
        plane.attrib.update(
            {
                "TheC": "0",
                "TheZ": str(z_index),
                "TheT": "0",
                "PositionX": f"{float(translation_um['x']):.17g}",
                "PositionY": f"{float(translation_um['y']):.17g}",
                "PositionZ": f"{float(translation_um['z'] + z_index * scale_um['z']):.17g}",
            }
        )
        pixels.append(plane)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


def _write_ome_zyx(
    output_path: Path,
    data: np.ndarray,
    *,
    source_path: Path,
    translation_um: dict[str, float],
    scale_um: dict[str, float],
    target_channel: int,
    compression: int | None,
    compression_level: float | None,
) -> None:
    import tifffile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ome_xml = _ome_xml_for_corrected_tile(
        source_path=source_path,
        output_path=output_path,
        output_shape_zyx=tuple(int(value) for value in data.shape),
        translation_um=translation_um,
        scale_um=scale_um,
        target_channel=target_channel,
    )
    compressionargs = None if compression_level is None else {"level": float(compression_level)}
    tifffile.imwrite(
        output_path,
        data,
        bigtiff=True,
        ome=False,
        photometric="minisblack",
        compression=compression,
        compressionargs=compressionargs,
        description=ome_xml.encode("utf-8"),
        metadata=None,
    )


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
    reference_gpu = reference_gpu[int(halo_before):z_stop]
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


def _output_tile_path(tile: legacy.TileMetadata, output_tile_dir: Path, *, target_channel: int, reference_channel: int) -> Path:
    stem = tile.path.name.removesuffix(".ome.tif").removesuffix(".tif")
    return output_tile_dir / f"{stem}.ch{target_channel}-minus-ch{reference_channel}.ome.tif"


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
        adapted.pop("path", None)
        adapted["source_tile"] = output_record["source_tile"]
        adapted["source_path"] = output_record["source_path"]
        adapted["source_view"] = output_record.get("side")
        adapted["stage_translation_um"] = output_record["translation_um"]
        adapted["stage_scale_um"] = output_record["scale_um"]
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
    crop_yx_px: int = 20,
    z_chunk: int = 64,
    compression: int | None = DEFAULT_COMPRESSION,
    compression_level: float | None = DEFAULT_COMPRESSION_LEVEL,
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

    output_records: list[dict[str, Any]] = []
    summary_records: list[SubtractedTileRecord] = []
    for tile_index, (tile, source_record) in enumerate(selected):
        array, store = legacy.open_tile_array(tile, source_level=source_level)
        try:
            source_shape = tuple(int(value) for value in array.shape)
            source_tile = legacy.fusion_tile_for_source_array(tile, source_shape, source_level=source_level)
            source_shape_zyx = _shape_zyx(tile, source_shape)
            target = _channel_view(array, tile, target_channel)
            reference = _channel_view(array, tile, reference_channel)
            output_path = _output_tile_path(
                tile,
                output_tile_dir,
                target_channel=target_channel,
                reference_channel=reference_channel,
            )
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
            )
            _write_ome_zyx(
                output_path,
                corrected,
                source_path=tile.path,
                translation_um=translation_um,
                scale_um=scale_um,
                target_channel=target_channel,
                compression=compression,
                compression_level=compression_level,
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
            "crop_yx_px": int(crop_yx_px),
            "z_chunk": int(z_chunk),
            "compression": compression,
            "compression_level": compression_level,
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
