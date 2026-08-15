from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
from typing import Any, Iterable
from xml.etree import ElementTree

import numpy as np
import zarr

from squisher.jpegxr_zarr import DEFAULT_JPEGXR_LEVEL, jpegxr_sharding_codec
from squisher_deconv.metadata import json_dumps_strict
from squisher_deconv.source import TiffLogicalSource


_DIMENSION_NAMES = ("c", "z", "y", "x")
_INNER_CHUNKS_CZYX = (1, 1, 240, 240)
_MAX_SHARD_SHAPE_CZYX = (1, 48, 960, 960)
_PYRAMID_FACTORS_YX = (2, 4)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def write_streamed_ome_zarr(
    path: Path,
    *,
    source: TiffLogicalSource,
    core_plane_chunks: Iterable[np.ndarray],
    output_mode: str,
    provenance: dict[str, Any],
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.partial")
    backup_path = path.with_name(f".{path.name}.backup")
    if partial_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing partial output {partial_path}")
    if backup_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup output {backup_path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output {path}; pass --overwrite to replace it.")

    dtype = np.dtype(np.uint16 if output_mode == "u16" else np.float32)
    json_dumps_strict(provenance, context="OME-Zarr deconvolution provenance")
    pyramid_factors = tuple(
        factor for factor in _PYRAMID_FACTORS_YX if source.height // factor and source.width // factor
    )
    attrs = _ome_zarr_attrs(
        source=source,
        provenance=provenance,
        output_mode=output_mode,
        pyramid_factors=pyramid_factors,
        jpegxr_level=jpegxr_level,
    )
    json_dumps_strict(attrs, context="OME-Zarr deconvolution metadata")
    try:
        root = zarr.open_group(str(partial_path), mode="w", zarr_format=3)
        root.attrs.update(attrs)
        arrays = {
            factor: _create_level_array(
                partial_path,
                dataset_path=str(level),
                shape=(
                    source.channels,
                    source.z_count,
                    source.height // factor,
                    source.width // factor,
                ),
                dtype=dtype,
                jpegxr_level=jpegxr_level,
            )
            for level, factor in enumerate((1, *pyramid_factors))
        }

        z_offset = 0
        for chunk in core_plane_chunks:
            if chunk.ndim != 3:
                raise ValueError(f"Expected flattened core plane chunk, got {chunk.shape}")
            if chunk.shape[0] % source.channels:
                raise ValueError(
                    f"Chunk has {chunk.shape[0]} flattened plane(s), not divisible by channels={source.channels}"
                )
            z_count = chunk.shape[0] // source.channels
            z_stop = z_offset + z_count
            if z_stop > source.z_count:
                raise ValueError(f"Chunk would write past z_count={source.z_count}: stop={z_stop}")
            zcyx = chunk.astype(dtype, copy=False).reshape(
                z_count, source.channels, source.height, source.width
            )
            czyx = np.moveaxis(zcyx, 1, 0)
            arrays[1][:, z_offset:z_stop, :, :] = czyx
            for factor in pyramid_factors:
                arrays[factor][:, z_offset:z_stop, :, :] = _downsample_xy_mean(czyx, factor, dtype=dtype)
            z_offset = z_stop
        if z_offset != source.z_count:
            raise ValueError(f"Expected to write {source.z_count} z plane(s), wrote {z_offset}")
        root.attrs["squisher_complete"] = True
        if path.exists():
            path.rename(backup_path)
        try:
            partial_path.rename(path)
        except BaseException:
            if backup_path.exists():
                backup_path.rename(path)
            raise
        if backup_path.exists():
            _remove_path(backup_path)
    except BaseException:
        shutil.rmtree(partial_path, ignore_errors=True)
        raise


def write_streamed_ome_zarr_with_sidecar(
    path: Path,
    *,
    sidecar_path: Path,
    sidecar_text: str,
    source: TiffLogicalSource,
    core_plane_chunks: Iterable[np.ndarray],
    output_mode: str,
    provenance: dict[str, Any],
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
    overwrite: bool = False,
) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    partial_sidecar = sidecar_path.with_name(f".{sidecar_path.name}.partial")
    staged_output = path.with_name(f".{path.name}.staged")
    output_backup = path.with_name(f".{path.name}.backup")
    sidecar_backup = sidecar_path.with_name(f".{sidecar_path.name}.backup")
    if partial_sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite existing partial sidecar {partial_sidecar}")
    for intermediate in (staged_output, output_backup, sidecar_backup):
        if intermediate.exists():
            raise FileExistsError(f"Refusing to overwrite existing transaction artifact {intermediate}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output {path}; pass --overwrite to replace it.")
    if sidecar_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing sidecar {sidecar_path}; pass --overwrite to replace it.")
    output_promoted = False
    try:
        partial_sidecar.write_text(sidecar_text)
        write_streamed_ome_zarr(
            staged_output,
            source=source,
            core_plane_chunks=core_plane_chunks,
            output_mode=output_mode,
            provenance=provenance,
            jpegxr_level=jpegxr_level,
            overwrite=False,
        )
        if path.exists():
            path.rename(output_backup)
        if sidecar_path.exists():
            sidecar_path.rename(sidecar_backup)
        staged_output.rename(path)
        output_promoted = True
        partial_sidecar.replace(sidecar_path)
    except BaseException as error:
        rollback_errors: list[BaseException] = []
        if output_promoted and path.exists():
            try:
                path.rename(staged_output)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if sidecar_backup.exists():
            try:
                sidecar_backup.rename(sidecar_path)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if output_backup.exists() and not path.exists():
            try:
                output_backup.rename(path)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        for intermediate in (partial_sidecar, staged_output):
            try:
                _remove_path(intermediate)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise BaseExceptionGroup("output publication and rollback failed", [error, *rollback_errors])
        raise
    if output_backup.exists():
        _remove_path(output_backup)
    if sidecar_backup.exists():
        _remove_path(sidecar_backup)


def _create_level_array(
    root_path: Path,
    *,
    dataset_path: str,
    shape: tuple[int, int, int, int],
    dtype: np.dtype[Any],
    jpegxr_level: float,
) -> Any:
    inner_chunks = tuple(min(size, target) for size, target in zip(shape, _INNER_CHUNKS_CZYX, strict=True))
    shard_shape = tuple(
        _bounded_shard_size(size, inner, maximum)
        for size, inner, maximum in zip(shape, inner_chunks, _MAX_SHARD_SHAPE_CZYX, strict=True)
    )
    return zarr.open(
        str(root_path / dataset_path),
        mode="w",
        shape=shape,
        chunks=shard_shape,
        dtype=dtype,
        zarr_format=3,
        dimension_names=_DIMENSION_NAMES,
        codecs=[jpegxr_sharding_codec(inner_chunks, level=jpegxr_level)],
    )


def _bounded_shard_size(size: int, inner: int, maximum: int) -> int:
    return max(inner, (min(size, maximum) // inner) * inner)


def _downsample_xy_mean(czyx: np.ndarray, factor: int, *, dtype: np.dtype[Any]) -> np.ndarray:
    y_stop = (czyx.shape[2] // factor) * factor
    x_stop = (czyx.shape[3] // factor) * factor
    reduced = czyx[:, :, :y_stop, :x_stop].reshape(
        czyx.shape[0],
        czyx.shape[1],
        y_stop // factor,
        factor,
        x_stop // factor,
        factor,
    ).mean(axis=(3, 5), dtype=np.float32)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        reduced = np.clip(np.rint(reduced), limits.min, limits.max)
    return reduced.astype(dtype, copy=False)


def _ome_zarr_attrs(
    *,
    source: TiffLogicalSource,
    provenance: dict[str, Any],
    output_mode: str,
    pyramid_factors: tuple[int, ...],
    jpegxr_level: float,
) -> dict[str, Any]:
    source_ome = _source_ome_metadata(source)
    spatial_scales = source_ome["spatial_scales_micrometer"]
    spatial_translation = source_ome["spatial_translation_micrometer"]
    axes = [{"name": "c", "type": "channel"}]
    axes.extend(
        {
            "name": axis,
            "type": "space",
            **({"unit": "micrometer"} if spatial_scales[index] is not None else {}),
        }
        for index, axis in enumerate(("z", "y", "x"))
    )
    base_scale = [1.0, *(1.0 if value is None else value for value in spatial_scales)]
    base_translation = [0.0, *(0.0 if value is None else value for value in spatial_translation)]
    include_translation = any(value is not None for value in spatial_translation)
    datasets = []
    for level, factor in enumerate((1, *pyramid_factors)):
        transforms: list[dict[str, Any]] = [
            {
                "type": "scale",
                "scale": [base_scale[0], base_scale[1], base_scale[2] * factor, base_scale[3] * factor],
            }
        ]
        if include_translation:
            transforms.append({"type": "translation", "translation": base_translation})
        datasets.append({"path": str(level), "coordinateTransformations": transforms})

    source_metadata = asdict(source.metadata)
    return {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "name": source_ome["image_name"] or source.path.stem,
                    "axes": axes,
                    "datasets": datasets,
                    "type": "mean",
                    "metadata": {"description": "XY mean pyramid generated directly from scale 0"},
                }
            ],
        },
        "squisher_deconv": {
            "output_mode": output_mode,
            "provenance": provenance,
            "source_metadata_hash": source.metadata.metadata_hash,
            "source_metadata": source_metadata,
            "source_ome": source_ome,
            "source_metadata_summary": _source_metadata_summary(source),
            "storage": {
                "format": "OME-Zarr",
                "zarr_format": 3,
                "ome_ngff_version": "0.5",
                "dimension_names": list(_DIMENSION_NAMES),
                "inner_chunks_czyx": list(_INNER_CHUNKS_CZYX),
                "max_shard_shape_czyx": list(_MAX_SHARD_SHAPE_CZYX),
                "pyramid_downsample_factors_yx": list(pyramid_factors),
                "pyramid_method": "mean_from_scale0",
                "codec": {"name": "squisher.jpegxr", "level": jpegxr_level, "checksum": "crc32c"},
            },
        },
    }


def _source_ome_metadata(source: TiffLogicalSource) -> dict[str, Any]:
    ome_xml = source.metadata.ome_xml
    if ome_xml is None:
        return {
            "available": False,
            "image_name": source.path.stem,
            "image_attributes": {},
            "pixels_attributes": {},
            "channels": [],
            "channel_names": [],
            "first_plane_attributes": {},
            "spatial_scales_micrometer": [None, None, None],
            "spatial_translation_micrometer": [None, None, None],
        }
    try:
        ome = ElementTree.fromstring(ome_xml)
    except ElementTree.ParseError as exc:
        raise ValueError(f"Invalid OME-XML metadata in {source.path}: {exc}") from exc
    image = next((element for element in ome.iter() if _local_name(element.tag) == "Image"), None)
    pixels = (
        next((element for element in image if _local_name(element.tag) == "Pixels"), None)
        if image is not None
        else None
    )
    if image is None or pixels is None:
        raise ValueError(f"OME-XML metadata in {source.path} does not contain an Image/Pixels element.")
    channels = [dict(element.attrib) for element in pixels if _local_name(element.tag) == "Channel"]
    first_plane = next((element for element in pixels if _local_name(element.tag) == "Plane"), None)
    plane_attributes = {} if first_plane is None else dict(first_plane.attrib)
    scales = [
        _ome_length_micrometers(
            pixels.attrib.get(f"PhysicalSize{axis.upper()}"),
            pixels.attrib.get(f"PhysicalSize{axis.upper()}Unit"),
            context=f"PhysicalSize{axis.upper()} in {source.path}",
        )
        for axis in ("z", "y", "x")
    ]
    translation = [
        _ome_length_micrometers(
            plane_attributes.get(f"Position{axis.upper()}"),
            plane_attributes.get(f"Position{axis.upper()}Unit"),
            context=f"Position{axis.upper()} in {source.path}",
        )
        for axis in ("z", "y", "x")
    ]
    return {
        "available": True,
        "image_name": image.attrib.get("Name", source.path.stem),
        "image_attributes": dict(image.attrib),
        "pixels_attributes": dict(pixels.attrib),
        "channels": channels,
        "channel_names": [channel.get("Name", channel.get("ID", "")) for channel in channels],
        "first_plane_attributes": plane_attributes,
        "spatial_scales_micrometer": scales,
        "spatial_translation_micrometer": translation,
    }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ome_length_micrometers(value: str | None, unit: str | None, *, context: str) -> float | None:
    if value is None:
        return None
    normalized_unit = "µm" if unit is None else unit.strip()
    factors = {
        "µm": 1.0,
        "um": 1.0,
        "micrometer": 1.0,
        "micrometre": 1.0,
        "nm": 1e-3,
        "mm": 1e3,
        "m": 1e6,
    }
    if normalized_unit not in factors:
        raise ValueError(f"Unsupported OME length unit {normalized_unit!r} for {context}.")
    return float(value) * factors[normalized_unit]


def _source_metadata_summary(source: TiffLogicalSource) -> dict[str, Any]:
    metadata = asdict(source.metadata)
    return {
        "source": str(source.path),
        "raw_shape": metadata["raw_shape"],
        "raw_dtype": metadata["raw_dtype"],
        "selected_tags": metadata["tags"],
        "has_shaped_metadata": bool(metadata["shaped_metadata"]),
        "has_imagej_metadata": metadata["imagej_metadata"] is not None,
        "has_ome_xml": metadata["ome_xml"] is not None,
    }
