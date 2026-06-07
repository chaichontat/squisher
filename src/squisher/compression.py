from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from typing import Any, NotRequired, TypedDict
import xml.etree.ElementTree as ET

import imagecodecs
from loguru import logger
import numpy as np
import numpy.typing as npt
import tifffile
from tifffile import OmeXml, TiffFile, TiffWriter


JPEG_XR_KWARGS = {"photometric": "minisblack", "compression": 22610}
CZI_PROGRESS_INTERVAL = 100
DEFAULT_ZARR_CHUNKS_TCZYX = (1, 1, 1, 4096, 4096)
DEFAULT_MIN_ZARR_CHUNK_PIXELS = 16 * 1024 * 1024
OME_ORIGINAL_METADATA_NAMESPACE = "openmicroscopy.org/OriginalMetadata"
CZI_RAW_METADATA_NAMESPACE = "fishtools/czi/raw-metadata"
OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
SUPPORTED_MULTI_DIMS = {"X", "Y", "Z", "C", "T", "M"}
SUPPORTED_SINGLETON_DIMS = {"S", "R", "I", "H", "V", "B"}
OUTPUT_FORMATS = {"ome-tiff", "ome-zarr"}
ZARR_COMPRESSORS = {"jpegxr", "jpegxl"}


class CziTile(TypedDict):
    index: int
    scene: int
    mosaic_index: NotRequired[int]
    index_zyx: NotRequired[tuple[int, int, int]]
    x: int
    y: int
    width: int
    height: int
    position_x: float
    position_y: float


def compress_czi_to_ome_tiff(
    path: Path,
    *,
    level: float,
    out_dir: Path | None = None,
    output_format: str = "ome-tiff",
    tile_size: int = 512,
    zarr_chunks: tuple[int, int, int, int, int] = DEFAULT_ZARR_CHUNKS_TCZYX,
    min_zarr_chunk_pixels: int = DEFAULT_MIN_ZARR_CHUNK_PIXELS,
    zarr_compressor: str = "jpegxr",
    maxworkers: int = 4,
    tile_workers: int = 1,
    resume: bool = False,
    overwrite: bool = False,
    thumbnails: bool = True,
    thumbnail_size: int = 512,
) -> bool:
    from aicspylibczi import CziFile

    if path.suffix.lower() != ".czi":
        raise ValueError(f"Expected a .czi input file, got {path}")
    _validate_output_format(output_format)
    if tile_size % 16 != 0:
        raise ValueError(f"TIFF tile size must be a multiple of 16, got {tile_size}")
    if output_format == "ome-zarr":
        _validate_zarr_chunks(
            zarr_chunks,
            min_zarr_chunk_pixels=min_zarr_chunk_pixels,
            compressor=zarr_compressor,
        )
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    tiles = _czi_tiles(path)
    tile_count = len(tiles)
    output_dir = _czi_output_dir(path, out_dir=out_dir, tile_count=tile_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [_czi_tile_output_path(path, tile, tile_count, output_dir=output_dir, output_format=output_format) for tile in tiles]
    logger.info(
        "Compressing {} tile(s) from {} to {} as {} with level={} tile_size={} resume={}",
        tile_count,
        path,
        output_dir,
        output_format,
        level,
        tile_size,
        resume,
    )
    existing = [out for out in outputs if out.exists()]
    if existing and overwrite:
        for out in existing:
            logger.warning("Overwriting existing {} output {}", output_format, out)
    elif existing and not resume:
        raise FileExistsError(f"Refusing to overwrite existing {output_format} output(s): {existing}")

    reader = CziFile(path)
    dims = reader.get_dims_shape()[0]
    plane_count = _czi_plane_count(dims)
    if resume:
        pending_tiles = []
        incomplete_existing = []
        for tile, out in zip(tiles, outputs, strict=True):
            if _is_complete_output(out, output_format=output_format, ome_shape=_czi_ome_shape(dims, tile)):
                continue
            pending_tiles.append(tile)
            if out.exists():
                incomplete_existing.append(out)
        if incomplete_existing:
            raise FileExistsError(f"Refusing to overwrite incomplete {output_format} output(s): {incomplete_existing}")
    else:
        pending_tiles = tiles
    logger.info("{} complete tile(s), {} pending tile(s)", tile_count - len(pending_tiles), len(pending_tiles))

    if thumbnails:
        _validate_thumbnail_size(thumbnail_size)

    if not pending_tiles:
        logger.info("No pending tile {} outputs for {}", output_format, path)
        if thumbnails:
            for out in outputs:
                write_center_z_thumbnail(out, max_size=thumbnail_size, overwrite=False)
        return True

    effective_tile_workers, effective_maxworkers = _effective_czi_workers(
        len(pending_tiles), tile_workers, maxworkers
    )
    settings = _compression_provenance(
        path,
        output_dir=output_dir,
        output_format=output_format,
        level=level,
        tile_size=tile_size,
        zarr_chunks=zarr_chunks,
        min_zarr_chunk_pixels=min_zarr_chunk_pixels,
        zarr_compressor=zarr_compressor,
        requested_tiff_maxworkers=maxworkers,
        effective_tiff_maxworkers=effective_maxworkers,
        requested_tile_workers=tile_workers,
        effective_tile_workers=effective_tile_workers,
        resume=resume,
        overwrite=overwrite,
        thumbnails=thumbnails,
        thumbnail_size=thumbnail_size,
        tile_count=tile_count,
        plane_count=plane_count,
    )
    logger.info(
        "Using {} CZI tile worker(s) and {} codec worker(s) per tile",
        effective_tile_workers,
        effective_maxworkers,
    )
    if effective_tile_workers == 1:
        for tile in pending_tiles:
            out = _write_czi_tile(
                reader,
                path,
                tile=tile,
                tile_count=tile_count,
                output_dir=output_dir,
                output_format=output_format,
                level=level,
                tile_size=tile_size,
                zarr_chunks=zarr_chunks,
                zarr_compressor=zarr_compressor,
                maxworkers=effective_maxworkers,
                dims=dims,
                provenance=settings,
            )
            if thumbnails:
                write_center_z_thumbnail(out, max_size=thumbnail_size, overwrite=True)
        if thumbnails and resume:
            _write_missing_resume_thumbnails(outputs, pending_tiles, tiles, thumbnail_size)
        logger.success("Finished compression for {}", path)
        return True

    del reader
    pool = ProcessPoolExecutor(effective_tile_workers, mp_context=get_context("spawn"))
    futures = []
    try:
        futures = [
            pool.submit(
                _write_czi_tile_process,
                path,
                tile=tile,
                tile_count=tile_count,
                output_dir=output_dir,
                output_format=output_format,
                level=level,
                tile_size=tile_size,
                zarr_chunks=zarr_chunks,
                zarr_compressor=zarr_compressor,
                maxworkers=effective_maxworkers,
                provenance=settings,
            )
            for tile in pending_tiles
        ]
        for future in as_completed(futures):
            out = future.result()
            logger.info("Completed {}", out.name)
            if thumbnails:
                write_center_z_thumbnail(out, max_size=thumbnail_size, overwrite=True)
    except BaseException as exc:
        for future in futures:
            future.cancel()
        if isinstance(exc, KeyboardInterrupt):
            logger.warning("KeyboardInterrupt received; cancelling pending CZI tile compression workers.")
        elif isinstance(exc, Exception):
            logger.exception("CZI tile compression failed; cancelling pending worker processes.")
        else:
            logger.warning("CZI tile compression interrupted; cancelling pending worker processes.")
        terminate_workers = getattr(pool, "terminate_workers", None)
        if terminate_workers is None:
            processes = list((getattr(pool, "_processes", None) or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            pool.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                process.join(timeout=1)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
        else:
            terminate_workers()
        raise
    else:
        pool.shutdown(wait=True)
    if thumbnails and resume:
        _write_missing_resume_thumbnails(outputs, pending_tiles, tiles, thumbnail_size)
    logger.success("Finished compression for {}", path)
    return True


def verify_czi_ome_tiff_outputs(
    path: Path,
    *,
    out_dir: Path | None = None,
    decode_samples: bool = False,
    max_sample_mae: float | None = None,
    max_sample_max_abs: float | None = None,
) -> bool:
    from aicspylibczi import CziFile

    if path.suffix.lower() != ".czi":
        raise ValueError(f"Expected a .czi input file, got {path}")

    reader = CziFile(path)
    dims = reader.get_dims_shape()[0]
    _validate_supported_czi_dims(path, dims)
    tiles = _czi_tiles(path)
    tile_count = len(tiles)
    plane_count = _czi_plane_count(dims)
    output_dir = _czi_output_dir(path, out_dir=out_dir, tile_count=tile_count)
    logger.info("Verifying {} tile output(s) for {} in {}", tile_count, path, output_dir)

    errors = []
    for tile in tiles:
        out = _czi_tile_ome_tiff_path(path, tile, tile_count, output_dir=output_dir)
        if not out.exists():
            errors.append(f"{out.name}: missing")
            continue
        tile_errors = _verify_ome_tiff(path, out, tile=tile, plane_count=plane_count)
        errors.extend(tile_errors)
        if decode_samples and not tile_errors:
            errors.extend(
                _decode_sample_pages(
                    out,
                    reader=reader,
                    dims=dims,
                    tile=tile,
                    max_sample_mae=max_sample_mae,
                    max_sample_max_abs=max_sample_max_abs,
                )
            )

    if errors:
        logger.error("OME-TIFF verification failed for {} with {} error(s)", path, len(errors))
        preview = "\n".join(f"- {error}" for error in errors[:20])
        suffix = f"\n... {len(errors) - 20} more error(s)" if len(errors) > 20 else ""
        raise ValueError(f"OME-TIFF verification failed for {path}:\n{preview}{suffix}")
    logger.success("Verified {} OME-TIFF output(s) for {}", tile_count, path)
    return True


def _write_czi_tile(
    reader: Any,
    path: Path,
    *,
    tile: CziTile,
    tile_count: int,
    output_dir: Path,
    output_format: str,
    level: float,
    tile_size: int,
    zarr_chunks: tuple[int, int, int, int, int],
    zarr_compressor: str,
    maxworkers: int,
    dims: dict[str, tuple[int, int]],
    provenance: dict[str, str],
) -> Path:
    if output_format == "ome-zarr":
        return _write_czi_tile_to_ome_zarr(
            reader,
            path,
            tile=tile,
            tile_count=tile_count,
            output_dir=output_dir,
            level=level,
            chunks=zarr_chunks,
            compressor=zarr_compressor,
            dims=dims,
            provenance=provenance,
        )
    if output_format != "ome-tiff":
        raise ValueError(f"Unsupported output format {output_format!r}")

    t_indexes = _czi_dim_indexes(dims, "T")
    c_indexes = _czi_dim_indexes(dims, "C")
    z_indexes = _czi_dim_indexes(dims, "Z")
    out = _czi_tile_ome_tiff_path(path, tile, tile_count, output_dir=output_dir)
    write_out = _temporary_output_path(out)
    plane_count = _czi_plane_count(dims)
    ome_shape = (len(t_indexes), len(c_indexes), len(z_indexes), tile["height"], tile["width"])

    logger.info(
        f"Writing tile {tile['index'] + 1}/{tile_count} "
        f"at x={_format_signed_coordinate(tile['x'])} y={_format_signed_coordinate(tile['y'])} "
        f"size={tile['width']}x{tile['height']} to {out.name}"
    )

    _remove_output_if_exists(write_out)
    try:
        plane_index = 0
        raw_bytes = 0
        with TiffWriter(write_out, bigtiff=True, mode="x") as writer:
            for t in t_indexes:
                for c in c_indexes:
                    for z in z_indexes:
                        plane = _as_grayscale_plane(reader.read_image(**_czi_read_kwargs(dims, tile, t, c, z))[0])
                        raw_bytes += plane.nbytes
                        writer.write(
                            plane,
                            description=_first_plane_ome_xml(path, reader, tile, ome_shape, plane.dtype, provenance)
                            if plane_index == 0
                            else None,
                            metadata=None,
                            tile=(tile_size, tile_size),
                            **JPEG_XR_KWARGS,
                            compressionargs={"level": _compression_level(level)},
                            maxworkers=maxworkers,
                        )
                        plane_index += 1
                        if plane_index == 1 or plane_index % CZI_PROGRESS_INTERVAL == 0 or plane_index == plane_count:
                            logger.info(f"{out.name}: wrote plane {plane_index}/{plane_count} (T={t}, C={c}, Z={z})")
        _replace_output(write_out, out)
    except BaseException:
        _remove_output_if_exists(write_out)
        raise
    compressed_bytes = out.stat().st_size
    ratio = raw_bytes / compressed_bytes if compressed_bytes else 0.0
    saved_fraction = 1.0 - (compressed_bytes / raw_bytes) if raw_bytes else 0.0
    logger.success(
        f"Finished {out.name} compressed={_format_bytes(compressed_bytes)} "
        f"raw={_format_bytes(raw_bytes)} ratio={ratio:.2f}:1 saved={saved_fraction:.1%}"
    )
    return out


def _write_czi_tile_to_ome_zarr(
    reader: Any,
    path: Path,
    *,
    tile: CziTile,
    tile_count: int,
    output_dir: Path,
    level: float,
    chunks: tuple[int, int, int, int, int],
    compressor: str,
    dims: dict[str, tuple[int, int]],
    provenance: dict[str, str],
) -> Path:
    import zarr

    t_indexes = _czi_dim_indexes(dims, "T")
    c_indexes = _czi_dim_indexes(dims, "C")
    z_indexes = _czi_dim_indexes(dims, "Z")
    out = _czi_tile_ome_zarr_path(path, tile, tile_count, output_dir=output_dir)
    write_out = _temporary_output_path(out)
    ome_shape = _czi_ome_shape(dims, tile)
    effective_chunks = tuple(min(chunk, size) for chunk, size in zip(chunks, ome_shape, strict=True))

    logger.info(
        f"Writing tile {tile['index'] + 1}/{tile_count} "
        f"at x={_format_signed_coordinate(tile['x'])} y={_format_signed_coordinate(tile['y'])} "
        f"size={tile['width']}x{tile['height']} to {out.name}"
    )

    _remove_output_if_exists(write_out)
    try:
        first_plane = _as_grayscale_plane(
            reader.read_image(**_czi_read_kwargs(dims, tile, t_indexes[0], c_indexes[0], z_indexes[0]))[0]
        )
        raw_bytes = 0
        root = zarr.open_group(str(write_out), mode="w", zarr_format=2)
        root.attrs.update(_ome_zarr_root_attrs(path, reader, tile, ome_shape, provenance))
        array = root.create_array(
            "0",
            shape=ome_shape,
            chunks=effective_chunks,
            dtype=first_plane.dtype,
            compressor=_zarr_numcodecs_compressor(compressor, level),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

        plane_index = 0
        plane_count = _czi_plane_count(dims)
        for t_offset, t in enumerate(t_indexes):
            for c_offset, c in enumerate(c_indexes):
                for z_offset, z in enumerate(z_indexes):
                    if plane_index == 0:
                        plane = first_plane
                    else:
                        plane = _as_grayscale_plane(reader.read_image(**_czi_read_kwargs(dims, tile, t, c, z))[0])
                    raw_bytes += plane.nbytes
                    array[t_offset, c_offset, z_offset, :, :] = plane
                    plane_index += 1
                    if plane_index == 1 or plane_index % CZI_PROGRESS_INTERVAL == 0 or plane_index == plane_count:
                        logger.info(f"{out.name}: wrote plane {plane_index}/{plane_count} (T={t}, C={c}, Z={z})")
        root.attrs["squisher_complete"] = True
        _replace_output(write_out, out)
    except BaseException:
        _remove_output_if_exists(write_out)
        raise

    compressed_bytes = _directory_size(out)
    ratio = raw_bytes / compressed_bytes if compressed_bytes else 0.0
    saved_fraction = 1.0 - (compressed_bytes / raw_bytes) if raw_bytes else 0.0
    logger.success(
        f"Finished {out.name} compressed={_format_bytes(compressed_bytes)} "
        f"raw={_format_bytes(raw_bytes)} ratio={ratio:.2f}:1 saved={saved_fraction:.1%}"
    )
    return out


def _ome_zarr_root_attrs(
    path: Path,
    reader: Any,
    tile: CziTile,
    shape: tuple[int, int, int, int, int],
    provenance: dict[str, str],
) -> dict[str, Any]:
    scale_x = _czi_scale_um(reader.meta, "X") or 1.0
    scale_y = _czi_scale_um(reader.meta, "Y") or 1.0
    scale_z = _czi_scale_um(reader.meta, "Z") or 1.0
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": f"{path.stem}:{tile['index']:03d}",
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, scale_z, scale_y, scale_x]},
                            {
                                "type": "translation",
                                "translation": [
                                    0.0,
                                    0.0,
                                    0.0,
                                    float(tile["position_y"]) * scale_y,
                                    float(tile["position_x"]) * scale_x,
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
        "omero": {"channels": [{"label": f"C{index}"} for index in range(shape[1])]},
        "squisher": {
            "source_file": path.name,
            "source_format": "CZI",
            "output_tile_index": tile["index"],
            "czi_scene": tile["scene"],
            "czi_mosaic_index": tile.get("mosaic_index"),
            "czi_tile_x": tile["x"],
            "czi_tile_y": tile["y"],
            "czi_tile_width": tile["width"],
            "czi_tile_height": tile["height"],
            "provenance": provenance,
        },
    }


def _verify_ome_tiff(path: Path, out: Path, *, tile: CziTile, plane_count: int) -> list[str]:
    errors = []
    try:
        with TiffFile(out) as tif:
            expected_shape = (tile["height"], tile["width"])
            if len(tif.pages) != plane_count:
                errors.append(f"{out.name}: expected {plane_count} pages, found {len(tif.pages)}")
            for page_index, page in enumerate(tif.pages):
                if page.compression != 22610:
                    errors.append(f"{out.name}: page {page_index} expected compression 22610, found {page.compression}")
                if not page.is_tiled:
                    errors.append(f"{out.name}: page {page_index} is not tiled")
                if page.shape != expected_shape:
                    errors.append(f"{out.name}: page {page_index} expected shape {expected_shape}, found {page.shape}")
            errors.extend(_verify_ome_metadata(path, out.name, tif.ome_metadata, tile=tile, plane_count=plane_count))
    except (OSError, ValueError, KeyError, IndexError, tifffile.TiffFileError) as exc:
        errors.append(f"{out.name}: unreadable TIFF/OME metadata: {exc}")
    return errors


def _verify_ome_metadata(
    path: Path,
    output_name: str,
    ome_metadata: str | None,
    *,
    tile: CziTile,
    plane_count: int,
) -> list[str]:
    if not ome_metadata:
        return [f"{output_name}: missing OME metadata"]

    errors = []
    root = ET.fromstring(ome_metadata)
    map_values = _map_annotation_values(root)
    expected_values = {
        "squisher.source_file": path.name,
        "squisher.source_format": "CZI",
        "squisher.output_tile_index": str(tile["index"]),
        "czi.scene": str(tile["scene"]),
        "czi.tile_width": str(tile["width"]),
        "czi.tile_height": str(tile["height"]),
    }
    if "mosaic_index" in tile:
        expected_values["czi.mosaic_index"] = str(tile["mosaic_index"])
    for key, expected in expected_values.items():
        if map_values.get(key) != expected:
            errors.append(f"{output_name}: expected MapAnnotation {key}={expected!r}, found {map_values.get(key)!r}")
    if not map_values.get("czi.global_metadata_xml", "").startswith("<ImageDocument"):
        errors.append(f"{output_name}: missing raw global CZI XML in MapAnnotation")
    if not map_values.get("czi.subblock_metadata_xml", "").startswith("<Subblocks"):
        errors.append(f"{output_name}: missing raw subblock CZI XML in MapAnnotation")

    raw_annotations = [
        annotation
        for annotation in root.findall(".//ome:XMLAnnotation", OME_NS)
        if annotation.attrib.get("Namespace") == CZI_RAW_METADATA_NAMESPACE
    ]
    if len(raw_annotations) != 1:
        errors.append(f"{output_name}: expected one raw CZI XMLAnnotation, found {len(raw_annotations)}")
        return errors
    if root.find(f".//ome:AnnotationRef[@ID='{raw_annotations[0].attrib['ID']}']", OME_NS) is None:
        errors.append(f"{output_name}: raw CZI XMLAnnotation is not linked by AnnotationRef")
    value = raw_annotations[0].find("ome:Value", OME_NS)
    if value is None or len(value) != 1 or value[0].tag != "CZIProvenanceMetadata":
        errors.append(f"{output_name}: malformed raw CZI XMLAnnotation value")
        return errors

    provenance = value[0]
    global_metadata = provenance.find("GlobalMetadata")
    subblock_metadata = provenance.find("SubblockMetadata")
    if global_metadata is None or len(global_metadata) != 1 or global_metadata[0].tag != "ImageDocument":
        errors.append(f"{output_name}: raw provenance is missing CZI ImageDocument")
    if subblock_metadata is None or len(subblock_metadata) != 1 or subblock_metadata[0].tag != "Subblocks":
        errors.append(f"{output_name}: raw provenance is missing CZI Subblocks")
    elif len(subblock_metadata[0]) != plane_count:
        errors.append(f"{output_name}: expected {plane_count} raw subblock records, found {len(subblock_metadata[0])}")
    return errors


def _map_annotation_values(root: ET.Element) -> dict[str, str]:
    return {
        item.attrib["K"]: item.text or ""
        for item in root.findall(".//ome:MapAnnotation/ome:Value/ome:M", OME_NS)
    }


def _decode_sample_pages(
    out: Path,
    *,
    reader: Any,
    dims: dict[str, tuple[int, int]],
    tile: CziTile,
    max_sample_mae: float | None,
    max_sample_max_abs: float | None,
) -> list[str]:
    errors = []
    t_indexes = _czi_dim_indexes(dims, "T")
    c_indexes = _czi_dim_indexes(dims, "C")
    z_indexes = _czi_dim_indexes(dims, "Z")
    plane_count = len(t_indexes) * len(c_indexes) * len(z_indexes)
    with TiffFile(out) as tif:
        for page_index in sorted({0, plane_count // 2, plane_count - 1}):
            try:
                decoded = tif.pages[page_index].asarray()
            except (OSError, ValueError, RuntimeError, tifffile.TiffFileError) as exc:
                errors.append(f"{out.name}: failed to decode page {page_index}: {exc}")
                continue
            t, c, z = _plane_coordinates(t_indexes, c_indexes, z_indexes, page_index)
            raw = _as_grayscale_plane(reader.read_image(**_czi_read_kwargs(dims, tile, t, c, z))[0])
            if decoded.shape != raw.shape:
                errors.append(f"{out.name}: decoded page {page_index} shape {decoded.shape} != CZI {raw.shape}")
                continue
            if decoded.dtype != raw.dtype:
                errors.append(f"{out.name}: decoded page {page_index} dtype {decoded.dtype} != CZI {raw.dtype}")
                continue
            diff = decoded.astype(np.float32) - raw.astype(np.float32)
            abs_diff = np.abs(diff)
            max_abs = float(np.max(abs_diff))
            mae = float(np.mean(abs_diff))
            rmse = float(np.sqrt(np.mean(diff * diff)))
            logger.info(
                "{} page {} source diff: T={} C={} Z={} max_abs={:.3f} mae={:.3f} rmse={:.3f}",
                out.name,
                page_index,
                t,
                c,
                z,
                max_abs,
                mae,
                rmse,
            )
            if max_sample_mae is not None and mae > max_sample_mae:
                errors.append(f"{out.name}: page {page_index} MAE {mae:.3f} > {max_sample_mae:.3f}")
            if max_sample_max_abs is not None and max_abs > max_sample_max_abs:
                errors.append(f"{out.name}: page {page_index} max abs {max_abs:.3f} > {max_sample_max_abs:.3f}")
    return errors


def write_center_z_thumbnail(path: Path, *, max_size: int = 512, overwrite: bool = True) -> Path:
    if max_size <= 0:
        raise ValueError(f"Thumbnail size must be > 0, got {max_size}")
    out = _thumbnail_path(path)
    if out.exists() and not overwrite:
        return out
    if path.name.endswith(".ome.zarr"):
        rgb = _center_zarr_thumbnail_rgb(path, max_size=max_size)
    else:
        rgb = _center_z_thumbnail_rgb(path, max_size=max_size)
    imagecodecs.imwrite(out, rgb)
    logger.info("Wrote thumbnail {}", out.name)
    return out


def _center_zarr_thumbnail_rgb(path: Path, *, max_size: int) -> npt.NDArray[np.uint8]:
    import zarr

    _register_imagecodecs_numcodecs()
    array = zarr.open(str(path / "0"), mode="r")
    if len(array.shape) != 5:
        raise ValueError(f"Expected OME-Zarr TCZYX array, found shape {array.shape}")
    _, channel_count, z_count, height, width = array.shape
    stride = max(1, int(np.ceil(max(height, width) / max_size)))
    channels = [
        _scale_thumbnail_plane(array[0, channel, z_count // 2, ::stride, ::stride])
        for channel in range(min(2, channel_count))
    ]
    if len(channels) == 1:
        return np.repeat(channels[0][..., None], 3, axis=2)
    rgb = np.zeros((*channels[0].shape, 3), dtype=np.uint8)
    rgb[..., 0] = channels[0]
    rgb[..., 2] = channels[0]
    rgb[..., 1] = channels[1]
    return rgb


def _center_z_thumbnail_rgb(path: Path, *, max_size: int) -> npt.NDArray[np.uint8]:
    with TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        if axes == "TCZYX":
            _, channel_count, z_count, height, width = series.shape
            page_indexes = [channel * z_count + z_count // 2 for channel in range(min(2, channel_count))]
        elif axes == "CZYX":
            channel_count, z_count, height, width = series.shape
            page_indexes = [channel * z_count + z_count // 2 for channel in range(min(2, channel_count))]
        elif axes == "CYX":
            channel_count, height, width = series.shape
            page_indexes = list(range(min(2, channel_count)))
        elif axes == "ZYX":
            z_count, height, width = series.shape
            page_indexes = [z_count // 2]
        elif axes == "YX":
            height, width = series.shape
            page_indexes = [0]
        else:
            raise ValueError(f"Expected OME-TIFF axes TCZYX, CZYX, CYX, ZYX, or YX for thumbnail, found {axes}")

        stride = max(1, int(np.ceil(max(height, width) / max_size)))
        channels = [_scale_thumbnail_plane(tif.pages[index].asarray()[::stride, ::stride]) for index in page_indexes]

    if len(channels) == 1:
        return np.repeat(channels[0][..., None], 3, axis=2)
    rgb = np.zeros((*channels[0].shape, 3), dtype=np.uint8)
    rgb[..., 0] = channels[0]
    rgb[..., 2] = channels[0]
    rgb[..., 1] = channels[1]
    return rgb


def _scale_thumbnail_plane(plane: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    data = plane.astype(np.float32, copy=False)
    sample = data[:: max(1, data.shape[0] // 512), :: max(1, data.shape[1] // 512)]
    nonzero = sample[sample > 0]
    ref = nonzero if nonzero.size else sample.reshape(-1)
    if not ref.size:
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(ref, (1, 99.8))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(ref))
        hi = float(np.max(ref))
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip((data - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _validate_thumbnail_size(thumbnail_size: int) -> None:
    if thumbnail_size <= 0:
        raise ValueError(f"Thumbnail size must be > 0, got {thumbnail_size}")


def _write_missing_resume_thumbnails(
    outputs: list[Path],
    pending_tiles: list[CziTile],
    tiles: list[CziTile],
    thumbnail_size: int,
) -> None:
    for tile, out in zip(tiles, outputs, strict=True):
        if tile not in pending_tiles and out.exists():
            write_center_z_thumbnail(out, max_size=thumbnail_size, overwrite=False)


def _write_czi_tile_process(
    path: Path,
    *,
    tile: CziTile,
    tile_count: int,
    output_dir: Path,
    output_format: str,
    level: float,
    tile_size: int,
    zarr_chunks: tuple[int, int, int, int, int],
    zarr_compressor: str,
    maxworkers: int,
    provenance: dict[str, str],
) -> Path:
    from aicspylibczi import CziFile

    reader = CziFile(path)
    return _write_czi_tile(
        reader,
        path,
        tile=tile,
        tile_count=tile_count,
        output_dir=output_dir,
        output_format=output_format,
        level=level,
        tile_size=tile_size,
        zarr_chunks=zarr_chunks,
        zarr_compressor=zarr_compressor,
        maxworkers=maxworkers,
        dims=reader.get_dims_shape()[0],
        provenance=provenance,
    )


def _first_plane_ome_xml(
    path: Path,
    reader: Any,
    tile: CziTile,
    shape: tuple[int, int, int, int, int],
    dtype: np.dtype[Any],
    provenance: dict[str, str],
) -> bytes:
    subblock_metadata = _czi_subblock_metadata(reader, tile)
    return _czi_ome_xml(
        path,
        reader.meta,
        tile=tile,
        shape=shape,
        dtype=dtype,
        subblock_metadata=subblock_metadata,
        provenance=provenance,
    )


def _czi_read_kwargs(dims: dict[str, tuple[int, int]], tile: CziTile, t: int, c: int, z: int) -> dict[str, int]:
    read_kwargs = {}
    if "C" in dims:
        read_kwargs["C"] = c
    if "Z" in dims:
        read_kwargs["Z"] = z
    if "T" in dims:
        read_kwargs["T"] = t
    if "mosaic_index" in tile:
        read_kwargs["M"] = tile["mosaic_index"]
    elif "S" in dims:
        read_kwargs["S"] = tile["scene"]
    return read_kwargs


def _czi_ome_xml(
    path: Path,
    czi_metadata: Any,
    *,
    tile: CziTile,
    shape: tuple[int, int, int, int, int],
    dtype: np.dtype[Any],
    subblock_metadata: Any,
    provenance: dict[str, str],
) -> bytes:
    plane_count = shape[0] * shape[1] * shape[2]
    metadata = _czi_ome_metadata(
        path,
        czi_metadata,
        tile=tile,
        plane_count=plane_count,
        subblock_metadata=subblock_metadata,
        provenance=provenance,
    )
    metadata.pop("axes", None)

    ome = OmeXml()
    ome.addimage(
        dtype=dtype,
        shape=shape,
        storedshape=(plane_count, 1, 1, shape[3], shape[4], 1),
        axes="TCZYX",
        **metadata,
    )
    _add_czi_raw_metadata_annotation(ome, czi_metadata, subblock_metadata)
    return ome.tostring().encode("utf-8")


def _czi_ome_metadata(
    path: Path,
    czi_metadata: Any,
    *,
    tile: CziTile,
    plane_count: int,
    subblock_metadata: Any,
    provenance: dict[str, str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "axes": "TCZYX",
        "Name": f"{path.stem}:{tile['index']:03d}",
        "StructuredAnnotations": {
            "MapAnnotation": {
                "Namespace": OME_ORIGINAL_METADATA_NAMESPACE,
                "Value": _czi_original_metadata(path, czi_metadata, subblock_metadata, tile=tile, provenance=provenance),
            }
        },
        "Plane": {
            "PositionX": [float(tile["position_x"])] * plane_count,
            "PositionXUnit": ["reference frame"] * plane_count,
            "PositionY": [float(tile["position_y"])] * plane_count,
            "PositionYUnit": ["reference frame"] * plane_count,
        },
    }

    scale_x = _czi_scale_um(czi_metadata, "X")
    scale_y = _czi_scale_um(czi_metadata, "Y")
    scale_z = _czi_scale_um(czi_metadata, "Z")
    if scale_x is not None:
        metadata["PhysicalSizeX"] = scale_x
        metadata["PhysicalSizeXUnit"] = "µm"
        metadata["Plane"]["PositionX"] = [tile["position_x"] * scale_x] * plane_count
        metadata["Plane"]["PositionXUnit"] = ["µm"] * plane_count
    if scale_y is not None:
        metadata["PhysicalSizeY"] = scale_y
        metadata["PhysicalSizeYUnit"] = "µm"
        metadata["Plane"]["PositionY"] = [tile["position_y"] * scale_y] * plane_count
        metadata["Plane"]["PositionYUnit"] = ["µm"] * plane_count
    if scale_z is not None:
        metadata["PhysicalSizeZ"] = scale_z
        metadata["PhysicalSizeZUnit"] = "µm"
    return metadata


def _czi_original_metadata(
    path: Path,
    czi_metadata: Any,
    subblock_metadata: Any,
    *,
    tile: CziTile,
    provenance: dict[str, str],
) -> dict[str, str]:
    values = {
        "squisher.source_file": path.name,
        "squisher.source_format": "CZI",
        "squisher.output_tile_index": str(tile["index"]),
        "czi.scene": str(tile["scene"]),
        "czi.tile_x": str(tile["x"]),
        "czi.tile_y": str(tile["y"]),
        "czi.tile_width": str(tile["width"]),
        "czi.tile_height": str(tile["height"]),
        "czi.global_metadata_xml": _metadata_text(czi_metadata),
        "czi.subblock_metadata_xml": _metadata_text(subblock_metadata),
        **provenance,
    }
    if "mosaic_index" in tile:
        values["czi.mosaic_index"] = str(tile["mosaic_index"])
    if "index_zyx" in tile:
        values["squisher.tile_index_zyx"] = json.dumps(tile["index_zyx"])
    placement_path = path.with_name(f"{path.stem}_placement.json")
    if placement_path.exists():
        values["squisher.placement_json"] = placement_path.read_text()
    return values


def _add_czi_raw_metadata_annotation(ome: OmeXml, czi_metadata: Any, subblock_metadata: Any) -> None:
    annotation_id = f"Annotation:{len(ome.annotations)}"
    global_metadata = _xml_metadata_text(czi_metadata, "CZI global metadata")
    subblock_metadata_text = _xml_metadata_text(subblock_metadata, "CZI subblock metadata")
    ome.annotations.append(
        f'<XMLAnnotation ID="{annotation_id}" Namespace="{CZI_RAW_METADATA_NAMESPACE}">'
        "<Value>"
        '<CZIProvenanceMetadata xmlns="">'
        f"<GlobalMetadata>{global_metadata}</GlobalMetadata>"
        f"<SubblockMetadata>{subblock_metadata_text}</SubblockMetadata>"
        "</CZIProvenanceMetadata>"
        "</Value>"
        "</XMLAnnotation>"
    )
    ome.images[-1] = ome.images[-1].replace(
        "</Image>",
        f'<AnnotationRef ID="{annotation_id}"/></Image>',
        1,
    )


def _czi_tiles(path: Path) -> list[CziTile]:
    from aicspylibczi import CziFile

    czi_file = CziFile(path)
    dims = czi_file.get_dims_shape()[0]
    _validate_supported_czi_dims(path, dims)
    scene = _first_czi_dim(dims, "S")
    channel = _first_czi_dim(dims, "C")
    z = _first_czi_dim(dims, "Z")
    t = _first_czi_dim(dims, "T")
    placement_origins = _placement_origins(path)

    try:
        boxes = czi_file.get_all_mosaic_tile_bounding_boxes(C=channel, Z=z, T=t)
    except RuntimeError:
        boxes = {}

    if boxes:
        xs = sorted({int(box.x) for box in boxes.values()})
        ys = sorted({int(box.y) for box in boxes.values()})
        tiles = []
        for index, (info, box) in enumerate(sorted(boxes.items(), key=lambda item: (item[1].y, item[1].x))):
            index_zyx = (0, ys.index(int(box.y)), xs.index(int(box.x)))
            origin = placement_origins.get(index_zyx)
            tiles.append(
                CziTile(
                    index=index,
                    scene=scene,
                    mosaic_index=int(info.dimension_coordinates["M"]),
                    index_zyx=index_zyx,
                    y=int(box.y),
                    x=int(box.x),
                    height=int(box.h),
                    width=int(box.w),
                    position_y=float(origin[1]) if origin is not None else float(int(box.y) - min(ys)),
                    position_x=float(origin[2]) if origin is not None else float(int(box.x) - min(xs)),
                )
            )
        _validate_placement_origins(path, tiles, placement_origins)
        return tiles

    try:
        box = czi_file.get_tile_bounding_box(C=channel, Z=z, T=t)
        x, y, width, height = int(box.x), int(box.y), int(box.w), int(box.h)
    except RuntimeError:
        x = y = 0
        width = int(dims["X"][1] - dims["X"][0])
        height = int(dims["Y"][1] - dims["Y"][0])
    return [
        CziTile(
            index=0,
            scene=scene,
            y=y,
            x=x,
            height=height,
            width=width,
            position_y=0.0,
            position_x=0.0,
        )
    ]


def _validate_supported_czi_dims(path: Path, dims: dict[str, tuple[int, int]]) -> None:
    unsupported = []
    for dim, (start, end) in dims.items():
        size = int(end) - int(start)
        if dim in SUPPORTED_MULTI_DIMS:
            continue
        if dim == "S" and size > 1:
            unsupported.append(f"S={start}:{end} ({size} scenes)")
        elif dim not in SUPPORTED_SINGLETON_DIMS and size > 1:
            unsupported.append(f"{dim}={start}:{end} ({size})")
        elif dim in SUPPORTED_SINGLETON_DIMS and size > 1:
            unsupported.append(f"{dim}={start}:{end} ({size})")
    if unsupported:
        raise ValueError(
            f"Unsupported CZI dimensions for {path}: {', '.join(unsupported)}. "
            "squisher currently supports X/Y image axes, arbitrary C/Z/T, mosaic M tiles, "
            "and only singleton S/R/I/H/V/B dimensions."
        )


def _plane_coordinates(t_indexes: list[int], c_indexes: list[int], z_indexes: list[int], page_index: int) -> tuple[int, int, int]:
    z_count = len(z_indexes)
    c_count = len(c_indexes)
    t_offset, within_t = divmod(page_index, c_count * z_count)
    c_offset, z_offset = divmod(within_t, z_count)
    return t_indexes[t_offset], c_indexes[c_offset], z_indexes[z_offset]


def _validate_placement_origins(
    path: Path,
    tiles: list[CziTile],
    placement_origins: dict[tuple[int, int, int], tuple[float, float, float]],
) -> None:
    if not placement_origins:
        return
    tile_indices = {tile["index_zyx"] for tile in tiles}
    missing = sorted(tile_indices - set(placement_origins))
    extra = sorted(set(placement_origins) - tile_indices)
    if missing or extra:
        raise ValueError(
            f"Placement JSON for {path} does not match CZI tiles; missing origins: {missing}; extra origins: {extra}"
        )


def _placement_origins(path: Path) -> dict[tuple[int, int, int], tuple[float, float, float]]:
    placement_path = path.with_name(f"{path.stem}_placement.json")
    if not placement_path.exists():
        return {}
    payload = json.loads(placement_path.read_text())
    origins = {}
    for record in payload.get("placement", {}).get("origins", []):
        index = tuple(int(value) for value in record["index_zyx"])
        origin = tuple(float(value) for value in record["origin_zyx"])
        if len(index) != 3 or len(origin) != 3:
            raise ValueError(f"Invalid placement origin record in {placement_path}: {record}")
        origins[index] = origin
    return origins


def _as_grayscale_plane(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    data = np.squeeze(data)
    if data.ndim == 3 and data.shape[-1] == 1:
        return data[..., 0]
    if data.ndim == 2:
        return data
    raise ValueError(f"Expected a 2D grayscale CZI plane, found shape {data.shape}")


def _is_complete_ome_tiff(path: Path, *, plane_count: int) -> bool:
    try:
        with TiffFile(path) as tif:
            return len(tif.pages) == plane_count and all(page.compression == 22610 and page.is_tiled for page in tif.pages)
    except (OSError, ValueError, KeyError, IndexError, tifffile.TiffFileError):
        return False


def _is_complete_output(
    path: Path,
    *,
    output_format: str,
    ome_shape: tuple[int, int, int, int, int],
) -> bool:
    if output_format == "ome-tiff":
        return _is_complete_ome_tiff(path, plane_count=ome_shape[0] * ome_shape[1] * ome_shape[2])
    if output_format == "ome-zarr":
        return _is_complete_ome_zarr(path, shape=ome_shape)
    raise ValueError(f"Unsupported output format {output_format!r}")


def _is_complete_ome_zarr(path: Path, *, shape: tuple[int, int, int, int, int]) -> bool:
    try:
        import zarr

        _register_imagecodecs_numcodecs()
        if not path.is_dir():
            return False
        array = zarr.open(str(path / "0"), mode="r")
        attrs = zarr.open_group(str(path), mode="r").attrs
        return tuple(int(value) for value in array.shape) == shape and "multiscales" in attrs and attrs.get("squisher_complete") is True
    except (OSError, ValueError, KeyError, ImportError):
        return False


def _validate_output_format(output_format: str) -> None:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format {output_format!r}; expected one of {sorted(OUTPUT_FORMATS)}")


def _validate_zarr_chunks(
    chunks: tuple[int, int, int, int, int],
    *,
    min_zarr_chunk_pixels: int,
    compressor: str,
) -> None:
    if len(chunks) != 5:
        raise ValueError(f"Expected five OME-Zarr chunk sizes for TCZYX, got {chunks}")
    if any(chunk <= 0 for chunk in chunks):
        raise ValueError(f"OME-Zarr chunk sizes must be positive, got {chunks}")
    if compressor not in ZARR_COMPRESSORS:
        raise ValueError(f"Unsupported OME-Zarr compressor {compressor!r}; expected one of {sorted(ZARR_COMPRESSORS)}")
    if min_zarr_chunk_pixels <= 0:
        raise ValueError(f"Minimum Zarr chunk pixels must be > 0, got {min_zarr_chunk_pixels}")
    if compressor == "jpegxr" and chunks[2] != 1:
        raise ValueError("JPEG-XR OME-Zarr output requires z chunk size 1 because JPEG-XR chunks are 2D images")
    spatial_pixels = chunks[2] * chunks[3] * chunks[4]
    if spatial_pixels < min_zarr_chunk_pixels:
        raise ValueError(
            f"OME-Zarr spatial chunk size z*y*x={spatial_pixels} is smaller than "
            f"--min-zarr-chunk-pixels={min_zarr_chunk_pixels}; increase --zarr-chunk-y/x "
            "or lower the guard explicitly"
        )


def _zarr_numcodecs_compressor(name: str, level: float):
    import imagecodecs.numcodecs as imagecodecs_numcodecs

    _register_imagecodecs_numcodecs()
    if name == "jpegxr":
        return imagecodecs_numcodecs.Jpegxr(level=_compression_level(level), photometric="minisblack")
    if name == "jpegxl":
        normalized_level = _compression_level(level)
        return imagecodecs_numcodecs.Jpegxl(level=int(round(normalized_level * 100)), lossless=normalized_level >= 1.0)
    raise ValueError(f"Unsupported OME-Zarr compressor {name!r}")


def _register_imagecodecs_numcodecs() -> None:
    import imagecodecs.numcodecs as imagecodecs_numcodecs

    imagecodecs_numcodecs.register_codecs(verbose=False)


def _remove_output(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_output_if_exists(path: Path) -> None:
    if path.exists():
        _remove_output(path)


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def _replace_output(source: Path, target: Path) -> None:
    if not target.exists():
        source.replace(target)
        return

    backup = target.with_name(f".{target.name}.bak-{os.getpid()}")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup path {backup}")

    target.replace(backup)
    try:
        source.replace(target)
    except BaseException:
        _remove_output_if_exists(target)
        backup.replace(target)
        raise
    _remove_output(backup)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _compression_level(level: float) -> float:
    return level / 100 if level > 1 else level


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}TiB"


def _format_signed_coordinate(value: int) -> str:
    return f"{value:6d}"


def _compression_provenance(
    path: Path,
    *,
    output_dir: Path,
    output_format: str,
    level: float,
    tile_size: int,
    zarr_chunks: tuple[int, int, int, int, int],
    min_zarr_chunk_pixels: int,
    zarr_compressor: str,
    requested_tiff_maxworkers: int,
    effective_tiff_maxworkers: int,
    requested_tile_workers: int,
    effective_tile_workers: int,
    resume: bool,
    overwrite: bool,
    thumbnails: bool,
    thumbnail_size: int,
    tile_count: int,
    plane_count: int,
) -> dict[str, str]:
    normalized_level = _compression_level(level)
    compression_name = "JPEG-XR" if output_format == "ome-tiff" else zarr_compressor.upper()
    compression_tiff_tag = JPEG_XR_KWARGS["compression"] if output_format == "ome-tiff" else None
    return {
        "squisher.version": _squisher_version(),
        "squisher.created_at_utc": datetime.now(UTC).isoformat(),
        "squisher.source_path": str(path),
        "squisher.output_dir": str(output_dir),
        "squisher.output_format": output_format,
        "squisher.tile_count": str(tile_count),
        "squisher.plane_count": str(plane_count),
        "squisher.compression": compression_name,
        "squisher.compression_tiff_tag": str(compression_tiff_tag or ""),
        "squisher.compression_level_input": str(level),
        "squisher.compression_level_normalized": str(normalized_level),
        "squisher.tiff_tile_size": str(tile_size),
        "squisher.zarr_chunks_tczyx": json.dumps(zarr_chunks),
        "squisher.min_zarr_chunk_pixels": str(min_zarr_chunk_pixels),
        "squisher.zarr_compressor": zarr_compressor,
        "squisher.requested_tiff_maxworkers": str(requested_tiff_maxworkers),
        "squisher.effective_tiff_maxworkers": str(effective_tiff_maxworkers),
        "squisher.requested_czi_tile_workers": str(requested_tile_workers),
        "squisher.effective_czi_tile_workers": str(effective_tile_workers),
        "squisher.resume": json.dumps(resume),
        "squisher.overwrite": json.dumps(overwrite),
        "squisher.thumbnails": json.dumps(thumbnails),
        "squisher.thumbnail_size": str(thumbnail_size),
        "squisher.settings_json": json.dumps(
            {
                "source_path": str(path),
                "output_dir": str(output_dir),
                "output_format": output_format,
                "compression": compression_name,
                "compression_tiff_tag": compression_tiff_tag,
                "level_input": level,
                "level_normalized": normalized_level,
                "tiff_tile_size": tile_size,
                "zarr_chunks_tczyx": zarr_chunks,
                "min_zarr_chunk_pixels": min_zarr_chunk_pixels,
                "zarr_compressor": zarr_compressor,
                "requested_tiff_maxworkers": requested_tiff_maxworkers,
                "effective_tiff_maxworkers": effective_tiff_maxworkers,
                "requested_czi_tile_workers": requested_tile_workers,
                "effective_czi_tile_workers": effective_tile_workers,
                "resume": resume,
                "overwrite": overwrite,
                "thumbnails": thumbnails,
                "thumbnail_size": thumbnail_size,
                "tile_count": tile_count,
                "plane_count": plane_count,
            },
            sort_keys=True,
        ),
    }


def _squisher_version() -> str:
    try:
        return version("squisher")
    except PackageNotFoundError:
        return "unknown"


def _thumbnail_path(path: Path) -> Path:
    if path.name.endswith(".ome.zarr"):
        return path.with_name(path.name.removesuffix(".ome.zarr") + ".center-z.png")
    return path.with_name(path.name.removesuffix(".ome.tif") + ".center-z.png")


def _czi_dim_indexes(dims: dict[str, tuple[int, int]], dim: str) -> list[int]:
    start, end = dims.get(dim, (0, 1))
    return list(range(int(start), int(end)))


def _czi_plane_count(dims: dict[str, tuple[int, int]]) -> int:
    return (
        len(_czi_dim_indexes(dims, "T"))
        * len(_czi_dim_indexes(dims, "C"))
        * len(_czi_dim_indexes(dims, "Z"))
    )


def _first_czi_dim(dims: dict[str, tuple[int, int]], dim: str) -> int:
    return int(dims.get(dim, (0, 1))[0])


def _czi_subblock_metadata(reader: Any, tile: CziTile) -> ET.Element:
    if "mosaic_index" in tile:
        subblocks = reader.read_subblock_metadata(unified_xml=False, M=tile["mosaic_index"])
    else:
        subblocks = reader.read_subblock_metadata(unified_xml=False, S=tile["scene"])

    root = ET.Element("Subblocks")
    for dims, raw_metadata in subblocks:
        subblock = ET.Element("Subblock")
        for dim, number in dims.items():
            subblock.set(dim, str(number))
        if "S" not in dims:
            subblock.set("S", "0")
        if raw_metadata.strip():
            subblock.append(ET.fromstring(raw_metadata))
        else:
            subblock.set("MetadataEmpty", "true")
        root.append(subblock)
    return root


def _metadata_text(metadata: Any) -> str:
    if isinstance(metadata, ET.Element):
        return ET.tostring(metadata, encoding="unicode")
    if isinstance(metadata, str):
        return metadata
    return json.dumps(metadata, sort_keys=True)


def _xml_metadata_text(metadata: Any, name: str) -> str:
    if isinstance(metadata, ET.Element):
        return ET.tostring(metadata, encoding="unicode")
    if isinstance(metadata, str):
        ET.fromstring(metadata)
        return metadata
    raise TypeError(f"Expected XML metadata for {name}, found {type(metadata).__name__}")


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _czi_scale_um(metadata: Any, axis: str) -> float | None:
    if isinstance(metadata, ET.Element):
        root = metadata
    elif isinstance(metadata, str):
        root = ET.fromstring(metadata)
    else:
        raise TypeError(f"Expected XML CZI metadata, found {type(metadata).__name__}")

    for distance in root.iter():
        if _local_xml_name(distance.tag) != "Distance" or distance.attrib.get("Id") != axis:
            continue
        for child in distance.iter():
            if _local_xml_name(child.tag) == "Value" and child.text:
                value = float(child.text)
                return value * 1_000_000 if value > 0 else None
    return None


def _effective_czi_workers(tile_count: int, tile_workers: int, tiff_maxworkers: int) -> tuple[int, int]:
    cpu_count = os.cpu_count() or 1
    if tile_workers < 0:
        raise ValueError(f"CZI tile workers must be >= 0, got {tile_workers}")
    if tiff_maxworkers < 0:
        raise ValueError(f"TIFF maxworkers must be >= 0, got {tiff_maxworkers}")
    if tile_workers == 0 and tiff_maxworkers == 0:
        effective_tile_workers = min(tile_count, max(1, cpu_count // 8))
        return effective_tile_workers, max(1, min(4, cpu_count // effective_tile_workers))
    if tile_workers == 0:
        return max(1, min(tile_count, cpu_count // max(1, tiff_maxworkers))), tiff_maxworkers
    if tiff_maxworkers == 0:
        return min(tile_workers, tile_count), max(1, min(4, cpu_count // tile_workers))
    return min(tile_workers, tile_count), tiff_maxworkers


def _czi_output_dir(path: Path, *, out_dir: Path | None, tile_count: int) -> Path:
    if tile_count <= 1:
        return out_dir or path.parent
    if out_dir is None:
        return path.with_name(path.stem)
    return out_dir if out_dir.name == path.stem else out_dir / path.stem


def _czi_tile_output_path(
    path: Path,
    tile: CziTile,
    tile_count: int,
    *,
    output_dir: Path | None = None,
    output_format: str,
) -> Path:
    if output_format == "ome-tiff":
        return _czi_tile_ome_tiff_path(path, tile, tile_count, output_dir=output_dir)
    if output_format == "ome-zarr":
        return _czi_tile_ome_zarr_path(path, tile, tile_count, output_dir=output_dir)
    raise ValueError(f"Unsupported output format {output_format!r}")


def _czi_tile_ome_tiff_path(path: Path, tile: CziTile, tile_count: int, *, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or _czi_output_dir(path, out_dir=None, tile_count=tile_count)
    if tile_count == 1:
        return output_dir / f"{path.stem}.ome.tif"
    return output_dir / f"{path.stem}.{tile['index']:03d}.ome.tif"


def _czi_tile_ome_zarr_path(path: Path, tile: CziTile, tile_count: int, *, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or _czi_output_dir(path, out_dir=None, tile_count=tile_count)
    if tile_count == 1:
        return output_dir / f"{path.stem}.ome.zarr"
    return output_dir / f"{path.stem}.{tile['index']:03d}.ome.zarr"


def _czi_ome_shape(dims: dict[str, tuple[int, int]], tile: CziTile) -> tuple[int, int, int, int, int]:
    return (
        len(_czi_dim_indexes(dims, "T")),
        len(_czi_dim_indexes(dims, "C")),
        len(_czi_dim_indexes(dims, "Z")),
        tile["height"],
        tile["width"],
    )
