from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from queue import Full, Queue
import threading
from typing import Any, Iterable

import numpy as np
from tifffile import OmeXml, TiffWriter

from squisher_deconv.metadata import json_dumps_strict
from squisher_deconv.source import TiffLogicalSource

JPEG_XR_KWARGS = {"photometric": "minisblack", "compression": 22610}
FLOAT32_KWARGS = {"photometric": "minisblack"}
WRITE_QUEUE_DEPTH = 3
_WRITE_SENTINEL = object()


def write_streamed_ome_tiff(
    path: Path,
    *,
    source: TiffLogicalSource,
    core_plane_chunks: Iterable[np.ndarray],
    output_mode: str,
    provenance: dict[str, Any],
    overwrite: bool = False,
    write_queue_depth: int = WRITE_QUEUE_DEPTH,
) -> None:
    if write_queue_depth < 1:
        raise ValueError(f"write_queue_depth must be at least 1, got {write_queue_depth}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.partial")
    if partial_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing partial output {partial_path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output {path}; pass --overwrite to replace it.")
    dtype = np.dtype(np.uint16 if output_mode == "u16" else np.float32)
    total_planes = source.z_count * source.channels
    write_queue: Queue[np.ndarray | object] = Queue(maxsize=write_queue_depth)
    writer_errors: list[BaseException] = []
    plane_index = 0

    def writer_worker() -> None:
        nonlocal plane_index
        with TiffWriter(partial_path, bigtiff=True, mode="x") as writer:
            while True:
                item = write_queue.get()
                try:
                    if item is _WRITE_SENTINEL:
                        return
                    chunk = item
                    if not isinstance(chunk, np.ndarray):
                        raise TypeError(f"Expected queued ndarray chunk, got {type(chunk).__name__}")
                    _write_chunk(
                        writer,
                        chunk,
                        source=source,
                        dtype=dtype,
                        total_planes=total_planes,
                        provenance=provenance,
                        output_mode=output_mode,
                        plane_index_start=plane_index,
                    )
                    plane_index += int(chunk.shape[0])
                finally:
                    write_queue.task_done()

    def guarded_writer_worker() -> None:
        try:
            writer_worker()
        except BaseException as exc:
            writer_errors.append(exc)

    writer_thread = threading.Thread(target=guarded_writer_worker, daemon=True, name="squisher-ome-writer")
    writer_thread.start()
    try:
        for chunk in core_plane_chunks:
            if chunk.ndim != 3:
                raise ValueError(f"Expected flattened core plane chunk, got {chunk.shape}")
            _put_write_item(write_queue, chunk, writer_errors)
        _put_write_item(write_queue, _WRITE_SENTINEL, writer_errors)
        writer_thread.join()
        if writer_errors:
            raise writer_errors[0]
        if plane_index != total_planes:
            raise ValueError(f"Expected to write {total_planes} plane(s), wrote {plane_index}")
        partial_path.replace(path)
    except BaseException:
        if writer_thread.is_alive():
            _put_write_item(write_queue, _WRITE_SENTINEL, writer_errors, raise_writer_errors=False)
            writer_thread.join()
        partial_path.unlink(missing_ok=True)
        raise


def _put_write_item(
    write_queue: Queue[np.ndarray | object],
    item: np.ndarray | object,
    writer_errors: list[BaseException],
    *,
    raise_writer_errors: bool = True,
) -> None:
    while True:
        if writer_errors and raise_writer_errors:
            raise writer_errors[0]
        try:
            write_queue.put(item, timeout=0.1)
            return
        except Full:
            continue


def _write_chunk(
    writer: TiffWriter,
    chunk: np.ndarray,
    *,
    source: TiffLogicalSource,
    dtype: np.dtype[Any],
    total_planes: int,
    provenance: dict[str, Any],
    output_mode: str,
    plane_index_start: int,
) -> None:
    if chunk.ndim != 3:
        raise ValueError(f"Expected flattened core plane chunk, got {chunk.shape}")
    plane_index = plane_index_start
    for plane in chunk.astype(dtype, copy=False):
        write_kwargs = (
            {**JPEG_XR_KWARGS, "compressionargs": {"level": 0.75}} if output_mode == "u16" else FLOAT32_KWARGS
        )
        writer.write(
            plane,
            description=_ome_xml(source, dtype=dtype, total_planes=total_planes, provenance=provenance)
            if plane_index == 0
            else None,
            metadata=None,
            **write_kwargs,
        )
        plane_index += 1


def _ome_xml(
    source: TiffLogicalSource,
    *,
    dtype: np.dtype[Any],
    total_planes: int,
    provenance: dict[str, Any],
) -> str:
    ome = OmeXml()
    metadata = {
        "Name": source.path.stem,
        "StructuredAnnotations": {
            "MapAnnotation": {
                "Namespace": "squisher/deconv/provenance",
                "Value": {
                    "squisher.deconv.provenance": json_dumps_strict(
                        provenance,
                        context="OME deconvolution provenance",
                    ),
                    "squisher.deconv.source_metadata_hash": source.metadata.metadata_hash,
                    "squisher.deconv.source_metadata_summary": json_dumps_strict(
                        _source_metadata_summary(source),
                        context="OME source metadata summary",
                    ),
                },
            }
        },
    }
    ome.addimage(
        dtype=dtype,
        shape=(1, 1, total_planes, source.height, source.width),
        storedshape=(total_planes, 1, 1, source.height, source.width, 1),
        axes="TCZYX",
        **metadata,
    )
    return ome.tostring()


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
