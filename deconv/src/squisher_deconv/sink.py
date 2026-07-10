from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
import zarr
from numcodecs import Blosc

from squisher_deconv.metadata import json_dumps_strict
from squisher_deconv.source import TiffLogicalSource


def write_streamed_ome_zarr(
    path: Path,
    *,
    source: TiffLogicalSource,
    core_plane_chunks: Iterable[np.ndarray],
    output_mode: str,
    provenance: dict[str, Any],
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.partial")
    if partial_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing partial output {partial_path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output {path}; pass --overwrite to replace it.")

    dtype = np.dtype(np.uint16 if output_mode == "u16" else np.float32)
    json_dumps_strict(provenance, context="OME-Zarr deconvolution provenance")
    try:
        root = zarr.open_group(str(partial_path), mode="w", zarr_format=2)
        root.attrs.update(_ome_zarr_attrs(source=source, provenance=provenance, output_mode=output_mode))
        output = root.create_array(
            "0",
            shape=(source.channels, source.z_count, source.height, source.width),
            chunks=(1, min(source.z_count, 16), min(source.height, 512), min(source.width, 512)),
            dtype=dtype,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        )
        output.attrs["_ARRAY_DIMENSIONS"] = ["c", "z", "y", "x"]

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
            czyx = chunk.astype(dtype, copy=False).reshape(z_count, source.channels, source.height, source.width)
            output[:, z_offset:z_stop, :, :] = np.moveaxis(czyx, 1, 0)
            z_offset = z_stop
        if z_offset != source.z_count:
            raise ValueError(f"Expected to write {source.z_count} z plane(s), wrote {z_offset}")
        root.attrs["squisher_complete"] = True
        if path.exists():
            shutil.rmtree(path)
        partial_path.rename(path)
    except BaseException:
        shutil.rmtree(partial_path, ignore_errors=True)
        raise


def _ome_zarr_attrs(
    *,
    source: TiffLogicalSource,
    provenance: dict[str, Any],
    output_mode: str,
) -> dict[str, Any]:
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": source.path.stem,
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]}],
                    }
                ],
            }
        ],
        "squisher_deconv": {
            "output_mode": output_mode,
            "provenance": provenance,
            "source_metadata_hash": source.metadata.metadata_hash,
            "source_metadata_summary": _source_metadata_summary(source),
        },
    }


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
