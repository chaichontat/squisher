from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from squisher_lightsheet import pyramid as pyramid_core


def ome_tiff_rechunk_output_path(source: Path, output_dir: Path) -> Path:
    name = source.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.endswith(suffix):
            return output_dir / f"{name.removesuffix(suffix)}.ome.zarr"
    return output_dir / f"{source.stem}.ome.zarr"


def rechunk_ome_tiffs(
    *,
    inputs: list[Path],
    output_dir: Path,
    chunk_shape_zyx: tuple[int, int, int] = (12, 240, 240),
    pyramid_downsample_factors: tuple[int, ...] = (2, 4),
    overwrite: bool = False,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    sources = _ome_tiff_sources(inputs)
    if not sources:
        raise ValueError("No OME-TIFF inputs were provided")
    if any(value < 1 for value in chunk_shape_zyx):
        raise ValueError(f"chunk_shape_zyx must be positive, got {chunk_shape_zyx}")
    if any(value <= 1 for value in pyramid_downsample_factors):
        raise ValueError(f"pyramid_downsample_factors must be > 1, got {pyramid_downsample_factors}")
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(index, source, ome_tiff_rechunk_output_path(source, output_dir)) for index, source in enumerate(sources)]
    if workers == 1:
        outputs = []
        for index, source, output in jobs:
            if progress is not None:
                progress(f"Rechunking OME-TIFF {index + 1}/{len(sources)}: {source.name} -> {output.name}")
            outputs.append(
                rechunk_ome_tiff(
                    source=source,
                    output=output,
                    chunk_shape_zyx=chunk_shape_zyx,
                    pyramid_downsample_factors=pyramid_downsample_factors,
                    overwrite=overwrite,
                )
            )
    else:
        outputs_by_index = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, source, output in jobs:
                if progress is not None:
                    progress(f"Queued OME-TIFF {index + 1}/{len(sources)}: {source.name} -> {output.name}")
                future = executor.submit(
                    rechunk_ome_tiff,
                    source=source,
                    output=output,
                    chunk_shape_zyx=chunk_shape_zyx,
                    pyramid_downsample_factors=pyramid_downsample_factors,
                    overwrite=overwrite,
                )
                futures[future] = (index, source)
            for future in as_completed(futures):
                index, source = futures[future]
                outputs_by_index[index] = future.result()
                if progress is not None:
                    progress(f"Finished OME-TIFF {index + 1}/{len(sources)}: {source.name}")
        outputs = [outputs_by_index[index] for index in range(len(sources))]

    return {
        "artifact_type": "lightsheet.ome_tiff_rechunk.v1",
        "output_dir": str(output_dir),
        "chunk_shape_zyx": [int(value) for value in chunk_shape_zyx],
        "pyramid_downsample_factors": [int(value) for value in pyramid_downsample_factors],
        "workers": int(workers),
        "source_count": len(sources),
        "outputs": outputs,
    }


def rechunk_ome_tiff(
    *,
    source: Path,
    output: Path,
    chunk_shape_zyx: tuple[int, int, int],
    pyramid_downsample_factors: tuple[int, ...] = (2, 4),
    overwrite: bool = False,
) -> dict[str, Any]:
    import tifffile
    import zarr
    from numcodecs import Blosc

    source_levels = _ome_tiff_metadata(source)
    source_axes, source_shape, dtype = source_levels[0]
    axes, shape = _rechunk_output_axes_shape(source=source, axes=source_axes, shape=source_shape)
    chunks = _chunks_for_axes(axes=axes, shape=shape, chunk_shape_zyx=chunk_shape_zyx)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not overwrite:
            return _upgrade_existing_rechunked_output(
                source=source,
                output=output,
                axes=axes,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                source_levels=source_levels if axes == source_axes else [],
                chunk_shape_zyx=chunk_shape_zyx,
                pyramid_downsample_factors=pyramid_downsample_factors,
            )
        shutil.rmtree(output)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)

    store = tifffile.imread(source, aszarr=True, level=0)
    try:
        source_array = zarr.open(store, mode="r")
        if hasattr(source_array, "keys") and "0" in source_array:
            source_array = source_array["0"]
        root = zarr.open_group(str(temporary), mode="w", zarr_format=2)
        root.attrs.update(_ome_zarr_attrs(source=source, axes=axes, pyramid_downsample_factors=pyramid_downsample_factors))
        output_array = root.create_array(
            "0",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        )
        output_array.attrs["_ARRAY_DIMENSIONS"] = [axis.lower() for axis in axes]
        if axes == source_axes:
            pyramid_core.copy_z_slabs(
                source_array=source_array,
                output_array=output_array,
                axes=axes,
                shape=shape,
                chunks=chunks,
            )
        else:
            _copy_flattened_zyx_to_czyx(
                source_array=source_array,
                output_array=output_array,
                shape=shape,
                chunks=chunks,
            )
        pyramid_levels = _write_pyramid_levels(
            source=source,
            root=root,
            scale0=output_array,
            axes=axes,
            dtype=dtype,
            source_levels=source_levels if axes == source_axes else [],
            chunk_shape_zyx=chunk_shape_zyx,
            downsample_factors=pyramid_downsample_factors,
        )
        root.attrs["squisher_complete"] = True
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    return _rechunk_summary(
        source=source,
        output=output,
        axes=axes,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        pyramid_downsample_factors=pyramid_downsample_factors,
        pyramid_levels=pyramid_levels,
        read_strategy="z_slab_full_yx" if axes == source_axes else "flattened_zyx_to_czyx",
    )


def _upgrade_existing_rechunked_output(
    *,
    source: Path,
    output: Path,
    axes: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunks: tuple[int, ...],
    source_levels: list[tuple[str, tuple[int, ...], np.dtype]],
    chunk_shape_zyx: tuple[int, int, int],
    pyramid_downsample_factors: tuple[int, ...],
) -> dict[str, Any]:
    import zarr

    root = zarr.open_group(str(output), mode="a")
    if root.attrs.get("squisher_complete") is not True:
        raise FileExistsError(f"Refusing to reuse incomplete rechunked output {output}")
    if "0" not in root:
        raise ValueError(f"Existing rechunked output {output} is missing level 0")
    scale0 = root["0"]
    if tuple(int(value) for value in scale0.shape) != shape:
        raise ValueError(f"Existing rechunked output {output} level 0 shape {scale0.shape} does not match {shape}")
    if tuple(int(value) for value in scale0.chunks) != chunks:
        raise ValueError(f"Existing rechunked output {output} level 0 chunks {scale0.chunks} do not match {chunks}")
    if np.dtype(scale0.dtype) != np.dtype(dtype):
        raise ValueError(f"Existing rechunked output {output} level 0 dtype {scale0.dtype} does not match {dtype}")
    if scale0.attrs.get("_ARRAY_DIMENSIONS") != [axis.lower() for axis in axes]:
        raise ValueError(f"Existing rechunked output {output} level 0 axes do not match {axes}")

    root.attrs.update(_ome_zarr_attrs(source=source, axes=axes, pyramid_downsample_factors=pyramid_downsample_factors))
    pyramid_levels = _write_pyramid_levels(
        source=source,
        root=root,
        scale0=scale0,
        axes=axes,
        dtype=dtype,
        source_levels=source_levels,
        chunk_shape_zyx=chunk_shape_zyx,
        downsample_factors=pyramid_downsample_factors,
    )
    root.attrs["squisher_complete"] = True
    return _rechunk_summary(
        source=source,
        output=output,
        axes=axes,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        pyramid_downsample_factors=pyramid_downsample_factors,
        pyramid_levels=pyramid_levels,
        read_strategy="existing_level0_upgraded",
    )


def _rechunk_summary(
    *,
    source: Path,
    output: Path,
    axes: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunks: tuple[int, ...],
    pyramid_downsample_factors: tuple[int, ...],
    pyramid_levels: list[dict[str, Any]],
    read_strategy: str,
) -> dict[str, Any]:
    return {
        "source": str(source),
        "output": str(output),
        "axes": axes,
        "shape": [int(value) for value in shape],
        "dtype": str(np.dtype(dtype)),
        "chunks": [int(value) for value in chunks],
        "pyramid_downsample_factors": [int(value) for value in pyramid_downsample_factors],
        "pyramid_levels": pyramid_levels,
        "read_strategy": read_strategy,
    }


def _ome_tiff_sources(inputs: list[Path]) -> list[Path]:
    sources: list[Path] = []
    for path in inputs:
        if path.is_dir():
            sources.extend(sorted(path.glob("*.ome.tif")))
            sources.extend(sorted(path.glob("*.ome.tiff")))
        elif path.is_file():
            sources.append(path)
        else:
            raise FileNotFoundError(f"OME-TIFF input does not exist: {path}")
    return sorted(dict.fromkeys(sources))


def _ome_tiff_metadata(path: Path) -> list[tuple[str, tuple[int, ...], np.dtype]]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = []
        for level in series.levels:
            axes = str(level.axes)
            if axes not in {"ZYX", "CZYX"}:
                raise ValueError(f"Expected OME-TIFF axes ZYX or CZYX for {path}, got {axes!r}")
            levels.append((axes, tuple(int(value) for value in level.shape), np.dtype(level.dtype)))
        return levels


def _rechunk_output_axes_shape(*, source: Path, axes: str, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
    if axes == "ZYX":
        channels = _deconv_channel_count(source)
        if channels == 1:
            return axes, shape
        if shape[0] % channels:
            raise ValueError(f"{source} has {shape[0]} flattened planes, not divisible by channels={channels}")
        return "CZYX", (channels, shape[0] // channels, shape[1], shape[2])
    return axes, shape


def _deconv_channel_count(source: Path) -> int:
    sidecar = source.with_suffix(".deconv.json")
    if not sidecar.exists():
        return 1
    payload = json.loads(sidecar.read_text())
    try:
        channels = int(payload["provenance"]["run_settings"]["channels"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{sidecar} is missing a valid provenance.run_settings.channels value") from error
    if channels < 1:
        raise ValueError(f"{sidecar} records invalid channel count {channels}")
    return channels


def _chunks_for_axes(
    *,
    axes: str,
    shape: tuple[int, ...],
    chunk_shape_zyx: tuple[int, int, int],
) -> tuple[int, ...]:
    if axes == "ZYX":
        return tuple(min(int(chunk), int(size)) for chunk, size in zip(chunk_shape_zyx, shape, strict=True))
    if axes == "CZYX":
        spatial = tuple(min(int(chunk), int(size)) for chunk, size in zip(chunk_shape_zyx, shape[1:], strict=True))
        return (1, *spatial)
    raise ValueError(f"Expected axes ZYX or CZYX, got {axes!r}")


def _copy_flattened_zyx_to_czyx(
    *,
    source_array: Any,
    output_array: Any,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> None:
    channels, z_count, height, width = shape
    if tuple(int(value) for value in output_array.shape) != shape:
        raise ValueError(f"Expected CZYX output shape {shape}, got {output_array.shape}")
    for channel in range(channels):
        for z_slice in pyramid_core.axis_slices(size=z_count, step=chunks[1]):
            source_slice = slice(z_slice.start * channels + channel, z_slice.stop * channels, channels)
            slab = np.asarray(source_array[source_slice, :, :])
            for y_slice, x_slice in pyramid_core.spatial_slices(shape_yx=(height, width), chunks_yx=chunks[2:]):
                output_array[channel, z_slice, y_slice, x_slice] = slab[:, y_slice, x_slice]


def _write_pyramid_levels(
    *,
    source: Path,
    root: Any,
    scale0: Any,
    axes: str,
    dtype: np.dtype,
    source_levels: list[tuple[str, tuple[int, ...], np.dtype]],
    chunk_shape_zyx: tuple[int, int, int],
    downsample_factors: tuple[int, ...],
) -> list[dict[str, Any]]:
    import tifffile
    import zarr
    from numcodecs import Blosc

    levels = []
    for level_index, factor in enumerate(downsample_factors, start=1):
        path = str(level_index)
        shape = pyramid_core.xy_downsampled_shape(
            axes=axes,
            shape=tuple(int(value) for value in scale0.shape),
            factor=factor,
        )
        chunks = _chunks_for_axes(axes=axes, shape=shape, chunk_shape_zyx=chunk_shape_zyx)
        if path in root:
            output_array = root[path]
            if tuple(int(value) for value in output_array.shape) != shape:
                raise ValueError(f"Existing pyramid level {path} shape {output_array.shape} does not match {shape}")
            if tuple(int(value) for value in output_array.chunks) != chunks:
                raise ValueError(f"Existing pyramid level {path} chunks {output_array.chunks} do not match {chunks}")
            if np.dtype(output_array.dtype) != np.dtype(dtype):
                raise ValueError(f"Existing pyramid level {path} dtype {output_array.dtype} does not match {dtype}")
            output_array.attrs["_ARRAY_DIMENSIONS"] = [axis.lower() for axis in axes]
            source_label = "existing"
        else:
            output_array = root.create_array(
                path,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
            )
            output_array.attrs["_ARRAY_DIMENSIONS"] = [axis.lower() for axis in axes]

            source_level = _matching_source_level(source_levels, axes=axes, shape=shape, dtype=dtype)
            if source_level is None:
                pyramid_core.copy_xy_downsampled(
                    source_array=scale0,
                    output_array=output_array,
                    axes=axes,
                    factor=factor,
                    chunks=chunks,
                )
                source_label = "computed_from_level0"
            else:
                store = tifffile.imread(source, aszarr=True, level=source_level)
                try:
                    source_array = zarr.open(store, mode="r")
                    if hasattr(source_array, "keys") and "0" in source_array:
                        source_array = source_array["0"]
                    pyramid_core.copy_z_slabs(
                        source_array=source_array,
                        output_array=output_array,
                        axes=axes,
                        shape=shape,
                        chunks=chunks,
                    )
                finally:
                    close = getattr(store, "close", None)
                    if callable(close):
                        close()
                source_label = f"source_level_{source_level}"

        levels.append(
            {
                "path": path,
                "downsample_factor_yx": int(factor),
                "source": source_label,
                "shape": [int(value) for value in shape],
                "chunks": [int(value) for value in chunks],
            }
        )
    return levels


def _matching_source_level(
    source_levels: list[tuple[str, tuple[int, ...], np.dtype]],
    *,
    axes: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> int | None:
    for index, (level_axes, level_shape, level_dtype) in enumerate(source_levels[1:], start=1):
        if level_axes == axes and level_shape == shape and np.dtype(level_dtype) == np.dtype(dtype):
            return index
    return None


def _ome_zarr_attrs(*, source: Path, axes: str, pyramid_downsample_factors: tuple[int, ...]) -> dict[str, Any]:
    return {
        "multiscales": [
            {
                "version": "0.4",
                "datasets": [
                    {
                        "path": str(index),
                        "coordinateTransformations": [_scale_transform(axes=axes, factor=factor)],
                    }
                    for index, factor in enumerate((1, *pyramid_downsample_factors))
                ],
                "axes": [_ome_zarr_axis(axis) for axis in axes],
                "name": source.name,
            }
        ],
        "squisher_source": str(source),
        "squisher_rechunked_from": "ome-tiff",
    }


def _ome_zarr_axis(axis: str) -> dict[str, str]:
    if axis == "C":
        return {"name": "c", "type": "channel"}
    return {"name": axis.lower(), "type": "space"}


def _scale_transform(*, axes: str, factor: int) -> dict[str, Any]:
    scale = []
    for axis in axes:
        scale.append(float(factor) if axis in {"Y", "X"} else 1.0)
    return {"type": "scale", "scale": scale}
