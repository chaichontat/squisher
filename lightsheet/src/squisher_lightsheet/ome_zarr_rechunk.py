from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from squisher.jpegxr_zarr import register_jpegxr_codec
from squisher_lightsheet import ngff


DEFAULT_CHUNK_SHAPE_ZYX = (12, 480, 480)


def rechunk_ome_zarr(
    *,
    source: Path,
    destination: Path,
    start_level: int = 2,
    zstd_level: int = 3,
    chunk_shape_zyx: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE_ZYX,
    workers: int = 8,
    codec_concurrency: int = 4,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rebase an OME-Zarr pyramid suffix into a lossless Zstd Zarr v3 store."""

    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    source = source.resolve()
    destination = destination.resolve()
    _validate_options(
        source=source,
        destination=destination,
        start_level=start_level,
        zstd_level=zstd_level,
        chunk_shape_zyx=chunk_shape_zyx,
        workers=workers,
        codec_concurrency=codec_concurrency,
    )
    register_jpegxr_codec()
    source_root = zarr.open_group(str(source), mode="r")
    source_attrs = source_root.attrs.asdict()
    if source_attrs.get("squisher_complete") is False:
        raise ValueError(f"Source OME-Zarr is explicitly marked incomplete: {source}")

    multiscales = ngff.multiscales(source_root)
    if len(multiscales) != 1:
        raise ValueError(f"Expected exactly one OME-Zarr multiscale in {source}, found {len(multiscales)}")
    source_paths = ngff.dataset_paths(source_root)
    if start_level >= len(source_paths):
        raise ValueError(
            f"Source OME-Zarr has {len(source_paths)} pyramid level(s); cannot start at level {start_level}"
        )
    selected_paths = source_paths[start_level:]
    source_arrays = [source_root[path] for path in selected_paths]
    dimension_names = [
        _dimension_names(source_root, array, source=source, path=path)
        for path, array in zip(selected_paths, source_arrays, strict=True)
    ]
    base_shape = tuple(int(size) for size in source_arrays[0].shape)
    base_dimensions = dimension_names[0]
    for path, array, dimensions in zip(selected_paths, source_arrays, dimension_names, strict=True):
        if dimensions != base_dimensions:
            raise ValueError(
                f"Source level {path} dimensions {dimensions} do not match selected base "
                f"dimensions {base_dimensions}"
            )
        if np.dtype(array.dtype) != np.dtype(source_arrays[0].dtype):
            raise ValueError(
                f"Source level {path} dtype {array.dtype} does not match selected base "
                f"dtype {source_arrays[0].dtype}"
            )

    temporary = destination.with_name(f".{destination.name}.tmp")
    _prepare_output(destination=destination, temporary=temporary, overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)

    level_summaries: list[dict[str, Any]] = []
    try:
        output_root = zarr.open_group(str(temporary), mode="w", zarr_format=3)
        output_attrs = _rebased_root_attrs(
            source_attrs=source_attrs,
            multiscale=multiscales[0],
            start_level=start_level,
        )
        output_attrs["squisher_complete"] = False
        output_attrs["squisher_rechunk"] = {
            "source": str(source),
            "source_start_level": int(start_level),
            "codec": "zstd",
            "zstd_level": int(zstd_level),
        }
        output_root.attrs.update(output_attrs)

        with zarr.config.set({"async.concurrency": codec_concurrency}):
            for output_level, (source_path, source_array, dimensions) in enumerate(
                zip(selected_paths, source_arrays, dimension_names, strict=True)
            ):
                shape = tuple(int(size) for size in source_array.shape)
                chunks = _scaled_chunks(
                    shape=shape,
                    base_shape=base_shape,
                    dimension_names=dimensions,
                    chunk_shape_zyx=chunk_shape_zyx,
                    source_chunks=_storage_chunks(source_array),
                )
                output_path = str(output_level)
                output_array = output_root.create_array(
                    output_path,
                    shape=shape,
                    chunks=chunks,
                    dtype=source_array.dtype,
                    fill_value=source_array.fill_value,
                    attributes=source_array.attrs.asdict(),
                    dimension_names=dimensions,
                    serializer=BytesCodec(),
                    compressors=[ZstdCodec(level=zstd_level)],
                )
                if progress is not None:
                    progress(
                        f"Rechunking source level {source_path} to output level {output_path}: "
                        f"shape={shape}, chunks={chunks}"
                    )
                written_chunks = _copy_chunks(
                    source_array=source_array,
                    output_array=output_array,
                    shape=shape,
                    chunks=chunks,
                    workers=workers,
                )
                level_summaries.append(
                    {
                        "source_path": source_path,
                        "path": output_path,
                        "shape": list(shape),
                        "chunks": list(chunks),
                        "written_chunks": written_chunks,
                    }
                )

        output_root.attrs["squisher_complete"] = True
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "artifact_type": "squisher_lightsheet.ome_zarr_rechunk.v1",
        "source": str(source),
        "destination": str(destination),
        "source_start_level": int(start_level),
        "zstd_level": int(zstd_level),
        "workers": int(workers),
        "codec_concurrency": int(codec_concurrency),
        "levels": level_summaries,
    }


def _validate_options(
    *,
    source: Path,
    destination: Path,
    start_level: int,
    zstd_level: int,
    chunk_shape_zyx: tuple[int, int, int],
    workers: int,
    codec_concurrency: int,
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source OME-Zarr does not exist or is not a directory: {source}")
    if not destination.name.endswith(".zarr"):
        raise ValueError(f"Destination must be a .zarr directory, got {destination}")
    if source == destination:
        raise ValueError("Source and destination OME-Zarr paths must differ")
    if source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination OME-Zarr paths must not contain one another")
    if start_level < 0:
        raise ValueError(f"start_level must be non-negative, got {start_level}")
    if not 1 <= zstd_level <= 22:
        raise ValueError(f"zstd_level must be between 1 and 22, got {zstd_level}")
    if len(chunk_shape_zyx) != 3 or any(size < 1 for size in chunk_shape_zyx):
        raise ValueError(f"chunk_shape_zyx must contain three positive values, got {chunk_shape_zyx}")
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    if codec_concurrency < 1:
        raise ValueError(f"codec_concurrency must be positive, got {codec_concurrency}")


def _prepare_output(*, destination: Path, temporary: Path, overwrite: bool) -> None:
    existing = [path for path in (destination, temporary) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output path(s): {', '.join(str(path) for path in existing)}"
        )
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _dimension_names(root: Any, array: Any, *, source: Path, path: str) -> tuple[str, ...]:
    dimensions = getattr(array.metadata, "dimension_names", None)
    if dimensions is None:
        dimensions = array.attrs.get("_ARRAY_DIMENSIONS")
    if dimensions is None:
        axes = ngff.multiscales(root)[0].get("axes")
        if isinstance(axes, list):
            dimensions = [axis.get("name") if isinstance(axis, Mapping) else axis for axis in axes]
    if (
        dimensions is None
        or len(dimensions) != array.ndim
        or not all(isinstance(dimension, str) and dimension for dimension in dimensions)
    ):
        raise ValueError(f"Cannot determine dimension names for {source}/{path} with shape {array.shape}")
    return tuple(dimensions)


def _storage_chunks(array: Any) -> tuple[int, ...]:
    chunk_grid = getattr(array.metadata, "chunk_grid", None)
    chunk_shape = getattr(chunk_grid, "chunk_shape", None)
    if chunk_shape is None:
        chunk_shape = array.chunks
    return tuple(int(size) for size in chunk_shape)


def _scaled_chunks(
    *,
    shape: tuple[int, ...],
    base_shape: tuple[int, ...],
    dimension_names: tuple[str, ...],
    chunk_shape_zyx: tuple[int, int, int],
    source_chunks: tuple[int, ...],
) -> tuple[int, ...]:
    spatial_chunks = dict(zip(("z", "y", "x"), chunk_shape_zyx, strict=True))
    chunks = []
    for dimension, size, base_size, source_chunk in zip(
        dimension_names, shape, base_shape, source_chunks, strict=True
    ):
        axis = dimension.lower()
        if axis in spatial_chunks:
            scaled = round(spatial_chunks[axis] * size / base_size)
            chunks.append(min(size, max(1, scaled)))
        else:
            chunks.append(min(size, source_chunk))
    return tuple(chunks)


def _copy_chunks(
    *,
    source_array: Any,
    output_array: Any,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    workers: int,
) -> int:
    selections = _chunk_selections(shape=shape, chunks=chunks)

    def copy(selection: tuple[slice, ...]) -> bool:
        data = np.asarray(source_array[selection])
        if _is_fill_chunk(data, source_array.fill_value):
            return False
        output_array[selection] = data
        return True

    if workers == 1:
        return sum(copy(selection) for selection in selections)

    written = 0
    pending: set[Future[bool]] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for selection in selections:
            pending.add(executor.submit(copy, selection))
            if len(pending) < workers * 2:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            written += sum(future.result() for future in done)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            written += sum(future.result() for future in done)
    return written


def _chunk_selections(*, shape: tuple[int, ...], chunks: tuple[int, ...]) -> Iterator[tuple[slice, ...]]:
    counts = tuple(math.ceil(size / chunk) for size, chunk in zip(shape, chunks, strict=True))
    for flat_index in range(math.prod(counts)):
        coordinates = np.unravel_index(flat_index, counts)
        yield tuple(
            slice(coordinate * chunk, min(size, (coordinate + 1) * chunk))
            for coordinate, size, chunk in zip(coordinates, shape, chunks, strict=True)
        )


def _is_fill_chunk(data: np.ndarray, fill_value: Any) -> bool:
    if fill_value is None:
        return False
    try:
        if np.issubdtype(data.dtype, np.floating) and np.isnan(fill_value):
            return bool(np.isnan(data).all())
    except TypeError:
        return False
    return bool(np.equal(data, fill_value).all())


def _rebased_root_attrs(
    *,
    source_attrs: dict[str, Any],
    multiscale: dict[str, Any],
    start_level: int,
) -> dict[str, Any]:
    attrs = deepcopy(source_attrs)
    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list) or start_level >= len(datasets):
        raise ValueError("OME-Zarr multiscales metadata has insufficient datasets")
    rebased_multiscale = deepcopy(multiscale)
    rebased_multiscale["datasets"] = [
        {**deepcopy(dataset), "path": str(output_level)}
        for output_level, dataset in enumerate(datasets[start_level:])
    ]
    ome = attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        ome["multiscales"] = [rebased_multiscale]
    elif "multiscales" in attrs:
        attrs["multiscales"] = [rebased_multiscale]
    else:
        raise ValueError("OME-Zarr root is missing multiscales metadata")
    return attrs


__all__ = ["DEFAULT_CHUNK_SHAPE_ZYX", "rechunk_ome_zarr"]
