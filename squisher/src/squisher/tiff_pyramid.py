#!/usr/bin/env python
"""Write a new OME-TIFF with two XY pyramid levels stored as TIFF SubIFDs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import tifffile


TIFF_SUFFIXES = (".tif", ".tiff")
DEFAULT_GPU_BATCH_SIZE = 32
JPEGXR_PYRAMID_LEVEL = 0.65
SUBIFD_LEVEL_COUNT = 2


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def output_filename(source: Path) -> str:
    if source.name.endswith((".ome.tiff", ".ome.tif")):
        return source.name
    return source.stem + ".ome.tif"


def default_output_path(source: Path) -> Path:
    suffixes = "".join(source.suffixes)
    if suffixes.endswith(".ome.tiff"):
        root_name = source.name.removesuffix(".ome.tiff")
        return source.with_name(root_name + ".pyramid") / output_filename(source)
    if suffixes.endswith(".ome.tif"):
        root_name = source.name.removesuffix(".ome.tif")
        return source.with_name(root_name + ".pyramid") / output_filename(source)
    return source.with_name(source.stem + ".pyramid") / output_filename(source)


def directory_tiff_sources(folder: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in folder.iterdir()
        if path.is_file() and path.name.lower().endswith(TIFF_SUFFIXES)
    )


def pyramid_jobs(
    input_path: Path,
    output: Path | None,
    output_dir: Path | None,
    *,
    overwrite: bool,
) -> list[tuple[Path, Path]]:
    if output is not None and output_dir is not None:
        raise ValueError("Use either --output or --output-dir, not both")

    source = input_path.resolve()
    if source.is_dir():
        if output is not None:
            raise ValueError("--output can only be used with a single input file")
        sources = directory_tiff_sources(source)
        if not sources:
            log(f"Skipping {source}: no TIFF files found")
            return []
        if overwrite and output_dir is None:
            return [(item, item) for item in sources]
        if output_dir is not None:
            destination_dir = output_dir.resolve()
            return [(item, destination_dir / output_filename(item)) for item in sources]
        destination_dir = source.with_name(source.name + ".pyramid").resolve()
        return [(item, destination_dir / output_filename(item)) for item in sources]

    if output_dir is not None:
        return [(source, (output_dir / output_filename(source)).resolve())]
    if overwrite and output is None:
        return [(source, source)]
    destination = output.resolve() if output is not None else default_output_path(source).resolve()
    return [(source, destination)]


def expand_pyramid_jobs(
    input_paths: list[Path],
    output: Path | None,
    output_dir: Path | None,
    *,
    overwrite: bool,
) -> list[tuple[Path, Path]]:
    if output is not None and len(input_paths) > 1:
        raise ValueError("--output can only be used with a single input path")

    jobs = []
    outputs = set()
    for input_path in input_paths:
        for source, output_path in pyramid_jobs(input_path, output, output_dir, overwrite=overwrite):
            if output_path in outputs:
                raise ValueError("Multiple inputs resolve to the same output path")
            jobs.append((source, output_path))
            outputs.add(output_path)
    return jobs


def page_template(page: Any) -> tifffile.TiffPage:
    return getattr(page, "keyframe", page)


def page_yx_axes(data: np.ndarray, page: Any) -> tuple[int, int]:
    template = page_template(page)
    if data.ndim < 2:
        raise ValueError(f"Expected an image plane with at least 2 dimensions; got shape {data.shape}")
    if data.ndim == 2:
        return 0, 1
    if int(template.samplesperpixel) > 1 and int(template.planarconfig) == 1:
        return data.ndim - 3, data.ndim - 2
    return data.ndim - 2, data.ndim - 1


@lru_cache(maxsize=1)
def gpu_block_reduce_modules() -> tuple[Any, Any] | None:
    try:
        import cupy as cp
        from cucim.skimage.measure import block_reduce
    except ImportError:
        return None

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
    except cp.cuda.runtime.CUDARuntimeError:
        return None

    return cp, block_reduce


def restore_reduced_dtype(reduced: Any, dtype: np.dtype) -> Any:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        reduced = np.rint(reduced)
        reduced = np.clip(reduced, info.min, info.max)
    return reduced.astype(dtype, copy=False)


def block_mean_xy_numpy(moved: np.ndarray, *, factor: int, dtype: np.dtype) -> np.ndarray:
    y_size, x_size = moved.shape[-2:]
    y_starts = np.arange(0, y_size, factor)
    x_starts = np.arange(0, x_size, factor)
    y_counts = np.diff(np.append(y_starts, y_size)).astype(np.float32)
    x_counts = np.diff(np.append(x_starts, x_size)).astype(np.float32)

    working = moved.astype(np.float32, copy=False)
    y_sums = np.add.reduceat(working, y_starts, axis=-2)
    xy_sums = np.add.reduceat(y_sums, x_starts, axis=-1)
    reduced = xy_sums / y_counts[..., None] / x_counts
    return restore_reduced_dtype(reduced, dtype)


def block_mean_xy_gpu(moved: np.ndarray, *, factor: int, dtype: np.dtype) -> np.ndarray | None:
    modules = gpu_block_reduce_modules()
    if modules is None:
        return None

    cp, block_reduce = modules
    block_size = (1,) * (moved.ndim - 2) + (factor, factor)
    reduced = block_reduce(cp.asarray(moved, dtype=cp.float32), block_size=block_size, func=cp.mean)
    reduced = restore_reduced_dtype(reduced, dtype)
    return cp.asnumpy(reduced)


def block_mean_xy(data: np.ndarray, *, factor: int, page: Any | None = None) -> np.ndarray:
    y_axis, x_axis = page_yx_axes(data, page)
    moved = np.moveaxis(data, (y_axis, x_axis), (-2, -1))
    dtype = np.dtype(data.dtype)
    reduced = block_mean_xy_gpu(moved, factor=factor, dtype=dtype)
    if reduced is None:
        reduced = block_mean_xy_numpy(moved, factor=factor, dtype=dtype)
    return np.moveaxis(reduced, (-2, -1), (y_axis, x_axis))


def pyramid_reductions(
    data: np.ndarray,
    *,
    factor: int,
    level_count: int,
    page: Any | None = None,
) -> list[np.ndarray]:
    levels: list[np.ndarray] = []
    reduced = data
    for _level_index in range(level_count):
        reduced = block_mean_xy(reduced, factor=factor, page=page)
        levels.append(reduced)
    return levels


def page_resolution(page: Any) -> tuple[Any, Any] | None:
    template = page_template(page)
    try:
        return template.tags["XResolution"].value, template.tags["YResolution"].value
    except KeyError:
        return None


def scaled_resolution(resolution: tuple[Any, Any] | None, scale: int) -> tuple[Any, Any] | None:
    if resolution is None:
        return None

    def scale_axis(value: Any) -> Any:
        if isinstance(value, tuple) and len(value) == 2:
            numerator, denominator = value
            return numerator, denominator * scale
        return value / scale

    return scale_axis(resolution[0]), scale_axis(resolution[1])


def resolution_kwargs_from_tags(tags: Mapping[str, Any]) -> dict[str, Any]:
    x_resolution = tags.get("XResolution")
    y_resolution = tags.get("YResolution")
    if x_resolution is None or y_resolution is None:
        return {}
    kwargs: dict[str, Any] = {"resolution": (_rational_pair(x_resolution), _rational_pair(y_resolution))}
    if (resolution_unit := tags.get("ResolutionUnit")) is not None:
        kwargs["resolutionunit"] = resolution_unit
    return kwargs


def _rational_pair(value: Any) -> tuple[Any, Any]:
    if isinstance(value, list | tuple) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(f"Expected TIFF resolution tag as a two-value rational, got {value!r}")


def description_value(description: str | None) -> bytes | None:
    if description is None:
        return None
    return description.encode("utf-8")


def page_write_kwargs(page: Any, *, description: str | None, subifds: int | None) -> dict[str, Any]:
    template = page_template(page)
    kwargs: dict[str, Any] = {
        "photometric": template.photometric,
        "compression": template.compression,
        "predictor": template.predictor,
        "description": description_value(description),
        "metadata": None,
        "subifds": subifds,
    }
    if template.samplesperpixel > 1:
        kwargs["planarconfig"] = template.planarconfig
    if template.is_tiled:
        kwargs["tile"] = (int(template.tilelength), int(template.tilewidth))
    elif template.rowsperstrip is not None:
        kwargs["rowsperstrip"] = int(template.rowsperstrip)

    resolution = page_resolution(page)
    if resolution is not None:
        kwargs["resolution"] = resolution
        kwargs["resolutionunit"] = template.resolutionunit

    if "ExtraSamples" in template.tags:
        kwargs["extrasamples"] = tuple(template.tags["ExtraSamples"].value)
    if template.colormap is not None:
        kwargs["colormap"] = template.colormap

    return kwargs


def subifd_write_kwargs(base_kwargs: dict[str, Any], *, scale: int) -> dict[str, Any]:
    kwargs = dict(base_kwargs)
    kwargs["description"] = None
    kwargs["subifds"] = None
    kwargs["subfiletype"] = 1
    if kwargs.get("compression") == tifffile.COMPRESSION.JPEGXR_NDPI:
        kwargs["compressionargs"] = {"level": JPEGXR_PYRAMID_LEVEL}
    if "resolution" in kwargs:
        kwargs["resolution"] = scaled_resolution(kwargs["resolution"], scale)
    return kwargs


def write_subifd_pyramid(
    writer: tifffile.TiffWriter,
    plane: np.ndarray,
    *,
    base_kwargs: dict[str, Any],
    page: Any | None = None,
    factor: int = 2,
    levels: int = SUBIFD_LEVEL_COUNT,
    reduced_levels: list[np.ndarray] | None = None,
    maxworkers: int | None = None,
) -> None:
    if reduced_levels is None:
        reduced_levels = pyramid_reductions(plane, factor=factor, level_count=levels, page=page)
    elif len(reduced_levels) != levels:
        raise ValueError(f"Expected {levels} SubIFD levels, got {len(reduced_levels)}")

    for level_index, reduced in enumerate(reduced_levels):
        kwargs = subifd_write_kwargs(base_kwargs, scale=factor ** (level_index + 1))
        if maxworkers is None:
            writer.write(reduced, **kwargs)
        else:
            writer.write(reduced, maxworkers=maxworkers, **kwargs)


def series_has_subifds(series: tifffile.TiffPageSeries) -> bool:
    return any(getattr(page, "subifds", None) for page in series.pages)


def validate_source(tif: tifffile.TiffFile, source: Path) -> tuple[tifffile.TiffPageSeries, str]:
    if tif.ome_metadata is None:
        raise ValueError(f"{source} is missing OME metadata")
    if len(tif.series) != 1:
        raise ValueError(f"Expected exactly one OME image series in {source}; found {len(tif.series)}")
    series = tif.series[0]
    if "Y" not in series.axes or "X" not in series.axes:
        raise ValueError(f"Expected OME-TIFF series with X/Y axes; got axes {series.axes!r}")
    return series, tif.ome_metadata


def raw_segment_iterator(page: Any):
    filehandle = page.parent.filehandle
    for offset, bytecount in zip(page.dataoffsets, page.databytecounts):
        filehandle.seek(offset)
        yield filehandle.read(bytecount)


def terminate_process_pool(executor: ProcessPoolExecutor) -> None:
    processes = getattr(executor, "_processes", None)
    if processes:
        for process in list(processes.values()):
            if process.is_alive():
                process.terminate()
        for process in list(processes.values()):
            process.join(timeout=1)
        for process in list(processes.values()):
            if process.is_alive():
                process.kill()
    executor.shutdown(wait=False, cancel_futures=True)


def cleanup_temp_outputs(jobs: list[tuple[Path, Path]], temp_token: str) -> None:
    prefixes = {
        f".{output.resolve().name}.{temp_token}."
        for source, output in jobs
        if source.resolve() == output.resolve()
    }
    if not prefixes:
        return

    parents = {output.resolve().parent for source, output in jobs if source.resolve() == output.resolve()}
    for parent in parents:
        for path in parent.iterdir():
            if (
                path.is_file()
                and path.name.endswith(".tmp")
                and any(path.name.startswith(prefix) for prefix in prefixes)
            ):
                path.unlink(missing_ok=True)


def write_pyramid(
    source: Path,
    output: Path,
    *,
    factor: int,
    overwrite: bool,
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    tiff_maxworkers: int | None = None,
    temp_token: str | None = None,
) -> None:
    source = source.resolve()
    output = output.resolve()
    replacing_source = source == output
    if replacing_source and not overwrite:
        raise ValueError("Input and output paths must differ unless --overwrite is set")
    if factor < 2:
        raise ValueError("--factor must be at least 2")
    if gpu_batch_size < 1:
        raise ValueError("--gpu-batch-size must be at least 1")
    if tiff_maxworkers is not None and tiff_maxworkers < 1:
        raise ValueError("--tiff-maxworkers must be at least 1")
    if output.exists() and not overwrite:
        with tifffile.TiffFile(output) as output_tif:
            if len(output_tif.series) == 1 and series_has_subifds(output_tif.series[0]):
                log(f"Skipping {source}: {output} already contains SubIFD pyramid levels")
                return
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    if replacing_source:
        prefix = f".{output.name}.{temp_token}." if temp_token is not None else f".{output.name}."
        handle = tempfile.NamedTemporaryFile(
            prefix=prefix,
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        )
        handle.close()
        write_output = Path(handle.name)
    else:
        write_output = output

    try:
        with tifffile.TiffFile(source) as tif:
            series, ome_xml = validate_source(tif, source)
            if series_has_subifds(series):
                log(f"Skipping {source}: source already contains SubIFD pyramid levels")
                return
            level_count = SUBIFD_LEVEL_COUNT

            log(
                f"Writing {output}: planes={len(series.pages)}, shape={series.shape}, axes={series.axes}, "
                f"levels={level_count}, compression={series.pages[0].compression.name}, "
                f"gpu_batch_size={gpu_batch_size}, tiff_maxworkers={tiff_maxworkers}"
            )
            with tifffile.TiffWriter(write_output, bigtiff=True) as writer:
                batch_size = gpu_batch_size
                for batch_start in range(0, len(series.pages), batch_size):
                    batch_pages = list(series.pages[batch_start : batch_start + batch_size])
                    if len(batch_pages) == 1:
                        fullres_batch = batch_pages[0].asarray()[np.newaxis, ...]
                    else:
                        fullres_batch = np.stack([page.asarray() for page in batch_pages], axis=0)
                    reduced_batches = pyramid_reductions(
                        fullres_batch,
                        factor=factor,
                        level_count=level_count,
                        page=batch_pages[0],
                    )
                    for batch_offset, page in enumerate(batch_pages):
                        plane_index = batch_start + batch_offset
                        plane = fullres_batch[batch_offset]
                        reduced_levels = [level[batch_offset] for level in reduced_batches]
                        base_kwargs = page_write_kwargs(
                            page,
                            description=ome_xml if plane_index == 0 else None,
                            subifds=level_count,
                        )
                        writer.write(
                            raw_segment_iterator(page),
                            shape=page.shape,
                            dtype=page.dtype,
                            **base_kwargs,
                        )

                        write_subifd_pyramid(
                            writer,
                            plane,
                            base_kwargs=base_kwargs,
                            page=page,
                            factor=factor,
                            levels=level_count,
                            reduced_levels=reduced_levels,
                            maxworkers=tiff_maxworkers,
                        )

                        if (
                            plane_index < 3
                            or (plane_index + 1) % 100 == 0
                            or plane_index + 1 == len(series.pages)
                        ):
                            log(f"Copied plane {plane_index + 1}/{len(series.pages)}")

        if replacing_source:
            os.replace(write_output, output)
    finally:
        if replacing_source:
            write_output.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Source OME-TIFF files or folders.")
    parser.add_argument("-o", "--output", type=Path, help="Destination OME-TIFF file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination folder. Defaults to a new sibling *.pyramid folder.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=2,
        help="XY downsampling factor between the two fixed SubIFD levels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs. With no --output/--output-dir, rewrite input TIFFs via temporary files.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=DEFAULT_GPU_BATCH_SIZE,
        help="Number of planes to downsample per GPU batch.",
    )
    parser.add_argument(
        "--tiff-maxworkers",
        type=int,
        help="Maximum worker threads tifffile may use for compression.",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=8,
        help="Number of input TIFF files to process concurrently for folder inputs.",
    )
    return parser.parse_args()


def run_pyramid_jobs(
    inputs: list[Path],
    *,
    output: Path | None,
    output_dir: Path | None,
    factor: int,
    overwrite: bool,
    gpu_batch_size: int,
    tiff_maxworkers: int | None,
    file_workers: int,
) -> int:
    if file_workers < 1:
        raise ValueError("--file-workers must be at least 1")

    jobs = expand_pyramid_jobs(inputs, output, output_dir, overwrite=overwrite)
    if not jobs:
        log("No TIFF files to process")
        return 0

    temp_token = uuid.uuid4().hex
    if len(jobs) > 1 and file_workers > 1:
        log(f"Processing {len(jobs)} files with {file_workers} file workers")
        executor = ProcessPoolExecutor(max_workers=file_workers)
        futures = []
        try:
            for source, output in jobs:
                futures.append(
                    executor.submit(
                        write_pyramid,
                        source,
                        output,
                        factor=factor,
                        overwrite=overwrite,
                        gpu_batch_size=gpu_batch_size,
                        tiff_maxworkers=tiff_maxworkers,
                        temp_token=temp_token,
                    )
                )
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            log("Interrupted; terminating file workers")
            for future in futures:
                future.cancel()
            terminate_process_pool(executor)
            cleanup_temp_outputs(jobs, temp_token)
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    else:
        for source, output in jobs:
            write_pyramid(
                source,
                output,
                factor=factor,
                overwrite=overwrite,
                gpu_batch_size=gpu_batch_size,
                tiff_maxworkers=tiff_maxworkers,
                temp_token=temp_token,
            )
    log("Done")
    return 0


def main() -> int:
    args = parse_args()
    return run_pyramid_jobs(
        args.inputs,
        output=args.output,
        output_dir=args.output_dir,
        factor=args.factor,
        overwrite=args.overwrite,
        gpu_batch_size=args.gpu_batch_size,
        tiff_maxworkers=args.tiff_maxworkers,
        file_workers=args.file_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
