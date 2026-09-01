from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger
from skimage.measure import regionprops_table
from squisher.jpegxr_zarr import register_jpegxr_codec


MOMENT_SCHEMA = {
    "label": pl.UInt32,
    "area": pl.UInt64,
    "z_sum": pl.UInt64,
    "y_sum": pl.UInt64,
    "x_sum": pl.UInt64,
}
PLANE_SCHEMA = {
    "label": pl.UInt32,
    "plane_z": pl.UInt32,
    "plane_area": pl.UInt64,
}
CELL_COLUMNS = (
    "label",
    "area",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "plane_z",
    "plane_area",
)
INTENSITY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
ChunkBounds = tuple[int, int, int, int, int, int]


def _validate_intensity_names(names: Iterable[str]) -> tuple[str, ...]:
    validated = tuple(names)
    if len(set(validated)) != len(validated):
        raise ValueError(f"Intensity names must be unique, got {validated}.")
    invalid = [name for name in validated if not INTENSITY_NAME_PATTERN.fullmatch(name)]
    if invalid:
        raise ValueError(f"Intensity names may contain only letters, digits, and underscores, got {invalid}.")
    return validated


def _intensity_moment_schema(names: Iterable[str]) -> dict[str, type[pl.DataType]]:
    return {
        column: dtype
        for name in names
        for column, dtype in (
            (f"intensity_{name}_sum", pl.Float64),
            (f"intensity_{name}_min", pl.Float64),
            (f"intensity_{name}_max", pl.Float64),
            (f"intensity_{name}_has_nan", pl.Boolean),
        )
    }


def _intensity_output_columns(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        column
        for name in names
        for column in (
            f"intensity_{name}_min",
            f"intensity_{name}_mean",
            f"intensity_{name}_max",
        )
    )


def _intensity_names_from_columns(columns: Iterable[str]) -> tuple[str, ...]:
    prefix = "intensity_"
    suffix = "_sum"
    return tuple(
        column[len(prefix) : -len(suffix)]
        for column in columns
        if column.startswith(prefix) and column.endswith(suffix)
    )


def _validate_block(block: np.ndarray, offset_zyx: tuple[int, int, int]) -> None:
    if block.ndim != 3:
        raise ValueError(f"Expected a 3D label block, got shape={block.shape}.")
    if not np.issubdtype(block.dtype, np.integer):
        raise TypeError(f"Expected an integer label block, got dtype={block.dtype}.")
    if len(offset_zyx) != 3 or any(offset < 0 for offset in offset_zyx):
        raise ValueError(f"offset_zyx must contain three nonnegative integers, got {offset_zyx}.")
    if np.issubdtype(block.dtype, np.signedinteger) and int(block.min(initial=0)) < 0:
        raise ValueError("Label values must be nonnegative.")
    if int(block.max(initial=0)) > np.iinfo(np.uint32).max:
        raise ValueError("Label values must fit in uint32.")
    if block.shape[0] and offset_zyx[0] + block.shape[0] - 1 > np.iinfo(np.uint32).max:
        raise ValueError("Global Z coordinates must fit in uint32.")


def measure_block(
    block: np.ndarray,
    *,
    offset_zyx: tuple[int, int, int],
    intensity_image: np.ndarray | None = None,
    intensity_names: tuple[str, ...] = (),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Measure additive label moments and per-Z areas for one decoded ZYX block.

    Blocks must be disjoint. Label zero is background, and labels must already
    be globally consistent across chunks.
    """
    block = np.asarray(block)
    _validate_block(block, offset_zyx)
    intensity_names = _validate_intensity_names(intensity_names)
    properties = ["label", "area", "centroid"]
    if intensity_image is not None:
        intensity_image = np.asarray(intensity_image)
        if intensity_image.ndim == block.ndim:
            intensity_image = intensity_image[..., np.newaxis]
        if intensity_image.shape[:-1] != block.shape:
            raise ValueError(
                "Intensity and label blocks must have identical spatial shapes, got "
                f"intensity={intensity_image.shape}, labels={block.shape}."
            )
        if intensity_image.shape[-1] != len(intensity_names):
            raise ValueError(
                f"Intensity block has {intensity_image.shape[-1]} channels but "
                f"{len(intensity_names)} names were supplied."
            )
        if not np.issubdtype(intensity_image.dtype, np.number):
            raise TypeError(f"Expected a numeric intensity block, got dtype={intensity_image.dtype}.")
        properties.extend(("intensity_min", "intensity_mean", "intensity_max"))
    elif intensity_names:
        raise ValueError("Intensity names were supplied without an intensity image.")
    props = regionprops_table(
        block,
        intensity_image=intensity_image,
        properties=properties,
    )
    if len(props["label"]) == 0:
        schema = {**MOMENT_SCHEMA, **_intensity_moment_schema(intensity_names)}
        moments = pl.DataFrame(schema=schema)
    else:
        labels = np.asarray(props["label"], dtype=np.uint32)
        areas = np.rint(np.asarray(props["area"], dtype=np.float64)).astype(np.uint64)
        coordinate_sums = []
        for axis, offset in enumerate(offset_zyx):
            centroid = np.asarray(props[f"centroid-{axis}"], dtype=np.float64)
            local_sum = np.rint(centroid * areas).astype(np.uint64)
            offset_sum = np.uint64(offset) * areas
            if np.any(local_sum > np.iinfo(np.uint64).max - offset_sum):
                raise OverflowError("Global coordinate moments exceed uint64 capacity.")
            coordinate_sums.append(local_sum + offset_sum)
        moment_columns: dict[str, np.ndarray] = {
            "label": labels,
            "area": areas,
            "z_sum": coordinate_sums[0],
            "y_sum": coordinate_sums[1],
            "x_sum": coordinate_sums[2],
        }
        if intensity_image is not None:
            for channel, name in enumerate(intensity_names):
                intensity_min = np.asarray(props[f"intensity_min-{channel}"], dtype=np.float64)
                intensity_mean = np.asarray(props[f"intensity_mean-{channel}"], dtype=np.float64)
                intensity_max = np.asarray(props[f"intensity_max-{channel}"], dtype=np.float64)
                moment_columns.update(
                    {
                        f"intensity_{name}_sum": intensity_mean * areas,
                        f"intensity_{name}_min": intensity_min,
                        f"intensity_{name}_max": intensity_max,
                        f"intensity_{name}_has_nan": np.isnan(intensity_min),
                    }
                )
        moments = pl.DataFrame(moment_columns)

    plane_frames: list[pl.DataFrame] = []
    for local_z, plane in enumerate(block):
        plane_labels, counts = np.unique(plane, return_counts=True)
        foreground = plane_labels != 0
        if foreground.any():
            plane_frames.append(
                pl.DataFrame(
                    {
                        "label": plane_labels[foreground].astype(np.uint32, copy=False),
                        "plane_z": np.full(
                            int(foreground.sum()),
                            offset_zyx[0] + local_z,
                            dtype=np.uint32,
                        ),
                        "plane_area": counts[foreground].astype(np.uint64, copy=False),
                    }
                )
            )
    planes = pl.concat(plane_frames) if plane_frames else pl.DataFrame(schema=PLANE_SCHEMA)
    return moments, planes


def finalize_measurements(moments: pl.DataFrame, planes: pl.DataFrame) -> pl.DataFrame:
    """Finalize already-reduced additive moments into one row per label."""
    missing_moments = set(MOMENT_SCHEMA) - set(moments.columns)
    missing_planes = set(PLANE_SCHEMA) - set(planes.columns)
    if missing_moments or missing_planes:
        raise ValueError(
            f"Measurement schema mismatch: missing moments={sorted(missing_moments)}, "
            f"missing planes={sorted(missing_planes)}."
        )
    intensity_names = _intensity_names_from_columns(moments.columns)
    intensity_moments = {column for column in moments.columns if column.startswith("intensity_")}
    expected_intensity_moments = set(_intensity_moment_schema(intensity_names))
    if intensity_moments != expected_intensity_moments:
        missing = sorted(expected_intensity_moments - intensity_moments)
        unexpected = sorted(intensity_moments - expected_intensity_moments)
        raise ValueError(
            f"Intensity measurement schema mismatch: missing={missing}, unexpected={unexpected}."
        )
    if moments.is_empty():
        schema = {
            "label": pl.UInt32,
            "area": pl.UInt64,
            "centroid_z": pl.Float64,
            "centroid_y": pl.Float64,
            "centroid_x": pl.Float64,
            "plane_z": pl.UInt32,
            "plane_area": pl.UInt64,
        }
        if intensity_names:
            schema.update({column: pl.Float64 for column in _intensity_output_columns(intensity_names)})
        return pl.DataFrame(schema=schema)
    if planes.is_empty():
        raise ValueError("Non-empty label moments require per-Z plane measurements.")

    moments = moments.with_columns(
        (pl.col("z_sum") / pl.col("area")).alias("centroid_z"),
        (pl.col("y_sum") / pl.col("area")).alias("centroid_y"),
        (pl.col("x_sum") / pl.col("area")).alias("centroid_x"),
    )
    if intensity_names:
        moments = moments.with_columns(
            expression
            for name in intensity_names
            for expression in (
                pl.when(pl.col(f"intensity_{name}_has_nan"))
                .then(float("nan"))
                .otherwise(pl.col(f"intensity_{name}_min"))
                .alias(f"intensity_{name}_min"),
                pl.when(pl.col(f"intensity_{name}_has_nan"))
                .then(float("nan"))
                .otherwise(pl.col(f"intensity_{name}_sum") / pl.col("area"))
                .alias(f"intensity_{name}_mean"),
                pl.when(pl.col(f"intensity_{name}_has_nan"))
                .then(float("nan"))
                .otherwise(pl.col(f"intensity_{name}_max"))
                .alias(f"intensity_{name}_max"),
            )
        )
    best_planes = (
        planes.join(moments.select("label", "centroid_z"), on="label", how="inner")
        .with_columns(
            (pl.col("plane_z").cast(pl.Float64) - pl.col("centroid_z"))
            .abs()
            .alias("z_distance")
        )
        .sort(
            ["label", "plane_area", "z_distance", "plane_z"],
            descending=[False, True, False, False],
        )
        .unique(subset="label", keep="first", maintain_order=True)
        .select("label", "plane_z", "plane_area")
    )
    columns = CELL_COLUMNS + _intensity_output_columns(intensity_names)
    cells = moments.join(best_planes, on="label", how="inner").select(*columns).sort("label")
    if cells.height != moments.height:
        raise ValueError("Plane measurements do not cover every measured label.")
    return cells


def reduce_measurements(
    *,
    moments: Iterable[pl.DataFrame],
    planes: Iterable[pl.DataFrame],
) -> pl.DataFrame:
    """Reduce measurements from disjoint chunks, including seam-crossing labels."""
    moment_list = list(moments)
    plane_list = list(planes)
    if not moment_list or not plane_list:
        raise ValueError("No chunk measurements were supplied.")
    moment_frame = pl.concat(moment_list)
    reduced_moments = moment_frame.group_by("label").agg(_moment_aggregations(moment_frame.columns))
    reduced_planes = pl.concat(plane_list).group_by("label", "plane_z").agg(pl.col("plane_area").sum())
    return finalize_measurements(reduced_moments, reduced_planes)


def _moment_aggregations(columns: Iterable[str]) -> list[pl.Expr]:
    aggregations = [pl.col("area", "z_sum", "y_sum", "x_sum").sum()]
    for name in _intensity_names_from_columns(columns):
        aggregations.extend(
            (
                pl.col(f"intensity_{name}_sum").sum(),
                pl.col(f"intensity_{name}_min").min(),
                pl.col(f"intensity_{name}_max").max(),
                pl.col(f"intensity_{name}_has_nan").any(),
            )
        )
    return aggregations


def _chunk_bounds(shape: tuple[int, ...], chunks: tuple[int, ...]) -> list[ChunkBounds]:
    return [
        (
            z0,
            min(z0 + chunks[0], shape[0]),
            y0,
            min(y0 + chunks[1], shape[1]),
            x0,
            min(x0 + chunks[2], shape[2]),
        )
        for z0 in range(0, shape[0], chunks[0])
        for y0 in range(0, shape[1], chunks[1])
        for x0 in range(0, shape[2], chunks[2])
    ]


def _measure_source_chunk(
    labels_path: str,
    intensity_sources: tuple[tuple[str, str], ...],
    bounds: ChunkBounds,
    base_offset_zyx: tuple[int, int, int],
) -> tuple[ChunkBounds, pl.DataFrame, pl.DataFrame]:
    labels = zarr.open_array(labels_path, mode="r")
    z0, z1, y0, y1, x0, x1 = bounds
    block = np.asarray(labels[z0:z1, y0:y1, x0:x1])
    intensity_image = None
    intensity_names = tuple(name for name, _ in intensity_sources)
    if intensity_sources:
        intensity_arrays = [_open_image_array(Path(path))[0] for _, path in intensity_sources]
        intensity_image = np.empty(
            (*block.shape, len(intensity_arrays)),
            dtype=intensity_arrays[0].dtype,
        )
        for channel, array in enumerate(intensity_arrays):
            intensity_image[..., channel] = array[z0:z1, y0:y1, x0:x1]
    moments, planes = measure_block(
        block,
        offset_zyx=(
            base_offset_zyx[0] + z0,
            base_offset_zyx[1] + y0,
            base_offset_zyx[2] + x0,
        ),
        intensity_image=intensity_image,
        intensity_names=intensity_names,
    )
    return bounds, moments, planes


def _partial_paths(partial_dir: Path, bounds: ChunkBounds) -> tuple[Path, Path]:
    z0, _, y0, _, x0, _ = bounds
    suffix = f"{z0}-{y0}-{x0}.parquet"
    return partial_dir / f"m-{suffix}", partial_dir / f"p-{suffix}"


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=".props-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_identity(labels_path: Path, labels: zarr.Array) -> dict[str, object]:
    metadata_path = labels_path / "zarr.json"
    if not metadata_path.is_file():
        metadata_path = labels_path / ".zarray"
    metadata_sha256 = (
        hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        if metadata_path.is_file()
        else None
    )
    return {
        "schema_version": 1,
        "path": str(labels_path.resolve()),
        "shape": list(labels.shape),
        "chunks": list(labels.chunks),
        "dtype": str(labels.dtype),
        "metadata_sha256": metadata_sha256,
        "artifact_key": labels.attrs.get("squisher_postproc_key")
        or labels.attrs.get("squisher_run_key"),
    }


def _open_image_array(path: Path) -> tuple[zarr.Array, Path]:
    """Open a direct Zarr array or the first dataset of an OME-Zarr group."""
    register_jpegxr_codec()
    node = zarr.open(path, mode="r")
    if isinstance(node, zarr.Array):
        return node, path

    attrs = node.attrs.asdict() if hasattr(node.attrs, "asdict") else dict(node.attrs)
    multiscales = attrs.get("multiscales")
    if multiscales is None and isinstance(attrs.get("ome"), dict):
        multiscales = attrs["ome"].get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError(f"Intensity Zarr group lacks OME multiscales metadata: {path}")
    datasets = multiscales[0].get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Intensity OME-Zarr does not list a level-0 dataset: {path}")
    level_path = datasets[0].get("path") if isinstance(datasets[0], dict) else None
    if not isinstance(level_path, str) or not level_path:
        raise ValueError(f"Intensity OME-Zarr has an invalid level-0 dataset path: {path}")
    array = node[level_path]
    if not isinstance(array, zarr.Array):
        raise ValueError(f"Intensity OME-Zarr level-0 dataset is not an array: {path / level_path}")
    return array, path / level_path


def _prepare_partial_dir(
    partial_dir: Path,
    identity: dict[str, object],
    *,
    resume: bool,
) -> None:
    manifest_path = partial_dir / "run.json"
    if partial_dir.exists() and not resume:
        shutil.rmtree(partial_dir)
    if partial_dir.exists():
        if not manifest_path.is_file():
            raise ValueError(f"Regionprops partial directory lacks run identity: {partial_dir}")
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError(f"Regionprops partial directory belongs to a different source: {partial_dir}")
        return
    partial_dir.mkdir(parents=True)
    manifest_path.write_text(json.dumps(identity, indent=2) + "\n")


def _write_chunk_result(
    partial_dir: Path,
    result: tuple[ChunkBounds, pl.DataFrame, pl.DataFrame],
) -> None:
    bounds, moments, planes = result
    moment_path, plane_path = _partial_paths(partial_dir, bounds)
    _write_parquet_atomic(moments, moment_path)
    _write_parquet_atomic(planes, plane_path)


def _measure_pending_chunks(
    labels_path: Path,
    intensity_sources: tuple[tuple[str, str], ...],
    pending: list[ChunkBounds],
    partial_dir: Path,
    *,
    workers: int,
    offset_zyx: tuple[int, int, int],
) -> None:
    if workers == 1:
        for index, bounds in enumerate(pending, start=1):
            _write_chunk_result(
                partial_dir,
                _measure_source_chunk(
                    str(labels_path),
                    intensity_sources,
                    bounds,
                    offset_zyx,
                ),
            )
            if index % 10 == 0 or index == len(pending):
                logger.info(f"Measured pending chunks={index}/{len(pending)}")
        return

    # Polars owns native worker threads; fork can deadlock after those threads
    # have initialized in the parent process.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        bounds_iter = iter(pending)
        active: dict[Future[tuple[ChunkBounds, pl.DataFrame, pl.DataFrame]], ChunkBounds] = {}
        for _ in range(min(workers, len(pending))):
            bounds = next(bounds_iter)
            active[
                executor.submit(
                    _measure_source_chunk,
                    str(labels_path),
                    intensity_sources,
                    bounds,
                    offset_zyx,
                )
            ] = bounds

        completed = 0
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                del active[future]
                _write_chunk_result(partial_dir, future.result())
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    logger.info(f"Measured pending chunks={completed}/{len(pending)}")
                try:
                    bounds = next(bounds_iter)
                except StopIteration:
                    continue
                active[
                    executor.submit(
                        _measure_source_chunk,
                        str(labels_path),
                        intensity_sources,
                        bounds,
                        offset_zyx,
                    )
                ] = bounds


def _reduce_partial_files(partial_dir: Path) -> pl.DataFrame:
    moment_scan = pl.scan_parquet(str(partial_dir / "m-*.parquet"))
    moments = (
        moment_scan.group_by("label")
        .agg(_moment_aggregations(moment_scan.collect_schema().names()))
        .collect(engine="streaming")
    )
    planes = (
        pl.scan_parquet(str(partial_dir / "p-*.parquet"))
        .group_by("label", "plane_z")
        .agg(pl.col("plane_area").sum())
        .collect(engine="streaming")
    )
    return finalize_measurements(moments, planes)


def measure_zarr(
    labels_path: Path,
    output_path: Path,
    *,
    intensity_paths: Mapping[str, Path] | None = None,
    workers: int = 2,
    offset_zyx: tuple[int, int, int] = (0, 0, 0),
    resume: bool = True,
    overwrite: bool = False,
) -> Path:
    """Measure a 3-D label Zarr with bounded process concurrency.

    Resume assumes the label data remain immutable while partial files exist;
    the stored identity covers array metadata and Squisher artifact identity,
    not a full hash of all label chunks.
    """
    labels_path = Path(labels_path)
    output_path = Path(output_path)
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}.")
    labels = zarr.open_array(labels_path, mode="r")
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected a 3-D integer label Zarr, got shape={labels.shape}, dtype={labels.dtype}.")
    _validate_block(np.empty((0, 0, 0), dtype=labels.dtype), offset_zyx)
    intensity_paths = dict(intensity_paths or {})
    intensity_names = tuple(sorted(_validate_intensity_names(intensity_paths)))
    intensity_sources: list[tuple[str, Path, Path, zarr.Array]] = []
    for name in intensity_names:
        intensity_path = Path(intensity_paths[name])
        intensity, intensity_array_path = _open_image_array(intensity_path)
        if intensity.ndim != 3 or not np.issubdtype(intensity.dtype, np.number):
            raise ValueError(
                f"Expected intensity {name!r} to be a 3-D numeric Zarr, got "
                f"shape={intensity.shape}, dtype={intensity.dtype}."
            )
        if intensity.shape != labels.shape:
            raise ValueError(
                f"Intensity {name!r} and label Zarrs must have identical shapes, got "
                f"intensity={intensity.shape}, labels={labels.shape}."
            )
        if intensity_sources and intensity.dtype != intensity_sources[0][3].dtype:
            raise ValueError(
                "All intensity Zarrs must have identical dtypes to avoid lossy stacking, got "
                f"{intensity_sources[0][3].dtype} and {intensity.dtype}."
            )
        intensity_sources.append((name, intensity_path, intensity_array_path, intensity))
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    partial_dir = output_path.parent / f".{output_path.stem}-parts"
    identity = {
        **_source_identity(labels_path, labels),
        "offset_zyx": list(offset_zyx),
    }
    if intensity_sources:
        identity["intensities"] = {
            name: {
                **_source_identity(array_path, intensity),
                "path": str(source_path.resolve()),
                "array_path": str(array_path.resolve()),
            }
            for name, source_path, array_path, intensity in intensity_sources
        }
    _prepare_partial_dir(partial_dir, identity, resume=resume)

    bounds = _chunk_bounds(tuple(labels.shape), tuple(labels.chunks))
    pending = [
        chunk
        for chunk in bounds
        if not all(path.is_file() for path in _partial_paths(partial_dir, chunk))
    ]
    logger.info(f"Label chunks total={len(bounds)} pending={len(pending)} workers={workers}")
    _measure_pending_chunks(
        labels_path,
        tuple((name, str(source_path)) for name, source_path, _, _ in intensity_sources),
        pending,
        partial_dir,
        workers=workers,
        offset_zyx=offset_zyx,
    )

    if bounds:
        cells = _reduce_partial_files(partial_dir)
    else:
        cells = finalize_measurements(
            pl.DataFrame(
                schema={**MOMENT_SCHEMA, **_intensity_moment_schema(intensity_names)}
            ),
            pl.DataFrame(schema=PLANE_SCHEMA),
        )
    _write_parquet_atomic(cells, output_path)
    shutil.rmtree(partial_dir)
    logger.info(f"Measured {cells.height} labels into {output_path}")
    return output_path
