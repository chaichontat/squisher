from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import zarr
from loguru import logger
from skimage.measure import regionprops_table


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
ChunkBounds = tuple[int, int, int, int, int, int]


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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Measure additive label moments and per-Z areas for one decoded ZYX block.

    Blocks must be disjoint. Label zero is background, and labels must already
    be globally consistent across chunks.
    """
    block = np.asarray(block)
    _validate_block(block, offset_zyx)
    props = regionprops_table(block, properties=("label", "area", "centroid"))
    if len(props["label"]) == 0:
        moments = pl.DataFrame(schema=MOMENT_SCHEMA)
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
        moments = pl.DataFrame(
            {
                "label": labels,
                "area": areas,
                "z_sum": coordinate_sums[0],
                "y_sum": coordinate_sums[1],
                "x_sum": coordinate_sums[2],
            }
        )

    plane_frames: list[pl.DataFrame] = []
    for local_z, plane in enumerate(block):
        labels, counts = np.unique(plane, return_counts=True)
        foreground = labels != 0
        if foreground.any():
            plane_frames.append(
                pl.DataFrame(
                    {
                        "label": labels[foreground].astype(np.uint32, copy=False),
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
    if moments.is_empty():
        return pl.DataFrame(
            schema={
                "label": pl.UInt32,
                "area": pl.UInt64,
                "centroid_z": pl.Float64,
                "centroid_y": pl.Float64,
                "centroid_x": pl.Float64,
                "plane_z": pl.UInt32,
                "plane_area": pl.UInt64,
            }
        )
    if planes.is_empty():
        raise ValueError("Non-empty label moments require per-Z plane measurements.")

    moments = moments.with_columns(
        (pl.col("z_sum") / pl.col("area")).alias("centroid_z"),
        (pl.col("y_sum") / pl.col("area")).alias("centroid_y"),
        (pl.col("x_sum") / pl.col("area")).alias("centroid_x"),
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
    cells = (
        moments.join(best_planes, on="label", how="inner")
        .select(*CELL_COLUMNS)
        .sort("label")
    )
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
    reduced_moments = (
        pl.concat(moment_list)
        .group_by("label")
        .agg(pl.col("area", "z_sum", "y_sum", "x_sum").sum())
    )
    reduced_planes = (
        pl.concat(plane_list)
        .group_by("label", "plane_z")
        .agg(pl.col("plane_area").sum())
    )
    return finalize_measurements(reduced_moments, reduced_planes)


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
    bounds: ChunkBounds,
    base_offset_zyx: tuple[int, int, int],
) -> tuple[ChunkBounds, pl.DataFrame, pl.DataFrame]:
    labels = zarr.open_array(labels_path, mode="r")
    z0, z1, y0, y1, x0, x1 = bounds
    block = np.asarray(labels[z0:z1, y0:y1, x0:x1])
    moments, planes = measure_block(
        block,
        offset_zyx=(
            base_offset_zyx[0] + z0,
            base_offset_zyx[1] + y0,
            base_offset_zyx[2] + x0,
        ),
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
                _measure_source_chunk(str(labels_path), bounds, offset_zyx),
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
            active[executor.submit(_measure_source_chunk, str(labels_path), bounds, offset_zyx)] = bounds

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
                active[executor.submit(_measure_source_chunk, str(labels_path), bounds, offset_zyx)] = bounds


def _reduce_partial_files(partial_dir: Path) -> pl.DataFrame:
    moments = (
        pl.scan_parquet(str(partial_dir / "m-*.parquet"))
        .group_by("label")
        .agg(pl.col("area", "z_sum", "y_sum", "x_sum").sum())
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
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    partial_dir = output_path.parent / f".{output_path.stem}-parts"
    identity = {
        **_source_identity(labels_path, labels),
        "offset_zyx": list(offset_zyx),
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
        pending,
        partial_dir,
        workers=workers,
        offset_zyx=offset_zyx,
    )

    if bounds:
        cells = _reduce_partial_files(partial_dir)
    else:
        cells = finalize_measurements(
            pl.DataFrame(schema=MOMENT_SCHEMA),
            pl.DataFrame(schema=PLANE_SCHEMA),
        )
    _write_parquet_atomic(cells, output_path)
    shutil.rmtree(partial_dir)
    logger.info(f"Measured {cells.height} labels into {output_path}")
    return output_path
