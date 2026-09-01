"""Bounded overlap-band evidence and reciprocal-IoU label matching."""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

from squisher_segment.segmentation.distributed.cache_utils import atomic_write_text

logger = logging.getLogger(__name__)

FACE_MODE = "face"
IOU_MODE = "overlap-iou"
STITCH_MODES = (FACE_MODE, IOU_MODE)
EVIDENCE_SCHEMA = 1
MAX_CHUNK_BYTES = 32 * 1024**2
MAX_PAIR_COUNTS = 1_000_000


def validate_stitch_options(mode: str, threshold: float) -> tuple[str, float]:
    """Validate the public stitching contract and return canonical values."""
    if not isinstance(mode, str):
        raise ValueError("stitch_mode must be a string.")
    canonical = mode.strip().lower()
    if canonical not in STITCH_MODES:
        raise ValueError(f"stitch_mode must be one of {STITCH_MODES}; got {mode!r}.")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("stitch_iou_threshold must be between 0 and 1.")
    return canonical, float(threshold)


def plan_overlap_evidence(
    selected_crops: Mapping[tuple[int, ...], tuple[slice, ...]],
    *,
    shape: tuple[int, int, int],
    blocksize: tuple[int, int, int],
) -> tuple[list[list[dict[str, Any]]], dict[tuple[int, int, int], list[dict[str, Any]]]]:
    """Enumerate selected neighbors and assign each pair-side to one block."""
    crops = {
        tuple(int(value) for value in index[:3]): tuple(crop[:3])
        for index, crop in selected_crops.items()
    }
    rows: list[list[dict[str, Any]]] = [[], [], []]
    by_block: dict[tuple[int, int, int], list[dict[str, Any]]] = {
        index: [] for index in crops
    }
    for first in sorted(crops):
        for axis in range(3):
            neighbor = list(first)
            neighbor[axis] += 1
            second = tuple(neighbor)
            if second not in crops:
                continue

            bounds: list[list[int]] = []
            for dim in range(3):
                if dim == axis:
                    start = max(crops[first][dim].start, crops[second][dim].start)
                    stop = min(crops[first][dim].stop, crops[second][dim].stop)
                else:
                    start = first[dim] * blocksize[dim]
                    stop = min(start + blocksize[dim], shape[dim])
                    start = max(start, crops[first][dim].start, crops[second][dim].start)
                    stop = min(stop, crops[first][dim].stop, crops[second][dim].stop)
                if stop <= start:
                    raise ValueError(
                        f"Neighbor blocks {first} and {second} have no valid evidence on axis {dim}."
                    )
                bounds.append([int(start), int(stop)])

            row_index = len(rows[axis])
            axis_order = (axis,) + tuple(dim for dim in range(3) if dim != axis)
            valid_shape = [bounds[dim][1] - bounds[dim][0] for dim in axis_order]
            row = {
                "blocks": [first, second],
                "slices": bounds,
                "valid_shape": valid_shape,
            }
            rows[axis].append(row)
            for side, block in enumerate((first, second)):
                by_block[block].append(
                    {
                        "axis": axis,
                        "row": row_index,
                        "side": side,
                        "slices": bounds,
                        "valid_shape": valid_shape,
                    }
                )
    return rows, by_block


def _spatial_chunks(shape: Sequence[int]) -> tuple[int, ...]:
    chunks = [max(1, int(size)) for size in shape]
    while int(np.prod(chunks, dtype=np.int64)) * np.dtype(np.uint32).itemsize > MAX_CHUNK_BYTES:
        largest = max(range(len(chunks)), key=chunks.__getitem__)
        chunks[largest] = max(1, (chunks[largest] + 1) // 2)
    return tuple(chunks)


def _evidence_codecs() -> list[Any]:
    return [
        zarr.codecs.BytesCodec(),
        zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=1,
            shuffle=zarr.codecs.BloscShuffle.noshuffle,
            typesize=np.dtype(np.uint32).itemsize,
        ),
    ]


def initialize_overlap_evidence(
    directory: Path,
    rows: list[list[dict[str, Any]]],
    *,
    run_key: str,
    resume: bool,
) -> None:
    """Create or validate the three pair-centric evidence arrays."""
    payload = {"schema": EVIDENCE_SCHEMA, "run_key": run_key, "rows": rows}
    metadata_path = directory / "metadata.json"
    if resume:
        if not metadata_path.is_file():
            raise RuntimeError(f"Cannot resume: overlap evidence metadata is missing at {metadata_path}.")
        saved = json.loads(metadata_path.read_text())
        if saved != json.loads(json.dumps(payload)):
            raise RuntimeError("Cannot resume: overlap evidence geometry or run identity changed.")
        for axis, axis_rows in enumerate(rows):
            if not axis_rows:
                continue
            axis_path = directory / f"axis-{axis}.zarr"
            if not axis_path.exists():
                raise RuntimeError(f"Cannot resume: overlap evidence axis {axis} is missing.")
            array = zarr.open_array(axis_path, mode="r")
            spatial_shape = tuple(
                max(row["valid_shape"][dim] for row in axis_rows) for dim in range(3)
            )
            if array.shape != (len(axis_rows), 2, *spatial_shape):
                raise RuntimeError(f"Cannot resume: overlap evidence axis {axis} shape changed.")
            if dict(array.attrs) != {"schema": EVIDENCE_SCHEMA, "run_key": run_key}:
                raise RuntimeError(
                    f"Cannot resume: overlap evidence axis {axis} identity changed."
                )
        return

    if directory.exists():
        raise RuntimeError(f"Fresh overlap-IoU run found stale evidence at {directory}.")
    directory.mkdir(parents=True)
    codecs = _evidence_codecs()
    for axis, axis_rows in enumerate(rows):
        if not axis_rows:
            continue
        spatial_shape = tuple(max(row["valid_shape"][dim] for row in axis_rows) for dim in range(3))
        array = zarr.create_array(
            directory / f"axis-{axis}.zarr",
            shape=(len(axis_rows), 2, *spatial_shape),
            chunks=(1, 1, *_spatial_chunks(spatial_shape)),
            dtype=np.uint32,
            serializer=codecs[0],
            compressors=tuple(codecs[1:]),
            config={"write_empty_chunks": False},
        )
        array.attrs.update(schema=EVIDENCE_SCHEMA, run_key=run_key)
    atomic_write_text(metadata_path, json.dumps(payload, indent=2))


def _marker_path(directory: Path, block_index: tuple[int, ...]) -> Path:
    name = "b-" + "-".join(str(value) for value in block_index[:3]) + ".json"
    return directory / "markers" / name


def write_block_marker(directory: Path, block_index: tuple[int, ...], run_key: str) -> None:
    path = _marker_path(directory, block_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({"schema": EVIDENCE_SCHEMA, "run_key": run_key}))


def block_marker_matches(directory: Path, block_index: tuple[int, ...], run_key: str) -> bool:
    path = _marker_path(directory, block_index)
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    return payload == {"schema": EVIDENCE_SCHEMA, "run_key": run_key}


def write_block_evidence(
    directory: Path,
    block_index: tuple[int, ...],
    assignments: Sequence[Mapping[str, Any]],
    labels: NDArray[np.uint32],
    crop: tuple[slice, ...],
    *,
    run_key: str,
) -> None:
    """Write every exclusively-owned pair side, then publish one block marker."""
    for assignment in assignments:
        axis = int(assignment["axis"])
        bounds = assignment["slices"]
        local_slices = tuple(
            slice(int(bound[0]) - crop[dim].start, int(bound[1]) - crop[dim].start)
            for dim, bound in enumerate(bounds)
        )
        band = np.moveaxis(np.asarray(labels[local_slices], dtype=np.uint32), axis, 0)
        valid_shape = tuple(int(value) for value in assignment["valid_shape"])
        if band.shape != valid_shape:
            raise ValueError(
                f"Overlap evidence shape {band.shape} does not match planned {valid_shape} "
                f"for block {block_index}."
            )
        array = zarr.open_array(directory / f"axis-{axis}.zarr", mode="r+")
        target = (int(assignment["row"]), int(assignment["side"])) + tuple(
            slice(0, size) for size in valid_shape
        )
        array[target] = band
    write_block_marker(directory, block_index, run_key)


def _add_counts(total: NDArray[np.int64], values: NDArray[np.uint32]) -> NDArray[np.int64]:
    counts = np.bincount(values.ravel())
    if counts.size > total.size:
        total = np.pad(total, (0, counts.size - total.size))
    total[: counts.size] += counts
    return total


def _fraction_order(first: tuple[int, int], second: tuple[int, int]) -> int:
    left = first[0] * second[1]
    right = second[0] * first[1]
    return (left > right) - (left < right)


def _reciprocal_pairs(
    area_first: NDArray[np.int64],
    area_second: NDArray[np.int64],
    intersections: Mapping[tuple[int, int], int],
    threshold: float,
) -> NDArray[np.uint32]:
    candidates: list[tuple[int, int, int, int]] = []
    for (first, second), intersection in intersections.items():
        union = int(area_first[first] + area_second[second] - intersection)
        if union > 0 and intersection / union >= threshold:
            candidates.append((first, second, intersection, union))

    best_first: dict[int, tuple[int, int, int, bool]] = {}
    best_second: dict[int, tuple[int, int, int, bool]] = {}

    def update(
        best: dict[int, tuple[int, int, int, bool]],
        owner: int,
        other: int,
        intersection: int,
        union: int,
    ) -> None:
        current = best.get(owner)
        if current is None:
            best[owner] = (other, intersection, union, False)
            return
        order = _fraction_order((intersection, union), (current[1], current[2]))
        if order > 0:
            best[owner] = (other, intersection, union, False)
        elif order == 0 and other != current[0]:
            best[owner] = (current[0], current[1], current[2], True)

    for first, second, intersection, union in candidates:
        update(best_first, first, second, intersection, union)
        update(best_second, second, first, intersection, union)

    accepted = [
        (first, second)
        for first, second, _intersection, _union in candidates
        if best_first[first] == (second, _intersection, _union, False)
        and best_second[second] == (first, _intersection, _union, False)
    ]
    if not accepted:
        return np.empty((2, 0), dtype=np.uint32)
    return np.asarray(sorted(accepted), dtype=np.uint32).T


def match_overlap_iou(
    first: NDArray[np.uint32],
    second: NDArray[np.uint32],
    *,
    threshold: float,
) -> NDArray[np.uint32]:
    """Match one in-memory overlap band using unique reciprocal-best IoU."""
    _, threshold = validate_stitch_options(IOU_MODE, threshold)
    first = np.asarray(first, dtype=np.uint32)
    second = np.asarray(second, dtype=np.uint32)
    if first.shape != second.shape:
        raise ValueError(f"Overlap bands must have equal shapes; got {first.shape} and {second.shape}.")
    area_first = _add_counts(np.zeros(0, dtype=np.int64), first)
    area_second = _add_counts(np.zeros(0, dtype=np.int64), second)
    foreground = (first != 0) & (second != 0)
    intersections: dict[tuple[int, int], int] = {}
    if np.any(foreground):
        packed = (first[foreground].astype(np.uint64) << np.uint64(32)) | second[
            foreground
        ].astype(np.uint64)
        keys, counts = np.unique(packed, return_counts=True)
        intersections = {
            (int(key >> np.uint64(32)), int(key & np.uint64(0xFFFFFFFF))): int(count)
            for key, count in zip(keys, counts, strict=True)
        }
    return _reciprocal_pairs(area_first, area_second, intersections, threshold)


def _chunk_slices(shape: Sequence[int], chunks: Sequence[int]):
    ranges = [range(0, int(size), int(chunk)) for size, chunk in zip(shape, chunks, strict=True)]
    for starts in itertools.product(*ranges):
        yield tuple(
            slice(start, min(start + int(chunk), int(size)))
            for start, chunk, size in zip(starts, chunks, shape, strict=True)
        )


def _match_stored_row(
    array: zarr.Array,
    row_index: int,
    valid_shape: tuple[int, int, int],
    threshold: float,
) -> tuple[NDArray[np.uint32], int, int]:
    area_first = np.zeros(0, dtype=np.int64)
    area_second = np.zeros(0, dtype=np.int64)
    intersections: dict[tuple[int, int], int] = {}
    peak_scratch = 0
    for slices in _chunk_slices(valid_shape, array.chunks[2:]):
        first = np.asarray(array[(row_index, 0, *slices)], dtype=np.uint32)
        second = np.asarray(array[(row_index, 1, *slices)], dtype=np.uint32)
        area_first = _add_counts(area_first, first)
        area_second = _add_counts(area_second, second)
        foreground = (first != 0) & (second != 0)
        peak_scratch = max(peak_scratch, first.nbytes + second.nbytes + foreground.nbytes)
        if np.any(foreground):
            packed = (first[foreground].astype(np.uint64) << np.uint64(32)) | second[
                foreground
            ].astype(np.uint64)
            keys, counts = np.unique(packed, return_counts=True)
            peak_scratch = max(peak_scratch, first.nbytes + second.nbytes + packed.nbytes)
            new_pairs = sum(
                (int(key >> np.uint64(32)), int(key & np.uint64(0xFFFFFFFF)))
                not in intersections
                for key in keys
            )
            if len(intersections) + new_pairs > MAX_PAIR_COUNTS:
                raise MemoryError(
                    f"One overlap row exceeds the bounded {MAX_PAIR_COUNTS:,} label-pair limit."
                )
            for key, count in zip(keys, counts, strict=True):
                pair = (int(key >> np.uint64(32)), int(key & np.uint64(0xFFFFFFFF)))
                intersections[pair] = intersections.get(pair, 0) + int(count)
    return _reciprocal_pairs(area_first, area_second, intersections, threshold), len(intersections), peak_scratch


def match_evidence(
    directory: Path,
    *,
    threshold: float,
    nblocks: tuple[int, int, int],
    label_bits: int,
) -> list[NDArray[np.uint32]]:
    """Stream stored bands and encode accepted local pairs as global IDs."""
    _, threshold = validate_stitch_options(IOU_MODE, threshold)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError(f"Unsupported overlap evidence schema: {metadata.get('schema')!r}.")
    started = time.perf_counter()
    result: list[NDArray[np.uint32]] = []
    candidate_count = 0
    accepted_count = 0
    peak_scratch = 0
    logical_bytes = 0
    for axis, rows in enumerate(metadata["rows"]):
        if not rows:
            continue
        array = zarr.open_array(directory / f"axis-{axis}.zarr", mode="r")
        for row_index, row in enumerate(rows):
            valid_shape = tuple(int(value) for value in row["valid_shape"])
            local_pairs, candidates, scratch = _match_stored_row(
                array, row_index, valid_shape, threshold
            )
            candidate_count += candidates
            peak_scratch = max(peak_scratch, scratch)
            logical_bytes += 2 * int(np.prod(valid_shape, dtype=np.int64)) * 4
            if not local_pairs.size:
                continue
            blocks = [tuple(int(value) for value in block) for block in row["blocks"]]
            encoded = local_pairs.copy()
            for side, block in enumerate(blocks):
                token = int(np.ravel_multi_index(block, nblocks))
                encoded[side] |= np.uint32(token << label_bits)
            result.append(encoded)
            accepted_count += encoded.shape[1]
    logger.info(
        "Overlap-IoU matching: %.2fs, %.2f GiB logical evidence, %d candidates, "
        "%d accepted, %.1f MiB peak chunk scratch",
        time.perf_counter() - started,
        logical_bytes / 1024**3,
        candidate_count,
        accepted_count,
        peak_scratch / 1024**2,
    )
    return result
