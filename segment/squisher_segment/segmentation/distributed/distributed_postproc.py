"""
Distributed 3D post-processing pipeline for segmentation masks.

Applies the 4-phase postproc3d pipeline to chunked zarr data using Dask,
with stitching via face-matching union-find.

Design Rationale
----------------
The postproc3d pipeline consists of 4 phases:
  1. gaussian_smooth_labels - Gaussian-weighted voting to smooth jagged boundaries
  2. relabel_connected_components - Assign unique IDs to disconnected fragments
  3. compute_metadata_and_adjacency - Compute volumes and contact areas
  4. donate_small_cells - Absorb tiny fragments into neighboring cells

For very large volumes (>4000^3 voxels), running these phases on the full volume
is infeasible due to memory constraints. This module implements a chunked approach.

Key Design Decisions
--------------------
1. **Wrap all 4 phases in one function per chunk**
   Rather than chunking each phase separately (which would require complex
   cross-chunk coordination for phases 2-4), we run the entire pipeline on
   each overlapped chunk. This keeps the per-chunk logic identical to the
   non-distributed case.

2. **Use the same overlap removal as distributed_segmentation**
   - Each chunk is read with ZYX overlap (configurable via `margin` → `overlap = 2*margin`)
   - Core blocks default to the input segmentation's ZYX chunks
   - After processing, overlaps are removed using `remove_overlaps`, matching the stitching behavior
     of `distributed_segmentation`
   - The overlap must be large enough that the trimmed core region has correct context from all directions

3. **Reuse label encoding from distributed_segmentation**
   After processing, chunks have locally-processed labels that need stitching:
   - global_segment_ids() encodes block index into bit-packed label IDs (avoids collisions)
   - block_faces() extracts and shrinks boundary faces on workers
   - neighboring face tasks return only unique touching label pairs
   - scipy.sparse.csgraph.connected_components() determines merges (union-find) in this compact space
   - A sorted sparse mapping is applied chunkwise via dask.array.map_blocks

4. **Overlap sizing for Gaussian smoothing**
   Gaussian smoothing with per-axis sigma has effective radius ~4*max(sigma) voxels.
   With default sigma=(1,2,2), the radius is ~8 voxels. The default 60-voxel
   halo sits comfortably beyond that radius in every axis. This ensures Gaussian
   voting at the core boundary has essentially identical context from neighboring
   chunks.

5. **Small cell donation at boundaries**
   Small cells near chunk boundaries might have their best neighbor in another
   chunk. Overlap provides local context, but donation remains approximate for
   components whose relevant volume or adjacency extends beyond the halo.

Pipeline Flow
-------------
```
Input zarr (from distributed_segmentation)
        |
        v
+--------------------------------------------------+
| Per-chunk (parallel via Dask):                   |
|   1. Read chunk with overlap                     |
|   2. gaussian_smooth_labels_cupy (sigma=4)       |
|   3. relabel_connected_components                |
|   4. compute_metadata_and_adjacency              |
|   5. donate_small_cells                          |
|   6. Remove overlaps (match distributed_segmentation) |
|   7. Assign globally unique IDs                  |
|   8. Extract faces, write to temp zarr           |
+--------------------------------------------------+
        |
        v
Stitch:
   - Worker-side neighboring face tasks emit unique label pairs
   - connected_components merges a compact sparse graph
   - Apply sorted sparse remap via dask.array.map_blocks
        |
        v
Output zarr (post-processed masks)
```

Usage
-----
CLI:
    python -m squisher_segment.segmentation.distributed.distributed_postproc \\
        run /path/to/segmentation.zarr \\
        --blocksize 120 712 504 --sigma \"1,2,2\" --v-min 8000

Programmatic:
    from squisher_segment.segmentation.distributed.distributed_postproc import distributed_postproc
    result = distributed_postproc(input_zarr, write_path, sigma=(1, 2, 2), V_min=8000, ...)
"""

import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cupy as cp
import dask.array
import numpy as np
import click
import zarr
from loguru import logger
from numpy.typing import NDArray

from squisher_segment.segment.postproc3d import (  # noqa: F401
    absorb_encircled_rois,
    compute_metadata_and_adjacency,
    donate_small_cells,
    gaussian_erosion_to_margin_and_scale,
    gaussian_smooth_labels_cupy,
    relabel_connected_components,
)
from squisher_segment.segmentation.distributed.gpu_cluster import cluster, myGPUCluster, myLocalCluster
from squisher_segment.segmentation.distributed.merge_utils import (
    GLOBAL_LABEL_BITS,
    block_faces,
    create_zarr_array,
    determine_sparse_merge_relabeling,
    find_label_pairs_across_boundary,
    get_block_crops,
    get_nblocks,
    global_segment_ids,
    label_zarr_codecs,
    remove_overlaps,
    shrink_labels,
    sparse_relabel_and_write,
    write_dask_to_zarr,
)


@contextmanager
def progress_bar(total: int):
    def _advance(*args: Any, **kwargs: Any) -> None:
        return None

    yield _advance


def _parse_sigma_option(val: str) -> float | tuple[float, float, float]:
    """
    Parse --sigma value as either a scalar or a Z,Y,X triple.

    Examples:
    - "4" or "4.0" -> 4.0
    - "2,1.5,1.5" or "2 1.5 1.5" -> (2.0, 1.5, 1.5)
    """
    s = val.strip().replace(" ", ",")
    parts = [p for p in s.split(",") if p]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 3:
        z, y, x = (float(p) for p in parts)
        return (z, y, x)
    raise click.BadParameter("sigma must be a number or a 'z,y,x' triple")


def _resolve_blocksize(
    input_zarr: zarr.Array,
    requested: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Resolve ZYX cores without silently reverting to full-volume reads."""
    source = input_zarr.chunks if requested is None else requested
    values = tuple(int(value) for value in source)
    if len(values) != input_zarr.ndim:
        raise ValueError(f"blocksize must have {input_zarr.ndim} dimensions; got {values}.")
    if any(value <= 0 for value in values):
        raise ValueError(f"blocksize extents must be positive; got {values}.")
    return tuple(
        min(value, int(size))
        for value, size in zip(values, input_zarr.shape, strict=True)
    )


def _validate_tiling(
    shape: tuple[int, ...],
    blocksize: tuple[int, int, int],
    overlap: int,
) -> NDArray[np.int_]:
    """Reject grids that overlap removal or global ID packing cannot represent."""
    if overlap < 0:
        raise ValueError(f"overlap must be nonnegative; got {overlap}.")
    invalid_axes = [
        axis
        for axis, (size, core) in enumerate(zip(shape, blocksize, strict=True))
        if size > core and overlap >= core
    ]
    if invalid_axes:
        raise ValueError(
            f"overlap {overlap} must be smaller than tiled core extents; "
            f"invalid axes {invalid_axes} for blocksize {blocksize}."
        )

    nblocks = get_nblocks(shape, np.asarray(blocksize))
    block_count = int(np.prod(nblocks, dtype=np.int64))
    max_blocks = 1 << (32 - GLOBAL_LABEL_BITS)
    if block_count > max_blocks:
        raise ValueError(
            f"Postproc grid has {block_count} blocks, exceeding the {max_blocks} "
            f"blocks representable with {GLOBAL_LABEL_BITS} label bits."
        )
    return nblocks


def _postproc_face(
    result: tuple[list[NDArray[Any]], NDArray[np.uint32], NDArray[np.uint64]],
    face_index: int,
    axis: int,
) -> NDArray[np.uint32]:
    """Extract one worker result face and normalize its boundary axis."""
    return np.moveaxis(result[0][face_index], axis, 0)


def _postproc_box_stats(
    result: tuple[list[NDArray[Any]], NDArray[np.uint32], NDArray[np.uint64]],
) -> tuple[NDArray[np.uint32], NDArray[np.uint64]]:
    """Extract aligned global IDs and exact core voxel counts."""
    return result[1], result[2]


def _aggregate_final_label_volumes(
    global_ids: NDArray[np.uint32],
    core_counts: NDArray[np.uint64],
    mapping: NDArray[np.uint32],
) -> NDArray[np.uint64]:
    """Sum non-overlapping core counts through the global-to-final mapping."""
    if global_ids.ndim != 1 or core_counts.shape != global_ids.shape:
        raise ValueError("Global IDs and core counts must be aligned 1D arrays.")
    if mapping.ndim != 2 or mapping.shape[0] != 2:
        raise ValueError("Label mapping must have global and final ID rows.")

    order = np.argsort(global_ids)
    sorted_ids = global_ids[order]
    if not np.array_equal(sorted_ids, mapping[0]):
        raise ValueError("Label mapping must exactly cover the counted global IDs.")

    volumes = np.zeros(int(mapping[1].max()) + 1, dtype=np.uint64)
    np.add.at(volumes, mapping[1], core_counts[order])
    return volumes


def _boundary_label_pairs(
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
) -> NDArray[np.uint32]:
    """Return unique global-label contacts for two normalized block faces."""
    if left.shape != right.shape or left.ndim != 3 or left.shape[0] != 1:
        raise ValueError(
            "Adjacent faces must have matching 3D shapes with a singleton boundary axis; "
            f"got {left.shape} and {right.shape}."
        )
    paired = np.concatenate((left, right), axis=0)
    return find_label_pairs_across_boundary(paired).astype(np.uint32, copy=False)


def _boundary_label_contacts(
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
) -> tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint64]]:
    """Return conservative stitch pairs plus counted raw face overlaps."""
    if left.shape != right.shape or left.ndim != 3 or left.shape[0] != 1:
        raise ValueError(
            "Adjacent faces must have matching 3D shapes with a singleton boundary axis; "
            f"got {left.shape} and {right.shape}."
        )
    robust_pairs = _boundary_label_pairs(
        shrink_labels(left, 1.0),
        shrink_labels(right, 1.0),
    )

    side0, side1 = left[0], right[0]
    foreground = (side0 > 0) & (side1 > 0)
    if not np.any(foreground):
        return (
            robust_pairs,
            np.empty((2, 0), dtype=np.uint32),
            np.empty(0, dtype=np.uint64),
        )

    raw_pairs = np.sort(
        np.vstack((side0[foreground], side1[foreground])).astype(np.uint32),
        axis=0,
    )
    raw_pairs, raw_counts = np.unique(raw_pairs, axis=1, return_counts=True)
    return robust_pairs, raw_pairs, raw_counts.astype(np.uint64, copy=False)


def _select_small_boundary_merges(
    mapping: NDArray[np.uint32],
    final_volumes: NDArray[np.uint64],
    raw_pairs: NDArray[np.uint32],
    contact_counts: NDArray[np.uint64],
    *,
    V_min: int,
) -> NDArray[np.uint32]:
    """Choose one strongest large recipient for each small seam component."""
    if raw_pairs.ndim != 2 or raw_pairs.shape[0] != 2:
        raise ValueError(f"Raw boundary pairs must have shape (2, N); got {raw_pairs.shape}.")
    if contact_counts.shape != (raw_pairs.shape[1],):
        raise ValueError("Raw boundary pairs and contact counts must be aligned.")
    if V_min <= 0 or raw_pairs.shape[1] == 0:
        return np.empty((2, 0), dtype=np.uint32)

    positions = np.searchsorted(mapping[0], raw_pairs)
    if np.any(positions == mapping.shape[1]) or not np.array_equal(
        mapping[0, positions], raw_pairs
    ):
        raise ValueError("Raw boundary pairs contain a global ID absent from the mapping.")
    final_pairs = mapping[1, positions]
    side0_small = final_volumes[final_pairs[0]] < V_min
    side1_small = final_volumes[final_pairs[1]] < V_min
    eligible = (final_pairs[0] != final_pairs[1]) & (side0_small ^ side1_small)
    if not np.any(eligible):
        return np.empty((2, 0), dtype=np.uint32)

    eligible_indices = np.flatnonzero(eligible)
    small_final = np.where(side0_small[eligible], final_pairs[0, eligible], final_pairs[1, eligible])
    large_final = np.where(side0_small[eligible], final_pairs[1, eligible], final_pairs[0, eligible])
    keys = (small_final.astype(np.uint64) << np.uint64(32)) | large_final.astype(np.uint64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    totals = np.zeros(unique_keys.size, dtype=np.uint64)
    np.add.at(totals, inverse, contact_counts[eligible])

    unique_small = (unique_keys >> np.uint64(32)).astype(np.uint32)
    unique_large = unique_keys.astype(np.uint32)
    order = np.lexsort((unique_large, np.iinfo(np.uint64).max - totals, unique_small))
    _, first = np.unique(unique_small[order], return_index=True)
    winning_groups = order[first]

    first_pair = np.full(unique_keys.size, keys.size, dtype=np.int64)
    np.minimum.at(first_pair, inverse, np.arange(keys.size))
    return raw_pairs[:, eligible_indices[first_pair[winning_groups]]]


def _drop_small_final_labels(
    mapping: NDArray[np.uint32],
    final_volumes: NDArray[np.uint64],
    *,
    V_min: int,
) -> NDArray[np.uint32]:
    """Map unrecoverable sub-threshold components to background and compact IDs."""
    if V_min <= 0:
        return mapping
    keep = np.flatnonzero(final_volumes >= V_min).astype(np.uint32, copy=False)
    translate = np.zeros(final_volumes.size, dtype=np.uint32)
    translate[keep] = np.arange(1, keep.size + 1, dtype=np.uint32)
    filtered = mapping.copy()
    filtered[1] = translate[mapping[1]]
    return filtered


def process_postproc_block(
    block_index: tuple[int, ...],
    crop: tuple[slice, ...],
    input_zarr: zarr.Array,
    output_zarr: zarr.Array,
    blocksize: tuple[int, ...],
    overlap: int,
    nblocks: NDArray[np.int_],
    postproc_kwargs: dict[str, Any],
) -> tuple[list[NDArray[Any]], NDArray[np.uint32], NDArray[np.uint64]]:
    """
    Process one block through all 4 post-processing phases.

    Parameters
    ----------
    block_index : tuple of int
        The (z, y, x) index of this block in the block grid
    crop : tuple of slice
        The crop coordinates (with overlap) to read from input
    input_zarr : zarr.Array
        Input segmentation masks
    output_zarr : zarr.Array
        Output zarr to write results (after overlap removal)
    blocksize : tuple of int
        Target block size (without overlap)
    overlap : int
        Number of voxels of spatial overlap used when building crops
    nblocks : NDArray
        Number of blocks along each axis
    postproc_kwargs : dict
        Parameters for post-processing (sigma, V_min, bg_scale, etc.)

    Returns
    -------
    tuple[list[NDArray], NDArray[np.uint32], NDArray[np.uint64]]
        Raw boundary faces, global IDs, and aligned core voxel counts.
    """
    t_block_start = time.perf_counter()
    logger.debug(f"Processing block {block_index}")

    # 1. Read chunk
    t0 = time.perf_counter()
    masks = np.asarray(input_zarr[crop])
    t_read = time.perf_counter()
    logger.debug(f"  Block {block_index}: read {masks.shape} in {(t_read - t0) * 1000:.1f} ms")
    logger.debug(
        f"  Block {block_index}: dtype={masks.dtype}, contiguous={masks.flags.c_contiguous}, "
        f"strides={masks.strides}, max_label={int(masks.max())}"
    )

    # Ensure integer dtype
    if not np.issubdtype(masks.dtype, np.integer):
        masks = masks.astype(np.int32)

    # Relabel to sequential values for efficient processing.
    # Input from distributed_segmentation has sparse bit-packed global IDs which
    # cause O(max_label) allocations in postproc functions. Sequential labels
    # reduce max_label from millions to the actual label count (~1000).
    # Use CuPy for fast GPU-accelerated unique (~100ms vs 4-8s on CPU).
    max_label_before = int(masks.max())
    t_relabel = time.perf_counter()
    masks_gpu = cp.asarray(masks)
    _, inverse = cp.unique(masks_gpu, return_inverse=True)
    masks = cp.asnumpy(inverse.reshape(masks.shape)).astype(np.int32)
    del masks_gpu, inverse
    t_relabel_done = time.perf_counter()
    logger.debug(
        f"  Block {block_index}: relabeled {max_label_before} -> {int(masks.max())} "
        f"in {(t_relabel_done - t_relabel) * 1000:.1f} ms"
    )

    # 2. Run 4-phase pipeline
    # Phase 1: Gaussian smooth
    sigma = postproc_kwargs.get("sigma", 4.0)
    max_expansion = postproc_kwargs.get("max_expansion", 1)

    # Compute bg_scale from FWHM fraction for slight dilation (matches profile script)
    # Use max sigma if tuple, since gaussian_erosion_to_margin_and_scale expects scalar
    sigma_for_scale = max(sigma) if isinstance(sigma, tuple) else sigma
    _, bg_scale = gaussian_erosion_to_margin_and_scale(sigma=sigma_for_scale, fwhm_fraction=-0.1)
    # Allow override if explicitly provided
    bg_scale = postproc_kwargs.get("bg_scale", bg_scale)

    t1 = time.perf_counter()
    try:
        masks = gaussian_smooth_labels_cupy(
            masks,
            sigma=sigma,
            in_place=True,
            bg_scale=bg_scale,
            max_expansion=max_expansion,
        )
    except ImportError:
        # Fall back to CPU if CuPy/CUDA not available
        from squisher_segment.segment.postproc3d import gaussian_smooth_labels

        masks = gaussian_smooth_labels(
            masks,
            sigma=sigma,
            in_place=True,
            bg_scale=bg_scale,
            max_expansion=max_expansion,
        )
    t_phase1 = time.perf_counter()
    logger.debug(f"  Block {block_index}: Phase 1 (gaussian_smooth) in {(t_phase1 - t1) * 1000:.1f} ms")
    logger.debug(f"  Block {block_index}: max_label after Phase 1 = {int(masks.max())}")

    # Phase 1.5: Absorb encircled ROIs (per 2D slice)
    t15_start = time.perf_counter()
    masks = absorb_encircled_rois(masks, in_place=True)
    t_phase15 = time.perf_counter()
    logger.debug(
        f"  Block {block_index}: Phase 1.5 (absorb_encircled) in {(t_phase15 - t15_start) * 1000:.1f} ms"
    )
    logger.debug(f"  Block {block_index}: max_label after Phase 1.5 = {int(masks.max())}")

    # Phase 2: Relabel connected components
    t2_start = time.perf_counter()
    masks = relabel_connected_components(masks, in_place=True)
    t_phase2 = time.perf_counter()
    logger.debug(f"  Block {block_index}: Phase 2 (relabel_cc) in {(t_phase2 - t2_start) * 1000:.1f} ms")
    logger.debug(f"  Block {block_index}: max_label after Phase 2 = {int(masks.max())}")

    # Phase 3: Compute metadata
    V_min = postproc_kwargs.get("V_min", 8000)
    min_contact_fraction = postproc_kwargs.get("min_contact_fraction", 0.0)

    t3_start = time.perf_counter()
    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)
    t_phase3 = time.perf_counter()
    logger.debug(f"  Block {block_index}: Phase 3 (metadata) in {(t_phase3 - t3_start) * 1000:.1f} ms")

    # Phase 4: Donate small cells
    t4_start = time.perf_counter()
    masks = donate_small_cells(
        masks,
        volumes=volumes,
        adjacency=adjacency,
        contact_areas=contact_areas,
        V_min=V_min,
        min_contact_fraction=min_contact_fraction,
        in_place=True,
    )
    t_phase4 = time.perf_counter()
    logger.debug(f"  Block {block_index}: Phase 4 (donate_small) in {(t_phase4 - t4_start) * 1000:.1f} ms")

    if V_min > 0:
        # After donation, only labels with volume >= V_min remain.
        n_labels = int(np.count_nonzero(volumes >= V_min))
    else:
        # No donation happened; all non-zero volumes remain as labels.
        n_labels = int(np.count_nonzero(volumes > 0))
    logger.debug(f"  Block {block_index}: {n_labels} labels after postproc")

    del volumes, adjacency, contact_areas

    # 3. Remove overlaps to match distributed_segmentation behavior
    t_overlap_start = time.perf_counter()
    masks_cropped, crop_trimmed = remove_overlaps(
        masks,
        crop,
        overlap,
        blocksize,
    )
    # Make masks_cropped independent so we can free the original masks array
    masks_cropped = masks_cropped.copy()
    del masks
    crop_trimmed = tuple(crop_trimmed)
    t_overlap = time.perf_counter()
    logger.debug(f"  Block {block_index}: remove_overlaps in {(t_overlap - t_overlap_start) * 1000:.1f} ms")

    # 4. Find existing local labels (O(N) via bincount, output size = max_label)
    # Do this BEFORE global_segment_ids to avoid O(N log N) unique on huge IDs
    t_unique_start = time.perf_counter()
    max_local = int(masks_cropped.max())
    counts = np.bincount(masks_cropped.ravel(), minlength=max_local + 1)
    local_ids = np.nonzero(counts)[0]
    local_ids = local_ids[local_ids > 0]  # Exclude background
    core_counts = counts[local_ids].astype(np.uint64, copy=False)
    t_unique = time.perf_counter()
    logger.debug(f"  Block {block_index}: find local IDs in {(t_unique - t_unique_start) * 1000:.1f} ms")

    # 5. Assign globally unique IDs
    t_global_start = time.perf_counter()
    masks_global, remap = global_segment_ids(masks_cropped, block_index, nblocks)
    del masks_cropped  # No longer needed after global_segment_ids
    # Convert local IDs to global IDs using remap
    box_ids = remap[local_ids].astype(np.uint32)
    t_global = time.perf_counter()
    logger.debug(f"  Block {block_index}: global_segment_ids in {(t_global - t_global_start) * 1000:.1f} ms")

    # 6. Extract raw faces for worker-side conservative matching and recovery.
    t_faces_start = time.perf_counter()
    faces = block_faces(masks_global)
    t_faces = time.perf_counter()
    logger.debug(f"  Block {block_index}: block_faces in {(t_faces - t_faces_start) * 1000:.1f} ms")

    # 7. Write to output zarr (masks_global is already uint32 from global_segment_ids)
    t_write_start = time.perf_counter()
    output_zarr[crop_trimmed] = masks_global
    del masks_global
    t_write = time.perf_counter()
    logger.debug(f"  Block {block_index}: write zarr in {(t_write - t_write_start) * 1000:.1f} ms")

    t_block_end = time.perf_counter()
    logger.debug(
        f"  Block {block_index}: TOTAL {(t_block_end - t_block_start) * 1000:.1f} ms "
        f"(phases: {(t_phase4 - t1) * 1000:.1f} ms, overhead: {((t_block_end - t_block_start) - (t_phase4 - t1)) * 1000:.1f} ms)"
    )

    cp.get_default_memory_pool().free_all_blocks()
    return faces, box_ids, core_counts


def _copy_zarr_metadata(
    input_zarr: zarr.Array,
    output_path: Path,
    input_path: Path | None = None,
    nblocks: tuple[int, ...] | None = None,
    mapping_filename: str | None = None,
    volumes_filename: str | None = None,
    postproc_params: dict[str, Any] | None = None,
) -> None:
    """Copy metadata from input zarr to output zarr, including source mtime and label mapping info."""
    output_zarr = zarr.open(output_path, mode="r+")

    # Copy all attributes from input
    for key, value in input_zarr.attrs.items():
        output_zarr.attrs[key] = value

    # Add source file mtime if we can determine the input path
    if input_path is not None:
        try:
            input_mtime = os.path.getmtime(input_path)
            output_zarr.attrs["source_mtime"] = input_mtime
            output_zarr.attrs["source_path"] = str(input_path)
        except OSError:
            pass  # Can't get mtime, skip

    # Add processing metadata
    output_zarr.attrs["postproc_version"] = "distributed_postproc_v1"

    # Add postproc parameters
    if postproc_params is not None:
        output_zarr.attrs["postproc_params"] = postproc_params

    # Add label mapping metadata if provided
    if mapping_filename is not None and nblocks is not None:
        output_zarr.attrs["label_mapping"] = {
            "file": mapping_filename,
            "format": "sorted_global_and_final_rows",
            "label_bits": GLOBAL_LABEL_BITS,
            "nblocks": list(nblocks),
            "rows": ["global_id", "final_label"],
            "decode_global_id": (
                "local = gid & 0xFFFF; block_token = gid >> 16; "
                "block_idx = np.unravel_index(block_token, nblocks)"
            ),
        }

    if volumes_filename is not None:
        output_zarr.attrs["label_volumes"] = {
            "file": volumes_filename,
            "format": "npy_uint64_indexed_by_final_label",
            "units": "voxels",
            "background_index": 0,
        }


def _run_distributed_postproc(
    input_zarr: zarr.Array,
    write_path: Path | str,
    temporary_directory: Path,
    blocksize: tuple[int, int, int] | None = None,
    margin: int = 30,
    sigma: float | tuple[float, float, float] = (1.5, 3, 3),
    V_min: int = 1000,
    bg_scale: float | None = None,
    max_expansion: int = 1,
    min_contact_fraction: float = 0.0,
    input_path: Path | None = None,
    cluster: myLocalCluster | myGPUCluster | None = None,
) -> zarr.Array:
    """
    Distributed post-processing of 3D segmentation masks.

    Applies Gaussian smoothing, connected component relabeling, and small cell
    donation in a tiled manner with overlap, then stitches results.

    Parameters
    ----------
    input_zarr : zarr.Array
        Input segmentation masks (3D integer array)
    write_path : Path or str
        Output path for final post-processed zarr
    blocksize : tuple of int, optional
        ZYX core block size for tiled processing. If None, inherits the input
        segmentation chunks so postprocessing uses the segmentation grid.
    margin : int
        Margin parameter (in voxels) used to derive the spatial overlap
        between blocks (default 30). The actual overlap passed to
        `get_block_crops` and `remove_overlaps` is `overlap = 2*margin`.
    sigma : float or tuple
        Gaussian smoothing sigma (default (1.5, 3, 3) for ZYX)
    V_min : int
        Minimum volume threshold for small cell donation (default 2000)
    bg_scale : float, optional
        Background scale factor for Gaussian voting. If None (default),
        computed from sigma with fwhm_fraction=-0.1 for slight dilation.
    max_expansion : int
        Maximum expansion for Gaussian smooth (default 1)
    min_contact_fraction : float
        Minimum contact fraction for donation (default 0.0)
    input_path : Path, optional
        Path to input zarr for metadata copying (source mtime)
    cluster : cluster object, optional
        Existing Dask cluster to use
    temporary_directory : Path
        Isolated directory for temporary files.

    Returns
    -------
    zarr.Array
        Post-processed segmentation masks
    """
    write_path = Path(write_path)

    if input_zarr.ndim != 3:
        raise ValueError("distributed_postproc expects a 3D ZYX zarr array.")

    blocksize = _resolve_blocksize(input_zarr, blocksize)
    overlap = 2 * margin
    nblocks = _validate_tiling(input_zarr.shape, blocksize, overlap)

    logger.info(
        f"Starting distributed postproc: blocksize={blocksize}, "
        f"overlap={overlap}, margin={margin}, sigma={sigma}, V_min={V_min}"
    )

    temporary_directory = Path(temporary_directory)

    # Get block indices and crops
    block_indices, block_crops = get_block_crops(input_zarr.shape, np.array(blocksize), overlap, mask=None)

    logger.info(f"Processing {len(block_indices)} blocks")

    # Create temp zarr for unstitched output
    temp_zarr_path = temporary_directory / "postproc_unstitched.zarr"
    temp_zarr = create_zarr_array(
        temp_zarr_path,
        shape=tuple(int(s) for s in input_zarr.shape),
        chunks=blocksize,
        dtype=np.uint32,
        overwrite=True,
        codecs=label_zarr_codecs(np.uint32),
    )

    # Prepare postproc kwargs
    postproc_kwargs: dict[str, Any] = {
        "sigma": sigma,
        "V_min": V_min,
        "max_expansion": max_expansion,
        "min_contact_fraction": min_contact_fraction,
    }
    # Only include bg_scale if explicitly provided; otherwise computed from sigma
    if bg_scale is not None:
        postproc_kwargs["bg_scale"] = bg_scale
    metadata_params = {
        **postproc_kwargs,
        "margin": margin,
        "overlap": overlap,
        "blocksize": list(blocksize),
    }

    # Shuffle block order for better load balancing across workers
    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(block_indices))
    block_indices = [block_indices[i] for i in shuffle_idx]
    block_crops = [block_crops[i] for i in shuffle_idx]

    # Map over blocks
    assert cluster is not None
    t_submit = time.perf_counter()
    block_futures = cluster.client.map(
        process_postproc_block,
        block_indices,
        block_crops,
        input_zarr=input_zarr,
        output_zarr=temp_zarr,
        blocksize=blocksize,
        overlap=overlap,
        nblocks=nblocks,
        postproc_kwargs=postproc_kwargs,
    )
    logger.debug(f"[timing] submit: {time.perf_counter() - t_submit:.2f}s")

    # Extract faces on their producing workers, then transfer only unique label
    # contacts to the driver. Releasing block_futures lets Dask discard each
    # full face result as soon as its dependent extraction tasks finish.
    future_lookup = dict(zip(block_indices, block_futures, strict=True))
    box_stats_futures = [
        cluster.client.submit(_postproc_box_stats, future)
        for future in block_futures
    ]
    pair_futures = []
    for block_index, block_future in future_lookup.items():
        for axis in range(3):
            neighbor_index = list(block_index)
            neighbor_index[axis] += 1
            neighbor_future = future_lookup.get(tuple(neighbor_index))
            if neighbor_future is None:
                continue
            left_face = cluster.client.submit(
                _postproc_face,
                block_future,
                2 * axis + 1,
                axis,
            )
            right_face = cluster.client.submit(
                _postproc_face,
                neighbor_future,
                2 * axis,
                axis,
            )
            pair_futures.append(
                cluster.client.submit(_boundary_label_contacts, left_face, right_face)
            )
            del left_face, right_face

    t_gather = time.perf_counter()
    with progress_bar(len(block_indices)) as submit:
        for future in block_futures:
            future.add_done_callback(submit)
        del future_lookup, block_futures
        compact_results = cluster.client.gather(box_stats_futures + pair_futures)
    gather_time = time.perf_counter() - t_gather
    logger.debug(f"[timing] gather: {gather_time:.2f}s")

    n_box_results = len(box_stats_futures)
    box_stats = [stats for stats in compact_results[:n_box_results] if stats[0].size]
    box_ids_list = [ids for ids, _ in box_stats]
    box_counts_list = [counts for _, counts in box_stats]
    boundary_results = compact_results[n_box_results:]
    label_pairs = [result[0] for result in boundary_results if result[0].size]
    raw_pairs_list = [result[1] for result in boundary_results if result[1].size]
    raw_counts_list = [result[2] for result in boundary_results if result[1].size]
    del box_stats_futures, pair_futures, compact_results, box_stats, boundary_results

    total_pair_bytes = sum(pairs.nbytes for pairs in label_pairs)
    logger.debug(f"[timing] boundary_pairs: {total_pair_bytes / 1e6:.1f} MB")

    if not box_ids_list:
        logger.warning("No labels found in any block")
        # Just copy temp to output
        out = create_zarr_array(
            write_path,
            shape=tuple(int(s) for s in temp_zarr.shape),
            chunks=tuple(int(c) for c in temp_zarr.chunks),
            dtype=np.uint32,
            overwrite=True,
            codecs=label_zarr_codecs(np.uint32),
        )
        write_dask_to_zarr(dask.array.from_zarr(temp_zarr), out)
        volumes_filename = "volumes.npy"
        np.save(
            write_path / volumes_filename,
            np.asarray([np.prod(input_zarr.shape)], dtype=np.uint64),
        )
        _copy_zarr_metadata(
            input_zarr,
            write_path,
            input_path=input_path,
            volumes_filename=volumes_filename,
            postproc_params=metadata_params,
        )
        return zarr.open(write_path, mode="r")

    temporary_mapping_path = temporary_directory / "label_mapping.npy"
    used_labels = np.concatenate(box_ids_list).astype(np.uint32, copy=False)
    core_counts = np.concatenate(box_counts_list).astype(np.uint64, copy=False)
    mapping = determine_sparse_merge_relabeling(used_labels, label_pairs)
    final_volumes = _aggregate_final_label_volumes(used_labels, core_counts, mapping)
    if raw_pairs_list:
        raw_pairs = np.concatenate(raw_pairs_list, axis=1)
        raw_counts = np.concatenate(raw_counts_list)
        recovery_pairs = _select_small_boundary_merges(
            mapping,
            final_volumes,
            raw_pairs,
            raw_counts,
            V_min=V_min,
        )
        if recovery_pairs.size:
            label_pairs.append(recovery_pairs)
            mapping = determine_sparse_merge_relabeling(used_labels, label_pairs)
            final_volumes = _aggregate_final_label_volumes(used_labels, core_counts, mapping)
            logger.info(
                f"Recovered {recovery_pairs.shape[1]} small seam components "
                "through strongest raw boundary overlap"
            )
        del raw_pairs, raw_counts, recovery_pairs
    remaining_small = int(np.count_nonzero((final_volumes > 0) & (final_volumes < V_min)))
    if remaining_small:
        mapping = _drop_small_final_labels(mapping, final_volumes, V_min=V_min)
        final_volumes = _aggregate_final_label_volumes(used_labels, core_counts, mapping)
        logger.info(f"Dropped {remaining_small} small seam components without a large recipient")
    final_volumes[0] += np.uint64(int(np.prod(input_zarr.shape)) - int(core_counts.sum()))
    np.save(temporary_mapping_path, mapping)
    sparse_relabel_and_write(
        temp_zarr,
        temporary_mapping_path,
        write_path,
    )
    logger.info(
        f"Relabeled {mapping.shape[1]} global IDs to "
        f"{int(mapping[1].max())} final labels"
    )
    del (
        label_pairs,
        raw_pairs_list,
        raw_counts_list,
        box_ids_list,
        box_counts_list,
        used_labels,
        core_counts,
        mapping,
    )

    # Copy label mapping to sidecar file before temp dir cleanup
    mapping_filename = "label_mapping.npy"
    mapping_path = write_path / mapping_filename
    shutil.copy(temporary_mapping_path, mapping_path)
    logger.info(f"Saved label mapping to {mapping_path}")

    volumes_filename = "volumes.npy"
    volumes_path = write_path / volumes_filename
    np.save(volumes_path, final_volumes)
    logger.info(f"Saved final label volumes to {volumes_path}")
    del final_volumes

    logger.info(f"Post-processing complete. Output saved to {write_path}")

    _copy_zarr_metadata(
        input_zarr,
        write_path,
        input_path=input_path,
        nblocks=tuple(nblocks.tolist()),
        mapping_filename=mapping_filename,
        volumes_filename=volumes_filename,
        postproc_params=metadata_params,
    )

    return zarr.open_array(write_path, mode="r")


def _publish_postproc_output(staged: Path, output: Path, *, overwrite: bool) -> None:
    """Expose a complete staged output while retaining the prior output on failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        os.replace(staged, output)
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    backup = staged.parent / "prior.zarr"
    os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def _local_store_path(array: zarr.Array) -> Path | None:
    root = getattr(getattr(array, "store", None), "root", None)
    if isinstance(root, (str, os.PathLike)):
        return Path(root)
    return None


@cluster
def distributed_postproc(
    input_zarr: zarr.Array,
    write_path: Path | str,
    blocksize: tuple[int, int, int] | None = None,
    margin: int = 30,
    sigma: float | tuple[float, float, float] = (1.5, 3, 3),
    V_min: int = 1000,
    bg_scale: float | None = None,
    max_expansion: int = 1,
    min_contact_fraction: float = 0.0,
    input_path: Path | None = None,
    cluster: myLocalCluster | myGPUCluster | None = None,
    cluster_kwargs: dict[str, Any] | None = None,
    temporary_directory: Path | None = None,
    overwrite: bool = False,
) -> zarr.Array:
    """Run postprocessing in an isolated workspace and publish one complete Zarr."""
    output = Path(write_path)
    source = input_path if input_path is not None else _local_store_path(input_zarr)
    if source is not None and source.resolve() == output.resolve():
        raise ValueError("Postprocessing input and output paths must differ.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    workspace_parent = output.parent if temporary_directory is None else Path(temporary_directory)
    workspace_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pp-", dir=workspace_parent) as workspace_name:
        workspace = Path(workspace_name)
        staged = workspace / "result.zarr"
        _run_distributed_postproc(
            input_zarr=input_zarr,
            write_path=staged,
            blocksize=blocksize,
            margin=margin,
            sigma=sigma,
            V_min=V_min,
            bg_scale=bg_scale,
            max_expansion=max_expansion,
            min_contact_fraction=min_contact_fraction,
            input_path=input_path,
            cluster=cluster,
            temporary_directory=workspace,
        )
        _publish_postproc_output(staged, output, overwrite=overwrite)

    return zarr.open_array(output, mode="r")


@click.group()
def cli() -> None:
    """Distributed 3D post-processing for segmentation masks."""


@cli.command("run")
@click.argument(
    "input_zarr_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--output-path", default=None, type=click.Path(path_type=Path), help="Output path.")
@click.option(
    "--blocksize",
    default=None,
    type=(int, int, int),
    metavar="Z Y X",
    help="Optional ZYX core block size; defaults to input chunks.",
)
@click.option(
    "--sigma",
    default="1,2,2",
    show_default=True,
    type=str,
    help="Gaussian smoothing sigma; scalar or 'z,y,x' triple.",
)
@click.option("--v-min", default=500, show_default=True, type=int, help="Minimum volume threshold for small cell donation.")
@click.option(
    "--margin",
    default=30,
    show_default=True,
    type=int,
    help="Margin parameter (overlap = 2*margin for overlap removal).",
)
@click.option("--workers-per-gpu", default=1, show_default=True, type=int, help="Workers per GPU.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite existing output.")
def run(
    input_zarr_path: Path,
    output_path: Path | None,
    blocksize: tuple[int, int, int] | None,
    sigma: str,
    v_min: int,
    margin: int,
    workers_per_gpu: int,
    overwrite: bool,
) -> None:
    """Post-process one 3D segmentation .zarr input."""
    input_zarr = zarr.open(input_zarr_path, mode="r")

    resolved_output_path = output_path
    if resolved_output_path is None:
        sigma_str = sigma.replace(",", "-").replace(" ", "")
        resolved_output_path = input_zarr_path.parent / f"{input_zarr_path.stem}_postproc_s{sigma_str}_v{v_min}.zarr"

    distributed_postproc(
        input_zarr=input_zarr,
        write_path=resolved_output_path,
        blocksize=blocksize,
        margin=margin,
        sigma=_parse_sigma_option(sigma),
        V_min=v_min,
        input_path=input_zarr_path,
        cluster_kwargs={"workers_per_gpu": workers_per_gpu, "threads_per_worker": 1},
        overwrite=overwrite,
    )


if __name__ == "__main__":
    cp.cuda.set_allocator(None)
    cli()
