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
   - Each chunk is read with XY overlap by default (configurable via `margin` → `overlap = 2*margin`)
   - Chunks always span the full Z extent (no chunking in Z)
   - After processing, overlaps are removed using `remove_overlaps`, matching the stitching behavior
     of `distributed_segmentation`
   - The overlap must be large enough that the trimmed core region has correct context from all directions

3. **Reuse stitching from distributed_segmentation**
   After processing, chunks have locally-processed labels that need stitching:
   - global_segment_ids() encodes block index into bit-packed label IDs (avoids collisions)
   - block_faces() extracts boundary faces
   - adjacent_faces() pairs faces between neighboring blocks
   - block_face_adjacency_graph() builds a compact adjacency graph over only used labels
   - scipy.sparse.csgraph.connected_components() determines merges (union-find) in this compact space
   - Final relabeling LUT is applied over global IDs via dask.array.map_blocks

4. **Overlap sizing for Gaussian smoothing**
   Gaussian smoothing with per-axis sigma has effective radius ~4*max(sigma) voxels.
   With default sigma=(1,2,2), the radius is ~8 voxels. With 10% overlap on
   2048×2048 XY blocks (~205px) and 50px margin crop, the core region still sits
   well inside the chunk boundary, comfortably beyond 8px. This ensures Gaussian
   voting at the core boundary has essentially identical context from neighboring
   chunks.

5. **Small cell donation at boundaries**
   Small cells near chunk boundaries might have their best neighbor in another
   chunk. With sufficient overlap (>100px), small cells in the core region
   have their full neighborhood visible, so donation decisions are correct.

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
Stitch (reuse from distributed_segmentation):
   - adjacent_faces -> block_face_adjacency_graph
   - connected_components (union-find)
   - Apply remap via dask.array.map_blocks
        |
        v
Output zarr (post-processed masks)
```

Usage
-----
CLI:
    python -m fishtools.segmentation.distributed.distributed_postproc \\
        run /path/to/segmentation.zarr \\
        --blocksize 512 --sigma \"1,2,2\" --v-min 8000

Programmatic:
    from fishtools.segmentation.distributed.distributed_postproc import distributed_postproc
    result = distributed_postproc(input_zarr, write_path, sigma=(1, 2, 2), V_min=8000, ...)
"""

import os
import shutil
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

from fishtools.segment.postproc3d import (  # noqa: F401
    absorb_encircled_rois,
    compute_metadata_and_adjacency,
    donate_small_cells,
    gaussian_erosion_to_margin_and_scale,
    gaussian_smooth_labels_cupy,
    relabel_connected_components,
)
from fishtools.segmentation.distributed.gpu_cluster import cluster, myGPUCluster, myLocalCluster
from fishtools.segmentation.distributed.merge_utils import (
    block_faces,
    get_block_crops,
    get_nblocks,
    global_segment_ids,
    remove_overlaps,
    stitch_labels,
    create_zarr_array,
    label_zarr_codecs,
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


def process_postproc_block(
    block_index: tuple[int, ...],
    crop: tuple[slice, ...],
    input_zarr: zarr.Array,
    output_zarr: zarr.Array,
    blocksize: tuple[int, ...],
    overlap: int,
    nblocks: NDArray[np.int_],
    postproc_kwargs: dict[str, Any],
) -> tuple[list[NDArray[Any]], NDArray[np.uint32]]:
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
    tuple[list[NDArray], list[tuple[slice, ...]], NDArray[np.uint32]]
        (faces, boxes, box_ids) for stitching
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
        from fishtools.segment.postproc3d import gaussian_smooth_labels

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

    # 6. Extract faces for stitching
    # Pre-shrink faces here to parallelize the expensive distance_transform_edt
    # calls across workers (instead of doing it on the driver).
    # Note: shrink_labels already returns independent copies (not views), so no
    # additional .copy() is needed when shrink=True.
    t_faces_start = time.perf_counter()
    faces = block_faces(masks_global, shrink=True)
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
    return faces, box_ids


def _copy_zarr_metadata(
    input_zarr: zarr.Array,
    output_path: Path,
    input_path: Path | None = None,
    nblocks: tuple[int, ...] | None = None,
    mapping_filename: str | None = None,
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
            "label_bits": 16,
            "nblocks": list(nblocks),
            "decode_global_id": "local = gid & 0xFFFF; block_token = gid >> 16; block_idx = np.unravel_index(block_token, nblocks)",
        }


@cluster
def distributed_postproc(
    input_zarr: zarr.Array,
    write_path: Path | str,
    blocksize: tuple[int, ...] | None = None,
    margin: int = 100,
    sigma: float | tuple[float, float, float] = (1.5, 3, 3),
    V_min: int = 1000,
    bg_scale: float | None = None,
    max_expansion: int = 1,
    min_contact_fraction: float = 0.0,
    input_path: Path | None = None,
    cluster: myLocalCluster | myGPUCluster | None = None,
    cluster_kwargs: dict[str, Any] | None = None,
    temporary_directory: Path | None = None,
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
        ZYX block size for tiled processing. If None, uses
        (Z_full, 2048, 2048) clipped to the volume shape. Z is
        never chunked; chunks always span the full Z extent.
    margin : int
        Margin parameter (in voxels) used to derive the spatial overlap
        between blocks (default 100). The actual overlap passed to
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
    cluster_kwargs : dict, optional
        Arguments for cluster creation if cluster not provided
    temporary_directory : Path, optional
        Directory for temporary files

    Returns
    -------
    zarr.Array
        Post-processed segmentation masks
    """
    write_path = Path(write_path)

    if input_zarr.ndim != 3:
        raise ValueError("distributed_postproc expects a 3D ZYX zarr array.")

    z_dim, y_dim, x_dim = map(int, input_zarr.shape)

    if blocksize is None:
        y_block = min(2048, y_dim)
        x_block = min(2048, x_dim)
    else:
        if len(blocksize) != 3:
            raise ValueError("blocksize must be a 3-tuple (Z, Y, X).")
        _, y_block_raw, x_block_raw = blocksize
        y_block = min(int(y_block_raw), y_dim)
        x_block = min(int(x_block_raw), x_dim)

    blocksize = (z_dim, y_block, x_block)
    overlap = 2 * margin

    logger.info(
        f"Starting distributed postproc: blocksize={blocksize}, "
        f"overlap={overlap}, margin={margin}, sigma={sigma}, V_min={V_min}"
    )

    # Set up temporary directory
    if temporary_directory is None:
        temporary_directory = write_path.parent / "postproc_temp"
    temporary_directory = Path(temporary_directory)
    temporary_directory.mkdir(parents=True, exist_ok=True)

    # Get block indices and crops
    block_indices, block_crops = get_block_crops(input_zarr.shape, np.array(blocksize), overlap, mask=None)
    nblocks = get_nblocks(input_zarr.shape, np.array(blocksize))

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

    # Shuffle block order for better load balancing across workers
    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(block_indices))
    block_indices = [block_indices[i] for i in shuffle_idx]
    block_crops = [block_crops[i] for i in shuffle_idx]

    # Map over blocks
    assert cluster is not None
    t_submit = time.perf_counter()
    futures = cluster.client.map(
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

    # Gather results
    t_gather = time.perf_counter()
    with progress_bar(len(block_indices)) as submit:
        [fut.add_done_callback(submit) for fut in futures]
        results = cluster.client.gather(futures)
    gather_time = time.perf_counter() - t_gather
    logger.debug(f"[timing] gather: {gather_time:.2f}s")

    del futures

    # Filter to non-empty blocks only (must keep faces aligned with block_indices)
    faces_list, box_ids_list, non_empty_indices = [], [], []
    for i, (faces, box_ids) in enumerate(results):
        if len(box_ids) > 0:
            faces_list.append(faces)
            box_ids_list.append(box_ids)
            non_empty_indices.append(block_indices[i])

    # Calculate face data size for profiling
    total_face_bytes = sum(sum(f.nbytes for f in faces) for faces in faces_list)
    logger.debug(
        f"[timing] face_data: {total_face_bytes / 1e6:.1f} MB "
        f"({total_face_bytes / 1e6 / gather_time:.1f} MB/s)"
    )

    if len(box_ids_list) == 0:
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
        dask.array.to_zarr(dask.array.from_zarr(temp_zarr), out, overwrite=False)
        _copy_zarr_metadata(
            input_zarr, write_path, input_path=input_path, postproc_params={**postproc_kwargs, "margin": margin}
        )
        return zarr.open(write_path, mode="r")

    new_labeling_path = temporary_directory / "new_labeling.npy"
    final_seg_zarr, new_labeling = stitch_labels(
        block_indices=non_empty_indices,
        faces_list=faces_list,
        box_ids_list=box_ids_list,
        temp_zarr=temp_zarr,
        write_path=write_path,
        lut_path=new_labeling_path,
        pre_shrunk=True,
    )

    # Free memory no longer needed on the driver
    del results, faces_list, box_ids_list

    # Copy label mapping to sidecar file before temp dir cleanup
    mapping_filename = f"{write_path.stem}_label_mapping.npy"
    mapping_path = write_path.parent / mapping_filename
    shutil.copy(new_labeling_path, mapping_path)
    logger.info(f"Saved label mapping to {mapping_path}")

    # Clean up temporary directory
    shutil.rmtree(temporary_directory, ignore_errors=True)

    logger.info(f"Post-processing complete. Output saved to {write_path}")

    _copy_zarr_metadata(
        input_zarr,
        write_path,
        input_path=input_path,
        nblocks=tuple(nblocks.tolist()),
        mapping_filename=mapping_filename,
        postproc_params={**postproc_kwargs, "margin": margin},
    )


@click.group()
def cli() -> None:
    """Distributed 3D post-processing for segmentation masks."""


@cli.command("run")
@click.argument(
    "input_zarr_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--output-path", default=None, type=click.Path(path_type=Path), help="Output path.")
@click.option("--blocksize", default=1024, show_default=True, type=int, help="XY block size for tiled processing.")
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
    default=50,
    show_default=True,
    type=int,
    help="Margin parameter (overlap = 2*margin for overlap removal).",
)
@click.option("--workers-per-gpu", default=4, show_default=True, type=int, help="Workers per GPU.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite existing output.")
def run(
    input_zarr_path: Path,
    output_path: Path | None,
    blocksize: int,
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

    if resolved_output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {resolved_output_path}")

    distributed_postproc(
        input_zarr=input_zarr,
        write_path=resolved_output_path,
        blocksize=(input_zarr.shape[0], blocksize, blocksize),
        margin=margin,
        sigma=_parse_sigma_option(sigma),
        V_min=v_min,
        input_path=input_zarr_path,
        cluster_kwargs={"workers_per_gpu": workers_per_gpu, "threads_per_worker": 1},
    )


if __name__ == "__main__":
    cp.cuda.set_allocator(None)
    cli()
