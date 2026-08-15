"""
Shared merging utilities for distributed segmentation pipelines.

Provides:
- Block overlap removal and face extraction
- Face adjacency graph construction for stitching (in compact label space)
- Label merging via union-find (connected components) with a global-ID LUT
- Bounding box merging
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import dask.array
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import zarr
from numpy.typing import NDArray



def _zarr_codecs(dtype: np.dtype | type[np.generic] | str, *, label: bool = False) -> list[Any]:
    typesize = np.dtype(dtype).itemsize
    shuffle = zarr.codecs.BloscShuffle.bitshuffle if label or typesize == 1 else zarr.codecs.BloscShuffle.shuffle
    return [
        zarr.codecs.BytesCodec(),
        zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=7 if label else 5,
            shuffle=shuffle,
            typesize=typesize,
        ),
    ]


def label_zarr_codecs(dtype: np.dtype | type[np.generic] | str) -> list[Any]:
    return _zarr_codecs(dtype, label=True)


def create_zarr_array(
    write_path: Path | str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype | type[np.generic] | str,
    overwrite: bool,
    codecs: list[Any] | None = None,
) -> zarr.Array:
    codecs = _zarr_codecs(dtype) if codecs is None else codecs
    return zarr.create_array(
        str(write_path),
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        serializer=codecs[0],
        compressors=tuple(codecs[1:]),
        overwrite=overwrite,
        config={"write_empty_chunks": False},
    )

logger = logging.getLogger(__name__)
GLOBAL_LABEL_BITS = 16


def determine_merge_relabeling(
    block_indices: list[tuple[int, ...]],
    faces: list[list[NDArray[Any]]],
    used_labels: NDArray[Any],
    pre_shrunk: bool = False,
) -> NDArray[np.uint32]:
    """
    Determine boundary segment mergers and return a relabeling lookup table.

    Labels are always compacted to contiguous range [0..N] since bit-packed
    global IDs create sparse ID spaces that would otherwise produce nonsensical
    component counts.

    Parameters
    ----------
    block_indices
        Block grid indices corresponding to ``faces``.
    faces
        Boundary faces per block (from ``block_faces``).
    used_labels
        1D array of labels present in the volume.
    pre_shrunk
        If True, faces were already shrunk on workers; skip shrinking when
        processing face pairs. This parallelizes expensive distance transforms.

    Returns
    -------
    new_labeling : ndarray[uint32]
        LUT mapping old global label IDs to new IDs. Background (0) is
        preserved as 0. The LUT is defined on the range
        ``[0 .. max(used_labels)]``.
    """
    used_labels = used_labels.astype(np.uint32)
    used_labels = used_labels[used_labels > 0]
    if used_labels.size == 0:
        return np.array([0], dtype=np.uint32)

    # Work in a compact label space for the adjacency graph to avoid
    # allocating components proportional to max(global_id). We compress
    # the set of actually-used global IDs to [1..N] (0 reserved for
    # background), build the graph there, then expand back to a LUT over
    # global IDs.
    unique_labels = np.unique(used_labels)

    t0 = time.perf_counter()
    faces_paired = adjacent_faces(block_indices, faces)
    logger.info(f"[timing] adjacent_faces: {time.perf_counter() - t0:.2f}s ({len(faces_paired)} face pairs)")

    label_groups = block_face_adjacency_graph(faces_paired, unique_labels, pre_shrunk=pre_shrunk)

    t0 = time.perf_counter()
    components = scipy.sparse.csgraph.connected_components(label_groups, directed=False)[1]
    # components has length N_labels + 1. Index 0 is the background node
    # (no edges); indices 1..N correspond to unique_labels[0..N-1].

    # Build LUT over the dense global-ID domain [0 .. max_label]. Unused
    # entries remain 0.
    max_label = int(unique_labels.max())
    lut = np.zeros(max_label + 1, dtype=np.uint32)

    # Map each global label -> component ID for its compact node index.
    # Node index for unique_labels[i] is i + 1.
    lut[unique_labels] = components[1 : unique_labels.size + 1].astype(np.uint32)

    # Ensure background stays 0 explicitly.
    lut[0] = 0

    return lut


def relabel_and_write(
    temp_zarr: zarr.Array,
    new_labeling_path: Path | str,
    write_path: Path | str,
) -> None:
    """
    Apply a relabeling LUT across a temporary zarr store and write output.

    Uses memory-mapped array access for efficiency - the OS handles page caching
    so each block only loads the portion of the LUT it needs.
    """
    write_path = Path(write_path)
    new_labeling_path = Path(new_labeling_path)

    def apply_lut_mmap(block: np.ndarray, lut_path: str) -> np.ndarray:
        """Apply LUT using memory-mapped access for efficiency."""
        lut = np.load(lut_path, mmap_mode="r")
        return lut[block].astype(np.uint32)

    segmentation_da = dask.array.from_zarr(temp_zarr)
    relabeled = dask.array.map_blocks(
        apply_lut_mmap,
        segmentation_da,
        lut_path=str(new_labeling_path),
        dtype=np.uint32,
        chunks=segmentation_da.chunks,
    )
    write_path.parent.mkdir(parents=True, exist_ok=True)
    out = create_zarr_array(
        write_path,
        shape=tuple(int(s) for s in relabeled.shape),
        chunks=tuple(int(c[0]) for c in relabeled.chunks),
        dtype=np.uint32,
        overwrite=True,
        codecs=label_zarr_codecs(np.uint32),
    )
    dask.array.to_zarr(relabeled, out, overwrite=False)


def get_block_crops(
    shape: tuple[int, ...],
    blocksize: np.ndarray,
    overlap: int,
    mask: NDArray[Any] | None,
) -> tuple[list[tuple[int, ...]], list[tuple[slice, ...]]]:
    """Given a voxel grid shape, blocksize, and overlap size, construct slice tuples per block."""
    blocksize = np.array(blocksize)
    if mask is not None:
        ratio = np.array(mask.shape) / shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start = blocksize * index - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_crop = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_crop]):
                foreground = False
        if foreground:
            indices.append(index)
            crops.append(crop)
    return indices, crops


def get_nblocks(shape: tuple[int, ...], blocksize: np.ndarray) -> NDArray[np.int_]:
    """Number of blocks along each axis."""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def block_faces(
    segmentation: NDArray[Any], *, shrink: bool = False, shrink_threshold: float = 1.0
) -> list[NDArray[Any]]:
    """Extract start/end faces along each axis for a segmented block.

    Parameters
    ----------
    segmentation
        The segmented block array.
    shrink
        If True, apply shrink_labels to each face to erode label boundaries.
        This moves expensive distance transform computation to workers.
    shrink_threshold
        Distance threshold for shrinking (only used if shrink=True).
    """
    faces = []
    for iii in range(segmentation.ndim):
        a = [slice(None)] * segmentation.ndim
        a[iii] = slice(0, 1)
        face = segmentation[tuple(a)]
        if shrink:
            face = shrink_labels(face, shrink_threshold)
        faces.append(face)
        a = [slice(None)] * segmentation.ndim
        a[iii] = slice(-1, None)
        face = segmentation[tuple(a)]
        if shrink:
            face = shrink_labels(face, shrink_threshold)
        faces.append(face)
    return faces


def bounding_boxes_in_global_coordinates(
    segmentation: NDArray[Any], crop: tuple[slice, ...]
) -> list[tuple[slice, ...]]:
    """Compute bounding boxes for all segments in global coordinates."""
    import cupy as cp

    try:
        from cupyx.scipy import ndimage as cpx_ndimage
    except ImportError:
        raise ImportError("CuPy >=14.0.0a1 needed for GPU-accelerated bounding box computation.")

    seg_gpu = cp.asarray(segmentation)
    boxes_local = cpx_ndimage.find_objects(seg_gpu)
    # filter out labels that are not present in this block
    boxes_local = [b for b in boxes_local if b is not None]
    translate = lambda a, b: slice(a.start + b.start, a.start + b.stop)
    return [tuple(translate(a, b) for a, b in zip(crop, box)) for box in boxes_local]


def global_segment_ids(
    segmentation: NDArray[Any],
    block_index: tuple[int, ...],
    nblocks: NDArray[np.int_],
    label_bits: int = GLOBAL_LABEL_BITS,
) -> tuple[NDArray[np.uint32], list[np.uint32]]:
    """
    Generate globally unique segment IDs using bit-packing.

    Encodes block index in upper bits and local label in lower bits to create
    globally unique IDs that fit within uint32 without overflow.

    Parameters
    ----------
    segmentation : ndarray
        Local segmentation mask with integer labels.
    block_index : tuple of int
        The (z, y, x) index of this block in the block grid.
    nblocks : ndarray
        Number of blocks along each axis.
    label_bits : int
        Number of bits reserved for local labels (default 16 = up to 65535 labels).
        Remaining bits (32 - label_bits) are used for block index. Increasing
        ``label_bits`` allows more distinct labels per block but fewer total
        blocks in the volume; decreasing it does the opposite. In typical
        tilings with 2048×2048 XY blocks and modest label densities, the
        default is a good compromise.

    Returns
    -------
    segmentation_global : ndarray[uint32]
        Segmentation with globally unique IDs.
    remap : list[uint32]
        Mapping from local indices to global IDs.

    Raises
    ------
    ValueError
        If block index or label count exceeds the bit allocation.
    """
    block_token = int(np.ravel_multi_index(block_index, tuple(nblocks.tolist())))
    max_blocks = 1 << (32 - label_bits)
    max_labels = 1 << label_bits

    if block_token >= max_blocks:
        raise ValueError(
            f"Block index {block_token} exceeds max {max_blocks - 1} with {label_bits} label bits. "
            f"Consider reducing label_bits or using fewer/larger blocks."
        )

    # Assume labels are sequential 0..max_label (from prior relabeling in postproc).
    # This avoids O(N log N) np.unique on 51M elements, making this O(N).
    max_label = int(segmentation.max())
    if max_label >= max_labels:
        raise ValueError(
            f"Label count {max_label} exceeds max {max_labels} with {label_bits} label bits. "
            f"Consider increasing label_bits or using smaller blocks."
        )

    # Pack: upper bits = block index, lower bits = label value directly
    # Since labels are sequential, we can index directly without unique_inverse
    remap = np.arange(max_label + 1, dtype=np.uint32) | np.uint32(block_token << label_bits)
    remap[0] = 0  # Background stays 0

    # Direct indexing - O(N) instead of O(N log N)
    segmentation_global = remap[segmentation.ravel()].reshape(segmentation.shape)
    return segmentation_global, remap


def decode_block_global_labels(
    segmentation: NDArray[Any],
    *,
    label_bits: int = GLOBAL_LABEL_BITS,
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Decode one temporary block into bounded local labels and global IDs.

    Temporary segmentation blocks contain one bit-packed block token in their
    upper bits and the original Cellpose label in their lower bits. Decoding
    the lower bits keeps bounding-box allocation proportional to local labels
    instead of the largest global ID.
    """
    if not np.issubdtype(segmentation.dtype, np.integer):
        raise TypeError(f"Expected integer segmentation labels, got {segmentation.dtype}.")
    if label_bits <= 0 or label_bits >= 32:
        raise ValueError("label_bits must be between 1 and 31.")

    if np.issubdtype(segmentation.dtype, np.signedinteger) and int(segmentation.min()) < 0:
        raise ValueError("Segmentation labels cannot be negative.")
    max_global_id = int(segmentation.max())
    if max_global_id > np.iinfo(np.uint32).max:
        raise ValueError("Segmentation labels exceed the uint32 encoding range.")

    encoded = segmentation.astype(np.uint32, copy=False)
    label_mask = np.uint32((1 << label_bits) - 1)
    local_labels = np.bitwise_and(encoded, label_mask)
    if max_global_id == 0:
        return local_labels, np.empty(0, dtype=np.uint32)

    min_global_id = int(
        np.min(encoded, where=encoded != 0, initial=np.iinfo(np.uint32).max)
    )
    min_token = min_global_id >> label_bits
    max_token = max_global_id >> label_bits
    if min_token != max_token:
        raise ValueError("A temporary segmentation block contains labels from multiple block tokens.")

    present_local_ids = np.flatnonzero(np.bincount(local_labels.ravel()))
    present_local_ids = present_local_ids[present_local_ids != 0].astype(np.uint32, copy=False)
    global_ids = present_local_ids | np.uint32(min_token << label_bits)
    return local_labels, global_ids


def remove_overlaps(
    array: NDArray[Any],
    crop: tuple[slice, ...],
    overlap: int,
    blocksize: tuple[int, ...],
) -> tuple[NDArray[Any], list[slice]]:
    """
    Remove overlapping regions from segmented block.

    Overlapping regions are added during processing to provide context for boundary
    pixels/voxels during segmentation. After segmentation is complete, these overlaps
    must be removed to prevent double-counting during block stitching.
    """
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        if crop[axis].start != 0:
            slc = [slice(None)] * array.ndim
            slc[axis] = slice(overlap, None)
            array = array[tuple(slc)]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlap, b)
        if array.shape[axis] > blocksize[axis]:
            slc = [slice(None)] * array.ndim
            slc[axis] = slice(None, blocksize[axis])
            array = array[tuple(slc)]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed


def adjacent_faces(
    block_indices: list[tuple[int, ...]], faces: list[list[NDArray[Any]]]
) -> list[NDArray[Any]]:
    """Find faces which touch and pair them together in new data structure."""
    face_pairs = []
    faces_index_lookup = {a: b for a, b in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2 * ax + 1]
                b = faces_index_lookup[neighbor_index][2 * ax]
                face_pairs.append(np.concatenate((a, b), axis=ax))
            except KeyError:
                continue
    return face_pairs


def shrink_labels(plane: NDArray[Any], threshold: float) -> NDArray[Any]:
    """Shrink labels in plane by some distance from their boundary.

    This erosion helps ensure robust matching by avoiding edge artifacts
    where labels may be slightly misaligned at chunk boundaries.
    """
    squeezed = plane.squeeze()
    # Degenerate case: single voxel, nothing to shrink.
    if squeezed.ndim == 0:
        return plane.copy()

    gradmag = np.linalg.norm(np.gradient(squeezed), axis=0)
    shrunk_labels = np.copy(squeezed)
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def _find_label_pairs_across_boundary(face: NDArray[Any], structure: NDArray[Any]) -> NDArray[np.int64]:
    """
    Find pairs of labels that touch across a face boundary.

    Local implementation equivalent to dask_image.ndmeasure._utils._label._across_block_label_grouping
    to avoid private API dependency.

    Parameters
    ----------
    face : ndarray
        Face array with shape (..., 2, ...) where axis with size 2 represents
        the two sides of the boundary.
    structure : ndarray
        Binary structure for connectivity (unused, kept for API compatibility).

    Returns
    -------
    pairs : ndarray of shape (2, n_pairs)
        Array of (label_a, label_b) pairs that touch across the boundary.
    """
    # Find the axis with size 2 (the boundary axis)
    boundary_axis = None
    for i, s in enumerate(face.shape):
        if s == 2:
            boundary_axis = i
            break

    if boundary_axis is None:
        return np.empty((2, 0), dtype=np.int64)

    # Get slices for each side
    sl0 = tuple(slice(0, 1) if i == boundary_axis else slice(None) for i in range(face.ndim))
    sl1 = tuple(slice(1, 2) if i == boundary_axis else slice(None) for i in range(face.ndim))

    side0 = face[sl0].squeeze(axis=boundary_axis)
    side1 = face[sl1].squeeze(axis=boundary_axis)

    # Find where both sides have non-zero labels
    mask = (side0 > 0) & (side1 > 0)

    pairs_a = side0[mask].ravel()
    pairs_b = side1[mask].ravel()

    if len(pairs_a) == 0:
        return np.empty((2, 0), dtype=np.int64)

    # Return unique pairs (sorted to ensure consistency)
    pairs = np.vstack([pairs_a, pairs_b]).astype(np.int64)
    # Make pairs canonical (smaller label first) for deduplication
    pairs = np.sort(pairs, axis=0)
    pairs = np.unique(pairs, axis=1)
    return pairs


def _compress_label_pairs_to_compact_space(
    pairs: NDArray[np.int64],
    unique_labels: NDArray[np.uint32],
) -> NDArray[np.int64]:
    """
    Map global label ID pairs to a compact index space [1..N].

    Parameters
    ----------
    pairs
        Array of shape (2, n_pairs) with global label IDs (>0).
    unique_labels
        Sorted 1D array of unique global label IDs present anywhere in the volume.

    Returns
    -------
    compact_pairs : ndarray[int64]
        Same shape as ``pairs``; each label value is replaced by its compact
        index in [1..N]. Index 0 is reserved for background and never used.
    """
    if pairs.size == 0:
        return pairs

    flat = pairs.ravel()
    # Map global IDs -> compact indices via searchsorted; unique_labels is sorted.
    idx = np.searchsorted(unique_labels, flat)

    # Sanity check: every label appearing in faces must be present in unique_labels.
    if not np.array_equal(unique_labels[idx], flat):
        raise ValueError("Encountered face labels that are not present in used_labels.")

    compact_flat = (idx + 1).astype(np.int64)  # reserve 0 for background
    return compact_flat.reshape(pairs.shape)


def _process_single_face(
    face: NDArray[Any],
    structure: NDArray[Any],
    unique_labels: NDArray[np.uint32],
    pre_shrunk: bool = False,
) -> NDArray[np.int64] | None:
    """Process a single face pair to find which labels should merge.

    Algorithm:
    1. Split the 2-pixel-thick face slab into two 1-pixel slices (one from each block)
    2. Shrink labels via distance transform to handle boundary artifacts (unless pre_shrunk)
    3. Concatenate slices and find label pairs that touch across the boundary
    4. Map global label IDs to compact [1..N] space for sparse graph construction

    Parameters
    ----------
    face
        2-pixel-thick face slab (concatenation of adjacent block boundaries).
    structure
        Binary structure for connectivity analysis.
    unique_labels
        Sorted array of unique global label IDs.
    pre_shrunk
        If True, faces were already shrunk on workers; skip shrinking here.

    Returns (2, M) array of compact label pairs, or None if no pairs found.
    """
    # Split face into the two adjacent block boundaries
    sl0 = tuple(slice(0, 1) if d == 2 else slice(None) for d in face.shape)
    sl1 = tuple(slice(1, 2) if d == 2 else slice(None) for d in face.shape)

    if pre_shrunk:
        # Faces already shrunk on workers - just extract slices
        a = face[sl0]
        b = face[sl1]
    else:
        a = shrink_labels(face[sl0], 1.0)
        b = shrink_labels(face[sl1], 1.0)

    face_combined = np.concatenate((a, b), axis=np.argmin(a.shape))
    mapped_global = _find_label_pairs_across_boundary(face_combined, structure)

    if mapped_global.size == 0:
        return None

    mapped_compact = _compress_label_pairs_to_compact_space(mapped_global, unique_labels)
    return mapped_compact if mapped_compact.size > 0 else None


def block_face_adjacency_graph(
    faces: list[NDArray[Any]],
    unique_labels: NDArray[np.uint32],
    n_workers: int | None = None,
    pre_shrunk: bool = False,
) -> scipy.sparse.csr_matrix:
    """
    Build adjacency graph from face pairs in a compact label space.

    Finds which labels touch across face boundaries to build a sparse
    adjacency matrix for union-find.

    Parameters
    ----------
    faces
        List of paired faces across all blocks (output of ``adjacent_faces``).
    unique_labels
        Sorted 1D array of unique global label IDs (>0) present in the volume.
        These are mapped to node indices [1..N]; index 0 is reserved for
        background and never used.
    n_workers
        Number of parallel workers for face processing. Defaults to cpu_count - 4.
    pre_shrunk
        If True, faces were already shrunk on workers; skip shrinking in
        _process_single_face. This parallelizes expensive distance transforms.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 8) - 4)

    nlabels = int(unique_labels.size)

    if nlabels == 0:
        return scipy.sparse.csr_matrix((1, 1), dtype=np.int32)

    structure = scipy.ndimage.generate_binary_structure(3, 1)
    n_faces = len(faces)

    # Process faces in parallel
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(
            executor.map(lambda f: _process_single_face(f, structure, unique_labels, pre_shrunk), faces)
        )
    t_parallel = time.perf_counter() - t0

    # Filter out None results and collect mappings
    all_mappings = [r for r in results if r is not None]

    logger.info(f"block_face_adjacency_graph ({n_faces} faces, {n_workers} workers):")
    logger.info(
        f"  parallel processing: {t_parallel:.2f}s ({t_parallel / max(n_faces, 1) * 1000:.1f}ms/face)"
    )

    if not all_mappings:
        return scipy.sparse.csr_matrix((nlabels + 1, nlabels + 1), dtype=np.int32)

    t0 = time.perf_counter()
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    result = scipy.sparse.coo_matrix((v, (i, j)), shape=(nlabels + 1, nlabels + 1)).tocsr()
    logger.info(f"  graph_construction: {time.perf_counter() - t0:.2f}s")

    return result


def merge_boxes(boxes: NDArray[Any]) -> tuple[slice, ...]:
    """Take union of two or more parallelpipeds."""
    box_union = boxes[0]
    for iii in range(1, len(boxes)):
        local_union = []
        for s1, s2 in zip(box_union, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            local_union.append(slice(start, stop))
        box_union = tuple(local_union)
    return box_union


def merge_all_boxes(boxes: list[tuple[slice, ...]], box_ids: NDArray[Any]) -> list[tuple[slice, ...]]:
    """Merge all boxes that map to the same box_ids.

    Uses vectorized sorting + group boundaries instead of Python loop over unique IDs.
    Complexity: O(n log n) instead of O(n * k) where k = number of unique IDs.
    """
    if len(boxes) == 0:
        return []

    # Convert slices to numeric bounds: [start0, stop0, start1, stop1, ...]
    ndim = len(boxes[0])
    bounds = np.array([[s.start, s.stop] for box in boxes for s in box], dtype=np.int64).reshape(
        len(boxes), ndim * 2
    )
    box_ids_arr = np.asarray(box_ids, dtype=np.int64)

    # Sort by box_id
    order = np.argsort(box_ids_arr)
    sorted_ids = box_ids_arr[order]
    sorted_bounds = bounds[order]

    # Find group boundaries using diff
    breaks = np.concatenate([[0], np.nonzero(np.diff(sorted_ids))[0] + 1, [len(sorted_ids)]])

    merged = []
    for i in range(len(breaks) - 1):
        group = sorted_bounds[breaks[i] : breaks[i + 1]]
        # Min of starts (even columns), max of stops (odd columns)
        mins = group[:, 0::2].min(axis=0)
        maxs = group[:, 1::2].max(axis=0)
        merged.append(tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs)))

    return merged


def stitch_labels(
    block_indices: list[tuple[int, int, int]],
    faces_list: list[list[NDArray[Any]]],
    box_ids_list: list[NDArray[np.uint32]],
    temp_zarr: zarr.Array,
    write_path: Path | str,
    lut_path: Path | str,
    *,
    pre_shrunk: bool,
) -> tuple[zarr.Array, NDArray[np.uint32]]:
    """
    Shared driver-side stitching: merge labels across block boundaries and write output.

    Steps:
    - Concatenate label IDs
    - determine_merge_relabeling(...)
    - Save LUT
    - relabel_and_write(...)

    Returns:
        (final_zarr, new_labeling LUT)
    """
    if box_ids_list:
        all_box_ids = np.concatenate(box_ids_list).astype(np.uint32)
    else:
        all_box_ids = np.empty(0, dtype=np.uint32)

    new_labeling = determine_merge_relabeling(
        [(bi[0], bi[1], bi[2]) for bi in block_indices],
        faces_list,
        all_box_ids,
        pre_shrunk=pre_shrunk,
    )

    lut_path = Path(lut_path)
    np.save(lut_path, new_labeling)

    n_final_labels = int(new_labeling.max())
    logger.info(f"Relabeling to {n_final_labels} final labels (merged from {len(all_box_ids)} IDs)")

    relabel_and_write(temp_zarr, lut_path, write_path)

    return zarr.open(write_path, mode="r"), new_labeling


def merge_boxes_for_labels(
    boxes_list: list[list[tuple[slice, ...]]],
    box_ids_list: list[NDArray[np.uint32]],
    new_labeling: NDArray[np.uint32],
) -> list[tuple[slice, ...]]:
    """
    Merge bounding boxes according to the label remapping from stitching.

    Args:
        boxes_list: Per-block lists of bounding boxes
        box_ids_list: Per-block arrays of global label IDs
        new_labeling: LUT from stitch_labels mapping old IDs to merged IDs

    Returns:
        Merged bounding boxes (one per final label)
    """
    boxes = [box for sublist in boxes_list for box in sublist]
    if not box_ids_list:
        return []
    box_ids = np.concatenate(box_ids_list).astype(np.uint32)
    return merge_all_boxes(boxes, new_labeling[box_ids])
