"""
3D Post-Processing Pipeline for Cellpose Segmentation Masks.

This module implements a 4-phase pipeline to refine 3D segmentation masks produced
by Cellpose. The pipeline addresses common artifacts from voxel-wise flow-field
reconstruction: jagged boundaries, disconnected fragments, and small spurious cells.

Overview
--------
The pipeline processes masks through 4 sequential phases:

Phase 1: Gaussian Smooth Labels (`gaussian_smooth_labels`, `gaussian_smooth_labels_cupy`)
    Smooths jagged label boundaries using Gaussian-weighted voting. For each voxel,
    computes a score for each label as the Gaussian-blurred indicator function, then
    assigns the voxel to the label with highest score. This acts as a low-pass filter
    on label boundaries while preserving discrete identities.

    Key parameters:
    - sigma: Gaussian standard deviation in voxel units. Can be a scalar
      (isotropic smoothing) or a 3-tuple ``(sigma_z, sigma_y, sigma_x)``
      matching the ``(Z, Y, X)`` array axes. Features smaller than the
      corresponding sigma along each axis are smoothed away. The default
      is anisotropic ``(1.5, 3.0, 3.0)`` in ZYX order.
    - bg_scale: Multiplicative factor on background scores (values <1 favor expansion).
    - max_expansion: Limits how far labels can expand beyond original foreground.

    Mathematical basis: For a planar boundary, the Gaussian-blurred indicator gives
    a CDF profile. The threshold level determines boundary position, allowing
    controlled erosion/dilation via bg_scale.

Phase 2: Relabel Connected Components (`relabel_connected_components`)
    Assigns unique IDs to disconnected fragments. After Gaussian smoothing, a single
    original label may have become multiple disconnected pieces (e.g., a thin bridge
    was smoothed away). This phase ensures each piece gets a unique ID so Phase 4
    can handle them independently.

Phase 3: Compute Metadata (`compute_metadata_and_adjacency`)
    Computes per-label statistics needed for Phase 4:
    - volumes: Voxel count per label (via bincount)
    - adjacency: Which labels are 6-connected neighbors
    - contact_areas: Size of shared interface between adjacent labels (in voxel-faces)

    Uses sparse contact computation with O(B) memory where B = boundary faces,
    instead of O(K²) dense matrix. Scales to any number of labels.

Phase 4: Donate Small Cells (`donate_small_cells`)
    Removes tiny fragments by "donating" their voxels to the neighbor with largest
    contact area. Labels with volume < V_min are candidates. The donation is
    "upward only" - small labels donate to larger labels, preventing cycles.

    Key parameters:
    - V_min: Volume threshold. Labels with 0 < volume < V_min are removed.
    - min_contact_fraction: Optional minimum contact_area/volume ratio to donate.

Additional Utilities
--------------------
- `smooth_masks_3d`: Alternative iterative smoothing using morphological operations
- `opening_correct_protrusions`: Detects and reassigns thin protrusions via opening
- `gaussian_erosion_to_margin_and_scale`: Computes bg_scale from desired erosion distance

Typical Usage
-------------
```python
from fishtools.segment.postproc3d import (
    gaussian_smooth_labels_cupy,
    relabel_connected_components,
    compute_metadata_and_adjacency,
    donate_small_cells,
)

# Phase 1: Smooth boundaries
masks = gaussian_smooth_labels_cupy(masks, sigma=3.5, bg_scale=0.9)

# Phase 2: Relabel disconnected fragments
masks = relabel_connected_components(masks)

# Phase 3: Compute metadata
volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

# Phase 4: Remove small cells
masks = donate_small_cells(masks, volumes, adjacency, contact_areas, V_min=8000)
```

For large volumes, see `fishtools.segmentation.distributed.distributed_postproc`
which applies this pipeline in a chunked/tiled manner with Dask.

Performance Notes
-----------------
- `gaussian_smooth_labels_cupy` uses GPU acceleration via CuPy (~10-100x faster)
- Contact computation uses sparse edge-list aggregation: O(B) memory, O(B log B) time
- Memory usage is O(N) for streaming argmax in Gaussian smoothing
- `absorb_encircled_rois` uses O(K) memory per slice via min/max neighbor tracking
"""

import math
from collections.abc import Sequence

import numpy as np
from numba import njit, prange  # type: ignore[import]

try:  # Provided by kernprof/line_profiler when profiling
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - fallback when not profiling

    def profile(func):
        return func


def _normalize_sigma_zyx(sigma: float | Sequence[float]) -> tuple[float, float, float]:
    """
    Normalize a sigma specification to a 3-tuple in ZYX order.

    Parameters
    ----------
    sigma
        Either a scalar (isotropic smoothing) or a sequence of three
        values ``(sigma_z, sigma_y, sigma_x)`` matching the
        ``(Z, Y, X)`` axes of the masks.

    Returns
    -------
    tuple[float, float, float]
        Normalized per-axis sigmas in ZYX order.
    """
    if isinstance(sigma, (int, float)):
        v = float(sigma)
        return (v, v, v)

    if not isinstance(sigma, Sequence):
        raise TypeError("sigma must be a float or a sequence of three floats (sigma_z, sigma_y, sigma_x)")

    if len(sigma) != 3:
        raise ValueError("sigma sequence must have length 3 in ZYX order (sigma_z, sigma_y, sigma_x)")

    sz, sy, sx = (float(s) for s in sigma)
    return (sz, sy, sx)


@njit(cache=False, nogil=True, parallel=True)  # pragma: no cover - exercised via runtime profiling
def _contact_matrix_numba(masks: np.ndarray, max_label: int) -> np.ndarray:
    """
    Dense symmetric contact matrix using 6-connected (face) neighbors.

    Parameters
    ----------
    masks
        3D int32 label volume, background = 0.
    max_label
        Maximum label ID present in ``masks``.

    Returns
    -------
    contact : np.ndarray
        2D array where contact[k, neighbor_label] is the 6-connected interface size
        between labels k and neighbor_label (in voxel-face units).
    """
    Z, Y, X = masks.shape
    contact = np.zeros((max_label + 1, max_label + 1), np.int64)

    # Forward 6-connected directions
    for dz, dy, dx in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        zmax = Z - dz
        ymax = Y - dy
        xmax = X - dx
        for z in prange(zmax):
            for y in prange(ymax):
                for x in prange(xmax):
                    a = masks[z, y, x]
                    b = masks[z + dz, y + dy, x + dx]
                    if a != b and a > 0 and b > 0:
                        # Label-to-label contact
                        if a < b:
                            k = a
                            neighbor_label = b
                        else:
                            k = b
                            neighbor_label = a
                        contact[k, neighbor_label] += 1
                        contact[neighbor_label, k] += 1

    return contact


@njit(cache=False, nogil=True, parallel=True)  # pragma: no cover
def _count_boundary_faces(masks: np.ndarray) -> int:
    """Count total label-to-label boundary faces (6-connected)."""
    Z, Y, X = masks.shape
    count = 0

    # Single prange loop - count all directions per z slice
    for z in prange(Z):
        local = 0
        for y in range(Y):
            for x in range(X):
                a = masks[z, y, x]
                if a <= 0:
                    continue
                # X direction (forward only)
                if x + 1 < X:
                    b = masks[z, y, x + 1]
                    if b > 0 and b != a:
                        local += 1
                # Y direction (forward only)
                if y + 1 < Y:
                    b = masks[z, y + 1, x]
                    if b > 0 and b != a:
                        local += 1
                # Z direction (forward only)
                if z + 1 < Z:
                    b = masks[z + 1, y, x]
                    if b > 0 and b != a:
                        local += 1
        count += local

    return count


@njit(cache=False, nogil=True)  # pragma: no cover
def _collect_boundary_pairs(masks: np.ndarray, out_a: np.ndarray, out_b: np.ndarray) -> int:
    """
    Collect all label-to-label boundary pairs into pre-allocated arrays.

    Each boundary face is recorded once with (min(a,b), max(a,b)).
    Returns the number of pairs written.
    """
    Z, Y, X = masks.shape
    idx = 0

    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                a = masks[z, y, x]
                if a <= 0:
                    continue
                # X direction (forward only)
                if x + 1 < X:
                    b = masks[z, y, x + 1]
                    if b > 0 and b != a:
                        if a < b:
                            out_a[idx] = a
                            out_b[idx] = b
                        else:
                            out_a[idx] = b
                            out_b[idx] = a
                        idx += 1
                # Y direction (forward only)
                if y + 1 < Y:
                    b = masks[z, y + 1, x]
                    if b > 0 and b != a:
                        if a < b:
                            out_a[idx] = a
                            out_b[idx] = b
                        else:
                            out_a[idx] = b
                            out_b[idx] = a
                        idx += 1
                # Z direction (forward only)
                if z + 1 < Z:
                    b = masks[z + 1, y, x]
                    if b > 0 and b != a:
                        if a < b:
                            out_a[idx] = a
                            out_b[idx] = b
                        else:
                            out_a[idx] = b
                            out_b[idx] = a
                        idx += 1

    return idx


def _aggregate_boundary_pairs(
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    max_label: int,
) -> tuple[list[set[int]], list[dict[int, int]]]:
    """
    Convert edge list to adjacency sets and contact_areas dicts.

    Uses lexsort + run-length encoding for O(B log B) aggregation.
    """
    adjacency: list[set[int]] = [set() for _ in range(max_label + 1)]
    contact_areas: list[dict[int, int]] = [dict() for _ in range(max_label + 1)]

    if len(pairs_a) == 0:
        return adjacency, contact_areas

    # Sort by (a, b) - lexsort sorts by last key first, so (pairs_b, pairs_a)
    order = np.lexsort((pairs_b, pairs_a))
    sorted_a = pairs_a[order]
    sorted_b = pairs_b[order]

    # Find where pairs change using diff
    diff_a = np.diff(sorted_a)
    diff_b = np.diff(sorted_b)
    change_mask = (diff_a != 0) | (diff_b != 0)
    change_idx = np.nonzero(change_mask)[0] + 1

    # Run starts and lengths
    starts = np.concatenate([[0], change_idx])
    ends = np.concatenate([change_idx, [len(sorted_a)]])
    counts = ends - starts

    # Unique pairs
    unique_a = sorted_a[starts]
    unique_b = sorted_b[starts]

    # Build output structures
    for i in range(len(unique_a)):
        a, b, c = int(unique_a[i]), int(unique_b[i]), int(counts[i])
        adjacency[a].add(b)
        adjacency[b].add(a)
        contact_areas[a][b] = c
        contact_areas[b][a] = c

    return adjacency, contact_areas


@njit(cache=False)
def _compute_slice_encircled_inner(
    slice_2d,
    neighbor_min,
    neighbor_max,
    touches_background,
    touches_edge,
    is_encircled,
    encircling_parent,
):
    """JIT-compiled inner loop for encirclement detection."""
    Y, X = slice_2d.shape

    # Horizontal neighbors (4-connectivity)
    for y in range(Y):
        for x in range(X - 1):
            a = slice_2d[y, x]
            b = slice_2d[y, x + 1]
            if a != b:
                if a > 0 and b > 0:
                    if b < neighbor_min[a]:
                        neighbor_min[a] = b
                    if b > neighbor_max[a]:
                        neighbor_max[a] = b
                    if a < neighbor_min[b]:
                        neighbor_min[b] = a
                    if a > neighbor_max[b]:
                        neighbor_max[b] = a
                elif a > 0:
                    touches_background[a] = True
                elif b > 0:
                    touches_background[b] = True

    # Vertical neighbors (4-connectivity)
    for y in range(Y - 1):
        for x in range(X):
            a = slice_2d[y, x]
            b = slice_2d[y + 1, x]
            if a != b:
                if a > 0 and b > 0:
                    if b < neighbor_min[a]:
                        neighbor_min[a] = b
                    if b > neighbor_max[a]:
                        neighbor_max[a] = b
                    if a < neighbor_min[b]:
                        neighbor_min[b] = a
                    if a > neighbor_max[b]:
                        neighbor_max[b] = a
                elif a > 0:
                    touches_background[a] = True
                elif b > 0:
                    touches_background[b] = True

    # Edge detection: y=0, y=Y-1, x=0, x=X-1
    for x in range(X):
        if slice_2d[0, x] > 0:
            touches_edge[slice_2d[0, x]] = True
        if slice_2d[Y - 1, x] > 0:
            touches_edge[slice_2d[Y - 1, x]] = True
    for y in range(Y):
        if slice_2d[y, 0] > 0:
            touches_edge[slice_2d[y, 0]] = True
        if slice_2d[y, X - 1] > 0:
            touches_edge[slice_2d[y, X - 1]] = True

    # Determine encirclement: exactly one neighbor, no bg/edge contact
    max_label = len(neighbor_min) - 1
    for k in range(1, max_label + 1):
        if touches_background[k] or touches_edge[k]:
            continue
        if neighbor_min[k] == neighbor_max[k] and neighbor_min[k] > 0:
            is_encircled[k] = True
            encircling_parent[k] = neighbor_min[k]


def _compute_slice_encircled_numba(
    slice_2d: np.ndarray,
    max_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute encirclement info for a single slice (4-connected).

    Uses O(K) memory instead of O(K²) by tracking min/max neighbors.
    A label is encircled if neighbor_min == neighbor_max (exactly one neighbor)
    and it doesn't touch background or edge.
    """
    INT32_MAX = 2147483647

    neighbor_min = np.full(max_label + 1, INT32_MAX, dtype=np.int32)
    neighbor_max = np.zeros(max_label + 1, dtype=np.int32)
    touches_background = np.zeros(max_label + 1, dtype=np.bool_)
    touches_edge = np.zeros(max_label + 1, dtype=np.bool_)
    is_encircled = np.zeros(max_label + 1, dtype=np.bool_)
    encircling_parent = np.zeros(max_label + 1, dtype=np.int32)

    _compute_slice_encircled_inner(
        slice_2d,
        neighbor_min,
        neighbor_max,
        touches_background,
        touches_edge,
        is_encircled,
        encircling_parent,
    )

    return is_encircled, encircling_parent


@profile
def compute_metadata_and_adjacency(
    masks: np.ndarray,
    compute_contact_areas: bool = True,
) -> tuple[np.ndarray, list[set[int]], list[dict[int, int]]]:
    """
    Phase 3 helper: compute per-label volumes and adjacency graph.

    This operates on the 3D label volume produced by
    ``resize_and_compute_masks(..., do_3D=True)`` and is designed to be
    a performance-measurable step in a phased 3D mask post-processing pipeline.

    Parameters
    ----------
    masks
        3D integer label volume, shape (Z, Y, X) with background = 0.
    compute_contact_areas
        If True, also compute approximate 6-connected contact areas
        between labels for downstream use (e.g. small-cell donation).

    Returns
    -------
    volumes
        1D array of length ``max_label + 1`` where ``volumes[k]`` is the
        voxel count of label ``k``.
    adjacency
        ``adjacency[k]`` is a set of labels that touch label ``k`` via
        6-connected (face-adjacent) contacts (labels > 0). Adjacency is
        only populated when ``compute_contact_areas`` is True.
    contact_areas
        ``contact_areas[k][l]`` is the approximate 6-connected interface
        size between labels ``k`` and ``l`` (in voxel-face units). If
        ``compute_contact_areas`` is False, the list contains empty
        dicts.
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("masks must be a 3D array")

    max_label = int(masks.max())
    volumes = np.bincount(masks.ravel(), minlength=max_label + 1).astype(np.int64)

    adjacency: list[set[int]] = [set() for _ in range(max_label + 1)]
    contact_areas: list[dict[int, int]] = [dict() for _ in range(max_label + 1)]

    if max_label == 0:
        return volumes, adjacency, contact_areas

    if not compute_contact_areas:
        return volumes, adjacency, contact_areas

    if not np.issubdtype(masks.dtype, np.integer):
        raise ValueError("masks must have an integer dtype for contact area computation")

    # Sparse contact computation: O(B) memory instead of O(K²)
    # Phase 1: Count boundary faces (parallel)
    masks_int = masks.astype(np.int32, copy=False)
    n_boundary = _count_boundary_faces(masks_int)

    if n_boundary == 0:
        return volumes, adjacency, contact_areas

    # Phase 2: Collect boundary pairs (sequential but cache-friendly)
    pairs_a = np.empty(n_boundary, dtype=np.int32)
    pairs_b = np.empty(n_boundary, dtype=np.int32)
    n_written = _collect_boundary_pairs(masks_int, pairs_a, pairs_b)

    # Trim to actual size (should match, but be safe)
    if n_written < n_boundary:
        pairs_a = pairs_a[:n_written]
        pairs_b = pairs_b[:n_written]

    # Phase 3: Aggregate to adjacency/contact_areas using NumPy (O(B log B))
    adjacency, contact_areas = _aggregate_boundary_pairs(pairs_a, pairs_b, max_label)

    return volumes, adjacency, contact_areas


@profile
def donate_small_cells(
    masks: np.ndarray,
    volumes: np.ndarray,
    adjacency: list[set[int]],
    contact_areas: list[dict[int, int]],
    V_min: int,
    min_contact_fraction: float = 0.0,
    in_place: bool = True,
) -> np.ndarray:
    """
    Phase 2 helper: remove very small cells and donate their voxels.

    This assumes metadata from :func:`compute_metadata_and_adjacency`
    and operates only on labels with volume ``0 < V(k) < V_min``.

    For each such label ``k``:

    * If it has no neighbors, all its voxels are set to background.
    * Otherwise, its voxels are reassigned to the neighbor with the
      largest measured 6-connected contact area, provided the contact
      area is sufficiently large relative to ``V(k)`` when
      ``min_contact_fraction > 0``. If no suitable neighbor is found,
      the label is dropped to background.

    Parameters
    ----------
    masks
        3D integer label volume, background = 0.
    volumes
        Volumes per label from :func:`compute_metadata_and_adjacency`.
    adjacency
        6-connected adjacency graph per label (face-adjacent neighbors).
    contact_areas
        6-connected contact areas per label pair.
    V_min
        Minimum volume threshold (in voxels). Labels with
        ``0 < V(k) < V_min`` are candidates for removal.
    min_contact_fraction
        Optional lower bound on ``max_contact_area / V(k)``. If the best
        contact area is smaller than this fraction of ``V(k)``, the
        label is dropped to background instead of being donated.
    in_place
        When True, modify ``masks`` in place, otherwise operate on a
        copy.

    Returns
    -------
    masks_out
        Label volume with small labels removed or donated.
    """
    if V_min <= 0:
        return masks

    masks_out = masks if in_place else masks.copy()
    max_label = len(volumes) - 1
    if max_label <= 0:
        return masks_out

    small_labels = np.where((volumes > 0) & (volumes < V_min))[0]
    if small_labels.size == 0:
        return masks_out

    # Build lookup table: label -> recipient (0 means drop to background)
    # This allows vectorized remapping
    remap = np.arange(max_label + 1, dtype=masks_out.dtype)  # Identity by default

    for k in small_labels:
        k = int(k)
        neighbors_k = adjacency[k] if k < len(adjacency) else set()
        # Only allow donation to non-small neighbors (upward donation).
        # This prevents small↔small donation cycles where fragments swap
        # labels without being truly removed.
        neighbors_k = {
            int(label) for label in neighbors_k if label > 0 and label != k and volumes[int(label)] >= V_min
        }

        if not neighbors_k:
            remap[k] = 0
            continue

        areas_k = contact_areas[k] if k < len(contact_areas) else {}
        best_recipient = 0
        best_area = 0
        for neighbor in neighbors_k:
            area = areas_k.get(neighbor, 0)
            if area > best_area:
                best_area = area
                best_recipient = neighbor

        if best_recipient == 0 or best_area == 0:
            remap[k] = 0
            continue

        volume_k = volumes[k]
        if volume_k > 0 and min_contact_fraction > 0.0:
            if best_area / float(volume_k) < min_contact_fraction:
                remap[k] = 0
                continue

        remap[k] = best_recipient

    # Apply remapping to the full volume. remap is the identity for
    # non-small labels, so this is equivalent to selectively updating
    # only small labels but avoids the costly np.isin over the volume.
    masks_out[...] = remap[masks_out]

    return masks_out


@profile
def absorb_encircled_rois(masks, in_place=True):
    """
    Absorb ROIs encircled by a single parent in each 2D slice.

    For each z-slice independently: if label k only touches one other
    label p in that slice (via 4-connectivity), and k doesn't touch
    background or the xy edges in that slice, then k's voxels in that
    slice are absorbed into p.

    This allows partial absorption - the same ROI can be absorbed in
    some slices but not others.

    Parameters
    ----------
    masks
        3D integer label volume, shape (Z, Y, X), background = 0.
        Labels should be sequential (1, 2, 3, ...). Large label values
        (e.g., bitmask-encoded) will cause O(max_label) memory allocation.
    in_place
        When True, modify ``masks`` in place; otherwise operate on a copy.

    Returns
    -------
    np.ndarray
        Label volume with encircled ROIs absorbed per slice.
    """
    masks_out = masks if in_place else masks.copy()
    Z, _, _ = masks_out.shape

    for z in range(Z):
        slice_2d = masks_out[z]
        max_label = int(slice_2d.max())
        if max_label == 0:
            continue

        is_encircled, encircling_parent = _compute_slice_encircled_numba(
            slice_2d.astype(np.int32, copy=False), max_label
        )

        remap = np.arange(max_label + 1, dtype=slice_2d.dtype)
        remap[is_encircled] = encircling_parent[is_encircled]
        slice_2d[...] = remap[slice_2d]

    return masks_out


def smooth_masks_3d(
    masks: np.ndarray,
    num_iterations: int = 1,
    min_neighbors_to_fill: int = 18,
    interface_margin: int = 1,
    min_labeled_for_flip: int = 4,
    flip_kernel_size: int = 3,
    in_place: bool = True,
) -> np.ndarray:
    """
    Phase 3 helper: label-preserving 3D boundary smoothing.

    This operates at label/background interfaces AND label/label interfaces,
    applying a small number of iterations of three local operations:

    * Erode thin protrusions (label -> 0) where a boundary voxel sees
      more background than self and does not touch any other label.
    * Smooth label-label interfaces by flipping voxels to the neighboring
      label when that label has strong local majority (Case B).
    * Fill small internal holes (0 -> label) when a background voxel is
      surrounded exclusively by a single label with sufficient support.

    Multi-label junctions (3+ labels meeting) are left unchanged to
    preserve topology.

    Parameters
    ----------
    masks
        3D integer label volume, background = 0.
    num_iterations
        Number of smoothing iterations (typically 1 or 2). More iterations
        produce smoother boundaries but may over-smooth fine structures.
    min_neighbors_to_fill
        Minimum number of neighbors (out of 26) with a given label required
        to fill a background voxel with that label. Higher values are more
        conservative (only fill deep holes). Range: 1-26, default 18.
    interface_margin
        Margin for interface flipping between two labels. A voxel of label A
        flips to label B when: count(B neighbors) > count(A neighbors) + margin.

        - Lower values (0-1): More aggressive smoothing, interfaces move more
        - Higher values (3-5): Conservative, only flip obvious outliers
        - Default 1 means ~54% local majority triggers a flip

        Only applies at two-label interfaces; multi-label junctions (3+) are
        never modified regardless of this setting.
    min_labeled_for_flip
        Edge guard threshold for interface flipping. Only flip when the total
        count of labeled neighbors (self + other) >= this value. Prevents
        spurious flips at cell edges where most neighbors are background.

        - Lower values (2-4): Allow flips even at sparse boundaries
        - Higher values (8-13): Only flip in dense label regions
        - Default 4; max useful value depends on flip_kernel_size
    flip_kernel_size
        Size of the cubic kernel used for interface flipping decisions.
        Must be an odd integer (3, 5, 7, ...). Larger kernels have a bigger
        "receptive field" and can resolve coarser jaggedness.

        - 3: 26 neighbors, fine-scale smoothing (default)
        - 5: 124 neighbors, resolves medium jaggedness
        - 7: 342 neighbors, resolves coarse jaggedness

        Erosion and hole-filling always use 3×3×3 regardless of this setting.
    in_place
        When True, modify ``masks`` in place, otherwise operate on a
        copy.

    Returns
    -------
    masks_out
        Smoothed label volume.
    """
    masks_out = masks if in_place else masks.copy()
    if masks_out.ndim != 3:
        raise ValueError("masks must be a 3D array")
    if flip_kernel_size < 3 or flip_kernel_size % 2 == 0:
        raise ValueError("flip_kernel_size must be an odd integer >= 3")

    Z, Y, X = masks_out.shape

    # 3×3×3 offsets for erosion and fill (always used)
    neighbor_offsets_3 = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    # Larger kernel offsets for interface flipping
    r = flip_kernel_size // 2
    neighbor_offsets_flip = [
        (dz, dy, dx)
        for dz in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    for _ in range(num_iterations):
        old = masks_out.copy()
        old_int32 = old.astype(np.int32)

        # Padding for 3×3×3 operations (erosion, fill)
        padded_3 = np.pad(old_int32, 1, constant_values=-1)

        # Accumulator arrays for erosion (max value is 26, uint8 suffices)
        n_self = np.zeros((Z, Y, X), dtype=np.uint8)
        n_bg = np.zeros((Z, Y, X), dtype=np.uint8)
        has_other = np.zeros((Z, Y, X), dtype=bool)

        # For fill: track min/max of non-zero neighbors
        min_nonzero_neighbor = np.full((Z, Y, X), np.iinfo(np.uint32).max, dtype=np.uint32)
        max_nonzero_neighbor = np.zeros((Z, Y, X), dtype=np.uint32)
        n_nonzero_neighbors = np.zeros((Z, Y, X), dtype=np.uint8)

        # Loop 1: 3×3×3 for erosion and fill
        for dz, dy, dx in neighbor_offsets_3:
            shifted = padded_3[1 + dz : 1 + dz + Z, 1 + dy : 1 + dy + Y, 1 + dx : 1 + dx + X]

            # Erosion: count self, background, detect other labels
            n_self += shifted == old
            n_bg += shifted == 0
            has_other |= (shifted > 0) & (shifted != old)

            # Fill: track min/max of non-zero neighbors
            nonzero_mask = shifted > 0
            min_nonzero_neighbor = np.where(
                nonzero_mask & (shifted < min_nonzero_neighbor), shifted, min_nonzero_neighbor
            )
            max_nonzero_neighbor = np.where(
                nonzero_mask & (shifted > max_nonzero_neighbor), shifted, max_nonzero_neighbor
            )
            n_nonzero_neighbors += nonzero_mask

        # Padding for flip kernel (may be larger)
        padded_flip = np.pad(old_int32, r, constant_values=-1)

        # For interface smoothing: track "other" labels with larger kernel
        other_min = np.full((Z, Y, X), np.iinfo(np.uint32).max, dtype=np.uint32)
        other_max = np.zeros((Z, Y, X), dtype=np.uint32)
        n_other = np.zeros((Z, Y, X), dtype=np.uint16)  # uint16 for larger kernels
        n_self_flip = np.zeros((Z, Y, X), dtype=np.uint16)

        # Loop 2: Larger kernel for interface flipping
        for dz, dy, dx in neighbor_offsets_flip:
            shifted = padded_flip[r + dz : r + dz + Z, r + dy : r + dy + Y, r + dx : r + dx + X]

            # Count self and other labels
            n_self_flip += shifted == old
            is_other = (shifted > 0) & (shifted != old)
            other_min = np.where(is_other & (shifted < other_min), shifted, other_min)
            other_max = np.where(is_other & (shifted > other_max), shifted, other_max)
            n_other += is_other

        # Identify boundary voxels (foreground with any different neighbor in 3×3×3)
        fg = old > 0
        boundary = fg & (n_self < 26)

        # Case A: Erosion mask - boundary, no other label touching, more bg than self
        erode_mask = boundary & ~has_other & (n_bg > n_self)

        # Case B: Interface flip - exactly one other label, strong majority
        # Detect exactly one other label (not Case C with 3+ labels)
        # Uses larger kernel counts (n_self_flip, n_other) for flip decision
        exactly_one_other = (n_other > 0) & (other_min == other_max)
        second_label = other_max  # The other label (valid when exactly_one_other)

        two_label_interface = (
            fg  # current voxel is labeled
            & exactly_one_other  # exactly one other label (not Case C)
            & ((n_self_flip + n_other) >= min_labeled_for_flip)  # edge guard
        )
        flip_mask = two_label_interface & (n_other > n_self_flip + interface_margin)

        # Fill mask: background, all nonzero neighbors same label, enough support
        all_same_label = min_nonzero_neighbor == max_nonzero_neighbor
        fill_mask = (
            (old == 0)
            & (max_nonzero_neighbor > 0)
            & all_same_label
            & (n_nonzero_neighbors >= min_neighbors_to_fill)
        )

        # Early termination if no changes
        if not erode_mask.any() and not flip_mask.any() and not fill_mask.any():
            break

        # Apply changes (order matters)
        masks_out = old.copy()
        masks_out[erode_mask] = 0
        masks_out[flip_mask] = second_label[flip_mask]
        masks_out[fill_mask] = max_nonzero_neighbor[fill_mask]

    return masks_out


def opening_correct_protrusions(
    masks: np.ndarray,
    r_neck: int = 2,
    dominance_threshold: float = 0.5,
    min_protrusion_volume: int = 10,
    in_place: bool = True,
) -> np.ndarray:
    """
    Phase 2.5: Correct protrusions using morphological opening per label.

    Detects "necks" (local narrowing) by finding regions that don't survive
    morphological opening at scale r_neck. These thin regions are reassigned
    to the neighbor with dominant contact.

    Parameters
    ----------
    masks : np.ndarray
        3D label volume from Phase 2.
    r_neck : int
        Opening radius. Structures thinner than ~2*r_neck are removed.
        Larger = more aggressive neck detection.
    dominance_threshold : float
        Fraction of contact needed to donate (0.5 = majority rule).
    min_protrusion_volume : int
        Skip protrusions smaller than this (avoid noise).
    in_place : bool
        Modify masks in place or return copy.

    Returns
    -------
    masks_out : np.ndarray
        Corrected label volume.
    """
    from scipy.ndimage import binary_opening, find_objects, grey_dilation
    from scipy.ndimage import label as label_components

    masks_out = masks if in_place else masks.copy()
    if masks_out.ndim != 3:
        raise ValueError("masks must be a 3D array")

    max_label = int(masks_out.max())
    if max_label == 0:
        return masks_out

    struct = np.ones((3, 3, 3), dtype=bool)

    # Get all bounding boxes in one pass (O(N) instead of O(K×N))
    slices_list = find_objects(masks_out)

    for k in range(1, max_label + 1):
        if slices_list[k - 1] is None:
            continue  # Label doesn't exist

        # Expand bounding box by margin
        base_slices = slices_list[k - 1]
        margin = r_neck + 1
        slices = tuple(
            slice(max(0, s.start - margin), min(dim, s.stop + margin))
            for s, dim in zip(base_slices, masks_out.shape)
        )

        sub_masks = masks_out[slices].copy()  # Copy to avoid issues during modification
        B_k = sub_masks == k

        # Morphological opening (erode then dilate)
        B_k_open = binary_opening(B_k, structure=struct, iterations=r_neck)

        # Protrusion = thin parts removed by opening
        P_k = B_k & ~B_k_open
        P_k_volume = int(P_k.sum())

        if P_k_volume < min_protrusion_volume:
            continue

        # Safety: skip if opening removed too much (> 50% of cell)
        B_k_volume = int(B_k.sum())
        if P_k_volume > B_k_volume * 0.5:
            continue

        # Vectorized neighbor detection using grey_dilation (replaces 26 np.roll calls)
        dilated = grey_dilation(sub_masks, size=(3, 3, 3))
        neighbor_contact = P_k & (dilated > 0) & (dilated != k)

        P_k_contact = P_k & neighbor_contact
        if not P_k_contact.any():
            continue

        # Connected components
        labeled_P, num_comp = label_components(P_k_contact)

        # Pre-compute padded array for efficient neighbor lookup
        sz, sy, sx = sub_masks.shape
        padded = np.pad(sub_masks, 1, mode="constant", constant_values=0)

        for c in range(1, num_comp + 1):
            comp_mask = labeled_P == c

            # Vectorized contact counting using 6-connectivity slicing
            neighbors = []
            for dz, dy, dx in [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ]:
                shifted = padded[1 + dz : 1 + dz + sz, 1 + dy : 1 + dy + sy, 1 + dx : 1 + dx + sx]
                neighbors.append(shifted[comp_mask])

            all_neighbors = np.concatenate(neighbors)
            all_neighbors = all_neighbors[(all_neighbors > 0) & (all_neighbors != k)]

            if len(all_neighbors) == 0:
                continue

            # Bincount for O(n) counting instead of Python loop, using a
            # compact local label range to keep the bincount array small
            # even when global label IDs are large.
            neighbors_arr = all_neighbors.astype(np.int64)
            min_neighbor = int(neighbors_arr.min())
            shifted = neighbors_arr - min_neighbor
            counts = np.bincount(shifted)
            best_idx = int(counts.argmax())
            best_neighbor = min_neighbor + best_idx
            best_contact = int(counts[best_idx])
            total_contact = len(all_neighbors)

            if best_contact >= dominance_threshold * total_contact:
                # Direct coordinate assignment (avoids full-volume mask allocation)
                local_z, local_y, local_x = np.where(comp_mask)
                global_z = local_z + slices[0].start
                global_y = local_y + slices[1].start
                global_x = local_x + slices[2].start
                masks_out[global_z, global_y, global_x] = best_neighbor

    return masks_out


def gaussian_smooth_labels(
    masks: np.ndarray,
    sigma: float | Sequence[float] = (1.5, 3.0, 3.0),
    in_place: bool = True,
    bg_scale: float = 1.0,
    max_expansion: int = 2,
) -> np.ndarray:
    """
    Smooth label boundaries using Gaussian-weighted voting.

    Each voxel is assigned the label with highest Gaussian-weighted
    vote from its neighborhood. This smooths jagged boundaries, removes
    thin protrusions, and fills small holes - analogous to a Gaussian
    low-pass filter but preserving discrete label identities.

    Parameters
    ----------
    masks : np.ndarray
        3D integer label array, background = 0.
    sigma : float or sequence of float, optional
        Gaussian sigma in voxel units. Larger values = more smoothing.
        If a scalar is provided, smoothing is isotropic. If a sequence
        of length 3 is provided, it is interpreted as
        ``(sigma_z, sigma_y, sigma_x)`` matching the ``(Z, Y, X)`` axes.
        Features smaller than ~sigma along each axis will be removed/
        smoothed. Default is ``(1.5, 3.0, 3.0)``. Typical scalar range: 1-3.
    in_place : bool
        Modify masks in place or return copy.
    max_expansion : int
        Maximum voxels labels can expand beyond the original foreground.
        Set to 0 to disable expansion constraint. Default 2.

    Returns
    -------
    np.ndarray
        Smoothed label array.
    """
    from scipy.ndimage import binary_dilation, find_objects, gaussian_filter

    masks_out = masks if in_place else masks.copy()
    if masks_out.ndim != 3:
        raise ValueError("masks must be a 3D array")

    max_label = int(masks_out.max())
    if max_label == 0:
        return masks_out

    # Normalize sigma to ZYX per-axis form and derive a conservative
    # spatial margin from the largest sigma so the Gaussian kernel
    # support is fully contained in the expanded bounding boxes.
    sigma_zyx = _normalize_sigma_zyx(sigma)
    sigma_max = max(sigma_zyx)
    if sigma_max <= 0:
        raise ValueError("sigma must be positive (all components > 0)")

    # Compute dilated constraint mask to prevent labels from expanding too far
    original_foreground = masks_out > 0
    if max_expansion > 0:
        struct = np.ones((3, 3, 3), dtype=bool)
        dilated_constraint = binary_dilation(original_foreground, structure=struct, iterations=max_expansion)
    else:
        dilated_constraint = original_foreground

    # Get bounding boxes for all labels (O(N) total)
    slices_list = find_objects(masks_out)

    # Streaming argmax: track best score and winning label at each voxel
    best_score = np.full(masks_out.shape, -np.inf, dtype=np.float32)
    result = np.zeros_like(masks_out)

    # Background: full volume convolution
    bg_indicator = (masks_out == 0).astype(np.float32)
    bg_score = gaussian_filter(bg_indicator, sigma_zyx)
    if bg_scale != 1.0:
        bg_score *= float(bg_scale)
    update = bg_score > best_score
    result[update] = 0
    best_score[update] = bg_score[update]

    # Gaussian extends ~4σ effectively along each axis; use the largest
    # sigma to derive a scalar margin for bounding box expansion.
    margin = int(np.ceil(4 * sigma_max))

    for k in range(1, max_label + 1):
        sl = slices_list[k - 1] if k - 1 < len(slices_list) else None
        if sl is None:
            continue

        # Expand bounding box by margin
        base_slices = sl
        exp_slices = tuple(
            slice(max(0, s.start - margin), min(dim, s.stop + margin))
            for s, dim in zip(base_slices, masks_out.shape)
        )

        # Local indicator and Gaussian convolution
        local_mask = masks_out[exp_slices]
        local_indicator = (local_mask == k).astype(np.float32)
        local_score = gaussian_filter(local_indicator, sigma_zyx)

        # Update where this label wins
        local_best = best_score[exp_slices]
        update = local_score > local_best

        # Update result and best_score in expanded region
        result_slice = result[exp_slices]
        result_slice = np.where(update, k, result_slice)
        result[exp_slices] = result_slice
        best_score[exp_slices] = np.maximum(local_best, local_score)

    # Constrain result to dilated original foreground
    result[~dilated_constraint] = 0

    if in_place:
        masks[:] = result
        return masks
    return result


def gaussian_smooth_labels_cupy(
    masks: np.ndarray,
    sigma: float | Sequence[float] = (1.5, 3.0, 3.0),
    in_place: bool = True,
    bg_scale: float = 1.0,
    max_expansion: int = 2,
) -> np.ndarray:
    """
    GPU-accelerated Gaussian label smoothing using CuPy.

    This mirrors :func:`gaussian_smooth_labels` but performs the
    Gaussian convolutions on the GPU via ``cupyx.scipy.ndimage``.
    It keeps memory usage O(N) by streaming an argmax over labels
    rather than materializing a full [Z, Y, X, K] score tensor.

    Parameters
    ----------
    masks : np.ndarray
        3D integer label array, background = 0.
    sigma : float or sequence of float, optional
        Gaussian sigma in voxel units. Larger values = more smoothing.
        If a scalar is provided, smoothing is isotropic. If a sequence
        of length 3 is provided, it is interpreted as
        ``(sigma_z, sigma_y, sigma_x)`` matching the ``(Z, Y, X)`` axes.
    in_place : bool
        If True, write results back into ``masks``; otherwise return
        a new array.
    bg_scale : float
        Multiplicative factor applied to the background (label 0)
        score before voting. Values < 1.0 weaken background; values
        > 1.0 strengthen it.
    max_expansion : int
        Maximum voxels labels can expand beyond the original foreground.
        Set to 0 to disable expansion constraint. Default 2.

    Returns
    -------
    np.ndarray
        Smoothed label array on the host (NumPy).
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_dilation as cp_binary_dilation
        from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
        from scipy.ndimage import find_objects
    except Exception as e:  # pragma: no cover - import failure path
        raise ImportError("CuPy or cupyx.scipy.ndimage is not available") from e

    if masks.ndim != 3:
        raise ValueError("masks must be a 3D array")

    # Work on a copy on host, then move to device.
    masks_host = masks if in_place else masks.copy()
    max_label = int(masks_host.max())
    if max_label == 0:
        return masks_host

    # Normalize sigma to ZYX per-axis form and derive a conservative
    # spatial margin from the largest sigma so the Gaussian kernel
    # support is fully contained in the expanded bounding boxes.
    sigma_zyx = _normalize_sigma_zyx(sigma)
    sigma_max = max(sigma_zyx)
    if sigma_max <= 0:
        raise ValueError("sigma must be positive (all components > 0)")

    # Compute bounding boxes on CPU (cheap O(N) pass).
    slices_list = find_objects(masks_host)

    masks_dev = cp.asarray(masks_host, dtype=cp.int32)
    Z, Y, X = masks_dev.shape

    # Compute dilated constraint mask to prevent labels from expanding too far
    original_foreground = masks_dev > 0
    if max_expansion > 0:
        struct = cp.ones((3, 3, 3), dtype=bool)
        dilated_constraint = original_foreground
        for _ in range(max_expansion):
            dilated_constraint = cp_binary_dilation(dilated_constraint, structure=struct, iterations=1)
    else:
        dilated_constraint = original_foreground

    # Streaming argmax: track best score and winning label at each voxel
    best_score = cp.full((Z, Y, X), -cp.inf, dtype=cp.float32)
    result = cp.zeros((Z, Y, X), dtype=cp.int32)

    # Background: full volume convolution on device
    bg_indicator = (masks_dev == 0).astype(cp.float32)
    bg_score = cp_gaussian_filter(bg_indicator, sigma_zyx)
    del bg_indicator
    if bg_scale != 1.0:
        bg_score *= float(bg_scale)
    update = bg_score > best_score
    result[update] = 0
    best_score[update] = bg_score[update]
    del bg_score, update

    # Gaussian extends ~4σ effectively along each axis; use the largest
    # sigma to derive a scalar margin for bounding box expansion.
    margin = int(math.ceil(4.0 * sigma_max))

    for k in range(1, max_label + 1):
        sl = slices_list[k - 1] if k - 1 < len(slices_list) else None
        if sl is None:
            continue

        # Expand bounding box by margin, clipped to volume
        base_slices = sl
        exp_slices = tuple(
            slice(max(0, s.start - margin), min(dim, s.stop + margin))
            for s, dim in zip(base_slices, masks_dev.shape)
        )

        # Local indicator and Gaussian convolution on device
        local_mask = masks_dev[exp_slices]
        local_indicator = (local_mask == k).astype(cp.float32)
        if not cp.any(local_indicator):
            continue

        local_score = cp_gaussian_filter(local_indicator, sigma_zyx)

        # Update where this label wins
        local_best = best_score[exp_slices]
        update = local_score > local_best

        result_slice = result[exp_slices]
        result_slice = cp.where(update, k, result_slice)
        result[exp_slices] = result_slice
        best_score[exp_slices] = cp.maximum(local_best, local_score)

    # Constrain result to dilated original foreground
    result = cp.where(dilated_constraint, result, 0)

    # Copy result back to host, then drop large device arrays so they can be freed.
    result_host = cp.asnumpy(result).astype(masks_host.dtype, copy=False)
    del result, best_score, masks_dev, original_foreground, dilated_constraint

    if in_place:
        masks[:] = result_host
        return masks
    return result_host


def gaussian_erosion_to_margin_and_scale(
    sigma: float,
    erosion_voxels: float | None = None,
    fwhm_fraction: float | None = None,
) -> tuple[float, float]:
    """
    Compute Gaussian background margin and scale from a geometric erosion spec.

    This helper converts an intended erosion distance for a Gaussian-blurred
    binary object into parameters that can be used to bias background in the
    Gaussian voting step.

    Under a planar boundary model, blurring the indicator of a half-space
    with a Gaussian of standard deviation ``sigma`` gives a CDF profile
    along the normal direction. Thresholding that blurred signal at level
    ``t`` moves the effective boundary inward by

        d(t) = sigma * sqrt(2) * Phi^{-1}(t)

    where ``Phi`` is the standard normal CDF. This function inverts that
    relation for two common ways of specifying erosion:

    * ``erosion_voxels``: target erosion distance in voxels along the
      boundary normal.
    * ``fwhm_fraction``: target erosion as a fraction of the Gaussian
      FWHM radius (~2.355 * sigma).

    It returns two quantities for that target erosion:

    * ``margin`` (delta):

        s_bg >= s_cell + margin

      corresponds to a cell threshold

        t = (1 + margin) / 2

    * ``scale`` (alpha): a multiplicative factor on background scores

        s_cell >= alpha * s_bg

      which corresponds to a cell threshold

        t = alpha / (1 + alpha)

    Either ``erosion_voxels`` OR ``fwhm_fraction`` must be provided, but
    not both.

    Parameters
    ----------
    sigma
        Gaussian standard deviation in voxel units.
    erosion_voxels
        Desired erosion distance (in voxels) along the boundary normal.
    fwhm_fraction
        Desired erosion as a fraction of the Gaussian FWHM radius
        (~2.355 * sigma).

    Returns
    -------
    margin : float
        Additive margin delta such that background must exceed the cell
        score by at least ``margin`` to erode a planar boundary by the
        specified amount.
    scale : float
        Multiplicative factor alpha such that a decision rule
        ``s_cell >= alpha * s_bg`` has the same effective erosion.
    """
    if (erosion_voxels is None) == (fwhm_fraction is None):
        raise ValueError("Specify exactly one of erosion_voxels or fwhm_fraction")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Compute threshold t for the cell from the requested erosion
    if erosion_voxels is not None:
        # d = sigma * sqrt(2) * Phi^{-1}(t) => t = Phi(d / (sigma * sqrt(2)))
        u = erosion_voxels / (sigma * math.sqrt(2.0))
        t = 0.5 * (1.0 + math.erf(u / math.sqrt(2.0)))
    else:
        # Erosion specified as fraction of FWHM: d = f * sigma * sqrt(8 ln 2)
        # t = Phi(d / (sigma * sqrt(2))) = Phi(f * sqrt(4 ln 2))
        u = fwhm_fraction * math.sqrt(4.0 * math.log(2.0))
        t = 0.5 * (1.0 + math.erf(u / math.sqrt(2.0)))

    # Clamp to open interval (0,1) to avoid infinities
    eps = 1e-6
    if t <= eps:
        t = eps
    elif t >= 1.0 - eps:
        t = 1.0 - eps

    # margin delta: cell threshold t = (1 + delta) / 2 => delta = 2t - 1
    margin = 2.0 * t - 1.0

    # scale alpha: cell threshold t = alpha / (1 + alpha) => alpha = t / (1 - t)
    scale = t / (1.0 - t)

    return margin, scale


def relabel_connected_components(
    masks: np.ndarray,
    in_place: bool = True,
) -> np.ndarray:
    """
    Relabel so each connected component gets a unique label ID.

    If a label has become disconnected into multiple pieces, each
    piece receives a new unique label. This allows downstream
    functions like donate_small_cells to handle fragments properly.

    The first component of each original label keeps that label;
    additional disconnected components get new unique IDs.

    Parameters
    ----------
    masks : np.ndarray
        3D integer label volume, background = 0.
    in_place : bool
        When True, modify ``masks`` in place; otherwise operate on a copy.

    Returns
    -------
    np.ndarray
        Relabeled volume where each connected component has unique ID.
        When ``in_place`` is False, the returned array has dtype
        ``np.uint32``. When ``in_place`` is True, the input array is
        modified in place and retains its original dtype.
    """
    from scipy.ndimage import find_objects
    from scipy.ndimage import label as label_components

    if masks.ndim != 3:
        raise ValueError("masks must be a 3D array")

    max_label = int(masks.max())
    if max_label == 0:
        if in_place:
            return masks
        return masks.astype(np.uint32, copy=True)

    # Work on a copy unless explicitly in-place
    if in_place:
        result = masks
    else:
        result = masks.astype(np.uint32, copy=True)

    # Next available label ID for new components
    next_label = max_label

    # Compute bounding boxes once to avoid full-volume scans per label
    slices_list = find_objects(result)

    # Process each label independently so different labels are never merged
    for k in range(1, max_label + 1):
        sl = slices_list[k - 1] if k - 1 < len(slices_list) else None
        if sl is None:
            continue

        sub = result[sl]
        mask_k = sub == k
        if not np.any(mask_k):
            continue

        cc_k, num_cc = label_components(mask_k)
        if num_cc <= 1:
            continue  # already a single connected component

        # Keep first component as label k, give new IDs to additional ones
        first_seen = False
        for cc_id in range(1, num_cc + 1):
            comp_mask = cc_k == cc_id  # within sub-volume
            if not np.any(comp_mask):
                continue
            if not first_seen:
                # First component keeps original label k (no change needed)
                first_seen = True
                continue
            next_label += 1
            sub[comp_mask] = next_label

        # Write back updated sub-volume
        result[sl] = sub

    return result
