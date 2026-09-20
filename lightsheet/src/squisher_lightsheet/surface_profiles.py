from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import SimpleITK as sitk

from squisher_lightsheet import ngff
from squisher_lightsheet.artifact_io import atomic_output_directory, sha256_file
from squisher_lightsheet.surface_geometry import (
    ImageGeometry,
    SurfaceProjection,
    SurfaceProjector,
    load_surface,
)


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def nearest_depth_bin(r_vz_um: npt.ArrayLike, depth_um: npt.ArrayLike) -> IntArray:
    """Assign nonnegative distances to their nearest uniformly spaced depth center."""
    depth = np.asarray(depth_um, dtype=np.float64)
    if (
        depth.ndim != 1
        or len(depth) < 2
        or depth[0] != 0
        or not np.isfinite(depth).all()
        or not np.allclose(np.diff(depth), depth[1] - depth[0])
    ):
        raise ValueError("depth centers must be a finite uniform grid starting at zero")
    step = float(depth[1] - depth[0])
    if step <= 0:
        raise ValueError("depth centers must be a finite uniform grid starting at zero")
    values = np.asarray(r_vz_um, dtype=np.float64)
    represented_max = np.nextafter(depth[-1] + step / 2, np.inf)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > represented_max):
        raise ValueError("r_vz values lie outside the represented depth bins")
    midpoints = (depth[:-1] + depth[1:]) / 2
    return np.searchsorted(midpoints, values, side="left").astype(np.int64, copy=False)


def accumulate_samples(
    vertex_index: npt.ArrayLike,
    depth_index: npt.ArrayLike,
    intensity: npt.ArrayLike,
    vertex_count: int,
    depth_count: int,
) -> tuple[FloatArray, npt.NDArray[np.uint64]]:
    """Accumulate raw sums and counts without averaging intermediate batches."""
    vertices = np.asarray(vertex_index)
    depths = np.asarray(depth_index)
    values = np.asarray(intensity, dtype=np.float64)
    if vertex_count <= 0 or depth_count <= 0:
        raise ValueError("vertex_count and depth_count must be positive")
    if vertices.ndim != 1 or depths.shape != vertices.shape or values.shape != vertices.shape:
        raise ValueError("vertex_index, depth_index, and intensity must be matching 1-D arrays")
    if not np.issubdtype(vertices.dtype, np.integer) or not np.issubdtype(depths.dtype, np.integer):
        raise ValueError("vertex_index and depth_index must contain integers")
    if not np.isfinite(values).all():
        raise ValueError("intensity must contain only finite values")
    if len(vertices) and (
        vertices.min() < 0
        or vertices.max() >= vertex_count
        or depths.min() < 0
        or depths.max() >= depth_count
    ):
        raise ValueError("sample index lies outside the requested output shape")
    flat_index = vertices.astype(np.int64) * depth_count + depths.astype(np.int64)
    size = vertex_count * depth_count
    sums = np.bincount(flat_index, weights=values, minlength=size).reshape(vertex_count, depth_count)
    counts = np.bincount(flat_index, minlength=size).astype(np.uint64).reshape(vertex_count, depth_count)
    return sums, counts


def projection_membership(
    points_xyz_um: npt.ArrayLike,
    projection: SurfaceProjection,
    vertices_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    max_depth_um: float,
) -> npt.NDArray[np.bool_]:
    """Select tissue-side projections away from the open mesh rim and within depth."""
    points = np.asarray(points_xyz_um, dtype=np.float64)
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    triangles_index = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(max_depth_um) or max_depth_um < 0:
        raise ValueError("max_depth_um must be finite and nonnegative")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz_um must have shape (n_points, 3)")
    if len(projection.face_index) != len(points):
        raise ValueError("projection must contain one result per point")
    triangles = vertices[triangles_index[np.asarray(projection.face_index)]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("surface contains a degenerate selected face")
    normals /= lengths[:, None]
    signed_normal = np.einsum("ij,ij->i", points - np.asarray(projection.closest_xyz_um), normals)
    return (
        (signed_normal <= 0)
        & ~np.asarray(projection.on_surface_boundary, dtype=bool)
        & (np.asarray(projection.r_vz_um) <= max_depth_um)
    )


def mesh_graph(vertices_xyz_um: npt.ArrayLike, faces: npt.ArrayLike) -> sparse.csr_matrix:
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices_xyz_um must be finite with shape (n_vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("faces must be a nonempty array with shape (n_faces, 3)")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("faces contain an out-of-range vertex index")
    edges = np.sort(
        np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
        axis=1,
    )
    edges = np.unique(edges, axis=0)
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if np.any(weights == 0):
        raise ValueError("surface contains a zero-length mesh edge")
    row = np.concatenate((edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 1], edges[:, 0]))
    return sparse.csr_matrix((np.tile(weights, 2), (row, col)), shape=(len(vertices), len(vertices)))


def intrinsic_neighborhoods(
    vertices_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    radius_um: float,
    batch_size: int = 256,
) -> sparse.csr_matrix:
    """Build mesh-edge shortest-path neighborhoods in bounded Dijkstra batches."""
    if not np.isfinite(radius_um) or radius_um < 0:
        raise ValueError("radius_um must be finite and nonnegative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    graph = mesh_graph(vertices_xyz_um, faces)
    rows: list[IntArray] = []
    cols: list[IntArray] = []
    for start in range(0, graph.shape[0], batch_size):
        stop = min(start + batch_size, graph.shape[0])
        distance = dijkstra(graph, directed=False, indices=np.arange(start, stop), limit=radius_um)
        row, col = np.nonzero(np.isfinite(distance))
        rows.append(row.astype(np.int64) + start)
        cols.append(col.astype(np.int64))
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(row), dtype=np.uint8), (row, col)), shape=graph.shape)


def aggregate_neighborhood(
    neighborhood: sparse.csr_matrix,
    raw_sum: npt.ArrayLike,
    raw_count: npt.ArrayLike,
) -> tuple[FloatArray, npt.NDArray[np.uint32]]:
    """Pool raw sufficient statistics and range-check the persisted count dtype."""
    sums = np.asarray(raw_sum, dtype=np.float64)
    counts = np.asarray(raw_count)
    if sums.ndim != 2 or counts.shape != sums.shape:
        raise ValueError("raw_sum and raw_count must be matching vertex-by-depth arrays")
    if neighborhood.shape != (len(sums), len(sums)):
        raise ValueError("neighborhood shape must match the profile vertex dimension")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("raw_count must contain nonnegative integers")
    summed = np.asarray(neighborhood @ sums, dtype=np.float64)
    count64 = np.asarray(neighborhood @ counts.astype(np.uint64), dtype=np.uint64)
    if count64.max(initial=0) > np.iinfo(np.uint32).max:
        raise OverflowError("Aggregated sample count exceeds uint32")
    return summed, count64.astype(np.uint32)


def _source_geometry(source_zarr: Path, level: int) -> tuple[Any, str, FloatArray, FloatArray]:
    import zarr

    root = zarr.open_group(str(source_zarr), mode="r")
    if root.attrs.get("squisher_complete") is not True:
        raise ValueError(f"source OME-Zarr is not marked complete: {source_zarr}")
    dataset_path = ngff.level_path(root, level=level, context=source_zarr)
    array = root[dataset_path]
    axes = list(ngff.axes(root, array).lower())
    if axes != ["z", "y", "x"]:
        raise ValueError(f"expected a channel-separated ZYX source, found axes {axes}")
    dataset_index = ngff.dataset_paths(root).index(dataset_path)
    transform_axes, scale, origin, has_scale, has_origin = ngff.scale_translation(
        root, dataset_index=dataset_index
    )
    if transform_axes != axes or not has_scale or not has_origin:
        raise ValueError("source level requires explicit ZYX scale and translation")
    multiscale_axes = ngff.multiscales(root)[0]["axes"]
    units = [axis.get("unit") if isinstance(axis, dict) else None for axis in multiscale_axes]
    if units != ["micrometer"] * 3:
        raise ValueError(f"expected micrometer ZYX axes, found units {units}")
    if array.ndim != 3 or array.dtype != np.dtype(np.uint16):
        raise ValueError(f"expected a 3-D uint16 source array, found {array.shape} {array.dtype}")
    scale_zyx = np.asarray(scale, dtype=np.float64)
    origin_zyx = np.asarray(origin, dtype=np.float64)
    if not np.isfinite(scale_zyx).all() or np.any(scale_zyx <= 0) or not np.isfinite(origin_zyx).all():
        raise ValueError("source scale and translation must be finite with positive scale")
    return array, dataset_path, scale_zyx, origin_zyx


def _mask_sampler(mask_path: Path) -> tuple[FloatArray, ImageGeometry]:
    image = sitk.ReadImage(mask_path)
    if image.GetDimension() != 3:
        raise ValueError("mask must be three-dimensional")
    values = sitk.GetArrayFromImage(image)
    unique = np.unique(values)
    if not np.array_equal(unique, np.array([0, 1], dtype=values.dtype)):
        raise ValueError("mask must contain both binary values 0 and 1")
    geometry = ImageGeometry.from_sitk(image)
    coefficients = spline_filter(values, order=3, output=np.float64, mode="constant")
    return coefficients, geometry


def sample_mask(
    points_xyz_um: npt.ArrayLike, coefficients_zyx: npt.ArrayLike, geometry: ImageGeometry
) -> FloatArray:
    index_xyz = geometry.world_to_intrinsic(points_xyz_um) / geometry.spacing_xyz_um
    return map_coordinates(
        np.asarray(coefficients_zyx, dtype=np.float64),
        np.asarray(index_xyz)[..., ::-1].T,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _internal_empty_rows(count: npt.NDArray[np.uint32]) -> int:
    total = 0
    for row in count:
        occupied = np.flatnonzero(row)
        if len(occupied) and np.any(row[occupied[0] : occupied[-1] + 1] == 0):
            total += 1
    return total


def _extract_profiles(
    *,
    source_array: Any,
    source_scale_zyx_um: FloatArray,
    source_origin_zyx_um: FloatArray,
    mask_coefficients: FloatArray,
    mask_geometry: ImageGeometry,
    vertices_xyz_um: FloatArray,
    faces: IntArray,
    vertex_uv: FloatArray,
    z_start: int,
    z_stop: int,
    stride_zyx: tuple[int, int, int],
    depth_um: FloatArray,
    radius_um: float,
    projection_chunk_size: int,
    geodesic_batch_size: int,
) -> tuple[dict[str, npt.NDArray[Any]], dict[str, int]]:
    raw_sum = np.zeros((len(vertices_xyz_um), len(depth_um)), dtype=np.float64)
    raw_count = np.zeros(raw_sum.shape, dtype=np.uint64)
    counters = {"grid_samples": 0, "inside_cortex": 0, "tissue_nonrim_depth": 0}
    z_stride, y_stride, x_stride = stride_zyx
    y_index = np.arange(0, source_array.shape[1], y_stride)
    x_index = np.arange(0, source_array.shape[2], x_stride)
    grid_x, grid_y = np.meshgrid(
        source_origin_zyx_um[2] + x_index * source_scale_zyx_um[2],
        source_origin_zyx_um[1] + y_index * source_scale_zyx_um[1],
    )
    projector = SurfaceProjector(vertices_xyz_um, faces, vertex_uv)
    nearest_vertex = cKDTree(vertices_xyz_um)
    retained_max_depth = float(depth_um[-1] + (depth_um[1] - depth_um[0]) / 2)

    for z_index in range(z_start, z_stop, z_stride):
        plane = np.asarray(source_array[z_index])[::y_stride, ::x_stride]
        points = np.column_stack(
            (
                grid_x.ravel(),
                grid_y.ravel(),
                np.full(grid_x.size, source_origin_zyx_um[0] + z_index * source_scale_zyx_um[0]),
            )
        )
        values = plane.ravel().astype(np.float64, copy=False)
        counters["grid_samples"] += len(points)
        inside = sample_mask(points, mask_coefficients, mask_geometry) >= 0.5
        counters["inside_cortex"] += int(inside.sum())
        if not np.any(inside):
            continue
        points = points[inside]
        values = values[inside]
        projection = projector.project(points, chunk_size=projection_chunk_size)
        keep = projection_membership(points, projection, vertices_xyz_um, faces, retained_max_depth)
        counters["tissue_nonrim_depth"] += int(keep.sum())
        if not np.any(keep):
            continue
        assigned_vertex = nearest_vertex.query(projection.closest_xyz_um[keep], workers=-1)[1]
        depth_index = nearest_depth_bin(projection.r_vz_um[keep], depth_um)
        sums, counts = accumulate_samples(
            assigned_vertex, depth_index, values[keep], len(vertices_xyz_um), len(depth_um)
        )
        raw_sum += sums
        maximum = np.iinfo(np.uint64).max - raw_count
        if np.any(counts > maximum):
            raise OverflowError("Raw sample count exceeds uint64")
        raw_count += counts

    neighborhood = intrinsic_neighborhoods(vertices_xyz_um, faces, radius_um, batch_size=geodesic_batch_size)
    summed, count = aggregate_neighborhood(neighborhood, raw_sum, raw_count)
    intensity = np.full(summed.shape, np.nan, dtype=np.float32)
    np.divide(summed, count, out=intensity, where=count > 0)
    sample_volume_um3 = float(np.prod(source_scale_zyx_um * np.asarray(stride_zyx)))
    data: dict[str, npt.NDArray[Any]] = {
        "depth_um": depth_um.astype(np.float32),
        "intensity": intensity,
        "valid_count": count,
        "valid_volume_um3": (count.astype(np.float64) * sample_volume_um3).astype(np.float32),
        "vertex_index": np.arange(len(vertices_xyz_um), dtype=np.int64),
    }
    counters["neighborhood_links"] = int(neighborhood.nnz)
    counters["nonempty_bins"] = int(np.count_nonzero(count))
    counters["internal_empty_bins"] = _internal_empty_rows(count)
    return data, counters


def build_surface_profiles(
    *,
    surface_path: Path,
    chart_path: Path,
    mask_path: Path,
    source_zarr: Path,
    output_dir: Path,
    level: int = 2,
    z_start: int = 0,
    z_stop: int | None = None,
    stride_zyx: tuple[int, int, int] = (2, 4, 4),
    radius_um: float = 20.0,
    max_depth_um: float = 280.8,
    depth_step_um: float = 4.8,
    projection_chunk_size: int = 250_000,
    geodesic_batch_size: int = 256,
) -> Path:
    """Build exact closest-surface intensity profiles as one atomic artifact."""
    surface_path = Path(surface_path)
    chart_path = Path(chart_path)
    mask_path = Path(mask_path)
    source_zarr = Path(source_zarr)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if len(stride_zyx) != 3 or any(int(value) != value or value <= 0 for value in stride_zyx):
        raise ValueError("stride_zyx must contain three positive integers")
    stride_zyx = tuple(int(value) for value in stride_zyx)
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("radius_um must be finite and positive")
    if not np.isfinite(depth_step_um) or depth_step_um <= 0:
        raise ValueError("depth_step_um must be finite and positive")
    if not np.isfinite(max_depth_um) or max_depth_um <= depth_step_um:
        raise ValueError("max_depth_um must be finite and exceed one depth step")
    if projection_chunk_size < 1 or geodesic_batch_size < 1:
        raise ValueError("projection and geodesic batch sizes must be positive")

    source_array, dataset_path, source_scale, source_origin = _source_geometry(source_zarr, level)
    final_z = source_array.shape[0] if z_stop is None else z_stop
    if not 0 <= z_start < final_z <= source_array.shape[0]:
        raise ValueError(f"invalid Z interval [{z_start}, {final_z}) for {source_array.shape[0]} planes")
    depth = np.arange(0, max_depth_um, depth_step_um, dtype=np.float64)
    if len(depth) < 2:
        raise ValueError("depth parameters must produce at least two centers")
    mask_coefficients, mask_geometry = _mask_sampler(mask_path)
    surface = load_surface(surface_path)
    vertices, faces = surface.vertices_world_xyz_um, surface.faces
    with np.load(chart_path) as chart:
        uv = np.asarray(chart["vertex_uv"], dtype=np.float64)
        if "mesh_sha256" in chart.files and str(chart["mesh_sha256"]) != sha256_file(surface_path):
            raise ValueError("chart does not belong to the supplied surface mesh")
    data, counters = _extract_profiles(
        source_array=source_array,
        source_scale_zyx_um=source_scale,
        source_origin_zyx_um=source_origin,
        mask_coefficients=mask_coefficients,
        mask_geometry=mask_geometry,
        vertices_xyz_um=vertices,
        faces=faces,
        vertex_uv=uv,
        z_start=z_start,
        z_stop=final_z,
        stride_zyx=stride_zyx,
        depth_um=depth,
        radius_um=radius_um,
        projection_chunk_size=projection_chunk_size,
        geodesic_batch_size=geodesic_batch_size,
    )

    with atomic_output_directory(output_dir) as stage:
        profile_path = stage / "profiles.npz"
        np.savez_compressed(profile_path, **data)
        sample_spacing = source_scale * np.asarray(stride_zyx)
        metadata = {
            "artifact_type": "squisher_lightsheet.surface_profiles.v1",
            "surface": str(surface_path.resolve()),
            "surface_sha256": sha256_file(surface_path),
            "chart": str(chart_path.resolve()),
            "chart_sha256": sha256_file(chart_path),
            "vertices": len(vertices),
            "source_zarr": str(source_zarr.resolve()),
            "source_level": level,
            "source_dataset": dataset_path,
            "source_shape_zyx": [int(value) for value in source_array.shape],
            "source_scale_zyx_um": source_scale.tolist(),
            "source_origin_zyx_um": source_origin.tolist(),
            "source_dtype": str(source_array.dtype),
            "source_squisher_complete": True,
            "source_root_metadata_sha256": sha256_file(source_zarr / "zarr.json"),
            "source_array_metadata_sha256": sha256_file(source_zarr / dataset_path / "zarr.json"),
            "source_identity_limit": (
                "squisher_complete and Zarr metadata hashes do not prove voxel-content identity"
            ),
            "mask": str(mask_path.resolve()),
            "mask_sha256": sha256_file(mask_path),
            "z_interval": [z_start, final_z],
            "stride_zyx": list(stride_zyx),
            "sample_spacing_zyx_um": sample_spacing.tolist(),
            "sample_volume_um3": float(np.prod(sample_spacing)),
            "depth_centers_um": [float(depth[0]), float(depth[-1]), len(depth)],
            "requested_max_depth_um_exclusive": max_depth_um,
            "depth_assignment": ("nearest center; exact closest-surface r_vz <= final center plus half-bin"),
            "maximum_retained_r_vz_um": float(depth[-1] + depth_step_um / 2),
            "projection": "exact triangle closest point using one reusable AABB per surface",
            "selection": "cubic cortex score >=0.5, tissue side, non-boundary footpoint",
            "surface_assignment": "nearest physical mesh vertex to exact footpoint",
            "tangential_average": (
                "unweighted raw sums and counts over mesh-edge shortest-path distance <= radius"
            ),
            "radius_um": radius_um,
            "support": "observed sampled-volume bins only; internal empty bins are retained",
            "counters": counters,
            "output_sha256": sha256_file(profile_path),
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return (output_dir / "metadata.json").resolve()
