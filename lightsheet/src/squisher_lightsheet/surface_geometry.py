"""Shared geometry, charting, and exact triangle-surface projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import igl
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ImageGeometry:
    """A 3-D image's intrinsic-physical to world-physical transform."""

    origin_xyz_um: FloatArray
    spacing_xyz_um: FloatArray
    direction: FloatArray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_xyz_um, dtype=np.float64)
        spacing = np.asarray(self.spacing_xyz_um, dtype=np.float64)
        direction = np.asarray(self.direction, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("origin_xyz_um must be finite with shape (3,)")
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("spacing_xyz_um must be finite and positive with shape (3,)")
        if direction.shape != (3, 3) or not np.isfinite(direction).all():
            raise ValueError("direction must be finite with shape (3, 3)")
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-10, rtol=1e-10):
            raise ValueError("direction must be orthonormal")
        determinant = float(np.linalg.det(direction))
        if not np.isclose(abs(determinant), 1.0, atol=1e-10, rtol=1e-10):
            raise ValueError("direction determinant must be +1 or -1")
        object.__setattr__(self, "origin_xyz_um", origin)
        object.__setattr__(self, "spacing_xyz_um", spacing)
        object.__setattr__(self, "direction", direction)

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.direction))

    @classmethod
    def from_sitk(cls, image: object) -> "ImageGeometry":
        """Read millimetre-valued SimpleITK geometry as micrometres."""
        if getattr(image, "GetDimension")() != 3:
            raise ValueError("image must be three-dimensional")
        return cls(
            np.asarray(getattr(image, "GetOrigin")(), dtype=np.float64) * 1000.0,
            np.asarray(getattr(image, "GetSpacing")(), dtype=np.float64) * 1000.0,
            np.asarray(getattr(image, "GetDirection")(), dtype=np.float64).reshape(3, 3),
        )

    def intrinsic_to_world(self, points_xyz_um: npt.ArrayLike) -> FloatArray:
        points = np.asarray(points_xyz_um, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("intrinsic points must be finite with final dimension 3")
        return points @ self.direction.T + self.origin_xyz_um

    def world_to_intrinsic(self, points_xyz_um: npt.ArrayLike) -> FloatArray:
        points = np.asarray(points_xyz_um, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("world points must be finite with final dimension 3")
        return (points - self.origin_xyz_um) @ self.direction

    def index_zyx_to_intrinsic(self, index_zyx: npt.ArrayLike) -> FloatArray:
        indices = np.asarray(index_zyx, dtype=np.float64)
        if indices.shape[-1:] != (3,) or not np.isfinite(indices).all():
            raise ValueError("index_zyx must be finite with final dimension 3")
        return indices[..., ::-1] * self.spacing_xyz_um

    def index_zyx_to_world(self, index_zyx: npt.ArrayLike) -> FloatArray:
        return self.intrinsic_to_world(self.index_zyx_to_intrinsic(index_zyx))

    def intrinsic_to_index_zyx(self, points_xyz_um: npt.ArrayLike) -> FloatArray:
        points = np.asarray(points_xyz_um, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("intrinsic points must be finite with final dimension 3")
        return (points / self.spacing_xyz_um)[..., ::-1]

    def world_to_index_zyx(self, points_xyz_um: npt.ArrayLike) -> FloatArray:
        return self.intrinsic_to_index_zyx(self.world_to_intrinsic(points_xyz_um))

    def strided(self, stride_zyx: npt.ArrayLike) -> "ImageGeometry":
        stride = _stride_array(stride_zyx)
        return ImageGeometry(
            self.origin_xyz_um,
            self.spacing_xyz_um * stride[::-1],
            self.direction,
        )


def _stride_array(stride_zyx: npt.ArrayLike) -> IntArray:
    values = np.asarray(stride_zyx)
    if values.shape != (3,) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 1):
        raise ValueError("stride_zyx must contain three positive integers")
    return values.astype(np.int64, copy=False)


@dataclass(frozen=True)
class SurfaceProjection:
    face_index: IntArray
    barycentric: FloatArray
    closest_xyz_um: FloatArray
    uv: FloatArray
    r_vz_um: FloatArray
    on_surface_boundary: npt.NDArray[np.bool_]
    is_orthogonal: npt.NDArray[np.bool_]
    feature: npt.NDArray[np.str_]


@dataclass(frozen=True)
class SurfaceMesh:
    vertices_world_xyz_um: FloatArray
    vertices_intrinsic_xyz_um: FloatArray
    faces: IntArray
    geometry: ImageGeometry
    source_geometry: ImageGeometry
    cavity_direction_world_xyz: FloatArray


def load_surface(path: Path) -> SurfaceMesh:
    """Load a geometry-complete surface artifact."""
    with np.load(path) as artifact:
        required = {
            "vertices_world_xyz_um",
            "vertices_intrinsic_xyz_um",
            "faces",
            "image_origin_xyz_um",
            "image_spacing_xyz_um",
            "image_direction",
            "original_image_origin_xyz_um",
            "original_image_spacing_xyz_um",
            "original_image_direction",
            "cavity_direction_world_xyz",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"surface artifact is missing geometry fields: {sorted(missing)}")
        geometry = ImageGeometry(
            artifact["image_origin_xyz_um"],
            artifact["image_spacing_xyz_um"],
            artifact["image_direction"],
        )
        source_geometry = ImageGeometry(
            artifact["original_image_origin_xyz_um"],
            artifact["original_image_spacing_xyz_um"],
            artifact["original_image_direction"],
        )
        vertices_world, faces = mesh_arrays(artifact["vertices_world_xyz_um"], artifact["faces"])
        vertices_intrinsic = np.asarray(artifact["vertices_intrinsic_xyz_um"], dtype=np.float64)
        direction = np.asarray(artifact["cavity_direction_world_xyz"], dtype=np.float64)
    if vertices_intrinsic.shape != vertices_world.shape or not np.isfinite(vertices_intrinsic).all():
        raise ValueError("vertices_intrinsic_xyz_um must be finite and match world vertices")
    if (
        direction.shape != (3,)
        or not np.isfinite(direction).all()
        or not np.isclose(np.linalg.norm(direction), 1.0)
    ):
        raise ValueError("cavity_direction_world_xyz must be a finite unit vector")
    if not np.allclose(
        geometry.intrinsic_to_world(vertices_intrinsic),
        vertices_world,
        atol=1e-8,
        rtol=1e-10,
    ):
        raise ValueError("world and intrinsic vertices disagree with the recorded geometry")
    if not np.allclose(geometry.origin_xyz_um, source_geometry.origin_xyz_um) or not np.allclose(
        geometry.direction, source_geometry.direction
    ):
        raise ValueError("working geometry must preserve the source origin and direction")
    stride_xyz = geometry.spacing_xyz_um / source_geometry.spacing_xyz_um
    if np.any(stride_xyz < 1) or not np.allclose(stride_xyz, np.rint(stride_xyz)):
        raise ValueError("working spacing must be a positive integer stride of source spacing")
    return SurfaceMesh(
        vertices_world,
        vertices_intrinsic,
        faces,
        geometry,
        source_geometry,
        direction,
    )


def mesh_arrays(vertices_xyz_um: npt.ArrayLike, faces: npt.ArrayLike) -> tuple[FloatArray, IntArray]:
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    face_values = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices_xyz_um must be finite with shape (n_vertices, 3)")
    if face_values.ndim != 2 or face_values.shape[1] != 3 or len(face_values) == 0:
        raise ValueError("faces must be a nonempty array with shape (n_faces, 3)")
    if not (
        np.issubdtype(face_values.dtype, np.integer)
        or (
            np.issubdtype(face_values.dtype, np.floating)
            and np.isfinite(face_values).all()
            and np.equal(face_values, np.floor(face_values)).all()
        )
    ):
        raise ValueError("faces must contain integer vertex indices")
    triangles = face_values.astype(np.int64, copy=False)
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("faces contain an out-of-range vertex index")
    corners = vertices[triangles]
    twice_area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    if np.any(twice_area == 0):
        raise ValueError("faces must not contain geometrically degenerate triangles")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(triangles)


def mesh_topology(vertex_count: int, faces: npt.ArrayLike) -> dict[str, int]:
    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.sort(
        np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    graph = sparse.coo_matrix(
        (
            np.ones(2 * len(unique_edges), dtype=np.uint8),
            (
                np.concatenate((unique_edges[:, 0], unique_edges[:, 1])),
                np.concatenate((unique_edges[:, 1], unique_edges[:, 0])),
            ),
        ),
        shape=(vertex_count, vertex_count),
    )
    component_count, _ = connected_components(graph, directed=False)
    return {
        "vertices": vertex_count,
        "faces": len(triangles),
        "edges": len(unique_edges),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "connected_components": int(component_count),
        "euler_characteristic": int(vertex_count - len(unique_edges) + len(triangles)),
    }


def disk_boundary(vertices_xyz_um: npt.ArrayLike, faces: npt.ArrayLike) -> IntArray:
    vertices, triangles = mesh_arrays(vertices_xyz_um, faces)
    if not igl.is_edge_manifold(triangles)[0] or not np.all(igl.is_vertex_manifold(triangles)):
        raise ValueError("surface must be an oriented edge- and vertex-manifold")
    component_count, _ = igl.facet_components(triangles)
    loops = igl.boundary_loop_all(triangles)
    if component_count != 1 or len(loops) != 1 or len(loops[0]) < 3:
        raise ValueError("surface must be connected with exactly one boundary loop")
    if igl.euler_characteristic(triangles) != 1:
        raise ValueError("surface must have disk topology (Euler characteristic 1)")
    boundary = np.asarray(loops[0], dtype=np.int64)
    start = min(range(len(boundary)), key=lambda index: tuple(vertices[boundary[index]]))
    boundary = np.roll(boundary, -start)
    if tuple(vertices[boundary[-1]]) < tuple(vertices[boundary[1]]):
        boundary = np.concatenate((boundary[:1], boundary[:0:-1]))
    return np.ascontiguousarray(boundary)


def parameterize_disk(
    vertices_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    *,
    anchor_vertices_xyz_um: npt.ArrayLike | None = None,
) -> FloatArray:
    """Create a Tutte chart, optionally anchored in a separate coordinate frame."""
    vertices, triangles = mesh_arrays(vertices_xyz_um, faces)
    anchors = (
        vertices if anchor_vertices_xyz_um is None else np.asarray(anchor_vertices_xyz_um, dtype=np.float64)
    )
    if anchors.shape != vertices.shape or not np.isfinite(anchors).all():
        raise ValueError("anchor_vertices_xyz_um must be finite and match vertices")
    boundary = disk_boundary(anchors, triangles)
    boundary_uv = igl.map_vertices_to_circle(np.asfortranarray(anchors), boundary.astype(np.int32))
    adjacency = igl.adjacency_matrix(triangles).astype(np.float64).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degree == 0):
        raise ValueError("surface contains an unreferenced vertex")
    laplacian = sparse.diags(degree) - adjacency
    is_boundary = np.zeros(len(vertices), dtype=bool)
    is_boundary[boundary] = True
    interior = np.flatnonzero(~is_boundary)
    uv = np.empty((len(vertices), 2), dtype=np.float64)
    uv[boundary] = boundary_uv
    if len(interior):
        rhs = -laplacian[interior][:, boundary] @ boundary_uv
        uv[interior] = spsolve(laplacian[interior][:, interior], rhs)
    edge_1 = uv[triangles[:, 1]] - uv[triangles[:, 0]]
    edge_2 = uv[triangles[:, 2]] - uv[triangles[:, 0]]
    signed_area = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    tolerance = np.finfo(np.float64).eps * 100
    if np.any(np.abs(signed_area) <= tolerance) or not (np.all(signed_area > 0) or np.all(signed_area < 0)):
        raise ValueError("surface parameterization contains a degenerate or flipped triangle")
    return uv


def validate_manifold_disk(
    vertices_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    *,
    anchor_vertices_xyz_um: npt.ArrayLike | None = None,
) -> tuple[dict[str, int], FloatArray]:
    vertices, triangles = mesh_arrays(vertices_xyz_um, faces)
    topology = mesh_topology(len(vertices), triangles)
    vertex_uv = parameterize_disk(vertices, triangles, anchor_vertices_xyz_um=anchor_vertices_xyz_um)
    return topology, vertex_uv


def boundary_edges(faces: npt.ArrayLike) -> set[tuple[int, int]]:
    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.sort(
        np.concatenate((triangles[:, [1, 2]], triangles[:, [2, 0]], triangles[:, [0, 1]])), axis=1
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {tuple(edge) for edge in unique[counts == 1]}


class SurfaceProjector:
    """An exact closest-triangle projector whose AABB tree is built once."""

    def __init__(
        self,
        vertices_xyz_um: npt.ArrayLike,
        faces: npt.ArrayLike,
        vertex_uv: npt.ArrayLike,
        *,
        tolerance: float = 1e-10,
    ) -> None:
        self.vertices, self.faces = mesh_arrays(vertices_xyz_um, faces)
        self.vertex_uv = np.asarray(vertex_uv, dtype=np.float64)
        if self.vertex_uv.shape != (len(self.vertices), 2) or not np.isfinite(self.vertex_uv).all():
            raise ValueError("vertex_uv must be finite with shape (n_vertices, 2)")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self.tolerance = tolerance
        self._tree = igl.AABB()
        self._tree.init(self.vertices, self.faces)
        self._boundary_edges = boundary_edges(self.faces)
        self._boundary_vertices = np.fromiter(
            {vertex for edge in self._boundary_edges for vertex in edge}, dtype=np.int64
        )

    def project(
        self,
        points_xyz_um: npt.ArrayLike,
        *,
        chunk_size: int = 250_000,
    ) -> SurfaceProjection:
        vertices = self.vertices
        triangles = self.faces
        uv_vertices = self.vertex_uv
        tolerance = self.tolerance
        points = np.asarray(points_xyz_um, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_xyz_um must be finite with shape (n_points, 3)")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        squared_distance = np.empty(len(points), dtype=np.float64)
        face_index = np.empty(len(points), dtype=np.int64)
        closest = np.empty_like(points)
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            sqr_d, selected, closest_chunk = self._tree.squared_distance(
                vertices, triangles, np.ascontiguousarray(points[start:stop])
            )
            squared_distance[start:stop] = sqr_d
            face_index[start:stop] = selected
            closest[start:stop] = closest_chunk
        selected_faces = triangles[face_index]
        corners = vertices[selected_faces]
        barycentric = igl.barycentric_coordinates(closest, corners[:, 0], corners[:, 1], corners[:, 2])
        barycentric[np.abs(barycentric) <= tolerance] = 0.0
        barycentric /= barycentric.sum(axis=1, keepdims=True)
        uv = np.einsum("ni,nij->nj", barycentric, uv_vertices[selected_faces])
        zero_count = np.count_nonzero(barycentric <= tolerance, axis=1)
        feature = np.full(len(points), "face", dtype="<U6")
        feature[zero_count == 1] = "edge"
        feature[zero_count >= 2] = "vertex"
        on_boundary = np.zeros(len(points), dtype=bool)
        for bary_index, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
            candidates = np.flatnonzero(barycentric[:, bary_index] <= tolerance)
            on_boundary[candidates] |= np.fromiter(
                (
                    tuple(sorted((int(selected_faces[i, a]), int(selected_faces[i, b]))))
                    in self._boundary_edges
                    for i in candidates
                ),
                dtype=bool,
                count=len(candidates),
            )
        vertex_rows = np.flatnonzero(feature == "vertex")
        projected_vertices = selected_faces[vertex_rows, np.argmax(barycentric[vertex_rows], axis=1)]
        on_boundary[vertex_rows] |= np.isin(projected_vertices, self._boundary_vertices)
        displacement = points - closest
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        normal_component = np.einsum("ij,ij->i", displacement, normals)[:, None] * normals
        tangent_norm = np.linalg.norm(displacement - normal_component, axis=1)
        distance = np.sqrt(np.maximum(squared_distance, 0.0))
        is_orthogonal = tangent_norm <= tolerance * np.maximum(1.0, distance)
        return SurfaceProjection(
            face_index, barycentric, closest, uv, distance, on_boundary, is_orthogonal, feature
        )
