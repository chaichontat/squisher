from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from squisher_lightsheet.surface_geometry import (
    ImageGeometry,
    SurfaceProjector,
    load_surface,
    parameterize_disk,
)
from squisher_lightsheet.uv_surface import build_ventricular_uv_surface


def _write_mask(
    path: Path,
    mask: np.ndarray,
    *,
    direction: np.ndarray,
    origin: tuple[float, float, float] = (0.01, 0.02, 0.03),
) -> None:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing((0.001, 0.002, 0.003))
    image.SetOrigin(origin)
    image.SetDirection(tuple(direction.ravel()))
    sitk.WriteImage(image, path)


def test_image_geometry_round_trips_rotated_indices() -> None:
    direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    geometry = ImageGeometry(np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0]), direction)
    index_zyx = np.array([[4.0, 5.0, 6.0]])

    intrinsic = geometry.index_zyx_to_intrinsic(index_zyx)
    world = geometry.index_zyx_to_world(index_zyx)

    np.testing.assert_allclose(intrinsic, [[6.0, 10.0, 12.0]])
    np.testing.assert_allclose(world, [[0.0, 26.0, 42.0]])
    np.testing.assert_allclose(geometry.world_to_intrinsic(world), intrinsic)


def test_rotated_and_reflected_surface_preserves_intrinsic_chart_and_world_normals(
    tmp_path: Path,
) -> None:
    mask = np.zeros((5, 6, 9), dtype=np.uint8)
    mask[:, :, 2:6] = 1
    directions = {
        "identity": np.eye(3),
        "rotated": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        "reflected": np.diag([-1.0, 1.0, 1.0]),
    }
    charts: dict[str, np.ndarray] = {}
    for name, direction in directions.items():
        mask_path = tmp_path / f"{name}.nrrd"
        output_dir = tmp_path / name
        _write_mask(mask_path, mask, direction=direction)
        build_ventricular_uv_surface(mask_path=mask_path, output_dir=output_dir, smoothing_um=0)
        surface = load_surface(output_dir / "surface.npz")
        with np.load(output_dir / "chart.npz") as chart:
            charts[name] = chart["vertex_uv"]
        np.testing.assert_allclose(
            surface.vertices_world_xyz_um,
            surface.geometry.intrinsic_to_world(surface.vertices_intrinsic_xyz_um),
        )
        triangles = surface.vertices_world_xyz_um[surface.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        assert np.all(normals @ surface.cavity_direction_world_xyz > 0)

    np.testing.assert_allclose(charts["rotated"], charts["identity"], atol=1e-14)
    np.testing.assert_allclose(charts["reflected"], charts["identity"], atol=1e-14)


def test_surface_stride_starts_at_zero_and_preserves_origin_direction(tmp_path: Path) -> None:
    mask = np.zeros((7, 7, 10), dtype=np.uint8)
    mask[:, :, 2:8] = 1
    direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mask_path = tmp_path / "mask.nrrd"
    output_dir = tmp_path / "surface"
    _write_mask(mask_path, mask, direction=direction)

    build_ventricular_uv_surface(
        mask_path=mask_path,
        output_dir=output_dir,
        smoothing_um=0,
        stride_zyx=(2, 3, 2),
    )
    surface = load_surface(output_dir / "surface.npz")

    np.testing.assert_allclose(surface.geometry.origin_xyz_um, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(surface.geometry.spacing_xyz_um, [2.0, 6.0, 6.0])
    np.testing.assert_allclose(surface.geometry.direction, direction)
    np.testing.assert_allclose(surface.source_geometry.origin_xyz_um, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(surface.source_geometry.spacing_xyz_um, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(surface.source_geometry.direction, direction)
    assert surface.vertices_intrinsic_xyz_um[:, 1].min() == 0.0
    assert surface.vertices_intrinsic_xyz_um[:, 2].min() == 0.0
    np.testing.assert_allclose(surface.vertices_intrinsic_xyz_um[:, 1] % 3.0, 0.0)
    np.testing.assert_allclose(surface.vertices_intrinsic_xyz_um[:, 2] % 3.0, 0.0)


def test_exact_projector_distinguishes_rim_and_internal_edge() -> None:
    vertices = np.array(
        [
            [-2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 1.0, 0.0],
            [-2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    uv = parameterize_disk(vertices, faces)
    projector = SurfaceProjector(vertices, faces, uv)

    result = projector.project(np.array([[0.0, -3.0, 4.0], [1.0, -0.5, 3.0]]))

    np.testing.assert_allclose(result.r_vz_um, [np.sqrt(20.0), 3.0])
    np.testing.assert_array_equal(result.feature, ["edge", "edge"])
    np.testing.assert_array_equal(result.on_surface_boundary, [True, False])
    np.testing.assert_array_equal(result.is_orthogonal, [False, True])


def test_exact_projector_restricts_candidates_and_preserves_full_face_ids() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.9, 0.0, 0.0],
            [0.9, 2.0, 0.0],
            [0.9, 0.0, 2.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    uv = vertices[:, 1:]
    point = np.array([[1.0, 0.5, 0.5]])

    full = SurfaceProjector(vertices, faces, uv).project(point)
    wall = SurfaceProjector(vertices, faces, uv, face_mask=np.array([True, False])).project(point)

    np.testing.assert_array_equal(full.face_index, [1])
    np.testing.assert_allclose(full.r_vz_um, [0.1])
    np.testing.assert_array_equal(wall.face_index, [0])
    np.testing.assert_allclose(wall.r_vz_um, [1.0])
