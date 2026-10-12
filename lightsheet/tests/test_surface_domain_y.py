from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet.surface_geometry import ImageGeometry, SurfaceMesh, surface_projection_domain


@pytest.mark.parametrize('direction', [np.eye(3), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), np.diag([-1., 1., 1.])])
def test_positive_y_wall_stops_at_turnarounds(tmp_path: Path, direction: np.ndarray) -> None:
    y = np.array([4., 6., 4., 2., 4., 6., 4.])
    vertices = np.array([[x, height, z] for x in range(3) for z, height in enumerate(y)])
    faces = np.array([[row * 7 + z, row * 7 + z + 1, (row + 1) * 7 + z]
                      for row in range(2) for z in range(6)])
    geometry = ImageGeometry(np.array([10., 20., 30.]), np.ones(3), direction)
    surface = SurfaceMesh(geometry.intrinsic_to_world(vertices), vertices, faces, geometry, geometry, direction[:, 1])
    domain = surface_projection_domain(tmp_path / 'surface.npz', surface, name='ventricular-wall')
    expected = np.tile([False, True, True, True, True, True, False], 3)
    np.testing.assert_array_equal(domain.vertex_mask, expected)
    np.testing.assert_array_equal(domain.face_mask, expected[faces].all(axis=1))
    assert domain.rule == 'intrinsic per-X wall between cavity-Y turnarounds'
    full = surface_projection_domain(tmp_path / 'surface.npz', surface, name='full')
    assert full.vertex_mask.all() and full.face_mask.all()


def test_wall_domain_rejects_unsupported_cavity_direction(tmp_path: Path) -> None:
    geometry = ImageGeometry(np.zeros(3), np.ones(3), np.eye(3))
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    surface = SurfaceMesh(vertices, vertices, np.array([[0, 1, 2]]), geometry, geometry, np.array([0., 0., 1.]))
    with pytest.raises(ValueError, match='cavity direction'):
        surface_projection_domain(tmp_path / 'surface.npz', surface, name='ventricular-wall')
