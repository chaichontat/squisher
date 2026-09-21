from __future__ import annotations

import json

import numpy as np
import pytest
import SimpleITK as sitk
import zarr

from squisher_lightsheet.surface_intensity import (
    build_surface_intensity_map,
    first_inside_interval,
    project_max_intensity,
    vertex_normals,
)
from squisher_lightsheet.surface_geometry import ImageGeometry
from squisher_lightsheet.uv_surface import build_bilateral_ventricular_uv_surfaces


def test_first_inside_interval_does_not_reenter_tissue() -> None:
    inside = np.array([[True, True, False, True, True], [False, True, True, False, False]])

    np.testing.assert_array_equal(
        first_inside_interval(inside),
        [[True, True, False, False, False], [False, True, True, False, False]],
    )


def test_vertex_normals_preserve_oriented_plane_direction() -> None:
    vertices = np.array([[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 1.0, 1.0], [4.0, 0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    np.testing.assert_allclose(vertex_normals(vertices, faces), [[1.0, 0.0, 0.0]] * 4)


def test_project_max_intensity_uses_only_first_tissue_interval() -> None:
    intensity = np.zeros((1, 1, 6), dtype=np.uint16)
    intensity[0, 0, 1] = 100
    intensity[0, 0, 3] = 5
    intensity[0, 0, 4] = 7
    mask = np.zeros_like(intensity, dtype=np.uint8)
    mask[0, 0, [1, 3, 4]] = 1

    projected, valid_count, last_depth, reaches_cap = project_max_intensity(
        intensity_zyx=intensity,
        intensity_geometry=ImageGeometry(np.zeros(3), np.ones(3), np.eye(3)),
        mask_zyx=mask,
        mask_geometry=ImageGeometry(np.zeros(3), np.ones(3), np.eye(3)),
        vertices_xyz_um=np.array([[4.0, 0.0, 0.0]]),
        normals_xyz=np.array([[1.0, 0.0, 0.0]]),
        depth_um=np.arange(5, dtype=np.float64),
        batch_size=1,
    )

    np.testing.assert_array_equal(projected, [7.0])
    np.testing.assert_array_equal(valid_count, [2])
    np.testing.assert_array_equal(last_depth, [1.0])
    np.testing.assert_array_equal(reaches_cap, [False])


def test_build_surface_intensity_map_writes_bilateral_uv_artifact(tmp_path) -> None:
    mask = np.zeros((8, 8, 15), dtype=np.uint8)
    mask[:, :, 1:5] = 1
    mask[:, :, 10:14] = 1
    mask[:2, :, 1:14] = 1
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((0.001, 0.002, 0.003))
    image.SetOrigin((0.01, 0.02, 0.03))
    mask_path = tmp_path / "mask.nrrd"
    sitk.WriteImage(image, mask_path)

    surface_root = tmp_path / "surface"
    build_bilateral_ventricular_uv_surfaces(
        mask_path=mask_path,
        output_dir=surface_root,
        split_x_index=7,
        smoothing_um=0,
    )
    intensity = np.broadcast_to(np.arange(15, dtype=np.uint16), mask.shape).copy()
    source = tmp_path / "source.ome.zarr"
    group = zarr.open_group(str(source), mode="w")
    group.create_array("0", data=intensity)
    group.attrs.update(
        {
            "squisher_complete": True,
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": "/",
                        "axes": [
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [3.0, 2.0, 1.0]},
                                    {"type": "translation", "translation": [30.0, 20.0, 10.0]},
                                ],
                            }
                        ],
                    }
                ],
            },
        }
    )

    output = tmp_path / "intensity-map"
    metadata_path = build_surface_intensity_map(
        surface_root=surface_root,
        mask_path=mask_path,
        source_zarr=source,
        output_dir=output,
        level=0,
        max_depth_um=20,
        depth_step_um=1,
        uv_size=32,
        batch_size=32,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["artifact_type"] == "squisher_lightsheet.surface_intensity_map.v1"
    assert (output / "intensity-map.png").is_file()
    for half in ("low-x", "high-x"):
        assert metadata["halves"][half]["missing_vertex_values"] == 0
        with np.load(output / half / "intensity-map.npz") as result:
            assert result["uv_intensity_max"].shape == (32, 32)
            assert np.isfinite(result["vertex_intensity_max"]).all()


@pytest.mark.parametrize(
    "direction",
    [
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        np.diag([-1.0, 1.0, 1.0]),
    ],
    ids=["rotated", "reflected"],
)
def test_surface_intensity_uses_mask_direction_and_recorded_cavity_direction(
    tmp_path, direction: np.ndarray
) -> None:
    mask = np.zeros((8, 8, 15), dtype=np.uint8)
    mask[:, :, 1:5] = 1
    mask[:, :, 10:14] = 1
    mask[:2, :, 1:14] = 1
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((0.001, 0.002, 0.003))
    image.SetOrigin((0.02, 0.02, 0.02))
    image.SetDirection(tuple(direction.ravel()))
    mask_path = tmp_path / "mask.nrrd"
    sitk.WriteImage(image, mask_path)

    surface_root = tmp_path / "surface"
    build_bilateral_ventricular_uv_surfaces(
        mask_path=mask_path,
        output_dir=surface_root,
        split_x_index=7,
        smoothing_um=0,
    )
    source = tmp_path / "source.ome.zarr"
    group = zarr.open_group(str(source), mode="w")
    group.create_array("0", data=np.full((80, 80, 80), 100, dtype=np.uint16))
    group.attrs.update(
        {
            "squisher_complete": True,
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": "/",
                        "axes": [
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                                    {"type": "translation", "translation": [-20.0, -20.0, -20.0]},
                                ],
                            }
                        ],
                    }
                ],
            },
        }
    )

    output = tmp_path / "intensity-map"
    build_surface_intensity_map(
        surface_root=surface_root,
        mask_path=mask_path,
        source_zarr=source,
        output_dir=output,
        level=0,
        max_depth_um=20,
        depth_step_um=1,
        uv_size=32,
        batch_size=32,
    )

    for half in ("low-x", "high-x"):
        with np.load(output / half / "intensity-map.npz") as result:
            assert np.isfinite(result["vertex_intensity_max"]).all()
