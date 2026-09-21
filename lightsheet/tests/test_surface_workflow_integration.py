from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr

from squisher_lightsheet.surface_boundaries import build_surface_boundaries
from squisher_lightsheet.surface_boundary_qc import build_surface_boundary_qc
from squisher_lightsheet.surface_geometry import ImageGeometry, SurfaceProjector, parameterize_disk
from squisher_lightsheet.surface_profiles import build_surface_profiles


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rotated_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    geometry = ImageGeometry(np.array([4.0, 0.0, 0.0]), np.ones(3), direction)
    wall_x = np.array([1.0, 3.0, 0.0, 3.0, 1.0])
    intrinsic = np.array(
        [[wall_x[y], float(y), z] for z in (0.0, 4.0) for y in range(5)],
        dtype=np.float64,
    )
    world = geometry.intrinsic_to_world(intrinsic)
    faces = np.array(
        [face for y in range(4) for face in ([y, y + 1, y + 6], [y, y + 6, y + 5])],
        dtype=np.int64,
    )
    normals = np.cross(
        world[faces[:, 1]] - world[faces[:, 0]],
        world[faces[:, 2]] - world[faces[:, 0]],
    )
    if np.all(normals @ direction[:, 0] < 0):
        faces = faces[:, [0, 2, 1]]
    assert np.all(
        np.cross(
            world[faces[:, 1]] - world[faces[:, 0]],
            world[faces[:, 2]] - world[faces[:, 0]],
        )
        @ direction[:, 0]
        > 0
    )

    surface_path = root / "surface.npz"
    np.savez_compressed(
        surface_path,
        vertices_xyz_um=world,
        vertices_world_xyz_um=world,
        vertices_intrinsic_xyz_um=intrinsic,
        faces=faces,
        image_origin_xyz_um=geometry.origin_xyz_um,
        image_spacing_xyz_um=geometry.spacing_xyz_um,
        image_direction=geometry.direction,
        original_image_origin_xyz_um=geometry.origin_xyz_um,
        original_image_spacing_xyz_um=geometry.spacing_xyz_um,
        original_image_direction=geometry.direction,
        cavity_direction_world_xyz=direction[:, 0],
    )
    chart_path = root / "chart.npz"
    vertex_uv = parameterize_disk(world, faces, anchor_vertices_xyz_um=intrinsic)
    np.savez_compressed(
        chart_path,
        vertex_uv=vertex_uv,
        mesh_sha256=np.asarray(_sha256(surface_path)),
    )

    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    for y, limit in enumerate(wall_x.astype(int)):
        mask[:, y, : limit + 1] = 1
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetOrigin(tuple(geometry.origin_xyz_um / 1000.0))
    mask_image.SetSpacing(tuple(geometry.spacing_xyz_um / 1000.0))
    mask_image.SetDirection(tuple(direction.ravel()))
    mask_path = root / "mask.nrrd"
    sitk.WriteImage(mask_image, mask_path)

    points = np.array([[float(x), float(y), float(z)] for z in range(5) for y in range(5) for x in range(5)])
    radial_distance = SurfaceProjector(world, faces, vertex_uv).project(points).r_vz_um
    source_values = np.zeros(125, dtype=np.uint16)
    intrinsic_points = geometry.world_to_intrinsic(points)
    intrinsic_index = np.rint(intrinsic_points).astype(int)
    inside = (
        (intrinsic_index[:, 0] >= 0)
        & (intrinsic_index[:, 0] <= wall_x[intrinsic_index[:, 1]])
        & (intrinsic_index[:, 2] >= 0)
        & (intrinsic_index[:, 2] < 5)
    )
    source_values[inside] = np.rint(
        1 + 100 * np.exp(-0.5 * ((radial_distance[inside] - 0.632) / 0.14) ** 2)
    ).astype(np.uint16)
    source_values = source_values.reshape(5, 5, 5)
    source_path = root / "source.ome.zarr"
    group = zarr.open_group(str(source_path), mode="w")
    group.create_array("0", data=source_values)
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
                                    {"type": "translation", "translation": [0.0, 0.0, 0.0]},
                                ],
                            }
                        ],
                    }
                ],
            },
        }
    )
    return surface_path, chart_path, mask_path, source_path


def test_rotated_surface_profiles_boundaries_and_qc(tmp_path: Path) -> None:
    surface, chart, mask, source = _write_rotated_fixture(tmp_path)
    profiles = tmp_path / "profiles"
    boundaries = tmp_path / "boundaries"
    qc = tmp_path / "qc"

    build_surface_profiles(
        surface_path=surface,
        chart_path=chart,
        mask_path=mask,
        source_zarr=source,
        output_dir=profiles,
        level=0,
        stride_zyx=(1, 1, 1),
        radius_um=20.0,
        max_depth_um=1.7,
        depth_step_um=1 / np.sqrt(10),
    )
    build_surface_boundaries(
        profiles_path=profiles / "profiles.npz",
        surface_path=surface,
        output_dir=boundaries,
    )
    metadata_path = build_surface_boundary_qc(
        surface_path=surface,
        chart_path=chart,
        boundaries_path=boundaries / "boundaries.npz",
        mask_path=mask,
        source_zarr=source,
        output_dir=qc,
        source_level=0,
        intrinsic_z_indices=[1, 2, 3],
        grid_stride=1,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["reference_geometry"]["direction_determinant"] == 1.0
    assert len(metadata["sections"]) == 3
    assert set(path.name for path in qc.iterdir()) == {
        "boundary-qc.png",
        "boundary-qc.pdf",
        "boundary-qc.svg",
        "metadata.json",
    }
