from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from typer.testing import CliRunner

from squisher_lightsheet.cli import app
from squisher_lightsheet.surface_geometry import parameterize_disk
from squisher_lightsheet.uv_surface import (
    ARTIFACT_TYPE,
    _exclude_long_edge_faces,
    _keep_largest_face_component,
    build_bilateral_ventricular_uv_surfaces,
    build_ventricular_uv_surface,
)


def _write_mask(path: Path, mask: np.ndarray, *, direction: np.ndarray | None = None) -> None:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing((0.001, 0.002, 0.003))
    image.SetOrigin((0.01, 0.02, 0.03))
    if direction is not None:
        image.SetDirection(tuple(direction.ravel()))
    sitk.WriteImage(image, path)


def test_parameterize_disk_has_unit_boundary_and_no_flips() -> None:
    vertices = np.array(
        [
            [-2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 1.0, 0.0],
            [-2.0, 1.0, 0.0],
            [0.0, 0.0, 0.8],
        ]
    )
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

    uv = parameterize_disk(vertices, faces)

    np.testing.assert_allclose(np.linalg.norm(uv[:4], axis=1), 1.0)
    edge_1 = uv[faces[:, 1]] - uv[faces[:, 0]]
    edge_2 = uv[faces[:, 2]] - uv[faces[:, 0]]
    area = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    assert np.all(area > 0) or np.all(area < 0)


def test_uv_surface_cli_builds_atomic_hashed_artifact(tmp_path: Path) -> None:
    mask = np.zeros((5, 6, 9), dtype=np.uint8)
    mask[:, :, 2:6] = 1
    mask_path = tmp_path / "cortex.seg.nrrd"
    _write_mask(mask_path, mask)
    output_dir = tmp_path / "surface"

    result = CliRunner().invoke(
        app,
        [
            "uv-surface",
            "--mask",
            str(mask_path),
            "--output-dir",
            str(output_dir),
            "--smoothing-um",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(output_dir / "metadata.json")
    assert {path.name for path in output_dir.iterdir()} == {
        "surface.npz",
        "surface.ply",
        "chart.npz",
        "metadata.json",
    }
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["artifact_type"] == ARTIFACT_TYPE
    assert metadata["signed_distance_smoothing_um"] == 0.0
    assert metadata["topology"]["connected_components"] == 1
    assert metadata["topology"]["euler_characteristic"] == 1
    with np.load(output_dir / "surface.npz") as surface, np.load(output_dir / "chart.npz") as chart:
        assert chart["vertex_uv"].shape == (len(surface["vertices_xyz_um"]), 2)
        assert chart["uv_units"].item() == "dimensionless"


def test_uv_surface_rejects_non_x_monotone_mask_without_output(tmp_path: Path) -> None:
    mask = np.zeros((4, 4, 10), dtype=np.uint8)
    mask[:, :, 1:3] = 1
    mask[:, :, 6:8] = 1
    mask_path = tmp_path / "two-intervals.nrrd"
    _write_mask(mask_path, mask)
    output_dir = tmp_path / "surface"

    with pytest.raises(ValueError, match="not X-monotone"):
        build_ventricular_uv_surface(mask_path=mask_path, output_dir=output_dir)

    assert not output_dir.exists()


def test_uv_surface_allows_reviewed_outermost_crossing_and_records_every_ray_stage(
    tmp_path: Path,
) -> None:
    mask = np.zeros((4, 4, 10), dtype=np.uint8)
    mask[:, :, 1:3] = 1
    mask[:, :, 6:8] = 1
    mask_path = tmp_path / "two-intervals.nrrd"
    _write_mask(mask_path, mask)
    output_dir = tmp_path / "surface"

    metadata_path = build_ventricular_uv_surface(
        mask_path=mask_path,
        output_dir=output_dir,
        smoothing_um=0,
        allow_multi_interval_rays=True,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["allow_multi_interval_rays"] is True
    assert metadata["original_binary_multi_interval_x_rays"] == 16
    assert metadata["strided_binary_multi_interval_x_rays"] == 16
    assert metadata["multi_interval_refined_rays"] > 0


def test_uv_surface_refuses_existing_output_directory(tmp_path: Path) -> None:
    mask_path = tmp_path / "unused.nrrd"
    output_dir = tmp_path / "surface"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_ventricular_uv_surface(mask_path=mask_path, output_dir=output_dir)


def test_bilateral_surface_excludes_joined_cut_cap_and_orients_medial_faces(tmp_path: Path) -> None:
    mask = np.zeros((8, 8, 15), dtype=np.uint8)
    mask[:, :, 1:5] = 1
    mask[:, :, 10:14] = 1
    mask[:2, :, 1:14] = 1
    mask_path = tmp_path / "joined-hemispheres.nrrd"
    _write_mask(mask_path, mask)
    output_dir = tmp_path / "bilateral"

    manifest_path = build_bilateral_ventricular_uv_surfaces(
        mask_path=mask_path,
        output_dir=output_dir,
        split_x_index=7,
        smoothing_um=0,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["original_cut_crossing_rays"] == 16
    assert manifest["halves"]["low-x"]["medial_face"] == "+X"
    assert manifest["halves"]["high-x"]["medial_face"] == "-X"
    for half, expected_normal_sign in (("low-x", 1), ("high-x", -1)):
        with np.load(output_dir / half / "surface.npz") as surface:
            vertices = surface["vertices_xyz_um"]
            faces = surface["faces"]
        triangles = vertices[faces]
        normals_x = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 0]
        assert np.all(np.sign(normals_x) == expected_normal_sign)
        assert manifest["halves"][half]["topology"]["euler_characteristic"] == 1


def test_bilateral_surface_rejects_out_of_bounds_split(tmp_path: Path) -> None:
    mask = np.zeros((3, 3, 5), dtype=np.uint8)
    mask[:, :, 1:4] = 1
    mask_path = tmp_path / "mask.nrrd"
    _write_mask(mask_path, mask)

    with pytest.raises(ValueError, match="split_x_index"):
        build_bilateral_ventricular_uv_surfaces(
            mask_path=mask_path,
            output_dir=tmp_path / "bilateral",
            split_x_index=5,
        )


def test_bilateral_component_policy_keeps_largest_face_component() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])

    retained_vertices, retained_faces, retained_indices, diagnostics = _keep_largest_face_component(
        vertices, faces
    )

    np.testing.assert_array_equal(retained_vertices, vertices[:4])
    np.testing.assert_array_equal(retained_faces, faces[:2])
    np.testing.assert_array_equal(retained_indices, np.arange(4))
    assert diagnostics == {
        "source_face_components": 2,
        "removed_vertices": 3,
        "removed_faces": 1,
    }


def test_bilateral_surface_preserves_exact_intrinsic_rows_after_rotation(tmp_path: Path) -> None:
    mask = np.zeros((8, 8, 15), dtype=np.uint8)
    mask[:, :, 1:5] = 1
    mask[:, :, 10:14] = 1
    mask[:2, :, 1:14] = 1
    x_angle = 0.37
    y_angle = -0.21
    rotate_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x_angle), -np.sin(x_angle)],
            [0.0, np.sin(x_angle), np.cos(x_angle)],
        ]
    )
    rotate_y = np.array(
        [
            [np.cos(y_angle), 0.0, np.sin(y_angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(y_angle), 0.0, np.cos(y_angle)],
        ]
    )
    mask_path = tmp_path / "rotated.nrrd"
    _write_mask(mask_path, mask, direction=rotate_y @ rotate_x)
    output_dir = tmp_path / "bilateral"

    build_bilateral_ventricular_uv_surfaces(
        mask_path=mask_path,
        output_dir=output_dir,
        split_x_index=7,
        smoothing_um=0,
    )

    for half in ("low-x", "high-x"):
        with np.load(output_dir / half / "surface.npz") as surface:
            intrinsic_z = surface["vertices_intrinsic_xyz_um"][:, 2]
        assert len(np.unique(intrinsic_z)) == len(np.unique(np.round(intrinsic_z, 10)))


def test_bilateral_surface_can_exclude_discontinuity_spanning_faces() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    retained, diagnostics = _exclude_long_edge_faces(vertices, faces, 10.0)

    np.testing.assert_array_equal(retained, faces[:1])
    assert diagnostics == {
        "max_face_edge_um": 10.0,
        "excluded_long_edge_faces": 1,
    }


def test_bilateral_surface_rejects_nonpositive_max_face_edge() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _exclude_long_edge_faces(np.zeros((3, 3)), np.array([[0, 1, 2]]), 0.0)
