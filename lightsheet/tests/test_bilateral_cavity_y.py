from pathlib import Path
import json

import numpy as np
import pytest
import SimpleITK as sitk
from typer.testing import CliRunner

from squisher_lightsheet.cli import app
from squisher_lightsheet.uv_surface import build_bilateral_ventricular_uv_surfaces


@pytest.mark.parametrize("direction", [
    np.eye(3),
    np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., 1.]]),
])
def test_positive_y_cavity_preserves_anterior_wall_world_geometry_and_closed_shells(tmp_path: Path, direction):
    mask = np.zeros((10, 10, 16), np.uint8)
    mask[1:9, 2:5, 1:15] = 1
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((0.001, 0.002, 0.003))
    image.SetOrigin((0.01, 0.02, 0.03))
    image.SetDirection(tuple(direction.ravel()))
    mask_path = tmp_path / "cortex.nrrd"
    sitk.WriteImage(image, str(mask_path))
    output = tmp_path / "surface"
    result = CliRunner().invoke(app, [
        "bilateral-uv-surface", "--mask", str(mask_path),
        "--output-dir", str(output), "--split-x-index", "8",
        "--smoothing-um", "0", "--cavity-axis", "y", "--closed-shell",
    ])
    assert result.exit_code == 0, result.output or repr(result.exception)
    metadata = json.loads((output / "bilateral.json").read_text())
    assert metadata["cavity_axis"] == "y"
    assert metadata["excluded_cut_cap_refined_rays"] == 0
    for half in ("low-x", "high-x"):
        with np.load(output / half / "surface.npz") as surface:
            intrinsic = surface["vertices_intrinsic_xyz_um"]
            world, faces = surface["vertices_world_xyz_um"], surface["faces"]
            np.testing.assert_allclose(world, intrinsic @ direction.T + [10, 20, 30])
            np.testing.assert_allclose(surface["cavity_direction_world_xyz"], direction[:, 1])
            normals = np.cross(world[faces[:, 1]] - world[faces[:, 0]],
                               world[faces[:, 2]] - world[faces[:, 0]])
            assert np.all(normals @ direction[:, 1] > 0)
            assert intrinsic[:, 2].min() <= 3
            assert intrinsic[:, 2].max() >= 24
            assert intrinsic[:, 1].max() > 8.5
            if half == "low-x":
                assert intrinsic[:, 0].max() <= 7.5
            else:
                assert intrinsic[:, 0].min() >= 7.5
        assert metadata["halves"][half]["topology"]["euler_characteristic"] == 1
        with np.load(output / half / "shell.npz") as shell:
            vertices, shell_faces, cap = shell["vertices_xyz_um"], shell["faces"], shell["split_cap_faces"]
        shell_x = ((vertices - [10, 20, 30]) @ direction)[:, 0]
        edges = np.sort(shell_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
        triangles = vertices[shell_faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        assert np.einsum("ij,ij->", triangles[:, 0], normals) > 0
        assert (shell_x.max() <= 7.5 + 1e-9) if half == "low-x" else (shell_x.min() >= 7.5 - 1e-9)
        assert cap.any()
        np.testing.assert_allclose(shell_x[shell_faces[cap]], 7.5, atol=1e-9)
        assert not np.isclose(shell_x[shell_faces[~cap]], 7.5, atol=1e-9).all(axis=1).any()
        record = metadata["halves"][half]["shell"]
        assert record["topology"]["euler_characteristic"] == 2
        assert record["split_rim_vertices"] > 0 and record["split_rim_max_error_um"] < 0.01


def test_closed_shell_rejects_mask_on_array_faces(tmp_path: Path):
    mask = np.zeros((10, 10, 16), np.uint8)
    mask[:, 2:5, 1:15] = 1
    mask_path = tmp_path / "cortex.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_path))
    with pytest.raises(ValueError, match="closed_shell requires"):
        build_bilateral_ventricular_uv_surfaces(
            mask_path=mask_path, output_dir=tmp_path / "surface", split_x_index=8,
            cavity_axis="y", smoothing_um=0, closed_shell=True,
        )
    assert not (tmp_path / "surface").exists()


def test_positive_y_cavity_rejects_x_sheet_trimming(tmp_path: Path):
    with pytest.raises(ValueError, match="keep_main_sheet.*cavity_axis"):
        build_bilateral_ventricular_uv_surfaces(
            mask_path=tmp_path / "unused.nrrd", output_dir=tmp_path / "surface",
            split_x_index=8, cavity_axis="y", keep_main_sheet=True,
        )
