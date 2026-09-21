from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import zarr

from squisher_lightsheet.surface_boundary_qc import (
    build_surface_boundary_qc,
    evaluate_boundary_fields,
    projection_continuity_mask,
)


def _oriented_plane() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array([[2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 2.0], [2.0, 0.0, 2.0]])
    # Reversed winding makes the reflected intrinsic +X direction point along world -X.
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    uv = vertices[:, 1:]
    return vertices, faces, uv


def test_evaluate_boundary_fields_uses_implicit_distance_and_per_edge_validity() -> None:
    vertices, faces, uv = _oriented_plane()
    points = np.array([[3.0, 1.0, 1.0], [3.5, 1.0, 1.0], [4.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    all_vertices = np.ones(4, dtype=bool)
    outer_vertices = np.zeros(4, dtype=bool)

    fields, counts = evaluate_boundary_fields(
        points,
        vertices_world_xyz_um=vertices,
        faces=faces,
        vertex_uv=uv,
        cutoffs_um={"inner": np.ones(4), "outer": np.full(4, 1.5)},
        eligible_vertices={"inner": all_vertices, "outer": outer_vertices},
        inside_cortex=np.array([True, True, False, True]),
    )

    np.testing.assert_allclose(fields["inner"][:2], [0.0, 0.5], atol=1e-12)
    assert np.isnan(fields["inner"][2])  # outside cortex
    assert np.isnan(fields["inner"][3])  # wrong (cavity) side
    assert np.isnan(fields["outer"]).all()  # selected face includes an unsupported vertex
    assert counts == {
        "points": 4,
        "rim": 0,
        "tissue_side": 3,
        "inside_cortex": 3,
        "projection_continuous": 4,
        "valid_inner": 2,
        "valid_outer": 0,
    }


def test_projection_continuity_mask_rejects_both_sides_of_a_branch_jump() -> None:
    closest = np.zeros((2, 3, 3), dtype=np.float64)
    closest[..., 0] = np.arange(3)
    closest[1, 2, 2] = 100.0

    continuous = projection_continuity_mask(closest.reshape(-1, 3), (2, 3), maximum_step_um=10.0).reshape(
        2, 3
    )

    np.testing.assert_array_equal(continuous, [[True, True, False], [True, False, False]])


def _write_fixture(root: Path, *, source_dtype: np.dtype = np.dtype(np.uint16)) -> dict[str, Path]:
    mask_path = root / "mask.nrrd"
    mask = np.ones((3, 4, 5), dtype=np.uint8)
    mask[:, 0, :] = 0
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetSpacing((0.001, 0.001, 0.001))
    mask_image.SetOrigin((0.004, 0.0, 0.0))
    mask_image.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(mask_image, mask_path)

    vertices, faces, uv = _oriented_plane()
    surface_path = root / "surface.npz"
    np.savez_compressed(
        surface_path,
        vertices_xyz_um=vertices,
        vertices_world_xyz_um=vertices,
        vertices_intrinsic_xyz_um=np.column_stack((4.0 - vertices[:, 0], vertices[:, 1:])),
        faces=faces,
        image_origin_xyz_um=np.array([4.0, 0.0, 0.0]),
        image_spacing_xyz_um=np.array([2.0, 1.0, 1.0]),
        image_direction=np.diag([-1.0, 1.0, 1.0]),
        original_image_origin_xyz_um=np.array([4.0, 0.0, 0.0]),
        original_image_spacing_xyz_um=np.ones(3),
        original_image_direction=np.diag([-1.0, 1.0, 1.0]),
        cavity_direction_world_xyz=np.array([-1.0, 0.0, 0.0]),
    )
    chart_path = root / "chart.npz"
    surface_hash = hashlib.sha256(surface_path.read_bytes()).hexdigest()
    np.savez_compressed(chart_path, vertex_uv=uv, mesh_sha256=np.asarray(surface_hash))
    boundary_path = root / "boundaries.npz"
    support = np.ones(4, dtype=bool)
    np.savez_compressed(
        boundary_path,
        raw_inner_um=np.ones(4),
        raw_outer_um=np.full(4, 1.5),
        inner_um=np.ones(4),
        outer_um=np.full(4, 1.5),
        direct_supported_inner=support,
        direct_supported_outer=np.zeros(4, dtype=bool),
        direct_supported_joint=np.zeros(4, dtype=bool),
        wall_support=support,
        estimated_inner=support,
        estimated_outer=support,
        recovered_missing_measurement_inner=np.zeros(4, dtype=bool),
        recovered_missing_measurement_outer=np.array([False, True, False, False]),
    )
    boundary_hash = hashlib.sha256(boundary_path.read_bytes()).hexdigest()
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.surface_boundaries.v1",
                "surface_sha256": surface_hash,
                "outputs": {"boundaries.npz": boundary_hash},
            }
        )
        + "\n"
    )

    source_path = root / "source.ome.zarr"
    values = (np.indices((3, 4, 5)).sum(axis=0) + 1).astype(source_dtype)
    group = zarr.open_group(str(source_path), mode="w")
    group.create_array("0", data=values)
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
    return {
        "surface_path": surface_path,
        "chart_path": chart_path,
        "boundaries_path": boundary_path,
        "mask_path": mask_path,
        "source_zarr": source_path,
    }


def test_build_surface_boundary_qc_records_reflected_geometry_and_distinct_validity(tmp_path: Path) -> None:
    inputs = _write_fixture(tmp_path)
    output = tmp_path / "qc"
    with np.load(inputs["surface_path"]) as surface:
        # The mesh uses a 2x intrinsic-X working stride while QC samples the source mask grid.
        np.testing.assert_array_equal(surface["image_spacing_xyz_um"], [2.0, 1.0, 1.0])
        np.testing.assert_array_equal(surface["original_image_spacing_xyz_um"], [1.0, 1.0, 1.0])

    metadata_path = build_surface_boundary_qc(
        **inputs,
        output_dir=output,
        source_level=0,
        intrinsic_z_positions_um=[1.2],
        grid_stride=1,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["artifact_type"] == "squisher_lightsheet.surface_boundary_qc.v1"
    assert metadata["reference_geometry"]["direction_determinant"] == pytest.approx(-1.0)
    section = metadata["sections"][0]
    assert section["requested_position_z_um"] == pytest.approx(1.2)
    assert section["actual_index_z"] == 1
    assert section["actual_position_z_um"] == pytest.approx(1.0)
    assert section["modes"]["direct"]["contour_present"] == {"inner": True, "outer": False}
    assert section["modes"]["recovered"]["contour_present"] == {"inner": True, "outer": True}
    assert section["modes"]["direct"]["contour_components"] == {"inner": 1, "outer": 0}
    assert section["modes"]["recovered"]["contour_components"] == {"inner": 1, "outer": 1}
    assert section["modes"]["recovered"]["counts"]["inside_cortex"] == 15
    assert metadata["image_extent_intrinsic_xy_um_pixel_edges"] == [-0.5, 4.5, 3.5, -0.5]
    assert "binary mask" in metadata["reference_mask_overlay"]
    assert "no slab projection" in metadata["section_sampling"]
    assert set(path.name for path in output.iterdir()) == {
        "boundary-qc.png",
        "boundary-qc.pdf",
        "boundary-qc.svg",
        "metadata.json",
    }


def test_build_surface_boundary_qc_requires_one_section_coordinate_form(tmp_path: Path) -> None:
    inputs = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="exactly one"):
        build_surface_boundary_qc(
            **inputs,
            output_dir=tmp_path / "qc",
            source_level=0,
            intrinsic_z_indices=[1],
            intrinsic_z_positions_um=[1.0],
        )
    assert not (tmp_path / "qc").exists()


def test_build_surface_boundary_qc_defaults_to_level_two_and_rejects_non_uint16(
    tmp_path: Path,
) -> None:
    assert inspect.signature(build_surface_boundary_qc).parameters["source_level"].default == 2
    inputs = _write_fixture(tmp_path, source_dtype=np.dtype(np.float32))

    with pytest.raises(ValueError, match="3-D uint16"):
        build_surface_boundary_qc(
            **inputs,
            output_dir=tmp_path / "qc",
            source_level=0,
            intrinsic_z_indices=[1],
        )


def test_build_surface_boundary_qc_refuses_existing_output(tmp_path: Path) -> None:
    inputs = _write_fixture(tmp_path)
    output = tmp_path / "qc"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_surface_boundary_qc(
            **inputs,
            output_dir=output,
            source_level=0,
            intrinsic_z_indices=[1],
        )


def test_build_surface_boundary_qc_rejects_chart_for_another_surface(tmp_path: Path) -> None:
    inputs = _write_fixture(tmp_path)
    np.savez_compressed(
        inputs["chart_path"],
        vertex_uv=_oriented_plane()[2],
        mesh_sha256=np.asarray("0" * 64),
    )

    with pytest.raises(ValueError, match="chart mesh_sha256"):
        build_surface_boundary_qc(
            **inputs,
            output_dir=tmp_path / "qc",
            source_level=0,
            intrinsic_z_indices=[1],
        )
    assert not (tmp_path / "qc").exists()


def test_build_surface_boundary_qc_rejects_boundaries_for_another_surface(tmp_path: Path) -> None:
    inputs = _write_fixture(tmp_path)
    (inputs["boundaries_path"].parent / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.surface_boundaries.v1",
                "surface_sha256": "0" * 64,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="boundary metadata surface_sha256"):
        build_surface_boundary_qc(
            **inputs,
            output_dir=tmp_path / "qc",
            source_level=0,
            intrinsic_z_indices=[1],
        )
    assert not (tmp_path / "qc").exists()


def test_build_surface_boundary_qc_rolls_back_failed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_fixture(tmp_path)
    output = tmp_path / "qc"
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig

    def fail_on_pdf(self, path, *args, **kwargs):
        if Path(path).suffix == ".pdf":
            raise RuntimeError("synthetic export failure")
        return original_savefig(self, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail_on_pdf)
    with pytest.raises(RuntimeError, match="synthetic export failure"):
        build_surface_boundary_qc(
            **inputs,
            output_dir=output,
            source_level=0,
            intrinsic_z_indices=[1],
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".qc-*"))
