from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
import SimpleITK as sitk
import zarr

from squisher_lightsheet.surface_geometry import SurfaceProjector
from squisher_lightsheet.surface_profiles import (
    accumulate_samples,
    aggregate_neighborhood,
    build_surface_profiles,
    intrinsic_neighborhoods,
    nearest_depth_bin,
    projection_membership,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_nearest_depth_bin_uses_centers_and_includes_last_half_bin() -> None:
    depth = np.arange(4, dtype=np.float64) * 4.8

    np.testing.assert_array_equal(
        nearest_depth_bin(np.array([0.2, 2.4, 4.0, 14.3, 16.8]), depth),
        [0, 0, 1, 3, 3],
    )
    with pytest.raises(ValueError, match="outside"):
        nearest_depth_bin(np.array([16.8001]), depth)


def test_accumulate_samples_returns_float64_sums_and_raw_counts() -> None:
    sums, counts = accumulate_samples(
        vertex_index=np.array([0, 0, 0, 1]),
        depth_index=np.array([0, 0, 2, 1]),
        intensity=np.array([1, 2, 8, 5], dtype=np.uint16),
        vertex_count=2,
        depth_count=3,
    )

    assert sums.dtype == np.float64
    assert counts.dtype == np.uint64
    np.testing.assert_array_equal(sums, [[3, 0, 8], [0, 5, 0]])
    np.testing.assert_array_equal(counts, [[2, 0, 1], [0, 1, 0]])


def test_aggregate_neighborhood_pools_sums_and_counts_and_checks_overflow() -> None:
    neighborhood = sparse.csr_matrix(np.ones((2, 2), dtype=np.uint8))
    summed, count = aggregate_neighborhood(
        neighborhood,
        raw_sum=np.array([[10.0], [40.0]], dtype=np.float64),
        raw_count=np.array([[1], [10]], dtype=np.uint64),
    )

    np.testing.assert_array_equal(summed, [[50.0], [50.0]])
    np.testing.assert_array_equal(count, [[11], [11]])
    np.testing.assert_allclose(summed / count, [[50 / 11], [50 / 11]])

    with pytest.raises(OverflowError, match="uint32"):
        aggregate_neighborhood(
            sparse.eye(1, format="csr", dtype=np.uint8),
            raw_sum=np.zeros((1, 1)),
            raw_count=np.array([[np.iinfo(np.uint32).max + 1]], dtype=np.uint64),
        )


def test_intrinsic_neighborhood_does_not_bridge_close_components() -> None:
    vertices = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.1, 0, 0],
            [0.1, 1, 0],
            [0.1, 0, 1],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])

    neighborhood = intrinsic_neighborhoods(vertices, faces, radius_um=2, batch_size=2)

    assert neighborhood[0, 1] == 1
    assert neighborhood[0, 3] == 0


def test_projection_membership_keeps_only_tissue_nonrim_points() -> None:
    vertices = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
    faces = np.array([[0, 1, 2]])
    uv = vertices[:, 1:]
    points = np.array([[-1, 0.5, 0.5], [1, 0.5, 0.5], [-1, 1, 1]], dtype=float)
    projection = SurfaceProjector(vertices, faces, uv).project(points)

    keep = projection_membership(points, projection, vertices, faces, max_depth_um=2)

    np.testing.assert_array_equal(keep, [True, False, False])


def _write_source(path: Path, dtype: np.dtype = np.dtype(np.uint16)) -> np.ndarray:
    values = np.broadcast_to(np.arange(5, dtype=dtype), (3, 3, 5)).copy()
    root = zarr.open_group(str(path), mode="w")
    root.create_array("0", data=values)
    root.attrs.update(
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
    return values


def _write_surface(path: Path) -> tuple[Path, Path]:
    vertices = np.array([[2.0, -1.0, -1.0], [2.0, 3.0, -1.0], [2.0, 3.0, 3.0], [2.0, -1.0, 3.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv = vertices[:, 1:]
    path.mkdir()
    surface_path = path / "surface.npz"
    chart_path = path / "chart.npz"
    np.savez_compressed(
        surface_path,
        vertices_xyz_um=vertices,
        vertices_world_xyz_um=vertices,
        vertices_intrinsic_xyz_um=vertices,
        faces=faces,
        image_origin_xyz_um=np.zeros(3),
        image_spacing_xyz_um=np.ones(3),
        image_direction=np.eye(3),
        original_image_origin_xyz_um=np.zeros(3),
        original_image_spacing_xyz_um=np.ones(3),
        original_image_direction=np.eye(3),
        cavity_direction_world_xyz=np.array([1.0, 0.0, 0.0]),
    )
    np.savez_compressed(
        chart_path,
        vertex_uv=uv,
        mesh_sha256=np.asarray(_sha256(surface_path)),
    )
    return surface_path, chart_path


def test_build_surface_profiles_writes_fresh_atomic_artifact_with_source_metadata(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.ome.zarr"
    _write_source(source_path)
    mask = np.zeros((3, 3, 5), dtype=np.uint8)
    mask[:, :, :2] = 1
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetSpacing((0.001, 0.001, 0.001))
    mask_path = tmp_path / "mask.nrrd"
    sitk.WriteImage(mask_image, mask_path)
    surface_path, chart_path = _write_surface(tmp_path / "surface")
    output = tmp_path / "profiles"

    metadata_path = build_surface_profiles(
        surface_path=surface_path,
        chart_path=chart_path,
        mask_path=mask_path,
        source_zarr=source_path,
        output_dir=output,
        level=0,
        stride_zyx=(1, 1, 1),
        radius_um=0.1,
        max_depth_um=3.0,
        depth_step_um=1.0,
        projection_chunk_size=4,
        geodesic_batch_size=2,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["artifact_type"] == "squisher_lightsheet.surface_profiles.v1"
    assert metadata["source_shape_zyx"] == [3, 3, 5]
    assert metadata["source_scale_zyx_um"] == [1.0, 1.0, 1.0]
    assert metadata["source_squisher_complete"] is True
    assert "do not prove voxel-content identity" in metadata["source_identity_limit"]
    assert metadata["sample_volume_um3"] == 1.0
    assert metadata["maximum_retained_r_vz_um"] == 2.5
    assert metadata["surface"] == str(surface_path.resolve())
    assert metadata["chart"] == str(chart_path.resolve())
    assert set(path.name for path in output.iterdir()) == {"profiles.npz", "metadata.json"}
    with np.load(output / "profiles.npz") as profiles:
        assert profiles["intensity"].dtype == np.float32
        assert profiles["valid_count"].dtype == np.uint32
        assert profiles["depth_um"].tolist() == [0.0, 1.0, 2.0]
        assert np.isnan(profiles["intensity"][profiles["valid_count"] == 0]).all()
        assert int(profiles["valid_count"].sum()) > 0

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_surface_profiles(
            surface_path=surface_path,
            chart_path=chart_path,
            mask_path=mask_path,
            source_zarr=source_path,
            output_dir=output,
        )


def test_build_surface_profiles_defaults_to_level_two_and_requires_uint16(tmp_path: Path) -> None:
    assert inspect.signature(build_surface_profiles).parameters["level"].default == 2
    source_path = tmp_path / "uint8.ome.zarr"
    _write_source(source_path, np.dtype(np.uint8))

    with pytest.raises(ValueError, match="3-D uint16"):
        build_surface_profiles(
            surface_path=tmp_path / "unused-surface.npz",
            chart_path=tmp_path / "unused-chart.npz",
            mask_path=tmp_path / "unused-mask.nrrd",
            source_zarr=source_path,
            output_dir=tmp_path / "profiles",
            level=0,
        )
