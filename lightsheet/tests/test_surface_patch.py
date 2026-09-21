from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from squisher_lightsheet.artifact_io import sha256_file
from squisher_lightsheet.surface_geometry import ImageGeometry, load_surface
from squisher_lightsheet.uv_surface import (
    ARTIFACT_TYPE,
    MIN_PATCH_TRIANGLE_QUALITY,
    _splice_surface_patch,
    _validate_surface,
    _write_surface_bundle,
    build_repaired_ventricular_surface,
)


def _fixture(
    tmp_path: Path, *, reflection: bool = False, wrong_side: bool = False
) -> tuple[Path, Path, dict]:
    mask = np.zeros((7, 7, 9), dtype=np.uint8)
    mask[:, :, 6:] = 1
    if wrong_side:
        mask = 1 - mask
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((0.001, 0.001, 0.001))
    image.SetOrigin((0.011, 0.012, 0.013))
    if reflection:
        image.SetDirection((-1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    mask_path = tmp_path / "mask.nrrd"
    sitk.WriteImage(image, mask_path)
    geometry = ImageGeometry.from_sitk(image)
    intrinsic = np.asarray([[5.5, 1, 1], [5.5, 5, 1], [5.5, 5, 5], [5.5, 1, 5]])
    faces = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    replacement = np.asarray([[0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]], dtype=np.int64)
    if reflection:
        faces = faces[:, [0, 2, 1]]
        replacement = replacement[:, [0, 2, 1]]
    world = geometry.intrinsic_to_world(intrinsic)
    _, uv = _validate_surface(world, faces, "fixture", anchor_vertices=intrinsic)
    source = tmp_path / "source"
    source.mkdir()
    outputs = _write_surface_bundle(
        source, world, intrinsic, faces, uv, geometry, geometry, geometry.direction[:, 0]
    )
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": ARTIFACT_TYPE,
                "outputs": outputs,
                "mask_sha256": sha256_file(mask_path),
                "stride_zyx": [1, 1, 1],
                "signed_distance_smoothing_um": 0,
            }
        )
    )
    patch = {
        "parent_surface_sha256": np.asarray(outputs["surface.npz"]),
        "mask_sha256": np.asarray(sha256_file(mask_path)),
        "replaced_face_indices": np.asarray([0, 1], dtype=np.int64),
        "new_vertices_intrinsic_xyz_um": np.asarray([[5.5, 3, 3]]),
        "replacement_faces": replacement,
    }
    return source / "surface.npz", mask_path, patch


@pytest.mark.parametrize("reflection", [False, True])
def test_reviewed_patch_preserves_parent_and_accepts_negative_intrinsic_x_normals(
    tmp_path: Path, reflection: bool
) -> None:
    surface_path, mask_path, arrays = _fixture(tmp_path, reflection=reflection)
    patch_path = tmp_path / "reviewed-patch.npz"
    np.savez_compressed(patch_path, **arrays)
    output = tmp_path / "repaired"

    metadata_path = build_repaired_ventricular_surface(
        surface_path=surface_path,
        mask_path=mask_path,
        patch_path=patch_path,
        output_dir=output,
        max_surface_error_um=0.1,
    )

    original, repaired = load_surface(surface_path), load_surface(output / "surface.npz")
    np.testing.assert_array_equal(repaired.vertices_world_xyz_um[:4], original.vertices_world_xyz_um)
    np.testing.assert_array_equal(repaired.vertices_intrinsic_xyz_um[:4], original.vertices_intrinsic_xyz_um)
    np.testing.assert_array_equal(repaired.faces, arrays["replacement_faces"])
    metadata = json.loads(metadata_path.read_text())
    assert metadata["repair"]["appended_vertices"] == 1
    assert metadata["repair"]["seam_vertex_ids"] == [0, 1, 2, 3]
    assert metadata["repair"]["minimum_normal_outward_cosine"] > 0.99
    assert metadata["repair"]["minimum_patch_triangle_quality"] >= MIN_PATCH_TRIANGLE_QUALITY
    assert metadata["repair"]["self_intersections"] == 0
    assert sha256_file(output / "patch.npz") == sha256_file(patch_path)
    with np.load(output / "chart.npz") as chart:
        assert str(chart["mesh_sha256"]) == sha256_file(output / "surface.npz")
        assert np.isfinite(chart["vertex_uv"]).all()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_repaired_ventricular_surface(
            surface_path=surface_path,
            mask_path=mask_path,
            patch_path=patch_path,
            output_dir=output,
        )


@pytest.mark.parametrize("key", ["parent_surface_sha256", "mask_sha256"])
def test_patch_rejects_wrong_identity_before_output(tmp_path: Path, key: str) -> None:
    surface_path, mask_path, arrays = _fixture(tmp_path)
    arrays[key] = np.asarray("wrong")
    patch_path = tmp_path / "patch.npz"
    np.savez_compressed(patch_path, **arrays)
    with pytest.raises(ValueError, match=key):
        build_repaired_ventricular_surface(
            surface_path=surface_path,
            mask_path=mask_path,
            patch_path=patch_path,
            output_dir=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("wrong_side", [False, True])
def test_patch_rejects_off_surface_or_wrong_outward_orientation(tmp_path: Path, wrong_side: bool) -> None:
    surface_path, mask_path, arrays = _fixture(tmp_path, wrong_side=wrong_side)
    if not wrong_side:
        arrays["new_vertices_intrinsic_xyz_um"][0, 0] += 1
    patch_path = tmp_path / "patch.npz"
    np.savez_compressed(patch_path, **arrays)
    with pytest.raises(ValueError, match="-gradient" if wrong_side else "departs from the zero-level"):
        build_repaired_ventricular_surface(
            surface_path=surface_path,
            mask_path=mask_path,
            patch_path=patch_path,
            output_dir=tmp_path / "output",
            max_surface_error_um=0.1,
        )
    assert not (tmp_path / "output").exists()


def test_patch_requires_exact_seam_and_preserves_unaffected_connectivity() -> None:
    vertices = np.asarray([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    added = np.asarray([[0, 2 / 3, 1 / 3]])
    replacement = np.asarray([[0, 1, 4], [1, 2, 4], [2, 0, 4]], dtype=np.int64)
    combined, result, seam = _splice_surface_patch(vertices, faces, np.asarray([0]), added, replacement)
    np.testing.assert_array_equal(combined[:4], vertices)
    np.testing.assert_array_equal(result[0], faces[1])
    np.testing.assert_array_equal(seam, [0, 1, 2])
    with pytest.raises(ValueError, match="directed boundary"):
        _splice_surface_patch(vertices, faces, np.asarray([0]), added, replacement[:, [0, 2, 1]])
    with pytest.raises(ValueError, match="winding"):
        bad = replacement.copy()
        bad[0] = bad[0, [0, 2, 1]]
        _splice_surface_patch(vertices, faces, np.asarray([0]), added, bad)
    with pytest.raises(ValueError, match="outside the replaced strip"):
        bad = replacement.copy()
        bad[0, 0] = 3
        _splice_surface_patch(vertices, faces, np.asarray([0]), added, bad)


def test_patch_rejects_self_intersection_even_with_conforming_seam(tmp_path: Path) -> None:
    surface_path, mask_path, arrays = _fixture(tmp_path)
    # The fan center lies outside the square, folding triangles over one another.
    arrays["new_vertices_intrinsic_xyz_um"][0, 1] = 6
    patch_path = tmp_path / "patch.npz"
    np.savez_compressed(patch_path, **arrays)
    with pytest.raises(ValueError, match="self-intersections"):
        build_repaired_ventricular_surface(
            surface_path=surface_path,
            mask_path=mask_path,
            patch_path=patch_path,
            output_dir=tmp_path / "output",
            max_surface_error_um=0.1,
        )
    assert not (tmp_path / "output").exists()


def test_patch_rejects_near_duplicate_vertex_before_fem_use(tmp_path: Path) -> None:
    surface_path, mask_path, arrays = _fixture(tmp_path)
    arrays["new_vertices_intrinsic_xyz_um"][0, 1:] = [1 + 1e-6, 1 + 1e-6]
    patch_path = tmp_path / "sliver-patch.npz"
    np.savez_compressed(patch_path, **arrays)

    with pytest.raises(ValueError, match="triangle quality"):
        build_repaired_ventricular_surface(
            surface_path=surface_path,
            mask_path=mask_path,
            patch_path=patch_path,
            output_dir=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()
