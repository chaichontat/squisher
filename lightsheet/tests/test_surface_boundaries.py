from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet.artifact_io import sha256_file
from squisher_lightsheet.surface_boundaries import (
    ARTIFACT_TYPE,
    build_surface_boundaries,
    fit_log_boundary,
    fit_ordered_outer_boundary,
    measure_peak_boundaries,
    peak_fraction_crossing,
    wall_support,
)


def _strip_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsic = np.asarray(
        [[x, y, z] for z in (0.0, 10.0) for y, x in enumerate((1.0, 3.0, 0.0, 3.0, 1.0))],
        dtype=np.float64,
    )
    world = intrinsic.copy()
    world[:, 0] = np.arange(len(world), dtype=np.float64) * 0.2
    faces: list[list[int]] = []
    for y in range(4):
        faces.extend(([y, y + 1, 5 + y + 1], [y, 5 + y + 1, 5 + y]))
    return world, intrinsic, np.asarray(faces, dtype=np.int64)


def test_peak_fraction_crossing_interpolates_each_side_and_rejects_a_gap() -> None:
    depth = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
    signal = np.asarray([0.0, 4.0, 10.0, 4.0, 0.0])
    counts = np.ones(5, dtype=np.uint32)

    assert peak_fraction_crossing(depth, signal, counts, fraction=0.6, side="rising") == pytest.approx(
        13.333333333333334
    )
    assert peak_fraction_crossing(depth, signal, counts, fraction=0.75, side="falling") == pytest.approx(
        24.166666666666668
    )

    counts[1] = 0
    assert np.isnan(peak_fraction_crossing(depth, signal, counts, fraction=0.6, side="rising"))
    assert peak_fraction_crossing(depth, signal, counts, fraction=0.75, side="falling") == pytest.approx(
        24.166666666666668
    )


def test_measure_peak_boundaries_tracks_inner_outer_and_joint_support() -> None:
    depth = np.arange(5, dtype=np.float64) * 10.0
    intensity = np.tile([0.0, 4.0, 10.0, 4.0, 0.0], (3, 1))
    counts = np.ones_like(intensity, dtype=np.uint32)
    counts[1, 1] = 0
    counts[2, 3] = 0

    _, _, inner, outer, joint = measure_peak_boundaries(depth, intensity, counts)

    np.testing.assert_array_equal(inner, [True, False, True])
    np.testing.assert_array_equal(outer, [True, True, False])
    np.testing.assert_array_equal(joint, [True, False, False])


def test_measure_peak_boundaries_retains_zero_crossing_but_excludes_it_from_log_fit() -> None:
    depth = np.asarray([0.0, 10.0, 20.0])
    intensity = np.asarray([[6.0, 10.0, 0.0]])
    counts = np.ones_like(intensity, dtype=np.uint32)

    raw_inner, _, direct_inner, _, _ = measure_peak_boundaries(depth, intensity, counts)

    assert raw_inner[0] == 0.0
    assert not direct_inner[0]


def test_log_fit_requires_an_anchor_on_every_mesh_component() -> None:
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    raw = np.asarray([10.0, np.nan, np.nan, np.nan, np.nan, np.nan])

    with pytest.raises(ValueError, match="each mesh component"):
        fit_log_boundary(vertices, faces, raw, np.isfinite(raw), smooth_um=30.0)


@pytest.mark.parametrize("bad_depth", [0.0, -1.0])
def test_log_fit_rejects_nonpositive_direct_depth_before_log(bad_depth: float) -> None:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    raw = np.asarray([bad_depth, 10.0, 10.0])

    with pytest.raises(ValueError, match="positive before log"):
        fit_log_boundary(vertices, faces, raw, np.ones(3, dtype=bool), smooth_um=30.0)


def test_ordered_outer_fit_allows_equality() -> None:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    inner = np.full(3, 10.0)

    outer = fit_ordered_outer_boundary(
        vertices,
        faces,
        np.full(3, 5.0),
        np.ones(3, dtype=bool),
        inner,
        constraint_mask=np.ones(3, dtype=bool),
        smooth_um=80.0,
    )

    np.testing.assert_array_equal(outer, inner)


def test_ordered_outer_fit_only_constrains_selected_domain() -> None:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    inner = np.full(3, 10.0)
    constrained = np.asarray([True, False, False])

    outer = fit_ordered_outer_boundary(
        vertices,
        faces,
        np.full(3, 5.0),
        np.ones(3, dtype=bool),
        inner,
        constraint_mask=constrained,
        smooth_um=0.01,
    )

    assert outer[0] == inner[0]
    assert np.all(outer[~constrained] < inner[~constrained])


def test_wall_support_uses_intrinsic_turnarounds() -> None:
    world, intrinsic, _ = _strip_mesh()

    keep, bounds = wall_support(intrinsic)

    np.testing.assert_array_equal(keep, [False, True, True, True, False] * 2)
    np.testing.assert_allclose(bounds, [[0.0, 1.0, 3.0], [10.0, 1.0, 3.0]])
    world_keep, _ = wall_support(world)
    assert not np.array_equal(world_keep, keep)

    mirrored = intrinsic.copy()
    mirrored[:, 0] *= -1
    mirrored_keep, mirrored_bounds = wall_support(mirrored, cavity_x_sign=-1)
    np.testing.assert_array_equal(mirrored_keep, keep)
    np.testing.assert_allclose(mirrored_bounds, bounds)


def test_builder_preserves_raw_measurements_and_masks_fitted_caps_once(
    tmp_path: Path,
) -> None:
    world, intrinsic, faces = _strip_mesh()
    surface_path = tmp_path / "surface.npz"
    np.savez_compressed(
        surface_path,
        vertices_xyz_um=world,
        vertices_intrinsic_xyz_um=intrinsic,
        faces=faces,
    )
    depth = np.arange(5, dtype=np.float64) * 10.0
    intensity = np.tile([0.0, 4.0, 10.0, 4.0, 0.0], (len(world), 1))
    counts = np.ones_like(intensity, dtype=np.uint32)
    counts[2, 1] = 0
    counts[7, 3] = 0
    permutation = np.asarray([7, 1, 8, 3, 9, 0, 5, 2, 6, 4])
    profiles_path = tmp_path / "profiles.npz"
    np.savez_compressed(
        profiles_path,
        depth_um=depth,
        intensity=intensity[permutation],
        valid_count=counts[permutation],
        vertex_index=permutation,
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.surface_profiles.v1",
                "surface_sha256": sha256_file(surface_path),
                "output_sha256": sha256_file(profiles_path),
            }
        )
    )
    output_dir = tmp_path / "boundaries"

    metadata_path = build_surface_boundaries(
        profiles_path=profiles_path,
        surface_path=surface_path,
        output_dir=output_dir,
    )

    assert metadata_path == (output_dir / "metadata.json").resolve()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["artifact_type"] == ARTIFACT_TYPE
    assert metadata["inner"]["peak_fraction"] == 0.60
    assert metadata["outer"]["peak_fraction"] == 0.75
    with np.load(output_dir / "boundaries.npz") as result:
        assert set(result.files) == {
            "raw_inner_um",
            "raw_outer_um",
            "inner_um",
            "outer_um",
            "direct_supported_inner",
            "direct_supported_outer",
            "direct_supported_joint",
            "wall_support",
            "estimated_inner",
            "estimated_outer",
            "recovered_missing_measurement_inner",
            "recovered_missing_measurement_outer",
        }
        np.testing.assert_array_equal(result["wall_support"], [False, True, True, True, False] * 2)
        assert np.isfinite(result["raw_inner_um"][[0, 4, 5, 9]]).all()
        assert np.isnan(result["inner_um"][[0, 4, 5, 9]]).all()
        assert np.isnan(result["outer_um"][[0, 4, 5, 9]]).all()
        assert result["recovered_missing_measurement_inner"][2]
        assert result["recovered_missing_measurement_outer"][7]
        estimated = result["estimated_inner"] & result["estimated_outer"]
        assert np.all(result["outer_um"][estimated] >= result["inner_um"][estimated])

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_surface_boundaries(
            profiles_path=profiles_path,
            surface_path=surface_path,
            output_dir=output_dir,
        )

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["surface_sha256"] = "wrong"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="surface_sha256"):
        build_surface_boundaries(
            profiles_path=profiles_path,
            surface_path=surface_path,
            output_dir=tmp_path / "mismatched-boundaries",
        )
