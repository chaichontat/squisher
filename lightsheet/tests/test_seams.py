from __future__ import annotations

import numpy as np

from squisher_lightsheet import seams


def test_overlap_plus_margin_sampling_uses_margin_not_full_tile() -> None:
    settings = seams.RobustBoundarySettings(
        patch_shape_zyx=(16, 30, 40),
        max_patches_per_edge=4,
        overlap_margin_zyx=(0, 10, 0),
    )
    specs = seams.sample_boundary_patches_from_bounds(
        tile_shapes_zyx=[(64, 100, 100), (64, 100, 100)],
        bounds=[
            seams.TileBounds(start_zyx=(0.0, 0.0, 0.0), stop_zyx=(64.0, 100.0, 100.0)),
            seams.TileBounds(start_zyx=(0.0, 80.0, 0.0), stop_zyx=(64.0, 180.0, 100.0)),
        ],
        pairs=[(0, 1)],
        settings=settings,
    )

    assert specs
    first = specs[0]
    assert first.axis == "y"
    assert first.overlap_start_zyx[1] == 70
    assert first.fixed_slices[1] == slice(70, 100)
    assert first.moving_slices[1] == slice(0, 30)
    assert first.fixed_slices[2].stop <= 100
    assert first.moving_slices[2].stop <= 100


def test_overlap_plus_margin_sampling_is_deterministic() -> None:
    settings = seams.RobustBoundarySettings(
        patch_shape_zyx=(16, 30, 40),
        max_patches_per_edge=4,
        overlap_margin_zyx=(0, 10, 0),
    )
    kwargs = {
        "tile_shapes_zyx": [(64, 100, 100), (64, 100, 100)],
        "bounds": [
            seams.TileBounds(start_zyx=(0.0, 0.0, 0.0), stop_zyx=(64.0, 100.0, 100.0)),
            seams.TileBounds(start_zyx=(0.0, 80.0, 0.0), stop_zyx=(64.0, 180.0, 100.0)),
        ],
        "pairs": [(0, 1)],
        "settings": settings,
    }

    left = seams.sample_boundary_patches_from_bounds(**kwargs)
    right = seams.sample_boundary_patches_from_bounds(**kwargs)

    assert left == right


def test_named_settings_keep_distinct_patch_contracts() -> None:
    robust = seams.robust_boundary_settings()
    recovery = seams.overlap_recovery_settings()

    assert robust.patch_shape_zyx == (64, 512, 512)
    assert robust.max_patches_per_edge == 128
    assert recovery.patch_shape_zyx == (64, 256, 256)
    assert recovery.max_patches_per_edge == 128
    assert robust.overlap_margin_zyx != (0, 0, 0)
    assert recovery.overlap_margin_zyx != (0, 0, 0)


def test_128_patch_budget_generates_128_uniform_candidates() -> None:
    settings = seams.RobustBoundarySettings(
        patch_shape_zyx=(16, 30, 40),
        max_patches_per_edge=128,
        overlap_margin_zyx=(0, 10, 0),
    )
    specs = seams.sample_boundary_patches_from_bounds(
        tile_shapes_zyx=[(64, 256, 256), (64, 256, 256)],
        bounds=[
            seams.TileBounds(start_zyx=(0.0, 0.0, 0.0), stop_zyx=(64.0, 256.0, 256.0)),
            seams.TileBounds(start_zyx=(0.0, 200.0, 0.0), stop_zyx=(64.0, 456.0, 256.0)),
        ],
        pairs=[(0, 1)],
        settings=settings,
    )

    assert len(specs) == 128
    assert [spec.patch_index for spec in specs] == list(range(128))


def test_center_z_content_prefilter_rejects_low_signal_center_plane() -> None:
    settings = seams.RobustBoundarySettings(min_center_z_p99=180.0, min_center_z_std=8.0)
    low = np.full((5, 32, 32), 120, dtype=np.float32)
    high = low.copy()
    high[2, 8:24, 8:24] = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) + 200

    reason, _, _ = seams.center_z_content_prefilter_reason(low, high, settings)

    assert reason == "low_center_z_p99"


def boundary_constraint(
    *,
    patch_index: int,
    shift_zyx: tuple[float, float, float],
    weight: float = 2.0,
) -> seams.BoundaryConstraint:
    return seams.BoundaryConstraint(
        fixed=0,
        moving=1,
        pair=(0, 1),
        axis="x",
        patch_index=patch_index,
        shift_zyx=shift_zyx,
        weight=weight,
        correlation_before=0.0,
        correlation_after=1.0,
        improvement=1.0,
        fixed_nonzero_fraction=1.0,
        moving_nonzero_fraction=1.0,
        fixed_std=1.0,
        moving_std=1.0,
        accepted=True,
    )


def test_edges_without_inlier_cluster_are_downweighted_not_rejected() -> None:
    settings = seams.RobustBoundarySettings(
        min_inlier_patches_per_edge=3,
        weak_edge_weight_factor=0.25,
    )
    constraints = [
        boundary_constraint(patch_index=0, shift_zyx=(0.0, 0.0, 0.0), weight=4.0),
        boundary_constraint(patch_index=1, shift_zyx=(20.0, 0.0, 0.0), weight=2.0),
    ]

    updated = seams.mark_boundary_inliers_by_edge(constraints, settings)

    assert [constraint.accepted for constraint in updated] == [True, True]
    assert [constraint.weight for constraint in updated] == [1.0, 0.5]
    assert {constraint.edge_status for constraint in updated} == {"downweighted_no_inlier_cluster"}


def test_edges_with_inlier_cluster_reject_patch_outliers() -> None:
    settings = seams.RobustBoundarySettings(min_inlier_patches_per_edge=3)
    constraints = [
        boundary_constraint(patch_index=0, shift_zyx=(0.0, 0.0, 0.0)),
        boundary_constraint(patch_index=1, shift_zyx=(0.5, 0.5, 0.5)),
        boundary_constraint(patch_index=2, shift_zyx=(1.0, 1.0, 1.0)),
        boundary_constraint(patch_index=3, shift_zyx=(20.0, 0.0, 0.0)),
    ]

    updated = seams.mark_boundary_inliers_by_edge(constraints, settings)

    assert [constraint.accepted for constraint in updated] == [True, True, True, False]
    assert [constraint.edge_status for constraint in updated[:3]] == ["inlier_cluster"] * 3
    assert updated[3].reject_reason == "outlier_shift_cluster"


def test_center_z_gradient_metric_scores_center_plane_after_shift() -> None:
    fixed = np.zeros((3, 96, 96), dtype=np.float32)
    moving = np.zeros_like(fixed)
    yy, xx = np.mgrid[0:48, 0:40]
    texture = (1.0 + np.sin(yy / 3.0) + np.cos(xx / 5.0)).astype(np.float32)
    fixed[1, 24:72, 28:68] = texture
    moving[1, 24:72, 28:68] = texture
    fixed[0, 4:28, 4:28] = 1.0
    moving[0, 68:92, 68:92] = 1.0
    fixed[2, 68:92, 4:28] = 1.0
    moving[2, 4:28, 68:92] = 1.0

    before, after = seams.center_z_gradient_component_ncc_after_shift(fixed, moving, (0.0, 0.0, 0.0))
    expected = seams.center_z_gradient_component_ncc(fixed[1], moving[1])

    assert before == expected
    assert after == expected
    assert after > 0.99


def test_shifted_center_plane_matches_full_patch_shift() -> None:
    source = np.zeros((9, 32, 32), dtype=np.float32)
    zz, yy, xx = np.mgrid[0:9, 0:32, 0:32]
    source = (zz * 11.0 + yy * 3.0 + xx).astype(np.float32)
    shift = (1.2, -2.5, 3.0)

    center_only = seams.shifted_center_plane_cpu(source, shift)
    full_shift = seams.shift_array_cpu(source, shift)[source.shape[0] // 2]

    np.testing.assert_allclose(center_only, full_shift, atol=1e-5)
