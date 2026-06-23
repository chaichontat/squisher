from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from squisher_lightsheet import registration as stitch


def affine_param(z: float = 0.0, y: float = 0.0, x: float = 0.0) -> xr.DataArray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = (z, y, x)
    return xr.DataArray(
        matrix[None, :, :],
        dims=("t", "x_in", "x_out"),
        coords={"t": [0], "x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]},
    )


def tile(index: int, *, source_view: str | None = None) -> stitch.TileMetadata:
    return stitch.TileMetadata(
        path=Path(f"tile_{index}.ome.tif"),
        shape=(1, 64, 100, 100),
        axes="CZYX",
        spacing={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": float(index * 80), "x": 0.0},
        channels=("ch0",),
        tracks=(
            stitch.TrackMetadata(
                slug="track0",
                track_id="all",
                channels=(0,),
                channel_names=("ch0",),
            ),
        ),
        source_view=source_view,
    )


def constraint(
    fixed: int,
    moving: int,
    shift_zyx: tuple[float, float, float],
    *,
    weight: float = 1.0,
    source_label: str | None = None,
) -> stitch.BoundaryConstraint:
    return stitch.BoundaryConstraint(
        fixed=fixed,
        moving=moving,
        pair=(fixed, moving),
        axis="x",
        patch_index=0,
        shift_zyx=shift_zyx,
        weight=weight,
        correlation_before=0.1,
        correlation_after=0.9,
        improvement=0.8,
        fixed_nonzero_fraction=1.0,
        moving_nonzero_fraction=1.0,
        fixed_std=1.0,
        moving_std=1.0,
        accepted=True,
        source_label=source_label,
    )


def test_phase_peak_refinement_quantizes_to_tenth_pixel() -> None:
    true_offset = 0.24
    left = -((-1.0 - true_offset) ** 2)
    center = -(true_offset**2)
    right = -((1.0 - true_offset) ** 2)

    shift = stitch.refined_phase_shift_from_samples(
        3,
        16,
        left,
        center,
        right,
    )

    assert shift == pytest.approx(3.2)


def test_phase_peak_refinement_wraps_negative_tenth_pixel_shift() -> None:
    true_offset = 0.24
    left = -((-1.0 - true_offset) ** 2)
    center = -(true_offset**2)
    right = -((1.0 - true_offset) ** 2)

    shift = stitch.refined_phase_shift_from_samples(
        15,
        16,
        left,
        center,
        right,
    )

    assert shift == pytest.approx(-0.8)


def test_fixed_xy_solver_solves_z_only_and_keeps_xy_corrections_zero() -> None:
    settings = stitch.RobustBoundarySettings(
        max_correction_zyx=(16, 96, 96),
        max_final_residual_zyx=(4.0, 8.0, 8.0),
    )
    constraints = [
        constraint(0, 1, (3.0, 40.0, -25.0), weight=10.0),
        constraint(1, 2, (2.0, -30.0, 12.0), weight=10.0),
    ]

    corrections, filtered, anchor_tile = stitch.solve_tile_corrections_with_residual_rejection(
        [tile(0), tile(1), tile(2)],
        constraints,
        settings,
        fixed_axes={"y", "x"},
    )

    assert anchor_tile == 1
    assert all(item.accepted for item in filtered)
    assert [correction[1:] for correction in corrections] == [(0.0, 0.0)] * 3
    assert corrections[1][0] == 0.0
    assert corrections[0][0] == pytest.approx(-3.0)
    assert corrections[2][0] == pytest.approx(2.0)
    assert filtered[0].final_residual_zyx == pytest.approx((0.0, -40.0, 25.0))


def test_full_xyz_solver_solves_lateral_corrections_when_axes_are_not_fixed() -> None:
    settings = stitch.RobustBoundarySettings(
        max_correction_zyx=(16, 96, 96),
        max_final_residual_zyx=(4.0, 8.0, 8.0),
    )
    constraints = [
        constraint(0, 1, (3.0, 40.0, -25.0), weight=10.0),
        constraint(1, 2, (2.0, -30.0, 12.0), weight=10.0),
    ]

    corrections, filtered, anchor_tile = stitch.solve_tile_corrections_with_residual_rejection(
        [tile(0), tile(1), tile(2)],
        constraints,
        settings,
        fixed_axes=None,
    )

    assert anchor_tile == 1
    assert all(item.accepted for item in filtered)
    assert corrections[1] == pytest.approx((0.0, 0.0, 0.0))
    assert corrections[0] == pytest.approx((-3.0, -40.0, 25.0))
    assert corrections[2] == pytest.approx((2.0, -30.0, 12.0))
    assert filtered[0].final_residual_zyx == pytest.approx((0.0, 0.0, 0.0))


def test_reference_prior_penalizes_lateral_corrections_without_fixing_them() -> None:
    settings = stitch.RobustBoundarySettings(
        max_correction_zyx=(16, 96, 96),
        irls_iterations=1,
    )
    constraints = [constraint(0, 1, (0.0, 40.0, -20.0), weight=0.001)]

    corrections, filtered, anchor_tile = stitch.solve_tile_corrections_with_residual_rejection(
        [tile(0), tile(1)],
        constraints,
        settings,
        reference_prior_weights_zyx=(0.0, 0.01, 0.01),
        residual_reject_axes=set(),
    )

    assert anchor_tile == 0
    assert all(item.accepted for item in filtered)
    assert corrections[0] == pytest.approx((0.0, -10.0 / 3.0, 5.0 / 3.0))
    assert corrections[1] == pytest.approx((0.0, 10.0 / 3.0, -5.0 / 3.0))
    assert abs(filtered[0].final_residual_zyx[1]) > settings.max_final_residual_zyx[1]


def test_joint_huber_downweights_a_patch_outlying_in_any_solved_axis() -> None:
    settings = stitch.RobustBoundarySettings(
        max_correction_zyx=(16, 96, 96),
        max_final_residual_zyx=(4.0, 8.0, 8.0),
        irls_iterations=8,
    )
    constraints = [
        constraint(0, 1, (0.0, 0.0, 0.0), weight=1.0),
        constraint(0, 1, (0.0, 0.0, 0.0), weight=1.0),
        constraint(0, 1, (2.0, 0.0, 40.0), weight=1.0),
    ]

    corrections = stitch.solve_tile_corrections_zyx(
        2,
        constraints,
        settings,
        anchor_tile=0,
    )

    assert corrections[1][0] < 0.25
    assert corrections[1][2] < 4.1


def test_reference_geometry_solver_options_define_reference_prior_contract() -> None:
    fixed_xy = stitch.reference_geometry_solver_options("fixed-xy", 0.01)
    full_xyz = stitch.reference_geometry_solver_options("full-xyz", 0.01)
    penalized_xy = stitch.reference_geometry_solver_options("penalized-xy", 0.01)

    assert fixed_xy.fixed_axes == {"y", "x"}
    assert fixed_xy.reference_prior_weights_zyx is None
    assert fixed_xy.residual_reject_axes is None

    assert full_xyz.fixed_axes == set()
    assert full_xyz.reference_prior_weights_zyx is None
    assert full_xyz.residual_reject_axes is None

    assert penalized_xy.fixed_axes == set()
    assert penalized_xy.reference_prior_weights_zyx == (0.0, 0.01, 0.01)
    assert penalized_xy.residual_reject_axes == set()


def test_reference_geometry_solver_options_reject_negative_prior() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        stitch.reference_geometry_solver_options("penalized-xy", -0.01)


def test_axis_aligned_pairs_skip_cross_source_view_discontinuities() -> None:
    tiles = [
        tile(0, source_view="L"),
        tile(1, source_view="L"),
        tile(0, source_view="R"),
        tile(1, source_view="R"),
    ]
    tiles[2] = stitch.replace_tile_stage_transform(
        tiles[2],
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        stage_scale=None,
        source_view="R",
    )
    tiles[3] = stitch.replace_tile_stage_transform(
        tiles[3],
        translation={"z": 0.0, "y": 80.0, "x": 0.0},
        stage_scale=None,
        source_view="R",
    )

    pairs = stitch.axis_aligned_registration_pairs(tiles)

    assert pairs == [(0, 1), (2, 3)]


def test_reference_prior_refinement_starts_from_loaded_registration() -> None:
    loaded = [affine_param(10.0, 20.0, 30.0)]
    reference = [affine_param(1.0, 2.0, 3.0)]

    start, source = stitch.refinement_start_params(
        loaded,
        reference,
        reference_geometry_mode="penalized-xy",
    )

    assert start is loaded
    assert source == "registration"


def test_reference_params_are_only_fallback_when_no_registration_exists() -> None:
    reference = [affine_param(1.0, 2.0, 3.0)]

    start, source = stitch.refinement_start_params(
        None,
        reference,
        reference_geometry_mode="penalized-xy",
    )

    assert start is reference
    assert source == "reference"


def test_rigid_initial_alignment_reduces_reference_drift() -> None:
    params = [
        affine_param(0.0, 0.0, 0.0),
        affine_param(0.0, 10.0, 0.0),
        affine_param(0.0, 0.0, 10.0),
    ]
    reference = [
        affine_param(5.0, 20.0, -3.0),
        affine_param(5.0, 30.0, -3.0),
        affine_param(5.0, 20.0, 7.0),
    ]

    aligned, summary = stitch.align_params_to_reference(
        params,
        reference,
        method="rigid",
    )

    assert summary["method"] == "rigid"
    np.testing.assert_allclose(
        np.asarray([stitch.affine_translation_zyx(param) for param in aligned]),
        np.asarray([stitch.affine_translation_zyx(param) for param in reference]),
        atol=1e-9,
    )
    assert summary["drift_after_um"]["y"]["max_abs"] < 1e-9
    assert summary["drift_after_um"]["x"]["max_abs"] < 1e-9


def test_refinement_alignment_returns_persistable_summary() -> None:
    params = [
        affine_param(0.0, 0.0, 0.0),
        affine_param(0.0, 10.0, 0.0),
        affine_param(0.0, 0.0, 10.0),
    ]
    reference = [
        affine_param(1.0, 20.0, -5.0),
        affine_param(1.0, 30.0, -5.0),
        affine_param(1.0, 20.0, 5.0),
    ]

    aligned, summary = stitch.align_refinement_start_to_reference(
        params,
        reference,
        method="rigid",
        source="registration",
    )

    assert aligned is not params
    assert summary is not None
    assert summary["method"] == "rigid"
    assert summary["drift_before_um"]["y"]["p95_abs"] > summary["drift_after_um"]["y"]["p95_abs"]


def test_combined_channel_constraint_averages_track_measurements_without_row_duplication() -> None:
    settings = stitch.RobustBoundarySettings()
    track1 = constraint(0, 1, (0.2, -1.0, 3.0), weight=0.4, source_label="track1")
    track2 = constraint(0, 1, (0.4, -3.0, 5.0), weight=0.4, source_label="track2")

    combined = stitch.combine_channel_boundary_constraints(
        [track1, track2],
        source_label="track1+track2",
        settings=settings,
    )

    assert combined.accepted
    assert combined.source_label == "track1+track2"
    assert combined.shift_zyx == pytest.approx((0.3, -2.0, 4.0))
    assert combined.weight == pytest.approx(0.4)


def test_apply_reference_fixed_axes_overwrites_xy_in_physical_units() -> None:
    params = [affine_param(10.0, 20.0, 30.0), affine_param(11.0, 21.0, 31.0)]
    reference = [affine_param(1.0, 200.0, 300.0), affine_param(2.0, 201.0, 301.0)]

    constrained = stitch.apply_reference_fixed_axes(params, reference, {"y", "x"})

    assert [stitch.affine_translation_zyx(param) for param in constrained] == [
        (10.0, 200.0, 300.0),
        (11.0, 201.0, 301.0),
    ]


def test_reference_geometry_payload_records_zero_fixed_xy_drift(tmp_path: Path) -> None:
    params = [affine_param(10.0, 200.0, 300.0), affine_param(11.0, 201.0, 301.0)]
    reference = [affine_param(1.0, 200.0, 300.0), affine_param(2.0, 201.0, 301.0)]
    constraints = [
        constraint(0, 1, (1.0, 10.0, 10.0), source_label="track1"),
        constraint(0, 1, (1.0, -8.0, -8.0), source_label="track2"),
    ]

    payload = stitch.reference_geometry_constraint(
        mode="fixed-xy",
        reference_input=tmp_path / "registration.track0.json",
        fixed_axes={"y", "x"},
        params=params,
        reference_params=reference,
        constraints=constraints,
        shared_geometry_tracks=("track1", "track2"),
    )

    assert payload.fixed_axes == ("y", "x")
    assert payload.shared_geometry_tracks == ("track1", "track2")
    assert payload.drift_from_reference_um is not None
    assert payload.drift_from_reference_um["y"]["max_abs"] == 0.0
    assert payload.drift_from_reference_um["x"]["max_abs"] == 0.0
    assert payload.drift_from_reference_um["z"]["median"] == 9.0
    assert payload.constraint_counts_by_track == {
        "track1": {"accepted": 1, "total": 1},
        "track2": {"accepted": 1, "total": 1},
    }
