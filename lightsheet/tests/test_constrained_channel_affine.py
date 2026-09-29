from __future__ import annotations

import json

import numpy as np

from squisher_lightsheet.constrained_channel_affine import (
    _fit_model,
    _fixed_world_residual_affine,
    _group_balanced_weights,
    _read_direct_anchor_observations,
    _read_seam_constraints,
)
from squisher_lightsheet.artifact_io import sha256_file


def test_joint_fit_recovers_supported_tile_offsets_and_depth_shear() -> None:
    tile_names = ["left", "right"]
    observations = []

    def correction(tile: str, point: np.ndarray) -> np.ndarray:
        offset = 1.0 if tile == "left" else 3.0
        return np.asarray([0.2 * point[0] + offset, -0.1 * point[0] + 0.5 * offset])

    for tile_index, tile in enumerate(tile_names):
        for depth in (0, 10, 20):
            for y, x in ((0, 0), (0, 10), (10, 0), (10, 10)):
                point = np.asarray([depth, y + 20 * tile_index, x], dtype=np.float64)
                delta = correction(tile, point)
                observations.append(
                    {
                        "tile": tile,
                        "window": f"{tile}-{depth}-{y}-{x}",
                        "depth_band": depth,
                        "baseline_world_um_zyx": point,
                        "target_world_um_zyx": point + np.r_[0.0, delta],
                        "delta_world_um_zyx": np.r_[0.0, delta],
                        "independent": True,
                        "selected_attempt": "measured",
                        "spacing_um_zyx": [1.0, 1.0, 1.0],
                    }
                )
    fixed_point = np.asarray([10.0, 10.0, 5.0])
    moving_point = np.asarray([10.0, 20.0, 5.0])
    seams = [
        {
            "fixed_tile": "left",
            "moving_tile": "right",
            "z_start": 8,
            "z_stop": 12,
            "fixed_world_um_zyx": fixed_point,
            "moving_world_um_zyx": moving_point,
            "target_delta_um_zyx": np.r_[
                0.0,
                correction("left", fixed_point) - correction("right", moving_point),
            ],
        }
    ]

    coefficients, _diagnostics, center, scale = _fit_model(
        tile_names=tile_names,
        observations=observations,
        seams=seams,
        gradient_lambdas=(0.0,),
        offset_lambda=0.0,
    )

    for row in observations:
        start = tile_names.index(row["tile"]) * 4
        normalized = (row["baseline_world_um_zyx"] - center) / scale
        predicted = np.r_[normalized, 1.0] @ coefficients[start : start + 4]
        np.testing.assert_allclose(predicted, row["delta_world_um_zyx"][1:3], atol=1e-7)


def test_group_balanced_weights_give_each_group_equal_total_weight() -> None:
    groups = [("anchor", "dense")] * 4 + [("anchor", "sparse")] * 2
    weights = _group_balanced_weights(groups, family_budgets={"anchor": 1.0})

    np.testing.assert_allclose(np.square(weights[:4]).sum(), np.square(weights[4:]).sum())


def test_group_balanced_weights_give_families_the_requested_objective_budget() -> None:
    groups = (
        [("anchor", "a")] * 6
        + [("anchor", "b")] * 2
        + [("seam", "x")] * 3
    )
    weights = _group_balanced_weights(
        groups, family_budgets={"anchor": 0.25, "seam": 0.75}
    )

    anchor_objective = np.square(weights[:8]).sum()
    seam_objective = np.square(weights[8:]).sum()
    np.testing.assert_allclose(seam_objective / anchor_objective, 3.0)


def test_fixed_world_residual_affine_preserves_rotated_baseline_z_row() -> None:
    coefficients = np.asarray(
        [
            [0.3, -0.2],
            [0.1, 0.0],
            [0.0, 0.4],
            [2.0, -3.0],
        ]
    )
    residual = _fixed_world_residual_affine(
        coefficients,
        coordinate_center_um=np.asarray([10.0, 20.0, 30.0]),
        coordinate_scale_um=np.asarray([5.0, 10.0, 20.0]),
        lateral_scale_um=np.asarray([0.3, 0.4]),
    )
    baseline = np.eye(4)
    angle = np.deg2rad(7.0)
    baseline[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    baseline[:3, 3] = [4.0, 5.0, 6.0]

    updated = residual @ baseline

    np.testing.assert_allclose(updated[0], baseline[0], atol=1e-12, rtol=0)
    assert not np.allclose(updated[1:3], baseline[1:3])


def test_seam_constraint_uses_shift_applied_to_moving(tmp_path) -> None:
    def tile(name: str, x: float) -> dict[str, object]:
        return {
            "tile": name,
            "axes": "CZYX",
            "shape": [1, 20, 30, 40],
            "stage_scale_um": {"z": 2.0, "y": 1.0, "x": 0.5},
            "stage_translation_um": {"z": 0.0, "y": 0.0, "x": x},
            "registered_affine": {"matrix": np.eye(4).tolist()},
        }

    artifact = tmp_path / "seams.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.level0_phase_recovery_measurements.v1",
                "rows": [
                    {
                        "status": "accepted",
                        "fixed_tile": "sample-488.001.ome.zarr",
                        "moving_tile": "sample-488.002.ome.zarr",
                        "fixed_slices_zyx": [[0, 10], [0, 10], [0, 10]],
                        "moving_slices_zyx": [[0, 10], [0, 10], [10, 20]],
                        "phase_shift_zyx": [0.0, 2.0, -4.0],
                    }
                ],
            }
        )
    )

    constraints = _read_seam_constraints(
        artifact,
        baseline_by_tile={
            "sample-561638.001.ome.zarr": tile("sample-561638.001.ome.zarr", 0.0),
            "sample-561638.002.ome.zarr": tile("sample-561638.002.ome.zarr", -5.0),
        },
    )

    row = constraints[0]
    # phase_cross_correlation returns the shift applied to moving, so its
    # corresponding source point is moving_center - shift.
    np.testing.assert_allclose(row["target_delta_um_zyx"], [0.0, -2.0, 2.0])
    assert row["fixed_tile"] == "sample-561638.001.ome.zarr"
    assert row["moving_tile"] == "sample-561638.002.ome.zarr"


def test_direct_anchor_uses_registered_grid_shift_and_verified_baseline(tmp_path) -> None:
    tile_name = "sample-561638.001.ome.zarr"
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": tile_name,
                        "axes": "CZYX",
                        "shape": [1, 20, 30, 40],
                        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                        "stage_translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                        "registered_affine": {"matrix": np.eye(4).tolist()},
                    }
                ]
            }
        )
    )
    artifact = tmp_path / "anchors.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.registered_grid_channel_anchors.v1",
                "moving_baseline_registration": str(baseline.resolve()),
                "moving_baseline_sha256": sha256_file(baseline),
                "rows": [
                    {
                        "status": "accepted",
                        "moving_tile": tile_name,
                        "registered_center_um_zyx": [10.0, 20.0, 30.0],
                        "moving_local_center_px_zyx": [5.0, 10.0, 10.0],
                        "shift_to_apply_moving_grid_px_zyx": [2.0, 3.0, -4.0],
                        "grid_spacing_um_zyx": [1.0, 2.0, 3.0],
                        "uncertainty_grid_px_yx": [0.25, 0.5],
                    }
                ],
            }
        )
    )

    rows = _read_direct_anchor_observations(
        artifact,
        baseline_by_tile={tile_name: json.loads(baseline.read_text())["tiles"][0]},
        expected_baseline=baseline,
    )

    np.testing.assert_allclose(rows[0]["delta_world_um_zyx"], [0.0, 6.0, -12.0])
    np.testing.assert_allclose(rows[0]["baseline_world_um_zyx"], [10.0, 20.0, 30.0])
    assert rows[0]["depth_band"] == 5
