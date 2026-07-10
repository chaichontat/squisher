from __future__ import annotations

import json

import numpy as np

from squisher_lightsheet.channel_optimization import (
    apply_direct_overrides,
    apply_seam_overrides,
    load_seam_constraints,
)


def test_load_seam_constraints_accepts_jsonl_starting_with_object(tmp_path) -> None:
    seam_path = tmp_path / "boundary_residuals.jsonl"
    seam_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "accepted": True,
                        "fixed": 0,
                        "moving": 1,
                        "pair": [0, 1],
                        "axis": "x",
                        "patch_index": 0,
                        "shift_zyx": [0.0, 1.0, -8.0],
                        "correlation_before": 0.2,
                        "correlation_after": 0.8,
                    }
                ),
                json.dumps(
                    {
                        "accepted": False,
                        "fixed": 0,
                        "moving": 1,
                        "pair": [0, 1],
                        "axis": "x",
                        "patch_index": 1,
                        "shift_zyx": [0.0, 0.0, 0.0],
                        "correlation_before": 0.1,
                        "correlation_after": 0.1,
                    }
                ),
            ]
        )
        + "\n"
    )
    source_registration = {
        "tiles": [
            {"tile": "tile-0", "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0}},
            {"tile": "tile-1", "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0}},
        ]
    }

    constraints = load_seam_constraints(
        seam_path,
        source_registration,
        tile_names=["tile-0", "tile-1"],
        min_corr_after=0.45,
        max_shift_px=np.asarray([3.0, 12.0, 12.0]),
    )

    assert len(constraints) == 1
    assert constraints[0]["fixed"] == "tile-0"
    assert constraints[0]["moving"] == "tile-1"
    np.testing.assert_allclose(constraints[0]["target_correction_delta_px"], [0.0, 1.0, -8.0])


def test_load_seam_constraints_allows_all_edges_to_be_filtered_out(tmp_path) -> None:
    seam_path = tmp_path / "boundary_residuals.jsonl"
    seam_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "fixed": 0,
                "moving": 1,
                "pair": [0, 1],
                "axis": "x",
                "patch_index": 0,
                "shift_zyx": [0.0, 1.0, -90.0],
                "correlation_before": 0.2,
                "correlation_after": 0.8,
            }
        )
        + "\n"
    )
    source_registration = {
        "tiles": [
            {"tile": "tile-0", "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0}},
            {"tile": "tile-1", "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0}},
        ]
    }

    constraints = load_seam_constraints(
        seam_path,
        source_registration,
        tile_names=["tile-0", "tile-1"],
        min_corr_after=0.45,
        max_shift_px=np.asarray([3.0, 12.0, 12.0]),
    )

    assert constraints == []


def test_apply_seam_overrides_rejects_pair(tmp_path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "id": "bad_pair",
                        "action": "reject_pair",
                        "fixed_tile": "tile-0",
                        "moving_tile": "tile-1",
                    }
                ]
            }
        )
        + "\n"
    )
    constraints = [
        {
            "fixed": "tile-0",
            "moving": "tile-1",
            "fixed_index": 0,
            "moving_index": 1,
            "pair": [0, 1],
            "target_correction_delta_px": np.asarray([0.0, 1.0, 2.0]),
            "weight": 1.0,
        },
        {
            "fixed": "tile-1",
            "moving": "tile-2",
            "fixed_index": 1,
            "moving_index": 2,
            "pair": [1, 2],
            "target_correction_delta_px": np.asarray([0.0, 3.0, 4.0]),
            "weight": 1.0,
        },
    ]

    updated, summary = apply_seam_overrides(constraints, overrides, tile_names=["tile-0", "tile-1", "tile-2"])

    assert len(updated) == 1
    assert updated[0]["fixed"] == "tile-1"
    assert updated[0]["moving"] == "tile-2"
    assert summary["applied"][0]["removed_constraints"] == 1


def test_apply_seam_overrides_force_accept_cluster_from_selected_source_patches(tmp_path) -> None:
    source_jsonl = tmp_path / "source.jsonl"
    source_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "fixed_tile": "tile-0",
                        "moving_tile": "tile-1",
                        "patch_index": 7,
                        "shift_zyx": [0.0, 10.0, 20.0],
                    }
                ),
                json.dumps(
                    {
                        "fixed_tile": "tile-0",
                        "moving_tile": "tile-1",
                        "patch_index": 8,
                        "shift_zyx": [2.0, 12.0, 24.0],
                    }
                ),
                json.dumps(
                    {
                        "fixed_tile": "tile-9",
                        "moving_tile": "tile-1",
                        "patch_index": 7,
                        "shift_zyx": [99.0, 99.0, 99.0],
                    }
                ),
            ]
        )
        + "\n"
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "defaults": {"require_selected_patches_exist": True},
                "overrides": [
                    {
                        "id": "curated_pair",
                        "action": "force_accept_cluster",
                        "fixed_tile": "tile-0",
                        "moving_tile": "tile-1",
                        "source_jsonl": str(source_jsonl),
                        "patch_indices": [7, 8],
                        "solver_weight": 2.0,
                    }
                ],
            }
        )
        + "\n"
    )
    constraints = [
        {
            "fixed": "tile-0",
            "moving": "tile-1",
            "fixed_index": 0,
            "moving_index": 1,
            "pair": [0, 1],
            "target_correction_delta_px": np.asarray([9.0, 9.0, 9.0]),
            "weight": 1.0,
        }
    ]

    updated, summary = apply_seam_overrides(constraints, overrides, tile_names=["tile-0", "tile-1"])

    assert len(updated) == 1
    constraint = updated[0]
    assert constraint["source"] == "seam_constraint_override"
    assert constraint["override_id"] == "curated_pair"
    assert constraint["patch_indices"] == [7, 8]
    assert constraint["weight"] == 2.0
    np.testing.assert_allclose(constraint["target_correction_delta_px"], [1.0, 11.0, 22.0])
    assert summary["applied"][0]["removed_constraints"] == 1
    assert summary["applied"][0]["source_patch_indices"] == [7, 8]


def test_apply_direct_overrides_drops_tile_direct_anchor(tmp_path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "id": "drop_bad_anchor",
                        "action": "drop_direct_anchor",
                        "tile": "tile-0",
                    }
                ]
            }
        )
        + "\n"
    )
    constraints = [
        {"tile": "tile-0", "tile_index": 0, "target_correction_px": np.zeros(3), "weight": 1.0},
        {"tile": "tile-1", "tile_index": 1, "target_correction_px": np.zeros(3), "weight": 1.0},
    ]

    updated, summary = apply_direct_overrides(constraints, overrides, tile_names=["tile-0", "tile-1"])

    assert [constraint["tile"] for constraint in updated] == ["tile-1"]
    assert summary["applied"][0]["removed_constraints"] == 1
