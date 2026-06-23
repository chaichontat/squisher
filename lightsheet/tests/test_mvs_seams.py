from __future__ import annotations

import numpy as np

from squisher_lightsheet import mvs_seams


def mvs_registration_payload() -> dict:
    return {
        "spacing_um": {"z": 2.0, "y": 1.0, "x": 0.5},
        "tiles": [
            {"tile": "tile0.ome.tif"},
            {"tile": "tile1.ome.tif"},
            {"tile": "tile2.ome.tif"},
        ],
        "metrics": {
            "pairwise_registration": {
                "edges": [
                    {
                        "source": 0,
                        "target": 1,
                        "quality": {"data": [0.6]},
                        "attrs": {
                            "transform": {
                                "data": [
                                    [
                                        [1.0, 0.0, 0.0, -4.0],
                                        [0.0, 1.0, 0.0, -3.0],
                                        [0.0, 0.0, 1.0, -2.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                ]
                            }
                        },
                    },
                    {
                        "source": 1,
                        "target": 2,
                        "quality": {"data": [0.2]},
                        "attrs": {
                            "transform": {
                                "data": [
                                    [
                                        [1.0, 0.0, 0.0, -2.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                ]
                            }
                        },
                    },
                ]
            },
            "groupwise_resolution": {
                "metrics": {
                    "used_edges": {"0": [[0, 1], [1, 2]]},
                    "edge_residuals": {"0": {"(0, 1)": 0.5, "(1, 2)": 12.0}},
                }
            },
        },
    }


def test_mvs_pairwise_transform_converts_to_moving_minus_fixed_delta_px() -> None:
    constraints = mvs_seams.mvs_seam_constraints(
        mvs_registration_payload(),
        tile_names=["tile0.ome.tif", "tile1.ome.tif", "tile2.ome.tif"],
        spacing_um_zyx=np.asarray([2.0, 1.0, 0.5]),
        min_quality=0.25,
    )

    assert len(constraints) == 1
    assert constraints[0]["fixed"] == "tile0.ome.tif"
    assert constraints[0]["moving"] == "tile1.ome.tif"
    np.testing.assert_allclose(constraints[0]["target_correction_delta_px"], [2.0, 3.0, 4.0])


def test_mvs_seam_constraints_ignore_gradient_ncc_scores() -> None:
    payload = mvs_registration_payload()
    payload["metrics"]["pairwise_registration"]["edges"][0]["attrs"]["gradient_component_ncc_after"] = 0.10
    payload["metrics"]["pairwise_registration"]["edges"][1]["attrs"]["gradient_component_ncc_after"] = 0.70

    constraints = mvs_seams.mvs_seam_constraints(
        payload,
        tile_names=["tile0.ome.tif", "tile1.ome.tif", "tile2.ome.tif"],
        spacing_um_zyx=np.asarray([2.0, 1.0, 0.5]),
        min_quality=0.25,
    )

    assert len(constraints) == 1
    assert constraints[0]["fixed"] == "tile0.ome.tif"
    assert constraints[0]["moving"] == "tile1.ome.tif"
    assert constraints[0]["corr_after"] == 0.6
    assert constraints[0]["mvs_quality"] == 0.6
    assert constraints[0]["score_source"] == "mvs_quality"


def test_mvs_seam_constraints_ignore_phase_refined_target_delta() -> None:
    payload = mvs_registration_payload()
    edge = payload["metrics"]["pairwise_registration"]["edges"][0]
    edge["attrs"]["gradient_component_ncc_after"] = 0.7
    edge["attrs"]["target_correction_delta_px_zyx"] = [1.0, -2.0, 3.0]
    edge["attrs"]["phase_refined_bad_gradient"] = True
    edge["attrs"]["phase_refined_shift_native_px_zyx"] = [0.0, 5.0, -1.0]

    constraints = mvs_seams.mvs_seam_constraints(
        payload,
        tile_names=["tile0.ome.tif", "tile1.ome.tif", "tile2.ome.tif"],
        spacing_um_zyx=np.asarray([2.0, 1.0, 0.5]),
        min_quality=0.25,
    )

    assert len(constraints) == 1
    np.testing.assert_allclose(constraints[0]["target_correction_delta_px"], [2.0, 3.0, 4.0])
    assert "phase_refined_bad_gradient" not in constraints[0]
    assert "phase_refined_shift_native_px_zyx" not in constraints[0]


def test_recover_anchor_shift_uses_mvs_seam_graph_from_direct_anchor() -> None:
    recovered, diagnostics = mvs_seams.recover_anchor_shifts_from_mvs_seams(
        direct_anchor_shift_um_by_tile={"tile0.ome.tif": np.asarray([10.0, 20.0, 30.0])},
        mvs_registration=mvs_registration_payload(),
        spacing_um_zyx=np.asarray([2.0, 1.0, 0.5]),
        min_quality=0.25,
    )

    np.testing.assert_allclose(recovered["tile1.ome.tif"], [14.0, 23.0, 32.0])
    assert "tile2.ome.tif" not in recovered
    assert diagnostics["mvs_pairwise_edge_count"] == 1
    assert diagnostics["recovered_tile_count"] == 1


def test_mvs_used_edge_audit_reports_edges_dropped_by_global_optimization() -> None:
    payload = mvs_registration_payload()
    payload["metrics"]["groupwise_resolution"]["metrics"]["used_edges"] = {"0": [[0, 1]]}

    audit = mvs_seams.mvs_used_edge_audit(payload)

    assert audit["measured_edge_count"] == 2
    assert audit["used_edge_count"] == 1
    assert audit["dropped_edge_count"] == 1
    assert audit["dropped_edges"][0]["pair"] == [1, 2]
    assert audit["dropped_edges"][0]["residual_um"] == 12.0
    assert audit["max_measured_residual_edge"]["pair"] == [1, 2]
    assert audit["max_used_residual_edge"]["pair"] == [0, 1]
