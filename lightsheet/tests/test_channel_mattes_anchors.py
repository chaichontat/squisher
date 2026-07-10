from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from squisher_lightsheet.channel_mattes_anchors import (
    MattesAnchorParameters,
    RoundTiles,
    add_total_shift_fields,
    candidate_rows_for_site,
    complete_linkage_inliers,
    median_anchor_from_inlier_blocks,
    ranked_measured_results,
    rename_fit_result,
    start_shift_from_row,
    starts_for_row,
)


def test_start_shift_prefers_phasecorr_direct_over_phasecorr_init() -> None:
    row = {
        "phasecorr_direct_shift_level_vox_zyx": [1.0, 2.0, 3.0],
        "phasecorr_init_shift_level_vox_zyx": [9.0, 9.0, 9.0],
    }

    assert np.allclose(start_shift_from_row(row), [1.0, 2.0, 3.0])


def test_starts_for_row_preserves_current_multiple_start_policy() -> None:
    row = {
        "phasecorr_direct_shift_level_vox_zyx": [1.0, 2.0, 3.0],
        "residual_shift_level_vox_zyx": [4.0, 5.0, 6.0],
    }
    known_good = [
        {
            "block_index": 0,
            "best_by_selection_metric": {"shift_to_apply_moving_zyx_level3_px": [7.0, 8.0, 9.0]},
        },
        {
            "block_index": 1,
            "best_by_selection_metric": {"shift_to_apply_moving_zyx_level3_px": [9.0, 10.0, 11.0]},
        },
    ]

    starts = starts_for_row(row, known_good)

    assert [start["name"] for start in starts] == [
        "block_phasecorr",
        "block_previous_residual",
        "good_chunk0_best_shift",
        "good_chunk1_best_shift",
        "good_chunk_median",
    ]
    assert starts[-1]["shift_to_apply_moving_zyx_level3_px"] == [8.0, 9.0, 10.0]


def test_starts_for_row_deduplicates_by_rounded_shift() -> None:
    row = {
        "phasecorr_direct_shift_level_vox_zyx": [1.0, 2.0, 3.0],
        "residual_shift_level_vox_zyx": [1.0004, 2.0004, 3.0004],
    }

    starts = starts_for_row(row, [])

    assert [start["name"] for start in starts] == ["block_phasecorr"]


def test_complete_linkage_inliers_matches_largest_compatible_group() -> None:
    parameters = MattesAnchorParameters(min_inlier_chunks=3, inlier_threshold_level_px_zyx=(3.0, 12.0, 12.0))
    records = [
        {
            "block_index": 0,
            "best_by_selection_metric": {
                "mip_gradient_component_ncc_moving": 0.2,
                "shift_to_apply_moving_zyx_level3_px": [0.0, 0.0, 0.0],
            },
        },
        {
            "block_index": 1,
            "best_by_selection_metric": {
                "mip_gradient_component_ncc_moving": 0.2,
                "shift_to_apply_moving_zyx_level3_px": [1.0, 8.0, 8.0],
            },
        },
        {
            "block_index": 2,
            "best_by_selection_metric": {
                "mip_gradient_component_ncc_moving": 0.2,
                "shift_to_apply_moving_zyx_level3_px": [2.0, 10.0, 10.0],
            },
        },
        {
            "block_index": 3,
            "best_by_selection_metric": {
                "mip_gradient_component_ncc_moving": 0.2,
                "shift_to_apply_moving_zyx_level3_px": [20.0, 80.0, 80.0],
            },
        },
    ]

    assert complete_linkage_inliers(records, parameters=parameters) == [0, 1, 2]


def test_complete_linkage_inliers_caps_at_max_inlier_chunks() -> None:
    parameters = MattesAnchorParameters(
        min_inlier_chunks=3,
        max_inlier_chunks=5,
        inlier_threshold_level_px_zyx=(3.0, 12.0, 12.0),
    )
    records = [
        {
            "block_index": index,
            "best_by_selection_metric": {
                "mip_gradient_component_ncc_moving": 0.2,
                "shift_to_apply_moving_zyx_level3_px": [float(index) * 0.1, float(index), float(index)],
            },
        }
        for index in range(6)
    ]

    assert complete_linkage_inliers(records, parameters=parameters) == [0, 1, 2, 3, 4]


def test_median_anchor_from_inlier_blocks_uses_selected_inlier_median() -> None:
    records = [
        {
            "block_index": 0,
            "best_by_selection_metric": {
                "metric": "mattes_mi",
                "optimizer": "regular_step",
                "mip_gradient_component_ncc_moving": 0.3,
                "shift_to_apply_moving_zyx_level3_px": [0.0, 10.0, 10.0],
            },
        },
        {
            "block_index": 1,
            "best_by_selection_metric": {
                "metric": "mattes_mi",
                "optimizer": "regular_step",
                "mip_gradient_component_ncc_moving": 0.4,
                "shift_to_apply_moving_zyx_level3_px": [2.0, 12.0, 12.0],
            },
        },
        {
            "block_index": 2,
            "best_by_selection_metric": {
                "metric": "mattes_mi",
                "optimizer": "regular_step",
                "mip_gradient_component_ncc_moving": 0.5,
                "shift_to_apply_moving_zyx_level3_px": [100.0, 100.0, 100.0],
            },
        },
    ]

    anchor = median_anchor_from_inlier_blocks(records, [0, 1])

    assert anchor is not None
    assert anchor["aggregation"] == "median"
    assert anchor["inlier_chunk_count"] == 2
    assert anchor["shift_to_apply_moving_zyx_level3_px"] == [1.0, 11.0, 11.0]


def test_candidate_rows_merge_existing_seed_by_world_center() -> None:
    parameters = MattesAnchorParameters(
        candidate_source="structure_tensor",
        level=3,
        patch_shape_zyx=(12, 160, 160),
        max_attempted_chunks=24,
    )
    reference = RoundTiles(
        label="488",
        position_path=Path("/reference.positions.json"),
        payload={},
        poses={"L:0001": SimpleNamespace(spacing_um=np.asarray([1.0, 1.0, 1.0]))},
    )
    seed = {
        "tile_site": "L:0001",
        "block_index": 99,
        "world_origin_um_zyx": [0.0, 0.0, 0.0],
        "patch_shape_zyx": [12, 160, 160],
        "level_spacing_um_zyx": [8.0, 8.0, 8.0],
        "phasecorr_init_shift_level_vox_zyx": [1.0, 2.0, 3.0],
    }
    candidate_metrics = [
        {
            "center_um_zyx": [48.0, 640.0, 640.0],
            "structure_tensor_score": 5.0,
            "high_frequency_content_score": 4.0,
            "content_score": 3.0,
        }
    ]

    rows = candidate_rows_for_site(
        "L:0001",
        [seed],
        reference=reference,
        parameters=parameters,
        candidate_metrics=candidate_metrics,
    )

    assert rows[0]["source_anchor_block_index"] == 99
    assert rows[0]["source"] == "generated_gpu_highpass_structure_tensor_candidate_matched_existing_anchor"
    assert rows[0]["block_index"] == 0
    assert rows[0]["phasecorr_init_shift_level_vox_zyx"] == [1.0, 2.0, 3.0]
    assert rows[0]["applied_translation_um_zyx"] == [0.0, 0.0, 0.0]


def test_candidate_rows_inherit_site_bootstrap_for_unmatched_generated_candidates() -> None:
    parameters = MattesAnchorParameters(
        candidate_source="structure_tensor",
        level=3,
        patch_shape_zyx=(12, 160, 160),
        max_attempted_chunks=24,
    )
    reference = RoundTiles(
        label="488",
        position_path=Path("/reference.positions.json"),
        payload={},
        poses={"L:0001": SimpleNamespace(spacing_um=np.asarray([1.0, 1.0, 1.0]))},
    )
    seed = {
        "tile_site": "L:0001",
        "block_index": 99,
        "world_origin_um_zyx": [10_000.0, 10_000.0, 10_000.0],
        "patch_shape_zyx": [12, 160, 160],
        "level_spacing_um_zyx": [8.0, 8.0, 8.0],
        "bootstrap_translation_um_zyx": [-2.0, -0.5, 0.5],
    }
    candidate_metrics = [
        {
            "center_um_zyx": [48.0, 640.0, 640.0],
            "structure_tensor_score": 5.0,
            "high_frequency_content_score": 4.0,
            "content_score": 3.0,
        }
    ]

    rows = candidate_rows_for_site(
        "L:0001",
        [seed],
        reference=reference,
        parameters=parameters,
        candidate_metrics=candidate_metrics,
    )

    assert rows[0]["bootstrap_translation_um_zyx"] == [-2.0, -0.5, 0.5]
    assert rows[0]["applied_translation_um_zyx"] == [-2.0, -0.5, 0.5]


def test_candidate_rows_default_to_seed_rows_with_applied_translation() -> None:
    parameters = MattesAnchorParameters(level=3, patch_shape_zyx=(12, 160, 160), max_attempted_chunks=1)
    reference = RoundTiles(
        label="488",
        position_path=Path("/reference.positions.json"),
        payload={},
        poses={"L:0001": SimpleNamespace(spacing_um=np.asarray([1.0, 1.0, 1.0]))},
    )
    seed_rows = [
        {
            "tile_site": "L:0001",
            "block_index": 2,
            "world_origin_um_zyx": [20.0, 0.0, 0.0],
            "bootstrap_translation_um_zyx": [-2.0, -0.5, 0.5],
            "anchor_prior_offset_um_zyx": [0.0, 1.0, 2.0],
        },
        {
            "tile_site": "L:0001",
            "block_index": 1,
            "world_origin_um_zyx": [10.0, 0.0, 0.0],
            "bootstrap_translation_um_zyx": [-2.0, -0.5, 0.5],
            "applied_translation_um_zyx": [9.0, 9.0, 9.0],
        },
    ]

    rows = candidate_rows_for_site("L:0001", seed_rows, reference=reference, parameters=parameters)

    assert len(rows) == 1
    assert rows[0]["block_index"] == 1
    assert rows[0]["applied_translation_um_zyx"] == [9.0, 9.0, 9.0]


def test_add_total_shift_fields_records_applied_plus_residual() -> None:
    result = {"shift_to_apply_moving_um_zyx": [1.0, 2.0, 3.0]}
    row = {"applied_translation_um_zyx": [-2.0, -0.5, 0.5]}

    add_total_shift_fields(result, row)

    assert result["applied_translation_um_zyx"] == [-2.0, -0.5, 0.5]
    assert result["total_shift_to_apply_moving_um_zyx"] == [-1.0, 1.5, 3.5]


def test_ranked_measured_results_uses_gradient_ncc_descending() -> None:
    results = [
        {"status": "measured", "mip_gradient_component_ncc_moving": 0.2, "name": "low"},
        {"status": "failed", "mip_gradient_component_ncc_moving": 1.0, "name": "failed"},
        {"status": "measured", "mip_gradient_component_ncc_moving": None, "name": "none"},
        {"status": "measured", "mip_gradient_component_ncc_moving": 0.7, "name": "high"},
    ]

    ranked = ranked_measured_results(results)

    assert [item["name"] for item in ranked] == ["high", "low"]


def test_rename_fit_result_preserves_current_level3_key_contract() -> None:
    result = rename_fit_result(
        {
            "initial_shift_to_apply_moving_zyx_level2_px": [1, 2, 3],
            "residual_shift_to_apply_moving_zyx_level2_px": [4, 5, 6],
            "shift_to_apply_moving_zyx_level2_px": [7, 8, 9],
            "metric": "mattes_mi",
        }
    )

    assert result["initial_shift_to_apply_moving_zyx_level3_px"] == [1, 2, 3]
    assert result["residual_shift_to_apply_moving_zyx_level3_px"] == [4, 5, 6]
    assert result["shift_to_apply_moving_zyx_level3_px"] == [7, 8, 9]
    assert not any(key.endswith("level2_px") for key in result)
