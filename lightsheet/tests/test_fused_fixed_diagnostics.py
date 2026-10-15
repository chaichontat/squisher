from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet.fused_fixed_diagnostics import write_overlap_disagreement


@pytest.fixture
def artifacts(tmp_path: Path):
    fixed = tmp_path / "fixed.zarr"
    sources = []
    geometry = []
    for index in range(2):
        source = {
            "tile": f"source{index}",
            "path": str(tmp_path / f"raw{index}.zarr"),
            "axes": "CZYX",
            "shape": [2, 40, 30, 20],
            "scale_um": {"z": 0.6, "y": 0.2, "x": 0.3},
            "translation_um": {"z": 0, "y": 0, "x": 0},
        }
        sources.append(source)
        geometry.append(source | {"tile": f"tile{index}"})
    source_path = tmp_path / "sources.json"
    geometry_path = tmp_path / "geometry.json"
    source_path.write_text(json.dumps({"tiles": sources}))
    geometry_path.write_text(json.dumps({"tiles": geometry}))
    summary = {
        "fixed_fused": str(fixed),
        "moving_position": str(geometry_path),
        "moving_source_position": str(source_path),
        "windows": [],
    }

    def add(
        name,
        start=(0, 0, 0),
        stop=(12, 12, 12),
        *,
        source=0,
        channel=0,
        matrix=None,
        offset=(0, 0, 0),
        status="accepted",
    ):
        matrix = np.eye(3) if matrix is None else np.asarray(matrix)
        start, stop, offset = np.asarray(start), np.asarray(stop), np.asarray(offset)
        center = (stop - start - 1) / 2
        fixed_start = start + [2, 3, 4]
        translation = offset + matrix @ start - fixed_start - center + matrix @ center
        row = {
            "status": status,
            "moving_tile": f"tile{source}",
            "moving_path": sources[source]["path"],
            "moving_channel": channel,
            "moving_start_l0_zyx": start.tolist(),
            "moving_stop_l0_zyx": stop.tolist(),
            "moving_shape_zyx": (stop - start).tolist(),
            "fixed_fused": str(fixed),
            "fixed_level": 0,
            "fixed_start_zyx": fixed_start.tolist(),
            "fused_scale_zyx": [0.6, 0.2, 0.3],
            "fused_translation_zyx": [10, 20, 30],
            "selected_moving_l0_to_fixed_fused_l0_matrix_zyx": matrix.tolist(),
            "selected_moving_l0_to_fixed_fused_l0_offset_zyx": offset.tolist(),
            "selected_local_matrix_zyx": matrix.tolist(),
            "selected_local_translation_zyx": translation.tolist(),
            "selected_attempt": "interpolated_method6_spatial_recovery",
            "outlier_recovery": {"decision": "original_retained", "residual_verdict": "unverified"},
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(row))
        summary["windows"].append({"level0_json": str(path), "status": status})
        return path, row

    def run():
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(summary))
        result = write_overlap_disagreement(path, tmp_path / "diagnostic.json")
        return json.loads(result.read_text())

    return add, run, geometry_path, summary


def test_differently_rebased_identical_global_maps_have_zero_disagreement(artifacts):
    add, run, _geometry, _summary = artifacts
    first_path, first = add("first", matrix=[[1, 0, 0], [0.03, 1, 0], [0, 0, 1]], offset=[1, 2, 3])
    _second_path, second = add(
        "second",
        start=(8, 0, 0),
        stop=(20, 12, 12),
        matrix=[[1, 0, 0], [0.03, 1, 0], [0, 0, 1]],
        offset=[1, 2, 3],
    )
    assert first["selected_local_translation_zyx"] != second["selected_local_translation_zyx"]
    before = first_path.read_text()
    result = run()
    assert result["overlap_pair_count"] == 1
    pair = result["pairs"][0]
    assert pair["overlap_start_raw_l0_zyx"] == [8, 0, 0]
    assert pair["overlap_stop_raw_l0_zyx"] == [12, 12, 12]
    assert len(pair["samples"]) == 9
    assert all(sample["disagreement_um"] == 0 for sample in pair["samples"])
    assert pair["first"]["outlier_recovery"] == first["outlier_recovery"]
    assert first_path.read_text() == before


def test_known_shear_translation_uses_common_points_and_anisotropic_fixed_spacing(artifacts):
    add, run, _geometry, _summary = artifacts
    add("first")
    matrix = np.eye(3)
    matrix[0, 2] = 0.1
    add("second", start=(8, 0, 0), stop=(20, 12, 12), matrix=matrix, offset=[2, -3, 4])
    for sample in run()["pairs"][0]["samples"]:
        point = np.asarray(sample["source_raw_l0_zyx"])
        expected = np.array([0.6 * (2 + 0.1 * point[2]), -0.6, 1.2])
        np.testing.assert_allclose(sample["second_minus_first_zyx_um"], expected, atol=1e-12)
        assert sample["disagreement_um"] == pytest.approx(np.linalg.norm(expected))
        np.testing.assert_allclose(sample["first_target_zyx_um"], [10, 20, 30] + point * [0.6, 0.2, 0.3])


def test_negative_source_orientation_evaluates_identical_raw_voxel_centers(artifacts):
    add, run, geometry_path, _summary = artifacts
    geometry = json.loads(geometry_path.read_text())
    geometry["tiles"][0]["scale_um"] = {"z": -0.6, "y": 0.2, "x": -0.3}
    geometry_path.write_text(json.dumps(geometry))
    add("first")
    add("second", start=(8, 0, 0), stop=(20, 12, 12), offset=[1, 2, 3])
    pair = run()["pairs"][0]
    assert pair["overlap_start_raw_l0_zyx"] == [28, 0, 8]
    assert pair["overlap_stop_raw_l0_zyx"] == [32, 12, 20]
    for sample in pair["samples"]:
        raw = np.asarray(sample["source_raw_l0_zyx"])
        oriented = raw * np.array([-1, 1, -1]) + [39, 0, 19]
        first_is_shifted = pair["first"]["window_json"].endswith("second.json")
        expected = [10, 20, 30] + (oriented + ([1, 2, 3] if first_is_shifted else [0, 0, 0])) * [
            0.6,
            0.2,
            0.3,
        ]
        np.testing.assert_allclose(sample["first_target_zyx_um"], expected)
        assert sample["disagreement_um"] == pytest.approx(np.linalg.norm([0.6, 0.4, 0.9]))


def test_disjoint_domains_different_sources_channels_and_rejections_do_not_pair(artifacts):
    add, run, _geometry, _summary = artifacts
    add("first")
    add("disjoint", start=(12, 0, 0), stop=(24, 12, 12))
    add("other_source", source=1)
    add("other_channel", channel=1)
    path, _row = add("rejected", status="rejected")
    path.write_text("incomplete rejected transform")
    result = run()
    assert result["accepted_window_count"] == 4
    assert result["same_source_channel_group_count"] == 3
    assert result["pairs"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("selected_moving_l0_to_fixed_fused_l0_offset_zyx", None),
        ("selected_moving_l0_to_fixed_fused_l0_matrix_zyx", np.zeros((3, 3)).tolist()),
        ("fused_scale_zyx", [0, 0.2, 0.3]),
        ("fused_translation_zyx", [float("nan"), 0, 0]),
        ("moving_stop_l0_zyx", [50, 12, 12]),
        ("moving_start_l0_zyx", [0.5, 0, 0]),
        ("moving_channel", 2),
        ("fixed_level", 2),
        ("status", "rejected"),
    ],
)
def test_incomplete_or_malformed_accepted_state_rejects_even_without_overlap(artifacts, field, value):
    add, run, _geometry, _summary = artifacts
    path, row = add("first")
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        run()


def test_missing_position_state_rejects_instead_of_reporting_no_overlap(artifacts):
    add, run, _geometry, summary = artifacts
    add("first")
    del summary["moving_source_position"]
    with pytest.raises(ValueError, match="nonempty path"):
        run()


def test_incomplete_summary_rejects(artifacts):
    _add, run, _geometry, summary = artifacts
    summary["windows"] = [{"status": "pending"}]
    with pytest.raises(ValueError, match="completed window statuses"):
        run()


def test_duplicate_accepted_window_rejects(artifacts):
    add, run, _geometry, summary = artifacts
    add("first")
    summary["windows"].append(summary["windows"][0].copy())
    with pytest.raises(ValueError, match="repeats accepted window"):
        run()


def test_empty_completed_summary_reports_no_overlap(artifacts):
    _add, run, _geometry, _summary = artifacts
    result = run()
    assert result["accepted_window_count"] == 0
    assert result["overlap_pair_count"] == 0


@pytest.mark.parametrize("field,value", [("shape", None), ("shape", [2, 40.5, 30, 20]), ("axes", "YX")])
def test_invalid_required_source_metadata_rejects(artifacts, field, value):
    add, run, _geometry, summary = artifacts
    add("first")
    path = Path(summary["moving_source_position"])
    payload = json.loads(path.read_text())
    payload["tiles"][0][field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        run()


def test_inconsistent_fixed_grid_rejects(artifacts):
    add, run, _geometry, _summary = artifacts
    add("first")
    path, row = add("second")
    row["fused_translation_zyx"] = [0, 0, 0]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="fixed geometry disagrees"):
        run()


def test_output_cannot_overwrite_source_position_metadata(artifacts, tmp_path):
    add, _run, _geometry, summary = artifacts
    add("first")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    source_position = Path(summary["moving_source_position"])
    before = source_position.read_text()
    with pytest.raises(ValueError, match="differ from its input artifacts"):
        write_overlap_disagreement(summary_path, source_position)
    assert source_position.read_text() == before
