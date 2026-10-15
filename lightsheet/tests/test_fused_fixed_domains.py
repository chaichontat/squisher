import json
from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet.fused_fixed_domains import assemble_bounded_domains, rebase_domain
from squisher_lightsheet.fusion import legacy


def _record(start, stop, fixed_start, *, translation=None):
    return {
        "status": "accepted",
        "moving_tile": "tile.ome.zarr",
        "moving_path": "/source/tile.ome.zarr",
        "moving_channel": 0,
        "fixed_fused": "/source/fixed.ome.zarr",
        "quadrant": "qy0_qx0",
        "moving_start_l0_zyx": start,
        "moving_stop_l0_zyx": stop,
        "moving_shape_zyx": (np.array(stop) - start).tolist(),
        "fixed_start_zyx": fixed_start,
        "selected_local_matrix_zyx": np.eye(3).tolist(),
        "selected_local_translation_zyx": [0, 0, 0] if translation is None else translation,
        "fused_scale_zyx": [0.6, 0.3, 0.3],
        "fused_translation_zyx": [-10, -20, 0],
        "cache_config": {"local_3d_refinement": True},
        "selected_attempt": "local_3d_affine_refinement",
    }


def _plan(tmp_path: Path):
    parent = tmp_path / "parent.json"
    child = tmp_path / "child.json"
    parent.write_text(json.dumps(_record([0, 0, 0], [12, 8, 8], [20, 30, 40])))
    child.write_text(json.dumps(_record([2, 0, 0], [7, 8, 8], [22, 30, 40], translation=[0.2, 0.1, -0.1])))
    domains = [
        {
            "parent_window_json": str(parent),
            "quadrant": "qy0_qx0",
            "transform_record_json": str(transform),
            "domain_policy": policy,
            "source_start_zyx": [lo, 0, 0],
            "source_stop_zyx": [hi, 8, 8],
        }
        for transform, policy, lo, hi in [
            (parent, "retained_published_mapping", 0, 3),
            (child, "validated_refinement", 2, 7),
            (parent, "retained_published_mapping", 6, 12),
        ]
    ]
    plan = {
        "artifact_type": "lightsheet.fused_fixed_domain_replacement_plan.v1",
        "fixed_fused": "/source/fixed.ome.zarr",
        "moving_channel": 0,
        "fixed_roi_start_zyx": [20, 30, 40],
        "fixed_roi_stop_zyx": [32, 38, 48],
        "protected_fixed_start_zyx": [23, 30, 40],
        "protected_fixed_stop_zyx": [26, 38, 48],
        "domains": domains,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


def test_retained_domain_preserves_source_to_fixed_map_with_shear() -> None:
    row = _record([960, 0, 0], [1488, 528, 528], [954, 5581, 6400], translation=[2, -3, 9])
    matrix = np.array([[1.001, 0.003, 0], [-0.002, 1, 0.001], [0, 0.001, 0.999]])
    row["selected_local_matrix_zyx"] = matrix.tolist()
    result = rebase_domain(row, start=np.array([1206, 0, 0]), stop=np.array([1488, 528, 528]))
    points = np.array([[1206, 0, 0], [1487, 527, 527], [1300, 200, 300]], dtype=float)
    center = (np.array(row["moving_shape_zyx"]) - 1) / 2
    expected = (points - np.array(row["moving_start_l0_zyx"]) - center) @ matrix.T + center
    expected += np.array(row["selected_local_translation_zyx"]) + np.array(row["fixed_start_zyx"])
    pull = np.array(result["selected_fixed_fused_l0_to_moving_l0_pull_matrix_zyx"])
    offset = np.array(result["selected_fixed_fused_l0_to_moving_l0_pull_offset_zyx"])
    np.testing.assert_allclose(expected @ pull.T + offset, points, atol=1e-9)
    assert result["moving_shape_zyx"] == [282, 528, 528]


def test_assembly_preserves_coverage_and_distinguishes_retention_from_refitting(tmp_path: Path) -> None:
    path, _ = _plan(tmp_path)
    summary = json.loads(assemble_bounded_domains(plan_path=path, output_dir=tmp_path / "out").read_text())
    assert summary["parent_source_coverage_complete"]
    assert summary["superseded_parent_support_excluded_from_protected_field"]
    assert len(summary["windows"]) == 3
    rows = [json.loads(Path(row["level0_json"]).read_text()) for row in summary["windows"]]
    assert [row["domain_policy"] for row in rows] == [
        "retained_published_mapping",
        "validated_refinement",
        "retained_published_mapping",
    ]
    assert [row["moving_shape_zyx"][0] for row in rows] == [3, 5, 6]
    assert all("method8_attempts" not in row for row in rows)


@pytest.mark.parametrize(
    "failure", ["gap", "old_parent_in_protected", "wrong_retained_transform", "rejected_refinement"]
)
def test_assembly_rejects_invalid_replacement_contract(tmp_path: Path, failure: str) -> None:
    path, plan = _plan(tmp_path)
    if failure == "gap":
        plan["domains"][0]["source_stop_zyx"][0] = 1
    elif failure == "old_parent_in_protected":
        plan["domains"][0]["source_stop_zyx"][0] = 12
    elif failure == "wrong_retained_transform":
        plan["domains"][0]["transform_record_json"] = str(tmp_path / "child.json")
    else:
        child = tmp_path / "child.json"
        row = json.loads(child.read_text())
        row["status"] = "rejected"
        child.write_text(json.dumps(row))
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        assemble_bounded_domains(plan_path=path, output_dir=tmp_path / "out")


def test_bounded_fusion_grid_uses_recorded_spacing_and_native_origin(tmp_path: Path) -> None:
    import zarr

    group = zarr.open_group(tmp_path / "template.zarr", mode="w")
    group.create_array("0", shape=(20, 30, 40), dtype="uint16")
    group.attrs["multiscales"] = [
        {
            "axes": [{"name": axis, "type": "space", "unit": "micrometer"} for axis in "zyx"],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [0.6, 0.3, 0.2]},
                        {"type": "translation", "translation": [-19.7, -0.3, 0]},
                    ],
                }
            ],
        }
    ]
    props = legacy.output_stack_properties_from_ome_zarr_template(
        tmp_path / "template.zarr", {"z": 1, "y": 1, "x": 1}, bounds_zyx=(2, 3, 4, 12, 13, 14)
    )
    assert props["shape"] == {"z": 10, "y": 10, "x": 10}
    np.testing.assert_allclose([props["origin"][axis] for axis in "zyx"], [-18.5, 0.6, 0.8])
    assert props["spacing"] == {"z": 0.6, "y": 0.3, "x": 0.2}
    with pytest.raises(ValueError, match="outside template"):
        legacy.output_stack_properties_from_ome_zarr_template(
            tmp_path / "template.zarr", {}, bounds_zyx=(2, 3, 4, 21, 13, 14)
        )
