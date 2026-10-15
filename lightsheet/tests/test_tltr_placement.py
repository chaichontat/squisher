from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from squisher_lightsheet import tltr_placement


def _registered(tile: str, *, x: float, path: Path) -> dict:
    return {
        "tile": tile,
        "path": str(path),
        "translation_um": {"z": 0.0, "y": 0.0, "x": x},
        "scale_um": {"z": 1.8, "y": 0.3, "x": 0.3},
        "shape": [1, 10, 8, 8],
        "axes": "CZYX",
        "channels": ["0"],
        "tracks": [],
    }


NONE = {"TL": set(), "TR": set()}


def _metadata() -> dict:
    # Metadata abuts TR beside TL along X; Z and Y centroids already agree.
    return {
        "tiles": [
            {"tile": "TL.000.ome.tif", "side": "L", "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
             "scale_um": {"z": 1.8, "y": 0.3, "x": 0.3}},
            {"tile": "TR.000.ome.tif", "side": "R", "translation_um": {"z": 0.0, "y": 0.0, "x": 100.0},
             "scale_um": {"z": 1.8, "y": 0.3, "x": 0.3}},
        ]
    }


def test_sides_are_centred_on_their_metadata_placement_and_tr_takes_the_shift(tmp_path: Path) -> None:
    left = [_registered("TL.000.ome.zarr", x=5.0, path=tmp_path / "TL.000.ome.zarr")]
    right = [_registered("TR.000.ome.zarr", x=-7.0, path=tmp_path / "TR.000.ome.zarr")]

    payload = tltr_placement.join_sides(
        metadata=_metadata(), left_records=left, right_records=right, right_shift_zyx_um=np.array([1.0, 2.0, 3.0]), excluded=NONE
    )

    by_side = {record["source_side"]: record for record in payload["tiles"]}
    assert by_side["TL"]["translation_um"]["x"] == pytest.approx(0.0)
    assert by_side["TR"]["translation_um"] == pytest.approx({"z": 1.0, "y": 2.0, "x": 103.0})
    assert by_side["TR"]["side"] == "R" and by_side["TL"]["side"] == "L"
    assert payload["diagnostics"]["join_axis"] == "x" and payload["diagnostics"]["right_flip_axes"] == []


def test_joining_refuses_registered_tiles_that_differ_from_the_metadata(tmp_path: Path) -> None:
    left = [_registered("TL.001.ome.zarr", x=0.0, path=tmp_path / "a")]
    right = [_registered("TR.000.ome.zarr", x=0.0, path=tmp_path / "b")]

    with pytest.raises(ValueError, match="TL metadata tiles"):
        tltr_placement.join_sides(
            metadata=_metadata(), left_records=left, right_records=right, right_shift_zyx_um=np.zeros(3), excluded=NONE
        )


def test_tiles_native_registration_excluded_are_left_out_of_the_centroid(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["tiles"].append(
        {"tile": "TL.001.ome.tif", "side": "L", "translation_um": {"z": 0.0, "y": 0.0, "x": 500.0},
         "scale_um": {"z": 1.8, "y": 0.3, "x": 0.3}}
    )
    left = [_registered("TL.000.ome.zarr", x=5.0, path=tmp_path / "a")]
    right = [_registered("TR.000.ome.zarr", x=0.0, path=tmp_path / "b")]

    payload = tltr_placement.join_sides(
        metadata=metadata,
        left_records=left,
        right_records=right,
        right_shift_zyx_um=np.zeros(3),
        excluded={"TL": {"TL.001.ome.zarr"}, "TR": set()},
    )

    # TL.001 was excluded, so TL.000 alone defines the TL centroid and lands on its metadata placement.
    assert next(r for r in payload["tiles"] if r["source_side"] == "TL")["translation_um"]["x"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="TL metadata tiles"):
        tltr_placement.join_sides(
            metadata=metadata, left_records=left, right_records=right, right_shift_zyx_um=np.zeros(3), excluded=NONE
        )


def _coarse(tmp_path: Path, *, phase: list[float] | None) -> dict:
    payload = tltr_placement.join_sides(
        metadata=_metadata(),
        left_records=[_registered("TL.000.ome.zarr", x=0.0, path=tmp_path / "TL.000.ome.zarr")],
        right_records=[_registered("TR.000.ome.zarr", x=0.0, path=tmp_path / "TR.000.ome.zarr")],
        right_shift_zyx_um=np.zeros(3),
        excluded=NONE,
    )
    if phase is not None:
        payload["diagnostics"]["level2_zyx_phase_alignment"] = {"shift_to_apply_R_um_zyx": phase}
        for record in payload["tiles"]:
            if record["side"] == "R":
                record["translation_um"] = {
                    axis: record["translation_um"][axis] + value for axis, value in zip("zyx", phase, strict=True)
                }
    return payload


def _method8(moving: str = "TR.000.ome.zarr") -> dict:
    return {
        "windows": [
            {"status": "accepted", "fixed_tile": "TL.000.ome.zarr", "moving_tile": moving,
             "local_translation_zyx": [1.0, 0.0, 10.0]},
            {"status": "rejected", "fixed_tile": "x", "moving_tile": "y"},
        ]
    }


def test_placement_adds_the_method8_median_to_tr_and_records_the_net_shift(tmp_path: Path) -> None:
    payload = tltr_placement.build_position_payload(
        coarse=_coarse(tmp_path, phase=[0.5, -1.0, 4.0]), method8_summary=_method8(), deconv_root=tmp_path, inputs={}
    )

    right = next(record for record in payload["tiles"] if record["source_side"] == "TR")
    # 1 Z voxel = 1.8 um and 10 X voxels = 3 um in TL voxels, on top of the phase shift.
    assert right["translation_um"] == pytest.approx({"z": 2.3, "y": -1.0, "x": 107.0})
    assert payload["alignment"]["net_right_shift_after_centroid_um_zyx"] == pytest.approx([2.3, -1.0, 7.0])
    assert right["path"] == str(tmp_path / "TR.000.ome.zarr")


def test_placement_rejects_method8_rows_outside_the_tl_tr_pairing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not pair a TL fixed tile with a TR moving tile"):
        tltr_placement.build_position_payload(
            coarse=_coarse(tmp_path, phase=None),
            method8_summary=_method8(moving="TL.000.ome.zarr"),
            deconv_root=tmp_path,
            inputs={},
        )


def test_publication_writes_positions_identity_registration_and_overlay(tmp_path: Path) -> None:
    deconv = tmp_path / "deconv"
    for name in ("TL.000.ome.zarr", "TR.000.ome.zarr"):
        group = zarr.open_group(str(deconv / name), mode="w")
        group.create_array("2", data=np.ones((1, 10, 8, 8), dtype=np.uint16))
    coarse = tmp_path / "coarse.json"
    coarse.write_text(json.dumps(_coarse(tmp_path, phase=None)))
    method8 = tmp_path / "method8.json"
    method8.write_text(json.dumps(_method8()))

    outputs = tltr_placement.publish_tltr_placement(
        coarse=coarse, method8_summary=method8, deconv_root=deconv, output_dir=tmp_path / "final", qc_channel=0, qc_level=2
    )

    registration = json.loads(Path(outputs["registration"]).read_text())
    assert {record["source_view"] for record in registration["tiles"]} == {"L", "R"}
    assert Path(outputs["overlay"]).name == "tltr_channel0_level2_zmedian_TLgreen_TRred.png"
    split = tltr_placement.split_sides(Path(outputs["positions"]), tmp_path / "sides")
    assert [len(json.loads(path.read_text())["tiles"]) for path in split.values()] == [1, 1]


def test_reviewed_affine_replaces_median_and_preserves_source_stage(tmp_path: Path) -> None:
    coarse = _coarse(tmp_path, phase=[0.5, -1.0, 4.0])
    affine = np.eye(4)
    affine[0, 1] = 0.002
    affine[:3, 3] = [2, -3, 4]
    payload = tltr_placement.build_position_payload(
        coarse=coarse, method8_summary=_method8(), deconv_root=tmp_path, inputs={},
        global_affine=affine,
    )
    registration = tltr_placement.build_placement_registration(payload, deconv_root=tmp_path)
    for original, placed, registered in zip(coarse["tiles"], payload["tiles"], registration["tiles"], strict=True):
        assert placed["translation_um"] == original["translation_um"]
        assert registered["stage_translation_um"] == original["translation_um"]
        expected = affine if original["source_side"] == "TR" else np.eye(4)
        np.testing.assert_allclose(registered["registered_affine"]["matrix"], expected)
    # 638's stage-conjugated calibration must retain the side transform exactly once.
    from squisher_lightsheet.channel_affine import compose_registration_affine
    record = registration["tiles"][1]
    stage = np.array([record["stage_translation_um"][axis] for axis in "zyx"])
    stage_matrix = np.eye(4)
    stage_matrix[:3, 3] = stage
    channel = np.eye(4)
    channel[1, 2] = 0.003
    channel[:3, 3] = [1, 2, 3]
    adapted = compose_registration_affine(reference_affine=record["registered_affine"],
        channel_affine_um=channel, stage_translation_um_zyx=stage)
    np.testing.assert_allclose(np.asarray(adapted["matrix"]) @ stage_matrix,
        affine @ stage_matrix @ channel)


def test_affine_initializer_maps_native_points_to_the_canonical_frame() -> None:
    from squisher_lightsheet._backends import fused_fixed_trial as trial
    record = _registered("TR.000.ome.zarr", x=100, path=Path("/data/a"))
    affine = np.eye(4)
    affine[0, 1] = 0.002
    affine[2, 0] = -0.001
    affine[:3, 3] = [2, -3, 4]
    record["placement_affine_zyx_um"] = affine.tolist()
    start, shape = np.array([10, 20, 30]), np.array([16, 20, 24])
    factor = np.array([4, 4, 4])
    scale = np.array([1.8, .3, .3])
    fused_scale = scale * factor
    fused_origin = np.array([-5, -8, -10])
    fixed_start = np.array([3, 4, 5])
    matrix, translation = trial._placement_crop_initializer(
        record, moving_start_l0=start, fit_shape=shape, level_factor=factor,
        fused_scale=fused_scale, fused_translation=fused_origin, fixed_start=fixed_start,
        channel_matrix=np.eye(3),
    )
    center = (shape - 1) / 2
    for point in (np.zeros(3), center, shape - 1):
        world = np.array([0, 0, 100]) + scale * (start + factor * point)
        expected = (affine[:3, :3] @ world + affine[:3, 3] - fused_origin) / fused_scale
        actual = matrix @ (point - center) + center + translation + fixed_start
        np.testing.assert_allclose(actual, expected, atol=1e-5)


def _affine_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    coarse = tmp_path / "coarse.json"
    coarse.write_text(json.dumps(_coarse(tmp_path, phase=None)))
    split = tltr_placement.split_sides(coarse, tmp_path / "sides")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_method8()))
    candidate = tmp_path / "affine.json"
    candidate.write_text(json.dumps({
        "artifact_type": "lightsheet.spatial_overlap_global_affine.v1",
        "coordinate_contract": "target_physical_zyx_um = matrix_zyx @ source_physical_zyx_um + translation_zyx_um",
        "source_position": str(split["TR"]), "target_position": str(split["TL"]),
        "matrix_zyx": np.eye(3).tolist(), "translation_zyx_um": [1, 2, 3],
        "fit": {"accepted_correspondence_count": 1},
    }))
    return coarse, summary, candidate


@pytest.mark.parametrize("change, message", [
    ({"matrix_zyx": np.diag([-1, 1, 1]).tolist()}, "preserve orientation"),
    ({"matrix_zyx": np.diag([0, 1, 1]).tolist()}, "preserve orientation"),
    ({"matrix_zyx": np.diag([2, 1, 1]).tolist()}, "10% scale"),
    ({"translation_zyx_um": [float("nan"), 0, 0]}, "finite"),
    ({"coordinate_contract": "inverse"}, "source-to-target"),
    ({"fit": {"accepted_correspondence_count": 2}}, "correspondence count"),
])
def test_reviewed_affine_rejects_invalid_contract(tmp_path: Path, change: dict, message: str) -> None:
    coarse, summary, candidate = _affine_inputs(tmp_path)
    candidate.write_text(json.dumps(json.loads(candidate.read_text()) | change))
    with pytest.raises(ValueError, match=message):
        tltr_placement.load_reviewed_global_affine(candidate, coarse=coarse, method8_summary=summary)


def test_reviewed_affine_rejects_stale_coarse_inputs(tmp_path: Path) -> None:
    coarse, summary, candidate = _affine_inputs(tmp_path)
    payload = json.loads(coarse.read_text())
    payload["tiles"][1]["translation_um"]["x"] += 1
    coarse.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inputs differ"):
        tltr_placement.load_reviewed_global_affine(candidate, coarse=coarse, method8_summary=summary)


def test_affine_publication_uses_full_volume_resampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from squisher_lightsheet import legacy_runner
    coarse, summary, candidate = _affine_inputs(tmp_path)
    deconv = tmp_path / "deconv"
    for name in ("TL.000.ome.zarr", "TR.000.ome.zarr"):
        (deconv / name).mkdir(parents=True)
    rendered = []
    monkeypatch.setattr(legacy_runner, "run_legacy_script", lambda script, args: rendered.append((script, args)))
    result = tltr_placement.publish_tltr_placement(
        coarse=coarse, method8_summary=summary, deconv_root=deconv, output_dir=tmp_path / "final",
        qc_channel=0, qc_level=2, reviewed_global_affine=candidate,
    )
    assert "--full-affine-planes" in rendered[0][1]
    assert "--center-y-xz" in rendered[0][1]
    registration = json.loads(Path(result["registration"]).read_text())
    np.testing.assert_allclose(np.asarray(registration["tiles"][1]["registered_affine"]["matrix"])[:3, 3], [1, 2, 3])
    assert "qc-affine" in result["overlay"]


def test_level2_refinement_preserves_canonical_map_at_native_spacing() -> None:
    from squisher_lightsheet._backends import fused_fixed_trial as trial
    matrix = np.eye(3)
    matrix[0, 1] = .002
    level2 = {"local_matrix_zyx": matrix.tolist(), "local_translation_zyx": [.3, -.2, 1.1],
        "level_factor_zyx": [4, 4, 4], "fixed_shape_zyx": [16, 20, 24], "fixed_start_zyx": [10, 20, 30]}
    shape = np.array([64, 80, 96])
    fixed_start = np.array([42, 79, 121])
    native_matrix, native_translation = trial._placed_level2_to_level0_initializer(
        level2, crop_shape=shape, fixed_start=fixed_start,
    )
    center2 = (np.array(level2["fixed_shape_zyx"]) - 1) / 2
    center0 = (shape - 1) / 2
    for point in (np.zeros(3), center0, shape - 1):
        expected = 4 * (matrix @ (point / 4 - center2) + center2 + level2["local_translation_zyx"] + np.array(level2["fixed_start_zyx"]))
        actual = native_matrix @ (point - center0) + center0 + native_translation + fixed_start
        np.testing.assert_allclose(actual, expected, atol=1e-5)
