from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tifffile
import zarr

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


def test_mvs_seam_constraints_use_phase_refined_target_delta() -> None:
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
    np.testing.assert_allclose(constraints[0]["target_correction_delta_px"], [1.0, -2.0, 3.0])
    assert constraints[0]["target_correction_delta_source"] == "target_correction_delta_px_zyx"
    assert constraints[0]["phase_refined_bad_gradient"] is True
    assert constraints[0]["phase_refined_shift_native_px_zyx"] == [0.0, 5.0, -1.0]


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


def test_recover_anchor_shift_can_use_dropped_mvs_edges_at_lower_weight() -> None:
    payload = mvs_registration_payload()
    payload["metrics"]["pairwise_registration"]["edges"][1]["quality"] = {"data": [0.6]}
    payload["metrics"]["groupwise_resolution"]["metrics"]["used_edges"] = {"0": [[0, 1]]}

    recovered, diagnostics = mvs_seams.recover_anchor_shifts_from_mvs_seams(
        direct_anchor_shift_um_by_tile={"tile0.ome.tif": np.asarray([10.0, 20.0, 30.0])},
        mvs_registration=payload,
        spacing_um_zyx=np.asarray([2.0, 1.0, 0.5]),
        min_quality=0.25,
        used_edges_only=False,
        unused_edge_weight_scale=0.1,
    )

    np.testing.assert_allclose(recovered["tile1.ome.tif"], [14.0, 23.0, 32.0])
    np.testing.assert_allclose(recovered["tile2.ome.tif"], [16.0, 23.0, 32.0])
    assert diagnostics["mvs_pairwise_edge_count"] == 2
    assert diagnostics["recovered_tile_count"] == 2
    assert diagnostics["unused_edge_weight_scale"] == 0.1


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


def level0_registration_payload(tmp_path) -> dict:
    return {
        "input_dir": str(tmp_path),
        "spacing_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        "tiles": [
            {
                "tile": "tile0.ome.tif",
                "stage_translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                "registered_affine": {
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
            },
            {
                "tile": "tile1.ome.tif",
                "stage_translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                "registered_affine": {
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
            },
        ],
        "metrics": {
            "pairwise_registration": {
                "edges": [
                    {
                        "source": 0,
                        "target": 1,
                        "quality": {"data": [0.8]},
                        "attrs": {
                            "bbox": {"data": [[[2.0, 6.0, 6.0], [8.0, 34.0, 34.0]]]},
                            "transform": {
                                "data": [
                                    [
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                ]
                            },
                        },
                    }
                ]
            },
            "groupwise_resolution": {"metrics": {"used_edges": {"0": [[0, 1]]}}},
        },
    }


def write_level0_tiles(tmp_path) -> None:
    z, y, x = np.indices((10, 40, 40), dtype=np.float32)
    channel0 = z * 10.0 + y + x
    channel1 = channel0 + 100.0
    data = np.stack([channel0, channel1]).astype(np.float32)
    tifffile.imwrite(tmp_path / "tile0.ome.tif", data, metadata={"axes": "CZYX"})
    tifffile.imwrite(tmp_path / "tile1.ome.tif", data, metadata={"axes": "CZYX"})


def install_fake_level0_phase(monkeypatch, shift=(0.0, 2.0, 3.0)) -> None:
    monkeypatch.setattr(mvs_seams, "_sample_registered_center_patch", lambda **kwargs: np.ones((8, 8), dtype=np.float32))
    monkeypatch.setattr(
        mvs_seams,
        "_phase_refine_shift",
        lambda fixed, moving, **_kwargs: (tuple(float(value) for value in shift), 0.9, 0.01, 0.5),
    )


def test_select_level0_seam_span_candidate_uses_full_bbox_yx_and_requested_z() -> None:
    payload = level0_registration_payload(None)
    edge = payload["metrics"]["pairwise_registration"]["edges"][0]

    candidates = mvs_seams._select_level0_seam_span_candidate(
        edge=edge,
        spacing_um_zyx=np.asarray([1.0, 1.0, 1.0]),
        patch_shape_zyx=(10, 480, 480),
    )

    assert len(candidates) == 1
    assert candidates[0].center_px_zyx == (5.0, 20.0, 20.0)
    assert candidates[0].start_px_zyx == (0, 6, 6)
    assert candidates[0].shape_zyx == (10, 28, 28)


def test_refine_level0_reads_candidate_patches_directly_from_sources(tmp_path, monkeypatch) -> None:
    write_level0_tiles(tmp_path)
    registration = tmp_path / "registration.level2.json"
    registration.write_text(json.dumps(level0_registration_payload(tmp_path)))
    output = tmp_path / "registration.level0-refined.json"
    install_fake_level0_phase(monkeypatch)

    opened_levels = []
    original_open = mvs_seams._open_image_level_accessor

    def counting_open(path, *, level):
        opened_levels.append((path.name, int(level)))
        return original_open(path, level=level)

    monkeypatch.setattr(mvs_seams, "_open_image_level_accessor", counting_open)
    sample_payloads = []
    original_sample_candidate_patch = mvs_seams._sample_candidate_patch

    def recording_sample_candidate_patch(*args, **kwargs):
        sample_payloads.append(kwargs["registration_payload"])
        return original_sample_candidate_patch(*args, **kwargs)

    monkeypatch.setattr(mvs_seams, "_sample_candidate_patch", recording_sample_candidate_patch)

    refined, diagnostics = mvs_seams.refine_mvs_registration_level0(
        registration_input=registration,
        output_registration=output,
        patch_shape_zyx=(4, 12, 12),
        patches_per_edge=3,
        min_inliers=1,
    )

    assert output.exists()
    assert opened_levels.count(("tile0.ome.tif", 0)) == 1
    assert opened_levels.count(("tile1.ome.tif", 0)) == 1
    assert "cache_arrays" not in diagnostics["measured_edges"][0]
    assert diagnostics["settings"]["patch_read_strategy"] == "direct_registered_source_patch"
    assert diagnostics["settings"]["phase_highpass_sigma_zyx"] == [0.0, 10.0, 10.0]
    assert diagnostics["settings"]["phase_upsample_factor"] == 10
    summary = diagnostics["optimization_summary"]
    assert summary["constraint_count"] == 1
    assert summary["accepted_constraint_count"] == 1
    assert summary["rejected_constraint_count"] == 0
    assert summary["constraint_counts_by_axis"]["mvs_level0"]["accepted_count"] == 1
    np.testing.assert_allclose(summary["residual_stats"]["max_abs_px_zyx"], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(summary["correction_summary"]["px"]["max_zyx"], [0.0, 2.0, 3.0])
    assert summary["connectivity"]["connected_tile_count"] == 2
    assert diagnostics["contact_sheet"]["render"] == (
        "seed registration coordinates with local phase-correlation shift applied directly; source green, target red"
    )
    assert diagnostics["contact_sheet"]["panels"][0]["render_shift_px_zyx"] == [0.0, 2.0, 3.0]
    assert diagnostics["contact_sheet"]["sample_registration"] == str(registration.resolve())
    assert sample_payloads[-1] is not refined
    np.testing.assert_allclose(
        np.asarray(sample_payloads[-1]["tiles"][1]["registered_affine"]["matrix"], dtype=float)[:3, 3],
        [0.0, 0.0, 0.0],
    )
    assert (tmp_path / "registration.level0-refined.level0-contact-sheet" / "level0_refinement_local_phase_contact_sheet.png").exists()
    np.testing.assert_allclose(
        np.asarray(refined["tiles"][1]["registered_affine"]["matrix"], dtype=float)[:3, 3],
        [0.0, 2.0, 3.0],
    )
    assert refined["tiles"][1]["stage_translation_um"] == {"z": 0.0, "y": 0.0, "x": 0.0}


def test_measure_level0_candidate_ignores_gradient_score(monkeypatch) -> None:
    monkeypatch.setattr(
        mvs_seams,
        "_phase_refine_shift",
        lambda fixed, moving, **_kwargs: ((0.0, 2.0, 3.0), 0.9, 0.01, 0.5),
    )
    candidate = mvs_seams.Level0PatchCandidate(
        patch_index=0,
        center_px_zyx=(2.0, 6.0, 6.0),
        start_px_zyx=(0, 0, 0),
        shape_zyx=(4, 12, 12),
        scout_score=1.0,
    )

    measurement = mvs_seams._measure_level0_candidate(
        fixed_patch=np.ones((4, 12, 12), dtype=np.float32),
        moving_patch=np.ones((4, 12, 12), dtype=np.float32),
        candidate=candidate,
        max_phase_shift_zyx=(3.0, 64.0, 64.0),
        phase_highpass_sigma_zyx=(0.0, 10.0, 10.0),
        phase_upsample_factor=10,
    )

    assert measurement["accepted"] is True
    assert measurement["reject_reason"] is None
    assert measurement["gradient_component_ncc_before"] is None
    assert measurement["gradient_component_ncc_after"] is None
    assert measurement["phase_crop_start_offset_zyx"] == [0, 0, 0]
    assert measurement["phase_crop_shape_zyx"] == [4, 12, 12]


def test_measure_level0_candidate_phase_correlates_seed_shared_support_crop(monkeypatch) -> None:
    calls = []

    def fake_phase(fixed, moving, **_kwargs):
        calls.append((fixed.shape, moving.shape))
        assert np.count_nonzero(fixed[:, :4, :]) > 0
        assert np.count_nonzero(moving[:, 10:, :]) > 0
        return (0.0, 2.0, 0.0), 0.9, float("nan"), 0.5

    monkeypatch.setattr(mvs_seams, "_phase_refine_shift", fake_phase)
    candidate = mvs_seams.Level0PatchCandidate(
        patch_index=0,
        center_px_zyx=(2.0, 10.0, 10.0),
        start_px_zyx=(5, 100, 200),
        shape_zyx=(4, 20, 20),
        scout_score=1.0,
    )
    fixed = np.zeros((4, 20, 20), dtype=np.float32)
    moving = np.zeros_like(fixed)
    fixed[:, :16, :] = 1.0
    moving[:, 10:, :] = 1.0

    measurement = mvs_seams._measure_level0_candidate(
        fixed_patch=fixed,
        moving_patch=moving,
        candidate=candidate,
        max_phase_shift_zyx=(1.0, 4.0, 4.0),
        phase_highpass_sigma_zyx=(0.0, 10.0, 10.0),
        phase_upsample_factor=10,
    )

    assert calls == [((4, 14, 20), (4, 14, 20))]
    assert measurement["accepted"] is True
    assert measurement["phase_crop_start_offset_zyx"] == [0, 6, 0]
    assert measurement["phase_crop_start_px_zyx"] == [5, 106, 200]
    assert measurement["phase_crop_shape_zyx"] == [4, 14, 20]
    np.testing.assert_allclose(measurement["seed_valid_overlap_fraction"], 6 / 14)
    assert measurement["shifted_valid_overlap_fraction"] > 0.0


def test_measure_level0_candidate_rejects_shift_at_phase_search_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        mvs_seams,
        "_phase_refine_shift",
        lambda fixed, moving, **_kwargs: ((0.0, -64.0, 32.0), 0.9, float("nan"), 0.5),
    )
    candidate = mvs_seams.Level0PatchCandidate(
        patch_index=0,
        center_px_zyx=(2.0, 80.0, 80.0),
        start_px_zyx=(0, 0, 0),
        shape_zyx=(4, 160, 160),
        scout_score=1.0,
    )

    measurement = mvs_seams._measure_level0_candidate(
        fixed_patch=np.ones((4, 160, 160), dtype=np.float32),
        moving_patch=np.ones((4, 160, 160), dtype=np.float32),
        candidate=candidate,
        max_phase_shift_zyx=(3.0, 64.0, 64.0),
        phase_highpass_sigma_zyx=(0.0, 10.0, 10.0),
        phase_upsample_factor=10,
    )

    assert measurement["accepted"] is False
    assert measurement["reject_reason"] == "phase_shift_out_of_bounds"
    assert measurement["shifted_valid_overlap_fraction"] > 0.05


def test_measure_level0_candidate_rejects_low_shifted_valid_overlap(monkeypatch) -> None:
    monkeypatch.setattr(
        mvs_seams,
        "_phase_refine_shift",
        lambda fixed, moving, **_kwargs: ((0.0, 39.0, 0.0), 0.9, float("nan"), 0.5),
    )
    candidate = mvs_seams.Level0PatchCandidate(
        patch_index=0,
        center_px_zyx=(2.0, 20.0, 20.0),
        start_px_zyx=(0, 0, 0),
        shape_zyx=(4, 40, 40),
        scout_score=1.0,
    )

    measurement = mvs_seams._measure_level0_candidate(
        fixed_patch=np.ones((4, 40, 40), dtype=np.float32),
        moving_patch=np.ones((4, 40, 40), dtype=np.float32),
        candidate=candidate,
        max_phase_shift_zyx=(3.0, 64.0, 64.0),
        phase_highpass_sigma_zyx=(0.0, 10.0, 10.0),
        phase_upsample_factor=10,
    )

    assert measurement["accepted"] is False
    assert measurement["reject_reason"] == "low_shifted_valid_overlap"
    assert measurement["shifted_valid_overlap_fraction"] < 0.05


def test_retry_face_orientation_uses_yx_bbox_extents() -> None:
    spacing = np.asarray([1.0, 2.0, 1.0])
    normal_axis, along_axis, plane, orientation, extents = mvs_seams._seam_face_orientation(
        {"attrs": {"bbox": {"data": [[[0.0, 0.0, 0.0], [20.0, 20.0, 200.0]]]}}},
        spacing,
    )

    assert normal_axis == 1
    assert along_axis == 2
    assert plane == "xz"
    assert orientation == "normal_y"
    assert extents == [20.0, 10.0, 200.0]

    normal_axis, along_axis, plane, orientation, _extents = mvs_seams._seam_face_orientation(
        {"attrs": {"bbox": {"data": [[[0.0, 0.0, 0.0], [20.0, 200.0, 20.0]]]}}},
        spacing,
    )

    assert normal_axis == 2
    assert along_axis == 1
    assert plane == "yz"
    assert orientation == "normal_x"


def test_cached_tiff_reader_uses_pyramid_level_once(tmp_path, monkeypatch) -> None:
    path = tmp_path / "pyramid.ome.tif"
    level0 = np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16)
    level1 = level0[:, ::2, ::2] + 1000
    with tifffile.TiffWriter(path, ome=True) as tif:
        tif.write(level0, metadata={"axes": "ZYX"}, subifds=1)
        tif.write(level1, subfiletype=1)
    with tifffile.TiffFile(path) as tif:
        assert len(tif.series[0].levels) == 2

    calls = []
    original_imread = tifffile.imread

    def recording_imread(*args, **kwargs):
        calls.append(kwargs.get("level"))
        return original_imread(*args, **kwargs)

    monkeypatch.setattr(tifffile, "imread", recording_imread)
    cache = mvs_seams._ImageLevelReaderCache()
    try:
        axes, source_level, shape = cache.metadata(path, level=1)
        first = cache.read_crop(
            path,
            level=source_level,
            axes=axes,
            channel=0,
            slices_zyx=(slice(1, 3), slice(2, 5), slice(3, 7)),
        )
        second = cache.read_crop(
            path,
            level=source_level,
            axes=axes,
            channel=0,
            slices_zyx=(slice(1, 2), slice(0, 2), slice(0, 2)),
        )
    finally:
        cache.close()

    assert axes == "ZYX"
    assert source_level == 1
    assert shape == level1.shape
    np.testing.assert_array_equal(first, level1[1:3, 2:5, 3:7].astype(np.float32))
    np.testing.assert_array_equal(second, level1[1:2, 0:2, 0:2].astype(np.float32))
    assert calls == [1]


def test_cached_zarr_reader_uses_pyramid_level_once(tmp_path, monkeypatch) -> None:
    path = tmp_path / "pyramid.ome.zarr"
    level0 = np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16)
    level1 = level0[:, ::2, ::2] + 1000
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    for name, data in (("0", level0), ("1", level1)):
        array = root.create_array(
            name,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(2, 4, 4),
        )
        array[:] = data
        array.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    root.attrs["multiscales"] = [
        {
            "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
            "datasets": [{"path": "0"}, {"path": "1"}],
        }
    ]

    cache = mvs_seams._ImageLevelReaderCache()
    try:
        axes, source_level, shape = cache.metadata(path, level=1)
        first = cache.read_crop(
            path,
            level=source_level,
            axes=axes,
            channel=0,
            slices_zyx=(slice(1, 3), slice(2, 5), slice(3, 7)),
        )
        second = cache.read_crop(
            path,
            level=source_level,
            axes=axes,
            channel=0,
            slices_zyx=(slice(1, 2), slice(0, 2), slice(0, 2)),
        )
    finally:
        cache.close()

    assert axes == "ZYX"
    assert source_level == 1
    assert shape == level1.shape
    np.testing.assert_array_equal(first, level1[1:3, 2:5, 3:7].astype(np.float32))
    np.testing.assert_array_equal(second, level1[1:2, 0:2, 0:2].astype(np.float32))


def test_cached_zarr_reader_uses_multiscales_dataset_paths(tmp_path) -> None:
    path = tmp_path / "pyramid.ome.zarr"
    level0 = np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16)
    level1 = level0[:, ::2, ::2] + 1000
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    for name, data in (("scale0", level0), ("scale1", level1)):
        array = root.create_array(name, shape=data.shape, dtype=data.dtype, chunks=(2, 4, 4))
        array[:] = data
        array.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    root.attrs["multiscales"] = [
        {
            "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
            "datasets": [{"path": "scale0"}, {"path": "scale1"}],
        }
    ]

    cache = mvs_seams._ImageLevelReaderCache()
    try:
        axes, source_level, shape = cache.metadata(path, level=1)
        crop = cache.read_crop(
            path,
            level=source_level,
            axes=axes,
            channel=0,
            slices_zyx=(slice(0, 1), slice(0, 2), slice(0, 2)),
        )
    finally:
        cache.close()

    assert source_level == 1
    assert shape == level1.shape
    np.testing.assert_array_equal(crop, level1[0:1, 0:2, 0:2].astype(np.float32))


def test_cached_zarr_reader_rejects_missing_pyramid_level(tmp_path) -> None:
    path = tmp_path / "pyramid.ome.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    array = root.create_array("0", shape=(4, 16, 16), dtype=np.uint16, chunks=(2, 4, 4))
    array.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    root.attrs["multiscales"] = [
        {
            "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
            "datasets": [{"path": "0"}],
        }
    ]

    with pytest.raises(ValueError, match="requested level 1"):
        mvs_seams._ImageLevelReaderCache().metadata(path, level=1)


def test_image_level_reader_cache_first_open_is_thread_safe(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tile.ome.zarr"
    open_count = 0

    def fake_open(*_args, **_kwargs):
        nonlocal open_count
        open_count += 1
        return mvs_seams._ImageLevelAccessor(
            axes="ZYX",
            level=1,
            shape=(4, 8, 8),
            array=np.zeros((4, 8, 8), dtype=np.uint16),
        )

    monkeypatch.setattr(mvs_seams, "_open_image_level_accessor", fake_open)
    cache = mvs_seams._ImageLevelReaderCache()
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(
                executor.map(
                    lambda _index: cache.read_crop(
                        path,
                        level=1,
                        axes="ZYX",
                        channel=0,
                        slices_zyx=(slice(0, 1), slice(0, 1), slice(0, 1)),
                    ),
                    range(8),
                )
            )
    finally:
        cache.close()

    assert open_count == 1


def test_sample_candidate_patch_scales_level_shape_for_pyramid_level(monkeypatch) -> None:
    captured = {}

    def fake_sample_registered_patch(**kwargs):
        captured.update(kwargs)
        return np.zeros(kwargs["patch_shape_zyx"], dtype=np.float32)

    monkeypatch.setattr(mvs_seams, "_sample_registered_patch", fake_sample_registered_patch)
    candidate = mvs_seams.Level0PatchCandidate(
        patch_index=0,
        center_px_zyx=(5.0, 240.0, 240.0),
        start_px_zyx=(0, 0, 0),
        shape_zyx=(10, 480, 480),
        scout_score=1.0,
    )

    patch = mvs_seams._sample_candidate_patch(
        registration_payload={},
        tile_record={},
        candidate=candidate,
        margin_zyx=(3, 32, 32),
        spacing_um_zyx=np.ones(3),
        channel=0,
        level=1,
        native_shift_scale_zyx=np.asarray([1.0, 2.0, 2.0]),
    )

    assert captured["level"] == 1
    assert captured["patch_shape_zyx"] == (16, 304, 304)
    assert patch.shape == (16, 304, 304)


def test_native_shift_scale_uses_actual_pyramid_shapes(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.ome.zarr"
    target_path = tmp_path / "target.ome.zarr"
    source_path.mkdir()
    target_path.mkdir()
    payload = {"input_dir": str(tmp_path)}
    tiles = [{"tile": source_path.name}, {"tile": target_path.name}]

    def fake_metadata(_path, *, level, image_reader_cache):
        del image_reader_cache
        if int(level) == 0:
            return "ZYX", 0, (4, 16, 16)
        return "ZYX", 1, (2, 8, 8)

    monkeypatch.setattr(mvs_seams, "_cached_image_level_metadata", fake_metadata)

    scale = mvs_seams._native_shift_scale_zyx_for_edge_level(
        registration_payload=payload,
        tiles=tiles,
        source=0,
        target=1,
        level=1,
    )

    np.testing.assert_allclose(scale, [2.0, 2.0, 2.0])


def test_retry_face_candidates_vary_z_and_along_seam_axis(monkeypatch) -> None:
    def fake_plane(**kwargs):
        values = np.zeros((len(kwargs["axis0_values_px"]), len(kwargs["axis1_values_px"])), dtype=np.float32)
        for z_index, along_index in ((10, 10), (25, 40), (40, 80), (55, 110)):
            values[z_index, along_index] = 10.0
        return values

    monkeypatch.setattr(mvs_seams, "_sample_registered_plane_grid", fake_plane)
    candidates, info = mvs_seams._select_level0_retry_face_candidates(
        registration_payload={"input_dir": "."},
        edge={
            "source": 0,
            "target": 1,
            "attrs": {"bbox": {"data": [[[0.0, 0.0, 0.0], [140.0, 20.0, 200.0]]]}},
        },
        tiles=[{"tile": "tile0.ome.zarr"}, {"tile": "tile1.ome.zarr"}],
        spacing_um_zyx=np.ones(3),
        channel=0,
        patch_shape_zyx=(10, 20, 20),
        patches_per_edge=4,
    )

    assert info["mode"] == "seam_face_content"
    assert info["orientation"] == "normal_y"
    assert info["plane"] == "xz"
    assert len(candidates) == 4
    assert len({candidate.center_px_zyx[0] for candidate in candidates}) > 1
    assert len({candidate.center_px_zyx[2] for candidate in candidates}) > 1
    assert {candidate.center_px_zyx[1] for candidate in candidates} == {10.0}


def test_refine_level0_retries_low_inlier_edges_with_more_patches(tmp_path, monkeypatch) -> None:
    registration = tmp_path / "registration.level2.json"
    registration.write_text(json.dumps(level0_registration_payload(tmp_path)))
    output = tmp_path / "registration.level0-refined.json"
    select_counts = []

    def fake_select_candidates(*, patches_per_edge, **_kwargs):
        select_counts.append(patches_per_edge)
        return [
            mvs_seams.Level0PatchCandidate(
                patch_index=index,
                center_px_zyx=(2.0, float(index), 0.0),
                start_px_zyx=(0, index, 0),
                shape_zyx=(4, 12, 12),
                scout_score=float(index),
            )
            for index in range(patches_per_edge)
        ]

    def fake_retry_candidates(*, patches_per_edge, **_kwargs):
        return (
            [
                mvs_seams.Level0PatchCandidate(
                    patch_index=index,
                    center_px_zyx=(float(index), float(index), 0.0),
                    start_px_zyx=(index, index, 0),
                    shape_zyx=(4, 12, 12),
                    scout_score=float(index),
                )
                for index in range(patches_per_edge)
            ],
            {"mode": "seam_face_content", "orientation": "normal_y", "plane": "xz"},
        )

    def fake_measure(*, candidate, **_kwargs):
        return {
            "patch_index": int(candidate.patch_index),
            "center_px_zyx": [float(value) for value in candidate.center_px_zyx],
            "start_px_zyx": [int(value) for value in candidate.start_px_zyx],
            "shape_zyx": [int(value) for value in candidate.shape_zyx],
            "scout_score": float(candidate.scout_score),
            "shift_px_zyx": [0.0, 0.0, 0.0],
            "accepted": True,
            "reject_reason": None,
        }

    monkeypatch.setattr(mvs_seams, "_select_level0_candidates", fake_select_candidates)
    monkeypatch.setattr(mvs_seams, "_select_level0_retry_face_candidates", fake_retry_candidates)
    monkeypatch.setattr(
        mvs_seams,
        "_sample_candidate_patch",
        lambda **_kwargs: np.ones((4, 12, 12), dtype=np.float32),
    )
    monkeypatch.setattr(mvs_seams, "_measure_level0_candidate", fake_measure)

    _refined, diagnostics = mvs_seams.refine_mvs_registration_level0(
        registration_input=registration,
        output_registration=output,
        patch_shape_zyx=(4, 12, 12),
        patches_per_edge=2,
        retry_patches_per_edge=5,
        min_inliers=3,
        render_contact_sheet=False,
    )

    edge = diagnostics["measured_edges"][0]
    assert select_counts == [2]
    assert diagnostics["settings"]["retry_patches_per_edge"] == 5
    assert edge["status"] == "accepted"
    assert edge["accepted_patch_count"] == 3
    assert edge["inlier_patch_count"] == 3
    assert edge["retried_with_patch_count"] == 3
    assert edge["retry_candidate_info"]["mode"] == "seam_face_content"
    assert edge["retry_candidate_info"]["measured_candidate_count"] == 3
    assert edge["retry_candidate_info"]["early_stopped"] is True
    assert len(edge["patches"]) == 3


def test_refine_level0_uses_low_weight_level2_fallback_for_rejected_edges(tmp_path, monkeypatch) -> None:
    registration_payload = level0_registration_payload(tmp_path)
    registration_payload["metrics"]["pairwise_registration"]["edges"][0]["attrs"]["transform"]["data"][0][2][3] = -5.0
    registration = tmp_path / "registration.level2.json"
    registration.write_text(json.dumps(registration_payload))
    output = tmp_path / "registration.level0-refined.json"

    monkeypatch.setattr(
        mvs_seams,
        "_select_level0_candidates",
        lambda **_kwargs: [
            mvs_seams.Level0PatchCandidate(
                patch_index=0,
                center_px_zyx=(2.0, 6.0, 6.0),
                start_px_zyx=(0, 0, 0),
                shape_zyx=(4, 12, 12),
                scout_score=1.0,
            )
        ],
    )
    monkeypatch.setattr(mvs_seams, "_sample_candidate_patch", lambda **_kwargs: np.ones((4, 12, 12), dtype=np.float32))
    monkeypatch.setattr(
        mvs_seams,
        "_measure_level0_candidate",
        lambda **_kwargs: {
            "patch_index": 0,
            "center_px_zyx": [2.0, 6.0, 6.0],
            "start_px_zyx": [0, 0, 0],
            "shape_zyx": [4, 12, 12],
            "scout_score": 1.0,
            "shift_px_zyx": [0.0, 0.0, 0.0],
            "accepted": False,
            "reject_reason": "low_content",
        },
    )

    refined, diagnostics = mvs_seams.refine_mvs_registration_level0(
        registration_input=registration,
        output_registration=output,
        patch_shape_zyx=(4, 12, 12),
        patches_per_edge=1,
        retry_patches_per_edge=1,
        min_inliers=1,
        fallback_level2_weight_scale=0.1,
        render_contact_sheet=False,
    )

    fallback_constraints = [
        constraint
        for constraint in diagnostics["constraints"]
        if constraint["axis"] == "mvs_level2_fallback"
    ]
    assert diagnostics["accepted_edge_count"] == 0
    assert diagnostics["fallback_level2_constraint_count"] == 1
    assert len(fallback_constraints) == 1
    assert fallback_constraints[0]["source_label"] == mvs_seams.LEVEL2_FALLBACK_SOURCE_LABEL
    assert fallback_constraints[0]["accepted"] is True
    np.testing.assert_allclose(fallback_constraints[0]["shift_px_zyx"], [0.0, 0.0, 5.0])
    np.testing.assert_allclose(fallback_constraints[0]["weight"], 0.08)
    np.testing.assert_allclose(
        np.asarray(refined["tiles"][1]["registered_affine"]["matrix"], dtype=float)[:3, 3],
        [0.0, 0.0, 5.0],
    )


def test_refine_level0_can_rescue_rejected_edges_at_fallback_refinement_level(tmp_path, monkeypatch) -> None:
    registration_payload = level0_registration_payload(tmp_path)
    registration_payload["metrics"]["pairwise_registration"]["edges"][0]["attrs"]["transform"]["data"][0][2][3] = -5.0
    registration = tmp_path / "registration.level2.json"
    registration.write_text(json.dumps(registration_payload))
    output = tmp_path / "registration.level0-refined.json"
    candidate = mvs_seams.Level0PatchCandidate(
        patch_index=0,
        center_px_zyx=(2.0, 6.0, 6.0),
        start_px_zyx=(0, 0, 0),
        shape_zyx=(4, 12, 12),
        scout_score=1.0,
    )

    monkeypatch.setattr(mvs_seams, "_select_level0_candidates", lambda **_kwargs: [candidate])
    monkeypatch.setattr(
        mvs_seams,
        "_select_level0_retry_face_candidates",
        lambda **_kwargs: ([candidate], {"mode": "test_level1_fallback"}),
    )
    monkeypatch.setattr(
        mvs_seams,
        "_native_shift_scale_zyx_for_edge_level",
        lambda **kwargs: np.asarray([1.0, 1.0, 1.0])
        if kwargs["level"] == 0
        else np.asarray([1.0, 2.0, 2.0]),
    )
    sample_calls = []

    def fake_sample_candidate_patch(**kwargs):
        sample_calls.append(kwargs)
        return np.full((4, 12, 12), float(kwargs["level"]), dtype=np.float32)

    monkeypatch.setattr(mvs_seams, "_sample_candidate_patch", fake_sample_candidate_patch)

    def fake_measure(*, fixed_patch, candidate, **_kwargs):
        level = int(fixed_patch[0, 0, 0])
        accepted = level == 1
        return {
            "patch_index": int(candidate.patch_index),
            "center_px_zyx": [float(value) for value in candidate.center_px_zyx],
            "start_px_zyx": [int(value) for value in candidate.start_px_zyx],
            "shape_zyx": [int(value) for value in candidate.shape_zyx],
            "scout_score": float(candidate.scout_score),
            "shift_px_zyx": [0.0, 2.0, 3.0] if accepted else [0.0, 0.0, 0.0],
            "phase_input_origin_px_zyx": [0, 0, 0],
            "phase_crop_start_offset_zyx": [0, 0, 0],
            "phase_crop_shape_zyx": [4, 12, 12],
            "accepted": accepted,
            "reject_reason": None if accepted else "low_content",
        }

    monkeypatch.setattr(mvs_seams, "_measure_level0_candidate", fake_measure)

    refined, diagnostics = mvs_seams.refine_mvs_registration_level0(
        registration_input=registration,
        output_registration=output,
        patch_shape_zyx=(4, 12, 12),
        patches_per_edge=1,
        retry_patches_per_edge=1,
        min_inliers=1,
        fallback_refinement_levels=(1,),
        fallback_level2_weight_scale=0.1,
        render_contact_sheet=True,
        contact_sheet_output_dir=tmp_path / "contact-sheet",
    )

    edge = diagnostics["measured_edges"][0]
    level1_constraints = [
        constraint for constraint in diagnostics["constraints"] if constraint["axis"] == "mvs_level1"
    ]
    fallback_constraints = [
        constraint
        for constraint in diagnostics["constraints"]
        if constraint["axis"] == "mvs_level2_fallback"
    ]
    assert edge["status"] == "accepted"
    assert edge["refinement_level"] == 1
    assert edge["edge_shift_level_px_zyx"] == [0.0, 2.0, 3.0]
    assert edge["edge_shift_px_zyx"] == [0.0, 4.0, 6.0]
    assert edge["fallback_refinement_attempts"][0]["status"] == "accepted"
    assert diagnostics["fallback_refinement_rescued_edge_count"] == 1
    assert diagnostics["contact_sheet"]["panels"][0]["refinement_level"] == 1
    assert sample_calls[-2]["level"] == 1
    assert sample_calls[-1]["level"] == 1
    assert diagnostics["accepted_edge_count_by_refinement_level"] == {"1": 1}
    assert diagnostics["fallback_level2_constraint_count"] == 0
    assert len(level1_constraints) == 1
    assert len(fallback_constraints) == 0
    assert level1_constraints[0]["source_label"] == "mvs_level1_fallback_refinement"
    np.testing.assert_allclose(level1_constraints[0]["shift_px_zyx"], [0.0, 4.0, 6.0])
    np.testing.assert_allclose(
        np.asarray(refined["tiles"][1]["registered_affine"]["matrix"], dtype=float)[:3, 3],
        [0.0, 4.0, 6.0],
    )


def test_refine_level0_uses_dropped_level2_edges_as_low_weight_fallback(tmp_path, monkeypatch) -> None:
    registration_payload = level0_registration_payload(tmp_path)
    tile2 = copy.deepcopy(registration_payload["tiles"][1])
    tile2["tile"] = "tile2.ome.tif"
    registration_payload["tiles"].append(tile2)
    registration_payload["metrics"]["pairwise_registration"]["edges"].append(
        {
            "source": 1,
            "target": 2,
            "quality": {"data": [0.8]},
            "attrs": {
                "bbox": {"data": [[[2.0, 6.0, 6.0], [8.0, 34.0, 34.0]]]},
                "transform": {
                    "data": [
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, -7.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ]
                },
            },
        }
    )
    registration_payload["metrics"]["groupwise_resolution"]["metrics"]["used_edges"] = {"0": [[0, 1]]}
    registration = tmp_path / "registration.level2.json"
    registration.write_text(json.dumps(registration_payload))
    output = tmp_path / "registration.level0-refined.json"

    monkeypatch.setattr(
        mvs_seams,
        "_select_level0_candidates",
        lambda **_kwargs: [
            mvs_seams.Level0PatchCandidate(
                patch_index=0,
                center_px_zyx=(2.0, 6.0, 6.0),
                start_px_zyx=(0, 0, 0),
                shape_zyx=(4, 12, 12),
                scout_score=1.0,
            )
        ],
    )
    monkeypatch.setattr(mvs_seams, "_sample_candidate_patch", lambda **_kwargs: np.ones((4, 12, 12), dtype=np.float32))
    monkeypatch.setattr(
        mvs_seams,
        "_measure_level0_candidate",
        lambda **_kwargs: {
            "patch_index": 0,
            "center_px_zyx": [2.0, 6.0, 6.0],
            "start_px_zyx": [0, 0, 0],
            "shape_zyx": [4, 12, 12],
            "scout_score": 1.0,
            "shift_px_zyx": [0.0, 0.0, 0.0],
            "accepted": True,
            "reject_reason": None,
        },
    )

    refined, diagnostics = mvs_seams.refine_mvs_registration_level0(
        registration_input=registration,
        output_registration=output,
        patch_shape_zyx=(4, 12, 12),
        patches_per_edge=1,
        retry_patches_per_edge=1,
        min_inliers=1,
        fallback_level2_weight_scale=0.1,
        render_contact_sheet=False,
    )

    fallback_constraints = [
        constraint
        for constraint in diagnostics["constraints"]
        if constraint["axis"] == "mvs_level2_fallback"
    ]
    assert diagnostics["accepted_edge_count"] == 1
    assert diagnostics["fallback_level2_constraint_count"] == 1
    assert fallback_constraints[0]["pair"] == [1, 2]
    np.testing.assert_allclose(fallback_constraints[0]["shift_px_zyx"], [0.0, 0.0, 7.0])
    np.testing.assert_allclose(
        np.asarray(refined["tiles"][2]["registered_affine"]["matrix"], dtype=float)[:3, 3],
        [0.0, 0.0, 7.0],
    )


def test_refine_level0_default_does_not_create_seam_cache(tmp_path, monkeypatch) -> None:
    write_level0_tiles(tmp_path)
    registration = tmp_path / "registration.level2.json"
    registration.write_text(json.dumps(level0_registration_payload(tmp_path)))
    output = tmp_path / "registration.level0-refined.json"
    install_fake_level0_phase(monkeypatch, shift=(0.0, 0.0, 0.0))

    mvs_seams.refine_mvs_registration_level0(
        registration_input=registration,
        output_registration=output,
        patch_shape_zyx=(4, 12, 12),
        patches_per_edge=1,
        min_inliers=1,
    )

    assert not (tmp_path / ".registration.level0-refined.seam-cache.zarr").exists()
    assert (tmp_path / "registration.level0-refined.level0-contact-sheet" / "level0_refinement_local_phase_contact_sheet.png").exists()


def test_refine_level0_direct_source_reads_do_not_create_seam_cache(tmp_path, monkeypatch) -> None:
    write_level0_tiles(tmp_path)
    registration = tmp_path / "registration.level2.json"
    registration.write_text(json.dumps(level0_registration_payload(tmp_path)))
    output = tmp_path / "registration.level0-refined.json"
    install_fake_level0_phase(monkeypatch, shift=(0.0, 0.0, 0.0))

    open_count = 0
    original_open = mvs_seams._open_image_level_accessor

    def counting_open(path, *, level):
        nonlocal open_count
        open_count += 1
        return original_open(path, level=level)

    monkeypatch.setattr(mvs_seams, "_open_image_level_accessor", counting_open)
    kwargs = {
        "registration_input": registration,
        "output_registration": output,
        "patch_shape_zyx": (4, 12, 12),
        "patches_per_edge": 1,
        "min_inliers": 1,
        "render_contact_sheet": False,
    }

    mvs_seams.refine_mvs_registration_level0(**kwargs)
    assert open_count == 2
    mvs_seams.refine_mvs_registration_level0(**kwargs)
    assert open_count == 4
    assert not (tmp_path / ".registration.level0-refined.seam-cache.zarr").exists()


def test_level0_solver_allows_and_clamps_bounded_z_correction() -> None:
    constraint = mvs_seams._edge_constraint_from_measurement(
        edge={"source": 0, "target": 1},
        edge_shift_px_zyx=np.asarray([9.0, 0.0, 0.0]),
        weight=1.0,
    )

    corrections, _constraints, _anchor, _connectivity = mvs_seams._solve_level0_edge_corrections(
        n_tiles=2,
        constraints=[constraint],
        spacing_um_zyx=np.ones(3),
        max_correction_zyx=(4.0, 64.0, 64.0),
        max_final_residual_zyx=(20.0, 8.0, 8.0),
    )

    np.testing.assert_allclose(corrections[1], [4.0, 0.0, 0.0])


def test_level0_solver_allows_small_disconnected_island_budget() -> None:
    constraints = [
        mvs_seams._edge_constraint_from_measurement(
            edge={"source": 0, "target": 1},
            edge_shift_px_zyx=np.asarray([0.0, 1.0, 0.0]),
            weight=1.0,
        ),
        mvs_seams._edge_constraint_from_measurement(
            edge={"source": 2, "target": 3},
            edge_shift_px_zyx=np.asarray([0.0, 1.0, 0.0]),
            weight=1.0,
        ),
    ]

    with pytest.raises(RuntimeError, match="disconnected island"):
        mvs_seams._solve_level0_edge_corrections(
            n_tiles=4,
            constraints=constraints,
            spacing_um_zyx=np.ones(3),
            max_correction_zyx=(4.0, 64.0, 64.0),
            max_final_residual_zyx=(20.0, 8.0, 8.0),
            max_disconnected_island_size=1,
        )

    _corrections, _constraints, _anchor, connectivity = mvs_seams._solve_level0_edge_corrections(
        n_tiles=4,
        constraints=constraints,
        spacing_um_zyx=np.ones(3),
        max_correction_zyx=(4.0, 64.0, 64.0),
        max_final_residual_zyx=(20.0, 8.0, 8.0),
    )
    assert connectivity["disconnected_tile_count"] == 2
    assert connectivity["largest_disconnected_island_size"] == 2
