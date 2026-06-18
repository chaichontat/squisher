from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as rough_legacy
from squisher_lightsheet import tile_phase as tile_phase_module
from squisher_lightsheet.tile_phase import (
    adapt_registration_from_reference,
    candidate_patch_slices,
    corresponding_moving_path,
    estimate_patch_shift_zyx_px,
    estimate_tile_shift_zyx_px,
    make_moving_tile_name,
    measure_patch_tile_shift,
    select_inlier_patch_measurements,
)


def test_estimate_tile_shift_zyx_px_recovers_synthetic_translation() -> None:
    z, y, x = np.mgrid[:32, :48, :48]
    fixed = np.exp(-(((z - 16) ** 2) / 12.0 + ((y - 24) ** 2 + (x - 24) ** 2) / 150.0)).astype(np.float32)
    moving = np.roll(fixed, shift=2, axis=0)
    moving = np.roll(moving, shift=-4, axis=1)
    moving = np.roll(moving, shift=5, axis=2)

    shift, details = estimate_tile_shift_zyx_px(fixed, moving, upsample_factor=10)

    np.testing.assert_allclose(shift, [-2, 4, -5], atol=0.25)
    assert details["corr_after"] > details["corr_before"]


def test_token_rewrite_resolves_corresponding_405_tile(tmp_path) -> None:
    reference = tmp_path / "230Tnc-CL-488514561638" / "230Tnc-CL-488514561638.000.ome.tif"
    moving = tmp_path / "230Tnc-CL-405" / "230Tnc-CL-405.000.ome.tif"
    moving.parent.mkdir(parents=True)
    moving.touch()

    assert corresponding_moving_path(reference, reference_token="488514561638", moving_token="405") == moving
    assert (
        make_moving_tile_name(
            "230Tnc-CL-488514561638.000.ome.tif",
            reference_token="488514561638",
            moving_token="405",
        )
        == "230Tnc-CL-405.000.ome.tif"
    )


def test_patch_shift_uses_gpu_phase_helper_for_synthetic_translation(monkeypatch) -> None:
    from skimage.registration import phase_cross_correlation

    z, y, x = np.mgrid[:32, :96, :96]
    fixed = np.exp(-(((z - 16) ** 2) / 14.0 + ((y - 48) ** 2 + (x - 48) ** 2) / 220.0)).astype(np.float32)
    moving = np.roll(fixed, shift=3, axis=0)
    moving = np.roll(moving, shift=-7, axis=1)
    moving = np.roll(moving, shift=9, axis=2)

    def fake_gpu_phase(fixed_norm: np.ndarray, moving_norm: np.ndarray) -> tuple[tuple[float, float, float], float]:
        shift, _error, _phase = phase_cross_correlation(fixed_norm, moving_norm, upsample_factor=1)
        return tuple(float(value) for value in shift), 1.0

    monkeypatch.setattr(tile_phase_module.stitch_legacy, "phase_correlation_shift_gpu", fake_gpu_phase)

    shift, details = estimate_patch_shift_zyx_px(fixed, moving)

    np.testing.assert_allclose(shift, [-3, 7, -9], atol=0.25)
    assert details["corr_after"] > details["corr_before"]


def test_patch_mode_composes_coarse_seed_and_residual(monkeypatch) -> None:
    reference = rough_legacy.TileRecord(
        tile="ref",
        side="L",
        path=Path("ref.tif"),
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([64, 128, 128]),
        axes="ZYX",
    )
    moving = rough_legacy.TileRecord(
        tile="moving",
        side="L",
        path=Path("moving.tif"),
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([64, 128, 128]),
        axes="ZYX",
    )

    monkeypatch.setattr(
        tile_phase_module.rough_legacy,
        "sampled_tile_volume",
        lambda *_args, **_kwargs: np.ones((16, 32, 32), dtype=np.float32),
    )
    monkeypatch.setattr(
        tile_phase_module,
        "estimate_tile_shift_zyx_px_gpu",
        lambda *_args, **_kwargs: (np.asarray([-1.0, 2.0, -3.0]), {"corr_before": 0.1, "corr_after": 0.5}),
    )
    monkeypatch.setattr(
        tile_phase_module,
        "candidate_patch_slices",
        lambda *_args, **_kwargs: [
            {
                "fixed_slices": (slice(16, 32), slice(48, 80), slice(48, 80)),
                "content_score": 1.0,
                "positive_fraction": 0.5,
            },
            {
                "fixed_slices": (slice(32, 48), slice(48, 80), slice(48, 80)),
                "content_score": 0.9,
                "positive_fraction": 0.4,
            },
            {
                "fixed_slices": (slice(40, 56), slice(48, 80), slice(48, 80)),
                "content_score": 0.8,
                "positive_fraction": 0.3,
            },
        ],
    )
    monkeypatch.setattr(
        tile_phase_module,
        "sampled_tile_volume_from_subifd",
        lambda *_args, **_kwargs: (np.ones((16, 32, 32), dtype=np.float32), np.asarray([4.0, 4.0, 4.0]), 2, 3),
    )
    monkeypatch.setattr(
        tile_phase_module,
        "read_tile_patch",
        lambda *_args, **_kwargs: np.ones((16, 32, 32), dtype=np.float32),
    )
    monkeypatch.setattr(
        tile_phase_module,
        "estimate_patch_shift_zyx_px",
        lambda *_args, **_kwargs: (
            np.asarray([1.0, -2.0, 3.0]),
            {"peak": 1.0, "corr_before": 0.2, "corr_after": 0.8},
        ),
    )

    shift, details = measure_patch_tile_shift(
        reference_tile=reference,
        moving_tile=moving,
        reference_channel=3,
        patch_shape_zyx=(16, 32, 32),
        coarse_level=2,
        upsample_factor=10,
        max_candidate_patches=2,
        min_inliers=2,
    )

    np.testing.assert_allclose(shift, [-3.0, 6.0, -9.0])
    assert details["n_inliers"] == 2
    assert details["n_measured"] == 2
    assert details["early_stop_after_patch"] == 1
    assert details["patches"][0]["moving_slices_zyx"] == [[20, 36], [40, 72], [60, 92]]
    assert details["patches"][2]["reason"] == "skipped_after_enough_inliers"


def test_inlier_selection_returns_cluster_median_and_rejects_outlier() -> None:
    shifts = np.asarray(
        [
            [10.0, -20.0, 4.0],
            [11.0, -18.0, 6.0],
            [9.5, -21.0, 5.0],
            [40.0, 50.0, -90.0],
        ]
    )

    inliers, median = select_inlier_patch_measurements(shifts, min_inliers=2)

    assert inliers.tolist() == [True, True, True, False]
    np.testing.assert_allclose(median, [10.0, -20.0, 5.0])


def test_inlier_selection_fails_with_too_few_inliers() -> None:
    shifts = np.asarray([[0.0, 0.0, 0.0], [20.0, 40.0, 40.0]])

    with pytest.raises(ValueError, match="require 2"):
        select_inlier_patch_measurements(shifts, min_inliers=2)


def test_candidate_patch_slices_filters_shifted_moving_out_of_bounds() -> None:
    scout = np.ones((4, 8, 8), dtype=np.float32)

    candidates = candidate_patch_slices(
        scout,
        tile_shape_zyx=np.asarray([64, 128, 128]),
        patch_shape_zyx=(32, 64, 64),
        scout_scale_zyx=np.asarray([16.0, 16.0, 16.0]),
        max_candidates=24,
        moving_shape_zyx=np.asarray([64, 128, 128]),
        shift_zyx_px=np.asarray([-32.0, -64.0, -64.0]),
    )

    assert candidates
    for candidate in candidates:
        moving_slices = tile_phase_module.shifted_slices_zyx(
            candidate["fixed_slices"],
            shift_zyx_px=np.asarray([-32.0, -64.0, -64.0]),
        )
        assert tile_phase_module.slices_within_shape(moving_slices, np.asarray([64, 128, 128]))


def test_adapt_registration_copies_affine_and_replaces_405_stage(tmp_path) -> None:
    reference_registration = tmp_path / "registration.track0.json"
    output_registration = tmp_path / "registration.405.json"
    affine = {
        "dims": ["x_in", "x_out"],
        "coords": {"x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]},
        "matrix": [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 4.0], [0.0, 0.0, 0.0, 1.0]],
    }
    reference_registration.write_text(
        json.dumps(
            {
                "input_dir": str(tmp_path / "230Tnc-CL-488514561638"),
                "tiles": [
                    {
                        "tile": "230Tnc-CL-488514561638.000.ome.tif",
                        "source_view": "L",
                        "stage_translation_um": {"z": 1.0, "y": 2.0, "x": 3.0},
                        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                        "registered_affine": affine,
                    }
                ],
            }
        )
    )
    moving_path = tmp_path / "230Tnc-CL-405" / "230Tnc-CL-405.000.ome.tif"
    position_payload = {
        "tiles": [
            {
                "tile": "230Tnc-CL-405.000.ome.tif",
                "path": str(moving_path),
                "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                "scale_um": {"z": 2.0, "y": 0.5, "x": 0.5},
            }
        ]
    }
    summary = {
        "output_position": str(tmp_path / "positions.json"),
        "summary_path": str(tmp_path / "tile_phase_alignment.json"),
        "measurements": [
            {
                "tile": "230Tnc-CL-405.000.ome.tif",
                "reference_tile": "230Tnc-CL-488514561638.000.ome.tif",
                "shift_px_zyx": [1.0, 2.0, 3.0],
                "shift_um_zyx": [2.0, 1.0, 1.5],
                "n_inliers": 2,
            }
        ],
    }

    adapt_registration_from_reference(
        reference_registration_input=reference_registration,
        output_registration=output_registration,
        adapted_position_payload=position_payload,
        reference_token="488514561638",
        moving_token="405",
        adapted_to_position=tmp_path / "positions.json",
        tile_phase_summary=summary,
    )

    adapted = json.loads(output_registration.read_text())
    assert adapted["input_dir"] == str(moving_path.parent)
    assert adapted["adapted_from"] == str(reference_registration.resolve())
    assert adapted["tiles"][0]["tile"] == "230Tnc-CL-405.000.ome.tif"
    assert adapted["tiles"][0]["path"] == str(moving_path)
    assert adapted["tiles"][0]["stage_translation_um"] == {"z": 10.0, "y": 20.0, "x": 30.0}
    assert adapted["tiles"][0]["stage_scale_um"] == {"z": 2.0, "y": 0.5, "x": 0.5}
    assert adapted["tiles"][0]["registered_affine"] == affine
