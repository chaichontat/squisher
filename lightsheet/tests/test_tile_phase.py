from __future__ import annotations

import json
from pathlib import Path
import inspect

import numpy as np
import pytest

from squisher_lightsheet import cli as cli_module
from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as rough_legacy
from squisher_lightsheet import tile_phase as tile_phase_module
from squisher_lightsheet.tile_phase import (
    adapt_registration_from_reference,
    candidate_patch_slices,
    corresponding_moving_path,
    estimate_patch_shift_zyx_px,
    estimate_tile_shift_zyx_px,
    align_tiles_to_reference,
    make_moving_tile_name,
    measure_patch_tile_shift,
    patch_quality_rejection_reasons,
    phase_cache_keys_match,
    rescore_cached_patch_attempt,
    select_inlier_patch_measurements,
    tile_phase_cache_key,
)


def _write_test_ome_zarr(path: Path, *, axes: str, levels: list[np.ndarray]) -> None:
    import zarr

    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    datasets = []
    axis_names = [{"name": axis.lower(), "type": "channel" if axis == "C" else "space"} for axis in axes]
    for index, data in enumerate(levels):
        chunks = tuple(min(2, int(size)) for size in data.shape)
        array = group.create_array(
            str(index),
            data=data,
            chunks=chunks,
            dimension_names=tuple(axis.lower() for axis in axes),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = [axis.lower() for axis in axes]
        datasets.append({"path": str(index), "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(axes)}]})
    group.attrs["multiscales"] = [{"version": "0.4", "axes": axis_names, "datasets": datasets}]


def test_flattened_channel_count_reads_deconvolution_run_settings(tmp_path: Path) -> None:
    path = tmp_path / "tile.ome.tif"
    tile_phase_module.deconvolution_sidecar(path).write_text(
        json.dumps({"provenance": {"run_settings": {"channels": 2}}})
    )

    assert tile_phase_module.flattened_channel_count(path) == 2


def test_estimate_tile_shift_zyx_px_recovers_synthetic_translation() -> None:
    z, y, x = np.mgrid[:32, :48, :48]
    fixed = np.exp(-(((z - 16) ** 2) / 12.0 + ((y - 24) ** 2 + (x - 24) ** 2) / 150.0)).astype(np.float32)
    moving = np.roll(fixed, shift=2, axis=0)
    moving = np.roll(moving, shift=-4, axis=1)
    moving = np.roll(moving, shift=5, axis=2)

    shift, details = estimate_tile_shift_zyx_px(fixed, moving, upsample_factor=10)

    np.testing.assert_allclose(shift, [-2, 4, -5], atol=0.25)
    assert details["corr_after"] > details["corr_before"]


def test_tile_phase_reads_zyx_and_czyx_ome_zarr_tiles(tmp_path) -> None:
    zyx_path = tmp_path / "Image_14.000.ome.zarr"
    czyx_path = tmp_path / "Image_10.000.ome.zarr"
    z, y, x = np.mgrid[:4, :6, :6]
    zyx_level0 = (z * 100 + y * 10 + x).astype(np.float32)
    czyx_level0 = np.stack([zyx_level0, zyx_level0 + 1000.0]).astype(np.float32)
    czyx_level1 = czyx_level0[:, ::2, ::2, ::2]
    _write_test_ome_zarr(zyx_path, axes="ZYX", levels=[zyx_level0, zyx_level0[::2, ::2, ::2]])
    _write_test_ome_zarr(czyx_path, axes="CZYX", levels=[czyx_level0, czyx_level1])
    reference = rough_legacy.TileRecord(
        tile=zyx_path.name,
        side="L",
        path=zyx_path,
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([4, 6, 6]),
        axes="ZYX",
    )
    moving = rough_legacy.TileRecord(
        tile=czyx_path.name,
        side="L",
        path=czyx_path,
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([4, 6, 6]),
        axes="CZYX",
    )

    fixed_patch = tile_phase_module.read_tile_patch(
        reference,
        channel=0,
        slices_zyx=(slice(1, 3), slice(2, 5), slice(1, 4)),
    )
    moving_patch = tile_phase_module.read_tile_patch(
        moving,
        channel=1,
        slices_zyx=(slice(1, 3), slice(2, 5), slice(1, 4)),
    )
    moving_indexed_patch = tile_phase_module.read_tile_indexed_z_patch(
        moving,
        channel=1,
        z_indices=np.arange(1, 3, dtype=np.int64),
        y_slice=slice(2, 5),
        x_slice=slice(1, 4),
    )
    moving_from_path = tile_phase_module.make_moving_tile_record(reference, czyx_path)
    scout, scale_zyx, source_level, available_levels, z_indices_l0 = tile_phase_module.sampled_tile_volume_from_subifd(
        moving,
        channel=1,
        requested_level=1,
    )
    position_record = {
        "tile": zyx_path.name,
        "path": str(zyx_path),
        "axes": "ZYX",
        "shape": [4, 6, 6],
        "translation_um": {"z": 1.0, "y": 2.0, "x": 3.0},
    }
    tile_phase_module._apply_shift_to_position_record(
        position_record,
        moving_tile_name=czyx_path.name,
        moving_path=czyx_path,
        shift_um=np.asarray([0.5, 1.5, 2.5]),
    )

    np.testing.assert_array_equal(fixed_patch, zyx_level0[1:3, 2:5, 1:4])
    np.testing.assert_array_equal(moving_patch, czyx_level0[1, 1:3, 2:5, 1:4])
    np.testing.assert_array_equal(moving_indexed_patch, czyx_level0[1, 1:3, 2:5, 1:4])
    np.testing.assert_array_equal(scout, czyx_level1[1])
    np.testing.assert_array_equal(scale_zyx, [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(z_indices_l0, [0, 2])
    np.testing.assert_array_equal(moving_from_path.shape_zyx, [4, 6, 6])
    assert moving_from_path.axes == "CZYX"
    assert position_record["axes"] == "CZYX"
    assert position_record["shape"] == [2, 4, 6, 6]
    assert position_record["channels"] == ["0", "1"]
    assert position_record["tracks"][0]["channels"] == [0, 1]
    assert position_record["translation_um"] == {"z": 1.5, "y": 3.5, "x": 5.5}
    assert source_level == 1
    assert available_levels == 2


@pytest.mark.parametrize(
    ("indices", "expected_indexer", "expected_reverse"),
    [
        (np.asarray([2, 3, 4]), slice(2, 5), False),
        (np.asarray([4, 3, 2]), slice(2, 5), True),
        (np.asarray([3]), slice(3, 4), False),
    ],
)
def test_contiguous_raw_z_indices_use_slice(
    indices: np.ndarray,
    expected_indexer: slice,
    expected_reverse: bool,
) -> None:
    indexer, reverse = tile_phase_module.raw_z_indexer(indices)

    assert indexer == expected_indexer
    assert reverse is expected_reverse


def test_sparse_raw_z_indices_remain_fancy_indexed() -> None:
    indices = np.asarray([1, 3, 6])

    indexer, reverse = tile_phase_module.raw_z_indexer(indices)

    np.testing.assert_array_equal(indexer, indices)
    assert reverse is False


def test_position_record_loader_accepts_side_less_canonical_positions(tmp_path) -> None:
    path = tmp_path / "fixed.000.ome.zarr"
    _write_test_ome_zarr(
        path,
        axes="CZYX",
        levels=[np.zeros((1, 4, 6, 8), dtype=np.uint16)],
    )

    tile = tile_phase_module.tile_record_from_position_record(
        {
            "tile": path.name,
            "path": str(path),
            "translation_um": {"z": 1.0, "y": 2.0, "x": 3.0},
            "scale_um": {"z": 0.6, "y": 0.3, "x": 0.3},
        }
    )

    assert tile.side == "all"
    assert tile.axes == "CZYX"
    np.testing.assert_array_equal(tile.shape_zyx, [4, 6, 8])


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


def test_patch_shift_correlation_improvement_uses_shifted_overlap_support(monkeypatch) -> None:
    z, y, x = np.mgrid[:4, :8, :8]
    fixed = (z + y * 2 + x * 3).astype(np.float32)
    moving = fixed.copy()
    moving[:, :2, :] = moving[:, :2, ::-1]

    monkeypatch.setattr(tile_phase_module, "normalize_volume_for_phase", lambda volume: volume.astype(np.float32))
    monkeypatch.setattr(
        tile_phase_module.stitch_legacy,
        "phase_correlation_shift_gpu",
        lambda *_args, **_kwargs: ((0.0, 2.0, 0.0), 1.0),
    )

    _shift, details = estimate_patch_shift_zyx_px(fixed, moving)

    assert details["corr_valid_overlap_fraction"] < 1.0
    assert details["corr_before"] != details["corr_before_full_support"]


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
                "fixed_slices": (slice(32, 48), slice(80, 112), slice(48, 80)),
                "content_score": 0.9,
                "positive_fraction": 0.4,
            },
            {
                "fixed_slices": (slice(40, 56), slice(48, 80), slice(80, 112)),
                "content_score": 0.8,
                "positive_fraction": 0.3,
            },
            {
                "fixed_slices": (slice(8, 24), slice(48, 80), slice(48, 80)),
                "content_score": 0.7,
                "positive_fraction": 0.3,
            },
            {
                "fixed_slices": (slice(24, 40), slice(48, 80), slice(48, 80)),
                "content_score": 0.6,
                "positive_fraction": 0.3,
            },
            {
                "fixed_slices": (slice(0, 16), slice(48, 80), slice(48, 80)),
                "content_score": 0.5,
                "positive_fraction": 0.3,
            },
            {
                "fixed_slices": (slice(4, 20), slice(48, 80), slice(48, 80)),
                "content_score": 0.4,
                "positive_fraction": 0.3,
            },
        ],
    )
    monkeypatch.setattr(
        tile_phase_module,
        "sampled_tile_volume_from_subifd",
        lambda *_args, **_kwargs: (
            np.ones((16, 32, 32), dtype=np.float32),
            np.asarray([4.0, 4.0, 4.0]),
            2,
            3,
            np.arange(16, dtype=np.int64) * 4,
        ),
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
        max_candidate_patches=7,
        min_inliers=3,
    )

    np.testing.assert_allclose(shift, [-3.0, 6.0, -9.0])
    assert details["n_inliers"] == 3
    assert details["n_measured"] == 3
    assert details["early_stop_after_patch"] == 2
    assert details["patches"][0]["moving_slices_zyx"] == [[20, 36], [40, 72], [60, 92]]
    assert details["patches"][3]["reason"] == "skipped_after_enough_inliers"


def test_sparse_scout_patch_rows_store_sampled_z_indices(monkeypatch) -> None:
    fixed = np.ones((32, 240, 240), dtype=np.float32)
    moving = np.ones_like(fixed)
    scale = np.asarray([3160.0 / 31.0, 4.0, 4.0])
    z_indices = np.rint(np.linspace(0, 3160, 32)).astype(np.int64)

    monkeypatch.setattr(
        tile_phase_module,
        "estimate_tile_shift_zyx_px_gpu",
        lambda *_args, **_kwargs: (np.asarray([0.0, 0.0, 0.0]), {"peak": 1.0, "corr_before": 0.2, "corr_after": 0.8}),
    )
    shift, details = tile_phase_module.measure_sparse_scout_patch_shift(
        fixed_coarse=fixed,
        moving_coarse=moving,
        fixed_coarse_scale_zyx=scale,
        fixed_z_indices_l0=z_indices,
        moving_z_indices_l0=z_indices,
        coarse_shift_coarse_px=np.zeros(3),
        coarse_details={"corr_before": 0.1, "corr_after": 0.7},
        patch_shape_zyx=(96, 320, 320),
        max_candidate_patches=3,
        min_inliers=3,
    )

    np.testing.assert_allclose(shift, [0.0, 0.0, 0.0])
    patch = details["patches"][0]
    assert patch["patch_source"] == "sparse_subifd_scout"
    assert patch["fixed_slices_zyx"][0] == [0, 3161]
    assert patch["moving_slices_zyx"][0] == [0, 3161]
    assert patch["fixed_z_indices_l0"] == z_indices.tolist()
    assert patch["moving_z_indices_l0"] == z_indices.tolist()


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


def test_complete_linkage_rejects_chained_inlier_cluster() -> None:
    shifts = np.asarray([[0.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 24.0, 0.0]])

    with pytest.raises(ValueError, match="require 3"):
        select_inlier_patch_measurements(
            shifts,
            thresholds_zyx=np.asarray([3.0, 12.0, 12.0]),
            min_inliers=3,
        )


def test_realized_seed_shift_uses_integer_patch_origin() -> None:
    fixed_slices = (slice(20, 52), slice(100, 164), slice(200, 264))

    moving_slices, realized = tile_phase_module.shifted_slices_with_realized_shift(
        fixed_slices,
        requested_shift_zyx_px=np.asarray([3.4, -5.6, 9.2]),
    )

    expected = np.asarray(
        [fixed.start - moving.start for fixed, moving in zip(fixed_slices, moving_slices, strict=True)],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(realized, expected)


def test_patch_quality_gates_reject_low_correlation_and_periodic_wrap() -> None:
    reasons = patch_quality_rejection_reasons(
        details={"corr_before": 0.1, "corr_after": 0.047},
        residual_shift_px=np.asarray([1.0, 1.0, 1.0]),
        patch_shape_zyx=(32, 512, 512),
    )
    assert "corr_after_below_threshold" in reasons

    reasons = patch_quality_rejection_reasons(
        details={"corr_before": 0.4, "corr_after": 0.3},
        residual_shift_px=np.asarray([1.0, 1.0, 1.0]),
        patch_shape_zyx=(32, 512, 512),
    )
    assert "corr_improvement_below_threshold" in reasons

    reasons = patch_quality_rejection_reasons(
        details={"corr_before": 0.2, "corr_after": 0.3},
        residual_shift_px=np.asarray([15.0, 1.0, 1.0]),
        patch_shape_zyx=(32, 512, 512),
    )
    assert "residual_near_periodic_wrap" in reasons


def test_patch_quality_rejects_weak_zero_residual_gradient_match() -> None:
    reasons = patch_quality_rejection_reasons(
        details={
            "corr_before": 0.70,
            "corr_after": 0.70,
            "gradient_component_ncc_before": 0.05,
            "gradient_component_ncc_after": 0.05,
        },
        residual_shift_px=np.asarray([0.0, 0.0, 0.0]),
        patch_shape_zyx=(32, 512, 512),
    )

    assert "weak_gradient_component_ncc_zero_residual" in reasons


def test_patch_quality_keeps_low_gradient_nonzero_refinement() -> None:
    reasons = patch_quality_rejection_reasons(
        details={
            "corr_before": 0.70,
            "corr_after": 0.70,
            "gradient_component_ncc_before": 0.05,
            "gradient_component_ncc_after": 0.05,
        },
        residual_shift_px=np.asarray([1.0, 0.0, 0.0]),
        patch_shape_zyx=(32, 512, 512),
    )

    assert "weak_gradient_component_ncc_zero_residual" not in reasons


def test_tile_phase_cli_min_inliers_default_and_floor() -> None:
    parameter = inspect.signature(cli_module.tile_phase_align).parameters["min_inliers"]
    assert parameter.default == 3
    assert inspect.signature(cli_module.tile_phase_align).parameters["moving_channel"].default == 0

    with pytest.raises(ValueError, match="patch-mode min_inliers must be >= 3"):
        align_tiles_to_reference(
            reference_position=Path("unused.json"),
            output_position=Path("out.json"),
            output_dir=Path("out"),
            patch_shape_zyx=(32, 512, 512),
            min_inliers=2,
        )


def test_cache_key_changes_when_position_or_tile_stat_changes(tmp_path) -> None:
    reference = tmp_path / "sample-488514561638.ome.tif"
    moving = tmp_path / "sample-405.ome.tif"
    reference.write_bytes(b"reference")
    moving.write_bytes(b"moving")
    position = tmp_path / "positions.json"
    position.write_text(json.dumps({"tiles": [{"tile": reference.name, "path": str(reference)}]}))
    kwargs = {
        "reference_position": position,
        "reference_channel": 3,
        "moving_channel": 0,
        "reference_token": "488514561638",
        "moving_token": "405",
        "level": 0,
        "upsample_factor": 10,
        "patch_shape_zyx": (32, 512, 512),
        "min_inliers": 3,
        "max_candidate_patches": 24,
        "coarse_level": 4,
        "scout_z_samples": 32,
    }

    before = tile_phase_cache_key(**kwargs)
    moving.write_bytes(b"moving changed")
    after_tile_change = tile_phase_cache_key(**kwargs)
    position.write_text(json.dumps({"tiles": [{"tile": reference.name, "path": str(reference), "note": "changed"}]}))
    after_position_change = tile_phase_cache_key(**kwargs)

    assert before != after_tile_change
    assert after_tile_change != after_position_change


def test_phase_cache_key_ignores_quality_metric_only_changes() -> None:
    current = {
        "cache_version": "tile_phase_robust_v4",
        "reference_position": "/tmp/positions.json",
        "patch_shape_zyx": [32, 512, 512],
        "quality_thresholds": {"corr_improvement_metric": "same_shifted_overlap_support"},
    }
    stored = {
        "cache_version": "tile_phase_robust_v4",
        "reference_position": "/tmp/positions.json",
        "patch_shape_zyx": [32, 512, 512],
        "quality_thresholds": {"min_corr_improvement": 0.0},
    }

    assert phase_cache_keys_match(stored, current)

    stored["patch_shape_zyx"] = [16, 512, 512]
    assert not phase_cache_keys_match(stored, current)


def test_cache_loader_rejects_invalid_measurement_status(tmp_path) -> None:
    cache_path = tmp_path / "tile_phase_measurement_cache.json"
    cache_key = {"cache_version": "test"}
    cache_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "measurements": [
                    {
                        "tile": "tile-405.ome.tif",
                        "measurement_status": "failed",
                        "shift_um_zyx": [0.0, 0.0, 0.0],
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="Invalid tile-phase measurement status"):
        tile_phase_module.load_tile_phase_cache(cache_path, cache_key)


def test_cached_patch_residuals_are_rescored_without_phase_recompute(monkeypatch) -> None:
    reference = rough_legacy.TileRecord(
        tile="ref",
        side="L",
        path=Path("ref.tif"),
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([8, 16, 16]),
        axes="ZYX",
    )
    moving = rough_legacy.TileRecord(
        tile="moving",
        side="L",
        path=Path("moving.tif"),
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([8, 16, 16]),
        axes="ZYX",
    )
    fixed_patch = np.zeros((4, 8, 8), dtype=np.float32)
    fixed_patch[:, 2:6, 2:6] = 10
    moving_patch = fixed_patch.copy()
    cached_attempt = {
        "patch_details": {
            "mode": "l0_patch_phase_robust_v2",
            "measurement_status": "direct_failed",
            "patch_shape_zyx": [4, 8, 8],
            "patches": [
                {
                        "patch_index": index,
                        "fixed_slices_zyx": [[0, 4], [index * 2, index * 2 + 8], [0, 8]],
                        "moving_slices_zyx": [[0, 4], [index * 2, index * 2 + 8], [0, 8]],
                        "residual_shift_px_zyx": [0.0, 0.0, 0.0],
                        "total_shift_px_zyx": [1.0 + index * 0.1, 2.0, 3.0],
                }
                for index in range(3)
            ],
        }
    }

    monkeypatch.setattr(
        tile_phase_module,
        "read_tile_patch",
        lambda tile, **_kwargs: fixed_patch if tile.tile == "ref" else moving_patch,
    )
    monkeypatch.setattr(
        tile_phase_module,
        "estimate_patch_shift_zyx_px",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("phase recomputed")),
    )

    shift, details = rescore_cached_patch_attempt(
        reference_tile=reference,
        moving_tile=moving,
        reference_channel=3,
        cached_attempt=cached_attempt,
        min_inliers=3,
    )

    np.testing.assert_allclose(shift, [1.1, 2.0, 3.0])
    assert details["measurement_status"] == "direct_accepted"
    assert details["cache_source"] == "cached_patch_residual_rescore"
    assert details["n_inliers"] == 3
    assert details["early_stop_after_patch"] == 2


def test_shift_field_fallback_is_called_with_three_inliers(tmp_path, monkeypatch) -> None:
    reference_path = tmp_path / "sample-488514561638.ome.tif"
    moving_path = tmp_path / "sample-405.ome.tif"
    reference_path.write_bytes(b"reference")
    moving_path.write_bytes(b"moving")
    position = tmp_path / "positions.json"
    position.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": reference_path.name,
                        "side": "L",
                        "path": str(reference_path),
                        "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                        "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                    }
                ]
            }
        )
    )
    reference_tile = rough_legacy.TileRecord(
        tile=reference_path.name,
        side="L",
        path=reference_path,
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([32, 64, 64]),
        axes="ZYX",
    )
    moving_tile = rough_legacy.TileRecord(
        tile=moving_path.name,
        side="L",
        path=moving_path,
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([32, 64, 64]),
        axes="ZYX",
    )
    seen = {}

    monkeypatch.setattr(tile_phase_module.rough_legacy, "load_tiles", lambda _payload: [reference_tile])
    monkeypatch.setattr(tile_phase_module, "make_moving_tile_record", lambda _reference_tile, _moving_path: moving_tile)
    monkeypatch.setattr(
        tile_phase_module,
        "measure_patch_tile_shift",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("direct failed")),
    )

    def fake_fallback(**kwargs):
        seen["min_inliers"] = kwargs["min_inliers"]
        return np.asarray([1.0, 2.0, 3.0]), {
            "corr_before": None,
            "corr_after": None,
            "n_inliers": 3,
        }

    monkeypatch.setattr(tile_phase_module, "infer_shift_from_adjacent_tiles", fake_fallback)

    align_tiles_to_reference(
        reference_position=position,
        output_position=tmp_path / "out.positions.json",
        output_dir=tmp_path / "phase",
        patch_shape_zyx=(16, 32, 32),
        min_inliers=3,
    )

    assert seen["min_inliers"] == 3


def test_failed_direct_attempt_is_cached_with_patch_correlations(tmp_path, monkeypatch) -> None:
    reference_path = tmp_path / "sample-488514561638.ome.tif"
    moving_path = tmp_path / "sample-405.ome.tif"
    reference_path.write_bytes(b"reference")
    moving_path.write_bytes(b"moving")
    position = tmp_path / "positions.json"
    position.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": reference_path.name,
                        "side": "L",
                        "path": str(reference_path),
                        "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                        "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                    }
                ]
            }
        )
    )
    reference_tile = rough_legacy.TileRecord(
        tile=reference_path.name,
        side="L",
        path=reference_path,
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([32, 64, 64]),
        axes="ZYX",
    )
    moving_tile = rough_legacy.TileRecord(
        tile=moving_path.name,
        side="L",
        path=moving_path,
        translation_zyx_um=np.zeros(3),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([32, 64, 64]),
        axes="ZYX",
    )
    direct_details = {
        "measurement_status": "direct_failed",
        "coarse_corr_before": 0.1,
        "coarse_corr_after": 0.2,
        "n_inliers": 1,
        "patches": [
            {
                "patch_index": 0,
                "status": "rejected",
                "reason": "quality_threshold",
                "corr_before": 0.05,
                "corr_after": 0.04,
            }
        ],
    }

    monkeypatch.setattr(tile_phase_module.rough_legacy, "load_tiles", lambda _payload: [reference_tile])
    monkeypatch.setattr(tile_phase_module, "make_moving_tile_record", lambda _reference_tile, _moving_path: moving_tile)
    monkeypatch.setattr(
        tile_phase_module,
        "measure_patch_tile_shift",
        lambda **_kwargs: (_ for _ in ()).throw(
            tile_phase_module.TilePhaseMeasurementError("direct failed", details=direct_details)
        ),
    )
    monkeypatch.setattr(
        tile_phase_module,
        "infer_shift_from_adjacent_tiles",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("fallback failed")),
    )

    output_dir = tmp_path / "phase"
    with pytest.raises(RuntimeError, match="Tile phase alignment failed"):
        align_tiles_to_reference(
            reference_position=position,
            output_position=tmp_path / "out.positions.json",
            output_dir=output_dir,
            patch_shape_zyx=(16, 32, 32),
            min_inliers=3,
        )

    cache = json.loads((output_dir / "tile_phase_measurement_cache.json").read_text())
    assert cache["measurements"] == []
    assert cache["attempts"][0]["measurement_status"] == "direct_failed"
    assert cache["attempts"][0]["patch_details"]["patches"][0]["corr_after"] == 0.04
    assert cache["attempts"][0]["fallback_error"] == "fallback failed"


def test_shift_field_fallback_uses_non_adjacent_same_side_successes() -> None:
    failed = rough_legacy.TileRecord(
        tile="failed",
        side="L",
        path=Path("failed.tif"),
        translation_zyx_um=np.asarray([0.0, 0.0, 0.0]),
        scale_zyx_um=np.ones(3),
        shape_zyx=np.asarray([32, 512, 512]),
        axes="ZYX",
    )
    successful = []
    for index, (translation, shift) in enumerate(
        [
            ([0.0, 10_000.0, 0.0], [10.0, 20.0, 30.0]),
            ([0.0, 20_000.0, 0.0], [11.0, 21.0, 31.0]),
            ([0.0, 30_000.0, 0.0], [9.5, 19.0, 29.0]),
        ]
    ):
        tile = rough_legacy.TileRecord(
            tile=f"success-{index}",
            side="L",
            path=Path(f"success-{index}.tif"),
            translation_zyx_um=np.asarray(translation, dtype=np.float64),
            scale_zyx_um=np.ones(3),
            shape_zyx=np.asarray([32, 512, 512]),
            axes="ZYX",
        )
        successful.append((tile, np.asarray(shift, dtype=np.float64)))

    shift_px, details = tile_phase_module.infer_shift_from_adjacent_tiles(
        failed_tile=failed,
        successful_tiles=successful,
        patch_shape_zyx=(32, 512, 512),
        min_inliers=3,
    )

    np.testing.assert_allclose(shift_px, [10.0, 20.0, 30.0])
    assert details["mode"] == "same_side_shift_field_fallback"
    assert details["n_inliers"] == 3
    assert all(row["reason"] == "same_side_shift_field_sample" for row in details["neighbors"])


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


def test_adapt_registration_writes_stage_records_with_identity_affine(tmp_path) -> None:
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
                "side": "L",
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
                "n_inliers": 4,
                "measurement_status": "direct_accepted",
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
    assert adapted["tiles"][0]["source_view"] == "L"
    assert adapted["tiles"][0]["stage_translation_um"] == {"z": 10.0, "y": 20.0, "x": 30.0}
    assert adapted["tiles"][0]["stage_scale_um"] == {"z": 2.0, "y": 0.5, "x": 0.5}
    assert adapted["tiles"][0]["registered_affine"]["matrix"] == np.eye(4).tolist()
    assert adapted["transform_contract"]["registered_affine_copied_exactly"] is False
    assert adapted["transform_contract"]["registered_affine_source"] == "identity_stage_translation_baked"
    assert adapted["tile_phase_summary"]["measurements"][0]["measurement_status"] == "direct_accepted"


def test_adapt_registration_rejects_failed_measurement(tmp_path) -> None:
    reference_registration = tmp_path / "registration.track0.json"
    output_registration = tmp_path / "registration.405.json"
    reference_registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "230Tnc-CL-488514561638.000.ome.tif",
                        "stage_translation_um": {"z": 1.0, "y": 2.0, "x": 3.0},
                        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                        "registered_affine": {"matrix": [[1.0]]},
                    }
                ]
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
        "measurements": [
            {
                "tile": "230Tnc-CL-405.000.ome.tif",
                "reference_tile": "230Tnc-CL-488514561638.000.ome.tif",
                "shift_px_zyx": [1.0, 2.0, 3.0],
                "shift_um_zyx": [2.0, 1.0, 1.5],
                "measurement_status": "failed",
            }
        ],
    }

    with pytest.raises(ValueError, match="Refusing to adapt registration"):
        adapt_registration_from_reference(
            reference_registration_input=reference_registration,
            output_registration=output_registration,
            adapted_position_payload=position_payload,
            reference_token="488514561638",
            moving_token="405",
            adapted_to_position=tmp_path / "positions.json",
            tile_phase_summary=summary,
        )

    assert not output_registration.exists()


def test_adapt_registration_rejects_fallback_measurement(tmp_path) -> None:
    reference_registration = tmp_path / "registration.track0.json"
    output_registration = tmp_path / "registration.405.json"
    reference_registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "230Tnc-CL-488514561638.000.ome.tif",
                        "stage_translation_um": {"z": 1.0, "y": 2.0, "x": 3.0},
                        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                    }
                ]
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
        "measurements": [
            {
                "tile": "230Tnc-CL-405.000.ome.tif",
                "reference_tile": "230Tnc-CL-488514561638.000.ome.tif",
                "shift_px_zyx": [1.0, 2.0, 3.0],
                "shift_um_zyx": [2.0, 1.0, 1.5],
                "n_inliers": 4,
                "measurement_status": "fallback_accepted",
                "fallback": True,
            }
        ],
    }

    with pytest.raises(ValueError, match="status=fallback_accepted"):
        adapt_registration_from_reference(
            reference_registration_input=reference_registration,
            output_registration=output_registration,
            adapted_position_payload=position_payload,
            reference_token="488514561638",
            moving_token="405",
            adapted_to_position=tmp_path / "positions.json",
            tile_phase_summary=summary,
        )

    assert not output_registration.exists()
