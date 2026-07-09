from pathlib import Path
from types import SimpleNamespace

import numpy as np

import squisher_lightsheet.cross_register_method8 as tile_quadrant_method8


def _position_record(
    tmp_path: Path,
    *,
    tile: str,
    translation_zyx_um: tuple[float, float, float],
    shape: tuple[int, ...] = (3161, 960, 960),
    axes: str = "ZYX",
    scale_zyx_um: tuple[float, float, float] = (0.6, 0.28780706498795744, 0.28780706498795744),
) -> dict[str, object]:
    return {
        "tile": tile,
        "path": str(tmp_path / tile),
        "side": "L",
        "shape": list(shape),
        "axes": axes,
        "translation_um": dict(zip(("z", "y", "x"), translation_zyx_um, strict=True)),
        "scale_um": dict(zip(("z", "y", "x"), scale_zyx_um, strict=True)),
    }


def test_quadrant_z_windows_cover_960_tile_with_480_step_and_528_window() -> None:
    windows = tile_quadrant_method8.quadrant_z_windows((3161, 960, 960))

    assert len(windows) == 28
    assert sorted({tuple(window["fixed_start_zyx"][1:3]) for window in windows}) == [
        (0, 0),
        (0, 432),
        (432, 0),
        (432, 432),
    ]
    assert sorted({window["fixed_start_zyx"][0] for window in windows}) == [0, 480, 960, 1440, 1920, 2400, 2633]
    for window in windows:
        start = np.asarray(window["fixed_start_zyx"])
        stop = np.asarray(window["fixed_stop_zyx"])
        np.testing.assert_array_equal(stop - start, [528, 528, 528])
        assert np.all(start >= 0)
        assert np.all(stop <= [3161, 960, 960])


def test_preseeded_level0_model_uses_coarse_position_translation_at_tile_center(tmp_path: Path) -> None:
    fixed = _position_record(
        tmp_path,
        tile="Image_14.000.ome.zarr",
        translation_zyx_um=(0.0, 100.0, 200.0),
        scale_zyx_um=(2.0, 1.0, 0.5),
    )
    moving = _position_record(
        tmp_path,
        tile="Image_10.000.ome.zarr",
        translation_zyx_um=(4.0, 112.0, 207.0),
        scale_zyx_um=(2.0, 1.0, 0.5),
    )

    matrix, translation = tile_quadrant_method8.preseeded_level0_model(
        fixed_record=fixed,
        moving_record=moving,
        preseed_matrix_zyx=tile_quadrant_method8.README_PRESEED_MATRIX_ZYX,
    )

    np.testing.assert_allclose(matrix, tile_quadrant_method8.README_PRESEED_MATRIX_ZYX)
    np.testing.assert_allclose(translation, [2.0, 12.0, 14.0])


def test_scaled_local_initial_model_conjugates_affine_by_fit_downsample() -> None:
    full_matrix = np.asarray(
        [[1.0, 0.1, 0.0], [0.2, 1.0, -0.3], [0.0, 0.4, 1.0]],
        dtype=np.float64,
    )
    full_translation = np.asarray([4.0, 6.0, 8.0], dtype=np.float64)
    factors = (2, 3, 4)

    local_matrix, local_translation, fit_matrix, fit_translation = tile_quadrant_method8.scaled_local_initial_model(
        full_matrix_zyx=full_matrix,
        full_translation_zyx=full_translation,
        fixed_start_zyx=np.asarray([10, 20, 30]),
        moving_start_zyx=np.asarray([10, 20, 30]),
        full_shape_zyx=np.asarray([100, 100, 100]),
        crop_shape_zyx=np.asarray([20, 20, 20]),
        fit_downsample_zyx=factors,
    )

    scale = np.diag(1.0 / np.asarray(factors, dtype=np.float64))
    inverse_scale = np.diag(np.asarray(factors, dtype=np.float64))
    np.testing.assert_allclose(fit_matrix, scale @ local_matrix @ inverse_scale)
    np.testing.assert_allclose(fit_translation, scale @ local_translation)


def test_obviously_empty_detects_flat_nonzero_background() -> None:
    stats = tile_quadrant_method8._raw_empty_stats(np.full((4, 5, 6), 123.0, dtype=np.float32))

    assert stats["p50"] == 123.0
    assert tile_quadrant_method8._obviously_empty(stats, min_dynamic_range=1.0, min_std=0.25)


def test_obviously_empty_keeps_structured_nonzero_block() -> None:
    volume = np.zeros((4, 5, 6), dtype=np.float32)
    volume[:, :, :3] = 100.0
    volume[:, :, 3:] = 140.0
    stats = tile_quadrant_method8._raw_empty_stats(volume)

    assert stats["dynamic_range_p99_p01"] > 1.0
    assert not tile_quadrant_method8._obviously_empty(stats, min_dynamic_range=1.0, min_std=0.25)


def test_resume_cache_reuses_completed_rows_and_reruns_errors() -> None:
    cache_config = {"fixed_mask_threshold": 3000.0, "fixed_mask_min_voxels": 256}
    precheck = {"version": tile_quadrant_method8.EMPTY_PRECHECK_VERSION}

    assert tile_quadrant_method8._is_resumable_row(
        {
            "status": "accepted",
            "empty_precheck": precheck,
            "cache_config": cache_config,
            "quality_mask": "fixed_threshold_mask",
            "native_return_code": 0,
        },
        cache_config=cache_config,
    )
    assert tile_quadrant_method8._is_resumable_row(
        {
            "status": "rejected",
            "rejection_reason": "fixed_threshold_fit_mask_empty",
            "empty_precheck": precheck,
            "cache_config": cache_config,
        },
        cache_config=cache_config,
    )
    assert not tile_quadrant_method8._is_resumable_row(
        {
            "status": "error",
            "error": "cupy.cuda.memory.OutOfMemoryError: out of memory",
            "empty_precheck": precheck,
            "cache_config": cache_config,
            "quality_mask": "fixed_threshold_mask",
        },
        cache_config=cache_config,
    )
    assert not tile_quadrant_method8._is_resumable_row(
        {
            "status": "error",
            "error": "native_process_terminated: exitcode=-9",
            "empty_precheck": precheck,
            "cache_config": cache_config,
            "quality_mask": "fixed_threshold_mask",
        },
        cache_config=cache_config,
    )


def test_resume_cache_reruns_rows_from_different_fixed_mask_config() -> None:
    precheck = {"version": tile_quadrant_method8.EMPTY_PRECHECK_VERSION}
    old_config = {"fixed_mask_threshold": 2000.0, "fixed_mask_min_voxels": 256}
    new_config = {"fixed_mask_threshold": 3000.0, "fixed_mask_min_voxels": 256}

    assert not tile_quadrant_method8._is_resumable_row(
        {
            "status": "accepted",
            "empty_precheck": precheck,
            "cache_config": old_config,
            "quality_mask": "fixed_threshold_mask",
        },
        cache_config=new_config,
    )


def test_cache_config_includes_inputs_and_window_geometry(tmp_path: Path) -> None:
    args = SimpleNamespace(
        fixed_position=tmp_path / "fixed.positions.json",
        moving_position=tmp_path / "moving.positions.json",
        fixed_channel=0,
        moving_channel=0,
        core_shape_zyx=(480, 480, 480),
        window_shape_zyx=(528, 528, 528),
        fit_downsample_zyx=(1, 1, 1),
        native_lib_dir=tmp_path,
        ftol=1e-4,
        max_iterations=300,
        min_corr=0.15,
        min_grad_ncc=0.24,
        empty_precheck_level=-1,
        empty_precheck_min_dynamic_range=1.0,
        empty_precheck_min_std=0.25,
        fixed_mask_threshold=3000.0,
        fixed_mask_level=2,
        fixed_mask_min_voxels=256,
        fixed_mask_max_masked_fraction=0.95,
    )

    config = tile_quadrant_method8._cache_config_from_args(
        args,
        preseed_matrix_zyx=tile_quadrant_method8.README_PRESEED_MATRIX_ZYX,
    )

    assert config["fixed_position"].endswith("fixed.positions.json")
    assert config["moving_position"].endswith("moving.positions.json")
    assert config["core_shape_zyx"] == [480, 480, 480]
    assert config["window_shape_zyx"] == [528, 528, 528]


def test_resume_cache_requires_mask_evidence_for_masked_rows() -> None:
    cache_config = {"fixed_mask_threshold": 3000.0, "fixed_mask_min_voxels": 256}
    precheck = {"version": tile_quadrant_method8.EMPTY_PRECHECK_VERSION}

    assert not tile_quadrant_method8._is_resumable_row(
        {"status": "accepted", "empty_precheck": precheck, "cache_config": cache_config},
        cache_config=cache_config,
    )


def test_native_phase_only_attempts_are_never_resumable() -> None:
    cache_config = {"fixed_mask_threshold": 3000.0, "fixed_mask_min_voxels": 256}
    precheck = {"version": tile_quadrant_method8.EMPTY_PRECHECK_VERSION}

    assert not tile_quadrant_method8._is_resumable_row(
        {
            "status": "accepted",
            "empty_precheck": precheck,
            "cache_config": cache_config,
            "quality_mask": "fixed_threshold_mask",
            "native_return_code": None,
        },
        cache_config=cache_config,
    )
