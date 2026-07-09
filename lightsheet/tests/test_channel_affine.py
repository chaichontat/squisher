from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from squisher_lightsheet._legacy.rough_align_tltr_center_z_phase import TileRecord
from squisher_lightsheet import channel_affine
from squisher_lightsheet import native_reg3dgpu
from squisher_lightsheet.channel_affine import (
    _native_window_content_stats,
    _native_window_low_content_reason,
    align_tiles_to_reference_affine,
    center_model_to_homogeneous_um,
    compose_registration_affine,
    crop_center_zyx,
    full_model_to_local,
    grid_fanout_order,
    homogeneous_um_to_center_model,
    level0_model_to_sampled,
    local_model_to_full,
    model_to_level0,
    moving_crop_start_for_fixed_crop,
    output_to_input_from_model,
    output_to_input_to_model,
    rigid_group_mean,
    select_content_fixed_crop_candidates_l0,
    select_content_fixed_crop_start_l0,
)


def _tile(name: str, y: float, x: float) -> TileRecord:
    return TileRecord(
        tile=name,
        side="L",
        path=Path(f"{name}.ome.zarr"),
        translation_zyx_um=np.asarray([0.0, y, x], dtype=np.float64),
        scale_zyx_um=np.ones(3, dtype=np.float64),
        shape_zyx=np.asarray([10, 10, 10], dtype=np.int64),
        axes="ZYX",
    )


def _position_record(tile: TileRecord) -> dict[str, object]:
    return {
        "tile": tile.tile,
        "side": tile.side,
        "path": str(tile.path),
        "translation_um": {"z": 0.0, "y": float(tile.translation_zyx_um[1]), "x": float(tile.translation_zyx_um[2])},
        "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        "shape": [10, 10, 10],
        "axes": "ZYX",
    }


def _skip_without_cuda() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CUDA device unavailable")
        cp.ones((1,), dtype=cp.float32).sum().item()
    except (cp.cuda.runtime.CUDARuntimeError, cp.cuda.driver.CUDADriverError) as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")


def _gradient_component_ncc_cupy_sobel_interior(
    fixed: np.ndarray, moving: np.ndarray, fixed_mask: np.ndarray | None = None
) -> dict[str, object]:
    cp = pytest.importorskip("cupy")
    from cupyx.scipy import ndimage as gpu_ndimage

    fixed_gpu = cp.asarray(fixed, dtype=cp.float32)
    moving_gpu = cp.asarray(moving, dtype=cp.float32)
    interior = cp.zeros(fixed_gpu.shape, dtype=cp.bool_)
    interior[1:-1, 1:-1, 1:-1] = True
    mask = interior & cp.isfinite(fixed_gpu) & cp.isfinite(moving_gpu) & (moving_gpu != 0)
    if fixed_mask is not None:
        mask &= cp.asarray(fixed_mask, dtype=cp.bool_)
    values: list[float] = []
    for axis in range(3):
        fixed_gradient = gpu_ndimage.sobel(fixed_gpu, axis=axis)
        moving_gradient = gpu_ndimage.sobel(moving_gpu, axis=axis)
        axis_mask = mask & cp.isfinite(fixed_gradient) & cp.isfinite(moving_gradient)
        count = int(cp.asnumpy(cp.count_nonzero(axis_mask)))
        if count < 256:
            values.append(float("nan"))
            continue
        fixed_values = cp.where(axis_mask, fixed_gradient, 0.0)
        moving_values = cp.where(axis_mask, moving_gradient, 0.0)
        fixed_mean = cp.sum(fixed_values) / cp.float32(count)
        moving_mean = cp.sum(moving_values) / cp.float32(count)
        fixed_centered = cp.where(axis_mask, fixed_gradient - fixed_mean, 0.0)
        moving_centered = cp.where(axis_mask, moving_gradient - moving_mean, 0.0)
        denominator = cp.sqrt(cp.sum(fixed_centered * fixed_centered) * cp.sum(moving_centered * moving_centered))
        values.append(
            float("nan")
            if float(cp.asnumpy(denominator)) == 0.0
            else float(cp.asnumpy(cp.sum(fixed_centered * moving_centered) / denominator))
        )
    finite = [value for value in values if np.isfinite(value)]
    cp.get_default_memory_pool().free_all_blocks()
    return {"mean": float(np.mean(finite)) if finite else float("nan"), "zyx_components": values}


def _registration_record(tile: TileRecord, *, registered_translation_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict[str, object]:
    record = _position_record(tile)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(registered_translation_zyx, dtype=np.float64)
    record["registered_affine"] = {"matrix": matrix.tolist()}
    return record


def test_fused_gradient_component_ncc_matches_cupy_sobel_interior_reference() -> None:
    _skip_without_cuda()
    rng = np.random.default_rng(42)
    fixed = rng.normal(size=(12, 13, 14)).astype(np.float32)
    moving = (0.8 * fixed + 0.2 * rng.normal(size=fixed.shape)).astype(np.float32)
    fixed[2, 3, 4] = np.nan
    moving[5, 6, 7] = np.inf
    moving[1:4, 2:5, 3:6] = 0.0

    fused = channel_affine.gradient_component_ncc_3d_gpu_fused(fixed, moving)
    reference = _gradient_component_ncc_cupy_sobel_interior(fixed, moving)

    assert fused["backend"] == "cupy_fused_sobel_interior"
    np.testing.assert_allclose(fused["zyx_components"], reference["zyx_components"], rtol=2e-4, atol=2e-4)
    assert fused["mean"] == pytest.approx(reference["mean"], rel=2e-4, abs=2e-4)


def test_corr_gpu_matches_cpu_masked_corr() -> None:
    _skip_without_cuda()
    rng = np.random.default_rng(45)
    fixed = rng.normal(size=(16, 17, 18)).astype(np.float32)
    moving = (0.5 * fixed + 0.5 * rng.normal(size=fixed.shape)).astype(np.float32)
    fixed[1, 2, 3] = np.nan
    moving[4, 5, 6] = np.inf
    moving[2:5, 3:6, 4:7] = 0.0

    gpu = channel_affine._corr_gpu(fixed, moving)
    cpu = channel_affine._corr(fixed, moving)

    assert gpu == pytest.approx(cpu, rel=2e-6, abs=2e-6)


def test_corr_gpu_honors_fixed_mask() -> None:
    _skip_without_cuda()
    rng = np.random.default_rng(48)
    fixed = rng.normal(size=(12, 13, 14)).astype(np.float32)
    moving = (0.7 * fixed + 0.3 * rng.normal(size=fixed.shape)).astype(np.float32)
    fixed_mask = fixed > np.percentile(fixed, 55.0)

    gpu = channel_affine._corr_gpu(fixed, moving, fixed_mask=fixed_mask)
    cpu = channel_affine.corrcoef_on_mask(
        fixed, moving, np.isfinite(fixed) & np.isfinite(moving) & (moving != 0) & fixed_mask
    )

    assert gpu == pytest.approx(cpu, rel=2e-6, abs=2e-6)


def test_fused_gradient_component_ncc_honors_fixed_mask() -> None:
    _skip_without_cuda()
    rng = np.random.default_rng(49)
    fixed = rng.normal(size=(14, 15, 16)).astype(np.float32)
    moving = (0.9 * fixed + 0.1 * rng.normal(size=fixed.shape)).astype(np.float32)
    fixed_mask = fixed > np.percentile(fixed, 35.0)

    fused = channel_affine.gradient_component_ncc_3d_gpu_fused(fixed, moving, fixed_mask=fixed_mask)
    reference = _gradient_component_ncc_cupy_sobel_interior(fixed, moving, fixed_mask=fixed_mask)

    assert fused["fixed_masked"] is True
    np.testing.assert_allclose(fused["zyx_components"], reference["zyx_components"], rtol=2e-4, atol=2e-4)
    assert fused["mean"] == pytest.approx(reference["mean"], rel=2e-4, abs=2e-4)


def test_corr_gpu_accepts_cupy_arrays_without_changing_mask_contract() -> None:
    _skip_without_cuda()
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(46)
    fixed = rng.normal(size=(9, 10, 11)).astype(np.float32)
    moving = (0.25 * fixed + rng.normal(size=fixed.shape)).astype(np.float32)
    fixed[0, 0, 0] = np.nan
    moving[1, 1, 1] = np.inf
    moving[2:4, 2:4, 2:4] = 0.0

    gpu = channel_affine._corr_gpu(cp.asarray(fixed), cp.asarray(moving))
    cpu = channel_affine._corr(fixed, moving)

    assert gpu == pytest.approx(cpu, rel=2e-6, abs=2e-6)


def test_corr_gpu_returns_none_for_zero_denominator() -> None:
    _skip_without_cuda()
    fixed = np.ones((4, 4, 4), dtype=np.float32)
    moving = np.ones((4, 4, 4), dtype=np.float32)

    assert channel_affine._corr_gpu(fixed, moving) is None


def test_block_mean_downsample_zyx_cupy_matches_reference() -> None:
    _skip_without_cuda()
    cp = pytest.importorskip("cupy")
    values = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    expected = values.reshape(2, 2, 3, 2, 4, 2).mean(axis=(1, 3, 5))

    reduced = channel_affine._block_mean_downsample_zyx_cupy(cp.asarray(values), (2, 2, 2))

    np.testing.assert_allclose(cp.asnumpy(reduced), expected, rtol=1e-6, atol=1e-6)


def test_robust_norm_and_content_stats_cupy_matches_numpy_wrapper() -> None:
    _skip_without_cuda()
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(47)
    values = np.abs(rng.normal(size=(8, 9, 10))).astype(np.float32)
    values[0, 0, 0] = np.nan

    gpu_values, gpu_stats = channel_affine._robust_norm_and_content_stats_cupy(cp.asarray(values))
    cpu_values, cpu_stats = channel_affine._robust_norm_and_content_stats_gpu(values)

    np.testing.assert_allclose(cp.asnumpy(gpu_values), cpu_values, rtol=1e-6, atol=1e-6)
    assert gpu_stats == pytest.approx(cpu_stats, rel=1e-6, abs=1e-6)


def test_fused_gradient_component_ncc_returns_nan_for_too_few_voxels() -> None:
    _skip_without_cuda()
    fixed = np.ones((4, 4, 4), dtype=np.float32)
    moving = np.ones((4, 4, 4), dtype=np.float32)

    fused = channel_affine.gradient_component_ncc_3d_gpu_fused(fixed, moving)

    assert np.isnan(fused["mean"])
    assert all(np.isnan(value) for value in fused["zyx_components"])


def test_fused_gradient_component_ncc_keeps_components_with_exact_minimum_support() -> None:
    _skip_without_cuda()
    rng = np.random.default_rng(43)
    fixed = rng.normal(size=(10, 10, 6)).astype(np.float32)
    moving = (fixed * 1.5 + 0.1 * rng.normal(size=fixed.shape)).astype(np.float32)

    fused = channel_affine.gradient_component_ncc_3d_gpu_fused(fixed, moving)
    reference = _gradient_component_ncc_cupy_sobel_interior(fixed, moving)

    assert all(np.isfinite(value) for value in fused["zyx_components"])
    np.testing.assert_allclose(fused["zyx_components"], reference["zyx_components"], rtol=2e-4, atol=2e-4)


def test_fused_gradient_component_ncc_uses_axis_specific_finite_support() -> None:
    _skip_without_cuda()
    rng = np.random.default_rng(44)
    fixed = rng.normal(size=(10, 10, 10)).astype(np.float32)
    moving = (0.7 * fixed + 0.3 * rng.normal(size=fixed.shape)).astype(np.float32)
    moving[5, 2:8, 2:8] = np.nan

    fused = channel_affine.gradient_component_ncc_3d_gpu_fused(fixed, moving)
    reference = _gradient_component_ncc_cupy_sobel_interior(fixed, moving)

    np.testing.assert_allclose(fused["zyx_components"], reference["zyx_components"], rtol=2e-4, atol=2e-4)
    assert fused["mean"] == pytest.approx(reference["mean"], rel=2e-4, abs=2e-4)


def test_output_to_input_affine_round_trips_model_parameters() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.01, 1.002, 0.003],
            [-0.02, -0.004, 0.999],
        ],
        dtype=np.float32,
    )
    translation = np.asarray([0.5, 48.0, -22.0], dtype=np.float32)

    output_to_input, offset = output_to_input_from_model(matrix, translation, (200, 240, 240))
    recovered_matrix, recovered_translation = output_to_input_to_model(output_to_input, offset, (200, 240, 240))

    np.testing.assert_allclose(recovered_matrix, matrix, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(recovered_translation, translation, rtol=1e-6, atol=1e-5)


def test_crop_center_zyx_uses_actual_chunk_center() -> None:
    center = crop_center_zyx(np.asarray([100, 52, 322]), np.asarray([200, 480, 480]))

    np.testing.assert_allclose(center, [199.5, 291.5, 561.5])


def test_native_window_content_gate_rejects_empty_windows() -> None:
    stats = _native_window_content_stats(np.zeros((4, 8, 8), dtype=np.float32))

    assert stats["std"] == 0.0
    assert stats["active_fraction"] == 0.0
    assert _native_window_low_content_reason(stats, prefix="moving") == "moving_constant"


def test_model_to_level0_scales_translation_and_cross_axis_terms() -> None:
    sampled_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.02, 1.0, 0.01],
            [-0.03, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sampled_translation = np.asarray([1.0, 49.0, -22.0], dtype=np.float32)

    matrix, translation = model_to_level0(
        model_matrix=sampled_matrix,
        model_translation=sampled_translation,
        sampled_factor_zyx=np.asarray([16.0, 4.0, 4.0], dtype=np.float32),
    )

    np.testing.assert_allclose(translation, [16.0, 196.0, -88.0], rtol=1e-6)
    np.testing.assert_allclose(matrix[1, 0], 0.02 * 4.0 / 16.0, rtol=1e-6)
    np.testing.assert_allclose(matrix[2, 0], -0.03 * 4.0 / 16.0, rtol=1e-6)
    np.testing.assert_allclose(matrix[1, 2], 0.01, rtol=1e-6)


def test_level0_and_sampled_conversion_support_different_fixed_moving_factors() -> None:
    sampled_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.02, 1.0, 0.01],
            [-0.03, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sampled_translation = np.asarray([1.0, 49.0, -22.0], dtype=np.float32)
    fixed_factor = np.asarray([16.0, 4.0, 4.0], dtype=np.float32)
    moving_factor = np.asarray([8.0, 2.0, 2.0], dtype=np.float32)

    level0_matrix, level0_translation = model_to_level0(
        model_matrix=sampled_matrix,
        model_translation=sampled_translation,
        fixed_sampled_factor_zyx=fixed_factor,
        moving_sampled_factor_zyx=moving_factor,
    )
    recovered_matrix, recovered_translation = level0_model_to_sampled(
        level0_matrix=level0_matrix,
        level0_translation=level0_translation,
        fixed_sampled_factor_zyx=fixed_factor,
        moving_sampled_factor_zyx=moving_factor,
    )

    np.testing.assert_allclose(recovered_matrix, sampled_matrix, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(recovered_translation, sampled_translation, rtol=1e-6, atol=1e-5)


def test_fit_downsample_native_pull_converts_back_to_full_crop_coordinates() -> None:
    factors = (2, 1, 1)
    full_matrix = np.asarray(
        [
            [1.0, 0.002, 0.003],
            [-0.006, 1.0, -0.005],
            [0.004, 0.007, 1.0],
        ],
        dtype=np.float64,
    )
    full_offset = np.asarray([4.0, -12.0, 8.0], dtype=np.float64)

    fit_matrix, fit_offset = channel_affine._model_to_fit_downsample(full_matrix, full_offset, factors)
    restored_matrix, restored_offset = channel_affine._native_pull_from_fit_downsample(fit_matrix, fit_offset, factors)

    np.testing.assert_allclose(restored_matrix, full_matrix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(restored_offset, full_offset, rtol=1e-12, atol=1e-12)


def test_full_and_local_crop_affine_round_trip() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.01, 1.0, 0.02],
            [-0.02, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation = np.asarray([0.0, 188.0, -82.0], dtype=np.float32)
    fixed_start = np.asarray([100, 220, 300], dtype=np.int64)
    moving_start = np.asarray([100, 32, 382], dtype=np.int64)

    local_matrix, local_translation = full_model_to_local(
        full_matrix=matrix,
        full_translation=translation,
        fixed_start_zyx=fixed_start,
        moving_start_zyx=moving_start,
        full_shape_zyx=np.asarray([3161, 960, 960]),
        crop_shape_zyx=np.asarray([200, 480, 480]),
    )
    recovered_matrix, recovered_translation = local_model_to_full(
        local_matrix=local_matrix,
        local_translation=local_translation,
        fixed_start_zyx=fixed_start,
        moving_start_zyx=moving_start,
        full_shape_zyx=np.asarray([3161, 960, 960]),
        crop_shape_zyx=np.asarray([200, 480, 480]),
    )

    np.testing.assert_allclose(recovered_matrix, matrix, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(recovered_translation, translation, rtol=1e-6, atol=1e-5)


def test_native_pull_matrix_order_converts_to_centered_model() -> None:
    pull_matrix_zyx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.9999842643737793, 0.005607467610388994],
            [0.0, -0.005607467610388994, 0.9999842643737793],
        ],
        dtype=np.float32,
    )
    pull_offset_zyx = np.asarray([0.0, 2.0, -3.0], dtype=np.float32)

    model_matrix, model_translation = channel_affine.output_to_input_to_model(
        pull_matrix_zyx,
        pull_offset_zyx,
        (128, 256, 256),
    )
    recovered_pull_matrix, recovered_pull_offset = channel_affine.output_to_input_from_model(
        model_matrix,
        model_translation,
        (128, 256, 256),
    )

    np.testing.assert_allclose(recovered_pull_matrix, pull_matrix_zyx, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(recovered_pull_offset, pull_offset_zyx, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(model_matrix, np.linalg.inv(pull_matrix_zyx), rtol=1e-6, atol=1e-6)


def test_native_method8_device_accepts_cupy_buffers() -> None:
    _skip_without_cuda()
    cp = pytest.importorskip("cupy")
    z, y, x = 6, 7, 8
    fixed = np.empty((z, y, x), dtype=np.float32)
    for zz in range(z):
        for yy in range(y):
            for xx in range(x):
                fixed[zz, yy, xx] = (
                    0.17 * float(xx + 1)
                    + 0.11 * float(yy * yy + 1)
                    + 0.07 * float(zz + 3)
                    + 0.03 * float((xx * yy + zz) % 5)
                )
    fixed_gpu = cp.ascontiguousarray(cp.asarray(fixed, dtype=cp.float32))
    moving_gpu = cp.ascontiguousarray(cp.asarray(fixed, dtype=cp.float32))
    fixed_mask_gpu = fixed_gpu > cp.percentile(fixed_gpu, 20)

    result = native_reg3dgpu.register_method8_device(
        fixed_gpu,
        moving_gpu,
        fixed_mask_zyx=fixed_mask_gpu,
        max_iterations=3,
        ftol=1e-2,
        device=0,
    )

    assert result.return_code == 0
    assert isinstance(result.registered_zyx, cp.ndarray)
    assert result.registered_zyx.shape == fixed.shape
    assert bool(cp.asnumpy(cp.isfinite(result.registered_zyx).all()))
    np.testing.assert_allclose(np.diag(result.matrix_xyz_3x4[:, :3]), [1.0, 1.0, 1.0], atol=1e-5)


def test_moving_crop_start_uses_inverse_affine() -> None:
    start = moving_crop_start_for_fixed_crop(
        fixed_start_zyx=np.asarray([100, 240, 240]),
        crop_shape_zyx=(200, 480, 480),
        full_matrix=np.eye(3, dtype=np.float32),
        full_translation=np.asarray([0.0, 188.0, -82.0], dtype=np.float32),
        fixed_shape_zyx=np.asarray([3161, 960, 960]),
        moving_shape_zyx=np.asarray([3161, 960, 960]),
    )

    np.testing.assert_array_equal(start, [100, 52, 322])


def test_content_crop_selection_prefers_high_content_supported_window() -> None:
    _skip_without_cuda()
    fixed = np.zeros((20, 24, 24), dtype=np.float32)
    moving = np.full_like(fixed, 0.01)
    fixed[10:14, 12:18, 4:10] = 2.0
    moving[10:14, 12:18, 4:10] = 2.0

    start, details = select_content_fixed_crop_start_l0(
        fixed_sampled=fixed,
        moving_registered_sampled=moving,
        sampled_factor_zyx=np.asarray([10.0, 4.0, 4.0]),
        sampled_z_l0=np.arange(20, dtype=np.int64) * 10,
        tile_shape_zyx=np.asarray([200, 96, 96]),
        crop_shape_zyx=(80, 48, 48),
    )

    assert details["score"] > 0.0
    assert int(start[0]) <= 100 <= int(start[0] + 80)
    assert int(start[1]) <= 48 <= int(start[1] + 48)
    assert int(start[2]) < 40 and int(start[2] + 48) > 16


def test_content_crop_candidates_corner_mode_prefers_spatial_corners(monkeypatch) -> None:
    fixed = np.zeros((3, 80, 80), dtype=np.float32)
    moving = np.zeros_like(fixed)
    foreground = np.zeros(fixed.shape, dtype=bool)
    foreground[:, ::2, ::2] = True

    def fake_maps(_fixed_sampled: np.ndarray, _moving_registered_sampled: np.ndarray) -> dict[str, np.ndarray | float]:
        return {
            "finite": np.ones(fixed.shape, dtype=bool),
            "moving_support": np.ones(fixed.shape, dtype=bool),
            "fixed_foreground": foreground,
            "moving_foreground": foreground,
            "mutual_foreground": foreground,
            "fixed_edge": np.ones(fixed.shape, dtype=np.float32),
            "moving_edge": np.ones(fixed.shape, dtype=np.float32),
            "fixed_foreground_threshold": 0.5,
            "moving_foreground_threshold": 0.5,
        }

    monkeypatch.setattr(channel_affine, "_content_selection_maps_gpu", fake_maps)

    candidates = select_content_fixed_crop_candidates_l0(
        fixed_sampled=fixed,
        moving_registered_sampled=moving,
        sampled_factor_zyx=np.asarray([1.0, 1.0, 1.0]),
        sampled_z_l0=np.arange(3, dtype=np.int64),
        tile_shape_zyx=np.asarray([3, 80, 80]),
        crop_shape_zyx=(3, 20, 20),
        max_candidates=4,
        selection_mode="corners",
    )

    starts = [start.tolist() for start, _details in candidates]
    assert starts == [[0, 0, 0], [0, 0, 60], [0, 60, 0], [0, 60, 60]]
    assert [details["corner_target_l0_yx"] for _start, details in candidates] == [
        [9.5, 9.5],
        [9.5, 69.5],
        [69.5, 9.5],
        [69.5, 69.5],
    ]


def test_center_model_to_homogeneous_um_preserves_translation_convention() -> None:
    affine = center_model_to_homogeneous_um(
        matrix_px=np.eye(3, dtype=np.float32),
        translation_px=np.asarray([1.0, 2.0, -3.0], dtype=np.float32),
        shape_zyx=np.asarray([100, 200, 300]),
        fixed_scale_um_zyx=np.asarray([0.6, 0.25, 0.25]),
        moving_scale_um_zyx=np.asarray([0.6, 0.25, 0.25]),
    )

    np.testing.assert_allclose(affine[:3, :3], np.eye(3), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(affine[:3, 3], [0.6, 0.5, -0.75], rtol=1e-6, atol=1e-6)


def test_homogeneous_um_to_center_model_round_trips() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.01, 0.0],
            [-0.02, 1.0, 0.03],
            [0.0, -0.01, 0.99],
        ],
        dtype=np.float32,
    )
    translation = np.asarray([2.0, -4.0, 8.0], dtype=np.float32)
    shape = np.asarray([100, 200, 300])
    scale = np.asarray([0.6, 0.25, 0.25])

    affine = center_model_to_homogeneous_um(
        matrix_px=matrix,
        translation_px=translation,
        shape_zyx=shape,
        fixed_scale_um_zyx=scale,
        moving_scale_um_zyx=scale,
    )
    recovered_matrix, recovered_translation = homogeneous_um_to_center_model(
        homogeneous_um=affine,
        shape_zyx=shape,
        fixed_scale_um_zyx=scale,
        moving_scale_um_zyx=scale,
    )

    np.testing.assert_allclose(recovered_matrix, matrix, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(recovered_translation, translation, rtol=1e-6, atol=1e-5)


def test_compose_registration_affine_left_multiplies_reference_transform() -> None:
    reference_affine = {
        "dims": ["x_in", "x_out"],
        "coords": {"x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]},
        "matrix": np.diag([1.0, 2.0, 3.0, 1.0]).tolist(),
    }
    channel_affine = np.eye(4, dtype=np.float64)
    channel_affine[:3, 3] = [1.0, 2.0, 3.0]

    composed = compose_registration_affine(reference_affine=reference_affine, channel_affine_um=channel_affine)

    np.testing.assert_allclose(np.asarray(composed["matrix"])[:3, 3], [1.0, 4.0, 9.0])


def test_compose_registration_affine_conjugates_channel_affine_through_stage() -> None:
    reference_affine = {
        "dims": ["x_in", "x_out"],
        "coords": {"x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]},
        "matrix": np.diag([1.0, 2.0, 3.0, 1.0]).tolist(),
    }
    channel_affine = np.eye(4, dtype=np.float64)
    channel_affine[:3, :3] = np.diag([1.0, 1.1, 0.9])
    channel_affine[:3, 3] = [1.0, 2.0, 3.0]
    stage = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    local_point = np.asarray([4.0, 5.0, 6.0, 1.0], dtype=np.float64)

    composed = compose_registration_affine(
        reference_affine=reference_affine,
        channel_affine_um=channel_affine,
        stage_translation_um_zyx=stage,
    )

    expected = (
        np.asarray(reference_affine["matrix"], dtype=np.float64)
        @ np.r_[stage + (channel_affine @ local_point)[:3], 1.0]
    )
    actual = np.asarray(composed["matrix"], dtype=np.float64) @ np.r_[stage + local_point[:3], 1.0]
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_compose_registration_affine_uses_distinct_moving_stage() -> None:
    reference_affine = {
        "dims": ["x_in", "x_out"],
        "coords": {"x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]},
        "matrix": np.diag([1.0, 2.0, 3.0, 1.0]).tolist(),
    }
    channel_affine = np.eye(4, dtype=np.float64)
    channel_affine[:3, :3] = np.diag([1.0, 1.1, 0.9])
    channel_affine[:3, 3] = [1.0, 2.0, 3.0]
    reference_stage = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    moving_stage = np.asarray([7.0, 11.0, 13.0], dtype=np.float64)
    local_point = np.asarray([4.0, 5.0, 6.0, 1.0], dtype=np.float64)

    composed = compose_registration_affine(
        reference_affine=reference_affine,
        channel_affine_um=channel_affine,
        stage_translation_um_zyx=reference_stage,
        moving_stage_translation_um_zyx=moving_stage,
    )

    expected = (
        np.asarray(reference_affine["matrix"], dtype=np.float64)
        @ np.r_[reference_stage + (channel_affine @ local_point)[:3], 1.0]
    )
    actual = np.asarray(composed["matrix"], dtype=np.float64) @ np.r_[moving_stage + local_point[:3], 1.0]
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_grid_fanout_order_starts_at_center_and_expands_by_neighbors() -> None:
    tiles = [_tile(f"tile_{row}_{col}", row * 100.0, col * 100.0) for row in range(3) for col in range(3)]
    records = [_position_record(tile) for tile in tiles]

    order, metadata = grid_fanout_order(records, tiles)

    assert records[order[0]]["tile"] == "tile_1_1"
    assert {records[index]["tile"] for index in order[1:5]} == {"tile_0_1", "tile_1_0", "tile_1_2", "tile_2_1"}
    assert [row["grid_ring"] for row in metadata] == sorted(row["grid_ring"] for row in metadata)


def test_grid_fanout_order_uses_registered_stage_translation_centers() -> None:
    tiles = [_tile("left", 0.0, 0.0), _tile("center", 0.0, 100.0), _tile("right", 0.0, 200.0)]
    records = []
    for tile in tiles:
        record = _position_record(tile)
        record["stage_translation_um"] = record.pop("translation_um")
        record["registered_affine"] = {"matrix": np.eye(4, dtype=np.float64).tolist()}
        records.append(record)
    records[0]["registered_affine"]["matrix"][1][3] = 1000.0

    order, _metadata = grid_fanout_order(records, tiles)

    assert records[order[0]]["tile"] == "center"


def test_rigid_group_mean_averages_rotation_with_quaternions() -> None:
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    first[:3, :3] = Rotation.from_euler("z", -10.0, degrees=True).as_matrix()
    second[:3, :3] = Rotation.from_euler("z", 10.0, degrees=True).as_matrix()
    first[:3, 3] = [0.0, 2.0, -4.0]
    second[:3, 3] = [0.0, 6.0, -8.0]

    mean, stats = rigid_group_mean([first, second])

    assert stats["method"] == "rigid_quaternion_rotation_mean"
    assert stats["count"] == 2
    np.testing.assert_allclose(mean[:3, :3], np.eye(3), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(mean[:3, 3], [0.0, 4.0, -6.0], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(mean[:3, :3].T @ mean[:3, :3], np.eye(3), rtol=1e-8, atol=1e-8)


def test_affine_batch_uses_running_prior_after_five_accepted_tiles(tmp_path, monkeypatch) -> None:
    tiles = [_tile(f"tile_{index}", float(index // 3) * 100.0, float(index % 3) * 100.0) for index in range(6)]
    payload = {"tiles": [_position_record(tile) for tile in tiles]}
    position_path = tmp_path / "position.json"
    output_path = tmp_path / "aligned.json"
    output_dir = tmp_path / "out"
    position_path.write_text(json.dumps(payload))
    prior_used: list[bool] = []

    def fake_moving_path(path: Path, *, reference_token: str, moving_token: str) -> Path:
        return path.with_name(path.name.replace(reference_token, moving_token))

    def fake_moving_tile(reference_tile: TileRecord, moving_path: Path) -> TileRecord:
        return TileRecord(
            tile=moving_path.name,
            side=reference_tile.side,
            path=moving_path,
            translation_zyx_um=reference_tile.translation_zyx_um.copy(),
            scale_zyx_um=reference_tile.scale_zyx_um.copy(),
            shape_zyx=reference_tile.shape_zyx.copy(),
            axes=reference_tile.axes,
        )

    def fake_measure(**kwargs):
        prior = kwargs["prior_channel_affine_um"]
        assert kwargs["fit_mode"] == "rigid"
        prior_used.append(prior is not None)
        affine = np.eye(4, dtype=np.float64)
        return {
            "reference_tile": kwargs["reference_tile"].tile,
            "moving_tile": kwargs["moving_tile"].tile,
            "reference_path": str(kwargs["reference_tile"].path),
            "moving_path": str(kwargs["moving_tile"].path),
            "status": "accepted",
            "running_affine_prior": {"used": prior is not None},
            "refined_translation_um_zyx": [0.0, 0.0, 0.0],
            "channel_affine_um_zyx_homogeneous": affine.tolist(),
            "corr_refined_affine": 1.0,
            "gradient_component_ncc_refined_affine": {"mean": 1.0},
        }

    monkeypatch.setattr(channel_affine, "corresponding_moving_path", fake_moving_path)
    monkeypatch.setattr(channel_affine, "make_moving_tile_record", fake_moving_tile)
    monkeypatch.setattr(channel_affine, "_measure_tile_affine", fake_measure)

    align_tiles_to_reference_affine(
        reference_position=position_path,
        output_position=output_path,
        output_dir=output_dir,
        reference_token="tile",
        moving_token="moving",
        running_average_min_inliers=5,
        render_contact_sheet=False,
    )

    assert prior_used.count(False) == 5
    assert prior_used[-1] is True
    summary = json.loads((output_dir / "tile_affine_alignment.json").read_text())
    assert summary["running_affine_mean"]["count"] == 6
    assert summary["running_affine_mean"]["method"] == "rigid_quaternion_rotation_mean"
    assert summary["running_affine_mean"]["prior_transform_family"] == "rigid"


def test_affine_12dof_batch_still_uses_rigid_quaternion_prior(tmp_path, monkeypatch) -> None:
    tiles = [_tile(f"tile_{index}", float(index // 3) * 100.0, float(index % 3) * 100.0) for index in range(6)]
    payload = {"tiles": [_position_record(tile) for tile in tiles]}
    position_path = tmp_path / "position.json"
    output_path = tmp_path / "aligned.json"
    output_dir = tmp_path / "out"
    position_path.write_text(json.dumps(payload))

    def fake_moving_path(path: Path, *, reference_token: str, moving_token: str) -> Path:
        return path.with_name(path.name.replace(reference_token, moving_token))

    def fake_moving_tile(reference_tile: TileRecord, moving_path: Path) -> TileRecord:
        return TileRecord(
            tile=moving_path.name,
            side=reference_tile.side,
            path=moving_path,
            translation_zyx_um=reference_tile.translation_zyx_um.copy(),
            scale_zyx_um=reference_tile.scale_zyx_um.copy(),
            shape_zyx=reference_tile.shape_zyx.copy(),
            axes=reference_tile.axes,
        )

    def fake_measure(**kwargs):
        assert kwargs["fit_mode"] == "affine-12dof"
        affine = np.eye(4, dtype=np.float64)
        return {
            "reference_tile": kwargs["reference_tile"].tile,
            "moving_tile": kwargs["moving_tile"].tile,
            "reference_path": str(kwargs["reference_tile"].path),
            "moving_path": str(kwargs["moving_tile"].path),
            "status": "accepted",
            "running_affine_prior": {"used": kwargs["prior_channel_affine_um"] is not None},
            "refined_translation_um_zyx": [0.0, 0.0, 0.0],
            "channel_affine_um_zyx_homogeneous": affine.tolist(),
            "corr_refined_affine": 1.0,
            "gradient_component_ncc_refined_affine": {"mean": 1.0},
        }

    monkeypatch.setattr(channel_affine, "corresponding_moving_path", fake_moving_path)
    monkeypatch.setattr(channel_affine, "make_moving_tile_record", fake_moving_tile)
    monkeypatch.setattr(channel_affine, "_measure_tile_affine", fake_measure)

    align_tiles_to_reference_affine(
        reference_position=position_path,
        output_position=output_path,
        output_dir=output_dir,
        reference_token="tile",
        moving_token="moving",
        fit_mode="affine-12dof",
        running_average_min_inliers=5,
        render_contact_sheet=False,
    )

    summary = json.loads((output_dir / "tile_affine_alignment.json").read_text())
    assert summary["fit_mode"] == "affine-12dof"
    assert summary["running_affine_mean"]["fit_mode"] == "affine-12dof"
    assert summary["running_affine_mean"]["method"] == "rigid_quaternion_rotation_mean"
    assert summary["running_affine_mean"]["prior_transform_family"] == "rigid"
