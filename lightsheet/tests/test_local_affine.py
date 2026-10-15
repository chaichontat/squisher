import numpy as np
import pytest

from squisher_lightsheet.channel_affine import output_to_input_from_model
from squisher_lightsheet.local_affine import affine_from_local_shifts


def test_local_correspondences_recover_translation_and_shear_in_pull_coordinates() -> None:
    shape = (528, 528, 528)
    centers = np.array(
        [[80, 90, 100], [420, 90, 100], [80, 430, 100], [80, 90, 430], [300, 300, 300]], dtype=float
    )
    base_matrix = np.array([[1, 0.003, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    base_translation = np.array([2, -3, 4], dtype=float)
    shifts = np.column_stack(
        (1.5 + centers[:, 1] * 0.002, np.full(len(centers), -0.5), np.full(len(centers), 0.25))
    )
    matrix, translation = affine_from_local_shifts(
        matrix_zyx=base_matrix,
        translation_zyx=base_translation,
        shape_zyx=shape,
        centers_zyx=centers,
        shifts_zyx=shifts,
    )
    before_pull, before_offset = output_to_input_from_model(base_matrix, base_translation, shape)
    after_pull, after_offset = output_to_input_from_model(matrix, translation, shape)
    np.testing.assert_allclose(
        centers @ after_pull.T + after_offset, (centers - shifts) @ before_pull.T + before_offset, atol=1e-4
    )


def test_local_affine_rejects_coplanar_correspondences() -> None:
    with pytest.raises(ValueError, match="noncoplanar"):
        affine_from_local_shifts(
            matrix_zyx=np.eye(3),
            translation_zyx=np.zeros(3),
            shape_zyx=(100, 100, 100),
            centers_zyx=np.array([[20, 20, 20], [20, 80, 20], [20, 20, 80], [20, 80, 80]]),
            shifts_zyx=np.ones((4, 3)),
        )


def test_local_affine_rejects_nonfinite_correspondences() -> None:
    with pytest.raises(ValueError, match="finite"):
        affine_from_local_shifts(
            matrix_zyx=np.eye(3),
            translation_zyx=np.zeros(3),
            shape_zyx=(100, 100, 100),
            centers_zyx=np.array([[20, 20, 20], [80, 20, 20], [20, 80, 20], [20, 20, 80]]),
            shifts_zyx=np.full((4, 3), np.nan),
        )


@pytest.mark.parametrize("estimator", ["least-squares", "restrained"])
@pytest.mark.parametrize("z_shift", [2.0, 2.35])
def test_local_gpu_refinement_recovers_known_image_translation(z_shift: float, estimator: str) -> None:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter, shift
    from squisher_lightsheet.local_affine import refine_local_affine_gpu

    context = gaussian_filter(
        cp.asarray(np.random.default_rng(42).normal(size=(80, 80, 80)), dtype=cp.float32), 1.0
    )
    shifted = shift(context, (z_shift, 0, 0), order=1, mode="constant", cval=0, prefilter=False)
    fixed = context[8:72, 8:72, 8:72]
    moving = shifted[8:72, 8:72, 8:72]
    result = refine_local_affine_gpu(
        fixed,
        moving,
        fixed_mask=cp.ones(fixed.shape, dtype=cp.bool_),
        initial_matrix=np.eye(3),
        initial_translation=np.zeros(3),
        spacing_um_zyx=np.ones(3),
        patch_span_um=20,
        max_shift_um=4,
        estimator=estimator,
    )

    assert result.accepted, result.rejection_reason
    np.testing.assert_allclose(result.matrix_zyx, np.eye(3), atol=0.003)
    np.testing.assert_allclose(result.translation_zyx, [-z_shift, 0, 0], atol=0.1)
    assert sum(row["role"] == "validation" and row["confident"] for row in result.after) >= 4


def test_local_gpu_validation_rejects_spatially_inconsistent_image_shifts() -> None:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter, shift
    from squisher_lightsheet.local_affine import refine_local_affine_gpu

    fixed = gaussian_filter(
        cp.asarray(np.random.default_rng(17).normal(size=(64, 64, 64)), dtype=cp.float32), 1.0
    )
    displaced = shift(fixed, (3, 0, 0), order=1, mode="constant", cval=0, prefilter=False)
    central_band = (cp.arange(64)[None, None, :] >= 22) & (cp.arange(64)[None, None, :] < 42)
    moving = cp.where(central_band, displaced, fixed)
    result = refine_local_affine_gpu(
        fixed,
        moving,
        fixed_mask=cp.ones(fixed.shape, dtype=cp.bool_),
        initial_matrix=np.eye(3),
        initial_translation=np.zeros(3),
        spacing_um_zyx=np.ones(3),
        patch_span_um=20,
        max_shift_um=4,
    )

    assert not result.accepted
    assert result.rejection_reason == "local_3d_validation_residual"
    assert result.after


def _restrained(centers, shifts, spacing=None, matrix=None, translation=None):
    from squisher_lightsheet.local_affine import affine_from_restrained_local_shifts

    spacing = np.ones(3) if spacing is None else np.asarray(spacing)
    return affine_from_restrained_local_shifts(
        matrix_zyx=np.eye(3) if matrix is None else matrix,
        translation_zyx=np.zeros(3) if translation is None else translation,
        shape_zyx=(100, 100, 100),
        centers_zyx=centers,
        shifts_zyx=shifts,
        spacing_um_zyx=spacing,
        capture_spacing_um_zyx=spacing * [1, 2, 2],
    )


def _corners():
    from itertools import product

    return np.array(list(product([20.0, 80.0], repeat=3)))


def test_restrained_translation_is_unpenalized_with_nonidentity_initializer():
    centers = _corners()
    shifts = np.tile([4.0, -3.0, 2.0], (len(centers), 1))
    initial = np.array([[1.0, 0.04, 0.0], [0.0, 1.0, 0.03], [0.0, 0.0, 1.0]])
    offset = np.array([2.0, 3.0, -5.0])
    matrix, translation, diagnostics = _restrained(centers, shifts, matrix=initial, translation=offset)
    pull, origin = output_to_input_from_model(initial, offset, (100, 100, 100))
    actual_pull, actual_origin = output_to_input_from_model(matrix, translation, (100, 100, 100))
    np.testing.assert_allclose(actual_pull, pull, atol=1e-7)
    np.testing.assert_allclose(actual_origin, origin - pull @ shifts[0], atol=1e-6)
    assert diagnostics["converged"]


def test_restrained_shrinks_unsupported_shear():
    centers = _corners()
    shifts = np.zeros_like(centers)
    shifts[:, 0] = 0.02 * (centers[:, 1] - 50)
    matrix, _, _ = _restrained(centers, shifts)
    legacy, _ = affine_from_local_shifts(
        matrix_zyx=np.eye(3),
        translation_zyx=np.zeros(3),
        shape_zyx=(100, 100, 100),
        centers_zyx=centers,
        shifts_zyx=shifts,
    )
    assert abs(matrix[0, 1]) < abs(legacy[0, 1]) * 0.5


def test_restrained_resists_outlier():
    centers = np.tile(_corners(), (4, 1))
    shifts = np.tile([1.0, -0.5, 0.25], (len(centers), 1))
    shifts[0] += [40.0, -30.0, 20.0]
    _, translation, diagnostics = _restrained(centers, shifts)
    np.testing.assert_allclose(translation, [1.0, -0.5, 0.25], atol=0.25)
    assert diagnostics["robust_weights"][0] < 0.1


def test_restrained_is_equivariant_to_physical_unit_change():
    centers = _corners()
    shifts = np.random.default_rng(4).normal(size=centers.shape) + [1.0, 2.0, -1.0]
    first = _restrained(centers, shifts, spacing=[2.0, 0.3, 0.3])
    second = _restrained(centers, shifts, spacing=[2000.0, 300.0, 300.0])
    np.testing.assert_allclose(first[0], second[0], atol=1e-7)
    np.testing.assert_allclose(first[1], second[1], atol=1e-6)


def test_restrained_rejects_ill_conditioned_support():
    centers = _corners()
    centers[:, 0] = 50 + (centers[:, 0] - 50) * 0.0001
    with pytest.raises(ValueError, match="condition"):
        _restrained(centers, np.ones_like(centers))


def test_restrained_cache_policy_preserves_legacy_cache():
    from squisher_lightsheet.local_affine import local_affine_cache_config

    assert local_affine_cache_config("least-squares") == {}
    assert local_affine_cache_config("restrained")["estimator"] == "restrained-v1"
    with pytest.raises(ValueError, match="estimator"):
        local_affine_cache_config("unknown")


@pytest.mark.parametrize("estimator", ["least-squares", "restrained"])
def test_local_gpu_records_actual_sparse_mask_centroid(estimator):
    import cupy as cp
    from squisher_lightsheet.local_affine import refine_local_affine_gpu

    fixed = cp.zeros((64, 64, 64), dtype=cp.float32)
    mask = cp.zeros_like(fixed, dtype=cp.bool_)
    mask[:4, :8, :8] = True
    result = refine_local_affine_gpu(
        fixed,
        fixed,
        fixed_mask=mask,
        initial_matrix=np.eye(3),
        initial_translation=np.zeros(3),
        spacing_um_zyx=np.ones(3),
        patch_span_um=20,
        max_shift_um=4,
        estimator=estimator,
    )
    assert not result.accepted
    record = result.before[0]
    assert record["center_zyx"] == [9.5, 9.5, 9.5]
    assert record["support_center_zyx"] == [1.5, 3.5, 3.5]
    assert record["support_count"] == 256
    assert record["support_fraction"] == 256 / 8000
    assert result.before[-1]["support_center_zyx"] is None


def test_restrained_requires_six_fitting_points():
    with pytest.raises(ValueError, match="six noncoplanar"):
        _restrained(_corners()[:5], np.ones((5, 3)))


def test_restrained_prior_has_literal_gaussian_sd():
    centers = _corners()
    shifts = np.zeros_like(centers)
    shifts[:, 0] = 0.02 * (centers[:, 1] - 50)
    _, _, diagnostics = _restrained(centers, shifts)
    # In the quadratic branch, the physical strain normal equation adds 1/SD².
    expected = 0.02 * (8 * 30**2) / (8 * 30**2 + 1 / 0.01**2)
    np.testing.assert_allclose(diagnostics["strain_physical"][0][1], expected, atol=1e-12)
    assert diagnostics["policy"]["strain_penalty_factor"] == 0.5


def test_restrained_gpu_conditions_actual_support_centroids():
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    from squisher_lightsheet.local_affine import refine_local_affine_gpu

    fixed = gaussian_filter(
        cp.asarray(np.random.default_rng(37).normal(size=(64, 64, 64)), dtype=cp.float32), 1.0
    )
    mask = cp.broadcast_to((cp.arange(64)[None, None, :] % 22) < 16, fixed.shape)
    result = refine_local_affine_gpu(
        fixed,
        fixed,
        fixed_mask=mask,
        initial_matrix=np.eye(3),
        initial_translation=np.zeros(3),
        spacing_um_zyx=np.ones(3),
        patch_span_um=20,
        max_shift_um=4,
        estimator="restrained",
    )
    assert result.accepted, result.rejection_reason
    fitting = [row for row in result.before if row["role"] == "fit" and row["confident"]]
    actual = np.array([row["support_center_zyx"] for row in fitting])
    nominal = np.array([row["center_zyx"] for row in fitting])
    design = np.column_stack(((actual - 31.5) / 31.5, np.ones(len(actual))))
    nominal_design = np.column_stack(((nominal - 31.5) / 31.5, np.ones(len(actual))))
    np.testing.assert_allclose(result.diagnostics["normalized_design_condition"], np.linalg.cond(design))
    assert abs(np.linalg.cond(design) - np.linalg.cond(nominal_design)) > 0.01
    np.testing.assert_allclose(result.matrix_zyx, np.eye(3), atol=0.001)
    np.testing.assert_allclose(result.translation_zyx, np.zeros(3), atol=0.01)
