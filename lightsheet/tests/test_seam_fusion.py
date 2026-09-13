"""Seam fusion keeps one source away from bounded feathered interfaces."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from squisher_lightsheet.seam_fusion import feather_ownership, sharpness_seam_fusion


def test_blending_is_confined_to_the_interface():
    ownership = np.zeros((32, 128), dtype=np.int32)
    ownership[:, 64:] = 1
    valid = np.ones((2, 32, 128), dtype=bool)
    weights = feather_ownership(ownership, valid, (3, 8))
    np.testing.assert_array_equal(weights[0, :, :56], 1)
    np.testing.assert_array_equal(weights[0, :, 72:], 0)
    assert np.all((weights[0, :, 60:68] > 0) & (weights[0, :, 60:68] < 1))
    np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-6)


def test_missing_source_never_contributes_even_at_a_seam():
    ownership = np.zeros((32, 128), dtype=np.int32)
    ownership[:, 64:] = 1
    valid = np.ones((2, 32, 128), dtype=bool)
    valid[0, :, 64:] = False
    valid[1, :, :32] = False
    valid[:, :3] = False
    ownership[:3] = -1
    weights = feather_ownership(ownership, valid, (3, 8))
    np.testing.assert_array_equal(weights[~valid], 0)
    np.testing.assert_allclose(weights[:, 3:].sum(axis=0), 1, atol=1e-6)


def test_sharper_source_wins_even_when_the_blurred_source_is_brighter():
    rng = np.random.default_rng(321)
    sharp = 100 + 30 * gaussian_filter(rng.normal(size=(96, 256)), 0.7)
    blurred = 2 * gaussian_filter(sharp, 3)
    views = np.stack([blurred, sharp]).astype(np.float32)
    fused = sharpness_seam_fusion(
        views,
        np.ones_like(views),
        sigma_1=(2, 2),
        sigma_2=(5, 5),
        feather_radius=(3, 8),
    )
    np.testing.assert_allclose(fused[32:-32, 32:-32], sharp[32:-32, 32:-32], rtol=1e-6)


def test_chunk_halo_preserves_the_same_fusion_at_chunk_boundaries():
    rng = np.random.default_rng(123)
    texture = 100 + gaussian_filter(rng.normal(size=(96, 192)), 0.6) * 30
    blurred = gaussian_filter(texture, 3)
    views = np.stack([texture, blurred]).astype(np.float32)
    views[:, :, 96:] = views[::-1, :, 96:]
    kwargs = dict(sigma_1=(1, 1), sigma_2=(3, 3), feather_radius=(3, 5))
    full = sharpness_seam_fusion(views.copy(), np.ones_like(views), **kwargs)
    halo = sharpness_seam_fusion.required_overlap(kwargs)["x"]
    parts = []
    for start, stop in [(0, 96), (96, 192)]:
        lo, hi = max(0, start - halo), min(192, stop + halo)
        part = sharpness_seam_fusion(views[:, :, lo:hi].copy(), np.ones_like(views[:, :, lo:hi]), **kwargs)
        parts.append(part[:, start - lo : stop - lo])
    np.testing.assert_allclose(np.concatenate(parts, axis=1), full, rtol=1e-6, atol=1e-5)


def test_zero_width_uses_exact_source_ownership():
    ownership = np.zeros((16, 32), dtype=np.int32)
    ownership[:, 16:] = 1
    weights = feather_ownership(ownership, np.ones((2, 16, 32), dtype=bool), (0, 0))
    np.testing.assert_array_equal(weights[0], ownership == 0)
    np.testing.assert_array_equal(weights[1], ownership == 1)


@pytest.mark.gpu
def test_gpu_sharpness_and_missing_support_match_cpu():
    import cupy as cp

    rng = np.random.default_rng(8)
    sharp = (100 + 30 * gaussian_filter(rng.normal(size=(24, 48, 96)), 0.7)).astype(np.float32)
    views = np.stack([2 * gaussian_filter(sharp, 3), sharp])
    views[0, :, :, :10] = np.nan
    valid = np.isfinite(views).astype(np.float32)
    kwargs = dict(sigma_1=(1, 2, 2), sigma_2=(2, 4, 4), feather_radius=(2, 4, 4))
    expected = sharpness_seam_fusion(views, valid, **kwargs)
    actual = cp.asnumpy(sharpness_seam_fusion(cp.asarray(views), cp.asarray(valid), **kwargs))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


def test_native_only_source_support_is_preserved():
    views = np.full((2, 24, 32), np.nan, dtype=np.float32)
    views[0, :, 1] = 100
    views[1, :, 5:] = 200
    weights = np.isfinite(views).astype(np.float32)
    actual = sharpness_seam_fusion(views, weights, sigma_1=(1, 1), sigma_2=(2, 2), feather_radius=(2, 2))
    np.testing.assert_array_equal(actual[:, 1], 100)
    np.testing.assert_array_equal(actual[:, 5:], 200)
    np.testing.assert_array_equal(actual[:, 2:5], 0)
