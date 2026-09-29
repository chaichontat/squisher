"""Content weighting must ignore missing pixels at tile borders."""

import numpy as np
import pytest

from squisher_lightsheet.fusion import coarse_preibisch_content_weights


def test_coarse_preibisch_reports_exact_truncated_support():
    production = coarse_preibisch_content_weights.required_overlap(
        {"sigma_1": 7, "sigma_2": 17, "stride_zyx": (1, 8, 8)}
    )
    reduced = coarse_preibisch_content_weights.required_overlap(
        {"sigma_1": 1, "sigma_2": 2, "stride_zyx": (1, 8, 8)}
    )

    assert production == {"z": 1, "y": 104, "x": 104}
    assert reduced == {"z": 1, "y": 32, "x": 32}


def test_coarse_preibisch_exact_halo_preserves_stride8_partition():
    rng = np.random.default_rng(230514)
    views = np.stack(
        [
            2000 + 400 * rng.standard_normal((272, 336)),
            2700 + 600 * rng.standard_normal((272, 336)),
        ]
    ).astype(np.float32)
    blending = np.ones_like(views)
    kwargs = {
        "sigma_1": 7,
        "sigma_2": 17,
        "stride_zyx": (1, 8, 8),
        "softmax_exponent": 2,
    }

    def fused_core(x_start: int, x_stop: int) -> np.ndarray:
        patch = views[:, :, x_start:x_stop]
        weights = coarse_preibisch_content_weights(
            patch,
            blending[:, :, x_start:x_stop],
            **kwargs,
        )
        return np.sum(weights * patch, axis=0)[104:168, 104:-104]

    joint = fused_core(0, 336)
    split = np.concatenate([fused_core(0, 272), fused_core(64, 336)], axis=1)

    np.testing.assert_allclose(split, joint, rtol=0, atol=1e-3)


@pytest.mark.parametrize("stride", [(1, 1, 1), (1, 8, 8)])
def test_constant_tiles_do_not_gain_content_at_their_borders(stride):
    views = np.full((2, 32, 256), 100, dtype=np.float32)
    blending = np.ones_like(views)
    views[1, :, :80] = np.nan
    blending[1, :, :80] = 0
    views[1, :, 80:] = 200

    weights = coarse_preibisch_content_weights(
        views,
        blending,
        sigma_1=7,
        sigma_2=17,
        stride_zyx=stride,
        softmax_exponent=2,
    )

    np.testing.assert_allclose(weights[0, :, :80], 1)
    np.testing.assert_allclose(weights[1, :, :80], 0)
    np.testing.assert_allclose(weights[:, :, 80:], 0.5, atol=1e-6)
    assert np.isfinite(weights).all()
