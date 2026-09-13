"""Content weighting must ignore missing pixels at tile borders."""

import numpy as np
import pytest

from squisher_lightsheet.fusion import coarse_preibisch_content_weights


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
