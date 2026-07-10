from __future__ import annotations

import numpy as np

from squisher_lightsheet import phase_metrics
from squisher_lightsheet import tile_phase
from squisher_lightsheet import track_z
from squisher_lightsheet._legacy import rough_align_tltr_center_z_phase as rough_legacy


def test_robust_normalize_matches_existing_wrappers() -> None:
    image = np.asarray(
        [
            [[0.0, 1.0], [2.0, np.nan]],
            [[4.0, 8.0], [16.0, 32.0]],
        ],
        dtype=np.float32,
    )

    expected = phase_metrics.robust_normalize(image)

    assert expected.dtype == np.float32
    assert np.array_equal(tile_phase.normalize_volume_for_phase(image), expected)
    assert np.array_equal(track_z.robust_normalize(image), expected)


def test_corrcoef_on_mask_matches_existing_wrappers() -> None:
    fixed = np.arange(12, dtype=np.float32).reshape(3, 4)
    moving = fixed * 2.0
    mask = np.ones(fixed.shape, dtype=bool)

    expected = phase_metrics.corrcoef_on_mask(fixed, moving, mask)

    assert np.isclose(expected, 1.0)
    assert tile_phase.corrcoef_on_mask(fixed, moving, mask) == expected
    assert track_z.corrcoef_masked(fixed, moving, mask) == expected
    assert rough_legacy.corrcoef_on_mask(fixed, moving, mask) == expected


def test_corrcoef_on_mask_rejects_too_few_or_constant_pixels() -> None:
    fixed = np.ones((3, 4), dtype=np.float32)
    moving = np.ones((3, 4), dtype=np.float32)

    assert phase_metrics.corrcoef_on_mask(fixed, moving, np.zeros((3, 4), dtype=bool)) is None
    assert phase_metrics.corrcoef_on_mask(fixed, moving, np.ones((3, 4), dtype=bool)) is None
