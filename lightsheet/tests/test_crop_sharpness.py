"""Native crop preferences preserve fine detail and stable overlap ownership."""

import numpy as np
from scipy.ndimage import gaussian_filter
from squisher_lightsheet.crop_sharpness import native_crop_scores, crop_sharpness_fusion, overlap_crop_boxes


def test_native_crops_detect_detail_that_striding_would_erase():
    x = np.arange(128)
    sharp = np.broadcast_to(100 + 20 * np.sin(np.pi * x / 2), (64, 128)).astype(np.float32)
    blurred = 2 * gaussian_filter(sharp, 2)
    scores = native_crop_scores(np.stack([sharp, blurred]), (2, 2))
    assert scores[0] > 10 * scores[1]


def test_crop_preference_selects_the_same_source_across_chunk_boundaries():
    rng = np.random.default_rng(1)
    views = rng.uniform(10, 200, size=(3, 48, 96)).astype(np.float32)
    valid = np.ones_like(views)
    valid[1, :, 64:] = 0
    kwargs = dict(
        pair_preferences=[[0, -1, 1], [1, 0, 1], [-1, -1, 0]], source_indices=[0, 1, 2], feather_radius=(2, 4)
    )
    full = crop_sharpness_fusion(views, valid, **kwargs)
    np.testing.assert_array_equal(full[:, :60], views[1, :, :60])
    np.testing.assert_array_equal(full[:, 68:], views[0, :, 68:])
    parts = []
    for start, stop in [(0, 48), (48, 96)]:
        lo, hi = max(0, start - 4), min(96, stop + 4)
        part = crop_sharpness_fusion(views[:, :, lo:hi], valid[:, :, lo:hi], **kwargs)
        parts.append(part[:, start - lo : stop - lo])
    np.testing.assert_allclose(np.concatenate(parts, axis=1), full, rtol=1e-6)


def test_overlap_crops_keep_native_spacing_inside_both_sources():
    first = dict(origin=dict(z=0, y=0, x=0), spacing=dict(z=2, y=0.3, x=0.3), shape=dict(z=64, y=512, x=512))
    second = first | dict(origin=dict(z=0, y=0, x=80))
    spacing = np.array([2, 0.3, 0.3])
    boxes = overlap_crop_boxes(first, second, spacing)
    assert len(boxes) == 3
    for box in boxes:
        np.testing.assert_array_equal(list(box["spacing"].values()), spacing)
        low = np.array(list(box["origin"].values()))
        high = low + (np.array(list(box["shape"].values())) - 1) * spacing
        assert np.all(low >= np.array([0, 0, 80]) - 1e-6)
        assert np.all(high <= np.array([126, 153.3, 153.3]) + 1e-6)
