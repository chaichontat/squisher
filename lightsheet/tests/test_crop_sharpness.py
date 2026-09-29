"""Native crop preferences preserve fine detail and stable overlap ownership."""

import json

import numpy as np
import squisher_lightsheet.crop_sharpness as crop_sharpness
from scipy.ndimage import gaussian_filter
from squisher_lightsheet.crop_sharpness import native_crop_scores, crop_sharpness_fusion, overlap_crop_boxes


def test_native_crops_detect_detail_that_striding_would_erase():
    x = np.arange(128)
    sharp = np.broadcast_to(100 + 20 * np.sin(np.pi * x / 2), (64, 128)).astype(np.float32)
    blurred = 2 * gaussian_filter(sharp, 2)
    scores = native_crop_scores(np.stack([sharp, blurred]), (2, 2))
    assert scores[0] > 10 * scores[1]


def test_native_crops_without_shared_support_are_unscored():
    views = np.full((2, 8, 8), np.nan, dtype=np.float32)
    assert native_crop_scores(views, (1, 1)) is None


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


def test_crop_preference_reduction_does_not_allocate_a_full_view_stack(monkeypatch):
    """The final weighted reduction must scale with one view, not all views."""
    views = np.ones((3, 4, 5), dtype=np.float32)
    valid = np.ones_like(views)
    original_zeros_like = np.zeros_like
    allocated_shapes = []

    def recording_zeros_like(array, *args, **kwargs):
        allocated_shapes.append(array.shape)
        return original_zeros_like(array, *args, **kwargs)

    monkeypatch.setattr(crop_sharpness.np, "zeros_like", recording_zeros_like)
    crop_sharpness_fusion(
        views,
        valid,
        pair_preferences=[[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
        source_indices=[0, 1, 2],
        feather_radius=(1, 1),
    )

    assert views.shape[1:] in allocated_shapes
    assert views.shape not in allocated_shapes


def test_registered_view_correction_is_applied_after_ownership(tmp_path):
    artifact = tmp_path / "registered-view-correction.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "squisher.registered-view-correction",
                "coordinate": "registered-physical-zyx-um",
                "moving_view": "R",
                "origin_zyx_um": [0, 0, 0],
                "spacing_zyx_um": [1, 1, 1],
                "shape_zyx": [2, 2, 2],
                "intercept_log_ratio": float(np.log(2)),
                "deviation_log_ratio": np.zeros((2, 2, 2)).tolist(),
                "common_scale": 0.5,
            }
        )
    )
    views = np.stack([np.full((2, 3, 4), 10, dtype=np.float32), np.full((2, 3, 4), 20, dtype=np.float32)])
    valid = np.ones_like(views)

    fused = crop_sharpness_fusion(
        views,
        valid,
        pair_preferences=[[0, -1], [1, 0]],
        source_indices=[0, 1],
        source_views=["L", "R"],
        feather_radius=(1, 1, 1),
        registered_view_correction=str(artifact),
        output_origin_zyx_um=(10, 10, 10),
        output_spacing_zyx_um=(1, 1, 1),
    )

    np.testing.assert_allclose(fused, 5)


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
