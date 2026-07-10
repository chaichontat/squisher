from __future__ import annotations

import numpy as np

from squisher_lightsheet.track_z import (
    TrackMeasurement,
    estimate_z_shift_px,
    smooth_track_measurements,
)


def test_estimate_z_shift_ignores_xy_offset_and_recovers_z_shift() -> None:
    z, y, x = np.mgrid[:32, :48, :48]
    fixed = np.exp(-(((z - 16) ** 2) / 12.0 + ((y - 24) ** 2 + (x - 24) ** 2) / 150.0)).astype(np.float32)
    moving = np.roll(fixed, shift=-3, axis=0)
    moving = np.roll(moving, shift=5, axis=1)
    moving = np.roll(moving, shift=-4, axis=2)

    dz_px, details = estimate_z_shift_px(fixed, moving, max_shift_px=6, min_voxels=128)

    np.testing.assert_allclose(dz_px, 3.0, atol=0.25)
    assert details["corr_after"] > details["corr_before"]


def test_smooth_track_measurements_rejects_large_outlier() -> None:
    measurements = [
        TrackMeasurement(
            tile=f"tile-{index}",
            side="L",
            path=f"/tmp/tile-{index}.tif",
            tile_center_y_um=float(index * 100),
            tile_center_x_um=0.0,
            moving_track="track0",
            moving_channels=(0, 1),
            dz_px=float(value),
            dz_um=float(value),
            smoothed_dz_um=None,
            residual_dz_um=None,
            score=0.8,
            valid=True,
            outlier=False,
            reason=None,
        )
        for index, value in enumerate([1.0, 1.1, 0.9, 1.0, 10.0])
    ]

    smoothed = smooth_track_measurements(measurements, smooth_sigma_tiles=1.5, outlier_mad=3.0)

    assert smoothed[-1].outlier is True
    assert smoothed[-1].valid is False
    assert smoothed[-1].reason == "outlier"
    assert smoothed[-1].smoothed_dz_um < 2.0
