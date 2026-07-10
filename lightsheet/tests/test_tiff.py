from __future__ import annotations

import numpy as np
import pytest
import tifffile

from squisher_lightsheet.tiff import (
    choose_tiff_source_level,
    spatial_shape_array_zyx_from_axes,
    spatial_shape_zyx_from_axes,
    tiff_level_factors_zyx,
    tiff_series_level_count,
)


def test_spatial_shape_zyx_from_axes_supports_czyx_and_zyx() -> None:
    assert spatial_shape_zyx_from_axes((2, 3, 4, 5), "CZYX") == (3, 4, 5)
    assert spatial_shape_zyx_from_axes((3, 4, 5), "ZYX") == (3, 4, 5)
    np.testing.assert_array_equal(
        spatial_shape_array_zyx_from_axes((2, 3, 4, 5), "CZYX"),
        np.asarray((3, 4, 5), dtype=np.int64),
    )


def test_spatial_shape_zyx_from_axes_rejects_unsupported_axes() -> None:
    with pytest.raises(ValueError, match="Expected CZYX or ZYX axes"):
        spatial_shape_zyx_from_axes((3, 4), "YX")


def test_tiff_series_level_count_counts_single_level_series(tmp_path) -> None:
    path = tmp_path / "tile.tif"
    tifffile.imwrite(path, np.zeros((2, 3), dtype=np.uint16))

    assert tiff_series_level_count(path) == 1


def test_choose_tiff_source_level_uses_available_level_not_exceeding_desired_factor(tmp_path) -> None:
    path = tmp_path / "tile.tif"
    tifffile.imwrite(path, np.zeros((2, 3, 4), dtype=np.uint16), metadata={"axes": "ZYX"})

    assert tiff_level_factors_zyx(path)[0] == (1, 1, 1)
    source_level, source_factor = choose_tiff_source_level(path, np.asarray([4, 4, 4], dtype=np.int64))

    assert source_level == 0
    np.testing.assert_array_equal(source_factor, np.asarray([1, 1, 1], dtype=np.int64))
