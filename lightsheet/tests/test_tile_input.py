from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet import tile_input


@pytest.mark.parametrize(
    ("axes", "shape", "expected"),
    [
        ("ZYX", (3, 4, 5), (3, 4, 5)),
        ("CZYX", (2, 3, 4, 5), (3, 4, 5)),
        ("ZCYX", (3, 2, 4, 5), (3, 4, 5)),
    ],
)
def test_spatial_shape_zyx_uses_explicit_axes(axes, shape, expected) -> None:
    assert tile_input.spatial_shape_zyx(shape, axes) == expected


def test_source_axes_handles_pyramid_level_without_channel_axis() -> None:
    assert tile_input.source_axes("ZCYX", (3, 4, 5)) == "ZYX"


def test_channel_view_selects_supported_axes() -> None:
    zyx = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    czyx = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    zcyx = np.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)

    assert tile_input.channel_view(zyx, "ZYX", 0, path=Path("tile.tif")) is zyx
    np.testing.assert_array_equal(
        tile_input.channel_view(czyx, "CZYX", 1, path=Path("tile.tif")), czyx[1]
    )
    np.testing.assert_array_equal(
        tile_input.channel_view(zcyx, "ZCYX", 1, path=Path("tile.tif")), zcyx[:, 1]
    )
