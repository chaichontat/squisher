from dataclasses import replace
from pathlib import Path

import pytest
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


def _tiles():
    tiles = []
    for name, channels in (("tl", ("561", "638", "514", "488")), ("tr", ("561", "638"))):
        tiles.append(legacy.TileMetadata(
            path=Path(f"/data/{name}.ome.zarr"), shape=(len(channels), 10, 8, 8), axes="CZYX",
            spacing={"z": 1.8, "y": .3, "x": .3}, translation={"z": 0, "y": 0, "x": 0},
            channels=channels, tracks=(),
        ))
    return tiles


def test_explicit_shared_channel_allows_different_acquisition_channel_counts():
    tiles = _tiles()
    legacy.validate_tiles_metadata(tiles, require_same_shape=False, selected_channels=(0, 1))
    with pytest.raises(ValueError, match="channel/y/x"):
        legacy.validate_tiles_metadata(tiles, require_same_shape=False)


@pytest.mark.parametrize("mutate, message", [
    (lambda tile: tile, "missing selected channel"),
    (lambda tile: replace(tile, channels=("514", "638")), "selected channel metadata"),
    (lambda tile: replace(tile, shape=(2, 10, 9, 8)), "y/x shape"),
])
def test_selected_channel_retains_required_geometry_and_label_contracts(mutate, message):
    tiles = _tiles()
    tiles[1] = mutate(tiles[1])
    selected = (2,) if message == "missing selected channel" else (0,)
    with pytest.raises(ValueError, match=message):
        legacy.validate_tiles_metadata(tiles, require_same_shape=False, selected_channels=selected)
