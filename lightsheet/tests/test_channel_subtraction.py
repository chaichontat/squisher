from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from squisher_lightsheet import channel_subtraction
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


class DummyStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("axes", ["CZYX", "ZCYX"])
def test_subtract_channel_tiles_writes_cropped_parseable_position_file(monkeypatch, tmp_path, axes) -> None:
    source_tile = tmp_path / "tile.ome.tif"
    plane_count = 4 * 3
    tifffile.imwrite(
        source_tile,
        np.zeros((4, 3, 6, 6), dtype=np.uint16),
        ome=True,
        metadata={
            "axes": "CZYX",
            "PhysicalSizeX": 0.25,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeZ": 1.0,
            "Plane": {
                "PositionX": [30.0] * plane_count,
                "PositionY": [20.0] * plane_count,
                "PositionZ": [10.0 + (index % 3) for index in range(plane_count)],
            },
        },
    )
    position_input = tmp_path / "input.positions.json"
    position_input.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.position.v1",
                "units": "micrometer",
                "tiles": [
                    {
                        "tile": source_tile.name,
                        "side": "L",
                        "path": str(source_tile),
                        "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                        "scale_um": {"z": 1.0, "y": 0.5, "x": 0.25},
                    }
                ],
            }
        )
        + "\n"
    )
    registration_input = tmp_path / "input.registration.json"
    registration_input.write_text(
        json.dumps(
            {
                "input_dir": str(tmp_path),
                "tiles": [
                    {
                        "tile": source_tile.name,
                        "source_view": "L",
                        "stage_translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                        "stage_scale_um": {"z": 1.0, "y": 0.5, "x": 0.25},
                        "registered_affine": {"matrix": [[1, 0, 0, 0]]},
                    }
                ],
            }
        )
        + "\n"
    )
    data = np.zeros((4, 3, 6, 6), dtype=np.uint16)
    data[2] = 100
    data[3] = 20
    if axes == "ZCYX":
        data = data.transpose(1, 0, 2, 3)
    tile = legacy.TileMetadata(
        path=source_tile,
        shape=data.shape,
        axes=axes,
        spacing={"z": 1.0, "y": 0.5, "x": 0.25},
        translation={"z": 10.0, "y": 20.0, "x": 30.0},
        channels=("0", "1", "2", "3"),
        tracks=(legacy.TrackMetadata(slug="track0", track_id="all", channels=(0, 1, 2, 3), channel_names=("0", "1", "2", "3")),),
    )
    store = DummyStore()

    monkeypatch.setattr(channel_subtraction.legacy, "read_position_input_tiles", lambda _path: [tile])
    monkeypatch.setattr(channel_subtraction.legacy, "open_tile_array", lambda *_args, **_kwargs: (data, store))

    def fake_subtract(target, reference, **kwargs):
        corrected = (
            target.astype(np.float32)
            - kwargs["target_background"]
            - kwargs["alpha"] * np.maximum(reference.astype(np.float32) - kwargs["reference_background"], 0)
            - kwargs["beta"]
        )
        corrected = corrected[:, 1:-1, 1:-1]
        return np.rint(np.maximum(corrected, 0)).astype(np.uint16)

    monkeypatch.setattr(channel_subtraction, "subtract_spillover_array_gpu", fake_subtract)

    result = channel_subtraction.subtract_channel_tiles(
        position_input=position_input,
        output_dir=tmp_path / "subtracted",
        registration_input=registration_input,
        target_channel=2,
        reference_channel=3,
        source_level=0,
        reference_shift_zyx_px=(0.0, 0.0, 0.0),
        alpha=2.0,
        beta=-1.0,
        target_background=10.0,
        reference_background=5.0,
        crop_yx_px=1,
    )

    payload = json.loads(result.position_output.read_text())
    assert payload["tiles"][0]["side"] == "L"
    assert payload["tiles"][0]["translation_um"] == {"z": 10.0, "y": 20.5, "x": 30.25}
    assert payload["tiles"][0]["scale_um"] == {"z": 1.0, "y": 0.5, "x": 0.25}
    output_tile = Path(payload["tiles"][0]["path"])
    assert output_tile.exists()
    assert store.closed is True

    output = tifffile.imread(output_tile)
    assert output.shape == (3, 4, 4)
    assert np.all(output == 61)
    with tifffile.TiffFile(output_tile) as tif:
        assert tif.pages[0].compression.value == channel_subtraction.DEFAULT_COMPRESSION
        assert tif.ome_metadata is not None
        assert f'SizeX="{output.shape[2]}"' in tif.ome_metadata
        assert 'SizeC="1"' in tif.ome_metadata

    parsed = legacy.parse_ome_metadata(output_tile)
    assert parsed.axes == "ZYX"
    assert parsed.translation == {"z": 10.0, "y": 20.5, "x": 30.25}

    assert result.registration_output is not None
    registration_payload = json.loads(result.registration_output.read_text())
    assert registration_payload["input_dir"].endswith("/subtracted/tiles")
    assert registration_payload["tiles"][0]["tile"] == output_tile.name
    assert registration_payload["tiles"][0]["source_tile"] == source_tile.name
    assert registration_payload["tiles"][0]["stage_translation_um"] == {"z": 10.0, "y": 20.5, "x": 30.25}
    assert registration_payload["tiles"][0]["registered_affine"] == {"matrix": [[1, 0, 0, 0]]}
