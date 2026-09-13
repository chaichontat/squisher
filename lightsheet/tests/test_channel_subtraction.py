from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from squisher_lightsheet import channel_subtraction
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


class DummyStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ConstantCorrection:
    def __init__(self, value: float) -> None:
        self.value = value

    def block(self, *, z_slice, y_slice, x_slice):
        return np.full(
            (
                z_slice.stop - z_slice.start,
                y_slice.stop - y_slice.start,
                x_slice.stop - x_slice.start,
            ),
            self.value,
            dtype=np.float32,
        )


def test_subtraction_applies_target_and_reference_corrections_before_subtracting(monkeypatch) -> None:
    target = np.full((3, 4, 5), 20, dtype=np.uint16)
    reference = np.full((3, 4, 5), 4, dtype=np.uint16)

    def fake_subtract(target_slab, reference_halo, **kwargs):
        z_stop = reference_halo.shape[0] - kwargs["halo_after"]
        return target_slab - reference_halo[kwargs["halo_before"] : z_stop]

    monkeypatch.setattr(channel_subtraction, "_subtract_slab_gpu", fake_subtract)
    actual = channel_subtraction.subtract_spillover_array_gpu(
        target,
        reference,
        reference_shift_zyx_px=(0.0, 0.0, 0.0),
        alpha=1.0,
        beta=0.0,
        target_background=0.0,
        reference_background=0.0,
        crop_yx_px=0,
        z_chunk=2,
        output_dtype=np.dtype(np.float32),
        target_correction=ConstantCorrection(2.0),
        reference_correction=ConstantCorrection(3.0),
    )

    np.testing.assert_array_equal(actual, 28.0)


def test_corrected_zarr_preserves_single_channel_spatial_metadata(tmp_path: Path) -> None:
    output = tmp_path / "corrected.ome.zarr"
    data = np.ones((3, 4, 5), dtype=np.uint16)

    channel_subtraction._write_ome_zarr_czyx(
        output,
        data,
        translation_um={"z": 10.0, "y": 20.0, "x": 30.0},
        scale_um={"z": 1.0, "y": 0.5, "x": 0.25},
        channel_name="514",
        zstd_level=3,
        chunk_shape_zyx=(2, 3, 4),
    )

    root = zarr.open_group(str(output), mode="r")
    level0 = root["0"]
    assert root.attrs["squisher_complete"] is True
    assert root.attrs["omero"]["channels"] == [{"label": "514"}]
    assert level0.shape == (1, 3, 4, 5)
    assert level0.chunks == (1, 2, 3, 4)
    assert level0.metadata.dimension_names == ("c", "z", "y", "x")
    np.testing.assert_array_equal(level0[0], data)
    parsed = legacy.parse_ome_metadata(output)
    assert parsed.axes == "CZYX"
    assert parsed.translation == {"z": 10.0, "y": 20.0, "x": 30.0}
    assert parsed.spacing == {"z": 1.0, "y": 0.5, "x": 0.25}


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
        tracks=(
            legacy.TrackMetadata(
                slug="track0", track_id="all", channels=(0, 1, 2, 3), channel_names=("0", "1", "2", "3")
            ),
        ),
    )
    store = DummyStore()

    monkeypatch.setattr(channel_subtraction.legacy, "read_position_input_tiles", lambda _path: [tile])
    monkeypatch.setattr(
        channel_subtraction.legacy, "open_tile_array", lambda *_args, **_kwargs: (data, store)
    )

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
    assert payload["tiles"][0]["axes"] == "CZYX"
    assert payload["tiles"][0]["shape"] == [1, 3, 4, 4]
    output_tile = Path(payload["tiles"][0]["path"])
    assert output_tile.exists()
    assert store.closed is True

    root = zarr.open_group(str(output_tile), mode="r")
    output = np.asarray(root["0"][0])
    assert output.shape == (3, 4, 4)
    assert np.all(output == 61)
    assert output_tile.name.endswith(".ome.zarr")
    assert root.attrs["squisher_complete"] is True

    parsed = legacy.parse_ome_metadata(output_tile)
    assert parsed.axes == "CZYX"
    assert parsed.translation == {"z": 10.0, "y": 20.5, "x": 30.25}

    assert result.registration_output is not None
    registration_payload = json.loads(result.registration_output.read_text())
    assert registration_payload["input_dir"].endswith("/subtracted/tiles")
    assert registration_payload["tiles"][0]["tile"] == output_tile.name
    assert registration_payload["tiles"][0]["path"] == str(output_tile)
    assert registration_payload["tiles"][0]["axes"] == "CZYX"
    assert registration_payload["tiles"][0]["shape"] == [1, 3, 4, 4]
    assert registration_payload["tiles"][0]["source_tile"] == source_tile.name
    assert registration_payload["tiles"][0]["stage_translation_um"] == {"z": 10.0, "y": 20.5, "x": 30.25}
    assert registration_payload["tiles"][0]["registered_affine"] == {"matrix": [[1, 0, 0, 0]]}
