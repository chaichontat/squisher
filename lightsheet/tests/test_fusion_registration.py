import sys

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


def test_fusion_uses_explicit_registration_for_multitrack_input(tmp_path, monkeypatch):
    positions = tmp_path / "positions.json"
    positions.write_text("{}")
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    tile = legacy.TileMetadata(
        path=tmp_path / "tile.000.ome.tif",
        shape=(2, 4, 8, 8),
        axes="CZYX",
        spacing={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        channels=("561", "488"),
        tracks=(
            legacy.TrackMetadata(slug="track0", track_id="0", channels=(0,), channel_names=("561",)),
            legacy.TrackMetadata(slug="track1", track_id="1", channels=(1,), channel_names=("488",)),
        ),
    )
    monkeypatch.setattr(legacy, "read_position_input_tiles", lambda *args, **kwargs: [tile])
    calls = []
    monkeypatch.setattr(legacy, "run_stitch_once", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stitch",
            str(tmp_path),
            "--position-input",
            str(positions),
            "--registration-input",
            str(registration),
            "--channels",
            "0",
        ],
    )
    assert legacy.main() == 0
    assert len(calls) == 1
    assert calls[0]["registration_input"] == registration
