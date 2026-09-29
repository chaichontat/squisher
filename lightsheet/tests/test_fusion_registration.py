import json
import sys

import numpy as np
import pytest

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


def _registration_record(tile: legacy.TileMetadata, *, stage_x: float, affine_x: float) -> dict:
    matrix = np.eye(4)
    matrix[2, 3] = affine_x
    return {
        "tile": tile.path.name,
        "stage_translation_um": {"z": 0.0, "y": 0.0, "x": stage_x},
        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        "registered_affine": {
            "dims": ["x_in", "x_out"],
            "coords": {
                "x_in": ["z", "y", "x", "1"],
                "x_out": ["z", "y", "x", "1"],
            },
            "matrix": matrix.tolist(),
        },
    }


def test_registration_accepts_prod2_stage_position_convention(tmp_path):
    tile = legacy.TileMetadata(
        path=tmp_path / "tile.000.ome.zarr",
        shape=(1, 4, 8, 8),
        axes="CZYX",
        spacing={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 20.0},
        channels=("561",),
        tracks=(),
    )
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps({"tiles": [_registration_record(tile, stage_x=20.0, affine_x=7.0)]})
    )

    (param,) = legacy.load_registration_params(registration, [tile])

    assert float(param.data[2, 3]) == 7.0


def test_registration_matches_canonical_tile_by_source_path_basename(tmp_path):
    tile = legacy.TileMetadata(
        path=tmp_path / "moving.000.ome.zarr",
        shape=(1, 4, 8, 8),
        axes="CZYX",
        spacing={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 20.0},
        channels=("514",),
        tracks=(),
    )
    record = _registration_record(tile, stage_x=20.0, affine_x=7.0)
    record["tile"] = "canonical-reference.000.ome.zarr"
    record["path"] = str(tile.path)
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"tiles": [record]}))

    (param,) = legacy.load_registration_params(registration, [tile])

    assert float(param.data[2, 3]) == 7.0


def test_registration_rejects_ambiguous_tile_and_source_path_identities(tmp_path):
    tile = legacy.TileMetadata(
        path=tmp_path / "moving.000.ome.zarr",
        shape=(1, 4, 8, 8),
        axes="CZYX",
        spacing={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 20.0},
        channels=("514",),
        tracks=(),
    )
    exact = _registration_record(tile, stage_x=20.0, affine_x=7.0)
    exact["path"] = str(tmp_path / "other.ome.zarr")
    fallback = _registration_record(tile, stage_x=20.0, affine_x=8.0)
    fallback["tile"] = "canonical-reference.000.ome.zarr"
    fallback["path"] = str(tile.path)
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"tiles": [exact, fallback]}))

    with pytest.raises(ValueError, match="ambiguous registration records"):
        legacy.load_registration_params(registration, [tile])


def test_registration_rejects_position_with_registered_translation_baked_in(tmp_path):
    tile = legacy.TileMetadata(
        path=tmp_path / "tile.000.ome.zarr",
        shape=(1, 4, 8, 8),
        axes="CZYX",
        spacing={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 27.0},
        channels=("561",),
        tracks=(),
    )
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps({"tiles": [_registration_record(tile, stage_x=20.0, affine_x=7.0)]})
    )

    with pytest.raises(ValueError, match="position translation differs from registration stage_translation_um"):
        legacy.load_registration_params(registration, [tile])
