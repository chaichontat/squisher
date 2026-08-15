from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

import squisher_lightsheet.method8_stitch_register as stitch_register
import squisher_lightsheet.overlap_screen as overlap_screen
from squisher_lightsheet.method8_stitch_register import TileInfo


def _write_multiscale_tile(path: Path, *, x_um: float) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    shapes = ((1, 128, 256, 256), (1, 128, 128, 128), (1, 128, 64, 64))
    xy_scales = (0.3, 0.6, 1.2)
    datasets = []
    for level, (shape, xy_scale) in enumerate(zip(shapes, xy_scales, strict=True)):
        root.create_array(
            str(level),
            shape=shape,
            chunks=(1, 1, shape[-2], shape[-1]),
            dtype="uint16",
            dimension_names=("c", "z", "y", "x"),
            fill_value=0,
        )
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 0.6, xy_scale, xy_scale]},
                    {"type": "translation", "translation": [0.0, 0.0, 0.0, x_um]},
                ],
            }
        )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets,
            }
        ],
    }


def _tile(tile_id: str, x_um: float) -> TileInfo:
    return TileInfo(
        tile_id=tile_id,
        tile_name=f"sample.{tile_id}.ome.zarr",
        path=Path(f"sample.{tile_id}.ome.zarr"),
        start_um_zyx=np.asarray([0.0, 0.0, x_um]),
        spacing_um_zyx=np.asarray([0.6, 0.3, 0.3]),
        shape_zyx=np.asarray([128, 256, 256]),
        channel=0,
    )


def test_screen_level2_overlaps_writes_every_pair_chunk_decision(tmp_path) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    names = ["sample.000.ome.zarr", "sample.001.ome.zarr"]
    for name, x_um in zip(names, (0.0, 57.6), strict=True):
        _write_multiscale_tile(zarr_dir / name, x_um=x_um)
        root = zarr.open_group(str(zarr_dir / name), mode="r+")
        root["2"][0, 31] = 2000
    positions = tmp_path / "positions.json"
    positions.write_text(
        json.dumps(
            {
                "units": "micrometer",
                "tiles": [
                    {
                        "tile": name,
                        "translation_um": {"z": 0.0, "y": 0.0, "x": x_um},
                        "scale_um": {"z": 0.6, "y": 0.3, "x": 0.3},
                    }
                    for name, x_um in zip(names, (0.0, 57.6), strict=True)
                ],
            }
        )
        + "\n"
    )
    output = tmp_path / "screen.json"

    overlap_screen.screen_level2_overlaps(
        position_json=positions,
        zarr_dir=zarr_dir,
        output=output,
        threshold=1542.0,
        level=2,
        channel=0,
        z_chunks=2,
        min_foreground_pixels=256,
        min_foreground_fraction=0.05,
    )

    payload = json.loads(output.read_text())
    assert payload["pair_count"] == 1
    assert payload["unit_count"] == 2
    assert payload["accepted_unit_count"] == 1
    assert [sample["status"] for sample in payload["pairs"][0]["samples"]] == [
        "accepted",
        "low_content",
    ]


def test_measurement_skips_level0_read_for_low_content_units(tmp_path, monkeypatch) -> None:
    position_json = tmp_path / "positions.json"
    position_json.write_text("{}\n")
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    screen = tmp_path / "screen.json"
    screen.write_text("{}\n")
    tiles = {"000": _tile("000", 0.0), "001": _tile("001", 57.6)}
    decisions = {
        ("000-001", 0): {
            "chunk": 0,
            "z_start": 0,
            "z_stop": 64,
            "status": "low_content",
            "reason": overlap_screen.LOW_CONTENT_REASON,
        },
        ("000-001", 1): {
            "chunk": 1,
            "z_start": 64,
            "z_stop": 128,
            "status": "accepted",
            "reason": None,
        },
    }
    measured_chunks: list[tuple[int, int]] = []

    monkeypatch.setattr(stitch_register, "_load_tiles", lambda *_args, **_kwargs: tiles)
    monkeypatch.setattr(stitch_register, "_all_adjacent_pairs", lambda _tiles: ["000-001"])
    monkeypatch.setattr(overlap_screen, "load_level2_screen", lambda *_args, **_kwargs: decisions)

    def fake_measure_z_chunk(**kwargs):
        measured_chunks.append((kwargs["z_start"], kwargs["z_stop"]))
        return {
            "fixed_tile": tiles["000"].tile_name,
            "moving_tile": tiles["001"].tile_name,
            "z_start": kwargs["z_start"],
            "z_stop": kwargs["z_stop"],
            "seam_axis": "x",
            "phase_shift_zyx": [0.0, 0.0, 0.0],
            "local_translation_zyx": None,
            "gradient_component_ncc_phase_mean": 0.5,
            "status": "accepted",
            "rejection_reason": None,
        }

    monkeypatch.setattr(stitch_register, "_measure_z_chunk", fake_measure_z_chunk)
    output = tmp_path / "measurements.json"

    stitch_register.measure_method8_zcoverage(
        position_json=position_json,
        zarr_dir=zarr_dir,
        output=output,
        level2_screen=screen,
        z_chunks=2,
        fixed_mask_threshold=1542.0,
        phase_recovery_shifted_crop=False,
    )

    assert measured_chunks == [(64, 128)]
    rows = json.loads(output.read_text())["rows"]
    assert rows[0]["measurement_mode"] == "level2_overlap_screen"
    assert rows[0]["rejection_reason"] == overlap_screen.LOW_CONTENT_REASON
    assert rows[1]["status"] == "accepted"


def test_load_level2_screen_fails_closed_on_missing_unit(tmp_path, monkeypatch) -> None:
    position_json = tmp_path / "positions.json"
    position_json.write_text("{}\n")
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    monkeypatch.setattr(overlap_screen, "registration_input_fingerprint", lambda *_args: "fingerprint")
    screen = tmp_path / "screen.json"
    screen.write_text(
        json.dumps(
            {
                "artifact_type": overlap_screen.ARTIFACT_TYPE,
                "position_json": str(position_json),
                "zarr_dir": str(zarr_dir),
                "input_fingerprint": "fingerprint",
                "settings": {"level": 2, "channel": 0, "threshold": 1542.0, "z_chunks": 2},
                "pairs": [
                    {
                        "pair": "000-001",
                        "samples": [
                            {
                                "chunk": 0,
                                "status": "accepted",
                                "reason": None,
                            }
                        ],
                    }
                ],
            }
        )
        + "\n"
    )

    try:
        overlap_screen.load_level2_screen(
            screen,
            position_json=position_json,
            zarr_dir=zarr_dir,
            expected_pairs=["000-001"],
            z_chunks=2,
            channel=0,
            threshold=1542.0,
        )
    except ValueError as error:
        assert "does not contain 2 samples" in str(error)
    else:
        raise AssertionError("incomplete screen must fail closed")
