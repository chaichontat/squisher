import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from squisher_lightsheet import method8_stitch_register as reg
from squisher_lightsheet import overlap_screen, stitch_cli


@pytest.fixture
def tiff_tiles(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    records = []
    for index in range(2):
        path = tiles / f"tile.{index:03}.ome.tif"
        with tifffile.TiffWriter(path, ome=True) as writer:
            data = np.full((2, 128, 256, 256), 7, dtype=np.uint16)
            writer.write(
                data,
                subifds=2,
                photometric="minisblack",
                tile=(64, 64),
                compression="zlib",
                metadata={"axes": "CZYX"},
            )
            writer.write(
                data[:, :, ::2, ::2],
                subfiletype=1,
                photometric="minisblack",
                tile=(64, 64),
                compression="zlib",
            )
            reduced = np.full((2, 128, 64, 64), 201, dtype=np.uint16)
            reduced[1] = 199
            writer.write(reduced, subfiletype=1, photometric="minisblack", tile=(64, 64), compression="zlib")
        records.append(
            {
                "tile": path.name,
                "translation_um": {"z": 0, "y": 0, "x": index * 57.6},
                "scale_um": {"z": 1.5, "y": 0.3, "x": 0.3},
            }
        )
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"units": "micrometer", "tiles": records}))
    return positions, tiles


def test_registration_reads_declared_tiff_level_and_physical_scale(tiff_tiles):
    positions, tiles = tiff_tiles
    loaded = reg._load_tiles(positions, tiles, channel=0, level=2)
    tile = loaded["000"]
    assert tile.level == 2
    assert tile.tile_name == "tile.000.ome.tif"
    np.testing.assert_allclose(tile.spacing_um_zyx, [1.5, 1.2, 1.2])
    np.testing.assert_array_equal(tile.shape_zyx, [128, 64, 64])
    crop = reg._read_tile_crop(tile, (slice(1, 3), slice(2, 5), slice(3, 7)))
    np.testing.assert_array_equal(crop, np.full((2, 3, 4), 201, dtype=np.float32))
    payload, names, indexes, spacing, solver_tiles = reg._load_position_tiles(
        positions, tiles, channel=0, level=2
    )
    np.testing.assert_allclose(spacing, [1.5, 1.2, 1.2])
    assert payload["tiles"][0]["scale_um"]["x"] == 0.3
    assert solver_tiles[0].shape == (128, 64, 64)
    with pytest.raises(ValueError, match="level"):
        reg._load_tiles(positions, tiles, channel=0, level=3)


def test_level2_screen_applies_reviewed_threshold_per_channel(tiff_tiles, tmp_path):
    positions, tiles = tiff_tiles
    for channel, accepted in [(0, 2), (1, 0)]:
        output = tmp_path / f"screen{channel}.json"
        overlap_screen.screen_level2_overlaps(
            position_json=positions,
            zarr_dir=tiles,
            output=output,
            threshold=200,
            level=2,
            registration_level=2,
            channel=channel,
            z_chunks=2,
        )
        result = json.loads(output.read_text())
        assert result["accepted_unit_count"] == accepted
        assert result["settings"]["registration_level"] == 2
        assert result["settings"]["threshold"] == 200


def test_register_cli_passes_level_and_reviewed_threshold(tmp_path, monkeypatch):
    positions = tmp_path / "positions.json"
    positions.write_text("{}")
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            **{
                key: tmp_path / key
                for key in [
                    "threshold_record",
                    "measurement_summary",
                    "optimized_positions",
                    "diagnostics",
                    "constraints_jsonl",
                    "tile_corrections",
                    "canonical_positions",
                    "registration_json",
                ]
            }
        )

    monkeypatch.setattr(stitch_cli, "run_registration_workflow", run)
    result = CliRunner().invoke(
        stitch_cli.app,
        [
            "register",
            "--position-json",
            str(positions),
            "--zarr-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--threshold",
            "200",
            "--level",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["level"] == 2
    assert captured["threshold"] == 200
