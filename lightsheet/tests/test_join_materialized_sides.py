from __future__ import annotations

import json
from pathlib import Path

import pytest

from squisher_lightsheet.tile_quadrant_fusion import join_materialized_sides

GRID = {"core_shape_zyx": [480, 480, 480], "window_shape_zyx": [528, 528, 528], "level_factor_zyx": [1, 1, 1]}


def _side(
    root: Path,
    name: str,
    tiles: list[str],
    *,
    fixed_fused: str = "/fused/561.ome.zarr",
    status: str = "complete",
    source_channel: int = 2,
) -> Path:
    output = root / name
    (output / "materialized_tiles").mkdir(parents=True)
    records = []
    for tile in tiles:
        path = output / "materialized_tiles" / tile
        path.mkdir()
        records.append({"tile": tile, "path": str(path), "source_view": name})
    common = {"fixed_fused": fixed_fused, "materialization_grid": GRID, "source_channel": source_channel}
    (output / "fused_fixed_materialized_chunks.positions.json").write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.fused_fixed_overlapping_materialized_positions.v1",
                "materialization_grid": GRID,
                "tiles": [record | {"translation_um": {"z": 0, "y": 0, "x": 0}} for record in records],
            }
        )
    )
    (output / "fused_fixed_materialized_chunks.registration.json").write_text(
        json.dumps(
            common
            | {
                "artifact_type": "squisher_lightsheet.fused_fixed_overlapping_materialized_registration.v1",
                "input_dir": str(output / "materialized_tiles"),
                "tiles": records,
            }
        )
    )
    (output / "fused_fixed_materialized_chunks.summary.json").write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.fused_fixed_overlapping_materialization_summary.v1",
                "status": status,
                "source_channel": source_channel,
                "output_codec": "jpegxr",
                "materialization_grid": GRID,
                "tile_count": len(tiles),
            }
        )
    )
    return output


def test_join_materialized_sides_concatenates_disjoint_complete_sides(tmp_path: Path) -> None:
    left = _side(tmp_path, "CL", ["cl.a.ome.zarr", "cl.b.ome.zarr"])
    right = _side(tmp_path, "CR", ["cr.a.ome.zarr"])

    outputs = join_materialized_sides(side_output_dirs=[left, right], output_dir=tmp_path / "joined")

    registration = json.loads(outputs["registration"].read_text())
    positions = json.loads(outputs["position"].read_text())
    assert [tile["tile"] for tile in registration["tiles"]] == ["cl.a.ome.zarr", "cl.b.ome.zarr", "cr.a.ome.zarr"]
    assert [tile["tile"] for tile in positions["tiles"]] == ["cl.a.ome.zarr", "cl.b.ome.zarr", "cr.a.ome.zarr"]
    assert registration["tiles"][2]["path"] == str(right / "materialized_tiles" / "cr.a.ome.zarr")
    assert registration["fixed_fused"] == "/fused/561.ome.zarr"
    assert registration["joined_sides"] == [str(left.resolve()), str(right.resolve())]


def test_join_materialized_sides_rejects_duplicate_tiles(tmp_path: Path) -> None:
    left = _side(tmp_path, "CL", ["same.ome.zarr"])
    right = _side(tmp_path, "CR", ["same.ome.zarr"])

    with pytest.raises(ValueError, match="duplicate"):
        join_materialized_sides(side_output_dirs=[left, right], output_dir=tmp_path / "joined")


def test_join_materialized_sides_rejects_different_fixed_grids(tmp_path: Path) -> None:
    left = _side(tmp_path, "CL", ["cl.a.ome.zarr"])
    right = _side(tmp_path, "CR", ["cr.a.ome.zarr"], fixed_fused="/fused/other.ome.zarr")

    with pytest.raises(ValueError, match="fixed_fused"):
        join_materialized_sides(side_output_dirs=[left, right], output_dir=tmp_path / "joined")


def test_join_materialized_sides_rejects_incomplete_side(tmp_path: Path) -> None:
    left = _side(tmp_path, "CL", ["cl.a.ome.zarr"])
    right = _side(tmp_path, "CR", ["cr.a.ome.zarr"], status="in_progress")

    with pytest.raises(ValueError, match="complete"):
        join_materialized_sides(side_output_dirs=[left, right], output_dir=tmp_path / "joined")


def test_join_materialized_sides_records_each_side_s_source_channel(tmp_path: Path) -> None:
    # 229Ptprz1: 514 is channel 2 of the CL multichannel acquisition and channel 0 of the separate CR acquisition.
    left = _side(tmp_path, "CL", ["cl.a.ome.zarr"], source_channel=2)
    right = _side(tmp_path, "CR", ["cr.a.ome.zarr"], source_channel=0)

    outputs = join_materialized_sides(side_output_dirs=[left, right], output_dir=tmp_path / "joined")

    summary = json.loads(outputs["summary"].read_text())
    assert "source_channel" not in summary
    assert summary["source_channels"] == {str(left.resolve()): 2, str(right.resolve()): 0}
