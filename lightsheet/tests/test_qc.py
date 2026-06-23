from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from squisher_lightsheet import qc
from squisher_lightsheet.qc import side_by_tile


def test_side_lookup_uses_resolved_paths_for_duplicate_basenames() -> None:
    payload = {
        "tiles": [
            {"tile": "tile001.ome.tif", "side": "L", "path": "/sample/TL/tile001.ome.tif"},
            {"tile": "tile001.ome.tif", "side": "R", "path": "/sample/TR/tile001.ome.tif"},
        ]
    }

    assert side_by_tile(payload) == {
        str(Path("/sample/TL/tile001.ome.tif").resolve()): "L",
        str(Path("/sample/TR/tile001.ome.tif").resolve()): "R",
    }


def test_place_global_projections_uses_max_compositing() -> None:
    projections = qc.empty_projection_canvases(np.asarray([3, 4, 5]))
    volume = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)

    qc.place_global_projections(
        projections,
        side="L",
        volume=volume,
        start_zyx=np.asarray([1, 1, 1]),
    )

    assert np.array_equal(projections["L"]["xy"][1:3, 1:4], volume.max(axis=0))
    assert np.array_equal(projections["L"]["xz"][1:3, 1:4], volume.max(axis=1))
    assert np.array_equal(projections["L"]["yz"][1:3, 1:3], volume.max(axis=2))


def test_write_contact_sheet_stacks_images_with_titles(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    output = tmp_path / "sheet.png"
    Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(first)
    Image.fromarray(np.zeros((5, 10, 3), dtype=np.uint8)).save(second)

    qc.write_contact_sheet(output, [("first", first), ("second", second)])

    sheet = Image.open(output)
    assert sheet.size == (20, 76)
