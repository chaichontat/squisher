from __future__ import annotations

from pathlib import Path

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
