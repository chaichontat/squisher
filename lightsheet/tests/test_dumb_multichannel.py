from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import tifffile
from typer.testing import CliRunner

from squisher_lightsheet import cli
from squisher_lightsheet import ome_metadata_dumb_stitch as ds


def test_multichannel_tiff_preserves_order_and_intensities(tmp_path, monkeypatch):
    tile = ds.TileMetadata(
        path=tmp_path / "tile.ome.tif",
        name="tile.ome.tif",
        axes="CZYX",
        shape=(4, 1, 2, 3),
        spacing_um_zyx=(1.0, 2.0, 3.0),
        translation_um_zyx=(0.0, 0.0, 0.0),
    )
    channels = (3, 0, 2, 1)
    planes = {c: np.array([[0, 17, 1024], [4096, 32000 + c, 65535]], dtype=np.float32) for c in channels}
    profile = ds.BasicProfile(
        flatfield=np.full((2, 3), 2, dtype=np.float32),
        darkfield=np.full((2, 3), 1, dtype=np.float32),
        flatfield_path=Path("flat.tif"),
        darkfield_path=Path("dark.tif"),
    )
    monkeypatch.setattr(ds, "tile_paths_from_dir", lambda _: [tile.path])
    monkeypatch.setattr(ds, "read_tile_metadata", lambda *a, **k: tile)
    monkeypatch.setattr(ds, "_read_planes", lambda *a, **k: planes)
    monkeypatch.setattr(ds, "load_basic_profile", lambda *a, **k: profile)
    result = ds.render_ome_metadata_dumb_stitch(
        input_dirs_by_view={"L": tmp_path},
        output_dir=tmp_path / "out",
        channels=channels,
        basic_dir=tmp_path,
        write_tiff=True,
        tiff_layout="channels",
        output_prefix="q",
    )
    outputs = [p for p in result.output_paths if p.name.endswith(".ome.tif")]
    assert len(outputs) == 2
    for case, dtype in (("raw", np.uint16), ("basic", np.float32)):
        path = next(p for p in outputs if f"_{case}_" in p.name)
        expected = np.stack([planes[c] if case == "raw" else (planes[c] - 1) / 2 for c in channels]).astype(
            dtype
        )
        with tifffile.TiffFile(path) as tif:
            assert tif.series[0].axes == "CYX"
            assert tif.series[0].dtype == dtype
            assert all(page.keyframe.is_tiled for page in tif.pages)
            np.testing.assert_array_equal(tif.asarray(), expected)
            pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
            assert float(pixels.attrib["PhysicalSizeY"]) == 2
            assert float(pixels.attrib["PhysicalSizeX"]) == 3
            assert [c.attrib["Name"] for c in pixels.findall("{*}Channel")] == [f"ch{c}" for c in channels]


def test_cli_accepts_multichannel_tiff_layout(tmp_path, monkeypatch):
    captured = {}

    def render(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            manifest_path=tmp_path / "manifest.json",
            contact_sheet_path=tmp_path / "contact.png",
            output_paths=[],
        )

    monkeypatch.setattr(cli, "render_ome_metadata_dumb_stitch", render)
    result = CliRunner().invoke(
        cli.app,
        [
            "ome-metadata-dumb-stitch",
            "--input-dir",
            f"L={tmp_path}",
            "--channels",
            "0,1,2,3",
            "--write-tiff",
            "--tiff-layout",
            "channels",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["tiff_layout"] == "channels"
    assert captured["channels"] == (0, 1, 2, 3)
