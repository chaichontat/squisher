from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest
import tifffile

from squisher_lightsheet.post_basic_qc import write_post_basic_qc


def test_qc_uses_channel_geometry_source_identity_and_unit_fields(tmp_path: Path, monkeypatch) -> None:
    sources = [str(tmp_path / name) for name in ("alpha.ome.tif", "beta.ome.tif", "excluded.ome.tif")]
    (tmp_path / "tile-gains.json").write_text(
        json.dumps(
            {
                "tiles": [
                    {"source": sources[0], "gains": [1.25, 1, 1, 1, 1]},
                    {"source": sources[1], "gains": [1, 1, 1, 1, 1]},
                    {"source": sources[2], "gains": [1, 1, 1, 1, 1]},
                ]
            }
        )
    )
    # Ownership is in sampled-source order, deliberately reversed from gain-file order.
    owner = np.full((8, 12), -1, dtype=np.int32)
    owner[:, :4] = 0
    owner[:, 4:8] = 1
    channels = {}
    flatfields = {}
    for channel in (0, 2, 4):
        artifacts = {"display_range": {"values": [0, 100]}}
        for key, array in [
            ("basic_tiff", np.full((8, 12), 20, np.float32)),
            ("corrected_tiff", np.full((8, 12), 25, np.float32)),
            ("mask", np.ones((6, 10), np.float32)),
            ("owner", owner),
        ]:
            path = tmp_path / f"{key}-{channel}.tif"
            tifffile.imwrite(path, array)
            artifacts[key] = {"path": path.name}
        flatfields[channel] = np.ones((6, 10), np.float32)
        channels[str(channel)] = {
            "label": f"Dye <{channel}>",
            "coefficient": [0.0] * 5,
            "artifacts": artifacts,
            "sampling": {"sampled_sources": [sources[1], sources[0]]},
            "tile_gains_applied": channel != 4,
            "excluded_sources": [sources[2]],
            "tile_gain_unestimated_sources": [sources[1]] if channel == 0 else [],
        }
    channels["0"]["source_fields_applied"] = True
    channels["0"]["source_fields"] = [{"source": source, "coefficient": [0.0] * 5} for source in sources]
    manifest = {
        "channel_results": channels,
        "fixed_spacing_um": [1, 2, 3],
        "fixed_origin_um": [0, 100, 200],
        "fixed_z": 7,
        "fixed_z_um": 7,
        "stride": 2,
        "inputs": {},
    }
    figures = []
    original_savefig = Figure.savefig

    def capture(self, path, **kwargs):
        if Path(path).suffix == ".png" and Path(path).stem.startswith("channel_"):
            gain_ax = next(ax for ax in self.axes if ax.get_title(loc="left").startswith("(c)"))
            figures.append((gain_ax.images[0].get_array().copy(), gain_ax.images[0].get_extent()))
        return original_savefig(self, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    original_family = list(mpl.rcParams["font.family"])
    write_post_basic_qc(tmp_path, manifest=manifest, flatfields=flatfields)
    assert mpl.rcParams["font.family"] == original_family
    np.testing.assert_array_equal(figures[0][0][:, :4], 0)
    np.testing.assert_array_equal(figures[0][0][:, 4:8], 25)
    assert figures[0][0].mask[:, 8:].all()
    np.testing.assert_allclose(figures[0][1], [0.2, 0.272, 0.132, 0.1])
    metadata = json.loads((tmp_path / "qc/qc_manifest.json").read_text())
    assert metadata["channels"]["2"]["camera_shape_yx"] == [6, 10]
    assert metadata["channels"]["2"]["mask_half_ranges"] == [0.01, 0.01]
    for channel in (0, 2, 4):
        for extension in ("png", "pdf", "svg"):
            assert (tmp_path / "qc" / f"field_{channel}.{extension}").is_file()
            path = tmp_path / "qc" / f"channel_{channel}.{extension}"
            assert path.stat().st_size > 0
        with Image.open(tmp_path / "qc" / f"channel_{channel}.png") as image:
            assert image.info["dpi"] == pytest.approx((300, 300), abs=0.01)
    with (tmp_path / "qc/correction_magnitude.csv").open() as handle:
        field_rows = list(csv.DictReader(handle))
    assert len(field_rows) == 9
    assert all(float(row["minimum"]) == float(row["maximum"]) == 0 for row in field_rows)
    with (tmp_path / "qc/tile_gains.csv").open() as handle:
        gain_rows = list(csv.DictReader(handle))
    assert (
        next(r for r in gain_rows if r["channel"] == "0" and r["source"] == sources[1])["status"]
        == "identity: no training-pair support"
    )
    assert (
        next(r for r in gain_rows if r["channel"] == "4" and r["source"] == sources[0])["status"]
        == "identity: not fitted"
    )
    assert all(r["status"] == "excluded" for r in gain_rows if r["source"] == sources[2])
    with (tmp_path / "qc/source_fields.csv").open() as handle:
        source_rows = list(csv.DictReader(handle))
    assert {row["source"] for row in source_rows} == set(sources)
