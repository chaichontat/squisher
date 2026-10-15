from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from squisher_lightsheet import clcr_placement, placement_common
from squisher_lightsheet.cli import app


def _record(tile: str, *, x: float, channels: int, scale_x: float = 0.3) -> dict:
    return {
        "tile": tile,
        "path": f"/raw/{tile}",
        "translation_um": {"z": 0.0, "y": 0.0, "x": x},
        "scale_um": {"z": 2.0, "y": 0.3, "x": scale_x},
        "shape": [channels, 4, 8, 8],
        "axes": "CZYX",
        "channels": [str(index) for index in range(channels)],
        "tracks": [{"slug": "track0", "track_id": "Track:0", "channels": list(range(min(channels, 2)))}],
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _inputs(tmp_path: Path, *, accepted_moving: str = "CR-a.000.ome.zarr") -> dict[str, Path]:
    """CL from a four-channel acquisition and CR from a two-channel one, as 229Ptprz1."""
    deconv = tmp_path / "deconv"
    left = [_record("CL-main.000.ome.tif", x=0.0, channels=4), _record("CL-main.001.ome.tif", x=2.0, channels=4)]
    right = [_record("CR-a.000.ome.tif", x=0.0, channels=2)]
    for record in left + right:
        name = placement_common.source_zarr_name(record)
        group = zarr.open_group(str(deconv / name), mode="w")
        group.create_array("2", data=np.ones(tuple(record["shape"]), dtype=np.uint16))
    return {
        "left": _write(tmp_path / "cl.json", {"tiles": left}),
        "right": _write(tmp_path / "cr.json", {"tiles": right}),
        "manifest": _write(tmp_path / "manifest.json", {"metadata_centers_um_zyx": {"CL": [0, 0, 10], "CR": [0, 0, 20]}}),
        "phase": _write(
            tmp_path / "phase.json",
            {
                "artifact_type": "squisher_lightsheet.registered_lr_phasecorr.v1",
                "fixed": "CL",
                "moving": "CR",
                "total_shift_to_apply_to_cr_zyx_um": [1.0, 2.0, 3.0],
            },
        ),
        "method8": _write(
            tmp_path / "method8.json",
            {
                "windows": [
                    {
                        "status": "accepted",
                        "fixed_tile": "CL-main.000.ome.zarr",
                        "moving_tile": accepted_moving,
                        "local_translation_zyx": [1.0, 0.0, 10.0],
                    },
                    {"status": "rejected", "fixed_tile": "x", "moving_tile": "y"},
                ]
            },
        ),
        "deconv": deconv,
    }


def _publish(inputs: dict[str, Path], output: Path) -> dict[str, str]:
    return clcr_placement.publish_clcr_placement(
        left_position=inputs["left"],
        right_position=inputs["right"],
        alignment_manifest=inputs["manifest"],
        phasecorr=inputs["phase"],
        method8_summary=inputs["method8"],
        deconv_root=inputs["deconv"],
        output_dir=output,
        qc_channel=0,
        qc_level=2,
    )


def test_placement_keeps_each_side_s_channels_and_adds_the_cr_shift(tmp_path: Path) -> None:
    outputs = _publish(_inputs(tmp_path), tmp_path / "out")

    positions = json.loads(Path(outputs["positions"]).read_text())
    by_tile = {record["tile"]: record for record in positions["tiles"]}
    assert by_tile["CL-main.000.ome.zarr"]["shape"][0] == 4
    assert by_tile["CR-a.000.ome.zarr"]["shape"][0] == 2
    assert by_tile["CR-a.000.ome.zarr"]["side"] == "R"
    # CL centroid moves to its metadata centre; CR additionally takes the align-registered-lr shift plus the median
    # Method8 translation in CL voxels (1 * 2.0 um in Z, 10 * 0.3 um in X).
    # The CL tiles at x 0 and 2 um, 2.4 um wide, centre at 2.2 um, which moves to the 10 um metadata centre.
    assert by_tile["CL-main.000.ome.zarr"]["translation_um"]["x"] == pytest.approx(10.0 - 2.2)
    net = positions["alignment"]["net_cr_shift_after_centroid_um_zyx"]
    assert net == pytest.approx([3.0, 2.0, 6.0])
    registration = json.loads(Path(outputs["registration"]).read_text())
    assert {record["source_view"] for record in registration["tiles"]} == {"L", "R"}
    assert Path(outputs["overlay"]).is_file()


def test_placement_rejects_method8_rows_outside_the_cl_cr_pairing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not pair a CL fixed tile with a CR moving tile"):
        _publish(_inputs(tmp_path, accepted_moving="CL-main.001.ome.zarr"), tmp_path / "out")


def test_placement_rejects_sides_with_different_scales(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _write(inputs["right"], {"tiles": [_record("CR-a.000.ome.tif", x=0.0, channels=2, scale_x=0.31)]})

    with pytest.raises(ValueError, match="absolute scales differ"):
        _publish(inputs, tmp_path / "out")


def test_cli_publishes_placement(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "clcr-placement",
            "--left-position", str(inputs["left"]),
            "--right-position", str(inputs["right"]),
            "--alignment-manifest", str(inputs["manifest"]),
            "--phasecorr", str(inputs["phase"]),
            "--method8-summary", str(inputs["method8"]),
            "--deconv-root", str(inputs["deconv"]),
            "--output-dir", str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "registration.json").is_file()
