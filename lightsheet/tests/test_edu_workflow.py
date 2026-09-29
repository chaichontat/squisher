from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from squisher_lightsheet import edu_workflow
from squisher_lightsheet._legacy.stitch_20x_tl_multiview import TileMetadata, TrackMetadata
from squisher_lightsheet.cli import app


def _metadata(path: Path, *, channels: int, names: tuple[str, ...]) -> TileMetadata:
    axes = "ZYX" if channels == 1 else "CZYX"
    shape = (8, 16, 16) if channels == 1 else (channels, 8, 16, 16)
    return TileMetadata(
        path=path,
        shape=shape,
        axes=axes,
        spacing={"z": 1.0, "y": 0.5, "x": 0.5},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        channels=names,
        tracks=(
            TrackMetadata(
                slug="track0",
                track_id="all",
                channels=tuple(range(channels)),
                channel_names=names,
            ),
        ),
    )


def test_discover_inputs_from_one_multichannel_tiff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    multi = tmp_path / "sample-561638514"
    dapi = tmp_path / "sample-405"
    multi.mkdir()
    dapi.mkdir()
    multi_tiffs = tuple(multi / f"tile.{index:03d}.ome.tif" for index in range(2))
    dapi_tiffs = tuple(dapi / f"tile.{index:03d}.ome.tif" for index in range(3))
    for path in (*multi_tiffs, *dapi_tiffs):
        path.touch()

    def fake_parse(path: Path) -> TileMetadata:
        if path.parent == multi:
            return _metadata(path, channels=3, names=("561", "638", "514"))
        return _metadata(path, channels=1, names=("405",))

    monkeypatch.setattr(edu_workflow, "parse_ome_metadata", fake_parse)
    inputs = edu_workflow.discover_inputs(multi_tiffs[0])

    assert inputs.multi_dir == multi
    assert inputs.dapi_dir == dapi
    assert inputs.multi_tiffs == multi_tiffs
    assert inputs.dapi_tiffs == dapi_tiffs


def test_discover_inputs_rejects_ambiguous_405_acquisitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    multi = tmp_path / "sample-561638514"
    multi.mkdir()
    source = multi / "tile.000.ome.tif"
    source.touch()
    for name in ("sample-405-1", "sample-405-2"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "tile.000.ome.tif").touch()

    def fake_parse(path: Path) -> TileMetadata:
        if path.parent == multi:
            return _metadata(path, channels=3, names=("561", "638", "514"))
        return _metadata(path, channels=1, names=("405",))

    monkeypatch.setattr(edu_workflow, "parse_ome_metadata", fake_parse)
    with pytest.raises(ValueError, match="found multi=1, 405=2"):
        edu_workflow.discover_inputs(source)


def test_threshold_gate_requires_all_positive_values(tmp_path: Path) -> None:
    gate = tmp_path / "thresholds.json"
    with pytest.raises(edu_workflow.ReviewRequired):
        edu_workflow._gate_numbers(gate, ("561", "405"), message="review")
    payload = json.loads(gate.read_text())
    payload["thresholds"] = {"561": 700, "405": 2000}
    gate.write_text(json.dumps(payload))

    assert edu_workflow._gate_numbers(gate, ("561", "405"), message="review") == {
        "561": 700.0,
        "405": 2000.0,
    }


def test_approval_gate_is_bound_to_reviewed_identity(tmp_path: Path) -> None:
    gate = tmp_path / "approval.json"
    identity = {"405": "abc", "514": "def"}
    with pytest.raises(edu_workflow.ReviewRequired):
        edu_workflow._gate_approval(gate, identity, message="review")
    payload = json.loads(gate.read_text())
    payload["approved"] = True
    gate.write_text(json.dumps(payload))
    edu_workflow._gate_approval(gate, identity, message="review")

    with pytest.raises(ValueError, match="no longer matches"):
        edu_workflow._gate_approval(gate, {"405": "changed", "514": "def"}, message="review")




def test_metadata_grid_uses_transformed_exclusive_tile_extents(tmp_path: Path) -> None:
    tile_dir = tmp_path / "tiles"
    source = tile_dir / "tile.000.ome.zarr"
    source.mkdir(parents=True)
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "spacing_um": {"z": 1.0, "y": 0.5, "x": 0.5},
                "tiles": [
                    {
                        "path": str(source),
                        "shape": [3, 8, 16, 20],
                        "stage_scale_um": {"z": 1.0, "y": 0.5, "x": 0.5},
                        "stage_translation_um": {"z": 2.0, "y": 4.0, "x": 6.0},
                        "registered_affine": {"matrix": np.eye(4).tolist()},
                    }
                ],
            }
        )
    )
    output = tmp_path / "grid.ome.zarr"

    edu_workflow._write_metadata_grid(registration, tile_dir, output)

    root = json.loads((output / "zarr.json").read_text())
    level = json.loads((output / "0/zarr.json").read_text())
    transforms = root["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    assert level["shape"] == [8, 16, 20]
    assert transforms == [
        {"scale": [1.0, 0.5, 0.5], "type": "scale"},
        {"translation": [2.0, 4.0, 6.0], "type": "translation"},
    ]


def test_cli_reports_review_gate_with_distinct_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "tile.000.ome.tif"
    source.touch()
    gate = tmp_path / "native-thresholds.json"

    def stop_at_gate(*_args: object, **_kwargs: object) -> Path:
        raise edu_workflow.ReviewRequired("Review native TIFFs", gate)

    monkeypatch.setattr(edu_workflow, "run_edu_workflow", stop_at_gate)
    result = CliRunner().invoke(app, ["edu-workflow", str(source)])

    assert result.exit_code == 3
    assert str(gate) in result.stdout


def test_manual_initialization_args_pin_master_file_by_content(tmp_path: Path) -> None:
    paths = edu_workflow.EduPaths(root=tmp_path, multi=tmp_path / "multi", dapi=tmp_path / "405")

    assert edu_workflow._manual_initialization_args(paths, "405") == []

    paths.manual_initializations.write_text('{"artifact_type": "x", "channels": {}}')

    assert edu_workflow._manual_initialization_args(paths, "405") == [
        "--manual-initializations",
        paths.manual_initializations,
        "--manual-initializations-channel",
        "405",
    ]


class _RecordingRunner:
    def __init__(self) -> None:
        self.commands: dict[str, list[str]] = {}

    def command(self, name, argv, **_kwargs) -> None:
        self.commands[name] = [str(value) for value in argv]


@pytest.mark.parametrize(("budget", "expected"), [(None, []), ("18", ["--fusion-gpu-memory-max-gib", "18"])])
def test_fuse_source_passes_the_gpu_memory_budget_only_when_given(
    tmp_path: Path, budget: str | None, expected: list[str]
) -> None:
    paths = edu_workflow.EduPaths(root=tmp_path, multi=tmp_path / "multi", dapi=tmp_path / "405")
    runner = _RecordingRunner()

    edu_workflow._fuse_source(
        runner,
        paths,
        label="514",
        source_dir=tmp_path / "tiles",
        position=tmp_path / "positions.json",
        registration=tmp_path / "registration.json",
        channel=0,
        correction=None,
        primary_gpu_env={},
        gpu_memory_max_gib=budget,
    )

    argv = runner.commands["fuse-514"]
    assert "--fusion-gpu-memory-max-gib" not in argv[: len(argv) - len(expected)]
    assert argv[len(argv) - len(expected) :] == expected
