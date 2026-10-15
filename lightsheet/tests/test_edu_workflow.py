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
    transforms = root["attributes"]["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"]
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


def test_registration_stage_rejects_changed_position_at_same_path(tmp_path: Path) -> None:
    from squisher_lightsheet.artifact_io import registration_input_fingerprint

    paths = edu_workflow.EduPaths(tmp_path, tmp_path / "multi", tmp_path / "405")
    position = tmp_path / "positions.json"
    position.write_text('{"tiles": [], "revision": 1}')
    output = tmp_path / "registration.json"
    output.write_text("{}")
    runner = edu_workflow.StageRunner(paths, dry_run=False)

    def identity():
        return registration_input_fingerprint(position, tmp_path / "tiles")

    runner.command("register", ["/bin/true"], required=[output], input_identity=identity)
    runner.command("register", ["/bin/true"], required=[output], input_identity=identity)
    position.write_text('{"tiles": [], "revision": 2}')

    with pytest.raises(ValueError, match="inputs changed"):
        runner.command("register", ["/bin/true"], required=[output], input_identity=identity)
    assert output.read_text() == "{}"


def test_fusion_rejects_changed_registration_at_same_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = edu_workflow.EduPaths(tmp_path, tmp_path / "multi", tmp_path / "405")
    source = tmp_path / "tiles"
    tile = source / "tile.ome.zarr"
    (tile / "0").mkdir(parents=True)
    (tile / "zarr.json").write_text('{"attributes": {"squisher_complete": true}}')
    (tile / "0/zarr.json").write_text("{}")
    position = tmp_path / "positions.json"
    registration = tmp_path / "registration.json"
    position.write_text("{}")
    registration.write_text('{"revision": 1}')
    fused = paths.fused("561")
    fused.mkdir(parents=True)
    (fused / "zarr.json").write_text('{"attributes": {"squisher_complete": true}}')
    (paths.grid("561") / "0").mkdir(parents=True)
    (paths.grid("561") / "zarr.json").write_text("{}")
    (paths.grid("561") / "0/zarr.json").write_text("{}")
    popen = edu_workflow.subprocess.Popen
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        return popen(["/bin/true"], **kwargs)

    monkeypatch.setattr(edu_workflow.subprocess, "Popen", fake_popen)
    runner = edu_workflow.StageRunner(paths, dry_run=False)

    def fuse():
        edu_workflow._fuse_source(
            runner,
            paths,
            label="561",
            source_dir=source,
            position=position,
            registration=registration,
            channel=0,
            correction=None,
            primary_gpu_env={},
        )

    fuse()
    fuse()
    assert len(calls) == 1
    registration.write_text('{"revision": 2}')
    with pytest.raises(ValueError, match="inputs changed"):
        fuse()
    assert len(calls) == 1


@pytest.mark.parametrize("label,channel", [("514", 2), ("405", 0)])
@pytest.mark.parametrize(
    "changed", [None, "registration", "correction", "summary", "window", "tile", "settings", "unbound"]
)
def test_materialized_fusion_reuse_is_bound_to_durable_inputs_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
    channel: int,
    changed: str | None,
) -> None:
    paths = edu_workflow.EduPaths(tmp_path, tmp_path / "multi", tmp_path / "405")
    tile = paths.deconv(label) / "tile.ome.zarr"
    (tile / "0").mkdir(parents=True)
    (tile / "zarr.json").write_text('{"attributes": {"squisher_complete": true}}')
    (tile / "0/zarr.json").write_text("{}")
    paths.registration_dir(label).mkdir(parents=True)
    paths.registered_positions(label).write_text(json.dumps({"tiles": [{"tile": "tile.ome.zarr"}]}))
    paths.registration(label).write_text('{"revision": 1}')
    correction = paths.residual(label) / "correction.json"
    correction.parent.mkdir(parents=True)
    correction.write_text('{"revision": 1}')
    window = tmp_path / "window.json"
    window.write_text('{"revision": 1}')
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"windows": [{"level0_json": str(window)}], "revision": 1}))
    (paths.fused("561") / "0").mkdir(parents=True)
    (paths.fused("561") / "zarr.json").write_text('{"attributes": {"squisher_complete": true}}')
    (paths.fused("561") / "0/zarr.json").write_text("{}")
    materialized = paths.multi / f"materialized-{label}"
    popen = edu_workflow.subprocess.Popen
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        if command[1] == "fused-fixed-materialize-overlap":
            (materialized / "materialized_tiles").mkdir(parents=True)
            for name in ("positions", "registration", "summary"):
                (materialized / f"fused_fixed_materialized_chunks.{name}.json").write_text("{}")
        elif command[1] == "fuse":
            paths.fused(label).mkdir(parents=True)
            (paths.fused(label) / "zarr.json").write_text('{"attributes": {"squisher_complete": true}}')
        else:
            pytest.fail(f"Unexpected command: {command}")
        return popen(["/bin/true"], **kwargs)

    monkeypatch.setattr(edu_workflow.subprocess, "Popen", fake_popen)
    runner = edu_workflow.StageRunner(paths, dry_run=False)

    def run():
        edu_workflow._run_materialized_fusion(
            runner,
            paths,
            label=label,
            channel=channel,
            summary=summary,
            primary_gpu_env={},
        )

    run()
    assert len(calls) == 2
    assert not materialized.exists()
    run()
    assert len(calls) == 2
    if changed is not None:
        if changed == "settings":
            channel += 1
        elif changed == "unbound":
            marker = runner._marker(f"fuse-{label}")
            payload = json.loads(marker.read_text())
            payload.pop("inputs")
            marker.write_text(json.dumps(payload))
        else:
            target = {
                "registration": paths.registration(label),
                "correction": correction,
                "summary": summary,
                "window": window,
                "tile": tile / "0/zarr.json",
            }[changed]
            payload = json.loads(target.read_text())
            payload["revision"] = 2
            target.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="inputs changed"):
            run()
        assert len(calls) == 2
        assert not materialized.exists()
        edu_workflow._require_complete_fusion(paths.fused(label))


def test_completed_materialized_fusion_without_marker_is_not_adopted(tmp_path: Path) -> None:
    paths = edu_workflow.EduPaths(tmp_path, tmp_path / "multi", tmp_path / "405")
    paths.fused("405").mkdir(parents=True)
    (paths.fused("405") / "zarr.json").write_text('{"attributes": {"squisher_complete": true}}')
    runner = edu_workflow.StageRunner(paths, dry_run=False)
    with pytest.raises(ValueError, match="no recorded input identity"):
        edu_workflow._run_materialized_fusion(
            runner,
            paths,
            label="405",
            channel=0,
            summary=tmp_path / "summary.json",
            primary_gpu_env={},
        )


def test_segmentation_runtime_report_cannot_reuse_unverified_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import zarr
    import torch
    from types import SimpleNamespace
    from squisher_segment.segment import model_artifacts

    model = tmp_path / "checkpoint"
    model.write_bytes(b"checkpoint")
    device_name = "CPU fixture GPU"
    plan = model_artifacts.plan_path_for_device(model, device_name)
    plan.write_bytes(b"plan")
    monkeypatch.setattr(edu_workflow, "DEFAULT_CELLPOSE_MODEL", model)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: device_name)
    monkeypatch.setattr(
        zarr, "open_group", lambda *_args, **_kwargs: {"0": SimpleNamespace(shape=(8, 2048, 2048))}
    )
    paths = edu_workflow.EduPaths(tmp_path, tmp_path / "multi", tmp_path / "405")
    report = tmp_path / "cellpose-subset/runtime-plan.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"gpu_uuid": "fixture-uuid", "plan": str(plan)}))

    class ReuseRunner:
        def action(self, name, _identity, _action, *, required, validate):
            assert name == "cellpose-runtime"
            validate()
            raise RuntimeError("Unverified plan was accepted")

    with pytest.raises(ValueError, match="squisher-segment build-trt"):
        edu_workflow._run_segmentation(
            ReuseRunner(),
            paths,
            fusion_identity={label: label for label in ("405", "561", "638")},
            primary_gpu_env={"CUDA_VISIBLE_DEVICES": "fixture-uuid"},
        )


@pytest.mark.parametrize(
    "changed", [None, "missing_sidecar", "invalid_sidecar", "checkpoint", "plan", "runtime", "report"]
)
def test_cellpose_runtime_revalidates_bound_plan_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed: str | None,
) -> None:
    import os
    import sys
    import torch
    from types import ModuleType
    from squisher_segment.segment import model_artifacts

    model = tmp_path / "checkpoint"
    model.write_bytes(b"checkpoint-v1")
    monkeypatch.setattr(edu_workflow, "DEFAULT_CELLPOSE_MODEL", model)
    device_name = "CPU fixture GPU"

    def get_device_name(_device):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "fixture-uuid"
        return device_name

    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)
    monkeypatch.setattr(model_artifacts, "_build_runtime_versions", lambda: {"runtime": "fixture-v1"})
    monkeypatch.setitem(
        sys.modules,
        "squisher_segment.segmentation.distributed.distributed_postproc",
        ModuleType("squisher_segment.segmentation.distributed.distributed_postproc"),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original-uuid")
    plan = model_artifacts.plan_path_for_device(model, device_name)
    plan.write_bytes(b"plan-v1")
    metadata = model_artifacts.write_trt_plan_metadata(
        model,
        device_name,
        backend="sam",
        bsize=256,
        batch_size=1,
        opset=17,
        vram_mb=1024,
    )
    paths = edu_workflow.EduPaths(tmp_path, tmp_path / "multi", tmp_path / "405")
    runner = edu_workflow.StageRunner(paths, dry_run=False)
    report = tmp_path / "runtime-plan.json"

    def run():
        edu_workflow._run_cellpose_runtime(
            runner,
            report,
            primary_gpu_env={"CUDA_VISIBLE_DEVICES": "fixture-uuid"},
        )

    run()
    payload = json.loads(report.read_text())
    assert payload["plan_metadata"] == metadata
    assert payload["model_sha256"] == model_artifacts.file_sha256(model)
    first_mtime = report.stat().st_mtime_ns
    run()
    assert report.stat().st_mtime_ns == first_mtime
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original-uuid"
    sidecar = model_artifacts.plan_metadata_path(plan)
    if changed is not None:
        if changed == "missing_sidecar":
            sidecar.unlink()
        elif changed == "invalid_sidecar":
            sidecar.write_text('{"schema_version": 1, "status": "invalid"}')
        elif changed == "checkpoint":
            stat = model.stat()
            model.write_bytes(b"checkpoint-v2")
            os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        elif changed == "plan":
            plan.write_bytes(b"plan-v2")
        elif changed == "runtime":
            monkeypatch.setattr(model_artifacts, "_build_runtime_versions", lambda: {"runtime": "fixture-v2"})
        elif changed == "report":
            payload["plan_metadata"] = {}
            report.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="build-trt|runtime identity"):
            run()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "original-uuid"
        assert plan.read_bytes() == (b"plan-v2" if changed == "plan" else b"plan-v1")
