from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from loguru import logger
from typer.testing import CliRunner
from typer.main import get_command

import squisher_lightsheet.cli as cli_module
from squisher_lightsheet.cli import app


README = Path(__file__).resolve().parents[1] / "README.md"


def invoke_with_log_messages(args: list[str]):
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="INFO")
    try:
        result = CliRunner().invoke(app, args)
    finally:
        logger.remove(sink_id)
    return result, "\n".join(messages)


def cli_options(command_name: str) -> set[str]:
    command = get_command(app).commands[command_name]
    return {
        option
        for parameter in command.params
        for option in (*getattr(parameter, "opts", ()), *getattr(parameter, "secondary_opts", ()))
        if option.startswith("--")
    }


def readme_command_options(command_name: str) -> set[str]:
    text = README.read_text()
    for match in re.finditer(r"```bash\n(?P<body>.*?)\n```", text, flags=re.DOTALL):
        body = match.group("body")
        if f"lightsheet {command_name}" in body:
            return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", body))
    raise AssertionError(f"README has no bash block for lightsheet {command_name}")


def test_cli_exposes_lean_stitching_subcommands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "position",
        "plot-zeiss-positions",
        "zeiss-tile-positions",
        "single-position",
        "rough-phase",
        "fuse",
        "pyramid",
        "qc",
        "fused-tile-index-qc",
        "live-fusion-preview",
        "fused-xyz-overlay-qc",
        "registration-center-z-spotcheck",
        "ome-metadata-dumb-stitch",
        "mvs-edge-audit",
        "mvs-refine-level0",
        "rechunk-ome-tiff",
        "subtract-channel",
        "track-z-diagnostics",
        "align-lr-dumb-stitch",
        "tile-phase-align",
        "channel-affine-registration",
        "cross-register-method8",
        "run-tltr",
    ):
        assert command in result.stdout


def test_channel_affine_registration_cli_forwards_required_contract(
    tmp_path: Path, monkeypatch
) -> None:
    window_dir = tmp_path / "windows"
    window_dir.mkdir()
    reference = tmp_path / "reference.json"
    reference.touch()
    fixed_fused = tmp_path / "fixed.ome.zarr"
    fixed_fused.mkdir()
    output = tmp_path / "output.json"
    captured = {}

    def fake_write(**kwargs):
        captured.update(kwargs)
        return output.resolve()

    monkeypatch.setattr(cli_module, "write_global_channel_affine_registration", fake_write)

    result = CliRunner().invoke(
        app,
        [
            "channel-affine-registration",
            "--window-dir",
            str(window_dir),
            "--reference-registration",
            str(reference),
            "--output-registration",
            str(output),
            "--expected-moving-channel",
            "1",
            "--expected-fixed-fused",
            str(fixed_fused),
            "--source-label",
            "638",
            "--target-label",
            "561",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == str(output.resolve())
    assert captured == {
        "window_dir": window_dir,
        "reference_registration_input": reference,
        "output_registration": output,
        "expected_moving_channel": 1,
        "expected_fixed_fused": fixed_fused,
        "source_label": "638",
        "target_label": "561",
    }


def test_cli_registers_jpegxr_codec_before_command(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli_module, "register_jpegxr_codec", lambda: calls.append("codec"))
    monkeypatch.setattr(
        cli_module,
        "create_single_position_file",
        lambda **kwargs: calls.append("command") or kwargs["output"],
    )

    result = CliRunner().invoke(
        app,
        [
            "single-position",
            "--input-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "position.json"),
            "--side",
            "L",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["codec", "command"]


def test_single_position_cli_forwards_metadata_position_options(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "R.positions.json"
    captured = {}

    def fake_create_single_position_file(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(cli_module, "create_single_position_file", fake_create_single_position_file)

    result = CliRunner().invoke(
        app,
        [
            "single-position",
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--side",
            "R",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "input_dir": tmp_path,
        "output": output,
        "side": "R",
        "plot_title": "metadata tile positions",
        "progress": cli_module.typer.echo,
    }


def test_plot_zeiss_positions_cli_forwards_options(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.pos"
    output_path = tmp_path / "positions.png"
    tile_positions = tmp_path / "tiles.json"
    input_path.touch()
    tile_positions.touch()
    captured = {}

    def fake_plot_xy_positions(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return output_path.resolve()

    monkeypatch.setattr(cli_module, "plot_xy_positions", fake_plot_xy_positions)

    result = CliRunner().invoke(
        app,
        [
            "plot-zeiss-positions",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--title",
            "Acquisition XY",
            "--tile-positions",
            str(tile_positions),
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "args": (input_path, output_path),
        "kwargs": {"title": "Acquisition XY", "tile_positions": tile_positions},
    }
    assert result.stdout.strip() == str(output_path.resolve())


def test_zeiss_tile_positions_cli_forwards_options(tmp_path: Path, monkeypatch) -> None:
    pos_input = tmp_path / "input.pos"
    pos_input.touch()
    output = tmp_path / "positions.json"
    captured = {}

    def fake_create_zeiss_tile_position_file(**kwargs):
        captured.update(kwargs)
        return output.resolve()

    monkeypatch.setattr(
        cli_module,
        "create_zeiss_tile_position_file",
        fake_create_zeiss_tile_position_file,
    )

    result = CliRunner().invoke(
        app,
        [
            "zeiss-tile-positions",
            "--pos-input",
            str(pos_input),
            "--output",
            str(output),
            "--side",
            "R",
            "--overlap-fraction",
            "0.25",
            "--min-hull-overlap-fraction",
            "0.03",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "pos_input": pos_input,
        "output": output,
        "side": "R",
        "overlap_fraction": 0.25,
        "min_hull_overlap_fraction": 0.03,
    }
    assert result.stdout.strip() == str(output.resolve())


def test_cli_does_not_expose_generic_register_command() -> None:
    result = CliRunner().invoke(app, ["register", "--help"])

    assert result.exit_code == 2
    assert "No such command 'register'" in result.stderr


def test_cross_register_method8_group_exposes_stage_subcommands() -> None:
    result = CliRunner().invoke(app, ["cross-register-method8", "--help"])

    assert result.exit_code == 0
    for command in ("coarse", "method8", "materialize", "manifest"):
        assert command in result.stdout


def test_cross_register_method8_coarse_forwards_tile_phase_options(tmp_path: Path, monkeypatch) -> None:
    fixed_position = tmp_path / "Image_14.positions.json"
    fixed_position.write_text('{"tiles":[]}\n')
    output_dir = tmp_path / "run"
    captured = {}

    def fake_align_tiles_to_reference(**kwargs):
        captured.update(kwargs)
        return kwargs["output_position"]

    monkeypatch.setattr(cli_module, "align_tiles_to_reference", fake_align_tiles_to_reference)

    result = CliRunner().invoke(
        app,
        [
            "cross-register-method8",
            "coarse",
            "--fixed-position",
            str(fixed_position),
            "--output-dir",
            str(output_dir),
            "--fixed-token",
            "Image_14",
            "--moving-token",
            "Image_10",
            "--fixed-channel",
            "0",
            "--moving-channel",
            "1",
            "--patch-shape-zyx",
            "96,320,320",
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert captured["reference_position"] == fixed_position
    assert captured["reference_token"] == "Image_14"
    assert captured["moving_token"] == "Image_10"
    assert captured["reference_channel"] == 0
    assert captured["moving_channel"] == 1
    assert captured["patch_shape_zyx"] == (96, 320, 320)
    assert captured["workers"] == 2
    payload = json.loads(result.stdout)
    assert payload["coarse_position"].endswith("Image_10.ch1.coarse-aligned-to-Image_14.positions.json")


def test_cross_register_method8_method8_forwards_runner_options(tmp_path: Path, monkeypatch) -> None:
    fixed_position = tmp_path / "fixed.positions.json"
    moving_position = tmp_path / "moving.positions.json"
    fixed_position.write_text('{"tiles":[]}\n')
    moving_position.write_text('{"tiles":[]}\n')
    output_dir = tmp_path / "run"
    captured = {}

    def fake_run_tile_quadrant_method8(**kwargs):
        captured.update(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        path = kwargs["output_dir"] / "tile_quadrant_method8_summary.json"
        path.write_text("{}\n")
        return path

    monkeypatch.setattr(cli_module, "run_tile_quadrant_method8", fake_run_tile_quadrant_method8)

    result = CliRunner().invoke(
        app,
        [
            "cross-register-method8",
            "method8",
            "--fixed-position",
            str(fixed_position),
            "--coarse-moving-position",
            str(moving_position),
            "--output-dir",
            str(output_dir),
            "--native-lib-dir",
            str(tmp_path),
            "--fixed-mask-threshold",
            "2500",
            "--core-shape-zyx",
            "480,480,480",
            "--window-shape-zyx",
            "528,528,528",
            "--workers",
            "4",
            "--devices",
            "0,1",
            "--no-resume",
        ],
    )

    assert result.exit_code == 0
    assert captured["fixed_position"] == fixed_position
    assert captured["moving_position"] == moving_position
    assert captured["fixed_mask_threshold"] == 2500.0
    assert captured["core_shape_zyx"] == (480, 480, 480)
    assert captured["window_shape_zyx"] == (528, 528, 528)
    assert captured["workers"] == 4
    assert captured["devices"] == "0,1"
    assert captured["resume"] is False


def test_cross_register_method8_method8_forwards_default_geometry_and_mask_off(
    tmp_path: Path, monkeypatch
) -> None:
    fixed_position = tmp_path / "fixed.positions.json"
    moving_position = tmp_path / "moving.positions.json"
    fixed_position.write_text('{"tiles":[]}\n')
    moving_position.write_text('{"tiles":[]}\n')
    output_dir = tmp_path / "run"
    captured = {}

    def fake_run_tile_quadrant_method8(**kwargs):
        captured.update(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        path = kwargs["output_dir"] / "tile_quadrant_method8_summary.json"
        path.write_text("{}\n")
        return path

    monkeypatch.setattr(cli_module, "run_tile_quadrant_method8", fake_run_tile_quadrant_method8)

    result = CliRunner().invoke(
        app,
        [
            "cross-register-method8",
            "method8",
            "--fixed-position",
            str(fixed_position),
            "--coarse-moving-position",
            str(moving_position),
            "--output-dir",
            str(output_dir),
            "--native-lib-dir",
            str(tmp_path),
            "--fixed-mask-threshold",
            "none",
        ],
    )

    assert result.exit_code == 0
    assert captured["core_shape_zyx"] == (480, 480, 480)
    assert captured["window_shape_zyx"] == (528, 528, 528)
    assert captured["fixed_mask_threshold"] is None


def test_cross_register_method8_materialize_forwards_export_options(tmp_path: Path, monkeypatch) -> None:
    window_dir = tmp_path / "window_json"
    window_dir.mkdir()
    coarse_position = tmp_path / "coarse.positions.json"
    fixed_registration = tmp_path / "fixed.registration.json"
    coarse_position.write_text('{"tiles":[]}\n')
    fixed_registration.write_text('{"tiles":[]}\n')
    output_dir = tmp_path / "run"
    captured = {}

    def fake_export_tile_quadrant_materialized_chunks(**kwargs):
        captured.update(kwargs)
        output = kwargs["output_dir"]
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "position": output / "tile_quadrant_materialized_chunks.positions.json",
            "registration": output / "tile_quadrant_materialized_chunks.registration.json",
            "summary": output / "tile_quadrant_materialized_chunks.summary.json",
        }
        for path in paths.values():
            path.write_text("{}\n")
        return paths

    monkeypatch.setattr(
        cli_module,
        "export_tile_quadrant_materialized_chunks",
        fake_export_tile_quadrant_materialized_chunks,
    )

    result = CliRunner().invoke(
        app,
        [
            "cross-register-method8",
            "materialize",
            "--window-json-dir",
            str(window_dir),
            "--coarse-moving-position",
            str(coarse_position),
            "--fixed-registration",
            str(fixed_registration),
            "--output-dir",
            str(output_dir),
            "--fusion-channel",
            "1",
            "--channel-source-shift-px-zyx",
            "2.3,1.3,1.1",
            "--jpegxr-level",
            "0.85",
        ],
    )

    assert result.exit_code == 0
    assert captured["window_json_dir"] == window_dir
    assert captured["moving_position_input"] == coarse_position
    assert captured["fixed_registration_input"] == fixed_registration
    assert captured["output_dir"] == output_dir / "materialized_fusion_inputs_ch1"
    assert captured["channel_source_shift_px_zyx"] == (2.3, 1.3, 1.1)
    assert captured["include_quality_gate_rejected"] is False
    assert captured["jpegxr_level"] == 0.85


def test_fused_fixed_materialize_overlap_forwards_grid_and_export_options(
    tmp_path: Path, monkeypatch
) -> None:
    source_registration = tmp_path / "core.registration.json"
    moving_position = tmp_path / "moving.positions.json"
    source_registration.write_text('{"tiles":[]}\n')
    moving_position.write_text('{"tiles":[]}\n')
    output_dir = tmp_path / "overlap"
    captured = {}

    def fake_export_fused_fixed_overlapping_materialized_chunks(**kwargs):
        captured.update(kwargs)
        output_dir.mkdir(parents=True)
        outputs = {
            "position": output_dir / "positions.json",
            "registration": output_dir / "registration.json",
            "summary": output_dir / "summary.json",
        }
        for path in outputs.values():
            path.write_text("{}\n")
        return outputs

    monkeypatch.setattr(
        cli_module,
        "export_fused_fixed_overlapping_materialized_chunks",
        fake_export_fused_fixed_overlapping_materialized_chunks,
    )

    result = CliRunner().invoke(
        app,
        [
            "fused-fixed-materialize-overlap",
            "--source-registration",
            str(source_registration),
            "--moving-position",
            str(moving_position),
            "--output-dir",
            str(output_dir),
            "--output-codec",
            "zstd",
            "--workers",
            "6",
            "--max-tiles",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["source_registration_input"] == source_registration
    assert captured["source_summary_input"] is None
    assert captured["moving_position_input"] == moving_position
    assert captured["output_dir"] == output_dir
    assert captured["core_shape_zyx"] == (480, 480, 480)
    assert captured["window_shape_zyx"] == (528, 528, 528)
    assert captured["level_factor_zyx"] == (4, 4, 4)
    assert captured["source_channel"] == 0
    assert captured["output_codec"] == "zstd"
    assert captured["jpegxr_level"] == 0.7
    assert captured["workers"] == 6
    assert captured["max_tiles"] == 2
    assert captured["resume"] is False


def test_cross_register_method8_manifest_validates_expected_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    materialized_dir = output_dir / "materialized_fusion_inputs_ch1"
    paths = [
        output_dir / "Image_10.ch0.coarse-aligned-to-Image_14.positions.json",
        output_dir / "tile_quadrant_method8_summary.json",
        materialized_dir / "tile_quadrant_materialized_chunks.positions.json",
        materialized_dir / "tile_quadrant_materialized_chunks.registration.json",
        materialized_dir / "tile_quadrant_materialized_chunks.summary.json",
    ]
    (output_dir / "window_json").mkdir(parents=True)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    result = CliRunner().invoke(app, ["cross-register-method8", "manifest", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    manifest = json.loads((output_dir / "cross_register_method8_manifest.json").read_text())
    assert manifest["artifact_type"] == "squisher_lightsheet.cross_register_method8_manifest.v1"
    assert manifest["stages"]["manifest"]["materialized_registration"].endswith(
        "tile_quadrant_materialized_chunks.registration.json"
    )


def test_readme_mvs_refine_level0_options_match_cli() -> None:
    assert readme_command_options("mvs-refine-level0") <= cli_options("mvs-refine-level0")


def test_run_tltr_dry_run_reports_expected_paths(tmp_path) -> None:
    left = tmp_path / "TL"
    right = tmp_path / "TR"
    left.mkdir()
    right.mkdir()
    output_prefix = tmp_path / "sample"

    result, logs = invoke_with_log_messages(
        [
            "run-tltr",
            "--left-dir",
            str(left),
            "--right-dir",
            str(right),
            "--output-prefix",
            str(output_prefix),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "sample.positions.json" in logs
    assert "sample.roughPhase.positions.json" in logs
    assert "level2_phase_corrected_zyx_yellowOverlay_ch0.png" in logs
    assert "workflow_summary.json" in result.stdout


def test_align_lr_dumb_stitch_dry_run_reports_expected_paths(tmp_path) -> None:
    left = tmp_path / "TL"
    right = tmp_path / "TR"
    left.mkdir()
    right.mkdir()
    output_prefix = tmp_path / "sample"

    result, logs = invoke_with_log_messages(
        [
            "align-lr-dumb-stitch",
            "--left-dir",
            str(left),
            "--right-dir",
            str(right),
            "--output-prefix",
            str(output_prefix),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "sample.positions.json" in logs
    assert "sample.roughPhase.positions.json" in logs
    assert "level2_phase_corrected_zyx_yellowOverlay_ch0.png" in logs
    assert "lr_dumb_stitch_alignment_summary.json" in result.stdout


def test_align_405_to_488_exposes_stage3_mattes_command() -> None:
    result = CliRunner().invoke(app, ["align-405-to-488", "--help"])

    assert result.exit_code == 0
    assert "measure-stage3-mattes" in result.stdout
    assert "render-candidate-grid" in result.stdout


def test_fused_fixed_contact_sheet_cli_accepts_incomplete_run(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "active-run"
    run_dir.mkdir()
    renderer_python = tmp_path / "python"
    renderer_python.touch()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="/tmp/accepted_contact_sheet.png\n", stderr="")

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_module, "FUSED_FIXED_CONTACT_SHEET_SCRIPT", Path(__file__))

    result = CliRunner().invoke(
        app,
        [
            "fused-fixed-contact-sheet",
            "--run-dir",
            str(run_dir),
            "--limit",
            "8",
            "--renderer-python",
            str(renderer_python),
        ],
    )

    assert result.exit_code == 0
    assert "accepted_contact_sheet.png" in result.stdout
    assert captured["command"][captured["command"].index("--run-dir") + 1] == str(run_dir.resolve())
    assert captured["command"][captured["command"].index("--limit") + 1] == "8"
    assert captured["command"][0] == str(renderer_python)
    assert captured["kwargs"] == {"check": True, "text": True, "capture_output": True}


def test_mvs_edge_audit_cli_reports_dropped_edges(tmp_path) -> None:
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "metrics": {
                    "pairwise_registration": {
                        "edges": [
                            {"source": 0, "target": 1, "quality": {"data": [0.8]}, "attrs": {}},
                            {"source": 1, "target": 2, "quality": {"data": [0.9]}, "attrs": {}},
                        ]
                    },
                    "groupwise_resolution": {
                        "metrics": {
                            "used_edges": {"0": [[0, 1]]},
                            "edge_residuals": {"0": {"(0, 1)": 0.1, "(1, 2)": 9.0}},
                        }
                    },
                }
            }
        )
    )

    result = CliRunner().invoke(app, ["mvs-edge-audit", "--registration-input", str(registration)])

    assert result.exit_code == 0
    audit = json.loads(result.stdout)
    assert audit["measured_edge_count"] == 2
    assert audit["used_edge_count"] == 1
    assert audit["dropped_edges"][0]["pair"] == [1, 2]


def test_mvs_edge_audit_cli_requires_registration_input() -> None:
    result = CliRunner().invoke(app, ["mvs-edge-audit"])

    assert result.exit_code == 2
    assert "--registration-input" in result.output


def test_mvs_refine_level0_defaults_to_more_candidate_patches() -> None:
    result = CliRunner().invoke(app, ["mvs-refine-level0", "--help"])

    assert result.exit_code == 0
    assert "--patches-per-" in result.stdout
    assert "[default: 12]" in result.stdout
    assert "--fallback-refi" in result.stdout
    assert "level-2" in result.stdout


def test_mvs_refine_level0_passes_repeated_fallback_refinement_levels(tmp_path, monkeypatch) -> None:
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    output = tmp_path / "refined.json"
    captured = {}

    def fake_refine_mvs_registration_level0(**kwargs):
        captured.update(kwargs)
        return {"metrics": {"level0_refinement": {"output_registration": str(output)}}}, {}

    monkeypatch.setattr(cli_module, "refine_mvs_registration_level0", fake_refine_mvs_registration_level0)

    result = CliRunner().invoke(
        app,
        [
            "mvs-refine-level0",
            "--registration-input",
            str(registration),
            "--output-registration",
            str(output),
            "--fallback-refinement-level",
            "1",
            "--fallback-refinement-level",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert captured["fallback_refinement_levels"] == (1, 2)


def test_rechunk_ome_tiff_cli_exposes_chunk_default() -> None:
    result = CliRunner().invoke(app, ["rechunk-ome-tiff", "--help"])

    assert result.exit_code == 0
    assert "--workers" in result.stdout
    assert "12,240,240" in result.stdout


def test_fuse_cli_parses_output_chunksize_zyx(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_fuse_tiles(**kwargs):
        captured.update(kwargs)
        return "ok"

    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    position = tmp_path / "positions.json"
    registration = tmp_path / "registration.json"
    position.write_text("{}")
    registration.write_text("{}")
    monkeypatch.setattr(cli_module, "fuse_tiles", fake_fuse_tiles)

    result = CliRunner().invoke(
        app,
        [
            "fuse",
            str(input_dir),
            "--position-input",
            str(position),
            "--registration-input",
            str(registration),
            "--output",
            str(tmp_path / "fused.ome.zarr"),
            "--output-chunksize-zyx",
            "12,960,960",
            "--resume-fusion",
        ],
    )

    assert result.exit_code == 0
    assert captured["output_chunksize_zyx"] == (12, 960, 960)
    assert captured["batch_size"] == 1
    assert captured["resume_fusion"] is True


def test_fuse_cli_uses_level0_output_chunksize_by_default(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_fuse_tiles(**kwargs):
        captured.update(kwargs)
        return "ok"

    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    position = tmp_path / "positions.json"
    registration = tmp_path / "registration.json"
    position.write_text("{}")
    registration.write_text("{}")
    monkeypatch.setattr(cli_module, "fuse_tiles", fake_fuse_tiles)

    result = CliRunner().invoke(
        app,
        [
            "fuse",
            str(input_dir),
            "--position-input",
            str(position),
            "--registration-input",
            str(registration),
            "--output",
            str(tmp_path / "fused.ome.zarr"),
        ],
    )

    assert result.exit_code == 0
    assert captured["output_chunksize_zyx"] == (12, 960, 960)


def test_registration_center_z_spotcheck_cli_passes_channels(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_render_registration_center_z_spotcheck(**kwargs):
        captured.update(kwargs)
        return [tmp_path / "ch0.png", tmp_path / "ch1.png"]

    position = tmp_path / "positions.json"
    registration = tmp_path / "registration.json"
    position.write_text("{}")
    registration.write_text("{}")
    monkeypatch.setattr(
        cli_module,
        "render_registration_center_z_spotcheck",
        fake_render_registration_center_z_spotcheck,
    )

    result = CliRunner().invoke(
        app,
        [
            "registration-center-z-spotcheck",
            "--position-input",
            str(position),
            "--registration-input",
            str(registration),
            "--output-dir",
            str(tmp_path),
            "--channel",
            "0",
            "--channel",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured["channels"] == [0, 1]
    assert captured["level"] == 4
    assert "ch0.png" in result.stdout
    assert "ch1.png" in result.stdout


def test_fused_tile_index_qc_cli_dispatches_renderer(tmp_path, monkeypatch) -> None:
    captured = {}
    fused = tmp_path / "fused.ome.zarr"
    fused.mkdir()
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    output = tmp_path / "overlay.png"

    def fake_render_fused_tile_index_overlay(**kwargs):
        captured.update(kwargs)
        return output

    monkeypatch.setattr(cli_module, "render_fused_tile_index_overlay", fake_render_fused_tile_index_overlay)

    result = CliRunner().invoke(
        app,
        [
            "fused-tile-index-qc",
            str(fused),
            "--registration-input",
            str(registration),
            "--output",
            str(output),
            "--level",
            "2",
            "--z-index",
            "7",
            "--no-labels",
            "--no-markers",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "fused_zarr": fused,
        "registration_input": registration,
        "output": output,
        "level": 2,
        "z_index": 7,
        "draw_labels": False,
        "draw_markers": False,
    }
    assert str(output.resolve()) in result.stdout


def test_ome_metadata_dumb_stitch_cli_dispatches_renderer(tmp_path, monkeypatch) -> None:
    captured = {}
    left = tmp_path / "L"
    right = tmp_path / "R"
    basic = tmp_path / "basic"
    output_dir = tmp_path / "qc"
    left.mkdir()
    right.mkdir()
    basic.mkdir()
    manifest = output_dir / "manifest.json"
    contact_sheet = output_dir / "contact.png"

    def fake_render_ome_metadata_dumb_stitch(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            manifest_path=manifest,
            contact_sheet_path=contact_sheet,
            output_paths=[manifest, contact_sheet],
        )

    monkeypatch.setattr(cli_module, "render_ome_metadata_dumb_stitch", fake_render_ome_metadata_dumb_stitch)

    result = CliRunner().invoke(
        app,
        [
            "ome-metadata-dumb-stitch",
            "--input-dir",
            f"L={left}",
            "--input-dir",
            f"R={right}",
            "--output-dir",
            str(output_dir),
            "--channels",
            "0,1",
            "--level",
            "2",
            "--write-tiff",
            "--basic-dir",
            str(basic),
        ],
    )

    assert result.exit_code == 0
    assert captured["input_dirs_by_view"] == {"L": left, "R": right}
    assert captured["output_dir"] == output_dir
    assert captured["channels"] == (0, 1)
    assert captured["basic_dir"] == basic
    assert captured["level"] == 2
    assert captured["center_z_index"] is None
    assert captured["output_prefix"] == "ome_metadata_dumb_stitch"
    assert captured["draw_tile_labels"] is False
    assert captured["draw_tile_outlines"] is False
    assert captured["write_tiff"] is True
    assert captured["progress"] == cli_module._log_progress
    payload = json.loads(result.stdout)
    assert payload["manifest"].endswith("manifest.json")
    assert payload["contact_sheet"].endswith("contact.png")


def test_live_fusion_preview_cli_dispatches_renderer(tmp_path, monkeypatch) -> None:
    captured = {}
    log = tmp_path / "fusion.log"
    log.write_text("streaming write to /tmp/fused.ome.zarr\n")
    output = tmp_path / "preview.png"

    def fake_render_live_fusion_preview(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output=output)

    monkeypatch.setattr(cli_module, "render_live_fusion_preview", fake_render_live_fusion_preview)

    result = CliRunner().invoke(
        app,
        [
            "live-fusion-preview",
            "--log",
            str(log),
            "--output",
            str(output),
            "--level",
            "1",
            "--channel",
            "2",
            "--color",
            "gray",
            "--stride",
            "3",
            "--z-start",
            "4",
            "--z-step",
            "5",
            "--panels",
            "2",
            "--high-percentile",
            "99.0",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "source_zarr": None,
        "log_path": log,
        "output": output,
        "output_dir": None,
        "level": 1,
        "channel": 2,
        "color": "gray",
        "stride": 3,
        "z_start": 4,
        "z_step": 5,
        "max_panels": 2,
        "high_percentile": 99.0,
    }
    assert str(output) in result.stdout


def test_tile_phase_align_exposes_affine_mode() -> None:
    result = CliRunner().invoke(app, ["tile-phase-align", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--alignment-mode" in result.stdout
    assert "--affine-init-z-samples" in result.stdout
    assert "--affine-fit-mode" in result.stdout
    assert "--affine-tile-order" in result.stdout
    assert "--affine-running-average-min-inliers" in result.stdout
    assert "200,480,480" in result.stdout
