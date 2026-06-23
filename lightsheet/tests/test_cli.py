from __future__ import annotations

import json

from typer.testing import CliRunner

from squisher_lightsheet.cli import app


def test_cli_exposes_lean_stitching_subcommands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "position",
        "rough-phase",
        "register",
        "fuse",
        "pyramid",
        "qc",
        "mvs-edge-audit",
        "subtract-channel",
        "track-z-diagnostics",
        "align-lr-dumb-stitch",
        "tile-phase-align",
        "run-tltr",
    ):
        assert command in result.stdout


def test_run_tltr_dry_run_reports_expected_paths(tmp_path) -> None:
    left = tmp_path / "TL"
    right = tmp_path / "TR"
    left.mkdir()
    right.mkdir()
    output_prefix = tmp_path / "sample"

    result = CliRunner().invoke(
        app,
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
    assert "sample.positions.json" in result.stdout
    assert "sample.roughPhase.positions.json" in result.stdout
    assert "level2_phase_corrected_zyx_yellowOverlay_ch0.png" in result.stdout
    assert "workflow_summary.json" in result.stdout


def test_align_lr_dumb_stitch_dry_run_reports_expected_paths(tmp_path) -> None:
    left = tmp_path / "TL"
    right = tmp_path / "TR"
    left.mkdir()
    right.mkdir()
    output_prefix = tmp_path / "sample"

    result = CliRunner().invoke(
        app,
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
    assert "sample.positions.json" in result.stdout
    assert "sample.roughPhase.positions.json" in result.stdout
    assert "level2_phase_corrected_zyx_yellowOverlay_ch0.png" in result.stdout
    assert "lr_dumb_stitch_alignment_summary.json" in result.stdout


def test_align_405_to_488_exposes_stage3_mattes_command() -> None:
    result = CliRunner().invoke(app, ["align-405-to-488", "--help"])

    assert result.exit_code == 0
    assert "measure-stage3-mattes" in result.stdout
    assert "render-candidate-grid" in result.stdout


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
