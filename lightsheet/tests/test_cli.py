from __future__ import annotations

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
        "track-z-diagnostics",
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
    assert "sample.overlap25.positions.json" in result.stdout
    assert "sample.overlap25.roughPhase.positions.json" in result.stdout
    assert "workflow_summary.json" in result.stdout
