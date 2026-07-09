from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import squisher_lightsheet.method8_stitch_register as stitch_register
import squisher_lightsheet.stitch_cli as stitch_cli
from squisher_lightsheet.method8_stitch_register import constraints_from_method8_summary
from squisher_lightsheet.stitch_cli import app


def _summary_row(
    *,
    status: str,
    method8_shift: list[float] | None,
    phase_shift: list[float],
    method8_grad: float | None,
    method8_corr: float | None,
    rejection_reason: str | None = None,
    corr_initial: float | None = 0.20,
) -> dict:
    return {
        "fixed_tile": "Image_14.000.ome.zarr",
        "moving_tile": "Image_14.001.ome.zarr",
        "seam_axis": "x",
        "status": status,
        "rejection_reason": rejection_reason,
        "fixed_slices_zyx": [[0, 10], [0, 10], [0, 10]],
        "moving_slices_zyx": [[0, 10], [0, 10], [0, 10]],
        "local_translation_zyx": method8_shift,
        "phase_shift_zyx": phase_shift,
        "gradient_component_ncc_initial_mean": 0.20,
        "gradient_component_ncc_method8_mean": method8_grad,
        "gradient_component_ncc_phase_mean": 0.50,
        "corr_initial": corr_initial,
        "corr_method8": method8_corr,
        "corr_phase": 0.60,
        "fixed_threshold_mask": {
            "source": {"unmasked_fraction": 0.5},
            "fit": {"unmasked_fraction": 0.5},
        },
        "fixed_content": {"std": 1.0},
        "moving_content": {"std": 1.0},
    }


def test_lightsheet_stitch_exposes_register_subcommand() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "register" in result.stdout

    register_help = CliRunner().invoke(app, ["register", "--help"])
    assert register_help.exit_code == 0
    assert "--threshold" in register_help.stdout
    assert "--fixed-mask-threshold" not in register_help.stdout
    assert "[default: 3000.0]" in register_help.stdout
    assert "--phase-fallback" in register_help.stdout
    assert "[default: 0.1]" in register_help.stdout


def test_constraints_use_median_method8_shift_per_edge() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 2500.0},
        "rows": [
            _summary_row(
                status="accepted",
                method8_shift=[1.0, 2.0, 3.0],
                phase_shift=[10.0, 10.0, 10.0],
                method8_grad=0.55,
                method8_corr=0.65,
            ),
            _summary_row(
                status="accepted",
                method8_shift=[3.0, 6.0, 9.0],
                phase_shift=[10.0, 10.0, 10.0],
                method8_grad=0.55,
                method8_corr=0.65,
            ),
            _summary_row(
                status="accepted",
                method8_shift=[100.0, 100.0, 100.0],
                phase_shift=[10.0, 10.0, 10.0],
                method8_grad=0.55,
                method8_corr=0.65,
            ),
        ],
    }

    constraints, rejections, sources = constraints_from_method8_summary(
        summary, tile_index={"000": 0, "001": 1}
    )

    assert not rejections
    assert sources["method8"] == 1
    assert constraints[0].shift_zyx == (3.0, 6.0, 9.0)
    assert constraints[0].source_label == "image14_method8_mask2500_phase_gated_median_n3"


def test_phase_fallback_is_downweighted_when_method8_edge_is_unusable() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 3000.0},
        "rows": [
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[4.0, 5.0, 6.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="low_corr_method8",
            ),
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[8.0, 9.0, 10.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="low_corr_method8",
            ),
        ],
    }

    constraints, rejections, sources = constraints_from_method8_summary(
        summary,
        tile_index={"000": 0, "001": 1},
        phase_fallback_weight_scale=0.1,
    )

    assert rejections["low_corr_method8"] == 2
    assert sources["phase_fallback"] == 1
    assert constraints[0].shift_zyx == (6.0, 7.0, 8.0)
    assert constraints[0].weight == pytest.approx(0.035)
    assert constraints[0].source_label == "image14_phase_fallback_mask3000_phase_gated_median_n2"


def test_phase_fallback_rejects_rows_with_missing_initial_correlation() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 3000.0},
        "rows": [
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[4.0, 5.0, 6.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="low_corr_method8",
                corr_initial=None,
            )
        ],
    }

    constraints, rejections, sources = constraints_from_method8_summary(
        summary,
        tile_index={"000": 0, "001": 1},
    )

    assert constraints == []
    assert rejections["low_corr_method8"] == 1
    assert sources["missing_edges_after_fallback"] == 1


def test_register_uses_threshold_in_default_method8_summary_path(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_measure_method8_zcoverage(**kwargs):
        captured["method8_output"] = kwargs["output"]
        captured["threshold"] = kwargs["fixed_mask_threshold"]
        return kwargs["output"]

    def fake_optimize_positions_from_method8_summary(**kwargs):
        captured["optimized_from"] = kwargs["method8_summary"]
        return stitch_register.Method8RegistrationOutputs(
            method8_summary=kwargs["method8_summary"],
            optimized_positions=tmp_path / "positions.json",
            diagnostics=tmp_path / "diagnostics.json",
            constraints_jsonl=tmp_path / "constraints.jsonl",
            tile_corrections=tmp_path / "corrections.json",
        )

    monkeypatch.setattr(stitch_register, "measure_method8_zcoverage", fake_measure_method8_zcoverage)
    monkeypatch.setattr(
        stitch_register,
        "optimize_positions_from_method8_summary",
        fake_optimize_positions_from_method8_summary,
    )
    monkeypatch.setattr(stitch_register, "DEFAULT_OUTPUT_PARENT_DIR", tmp_path)

    stitch_register.register_image14_method8(
        position_json=tmp_path / "positions.json",
        zarr_dir=tmp_path / "zarr",
        fixed_mask_threshold=2500.0,
    )

    expected_output_dir = tmp_path / "optimized-mask2500-phase-gated"
    assert captured["threshold"] == 2500.0
    assert (
        captured["method8_output"]
        == expected_output_dir / "image14_method8_all_adjacent_zcoverage_mask2500_summary.json"
    )
    assert captured["optimized_from"] == captured["method8_output"]


def test_register_uses_summary_threshold_for_default_output_dir(tmp_path, monkeypatch) -> None:
    captured = {}
    summary = tmp_path / "summary.json"
    summary.write_text('{"settings": {"fixed_mask_threshold": 2500.0}, "rows": []}\n')

    def fake_optimize_positions_from_method8_summary(**kwargs):
        captured.update(kwargs)
        return stitch_register.Method8RegistrationOutputs(
            method8_summary=kwargs["method8_summary"],
            optimized_positions=kwargs["output_dir"] / "positions.json",
            diagnostics=kwargs["output_dir"] / "diagnostics.json",
            constraints_jsonl=kwargs["output_dir"] / "constraints.jsonl",
            tile_corrections=kwargs["output_dir"] / "corrections.json",
        )

    monkeypatch.setattr(
        stitch_register,
        "optimize_positions_from_method8_summary",
        fake_optimize_positions_from_method8_summary,
    )
    monkeypatch.setattr(stitch_register, "DEFAULT_OUTPUT_PARENT_DIR", tmp_path)

    stitch_register.register_image14_method8(
        position_json=tmp_path / "positions.json",
        zarr_dir=tmp_path / "zarr",
        method8_summary=summary,
    )

    assert captured["method8_summary"] == summary
    assert captured["output_dir"] == tmp_path / "optimized-mask2500-phase-gated"


def test_cli_register_forwards_threshold_and_pairs(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_register_image14_method8(**kwargs):
        captured.update(kwargs)
        return stitch_register.Method8RegistrationOutputs(
            method8_summary=tmp_path / "summary.json",
            optimized_positions=tmp_path / "positions.json",
            diagnostics=tmp_path / "diagnostics.json",
            constraints_jsonl=tmp_path / "constraints.jsonl",
            tile_corrections=tmp_path / "corrections.json",
        )

    monkeypatch.setattr(stitch_cli, "register_image14_method8", fake_register_image14_method8)

    result = CliRunner().invoke(
        app,
        [
            "register",
            "--position-json",
            str(tmp_path / "positions.json"),
            "--zarr-dir",
            str(tmp_path / "zarr"),
            "--threshold",
            "2500",
            "--pair",
            "061-062",
        ],
    )

    assert result.exit_code == 0
    assert captured["fixed_mask_threshold"] == 2500.0
    assert captured["pairs"] == ("061-062",)
    assert captured["all_adjacent"] is False


def test_optimization_removes_stale_outputs_before_solver_failure(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    summary = tmp_path / "summary.json"
    summary.write_text('{"settings": {"fixed_mask_threshold": 2500.0}, "rows": []}\n')
    paths = stitch_register._optimized_output_paths(output_dir, "mask2500")
    for path in paths.values():
        path.write_text("stale")

    constraint = stitch_register.BoundaryConstraint(
        fixed=0,
        moving=1,
        pair=(0, 1),
        axis="x",
        patch_index=0,
        shift_zyx=(1.0, 2.0, 3.0),
        weight=1.0,
        correlation_before=0.1,
        correlation_after=0.2,
        improvement=0.1,
        fixed_nonzero_fraction=0.5,
        moving_nonzero_fraction=0.5,
        fixed_std=1.0,
        moving_std=1.0,
        accepted=True,
    )

    monkeypatch.setattr(
        stitch_register,
        "_load_position_tiles",
        lambda *_args, **_kwargs: (
            {},
            ["Image_14.000.ome.zarr", "Image_14.001.ome.zarr"],
            {"000": 0},
            np.ones(3),
            [],
        ),
    )
    monkeypatch.setattr(
        stitch_register,
        "constraints_from_method8_summary",
        lambda *_args, **_kwargs: ([constraint], {}, {"method8": 1}),
    )

    def fail_solver(*_args, **_kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(
        stitch_register.stitch_legacy, "solve_tile_corrections_with_multiview_stitcher", fail_solver
    )

    with pytest.raises(RuntimeError, match="solver failed"):
        stitch_register.optimize_positions_from_method8_summary(
            method8_summary=summary,
            position_json=tmp_path / "positions.json",
            zarr_dir=tmp_path / "zarr",
            output_dir=output_dir,
        )

    assert all(not path.exists() for path in paths.values())


def test_pyproject_exposes_lightsheet_stitch_entrypoint() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()

    assert 'lightsheet-stitch = "squisher_lightsheet.stitch_cli:main"' in pyproject
