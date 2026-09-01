from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import squisher_lightsheet.cli as cli_module
import squisher_lightsheet.fused_fixed as fused_fixed
from squisher_lightsheet.cli import app


def _inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "fixed_position": tmp_path / "fixed.positions.json",
        "moving_position": tmp_path / "moving.positions.json",
        "moving_source_position": tmp_path / "moving.registration.json",
        "fixed_fused": tmp_path / "fixed.ome.zarr",
        "sweep_runner": tmp_path / "sweep.py",
        "native_lib_dir": tmp_path / "native",
    }
    for key, path in paths.items():
        if key in {"fixed_fused", "native_lib_dir"}:
            path.mkdir()
        else:
            path.write_text("{}\n")
    return paths


def test_cross_register_exposes_method6() -> None:
    result = CliRunner().invoke(app, ["cross-register", "--help"])

    assert result.exit_code == 0
    assert "method6" in result.stdout


def test_method6_cli_forwards_canonical_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _inputs(tmp_path)
    output_dir = tmp_path / "fit"
    output_registration = tmp_path / "registration.json"
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return output_dir / fused_fixed.SUMMARY_NAME, output_registration

    monkeypatch.setattr(cli_module, "run_fused_fixed_method6", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "cross-register",
            "method6",
            "--fixed-position",
            str(paths["fixed_position"]),
            "--moving-position",
            str(paths["moving_position"]),
            "--moving-source-position",
            str(paths["moving_source_position"]),
            "--fixed-fused",
            str(paths["fixed_fused"]),
            "--output-dir",
            str(output_dir),
            "--output-registration",
            str(output_registration),
            "--fixed-mask-threshold",
            "50",
            "--source-label",
            "638",
            "--target-label",
            "571",
            "--moving-channel",
            "0",
            "--sweep-runner",
            str(paths["sweep_runner"]),
            "--native-lib-dir",
            str(paths["native_lib_dir"]),
            "--workers",
            "2",
            "--devices",
            "0,1",
        ],
    )

    assert result.exit_code == 0
    assert captured["fixed_mask_threshold"] == 50.0
    assert captured["moving_channel"] == 0
    assert captured["source_label"] == "638"
    assert captured["target_label"] == "571"
    assert captured["core_shape_zyx"] == (480, 480, 480)
    assert captured["window_shape_zyx"] == (528, 528, 528)
    assert captured["fit_downsample_zyx"] == (1, 1, 1)
    assert captured["fixed_mask_level"] == 2
    assert captured["devices"] == (0, 1)
    assert json.loads(result.stdout) == {
        "summary": str(output_dir / fused_fixed.SUMMARY_NAME),
        "registration": str(output_registration),
    }


def test_method6_runner_locks_mode_and_aggregates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _inputs(tmp_path)
    output_dir = tmp_path / "fit"
    output_registration = tmp_path / "registration.json"
    captured_command: list[str] = []
    captured_affine = {}

    def fake_subprocess_run(command, **kwargs):
        captured_command.extend(command)
        assert kwargs == {"check": True}
        output_dir.mkdir()
        (output_dir / "window_json").mkdir()
        summary = {
            "cache_config": {
                "native_method": "method6",
                "fit_intensity_transform": "log1p",
                "level0_initializer": "level2-method8",
                "moving_channel": 1,
                "fixed_mask_threshold": 50.0,
                "moving_position": str(paths["moving_position"].resolve()),
                "moving_source_position": str(paths["moving_source_position"].resolve()),
                "fixed_fused": str(paths["fixed_fused"].resolve()),
            }
        }
        (output_dir / fused_fixed.SUMMARY_NAME).write_text(json.dumps(summary))

    def fake_write(**kwargs):
        captured_affine.update(kwargs)
        return output_registration.resolve()

    monkeypatch.setattr(fused_fixed.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(fused_fixed, "write_global_channel_affine_registration", fake_write)
    summary, registration = fused_fixed.run_fused_fixed_method6(
        fixed_position=paths["fixed_position"],
        moving_position=paths["moving_position"],
        moving_source_position=paths["moving_source_position"],
        fixed_fused=paths["fixed_fused"],
        output_dir=output_dir,
        output_registration=output_registration,
        moving_channel=1,
        fixed_mask_threshold=50.0,
        source_label="638",
        target_label="561",
        sweep_runner=paths["sweep_runner"],
        native_lib_dir=paths["native_lib_dir"],
        workers=2,
        devices=(0, 1),
    )

    assert captured_command[captured_command.index("--native-method") + 1] == "method6"
    assert captured_command[captured_command.index("--fit-intensity-transform") + 1] == "log1p"
    assert captured_command[captured_command.index("--level0-initializer") + 1] == "level2-method8"
    assert captured_command[captured_command.index("--starting-affine-matrix-zyx") + 1] == (
        "1,0,0,0,1,0,0,0,1"
    )
    assert captured_affine == {
        "window_dir": output_dir / "window_json",
        "reference_registration_input": paths["moving_source_position"],
        "output_registration": output_registration,
        "expected_moving_channel": 1,
        "expected_fixed_fused": paths["fixed_fused"],
        "source_label": "638",
        "target_label": "561",
    }
    assert summary == (output_dir / fused_fixed.SUMMARY_NAME).resolve()
    assert registration == output_registration.resolve()


def test_method6_summary_rejects_wrong_intensity_transform(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    summary = tmp_path / fused_fixed.SUMMARY_NAME
    summary.write_text(
        json.dumps(
            {
                "cache_config": {
                    "native_method": "method6",
                    "fit_intensity_transform": "linear",
                    "level0_initializer": "level2-method8",
                    "moving_channel": 0,
                    "fixed_mask_threshold": 50.0,
                    "moving_position": str(paths["moving_position"]),
                    "moving_source_position": str(paths["moving_source_position"]),
                    "fixed_fused": str(paths["fixed_fused"]),
                }
            }
        )
    )

    with pytest.raises(ValueError, match="fit_intensity_transform"):
        fused_fixed.validate_method6_summary(
            summary,
            moving_position=paths["moving_position"],
            moving_source_position=paths["moving_source_position"],
            fixed_fused=paths["fixed_fused"],
            moving_channel=0,
            fixed_mask_threshold=50.0,
        )


def test_method6_summary_does_not_fallback_outside_cache_config(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    summary = tmp_path / fused_fixed.SUMMARY_NAME
    summary.write_text(
        json.dumps(
            {
                "native_method": "method6",
                "cache_config": {
                    "fit_intensity_transform": "log1p",
                    "level0_initializer": "level2-method8",
                    "moving_channel": 0,
                    "fixed_mask_threshold": 50.0,
                    "moving_position": str(paths["moving_position"]),
                    "moving_source_position": str(paths["moving_source_position"]),
                    "fixed_fused": str(paths["fixed_fused"]),
                },
            }
        )
    )

    with pytest.raises(ValueError, match="native_method"):
        fused_fixed.validate_method6_summary(
            summary,
            moving_position=paths["moving_position"],
            moving_source_position=paths["moving_source_position"],
            fixed_fused=paths["fixed_fused"],
            moving_channel=0,
            fixed_mask_threshold=50.0,
        )
