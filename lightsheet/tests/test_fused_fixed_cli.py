from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import squisher_lightsheet.cli as cli_module
import squisher_lightsheet.fused_fixed as fused_fixed
from squisher_lightsheet.complete_recovery import CompleteRecoverySettings
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
    for command in (
        "method6",
        "recover-native",
        "complete-recovery",
        "prefusion-qc",
        "interpolated-window-qc",
    ):
        assert command in result.stdout


def test_root_exposes_fused_middle_z_qc() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "fused-middle-z-qc" in result.stdout


def test_recover_native_cli_forwards_only_native_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.json"
    adjacency = tmp_path / "adjacency.json"
    sweep_runner = tmp_path / "sweep.py"
    for path in (summary, adjacency, sweep_runner):
        path.write_text("{}\n")
    output_dir = tmp_path / "recovered"
    captured = {}

    def fake_recover_native(**kwargs):
        captured.update(kwargs)
        return output_dir / "fused_fixed_method8_summary.json"

    monkeypatch.setattr(cli_module, "recover_native", fake_recover_native)
    result = CliRunner().invoke(
        app,
        [
            "cross-register",
            "recover-native",
            "--summary",
            str(summary),
            "--output-dir",
            str(output_dir),
            "--adjacency-json",
            str(adjacency),
            "--sweep-runner",
            str(sweep_runner),
            "--devices",
            "0,1",
            "--maximum-outlier-px",
            "4.5",
            "--maximum-refit-displacement-px",
            "9",
            "--source-cache-max-gib",
            "8",
            "--rerun-all-eligible",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "summary": summary,
        "output_dir": output_dir,
        "maximum_outlier_px": 4.5,
        "maximum_refit_displacement_px": 9.0,
        "devices": (0, 1),
        "adjacency_json": adjacency,
        "tile_filter": None,
        "rerun_all_eligible": True,
        "source_cache_max_gib": 8.0,
        "sweep_runner": sweep_runner,
        "manual_initializations": None,
        "manual_initializations_channel": None,
        "dry_run": False,
        "resume": True,
    }
    assert json.loads(result.stdout) == {
        "summary": str(output_dir / "fused_fixed_method8_summary.json")
    }


def test_complete_recovery_cli_forwards_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_summary = tmp_path / "native.json"
    native_summary.write_text("{}\n")
    output_dir = tmp_path / "complete"
    captured = {}

    def fake_complete_recovery(**kwargs):
        captured.update(kwargs)
        return output_dir / "fused_fixed_method8_summary.json"

    monkeypatch.setattr(cli_module, "complete_recovery", fake_complete_recovery)
    result = CliRunner().invoke(
        app,
        [
            "cross-register",
            "complete-recovery",
            "--native-summary",
            str(native_summary),
            "--output-dir",
            str(output_dir),
            "--maximum-outlier-px",
            "4.25",
            "--minimum-linear-samples",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "native_summary": native_summary,
        "output_dir": output_dir,
        "settings": CompleteRecoverySettings(
            minimum_linear_samples=10,
            maximum_outlier_px=4.25,
        ),
    }
    assert json.loads(result.stdout) == {
        "summary": str(output_dir / "fused_fixed_method8_summary.json")
    }


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
            "--fit-intensity-transform",
            "linear",
            "--min-grad-ncc",
            "0.20",
            "--starting-affine-matrix-zyx",
            "1,0.01,0,0,1,0,0,0,1",
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
    assert captured["fit_intensity_transform"] == "linear"
    assert captured["min_grad_ncc"] == 0.20
    assert captured["starting_affine_matrix_zyx"] == (
        1.0,
        0.01,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert captured["fixed_mask_level"] == 2
    assert captured["devices"] == (0, 1)
    assert json.loads(result.stdout) == {
        "summary": str(output_dir / fused_fixed.SUMMARY_NAME),
        "registration": str(output_registration),
    }


@pytest.mark.parametrize(
    "matrix",
    [
        "1,0,0,0,1,0,0,0",
        "1,0,0,0,0,0,0,0,1",
        "1,0,0,0,1,0,0,0,nan",
    ],
)
def test_method6_cli_rejects_invalid_starting_affine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix: str
) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "run_fused_fixed_method6",
        lambda **_kwargs: pytest.fail("invalid affine reached the Method 6 owner"),
    )

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
            str(tmp_path / "fit"),
            "--output-registration",
            str(tmp_path / "registration.json"),
            "--fixed-mask-threshold",
            "50",
            "--source-label",
            "405",
            "--target-label",
            "561",
            "--starting-affine-matrix-zyx",
            matrix,
        ],
    )

    assert result.exit_code == 2
    assert "starting affine" in (result.stdout + result.stderr).lower()


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
                "fit_intensity_transform": "linear",
                "starting_affine_matrix_zyx": [
                    1.0,
                    0.01,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "min_grad_ncc": 0.20,
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
        fit_intensity_transform="linear",
        starting_affine_matrix_zyx=(1.0, 0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        min_grad_ncc=0.20,
        sweep_runner=paths["sweep_runner"],
        native_lib_dir=paths["native_lib_dir"],
        workers=2,
        devices=(0, 1),
    )

    assert captured_command[captured_command.index("--native-method") + 1] == "method6"
    assert captured_command[captured_command.index("--fit-intensity-transform") + 1] == "linear"
    assert captured_command[captured_command.index("--level0-initializer") + 1] == "level2-method8"
    assert captured_command[captured_command.index("--starting-affine-matrix-zyx") + 1] == (
        "1.0,0.01,0.0,0.0,1.0,0.0,0.0,0.0,1.0"
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


def test_method6_runner_composes_fixed_fusion_registration_into_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    tile = {
        "tile": "tile.000.ome.zarr",
        "path": str(tmp_path / "tile.000.ome.zarr"),
        "axes": "CZYX",
        "shape": [1, 10, 20, 30],
        "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
        "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
    }
    paths["fixed_position"].write_text(json.dumps({"tiles": [tile]}))
    moving_tile = json.loads(json.dumps(tile))
    moving_tile["translation_um"]["x"] = 2.0
    paths["moving_position"].write_text(json.dumps({"tiles": [moving_tile]}))
    fixed_registration = tmp_path / "fixed.registration.json"
    fixed_registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": tile["tile"],
                        "stage_translation_um": tile["translation_um"],
                        "registered_affine": {
                            "matrix": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 100.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        },
                    }
                ]
            }
        )
    )
    provenance = paths["fixed_fused"] / "provenance"
    provenance.mkdir()
    (provenance / "manifest.json").write_text(
        json.dumps({"inputs": {"registration_input": str(fixed_registration)}})
    )
    output_dir = tmp_path / "fit"
    output_registration = tmp_path / "registration.json"
    effective_paths = {}

    def fake_subprocess_run(command, **kwargs):
        assert kwargs == {"check": True}
        effective_fixed = Path(command[command.index("--fixed-position") + 1])
        effective_moving = Path(command[command.index("--moving-position") + 1])
        effective_paths.update(fixed=effective_fixed, moving=effective_moving)
        output_dir.mkdir(exist_ok=True)
        (output_dir / "window_json").mkdir()
        (output_dir / fused_fixed.SUMMARY_NAME).write_text(
            json.dumps(
                {
                    "cache_config": {
                        "native_method": "method6",
                        "fit_intensity_transform": "log1p",
                        "starting_affine_matrix_zyx": [
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        "min_grad_ncc": 0.24,
                        "level0_initializer": "level2-method8",
                        "moving_channel": 0,
                        "fixed_mask_threshold": 50.0,
                        "moving_position": str(effective_moving),
                        "moving_source_position": str(paths["moving_source_position"].resolve()),
                        "fixed_fused": str(paths["fixed_fused"].resolve()),
                    }
                }
            )
        )

    monkeypatch.setattr(fused_fixed.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        fused_fixed,
        "write_global_channel_affine_registration",
        lambda **_kwargs: output_registration.resolve(),
    )
    fused_fixed.run_fused_fixed_method6(
        fixed_position=paths["fixed_position"],
        moving_position=paths["moving_position"],
        moving_source_position=paths["moving_source_position"],
        fixed_fused=paths["fixed_fused"],
        output_dir=output_dir,
        output_registration=output_registration,
        moving_channel=0,
        fixed_mask_threshold=50.0,
        source_label="514",
        target_label="561",
        sweep_runner=paths["sweep_runner"],
        native_lib_dir=paths["native_lib_dir"],
    )

    assert effective_paths["fixed"] != paths["fixed_position"]
    assert effective_paths["moving"] != paths["moving_position"]
    effective_fixed = json.loads(effective_paths["fixed"].read_text())
    effective_moving = json.loads(effective_paths["moving"].read_text())
    assert effective_fixed["tiles"][0]["translation_um"]["x"] == pytest.approx(100.0)
    assert effective_moving["tiles"][0]["translation_um"]["x"] == pytest.approx(102.0)
    assert effective_fixed["fused_coordinate_adaptation"]["registration_input"] == str(
        fixed_registration
    )


def test_fusion_registration_resolves_artifact_without_inputs(tmp_path: Path) -> None:
    fixed_fused = tmp_path / "fixed.ome.zarr"
    provenance = fixed_fused / "provenance"
    provenance.mkdir(parents=True)
    registration = tmp_path / "fixed.registration.json"
    registration.write_text("{}\n")
    (provenance / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "role": "registration_input",
                        "source_path": str(registration),
                    }
                ]
            }
        )
    )

    assert fused_fixed._registration_from_fusion_provenance(fixed_fused) == (
        registration.resolve()
    )


def test_method6_uses_fixed_frame_affine_for_disjoint_moving_tiles(tmp_path: Path) -> None:
    fixed_position = tmp_path / "fixed.positions.json"
    moving_position = tmp_path / "moving.positions.json"
    fixed_fused = tmp_path / "fixed.ome.zarr"
    output_dir = tmp_path / "fit"
    fixed_tile = {
        "tile": "fixed.000.ome.zarr",
        "path": str(tmp_path / "fixed.000.ome.zarr"),
        "axes": "CZYX",
        "shape": [1, 10, 20, 30],
        "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
        "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
    }
    moving_tile = {
        **fixed_tile,
        "tile": "moving.000.ome.zarr",
        "path": str(tmp_path / "moving.000.ome.zarr"),
        "translation_um": {"z": 0.0, "y": 0.0, "x": 2.0},
    }
    fixed_position.write_text(json.dumps({"tiles": [fixed_tile]}))
    moving_position.write_text(json.dumps({"tiles": [moving_tile]}))
    registration = tmp_path / "fixed.registration.json"
    registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": fixed_tile["tile"],
                        "stage_translation_um": fixed_tile["translation_um"],
                        "registered_affine": {
                            "matrix": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 100.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        },
                    },
                    {
                        "tile": moving_tile["tile"],
                        "stage_translation_um": moving_tile["translation_um"],
                        "registered_affine": {
                            "matrix": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        },
                    },
                ]
            }
        )
    )
    provenance = fixed_fused / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "manifest.json").write_text(
        json.dumps({"inputs": {"registration_input": str(registration)}})
    )

    effective_fixed_path, effective_moving_path = fused_fixed._prepare_fused_coordinate_positions(
        fixed_position=fixed_position,
        moving_position=moving_position,
        fixed_fused=fixed_fused,
        output_dir=output_dir,
    )

    effective_fixed = json.loads(effective_fixed_path.read_text())
    effective_moving = json.loads(effective_moving_path.read_text())
    assert effective_fixed["tiles"][0]["translation_um"]["x"] == pytest.approx(100.0)
    assert effective_moving["tiles"][0]["translation_um"]["x"] == pytest.approx(102.0)
    assert effective_moving["fused_coordinate_adaptation"]["affine_source"] == (
        "common_fixed_frame_registered_affine"
    )


def test_method6_rejects_disjoint_tiles_without_common_fixed_frame_affine(
    tmp_path: Path,
) -> None:
    fixed_position = tmp_path / "fixed.positions.json"
    moving_position = tmp_path / "moving.positions.json"
    fixed_fused = tmp_path / "fixed.ome.zarr"
    fixed_tiles = []
    registration_tiles = []
    for index, shift in enumerate((100.0, 110.0)):
        tile = {
            "tile": f"fixed.{index:03d}.ome.zarr",
            "path": str(tmp_path / f"fixed.{index:03d}.ome.zarr"),
            "axes": "CZYX",
            "shape": [1, 10, 20, 30],
            "translation_um": {"z": 0.0, "y": 0.0, "x": float(index * 30)},
            "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        }
        fixed_tiles.append(tile)
        registration_tiles.append(
            {
                "tile": tile["tile"],
                "stage_translation_um": tile["translation_um"],
                "registered_affine": {
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, shift],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
            }
        )
    moving_tile = {
        **fixed_tiles[0],
        "tile": "moving.000.ome.zarr",
        "path": str(tmp_path / "moving.000.ome.zarr"),
    }
    fixed_position.write_text(json.dumps({"tiles": fixed_tiles}))
    moving_position.write_text(json.dumps({"tiles": [moving_tile]}))
    registration = tmp_path / "fixed.registration.json"
    registration.write_text(json.dumps({"tiles": registration_tiles}))
    provenance = fixed_fused / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "manifest.json").write_text(
        json.dumps({"inputs": {"registration_input": str(registration)}})
    )

    with pytest.raises(ValueError, match="no single fused-frame transform"):
        fused_fixed._prepare_fused_coordinate_positions(
            fixed_position=fixed_position,
            moving_position=moving_position,
            fixed_fused=fixed_fused,
            output_dir=tmp_path / "fit",
        )


def test_method6_uses_canonical_fixed_tile_affines_across_acquisitions(tmp_path: Path) -> None:
    fixed_position = tmp_path / "fixed.positions.json"
    moving_position = tmp_path / "moving.positions.json"
    fixed_fused = tmp_path / "fixed.ome.zarr"
    fixed_tiles = []
    moving_tiles = []
    registration_tiles = []
    for index, shift in enumerate((100.0, 110.0)):
        stage_x = float(index * 30)
        fixed = {
            "tile": f"sample-TL-514.{index:03d}.ome.zarr",
            "path": str(tmp_path / f"fixed.{index:03d}.ome.zarr"),
            "source_view": "L",
            "axes": "CZYX",
            "shape": [1, 10, 20, 30],
            "translation_um": {"z": 0.0, "y": 0.0, "x": stage_x},
            "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        }
        moving = {
            **fixed,
            "tile": f"sample-TL-488.{index:03d}.ome.zarr",
            "path": str(tmp_path / f"moving.{index:03d}.ome.zarr"),
            "translation_um": {"z": 0.0, "y": 0.0, "x": stage_x + 2.0},
        }
        fixed_tiles.append(fixed)
        moving_tiles.append(moving)
        registration_tiles.append(
            {
                **fixed,
                "stage_translation_um": fixed["translation_um"],
                "registered_affine": {
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, shift],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
            }
        )
    fixed_position.write_text(json.dumps({"tiles": fixed_tiles}))
    moving_position.write_text(json.dumps({"tiles": moving_tiles}))
    registration = tmp_path / "fixed.registration.json"
    registration.write_text(json.dumps({"tiles": registration_tiles}))
    provenance = fixed_fused / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "manifest.json").write_text(
        json.dumps({"inputs": {"registration_input": str(registration)}})
    )

    _, effective_moving_path = fused_fixed._prepare_fused_coordinate_positions(
        fixed_position=fixed_position,
        moving_position=moving_position,
        fixed_fused=fixed_fused,
        output_dir=tmp_path / "fit",
    )

    effective_moving = json.loads(effective_moving_path.read_text())
    assert [row["translation_um"]["x"] for row in effective_moving["tiles"]] == pytest.approx(
        [102.0, 142.0]
    )
    assert effective_moving["fused_coordinate_adaptation"]["affine_source"] == (
        "matching_canonical_fixed_tile_registered_affine"
    )


@pytest.mark.parametrize("with_template", [True, False])
def test_method6_resolves_materialized_fixed_fusion_frame_through_grid_template(
    tmp_path: Path, with_template: bool
) -> None:
    fixed_position = tmp_path / "fixed.positions.json"
    moving_position = tmp_path / "moving.positions.json"
    tile = {
        "tile": "sample-CL.000.ome.zarr",
        "path": str(tmp_path / "sample-CL.000.ome.zarr"),
        "source_view": "L",
        "axes": "CZYX",
        "shape": [1, 10, 20, 30],
        "translation_um": {"z": 0.0, "y": 0.0, "x": 5.0},
        "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
    }
    fixed_position.write_text(json.dumps({"tiles": [tile]}))
    moving_position.write_text(
        json.dumps({"tiles": [tile | {"translation_um": {"z": 0.0, "y": 0.0, "x": 7.0}}]})
    )

    def write_fusion(path: Path, registration_tiles: list[dict], inputs: dict) -> None:
        registration = path.with_suffix(".registration.json")
        registration.write_text(json.dumps({"tiles": registration_tiles}))
        (path / "provenance").mkdir(parents=True)
        (path / "provenance" / "manifest.json").write_text(
            json.dumps({"inputs": inputs | {"registration_input": str(registration)}})
        )

    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    shifted = [row.copy() for row in identity]
    shifted[2][3] = 3.0
    template = tmp_path / "template.ome.zarr"
    write_fusion(
        template,
        [
            {
                "tile": tile["tile"],
                "stage_translation_um": tile["translation_um"],
                "registered_affine": {"matrix": identity},
            }
        ],
        {"output_grid_template": None},
    )
    fixed_fused = tmp_path / "materialized-fused.ome.zarr"
    write_fusion(
        fixed_fused,
        [
            {
                "tile": "sample-CL.000.qy0_qx0.z00000.ome.zarr",
                "moving_tile": tile["tile"],
                "stage_translation_um": tile["translation_um"],
                "registered_affine": {"matrix": shifted},
            }
        ],
        {"output_grid_template": {"path": str(template), "level": 0} if with_template else None},
    )

    if not with_template:
        with pytest.raises(ValueError, match="missing fixed tiles"):
            fused_fixed._prepare_fused_coordinate_positions(
                fixed_position=fixed_position,
                moving_position=moving_position,
                fixed_fused=fixed_fused,
                output_dir=tmp_path / "fit",
            )
        return

    assert fused_fixed._prepare_fused_coordinate_positions(
        fixed_position=fixed_position,
        moving_position=moving_position,
        fixed_fused=fixed_fused,
        output_dir=tmp_path / "fit",
    ) == (fixed_position, moving_position)


def test_method6_rejects_ambiguous_fixed_coordinate_frame(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    tile = {
        "tile": "tile.000.ome.zarr",
        "path": str(tmp_path / "tile.000.ome.zarr"),
        "axes": "CZYX",
        "shape": [1, 10, 20, 30],
        "translation_um": {"z": 0.0, "y": 0.0, "x": 5.0},
        "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
    }
    paths["fixed_position"].write_text(json.dumps({"tiles": [tile]}))
    paths["moving_position"].write_text(json.dumps({"tiles": [tile]}))
    fixed_registration = tmp_path / "fixed.registration.json"
    fixed_registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": tile["tile"],
                        "stage_translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                        "registered_affine": {
                            "matrix": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 100.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        },
                    }
                ]
            }
        )
    )
    provenance = paths["fixed_fused"] / "provenance"
    provenance.mkdir()
    (provenance / "manifest.json").write_text(
        json.dumps({"inputs": {"registration_input": str(fixed_registration)}})
    )

    with pytest.raises(ValueError, match="neither registration-stage coordinates"):
        fused_fixed.run_fused_fixed_method6(
            fixed_position=paths["fixed_position"],
            moving_position=paths["moving_position"],
            moving_source_position=paths["moving_source_position"],
            fixed_fused=paths["fixed_fused"],
            output_dir=tmp_path / "fit",
            output_registration=tmp_path / "registration.json",
            moving_channel=0,
            fixed_mask_threshold=50.0,
            source_label="514",
            target_label="561",
            sweep_runner=paths["sweep_runner"],
            native_lib_dir=paths["native_lib_dir"],
        )


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


def test_method6_summary_rejects_wrong_starting_affine(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    summary = tmp_path / fused_fixed.SUMMARY_NAME
    summary.write_text(
        json.dumps(
            {
                "cache_config": {
                    "native_method": "method6",
                    "fit_intensity_transform": "log1p",
                    "starting_affine_matrix_zyx": [
                        1.0,
                        0.01,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "min_grad_ncc": 0.24,
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

    with pytest.raises(ValueError, match="starting_affine_matrix_zyx"):
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
