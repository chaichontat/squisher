from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

import squisher_lightsheet.method8_stitch_register as stitch_register
import squisher_lightsheet.artifact_io as artifact_io
import squisher_lightsheet.registration_workflow as registration_workflow
from squisher_lightsheet.registration_workflow import (
    derive_registration_threshold,
    run_registration_workflow,
    write_canonical_registration,
)


def _write_tile(path: Path, data: np.ndarray) -> None:
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.create_array(
        "0",
        data=data,
        chunks=(1, 1, data.shape[-2], data.shape[-1]),
        dimension_names=("c", "z", "y", "x"),
    )
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 0.6, 0.3, 0.3]}
                        ],
                    }
                ],
            }
        ],
    }


def _write_positions(path: Path, zarr_dir: Path, names: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "units": "micrometer",
                "tiles": [
                    {
                        "tile": name,
                        "path": str(zarr_dir / name),
                        "translation_um": {"z": 0.0, "y": 0.0, "x": float(index * 1.2)},
                        "scale_um": {"z": 0.6, "y": 0.3, "x": 0.3},
                    }
                    for index, name in enumerate(names)
                ],
            }
        )
        + "\n"
    )


def _write_screen(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    return path


def test_registration_fingerprint_maps_tiff_names_to_zarr(tmp_path) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    zarr_name = "sample.000.ome.zarr"
    _write_tile(zarr_dir / zarr_name, np.ones((1, 3, 8, 8), dtype=np.uint16))
    positions = tmp_path / "positions.json"
    _write_positions(positions, zarr_dir, ["sample.000.ome.tif"])

    fingerprint = artifact_io.registration_input_fingerprint(positions, zarr_dir)

    assert fingerprint["tiles"][0]["store"]["path"] == str((zarr_dir / zarr_name).resolve())


def test_derive_registration_threshold_reads_czyx_level_zero(tmp_path) -> None:
    rng = np.random.default_rng(42)
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    names = ["sample.000.ome.zarr", "sample.001.ome.zarr"]
    for name in names:
        background = rng.normal(100, 8, size=(1, 3, 16, 8))
        foreground = rng.normal(1000, 20, size=(1, 3, 16, 8))
        data = np.concatenate([background, foreground], axis=-1).clip(0).astype(np.uint16)
        _write_tile(zarr_dir / name, data)
    positions = tmp_path / "positions.json"
    _write_positions(positions, zarr_dir, names)
    output = tmp_path / "threshold.json"

    result = derive_registration_threshold(
        position_json=positions,
        zarr_dir=zarr_dir,
        output=output,
        channel=0,
        method="minimum",
        sample_step_yx=1,
    )

    assert 120 < result.threshold < 900
    assert result.tile_count == 2
    assert result.sampled_value_count == 2 * 16 * 16
    payload = json.loads(output.read_text())
    assert payload["method"] == "skimage.filters.threshold_minimum"
    assert payload["source_level"] == 0
    assert payload["axes"] == "CZYX"


def test_write_canonical_registration_emits_fusion_contract(tmp_path) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    names = ["sample.000.ome.zarr", "sample.001.ome.zarr"]
    for name in names:
        _write_tile(zarr_dir / name, np.ones((1, 3, 8, 8), dtype=np.uint16))
    optimized = tmp_path / "optimized.json"
    _write_positions(optimized, zarr_dir, [name.replace(".ome.zarr", ".ome.tif") for name in names])
    summary = tmp_path / "measurements.json"
    summary.write_text(json.dumps({"settings": {"channel": 0, "z_chunks": 6}}) + "\n")
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(json.dumps({"connected_tile_count": 2, "tile_count": 2}) + "\n")
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps({"threshold": 250.0, "method": "test"}) + "\n")
    position_output = tmp_path / "registration.positions.json"
    registration_output = tmp_path / "registration.json"

    write_canonical_registration(
        optimized_position=optimized,
        zarr_dir=zarr_dir,
        position_output=position_output,
        registration_output=registration_output,
        measurement_summary=summary,
        diagnostics=diagnostics,
        threshold_record=threshold,
    )

    position_payload = json.loads(position_output.read_text())
    assert position_payload["registration_run"]["settings"] == {"channel": 0, "z_chunks": 6}
    assert "Method8" not in position_payload["registration_run"]["method"]
    assert position_payload["input_dir"] == str(zarr_dir.resolve())
    assert [tile["tile"] for tile in position_payload["tiles"]] == names
    assert [tile["path"] for tile in position_payload["tiles"]] == [
        str((zarr_dir / name).resolve()) for name in names
    ]
    assert position_payload["tiles"][0]["shape"] == [1, 3, 8, 8]
    assert position_payload["tiles"][0]["axes"] == "CZYX"
    assert position_payload["tiles"][1]["translation_um"] == {"z": 0.0, "y": 0.0, "x": 1.2}
    registration = json.loads(registration_output.read_text())
    assert registration["registered_transform_key"] == "registered_affine"
    assert [tile["tile"] for tile in registration["tiles"]] == names
    assert registration["tiles"][0]["shape"] == [1, 3, 8, 8]
    assert registration["tiles"][0]["axes"] == "CZYX"
    assert registration["tiles"][0]["registered_affine"]["matrix"] == np.eye(4).tolist()
    assert registration["metrics"]["registration_run"]["threshold"]["threshold"] == 250.0


def test_write_canonical_registration_refuses_existing_outputs(tmp_path) -> None:
    existing = tmp_path / "registration.json"
    existing.write_text("keep")

    with pytest.raises(FileExistsError, match="registration.json"):
        write_canonical_registration(
            optimized_position=tmp_path / "optimized.json",
            zarr_dir=tmp_path,
            position_output=tmp_path / "positions.json",
            registration_output=existing,
            measurement_summary=tmp_path / "summary.json",
            diagnostics=tmp_path / "diagnostics.json",
            threshold_record=tmp_path / "threshold.json",
        )

    assert existing.read_text() == "keep"


def test_write_canonical_registration_requires_explicit_connectivity(tmp_path) -> None:
    optimized = tmp_path / "optimized.json"
    optimized.write_text(json.dumps({"tiles": []}) + "\n")
    summary = tmp_path / "measurements.json"
    summary.write_text(json.dumps({"settings": {}}) + "\n")
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(json.dumps({}) + "\n")
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps({"threshold": 250.0}) + "\n")

    with pytest.raises(ValueError, match="tile_count"):
        write_canonical_registration(
            optimized_position=optimized,
            zarr_dir=tmp_path,
            position_output=tmp_path / "registration.positions.json",
            registration_output=tmp_path / "registration.json",
            measurement_summary=summary,
            diagnostics=diagnostics,
            threshold_record=threshold,
        )


def test_write_canonical_registration_allows_explicit_disconnected_override(tmp_path) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    names = ["sample.000.ome.zarr", "sample.001.ome.zarr"]
    for name in names:
        _write_tile(zarr_dir / name, np.ones((1, 3, 8, 8), dtype=np.uint16))
    optimized = tmp_path / "optimized.json"
    _write_positions(optimized, zarr_dir, names)
    summary = tmp_path / "measurements.json"
    summary.write_text(json.dumps({"settings": {"channel": 1}}) + "\n")
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(json.dumps({"tile_count": 2, "connected_tile_count": 1}) + "\n")
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps({"threshold": 90.0}) + "\n")
    position_output = tmp_path / "registration.positions.json"
    registration_output = tmp_path / "registration.json"

    with pytest.raises(ValueError, match="registration graph is not fully connected"):
        write_canonical_registration(
            optimized_position=optimized,
            zarr_dir=zarr_dir,
            position_output=position_output,
            registration_output=registration_output,
            measurement_summary=summary,
            diagnostics=diagnostics,
            threshold_record=threshold,
        )

    write_canonical_registration(
        optimized_position=optimized,
        zarr_dir=zarr_dir,
        position_output=position_output,
        registration_output=registration_output,
        measurement_summary=summary,
        diagnostics=diagnostics,
        threshold_record=threshold,
        allow_disconnected=True,
    )

    registration_run = json.loads(registration_output.read_text())["metrics"]["registration_run"]
    assert registration_run["connectivity"] == {
        "allow_disconnected": True,
        "tile_count": 2,
        "connected_tile_count": 1,
    }


def test_run_registration_workflow_rejects_summary_from_different_input(tmp_path) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    positions = tmp_path / "positions.json"
    _write_positions(positions, zarr_dir, [])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_screen(output_dir / "level2-screen.json")
    (output_dir / "registration.measurements.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "lightsheet.level0_phase_recovery_measurements.v1",
                "position_json": str(tmp_path / "other.positions.json"),
                "zarr_dir": str(zarr_dir),
                "settings": {"fixed_mask_threshold": 250.0, "channel": 0},
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="different position file"):
        run_registration_workflow(
            position_json=positions,
            zarr_dir=zarr_dir,
            output_dir=output_dir,
            threshold=250.0,
        )


def test_measurement_resume_rejects_position_content_changed_in_place(tmp_path) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    name = "sample.000.ome.zarr"
    _write_tile(zarr_dir / name, np.ones((1, 3, 8, 8), dtype=np.uint16))
    positions = tmp_path / "positions.json"
    _write_positions(positions, zarr_dir, [name])
    payload = {
        "artifact_type": "lightsheet.level0_phase_recovery_measurements.v1",
        "position_json": str(positions.resolve()),
        "zarr_dir": str(zarr_dir.resolve()),
        "input_fingerprint": artifact_io.registration_input_fingerprint(positions, zarr_dir),
        "settings": {"channel": 0},
    }
    _write_positions(positions, zarr_dir, [name])
    position_payload = json.loads(positions.read_text())
    position_payload["tiles"][0]["translation_um"]["x"] = 99.0
    positions.write_text(json.dumps(position_payload) + "\n")

    with pytest.raises(ValueError, match="input fingerprint differs"):
        registration_workflow._validate_measurement_summary(
            payload,
            artifact=tmp_path / "measurements.json",
            position_json=positions,
            zarr_dir=zarr_dir,
            expected_settings={"channel": 0},
        )


def test_optimization_resume_rejects_measurement_summary_changed_in_place(tmp_path) -> None:
    method8_summary = tmp_path / "measurements.json"
    method8_summary.write_text('{"rows": []}\n')
    position_json = tmp_path / "positions.json"
    position_json.write_text('{"tiles": []}\n')
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    outputs = {
        "positions": tmp_path / "optimized.json",
        "constraints_jsonl": tmp_path / "constraints.jsonl",
        "corrections": tmp_path / "corrections.json",
    }
    payload = {
        "artifact_type": "lightsheet.level0_phase_recovery_tile_optimization.v1",
        "method8_summary": str(method8_summary.resolve()),
        "method8_summary_sha256": artifact_io.sha256_file(method8_summary),
        "position_json": str(position_json.resolve()),
        "settings": {"zarr_dir": str(zarr_dir.resolve())},
        "tile_count": 0,
        "connected_tile_count": 0,
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    method8_summary.write_text('{"rows": [{"changed": true}]}\n')

    with pytest.raises(ValueError, match="measurement summary content differs"):
        registration_workflow._validate_optimization_diagnostics(
            payload,
            artifact=tmp_path / "optimization.json",
            method8_summary=method8_summary,
            position_json=position_json,
            zarr_dir=zarr_dir,
            expected_settings={},
            expected_outputs=outputs,
        )


def test_run_registration_workflow_rejects_non_human_threshold_record(tmp_path) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    positions = tmp_path / "positions.json"
    _write_positions(positions, zarr_dir, [])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_screen(output_dir / "level2-screen.json")
    (output_dir / "registration.threshold.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "lightsheet.registration_threshold.v1",
                "method": "skimage.filters.threshold_minimum",
                "threshold": 250.0,
                "sample_step_yx": 4,
                "channel": 0,
                "position_json": str(positions.resolve()),
                "zarr_dir": str(zarr_dir.resolve()),
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="threshold method"):
        run_registration_workflow(
            position_json=positions,
            zarr_dir=zarr_dir,
            output_dir=output_dir,
            threshold=250.0,
        )


def test_run_registration_workflow_rejects_threshold_record_without_value(tmp_path) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    positions = tmp_path / "positions.json"
    _write_positions(positions, zarr_dir, [])
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_screen(output_dir / "level2-screen.json")
    (output_dir / "registration.threshold.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "lightsheet.registration_threshold.v1",
                "method": "skimage.filters.threshold_minimum",
                "sample_step_yx": 4,
                "channel": 0,
                "position_json": str(positions.resolve()),
                "zarr_dir": str(zarr_dir.resolve()),
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="missing threshold"):
        run_registration_workflow(
            position_json=positions,
            zarr_dir=zarr_dir,
            output_dir=output_dir,
            threshold=250.0,
        )


def test_write_canonical_registration_rolls_back_partial_output_set(tmp_path, monkeypatch) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    name = "sample.000.ome.zarr"
    _write_tile(zarr_dir / name, np.ones((1, 3, 8, 8), dtype=np.uint16))
    optimized = tmp_path / "optimized.json"
    _write_positions(optimized, zarr_dir, [name])
    summary = tmp_path / "measurements.json"
    summary.write_text(json.dumps({"settings": {}}) + "\n")
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(json.dumps({"tile_count": 1, "connected_tile_count": 1}) + "\n")
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps({"threshold": 250.0}) + "\n")
    position_output = tmp_path / "registration.positions.json"
    registration_output = tmp_path / "registration.json"
    original_replace = Path.replace

    def fail_second_promotion(self: Path, target: Path) -> Path:
        if target == registration_output:
            raise OSError("forced registration promotion failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_second_promotion)

    with pytest.raises(OSError, match="forced registration promotion failure"):
        write_canonical_registration(
            optimized_position=optimized,
            zarr_dir=zarr_dir,
            position_output=position_output,
            registration_output=registration_output,
            measurement_summary=summary,
            diagnostics=diagnostics,
            threshold_record=threshold,
        )

    assert not position_output.exists()
    assert not registration_output.exists()


def test_run_registration_workflow_emits_complete_cli_contract(tmp_path, monkeypatch) -> None:
    zarr_dir = tmp_path / "tiles"
    zarr_dir.mkdir()
    names = ["sample.000.ome.zarr", "sample.001.ome.zarr"]
    for name in names:
        _write_tile(zarr_dir / name, np.ones((1, 3, 8, 8), dtype=np.uint16))
    positions = tmp_path / "positions.json"
    _write_positions(positions, zarr_dir, names)
    output_dir = tmp_path / "run"

    def fake_screen_level2_overlaps(**kwargs):
        return _write_screen(kwargs["output"])

    monkeypatch.setattr(
        registration_workflow,
        "screen_level2_overlaps",
        fake_screen_level2_overlaps,
    )

    def fake_register_level0_phase_recovery(**kwargs):
        output = kwargs["output_dir"]
        summary = output / "registration.measurements.json"
        summary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "lightsheet.level0_phase_recovery_measurements.v1",
                    "position_json": str(positions.resolve()),
                    "zarr_dir": str(zarr_dir.resolve()),
                    "input_fingerprint": artifact_io.registration_input_fingerprint(positions, zarr_dir),
                    "settings": {
                        "pairs": [],
                        "all_adjacent": True,
                        "channel": kwargs["channel"],
                        "z_chunks": kwargs["z_chunks"],
                        "device": kwargs["device"],
                        "method8": kwargs["method8"],
                        "max_iterations": kwargs["max_iterations"],
                        "ftol": kwargs["ftol"],
                        "min_corr": kwargs["min_corr"],
                        "min_grad_ncc": kwargs["min_grad_ncc"],
                        "fixed_mask_threshold": kwargs["fixed_mask_threshold"],
                        "fixed_mask_min_voxels": kwargs["fixed_mask_min_voxels"],
                        "fixed_mask_max_masked_fraction": kwargs["fixed_mask_max_masked_fraction"],
                        "phase_recovery_shifted_crop": True,
                        "phase_recovery_min_prior_edges_per_axis": kwargs[
                            "phase_recovery_min_prior_edges_per_axis"
                        ],
                        "phase_recovery_min_phase_grad": kwargs["min_phase_grad"],
                        "phase_recovery_min_phase_corr": kwargs["min_phase_corr"],
                        "native_lib_dir": str(kwargs["native_lib_dir"]),
                        "level2_screen": str(kwargs["level2_screen"].resolve()),
                        "level2_screen_sha256": artifact_io.sha256_file(kwargs["level2_screen"]),
                    }
                }
            )
            + "\n"
        )
        optimized = output / "registration.optimized.positions.json"
        optimized.write_text(positions.read_text())
        constraints = output / "registration.constraints.jsonl"
        constraints.write_text("")
        corrections = output / "registration.tile-corrections.json"
        corrections.write_text("{}\n")
        diagnostics = output / "registration.optimization.diagnostics.json"
        diagnostics.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "lightsheet.level0_phase_recovery_tile_optimization.v1",
                    "method8_summary": str(summary.resolve()),
                    "method8_summary_sha256": artifact_io.sha256_file(summary),
                    "position_json": str(positions.resolve()),
                    "settings": {
                        "zarr_dir": str(zarr_dir.resolve()),
                        "max_grad_regression": kwargs["max_grad_regression"],
                        "max_corr_regression": kwargs["max_corr_regression"],
                        "phase_fallback": kwargs["phase_fallback"],
                        "min_phase_grad": kwargs["min_phase_grad"],
                        "min_phase_corr": kwargs["min_phase_corr"],
                        "phase_fallback_weight_scale": kwargs["phase_fallback_weight_scale"],
                    },
                    "tile_count": 2,
                    "connected_tile_count": 2,
                    "outputs": {
                        "positions": str(optimized.resolve()),
                        "constraints_jsonl": str(constraints.resolve()),
                        "corrections": str(corrections.resolve()),
                    },
                }
            )
            + "\n"
        )
        return stitch_register.Method8RegistrationOutputs(
            method8_summary=summary,
            optimized_positions=optimized,
            diagnostics=diagnostics,
            constraints_jsonl=constraints,
            tile_corrections=corrections,
        )

    monkeypatch.setattr(
        stitch_register,
        "register_level0_phase_recovery",
        fake_register_level0_phase_recovery,
    )

    outputs = run_registration_workflow(
        position_json=positions,
        zarr_dir=zarr_dir,
        output_dir=output_dir,
        threshold=250.0,
        channel=0,
        z_chunks=6,
    )

    assert outputs.threshold_record.name == "registration.threshold.json"
    assert outputs.canonical_positions.name == "registration.positions.json"
    assert outputs.registration_json.name == "registration.json"
    threshold_record = json.loads(outputs.threshold_record.read_text())
    assert threshold_record["method"] == "human_reviewed_threshold"
    assert "reviewed_dumb_tiff" not in threshold_record
    assert "reviewed_dumb_tiff_sha256" not in threshold_record
    assert (output_dir / "level2-screen.json").is_file()
    assert json.loads(outputs.registration_json.read_text())["tiles"][0]["axes"] == "CZYX"
