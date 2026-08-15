from __future__ import annotations

import json
from pathlib import Path

import pytest

from squisher_lightsheet.fusion_provenance import write_fusion_provenance


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _write_output_store(path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": axis, "type": "space"} for axis in ("z", "y", "x")],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [0.6, 0.3, 0.3]},
                            {"type": "translation", "translation": [1.0, 2.0, 3.0]},
                        ],
                    }
                ],
            }
        ],
    }
    zarr.open_array(
        str(path / "0"),
        mode="w",
        shape=(8, 12, 16),
        chunks=(4, 6, 8),
        dtype="uint16",
        zarr_format=3,
        dimension_names=("z", "y", "x"),
    )


def _write_deconv_store(path: Path, basic_profile: Path, scaling: Path) -> None:
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update(
        {
            "squisher_complete": True,
            "squisher_deconv": {
                "output_mode": "u16",
                "provenance": {
                    "tool": "squisher-deconv",
                    "created_utc": "2026-07-01T00:00:00+00:00",
                    "source": "/data/tile.ome.tif",
                    "run_settings": {
                        "channels": 2,
                        "halo": 20,
                        "slab_depth": 200,
                        "iterations": 1,
                        "output_mode": "u16",
                    },
                    "psfs": [{"path": "/data/psf.tif"}],
                    "basic_profiles": [{"path": str(basic_profile)}],
                    "scaling": {"path": str(scaling)},
                    "versions": {"squisher-deconv": "0.1.0"},
                },
                "source_metadata_hash": "historical-only",
                "source_metadata": {"ome_xml": "intentionally not copied"},
                "source_metadata_summary": {"raw_shape": [2, 8, 12, 16]},
                "source_ome": {"channel_names": ["638", "561"]},
                "storage": {"format": "OME-Zarr", "zarr_format": 3},
            },
        }
    )
    zarr.open_array(
        str(path / "0"),
        mode="w",
        shape=(2, 8, 12, 16),
        chunks=(1, 4, 6, 8),
        dtype="uint16",
        zarr_format=3,
        dimension_names=("c", "z", "y", "x"),
    )


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    basic_dir = tmp_path / "basic"
    basic_profile = basic_dir / "profile-ch0.pkl"
    basic_dir.mkdir()
    basic_profile.write_bytes(b"profile")
    _write_json(
        basic_dir / "fit-sampling.json",
        {
            "artifact_type": "squisher_deconv.basic_fit.v1",
            "basic_settings": {"autotune": True, "smoothness_flatfield": 7.0},
            "outputs": {"profiles": [str(basic_profile)]},
        },
    )

    scale_dir = tmp_path / "scale"
    scaling = _write_json(scale_dir / "scaling.json", {"offset": [1.0], "scale": [2.0]})
    _write_json(
        scale_dir / "sample-manifest.json",
        {
            "artifact_type": "squisher_deconv.sample_scale.v1",
            "sample_count": 12,
            "scaling": {"offset": [1.0], "scale": [2.0]},
        },
    )

    deconv = tmp_path / "deconv" / "tile.ome.zarr"
    _write_deconv_store(deconv, basic_profile, scaling)

    accepted_window = _write_json(
        tmp_path / "registration" / "windows" / "accepted.json",
        {
            "artifact_type": "lightsheet.fused_fixed_window.v1",
            "status": "accepted",
            "selected_attempt": "native_recovery",
            "fixed_threshold_mask": {"threshold": 50.0},
        },
    )
    rejected_window = _write_json(
        tmp_path / "registration" / "windows" / "rejected.json",
        {
            "artifact_type": "lightsheet.fused_fixed_window.v1",
            "status": "rejected",
            "rejection_reason": "fixed_threshold_mask_too_masked",
        },
    )
    summary = _write_json(
        tmp_path / "registration" / "summary.json",
        {
            "artifact_type": "lightsheet.fused_fixed_summary.v1",
            "aggregate": {"accepted_window_count": 99},
            "windows": [
                {"status": "accepted", "level0_json": str(accepted_window)},
                {"status": "rejected", "level0_json": str(rejected_window)},
            ],
        },
    )
    position = _write_json(
        tmp_path / "materialized" / "chunks.positions.json",
        {
            "artifact_type": "squisher_lightsheet.fused_fixed_overlapping_materialized_positions.v1",
            "units": "micrometer",
            "source_summary_input": str(summary),
            "materialization_grid": {"core_shape_zyx": [480, 480, 480], "window_shape_zyx": [528, 528, 528]},
            "tiles": [
                {
                    "tile": "chunk.ome.zarr",
                    "materialized_source_path": str(deconv),
                    "method8_window_json": str(accepted_window),
                }
            ],
        },
    )
    registration = _write_json(
        tmp_path / "materialized" / "chunks.registration.json",
        {
            "artifact_type": "squisher_lightsheet.fused_fixed_overlapping_materialized_registration.v1",
            "source_summary_input": str(summary),
            "materialization_grid": {"core_shape_zyx": [480, 480, 480], "window_shape_zyx": [528, 528, 528]},
            "tiles": [
                {
                    "tile": "chunk.ome.zarr",
                    "status": "accepted",
                    "materialized_source_path": str(deconv),
                    "method8_window_json": str(accepted_window),
                    "registered_affine": {"matrix": [[1.0, 0.0], [0.0, 1.0]]},
                }
            ],
        },
    )
    return position, registration, deconv, summary


def test_write_fusion_provenance_bundles_registration_and_preprocessing(tmp_path: Path) -> None:
    position, registration, deconv, summary = _fixture_inputs(tmp_path)
    output = tmp_path / "fused.ch0.ome.zarr"
    _write_output_store(output)

    index = write_fusion_provenance(
        output=output,
        input_dir=tmp_path / "materialized",
        position_input=position,
        registration_input=registration,
        channel=0,
        requested_settings={"fusion_level": 0, "output_chunksize_zyx": [4, 6, 8]},
        resolved_settings={"batch_size": 1, "output_chunksize_zyx": [4, 6, 8]},
    )

    root_payload = json.loads((output / "zarr.json").read_text())
    assert root_payload["attributes"]["squisher_complete"] is True
    assert root_payload["attributes"]["squisher_fusion"] == index
    assert index["recording_mode"] == "inline"
    assert index["status"] == "complete"
    assert index["workflow"] == "fused_fixed_cross_channel"
    assert index["registration_counts"] == {"accepted": 1, "rejected": 1}

    manifest = json.loads((output / "provenance" / "manifest.json").read_text())
    assert manifest["recording_mode"] == "inline"
    source_paths = {artifact["source_path"] for artifact in manifest["artifacts"]}
    assert str(position.resolve()) in source_paths
    assert str(registration.resolve()) in source_paths
    assert str(summary.resolve()) in source_paths
    assert len(manifest["preprocessing"]["deconvolution"]["sources"]) == 1
    deconv_record = manifest["preprocessing"]["deconvolution"]["sources"][0]
    assert deconv_record["path"] == str(deconv.resolve())
    assert "source_metadata" not in deconv_record
    assert manifest["preprocessing"]["deconvolution"]["cohorts"][0]["settings"]["run_settings"]["halo"] == 20
    basic_artifacts = [
        artifact for artifact in manifest["artifacts"] if artifact["role"] == "basic_fit_manifest"
    ]
    assert len(basic_artifacts) == 1

    assert all("sha256" not in artifact and "checksum" not in artifact for artifact in manifest["artifacts"])
    assert all(
        "sha256" not in artifact and "checksum" not in artifact
        for artifact in manifest["external_artifacts"]
    )


def test_missing_transitive_json_records_partial_coverage(tmp_path: Path) -> None:
    position, registration, _deconv, _summary = _fixture_inputs(tmp_path)
    payload = json.loads(registration.read_text())
    payload["source_summary_input"] = str(tmp_path / "missing-summary.json")
    registration.write_text(json.dumps(payload) + "\n")
    output = tmp_path / "fused.ch0.ome.zarr"
    _write_output_store(output)

    index = write_fusion_provenance(
        output=output,
        input_dir=tmp_path,
        position_input=position,
        registration_input=registration,
        channel=0,
        requested_settings={},
        resolved_settings={},
    )

    manifest = json.loads((output / "provenance" / "manifest.json").read_text())
    assert index["status"] == "partial"
    assert any(item["path"].endswith("missing-summary.json") for item in manifest["coverage"]["unresolved"])


def test_historical_nan_json_is_preserved_verbatim(tmp_path: Path) -> None:
    position, registration, _deconv, summary = _fixture_inputs(tmp_path)
    summary_payload = json.loads(summary.read_text())
    window_path = Path(summary_payload["windows"][0]["level0_json"])
    window_path.write_text('{"artifact_type":"legacy.window.v1","score":NaN}\n')
    output = tmp_path / "fused.ch0.ome.zarr"
    _write_output_store(output)

    index = write_fusion_provenance(
        output=output,
        input_dir=tmp_path,
        position_input=position,
        registration_input=registration,
        channel=0,
        requested_settings={},
        resolved_settings={},
    )

    assert index["status"] == "complete"
    manifest = json.loads((output / "provenance" / "manifest.json").read_text())
    artifact = next(item for item in manifest["artifacts"] if item["source_path"] == str(window_path.resolve()))
    assert (output / artifact["bundled_path"]).read_text() == window_path.read_text()
