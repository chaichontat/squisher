from __future__ import annotations

import hashlib
from pathlib import Path
import json

import numpy as np
import pytest
import zarr

from squisher_lightsheet.cross_dataset_match import compose_base_corrections, cross_dataset_match
from squisher_lightsheet.registered_view_correction import (
    build_registered_view_correction,
    load_registered_view_correction,
)
from squisher_lightsheet.residual_correction import _source_identity, load_residual_corrections


def _payload(channel: int, shared, rows, *, global_scale: float):
    return {
        "schema_version": 3,
        "kind": "squisher.residual-correction",
        "channel": channel,
        "coordinate": "normalized-original-raw-zyx",
        "basis": "cosine-xy2-z-scaled",
        "coefficient": shared,
        "global_scale": global_scale,
        "sources": rows,
    }


def _row(path: Path, *, gain: float, coefficient):
    return {
        "path": str(path.resolve()),
        "source_sha256": f"sha-{path.name}",
        "shape_zyx": [7, 9, 11],
        "gain": gain,
        "coefficient": coefficient,
        "training_pairs": 4,
    }


def _write_zarr(
    path: Path,
    data: np.ndarray,
    axes: str,
    scale: tuple[float, ...] | None = None,
) -> None:
    root = zarr.open_group(path, mode="w-")
    root.create_array("0", data=data, chunks=data.shape, dimension_names=list(axes))
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": axis, "type": "channel" if axis == "c" else "space"} for axis in axes],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": list(scale or (1.0,) * len(axes))},
                            {"type": "translation", "translation": [0.0] * len(axes)},
                        ],
                    }
                ],
            }
        ],
    }


def test_compose_base_corrections_preserves_each_view(tmp_path: Path) -> None:
    tl_sources = [tmp_path / "tl-0.ome.zarr", tmp_path / "tl-1.ome.zarr"]
    tr_sources = [tmp_path / "tr-0.ome.zarr", tmp_path / "tr-1.ome.zarr"]
    records = [
        {
            "path": str(source),
            "tile": source.name,
            "source_view": view,
            "shape": [2, 7, 9, 11],
        }
        for view, sources in (("TL", tl_sources), ("TR", tr_sources))
        for source in sources
    ]
    tl_shared = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    tr_shared = [[-0.2, 0.1, 0.0, 0.2, -0.1], [0.05, 0.0, 0.0, 0.0, 0.0]]
    tl_rows = [
        _row(source, gain=gain, coefficient=[[0.01 * index, 0, 0, 0, 0]])
        for index, (source, gain) in enumerate(zip(tl_sources, (0.8, 1.2), strict=True), 1)
    ]
    tr_rows = [
        _row(
            source,
            gain=gain,
            coefficient=[[0, 0.02 * index, 0, 0, 0], [0, 0, 0.01 * index, 0, 0]],
        )
        for index, (source, gain) in enumerate(zip(tr_sources, (0.9, 1.1), strict=True), 1)
    ]
    payloads = {
        "TL": _payload(0, tl_shared, tl_rows, global_scale=0.7),
        "TR": _payload(0, tr_shared, tr_rows, global_scale=0.6),
    }

    result = compose_base_corrections(
        records=records,
        correction_payloads=payloads,
        channel=0,
    )

    assert np.count_nonzero(result["coefficient"]) == 0
    output = {row["path"]: row for row in result["sources"]}
    for view, payload in payloads.items():
        for base in payload["sources"]:
            actual = output[str(Path(base["path"]).resolve())]
            rows = max(len(payload["coefficient"]), len(result["coefficient"]))
            expected_field = np.zeros((rows, 5))
            expected_field[: len(payload["coefficient"])] += payload["coefficient"]
            expected_field[: len(base["coefficient"])] += base["coefficient"]
            np.testing.assert_allclose(actual["coefficient"], expected_field)
            expected_gain = payload["global_scale"] * base["gain"]
            assert result["global_scale"] * actual["gain"] == pytest.approx(
                result["global_scale"] * expected_gain
            )

    tl_ratio = output[str(tl_sources[1].resolve())]["gain"] / output[str(tl_sources[0].resolve())]["gain"]
    tr_ratio = output[str(tr_sources[1].resolve())]["gain"] / output[str(tr_sources[0].resolve())]["gain"]
    assert tl_ratio == pytest.approx(1.2 / 0.8)
    assert tr_ratio == pytest.approx(1.1 / 0.9)
    assert result["fit"]["method"] == "cross-dataset-match-base-corrections"
    assert result["fit"]["fixed_per_dataset_corrections"] is True
    assert result["fit"]["per_tile_gains_fitted"] is False
    assert result["fit"]["common_saturation_scale"] == result["global_scale"]
    upper_bounds = []
    for row in result["sources"]:
        coefficient = np.asarray(row["coefficient"])
        log_bound = sum(
            np.abs(coefficient[degree]).sum() / (1 + degree**2) for degree in range(len(coefficient))
        )
        upper_bounds.append(result["global_scale"] * row["gain"] * np.exp(log_bound))
    assert max(upper_bounds) <= 1 + 1e-12


def test_compose_rejects_missing_view_correction(tmp_path: Path) -> None:
    source = tmp_path / "tl.ome.zarr"
    records = [{"path": str(source), "tile": source.name, "source_view": "TL", "shape": [1, 7, 9, 11]}]
    payload = _payload(
        0,
        [[0, 0, 0, 0, 0]],
        [_row(source, gain=1, coefficient=[[0, 0, 0, 0, 0]])],
        global_scale=1,
    )

    with pytest.raises(ValueError, match="exactly match"):
        compose_base_corrections(
            records=records,
            correction_payloads={"TL": payload, "TR": payload},
            channel=0,
        )


def test_registered_field_varies_spatially_and_uses_held_out_blocks(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.ome.zarr"
    _write_zarr(
        fixed,
        np.zeros((7, 7, 7), dtype=np.uint16),
        "zyx",
        scale=(100, 100, 100),
    )
    registration = tmp_path / "registration.json"
    registration.write_text("{}\n")
    rows = []
    for z in (100, 300, 500):
        for y in (100, 300, 500):
            for x in (100, 300, 500):
                for repeat in range(4):
                    rows.append(
                        {
                            "center_zyx_um": [z, y, x],
                            "first_source": f"tl-{repeat % 2}",
                            "second_source": f"tr-{repeat // 2}",
                            "first_view": "TL",
                            "second_view": "TR",
                            "log_ratio_second_over_first": float(np.log(1.2) + (y - 300) / 1000),
                        }
                    )

    payload, _deviation, _support = build_registered_view_correction(
        rows=rows,
        moving_view="TR",
        fixed_fused=fixed,
        registration=registration,
        grid_spacing_zyx_um=(100, 100, 100),
    )
    assert payload["fit"]["spatial_cross_validation"]["folds"] >= 2
    artifact = tmp_path / "registered-view-correction.json"
    artifact.write_text(json.dumps(payload))
    correction = load_registered_view_correction(artifact)
    low_y = correction.log_ratio_block(
        origin_zyx_um=(300, 100, 300),
        spacing_zyx_um=(100, 100, 100),
        shape_zyx=(1, 1, 1),
    )
    high_y = correction.log_ratio_block(
        origin_zyx_um=(300, 500, 300),
        spacing_zyx_um=(100, 100, 100),
        shape_zyx=(1, 1, 1),
    )
    assert high_y.item() > low_y.item()


def test_cross_dataset_match_writes_base_correction_and_registered_field(tmp_path: Path) -> None:
    zz, yy, xx = np.meshgrid(np.arange(9), np.arange(32), np.arange(32), indexing="ij")
    scene = 1000 + 120 * np.sin(xx / 3) + 90 * np.cos(yy / 4) + 20 * zz
    tl_source = tmp_path / "tl.ome.zarr"
    tr_source = tmp_path / "tr.ome.zarr"
    _write_zarr(tl_source, scene[None].astype(np.float32), "czyx")
    _write_zarr(tr_source, (scene * 1.25)[None].astype(np.float32), "czyx")
    fixed = tmp_path / "fixed.ome.zarr"
    _write_zarr(fixed, np.zeros_like(scene, dtype=np.uint16), "zyx")
    records = [
        {
            "path": str(source),
            "tile": source.name,
            "source_view": view,
            "axes": "CZYX",
            "shape": [1, *scene.shape],
            "translation_um": {axis: 0 for axis in "zyx"},
            "scale_um": {axis: 1 for axis in "zyx"},
        }
        for source, view in ((tl_source, "TL"), (tr_source, "TR"))
    ]
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"tiles": records}) + "\n")
    correction_paths = {}
    for source, view in ((tl_source, "TL"), (tr_source, "TR")):
        correction = tmp_path / f"{view}.correction.json"
        correction.write_text(
            json.dumps(
                _payload(
                    0,
                    [[0, 0, 0, 0, 0]],
                    [
                        {
                            **_row(source, gain=1, coefficient=[[0, 0, 0, 0, 0]]),
                            "source_sha256": _source_identity(source),
                            "shape_zyx": list(scene.shape),
                        }
                    ],
                    global_scale=1,
                )
            )
            + "\n"
        )
        correction_paths[view] = correction

    output = tmp_path / "matched"
    manifest = cross_dataset_match(
        registration=registration,
        fixed_fused=fixed,
        output_dir=output,
        corrections_by_view=correction_paths,
        reference_view="TL",
        moving_view="TR",
        source_level=0,
        stride=1,
        z_percentiles=(25, 50, 75),
        workers=1,
    )

    assert manifest == output / "manifest.json"
    match = json.loads((output / "match.json").read_text())
    assert match["model"] == "registered-spatial-field"
    assert match["mean_moving_factor_not_applied_to_base_correction"] == pytest.approx(0.8, rel=2e-3)
    assert match["summary"]["after_median_abs_log2"] < 0.002
    assert (output / "cross-dataset-match-qc.png").is_file()
    manifest_payload = json.loads(manifest.read_text())
    assert manifest_payload["artifact_type"] == "squisher_lightsheet.cross_dataset_match.v2"
    assert manifest_payload["registered_view_correction"]["path"] == ("registered-view-correction.json")
    base_payload = json.loads((output / "correction.json").read_text())
    assert base_payload["fit"]["moving_factor_applied"] is False
    assert "moving_factor" not in base_payload["fit"]
    loaded = load_residual_corrections(
        output / "correction.json",
        sources=[tl_source, tr_source],
        shapes_zyx=[scene.shape, scene.shape],
        channel=0,
    )
    assert loaded[str(tl_source.resolve())].multiplier == pytest.approx(1)
    assert loaded[str(tr_source.resolve())].multiplier == pytest.approx(1)
    registered = load_registered_view_correction(output / "registered-view-correction.json")
    assert registered.moving_view == "TR"
    assert registered.intercept_log_ratio == pytest.approx(np.log(1.25), rel=2e-3)
    assert registered.registration_sha256 == match["registration_sha256"]
    assert registered.registration_sha256 == hashlib.sha256(registration.read_bytes()).hexdigest()
    field = registered.log_ratio_block(
        origin_zyx_um=(0, 0, 0),
        spacing_zyx_um=(1, 1, 1),
        shape_zyx=(1, 1, 1),
    )
    assert np.exp(-field[0, 0, 0]) == pytest.approx(0.8, rel=2e-3)

    identity_output = tmp_path / "matched-identity"
    cross_dataset_match(
        registration=registration,
        fixed_fused=fixed,
        output_dir=identity_output,
        corrections_by_view={},
        reference_view="TL",
        moving_view="TR",
        source_level=0,
        stride=1,
        z_percentiles=(25, 50, 75),
        workers=1,
    )
    identity_manifest = json.loads((identity_output / "manifest.json").read_text())
    assert identity_manifest["base_corrections"] == {
        "TL": {"mode": "identity", "sources": 1},
        "TR": {"mode": "identity", "sources": 1},
    }
    identity_correction = json.loads((identity_output / "correction.json").read_text())
    assert all(np.count_nonzero(row["coefficient"]) == 0 for row in identity_correction["sources"])

    tl_scaled = tmp_path / "TL.scaled.correction.json"
    tl_scaled_payload = json.loads(correction_paths["TL"].read_text())
    tl_scaled_payload["global_scale"] = 0.5
    tl_scaled.write_text(json.dumps(tl_scaled_payload) + "\n")
    mixed_output = tmp_path / "matched-mixed"
    cross_dataset_match(
        registration=registration,
        fixed_fused=fixed,
        output_dir=mixed_output,
        corrections_by_view={"TL": tl_scaled},
        reference_view="TL",
        moving_view="TR",
        source_level=0,
        stride=1,
        z_percentiles=(25, 50, 75),
        workers=1,
    )
    mixed_manifest = json.loads((mixed_output / "manifest.json").read_text())
    assert mixed_manifest["base_corrections"]["TL"]["mode"] == "supplied"
    assert mixed_manifest["base_corrections"]["TR"] == {"mode": "identity", "sources": 1}
    mixed_match = json.loads((mixed_output / "match.json").read_text())
    assert mixed_match["mean_moving_factor_not_applied_to_base_correction"] == pytest.approx(0.4, rel=2e-3)
    mixed_loaded = load_residual_corrections(
        mixed_output / "correction.json",
        sources=[tl_source, tr_source],
        shapes_zyx=[scene.shape, scene.shape],
        channel=0,
    )
    assert mixed_loaded[str(tl_source.resolve())].multiplier == pytest.approx(0.5)
    assert mixed_loaded[str(tr_source.resolve())].multiplier == pytest.approx(1)
    mixed_registered = load_registered_view_correction(mixed_output / "registered-view-correction.json")
    assert np.exp(-mixed_registered.intercept_log_ratio) == pytest.approx(0.4, rel=2e-3)


def test_cross_dataset_match_accepts_fit_residual_corrected_materialized_sources(
    tmp_path: Path,
) -> None:
    zz, yy, xx = np.meshgrid(np.arange(9), np.arange(32), np.arange(32), indexing="ij")
    scene = 1000 + 120 * np.sin(xx / 3) + 90 * np.cos(yy / 4) + 20 * zz
    fixed = tmp_path / "fixed.ome.zarr"
    _write_zarr(fixed, np.zeros_like(scene, dtype=np.uint16), "zyx")
    records = []
    for view, factor in (("L", 1.0), ("R", 1.25)):
        source = tmp_path / f"{view}.materialized.ome.zarr"
        _write_zarr(source, (scene * factor).astype(np.float32), "zyx")
        records.append(
            {
                "path": str(source),
                "tile": source.name,
                "moving_tile": f"{view}.source.ome.zarr",
                "source_view": view,
                "axes": "ZYX",
                "shape": list(scene.shape),
                "translation_um": {axis: 0 for axis in "zyx"},
                "scale_um": {axis: 1 for axis in "zyx"},
                "materialized_source_start_zyx": [0, 0, 0],
                "materialized_source_stop_zyx": list(scene.shape),
                "materialized_fixed_origin_um": {axis: 0 for axis in "zyx"},
                "materialized_fixed_spacing_um": {axis: 1 for axis in "zyx"},
                "materialized_fixed_shape_zyx": list(scene.shape),
            }
        )
    registration = tmp_path / "materialized.registration.json"
    registration.write_text(json.dumps({"tiles": records}) + "\n")

    output = tmp_path / "matched-materialized"
    cross_dataset_match(
        registration=registration,
        fixed_fused=fixed,
        output_dir=output,
        corrections_by_view={},
        reference_view="L",
        moving_view="R",
        source_level=0,
        stride=1,
        z_percentiles=(25, 50, 75),
        workers=1,
    )

    match = json.loads((output / "match.json").read_text())
    assert match["summary"]["after_median_abs_log2"] < 0.002
    correction = json.loads((output / "correction.json").read_text())
    assert all(np.count_nonzero(row["coefficient"]) == 0 for row in correction["sources"])
    registered = load_registered_view_correction(output / "registered-view-correction.json")
    assert np.exp(-registered.intercept_log_ratio) == pytest.approx(0.8, rel=2e-3)


def test_cross_dataset_match_rejects_double_correction_of_materialized_sources(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "materialized.registration.json"
    registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "path": str(tmp_path / "materialized.ome.zarr"),
                        "tile": "materialized.ome.zarr",
                        "source_view": "L",
                        "shape": [4, 4, 4],
                        "materialized_source_start_zyx": [0, 0, 0],
                    },
                    {
                        "path": str(tmp_path / "other.ome.zarr"),
                        "tile": "other.ome.zarr",
                        "source_view": "R",
                        "shape": [4, 4, 4],
                        "materialized_source_start_zyx": [0, 0, 0],
                    },
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="already contain fit-residual"):
        cross_dataset_match(
            registration=registration,
            fixed_fused=tmp_path / "fixed.ome.zarr",
            output_dir=tmp_path / "output",
            corrections_by_view={"L": tmp_path / "correction.json"},
            reference_view="L",
            moving_view="R",
        )
