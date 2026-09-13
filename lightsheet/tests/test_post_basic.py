from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import zarr

from squisher_deconv.tile_gains import load_tile_gains
from squisher_lightsheet.post_basic import (
    PostBasicChannel,
    _Grid,
    _fixed_to_source_pull,
    _plan_record,
    _pair_folds,
    _SampledTile,
    _seam_samples,
    parse_channel_spec,
    run_post_basic,
)


LOG_GAINS = np.asarray(
    [[0.15, -0.10, 0.04], [-0.12, 0.08, -0.02], [0.03, -0.06, 0.11]],
    dtype=np.float64,
)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("edge", [0, 1])
def test_seams_include_neighbors_outside_overlap(axis: int, edge: int) -> None:
    owner = np.zeros((64, 64), dtype=np.int32)
    region = [slice(None), slice(None)]
    region[axis] = slice(16, None) if edge == 0 else slice(None, 48)
    owner[tuple(region)] = 1
    tiles = [
        _SampledTile(
            source=Path(f"tile-{index}.ome.tif"),
            lo=np.array([16, 16]),
            hi=np.array([48, 48]),
            data=np.full((32, 32), 100 * (index + 1), dtype=np.float32),
            yx=np.zeros((32, 32, 2)),
            valid=np.ones((32, 32), dtype=bool),
        )
        for index in range(2)
    ]
    rows = _seam_samples(tiles, owner, cutoff=1)
    assert len(rows) == 1
    # The seam is entirely outside the overlap's internal adjacency pairs.
    assert len(rows[0]["target"]) == (32 if edge == 0 else 24)
    np.testing.assert_allclose(rows[0]["target"], np.log(2))


def _write_grid(path: Path, shape: tuple[int, int, int]) -> None:
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.create_array(
        "0",
        shape=shape,
        chunks=shape,
        dtype="uint16",
        dimension_names=("z", "y", "x"),
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": axis} for axis in "zyx"],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
                        ],
                    }
                ],
            }
        ],
    }


def _write_raw_tile(path: Path, plane: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        plane[None, None].astype(np.uint16),
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "CZYX",
            "PhysicalSizeZ": 1.0,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
            "Plane": {
                "PositionZ": [0.0],
                "PositionY": [0.0],
                "PositionX": [0.0],
            },
        },
    )


def _synthetic_inputs(
    tmp_path: Path, *, log_gains: np.ndarray = LOG_GAINS
) -> tuple[Path, Path, Path, Path, list[Path]]:
    raw_dir = tmp_path / "raw"
    basic_dir = tmp_path / "basic"
    raw_dir.mkdir()
    basic_dir.mkdir()
    fixed = tmp_path / "fixed.ome.zarr"
    _write_grid(fixed, (1, 128, 128))

    size = 64
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    local_y = yy / (size - 1)
    local_x = xx / (size - 1)
    field = np.exp(0.09 * np.cos(np.pi * local_y) - 0.14 * np.cos(np.pi * local_x))
    records = []
    sources = []
    for row in range(3):
        for column in range(3):
            y0, x0 = 32 * row, 32 * column
            global_y, global_x = yy + y0, xx + x0
            checker = ((global_y // 4 + global_x // 4) % 2).astype(np.float64)
            global_signal = 2200 + 4200 * checker + 120 * np.sin(global_y / 7) + 90 * np.cos(global_x / 9)
            gain = np.exp(log_gains[row, column])
            plane = np.rint(global_signal / (field * gain))
            source = raw_dir / f"tile-{row}-{column}.ome.tif"
            _write_raw_tile(source, plane)
            sources.append(source)
            records.append(
                {
                    "tile": source.with_suffix("").with_suffix("").name + ".ome.zarr",
                    "stage_translation_um": {"z": 0.0, "y": float(y0), "x": float(x0)},
                    "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                    "registered_affine": {"matrix": np.eye(4).tolist()},
                }
            )
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"tiles": records}) + "\n")

    flat = np.ones((size, size), dtype=np.float32)
    dark = np.zeros((size, size), dtype=np.float32)
    tifffile.imwrite(basic_dir / "multi-ch0-flatfield.tif", flat, compression="zstd")
    tifffile.imwrite(basic_dir / "multi-ch0-darkfield.tif", dark, compression="zstd")
    (basic_dir / "multi-ch0.pkl").write_bytes(
        pickle.dumps({"basic": SimpleNamespace(flatfield=flat, darkfield=dark)})
    )
    return fixed, raw_dir, basic_dir, registration, sources


def test_cropped_affine_pull_adds_original_source_crop_start() -> None:
    affine = np.eye(4)
    affine[:3, 3] = [4.0, -3.0, 2.0]
    record = {
        "tile": "crop.ome.zarr",
        "stage_translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
        "stage_scale_um": {"z": 2.0, "y": 3.0, "x": 4.0},
        "registered_affine": {"matrix": affine.tolist()},
        "materialized_source_start_zyx": [7, 11, 13],
    }

    matrix, offset = _fixed_to_source_pull(
        record,
        fixed_scale=np.asarray([5.0, 6.0, 7.0]),
        fixed_origin=np.asarray([1.0, 2.0, 3.0]),
    )
    fixed_index = np.asarray([2.0, 3.0, 4.0])
    fixed_world = np.asarray([1.0, 2.0, 3.0]) + np.asarray([5.0, 6.0, 7.0]) * fixed_index
    expected = np.asarray([7.0, 11.0, 13.0]) + (
        np.linalg.inv(affine)[:3, :3] @ fixed_world
        + np.linalg.inv(affine)[:3, 3]
        - np.asarray([10.0, 20.0, 30.0])
    ) / np.asarray([2.0, 3.0, 4.0])
    np.testing.assert_allclose(matrix @ fixed_index + offset, expected)


def test_cropped_plan_uses_materialized_fixed_extent() -> None:
    record = {
        "tile": "crop.ome.zarr",
        "shape": [1, 5, 7],
        "spacing_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        "materialized_fixed_origin_um": {"z": 0.0, "y": 20.0, "x": 30.0},
        "materialized_fixed_spacing_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        "materialized_fixed_shape_zyx": [1, 5, 7],
        "stage_translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        "registered_affine": {"matrix": np.eye(4).tolist()},
        "materialized_source_start_zyx": [0, 10, 10],
        "materialized_source_stop_zyx": [1, 90, 90],
    }
    grid = _Grid(
        shape=np.asarray([1, 100, 100]),
        scale=np.ones(3),
        origin=np.zeros(3),
        z=0,
        stride=1,
    )

    plan = _plan_record(
        record=record,
        source=Path("crop.ome.tif"),
        source_shape=(1, 100, 100),
        grid=grid,
    )

    assert plan is not None
    np.testing.assert_array_equal(plan.lo, [20, 30])
    np.testing.assert_array_equal(plan.hi, [25, 37])


def test_cropped_plan_without_fixed_extent_uses_affine() -> None:
    affine = np.diag([1.0, 2.0, 0.5, 1.0])
    affine[:3, 3] = [0.0, 10.0, 20.0]
    record = {
        "tile": "crop.ome.zarr",
        "shape": [1, 4, 6],
        "axes": "ZYX",
        "stage_translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        "registered_affine": {"matrix": affine.tolist()},
        "materialized_source_start_zyx": [0, 2, 4],
        "materialized_source_stop_zyx": [1, 6, 10],
    }
    grid = _Grid(
        shape=np.asarray([1, 100, 100]),
        scale=np.ones(3),
        origin=np.zeros(3),
        z=0,
        stride=1,
    )

    plan = _plan_record(
        record=record,
        source=Path("crop.ome.tif"),
        source_shape=(1, 20, 30),
        grid=grid,
    )

    assert plan is not None
    np.testing.assert_array_equal(plan.lo, [10, 20])
    np.testing.assert_array_equal(plan.hi, [17, 23])


def test_post_basic_runs_raw_sampling_fit_and_profile_composition(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, sources = _synthetic_inputs(tmp_path)
    output = tmp_path / "post-basic"

    manifest_path = run_post_basic(
        fixed_fused=fixed,
        raw_dir=raw_dir,
        basic_dir=basic_dir,
        output_dir=output,
        channels=(PostBasicChannel(label="561", index=0, registration=registration),),
        tile_gain_channels=frozenset({0}),
        fixed_z=0,
        stride=1,
        workers=2,
        seed=17,
    )

    manifest = json.loads(manifest_path.read_text())
    result = manifest["channel_results"]["0"]
    assert manifest["status"] == "complete"
    qc = manifest["qc"]
    for artifact in qc["artifacts"].values():
        path = output / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    for extension in ("png", "pdf", "svg"):
        assert (output / "qc" / f"channel_0.{extension}").is_file()
    with (output / "qc/tile_gains.csv").open() as handle:
        gain_rows = list(csv.DictReader(handle))
    assert {row["source"] for row in gain_rows} == {str(source.resolve()) for source in sources}
    owner = tifffile.imread(output / result["artifacts"]["owner"]["path"])
    assert owner.dtype == np.int32
    assert owner.max() < len(result["sampling"]["sampled_sources"])
    assert np.isfinite(result["heldout_after"]["median"])
    assert result["field_range"][0] > 0
    assert result["tile_gain_zero_support_policy"] == "identity"
    assert isinstance(result["tile_gain_unestimated_sources"], list)
    assert (output / result["artifacts"]["comparison"]["path"]).is_file()
    assert (output / result["artifacts"]["mask"]["path"]).is_file()

    gains = load_tile_gains(output / "tile-gains.json", inputs=sources, channels=1)
    assert set(gains) == {str(source.resolve()) for source in sources}
    fitted_log_gains = np.asarray([np.log(gains[str(source.resolve())][0]) for source in sources])
    assert np.corrcoef(fitted_log_gains, LOG_GAINS.ravel())[0, 1] > 0.5
    loaded = pickle.loads((output / "multi-ch0.pkl").read_bytes())
    mask = tifffile.imread(output / "mask-ch0.tif")
    np.testing.assert_allclose(loaded["basic"].flatfield, 1.0 / mask, rtol=1e-6)
    np.testing.assert_array_equal(loaded["basic"].darkfield, np.zeros((64, 64), np.float32))


def test_post_basic_recovers_shared_camera_field_on_heldout_pairs(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(
        tmp_path, log_gains=np.zeros((3, 3), dtype=np.float64)
    )

    manifest_path = run_post_basic(
        fixed_fused=fixed,
        raw_dir=raw_dir,
        basic_dir=basic_dir,
        output_dir=tmp_path / "post-basic",
        channels=(PostBasicChannel(label="561", index=0, registration=registration),),
        fixed_z=0,
        stride=1,
        workers=2,
        seed=17,
    )

    result = json.loads(manifest_path.read_text())["channel_results"]["0"]
    assert result["heldout_after"]["median"] < result["heldout_before"]["median"]
    yx = np.linspace(0, 1, 64)
    yy, xx = np.meshgrid(yx, yx, indexing="ij")
    expected_log_field = 0.09 * np.cos(np.pi * yy) - 0.14 * np.cos(np.pi * xx)
    fitted_log_field = np.log(tifffile.imread(tmp_path / "post-basic/mask-ch0.tif"))
    assert np.corrcoef(expected_log_field.ravel(), fitted_log_field.ravel())[0, 1] > 0.8


def test_post_basic_refuses_existing_output(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(tmp_path)
    output = tmp_path / "post-basic"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    try:
        run_post_basic(
            fixed_fused=fixed,
            raw_dir=raw_dir,
            basic_dir=basic_dir,
            output_dir=output,
            channels=(PostBasicChannel(label="561", index=0, registration=registration),),
            tile_gain_channels=frozenset(),
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output was accepted")
    assert marker.read_text() == "keep"


def test_post_basic_rejects_sidecar_that_differs_from_pickle(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(tmp_path)
    profile_path = basic_dir / "multi-ch0.pkl"
    payload = pickle.loads(profile_path.read_bytes())
    payload["basic"].flatfield *= 2
    profile_path.write_bytes(pickle.dumps(payload))
    output = tmp_path / "post-basic"

    try:
        run_post_basic(
            fixed_fused=fixed,
            raw_dir=raw_dir,
            basic_dir=basic_dir,
            output_dir=output,
            channels=(PostBasicChannel(label="561", index=0, registration=registration),),
        )
    except ValueError as error:
        assert "flatfield TIFF does not match" in str(error)
    else:
        raise AssertionError("mismatched TIFF and pickle profiles were accepted")
    assert not output.exists()


def test_post_basic_validates_unfitted_profile_sidecars(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(tmp_path)
    flat = np.ones((64, 64), dtype=np.float32)
    dark = np.zeros((64, 64), dtype=np.float32)
    tifffile.imwrite(basic_dir / "multi-ch1-flatfield.tif", flat, compression="zstd")
    tifffile.imwrite(basic_dir / "multi-ch1-darkfield.tif", dark, compression="zstd")
    (basic_dir / "multi-ch1.pkl").write_bytes(
        pickle.dumps({"basic": SimpleNamespace(flatfield=flat * 2, darkfield=dark)})
    )
    output = tmp_path / "post-basic"

    try:
        run_post_basic(
            fixed_fused=fixed,
            raw_dir=raw_dir,
            basic_dir=basic_dir,
            output_dir=output,
            channels=(PostBasicChannel(label="561", index=0, registration=registration),),
        )
    except ValueError as error:
        assert "Channel 1 BaSiC flatfield TIFF does not match" in str(error)
    else:
        raise AssertionError("an unfitted mismatched profile was accepted")
    assert not output.exists()


def test_tile_gain_channel_requires_every_raw_source_on_sampled_plane(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(tmp_path)
    extra = raw_dir / "extra.ome.tif"
    _write_raw_tile(extra, np.full((64, 64), 3000, dtype=np.uint16))
    payload = json.loads(registration.read_text())
    payload["tiles"].append(
        {
            "tile": "extra.ome.zarr",
            "stage_translation_um": {"z": 10.0, "y": 0.0, "x": 0.0},
            "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
            "registered_affine": {"matrix": np.eye(4).tolist()},
        }
    )
    registration.write_text(json.dumps(payload) + "\n")

    try:
        run_post_basic(
            fixed_fused=fixed,
            raw_dir=raw_dir,
            basic_dir=basic_dir,
            output_dir=tmp_path / "post-basic",
            channels=(PostBasicChannel(label="561", index=0, registration=registration),),
            tile_gain_channels=frozenset({0}),
            fixed_z=0,
            stride=1,
        )
    except ValueError as error:
        assert "did not sample every raw source" in str(error)
    else:
        raise AssertionError("an unsampled source received an implicit unit gain")


def test_parse_channel_spec_keeps_equals_in_registration_path() -> None:
    assert parse_channel_spec("561=0=/tmp/a=b.json") == PostBasicChannel(
        label="561", index=0, registration=Path("/tmp/a=b.json")
    )


def test_tile_gain_channel_honors_explicit_registration_exclusions(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, sources = _synthetic_inputs(tmp_path)
    extra = raw_dir / "extra.ome.tif"
    _write_raw_tile(extra, np.full((64, 64), 3000, dtype=np.uint16))
    payload = json.loads(registration.read_text())
    payload["metrics"] = {"registration_run": {"connectivity": {"excluded_tiles": [extra.name]}}}
    registration.write_text(json.dumps(payload) + "\n")
    output = tmp_path / "post-basic"
    manifest_path = run_post_basic(
        fixed_fused=fixed,
        raw_dir=raw_dir,
        basic_dir=basic_dir,
        output_dir=output,
        channels=(PostBasicChannel(label="561", index=0, registration=registration),),
        tile_gain_channels=frozenset({0}),
        fixed_z=0,
        stride=1,
        workers=2,
        seed=17,
    )
    result = json.loads(manifest_path.read_text())["channel_results"]["0"]
    assert result["excluded_sources"] == [str(extra.resolve())]
    assert result["excluded_source_gain_policy"] == "identity"
    gains = load_tile_gains(output / "tile-gains.json", inputs=[*sources, extra], channels=1)
    assert gains[str(extra.resolve())][0] == 1.0


def test_post_basic_accepts_ome_channel_ids_on_multichannel_records(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(tmp_path)
    payload = json.loads(registration.read_text())
    for record in payload["tiles"]:
        record.update(axes="CZYX", shape=[1, 1, 64, 64], channels=["Channel:0:0"])
    registration.write_text(json.dumps(payload) + "\n")
    manifest = run_post_basic(
        fixed_fused=fixed,
        raw_dir=raw_dir,
        basic_dir=basic_dir,
        output_dir=tmp_path / "post-basic",
        channels=(PostBasicChannel(label="561", index=0, registration=registration),),
        fixed_z=0,
        stride=1,
        workers=2,
        seed=17,
    )
    assert json.loads(manifest.read_text())["status"] == "complete"


def test_post_basic_accepts_ome_channel_id_on_single_channel_records(tmp_path: Path) -> None:
    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(tmp_path)
    payload = json.loads(registration.read_text())
    for record in payload["tiles"]:
        record.update(axes="ZYX", shape=[1, 64, 64], channels=["Channel:0:0"])
    registration.write_text(json.dumps(payload) + "\n")
    manifest = run_post_basic(
        fixed_fused=fixed,
        raw_dir=raw_dir,
        basic_dir=basic_dir,
        output_dir=tmp_path / "post-basic",
        channels=(PostBasicChannel(label="405", index=0, registration=registration),),
        fixed_z=0,
        stride=1,
        workers=2,
        seed=17,
    )
    assert json.loads(manifest.read_text())["status"] == "complete"


def test_post_basic_does_not_publish_when_qc_rendering_fails(tmp_path: Path, monkeypatch) -> None:
    import squisher_lightsheet.post_basic as module

    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(tmp_path)
    output = tmp_path / "post-basic"

    def fail_qc(*args, **kwargs):
        raise RuntimeError("QC rendering failed")

    monkeypatch.setattr(module, "write_post_basic_qc", fail_qc)
    with pytest.raises(RuntimeError, match="QC rendering failed"):
        run_post_basic(
            fixed_fused=fixed,
            raw_dir=raw_dir,
            basic_dir=basic_dir,
            output_dir=output,
            channels=(PostBasicChannel(label="561", index=0, registration=registration),),
            fixed_z=0,
            stride=1,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".post-basic-*"))


def test_repeated_source_pairs_stay_in_one_fold_across_z_planes() -> None:
    rows = [{"first": pair, "second": pair + 1, "z": z} for z in (2, 3, 4) for pair in range(5)]
    folds = _pair_folds(rows, seed=17).reshape(3, 5)
    np.testing.assert_array_equal(folds[0], folds[1])
    np.testing.assert_array_equal(folds[0], folds[2])
    assert set(folds[0]) == set(range(5))


@pytest.mark.parametrize("z_degree", [0, 1])
def test_post_basic_pools_requested_z_planes_with_plane_heldout_validation(
    tmp_path: Path, z_degree: int
) -> None:
    fixed, raw_dir, basic_dir, registration, sources = _synthetic_inputs(tmp_path)
    _write_grid(fixed, (11, 128, 128))
    for source in sources:
        plane = tifffile.imread(source).squeeze()
        tifffile.imwrite(
            source,
            np.repeat(plane[None, None], 11, axis=1),
            ome=True,
            photometric="minisblack",
            metadata={
                "axes": "CZYX",
                "PhysicalSizeZ": 1.0,
                "PhysicalSizeY": 1.0,
                "PhysicalSizeX": 1.0,
                "Plane": {"PositionZ": list(range(11)), "PositionY": [0.0] * 11, "PositionX": [0.0] * 11},
            },
        )
    manifest_path = run_post_basic(
        fixed_fused=fixed,
        raw_dir=raw_dir,
        basic_dir=basic_dir,
        output_dir=tmp_path / "post-basic",
        channels=(PostBasicChannel(label="561", index=0, registration=registration),),
        tile_gain_channels=frozenset({0}),
        z_percentiles=(20, 30, 40, 50, 60, 70),
        z_degree=z_degree,
        stride=1,
        workers=2,
        seed=17,
    )
    manifest = json.loads(manifest_path.read_text())
    from squisher_deconv.basic_profiles import load_basic_profile_arrays

    profile = load_basic_profile_arrays(manifest_path.parent / "multi-ch0.pkl")
    assert (profile.residual_coefficient is not None) == bool(z_degree)
    if z_degree:
        assert profile.residual_coefficient.shape == (2, 5)
        np.testing.assert_array_equal(
            profile.flatfield, tifffile.imread(basic_dir / "multi-ch0-flatfield.tif")
        )
    assert manifest["sampled_z"] == [2, 3, 4, 5, 6, 7]
    assert manifest["fixed_z"] == 5
    result = manifest["channel_results"]["0"]
    assert len(result["planes"]) == 6
    for plane in result["planes"]:
        prefix = "" if plane["z"] == manifest["fixed_z"] else f"planes/z{plane['z']}/"
        for name, artifact in plane["artifacts"].items():
            if name == "display_range":
                continue
            assert artifact["path"].startswith(prefix)
            path = manifest_path.parent / artifact["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        if plane["z"] == manifest["fixed_z"]:
            assert plane["artifacts"] == result["artifacts"]
    assert len(result["z_validation"]) == 6
    for evaluation in result["z_validation"]:
        assert evaluation["heldout_z"] not in evaluation["training_z"]
        assert evaluation["single_plane_z"] in evaluation["training_z"]
        assert len(evaluation["training_z"]) == 5
        assert evaluation["shared"]["pairs"] > 0
    source_folds = {}
    for pair in result["pairs"]:
        identity = tuple(pair["sources"])
        assert source_folds.setdefault(identity, pair["fold"]) == pair["fold"]
    gains = load_tile_gains(manifest_path.parent / "tile-gains.json", inputs=sources, channels=1)
    assert len(gains) == len(sources)
    assert (manifest_path.parent / "qc/z_validation.csv").is_file()


@pytest.mark.parametrize(
    "percentiles,fixed_z,match",
    [
        ((20, 70), 0, "either fixed_z or z_percentiles"),
        ((-1, 70), None, "finite values"),
        ((float("nan"), 70), None, "finite values"),
        ((50,), None, "at least two distinct"),
        ((20, 30), None, "at least two distinct"),
    ],
)
def test_post_basic_rejects_invalid_plane_selection(tmp_path, percentiles, fixed_z, match) -> None:
    fixed, raw_dir, basic_dir, registration, _sources = _synthetic_inputs(tmp_path)
    with pytest.raises(ValueError, match=match):
        run_post_basic(
            fixed_fused=fixed,
            raw_dir=raw_dir,
            basic_dir=basic_dir,
            output_dir=tmp_path / "post-basic",
            channels=(PostBasicChannel(label="561", index=0, registration=registration),),
            fixed_z=fixed_z,
            z_percentiles=percentiles,
        )


def test_regularized_z_field_predicts_depth_dependent_overlap_ratios():
    from squisher_lightsheet.post_basic import _fit_seams, _row_design

    rng = np.random.default_rng(8)
    coefficient = np.zeros(10)
    coefficient[5] = 0.7
    rows = []
    for z in np.linspace(0, 1, 9):
        for pair in range(10):
            row = {
                "first": pair,
                "second": pair + 1,
                "pair_id": str(pair),
                "z": int(z * 8),
                "ya": rng.random((50, 2)),
                "yb": rng.random((50, 2)),
                "za": np.full(50, z),
                "zb": np.full(50, z),
            }
            row["target"] = _row_design(row, 1) @ coefficient
            rows.append(row)
    sources = [Path(str(i)) for i in range(11)]
    shared, _, _ = _fit_seams(rows=rows, sources=sources, seed=8, fit_tile_gains=False)
    varying, _, _ = _fit_seams(rows=rows, sources=sources, seed=8, fit_tile_gains=False, z_degree=1)
    shared_error = np.mean(
        np.concatenate([(_row_design(row, 0) @ shared - row["target"]) ** 2 for row in rows])
    )
    varying_error = np.mean(
        np.concatenate([(_row_design(row, 1) @ varying - row["target"]) ** 2 for row in rows])
    )
    assert varying_error < shared_error * 0.1


def test_z_profile_crop_uses_original_camera_coordinates(tmp_path):
    from squisher_deconv.basic_profiles import compose_z_profile
    from squisher_deconv.residual_field import residual_plane
    from squisher_lightsheet.ome_metadata_dumb_stitch import load_basic_profile, apply_basic
    from squisher_lightsheet.post_basic import _read_corrected_crop

    base_dir = tmp_path / "base"
    corrected_dir = tmp_path / "corrected"
    base_dir.mkdir()
    corrected_dir.mkdir()
    flat = np.full((9, 13), 2, dtype=np.float32)
    dark = np.full((9, 13), 5, dtype=np.float32)
    profile_path = base_dir / "field-ch0.pkl"
    profile_path.write_bytes(pickle.dumps({"basic": SimpleNamespace(flatfield=flat, darkfield=dark)}))
    coefficient = np.zeros((2, 5))
    coefficient[1, 0] = 0.4
    compose_z_profile(profile_path, corrected_dir / profile_path.name, coefficient=coefficient, provenance={})
    tifffile.imwrite(corrected_dir / "field-ch0-flatfield.tif", flat)
    tifffile.imwrite(corrected_dir / "field-ch0-darkfield.tif", dark)
    profile = load_basic_profile(corrected_dir, 0)
    source = tmp_path / "source.ome.tif"
    raw = np.full((11, 9, 13), 105, dtype=np.uint16)
    tifffile.imwrite(source, raw, photometric="minisblack", metadata={"axes": "ZYX"})
    with tifffile.TiffFile(source) as tif:
        actual = _read_corrected_crop(
            tif,
            channel=0,
            source_shape=raw.shape,
            lo=np.asarray([6, 2, 3]),
            hi=np.asarray([9, 7, 11]),
            profile=profile,
        )
    expected = np.stack(
        [50 * residual_plane(coefficient, z=z, shape_zyx=raw.shape)[2:7, 3:11] for z in range(6, 9)]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    with pytest.raises(ValueError, match="original raw Z"):
        apply_basic(raw[0], profile)


def test_lower_xy_degree_fits_only_identifiable_first_order_modes():
    from squisher_lightsheet.post_basic import _fit_seams, _row_design

    rng = np.random.default_rng(19)
    truth = np.asarray([0.08, 0, -0.12, 0, 0, 0.03, 0, -0.02, 0, 0])
    rows = []
    for z in (0.0, 0.5, 1.0):
        for pair in range(10):
            row = {
                "first": pair,
                "second": pair + 1,
                "pair_id": str(pair),
                "z": int(z * 10),
                "ya": rng.random((100, 2)),
                "yb": rng.random((100, 2)),
                "za": np.full(100, z),
                "zb": np.full(100, z),
            }
            row["target"] = _row_design(row, 1) @ truth
            rows.append(row)
    coefficient, _, result = _fit_seams(
        rows=rows,
        sources=[Path(str(i)) for i in range(11)],
        seed=17,
        fit_tile_gains=False,
        z_degree=1,
        xy_degree=1,
        field_penalty=0.03,
    )
    np.testing.assert_array_equal(coefficient.reshape(2, 5)[:, [1, 3, 4]], 0)
    assert result["selected_penalty"] == 0.03
    assert result["degree"] == 1
    assert result["heldout_after"]["median"] < result["heldout_before"]["median"] * 0.2


def test_source_fields_recover_source_specific_spatial_residuals():
    from squisher_deconv.residual_field import cosine_basis
    from squisher_lightsheet.post_basic import _fit_seams

    rng = np.random.default_rng(41)
    source_count = 8
    truth = rng.normal(0, 0.18, size=(source_count, 10))
    truth[:, [1, 3, 4, 6, 8, 9]] = 0
    truth -= truth.mean(axis=0)
    rows = []
    for z in (0.0, 0.5, 1.0):
        for first in range(source_count):
            for second in range(first + 1, source_count):
                ya, yb = rng.random((2, 160, 2))
                za = np.full(160, z)
                zb = np.full(160, z)
                rows.append(
                    {
                        "first": first,
                        "second": second,
                        "pair_id": f"{first}-{second}",
                        "z": int(10 * z),
                        "ya": ya,
                        "yb": yb,
                        "za": za,
                        "zb": zb,
                        "target": cosine_basis(ya, z=za, z_degree=1) @ truth[first]
                        - cosine_basis(yb, z=zb, z_degree=1) @ truth[second],
                    }
                )
    sources = [Path(str(index)) for index in range(source_count)]

    _, _, shared = _fit_seams(
        rows=rows,
        sources=sources,
        seed=41,
        fit_tile_gains=False,
        z_degree=1,
        xy_degree=1,
        field_penalty=0.3,
    )
    _, _, fitted = _fit_seams(
        rows=rows,
        sources=sources,
        seed=41,
        fit_tile_gains=False,
        fit_source_fields=True,
        z_degree=1,
        xy_degree=1,
        field_penalty=0.3,
        source_field_penalty=0.003,
    )

    source_fields = np.asarray([row["coefficient"] for row in fitted["source_fields"]])
    assert source_fields.shape == (source_count, 10)
    assert np.linalg.norm(source_fields[:, 5:]) > 0
    assert fitted["heldout_with_source_field"]["median"] < shared["heldout_after"]["median"] * 0.25


def test_z_validation_accepts_sparse_per_plane_pair_support():
    from squisher_lightsheet.post_basic import _validate_z_models

    rng = np.random.default_rng(53)
    plane_pairs = (
        ((0, 1), (1, 2), (2, 3), (0, 3)),
        ((0, 1), (1, 2), (2, 3), (0, 2)),
        ((0, 1), (1, 2), (2, 3), (1, 3)),
    )
    rows = []
    for z, pairs in enumerate(plane_pairs):
        for first, second in pairs:
            ya, yb = rng.random((2, 40, 2))
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "pair_id": f"{first}-{second}",
                    "z": z,
                    "ya": ya,
                    "yb": yb,
                    "za": np.full(40, z / 2),
                    "zb": np.full(40, z / 2),
                    "target": np.zeros(40),
                }
            )

    result = _validate_z_models(
        rows,
        sources=[Path(str(index)) for index in range(4)],
        reference_z=1,
        seed=53,
        fit_tile_gains=True,
        fit_source_fields=True,
        z_degree=1,
        field_penalty=0.03,
        source_field_penalty=0.03,
    )

    assert len(result) == 3
