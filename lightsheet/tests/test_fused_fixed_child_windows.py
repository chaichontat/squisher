import json
from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet._backends.fused_fixed_sweep import _parent_window_path, _windows
from squisher_lightsheet._backends.fused_fixed_trial import (
    _global_index_affine_to_local_model,
    _local_model_to_global_index_affine,
)


def test_child_affine_rebase_preserves_parent_mapping_at_native_points() -> None:
    parent_shape = np.array([528, 528, 528])
    parent_fixed = np.array([954, 5581, 6400])
    parent_moving = np.array([960, 0, 0])
    parent_matrix = np.array([[1.002, 0.003, 0], [-0.001, 1.0, 0.002], [0, 0.001, 0.999]])
    parent_translation = np.array([2.1, -3.8, 9.9])
    matrix, offset = _local_model_to_global_index_affine(
        matrix=parent_matrix,
        translation=parent_translation,
        fixed_start_zyx=parent_fixed,
        moving_start_zyx=parent_moving,
        crop_shape_zyx=parent_shape,
    )
    for z_start in (960, 1120, 1280):
        child_shape = np.array([208, 528, 528])
        child_moving = np.array([z_start, 0, 0])
        child_fixed = parent_fixed + child_moving - parent_moving + np.array([1, -2, 3])
        child_matrix, child_translation = _global_index_affine_to_local_model(
            matrix=matrix,
            offset=offset,
            fixed_start_zyx=child_fixed,
            moving_start_zyx=child_moving,
            crop_shape_zyx=child_shape,
        )
        points = np.array([[0, 0, 0], [207, 527, 527], [80, 120, 300]], dtype=float)
        parent_center = (parent_shape - 1) / 2
        expected = (points + child_moving - parent_moving - parent_center) @ parent_matrix.T
        expected += parent_center + parent_translation + parent_fixed
        child_center = (child_shape - 1) / 2
        actual = (points - child_center) @ child_matrix.T + child_center + child_translation + child_fixed
        np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_shorter_native_domains_cover_parent_without_gaps_or_extension() -> None:
    windows = _windows((2500, 960, 960), core_shape_zyx=(160, 480, 480), window_shape_zyx=(208, 528, 528))
    children = [
        row for row in windows if row["quadrant"] == "qy0_qx0" and row["start_zyx"][0] in (960, 1120, 1280)
    ]
    coverage = np.zeros(2500, dtype=bool)
    for row in children:
        start = row["start_zyx"][0]
        coverage[start : start + row["shape_zyx"][0]] = True
    np.testing.assert_array_equal(np.flatnonzero(coverage), np.arange(960, 1488))


def test_child_selects_only_accepted_containing_parent(tmp_path: Path) -> None:
    tile = "235-4-405.074.ome.zarr"
    path = tmp_path / "235-4-405.074.qy0_qx0.z00960.json"
    row = {
        "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
        "fixed_fused": str(tmp_path / "fixed.zarr"),
        "moving_path": str(tmp_path / "moving.zarr"),
        "moving_channel": 0,
        "status": "accepted",
        "moving_tile": tile,
        "moving_start_l0_zyx": [960, 0, 0],
        "moving_stop_l0_zyx": [1488, 528, 528],
    }
    path.write_text(json.dumps(row))
    assert (
        _parent_window_path(
            tmp_path,
            tile_name=tile,
            quadrant="qy0_qx0",
            start_zyx=[1120, 0, 0],
            shape_zyx=[208, 528, 528],
            fixed_fused=tmp_path / "fixed.zarr",
            moving_path=tmp_path / "moving.zarr",
            moving_channel=0,
        )
        == path
    )
    with pytest.raises(ValueError, match="exactly one accepted parent"):
        _parent_window_path(
            tmp_path,
            tile_name=tile,
            quadrant="qy0_qx0",
            start_zyx=[1400, 0, 0],
            shape_zyx=[208, 528, 528],
            fixed_fused=tmp_path / "fixed.zarr",
            moving_path=tmp_path / "moving.zarr",
            moving_channel=0,
        )
    row["status"] = "rejected"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="exactly one accepted parent"):
        _parent_window_path(
            tmp_path,
            tile_name=tile,
            quadrant="qy0_qx0",
            start_zyx=[1120, 0, 0],
            shape_zyx=[208, 528, 528],
            fixed_fused=tmp_path / "fixed.zarr",
            moving_path=tmp_path / "moving.zarr",
            moving_channel=0,
        )


@pytest.mark.parametrize("requested,cached", [(True, False), (False, True)])
def test_summary_rejects_different_local_registration_criterion(
    tmp_path: Path, requested: bool, cached: bool
) -> None:
    from squisher_lightsheet.fused_fixed import validate_method6_summary

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "cache_config": {
                    "native_method": "method6",
                    "fit_intensity_transform": "log1p",
                    "level0_initializer": "level2-method8",
                    "moving_channel": 0,
                    "local_3d_refinement": cached,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="local_3d_refinement"):
        validate_method6_summary(
            summary,
            moving_position=tmp_path / "moving.json",
            moving_source_position=tmp_path / "source.json",
            fixed_fused=tmp_path / "fixed.zarr",
            moving_channel=0,
            fixed_mask_threshold=1000,
            local_3d_refinement=requested,
        )


def _write_parent(tmp_path: Path, z: int = 1972) -> Path:
    path = tmp_path / f"235-4-405.074.qy0_qx0.z{z:05d}.json"
    path.write_text(json.dumps({
        "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
        "fixed_fused": str(tmp_path / "fixed.zarr"),
        "moving_path": str(tmp_path / "moving.zarr"),
        "moving_channel": 0, "status": "accepted",
        "moving_tile": "235-4-405.074.ome.zarr",
        "moving_start_l0_zyx": [z, 0, 0],
        "moving_stop_l0_zyx": [z + 528, 528, 528],
    }))
    return path


def _split_parents(tmp_path: Path) -> list[dict]:
    from squisher_lightsheet._backends.fused_fixed_sweep import _parent_windows

    return _parent_windows(
        tmp_path, tile_name="235-4-405.074.ome.zarr", tile_shape_zyx=(2500, 960, 960),
        core_shape_zyx=(80, 480, 480), window_shape_zyx=(128, 528, 528),
        fixed_fused=tmp_path / "fixed.zarr", moving_path=tmp_path / "moving.zarr", moving_channel=0,
    )


def test_parent_relative_tail_children_preserve_complete_source_domain(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    children = _split_parents(tmp_path)
    assert [r["start_zyx"][0] for r in children] == [1972, 2052, 2132, 2212, 2292, 2372]
    assert all(r["parent_path"] == parent for r in children)
    coverage = np.zeros(2500, bool)
    for row in children:
        assert row["quadrant"] == "qy0_qx0"
        assert row["shape_zyx"] == [128, 528, 528]
        coverage[row["start_zyx"][0]:row["start_zyx"][0] + 128] = True
    np.testing.assert_array_equal(np.flatnonzero(coverage), np.arange(1972, 2500))


def test_overlapping_parents_keep_explicit_parent_initializers(tmp_path: Path) -> None:
    a, b = _write_parent(tmp_path, 1920), _write_parent(tmp_path, 1972)
    children = _split_parents(tmp_path)
    assert len(children) == 12
    assert {r["parent_path"] for r in children} == {a, b}
    # The existing implicit lookup still rejects this ambiguous child.
    with pytest.raises(ValueError, match="exactly one accepted parent"):
        _parent_window_path(
            tmp_path, tile_name="235-4-405.074.ome.zarr", quadrant="qy0_qx0",
            start_zyx=[2000, 0, 0], shape_zyx=[128, 528, 528],
            fixed_fused=tmp_path / "fixed.zarr", moving_path=tmp_path / "moving.zarr", moving_channel=0,
        )


@pytest.mark.parametrize("change, message", [
    ({"moving_channel": 1}, "source/reference/channel"),
    ({"moving_stop_l0_zyx": [2501, 528, 528]}, "native source bounds"),
    ({"moving_stop_l0_zyx": [2500, 527, 528]}, "unchanged XY"),
    ({"artifact_type": "unrelated"}, "unsupported accepted parent"),
])
def test_parent_split_rejects_invalid_input_contract(tmp_path: Path, change: dict, message: str) -> None:
    path = _write_parent(tmp_path)
    row = json.loads(path.read_text())
    row.update(change)
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=message):
        _split_parents(tmp_path)


def test_parent_split_omits_rejected_parents_and_rejects_identity_collision(tmp_path: Path) -> None:
    path = _write_parent(tmp_path)
    row = json.loads(path.read_text())
    row["status"] = "rejected"
    path.write_text(json.dumps(row))
    assert _split_parents(tmp_path) == []
    _write_parent(tmp_path, 0)
    _write_parent(tmp_path, 400)
    with pytest.raises(ValueError, match="duplicate child identity"):
        _split_parents(tmp_path)


@pytest.mark.parametrize("with_parent", [True, False])
@pytest.mark.parametrize("refusion", [True, False])
def test_method6_parent_split_cli_requires_and_forwards_parent(tmp_path: Path, monkeypatch, with_parent: bool, refusion: bool) -> None:
    from typer.testing import CliRunner
    from squisher_lightsheet.cli import app

    for name in ("fixed.json", "moving.json", "source.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "fixed.zarr").mkdir()
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return tmp_path / "summary.json", tmp_path / "registration.json"

    monkeypatch.setattr("squisher_lightsheet.cli.run_fused_fixed_method6", run)
    args = [
        "cross-register", "method6",
        "--fixed-position", str(tmp_path / "fixed.json"),
        "--moving-position", str(tmp_path / "moving.json"),
        "--moving-source-position", str(tmp_path / "source.json"),
        "--fixed-fused", str(tmp_path / "fixed.zarr"),
        "--output-dir", str(tmp_path / "out"),
        "--output-registration", str(tmp_path / "out.json"),
        "--fixed-mask-threshold", "1000", "--native-lib-dir", str(tmp_path),
        "--source-label", "405", "--target-label", "561", "--split-parent-windows",
    ]
    if with_parent:
        args.extend(["--parent-window-dir", str(tmp_path)])
    if refusion:
        args.extend(["--parent-reference-fused", str(tmp_path / "fixed.zarr")])
    result = CliRunner().invoke(app, args)
    if with_parent:
        assert result.exit_code == 0, result.output
        assert calls[0]["split_parent_windows"] is True
        assert calls[0]["parent_window_dir"] == tmp_path
        assert calls[0]["parent_reference_fused"] == (tmp_path / "fixed.zarr" if refusion else None)
    else:
        assert result.exit_code == 2
        assert "requires --parent-window-dir" in result.output
        assert not calls


@pytest.mark.parametrize("requested,cached", [(True, False), (False, True)])
def test_summary_rejects_different_parent_grid(tmp_path: Path, requested: bool, cached: bool) -> None:
    from squisher_lightsheet.fused_fixed import validate_method6_summary

    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"cache_config": {
        "native_method": "method6", "fit_intensity_transform": "log1p",
        "level0_initializer": "window-parent", "moving_channel": 0,
        "split_parent_windows": cached,
    }}))
    with pytest.raises(ValueError, match="split_parent_windows"):
        validate_method6_summary(
            path, moving_position=tmp_path / "moving.json",
            moving_source_position=tmp_path / "source.json",
            fixed_fused=tmp_path / "fixed.zarr", moving_channel=0,
            fixed_mask_threshold=1000, level0_initializer="window-parent",
            split_parent_windows=requested,
        )


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("refusion", [False, True])
def test_worker_cache_preserves_parent_grid_and_matches_sweep(tmp_path: Path, split: bool, refusion: bool) -> None:
    from squisher_lightsheet._backends.fused_fixed_sweep import (
        _cache_config_from_args, _namespace_for_task, build_parser,
    )
    from squisher_lightsheet._backends.fused_fixed_trial import _cache_config

    reference = tmp_path / "reference.zarr"
    if refusion:
        _write_reference_grid(reference)
    args = build_parser().parse_args([
        "--parent-window-dir", str(tmp_path), "--level0-initializer", "window-parent",
        *(["--split-parent-windows"] if split else []),
        *(["--parent-reference-fused", str(reference)] if refusion else []),
    ])
    task = {"tile": "074", "start_zyx": [960, 0, 0], "shape_zyx": [128, 528, 528]}
    worker = _namespace_for_task(
        vars(args), task, mode="level0-from-level2", output=tmp_path / "window.json",
        level2_result=None, device=0,
    )
    actual = _cache_config(worker)
    actual.pop("initializer_sha256")
    assert actual == _cache_config_from_args(args)
    assert actual["version"].endswith("_selected_parent_affine_v1")
    if refusion:
        assert actual["parent_reference_fused"] == str(reference)
        metadata = json.loads((reference / "zarr.json").read_text())
        metadata["attributes"]["review_revision"] = 2
        (reference / "zarr.json").write_text(json.dumps(metadata))
        assert _cache_config_from_args(args) != actual


def _write_reference_grid(path: Path, *, origin_z: float = 0, unit: str = "micrometer") -> None:
    import zarr

    root = zarr.open_group(path, mode="w")
    root.create_array("0", shape=(2500, 960, 960), chunks=(24, 960, 960), dtype="uint16")
    root.attrs.update({"squisher_complete": True, "ome": {"version": "0.5", "multiscales": [{
        "axes": [{"name": a, "type": "space", "unit": unit} for a in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [0.6, 0.3, 0.3]},
            {"type": "translation", "translation": [origin_z, 0, 0]},
        ]}],
    }]}})


def test_parent_refusion_requires_explicit_matching_reference(tmp_path: Path) -> None:
    from squisher_lightsheet._backends.fused_fixed_sweep import _parent_windows

    old, current = tmp_path / "fixed.zarr", tmp_path / "current.zarr"
    _write_reference_grid(old)
    _write_reference_grid(current)
    _write_parent(tmp_path)
    kwargs = dict(tile_name="235-4-405.074.ome.zarr", tile_shape_zyx=(2500, 960, 960),
                  core_shape_zyx=(80, 480, 480), window_shape_zyx=(128, 528, 528),
                  fixed_fused=current, moving_path=tmp_path / "moving.zarr", moving_channel=0)
    with pytest.raises(ValueError, match="source/reference/channel"):
        _parent_windows(tmp_path, **kwargs)
    children = _parent_windows(tmp_path, parent_reference_fused=old, **kwargs)
    assert len(children) == 6
    assert {r["parent_path"] for r in children} == {tmp_path / "235-4-405.074.qy0_qx0.z01972.json"}


@pytest.mark.parametrize("origin_z, unit", [(1.0, "micrometer"), (0.0, "nanometer")])
def test_parent_refusion_rejects_different_physical_grid(tmp_path: Path, origin_z: float, unit: str) -> None:
    from squisher_lightsheet._backends.fused_fixed_sweep import _parent_windows

    old, current = tmp_path / "fixed.zarr", tmp_path / "current.zarr"
    _write_reference_grid(old)
    _write_reference_grid(current, origin_z=origin_z, unit=unit)
    _write_parent(tmp_path)
    with pytest.raises(ValueError, match="grid"):
        _parent_windows(tmp_path, tile_name="235-4-405.074.ome.zarr", tile_shape_zyx=(2500, 960, 960),
                        core_shape_zyx=(80, 480, 480), window_shape_zyx=(128, 528, 528),
                        fixed_fused=current, parent_reference_fused=old,
                        moving_path=tmp_path / "moving.zarr", moving_channel=0)


def test_parent_refusion_rejects_incomplete_reference(tmp_path: Path) -> None:
    import zarr
    from squisher_lightsheet._backends.fused_fixed_sweep import _parent_windows

    old, current = tmp_path / "fixed.zarr", tmp_path / "current.zarr"
    _write_reference_grid(old)
    _write_reference_grid(current)
    zarr.open_group(old, mode="r+").attrs["squisher_complete"] = False
    _write_parent(tmp_path)
    with pytest.raises(ValueError, match="complete"):
        _parent_windows(tmp_path, tile_name="235-4-405.074.ome.zarr", tile_shape_zyx=(2500, 960, 960),
                        core_shape_zyx=(80, 480, 480), window_shape_zyx=(128, 528, 528),
                        fixed_fused=current, parent_reference_fused=old,
                        moving_path=tmp_path / "moving.zarr", moving_channel=0)


def _write_recovery_initializer(directory: Path, tile: str, quadrant: str, start_zyx: list[int]) -> None:
    directory.mkdir(exist_ok=True)
    (directory / f"{tile.removesuffix('.ome.zarr')}.{quadrant}.z{start_zyx[0]:05d}.json").write_text(json.dumps({
        "artifact_type": "lightsheet.fused_fixed_native_spatial_recovery_initializer.v1",
        "status": "accepted", "moving_tile": tile, "moving_start_l0_zyx": start_zyx,
        "level_factor_zyx": [1, 1, 1], "local_matrix_zyx": np.eye(3).tolist(),
        "local_translation_zyx": [0.0, 0.0, 0.0],
    }))


def _interpolated_tasks(tmp_path: Path, monkeypatch, tile: str) -> list[dict]:
    from types import SimpleNamespace

    from squisher_lightsheet._backends import fused_fixed_trial as trial
    from squisher_lightsheet._backends.fused_fixed_sweep import _build_tasks, build_parser

    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps({"tiles": [{"tile": tile}]}))
    monkeypatch.setattr(trial, "source_record_with_geometry", lambda source, record: record)
    monkeypatch.setattr(trial, "tile_record_from_position_record", lambda record: SimpleNamespace(
        shape_zyx=(2256, 960, 960), path=tmp_path / "moving.zarr"))
    args = build_parser().parse_args([
        "--moving-position", str(positions), "--moving-source-position", str(positions),
        "--output-dir", str(tmp_path / "out"), "--core-shape-zyx", "80,480,480",
        "--window-shape-zyx", "128,528,528", "--level0-initializer", "window-interpolated",
        "--window-initializer-dir", str(tmp_path / "initializers"),
    ])
    return _build_tasks(args)


def test_native_rerun_tasks_include_off_grid_split_parent_children(tmp_path: Path, monkeypatch) -> None:
    # Children of a tail parent at z 1728 start at 1808, off the regular 80-step grid of a 2256-plane tile.
    tile = "235-2-405-2.017.ome.zarr"
    for start in ([1440, 432, 0], [1808, 432, 0], [2128, 0, 432]):
        _write_recovery_initializer(tmp_path / "initializers", tile, f"qy{start[1] // 432}_qx{start[2] // 432}", start)
    tasks = _interpolated_tasks(tmp_path, monkeypatch, tile)
    assert sorted((t["quadrant"], t["start_zyx"]) for t in tasks) == [
        ("qy0_qx1", [2128, 0, 432]), ("qy1_qx0", [1440, 432, 0]), ("qy1_qx0", [1808, 432, 0])]
    assert all(t["shape_zyx"] == [128, 528, 528] for t in tasks)


@pytest.mark.parametrize("start", [[2200, 432, 0], [1808, 500, 0]])
def test_native_rerun_rejects_initializer_outside_its_tile_window(tmp_path: Path, monkeypatch, start) -> None:
    tile = "235-2-405-2.017.ome.zarr"
    _write_recovery_initializer(tmp_path / "initializers", tile, "qy1_qx0", start)
    with pytest.raises(ValueError, match="initializer"):
        _interpolated_tasks(tmp_path, monkeypatch, tile)
