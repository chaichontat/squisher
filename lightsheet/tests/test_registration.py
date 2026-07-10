from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.registration import register_tiles


def test_legacy_registration_defaults_to_level_zero(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", ["stitch", str(tmp_path)])

    args = legacy.parse_args()

    assert args.coarse_reg_res_levels == (0,)
    assert args.registration_pair_mode == "axis-aligned"
    assert args.n_parallel_pairwise_regs == 1
    assert legacy.resolve_pairwise_registration_jobs(args) == 1


def test_register_rejects_multi_track_position_inputs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "squisher_lightsheet.registration.legacy.read_position_input_tiles",
        lambda _path: [SimpleNamespace(tracks=[object(), object()])],
    )

    with pytest.raises(ValueError, match="does not support multi-track"):
        register_tiles(
            run_dir=tmp_path,
            position_input=Path("positions.json"),
            registration_output=tmp_path / "registration.json",
        )


def test_register_channel_selection_allows_multi_track_position_inputs(monkeypatch, tmp_path) -> None:
    captured = {}

    monkeypatch.setattr(
        "squisher_lightsheet.registration.legacy.read_position_input_tiles",
        lambda _path: [SimpleNamespace(tracks=[object(), object(), object()])],
    )

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
        log_file=tmp_path / "progress.log",
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert captured["args"][captured["args"].index("--channels") + 1] == "3"
    assert captured["args"][captured["args"].index("--log-file") + 1] == str(tmp_path / "progress.log")
    assert captured["args"][captured["args"].index("--coarse-reg-res-levels") + 1] == "0"
    assert captured["args"][captured["args"].index("--n-parallel-pairwise-regs") + 1] == "1"
    assert "--gpu-pairwise-phase-correlation" not in captured["args"]
    assert "--registration-pair-mode" not in captured["args"]
    assert "--reg-res-level" not in captured["args"]


def test_register_passes_explicit_registration_pair_file(monkeypatch, tmp_path) -> None:
    captured = {}
    pair_file = tmp_path / "pairs.json"
    pair_file.write_text('{"pairs": [[0, 1]]}\n')

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
        registration_pair_file=pair_file,
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert captured["args"][captured["args"].index("--registration-pair-file") + 1] == str(pair_file)


def test_register_passes_parallel_registration_controls(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
        pairwise_jobs=12,
        dask_num_workers=48,
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert captured["args"][captured["args"].index("--n-parallel-pairwise-regs") + 1] == "12"
    assert captured["args"][captured["args"].index("--dask-num-workers") + 1] == "48"


def test_register_passes_explicit_registration_level(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
        level=2,
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert captured["args"][captured["args"].index("--coarse-reg-res-levels") + 1] == "2"


def test_registration_source_level_uses_representative_tile(monkeypatch, tmp_path) -> None:
    tiles = [
        legacy.TileMetadata(
            path=tmp_path / f"tile{index}.ome.tif",
            shape=(5, 6, 7),
            axes="ZYX",
            spacing={"z": 1.0, "y": 0.5, "x": 0.5},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            channels=("0",),
            tracks=(),
            stage_scale=None,
            source_view=None,
        )
        for index in range(3)
    ]
    calls: list[Path] = []

    def fake_level_count(path: Path) -> int:
        calls.append(path)
        return 3

    monkeypatch.setattr(legacy, "tiff_series_level_count", fake_level_count)

    assert legacy.registration_source_level_for_tiles(tiles, (2,), None) == (2, 3, 2)
    assert calls == [tiles[0].path]


def test_source_tile_uses_opened_zyx_axes_for_czyx_metadata(tmp_path) -> None:
    tile = legacy.TileMetadata(
        path=tmp_path / "tile.ome.tif",
        shape=(2, 5, 6, 7),
        axes="CZYX",
        spacing={"z": 1.0, "y": 0.5, "x": 0.5},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        channels=("0", "1"),
        tracks=(),
        stage_scale=None,
        source_view=None,
    )

    source_tile = legacy.fusion_tile_for_source_array(tile, (5, 3, 4), source_level=1)

    assert source_tile.axes == "ZYX"
    assert source_tile.shape == (5, 3, 4)
    assert source_tile.spacing == {"z": 1.0, "y": 1.0, "x": 0.875}


def test_register_passes_registration_cache_controls(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
        registration_cache_max_gib=32.0,
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert captured["args"][captured["args"].index("--registration-cache-max-gib") + 1] == "32.0"


def test_register_defaults_to_translation_groupwise_transform(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert "--groupwise-transform" not in captured["args"]


def test_register_passes_explicit_rigid_groupwise_transform(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
        groupwise_transform="rigid",
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert captured["args"][captured["args"].index("--groupwise-transform") + 1] == "rigid"


def test_register_passes_mvs_quality_filter_controls(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
        mvs_post_quality_filter=False,
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert "--no-mvs-post-quality-filter" in captured["args"]


def test_register_passes_mvs_quality_threshold(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        channels=(3,),
        mvs_post_quality_threshold=0.05,
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert captured["args"][captured["args"].index("--mvs-post-quality-threshold") + 1] == "0.05"


def test_register_shared_geometry_passes_reference_flags(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_legacy_script(script_name, args, dry_run=False):
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "command"

    monkeypatch.setattr("squisher_lightsheet.registration.run_legacy_script", fake_run_legacy_script)

    register_tiles(
        run_dir=tmp_path,
        position_input=Path("positions.json"),
        registration_output=tmp_path / "registration.json",
        reference_registration_input=tmp_path / "registration.track0.json",
        reference_geometry_mode="penalized-xy",
        reference_xy_prior_weight=0.01,
        reference_initial_alignment="rigid",
        shared_geometry_tracks=("track1", "track2"),
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert "--reference-registration-input" in captured["args"]
    assert str(tmp_path / "registration.track0.json") in captured["args"]
    assert captured["args"][captured["args"].index("--reference-geometry-mode") + 1] == "penalized-xy"
    assert captured["args"][captured["args"].index("--reference-xy-prior-weight") + 1] == "0.01"
    assert captured["args"][captured["args"].index("--reference-initial-alignment") + 1] == "rigid"
    assert captured["args"][captured["args"].index("--shared-geometry-tracks") + 1] == "track1,track2"
