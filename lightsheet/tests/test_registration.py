from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from squisher_lightsheet.registration import register_tiles


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


def test_register_passes_translation_groupwise_transform(monkeypatch, tmp_path) -> None:
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
        groupwise_transform="translation",
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert captured["args"][captured["args"].index("--groupwise-transform") + 1] == "translation"


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


def test_register_rejects_non_default_level_for_current_legacy_cli(tmp_path) -> None:
    with pytest.raises(ValueError, match="only supports level 4"):
        register_tiles(
            run_dir=tmp_path,
            position_input=Path("positions.json"),
            registration_output=tmp_path / "registration.json",
            level=2,
            channels=(3,),
        )
