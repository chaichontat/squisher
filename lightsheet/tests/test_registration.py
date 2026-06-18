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
        shared_geometry_tracks=("track1", "track2"),
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    assert "--reference-registration-input" in captured["args"]
    assert str(tmp_path / "registration.track0.json") in captured["args"]
    assert captured["args"][captured["args"].index("--reference-geometry-mode") + 1] == "penalized-xy"
    assert captured["args"][captured["args"].index("--reference-xy-prior-weight") + 1] == "0.01"
    assert captured["args"][captured["args"].index("--shared-geometry-tracks") + 1] == "track1,track2"
