from __future__ import annotations

import json

import pytest

from squisher_lightsheet import lr_alignment


def test_run_lr_dumb_stitch_alignment_writes_summary(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(lr_alignment, "create_position_file", lambda **kwargs: calls.append(("position", kwargs)))
    monkeypatch.setattr(lr_alignment, "rough_phase_align", lambda **kwargs: calls.append(("rough_phase", kwargs)))

    left = tmp_path / "TL"
    right = tmp_path / "TR"
    left.mkdir()
    right.mkdir()

    paths = lr_alignment.run_lr_dumb_stitch_alignment(
        left_dir=left,
        right_dir=right,
        output_prefix=tmp_path / "sample",
        channel=2,
    )

    assert [name for name, _kwargs in calls] == ["position", "rough_phase"]
    assert calls[1][1]["level"] == 2
    assert calls[1][1]["z_slab_planes"] == 8
    assert calls[1][1]["phase_downsample_zyx"] == (4, 4, 4)
    assert paths.metadata_position.name == "sample.positions.json"
    assert paths.phase_position.name == "sample.roughPhase.positions.json"
    assert paths.phase_qc_dir.name == "level2-rough-phase"
    assert paths.initial_overlay.name == "level2_metadata_initial_zyx_yellowOverlay_ch2.png"
    assert paths.corrected_overlay.name == "level2_phase_corrected_zyx_yellowOverlay_ch2.png"
    assert paths.phase_summary_json.name == "level2_zyx_phase_alignment_ch2.json"

    summary = json.loads(paths.summary_json.read_text())
    assert summary["workflow"] == "lr_dumb_stitch_alignment"
    assert summary["outputs"]["metadata_position"] == str(paths.metadata_position.resolve())
    assert summary["outputs"]["phase_position"] == str(paths.phase_position.resolve())
    assert summary["outputs"]["initial_overlay"] == str(paths.initial_overlay.resolve())
    assert summary["outputs"]["corrected_overlay"] == str(paths.corrected_overlay.resolve())
    assert summary["outputs"]["phase_summary_json"] == str(paths.phase_summary_json.resolve())
    assert summary["parameters"]["channel"] == 2
    assert summary["parameters"]["level"] == 2
    assert summary["parameters"]["xy_downsample_factor"] == 4
    assert summary["parameters"]["phase_dimensions"] == "zyx"
    assert summary["parameters"]["z_sampling"] == "native_center_z_slab"
    assert summary["parameters"]["z_slab_planes"] == 8
    assert summary["parameters"]["phase_downsample_zyx"] == [4, 4, 4]
    assert {"position", "rough_phase"} <= set(summary["commands"])
    assert "--z-slab-planes 8" in summary["commands"]["rough_phase"]
    assert "--phase-downsample-zyx 4,4,4" in summary["commands"]["rough_phase"]


def test_run_lr_dumb_stitch_alignment_validates_fraction(tmp_path) -> None:
    with pytest.raises(ValueError, match="overlap_fraction"):
        lr_alignment.run_lr_dumb_stitch_alignment(
            left_dir=tmp_path,
            right_dir=tmp_path,
            output_prefix=tmp_path / "sample",
            overlap_fraction=1.0,
            dry_run=True,
        )
