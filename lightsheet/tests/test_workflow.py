from __future__ import annotations

import json

import pytest

from squisher_lightsheet import workflow


def test_run_tltr_summary_records_actual_fused_channel_outputs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(workflow, "create_position_file", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "rough_phase_align", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "register_tiles", lambda **_kwargs: "registration command")
    monkeypatch.setattr(workflow, "render_registration_qc", lambda **_kwargs: "qc command")
    monkeypatch.setattr(workflow, "fuse_tiles", lambda **_kwargs: "fusion command")
    monkeypatch.setattr(workflow, "add_pyramids", lambda **_kwargs: "pyramid command")

    left = tmp_path / "TL"
    right = tmp_path / "TR"
    left.mkdir()
    right.mkdir()
    paths = workflow.run_tltr_workflow(
        left_dir=left,
        right_dir=right,
        output_prefix=tmp_path / "sample",
        channel=0,
        do_fuse=True,
        do_pyramid=True,
    )

    summary = json.loads(paths.summary_json.read_text())
    assert summary["outputs"]["fusion_base_output"].endswith("fused.ome.zarr")
    assert summary["outputs"]["fusion_outputs"] == [str((paths.registration_dir / "fused.ch0.ome.zarr").resolve())]
    assert "search_margin_px" in summary["parameters"]
    assert "phase_upsample_factor" in summary["parameters"]
    assert {"position", "rough_phase", "registration", "qc", "fusion", "pyramid"} <= set(summary["commands"])


def test_run_tltr_pyramid_requires_fusion(tmp_path) -> None:
    with pytest.raises(ValueError, match="do_pyramid requires do_fuse"):
        workflow.run_tltr_workflow(
            left_dir=tmp_path,
            right_dir=tmp_path,
            output_prefix=tmp_path / "sample",
            do_pyramid=True,
            dry_run=True,
        )
