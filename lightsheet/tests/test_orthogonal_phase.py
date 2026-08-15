from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet import orthogonal_phase


def tile(name: str, *, y: float = 0.0, x: float = 0.0):
    return orthogonal_phase.legacy.TileRecord(
        tile=name,
        side="L",
        path=Path(f"/{name}.ome.zarr"),
        translation_zyx_um=np.asarray([0.0, y, x]),
        scale_zyx_um=np.asarray([1.0, 1.0, 1.0]),
        shape_zyx=np.asarray([10, 10, 10]),
        axes="ZYX",
    )


def test_render_dumb_plane_max_composites_adjacent_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiles = [tile("left"), tile("right")]
    monkeypatch.setattr(
        orthogonal_phase,
        "_sample_plane",
        lambda record, **_kwargs: np.full((2, 3), 1 if record.tile == "left" else 2),
    )

    image, coverage, contributors = orthogonal_phase.render_dumb_plane(
        tiles,
        channel=0,
        plane="zx",
        z_world=np.asarray([1.0, 2.0]),
        lateral_world=np.asarray([1.0, 2.0, 3.0]),
        cross_world=1.0,
    )

    assert np.all(image == 2)
    assert np.all(coverage)
    assert contributors == ["left", "right"]


def test_select_center_neighborhood_uses_center_tile_and_adjacent_tiles() -> None:
    tiles = [
        tile(f"tile-{row}-{column}", y=row * 10, x=column * 10) for row in range(3) for column in range(3)
    ]

    neighborhood, center = orthogonal_phase.select_center_neighborhood(tiles)

    assert center.tile == "tile-1-1"
    assert {record.tile for record in neighborhood} == {record.tile for record in tiles}


def test_analyze_plane_preserves_manual_joint_phase_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiles = [tile("left"), tile("right", x=5.0)]
    monkeypatch.setattr(
        orthogonal_phase,
        "render_dumb_plane",
        lambda *_args, **_kwargs: (
            np.arange(100, dtype=np.float32).reshape(10, 10),
            np.ones((10, 10), dtype=bool),
            ["left", "right"],
        ),
    )

    monkeypatch.setattr(
        orthogonal_phase,
        "phasecorr_shift_gpu",
        lambda *_args, **_kwargs: (np.asarray([2.0, 0.0, 5.0]), {}),
    )
    correlations = iter([0.1, 0.2])
    monkeypatch.setattr(
        orthogonal_phase.phase_metrics,
        "corrcoef_on_mask",
        lambda *_args, **_kwargs: next(correlations),
    )
    shifts = []
    expanded = orthogonal_phase._expanded_shifted_pair

    def capture_shift(fixed, moving, shift):
        shifts.append(np.asarray(shift).copy())
        return expanded(fixed, moving, shift)

    monkeypatch.setattr(orthogonal_phase, "_expanded_shifted_pair", capture_shift)
    monkeypatch.setattr(
        orthogonal_phase.qc,
        "write_overlay_scaled",
        lambda *_args, **_kwargs: None,
    )
    output_dir = tmp_path / "orthogonal"
    output_dir.mkdir()

    result, _panels = orthogonal_phase._analyze_plane(
        tiles,
        tiles,
        fixed_channel=0,
        moving_channel=0,
        plane="zx",
        cross_world=1.0,
        region_start=np.asarray([0.0, 0.0, 0.0]),
        region_stop=np.asarray([10.0, 10.0, 10.0]),
        z_spacing=1.0,
        lateral_spacing=1.0,
        max_shift_um=10.0,
        fixed_transform="identity",
        moving_transform="identity",
        output_dir=output_dir,
    )

    assert np.array_equal(shifts[2], [2.0, 5.0])
    assert np.array_equal(shifts[3], [2.0, 5.0])
    assert result["shift_to_apply_moving_um"] == [2.0, 5.0]
    assert result["diagnostic_only_lateral_shift_um"] == 5.0


def test_orthogonal_phase_applies_consensus_z_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiles = [
        tile(f"tile-{row}-{column}", y=row * 10, x=column * 10) for row in range(3) for column in range(3)
    ]
    monkeypatch.setattr(
        orthogonal_phase,
        "position_tiles",
        lambda _payload, **_kwargs: tiles,
    )

    def fake_analyze(*_args, plane: str, output_dir: Path, **_kwargs):
        z_shift, lateral_shift = {
            "zx": (-55.0, 48.0),
            "zy": (-67.0, 61.0),
        }[plane]
        return (
            {
                "shift_to_apply_moving_um": [z_shift, lateral_shift],
                "final_applied_axes": ["z"],
                "diagnostic_only_lateral_shift_um": lateral_shift,
            },
            [],
        )

    monkeypatch.setattr(orthogonal_phase, "_analyze_plane", fake_analyze)
    monkeypatch.setattr(
        orthogonal_phase.qc,
        "write_contact_sheet",
        lambda output, _panels: output.write_bytes(b"qc"),
    )

    result = orthogonal_phase.run_orthogonal_dumb_phase(
        fixed_payload={},
        moving_payload={},
        fixed_tile_dir=None,
        moving_tile_dir=None,
        fixed_channel=0,
        moving_channel=0,
        fixed_transform="identity",
        moving_transform="identity",
        output_dir=tmp_path / "orthogonal",
        max_shift_um=100.0,
    )

    summary = json.loads(result.summary.read_text())
    assert result.z_residual_um == -61.0
    assert summary["applied_z_residual_um"] == -61.0
    assert summary["z_disagreement_um"] == 12.0
    assert summary["lateral_components_applied"] is False
    assert summary["results"]["zx"]["diagnostic_only_lateral_shift_um"] == 48.0
    assert summary["results"]["zy"]["diagnostic_only_lateral_shift_um"] == 61.0


def test_orthogonal_phase_records_disagreement_for_human_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiles = [
        tile(f"tile-{row}-{column}", y=row * 10, x=column * 10) for row in range(3) for column in range(3)
    ]
    monkeypatch.setattr(
        orthogonal_phase,
        "position_tiles",
        lambda _payload, **_kwargs: tiles,
    )
    monkeypatch.setattr(
        orthogonal_phase,
        "_analyze_plane",
        lambda *_args, plane, **_kwargs: (
            {"shift_to_apply_moving_um": [-30.0 if plane == "zx" else -60.0, 0.0]},
            [],
        ),
    )

    monkeypatch.setattr(
        orthogonal_phase.qc,
        "write_contact_sheet",
        lambda output, _panels: output.write_bytes(b"qc"),
    )
    result = orthogonal_phase.run_orthogonal_dumb_phase(
        fixed_payload={},
        moving_payload={},
        fixed_tile_dir=None,
        moving_tile_dir=None,
        fixed_channel=0,
        moving_channel=0,
        fixed_transform="identity",
        moving_transform="identity",
        output_dir=tmp_path / "orthogonal",
        max_shift_um=100.0,
    )

    summary = json.loads(result.summary.read_text())
    assert result.z_residual_um == -45.0
    assert summary["z_disagreement_um"] == 30.0
    assert summary["human_review_required"] is True
