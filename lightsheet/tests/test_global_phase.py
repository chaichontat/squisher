from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from squisher_lightsheet import cli as cli_module
from squisher_lightsheet import global_phase
from squisher_lightsheet.cli import app
from squisher_lightsheet.orthogonal_phase import OrthogonalPhaseResult


def canvas(
    image: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (0.6, 4.0, 4.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> global_phase.GlobalPhaseCanvas:
    return global_phase.GlobalPhaseCanvas(
        image=image.astype(np.float32),
        coverage=np.ones(image.shape, dtype=bool),
        spacing_zyx_um=np.asarray(spacing, dtype=np.float64),
        global_min_zyx_um=np.asarray(origin, dtype=np.float64),
        slab={"native_z_spacing_um": spacing[0]},
        tile_count=2,
    )


def write_position(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source": "test",
                "tiles": [
                    {
                        "tile": "tile-0",
                        "translation_um": {"z": 1.0, "y": 2.0, "x": 3.0},
                    },
                    {
                        "tile": "tile-1",
                        "translation_um": {"z": 4.0, "y": 5.0, "x": 6.0},
                    },
                ],
            }
        )
        + "\n"
    )


def test_global_phase_applies_phase_and_origin_shift_to_moving_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_position = tmp_path / "fixed.json"
    moving_position = tmp_path / "moving.json"
    write_position(fixed_position)
    write_position(moving_position)
    fixed = canvas(np.arange(120).reshape(4, 5, 6), origin=(1.2, 8.0, 12.0))
    moving = canvas(np.arange(120).reshape(4, 5, 6), origin=(0.6, 4.0, 8.0))

    def fake_render(position: Path, **_kwargs) -> global_phase.GlobalPhaseCanvas:
        return fixed if position == fixed_position else moving

    monkeypatch.setattr(global_phase, "render_position_canvas", fake_render)
    phase_kwargs = {}

    def fake_phasecorr(*_args, **kwargs):
        phase_kwargs.update(kwargs)
        return np.asarray([0.0, 1.0, -1.0]), {"peak_value": 0.9}

    monkeypatch.setattr(global_phase, "phasecorr_shift_gpu", fake_phasecorr)
    correlations = iter([0.1, 0.9])
    monkeypatch.setattr(
        global_phase.phase_metrics,
        "corrcoef_on_mask",
        lambda *_args, **_kwargs: next(correlations),
    )
    orthogonal_kwargs = {}

    def fake_orthogonal(**kwargs) -> OrthogonalPhaseResult:
        orthogonal_kwargs.update(kwargs)
        orthogonal_dir = kwargs["output_dir"]
        orthogonal_dir.mkdir(parents=True)
        summary = orthogonal_dir / "orthogonal.summary.json"
        contact = orthogonal_dir / "orthogonal.png"
        summary.write_text("{}\n")
        contact.write_text("qc\n")
        return OrthogonalPhaseResult(-2.0, summary.resolve(), contact.resolve())

    monkeypatch.setattr(global_phase, "run_orthogonal_dumb_phase", fake_orthogonal)
    output_position = tmp_path / "run" / "global-phase.positions.json"
    result = global_phase.run_global_phase(
        fixed_position=fixed_position,
        moving_position=moving_position,
        output_dir=output_position.parent,
        output_position=output_position,
        fixed_intensity_transform="identity",
    )

    output = json.loads(result.output_position.read_text())
    expected_xy_shift = np.asarray([0.0, 8.0, 0.0])
    expected_shift = np.asarray([-2.0, 8.0, 0.0])
    assert np.allclose(
        [output["tiles"][0]["translation_um"][axis] for axis in "zyx"],
        np.asarray([1.0, 2.0, 3.0]) + expected_shift,
    )
    assert np.allclose(
        [output["tiles"][1]["translation_um"][axis] for axis in "zyx"],
        np.asarray([4.0, 5.0, 6.0]) + expected_shift,
    )
    assert json.loads(fixed_position.read_text())["tiles"][0]["translation_um"] == {
        "z": 1.0,
        "y": 2.0,
        "x": 3.0,
    }
    summary = json.loads(result.summary.read_text())
    assert summary["phase_shift_to_apply_moving_level_px_zyx"] == [0.0, 1.0, -1.0]
    assert summary["global_phase_applied_axes"] == ["y", "x"]
    assert np.allclose(summary["total_shift_to_apply_moving_zyx_um"], expected_shift)
    assert np.allclose(summary["xy_total_shift_to_apply_moving_zyx_um"], expected_xy_shift)
    assert summary["orthogonal_z_residual_um"] == -2.0
    assert summary["orthogonal_lateral_components_applied"] is False
    assert np.allclose(
        [orthogonal_kwargs["moving_payload"]["tiles"][0]["translation_um"][axis] for axis in "zyx"],
        np.asarray([1.0, 2.0, 3.0]) + expected_xy_shift,
    )
    assert result.before_overlay.exists()
    assert result.after_overlay.exists()
    assert np.allclose(phase_kwargs["search_center_zyx"], [0.0, -1.0, -1.0])
    assert np.allclose(phase_kwargs["max_shift_from_center_zyx"], [0.0, 25.0, 25.0])


def test_global_phase_rejects_mismatched_physical_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_position = tmp_path / "fixed.json"
    moving_position = tmp_path / "moving.json"
    write_position(fixed_position)
    write_position(moving_position)
    fixed = canvas(np.ones((3, 4, 5)), spacing=(0.6, 4.0, 4.0))
    moving = canvas(np.ones((3, 4, 5)), spacing=(0.6, 8.0, 8.0))
    monkeypatch.setattr(
        global_phase,
        "render_position_canvas",
        lambda position, **_kwargs: fixed if position == fixed_position else moving,
    )

    with pytest.raises(ValueError, match="matching physical spacing"):
        global_phase.run_global_phase(
            fixed_position=fixed_position,
            moving_position=moving_position,
            output_dir=tmp_path / "run",
            output_position=tmp_path / "run" / "positions.json",
        )


def test_global_phase_rejects_translation_without_coverage_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_position = tmp_path / "fixed.json"
    moving_position = tmp_path / "moving.json"
    write_position(fixed_position)
    write_position(moving_position)
    rendered = canvas(np.ones((3, 4, 5)))
    monkeypatch.setattr(global_phase, "render_position_canvas", lambda *_args, **_kwargs: rendered)
    monkeypatch.setattr(
        global_phase,
        "phasecorr_shift_gpu",
        lambda *_args, **_kwargs: (np.asarray([0.0, 10.0, 0.0]), {"peak_value": 0.1}),
    )

    with pytest.raises(ValueError, match="no fixed/moving coverage overlap"):
        global_phase.run_global_phase(
            fixed_position=fixed_position,
            moving_position=moving_position,
            output_dir=tmp_path / "run",
            output_position=tmp_path / "run" / "positions.json",
        )


def test_global_phase_rejects_nonimproving_correlation_without_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_position = tmp_path / "fixed.json"
    moving_position = tmp_path / "moving.json"
    write_position(fixed_position)
    write_position(moving_position)
    rendered = canvas(np.arange(120).reshape(4, 5, 6))
    monkeypatch.setattr(global_phase, "render_position_canvas", lambda *_args, **_kwargs: rendered)
    monkeypatch.setattr(
        global_phase,
        "phasecorr_shift_gpu",
        lambda *_args, **_kwargs: (np.asarray([0.0, 0.0, 0.0]), {"peak_value": 0.1}),
    )
    correlations = iter([0.9, 0.8])
    monkeypatch.setattr(
        global_phase.phase_metrics,
        "corrcoef_on_mask",
        lambda *_args, **_kwargs: next(correlations),
    )
    output_dir = tmp_path / "run"
    output_position = output_dir / "positions.json"

    with pytest.raises(ValueError, match="did not improve correlation"):
        global_phase.run_global_phase(
            fixed_position=fixed_position,
            moving_position=moving_position,
            output_dir=output_dir,
            output_position=output_position,
        )

    assert not output_position.exists()
    assert not (output_dir / "global-phase.summary.json").exists()


def test_global_phase_rejects_out_of_window_phase_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_position = tmp_path / "fixed.json"
    moving_position = tmp_path / "moving.json"
    write_position(fixed_position)
    write_position(moving_position)
    rendered = canvas(np.ones((3, 4, 5)))
    monkeypatch.setattr(global_phase, "render_position_canvas", lambda *_args, **_kwargs: rendered)
    monkeypatch.setattr(
        global_phase,
        "phasecorr_shift_gpu",
        lambda *_args, **_kwargs: (np.asarray([0.0, 26.0, 0.0]), {"peak_value": 0.1}),
    )
    output_position = tmp_path / "run" / "positions.json"

    with pytest.raises(ValueError, match="out-of-window shift"):
        global_phase.run_global_phase(
            fixed_position=fixed_position,
            moving_position=moving_position,
            output_dir=output_position.parent,
            output_position=output_position,
        )

    assert not output_position.exists()


def test_expanded_before_qc_respects_physical_origin_offset() -> None:
    image = np.ones((2, 3, 4), dtype=np.float32)
    fixed, moving = global_phase.expanded_shifted_pair(
        image,
        image,
        np.asarray([0.0, 2.0, -1.0]),
    )

    assert fixed.shape == moving.shape == (2, 5, 5)
    assert int(np.count_nonzero(fixed * moving)) == 6


def test_global_phase_log1p_rejects_negative_intensities() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        global_phase.apply_intensity_transform(np.asarray([-1.0], dtype=np.float32), "log1p")


def test_render_position_canvas_resolves_tiff_names_from_tile_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    position = tmp_path / "positions.json"
    position.write_text('{"tiles":[{"tile":"sample.000.ome.tif","path":"/raw/sample.000.ome.tif"}]}\n')
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    (tile_dir / "sample.000.ome.zarr").mkdir()
    captured = {}
    monkeypatch.setattr(global_phase, "register_jpegxr_codec", lambda: None)

    def fake_load_tiles(payload):
        captured.update(payload)
        return ["tile"]

    geometry = SimpleNamespace(
        level_spacing_zyx_um=np.asarray([9.6, 4.0, 4.0]),
        global_min_zyx_um=np.asarray([1.0, 2.0, 3.0]),
    )
    monkeypatch.setattr(global_phase.legacy, "load_tiles", fake_load_tiles)
    monkeypatch.setattr(global_phase.legacy, "build_geometry", lambda *_args, **_kwargs: geometry)
    monkeypatch.setattr(
        global_phase.legacy,
        "render_center_z_slab_canvases",
        lambda *_args, **_kwargs: (
            {"L": np.ones((2, 3, 4)), "R": np.zeros((2, 3, 4))},
            {"L": np.ones((2, 3, 4), dtype=bool), "R": np.zeros((2, 3, 4), dtype=bool)},
            [],
            {"native_z_spacing_um": 0.6, "slab_range_z_px": [3, 5]},
        ),
    )

    rendered = global_phase.render_position_canvas(
        position, tile_dir=tile_dir, channel=0, level=4, z_slab_planes=2
    )

    assert captured["tiles"][0]["path"] == str(tile_dir / "sample.000.ome.zarr")
    assert captured["tiles"][0]["side"] == "L"
    assert rendered.spacing_zyx_um.tolist() == [0.6, 4.0, 4.0]
    assert rendered.global_min_zyx_um.tolist() == [2.8, 2.0, 3.0]


def test_cross_register_global_phase_cli_forwards_contract_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_position = tmp_path / "fixed.json"
    moving_position = tmp_path / "moving.json"
    write_position(fixed_position)
    write_position(moving_position)
    output_dir = tmp_path / "run"
    fixed_tile_dir = tmp_path / "fixed-tiles"
    moving_tile_dir = tmp_path / "moving-tiles"
    fixed_tile_dir.mkdir()
    moving_tile_dir.mkdir()
    captured = {}

    def fake_run_global_phase(**kwargs) -> global_phase.GlobalPhaseResult:
        captured.update(kwargs)
        output_dir.mkdir()
        result = global_phase.GlobalPhaseResult(
            output_position=kwargs["output_position"].resolve(),
            summary=(output_dir / "global-phase.summary.json").resolve(),
            fixed_mip=(output_dir / "fixed.phase.tif").resolve(),
            moving_mip=(output_dir / "moving.phase.tif").resolve(),
            before_overlay=(output_dir / "before.png").resolve(),
            after_overlay=(output_dir / "after.png").resolve(),
            orthogonal_summary=(output_dir / "orthogonal" / "orthogonal.summary.json").resolve(),
            orthogonal_contact_sheet=(output_dir / "orthogonal" / "orthogonal.png").resolve(),
        )
        result.orthogonal_summary.parent.mkdir()
        for path in (
            result.output_position,
            result.summary,
            result.fixed_mip,
            result.moving_mip,
            result.before_overlay,
            result.after_overlay,
            result.orthogonal_summary,
            result.orthogonal_contact_sheet,
        ):
            path.write_text("{}\n")
        return result

    monkeypatch.setattr(cli_module, "run_global_phase", fake_run_global_phase)
    result = CliRunner().invoke(
        app,
        [
            "cross-register",
            "global-phase",
            "--fixed-position",
            str(fixed_position),
            "--moving-position",
            str(moving_position),
            "--output-dir",
            str(output_dir),
            "--fixed-tile-dir",
            str(fixed_tile_dir),
            "--moving-tile-dir",
            str(moving_tile_dir),
            "--fixed-channel",
            "2",
            "--moving-channel",
            "1",
            "--level",
            "3",
            "--z-slab-planes",
            "16",
            "--fft-highpass-sigma-zyx",
            "1,2,3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["fixed_position"] == fixed_position
    assert captured["moving_position"] == moving_position
    assert captured["fixed_tile_dir"] == fixed_tile_dir
    assert captured["moving_tile_dir"] == moving_tile_dir
    assert captured["fixed_channel"] == 2
    assert captured["moving_channel"] == 1
    assert captured["level"] == 3
    assert captured["z_slab_planes"] == 16
    assert captured["fixed_intensity_transform"] == "log1p"
    assert captured["moving_intensity_transform"] == "identity"
    assert captured["fft_highpass_sigma_zyx"] == (1.0, 2.0, 3.0)
    assert captured["max_residual_shift_um"] == 100.0
    assert captured["orthogonal_lateral_factor"] == 4
    manifest = json.loads((output_dir / "cross-register.manifest.json").read_text())
    assert manifest["stages"]["global-phase"]["fixed_channel"] == 2
    assert manifest["stages"]["global-phase"]["orthogonal_contact_sheet"] == str(
        (output_dir / "orthogonal" / "orthogonal.png").resolve()
    )


def test_cross_register_group_exposes_global_phase() -> None:
    result = CliRunner().invoke(app, ["cross-register", "--help"])

    assert result.exit_code == 0
    assert "global-phase" in result.stdout


def test_cross_register_global_phase_rejects_output_alias_before_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_position = tmp_path / "fixed.json"
    moving_position = tmp_path / "moving.json"
    write_position(fixed_position)
    write_position(moving_position)
    original = fixed_position.read_text()
    monkeypatch.setattr(
        cli_module,
        "run_global_phase",
        lambda **_kwargs: pytest.fail("global phase must not run with aliased paths"),
    )

    result = CliRunner().invoke(
        app,
        [
            "cross-register",
            "global-phase",
            "--fixed-position",
            str(fixed_position),
            "--moving-position",
            str(moving_position),
            "--output-dir",
            str(tmp_path / "run"),
            "--output-position",
            str(fixed_position),
            "--overwrite",
        ],
    )

    assert result.exit_code == 2
    assert "Cross-register paths must be distinct" in result.output
    assert "aliases fixed position" in result.output
    assert fixed_position.read_text() == original
