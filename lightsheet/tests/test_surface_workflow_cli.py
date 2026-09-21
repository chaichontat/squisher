from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import squisher_lightsheet.cli as cli_module
from squisher_lightsheet.cli import app


def _file(path: Path) -> Path:
    path.touch()
    return path


def test_eomes_profiles_cli_forwards_canonical_defaults(tmp_path: Path, monkeypatch) -> None:
    surface = _file(tmp_path / "surface.npz")
    chart = _file(tmp_path / "chart.npz")
    mask = _file(tmp_path / "mask.nrrd")
    source = tmp_path / "source.ome.zarr"
    source.mkdir()
    output = tmp_path / "profiles"
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return output / "metadata.json"

    monkeypatch.setattr(cli_module, "build_surface_profiles", fake_build)
    result = CliRunner().invoke(
        app,
        [
            "eomes-profiles",
            "--surface",
            str(surface),
            "--chart",
            str(chart),
            "--mask",
            str(mask),
            "--source-zarr",
            str(source),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "surface_path": surface,
        "chart_path": chart,
        "mask_path": mask,
        "source_zarr": source,
        "output_dir": output,
        "level": 2,
        "z_start": 0,
        "z_stop": None,
        "stride_zyx": (2, 4, 4),
        "radius_um": 20.0,
        "max_depth_um": 280.8,
        "depth_step_um": 4.8,
        "projection_domain": "full",
    }


def test_eomes_boundaries_cli_forwards_prod2_parameters(tmp_path: Path, monkeypatch) -> None:
    profiles = _file(tmp_path / "profiles.npz")
    surface = _file(tmp_path / "surface.npz")
    output = tmp_path / "boundaries"
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return output / "metadata.json"

    monkeypatch.setattr(cli_module, "build_surface_boundaries", fake_build)
    result = CliRunner().invoke(
        app,
        [
            "eomes-boundaries",
            "--profiles",
            str(profiles),
            "--surface",
            str(surface),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "profiles_path": profiles,
        "surface_path": surface,
        "output_dir": output,
        "inner_fraction": 0.60,
        "inner_smooth_um": 30.0,
        "outer_fraction": 0.75,
        "outer_smooth_um": 80.0,
        "min_crossing_samples": 5,
        "max_recovery_distance_um": 20.0,
        "trim_caps": True,
    }


def test_eomes_boundary_qc_cli_forwards_sections(tmp_path: Path, monkeypatch) -> None:
    surface = _file(tmp_path / "surface.npz")
    chart = _file(tmp_path / "chart.npz")
    boundaries = _file(tmp_path / "boundaries.npz")
    mask = _file(tmp_path / "mask.nrrd")
    source = tmp_path / "source.ome.zarr"
    source.mkdir()
    output = tmp_path / "qc"
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return output / "metadata.json"

    monkeypatch.setattr(cli_module, "build_surface_boundary_qc", fake_build)
    result = CliRunner().invoke(
        app,
        [
            "eomes-boundary-qc",
            "--surface",
            str(surface),
            "--chart",
            str(chart),
            "--boundaries",
            str(boundaries),
            "--mask",
            str(mask),
            "--source-zarr",
            str(source),
            "--output-dir",
            str(output),
            "--intrinsic-z-index",
            "4",
            "--intrinsic-z-index",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "surface_path": surface,
        "chart_path": chart,
        "boundaries_path": boundaries,
        "mask_path": mask,
        "source_zarr": source,
        "output_dir": output,
        "source_level": 2,
        "intrinsic_z_indices": [4, 8],
        "intrinsic_z_positions_um": None,
        "grid_stride": 2,
    }
