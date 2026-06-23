from __future__ import annotations

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

import squisher_deconv.cli as cli
from squisher_deconv.cli import app
from squisher_deconv.deconvolution import infer_psf_halo


def test_cli_help_exposes_basic_engine_options_without_fiducials() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--basic" in result.output
    assert "--engine" in result.output
    assert "--n-fids" not in result.output


def test_gpu_engine_requires_one_basic_profile_per_channel(tmp_path) -> None:
    runner = CliRunner()
    src = tmp_path / "tile.tif"
    psf = tmp_path / "psf.tif"
    basic = tmp_path / "basic-c0.pkl"
    tifffile.imwrite(src, np.zeros((2, 3, 3), dtype=np.uint16), photometric="minisblack")
    tifffile.imwrite(psf, np.ones((3, 3, 3), dtype=np.float32), photometric="minisblack")
    basic.write_bytes(b"not read before validation")

    result = runner.invoke(
        app,
        [
            "sample-scale",
            str(src),
            "--out-dir",
            str(tmp_path / "scale"),
            "--planes",
            "1",
            "--channels",
            "2",
            "--psf",
            str(psf),
            "--basic",
            str(basic),
            "--engine",
            "gpu",
        ],
    )

    assert result.exit_code != 0
    assert "requires exactly 2 --basic profile path" in result.output


def test_sample_scale_cli_runs_with_explicit_scipy_debug_engine(tmp_path) -> None:
    runner = CliRunner()
    src = tmp_path / "tile.tif"
    psf = tmp_path / "psf.tif"
    tifffile.imwrite(
        src,
        np.arange(2 * 2 * 3 * 3, dtype=np.uint16).reshape(4, 3, 3),
        photometric="minisblack",
    )
    psf_data = np.zeros((3, 3, 3), dtype=np.float32)
    psf_data[1, 1, 1] = 1.0
    tifffile.imwrite(psf, psf_data, photometric="minisblack")

    result = runner.invoke(
        app,
        [
            "sample-scale",
            str(src),
            "--out-dir",
            str(tmp_path / "scale"),
            "--planes",
            "1",
            "--channels",
            "2",
            "--psf",
            str(psf),
            "--engine",
            "scipy",
            "--halo",
            "0",
            "--p-low",
            "0",
            "--p-high",
            "1",
            "--bins",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "scale" / "scaling.json").exists()


def test_infer_psf_halo_uses_psf_z_extent(tmp_path) -> None:
    psf = tmp_path / "psf.tif"
    tifffile.imwrite(psf, np.ones((4, 3, 3), dtype=np.float32), photometric="minisblack")

    assert infer_psf_halo(psf) == 6


def test_sample_scale_cli_infers_halo_without_eager_deconvolver_init(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    src = tmp_path / "tile.tif"
    psf = tmp_path / "psf.tif"
    captured: dict[str, object] = {}
    tifffile.imwrite(src, np.zeros((2, 3, 3), dtype=np.uint16), photometric="minisblack")
    tifffile.imwrite(psf, np.ones((4, 3, 3), dtype=np.float32), photometric="minisblack")

    def factory_builder(**kwargs):
        def fail_if_initialized(device: int):
            pytest.fail(f"deconvolver should not be initialized during CLI halo inference for device {device}")

        return fail_if_initialized

    def capture_workflow(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "_build_deconvolver_factory", factory_builder)
    monkeypatch.setattr(cli, "sample_scale_workflow", capture_workflow)

    result = runner.invoke(
        app,
        [
            "sample-scale",
            str(src),
            "--out-dir",
            str(tmp_path / "scale"),
            "--planes",
            "1",
            "--channels",
            "1",
            "--psf",
            str(psf),
            "--engine",
            "scipy",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["halo"] == 6
    assert captured["deconvolver"] is None
    assert callable(captured["deconvolver_factory"])


def test_run_cli_requires_scaling_for_u16_output(tmp_path) -> None:
    runner = CliRunner()
    src = tmp_path / "tile.tif"
    psf = tmp_path / "psf.tif"
    tifffile.imwrite(src, np.zeros((2, 3, 3), dtype=np.uint16), photometric="minisblack")
    tifffile.imwrite(psf, np.ones((3, 3, 3), dtype=np.float32), photometric="minisblack")

    result = runner.invoke(
        app,
        [
            "run",
            str(src),
            "--out-dir",
            str(tmp_path / "out"),
            "--channels",
            "1",
            "--psf",
            str(psf),
            "--engine",
            "scipy",
        ],
    )

    assert result.exit_code != 0
    assert "--output-mode u16 requires --scaling" in result.output
