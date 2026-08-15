from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from typer.testing import CliRunner

import squisher_deconv.cli as cli
from squisher_deconv.basic import BasicFitOutputs
from squisher_deconv.cli import app
from squisher_deconv.deconvolution import infer_psf_halo, infer_psf_halo_many
from squisher_deconv.planning import output_sidecar_path


def test_cli_help_exposes_gpu_options_without_fiducials() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--basic" in result.output
    assert "--iter" in result.output
    assert "--engine" not in result.output
    assert "--n-fids" not in result.output


def test_basic_cli_runs_joint_autotune_darkfield_workflow(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    src = tmp_path / "tile.ome.tif"
    tifffile.imwrite(
        src,
        np.zeros((1, 2, 4, 4), dtype=np.uint16),
        ome=True,
        metadata={"axes": "CZYX"},
        photometric="minisblack",
    )
    captured: dict[str, object] = {}

    def capture_workflow(**kwargs):
        captured.update(kwargs)
        out_dir = Path(kwargs["out_dir"])
        return BasicFitOutputs(
            profile_paths=(out_dir / "sample-ch0.pkl",),
            flatfield_paths=(out_dir / "sample-ch0-flatfield.tif",),
            darkfield_paths=(out_dir / "sample-ch0-darkfield.tif",),
            png_paths=(out_dir / "sample-ch0.png",),
            manifest=out_dir / "sample-sampling.json",
        )

    monkeypatch.setattr(cli, "fit_basic_profiles", capture_workflow)

    result = runner.invoke(
        app,
        [
            "basic",
            str(src),
            "--out-dir",
            str(tmp_path / "basic"),
            "--label",
            "sample",
            "--channels",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["inputs"] == [src]
    assert captured["samples"] == 500
    assert captured["cache_samples_per_channel"] == 500
    assert captured["exclude_blank_slices"] is True
    assert captured["exclude_edge_slices"] is True
    assert captured["device"] == "cuda"
    assert str(tmp_path / "basic" / "sample-sampling.json") in result.output


def test_basic_cli_does_not_expose_device_override() -> None:
    runner = CliRunner()

    help_result = runner.invoke(app, ["basic", "--help"])
    override_result = runner.invoke(app, ["basic", "--device", "cpu"])

    assert help_result.exit_code == 0
    assert "--device" not in help_result.output
    assert override_result.exit_code != 0
    assert "No such option: --device" in override_result.output


def test_cli_rejects_partial_basic_profiles(tmp_path) -> None:
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
            "--psf",
            str(psf),
            "--basic",
            str(basic),
        ],
    )

    assert result.exit_code != 0
    assert "Expected exactly 2 --basic profile path" in result.output


def test_cli_requires_one_psf_per_channel(tmp_path) -> None:
    runner = CliRunner()
    src = tmp_path / "tile.tif"
    psf = tmp_path / "psf.tif"
    tifffile.imwrite(src, np.zeros((2, 3, 3), dtype=np.uint16), photometric="minisblack")
    tifffile.imwrite(psf, np.ones((3, 3, 3), dtype=np.float32), photometric="minisblack")

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
        ],
    )

    assert result.exit_code != 0
    assert "Expected exactly 2 --psf path" in result.output


def test_deconvolver_factory_passes_iterations_to_gpu_backend(tmp_path) -> None:
    from squisher_deconv.gpu import CupyDeconvolverFactory

    psf = tmp_path / "psf.tif"
    basic = tmp_path / "basic.pkl"

    factory = cli._build_deconvolver_factory(
        basic=[basic],
        channels=1,
        psfs=[psf],
        iterations=3,
    )

    assert isinstance(factory, CupyDeconvolverFactory)
    assert factory.iterations == 3


def test_deconvolver_factory_allows_no_basic_profiles(tmp_path) -> None:
    from squisher_deconv.gpu import CupyDeconvolverFactory

    psf = tmp_path / "psf.tif"

    factory = cli._build_deconvolver_factory(
        basic=None,
        channels=1,
        psfs=[psf],
        iterations=1,
    )

    assert isinstance(factory, CupyDeconvolverFactory)
    assert factory.basic_paths == ()


def test_infer_psf_halo_uses_psf_z_extent(tmp_path) -> None:
    psf = tmp_path / "psf.tif"
    tifffile.imwrite(psf, np.ones((4, 3, 3), dtype=np.float32), photometric="minisblack")

    assert infer_psf_halo(psf) == 6


def test_infer_psf_halo_many_uses_largest_z_extent(tmp_path) -> None:
    psf_small = tmp_path / "psf-small.tif"
    psf_large = tmp_path / "psf-large.tif"
    tifffile.imwrite(psf_small, np.ones((3, 3, 3), dtype=np.float32), photometric="minisblack")
    tifffile.imwrite(psf_large, np.ones((5, 3, 3), dtype=np.float32), photometric="minisblack")

    assert infer_psf_halo_many([psf_small, psf_large]) == 8


def test_sample_scale_cli_infers_halo_without_eager_deconvolver_init(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    src = tmp_path / "tile.tif"
    psf = tmp_path / "psf.tif"
    captured: dict[str, object] = {}
    tifffile.imwrite(src, np.zeros((2, 3, 3), dtype=np.uint16), photometric="minisblack")
    tifffile.imwrite(psf, np.ones((4, 3, 3), dtype=np.float32), photometric="minisblack")

    def factory_builder(**kwargs):
        def fail_if_initialized(device: int):
            pytest.fail(
                f"deconvolver should not be initialized during CLI halo inference for device {device}"
            )

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
            "--devices",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["halo"] == 6
    assert captured["iterations"] == 1
    assert captured["deconvolver"] is None
    assert callable(captured["deconvolver_factory"])


def test_run_cli_requires_scaling(tmp_path) -> None:
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
        ],
    )

    assert result.exit_code != 0
    assert "run requires --scaling" in result.output


def test_qc_cli_renders_selected_finished_tiles_without_opening_every_tiff(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    deconv_dir = raw_dir / "squisher-deconv-run-u16"
    qc_dir = tmp_path / "qc"
    raw_dir.mkdir()
    deconv_dir.mkdir()

    for index in range(4):
        raw = np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4) + index
        deconv = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4) + index + 1
        raw_path = raw_dir / f"Image_99.{index:03d}.ome.tif"
        deconv_path = deconv_dir / f"Image_99.{index:03d}.ome.zarr"
        tifffile.imwrite(raw_path, raw, photometric="minisblack", metadata={"axes": "CZYX"})
        root = zarr.open_group(str(deconv_path), mode="w", zarr_format=2)
        array = root.create_array("0", data=deconv.reshape(2, 2, 4, 4), chunks=(1, 1, 4, 4))
        array.attrs["_ARRAY_DIMENSIONS"] = ["c", "z", "y", "x"]
        output_sidecar_path(deconv_path).write_text(
            json.dumps({"provenance": {"run_settings": {"channels": 2}}})
        )

    import squisher_deconv.qc as qc_module

    assert qc_module.decon_channel_count(deconv_dir / "Image_99.000.ome.zarr") == 2

    real_tiff_file = qc_module.tifffile.TiffFile
    opened: list[str] = []

    def recording_tiff_file(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_tiff_file(path, *args, **kwargs)

    monkeypatch.setattr(qc_module.tifffile, "TiffFile", recording_tiff_file)

    result = runner.invoke(
        app,
        [
            "qc",
            "--raw-dir",
            str(raw_dir),
            "--deconv-dir",
            str(deconv_dir),
            "--qc-dir",
            str(qc_dir),
            "--image-prefix",
            "Image_99",
            "--tile-count",
            "2",
            "--channel",
            "1",
            "--z-plane",
            "0",
            "--z-plane",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (qc_dir / "manifest.json").exists()
    assert len(list(qc_dir.glob("*.before-after.png"))) == 2
    assert len(opened) == 2
    assert sorted(opened) == [
        "Image_99.000.ome.tif",
        "Image_99.003.ome.tif",
    ]
