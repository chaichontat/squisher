from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import tifffile
from typer.testing import CliRunner

from lightsheet_psf.cli import app
from lightsheet_psf.centroid_qc import choose_spots
from lightsheet_psf.io import load_image_zyx
from lightsheet_psf.median import add_quality_flags, build_medians, crop_and_align
from lightsheet_psf.radial import peak_normalize, radial_average_xy, sum_normalize
from lightsheet_psf.shear import shear_for_volume


def sheared_gaussian(shape: tuple[int, int, int], slope_x_px_per_z: float) -> np.ndarray:
    z_size, y_size, x_size = shape
    zc = (z_size - 1) / 2.0
    yc = (y_size - 1) / 2.0
    xc = (x_size - 1) / 2.0
    z, y, x = np.indices(shape, dtype=np.float64)
    x_center = xc + slope_x_px_per_z * (z - zc)
    return np.exp(-(((z - zc) / 2.0) ** 2 + ((y - yc) / 2.2) ** 2 + ((x - x_center) / 2.4) ** 2))


def test_shear_for_volume_recovers_known_xz_slope() -> None:
    volume = sheared_gaussian((13, 19, 21), slope_x_px_per_z=-0.25)

    metrics = shear_for_volume(
        volume,
        spacing_zyx_um=(1.0, 1.0, 1.0),
        z_fraction_threshold=0.1,
        mode="sum_y",
    )

    assert metrics["slope_x_px_per_z_plane"] == pytest.approx(-0.25, abs=0.02)
    assert metrics["slope_x_um_per_z_um"] == pytest.approx(-0.25, abs=0.02)


def test_radial_average_xy_preserves_center_and_normalizers_validate() -> None:
    volume = sheared_gaussian((9, 17, 17), slope_x_px_per_z=0.0)

    radial = radial_average_xy(volume)

    assert np.unravel_index(int(np.argmax(radial)), radial.shape) == (4, 8, 8)
    assert float(peak_normalize(radial).max()) == pytest.approx(1.0)
    assert float(sum_normalize(radial).sum()) == pytest.approx(1.0)


def test_cli_help_has_no_dataset_defaults() -> None:
    result = CliRunner().invoke(app, ["radialize", "--help"])

    assert result.exit_code == 0
    assert "20260606" not in result.stdout
    assert "Image_58" not in result.stdout
    assert "--spacing-zyx-um" in result.stdout


def test_cli_exposes_acquisition_side_commands_without_dataset_defaults() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("detect-beads", "make-median", "render-centroid-qc"):
        assert command in result.stdout
    assert "Image_58" not in result.stdout


def test_load_image_zyx_reads_ome_tiff_stack(tmp_path) -> None:
    volume = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
    path = tmp_path / "stack.ome.tif"
    tifffile.imwrite(path, volume, metadata={"axes": "ZYX"})

    loaded = load_image_zyx(path)

    assert np.array_equal(loaded, volume)


def test_measure_shear_cli_writes_report_with_manifest(tmp_path) -> None:
    median = sheared_gaussian((9, 13, 15), slope_x_px_per_z=0.2).astype(np.float32)
    median_path = tmp_path / "median.tif"
    crops_path = tmp_path / "crops.npy"
    quality_path = tmp_path / "quality.csv"
    report_path = tmp_path / "shear.json"
    png_path = tmp_path / "shear.png"
    tifffile.imwrite(median_path, median, metadata={"axes": "ZYX"})
    np.save(crops_path, np.stack([median, median], axis=0))
    quality_path.write_text("good_quality\ntrue\nfalse\n")

    result = CliRunner().invoke(
        app,
        [
            "measure-shear",
            "--median-psf",
            str(median_path),
            "--crops-npy",
            str(crops_path),
            "--quality-csv",
            str(quality_path),
            "--output-json",
            str(report_path),
            "--output-png",
            str(png_path),
            "--spacing-zyx-um",
            "1,1,1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    report = json.loads(report_path.read_text())
    assert report["command"] == "measure-shear"
    assert report["good_quality_count"] == 1
    assert report["median_psf_shear"]["slope_x_px_per_z_plane"] == pytest.approx(0.2, abs=0.04)
    assert "input_sha256" in report
    assert png_path.exists()
    assert json.loads(result.stdout)["median_psf_shear"]["slope_x_px_per_z_plane"] == pytest.approx(
        0.2, abs=0.04
    )


def test_radialize_cli_writes_sum_normalized_outputs(tmp_path) -> None:
    volume = sheared_gaussian((9, 15, 15), slope_x_px_per_z=-0.15).astype(np.float32)
    peak_path = tmp_path / "peak.tif"
    raw_path = tmp_path / "raw.tif"
    out_dir = tmp_path / "out"
    tifffile.imwrite(peak_path, peak_normalize(volume).astype(np.float32), metadata={"axes": "ZYX"})
    tifffile.imwrite(raw_path, (volume * 100).astype(np.float32), metadata={"axes": "ZYX"})

    result = CliRunner().invoke(
        app,
        [
            "radialize",
            "--peak-psf",
            str(peak_path),
            "--raw-psf",
            str(raw_path),
            "--out-dir",
            str(out_dir),
            "--spacing-zyx-um",
            "1,1,1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    report = json.loads((out_dir / "deskew_radial_reskew_report.json").read_text())
    assert report["command"] == "radialize"
    assert report["sum_normalized_sums"]["radial_reskewed"] == pytest.approx(1.0, abs=1e-6)
    assert (out_dir / "radial_symmetric_reskewed_sumnorm.tif").exists()
    assert "output_sha256" in report
    assert report["outputs"]["radial_reskewed"]["sumnorm"].endswith("radial_symmetric_reskewed_sumnorm.tif")
    assert json.loads(result.stdout)["outputs"]["radial_reskewed"]["sumnorm"].endswith(
        "radial_symmetric_reskewed_sumnorm.tif"
    )


def test_radialize_rejects_nan_raw_peak(tmp_path) -> None:
    volume = sheared_gaussian((9, 15, 15), slope_x_px_per_z=0.0).astype(np.float32)
    peak_path = tmp_path / "peak.tif"
    raw_path = tmp_path / "raw_nan.tif"
    tifffile.imwrite(peak_path, peak_normalize(volume).astype(np.float32), metadata={"axes": "ZYX"})
    tifffile.imwrite(raw_path, np.full_like(volume, np.nan, dtype=np.float32), metadata={"axes": "ZYX"})

    result = CliRunner().invoke(
        app,
        [
            "radialize",
            "--peak-psf",
            str(peak_path),
            "--raw-psf",
            str(raw_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--spacing-zyx-um",
            "1,1,1",
        ],
    )

    assert result.exit_code != 0
    assert "positive finite peak" in result.output


def test_add_quality_flags_and_crop_alignment_on_synthetic_stack() -> None:
    stack = np.zeros((9, 15, 15), dtype=np.float32)
    stack[4, 7, 7] = 100.0
    beads = pd.DataFrame(
        {
            "id": [1],
            "z": [4.0],
            "y": [7.0],
            "x": [7.0],
            "peak_intensity": [100.0],
        }
    )

    quality = add_quality_flags(beads, stack.shape, (5, 5, 5), min_xy_distance=2.0)
    crop = crop_and_align(stack, quality.iloc[0], (5, 5, 5))

    assert bool(quality.loc[0, "basic_quality"])
    assert bool(quality.loc[0, "full_crop"])
    assert np.unravel_index(int(np.nanargmax(crop)), crop.shape) == (2, 2, 2)


def test_build_medians_outputs_quality_columns_and_peaknorm_crops() -> None:
    stack = np.zeros((25, 35, 35), dtype=np.float32)
    coords = [(12, 10, 10), (12, 24, 10), (12, 10, 24), (12, 24, 24)]
    for z, y, x in coords:
        stack[z, y, x] = 100.0
        for dz, value in ((1, 70.0), (2, 50.0), (3, 35.0), (4, 20.0)):
            stack[z - dz, y, x] = value
            stack[z + dz, y, x] = value
        stack[z, y - 1, x] = 50.0
        stack[z, y + 1, x] = 50.0
        stack[z, y, x - 1] = 50.0
        stack[z, y, x + 1] = 50.0
    beads = pd.DataFrame(
        {
            "id": list(range(len(coords))),
            "z": [float(z) for z, _, _ in coords],
            "y": [float(y) for _, y, _ in coords],
            "x": [float(x) for _, _, x in coords],
            "peak_intensity": [100.0] * len(coords),
        }
    )
    quality = add_quality_flags(beads, stack.shape, (9, 9, 9), min_xy_distance=5.0)

    crops, raw_median, norm_median = build_medians(
        stack,
        quality,
        (9, 9, 9),
        size_mad_mult=2.0,
        z_asym_mad_mult=1.5,
        z_near_ratio_floor=0.0,
        require_full_crop=False,
        min_z_fwhm_px=None,
        max_z_fwhm_px=None,
        min_z_support_span_px=None,
        max_z_support_span_px=None,
        min_z_pre_tail_fraction=None,
        max_central_z_peak_offset_px=None,
    )

    assert crops.shape[0] == len(coords)
    assert raw_median.shape == (9, 9, 9)
    assert float(norm_median.max()) == pytest.approx(1.0)
    assert int(quality["good_quality"].sum()) == len(coords)


def test_build_medians_can_reject_padded_or_axially_streaked_crops() -> None:
    stack = np.zeros((25, 35, 35), dtype=np.float32)
    coords = [(12, 10, 10), (12, 24, 10)]
    for z, y, x in coords:
        stack[z, y, x] = 100.0
        for dz, value in ((1, 70.0), (2, 50.0), (3, 35.0), (4, 20.0)):
            stack[z - dz, y, x] = value
            stack[z + dz, y, x] = value
    beads = pd.DataFrame(
        {
            "id": [0, 1],
            "z": [12.0, 12.0],
            "y": [10.0, 24.0],
            "x": [10.0, 24.0],
            "peak_intensity": [100.0, 100.0],
        }
    )
    quality = add_quality_flags(beads, stack.shape, (9, 9, 9), min_xy_distance=5.0)

    build_medians(
        stack,
        quality,
        (9, 9, 9),
        size_mad_mult=2.0,
        z_asym_mad_mult=1.5,
        z_near_ratio_floor=0.0,
        require_full_crop=True,
        min_z_fwhm_px=None,
        max_z_fwhm_px=20.0,
        min_z_support_span_px=None,
        max_z_support_span_px=20.0,
        min_z_pre_tail_fraction=None,
        max_central_z_peak_offset_px=None,
    )

    assert "full_crop_required_ok" in quality
    assert "z_fwhm_max_ok" in quality
    assert "z_support_span_max_ok" in quality


def test_choose_spots_returns_representative_good_spots() -> None:
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "good_quality": [True, True, False],
            "full_crop": [True, True, True],
            "peak_intensity": [10.0, 20.0, 30.0],
            "z": [5.0, 6.0, 7.0],
            "y": [5.0, 6.0, 7.0],
            "x": [5.0, 6.0, 7.0],
            "z_round": [5, 6, 7],
            "y_round": [5, 6, 7],
            "x_round": [5, 6, 7],
        }
    )

    selected = choose_spots(df, (5, 5, 5), (10, 10, 10))

    assert set(selected["id"]) == {0, 1}
