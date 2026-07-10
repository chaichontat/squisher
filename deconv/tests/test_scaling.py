from __future__ import annotations

import numpy as np
import pytest
import tifffile

from squisher_deconv.scaling import ScalingParameters, collate_scaling, load_scaling, quantize_global, write_scaling_artifacts


def test_scaling_artifacts_match_synthetic_quantiles(tmp_path) -> None:
    sample = tmp_path / "sample.tif"
    data = np.array(
        [
            [[0, 1], [2, 3]],
            [[10, 11], [12, 13]],
            [[4, 5], [6, 7]],
            [[14, 15], [16, 17]],
        ],
        dtype=np.float32,
    )
    tifffile.imwrite(sample, data, photometric="minisblack")

    params = collate_scaling(
        [sample],
        channels=2,
        out_dir=tmp_path / "scale",
        p_low=0.0,
        p_high=1.0,
        gamma=1.0,
        bins=4,
        manifest={"seed": 1},
    )

    assert np.allclose(params.offset, [0.0, 10.0])
    assert np.allclose(params.scale, [65535.0 / 7.0, 65535.0 / 7.0])
    assert (tmp_path / "scale" / "scaling.json").exists()
    assert (tmp_path / "scale" / "scaling.txt").exists()
    assert (tmp_path / "scale" / "histogram.csv").exists()
    assert (tmp_path / "scale" / "sample-manifest.json").exists()
    assert (tmp_path / "scale" / "scaling-qc.png").exists()

    loaded = load_scaling(tmp_path / "scale" / "scaling.json")
    quantized = quantize_global(data.reshape(2, 2, 2, 2), loaded)
    assert quantized.shape == (4, 2, 2)
    assert quantized.dtype == np.uint16


def test_scaling_manifest_serialization_fails_before_writing_artifacts(tmp_path) -> None:
    params = ScalingParameters(
        offset=np.array([0], dtype=np.float32),
        scale=np.array([1], dtype=np.float32),
        p_low=0,
        p_high=1,
        gamma=1,
        i_max=65535,
    )

    with pytest.raises(TypeError, match="Sample scaling manifest must be JSON serializable"):
        write_scaling_artifacts(
            tmp_path / "scale",
            params,
            histograms=[(np.array([1]), np.array([0, 1]))],
            manifest={"bad": object()},
            sample_paths=[],
        )

    assert not (tmp_path / "scale" / "scaling.json").exists()
    assert not (tmp_path / "scale" / "sample-manifest.json").exists()


def test_load_scaling_rejects_offset_scale_channel_mismatch(tmp_path) -> None:
    path = tmp_path / "scaling.json"
    path.write_text(
        """
        {
          "offset": [0.0, 1.0],
          "scale": [2.0],
          "p_low": 0.0,
          "p_high": 1.0,
          "gamma": 1.0,
          "i_max": 65535
        }
        """
    )

    with pytest.raises(ValueError, match="offset/scale channel count mismatch"):
        load_scaling(path)


def test_load_scaling_rejects_non_finite_and_non_positive_scale(tmp_path) -> None:
    path = tmp_path / "scaling.json"
    path.write_text(
        """
        {
          "offset": [0.0],
          "scale": [0.0],
          "p_low": 0.0,
          "p_high": 1.0,
          "gamma": 1.0,
          "i_max": 65535
        }
        """
    )

    with pytest.raises(ValueError, match="scale must be strictly positive"):
        load_scaling(path)

    path.write_text(
        """
        {
          "offset": [NaN],
          "scale": [1.0],
          "p_low": 0.0,
          "p_high": 1.0,
          "gamma": 1.0,
          "i_max": 65535
        }
        """
    )

    with pytest.raises(ValueError, match="offset must contain only finite values"):
        load_scaling(path)


def test_collate_scaling_rejects_non_finite_sample_values(tmp_path) -> None:
    sample = tmp_path / "sample.tif"
    tifffile.imwrite(sample, np.array([[[np.nan]]], dtype=np.float32), photometric="minisblack")

    with pytest.raises(ValueError, match="sample values must contain only finite values"):
        collate_scaling(
            [sample],
            channels=1,
            out_dir=tmp_path / "scale",
            p_low=0.0,
            p_high=1.0,
            gamma=1.0,
            bins=4,
            manifest={"seed": 1},
        )

    assert not (tmp_path / "scale" / "scaling.json").exists()
