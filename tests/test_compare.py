import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pylibCZIrw import czi

from squisher.compare import (
    compare_czi_compression,
    comparison_output_dir,
    compression_levels,
    render_sweep_figure,
)


def test_compression_levels_include_end_of_sweep() -> None:
    assert compression_levels(min_level=0.65, max_level=0.90, level_step=0.05) == [
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]


def test_comparison_output_dir_uses_stem_folder_for_multi_tile_inputs(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"

    assert comparison_output_dir(czi_path, out_dir=None, crop_size=256, tile_count=2) == tmp_path / "sample"
    assert (
        comparison_output_dir(czi_path, out_dir=tmp_path / "comparisons", crop_size=256, tile_count=2)
        == tmp_path / "comparisons" / "sample"
    )
    assert (
        comparison_output_dir(czi_path, out_dir=tmp_path / "comparisons" / "sample", crop_size=256, tile_count=2)
        == tmp_path / "comparisons" / "sample"
    )


def test_comparison_output_dir_preserves_single_tile_legacy_default(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"

    assert comparison_output_dir(czi_path, out_dir=None, crop_size=128, tile_count=1) == (
        tmp_path / "sample_compression_sweep_128_crops"
    )
    assert comparison_output_dir(czi_path, out_dir=tmp_path / "custom", crop_size=128, tile_count=1) == (
        tmp_path / "custom"
    )


def test_compare_rejects_effective_tile_size_that_is_not_multiple_of_16(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"not checked before tile validation")

    with pytest.raises(ValueError, match="Effective TIFF tile size"):
        compare_czi_compression(czi_path, crop_size=17, tile_size=256)


def test_compare_czi_compression_writes_artifacts(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    with czi.create_czi(str(czi_path)) as writer:
        for t_index in range(2):
            assert writer.write(
                np.arange(16 * 16, dtype=np.uint16).reshape((16, 16)) + t_index * 1000,
                plane={"T": t_index, "C": 0, "Z": 0},
                scene=0,
            )

    out_dir = compare_czi_compression(
        czi_path,
        count=1,
        crop_size=16,
        min_level=0.90,
        max_level=0.90,
        level_step=0.05,
        tile_size=16,
        keep_encoded_tiffs=False,
        t=1,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["count"] == 1
    assert manifest["raw_dtypes"] == ["uint16"]
    assert len(manifest["records"]) == 1
    record = manifest["records"][0]
    assert record["t"] == 1
    assert record["channel"] == 0
    assert record["z"] == 0
    assert "_t1_" in record["png"]
    assert len(record["levels"]) == 1
    level_record = record["levels"][0]
    assert level_record["raw_dtype"] == "uint16"
    assert level_record["raw_bytes"] == 16 * 16 * 2
    assert level_record["encoded_tiff"] is None
    figure_path = out_dir / record["png"]
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0

    csv_rows = list(csv.DictReader((out_dir / "size_metrics.csv").open()))
    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert row["t"] == "1"
    assert row["channel"] == "0"
    assert row["z"] == "0"
    assert row["raw_dtype"] == level_record["raw_dtype"]
    assert int(row["raw_bytes"]) == level_record["raw_bytes"]
    assert int(row["encoded_bytes"]) == level_record["encoded_bytes"]
    assert float(row["mae"]) == pytest.approx(level_record["mae"])
    assert float(row["rmse"]) == pytest.approx(level_record["rmse"])
    assert len(list(out_dir.glob("*_sweep.png"))) == 1
    assert list((out_dir / "encoded_tiffs").glob("*.tif")) == []


def test_render_sweep_figure_uses_two_aligned_rows() -> None:
    raw = np.arange(16 * 16, dtype=np.uint16).reshape((16, 16))
    decoded_by_level = {
        0.65: raw + 1,
        0.70: raw + 2,
    }
    level_records = [
        {"level": 0.65, "encoded_bytes": 100, "raw_bytes": 512, "ratio_raw_to_encoded": 5.12, "mae": 1.0, "rmse": 1.0},
        {"level": 0.70, "encoded_bytes": 120, "raw_bytes": 512, "ratio_raw_to_encoded": 4.27, "mae": 2.0, "rmse": 2.0},
    ]

    figure, window, diff_limit = render_sweep_figure(
        raw,
        decoded_by_level=decoded_by_level,
        level_records=level_records,
        crop_label="CROP 00 TILE 000 C0 Z0000 Y0000 X0000",
    )

    assert figure.ndim == 3
    assert figure.shape[2] == 3
    assert figure.shape[0] > raw.shape[0] * 2
    assert figure.shape[1] > raw.shape[1] * 3
    assert np.unique(figure.reshape(-1, 3), axis=0).shape[0] > 1
    assert window[0] >= 0
    assert window[1] > window[0]
    assert diff_limit > 0
