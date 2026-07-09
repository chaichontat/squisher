from __future__ import annotations

import json

import numpy as np
import pytest

from squisher_lightsheet import tile_quadrant_fusion
from squisher_lightsheet.tile_quadrant_fusion import (
    _materialize_crop_ome_zarr,
    _requested_source_window_zyx,
    export_tile_quadrant_materialized_chunks,
)


def test_requested_source_window_uses_image10_local_requested_fixed_window() -> None:
    row = {
        "fixed_start_zyx": [2400, 432, 432],
        "fixed_stop_zyx": [2928, 960, 888],
        "moving_start_zyx": [2400, 233, 501],
        "moving_stop_zyx": [2928, 761, 957],
        "requested_fixed_start_zyx": [2400, 432, 432],
        "requested_window_shape_zyx": [528, 528, 528],
    }

    start, stop, shape = _requested_source_window_zyx(row, np.asarray([3161, 960, 960], dtype=np.int64))

    np.testing.assert_array_equal(start, [2400, 432, 432])
    np.testing.assert_array_equal(stop, [2928, 960, 960])
    np.testing.assert_array_equal(shape, [528, 528, 528])


def test_materialize_crop_writes_image10_local_crop_without_padding(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_root = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source = source_root.create_array(
        "0",
        shape=(1, 4, 5, 6),
        chunks=(1, 2, 5, 6),
        dtype="uint16",
        dimension_names=("c", "z", "y", "x"),
    )
    source[:] = np.arange(np.prod(source.shape), dtype=np.uint16).reshape(source.shape)
    source_root.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]

    output_path = tmp_path / "out.ome.zarr"
    shape = _materialize_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr", "axes": "CZYX"},
        start_zyx=np.asarray([1, 1, 2], dtype=np.int64),
        stop_zyx=np.asarray([3, 4, 5], dtype=np.int64),
        spacing_zyx=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
    )

    output = zarr.open_group(str(output_path), mode="r")["0"]
    actual = output[:]
    expected = source[:, 1:3, 1:4, 2:5]
    assert tuple(shape) == expected.shape
    np.testing.assert_array_equal(actual, expected)
    codecs = output.metadata.to_dict()["codecs"]
    assert codecs[-1]["name"] == "blosc"
    assert codecs[-1]["configuration"]["cname"] == "zstd"
    assert codecs[-1]["configuration"]["clevel"] == 3


def test_export_composes_method8_transform_into_registered_affine(monkeypatch, tmp_path) -> None:
    window_dir = tmp_path / "windows"
    window_dir.mkdir()
    accepted = {
        "status": "accepted",
        "fixed_tile": "Image_14.000.ome.zarr",
        "moving_tile": "Image_10.000.ome.zarr",
        "quadrant": "qy1_qx1",
        "fixed_start_zyx": [2, 3, 4],
        "fixed_stop_zyx": [6, 7, 8],
        "moving_start_zyx": [1, 2, 3],
        "moving_stop_zyx": [5, 6, 7],
        "requested_fixed_start_zyx": [4, 2, 1],
        "requested_window_shape_zyx": [4, 4, 4],
        "full_matrix_zyx": [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]],
        "full_translation_zyx": [1.0, 2.0, 3.0],
    }
    rejected = accepted | {"status": "rejected", "rejection_reason": "fixed_threshold_mask_too_masked", "quadrant": "qy0_qx0"}
    (window_dir / "accepted.json").write_text(json.dumps(accepted) + "\n")
    (window_dir / "rejected.json").write_text(json.dumps(rejected) + "\n")
    moving_path = tmp_path / "moving.positions.json"
    source_dir = tmp_path / "source.ome.zarr"
    source_dir.mkdir()
    moving_path.write_text(
        json.dumps(
            {
                "units": "micrometer",
                "tiles": [
                    {
                        "tile": "Image_10.000.ome.zarr",
                        "path": str(source_dir),
                        "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                        "scale_um": {"z": 1.0, "y": 2.0, "x": 5.0},
                        "shape": [2, 10, 10, 10],
                        "axes": "CZYX",
                        "channels": ["0", "1"],
                        "tracks": [{"slug": "track0", "track_id": "all", "channels": [0, 1], "channel_names": ["0", "1"]}],
                    }
                ],
            }
        )
        + "\n"
    )
    fixed_path = tmp_path / "fixed.registration.json"
    fixed_registered_affine = np.eye(4, dtype=np.float64)
    fixed_registered_affine[:3, 3] = [7.0, 11.0, 13.0]
    fixed_path.write_text(
        json.dumps(
            {
                "input_dir": "/fixed",
                "registered_transform_key": "registered_affine",
                "tiles": [
                    {
                        "tile": "Image_14.000.ome.zarr",
                        "stage_translation_um": {"z": 100.0, "y": 200.0, "x": 300.0},
                        "stage_scale_um": {"z": 2.0, "y": 4.0, "x": 5.0},
                        "shape": [10, 10, 10],
                        "axes": "ZYX",
                        "registered_affine": {"matrix": fixed_registered_affine.tolist()},
                    }
                ],
            }
        )
        + "\n"
    )
    calls = []

    def fake_materialize(**kwargs):
        calls.append(kwargs)
        return [2, 4, 4, 4]

    monkeypatch.setattr(tile_quadrant_fusion, "_materialize_crop_ome_zarr", fake_materialize)

    outputs = export_tile_quadrant_materialized_chunks(
        window_json_dir=window_dir,
        moving_position_input=moving_path,
        fixed_registration_input=fixed_path,
        output_dir=tmp_path / "out",
        channel_source_shift_px_zyx=(1.0, 2.0, 3.0),
    )

    position = json.loads(outputs["position"].read_text())
    registration = json.loads(outputs["registration"].read_text())
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["start_zyx"], [4, 2, 1])
    np.testing.assert_array_equal(calls[0]["stop_zyx"], [8, 6, 5])
    assert len(position["tiles"]) == 1
    assert position["tiles"][0]["materialized_source_start_zyx"] == [4, 2, 1]
    assert position["tiles"][0]["shape"] == [2, 4, 4, 4]
    assert position["tiles"][0]["translation_um"] == {"z": 14.0, "y": 24.0, "x": 35.0}

    fixed_stage = np.eye(4, dtype=np.float64)
    fixed_stage[:3, 3] = [100.0, 200.0, 300.0]
    fixed_scale = np.diag([2.0, 4.0, 5.0, 1.0])
    moving_stage_inv = np.eye(4, dtype=np.float64)
    moving_stage_inv[:3, 3] = [-10.0, -20.0, -30.0]
    moving_scale_inv = np.diag([1.0, 0.5, 0.2, 1.0])
    method8 = np.eye(4, dtype=np.float64)
    method8[:3, :3] = np.diag([1.0, 1.5, 1.0])
    method8[:3, 3] = [1.0, -0.25, 3.0]
    channel_shift = np.eye(4, dtype=np.float64)
    channel_shift[:3, 3] = [1.0, 4.0, 15.0]
    expected = fixed_registered_affine @ fixed_stage @ fixed_scale @ method8 @ moving_scale_inv @ moving_stage_inv @ channel_shift
    matrix = np.asarray(registration["tiles"][0]["registered_affine"]["matrix"])
    np.testing.assert_allclose(matrix, expected)
    assert registration["method8_transform_usage"].startswith("registered_affine maps")
