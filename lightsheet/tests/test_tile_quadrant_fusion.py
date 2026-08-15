from __future__ import annotations

import json

import numpy as np
import pytest

from squisher_lightsheet import tile_quadrant_fusion
from squisher_lightsheet.tile_quadrant_fusion import (
    _fused_fixed_registered_affine_um,
    _fused_fixed_overlapping_window_zyx,
    _materialize_crop_ome_zarr,
    _materialize_downsampled_channel_crop_ome_zarr,
    _materialize_native_source_group,
    _requested_source_window_zyx,
    _shape_zyx_from_record,
    export_fused_fixed_overlapping_materialized_chunks,
    export_tile_quadrant_materialized_chunks,
)


def _source_ome_zarr(path, data: np.ndarray, *, axes: str, chunks: tuple[int, ...]):
    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    array = root.create_array(
        "0",
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        dimension_names=tuple(axis.lower() for axis in axes),
    )
    array[:] = data
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    return array


def _assert_sharded_layout(array, *, chunks: tuple[int, ...], shards: tuple[int, ...]) -> None:
    assert array.chunks == chunks
    assert array.metadata.shards == shards
    codecs = list(array.metadata.to_dict()["codecs"])
    assert len(codecs) == 1
    assert codecs[0]["name"] == "sharding_indexed"
    assert codecs[0]["configuration"]["chunk_shape"] == chunks
    inner_codecs = list(codecs[0]["configuration"]["codecs"])
    assert inner_codecs == [
        {"name": "squisher.jpegxr", "configuration": {"level": 0.7}},
        {"name": "crc32c"},
    ]


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


def test_fused_fixed_overlapping_window_maps_core_grid_to_cross_registration_grid() -> None:
    start, stop = _fused_fixed_overlapping_window_zyx(
        core_start_zyx=np.asarray([2505, 960, 480], dtype=np.int64),
        source_shape_zyx=np.asarray([2985, 1440, 960], dtype=np.int64),
        core_shape_zyx=np.asarray([480, 480, 480], dtype=np.int64),
        window_shape_zyx=np.asarray([528, 528, 528], dtype=np.int64),
    )

    np.testing.assert_array_equal(start, [2457, 912, 432])
    np.testing.assert_array_equal(stop, [2985, 1440, 960])


def test_shape_zyx_resolves_sparse_position_record_from_ome_zarr(tmp_path) -> None:
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.zeros((1, 8, 10, 12), dtype=np.uint16)
    _source_ome_zarr(source_path, source_data, axes="CZYX", chunks=(1, 4, 5, 6))

    shape = _shape_zyx_from_record({"tile": "source", "path": str(source_path)})

    np.testing.assert_array_equal(shape, [8, 10, 12])


def test_fused_fixed_registered_affine_maps_local_model_to_fixed_physical_space() -> None:
    row = {
        "fixed_start_zyx": [100, 200, 300],
        "moving_shape_zyx": [480, 480, 480],
        "fused_scale_zyx": [0.6, 0.3, 0.2],
        "fused_translation_zyx": [-5.0, 7.0, 11.0],
        "selected_local_matrix_zyx": [
            [1.0, 0.01, 0.02],
            [0.01, 1.0, 0.03],
            [0.02, 0.03, 1.0],
        ],
        "selected_local_translation_zyx": [2.0, 3.0, 4.0],
    }

    affine, fixed_origin = _fused_fixed_registered_affine_um(row)

    scale = np.asarray(row["fused_scale_zyx"])
    fixed_translation = np.asarray(row["fused_translation_zyx"])
    fixed_start = np.asarray(row["fixed_start_zyx"])
    matrix = np.asarray(row["selected_local_matrix_zyx"])
    translation = np.asarray(row["selected_local_translation_zyx"])
    center = (np.asarray(row["moving_shape_zyx"]) - 1.0) / 2.0
    np.testing.assert_allclose(fixed_origin, fixed_translation + fixed_start * scale)
    for local_pixel in (np.zeros(3), center, np.asarray([479.0, 479.0, 479.0])):
        stage_um = fixed_origin + scale * local_pixel
        actual = affine @ np.r_[stage_um, 1.0]
        fixed_local = matrix @ (local_pixel - center) + center + translation
        expected = fixed_translation + scale * (fixed_start + fixed_local)
        np.testing.assert_allclose(actual[:3], expected)


def test_export_fused_fixed_materializes_overlap_without_changing_registered_world_mapping(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "source.ome.zarr"
    source.mkdir()
    moving_position = tmp_path / "moving.positions.json"
    moving_position.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "tile.ome.tif",
                        "path": str(source),
                        "axes": "CZYX",
                        "shape": [2, 1440, 1440, 1440],
                        "scale_um": {"z": 0.6, "y": 0.3, "x": 0.2},
                    }
                ]
            }
        )
        + "\n"
    )
    window_json = tmp_path / "window.json"
    window_json.write_text(
        json.dumps(
            {
                "moving_start_l0_zyx": [960, 960, 960],
                "moving_stop_l0_zyx": [1440, 1440, 1440],
            }
        )
        + "\n"
    )
    registered_affine = np.eye(4, dtype=np.float64)
    registered_affine[:3, 3] = [7.0, 8.0, 9.0]
    registration_input = tmp_path / "core.registration.json"
    registration_input.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "tile.qy1_qx1.z00480.ome.zarr",
                        "path": "/old/core.ome.zarr",
                        "moving_tile": "tile.ome.tif",
                        "method8_window_json": str(window_json),
                        "shape": [120, 120, 120],
                        "axes": "ZYX",
                        "stage_translation_um": {"z": 1000.0, "y": 2000.0, "x": 3000.0},
                        "stage_scale_um": {"z": 2.4, "y": 1.2, "x": 0.8},
                        "registered_affine": {"matrix": registered_affine.tolist()},
                    }
                ]
            }
        )
        + "\n"
    )
    calls = []

    def fake_materialize(**kwargs):
        calls.append(kwargs)
        return [132, 132, 132]

    monkeypatch.setattr(
        tile_quadrant_fusion,
        "_materialize_downsampled_channel_crop_ome_zarr",
        fake_materialize,
    )

    outputs = export_fused_fixed_overlapping_materialized_chunks(
        source_registration_input=registration_input,
        moving_position_input=moving_position,
        output_dir=tmp_path / "out",
        output_codec="zstd",
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["start_zyx"], [912, 912, 912])
    np.testing.assert_array_equal(calls[0]["stop_zyx"], [1440, 1440, 1440])
    position = json.loads(outputs["position"].read_text())
    assert position["units"] == "micrometer"
    registration = json.loads(outputs["registration"].read_text())
    record = registration["tiles"][0]
    assert record["shape"] == [132, 132, 132]
    assert record["materialized_source_start_zyx"] == [912, 912, 912]
    assert record["stage_translation_um"] == {"z": 971.2, "y": 1985.6, "x": 2990.4}
    np.testing.assert_array_equal(record["registered_affine"]["matrix"], registered_affine)
    assert registration["materialization_grid"]["window_shape_zyx"] == [528, 528, 528]
    summary = json.loads(outputs["summary"].read_text())
    assert summary["output_codec"] == "zstd"


def test_export_fused_fixed_uses_explicit_pixel_source_registration(monkeypatch, tmp_path) -> None:
    stale_tiff = tmp_path / "tile.ome.tif"
    source_zarr = tmp_path / "deconv" / "tile.ome.zarr"
    moving_position = tmp_path / "moving.positions.json"
    moving_position.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "tile.ome.tif",
                        "path": str(stale_tiff),
                        "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                        "scale_um": {"z": 0.6, "y": 0.3, "x": 0.2},
                    }
                ]
            }
        )
        + "\n"
    )
    moving_sources = tmp_path / "moving.registration.json"
    moving_sources.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "tile.ome.zarr",
                        "path": str(source_zarr),
                        "shape": [3, 1440, 1440, 1440],
                        "axes": "CZYX",
                    }
                ]
            }
        )
        + "\n"
    )
    window_json = tmp_path / "window.json"
    window_json.write_text(
        json.dumps(
            {
                "moving_start_l0_zyx": [960, 960, 960],
                "moving_stop_l0_zyx": [1440, 1440, 1440],
            }
        )
        + "\n"
    )
    registration_input = tmp_path / "core.registration.json"
    registration_input.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "tile.qy1_qx1.z00960.ome.zarr",
                        "moving_tile": "tile.ome.tif",
                        "method8_window_json": str(window_json),
                        "stage_translation_um": {"z": 100.0, "y": 200.0, "x": 300.0},
                        "stage_scale_um": {"z": 2.4, "y": 1.2, "x": 0.8},
                    }
                ]
            }
        )
        + "\n"
    )
    calls = []

    def fake_materialize(**kwargs):
        calls.append(kwargs)
        return [132, 132, 132]

    monkeypatch.setattr(
        tile_quadrant_fusion,
        "_materialize_downsampled_channel_crop_ome_zarr",
        fake_materialize,
    )

    outputs = export_fused_fixed_overlapping_materialized_chunks(
        source_registration_input=registration_input,
        moving_position_input=moving_position,
        moving_source_input=moving_sources,
        output_dir=tmp_path / "out",
        output_codec="zstd",
    )

    assert calls[0]["source_path"] == source_zarr
    summary = json.loads(outputs["summary"].read_text())
    assert summary["moving_source_input"] == str(moving_sources.resolve())


def test_export_fused_fixed_allows_overlap_with_full_extent_axis(tmp_path) -> None:
    moving_position = tmp_path / "moving.positions.json"
    moving_position.write_text(json.dumps({"tiles": []}) + "\n")
    registration_input = tmp_path / "core.registration.json"
    registration_input.write_text(json.dumps({"tiles": []}) + "\n")

    outputs = export_fused_fixed_overlapping_materialized_chunks(
        source_registration_input=registration_input,
        moving_position_input=moving_position,
        output_dir=tmp_path / "out",
        core_shape_zyx=(320, 640, 640),
        window_shape_zyx=(352, 704, 640),
        output_codec="zstd",
    )

    registration = json.loads(outputs["registration"].read_text())
    assert registration["materialization_grid"]["core_shape_zyx"] == [320, 640, 640]
    assert registration["materialization_grid"]["window_shape_zyx"] == [352, 704, 640]

    with pytest.raises(ValueError, match="must be larger on at least one axis"):
        export_fused_fixed_overlapping_materialized_chunks(
            source_registration_input=registration_input,
            moving_position_input=moving_position,
            output_dir=tmp_path / "no-overlap",
            core_shape_zyx=(320, 640, 640),
            window_shape_zyx=(320, 640, 640),
            output_codec="zstd",
        )


def test_export_fused_fixed_builds_registration_from_selected_window_summary(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source.ome.zarr"
    source.mkdir()
    moving_position = tmp_path / "moving.positions.json"
    moving_position.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "tile.ome.tif",
                        "path": str(source),
                        "axes": "CZYX",
                        "shape": [2, 1440, 1440, 1440],
                        "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                        "scale_um": {"z": 0.6, "y": 0.3, "x": 0.2},
                    }
                ]
            }
        )
        + "\n"
    )
    accepted = {
        "status": "accepted",
        "moving_tile": "tile.ome.tif",
        "quadrant": "qy2_qx1",
        "moving_start_l0_zyx": [912, 912, 912],
        "moving_stop_l0_zyx": [1440, 1440, 1440],
        "moving_shape_zyx": [528, 528, 528],
        "fixed_start_zyx": [100, 200, 300],
        "fused_scale_zyx": [0.6, 0.3, 0.2],
        "fused_translation_zyx": [-5.0, 7.0, 11.0],
        "selected_attempt": "measured_inlier",
        "selected_local_matrix_zyx": np.eye(3).tolist(),
        "selected_local_translation_zyx": [2.0, 3.0, 4.0],
    }
    accepted_path = tmp_path / "accepted.json"
    rejected_path = tmp_path / "rejected.json"
    accepted_path.write_text(json.dumps(accepted) + "\n")
    rejected_path.write_text(json.dumps(accepted | {"status": "rejected"}) + "\n")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "fixed_fused": "/fixed.ome.zarr",
                "windows": [
                    {"level0_json": str(accepted_path)},
                    {"level0_json": str(rejected_path)},
                ],
            }
        )
        + "\n"
    )
    calls = []

    def fake_materialize(**kwargs):
        calls.append(kwargs)
        return [132, 132, 132]

    monkeypatch.setattr(
        tile_quadrant_fusion,
        "_materialize_downsampled_channel_crop_ome_zarr",
        fake_materialize,
    )

    outputs = export_fused_fixed_overlapping_materialized_chunks(
        source_summary_input=summary_path,
        moving_position_input=moving_position,
        output_dir=tmp_path / "out",
        output_codec="zstd",
    )

    assert len(calls) == 1
    registration = json.loads(outputs["registration"].read_text())
    assert registration["source_summary_input"] == str(summary_path.resolve())
    assert len(registration["tiles"]) == 1
    record = registration["tiles"][0]
    assert record["transform_source"] == "measured_inlier"
    np.testing.assert_allclose(
        [record["stage_translation_um"][axis] for axis in "zyx"],
        [55.0, 67.0, 71.0],
    )
    assert record["materialized_source_start_zyx"] == [912, 912, 912]


def test_materialize_crop_writes_image10_local_crop_without_padding(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.arange(1 * 4 * 5 * 6, dtype=np.uint16).reshape(1, 4, 5, 6)
    source = _source_ome_zarr(source_path, source_data, axes="CZYX", chunks=(1, 2, 5, 6))

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
    np.testing.assert_allclose(actual, expected, atol=4)
    _assert_sharded_layout(output, chunks=(1, 1, 3, 3), shards=(1, 2, 3, 3))


def test_materialize_downsampled_channel_crop_writes_zyx_zstd(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.arange(2 * 8 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8, 8)
    _source_ome_zarr(source_path, source_data, axes="CZYX", chunks=(1, 4, 4, 4))

    output_path = tmp_path / "out.ome.zarr"
    shape = _materialize_downsampled_channel_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr"},
        source_channel=1,
        start_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        stop_zyx=np.asarray([8, 8, 8], dtype=np.int64),
        level_factor_zyx=np.asarray([2, 2, 2], dtype=np.int64),
        spacing_zyx=np.asarray([1.2, 0.6, 0.6], dtype=np.float64),
        output_codec="zstd",
        zstd_level=3,
    )

    output = zarr.open_group(str(output_path), mode="r")["0"]
    expected = source_data[1].reshape(4, 2, 4, 2, 4, 2).mean(axis=(1, 3, 5))
    assert shape == [4, 4, 4]
    assert output.metadata.dimension_names == ("z", "y", "x")
    np.testing.assert_array_equal(output[:], expected.astype(np.uint16))
    assert [codec["name"] for codec in output.metadata.to_dict()["codecs"]] == ["bytes", "zstd"]


def test_materialize_downsampled_channel_crop_writes_explicit_zstd(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.arange(2 * 8 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8, 8)
    _source_ome_zarr(source_path, source_data, axes="CZYX", chunks=(1, 4, 4, 4))

    output_path = tmp_path / "out.ome.zarr"
    shape = _materialize_downsampled_channel_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr"},
        source_channel=1,
        start_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        stop_zyx=np.asarray([8, 8, 8], dtype=np.int64),
        level_factor_zyx=np.asarray([2, 2, 2], dtype=np.int64),
        spacing_zyx=np.asarray([1.2, 0.6, 0.6], dtype=np.float64),
        output_codec="zstd",
        zstd_level=3,
    )

    output = zarr.open_group(str(output_path), mode="r")["0"]
    expected = source_data[1].reshape(4, 2, 4, 2, 4, 2).mean(axis=(1, 3, 5))
    assert shape == [4, 4, 4]
    np.testing.assert_array_equal(output[:], expected.astype(np.uint16))
    assert [codec["name"] for codec in output.metadata.to_dict()["codecs"]] == ["bytes", "zstd"]


def test_materialize_native_channel_crop_writes_explicit_jpegxr(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.arange(2 * 4 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4, 4)
    _source_ome_zarr(source_path, source_data, axes="CZYX", chunks=(1, 2, 4, 4))

    output_path = tmp_path / "out.ome.zarr"
    _materialize_downsampled_channel_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr"},
        source_channel=1,
        start_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        stop_zyx=np.asarray([4, 4, 4], dtype=np.int64),
        level_factor_zyx=np.asarray([1, 1, 1], dtype=np.int64),
        spacing_zyx=np.asarray([0.6, 0.3, 0.3], dtype=np.float64),
        output_codec="jpegxr",
    )

    output = zarr.open_group(str(output_path), mode="r")["0"]
    np.testing.assert_allclose(output[:], source_data[1], atol=16)
    _assert_sharded_layout(output, chunks=(1, 4, 4), shards=(4, 4, 4))


def test_native_source_group_reuses_full_xy_slabs_and_writes_exact_zstd(
    monkeypatch, tmp_path
) -> None:
    zarr = pytest.importorskip("zarr")
    source_data = np.arange(1 * 4 * 6 * 6, dtype=np.uint16).reshape(1, 4, 6, 6)

    class RecordingArray:
        shape = source_data.shape
        chunks = (1, 2, 6, 6)
        dtype = source_data.dtype

        def __init__(self) -> None:
            self.selections = []

        def __getitem__(self, selection):
            self.selections.append(selection)
            return source_data[selection]

    source = RecordingArray()
    monkeypatch.setattr(
        tile_quadrant_fusion,
        "_ome_downsample_source",
        lambda _path, _factors: (source, "CZYX", 0, np.ones(3, dtype=np.int64)),
    )
    base_task = {
        "source_path": tmp_path / "source.ome.zarr",
        "source_record": {"tile": "source.ome.zarr", "axes": "CZYX"},
        "source_channel": 0,
        "level_factor_zyx": np.ones(3, dtype=np.int64),
        "spacing_zyx": np.ones(3, dtype=np.float64),
        "output_codec": "zstd",
        "zstd_level": 3,
        "jpegxr_level": 0.7,
    }
    tasks = [
        base_task
        | {
            "output_path": tmp_path / "left.ome.zarr",
            "start_zyx": np.asarray([0, 0, 0]),
            "stop_zyx": np.asarray([4, 4, 4]),
        },
        base_task
        | {
            "output_path": tmp_path / "right.ome.zarr",
            "start_zyx": np.asarray([0, 2, 2]),
            "stop_zyx": np.asarray([4, 6, 6]),
        },
    ]

    shapes = _materialize_native_source_group(tasks)

    assert shapes == [[4, 4, 4], [4, 4, 4]]
    assert len(source.selections) == 2
    assert all(selection[2:] == (slice(None), slice(None)) for selection in source.selections)
    np.testing.assert_array_equal(
        zarr.open_group(str(tmp_path / "left.ome.zarr"), mode="r")["0"][:],
        source_data[0, :, :4, :4],
    )
    right_root = zarr.open_group(str(tmp_path / "right.ome.zarr"), mode="r")
    np.testing.assert_array_equal(right_root["0"][:], source_data[0, :, 2:, 2:])
    assert right_root.attrs["squisher_complete"] is True
    assert [codec["name"] for codec in right_root["0"].metadata.to_dict()["codecs"]] == [
        "bytes",
        "zstd",
    ]
    completion_path = tmp_path / "right.ome.zarr" / "squisher.complete.json"
    completion = json.loads(completion_path.read_text())
    assert completion["payload"]
    first_payload = tmp_path / "right.ome.zarr" / completion["payload"][0]["path"]
    first_payload.unlink()
    assert tile_quadrant_fusion._completed_materialization_shape(tasks[1]) is None


def test_native_source_group_rejects_crop_outside_source(monkeypatch, tmp_path) -> None:
    source_data = np.zeros((1, 4, 6, 6), dtype=np.uint16)

    class Source:
        shape = source_data.shape
        chunks = (1, 2, 6, 6)
        dtype = source_data.dtype

        def __getitem__(self, selection):
            return source_data[selection]

    monkeypatch.setattr(
        tile_quadrant_fusion,
        "_ome_downsample_source",
        lambda _path, _factors: (Source(), "CZYX", 0, np.ones(3, dtype=np.int64)),
    )
    task = {
        "source_path": tmp_path / "source.ome.zarr",
        "output_path": tmp_path / "out.ome.zarr",
        "source_record": {"tile": "source.ome.zarr", "axes": "CZYX"},
        "source_channel": 0,
        "start_zyx": np.asarray([0, 0, 0]),
        "stop_zyx": np.asarray([5, 4, 4]),
        "level_factor_zyx": np.ones(3, dtype=np.int64),
        "spacing_zyx": np.ones(3, dtype=np.float64),
        "output_codec": "zstd",
        "zstd_level": 3,
        "jpegxr_level": 0.7,
    }

    with pytest.raises(ValueError, match="outside source shape"):
        _materialize_native_source_group([task])
    assert not (tmp_path / "out.ome.zarr").exists()


def test_fused_fixed_resume_rejects_changed_plan(tmp_path) -> None:
    moving_position = tmp_path / "moving.positions.json"
    registration_input = tmp_path / "core.registration.json"
    moving_position.write_text(json.dumps({"tiles": []}) + "\n")
    registration_input.write_text(json.dumps({"tiles": []}) + "\n")
    output_dir = tmp_path / "out"
    export_fused_fixed_overlapping_materialized_chunks(
        source_registration_input=registration_input,
        moving_position_input=moving_position,
        output_dir=output_dir,
        output_codec="zstd",
    )

    with pytest.raises(ValueError, match="materialization plan differs"):
        export_fused_fixed_overlapping_materialized_chunks(
            source_registration_input=registration_input,
            moving_position_input=moving_position,
            output_dir=output_dir,
            output_codec="jpegxr",
            resume=True,
        )


def test_materialize_downsampled_crop_reuses_matching_xy_pyramid(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    level0 = np.zeros((2, 8, 8, 8), dtype=np.uint16)
    _source_ome_zarr(source_path, level0, axes="CZYX", chunks=(1, 4, 4, 4))
    root = zarr.open_group(str(source_path), mode="a")
    level2_data = np.arange(2 * 8 * 2 * 2, dtype=np.uint16).reshape(2, 8, 2, 2)
    level2 = root.create_array(
        "2",
        shape=level2_data.shape,
        chunks=(1, 4, 2, 2),
        dtype=level2_data.dtype,
        dimension_names=("c", "z", "y", "x"),
    )
    level2[:] = level2_data
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}, {"path": "2"}]}]

    output_path = tmp_path / "out.ome.zarr"
    shape = _materialize_downsampled_channel_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr"},
        source_channel=1,
        start_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        stop_zyx=np.asarray([8, 8, 8], dtype=np.int64),
        level_factor_zyx=np.asarray([2, 4, 4], dtype=np.int64),
        spacing_zyx=np.asarray([1.2, 1.2, 1.2], dtype=np.float64),
        output_codec="zstd",
        zstd_level=3,
    )

    output_root = zarr.open_group(str(output_path), mode="r")
    expected = level2_data[1].reshape(4, 2, 2, 2).mean(axis=1).astype(np.uint16)
    assert shape == [4, 2, 2]
    np.testing.assert_array_equal(output_root["0"][:], expected)
    assert output_root.attrs["squisher_materialization"]["source_level"] == 1
    assert output_root.attrs["squisher_materialization"]["source_factor_zyx"] == [1, 4, 4]


def test_materialize_czyx_crop_uses_fusion_inner_chunks_and_shards(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.arange(2 * 60 * 8 * 9, dtype=np.uint16).reshape(2, 60, 8, 9)
    source = _source_ome_zarr(source_path, source_data, axes="CZYX", chunks=(1, 1, 2, 3))

    output_path = tmp_path / "out.ome.zarr"
    _materialize_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr", "axes": "CZYX"},
        start_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        stop_zyx=np.asarray([60, 8, 9], dtype=np.int64),
        spacing_zyx=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
    )

    output = zarr.open_group(str(output_path), mode="r")["0"]
    _assert_sharded_layout(output, chunks=(1, 1, 8, 9), shards=(1, 48, 8, 9))
    np.testing.assert_allclose(output[:], source[:], atol=9)


def test_materialize_zyx_crop_uses_fusion_inner_chunks_and_shards(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.arange(60 * 8 * 9, dtype=np.uint16).reshape(60, 8, 9)
    source = _source_ome_zarr(source_path, source_data, axes="ZYX", chunks=(1, 2, 3))

    output_path = tmp_path / "out.ome.zarr"
    _materialize_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr", "axes": "ZYX"},
        start_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        stop_zyx=np.asarray([60, 8, 9], dtype=np.int64),
        spacing_zyx=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
    )

    output = zarr.open_group(str(output_path), mode="r")["0"]
    _assert_sharded_layout(output, chunks=(1, 8, 9), shards=(48, 8, 9))
    np.testing.assert_allclose(output[:], source[:], atol=9)


def test_materialize_small_crop_caps_inner_and_shard_z(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.arange(1 * 7 * 4 * 5, dtype=np.uint16).reshape(1, 7, 4, 5)
    source = _source_ome_zarr(source_path, source_data, axes="CZYX", chunks=(1, 1, 2, 2))

    output_path = tmp_path / "out.ome.zarr"
    _materialize_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr", "axes": "CZYX"},
        start_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        stop_zyx=np.asarray([7, 4, 5], dtype=np.int64),
        spacing_zyx=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
    )

    output = zarr.open_group(str(output_path), mode="r")["0"]
    _assert_sharded_layout(output, chunks=(1, 1, 4, 5), shards=(1, 7, 4, 5))
    np.testing.assert_allclose(output[:], source[:], atol=6)


def test_materialize_nonstandard_crop_z_keeps_shard_divisible_by_inner_z(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    source_data = np.arange(20 * 4 * 5, dtype=np.uint16).reshape(20, 4, 5)
    source = _source_ome_zarr(source_path, source_data, axes="ZYX", chunks=(1, 2, 2))

    output_path = tmp_path / "out.ome.zarr"
    _materialize_crop_ome_zarr(
        source_path=source_path,
        output_path=output_path,
        source_record={"tile": "source.ome.zarr", "axes": "ZYX"},
        start_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        stop_zyx=np.asarray([20, 4, 5], dtype=np.int64),
        spacing_zyx=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
    )

    output = zarr.open_group(str(output_path), mode="r")["0"]
    _assert_sharded_layout(output, chunks=(1, 4, 5), shards=(20, 4, 5))
    np.testing.assert_allclose(output[:], source[:], atol=6)


@pytest.mark.parametrize(
    ("moving_scale_zyx", "expected_chunk_origin_zyx", "expected_channel_shift_um_zyx"),
    [
        ((1.0, 2.0, 5.0), (14.0, 24.0, 35.0), (1.0, 4.0, 15.0)),
        ((-1.0, 2.0, -5.0), (6.0, 24.0, 25.0), (-1.0, 4.0, -15.0)),
    ],
)
def test_export_composes_method8_transform_into_registered_affine(
    monkeypatch,
    tmp_path,
    moving_scale_zyx,
    expected_chunk_origin_zyx,
    expected_channel_shift_um_zyx,
) -> None:
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
    rejected = accepted | {
        "status": "rejected",
        "rejection_reason": "fixed_threshold_mask_too_masked",
        "quadrant": "qy0_qx0",
    }
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
                        "scale_um": dict(zip(("z", "y", "x"), moving_scale_zyx, strict=True)),
                        "shape": [2, 10, 10, 10],
                        "axes": "CZYX",
                        "channels": ["0", "1"],
                        "tracks": [
                            {
                                "slug": "track0",
                                "track_id": "all",
                                "channels": [0, 1],
                                "channel_names": ["0", "1"],
                            }
                        ],
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
    summary = json.loads(outputs["summary"].read_text())
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["start_zyx"], [4, 2, 1])
    np.testing.assert_array_equal(calls[0]["stop_zyx"], [8, 6, 5])
    assert len(position["tiles"]) == 1
    assert position["tiles"][0]["materialized_source_start_zyx"] == [4, 2, 1]
    assert position["tiles"][0]["shape"] == [2, 4, 4, 4]
    assert position["tiles"][0]["translation_um"] == dict(
        zip(("z", "y", "x"), expected_chunk_origin_zyx, strict=True)
    )

    fixed_stage = np.eye(4, dtype=np.float64)
    fixed_stage[:3, 3] = [100.0, 200.0, 300.0]
    fixed_scale = np.diag([2.0, 4.0, 5.0, 1.0])
    moving_stage_inv = np.eye(4, dtype=np.float64)
    moving_stage_inv[:3, 3] = [-10.0, -20.0, -30.0]
    moving_scale_inv = np.diag([*(1.0 / np.asarray(moving_scale_zyx)), 1.0])
    method8 = np.eye(4, dtype=np.float64)
    method8[:3, :3] = np.diag([1.0, 1.5, 1.0])
    method8[:3, 3] = [1.0, -0.25, 3.0]
    channel_shift = np.eye(4, dtype=np.float64)
    channel_shift[:3, 3] = expected_channel_shift_um_zyx
    expected = (
        fixed_registered_affine
        @ fixed_stage
        @ fixed_scale
        @ method8
        @ moving_scale_inv
        @ moving_stage_inv
        @ channel_shift
    )
    matrix = np.asarray(registration["tiles"][0]["registered_affine"]["matrix"])
    np.testing.assert_allclose(matrix, expected)
    assert registration["method8_transform_usage"].startswith("registered_affine maps")
    assert summary["diagnostics"][0]["materialized_inner_chunks"] == [1, 1, 4, 4]
    assert summary["diagnostics"][0]["materialized_shard_chunks"] == [1, 4, 4, 4]
