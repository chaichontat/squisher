import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import squisher
from squisher.video import (
    _channel_stream_indices,
    _colorize,
    _open_ome_zarr_level,
    _percentiles_for_zarr_indices,
    _read_zarr_batch,
    _read_zarr_plane,
    _zarr_chunk_preserving_index_batches,
    render_zarr_video,
)


video_module = importlib.import_module("squisher.video")


class FakeZarrArray:
    def __init__(
        self,
        data: np.ndarray,
        dimension_names: tuple[str, ...],
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        self.data = data
        self.shape = data.shape
        self.ndim = data.ndim
        self.dtype = data.dtype
        self.chunks = chunks or data.shape
        self.metadata = SimpleNamespace(dimension_names=dimension_names)
        self.attrs = {}
        self.read_keys = []

    def __getitem__(self, key):
        self.read_keys.append(key)
        return self.data[key]


def test_channel_stream_indices_support_interleaved_layout() -> None:
    np.testing.assert_array_equal(
        _channel_stream_indices(page_count=6, channels=2, channel=0, channel_layout="interleaved"),
        np.array([0, 2, 4]),
    )
    np.testing.assert_array_equal(
        _channel_stream_indices(page_count=6, channels=2, channel=1, channel_layout="interleaved"),
        np.array([1, 3, 5]),
    )


def test_channel_stream_indices_support_contiguous_layout() -> None:
    np.testing.assert_array_equal(
        _channel_stream_indices(page_count=6, channels=2, channel=0, channel_layout="contiguous"),
        np.array([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        _channel_stream_indices(page_count=6, channels=2, channel=1, channel_layout="contiguous"),
        np.array([3, 4, 5]),
    )


def test_green_magenta_overlay_makes_double_positive_white() -> None:
    green = _colorize(np.array([[255]], dtype=np.uint8), "green")
    magenta = _colorize(np.array([[255]], dtype=np.uint8), "magenta")

    np.testing.assert_array_equal(np.maximum(green, magenta), np.array([[[255, 255, 255]]], dtype=np.uint8))


def test_read_zarr_batch_supports_tczyx_channel_selection() -> None:
    data = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)
    array = FakeZarrArray(data, ("t", "c", "z", "y", "x"))

    batch = _read_zarr_batch(array, np.array([1, 2]), channel=2)

    np.testing.assert_array_equal(batch, data[0, 2, 1:3, :, :])


def test_read_zarr_batch_supports_zarr_v3_tczyx_dimension_names(tmp_path: Path) -> None:
    import zarr

    data = np.arange(1 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(1, 3, 4, 5, 6)
    root = zarr.open_group(str(tmp_path / "sample.ome.zarr"), mode="w", zarr_format=3)
    array = root.create_array(
        "0",
        data=data,
        chunks=(1, 1, 2, 5, 6),
        dimension_names=("t", "c", "z", "y", "x"),
    )

    batch = _read_zarr_batch(array, np.array([1, 2]), channel=2)

    np.testing.assert_array_equal(batch, data[0, 2, 1:3, :, :])


def test_read_zarr_batch_supports_array_dimensions_attrs() -> None:
    data = np.arange(1 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(1, 3, 4, 5, 6)
    array = FakeZarrArray(data, ())
    array.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

    batch = _read_zarr_batch(array, np.array([1, 2]), channel=2)

    np.testing.assert_array_equal(batch, data[0, 2, 1:3, :, :])


def test_read_zarr_batch_reads_each_sparse_z_chunk_once() -> None:
    data = np.arange(1 * 1 * 8 * 5 * 6, dtype=np.uint16).reshape(1, 1, 8, 5, 6)
    array = FakeZarrArray(data, ("t", "c", "z", "y", "x"), chunks=(1, 1, 4, 5, 6))

    batch = _read_zarr_batch(array, np.array([0, 2, 3, 5]), channel=0)

    np.testing.assert_array_equal(batch, data[0, 0, [0, 2, 3, 5], :, :])
    assert array.read_keys == [
        (0, 0, slice(0, 4), slice(None), slice(None)),
        (0, 0, slice(5, 6), slice(None), slice(None)),
    ]


def test_zarr_render_batches_preserve_chunks_while_meeting_minimum_batch_size() -> None:
    batches = list(
        _zarr_chunk_preserving_index_batches(
            np.arange(8, 40, dtype=int),
            z_chunk_size=16,
            min_batch_size=4,
        )
    )

    assert [batch.tolist() for batch in batches] == [
        list(range(8, 16)),
        list(range(16, 32)),
        list(range(32, 40)),
    ]


def test_zarr_render_batches_can_merge_short_whole_chunk_groups() -> None:
    batches = list(
        _zarr_chunk_preserving_index_batches(
            np.array([0, 4], dtype=int),
            z_chunk_size=4,
            min_batch_size=4,
        )
    )

    assert [batch.tolist() for batch in batches] == [[0, 4]]


def test_zarr_percentile_sampling_reads_each_sparse_z_chunk_once() -> None:
    data = np.arange(1 * 1 * 8 * 5 * 6, dtype=np.uint16).reshape(1, 1, 8, 5, 6)
    array = FakeZarrArray(data, ("t", "c", "z", "y", "x"), chunks=(1, 1, 4, 5, 6))

    low, high = _percentiles_for_zarr_indices(
        array,
        np.array([0, 2, 3, 5]),
        0.0,
        100.0,
        False,
        4,
        "primary",
        0,
    )

    assert low == 0.0
    assert high == float(data[0, 0, 5].max())
    assert array.read_keys == [
        (0, 0, slice(0, 4), slice(None), slice(None)),
        (0, 0, slice(5, 6), slice(None), slice(None)),
    ]


def test_open_ome_zarr_level_uses_multiscales_dataset_path_axes_and_scale(tmp_path: Path) -> None:
    import zarr

    root = zarr.open_group(str(tmp_path / "sample.ome.zarr"), mode="w", zarr_format=3)
    data = np.zeros((1, 2, 3, 4, 5), dtype=np.uint16)
    root.create_array("scale0", data=data, chunks=(1, 1, 2, 4, 5))
    root.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "scale0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 3.0, 2.0, 1.0]},
                    ],
                }
            ],
        }
    ]

    level = _open_ome_zarr_level(tmp_path / "sample.ome.zarr", 0)

    assert level.array.shape == data.shape
    assert level.dims == ("t", "c", "z", "y", "x")
    assert level.scale_zyx_um == (3.0, 2.0, 1.0)


def test_zarr_percentiles_use_resolved_multiscales_axes(tmp_path: Path) -> None:
    import zarr

    root = zarr.open_group(str(tmp_path / "sample.ome.zarr"), mode="w", zarr_format=3)
    data = np.arange(1 * 1 * 3 * 4 * 5, dtype=np.uint16).reshape(1, 1, 3, 4, 5)
    root.create_array("scale0", data=data, chunks=(1, 1, 2, 4, 5))
    root.attrs["multiscales"] = [
        {
            "axes": [{"name": name} for name in ("t", "c", "z", "y", "x")],
            "datasets": [{"path": "scale0"}],
        }
    ]
    level = _open_ome_zarr_level(tmp_path / "sample.ome.zarr", 0)

    low, high = _percentiles_for_zarr_indices(
        level.array,
        np.array([0, 2]),
        0.0,
        100.0,
        False,
        2,
        "primary",
        0,
        dims=level.dims,
    )

    assert low == 0.0
    assert high == float(data[0, 0, 2].max())


def test_zarr_read_helpers_reject_metadata_less_arrays() -> None:
    array = FakeZarrArray(np.zeros((1, 3, 4, 5, 6), dtype=np.uint16), ())

    with pytest.raises(ValueError, match="missing dimension names"):
        _read_zarr_batch(array, np.array([0]), channel=0)


def test_zarr_read_helpers_reject_nonzero_channel_for_zyx() -> None:
    array = FakeZarrArray(np.zeros((4, 5, 6), dtype=np.uint16), ("z", "y", "x"))

    with pytest.raises(ValueError, match="channel must be 0"):
        _read_zarr_plane(array, 0, channel=1)
    with pytest.raises(ValueError, match="channel must be 0"):
        _read_zarr_batch(array, np.array([0, 1]), channel=1)


def test_render_zarr_video_accepts_tczyx_without_reading_all_channels(tmp_path: Path, monkeypatch) -> None:
    import zarr

    root = zarr.open_group(str(tmp_path / "sample.ome.zarr"), mode="w", zarr_format=3)
    root.create_array(
        "0",
        data=np.zeros((1, 3, 4, 5, 6), dtype=np.uint16),
        chunks=(1, 1, 2, 5, 6),
        dimension_names=("t", "c", "z", "y", "x"),
    )
    root.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "t"},
                {"name": "c"},
                {"name": "z"},
                {"name": "y"},
                {"name": "x"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 3.0, 2.0, 1.0]},
                    ],
                }
            ],
        }
    ]
    output = tmp_path / "movie.mp4"
    percentile_channels = []
    rendered_channels = []
    rendered_pixel_sizes = []

    class FakeStdin:
        def write(self, frame) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeProcess:
        stdin = FakeStdin()

        def wait(self) -> int:
            return 0

    def fake_render_zarr_batch_gpu(array, indices, size, low, high, color, channel, **kwargs):
        rendered_channels.append(channel)
        rendered_pixel_sizes.append((kwargs["y_um"], kwargs["x_um"]))
        return np.zeros((len(indices), size, size), dtype=np.uint8)

    def fake_percentiles_for_zarr_indices(
        array, indices, low, high, nonzero, sample_frames, label, channel, **kwargs
    ):
        percentile_channels.append(channel)
        return 0.0, 1.0

    monkeypatch.setattr(video_module, "_open_ffmpeg", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(video_module, "_percentiles_for_zarr_indices", fake_percentiles_for_zarr_indices)
    monkeypatch.setattr(video_module, "_render_zarr_batch_gpu", fake_render_zarr_batch_gpu)

    assert render_zarr_video(tmp_path / "sample.ome.zarr", out=output, size=8, channel=2) == output
    assert percentile_channels == [2]
    assert rendered_channels == [2]
    assert rendered_pixel_sizes == [(2.0, 1.0)]


def test_video_cli_forwards_interleaved_overlay_options(tmp_path: Path, monkeypatch) -> None:
    tiff_path = tmp_path / "sample.ome.tif"
    out_path = tmp_path / "movie.mp4"
    tiff_path.write_bytes(b"not read by fake renderer")
    captured = {}

    def fake_render_tiff_video(path: Path, **kwargs) -> Path:
        captured["path"] = path
        captured.update(kwargs)
        return out_path

    monkeypatch.setattr(squisher, "render_tiff_video", fake_render_tiff_video)

    result = CliRunner().invoke(
        squisher.app,
        [
            "video",
            str(tiff_path),
            "--out",
            str(out_path),
            "--channels",
            "2",
            "--channel-layout",
            "interleaved",
            "--channel",
            "0",
            "--overlay-channel",
            "1",
            "--color",
            "green",
            "--overlay-color",
            "magenta",
            "--size",
            "960",
        ],
    )

    assert result.exit_code == 0
    assert captured["path"] == tiff_path
    assert captured["out"] == out_path
    assert captured["channels"] == 2
    assert captured["channel_layout"] == "interleaved"
    assert captured["channel"] == 0
    assert captured["overlay_channel"] == 1
    assert captured["color"] == "green"
    assert captured["overlay_color"] == "magenta"
    assert captured["size"] == 960


def test_video_cli_defaults_tiff_color_to_green(tmp_path: Path, monkeypatch) -> None:
    tiff_path = tmp_path / "sample.ome.tif"
    out_path = tmp_path / "movie.mp4"
    tiff_path.write_bytes(b"not read by fake renderer")
    captured = {}

    def fake_render_tiff_video(path: Path, **kwargs) -> Path:
        captured["path"] = path
        captured.update(kwargs)
        return out_path

    monkeypatch.setattr(squisher, "render_tiff_video", fake_render_tiff_video)

    result = CliRunner().invoke(squisher.app, ["video", str(tiff_path), "--out", str(out_path)])

    assert result.exit_code == 0
    assert captured["color"] == "green"


def test_video_cli_dispatches_zarr_inputs(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "sample.ome.zarr"
    out_path = tmp_path / "movie.mp4"
    zarr_path.mkdir()
    captured = {}

    def fake_render_zarr_video(path: Path, **kwargs) -> Path:
        captured["path"] = path
        captured.update(kwargs)
        return out_path

    monkeypatch.setattr(squisher, "render_zarr_video", fake_render_zarr_video)

    result = CliRunner().invoke(
        squisher.app,
        [
            "video",
            str(zarr_path),
            "--out",
            str(out_path),
            "--zarr-level",
            "2",
            "--channel",
            "1",
            "--size",
            "960",
            "--color",
            "gray",
        ],
    )

    assert result.exit_code == 0
    assert captured["path"] == zarr_path
    assert captured["out"] == out_path
    assert captured["zarr_level"] == 2
    assert captured["channel"] == 1
    assert captured["size"] == 960
    assert captured["color"] == "gray"


def test_video_cli_defaults_zarr_color_to_gray(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "sample.ome.zarr"
    out_path = tmp_path / "movie.mp4"
    zarr_path.mkdir()
    captured = {}

    def fake_render_zarr_video(path: Path, **kwargs) -> Path:
        captured["path"] = path
        captured.update(kwargs)
        return out_path

    monkeypatch.setattr(squisher, "render_zarr_video", fake_render_zarr_video)

    result = CliRunner().invoke(squisher.app, ["video", str(zarr_path), "--out", str(out_path)])

    assert result.exit_code == 0
    assert captured["color"] == "gray"
