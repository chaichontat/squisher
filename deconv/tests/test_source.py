from __future__ import annotations

import numpy as np
import pytest
import tifffile

from squisher_deconv.source import TiffLogicalSource


def test_read_window_reads_only_requested_flattened_pages(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tile.tif"
    payload = np.arange(4 * 2 * 3 * 4, dtype=np.uint16).reshape(8, 3, 4)
    tifffile.imwrite(path, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    def fail_imread(*args, **kwargs):
        raise AssertionError("read_window must not use full-stack tifffile.imread")

    monkeypatch.setattr(tifffile, "imread", fail_imread)

    source = TiffLogicalSource.open(path, channels=2)
    window = source.read_window(1, 3)

    assert np.array_equal(window, payload[2:6].reshape(2, 2, 3, 4))


def test_read_window_honors_explicit_czyx_page_order(tmp_path) -> None:
    path = tmp_path / "tile-czyx.ome.tif"
    payload = np.arange(2 * 3 * 2 * 4, dtype=np.uint16).reshape(2, 3, 2, 4)
    tifffile.imwrite(path, payload, metadata={"axes": "CZYX"}, photometric="minisblack")

    source = TiffLogicalSource.open(path, channels=2)
    window = source.read_window(1, 3)

    expected = np.stack(
        [
            np.stack([payload[0, 1], payload[1, 1]]),
            np.stack([payload[0, 2], payload[1, 2]]),
        ]
    )
    assert source.z_count == 3
    assert np.array_equal(window, expected)


def test_open_rejects_explicit_channel_axis_mismatch(tmp_path) -> None:
    path = tmp_path / "tile-czyx.ome.tif"
    payload = np.arange(2 * 3 * 2 * 4, dtype=np.uint16).reshape(2, 3, 2, 4)
    tifffile.imwrite(path, payload, metadata={"axes": "CZYX"}, photometric="minisblack")

    with pytest.raises(ValueError, match="declares 2 channel\\(s\\).*channels=1"):
        TiffLogicalSource.open(path, channels=1)


def test_read_window_rejects_non_page_addressable_stack(tmp_path) -> None:
    path = tmp_path / "tile.tif"
    payload = np.arange(4 * 2 * 3 * 4, dtype=np.uint16).reshape(8, 3, 4)
    tifffile.imwrite(path, payload, metadata={"axes": "ZYX"}, contiguous=True)

    with pytest.raises(ValueError, match="requires one page per flattened"):
        TiffLogicalSource.open(path, channels=2)


def test_read_window_rejects_extra_tiff_pages(tmp_path) -> None:
    path = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(path, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint16), append=True, photometric="minisblack")

    with pytest.raises(ValueError, match="exposes 5 TIFF page\\(s\\).*declares 4 flattened plane"):
        TiffLogicalSource.open(path, channels=2)
