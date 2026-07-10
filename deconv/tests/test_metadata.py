from __future__ import annotations

import numpy as np
import pytest
import zarr

from squisher_deconv.metadata import SourceMetadata, json_dumps_strict
from squisher_deconv.sink import write_streamed_ome_zarr
from squisher_deconv.source import TiffLogicalSource


def test_json_dumps_strict_rejects_non_serializable_values() -> None:
    with pytest.raises(TypeError, match="Test payload must be JSON serializable"):
        json_dumps_strict({"bad": object()}, context="Test payload")


def test_ome_zarr_provenance_serialization_failure_removes_partial_output(tmp_path) -> None:
    source = _source(tmp_path, channels=1, z_count=1, height=2, width=2)
    output = tmp_path / "out.ome.zarr"

    with pytest.raises(TypeError, match="OME-Zarr deconvolution provenance must be JSON serializable"):
        write_streamed_ome_zarr(
            output,
            source=source,
            core_plane_chunks=[np.zeros((1, 2, 2), dtype=np.uint16)],
            output_mode="u16",
            provenance={"bad": object()},
        )

    assert not output.exists()
    assert not (tmp_path / ".out.ome.zarr.partial").exists()


def test_streamed_ome_zarr_writes_czyx_array_with_metadata(tmp_path) -> None:
    source = _source(tmp_path, channels=2, z_count=2, height=5, width=6)
    output = tmp_path / "out.ome.zarr"
    payload = np.arange(2 * 2 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)

    write_streamed_ome_zarr(
        output,
        source=source,
        core_plane_chunks=[payload],
        output_mode="u16",
        provenance={"tool": "test"},
    )

    root = zarr.open_group(str(output), mode="r")
    array = root["0"]
    assert tuple(array.shape) == (2, 2, 5, 6)
    assert tuple(array.chunks) == (1, 2, 5, 6)
    assert np.dtype(array.dtype) == np.uint16
    assert array.attrs["_ARRAY_DIMENSIONS"] == ["c", "z", "y", "x"]
    assert root.attrs["squisher_complete"] is True
    assert root.attrs["multiscales"][0]["datasets"][0]["path"] == "0"
    assert root.attrs["squisher_deconv"]["provenance"] == {"tool": "test"}
    assert root.attrs["squisher_deconv"]["source_metadata_summary"]["raw_shape"] == [4, 5, 6]
    np.testing.assert_array_equal(np.moveaxis(array[:], 0, 1).reshape(4, 5, 6), payload)


def _source(tmp_path, *, channels: int, z_count: int, height: int, width: int) -> TiffLogicalSource:
    return TiffLogicalSource(
        path=tmp_path / "source.tif",
        channels=channels,
        z_count=z_count,
        height=height,
        width=width,
        dtype="uint16",
        metadata=SourceMetadata(
            shaped_metadata=[],
            imagej_metadata=None,
            ome_xml=None,
            tags={},
            raw_shape=(z_count * channels, height, width),
            raw_dtype="uint16",
            metadata_hash="hash",
        ),
    )
