from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import pytest

from squisher_deconv.metadata import SourceMetadata, json_dumps_strict
from squisher_deconv.sink import write_streamed_ome_tiff
from squisher_deconv.source import TiffLogicalSource


def test_json_dumps_strict_rejects_non_serializable_values() -> None:
    with pytest.raises(TypeError, match="Test payload must be JSON serializable"):
        json_dumps_strict({"bad": object()}, context="Test payload")


def test_ome_provenance_serialization_failure_removes_partial_output(tmp_path) -> None:
    source = TiffLogicalSource(
        path=tmp_path / "source.tif",
        channels=1,
        z_count=1,
        height=2,
        width=2,
        dtype="uint16",
        metadata=SourceMetadata(
            shaped_metadata=[],
            imagej_metadata=None,
            ome_xml=None,
            tags={},
            raw_shape=(1, 2, 2),
            raw_dtype="uint16",
            metadata_hash="hash",
        ),
    )
    output = tmp_path / "out.tif"

    with pytest.raises(TypeError, match="OME deconvolution provenance must be JSON serializable"):
        write_streamed_ome_tiff(
            output,
            source=source,
            core_plane_chunks=[np.zeros((1, 2, 2), dtype=np.uint16)],
            output_mode="u16",
            provenance={"bad": object()},
        )

    assert not output.exists()
    assert not (tmp_path / ".out.tif.partial").exists()


def test_streamed_ome_tiff_writes_from_dedicated_thread(tmp_path, monkeypatch) -> None:
    source = TiffLogicalSource(
        path=tmp_path / "source.tif",
        channels=1,
        z_count=2,
        height=2,
        width=2,
        dtype="uint16",
        metadata=SourceMetadata(
            shaped_metadata=[],
            imagej_metadata=None,
            ome_xml=None,
            tags={},
            raw_shape=(2, 2, 2),
            raw_dtype="uint16",
            metadata_hash="hash",
        ),
    )
    first_write_started = threading.Event()
    second_chunk_queued = threading.Event()
    writer_thread_names: list[str] = []

    class BlockingWriter:
        def __init__(self, path, **kwargs) -> None:
            self.path = Path(path)

        def __enter__(self):
            self.path.write_bytes(b"partial")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def write(self, plane, **kwargs) -> None:
            writer_thread_names.append(threading.current_thread().name)
            if not first_write_started.is_set():
                first_write_started.set()
                assert second_chunk_queued.wait(timeout=1)

    monkeypatch.setattr("squisher_deconv.sink.TiffWriter", BlockingWriter)

    def chunks():
        yield np.ones((1, 2, 2), dtype=np.uint16)
        assert first_write_started.wait(timeout=1)
        yield np.ones((1, 2, 2), dtype=np.uint16)
        second_chunk_queued.set()

    write_streamed_ome_tiff(
        tmp_path / "out.tif",
        source=source,
        core_plane_chunks=chunks(),
        output_mode="float32",
        provenance={},
    )

    assert (tmp_path / "out.tif").exists()
    assert writer_thread_names
    assert set(writer_thread_names) == {"squisher-ome-writer"}
