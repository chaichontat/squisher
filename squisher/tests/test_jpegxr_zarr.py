from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
import zarr

from squisher.jpegxr_zarr import (
    JPEGXR_CODEC_NAME,
    JpegxrCodec,
    jpegxr_sharding_codec,
    register_jpegxr_codec,
)


def _data(shape: tuple[int, ...]) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    return (values * 7) % 60_000


def _sharded_array(path: Path, *, shape: tuple[int, ...], level: float = 1.0):
    inner_chunks = tuple(1 if axis < len(shape) - 2 else size for axis, size in enumerate(shape))
    return zarr.open(
        str(path),
        mode="w",
        shape=shape,
        chunks=shape,
        dtype="uint16",
        zarr_format=3,
        codecs=[jpegxr_sharding_codec(inner_chunks, level=level)],
    )


def test_jpegxr_shard_roundtrips_independent_2d_planes(tmp_path: Path) -> None:
    register_jpegxr_codec()
    data = _data((1, 5, 32, 40))
    array = _sharded_array(tmp_path / "stack.zarr", shape=data.shape)

    array[:] = data

    np.testing.assert_array_equal(array[:], data)
    codec = array.metadata.to_dict()["codecs"][0]
    assert codec["name"] == "sharding_indexed"
    assert codec["configuration"]["chunk_shape"] == (1, 1, 32, 40)
    assert [item["name"] for item in codec["configuration"]["codecs"]] == [
        JPEGXR_CODEC_NAME,
        "crc32c",
    ]


def test_installed_entry_point_opens_jpegxr_shard_without_manual_registration(tmp_path: Path) -> None:
    register_jpegxr_codec()
    path = tmp_path / "entrypoint.zarr"
    data = _data((3, 16, 16))
    array = _sharded_array(path, shape=data.shape)
    array[:] = data

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import zarr; print(int(zarr.open_array({str(path)!r}, mode='r')[:].sum()))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert int(result.stdout) == int(data.sum())


def test_jpegxr_shard_reads_and_updates_one_plane_independently(tmp_path: Path, monkeypatch) -> None:
    register_jpegxr_codec()
    data = _data((5, 32, 40))
    array = _sharded_array(tmp_path / "partial.zarr", shape=data.shape)
    array[:] = data
    encode_threads: list[str] = []
    decode_threads: list[str] = []
    original_encode = JpegxrCodec._encode_sync
    original_decode = JpegxrCodec._decode_sync

    def record_encode(self, chunk_array, chunk_spec):
        encode_threads.append(threading.current_thread().name)
        return original_encode(self, chunk_array, chunk_spec)

    def record_decode(self, chunk_bytes, chunk_spec):
        decode_threads.append(threading.current_thread().name)
        return original_decode(self, chunk_bytes, chunk_spec)

    monkeypatch.setattr(JpegxrCodec, "_encode_sync", record_encode)
    monkeypatch.setattr(JpegxrCodec, "_decode_sync", record_decode)

    np.testing.assert_array_equal(array[2], data[2])
    array[2] = data[2]

    assert len(decode_threads) == 1
    assert len(encode_threads) == 1
    assert all(name != "zarr_io" for name in (*encode_threads, *decode_threads))


def test_jpegxr_shard_handles_partial_final_z_extent(tmp_path: Path) -> None:
    register_jpegxr_codec()
    data = _data((5, 32, 40))
    array = zarr.open(
        str(tmp_path / "edge.zarr"),
        mode="w",
        shape=data.shape,
        chunks=(6, 32, 40),
        dtype=data.dtype,
        zarr_format=3,
        codecs=[jpegxr_sharding_codec((1, 32, 40), level=1.0)],
    )

    array[:] = data

    np.testing.assert_array_equal(array[:], data)


def test_jpegxr_shard_crc_rejects_corrupted_plane(tmp_path: Path) -> None:
    register_jpegxr_codec()
    path = tmp_path / "corrupt.zarr"
    data = _data((3, 32, 40))
    array = _sharded_array(path, shape=data.shape)
    array[:] = data
    shard = next(item for item in (path / "c").rglob("*") if item.is_file())
    payload = bytearray(shard.read_bytes())
    payload[100] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(Exception, match="checksum"):
        np.asarray(array[0])


def test_jpegxr_codec_rejects_multi_plane_inner_chunk(tmp_path: Path) -> None:
    register_jpegxr_codec()

    with pytest.raises(ValueError, match="exactly one 2D plane"):
        zarr.open(
            str(tmp_path / "invalid.zarr"),
            mode="w",
            shape=(3, 16, 16),
            chunks=(3, 16, 16),
            dtype="uint16",
            zarr_format=3,
            codecs=[JpegxrCodec()],
        )


def test_jpegxr_codec_rejects_unsupported_dtype(tmp_path: Path) -> None:
    register_jpegxr_codec()

    with pytest.raises(ValueError, match="supports uint8, uint16, float16, and float32"):
        zarr.open(
            str(tmp_path / "signed.zarr"),
            mode="w",
            shape=(1, 16, 16),
            chunks=(1, 16, 16),
            dtype=np.int16,
            zarr_format=3,
            codecs=[JpegxrCodec()],
        )


@pytest.mark.parametrize("level", [-0.01, 1.01, float("nan")])
def test_jpegxr_codec_rejects_invalid_level(level: float) -> None:
    with pytest.raises(ValueError, match="level must be finite and between 0 and 1"):
        JpegxrCodec(level=level)
