"""JPEG-XR image-plane codec and indexed 3D sharding for Zarr v3."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import Any, ClassVar

import imagecodecs
import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.codecs import Crc32cCodec, ShardingCodec


DEFAULT_JPEGXR_LEVEL = 0.7
JPEGXR_CODEC_NAME = "squisher.jpegxr"
_SUPPORTED_DTYPES = frozenset(np.dtype(dtype) for dtype in (np.uint8, np.uint16, np.float16, np.float32))
_REGISTERED = False


@dataclass(frozen=True)
class JpegxrCodec(ArrayBytesCodec):
    """Encode exactly one two-dimensional image plane as raw JPEG-XR bytes."""

    level: float = DEFAULT_JPEGXR_LEVEL
    is_fixed_size: ClassVar[bool] = False

    def __post_init__(self) -> None:
        level = float(self.level)
        if not math.isfinite(level) or not 0.0 <= level <= 1.0:
            raise ValueError(f"JPEG-XR level must be finite and between 0 and 1, got {self.level!r}")
        if not imagecodecs.JPEGXR.available:
            raise ValueError("imagecodecs was built without JPEG-XR support")
        object.__setattr__(self, "level", level)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JpegxrCodec:
        if data.get("name") != JPEGXR_CODEC_NAME:
            raise ValueError(f"Expected codec name {JPEGXR_CODEC_NAME!r}, got {data.get('name')!r}")
        configuration = data.get("configuration") or {}
        if not isinstance(configuration, dict):
            raise ValueError("JPEG-XR codec configuration must be an object")
        return cls(**configuration)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": JPEGXR_CODEC_NAME,
            "configuration": {
                "level": self.level,
            },
        }

    def validate(self, *, shape: tuple[int, ...], dtype: Any, chunk_grid: Any) -> None:
        chunk_shape = tuple(int(size) for size in chunk_grid.chunk_shape)
        _validate_plane_chunk_shape(chunk_shape)
        native_dtype = _native_dtype(dtype)
        if native_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"JPEG-XR codec supports uint8, uint16, float16, and float32; got {native_dtype}"
            )

    def _encode_sync(self, chunk_array: Any, chunk_spec: Any) -> Any:
        array = chunk_array.as_numpy_array()
        _validate_plane_chunk_shape(tuple(int(size) for size in array.shape))
        if array.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"JPEG-XR codec supports uint8, uint16, float16, and float32; got {array.dtype}")
        encoded = imagecodecs.jpegxr_encode(
            np.ascontiguousarray(array.reshape(array.shape[-2:])),
            level=self.level,
            photometric="GRAY",
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(self, chunk_array: Any, chunk_spec: Any) -> Any:
        return await asyncio.to_thread(self._encode_sync, chunk_array, chunk_spec)

    def _decode_sync(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        shape = tuple(int(size) for size in chunk_spec.shape)
        _validate_plane_chunk_shape(shape)
        output = chunk_spec.prototype.nd_buffer.empty(
            shape=shape,
            dtype=chunk_spec.dtype.to_native_dtype(),
            order=chunk_spec.order,
        )
        imagecodecs.jpegxr_decode(
            chunk_bytes.as_numpy_array(),
            out=output.as_numpy_array().reshape(shape[-2:]),
        )
        return output

    async def _decode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: Any) -> int:
        del input_byte_length, chunk_spec
        raise NotImplementedError


def jpegxr_sharding_codec(
    inner_chunk_shape: tuple[int, ...],
    *,
    level: float = DEFAULT_JPEGXR_LEVEL,
) -> ShardingCodec:
    """Pool independently encoded JPEG-XR planes in a CRC-protected indexed shard."""

    _validate_plane_chunk_shape(inner_chunk_shape)
    return ShardingCodec(
        chunk_shape=inner_chunk_shape,
        codecs=[JpegxrCodec(level=level), Crc32cCodec()],
    )


def jpegxr_plane_chunk_shape(
    chunk_shape: tuple[int, ...],
    dimension_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Keep Y/X chunking while making every other inner-chunk axis singleton."""

    if len(chunk_shape) != len(dimension_names) or tuple(name.lower() for name in dimension_names[-2:]) != (
        "y",
        "x",
    ):
        raise ValueError(
            f"JPEG-XR chunks require matching dimensions ending in y/x, got {dimension_names} and {chunk_shape}"
        )
    return tuple(
        int(size) if name.lower() in {"y", "x"} else 1
        for name, size in zip(dimension_names, chunk_shape, strict=True)
    )


def register_jpegxr_codec() -> None:
    """Register the codec directly when package entry-point discovery is unavailable."""

    global _REGISTERED
    if _REGISTERED:
        return
    from zarr.registry import register_codec

    register_codec(JPEGXR_CODEC_NAME, JpegxrCodec)
    _REGISTERED = True


def _validate_plane_chunk_shape(shape: tuple[int, ...]) -> None:
    if len(shape) < 2 or any(size != 1 for size in shape[:-2]):
        raise ValueError(f"JPEG-XR inner chunks must contain exactly one 2D plane, got {shape}")


def _native_dtype(dtype: Any) -> np.dtype[Any]:
    if hasattr(dtype, "to_native_dtype"):
        dtype = dtype.to_native_dtype()
    return np.dtype(dtype)


__all__ = [
    "DEFAULT_JPEGXR_LEVEL",
    "JPEGXR_CODEC_NAME",
    "JpegxrCodec",
    "jpegxr_plane_chunk_shape",
    "jpegxr_sharding_codec",
    "register_jpegxr_codec",
]
