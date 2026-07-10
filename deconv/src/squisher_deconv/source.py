from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import tifffile

from squisher_deconv.metadata import SourceMetadata, json_dumps_strict, read_source_metadata


@dataclass(frozen=True, slots=True)
class TiffLogicalSource:
    path: Path
    channels: int
    z_count: int
    height: int
    width: int
    dtype: str
    metadata: SourceMetadata
    axes: str = "ZYX"
    _leading_shape: tuple[int, ...] = (1,)
    _z_axis: int | None = None
    _c_axis: int | None = None

    @classmethod
    def open(cls, path: Path, *, channels: int, metadata_mode: str = "full") -> "TiffLogicalSource":
        if metadata_mode not in {"full", "summary"}:
            raise ValueError(f"Unsupported metadata_mode={metadata_mode!r}; expected 'full' or 'summary'.")
        metadata = read_source_metadata(path) if metadata_mode == "full" else _read_summary_metadata(path)
        shape = metadata.raw_shape
        if len(shape) < 3:
            raise ValueError(f"{path} must have at least flattened planes plus Y/X, got shape {shape}")
        height, width = int(shape[-2]), int(shape[-1])
        plane_count = int(np.prod(shape[:-2]))
        if plane_count % channels:
            raise ValueError(f"{path} has {plane_count} plane(s), not divisible by channels={channels}")
        with tifffile.TiffFile(path) as tif:
            page_count = len(tif.pages)
            axes = tif.series[0].axes
        if page_count != plane_count:
            raise ValueError(
                f"{path} exposes {page_count} TIFF page(s), but its shaped data declares {plane_count} "
                "flattened plane(s). squisher-deconv requires one page per flattened z/channel plane "
                "so slabs can be streamed without reading the full stack."
            )
        leading_shape = tuple(int(v) for v in shape[:-2])
        z_count, z_axis, c_axis = _logical_axes(path, axes=axes, leading_shape=leading_shape, channels=channels)
        return cls(
            path=Path(path),
            channels=int(channels),
            z_count=z_count,
            height=height,
            width=width,
            dtype=metadata.raw_dtype,
            metadata=metadata,
            axes=axes,
            _leading_shape=leading_shape,
            _z_axis=z_axis,
            _c_axis=c_axis,
        )

    def read_window(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop > self.z_count or start >= stop:
            raise ValueError(f"Invalid z window [{start}, {stop}) for z_count={self.z_count}")
        page_keys = self._page_keys(start, stop)
        with tifffile.TiffFile(self.path) as tif:
            if len(tif.pages) <= max(page_keys):
                raise ValueError(
                    f"{self.path} exposes {len(tif.pages)} TIFF page(s), but window [{start}, {stop}) "
                    f"requires flattened page {max(page_keys)}."
                )
            if _is_contiguous(page_keys):
                window = tif.asarray(key=slice(page_keys[0], page_keys[-1] + 1))
            else:
                window = tif.asarray(key=page_keys)
        return window.reshape(stop - start, self.channels, self.height, self.width)

    def _page_keys(self, start: int, stop: int) -> list[int]:
        if self._c_axis is None:
            flat_start = start * self.channels
            flat_stop = stop * self.channels
            return list(range(flat_start, flat_stop))
        if self._z_axis is None:
            raise RuntimeError("Explicit channel-axis sources must also have a z axis.")
        keys: list[int] = []
        for z_index in range(start, stop):
            for channel in range(self.channels):
                index = [0] * len(self._leading_shape)
                index[self._z_axis] = z_index
                index[self._c_axis] = channel
                keys.append(int(np.ravel_multi_index(tuple(index), self._leading_shape)))
        return keys


def _logical_axes(
    path: Path,
    *,
    axes: str,
    leading_shape: tuple[int, ...],
    channels: int,
) -> tuple[int, int | None, int | None]:
    if len(axes) != len(leading_shape) + 2 or not axes.endswith("YX"):
        return int(np.prod(leading_shape) // channels), None, None
    leading_axes = axes[:-2]
    if "C" not in leading_axes:
        return int(np.prod(leading_shape) // channels), None, None
    if set(leading_axes) != {"Z", "C"} or len(leading_axes) != 2:
        raise ValueError(
            f"{path} has unsupported axes {axes!r}; explicit channel-axis sources must be ZCYX or CZYX."
        )
    z_axis = leading_axes.index("Z")
    c_axis = leading_axes.index("C")
    explicit_channels = int(leading_shape[c_axis])
    if explicit_channels != channels:
        raise ValueError(
            f"{path} declares {explicit_channels} channel(s) from axes {axes!r}, but channels={channels} was requested."
        )
    return int(leading_shape[z_axis]), z_axis, c_axis


def _is_contiguous(keys: list[int]) -> bool:
    return keys == list(range(keys[0], keys[-1] + 1))


def _read_summary_metadata(path: Path) -> SourceMetadata:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        payload = {
            "metadata_mode": "summary",
            "raw_shape": tuple(int(v) for v in series.shape),
            "raw_dtype": str(series.dtype),
            "axes": series.axes,
            "page_count": len(tif.pages),
        }
    encoded = json_dumps_strict(payload, context=f"Summary source metadata for {path}").encode("utf-8")
    return SourceMetadata(
        shaped_metadata=[],
        imagej_metadata=None,
        ome_xml=None,
        tags={"axes": payload["axes"], "page_count": payload["page_count"], "metadata_mode": "summary"},
        raw_shape=payload["raw_shape"],
        raw_dtype=payload["raw_dtype"],
        metadata_hash=hashlib.sha256(encoded).hexdigest(),
    )
