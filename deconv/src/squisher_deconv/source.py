from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from xml.etree import ElementTree

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
        if metadata_mode == "summary":
            with tifffile.TiffFile(path) as tif:
                metadata, axes = _summary_metadata(path, tif=tif)
            page_count = None
        else:
            metadata = read_source_metadata(path)
            with tifffile.TiffFile(path) as tif:
                page_count = len(tif.pages)
                axes = tif.series[0].axes
        shape = metadata.raw_shape
        if len(shape) < 3:
            raise ValueError(f"{path} must have at least flattened planes plus Y/X, got shape {shape}")
        height, width = int(shape[-2]), int(shape[-1])
        plane_count = int(np.prod(shape[:-2]))
        if plane_count % channels:
            raise ValueError(f"{path} has {plane_count} plane(s), not divisible by channels={channels}")
        if page_count is not None and page_count != plane_count:
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

    def page_key(self, *, channel: int, z: int) -> int:
        if not 0 <= channel < self.channels:
            raise ValueError(f"Channel {channel} is outside channel count {self.channels}")
        if not 0 <= z < self.z_count:
            raise ValueError(f"Z index {z} is outside z_count={self.z_count}")
        if self._c_axis is None:
            return z * self.channels + channel
        if self._z_axis is None:
            raise RuntimeError("Explicit channel-axis sources must also have a z axis.")
        index = [0] * len(self._leading_shape)
        index[self._z_axis] = z
        index[self._c_axis] = channel
        return int(np.ravel_multi_index(tuple(index), self._leading_shape))

    def _page_keys(self, start: int, stop: int) -> list[int]:
        if self._c_axis is None:
            flat_start = start * self.channels
            flat_stop = stop * self.channels
            return list(range(flat_start, flat_stop))
        return [
            self.page_key(channel=channel, z=z_index)
            for z_index in range(start, stop)
            for channel in range(self.channels)
        ]


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


def _summary_metadata(path: Path, *, tif: tifffile.TiffFile) -> tuple[SourceMetadata, str]:
    ome_xml = tif.ome_metadata
    if ome_xml is None:
        raise ValueError(f"{path} is missing OME-XML metadata; summary inspection requires OME-TIFF input.")
    raw_shape, axes = _ome_pixels_shape(path, ome_xml=ome_xml)
    raw_dtype = str(tif.pages[0].dtype)
    payload = {
        "metadata_mode": "summary",
        "raw_shape": raw_shape,
        "raw_dtype": raw_dtype,
        "axes": axes,
        "page_count_validated": False,
    }
    encoded = json_dumps_strict(payload, context=f"Summary source metadata for {path}").encode("utf-8")
    return SourceMetadata(
        shaped_metadata=[],
        imagej_metadata=None,
        ome_xml=None,
        tags={
            "axes": payload["axes"],
            "metadata_mode": "summary",
            "page_count_validated": False,
        },
        raw_shape=raw_shape,
        raw_dtype=raw_dtype,
        metadata_hash=hashlib.sha256(encoded).hexdigest(),
    ), axes


def _ome_pixels_shape(path: Path, *, ome_xml: str) -> tuple[tuple[int, ...], str]:
    try:
        root = ElementTree.fromstring(ome_xml)
    except ElementTree.ParseError as exc:
        raise ValueError(f"Invalid OME-XML metadata in {path}: {exc}") from exc
    pixels = next((element for element in root.iter() if _local_name(element.tag) == "Pixels"), None)
    if pixels is None:
        raise ValueError(f"OME-XML metadata in {path} does not contain a Pixels element.")

    order = pixels.attrib.get("DimensionOrder")
    if order is None or not order.startswith("XY"):
        raise ValueError(f"OME-XML metadata in {path} has unsupported DimensionOrder={order!r}.")
    sizes = {
        axis: _ome_axis_size(path, pixels=pixels, axis=axis)
        for axis in ("X", "Y", "Z", "C", "T")
    }
    leading_axes = [axis for axis in reversed(order[2:]) if sizes[axis] > 1]
    axes = "".join(leading_axes) + "YX"
    raw_shape = tuple(sizes[axis] for axis in leading_axes) + (sizes["Y"], sizes["X"])
    return raw_shape, axes


def _ome_axis_size(path: Path, *, pixels: ElementTree.Element, axis: str) -> int:
    value = pixels.attrib.get(f"Size{axis}")
    if value is None:
        raise ValueError(f"OME-XML metadata in {path} is missing Size{axis}.")
    try:
        size = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid Size{axis} in {path}: {value!r}") from exc
    if size < 1:
        raise ValueError(f"Invalid Size{axis} in {path}: {size}")
    return size


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
