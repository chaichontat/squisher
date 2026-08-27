"""Open local TIFF and OME-Zarr tiles behind one axes/channel contract.

This module owns storage dispatch and array normalization. Positioning,
registration, and channel alignment remain the caller's responsibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from squisher_lightsheet import ngff
from squisher_lightsheet.tiff_input import TiffInputHandler


SUPPORTED_AXES = {"ZYX", "CZYX", "ZCYX"}


def is_ome_zarr_path(path: Path) -> bool:
    return path.is_dir() and (
        path.name.endswith(".zarr") or (path / "zarr.json").exists() or (path / ".zgroup").exists()
    )


def open_ome_zarr_level_array(path: Path, *, source_level: int = 0) -> Any:
    import zarr

    group = zarr.open_group(str(path), mode="r")
    dataset_paths = ngff.dataset_paths(group)
    if not dataset_paths:
        raise ValueError(f"{path} does not contain any OME-Zarr datasets")
    if source_level < 0 or source_level >= len(dataset_paths):
        raise ValueError(f"{path} has {len(dataset_paths)} OME-Zarr level(s), cannot open level {source_level}")
    return group[dataset_paths[source_level]]


def open_array(
    path: Path,
    *,
    source_level: int = 0,
    tiff_inputs: TiffInputHandler | None = None,
) -> tuple[Any, Any | None]:
    if is_ome_zarr_path(path):
        return open_ome_zarr_level_array(path, source_level=source_level), None
    return (tiff_inputs or TiffInputHandler()).open(path, level=source_level)


def source_axes(metadata_axes: str, source_shape: tuple[int, ...]) -> str:
    if metadata_axes not in SUPPORTED_AXES:
        raise ValueError(f"Expected CZYX, ZCYX, or ZYX axes, got {metadata_axes!r}")
    if len(source_shape) == len(metadata_axes):
        return metadata_axes
    if "C" in metadata_axes and len(source_shape) == 3:
        return "ZYX"
    raise ValueError(
        f"Expected source shape rank to match {metadata_axes} or ZYX, got shape {source_shape}"
    )


def spatial_shape_zyx(shape: tuple[int, ...], axes: str) -> tuple[int, int, int]:
    if axes not in SUPPORTED_AXES or len(shape) != len(axes):
        raise ValueError(f"Expected CZYX, ZCYX, or ZYX shape, got axes={axes!r}, shape={shape}")
    return tuple(int(shape[axes.index(dim)]) for dim in "ZYX")


def validate_channel(path: Path, shape: tuple[int, ...], axes: str, channel: int) -> None:
    spatial_shape_zyx(shape, axes)
    channel_count = 1 if axes == "ZYX" else int(shape[axes.index("C")])
    if not 0 <= channel < channel_count:
        raise ValueError(f"Channel {channel} out of range for {axes} tile {path}; shape={shape}")


def channel_view(array: Any, axes: str, channel: int, *, path: Path) -> Any:
    shape = tuple(int(value) for value in array.shape)
    validate_channel(path, shape, axes, channel)
    if axes == "ZYX":
        return array
    selection = [slice(None)] * len(shape)
    selection[axes.index("C")] = channel
    return array[tuple(selection)]
