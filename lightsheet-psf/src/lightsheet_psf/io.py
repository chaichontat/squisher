from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from aicspylibczi import CziFile


def load_image_zyx(path: Path, channel: int | None = None) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        return load_tiff_zyx(path, channel=channel)
    return load_czi_zyx(path, channel=channel)


def _select_zyx(path: Path, arr: np.ndarray, dims: str, channel: int | None) -> np.ndarray:
    axes = {axis: idx for idx, axis in enumerate(dims)}
    missing = {"Z", "Y", "X"} - set(axes)
    if missing:
        raise ValueError(f"{path} is missing expected axes: {sorted(missing)}")

    selectors: list[int | slice] = []
    remaining_dims: list[str] = []
    for idx, axis in enumerate(dims):
        if axis in {"Z", "Y", "X"}:
            selectors.append(slice(None))
            remaining_dims.append(axis)
        elif axis == "C" and arr.shape[idx] != 1:
            if channel is None:
                raise ValueError(f"{path} has {arr.shape[idx]} channels; pass --channel.")
            if channel < 0 or channel >= arr.shape[idx]:
                raise ValueError(f"--channel must be in [0, {arr.shape[idx] - 1}], got {channel}.")
            selectors.append(channel)
        else:
            if arr.shape[idx] != 1:
                raise ValueError(
                    f"{path} has non-singleton axis {axis} with size {arr.shape[idx]}; "
                    "choose a scene/channel/timepoint before extracting beads."
                )
            selectors.append(0)

    selected = arr[tuple(selectors)]
    selected_axes = {axis: idx for idx, axis in enumerate(remaining_dims)}
    zyx_axes = [selected_axes[axis] for axis in "ZYX"]
    return np.moveaxis(selected, zyx_axes, [0, 1, 2]).reshape(
        selected.shape[selected_axes["Z"]],
        selected.shape[selected_axes["Y"]],
        selected.shape[selected_axes["X"]],
    )


def load_tiff_zyx(path: Path, channel: int | None = None) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        arr = series.asarray()
        dims = series.axes
    return _select_zyx(path, arr, dims, channel)


def load_czi_zyx(path: Path, channel: int | None = None) -> np.ndarray:
    czi = CziFile(str(path))
    arr, _ = czi.read_image()
    return _select_zyx(path, arr, czi.dims, channel)
