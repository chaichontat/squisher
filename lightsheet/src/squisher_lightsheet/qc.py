from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from squisher_lightsheet.legacy_runner import run_legacy_script


SIDES = ("L", "R")


@dataclass(frozen=True)
class LiveFusionPreviewResult:
    output: Path
    source_zarr: Path
    shape: tuple[int, ...]
    chunks: tuple[int, ...] | None
    dtype: str
    stride: int
    sampled_plane_count: int
    selected_z: tuple[int, ...]
    selected_nonzero_pixels: tuple[int, ...]


def scale_u8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(positive, [1.0, 99.8])
    scaled = np.clip((image - low) / max(float(high - low), 1.0), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def scale_u8_with_limits(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size == 0:
        return np.zeros(image.shape, dtype=np.uint8), (0.0, 1.0)
    low, high = np.percentile(sample, [1.0, 99.8])
    if not np.isfinite(high) or high <= low:
        low = float(np.min(sample))
        high = float(np.max(sample)) if float(np.max(sample)) > low else low + 1.0
    scaled = np.clip((image - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8), (float(low), float(high))


def scale_positive_u8(image: np.ndarray, high_percentile: float = 99.7) -> np.ndarray:
    if high_percentile <= 0.0 or high_percentile > 100.0:
        raise ValueError(f"high_percentile must be in (0, 100], got {high_percentile}")
    image = np.asarray(image, dtype=np.float32)
    positive = image[np.isfinite(image) & (image > 0)]
    if positive.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    high = float(np.percentile(positive, high_percentile))
    if not np.isfinite(high) or high <= 0:
        high = float(positive.max())
    if high <= 0:
        return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip(image / high, 0.0, 1.0) * 255).astype(np.uint8)


def place_max(canvas: np.ndarray, image: np.ndarray, start_yx: tuple[int, int]) -> None:
    y0, x0 = start_yx
    if y0 >= canvas.shape[0] or x0 >= canvas.shape[1]:
        return
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    y_size = min(image.shape[0] - src_y0, canvas.shape[0] - dst_y0)
    x_size = min(image.shape[1] - src_x0, canvas.shape[1] - dst_x0)
    if y_size <= 0 or x_size <= 0:
        return
    canvas[dst_y0 : dst_y0 + y_size, dst_x0 : dst_x0 + x_size] = np.maximum(
        canvas[dst_y0 : dst_y0 + y_size, dst_x0 : dst_x0 + x_size],
        image[src_y0 : src_y0 + y_size, src_x0 : src_x0 + x_size],
    )


def rgb_overlay_image(*, left: np.ndarray, right: np.ndarray):
    from PIL import Image

    rgb = np.zeros((*left.shape, 3), dtype=np.uint8)
    rgb[..., 0] = scale_u8(right)
    rgb[..., 1] = scale_u8(left)
    return Image.fromarray(rgb)


def write_overlay(path: Path, *, left: np.ndarray, right: np.ndarray) -> None:
    rgb_overlay_image(left=left, right=right).save(path)


def write_overlay_scaled(path: Path, *, left: np.ndarray, right: np.ndarray, y_scale: float = 1.0) -> None:
    from PIL import Image

    image = rgb_overlay_image(left=left, right=right)
    if not np.isclose(y_scale, 1.0):
        image = image.resize(
            (image.width, max(1, int(round(image.height * y_scale)))),
            Image.Resampling.BILINEAR,
        )
    image.save(path)


def write_contact_sheet(output: Path, images: list[tuple[str, Path]]) -> None:
    from PIL import Image, ImageDraw

    opened = [(title, Image.open(path).convert("RGB")) for title, path in images]
    target_width = max(image.width for _, image in opened)
    resized = []
    for title, image in opened:
        if image.width != target_width:
            height = max(1, int(round(image.height * target_width / image.width)))
            image = image.resize((target_width, height), Image.Resampling.LANCZOS)
        resized.append((title, image))
    title_height = 28
    sheet = Image.new("RGB", (target_width, sum(image.height + title_height for _, image in resized)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for title, image in resized:
        draw.text((8, y + 7), title, fill=(0, 0, 0))
        y += title_height
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output)


def _open_zarr_level(source_zarr: Path, level: int):
    import zarr

    level_path = source_zarr / str(level)
    if level_path.exists():
        return zarr.open_array(str(level_path), mode="r")
    if level == 0:
        return zarr.open_array(str(source_zarr), mode="r")
    raise ValueError(f"{source_zarr} has no pyramid level {level}")


def _plane_from_fused_array(array, *, z: int, channel: int, stride: int) -> np.ndarray:
    if array.ndim == 3:
        return np.asarray(array[z, ::stride, ::stride])
    if array.ndim == 4:
        if channel < 0 or channel >= array.shape[0]:
            raise ValueError(f"channel {channel} is outside CZYX range [0, {array.shape[0]})")
        return np.asarray(array[channel, z, ::stride, ::stride])
    if array.ndim == 5:
        if channel < 0 or channel >= array.shape[1]:
            raise ValueError(f"channel {channel} is outside TCZYX range [0, {array.shape[1]})")
        return np.asarray(array[0, channel, z, ::stride, ::stride])
    raise ValueError(f"Unsupported fused array shape {array.shape}; expected ZYX, CZYX, or TCZYX")


def render_live_fusion_preview(
    *,
    source_zarr: Path | None = None,
    log_path: Path | None = None,
    output: Path | None = None,
    output_dir: Path | None = None,
    level: int = 0,
    channel: int = 0,
    color: Literal["red", "green", "blue", "gray"] = "red",
    stride: int = 10,
    z_start: int = 6,
    z_step: int = 24,
    max_panels: int = 6,
    high_percentile: float = 99.7,
) -> LiveFusionPreviewResult:
    """Render a contact sheet from nonzero planes already written in a live fused OME-Zarr."""
    if source_zarr is None:
        if log_path is None:
            raise ValueError("Either source_zarr or log_path is required")
        matches = re.findall(r"streaming write to (/.+)", log_path.read_text())
        if not matches:
            raise ValueError(f"Could not find a 'streaming write to ...' output path in {log_path}")
        source_zarr = Path(matches[-1].strip())
    if stride < 1:
        raise ValueError("stride must be positive")
    if z_step < 1:
        raise ValueError("z_step must be positive")
    if max_panels < 1:
        raise ValueError("max_panels must be positive")

    array = _open_zarr_level(source_zarr, level)
    z_size = int(array.shape[-3])
    written: list[tuple[int, int]] = []
    for z in range(z_start, z_size, z_step):
        plane = _plane_from_fused_array(array, z=z, channel=channel, stride=stride)
        nonzero = int(np.count_nonzero(plane))
        if nonzero:
            written.append((z, nonzero))
    if not written:
        raise ValueError(f"No nonzero planes found in {source_zarr} level {level}")

    if len(written) <= max_panels:
        selected = written
    else:
        indices = np.linspace(0, len(written) - 1, max_panels).round().astype(int)
        selected = [written[int(index)] for index in indices]

    from PIL import Image, ImageDraw

    tiles = []
    labels = []
    for z, nonzero in selected:
        image_u8 = scale_positive_u8(
            _plane_from_fused_array(array, z=z, channel=channel, stride=stride),
            high_percentile=high_percentile,
        )
        rgb = np.zeros((*image_u8.shape, 3), dtype=np.uint8)
        if color == "red":
            rgb[..., 0] = image_u8
        elif color == "green":
            rgb[..., 1] = image_u8
        elif color == "blue":
            rgb[..., 2] = image_u8
        elif color == "gray":
            rgb[...] = image_u8[..., None]
        else:
            raise ValueError(f"Unsupported preview color {color!r}")
        tiles.append(Image.fromarray(rgb))
        labels.append(f"z={z} nnz={nonzero}")

    tile_width, tile_height = tiles[0].size
    cols = min(3, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    label_height = 28
    sheet = Image.new("RGB", (cols * tile_width, rows * (tile_height + label_height)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, tile in enumerate(tiles):
        x = (index % cols) * tile_width
        y = (index // cols) * (tile_height + label_height)
        sheet.paste(tile, (x, y + label_height))
        draw.text((x + 8, y + 7), labels[index], fill=(255, 255, 255))

    if output is None:
        target_dir = source_zarr.parent if output_dir is None else output_dir
        stem = source_zarr.name.removesuffix(".ome.zarr").removesuffix(".zarr")
        output = target_dir / f"{stem}.live-written-z-preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return LiveFusionPreviewResult(
        output=output.resolve(),
        source_zarr=source_zarr.resolve(),
        shape=tuple(int(value) for value in array.shape),
        chunks=None if getattr(array, "chunks", None) is None else tuple(int(value) for value in array.chunks),
        dtype=str(array.dtype),
        stride=int(stride),
        sampled_plane_count=len(written),
        selected_z=tuple(int(z) for z, _ in selected),
        selected_nonzero_pixels=tuple(int(nonzero) for _, nonzero in selected),
    )


def _tile_index_label(tile_name: str) -> str:
    basename = Path(tile_name).name
    match = re.search(r"^Image_\d+\.(\d+)(?:\.|$)", basename)
    if match is None:
        match = re.search(r"^Image_\d+\.(\d+)_", basename)
    if match is not None:
        return str(int(match.group(1)))
    match = re.search(r"\.(\d+)\.ome\.zarr$", basename)
    if match is None:
        match = re.search(r"\.(\d+)\.ome\.tif$", basename)
    if match is None:
        match = re.search(r"(\d+)", basename)
    return str(int(match.group(1))) if match else basename


def _ngff_multiscale(root_attrs: dict[str, Any]) -> dict[str, Any]:
    if "ome" in root_attrs:
        multiscales = root_attrs["ome"].get("multiscales")
    else:
        multiscales = root_attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("OME-Zarr root metadata is missing multiscales")
    multiscale = multiscales[0]
    if not isinstance(multiscale, dict):
        raise ValueError("OME-Zarr multiscales[0] must be an object")
    return multiscale


def _ngff_axes(multiscale: dict[str, Any], ndim: int) -> list[str]:
    axes = multiscale.get("axes")
    if isinstance(axes, list) and len(axes) == ndim:
        names = [axis.get("name") if isinstance(axis, dict) else axis for axis in axes]
        if all(isinstance(name, str) for name in names):
            return list(names)
    defaults = {
        2: ["y", "x"],
        3: ["z", "y", "x"],
        4: ["c", "z", "y", "x"],
        5: ["t", "c", "z", "y", "x"],
    }
    if ndim not in defaults:
        raise ValueError(f"Cannot infer fused OME-Zarr axes for ndim={ndim}")
    return defaults[ndim]


def _ngff_scale_translation(multiscale: dict[str, Any], level: int, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("OME-Zarr multiscales[0] is missing datasets")
    dataset = next((item for item in datasets if isinstance(item, dict) and str(item.get("path")) == str(level)), None)
    if dataset is None:
        raise ValueError(f"OME-Zarr multiscales[0] has no dataset path {level!r}")
    scale = np.ones(ndim, dtype=np.float64)
    translation = np.zeros(ndim, dtype=np.float64)
    transforms = dataset.get("coordinateTransformations", [])
    if not isinstance(transforms, list):
        raise ValueError(f"OME-Zarr dataset {level!r} coordinateTransformations must be a list")
    for transform in transforms:
        if not isinstance(transform, dict):
            continue
        values = transform.get("scale" if transform.get("type") == "scale" else "translation")
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (ndim,):
            raise ValueError(
                f"OME-Zarr dataset {level!r} transform {transform.get('type')!r} "
                f"has {array.size} values for ndim={ndim}"
            )
        if transform.get("type") == "scale":
            scale = array
        elif transform.get("type") == "translation":
            translation = array
    return scale, translation


def _zyx_values(record: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict) and all(dim in value for dim in ("z", "y", "x")):
            return np.asarray([value[dim] for dim in ("z", "y", "x")], dtype=np.float64)
    raise ValueError(f"Tile record {record.get('tile')!r} is missing one of {keys}")


def _tile_shape_zyx(record: dict[str, Any]) -> np.ndarray:
    shape = record.get("shape")
    if not isinstance(shape, list | tuple) or len(shape) < 3:
        raise ValueError(f"Tile record {record.get('tile')!r} is missing a z/y/x shape")
    return np.asarray(shape[-3:], dtype=np.float64)


def _registered_affine_zyx(record: dict[str, Any]) -> np.ndarray:
    affine = record.get("registered_affine")
    if affine is None:
        return np.eye(4, dtype=np.float64)
    matrix = affine.get("matrix") if isinstance(affine, dict) else affine
    array = np.asarray(matrix, dtype=np.float64)
    while array.ndim > 2:
        array = array[0]
    if array.shape[0] < 4 or array.shape[1] < 4:
        raise ValueError(f"Tile record {record.get('tile')!r} registered_affine must be at least 4x4")
    return array[:4, :4]


def _tile_registered_center_um_zyx(record: dict[str, Any]) -> np.ndarray:
    shape_zyx = _tile_shape_zyx(record)
    stage_translation = _zyx_values(record, "stage_translation_um", "translation_um")
    stage_scale = _zyx_values(record, "stage_scale_um", "scale_um", "spacing_um")
    sim_translation = stage_translation.copy()
    negative_axes = stage_scale < 0
    sim_translation[negative_axes] = stage_translation[negative_axes] + shape_zyx[negative_axes] * stage_scale[negative_axes]
    stage_center = sim_translation + 0.5 * shape_zyx * np.abs(stage_scale)
    return (_registered_affine_zyx(record) @ np.r_[stage_center, 1.0])[:3]


def default_fused_tile_index_overlay_path(fused_zarr: Path, level: int) -> Path:
    name = fused_zarr.name
    if name.endswith(".ome.zarr"):
        stem = name.removesuffix(".ome.zarr")
    elif name.endswith(".zarr"):
        stem = name.removesuffix(".zarr")
    else:
        stem = name
    if ".level0." in stem:
        stem = stem.replace(".level0.", f".level{level}.", 1)
        return fused_zarr.with_name(f"{stem}.center-z.tile-index.png")
    return fused_zarr.with_name(f"{stem}.level{level}.center-z.tile-index.png")


def render_fused_tile_index_overlay(
    *,
    fused_zarr: Path,
    registration_input: Path,
    output: Path | None = None,
    summary_output: Path | None = None,
    level: int = 2,
    z_index: int | None = None,
) -> Path:
    """Render a fused OME-Zarr center-z PNG with numeric tile-index labels."""
    from PIL import Image, ImageDraw, ImageFont
    import zarr

    output = default_fused_tile_index_overlay_path(fused_zarr, level) if output is None else output
    summary_output = output.with_suffix(".json") if summary_output is None else summary_output
    root = zarr.open_group(fused_zarr, mode="r")
    level_key = str(level)
    if level_key not in root:
        raise ValueError(f"{fused_zarr} does not contain pyramid level {level}")
    array = root[level_key]
    multiscale = _ngff_multiscale(dict(root.attrs))
    axes = _ngff_axes(multiscale, len(array.shape))
    for dim in ("z", "y", "x"):
        if dim not in axes:
            raise ValueError(f"{fused_zarr}/{level} axes {axes} do not contain {dim!r}")
    z_axis = axes.index("z")
    y_axis = axes.index("y")
    x_axis = axes.index("x")
    z_index = array.shape[z_axis] // 2 if z_index is None else int(z_index)
    if z_index < 0 or z_index >= array.shape[z_axis]:
        raise ValueError(f"z_index {z_index} is outside level {level} z range [0, {array.shape[z_axis]})")

    indexer: list[int | slice] = []
    for axis in range(len(array.shape)):
        if axis == z_axis:
            indexer.append(z_index)
        elif axis in (y_axis, x_axis):
            indexer.append(slice(None))
        else:
            indexer.append(0)
    plane = np.asarray(array[tuple(indexer)])
    remaining_axes = [axis for axis in range(len(array.shape)) if axis in (y_axis, x_axis)]
    if remaining_axes == [x_axis, y_axis]:
        plane = plane.T
    elif remaining_axes != [y_axis, x_axis]:
        raise ValueError(f"Cannot reduce axes {axes} to a y/x plane")

    scaled, limits = scale_u8_with_limits(plane)
    image = Image.fromarray(scaled, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    scale, translation = _ngff_scale_translation(multiscale, level, len(array.shape))
    axis_to_index = {axis: index for index, axis in enumerate(axes)}
    scale_zyx = np.asarray([scale[axis_to_index[dim]] for dim in ("z", "y", "x")], dtype=np.float64)
    translation_zyx = np.asarray([translation[axis_to_index[dim]] for dim in ("z", "y", "x")], dtype=np.float64)

    payload = json.loads(registration_input.read_text())
    records = []
    for record in payload.get("tiles", []):
        if not isinstance(record, dict):
            raise ValueError(f"{registration_input} contains a non-object tile record")
        label = _tile_index_label(str(record.get("tile", Path(str(record.get("path", ""))).name)))
        center_um = _tile_registered_center_um_zyx(record)
        center_level_zyx = (center_um - translation_zyx) / scale_zyx
        y = int(round(center_level_zyx[1]))
        x = int(round(center_level_zyx[2]))
        drawn = 0 <= x < image.width and 0 <= y < image.height
        if drawn:
            bbox = draw.textbbox((x, y), label, font=font, stroke_width=3)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = (
                min(max(2, x - text_width // 2), max(2, image.width - text_width - 2)),
                min(max(2, y - text_height // 2), max(2, image.height - text_height - 2)),
            )
            draw.text(position, label, fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 210, 0), outline=(0, 0, 0))
        records.append(
            {
                "tile": record.get("tile"),
                "label": label,
                "center_registered_um_zyx": [float(value) for value in center_um],
                "center_level_zyx": [float(value) for value in center_level_zyx],
                "drawn": drawn,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    summary_output.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.fused_tile_index_overlay.v1",
                "source_zarr": str(fused_zarr.resolve()),
                "registration_input": str(registration_input.resolve()),
                "overlay_png": str(output.resolve()),
                "level": level,
                "z_index": z_index,
                "axes": axes,
                "shape": [int(value) for value in array.shape],
                "intensity_percentiles": {"low": limits[0], "high": limits[1]},
                "coordinate_rule": (
                    "level_zyx = (registered_affine @ stage_center_um - "
                    "ngff_level_translation_um) / ngff_level_scale_um"
                ),
                "labels": "numeric tile index only",
                "drawn_count": sum(1 for record in records if record["drawn"]),
                "tile_count": len(records),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def _fused_level_array(root: Path, level: int):
    import zarr

    group = zarr.open_group(root, mode="r")
    level_key = str(level)
    if level_key not in group:
        raise ValueError(f"{root} does not contain pyramid level {level}")
    return group[level_key]


def _fused_axes(array) -> list[str]:
    defaults = {
        2: ["y", "x"],
        3: ["z", "y", "x"],
        4: ["c", "z", "y", "x"],
        5: ["t", "c", "z", "y", "x"],
    }
    if array.ndim not in defaults:
        raise ValueError(f"Cannot infer fused OME-Zarr axes for ndim={array.ndim}")
    return defaults[array.ndim]


def _zyx_shape(array) -> tuple[int, int, int]:
    axes = _fused_axes(array)
    for dim in ("z", "y", "x"):
        if dim not in axes:
            raise ValueError(f"Fused array axes {axes} do not contain {dim!r}")
    return tuple(int(array.shape[axes.index(dim)]) for dim in ("z", "y", "x"))


def _array_indexer_for_zyx(array, z_sel: int | slice, y_sel: int | slice, x_sel: int | slice) -> tuple[int | slice, ...]:
    axes = _fused_axes(array)
    indexer: list[int | slice] = []
    for axis in axes:
        if axis == "z":
            indexer.append(z_sel)
        elif axis == "y":
            indexer.append(y_sel)
        elif axis == "x":
            indexer.append(x_sel)
        else:
            indexer.append(0)
    return tuple(indexer)


def _read_zyx_crop(
    array,
    *,
    axis: Literal["xy", "xz", "yz"],
    z: int,
    y: int,
    x: int,
    panel_size: int,
) -> np.ndarray:
    shape_zyx = _zyx_shape(array)
    half = panel_size // 2
    z_slice = slice(max(0, z - half), min(shape_zyx[0], z - half + panel_size))
    y_slice = slice(max(0, y - half), min(shape_zyx[1], y - half + panel_size))
    x_slice = slice(max(0, x - half), min(shape_zyx[2], x - half + panel_size))
    if axis == "xy":
        data = np.asarray(array[_array_indexer_for_zyx(array, z, y_slice, x_slice)])
        center_yx = (y - int(y_slice.start), x - int(x_slice.start))
    elif axis == "xz":
        data = np.asarray(array[_array_indexer_for_zyx(array, z_slice, y, x_slice)])
        axes = _fused_axes(array)
        remaining = [dim for dim in axes if dim in {"z", "x"}]
        data = data.transpose([remaining.index("z"), remaining.index("x")]) if remaining != ["z", "x"] else data
        center_yx = (z - int(z_slice.start), x - int(x_slice.start))
    elif axis == "yz":
        data = np.asarray(array[_array_indexer_for_zyx(array, z_slice, y_slice, x)])
        axes = _fused_axes(array)
        remaining = [dim for dim in axes if dim in {"z", "y"}]
        data = data.transpose([remaining.index("z"), remaining.index("y")]) if remaining != ["z", "y"] else data
        center_yx = (z - int(z_slice.start), y - int(y_slice.start))
    else:
        raise ValueError(axis)
    return _crop_with_padding(data, center_yx, panel_size)


def _read_xy_plane(array, z: int) -> np.ndarray:
    data = np.asarray(array[_array_indexer_for_zyx(array, z, slice(None), slice(None))])
    axes = _fused_axes(array)
    remaining = [dim for dim in axes if dim in {"y", "x"}]
    return data.transpose([remaining.index("y"), remaining.index("x")]) if remaining != ["y", "x"] else data


def _crop_with_padding(plane: np.ndarray, center_yx: tuple[int, int], panel_size: int) -> np.ndarray:
    y, x = center_yx
    half = panel_size // 2
    y0 = int(y) - half
    x0 = int(x) - half
    y1 = y0 + panel_size
    x1 = x0 + panel_size
    out = np.zeros((panel_size, panel_size), dtype=plane.dtype)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(plane.shape[0], y1)
    src_x1 = min(plane.shape[1], x1)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    out[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = plane[src_y0:src_y1, src_x0:src_x1]
    return out


def _overlay_rgb(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    ref, _ = scale_u8_with_limits(reference)
    mov, _ = scale_u8_with_limits(moving)
    rgb = np.zeros((*ref.shape, 3), dtype=np.uint8)
    rgb[..., 0] = ref
    rgb[..., 1] = mov
    rgb[..., 2] = ((ref.astype(np.uint16) + mov.astype(np.uint16)) // 8).astype(np.uint8)
    return rgb


def _label_panel(image, label: str):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, bbox[2] + 8, bbox[3] + 6), fill=(0, 0, 0, 180))
    draw.text((4, 3), label, fill=(255, 255, 255, 255), font=font)
    return image


def _write_panel_sheet(panels: list[tuple[str, np.ndarray]], *, output: Path, panel_size: int, cols: int = 3) -> None:
    from PIL import Image

    rows = int(np.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (cols * panel_size, rows * panel_size), "black")
    for index, (label, rgb) in enumerate(panels):
        panel = _label_panel(Image.fromarray(rgb, mode="RGB"), label)
        sheet.paste(panel, ((index % cols) * panel_size, (index // cols) * panel_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def render_fused_xyz_overlay_qc(
    *,
    reference_zarr: Path,
    moving_zarr: Path,
    output_dir: Path,
    level: int = 0,
    thumb_level: int = 2,
    panel_size: int = 512,
) -> Path:
    """Render native crop overlays spanning x/y/z from two same-grid fused Zarrs."""
    if panel_size < 1:
        raise ValueError("panel_size must be positive")
    reference = _fused_level_array(reference_zarr, level)
    moving = _fused_level_array(moving_zarr, level)
    if _zyx_shape(reference) != _zyx_shape(moving):
        raise ValueError(f"Fused level {level} shape mismatch: reference={reference.shape}, moving={moving.shape}")
    if level == thumb_level:
        reference_thumb = reference
        moving_thumb = moving
    else:
        reference_thumb = _fused_level_array(reference_zarr, thumb_level)
        moving_thumb = _fused_level_array(moving_zarr, thumb_level)
    if _zyx_shape(reference_thumb) != _zyx_shape(moving_thumb):
        raise ValueError(
            f"Fused thumb level {thumb_level} shape mismatch: reference={reference_thumb.shape}, moving={moving_thumb.shape}"
        )

    shape_zyx = _zyx_shape(reference)
    thumb_shape_zyx = _zyx_shape(reference_thumb)
    scale_zyx = np.maximum(1, np.asarray(shape_zyx, dtype=np.float64) / np.asarray(thumb_shape_zyx, dtype=np.float64))

    z_values = [int(value) for value in np.linspace(panel_size // 2, max(panel_size // 2, shape_zyx[0] - panel_size // 2 - 1), 5)]
    panel_records: list[dict[str, Any]] = []
    panels_by_axis: dict[str, list[tuple[str, np.ndarray]]] = {"xy": [], "xz": [], "yz": []}
    for z in z_values:
        z_thumb = int(np.clip(round(z / scale_zyx[0]), 0, thumb_shape_zyx[0] - 1))
        slab = np.maximum(_read_xy_plane(reference_thumb, z_thumb), _read_xy_plane(moving_thumb, z_thumb))
        y_edges = np.linspace(0, slab.shape[0], 4, dtype=int)
        for band_index in range(3):
            band = slab[y_edges[band_index] : y_edges[band_index + 1]]
            if band.size:
                yy, xx = np.unravel_index(int(np.argmax(band)), band.shape)
                y = int(round((yy + y_edges[band_index]) * scale_zyx[1]))
                x = int(round(xx * scale_zyx[2]))
            else:
                y = shape_zyx[1] // 2
                x = shape_zyx[2] // 2
            ref_crop = _read_zyx_crop(reference, axis="xy", z=z, y=y, x=x, panel_size=panel_size)
            mov_crop = _read_zyx_crop(moving, axis="xy", z=z, y=y, x=x, panel_size=panel_size)
            label = f"XY z={z} y={y} x={x}"
            panels_by_axis["xy"].append((label, _overlay_rgb(ref_crop, mov_crop)))
            panel_records.append({"axis": "xy", "z": z, "y": y, "x": x})

    y_values = [int(value) for value in np.linspace(panel_size // 2, max(panel_size // 2, shape_zyx[1] - panel_size // 2 - 1), 3)]
    x_values = [int(value) for value in np.linspace(panel_size // 2, max(panel_size // 2, shape_zyx[2] - panel_size // 2 - 1), 3)]
    for index, y in enumerate(y_values):
        z = z_values[index % len(z_values)]
        x = x_values[index % len(x_values)]
        ref_crop = _read_zyx_crop(reference, axis="xz", z=z, y=y, x=x, panel_size=panel_size)
        mov_crop = _read_zyx_crop(moving, axis="xz", z=z, y=y, x=x, panel_size=panel_size)
        label = f"XZ z={z} y={y} x={x}"
        panels_by_axis["xz"].append((label, _overlay_rgb(ref_crop, mov_crop)))
        panel_records.append({"axis": "xz", "z": z, "y": y, "x": x})
    for index, x in enumerate(x_values):
        z = z_values[index % len(z_values)]
        y = y_values[index % len(y_values)]
        ref_crop = _read_zyx_crop(reference, axis="yz", z=z, y=y, x=x, panel_size=panel_size)
        mov_crop = _read_zyx_crop(moving, axis="yz", z=z, y=y, x=x, panel_size=panel_size)
        label = f"YZ z={z} y={y} x={x}"
        panels_by_axis["yz"].append((label, _overlay_rgb(ref_crop, mov_crop)))
        panel_records.append({"axis": "yz", "z": z, "y": y, "x": x})

    output_dir.mkdir(parents=True, exist_ok=True)
    xy_path = output_dir / "native_xy_zspan_contentful_overlay.png"
    xz_path = output_dir / "native_xz_spanning_overlay.png"
    yz_path = output_dir / "native_yz_spanning_overlay.png"
    combined_path = output_dir / "native_xyz_spanning_overlay_contact_sheet.png"
    _write_panel_sheet(panels_by_axis["xy"], output=xy_path, panel_size=panel_size)
    _write_panel_sheet(panels_by_axis["xz"], output=xz_path, panel_size=panel_size)
    _write_panel_sheet(panels_by_axis["yz"], output=yz_path, panel_size=panel_size)
    combined = panels_by_axis["xy"][:6] + panels_by_axis["xz"] + panels_by_axis["yz"]
    _write_panel_sheet(combined, output=combined_path, panel_size=panel_size)
    (output_dir / "native_xyz_spanning_overlay_contact_sheet.json").write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.fused_xyz_overlay_qc.v1",
                "reference_zarr": str(reference_zarr.resolve()),
                "moving_zarr": str(moving_zarr.resolve()),
                "level": int(level),
                "thumb_level": int(thumb_level),
                "panel_size": int(panel_size),
                "color": {"red": "reference", "green": "moving"},
                "outputs": {
                    "combined": str(combined_path.resolve()),
                    "xy": str(xy_path.resolve()),
                    "xz": str(xz_path.resolve()),
                    "yz": str(yz_path.resolve()),
                },
                "panels": panel_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return combined_path.resolve()


def empty_projection_canvases(
    shape_zyx: np.ndarray,
    *,
    sides: tuple[str, ...] = SIDES,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        side: {
            "xy": np.zeros((shape_zyx[1], shape_zyx[2]), dtype=np.float32),
            "xz": np.zeros((shape_zyx[0], shape_zyx[2]), dtype=np.float32),
            "yz": np.zeros((shape_zyx[0], shape_zyx[1]), dtype=np.float32),
        }
        for side in sides
    }


def place_global_projections(
    projections: dict[str, dict[str, np.ndarray]],
    *,
    side: str,
    volume: np.ndarray,
    start_zyx: np.ndarray,
) -> None:
    place_max(projections[side]["xy"], volume.max(axis=0), (int(start_zyx[1]), int(start_zyx[2])))
    place_max(projections[side]["xz"], volume.max(axis=1), (int(start_zyx[0]), int(start_zyx[2])))
    place_max(projections[side]["yz"], volume.max(axis=2), (int(start_zyx[0]), int(start_zyx[1])))


def side_by_tile(payload: dict[str, Any]) -> dict[str, str]:
    from squisher_lightsheet._legacy import render_lr_level4_registration_qc as legacy

    return legacy.side_by_tile(payload)


def render_registration_qc(
    *,
    position_input: Path,
    registration_input: Path,
    output_dir: Path,
    channel: int = 0,
    level: int = 4,
    center_y_xz: bool = True,
    dry_run: bool = False,
) -> str:
    args = [
        "--position-input",
        str(position_input),
        "--registration-input",
        str(registration_input),
        "--output-dir",
        str(output_dir),
        "--channel",
        str(channel),
        "--level",
        str(level),
    ]
    if center_y_xz:
        args.append("--center-y-xz")
    return run_legacy_script("render_lr_level4_registration_qc.py", args, dry_run=dry_run)


def _write_position_subset_for_registration(
    *,
    position_input: Path,
    registration_input: Path,
    output_dir: Path,
    side: str,
) -> Path:
    position_payload = json.loads(position_input.read_text())
    registration_payload = json.loads(registration_input.read_text())
    registered_tiles = {str(record["tile"]) for record in registration_payload["tiles"]}

    filtered_tiles = []
    for record in position_payload["tiles"]:
        tile_name = Path(record["path"]).name
        if tile_name not in registered_tiles:
            continue
        filtered = dict(record)
        filtered["side"] = side
        filtered_tiles.append(filtered)

    if not filtered_tiles:
        raise ValueError(f"No position records in {position_input} matched registration tiles in {registration_input}")
    missing = sorted(registered_tiles - {Path(record["path"]).name for record in filtered_tiles})
    if missing:
        raise ValueError(f"{position_input} is missing position records for registered tiles: {missing}")

    subset = dict(position_payload)
    subset["tiles"] = filtered_tiles
    subset["tile_count"] = len(filtered_tiles)
    subset["derived_by"] = "squisher_lightsheet.registration_center_z_spotcheck"
    subset["source_position_input"] = str(position_input.resolve())
    subset["source_registration_input"] = str(registration_input.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "registration-center-z-spotcheck.positions.json"
    path.write_text(json.dumps(subset, indent=2) + "\n")
    return path


def render_registration_center_z_spotcheck(
    *,
    position_input: Path,
    registration_input: Path,
    output_dir: Path,
    channels: list[int] | None = None,
    level: int = 4,
    center_z_um: float | None = None,
    side: str = "L",
    dry_run: bool = False,
) -> list[Path]:
    subset_position = _write_position_subset_for_registration(
        position_input=position_input,
        registration_input=registration_input,
        output_dir=output_dir,
        side=side,
    )
    rendered: list[Path] = []
    for channel in channels or [0, 1]:
        args = [
            "--position-input",
            str(subset_position),
            "--registration-input",
            str(registration_input),
            "--output-dir",
            str(output_dir),
            "--channel",
            str(channel),
            "--level",
            str(level),
            "--full-affine-planes",
            "--center-z-only",
            "--skip-global-projections",
        ]
        if center_z_um is not None:
            args.extend(["--center-z-um", str(center_z_um)])
        run_legacy_script("render_lr_level4_registration_qc.py", args, dry_run=dry_run)
        rendered.append(output_dir / f"level{level}_registered_lr_fullAffine_centerZ_xy_yellowOverlay_ch{channel}.png")
    return rendered
