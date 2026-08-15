from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from squisher.jpegxr_zarr import register_jpegxr_codec
from tifffile import TiffFile, imread, imwrite

from squisher_lightsheet.geometry import orient_plane_yx, signed_bounds
from squisher_lightsheet import ngff


OME_NS = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
SPATIAL_AXES = ("z", "y", "x")


@dataclass(frozen=True)
class TileMetadata:
    path: Path
    name: str
    axes: str
    shape: tuple[int, ...]
    spacing_um_zyx: tuple[float, float, float]
    translation_um_zyx: tuple[float, float, float]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        if self.axes == "CZYX":
            return (int(self.shape[1]), int(self.shape[2]), int(self.shape[3]))
        if self.axes == "ZYX":
            return (int(self.shape[0]), int(self.shape[1]), int(self.shape[2]))
        raise ValueError(f"{self.path} has unsupported axes {self.axes!r}; expected CZYX or ZYX")

    @property
    def channel_count(self) -> int:
        if self.axes == "CZYX":
            return int(self.shape[0])
        if self.axes == "ZYX":
            return 1
        raise ValueError(f"{self.path} has unsupported axes {self.axes!r}; expected CZYX or ZYX")


@dataclass(frozen=True)
class BasicProfile:
    flatfield: np.ndarray
    darkfield: np.ndarray | None
    flatfield_path: Path
    darkfield_path: Path | None


@dataclass(frozen=True)
class DumbStitchResult:
    manifest_path: Path
    contact_sheet_path: Path
    output_paths: list[Path]


def parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not channels or any(channel < 0 for channel in channels):
        raise ValueError(f"Expected comma-separated non-negative channel indexes, got {value!r}")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Duplicate channel in {value!r}")
    return channels


def parse_view_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected VIEW=DIR, got {value!r}")
    view, path = value.split("=", 1)
    view = view.strip()
    if not view:
        raise ValueError(f"Missing view name in {value!r}")
    if not path:
        raise ValueError(f"Missing directory in {value!r}")
    return view, Path(path).expanduser()


def tile_paths_from_dir(input_dir: Path) -> list[Path]:
    paths = [
        *sorted(input_dir.glob("*.ome.tif")),
        *sorted(path for path in input_dir.glob("*.ome.zarr") if path.is_dir()),
    ]
    if not paths:
        raise FileNotFoundError(f"No *.ome.tif or *.ome.zarr tiles found in {input_dir}")
    return paths


def _float_attr(element: ET.Element, name: str, default: float = 0.0) -> float:
    value = element.attrib.get(name)
    return default if value is None else float(value)


def _ome_tiff_metadata(path: Path) -> TileMetadata:
    with TiffFile(path) as tif:
        if tif.ome_metadata is None:
            raise ValueError(f"{path} is missing OME metadata")
        root = ET.fromstring(tif.ome_metadata)
        pixels = root.find(f".//{OME_NS}Pixels")
        if pixels is None:
            raise ValueError(f"{path} OME metadata is missing Pixels")
        size_t = int(pixels.attrib["SizeT"])
        size_c = int(pixels.attrib["SizeC"])
        size_z = int(pixels.attrib["SizeZ"])
        size_y = int(pixels.attrib["SizeY"])
        size_x = int(pixels.attrib["SizeX"])
        if size_t != 1:
            raise ValueError(f"{path} has unsupported SizeT={size_t}; expected a single timepoint")
        axes = "CZYX" if size_c > 1 else "ZYX"
        shape = (size_c, size_z, size_y, size_x) if size_c > 1 else (size_z, size_y, size_x)
        plane = pixels.find(f"{OME_NS}Plane")
        if plane is None:
            raise ValueError(f"{path} OME metadata is missing Plane position metadata")
        return TileMetadata(
            path=path,
            name=path.name,
            axes=axes,
            shape=shape,
            spacing_um_zyx=(
                _float_attr(pixels, "PhysicalSizeZ", 1.0),
                _float_attr(pixels, "PhysicalSizeY", 1.0),
                _float_attr(pixels, "PhysicalSizeX", 1.0),
            ),
            translation_um_zyx=(
                _float_attr(plane, "PositionZ", 0.0),
                _float_attr(plane, "PositionY", 0.0),
                _float_attr(plane, "PositionX", 0.0),
            ),
        )


def _ome_zarr_metadata(path: Path, *, level: int) -> TileMetadata:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    names, scale, translation_values, _has_scale, _has_translation = ngff.scale_translation(
        root, dataset_index=level
    )
    array = ngff.level_array(root, level=level, context=path)
    scale_by_axis = dict(zip(names, scale, strict=True))
    translation_by_axis = dict(zip(names, translation_values, strict=True))
    try:
        spacing = tuple(scale_by_axis[axis] for axis in SPATIAL_AXES)
        translation = tuple(translation_by_axis[axis] for axis in SPATIAL_AXES)
    except KeyError as error:
        raise ValueError(f"OME-Zarr axes must include z, y, and x; found {names}") from error
    return TileMetadata(
        path=path,
        name=path.name,
        axes="".join(name.upper() for name in names),
        shape=tuple(int(value) for value in array.shape),
        spacing_um_zyx=spacing,
        translation_um_zyx=translation,
    )


def read_tile_metadata(path: Path, *, level: int = 0) -> TileMetadata:
    if path.name.endswith(".ome.zarr"):
        return _ome_zarr_metadata(path, level=level)
    if level != 0:
        raise ValueError(f"{path} is OME-TIFF; pyramid level selection requires OME-Zarr input")
    return _ome_tiff_metadata(path)


def _read_planes(
    tile: TileMetadata, *, channels: tuple[int, ...], z_index: int, level: int = 0
) -> dict[int, np.ndarray]:
    if tile.path.name.endswith(".ome.zarr"):
        import zarr

        root = zarr.open_group(str(tile.path), mode="r")
        array = ngff.level_array(root, level=level, context=tile.path)
        if tile.axes == "CZYX":
            return {channel: np.asarray(array[channel, z_index], dtype=np.float32) for channel in channels}
        if tile.axes == "ZYX":
            if channels != (0,):
                raise ValueError(f"Channels {channels} out of range for single-channel tile {tile.path}")
            return {0: np.asarray(array[z_index], dtype=np.float32)}
    with TiffFile(tile.path) as tif:
        if tile.axes == "CZYX":
            z_count = tile.shape_zyx[0]
            return {
                channel: np.asarray(tif.pages[channel * z_count + z_index].asarray(), dtype=np.float32)
                for channel in channels
            }
        if tile.axes == "ZYX":
            if channels != (0,):
                raise ValueError(f"Channels {channels} out of range for single-channel tile {tile.path}")
            return {0: np.asarray(tif.pages[z_index].asarray(), dtype=np.float32)}
    raise ValueError(f"{tile.path} has unsupported axes {tile.axes!r}; expected CZYX or ZYX")


def _find_profile_path(basic_dir: Path, channel: int, suffix: str) -> Path | None:
    matches = sorted(basic_dir.glob(f"*-ch{channel}-{suffix}.tif"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one ch{channel} {suffix} TIFF in {basic_dir}, found {matches}")
    return matches[0]


def load_basic_profile(basic_dir: Path, channel: int) -> BasicProfile:
    flatfield_path = _find_profile_path(basic_dir, channel, "flatfield")
    if flatfield_path is None:
        raise FileNotFoundError(f"No *-ch{channel}-flatfield.tif found in {basic_dir}")
    darkfield_path = _find_profile_path(basic_dir, channel, "darkfield")
    flatfield = np.asarray(imread(flatfield_path), dtype=np.float32)
    darkfield = None if darkfield_path is None else np.asarray(imread(darkfield_path), dtype=np.float32)
    if flatfield.ndim != 2 or not np.all(np.isfinite(flatfield)) or np.any(flatfield <= 0):
        raise ValueError(f"{flatfield_path} must be a positive finite 2D flatfield")
    if darkfield is not None and darkfield.shape != flatfield.shape:
        raise ValueError(
            f"{darkfield_path} shape {darkfield.shape} does not match flatfield {flatfield.shape}"
        )
    return BasicProfile(
        flatfield=flatfield, darkfield=darkfield, flatfield_path=flatfield_path, darkfield_path=darkfield_path
    )


def apply_basic(plane: np.ndarray, profile: BasicProfile) -> np.ndarray:
    if profile.flatfield.shape != plane.shape:
        raise ValueError(
            f"BaSiC profile shape {profile.flatfield.shape} does not match tile plane {plane.shape}"
        )
    corrected = plane.astype(np.float32, copy=False)
    if profile.darkfield is not None:
        corrected = corrected - profile.darkfield
    return corrected / profile.flatfield


def canvas_for_tiles(
    tiles: list[TileMetadata],
) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float], tuple[float, float]]:
    if not tiles:
        raise ValueError("No tiles to render")
    pixel_um_yx = np.abs(tiles[0].spacing_um_zyx[1:])
    bounds_min = np.full(2, np.inf, dtype=np.float64)
    bounds_max = np.full(2, -np.inf, dtype=np.float64)
    for tile in tiles:
        tile_pixel_um_yx = np.abs(tile.spacing_um_zyx[1:])
        if not np.allclose(tile_pixel_um_yx, pixel_um_yx, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"Inconsistent absolute YX spacing in {tile.path}: {tile_pixel_um_yx} vs {pixel_um_yx}"
            )
        tile_min, tile_max = signed_bounds(
            np.asarray(tile.translation_um_zyx[1:], dtype=np.float64),
            np.asarray(tile.spacing_um_zyx[1:], dtype=np.float64),
            np.asarray(tile.shape_zyx[1:], dtype=np.float64),
        )
        bounds_min = np.minimum(bounds_min, tile_min)
        bounds_max = np.maximum(bounds_max, tile_max)
    shape_yx = tuple(np.ceil((bounds_max - bounds_min) / np.asarray(pixel_um_yx)).astype(int).tolist())
    return (
        shape_yx,
        tuple(float(v) for v in pixel_um_yx),
        tuple(float(v) for v in bounds_min),
        tuple(float(v) for v in bounds_max),
    )


def stretch_uint8(images: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[float, float]]:
    values = [
        image[np.isfinite(image) & (image > 0)].ravel()
        for image in images
        if np.any(np.isfinite(image) & (image > 0))
    ]
    low, high = np.percentile(np.concatenate(values), [0.5, 99.8]) if values else (0.0, 1.0)
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    stretched = [
        np.clip((image - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8) for image in images
    ]
    return stretched, (float(low), float(high))


def _annotation_suffix(*, draw_tile_labels: bool, draw_tile_outlines: bool) -> str:
    if not draw_tile_labels and not draw_tile_outlines:
        return "_noLines"
    parts = []
    if draw_tile_labels:
        parts.append("labels")
    if draw_tile_outlines:
        parts.append("outlines")
    return "_" + "_".join(parts)


def _tile_number(name: str) -> str:
    return name.removesuffix(".ome.zarr").removesuffix(".ome.tif").rsplit(".", 1)[-1]


def annotate_tiles(
    image: np.ndarray,
    placements: list[dict[str, object]],
    *,
    draw_tile_labels: bool,
    draw_tile_outlines: bool,
) -> np.ndarray:
    if not draw_tile_labels and not draw_tile_outlines:
        return image
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for placement in placements:
        x0 = int(placement["x0"])
        y0 = int(placement["y0"])
        height, width = (int(v) for v in placement["shape_yx"])
        if draw_tile_outlines:
            draw.rectangle([x0, y0, x0 + width - 1, y0 + height - 1], outline=(255, 255, 255), width=1)
        if draw_tile_labels:
            draw.text((x0 + 5, y0 + 5), _tile_number(str(placement["tile"])), fill=(255, 255, 255))
    return np.asarray(pil)


def _write_contact_sheet(items: list[tuple[str, Path]], output: Path) -> None:
    if not items:
        raise ValueError("No RGB images were generated for the contact sheet; pass at least two channels")
    thumbs = []
    for label, path in items:
        image = Image.open(path).convert("RGB")
        scale = min(1.0, 1100 / max(1, image.width))
        if scale < 1.0:
            image = image.resize(
                (int(round(image.width * scale)), int(round(image.height * scale))), Image.Resampling.BILINEAR
            )
        thumbs.append((label, image))
    columns = min(2, len(thumbs))
    rows = int(math.ceil(len(thumbs) / columns))
    cell_width = max(image.width for _label, image in thumbs)
    cell_height = max(image.height for _label, image in thumbs) + 28
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(thumbs):
        row, column = divmod(index, columns)
        x = column * cell_width
        y = row * cell_height
        draw.text((x + 5, y + 5), label, fill=(0, 0, 0))
        sheet.paste(image, (x, y + 28))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _paste(mosaic: np.ndarray, plane: np.ndarray, *, y0: int, x0: int) -> None:
    y1 = min(mosaic.shape[0], y0 + plane.shape[0])
    x1 = min(mosaic.shape[1], x0 + plane.shape[1])
    if y1 > y0 and x1 > x0:
        mosaic[y0:y1, x0:x1] = plane[: y1 - y0, : x1 - x0]


def _exact_uint16(values: np.ndarray) -> np.ndarray:
    """Convert native intensities without silently clipping, rounding, or rescaling."""
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or np.any(values > np.iinfo(np.uint16).max)
        or not np.array_equal(values, np.rint(values))
    ):
        raise ValueError("raw dumb-stitch intensities cannot be represented exactly as uint16")
    return values.astype(np.uint16)


def _render_view(
    *,
    view: str,
    input_dir: Path,
    output_dir: Path,
    channels: tuple[int, ...],
    basic_dir: Path | None,
    level: int,
    center_z_index: int | None,
    draw_tile_labels: bool,
    draw_tile_outlines: bool,
    write_tiff: bool,
    output_prefix: str,
    progress,
) -> tuple[dict[str, object], list[tuple[str, Path]]]:
    tiles = [read_tile_metadata(path, level=level) for path in tile_paths_from_dir(input_dir)]
    shape_yx, pixel_um_yx, bounds_min_yx, bounds_max_yx = canvas_for_tiles(tiles)
    for tile in tiles:
        for channel in channels:
            if channel >= tile.channel_count:
                raise ValueError(
                    f"Channel {channel} out of range for {tile.path} with {tile.channel_count} channel(s)"
                )
    cases = ["raw", *(["basic"] if basic_dir is not None else [])]
    profiles = (
        {channel: load_basic_profile(basic_dir, channel) for channel in channels}
        if basic_dir is not None
        else {}
    )
    mosaics = {
        (case, channel): np.zeros(shape_yx, dtype=np.float32) for case in cases for channel in channels
    }
    placements: list[dict[str, object]] = []
    for index, tile in enumerate(tiles):
        z_count = tile.shape_zyx[0]
        z_index = z_count // 2 if center_z_index is None else center_z_index
        if z_index < 0 or z_index >= z_count:
            raise ValueError(f"Center z {z_index} is outside {tile.path} z_count={z_count}")
        tile_min_yx, _ = signed_bounds(
            np.asarray(tile.translation_um_zyx[1:], dtype=np.float64),
            np.asarray(tile.spacing_um_zyx[1:], dtype=np.float64),
            np.asarray(tile.shape_zyx[1:], dtype=np.float64),
        )
        y0 = int(round((tile_min_yx[0] - bounds_min_yx[0]) / pixel_um_yx[0]))
        x0 = int(round((tile_min_yx[1] - bounds_min_yx[1]) / pixel_um_yx[1]))
        raw_planes = _read_planes(tile, channels=channels, z_index=z_index, level=level)
        for channel in channels:
            raw_plane = raw_planes[channel]
            plane = orient_plane_yx(raw_plane, tile.spacing_um_zyx[1:])
            _paste(mosaics[("raw", channel)], plane, y0=y0, x0=x0)
            if basic_dir is not None:
                corrected = orient_plane_yx(
                    apply_basic(raw_plane, profiles[channel]), tile.spacing_um_zyx[1:]
                )
                _paste(mosaics[("basic", channel)], corrected, y0=y0, x0=x0)
        placements.append(
            {
                "tile": tile.name,
                "path": str(tile.path),
                "center_z_index": int(z_index),
                "x0": int(x0),
                "y0": int(y0),
                "shape_yx": [int(tile.shape_zyx[1]), int(tile.shape_zyx[2])],
                "translation_um_zyx": [float(v) for v in tile.translation_um_zyx],
            }
        )
        if progress is not None and index % 5 == 0:
            progress(f"{view}: placed {index}/{len(tiles)}")
    outputs: dict[str, str] = {}
    display_ranges = {}
    contact_items: list[tuple[str, Path]] = []
    suffix = _annotation_suffix(draw_tile_labels=draw_tile_labels, draw_tile_outlines=draw_tile_outlines)
    for case in cases:
        stretched, display_range = stretch_uint8([mosaics[(case, channel)] for channel in channels])
        display_ranges[case] = list(display_range)
        for channel, image in zip(channels, stretched, strict=True):
            if write_tiff:
                tiff_path = (
                    output_dir
                    / f"{output_prefix}-{view}_{case}_ch{channel}_omeMetadata_noBlend{suffix}.ome.tif"
                )
                imwrite(
                    tiff_path,
                    (
                        _exact_uint16(mosaics[(case, channel)])
                        if case == "raw"
                        else mosaics[(case, channel)]
                    ),
                    ome=True,
                    photometric="minisblack",
                    tile=(256, 256),
                    compression="zlib",
                    metadata={
                        "axes": "YX",
                        "PhysicalSizeY": pixel_um_yx[0],
                        "PhysicalSizeYUnit": "µm",
                        "PhysicalSizeX": pixel_um_yx[1],
                        "PhysicalSizeXUnit": "µm",
                    },
                )
                outputs[f"{case}_ch{channel}_tiff"] = str(tiff_path)
            image = annotate_tiles(
                image, placements, draw_tile_labels=draw_tile_labels, draw_tile_outlines=draw_tile_outlines
            )
            path = output_dir / f"{output_prefix}-{view}_{case}_ch{channel}_omeMetadata_noBlend{suffix}.png"
            Image.fromarray(image).save(path)
            outputs[f"{case}_ch{channel}"] = str(path)
            if len(channels) == 1:
                contact_items.append((f"{view} {case} ch{channel}", path))
        if len(channels) >= 2:
            rgb = np.zeros((*shape_yx, 3), dtype=np.uint8)
            rgb[..., 1] = stretched[0]
            rgb[..., 0] = stretched[1]
            rgb = annotate_tiles(
                rgb, placements, draw_tile_labels=draw_tile_labels, draw_tile_outlines=draw_tile_outlines
            )
            path = (
                output_dir
                / f"{output_prefix}-{view}_{case}_ch{channels[0]}green_ch{channels[1]}red_omeMetadata_noBlend{suffix}.png"
            )
            Image.fromarray(rgb).save(path)
            outputs[f"{case}_rgb"] = str(path)
            contact_items.append((f"{view} {case}", path))
    return (
        {
            "view": view,
            "input_dir": str(input_dir),
            "level": int(level),
            "tile_count": len(tiles),
            "shape_yx_px": [int(shape_yx[0]), int(shape_yx[1])],
            "pixel_um_yx": [float(v) for v in pixel_um_yx],
            "bounds_yx_um": {
                "min": [float(v) for v in bounds_min_yx],
                "max": [float(v) for v in bounds_max_yx],
            },
            "placements": placements,
            "display_ranges": display_ranges,
            "outputs": outputs,
        },
        contact_items,
    )


def render_ome_metadata_dumb_stitch(
    *,
    input_dirs_by_view: dict[str, Path],
    output_dir: Path,
    channels: tuple[int, ...],
    basic_dir: Path | None = None,
    level: int = 0,
    center_z_index: int | None = None,
    output_prefix: str = "ome_metadata_dumb_stitch",
    draw_tile_labels: bool = False,
    draw_tile_outlines: bool = False,
    write_tiff: bool = False,
    progress=None,
) -> DumbStitchResult:
    register_jpegxr_codec()
    if not input_dirs_by_view:
        raise ValueError("At least one input view is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    views = []
    contact_items: list[tuple[str, Path]] = []
    output_paths: list[Path] = []
    for view, input_dir in input_dirs_by_view.items():
        summary, view_contact_items = _render_view(
            view=view,
            input_dir=input_dir,
            output_dir=output_dir,
            channels=channels,
            basic_dir=basic_dir,
            level=level,
            center_z_index=center_z_index,
            draw_tile_labels=draw_tile_labels,
            draw_tile_outlines=draw_tile_outlines,
            write_tiff=write_tiff,
            output_prefix=output_prefix,
            progress=progress,
        )
        views.append(summary)
        contact_items.extend(view_contact_items)
        output_paths.extend(Path(path) for path in summary["outputs"].values())
    suffix = _annotation_suffix(draw_tile_labels=draw_tile_labels, draw_tile_outlines=draw_tile_outlines)
    contact_sheet_path = output_dir / f"{output_prefix}_omeMetadata_noBlend{suffix}_contact_sheet.png"
    _write_contact_sheet(contact_items, contact_sheet_path)
    manifest_path = output_dir / f"{output_prefix}_omeMetadata_noBlend{suffix}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.ome_metadata_dumb_stitch.v1",
                "created_at_unix": time.time(),
                "metadata_source": (
                    "OME physical metadata: OME-TIFF Plane PositionX/Y/Z plus PhysicalSizeX/Y/Z, "
                    "or OME-Zarr multiscales scale/translation transforms"
                ),
                "channels": [int(channel) for channel in channels],
                "basic_dir": None if basic_dir is None else str(basic_dir),
                "level": int(level),
                "center_z_index": center_z_index,
                "draw_tile_labels": bool(draw_tile_labels),
                "draw_tile_outlines": bool(draw_tile_outlines),
                "write_tiff": bool(write_tiff),
                "views": views,
                "contact_sheet": str(contact_sheet_path),
            },
            indent=2,
        )
        + "\n"
    )
    return DumbStitchResult(
        manifest_path=manifest_path,
        contact_sheet_path=contact_sheet_path,
        output_paths=[*output_paths, contact_sheet_path, manifest_path],
    )
