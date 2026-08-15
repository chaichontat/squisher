from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw, ImageFont

from squisher_deconv.planning import output_sidecar_path

DEFAULT_Z_PLANES = (200, 1000, 2000, 3000)
GUTTER = 32
TOP_MARGIN = 144
ROW_TITLE_HEIGHT = 96
TITLE_FONT_SIZE = 52
LABEL_FONT_SIZE = 40


def render_before_after_qc(
    *,
    raw_dir: Path,
    deconv_dir: Path | None = None,
    qc_dir: Path | None = None,
    image_prefix: str | None = None,
    z_planes: list[int] | None = None,
    tile_count: int = 5,
    channel: int = 0,
) -> list[dict[str, object]]:
    if tile_count <= 0:
        raise ValueError(f"tile_count must be positive, got {tile_count}")
    if channel < 0:
        raise ValueError(f"channel must be non-negative, got {channel}")
    resolved_z_planes = list(DEFAULT_Z_PLANES if z_planes is None else z_planes)
    if not resolved_z_planes:
        raise ValueError("At least one z plane is required")

    resolved_deconv_dir = deconv_dir or raw_dir / "squisher-deconv-run-u16"
    resolved_qc_dir = qc_dir or raw_dir / "squisher-deconv-run-u16-qc"
    prefix = infer_image_prefix(raw_dir, image_prefix)
    manifest = [
        render_tile(
            tile,
            raw_dir=raw_dir,
            deconv_dir=resolved_deconv_dir,
            qc_dir=resolved_qc_dir,
            prefix=prefix,
            z_planes=resolved_z_planes,
            channel=channel,
        )
        for tile in finished_tiles(resolved_deconv_dir, prefix, tile_count)
    ]
    manifest_path = resolved_qc_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def read_page(tif: tifffile.TiffFile, z: int, *, channel: int) -> np.ndarray:
    return tif.pages[page_index_for_channel_z(tif, z, channel=channel)].asarray()


def page_index_for_channel_z(tif: tifffile.TiffFile, z: int, *, channel: int) -> int:
    path = Path(tif.filehandle.path)
    axes = tif.series[0].axes
    shape = tuple(map(int, tif.series[0].shape))
    if axes == "CZYX":
        channels, z_count, _, _ = shape
        if channel >= channels:
            raise ValueError(f"{path} has {channels} channel(s), cannot read channel {channel}")
        index = channel * z_count + z
    elif axes == "ZYX":
        channels = decon_channel_count(path)
        if channel >= channels:
            raise ValueError(f"{path} has {channels} channel(s), cannot read channel {channel}")
        index = z * channels if channels > 1 else z
        if channels > 1:
            index += channel
    else:
        raise ValueError(f"{path} axes must be CZYX or ZYX, got {axes}")
    if index >= len(tif.pages):
        raise ValueError(f"{path} has {len(tif.pages)} pages, cannot read page {index}")
    return index


def decon_channel_count(path: Path) -> int:
    sidecar = output_sidecar_path(path) if path.name.endswith(".ome.zarr") else path.with_suffix(".deconv.json")
    if not sidecar.exists():
        return 1
    payload = json.loads(sidecar.read_text())
    try:
        channels = int(payload["provenance"]["run_settings"]["channels"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{sidecar} is missing a valid provenance.run_settings.channels value") from error
    if channels <= 0:
        raise ValueError(f"{sidecar} records invalid channel count {channels}")
    return channels


def stretch(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, [0.5, 99.8])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    stretched = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(stretched * 255).astype(np.uint8)


def infer_image_prefix(raw_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    first = next(iter(sorted(raw_dir.glob("*.ome.tif"))), None)
    if first is None:
        raise FileNotFoundError(f"No *.ome.tif files found in {raw_dir}")
    return first.name.split(".")[0]


def finished_tiles(deconv_dir: Path, prefix: str, tile_count: int) -> list[str]:
    tiles = sorted(
        path.name.removeprefix(f"{prefix}.").removesuffix(".ome.zarr")
        for path in deconv_dir.glob(f"{prefix}.*.ome.zarr")
        if output_sidecar_path(path).exists()
    )
    if len(tiles) < tile_count:
        raise RuntimeError(f"Only found {len(tiles)} finished tile(s) in {deconv_dir}")
    indexes = np.linspace(0, len(tiles) - 1, tile_count, dtype=int)
    return [tiles[int(index)] for index in indexes]


def read_ome_zarr_plane(path: Path, z: int, *, channel: int) -> np.ndarray:
    array = zarr.open_group(str(path), mode="r")["0"]
    axes = list(getattr(array.metadata, "dimension_names", None) or array.attrs.get("_ARRAY_DIMENSIONS", []))
    if axes != ["c", "z", "y", "x"]:
        raise ValueError(f"{path}/0 axes must be ['c', 'z', 'y', 'x'], got {axes}")
    channels, z_count, _, _ = map(int, array.shape)
    if channel >= channels:
        raise ValueError(f"{path} has {channels} channel(s), cannot read channel {channel}")
    if z >= z_count:
        raise ValueError(f"{path} has {z_count} z plane(s), cannot read z={z}")
    return np.asarray(array[channel, z, :, :])


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_tile(
    tile: str,
    *,
    raw_dir: Path,
    deconv_dir: Path,
    qc_dir: Path,
    prefix: str,
    z_planes: list[int],
    channel: int,
) -> dict[str, object]:
    raw_path = raw_dir / f"{prefix}.{tile}.ome.tif"
    deconv_path = deconv_dir / f"{prefix}.{tile}.ome.zarr"
    if not raw_path.exists() or not deconv_path.exists():
        raise FileNotFoundError(f"Missing raw/deconv pair for tile {tile}")

    with tifffile.TiffFile(raw_path) as raw_tif:
        deconv_planes = [(z, read_ome_zarr_plane(deconv_path, z, channel=channel)) for z in z_planes]
        stats = [
            {"z": z, "min": int(plane.min()), "max": int(plane.max()), "mean": float(plane.mean())}
            for z, plane in deconv_planes
        ]
        if any(item["max"] == 0 for item in stats):
            raise RuntimeError(f"{deconv_path} still has a zero rendered z plane: {stats}")

        first = read_page(raw_tif, z_planes[0], channel=channel)
        height, width = first.shape
        canvas_width = width * 2 + GUTTER
        canvas_height = TOP_MARGIN + len(z_planes) * (ROW_TITLE_HEIGHT + height)
        canvas = Image.new("L", (canvas_width, canvas_height), color=255)
        draw = ImageDraw.Draw(canvas)
        title_font = load_font(TITLE_FONT_SIZE)
        label_font = load_font(LABEL_FONT_SIZE)
        draw.text(
            (8, 16),
            f"{deconv_path.name} ch{channel}: raw vs corrected exact-backend deconv",
            fill=0,
            font=title_font,
        )
        for row, (z, deconv) in enumerate(deconv_planes):
            raw = first if row == 0 else read_page(raw_tif, z, channel=channel)
            y0 = TOP_MARGIN + row * (ROW_TITLE_HEIGHT + height)
            panels = [(raw, "raw", 0), (deconv, "deconv u16", width + GUTTER)]
            for image, label, x0 in panels:
                draw.text(
                    (x0, y0 + 8),
                    f"z={z} {label}  min={int(image.min())} max={int(image.max())}",
                    fill=0,
                    font=label_font,
                )
                panel = Image.fromarray(stretch(image), mode="L")
                canvas.paste(panel, (x0, y0 + ROW_TITLE_HEIGHT))

    qc_dir.mkdir(parents=True, exist_ok=True)
    out = qc_dir / f"{deconv_path.stem}.before-after.png"
    canvas.save(out)
    return {
        "tile": tile,
        "png": str(out),
        "layout": {"columns": ["raw", "deconv_u16"], "z_planes": z_planes, "tile_shape": [height, width]},
        "image_shape": [canvas_height, canvas_width],
        "native_panel_pixels": [height, width],
        "deconv_stats": stats,
    }
