from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import random

import imagecodecs
from loguru import logger
import numpy as np
import tifffile

from squisher.compression import _czi_tiles


JPEG_XR_COMPRESSION = 22610


@dataclass(frozen=True)
class CropSelection:
    crop_index: int
    tile_index: int
    mosaic_index: int | None
    t: int
    channel: int
    z: int
    y: int
    x: int
    height: int
    width: int


def compare_czi_compression(
    path: Path,
    *,
    out_dir: Path | None = None,
    count: int = 6,
    crop_size: int = 256,
    min_level: float = 0.65,
    max_level: float = 0.90,
    level_step: float = 0.05,
    seed: int = 20260604,
    tile_size: int = 256,
    keep_encoded_tiffs: bool = True,
    t: int | None = None,
    channel: int | None = None,
    z: int | None = None,
    max_attempts: int = 500,
) -> Path:
    from aicspylibczi import CziFile

    if path.suffix.lower() != ".czi":
        raise ValueError(f"Expected a .czi input file, got {path}")
    if count <= 0:
        raise ValueError(f"Crop count must be positive, got {count}")
    if crop_size <= 0:
        raise ValueError(f"Crop size must be positive, got {crop_size}")
    if tile_size % 16 != 0:
        raise ValueError(f"TIFF tile size must be a multiple of 16, got {tile_size}")
    effective_tile_size = min(tile_size, crop_size)
    if effective_tile_size % 16 != 0:
        raise ValueError(f"Effective TIFF tile size must be a multiple of 16, got {effective_tile_size}")
    if max_attempts < count:
        raise ValueError(f"max_attempts must be at least count ({count}), got {max_attempts}")

    reader = CziFile(path)
    dims = reader.get_dims_shape()[0]
    t_indexes = _selected_indexes(dims, "T", t)
    channels = _selected_indexes(dims, "C", channel)
    z_indexes = _selected_indexes(dims, "Z", z)
    tiles = _czi_tiles(path)
    tile_count = len(tiles)
    levels = compression_levels(min_level=min_level, max_level=max_level, level_step=level_step)
    out_dir = comparison_output_dir(path, out_dir=out_dir, crop_size=crop_size, tile_count=tile_count)
    encoded_dir = out_dir / "encoded_tiffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir.mkdir(exist_ok=True)
    logger.info(
        "Comparing JPEG-XR compression for {} crop(s) from {} at levels {} into {}",
        count,
        path,
        levels,
        out_dir,
    )
    selections = random_crop_selections(
        tiles=tiles,
        t_indexes=t_indexes,
        channels=channels,
        z_indexes=z_indexes,
        count=count,
        crop_size=crop_size,
        seed=seed,
        max_attempts=max_attempts,
    )

    records = []
    for selection in selections:
        raw = _read_czi_crop(reader, selection, crop_size=crop_size)
        raw_crop_bytes = raw.nbytes
        decoded_by_level: dict[float, np.ndarray] = {}
        level_records = []
        crop_stem = comparison_crop_stem(selection, tile_count=tile_count)
        for level in levels:
            level_tag = f"{int(round(level * 100)):02d}"
            tiff_path = encoded_dir / f"{crop_stem}_level{level_tag}.tif"
            decoded = write_jpeg_xr_tiff_crop(
                tiff_path,
                raw,
                level=level,
                tile_size=min(tile_size, crop_size),
            )
            decoded_by_level[level] = decoded
            diff = decoded.astype(np.float32) - raw.astype(np.float32)
            encoded_bytes = tiff_path.stat().st_size
            level_records.append(
                {
                    "level": level,
                    "encoded_tiff": str(tiff_path.relative_to(out_dir)) if keep_encoded_tiffs else None,
                    "encoded_bytes": encoded_bytes,
                    "raw_bytes": raw_crop_bytes,
                    "raw_dtype": str(raw.dtype),
                    "ratio_raw_to_encoded": raw_crop_bytes / encoded_bytes,
                    "encoded_fraction_of_raw": encoded_bytes / raw_crop_bytes,
                    "saved_fraction_vs_raw": 1.0 - encoded_bytes / raw_crop_bytes,
                    "mae": float(np.mean(np.abs(diff))),
                    "rmse": float(np.sqrt(np.mean(diff * diff))),
                }
            )
            if not keep_encoded_tiffs:
                tiff_path.unlink()

        figure, window, diff_limit = render_sweep_figure(
            raw,
            decoded_by_level=decoded_by_level,
            level_records=level_records,
            crop_label=(
                f"CROP {selection.crop_index:02d} TILE {selection.tile_index:03d} "
                f"T{selection.t} C{selection.channel} Z{selection.z:04d} Y{selection.y:04d} X{selection.x:04d}"
            ),
        )
        figure_path = out_dir / f"{crop_stem}_sweep.png"
        imagecodecs.imwrite(figure_path, figure)
        logger.info("Wrote comparison figure {}", figure_path.name)
        records.append(
            {
                "png": figure_path.name,
                "tile_index": selection.tile_index,
                "mosaic_index": selection.mosaic_index,
                "t": selection.t,
                "channel": selection.channel,
                "z": selection.z,
                "y": selection.y,
                "x": selection.x,
                "raw_source": str(path),
                "window_lo": window[0],
                "window_hi": window[1],
                "diff_abs_p995_limit": diff_limit,
                "levels": level_records,
            }
        )

    manifest = {
        "description": (
            "Each PNG has two aligned rows. The top row is raw CZI followed by decoded JPEG-XR crops. "
            "The bottom row is the corresponding compressed-minus-raw diff. Each level title includes "
            "encoded size, raw/encoded ratio, MAE, and RMSE."
        ),
        "levels": levels,
        "seed": seed,
        "crop_size": crop_size,
        "raw_dtypes": sorted({record["levels"][0]["raw_dtype"] for record in records}),
        "count": len(records),
        "records": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_size_metrics_csv(out_dir / "size_metrics.csv", records)
    logger.success("Wrote {} comparison figure(s) to {}", len(records), out_dir)
    logger.info("Wrote manifest to {}", out_dir / "manifest.json")
    logger.info("Wrote size metrics to {}", out_dir / "size_metrics.csv")
    return out_dir


def comparison_crop_stem(selection: CropSelection, *, tile_count: int) -> str:
    tile = f"_tile{selection.tile_index:03d}" if tile_count > 1 else ""
    return (
        f"crop_{selection.crop_index:02d}{tile}"
        f"_t{selection.t}_c{selection.channel}_z{selection.z:04d}_y{selection.y:04d}_x{selection.x:04d}"
    )


def comparison_output_dir(path: Path, *, out_dir: Path | None, crop_size: int, tile_count: int) -> Path:
    if tile_count <= 1:
        return out_dir or path.with_name(f"{path.stem}_compression_sweep_{crop_size}_crops")
    if out_dir is None:
        return path.with_name(path.stem)
    return out_dir if out_dir.name == path.stem else out_dir / path.stem


def compression_levels(*, min_level: float, max_level: float, level_step: float) -> list[float]:
    if min_level < 0 or max_level > 100:
        raise ValueError(f"Compression levels must be in [0, 100], got {min_level}..{max_level}")
    if max_level < min_level:
        raise ValueError(f"max_level must be >= min_level, got {max_level} < {min_level}")
    if level_step <= 0:
        raise ValueError(f"level_step must be positive, got {level_step}")

    levels = []
    current = min_level
    epsilon = level_step / 1_000
    while current <= max_level + epsilon:
        levels.append(round(current, 10))
        current += level_step
    return levels


def random_crop_selections(
    *,
    tiles: list[dict],
    t_indexes: list[int],
    channels: list[int],
    z_indexes: list[int],
    count: int,
    crop_size: int,
    seed: int,
    max_attempts: int,
) -> list[CropSelection]:
    rng = random.Random(seed)
    selections = []
    seen = set()
    candidates = [tile for tile in tiles if tile["height"] >= crop_size and tile["width"] >= crop_size]
    if not candidates:
        raise ValueError(f"No CZI tile is large enough for {crop_size}x{crop_size} crops")

    for _attempt in range(max_attempts):
        if len(selections) >= count:
            break
        tile = rng.choice(candidates)
        t = rng.choice(t_indexes)
        channel = rng.choice(channels)
        z = rng.choice(z_indexes)
        y = rng.randint(0, tile["height"] - crop_size)
        x = rng.randint(0, tile["width"] - crop_size)
        key = (tile["index"], t, channel, z // 50, y // crop_size, x // crop_size)
        if key in seen:
            continue
        seen.add(key)
        selections.append(
            CropSelection(
                crop_index=len(selections),
                tile_index=tile["index"],
                mosaic_index=tile.get("mosaic_index"),
                t=t,
                channel=channel,
                z=z,
                y=y,
                x=x,
                height=tile["height"],
                width=tile["width"],
            )
        )

    if len(selections) < count:
        raise ValueError(f"Only selected {len(selections)} crops after {max_attempts} attempts")
    return selections


def write_jpeg_xr_tiff_crop(path: Path, crop: np.ndarray, *, level: float, tile_size: int) -> np.ndarray:
    tifffile.imwrite(
        path,
        crop,
        photometric="minisblack",
        compression=JPEG_XR_COMPRESSION,
        compressionargs={"level": _normalized_compression_level(level)},
        tile=(tile_size, tile_size),
        metadata=None,
    )
    with tifffile.TiffFile(path) as tif:
        return tif.pages[0].asarray()


def render_sweep_figure(
    raw: np.ndarray,
    *,
    decoded_by_level: dict[float, np.ndarray],
    level_records: list[dict],
    crop_label: str,
) -> tuple[np.ndarray, tuple[float, float], float]:
    crop_size = raw.shape[0]
    if raw.shape[0] != raw.shape[1]:
        raise ValueError(f"Expected square crop, got {raw.shape}")

    levels = [record["level"] for record in level_records]
    decoded = [decoded_by_level[level] for level in levels]
    lo, hi = _shared_window([raw, *decoded])
    diffs = [image.astype(np.float32) - raw.astype(np.float32) for image in decoded]
    diff_limit = _diff_limit(diffs)

    gap = 8
    left_label_width = 76
    title_height = 112
    row_label_height = 22
    between_rows = 12
    column_width = max(crop_size, 256)
    cols = 1 + len(levels)
    width = left_label_width + cols * column_width + (cols - 1) * gap
    height = title_height + row_label_height + crop_size + between_rows + row_label_height + crop_size
    figure = np.empty((height, width, 3), dtype=np.uint8)
    figure[...] = np.array([250, 250, 250], dtype=np.uint8)

    col_xs = [left_label_width + index * (column_width + gap) for index in range(cols)]
    raw_bytes = int(level_records[0]["raw_bytes"])
    _draw_multiline(
        figure,
        left_label_width,
        8,
        [crop_label, f"WINDOW {lo:.0f}-{hi:.0f} DIFF P99.5 +/-{diff_limit:.1f}"],
    )
    _draw_multiline(
        figure,
        col_xs[0] + 4,
        54,
        ["RAW", f"{raw.dtype} {raw_bytes / 1024:.1f}KB", "REFERENCE"],
        color=(20, 55, 120),
    )
    for index, record in enumerate(level_records, start=1):
        _draw_multiline(
            figure,
            col_xs[index] + 4,
            54,
            [
                f"L{record['level']:.2f} {record['encoded_bytes'] / 1024:.1f}KB",
                f"RAW/ENC {record['ratio_raw_to_encoded']:.1f}X",
                f"MAE {record['mae']:.2f} RMSE {record['rmse']:.2f}",
            ],
            color=(20, 55, 120),
        )

    top_label_y = title_height
    top_y = top_label_y + row_label_height
    bottom_label_y = top_y + crop_size + between_rows
    bottom_y = bottom_label_y + row_label_height
    _draw_text(figure, 8, top_label_y + 4, "IMAGE")
    _draw_text(figure, 8, bottom_label_y + 4, "DIFF")
    _draw_text(figure, col_xs[0] + 4, bottom_label_y + 4, "REFERENCE", color=(90, 90, 90))

    figure[top_y : top_y + crop_size, col_xs[0] : col_xs[0] + crop_size] = _gray_rgb(_scale_u8(raw, lo, hi))
    figure[bottom_y : bottom_y + crop_size, col_xs[0] : col_xs[0] + crop_size] = 255
    for index, image in enumerate(decoded, start=1):
        x0 = col_xs[index]
        figure[top_y : top_y + crop_size, x0 : x0 + crop_size] = _gray_rgb(_scale_u8(image, lo, hi))
        figure[bottom_y : bottom_y + crop_size, x0 : x0 + crop_size] = _diff_rgb(
            image.astype(np.float32) - raw.astype(np.float32),
            diff_limit,
        )

    _draw_image_borders(figure, col_xs=col_xs, top_y=top_y, bottom_y=bottom_y, crop_size=crop_size)
    return figure, (lo, hi), diff_limit


def write_size_metrics_csv(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "crop",
        "tile_index",
        "t",
        "channel",
        "z",
        "y",
        "x",
        "level",
        "encoded_bytes",
        "raw_bytes",
        "raw_dtype",
        "ratio_raw_to_encoded",
        "encoded_fraction_of_raw",
        "saved_fraction_vs_raw",
        "mae",
        "rmse",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for crop_index, record in enumerate(records):
            for level_record in record["levels"]:
                writer.writerow(
                    {
                        "crop": crop_index,
                        "tile_index": record["tile_index"],
                        "t": record["t"],
                        "channel": record["channel"],
                        "z": record["z"],
                        "y": record["y"],
                        "x": record["x"],
                        "level": level_record["level"],
                        "encoded_bytes": level_record["encoded_bytes"],
                        "raw_bytes": level_record["raw_bytes"],
                        "raw_dtype": level_record["raw_dtype"],
                        "ratio_raw_to_encoded": level_record["ratio_raw_to_encoded"],
                        "encoded_fraction_of_raw": level_record["encoded_fraction_of_raw"],
                        "saved_fraction_vs_raw": level_record["saved_fraction_vs_raw"],
                        "mae": level_record["mae"],
                        "rmse": level_record["rmse"],
                    }
                )


def _read_czi_crop(reader, selection: CropSelection, *, crop_size: int) -> np.ndarray:
    read_kwargs = {"C": selection.channel, "Z": selection.z}
    if "T" in reader.get_dims_shape()[0]:
        read_kwargs["T"] = selection.t
    if selection.mosaic_index is not None:
        read_kwargs["M"] = selection.mosaic_index
    plane = np.squeeze(np.asarray(reader.read_image(**read_kwargs)[0]))
    crop = plane[selection.y : selection.y + crop_size, selection.x : selection.x + crop_size]
    if crop.shape != (crop_size, crop_size):
        raise ValueError(f"Expected crop shape {(crop_size, crop_size)}, got {crop.shape}")
    return crop


def _selected_indexes(dims: dict[str, tuple[int, int]], axis: str, selected: int | None) -> list[int]:
    start, stop = dims.get(axis, (0, 1))
    if selected is None:
        return list(range(start, stop))
    if selected < start or selected >= stop:
        raise ValueError(f"Selected {axis} index {selected} outside available range [{start}, {stop})")
    return [selected]


def _normalized_compression_level(level: float) -> float:
    return level / 100 if level > 1 else level


def _shared_window(arrays: list[np.ndarray]) -> tuple[float, float]:
    combined = np.concatenate([array.astype(np.float32, copy=False).reshape(-1) for array in arrays])
    finite = combined[np.isfinite(combined)]
    nonzero = finite[finite > 0]
    ref = nonzero if nonzero.size else finite
    if not ref.size:
        return 0.0, 1.0
    lo, hi = np.percentile(ref, (0.5, 99.8))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(ref))
        hi = float(np.max(ref))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _scale_u8(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    data = array.astype(np.float32, copy=False)
    return np.clip((data - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _diff_limit(diffs: list[np.ndarray]) -> float:
    values = np.concatenate([np.abs(diff).reshape(-1) for diff in diffs])
    limit = float(np.percentile(values, 99.5)) if values.size else 1.0
    if limit <= 0 or not np.isfinite(limit):
        limit = float(np.max(values)) if values.size else 1.0
    return limit if limit > 0 else 1.0


def _gray_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)


def _diff_rgb(diff: np.ndarray, limit: float) -> np.ndarray:
    norm = np.clip(diff.astype(np.float32, copy=False) / limit, -1, 1)
    rgb = np.zeros((*diff.shape, 3), dtype=np.uint8)
    pos = np.clip(norm, 0, 1)
    neg = np.clip(-norm, 0, 1)
    rgb[..., 0] = (pos * 255).astype(np.uint8)
    rgb[..., 1] = ((1 - np.abs(norm)) * 40).astype(np.uint8)
    rgb[..., 2] = (neg * 255).astype(np.uint8)
    return rgb


def _draw_image_borders(
    figure: np.ndarray,
    *,
    col_xs: list[int],
    top_y: int,
    bottom_y: int,
    crop_size: int,
) -> None:
    for row_y in (top_y, bottom_y):
        for x0 in col_xs:
            figure[row_y - 1 : row_y + crop_size + 1, x0 - 1 : x0] = 0
            figure[row_y - 1 : row_y + crop_size + 1, x0 + crop_size : x0 + crop_size + 1] = 0
            figure[row_y - 1 : row_y, x0 - 1 : x0 + crop_size + 1] = 0
            figure[row_y + crop_size : row_y + crop_size + 1, x0 - 1 : x0 + crop_size + 1] = 0


def _draw_multiline(
    image: np.ndarray,
    x: int,
    y: int,
    lines: list[str],
    *,
    color: tuple[int, int, int] = (0, 0, 0),
    scale: int = 2,
    line_gap: int = 5,
) -> None:
    line_height = 7 * scale + line_gap
    for index, line in enumerate(lines):
        _draw_text(image, x, y + index * line_height, line, color=color, scale=scale)


def _draw_text(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    *,
    color: tuple[int, int, int] = (0, 0, 0),
    scale: int = 2,
) -> None:
    cursor_x = x
    for char in text.upper():
        pattern = FONT.get(char, FONT[" "])
        for row, bits in enumerate(pattern):
            for col, bit in enumerate(bits):
                if bit == "1":
                    y0 = y + row * scale
                    x0 = cursor_x + col * scale
                    if y0 < image.shape[0] and x0 < image.shape[1]:
                        image[y0 : min(y0 + scale, image.shape[0]), x0 : min(x0 + scale, image.shape[1])] = color
        cursor_x += 6 * scale


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "%": ["11001", "11010", "00100", "01000", "01011", "10011", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}
