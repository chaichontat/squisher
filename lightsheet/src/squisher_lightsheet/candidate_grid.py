from __future__ import annotations

import copy
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from squisher_lightsheet.qc import render_registration_qc


DIMENSIONS = ("z", "y", "x")


def render_candidate_grid(
    *,
    position_input: Path,
    registration_input: Path,
    candidate_json: Path,
    output_dir: Path,
    channel: int = 0,
    level: int = 4,
    render_jobs: int = 1,
    crop_fraction_yx: tuple[float, float] = (0.5, 0.5),
    center_y_xz: bool = False,
) -> Path:
    if render_jobs < 1:
        raise ValueError(f"render_jobs must be at least 1, got {render_jobs}")
    if any(value <= 0.0 or value > 1.0 for value in crop_fraction_yx):
        raise ValueError(f"crop_fraction_yx values must be in (0, 1], got {crop_fraction_yx}")

    position_payload = json.loads(position_input.read_text())
    registration_payload = json.loads(registration_input.read_text())
    candidate_spec = json.loads(candidate_json.read_text())
    candidate_tiles = _candidate_tiles(candidate_spec)

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = list(_variant_jobs(candidate_tiles))
    with ThreadPoolExecutor(max_workers=render_jobs) as pool:
        futures = [
            pool.submit(
                _write_and_render_variant,
                position_payload,
                registration_payload,
                output_dir,
                name,
                corrections,
                channel,
                level,
                crop_fraction_yx,
                center_y_xz,
            )
            for name, corrections in jobs
        ]
        records = [future.result() for future in as_completed(futures)]

    records.sort(key=lambda record: record["name"])
    for record in records:
        record["top_left_boundaries"] = str(_draw_boundaries(Path(record["top_left_quadrant"]), level, channel))

    summary_path = output_dir / "candidate_grid_summary.json"
    summary_path.write_text(json.dumps(records, indent=2) + "\n")
    _write_sheet(records, "top_left_quadrant", output_dir / "candidate_grid_sheet.png")
    _write_sheet(records, "top_left_boundaries", output_dir / "candidate_grid_boundaries_sheet.png")
    return summary_path


def _candidate_tiles(candidate_spec: dict[str, Any]) -> list[dict[str, Any]]:
    tiles = candidate_spec["tiles"]
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("Candidate spec must contain a non-empty 'tiles' list")
    for tile in tiles:
        if not isinstance(tile.get("candidates"), dict) or not tile["candidates"]:
            raise ValueError(f"Candidate tile {tile.get('tile', '<missing>')} has no candidates")
        for name, vector in tile["candidates"].items():
            if len(vector) != 3:
                raise ValueError(f"Candidate {tile['tile']}:{name} must have z,y,x correction values")
    return tiles


def _variant_jobs(candidate_tiles: list[dict[str, Any]]) -> list[tuple[str, dict[str, list[float]]]]:
    choices = [list(tile["candidates"].items()) for tile in candidate_tiles]
    jobs = []
    for selected in itertools.product(*choices):
        name_parts = []
        corrections = {}
        for tile, (candidate_name, correction) in zip(candidate_tiles, selected, strict=True):
            label = tile.get("label", _tile_label(tile["tile"]))
            name_parts.append(f"{label}_{candidate_name}")
            corrections[tile["tile"]] = [float(value) for value in correction]
        jobs.append(("__".join(name_parts), corrections))
    return jobs


def _write_and_render_variant(
    position_payload: dict[str, Any],
    registration_payload: dict[str, Any],
    output_dir: Path,
    name: str,
    corrections_px_zyx: dict[str, list[float]],
    channel: int,
    level: int,
    crop_fraction_yx: tuple[float, float],
    center_y_xz: bool,
) -> dict[str, Any]:
    variant_dir = output_dir / name
    qc_dir = variant_dir / "registration-qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    position_variant = copy.deepcopy(position_payload)
    registration_variant = copy.deepcopy(registration_payload)
    for tile_name, correction_px_zyx in corrections_px_zyx.items():
        _apply_correction(position_variant, tile_name, correction_px_zyx, translation_key="translation_um")
        _apply_correction(registration_variant, tile_name, correction_px_zyx, translation_key="stage_translation_um")

    position_path = variant_dir / "405.to488.optimized.positions.json"
    registration_path = variant_dir / "405-to488.optimized.registration.json"
    position_path.write_text(json.dumps(position_variant, indent=2) + "\n")
    registration_path.write_text(json.dumps(registration_variant, indent=2) + "\n")

    render_registration_qc(
        position_input=position_path,
        registration_input=registration_path,
        output_dir=qc_dir,
        channel=channel,
        level=level,
        center_y_xz=center_y_xz,
    )
    xy_path = qc_dir / f"level{level}_registered_lr_xy_isoZ_yellowOverlay_ch{channel}.png"
    return {
        "name": name,
        "output_dir": str(variant_dir),
        "position": str(position_path),
        "registration": str(registration_path),
        "xy": str(xy_path),
        "top_left_quadrant": str(_crop_top_left(xy_path, crop_fraction_yx)),
        "corrections_px_zyx": corrections_px_zyx,
    }


def _apply_correction(
    payload: dict[str, Any],
    tile_name: str,
    correction_px_zyx: list[float],
    *,
    translation_key: str,
) -> None:
    for tile in payload["tiles"]:
        if tile["tile"] != tile_name:
            continue
        scale = [float(tile[_scale_key(translation_key)][dim]) for dim in DIMENSIONS]
        for dim, correction_px, scale_um in zip(DIMENSIONS, correction_px_zyx, scale, strict=True):
            tile[translation_key][dim] = float(tile[translation_key][dim]) + float(correction_px) * scale_um
        return
    raise ValueError(f"{tile_name} is missing from payload")


def _scale_key(translation_key: str) -> str:
    if translation_key == "translation_um":
        return "scale_um"
    if translation_key == "stage_translation_um":
        return "stage_scale_um"
    raise ValueError(f"Unsupported translation key {translation_key!r}")


def _crop_top_left(image_path: Path, crop_fraction_yx: tuple[float, float]) -> Path:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    crop_height = max(1, int(round(height * crop_fraction_yx[0])))
    crop_width = max(1, int(round(width * crop_fraction_yx[1])))
    output = image_path.with_name(image_path.stem + ".top_left_quadrant.png")
    image.crop((0, 0, crop_width, crop_height)).save(output)
    return output


def _draw_boundaries(crop_path: Path, level: int, channel: int) -> Path:
    summary_path = crop_path.parent / f"level{level}_registered_lr_isoZ_yellowOverlay_ch{channel}.json"
    render_summary = json.loads(summary_path.read_text())
    image = Image.open(crop_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for tile in render_summary["tiles"]:
        z_start, y_start, x_start = tile["level_start_zyx"]
        z_size, y_size, x_size = tile["sampled_shape_zyx"]
        del z_start, z_size
        if x_start + x_size < 0 or y_start + y_size < 0 or x_start > width or y_start > height:
            continue
        color = (255, 255, 255) if tile["side"] == "L" else (255, 80, 80)
        draw.rectangle((x_start, y_start, x_start + x_size, y_start + y_size), outline=color, width=2)
        draw.text((max(2, x_start + 3), max(2, y_start + 3)), _tile_label(tile["tile"]), fill=color)
    output = crop_path.with_name(crop_path.stem + ".boundaries.png")
    image.save(output)
    return output


def _write_sheet(records: list[dict[str, Any]], image_key: str, output: Path) -> None:
    thumbs = []
    for record in records:
        image = Image.open(record[image_key]).convert("RGB")
        image.thumbnail((560, 360), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (560, 390), "white")
        canvas.paste(image, (0, 30))
        ImageDraw.Draw(canvas).text((4, 6), record["name"], fill="black")
        thumbs.append(canvas)

    cols = 4
    sheet = Image.new("RGB", (cols * 560, ((len(thumbs) + cols - 1) // cols) * 390), "white")
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % cols) * 560, (index // cols) * 390))
    sheet.save(output)


def _tile_label(tile_name: str) -> str:
    basename = Path(tile_name).name
    side = "L" if "-CL-" in basename else "R" if "-CR-" in basename else ""
    parts = basename.split(".")
    return f"{side}{parts[-3]}" if len(parts) >= 3 else basename
