from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from squisher_lightsheet.legacy_runner import run_legacy_script


SIDES = ("L", "R")


def scale_u8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(positive, [1.0, 99.8])
    scaled = np.clip((image - low) / max(float(high - low), 1.0), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


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
