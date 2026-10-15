#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from PIL import Image, ImageDraw

from squisher_lightsheet.channel_affine import _read_level0_crop, output_to_input_from_model
from squisher_lightsheet.tile_phase import tile_record_from_position_record
from squisher_lightsheet.qc import load_qc_font


DEFAULT_SUMMARY = Path(
    "/home/chaichontat/nvme/lightsheet/working/eduseg/Image_14/"
    "image10-ch0-to-image14-fused-fixed-method8-c528-overlap10/fused_fixed_method8_summary.json"
)

LEVEL0_ARTIFACT_TYPES = {
    "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
    "lightsheet.image10_to_fused_image14_method8_trial.v1",
}
NATIVE_METHOD_LABELS = {
    "method6": "method 6",
    "method8": "method 8",
    "method10-mattes": "method 10 Mattes",
    "method11-mattes": "method 11 Mattes",
}
NATIVE_RENDER_FIELDS = (
    "native_pretranslated_fixed_start_zyx",
    "native_pretranslated_fixed_stop_zyx",
    "native_pretranslated_moving_start_zyx",
    "native_pretranslated_moving_stop_zyx",
    "native_pretranslated_crop_shape_zyx",
    "native_initial_matrix_zyx",
    "native_initial_translation_zyx",
    "native_cropped_local_matrix_zyx",
    "native_cropped_local_translation_zyx",
)


def _display_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(array, dtype=np.float32).ravel() for array in arrays])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.8])
    return float(low), float(max(high, low + 1.0))


def _robust_u8(a: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    values = np.asarray(a, dtype=np.float32)
    low, high = limits
    return np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _overlay(
    red_source: np.ndarray,
    green_source: np.ndarray,
    *,
    red_limits: tuple[float, float],
    green_limits: tuple[float, float],
) -> Image.Image:
    red = _robust_u8(red_source, red_limits)
    green = _robust_u8(green_source, green_limits)
    rgb = np.zeros((*red.shape, 3), dtype=np.uint8)
    rgb[..., 0] = red
    rgb[..., 1] = green
    return Image.fromarray(rgb)


def _native_method_provenance(row: dict[str, Any]) -> tuple[str, str]:
    method = row.get("native_method")
    if isinstance(method, str) and method in NATIVE_METHOD_LABELS:
        return method, NATIVE_METHOD_LABELS[method]
    selected = row.get("selected_attempt")
    if isinstance(selected, str):
        for key, label in NATIVE_METHOD_LABELS.items():
            if selected.startswith(f"native_{key.replace('-', '_')}_"):
                return key, label
    return "native-registration", "native registration"


def _render_transforms(
    row: dict[str, Any],
    *,
    before_transform: str = "phase-correlation",
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    if before_transform == "linear-initializer":
        before_matrix = row.get("fit_initial_matrix_zyx")
        before_translation = row.get("fit_initial_translation_zyx")
        if before_matrix is None or before_translation is None:
            raise ValueError(f"{row.get('moving_tile')} has no fitted linear initializer transform")
    else:
        phase = row.get("phase_priming")
        if not isinstance(phase, dict) or "local_matrix_zyx" not in phase or "local_translation_zyx" not in phase:
            raise ValueError(f"{row.get('moving_tile')} has no phase-correlation transform")
        before_matrix = phase["local_matrix_zyx"]
        before_translation = phase["local_translation_zyx"]
    if "selected_local_matrix_zyx" not in row or "selected_local_translation_zyx" not in row:
        raise ValueError(f"{row.get('moving_tile')} has no selected native-registration transform")
    return (
        (
            np.asarray(before_matrix, dtype=np.float32),
            np.asarray(before_translation, dtype=np.float32),
        ),
        (
            np.asarray(row["selected_local_matrix_zyx"], dtype=np.float32),
            np.asarray(row["selected_local_translation_zyx"], dtype=np.float32),
        ),
    )


def _render_frame(
    row: dict[str, Any],
    *,
    before_transform: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    bool,
]:
    """Resolve crop bounds and transforms from one consistent coordinate frame."""
    selected_name = row.get("selected_attempt")
    if before_transform != "phase-correlation" or not (
        isinstance(selected_name, str) and selected_name.startswith("native_")
    ):
        fixed_start = np.asarray(row["fixed_start_zyx"], dtype=np.int64)
        fixed_stop = np.asarray(row["fixed_stop_zyx"], dtype=np.int64)
        moving_start_values = row.get("moving_start_l0_zyx", row.get("moving_start_zyx"))
        if moving_start_values is None:
            raise ValueError(f"{row.get('moving_tile')} has no moving_start_l0_zyx or moving_start_zyx")
        before, after = _render_transforms(row, before_transform=before_transform)
        return fixed_start, np.asarray(moving_start_values, dtype=np.int64), fixed_stop - fixed_start, before, after, False

    attempts = row.get("method8_attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"{row.get('moving_tile')} selected {selected_name} but has no method8_attempts")
    matches = [attempt for attempt in attempts if isinstance(attempt, dict) and attempt.get("name") == selected_name]
    if len(matches) != 1:
        raise ValueError(
            f"{row.get('moving_tile')} selected {selected_name} but found {len(matches)} matching attempts"
        )
    attempt = matches[0]
    missing = [key for key in NATIVE_RENDER_FIELDS if attempt.get(key) is None]
    if missing:
        raise ValueError(f"{row.get('moving_tile')} selected native attempt is missing {', '.join(missing)}")

    fixed_start = np.asarray(attempt["native_pretranslated_fixed_start_zyx"], dtype=np.int64)
    fixed_stop = np.asarray(attempt["native_pretranslated_fixed_stop_zyx"], dtype=np.int64)
    moving_start = np.asarray(attempt["native_pretranslated_moving_start_zyx"], dtype=np.int64)
    moving_stop = np.asarray(attempt["native_pretranslated_moving_stop_zyx"], dtype=np.int64)
    shape = np.asarray(attempt["native_pretranslated_crop_shape_zyx"], dtype=np.int64)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError(f"{row.get('moving_tile')} has invalid native crop shape {shape.tolist()}")
    if not np.array_equal(fixed_stop - fixed_start, shape) or not np.array_equal(moving_stop - moving_start, shape):
        raise ValueError(f"{row.get('moving_tile')} native crop bounds do not match its crop shape")
    before = (
        np.asarray(attempt["native_initial_matrix_zyx"], dtype=np.float32),
        np.asarray(attempt["native_initial_translation_zyx"], dtype=np.float32),
    )
    after = (
        np.asarray(attempt["native_cropped_local_matrix_zyx"], dtype=np.float32),
        np.asarray(attempt["native_cropped_local_translation_zyx"], dtype=np.float32),
    )
    return fixed_start, moving_start, shape, before, after, True


def _render_one(
    row: dict[str, Any],
    *,
    moving_position: Path,
    device: int,
    z_fractions: tuple[float, ...],
    z_selection: str,
    before_transform: str,
    display_intensity_transform: str = "linear",
) -> Image.Image:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform

    from squisher_lightsheet._backends import fused_fixed_trial as trial
    fixed_root = zarr.open_group(row["fixed_fused"], mode="r")
    fixed_start, moving_start, shape, before, final_transform, native_frame = _render_frame(
        row,
        before_transform=before_transform,
    )
    fixed = np.asarray(trial._read_fused_crop(fixed_root["0"], fixed_start, shape), dtype=np.float32)

    tile_no = str(row["moving_tile"]).split(".")[-3]
    moving_record = trial._record_by_tile_number(moving_position, tile_no)
    moving_tile = tile_record_from_position_record(moving_record)
    moving, _ = _read_level0_crop(
        moving_tile,
        channel=int(row["moving_channel"]),
        start_zyx=moving_start,
        crop_shape_zyx=tuple(int(value) for value in shape),
    )

    if display_intensity_transform == "log1p":
        fixed = np.log1p(np.maximum(fixed, 0.0)).astype(np.float32, copy=False)
        moving = np.log1p(np.maximum(moving, 0.0)).astype(np.float32, copy=False)
    elif display_intensity_transform != "linear":
        raise ValueError(
            "display_intensity_transform must be 'linear' or 'log1p', "
            f"got {display_intensity_transform!r}"
        )

    with cp.cuda.Device(int(device)):
        moving_gpu = cp.asarray(moving, dtype=cp.float32)
        registered = []
        for matrix, translation in (before, final_transform):
            pull_matrix, pull_offset = output_to_input_from_model(
                matrix,
                translation,
                tuple(int(value) for value in shape),
            )
            registered.append(
                cp.asnumpy(
                    gpu_affine_transform(
                        moving_gpu,
                        cp.asarray(pull_matrix, dtype=cp.float32),
                        cp.asarray(pull_offset, dtype=cp.float32),
                        output_shape=tuple(int(value) for value in shape),
                        order=1,
                        mode="constant",
                        cval=0.0,
                    )
                )
            )
    before_registered, final_registered = registered

    if z_selection == "bright-moving":
        from skimage.filters import threshold_multiotsu

        sampled = moving[::4, ::4, ::4]
        bright_threshold = float(threshold_multiotsu(sampled, classes=3, nbins=4096)[-1])
        z_scores = np.maximum(moving[:, ::4, ::4] - bright_threshold, 0.0).mean(axis=(1, 2))
        z_indices = [int(band[np.argmax(z_scores[band])]) for band in np.array_split(np.arange(shape[0]), 3)]
    else:
        z_indices = [int(round((int(shape[0]) - 1) * fraction)) for fraction in z_fractions]
    fixed_limits = _display_limits(*(fixed[z] for z in z_indices))
    moving_limits = _display_limits(
        *(before_registered[z] for z in z_indices),
        *(final_registered[z] for z in z_indices),
    )
    _native_method, native_method_label = _native_method_provenance(row)
    failed_reason = row.get("rendered_failed_reason")
    after_label = f"after: {native_method_label}"
    if row.get("rendered_failed_attempt") is True:
        after_label += f" FAILED ({failed_reason})" if failed_reason else " FAILED"
    labels = [
        (
            "before: native initializer"
            if native_frame
            else "before: linear interpolation"
            if before_transform == "linear-initializer"
            else "before: phase correlation"
        ),
        after_label,
    ]
    panel_w = int(shape[2])
    panel_h = int(shape[1]) + 24
    panel_font = load_qc_font(15)
    sheet = Image.new("RGB", (panel_w * len(labels), panel_h * len(z_indices)), (0, 0, 0))
    for row_index, z in enumerate(z_indices):
        panels = [
            _overlay(before_registered[z], fixed[z], red_limits=moving_limits, green_limits=fixed_limits),
            _overlay(final_registered[z], fixed[z], red_limits=moving_limits, green_limits=fixed_limits),
        ]
        for col, panel in enumerate(panels):
            canvas = Image.new("RGB", (panel_w, panel_h), (0, 0, 0))
            canvas.paste(panel, (0, 24))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 5), f"z={z} {labels[col]}", fill=(255, 255, 255), font=panel_font)
            sheet.paste(canvas, (col * panel_w, row_index * panel_h))
    return sheet


def _select_rows(
    summary: dict[str, Any],
    *,
    limit: int,
    tile_filter: set[str] | None = None,
    fixed_fused: Path | None = None,
    status: str | None = "accepted",
    rejection_reason: str | None = None,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in summary.get("windows", [])
        if (status is None or row.get("status") == status)
        and (rejection_reason is None or row.get("rejection_reason") == rejection_reason)
        and (tile_filter is None or str(row.get("tile", "")).split(".")[-3] in tile_filter)
        and row.get("level0_json")
        and Path(row["level0_json"]).exists()
    ]
    rows.sort(
        key=lambda row: (
            float(row.get("selected_corr") if row.get("selected_corr") is not None else float("inf")),
            float(row.get("selected_grad_ncc") if row.get("selected_grad_ncc") is not None else float("inf")),
        )
    )
    selected = []
    for summary_row in rows:
        payload = json.loads(Path(summary_row["level0_json"]).read_text())
        phase_priming = payload.get("phase_priming")
        has_phase = isinstance(phase_priming, dict) and "local_matrix_zyx" in phase_priming and "local_translation_zyx" in phase_priming
        has_selected = "selected_local_matrix_zyx" in payload and "selected_local_translation_zyx" in payload
        if has_phase and not has_selected and payload.get("status") == "rejected":
            attempts = payload.get("method8_attempts")
            renderable = [
                attempt
                for attempt in attempts if isinstance(attempt, dict)
                and isinstance(attempt.get("name"), str)
                and attempt["name"].startswith("native_")
                and attempt.get("status") == "rejected"
                and attempt.get("local_matrix_zyx") is not None
                and attempt.get("local_translation_zyx") is not None
                and all(attempt.get(key) is not None for key in NATIVE_RENDER_FIELDS)
            ] if isinstance(attempts, list) else []
            if len(renderable) == 1:
                attempt = renderable[0]
                payload["selected_attempt"] = attempt["name"]
                payload["selected_local_matrix_zyx"] = attempt["local_matrix_zyx"]
                payload["selected_local_translation_zyx"] = attempt["local_translation_zyx"]
                payload["rendered_failed_attempt"] = True
                payload["rendered_failed_reason"] = attempt.get("rejection_reason")
                has_selected = True
        if not (has_phase and has_selected):
            continue
        if fixed_fused is not None:
            payload["fixed_fused"] = str(fixed_fused)
        payload["quadrant"] = summary_row.get("quadrant")
        payload["summary_status"] = summary_row.get("status")
        payload["summary_rejection_reason"] = summary_row.get("rejection_reason")
        payload["summary_selected_corr"] = summary_row.get("selected_corr")
        payload["summary_selected_grad_ncc"] = summary_row.get("selected_grad_ncc")
        payload["summary_initializer_displacement_px"] = summary_row.get("initializer_displacement_px")
        payload["summary_neighbor_displacement_px"] = summary_row.get("neighbor_displacement_px")
        selected.append(payload)
        if len(selected) >= int(limit):
            break
    return selected


def _partial_summary(run_dir: Path) -> dict[str, Any]:
    paths = sorted((run_dir / "window_json").glob("*.json"))
    windows: list[dict[str, Any]] = []
    first_payload: dict[str, Any] | None = None
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("artifact_type") not in LEVEL0_ARTIFACT_TYPES:
            continue
        if first_payload is None:
            first_payload = payload
        windows.append(
            {
                "status": payload.get("status"),
                "rejection_reason": payload.get("rejection_reason"),
                "tile": payload.get("moving_tile"),
                "quadrant": payload.get("quadrant"),
                "level0_json": str(path.resolve()),
                "selected_corr": payload.get("selected_corr_refined"),
                "selected_grad_ncc": payload.get("selected_gradient_component_ncc_mean"),
            }
        )
    if first_payload is None:
        raise ValueError(f"No completed level-0 window JSON files found in {run_dir / 'window_json'}")
    cache = first_payload.get("cache_config")
    if not isinstance(cache, dict):
        raise ValueError(f"Completed window {paths[0]} has no cache_config provenance")
    moving_position = cache.get("moving_source_position") or cache.get("moving_position")
    if not isinstance(moving_position, str):
        raise ValueError(f"Completed window {paths[0]} has no moving position provenance")
    return {
        "partial": True,
        "run_dir": str(run_dir.resolve()),
        "fixed_fused": str(first_payload["fixed_fused"]),
        "moving_position": moving_position,
        "windows": windows,
    }


def _load_render_input(*, summary_path: Path | None, run_dir: Path | None) -> tuple[dict[str, Any], Path | None]:
    if (summary_path is None) == (run_dir is None):
        raise ValueError("Pass exactly one of --summary or --run-dir")
    if summary_path is not None:
        return json.loads(summary_path.read_text()), summary_path.resolve()
    assert run_dir is not None
    final_summary = run_dir / "fused_fixed_method8_summary.json"
    if final_summary.is_file():
        summary = json.loads(final_summary.read_text())
        summarized_windows = {
            Path(row["level0_json"]).resolve()
            for row in summary.get("windows", [])
            if isinstance(row, dict) and isinstance(row.get("level0_json"), str)
        }
        completed_windows = {path.resolve() for path in (run_dir / "window_json").glob("*.json")}
        if completed_windows.issubset(summarized_windows):
            return summary, final_summary.resolve()
    return _partial_summary(run_dir), None


def run(args: argparse.Namespace) -> Path:
    summary, summary_path = _load_render_input(
        summary_path=None if args.summary is None else Path(args.summary),
        run_dir=None if args.run_dir is None else Path(args.run_dir),
    )
    moving_position = Path(summary["moving_position"])
    fixed_fused = Path(args.fixed_fused).resolve() if args.fixed_fused is not None else None
    tile_filter = None
    if args.tile_filter:
        tile_filter = {value.strip().zfill(3) for value in args.tile_filter.split(",") if value.strip()}
    rows = _select_rows(
        summary,
        limit=int(args.limit),
        tile_filter=tile_filter,
        fixed_fused=fixed_fused,
        status=args.status,
        rejection_reason=args.rejection_reason,
    )
    output_dir = Path(args.output_dir)
    z_fractions = tuple(float(value) for value in args.z_fractions.split(","))
    if not z_fractions or any(not 0.0 <= value <= 1.0 for value in z_fractions):
        raise ValueError(f"--z-fractions must contain comma-separated values in [0,1], got {args.z_fractions!r}")
    thumbs = []
    font = load_qc_font(18)
    for index, row in enumerate(rows, start=1):
        image = _render_one(
            row,
            moving_position=moving_position,
            device=int(args.device),
            z_fractions=z_fractions,
            z_selection=args.z_selection,
            before_transform=args.before_transform,
        )
        image.thumbnail((1024, 768), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1024, 800), (0, 0, 0))
        canvas.paste(image, ((1024 - image.width) // 2, 28))
        draw = ImageDraw.Draw(canvas)
        displacement_text = ""
        if row.get("summary_initializer_displacement_px") is not None:
            displacement_text = (
                f" displacement init={float(row['summary_initializer_displacement_px']):.1f}px"
                f" neighbor={float(row['summary_neighbor_displacement_px']):.1f}px"
            )
        draw.text(
            (8, 8),
            f"{index:02d} {row['moving_tile']} z={row.get('moving_start_l0_zyx', row.get('moving_start_zyx'))[0]} "
            f"status={row.get('status')} reason={row.get('rejection_reason')} "
            f"corr={float(row.get('corr_refined', float('nan'))):.4f} "
            f"grad={float(row.get('gradient_component_ncc_mean', float('nan'))):.4f}{displacement_text}",
            fill=(255, 255, 255),
            font=font,
        )
        thumbs.append(canvas)
        print(f"rendered {row['moving_tile']} {row.get('quadrant')}", flush=True)

    if not thumbs:
        raise ValueError("No rows were available for contact sheet rendering")
    cols = int(args.columns)
    rows_count = int(np.ceil(len(thumbs) / cols))
    combined = Image.new("RGB", (cols * 1024, rows_count * 800), (0, 0, 0))
    for index, thumb in enumerate(thumbs):
        y, x = divmod(index, cols)
        combined.paste(thumb, (x * 1024, y * 800))
    name_parts = [args.status or "all"]
    if args.rejection_reason:
        name_parts.append(args.rejection_reason)
    output = output_dir / f"{'_'.join(name_parts)}_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output, dpi=(300, 300))
    native_methods = sorted({_native_method_provenance(row)[0] for row in rows})
    after_panel = (
        f"{native_methods[0].replace('-', '_')}_overlay"
        if len(native_methods) == 1
        else "native_registration_overlay"
    )
    index_path = output_dir / "lowest_grad_contact_sheet.index.json"
    index_path.write_text(
        json.dumps(
            {
                "summary": None if summary_path is None else str(summary_path),
                "run_dir": summary.get("run_dir"),
                "partial": bool(summary.get("partial", False)),
                "fixed_fused": str(fixed_fused) if fixed_fused is not None else summary.get("fixed_fused"),
                "status": args.status,
                "tile_filter": None if tile_filter is None else sorted(tile_filter),
                "rejection_reason": args.rejection_reason,
                "z_fractions": list(z_fractions),
                "z_selection": args.z_selection,
                "native_methods": native_methods,
                "panels": [f"{args.before_transform.replace('-', '_')}_overlay", after_panel],
            },
            indent=2,
        )
        + "\n"
    )
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render native contact sheets for lowest-gradient fused-fixed method8 windows.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary", type=Path)
    source.add_argument("--run-dir", type=Path, help="Render completed windows even before the final summary exists.")
    parser.add_argument("--fixed-fused", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY.parent / "contact_sheets")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tile-filter", default=None, help="Comma-separated three-digit tile numbers.")
    parser.add_argument("--z-fractions", default="0.25,0.5,0.75")
    parser.add_argument("--z-selection", choices=("fractions", "bright-moving"), default="fractions")
    parser.add_argument(
        "--before-transform",
        choices=("phase-correlation", "linear-initializer"),
        default="phase-correlation",
    )
    parser.add_argument("--status", default="accepted")
    parser.add_argument("--rejection-reason", default=None)
    return parser


def main() -> None:
    output = run(build_parser().parse_args())
    print(output, flush=True)


if __name__ == "__main__":
    main()
