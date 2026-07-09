from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Literal

import typer
import numpy as np
from loguru import logger

from squisher_lightsheet.channel_alignment import (
    optimize_405_to_488_translations,
    recover_405_488_mvs_seam_anchors,
    refine_405_488_level0_anchors,
    run_405_to_488_workflow,
)
from squisher_lightsheet.channel_mattes_anchors import (
    MattesAnchorParameters,
    measure_405_to_488_mattes_anchors,
)
from squisher_lightsheet.channel_subtraction import (
    DEFAULT_COMPRESSION as DEFAULT_SUBTRACTION_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL as DEFAULT_SUBTRACTION_COMPRESSION_LEVEL,
    subtract_channel_tiles,
)
from squisher_lightsheet.candidate_grid import render_candidate_grid
from squisher_lightsheet.channel_affine import (
    align_tiles_to_reference_affine,
)
from squisher_lightsheet.cross_register_method8 import (
    DEFAULT_LIB_DIR as DEFAULT_CROSS_REGISTER_METHOD8_LIB_DIR,
    run_tile_quadrant_method8,
)
from squisher_lightsheet.fusion import fuse_tiles
from squisher_lightsheet.tile_quadrant_fusion import export_tile_quadrant_materialized_chunks
from squisher_lightsheet.lr_alignment import (
    DEFAULT_LEVEL as DEFAULT_LR_ALIGNMENT_LEVEL,
    DEFAULT_PHASE_DOWNSAMPLE_ZYX,
    DEFAULT_Z_SLAB_PLANES,
    run_lr_dumb_stitch_alignment,
)
from squisher_lightsheet.modes import ModeName
from squisher_lightsheet.mvs_seams import (
    DEFAULT_LEVEL0_FALLBACK_LEVEL2_WEIGHT_SCALE,
    DEFAULT_LEVEL0_MAX_DISCONNECTED_ISLAND_SIZE,
    DEFAULT_LEVEL0_PATCHES_PER_EDGE,
    DEFAULT_LEVEL0_RETRY_PATCHES_PER_EDGE,
    load_mvs_registration,
    mvs_used_edge_audit,
    refine_mvs_registration_level0,
    score_mvs_edges_with_gradient_ncc,
)
from squisher_lightsheet.ome_rechunk import rechunk_ome_tiffs
from squisher_lightsheet.parsing import parse_source_view_path_entry
from squisher_lightsheet.positions import create_position_file
from squisher_lightsheet.pyramid import add_pyramids
from squisher_lightsheet.qc import (
    render_live_fusion_preview,
    render_fused_xyz_overlay_qc,
    render_fused_tile_index_overlay,
    render_registration_center_z_spotcheck,
    render_registration_qc,
)
from squisher_lightsheet.registration import register_tiles
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy_registration
from squisher_lightsheet.rough_phase import rough_phase_align
from squisher_lightsheet.tile_phase import align_tiles_to_reference, parse_shape_zyx
from squisher_lightsheet.track_z import run_track_z_diagnostics
from squisher_lightsheet.workflow import DEFAULT_LEVEL, DEFAULT_OVERLAP_FRACTION, run_tltr_workflow


app = typer.Typer(no_args_is_help=True)
align_405_to_488_app = typer.Typer(no_args_is_help=True)
cross_register_method8_app = typer.Typer(no_args_is_help=True)
app.add_typer(align_405_to_488_app, name="align-405-to-488")
app.add_typer(cross_register_method8_app, name="cross-register-method8")


def _parse_source_view_flatfield_dir(value: str) -> tuple[str, Path]:
    return parse_source_view_path_entry(value, error_factory=typer.BadParameter)


def _parse_source_view_flatfield_dirs(values: list[str] | None) -> dict[str, Path] | None:
    if values is None:
        return None
    parsed: dict[str, Path] = {}
    for value in values:
        view, path = _parse_source_view_flatfield_dir(value)
        if view in parsed:
            raise typer.BadParameter(f"Duplicate --flatfield-dir-by-source-view entry for {view!r}")
        parsed[view] = path
    return parsed


def _log_progress(message: str) -> None:
    logger.info(message)


def _parse_float_zyx(text: str) -> tuple[float, float, float]:
    values = tuple(float(part) for part in text.split(","))
    if len(values) != 3:
        raise typer.BadParameter(f"Expected comma-separated z,y,x values, got {text!r}")
    return values


def _parse_optional_float(text: str) -> float | None:
    if text.strip().lower() in {"none", "off", "disabled"}:
        return None
    return float(text)


def _parse_int_zyx(text: str, option_name: str = "value") -> tuple[int, int, int]:
    values = tuple(int(part) for part in text.split(","))
    if len(values) != 3:
        raise typer.BadParameter(f"{option_name} expects comma-separated z,y,x values, got {text!r}")
    return values


def _parse_float_yx(text: str) -> tuple[float, float]:
    values = tuple(float(part) for part in text.split(","))
    if len(values) != 2:
        raise typer.BadParameter(f"Expected comma-separated y,x values, got {text!r}")
    return values


def _format_int_zyx(values: tuple[int, int, int]) -> str:
    return ",".join(str(value) for value in values)


def _parse_devices(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise typer.BadParameter("Expected at least one comma-separated CUDA device id")
    if any(value < 0 for value in values):
        raise typer.BadParameter(f"CUDA device ids must be non-negative, got {text!r}")
    return values


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _cross_register_manifest_path(output_dir: Path) -> Path:
    return output_dir / "cross_register_method8_manifest.json"


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"{path} does not contain a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    tmp.replace(path)
    return path.resolve()


def _update_cross_register_manifest(
    output_dir: Path,
    *,
    stage: str,
    updates: dict[str, object],
) -> Path:
    manifest_path = _cross_register_manifest_path(output_dir)
    manifest = _read_json_if_exists(manifest_path)
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        stages = {}
    stages[stage] = updates
    manifest.update(
        {
            "artifact_type": "squisher_lightsheet.cross_register_method8_manifest.v1",
            "output_dir": str(output_dir.resolve()),
            "stages": stages,
        }
    )
    return _write_json_atomic(manifest_path, manifest)


def _default_cross_register_coarse_position(
    output_dir: Path,
    *,
    moving_token: str,
    moving_channel: int,
    fixed_token: str,
) -> Path:
    return output_dir / f"{moving_token}.ch{moving_channel}.coarse-aligned-to-{fixed_token}.positions.json"


def _default_materialized_output_dir(output_dir: Path, *, fusion_channel: int) -> Path:
    return output_dir / f"materialized_fusion_inputs_ch{fusion_channel}"


def _require_empty_or_overwrite(path: Path, *, overwrite: bool, label: str) -> None:
    if not path.exists():
        return
    has_content = any(path.iterdir()) if path.is_dir() else True
    if has_content and not overwrite:
        raise typer.BadParameter(f"{label} already exists at {path}; pass --overwrite to replace it")
    if overwrite:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _require_paths_exist(paths: dict[str, Path]) -> None:
    missing = {label: path for label, path in paths.items() if not path.exists()}
    if missing:
        details = ", ".join(f"{label}={path}" for label, path in missing.items())
        raise typer.BadParameter(f"Missing required cross-register artifact(s): {details}")


@cross_register_method8_app.command("coarse")
def cross_register_method8_coarse(
    fixed_position: Annotated[Path, typer.Option("--fixed-position", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    output_position: Annotated[Path | None, typer.Option("--output-position", dir_okay=False)] = None,
    fixed_channel: Annotated[int, typer.Option("--fixed-channel", min=0)] = 0,
    moving_channel: Annotated[int, typer.Option("--moving-channel", min=0)] = 0,
    fixed_token: Annotated[str, typer.Option("--fixed-token")] = "Image_14",
    moving_token: Annotated[str, typer.Option("--moving-token")] = "Image_10",
    level: Annotated[int, typer.Option("--level", min=0)] = 2,
    coarse_level: Annotated[int, typer.Option("--coarse-level", min=0)] = 2,
    upsample_factor: Annotated[int, typer.Option("--upsample-factor", min=1)] = 10,
    patch_shape_zyx: Annotated[str | None, typer.Option("--patch-shape-zyx")] = "96,320,320",
    min_inliers: Annotated[int, typer.Option("--min-inliers", min=3)] = 3,
    max_candidate_patches: Annotated[int, typer.Option("--max-candidate-patches", min=1)] = 24,
    scout_z_samples: Annotated[int | None, typer.Option("--scout-z-samples", min=1)] = 32,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    fixed_registration: Annotated[
        Path | None,
        typer.Option("--fixed-registration", exists=True, dir_okay=False, readable=True),
    ] = None,
    output_registration: Annotated[Path | None, typer.Option("--output-registration", dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    resolved_output_position = output_position or _default_cross_register_coarse_position(
        output_dir,
        moving_token=moving_token,
        moving_channel=moving_channel,
        fixed_token=fixed_token,
    )
    _require_empty_or_overwrite(resolved_output_position, overwrite=overwrite, label="coarse output position")
    if output_registration is not None:
        _require_empty_or_overwrite(output_registration, overwrite=overwrite, label="coarse output registration")
    path = align_tiles_to_reference(
        reference_position=fixed_position,
        output_position=resolved_output_position,
        output_dir=output_dir,
        reference_channel=fixed_channel,
        moving_channel=moving_channel,
        reference_token=fixed_token,
        moving_token=moving_token,
        level=level,
        upsample_factor=upsample_factor,
        patch_shape_zyx=None if patch_shape_zyx is None else parse_shape_zyx(patch_shape_zyx),
        min_inliers=min_inliers,
        max_candidate_patches=max_candidate_patches,
        coarse_level=coarse_level,
        scout_z_samples=scout_z_samples,
        workers=workers,
        reference_registration_input=fixed_registration,
        output_registration=output_registration,
    )
    manifest = _update_cross_register_manifest(
        output_dir,
        stage="coarse",
        updates={
            "fixed_position": fixed_position.resolve(),
            "output_position": path,
            "fixed_registration": None if fixed_registration is None else fixed_registration.resolve(),
            "output_registration": None if output_registration is None else output_registration.resolve(),
            "fixed_token": fixed_token,
            "moving_token": moving_token,
            "fixed_channel": fixed_channel,
            "moving_channel": moving_channel,
            "level": level,
            "coarse_level": coarse_level,
            "patch_shape_zyx": None if patch_shape_zyx is None else list(parse_shape_zyx(patch_shape_zyx)),
        },
    )
    typer.echo(json.dumps({"coarse_position": str(path), "manifest": str(manifest)}, indent=2))


@cross_register_method8_app.command("method8")
def cross_register_method8_method8(
    fixed_position: Annotated[Path, typer.Option("--fixed-position", exists=True, dir_okay=False, readable=True)],
    coarse_moving_position: Annotated[
        Path,
        typer.Option("--coarse-moving-position", exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    fixed_channel: Annotated[int, typer.Option("--fixed-channel", min=0)] = 0,
    moving_channel: Annotated[int, typer.Option("--moving-channel", min=0)] = 0,
    core_shape_zyx: Annotated[str, typer.Option("--core-shape-zyx")] = "480,480,480",
    window_shape_zyx: Annotated[str, typer.Option("--window-shape-zyx")] = "528,528,528",
    fit_downsample_zyx: Annotated[str, typer.Option("--fit-downsample-zyx")] = "1,1,1",
    preseed_matrix_zyx: Annotated[
        str | None,
        typer.Option("--preseed-matrix-zyx", help="Nine comma-separated row-major zyx values."),
    ] = None,
    native_lib_dir: Annotated[
        Path,
        typer.Option("--native-lib-dir", exists=True, file_okay=False, readable=True),
    ] = DEFAULT_CROSS_REGISTER_METHOD8_LIB_DIR,
    ftol: Annotated[float, typer.Option("--ftol", min=0.0)] = 1e-4,
    max_iterations: Annotated[int, typer.Option("--max-iterations", min=1)] = 300,
    min_corr: Annotated[float, typer.Option("--min-corr")] = 0.15,
    min_grad_ncc: Annotated[float, typer.Option("--min-grad-ncc")] = 0.24,
    empty_precheck_level: Annotated[int, typer.Option("--empty-precheck-level")] = -1,
    empty_precheck_min_dynamic_range: Annotated[float, typer.Option("--empty-precheck-min-dynamic-range")] = 1.0,
    empty_precheck_min_std: Annotated[float, typer.Option("--empty-precheck-min-std")] = 0.25,
    fixed_mask_threshold: Annotated[
        str,
        typer.Option("--fixed-mask-threshold", help="Fixed-mask threshold; use none/off/disabled to turn it off."),
    ] = "3000",
    fixed_mask_level: Annotated[int, typer.Option("--fixed-mask-level", min=0)] = 2,
    fixed_mask_min_voxels: Annotated[int, typer.Option("--fixed-mask-min-voxels", min=1)] = 256,
    fixed_mask_max_masked_fraction: Annotated[
        float,
        typer.Option("--fixed-mask-max-masked-fraction", min=0.0, max=1.0),
    ] = 0.95,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    devices: Annotated[str, typer.Option("--devices")] = "0",
    tile_filter: Annotated[str | None, typer.Option("--tile-filter")] = None,
    max_windows: Annotated[int | None, typer.Option("--max-windows", min=1)] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    summary_path = output_dir / "tile_quadrant_method8_summary.json"
    window_json_dir = output_dir / "window_json"
    if overwrite:
        _require_empty_or_overwrite(summary_path, overwrite=True, label="method8 summary")
        _require_empty_or_overwrite(window_json_dir, overwrite=True, label="method8 window JSON directory")
        _require_empty_or_overwrite(
            output_dir / "interpolated_prior_models",
            overwrite=True,
            label="obsolete interpolation prior directory",
        )
        _require_empty_or_overwrite(
            output_dir / "interpolation_rescue_window_json",
            overwrite=True,
            label="obsolete interpolation rescue directory",
        )
    elif not resume:
        _require_empty_or_overwrite(summary_path, overwrite=False, label="method8 summary")
        _require_empty_or_overwrite(window_json_dir, overwrite=False, label="method8 window JSON directory")
    path = run_tile_quadrant_method8(
        fixed_position=fixed_position,
        moving_position=coarse_moving_position,
        output_dir=output_dir,
        fixed_channel=fixed_channel,
        moving_channel=moving_channel,
        core_shape_zyx=parse_shape_zyx(core_shape_zyx),
        window_shape_zyx=parse_shape_zyx(window_shape_zyx),
        fit_downsample_zyx=parse_shape_zyx(fit_downsample_zyx),
        preseed_matrix_zyx=preseed_matrix_zyx,
        native_lib_dir=native_lib_dir,
        ftol=ftol,
        max_iterations=max_iterations,
        min_corr=min_corr,
        min_grad_ncc=min_grad_ncc,
        empty_precheck_level=empty_precheck_level,
        empty_precheck_min_dynamic_range=empty_precheck_min_dynamic_range,
        empty_precheck_min_std=empty_precheck_min_std,
        fixed_mask_threshold=_parse_optional_float(fixed_mask_threshold),
        fixed_mask_level=fixed_mask_level,
        fixed_mask_min_voxels=fixed_mask_min_voxels,
        fixed_mask_max_masked_fraction=fixed_mask_max_masked_fraction,
        workers=workers,
        devices=devices,
        tile_filter=tile_filter,
        max_windows=max_windows,
        resume=resume,
    )
    manifest = _update_cross_register_manifest(
        output_dir,
        stage="method8",
        updates={
            "fixed_position": fixed_position.resolve(),
            "coarse_moving_position": coarse_moving_position.resolve(),
            "summary": path,
            "window_json_dir": window_json_dir.resolve(),
            "fixed_channel": fixed_channel,
            "moving_channel": moving_channel,
            "core_shape_zyx": list(parse_shape_zyx(core_shape_zyx)),
            "window_shape_zyx": list(parse_shape_zyx(window_shape_zyx)),
            "fixed_mask_threshold": _parse_optional_float(fixed_mask_threshold),
            "workers": workers,
            "devices": list(_parse_devices(devices)),
            "resume": resume,
        },
    )
    typer.echo(json.dumps({"method8_summary": str(path), "window_json_dir": str(window_json_dir.resolve()), "manifest": str(manifest)}, indent=2))


@cross_register_method8_app.command("materialize")
def cross_register_method8_materialize(
    window_json_dir: Annotated[Path, typer.Option("--window-json-dir", exists=True, file_okay=False, readable=True)],
    coarse_moving_position: Annotated[
        Path,
        typer.Option("--coarse-moving-position", exists=True, dir_okay=False, readable=True),
    ],
    fixed_registration: Annotated[
        Path,
        typer.Option("--fixed-registration", exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    fusion_channel: Annotated[int, typer.Option("--fusion-channel", min=0)] = 1,
    materialized_output_dir: Annotated[
        Path | None,
        typer.Option("--materialized-output-dir", file_okay=False),
    ] = None,
    channel_source_shift_px_zyx: Annotated[
        str | None,
        typer.Option("--channel-source-shift-px-zyx"),
    ] = None,
    include_quality_gate_rejected: Annotated[
        bool,
        typer.Option("--include-quality-gate-rejected/--skip-quality-gate-rejected"),
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    resolved_output_dir = materialized_output_dir or _default_materialized_output_dir(
        output_dir,
        fusion_channel=fusion_channel,
    )
    _require_empty_or_overwrite(resolved_output_dir, overwrite=overwrite, label="materialized output directory")
    outputs = export_tile_quadrant_materialized_chunks(
        window_json_dir=window_json_dir,
        moving_position_input=coarse_moving_position,
        fixed_registration_input=fixed_registration,
        output_dir=resolved_output_dir,
        channel_source_shift_px_zyx=(
            None if channel_source_shift_px_zyx is None else _parse_float_zyx(channel_source_shift_px_zyx)
        ),
        include_quality_gate_rejected=include_quality_gate_rejected,
    )
    manifest = _update_cross_register_manifest(
        output_dir,
        stage="materialize",
        updates={
            "window_json_dir": window_json_dir.resolve(),
            "coarse_moving_position": coarse_moving_position.resolve(),
            "fixed_registration": fixed_registration.resolve(),
            "fusion_channel": fusion_channel,
            "materialized_output_dir": resolved_output_dir.resolve(),
            "materialized_tile_dir": (resolved_output_dir / "materialized_tiles").resolve(),
            "position": outputs["position"],
            "registration": outputs["registration"],
            "summary": outputs["summary"],
            "channel_source_shift_px_zyx": (
                None if channel_source_shift_px_zyx is None else list(_parse_float_zyx(channel_source_shift_px_zyx))
            ),
            "include_quality_gate_rejected": include_quality_gate_rejected,
        },
    )
    typer.echo(
        json.dumps(
            {
                "position": str(outputs["position"]),
                "registration": str(outputs["registration"]),
                "summary": str(outputs["summary"]),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


@cross_register_method8_app.command("manifest")
def cross_register_method8_manifest(
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    fixed_token: Annotated[str, typer.Option("--fixed-token")] = "Image_14",
    moving_token: Annotated[str, typer.Option("--moving-token")] = "Image_10",
    fixed_channel: Annotated[int, typer.Option("--fixed-channel", min=0)] = 0,
    moving_channel: Annotated[int, typer.Option("--moving-channel", min=0)] = 0,
    fusion_channel: Annotated[int, typer.Option("--fusion-channel", min=0)] = 1,
    coarse_position: Annotated[Path | None, typer.Option("--coarse-position", dir_okay=False)] = None,
    method8_summary: Annotated[Path | None, typer.Option("--method8-summary", dir_okay=False)] = None,
    window_json_dir: Annotated[Path | None, typer.Option("--window-json-dir", file_okay=False)] = None,
    materialized_output_dir: Annotated[Path | None, typer.Option("--materialized-output-dir", file_okay=False)] = None,
    validate: Annotated[bool, typer.Option("--validate/--no-validate")] = True,
) -> None:
    resolved_coarse = coarse_position or _default_cross_register_coarse_position(
        output_dir,
        moving_token=moving_token,
        moving_channel=moving_channel,
        fixed_token=fixed_token,
    )
    resolved_method8_summary = method8_summary or (output_dir / "tile_quadrant_method8_summary.json")
    resolved_window_json_dir = window_json_dir or (output_dir / "window_json")
    resolved_materialized_dir = materialized_output_dir or _default_materialized_output_dir(
        output_dir,
        fusion_channel=fusion_channel,
    )
    materialized_position = resolved_materialized_dir / "tile_quadrant_materialized_chunks.positions.json"
    materialized_registration = resolved_materialized_dir / "tile_quadrant_materialized_chunks.registration.json"
    materialized_summary = resolved_materialized_dir / "tile_quadrant_materialized_chunks.summary.json"
    if validate:
        _require_paths_exist(
            {
                "coarse_position": resolved_coarse,
                "method8_summary": resolved_method8_summary,
                "window_json_dir": resolved_window_json_dir,
                "materialized_position": materialized_position,
                "materialized_registration": materialized_registration,
                "materialized_summary": materialized_summary,
            }
        )
    manifest = _update_cross_register_manifest(
        output_dir,
        stage="manifest",
        updates={
            "fixed_token": fixed_token,
            "moving_token": moving_token,
            "fixed_channel": fixed_channel,
            "moving_channel": moving_channel,
            "fusion_channel": fusion_channel,
            "coarse_position": resolved_coarse.resolve(),
            "method8_summary": resolved_method8_summary.resolve(),
            "window_json_dir": resolved_window_json_dir.resolve(),
            "materialized_output_dir": resolved_materialized_dir.resolve(),
            "materialized_tile_dir": (resolved_materialized_dir / "materialized_tiles").resolve(),
            "materialized_position": materialized_position.resolve(),
            "materialized_registration": materialized_registration.resolve(),
            "materialized_summary": materialized_summary.resolve(),
            "validated": validate,
        },
    )
    typer.echo(manifest)


@app.command()
def position(
    left_dir: Annotated[Path, typer.Option("--left-dir", exists=True, file_okay=False, readable=True)],
    right_dir: Annotated[Path, typer.Option("--right-dir", exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    mode: Annotated[ModeName, typer.Option("--mode")] = "tltr_x_join_center_z_phase",
    overlap_fraction: Annotated[float | None, typer.Option("--overlap-fraction", min=0.0, max=0.999)] = None,
    plot_title: Annotated[str, typer.Option("--plot-title")] = "metadata joined tile positions",
) -> None:
    create_position_file(
        left_dir=left_dir,
        right_dir=right_dir,
        output=output,
        mode=mode,
        overlap_fraction=overlap_fraction,
        plot_title=plot_title,
    )
    typer.echo(output.resolve())


@app.command("rough-phase")
def rough_phase(
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    output_position: Annotated[Path, typer.Option("--output-position")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LR_ALIGNMENT_LEVEL,
    search_margin_px: Annotated[int, typer.Option("--search-margin-px", min=0)] = 64,
    upsample_factor: Annotated[int, typer.Option("--upsample-factor", min=1)] = 10,
    seam_fraction: Annotated[float, typer.Option("--seam-fraction", min=0.0, max=1.0)] = 0.10,
    crop_overlap: Annotated[bool, typer.Option("--crop-overlap/--no-crop-overlap")] = True,
    z_slab_planes: Annotated[int, typer.Option("--z-slab-planes", min=1)] = DEFAULT_Z_SLAB_PLANES,
    phase_downsample_yx: Annotated[int, typer.Option("--phase-downsample-yx", min=1)] = 1,
    phase_downsample_zyx: Annotated[str | None, typer.Option("--phase-downsample-zyx")] = _format_int_zyx(DEFAULT_PHASE_DOWNSAMPLE_ZYX),
) -> None:
    rough_phase_align(
        position_input=position_input,
        output_position=output_position,
        output_dir=output_dir,
        channel=channel,
        level=level,
        search_margin_px=search_margin_px,
        upsample_factor=upsample_factor,
        seam_fraction=seam_fraction,
        crop_overlap=crop_overlap,
        z_slab_planes=z_slab_planes,
        phase_downsample_yx=phase_downsample_yx,
        phase_downsample_zyx=_parse_int_zyx(phase_downsample_zyx) if phase_downsample_zyx is not None else None,
    )
    typer.echo(output_position.resolve())


@app.command()
def register(
    run_dir: Annotated[Path, typer.Argument()],
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    registration_output: Annotated[Path, typer.Option("--registration-output")],
    level: Annotated[int, typer.Option("--level", min=0)] = legacy_registration.DEFAULT_COARSE_REG_RES_LEVELS[0],
    registration_pair_mode: Annotated[str, typer.Option("--registration-pair-mode")] = legacy_registration.DEFAULT_REGISTRATION_PAIR_MODE,
    robust_boundary_qc_dir: Annotated[Path | None, typer.Option("--robust-boundary-qc-dir")] = None,
    registration_plots_dir: Annotated[Path | None, typer.Option("--registration-plots-dir")] = None,
    skip_registration_plots: Annotated[bool, typer.Option("--skip-registration-plots/--registration-plots")] = True,
    dask_num_workers: Annotated[int | None, typer.Option("--dask-num-workers", min=1)] = None,
    pairwise_jobs: Annotated[int | None, typer.Option("--pairwise-jobs", min=0)] = legacy_registration.DEFAULT_N_PARALLEL_PAIRWISE_REGS,
    registration_cache_max_gib: Annotated[
        float,
        typer.Option("--registration-cache-max-gib", min=0.0),
    ] = legacy_registration.DEFAULT_REGISTRATION_CACHE_MAX_GIB,
    registration_pair_file: Annotated[
        Path | None,
        typer.Option("--registration-pair-file", exists=True, dir_okay=False, readable=True),
    ] = None,
    groupwise_transform: Annotated[str, typer.Option("--groupwise-transform")] = legacy_registration.MVS_GROUPWISE_TRANSFORM,
    mvs_post_quality_filter: Annotated[
        bool,
        typer.Option("--mvs-post-quality-filter/--no-mvs-post-quality-filter"),
    ] = True,
    mvs_post_quality_threshold: Annotated[
        float,
        typer.Option("--mvs-post-quality-threshold", min=0.0),
    ] = legacy_registration.MVS_POST_QUALITY_THRESHOLD,
    channel: Annotated[list[int] | None, typer.Option("--channel", min=0)] = None,
    reference_registration_input: Annotated[
        Path | None,
        typer.Option("--reference-registration-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    reference_geometry_mode: Annotated[str, typer.Option("--reference-geometry-mode")] = "none",
    reference_xy_prior_weight: Annotated[float | None, typer.Option("--reference-xy-prior-weight", min=0.0)] = None,
    reference_initial_alignment: Annotated[str, typer.Option("--reference-initial-alignment")] = "none",
    shared_geometry_tracks: Annotated[str | None, typer.Option("--shared-geometry-tracks")] = None,
    log_file: Annotated[Path | None, typer.Option("--log-file")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    register_tiles(
        run_dir=run_dir,
        position_input=position_input,
        registration_output=registration_output,
        level=level,
        registration_pair_mode=registration_pair_mode,
        robust_boundary_qc_dir=robust_boundary_qc_dir,
        registration_plots_dir=registration_plots_dir,
        skip_registration_plots=skip_registration_plots,
        dask_num_workers=dask_num_workers,
        pairwise_jobs=pairwise_jobs,
        registration_cache_max_gib=registration_cache_max_gib,
        registration_pair_file=registration_pair_file,
        groupwise_transform=groupwise_transform,
        mvs_post_quality_filter=mvs_post_quality_filter,
        mvs_post_quality_threshold=mvs_post_quality_threshold,
        reference_registration_input=reference_registration_input,
        reference_geometry_mode=reference_geometry_mode,
        reference_xy_prior_weight=reference_xy_prior_weight,
        reference_initial_alignment=reference_initial_alignment,
        shared_geometry_tracks=(
            tuple(item.strip() for item in shared_geometry_tracks.split(",") if item.strip())
            if shared_geometry_tracks is not None
            else None
        ),
        channels=tuple(channel) if channel is not None else None,
        log_file=log_file,
        dry_run=dry_run,
    )


@app.command("mvs-edge-audit")
def mvs_edge_audit(
    registration_input: Annotated[Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    audit = mvs_used_edge_audit(load_mvs_registration(registration_input))
    text = json.dumps(audit, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        typer.echo(output.resolve())
    else:
        typer.echo(text)


@app.command("mvs-refine-level0")
def mvs_refine_level0(
    registration_input: Annotated[Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)],
    output_registration: Annotated[Path, typer.Option("--output-registration")],
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    patch_shape_zyx: Annotated[str, typer.Option("--patch-shape-zyx")] = "12,320,320",
    span_entire_seam: Annotated[bool, typer.Option("--span-entire-seam/--scout-patches")] = False,
    patches_per_edge: Annotated[int, typer.Option("--patches-per-edge", min=1)] = DEFAULT_LEVEL0_PATCHES_PER_EDGE,
    retry_patches_per_edge: Annotated[int, typer.Option("--retry-patches-per-edge", min=1)] = DEFAULT_LEVEL0_RETRY_PATCHES_PER_EDGE,
    min_inliers: Annotated[int, typer.Option("--min-inliers", min=1)] = 3,
    max_phase_shift_zyx: Annotated[str, typer.Option("--max-phase-shift-zyx")] = "3,64,64",
    phase_highpass_sigma_zyx: Annotated[str, typer.Option("--phase-highpass-sigma-zyx")] = "0,10,10",
    phase_upsample_factor: Annotated[int, typer.Option("--phase-upsample-factor", min=1)] = 10,
    max_correction_zyx: Annotated[str, typer.Option("--max-correction-zyx")] = "4,64,64",
    max_final_residual_zyx: Annotated[str, typer.Option("--max-final-residual-zyx")] = "2,8,8",
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    used_edges_only: Annotated[bool, typer.Option("--used-edges-only/--all-edges")] = True,
    min_quality: Annotated[float, typer.Option("--min-quality")] = 0.25,
    max_edges: Annotated[int | None, typer.Option("--max-edges", min=1)] = None,
    max_disconnected_island_size: Annotated[
        int,
        typer.Option("--max-disconnected-island-size", min=0),
    ] = DEFAULT_LEVEL0_MAX_DISCONNECTED_ISLAND_SIZE,
    fallback_refinement_level: Annotated[
        list[int] | None,
        typer.Option(
            "--fallback-refinement-level",
            min=1,
            help="Retry native-level rejected seams at this pyramid level before level-2 fallback. May be passed multiple times.",
        ),
    ] = None,
    fallback_level2_weight_scale: Annotated[
        float,
        typer.Option("--fallback-level2-weight-scale", min=0.0),
    ] = DEFAULT_LEVEL0_FALLBACK_LEVEL2_WEIGHT_SCALE,
    contact_sheet: Annotated[bool, typer.Option("--contact-sheet/--no-contact-sheet")] = True,
    contact_sheet_output_dir: Annotated[Path | None, typer.Option("--contact-sheet-output-dir", file_okay=False)] = None,
    contact_sheet_max_panels: Annotated[int, typer.Option("--contact-sheet-max-panels", min=1)] = 128,
) -> None:
    refined, _diagnostics = refine_mvs_registration_level0(
        registration_input=registration_input,
        output_registration=output_registration,
        channel=channel,
        patch_shape_zyx=_parse_int_zyx(patch_shape_zyx),
        candidate_mode="seam-span" if span_entire_seam else "scout",
        patches_per_edge=patches_per_edge,
        retry_patches_per_edge=retry_patches_per_edge,
        min_inliers=min_inliers,
        max_phase_shift_zyx=_parse_float_zyx(max_phase_shift_zyx),
        phase_highpass_sigma_zyx=_parse_float_zyx(phase_highpass_sigma_zyx),
        phase_upsample_factor=phase_upsample_factor,
        max_correction_zyx=_parse_float_zyx(max_correction_zyx),
        max_final_residual_zyx=_parse_float_zyx(max_final_residual_zyx),
        workers=workers,
        used_edges_only=used_edges_only,
        min_quality=min_quality,
        max_edges=max_edges,
        max_disconnected_island_size=max_disconnected_island_size,
        fallback_refinement_levels=tuple(fallback_refinement_level or ()),
        fallback_level2_weight_scale=fallback_level2_weight_scale,
        render_contact_sheet=contact_sheet,
        contact_sheet_output_dir=contact_sheet_output_dir,
        contact_sheet_max_panels=contact_sheet_max_panels,
        progress=_log_progress,
    )
    typer.echo(refined["metrics"]["level0_refinement"]["output_registration"])


@app.command("rechunk-ome-tiff")
def rechunk_ome_tiff(
    inputs: Annotated[list[Path], typer.Argument()],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    chunk_shape_zyx: Annotated[str, typer.Option("--chunk-shape-zyx")] = "12,240,240",
    pyramid_downsample_factors: Annotated[str, typer.Option("--pyramid-downsample-factors")] = "2,4",
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    summary_output: Annotated[Path | None, typer.Option("--summary-output", dir_okay=False)] = None,
) -> None:
    summary = rechunk_ome_tiffs(
        inputs=inputs,
        output_dir=output_dir,
        chunk_shape_zyx=_parse_int_zyx(chunk_shape_zyx),
        pyramid_downsample_factors=tuple(
            int(value.strip()) for value in pyramid_downsample_factors.split(",") if value.strip()
        ),
        overwrite=overwrite,
        workers=workers,
        progress=_log_progress,
    )
    text = json.dumps(summary, indent=2)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(text + "\n")
        typer.echo(summary_output.resolve())
    else:
        typer.echo(text)


@app.command("mvs-score-gradient-ncc")
def mvs_score_gradient_ncc(
    registration_input: Annotated[Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    summary_output: Annotated[Path | None, typer.Option("--summary-output")] = None,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = 2,
    patch_shape_yx: Annotated[str, typer.Option("--patch-shape-yx")] = "256,256",
    phase_patch_shape_zyx: Annotated[str, typer.Option("--phase-patch-shape-zyx")] = "32,256,256",
    min_gradient_ncc: Annotated[float, typer.Option("--min-gradient-ncc")] = 0.15,
    max_phase_shift_native_zyx: Annotated[str, typer.Option("--max-phase-shift-native-zyx")] = "16,96,96",
    phase_refine_bad_gradients: Annotated[
        bool,
        typer.Option("--phase-refine-bad-gradients/--no-phase-refine-bad-gradients"),
    ] = False,
    max_cached_tiles: Annotated[int, typer.Option("--max-cached-tiles", min=1)] = 2,
    used_edges_only: Annotated[bool, typer.Option("--used-edges-only/--all-measured-edges")] = True,
) -> None:
    patch_shape = tuple(int(value) for value in patch_shape_yx.split(","))
    if len(patch_shape) != 2 or any(value < 1 for value in patch_shape):
        raise typer.BadParameter(f"Expected comma-separated positive y,x patch shape, got {patch_shape_yx!r}")
    phase_patch_shape = tuple(int(value) for value in phase_patch_shape_zyx.split(","))
    if len(phase_patch_shape) != 3 or any(value < 1 for value in phase_patch_shape):
        raise typer.BadParameter(
            f"Expected comma-separated positive z,y,x phase patch shape, got {phase_patch_shape_zyx!r}"
        )
    max_phase_shift = tuple(float(value) for value in max_phase_shift_native_zyx.split(","))
    if len(max_phase_shift) != 3 or any(value < 0 for value in max_phase_shift):
        raise typer.BadParameter(
            f"Expected comma-separated non-negative z,y,x max shift, got {max_phase_shift_native_zyx!r}"
        )
    payload, summary = score_mvs_edges_with_gradient_ncc(
        load_mvs_registration(registration_input),
        channel=channel,
        level=level,
        patch_shape_yx=(patch_shape[0], patch_shape[1]),
        phase_patch_shape_zyx=(phase_patch_shape[0], phase_patch_shape[1], phase_patch_shape[2]),
        min_gradient_ncc=min_gradient_ncc,
        max_phase_shift_native_zyx=(max_phase_shift[0], max_phase_shift[1], max_phase_shift[2]),
        phase_refine_bad_gradients=phase_refine_bad_gradients,
        max_cached_tiles=max_cached_tiles,
        used_edges_only=used_edges_only,
        progress=_log_progress,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(output.resolve())


@app.command()
def fuse(
    input_dir: Annotated[Path, typer.Argument()],
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    registration_input: Annotated[Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    channel: Annotated[list[int] | None, typer.Option("--channel", min=0)] = None,
    flatfield_dir_by_source_view: Annotated[
        list[str] | None,
        typer.Option("--flatfield-dir-by-source-view"),
    ] = None,
    fusion_level: Annotated[int, typer.Option("--fusion-level", min=0)] = 0,
    fusion_weight_mode: Annotated[
        str,
        typer.Option("--fusion-weight-mode"),
    ] = "content-preibisch-coarse",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    basic_cache_disk_dir: Annotated[
        Path | None,
        typer.Option("--basic-cache-disk-dir", file_okay=False),
    ] = None,
    output_chunksize_zyx: Annotated[
        str | None,
        typer.Option("--output-chunksize-zyx", help="Fusion output chunk size as z,y,x."),
    ] = None,
    output_grid_template: Annotated[
        Path | None,
        typer.Option("--output-grid-template", exists=True, file_okay=False, help="OME-Zarr whose scale-0 grid is reused."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    fuse_tiles(
        input_dir=input_dir,
        position_input=position_input,
        registration_input=registration_input,
        output=output,
        channels=channel,
        flatfield_dirs_by_source_view=_parse_source_view_flatfield_dirs(flatfield_dir_by_source_view),
        fusion_level=fusion_level,
        fusion_weight_mode=fusion_weight_mode,
        batch_size=batch_size,
        basic_cache_disk_dir=basic_cache_disk_dir,
        output_chunksize_zyx=(
            None if output_chunksize_zyx is None else _parse_int_zyx(output_chunksize_zyx, "--output-chunksize-zyx")
        ),
        output_grid_template=output_grid_template,
        dry_run=dry_run,
    )


@app.command("subtract-channel")
def subtract_channel(
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    output_position: Annotated[Path | None, typer.Option("--output-position")] = None,
    registration_input: Annotated[
        Path | None,
        typer.Option("--registration-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    output_registration: Annotated[Path | None, typer.Option("--output-registration")] = None,
    target_channel: Annotated[int, typer.Option("--target-channel", min=0)] = 2,
    reference_channel: Annotated[int, typer.Option("--reference-channel", min=0)] = 3,
    source_level: Annotated[int, typer.Option("--source-level", min=0)] = 2,
    reference_shift_zyx_px: Annotated[str, typer.Option("--reference-shift-zyx-px")] = "0,0,0",
    alpha: Annotated[float, typer.Option("--alpha")] = 0.0,
    beta: Annotated[float, typer.Option("--beta")] = 0.0,
    target_background: Annotated[float, typer.Option("--target-background")] = 0.0,
    reference_background: Annotated[float, typer.Option("--reference-background")] = 0.0,
    crop_yx_px: Annotated[int, typer.Option("--crop-yx-px", min=0)] = 20,
    z_chunk: Annotated[int, typer.Option("--z-chunk", min=1)] = 64,
    compression: Annotated[int | None, typer.Option("--compression")] = DEFAULT_SUBTRACTION_COMPRESSION,
    compression_level: Annotated[float | None, typer.Option("--compression-level")] = (
        DEFAULT_SUBTRACTION_COMPRESSION_LEVEL
    ),
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    limit_tiles: Annotated[int | None, typer.Option("--limit-tiles", min=1)] = None,
) -> None:
    """Write spillover-subtracted tiles for later BaSiC correction and fusion.

    The command is a producer stage: it reads raw target/reference channels from
    each source tile, shifts the reference into target local coordinates, applies
    the subtraction model, crops y/x borders, and writes corrected single-channel
    OME-TIFFs plus position/registration metadata. Downstream fusion treats the
    output as an ordinary single-channel acquisition.
    """
    result = subtract_channel_tiles(
        position_input=position_input,
        output_dir=output_dir,
        output_position=output_position,
        registration_input=registration_input,
        output_registration=output_registration,
        target_channel=target_channel,
        reference_channel=reference_channel,
        source_level=source_level,
        reference_shift_zyx_px=_parse_float_zyx(reference_shift_zyx_px),
        alpha=alpha,
        beta=beta,
        target_background=target_background,
        reference_background=reference_background,
        crop_yx_px=crop_yx_px,
        z_chunk=z_chunk,
        compression=compression,
        compression_level=compression_level,
        overwrite=overwrite,
        limit_tiles=limit_tiles,
        progress=_log_progress,
    )
    typer.echo(result.position_output.resolve())
    if result.registration_output is not None:
        typer.echo(result.registration_output.resolve())


@app.command()
def pyramid(
    ome_zarr: Annotated[list[Path], typer.Argument()],
    template: Annotated[Path | None, typer.Option("--template", exists=True, file_okay=False)] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    add_pyramids(ome_zarrs=ome_zarr, template=template, dry_run=dry_run)


@app.command()
def qc(
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    registration_input: Annotated[Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    center_y_xz: Annotated[bool, typer.Option("--center-y-xz/--no-center-y-xz")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    render_registration_qc(
        position_input=position_input,
        registration_input=registration_input,
        output_dir=output_dir,
        channel=channel,
        level=level,
        center_y_xz=center_y_xz,
        dry_run=dry_run,
    )


@app.command("fused-tile-index-qc")
def fused_tile_index_qc(
    fused_zarr: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    registration_input: Annotated[Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    level: Annotated[int, typer.Option("--level", min=0)] = 2,
    z_index: Annotated[int | None, typer.Option("--z-index", min=0)] = None,
) -> None:
    path = render_fused_tile_index_overlay(
        fused_zarr=fused_zarr,
        registration_input=registration_input,
        output=output,
        level=level,
        z_index=z_index,
    )
    typer.echo(path.resolve())


@app.command("live-fusion-preview")
def live_fusion_preview(
    source_zarr: Annotated[Path | None, typer.Option("--zarr", exists=True, file_okay=False, readable=True)] = None,
    log_path: Annotated[Path | None, typer.Option("--log", exists=True, dir_okay=False, readable=True)] = None,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
    level: Annotated[int, typer.Option("--level", min=0)] = 0,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    color: Annotated[Literal["red", "green", "blue", "gray"], typer.Option("--color")] = "red",
    stride: Annotated[int, typer.Option("--stride", min=1)] = 10,
    z_start: Annotated[int, typer.Option("--z-start", min=0)] = 6,
    z_step: Annotated[int, typer.Option("--z-step", min=1)] = 24,
    panels: Annotated[int, typer.Option("--panels", min=1)] = 6,
    high_percentile: Annotated[float, typer.Option("--high-percentile", min=0.000001, max=100.0)] = 99.7,
) -> None:
    if source_zarr is None and log_path is None:
        raise typer.BadParameter("Pass either --zarr or --log.")
    if source_zarr is not None and log_path is not None:
        raise typer.BadParameter("Pass only one of --zarr or --log.")
    result = render_live_fusion_preview(
        source_zarr=source_zarr,
        log_path=log_path,
        output=output,
        output_dir=output_dir,
        level=level,
        channel=channel,
        color=color,
        stride=stride,
        z_start=z_start,
        z_step=z_step,
        max_panels=panels,
        high_percentile=high_percentile,
    )
    typer.echo(result.output)


@app.command("fused-xyz-overlay-qc")
def fused_xyz_overlay_qc(
    reference_zarr: Annotated[Path, typer.Option("--reference-zarr", exists=True, file_okay=False, readable=True)],
    moving_zarr: Annotated[Path, typer.Option("--moving-zarr", exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    level: Annotated[int, typer.Option("--level", min=0)] = 0,
    thumb_level: Annotated[int, typer.Option("--thumb-level", min=0)] = 2,
    panel_size: Annotated[int, typer.Option("--panel-size", min=1)] = 512,
) -> None:
    path = render_fused_xyz_overlay_qc(
        reference_zarr=reference_zarr,
        moving_zarr=moving_zarr,
        output_dir=output_dir,
        level=level,
        thumb_level=thumb_level,
        panel_size=panel_size,
    )
    typer.echo(path.resolve())


@app.command("registration-center-z-spotcheck")
def registration_center_z_spotcheck(
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    registration_input: Annotated[Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    channel: Annotated[list[int] | None, typer.Option("--channel", min=0)] = None,
    level: Annotated[int, typer.Option("--level", min=0)] = 4,
    center_z_um: Annotated[float | None, typer.Option("--center-z-um")] = None,
    side: Annotated[str, typer.Option("--side")] = "L",
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    paths = render_registration_center_z_spotcheck(
        position_input=position_input,
        registration_input=registration_input,
        output_dir=output_dir,
        channels=channel,
        level=level,
        center_z_um=center_z_um,
        side=side,
        dry_run=dry_run,
    )
    for path in paths:
        typer.echo(path.resolve())


@app.command("track-z-diagnostics")
def track_z_diagnostics(
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    reference_channel: Annotated[int, typer.Option("--reference-channel", min=0)] = 3,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    max_z_shift_px: Annotated[int, typer.Option("--max-z-shift-px", min=0)] = 8,
    min_score: Annotated[float, typer.Option("--min-score")] = 0.05,
    min_voxels: Annotated[int, typer.Option("--min-voxels", min=1)] = 2048,
    smooth_sigma_tiles: Annotated[float, typer.Option("--smooth-sigma-tiles", min=0.001)] = 1.5,
    outlier_mad: Annotated[float, typer.Option("--outlier-mad", min=0.001)] = 4.0,
) -> None:
    summary = run_track_z_diagnostics(
        position_input=position_input,
        output_dir=output_dir,
        reference_channel=reference_channel,
        level=level,
        max_z_shift_px=max_z_shift_px,
        min_score=min_score,
        min_voxels=min_voxels,
        smooth_sigma_tiles=smooth_sigma_tiles,
        outlier_mad=outlier_mad,
    )
    typer.echo(summary)


@app.command("align-lr-dumb-stitch")
def align_lr_dumb_stitch(
    left_dir: Annotated[Path, typer.Option("--left-dir", exists=True, file_okay=False, readable=True)],
    right_dir: Annotated[Path, typer.Option("--right-dir", exists=True, file_okay=False, readable=True)],
    output_prefix: Annotated[Path, typer.Option("--output-prefix")],
    overlap_fraction: Annotated[float, typer.Option("--overlap-fraction", min=0.0, max=0.999)] = DEFAULT_OVERLAP_FRACTION,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LR_ALIGNMENT_LEVEL,
    search_margin_px: Annotated[int, typer.Option("--search-margin-px", min=0)] = 64,
    phase_upsample_factor: Annotated[int, typer.Option("--phase-upsample-factor", min=1)] = 10,
    seam_fraction: Annotated[float, typer.Option("--seam-fraction", min=0.0, max=1.0)] = 0.10,
    z_slab_planes: Annotated[int, typer.Option("--z-slab-planes", min=1)] = DEFAULT_Z_SLAB_PLANES,
    phase_downsample_zyx: Annotated[str, typer.Option("--phase-downsample-zyx")] = _format_int_zyx(DEFAULT_PHASE_DOWNSAMPLE_ZYX),
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    paths = run_lr_dumb_stitch_alignment(
        left_dir=left_dir,
        right_dir=right_dir,
        output_prefix=output_prefix,
        overlap_fraction=overlap_fraction,
        channel=channel,
        level=level,
        search_margin_px=search_margin_px,
        phase_upsample_factor=phase_upsample_factor,
        seam_fraction=seam_fraction,
        z_slab_planes=z_slab_planes,
        phase_downsample_zyx=_parse_int_zyx(phase_downsample_zyx),
        dry_run=dry_run,
    )
    typer.echo(paths.summary_json.resolve())


@app.command("tile-phase-align")
def tile_phase_align(
    reference_position: Annotated[Path, typer.Option("--reference-position", exists=True, dir_okay=False, readable=True)],
    output_position: Annotated[Path, typer.Option("--output-position")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    reference_channel: Annotated[int, typer.Option("--reference-channel", min=0)] = 3,
    moving_channel: Annotated[int, typer.Option("--moving-channel", min=0)] = 0,
    reference_token: Annotated[str, typer.Option("--reference-token")] = "488514561638",
    moving_token: Annotated[str, typer.Option("--moving-token")] = "405",
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    upsample_factor: Annotated[int, typer.Option("--upsample-factor", min=1)] = 10,
    patch_shape_zyx: Annotated[str | None, typer.Option("--patch-shape-zyx")] = None,
    min_inliers: Annotated[int, typer.Option("--min-inliers", min=3)] = 3,
    max_candidate_patches: Annotated[int, typer.Option("--max-candidate-patches", min=1)] = 24,
    coarse_level: Annotated[int, typer.Option("--coarse-level", min=0)] = DEFAULT_LEVEL,
    scout_z_samples: Annotated[int | None, typer.Option("--scout-z-samples", min=1)] = 32,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    reference_registration_input: Annotated[
        Path | None,
        typer.Option("--reference-registration-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    output_registration: Annotated[Path | None, typer.Option("--output-registration")] = None,
    alignment_mode: Annotated[
        Literal["phase", "affine"],
        typer.Option("--alignment-mode", help="Cross-channel tile registration mode."),
    ] = "phase",
    affine_init_level: Annotated[int, typer.Option("--affine-init-level", min=0)] = 2,
    affine_init_z_samples: Annotated[int, typer.Option("--affine-init-z-samples", min=1)] = 200,
    affine_refine_crop_shape_zyx: Annotated[str, typer.Option("--affine-refine-crop-shape-zyx")] = "200,480,480",
    affine_max_iterations: Annotated[int, typer.Option("--affine-max-iterations", min=0)] = 20,
    affine_contact_sheet: Annotated[bool, typer.Option("--affine-contact-sheet/--no-affine-contact-sheet")] = True,
    affine_fit_mode: Annotated[
        Literal["rigid", "affine-12dof"],
        typer.Option("--affine-fit-mode", help="Transform family for affine-mode cross-channel registration."),
    ] = "rigid",
    affine_tile_order: Annotated[
        Literal["input", "grid-fanout"],
        typer.Option("--affine-tile-order", help="Tile processing order for affine mode."),
    ] = "grid-fanout",
    affine_running_average_min_inliers: Annotated[
        int,
        typer.Option("--affine-running-average-min-inliers", min=1),
    ] = 5,
) -> None:
    if alignment_mode == "affine":
        if (output_registration is None) != (reference_registration_input is None):
            raise typer.BadParameter(
                "affine alignment mode requires both --reference-registration-input and --output-registration "
                "when writing a registration"
            )
        path = align_tiles_to_reference_affine(
            reference_position=reference_position,
            output_position=output_position,
            output_dir=output_dir,
            reference_channel=reference_channel,
            moving_channel=moving_channel,
            reference_token=reference_token,
            moving_token=moving_token,
            init_level=affine_init_level,
            init_z_samples=affine_init_z_samples,
            refine_crop_shape_zyx=parse_shape_zyx(affine_refine_crop_shape_zyx),
            max_iterations=affine_max_iterations,
            render_contact_sheet=affine_contact_sheet,
            fit_mode=affine_fit_mode,
            tile_order=affine_tile_order,
            running_average_min_inliers=affine_running_average_min_inliers,
            reference_registration_input=reference_registration_input,
            output_registration=output_registration,
            progress=_log_progress,
        )
    else:
        path = align_tiles_to_reference(
            reference_position=reference_position,
            output_position=output_position,
            output_dir=output_dir,
            reference_channel=reference_channel,
            moving_channel=moving_channel,
            reference_token=reference_token,
            moving_token=moving_token,
            level=level,
            upsample_factor=upsample_factor,
            patch_shape_zyx=None if patch_shape_zyx is None else parse_shape_zyx(patch_shape_zyx),
            min_inliers=min_inliers,
            max_candidate_patches=max_candidate_patches,
            coarse_level=coarse_level,
            scout_z_samples=scout_z_samples,
            workers=workers,
            reference_registration_input=reference_registration_input,
            output_registration=output_registration,
        )
    typer.echo(path)


@align_405_to_488_app.command("refine-level0")
def align_405_to_488_refine_level0(
    coarse_json: Annotated[Path, typer.Option("--coarse-json", exists=True, dir_okay=False, readable=True)],
    anchor_table: Annotated[Path, typer.Option("--anchor-table", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    native_patch_shape_zyx: Annotated[str, typer.Option("--native-patch-shape-zyx")] = "12,320,320",
    upsample_factor: Annotated[int, typer.Option("--upsample-factor", min=1)] = 10,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    script_dir: Annotated[Path, typer.Option("--script-dir", exists=True, file_okay=False, readable=True)] = ...,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    command = refine_405_488_level0_anchors(
        script_dir=script_dir,
        coarse_json=coarse_json,
        anchor_table=anchor_table,
        output_dir=output_dir,
        native_patch_shape_zyx=parse_shape_zyx(native_patch_shape_zyx),
        upsample_factor=upsample_factor,
        workers=workers,
        dry_run=dry_run,
    )
    typer.echo(command)


@align_405_to_488_app.command("measure-stage3-mattes")
def align_405_to_488_measure_stage3_mattes(
    records_jsonl: Annotated[Path, typer.Option("--records-jsonl", exists=True, dir_okay=False, readable=True)],
    reference_position: Annotated[
        Path,
        typer.Option("--reference-position", exists=True, dir_okay=False, readable=True),
    ],
    moving_position: Annotated[Path, typer.Option("--moving-position", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    preserve_moving_geometry: Annotated[
        bool,
        typer.Option(
            "--preserve-moving-geometry/--canonicalize-moving-geometry",
            help="Use moving position-file geometry as-is, or snap moving tile origins to reference sites.",
        ),
    ] = True,
    candidate_source: Annotated[
        Literal["seed_rows", "structure_tensor"],
        typer.Option("--candidate-source", help="Use seed_rows for current workflow fidelity, or structure_tensor."),
    ] = "seed_rows",
    level: Annotated[int, typer.Option("--level", min=0)] = 3,
    patch_shape_zyx: Annotated[str, typer.Option("--patch-shape-zyx")] = "12,160,160",
    reference_channel: Annotated[int, typer.Option("--reference-channel", min=0)] = 3,
    moving_channel: Annotated[int, typer.Option("--moving-channel", min=0)] = 0,
    max_attempted_chunks: Annotated[int, typer.Option("--max-attempted-chunks", min=1)] = 24,
    min_inlier_chunks: Annotated[int, typer.Option("--min-inlier-chunks", min=1)] = 3,
    max_inlier_chunks: Annotated[int, typer.Option("--max-inlier-chunks", min=1)] = 5,
    min_good_mip_gradient_ncc: Annotated[float, typer.Option("--min-good-mip-gradient-ncc")] = 0.15,
    high_frequency_content_sigma_zyx: Annotated[
        str,
        typer.Option("--high-frequency-content-sigma-zyx"),
    ] = "0,3,3",
    min_high_frequency_content_score: Annotated[
        float,
        typer.Option("--min-high-frequency-content-score"),
    ] = 0.001,
    candidate_pool_size: Annotated[int, typer.Option("--candidate-pool-size", min=1)] = 36,
    edge_inset_fraction: Annotated[float, typer.Option("--edge-inset-fraction", min=0.0, max=0.49)] = 0.0,
    inlier_threshold_level_px_zyx: Annotated[str, typer.Option("--inlier-threshold-level-px-zyx")] = "3,12,12",
) -> None:
    output = measure_405_to_488_mattes_anchors(
        records_jsonl=records_jsonl,
        reference_position=reference_position,
        moving_position=moving_position,
        output_dir=output_dir,
        preserve_moving_geometry=preserve_moving_geometry,
        parameters=MattesAnchorParameters(
            candidate_source=candidate_source,
            level=level,
            patch_shape_zyx=parse_shape_zyx(patch_shape_zyx),
            reference_channel=reference_channel,
            moving_channel=moving_channel,
            max_attempted_chunks=max_attempted_chunks,
            min_inlier_chunks=min_inlier_chunks,
            max_inlier_chunks=max_inlier_chunks,
            min_good_mip_gradient_ncc=min_good_mip_gradient_ncc,
            high_frequency_content_sigma_zyx=_parse_float_zyx(high_frequency_content_sigma_zyx),
            min_high_frequency_content_score=min_high_frequency_content_score,
            candidate_pool_size=candidate_pool_size,
            edge_inset_fraction=edge_inset_fraction,
            inlier_threshold_level_px_zyx=_parse_float_zyx(inlier_threshold_level_px_zyx),
        ),
    )
    typer.echo(output.resolve())


@align_405_to_488_app.command("recover-mvs-seams")
def align_405_to_488_recover_mvs_seams(
    level0_refined_json: Annotated[
        Path,
        typer.Option("--level0-refined-json", exists=True, dir_okay=False, readable=True),
    ],
    mvs_registration: Annotated[
        Path,
        typer.Option("--mvs-registration", exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    command = recover_405_488_mvs_seam_anchors(
        level0_refined_json=level0_refined_json,
        mvs_registration=mvs_registration,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    typer.echo(command)


@align_405_to_488_app.command("optimize")
def align_405_to_488_optimize(
    anchor_table: Annotated[Path, typer.Option("--anchor-table", exists=True, dir_okay=False, readable=True)],
    phase_position: Annotated[Path, typer.Option("--phase-position", exists=True, dir_okay=False, readable=True)],
    source_405_registration: Annotated[
        Path,
        typer.Option("--source-405-registration", exists=True, dir_okay=False, readable=True),
    ],
    seam_residuals: Annotated[Path, typer.Option("--seam-residuals", exists=True, dir_okay=False, readable=True)],
    output_position: Annotated[Path, typer.Option("--output-position")],
    output_registration: Annotated[Path, typer.Option("--output-registration")],
    diagnostics: Annotated[Path, typer.Option("--diagnostics")],
    seam_overrides: Annotated[
        Path | None,
        typer.Option("--seam-overrides", exists=True, dir_okay=False, readable=True),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    command = optimize_405_to_488_translations(
        anchor_table=anchor_table,
        phase_position=phase_position,
        source_405_registration=source_405_registration,
        seam_residuals=seam_residuals,
        seam_overrides=seam_overrides,
        output_position=output_position,
        output_registration=output_registration,
        diagnostics=diagnostics,
        dry_run=dry_run,
    )
    typer.echo(command)


@align_405_to_488_app.command("render-candidate-grid")
def align_405_to_488_render_candidate_grid(
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    registration_input: Annotated[
        Path,
        typer.Option("--registration-input", exists=True, dir_okay=False, readable=True),
    ],
    candidate_json: Annotated[Path, typer.Option("--candidate-json", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = 4,
    render_jobs: Annotated[int, typer.Option("--render-jobs", min=1)] = 1,
    crop_fraction_yx: Annotated[str, typer.Option("--crop-fraction-yx")] = "0.5,0.5",
    center_y_xz: Annotated[bool, typer.Option("--center-y-xz/--no-center-y-xz")] = False,
) -> None:
    summary = render_candidate_grid(
        position_input=position_input,
        registration_input=registration_input,
        candidate_json=candidate_json,
        output_dir=output_dir,
        channel=channel,
        level=level,
        render_jobs=render_jobs,
        crop_fraction_yx=_parse_float_yx(crop_fraction_yx),
        center_y_xz=center_y_xz,
    )
    typer.echo(summary.resolve())


@align_405_to_488_app.command("run")
def align_405_to_488_run(
    all_tiles_json: Annotated[Path, typer.Option("--all-tiles-json", exists=True, dir_okay=False, readable=True)],
    coarse_anchor_table: Annotated[
        Path,
        typer.Option("--coarse-anchor-table", exists=True, dir_okay=False, readable=True),
    ],
    phase_position: Annotated[Path, typer.Option("--phase-position", exists=True, dir_okay=False, readable=True)],
    source_405_registration: Annotated[
        Path,
        typer.Option("--source-405-registration", exists=True, dir_okay=False, readable=True),
    ],
    seam_residuals: Annotated[Path, typer.Option("--seam-residuals", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    script_dir: Annotated[Path, typer.Option("--script-dir", exists=True, file_okay=False, readable=True)] = ...,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    commands = run_405_to_488_workflow(
        script_dir=script_dir,
        all_tiles_json=all_tiles_json,
        coarse_anchor_table=coarse_anchor_table,
        phase_position=phase_position,
        source_405_registration=source_405_registration,
        seam_residuals=seam_residuals,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    for name, command in commands.items():
        typer.echo(f"{name}: {command}")


@app.command("run-tltr")
def run_tltr(
    left_dir: Annotated[Path, typer.Option("--left-dir", exists=True, file_okay=False, readable=True)],
    right_dir: Annotated[Path, typer.Option("--right-dir", exists=True, file_okay=False, readable=True)],
    output_prefix: Annotated[Path, typer.Option("--output-prefix")],
    overlap_fraction: Annotated[float, typer.Option("--overlap-fraction", min=0.0, max=0.999)] = DEFAULT_OVERLAP_FRACTION,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    rough_phase_level: Annotated[int, typer.Option("--rough-phase-level", min=0)] = DEFAULT_LR_ALIGNMENT_LEVEL,
    search_margin_px: Annotated[int, typer.Option("--search-margin-px", min=0)] = 64,
    phase_upsample_factor: Annotated[int, typer.Option("--phase-upsample-factor", min=1)] = 10,
    seam_fraction: Annotated[float, typer.Option("--seam-fraction", min=0.0, max=1.0)] = 0.10,
    z_slab_planes: Annotated[int, typer.Option("--z-slab-planes", min=1)] = DEFAULT_Z_SLAB_PLANES,
    phase_downsample_zyx: Annotated[str, typer.Option("--phase-downsample-zyx")] = _format_int_zyx(DEFAULT_PHASE_DOWNSAMPLE_ZYX),
    registration_pair_mode: Annotated[str, typer.Option("--registration-pair-mode")] = legacy_registration.DEFAULT_REGISTRATION_PAIR_MODE,
    registration_pair_file: Annotated[
        Path | None,
        typer.Option("--registration-pair-file", exists=True, dir_okay=False, readable=True),
    ] = None,
    skip_registration_plots: Annotated[bool, typer.Option("--skip-registration-plots/--registration-plots")] = True,
    dask_registration_workers: Annotated[int | None, typer.Option("--dask-registration-workers", min=1)] = None,
    pairwise_jobs: Annotated[int | None, typer.Option("--pairwise-jobs", min=0)] = legacy_registration.DEFAULT_N_PARALLEL_PAIRWISE_REGS,
    fuse_output: Annotated[bool, typer.Option("--fuse/--no-fuse")] = False,
    pyramid_output: Annotated[bool, typer.Option("--pyramid/--no-pyramid")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    paths = run_tltr_workflow(
        left_dir=left_dir,
        right_dir=right_dir,
        output_prefix=output_prefix,
        overlap_fraction=overlap_fraction,
        channel=channel,
        level=level,
        rough_phase_level=rough_phase_level,
        search_margin_px=search_margin_px,
        phase_upsample_factor=phase_upsample_factor,
        seam_fraction=seam_fraction,
        z_slab_planes=z_slab_planes,
        phase_downsample_zyx=_parse_int_zyx(phase_downsample_zyx),
        registration_pair_mode=registration_pair_mode,
        registration_pair_file=registration_pair_file,
        skip_registration_plots=skip_registration_plots,
        dask_registration_workers=dask_registration_workers,
        pairwise_jobs=pairwise_jobs,
        do_fuse=fuse_output,
        do_pyramid=pyramid_output,
        dry_run=dry_run,
    )
    typer.echo(paths.summary_json.resolve())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
