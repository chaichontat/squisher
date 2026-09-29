from __future__ import annotations

import json
import shutil
from pathlib import Path
import subprocess
import sys
from typing import Annotated, Literal

import typer

from squisher.jpegxr_zarr import DEFAULT_JPEGXR_LEVEL, register_jpegxr_codec
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
    DEFAULT_CHUNK_SHAPE_ZYX as DEFAULT_SUBTRACTION_CHUNK_SHAPE_ZYX,
    DEFAULT_ZSTD_LEVEL as DEFAULT_SUBTRACTION_ZSTD_LEVEL,
    join_paired_subtraction_sides,
    subtract_channel_tiles,
    subtract_paired_channel_tiles,
)
from squisher_lightsheet.candidate_grid import render_candidate_grid
from squisher_lightsheet.channel_affine import (
    apply_channel_affine_registration,
    align_tiles_to_reference_affine,
    write_global_channel_affine_registration,
)
from squisher_lightsheet.constrained_channel_affine import (
    write_constrained_tile_channel_affine_registration,
)
from squisher_lightsheet.complete_recovery import (
    CompleteRecoverySettings,
    DEFAULT_COMPLETE_RECOVERY_SETTINGS,
    complete_recovery,
)
from squisher_lightsheet.cross_register_method8 import (
    DEFAULT_LIB_DIR as DEFAULT_CROSS_REGISTER_METHOD8_LIB_DIR,
    run_tile_quadrant_method8,
)
from squisher_lightsheet.fusion import DEFAULT_OUTPUT_CHUNKSIZE_ZYX, fuse_tiles, parse_fusion_batch_size
from squisher_lightsheet.fused_subtraction import (
    DEFAULT_QUANTILE as DEFAULT_SUBTRACTION_QUANTILE,
    fit_fused_subtraction,
    subtract_fused_volumes,
)
from squisher_lightsheet.fused_fixed import (
    DEFAULT_STARTING_AFFINE_MATRIX_ZYX,
    DEFAULT_SWEEP_RUNNER,
    normalize_starting_affine_matrix_zyx,
    run_fused_fixed_method6,
)
from squisher_lightsheet.fused_fixed_recovery import (
    DEFAULT_MAXIMUM_OUTLIER_PX as DEFAULT_NATIVE_RECOVERY_MAXIMUM_OUTLIER_PX,
    DEFAULT_MAXIMUM_REFIT_DISPLACEMENT_PX,
    DEFAULT_NATIVE_DEVICE_SPEC,
    recover_native,
)
from squisher_lightsheet.fused_fixed_qc import (
    DEFAULT_INTERPOLATED_QC_COLUMNS,
    render_interpolated_window_qc,
    render_prefusion_qc,
)
from squisher_lightsheet.global_phase import (
    DEFAULT_HIGHPASS_SIGMA_PX as DEFAULT_GLOBAL_PHASE_HIGHPASS_SIGMA_PX,
    DEFAULT_LEVEL as DEFAULT_GLOBAL_PHASE_LEVEL,
    DEFAULT_MAX_RESIDUAL_SHIFT_UM,
    DEFAULT_ORTHOGONAL_LATERAL_FACTOR,
    DEFAULT_Z_SLAB_PLANES as DEFAULT_GLOBAL_PHASE_Z_SLAB_PLANES,
    run_global_phase,
)
from squisher_lightsheet.tile_quadrant_fusion import (
    DEFAULT_FUSED_FIXED_LEVEL_FACTOR_ZYX,
    export_fused_fixed_overlapping_materialized_chunks,
    export_tile_quadrant_materialized_chunks,
    join_materialized_sides,
)
from squisher_lightsheet.lr_alignment import (
    DEFAULT_LEVEL as DEFAULT_LR_ALIGNMENT_LEVEL,
    DEFAULT_PHASE_DOWNSAMPLE_ZYX,
    DEFAULT_Z_SLAB_PLANES,
    run_lr_dumb_stitch_alignment,
)
from squisher_lightsheet.lr_registered_alignment import align_registered_lr_fusions
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
from squisher_lightsheet.nrrd_export import (
    DEFAULT_GZIP_LEVEL,
    DEFAULT_SLAB_PLANES,
    export_nrrd,
)
from squisher_lightsheet.ome_metadata_dumb_stitch import (
    parse_channels as parse_ome_metadata_dumb_stitch_channels,
    parse_view_dir as parse_ome_metadata_dumb_stitch_view_dir,
    render_ome_metadata_dumb_stitch,
)
from squisher_lightsheet.ome_zarr_rechunk import DEFAULT_CHUNK_SHAPE_ZYX, rechunk_ome_zarr
from squisher_lightsheet.parsing import parse_source_view_path_entry
from squisher_lightsheet.planar_tilt import write_planar_tilt_fit
from squisher_lightsheet.positions import create_position_file, create_single_position_file
from squisher_lightsheet.pyramid import add_pyramids
from squisher_lightsheet.registration_join import write_joined_channel_affine_registration
from squisher_lightsheet.registered_grid_anchors import (
    DEFAULT_LEVEL as DEFAULT_REGISTERED_GRID_ANCHOR_LEVEL,
    write_registered_grid_channel_anchors,
)
from squisher_lightsheet.qc import (
    DEFAULT_FUSED_MIDDLE_Z_LEVEL,
    render_live_fusion_preview,
    render_fused_middle_z_qc,
    render_fused_xyz_overlay_qc,
    render_fused_tile_index_overlay,
    render_registration_center_z_spotcheck,
    render_registration_qc,
    write_fused_threshold_review_tiff,
)
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy_registration
from squisher_lightsheet.rough_phase import rough_phase_align
from squisher_lightsheet.surface_intensity import build_surface_intensity_map
from squisher_lightsheet.surface_boundaries import (
    DEFAULT_INNER_FRACTION,
    DEFAULT_INNER_SMOOTH_UM,
    DEFAULT_MAX_RECOVERY_DISTANCE_UM,
    DEFAULT_MIN_CROSSING_SAMPLES,
    DEFAULT_OUTER_FRACTION,
    DEFAULT_OUTER_SMOOTH_UM,
    build_surface_boundaries,
)
from squisher_lightsheet.surface_boundary_qc import build_surface_boundary_qc
from squisher_lightsheet.surface_profiles import build_surface_profiles
from squisher_lightsheet.tile_phase import align_tiles_to_reference, parse_shape_zyx
from squisher_lightsheet.track_z import run_track_z_diagnostics
from squisher_lightsheet.uv_surface import (
    DEFAULT_SMOOTHING_UM,
    build_bilateral_ventricular_uv_surfaces,
    build_ventricular_uv_surface,
)
from squisher_lightsheet.workflow import DEFAULT_LEVEL, DEFAULT_OVERLAP_FRACTION, run_tltr_workflow
from squisher_lightsheet.zeiss_positions import plot_xy_positions
from squisher_lightsheet.zeiss_tile_positions import create_zeiss_tile_position_file


app = typer.Typer(no_args_is_help=True)
align_405_to_488_app = typer.Typer(no_args_is_help=True)
cross_register_app = typer.Typer(no_args_is_help=True)
cross_register_method8_app = typer.Typer(no_args_is_help=True)
app.add_typer(align_405_to_488_app, name="align-405-to-488")
app.add_typer(cross_register_app, name="cross-register")
app.add_typer(cross_register_method8_app, name="cross-register-method8")

FUSED_FIXED_CONTACT_SHEET_SCRIPT = Path(
    "/home/chaichontat/nvme/lightsheet/scripts/render_fused_fixed_method8_contact_sheets.py"
)


@app.callback()
def lightsheet() -> None:
    """Lightsheet metadata, registration, fusion, and QC workflows."""
    register_jpegxr_codec()


@app.command("edu-workflow")
def edu_workflow(
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="One OME-TIFF from the EDU three-channel or sibling 405 acquisition.",
        ),
    ],
    output_root: Annotated[Path | None, typer.Option("--output-root", file_okay=False)] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    allow_disconnected_561: Annotated[
        bool,
        typer.Option("--allow-disconnected-561", help="Accept incomplete connectivity for 561 registration."),
    ] = False,
    allow_disconnected_514: Annotated[
        bool,
        typer.Option("--allow-disconnected-514", help="Accept incomplete connectivity for 514 registration."),
    ] = False,
    allow_disconnected_405: Annotated[
        bool,
        typer.Option("--allow-disconnected-405", help="Accept incomplete connectivity for 405 registration."),
    ] = False,
) -> None:
    """Run or resume the complete WORKFLOW_EDU.md pipeline from one OME-TIFF."""
    from squisher_lightsheet.edu_workflow import ReviewRequired, run_edu_workflow

    try:
        completion = run_edu_workflow(
            source,
            output_root=output_root,
            dry_run=dry_run,
            allow_disconnected_561=allow_disconnected_561,
            allow_disconnected_514=allow_disconnected_514,
            allow_disconnected_405=allow_disconnected_405,
        )
    except ReviewRequired as gate:
        typer.echo(str(gate))
        raise typer.Exit(3) from gate
    typer.echo(completion)


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


def _parse_labeled_paths(
    values: list[str], *, option: str, kind: Literal["file", "directory"] = "file"
) -> list[tuple[str, Path]]:
    parsed = []
    labels = set()
    for value in values:
        label, separator, path_value = value.partition("=")
        if not separator or not label or not path_value:
            raise typer.BadParameter(f"{option} must be LABEL=PATH, got {value!r}")
        if label in labels:
            raise typer.BadParameter(f"duplicate {option} label {label!r}")
        path = Path(path_value)
        if kind == "file" and not path.is_file():
            raise typer.BadParameter(f"{option} path does not exist: {path}")
        if kind == "directory" and not path.is_dir():
            raise typer.BadParameter(f"{option} directory does not exist: {path}")
        labels.add(label)
        parsed.append((label, path))
    return parsed


def _log_progress(message: str) -> None:
    logger.info(message)


@app.command("uv-surface")
def uv_surface(
    mask: Annotated[Path, typer.Option("--mask", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    smoothing_um: Annotated[float, typer.Option("--smoothing-um", min=0)] = DEFAULT_SMOOTHING_UM,
    stride_zyx: Annotated[str, typer.Option("--stride-zyx")] = "1,1,1",
    allow_multi_interval_rays: Annotated[
        bool,
        typer.Option(
            "--allow-multi-interval-rays",
            help="Use the outermost +X crossing when a source or strided ray has multiple intervals.",
        ),
    ] = False,
) -> None:
    """Build a +X ventricular mesh and geometric UV chart from a binary NRRD mask."""
    typer.echo(
        build_ventricular_uv_surface(
            mask_path=mask,
            output_dir=output_dir,
            smoothing_um=smoothing_um,
            stride_zyx=parse_shape_zyx(stride_zyx),
            allow_multi_interval_rays=allow_multi_interval_rays,
        )
    )


@app.command("bilateral-uv-surface")
def bilateral_uv_surface(
    mask: Annotated[Path, typer.Option("--mask", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    split_x_index: Annotated[int, typer.Option("--split-x-index", min=1)],
    smoothing_um: Annotated[float, typer.Option("--smoothing-um", min=0)] = DEFAULT_SMOOTHING_UM,
    keep_main_sheet: Annotated[
        bool,
        typer.Option(
            "--keep-main-sheet",
            help="Trim the inferior/posterior wrap using the split bridge's upper Z envelope.",
        ),
    ] = False,
    main_sheet_min_z_index: Annotated[
        int | None,
        typer.Option(
            "--main-sheet-min-z-index",
            min=0,
            help="Optional original-grid lower Z bound applied with --keep-main-sheet.",
        ),
    ] = None,
    max_face_edge_um: Annotated[
        float | None,
        typer.Option(
            "--max-face-edge-um",
            min=0,
            help="Drop discontinuity-spanning faces before retaining the largest component.",
        ),
    ] = None,
) -> None:
    """Build separate medial UV charts for X-separated cerebral hemispheres."""
    typer.echo(
        build_bilateral_ventricular_uv_surfaces(
            mask_path=mask,
            output_dir=output_dir,
            split_x_index=split_x_index,
            smoothing_um=smoothing_um,
            keep_main_sheet=keep_main_sheet,
            main_sheet_min_z_index=main_sheet_min_z_index,
            max_face_edge_um=max_face_edge_um,
        )
    )


@app.command("surface-intensity-map")
def surface_intensity_map(
    surface_root: Annotated[Path, typer.Option("--surface-root", exists=True, file_okay=False)],
    mask: Annotated[Path, typer.Option("--mask", exists=True, dir_okay=False, readable=True)],
    source_zarr: Annotated[Path, typer.Option("--source-zarr", exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    level: Annotated[int, typer.Option("--level", min=0)] = 2,
    max_depth_um: Annotated[float, typer.Option("--max-depth-um", min=0)] = 2500.0,
    depth_step_um: Annotated[float, typer.Option("--depth-step-um", min=0)] = 4.8,
    uv_size: Annotated[int, typer.Option("--uv-size", min=2)] = 512,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 2048,
) -> None:
    """Max-project tissue-side intensity rays onto bilateral surface UV charts."""
    typer.echo(
        build_surface_intensity_map(
            surface_root=surface_root,
            mask_path=mask,
            source_zarr=source_zarr,
            output_dir=output_dir,
            level=level,
            max_depth_um=max_depth_um,
            depth_step_um=depth_step_um,
            uv_size=uv_size,
            batch_size=batch_size,
        )
    )


@app.command("eomes-profiles")
def eomes_profiles(
    surface: Annotated[Path, typer.Option("--surface", exists=True, dir_okay=False, readable=True)],
    chart: Annotated[Path, typer.Option("--chart", exists=True, dir_okay=False, readable=True)],
    mask: Annotated[Path, typer.Option("--mask", exists=True, dir_okay=False, readable=True)],
    source_zarr: Annotated[Path, typer.Option("--source-zarr", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    level: Annotated[int, typer.Option("--level", min=0)] = 2,
    z_start: Annotated[int, typer.Option("--z-start", min=0)] = 0,
    z_stop: Annotated[int | None, typer.Option("--z-stop", min=1)] = None,
    stride_zyx: Annotated[str, typer.Option("--stride-zyx")] = "2,4,4",
    radius_um: Annotated[float, typer.Option("--radius-um", min=0)] = 20.0,
    max_depth_um: Annotated[float, typer.Option("--max-depth-um", min=0)] = 280.8,
    depth_step_um: Annotated[float, typer.Option("--depth-step-um", min=0)] = 4.8,
    projection_domain: Annotated[
        Literal["full", "ventricular-wall"], typer.Option("--projection-domain")
    ] = "full",
) -> None:
    """Bin 561 intensity by exact closest-surface distance and mesh neighborhood."""
    typer.echo(
        build_surface_profiles(
            surface_path=surface,
            chart_path=chart,
            mask_path=mask,
            source_zarr=source_zarr,
            output_dir=output_dir,
            level=level,
            z_start=z_start,
            z_stop=z_stop,
            stride_zyx=parse_shape_zyx(stride_zyx),
            radius_um=radius_um,
            max_depth_um=max_depth_um,
            depth_step_um=depth_step_um,
            projection_domain=projection_domain,
        )
    )


@app.command("eomes-boundaries")
def eomes_boundaries(
    profiles: Annotated[Path, typer.Option("--profiles", exists=True, dir_okay=False, readable=True)],
    surface: Annotated[Path, typer.Option("--surface", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    inner_fraction: Annotated[float, typer.Option("--inner-fraction", min=0, max=1)] = DEFAULT_INNER_FRACTION,
    inner_smooth_um: Annotated[float, typer.Option("--inner-smooth-um", min=0)] = DEFAULT_INNER_SMOOTH_UM,
    outer_fraction: Annotated[float, typer.Option("--outer-fraction", min=0, max=1)] = DEFAULT_OUTER_FRACTION,
    outer_smooth_um: Annotated[float, typer.Option("--outer-smooth-um", min=0)] = DEFAULT_OUTER_SMOOTH_UM,
    min_crossing_samples: Annotated[int, typer.Option("--min-crossing-samples", min=1)] = (
        DEFAULT_MIN_CROSSING_SAMPLES
    ),
    max_recovery_distance_um: Annotated[
        float, typer.Option("--max-recovery-distance-um", min=0)
    ] = DEFAULT_MAX_RECOVERY_DISTANCE_UM,
    trim_caps: Annotated[bool, typer.Option("--trim-caps/--no-trim-caps")] = True,
) -> None:
    """Measure and recover ordered Eomes inner and outer surface boundaries."""
    typer.echo(
        build_surface_boundaries(
            profiles_path=profiles,
            surface_path=surface,
            output_dir=output_dir,
            inner_fraction=inner_fraction,
            inner_smooth_um=inner_smooth_um,
            outer_fraction=outer_fraction,
            outer_smooth_um=outer_smooth_um,
            min_crossing_samples=min_crossing_samples,
            max_recovery_distance_um=max_recovery_distance_um,
            trim_caps=trim_caps,
        )
    )


@app.command("eomes-boundary-qc")
def eomes_boundary_qc(
    surface: Annotated[Path, typer.Option("--surface", exists=True, dir_okay=False, readable=True)],
    chart: Annotated[Path, typer.Option("--chart", exists=True, dir_okay=False, readable=True)],
    boundaries: Annotated[Path, typer.Option("--boundaries", exists=True, dir_okay=False, readable=True)],
    mask: Annotated[Path, typer.Option("--mask", exists=True, dir_okay=False, readable=True)],
    source_zarr: Annotated[Path, typer.Option("--source-zarr", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    source_level: Annotated[int, typer.Option("--source-level", min=0)] = 2,
    intrinsic_z_index: Annotated[list[int] | None, typer.Option("--intrinsic-z-index", min=0)] = None,
    intrinsic_z_position_um: Annotated[
        list[float] | None, typer.Option("--intrinsic-z-position-um", min=0)
    ] = None,
    grid_stride: Annotated[int, typer.Option("--grid-stride", min=1)] = 2,
) -> None:
    """Render direct and recovered implicit Eomes contours on matched sections."""
    typer.echo(
        build_surface_boundary_qc(
            surface_path=surface,
            chart_path=chart,
            boundaries_path=boundaries,
            mask_path=mask,
            source_zarr=source_zarr,
            output_dir=output_dir,
            source_level=source_level,
            intrinsic_z_indices=intrinsic_z_index,
            intrinsic_z_positions_um=intrinsic_z_position_um,
            grid_stride=grid_stride,
        )
    )


@app.command("fit-planar-tilt")
def fit_planar_tilt(
    mask: Annotated[Path, typer.Option("--mask", exists=True, dir_okay=False, readable=True)],
    source: Annotated[
        list[Path],
        typer.Option(
            "--source",
            exists=True,
            file_okay=False,
            readable=True,
            help="Repeat for each OME-Zarr source.",
        ),
    ],
    window: Annotated[
        list[str],
        typer.Option("--window", help="Repeat once per source as MIN,MAX."),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    level: Annotated[int, typer.Option("--level", min=0)] = 2,
    z_depth_um: Annotated[float, typer.Option("--z-depth-um", min=0.001)] = 360.0,
    xy_step_um: Annotated[float, typer.Option("--xy-step-um", min=0.001)] = 5.0,
) -> None:
    """Fit a two-axis tissue-plane normal from masked OME-Zarr sources."""
    if not np.isfinite(z_depth_um):
        raise typer.BadParameter("Must be finite.", param_hint="--z-depth-um")
    if not np.isfinite(xy_step_um):
        raise typer.BadParameter("Must be finite.", param_hint="--xy-step-um")
    if len(source) != len(window):
        raise typer.BadParameter("Provide exactly one --window for each --source.", param_hint="--window")
    windows = []
    for value in window:
        try:
            lower, upper = (float(piece) for piece in value.split(","))
        except ValueError as error:
            raise typer.BadParameter(
                f"Expected MIN,MAX, received {value!r}.", param_hint="--window"
            ) from error
        if not np.isfinite((lower, upper)).all() or lower >= upper:
            raise typer.BadParameter(
                f"Expected finite MIN,MAX with MIN < MAX, received {value!r}.",
                param_hint="--window",
            )
        windows.append((lower, upper))

    typer.echo(
        write_planar_tilt_fit(
            mask_path=mask,
            source_paths=source,
            windows=windows,
            output_path=output,
            level=level,
            z_depth_um=z_depth_um,
            xy_step_um=xy_step_um,
        )
    )


@app.command("fit-residual")
@app.command("basic-residual")
def fit_residual(
    fixed_fused: Annotated[Path, typer.Option("--fixed-fused", exists=True, file_okay=False, readable=True)],
    registration: Annotated[Path, typer.Option("--registration", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    source_level: Annotated[int, typer.Option("--source-level", min=0)] = 2,
    stride: Annotated[int, typer.Option("--stride", min=1)] = 4,
    xy_degree: Annotated[int, typer.Option("--xy-degree", min=1, max=2)] = 1,
    z_degree: Annotated[int, typer.Option("--z-degree", min=0, max=3)] = 1,
    field_penalty: Annotated[
        float | None,
        typer.Option("--field-penalty", min=0, help="Omit to select by held-out pairs."),
    ] = None,
    source_field_penalty: Annotated[
        float | None,
        typer.Option(
            "--source-field-penalty",
            min=0,
            help="Regularization for per-source spatial fields; omit to select by held-out pairs.",
        ),
    ] = None,
    z_percentile: Annotated[
        list[float] | None,
        typer.Option(
            "--z-percentile",
            min=0,
            max=100,
            help="Repeat to fit one model across selected fixed-grid depths.",
        ),
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 8,
    validation_workers: Annotated[int, typer.Option("--validation-workers", min=1)] = 1,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Fit shared and source-specific residual fields from deconvolved tiles."""
    from squisher_lightsheet.residual_correction import (
        DEFAULT_Z_PERCENTILES,
        fit_residual_correction,
    )

    manifest_path = fit_residual_correction(
        registration=registration,
        fixed_fused=fixed_fused,
        output_dir=output_dir,
        channel=channel,
        source_level=source_level,
        stride=stride,
        xy_degree=xy_degree,
        z_degree=z_degree,
        field_penalty=field_penalty,
        source_field_penalty=source_field_penalty,
        z_percentiles=DEFAULT_Z_PERCENTILES if z_percentile is None else z_percentile,
        workers=workers,
        validation_workers=validation_workers,
        seed=seed,
    )
    typer.echo(manifest_path)
    typer.echo(f"QC figures: {manifest_path.parent / 'qc'}")


@app.command("cross-dataset-match")
def cross_dataset_match_command(
    fixed_fused: Annotated[Path, typer.Option("--fixed-fused", exists=True, file_okay=False, readable=True)],
    registration: Annotated[Path, typer.Option("--registration", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    reference_view: Annotated[str, typer.Option("--reference-view")],
    moving_view: Annotated[str, typer.Option("--moving-view")],
    correction_by_source_view: Annotated[
        list[str] | None,
        typer.Option(
            "--correction-by-source-view",
            help="Optional; repeat as VIEW=/path/to/existing/correction.json.",
        ),
    ] = None,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    source_level: Annotated[int, typer.Option("--source-level", min=0)] = 2,
    stride: Annotated[int, typer.Option("--stride", min=1)] = 4,
    z_percentile: Annotated[
        list[float] | None,
        typer.Option(
            "--z-percentile",
            min=0,
            max=100,
            help="Repeat to sample cross-view overlaps at selected fused-grid depths.",
        ),
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 8,
    grid_spacing_zyx_um: Annotated[
        str,
        typer.Option(
            "--grid-spacing-zyx-um",
            help="Registered correction grid spacing as z,y,x micrometers.",
        ),
    ] = "100,200,200",
) -> None:
    """Match two basic-residual-corrected datasets with a registered spatial field."""
    from squisher_lightsheet.cross_dataset_match import (
        DEFAULT_Z_PERCENTILES,
        cross_dataset_match,
    )

    corrections = {}
    for value in correction_by_source_view or []:
        view, path = parse_source_view_path_entry(value, error_factory=typer.BadParameter)
        if view in corrections:
            raise typer.BadParameter(
                f"Duplicate --correction-by-source-view entry for {view!r}",
                param_hint="--correction-by-source-view",
            )
        if not path.is_file():
            raise typer.BadParameter(
                f"Correction file does not exist: {path}",
                param_hint="--correction-by-source-view",
            )
        corrections[view] = path
    manifest_path = cross_dataset_match(
        registration=registration,
        fixed_fused=fixed_fused,
        output_dir=output_dir,
        corrections_by_view=corrections,
        reference_view=reference_view,
        moving_view=moving_view,
        channel=channel,
        source_level=source_level,
        stride=stride,
        z_percentiles=DEFAULT_Z_PERCENTILES if z_percentile is None else z_percentile,
        workers=workers,
        grid_spacing_zyx_um=_parse_float_zyx(grid_spacing_zyx_um),
    )
    typer.echo(manifest_path)
    typer.echo(f"Base correction: {manifest_path.parent / 'correction.json'}")
    typer.echo(f"Registered correction: {manifest_path.parent / 'registered-view-correction.json'}")
    typer.echo(f"QC figure: {manifest_path.parent / 'cross-dataset-match-qc.png'}")


@app.command("registered-grid-channel-anchors")
def registered_grid_channel_anchors(
    fixed_registration: Annotated[
        Path,
        typer.Option("--fixed-registration", exists=True, dir_okay=False, readable=True),
    ],
    moving_baseline_registration: Annotated[
        Path,
        typer.Option("--moving-baseline-registration", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output")],
    fixed_threshold: Annotated[
        float,
        typer.Option(
            "--fixed-threshold",
            min=0,
            help="Required human-reviewed threshold in the fixed tile intensity space.",
        ),
    ],
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_REGISTERED_GRID_ANCHOR_LEVEL,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 4,
) -> None:
    """Measure verified matched-tile anchors on a shared registered grid."""
    typer.echo(
        write_registered_grid_channel_anchors(
            fixed_registration=fixed_registration,
            moving_baseline_registration=moving_baseline_registration,
            output=output,
            fixed_threshold=fixed_threshold,
            level=level,
            workers=workers,
            progress=_log_progress,
        )
    )


@app.command("channel-affine-registration")
def channel_affine_registration(
    window_dir: Annotated[Path, typer.Option("--window-dir", exists=True, file_okay=False, readable=True)],
    reference_registration: Annotated[
        Path,
        typer.Option("--reference-registration", exists=True, dir_okay=False, readable=True),
    ],
    output_registration: Annotated[Path, typer.Option("--output-registration")],
    expected_moving_channel: Annotated[int, typer.Option("--expected-moving-channel", min=0)],
    expected_fixed_fused: Annotated[
        Path,
        typer.Option("--expected-fixed-fused", exists=True, file_okay=False, readable=True),
    ],
    source_label: Annotated[str, typer.Option("--source-label")],
    target_label: Annotated[str, typer.Option("--target-label")],
    model: Annotated[
        Literal["global", "constrained-tile"],
        typer.Option("--model", help="Final full-tile model fitted from the accepted windows."),
    ] = "global",
    baseline_registration: Annotated[
        Path | None,
        typer.Option(
            "--baseline-registration",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Provisional global registration required by --model constrained-tile.",
        ),
    ] = None,
    seam_measurements: Annotated[
        Path | None,
        typer.Option(
            "--seam-measurements",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Measured same-channel overlap constraints required by --model constrained-tile.",
        ),
    ] = None,
    verified_anchor: Annotated[
        Path | None,
        typer.Option(
            "--verified-anchor",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Verified registered-grid image anchors required by --model constrained-tile.",
        ),
    ] = None,
) -> None:
    """Finalize accepted fused-fixed windows into a full-tile registration."""
    if model == "constrained-tile":
        if baseline_registration is None or seam_measurements is None or verified_anchor is None:
            raise typer.BadParameter(
                "--model constrained-tile requires --baseline-registration, --seam-measurements, "
                "and --verified-anchor"
            )
        typer.echo(
            write_constrained_tile_channel_affine_registration(
                window_dir=window_dir,
                reference_registration_input=reference_registration,
                baseline_registration_input=baseline_registration,
                seam_measurements_input=seam_measurements,
                verified_anchor_input=verified_anchor,
                output_registration=output_registration,
                expected_moving_channel=expected_moving_channel,
                expected_fixed_fused=expected_fixed_fused,
                source_label=source_label,
                target_label=target_label,
            )
        )
        return
    typer.echo(
        write_global_channel_affine_registration(
            window_dir=window_dir,
            reference_registration_input=reference_registration,
            output_registration=output_registration,
            expected_moving_channel=expected_moving_channel,
            expected_fixed_fused=expected_fixed_fused,
            source_label=source_label,
            target_label=target_label,
        )
    )


@app.command("apply-channel-affine")
def apply_channel_affine(
    reference_registration: Annotated[
        Path,
        typer.Option("--reference-registration", exists=True, dir_okay=False, readable=True),
    ],
    calibration_registration: Annotated[
        Path,
        typer.Option("--calibration-registration", exists=True, dir_okay=False, readable=True),
    ],
    expected_calibration_sha256: Annotated[str, typer.Option("--expected-calibration-sha256")],
    output_registration: Annotated[Path, typer.Option("--output-registration")],
    expected_moving_channel: Annotated[int, typer.Option("--expected-moving-channel", min=0)],
    source_label: Annotated[str, typer.Option("--source-label")],
    target_label: Annotated[str, typer.Option("--target-label")],
) -> None:
    """Apply a pinned physical channel affine to a registration."""
    typer.echo(
        apply_channel_affine_registration(
            reference_registration_input=reference_registration,
            calibration_registration_input=calibration_registration,
            expected_calibration_sha256=expected_calibration_sha256,
            output_registration=output_registration,
            expected_moving_channel=expected_moving_channel,
            source_label=source_label,
            target_label=target_label,
        )
    )


@app.command("join-channel-affine-registrations")
def join_channel_affine_registrations(
    reference_registration: Annotated[
        Path,
        typer.Option("--reference-registration", exists=True, dir_okay=False, readable=True),
    ],
    side_registration: Annotated[
        list[Path],
        typer.Option(
            "--side-registration",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Repeat for each disjoint side-specific channel-affine registration.",
        ),
    ],
    output_registration: Annotated[Path, typer.Option("--output-registration")],
    source_label: Annotated[str, typer.Option("--source-label")],
    target_label: Annotated[str, typer.Option("--target-label")],
) -> None:
    """Join side-specific channel affines over one reference registration."""
    typer.echo(
        write_joined_channel_affine_registration(
            reference_registration_input=reference_registration,
            side_registration_inputs=side_registration,
            output_registration=output_registration,
            source_label=source_label,
            target_label=target_label,
        )
    )


@app.command("fused-fixed-contact-sheet")
def fused_fixed_contact_sheet(
    run_dir: Annotated[Path, typer.Option("--run-dir", exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
    limit: Annotated[int, typer.Option("--limit", min=1)] = 24,
    columns: Annotated[int, typer.Option("--columns", min=1)] = 2,
    device: Annotated[int, typer.Option("--device", min=0)] = 0,
    renderer_python: Annotated[
        Path | None,
        typer.Option("--renderer-python", exists=True, dir_okay=False),
    ] = None,
    status: Annotated[str, typer.Option("--status")] = "accepted",
    rejection_reason: Annotated[str | None, typer.Option("--rejection-reason")] = None,
    tile_filter: Annotated[str | None, typer.Option("--tile-filter")] = None,
    before_transform: Annotated[
        Literal["phase-correlation", "linear-initializer"],
        typer.Option("--before-transform"),
    ] = "phase-correlation",
) -> None:
    """Render completed fused-fixed windows, including an incomplete active run."""
    if not FUSED_FIXED_CONTACT_SHEET_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Missing fused-fixed contact-sheet renderer: {FUSED_FIXED_CONTACT_SHEET_SCRIPT}"
        )
    resolved_output = (output_dir or run_dir / "contact_sheets").resolve()
    command = [
        str(renderer_python or sys.executable),
        str(FUSED_FIXED_CONTACT_SHEET_SCRIPT),
        "--run-dir",
        str(run_dir.resolve()),
        "--output-dir",
        str(resolved_output),
        "--limit",
        str(limit),
        "--columns",
        str(columns),
        "--device",
        str(device),
        "--status",
        status,
        "--before-transform",
        before_transform,
    ]
    if rejection_reason is not None:
        command.extend(["--rejection-reason", rejection_reason])
    if tile_filter is not None:
        command.extend(["--tile-filter", tile_filter])
    logger.info("{}", " ".join(command))
    subprocess.run(command, check=True)


@app.command("fused-fixed-materialize-overlap")
def fused_fixed_materialize_overlap(
    moving_position: Annotated[
        Path,
        typer.Option("--moving-position", exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    output_codec: Annotated[
        Literal["zstd", "jpegxr"],
        typer.Option("--output-codec"),
    ],
    moving_source_registration: Annotated[
        Path | None,
        typer.Option(
            "--moving-source-registration",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Registration artifact that maps moving tile identities to pixel-source paths.",
        ),
    ] = None,
    moving_source_dir: Annotated[
        Path | None,
        typer.Option(
            "--moving-source-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Read replacement OME-Zarr tiles with matching identities and spatial shapes; preserve saved geometry.",
        ),
    ] = None,
    residual_correction: Annotated[
        Path | None,
        typer.Option(
            "--residual-correction",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Apply a fitted post-deconvolution correction to native source blocks before materialization.",
        ),
    ] = None,
    source_registration: Annotated[
        Path | None,
        typer.Option("--source-registration", exists=True, dir_okay=False, readable=True),
    ] = None,
    source_summary: Annotated[
        Path | None,
        typer.Option("--source-summary", exists=True, dir_okay=False, readable=True),
    ] = None,
    source_channel: Annotated[int, typer.Option("--source-channel", min=0)] = 0,
    core_shape_zyx: Annotated[str, typer.Option("--core-shape-zyx")] = "480,480,480",
    window_shape_zyx: Annotated[str, typer.Option("--window-shape-zyx")] = "528,528,528",
    level_factor_zyx: Annotated[str, typer.Option("--level-factor-zyx")] = ",".join(
        map(str, DEFAULT_FUSED_FIXED_LEVEL_FACTOR_ZYX)
    ),
    zstd_level: Annotated[int, typer.Option("--zstd-level", min=1, max=22)] = 3,
    jpegxr_level: Annotated[float, typer.Option("--jpegxr-level", min=0.0, max=1.0)] = DEFAULT_JPEGXR_LEVEL,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    max_tiles: Annotated[int | None, typer.Option("--max-tiles", min=1)] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Rematerialize fused-fixed cores with cross-registration-style overlap."""
    if (source_registration is None) == (source_summary is None):
        raise typer.BadParameter("pass exactly one of --source-registration or --source-summary")
    if resume and overwrite:
        raise typer.BadParameter("--resume and --overwrite are mutually exclusive")
    if not resume:
        _require_empty_or_overwrite(
            output_dir, overwrite=overwrite, label="overlapping materialized output directory"
        )
    outputs = export_fused_fixed_overlapping_materialized_chunks(
        source_registration_input=source_registration,
        source_summary_input=source_summary,
        moving_position_input=moving_position,
        moving_source_input=moving_source_registration,
        moving_source_dir=moving_source_dir,
        residual_correction=residual_correction,
        output_dir=output_dir,
        source_channel=source_channel,
        core_shape_zyx=parse_shape_zyx(core_shape_zyx),
        window_shape_zyx=parse_shape_zyx(window_shape_zyx),
        level_factor_zyx=parse_shape_zyx(level_factor_zyx),
        output_codec=output_codec,
        zstd_level=zstd_level,
        jpegxr_level=jpegxr_level,
        workers=workers,
        max_tiles=max_tiles,
        resume=resume,
    )
    typer.echo(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


def _parse_float_zyx(text: str) -> tuple[float, float, float]:
    values = tuple(float(part) for part in text.split(","))
    if len(values) != 3:
        raise typer.BadParameter(f"Expected comma-separated z,y,x values, got {text!r}")
    return values


def _parse_starting_affine_matrix_zyx(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in text.split(","))
        return normalize_starting_affine_matrix_zyx(values)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


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


def _global_cross_register_manifest_path(output_dir: Path) -> Path:
    return output_dir / "cross-register.manifest.json"


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


def _update_global_cross_register_manifest(
    output_dir: Path,
    *,
    stage: str,
    updates: dict[str, object],
) -> Path:
    manifest_path = _global_cross_register_manifest_path(output_dir)
    manifest = _read_json_if_exists(manifest_path)
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        stages = {}
    stages[stage] = updates
    manifest.update(
        {
            "artifact_type": "squisher_lightsheet.cross_register_manifest.v1",
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


def _require_distinct_paths(
    *,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
) -> None:
    resolved_inputs = {label: path.resolve() for label, path in inputs.items()}
    resolved_outputs = {label: path.resolve() for label, path in outputs.items()}
    aliases = []
    for output_label, output_path in resolved_outputs.items():
        for input_label, input_path in resolved_inputs.items():
            if output_path == input_path:
                aliases.append(f"{output_label} aliases {input_label} at {output_path}")
    output_items = list(resolved_outputs.items())
    for index, (left_label, left_path) in enumerate(output_items):
        for right_label, right_path in output_items[index + 1 :]:
            if left_path == right_path:
                aliases.append(f"{left_label} aliases {right_label} at {left_path}")
    if aliases:
        raise typer.BadParameter("Cross-register paths must be distinct: " + "; ".join(aliases))


@cross_register_app.command("global-phase")
def cross_register_global_phase(
    fixed_position: Annotated[
        Path, typer.Option("--fixed-position", exists=True, dir_okay=False, readable=True)
    ],
    moving_position: Annotated[
        Path, typer.Option("--moving-position", exists=True, dir_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    fixed_tile_dir: Annotated[
        Path | None,
        typer.Option("--fixed-tile-dir", exists=True, file_okay=False, readable=True),
    ] = None,
    moving_tile_dir: Annotated[
        Path | None,
        typer.Option("--moving-tile-dir", exists=True, file_okay=False, readable=True),
    ] = None,
    output_position: Annotated[Path | None, typer.Option("--output-position", dir_okay=False)] = None,
    fixed_channel: Annotated[int, typer.Option("--fixed-channel", min=0)] = 0,
    moving_channel: Annotated[int, typer.Option("--moving-channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_GLOBAL_PHASE_LEVEL,
    z_slab_planes: Annotated[int, typer.Option("--z-slab-planes", min=1)] = (
        DEFAULT_GLOBAL_PHASE_Z_SLAB_PLANES
    ),
    fixed_intensity_transform: Annotated[
        Literal["identity", "log1p"], typer.Option("--fixed-intensity-transform")
    ] = "log1p",
    moving_intensity_transform: Annotated[
        Literal["identity", "log1p"], typer.Option("--moving-intensity-transform")
    ] = "identity",
    highpass_sigma_px: Annotated[
        float, typer.Option("--highpass-sigma-px", min=0.0)
    ] = DEFAULT_GLOBAL_PHASE_HIGHPASS_SIGMA_PX,
    max_residual_shift_um: Annotated[
        float, typer.Option("--max-residual-shift-um", min=0.0)
    ] = DEFAULT_MAX_RESIDUAL_SHIFT_UM,
    orthogonal_lateral_factor: Annotated[
        int, typer.Option("--orthogonal-lateral-factor", min=1)
    ] = DEFAULT_ORTHOGONAL_LATERAL_FACTOR,
    orthogonal_z: Annotated[bool, typer.Option("--orthogonal-z/--no-orthogonal-z")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    resolved_output_position = output_position or output_dir / "global-phase.positions.json"
    outputs = {
        "output position": resolved_output_position,
        "summary": output_dir / "global-phase.summary.json",
        "fixed phase MIP": output_dir / "fixed.phase.tif",
        "moving phase MIP": output_dir / "moving.phase.tif",
        "before overlay": output_dir / "before.png",
        "after overlay": output_dir / "after.png",
        "manifest": _global_cross_register_manifest_path(output_dir),
    }
    if orthogonal_z:
        outputs.update(
            {
                "orthogonal summary": output_dir / "orthogonal" / "orthogonal.summary.json",
                "orthogonal contact sheet": output_dir / "orthogonal" / "orthogonal.png",
            }
        )
        for plane in ("zx", "zy"):
            for artifact in ("fixed.tif", "moving.tif", "before.png", "after.png"):
                outputs[f"orthogonal {plane} {artifact}"] = output_dir / "orthogonal" / f"{plane}.{artifact}"
    _require_distinct_paths(
        inputs={"fixed position": fixed_position, "moving position": moving_position},
        outputs=outputs,
    )
    for label, path in outputs.items():
        if label == "manifest":
            continue
        _require_empty_or_overwrite(path, overwrite=overwrite, label=label)
    result = run_global_phase(
        fixed_position=fixed_position,
        moving_position=moving_position,
        output_dir=output_dir,
        output_position=resolved_output_position,
        fixed_tile_dir=fixed_tile_dir,
        moving_tile_dir=moving_tile_dir,
        fixed_channel=fixed_channel,
        moving_channel=moving_channel,
        level=level,
        z_slab_planes=z_slab_planes,
        fixed_intensity_transform=fixed_intensity_transform,
        moving_intensity_transform=moving_intensity_transform,
        highpass_sigma_px=highpass_sigma_px,
        max_residual_shift_um=max_residual_shift_um,
        orthogonal_lateral_factor=orthogonal_lateral_factor,
        orthogonal_z=orthogonal_z,
    )
    manifest = _update_global_cross_register_manifest(
        output_dir,
        stage="global-phase",
        updates={
            "fixed_position": fixed_position.resolve(),
            "moving_position": moving_position.resolve(),
            "fixed_tile_dir": None if fixed_tile_dir is None else fixed_tile_dir.resolve(),
            "moving_tile_dir": None if moving_tile_dir is None else moving_tile_dir.resolve(),
            "output_position": result.output_position,
            "summary": result.summary,
            "fixed_mip": result.fixed_mip,
            "moving_mip": result.moving_mip,
            "before_overlay": result.before_overlay,
            "after_overlay": result.after_overlay,
            "orthogonal_summary": result.orthogonal_summary,
            "orthogonal_contact_sheet": result.orthogonal_contact_sheet,
            "fixed_channel": fixed_channel,
            "moving_channel": moving_channel,
            "level": level,
            "z_slab_planes": z_slab_planes,
            "fixed_intensity_transform": fixed_intensity_transform,
            "moving_intensity_transform": moving_intensity_transform,
            "highpass_sigma_px": highpass_sigma_px,
            "max_residual_shift_um": max_residual_shift_um,
            "orthogonal_lateral_factor": orthogonal_lateral_factor,
            "orthogonal_z": orthogonal_z,
        },
    )
    typer.echo(
        json.dumps(
            {
                "output_position": str(result.output_position),
                "summary": str(result.summary),
                "before_overlay": str(result.before_overlay),
                "after_overlay": str(result.after_overlay),
                "orthogonal_summary": (
                    None if result.orthogonal_summary is None else str(result.orthogonal_summary)
                ),
                "orthogonal_contact_sheet": (
                    None if result.orthogonal_contact_sheet is None else str(result.orthogonal_contact_sheet)
                ),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


@cross_register_app.command("method6")
def cross_register_method6(
    fixed_position: Annotated[
        Path, typer.Option("--fixed-position", exists=True, dir_okay=False, readable=True)
    ],
    moving_position: Annotated[
        Path, typer.Option("--moving-position", exists=True, dir_okay=False, readable=True)
    ],
    moving_source_position: Annotated[
        Path,
        typer.Option("--moving-source-position", exists=True, dir_okay=False, readable=True),
    ],
    fixed_fused: Annotated[Path, typer.Option("--fixed-fused", exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    output_registration: Annotated[Path, typer.Option("--output-registration", dir_okay=False)],
    fixed_mask_threshold: Annotated[float, typer.Option("--fixed-mask-threshold", min=0.0)],
    source_label: Annotated[str, typer.Option("--source-label")],
    target_label: Annotated[str, typer.Option("--target-label")],
    moving_channel: Annotated[int, typer.Option("--moving-channel", min=0)] = 0,
    sweep_runner: Annotated[
        Path,
        typer.Option("--sweep-runner", exists=True, dir_okay=False, readable=True),
    ] = DEFAULT_SWEEP_RUNNER,
    window_filter_json: Annotated[
        Path | None,
        typer.Option("--window-filter-json", exists=True, dir_okay=False, readable=True),
    ] = None,
    core_shape_zyx: Annotated[str, typer.Option("--core-shape-zyx")] = "480,480,480",
    window_shape_zyx: Annotated[str, typer.Option("--window-shape-zyx")] = "528,528,528",
    fit_downsample_zyx: Annotated[str, typer.Option("--fit-downsample-zyx")] = "1,1,1",
    native_lib_dir: Annotated[
        Path,
        typer.Option("--native-lib-dir", exists=True, file_okay=False, readable=True),
    ] = DEFAULT_CROSS_REGISTER_METHOD8_LIB_DIR,
    ftol: Annotated[float, typer.Option("--ftol", min=0.0)] = 1e-4,
    max_iterations: Annotated[int, typer.Option("--max-iterations", min=1)] = 300,
    phase_upsample_factor: Annotated[int, typer.Option("--phase-upsample-factor", min=1)] = 10,
    min_corr: Annotated[float, typer.Option("--min-corr")] = 0.15,
    min_grad_ncc: Annotated[float, typer.Option("--min-grad-ncc")] = 0.24,
    fit_intensity_transform: Annotated[
        Literal["linear", "log1p"], typer.Option("--fit-intensity-transform")
    ] = "log1p",
    starting_affine_matrix_zyx: Annotated[
        str,
        typer.Option(
            "--starting-affine-matrix-zyx",
            help="Nine comma-separated row-major values for the native Method 6 initializer.",
        ),
    ] = ",".join(map(str, DEFAULT_STARTING_AFFINE_MATRIX_ZYX)),
    fixed_mask_level: Annotated[int, typer.Option("--fixed-mask-level", min=0)] = 2,
    fixed_mask_min_voxels: Annotated[int, typer.Option("--fixed-mask-min-voxels", min=1)] = 256,
    fixed_mask_max_masked_fraction: Annotated[
        float,
        typer.Option("--fixed-mask-max-masked-fraction", min=0.0, max=1.0),
    ] = 0.95,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    max_tasks_per_worker: Annotated[int, typer.Option("--max-tasks-per-worker", min=1)] = 10,
    devices: Annotated[
        str,
        typer.Option(
            "--devices",
            help=(
                "Comma-separated local CUDA indices when CUDA_VISIBLE_DEVICES is set; "
                "physical indices otherwise."
            ),
        ),
    ] = "0",
    max_windows: Annotated[int | None, typer.Option("--max-windows", min=1)] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Fit fused-fixed Method 6, then write one canonical channel registration."""
    if output_registration.exists() and not overwrite:
        raise typer.BadParameter(
            f"output registration already exists at {output_registration}; pass --overwrite to replace it"
        )
    _require_distinct_paths(
        inputs={
            "fixed position": fixed_position,
            "moving position": moving_position,
            "moving source position": moving_source_position,
            "fixed fused": fixed_fused,
        },
        outputs={"output registration": output_registration},
    )
    summary, registration = run_fused_fixed_method6(
        fixed_position=fixed_position,
        moving_position=moving_position,
        moving_source_position=moving_source_position,
        fixed_fused=fixed_fused,
        output_dir=output_dir,
        output_registration=output_registration,
        moving_channel=moving_channel,
        fixed_mask_threshold=fixed_mask_threshold,
        source_label=source_label,
        target_label=target_label,
        sweep_runner=sweep_runner,
        window_filter_json=window_filter_json,
        core_shape_zyx=parse_shape_zyx(core_shape_zyx),
        window_shape_zyx=parse_shape_zyx(window_shape_zyx),
        fit_downsample_zyx=parse_shape_zyx(fit_downsample_zyx),
        native_lib_dir=native_lib_dir,
        ftol=ftol,
        max_iterations=max_iterations,
        phase_upsample_factor=phase_upsample_factor,
        min_corr=min_corr,
        min_grad_ncc=min_grad_ncc,
        fit_intensity_transform=fit_intensity_transform,
        starting_affine_matrix_zyx=_parse_starting_affine_matrix_zyx(starting_affine_matrix_zyx),
        fixed_mask_level=fixed_mask_level,
        fixed_mask_min_voxels=fixed_mask_min_voxels,
        fixed_mask_max_masked_fraction=fixed_mask_max_masked_fraction,
        workers=workers,
        max_tasks_per_worker=max_tasks_per_worker,
        devices=_parse_devices(devices),
        max_windows=max_windows,
        resume=resume,
    )
    typer.echo(
        json.dumps(
            {"summary": str(summary), "registration": str(registration)},
            indent=2,
        )
    )


@cross_register_app.command("recover-native")
def cross_register_recover_native(
    summary: Annotated[Path, typer.Option("--summary", exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
    maximum_outlier_px: Annotated[
        float, typer.Option("--maximum-outlier-px", min=0.0)
    ] = DEFAULT_NATIVE_RECOVERY_MAXIMUM_OUTLIER_PX,
    maximum_refit_displacement_px: Annotated[
        float, typer.Option("--maximum-refit-displacement-px", min=0.0)
    ] = DEFAULT_MAXIMUM_REFIT_DISPLACEMENT_PX,
    devices: Annotated[
        str,
        typer.Option(
            "--devices",
            help=(
                "Comma-separated local CUDA indices when CUDA_VISIBLE_DEVICES is set; "
                "physical indices otherwise."
            ),
        ),
    ] = DEFAULT_NATIVE_DEVICE_SPEC,
    adjacency_json: Annotated[
        Path | None,
        typer.Option("--adjacency-json", exists=True, dir_okay=False, readable=True),
    ] = None,
    tile_filter: Annotated[str | None, typer.Option("--tile-filter")] = None,
    rerun_all_eligible: Annotated[bool, typer.Option("--rerun-all-eligible")] = False,
    source_cache_max_gib: Annotated[
        float, typer.Option("--source-cache-max-gib", min=0.0)
    ] = 0.0,
    sweep_runner: Annotated[
        Path,
        typer.Option("--sweep-runner", exists=True, dir_okay=False, readable=True),
    ] = DEFAULT_SWEEP_RUNNER,
    manual_initializations: Annotated[
        Path | None,
        typer.Option(
            "--manual-initializations",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Dataset master file of curated window initializations.",
        ),
    ] = None,
    manual_initializations_channel: Annotated[
        str | None, typer.Option("--manual-initializations-channel")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
) -> None:
    """Recover native registration windows from spatial affine consensus."""
    if output_dir is None and not dry_run:
        raise typer.BadParameter("--output-dir is required unless --dry-run is used")
    result = recover_native(
        summary=summary,
        output_dir=output_dir,
        maximum_outlier_px=maximum_outlier_px,
        maximum_refit_displacement_px=maximum_refit_displacement_px,
        devices=_parse_devices(devices),
        adjacency_json=adjacency_json,
        tile_filter=tile_filter,
        rerun_all_eligible=rerun_all_eligible,
        source_cache_max_gib=source_cache_max_gib,
        sweep_runner=sweep_runner,
        manual_initializations=manual_initializations,
        manual_initializations_channel=manual_initializations_channel,
        dry_run=dry_run,
        resume=resume,
    )
    if result is not None:
        typer.echo(json.dumps({"summary": str(result)}, indent=2))


@cross_register_app.command("complete-recovery")
def cross_register_complete_recovery(
    native_summary: Annotated[
        Path,
        typer.Option("--native-summary", exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    minimum_linear_samples: Annotated[
        int, typer.Option("--minimum-linear-samples", min=2)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.minimum_linear_samples,
    minimum_smoothing_samples: Annotated[
        int, typer.Option("--minimum-smoothing-samples", min=2)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.minimum_smoothing_samples,
    linear_loss_scale_um: Annotated[
        float, typer.Option("--linear-loss-scale-um", min=0.0)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.linear_loss_scale_um,
    outlier_mad: Annotated[
        float, typer.Option("--outlier-mad", min=0.0)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.outlier_mad,
    minimum_linear_outlier_um: Annotated[
        float, typer.Option("--minimum-linear-outlier-um", min=0.0)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.minimum_linear_outlier_um,
    maximum_linear_outlier_um: Annotated[
        float, typer.Option("--maximum-linear-outlier-um", min=0.0)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.maximum_linear_outlier_um,
    maximum_linear_median_residual_um: Annotated[
        float, typer.Option("--maximum-linear-median-residual-um", min=0.0)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.maximum_linear_median_residual_um,
    maximum_outlier_px: Annotated[
        float, typer.Option("--maximum-outlier-px", min=0.0)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.maximum_outlier_px,
    replacement_image_ncc_margin: Annotated[
        float, typer.Option("--replacement-image-ncc-margin", min=0.0)
    ] = DEFAULT_COMPLETE_RECOVERY_SETTINGS.replacement_image_ncc_margin,
) -> None:
    """Publish the fixed four-stage recovery policy from a native checkpoint."""
    summary = complete_recovery(
        native_summary=native_summary,
        output_dir=output_dir,
        settings=CompleteRecoverySettings(
            minimum_linear_samples=minimum_linear_samples,
            minimum_smoothing_samples=minimum_smoothing_samples,
            linear_loss_scale_um=linear_loss_scale_um,
            outlier_mad=outlier_mad,
            minimum_linear_outlier_um=minimum_linear_outlier_um,
            maximum_linear_outlier_um=maximum_linear_outlier_um,
            maximum_linear_median_residual_um=maximum_linear_median_residual_um,
            maximum_outlier_px=maximum_outlier_px,
            replacement_image_ncc_margin=replacement_image_ncc_margin,
        ),
    )
    typer.echo(json.dumps({"summary": str(summary)}, indent=2))


@cross_register_app.command("prefusion-qc")
def cross_register_prefusion_qc(
    summaries: Annotated[list[str], typer.Option("--summary", help="Recovery summary as LABEL=PATH.")],
    fixed_volume: Annotated[
        Path, typer.Option("--fixed-volume", exists=True, file_okay=False, readable=True)
    ],
    fixed_threshold: Annotated[float, typer.Option("--fixed-threshold", min=0.0)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Render pre-fusion dumb stitches and geometric tissue-hole maps."""
    output = render_prefusion_qc(
        summaries=_parse_labeled_paths(summaries, option="--summary"),
        fixed_volume=fixed_volume,
        fixed_threshold=fixed_threshold,
        output_dir=output_dir,
    )
    typer.echo(json.dumps({"contact_sheet": str(output)}, indent=2))


@cross_register_app.command("interpolated-window-qc")
def cross_register_interpolated_window_qc(
    summary: Annotated[Path, typer.Option("--summary", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    moving_label: Annotated[str, typer.Option("--moving-label")],
    fixed_label: Annotated[str, typer.Option("--fixed-label")],
    dataset_label: Annotated[str, typer.Option("--dataset-label")] = "",
    columns: Annotated[int, typer.Option("--columns", min=1)] = DEFAULT_INTERPOLATED_QC_COLUMNS,
) -> None:
    """Render overlays for windows accepted through interpolation."""
    result = render_interpolated_window_qc(
        summary_path=summary,
        output=output,
        moving_label=moving_label,
        fixed_label=fixed_label,
        dataset_label=dataset_label,
        columns=columns,
    )
    typer.echo(json.dumps({"contact_sheet": str(result)}, indent=2))


@cross_register_method8_app.command("coarse")
def cross_register_method8_coarse(
    fixed_position: Annotated[
        Path, typer.Option("--fixed-position", exists=True, dir_okay=False, readable=True)
    ],
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
        _require_empty_or_overwrite(
            output_registration, overwrite=overwrite, label="coarse output registration"
        )
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
    fixed_position: Annotated[
        Path, typer.Option("--fixed-position", exists=True, dir_okay=False, readable=True)
    ],
    coarse_moving_position: Annotated[
        Path,
        typer.Option("--coarse-moving-position", exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    fixed_channel: Annotated[int, typer.Option("--fixed-channel", min=0)] = 0,
    moving_channel: Annotated[int, typer.Option("--moving-channel", min=0)] = 0,
    pair_mode: Annotated[
        Literal["tile-number", "spatial-overlap"],
        typer.Option("--pair-mode", help="Pair corresponding tile numbers or every spatial overlap."),
    ] = "tile-number",
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
    empty_precheck_min_dynamic_range: Annotated[
        float, typer.Option("--empty-precheck-min-dynamic-range")
    ] = 1.0,
    empty_precheck_min_std: Annotated[float, typer.Option("--empty-precheck-min-std")] = 0.25,
    fixed_mask_threshold: Annotated[
        str,
        typer.Option(
            "--fixed-mask-threshold", help="Fixed-mask threshold; use none/off/disabled to turn it off."
        ),
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
        pair_mode=pair_mode,
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
    summary_payload = json.loads(path.read_text())
    global_affine = summary_payload.get("global_affine")
    manifest = _update_cross_register_manifest(
        output_dir,
        stage="method8",
        updates={
            "fixed_position": fixed_position.resolve(),
            "coarse_moving_position": coarse_moving_position.resolve(),
            "summary": path,
            "window_json_dir": window_json_dir.resolve(),
            "global_affine": global_affine,
            "fixed_channel": fixed_channel,
            "moving_channel": moving_channel,
            "pair_mode": pair_mode,
            "core_shape_zyx": list(parse_shape_zyx(core_shape_zyx)),
            "window_shape_zyx": list(parse_shape_zyx(window_shape_zyx)),
            "fixed_mask_threshold": _parse_optional_float(fixed_mask_threshold),
            "workers": workers,
            "devices": list(_parse_devices(devices)),
            "resume": resume,
        },
    )
    typer.echo(
        json.dumps(
            {
                "method8_summary": str(path),
                "window_json_dir": str(window_json_dir.resolve()),
                "global_affine": global_affine,
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


@app.command("fused-middle-z-qc")
def fused_middle_z_qc(
    channels: Annotated[list[str], typer.Option("--channel", help="Fused channel as LABEL=PATH.")],
    fixed_volume: Annotated[
        Path, typer.Option("--fixed-volume", exists=True, file_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_FUSED_MIDDLE_Z_LEVEL,
    contact_sheet_name: Annotated[str | None, typer.Option("--contact-sheet-name")] = None,
) -> None:
    """Render same-grid fused channels at their shared middle-Z plane."""
    result = render_fused_middle_z_qc(
        channels=_parse_labeled_paths(channels, option="--channel", kind="directory"),
        fixed_volume=fixed_volume,
        output_dir=output_dir,
        level=level,
        contact_sheet_name=contact_sheet_name,
    )
    typer.echo(
        json.dumps(
            {
                "images": [str(path) for path in result.images],
                "contact_sheet": (None if result.contact_sheet is None else str(result.contact_sheet)),
            },
            indent=2,
        )
    )


@app.command("threshold-review")
def threshold_review(
    fixed_fused: Annotated[
        Path,
        typer.Option("--fixed-fused", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    level: Annotated[int, typer.Option("--level", min=0)] = 2,
    z_index: Annotated[int | None, typer.Option("--z-index", min=0)] = None,
) -> None:
    """Write an unscaled fixed-channel OME-TIFF for manual mask-threshold review."""
    review_tiff, manifest = write_fused_threshold_review_tiff(
        fused_zarr=fixed_fused,
        output=output,
        level=level,
        z_index=z_index,
    )
    typer.echo(
        json.dumps(
            {"review_tiff": str(review_tiff), "manifest": str(manifest)},
            indent=2,
        )
    )


@cross_register_method8_app.command("materialize")
def cross_register_method8_materialize(
    window_json_dir: Annotated[
        Path, typer.Option("--window-json-dir", exists=True, file_okay=False, readable=True)
    ],
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
    jpegxr_level: Annotated[float, typer.Option("--jpegxr-level", min=0.0, max=1.0)] = (DEFAULT_JPEGXR_LEVEL),
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    resolved_output_dir = materialized_output_dir or _default_materialized_output_dir(
        output_dir,
        fusion_channel=fusion_channel,
    )
    _require_empty_or_overwrite(
        resolved_output_dir, overwrite=overwrite, label="materialized output directory"
    )
    outputs = export_tile_quadrant_materialized_chunks(
        window_json_dir=window_json_dir,
        moving_position_input=coarse_moving_position,
        fixed_registration_input=fixed_registration,
        output_dir=resolved_output_dir,
        channel_source_shift_px_zyx=(
            None if channel_source_shift_px_zyx is None else _parse_float_zyx(channel_source_shift_px_zyx)
        ),
        include_quality_gate_rejected=include_quality_gate_rejected,
        jpegxr_level=jpegxr_level,
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
                None
                if channel_source_shift_px_zyx is None
                else list(_parse_float_zyx(channel_source_shift_px_zyx))
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
    materialized_output_dir: Annotated[
        Path | None, typer.Option("--materialized-output-dir", file_okay=False)
    ] = None,
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
    materialized_registration = (
        resolved_materialized_dir / "tile_quadrant_materialized_chunks.registration.json"
    )
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


@app.command("plot-zeiss-positions")
def plot_zeiss_positions(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    title: Annotated[str, typer.Option("--title")] = "Zeiss XY positions",
    tile_positions: Annotated[
        Path | None,
        typer.Option("--tile-positions", exists=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    output_path = plot_xy_positions(
        input_path,
        output,
        title=title,
        tile_positions=tile_positions,
    )
    typer.echo(output_path)


@app.command("zeiss-tile-positions")
def zeiss_tile_positions(
    pos_input: Annotated[Path, typer.Option("--pos-input", exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    side: Annotated[str, typer.Option("--side")] = "single",
    overlap_fraction: Annotated[float, typer.Option("--overlap-fraction", min=0.0, max=0.99)] = 0.2,
    min_hull_overlap_fraction: Annotated[
        float, typer.Option("--min-hull-overlap-fraction", min=0.0, max=0.99)
    ] = 0.02,
) -> None:
    output_path = create_zeiss_tile_position_file(
        pos_input=pos_input,
        output=output,
        side=side,
        overlap_fraction=overlap_fraction,
        min_hull_overlap_fraction=min_hull_overlap_fraction,
    )
    typer.echo(output_path)


@app.command("single-position")
def single_position(
    input_dir: Annotated[Path, typer.Option("--input-dir", exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    side: Annotated[str, typer.Option("--side")],
    plot_title: Annotated[str, typer.Option("--plot-title")] = "metadata tile positions",
) -> None:
    create_single_position_file(
        input_dir=input_dir,
        output=output,
        side=side,
        plot_title=plot_title,
        progress=typer.echo,
    )
    typer.echo(output.resolve())


@app.command("rough-phase")
def rough_phase(
    position_input: Annotated[
        Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)
    ],
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
    phase_downsample_zyx: Annotated[str | None, typer.Option("--phase-downsample-zyx")] = _format_int_zyx(
        DEFAULT_PHASE_DOWNSAMPLE_ZYX
    ),
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
        phase_downsample_zyx=_parse_int_zyx(phase_downsample_zyx)
        if phase_downsample_zyx is not None
        else None,
    )
    typer.echo(output_position.resolve())


@app.command("mvs-edge-audit")
def mvs_edge_audit(
    registration_input: Annotated[
        Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)
    ],
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
    registration_input: Annotated[
        Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)
    ],
    output_registration: Annotated[Path, typer.Option("--output-registration")],
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    patch_shape_zyx: Annotated[str, typer.Option("--patch-shape-zyx")] = "12,320,320",
    span_entire_seam: Annotated[bool, typer.Option("--span-entire-seam/--scout-patches")] = False,
    patches_per_edge: Annotated[
        int, typer.Option("--patches-per-edge", min=1)
    ] = DEFAULT_LEVEL0_PATCHES_PER_EDGE,
    retry_patches_per_edge: Annotated[
        int, typer.Option("--retry-patches-per-edge", min=1)
    ] = DEFAULT_LEVEL0_RETRY_PATCHES_PER_EDGE,
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
    contact_sheet_output_dir: Annotated[
        Path | None, typer.Option("--contact-sheet-output-dir", file_okay=False)
    ] = None,
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


@app.command("rechunk-ome-zarr")
def rechunk_ome_zarr_command(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    destination: Annotated[Path, typer.Argument(file_okay=False)],
    start_level: Annotated[int, typer.Option("--start-level", min=0)] = 2,
    zstd_level: Annotated[int, typer.Option("--zstd-level", min=1, max=22)] = 3,
    chunk_shape_zyx: Annotated[str, typer.Option("--chunk-shape-zyx")] = ",".join(
        str(size) for size in DEFAULT_CHUNK_SHAPE_ZYX
    ),
    workers: Annotated[int, typer.Option("--workers", min=1)] = 8,
    codec_concurrency: Annotated[int, typer.Option("--codec-concurrency", min=1)] = 4,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    summary_output: Annotated[Path | None, typer.Option("--summary-output", dir_okay=False)] = None,
) -> None:
    """Rebase an OME-Zarr pyramid suffix into a lossless Zstd store."""
    summary = rechunk_ome_zarr(
        source=source,
        destination=destination,
        start_level=start_level,
        zstd_level=zstd_level,
        chunk_shape_zyx=_parse_int_zyx(chunk_shape_zyx),
        overwrite=overwrite,
        workers=workers,
        codec_concurrency=codec_concurrency,
        progress=_log_progress,
    )
    text = json.dumps(summary, indent=2)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(text + "\n")
        typer.echo(summary_output.resolve())
    else:
        typer.echo(text)


@app.command("fit-fused-subtraction")
def fit_fused_subtraction_command(
    target: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    reference: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    quantile: Annotated[float, typer.Option("--quantile", min=0.0, max=1.0)] = DEFAULT_SUBTRACTION_QUANTILE,
    planes: Annotated[int, typer.Option("--planes", min=1)] = 16,
    stride_yx: Annotated[int, typer.Option("--stride-yx", min=1)] = 32,
    max_samples: Annotated[int, typer.Option("--max-samples", min=2)] = 50_000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Fit target ~ alpha * reference + beta on the lower envelope of two completed same-grid fused volumes."""
    summary = fit_fused_subtraction(
        target=target,
        reference=reference,
        output_dir=output_dir,
        quantile=quantile,
        planes=planes,
        stride_yx=stride_yx,
        max_samples=max_samples,
        seed=seed,
    )
    typer.echo(json.dumps({key: summary[key] for key in summary if key != "per_plane"}, indent=2))


@app.command("subtract-fused")
def subtract_fused_command(
    model: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Argument(file_okay=False)],
    zstd_level: Annotated[int, typer.Option("--zstd-level", min=1, max=22)] = 3,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 8,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    summary_output: Annotated[Path | None, typer.Option("--summary-output", dir_okay=False)] = None,
) -> None:
    """Stream a fitted fused-volume subtraction over the shared multiscale grid."""
    summary = subtract_fused_volumes(
        model=model,
        output=output,
        zstd_level=zstd_level,
        workers=workers,
        overwrite=overwrite,
        progress=_log_progress,
    )
    text = json.dumps(summary, indent=2)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(text + "\n")
        typer.echo(summary_output.resolve())
    else:
        typer.echo(text)


@app.command("export-nrrd")
def export_nrrd_command(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    level: Annotated[int, typer.Option("--level", min=0)] = 0,
    channel: Annotated[str | None, typer.Option("--channel")] = None,
    gzip_level: Annotated[int, typer.Option("--gzip-level", min=0, max=9)] = DEFAULT_GZIP_LEVEL,
    slab_planes: Annotated[int, typer.Option("--slab-planes", min=1)] = DEFAULT_SLAB_PLANES,
    summary_output: Annotated[Path | None, typer.Option("--summary-output", dir_okay=False)] = None,
) -> None:
    """Export one OME-Zarr pyramid level as a validated Slicer NRRD."""
    summary = export_nrrd(
        source=source,
        output=output,
        level=level,
        channel=channel,
        gzip_level=gzip_level,
        slab_planes=slab_planes,
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
    registration_input: Annotated[
        Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)
    ],
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
    position_input: Annotated[
        Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)
    ],
    registration_input: Annotated[
        Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Canonical output directory, or an explicit .ome.zarr/.zarr base path.",
        ),
    ],
    channel: Annotated[list[int] | None, typer.Option("--channel", min=0)] = None,
    flatfield_dir_by_source_view: Annotated[
        list[str] | None,
        typer.Option("--flatfield-dir-by-source-view"),
    ] = None,
    residual_correction: Annotated[
        Path | None, typer.Option("--residual-correction", exists=True, dir_okay=False)
    ] = None,
    registered_view_correction: Annotated[
        Path | None,
        typer.Option("--registered-view-correction", exists=True, dir_okay=False, readable=True),
    ] = None,
    fusion_level: Annotated[int, typer.Option("--fusion-level", min=0)] = 0,
    fusion_weight_mode: Annotated[
        str,
        typer.Option("--fusion-weight-mode"),
    ] = "crop-sharpness-seam",
    seam_width_um: Annotated[
        float,
        typer.Option(
            "--seam-width-um",
            min=0.0,
            help="Full interface blend width for sharpness-seam fusion, in micrometers.",
        ),
    ] = 10.0,
    batch_size: Annotated[str, typer.Option("--batch-size")] = "auto",
    source_cache_max_gib: Annotated[
        float,
        typer.Option(
            "--source-cache-max-gib",
            min=0.0,
            help="Total host-memory budget for session-owned raw source chunks; 0 disables caching.",
        ),
    ] = 64.0,
    pipeline_staging_max_gib: Annotated[
        float,
        typer.Option(
            "--pipeline-staging-max-gib",
            min=0.0,
            help="Host-memory budget for overlapped prepare/GPU/write stages; 0 disables pipelining.",
        ),
    ] = 8.0,
    fusion_gpu_memory_max_gib: Annotated[
        float,
        typer.Option(
            "--fusion-gpu-memory-max-gib",
            min=0.0,
            help=(
                "GPU working-memory budget for internal x/y fusion sub-blocks; "
                "0 keeps one GPU computation per output chunk."
            ),
        ),
    ] = 0.0,
    per_chunk_cupy_cleanup: Annotated[
        bool,
        typer.Option(
            "--per-chunk-cupy-cleanup/--retain-cupy-pool",
            help=(
                "Release unused CuPy allocations after each output block; "
                "retain the pool only for measured allocator-reuse workloads."
            ),
        ),
    ] = True,
    output_chunksize_zyx: Annotated[
        str,
        typer.Option("--output-chunksize-zyx", help="Fusion output chunk size as z,y,x."),
    ] = ",".join(str(value) for value in DEFAULT_OUTPUT_CHUNKSIZE_ZYX),
    output_grid_template: Annotated[
        Path | None,
        typer.Option(
            "--output-grid-template",
            exists=True,
            file_okay=False,
            help="OME-Zarr whose scale-0 grid is reused.",
        ),
    ] = None,
    output_grid_template_level: Annotated[
        int,
        typer.Option(
            "--output-grid-template-level", min=0, help="Pyramid level to reuse from --output-grid-template."
        ),
    ] = 0,
    output_codec: Annotated[
        Literal["auto", "zstd", "jpegxr"],
        typer.Option(
            "--output-codec",
            help="auto uses JPEG-XR for native level 0 and Zstd for downsampled preview output.",
        ),
    ] = "auto",
    zstd_level: Annotated[int, typer.Option("--zstd-level", min=-7, max=22)] = 3,
    jpegxr_level: Annotated[float, typer.Option("--jpegxr-level", min=0.0, max=1.0)] = (DEFAULT_JPEGXR_LEVEL),
    resume_fusion: Annotated[
        bool,
        typer.Option(
            "--resume-fusion",
            help="Resume only a preserved temporary fusion whose inputs and pixel-affecting settings match.",
        ),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    fuse_tiles(
        input_dir=input_dir,
        position_input=position_input,
        registration_input=registration_input,
        output=output,
        channels=channel,
        flatfield_dirs_by_source_view=_parse_source_view_flatfield_dirs(flatfield_dir_by_source_view),
        residual_correction=residual_correction,
        registered_view_correction=registered_view_correction,
        fusion_level=fusion_level,
        fusion_weight_mode=fusion_weight_mode,
        seam_width_um=seam_width_um,
        batch_size=parse_fusion_batch_size(batch_size),
        source_cache_max_gib=source_cache_max_gib,
        pipeline_staging_max_gib=pipeline_staging_max_gib,
        fusion_gpu_memory_max_gib=fusion_gpu_memory_max_gib,
        per_chunk_cupy_cleanup=per_chunk_cupy_cleanup,
        output_chunksize_zyx=_parse_int_zyx(output_chunksize_zyx, "--output-chunksize-zyx"),
        output_grid_template=output_grid_template,
        output_grid_template_level=output_grid_template_level,
        output_codec=output_codec,
        zstd_level=zstd_level,
        jpegxr_level=jpegxr_level,
        resume_fusion=resume_fusion,
        dry_run=dry_run,
    )


@app.command("subtract-channel")
def subtract_channel(
    position_input: Annotated[
        Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    output_position: Annotated[Path | None, typer.Option("--output-position")] = None,
    registration_input: Annotated[
        Path | None,
        typer.Option("--registration-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    reference_position_input: Annotated[
        Path | None,
        typer.Option("--reference-position-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    reference_registration_input: Annotated[
        Path | None,
        typer.Option("--reference-registration-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    subtraction_model: Annotated[
        Path | None,
        typer.Option("--subtraction-model", exists=True, dir_okay=False, readable=True),
    ] = None,
    side: Annotated[Literal["L", "R"] | None, typer.Option("--side")] = None,
    output_registration: Annotated[Path | None, typer.Option("--output-registration")] = None,
    target_channel: Annotated[int, typer.Option("--target-channel", min=0)] = 2,
    reference_channel: Annotated[int, typer.Option("--reference-channel", min=0)] = 3,
    source_level: Annotated[int, typer.Option("--source-level", min=0)] = 2,
    reference_shift_zyx_px: Annotated[str, typer.Option("--reference-shift-zyx-px")] = "0,0,0",
    alpha: Annotated[float, typer.Option("--alpha")] = 0.0,
    beta: Annotated[float, typer.Option("--beta")] = 0.0,
    target_background: Annotated[float, typer.Option("--target-background")] = 0.0,
    reference_background: Annotated[float, typer.Option("--reference-background")] = 0.0,
    target_residual_correction: Annotated[
        Path | None,
        typer.Option("--target-residual-correction", exists=True, dir_okay=False, readable=True),
    ] = None,
    reference_residual_correction: Annotated[
        Path | None,
        typer.Option("--reference-residual-correction", exists=True, dir_okay=False, readable=True),
    ] = None,
    crop_yx_px: Annotated[int, typer.Option("--crop-yx-px", min=0)] = 20,
    z_chunk: Annotated[int, typer.Option("--z-chunk", min=1)] = 64,
    chunk_shape_zyx: Annotated[str, typer.Option("--chunk-shape-zyx")] = _format_int_zyx(
        DEFAULT_SUBTRACTION_CHUNK_SHAPE_ZYX
    ),
    zstd_level: Annotated[int, typer.Option("--zstd-level", min=1, max=22)] = DEFAULT_SUBTRACTION_ZSTD_LEVEL,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    limit_tiles: Annotated[int | None, typer.Option("--limit-tiles", min=1)] = None,
) -> None:
    """Write spillover-subtracted tiles for later BaSiC correction and fusion.

    The command is a producer stage: it reads raw target/reference channels from
    each source tile, shifts the reference into target local coordinates, applies
    the subtraction model, crops y/x borders, and writes corrected single-channel
    OME-Zarrs plus position/registration metadata. Downstream fusion treats the
    output as an ordinary single-channel acquisition.
    """
    paired_values = (reference_position_input, reference_registration_input, subtraction_model, side)
    if any(value is not None for value in paired_values):
        if not all(value is not None for value in paired_values):
            raise typer.BadParameter(
                "paired mode requires --reference-position-input, --reference-registration-input, "
                "--subtraction-model, and --side"
            )
        if registration_input is None:
            raise typer.BadParameter("paired mode requires --registration-input")
        if target_residual_correction is None or reference_residual_correction is None:
            raise typer.BadParameter(
                "paired mode requires --target-residual-correction and --reference-residual-correction"
            )
        if overwrite:
            raise typer.BadParameter("paired mode does not permit --overwrite")
        result = subtract_paired_channel_tiles(
            position_input=position_input,
            registration_input=registration_input,
            reference_position_input=reference_position_input,
            reference_registration_input=reference_registration_input,
            subtraction_model=subtraction_model,
            output_dir=output_dir,
            side=side,
            target_residual_correction=target_residual_correction,
            reference_residual_correction=reference_residual_correction,
            target_channel=target_channel,
            reference_channel=reference_channel,
            crop_yx_px=crop_yx_px,
            z_chunk=z_chunk,
            chunk_shape_zyx=parse_shape_zyx(chunk_shape_zyx),
            zstd_level=zstd_level,
            limit_tiles=limit_tiles,
            progress=_log_progress,
        )
    else:
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
            target_residual_correction=target_residual_correction,
            reference_residual_correction=reference_residual_correction,
            crop_yx_px=crop_yx_px,
            z_chunk=z_chunk,
            chunk_shape_zyx=parse_shape_zyx(chunk_shape_zyx),
            zstd_level=zstd_level,
            overwrite=overwrite,
            limit_tiles=limit_tiles,
            progress=_log_progress,
        )
    typer.echo(result.position_output.resolve())
    if result.registration_output is not None:
        typer.echo(result.registration_output.resolve())


@app.command("join-subtracted-sides")
def join_subtracted_sides(
    side_output_dir: Annotated[
        list[Path],
        typer.Option(
            "--side-output-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Completed paired-subtraction side output; repeat for L and R.",
        ),
    ],
    output_root: Annotated[Path, typer.Option("--output-root", file_okay=False)],
) -> None:
    """Join complete L/R subtraction tiles and metadata without refitting geometry."""
    result = join_paired_subtraction_sides(
        side_output_dirs=side_output_dir,
        output_root=output_root,
    )
    typer.echo(result.position_output)
    typer.echo(result.registration_output)
    typer.echo(result.summary_output)


@app.command("join-materialized-sides")
def join_materialized_sides_command(
    side_output_dir: Annotated[
        list[Path],
        typer.Option(
            "--side-output-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Completed fused-fixed-materialize-overlap output; repeat for each dataset.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Join per-dataset materializations on one fixed grid into a single fusion input."""
    outputs = join_materialized_sides(side_output_dirs=side_output_dir, output_dir=output_dir)
    typer.echo(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


@app.command()
def pyramid(
    ome_zarr: Annotated[list[Path], typer.Argument()],
    template: Annotated[Path | None, typer.Option("--template", exists=True, file_okay=False)] = None,
    jpegxr_level: Annotated[float, typer.Option("--jpegxr-level", min=0.0, max=1.0)] = (DEFAULT_JPEGXR_LEVEL),
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = False,
) -> None:
    add_pyramids(
        ome_zarrs=ome_zarr,
        template=template,
        jpegxr_level=jpegxr_level,
        dry_run=dry_run,
    )


@app.command()
def qc(
    position_input: Annotated[
        Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)
    ],
    registration_input: Annotated[
        Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)
    ],
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
    registration_input: Annotated[
        Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    level: Annotated[int, typer.Option("--level", min=0)] = 2,
    z_index: Annotated[int | None, typer.Option("--z-index", min=0)] = None,
    draw_labels: Annotated[bool, typer.Option("--labels/--no-labels")] = True,
    draw_markers: Annotated[bool, typer.Option("--markers/--no-markers")] = True,
) -> None:
    path = render_fused_tile_index_overlay(
        fused_zarr=fused_zarr,
        registration_input=registration_input,
        output=output,
        level=level,
        z_index=z_index,
        draw_labels=draw_labels,
        draw_markers=draw_markers,
    )
    typer.echo(path.resolve())


@app.command("live-fusion-preview")
def live_fusion_preview(
    source_zarr: Annotated[
        Path | None, typer.Option("--zarr", exists=True, file_okay=False, readable=True)
    ] = None,
    log_path: Annotated[
        Path | None, typer.Option("--log", exists=True, dir_okay=False, readable=True)
    ] = None,
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
    reference_zarr: Annotated[
        Path, typer.Option("--reference-zarr", exists=True, file_okay=False, readable=True)
    ],
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
    position_input: Annotated[
        Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)
    ],
    registration_input: Annotated[
        Path, typer.Option("--registration-input", exists=True, dir_okay=False, readable=True)
    ],
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


@app.command("ome-metadata-dumb-stitch")
def ome_metadata_dumb_stitch(
    input_dir: Annotated[
        list[str] | None,
        typer.Option("--input-dir", help="Repeatable VIEW=DIR source tile directory entry."),
    ] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)] = Path(
        "ome-metadata-dumb-stitch-qc"
    ),
    channels: Annotated[str, typer.Option("--channels", help="Comma-separated channel indexes.")] = "0,1",
    basic_dir: Annotated[
        Path | None, typer.Option("--basic-dir", exists=True, file_okay=False, readable=True)
    ] = None,
    level: Annotated[int, typer.Option("--level", min=0, help="OME-Zarr pyramid level to render.")] = 0,
    center_z_index: Annotated[
        int | None,
        typer.Option(
            "--center-z-index", min=0, help="Z index at the selected level; defaults to its center."
        ),
    ] = None,
    output_prefix: Annotated[str, typer.Option("--output-prefix")] = "ome_metadata_dumb_stitch",
    draw_tile_labels: Annotated[bool, typer.Option("--draw-tile-labels/--no-draw-tile-labels")] = False,
    draw_tile_outlines: Annotated[bool, typer.Option("--draw-tile-outlines/--no-draw-tile-outlines")] = False,
    tiff_layout: Annotated[
        Literal["separate", "channels"],
        typer.Option("--tiff-layout", help="Write separate TIFFs or one CYX TIFF per view and correction."),
    ] = "separate",
    write_tiff: Annotated[
        bool,
        typer.Option(
            "--write-tiff/--no-write-tiff",
            help=(
                "Also write tiled native-intensity OME-TIFF mosaics; raw outputs are exact uint16, "
                "and corrected outputs remain float32."
            ),
        ),
    ] = False,
) -> None:
    if not input_dir:
        raise typer.BadParameter("Pass at least one --input-dir VIEW=DIR entry")
    input_dirs_by_view: dict[str, Path] = {}
    for value in input_dir:
        try:
            view, path = parse_ome_metadata_dumb_stitch_view_dir(value)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        if view in input_dirs_by_view:
            raise typer.BadParameter(f"Duplicate --input-dir view {view!r}")
        if not path.is_dir():
            raise typer.BadParameter(f"--input-dir {view} path is not a directory: {path}")
        input_dirs_by_view[view] = path
    try:
        parsed_channels = parse_ome_metadata_dumb_stitch_channels(channels)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    result = render_ome_metadata_dumb_stitch(
        input_dirs_by_view=input_dirs_by_view,
        output_dir=output_dir,
        channels=parsed_channels,
        basic_dir=basic_dir,
        level=level,
        center_z_index=center_z_index,
        output_prefix=output_prefix,
        draw_tile_labels=draw_tile_labels,
        draw_tile_outlines=draw_tile_outlines,
        write_tiff=write_tiff,
        tiff_layout=tiff_layout,
        progress=_log_progress,
    )
    typer.echo(
        json.dumps(
            {
                "manifest": str(result.manifest_path.resolve()),
                "contact_sheet": str(result.contact_sheet_path.resolve()),
                "outputs": [str(path.resolve()) for path in result.output_paths],
            },
            indent=2,
        )
    )


@app.command("track-z-diagnostics")
def track_z_diagnostics(
    position_input: Annotated[
        Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)
    ],
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
    overlap_fraction: Annotated[
        float, typer.Option("--overlap-fraction", min=0.0, max=0.999)
    ] = DEFAULT_OVERLAP_FRACTION,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LR_ALIGNMENT_LEVEL,
    search_margin_px: Annotated[int, typer.Option("--search-margin-px", min=0)] = 64,
    phase_upsample_factor: Annotated[int, typer.Option("--phase-upsample-factor", min=1)] = 10,
    seam_fraction: Annotated[float, typer.Option("--seam-fraction", min=0.0, max=1.0)] = 0.10,
    z_slab_planes: Annotated[int, typer.Option("--z-slab-planes", min=1)] = DEFAULT_Z_SLAB_PLANES,
    phase_downsample_zyx: Annotated[str, typer.Option("--phase-downsample-zyx")] = _format_int_zyx(
        DEFAULT_PHASE_DOWNSAMPLE_ZYX
    ),
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


@app.command("align-registered-lr")
def align_registered_lr(
    metadata_position: Annotated[
        Path, typer.Option("--metadata-position", exists=True, dir_okay=False, readable=True)
    ],
    left_position: Annotated[
        Path, typer.Option("--left-position", exists=True, dir_okay=False, readable=True)
    ],
    right_position: Annotated[
        Path, typer.Option("--right-position", exists=True, dir_okay=False, readable=True)
    ],
    left_fusion: Annotated[Path, typer.Option("--left-fusion", exists=True, file_okay=False, readable=True)],
    right_fusion: Annotated[
        Path, typer.Option("--right-fusion", exists=True, file_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    fusion_level: Annotated[int, typer.Option("--fusion-level", min=0)] = 0,
    manual_cr_shift_zyx_um: Annotated[
        str | None,
        typer.Option(
            "--manual-cr-shift-zyx-um",
            help="Manual CR correction as comma-separated z,y,x micrometers; skips phase correlation.",
        ),
    ] = None,
) -> None:
    """Place registered CR into CL space from fused z-median phase correlation."""
    manifest = align_registered_lr_fusions(
        metadata_position=metadata_position,
        left_position=left_position,
        right_position=right_position,
        left_fusion=left_fusion,
        right_fusion=right_fusion,
        output_dir=output_dir,
        fusion_level=fusion_level,
        manual_cr_shift_zyx_um=(
            _parse_float_zyx(manual_cr_shift_zyx_um) if manual_cr_shift_zyx_um is not None else None
        ),
    )
    typer.echo(manifest.resolve())


@app.command("tile-phase-align")
def tile_phase_align(
    reference_position: Annotated[
        Path, typer.Option("--reference-position", exists=True, dir_okay=False, readable=True)
    ],
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
    affine_refine_crop_shape_zyx: Annotated[
        str, typer.Option("--affine-refine-crop-shape-zyx")
    ] = "200,480,480",
    affine_max_iterations: Annotated[int, typer.Option("--affine-max-iterations", min=0)] = 20,
    affine_contact_sheet: Annotated[
        bool, typer.Option("--affine-contact-sheet/--no-affine-contact-sheet")
    ] = True,
    affine_fit_mode: Annotated[
        Literal["rigid", "affine-12dof"],
        typer.Option(
            "--affine-fit-mode", help="Transform family for affine-mode cross-channel registration."
        ),
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
    script_dir: Annotated[
        Path, typer.Option("--script-dir", exists=True, file_okay=False, readable=True)
    ] = ...,
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
    records_jsonl: Annotated[
        Path, typer.Option("--records-jsonl", exists=True, dir_okay=False, readable=True)
    ],
    reference_position: Annotated[
        Path,
        typer.Option("--reference-position", exists=True, dir_okay=False, readable=True),
    ],
    moving_position: Annotated[
        Path, typer.Option("--moving-position", exists=True, dir_okay=False, readable=True)
    ],
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
        typer.Option(
            "--candidate-source", help="Use seed_rows for current workflow fidelity, or structure_tensor."
        ),
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
    inlier_threshold_level_px_zyx: Annotated[
        str, typer.Option("--inlier-threshold-level-px-zyx")
    ] = "3,12,12",
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
    phase_position: Annotated[
        Path, typer.Option("--phase-position", exists=True, dir_okay=False, readable=True)
    ],
    source_405_registration: Annotated[
        Path,
        typer.Option("--source-405-registration", exists=True, dir_okay=False, readable=True),
    ],
    seam_residuals: Annotated[
        Path, typer.Option("--seam-residuals", exists=True, dir_okay=False, readable=True)
    ],
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
    position_input: Annotated[
        Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)
    ],
    registration_input: Annotated[
        Path,
        typer.Option("--registration-input", exists=True, dir_okay=False, readable=True),
    ],
    candidate_json: Annotated[
        Path, typer.Option("--candidate-json", exists=True, dir_okay=False, readable=True)
    ],
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
    all_tiles_json: Annotated[
        Path, typer.Option("--all-tiles-json", exists=True, dir_okay=False, readable=True)
    ],
    coarse_anchor_table: Annotated[
        Path,
        typer.Option("--coarse-anchor-table", exists=True, dir_okay=False, readable=True),
    ],
    phase_position: Annotated[
        Path, typer.Option("--phase-position", exists=True, dir_okay=False, readable=True)
    ],
    source_405_registration: Annotated[
        Path,
        typer.Option("--source-405-registration", exists=True, dir_okay=False, readable=True),
    ],
    seam_residuals: Annotated[
        Path, typer.Option("--seam-residuals", exists=True, dir_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    script_dir: Annotated[
        Path, typer.Option("--script-dir", exists=True, file_okay=False, readable=True)
    ] = ...,
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
    overlap_fraction: Annotated[
        float, typer.Option("--overlap-fraction", min=0.0, max=0.999)
    ] = DEFAULT_OVERLAP_FRACTION,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    rough_phase_level: Annotated[
        int, typer.Option("--rough-phase-level", min=0)
    ] = DEFAULT_LR_ALIGNMENT_LEVEL,
    search_margin_px: Annotated[int, typer.Option("--search-margin-px", min=0)] = 64,
    phase_upsample_factor: Annotated[int, typer.Option("--phase-upsample-factor", min=1)] = 10,
    seam_fraction: Annotated[float, typer.Option("--seam-fraction", min=0.0, max=1.0)] = 0.10,
    z_slab_planes: Annotated[int, typer.Option("--z-slab-planes", min=1)] = DEFAULT_Z_SLAB_PLANES,
    phase_downsample_zyx: Annotated[str, typer.Option("--phase-downsample-zyx")] = _format_int_zyx(
        DEFAULT_PHASE_DOWNSAMPLE_ZYX
    ),
    registration_pair_mode: Annotated[
        str, typer.Option("--registration-pair-mode")
    ] = legacy_registration.DEFAULT_REGISTRATION_PAIR_MODE,
    registration_pair_file: Annotated[
        Path | None,
        typer.Option("--registration-pair-file", exists=True, dir_okay=False, readable=True),
    ] = None,
    skip_registration_plots: Annotated[
        bool, typer.Option("--skip-registration-plots/--registration-plots")
    ] = True,
    dask_registration_workers: Annotated[
        int | None, typer.Option("--dask-registration-workers", min=1)
    ] = None,
    pairwise_jobs: Annotated[
        int | None, typer.Option("--pairwise-jobs", min=0)
    ] = legacy_registration.DEFAULT_N_PARALLEL_PAIRWISE_REGS,
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
