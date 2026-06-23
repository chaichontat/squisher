from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

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
from squisher_lightsheet.fusion import fuse_tiles
from squisher_lightsheet.lr_alignment import (
    DEFAULT_LEVEL as DEFAULT_LR_ALIGNMENT_LEVEL,
    DEFAULT_PHASE_DOWNSAMPLE_ZYX,
    DEFAULT_Z_SLAB_PLANES,
    run_lr_dumb_stitch_alignment,
)
from squisher_lightsheet.modes import ModeName
from squisher_lightsheet.mvs_seams import load_mvs_registration, mvs_used_edge_audit, score_mvs_edges_with_gradient_ncc
from squisher_lightsheet.parsing import parse_source_view_path_entry
from squisher_lightsheet.positions import create_position_file
from squisher_lightsheet.pyramid import add_pyramids
from squisher_lightsheet.qc import render_registration_qc
from squisher_lightsheet.registration import register_tiles
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy_registration
from squisher_lightsheet.rough_phase import rough_phase_align
from squisher_lightsheet.tile_phase import align_tiles_to_reference, parse_shape_zyx
from squisher_lightsheet.track_z import run_track_z_diagnostics
from squisher_lightsheet.workflow import DEFAULT_LEVEL, DEFAULT_OVERLAP_FRACTION, run_tltr_workflow


app = typer.Typer(no_args_is_help=True)
align_405_to_488_app = typer.Typer(no_args_is_help=True)
app.add_typer(align_405_to_488_app, name="align-405-to-488")


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


def _parse_float_zyx(text: str) -> tuple[float, float, float]:
    values = tuple(float(part) for part in text.split(","))
    if len(values) != 3:
        raise typer.BadParameter(f"Expected comma-separated z,y,x values, got {text!r}")
    return values


def _parse_int_zyx(text: str) -> tuple[int, int, int]:
    values = tuple(int(part) for part in text.split(","))
    if len(values) != 3:
        raise typer.BadParameter(f"Expected comma-separated z,y,x values, got {text!r}")
    return values


def _parse_float_yx(text: str) -> tuple[float, float]:
    values = tuple(float(part) for part in text.split(","))
    if len(values) != 2:
        raise typer.BadParameter(f"Expected comma-separated y,x values, got {text!r}")
    return values


def _format_int_zyx(values: tuple[int, int, int]) -> str:
    return ",".join(str(value) for value in values)


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
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    registration_pair_mode: Annotated[str, typer.Option("--registration-pair-mode")] = legacy_registration.DEFAULT_REGISTRATION_PAIR_MODE,
    robust_boundary_qc_dir: Annotated[Path | None, typer.Option("--robust-boundary-qc-dir")] = None,
    registration_plots_dir: Annotated[Path | None, typer.Option("--registration-plots-dir")] = None,
    skip_registration_plots: Annotated[bool, typer.Option("--skip-registration-plots/--registration-plots")] = True,
    dask_num_workers: Annotated[int | None, typer.Option("--dask-num-workers", min=1)] = None,
    pairwise_jobs: Annotated[int | None, typer.Option("--pairwise-jobs", min=0)] = None,
    registration_pair_file: Annotated[
        Path | None,
        typer.Option("--registration-pair-file", exists=True, dir_okay=False, readable=True),
    ] = None,
    groupwise_transform: Annotated[str, typer.Option("--groupwise-transform")] = legacy_registration.MVS_GROUPWISE_TRANSFORM,
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
        registration_pair_file=registration_pair_file,
        groupwise_transform=groupwise_transform,
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
        progress=typer.echo,
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
        progress=typer.echo,
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
    reference_token: Annotated[str, typer.Option("--reference-token")] = "488514561638",
    moving_token: Annotated[str, typer.Option("--moving-token")] = "405",
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    upsample_factor: Annotated[int, typer.Option("--upsample-factor", min=1)] = 10,
    patch_shape_zyx: Annotated[str | None, typer.Option("--patch-shape-zyx")] = None,
    min_inliers: Annotated[int, typer.Option("--min-inliers", min=3)] = 3,
    max_candidate_patches: Annotated[int, typer.Option("--max-candidate-patches", min=1)] = 24,
    coarse_level: Annotated[int, typer.Option("--coarse-level", min=0)] = DEFAULT_LEVEL,
    reference_registration_input: Annotated[
        Path | None,
        typer.Option("--reference-registration-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    output_registration: Annotated[Path | None, typer.Option("--output-registration")] = None,
) -> None:
    path = align_tiles_to_reference(
        reference_position=reference_position,
        output_position=output_position,
        output_dir=output_dir,
        reference_channel=reference_channel,
        reference_token=reference_token,
        moving_token=moving_token,
        level=level,
        upsample_factor=upsample_factor,
        patch_shape_zyx=None if patch_shape_zyx is None else parse_shape_zyx(patch_shape_zyx),
        min_inliers=min_inliers,
        max_candidate_patches=max_candidate_patches,
        coarse_level=coarse_level,
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
    registration_pair_mode: Annotated[str, typer.Option("--registration-pair-mode")] = "robust-boundary",
    registration_pair_file: Annotated[
        Path | None,
        typer.Option("--registration-pair-file", exists=True, dir_okay=False, readable=True),
    ] = None,
    skip_registration_plots: Annotated[bool, typer.Option("--skip-registration-plots/--registration-plots")] = True,
    dask_registration_workers: Annotated[int | None, typer.Option("--dask-registration-workers", min=1)] = None,
    pairwise_jobs: Annotated[int | None, typer.Option("--pairwise-jobs", min=0)] = None,
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
