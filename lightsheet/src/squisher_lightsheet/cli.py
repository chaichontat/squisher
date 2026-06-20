from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from squisher_lightsheet.fusion import fuse_tiles
from squisher_lightsheet.modes import ModeName
from squisher_lightsheet.positions import create_position_file
from squisher_lightsheet.pyramid import add_pyramids
from squisher_lightsheet.qc import render_registration_qc
from squisher_lightsheet.registration import register_tiles
from squisher_lightsheet.rough_phase import rough_phase_align
from squisher_lightsheet.tile_phase import align_tiles_to_reference, parse_shape_zyx
from squisher_lightsheet.track_z import run_track_z_diagnostics
from squisher_lightsheet.workflow import DEFAULT_LEVEL, DEFAULT_OVERLAP_FRACTION, run_tltr_workflow


app = typer.Typer(no_args_is_help=True)


def _parse_source_view_flatfield_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise typer.BadParameter(f"Expected source-view flatfield entry as VIEW=DIR, got {value!r}")
    view, path = value.split("=", 1)
    view = view.strip()
    if not view:
        raise typer.BadParameter(f"Missing source-view name in {value!r}")
    if not path:
        raise typer.BadParameter(f"Missing flatfield directory in {value!r}")
    return view, Path(path)


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
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    search_margin_px: Annotated[int, typer.Option("--search-margin-px", min=0)] = 64,
    upsample_factor: Annotated[int, typer.Option("--upsample-factor", min=1)] = 10,
    seam_fraction: Annotated[float, typer.Option("--seam-fraction", min=0.0, max=1.0)] = 0.10,
    crop_overlap: Annotated[bool, typer.Option("--crop-overlap/--no-crop-overlap")] = True,
    z_slab_planes: Annotated[int, typer.Option("--z-slab-planes", min=1)] = 1,
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
    )
    typer.echo(output_position.resolve())


@app.command()
def register(
    run_dir: Annotated[Path, typer.Argument()],
    position_input: Annotated[Path, typer.Option("--position-input", exists=True, dir_okay=False, readable=True)],
    registration_output: Annotated[Path, typer.Option("--registration-output")],
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    registration_pair_mode: Annotated[str, typer.Option("--registration-pair-mode")] = "robust-boundary",
    robust_boundary_qc_dir: Annotated[Path | None, typer.Option("--robust-boundary-qc-dir")] = None,
    registration_plots_dir: Annotated[Path | None, typer.Option("--registration-plots-dir")] = None,
    skip_registration_plots: Annotated[bool, typer.Option("--skip-registration-plots/--registration-plots")] = True,
    dask_num_workers: Annotated[int | None, typer.Option("--dask-num-workers", min=1)] = None,
    pairwise_jobs: Annotated[int | None, typer.Option("--pairwise-jobs", min=0)] = None,
    reference_registration_input: Annotated[
        Path | None,
        typer.Option("--reference-registration-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    reference_geometry_mode: Annotated[str, typer.Option("--reference-geometry-mode")] = "none",
    reference_xy_prior_weight: Annotated[float | None, typer.Option("--reference-xy-prior-weight", min=0.0)] = None,
    shared_geometry_tracks: Annotated[str | None, typer.Option("--shared-geometry-tracks")] = None,
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
        reference_registration_input=reference_registration_input,
        reference_geometry_mode=reference_geometry_mode,
        reference_xy_prior_weight=reference_xy_prior_weight,
        shared_geometry_tracks=(
            tuple(item.strip() for item in shared_geometry_tracks.split(",") if item.strip())
            if shared_geometry_tracks is not None
            else None
        ),
        dry_run=dry_run,
    )


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
        fusion_weight_mode=fusion_weight_mode,
        batch_size=batch_size,
        basic_cache_disk_dir=basic_cache_disk_dir,
        dry_run=dry_run,
    )


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


@app.command("run-tltr")
def run_tltr(
    left_dir: Annotated[Path, typer.Option("--left-dir", exists=True, file_okay=False, readable=True)],
    right_dir: Annotated[Path, typer.Option("--right-dir", exists=True, file_okay=False, readable=True)],
    output_prefix: Annotated[Path, typer.Option("--output-prefix")],
    overlap_fraction: Annotated[float, typer.Option("--overlap-fraction", min=0.0, max=0.999)] = DEFAULT_OVERLAP_FRACTION,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    search_margin_px: Annotated[int, typer.Option("--search-margin-px", min=0)] = 64,
    phase_upsample_factor: Annotated[int, typer.Option("--phase-upsample-factor", min=1)] = 10,
    seam_fraction: Annotated[float, typer.Option("--seam-fraction", min=0.0, max=1.0)] = 0.10,
    registration_pair_mode: Annotated[str, typer.Option("--registration-pair-mode")] = "robust-boundary",
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
        search_margin_px=search_margin_px,
        phase_upsample_factor=phase_upsample_factor,
        seam_fraction=seam_fraction,
        registration_pair_mode=registration_pair_mode,
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
