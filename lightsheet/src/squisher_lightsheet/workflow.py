from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from squisher_lightsheet.artifacts import write_workflow_summary
from squisher_lightsheet.fusion import channel_output_paths, fuse_tiles
from squisher_lightsheet.lr_alignment import (
    DEFAULT_LEVEL as DEFAULT_ROUGH_PHASE_LEVEL,
    DEFAULT_OVERLAP_FRACTION,
    DEFAULT_PHASE_DOWNSAMPLE_ZYX,
    DEFAULT_Z_SLAB_PLANES,
    LRDumbStitchAlignmentPaths,
    lr_dumb_stitch_alignment_commands,
    lr_dumb_stitch_alignment_paths,
    run_lr_dumb_stitch_alignment,
)
from squisher_lightsheet.pyramid import add_pyramids
from squisher_lightsheet.qc import render_registration_qc
from squisher_lightsheet.registration import register_tiles


DEFAULT_LEVEL = 4


@dataclass(frozen=True)
class WorkflowPaths:
    metadata_position: Path
    phase_position: Path
    phase_qc_dir: Path
    initial_overlay: Path
    corrected_overlay: Path
    phase_summary_json: Path
    registration_dir: Path
    registration_json: Path
    registration_plots_dir: Path
    robust_boundary_qc_dir: Path
    registration_qc_dir: Path
    summary_json: Path
    fusion_base_output: Path


def workflow_paths(
    output_prefix: Path,
    *,
    level: int = DEFAULT_LEVEL,
    rough_phase_level: int = DEFAULT_ROUGH_PHASE_LEVEL,
    z_slab_planes: int = DEFAULT_Z_SLAB_PLANES,
    channel: int = 0,
) -> WorkflowPaths:
    run_dir = output_prefix.with_name(f"{output_prefix.name}-roughPhase-registration-level{level}")
    lr_paths = lr_dumb_stitch_alignment_paths(
        output_prefix,
        level=rough_phase_level,
        channel=channel,
        z_slab_planes=z_slab_planes,
        run_dir=run_dir,
    )
    return WorkflowPaths(
        metadata_position=lr_paths.metadata_position,
        phase_position=lr_paths.phase_position,
        phase_qc_dir=lr_paths.phase_qc_dir,
        initial_overlay=lr_paths.initial_overlay,
        corrected_overlay=lr_paths.corrected_overlay,
        phase_summary_json=lr_paths.phase_summary_json,
        registration_dir=run_dir,
        registration_json=run_dir / "registration.json",
        registration_plots_dir=run_dir / "registration-plots",
        robust_boundary_qc_dir=run_dir / "robust-boundary-qc",
        registration_qc_dir=run_dir / f"level{level}-registration-qc",
        summary_json=run_dir / "workflow_summary.json",
        fusion_base_output=run_dir / "fused.ome.zarr",
    )


def run_tltr_workflow(
    *,
    left_dir: Path,
    right_dir: Path,
    output_prefix: Path,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
    channel: int = 0,
    level: int = DEFAULT_LEVEL,
    rough_phase_level: int = DEFAULT_ROUGH_PHASE_LEVEL,
    search_margin_px: int = 64,
    phase_upsample_factor: int = 10,
    seam_fraction: float = 0.10,
    z_slab_planes: int = DEFAULT_Z_SLAB_PLANES,
    phase_downsample_zyx: tuple[int, int, int] = DEFAULT_PHASE_DOWNSAMPLE_ZYX,
    registration_pair_mode: str = "robust-boundary",
    registration_pair_file: Path | None = None,
    skip_registration_plots: bool = True,
    dask_registration_workers: int | None = None,
    pairwise_jobs: int | None = None,
    do_fuse: bool = False,
    do_pyramid: bool = False,
    dry_run: bool = False,
) -> WorkflowPaths:
    if do_pyramid and not do_fuse:
        raise ValueError("do_pyramid requires do_fuse")

    paths = workflow_paths(
        output_prefix.resolve(),
        level=level,
        rough_phase_level=rough_phase_level,
        z_slab_planes=z_slab_planes,
        channel=channel,
    )
    commands: dict[str, str] = {}
    lr_paths = LRDumbStitchAlignmentPaths(
        metadata_position=paths.metadata_position,
        phase_position=paths.phase_position,
        phase_qc_dir=paths.phase_qc_dir,
        initial_overlay=paths.initial_overlay,
        corrected_overlay=paths.corrected_overlay,
        phase_summary_json=paths.phase_summary_json,
        summary_json=paths.registration_dir / "lr_dumb_stitch_alignment_summary.json",
    )
    outputs = {
        "metadata_position": str(paths.metadata_position.resolve()),
        "phase_position": str(paths.phase_position.resolve()),
        "phase_qc_dir": str(paths.phase_qc_dir.resolve()),
        "initial_overlay": str(paths.initial_overlay.resolve()),
        "corrected_overlay": str(paths.corrected_overlay.resolve()),
        "phase_summary_json": str(paths.phase_summary_json.resolve()),
        "registration_dir": str(paths.registration_dir.resolve()),
        "registration_json": str(paths.registration_json.resolve()),
        "registration_plots_dir": str(paths.registration_plots_dir.resolve()),
        "robust_boundary_qc_dir": str(paths.robust_boundary_qc_dir.resolve()),
        "registration_qc_dir": str(paths.registration_qc_dir.resolve()),
        "summary_json": str(paths.summary_json.resolve()),
    }
    parameters = {
        "overlap_fraction": overlap_fraction,
        "channel": channel,
        "level": level,
        "rough_phase_level": rough_phase_level,
        "rough_phase_xy_downsample_factor": 2**rough_phase_level,
        "rough_phase_dimensions": "zyx" if z_slab_planes > 1 else "yx",
        "rough_phase_z_sampling": "native_center_z_slab",
        "rough_phase_z_slab_planes": z_slab_planes,
        "rough_phase_downsample_zyx": list(phase_downsample_zyx),
        "mode": "tltr_x_join_center_z_phase",
        "search_margin_px": search_margin_px,
        "phase_upsample_factor": phase_upsample_factor,
        "seam_fraction": seam_fraction,
        "registration_pair_mode": registration_pair_mode,
        "skip_registration_plots": skip_registration_plots,
        "do_fuse": do_fuse,
        "do_pyramid": do_pyramid,
    }
    commands.update(
        lr_dumb_stitch_alignment_commands(
            left_dir=left_dir,
            right_dir=right_dir,
            paths=lr_paths,
            overlap_fraction=overlap_fraction,
            channel=channel,
            level=rough_phase_level,
            search_margin_px=search_margin_px,
            phase_upsample_factor=phase_upsample_factor,
            seam_fraction=seam_fraction,
            z_slab_planes=z_slab_planes,
            phase_downsample_zyx=phase_downsample_zyx,
        )
    )
    run_lr_dumb_stitch_alignment(
        left_dir=left_dir,
        right_dir=right_dir,
        output_prefix=output_prefix,
        overlap_fraction=overlap_fraction,
        channel=channel,
        level=rough_phase_level,
        search_margin_px=search_margin_px,
        phase_upsample_factor=phase_upsample_factor,
        seam_fraction=seam_fraction,
        z_slab_planes=z_slab_planes,
        phase_downsample_zyx=phase_downsample_zyx,
        paths=lr_paths,
        dry_run=dry_run,
    )
    commands["registration"] = register_tiles(
        run_dir=paths.registration_dir,
        position_input=paths.phase_position,
        registration_output=paths.registration_json,
        level=level,
        registration_pair_mode=registration_pair_mode,
        registration_pair_file=registration_pair_file,
        robust_boundary_qc_dir=paths.robust_boundary_qc_dir,
        registration_plots_dir=paths.registration_plots_dir,
        skip_registration_plots=skip_registration_plots,
        dask_num_workers=dask_registration_workers,
        pairwise_jobs=pairwise_jobs,
        dry_run=dry_run,
    )
    commands["qc"] = render_registration_qc(
        position_input=paths.phase_position,
        registration_input=paths.registration_json,
        output_dir=paths.registration_qc_dir,
        channel=channel,
        level=level,
        center_y_xz=True,
        dry_run=dry_run,
    )
    if do_fuse:
        fused_outputs = channel_output_paths(paths.fusion_base_output, [channel])
        outputs["fusion_base_output"] = str(paths.fusion_base_output.resolve())
        outputs["fusion_outputs"] = [str(path.resolve()) for path in fused_outputs]
        commands["fusion"] = fuse_tiles(
            input_dir=paths.registration_dir,
            position_input=paths.phase_position,
            registration_input=paths.registration_json,
            output=paths.fusion_base_output,
            channels=[channel],
            dry_run=dry_run,
        )
        if do_pyramid:
            commands["pyramid"] = add_pyramids(ome_zarrs=fused_outputs, dry_run=dry_run)
    if not dry_run:
        write_workflow_summary(
            paths.summary_json,
            workflow="tltr_x_join_center_z_phase",
            inputs={"left_dir": str(left_dir.resolve()), "right_dir": str(right_dir.resolve())},
            outputs=outputs,
            parameters=parameters,
            commands=commands,
        )
    return paths
