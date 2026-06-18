from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from squisher_lightsheet.artifacts import write_workflow_summary
from squisher_lightsheet.fusion import channel_output_paths, fuse_tiles
from squisher_lightsheet.legacy_runner import command_text
from squisher_lightsheet.positions import create_position_file
from squisher_lightsheet.pyramid import add_pyramids
from squisher_lightsheet.qc import render_registration_qc
from squisher_lightsheet.registration import register_tiles
from squisher_lightsheet.rough_phase import rough_phase_align


DEFAULT_OVERLAP_FRACTION = 0.25
DEFAULT_LEVEL = 4


@dataclass(frozen=True)
class WorkflowPaths:
    metadata_position: Path
    phase_position: Path
    phase_qc_dir: Path
    registration_dir: Path
    registration_json: Path
    registration_plots_dir: Path
    robust_boundary_qc_dir: Path
    registration_qc_dir: Path
    summary_json: Path
    fusion_base_output: Path


def overlap_tag(overlap_fraction: float) -> str:
    percent = overlap_fraction * 100.0
    if percent.is_integer():
        return f"overlap{int(percent)}"
    return f"overlap{str(percent).replace('.', 'p')}"


def workflow_paths(output_prefix: Path, *, overlap_fraction: float, level: int = DEFAULT_LEVEL) -> WorkflowPaths:
    tag = overlap_tag(overlap_fraction)
    metadata_position = output_prefix.with_name(f"{output_prefix.name}.{tag}.positions.json")
    phase_position = output_prefix.with_name(f"{output_prefix.name}.{tag}.roughPhase.positions.json")
    run_dir = output_prefix.with_name(f"{output_prefix.name}-{tag}-roughPhase-registration-level{level}")
    return WorkflowPaths(
        metadata_position=metadata_position,
        phase_position=phase_position,
        phase_qc_dir=run_dir / f"level{level}-rough-phase",
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
    search_margin_px: int = 64,
    phase_upsample_factor: int = 10,
    seam_fraction: float = 0.10,
    registration_pair_mode: str = "robust-boundary",
    skip_registration_plots: bool = True,
    dask_registration_workers: int | None = None,
    pairwise_jobs: int | None = None,
    do_fuse: bool = False,
    do_pyramid: bool = False,
    dry_run: bool = False,
) -> WorkflowPaths:
    if do_pyramid and not do_fuse:
        raise ValueError("do_pyramid requires do_fuse")

    paths = workflow_paths(output_prefix.resolve(), overlap_fraction=overlap_fraction, level=level)
    commands: dict[str, str] = {}
    outputs = {
        "metadata_position": str(paths.metadata_position.resolve()),
        "phase_position": str(paths.phase_position.resolve()),
        "phase_qc_dir": str(paths.phase_qc_dir.resolve()),
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
        "mode": "tltr_x_join_center_z_phase",
        "search_margin_px": search_margin_px,
        "phase_upsample_factor": phase_upsample_factor,
        "seam_fraction": seam_fraction,
        "registration_pair_mode": registration_pair_mode,
        "skip_registration_plots": skip_registration_plots,
        "do_fuse": do_fuse,
        "do_pyramid": do_pyramid,
    }
    commands["position"] = command_text(
        [
            "lightsheet",
            "position",
            "--left-dir",
            str(left_dir.resolve()),
            "--right-dir",
            str(right_dir.resolve()),
            "--output",
            str(paths.metadata_position.resolve()),
            "--mode",
            "tltr_x_join_center_z_phase",
            "--overlap-fraction",
            str(overlap_fraction),
        ]
    )
    commands["rough_phase"] = command_text(
        [
            "lightsheet",
            "rough-phase",
            "--position-input",
            str(paths.metadata_position.resolve()),
            "--output-position",
            str(paths.phase_position.resolve()),
            "--output-dir",
            str(paths.phase_qc_dir.resolve()),
            "--channel",
            str(channel),
            "--level",
            str(level),
            "--search-margin-px",
            str(search_margin_px),
            "--upsample-factor",
            str(phase_upsample_factor),
            "--seam-fraction",
            str(seam_fraction),
        ]
    )
    if dry_run:
        print(paths, flush=True)
    else:
        create_position_file(
            left_dir=left_dir,
            right_dir=right_dir,
            output=paths.metadata_position,
            mode="tltr_x_join_center_z_phase",
            overlap_fraction=overlap_fraction,
            plot_title=f"{output_prefix.name} metadata positions",
        )
        rough_phase_align(
            position_input=paths.metadata_position,
            output_position=paths.phase_position,
            output_dir=paths.phase_qc_dir,
            channel=channel,
            level=level,
            search_margin_px=search_margin_px,
            upsample_factor=phase_upsample_factor,
            seam_fraction=seam_fraction,
        )
    commands["registration"] = register_tiles(
        run_dir=paths.registration_dir,
        position_input=paths.phase_position,
        registration_output=paths.registration_json,
        level=level,
        registration_pair_mode=registration_pair_mode,
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
