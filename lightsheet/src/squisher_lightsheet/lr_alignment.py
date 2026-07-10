from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from squisher_lightsheet.artifacts import write_workflow_summary
from squisher_lightsheet.legacy_runner import command_text
from squisher_lightsheet.positions import create_position_file
from squisher_lightsheet.rough_phase import rough_phase_align


DEFAULT_OVERLAP_FRACTION = 0.25
DEFAULT_LEVEL = 2
DEFAULT_Z_SLAB_PLANES = 8
DEFAULT_PHASE_DOWNSAMPLE_ZYX = (4, 4, 4)


@dataclass(frozen=True)
class LRDumbStitchAlignmentPaths:
    metadata_position: Path
    phase_position: Path
    phase_qc_dir: Path
    initial_overlay: Path
    corrected_overlay: Path
    phase_summary_json: Path
    summary_json: Path


def lr_phase_plane(*, z_slab_planes: int) -> str:
    return "zyx" if z_slab_planes > 1 else "xy"


def lr_dumb_stitch_alignment_paths(
    output_prefix: Path,
    *,
    level: int = DEFAULT_LEVEL,
    channel: int = 0,
    z_slab_planes: int = DEFAULT_Z_SLAB_PLANES,
    run_dir: Path | None = None,
) -> LRDumbStitchAlignmentPaths:
    metadata_position = output_prefix.with_name(f"{output_prefix.name}.positions.json")
    phase_position = output_prefix.with_name(f"{output_prefix.name}.roughPhase.positions.json")
    if run_dir is None:
        run_dir = output_prefix.with_name(f"{output_prefix.name}-roughPhase-registration-level{level}")
    phase_qc_dir = run_dir / f"level{level}-rough-phase"
    phase_plane = lr_phase_plane(z_slab_planes=z_slab_planes)
    return LRDumbStitchAlignmentPaths(
        metadata_position=metadata_position,
        phase_position=phase_position,
        phase_qc_dir=phase_qc_dir,
        initial_overlay=phase_qc_dir / f"level{level}_metadata_initial_{phase_plane}_yellowOverlay_ch{channel}.png",
        corrected_overlay=phase_qc_dir / f"level{level}_phase_corrected_{phase_plane}_yellowOverlay_ch{channel}.png",
        phase_summary_json=phase_qc_dir / f"level{level}_{phase_plane}_phase_alignment_ch{channel}.json",
        summary_json=run_dir / "lr_dumb_stitch_alignment_summary.json",
    )


def lr_dumb_stitch_alignment_commands(
    *,
    left_dir: Path,
    right_dir: Path,
    paths: LRDumbStitchAlignmentPaths,
    overlap_fraction: float,
    channel: int,
    level: int,
    search_margin_px: int,
    phase_upsample_factor: int,
    seam_fraction: float,
    z_slab_planes: int,
    phase_downsample_zyx: tuple[int, int, int],
) -> dict[str, str]:
    return {
        "position": command_text(
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
        ),
        "rough_phase": command_text(
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
                "--z-slab-planes",
                str(z_slab_planes),
                "--phase-downsample-zyx",
                ",".join(str(value) for value in phase_downsample_zyx),
            ]
        ),
    }


def run_lr_dumb_stitch_alignment(
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
    z_slab_planes: int = DEFAULT_Z_SLAB_PLANES,
    phase_downsample_zyx: tuple[int, int, int] = DEFAULT_PHASE_DOWNSAMPLE_ZYX,
    paths: LRDumbStitchAlignmentPaths | None = None,
    dry_run: bool = False,
) -> LRDumbStitchAlignmentPaths:
    if overlap_fraction < 0.0 or overlap_fraction >= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")
    if level < 0:
        raise ValueError("level must be non-negative")
    if search_margin_px < 0:
        raise ValueError("search_margin_px must be non-negative")
    if phase_upsample_factor < 1:
        raise ValueError("phase_upsample_factor must be >= 1")
    if not 0.0 < seam_fraction <= 1.0:
        raise ValueError("seam_fraction must be in (0, 1]")
    if z_slab_planes < 1:
        raise ValueError("z_slab_planes must be >= 1")
    if len(phase_downsample_zyx) != 3 or min(phase_downsample_zyx) < 1:
        raise ValueError("phase_downsample_zyx must contain three values >= 1")

    if paths is None:
        paths = lr_dumb_stitch_alignment_paths(
            output_prefix.resolve(),
            level=level,
            channel=channel,
            z_slab_planes=z_slab_planes,
        )
    commands = lr_dumb_stitch_alignment_commands(
        left_dir=left_dir,
        right_dir=right_dir,
        paths=paths,
        overlap_fraction=overlap_fraction,
        channel=channel,
        level=level,
        search_margin_px=search_margin_px,
        phase_upsample_factor=phase_upsample_factor,
        seam_fraction=seam_fraction,
        z_slab_planes=z_slab_planes,
        phase_downsample_zyx=phase_downsample_zyx,
    )
    outputs = {
        "metadata_position": str(paths.metadata_position.resolve()),
        "phase_position": str(paths.phase_position.resolve()),
        "phase_qc_dir": str(paths.phase_qc_dir.resolve()),
        "initial_overlay": str(paths.initial_overlay.resolve()),
        "corrected_overlay": str(paths.corrected_overlay.resolve()),
        "phase_summary_json": str(paths.phase_summary_json.resolve()),
        "summary_json": str(paths.summary_json.resolve()),
    }
    parameters = {
        "overlap_fraction": overlap_fraction,
        "channel": channel,
        "level": level,
        "xy_downsample_factor": 2**level,
        "phase_dimensions": "zyx" if z_slab_planes > 1 else "yx",
        "z_sampling": "native_center_z_slab",
        "z_slab_planes": z_slab_planes,
        "mode": "tltr_x_join_center_z_phase",
        "search_margin_px": search_margin_px,
        "phase_upsample_factor": phase_upsample_factor,
        "seam_fraction": seam_fraction,
        "phase_downsample_zyx": list(phase_downsample_zyx),
    }

    if dry_run:
        logger.info(paths)
        return paths

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
        z_slab_planes=z_slab_planes,
        phase_downsample_zyx=phase_downsample_zyx,
    )
    write_workflow_summary(
        paths.summary_json,
        workflow="lr_dumb_stitch_alignment",
        inputs={"left_dir": str(left_dir.resolve()), "right_dir": str(right_dir.resolve())},
        outputs=outputs,
        parameters=parameters,
        commands=commands,
    )
    return paths
