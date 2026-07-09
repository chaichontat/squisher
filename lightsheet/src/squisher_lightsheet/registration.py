from __future__ import annotations

from pathlib import Path

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.legacy_runner import run_legacy_script
from squisher_lightsheet import seams


TrackMetadata = legacy.TrackMetadata
TileMetadata = legacy.TileMetadata
RobustBoundarySettings = seams.RobustBoundarySettings
BoundaryPatchSpec = seams.BoundaryPatchSpec
BoundaryConstraint = seams.BoundaryConstraint
affine_translation_zyx = legacy.affine_translation_zyx
refined_phase_shift_from_samples = seams.refined_phase_shift_from_samples
combine_channel_boundary_constraints = legacy.combine_channel_boundary_constraints
solve_tile_corrections_zyx = legacy.solve_tile_corrections_zyx
solve_tile_corrections_with_residual_rejection = legacy.solve_tile_corrections_with_residual_rejection
solve_tile_corrections_with_multiview_stitcher = legacy.solve_tile_corrections_with_multiview_stitcher
axis_aligned_registration_pairs = legacy.axis_aligned_registration_pairs
replace_tile_stage_transform = legacy.replace_tile_stage_transform
apply_reference_fixed_axes = legacy.apply_reference_fixed_axes
reference_geometry_solver_options = legacy.reference_geometry_solver_options
reference_geometry_constraint = legacy.reference_geometry_constraint
refinement_start_params = legacy.refinement_start_params
align_params_to_reference = legacy.align_params_to_reference
align_refinement_start_to_reference = legacy.align_refinement_start_to_reference


def ensure_single_track_position_input(position_input: Path) -> None:
    tiles = legacy.read_position_input_tiles(position_input.resolve())
    if len(tiles[0].tracks) > 1:
        raise ValueError(
            "squisher-lightsheet v1 does not support multi-track registration outputs; "
            "run the underlying legacy stitcher directly or split the input by track first"
        )


def register_tiles(
    *,
    run_dir: Path,
    position_input: Path,
    registration_output: Path,
    level: int = 0,
    registration_pair_mode: str = legacy.DEFAULT_REGISTRATION_PAIR_MODE,
    robust_boundary_qc_dir: Path | None = None,
    registration_plots_dir: Path | None = None,
    skip_registration_plots: bool = True,
    dask_num_workers: int | None = None,
    pairwise_jobs: int | None = legacy.DEFAULT_N_PARALLEL_PAIRWISE_REGS,
    registration_cache_max_gib: float | None = legacy.DEFAULT_REGISTRATION_CACHE_MAX_GIB,
    registration_pair_file: Path | None = None,
    groupwise_transform: str = legacy.MVS_GROUPWISE_TRANSFORM,
    mvs_post_quality_filter: bool = True,
    mvs_post_quality_threshold: float | None = legacy.MVS_POST_QUALITY_THRESHOLD,
    reference_registration_input: Path | None = None,
    reference_geometry_mode: str = "none",
    reference_xy_prior_weight: float | None = None,
    reference_initial_alignment: str = "none",
    shared_geometry_tracks: tuple[str, ...] | None = None,
    channels: tuple[int, ...] | None = None,
    log_file: Path | None = None,
    dry_run: bool = False,
) -> str:
    if not dry_run and shared_geometry_tracks is None and channels is None:
        ensure_single_track_position_input(position_input)
    if robust_boundary_qc_dir is not None and robust_boundary_qc_dir != run_dir / "robust-boundary-qc":
        raise ValueError("The current legacy registration CLI writes robust-boundary QC to RUN_DIR/robust-boundary-qc")
    if registration_plots_dir is not None and registration_plots_dir != run_dir / "registration-plots":
        raise ValueError("The current legacy registration CLI writes registration plots to RUN_DIR/registration-plots")
    if not skip_registration_plots:
        raise ValueError("The current legacy registration CLI does not expose registration plot generation")
    args = [
        str(run_dir),
        "--position-input",
        str(position_input),
        "--register",
        "--register-only",
        "--registration-output",
        str(registration_output),
        "--coarse-reg-res-levels",
        str(level),
    ]
    if reference_registration_input is not None:
        args.extend(["--reference-registration-input", str(reference_registration_input)])
    if reference_geometry_mode != "none":
        args.extend(["--reference-geometry-mode", reference_geometry_mode])
    if reference_xy_prior_weight is not None:
        args.extend(["--reference-xy-prior-weight", str(reference_xy_prior_weight)])
    if reference_initial_alignment != "none":
        args.extend(["--reference-initial-alignment", reference_initial_alignment])
    if shared_geometry_tracks is not None:
        args.extend(["--shared-geometry-tracks", ",".join(shared_geometry_tracks)])
    if channels is not None:
        args.append("--channels")
        args.extend(str(channel) for channel in channels)
    if registration_pair_mode != legacy.DEFAULT_REGISTRATION_PAIR_MODE:
        args.extend(["--registration-pair-mode", registration_pair_mode])
    if registration_pair_file is not None:
        args.extend(["--registration-pair-file", str(registration_pair_file)])
    if pairwise_jobs is not None:
        args.extend(["--n-parallel-pairwise-regs", str(pairwise_jobs)])
    if dask_num_workers is not None:
        args.extend(["--dask-num-workers", str(dask_num_workers)])
    if registration_cache_max_gib != legacy.DEFAULT_REGISTRATION_CACHE_MAX_GIB:
        if registration_cache_max_gib is None:
            raise ValueError("registration_cache_max_gib cannot be None for the legacy registration CLI")
        args.extend(["--registration-cache-max-gib", str(registration_cache_max_gib)])
    if groupwise_transform != legacy.MVS_GROUPWISE_TRANSFORM:
        args.extend(["--groupwise-transform", groupwise_transform])
    if not mvs_post_quality_filter:
        args.append("--no-mvs-post-quality-filter")
    elif mvs_post_quality_threshold != legacy.MVS_POST_QUALITY_THRESHOLD:
        if mvs_post_quality_threshold is None:
            raise ValueError("mvs_post_quality_threshold cannot be None when the quality filter is enabled")
        args.extend(["--mvs-post-quality-threshold", str(mvs_post_quality_threshold)])
    if log_file is not None:
        args.extend(["--log-file", str(log_file)])
    if dry_run:
        args.append("--dry-run")
    return run_legacy_script("stitch_20x_tl_multiview.py", args, dry_run=dry_run)
