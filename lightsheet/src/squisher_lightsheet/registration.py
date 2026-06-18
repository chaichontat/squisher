from __future__ import annotations

from pathlib import Path

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.legacy_runner import run_legacy_script


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
    level: int = 4,
    registration_pair_mode: str = "robust-boundary",
    robust_boundary_qc_dir: Path | None = None,
    registration_plots_dir: Path | None = None,
    skip_registration_plots: bool = True,
    dask_num_workers: int | None = None,
    pairwise_jobs: int | None = None,
    reference_registration_input: Path | None = None,
    reference_geometry_mode: str = "none",
    reference_xy_prior_weight: float | None = None,
    shared_geometry_tracks: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> str:
    if not dry_run and shared_geometry_tracks is None:
        ensure_single_track_position_input(position_input)

    args = [
        str(run_dir),
        "--position-input",
        str(position_input),
        "--register",
        "--register-only",
        "--registration-output",
        str(registration_output),
        "--registration-pair-mode",
        registration_pair_mode,
        "--reg-res-level",
        str(level),
    ]
    if skip_registration_plots:
        args.append("--skip-registration-plots")
    else:
        args.append("--no-skip-registration-plots")
    if registration_plots_dir is not None:
        args.extend(["--registration-plots-dir", str(registration_plots_dir)])
    if robust_boundary_qc_dir is not None:
        args.extend(["--robust-boundary-qc-dir", str(robust_boundary_qc_dir)])
    if dask_num_workers is not None:
        args.extend(["--dask-num-workers", str(dask_num_workers)])
    if pairwise_jobs is not None:
        args.extend(["--n-parallel-pairwise-regs", str(pairwise_jobs)])
    if reference_registration_input is not None:
        args.extend(["--reference-registration-input", str(reference_registration_input)])
    if reference_geometry_mode != "none":
        args.extend(["--reference-geometry-mode", reference_geometry_mode])
    if reference_xy_prior_weight is not None:
        args.extend(["--reference-xy-prior-weight", str(reference_xy_prior_weight)])
    if shared_geometry_tracks is not None:
        args.extend(["--shared-geometry-tracks", ",".join(shared_geometry_tracks)])
    if dry_run:
        args.append("--dry-run")
    return run_legacy_script("stitch_20x_tl_multiview.py", args, dry_run=dry_run)
