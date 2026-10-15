"""The canonical 561 frame (WORKFLOW_HCR.md steps 5-7): CL/CR placement and the joint 561 and 638 fusions that
every later channel is fused onto."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from squisher_lightsheet._legacy.stitch_20x_tl_multiview import (
    file_resume_identity,
    source_tile_resume_identity,
)
from squisher_lightsheet.edu_workflow import (
    DEFAULT_638_REFERENCE,
    DEFAULT_638_REFERENCE_SHA256,
)
from squisher_lightsheet.geometry import signed_bounds
from squisher_lightsheet.hcr_acquisition import ChannelSource, HcrPaths, require_complete_fusion
from squisher_lightsheet.ngff import scale_translation
from squisher_lightsheet.tltr_placement import (
    publish_tltr_placement,
    render_coarse_overlay,
    split_sides,
    write_joined_sides,
)
from squisher_lightsheet.workflow_runner import (
    StageRunner,
    atomic_json,
    gate_approval,
    gate_numbers,
    sha256_file,
)


def single_561_source(sources: list[ChannelSource], side: str) -> ChannelSource:
    matches = [source for source in sources if source.side == side and source.label == "561"]
    if len(matches) != 1:
        raise ValueError(f"Side {side} needs exactly one 561 source, found {len(matches)}")
    return matches[0]


def _link_tiles(directory: Path, sources: list[Path]) -> None:
    """A directory of symlinks to every deconvolved tile of the given acquisitions, for joint fusion."""
    directory.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for tile in sorted(source.glob("*.ome.zarr")):
            link = directory / tile.name
            if link.is_symlink() and link.resolve() == tile.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise ValueError(f"{link} already exists and does not point to {tile}")
            link.symlink_to(tile.resolve(), target_is_directory=True)


def place_clcr(
    runner: StageRunner,
    paths: HcrPaths,
    manifest: dict[str, Any],
    sources: list[ChannelSource],
    thresholds: dict[str, float],
    *,
    all_gpu_env: dict[str, str],
) -> dict[str, Path]:
    """WORKFLOW_HCR.md step 5 for CL/CR: coarse placement, Method8 spatial-overlap correction, published placement."""
    root = paths.root
    placement = root / "placement"
    fixed, moving = single_561_source(sources, "CL"), single_561_source(sources, "CR")
    raw = {acquisition["side"]: acquisition["directory"] for acquisition in manifest["acquisitions"]
           if any(channel["label"] == "561" for channel in acquisition["channels"])}
    metadata = placement / "metadata.positions.json"
    runner.command(
        "placement-metadata",
        [
            "lightsheet",
            "position",
            "--left-dir",
            raw["CL"],
            "--right-dir",
            raw["CR"],
            "--output",
            metadata,
            "--mode",
            "lr_z_endview_flip_xz",
        ],
        required=[metadata],
        input_identity=lambda: {
            side: [
                source_tile_resume_identity(path, require_completion=False)
                for path in sorted(Path(directory).glob("*.ome.tif"))
            ]
            for side, directory in raw.items()
        },
    )
    coarse_request = root / "placement-coarse.json"
    if not coarse_request.exists():
        atomic_json(
            coarse_request,
            {"artifact_type": "squisher_lightsheet.hcr_coarse_placement.v1", "manual_cr_shift_zyx_um": None},
        )
    manual = json.loads(coarse_request.read_text())["manual_cr_shift_zyx_um"]
    # Each manual shift aligns into its own folder, so the author can refine it before accepting one.
    attempt = "coarse" if manual is None else "coarse-manual_" + "_".join(f"{float(value):+.3f}" for value in manual)
    coarse = placement / attempt
    align: list[str | Path] = [
        "lightsheet",
        "align-registered-lr",
        "--metadata-position",
        metadata,
        "--left-position",
        paths.registered_positions(fixed.acquisition, "561"),
        "--right-position",
        paths.registered_positions(moving.acquisition, "561"),
        "--left-fusion",
        paths.side_fused("CL", "561", fixed.index),
        "--right-fusion",
        paths.side_fused("CR", "561", moving.index),
        "--output-dir",
        coarse,
        "--fusion-level",
        "2",
    ]
    if manual is not None:
        align.append("--manual-cr-shift-zyx-um=" + ",".join(str(float(value)) for value in manual))
    runner.command(
        f"placement-align-{attempt}",
        align,
        required=[coarse / "phase.json", coarse / "manifest.json", coarse / "cl.positions.json", coarse / "cr.positions.json"],
        input_identity=lambda: {
            "metadata": file_resume_identity(metadata),
            "positions": [
                file_resume_identity(paths.registered_positions(source.acquisition, "561")) for source in (fixed, moving)
            ],
            "fusions": [
                source_tile_resume_identity(paths.side_fused(source.side, "561", source.index), require_completion=False)
                for source in (fixed, moving)
            ],
        },
    )
    gate_approval(
        root / "placement-coarse-approval.json",
        {"phase": sha256_file(coarse / "phase.json")},
        message=(
            f"Review {coarse / 'phase-overlay.png'} and phase.json (phase_candidate_accepted); approve, or set "
            f"manual_cr_shift_zyx_um in {coarse_request} and delete this approval file"
        ),
        artifact_type="squisher_lightsheet.hcr_approval.v1",
    )
    method8, joined = _method8_and_link(
        runner,
        paths,
        fixed,
        moving,
        fixed_position=coarse / "cl.positions.json",
        moving_position=coarse / "cr.positions.json",
        threshold=thresholds["CL/561"],
        all_gpu_env=all_gpu_env,
    )
    final = placement / "final"
    runner.command(
        "placement-publish",
        [
            "lightsheet",
            "clcr-placement",
            "--left-position",
            coarse / "cl.positions.json",
            "--right-position",
            coarse / "cr.positions.json",
            "--alignment-manifest",
            coarse / "manifest.json",
            "--phasecorr",
            coarse / "phase.json",
            "--method8-summary",
            method8 / "tile_quadrant_method8_summary.json",
            "--deconv-root",
            joined,
            "--output-dir",
            final,
            "--qc-channel",
            str(fixed.index),
            "--qc-level",
            "2",
        ],
        required=[final / "positions.json", final / "registration.json"],
        input_identity=lambda: {
            "coarse": {name: file_resume_identity(coarse / name) for name in ("cl.positions.json", "cr.positions.json", "manifest.json", "phase.json")},
            "method8": file_resume_identity(method8 / "tile_quadrant_method8_summary.json"),
        },
    )
    gate_approval(
        root / "placement-approval.json",
        {"positions": sha256_file(final / "positions.json")},
        message=f"Review the CL/CR placement overlay in {final / 'qc-zmedian'} and the Method8 summary, then set approved=true",
        artifact_type="squisher_lightsheet.hcr_approval.v1",
    )
    return {"positions": final / "positions.json", "registration": final / "registration.json", "deconv": joined}


def _method8_and_link(
    runner: StageRunner,
    paths: HcrPaths,
    fixed: ChannelSource,
    moving: ChannelSource,
    *,
    fixed_position: Path,
    moving_position: Path,
    threshold: float,
    all_gpu_env: dict[str, str],
) -> tuple[Path, Path]:
    """Method8 spatial-overlap windows between the coarsely placed sides, and a joined view of both sides' tiles."""
    placement = paths.root / "placement"
    method8 = placement / "method8"
    runner.command(
        "placement-method8",
        [
            "lightsheet",
            "cross-register-method8",
            "method8",
            "--fixed-position",
            fixed_position,
            "--coarse-moving-position",
            moving_position,
            "--output-dir",
            method8,
            "--fixed-channel",
            str(fixed.index),
            "--moving-channel",
            str(moving.index),
            "--pair-mode",
            "spatial-overlap",
            "--preseed-matrix-zyx",
            "1,0,0,0,1,0,0,0,1",
            "--core-shape-zyx",
            "480,480,480",
            "--window-shape-zyx",
            "528,528,528",
            "--fit-downsample-zyx",
            "1,1,1",
            "--fixed-mask-threshold",
            str(threshold),
            "--fixed-mask-level",
            "2",
            "--workers",
            "2",
            "--devices",
            "0,1",
            "--resume",
        ],
        required=[method8 / "tile_quadrant_method8_summary.json"],
        env=all_gpu_env,
        input_identity=lambda: {path.name: file_resume_identity(path) for path in (fixed_position, moving_position)},
    )
    joined = placement / "deconv-561"
    acquisitions = [paths.deconv(fixed.acquisition), paths.deconv(moving.acquisition)]
    runner.action(
        "placement-joined-561",
        {"acquisitions": [str(path) for path in acquisitions]},
        lambda: _link_tiles(joined, acquisitions),
        required=[joined],
    )
    return method8 / "tile_quadrant_method8_summary.json", joined


def place_tltr(
    runner: StageRunner,
    paths: HcrPaths,
    manifest: dict[str, Any],
    sources: list[ChannelSource],
    thresholds: dict[str, float],
    *,
    all_gpu_env: dict[str, str],
) -> dict[str, Path]:
    """WORKFLOW_HCR.md step 5 for TL/TR: centre-Z slab phase (or a manual shift), Method8, published placement."""
    root = paths.root
    placement = root / "placement"
    fixed, moving = single_561_source(sources, "TL"), single_561_source(sources, "TR")
    raw = {
        acquisition["side"]: acquisition["directory"]
        for acquisition in manifest["acquisitions"]
        if any(channel["label"] == "561" for channel in acquisition["channels"])
    }
    metadata = placement / "metadata.positions.json"
    runner.command(
        "placement-metadata",
        [
            "lightsheet",
            "position",
            "--left-dir",
            raw["TL"],
            "--right-dir",
            raw["TR"],
            "--output",
            metadata,
            "--mode",
            "tltr_x_join_center_z_phase",
        ],
        required=[metadata],
        input_identity=lambda: {
            side: [
                source_tile_resume_identity(path, require_completion=False)
                for path in sorted(Path(directory).glob("*.ome.tif"))
            ]
            for side, directory in raw.items()
        },
    )
    coarse_request = root / "placement-coarse.json"
    if not coarse_request.exists():
        atomic_json(
            coarse_request,
            {"artifact_type": "squisher_lightsheet.hcr_coarse_placement.v1", "manual_tr_shift_zyx_um": None},
        )
    manual = json.loads(coarse_request.read_text())["manual_tr_shift_zyx_um"]
    # Each manual shift places TR into its own folder, so the author can refine it before accepting one.
    attempt = "coarse" if manual is None else "coarse-manual_" + "_".join(f"{float(value):+.3f}" for value in manual)
    directory = placement / attempt
    joined_sides = directory / "joined.positions.json"
    registered = {source.side: paths.registered_positions(source.acquisition, "561") for source in (fixed, moving)}
    runner.action(
        f"placement-join-{attempt}",
        {
            "metadata": sha256_file(metadata),
            "registered": {side: sha256_file(path) for side, path in registered.items()},
            "manual": manual,
        },
        lambda: write_joined_sides(
            metadata=metadata,
            left_position=registered["TL"],
            right_position=registered["TR"],
            right_shift_zyx_um=np.zeros(3) if manual is None else np.asarray(manual, dtype=np.float64),
            output=joined_sides,
        ),
        required=[joined_sides],
    )
    coarse = directory / "coarse.positions.json"
    if manual is None:
        runner.command(
            f"placement-phase-{attempt}",
            [
                "lightsheet",
                "rough-phase",
                "--position-input",
                joined_sides,
                "--output-position",
                coarse,
                "--output-dir",
                directory / "phase",
                "--channel",
                str(fixed.index),
                "--level",
                "2",
            ],
            required=[coarse],
            input_identity=lambda: file_resume_identity(joined_sides),
        )
    else:
        coarse = joined_sides
        # A manual shift has no phase overlays; render the shifted placement itself for review.
        runner.action(
            f"placement-overlay-{attempt}",
            {"coarse": sha256_file(coarse)},
            lambda: render_coarse_overlay(coarse, directory / "phase", channel=fixed.index, level=2),
            required=[directory / "phase" / "zmedian_overlay.json"],
        )
    gate_approval(
        root / "placement-coarse-approval.json",
        {"coarse": sha256_file(coarse)},
        message=(
            f"Review the TL/TR coarse placement overlays in {directory / 'phase'} and the shift recorded in "
            f"{coarse.name}; approve, or set manual_tr_shift_zyx_um in {coarse_request} and delete this approval file"
        ),
        artifact_type="squisher_lightsheet.hcr_approval.v1",
    )
    sides = directory / "sides"
    runner.action(
        f"placement-split-{attempt}",
        {"coarse": sha256_file(coarse)},
        lambda: split_sides(coarse, sides),
        required=[sides / "tl.positions.json", sides / "tr.positions.json"],
    )
    method8, joined = _method8_and_link(
        runner,
        paths,
        fixed,
        moving,
        fixed_position=sides / "tl.positions.json",
        moving_position=sides / "tr.positions.json",
        threshold=thresholds["TL/561"],
        all_gpu_env=all_gpu_env,
    )
    final = placement / "final"
    decision = json.loads(coarse_request.read_text()).get("reviewed_global_affine")
    reviewed_affine = None
    affine_identity = {}
    if decision is not None:
        reviewed_affine = Path(decision["path"])
        if sha256_file(reviewed_affine) != decision["sha256"]:
            raise ValueError("Reviewed TL/TR global affine has changed since selection")
        affine_identity = {"reviewed_global_affine": decision}
    runner.action(
        "placement-publish",
        {"coarse": sha256_file(coarse), "method8": sha256_file(method8), **affine_identity},
        lambda: publish_tltr_placement(
            coarse=coarse, method8_summary=method8, deconv_root=joined, output_dir=final, qc_channel=fixed.index, qc_level=2,
            **({"reviewed_global_affine": reviewed_affine} if reviewed_affine is not None else {}),
        ),
        required=[final / "positions.json", final / "registration.json"],
    )
    gate_approval(
        root / "placement-approval.json",
        {"positions": sha256_file(final / "positions.json"),
         **({"registration": sha256_file(final / "registration.json")} if reviewed_affine is not None else {})},
        message=f"Review the TL/TR placement overlay in {final / ('qc-affine' if reviewed_affine is not None else 'qc-zmedian')} and the Method8 summary, then set approved=true",
        artifact_type="squisher_lightsheet.hcr_approval.v1",
    )
    return {"positions": final / "positions.json", "registration": final / "registration.json", "deconv": joined}


def shared_slab_percentiles(
    grid: Path,
    positions: Path,
    *,
    side_field: str = "source_side",
    sides: tuple[str, str],
    margin_um: float = 10.0,
    planes: int = 16,
) -> list[float]:
    """Z percentiles of the combined grid that sample only the slab both sides cover, ``margin_um`` inside it.

    Placed tiles name their side in ``source_side`` (CL/CR); joined materialized windows in ``source_view`` (L/R).
    """
    group = zarr.open_group(str(grid), mode="r")
    _names, scale, origin, _has_scale, _has_origin = scale_translation(group, dataset_index=0)
    grid_planes = group["0"].shape[0]
    placement = json.loads(positions.read_text())
    tiles = placement["tiles"]
    affine = placement.get("alignment", {}).get("global_affine_zyx_um")

    def z_range(side: str) -> tuple[float, float]:
        bounds = [
            signed_bounds(
                np.asarray([tile["translation_um"][axis] for axis in "zyx"]),
                np.asarray([tile["scale_um"][axis] for axis in "zyx"]),
                np.asarray(tile["shape"][-3:]),
            )
            for tile in tiles
            if tile[side_field] == side
        ]
        if affine is not None and side == "TR":
            matrix = np.asarray(affine)
            from itertools import product
            corners = np.array([point for low, high in bounds for point in product(*zip(low, high, strict=True))])
            transformed = corners @ matrix[:3, :3].T + matrix[:3, 3]
            return float(transformed[:, 0].min()), float(transformed[:, 0].max())
        return min(float(low[0]) for low, _high in bounds), max(float(high[0]) for _low, high in bounds)

    # Tiles may start outside the grid; sample only planes the grid holds.
    ranges = [z_range(side) for side in sides]
    low = max(*(start for start, _stop in ranges), origin[0])
    high = min(*(stop for _start, stop in ranges), origin[0] + scale[0] * (grid_planes - 1))
    if high - low <= 2 * margin_um:
        raise ValueError(f"{sides[0]} and {sides[1]} share no Z slab thicker than {2 * margin_um} um in {grid}")
    indices = np.unique(np.rint((np.linspace(low + margin_um, high - margin_um, planes) - origin[0]) / scale[0]))
    return [(int(index) + 0.5) / (grid_planes - 1) * 100 for index in indices]


def require_held_out_improvement(match: Path) -> None:
    held_out = json.loads(match.read_text())["summary"]["spatial_cross_validation"]
    if not held_out["after_median_abs_log2"] < held_out["before_median_abs_log2"]:
        raise ValueError(
            f"The registered CR field does not improve held-out CL/CR agreement "
            f"({held_out['before_median_abs_log2']} -> {held_out['after_median_abs_log2']}): {match}"
        )


def reviewed_native_561_corrections(
    runner: StageRunner, *, canonical: Path, registration: Path,
    corrections: dict[str, Path], channel: int,
) -> Path | None:
    """Retain native 561 corrections after explicit review of a rejected cross-side intensity fit."""
    approval = canonical / "intensity-561-native-only-approval.json"
    if not approval.exists():
        return None
    match = canonical / "match-561"
    identity = {
        "registration": sha256_file(registration),
        "rejected_match": sha256_file(match / "match.json"),
        "rejected_field": sha256_file(match / "registered-view-correction.json"),
        **{f"native_{view}": sha256_file(path) for view, path in corrections.items()},
    }
    gate_approval(
        approval, identity,
        message="Review the rejected 561 intensity fit and approve retaining native corrections without extra scaling",
        artifact_type="squisher_lightsheet.hcr_native_only_561_approval.v1",
    )
    output = canonical / "native-corrections-561" / "correction.json"

    def compose() -> None:
        from squisher_lightsheet.cross_dataset_match import compose_base_corrections
        payload = compose_base_corrections(
            records=json.loads(registration.read_text())["tiles"],
            correction_payloads={view: json.loads(path.read_text()) for view, path in corrections.items()},
            channel=channel,
        )
        payload["fit"]["reviewed_native_only"] = {"approval": str(approval), "identity": identity}
        atomic_json(output, payload)

    runner.action(
        "canonical-native-corrections-561", {"inputs": identity, "approval": sha256_file(approval), "channel": channel},
        compose, required=[output],
    )
    return output


def fuse_canonical(
    runner: StageRunner,
    paths: HcrPaths,
    sources: list[ChannelSource],
    placement: dict[str, Path],
    *,
    sides: tuple[str, str],
    all_gpu_env: dict[str, str],
    primary_gpu_env: dict[str, str],
) -> dict[str, Path]:
    """WORKFLOW_HCR.md steps 6-7 for the 561 track: cross-dataset intensity match, then the seam fusion on the 561
    grid. ``sides`` are the layout's fixed and moving sides (CL/CR or TL/TR)."""
    canonical = paths.root / "canonical"
    joined, positions = placement["deconv"], placement["positions"]
    labels = ["561"] + (["638"] if any(source.label == "638" for source in sources) else [])
    by_label: dict[str, dict[str, ChannelSource]] = {
        label: {source.side: source for source in sources if source.label == label} for label in labels
    }
    registrations = {"561": placement["registration"]}
    if "638" in labels:
        registrations["638"] = placement["registration"].with_name("registration-638.json")
        runner.command(
            "canonical-638-affine",
            [
                "lightsheet",
                "apply-channel-affine",
                "--reference-registration",
                placement["registration"],
                "--calibration-registration",
                DEFAULT_638_REFERENCE,
                "--expected-calibration-sha256",
                DEFAULT_638_REFERENCE_SHA256,
                "--output-registration",
                registrations["638"],
                "--expected-moving-channel",
                str(by_label["638"][sides[0]].index),
                "--source-label",
                "638",
                "--target-label",
                "561",
            ],
            required=[registrations["638"]],
            input_identity=lambda: file_resume_identity(placement["registration"]),
        )
    index_561 = by_label["561"][sides[0]].index
    preview = canonical / "preview-561" / f"fused.ch{index_561}.ome.zarr"

    def tiles_identity() -> list[dict[str, Any]]:
        return [source_tile_resume_identity(path, require_completion=True) for path in sorted(joined.glob("*.ome.zarr"))]

    runner.command(
        "canonical-preview-561",
        [
            "lightsheet",
            "fuse",
            joined,
            "--position-input",
            positions,
            "--registration-input",
            placement["registration"],
            "--output",
            preview.parent,
            "--channel",
            str(index_561),
            "--fusion-level",
            "2",
            "--fusion-weight-mode",
            "content-preibisch-coarse",
            "--output-codec",
            "zstd",
            "--fusion-gpu-memory-max-gib",
            "8",
        ],
        required=[preview / "zarr.json"],
        env=primary_gpu_env,
        validate=lambda: require_complete_fusion(preview),
        input_identity=lambda: {"tiles": tiles_identity(), "positions": file_resume_identity(positions),
                                "registration": file_resume_identity(placement["registration"])},
    )
    fused: dict[str, Path] = {}
    for label in labels:
        index = by_label[label][sides[0]].index
        match = canonical / f"match-{label}"
        corrections = {
            view: paths.residual(by_label[label][side].acquisition, label) / "correction.json"
            for view, side in zip(("L", "R"), sides, strict=True)
        }
        native_correction = (
            reviewed_native_561_corrections(runner, canonical=canonical, registration=registrations[label],
                                           corrections=corrections, channel=index)
            if label == "561" else None
        )
        if native_correction is None:
            percentiles = [] if runner.dry_run else shared_slab_percentiles(preview, positions, sides=sides)
            runner.command(
                f"canonical-match-{label}",
                [
                    "lightsheet",
                    "cross-dataset-match",
                    "--fixed-fused",
                    preview,
                    "--registration",
                    registrations[label],
                    "--output-dir",
                    match,
                    *(
                        argument
                        for view, path in corrections.items()
                        for argument in ("--correction-by-source-view", f"{view}={path}")
                    ),
                    "--reference-view",
                    "L",
                    "--moving-view",
                    "R",
                    "--channel",
                    str(index),
                    "--source-level",
                    "2",
                    "--stride",
                    "4",
                    *(argument for value in percentiles for argument in ("--z-percentile", f"{value:.6f}")),
                    "--workers",
                    "8",
                ],
                required=[match / "match.json", match / "correction.json", match / "registered-view-correction.json"],
                validate=lambda match=match: require_held_out_improvement(match / "match.json"),
                input_identity=lambda label=label, corrections=corrections: {
                    "preview": source_tile_resume_identity(preview, require_completion=False),
                    "registration": file_resume_identity(registrations[label]),
                    "corrections": {view: file_resume_identity(path) for view, path in corrections.items()},
                },
            )
        correction = native_correction or match / "correction.json"
        registered_field = (
            [] if native_correction is not None else ["--registered-view-correction", match / "registered-view-correction.json"]
        )
        match_artifacts = (
            {"correction.json": correction, "native_only_approval": canonical / "intensity-561-native-only-approval.json"}
            if native_correction is not None else
            {name: match / name for name in ("correction.json", "registered-view-correction.json")}
        )
        output = canonical / f"fusion-{label}" / f"fused.ch{index}.ome.zarr"
        # 638 lands on the completed 561 grid so the two channels share voxels.
        template = (
            [] if label == "561" else ["--output-grid-template", fused["561"], "--output-grid-template-level", "0"]
        )
        runner.command(
            f"canonical-fuse-{label}",
            [
                "lightsheet",
                "fuse",
                joined,
                "--position-input",
                positions,
                "--registration-input",
                registrations[label],
                "--residual-correction",
                correction,
                *registered_field,
                "--output",
                output.parent,
                "--channel",
                str(index),
                "--fusion-level",
                "0",
                "--fusion-weight-mode",
                "crop-sharpness-seam",
                "--seam-width-um",
                "10",
                "--batch-size",
                "auto",
                "--source-cache-max-gib",
                "24",
                "--fusion-gpu-memory-max-gib",
                "8",
                "--output-chunksize-zyx",
                "24,960,960",
                "--output-codec",
                "jpegxr",
                "--jpegxr-level",
                "0.7",
                *template,
                "--resume-fusion",
            ],
            required=[output / "zarr.json"],
            env=all_gpu_env,
            validate=lambda output=output: require_complete_fusion(output),
            input_identity=lambda label=label, match_artifacts=match_artifacts: {
                "tiles": tiles_identity(),
                "positions": file_resume_identity(positions),
                "registration": file_resume_identity(registrations[label]),
                "match": {
                    name: file_resume_identity(path) for name, path in match_artifacts.items()
                },
            },
        )
        fused[label] = output
    review = canonical / "fusion-561-threshold-review.level2.ome.tif"
    runner.command(
        "canonical-threshold-review-561",
        ["lightsheet", "threshold-review", "--fixed-fused", fused["561"], "--output", review, "--level", "2"],
        required=[review],
        input_identity=lambda: source_tile_resume_identity(fused["561"], require_completion=False),
    )
    return fused | {"threshold_review": review, "preview": preview}



def fused_threshold(
    root: Path, label: str, fused: Path, review: Path, *, accepted_native: dict[str, float] | None = None
) -> float:
    """The fixed-mask threshold of a completed fused volume that a later label registers to.

    Fused 561 reuses one accepted native 561 threshold unchanged (WORKFLOW_HCR.md step 6); fused 514 and 488 take a
    threshold read from their own review TIFF.
    """
    path = root / f"fused-{label}-threshold.json"
    value = gate_numbers(
        path,
        [label],
        message=f"Read {review} and fill the fixed-mask threshold of fused {label}"
        + ("" if accepted_native is None else f", one of the accepted native values {accepted_native}"),
        artifact_type="squisher_lightsheet.hcr_thresholds.v1",
        identity={
            "fusion": source_tile_resume_identity(fused, require_completion=True),
            "review": sha256_file(review),
        },
    )[label]
    if accepted_native is not None and value not in accepted_native.values():
        raise ValueError(f"Fused 561 threshold {value} in {path} is not an accepted native 561 threshold {accepted_native}")
    return value

