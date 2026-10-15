"""Cross-registered HCR channels (WORKFLOW_HCR.md steps 8-9): each label registered to its
fixed fused volume on both sides, recovered in rounds, screened, materialized, matched, and fused on the
canonical grid."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet._legacy.stitch_20x_tl_multiview import (
    file_resume_identity,
    source_tile_resume_identity,
)
from squisher_lightsheet.acquisition_offset import run_acquisition_offset, shifted_acquisition_positions
from squisher_lightsheet.artifact_io import registration_input_fingerprint
from squisher_lightsheet.edu_workflow import STARTING_MATRIX_514
from squisher_lightsheet.fused_fixed_recovery import (
    MANUAL_INITIALIZATIONS_ARTIFACT,
    propose_manual_initializations,
)
from squisher_lightsheet.hcr_acquisition import ChannelSource, HcrPaths, require_complete_fusion
from squisher_lightsheet.hcr_manifest import LAYOUT_SIDES
from squisher_lightsheet.hcr_reference import (
    require_held_out_improvement,
    shared_slab_percentiles,
    single_561_source,
)
from squisher_lightsheet.registration_screen import ARTIFACT_TYPE as SCREEN_ARTIFACT_TYPE
from squisher_lightsheet.registration_screen import screen_recovery, seedable_window_count
from squisher_lightsheet.tltr_placement import split_sides, write_joined_sides
from squisher_lightsheet.workflow_runner import (
    StageRunner,
    atomic_json,
    gate_approval,
    gate_choice,
    json_sha256,
    sha256_file,
)


@dataclass(frozen=True)
class CrossRegistration:
    """How one label registers to its fixed fused volume (WORKFLOW_HCR.md steps 8-9)."""

    starting_matrix: str
    rough_intensity_transform: str
    fit_intensity_transform: str
    minimum_gradient_ncc: float


IDENTITY_MATRIX = "1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0"
CROSS_REGISTRATION = {
    "514": CrossRegistration(STARTING_MATRIX_514, "log1p", "log1p", 0.24),
    "488": CrossRegistration(IDENTITY_MATRIX, "log1p", "log1p", 0.24),
}
# The EDU initializers live in acquisition axes; lr_z_endview_flip_xz flips CR's Z and X in the combined frame.
SIDE_AXIS_SIGNS = {"CL": (1.0, 1.0, 1.0), "CR": (-1.0, 1.0, -1.0), "TL": (1.0, 1.0, 1.0), "TR": (1.0, 1.0, 1.0)}
LAYOUT_MODES = {"CL/CR": "lr_z_endview_flip_xz", "TL/TR": "tltr_x_join_center_z_phase"}
# Process recovery for commands that resume finished windows: 230Tnc recover-native died silently four times on CR.
PROCESS_ATTEMPTS = 20


# 514 registers to fused 561 and 488 to fused 514.
FIXED_LABELS = {"514": "561", "488": "514"}


def side_starting_matrix(matrix: str, side: str) -> str:
    """Conjugate an acquisition-axes initializer by the side's combined-frame axis signs."""
    signs = np.asarray(SIDE_AXIS_SIGNS[side])
    values = np.asarray([float(value) for value in matrix.split(",")]).reshape(3, 3) * np.outer(signs, signs)
    return ",".join(repr(float(value)) for value in values.ravel())


def _recovery_inputs(summary: Path) -> object:
    payload = json.loads(summary.read_text())
    return {
        "summary": file_resume_identity(summary),
        "windows": [file_resume_identity(Path(window["level0_json"])) for window in payload["windows"]],
    }


def _seed_entries(master: Path, channel: str) -> dict[str, Any]:
    if not master.is_file():
        return {}
    return json.loads(master.read_text())["channels"].get(channel, {})


def _write_seeds(path: Path, channel: str, entries: dict[str, Any]) -> None:
    atomic_json(path, {"artifact_type": MANUAL_INITIALIZATIONS_ARTIFACT, "channels": {channel: entries}})


def _snapshot_native(cache: Path, snapshot: Path) -> None:
    """Copy one round's native recovery out of the shared resume cache, pointing its records at the copy."""
    temporary = snapshot.with_name(f".{snapshot.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(cache, temporary)
    prefix, replacement = json.dumps(str(cache.resolve()))[1:-1], json.dumps(str(snapshot.resolve()))[1:-1]
    for path in temporary.rglob("*.json"):
        path.write_text(path.read_text().replace(prefix, replacement))
    temporary.rename(snapshot)


def _recover_in_rounds(
    runner: StageRunner,
    paths: HcrPaths,
    source: ChannelSource,
    method6_summary: Path,
    *,
    primary_gpu_env: dict[str, str],
) -> Path:
    """Native and complete recovery, repeated while measured seeds keep reducing the rejected tissue windows.

    Each round snapshots its seeds from the dataset master file and recovers with them; ``propose-seeds`` then writes
    the next proposals into the master. A new round runs when the master differs from the round's snapshot, unless the
    proposals came from an automatic round that did not reduce the seedable windows. An author's edit to the master
    always starts a new round.

    recover-native resumes in one cache directory and reruns only windows whose initializer changed; it reads every
    round's seeds from one path because the path enters each initializer's provenance and so its resume hash. Each
    round's native output is then copied out of the cache, so completed rounds stay immutable inputs of their
    complete-recovery and screening stages.
    """
    channel = f"{source.label}-{source.side}"
    side_dir = paths.channel(source.label) / source.side
    cache = side_dir / "native-recovery-cache"
    current_seeds = side_dir / "seeds" / "current.json"
    master = paths.manual_initializations
    adjacency = paths.registration_dir(source.acquisition, source.label) / "registration.measurements.json"
    previous_count: int | None = None
    round_index = 0
    while True:
        seeds = side_dir / "seeds" / f"round-{round_index}.json"
        if not seeds.exists():
            _write_seeds(seeds, channel, _seed_entries(master, channel))
        snapshot = json.loads(seeds.read_text())["channels"][channel]
        _write_seeds(current_seeds, channel, snapshot)
        manual = ["--manual-initializations", current_seeds, "--manual-initializations-channel", channel]
        runner.command(
            f"{channel}-recover-native-round{round_index}",
            [
                "lightsheet",
                "cross-register",
                "recover-native",
                "--summary",
                method6_summary,
                "--output-dir",
                cache,
                "--adjacency-json",
                adjacency,
                "--devices",
                "0",
                "--resume",
                *(manual if snapshot else []),
            ],
            required=[cache / "fused_fixed_method8_summary.json"],
            env=primary_gpu_env,
            attempts=PROCESS_ATTEMPTS,
            input_identity=lambda seeds=seeds: {
                "method6": _recovery_inputs(method6_summary),
                "adjacency": file_resume_identity(adjacency),
                "seeds": file_resume_identity(seeds),
            },
        )
        native = side_dir / "native-recovery" / f"round-{round_index}"
        runner.action(
            f"{channel}-native-snapshot-round{round_index}",
            {"seeds": str(seeds), "cache": str(cache)},
            lambda native=native: _snapshot_native(cache, native),
            required=[native / "fused_fixed_method8_summary.json"],
        )
        complete = side_dir / "complete-recovery" / f"round-{round_index}"
        summary = complete / "fused_fixed_method8_summary.json"
        runner.command(
            f"{channel}-complete-recovery-round{round_index}",
            [
                "lightsheet",
                "cross-register",
                "complete-recovery",
                "--native-summary",
                native / "fused_fixed_method8_summary.json",
                "--output-dir",
                complete,
            ],
            required=[summary, complete / "manifest.json"],
            input_identity=lambda native=native: _recovery_inputs(native / "fused_fixed_method8_summary.json"),
        )
        proposed = side_dir / "seeds" / f"proposed-round-{round_index}.json"

        def propose(summary: Path = summary, proposed: Path = proposed) -> None:
            propose_manual_initializations(complete_summary=summary, master=master, channel=channel)
            _write_seeds(proposed, channel, _seed_entries(master, channel))

        runner.action(
            f"{channel}-propose-seeds-round{round_index}",
            {"summary": str(summary), "windows": _recovery_inputs(summary)},
            propose,
            required=[proposed],
        )
        current = _seed_entries(master, channel)
        count = seedable_window_count(summary)
        if current == snapshot:
            return summary
        automatic = current == json.loads(proposed.read_text())["channels"][channel]
        manual_changed = {
            name: entry for name, entry in current.items() if entry.get("seed_origin") == "manual"
        } != {
            name: entry for name, entry in snapshot.items() if entry.get("seed_origin") == "manual"
        }
        if automatic and not manual_changed and previous_count is not None and count >= previous_count:
            return summary
        previous_count = count
        round_index += 1


def _canonical_side_positions(runner: StageRunner, paths: HcrPaths, positions: Path, side: str) -> Path:
    """One side's tiles of the accepted combined 561 placement: the Method 6 fixed positions of every later label."""
    output = paths.root / "canonical" / f"positions-{side}.json"

    def write() -> None:
        payload = json.loads(positions.read_text())
        payload["tiles"] = [tile for tile in payload["tiles"] if tile["source_side"] == side]
        atomic_json(output, payload)

    runner.action(f"canonical-positions-{side}", {"positions": sha256_file(positions)}, write, required=[output])
    return output


def _screen_registration(
    runner: StageRunner,
    paths: HcrPaths,
    label: str,
    summaries: dict[str, Path],
    *,
    fixed_fused: Path,
    fixed_threshold: float,
) -> None:
    """WORKFLOW_HCR.md: list every recovery defect and render combined coverage through it before materialization."""
    key = "-".join(f"{side}-{summary.parent.name}" for side, summary in sorted(summaries.items()))
    screen = paths.channel(label) / "screen" / key
    report = screen / "report.json"
    datasets = {f"{label}-{side}": summary for side, summary in summaries.items()}
    # The report records what it screened, so the renders and the approval bind to these windows, fixed volume, and
    # threshold; a later change to any of them invalidates both instead of passing silently.
    screened = {
        "windows": {name: _recovery_inputs(summary) for name, summary in datasets.items()},
        "fixed_fused": source_tile_resume_identity(fixed_fused, require_completion=False),
        "fixed_threshold": fixed_threshold,
        "policy": SCREEN_ARTIFACT_TYPE,
    }
    runner.action(
        f"{label}-screen-{key}",
        screened,
        lambda: atomic_json(report, screen_recovery(datasets, fixed_fused=fixed_fused) | {"screened": screened}),
        required=[report],
    )
    for plane in json.loads(report.read_text())["qc_planes"]:
        output = screen / f"qc-z{plane:04d}"
        runner.command(
            f"{label}-screen-{key}-z{plane:04d}",
            [
                "lightsheet",
                "cross-register",
                "prefusion-qc",
                *(argument for name, summary in datasets.items() for argument in ("--summary", f"{name}={summary}")),
                "--fixed-volume",
                fixed_fused,
                "--fixed-threshold",
                str(fixed_threshold),
                "--output-dir",
                output,
                "--fixed-z-index",
                str(plane),
            ],
            required=[output / "prefusion-hole-review.json"],
            input_identity=lambda: file_resume_identity(report),
        )
    gate_approval(
        screen / "approval.json",
        {"report": sha256_file(report)},
        message=(
            f"Read {report} and the hole maps in {screen}: seed or accept every defect. To seed, add entries to "
            f"{paths.manual_initializations} and rerun; to accept the remaining defects, set approved=true"
        ),
        artifact_type="squisher_lightsheet.hcr_approval.v1",
    )


def _reviewed_offset_initializer(
    runner: StageRunner, *, source: ChannelSource, reference_label: str, measurement: Path,
    moving_placed: Path, output: Path, inputs: object,
) -> bool:
    """Publish an explicitly reviewed failed offset only as the initializer for local fused-fixed refinement."""
    approval = measurement.parent / "rough-initializer-approval.json"
    if not approval.exists():
        return False
    review = output.parent / "offset-review"
    identity = {
        "inputs": json_sha256(inputs), "moving": sha256_file(moving_placed),
        "measurement": sha256_file(measurement),
        "qc": sha256_file(review / f"held-out-{reference_label}-contact-sheet.png"),
        "render": sha256_file(review / "render.json"),
    }
    gate_approval(
        approval, identity,
        message=f"Visually review {source.label}-{source.side} offset and approve it only as a rough initializer",
        artifact_type="squisher_lightsheet.hcr_offset_initializer_approval.v1",
    )

    def publish() -> None:
        summary = json.loads(measurement.read_text())
        if summary["artifact_type"] != "squisher_lightsheet.acquisition_offset.v2":
            raise ValueError(f"Not an acquisition-offset measurement: {measurement}")
        payload = shifted_acquisition_positions(
            json.loads(moving_placed.read_text()), summary["median_correction_zyx_um"], summary_path=measurement,
        )
        payload["rough_placement_review"] = {
            "approval": str(approval), "identity": identity,
            "held_out_validation_passed": summary["validation"]["passed"],
        }
        atomic_json(output, payload)

    runner.action(
        f"{source.label}-{source.side}-reviewed-offset-initializer",
        {"inputs": identity, "approval": sha256_file(approval)}, publish, required=[output],
    )
    return True


def _offset_placement(
    runner: StageRunner,
    paths: HcrPaths,
    manifest: dict[str, Any],
    sources: list[ChannelSource],
    source: ChannelSource,
    reference: ChannelSource,
    *,
    moving_placed: Path,
    reference_positions: Path,
    reference_tiles: Path,
    output: Path,
    inputs: object,
) -> None:
    """Place a separate acquisition by its measured depth and XY offset to the reference acquisition's tiles.

    Which reference channel shares enough structure with the moving label is measured, not assumed: every channel
    imaged in the reference label's track is a candidate, each validated on held-out tiles. A single candidate is used
    if it validates; several go to the author as a choice bound to their measurements.
    """
    acquisition = next(
        item for item in manifest["acquisitions"] if Path(item["directory"]).name == reference.acquisition
    )
    track = next(track for track in acquisition["tracks"] if reference.index in track)
    candidates = sorted(
        (item for item in sources if item.acquisition == reference.acquisition and item.index in track),
        key=lambda item: item.index,
    )
    summaries = {}
    for candidate in candidates:
        directory = output.parent / f"offset-{candidate.label}"
        summaries[candidate.label] = directory / "acquisition-offset.json"
        runner.action(
            f"{source.label}-{source.side}-acquisition-offset-{candidate.label}",
            {"inputs": inputs, "moving_channel": source.index, "reference_channel": candidate.index},
            lambda directory=directory, candidate=candidate: run_acquisition_offset(
                moving_position=moving_placed,
                reference_position=reference_positions,
                moving_tile_dir=paths.deconv(source.acquisition),
                reference_tile_dir=reference_tiles,
                output_dir=directory,
                output_position=directory / "offset.positions.json",
                moving_channel=source.index,
                reference_channel=candidate.index,
            ),
            required=[summaries[candidate.label]],
        )
    measured = {label: json.loads(path.read_text()) for label, path in summaries.items()}
    if len(candidates) == 1:
        selected = candidates[0].label
    else:
        selected = gate_choice(
            paths.channel(source.label) / f"{source.side}-offset-reference.json",
            {label: _offset_metrics(summary) for label, summary in measured.items()},
            identity={label: sha256_file(path) for label, path in summaries.items()},
            message=(
                f"Compare the reference channels measured for {source.label}-{source.side} (held-out validation, peak "
                "margins, residuals; full reports beside each offset) and set selected to one that passes"
            ),
            artifact_type="squisher_lightsheet.hcr_offset_reference.v1",
        )
    if not measured[selected]["validation"]["passed"]:
        if _reviewed_offset_initializer(
            runner, source=source, reference_label=selected, measurement=summaries[selected],
            moving_placed=moving_placed, output=output, inputs=inputs,
        ):
            return
        prefix = "The only reference channel," if len(candidates) == 1 else "Selected reference channel"
        raise ValueError(f"{prefix} {selected} fails held-out validation: {summaries[selected]}")
    chosen = summaries[selected].parent / "offset.positions.json"
    runner.action(
        f"{source.label}-{source.side}-offset-placement",
        {"selected": selected, "positions": sha256_file(chosen)},
        lambda: shutil.copyfile(chosen, output),
        required=[output],
    )


def _offset_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    checks = summary["validation"]["checks"]
    return {
        "validation_passed": summary["validation"]["passed"],
        "held_out_tiles": summary["validation"]["held_out_tiles"],
        "failed_checks": sum(not check["passed"] for check in checks),
        "minimum_peak_margin": min(check["peak_margin"] for check in checks),
        "maximum_xy_residual_px": max(
            float(np.max(np.abs(np.subtract(check["shift_yx_px"], check["expected_shift_yx_px"])))) for check in checks
        ),
        "median_correction_zyx_um": summary["median_correction_zyx_um"],
        "correction_mad_zyx_um": summary["correction_mad_zyx_um"],
    }


def _place_label(
    runner: StageRunner,
    paths: HcrPaths,
    manifest: dict[str, Any],
    sources: list[ChannelSource],
    label: str,
    by_side: dict[str, ChannelSource],
    *,
    placement: dict[str, Path],
    placed: Path,
) -> None:
    """Place this label's native geometry on both sides with the accepted 561 placement's moving-side shift.

    Writes one ``<side>.positions.json`` per side into ``placed`` (229Tnc run-xreg.sh for CL/CR).
    """
    fixed_side, moving_side = LAYOUT_SIDES[manifest["layout"]]
    raw = {Path(acquisition["directory"]).name: acquisition["directory"] for acquisition in manifest["acquisitions"]}
    registered = {side: paths.registered_positions(source.acquisition, label) for side, source in by_side.items()}
    metadata = placed / "metadata.positions.json"
    runner.command(
        f"{label}-placement-metadata",
        [
            "lightsheet",
            "position",
            "--left-dir",
            raw[by_side[fixed_side].acquisition],
            "--right-dir",
            raw[by_side[moving_side].acquisition],
            "--output",
            metadata,
            "--mode",
            LAYOUT_MODES[manifest["layout"]],
        ],
        required=[metadata],
    )
    alignment = json.loads(placement["positions"].read_text())["alignment"]
    if manifest["layout"] == "TL/TR":
        shift = alignment["net_right_shift_after_centroid_um_zyx"]

        def place() -> None:
            write_joined_sides(
                metadata=metadata,
                left_position=registered["TL"],
                right_position=registered["TR"],
                right_shift_zyx_um=np.asarray(shift, dtype=np.float64),
                output=placed / "joined.positions.json",
            )
            split_sides(placed / "joined.positions.json", placed)

        runner.action(
            f"{label}-placement",
            {
                "metadata": sha256_file(metadata),
                "registered": {side: sha256_file(path) for side, path in registered.items()},
                "shift": shift,
            },
            place,
            required=[placed / "tl.positions.json", placed / "tr.positions.json"],
        )
        return
    shift = alignment["net_cr_shift_after_centroid_um_zyx"]
    runner.command(
        f"{label}-placement",
        [
            "lightsheet",
            "align-registered-lr",
            "--metadata-position",
            metadata,
            "--left-position",
            registered["CL"],
            "--right-position",
            registered["CR"],
            "--left-fusion",
            paths.side_fused("CL", "561", single_561_source(sources, "CL").index),
            "--right-fusion",
            paths.side_fused("CR", "561", single_561_source(sources, "CR").index),
            "--output-dir",
            placed,
            "--fusion-level",
            "2",
            "--manual-cr-shift-zyx-um=" + ",".join(f"{float(value):.6f}" for value in shift),
        ],
        required=[placed / "cl.positions.json", placed / "cr.positions.json"],
        input_identity=lambda: {
            "metadata": file_resume_identity(metadata),
            "positions": [file_resume_identity(registered[side]) for side in ("CL", "CR")],
        },
    )


def _reviewed_no_shift_rough(
    runner: StageRunner, *, label: str, side: str, moving_placed: Path, output: Path,
    inputs: dict[str, object],
) -> bool:
    """Retain a same-acquisition rough placement after explicit review of its rejected phase candidate."""
    approval = output.parent / "no-shift-approval.json"
    if not approval.exists():
        return False
    phase_name = f"{label}-{side}-global-phase"
    attempt = runner.stages / f"{phase_name}.started.json"
    if json.loads(attempt.read_text())["inputs"] != inputs:
        raise ValueError(f"Reviewed phase attempt used different inputs: {attempt}")
    identity = {
        "inputs": json_sha256(inputs),
        "phase_attempt": sha256_file(attempt),
        "phase_log": sha256_file(runner.logs / f"{phase_name}.log"),
        **{name: sha256_file(output.parent / f"{name}.png") for name in ("before", "after")},
    }
    gate_approval(
        approval, identity,
        message=f"Review {label}-{side} phase QC and approve retaining its rough placement without a shift",
        artifact_type="squisher_lightsheet.hcr_rough_no_shift_approval.v1",
    )

    def retain() -> None:
        payload = json.loads(moving_placed.read_text())
        payload["rough_placement_review"] = {
            "approval": str(approval), "identity": identity,
            "total_shift_to_apply_moving_zyx_um": [0.0, 0.0, 0.0],
        }
        atomic_json(output, payload)

    runner.action(
        f"{label}-{side}-reviewed-no-shift", {"inputs": identity, "approval": sha256_file(approval)},
        retain, required=[output],
    )
    return True


def cross_register_label(
    runner: StageRunner,
    paths: HcrPaths,
    manifest: dict[str, Any],
    sources: list[ChannelSource],
    label: str,
    *,
    placement: dict[str, Path],
    fused: dict[str, Path],
    fused_thresholds: dict[str, float],
    preview: Path,
    all_gpu_env: dict[str, str],
    primary_gpu_env: dict[str, str],
) -> tuple[Path, Path]:
    """Register one label on both sides to its fixed fused volume, screen, materialize, match, and fuse it."""
    reference_label = FIXED_LABELS[label]
    settings = CROSS_REGISTRATION[label]
    by_side = {source.side: source for source in sources if source.label == label}
    references = {source.side: source for source in sources if source.label == reference_label}
    directory = paths.channel(label)
    placed = directory / "placed"
    sides = LAYOUT_SIDES[manifest["layout"]]
    _place_label(runner, paths, manifest, sources, label, by_side, placement=placement, placed=placed)
    summaries: dict[str, Path] = {}
    rough: dict[str, Path] = {}
    for side, source in sorted(by_side.items()):
        reference = references[side]
        fixed_positions = _canonical_side_positions(runner, paths, placement["positions"], side)
        if reference_label == "561":
            reference_positions, reference_tiles = fixed_positions, placement["deconv"]
        else:
            reference_positions = paths.channel(reference_label) / side / "rough" / "rough.positions.json"
            reference_tiles = paths.deconv(reference.acquisition)
        side_dir = directory / side
        rough[side] = side_dir / "rough" / "rough.positions.json"
        moving_placed = placed / f"{side.lower()}.positions.json"
        rough_inputs = {
            "moving": registration_input_fingerprint(moving_placed, paths.deconv(source.acquisition)),
            "reference": registration_input_fingerprint(reference_positions, reference_tiles),
        }
        if source.acquisition == reference.acquisition:
            reviewed = _reviewed_no_shift_rough(
                runner, label=label, side=side, moving_placed=moving_placed, output=rough[side], inputs=rough_inputs,
            )
            if not reviewed:
                runner.command(
                    f"{label}-{side}-global-phase",
                    [
                        "lightsheet",
                        "cross-register",
                        "global-phase",
                        "--fixed-position",
                        reference_positions,
                        "--moving-position",
                        moving_placed,
                        "--fixed-tile-dir",
                        reference_tiles,
                        "--moving-tile-dir",
                        paths.deconv(source.acquisition),
                        "--output-dir",
                        rough[side].parent,
                        "--output-position",
                        rough[side],
                        "--fixed-channel",
                        str(reference.index),
                        "--moving-channel",
                        str(source.index),
                        "--level",
                        "2",
                        "--z-slab-planes",
                        "32",
                        "--fixed-intensity-transform",
                        "log1p",
                        "--moving-intensity-transform",
                        settings.rough_intensity_transform,
                        "--highpass-sigma-px",
                        "8",
                    ],
                    required=[rough[side]],
                    env=primary_gpu_env,
                    input_identity=lambda rough_inputs=rough_inputs: rough_inputs,
                )
        else:
            # A separate acquisition shares the stage lattice but starts its stacks at an unrecorded depth.
            _offset_placement(
                runner,
                paths,
                manifest,
                sources,
                source,
                reference,
                moving_placed=moving_placed,
                reference_positions=reference_positions,
                reference_tiles=reference_tiles,
                output=rough[side],
                inputs=rough_inputs,
            )
        world_affine = json.loads(placement["positions"].read_text())["alignment"].get("global_affine_zyx_um")
        if side == "TR" and world_affine is not None:
            canonical_rough = side_dir / "rough-canonical" / "rough.positions.json"

            def attach_affine(source=rough[side], output=canonical_rough, affine=world_affine) -> None:
                payload = json.loads(source.read_text())
                for tile in payload["tiles"]:
                    tile["placement_affine_zyx_um"] = affine
                atomic_json(output, payload)

            runner.action(
                f"{label}-{side}-canonical-initializer",
                {"rough": sha256_file(rough[side]), "affine": world_affine}, attach_affine,
                required=[canonical_rough],
            )
            rough[side] = canonical_rough
        method6 = side_dir / "method6"
        registration = paths.registration(source.acquisition, label)
        runner.command(
            f"{label}-{side}-method6",
            [
                "lightsheet",
                "cross-register",
                "method6",
                "--fixed-position",
                fixed_positions,
                "--moving-position",
                rough[side],
                "--moving-source-position",
                registration,
                "--fixed-fused",
                fused[reference_label],
                "--output-dir",
                method6,
                "--output-registration",
                method6 / "registration.json",
                "--moving-channel",
                str(source.index),
                "--fit-intensity-transform",
                settings.fit_intensity_transform,
                "--starting-affine-matrix-zyx",
                side_starting_matrix(settings.starting_matrix, side),
                "--min-corr",
                "0.15",
                "--min-grad-ncc",
                str(settings.minimum_gradient_ncc),
                "--fixed-mask-threshold",
                str(fused_thresholds[reference_label]),
                "--source-label",
                label,
                "--target-label",
                reference_label,
                "--workers",
                "1",
                "--devices",
                "0",
                "--resume",
            ],
            required=[method6 / "fused_fixed_method8_summary.json"],
            env=primary_gpu_env,
            attempts=PROCESS_ATTEMPTS,
            input_identity=lambda side=side, fixed_positions=fixed_positions, registration=registration: {
                "moving": registration_input_fingerprint(rough[side], paths.deconv(by_side[side].acquisition)),
                "registration": file_resume_identity(registration),
                "fixed_position": file_resume_identity(fixed_positions),
                "fixed_fused": source_tile_resume_identity(fused[reference_label], require_completion=False),
            },
        )
        summaries[side] = _recover_in_rounds(
            runner, paths, source, method6 / "fused_fixed_method8_summary.json", primary_gpu_env=primary_gpu_env
        )
    _screen_registration(
        runner,
        paths,
        label,
        summaries,
        fixed_fused=fused[reference_label],
        fixed_threshold=fused_thresholds[reference_label],
    )
    materialized: dict[str, Path] = {}
    for side, source in sorted(by_side.items()):
        output = directory / f"materialized-{side}"
        materialized[side] = output
        correction = paths.residual(source.acquisition, label) / "correction.json"
        registration = paths.registration(source.acquisition, label)
        runner.command(
            f"{label}-{side}-materialize",
            [
                "lightsheet",
                "fused-fixed-materialize-overlap",
                "--moving-position",
                rough[side],
                "--moving-source-registration",
                registration,
                "--source-summary",
                summaries[side],
                "--output-dir",
                output,
                "--source-channel",
                str(source.index),
                "--output-channel-label",
                label,
                # Applied here once; the fusion of materialized windows never reapplies it (WORKFLOW_HCR.md step 3).
                "--residual-correction",
                correction,
                "--core-shape-zyx",
                "480,480,480",
                "--window-shape-zyx",
                "528,528,528",
                "--level-factor-zyx",
                "1,1,1",
                "--output-codec",
                "zstd",
                "--zstd-level",
                "3",
                "--workers",
                "4",
                "--resume",
            ],
            required=[
                output / "fused_fixed_materialized_chunks.summary.json",
                output / "fused_fixed_materialized_chunks.positions.json",
                output / "fused_fixed_materialized_chunks.registration.json",
            ],
            env=primary_gpu_env,
            input_identity=lambda side=side, correction=correction, registration=registration: {
                "moving": registration_input_fingerprint(rough[side], paths.deconv(by_side[side].acquisition)),
                "registration": file_resume_identity(registration),
                "recovery": _recovery_inputs(summaries[side]),
                "correction": file_resume_identity(correction),
            },
        )
    joined = directory / "materialized-joined"
    joined_positions = joined / "fused_fixed_materialized_chunks.positions.json"
    joined_registration = joined / "fused_fixed_materialized_chunks.registration.json"
    runner.command(
        f"{label}-join",
        [
            "lightsheet",
            "join-materialized-sides",
            "--side-output-dir",
            materialized[sides[0]],
            "--side-output-dir",
            materialized[sides[1]],
            "--output-dir",
            joined,
        ],
        required=[joined_positions, joined_registration],
        input_identity=lambda: {
            side: file_resume_identity(output / "fused_fixed_materialized_chunks.summary.json")
            for side, output in materialized.items()
        },
    )
    match = directory / "match"
    # Materialized windows already carry each dataset's correction; only the registered CR field is fitted.
    percentiles = shared_slab_percentiles(preview, joined_positions, side_field="source_view", sides=("L", "R"))
    runner.command(
        f"{label}-match",
        [
            "lightsheet",
            "cross-dataset-match",
            "--fixed-fused",
            preview,
            "--registration",
            joined_registration,
            "--output-dir",
            match,
            "--reference-view",
            "L",
            "--moving-view",
            "R",
            "--channel",
            "0",
            "--source-level",
            "0",
            "--stride",
            "4",
            *(argument for value in percentiles for argument in ("--z-percentile", f"{value:.6f}")),
            "--workers",
            "8",
        ],
        required=[match / "match.json", match / "correction.json", match / "registered-view-correction.json"],
        validate=lambda: require_held_out_improvement(match / "match.json"),
        input_identity=lambda: {
            "preview": source_tile_resume_identity(preview, require_completion=False),
            "registration": file_resume_identity(joined_registration),
        },
    )
    output = directory / "fusion" / "fused.ch0.ome.zarr"
    runner.command(
        f"{label}-fuse",
        [
            "lightsheet",
            "fuse",
            joined,
            "--position-input",
            joined_positions,
            "--registration-input",
            joined_registration,
            "--residual-correction",
            match / "correction.json",
            "--registered-view-correction",
            match / "registered-view-correction.json",
            "--output",
            output.parent,
            "--channel",
            "0",
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
            "--output-grid-template",
            fused["561"],
            "--output-grid-template-level",
            "0",
            "--resume-fusion",
        ],
        required=[output / "zarr.json"],
        env=all_gpu_env,
        validate=lambda: require_complete_fusion(output),
        input_identity=lambda: {
            "joined": [file_resume_identity(path) for path in (joined_positions, joined_registration)],
            "match": [
                file_resume_identity(match / name) for name in ("correction.json", "registered-view-correction.json")
            ],
            "template": source_tile_resume_identity(fused["561"], require_completion=False),
        },
    )
    review = directory / f"fusion-{label}-threshold-review.level2.ome.tif"
    runner.command(
        f"{label}-threshold-review",
        ["lightsheet", "threshold-review", "--fixed-fused", output, "--output", review, "--level", "2"],
        required=[review],
        input_identity=lambda: source_tile_resume_identity(output, require_completion=False),
    )
    return output, review

