"""Resumable orchestration of WORKFLOW_HCR.md from an approved acquisition manifest.

The workflow stops at each human decision WORKFLOW_HCR.md requires. Rerunning the same command after answering the
emitted gate file resumes from the first incomplete stage. The manifest (``lightsheet hcr-manifest``) is the only
source of acquisitions, sides, and channel labels; it is re-verified against the tiles on every run.
"""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from squisher_lightsheet._legacy.stitch_20x_tl_multiview import (
    file_resume_identity,
    source_tile_resume_identity,
)
from squisher_lightsheet.hcr_acquisition import (
    ChannelSource,
    HcrPaths,
    channel_sources,
    fuse_side_561,
    preprocess_acquisition,
    register_native_geometry,
    registration_threshold_keys,
    require_complete_fusion,
)
from squisher_lightsheet.hcr_channels import CROSS_REGISTRATION, FIXED_LABELS, cross_register_label
from squisher_lightsheet.hcr_cleanup import execute_cleanup, plan_cleanup, verify_retained
from squisher_lightsheet.hcr_manifest import LAYOUT_SIDES, verify_manifest
from squisher_lightsheet.hcr_reference import fuse_canonical, fused_threshold, place_clcr, place_tltr
from squisher_lightsheet.ngff import require_same_multiscale_grid
from squisher_lightsheet.workflow_runner import (
    ReviewRequired,
    StageRunner,
    atomic_json,
    gate_approval,
    gate_numbers,
    gpu_uuids,
    json_sha256,
    sha256_file,
)


def supported_sources(manifest: dict[str, Any]) -> list[ChannelSource]:
    """The manifest's channel sources, refused before any processing unless the workflow can complete them.

    Every label must be acquired once on each side of the layout; 561 and 638 are fused jointly from both sides, so
    each needs one channel index on both; every other label needs the label it registers to.
    """
    if manifest["layout"] not in LAYOUT_SIDES:
        raise NotImplementedError(f"{manifest['layout']} placement is not packaged in hcr-workflow")
    layout_sides = LAYOUT_SIDES[manifest["layout"]]
    sources = channel_sources(manifest)
    labels = {source.label for source in sources}
    if "405" in labels:
        raise ValueError("HCR samples skip 405 (WORKFLOW_HCR.md); leave the 405 acquisitions out of the manifest")
    unknown = labels - {"561", "638", *CROSS_REGISTRATION}
    if unknown:
        raise ValueError(f"No HCR workflow path for labels {sorted(unknown)}")
    for label in sorted(labels):
        sides = [source for source in sources if source.label == label]
        if sorted(source.side for source in sides) != sorted(layout_sides):
            raise ValueError(
                f"{label} must be acquired once on each of {' and '.join(layout_sides)}, found {[s.side for s in sides]}"
            )
        if label in {"561", "638"} and sides[0].index != sides[1].index:
            raise ValueError(f"{label} is channel {sides[0].index} on {sides[0].side} but {sides[1].index} on "
                             f"{sides[1].side}; joint fusion needs one index")
        if label in CROSS_REGISTRATION and FIXED_LABELS[label] not in labels:
            raise ValueError(f"{label} registers to fused {FIXED_LABELS[label]}, which is not acquired")
    return sources


def _finalize(
    runner: StageRunner, paths: HcrPaths, fused: dict[str, Path]
) -> dict[str, Path]:
    """WORKFLOW_HCR.md steps 10-12: fused-volume subtraction, same-grid final QC, and the fused 561 NRRD export."""
    final = paths.root / "final"
    finals = dict(fused)
    if {"514", "488"} <= set(fused):
        model_dir = final / "subtraction-514-minus-488"
        model = model_dir / "fused_subtraction_model.json"
        runner.command(
            "subtraction-fit",
            ["lightsheet", "fit-fused-subtraction", fused["514"], fused["488"], "--output-dir", model_dir],
            required=[model, model_dir / "fused_subtraction_fit_qc.png"],
            input_identity=lambda: {
                label: source_tile_resume_identity(fused[label], require_completion=True) for label in ("514", "488")
            },
        )
        gate_approval(
            paths.root / "subtraction-approval.json",
            {"model": sha256_file(model)},
            message=f"Review {model_dir / 'fused_subtraction_fit_qc.png'} and the 514 = alpha * 488 + beta fit",
            artifact_type="squisher_lightsheet.hcr_approval.v1",
        )
        subtracted = final / "fusion-514-minus-488" / "fused.ch0.ome.zarr"
        runner.command(
            "subtraction",
            [
                "lightsheet",
                "subtract-fused",
                model,
                subtracted,
                "--workers",
                "16",
                "--summary-output",
                subtracted.parent / "fused_subtraction.summary.json",
            ],
            required=[subtracted / "zarr.json", subtracted.parent / "fused_subtraction.summary.json"],
            validate=lambda: require_complete_fusion(subtracted),
            input_identity=lambda: file_resume_identity(model),
        )
        finals["514-488"] = subtracted
    qc = final / "qc"
    middle = qc / "middle-z"
    runner.command(
        "final-qc-middle-z",
        [
            "lightsheet",
            "fused-middle-z-qc",
            *(argument for label, path in finals.items() for argument in ("--channel", f"{label}={path}")),
            "--fixed-volume",
            finals["561"],
            "--output-dir",
            middle,
        ],
        required=[middle / f"middle-z-{label}.png" for label in finals],
        input_identity=lambda: {
            label: source_tile_resume_identity(path, require_completion=True) for label, path in finals.items()
        },
    )
    # Every channel overlays the 561 frame.
    overlays = [("561", label) for label in finals if label != "561"]
    for reference, moving in overlays:
        output = qc / f"xyz-overlay-{moving}-on-{reference}"
        runner.command(
            f"final-qc-overlay-{moving}-on-{reference}",
            [
                "lightsheet",
                "fused-xyz-overlay-qc",
                "--reference-zarr",
                finals[reference],
                "--moving-zarr",
                finals[moving],
                "--output-dir",
                output,
            ],
            required=[output / "native_xyz_spanning_overlay_contact_sheet.png"],
            input_identity=lambda reference=reference, moving=moving: {
                label: source_tile_resume_identity(finals[label], require_completion=True)
                for label in (reference, moving)
            },
        )
    # Every final must lie on the canonical 561 multiscale grid at every level before its QC can be accepted.
    for path in finals.values():
        require_same_multiscale_grid(path, finals["561"])
    gate_approval(
        paths.root / "final-qc-approval.json",
        {str(path.relative_to(qc)): sha256_file(path) for path in sorted(qc.rglob("*.png"))},
        message=f"Review the same-grid middle-Z and XYZ overlay QC in {qc}, then set approved=true",
        artifact_type="squisher_lightsheet.hcr_approval.v1",
    )
    export = final / "export"
    nrrd, receipt = export / "561.level3.nrrd", export / "561.level3.json"
    runner.command(
        "export-nrrd-561",
        [
            "lightsheet",
            "export-nrrd",
            finals["561"],
            nrrd,
            "--level",
            "3",
            "--channel",
            "561",
            "--summary-output",
            receipt,
        ],
        required=[nrrd, receipt],
        validate=lambda: _require_validated_export(receipt, source=finals["561"], output=nrrd),
        input_identity=lambda: source_tile_resume_identity(finals["561"], require_completion=True),
    )
    return finals | {"nrrd_561": nrrd, "nrrd_561_receipt": receipt}


def _require_validated_export(receipt: Path, *, source: Path, output: Path) -> None:
    """Accept the export only as level 3 of the fused 561, written to ``output``, with a matching readback."""
    payload = json.loads(receipt.read_text())
    expected = {"source": str(source), "source_level": 3, "channel": "561", "output": str(output)}
    mismatched = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if mismatched:
        raise ValueError(f"NRRD export receipt {receipt} does not describe the expected export: {mismatched}")
    if payload.get("all_voxels_equal") is not True or not output.is_file():
        raise ValueError(f"NRRD export readback did not reproduce the source voxels, or {output} is missing")


def _cleanup(
    paths: HcrPaths,
    finals: dict[str, Path],
    *,
    manifest_identity: str,
    dry_run: bool,
    export: Path | None = None,
    export_receipt: Path | None = None,
) -> Path:
    """Plan, approve, and execute the retirement of large intermediates; the record is the terminal state.

    A dry run deletes nothing: it verifies a completed cleanup, or reports what an approved one would still delete.
    """
    directory = paths.root / "cleanup"
    plan, record = directory / "plan.json", directory / "record.json"
    if not plan.exists():
        assert export is not None and export_receipt is not None
        atomic_json(
            plan,
            plan_cleanup(
                paths.root,
                finals,
                canonical=finals["561"],
                export=export,
                export_receipt=export_receipt,
                manifest_identity=manifest_identity,
            ),
        )
    if dry_run:
        if record.is_file() and json.loads(record.read_text())["status"] == "complete":
            verify_retained(plan, record, manifest_identity=manifest_identity)
        else:
            count = len(json.loads(plan.read_text())["deletions"])
            print(f"[hcr] dry run: cleanup per {plan} would delete {count} payloads", flush=True)
        return record
    gate_approval(
        paths.root / "cleanup-approval.json",
        {"plan": sha256_file(plan)},
        message=(
            f"Review {plan}: it deletes the listed deconvolved tiles, side fusions, and materialized windows (sizes "
            "and final-fusion provenance coverage included). Set approved=true to delete them"
        ),
        artifact_type="squisher_lightsheet.hcr_approval.v1",
    )
    execute_cleanup(plan, record, manifest_identity=manifest_identity)
    verify_retained(plan, record, manifest_identity=manifest_identity)
    return record


def run_hcr_workflow(manifest_path: Path, *, output_root: Path | None = None, dry_run: bool = False) -> Path:
    """Run or resume WORKFLOW_HCR.md for the dataset an approved manifest describes."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)
    if manifest.get("approved") is not True:
        raise ReviewRequired("Review the acquisitions, sides, and channel labels, then set approved=true", manifest_path)
    supported_sources(manifest)
    root = (output_root or manifest_path.parent).expanduser().resolve()
    paths = HcrPaths(root=root)
    progress = root / "workflow-progress.json"
    if (root / "cleanup" / "record.json").is_file():
        # Cleanup retired the pixels upstream stages fingerprint: finish or verify it, and go no further.
        record = _cleanup(paths, {}, manifest_identity=json_sha256(manifest), dry_run=dry_run)
        print(f"[hcr] cleanup complete; retained finals verified against {record}", flush=True)
        return progress
    uuids = gpu_uuids()
    runner = StageRunner(
        logs=paths.logs,
        stages=paths.stages,
        prefix="hcr",
        artifact_type="squisher_lightsheet.hcr_stage.v1",
        dry_run=dry_run,
    )
    all_gpu_env = {"CUDA_VISIBLE_DEVICES": ",".join(uuids)}
    primary_gpu_env = {"CUDA_VISIBLE_DEVICES": uuids[0]}

    for acquisition in manifest["acquisitions"]:
        preprocess_acquisition(runner, paths, acquisition, all_gpu_env=all_gpu_env, primary_gpu_env=primary_gpu_env)
    for acquisition in manifest["acquisitions"]:
        name = Path(acquisition["directory"]).name
        review = paths.native_review(name)
        runner.command(
            f"{name}-native-review",
            [
                "lightsheet",
                "ome-metadata-dumb-stitch",
                "--input-dir",
                f"R={paths.deconv(name)}",
                "--output-dir",
                review,
                "--channels",
                ",".join(str(channel["index"]) for channel in acquisition["channels"]),
                "--level",
                "0",
                "--output-prefix",
                name,
                "--write-tiff",
            ],
            required=[review / f"{name}_omeMetadata_noBlend_noLines_manifest.json"],
            input_identity=lambda name=name: [
                source_tile_resume_identity(path, require_completion=True)
                for path in sorted(paths.deconv(name).glob("*.ome.zarr"))
            ],
        )
    thresholds = gate_numbers(
        root / "native-thresholds.json",
        registration_threshold_keys(manifest),
        message="Review the native-intensity TIFFs and fill one registration threshold per side and channel",
        artifact_type="squisher_lightsheet.hcr_thresholds.v1",
    )
    sources = register_native_geometry(runner, paths, manifest, thresholds, primary_gpu_env=primary_gpu_env)
    # WORKFLOW_HCR.md step 3: accept native geometry and every fit-residual correction before cross-registration.
    gate_approval(
        root / "native-geometry-approval.json",
        {
            f"{source.side}/{source.label}": json_sha256(
                {
                    "registration": sha256_file(paths.registration(source.acquisition, source.registration_label)),
                    "correction": sha256_file(paths.residual(source.acquisition, source.label) / "correction.json"),
                }
            )
            for source in sources
        },
        message="Review each registration's excluded tiles and every fit-residual report, then set approved=true",
        artifact_type="squisher_lightsheet.hcr_approval.v1",
    )
    sides = LAYOUT_SIDES[manifest["layout"]]
    if manifest["layout"] == "CL/CR":
        # align-registered-lr phase-correlates the dataset-local 561 mosaics; TL/TR correlates tiles directly.
        side_561 = {
            source.side: fuse_side_561(runner, paths, source, primary_gpu_env=primary_gpu_env)
            for source in sources
            if source.label == "561"
        }
        placement = place_clcr(runner, paths, manifest, sources, thresholds, all_gpu_env=all_gpu_env)
    else:
        side_561 = {}
        placement = place_tltr(runner, paths, manifest, sources, thresholds, all_gpu_env=all_gpu_env)
    canonical = fuse_canonical(
        runner, paths, sources, placement, sides=sides, all_gpu_env=all_gpu_env, primary_gpu_env=primary_gpu_env
    )
    fused = {label: canonical[label] for label in ("561", "638") if label in canonical}
    fused_thresholds = {
        "561": fused_threshold(
            root,
            "561",
            canonical["561"],
            canonical["threshold_review"],
            accepted_native={side: thresholds[f"{side}/561"] for side in sides},
        )
    }
    labels = {source.label for source in sources}
    registered = [label for label in ("514", "488") if label in labels]
    references = {FIXED_LABELS[label] for label in registered}
    for label in registered:
        fused[label], review = cross_register_label(
            runner,
            paths,
            manifest,
            sources,
            label,
            placement=placement,
            fused=fused,
            fused_thresholds=fused_thresholds,
            preview=canonical["preview"],
            all_gpu_env=all_gpu_env,
            primary_gpu_env=primary_gpu_env,
        )
        if label in references:
            fused_thresholds[label] = fused_threshold(root, label, fused[label], review)
    finals = _finalize(runner, paths, fused)
    progress_payload = {
        "artifact_type": "squisher_lightsheet.hcr_workflow_progress.v1",
        "manifest": str(manifest_path),
        "completed_through": "exported fused 561; finals " + ", ".join(finals),
        "side_561": {side: str(path) for side, path in side_561.items()},
        "placement": {key: str(path) for key, path in placement.items()},
        "canonical": {key: str(path) for key, path in canonical.items()},
        "finals": {label: str(path) for label, path in finals.items()},
        "fused_thresholds": fused_thresholds,
    }
    atomic_json(progress, progress_payload)
    stores = {label: path for label, path in finals.items() if label not in {"nrrd_561", "nrrd_561_receipt"}}
    record = _cleanup(
        paths,
        stores,
        manifest_identity=json_sha256(manifest),
        dry_run=dry_run,
        export=finals["nrrd_561"],
        export_receipt=finals["nrrd_561_receipt"],
    )
    atomic_json(progress, progress_payload | {"completed_through": "cleanup", "cleanup": str(record)})
    return progress
