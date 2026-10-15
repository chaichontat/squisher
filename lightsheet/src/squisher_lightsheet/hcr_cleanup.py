"""Retire an HCR dataset's large intermediates after its final QC and export are accepted (WORKFLOW_HCR.md cleanup).

Deconvolved tiles, side-local 561 fusions, and materialized registration windows are the pixel payloads the final
fusions were computed from; deconvolution's sampling caches go with them. Their deletion is planned as an exact inventory, approved by the author, executed with a
record of each deletion, and kept beside a copy of every deleted store's Zarr metadata. Everything else stays:
final stores and export, positions, registrations, recovery windows and seeds, corrections, preprocessing records,
stage identities, approvals, QC, and logs.

After cleanup the workflow's upstream stages can no longer fingerprint their inputs, so a completed record is the
dataset's terminal state: a rerun validates the retained finals and export receipt against the record and stops.
Recomputing anything upstream needs the retired pixels rebuilt explicitly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from squisher_lightsheet._legacy.stitch_20x_tl_multiview import source_tile_resume_identity
from squisher_lightsheet.ngff import require_same_multiscale_grid
from squisher_lightsheet.workflow_runner import atomic_json, sha256_file

PLAN_ARTIFACT_TYPE = "squisher_lightsheet.hcr_cleanup_plan.v1"
RECORD_ARTIFACT_TYPE = "squisher_lightsheet.hcr_cleanup_record.v1"
# Store metadata worth keeping once the chunks are gone: the root and each array's (pyramid level's) metadata.
METADATA_NAMES = ("zarr.json", ".zattrs", ".zgroup", ".zarray", "squisher.complete.json")


def _bytes(path: Path) -> int:
    result = subprocess.run(["du", "-s", "--block-size=1", str(path)], check=True, capture_output=True, text=True)
    return int(result.stdout.split()[0])


def retired_payloads(root: Path) -> list[tuple[str, Path]]:
    """The intermediates cleanup retires, by kind, under an HCR workflow root."""
    payloads = [("deconvolved_tile", path) for path in sorted(root.glob("acquisitions/*/deconvolved/*.ome.zarr"))]
    payloads += [("side_fusion", path) for path in sorted(root.glob("sides/*/fusion-*/*.ome.zarr"))]
    payloads += [
        ("materialized_windows", path) for path in sorted(root.glob("channels/*/materialized-*/materialized_tiles"))
    ]
    # The joined 561 view only links deconvolved tiles, which are retired above.
    payloads += [("tile_links", path) for path in sorted(root.glob("placement/deconv-*")) if path.is_dir()]
    # Deconvolution's sampling caches; the BaSiC models and scaling records beside them stay.
    payloads += [
        ("sampling_cache", path)
        for pattern in ("acquisitions/*/basic/sample-cache", "acquisitions/*/scale/float32-samples")
        for path in sorted(root.glob(pattern))
    ]
    return payloads


def _provenance(final: Path) -> dict[str, Any]:
    manifest = final / "provenance" / "manifest.json"
    if not manifest.is_file():
        return {"status": "missing", "unresolved": []}
    payload = json.loads(manifest.read_text())
    return {"status": payload["coverage"]["status"], "unresolved": payload["coverage"]["unresolved"]}


def plan_cleanup(
    root: Path,
    finals: dict[str, Path],
    *,
    canonical: Path,
    export: Path,
    export_receipt: Path,
    manifest_identity: str,
) -> dict[str, Any]:
    """Inventory what cleanup deletes and what it retains, with each final fusion's recorded provenance coverage.

    The plan binds the manifest that produced the dataset, every retained final, and the export's content, so the
    terminal state can be verified against them later.
    """
    return {
        "artifact_type": PLAN_ARTIFACT_TYPE,
        "root": str(root),
        "manifest_sha256": manifest_identity,
        "deletions": [
            {"kind": kind, "path": str(path), "bytes": _bytes(path)} for kind, path in retired_payloads(root)
        ],
        "canonical": str(canonical),
        "finals": {
            label: {
                "path": str(path),
                "identity": source_tile_resume_identity(path, require_completion=True),
                "provenance": _provenance(path),
            }
            for label, path in finals.items()
        },
        "export": {
            "nrrd": str(export),
            "nrrd_sha256": sha256_file(export),
            "receipt": str(export_receipt),
            "receipt_sha256": sha256_file(export_receipt),
        },
    }


def _stores(path: Path) -> list[Path]:
    """The OME-Zarr stores in a retired payload: the payload itself, or each store directly inside it."""
    if any((path / name).is_file() for name in METADATA_NAMES):
        return [path]
    return sorted(child for child in path.iterdir() if child.is_dir())


def _retain_metadata(path: Path, retained: Path) -> None:
    """Copy each store's root and array (pyramid level) metadata out of a payload before it is deleted."""
    for store in _stores(path):
        copied = 0
        for directory in [store, *sorted(child for child in store.iterdir() if child.is_dir())]:
            for name in METADATA_NAMES:
                source = directory / name
                if source.is_file():
                    destination = retained / directory.relative_to(path.parent) / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    copied += directory == store
        if not copied:
            raise ValueError(f"{store} holds no Zarr root metadata to retain; refusing to delete it")


def _verify_finals(plan: dict[str, Any]) -> None:
    canonical = Path(plan["canonical"])
    for label, final in plan["finals"].items():
        path = Path(final["path"])
        current = json.loads(json.dumps(source_tile_resume_identity(path, require_completion=True)))
        if current != final["identity"]:
            raise ValueError(f"Retained final {label} changed since the cleanup plan: {final['path']}")
        require_same_multiscale_grid(path, canonical)
    export = plan["export"]
    for key in ("nrrd", "receipt"):
        if not Path(export[key]).is_file() or sha256_file(Path(export[key])) != export[f"{key}_sha256"]:
            raise ValueError(f"Export {key} is missing or changed since the cleanup plan: {export[key]}")


def execute_cleanup(plan_path: Path, record_path: Path, *, manifest_identity: str) -> dict[str, Any]:
    """Delete the planned payloads, recording each step; resumes an interrupted cleanup from its record.

    The record exists before anything is deleted, so an interruption can never leave deleted inputs without it.
    """
    plan = json.loads(plan_path.read_text())
    if plan["manifest_sha256"] != manifest_identity:
        raise ValueError(f"{plan_path} was planned for a different manifest")
    _verify_finals(plan)
    if not record_path.is_file():
        atomic_json(
            record_path,
            {
                "artifact_type": RECORD_ARTIFACT_TYPE,
                "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
                "status": "in_progress",
                "entries": {},
            },
        )
    record = json.loads(record_path.read_text())
    if record["plan"]["sha256"] != sha256_file(plan_path):
        raise ValueError(f"{record_path} records a different cleanup plan than {plan_path}")
    root = Path(plan["root"])
    retained = record_path.parent / "retained-metadata"
    for deletion in plan["deletions"]:
        path = Path(deletion["path"])
        entry = record["entries"].get(deletion["path"])
        if entry is not None and entry["state"] == "deleted":
            continue
        if entry is None:
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"Planned deletion {path} is not a directory")
            if deletion["kind"] not in {"tile_links", "sampling_cache"}:
                _retain_metadata(path, retained / path.parent.relative_to(root))
            record["entries"][deletion["path"]] = {**deletion, "state": "deleting"}
            atomic_json(record_path, record)
        if path.exists():
            shutil.rmtree(path)
        record["entries"][deletion["path"]] |= {"state": "deleted", "deleted_at": datetime.now().astimezone().isoformat()}
        atomic_json(record_path, record)
    record["status"] = "complete"
    record["completed_at"] = datetime.now().astimezone().isoformat()
    atomic_json(record_path, record)
    return record


def verify_retained(plan_path: Path, record_path: Path, *, manifest_identity: str) -> None:
    """Fail unless cleanup completed for this manifest and every retained final and the export are unchanged."""
    record = json.loads(record_path.read_text())
    if record["status"] != "complete":
        raise ValueError(f"Cleanup is not complete: {record_path}")
    if record["plan"]["sha256"] != sha256_file(plan_path):
        raise ValueError(f"{record_path} records a different cleanup plan than {plan_path}")
    plan = json.loads(plan_path.read_text())
    if plan["manifest_sha256"] != manifest_identity:
        raise ValueError(f"{plan_path} was planned for a different manifest")
    _verify_finals(plan)
