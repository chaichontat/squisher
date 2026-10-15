from __future__ import annotations

import json
from pathlib import Path

import pytest
import zarr

from squisher_lightsheet import hcr_acquisition, hcr_cleanup, hcr_workflow
from squisher_lightsheet.hcr_cleanup import execute_cleanup, plan_cleanup, retired_payloads, verify_retained
from squisher_lightsheet.ngff import require_same_multiscale_grid
from squisher_lightsheet.workflow_runner import ReviewRequired

MANIFEST = "manifest-sha"


def _store(path: Path, *, complete: bool = True) -> Path:
    (path / "0" / "c").mkdir(parents=True)
    (path / "zarr.json").write_text(json.dumps({"attributes": {"squisher_complete": complete}}))
    (path / "0" / "zarr.json").write_text(json.dumps({"shape": [2, 2, 2]}))
    (path / "0" / "c" / "0").write_bytes(b"chunk")
    return path


def _fused(path: Path, *, origin_z: float = 0.0) -> Path:
    group = zarr.open_group(str(path), mode="w")
    for level, factor in enumerate((1, 2)):
        group.create_array(str(level), shape=(8, 8 // factor, 8 // factor), dtype="uint16")
    group.attrs["multiscales"] = [
        {
            "axes": ["z", "y", "x"],
            "datasets": [
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.8, 0.3 * factor, 0.3 * factor]},
                        {"type": "translation", "translation": [origin_z, 0.0, 0.0]},
                    ],
                }
                for level, factor in enumerate((1, 2))
            ],
        }
    ]
    group.attrs["squisher_complete"] = True
    return path


def _dataset(root: Path) -> dict[str, Path]:
    _store(root / "acquisitions" / "CL-main" / "deconvolved" / "CL-main.000.ome.zarr")
    (root / "acquisitions" / "CL-main" / "basic" / "sample-cache").mkdir(parents=True)
    (root / "acquisitions" / "CL-main" / "basic" / "CL-main-ch0.pkl").write_bytes(b"model")
    _store(root / "sides" / "CL" / "fusion-561" / "fused.ch0.ome.zarr")
    windows = root / "channels" / "514" / "materialized-CL" / "materialized_tiles"
    _store(windows / "w.ome.zarr")
    (root / "channels" / "514" / "materialized-CL" / "fused_fixed_materialized_chunks.summary.json").write_text("{}")
    links = root / "placement" / "deconv-561"
    links.mkdir(parents=True)
    (links / "CL-main.000.ome.zarr").symlink_to(root / "acquisitions" / "CL-main" / "deconvolved" / "CL-main.000.ome.zarr")
    export = root / "final" / "export"
    export.mkdir(parents=True)
    (export / "561.level3.nrrd").write_bytes(b"nrrd")
    (export / "561.level3.json").write_text(json.dumps({"all_voxels_equal": True}))
    return {
        "561": _fused(root / "canonical" / "fusion-561" / "fused.ch0.ome.zarr"),
        "514": _fused(root / "channels" / "514" / "fusion" / "fused.ch0.ome.zarr"),
        "nrrd": export / "561.level3.nrrd",
        "receipt": export / "561.level3.json",
    }


def _plan(root: Path, items: dict[str, Path]) -> Path:
    plan = root / "cleanup" / "plan.json"
    plan.parent.mkdir(exist_ok=True)
    finals = {label: items[label] for label in ("561", "514")}
    plan.write_text(
        json.dumps(
            plan_cleanup(
                root,
                finals,
                canonical=items["561"],
                export=items["nrrd"],
                export_receipt=items["receipt"],
                manifest_identity=MANIFEST,
            )
        )
    )
    return plan


def test_cleanup_retires_payloads_and_keeps_every_store_s_metadata(tmp_path: Path) -> None:
    items = _dataset(tmp_path)
    plan = _plan(tmp_path, items)
    record = tmp_path / "cleanup" / "record.json"

    result = execute_cleanup(plan, record, manifest_identity=MANIFEST)

    assert result["status"] == "complete"
    assert {entry["kind"] for entry in result["entries"].values()} == {
        "deconvolved_tile",
        "side_fusion",
        "materialized_windows",
        "tile_links",
        "sampling_cache",
    }
    assert not retired_payloads(tmp_path)
    retained = tmp_path / "cleanup" / "retained-metadata"
    assert (retained / "acquisitions/CL-main/deconvolved/CL-main.000.ome.zarr/0/zarr.json").is_file()
    assert not (retained / "acquisitions/CL-main/deconvolved/CL-main.000.ome.zarr/0/c").exists()
    # Each materialized window is a store of its own; its array metadata is kept too.
    assert (retained / "channels/514/materialized-CL/materialized_tiles/w.ome.zarr/0/zarr.json").is_file()
    assert (tmp_path / "acquisitions" / "CL-main" / "basic" / "CL-main-ch0.pkl").is_file()
    verify_retained(plan, record, manifest_identity=MANIFEST)


def test_cleanup_records_its_start_before_deleting_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = _dataset(tmp_path)
    plan = _plan(tmp_path, items)
    record = tmp_path / "cleanup" / "record.json"

    def interrupted(path: Path) -> None:
        raise RuntimeError("interrupted mid-deletion")

    monkeypatch.setattr(hcr_cleanup.shutil, "rmtree", interrupted)
    with pytest.raises(RuntimeError, match="interrupted mid-deletion"):
        execute_cleanup(plan, record, manifest_identity=MANIFEST)
    entries = json.loads(record.read_text())["entries"]
    assert [entry["state"] for entry in entries.values()] == ["deleting"]

    monkeypatch.undo()
    assert execute_cleanup(plan, record, manifest_identity=MANIFEST)["status"] == "complete"
    assert not retired_payloads(tmp_path)


@pytest.mark.parametrize(
    ("damage", "error"),
    [
        (lambda items: items["nrrd"].unlink(), "Export nrrd is missing or changed"),
        (lambda items: items["nrrd"].write_bytes(b"other"), "Export nrrd is missing or changed"),
        (lambda items: items["receipt"].write_text("{}"), "Export receipt is missing or changed"),
    ],
)
def test_verification_checks_the_export_itself(tmp_path: Path, damage, error: str) -> None:
    items = _dataset(tmp_path)
    plan = _plan(tmp_path, items)
    record = tmp_path / "cleanup" / "record.json"
    execute_cleanup(plan, record, manifest_identity=MANIFEST)

    damage(items)
    with pytest.raises(ValueError, match=error):
        verify_retained(plan, record, manifest_identity=MANIFEST)


def test_cleanup_belongs_to_the_manifest_that_planned_it(tmp_path: Path) -> None:
    items = _dataset(tmp_path)
    plan = _plan(tmp_path, items)

    with pytest.raises(ValueError, match="planned for a different manifest"):
        execute_cleanup(plan, tmp_path / "cleanup" / "record.json", manifest_identity="other")
    assert retired_payloads(tmp_path)


def test_a_final_off_the_canonical_grid_is_refused(tmp_path: Path) -> None:
    canonical = _fused(tmp_path / "561.zarr")
    shifted = _fused(tmp_path / "514.zarr", origin_z=100.0)

    with pytest.raises(ValueError, match="level 0 .* is not on"):
        require_same_multiscale_grid(shifted, canonical)
    require_same_multiscale_grid(_fused(tmp_path / "488.zarr"), canonical)


def test_workflow_cleanup_waits_for_approval_deletes_nothing_in_a_dry_run_then_becomes_terminal(
    tmp_path: Path,
) -> None:
    items = _dataset(tmp_path)
    paths = hcr_acquisition.HcrPaths(root=tmp_path)
    finals = {label: items[label] for label in ("561", "514")}

    def cleanup(*, dry_run: bool, with_finals: bool = True) -> Path:
        return hcr_workflow._cleanup(
            paths,
            finals if with_finals else {},
            manifest_identity=MANIFEST,
            dry_run=dry_run,
            export=items["nrrd"],
            export_receipt=items["receipt"],
        )

    cleanup(dry_run=True)
    assert retired_payloads(tmp_path)
    with pytest.raises(ReviewRequired, match="Set approved=true to delete them"):
        cleanup(dry_run=False)
    approval = tmp_path / "cleanup-approval.json"
    approval.write_text(json.dumps(json.loads(approval.read_text()) | {"approved": True}))
    cleanup(dry_run=True)
    assert retired_payloads(tmp_path)

    cleanup(dry_run=False)
    assert not retired_payloads(tmp_path)
    # A rerun only verifies the retained finals against the record.
    cleanup(dry_run=False, with_finals=False)
    cleanup(dry_run=True, with_finals=False)
