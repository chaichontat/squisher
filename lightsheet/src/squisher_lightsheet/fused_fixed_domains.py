from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from squisher_lightsheet._backends.fused_fixed_trial import (
    _global_index_affine_to_local_model,
    _local_model_to_global_index_affine,
)


def _record(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain an object")
    return value


def _bounds(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3 or not all(type(v) is int for v in value):
        raise ValueError(f"{name} must contain three integer ZYX coordinates")
    return np.asarray(value, dtype=np.int64)


def _path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must contain a path string")
    return Path(value).resolve()


def _intersects_fixed_roi(row: dict[str, object], start: np.ndarray, stop: np.ndarray) -> bool:
    pull = np.asarray(row["selected_fixed_fused_l0_to_moving_l0_pull_matrix_zyx"])
    offset = np.asarray(row["selected_fixed_fused_l0_to_moving_l0_pull_offset_zyx"])
    source_start = _bounds(row["moving_start_l0_zyx"], name="domain start")
    source_stop = _bounds(row["moving_stop_l0_zyx"], name="domain stop")
    result = linprog(
        np.zeros(3),
        A_ub=np.vstack((pull, -pull)),
        b_ub=np.r_[source_stop - 1 - offset, offset - source_start],
        bounds=list(zip(start, stop - 1, strict=True)),
        method="highs",
    )
    if result.status not in (0, 2):
        raise ValueError(f"domain intersection could not be established: {result.message}")
    return result.success


def rebase_domain(
    transform: dict[str, object],
    *,
    start: np.ndarray,
    stop: np.ndarray,
) -> dict[str, object]:
    """Retain a source-to-fixed mapping while changing its sampling domain."""
    original_start = _bounds(transform["moving_start_l0_zyx"], name="transform start")
    original_stop = _bounds(transform["moving_stop_l0_zyx"], name="transform stop")
    if np.any(start < original_start) or np.any(stop > original_stop) or np.any(stop <= start):
        raise ValueError("replacement domain lies outside its transform's recorded source domain")
    original_shape = _bounds(transform["moving_shape_zyx"], name="transform shape")
    if not np.array_equal(original_shape, original_stop - original_start):
        raise ValueError("transform source bounds and moving_shape_zyx disagree")
    matrix = np.asarray(transform["selected_local_matrix_zyx"], dtype=np.float64)
    translation = np.asarray(transform["selected_local_translation_zyx"], dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or translation.shape != (3,)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(translation))
    ):
        raise ValueError("transform requires a finite 3D affine")
    global_matrix, global_offset = _local_model_to_global_index_affine(
        matrix=matrix,
        translation=translation,
        fixed_start_zyx=_bounds(transform["fixed_start_zyx"], name="transform fixed start"),
        moving_start_zyx=original_start,
        crop_shape_zyx=original_shape,
    )
    shape = stop - start
    center = (shape - 1) / 2
    fixed_start = np.rint(global_matrix @ (start + center) + global_offset - center).astype(np.int64)
    local_matrix, local_translation = _global_index_affine_to_local_model(
        matrix=global_matrix,
        offset=global_offset,
        fixed_start_zyx=fixed_start,
        moving_start_zyx=start,
        crop_shape_zyx=shape,
    )
    pull = np.linalg.inv(global_matrix)
    row = {
        key: transform[key]
        for key in (
            "moving_tile",
            "moving_path",
            "moving_channel",
            "fixed_fused",
            "fused_scale_zyx",
            "fused_translation_zyx",
        )
    }
    row["schema_version"] = 1
    row["status"] = "accepted"
    row.update(
        {
            "artifact_type": "lightsheet.fused_fixed_explicit_domain.v1",
            "moving_start_l0_zyx": start.tolist(),
            "moving_stop_l0_zyx": stop.tolist(),
            "moving_shape_zyx": shape.tolist(),
            "fixed_start_zyx": fixed_start.tolist(),
            "fixed_stop_zyx": (fixed_start + shape).tolist(),
            "selected_local_matrix_zyx": local_matrix.tolist(),
            "selected_local_translation_zyx": local_translation.tolist(),
            "selected_moving_l0_to_fixed_fused_l0_matrix_zyx": global_matrix.tolist(),
            "selected_moving_l0_to_fixed_fused_l0_offset_zyx": global_offset.tolist(),
            "selected_fixed_fused_l0_to_moving_l0_pull_matrix_zyx": pull.tolist(),
            "selected_fixed_fused_l0_to_moving_l0_pull_offset_zyx": (-pull @ global_offset).tolist(),
        }
    )
    return row


def assemble_bounded_domains(*, plan_path: Path, output_dir: Path) -> Path:
    plan = _record(plan_path)
    if plan.get("artifact_type") != "lightsheet.fused_fixed_domain_replacement_plan.v1":
        raise ValueError("unsupported explicit-domain plan")
    domains = plan.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("domain plan requires a nonempty domains list")
    fixed_fused = _path(plan["fixed_fused"], name="fixed_fused")
    moving_channel = plan["moving_channel"]
    if type(moving_channel) is not int or moving_channel < 0:
        raise ValueError("domain plan requires a nonnegative acquisition source channel")
    roi_start = _bounds(plan["fixed_roi_start_zyx"], name="fixed ROI start")
    roi_stop = _bounds(plan["fixed_roi_stop_zyx"], name="fixed ROI stop")
    if np.any(roi_start < 0) or np.any(roi_stop <= roi_start):
        raise ValueError("bounded fixed ROI must be positive and ordered")
    protected_start = _bounds(plan["protected_fixed_start_zyx"], name="protected start")
    protected_stop = _bounds(plan["protected_fixed_stop_zyx"], name="protected stop")
    if (
        np.any(protected_start < roi_start)
        or np.any(protected_stop > roi_stop)
        or np.any(protected_stop <= protected_start)
    ):
        raise ValueError("protected field must be inside the bounded fusion ROI")
    replaced_parents: set[Path] = set()
    rows: list[dict[str, object]] = []
    coverage: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    identities: set[tuple[str, tuple[int, ...], tuple[int, ...]]] = set()
    for item in domains:
        if not isinstance(item, dict):
            raise ValueError("domain records must be objects")
        parent_path = _path(item["parent_window_json"], name="parent_window_json")
        transform_path = _path(item["transform_record_json"], name="transform_record_json")
        quadrant = item["quadrant"]
        if not isinstance(quadrant, str) or re.fullmatch(r"qy[0-9]+_qx[0-9]+", quadrant) is None:
            raise ValueError("domain quadrant must identify its published acquisition quadrant")
        policy = item["domain_policy"]
        if policy not in {"retained_published_mapping", "validated_refinement"}:
            raise ValueError(f"unsupported domain policy {policy!r}")
        parent, transform = _record(parent_path), _record(transform_path)
        for record_path, record in ((parent_path, parent), (transform_path, transform)):
            if record.get("status") != "accepted":
                raise ValueError(f"domain input {record_path} is not accepted")
            if (
                _path(record["fixed_fused"], name="record fixed_fused") != fixed_fused
                or record["moving_channel"] != moving_channel
            ):
                raise ValueError(f"domain input {record_path} reference/channel mismatch")
        if (
            parent["moving_tile"] != transform["moving_tile"]
            or parent["moving_path"] != transform["moving_path"]
        ):
            raise ValueError("parent and refinement source identities differ")
        if policy == "retained_published_mapping" and transform_path != parent_path:
            raise ValueError("retained mapping must use its published parent transform")
        if policy == "validated_refinement":
            replaced_parents.add(parent_path)
            cache = transform.get("cache_config")
            if (
                not isinstance(cache, dict)
                or cache.get("local_3d_refinement") is not True
                or transform.get("selected_attempt") != "local_3d_affine_refinement"
            ):
                raise ValueError("refinement must have passed the local 3D owning criterion")
        start = _bounds(item["source_start_zyx"], name="domain source start")
        stop = _bounds(item["source_stop_zyx"], name="domain source stop")
        parent_start = _bounds(parent["moving_start_l0_zyx"], name="parent source start")
        parent_stop = _bounds(parent["moving_stop_l0_zyx"], name="parent source stop")
        if np.any(start < parent_start) or np.any(stop > parent_stop) or np.any(stop <= start):
            raise ValueError("replacement domain is outside its parent")
        if not np.array_equal(start[1:], parent_start[1:]) or not np.array_equal(stop[1:], parent_stop[1:]):
            raise ValueError("bounded replacement supports axial domains with unchanged XY bounds")
        identity = (str(parent["moving_tile"]), tuple(start.tolist()), tuple(stop.tolist()))
        if identity in identities:
            raise ValueError("duplicate source-domain identity")
        identities.add(identity)
        if parent_path not in coverage:
            coverage[parent_path] = (
                parent_start,
                np.zeros(int(parent_stop[0] - parent_start[0]), dtype=bool),
            )
        covered_start, covered = coverage[parent_path]
        covered[int(start[0] - covered_start[0]) : int(stop[0] - covered_start[0])] = True
        row = rebase_domain(transform, start=start, stop=stop)
        row.update(
            {
                "parent_window_json": str(parent_path),
                "transform_record_json": str(transform_path),
                "domain_policy": policy,
                "quadrant": quadrant,
                "domain_acceptance": "published mapping retained without refitting"
                if policy == "retained_published_mapping"
                else "owning local 3D refinement accepted",
            }
        )
        rows.append(row)
    if any(not covered.all() for _, covered in coverage.values()):
        raise ValueError("replacement domains leave a gap in a parent source domain")
    for row in rows:
        if (
            row["domain_policy"] == "retained_published_mapping"
            and _path(row["parent_window_json"], name="row parent") in replaced_parents
            and _intersects_fixed_roi(row, protected_start, protected_stop)
        ):
            raise ValueError("a retained superseded parent would contribute inside the protected field")
    output_dir.mkdir(parents=True, exist_ok=True)
    window_dir = output_dir / "window_json"
    window_dir.mkdir(exist_ok=True)
    compact = []
    for row in rows:
        start = _bounds(row["moving_start_l0_zyx"], name="row start")
        stop = _bounds(row["moving_stop_l0_zyx"], name="row stop")
        stem = str(row["moving_tile"]).removesuffix(".ome.zarr")
        # Bounds identify domains; quadrant is retained acquisition provenance.
        domain_id = "_".join(str(v) for v in [*start, *stop])
        path = window_dir / f"{stem}.domain_{domain_id}.json"
        path.write_text(json.dumps(row, indent=2) + "\n")
        compact.append(
            {
                "level0_json": str(path.resolve()),
                "quadrant": row["quadrant"],
                "status": "accepted",
                "domain_policy": row["domain_policy"],
            }
        )
    summary = {
        "artifact_type": "lightsheet.fused_fixed_explicit_domain_summary.v1",
        "scope": "bounded_validation",
        "domain_plan": str(plan_path.resolve()),
        "fixed_fused": str(fixed_fused),
        "moving_channel": moving_channel,
        "fixed_roi_start_zyx": roi_start.tolist(),
        "fixed_roi_stop_zyx": roi_stop.tolist(),
        "protected_fixed_start_zyx": protected_start.tolist(),
        "protected_fixed_stop_zyx": protected_stop.tolist(),
        "superseded_parent_support_excluded_from_protected_field": True,
        "parent_source_coverage_complete": True,
        "windows": compact,
    }
    summary_path = output_dir / "fused_fixed_explicit_domain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path.resolve()
