from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from squisher_lightsheet.mvs_seams import DEFAULT_MVS_SEAM_QUALITY_THRESHOLD, mvs_seam_constraints


DIMENSIONS = ("z", "y", "x")
IDENTITY_AFFINE = {
    "dims": ["x_in", "x_out"],
    "coords": {
        "x_in": ["z", "y", "x", "1"],
        "x_out": ["z", "y", "x", "1"],
    },
    "matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
}


def zyx(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([values[dim] for dim in DIMENSIONS], dtype=np.float64)


def set_zyx(record: dict[str, Any], key: str, values: np.ndarray) -> None:
    record[key] = {dim: float(value) for dim, value in zip(DIMENSIONS, values, strict=True)}


def record_translation_um(record: dict[str, Any]) -> np.ndarray:
    if "translation_um" in record:
        return zyx(record, "translation_um")
    stage = zyx(record, "stage_translation_um")
    affine = np.asarray(record["registered_affine"]["matrix"], dtype=np.float64)
    return stage + affine[:3, 3]


def record_scale_um(record: dict[str, Any]) -> np.ndarray:
    if "scale_um" in record:
        return zyx(record, "scale_um")
    return zyx(record, "stage_scale_um")


def tile_records_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["tile"]): record for record in payload["tiles"]}


def moving_tile_to_reference_tile(tile: str, *, moving_token: str, reference_token: str) -> str:
    return tile.replace(moving_token, reference_token)


def load_direct_constraints(
    tile_phase_summary: dict[str, Any],
    reference_position: dict[str, Any],
    *,
    tile_names: list[str],
    moving_token: str,
    reference_token: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    reference_by_tile = tile_records_by_name(reference_position)
    measurements = {
        str(record["tile"]): record
        for record in tile_phase_summary["measurements"]
        if record.get("measurement_status") == "direct_accepted"
    }
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    tile_shift_px = np.zeros((len(tile_names), 3), dtype=np.float64)
    reference_translation = np.zeros((len(tile_names), 3), dtype=np.float64)
    constraints: list[dict[str, Any]] = []
    missing = sorted(set(tile_names) - set(measurements))
    if missing:
        raise ValueError(f"Missing direct tile-phase measurements for {missing}")

    for tile in tile_names:
        measurement = measurements[tile]
        index = tile_index[tile]
        reference_tile = measurement.get("reference_tile") or moving_tile_to_reference_tile(
            tile,
            moving_token=moving_token,
            reference_token=reference_token,
        )
        if reference_tile not in reference_by_tile:
            raise ValueError(f"Reference position is missing {reference_tile} for {tile}")
        scale = record_scale_um(reference_by_tile[reference_tile])
        reference_translation[index] = record_translation_um(reference_by_tile[reference_tile])
        tile_shift_px[index] = np.asarray(measurement["shift_px_zyx"], dtype=np.float64)
        inlier_patches = [
            patch
            for patch in measurement.get("patch_details", {}).get("patches", [])
            if patch.get("inlier")
        ]
        if not inlier_patches:
            raise ValueError(f"{tile} has no direct inlier patch details")
        for patch in inlier_patches:
            total_shift = np.asarray(patch["total_shift_px_zyx"], dtype=np.float64)
            constraints.append(
                {
                    "tile": tile,
                    "tile_index": index,
                    "patch_index": int(patch["patch_index"]),
                    "target_correction_px": total_shift - tile_shift_px[index],
                    "scale_um": scale,
                    "corr_after": float(patch["corr_after"]),
                    "corr_before": float(patch["corr_before"]),
                    "weight": max(float(patch["corr_after"]) - 0.15, 1e-3),
                }
            )
    return constraints, tile_shift_px, reference_translation


def load_seam_constraints(
    seam_residuals: Path,
    source_registration: dict[str, Any],
    *,
    tile_names: list[str],
    min_corr_after: float,
    max_shift_px: np.ndarray,
) -> list[dict[str, Any]]:
    maybe_json = seam_residuals.read_text()
    if maybe_json.lstrip().startswith("{"):
        try:
            payload = json.loads(maybe_json)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "pairwise_registration" in payload.get("metrics", {}):
            spacing_um = zyx(source_registration, "spacing_um") if "spacing_um" in source_registration else zyx(
                source_registration["tiles"][0],
                "stage_scale_um",
            )
            return mvs_seam_constraints(
                payload,
                tile_names=tile_names,
                spacing_um_zyx=spacing_um,
                min_quality=DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
                used_edges_only=True,
            )

    registration_tiles = [str(record["tile"]) for record in source_registration["tiles"]]
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    constraints: list[dict[str, Any]] = []
    for line in maybe_json.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("accepted"):
            continue
        if float(record.get("correlation_after") or float("nan")) < min_corr_after:
            continue
        fixed_tile = registration_tiles[int(record["fixed"])]
        moving_tile = registration_tiles[int(record["moving"])]
        if fixed_tile not in tile_index or moving_tile not in tile_index:
            continue
        shift = np.asarray(record["shift_zyx"], dtype=np.float64)
        if np.any(np.abs(shift) > max_shift_px):
            continue
        constraints.append(
            {
                "fixed": fixed_tile,
                "moving": moving_tile,
                "fixed_index": tile_index[fixed_tile],
                "moving_index": tile_index[moving_tile],
                "pair": [int(record["fixed"]), int(record["moving"])],
                "axis": record.get("axis"),
                "patch_index": int(record["patch_index"]),
                "target_correction_delta_px": shift,
                "corr_after": float(record["correlation_after"]),
                "corr_before": float(record["correlation_before"]),
                "weight": max(float(record["correlation_after"]) - 0.15, 1e-3),
            }
        )
    return constraints


def seam_pair_key(fixed_tile: str, moving_tile: str) -> tuple[str, str]:
    return (str(fixed_tile), str(moving_tile))


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record in {path} at line {line_number}: {error}") from error
    return records


def _source_patch_records(override: dict[str, Any], *, require_exists: bool) -> list[dict[str, Any]]:
    source_jsonl = override.get("source_jsonl")
    patch_indices = override.get("patch_indices") or []
    if not source_jsonl or not patch_indices:
        return []

    fixed_tile = str(override.get("source_fixed_tile", override["fixed_tile"]))
    moving_tile = str(override.get("source_moving_tile", override["moving_tile"]))
    requested = {int(value) for value in patch_indices}
    selected = [
        record
        for record in _load_jsonl_records(Path(source_jsonl))
        if str(record.get("fixed_tile")) == fixed_tile
        and str(record.get("moving_tile")) == moving_tile
        and int(record.get("patch_index", -1)) in requested
    ]
    found = {int(record["patch_index"]) for record in selected}
    missing = sorted(requested - found)
    if missing and require_exists:
        raise ValueError(
            f"Override {override.get('id', '<unnamed>')} selected missing patches "
            f"for {fixed_tile}->{moving_tile}: {missing}"
        )
    return selected


def _override_shift_px(
    override: dict[str, Any],
    *,
    source_records: list[dict[str, Any]],
) -> np.ndarray:
    if "shift_zyx_px" in override:
        shift = np.asarray(override["shift_zyx_px"], dtype=np.float64)
    else:
        if not source_records:
            raise ValueError(
                f"Override {override.get('id', '<unnamed>')} needs shift_zyx_px "
                "or source_jsonl plus patch_indices"
            )
        shifts = np.asarray([record["shift_zyx"] for record in source_records], dtype=np.float64)
        aggregation = str(override.get("aggregation", "median"))
        if aggregation != "median":
            raise ValueError(f"Unsupported seam override aggregation {aggregation!r}; only 'median' is supported")
        shift = np.median(shifts, axis=0)
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError(f"Override {override.get('id', '<unnamed>')} has invalid shift_zyx_px: {shift.tolist()}")
    return shift


def _component_mask(components: list[str] | None) -> list[bool]:
    if components is None:
        return [True, True, True]
    selected = set(components)
    unknown = selected - set(DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown seam override components: {sorted(unknown)}")
    return [dim in selected for dim in DIMENSIONS]


def apply_direct_overrides(
    direct_constraints: list[dict[str, Any]],
    seam_overrides: Path,
    *,
    tile_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(seam_overrides.read_text())
    tile_set = set(tile_names)
    constraints = list(direct_constraints)
    summary: dict[str, Any] = {
        "path": str(seam_overrides.resolve()),
        "input_constraint_count": len(direct_constraints),
        "applied": [],
    }
    for override in payload.get("overrides", []):
        action = str(override["action"])
        if action != "drop_direct_anchor":
            continue
        tile = str(override["tile"])
        if tile not in tile_set:
            raise ValueError(f"Override {override.get('id', '<unnamed>')} references unknown tile {tile}")
        before_count = len(constraints)
        constraints = [constraint for constraint in constraints if str(constraint["tile"]) != tile]
        summary["applied"].append(
            {
                "id": override.get("id"),
                "action": action,
                "tile": tile,
                "removed_constraints": before_count - len(constraints),
                "reason": override.get("reason"),
            }
        )
    summary["output_constraint_count"] = len(constraints)
    return constraints, summary


def apply_seam_overrides(
    seam_constraints: list[dict[str, Any]],
    seam_overrides: Path,
    *,
    tile_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(seam_overrides.read_text())
    defaults = payload.get("defaults", {})
    require_exists = bool(defaults.get("require_selected_patches_exist", True))
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    constraints = list(seam_constraints)
    summary: dict[str, Any] = {
        "path": str(seam_overrides.resolve()),
        "input_constraint_count": len(seam_constraints),
        "applied": [],
    }

    for override in payload.get("overrides", []):
        action = str(override["action"])
        if action == "drop_direct_anchor":
            continue
        fixed_tile = str(override["fixed_tile"])
        moving_tile = str(override["moving_tile"])
        if fixed_tile not in tile_index or moving_tile not in tile_index:
            raise ValueError(
                f"Override {override.get('id', '<unnamed>')} references a tile outside the position file: "
                f"{fixed_tile}->{moving_tile}"
            )
        pair_key = seam_pair_key(fixed_tile, moving_tile)
        before_count = len(constraints)

        if action == "reject_pair":
            constraints = [
                constraint
                for constraint in constraints
                if seam_pair_key(constraint["fixed"], constraint["moving"]) != pair_key
            ]
            summary["applied"].append(
                {
                    "id": override.get("id"),
                    "action": action,
                    "fixed_tile": fixed_tile,
                    "moving_tile": moving_tile,
                    "removed_constraints": before_count - len(constraints),
                    "reason": override.get("reason"),
                }
            )
            continue

        if action != "force_accept_cluster":
            raise ValueError(f"Unsupported seam override action {action!r}")

        conflict_policy = str(override.get("conflict_policy", defaults.get("conflict_policy", "replace_pair")))
        if conflict_policy != "replace_pair":
            raise ValueError(f"Unsupported seam override conflict_policy {conflict_policy!r}")
        constraints = [
            constraint
            for constraint in constraints
            if seam_pair_key(constraint["fixed"], constraint["moving"]) != pair_key
        ]
        source_records = _source_patch_records(override, require_exists=require_exists)
        shift = _override_shift_px(override, source_records=source_records)
        weight = float(override.get("solver_weight", defaults.get("solver_weight", 1.0)))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"Override {override.get('id', '<unnamed>')} has invalid solver_weight {weight}")
        constraints.append(
            {
                "fixed": fixed_tile,
                "moving": moving_tile,
                "fixed_index": tile_index[fixed_tile],
                "moving_index": tile_index[moving_tile],
                "pair": [tile_index[fixed_tile], tile_index[moving_tile]],
                "axis": override.get("axis"),
                "patch_index": -1,
                "target_correction_delta_px": shift,
                "component_mask": _component_mask(override.get("components")),
                "corr_after": None,
                "corr_before": None,
                "weight": weight,
                "source": "seam_constraint_override",
                "override_id": override.get("id"),
                "source_jsonl": override.get("source_jsonl"),
                "patch_indices": [int(value) for value in override.get("patch_indices", [])],
                "reason": override.get("reason"),
            }
        )
        summary["applied"].append(
            {
                "id": override.get("id"),
                "action": action,
                "fixed_tile": fixed_tile,
                "moving_tile": moving_tile,
                "removed_constraints": before_count - len(constraints) + 1,
                "added_constraints": 1,
                "shift_zyx_px": [float(value) for value in shift],
                "components": override.get("components", list(DIMENSIONS)),
                "solver_weight": weight,
                "source_patch_count": len(source_records),
                "source_patch_indices": [int(record["patch_index"]) for record in source_records],
                "source_patch_shifts_zyx": [
                    [float(value) for value in record["shift_zyx"]] for record in source_records
                ],
                "reason": override.get("reason"),
            }
        )

    summary["output_constraint_count"] = len(constraints)
    return constraints, summary


def load_boundary_constraints(path: Path | None, *, tile_names: list[str]) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    constraints = []
    for record in payload.get("constraints", []):
        tile = str(record["tile"])
        if tile not in tile_index:
            continue
        shift_yx = np.asarray(record["shift_yx_fullres_px"], dtype=np.float64)
        constraints.append(
            {
                "tile": tile,
                "tile_index": tile_index[tile],
                "target_correction_yx_px": shift_yx,
                "weight": max(float(record.get("weight", 1.0)), 1e-3),
                "metadata": record,
            }
        )
    return constraints


def apply_anchor_table_baseline(
    anchor_table: list[dict[str, Any]],
    position_payload: dict[str, Any],
    *,
    direct_anchor_sigma_px: np.ndarray,
    recovered_anchor_sigma_px: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    updated_position = json.loads(json.dumps(position_payload))
    position_by_tile = tile_records_by_name(updated_position)
    tile_names = [str(record["tile"]) for record in updated_position["tiles"]]
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    anchor_by_tile = {str(record["tile"]): record for record in anchor_table}
    missing = sorted(set(tile_names) - set(anchor_by_tile))
    if missing:
        raise ValueError(f"Anchor table is missing tiles from position file: {missing}")

    constraints: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    anchored_tiles: list[str] = []
    for tile in tile_names:
        record = anchor_by_tile[tile]
        source = str(record.get("source") or record.get("status") or "unknown")
        counts[source] += 1
        shift_um = record.get("shift_um_zyx")
        if shift_um is None:
            continue
        shift_um_zyx = np.asarray(shift_um, dtype=np.float64)
        if shift_um_zyx.shape != (3,) or not np.all(np.isfinite(shift_um_zyx)):
            raise ValueError(f"{tile} has invalid shift_um_zyx in anchor table: {shift_um}")

        position_record = position_by_tile[tile]
        set_zyx(
            position_record,
            "translation_um",
            zyx(position_record, "translation_um") + shift_um_zyx,
        )
        if source == "direct_405_to_488_level0_refined":
            sigma_px = direct_anchor_sigma_px
            weight = 1.0
        elif source in {"recovered_from_405_overlap_level0_anchor", "recovered_from_405_mvs_seam_anchor"}:
            sigma_px = recovered_anchor_sigma_px
            weight = 1.0
        else:
            continue
        anchored_tiles.append(tile)
        constraints.append(
            {
                "tile": tile,
                "tile_index": tile_index[tile],
                "patch_index": -1,
                "target_correction_px": np.zeros(3, dtype=np.float64),
                "sigma_px": sigma_px,
                "corr_after": None,
                "corr_before": None,
                "weight": weight,
                "source": source,
                "tile_site": record.get("tile_site"),
            }
        )

    summary = {
        "anchor_table_counts": dict(counts),
        "zero_correction_anchor_count": len(constraints),
        "zero_correction_anchor_tiles": anchored_tiles,
        "direct_anchor_sigma_px": [float(v) for v in direct_anchor_sigma_px],
        "recovered_anchor_sigma_px": [float(v) for v in recovered_anchor_sigma_px],
    }
    return updated_position, constraints, summary


def solve_translation_corrections(
    *,
    n_tiles: int,
    direct_constraints: list[dict[str, Any]],
    boundary_constraints: list[dict[str, Any]],
    seam_constraints: list[dict[str, Any]],
    direct_sigma_px: np.ndarray,
    boundary_sigma_px: np.ndarray,
    seam_sigma_px: np.ndarray,
    prior_sigma_px: np.ndarray,
) -> tuple[np.ndarray, Any]:
    def residual_vector(flat: np.ndarray) -> np.ndarray:
        corrections = flat.reshape(n_tiles, 3)
        residuals: list[np.ndarray] = []
        for constraint in direct_constraints:
            weight = np.sqrt(constraint["weight"])
            sigma_px = constraint.get("sigma_px", direct_sigma_px)
            residuals.append(
                weight
                * (corrections[constraint["tile_index"]] - constraint["target_correction_px"])
                / sigma_px
            )
        for constraint in boundary_constraints:
            weight = np.sqrt(constraint["weight"])
            residuals.append(
                weight
                * (
                    corrections[constraint["tile_index"], 1:3]
                    - constraint["target_correction_yx_px"]
                )
                / boundary_sigma_px
            )
        for constraint in seam_constraints:
            weight = np.sqrt(constraint["weight"])
            mask = np.asarray(constraint.get("component_mask", [True, True, True]), dtype=bool)
            delta = (
                corrections[constraint["moving_index"]]
                - corrections[constraint["fixed_index"]]
                - constraint["target_correction_delta_px"]
            )
            residuals.append(weight * delta[mask] / seam_sigma_px[mask])
        residuals.append((corrections / prior_sigma_px).reshape(-1))
        return np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in residuals])

    result = least_squares(
        residual_vector,
        np.zeros(n_tiles * 3, dtype=np.float64),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=500,
    )
    return result.x.reshape(n_tiles, 3), result


def residual_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    norms = np.linalg.norm(values, axis=1)
    return {
        "count": int(values.shape[0]),
        "median_norm_px": float(np.median(norms)),
        "p95_norm_px": float(np.percentile(norms, 95)),
        "max_norm_px": float(np.max(norms)),
        "median_abs_zyx_px": [float(v) for v in np.median(np.abs(values), axis=0)],
        "max_abs_zyx_px": [float(v) for v in np.max(np.abs(values), axis=0)],
    }


def build_diagnostics(
    *,
    tile_names: list[str],
    corrections_px: np.ndarray,
    direct_constraints: list[dict[str, Any]],
    boundary_constraints: list[dict[str, Any]],
    seam_constraints: list[dict[str, Any]],
    result: Any,
) -> dict[str, Any]:
    direct_residuals = []
    for constraint in direct_constraints:
        residual = corrections_px[constraint["tile_index"]] - constraint["target_correction_px"]
        direct_residuals.append(residual)
    seam_residuals = []
    for constraint in seam_constraints:
        mask = np.asarray(constraint.get("component_mask", [True, True, True]), dtype=bool)
        residual = (
            corrections_px[constraint["moving_index"]]
            - corrections_px[constraint["fixed_index"]]
            - constraint["target_correction_delta_px"]
        )
        residual = np.where(mask, residual, 0.0)
        seam_residuals.append(residual)
    boundary_residuals = []
    for constraint in boundary_constraints:
        residual = corrections_px[constraint["tile_index"], 1:3] - constraint["target_correction_yx_px"]
        boundary_residuals.append(residual)

    per_tile = []
    direct_by_tile: dict[str, list[np.ndarray]] = {tile: [] for tile in tile_names}
    boundary_by_tile: dict[str, list[np.ndarray]] = {tile: [] for tile in tile_names}
    seam_by_tile: dict[str, list[np.ndarray]] = {tile: [] for tile in tile_names}
    for constraint, residual in zip(direct_constraints, direct_residuals, strict=True):
        direct_by_tile[constraint["tile"]].append(residual)
    for constraint, residual in zip(boundary_constraints, boundary_residuals, strict=True):
        boundary_by_tile[constraint["tile"]].append(residual)
    for constraint, residual in zip(seam_constraints, seam_residuals, strict=True):
        seam_by_tile[constraint["fixed"]].append(residual)
        seam_by_tile[constraint["moving"]].append(residual)
    for tile_index, tile in enumerate(tile_names):
        direct = np.asarray(direct_by_tile[tile], dtype=np.float64)
        seam = np.asarray(seam_by_tile[tile], dtype=np.float64)
        boundary = np.asarray(boundary_by_tile[tile], dtype=np.float64)
        per_tile.append(
            {
                "tile": tile,
                "correction_px_zyx": [float(v) for v in corrections_px[tile_index]],
                "direct": residual_stats(direct),
                "boundary_yx": residual_stats(boundary),
                "seam": residual_stats(seam),
            }
        )

    return {
        "optimizer": {
            "method": "scipy.optimize.least_squares",
            "loss": "soft_l1",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
        },
        "constraint_counts": {
            "direct_patches": len(direct_constraints),
            "direct_constraint_sources": dict(
                Counter(str(constraint.get("source", "direct_patch")) for constraint in direct_constraints)
            ),
            "boundary_planes": len(boundary_constraints),
            "seam_patches": len(seam_constraints),
            "seam_pairs": len({tuple(constraint["pair"]) for constraint in seam_constraints}),
        },
        "overall_residuals": {
            "direct": residual_stats(np.asarray(direct_residuals, dtype=np.float64)),
            "boundary_yx": residual_stats(np.asarray(boundary_residuals, dtype=np.float64)),
            "seam": residual_stats(np.asarray(seam_residuals, dtype=np.float64)),
        },
        "per_tile": per_tile,
        "seam_pair_counts": dict(Counter(str(tuple(constraint["pair"])) for constraint in seam_constraints)),
    }


def write_outputs(
    *,
    output_position: Path,
    output_registration: Path,
    diagnostics_path: Path,
    position_payload: dict[str, Any],
    tile_names: list[str],
    corrections_px: np.ndarray,
    spacing_um: np.ndarray,
    diagnostics: dict[str, Any],
    run_inputs: dict[str, str | None],
    direct_constraint_description: str,
) -> None:
    output_position.parent.mkdir(parents=True, exist_ok=True)
    output_registration.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    updated_position = json.loads(json.dumps(position_payload))
    position_by_tile = tile_records_by_name(updated_position)
    for index, tile in enumerate(tile_names):
        record = position_by_tile[tile]
        translation = zyx(record, "translation_um") + corrections_px[index] * spacing_um
        set_zyx(record, "translation_um", translation)
    updated_position["source"] = f"{updated_position.get('source', 'position file')} + constrained 405-to-488 seam optimization"
    updated_position["derived_by"] = "squisher_lightsheet.channel_optimization.optimize_405_to_488_translation_mapping"
    updated_position["transform_contract"] = {
        "coordinate_frame": "488 registered-frame target coordinates",
        "optimized_unknown": "405 per-tile stage translations only",
        "registered_affine": "identity",
        "reference_affines_copied": False,
        "direct_constraints": direct_constraint_description,
        "seam_constraints": "405-to-405 seam continuity constraints",
    }
    updated_position["optimization_diagnostics"] = str(diagnostics_path.resolve())

    registration_records = []
    for tile in tile_names:
        record = position_by_tile[tile]
        registration_records.append(
            {
                "tile": tile,
                "source_view": record.get("side"),
                "path": record.get("path"),
                "stage_translation_um": record["translation_um"],
                "stage_scale_um": record["scale_um"],
                "registered_affine": IDENTITY_AFFINE,
            }
        )
    registration_payload = {
        "input_dir": str(Path(registration_records[0]["path"]).parent),
        "metadata_transform_key": "stage_metadata",
        "registered_transform_key": "translation_optimized_to_488_identity_affine",
        "spacing_um": {dim: float(value) for dim, value in zip(DIMENSIONS, spacing_um, strict=True)},
        "tiles": registration_records,
        "transform_contract": updated_position["transform_contract"],
        "optimization": {
            **run_inputs,
            "diagnostics": str(diagnostics_path.resolve()),
        },
        "metrics": diagnostics["overall_residuals"],
    }

    diagnostics_payload = {
        **diagnostics,
        "artifact_type": "lightsheet.405_to_488_translation_optimization.v1",
        "output_position": str(output_position.resolve()),
        "output_registration": str(output_registration.resolve()),
        "units": "pixel residuals for diagnostics; output translations in micrometers",
    }

    for path, payload in (
        (output_position, updated_position),
        (output_registration, registration_payload),
        (diagnostics_path, diagnostics_payload),
    ):
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
        tmp_path.replace(path)


def _parse_vector(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def optimize_405_to_488_translation_mapping(
    *,
    anchor_table: Path | None = None,
    tile_phase_summary: Path | None = None,
    phase_position: Path,
    reference_position: Path | None = None,
    reference_registration_target: Path | None = None,
    source_405_registration: Path,
    seam_residuals: Path,
    output_position: Path,
    output_registration: Path,
    diagnostics_path: Path,
    seam_overrides: Path | None = None,
    moving_token: str = "405",
    reference_token: str = "488514561638",
    direct_sigma_px: tuple[float, float, float] = (3.0, 12.0, 12.0),
    direct_anchor_sigma_px: tuple[float, float, float] = (0.25, 1.0, 1.0),
    recovered_anchor_sigma_px: tuple[float, float, float] = (1.0, 4.0, 4.0),
    boundary_constraints_path: Path | None = None,
    boundary_sigma_px: tuple[float, float] = (12.0, 12.0),
    seam_sigma_px: tuple[float, float, float] = (2.0, 6.0, 6.0),
    prior_sigma_px: tuple[float, float, float] = (6.0, 24.0, 24.0),
    min_seam_corr_after: float = 0.45,
    max_seam_shift_px: tuple[float, float, float] = (3.0, 12.0, 12.0),
) -> dict[str, Any]:
    if anchor_table is None and (tile_phase_summary is None or reference_position is None):
        raise ValueError("tile_phase_summary and reference_position are required without anchor_table")
    if anchor_table is not None and reference_position is not None:
        raise ValueError("reference_position is not used with anchor_table")

    phase_payload = json.loads(phase_position.read_text())
    source_registration = json.loads(source_405_registration.read_text())
    tile_names = [str(record["tile"]) for record in phase_payload["tiles"]]
    spacing_um = zyx(phase_payload["tiles"][0], "scale_um")
    anchor_summary = None

    if anchor_table is not None:
        anchor_payload = json.loads(anchor_table.read_text())
        if not isinstance(anchor_payload, list):
            raise ValueError(f"Anchor table must be a JSON list: {anchor_table}")
        phase_payload, direct_constraints, anchor_summary = apply_anchor_table_baseline(
            anchor_payload,
            phase_payload,
            direct_anchor_sigma_px=np.asarray(direct_anchor_sigma_px, dtype=np.float64),
            recovered_anchor_sigma_px=np.asarray(recovered_anchor_sigma_px, dtype=np.float64),
        )
        direct_constraint_description = "zero residual correction priors around the merged anchor-table baseline"
    else:
        tile_phase_payload = json.loads(tile_phase_summary.read_text())
        reference_payload = json.loads(reference_position.read_text())
        reference_target = (
            json.loads(reference_registration_target.read_text())
            if reference_registration_target is not None
            else reference_payload
        )
        direct_constraints, _, _ = load_direct_constraints(
            tile_phase_payload,
            reference_target,
            tile_names=tile_names,
            moving_token=moving_token,
            reference_token=reference_token,
        )
        direct_constraint_description = "405 tile to corresponding 488 channel-3 tile"

    seam_constraints = load_seam_constraints(
        seam_residuals,
        source_registration,
        tile_names=tile_names,
        min_corr_after=min_seam_corr_after,
        max_shift_px=np.asarray(max_seam_shift_px, dtype=np.float64),
    )
    seam_override_summary = None
    if seam_overrides is not None:
        direct_constraints, direct_override_summary = apply_direct_overrides(
            direct_constraints,
            seam_overrides,
            tile_names=tile_names,
        )
        seam_constraints, seam_override_summary = apply_seam_overrides(
            seam_constraints,
            seam_overrides,
            tile_names=tile_names,
        )
    boundary_constraints = load_boundary_constraints(boundary_constraints_path, tile_names=tile_names)
    corrections_px, result = solve_translation_corrections(
        n_tiles=len(tile_names),
        direct_constraints=direct_constraints,
        boundary_constraints=boundary_constraints,
        seam_constraints=seam_constraints,
        direct_sigma_px=np.asarray(direct_sigma_px, dtype=np.float64),
        boundary_sigma_px=np.asarray(boundary_sigma_px, dtype=np.float64),
        seam_sigma_px=np.asarray(seam_sigma_px, dtype=np.float64),
        prior_sigma_px=np.asarray(prior_sigma_px, dtype=np.float64),
    )
    diagnostics = build_diagnostics(
        tile_names=tile_names,
        corrections_px=corrections_px,
        direct_constraints=direct_constraints,
        boundary_constraints=boundary_constraints,
        seam_constraints=seam_constraints,
        result=result,
    )
    if anchor_summary is not None:
        diagnostics["anchor_table_baseline"] = anchor_summary
    if seam_override_summary is not None:
        diagnostics["seam_overrides"] = seam_override_summary
        diagnostics["direct_overrides"] = direct_override_summary
    run_inputs = {
        "anchor_table": str(anchor_table.resolve()) if anchor_table is not None else None,
        "tile_phase_summary": str(tile_phase_summary.resolve()) if tile_phase_summary is not None else None,
        "reference_position": str(reference_position.resolve()) if reference_position is not None else None,
        "reference_registration_target": (
            str(reference_registration_target.resolve()) if reference_registration_target is not None else None
        ),
        "seam_residuals": str(seam_residuals.resolve()),
        "seam_overrides": str(seam_overrides.resolve()) if seam_overrides is not None else None,
        "source_405_registration_for_seams": str(source_405_registration.resolve()),
    }
    write_outputs(
        output_position=output_position,
        output_registration=output_registration,
        diagnostics_path=diagnostics_path,
        position_payload=phase_payload,
        tile_names=tile_names,
        corrections_px=corrections_px,
        spacing_um=spacing_um,
        diagnostics=diagnostics,
        run_inputs=run_inputs,
        direct_constraint_description=direct_constraint_description,
    )
    return {
        **diagnostics,
        "output_position": str(output_position.resolve()),
        "output_registration": str(output_registration.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize 405 tile translations into the 488 frame while preserving 405 seam continuity."
    )
    parser.add_argument("--anchor-table", type=Path)
    parser.add_argument("--tile-phase-summary", type=Path)
    parser.add_argument("--phase-position", required=True, type=Path)
    parser.add_argument("--reference-position", type=Path)
    parser.add_argument(
        "--reference-registration-target",
        type=Path,
        help="Optional 488 registration JSON whose stage+affine translations define the target frame.",
    )
    parser.add_argument("--source-405-registration", required=True, type=Path)
    parser.add_argument("--seam-residuals", required=True, type=Path)
    parser.add_argument(
        "--seam-overrides",
        type=Path,
        help="Optional seam override JSON that rejects bad pairs or injects curated pair shifts.",
    )
    parser.add_argument("--output-position", required=True, type=Path)
    parser.add_argument("--output-registration", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--moving-token", default="405")
    parser.add_argument("--reference-token", default="488514561638")
    parser.add_argument("--direct-sigma-px", default="3,12,12")
    parser.add_argument(
        "--direct-anchor-sigma-px",
        default="0.25,1,1",
        help="Zero-correction sigma for direct_405_to_488_level0_refined anchors in anchor-table mode.",
    )
    parser.add_argument(
        "--recovered-anchor-sigma-px",
        default="1,4,4",
        help=(
            "Zero-correction sigma for recovered anchors in anchor-table mode, including "
            "recovered_from_405_mvs_seam_anchor."
        ),
    )
    parser.add_argument("--boundary-constraints", type=Path)
    parser.add_argument("--boundary-sigma-px", default="12,12")
    parser.add_argument("--seam-sigma-px", default="2,6,6")
    parser.add_argument("--prior-sigma-px", default="6,24,24")
    parser.add_argument("--min-seam-corr-after", type=float, default=0.45)
    parser.add_argument(
        "--max-seam-shift-px",
        default="3,12,12",
        help=(
            "Use only seam constraints whose measured correction is within this z/y/x pixel bound. "
            "Larger seam shifts are treated as conflicts with the direct 405-to-488 frame, not as hard constraints."
        ),
    )
    args = parser.parse_args()

    if args.anchor_table is None and (args.tile_phase_summary is None or args.reference_position is None):
        raise SystemExit("--tile-phase-summary and --reference-position are required without --anchor-table")
    if args.anchor_table is not None and args.reference_position is not None:
        raise SystemExit("--reference-position is not used with --anchor-table")

    diagnostics = optimize_405_to_488_translation_mapping(
        anchor_table=args.anchor_table,
        tile_phase_summary=args.tile_phase_summary,
        phase_position=args.phase_position,
        reference_position=args.reference_position,
        reference_registration_target=args.reference_registration_target,
        source_405_registration=args.source_405_registration,
        seam_residuals=args.seam_residuals,
        seam_overrides=args.seam_overrides,
        output_position=args.output_position,
        output_registration=args.output_registration,
        diagnostics_path=args.diagnostics,
        moving_token=args.moving_token,
        reference_token=args.reference_token,
        direct_sigma_px=_parse_vector(args.direct_sigma_px),
        direct_anchor_sigma_px=_parse_vector(args.direct_anchor_sigma_px),
        recovered_anchor_sigma_px=_parse_vector(args.recovered_anchor_sigma_px),
        boundary_constraints_path=args.boundary_constraints,
        boundary_sigma_px=_parse_vector(args.boundary_sigma_px),
        seam_sigma_px=_parse_vector(args.seam_sigma_px),
        prior_sigma_px=_parse_vector(args.prior_sigma_px),
        min_seam_corr_after=args.min_seam_corr_after,
        max_seam_shift_px=_parse_vector(args.max_seam_shift_px),
    )
    print(json.dumps(diagnostics["overall_residuals"], indent=2))
    print(args.output_position.resolve())
    print(args.output_registration.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
