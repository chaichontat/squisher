"""Constrained per-tile finalization for fused-fixed channel registration."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from squisher_lightsheet.artifact_io import sha256_file
from squisher_lightsheet.channel_affine import (
    RegistrationTransformContract,
    _czyx_tile_geometry,
    _finite_array,
    _record_registered_affine_um,
    _tile_record_placement_um,
    isolate_window_channel_affine_um,
)


ARTIFACT_TYPE = "squisher_lightsheet.constrained_tile_channel_affine_registration.v2"
SEAM_ARTIFACT_TYPE = "lightsheet.level0_phase_recovery_measurements.v1"
DIRECT_ANCHOR_ARTIFACT_TYPE = "squisher_lightsheet.registered_grid_channel_anchors.v1"
DEFAULT_GRADIENT_LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _records_by_tile(payload: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("tiles")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must contain a nonempty tiles list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        tile = row.get("tile") if isinstance(row, dict) else None
        if not isinstance(tile, str) or not tile:
            raise ValueError(f"{label} contains a tile without a valid identity")
        if tile in result:
            raise ValueError(f"{label} contains duplicate tile {tile!r}")
        result[tile] = row
    return result


def _source_geometry(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "registered_affine"}


def _pixel_to_world(record: dict[str, Any], point_px_zyx: np.ndarray) -> np.ndarray:
    _shape, scale = _czyx_tile_geometry(record)
    local_um = np.asarray(point_px_zyx, dtype=np.float64) * scale
    return (_tile_record_placement_um(record) @ np.r_[local_um, 1.0])[:3]


def _window_center_px(row: dict[str, Any]) -> np.ndarray:
    start = _finite_array(row.get("moving_start_l0_zyx"), shape=(3,), label="moving window start")
    stop = _finite_array(row.get("moving_stop_l0_zyx"), shape=(3,), label="moving window stop")
    if np.any(stop <= start):
        raise ValueError("Accepted window has an empty moving window")
    return start + (stop - start - 1.0) / 2.0


def _read_window_observations(
    *,
    window_dir: Path,
    reference_by_tile: dict[str, dict[str, Any]],
    baseline_by_tile: dict[str, dict[str, Any]],
    expected_moving_channel: int,
    expected_fixed_fused: Path,
) -> list[dict[str, Any]]:
    reference_by_path = {Path(str(row["path"])).resolve(): row for row in reference_by_tile.values()}
    observations: list[dict[str, Any]] = []
    for path in sorted(window_dir.glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "accepted":
            continue
        if row.get("moving_channel") != expected_moving_channel:
            raise ValueError(f"{path} has moving_channel={row.get('moving_channel')!r}")
        fixed_fused = Path(str(row.get("fixed_fused", "")))
        if not fixed_fused.is_absolute():
            fixed_fused = path.parent / fixed_fused
        if fixed_fused.resolve() != expected_fixed_fused.resolve():
            raise ValueError(f"{path} refers to unexpected fixed fusion {fixed_fused.resolve()}")
        moving_path = Path(str(row.get("moving_path", "")))
        if not moving_path.is_absolute():
            moving_path = path.parent / moving_path
        reference = reference_by_path.get(moving_path.resolve())
        if reference is None:
            raise ValueError(f"{path} moving source is absent from the reference registration")
        tile = str(reference["tile"])
        baseline = baseline_by_tile[tile]
        point_px = _window_center_px(row)
        _shape, scale = _czyx_tile_geometry(reference)
        local_um = point_px * scale
        isolated = isolate_window_channel_affine_um(row=row, reference_record=reference)
        target_world = (_tile_record_placement_um(reference) @ isolated @ np.r_[local_um, 1.0])[:3]
        baseline_world = _pixel_to_world(baseline, point_px)
        attempt = str(row.get("selected_attempt", ""))
        observations.append(
            {
                "tile": tile,
                "window": path.name,
                "depth_band": int(round(float(row["moving_start_l0_zyx"][0]))),
                "baseline_world_um_zyx": baseline_world,
                "target_world_um_zyx": target_world,
                "delta_world_um_zyx": target_world - baseline_world,
                "independent": not attempt.startswith("linear_interpolated_"),
                "selected_attempt": attempt,
            }
        )
    if not observations:
        raise ValueError(f"No accepted windows found in {window_dir}")
    return observations


def _slice_center(row: dict[str, Any], key: str) -> np.ndarray:
    slices = np.asarray(row.get(key), dtype=np.float64)
    if slices.shape != (3, 2) or not np.all(np.isfinite(slices)) or np.any(slices[:, 1] <= slices[:, 0]):
        raise ValueError(f"Seam row has invalid {key}")
    return (slices[:, 0] + slices[:, 1] - 1.0) / 2.0


def _numeric_tile_id(tile: str) -> str:
    name = Path(tile).name
    for suffix in (".ome.zarr", ".ome.tif", ".ome.tiff"):
        name = name.removesuffix(suffix)
    tile_id = name.rsplit(".", 1)[-1]
    if not tile_id.isdigit():
        raise ValueError(f"Tile identity {tile!r} does not end in a numeric ID")
    return tile_id


def _seam_tile_resolver(
    baseline_by_tile: dict[str, dict[str, Any]],
) -> dict[str, str]:
    resolver = {tile: tile for tile in baseline_by_tile}
    by_id: dict[str, list[str]] = defaultdict(list)
    for tile in baseline_by_tile:
        by_id[_numeric_tile_id(tile)].append(tile)
    for tile_id, tiles in by_id.items():
        if len(tiles) != 1:
            raise ValueError(
                f"Baseline registration has ambiguous numeric tile ID {tile_id}: {tiles}"
            )
        resolver[tile_id] = tiles[0]
    return resolver


def _read_seam_constraints(
    path: Path,
    *,
    baseline_by_tile: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("artifact_type") != SEAM_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported seam measurement artifact {payload.get('artifact_type')!r}")
    resolver = _seam_tile_resolver(baseline_by_tile)
    constraints: list[dict[str, Any]] = []
    for index, row in enumerate(payload.get("rows", [])):
        if row.get("status") != "accepted" or row.get("phase_shift_zyx") is None:
            continue
        measured_fixed_tile = str(row.get("fixed_tile"))
        measured_moving_tile = str(row.get("moving_tile"))
        fixed_tile = resolver.get(measured_fixed_tile)
        moving_tile = resolver.get(measured_moving_tile)
        if fixed_tile is None:
            fixed_tile = resolver.get(_numeric_tile_id(measured_fixed_tile))
        if moving_tile is None:
            moving_tile = resolver.get(_numeric_tile_id(measured_moving_tile))
        if fixed_tile is None or moving_tile is None:
            raise ValueError(
                "Seam measurement refers to a tile absent from the baseline registration: "
                f"{measured_fixed_tile!r}, {measured_moving_tile!r}"
            )
        shift = _finite_array(row["phase_shift_zyx"], shape=(3,), label="seam phase shift")
        fixed_px = _slice_center(row, "fixed_slices_zyx")
        moving_px = _slice_center(row, "moving_slices_zyx") - shift
        fixed_world = _pixel_to_world(baseline_by_tile[fixed_tile], fixed_px)
        moving_world = _pixel_to_world(baseline_by_tile[moving_tile], moving_px)
        constraints.append(
            {
                "row": index,
                "fixed_tile": fixed_tile,
                "moving_tile": moving_tile,
                "measured_fixed_tile": measured_fixed_tile,
                "measured_moving_tile": measured_moving_tile,
                "fixed_world_um_zyx": fixed_world,
                "moving_world_um_zyx": moving_world,
                "target_delta_um_zyx": moving_world - fixed_world,
                "z_start": int(row.get("z_start", 0)),
                "z_stop": int(row.get("z_stop", 0)),
                "gradient_ncc": row.get("gradient_component_ncc_phase_mean"),
            }
        )
    if not constraints:
        raise ValueError(f"No accepted phase seam constraints found in {path}")
    return constraints


def _read_direct_anchor_observations(
    path: Path,
    *,
    baseline_by_tile: dict[str, dict[str, Any]],
    expected_baseline: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("artifact_type") != DIRECT_ANCHOR_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported direct-anchor artifact {payload.get('artifact_type')!r}")
    recorded_baseline = Path(str(payload.get("moving_baseline_registration", "")))
    if recorded_baseline.resolve() != expected_baseline.resolve():
        raise ValueError(
            f"Direct anchors use moving baseline {recorded_baseline.resolve()}, "
            f"expected {expected_baseline.resolve()}"
        )
    if payload.get("moving_baseline_sha256") != sha256_file(expected_baseline):
        raise ValueError("Direct-anchor moving baseline hash differs from the current baseline")
    resolver = _seam_tile_resolver(baseline_by_tile)
    observations: list[dict[str, Any]] = []
    for index, row in enumerate(payload.get("rows", [])):
        if row.get("status") != "accepted":
            continue
        measured_tile = str(row.get("moving_tile"))
        tile = resolver.get(measured_tile)
        if tile is None:
            tile = resolver.get(_numeric_tile_id(measured_tile))
        if tile is None:
            raise ValueError(f"Direct anchor row {index} refers to unknown tile {measured_tile!r}")
        center = _finite_array(
            row.get("registered_center_um_zyx"),
            shape=(3,),
            label=f"Direct anchor row {index} registered center",
        )
        shift = _finite_array(
            row.get("shift_to_apply_moving_grid_px_zyx"),
            shape=(3,),
            label=f"Direct anchor row {index} shift",
        )
        grid_spacing = _finite_array(
            row.get("grid_spacing_um_zyx"),
            shape=(3,),
            label=f"Direct anchor row {index} grid spacing",
        )
        if np.any(grid_spacing <= 0):
            raise ValueError(f"Direct anchor row {index} grid spacing must be positive")
        _shape, tile_spacing = _czyx_tile_geometry(baseline_by_tile[tile])
        delta = np.asarray([0.0, shift[1] * grid_spacing[1], shift[2] * grid_spacing[2]])
        observations.append(
            {
                "tile": tile,
                "window": f"direct-anchor-row-{index}",
                "depth_band": int(row["moving_local_center_px_zyx"][0]),
                "baseline_world_um_zyx": center,
                "target_world_um_zyx": center + delta,
                "delta_world_um_zyx": delta,
                "independent": True,
                "selected_attempt": "verified_registered_grid_direct_anchor",
                "spacing_um_zyx": tile_spacing.tolist(),
                "anchor_row": index,
                "uncertainty_grid_px_yx": row.get("uncertainty_grid_px_yx"),
            }
        )
    if not observations:
        raise ValueError(f"No accepted verified direct anchors found in {path}")
    return observations


def _require_connected(tile_names: list[str], constraints: list[dict[str, Any]]) -> None:
    graph: dict[str, set[str]] = {tile: set() for tile in tile_names}
    for row in constraints:
        fixed, moving = row["fixed_tile"], row["moving_tile"]
        graph[fixed].add(moving)
        graph[moving].add(fixed)
    seen = {tile_names[0]}
    pending = [tile_names[0]]
    while pending:
        tile = pending.pop()
        unseen = graph[tile] - seen
        seen.update(unseen)
        pending.extend(unseen)
    missing = sorted(set(tile_names) - seen)
    if missing:
        raise ValueError(f"488 seam measurement graph is disconnected; missing tiles: {missing}")


def _group_balanced_weights(
    groups: Iterable[tuple[Any, ...]],
    *,
    family_budgets: dict[str, float] | None = None,
) -> np.ndarray:
    values = list(groups)
    counts = Counter(values)
    families = sorted({str(group[0]) for group in values})
    budgets = (
        {family: 1.0 / len(families) for family in families}
        if family_budgets is None
        else {family: float(family_budgets[family]) for family in families}
    )
    if any(not np.isfinite(value) or value <= 0 for value in budgets.values()):
        raise ValueError("Every observation-family budget must be finite and positive")
    budget_sum = sum(budgets.values())
    family_group_counts = Counter(str(group[0]) for group in counts)
    objective_weights = np.asarray(
        [
            (budgets[str(group[0])] / budget_sum)
            / (family_group_counts[str(group[0])] * counts[group])
            for group in values
        ],
        dtype=np.float64,
    )
    weights = np.sqrt(objective_weights)
    return weights / np.sqrt(np.mean(weights**2))


def _fit_huber_ridge(
    design: np.ndarray,
    target: np.ndarray,
    base_weights: np.ndarray,
    ridge: np.ndarray,
    *,
    huber_px: float = 2.0,
) -> np.ndarray:
    coefficients = np.zeros((design.shape[1], target.shape[1]), dtype=np.float64)
    for _iteration in range(30):
        predicted = design @ coefficients
        residual = target - predicted
        robust = np.ones_like(residual)
        outliers = np.abs(residual) > huber_px
        robust[outliers] = huber_px / np.abs(residual[outliers])
        updated = np.empty_like(coefficients)
        for axis in range(target.shape[1]):
            weights = base_weights * np.sqrt(robust[:, axis])
            augmented_design = np.vstack((design * weights[:, None], ridge))
            augmented_target = np.r_[target[:, axis] * weights, np.zeros(ridge.shape[0])]
            updated[:, axis] = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) < 1e-8:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def _design_problem(
    *,
    tile_names: list[str],
    observations: list[dict[str, Any]],
    seams: list[dict[str, Any]],
    coordinate_center: np.ndarray,
    coordinate_scale: np.ndarray,
    lateral_scale_um: np.ndarray,
    excluded_depth: int | None,
    depth_bands: list[int],
    family_budgets: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    width = len(tile_names) * 4
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[tuple[Any, ...]] = []
    provenance: list[tuple[str, int]] = []
    for index, row in enumerate(observations):
        if not row["independent"] or row["depth_band"] == excluded_depth:
            continue
        design = np.zeros(width, dtype=np.float64)
        start = tile_index[row["tile"]] * 4
        point = (row["baseline_world_um_zyx"] - coordinate_center) / coordinate_scale
        design[start : start + 4] = np.r_[point, 1.0]
        rows.append(design)
        targets.append(row["delta_world_um_zyx"][1:3] / lateral_scale_um)
        groups.append(("anchor", row["tile"], row["depth_band"]))
        provenance.append(("anchor", index))
    for index, row in enumerate(seams):
        seam_depth = (float(row["z_start"]) + float(row["z_stop"]) - 1.0) / 2.0
        seam_depth_band = min(depth_bands, key=lambda depth: abs(depth - seam_depth))
        if seam_depth_band == excluded_depth:
            continue
        design = np.zeros(width, dtype=np.float64)
        fixed_start = tile_index[row["fixed_tile"]] * 4
        moving_start = tile_index[row["moving_tile"]] * 4
        fixed_point = (row["fixed_world_um_zyx"] - coordinate_center) / coordinate_scale
        moving_point = (row["moving_world_um_zyx"] - coordinate_center) / coordinate_scale
        design[fixed_start : fixed_start + 4] = np.r_[fixed_point, 1.0]
        design[moving_start : moving_start + 4] -= np.r_[moving_point, 1.0]
        rows.append(design)
        targets.append(row["target_delta_um_zyx"][1:3] / lateral_scale_um)
        groups.append(("seam", tuple(sorted((row["fixed_tile"], row["moving_tile"])))))
        provenance.append(("seam", index))
    return (
        np.stack(rows),
        np.stack(targets),
        _group_balanced_weights(groups, family_budgets=family_budgets),
        provenance,
    )


def _anchor_validation_metrics(
    *,
    rows: list[dict[str, Any]],
    tile_names: list[str],
    coefficients: np.ndarray,
    coordinate_center: np.ndarray,
    coordinate_scale: np.ndarray,
    lateral_scale_um: np.ndarray,
) -> dict[str, Any]:
    by_tile: dict[str, list[np.ndarray]] = defaultdict(list)
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    for row in rows:
        start = tile_index[row["tile"]] * 4
        point = (row["baseline_world_um_zyx"] - coordinate_center) / coordinate_scale
        predicted = np.r_[point, 1.0] @ coefficients[start : start + 4]
        measured = row["delta_world_um_zyx"][1:3] / lateral_scale_um
        by_tile[row["tile"]].append(np.abs(measured - predicted))
    tile_p90 = np.stack(
        [np.quantile(np.stack(values), 0.9, axis=0) for values in by_tile.values()]
    )
    p90_yx = np.median(tile_p90, axis=0)
    return {
        "tile_balanced_p90_px_yx": p90_yx.tolist(),
        "selection_score_px": float(np.max(p90_yx)),
        "tile_count": len(by_tile),
        "row_count": len(rows),
    }


def _seam_validation_metrics(
    *,
    rows: list[dict[str, Any]],
    tile_names: list[str],
    coefficients: np.ndarray,
    coordinate_center: np.ndarray,
    coordinate_scale: np.ndarray,
    lateral_scale_um: np.ndarray,
) -> dict[str, Any]:
    if not rows:
        return {
            "before_p90_px_yx": None,
            "after_p90_px_yx": None,
            "row_count": 0,
        }
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    before: list[np.ndarray] = []
    after: list[np.ndarray] = []
    for row in rows:
        fixed = _predict_lateral_correction_px(
            tile=row["fixed_tile"],
            point_world_um_zyx=row["fixed_world_um_zyx"],
            tile_index=tile_index,
            coefficients=coefficients,
            coordinate_center=coordinate_center,
            coordinate_scale=coordinate_scale,
        )
        moving = _predict_lateral_correction_px(
            tile=row["moving_tile"],
            point_world_um_zyx=row["moving_world_um_zyx"],
            tile_index=tile_index,
            coefficients=coefficients,
            coordinate_center=coordinate_center,
            coordinate_scale=coordinate_scale,
        )
        mismatch = row["target_delta_um_zyx"][1:3] / lateral_scale_um
        before.append(np.abs(mismatch))
        after.append(np.abs(mismatch - (fixed - moving)))
    return {
        "before_p90_px_yx": np.quantile(np.stack(before), 0.9, axis=0).tolist(),
        "after_p90_px_yx": np.quantile(np.stack(after), 0.9, axis=0).tolist(),
        "row_count": len(rows),
    }


def _ridge_matrix(tile_count: int, *, gradient_lambda: float, offset_lambda: float) -> np.ndarray:
    diagonal = np.tile([gradient_lambda, gradient_lambda, gradient_lambda, offset_lambda], tile_count)
    return np.diag(diagonal)


def _fit_model(
    *,
    tile_names: list[str],
    observations: list[dict[str, Any]],
    seams: list[dict[str, Any]],
    gradient_lambdas: tuple[float, ...],
    offset_lambda: float,
    anchor_budget: float = 0.5,
    seam_budget: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    direct = [row for row in observations if row["independent"]]
    points = np.stack(
        [row["baseline_world_um_zyx"] for row in direct]
        + [row["fixed_world_um_zyx"] for row in seams]
        + [row["moving_world_um_zyx"] for row in seams]
    )
    coordinate_center = np.median(points, axis=0)
    coordinate_scale = np.ptp(points, axis=0)
    coordinate_scale = np.maximum(coordinate_scale, 1.0)
    if not direct:
        raise ValueError("Constrained affine fitting requires independent window measurements")
    spacings = np.stack(
        [np.asarray(row["spacing_um_zyx"], dtype=np.float64) for row in direct]
    )
    if not np.allclose(spacings, spacings[0], atol=1e-9, rtol=0):
        raise ValueError("Constrained affine fitting requires consistent tile spacing")
    lateral_scale_um = spacings[0, 1:3]
    depths = sorted({int(row["depth_band"]) for row in direct})
    if len(depths) < 3:
        raise ValueError("Constrained affine fitting requires at least three direct depth bands")

    scores: dict[float, list[float]] = defaultdict(list)
    folds_by_lambda: dict[float, list[dict[str, Any]]] = defaultdict(list)
    family_budgets = {"anchor": float(anchor_budget), "seam": float(seam_budget)}
    for gradient_lambda in gradient_lambdas:
        ridge = _ridge_matrix(
            len(tile_names), gradient_lambda=gradient_lambda, offset_lambda=offset_lambda
        )
        for held_depth in depths:
            design, target, weights, _provenance = _design_problem(
                tile_names=tile_names,
                observations=observations,
                seams=seams,
                coordinate_center=coordinate_center,
                coordinate_scale=coordinate_scale,
                lateral_scale_um=lateral_scale_um,
                excluded_depth=held_depth,
                depth_bands=depths,
                family_budgets=family_budgets,
            )
            coefficients = _fit_huber_ridge(design, target, weights, ridge)
            validation = [
                row for row in direct if int(row["depth_band"]) == held_depth
            ]
            held_seams = [
                row
                for row in seams
                if min(
                    depths,
                    key=lambda depth: abs(
                        depth - (float(row["z_start"]) + float(row["z_stop"]) - 1.0) / 2.0
                    ),
                )
                == held_depth
            ]
            anchor_metrics = _anchor_validation_metrics(
                rows=validation,
                tile_names=tile_names,
                coefficients=coefficients,
                coordinate_center=coordinate_center,
                coordinate_scale=coordinate_scale,
                lateral_scale_um=lateral_scale_um,
            )
            seam_metrics = _seam_validation_metrics(
                rows=held_seams,
                tile_names=tile_names,
                coefficients=coefficients,
                coordinate_center=coordinate_center,
                coordinate_scale=coordinate_scale,
                lateral_scale_um=lateral_scale_um,
            )
            if seam_metrics["row_count"]:
                seam_before = np.asarray(seam_metrics["before_p90_px_yx"])
                seam_after = np.asarray(seam_metrics["after_p90_px_yx"])
                seam_gate_passed = bool(np.all(seam_after <= seam_before + 0.25))
            else:
                seam_gate_passed = True
            scores[gradient_lambda].append(float(anchor_metrics["selection_score_px"]))
            folds_by_lambda[gradient_lambda].append(
                {
                    "held_depth_band": held_depth,
                    "anchors": anchor_metrics,
                    "seams": seam_metrics,
                    "seam_nonregression_gate_passed": seam_gate_passed,
                }
            )
    means = {value: float(np.mean(folds)) for value, folds in scores.items()}
    passing = [
        value
        for value in gradient_lambdas
        if all(fold["seam_nonregression_gate_passed"] for fold in folds_by_lambda[value])
    ]
    if not passing:
        raise ValueError("No gradient regularization candidate passes held-seam nonregression")
    best = min(passing, key=means.get)
    uncertainty = float(np.std(scores[best], ddof=1) / np.sqrt(len(scores[best])))
    eligible = [value for value in passing if means[value] <= means[best] + uncertainty]
    selected = max(eligible)
    design, target, weights, provenance = _design_problem(
        tile_names=tile_names,
        observations=observations,
        seams=seams,
        coordinate_center=coordinate_center,
        coordinate_scale=coordinate_scale,
        lateral_scale_um=lateral_scale_um,
        excluded_depth=None,
        depth_bands=depths,
        family_budgets=family_budgets,
    )
    coefficients = _fit_huber_ridge(
        design,
        target,
        weights,
        _ridge_matrix(len(tile_names), gradient_lambda=selected, offset_lambda=offset_lambda),
    )
    diagnostics = {
        "gradient_lambda_candidates": list(gradient_lambdas),
        "held_depth_bands": depths,
        "validation_selection_score_px_by_lambda": {
            str(key): value for key, value in scores.items()
        },
        "validation_mean_selection_score_px_by_lambda": {
            str(key): value for key, value in means.items()
        },
        "validation_folds_by_lambda": {
            str(key): value for key, value in folds_by_lambda.items()
        },
        "selected_gradient_lambda": selected,
        "selection_rule": (
            "strongest regularization within one standard error of minimum tile-balanced "
            "held-anchor axis-p90 score, among candidates passing held-seam axis-p90 "
            "nonregression (+0.25 px tolerance)"
        ),
        "validation_overlap_policy": (
            "entire depth bands are withheld for both anchors and seam segments; configured "
            "anchor blocks are spatially disjoint within each band"
        ),
        "offset_lambda": offset_lambda,
        "objective_family_budgets": family_budgets,
        "coordinate_center_um_zyx": coordinate_center.tolist(),
        "coordinate_scale_um_zyx": coordinate_scale.tolist(),
        "lateral_scale_um_yx": lateral_scale_um.tolist(),
    }
    residual = target - design @ coefficients
    diagnostics["fit_residual_abs_px"] = {
        "median_yx": np.median(np.abs(residual), axis=0).tolist(),
        "p90_yx": np.quantile(np.abs(residual), 0.9, axis=0).tolist(),
    }
    diagnostics["fit_row_provenance"] = [list(value) for value in provenance]
    return coefficients, diagnostics, coordinate_center, coordinate_scale


def _predict_lateral_correction_px(
    *,
    tile: str,
    point_world_um_zyx: np.ndarray,
    tile_index: dict[str, int],
    coefficients: np.ndarray,
    coordinate_center: np.ndarray,
    coordinate_scale: np.ndarray,
) -> np.ndarray:
    start = tile_index[tile] * 4
    point = (point_world_um_zyx - coordinate_center) / coordinate_scale
    return np.r_[point, 1.0] @ coefficients[start : start + 4]


def _write_fit_qc(
    *,
    output_registration: Path,
    tile_names: list[str],
    observations: list[dict[str, Any]],
    seams: list[dict[str, Any]],
    coefficients: np.ndarray,
    coordinate_center: np.ndarray,
    coordinate_scale: np.ndarray,
    lateral_scale_um: np.ndarray,
) -> tuple[Path, Path]:
    """Write exact before/after fit values and a compact review figure."""
    import matplotlib.pyplot as plt

    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    window_rows: list[dict[str, Any]] = []
    for row in observations:
        predicted = _predict_lateral_correction_px(
            tile=row["tile"],
            point_world_um_zyx=row["baseline_world_um_zyx"],
            tile_index=tile_index,
            coefficients=coefficients,
            coordinate_center=coordinate_center,
            coordinate_scale=coordinate_scale,
        )
        measured = row["delta_world_um_zyx"][1:3] / lateral_scale_um
        window_rows.append(
            {
                "tile": row["tile"],
                "window": row["window"],
                "depth_band": row["depth_band"],
                "independent": row["independent"],
                "selected_attempt": row["selected_attempt"],
                "measured_lateral_correction_px_yx": measured.tolist(),
                "fitted_lateral_correction_px_yx": predicted.tolist(),
                "after_fit_residual_px_yx": (measured - predicted).tolist(),
            }
        )

    seam_rows: list[dict[str, Any]] = []
    for row in seams:
        fixed = _predict_lateral_correction_px(
            tile=row["fixed_tile"],
            point_world_um_zyx=row["fixed_world_um_zyx"],
            tile_index=tile_index,
            coefficients=coefficients,
            coordinate_center=coordinate_center,
            coordinate_scale=coordinate_scale,
        )
        moving = _predict_lateral_correction_px(
            tile=row["moving_tile"],
            point_world_um_zyx=row["moving_world_um_zyx"],
            tile_index=tile_index,
            coefficients=coefficients,
            coordinate_center=coordinate_center,
            coordinate_scale=coordinate_scale,
        )
        before = row["target_delta_um_zyx"][1:3] / lateral_scale_um
        after = before - (fixed - moving)
        seam_rows.append(
            {
                "measurement_row": row["row"],
                "fixed_tile": row["fixed_tile"],
                "moving_tile": row["moving_tile"],
                "measured_fixed_tile": row["measured_fixed_tile"],
                "measured_moving_tile": row["measured_moving_tile"],
                "z_start": row["z_start"],
                "z_stop": row["z_stop"],
                "gradient_component_ncc_phase_mean": row["gradient_ncc"],
                "before_fit_mismatch_px_yx": before.tolist(),
                "after_fit_mismatch_px_yx": after.tolist(),
            }
        )

    stem = output_registration.name.removesuffix(".json")
    table_path = output_registration.with_name(f"{stem}.fit-qc.json")
    figure_path = output_registration.with_name(f"{stem}.fit-qc.png")
    table_path.write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.constrained_tile_channel_affine_fit_qc.v1",
                "registration": str(output_registration.resolve()),
                "units": "native moving-tile pixels",
                "window_rows": window_rows,
                "seam_rows": seam_rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    independent = [row for row in window_rows if row["independent"]]
    measured = np.asarray(
        [row["measured_lateral_correction_px_yx"] for row in independent], dtype=np.float64
    )
    fitted = np.asarray(
        [row["fitted_lateral_correction_px_yx"] for row in independent], dtype=np.float64
    )
    depth = np.asarray([row["depth_band"] for row in independent], dtype=np.float64)
    seam_before = np.linalg.norm(
        np.asarray([row["before_fit_mismatch_px_yx"] for row in seam_rows]), axis=1
    )
    seam_after = np.linalg.norm(
        np.asarray([row["after_fit_mismatch_px_yx"] for row in seam_rows]), axis=1
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, label in enumerate(("Y", "X")):
        axes[0, axis].scatter(measured[:, axis], fitted[:, axis], s=5, alpha=0.25)
        limits = [
            float(min(measured[:, axis].min(), fitted[:, axis].min())),
            float(max(measured[:, axis].max(), fitted[:, axis].max())),
        ]
        axes[0, axis].plot(limits, limits, color="black", linewidth=1)
        axes[0, axis].set(
            xlabel=f"measured {label} correction (px)", ylabel="after fitting (px)"
        )
        axes[1, 0].scatter(
            depth,
            measured[:, axis] - fitted[:, axis],
            s=5,
            alpha=0.2,
            label=label,
        )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set(xlabel="native Z start", ylabel="after-fit window residual (px)")
    axes[1, 0].legend()
    axes[1, 1].scatter(seam_before, seam_after, s=8, alpha=0.35)
    maximum = float(max(seam_before.max(initial=0), seam_after.max(initial=0), 1.0))
    axes[1, 1].plot([0, maximum], [0, maximum], color="black", linewidth=1)
    axes[1, 1].set(
        xlabel="488 seam mismatch before fit (px)",
        ylabel="488 seam mismatch after fit (px)",
    )
    figure.suptitle("Constrained tile affine: after-fitting QC")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return table_path.resolve(), figure_path.resolve()


def _fixed_world_residual_affine(
    coefficients_px: np.ndarray,
    *,
    coordinate_center_um: np.ndarray,
    coordinate_scale_um: np.ndarray,
    lateral_scale_um: np.ndarray,
) -> np.ndarray:
    slopes_um = lateral_scale_um[:, None] * coefficients_px[:3].T / coordinate_scale_um[None, :]
    offset_um = lateral_scale_um * coefficients_px[3] - slopes_um @ coordinate_center_um
    affine = np.eye(4, dtype=np.float64)
    affine[1:3, :3] += slopes_um
    affine[1:3, 3] = offset_um
    return affine


def write_constrained_tile_channel_affine_registration(
    *,
    window_dir: Path,
    reference_registration_input: Path,
    baseline_registration_input: Path,
    seam_measurements_input: Path,
    verified_anchor_input: Path,
    output_registration: Path,
    expected_moving_channel: int,
    expected_fixed_fused: Path,
    source_label: str,
    target_label: str,
    gradient_lambdas: tuple[float, ...] = DEFAULT_GRADIENT_LAMBDAS,
    offset_lambda: float = 0.1,
    anchor_budget: float = 0.5,
    seam_budget: float = 0.5,
) -> Path:
    """Fit lateral per-tile affines from absolute windows and measured 488 seams."""
    reference = json.loads(reference_registration_input.read_text())
    baseline = json.loads(baseline_registration_input.read_text())
    reference_by_tile = _records_by_tile(reference, label="reference registration")
    baseline_by_tile = _records_by_tile(baseline, label="baseline registration")
    if reference_by_tile.keys() != baseline_by_tile.keys():
        raise ValueError("Reference and baseline registrations must contain identical tile identities")
    for tile in reference_by_tile:
        if _source_geometry(reference_by_tile[tile]) != _source_geometry(baseline_by_tile[tile]):
            raise ValueError(f"Baseline source geometry differs for tile {tile!r}")
    tile_names = list(reference_by_tile)
    method6_observations = _read_window_observations(
        window_dir=window_dir,
        reference_by_tile=reference_by_tile,
        baseline_by_tile=baseline_by_tile,
        expected_moving_channel=expected_moving_channel,
        expected_fixed_fused=expected_fixed_fused,
    )
    for row in method6_observations:
        _shape, spacing = _czyx_tile_geometry(reference_by_tile[row["tile"]])
        row["spacing_um_zyx"] = spacing.tolist()
    observations = _read_direct_anchor_observations(
        verified_anchor_input,
        baseline_by_tile=baseline_by_tile,
        expected_baseline=baseline_registration_input,
    )
    seams = _read_seam_constraints(seam_measurements_input, baseline_by_tile=baseline_by_tile)
    direct_tiles = {str(row["tile"]) for row in observations if row["independent"]}
    seam_tiles = {
        str(tile)
        for row in seams
        for tile in (row["fixed_tile"], row["moving_tile"])
    }
    unresolved_tiles = [
        tile for tile in tile_names if tile not in direct_tiles and tile not in seam_tiles
    ]
    fit_tile_names = [tile for tile in tile_names if tile not in unresolved_tiles]
    _require_connected(fit_tile_names, seams)
    coefficients, fit_diagnostics, center, scale = _fit_model(
        tile_names=fit_tile_names,
        observations=observations,
        seams=seams,
        gradient_lambdas=gradient_lambdas,
        offset_lambda=offset_lambda,
        anchor_budget=anchor_budget,
        seam_budget=seam_budget,
    )

    adapted = copy.deepcopy(baseline)
    tile_diagnostics: dict[str, Any] = {}
    seam_degree = Counter(
        tile
        for row in seams
        for tile in (str(row["fixed_tile"]), str(row["moving_tile"]))
    )
    fit_tile_index = {tile: index for index, tile in enumerate(fit_tile_names)}
    for record in adapted["tiles"]:
        tile = str(record["tile"])
        _shape, spacing = _czyx_tile_geometry(reference_by_tile[tile])
        baseline_affine = _record_registered_affine_um(record)
        if tile in unresolved_tiles:
            block = None
            residual_affine = np.eye(4, dtype=np.float64)
            updated_affine = baseline_affine
        else:
            index = fit_tile_index[tile]
            block = coefficients[index * 4 : (index + 1) * 4]
            residual_affine = _fixed_world_residual_affine(
                block,
                coordinate_center_um=center,
                coordinate_scale_um=scale,
                lateral_scale_um=spacing[1:3],
            )
            updated_affine = residual_affine @ baseline_affine
        if not np.allclose(updated_affine[0], baseline_affine[0], atol=1e-12, rtol=0):
            raise RuntimeError(f"Constrained lateral fit changed the fixed-world Z row for {tile!r}")
        record["registered_affine"]["matrix"] = updated_affine.tolist()
        direct_count = sum(row["tile"] == tile and row["independent"] for row in observations)
        if tile in unresolved_tiles:
            support = "baseline_only_unresolved"
        elif direct_count:
            support = "direct_windows_and_measured_seams"
        else:
            support = "measured_seams_only"
        tile_diagnostics[tile] = {
            "verified_direct_anchor_count": direct_count,
            "predicted_window_count": sum(
                row["tile"] == tile and not row["independent"]
                for row in method6_observations
            ),
            "supported_by_direct_windows": direct_count > 0,
            "measured_seam_constraint_count": seam_degree[tile],
            "support": support,
            "normalized_lateral_coefficients_px": None if block is None else block.tolist(),
            "fixed_world_residual_affine_zyx": residual_affine.tolist(),
        }

    output_registration.parent.mkdir(parents=True, exist_ok=True)
    lateral_scale_um = np.asarray(fit_diagnostics["lateral_scale_um_yx"], dtype=np.float64)
    fit_qc_table, fit_qc_figure = _write_fit_qc(
        output_registration=output_registration,
        tile_names=fit_tile_names,
        observations=[row for row in observations if row["tile"] in fit_tile_index],
        seams=seams,
        coefficients=coefficients,
        coordinate_center=center,
        coordinate_scale=scale,
        lateral_scale_um=lateral_scale_um,
    )

    adapted["artifact_type"] = ARTIFACT_TYPE
    adapted["adapted_from"] = str(baseline_registration_input.resolve())
    adapted["adaptation_method"] = "joint_huber_tile_lateral_affine_with_measured_488_seams"
    adapted["transform_contract"] = RegistrationTransformContract(
        registered_affine_semantics="moving_tile_stage_um_to_reference_registered_um",
        source_space=f"{source_label}_stage_um",
        target_space=f"{target_label}_registered_um",
        composition_order=("fixed_world_residual_affine", "baseline_registered_affine"),
        registered_affine_contains_full_channel_affine=True,
        stage_translation_source="reference_registration_input",
    ).model_dump(mode="json", by_alias=True)
    adapted.setdefault("diagnostics", {})["constrained_tile_channel_affine"] = {
        "status": "awaiting_human_qc",
        "reference_registration": str(reference_registration_input.resolve()),
        "reference_registration_sha256": sha256_file(reference_registration_input),
        "baseline_registration": str(baseline_registration_input.resolve()),
        "baseline_registration_sha256": sha256_file(baseline_registration_input),
        "window_dir": str(window_dir.resolve()),
        "method6_role": "initializer_candidate_and_provenance_only",
        "verified_anchor_input": str(verified_anchor_input.resolve()),
        "verified_anchor_sha256": sha256_file(verified_anchor_input),
        "seam_measurements": str(seam_measurements_input.resolve()),
        "seam_measurements_sha256": sha256_file(seam_measurements_input),
        "seam_shift_convention": (
            "phase_shift_zyx is the array-coordinate shift applied to moving; "
            "the corresponding moving point is moving_slice_center - phase_shift_zyx"
        ),
        "fixed_fused": str(expected_fixed_fused.resolve()),
        "z_policy": "fixed-world registered-affine Z row copied exactly from baseline",
        "verified_direct_anchor_count": len(observations),
        "method6_direct_window_count": sum(
            row["independent"] for row in method6_observations
        ),
        "predicted_window_count": sum(
            not row["independent"] for row in method6_observations
        ),
        "seam_constraint_count": len(seams),
        "fit_tile_count": len(fit_tile_names),
        "unresolved_baseline_only_tiles": unresolved_tiles,
        "fit_qc_table": str(fit_qc_table),
        "fit_qc_figure": str(fit_qc_figure),
        "fit": fit_diagnostics,
        "tiles": tile_diagnostics,
    }
    temporary = output_registration.with_name(f".{output_registration.name}.tmp")
    temporary.write_text(json.dumps(adapted, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_registration)
    return output_registration.resolve()
