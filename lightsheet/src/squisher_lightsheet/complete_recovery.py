"""Publish the complete fused-fixed recovery policy."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np

from squisher_lightsheet import fused_fixed_recovery as recovery
from squisher_lightsheet.artifact_io import read_json_object
from squisher_lightsheet.fused_fixed import SUMMARY_NAME


PRODUCER_VERSION = "squisher-lightsheet.complete-recovery.4"
POLICY_ORDER = (
    "physical_linear_filter",
    "retained_affine_outlier_smoothing",
    "spatial_fallback_promotion",
    "linear_mapping_recovery",
    "manual_seed_arbitration",
)
SPATIAL_TARGET_REASON = "native_rerun_rejected"
LINEAR_TARGET_REASON = "linear_mapping_outlier"
NATIVE_RERUN_QUALITY_GATE_REASONS = frozenset(("quality_gate_grad_ncc", "quality_gate_corr"))
INTERPOLATED_BASELINE_ATTEMPT = "level2_initializer_only"
CONTRADICTED_INTERPOLATION_REASON = "spatial_fallback_contradicted_by_image"
COMPLETE_NATIVE_METHODS = frozenset(("method6", "method10-mattes"))


@dataclass(frozen=True)
class CompleteRecoverySettings:
    minimum_linear_samples: int = recovery.DEFAULT_MINIMUM_LINEAR_SAMPLES
    minimum_smoothing_samples: int = recovery.DEFAULT_MINIMUM_SMOOTHING_SAMPLES
    linear_loss_scale_um: float = recovery.DEFAULT_LINEAR_LOSS_SCALE_UM
    outlier_mad: float = recovery.DEFAULT_OUTLIER_MAD
    minimum_linear_outlier_um: float = recovery.DEFAULT_MINIMUM_LINEAR_OUTLIER_UM
    maximum_linear_outlier_um: float = recovery.DEFAULT_MAXIMUM_LINEAR_OUTLIER_UM
    maximum_linear_median_residual_um: float = (
        recovery.DEFAULT_MAXIMUM_LINEAR_MEDIAN_RESIDUAL_UM
    )
    maximum_outlier_px: float = recovery.DEFAULT_MAXIMUM_OUTLIER_PX
    replacement_image_ncc_margin: float = recovery.DEFAULT_REPLACEMENT_IMAGE_NCC_MARGIN


DEFAULT_COMPLETE_RECOVERY_SETTINGS = CompleteRecoverySettings()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_dataset_identity(path: Path) -> dict[str, Any]:
    """Hash metadata and provenance without claiming to hash image chunks."""
    candidates = (
        path / "provenance" / "manifest.json",
        path / "zarr.json",
        path / ".zattrs",
        path / ".zgroup",
    )
    return {
        "path": str(path.resolve()),
        "metadata_sha256": {
            str(candidate.relative_to(path)): _sha256(candidate)
            for candidate in candidates
            if candidate.is_file()
        },
        "pixel_content_identity": None,
    }


def _input_identity(
    summary_path: Path, settings: CompleteRecoverySettings
) -> dict[str, Any]:
    summary = read_json_object(summary_path)
    native = summary.get("outlier_recovery")
    if not isinstance(native, dict) or native.get("artifact_type") != (
        "lightsheet.fused_fixed_native_spatial_outlier_recovery.v1"
    ):
        raise ValueError(f"Native recovery checkpoint is not complete: {summary_path}")
    native_method = summary.get("cache_config", {}).get("native_method")
    if native_method not in COMPLETE_NATIVE_METHODS:
        raise ValueError(
            f"Complete recovery requires native Method 6 or Method 10 Mattes, got {native_method!r}"
        )
    compact_windows = summary.get("windows")
    if not isinstance(compact_windows, list):
        raise ValueError(f"Native recovery checkpoint has no windows: {summary_path}")
    aggregate_count = summary.get("aggregate", {}).get("window_count")
    if aggregate_count != len(compact_windows):
        raise ValueError(
            f"Native recovery window count mismatch: {aggregate_count} != "
            f"{len(compact_windows)}"
        )

    windows = []
    names: set[str] = set()
    for compact in compact_windows:
        path = Path(compact["level0_json"]).resolve()
        if path.name in names:
            raise ValueError(f"Duplicate recovery window filename: {path.name}")
        names.add(path.name)
        windows.append({"name": path.name, "path": str(path), "sha256": _sha256(path)})

    moving_position = Path(summary["moving_position"]).resolve()
    fixed_fused = Path(summary["fixed_fused"]).resolve()
    payload = {
        "producer_version": PRODUCER_VERSION,
        "producer_files_sha256": {
            Path(__file__).name: _sha256(Path(__file__)),
            Path(recovery.__file__).name: _sha256(Path(recovery.__file__)),
        },
        "policy_order": list(POLICY_ORDER),
        "settings": asdict(settings),
        "native_summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
        },
        "window_records": windows,
        "moving_position": {
            "path": str(moving_position),
            "sha256": _sha256(moving_position),
        },
        "fixed_dataset": _fixed_dataset_identity(fixed_fused),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"digest": hashlib.sha256(encoded).hexdigest(), **payload}


def _attempt_transform(attempt: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(attempt["local_matrix_zyx"], dtype=np.float64).ravel(),
            np.asarray(attempt["local_translation_zyx"], dtype=np.float64),
        ]
    )


def _quality_gate_rejected_native_rerun(
    row: dict[str, Any], *, native_method: str, interpolated: np.ndarray
) -> dict[str, Any] | None:
    """Return the native rerun when it failed only the correlation or gradient-NCC gate."""
    reason = (row.get("outlier_recovery") or {}).get("native_rerun_rejection_reason")
    if reason not in NATIVE_RERUN_QUALITY_GATE_REASONS:
        return None
    attempts = row.get("method8_attempts") or []
    prefix = recovery._native_attempt_prefix(native_method)
    native = [attempt for attempt in attempts if str(attempt["name"]).startswith(prefix)]
    baseline = [
        attempt for attempt in attempts if attempt["name"] == INTERPOLATED_BASELINE_ATTEMPT
    ]
    if len(native) != 1 or len(baseline) != 1:
        raise ValueError(
            f"Expected one native rerun and one {INTERPOLATED_BASELINE_ATTEMPT} attempt "
            f"for {row.get('moving_tile')}"
        )
    if native[0]["rejection_reason"] != reason:
        raise ValueError(f"Native rerun rejection mismatch for {row.get('moving_tile')}")
    if not np.allclose(_attempt_transform(baseline[0]), interpolated, atol=1e-4):
        raise ValueError(
            f"{INTERPOLATED_BASELINE_ATTEMPT} does not match the interpolated initializer "
            f"for {row.get('moving_tile')}"
        )
    transform = _attempt_transform(native[0])
    if not recovery._affine_is_plausible(transform):
        raise ValueError(f"Quality-gate rejection with implausible affine for {row.get('moving_tile')}")
    return {"attempt": native[0], "transform": transform}


def _promote_spatial_fallback(
    row: dict[str, Any],
    *,
    native_method: str,
    scorer: Any,
    image_ncc_margin: float,
) -> dict[str, Any]:
    """Promote the image-favored candidate unless the image contradicts it.

    The interpolated initializer is the default. A quality-gate-rejected native
    rerun replaces it only when every informative plane favors the rerun. The
    selected transform is then checked for confident residual misalignment; a
    contradicted window stays rejected instead of being published misaligned.
    """
    provenance = row.get("outlier_recovery") or {}
    matrix = np.asarray(
        provenance.get("interpolated_initializer_local_matrix_zyx"), dtype=np.float64
    )
    translation = np.asarray(
        provenance.get("interpolated_initializer_local_translation_zyx"),
        dtype=np.float64,
    )
    source_windows = provenance.get("source_good_window_json") or []
    if matrix.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"Invalid interpolated transform for {row.get('moving_tile')}")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(translation)):
        raise ValueError(f"Non-finite interpolated transform for {row.get('moving_tile')}")
    initializer_source = provenance.get("initializer_source", "interpolated")
    if len(source_windows) < 2 and initializer_source != "manual":
        raise ValueError(f"Insufficient interpolation support for {row.get('moving_tile')}")
    interpolated = np.concatenate([matrix.ravel(), translation])
    if not recovery._affine_is_plausible(interpolated):
        raise ValueError(f"Implausible interpolated transform for {row.get('moving_tile')}")

    rerun = _quality_gate_rejected_native_rerun(
        row, native_method=native_method, interpolated=interpolated
    )
    comparison = None
    select_rerun = False
    if rerun is not None:
        interpolated_planes, rerun_planes = scorer.plane_scores(
            row, [interpolated, rerun["transform"]]
        )
        select_rerun = recovery.image_prefers_replacement(
            interpolated_planes, rerun_planes, image_ncc_margin
        )
        comparison = {
            "interpolated_image_ncc_planes": interpolated_planes,
            "native_rerun_image_ncc_planes": rerun_planes,
            "replacement_image_ncc_margin": image_ncc_margin,
        }
    selected = rerun["transform"] if rerun is not None and select_rerun else interpolated
    residual = recovery.residual_verdict(scorer.residual_shifts(row, selected))
    promotion = {
        "artifact_type": "lightsheet.interpolated_spatial_fallback_promotion.v2",
        "policy": (
            "promote the interpolated initializer, or the quality-gate-rejected native "
            "rerun when every informative plane favors it, unless confident residual "
            "misalignment contradicts the selected transform"
        ),
        "previous_status": row["status"],
        "previous_rejection_reason": row["rejection_reason"],
        "native_rerun_rejection_reason": provenance.get("native_rerun_rejection_reason"),
        "source_good_window_count": len(source_windows),
        "initializer_source": initializer_source,
        "selected_candidate": "native_rerun" if select_rerun else "interpolated",
        "candidate_comparison": comparison,
        "residual": residual,
    }
    promoted = deepcopy(row)
    if residual["verdict"] == "contradicted":
        promoted.update(
            {"status": "rejected", "rejection_reason": CONTRADICTED_INTERPOLATION_REASON}
        )
        promoted["interpolated_fallback_promotion"] = {**promotion, "selected_attempt": None}
        return promoted
    if select_rerun:
        assert rerun is not None
        attempt = str(rerun["attempt"]["name"])
        recovery._set_selected_transform(
            promoted,
            matrix=rerun["transform"][:9].reshape(3, 3),
            translation=rerun["transform"][9:],
            attempt=attempt,
        )
        promoted.update(
            {
                "selected_corr_refined": rerun["attempt"].get("corr_refined"),
                "selected_gradient_component_ncc_mean": rerun["attempt"].get(
                    "gradient_component_ncc_mean"
                ),
                "selected_gradient_component_ncc_refined": rerun["attempt"].get(
                    "gradient_component_ncc_refined"
                ),
            }
        )
    else:
        attempt = recovery._interpolated_recovery_attempt(native_method)
        recovery._set_selected_transform(
            promoted, matrix=matrix, translation=translation, attempt=attempt
        )
    promoted.update({"status": "accepted", "rejection_reason": None})
    promoted["interpolated_fallback_promotion"] = {**promotion, "selected_attempt": attempt}
    return promoted


def _arbitrate_manual_seed(
    row: dict[str, Any], *, native_method: str, scorer: Any
) -> dict[str, Any]:
    """Publish the curated seed when the image verifies it but not the accepted fit.

    An accepted window with a manual initializer carries either the native refit
    from that seed or the retained original fit. Both can land away from a seed
    the residual check verifies; the seed then replaces the fit.
    """
    initializer_path = Path(row["outlier_recovery"]["initializer_json"])
    initializer = read_json_object(initializer_path)
    if initializer["recovery"].get("initializer_source") != "manual" or any(
        initializer[key] != row[key] for key in ("moving_tile", "moving_start_l0_zyx")
    ):
        raise ValueError(f"{initializer_path} is not the manual initializer of {row['moving_tile']}")
    seed = np.concatenate(
        [
            np.asarray(initializer["local_matrix_zyx"], dtype=np.float64).ravel(),
            np.asarray(initializer["local_translation_zyx"], dtype=np.float64),
        ]
    )
    fitted_residual = recovery.residual_verdict(
        scorer.residual_shifts(row, recovery._transform(row))
    )
    seed_residual = None
    if fitted_residual["verdict"] != "verified":
        seed_residual = recovery.residual_verdict(scorer.residual_shifts(row, seed))
    select_seed = seed_residual is not None and seed_residual["verdict"] == "verified"
    arbitrated = deepcopy(row)
    if select_seed:
        recovery._set_selected_transform(
            arbitrated,
            matrix=seed[:9].reshape(3, 3),
            translation=seed[9:],
            attempt=recovery._interpolated_recovery_attempt(native_method),
        )
    arbitrated["manual_seed_arbitration"] = {
        "artifact_type": "lightsheet.manual_seed_arbitration.v1",
        "policy": (
            "keep the accepted fit unless the residual check does not verify it "
            "and verifies the manual seed"
        ),
        "initializer_json": str(initializer_path),
        "previous_selected_attempt": row.get("selected_attempt"),
        "fitted_residual": fitted_residual,
        "seed_residual": seed_residual,
        "selected_candidate": "manual_seed" if select_seed else "fitted",
    }
    return arbitrated


def _write_completion_stage(
    *,
    summary: dict[str, Any],
    compact_rows: list[dict[str, Any]],
    summary_path: Path,
    output_dir: Path,
    report_key: str,
    report_name: str,
    report: dict[str, Any],
) -> Path:
    counts = Counter(str(row.get("status")) for row in compact_rows)
    return recovery._write_stage_artifacts(
        source_summary=summary,
        windows=compact_rows,
        input_summary=summary_path,
        output_dir=output_dir,
        report_key=report_key,
        report_name=report_name,
        report={
            **report,
            "accepted_count": counts["accepted"],
            "rejected_count": counts["rejected"],
            "error_count": counts["error"],
        },
        write_json=_write_json,
    )


def _promote_spatial_fallbacks(
    summary_path: Path, output_dir: Path, *, image_ncc_margin: float
) -> Path:
    summary_path = summary_path.resolve()
    summary = read_json_object(summary_path)
    native_method = recovery._native_method_from_summary(summary)
    scorer = None
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    window_dir = output_dir / "window_json"
    window_dir.mkdir(parents=True)

    compact_rows = []
    promoted_count = 0
    candidates: Counter[str] = Counter()
    contradicted_count = 0
    native_reasons: Counter[str] = Counter()
    support_counts: list[int] = []
    for compact in summary["windows"]:
        source_path = Path(compact["level0_json"])
        row = read_json_object(source_path)
        if row.get("status") == "rejected" and row.get("rejection_reason") in (
            SPATIAL_TARGET_REASON,
            "native_rerun_attempt_mismatch",
        ):
            if scorer is None:
                scorer = recovery.window_image_scorer(summary)
            row = _promote_spatial_fallback(
                row,
                native_method=native_method,
                scorer=scorer,
                image_ncc_margin=image_ncc_margin,
            )
            promotion = row["interpolated_fallback_promotion"]
            if row["status"] == "accepted":
                promoted_count += 1
                candidates[str(promotion["selected_candidate"])] += 1
            else:
                contradicted_count += 1
            native_reasons[str(promotion["native_rerun_rejection_reason"])] += 1
            support_counts.append(int(promotion["source_good_window_count"]))
        output_path = window_dir / source_path.name
        _write_json(output_path, row)
        compact_rows.append(
            recovery._compact_window_record(compact, row, output_path)
        )

    report = {
        "artifact_type": "lightsheet.interpolated_spatial_fallback_promotion_summary.v1",
        "input_summary": str(summary_path),
        "promoted_count": promoted_count,
        "selected_candidates": dict(sorted(candidates.items())),
        "contradicted_by_image_count": contradicted_count,
        "replacement_image_ncc_margin": image_ncc_margin,
        "native_rejection_reasons": dict(sorted(native_reasons.items())),
        "minimum_source_good_window_count": min(support_counts, default=None),
        "maximum_source_good_window_count": max(support_counts, default=None),
    }
    return _write_completion_stage(
        summary=summary,
        compact_rows=compact_rows,
        summary_path=summary_path,
        output_dir=output_dir,
        report_key="interpolated_fallback_promotion",
        report_name="interpolated_fallback_promotion.json",
        report=report,
    )


def _arbitrate_manual_seeds(summary_path: Path, output_dir: Path) -> Path:
    """Arbitrate every accepted curated window after all other recovery policies."""
    summary_path = summary_path.resolve()
    summary = read_json_object(summary_path)
    native_method = recovery._native_method_from_summary(summary)
    scorer = None
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    window_dir = output_dir / "window_json"
    window_dir.mkdir(parents=True)

    compact_rows = []
    candidates: Counter[str] = Counter()
    for compact in summary["windows"]:
        source_path = Path(compact["level0_json"])
        row = read_json_object(source_path)
        if (
            row.get("status") == "accepted"
            and (row.get("outlier_recovery") or {}).get("initializer_source") == "manual"
        ):
            if scorer is None:
                scorer = recovery.window_image_scorer(summary)
            row = _arbitrate_manual_seed(row, native_method=native_method, scorer=scorer)
            candidates[row["manual_seed_arbitration"]["selected_candidate"]] += 1
        output_path = window_dir / source_path.name
        _write_json(output_path, row)
        compact_rows.append(recovery._compact_window_record(compact, row, output_path))

    return _write_completion_stage(
        summary=summary,
        compact_rows=compact_rows,
        summary_path=summary_path,
        output_dir=output_dir,
        report_key="manual_seed_arbitration",
        report_name="manual_seed_arbitration.json",
        report={
            "artifact_type": "lightsheet.manual_seed_arbitration_summary.v1",
            "input_summary": str(summary_path),
            "selected_candidates": dict(sorted(candidates.items())),
        },
    )


def _is_native_source(row: dict[str, Any], native_method: str) -> bool:
    attempt = str(row.get("selected_attempt", ""))
    return row.get("status") == "accepted" and (
        attempt.startswith(recovery._native_attempt_prefix(native_method))
        or attempt.startswith("level2_initializer_")
    )


def _recover_linear_mapping_rejections(
    summary_path: Path, output_dir: Path, *, image_ncc_margin: float
) -> Path:
    summary_path = summary_path.resolve()
    summary = read_json_object(summary_path)
    native_method = recovery._native_method_from_summary(summary)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    records = [
        (Path(item["level0_json"]), read_json_object(Path(item["level0_json"])))
        for item in summary["windows"]
    ]
    by_tile: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, row in records:
        by_tile[str(row["moving_tile"])].append((path, row))

    predictions: dict[Path, tuple[np.ndarray, list[str]]] = {}
    image_scores: dict[Path, tuple[list[float], list[float]]] = {}
    scorer = None
    target_count = 0
    for tile, items in by_tile.items():
        targets = [
            (path, row)
            for path, row in items
            if row.get("status") == "rejected"
            and row.get("rejection_reason") == LINEAR_TARGET_REASON
        ]
        if not targets:
            continue
        target_count += len(targets)
        sources = [(path, row) for path, row in items if _is_native_source(row, native_method)]
        if len(sources) < 2:
            raise ValueError(f"{tile} has only {len(sources)} native interpolation sources")
        points = np.stack([recovery._point(row) for _path, row in sources])
        transforms = np.stack([recovery._transform(row) for _path, row in sources])
        normalized, origin, spacing = recovery._normalized_points(points)
        target_points = np.stack(
            [(recovery._point(row) - origin) / spacing for _path, row in targets]
        )
        fitted = recovery.predict_decomposed_affine_field(
            normalized, transforms, target_points=target_points
        )
        source_paths = [str(path.resolve()) for path, _row in sources]
        if scorer is None:
            scorer = recovery.window_image_scorer(summary)
        for (path, row), transform in zip(targets, fitted, strict=True):
            if not recovery._affine_is_plausible(transform):
                raise ValueError(f"Implausible interpolated transform for {path}")
            predictions[path] = (transform, source_paths)
            fitted_planes, replacement_planes = scorer.plane_scores(
                row, [recovery._transform(row), transform]
            )
            image_scores[path] = (fitted_planes, replacement_planes)

    if len(predictions) != target_count:
        raise ValueError(
            f"Expected {target_count} recoveries, found {len(predictions)}"
        )

    window_dir = output_dir / "window_json"
    window_dir.mkdir(parents=True)
    compact_rows = []
    support_counts: list[int] = []
    retained_count = 0
    for compact, (source_path, original_row) in zip(summary["windows"], records, strict=True):
        row = deepcopy(original_row)
        prediction = predictions.get(source_path)
        if prediction is not None:
            transform, source_paths = prediction
            fitted_planes, replacement_planes = image_scores[source_path]
            replace = recovery.image_prefers_replacement(
                fitted_planes, replacement_planes, image_ncc_margin
            )
            if replace:
                attempt = (
                    f"interpolated_{native_method.replace('-', '_')}_linear_mapping_recovery"
                )
                recovery._set_selected_transform(
                    row,
                    matrix=transform[:9].reshape(3, 3),
                    translation=transform[9:],
                    attempt=attempt,
                )
            else:
                attempt = original_row.get("selected_attempt")
                retained_count += 1
            row.update({"status": "accepted", "rejection_reason": None})
            row["linear_mapping_recovery"] = {
                "artifact_type": "lightsheet.linear_mapping_recovery.v2",
                "policy": (
                    "decomposed-affine interpolation from accepted native within-tile "
                    "windows, applied only when the image favors it over the fitted transform"
                ),
                "previous_status": original_row.get("status"),
                "previous_rejection_reason": original_row.get("rejection_reason"),
                "previous_selected_attempt": original_row.get("selected_attempt"),
                "selected_candidate": "interpolated" if replace else "fitted",
                "fitted_image_ncc_planes": fitted_planes,
                "replacement_image_ncc_planes": replacement_planes,
                "replacement_image_ncc_margin": image_ncc_margin,
                "source_window_json": source_paths,
                "source_window_count": len(source_paths),
                "selected_attempt": attempt,
            }
            support_counts.append(len(source_paths))
        output_path = window_dir / source_path.name
        _write_json(output_path, row)
        compact_rows.append(
            recovery._compact_window_record(compact, row, output_path)
        )

    report = {
        "artifact_type": "lightsheet.linear_mapping_recovery_summary.v1",
        "input_summary": str(summary_path),
        "recovered_count": len(predictions),
        "fitted_retained_count": retained_count,
        "replacement_image_ncc_margin": image_ncc_margin,
        "minimum_source_window_count": min(support_counts, default=None),
        "maximum_source_window_count": max(support_counts, default=None),
    }
    return _write_completion_stage(
        summary=summary,
        compact_rows=compact_rows,
        summary_path=summary_path,
        output_dir=output_dir,
        report_key="linear_mapping_recovery",
        report_name="linear_mapping_recovery.json",
        report=report,
    )


def _run_policy_stages(
    native_summary: Path,
    workspace: Path,
    settings: CompleteRecoverySettings,
) -> tuple[Path, list[tuple[str, Path]]]:
    filter_dir = workspace / "01-filter"
    filter_summary = recovery.filter_final_linear_outliers(
        SimpleNamespace(
            summary=native_summary,
            output_dir=filter_dir,
            minimum_linear_samples=settings.minimum_linear_samples,
            linear_loss_scale_um=settings.linear_loss_scale_um,
            outlier_mad=settings.outlier_mad,
            minimum_linear_outlier_um=settings.minimum_linear_outlier_um,
            maximum_linear_outlier_um=settings.maximum_linear_outlier_um,
            maximum_linear_median_residual_um=(
                settings.maximum_linear_median_residual_um
            ),
        )
    )
    smooth_dir = workspace / "02-smooth"
    smooth_summary = recovery.smooth_final_linear_mapping(
        SimpleNamespace(
            summary=filter_summary,
            output_dir=smooth_dir,
            minimum_smoothing_samples=settings.minimum_smoothing_samples,
            maximum_outlier_px=settings.maximum_outlier_px,
            replacement_image_ncc_margin=settings.replacement_image_ncc_margin,
        )
    )

    promote_dir = workspace / "03-promote"
    promote_summary = _promote_spatial_fallbacks(
        smooth_summary,
        promote_dir,
        image_ncc_margin=settings.replacement_image_ncc_margin,
    )

    linear_dir = workspace / "04-linear-recovery"
    linear_summary = _recover_linear_mapping_rejections(
        promote_summary,
        linear_dir,
        image_ncc_margin=settings.replacement_image_ncc_margin,
    )

    manual_dir = workspace / "05-manual-seed"
    final_summary = _arbitrate_manual_seeds(linear_summary, manual_dir)
    return final_summary, [
        ("physical_linear_filter", filter_dir / "method10_final_linear_filter.json"),
        (
            "retained_affine_outlier_smoothing",
            smooth_dir / "method10_final_linear_interpolation.json",
        ),
        (
            "spatial_fallback_promotion",
            promote_dir / "interpolated_fallback_promotion.json",
        ),
        ("linear_mapping_recovery", linear_dir / "linear_mapping_recovery.json"),
        ("manual_seed_arbitration", manual_dir / "manual_seed_arbitration.json"),
    ]


def _logicalize_temporary_paths(value: Any, workspace: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _logicalize_temporary_paths(item, workspace)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_logicalize_temporary_paths(item, workspace) for item in value]
    if isinstance(value, str):
        try:
            relative = Path(value).resolve().relative_to(workspace.resolve())
        except (ValueError, OSError):
            return value
        return f"recovery-stage://{relative.as_posix()}"
    return value


def _validate_final(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    compact_rows = summary.get("windows", [])
    if len(rows) != len(compact_rows):
        raise ValueError("Final recovery publication lost window records")
    for compact, row in zip(compact_rows, rows, strict=True):
        for key in ("status", "rejection_reason", "selected_attempt"):
            if compact.get(key) != row.get(key):
                raise ValueError(
                    f"Compact/full mismatch for {key} in {compact['level0_json']}"
                )
        if row.get("status") == "error":
            raise ValueError(
                f"Recovery contains an error window: {compact['level0_json']}"
            )
        if row.get("status") == "accepted":
            transform = recovery._transform(row)
            if not recovery._affine_is_plausible(transform):
                raise ValueError(
                    f"Recovery contains an implausible affine: {compact['level0_json']}"
                )


def _publish(
    source_summary: Path,
    reports: list[tuple[str, Path]],
    output_dir: Path,
    native_summary: Path,
    identity: dict[str, Any],
    workspace: Path,
) -> Path:
    source = read_json_object(source_summary)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.publishing-", dir=output_dir.parent
        )
    )
    try:
        window_dir = staging / "window_json"
        window_dir.mkdir()
        rows = []
        for compact in source["windows"]:
            input_path = Path(compact["level0_json"])
            row = _logicalize_temporary_paths(read_json_object(input_path), workspace)
            output_path = window_dir / input_path.name
            _write_json(output_path, row)
            compact["level0_json"] = str(
                (output_dir / "window_json" / input_path.name).resolve()
            )
            rows.append(row)

        source = _logicalize_temporary_paths(source, workspace)
        source["output_dir"] = str(output_dir.resolve())
        source["input_summary"] = str(native_summary.resolve())
        source["complete_recovery"] = {
            "artifact_type": "lightsheet.complete_recovery.v1",
            "producer_version": PRODUCER_VERSION,
            "identity": identity["digest"],
            "policy_order": list(POLICY_ORDER),
            "native_checkpoint": str(native_summary.resolve()),
        }
        _validate_final(source, rows)

        report_dir = staging / "reports"
        report_dir.mkdir()
        published_reports = {}
        for stage, path in reports:
            report = _logicalize_temporary_paths(read_json_object(path), workspace)
            report_path = report_dir / f"{stage}.json"
            _write_json(report_path, report)
            published_reports[stage] = str(
                (output_dir / "reports" / report_path.name).resolve()
            )
        source["complete_recovery"]["reports"] = published_reports

        summary_path = staging / SUMMARY_NAME
        _write_json(summary_path, source)
        final_window_sha256 = {
            path.name: _sha256(path) for path in sorted(window_dir.glob("*.json"))
        }
        report_sha256 = {
            path.name: _sha256(path) for path in sorted(report_dir.glob("*.json"))
        }
        manifest = {
            "artifact_type": "lightsheet.complete_recovery_manifest.v1",
            "status": "complete",
            "identity": identity,
            "summary_sha256": _sha256(summary_path),
            "final_window_sha256": final_window_sha256,
            "report_sha256": report_sha256,
        }
        _write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise FileExistsError(f"Recovery output already exists: {output_dir}")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / SUMMARY_NAME


def _matches_hashes(directory: Path, expected: object) -> bool:
    if not isinstance(expected, dict):
        return False
    actual_paths = sorted(path for path in directory.glob("*.json") if path.is_file())
    if {path.name for path in actual_paths} != set(expected):
        return False
    return all(_sha256(path) == expected[path.name] for path in actual_paths)


def complete_recovery(
    *,
    native_summary: Path,
    output_dir: Path,
    settings: CompleteRecoverySettings = DEFAULT_COMPLETE_RECOVERY_SETTINGS,
) -> Path:
    """Run the fixed four-stage policy and atomically publish its durable result."""
    native_summary = native_summary.resolve()
    output_dir = output_dir.resolve()
    identity = _input_identity(native_summary, settings)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json_object(manifest_path)
        final_summary = output_dir / SUMMARY_NAME
        valid_files = (
            final_summary.is_file()
            and _sha256(final_summary) == manifest.get("summary_sha256")
            and _matches_hashes(
                output_dir / "window_json", manifest.get("final_window_sha256")
            )
            and _matches_hashes(output_dir / "reports", manifest.get("report_sha256"))
        )
        if (
            manifest.get("status") == "complete"
            and manifest.get("identity", {}).get("digest") == identity["digest"]
            and valid_files
        ):
            return final_summary
        raise ValueError(f"Existing recovery artifact identity mismatch: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Incomplete recovery output exists: {output_dir}")

    with tempfile.TemporaryDirectory(
        prefix="squisher-complete-recovery-"
    ) as temporary:
        workspace = Path(temporary)
        final_summary, reports = _run_policy_stages(
            native_summary, workspace, settings
        )
        return _publish(
            final_summary,
            reports,
            output_dir,
            native_summary,
            identity,
            workspace,
        )
