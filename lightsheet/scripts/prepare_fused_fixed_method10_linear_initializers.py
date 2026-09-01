#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _tile_number(value: str) -> str:
    return value.split(".")[-3]


def _point(row: dict[str, Any]) -> np.ndarray:
    point = np.asarray(row["moving_start_l0_zyx"], dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"invalid window point for {row.get('moving_tile')}: {point}")
    return point


def _transform(row: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(row["selected_local_matrix_zyx"], dtype=np.float64)
    translation = np.asarray(row["selected_local_translation_zyx"], dtype=np.float64)
    if matrix.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"invalid transform shape for {row.get('moving_tile')}")
    transform = np.concatenate((matrix.ravel(), translation))
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"non-finite transform for {row.get('moving_tile')}")
    return transform


def fit_linear_transform(
    points_zyx: np.ndarray,
    transforms: np.ndarray,
    target_zyx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit transform = intercept + z/y/x slopes using ordinary least squares."""
    points = np.asarray(points_zyx, dtype=np.float64)
    values = np.asarray(transforms, dtype=np.float64)
    target = np.asarray(target_zyx, dtype=np.float64)
    origin = np.min(points, axis=0)
    spacing = np.ones(3, dtype=np.float64)
    varying_axes = []
    for axis in range(3):
        differences = np.diff(np.unique(points[:, axis]))
        if differences.size:
            spacing[axis] = float(np.median(differences[differences > 0]))
            varying_axes.append(axis)
    normalized = (points - origin) / spacing
    active_axes = []
    rank = 1
    for axis in varying_axes:
        candidate = np.column_stack((np.ones(len(points)), normalized[:, active_axes + [axis]]))
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank > rank:
            active_axes.append(axis)
            rank = candidate_rank
    design = np.column_stack((np.ones(len(points)), normalized[:, active_axes]))
    fitted, _residuals, fitted_rank, singular_values = np.linalg.lstsq(design, values, rcond=None)
    expected_rank = 1 + len(active_axes)
    if fitted_rank != expected_rank:
        raise ValueError(f"linear transform model requires rank {expected_rank}, got {fitted_rank}")
    coefficients = np.zeros((4, values.shape[1]), dtype=np.float64)
    coefficients[0] = fitted[0]
    coefficients[np.asarray(active_axes, dtype=np.int64) + 1] = fitted[1:]
    prediction = np.r_[1.0, (target - origin) / spacing] @ coefficients
    return prediction, {
        "origin_zyx": origin.tolist(),
        "spacing_zyx": spacing.tolist(),
        "coefficients_intercept_zyx": coefficients.tolist(),
        "design_rank": int(fitted_rank),
        "modeled_spatial_axes": ["zyx"[axis] for axis in active_axes],
        "constant_spatial_axes": ["zyx"[axis] for axis in range(3) if axis not in varying_axes],
        "dependent_spatial_axes": ["zyx"[axis] for axis in varying_axes if axis not in active_axes],
        "design_singular_values": singular_values.tolist(),
    }


def _affine_is_plausible(transform: np.ndarray) -> bool:
    matrix = transform[:9].reshape(3, 3)
    determinant = float(np.linalg.det(matrix))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return bool(
        np.all(np.isfinite(transform))
        and 1.0 / 1.1 <= determinant <= 1.1
        and singular_values.max() / singular_values.min() <= 1.1
        and singular_values.min() >= 1.0 / 1.1
        and singular_values.max() <= 1.1
    )


def run(input_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    initializer_paths = sorted(input_dir.glob("*.json"))
    if not initializer_paths:
        raise ValueError(f"no initializer JSON files found in {input_dir}")

    initializers = {path: _read_json(path) for path in initializer_paths}
    targets_by_tile: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, row in initializers.items():
        targets_by_tile.setdefault(_tile_number(str(row["moving_tile"])), []).append((path, row))

    source_rows: dict[Path, dict[str, Any]] = {}
    report_tiles = []
    written = 0
    for tile, targets in sorted(targets_by_tile.items()):
        robust_inliers = {
            Path(source)
            for _path, row in targets
            for source in row["recovery"]["source_good_window_json"]
        }
        for source in robust_inliers:
            source_rows.setdefault(source, _read_json(source))
        failures = []
        for path, initializer in targets:
            target_source = Path(initializer["recovery"]["source_window_json"])
            training_paths = sorted(robust_inliers - {target_source})
            try:
                training_rows = [source_rows[source] for source in training_paths]
                prediction, model = fit_linear_transform(
                    np.stack([_point(row) for row in training_rows]),
                    np.stack([_transform(row) for row in training_rows]),
                    np.asarray(initializer["moving_start_l0_zyx"], dtype=np.float64),
                )
                if not _affine_is_plausible(prediction):
                    raise ValueError("predicted affine is not plausible")
            except ValueError as exc:
                failures.append({"initializer": str(path), "error": str(exc)})
                continue

            output = dict(initializer)
            output["artifact_type"] = "lightsheet.fused_fixed_method10_linear_model_initializer.v1"
            output["local_matrix_zyx"] = prediction[:9].reshape(3, 3).tolist()
            output["local_translation_zyx"] = prediction[9:].tolist()
            recovery = dict(output["recovery"])
            recovery["method"] = "per-tile linear least-squares model over normalized ZYX using robust spatial inliers"
            recovery["robust_inlier_source"] = str(input_dir.resolve())
            recovery["training_window_count"] = len(training_paths)
            recovery["target_excluded_from_training"] = target_source in robust_inliers
            recovery["linear_model"] = model
            output["recovery"] = recovery
            _write_json(output_dir / path.name, output)
            written += 1

        report_tiles.append(
            {
                "tile": tile,
                "target_count": len(targets),
                "robust_inlier_count": len(robust_inliers),
                "written_count": len(targets) - len(failures),
                "failures": failures,
            }
        )

    report = {
        "artifact_type": "lightsheet.fused_fixed_method10_linear_model_initializer_summary.v1",
        "input_initializer_dir": str(input_dir.resolve()),
        "output_initializer_dir": str(output_dir.resolve()),
        "model": "ordinary least squares: transform parameter = intercept + beta_z*z + beta_y*y + beta_x*x",
        "target_count": len(initializer_paths),
        "written_count": written,
        "failure_count": len(initializer_paths) - written,
        "tiles": report_tiles,
    }
    report_path = output_dir / "linear_model_initializer_summary.json"
    _write_json(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit fused-fixed Method-10 initializers with per-tile linear models.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.input_dir.resolve(), args.output_dir.resolve()), flush=True)


if __name__ == "__main__":
    main()
