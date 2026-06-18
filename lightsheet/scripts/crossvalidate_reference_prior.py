#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as stitch


DEFAULT_POSITION_INPUT = Path(
    "/home/chaichontat/nvme/lightsheet/20260613/"
    "230Tnc-CLR-488514561638.overlap25.roughPhase.zSlab32.fullCanvas.positions.json"
)
DEFAULT_REFERENCE_REGISTRATION = Path(
    "/working/230Tnc/230Tnc-CLR-488514561638-registration/registration.track0.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/working/230Tnc/230Tnc-CLR-488514561638-shared561638-prior-cv"
)
DEFAULT_CONSTRAINTS = (
    (
        "track1",
        Path(
            "/working/230Tnc/"
            "230Tnc-CLR-488514561638-shared561638-penalizedxy001-zreject-488seed-registration/"
            "track1-robust-boundary-qc/boundary_residuals.jsonl"
        ),
    ),
    (
        "track2",
        Path(
            "/working/230Tnc/"
            "230Tnc-CLR-488514561638-shared561638-penalizedxy001-zreject-488seed-registration/"
            "track2-robust-boundary-qc/boundary_residuals.jsonl"
        ),
    ),
)
DEFAULT_PRIOR_WEIGHTS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def parse_constraint_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("constraints must be passed as label=/path/file.jsonl")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("constraint label cannot be empty")
    return label, Path(raw_path)


def read_constraints(paths: list[tuple[str, Path]]) -> list[stitch.BoundaryConstraint]:
    constraints: list[stitch.BoundaryConstraint] = []
    for label, path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("accepted"):
                    continue
                if float(row.get("weight", 0.0)) <= 0.0:
                    continue
                try:
                    constraints.append(
                        stitch.BoundaryConstraint(
                            fixed=int(row["fixed"]),
                            moving=int(row["moving"]),
                            pair=tuple(int(value) for value in row["pair"]),
                            axis=str(row["axis"]),
                            patch_index=int(row["patch_index"]),
                            shift_zyx=tuple(float(value) for value in row["shift_zyx"]),
                            weight=float(row["weight"]),
                            correlation_before=float(row["correlation_before"]),
                            correlation_after=float(row["correlation_after"]),
                            improvement=float(row["improvement"]),
                            fixed_nonzero_fraction=float(row["fixed_nonzero_fraction"]),
                            moving_nonzero_fraction=float(row["moving_nonzero_fraction"]),
                            fixed_std=float(row["fixed_std"]),
                            moving_std=float(row["moving_std"]),
                            accepted=True,
                            fixed_content_fraction=float(row.get("fixed_content_fraction", 0.0)),
                            moving_content_fraction=float(row.get("moving_content_fraction", 0.0)),
                            final_residual_zyx=tuple(float(value) for value in row["final_residual_zyx"])
                            if row.get("final_residual_zyx") is not None
                            else None,
                            source_label=label,
                        )
                    )
                except KeyError as exc:
                    raise ValueError(f"{path}:{line_number} is missing {exc}") from exc
    if not constraints:
        raise ValueError("No accepted constraints found")
    return constraints


def split_folds(
    constraints: list[stitch.BoundaryConstraint],
    *,
    n_folds: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    groups: dict[tuple[str, tuple[int, int], str], list[int]] = defaultdict(list)
    for index, constraint in enumerate(constraints):
        groups[(constraint.source_label or "", constraint.pair, constraint.axis)].append(index)

    folds = [-1 for _ in constraints]
    for indices in groups.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for offset, index in enumerate(shuffled):
            folds[index] = offset % n_folds

    if any(fold < 0 for fold in folds):
        raise RuntimeError("Internal error: not all constraints were assigned a fold")
    return folds


def residuals_px(
    corrections_zyx: list[tuple[float, float, float]],
    constraints: list[stitch.BoundaryConstraint],
) -> np.ndarray:
    corrections = np.asarray(corrections_zyx, dtype=float)
    residuals = []
    for constraint in constraints:
        residuals.append(
            corrections[constraint.moving] - corrections[constraint.fixed] - np.asarray(constraint.shift_zyx, dtype=float)
        )
    return np.asarray(residuals, dtype=float)


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(math.sqrt(np.sum(weights * np.square(values)) / total))


def residual_metrics(
    residual_um: np.ndarray,
    weights: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    abs_residual = np.abs(residual_um)
    norm_zyx = np.linalg.norm(residual_um, axis=1)
    norm_xy = np.linalg.norm(residual_um[:, 1:3], axis=1)
    return {
        f"{prefix}_n": float(residual_um.shape[0]),
        f"{prefix}_zyx_weighted_rms_um": weighted_rms(norm_zyx, weights),
        f"{prefix}_zyx_median_um": float(np.median(norm_zyx)),
        f"{prefix}_zyx_p95_um": float(np.percentile(norm_zyx, 95)),
        f"{prefix}_z_weighted_rms_um": weighted_rms(residual_um[:, 0], weights),
        f"{prefix}_z_median_abs_um": float(np.median(abs_residual[:, 0])),
        f"{prefix}_z_p95_abs_um": float(np.percentile(abs_residual[:, 0], 95)),
        f"{prefix}_xy_weighted_rms_um": weighted_rms(norm_xy, weights),
        f"{prefix}_xy_median_um": float(np.median(norm_xy)),
        f"{prefix}_xy_p95_um": float(np.percentile(norm_xy, 95)),
        f"{prefix}_y_p95_abs_um": float(np.percentile(abs_residual[:, 1], 95)),
        f"{prefix}_x_p95_abs_um": float(np.percentile(abs_residual[:, 2], 95)),
    }


def drift_metrics(corrections_zyx: list[tuple[float, float, float]], spacing_zyx: np.ndarray) -> dict[str, float]:
    drift_um = np.asarray(corrections_zyx, dtype=float) * spacing_zyx[None, :]
    metrics: dict[str, float] = {}
    for index, axis in enumerate(("z", "y", "x")):
        values = drift_um[:, index]
        abs_values = np.abs(values)
        metrics[f"drift_{axis}_median_um"] = float(np.median(values))
        metrics[f"drift_{axis}_p95_abs_um"] = float(np.percentile(abs_values, 95))
        metrics[f"drift_{axis}_max_abs_um"] = float(np.max(abs_values))
    xy_norm = np.linalg.norm(drift_um[:, 1:3], axis=1)
    metrics["drift_xy_p95_um"] = float(np.percentile(xy_norm, 95))
    metrics["drift_xy_max_um"] = float(np.max(xy_norm))
    return metrics


def fold_metric_row(
    *,
    mode: str,
    prior_weight: float | None,
    fold: int,
    corrections: list[tuple[float, float, float]],
    train_constraints: list[stitch.BoundaryConstraint],
    test_constraints: list[stitch.BoundaryConstraint],
    spacing_zyx: np.ndarray,
    train_connected_tile_count: int,
) -> dict[str, Any]:
    train_weights = np.asarray([constraint.weight for constraint in train_constraints], dtype=float)
    test_weights = np.asarray([constraint.weight for constraint in test_constraints], dtype=float)
    train_residual_um = residuals_px(corrections, train_constraints) * spacing_zyx[None, :]
    test_residual_um = residuals_px(corrections, test_constraints) * spacing_zyx[None, :]
    return {
        "mode": mode,
        "prior_weight": prior_weight,
        "fold": fold,
        "train_connected_tiles": train_connected_tile_count,
        **residual_metrics(train_residual_um, train_weights, prefix="train"),
        **residual_metrics(test_residual_um, test_weights, prefix="heldout"),
        **drift_metrics(corrections, spacing_zyx),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), row["prior_weight"])].append(row)

    summary_rows = []
    metric_names = [
        name
        for name in rows[0]
        if name not in {"mode", "prior_weight", "fold"}
        and isinstance(rows[0][name], (int, float))
    ]
    for (mode, prior_weight), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], -1.0 if item[0][1] is None else float(item[0][1])),
    ):
        summary: dict[str, Any] = {"mode": mode, "prior_weight": prior_weight, "folds": len(group)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = [
        row
        for row in summary_rows
        if row["mode"] == "penalized-xy" and row["prior_weight"] is not None and row["prior_weight"] > 0
    ]
    if not sweep:
        return
    weights = np.asarray([float(row["prior_weight"]) for row in sweep], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")
    ax.plot(weights, [row["heldout_z_p95_abs_um_mean"] for row in sweep], marker="o", label="heldout z p95")
    ax.plot(weights, [row["heldout_xy_p95_um_mean"] for row in sweep], marker="o", label="heldout xy p95")
    ax.set_xlabel("xy prior weight")
    ax.set_ylabel("held-out residual (um)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_sweep_heldout_residuals.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")
    ax.plot(weights, [row["drift_z_p95_abs_um_mean"] for row in sweep], marker="o", label="z drift p95")
    ax.plot(weights, [row["drift_xy_p95_um_mean"] for row in sweep], marker="o", label="xy drift p95")
    ax.set_xlabel("xy prior weight")
    ax.set_ylabel("drift from 488 geometry (um)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_sweep_drift_from_488.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xscale("log")
    ax1.plot(weights, [row["heldout_zyx_weighted_rms_um_mean"] for row in sweep], color="tab:blue", marker="o")
    ax1.set_xlabel("xy prior weight")
    ax1.set_ylabel("held-out weighted RMS (um)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(weights, [row["drift_xy_p95_um_mean"] for row in sweep], color="tab:red", marker="s")
    ax2.set_ylabel("xy drift p95 from 488 (um)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_sweep_tradeoff.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validate reference-geometry xy prior weights from cached boundary measurements."
    )
    parser.add_argument("--position-input", type=Path, default=DEFAULT_POSITION_INPUT)
    parser.add_argument("--reference-registration", type=Path, default=DEFAULT_REFERENCE_REGISTRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--constraint-jsonl",
        type=parse_constraint_arg,
        action="append",
        default=None,
        help="Accepted boundary measurements as label=/path/file.jsonl. Defaults to track1 and track2.",
    )
    parser.add_argument("--prior-weight", type=float, action="append", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    constraint_paths = args.constraint_jsonl or list(DEFAULT_CONSTRAINTS)
    prior_weights = args.prior_weight or list(DEFAULT_PRIOR_WEIGHTS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tiles from {args.position_input}", flush=True)
    tiles = stitch.read_position_input_tiles(args.position_input)
    print(f"Validating reference registration from {args.reference_registration}", flush=True)
    stitch.load_registration_params(args.reference_registration, tiles)
    spacing_zyx = np.asarray([tiles[0].spacing[axis] for axis in ("z", "y", "x")], dtype=float)

    print("Loading cached accepted constraints", flush=True)
    constraints = read_constraints(constraint_paths)
    folds = split_folds(constraints, n_folds=args.folds, seed=args.seed)
    anchor_tile = stitch.choose_anchor_tile(tiles, constraints)
    settings = stitch.RobustBoundarySettings()
    print(
        f"Using {len(constraints)} accepted constraints across {args.folds} folds; "
        f"anchor tile {anchor_tile}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    modes: list[tuple[str, float | None, set[str], tuple[float, float, float] | None]] = [
        ("fixed-xy", None, {"y", "x"}, None),
        ("full-xyz", 0.0, set(), (0.0, 0.0, 0.0)),
    ]
    modes.extend(
        ("penalized-xy", weight, set(), (0.0, float(weight), float(weight)))
        for weight in prior_weights
        if weight > 0
    )

    for mode, prior_weight, fixed_axes, prior_weights_zyx in modes:
        for fold in range(args.folds):
            train_constraints = [
                constraint
                for constraint, constraint_fold in zip(constraints, folds, strict=True)
                if constraint_fold != fold
            ]
            test_constraints = [
                constraint
                for constraint, constraint_fold in zip(constraints, folds, strict=True)
                if constraint_fold == fold
            ]
            corrections = stitch.solve_tile_corrections_zyx(
                len(tiles),
                train_constraints,
                settings,
                anchor_tile,
                fixed_axes=fixed_axes,
                reference_prior_weights_zyx=prior_weights_zyx,
            )
            rows.append(
                fold_metric_row(
                    mode=mode,
                    prior_weight=prior_weight,
                    fold=fold,
                    corrections=corrections,
                    train_constraints=train_constraints,
                    test_constraints=test_constraints,
                    spacing_zyx=spacing_zyx,
                    train_connected_tile_count=len(
                        stitch.anchor_connected_tiles(len(tiles), train_constraints, anchor_tile)
                    ),
                )
            )
        print(f"Finished {mode} prior={prior_weight}", flush=True)

    summary_rows = summarize_rows(rows)
    write_csv(args.output_dir / "prior_sweep_folds.csv", rows)
    write_csv(args.output_dir / "prior_sweep_summary.csv", summary_rows)
    (args.output_dir / "prior_sweep_summary.json").write_text(
        json.dumps(
            {
                "position_input": str(args.position_input),
                "reference_registration": str(args.reference_registration),
                "constraints": [{"label": label, "path": str(path)} for label, path in constraint_paths],
                "folds": args.folds,
                "seed": args.seed,
                "anchor_tile": anchor_tile,
                "settings": asdict(settings),
                "summary": summary_rows,
            },
            indent=2,
        )
        + "\n"
    )
    write_plots(args.output_dir, summary_rows)
    print(f"Wrote cross-validation sweep to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
