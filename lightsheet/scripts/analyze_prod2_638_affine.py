#!/usr/bin/env python
"""Assess convergence of accepted 638-to-561 local affine fits."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def tile_number(name: str) -> int:
    match = re.search(r"\.(\d{3})\.ome", name)
    if match is None:
        raise ValueError(f"Cannot parse tile number from {name!r}")
    return int(match.group(1))


def median_transform(values: np.ndarray) -> np.ndarray:
    return np.median(values, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted((args.run_dir / "window_json").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "accepted":
            continue
        matrix = np.asarray(row["selected_local_matrix_zyx"], dtype=np.float64)
        translation = np.asarray(row["selected_local_translation_zyx"], dtype=np.float64)
        if matrix.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(f"Unexpected affine shape in {path}")
        rows.append(
            {
                "tile": tile_number(row["moving_tile"]),
                "z": int(row["moving_start_l0_zyx"][0]),
                "matrix": matrix,
                "translation": translation,
                "corr": float(row["selected_corr_refined"]),
                "grad_ncc": float(row["selected_gradient_component_ncc_mean"]),
            }
        )
    rows.sort(key=lambda row: (row["tile"], row["z"]))
    if len(rows) < 2:
        raise ValueError(f"Need at least two accepted fits, found {len(rows)}")

    matrices = np.stack([row["matrix"] for row in rows])
    translations = np.stack([row["translation"] for row in rows])
    parameters = np.concatenate([matrices.reshape(len(rows), 9), translations], axis=1)
    final = median_transform(parameters)
    cumulative = np.stack([median_transform(parameters[: index + 1]) for index in range(len(rows))])
    matrix_error = np.linalg.norm(cumulative[:, :9] - final[:9], axis=1)
    translation_error = np.linalg.norm(cumulative[:, 9:] - final[9:], axis=1)

    split = len(rows) // 2
    early = median_transform(parameters[:split])
    late = median_transform(parameters[split:])
    checkpoints = sorted({value for value in (5, 10, 20, 30, 40, 50, len(rows)) if value <= len(rows)})

    median_abs_deviation = np.median(np.abs(parameters - final), axis=0)
    summary = {
        "artifact_type": "prod2.638_to_561_affine_convergence.v1",
        "run_dir": str(args.run_dir.resolve()),
        "accepted_fit_count": len(rows),
        "accepted_tile_count": len({row["tile"] for row in rows}),
        "tile_range": [min(row["tile"] for row in rows), max(row["tile"] for row in rows)],
        "final_componentwise_median": {
            "matrix_zyx": final[:9].reshape(3, 3).tolist(),
            "translation_zyx_px": final[9:].tolist(),
        },
        "component_mad": {
            "matrix_zyx": median_abs_deviation[:9].reshape(3, 3).tolist(),
            "translation_zyx_px": median_abs_deviation[9:].tolist(),
        },
        "early_late_difference": {
            "matrix_frobenius": float(np.linalg.norm(early[:9] - late[:9])),
            "translation_l2_px": float(np.linalg.norm(early[9:] - late[9:])),
            "early_count": split,
            "late_count": len(rows) - split,
        },
        "checkpoints": [
            {
                "accepted_fits": count,
                "matrix_frobenius_to_final": float(matrix_error[count - 1]),
                "translation_l2_px_to_final": float(translation_error[count - 1]),
            }
            for count in checkpoints
        ],
        "quality": {
            "median_corr": float(np.median([row["corr"] for row in rows])),
            "median_grad_ncc": float(np.median([row["grad_ncc"] for row in rows])),
        },
    }

    arial = fm.findfont("Arial", fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "ch3_638_affine_convergence.json"
    output_png = args.output_dir / "ch3_638_affine_convergence.png"
    summary["resolved_font"] = arial
    output_json.write_text(json.dumps(summary, indent=2) + "\n")

    x = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), constrained_layout=True)
    axes[0, 0].plot(x, matrix_error, color="#335c81", linewidth=1.5)
    axes[0, 0].set(xlabel="Accepted fits accumulated", ylabel="Matrix difference to final\n(Frobenius norm)")
    axes[0, 1].plot(x, translation_error, color="#8f4f55", linewidth=1.5)
    axes[0, 1].set(xlabel="Accepted fits accumulated", ylabel="Translation difference to final (px)")

    labels = [f"m{row + 1}{column + 1}" for row in range(3) for column in range(3)]
    final_matrix_delta = final[:9] - np.eye(3).reshape(-1)
    matrix_mad = median_abs_deviation[:9]
    axes[1, 0].errorbar(
        np.arange(9),
        final_matrix_delta,
        yerr=matrix_mad,
        fmt="o",
        color="#335c81",
        ecolor="#9cafbf",
        capsize=2,
    )
    axes[1, 0].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 0].set_xticks(np.arange(9), labels, rotation=45)
    axes[1, 0].set_ylabel("Median matrix − identity (MAD)")

    axes[1, 1].errorbar(
        np.arange(3),
        final[9:],
        yerr=median_abs_deviation[9:],
        fmt="o",
        color="#8f4f55",
        ecolor="#c7a3a6",
        capsize=3,
    )
    axes[1, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 1].set_xticks(np.arange(3), ["z", "y", "x"])
    axes[1, 1].set_ylabel("Centered translation (px; MAD)")
    fig.suptitle("638-to-561 affine convergence across Prod2 tiles", fontsize=11)
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(output_json)
    print(output_png)


if __name__ == "__main__":
    main()
