"""Screen complete-recovery registrations before materialization (WORKFLOW_HCR.md, "Screen the registration").

Materialization and fusion cannot be updated for a few changed windows, so every defect a recovery summary records is
listed before they run: windows rejected for a reason other than the fixed mask (the fusion leaves them uncovered
unless neighbours cover them), quality-gate-rejected native reruns that spatial fallback promoted without the residual
check verifying them, native reruns withheld for departing from the interpolated placement, linear-mapping recoveries
that kept their own fit without two image-comparison planes, and accepted windows whose final transform departs from
their tile's other accepted windows by more than the unverified-rerun departure bound. Each defect names its window
and the level-2 fixed plane through its centre.
Every accepted window that recovery repaired is reviewed too: the combined coverage maps are rendered at the fewest
level-2 planes that pass through every repaired window's core, besides the defect planes and planes spanning the
tissue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from squisher_lightsheet.complete_recovery import DEFAULT_MAXIMUM_UNVERIFIED_RERUN_DEPARTURE_PX
from squisher_lightsheet.fused_fixed_recovery import (
    DEFAULT_SEED_MAXIMUM_MASKED_FRACTION,
    MASK_REJECTION_REASONS,
    MINIMUM_IMAGE_EVIDENCE_PLANES,
    final_transform_departures,
)
from squisher_lightsheet.ngff import scale_translation

ARTIFACT_TYPE = "squisher_lightsheet.registration_screen.v3"
# A window's corner displacement from its tile's decomposed-affine field, judged on the scale of the unverified-rerun
# bound: on 230Tnc 405 it flags exactly the three windows promoted 173-467 px from their neighbours, against 5 px
# (recovery's own outlier tolerance) flagging 319 of 1,508.
DEPARTURE_BOUND_PX = DEFAULT_MAXIMUM_UNVERIFIED_RERUN_DEPARTURE_PX
QC_LEVEL = 2
REPRESENTATIVE_PLANES = 5
# recover-native records outlier_recovery on every window it reran; later stages add their own records, including the
# final linear smoothing that replaces a window's transform without any earlier record (230Tnc CR-405 tile 014).
REPAIR_RECORDS = (
    "outlier_recovery",
    "interpolated_fallback_promotion",
    "linear_mapping_recovery",
    "final_linear_interpolation",
)


def seedable_window_count(summary: Path) -> int:
    """Rejected tissue windows that propose-seeds can seed: not mask-rejected and mostly above the fixed threshold."""
    windows = json.loads(summary.read_text())["windows"]
    return sum(
        row["status"] != "accepted"
        and row.get("rejection_reason") not in MASK_REJECTION_REASONS
        and row.get("fixed_masked_fraction") is not None
        and row["fixed_masked_fraction"] < DEFAULT_SEED_MAXIMUM_MASKED_FRACTION
        for row in windows
    )


def _plane_factor(fixed_fused: Path, level: int) -> float:
    group = zarr.open_group(str(fixed_fused), mode="r")
    _names, base, *_ = scale_translation(group, dataset_index=0)
    _names, scaled, *_ = scale_translation(group, dataset_index=level)
    return float(scaled[-3]) / float(base[-3])


def _defects(row: dict[str, Any], window: dict[str, Any]) -> list[str]:
    if row["status"] != "accepted":
        return [] if row.get("rejection_reason") in MASK_REJECTION_REASONS else ["uncovered"]
    found = []
    promotion = window.get("interpolated_fallback_promotion")
    if promotion is not None:
        if promotion["selected_candidate"] == "native_rerun" and promotion["residual"]["verdict"] != "verified":
            found.append("unverified_native_rerun")
        if promotion.get("native_rerun_withheld_for_departure"):
            found.append("withheld_native_rerun")
    linear = window.get("linear_mapping_recovery")
    if (
        linear is not None
        and linear["selected_candidate"] == "fitted"
        and len(linear["fitted_image_ncc_planes"]) < MINIMUM_IMAGE_EVIDENCE_PLANES
    ):
        found.append("linear_fit_without_image_evidence")
    return found


def stabbing_planes(intervals: list[tuple[int, int]]) -> list[int]:
    """The fewest planes such that every half-open [start, stop) interval contains one (greedy by stop)."""
    planes: list[int] = []
    for start, stop in sorted(intervals, key=lambda interval: interval[1]):
        if not planes or planes[-1] < start:
            planes.append(stop - 1)
    return planes


def screen_recovery(summaries: dict[str, Path], *, fixed_fused: Path) -> dict[str, Any]:
    """List every screened defect and repaired window of each dataset and the QC planes that show them."""
    factor = _plane_factor(fixed_fused, QC_LEVEL)
    plane_count = int(zarr.open_group(str(fixed_fused), mode="r")[str(QC_LEVEL)].shape[-3])

    def plane(level0_z: float) -> int:
        return min(int(level0_z / factor), plane_count - 1)

    defects = []
    repaired = []
    unscored = []
    cores: list[tuple[int, int]] = []
    accepted_z: list[float] = []
    for dataset, summary in summaries.items():
        payload = json.loads(summary.read_text())
        margin = 0.5 * (payload["window_shape_zyx"][0] - payload["core_shape_zyx"][0])
        accepted: dict[Path, tuple[dict[str, Any], dict[str, Any]]] = {}
        for row in payload["windows"]:
            if row["status"] != "accepted" and row.get("rejection_reason") in MASK_REJECTION_REASONS:
                continue
            window = json.loads(Path(row["level0_json"]).read_text())
            start_z, stop_z = window["fixed_start_zyx"][0], window["fixed_stop_zyx"][0]
            centre_z = 0.5 * (start_z + stop_z)
            record = {
                "dataset": dataset,
                "window": row["level0_json"],
                "tile": row["tile"],
                "fixed_start_zyx": window["fixed_start_zyx"],
                "fixed_stop_zyx": window["fixed_stop_zyx"],
                "qc_plane": plane(centre_z),
            }
            if row["status"] == "accepted":
                accepted[Path(row["level0_json"])] = (window, record)
                accepted_z.append(centre_z)
                if any(key in window for key in REPAIR_RECORDS):
                    repaired.append(record | {"selected_attempt": window.get("selected_attempt")})
                    core = (plane(start_z + margin), plane(stop_z - margin - 1) + 1)
                    cores.append(core)
            for kind in _defects(row, window):
                defects.append(
                    record
                    | {
                        "kind": kind,
                        "rejection_reason": row.get("rejection_reason"),
                        "fixed_masked_fraction": row.get("fixed_masked_fraction"),
                    }
                )
        departures = final_transform_departures(
            [(path, window) for path, (window, _record) in accepted.items()], maximum_px=DEPARTURE_BOUND_PX
        )
        for path, departure in departures.items():
            record = accepted[path][1] | {"departure": departure}
            if departure["status"] == "insufficient_support":
                unscored.append(record)
            elif departure["departs"]:
                defects.append(record | {"kind": "neighbour_departure"})
    if not accepted_z:
        raise ValueError(f"No accepted windows in {sorted(summaries)}")
    tissue = {plane(z) for z in np.linspace(min(accepted_z), max(accepted_z), REPRESENTATIVE_PLANES)}
    planes = sorted({defect["qc_plane"] for defect in defects} | set(stabbing_planes(cores)) | tissue)
    counts: dict[str, int] = {}
    for defect in defects:
        counts[defect["kind"]] = counts.get(defect["kind"], 0) + 1
    return {
        "artifact_type": ARTIFACT_TYPE,
        "summaries": {dataset: str(path) for dataset, path in summaries.items()},
        "fixed_fused": str(fixed_fused),
        "qc_level": QC_LEVEL,
        "defect_counts": counts,
        "defects": defects,
        "departure_bound_px": DEPARTURE_BOUND_PX,
        "departure_unscored": unscored,
        "repaired_count": len(repaired),
        "repaired": repaired,
        "qc_planes": planes,
    }
