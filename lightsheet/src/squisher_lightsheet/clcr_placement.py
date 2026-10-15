"""Publish the accepted CL/CR placement as fusion inputs (WORKFLOW_HCR.md step 5).

Each side keeps its own tile geometry. CR is placed in CL space by the metadata centroid, the CR shift that
``align-registered-lr`` baked into the side positions (phase correlation or a reviewed manual shift), and the median
accepted Method8 spatial-overlap local translation. The outputs are one CZYX position file and an identity-affine
registration over a directory holding both sides' deconvolved tiles, plus a no-blend z-median overlay for review.
Every tile keeps its own shape, channels, and tracks, so the two sides may come from acquisitions with different
channel counts.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.placement_common import (
    RECORD_KEYS,
    accepted_method8_translations,
    build_identity_registration_payload,
    common_scale,
    dict_zyx,
    load_side_records,
    read_json,
    record_centers,
    render_zmedian_overlay,
    source_zarr_name,
    translation_stats,
    vector,
    write_json,
)

SIDE_VIEWS = {"CL": "L", "CR": "R"}
























def build_position_payload(
    *,
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    alignment_manifest: dict[str, Any],
    phasecorr: dict[str, Any],
    method8_summary: dict[str, Any],
    deconv_root: Path,
    inputs: dict[str, str],
) -> dict[str, Any]:
    if phasecorr.get("fixed") != "CL" or phasecorr.get("moving") != "CR":
        raise ValueError("Placement phase correlation must be CL-fixed and CR-moving")
    if phasecorr.get("artifact_type") != "squisher_lightsheet.registered_lr_phasecorr.v1":
        raise ValueError(f"Unsupported placement record {phasecorr.get('artifact_type')!r}")
    centers = alignment_manifest.get("metadata_centers_um_zyx")
    if not isinstance(centers, dict) or not {"CL", "CR"} <= set(centers):
        raise ValueError("Alignment manifest lacks CL and CR metadata_centers_um_zyx")
    left_scale = common_scale(left_records, "CL")
    right_scale = common_scale(right_records, "CR")
    if not np.allclose(np.abs(left_scale), np.abs(right_scale), rtol=1e-6, atol=1e-6):
        raise ValueError(f"CL/CR absolute scales differ: {np.abs(left_scale)} versus {np.abs(right_scale)}")

    # align-registered-lr bakes its whole CR shift, phase or manual, into the side positions.
    phase_shift_um = np.asarray(phasecorr["total_shift_to_apply_to_cr_zyx_um"], dtype=np.float64)
    translations = accepted_method8_translations(
        method8_summary,
        {source_zarr_name(record) for record in left_records},
        {source_zarr_name(record) for record in right_records},
        sides=("CL", "CR"),
    )
    # Method8 local translations are in fixed CL voxels, which run along +Z/Y/X on both sides.
    method8_median_um = np.median(translations, axis=0) * np.abs(left_scale)
    left_offset = np.asarray(centers["CL"], dtype=np.float64) - record_centers(left_records)
    right_offset = np.asarray(centers["CR"], dtype=np.float64) - record_centers(right_records)
    net_right_shift = phase_shift_um + method8_median_um

    tiles = []
    for side, records, offset in (("CL", left_records, left_offset), ("CR", right_records, right_offset + net_right_shift)):
        for record in records:
            name = source_zarr_name(record)
            tiles.append(
                {
                    "tile": name,
                    "path": str(deconv_root / name),
                    "side": SIDE_VIEWS[side],
                    "source_side": side,
                    "translation_um": dict_zyx(vector(record, "translation_um") + offset),
                    "scale_um": dict_zyx(vector(record, "scale_um")),
                    "spacing_um": dict_zyx(np.abs(vector(record, "scale_um"))),
                    **{key: deepcopy(record[key]) for key in RECORD_KEYS},
                    "source_tile": str(record.get("source_tile", record["tile"])),
                    "source_path": str(record.get("source_path", record.get("path", ""))),
                }
            )
    tiles.sort(key=lambda record: (record["source_side"], record["tile"]))
    return {
        "schema_version": 1,
        "artifact_type": "squisher_lightsheet.clcr_placement_positions.v1",
        "units": "micrometer",
        "source": (
            "CL/CR side-internal positions; CR placed in CL space by metadata centroid, the align-registered-lr "
            "shift, and median accepted Method8 local translation"
        ),
        "inputs": inputs,
        "alignment": {
            "fixed_side": "CL",
            "moving_side": "CR",
            "align_registered_lr_shift_um_zyx": phase_shift_um.astype(float).tolist(),
            "method8_local_translation_px_zyx": translation_stats(translations),
            "method8_median_shift_um_zyx": method8_median_um.astype(float).tolist(),
            "net_cr_shift_after_centroid_um_zyx": net_right_shift.astype(float).tolist(),
            "cl_global_centroid_offset_um_zyx": left_offset.astype(float).tolist(),
            "cr_global_centroid_plus_alignment_offset_um_zyx": (right_offset + net_right_shift).astype(float).tolist(),
            "method8_status_counts": dict(Counter(row.get("status") for row in method8_summary.get("windows", []))),
        },
        "tiles": tiles,
    }








def publish_clcr_placement(
    *,
    left_position: Path,
    right_position: Path,
    alignment_manifest: Path,
    phasecorr: Path,
    method8_summary: Path,
    deconv_root: Path,
    output_dir: Path,
    qc_channel: int,
    qc_level: int,
) -> dict[str, str]:
    """Write positions.json, registration.json, and the z-median overlay; return their paths."""
    deconv_root = deconv_root.resolve()
    position_payload = build_position_payload(
        left_records=load_side_records(left_position, "CL"),
        right_records=load_side_records(right_position, "CR"),
        alignment_manifest=read_json(alignment_manifest),
        phasecorr=read_json(phasecorr),
        method8_summary=read_json(method8_summary),
        deconv_root=deconv_root,
        inputs={
            "left_position": str(left_position.resolve()),
            "right_position": str(right_position.resolve()),
            "alignment_manifest": str(alignment_manifest.resolve()),
            "phasecorr": str(phasecorr.resolve()),
            "method8_summary": str(method8_summary.resolve()),
            "deconv_root": str(deconv_root),
        },
    )
    missing = [record["path"] for record in position_payload["tiles"] if not Path(record["path"]).exists()]
    if missing:
        raise ValueError(f"{len(missing)} placed tiles are absent from {deconv_root}: {missing[:3]}")
    positions = output_dir / "positions.json"
    registration = output_dir / "registration.json"
    write_json(positions, position_payload)
    write_json(registration, build_identity_registration_payload(
            position_payload,
            deconv_root=deconv_root,
            artifact_type="squisher_lightsheet.clcr_placement_identity_registration.v1",
        ))
    qc = render_zmedian_overlay(
        position_payload,
        output_dir=output_dir / "qc-zmedian",
        channel=qc_channel,
        level=qc_level,
        sides=("CL", "CR"),
        artifact_type="squisher_lightsheet.clcr_placement_zmedian_overlay.v1",
    )
    return {"positions": str(positions), "registration": str(registration), "overlay": qc["outputs"]["overlay"]}
