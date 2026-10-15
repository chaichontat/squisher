"""TL/TR cross-dataset placement (WORKFLOW_HCR.md step 5, ``tltr_x_join_center_z_phase``).

TL is fixed and TR moving; the sides join along X without the CL/CR flips. Each side keeps its own registered tile
geometry and is centred on its metadata placement (``lightsheet position --mode tltr_x_join_center_z_phase``, which
abuts TR beside TL with an assumed overlap). TR's shift then comes from a reviewed centre-Z slab phase correlation of
the two sides (``lightsheet rough-phase``) or an author's manual shift, plus the median accepted Method8
spatial-overlap local translation. Unlike ``run-tltr``, the sides are never jointly re-registered.

HCR Z spacing is the nominal 1.8 um step on both sides; recorded values (1.807 and 1.8065 um on 230Ptprz1) differ
by metadata noise, so placement keeps each tile's recorded scale without resampling.
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

SIDES = ("TL", "TR")
SIDE_VIEWS = {"TL": "L", "TR": "R"}
# rough-phase reads the join axis to phase-correlate a centre-Z slab rather than the CL/CR XZ seam.
JOIN_DIAGNOSTICS = {"join_axis": "x", "right_flip_axes": []}


def _stem(name: str) -> str:
    return Path(name).name.removesuffix(".ome.zarr").removesuffix(".ome.tif")


def join_sides(
    *,
    metadata: dict[str, Any],
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    right_shift_zyx_um: np.ndarray,
    excluded: dict[str, set[str]],
) -> dict[str, Any]:
    """Centre each side's registered tiles on its metadata placement and shift TR by ``right_shift_zyx_um``.

    ``excluded`` names each side's tiles that native registration excluded (and the author accepted); every other
    metadata tile must be registered, and centroids use the registered tiles only.
    """
    tiles = []
    offsets = {}
    for side, records in (("TL", left_records), ("TR", right_records)):
        view = SIDE_VIEWS[side]
        by_stem = {_stem(record["tile"]): record for record in records}
        accepted_out = {_stem(name) for name in excluded[side]}
        metadata_side = {_stem(record["tile"]): record for record in metadata["tiles"] if record["side"] == view}
        if set(by_stem) | accepted_out != set(metadata_side) or set(by_stem) & accepted_out:
            raise ValueError(
                f"{side} metadata tiles {sorted(metadata_side)} are not its registered tiles {sorted(by_stem)} plus "
                f"its excluded tiles {sorted(accepted_out)}"
            )
        placed = {stem: record for stem, record in metadata_side.items() if stem in by_stem}
        # Metadata records carry no shape; each tile's registered record supplies it.
        target = record_centers(
            [
                by_stem[stem] | {"translation_um": record["translation_um"], "scale_um": record["scale_um"]}
                for stem, record in placed.items()
            ]
        )
        offset = target - record_centers(records)
        if side == "TR":
            offset = offset + right_shift_zyx_um
        offsets[side] = offset
        for record in records:
            tiles.append(
                {
                    "tile": source_zarr_name(record),
                    "path": record["path"],
                    "side": view,
                    "source_side": side,
                    "translation_um": dict_zyx(vector(record, "translation_um") + offset),
                    "scale_um": dict_zyx(vector(record, "scale_um")),
                    "spacing_um": dict_zyx(np.abs(vector(record, "scale_um"))),
                    **{key: record[key] for key in RECORD_KEYS},
                    "source_tile": str(record.get("source_tile", record["tile"])),
                    "source_path": str(record.get("source_path", record["path"])),
                }
            )
    return {
        "schema_version": 1,
        "artifact_type": "lightsheet.position.v1",
        "units": "micrometer",
        "source": "TL/TR side-internal positions centred on metadata placement; TR shifted by right_shift_zyx_um",
        "diagnostics": {
            **JOIN_DIAGNOSTICS,
            "right_shift_zyx_um": np.asarray(right_shift_zyx_um, dtype=np.float64).tolist(),
            "centroid_offsets_um_zyx": {side: offset.tolist() for side, offset in offsets.items()},
        },
        "tiles": tiles,
    }


def coarse_right_shift(coarse: dict[str, Any]) -> np.ndarray:
    """TR's whole coarse shift after centring: the joined shift plus any rough-phase shift applied on top."""
    diagnostics = coarse["diagnostics"]
    shift = np.asarray(diagnostics["right_shift_zyx_um"], dtype=np.float64)
    phases = [value for key, value in diagnostics.items() if key.endswith("_phase_alignment")]
    if len(phases) > 1:
        raise ValueError(f"Coarse TL/TR positions record {len(phases)} phase alignments; expected at most one")
    if phases:
        phase = phases[0]
        if "shift_to_apply_R_um_zyx" not in phase:
            raise ValueError("TL/TR rough phase must be a centre-Z slab (ZYX) correlation")
        shift = shift + np.asarray(phase["shift_to_apply_R_um_zyx"], dtype=np.float64)
    return shift


def build_position_payload(
    *, coarse: dict[str, Any], method8_summary: dict[str, Any], deconv_root: Path, inputs: dict[str, str],
    global_affine: np.ndarray | None = None,
) -> dict[str, Any]:
    left = [record for record in coarse["tiles"] if record["source_side"] == "TL"]
    right = [record for record in coarse["tiles"] if record["source_side"] == "TR"]
    translations = accepted_method8_translations(
        method8_summary,
        {record["tile"] for record in left},
        {record["tile"] for record in right},
        sides=SIDES,
    )
    # Method8 local translations are in fixed TL voxels; the sides share orientation, so no axis flips.
    method8_median_um = np.median(translations, axis=0) * np.abs(vector(left[0], "scale_um"))
    tiles = []
    for record in left + right:
        translation = vector(record, "translation_um")
        if record["source_side"] == "TR" and global_affine is None:
            translation = translation + method8_median_um
        tiles.append(
            record
            | {
                "path": str(deconv_root / record["tile"]),
                "translation_um": dict_zyx(translation),
            }
        )
    tiles.sort(key=lambda record: (record["source_side"], record["tile"]))
    coarse_shift = coarse_right_shift(coarse)
    return {
        "schema_version": 1,
        "artifact_type": "squisher_lightsheet.tltr_placement_positions.v1",
        "units": "micrometer",
        "source": (
            "TL/TR side-internal positions; TR placed in TL space by metadata centroid, the reviewed phase or manual "
            + ("shift, and median accepted Method8 local translation" if global_affine is None else
             "shift; reviewed global affine is stored separately from source stage placement")
        ),
        "inputs": inputs,
        "alignment": {
            "placement_model": "median_translation" if global_affine is None else "global_affine",
            **({"global_affine_zyx_um": global_affine.tolist()} if global_affine is not None else {}),
            "fixed_side": "TL",
            "moving_side": "TR",
            "coarse_right_shift_um_zyx": coarse_shift.tolist(),
            "method8_local_translation_px_zyx": translation_stats(translations),
            "method8_median_shift_um_zyx": method8_median_um.tolist(),
            "net_right_shift_after_centroid_um_zyx": (coarse_shift + (method8_median_um if global_affine is None else 0)).tolist(),
            "method8_status_counts": dict(Counter(row.get("status") for row in method8_summary.get("windows", []))),
        },
        "tiles": tiles,
    }


def load_reviewed_global_affine(path: Path, *, coarse: Path, method8_summary: Path) -> np.ndarray:
    """Validate Method8's world-coordinate affine against the exact coarse side records."""
    candidate = read_json(path)
    if candidate.get("artifact_type") != "lightsheet.spatial_overlap_global_affine.v1" or candidate.get(
        "coordinate_contract"
    ) != "target_physical_zyx_um = matrix_zyx @ source_physical_zyx_um + translation_zyx_um":
        raise ValueError("Reviewed affine must be a Method8 physical ZYX source-to-target affine")
    payload = read_json(coarse)
    for side, key in (("TL", "target_position"), ("TR", "source_position")):
        records = read_json(Path(candidate[key]))["tiles"]
        expected = [r for r in payload["tiles"] if r["source_side"] == side]
        if records != expected:
            raise ValueError(f"Reviewed affine {side} inputs differ from current coarse placement")
    matrix = np.asarray(candidate["matrix_zyx"], dtype=np.float64)
    translation = np.asarray(candidate["translation_zyx_um"], dtype=np.float64)
    if matrix.shape != (3, 3) or translation.shape != (3,) or not (
        np.all(np.isfinite(matrix)) and np.all(np.isfinite(translation))
    ):
        raise ValueError("Reviewed affine must have a finite 3x3 matrix and ZYX translation")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if np.linalg.det(matrix) <= 0 or singular.min() < 1 / 1.1 or singular.max() > 1.1:
        raise ValueError("Reviewed TL/TR affine must preserve orientation and remain within 10% scale")
    accepted = sum(row.get("status") == "accepted" for row in read_json(method8_summary)["windows"])
    if candidate["fit"]["accepted_correspondence_count"] != accepted:
        raise ValueError("Reviewed affine correspondence count differs from current Method8 summary")
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = matrix, translation
    return result


def build_placement_registration(payload: dict[str, Any], *, deconv_root: Path) -> dict[str, Any]:
    registration = build_identity_registration_payload(
        payload, deconv_root=deconv_root, artifact_type=("squisher_lightsheet.tltr_placement_registration.v1"
            if payload["alignment"].get("global_affine_zyx_um") is not None
            else "squisher_lightsheet.tltr_placement_identity_registration.v1")
    )
    affine = payload["alignment"].get("global_affine_zyx_um")
    if affine is not None:
        for position, record in zip(payload["tiles"], registration["tiles"], strict=True):
            if position["source_side"] == "TR":
                record["registered_affine"]["matrix"] = deepcopy(affine)
        registration["metrics"]["registered_affine_note"] = (
            "TL identity; TR reviewed world affine applied to coarse stage placement, without median fine shift."
        )
        registration["metrics"]["inputs"] = deepcopy(payload["inputs"])
    return registration


def split_sides(coarse: Path, output_dir: Path) -> dict[str, Path]:
    """One position file per side, for Method8's fixed and moving inputs."""
    payload = read_json(coarse)
    outputs = {}
    for side in SIDES:
        outputs[side] = output_dir / f"{side.lower()}.positions.json"
        write_json(outputs[side], payload | {"tiles": [r for r in payload["tiles"] if r["source_side"] == side]})
    return outputs


def _excluded_tiles(position: Path) -> set[str]:
    """Tiles native registration excluded as disconnected, as recorded in its registered positions."""
    connectivity = read_json(position).get("registration_run", {}).get("connectivity", {})
    return set(connectivity.get("excluded_tiles", []))


def write_joined_sides(
    *, metadata: Path, left_position: Path, right_position: Path, right_shift_zyx_um: np.ndarray, output: Path
) -> None:
    write_json(
        output,
        join_sides(
            metadata=read_json(metadata),
            left_records=load_side_records(left_position, "TL"),
            right_records=load_side_records(right_position, "TR"),
            right_shift_zyx_um=np.asarray(right_shift_zyx_um, dtype=np.float64),
            excluded={"TL": _excluded_tiles(left_position), "TR": _excluded_tiles(right_position)},
        ),
    )


def render_coarse_overlay(coarse: Path, output_dir: Path, *, channel: int, level: int) -> dict[str, Any]:
    """The z-median overlay of a coarse TL/TR placement, for reviewing a manual shift before Method8."""
    return render_zmedian_overlay(
        read_json(coarse),
        output_dir=output_dir,
        channel=channel,
        level=level,
        sides=SIDES,
        artifact_type="squisher_lightsheet.tltr_coarse_zmedian_overlay.v1",
    )


def publish_tltr_placement(
    *, coarse: Path, method8_summary: Path, deconv_root: Path, output_dir: Path, qc_channel: int, qc_level: int,
    reviewed_global_affine: Path | None = None,
) -> dict[str, str]:
    """Write positions.json, registration.json, and the z-median overlay; return their paths."""
    deconv_root = deconv_root.resolve()
    affine = (load_reviewed_global_affine(reviewed_global_affine, coarse=coarse, method8_summary=method8_summary)
              if reviewed_global_affine is not None else None)
    inputs = {"coarse": str(coarse.resolve()), "method8_summary": str(method8_summary.resolve())}
    if reviewed_global_affine is not None:
        from squisher_lightsheet.workflow_runner import sha256_file
        inputs |= {"reviewed_global_affine": str(reviewed_global_affine.resolve()),
                   "reviewed_global_affine_sha256": sha256_file(reviewed_global_affine)}
    payload = build_position_payload(
        coarse=read_json(coarse),
        method8_summary=read_json(method8_summary),
        deconv_root=deconv_root,
        inputs=inputs, global_affine=affine,
    )
    missing = [record["path"] for record in payload["tiles"] if not Path(record["path"]).exists()]
    if missing:
        raise ValueError(f"{len(missing)} placed tiles are absent from {deconv_root}: {missing[:3]}")
    positions = output_dir / "positions.json"
    registration = output_dir / "registration.json"
    write_json(positions, payload)
    write_json(
        registration,
        build_placement_registration(payload, deconv_root=deconv_root),
    )
    if affine is not None:
        from squisher_lightsheet.legacy_runner import run_legacy_script
        qc_dir = output_dir / "qc-affine"
        run_legacy_script("render_lr_level4_registration_qc.py", [
            "--position-input", str(positions), "--registration-input", str(registration),
            "--output-dir", str(qc_dir), "--channel", str(qc_channel), "--level", str(qc_level),
            "--full-affine-planes", "--center-y-xz", "--skip-global-projections",
        ])
        overlay = qc_dir / f"level{qc_level}_registered_lr_fullAffine_planes_yellowOverlay_ch{qc_channel}.png"
        return {"positions": str(positions), "registration": str(registration), "overlay": str(overlay)}
    qc = render_zmedian_overlay(
        payload,
        output_dir=output_dir / "qc-zmedian",
        channel=qc_channel,
        level=qc_level,
        sides=SIDES,
        artifact_type="squisher_lightsheet.tltr_placement_zmedian_overlay.v1",
    )
    return {"positions": str(positions), "registration": str(registration), "overlay": qc["outputs"]["overlay"]}
