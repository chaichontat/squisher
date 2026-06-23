from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from squisher_lightsheet.channel_optimization import optimize_405_to_488_translation_mapping
from squisher_lightsheet.legacy_runner import command_text
from squisher_lightsheet.mvs_seams import (
    DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    load_mvs_registration,
    recover_anchor_shifts_from_mvs_seams,
)
from squisher_lightsheet.qc import render_registration_qc

import numpy as np


@dataclass(frozen=True)
class Alignment405To488Outputs:
    level0_refined_anchors: Path
    recovered_anchor_table: Path
    optimized_position: Path
    optimized_registration: Path
    diagnostics: Path


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def _run_script(script_dir: Path, script_name: str, args: list[str], *, dry_run: bool) -> str:
    command = [sys.executable, str(script_dir / script_name), *args]
    text = command_text(command)
    if dry_run:
        print(f"DRY RUN not executed: {text}", flush=True)
        return text
    print(text, flush=True)
    subprocess.run(command, check=True)
    return text


def refine_405_488_level0_anchors(
    *,
    coarse_json: Path,
    anchor_table: Path,
    output_dir: Path,
    native_patch_shape_zyx: tuple[int, int, int] = (12, 320, 320),
    upsample_factor: int = 10,
    workers: int = 1,
    script_dir: Path,
    dry_run: bool = False,
) -> str:
    args = [
        "--coarse-json",
        str(coarse_json),
        "--anchor-table",
        str(anchor_table),
        "--output-dir",
        str(output_dir),
        "--native-patch-shape-zyx",
        *(str(value) for value in native_patch_shape_zyx),
        "--upsample-factor",
        str(upsample_factor),
        "--workers",
        str(workers),
    ]
    return _run_script(script_dir, "refine_405_488_anchors_level0_subpixel.py", args, dry_run=dry_run)


def _load_direct_level0_anchor_shifts(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    payload = json.loads(path.read_text())
    grouped: dict[str, list[dict]] = {}
    for record in payload["records"]:
        if record.get("accepted"):
            grouped.setdefault(str(record["tile"]), []).append(record)
    anchors = {}
    metadata = {}
    for tile, records in grouped.items():
        shifts = np.asarray([record["refined_shift_um_zyx"] for record in records], dtype=float)
        anchors[tile] = np.median(shifts, axis=0)
        metadata[tile] = {
            "tile_site": records[0].get("tile_site"),
            "accepted_level0_block_count": len(records),
            "accepted_level0_blocks": [
                {
                    "coarse_block_index": int(record["coarse_block_index"]),
                    "refined_shift_um_zyx": record["refined_shift_um_zyx"],
                    "residual_shift_level0_px_zyx": record["residual_shift_level0_px_zyx"],
                    "qc_png": record.get("qc_png"),
                }
                for record in records
            ],
        }
    return anchors, metadata


def recover_405_488_mvs_seam_anchors(
    *,
    level0_refined_json: Path,
    mvs_registration: Path,
    output_dir: Path,
    min_quality: float = DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    dry_run: bool = False,
) -> str:
    summary = (
        "recover_405_488_mvs_seam_anchors("
        f"level0_refined_json={level0_refined_json}, "
        f"mvs_registration={mvs_registration}, "
        f"output_dir={output_dir})"
    )
    if dry_run:
        print(f"DRY RUN not executed: {summary}", flush=True)
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    direct_shifts_um, direct_metadata = _load_direct_level0_anchor_shifts(level0_refined_json)
    mvs_payload = load_mvs_registration(mvs_registration)
    spacing_um = np.asarray(
        [
            mvs_payload["spacing_um"]["z"],
            mvs_payload["spacing_um"]["y"],
            mvs_payload["spacing_um"]["x"],
        ],
        dtype=float,
    )
    recovered_shifts_um, diagnostics = recover_anchor_shifts_from_mvs_seams(
        direct_anchor_shift_um_by_tile=direct_shifts_um,
        mvs_registration=mvs_payload,
        spacing_um_zyx=spacing_um,
        min_quality=min_quality,
        used_edges_only=True,
    )
    rows = []
    for record in mvs_payload["tiles"]:
        tile = str(record["tile"])
        if tile in direct_shifts_um:
            rows.append(
                {
                    "tile_site": direct_metadata[tile].get("tile_site"),
                    "tile": tile,
                    "status": "accepted",
                    "source": "direct_405_to_488_level0_refined",
                    "shift_zyx_level3_px": (direct_shifts_um[tile] / spacing_um).tolist(),
                    "shift_um_zyx": direct_shifts_um[tile].tolist(),
                    "direct_anchor": direct_metadata[tile],
                }
            )
        elif tile in recovered_shifts_um:
            rows.append(
                {
                    "tile_site": None,
                    "tile": tile,
                    "status": "recovered",
                    "source": "recovered_from_405_mvs_seam_anchor",
                    "shift_zyx_level3_px": (recovered_shifts_um[tile] / spacing_um).tolist(),
                    "shift_um_zyx": recovered_shifts_um[tile].tolist(),
                    "recovery": {"method": "mvs_pairwise_registration"},
                }
            )
        else:
            rows.append(
                {
                    "tile_site": None,
                    "tile": tile,
                    "status": "unrecovered",
                    "source": "unrecovered",
                    "shift_zyx_level3_px": None,
                    "shift_um_zyx": None,
                    "recovery": {"reason": "not_connected_to_direct_anchor_by_inlier_mvs_seams"},
                }
            )
    anchor_table = output_dir / "merged_405_to_488_anchor_table.json"
    recovery_json = output_dir / "mvs_seam_recovery.json"
    anchor_table.write_text(json.dumps(rows, indent=2) + "\n")
    recovery_json.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.405_to_488_mvs_seam_anchor_recovery.v1",
                "level0_refined_json": str(level0_refined_json.resolve()),
                "mvs_registration": str(mvs_registration.resolve()),
                "min_quality": min_quality,
                "merged_anchor_table": str(anchor_table.resolve()),
                **diagnostics,
            },
            indent=2,
        )
        + "\n"
    )
    return str(anchor_table.resolve())


def optimize_405_to_488_translations(
    *,
    anchor_table: Path,
    phase_position: Path,
    source_405_registration: Path,
    seam_residuals: Path,
    output_position: Path,
    output_registration: Path,
    diagnostics: Path,
    seam_overrides: Path | None = None,
    qc_output_dir: Path | None = None,
    qc_channel: int = 0,
    qc_level: int = 4,
    direct_anchor_sigma_px: str = "0.25,1,1",
    recovered_anchor_sigma_px: str = "1,4,4",
    seam_sigma_px: str = "2,6,6",
    prior_sigma_px: str = "6,24,24",
    min_seam_corr_after: float = 0.45,
    max_seam_shift_px: str = "3,12,12",
    dry_run: bool = False,
) -> str:
    summary = (
        "optimize_405_to_488_translation_mapping("
        f"anchor_table={anchor_table}, "
        f"phase_position={phase_position}, "
        f"source_405_registration={source_405_registration}, "
        f"seam_residuals={seam_residuals}, "
        f"seam_overrides={seam_overrides}, "
        f"output_position={output_position}, "
        f"output_registration={output_registration}, "
        f"diagnostics={diagnostics}); "
        "render_registration_qc("
        f"position_input={output_position}, "
        f"registration_input={output_registration}, "
        f"output_dir={qc_output_dir or output_registration.parent / 'registration-qc'}, "
        f"channel={qc_channel}, "
        f"level={qc_level})"
    )
    if dry_run:
        print(f"DRY RUN not executed: {summary}", flush=True)
        return summary
    qc_dir = qc_output_dir or output_registration.parent / "registration-qc"
    optimize_405_to_488_translation_mapping(
        anchor_table=anchor_table,
        phase_position=phase_position,
        source_405_registration=source_405_registration,
        seam_residuals=seam_residuals,
        seam_overrides=seam_overrides,
        output_position=output_position,
        output_registration=output_registration,
        diagnostics_path=diagnostics,
        direct_anchor_sigma_px=_parse_float_tuple(direct_anchor_sigma_px),
        recovered_anchor_sigma_px=_parse_float_tuple(recovered_anchor_sigma_px),
        seam_sigma_px=_parse_float_tuple(seam_sigma_px),
        prior_sigma_px=_parse_float_tuple(prior_sigma_px),
        min_seam_corr_after=min_seam_corr_after,
        max_seam_shift_px=_parse_float_tuple(max_seam_shift_px),
    )
    render_registration_qc(
        position_input=output_position,
        registration_input=output_registration,
        output_dir=qc_dir,
        channel=qc_channel,
        level=qc_level,
        center_y_xz=True,
        dry_run=False,
    )
    return str(output_position.resolve())


def run_405_to_488_workflow(
    *,
    all_tiles_json: Path,
    coarse_anchor_table: Path,
    phase_position: Path,
    source_405_registration: Path,
    seam_residuals: Path,
    output_dir: Path,
    script_dir: Path,
    dry_run: bool = False,
) -> dict[str, str]:
    level0_dir = output_dir / "level0-direct-anchors"
    recovery_dir = output_dir / "level0-mvs-seam-recovery"
    optimization_dir = output_dir / "global-optimization"
    outputs = Alignment405To488Outputs(
        level0_refined_anchors=level0_dir / "level0_subpixel_refined_405_to_488_anchors.json",
        recovered_anchor_table=recovery_dir / "merged_405_to_488_anchor_table.json",
        optimized_position=optimization_dir / "405.to488.optimized.positions.json",
        optimized_registration=optimization_dir / "405-to488.optimized.registration.json",
        diagnostics=optimization_dir / "optimization_diagnostics.json",
    )
    commands = {
        "refine_level0": refine_405_488_level0_anchors(
            script_dir=script_dir,
            coarse_json=all_tiles_json,
            anchor_table=coarse_anchor_table,
            output_dir=level0_dir,
            dry_run=dry_run,
        ),
        "recover_mvs_seams": recover_405_488_mvs_seam_anchors(
            mvs_registration=source_405_registration,
            level0_refined_json=outputs.level0_refined_anchors,
            output_dir=recovery_dir,
            dry_run=dry_run,
        ),
        "optimize": optimize_405_to_488_translations(
            anchor_table=outputs.recovered_anchor_table,
            phase_position=phase_position,
            source_405_registration=source_405_registration,
            seam_residuals=seam_residuals,
            output_position=outputs.optimized_position,
            output_registration=outputs.optimized_registration,
            diagnostics=outputs.diagnostics,
            dry_run=dry_run,
        ),
    }
    return commands
