from __future__ import annotations

from pathlib import Path

from squisher_lightsheet.channel_alignment import (
    optimize_405_to_488_translations,
    recover_405_488_mvs_seam_anchors,
    refine_405_488_level0_anchors,
    run_405_to_488_workflow,
)


def test_refine_level0_command_uses_explicit_script_dir() -> None:
    command = refine_405_488_level0_anchors(
        coarse_json=Path("/run/all_tiles.json"),
        anchor_table=Path("/run/coarse_anchor_table.json"),
        output_dir=Path("/run/level0"),
        script_dir=Path("/scripts"),
        dry_run=True,
    )

    assert "/scripts/refine_405_488_anchors_level0_subpixel.py" in command
    assert "--coarse-json /run/all_tiles.json" in command
    assert "--anchor-table /run/coarse_anchor_table.json" in command
    assert "--native-patch-shape-zyx 12 320 320" in command


def test_recover_mvs_seams_uses_mvs_registration() -> None:
    command = recover_405_488_mvs_seam_anchors(
        level0_refined_json=Path("/run/level0/anchors.json"),
        mvs_registration=Path("/run/405.mvs.registration.json"),
        output_dir=Path("/run/recovery"),
        dry_run=True,
    )

    assert "recover_405_488_mvs_seam_anchors(" in command
    assert "level0_refined_json=/run/level0/anchors.json" in command
    assert "mvs_registration=/run/405.mvs.registration.json" in command


def test_optimize_uses_native_package_function_not_script_dir() -> None:
    command = optimize_405_to_488_translations(
        anchor_table=Path("/run/anchors.json"),
        phase_position=Path("/run/405.positions.json"),
        source_405_registration=Path("/run/source.registration.json"),
        seam_residuals=Path("/run/405.mvs.registration.json"),
        output_position=Path("/run/out.positions.json"),
        output_registration=Path("/run/out.registration.json"),
        diagnostics=Path("/run/diagnostics.json"),
        dry_run=True,
    )

    assert "optimize_405_to_488_translation_mapping(" in command
    assert "render_registration_qc(" in command
    assert "output_dir=/run/registration-qc" in command
    assert "optimize_405_to_488_translation_mapping.py" not in command
    assert "script_dir" not in command


def test_run_workflow_builds_refine_recover_optimize_commands() -> None:
    commands = run_405_to_488_workflow(
        all_tiles_json=Path("/run/all_tiles.json"),
        coarse_anchor_table=Path("/run/coarse_anchor_table.json"),
        phase_position=Path("/run/405.positions.json"),
        source_405_registration=Path("/run/source.registration.json"),
        seam_residuals=Path("/run/source.registration.json"),
        output_dir=Path("/run/align405"),
        script_dir=Path("/scripts"),
        dry_run=True,
    )

    assert set(commands) == {"refine_level0", "recover_mvs_seams", "optimize"}
    assert "/scripts/refine_405_488_anchors_level0_subpixel.py" in commands["refine_level0"]
    assert "/run/align405/level0-direct-anchors" in commands["refine_level0"]
    assert "/run/align405/level0-mvs-seam-recovery" in commands["recover_mvs_seams"]
    assert "anchor_table=/run/align405/level0-mvs-seam-recovery/merged_405_to_488_anchor_table.json" in commands[
        "optimize"
    ]
    assert "/run/align405/global-optimization/registration-qc" in commands["optimize"]
