from __future__ import annotations

from pathlib import Path

from squisher_lightsheet.legacy_runner import run_legacy_script


def render_registration_qc(
    *,
    position_input: Path,
    registration_input: Path,
    output_dir: Path,
    channel: int = 0,
    level: int = 4,
    center_y_xz: bool = True,
    dry_run: bool = False,
) -> str:
    args = [
        "--position-input",
        str(position_input),
        "--registration-input",
        str(registration_input),
        "--output-dir",
        str(output_dir),
        "--channel",
        str(channel),
        "--level",
        str(level),
    ]
    if center_y_xz:
        args.append("--center-y-xz")
    return run_legacy_script("render_lr_level4_registration_qc.py", args, dry_run=dry_run)
