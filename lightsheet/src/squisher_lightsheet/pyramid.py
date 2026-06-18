from __future__ import annotations

from pathlib import Path

from squisher_lightsheet.legacy_runner import run_legacy_script


def add_pyramids(*, ome_zarrs: list[Path], template: Path | None = None, dry_run: bool = False) -> str:
    args = []
    if template is not None:
        args.extend(["--template", str(template)])
    args.extend(str(path) for path in ome_zarrs)
    return run_legacy_script("add_ome_zarr_pyramid.py", args, dry_run=dry_run)
