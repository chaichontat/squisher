from __future__ import annotations

from pathlib import Path

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.legacy_runner import run_legacy_script


coarse_preibisch_content_weights = legacy.coarse_preibisch_content_weights
temporary_basic_disk_cache_dir = legacy.temporary_basic_disk_cache_dir


def channel_output_path(output: Path, channel: int) -> Path:
    name = output.name
    if name.endswith(".ome.zarr"):
        return output.with_name(f"{name.removesuffix('.ome.zarr')}.ch{channel}.ome.zarr")
    if name.endswith(".zarr"):
        return output.with_name(f"{name.removesuffix('.zarr')}.ch{channel}.zarr")
    return output.with_name(f"{name}.ch{channel}.zarr")


def channel_output_paths(output: Path, channels: list[int]) -> list[Path]:
    return [channel_output_path(output, channel) for channel in channels]


def fuse_tiles(
    *,
    input_dir: Path,
    position_input: Path,
    registration_input: Path,
    output: Path,
    channels: list[int] | None = None,
    fusion_weight_mode: str = "content-preibisch-coarse",
    batch_size: int = 4,
    basic_cache_tiles: int = 64,
    basic_cache_disk_dir: Path | None = None,
    flatfield_dirs_by_source_view: dict[str, Path] | None = None,
    dry_run: bool = False,
) -> str:
    args = [
        str(input_dir),
        "--position-input",
        str(position_input),
        "--registration-input",
        str(registration_input),
        "--output",
        str(output),
        "--fusion-weight-mode",
        fusion_weight_mode,
        "--batch-size",
        str(batch_size),
        "--basic-cache-tiles",
        str(basic_cache_tiles),
    ]
    if flatfield_dirs_by_source_view is not None:
        for view, flatfield_dir in flatfield_dirs_by_source_view.items():
            args.extend(["--flatfield-dir-by-source-view", f"{view}={flatfield_dir}"])
    if basic_cache_disk_dir is not None:
        args.extend(["--basic-cache-disk-dir", str(basic_cache_disk_dir)])
    if channels is not None:
        args.extend(["--channels", *(str(channel) for channel in channels)])
    if dry_run:
        args.append("--dry-run")
    return run_legacy_script("stitch_20x_tl_multiview.py", args, dry_run=dry_run)
