from __future__ import annotations

from pathlib import Path

from squisher_lightsheet.legacy_runner import run_legacy_script


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
    content_preibisch_sigma1: int = 7,
    content_preibisch_sigma2: int = 17,
    content_preibisch_coarse_stride: tuple[int, int, int] = (1, 8, 8),
    basic_cache_tiles: int = 128,
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
        "--content-preibisch-sigma1",
        str(content_preibisch_sigma1),
        "--content-preibisch-sigma2",
        str(content_preibisch_sigma2),
        "--content-preibisch-coarse-stride",
        *(str(value) for value in content_preibisch_coarse_stride),
        "--basic-cache-tiles",
        str(basic_cache_tiles),
    ]
    if channels is not None:
        args.extend(["--channels", *(str(channel) for channel in channels)])
    if dry_run:
        args.append("--dry-run")
    return run_legacy_script("stitch_20x_tl_multiview.py", args, dry_run=dry_run)
