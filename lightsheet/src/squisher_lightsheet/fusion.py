from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from squisher.jpegxr_zarr import DEFAULT_JPEGXR_LEVEL
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.legacy_runner import run_legacy_script


coarse_preibisch_content_weights = legacy.coarse_preibisch_content_weights
temporary_basic_disk_cache_dir = legacy.temporary_basic_disk_cache_dir
DEFAULT_OUTPUT_CHUNKSIZE_ZYX = (12, 960, 960)
OutputCodec = Literal["auto", "zstd", "jpegxr"]


def _materialization_level_factor_zyx(position_input: Path) -> tuple[int, int, int]:
    if not position_input.is_file():
        return (1, 1, 1)
    payload = json.loads(position_input.read_text())
    grid = payload.get("materialization_grid")
    if grid is None:
        return (1, 1, 1)
    factors = grid.get("level_factor_zyx") if isinstance(grid, dict) else None
    if (
        not isinstance(factors, list)
        or len(factors) != 3
        or any(not isinstance(value, int) or value < 1 for value in factors)
    ):
        raise ValueError(
            f"{position_input} materialization_grid.level_factor_zyx must contain three positive integers"
        )
    return tuple(factors)


def resolve_fusion_output_codec(
    *,
    position_input: Path,
    fusion_level: int,
    output_codec: OutputCodec,
) -> Literal["zstd", "jpegxr"]:
    """Resolve the standard codec from the actual output resolution contract."""
    materialization_factors = _materialization_level_factor_zyx(position_input)
    native_source = materialization_factors == (1, 1, 1) and fusion_level == 0
    if output_codec == "auto":
        return "jpegxr" if native_source else "zstd"
    return output_codec


def canonical_fusion_base_output(output: Path) -> Path:
    """Return the base OME-Zarr path that yields canonical per-channel outputs."""
    name = output.name
    if name.endswith(".ome.zarr") or name.endswith(".zarr"):
        return output
    return output / "fused.ome.zarr"


def channel_output_path(output: Path, channel: int) -> Path:
    return legacy.channel_output_path(canonical_fusion_base_output(output), channel, separate_channels=True)


def channel_output_paths(output: Path, channels: list[int]) -> list[Path]:
    return [channel_output_path(output, channel) for channel in channels]


def _load_basic_sampling_manifest(flatfield_dir: Path) -> dict:
    manifests = sorted(flatfield_dir.glob("*-sampling.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"{flatfield_dir} must contain exactly one BaSiC *-sampling.json manifest; found {len(manifests)}"
        )
    with manifests[0].open() as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifests[0]} must contain a JSON object")
    return manifest


def validate_source_view_flatfields(flatfield_dirs_by_source_view: dict[str, Path]) -> None:
    for view, flatfield_dir in flatfield_dirs_by_source_view.items():
        if "pooled" in flatfield_dir.name.lower():
            raise ValueError(
                f"source_view={view} uses pooled BaSiC directory {flatfield_dir}; "
                "use a separate sorted BaSiC profile for each source view"
            )
        manifest = _load_basic_sampling_manifest(flatfield_dir)
        if manifest.get("sort_intensity") is not True:
            raise ValueError(
                f"source_view={view} BaSiC manifest in {flatfield_dir} was not built with sort_intensity=true"
            )
        input_dirs = manifest.get("input_dirs")
        if not isinstance(input_dirs, list) or len(input_dirs) != 1:
            raise ValueError(
                f"source_view={view} BaSiC manifest in {flatfield_dir} must record "
                "exactly one input_dir; pooled L/R profiles are not allowed"
            )


def fuse_tiles(
    *,
    input_dir: Path,
    position_input: Path,
    registration_input: Path,
    output: Path,
    channels: list[int] | None = None,
    fusion_level: int = 0,
    fusion_weight_mode: str = "content-preibisch-coarse",
    batch_size: int = 1,
    basic_cache_tiles: int = 64,
    basic_cache_disk_dir: Path | None = None,
    output_chunksize_zyx: tuple[int, int, int] = DEFAULT_OUTPUT_CHUNKSIZE_ZYX,
    output_grid_template: Path | None = None,
    output_grid_template_level: int = 0,
    output_codec: OutputCodec = "auto",
    zstd_level: int = 3,
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
    flatfield_dirs_by_source_view: dict[str, Path] | None = None,
    resume_fusion: bool = False,
    dry_run: bool = False,
) -> str:
    output = canonical_fusion_base_output(output)
    resolved_output_codec = resolve_fusion_output_codec(
        position_input=position_input,
        fusion_level=fusion_level,
        output_codec=output_codec,
    )
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
        "--fusion-level",
        str(fusion_level),
        "--batch-size",
        str(batch_size),
        "--basic-cache-tiles",
        str(basic_cache_tiles),
        "--jpegxr-level",
        str(jpegxr_level),
        "--output-codec",
        resolved_output_codec,
        "--zstd-level",
        str(zstd_level),
    ]
    args.extend(["--output-chunksize", *(str(value) for value in output_chunksize_zyx)])
    if output_grid_template is not None:
        args.extend(["--output-grid-template", str(output_grid_template)])
        args.extend(["--output-grid-template-level", str(output_grid_template_level)])
    if flatfield_dirs_by_source_view is not None:
        validate_source_view_flatfields(flatfield_dirs_by_source_view)
        for view, flatfield_dir in flatfield_dirs_by_source_view.items():
            args.extend(["--flatfield-dir-by-source-view", f"{view}={flatfield_dir}"])
    if basic_cache_disk_dir is not None:
        args.extend(["--basic-cache-disk-dir", str(basic_cache_disk_dir)])
    if channels is not None:
        args.extend(["--channels", *(str(channel) for channel in channels)])
    if resume_fusion:
        args.append("--resume-fusion")
    if dry_run:
        args.append("--dry-run")
    return run_legacy_script("stitch_20x_tl_multiview.py", args, dry_run=dry_run)
