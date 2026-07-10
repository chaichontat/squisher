from __future__ import annotations

import json
from pathlib import Path

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.legacy_runner import run_legacy_script


coarse_preibisch_content_weights = legacy.coarse_preibisch_content_weights
temporary_basic_disk_cache_dir = legacy.temporary_basic_disk_cache_dir


def canonical_fusion_base_output(output: Path) -> Path:
    """Return the base OME-Zarr path that yields canonical per-channel outputs."""
    name = output.name
    if name.endswith(".ome.zarr") or name.endswith(".zarr"):
        return output
    return output / "fused.ome.zarr"


def channel_output_path(output: Path, channel: int) -> Path:
    output = canonical_fusion_base_output(output)
    name = output.name
    if name.endswith(".ome.zarr"):
        return output.with_name(f"{name.removesuffix('.ome.zarr')}.ch{channel}.ome.zarr")
    if name.endswith(".zarr"):
        return output.with_name(f"{name.removesuffix('.zarr')}.ch{channel}.zarr")
    return output.with_name(f"{name}.ch{channel}.zarr")


def channel_output_paths(output: Path, channels: list[int]) -> list[Path]:
    return [channel_output_path(output, channel) for channel in channels]


def _load_basic_sampling_manifest(flatfield_dir: Path) -> dict:
    manifests = sorted(flatfield_dir.glob("*-sampling.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"{flatfield_dir} must contain exactly one BaSiC *-sampling.json manifest; "
            f"found {len(manifests)}"
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
                f"source_view={view} BaSiC manifest in {flatfield_dir} was not built "
                "with sort_intensity=true"
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
    batch_size: int = 4,
    basic_cache_tiles: int = 64,
    basic_cache_disk_dir: Path | None = None,
    output_chunksize_zyx: tuple[int, int, int] | None = None,
    output_grid_template: Path | None = None,
    flatfield_dirs_by_source_view: dict[str, Path] | None = None,
    dry_run: bool = False,
) -> str:
    output = canonical_fusion_base_output(output)
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
    ]
    if output_chunksize_zyx is not None:
        args.extend(["--output-chunksize", *(str(value) for value in output_chunksize_zyx)])
    if output_grid_template is not None:
        args.extend(["--output-grid-template", str(output_grid_template)])
    if flatfield_dirs_by_source_view is not None:
        validate_source_view_flatfields(flatfield_dirs_by_source_view)
        for view, flatfield_dir in flatfield_dirs_by_source_view.items():
            args.extend(["--flatfield-dir-by-source-view", f"{view}={flatfield_dir}"])
    if basic_cache_disk_dir is not None:
        args.extend(["--basic-cache-disk-dir", str(basic_cache_disk_dir)])
    if channels is not None:
        args.extend(["--channels", *(str(channel) for channel in channels)])
    if dry_run:
        args.append("--dry-run")
    return run_legacy_script("stitch_20x_tl_multiview.py", args, dry_run=dry_run)
