from __future__ import annotations

import logging
from pathlib import Path

import click
from loguru import logger

from squisher_segment.segment.extract_core import (
    _is_random_access_volume_path,
    _stage_registered_volume,
    normalize_numeric_options,
    run_single_file_extract,
)


def run_extract(
    input_path: Path,
    *,
    mode: str,
    out: Path | None,
    dz: int,
    n: int,
    z_crops_per_file: int,
    anisotropy: int,
    channels: str | None,
    crop: int,
    threads: int,
    upscale: float | None,
    seed: int | None,
    label: str | None,
    masks: Path | None,
    stage: Path | None,
    enrich_boundaries: Path | None,
    aux_channel_stack: Path | None = None,
    ortho_depth: int | None = None,
) -> None:
    """Extract segmentation candidate TIFFs from one registered TIFF or Zarr volume."""
    mode = mode.lower().strip()
    if mode not in {"z", "ortho", "maxproj"}:
        raise click.BadParameter("Mode must be 'z', 'ortho', or 'maxproj'.")

    source_path = input_path.resolve()
    if source_path.is_dir() and not _is_random_access_volume_path(source_path):
        raise click.BadParameter("Input directory must be a .zarr store.")
    if not source_path.exists():
        raise FileNotFoundError(f"Input not found: {source_path}")

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    out_dir = out if out is not None else source_path.parent / "segment_extract"
    label_value = label or source_path.stem
    registered_path = source_path
    if stage is not None:
        registered_path = _stage_registered_volume(source_path, stage.resolve())

    use_zarr = _is_random_access_volume_path(registered_path)
    if ortho_depth is not None:
        if mode != "ortho" or not use_zarr:
            raise click.BadParameter("--ortho-depth is only valid for ortho extraction from Zarr input.")
        if ortho_depth < 1:
            raise click.BadParameter("--ortho-depth must be a positive integer.")
        if enrich_boundaries is not None:
            raise click.BadParameter("--ortho-depth cannot be combined with --enrich-boundaries.")
    upscale_value = normalize_numeric_options(
        mode=mode,
        dz=dz,
        anisotropy=anisotropy,
        upscale=upscale,
        use_zarr=use_zarr,
        has_max_from=False,
        ortho_anisotropy_default=6,
    )

    if out_dir.is_file():
        raise click.BadParameter("--out must point to a directory, not a file.")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{label_value}] Input: {source_path}")
    if registered_path != source_path:
        logger.info(f"[{label_value}] Staged input: {registered_path}")
    logger.info(f"[{label_value}] Output: {out_dir}")
    logger.info(f"[{label_value}] Upscale factor: {upscale_value}")

    run_single_file_extract(
        mode=mode,
        registered=registered_path,
        out=out_dir,
        dz=dz,
        n=n,
        z_crops_per_file=z_crops_per_file,
        anisotropy=anisotropy,
        channels=channels,
        crop=crop,
        threads=threads,
        upscale=upscale_value,
        seed=seed,
        max_from_path=None,
        aux_channel_stack=aux_channel_stack,
        label=label_value,
        masks=masks,
        enrich_boundaries=enrich_boundaries,
        ortho_depth=ortho_depth,
    )
