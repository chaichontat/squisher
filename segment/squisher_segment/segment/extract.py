from __future__ import annotations

import logging
from pathlib import Path

import click
from loguru import logger

from squisher_segment.segment.extract_core import (
    _is_zarr_path,
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
    enrich_boundaries: Path | None,
    aux_channel_stack: Path | None = None,
) -> None:
    """Extract segmentation candidate TIFFs from one registered TIFF or Zarr volume."""
    mode = mode.lower().strip()
    if mode not in {"z", "ortho", "maxproj"}:
        raise click.BadParameter("Mode must be 'z', 'ortho', or 'maxproj'.")

    input_path = input_path.resolve()
    if input_path.is_dir() and not _is_zarr_path(input_path):
        raise click.BadParameter("Input directory must be a .zarr store.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    use_zarr = _is_zarr_path(input_path)
    upscale_value = normalize_numeric_options(
        mode=mode,
        dz=dz,
        anisotropy=anisotropy,
        upscale=upscale,
        use_zarr=use_zarr,
        has_max_from=False,
        ortho_anisotropy_default=6,
    )

    out_dir = out if out is not None else input_path.parent / "segment_extract"
    if out_dir.is_file():
        raise click.BadParameter("--out must point to a directory, not a file.")
    out_dir.mkdir(parents=True, exist_ok=True)

    label_value = label or input_path.stem
    logger.info(f"[{label_value}] Input: {input_path}")
    logger.info(f"[{label_value}] Output: {out_dir}")
    logger.info(f"[{label_value}] Upscale factor: {upscale_value}")

    run_single_file_extract(
        mode=mode,
        registered=input_path,
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
    )
