import glob
import os
from pathlib import Path
from typing import Annotated

import typer

from squisher.compare import compare_czi_compression
from squisher.compression import (
    DEFAULT_MIN_ZARR_CHUNK_PIXELS,
    DEFAULT_ZARR_CHUNKS_TCZYX,
    compress_czi_to_ome_tiff,
    verify_czi_ome_tiff_outputs,
)
from squisher.logging import setup_cli_logging


app = typer.Typer(no_args_is_help=True)


@app.callback()
def callback() -> None:
    setup_cli_logging()


@app.command()
def compress(
    paths: Annotated[list[Path], typer.Argument(help="One or more CZI files or glob patterns.")],
    out_dir: Annotated[Path | None, typer.Option("--out-dir")] = None,
    output_format: Annotated[str, typer.Option("--output-format")] = "ome-tiff",
    level: Annotated[float, typer.Option("--level", min=0.0, max=100.0)] = 0.7,
    tile_size: Annotated[int, typer.Option("--tile-size")] = 512,
    zarr_chunk_z: Annotated[int, typer.Option("--zarr-chunk-z", min=1)] = DEFAULT_ZARR_CHUNKS_TCZYX[2],
    zarr_chunk_y: Annotated[int, typer.Option("--zarr-chunk-y", min=1)] = DEFAULT_ZARR_CHUNKS_TCZYX[3],
    zarr_chunk_x: Annotated[int, typer.Option("--zarr-chunk-x", min=1)] = DEFAULT_ZARR_CHUNKS_TCZYX[4],
    min_zarr_chunk_pixels: Annotated[int, typer.Option("--min-zarr-chunk-pixels", min=1)] = (
        DEFAULT_MIN_ZARR_CHUNK_PIXELS
    ),
    zarr_compressor: Annotated[str, typer.Option("--zarr-compressor")] = "jpegxr",
    tiff_maxworkers: Annotated[int, typer.Option("--tiff-maxworkers")] = 8,
    czi_tile_workers: Annotated[int, typer.Option("--czi-tile-workers")] = 8,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    thumbnails: Annotated[bool, typer.Option("--thumbnails/--no-thumbnails")] = True,
    thumbnail_size: Annotated[int, typer.Option("--thumbnail-size", min=1)] = 512,
    pos_path: Annotated[Path | None, typer.Option("--pos", exists=True, dir_okay=False, readable=True)] = None,
    pyramid: Annotated[
        bool,
        typer.Option("--pyramid/--no-pyramid", help="Write XY SubIFD pyramids for OME-TIFF outputs."),
    ] = True,
    delete_source: Annotated[
        bool,
        typer.Option("--delete/--no-delete", help="Delete each source CZI after successful compression."),
    ] = False,
) -> None:
    expanded_paths = _expand_input_paths(paths)
    if out_dir is not None:
        _validate_unique_output_stems(expanded_paths)

    compressed_paths = []
    for path in expanded_paths:
        try:
            compressed = compress_czi_to_ome_tiff(
                path,
                level=level,
                out_dir=out_dir,
                output_format=output_format,
                tile_size=tile_size,
                zarr_chunks=(1, 1, zarr_chunk_z, zarr_chunk_y, zarr_chunk_x),
                min_zarr_chunk_pixels=min_zarr_chunk_pixels,
                zarr_compressor=zarr_compressor,
                maxworkers=tiff_maxworkers,
                tile_workers=czi_tile_workers,
                overwrite=overwrite,
                thumbnails=thumbnails,
                thumbnail_size=thumbnail_size,
                pos_path=pos_path,
                pyramid=pyramid,
            )
        except Exception as exc:
            raise RuntimeError(f"Compression failed for {path}") from exc
        if compressed:
            compressed_paths.append(path)

    if delete_source:
        for path in compressed_paths:
            path.unlink()


@app.command()
def verify(
    paths: Annotated[list[Path], typer.Argument(help="One or more CZI files or glob patterns.")],
    out_dir: Annotated[Path | None, typer.Option("--out-dir")] = None,
    decode_samples: Annotated[bool, typer.Option("--decode-samples/--no-decode-samples")] = False,
    max_sample_mae: Annotated[float | None, typer.Option("--max-sample-mae", min=0.0)] = None,
    max_sample_max_abs: Annotated[float | None, typer.Option("--max-sample-max-abs", min=0.0)] = None,
) -> None:
    expanded_paths = _expand_input_paths(paths)
    if out_dir is not None:
        _validate_unique_output_stems(expanded_paths)

    for path in expanded_paths:
        verify_czi_ome_tiff_outputs(
            path,
            out_dir=out_dir,
            decode_samples=decode_samples,
            max_sample_mae=max_sample_mae,
            max_sample_max_abs=max_sample_max_abs,
        )


@app.command()
def compare(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    out_dir: Annotated[Path | None, typer.Option("--out-dir")] = None,
    count: Annotated[int, typer.Option("--count", min=1)] = 6,
    crop_size: Annotated[int, typer.Option("--crop-size", min=1)] = 256,
    min_level: Annotated[float, typer.Option("--min-level", min=0.0, max=100.0)] = 0.65,
    max_level: Annotated[float, typer.Option("--max-level", min=0.0, max=100.0)] = 0.90,
    level_step: Annotated[float, typer.Option("--level-step", min=0.0)] = 0.05,
    seed: Annotated[int, typer.Option("--seed")] = 20260604,
    tile_size: Annotated[int, typer.Option("--tile-size")] = 256,
    keep_encoded_tiffs: Annotated[bool, typer.Option("--keep-encoded-tiffs/--discard-encoded-tiffs")] = True,
    t: Annotated[int | None, typer.Option("--t")] = None,
    channel: Annotated[int | None, typer.Option("--channel")] = None,
    z: Annotated[int | None, typer.Option("--z")] = None,
    max_attempts: Annotated[int, typer.Option("--max-attempts", min=1)] = 500,
) -> None:
    compare_czi_compression(
        path,
        out_dir=out_dir,
        count=count,
        crop_size=crop_size,
        min_level=min_level,
        max_level=max_level,
        level_step=level_step,
        seed=seed,
        tile_size=tile_size,
        keep_encoded_tiffs=keep_encoded_tiffs,
        t=t,
        channel=channel,
        z=z,
        max_attempts=max_attempts,
    )


def _expand_input_paths(inputs: list[Path]) -> list[Path]:
    expanded_paths: list[Path] = []
    for input_path in inputs:
        pattern = os.fspath(input_path)
        if input_path.exists():
            expanded_paths.append(input_path)
        elif glob.has_magic(pattern):
            matches = [Path(match) for match in sorted(glob.glob(pattern))]
            if not matches:
                raise typer.BadParameter(f"No files matched input pattern: {input_path}")
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(input_path)

    if not expanded_paths:
        raise typer.BadParameter("At least one input file is required.")

    for path in expanded_paths:
        if not path.exists():
            raise typer.BadParameter(f"Input file does not exist: {path}")
        if path.is_dir():
            raise typer.BadParameter(f"Input path must be a file, got directory: {path}")
        if not os.access(path, os.R_OK):
            raise typer.BadParameter(f"Input file is not readable: {path}")

    return expanded_paths


def _validate_unique_output_stems(paths: list[Path]) -> None:
    stems: dict[str, Path] = {}
    for path in paths:
        if path.stem in stems:
            raise typer.BadParameter(
                f"Multiple inputs named {path.stem!r} would collide under --out-dir: {stems[path.stem]} and {path}"
            )
        stems[path.stem] = path


def main() -> None:
    app()
