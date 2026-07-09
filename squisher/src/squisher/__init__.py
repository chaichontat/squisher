import glob
import os
from pathlib import Path
from typing import Annotated, cast

import typer

from squisher.compare import compare_czi_compression
from squisher.compression import (
    DEFAULT_MIN_ZARR_CHUNK_PIXELS,
    DEFAULT_ZARR_CHUNKS_TCZYX,
    compress_czi_to_ome_tiff,
    verify_czi_ome_tiff_outputs,
)
from squisher.logging import setup_cli_logging
from squisher.video import ChannelLayout, Color, render_tiff_video, render_zarr_video


app = typer.Typer(no_args_is_help=True)
DEFAULT_PYRAMID_GPU_BATCH_SIZE = 32


@app.callback()
def callback() -> None:
    setup_cli_logging()


@app.command()
def compress(
    paths: Annotated[list[Path], typer.Argument(help="One or more CZI files or glob patterns.")],
    out_dir: Annotated[Path | None, typer.Option("--out-dir")] = None,
    output_format: Annotated[str, typer.Option("--output-format")] = "ome-tiff",
    level: Annotated[float, typer.Option("--level", min=0.0, max=100.0)] = 0.7,
    tile_size: Annotated[int | None, typer.Option("--tile-size")] = None,
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
    pos_path: Annotated[
        Path | None, typer.Option("--pos", exists=True, dir_okay=False, readable=True)
    ] = None,
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
def pyramid(
    inputs: Annotated[list[Path], typer.Argument(help="Source OME-TIFF files or folders.")],
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Destination OME-TIFF file.")] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Destination folder. Defaults to a new sibling *.pyramid folder."),
    ] = None,
    factor: Annotated[
        int,
        typer.Option("--factor", min=2, help="XY downsampling factor between the two fixed SubIFD levels."),
    ] = 2,
    overwrite: Annotated[
        bool, typer.Option("--overwrite/--no-overwrite", help="Replace existing outputs.")
    ] = False,
    gpu_batch_size: Annotated[
        int,
        typer.Option("--gpu-batch-size", min=1, help="Number of planes to downsample per GPU batch."),
    ] = DEFAULT_PYRAMID_GPU_BATCH_SIZE,
    tiff_maxworkers: Annotated[
        int | None,
        typer.Option(
            "--tiff-maxworkers", min=1, help="Maximum worker threads tifffile may use for compression."
        ),
    ] = None,
    file_workers: Annotated[
        int,
        typer.Option("--file-workers", min=1, help="Number of input TIFF files to process concurrently."),
    ] = 8,
) -> None:
    """Write two XY TIFF SubIFD pyramid levels into OME-TIFF files."""
    from squisher.tiff_pyramid import run_pyramid_jobs

    run_pyramid_jobs(
        inputs,
        output=output,
        output_dir=output_dir,
        factor=factor,
        overwrite=overwrite,
        gpu_batch_size=gpu_batch_size,
        tiff_maxworkers=tiff_maxworkers,
        file_workers=file_workers,
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


@app.command("video")
def video(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    size: Annotated[int | None, typer.Option("--size", min=1)] = None,
    fps: Annotated[int, typer.Option("--fps", min=1)] = 20,
    low: Annotated[float, typer.Option("--low", min=0.0, max=100.0)] = 20.0,
    high: Annotated[float, typer.Option("--high", min=0.0, max=100.0)] = 99.99,
    nonzero_percentiles: Annotated[bool, typer.Option("--nonzero-percentiles/--all-pixels")] = False,
    percentile_sample_frames: Annotated[int, typer.Option("--percentile-sample-frames", min=1)] = 64,
    channels: Annotated[int, typer.Option("--channels", min=1)] = 1,
    channel_layout: Annotated[
        str,
        typer.Option(
            "--channel-layout", help="Flattened page order: interleaved uses z*C+c; contiguous uses c*Z+z."
        ),
    ] = "interleaved",
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    overlay_channel: Annotated[int | None, typer.Option("--overlay-channel", min=0)] = None,
    page_offset: Annotated[int | None, typer.Option("--page-offset", min=0)] = None,
    overlay_page_offset: Annotated[int | None, typer.Option("--overlay-page-offset", min=0)] = None,
    frame_step: Annotated[int | None, typer.Option("--frame-step", min=1)] = None,
    zarr_level: Annotated[int, typer.Option("--zarr-level", min=0)] = 0,
    color: Annotated[str | None, typer.Option("--color")] = None,
    overlay_color: Annotated[str, typer.Option("--overlay-color")] = "magenta",
    reverse_z: Annotated[bool, typer.Option("--reverse-z/--forward-z")] = False,
    rotate_cw: Annotated[bool, typer.Option("--rotate-cw/--no-rotate-cw")] = False,
    scale_bar_um: Annotated[float | None, typer.Option("--scale-bar-um", min=0.0)] = None,
    ffmpeg: Annotated[str, typer.Option("--ffmpeg")] = "ffmpeg",
    encoder: Annotated[str, typer.Option("--encoder")] = "hevc_nvenc",
    crf: Annotated[int, typer.Option("--crf", min=0, max=51)] = 18,
    preset: Annotated[str, typer.Option("--preset")] = "medium",
) -> None:
    """Render a large-Z TIFF or OME-Zarr as a streamed z movie."""
    if channel_layout not in {"interleaved", "contiguous"}:
        raise typer.BadParameter("--channel-layout must be 'interleaved' or 'contiguous'.")
    if color is not None and color not in {"green", "magenta", "gray"}:
        raise typer.BadParameter("--color must be 'green', 'magenta', or 'gray'.")
    if overlay_color not in {"green", "magenta", "gray"}:
        raise typer.BadParameter("--overlay-color must be 'green', 'magenta', or 'gray'.")
    if path.is_dir():
        if channels != 1:
            raise typer.BadParameter("--channels is only supported for TIFF inputs.")
        if overlay_channel is not None or overlay_page_offset is not None:
            raise typer.BadParameter("overlay options are only supported for TIFF inputs.")
        if page_offset is not None:
            raise typer.BadParameter("--page-offset is only supported for TIFF inputs.")
        rendered = render_zarr_video(
            path,
            out=out,
            size=size or 960,
            fps=fps,
            low_percentile=low,
            high_percentile=high,
            nonzero_percentiles=nonzero_percentiles,
            percentile_sample_frames=percentile_sample_frames,
            zarr_level=zarr_level,
            channel=channel,
            frame_step=frame_step,
            color=cast(Color, color or "gray"),
            reverse_z=reverse_z,
            rotate_cw=rotate_cw,
            scale_bar_um=scale_bar_um,
            ffmpeg=ffmpeg,
            encoder=encoder,
            crf=crf,
            preset=preset,
        )
    else:
        rendered = render_tiff_video(
            path,
            out=out,
            size=size,
            fps=fps,
            low_percentile=low,
            high_percentile=high,
            nonzero_percentiles=nonzero_percentiles,
            percentile_sample_frames=percentile_sample_frames,
            channels=channels,
            channel_layout=cast(ChannelLayout, channel_layout),
            channel=channel,
            overlay_channel=overlay_channel,
            page_offset=page_offset,
            overlay_page_offset=overlay_page_offset,
            frame_step=frame_step,
            color=cast(Color, color or "green"),
            overlay_color=cast(Color, overlay_color),
            reverse_z=reverse_z,
            rotate_cw=rotate_cw,
            scale_bar_um=scale_bar_um,
            ffmpeg=ffmpeg,
            encoder=encoder,
            crf=crf,
            preset=preset,
        )
    typer.echo(str(rendered))


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
