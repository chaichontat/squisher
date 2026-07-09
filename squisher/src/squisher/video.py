from __future__ import annotations

import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.progress import track
import tifffile


Color = Literal["green", "magenta", "gray"]
ChannelLayout = Literal["interleaved", "contiguous"]

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
ZARR_RENDER_GPU_BATCH = 4


@dataclass(frozen=True)
class ZarrLevel:
    array: Any
    dims: tuple[str, ...]
    scale_zyx_um: tuple[float | None, float | None, float | None]


def render_tiff_video(
    path: Path,
    *,
    out: Path | None = None,
    size: int | None = None,
    fps: int = 20,
    low_percentile: float = 20.0,
    high_percentile: float = 99.99,
    nonzero_percentiles: bool = False,
    percentile_sample_frames: int = 64,
    channels: int = 1,
    channel_layout: ChannelLayout = "interleaved",
    channel: int = 0,
    overlay_channel: int | None = None,
    page_offset: int | None = None,
    overlay_page_offset: int | None = None,
    frame_step: int | None = None,
    color: Color = "green",
    overlay_color: Color = "magenta",
    reverse_z: bool = False,
    rotate_cw: bool = False,
    scale_bar_um: float | None = None,
    ffmpeg: str = "ffmpeg",
    encoder: str = "hevc_nvenc",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Stream TIFF pages into an RGB z movie without materializing the stack."""
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    if percentile_sample_frames < 1:
        raise ValueError(f"percentile_sample_frames must be >= 1, got {percentile_sample_frames}")
    if size is not None and size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    output = out or path.with_name(f"{path.stem}.z_h265_p{low_percentile:g}_p{high_percentile:g}.mp4")
    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError(f"No TIFF pages found in {path}")
        page_count = len(tif.pages)
        first_page = tif.pages[0]
        if len(first_page.shape) != 2:
            raise ValueError(f"Expected 2D TIFF pages, got first page shape {first_page.shape}")
        output_size = size or int(max(first_page.shape))
        y_um = _physical_size_um(tif.ome_metadata, "Y")
        x_um = _physical_size_um(tif.ome_metadata, "X")

        if page_offset is None:
            if frame_step is not None:
                raise ValueError("--frame-step requires --page-offset when not using channel layout")
            primary_indices = _channel_stream_indices(page_count, channels, channel, channel_layout)
        else:
            primary_indices = _page_stream_indices(page_count, page_offset, frame_step or 1)

        overlay_indices = None
        if overlay_channel is not None:
            if overlay_page_offset is None:
                if frame_step is not None:
                    raise ValueError("--frame-step requires --page-offset when not using channel layout")
                overlay_indices = _channel_stream_indices(
                    page_count, channels, overlay_channel, channel_layout
                )
            else:
                overlay_indices = _page_stream_indices(page_count, overlay_page_offset, frame_step or 1)
        elif overlay_page_offset is not None:
            overlay_indices = _page_stream_indices(page_count, overlay_page_offset, frame_step or 1)

        if overlay_indices is not None:
            frame_count = min(len(primary_indices), len(overlay_indices))
            primary_indices = primary_indices[:frame_count]
            overlay_indices = overlay_indices[:frame_count]

        print(
            f"Input: {path.resolve()} pages={page_count} shape={first_page.shape} dtype={first_page.dtype} "
            f"pixel_size_y={y_um} pixel_size_x={x_um}",
            flush=True,
        )
        print(
            f"Primary stream first={int(primary_indices[0])} color={color} frames={len(primary_indices)}",
            flush=True,
        )
        if overlay_indices is not None:
            print(
                f"Overlay stream first={int(overlay_indices[0])} color={overlay_color} frames={len(overlay_indices)}",
                flush=True,
            )

        primary_low, primary_high = _percentiles_for_indices(
            tif,
            primary_indices,
            low_percentile,
            high_percentile,
            nonzero_percentiles,
            percentile_sample_frames,
            "primary",
        )
        print(f"Primary percentiles low={primary_low} high={primary_high}", flush=True)
        overlay_low = overlay_high = None
        if overlay_indices is not None:
            overlay_low, overlay_high = _percentiles_for_indices(
                tif,
                overlay_indices,
                low_percentile,
                high_percentile,
                nonzero_percentiles,
                percentile_sample_frames,
                "overlay",
            )
            print(f"Overlay percentiles low={overlay_low} high={overlay_high}", flush=True)

        _, display_um = _resize_to_physical_canvas(
            tif.pages[int(primary_indices[0])].asarray(),
            output_size,
            y_um,
            x_um,
        )
        print(
            f"Display canvas={output_size}x{output_size} display_pixel_size_um={display_um}",
            flush=True,
        )

        if reverse_z:
            primary_indices = primary_indices[::-1]
            if overlay_indices is not None:
                overlay_indices = overlay_indices[::-1]
        frame_iter = (
            zip(primary_indices, overlay_indices, strict=True)
            if overlay_indices is not None
            else ((page_idx, None) for page_idx in primary_indices)
        )

        write_gray = color == "gray" and overlay_indices is None and scale_bar_um is None
        process = _open_ffmpeg(
            output,
            output_size,
            fps,
            ffmpeg,
            encoder,
            crf,
            preset,
            input_pix_fmt="gray" if write_gray else "rgb24",
        )
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe was not opened")
        try:
            render_start = time.monotonic()
            for frame_number, (page_idx, overlay_page_idx) in enumerate(
                track(
                    frame_iter,
                    total=len(primary_indices),
                    description="Rendering TIFF z",
                ),
                start=1,
            ):
                if write_gray:
                    frame, display_um = _scaled_gray_frame(
                        tif.pages[int(page_idx)].asarray(),
                        output_size,
                        y_um,
                        x_um,
                        primary_low,
                        primary_high,
                    )
                else:
                    frame, display_um = _scaled_color_frame(
                        tif.pages[int(page_idx)].asarray(),
                        output_size,
                        y_um,
                        x_um,
                        primary_low,
                        primary_high,
                        color,
                    )
                if overlay_page_idx is not None:
                    if overlay_low is None or overlay_high is None:
                        raise RuntimeError("Overlay stream was not initialized")
                    overlay_frame, _ = _scaled_color_frame(
                        tif.pages[int(overlay_page_idx)].asarray(),
                        output_size,
                        y_um,
                        x_um,
                        overlay_low,
                        overlay_high,
                        overlay_color,
                    )
                    frame = np.maximum(frame, overlay_frame)
                if rotate_cw:
                    frame = np.ascontiguousarray(np.rot90(frame, k=3))
                if scale_bar_um is not None:
                    frame = _draw_scale_bar(frame, scale_bar_um, display_um)
                process.stdin.write(frame.tobytes())
                _log_frame_progress("TIFF", frame_number, len(primary_indices), int(page_idx), render_start)
            process.stdin.close()
            return_code = process.wait()
        except BrokenPipeError as exc:
            return_code = process.wait()
            raise RuntimeError(f"ffmpeg exited early with status {return_code}") from exc
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"Wrote {output}", flush=True)
    return output


def render_zarr_video(
    path: Path,
    *,
    out: Path | None = None,
    size: int = 960,
    fps: int = 20,
    low_percentile: float = 20.0,
    high_percentile: float = 99.99,
    nonzero_percentiles: bool = False,
    percentile_sample_frames: int = 64,
    zarr_level: int = 0,
    channel: int = 0,
    frame_step: int | None = None,
    color: Color = "gray",
    reverse_z: bool = False,
    rotate_cw: bool = False,
    scale_bar_um: float | None = None,
    ffmpeg: str = "ffmpeg",
    encoder: str = "hevc_nvenc",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Stream a ZYX or TCZYX OME-Zarr pyramid level into a z movie."""
    if percentile_sample_frames < 1:
        raise ValueError(f"percentile_sample_frames must be >= 1, got {percentile_sample_frames}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if zarr_level < 0:
        raise ValueError(f"zarr_level must be >= 0, got {zarr_level}")
    if channel < 0:
        raise ValueError(f"channel must be >= 0, got {channel}")

    output = out or path.with_name(
        f"{path.name}.level{zarr_level}.z_h265_p{low_percentile:g}_p{high_percentile:g}.mp4"
    )
    level = _open_ome_zarr_level(path, zarr_level)
    array = level.array
    dims = level.dims
    z_count, y_size, x_size = _zarr_plane_shape(array, dims, channel)
    indices = _page_stream_indices(z_count, 0, frame_step or 1)
    print(
        f"Input: {path.resolve()} level={zarr_level} shape={array.shape} chunks={array.chunks} "
        f"dims={dims} channel={channel} dtype={array.dtype}",
        flush=True,
    )
    print(f"Primary stream first={int(indices[0])} color={color} frames={len(indices)}", flush=True)
    low, high = _percentiles_for_zarr_indices(
        array,
        indices,
        low_percentile,
        high_percentile,
        nonzero_percentiles,
        percentile_sample_frames,
        "primary",
        channel,
        dims=dims,
    )
    print(f"Primary percentiles low={low} high={high}", flush=True)

    _resized_shape_yx, display_um = _physical_canvas_geometry(
        (y_size, x_size),
        size,
        level.scale_zyx_um[1],
        level.scale_zyx_um[2],
    )
    print(f"Display canvas={size}x{size} source_plane={y_size}x{x_size}", flush=True)
    if reverse_z:
        indices = indices[::-1]

    write_gray = color == "gray" and scale_bar_um is None
    process = _open_ffmpeg(
        output,
        size,
        fps,
        ffmpeg,
        encoder,
        crf,
        preset,
        input_pix_fmt="gray" if write_gray else "rgb24",
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe was not opened")
    try:
        render_start = time.monotonic()
        frame_number = 0
        index_batches = list(
            _zarr_chunk_preserving_index_batches(
                indices,
                z_chunk_size=_zarr_z_chunk_size(array, dims),
                min_batch_size=ZARR_RENDER_GPU_BATCH,
            )
        )
        for batch_indices in track(
            index_batches,
            total=len(index_batches),
            description="Rendering Zarr z",
        ):
            frames = _render_zarr_batch_gpu(
                array,
                batch_indices,
                size,
                low,
                high,
                color,
                channel,
                write_gray=write_gray,
                rotate_cw=rotate_cw,
                dims=dims,
                y_um=level.scale_zyx_um[1],
                x_um=level.scale_zyx_um[2],
            )
            for z_index, frame in zip(batch_indices, frames, strict=True):
                frame_number += 1
                if scale_bar_um is not None:
                    frame = _draw_scale_bar(frame, scale_bar_um, display_um)
                process.stdin.write(memoryview(np.ascontiguousarray(frame)))
                _log_frame_progress("Zarr", frame_number, len(indices), int(z_index), render_start)
        process.stdin.close()
        return_code = process.wait()
    except BrokenPipeError as exc:
        return_code = process.wait()
        raise RuntimeError(f"ffmpeg exited early with status {return_code}") from exc
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"Wrote {output}", flush=True)
    return output


def _page_stream_indices(page_count: int, page_offset: int, frame_step: int) -> np.ndarray:
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    if page_offset < 0:
        raise ValueError(f"page_offset must be >= 0, got {page_offset}")
    if page_offset >= page_count:
        raise ValueError(f"page_offset {page_offset} is outside page count {page_count}")
    return np.arange(page_offset, page_count, frame_step, dtype=int)


def _log_frame_progress(
    source: str,
    frame_number: int,
    frame_count: int,
    source_index: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    fps = frame_number / elapsed
    remaining = (frame_count - frame_number) / fps if fps > 0 else float("inf")
    print(
        f"Rendered {source} frame {frame_number}/{frame_count} "
        f"source_index={source_index} elapsed={elapsed:.1f}s eta={remaining:.1f}s fps={fps:.2f}",
        flush=True,
    )


def _channel_stream_indices(
    page_count: int,
    channels: int,
    channel: int,
    channel_layout: ChannelLayout,
) -> np.ndarray:
    if channel < 0 or channel >= channels:
        raise ValueError(f"channel must be in [0, {channels}), got {channel}")
    if page_count % channels:
        raise ValueError(f"page count {page_count} is not divisible by channel count {channels}")
    z_count = page_count // channels
    if channel_layout == "interleaved":
        return np.arange(channel, page_count, channels, dtype=int)
    if channel_layout == "contiguous":
        start = channel * z_count
        return np.arange(start, start + z_count, dtype=int)
    raise ValueError(f"Unsupported channel layout {channel_layout!r}")


def _open_ome_zarr_level(path: Path, zarr_level: int) -> ZarrLevel:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    multiscales = list(root.attrs.get("multiscales", ()))
    if multiscales:
        multiscale = multiscales[0]
        datasets = list(multiscale.get("datasets", ()))
        if zarr_level >= len(datasets):
            raise ValueError(
                f"zarr_level {zarr_level} is outside multiscales dataset count {len(datasets)} for {path}"
            )
        dataset = datasets[zarr_level]
        dataset_path = dataset.get("path")
        if not dataset_path:
            raise ValueError(f"OME-Zarr multiscales dataset {zarr_level} is missing a path")
        array = root[str(dataset_path)]
        dims = _zarr_axes_from_multiscales(multiscale)
        if not dims:
            dims = _zarr_dimension_names(array)
        return ZarrLevel(
            array=array,
            dims=dims,
            scale_zyx_um=_zarr_scale_zyx_um(dims, dataset),
        )

    array = zarr.open_array(str(path / str(zarr_level)), mode="r")
    return ZarrLevel(
        array=array,
        dims=_zarr_dimension_names(array),
        scale_zyx_um=(None, None, None),
    )


def _zarr_axes_from_multiscales(multiscale: dict[str, Any]) -> tuple[str, ...]:
    axes = multiscale.get("axes")
    if not axes:
        return ()
    names = []
    for axis in axes:
        if isinstance(axis, str):
            names.append(axis.lower())
        elif isinstance(axis, dict) and axis.get("name"):
            names.append(str(axis["name"]).lower())
        else:
            return ()
    return tuple(names)


def _zarr_scale_zyx_um(
    dims: tuple[str, ...], dataset: dict[str, Any]
) -> tuple[float | None, float | None, float | None]:
    for transform in dataset.get("coordinateTransformations", ()):
        if transform.get("type") != "scale":
            continue
        scale = transform.get("scale")
        if not isinstance(scale, list | tuple) or len(scale) != len(dims):
            continue
        values = {dim: float(scale[index]) for index, dim in enumerate(dims)}
        return values.get("z"), values.get("y"), values.get("x")
    return None, None, None


def _zarr_dimension_names(array) -> tuple[str, ...]:
    names = tuple(getattr(array.metadata, "dimension_names", None) or ())
    if names:
        return names
    names = tuple(array.attrs.get("_ARRAY_DIMENSIONS", ()))
    if names:
        return names
    raise ValueError(
        "Zarr array is missing dimension names; expected explicit ('z', 'y', 'x') "
        "or ('t', 'c', 'z', 'y', 'x') axes metadata."
    )


def _zarr_plane_shape(array, dims: tuple[str, ...], channel: int) -> tuple[int, int, int]:
    if dims == ("z", "y", "x"):
        if channel != 0:
            raise ValueError("channel must be 0 for a 3D ZYX Zarr array")
        return int(array.shape[0]), int(array.shape[1]), int(array.shape[2])
    if dims == ("t", "c", "z", "y", "x"):
        channel_count = int(array.shape[1])
        if channel >= channel_count:
            raise ValueError(f"channel must be in [0, {channel_count}), got {channel}")
        return int(array.shape[2]), int(array.shape[3]), int(array.shape[4])
    raise ValueError(
        f"Expected Zarr dimension names ('z', 'y', 'x') or ('t', 'c', 'z', 'y', 'x'), got {dims}"
    )


def _zarr_z_chunk_size(array, dims: tuple[str, ...]) -> int:
    chunks = tuple(int(value) for value in getattr(array, "chunks", ()) or ())
    if dims == ("z", "y", "x") and len(chunks) == 3:
        return chunks[0]
    if dims == ("t", "c", "z", "y", "x") and len(chunks) == 5:
        return chunks[2]
    return 1


def _zarr_chunk_preserving_index_batches(
    indices: np.ndarray,
    *,
    z_chunk_size: int,
    min_batch_size: int,
) -> list[np.ndarray]:
    if z_chunk_size < 1:
        raise ValueError(f"z_chunk_size must be >= 1, got {z_chunk_size}")
    if min_batch_size < 1:
        raise ValueError(f"min_batch_size must be >= 1, got {min_batch_size}")
    if len(indices) == 0:
        return []

    batches = []
    current: list[int] = []
    current_chunk = int(indices[0]) // z_chunk_size
    for index in indices:
        chunk = int(index) // z_chunk_size
        if chunk != current_chunk:
            if len(current) >= min_batch_size:
                batches.append(np.asarray(current, dtype=int))
                current = []
            current_chunk = chunk
        current.append(int(index))
    if current:
        batches.append(np.asarray(current, dtype=int))
    return batches


def _percentile_from_histogram(histogram: np.ndarray, percentile: float) -> float:
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("Cannot compute percentiles from an empty histogram.")
    rank = int(np.ceil(percentile / 100.0 * total))
    return float(np.searchsorted(np.cumsum(histogram), rank, side="left"))


def _percentiles_for_indices(
    tif: tifffile.TiffFile,
    indices: np.ndarray,
    low_percentile: float,
    high_percentile: float,
    nonzero: bool,
    sample_frames: int,
    label: str,
) -> tuple[float, float]:
    sample_count = min(sample_frames, len(indices))
    sample_positions = np.unique(np.linspace(0, len(indices) - 1, sample_count, dtype=int))
    sample_indices = indices[sample_positions]
    histogram = np.zeros(65536, dtype=np.uint64)
    for page_idx in track(sample_indices, description=f"Sampling {label} TIFF pages"):
        values = tif.pages[int(page_idx)].asarray().reshape(-1)
        if nonzero:
            values = values[values > 0]
        if values.size:
            histogram += np.bincount(values, minlength=65536).astype(np.uint64)
    low = _percentile_from_histogram(histogram, low_percentile)
    high = _percentile_from_histogram(histogram, high_percentile)
    if high <= low:
        raise ValueError(f"Invalid {label} percentile range: low={low} high={high}")
    return low, high


def _percentiles_for_zarr_indices(
    array,
    indices: np.ndarray,
    low_percentile: float,
    high_percentile: float,
    nonzero: bool,
    sample_frames: int,
    label: str,
    channel: int,
    dims: tuple[str, ...] | None = None,
) -> tuple[float, float]:
    sample_count = min(sample_frames, len(indices))
    sample_positions = np.unique(np.linspace(0, len(indices) - 1, sample_count, dtype=int))
    sample_indices = indices[sample_positions]
    histogram = np.zeros(65536, dtype=np.uint64)
    resolved_dims = dims or _zarr_dimension_names(array)
    sample_batches = _zarr_chunk_preserving_index_batches(
        sample_indices,
        z_chunk_size=_zarr_z_chunk_size(array, resolved_dims),
        min_batch_size=1,
    )
    for sample_batch in track(sample_batches, description=f"Sampling {label} Zarr chunks"):
        planes = _read_zarr_batch(array, sample_batch, channel, dims=resolved_dims)
        for plane in planes:
            values = np.asarray(plane).reshape(-1)
            if nonzero:
                values = values[values > 0]
            if values.size:
                histogram += np.bincount(values, minlength=65536).astype(np.uint64)
    low = _percentile_from_histogram(histogram, low_percentile)
    high = _percentile_from_histogram(histogram, high_percentile)
    if high <= low:
        raise ValueError(f"Invalid {label} percentile range: low={low} high={high}")
    return low, high


def _read_zarr_plane(
    array,
    z_index: int,
    channel: int,
    *,
    dims: tuple[str, ...] | None = None,
) -> np.ndarray:
    dims = dims or _zarr_dimension_names(array)
    if dims == ("z", "y", "x"):
        if channel != 0:
            raise ValueError("channel must be 0 for a 3D ZYX Zarr array")
        return np.asarray(array[z_index, :, :])
    if dims == ("t", "c", "z", "y", "x"):
        return np.asarray(array[0, channel, z_index, :, :])
    raise ValueError(
        f"Expected Zarr dimension names ('z', 'y', 'x') or ('t', 'c', 'z', 'y', 'x'), got {dims}"
    )


def _read_zarr_batch(
    array,
    indices: np.ndarray,
    channel: int,
    *,
    dims: tuple[str, ...] | None = None,
) -> np.ndarray:
    dims = dims or _zarr_dimension_names(array)
    if len(indices) == 0:
        raise ValueError("indices must contain at least one z index")
    start = int(indices[0])
    stop = int(indices[-1]) + 1
    if len(indices) == stop - start and np.array_equal(indices, np.arange(start, stop)):
        return np.asarray(_read_zarr_span(array, dims, start, stop, channel))

    chunk_size = _zarr_z_chunk_size(array, dims)
    chunks: list[np.ndarray] = []
    for chunk_indices in _zarr_chunk_preserving_index_batches(
        indices,
        z_chunk_size=chunk_size,
        min_batch_size=1,
    ):
        chunk_start = int(np.min(chunk_indices))
        chunk_stop = int(np.max(chunk_indices)) + 1
        span = np.asarray(_read_zarr_span(array, dims, chunk_start, chunk_stop, channel))
        chunks.append(span[np.asarray(chunk_indices, dtype=int) - chunk_start])
    return np.concatenate(chunks, axis=0)


def _read_zarr_span(array, dims: tuple[str, ...], start: int, stop: int, channel: int) -> np.ndarray:
    if dims == ("z", "y", "x"):
        if channel != 0:
            raise ValueError("channel must be 0 for a 3D ZYX Zarr array")
        return np.asarray(array[start:stop, :, :])
    if dims == ("t", "c", "z", "y", "x"):
        return np.asarray(array[0, channel, start:stop, :, :])
    raise ValueError(
        f"Expected Zarr dimension names ('z', 'y', 'x') or ('t', 'c', 'z', 'y', 'x'), got {dims}"
    )


def _render_zarr_batch_gpu(
    array,
    indices: np.ndarray,
    size: int,
    low: float,
    high: float,
    color: Color,
    channel: int,
    *,
    write_gray: bool,
    rotate_cw: bool,
    dims: tuple[str, ...],
    y_um: float | None,
    x_um: float | None,
) -> np.ndarray:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage

    batch = cp.asarray(_read_zarr_batch(array, indices, channel, dims=dims))
    resized_shape_yx, _display_um = _physical_canvas_geometry(batch.shape[1:], size, y_um, x_um)
    if batch.shape[1:] != resized_shape_yx:
        zoom = (1.0, resized_shape_yx[0] / batch.shape[1], resized_shape_yx[1] / batch.shape[2])
        batch = cpx_ndimage.zoom(batch, zoom, order=1, mode="nearest", prefilter=False)
    batch = _center_crop_or_pad_gpu(batch, (batch.shape[0], size, size), cp)
    scaled = cp.clip((batch.astype(cp.float32) - low) / (high - low), 0.0, 1.0)
    gray = cp.rint(scaled * 255).astype(cp.uint8)
    if rotate_cw:
        gray = cp.rot90(gray, k=3, axes=(1, 2))
    if write_gray:
        return cp.asnumpy(cp.ascontiguousarray(gray))

    rgb = cp.zeros((*gray.shape, 3), dtype=cp.uint8)
    if color == "green":
        rgb[..., 1] = gray
    elif color == "magenta":
        rgb[..., 0] = gray
        rgb[..., 2] = gray
    elif color == "gray":
        rgb[...] = gray[..., None]
    else:
        raise ValueError(f"Unsupported color {color!r}")
    return cp.asnumpy(cp.ascontiguousarray(rgb))


def _center_crop_or_pad_gpu(array, shape: tuple[int, int, int], cp):
    if tuple(int(value) for value in array.shape) == shape:
        return array
    source_y0 = max((array.shape[1] - shape[1]) // 2, 0)
    source_x0 = max((array.shape[2] - shape[2]) // 2, 0)
    source_y1 = source_y0 + min(shape[1], array.shape[1])
    source_x1 = source_x0 + min(shape[2], array.shape[2])
    cropped = array[: shape[0], source_y0:source_y1, source_x0:source_x1]
    if tuple(int(value) for value in cropped.shape) == shape:
        return cropped
    output = cp.zeros(shape, dtype=array.dtype)
    target_y0 = (shape[1] - cropped.shape[1]) // 2
    target_x0 = (shape[2] - cropped.shape[2]) // 2
    output[:, target_y0 : target_y0 + cropped.shape[1], target_x0 : target_x0 + cropped.shape[2]] = cropped
    return output


def _physical_size_um(ome_xml: str | None, axis: str) -> float | None:
    if not ome_xml:
        return None
    root = ET.fromstring(ome_xml)
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    pixels = root.find(".//ome:Pixels", ns)
    if pixels is None:
        return None
    value = pixels.attrib.get(f"PhysicalSize{axis}")
    return None if value is None else float(value)


def _resize_to_physical_canvas(
    plane: np.ndarray,
    size: int,
    y_um: float | None,
    x_um: float | None,
) -> tuple[np.ndarray, float | None]:
    (resized_h, resized_w), output_um = _physical_canvas_geometry(plane.shape, size, y_um, x_um)
    if y_um is None or x_um is None:
        return _resize(plane, resized_w, resized_h), None
    resized = _resize(plane, resized_w, resized_h)
    canvas = np.zeros((size, size), dtype=resized.dtype)
    y0 = (size - resized_h) // 2
    x0 = (size - resized_w) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas, output_um


def _physical_canvas_geometry(
    shape_yx: tuple[int, int],
    size: int,
    y_um: float | None,
    x_um: float | None,
) -> tuple[tuple[int, int], float | None]:
    if y_um is None or x_um is None:
        return (size, size), None
    physical_h = shape_yx[0] * y_um
    physical_w = shape_yx[1] * x_um
    output_um = max(physical_h, physical_w) / size
    resized_h = min(size, max(1, int(round(physical_h / output_um))))
    resized_w = min(size, max(1, int(round(physical_w / output_um))))
    return (resized_h, resized_w), output_um


def _resize(plane: np.ndarray, width: int, height: int) -> np.ndarray:
    if plane.shape == (height, width):
        return plane
    pil_mode = "I;16" if plane.dtype == np.uint16 else None
    image = Image.fromarray(plane, mode=pil_mode)
    resample = (
        Image.Resampling.BOX
        if plane.shape[0] > height or plane.shape[1] > width
        else Image.Resampling.BILINEAR
    )
    return np.asarray(image.resize((width, height), resample=resample), dtype=plane.dtype)


def _scaled_color_frame(
    plane: np.ndarray,
    size: int,
    y_um: float | None,
    x_um: float | None,
    low: float,
    high: float,
    color: Color,
) -> tuple[np.ndarray, float | None]:
    resized, display_um = _resize_to_physical_canvas(plane, size, y_um, x_um)
    scaled = np.clip((resized.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return _colorize(np.rint(scaled * 255).astype(np.uint8), color), display_um


def _scaled_gray_frame(
    plane: np.ndarray,
    size: int,
    y_um: float | None,
    x_um: float | None,
    low: float,
    high: float,
) -> tuple[np.ndarray, float | None]:
    resized, display_um = _resize_to_physical_canvas(plane, size, y_um, x_um)
    scaled = np.clip((resized.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255).astype(np.uint8), display_um


def _colorize(gray: np.ndarray, color: Color) -> np.ndarray:
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    if color == "green":
        rgb[..., 1] = gray
    elif color == "magenta":
        rgb[..., 0] = gray
        rgb[..., 2] = gray
    elif color == "gray":
        rgb[:] = gray[..., np.newaxis]
    else:
        raise ValueError(f"Unsupported color {color!r}")
    return rgb


def _scale_bar_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, 22)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_scale_bar(frame: np.ndarray, length_um: float, display_pixel_size_um: float | None) -> np.ndarray:
    if display_pixel_size_um is None:
        return frame
    bar_width = int(round(length_um / display_pixel_size_um))
    if bar_width <= 0 or bar_width > frame.shape[1] - 40:
        return frame
    margin = 28
    bar_height = 3
    text_gap = 8
    label = f"{length_um:g} um"
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _scale_bar_font()
    box = draw.textbbox((0, 0), label, font=font)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    x1 = frame.shape[1] - margin
    x0 = x1 - bar_width
    y1 = frame.shape[0] - margin - text_h - text_gap
    y0 = y1 - bar_height
    text_x = x0 + (bar_width - text_w) // 2
    text_y = y1 + text_gap
    alpha = 204
    draw.rectangle((x0 + 2, y0 + 2, x1 + 2, y1 + 2), fill=(0, 0, 0, alpha))
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, alpha))
    draw.text((text_x + 1, text_y + 1), label, fill=(0, 0, 0, alpha), font=font)
    draw.text((text_x, text_y), label, fill=(255, 255, 255, alpha), font=font)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _open_ffmpeg(
    out: Path,
    size: int,
    fps: int,
    ffmpeg: str,
    encoder: str,
    crf: int,
    preset: str,
    *,
    input_pix_fmt: str = "rgb24",
) -> subprocess.Popen[bytes]:
    out.parent.mkdir(parents=True, exist_ok=True)
    quality_args = ["-crf", str(crf)]
    if encoder == "hevc_nvenc":
        quality_args = ["-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        input_pix_fmt,
        "-s",
        f"{size}x{size}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
        "-preset",
        preset,
        *quality_args,
        "-tag:v",
        "hvc1",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
