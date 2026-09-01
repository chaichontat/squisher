#!/usr/bin/env python
"""Fit one BaSiC profile from concatenated ch0/ch1 selected slices.

This is the parameterized version of the 230Tnc/231Tnc joint-channel workflow:
build per-channel selected-slice caches with the standard edge/content rejection,
take an even subset from each channel, fit one BaSiC model, and export that same
flatfield/darkfield under each channel name so existing fusion/benchmark code
can consume it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from basicpy import BaSiC
from tifffile import imwrite


REPO = Path("/home/chaichontat/nvme/lightsheet")


def parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(channels) < 2:
        raise argparse.ArgumentTypeError("expected at least two comma-separated channels")
    if len(set(channels)) != len(channels):
        raise argparse.ArgumentTypeError(f"duplicate channel in {value!r}")
    return channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dirs", type=Path, nargs="+", help="Directories containing source OME-TIFF tiles.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--channels", type=parse_channels, default=(0, 1))
    parser.add_argument("--cache-z-total-per-channel", type=int, default=500)
    parser.add_argument("--fit-total-slices", type=int, default=500)
    parser.add_argument("--source-level", type=int, default=0)
    parser.add_argument("--blank-slice-sample-stride", type=int, default=16)
    parser.add_argument("--blank-slice-min-relative-signal", type=float, default=0.10)
    parser.add_argument("--blank-slice-min-nonzero-fraction", type=float, default=1e-4)
    parser.add_argument("--edge-slice-min-profile-jump", type=float, default=0.05)
    parser.add_argument("--edge-slice-min-band-delta", type=float, default=0.35)
    parser.add_argument("--smoothness-flatfield", type=float, default=1.8)
    parser.add_argument("--fitting-mode", choices=("approximate", "ladmap"), default="approximate")
    parser.add_argument("--working-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.cache_z_total_per_channel <= 0:
        raise ValueError("--cache-z-total-per-channel must be positive")
    if args.fit_total_slices <= 0:
        raise ValueError("--fit-total-slices must be positive")
    if args.source_level < 0:
        raise ValueError("--source-level must be non-negative")
    return args


def load_fit_module():
    path = REPO / "scripts" / "fit_basic_ome_tiff_tiles.py"
    spec = importlib.util.spec_from_file_location("fit_basic_ome_tiff_tiles", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def even_indices(length: int, count: int) -> np.ndarray:
    if length < count:
        raise ValueError(f"Cannot draw {count} slices from cache with only {length} slices")
    return np.linspace(0, length - 1, count, dtype=np.int64)


def cache_paths(output_dir: Path, label: str, channel: int) -> tuple[Path, Path]:
    cache_dir = output_dir / "sample-cache"
    return (
        cache_dir / f"{label}-ch{channel}-selected-slices.npy",
        cache_dir / f"{label}-ch{channel}-selected-slices.json",
    )


def selected_slices_for_channel(
    fit_module: Any,
    files: list[Path],
    *,
    channel: int,
    source_level: int,
    source_axes: str,
    shape_czyx: tuple[int, int, int, int],
    tile_layouts: dict[Path, Any],
    tile_shapes_czyx_by_file: dict[str, list[int]],
    target_samples: int,
    args: argparse.Namespace,
):
    cache_key = fit_module.qc_cache_key(
        files,
        channel=channel,
        target_samples=target_samples,
        source_level=source_level,
        source_axes=source_axes,
        expected_shape_czyx=shape_czyx,
        stride=args.blank_slice_sample_stride,
        min_relative_signal=args.blank_slice_min_relative_signal,
        min_nonzero_fraction=args.blank_slice_min_nonzero_fraction,
        exclude_blank_slices=True,
        exclude_edge_slices=True,
        edge_min_profile_jump=args.edge_slice_min_profile_jump,
        edge_min_band_delta=args.edge_slice_min_band_delta,
        tile_shapes_czyx_by_file=tile_shapes_czyx_by_file,
    )
    qc_path = args.output_dir / f"{args.label}-ch{channel}-qc-cache.json"
    selection = None if args.overwrite else fit_module.read_qc_cache(qc_path, cache_key)
    if selection is None:
        selection = fit_module.select_training_slices(
            files,
            channel=channel,
            z_count=shape_czyx[1],
            target_samples=target_samples,
            fallback_z_indices=fit_module.sample_z_indices(shape_czyx[1], fit_module.resolve_nz(explicit_nz=25, z_total=target_samples, n_files=len(files), z_count=shape_czyx[1]))[1],
            source_axes=source_axes,
            expected_shape_czyx=shape_czyx,
            stride=args.blank_slice_sample_stride,
            source_level=source_level,
            min_relative_signal=args.blank_slice_min_relative_signal,
            min_nonzero_fraction=args.blank_slice_min_nonzero_fraction,
            exclude_blank_slices=True,
            exclude_edge_slices=True,
            edge_min_profile_jump=args.edge_slice_min_profile_jump,
            edge_min_band_delta=args.edge_slice_min_band_delta,
            tile_layouts=tile_layouts,
        )
        fit_module.write_qc_cache(qc_path, cache_key, selection)
    return selection


def ensure_channel_cache(
    fit_module: Any,
    files: list[Path],
    *,
    channel: int,
    source_level: int,
    source_axes: str,
    shape_czyx: tuple[int, int, int, int],
    source_dtype: str,
    tile_layouts: dict[Path, Any],
    tile_shapes_czyx_by_file: dict[str, list[int]],
    args: argparse.Namespace,
) -> tuple[Path, int]:
    data_path, meta_path = cache_paths(args.output_dir, args.label, channel)
    if data_path.exists() and not args.overwrite:
        cached = np.load(data_path, mmap_mode="r")
        if cached.ndim != 3:
            raise ValueError(f"{data_path} has unexpected shape {cached.shape}")
        print(f"Using existing channel cache {data_path}: shape={cached.shape}", flush=True)
        return data_path, int(cached.shape[0])

    selection = selected_slices_for_channel(
        fit_module,
        files,
        channel=channel,
        source_level=source_level,
        source_axes=source_axes,
        shape_czyx=shape_czyx,
        tile_layouts=tile_layouts,
        tile_shapes_czyx_by_file=tile_shapes_czyx_by_file,
        target_samples=args.cache_z_total_per_channel,
        args=args,
    )
    sample_key = fit_module.sample_cache_key(
        files,
        channel=channel,
        selected_slices=selection.selected,
        source_level=source_level,
        source_axes=source_axes,
        expected_shape_czyx=shape_czyx,
        source_dtype=source_dtype,
        tile_shapes_czyx_by_file=tile_shapes_czyx_by_file,
    )
    data = fit_module.read_channel_samples(
        files,
        channel,
        selection.selected,
        source_level=source_level,
        tile_layouts=tile_layouts,
        sample_cache_data_path=data_path,
        sample_cache_meta_path=meta_path,
        sample_cache_key_payload=sample_key,
    )
    count = int(data.shape[0])
    del data
    return data_path, count


def copy_to_combined_stack(
    channel_caches: dict[int, Path],
    *,
    total_slices: int,
    output_path: Path,
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    first_channel = next(iter(channel_caches))
    first = np.load(channel_caches[first_channel], mmap_mode="r")
    per_channel = total_slices // len(channel_caches)
    remainder = total_slices - per_channel * len(channel_caches)
    counts = {
        channel: per_channel + (1 if index < remainder else 0)
        for index, channel in enumerate(channel_caches)
    }
    shape = (total_slices, int(first.shape[1]), int(first.shape[2]))
    if output_path.exists() and not overwrite:
        existing = np.load(output_path, mmap_mode="r+")
        if tuple(existing.shape) == shape and existing.dtype == np.float32:
            print(f"Using existing combined stack {output_path}: shape={existing.shape}", flush=True)
            return existing, {f"ch{channel}": int(count) for channel, count in counts.items()}
        raise ValueError(f"Existing combined stack has unexpected shape/dtype: {existing.shape} {existing.dtype}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)
    offset = 0
    chunk = 25
    for channel, cache_path in channel_caches.items():
        cache = np.load(cache_path, mmap_mode="r")
        if cache.shape[1:] != first.shape[1:]:
            raise ValueError(f"Cache shape mismatch: {cache_path} has {cache.shape}, expected spatial {first.shape[1:]}")
        count = counts[channel]
        indices = even_indices(int(cache.shape[0]), count)
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            stack[offset + start : offset + stop] = cache[indices[start:stop]]
            print(f"Copied ch{channel} slices {start}:{stop}", flush=True)
        offset += count
    stack.flush()
    print(f"Wrote combined stack {output_path}: shape={stack.shape}", flush=True)
    return stack, {f"ch{channel}": int(count) for channel, count in counts.items()}


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(file for input_dir in input_dirs for file in input_dir.glob("*.ome.tif"))
    if not files:
        raise FileNotFoundError(f"No *.ome.tif files found in {input_dirs}")
    fit_module = load_fit_module()
    tile_layouts = fit_module.inspect_tile_layouts(files, source_level=args.source_level)
    source_axes, height, width, source_dtype = fit_module.common_layout_contract(tile_layouts)
    channel_count = min(layout.channel_count for layout in tile_layouts.values())
    z_count = min(layout.z_count for layout in tile_layouts.values())
    shape_czyx = (channel_count, z_count, height, width)
    tile_shapes_czyx_by_file = {
        str(file.resolve()): [int(value) for value in tile_layouts[file.resolve()].shape_czyx]
        for file in files
    }
    for channel in args.channels:
        if channel < 0 or channel >= channel_count:
            raise ValueError(f"Channel {channel} out of range for {channel_count} channels")

    channel_caches = {
        channel: ensure_channel_cache(
            fit_module,
            files,
            channel=channel,
            source_level=args.source_level,
            source_axes=source_axes,
            shape_czyx=shape_czyx,
            source_dtype=source_dtype,
            tile_layouts=tile_layouts,
            tile_shapes_czyx_by_file=tile_shapes_czyx_by_file,
            args=args,
        )[0]
        for channel in args.channels
    }
    combined_path = args.output_dir / "sample-cache" / f"{args.label}-combined-channels-selected-slices-n{args.fit_total_slices}.float32.npy"
    images, per_channel_counts = copy_to_combined_stack(
        channel_caches,
        total_slices=args.fit_total_slices,
        output_path=combined_path,
        overwrite=args.overwrite,
    )

    basic = BaSiC(
        max_iterations=1000,
        smoothness_flatfield=args.smoothness_flatfield,
        fitting_mode=args.fitting_mode,
        working_size=args.working_size,
        sort_intensity=True,
        get_darkfield=True,
        device=args.device,
    )
    print(f"Running joint-channel BaSiC autotune: label={args.label}", flush=True)
    basic.autotune(images, is_timelapse=False, skip_shape_warning=True)
    print(
        "Autotune selected "
        f"smoothness_flatfield={basic.smoothness_flatfield}, "
        f"smoothness_darkfield={basic.smoothness_darkfield}",
        flush=True,
    )
    basic.fit(images, skip_shape_warning=True)
    flatfield = np.asarray(basic.flatfield, dtype=np.float32)
    darkfield = np.asarray(basic.darkfield, dtype=np.float32)

    for channel in args.channels:
        channel_name = f"ch{channel}"
        imwrite(args.output_dir / f"{args.label}-{channel_name}-flatfield.tif", flatfield)
        imwrite(args.output_dir / f"{args.label}-{channel_name}-darkfield.tif", darkfield)

    basic = fit_module.make_basic_pickle_portable(basic)
    for channel in args.channels:
        channel_name = f"ch{channel}"
        with (args.output_dir / f"{args.label}-{channel_name}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "basic": basic,
                    "path": str(args.output_dir.resolve()),
                    "name": args.label,
                    "channel": channel_name,
                    "shared_profile": True,
                    "training_channels": [f"ch{channel}" for channel in args.channels],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                handle,
            )
        fit_module.plot_basic(basic)
        fit_module.plt.savefig(args.output_dir / f"{args.label}-{channel_name}.png", dpi=150, bbox_inches="tight")
        fit_module.plt.close()

    manifest = {
        "label": args.label,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Single shared BaSiC profile trained by concatenating selected planes from multiple channels.",
        "input_dirs": [str(path) for path in input_dirs],
        "sample_cache_dir": str((args.output_dir / "sample-cache").resolve()),
        "input_caches": {f"ch{channel}": str(path) for channel, path in channel_caches.items()},
        "combined_stack": str(combined_path),
        "combined_shape": [int(value) for value in images.shape],
        "combined_total_slices": int(args.fit_total_slices),
        "channel_cache_sample_policy": "evenly spaced slices from each cached channel stack",
        "per_channel_samples": per_channel_counts,
        "basic_settings": {
            "get_darkfield": True,
            "autotune": True,
            "autotune_is_timelapse": False,
            "sort_intensity": True,
            "fitting_mode": args.fitting_mode,
            "working_size": int(args.working_size),
            "device": args.device,
            "smoothness_flatfield": float(basic.smoothness_flatfield),
            "smoothness_darkfield": float(basic.smoothness_darkfield),
        },
        "outputs": {
            "shared_profile_saved_as_channels": [int(channel) for channel in args.channels],
            "flatfield_min": float(np.nanmin(flatfield)),
            "flatfield_max": float(np.nanmax(flatfield)),
            "darkfield_min": float(np.nanmin(darkfield)),
            "darkfield_max": float(np.nanmax(darkfield)),
            "finite_flatfield": bool(np.isfinite(flatfield).all()),
            "finite_darkfield": bool(np.isfinite(darkfield).all()),
        },
    }
    manifest_path = args.output_dir / f"{args.label}-sampling.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
