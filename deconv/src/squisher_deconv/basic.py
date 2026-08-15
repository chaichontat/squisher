from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Callable

import numpy as np
import tifffile

from squisher_deconv.gpu import _validate_basic_profile
from squisher_deconv.source import TiffLogicalSource


@dataclass(frozen=True, slots=True)
class BasicFitOutputs:
    profile_paths: tuple[Path, ...]
    flatfield_paths: tuple[Path, ...]
    darkfield_paths: tuple[Path, ...]
    png_paths: tuple[Path, ...]
    manifest: Path


@dataclass(frozen=True, slots=True)
class SliceStats:
    path: Path
    z: int
    signal: float
    nonzero_fraction: float
    edge_profile_jump: float
    edge_band_delta: float


class _TiffPlaneReader:
    """Retain TIFF directory indexes across all channel sampling passes."""

    def __init__(self) -> None:
        self._tiffs: dict[Path, tifffile.TiffFile] = {}

    def __enter__(self) -> "_TiffPlaneReader":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        for tif in self._tiffs.values():
            tif.close()

    def read(self, source: TiffLogicalSource, *, channel: int, z: int) -> np.ndarray:
        tif = self._tiffs.get(source.path)
        if tif is None:
            tif = tifffile.TiffFile(source.path)
            try:
                _validate_sampled_tiff(tif, source)
            except BaseException:
                tif.close()
                raise
            self._tiffs[source.path] = tif
        return _read_plane(tif, source, channel=channel, z=z)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _write_pickle_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(payload))
    temporary.replace(path)


def _file_record(source: TiffLogicalSource) -> dict[str, Any]:
    stat = source.path.stat()
    return {
        "path": str(source.path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "axes": source.axes,
        "shape_zcyx": [source.z_count, source.channels, source.height, source.width],
        "dtype": source.dtype,
        "layout_metadata_hash": source.metadata.metadata_hash,
    }


def _inspect_sources(
    inputs: list[Path],
    *,
    channels: int,
    progress: Callable[[str], None],
) -> list[TiffLogicalSource]:
    if not inputs:
        raise ValueError("at least one OME-TIFF input is required")
    representative = Path(inputs[0]).resolve()
    progress(f"basic inspect representative=1/{len(inputs)} path={representative}")
    first = TiffLogicalSource.open(representative, channels=channels, metadata_mode="summary")
    sources = [first, *(replace(first, path=Path(path).resolve()) for path in inputs[1:])]
    progress(
        f"basic inspect complete representative={representative} assumed_inputs={len(sources)} "
        f"channels={first.channels} xy={first.height}x{first.width} dtype={first.dtype}"
    )
    return sources


def _edge_stats(sample: np.ndarray, *, signal: float) -> tuple[float, float]:
    if sample.ndim != 2 or min(sample.shape) < 4:
        return 0.0, 0.0
    threshold = float(np.percentile(sample, 50.0)) + 0.25 * max(signal, 1.0)
    mask = np.asarray(sample > threshold)
    row_profile = mask.mean(axis=1)
    column_profile = mask.mean(axis=0)
    kernel_size = min(9, row_profile.size, column_profile.size)
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    row_smooth = np.convolve(row_profile, kernel, mode="same")
    column_smooth = np.convolve(column_profile, kernel, mode="same")
    profile_jump = max(
        float(np.max(np.abs(np.diff(row_smooth)))),
        float(np.max(np.abs(np.diff(column_smooth)))),
    )

    band_y = max(3, sample.shape[0] // 8)
    band_x = max(3, sample.shape[1] // 8)
    center_y = row_profile[sample.shape[0] // 3 : 2 * sample.shape[0] // 3]
    center_x = column_profile[sample.shape[1] // 3 : 2 * sample.shape[1] // 3]
    if center_y.size == 0 or center_x.size == 0:
        return profile_jump, 0.0
    band_delta = max(
        abs(float(row_profile[:band_y].mean()) - float(center_y.mean())),
        abs(float(row_profile[-band_y:].mean()) - float(center_y.mean())),
        abs(float(column_profile[:band_x].mean()) - float(center_x.mean())),
        abs(float(column_profile[-band_x:].mean()) - float(center_x.mean())),
    )
    return profile_jump, band_delta


def _slice_stats(path: Path, z: int, plane: np.ndarray, *, stride: int) -> SliceStats:
    sample = np.asarray(plane[::stride, ::stride], dtype=np.float32)
    flattened = sample.reshape(-1)
    signal = float(np.percentile(flattened, 99.0) - np.percentile(flattened, 50.0))
    nonzero_fraction = float(np.count_nonzero(flattened) / flattened.size)
    edge_profile_jump, edge_band_delta = _edge_stats(sample, signal=signal)
    return SliceStats(
        path=path,
        z=int(z),
        signal=signal,
        nonzero_fraction=nonzero_fraction,
        edge_profile_jump=edge_profile_jump,
        edge_band_delta=edge_band_delta,
    )


def _candidate_order(
    sources: list[TiffLogicalSource], *, seed: int
) -> list[tuple[TiffLogicalSource, int]]:
    rng = np.random.default_rng(seed)
    z_orders = [rng.permutation(source.z_count) for source in sources]
    return [
        (source, int(z_orders[source_index][z_offset]))
        for z_offset in range(max(source.z_count for source in sources))
        for source_index, source in enumerate(sources)
        if z_offset < source.z_count
    ]


def _sample_sources(
    sources: list[TiffLogicalSource], *, target: int, samples_per_tile: int
) -> list[TiffLogicalSource]:
    """Select evenly distributed tiles while amortizing each TIFF directory scan."""
    source_count = min(len(sources), math.ceil(target / samples_per_tile))
    indices = np.linspace(0, len(sources) - 1, source_count, dtype=np.int64)
    return [sources[int(index)] for index in indices]


def _read_stats_batch(
    candidates: list[tuple[TiffLogicalSource, int]],
    *,
    channel: int,
    stride: int,
    progress: Callable[[str], None],
    reader: _TiffPlaneReader,
) -> list[SliceStats]:
    by_source: dict[Path, tuple[TiffLogicalSource, list[int]]] = {}
    for source, z in candidates:
        entry = by_source.setdefault(source.path, (source, []))
        entry[1].append(z)
    stats = []
    for index, (source, z_indices) in enumerate(by_source.values(), start=1):
        progress(
            f"basic sample source={index}/{len(by_source)} channel={channel} "
            f"planes={len(z_indices)} path={source.path}"
        )
        for z in sorted(z_indices, key=lambda value: source.page_key(channel=channel, z=value)):
            page = reader.read(source, channel=channel, z=z)
            stats.append(_slice_stats(source.path, z, page, stride=stride))
    return stats


def _validate_sampled_tiff(tif: tifffile.TiffFile, source: TiffLogicalSource) -> None:
    page = tif.pages[0]
    if tuple(page.shape) != (source.height, source.width) or str(page.dtype) != source.dtype:
        raise ValueError(
            f"BaSiC sampled input page layout mismatch in {source.path}: expected "
            f"shape/dtype={(source.height, source.width)}/{source.dtype}, got "
            f"{tuple(page.shape)}/{page.dtype}"
        )


def _read_plane(
    tif: tifffile.TiffFile,
    source: TiffLogicalSource,
    *,
    channel: int,
    z: int,
) -> np.ndarray:
    page_key = source.page_key(channel=channel, z=z)
    try:
        page = tif.pages[page_key].asarray()
    except IndexError as exc:
        raise ValueError(
            f"BaSiC sampled input {source.path} does not contain required page {page_key} "
            f"for channel={channel}, z={z}"
        ) from exc
    if page.shape != (source.height, source.width) or str(page.dtype) != source.dtype:
        raise ValueError(
            f"BaSiC sampled page layout mismatch in {source.path} at channel={channel}, z={z}: "
            f"expected {(source.height, source.width)}/{source.dtype}, got {page.shape}/{page.dtype}"
        )
    return page


def _eligible_stats(
    stats: list[SliceStats],
    *,
    min_relative_signal: float,
    min_nonzero_fraction: float,
    exclude_blank_slices: bool,
    exclude_edge_slices: bool,
    edge_min_profile_jump: float,
    edge_min_band_delta: float,
) -> tuple[list[SliceStats], float]:
    signals = np.asarray([stat.signal for stat in stats], dtype=np.float64)
    positive = signals[signals > 0]
    reference = float(np.percentile(positive, 75.0)) if positive.size else 0.0
    signal_threshold = reference * min_relative_signal if exclude_blank_slices else 0.0
    eligible = [
        stat
        for stat in stats
        if (
            not exclude_blank_slices
            or (stat.signal >= signal_threshold and stat.nonzero_fraction >= min_nonzero_fraction)
        )
        and (
            not exclude_edge_slices
            or stat.edge_profile_jump < edge_min_profile_jump
            or stat.edge_band_delta < edge_min_band_delta
        )
    ]
    return eligible, signal_threshold


def _select_training_slices(
    sources: list[TiffLogicalSource],
    *,
    channel: int,
    target: int,
    stride: int,
    min_relative_signal: float,
    min_nonzero_fraction: float,
    exclude_blank_slices: bool,
    exclude_edge_slices: bool,
    edge_min_profile_jump: float,
    edge_min_band_delta: float,
    seed: int,
    progress: Callable[[str], None],
    reader: _TiffPlaneReader,
) -> tuple[list[SliceStats], dict[str, Any]]:
    candidates = _candidate_order(sources, seed=seed)
    if target > len(candidates):
        raise ValueError(
            f"requested {target} BaSiC samples for channel {channel}, but only {len(candidates)} tile-Z planes exist"
        )
    scanned: list[SliceStats] = []
    eligible: list[SliceStats] = []
    signal_threshold = 0.0
    batch_size = max(64, target)
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        scanned.extend(
            _read_stats_batch(
                batch,
                channel=channel,
                stride=stride,
                progress=progress,
                reader=reader,
            )
        )
        eligible, signal_threshold = _eligible_stats(
            scanned,
            min_relative_signal=min_relative_signal,
            min_nonzero_fraction=min_nonzero_fraction,
            exclude_blank_slices=exclude_blank_slices,
            exclude_edge_slices=exclude_edge_slices,
            edge_min_profile_jump=edge_min_profile_jump,
            edge_min_band_delta=edge_min_band_delta,
        )
        progress(f"basic channel={channel} scanned={len(scanned)} eligible={len(eligible)}/{target}")
        if len(eligible) >= target:
            break
    if len(eligible) < target:
        raise ValueError(
            f"blank/edge QC selected only {len(eligible)}/{target} slices for channel {channel} "
            f"after scanning all {len(scanned)} planes"
        )
    selected = sorted(eligible, key=lambda stat: stat.signal, reverse=True)[:target]
    selected.sort(key=lambda stat: (str(stat.path), stat.z))
    return selected, {
        "scanned": len(scanned),
        "eligible": len(eligible),
        "selected": len(selected),
        "signal_threshold": signal_threshold,
        "blank_rejected": sum(
            stat.signal < signal_threshold or stat.nonzero_fraction < min_nonzero_fraction
            for stat in scanned
        ),
        "edge_rejected": sum(
            stat.edge_profile_jump >= edge_min_profile_jump
            and stat.edge_band_delta >= edge_min_band_delta
            for stat in scanned
        )
        if exclude_edge_slices
        else 0,
    }


def _cache_key(
    sources: list[TiffLogicalSource],
    *,
    channel: int,
    target: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "inputs": [_file_record(source) for source in sources],
        "channel": int(channel),
        "target": int(target),
        "settings": settings,
    }


def _load_valid_cache(data_path: Path, metadata_path: Path, *, key: dict[str, Any]) -> np.ndarray | None:
    if not data_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("cache_key") != key:
        return None
    data = np.load(data_path, mmap_mode="r")
    if list(data.shape) != metadata.get("shape") or str(data.dtype) != metadata.get("dtype"):
        return None
    return data


def _write_channel_cache(
    *,
    selected: list[SliceStats],
    source_by_path: dict[Path, TiffLogicalSource],
    channel: int,
    data_path: Path,
    metadata_path: Path,
    key: dict[str, Any],
    reader: _TiffPlaneReader,
) -> np.ndarray:
    first = next(iter(source_by_path.values()))
    temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp")
    data = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.dtype(first.dtype),
        shape=(len(selected), first.height, first.width),
    )
    selected_by_path: dict[Path, list[tuple[int, SliceStats]]] = {}
    for index, stat in enumerate(selected):
        selected_by_path.setdefault(stat.path, []).append((index, stat))
    for path, indexed_stats in selected_by_path.items():
        source = source_by_path[path]
        for index, stat in sorted(
            indexed_stats,
            key=lambda item: source.page_key(channel=channel, z=item[1].z),
        ):
            data[index] = reader.read(source, channel=channel, z=stat.z)
    data.flush()
    del data
    temporary.replace(data_path)
    _write_json_atomic(
        metadata_path,
        {
            "cache_key": key,
            "data_path": str(data_path.resolve()),
            "shape": [len(selected), first.height, first.width],
            "dtype": str(np.dtype(first.dtype)),
            "selected": [{"tile": str(stat.path), "z": stat.z} for stat in selected],
        },
    )
    return np.load(data_path, mmap_mode="r")


def _channel_cache(
    sources: list[TiffLogicalSource],
    *,
    out_dir: Path,
    label: str,
    channel: int,
    target: int,
    sampling_settings: dict[str, Any],
    seed: int,
    progress: Callable[[str], None],
    reader: _TiffPlaneReader,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir = out_dir / "sample-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{label}-ch{channel}-selected-slices.npy"
    metadata_path = cache_dir / f"{label}-ch{channel}-selected-slices.json"
    key = _cache_key(sources, channel=channel, target=target, settings=sampling_settings)
    cached = _load_valid_cache(data_path, metadata_path, key=key)
    if cached is not None:
        progress(f"basic channel={channel} reusing cache={data_path}")
        return cached, {"selected": int(cached.shape[0]), "reused": True, "cache": str(data_path)}

    selected, diagnostics = _select_training_slices(
        sources,
        channel=channel,
        target=target,
        stride=int(sampling_settings["blank_slice_sample_stride"]),
        min_relative_signal=float(sampling_settings["blank_slice_min_relative_signal"]),
        min_nonzero_fraction=float(sampling_settings["blank_slice_min_nonzero_fraction"]),
        exclude_blank_slices=bool(sampling_settings["exclude_blank_slices"]),
        exclude_edge_slices=bool(sampling_settings["exclude_edge_slices"]),
        edge_min_profile_jump=float(sampling_settings["edge_slice_min_profile_jump"]),
        edge_min_band_delta=float(sampling_settings["edge_slice_min_band_delta"]),
        seed=seed,
        progress=progress,
        reader=reader,
    )
    source_by_path = {source.path: source for source in sources}
    data = _write_channel_cache(
        selected=selected,
        source_by_path=source_by_path,
        channel=channel,
        data_path=data_path,
        metadata_path=metadata_path,
        key=key,
        reader=reader,
    )
    return data, {**diagnostics, "reused": False, "cache": str(data_path)}


def _combined_stack(
    caches: dict[int, np.ndarray], *, samples: int, output: Path
) -> tuple[np.ndarray, dict[str, int]]:
    channels = list(caches)
    per_channel = samples // len(channels)
    remainder = samples % len(channels)
    counts = {
        channel: per_channel + (1 if index < remainder else 0)
        for index, channel in enumerate(channels)
    }
    first = caches[channels[0]]
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    combined = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(samples, first.shape[1], first.shape[2]),
    )
    offset = 0
    for channel in channels:
        cache = caches[channel]
        count = counts[channel]
        if cache.shape[0] < count:
            raise ValueError(f"channel {channel} cache has {cache.shape[0]} slices but {count} are required")
        indices = np.linspace(0, cache.shape[0] - 1, count, dtype=np.int64)
        combined[offset : offset + count] = cache[indices]
        offset += count
    combined.flush()
    del combined
    temporary.replace(output)
    return np.load(output, mmap_mode="r"), {f"ch{channel}": count for channel, count in counts.items()}


def _portable_basic(basic: Any) -> Any:
    if hasattr(basic, "device"):
        basic.device = "cpu"
    private = getattr(basic, "__pydantic_private__", None)
    if isinstance(private, dict):
        for name, value in private.items():
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                private[name] = value.detach().cpu()
    return basic


def _write_basic_qc_png(path: Path, basic: Any) -> None:
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    image = axes[0].imshow(basic.flatfield)
    fig.colorbar(image, ax=axes[0])
    axes[0].set_title("Flatfield")
    axes[0].axis("off")
    image = axes[1].imshow(basic.darkfield)
    fig.colorbar(image, ax=axes[1])
    axes[1].set_title("Darkfield")
    axes[1].axis("off")
    axes[2].plot(basic.baseline)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_basic_profiles(
    *,
    inputs: list[Path],
    out_dir: Path,
    label: str,
    channels: int,
    samples: int = 500,
    cache_samples_per_channel: int | None = None,
    samples_per_tile: int = 25,
    blank_slice_sample_stride: int = 16,
    blank_slice_min_relative_signal: float = 0.10,
    blank_slice_min_nonzero_fraction: float = 1e-4,
    exclude_blank_slices: bool = True,
    exclude_edge_slices: bool = True,
    edge_slice_min_profile_jump: float = 0.05,
    edge_slice_min_band_delta: float = 0.35,
    smoothness_flatfield: float = 1.8,
    fitting_mode: str = "approximate",
    working_size: int = 128,
    device: str = "cuda",
    seed: int = 20260709,
    basic_factory: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BasicFitOutputs:
    """Fit one autotuned darkfield BaSiC model shared across all channels."""
    if channels < 1 or samples < 1:
        raise ValueError("channels and samples must be positive")
    if cache_samples_per_channel is None:
        cache_samples_per_channel = int(np.ceil(samples / channels))
    if cache_samples_per_channel < 1:
        raise ValueError("cache_samples_per_channel must be positive")
    if samples_per_tile < 1:
        raise ValueError("samples_per_tile must be positive")
    if samples < channels:
        raise ValueError(f"samples={samples} must be at least channels={channels}")
    if cache_samples_per_channel < int(np.ceil(samples / channels)):
        raise ValueError("cache_samples_per_channel is too small for the requested joint fit")
    if blank_slice_sample_stride < 1:
        raise ValueError("blank_slice_sample_stride must be positive")
    if not 0 <= blank_slice_min_nonzero_fraction <= 1:
        raise ValueError("blank_slice_min_nonzero_fraction must be in [0, 1]")
    if fitting_mode not in {"approximate", "ladmap"}:
        raise ValueError("fitting_mode must be 'approximate' or 'ladmap'")

    profile_paths = tuple(out_dir / f"{label}-ch{channel}.pkl" for channel in range(channels))
    flatfield_paths = tuple(out_dir / f"{label}-ch{channel}-flatfield.tif" for channel in range(channels))
    darkfield_paths = tuple(out_dir / f"{label}-ch{channel}-darkfield.tif" for channel in range(channels))
    png_paths = tuple(out_dir / f"{label}-ch{channel}.png" for channel in range(channels))
    manifest_path = out_dir / f"{label}-sampling.json"
    existing = [
        path
        for path in (*profile_paths, *flatfield_paths, *darkfield_paths, *png_paths, manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing BaSiC output(s): {existing}")

    progress = progress or (lambda _message: None)
    progress(
        "basic start "
        f"inputs={len(inputs)} channels={channels} samples={samples} "
        f"cache_samples_per_channel={cache_samples_per_channel} device={device} "
        f"out_dir={out_dir.resolve()}"
    )
    sources = _inspect_sources(inputs, channels=channels, progress=progress)
    sampled_sources = _sample_sources(
        sources,
        target=cache_samples_per_channel,
        samples_per_tile=samples_per_tile,
    )
    progress(
        f"basic sampling inputs={len(sampled_sources)}/{len(sources)} "
        f"samples_per_tile={samples_per_tile}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling_settings = {
        "policy": "balanced_tile_z",
        "samples_per_tile": int(samples_per_tile),
        "sampled_inputs": [str(source.path) for source in sampled_sources],
        "blank_slice_sample_stride": int(blank_slice_sample_stride),
        "blank_slice_min_relative_signal": float(blank_slice_min_relative_signal),
        "blank_slice_min_nonzero_fraction": float(blank_slice_min_nonzero_fraction),
        "exclude_blank_slices": bool(exclude_blank_slices),
        "exclude_edge_slices": bool(exclude_edge_slices),
        "edge_slice_min_profile_jump": float(edge_slice_min_profile_jump),
        "edge_slice_min_band_delta": float(edge_slice_min_band_delta),
        "seed": int(seed),
    }
    caches = {}
    sampling_diagnostics = {}
    with _TiffPlaneReader() as reader:
        for channel in range(channels):
            cache, diagnostics = _channel_cache(
                sampled_sources,
                out_dir=out_dir,
                label=label,
                channel=channel,
                target=cache_samples_per_channel,
                sampling_settings=sampling_settings,
                seed=seed,
                progress=progress,
                reader=reader,
            )
            caches[channel] = cache
            sampling_diagnostics[f"ch{channel}"] = diagnostics

    combined_path = out_dir / "sample-cache" / f"{label}-joint-selected-slices-n{samples}.float32.npy"
    images, per_channel_samples = _combined_stack(caches, samples=samples, output=combined_path)
    if basic_factory is None:
        from basicpy import BaSiC

        basic_factory = BaSiC
    kwargs: dict[str, Any] = {
        "max_iterations": 1000,
        "smoothness_flatfield": smoothness_flatfield,
        "fitting_mode": fitting_mode,
        "working_size": working_size,
        "sort_intensity": True,
        "get_darkfield": True,
    }
    if device != "none":
        if "device" not in getattr(basic_factory, "model_fields", {}):
            raise ValueError(f"This BaSiCPy version does not support device={device!r}")
        kwargs["device"] = device
    basic = basic_factory(**kwargs)
    progress(f"basic autotune samples={images.shape} device={device}")
    basic.autotune(images, is_timelapse=False, skip_shape_warning=True)
    progress(
        "basic fit "
        f"smoothness_flatfield={basic.smoothness_flatfield} "
        f"smoothness_darkfield={basic.smoothness_darkfield}"
    )
    basic.fit(images, skip_shape_warning=True)
    flatfield = np.asarray(basic.flatfield, dtype=np.float32)
    darkfield = np.asarray(basic.darkfield, dtype=np.float32)
    _validate_basic_profile(manifest_path, dark=darkfield, flat=flatfield)
    basic = _portable_basic(basic)

    created_at = datetime.now(timezone.utc).isoformat()
    run_settings = {
        "autotune": True,
        "autotune_is_timelapse": False,
        "get_darkfield": True,
        "sort_intensity": True,
        "shared_profile": True,
        "channels": int(channels),
        "samples": int(samples),
        "cache_samples_per_channel": int(cache_samples_per_channel),
        "fitting_mode": fitting_mode,
        "working_size": int(working_size),
        "device": device,
        "smoothness_flatfield": float(basic.smoothness_flatfield),
        "smoothness_darkfield": float(basic.smoothness_darkfield),
        **sampling_settings,
    }
    for channel, (profile_path, flatfield_path, darkfield_path, png_path) in enumerate(
        zip(profile_paths, flatfield_paths, darkfield_paths, png_paths, strict=True)
    ):
        _write_pickle_atomic(
            profile_path,
            {
                "basic": basic,
                "path": str(out_dir.resolve()),
                "name": label,
                "channel": f"ch{channel}",
                "shared_profile": True,
                "training_channels": [f"ch{index}" for index in range(channels)],
                "created_at": created_at,
                "run_settings": run_settings,
                "manifest": str(manifest_path.resolve()),
            },
        )
        tifffile.imwrite(flatfield_path, flatfield, photometric="minisblack")
        tifffile.imwrite(darkfield_path, darkfield, photometric="minisblack")
        _write_basic_qc_png(png_path, basic)

    manifest = {
        "schema_version": 1,
        "artifact_type": "squisher_deconv.basic_fit.v1",
        "label": label,
        "created_at": created_at,
        "description": "Joint-channel autotuned BaSiC profile with fitted darkfield.",
        "inputs": [_file_record(source) for source in sources],
        "sample_cache_dir": str((out_dir / "sample-cache").resolve()),
        "combined_stack": str(combined_path.resolve()),
        "combined_shape": [int(value) for value in images.shape],
        "per_channel_samples": per_channel_samples,
        "sampling": {
            "settings": sampling_settings,
            "selected_per_channel": {
                channel: int(diagnostics["selected"])
                for channel, diagnostics in sampling_diagnostics.items()
            },
            "diagnostics": sampling_diagnostics,
        },
        "basic_settings": run_settings,
        "outputs": {
            "profiles": [str(path.resolve()) for path in profile_paths],
            "flatfields": [str(path.resolve()) for path in flatfield_paths],
            "darkfields": [str(path.resolve()) for path in darkfield_paths],
            "pngs": [str(path.resolve()) for path in png_paths],
            "flatfield_min": float(np.min(flatfield)),
            "flatfield_max": float(np.max(flatfield)),
            "darkfield_min": float(np.min(darkfield)),
            "darkfield_max": float(np.max(darkfield)),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    progress(
        f"basic complete manifest={manifest_path.resolve()} profiles={len(profile_paths)}"
    )
    return BasicFitOutputs(
        profile_paths=profile_paths,
        flatfield_paths=flatfield_paths,
        darkfield_paths=darkfield_paths,
        png_paths=png_paths,
        manifest=manifest_path,
    )
