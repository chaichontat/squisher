#!/usr/bin/env python
"""Register and fuse the 20x-TL-561638 OME-TIFF tiles with multiview-stitcher.

The tile translations are read from each tile's OME Plane PositionX/Y
metadata.  PositionZ is not present in these files, so z translation is set to
zero.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from contextlib import contextmanager, nullcontext
import copy
from datetime import datetime
import hashlib
import io
import json
import math
import os
import tempfile
import time
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from zarr.abc.codec import ArrayBytesCodec


try:
    profile
except NameError:
    def profile(func):
        return func


TRANSFORM_KEY = "stage_metadata"
REGISTERED_TRANSFORM_KEY = "translation_registered"
FUSION_BACKEND = "cupy"
DEFAULT_REGISTRATION_READ_CHUNK_Z = 128
PHASE_CORRELATION_UPSAMPLE_FACTOR = 10
NGFF_VERSION = "0.5"
ZSTD_LEVEL = 0
DEFAULT_JPEGXR_LEVEL = 0.7
JPEGXR_CODEC_NAME = "fishtools.jpegxr"
_JPEGXR_CODEC_REGISTERED = False
_LOG_FILE: Path | None = None


@dataclass(frozen=True)
class FilesystemProgressSnapshot:
    exists: bool
    files: int = 0
    dirs: int = 0
    bytes: int = 0
    newest_mtime: float | None = None


class ProfileFusionStop(RuntimeError):
    """Internal stop signal for bounded fusion profiling runs."""


@contextmanager
def cupy_cleanup_context(per_chunk_cleanup: bool):
    if per_chunk_cleanup:
        yield
        return

    from multiview_stitcher import misc_utils

    def skip_cupy_cleanup():
        return False

    originals: list[tuple[Any, str, Any]] = [
        (misc_utils, "clear_cupy_memory", misc_utils.clear_cupy_memory)
    ]
    try:
        from multiview_stitcher import weights

        if hasattr(weights, "clear_cupy_memory"):
            originals.append((weights, "clear_cupy_memory", weights.clear_cupy_memory))
    except ImportError:
        pass

    for module, name, _original in originals:
        setattr(module, name, skip_cupy_cleanup)
    try:
        yield
    finally:
        originals[0][2]()
        for module, name, original in originals:
            setattr(module, name, original)


def basic_array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()[:16]


def basic_disk_cache_paths(cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(cache_key.encode()).hexdigest()
    return cache_dir / f"{digest}.npy", cache_dir / f"{digest}.json"


def read_basic_disk_cache_entry(
    cache_dir: Path,
    cache_key: str,
) -> dict[str, Any] | None:
    data_path, metadata_path = basic_disk_cache_paths(cache_dir, cache_key)
    if not data_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("cache_key") != cache_key:
        raise ValueError(f"BaSiC disk cache metadata mismatch for {metadata_path}")
    data = np.load(data_path, allow_pickle=False)
    return {
        "data": data,
        "dims": tuple(metadata["dims"]),
        "offsets": {dim: int(offset) for dim, offset in metadata["offsets"].items()},
        "bytes": int(data.nbytes),
    }


def write_basic_disk_cache_entry(
    cache_dir: Path,
    cache_key: str,
    cached: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, metadata_path = basic_disk_cache_paths(cache_dir, cache_key)
    suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
    tmp_data_path = data_path.with_name(f"{data_path.name}{suffix}")
    tmp_metadata_path = metadata_path.with_name(f"{metadata_path.name}{suffix}")
    with tmp_data_path.open("wb") as handle:
        np.save(handle, cached["data"], allow_pickle=False)
    metadata = {
        "cache_key": cache_key,
        "dims": list(cached["dims"]),
        "offsets": {dim: int(offset) for dim, offset in cached["offsets"].items()},
        "shape": [int(value) for value in cached["data"].shape],
        "dtype": str(cached["data"].dtype),
        "bytes": int(cached["data"].nbytes),
    }
    tmp_metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n")
    os.replace(tmp_data_path, data_path)
    os.replace(tmp_metadata_path, metadata_path)


@contextmanager
def temporary_basic_disk_cache_dir(base_dir: Path | None, output: Path):
    root = output.parent if base_dir is None else base_dir
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.basic-cache-",
        dir=root,
    ) as cache_dir:
        yield Path(cache_dir)


def make_profile_limited_batch_func(
    max_chunks: int,
    process_batch_func: Any,
    *,
    skip_chunks: int = 0,
    log_func=None,
):
    state = {"processed": 0, "skipped": 0}
    logger = log if log_func is None else log_func

    def process_profile_limited_batch(func, block_ids, n_jobs=4, backend="threading"):
        block_ids = list(block_ids)
        if state["skipped"] < skip_chunks:
            remaining_skip = skip_chunks - state["skipped"]
            skipped_now = min(remaining_skip, len(block_ids))
            state["skipped"] += skipped_now
            block_ids = block_ids[skipped_now:]
            if state["skipped"] >= skip_chunks:
                logger(f"Profile fusion chunk skip complete: {state['skipped']}/{skip_chunks}")
            if not block_ids:
                return
        remaining = max_chunks - state["processed"]
        if remaining <= 0:
            raise ProfileFusionStop
        selected_block_ids = block_ids[:remaining]
        if selected_block_ids:
            process_batch_func(
                func,
                selected_block_ids,
                n_jobs=n_jobs,
                backend=backend,
            )
            state["processed"] += len(selected_block_ids)
            logger(f"Profile fusion chunk limit: {state['processed']}/{max_chunks}")
        if state["processed"] >= max_chunks:
            raise ProfileFusionStop

    return process_profile_limited_batch


def normalize_or_uniform_valid(weights, valid, xp):
    totals = xp.sum(weights, axis=0, keepdims=True)
    zero_quality = (totals <= 0.0) & xp.any(valid, axis=0, keepdims=True)
    weights = xp.where(zero_quality & valid, 1.0, weights)
    totals = xp.sum(weights, axis=0, keepdims=True)
    return xp.where(totals > 0.0, weights / totals, 0.0)


def coarse_preibisch_content_weights(
    transformed_views,
    blending_weights,
    *,
    sigma_1: int,
    sigma_2: int,
    stride_zyx: tuple[int, int, int],
    softmax_exponent: float = 2.0,
):
    try:
        import cupy as cp
    except ImportError:  # pragma: no cover - depends on runtime environment
        cp = None

    if cp is not None and isinstance(transformed_views, cp.ndarray):
        import cupyx.scipy.ndimage as cpx_ndimage

        xp = cp
        gaussian_filter = cpx_ndimage.gaussian_filter
    else:
        import scipy.ndimage as scipy_ndimage

        xp = np
        gaussian_filter = scipy_ndimage.gaussian_filter

    spatial_ndim = transformed_views.ndim - 1
    strides = tuple(int(value) for value in stride_zyx[-spatial_ndim:])
    if any(value <= 0 for value in strides):
        raise ValueError("content Preibisch coarse strides must be positive")

    ds_slices = (slice(None),) + tuple(slice(None, None, stride) for stride in strides)
    views_ds = transformed_views[ds_slices].astype(xp.float32, copy=False)
    blending_ds = blending_weights[ds_slices]
    valid_ds = (blending_ds > 1e-7) & ~xp.isnan(views_ds)
    filled = xp.where(valid_ds, views_ds, 0.0)

    sigma1_ds = (0.0,) + tuple(max(0.5, float(sigma_1) / stride) for stride in strides)
    sigma2_ds = (0.0,) + tuple(max(0.5, float(sigma_2) / stride) for stride in strides)
    low_pass = gaussian_filter(filled, sigma=sigma1_ds, mode="reflect")
    quality = gaussian_filter(xp.where(valid_ds, (filled - low_pass) ** 2, 0.0), sigma=sigma2_ds, mode="reflect")
    quality = xp.where(valid_ds, quality, 0.0)
    quality = normalize_or_uniform_valid(quality, valid_ds, xp)

    weights = quality
    for axis, stride in enumerate(strides, start=1):
        if stride > 1:
            weights = xp.repeat(weights, stride, axis=axis)
    crop = (slice(None),) + tuple(slice(0, size) for size in transformed_views.shape[1:])
    weights = weights[crop]

    valid_full = (blending_weights > 1e-7) & ~xp.isnan(transformed_views)
    weights = xp.where(valid_full, weights, 0.0)
    weights = normalize_or_uniform_valid(weights, valid_full, xp)
    if softmax_exponent > 1.0:
        eps = xp.asarray(1e-12, dtype=weights.dtype)
        weights = xp.where(valid_full, xp.power(xp.maximum(weights, eps), float(softmax_exponent)), 0.0)
        weights = normalize_or_uniform_valid(weights, valid_full, xp)
    return weights.astype(xp.float32, copy=False)


def fusion_weight_config(args: argparse.Namespace, resolved_chunksize: dict[str, int]) -> tuple[Any | None, dict[str, Any] | None]:
    if args.fusion_weight_mode == "geometric":
        return None, None

    from multiview_stitcher.fusion import _core as fusion_core

    if args.fusion_weight_mode == "content-dct":
        if args.content_dct_size <= 0:
            raise ValueError("--content-dct-size must be positive")
        if args.content_dct_exponent <= 0.0:
            raise ValueError("--content-dct-exponent must be positive")
        otf_support_fraction = (
            None
            if args.content_dct_otf_support_fraction < 0.0
            else float(args.content_dct_otf_support_fraction)
        )
        return fusion_core.weights.content_based_dct, {
            "dct_size": int(args.content_dct_size),
            "exponent": float(args.content_dct_exponent),
            "otf_support_fraction": otf_support_fraction,
            "output_chunksize": {dim: int(resolved_chunksize[dim]) for dim in ("z", "y", "x")},
        }

    if args.fusion_weight_mode == "content-preibisch":
        if args.content_preibisch_sigma1 <= 0.0:
            raise ValueError("--content-preibisch-sigma1 must be positive")
        if args.content_preibisch_sigma2 <= 0.0:
            raise ValueError("--content-preibisch-sigma2 must be positive")
        return fusion_core.weights.content_based, {
            "sigma_1": int(args.content_preibisch_sigma1),
            "sigma_2": int(args.content_preibisch_sigma2),
        }

    if args.fusion_weight_mode == "content-preibisch-coarse":
        if args.content_preibisch_sigma1 <= 0.0:
            raise ValueError("--content-preibisch-sigma1 must be positive")
        if args.content_preibisch_sigma2 <= 0.0:
            raise ValueError("--content-preibisch-sigma2 must be positive")
        if args.content_preibisch_softmax_exponent < 1.0:
            raise ValueError("--content-preibisch-softmax-exponent must be at least 1.0")
        strides = tuple(int(value) for value in args.content_preibisch_coarse_stride)
        if any(value <= 0 for value in strides):
            raise ValueError("--content-preibisch-coarse-stride values must be positive")
        return coarse_preibisch_content_weights, {
            "sigma_1": int(args.content_preibisch_sigma1),
            "sigma_2": int(args.content_preibisch_sigma2),
            "stride_zyx": strides,
            "softmax_exponent": float(args.content_preibisch_softmax_exponent),
        }

    raise ValueError(f"Unsupported fusion weight mode {args.fusion_weight_mode!r}")


@dataclass(frozen=True)
class JpegxrZarrV3Codec(ArrayBytesCodec):
    """Zarr v3 array-to-bytes JPEG-XR codec for 2D image-plane chunks."""

    is_fixed_size = False
    level: float = DEFAULT_JPEGXR_LEVEL
    photometric: str = "minisblack"

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        if data.get("name") != JPEGXR_CODEC_NAME:
            raise ValueError(f"Expected codec name {JPEGXR_CODEC_NAME!r}, got {data.get('name')!r}")
        configuration = data.get("configuration") or {}
        return cls(**configuration)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": JPEGXR_CODEC_NAME,
            "configuration": {
                "level": self.level,
                "photometric": self.photometric,
            },
        }

    def validate(self, *, shape, dtype, chunk_grid) -> None:
        del shape, dtype
        chunk_shape = tuple(chunk_grid.chunk_shape)
        if len(chunk_shape) < 2 or any(size != 1 for size in chunk_shape[:-2]):
            raise ValueError(
                "JPEG-XR Zarr v3 output requires image-plane chunks with all "
                f"non-y/x chunk axes set to 1; got chunk_shape={chunk_shape}"
            )

    def _encode_sync(self, chunk_array, chunk_spec):
        import imagecodecs
        import numpy as np

        plane = np.asarray(chunk_array.as_ndarray_like())
        while plane.ndim > 2:
            if plane.shape[0] != 1:
                raise ValueError(f"JPEG-XR chunk axes before y/x must be singleton, got {plane.shape}")
            plane = plane[0]
        encoded = imagecodecs.jpegxr_encode(
            np.ascontiguousarray(plane),
            level=self.level,
            photometric=self.photometric,
        )
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(self, chunk_array, chunk_spec):
        return self._encode_sync(chunk_array, chunk_spec)

    def _decode_sync(self, chunk_bytes, chunk_spec):
        import imagecodecs
        import numpy as np

        decoded = np.asarray(imagecodecs.jpegxr_decode(chunk_bytes.to_bytes()))
        decoded = decoded.reshape(chunk_spec.shape)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(decoded)

    async def _decode_single(self, chunk_bytes, chunk_spec):
        return self._decode_sync(chunk_bytes, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec) -> int:
        del input_byte_length, chunk_spec
        raise NotImplementedError


def register_jpegxr_zarr_v3_codec() -> None:
    global _JPEGXR_CODEC_REGISTERED
    if _JPEGXR_CODEC_REGISTERED:
        return
    from zarr.registry import register_codec

    register_codec(JPEGXR_CODEC_NAME, JpegxrZarrV3Codec)
    _JPEGXR_CODEC_REGISTERED = True


def configure_log_file(path: Path | None) -> None:
    global _LOG_FILE
    _LOG_FILE = path.resolve() if path is not None else None
    if _LOG_FILE is not None:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE.write_text("")


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()


def current_rss_gb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None

    for line in status_path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024 / 1024
    return None


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise RuntimeError("unreachable byte formatter state")


def filesystem_progress_snapshot(path: Path) -> FilesystemProgressSnapshot:
    if not path.exists():
        return FilesystemProgressSnapshot(exists=False)

    files = 0
    dirs = 0
    total_bytes = 0
    newest_mtime: float | None = None
    for root, dirnames, filenames in os.walk(path):
        dirs += len(dirnames)
        for filename in filenames:
            files += 1
            file_path = Path(root) / filename
            stat = file_path.stat()
            total_bytes += int(stat.st_size)
            newest_mtime = stat.st_mtime if newest_mtime is None else max(newest_mtime, stat.st_mtime)
    return FilesystemProgressSnapshot(
        exists=True,
        files=files,
        dirs=dirs,
        bytes=total_bytes,
        newest_mtime=newest_mtime,
    )


def format_filesystem_progress(path: Path, snapshot: FilesystemProgressSnapshot) -> str:
    if not snapshot.exists:
        return f"{path} not created yet"
    newest = "none"
    if snapshot.newest_mtime is not None:
        newest = datetime.fromtimestamp(snapshot.newest_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{path}: files={snapshot.files}, dirs={snapshot.dirs}, "
        f"bytes={format_bytes(snapshot.bytes)}, newest_file_mtime={newest}"
    )


@contextmanager
def heartbeat(label: str, *, every_seconds: int = 30):
    stop = threading.Event()
    started = time.monotonic()

    def emit() -> None:
        while not stop.wait(every_seconds):
            elapsed = time.monotonic() - started
            rss_gb = current_rss_gb()
            rss = f", rss={rss_gb:.1f} GiB" if rss_gb is not None else ""
            log(f"Still {label}: elapsed={elapsed:.0f}s{rss}")

    thread = threading.Thread(target=emit, name=f"{label} heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
        elapsed = time.monotonic() - started
        rss_gb = current_rss_gb()
        rss = f", rss={rss_gb:.1f} GiB" if rss_gb is not None else ""
        log(f"Finished {label}: elapsed={elapsed:.0f}s{rss}")


@contextmanager
def filesystem_progress(label: str, path: Path, *, every_seconds: int = 60):
    stop = threading.Event()
    started = time.monotonic()

    def emit(status: str) -> None:
        elapsed = time.monotonic() - started
        try:
            snapshot = filesystem_progress_snapshot(path)
        except OSError as exc:
            log(f"Filesystem progress {label} {status}: elapsed={elapsed:.0f}s, stat failed for {path}: {exc}")
            return
        log(
            f"Filesystem progress {label} {status}: "
            f"elapsed={elapsed:.0f}s, {format_filesystem_progress(path, snapshot)}"
        )

    def emit_until_done() -> None:
        while not stop.wait(every_seconds):
            emit("running")

    emit("start")
    thread = threading.Thread(target=emit_until_done, name=f"{label} filesystem progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
        emit("finished")


def dask_task_family(key: Any) -> str:
    head = key[0] if isinstance(key, tuple) else key
    text = str(head)
    family, separator, suffix = text.rpartition("-")
    if separator and suffix.replace(".", "").replace("_", "").isalnum():
        text = family
    return text[:100]


@contextmanager
def dask_progress(label: str, *, every_seconds: int = 10):
    from dask.callbacks import Callback

    class DetailedDaskLogger(Callback):
        def __init__(self, stage_label: str):
            super().__init__()
            self.stage_label = stage_label
            self.lock = threading.Lock()
            self.stop = threading.Event()
            self.started_at = time.monotonic()
            self.total_tasks = 0
            self.started_tasks = 0
            self.finished_tasks = 0
            self.running_families: Counter[str] = Counter()
            self.completed_families: Counter[str] = Counter()
            self.thread: threading.Thread | None = None

        def _start(self, dsk):
            self.total_tasks = len(dsk)
            log(f"Dask stage {self.stage_label}: total_tasks={self.total_tasks}")

        def _start_state(self, dsk, state):
            del dsk, state
            self.thread = threading.Thread(
                target=self._emit_until_done,
                name=f"{self.stage_label} dask logger",
                daemon=True,
            )
            self.thread.start()

        def _pretask(self, key, dsk, state):
            del dsk, state
            family = dask_task_family(key)
            with self.lock:
                self.started_tasks += 1
                self.running_families[family] += 1

        def _posttask(self, key, result, dsk, state, worker_id):
            del result, dsk, state, worker_id
            family = dask_task_family(key)
            with self.lock:
                if self.running_families[family] > 1:
                    self.running_families[family] -= 1
                else:
                    self.running_families.pop(family, None)
                self.finished_tasks += 1
                self.completed_families[family] += 1

        def _finish(self, dsk, state, failed):
            del dsk, state
            self.stop.set()
            if self.thread is not None:
                self.thread.join(timeout=1)
            status = "failed" if failed else "done"
            self._emit(status=status)

        def _emit_until_done(self) -> None:
            while not self.stop.wait(every_seconds):
                self._emit(status="running")

        def _emit(self, *, status: str) -> None:
            with self.lock:
                total_tasks = self.total_tasks
                started_tasks = self.started_tasks
                finished_tasks = self.finished_tasks
                running = ", ".join(
                    f"{family}={count}"
                    for family, count in self.running_families.most_common(5)
                )
                completed = ", ".join(
                    f"{family}={count}"
                    for family, count in self.completed_families.most_common(5)
                )
            elapsed = time.monotonic() - self.started_at
            percent = 100 * finished_tasks / total_tasks if total_tasks else 0.0
            rss_gb = current_rss_gb()
            rss = f", rss={rss_gb:.1f} GiB" if rss_gb is not None else ""
            log(
                f"Dask stage {self.stage_label} {status}: "
                f"elapsed={elapsed:.0f}s, started={started_tasks}/{total_tasks}, "
                f"finished={finished_tasks}/{total_tasks} ({percent:.1f}%), "
                f"running=[{running or 'none'}], completed_top=[{completed or 'none'}]{rss}"
            )

    with DetailedDaskLogger(label):
        yield


@dataclass(frozen=True)
class TrackMetadata:
    slug: str
    track_id: str
    channels: tuple[int, ...]
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class TileMetadata:
    path: Path
    shape: tuple[int, ...]
    axes: str
    spacing: dict[str, float]
    translation: dict[str, float]
    channels: tuple[str, ...]
    tracks: tuple[TrackMetadata, ...]
    stage_scale: dict[str, float] | None = None
    source_view: str | None = None


@dataclass(frozen=True)
class RobustBoundarySettings:
    patch_shape_zyx: tuple[int, int, int] = (64, 512, 512)
    max_patches_per_edge: int = 12
    min_nonzero_fraction: float = 0.05
    min_std: float = 1e-3
    content_mask_percentile: float = 60.0
    min_content_fraction: float = 0.002
    min_content_voxels: int = 4096
    min_correlation: float = 0.60
    min_improvement: float = 0.02
    min_stable_correlation: float = 0.75
    max_stable_shift_zyx: tuple[float, float, float] = (1.0, 2.0, 2.0)
    max_correction_zyx: tuple[float, float, float] = (16.0, 96.0, 96.0)
    max_final_residual_zyx: tuple[float, float, float] = (4.0, 8.0, 8.0)
    huber_delta: float = 4.0
    irls_iterations: int = 5
    reference_xy_prior_weight: float = 0.01


@dataclass(frozen=True)
class TileBounds:
    start_zyx: tuple[float, float, float]
    stop_zyx: tuple[float, float, float]


@dataclass(frozen=True)
class BoundaryPatchSpec:
    pair: tuple[int, int]
    axis: str
    patch_index: int
    fixed_slices: tuple[slice, slice, slice]
    moving_slices: tuple[slice, slice, slice]
    overlap_start_zyx: tuple[int, int, int]
    overlap_shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class BoundaryConstraint:
    fixed: int
    moving: int
    pair: tuple[int, int]
    axis: str
    patch_index: int
    shift_zyx: tuple[float, float, float]
    weight: float
    correlation_before: float
    correlation_after: float
    improvement: float
    fixed_nonzero_fraction: float
    moving_nonzero_fraction: float
    fixed_std: float
    moving_std: float
    accepted: bool
    fixed_content_fraction: float = 0.0
    moving_content_fraction: float = 0.0
    reject_reason: str | None = None
    final_residual_zyx: tuple[float, float, float] | None = None
    fixed_slices: tuple[slice, slice, slice] | None = None
    moving_slices: tuple[slice, slice, slice] | None = None
    source_label: str | None = None


@dataclass(frozen=True)
class ReferenceGeometryConstraint:
    mode: str
    reference_input: str
    fixed_axes: tuple[str, ...]
    shared_geometry_tracks: tuple[str, ...] = ()
    drift_from_reference_um: dict[str, Any] | None = None
    constraint_counts_by_track: dict[str, Any] | None = None
    reference_prior_weights_zyx: tuple[float, float, float] | None = None
    residual_reject_axes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ReferenceGeometrySolverOptions:
    fixed_axes: set[str]
    reference_prior_weights_zyx: tuple[float, float, float] | None
    residual_reject_axes: set[str] | None


@dataclass(frozen=True)
class RobustBoundaryRefinementResult:
    params: list[Any]
    constraints: list[BoundaryConstraint]
    corrections_zyx: list[tuple[float, float, float]]
    anchor_tile: int
    output_dir: Path
    summary: dict[str, Any]
    reference_geometry: ReferenceGeometryConstraint | None = None


@dataclass(frozen=True)
class TrackRunConfig:
    track: TrackMetadata
    output: Path
    registration_output: Path
    registration_input: Path | None
    registration_plots_dir: Path
    robust_boundary_qc_dir: Path
    selected_channels: tuple[int, ...] | None


@dataclass(frozen=True)
class PreStitchTileRotation:
    matrix_physical_zyx: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    rotation_vector_deg_zyx: tuple[float, float, float] | None = None
    source: str | None = None
    inverted: bool = False
    center_mode: str = "linear_only"


def configure_writable_caches(root: Path) -> None:
    cache_root = root / ".cache"
    os.environ.setdefault("ITKWASM_CACHE_DIR", str(cache_root / "itkwasm"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def track_slug(index: int, track_id: str) -> str:
    del track_id
    return f"track{index}"


def unique_ordered(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def normalized_czyx_shape(axes: str, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if axes == "CZYX":
        return tuple(int(value) for value in shape)
    if axes == "ZYX":
        z_count, height, width = shape
        return (1, int(z_count), int(height), int(width))
    raise ValueError(f"Expected CZYX or ZYX axes, got {axes}")


def tile_channel_count(tile: "TileMetadata") -> int:
    if tile.axes == "CZYX":
        return int(tile.shape[0])
    if tile.axes == "ZYX":
        return 1
    raise ValueError(f"Expected CZYX or ZYX axes, got {tile.axes}")


def parse_track_metadata(
    ome_xml: str,
    *,
    channel_count: int,
    channel_labels: tuple[str, ...],
) -> tuple[TrackMetadata, ...]:
    root = ET.fromstring(ome_xml)

    zeiss_channel_ids: list[str] = []
    zeiss_channel_names: dict[str, str] = {}
    for elem in root.iter():
        if local_name(elem.tag) != "Channel":
            continue
        zeiss_id = elem.attrib.get("Id")
        if zeiss_id is None:
            continue
        zeiss_channel_ids.append(zeiss_id)
        zeiss_channel_names.setdefault(zeiss_id, elem.attrib.get("Name", zeiss_id))

    zeiss_channel_ids = unique_ordered(zeiss_channel_ids)
    if len(zeiss_channel_ids) < channel_count:
        return (
            TrackMetadata(
                slug="track0",
                track_id="all",
                channels=tuple(range(channel_count)),
                channel_names=channel_labels,
            ),
        )

    channel_index_by_zeiss_id = {
        zeiss_id: index for index, zeiss_id in enumerate(zeiss_channel_ids[:channel_count])
    }
    tracks: list[TrackMetadata] = []
    assigned_channels: set[int] = set()
    for index, track_elem in enumerate(elem for elem in root.iter() if local_name(elem.tag) == "Track"):
        track_id = track_elem.attrib.get("Id") or f"Track:{index}"
        channels = []
        channel_names = []
        for child in track_elem.iter():
            if local_name(child.tag) != "ChannelRef":
                continue
            ref_id = child.attrib.get("Id")
            if ref_id not in channel_index_by_zeiss_id:
                continue
            channel = channel_index_by_zeiss_id[ref_id]
            channels.append(channel)
            channel_names.append(zeiss_channel_names.get(ref_id, channel_labels[channel]))
        channels_tuple = tuple(dict.fromkeys(channels))
        if not channels_tuple:
            continue
        assigned_channels.update(channels_tuple)
        tracks.append(
            TrackMetadata(
                slug=track_slug(index, track_id),
                track_id=track_id,
                channels=channels_tuple,
                channel_names=tuple(channel_names[: len(channels_tuple)]),
            )
        )

    missing_channels = tuple(index for index in range(channel_count) if index not in assigned_channels)
    if missing_channels:
        tracks.append(
            TrackMetadata(
                slug=f"track{len(tracks)}",
                track_id="unassigned",
                channels=missing_channels,
                channel_names=tuple(channel_labels[index] for index in missing_channels),
            )
        )

    if not tracks:
        return (
            TrackMetadata(
                slug="track0",
                track_id="all",
                channels=tuple(range(channel_count)),
                channel_names=channel_labels,
            ),
        )

    return tuple(tracks)


def parse_ome_metadata(path: Path) -> TileMetadata:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        source_axes = str(series.axes)
        source_shape = tuple(int(value) for value in series.shape)
        shape_czyx = normalized_czyx_shape(source_axes, source_shape)
        ome_xml = tif.ome_metadata
        if ome_xml is None:
            raise ValueError(f"{path} does not contain OME metadata")

    pixels_attrs: dict[str, str] | None = None
    first_plane_attrs: dict[str, str] | None = None
    channels: list[str] = []

    for _, elem in ET.iterparse(io.StringIO(ome_xml), events=("start",)):
        name = local_name(elem.tag)
        if name == "Pixels" and pixels_attrs is None:
            pixels_attrs = dict(elem.attrib)
        elif name == "Channel" and pixels_attrs is not None:
            channels.append(elem.attrib.get("Name") or elem.attrib.get("ID") or str(len(channels)))
        elif name == "Plane":
            first_plane_attrs = dict(elem.attrib)
            break

    if pixels_attrs is None:
        raise ValueError(f"{path} OME metadata does not contain a Pixels element")
    if first_plane_attrs is None:
        raise ValueError(f"{path} OME metadata does not contain a Plane element")

    channel_count = shape_czyx[0]
    if len(channels) < channel_count:
        channels.extend(str(index) for index in range(len(channels), channel_count))

    spacing = {
        "z": float(pixels_attrs["PhysicalSizeZ"]),
        "y": float(pixels_attrs["PhysicalSizeY"]),
        "x": float(pixels_attrs["PhysicalSizeX"]),
    }
    translation = {
        "z": float(first_plane_attrs.get("PositionZ", 0.0)),
        "y": float(first_plane_attrs["PositionY"]),
        "x": float(first_plane_attrs["PositionX"]),
    }

    return TileMetadata(
        path=path,
        shape=source_shape,
        axes=source_axes,
        spacing=spacing,
        translation=translation,
        channels=tuple(channels),
        tracks=parse_track_metadata(
            ome_xml,
            channel_count=channel_count,
            channel_labels=tuple(channels),
        ),
    )


def read_tiles_metadata(input_dir: Path) -> list[TileMetadata]:
    tile_paths = sorted(input_dir.glob("*.ome.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No *.ome.tif files found in {input_dir}")

    tiles = [parse_ome_metadata(path) for path in tile_paths]
    validate_tiles_metadata(tiles, require_same_shape=True)
    return tiles


def validate_tiles_metadata(
    tiles: list[TileMetadata],
    *,
    require_same_shape: bool,
    require_same_spacing: bool = True,
) -> None:
    if not tiles:
        raise ValueError("Expected at least one tile")

    axes = {tile.axes for tile in tiles}
    shapes = {tile.shape for tile in tiles}
    spacings = {tuple(tile.spacing.items()) for tile in tiles}
    channels = {tile.channels for tile in tiles}
    tracks = {tile.tracks for tile in tiles}

    if not axes <= {"CZYX", "ZYX"}:
        raise ValueError(f"Expected all tiles to have CZYX or ZYX axes, found {sorted(axes)}")
    if len(axes) != 1:
        raise ValueError(f"Expected all tiles to have the same axes, found {sorted(axes)}")
    if require_same_shape and len(shapes) != 1:
        raise ValueError(f"Expected all tiles to have the same shape, found {sorted(shapes)}")
    if axes == {"CZYX"}:
        c_yx_shapes = {(tile.shape[0], tile.shape[2], tile.shape[3]) for tile in tiles}
    else:
        c_yx_shapes = {(1, tile.shape[1], tile.shape[2]) for tile in tiles}
    if not require_same_shape and len(c_yx_shapes) != 1:
        raise ValueError(
            "Expected position-file tiles to have the same channel/y/x shape; "
            f"found {sorted(c_yx_shapes)}"
        )
    if require_same_spacing and len(spacings) != 1:
        raise ValueError("Expected all tiles to have the same physical spacing")
    if not require_same_spacing:
        yx_spacings = {(tile.spacing["y"], tile.spacing["x"]) for tile in tiles}
        if len(yx_spacings) != 1:
            raise ValueError("Expected position-file tiles to have the same y/x physical spacing")
    if len(channels) != 1:
        raise ValueError("Expected all tiles to have the same channel metadata")
    if len(tracks) != 1:
        raise ValueError("Expected all tiles to have the same track metadata")


def replace_tile_translation(tile: TileMetadata, translation: dict[str, float]) -> TileMetadata:
    return replace(tile, translation={dim: float(translation[dim]) for dim in ("z", "y", "x")})


def replace_tile_stage_transform(
    tile: TileMetadata,
    *,
    translation: dict[str, float],
    stage_scale: dict[str, float] | None,
    source_view: str | None = None,
) -> TileMetadata:
    normalized_stage_scale = None
    if stage_scale is not None:
        normalized_stage_scale = {dim: float(stage_scale[dim]) for dim in ("z", "y", "x")}
        for dim in ("z", "y", "x"):
            if not math.isclose(abs(normalized_stage_scale[dim]), tile.spacing[dim], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"{tile.path} stage scale {dim}={normalized_stage_scale[dim]} must match "
                    f"physical spacing magnitude {tile.spacing[dim]}"
                )
    return replace(
        tile,
        translation={dim: float(translation[dim]) for dim in ("z", "y", "x")},
        stage_scale=normalized_stage_scale,
        source_view=source_view,
    )


def tile_stage_scale(tile: TileMetadata) -> dict[str, float]:
    return tile.spacing if tile.stage_scale is None else tile.stage_scale


def _shape_zyx_from_axes(axes: str, shape: tuple[int, ...]) -> tuple[int, int, int]:
    if axes == "CZYX":
        return (int(shape[1]), int(shape[2]), int(shape[3]))
    if axes == "ZYX":
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    raise ValueError(f"Expected CZYX or ZYX axes, got {axes}")


def tile_shape_zyx(tile: TileMetadata) -> tuple[int, int, int]:
    return _shape_zyx_from_axes(tile.axes, tile.shape)


def tiff_series_level_count(path: Path) -> int:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        return len(tif.series[0].levels)


def fusion_source_level_for_tile(tile: TileMetadata, requested_level: int) -> tuple[int, int]:
    if requested_level < 0:
        raise ValueError("--fusion-level must be non-negative")
    if requested_level == 0:
        return 0, 1

    available_levels = tiff_series_level_count(tile.path)
    return min(requested_level, max(0, available_levels - 1)), available_levels


def registration_source_level_for_tiles(
    tiles: list[TileMetadata],
    reg_res_levels: tuple[int | None, ...],
    registration_binning: dict[str, int] | None,
) -> tuple[int, int, int]:
    if registration_binning is not None:
        return 0, 1, 0
    concrete_levels = [int(level) for level in reg_res_levels if level is not None]
    if not concrete_levels:
        return 0, 1, 0
    requested_level = min(concrete_levels)
    if requested_level <= 0:
        return 0, 1, 0
    available_level_counts = [tiff_series_level_count(tile.path) for tile in tiles]
    available_levels = min(available_level_counts)
    source_level = min(requested_level, max(0, available_levels - 1))
    if len(set(available_level_counts)) > 1:
        log(
            "Registration TIFF source levels differ across tiles; "
            f"using common source level {source_level} from min available {available_levels}"
        )
    return source_level, available_levels, source_level


def effective_registration_level(stage_reg_res_level: int | None, source_level_offset: int) -> int | None:
    if stage_reg_res_level is None:
        return None
    return max(0, int(stage_reg_res_level) - int(source_level_offset))


def fusion_tile_for_source_array(
    tile: TileMetadata,
    source_shape: tuple[int, ...],
    *,
    source_level: int,
) -> TileMetadata:
    if source_level == 0:
        return tile

    full_shape_zyx = tile_shape_zyx(tile)
    source_shape_zyx = _shape_zyx_from_axes(tile.axes, tuple(int(value) for value in source_shape))
    if any(size <= 0 for size in source_shape_zyx):
        raise ValueError(f"{tile.path} source level {source_level} has invalid shape {source_shape}")

    base_stage_scale = tile_stage_scale(tile)
    scaled_stage_scale = {
        dim: base_stage_scale[dim] * full_shape_zyx[index] / source_shape_zyx[index]
        for index, dim in enumerate(("z", "y", "x"))
    }
    scaled_spacing = {dim: abs(scaled_stage_scale[dim]) for dim in ("z", "y", "x")}
    return replace(
        tile,
        shape=tuple(int(value) for value in source_shape),
        spacing=scaled_spacing,
        stage_scale=scaled_stage_scale,
    )


def tile_sim_scale(tile: TileMetadata) -> dict[str, float]:
    stage_scale = tile_stage_scale(tile)
    return {dim: abs(stage_scale[dim]) for dim in ("z", "y", "x")}


def tile_sim_translation(tile: TileMetadata) -> dict[str, float]:
    stage_scale = tile_stage_scale(tile)
    shape_zyx = tile_shape_zyx(tile)
    translation = {}
    for index, dim in enumerate(("z", "y", "x")):
        if stage_scale[dim] < 0:
            translation[dim] = tile.translation[dim] + shape_zyx[index] * stage_scale[dim]
        else:
            translation[dim] = tile.translation[dim]
    return translation


def tile_flip_axes_zyx(tile: TileMetadata) -> tuple[bool, bool, bool]:
    stage_scale = tile_stage_scale(tile)
    return tuple(stage_scale[dim] < 0 for dim in ("z", "y", "x"))


def flip_spatial_array_for_stage_scale(array: Any, tile: TileMetadata, *, has_channel_axis: bool):
    flips = tile_flip_axes_zyx(tile)
    if not any(flips):
        return array

    slices: list[Any] = [slice(None)] * len(array.shape)
    spatial_offset = 1 if has_channel_axis else 0
    for index, should_flip in enumerate(flips):
        if should_flip:
            slices[spatial_offset + index] = slice(None, None, -1)
    try:
        return array[tuple(slices)]
    except Exception as exc:
        if exc.__class__.__name__ != "NegativeStepError":
            raise
        import dask.array as da

        return da.from_zarr(array)[tuple(slices)]


def read_position_input_tiles(position_input: Path) -> list[TileMetadata]:
    payload = json.loads(position_input.read_text())
    if payload.get("units") != "micrometer":
        raise ValueError(f"{position_input} must declare units='micrometer'")
    records = payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{position_input} must contain a non-empty tiles list")

    tiles: list[TileMetadata] = []
    seen_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{position_input} tile record {index} must be an object")
        raw_path = record.get("path")
        if raw_path is None:
            raise ValueError(f"{position_input} tile record {index} is missing path")
        tile_path = Path(raw_path)
        if not tile_path.is_absolute():
            tile_path = position_input.parent / tile_path
        tile_path = tile_path.resolve()
        if tile_path in seen_paths:
            raise ValueError(f"{position_input} lists duplicate tile path {tile_path}")
        seen_paths.add(tile_path)
        if not tile_path.exists():
            raise FileNotFoundError(f"{position_input} references missing tile {tile_path}")

        translation = record.get("translation_um")
        if not isinstance(translation, dict) or any(dim not in translation for dim in ("z", "y", "x")):
            raise ValueError(f"{position_input} tile record {index} must contain translation_um z/y/x")
        stage_scale = record.get("scale_um")
        if stage_scale is not None and (
            not isinstance(stage_scale, dict) or any(dim not in stage_scale for dim in ("z", "y", "x"))
        ):
            raise ValueError(f"{position_input} tile record {index} scale_um must contain z/y/x")
        source_view = record.get("side")
        if source_view is not None and not isinstance(source_view, str):
            raise ValueError(f"{position_input} tile record {index} side must be a string")
        tiles.append(
            replace_tile_stage_transform(
                parse_ome_metadata(tile_path),
                translation=translation,
                stage_scale=stage_scale,
                source_view=source_view,
            )
        )

    validate_tiles_metadata(tiles, require_same_shape=False, require_same_spacing=False)
    return tiles


def _resolve_registration_tile_path(registration_input: Path, input_dir: Path, record: dict[str, Any]) -> Path:
    raw_path = record.get("path") or record.get("tile")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{registration_input} registration tile record is missing tile/path")

    tile_path = Path(raw_path)
    if tile_path.is_absolute():
        candidates = [tile_path]
    else:
        candidates = [input_dir / tile_path]
        if len(tile_path.parts) == 1:
            candidates.extend(input_dir.parent.glob(f"*/{tile_path.name}"))

    existing = [candidate.resolve() for candidate in candidates if candidate.exists()]
    unique_existing = list(dict.fromkeys(existing))
    if len(unique_existing) == 1:
        return unique_existing[0]
    if not unique_existing:
        raise FileNotFoundError(
            f"{registration_input} tile record {raw_path!r} was not found under "
            f"{input_dir} or one sibling directory level"
        )
    raise ValueError(f"{registration_input} tile record {raw_path!r} resolved ambiguously: {unique_existing}")


def read_registration_input_tiles(registration_input: Path) -> list[TileMetadata]:
    payload = json.loads(registration_input.read_text())
    raw_input_dir = payload.get("input_dir")
    if not isinstance(raw_input_dir, str) or not raw_input_dir:
        raise ValueError(f"{registration_input} is missing input_dir")
    input_dir = Path(raw_input_dir)
    if not input_dir.is_absolute():
        input_dir = registration_input.parent / input_dir
    input_dir = input_dir.resolve()

    records = payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{registration_input} must contain a non-empty tiles list")

    tiles: list[TileMetadata] = []
    seen_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{registration_input} tile record {index} must be an object")

        tile_path = _resolve_registration_tile_path(registration_input, input_dir, record)
        if tile_path in seen_paths:
            raise ValueError(f"{registration_input} lists duplicate tile path {tile_path}")
        seen_paths.add(tile_path)

        translation = record.get("stage_translation_um")
        if not isinstance(translation, dict) or any(dim not in translation for dim in ("z", "y", "x")):
            raise ValueError(f"{registration_input} tile record {index} must contain stage_translation_um z/y/x")
        stage_scale = record.get("stage_scale_um")
        if not isinstance(stage_scale, dict) or any(dim not in stage_scale for dim in ("z", "y", "x")):
            raise ValueError(f"{registration_input} tile record {index} must contain stage_scale_um z/y/x")
        source_view = record.get("source_view")
        if source_view is not None and not isinstance(source_view, str):
            raise ValueError(f"{registration_input} tile record {index} source_view must be a string")

        tiles.append(
            replace_tile_stage_transform(
                parse_ome_metadata(tile_path),
                translation=translation,
                stage_scale=stage_scale,
                source_view=source_view,
            )
        )

    validate_tiles_metadata(tiles, require_same_shape=False, require_same_spacing=False)
    return tiles


def estimate_output_shape(tiles: list[TileMetadata]) -> dict[str, int]:
    first = tiles[0]
    shape = {}
    for dim in ("z", "y", "x"):
        spacing = first.spacing[dim]
        dim_index = {"z": 0, "y": 1, "x": 2}[dim]
        starts = []
        stops = []
        for tile in tiles:
            scale = tile_stage_scale(tile)[dim]
            shape_zyx = tile_shape_zyx(tile)
            edge_a = tile.translation[dim]
            edge_b = tile.translation[dim] + shape_zyx[dim_index] * scale
            starts.append(min(edge_a, edge_b))
            stops.append(max(edge_a, edge_b))
        shape[dim] = int(round((max(stops) - min(starts)) / spacing))
    return shape


def axis_aligned_registration_pairs(tiles: list[TileMetadata]) -> list[tuple[int, int]]:
    bounds = [tile_registered_bounds_zyx(tile) for tile in tiles]
    first = tiles[0]
    axis_tolerance = 0.1 * min(
        tile_shape_zyx(first)[1] * first.spacing["y"],
        tile_shape_zyx(first)[2] * first.spacing["x"],
    )
    pairs: list[tuple[int, int]] = []

    for left_index, left in enumerate(tiles):
        for right_index in range(left_index + 1, len(tiles)):
            right = tiles[right_index]
            left_bounds = bounds[left_index]
            right_bounds = bounds[right_index]
            overlap = {}
            for dim_index, dim in ((1, "y"), (2, "x")):
                overlap[dim] = min(left_bounds.stop_zyx[dim_index], right_bounds.stop_zyx[dim_index]) - max(
                    left_bounds.start_zyx[dim_index],
                    right_bounds.start_zyx[dim_index],
                )

            if overlap["y"] <= 0 or overlap["x"] <= 0:
                continue

            left_center_y = (left_bounds.start_zyx[1] + left_bounds.stop_zyx[1]) / 2.0
            right_center_y = (right_bounds.start_zyx[1] + right_bounds.stop_zyx[1]) / 2.0
            left_center_x = (left_bounds.start_zyx[2] + left_bounds.stop_zyx[2]) / 2.0
            right_center_x = (right_bounds.start_zyx[2] + right_bounds.stop_zyx[2]) / 2.0
            dy = abs((left_center_y - right_center_y) * left.spacing["y"])
            dx = abs((left_center_x - right_center_x) * left.spacing["x"])
            cross_view_pair = (
                left.source_view is not None
                and right.source_view is not None
                and left.source_view != right.source_view
            )
            if cross_view_pair or min(dy, dx) <= axis_tolerance:
                pairs.append((left_index, right_index))

    if not pairs:
        raise ValueError("Could not infer any axis-aligned overlapping registration tile pairs")
    return pairs


def spanning_tree_registration_pairs(tiles: list[TileMetadata]) -> list[tuple[int, int]]:
    pairs = axis_aligned_registration_pairs(tiles)
    neighbors: dict[int, list[int]] = {index: [] for index in range(len(tiles))}
    for left, right in pairs:
        neighbors[left].append(right)
        neighbors[right].append(left)

    tree: list[tuple[int, int]] = []
    seen = {0}
    queue = [0]
    while queue:
        current = queue.pop(0)
        for neighbor in sorted(neighbors[current]):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
            tree.append(tuple(sorted((current, neighbor))))

    if len(seen) != len(tiles):
        missing = sorted(set(range(len(tiles))) - seen)
        raise ValueError(f"Axis-aligned registration graph is disconnected; missing tiles: {missing}")
    return tree


def require_cuda_for_robust_boundary() -> None:
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("robust-boundary registration requires CuPy and CUDA") from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError("robust-boundary registration requires a CUDA-capable GPU") from exc
    if device_count < 1:
        raise RuntimeError("robust-boundary registration requires a CUDA-capable GPU")


def affine_translation_zyx(param: Any) -> tuple[float, float, float]:
    import numpy as np

    data = np.asarray(param.data if hasattr(param, "data") else param, dtype=float)
    while data.ndim > 2:
        data = data[0]
    if data.shape[0] < 3 or data.shape[1] < 4:
        raise ValueError(f"Expected affine matrix with at least 3 spatial rows and 4 columns, got {data.shape}")
    return (float(data[0, 3]), float(data[1, 3]), float(data[2, 3]))


def tile_registered_bounds_zyx(tile: TileMetadata, param: Any | None = None) -> TileBounds:
    shape_zyx = tile_shape_zyx(tile)
    affine_um = affine_translation_zyx(param) if param is not None else (0.0, 0.0, 0.0)
    starts = []
    stops = []
    stage_scale = tile_stage_scale(tile)
    for index, dim in enumerate(("z", "y", "x")):
        edge_a = (tile.translation[dim] + affine_um[index]) / tile.spacing[dim]
        edge_b = edge_a + shape_zyx[index] * stage_scale[dim] / tile.spacing[dim]
        starts.append(min(edge_a, edge_b))
        stops.append(max(edge_a, edge_b))
    return TileBounds(start_zyx=tuple(starts), stop_zyx=tuple(stops))


def boundary_axis_for_pair(bounds_a: TileBounds, bounds_b: TileBounds) -> str:
    centers_a = tuple((bounds_a.start_zyx[index] + bounds_a.stop_zyx[index]) / 2 for index in range(3))
    centers_b = tuple((bounds_b.start_zyx[index] + bounds_b.stop_zyx[index]) / 2 for index in range(3))
    deltas = [abs(centers_a[index] - centers_b[index]) for index in range(3)]
    return ("z", "y", "x")[int(max(range(3), key=lambda index: deltas[index]))]


def clipped_slice(start: int, size: int, limit: int) -> slice:
    start = max(0, min(start, limit))
    stop = max(start, min(start + size, limit))
    return slice(start, stop)


def evenly_spaced_starts(start: int, stop: int, size: int, count: int) -> list[int]:
    if stop <= start:
        return []
    span = stop - start
    if span <= size or count <= 1:
        return [start]
    last = stop - size
    return [int(round(start + (last - start) * index / (count - 1))) for index in range(count)]


def local_slices_for_world_patch(
    tile: TileMetadata,
    bounds: TileBounds,
    world_start_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    limits = tile_shape_zyx(tile)
    starts = [
        int(round(world_start_zyx[index] - bounds.start_zyx[index]))
        for index in range(3)
    ]
    return tuple(
        clipped_slice(starts[index], patch_shape_zyx[index], limits[index])
        for index in range(3)
    )


def slice_shape(slices: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    return tuple(int(slc.stop or 0) - int(slc.start or 0) for slc in slices)


def sample_boundary_patches(
    tiles: list[TileMetadata],
    params: list[Any] | None,
    pairs: list[tuple[int, int]],
    settings: RobustBoundarySettings,
) -> list[BoundaryPatchSpec]:
    bounds = [
        tile_registered_bounds_zyx(tile, None if params is None else params[index])
        for index, tile in enumerate(tiles)
    ]
    patch_specs: list[BoundaryPatchSpec] = []

    for fixed, moving in pairs:
        fixed_bounds = bounds[fixed]
        moving_bounds = bounds[moving]
        overlap_start = tuple(
            int(math.ceil(max(fixed_bounds.start_zyx[index], moving_bounds.start_zyx[index])))
            for index in range(3)
        )
        overlap_stop = tuple(
            int(math.floor(min(fixed_bounds.stop_zyx[index], moving_bounds.stop_zyx[index])))
            for index in range(3)
        )
        overlap_shape = tuple(overlap_stop[index] - overlap_start[index] for index in range(3))
        if any(size <= 0 for size in overlap_shape):
            continue

        patch_shape = tuple(
            min(settings.patch_shape_zyx[index], overlap_shape[index])
            for index in range(3)
        )
        axis = boundary_axis_for_pair(fixed_bounds, moving_bounds)
        varying_axes = [index for index, dim in enumerate(("z", "y", "x")) if dim != axis]
        counts = {index: 1 for index in range(3)}
        if settings.max_patches_per_edge > 1:
            first_axis = varying_axes[0]
            second_axis = varying_axes[1]
            counts[first_axis] = max(1, int(math.floor(math.sqrt(settings.max_patches_per_edge))))
            counts[second_axis] = max(1, settings.max_patches_per_edge // counts[first_axis])

        starts_by_axis = []
        for index in range(3):
            starts_by_axis.append(
                evenly_spaced_starts(
                    overlap_start[index],
                    overlap_stop[index],
                    patch_shape[index],
                    counts[index],
                )
            )

        import itertools

        for patch_index, patch_start in enumerate(itertools.product(*starts_by_axis)):
            fixed_slices = local_slices_for_world_patch(
                tiles[fixed],
                fixed_bounds,
                tuple(patch_start),
                patch_shape,
            )
            moving_slices = local_slices_for_world_patch(
                tiles[moving],
                moving_bounds,
                tuple(patch_start),
                patch_shape,
            )
            shared_shape = tuple(
                min(slice_shape(fixed_slices)[index], slice_shape(moving_slices)[index])
                for index in range(3)
            )
            if any(size <= 1 for size in shared_shape):
                continue
            fixed_slices = tuple(
                slice(fixed_slices[index].start, fixed_slices[index].start + shared_shape[index])
                for index in range(3)
            )
            moving_slices = tuple(
                slice(moving_slices[index].start, moving_slices[index].start + shared_shape[index])
                for index in range(3)
            )
            patch_specs.append(
                BoundaryPatchSpec(
                    pair=(fixed, moving),
                    axis=axis,
                    patch_index=patch_index,
                    fixed_slices=fixed_slices,
                    moving_slices=moving_slices,
                    overlap_start_zyx=tuple(int(value) for value in patch_start),
                    overlap_shape_zyx=shared_shape,
                )
            )

    return patch_specs


def phase_correlation_shift_gpu(fixed: Any, moving: Any) -> tuple[tuple[float, float, float], float]:
    import numpy as np
    import cupy as cp

    fixed_gpu = cp.asarray(np.asarray(fixed, dtype=np.float32))
    moving_gpu = cp.asarray(np.asarray(moving, dtype=np.float32))
    return phase_correlation_shift_gpu_arrays(fixed_gpu, moving_gpu)


def quantize_phase_shift(
    shift: float,
    *,
    upsample_factor: int = PHASE_CORRELATION_UPSAMPLE_FACTOR,
) -> float:
    if upsample_factor <= 1:
        quantized = float(round(shift))
    else:
        quantized = round(float(shift) * upsample_factor) / float(upsample_factor)
    return 0.0 if quantized == 0.0 else quantized


def quadratic_subpixel_peak_offset(left: float, center: float, right: float) -> float:
    if not all(math.isfinite(value) for value in (left, center, right)):
        return 0.0
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    if not math.isfinite(offset):
        return 0.0
    return min(0.5, max(-0.5, float(offset)))


def refined_phase_shift_from_samples(
    peak_index: int,
    size: int,
    left: float,
    center: float,
    right: float,
    *,
    upsample_factor: int = PHASE_CORRELATION_UPSAMPLE_FACTOR,
) -> float:
    refined = float(peak_index) + quadratic_subpixel_peak_offset(left, center, right)
    if refined > float(size) / 2.0:
        refined -= float(size)
    return quantize_phase_shift(refined, upsample_factor=upsample_factor)


def upsampled_dft_gpu(
    data: Any,
    upsampled_region_size: int,
    upsample_factor: int,
    axis_offsets: list[float],
) -> Any:
    import cupy as cp

    if len(axis_offsets) != data.ndim:
        raise ValueError("axis_offsets must match data dimensionality")
    float_dtype = data.real.dtype
    im2pi = 1j * 2.0 * math.pi
    for n_items, axis_offset in zip(data.shape[::-1], axis_offsets[::-1], strict=True):
        sample_positions = cp.arange(upsampled_region_size, dtype=float_dtype) - float(axis_offset)
        frequencies = cp.fft.fftfreq(n_items, d=float(upsample_factor)).astype(float_dtype, copy=False)
        kernel = cp.exp((-im2pi * sample_positions[:, None]) * frequencies[None, :])
        kernel = kernel.astype(data.dtype, copy=False)
        data = cp.tensordot(kernel, data, axes=(1, -1))
    return data


def refined_phase_shifts_gpu(
    image_product: Any,
    peak_index: tuple[int, ...],
    *,
    upsample_factor: int,
) -> tuple[float, ...]:
    import cupy as cp

    shape = tuple(int(value) for value in image_product.shape)
    shifts = []
    for index, size in zip(peak_index, shape, strict=True):
        shift = float(index)
        if index > size // 2:
            shift -= float(size)
        shifts.append(shift)

    if upsample_factor <= 1:
        return tuple(quantize_phase_shift(shift, upsample_factor=1) for shift in shifts)

    shifts = [
        quantize_phase_shift(shift, upsample_factor=upsample_factor)
        for shift in shifts
    ]
    upsampled_region_size = int(math.ceil(float(upsample_factor) * 1.5))
    dftshift = int(math.trunc(upsampled_region_size / 2.0))
    sample_region_offset = [
        float(dftshift) - shift * float(upsample_factor)
        for shift in shifts
    ]
    refined = upsampled_dft_gpu(
        image_product.conj(),
        upsampled_region_size,
        upsample_factor,
        sample_region_offset,
    ).conj()
    refined_peak = tuple(
        int(value)
        for value in cp.unravel_index(cp.argmax(cp.abs(refined)), refined.shape)
    )
    output = []
    for shift, maximum, size in zip(shifts, refined_peak, shape, strict=True):
        refined_shift = shift + (float(maximum) - float(dftshift)) / float(upsample_factor)
        if size == 1:
            refined_shift = 0.0
        output.append(quantize_phase_shift(refined_shift, upsample_factor=upsample_factor))
    return tuple(output)


def content_mask_gpu_array(values_gpu: Any, settings: RobustBoundarySettings) -> Any:
    import cupy as cp

    finite_mask = cp.isfinite(values_gpu)
    positive = values_gpu[finite_mask & (values_gpu > 0)]
    if int(positive.size) < settings.min_content_voxels:
        return cp.zeros(values_gpu.shape, dtype=cp.bool_)
    threshold = cp.percentile(positive, settings.content_mask_percentile)
    return finite_mask & (values_gpu > threshold)


def mask_fraction_gpu_array(mask_gpu: Any) -> float:
    import cupy as cp

    if mask_gpu.size == 0:
        return 0.0
    return float((cp.count_nonzero(mask_gpu) / mask_gpu.size).get())


def masked_centered_array(values_gpu: Any, mask_gpu: Any, min_voxels: int) -> Any:
    import cupy as cp

    centered = cp.zeros_like(values_gpu, dtype=cp.float32)
    if int(cp.count_nonzero(mask_gpu).get()) < min_voxels:
        return centered
    masked_values = values_gpu[mask_gpu]
    centered[mask_gpu] = masked_values - cp.mean(masked_values)
    return centered


def phase_correlation_shift_gpu_arrays(
    fixed_gpu: Any,
    moving_gpu: Any,
    fixed_mask_gpu: Any | None = None,
    moving_mask_gpu: Any | None = None,
    *,
    min_mask_voxels: int = 2,
    upsample_factor: int = PHASE_CORRELATION_UPSAMPLE_FACTOR,
) -> tuple[tuple[float, float, float], float]:
    import cupy as cp

    if fixed_mask_gpu is not None and moving_mask_gpu is not None:
        fixed_gpu = masked_centered_array(fixed_gpu, fixed_mask_gpu, min_mask_voxels)
        moving_gpu = masked_centered_array(moving_gpu, moving_mask_gpu, min_mask_voxels)
    else:
        fixed_gpu = fixed_gpu - cp.mean(fixed_gpu)
        moving_gpu = moving_gpu - cp.mean(moving_gpu)
    cross_power = cp.fft.fftn(fixed_gpu) * cp.conj(cp.fft.fftn(moving_gpu))
    magnitude = cp.abs(cross_power)
    cross_power = cross_power / cp.maximum(magnitude, cp.finfo(cp.float32).eps)
    corr = cp.real(cp.fft.ifftn(cross_power))
    peak_index = tuple(int(value) for value in cp.unravel_index(cp.argmax(corr), corr.shape))
    peak = float(corr[peak_index].get())
    shifts = refined_phase_shifts_gpu(
        cross_power,
        peak_index,
        upsample_factor=upsample_factor,
    )
    return (shifts[0], shifts[1], shifts[2]), peak


def cupy_pairwise_phase_correlation_registration(
    fixed_data: Any,
    moving_data: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Multiview-stitcher pairwise registration function backed by CuPy FFTs."""
    import cupy as cp
    from multiview_stitcher import param_utils

    fixed = cp.asarray(np.asarray(fixed_data.data, dtype=np.float32))
    moving = cp.asarray(np.asarray(moving_data.data, dtype=np.float32))
    finite = cp.isfinite(fixed) & cp.isfinite(moving)
    if int(cp.count_nonzero(finite).get()) < 8:
        return {
            "affine_matrix": param_utils.affine_from_translation([0.0] * fixed.ndim),
            "quality": float("nan"),
        }

    fixed = cp.where(cp.isfinite(fixed), fixed, 0.0)
    moving = cp.where(cp.isfinite(moving), moving, 0.0)
    shift, peak = phase_correlation_shift_gpu_arrays(fixed, moving)
    return {
        "affine_matrix": param_utils.affine_from_translation(list(shift)),
        "quality": float(peak),
    }


def pairwise_registration_function(*, use_gpu: bool) -> Any:
    from multiview_stitcher import registration

    return cupy_pairwise_phase_correlation_registration if use_gpu else registration.phase_correlation_registration


def integer_shift_slices(
    shape: tuple[int, ...],
    shift_zyx: tuple[float, float, float],
) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    source_slices = []
    destination_slices = []
    for size, shift in zip(shape, shift_zyx, strict=True):
        rounded = int(round(shift))
        if abs(rounded) >= size:
            return None
        if rounded > 0:
            source_slices.append(slice(0, size - rounded))
            destination_slices.append(slice(rounded, size))
        elif rounded < 0:
            source_slices.append(slice(-rounded, size))
            destination_slices.append(slice(0, size + rounded))
        else:
            source_slices.append(slice(None))
            destination_slices.append(slice(None))
    return tuple(source_slices), tuple(destination_slices)


def is_integer_shift(shift_zyx: tuple[float, float, float]) -> bool:
    return all(abs(float(shift) - round(float(shift))) < 1e-6 for shift in shift_zyx)


def shift_array_gpu_array(source: Any, shift_zyx: tuple[float, float, float]) -> Any:
    import cupy as cp

    if not is_integer_shift(shift_zyx):
        import cupyx.scipy.ndimage as cpx_ndimage

        is_bool = source.dtype == cp.bool_
        return cpx_ndimage.shift(
            source,
            shift=shift_zyx,
            order=0 if is_bool else 1,
            mode="constant",
            cval=0 if is_bool else 0.0,
            prefilter=False,
        )

    shifted = cp.zeros_like(source)
    slices = integer_shift_slices(tuple(source.shape), shift_zyx)
    if slices is None:
        return shifted
    source_slices, destination_slices = slices
    shifted[destination_slices] = source[source_slices]
    return shifted


def shift_array_cpu(array: Any, shift_zyx: tuple[float, float, float]) -> Any:
    import numpy as np

    source = np.asarray(array)
    if not is_integer_shift(shift_zyx):
        import scipy.ndimage as scipy_ndimage

        is_bool = source.dtype == np.bool_
        return scipy_ndimage.shift(
            source,
            shift=shift_zyx,
            order=0 if is_bool else 1,
            mode="constant",
            cval=0 if is_bool else 0.0,
            prefilter=False,
        )

    shifted = np.zeros_like(source)
    slices = integer_shift_slices(tuple(source.shape), shift_zyx)
    if slices is None:
        return shifted
    source_slices, destination_slices = slices
    shifted[destination_slices] = source[source_slices]
    return shifted


def patch_support_stats_gpu(values_gpu: Any) -> tuple[float, float]:
    import cupy as cp

    finite_mask = cp.isfinite(values_gpu)
    finite_count = int(cp.count_nonzero(finite_mask).get())
    if finite_count == 0:
        return 0.0, 0.0
    finite = values_gpu[finite_mask]
    nonzero_fraction = float((cp.count_nonzero(finite) / finite_count).get())
    return nonzero_fraction, float(cp.std(finite).get())


def normalized_cross_correlation_gpu_arrays(
    fixed_gpu: Any,
    moving_gpu: Any,
    mask_gpu: Any | None = None,
    *,
    min_voxels: int = 2,
) -> float:
    import cupy as cp

    finite_mask = cp.isfinite(fixed_gpu) & cp.isfinite(moving_gpu)
    if mask_gpu is not None:
        finite_mask &= mask_gpu
    if int(cp.count_nonzero(finite_mask).get()) < min_voxels:
        return float("nan")
    fixed_values = fixed_gpu[finite_mask]
    moving_values = moving_gpu[finite_mask]
    fixed_values = fixed_values - cp.mean(fixed_values)
    moving_values = moving_values - cp.mean(moving_values)
    denominator = cp.sqrt(
        cp.sum(fixed_values * fixed_values) * cp.sum(moving_values * moving_values)
    )
    if float(denominator.get()) == 0.0:
        return float("nan")
    return float((cp.sum(fixed_values * moving_values) / denominator).get())


def evaluate_boundary_patch_gpu(
    fixed_patch: Any,
    moving_patch: Any,
    settings: RobustBoundarySettings,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
    tuple[float, float, float],
    float,
    float,
]:
    import numpy as np
    import cupy as cp

    fixed_gpu = cp.asarray(np.asarray(fixed_patch, dtype=np.float32))
    moving_gpu = cp.asarray(np.asarray(moving_patch, dtype=np.float32))
    fixed_support = patch_support_stats_gpu(fixed_gpu)
    moving_support = patch_support_stats_gpu(moving_gpu)
    fixed_content_mask = content_mask_gpu_array(fixed_gpu, settings)
    moving_content_mask = content_mask_gpu_array(moving_gpu, settings)
    fixed_content = mask_fraction_gpu_array(fixed_content_mask)
    moving_content = mask_fraction_gpu_array(moving_content_mask)
    content_overlap_mask = fixed_content_mask & moving_content_mask
    corr_before = normalized_cross_correlation_gpu_arrays(
        fixed_gpu,
        moving_gpu,
        content_overlap_mask,
        min_voxels=settings.min_content_voxels,
    )
    if (
        fixed_support[0] < settings.min_nonzero_fraction
        or moving_support[0] < settings.min_nonzero_fraction
        or fixed_support[1] < settings.min_std
        or moving_support[1] < settings.min_std
        or fixed_content < settings.min_content_fraction
        or moving_content < settings.min_content_fraction
    ):
        return fixed_support, moving_support, (fixed_content, moving_content), corr_before, (0.0, 0.0, 0.0), float("nan"), corr_before
    shift, peak = phase_correlation_shift_gpu_arrays(
        fixed_gpu,
        moving_gpu,
        fixed_content_mask,
        moving_content_mask,
        min_mask_voxels=settings.min_content_voxels,
    )
    shifted = shift_array_gpu_array(moving_gpu, shift)
    shifted_moving_content_mask = shift_array_gpu_array(moving_content_mask, shift)
    shifted_content_overlap_mask = fixed_content_mask & shifted_moving_content_mask
    corr_after = normalized_cross_correlation_gpu_arrays(
        fixed_gpu,
        shifted,
        shifted_content_overlap_mask,
        min_voxels=settings.min_content_voxels,
    )
    return fixed_support, moving_support, (fixed_content, moving_content), corr_before, shift, peak, corr_after


def read_native_patch(array: Any, axes: str, channel: int, slices_zyx: tuple[slice, slice, slice]) -> Any:
    import numpy as np

    if axes == "CZYX":
        return np.asarray(array[(channel, *slices_zyx)])
    if axes == "ZYX":
        if channel != 0:
            raise ValueError(f"Channel {channel} out of range for single-channel ZYX tile")
        return np.asarray(array[slices_zyx])
    raise ValueError(f"Expected CZYX or ZYX axes, got {axes}")


def is_stable_aligned_patch(
    corr_before: float,
    shift_zyx: tuple[float, float, float],
    settings: RobustBoundarySettings,
) -> bool:
    return (
        math.isfinite(corr_before)
        and corr_before >= settings.min_stable_correlation
        and all(
            abs(shift_zyx[index]) <= settings.max_stable_shift_zyx[index]
            for index in range(3)
        )
    )


def accepted_boundary_weight(
    *,
    improvement: float,
    fixed_content: float,
    moving_content: float,
    peak: float,
    stable_alignment: bool,
    settings: RobustBoundarySettings,
) -> float:
    support = min(fixed_content, moving_content)
    peak_weight = max(0.0, peak) if math.isfinite(peak) else 1.0
    if stable_alignment:
        evidence = settings.min_improvement
    else:
        evidence = max(0.0, improvement)
    return evidence * support * peak_weight


def boundary_constraint_from_evaluation(
    *,
    spec: BoundaryPatchSpec,
    fixed_index: int,
    moving_index: int,
    fixed_slices: tuple[slice, slice, slice],
    moving_slices: tuple[slice, slice, slice],
    fixed_support: tuple[float, float],
    moving_support: tuple[float, float],
    fixed_content: float,
    moving_content: float,
    corr_before: float,
    shift: tuple[float, float, float],
    peak: float,
    corr_after: float,
    settings: RobustBoundarySettings,
    source_label: str | None = None,
) -> BoundaryConstraint:
    fixed_nonzero, fixed_std = fixed_support
    moving_nonzero, moving_std = moving_support
    reject_reason = None
    if fixed_nonzero < settings.min_nonzero_fraction or moving_nonzero < settings.min_nonzero_fraction:
        reject_reason = "low_nonzero_fraction"
    elif fixed_std < settings.min_std or moving_std < settings.min_std:
        reject_reason = "low_texture"
    elif fixed_content < settings.min_content_fraction or moving_content < settings.min_content_fraction:
        reject_reason = "low_content"

    stable_alignment = False
    if reject_reason is None:
        improvement = corr_after - corr_before if math.isfinite(corr_before) else corr_after
        stable_alignment = is_stable_aligned_patch(corr_before, shift, settings)
        if stable_alignment and (
            not math.isfinite(improvement)
            or improvement < settings.min_improvement
        ):
            shift = (0.0, 0.0, 0.0)
            corr_after = corr_before
            improvement = 0.0
        elif not math.isfinite(corr_after) or corr_after < settings.min_correlation:
            reject_reason = "weak_correlation"
        elif not math.isfinite(improvement) or improvement < settings.min_improvement:
            reject_reason = "weak_improvement"
    else:
        improvement = 0.0

    if reject_reason is None:
        weight = accepted_boundary_weight(
            improvement=improvement,
            fixed_content=fixed_content,
            moving_content=moving_content,
            peak=peak,
            stable_alignment=stable_alignment,
            settings=settings,
        )
        if weight <= 0:
            reject_reason = "zero_weight"
    else:
        weight = 0.0

    return BoundaryConstraint(
        fixed=fixed_index,
        moving=moving_index,
        pair=spec.pair,
        axis=spec.axis,
        patch_index=spec.patch_index,
        shift_zyx=shift,
        weight=weight,
        correlation_before=corr_before,
        correlation_after=corr_after,
        improvement=improvement,
        fixed_nonzero_fraction=fixed_nonzero,
        moving_nonzero_fraction=moving_nonzero,
        fixed_std=fixed_std,
        moving_std=moving_std,
        accepted=reject_reason is None,
        fixed_content_fraction=fixed_content,
        moving_content_fraction=moving_content,
        reject_reason=reject_reason,
        fixed_slices=fixed_slices,
        moving_slices=moving_slices,
        source_label=source_label,
    )


def evaluate_boundary_patch_constraint(
    *,
    arrays: list[Any],
    tiles: list[TileMetadata],
    channel: int,
    spec: BoundaryPatchSpec,
    settings: RobustBoundarySettings,
    source_label: str | None = None,
) -> BoundaryConstraint:
    fixed_index, moving_index = spec.pair
    fixed_patch = read_native_patch(
        arrays[fixed_index],
        tiles[fixed_index].axes,
        channel,
        spec.fixed_slices,
    )
    moving_patch = read_native_patch(
        arrays[moving_index],
        tiles[moving_index].axes,
        channel,
        spec.moving_slices,
    )
    (
        fixed_support,
        moving_support,
        (fixed_content, moving_content),
        corr_before,
        shift,
        peak,
        corr_after,
    ) = evaluate_boundary_patch_gpu(fixed_patch, moving_patch, settings)
    return boundary_constraint_from_evaluation(
        spec=spec,
        fixed_index=fixed_index,
        moving_index=moving_index,
        fixed_slices=spec.fixed_slices,
        moving_slices=spec.moving_slices,
        fixed_support=fixed_support,
        moving_support=moving_support,
        fixed_content=fixed_content,
        moving_content=moving_content,
        corr_before=corr_before,
        shift=shift,
        peak=peak,
        corr_after=corr_after,
        settings=settings,
        source_label=source_label,
    )


def finite_weighted_average(values: list[float], weights: np.ndarray) -> float:
    finite = np.asarray([math.isfinite(value) for value in values], dtype=bool)
    if not np.any(finite):
        return float("nan")
    selected_values = np.asarray(values, dtype=float)[finite]
    selected_weights = weights[finite]
    if float(np.sum(selected_weights)) <= 0.0:
        return float(np.mean(selected_values))
    return float(np.average(selected_values, weights=selected_weights))


def combine_channel_boundary_constraints(
    constraints: list[BoundaryConstraint],
    *,
    source_label: str,
    settings: RobustBoundarySettings,
) -> BoundaryConstraint:
    if not constraints:
        raise ValueError("Cannot combine an empty boundary constraint list")
    first = constraints[0]
    accepted = [constraint for constraint in constraints if constraint.accepted]
    if not accepted:
        best = max(
            constraints,
            key=lambda constraint: (
                constraint.correlation_after
                if math.isfinite(constraint.correlation_after)
                else float("-inf")
            ),
        )
        return replace(
            best,
            weight=0.0,
            accepted=False,
            reject_reason="all_channels_rejected",
            source_label=source_label,
        )

    weights = np.asarray([max(constraint.weight, 0.0) for constraint in accepted], dtype=float)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(accepted), dtype=float)
    shifts = np.asarray([constraint.shift_zyx for constraint in accepted], dtype=float)
    combined_shift = tuple(
        quantize_phase_shift(float(value))
        for value in np.average(shifts, axis=0, weights=weights)
    )
    disagreement = np.max(np.abs(shifts - np.asarray(combined_shift, dtype=float)), axis=0)
    residual_scale = np.asarray(settings.max_final_residual_zyx, dtype=float)
    normalized_disagreement = float(np.max(disagreement / residual_scale))
    consensus_weight = 1.0 / max(1.0, normalized_disagreement)
    combined_weight = float(np.sum([constraint.weight for constraint in accepted]) / len(constraints))
    combined_weight *= consensus_weight

    metric_weights = weights
    corr_before = finite_weighted_average(
        [constraint.correlation_before for constraint in accepted],
        metric_weights,
    )
    corr_after = finite_weighted_average(
        [constraint.correlation_after for constraint in accepted],
        metric_weights,
    )
    improvement = finite_weighted_average(
        [constraint.improvement for constraint in accepted],
        metric_weights,
    )
    return BoundaryConstraint(
        fixed=first.fixed,
        moving=first.moving,
        pair=first.pair,
        axis=first.axis,
        patch_index=first.patch_index,
        shift_zyx=combined_shift,
        weight=combined_weight,
        correlation_before=corr_before,
        correlation_after=corr_after,
        improvement=improvement,
        fixed_nonzero_fraction=finite_weighted_average(
            [constraint.fixed_nonzero_fraction for constraint in accepted],
            metric_weights,
        ),
        moving_nonzero_fraction=finite_weighted_average(
            [constraint.moving_nonzero_fraction for constraint in accepted],
            metric_weights,
        ),
        fixed_std=finite_weighted_average(
            [constraint.fixed_std for constraint in accepted],
            metric_weights,
        ),
        moving_std=finite_weighted_average(
            [constraint.moving_std for constraint in accepted],
            metric_weights,
        ),
        accepted=combined_weight > 0.0,
        fixed_content_fraction=finite_weighted_average(
            [constraint.fixed_content_fraction for constraint in accepted],
            metric_weights,
        ),
        moving_content_fraction=finite_weighted_average(
            [constraint.moving_content_fraction for constraint in accepted],
            metric_weights,
        ),
        reject_reason=None if combined_weight > 0.0 else "zero_weight",
        fixed_slices=first.fixed_slices,
        moving_slices=first.moving_slices,
        source_label=source_label,
    )


def build_boundary_constraints(
    tiles: list[TileMetadata],
    channel: int,
    patch_specs: list[BoundaryPatchSpec],
    settings: RobustBoundarySettings,
) -> list[BoundaryConstraint]:
    arrays = []
    stores = []
    constraints: list[BoundaryConstraint] = []
    try:
        for tile in tiles:
            array, store = open_tile_array(tile)
            arrays.append(array)
            stores.append(store)

        for spec_index, spec in enumerate(patch_specs, start=1):
            constraint = evaluate_boundary_patch_constraint(
                arrays=arrays,
                tiles=tiles,
                channel=channel,
                spec=spec,
                settings=settings,
            )
            constraints.append(constraint)
            if spec_index <= 3 or spec_index % 25 == 0 or spec_index == len(patch_specs):
                log(
                    "Boundary patch "
                    f"{spec_index}/{len(patch_specs)} pair={spec.pair} axis={spec.axis} "
                    f"shift_zyx={tuple(round(value, 3) for value in constraint.shift_zyx)} "
                    f"corr={constraint.correlation_before:.3f}->{constraint.correlation_after:.3f} "
                    f"{'accepted' if constraint.accepted else 'rejected=' + str(constraint.reject_reason)}"
                )
    finally:
        close_stores(stores)
    return constraints


def build_combined_boundary_constraints(
    tiles: list[TileMetadata],
    channels: tuple[int, ...],
    patch_specs: list[BoundaryPatchSpec],
    settings: RobustBoundarySettings,
    *,
    source_label: str,
) -> list[BoundaryConstraint]:
    if not channels:
        raise ValueError("At least one channel is required for combined boundary constraints")
    arrays = []
    stores = []
    constraints: list[BoundaryConstraint] = []
    try:
        for tile in tiles:
            array, store = open_tile_array(tile)
            arrays.append(array)
            stores.append(store)

        for spec_index, spec in enumerate(patch_specs, start=1):
            channel_constraints = [
                evaluate_boundary_patch_constraint(
                    arrays=arrays,
                    tiles=tiles,
                    channel=channel,
                    spec=spec,
                    settings=settings,
                    source_label=f"channel{channel}",
                )
                for channel in channels
            ]
            constraint = combine_channel_boundary_constraints(
                channel_constraints,
                source_label=source_label,
                settings=settings,
            )
            constraints.append(constraint)
            if spec_index <= 3 or spec_index % 25 == 0 or spec_index == len(patch_specs):
                accepted_channels = sum(item.accepted for item in channel_constraints)
                log(
                    "Combined boundary patch "
                    f"{spec_index}/{len(patch_specs)} pair={spec.pair} axis={spec.axis} "
                    f"channels={channels} accepted_channels={accepted_channels}/{len(channels)} "
                    f"shift_zyx={tuple(round(value, 3) for value in constraint.shift_zyx)} "
                    f"corr={constraint.correlation_before:.3f}->{constraint.correlation_after:.3f} "
                    f"{'accepted' if constraint.accepted else 'rejected=' + str(constraint.reject_reason)}"
                )
    finally:
        close_stores(stores)
    return constraints


def choose_anchor_tile(tiles: list[TileMetadata], constraints: list[BoundaryConstraint]) -> int:
    quality = [0.0 for _ in tiles]
    for constraint in constraints:
        if not constraint.accepted:
            continue
        quality[constraint.fixed] += constraint.weight
        quality[constraint.moving] += constraint.weight
    if any(value > 0 for value in quality):
        return int(max(range(len(tiles)), key=lambda index: quality[index]))

    centers = []
    for tile in tiles:
        bounds = tile_registered_bounds_zyx(tile)
        centers.append(tuple((bounds.start_zyx[index] + bounds.stop_zyx[index]) / 2 for index in range(3)))
    mosaic_center = tuple(sum(center[index] for center in centers) / len(centers) for index in range(3))
    distances = [
        sum((center[index] - mosaic_center[index]) ** 2 for index in range(3))
        for center in centers
    ]
    return int(min(range(len(tiles)), key=lambda index: distances[index]))


def anchor_connected_tiles(
    n_tiles: int,
    constraints: list[BoundaryConstraint],
    anchor_tile: int,
) -> set[int]:
    neighbors: dict[int, set[int]] = {index: set() for index in range(n_tiles)}
    for constraint in constraints:
        if not constraint.accepted:
            continue
        neighbors[constraint.fixed].add(constraint.moving)
        neighbors[constraint.moving].add(constraint.fixed)

    connected = {anchor_tile}
    queue = [anchor_tile]
    while queue:
        current = queue.pop(0)
        for neighbor in sorted(neighbors[current]):
            if neighbor in connected:
                continue
            connected.add(neighbor)
            queue.append(neighbor)
    return connected


def solve_tile_corrections_zyx(
    n_tiles: int,
    constraints: list[BoundaryConstraint],
    settings: RobustBoundarySettings,
    anchor_tile: int,
    *,
    fixed_axes: set[str] | None = None,
    reference_prior_weights_zyx: tuple[float, float, float] | None = None,
) -> list[tuple[float, float, float]]:
    import numpy as np
    from scipy.optimize import lsq_linear

    axis_to_index = {"z": 0, "y": 1, "x": 2}
    fixed_dim_indices = {axis_to_index[axis] for axis in (fixed_axes or set())}
    connected_tiles = anchor_connected_tiles(n_tiles, constraints, anchor_tile)
    accepted = [
        constraint
        for constraint in constraints
        if constraint.accepted
        and constraint.fixed in connected_tiles
        and constraint.moving in connected_tiles
    ]
    corrections = np.zeros((n_tiles, 3), dtype=float)
    if not accepted:
        return [tuple(float(value) for value in row) for row in corrections]

    clamp = np.asarray(settings.max_correction_zyx, dtype=float)
    prior_weights = np.asarray(
        reference_prior_weights_zyx or (0.0, 0.0, 0.0),
        dtype=float,
    )
    if np.any(prior_weights < 0.0):
        raise ValueError("Reference prior weights must be non-negative")

    active_dims = [dim for dim in range(3) if dim not in fixed_dim_indices]
    if not active_dims:
        return [tuple(float(value) for value in row) for row in corrections]

    n_constraints = len(accepted)
    fixed_tile_indices = np.fromiter(
        (constraint.fixed for constraint in accepted),
        dtype=np.intp,
        count=n_constraints,
    )
    moving_tile_indices = np.fromiter(
        (constraint.moving for constraint in accepted),
        dtype=np.intp,
        count=n_constraints,
    )
    measured_shifts = np.asarray(
        [constraint.shift_zyx for constraint in accepted],
        dtype=float,
    )
    base_edge_weights = np.asarray(
        [max(constraint.weight, 1e-6) for constraint in accepted],
        dtype=float,
    )
    edge_weights = base_edge_weights.copy()

    # Each dimension has the same edge rows, but dimensions with an absolute
    # prior include every tile as a variable. This avoids giving the selected
    # anchor an unintended infinite-precision prior in y/x.
    dimension_systems: dict[
        int,
        tuple[np.ndarray, np.ndarray, dict[int, int], np.ndarray],
    ] = {}
    design_cache: dict[bool, tuple[np.ndarray, dict[int, int]]] = {}
    sorted_connected_tiles = sorted(connected_tiles)
    for dim in active_dims:
        has_absolute_prior = prior_weights[dim] > 0.0
        cached_design = design_cache.get(has_absolute_prior)
        if cached_design is None:
            variable_tiles = (
                sorted_connected_tiles
                if has_absolute_prior
                else [tile for tile in sorted_connected_tiles if tile != anchor_tile]
            )
            if not variable_tiles:
                continue
            variable_index = {
                tile: index for index, tile in enumerate(variable_tiles)
            }
            edge_a = np.zeros((n_constraints, len(variable_tiles)), dtype=float)
            for row_index, constraint in enumerate(accepted):
                moving_index = variable_index.get(constraint.moving)
                fixed_index = variable_index.get(constraint.fixed)
                if moving_index is not None:
                    edge_a[row_index, moving_index] += 1.0
                if fixed_index is not None:
                    edge_a[row_index, fixed_index] -= 1.0
            a = (
                np.vstack((edge_a, np.eye(len(variable_tiles), dtype=float)))
                if has_absolute_prior
                else edge_a
            )
            design_cache[has_absolute_prior] = (a, variable_index)
        else:
            a, variable_index = cached_design

        edge_b = measured_shifts[:, dim]
        if has_absolute_prior:
            b = np.concatenate(
                (edge_b, np.zeros(len(variable_index), dtype=float))
            )
            prior_row_weights = np.full(
                len(variable_index),
                prior_weights[dim],
                dtype=float,
            )
        else:
            b = edge_b
            prior_row_weights = np.empty(0, dtype=float)

        dimension_systems[dim] = (a, b, variable_index, prior_row_weights)

    if not dimension_systems:
        return [tuple(float(value) for value in row) for row in corrections]

    # Preserve the existing scalar Huber control while respecting the existing
    # anisotropic residual thresholds. With the defaults this is (4, 8, 8) px.
    max_residual = np.asarray(settings.max_final_residual_zyx, dtype=float)
    if np.any(max_residual <= 0.0):
        raise ValueError("Final residual thresholds must be positive")
    z_scale = max(max_residual[0], np.finfo(float).eps)
    huber_delta_zyx = max_residual * (
        max(float(settings.huber_delta), np.finfo(float).eps) / z_scale
    )
    solved_dims = sorted(dimension_systems)

    for _ in range(max(1, settings.irls_iterations)):
        for dim in solved_dims:
            a, b, variable_index, prior_row_weights = dimension_systems[dim]
            row_weights = (
                np.concatenate((edge_weights, prior_row_weights))
                if prior_row_weights.size
                else edge_weights
            )
            sqrt_w = np.sqrt(row_weights)
            result = lsq_linear(
                a * sqrt_w[:, None],
                b * sqrt_w,
                bounds=(-clamp[dim], clamp[dim]),
                lsmr_tol="auto",
            )
            if not result.success:
                raise RuntimeError(
                    f"Tile correction solve failed for axis {('z', 'y', 'x')[dim]}: "
                    f"{result.message}"
                )
            solution = np.asarray(result.x, dtype=float)
            for tile, index in variable_index.items():
                corrections[tile, dim] = solution[index]
            if prior_weights[dim] <= 0.0:
                corrections[anchor_tile, dim] = 0.0

        # One robust inlier weight per patch vector. A gross mismatch in any
        # solved axis downweights that patch in z/y/x together.
        residuals = (
            corrections[moving_tile_indices]
            - corrections[fixed_tile_indices]
            - measured_shifts
        )
        normalized = np.abs(residuals[:, solved_dims]) / huber_delta_zyx[solved_dims]
        robust_scale = np.maximum(1.0, np.max(normalized, axis=1))
        next_edge_weights = base_edge_weights / robust_scale
        if np.allclose(next_edge_weights, edge_weights, rtol=1e-3, atol=1e-12):
            break
        edge_weights = next_edge_weights

    return [tuple(float(value) for value in row) for row in corrections]


def annotate_final_residuals(
    constraints: list[BoundaryConstraint],
    corrections_zyx: list[tuple[float, float, float]],
    settings: RobustBoundarySettings,
    connected_tiles: set[int] | None = None,
    *,
    reject_axes: set[str] | None = None,
) -> list[BoundaryConstraint]:
    axis_to_index = {"z": 0, "y": 1, "x": 2}
    selected_axes = {"z", "y", "x"} if reject_axes is None else set(reject_axes)
    unknown_axes = selected_axes - axis_to_index.keys()
    if unknown_axes:
        raise ValueError(f"Unsupported residual rejection axes: {sorted(unknown_axes)}")

    max_residual = settings.max_final_residual_zyx
    reject_dim_indices = {axis_to_index[axis] for axis in selected_axes}
    updated = []
    for constraint in constraints:
        residual = tuple(
            corrections_zyx[constraint.moving][index]
            - corrections_zyx[constraint.fixed][index]
            - constraint.shift_zyx[index]
            for index in range(3)
        )
        disconnected = (
            connected_tiles is not None
            and constraint.accepted
            and (
                constraint.fixed not in connected_tiles
                or constraint.moving not in connected_tiles
            )
        )
        high_residual = constraint.accepted and any(
            abs(residual[index]) > max_residual[index]
            for index in reject_dim_indices
        )
        reject = disconnected or high_residual
        if disconnected:
            reject_reason = "disconnected_from_anchor"
        elif high_residual:
            reject_reason = "high_final_residual"
        else:
            reject_reason = constraint.reject_reason
        updated.append(
            replace(
                constraint,
                accepted=constraint.accepted and not reject,
                reject_reason=reject_reason,
                final_residual_zyx=residual,
            )
        )
    return updated


def solve_tile_corrections_with_residual_rejection(
    tiles: list[TileMetadata],
    constraints: list[BoundaryConstraint],
    settings: RobustBoundarySettings,
    *,
    fixed_axes: set[str] | None = None,
    reference_prior_weights_zyx: tuple[float, float, float] | None = None,
    residual_reject_axes: set[str] | None = None,
) -> tuple[list[tuple[float, float, float]], list[BoundaryConstraint], int]:
    n_tiles = len(tiles)
    current = constraints
    anchor_tile = choose_anchor_tile(tiles, current)
    hard_reject_axes = (
        set(residual_reject_axes)
        if residual_reject_axes is not None
        else {"z", "y", "x"} - (fixed_axes or set())
    )

    for _ in range(len(constraints) + n_tiles + 1):
        connected_tiles = anchor_connected_tiles(n_tiles, current, anchor_tile)
        corrections_zyx = solve_tile_corrections_zyx(
            n_tiles,
            current,
            settings,
            anchor_tile,
            fixed_axes=fixed_axes,
            reference_prior_weights_zyx=reference_prior_weights_zyx,
        )
        updated = annotate_final_residuals(
            current,
            corrections_zyx,
            settings,
            connected_tiles,
            reject_axes=hard_reject_axes,
        )
        next_anchor_tile = choose_anchor_tile(tiles, updated)
        if all(
            before.accepted == after.accepted
            for before, after in zip(current, updated, strict=True)
        ) and next_anchor_tile == anchor_tile:
            return corrections_zyx, updated, anchor_tile
        current = updated
        anchor_tile = next_anchor_tile

    raise RuntimeError("Robust boundary residual rejection did not converge")


def apply_corrections_to_params(
    params: list[Any],
    corrections_zyx: list[tuple[float, float, float]],
    spacing: dict[str, float],
) -> list[Any]:
    corrected = []
    for param, correction in zip(params, corrections_zyx, strict=True):
        updated = param.copy(deep=True)
        for index, dim in enumerate(("z", "y", "x")):
            updated.data[(0,) * (updated.data.ndim - 2) + (index, 3)] += correction[index] * spacing[dim]
        corrected.append(updated)
    return corrected


def set_affine_translation_um(
    param: Any,
    translation_zyx_um: tuple[float, float, float],
) -> Any:
    updated = param.copy(deep=True)
    index_prefix = (0,) * (updated.data.ndim - 2)
    for index, value in enumerate(translation_zyx_um):
        updated.data[index_prefix + (index, 3)] = float(value)
    return updated


def apply_reference_fixed_axes(
    params: list[Any],
    reference_params: list[Any],
    fixed_axes: set[str],
) -> list[Any]:
    axis_to_index = {"z": 0, "y": 1, "x": 2}
    fixed_indices = {axis_to_index[axis] for axis in fixed_axes}
    constrained = []
    for param, reference in zip(params, reference_params, strict=True):
        translation = list(affine_translation_zyx(param))
        reference_translation = affine_translation_zyx(reference)
        for index in fixed_indices:
            translation[index] = reference_translation[index]
        constrained.append(set_affine_translation_um(param, tuple(translation)))
    return constrained


def reference_drift_summary_um(
    params: list[Any],
    reference_params: list[Any],
) -> dict[str, dict[str, float]]:
    translations = np.asarray([affine_translation_zyx(param) for param in params], dtype=float)
    references = np.asarray([affine_translation_zyx(param) for param in reference_params], dtype=float)
    drift = translations - references
    summary: dict[str, dict[str, float]] = {}
    for index, dim in enumerate(("z", "y", "x")):
        values = drift[:, index]
        abs_values = np.abs(values)
        summary[dim] = {
            "median": float(np.median(values)),
            "p95_abs": float(np.percentile(abs_values, 95)),
            "max_abs": float(np.max(abs_values)),
        }
    return summary


def reference_geometry_solver_options(
    mode: str,
    reference_xy_prior_weight: float,
) -> ReferenceGeometrySolverOptions:
    if reference_xy_prior_weight < 0.0:
        raise ValueError("--reference-xy-prior-weight must be non-negative")
    if mode not in {"fixed-xy", "full-xyz", "penalized-xy"}:
        raise ValueError(f"Unsupported reference geometry mode: {mode}")
    if mode == "fixed-xy":
        return ReferenceGeometrySolverOptions(
            fixed_axes={"y", "x"},
            reference_prior_weights_zyx=None,
            residual_reject_axes=None,
        )
    if mode == "penalized-xy":
        return ReferenceGeometrySolverOptions(
            fixed_axes=set(),
            reference_prior_weights_zyx=(
                0.0,
                float(reference_xy_prior_weight),
                float(reference_xy_prior_weight),
            ),
            # All solved axes participate in joint Huber IRLS. Disable only the
            # post-fit hard gate because the xy prior intentionally leaves
            # nonzero xy residuals when it resists a measured edge shift.
            residual_reject_axes=set(),
        )
    return ReferenceGeometrySolverOptions(
        fixed_axes=set(),
        reference_prior_weights_zyx=None,
        residual_reject_axes=None,
    )


def constraint_counts_by_source(
    constraints: list[BoundaryConstraint],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for constraint in constraints:
        label = constraint.source_label or "unlabeled"
        item = counts.setdefault(label, {"accepted": 0, "total": 0})
        item["total"] += 1
        if constraint.accepted:
            item["accepted"] += 1
    return counts


def reference_geometry_constraint(
    *,
    mode: str,
    reference_input: Path,
    fixed_axes: set[str],
    params: list[Any],
    reference_params: list[Any],
    constraints: list[BoundaryConstraint],
    shared_geometry_tracks: tuple[str, ...] = (),
    reference_prior_weights_zyx: tuple[float, float, float] | None = None,
    residual_reject_axes: set[str] | None = None,
) -> ReferenceGeometryConstraint:
    return ReferenceGeometryConstraint(
        mode=mode,
        reference_input=str(reference_input),
        fixed_axes=tuple(axis for axis in ("z", "y", "x") if axis in fixed_axes),
        shared_geometry_tracks=shared_geometry_tracks,
        drift_from_reference_um=reference_drift_summary_um(params, reference_params),
        constraint_counts_by_track=constraint_counts_by_source(constraints),
        reference_prior_weights_zyx=reference_prior_weights_zyx,
        residual_reject_axes=(
            tuple(axis for axis in ("z", "y", "x") if axis in residual_reject_axes)
            if residual_reject_axes is not None
            else None
        ),
    )


def write_reference_geometry_qc(
    output_dir: Path,
    reference_geometry: ReferenceGeometryConstraint | None,
) -> None:
    if reference_geometry is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": reference_geometry.mode,
        "reference_registration_input": reference_geometry.reference_input,
        "fixed_axes": list(reference_geometry.fixed_axes),
        "shared_geometry_tracks": list(reference_geometry.shared_geometry_tracks),
        "drift_from_reference_um": reference_geometry.drift_from_reference_um,
        "constraint_counts_by_track": reference_geometry.constraint_counts_by_track,
        "reference_prior_weights_zyx": reference_geometry.reference_prior_weights_zyx,
        "residual_reject_axes": (
            list(reference_geometry.residual_reject_axes)
            if reference_geometry.residual_reject_axes is not None
            else None
        ),
    }
    for filename in ("reference_drift.json", "shared_geometry_constraint_summary.json"):
        (output_dir / filename).write_text(json.dumps(payload, indent=2) + "\n")


def set_msims_affine_transform(msims: list[Any], params: list[Any], transform_key: str) -> None:
    from multiview_stitcher import msi_utils

    for msim, param in zip(msims, params, strict=True):
        msi_utils.set_affine_transform(
            msim,
            xaffine=spatial_affine_param(param),
            transform_key=transform_key,
        )


def set_msims_relative_affine_transform(
    msims: list[Any],
    params: list[Any],
    *,
    transform_key: str,
    base_transform_key: str,
) -> None:
    from multiview_stitcher import msi_utils

    for msim, param in zip(msims, params, strict=True):
        msi_utils.set_affine_transform(
            msim,
            xaffine=spatial_affine_param(param),
            transform_key=transform_key,
            base_transform_key=base_transform_key,
        )


def msim_full_transforms(msims: list[Any], transform_key: str) -> list[Any]:
    from multiview_stitcher import msi_utils

    return [msi_utils.get_transform_from_msim(msim, transform_key=transform_key) for msim in msims]


def full_params_relative_to_stage(msims: list[Any], full_params: list[Any]) -> list[Any]:
    from multiview_stitcher import msi_utils, param_utils

    relative = []
    for msim, full_param in zip(msims, full_params, strict=True):
        stage_param = msi_utils.get_transform_from_msim(msim, transform_key=TRANSFORM_KEY)
        relative.append(
            param_utils.rebase_affine(
                spatial_affine_param(full_param),
                param_utils.invert_xparams(spatial_affine_param(stage_param)),
            )
        )
    return relative


def clamp_relative_affine_translations(
    params: list[Any],
    allowed_axes: set[str],
    *,
    reference_params: list[Any] | None = None,
) -> list[Any]:
    axis_to_row = {"z": 0, "y": 1, "x": 2}
    blocked_rows = [row for axis, row in axis_to_row.items() if axis not in allowed_axes]
    if not blocked_rows:
        return params
    if reference_params is not None and len(reference_params) != len(params):
        raise ValueError(
            f"Reference registration length {len(reference_params)} does not match params length {len(params)}"
        )

    clamped = []
    for index, param in enumerate(params):
        reference_data = (
            np.asarray(reference_params[index].data if hasattr(reference_params[index], "data") else reference_params[index])
            if reference_params is not None
            else None
        )
        if not hasattr(param, "data"):
            data = np.asarray(param, dtype=np.float64).copy()
            for row in blocked_rows:
                data[..., row, 3] = 0.0 if reference_data is None else reference_data[..., row, 3]
            clamped.append(data)
            continue

        data = np.asarray(param.data, dtype=np.float64).copy()
        for row in blocked_rows:
            data[..., row, 3] = 0.0 if reference_data is None else reference_data[..., row, 3]
        clamped.append(param.copy(data=data))
    return clamped


def rotation_matrix_from_vector_deg_zyx(rotation_vector_deg_zyx: tuple[float, float, float]) -> np.ndarray:
    rotation_vector = np.deg2rad(np.asarray(rotation_vector_deg_zyx, dtype=np.float64))
    theta = float(np.linalg.norm(rotation_vector))
    if theta == 0.0:
        return np.eye(3, dtype=np.float64)
    axis = rotation_vector / theta
    z, y, x = axis
    cross = np.asarray([[0.0, -x, y], [x, 0.0, -z], [-y, z, 0.0]], dtype=np.float64)
    ident = np.eye(3, dtype=np.float64)
    return ident + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def tile_center_um_zyx(tile: TileMetadata) -> np.ndarray:
    shape_zyx = np.asarray(tile_shape_zyx(tile), dtype=np.float64)
    spacing_zyx = np.asarray([tile_stage_scale(tile)[dim] for dim in ("z", "y", "x")], dtype=np.float64)
    origin_zyx = np.asarray([tile.translation[dim] for dim in ("z", "y", "x")], dtype=np.float64)
    return origin_zyx + ((shape_zyx - 1.0) / 2.0) * spacing_zyx


def centered_rotation_affine_zyx(matrix_zyx: np.ndarray, center_zyx: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix_zyx, dtype=np.float64)
    center = np.asarray(center_zyx, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {matrix.shape}")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = matrix
    affine[:3, 3] = center - matrix @ center
    return affine


def apply_pre_stitch_tile_rotation_to_params(
    params: list[Any],
    tiles: list[TileMetadata],
    rotation: PreStitchTileRotation,
) -> list[Any]:
    if len(params) != len(tiles):
        raise ValueError(f"Expected one registration param per tile, got {len(params)} params for {len(tiles)} tiles")
    matrix = np.asarray(rotation.matrix_physical_zyx, dtype=np.float64)
    corrected = []
    for tile, param in zip(tiles, params, strict=True):
        if rotation.center_mode == "linear_only":
            rotation_affine = np.eye(4, dtype=np.float64)
            rotation_affine[:3, :3] = matrix
        elif rotation.center_mode == "tile_center":
            rotation_affine = centered_rotation_affine_zyx(matrix, tile_center_um_zyx(tile))
        else:
            raise ValueError(f"Unsupported pre-stitch tile rotation center_mode={rotation.center_mode!r}")
        updated = param.copy(deep=True)
        updated.data = np.matmul(updated.data, rotation_affine)
        corrected.append(updated)
    return corrected


def pre_stitch_tile_rotation_payload(rotation: PreStitchTileRotation) -> dict[str, Any]:
    return {
        "model": "shared_physical_affine_before_multiview_fusion",
        "matrix_physical_zyx": [list(row) for row in rotation.matrix_physical_zyx],
        "rotation_vector_deg_zyx": None
        if rotation.rotation_vector_deg_zyx is None
        else list(rotation.rotation_vector_deg_zyx),
        "source": rotation.source,
        "inverted": rotation.inverted,
        "center_mode": rotation.center_mode,
    }


def constraint_payload(constraint: BoundaryConstraint) -> dict[str, Any]:
    def slices_payload(slices: tuple[slice, slice, slice] | None) -> list[list[int | None]] | None:
        if slices is None:
            return None
        return [[slc.start, slc.stop, slc.step] for slc in slices]

    return {
        "fixed": constraint.fixed,
        "moving": constraint.moving,
        "pair": list(constraint.pair),
        "axis": constraint.axis,
        "patch_index": constraint.patch_index,
        "shift_zyx": list(constraint.shift_zyx),
        "weight": constraint.weight,
        "correlation_before": constraint.correlation_before,
        "correlation_after": constraint.correlation_after,
        "improvement": constraint.improvement,
        "fixed_nonzero_fraction": constraint.fixed_nonzero_fraction,
        "moving_nonzero_fraction": constraint.moving_nonzero_fraction,
        "fixed_std": constraint.fixed_std,
        "moving_std": constraint.moving_std,
        "fixed_content_fraction": constraint.fixed_content_fraction,
        "moving_content_fraction": constraint.moving_content_fraction,
        "accepted": constraint.accepted,
        "reject_reason": constraint.reject_reason,
        "final_residual_zyx": (
            list(constraint.final_residual_zyx)
            if constraint.final_residual_zyx is not None
            else None
        ),
        "fixed_slices_zyx": slices_payload(constraint.fixed_slices),
        "moving_slices_zyx": slices_payload(constraint.moving_slices),
        "source_label": constraint.source_label,
    }


def robust_summary(
    constraints: list[BoundaryConstraint],
    corrections_zyx: list[tuple[float, float, float]],
) -> dict[str, Any]:
    import numpy as np

    accepted = [constraint for constraint in constraints if constraint.accepted]
    residual_norms = [
        math.sqrt(sum((constraint.final_residual_zyx or (0.0, 0.0, 0.0))[index] ** 2 for index in range(3)))
        for constraint in accepted
    ]
    correction_norms = [
        math.sqrt(sum(value * value for value in correction))
        for correction in corrections_zyx
    ]
    return {
        "total_patches": len(constraints),
        "accepted_patches": len(accepted),
        "rejected_patches": len(constraints) - len(accepted),
        "median_accepted_residual_px": float(np.median(residual_norms)) if residual_norms else None,
        "p95_accepted_residual_px": float(np.percentile(residual_norms, 95)) if residual_norms else None,
        "max_correction_px": float(max(correction_norms)) if correction_norms else 0.0,
        "reject_reasons": dict(Counter(
            constraint.reject_reason
            for constraint in constraints
            if constraint.reject_reason is not None
        )),
        "constraint_status_counts": dict(Counter(
            "accepted" if constraint.accepted else "rejected"
            for constraint in constraints
        )),
    }


def registered_bounds_um(tile: TileMetadata, param: Any) -> tuple[np.ndarray, np.ndarray]:
    correction_um = np.asarray(affine_translation_zyx(param), dtype=np.float64)
    origin_um = np.asarray([tile_sim_translation(tile)[dim] for dim in ("z", "y", "x")], dtype=np.float64)
    scale_um = np.asarray([tile_sim_scale(tile)[dim] for dim in ("z", "y", "x")], dtype=np.float64)
    shape_zyx = np.asarray(tile_shape_zyx(tile), dtype=np.float64)
    start_um = origin_um + correction_um
    stop_um = start_um + shape_zyx * scale_um
    return np.minimum(start_um, stop_um), np.maximum(start_um, stop_um)


def preview_spacing_zyx_um(tile: TileMetadata, level: int) -> np.ndarray:
    base_spacing = np.asarray([tile.spacing[dim] for dim in ("z", "y", "x")], dtype=np.float64)
    return base_spacing * (2**int(level))


def source_sampling_steps_zyx(tile: TileMetadata, target_spacing_zyx_um: np.ndarray) -> tuple[int, int, int]:
    source_spacing = np.asarray([tile.spacing[dim] for dim in ("z", "y", "x")], dtype=np.float64)
    steps = np.rint(target_spacing_zyx_um / source_spacing).astype(int)
    return tuple(max(1, int(value)) for value in steps)


def read_registered_center_z_plane(
    tile: TileMetadata,
    param: Any,
    *,
    channel: int,
    level: int,
    global_center_z_um: float,
    global_min_zyx_um: np.ndarray,
    target_spacing_zyx_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    import dask.array as da

    source_level, available_levels = fusion_source_level_for_tile(tile, level)
    store = None
    try:
        zarray, store = open_tile_array(tile, source_level=source_level)
        source_tile = fusion_tile_for_source_array(
            tile,
            tuple(int(value) for value in zarray.shape),
            source_level=source_level,
        )
        start_um, stop_um = registered_bounds_um(source_tile, param)
        if global_center_z_um < start_um[0] or global_center_z_um >= stop_um[0]:
            return None

        shape_zyx = tile_shape_zyx(source_tile)
        steps_zyx = source_sampling_steps_zyx(source_tile, target_spacing_zyx_um)
        local_z = int(round((global_center_z_um - start_um[0]) / source_tile.spacing["z"]))
        local_z = min(max(local_z, 0), shape_zyx[0] - 1)
        z_flip, y_flip, x_flip = tile_flip_axes_zyx(source_tile)
        source_z = shape_zyx[0] - 1 - local_z if z_flip else local_z
        y_slice = slice(None, None, -steps_zyx[1] if y_flip else steps_zyx[1])
        x_slice = slice(None, None, -steps_zyx[2] if x_flip else steps_zyx[2])

        array = da.from_zarr(zarray)
        if source_tile.axes == "CZYX":
            if channel < 0 or channel >= source_tile.shape[0]:
                raise ValueError(f"Channel {channel} is outside {source_tile.path} channel count {source_tile.shape[0]}")
            plane = array[channel, source_z, y_slice, x_slice]
        elif source_tile.axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} is outside single-channel tile {source_tile.path}")
            plane = array[source_z, y_slice, x_slice]
        else:
            raise ValueError(f"Expected CZYX or ZYX axes in {source_tile.path}, got {source_tile.axes}")

        start_zyx = np.rint((start_um - global_min_zyx_um) / target_spacing_zyx_um).astype(int)
        metadata = {
            "tile": tile.path.name,
            "source_level": source_level,
            "available_levels": available_levels,
            "source_z_index": int(source_z),
            "registered_start_zyx_um": start_um.tolist(),
            "level_start_zyx": [int(value) for value in start_zyx],
            "source_sampling_steps_zyx": [int(value) for value in steps_zyx],
        }
        return np.asarray(plane.compute()).astype(np.float32, copy=False), start_zyx, metadata
    finally:
        if store is not None:
            close = getattr(store, "close", None)
            if callable(close):
                close()


def add_plane_to_average_canvas(
    sum_canvas: np.ndarray,
    count_canvas: np.ndarray,
    plane: np.ndarray,
    start_yx: tuple[int, int],
) -> None:
    y0, x0 = start_yx
    if y0 >= sum_canvas.shape[0] or x0 >= sum_canvas.shape[1]:
        return
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    y_size = min(plane.shape[0] - src_y0, sum_canvas.shape[0] - dst_y0)
    x_size = min(plane.shape[1] - src_x0, sum_canvas.shape[1] - dst_x0)
    if y_size <= 0 or x_size <= 0:
        return

    source = plane[src_y0 : src_y0 + y_size, src_x0 : src_x0 + x_size]
    valid = np.isfinite(source)
    if not np.any(valid):
        return
    target_sum = sum_canvas[dst_y0 : dst_y0 + y_size, dst_x0 : dst_x0 + x_size]
    target_count = count_canvas[dst_y0 : dst_y0 + y_size, dst_x0 : dst_x0 + x_size]
    target_sum[valid] += source[valid]
    target_count[valid] += 1


def write_registered_center_z_preview(
    output_dir: Path,
    tiles: list[TileMetadata],
    params: list[Any],
    *,
    channel: int,
    level: int = 4,
) -> Path:
    from PIL import Image

    if len(tiles) != len(params):
        raise ValueError(f"Cannot render registration preview: {len(tiles)} tiles but {len(params)} params")
    target_spacing_zyx_um = preview_spacing_zyx_um(tiles[0], level)
    bounds = [registered_bounds_um(tile, param) for tile, param in zip(tiles, params, strict=True)]
    global_min = np.min([bound[0] for bound in bounds], axis=0)
    global_max = np.max([bound[1] for bound in bounds], axis=0)
    shape_zyx = np.ceil((global_max - global_min) / target_spacing_zyx_um).astype(int)
    if np.any(shape_zyx <= 0):
        raise ValueError(f"Cannot render registration preview with invalid level shape {shape_zyx.tolist()}")

    center_z_um = float((global_min[0] + global_max[0]) / 2.0)
    sum_canvas = np.zeros((int(shape_zyx[1]), int(shape_zyx[2])), dtype=np.float32)
    count_canvas = np.zeros_like(sum_canvas, dtype=np.uint16)
    tile_rows: list[dict[str, Any]] = []

    for tile, param in zip(tiles, params, strict=True):
        result = read_registered_center_z_plane(
            tile,
            param,
            channel=channel,
            level=level,
            global_center_z_um=center_z_um,
            global_min_zyx_um=global_min,
            target_spacing_zyx_um=target_spacing_zyx_um,
        )
        if result is None:
            tile_rows.append({"tile": tile.path.name, "intersects_center_z": False})
            continue
        plane, start_zyx, metadata = result
        add_plane_to_average_canvas(sum_canvas, count_canvas, plane, (int(start_zyx[1]), int(start_zyx[2])))
        metadata["intersects_center_z"] = True
        metadata["sampled_plane_shape_yx"] = [int(value) for value in plane.shape]
        tile_rows.append(metadata)
        log(
            "Placed registration preview tile "
            f"{tile.path.name} start_yx={[int(start_zyx[1]), int(start_zyx[2])]} "
            f"shape_yx={[int(value) for value in plane.shape]}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        preview = np.where(count_canvas > 0, sum_canvas / np.maximum(count_canvas, 1), 0.0)
    image_path = output_dir / f"level{level}_registered_centerZ_placement_ch{channel}.png"
    Image.fromarray(scale_thumbnail_uint8(preview)).save(image_path)
    summary_path = output_dir / f"level{level}_registered_centerZ_placement_ch{channel}.json"
    summary_path.write_text(
        json.dumps(
            {
                "channel": channel,
                "level": level,
                "center_z_um": center_z_um,
                "global_min_zyx_um": global_min.tolist(),
                "global_max_zyx_um": global_max.tolist(),
                "target_spacing_zyx_um": target_spacing_zyx_um.tolist(),
                "level_shape_zyx": [int(value) for value in shape_zyx],
                "placed_tile_count": int(np.count_nonzero([row.get("intersects_center_z", False) for row in tile_rows])),
                "tiles": tile_rows,
                "image": str(image_path.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    log(f"Wrote registered center-z placement preview: {image_path}")
    return image_path


def write_registered_center_z_reference_overlay(
    output_dir: Path,
    tiles: list[TileMetadata],
    reference_params: list[Any],
    moving_params: list[Any],
    *,
    reference_channel: int,
    moving_channel: int,
    level: int = 4,
) -> Path:
    from PIL import Image

    if len(tiles) != len(reference_params) or len(tiles) != len(moving_params):
        raise ValueError(
            "Cannot render reference overlay: "
            f"{len(tiles)} tiles, {len(reference_params)} reference params, "
            f"{len(moving_params)} moving params"
        )
    target_spacing_zyx_um = preview_spacing_zyx_um(tiles[0], level)
    bounds = [
        registered_bounds_um(tile, param)
        for params in (reference_params, moving_params)
        for tile, param in zip(tiles, params, strict=True)
    ]
    global_min = np.min([bound[0] for bound in bounds], axis=0)
    global_max = np.max([bound[1] for bound in bounds], axis=0)
    center_z_um = float((global_min[0] + global_max[0]) / 2.0)
    shape_zyx = np.ceil((global_max - global_min) / target_spacing_zyx_um).astype(int)
    if np.any(shape_zyx <= 0):
        raise ValueError(f"Cannot render reference overlay with invalid level shape {shape_zyx.tolist()}")

    def render_canvas(params: list[Any], channel: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        sum_canvas = np.zeros((int(shape_zyx[1]), int(shape_zyx[2])), dtype=np.float32)
        count_canvas = np.zeros_like(sum_canvas, dtype=np.uint16)
        rows: list[dict[str, Any]] = []
        for tile, param in zip(tiles, params, strict=True):
            result = read_registered_center_z_plane(
                tile,
                param,
                channel=channel,
                level=level,
                global_center_z_um=center_z_um,
                global_min_zyx_um=global_min,
                target_spacing_zyx_um=target_spacing_zyx_um,
            )
            if result is None:
                rows.append({"tile": tile.path.name, "intersects_center_z": False})
                continue
            plane, start_zyx, metadata = result
            add_plane_to_average_canvas(sum_canvas, count_canvas, plane, (int(start_zyx[1]), int(start_zyx[2])))
            metadata["intersects_center_z"] = True
            metadata["sampled_plane_shape_yx"] = [int(value) for value in plane.shape]
            rows.append(metadata)
        with np.errstate(divide="ignore", invalid="ignore"):
            canvas = np.where(count_canvas > 0, sum_canvas / np.maximum(count_canvas, 1), 0.0)
        return canvas, rows

    reference_canvas, reference_rows = render_canvas(reference_params, reference_channel)
    moving_canvas, moving_rows = render_canvas(moving_params, moving_channel)
    rgb = np.zeros((*reference_canvas.shape, 3), dtype=np.uint8)
    rgb[..., 0] = scale_thumbnail_uint8(reference_canvas)
    rgb[..., 1] = scale_thumbnail_uint8(moving_canvas)

    image_path = output_dir / (
        f"level{level}_registered_centerZ_overlay_refCh{reference_channel}_red_"
        f"ch{moving_channel}_green.png"
    )
    Image.fromarray(rgb).save(image_path)
    summary_path = image_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "level": level,
                "center_z_um": center_z_um,
                "reference_channel": reference_channel,
                "moving_channel": moving_channel,
                "color_mapping": {
                    f"channel_{reference_channel}": "red",
                    f"channel_{moving_channel}": "green",
                    "overlap": "yellow",
                },
                "global_min_zyx_um": global_min.tolist(),
                "global_max_zyx_um": global_max.tolist(),
                "target_spacing_zyx_um": target_spacing_zyx_um.tolist(),
                "level_shape_zyx": [int(value) for value in shape_zyx],
                "reference_placed_tile_count": int(
                    np.count_nonzero([row.get("intersects_center_z", False) for row in reference_rows])
                ),
                "moving_placed_tile_count": int(
                    np.count_nonzero([row.get("intersects_center_z", False) for row in moving_rows])
                ),
                "reference_tiles": reference_rows,
                "moving_tiles": moving_rows,
                "image": str(image_path.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    log(f"Wrote registered center-z 488/channel overlay: {image_path}")
    return image_path


def write_robust_boundary_qc(
    output_dir: Path,
    tiles: list[TileMetadata],
    params: list[Any],
    *,
    channel: int,
    constraints: list[BoundaryConstraint],
    corrections_zyx: list[tuple[float, float, float]],
    summary: dict[str, Any],
    reference_geometry: ReferenceGeometryConstraint | None = None,
    reference_params: list[Any] | None = None,
    reference_channel: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals_path = output_dir / "boundary_residuals.jsonl"
    with residuals_path.open("w") as stream:
        for constraint in constraints:
            stream.write(json.dumps(constraint_payload(constraint)) + "\n")
    (output_dir / "tile_corrections.json").write_text(
        json.dumps(
            {
                "corrections_zyx_px": corrections_zyx,
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    write_registered_center_z_preview(output_dir, tiles, params, channel=channel, level=4)
    if reference_params is not None:
        write_registered_center_z_reference_overlay(
            output_dir,
            tiles,
            reference_params,
            params,
            reference_channel=reference_channel,
            moving_channel=channel,
            level=4,
        )
    write_reference_geometry_qc(output_dir, reference_geometry)
    log(f"Wrote robust boundary residual QC: {residuals_path}")


def refine_registration_with_robust_boundaries(
    tiles: list[TileMetadata],
    params: list[Any],
    *,
    channel: int,
    output_dir: Path,
    settings: RobustBoundarySettings,
    reference_params: list[Any] | None = None,
    reference_input: Path | None = None,
    fixed_reference_axes: set[str] | None = None,
    reference_prior_weights_zyx: tuple[float, float, float] | None = None,
    residual_reject_axes: set[str] | None = None,
    reference_geometry_mode: str | None = None,
    source_label: str | None = None,
    shared_geometry_tracks: tuple[str, ...] = (),
) -> RobustBoundaryRefinementResult:
    require_cuda_for_robust_boundary()
    pairs = axis_aligned_registration_pairs(tiles)
    patch_specs = sample_boundary_patches(tiles, params, pairs, settings)
    log(
        "Robust boundary refinement sampled "
        f"{len(patch_specs)} patch(es) from {len(pairs)} axis-aligned edge(s)"
    )
    constraints = build_boundary_constraints(tiles, channel, patch_specs, settings)
    if source_label is not None:
        constraints = [replace(constraint, source_label=source_label) for constraint in constraints]
    corrections_zyx, constraints, anchor_tile = solve_tile_corrections_with_residual_rejection(
        tiles,
        constraints,
        settings,
        fixed_axes=fixed_reference_axes,
        reference_prior_weights_zyx=reference_prior_weights_zyx,
        residual_reject_axes=residual_reject_axes,
    )
    corrected_params = apply_corrections_to_params(params, corrections_zyx, tiles[0].spacing)
    reference_geometry = None
    if reference_params is not None:
        if fixed_reference_axes:
            corrected_params = apply_reference_fixed_axes(corrected_params, reference_params, fixed_reference_axes)
        if reference_input is None:
            raise ValueError("reference_input is required when reference_params are provided")
        reference_geometry = reference_geometry_constraint(
            mode=(
                reference_geometry_mode
                or ("fixed-xy" if fixed_reference_axes == {"y", "x"} else "fixed-axes")
            ),
            reference_input=reference_input,
            fixed_axes=fixed_reference_axes or set(),
            params=corrected_params,
            reference_params=reference_params,
            constraints=constraints,
            shared_geometry_tracks=shared_geometry_tracks,
            reference_prior_weights_zyx=reference_prior_weights_zyx,
            residual_reject_axes=residual_reject_axes,
        )
    summary = robust_summary(constraints, corrections_zyx)
    write_robust_boundary_qc(
        output_dir,
        tiles,
        corrected_params,
        channel=channel,
        constraints=constraints,
        corrections_zyx=corrections_zyx,
        summary=summary,
        reference_geometry=reference_geometry,
        reference_params=reference_params,
    )
    log(
        "Robust boundary refinement summary: "
        f"accepted={summary['accepted_patches']}/{summary['total_patches']}, "
        f"median_residual={summary['median_accepted_residual_px']}, "
        f"p95_residual={summary['p95_accepted_residual_px']}, "
        f"max_correction_px={summary['max_correction_px']:.3f}, "
        f"anchor_tile={anchor_tile}"
    )
    return RobustBoundaryRefinementResult(
        params=corrected_params,
        constraints=constraints,
        corrections_zyx=corrections_zyx,
        anchor_tile=anchor_tile,
        output_dir=output_dir,
        summary=summary,
        reference_geometry=reference_geometry,
    )


def narrow_fusion_blending_widths(
    tiles: list[TileMetadata],
    params: list[Any] | None,
    *,
    default_width_voxels_zyx: tuple[float, float, float] = (64.0, 64.0, 64.0),
    max_overlap_fraction: float = 0.25,
) -> dict[str, float]:
    """Return multiview-stitcher blending widths in physical z/y/x units.

    The package expects physical widths. Choose widths in voxels first so
    anisotropic z/y/x spacing still blends over the same number of pixels.
    """
    pairs = axis_aligned_registration_pairs(tiles)
    bounds = [
        tile_registered_bounds_zyx(tile, None if params is None else params[index])
        for index, tile in enumerate(tiles)
    ]
    positive_overlaps: dict[str, list[float]] = {"z": [], "y": [], "x": []}
    for left, right in pairs:
        for dim_index, dim in enumerate(("z", "y", "x")):
            overlap = min(bounds[left].stop_zyx[dim_index], bounds[right].stop_zyx[dim_index]) - max(
                bounds[left].start_zyx[dim_index],
                bounds[right].start_zyx[dim_index],
            )
            if overlap > 0:
                positive_overlaps[dim].append(float(overlap))

    widths = {}
    for dim_index, dim in enumerate(("z", "y", "x")):
        if positive_overlaps[dim]:
            width_px = min(default_width_voxels_zyx[dim_index], max_overlap_fraction * min(positive_overlaps[dim]))
        else:
            width_px = default_width_voxels_zyx[dim_index]
        widths[dim] = width_px * tiles[0].spacing[dim]
    return widths


def print_plan(
    tiles: list[TileMetadata],
    output: Path,
    registration_output: Path,
    registration_plots_dir: Path,
    robust_boundary_qc_dir: Path,
    selected_channels: tuple[int, ...] | None,
    register_only: bool,
    register: bool,
    fusion_compressor: str,
    jpegxr_level: float,
    registration_pair_mode: str,
    registration_binning: tuple[int, int, int] | None,
    reg_res_level: int | None,
    n_parallel_pairwise_regs: int | None,
    dask_num_workers: int | None,
    registration_read_chunk_z: int,
) -> None:
    first = tiles[0]
    channel_count = tile_channel_count(first) if selected_channels is None else len(selected_channels)
    output_shape = estimate_output_shape(tiles)
    log(f"Input tiles: {len(tiles)}")
    log(f"Tile shape: {first.shape} ({first.axes})")
    log(f"Spacing um: {first.spacing}")
    if not register_only:
        log(f"Output zarr base: {output}")
        log(f"Fusion backend: {FUSION_BACKEND}")
        if fusion_compressor == "jpegxr":
            log(
                "Fusion output format: "
                f"OME-NGFF {NGFF_VERSION} / Zarr v3, JPEG-XR level "
                f"{compression_level(jpegxr_level):.3f} after GPU fusion"
            )
        else:
            log(
                "Fusion output format: "
                f"OME-NGFF {NGFF_VERSION} / Zarr v3, zstd level {ZSTD_LEVEL}"
            )
    if register or register_only:
        log(f"Registration binning (z, y, x): {registration_binning or 'auto'}")
        log(f"Registration resolution level: {reg_res_level if reg_res_level is not None else 'auto'}")
        log(f"Registration read chunk z: {registration_read_chunk_z}")
        log(f"Pairwise jobs: {n_parallel_pairwise_regs or 'auto'}")
        log(f"Dask registration workers: {dask_num_workers or 'default'}")
        log(f"Registration pair mode: {registration_pair_mode}")
        log(f"Registration metric plots: {registration_plots_dir}")
        if registration_pair_mode == "robust-boundary":
            log(f"Robust boundary QC: {robust_boundary_qc_dir}")
    if register_only:
        log(f"Registration output: {registration_output}")
    log(
        "Estimated metadata mosaic shape (c, z, y, x): "
        f"({channel_count}, {output_shape['z']}, {output_shape['y']}, {output_shape['x']})"
    )
    log("Tile stage transforms:")
    for tile in tiles:
        scale = tile_stage_scale(tile)
        scale_suffix = ""
        if scale != tile.spacing:
            scale_suffix = f", scale z={scale['z']:.6g}, y={scale['y']:.6g}, x={scale['x']:.6g} um/px"
        log(
            f"  {tile.path.name}: "
            f"z={tile.translation['z']:.3f}, "
            f"y={tile.translation['y']:.3f}, "
            f"x={tile.translation['x']:.3f} um"
            f"{scale_suffix}"
        )


def channel_labels_for_tiles(tiles: list[TileMetadata]) -> list[str]:
    channel_labels = list(tiles[0].channels)
    if not channel_labels:
        channel_labels = [str(index) for index in range(tile_channel_count(tiles[0]))]
    return channel_labels


def open_tile_array(tile: TileMetadata, *, source_level: int = 0):
    import tifffile
    import zarr

    store = tifffile.imread(tile.path, aszarr=True, level=source_level)
    zarray = zarr.open(store, mode="r")
    return zarray, store


def open_registration_tile_array(
    tile: TileMetadata,
    channel: int,
    *,
    read_chunk_z: int,
    source_level: int = 0,
):
    import dask.array as da
    import tifffile
    import zarr

    store = tifffile.imread(tile.path, aszarr=True, level=source_level)
    zarray = zarr.open(store, mode="r")
    zarr_chunks = tuple(int(chunk) for chunk in zarray.chunks)
    source_shape = tuple(int(value) for value in zarray.shape)
    source_tile = fusion_tile_for_source_array(tile, source_shape, source_level=source_level)
    if tile.axes == "CZYX":
        read_chunks = (
            1,
            min(read_chunk_z, source_tile.shape[1]),
            zarr_chunks[2],
            zarr_chunks[3],
        )
        return da.from_zarr(zarray, chunks=read_chunks)[channel : channel + 1, :, :, :], store, source_tile
    if tile.axes == "ZYX":
        if channel != 0:
            raise ValueError(f"Channel {channel} out of range for single-channel ZYX tile {tile.path}")
        read_chunks = (
            min(read_chunk_z, source_tile.shape[0]),
            zarr_chunks[1],
            zarr_chunks[2],
        )
        return da.from_zarr(zarray, chunks=read_chunks)[None, :, :, :], store, source_tile
    raise ValueError(f"Expected CZYX or ZYX axes in {tile.path}, got {tile.axes}")


def build_registration_msims(
    tiles: list[TileMetadata],
    channel: int,
    *,
    read_chunk_z: int,
    source_level: int = 0,
):
    from multiview_stitcher import msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils

    msims = []
    stores = []
    channel_labels = channel_labels_for_tiles(tiles)

    for tile in tiles:
        array, store, source_tile = open_registration_tile_array(
            tile,
            channel,
            read_chunk_z=read_chunk_z,
            source_level=source_level,
        )
        if not msims:
            log(
                "Registration source dask chunks: "
                f"c={array.chunks[0]}, z_max={max(array.chunks[1])}, "
                f"y={array.chunks[2]}, x={array.chunks[3]}"
            )
            log(
                "Registration source tile shape/spacing: "
                f"shape={source_tile.shape}, spacing={source_tile.spacing}"
            )
        stores.append(store)
        array = flip_spatial_array_for_stage_scale(array, source_tile, has_channel_axis=True)
        sim = si_utils.get_sim_from_array(
            array,
            dims=["c", "z", "y", "x"],
            scale=tile_sim_scale(source_tile),
            translation=tile_sim_translation(source_tile),
            transform_key=TRANSFORM_KEY,
            c_coords=[channel_labels[channel]],
        )
        msims.append(msi_utils.get_msim_from_sim(sim))

    return msims, stores, channel_labels[channel]


def build_fusion_sims(tiles: list[TileMetadata], channel: int, *, fusion_level: int = 0):
    from multiview_stitcher import spatial_image_utils as si_utils

    sims = []
    stores = []
    source_tiles = []
    channel_labels = channel_labels_for_tiles(tiles)
    source_level_counts: Counter[tuple[int, int]] = Counter()

    for tile in tiles:
        source_level, available_levels = fusion_source_level_for_tile(tile, fusion_level)
        array, store = open_tile_array(tile, source_level=source_level)
        stores.append(store)
        source_level_counts[(source_level, available_levels)] += 1
        source_tile = fusion_tile_for_source_array(
            tile,
            tuple(int(value) for value in array.shape),
            source_level=source_level,
        )
        source_tiles.append(source_tile)
        if tile.axes == "ZYX":
            array = flip_spatial_array_for_stage_scale(array, source_tile, has_channel_axis=False)
            dims = ["z", "y", "x"]
        else:
            array = flip_spatial_array_for_stage_scale(array, source_tile, has_channel_axis=True)
            dims = ["c", "z", "y", "x"]
        sim = si_utils.get_sim_from_array(
            array,
            dims=dims,
            scale=tile_sim_scale(source_tile),
            translation=tile_sim_translation(source_tile),
            transform_key=TRANSFORM_KEY,
            c_coords=channel_labels,
        )
        # Scalar selection preserves the original TIFF channel index in
        # multiview-stitcher's zarr-backed serializer. Drop singleton t too:
        # the package otherwise deep-copies TIFF stores while iterating over t.
        selected = sim.isel(c=channel, t=0)
        selected.attrs["basic_tile_cache_key"] = f"{tile.path.resolve()}::ch{channel}"
        if tile.source_view is not None:
            selected.attrs["source_view"] = tile.source_view
        sims.append(selected)

    return sims, stores, channel_labels[channel], source_level_counts, source_tiles


def channel_output_path(output: Path, channel: int, *, separate_channels: bool) -> Path:
    if not separate_channels:
        return output

    name = output.name
    if name.endswith(".ome.zarr"):
        name = f"{name.removesuffix('.ome.zarr')}.ch{channel}.ome.zarr"
    elif name.endswith(".zarr"):
        name = f"{name.removesuffix('.zarr')}.ch{channel}.zarr"
    else:
        name = f"{name}.ch{channel}.zarr"
    return output.with_name(name)


def insert_track_suffix(path: Path, track_slug_value: str) -> Path:
    name = path.name
    dotted_slug = f".{track_slug_value}."
    if dotted_slug in name or name.startswith(f"{track_slug_value}-"):
        return path
    if ".multiview-stitcher" in name:
        name = name.replace(".multiview-stitcher", f".{track_slug_value}.multiview-stitcher", 1)
    elif name.endswith(".ome.zarr"):
        name = f"{name.removesuffix('.ome.zarr')}.{track_slug_value}.ome.zarr"
    elif name.endswith(".zarr"):
        name = f"{name.removesuffix('.zarr')}.{track_slug_value}.zarr"
    elif name.endswith(".json"):
        name = f"{name.removesuffix('.json')}.{track_slug_value}.json"
    else:
        name = f"{name}.{track_slug_value}"
    return path.with_name(name)


def track_registration_plots_dir(path: Path, track_slug_value: str) -> Path:
    name = path.name
    if name.startswith(f"{track_slug_value}-") or name.endswith(f"-{track_slug_value}"):
        return path
    return path.with_name(f"{track_slug_value}-{name}")


def track_qc_dir(path: Path, track_slug_value: str) -> Path:
    return track_registration_plots_dir(path, track_slug_value)


def selected_track_metadata(
    tracks: tuple[TrackMetadata, ...],
    selected_channels: tuple[int, ...] | None,
) -> tuple[TrackMetadata, ...]:
    if selected_channels is None:
        return tracks

    selected = set(selected_channels)
    filtered = []
    for track in tracks:
        channels = tuple(channel for channel in track.channels if channel in selected)
        if not channels:
            continue
        name_by_channel = dict(zip(track.channels, track.channel_names, strict=False))
        names = tuple(name_by_channel.get(channel, str(channel)) for channel in channels)
        filtered.append(
            TrackMetadata(
                slug=track.slug,
                track_id=track.track_id,
                channels=channels,
                channel_names=names,
            )
        )
    return tuple(filtered)


def flatfield_path(flatfield_dir: Path, channel: int) -> Path:
    matches = sorted(flatfield_dir.glob(f"*-ch{channel}-flatfield.tif"))
    if not matches:
        raise FileNotFoundError(f"No *-ch{channel}-flatfield.tif found in {flatfield_dir}")
    if len(matches) > 1:
        raise ValueError(f"Expected one ch{channel} flatfield TIFF in {flatfield_dir}, found {matches}")
    return matches[0]


def flatfield_scale_path(flatfield_dir: Path, channel: int) -> Path | None:
    matches = sorted(flatfield_dir.glob(f"*-ch{channel}-*-corrected-max.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one ch{channel} corrected-max JSON in {flatfield_dir}, found {matches}")
    return matches[0]


def load_flatfield_pre_scale(flatfield_dir: Path, channel: int) -> tuple[float, Path | None]:
    path = flatfield_scale_path(flatfield_dir, channel)
    if path is None:
        return 1.0, None
    payload = json.loads(path.read_text())
    pre_scale = float(payload["pre_scale_for_uint16"])
    if not math.isfinite(pre_scale) or pre_scale <= 0:
        raise ValueError(f"{path} has invalid pre_scale_for_uint16={pre_scale}")
    return pre_scale, path


def downsample_flatfield_to_shape(flatfield: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    if flatfield.shape == expected_shape:
        return flatfield

    src_y, src_x = (int(flatfield.shape[0]), int(flatfield.shape[1]))
    dst_y, dst_x = (int(expected_shape[0]), int(expected_shape[1]))
    if dst_y <= 0 or dst_x <= 0 or dst_y > src_y or dst_x > src_x:
        raise ValueError(f"Cannot downsample flatfield shape {flatfield.shape} to {expected_shape}")

    factor_y = math.ceil(src_y / dst_y)
    factor_x = math.ceil(src_x / dst_x)
    if (src_y + factor_y - 1) // factor_y != dst_y or (src_x + factor_x - 1) // factor_x != dst_x:
        raise ValueError(f"Cannot downsample flatfield shape {flatfield.shape} to {expected_shape}")

    y_starts = np.arange(0, src_y, factor_y)
    x_starts = np.arange(0, src_x, factor_x)
    y_counts = np.diff(np.append(y_starts, src_y)).astype(np.float32)
    x_counts = np.diff(np.append(x_starts, src_x)).astype(np.float32)
    y_sums = np.add.reduceat(flatfield.astype(np.float32, copy=False), y_starts, axis=0)
    xy_sums = np.add.reduceat(y_sums, x_starts, axis=1)
    return xy_sums / y_counts[:, None] / x_counts[None, :]


@profile
def load_inverse_flatfield(flatfield_dir: Path, channel: int, tile_shape: tuple[int, ...]):
    import numpy as np
    import tifffile

    path = flatfield_path(flatfield_dir, channel)
    flatfield = np.asarray(tifffile.imread(path), dtype=np.float32)
    expected_shape = tile_shape[-2:]
    if flatfield.shape != expected_shape:
        try:
            flatfield = downsample_flatfield_to_shape(flatfield, expected_shape)
        except ValueError as exc:
            raise ValueError(f"{path} has shape {flatfield.shape}; expected {expected_shape}") from exc
        log(f"Downsampled BaSiC flatfield {path} to shape {expected_shape}")
    if not np.all(np.isfinite(flatfield)) or np.any(flatfield <= 0):
        raise ValueError(f"{path} must contain positive finite flatfield values")

    pre_scale, scale_path = load_flatfield_pre_scale(flatfield_dir, channel)
    inverse = 1.0 / (pre_scale * flatfield)
    return inverse.astype(np.float32, copy=False), path, pre_scale, scale_path


def load_fusion_inverse_flatfields(
    tiles: list[TileMetadata],
    *,
    flatfield_dir: Path,
    flatfield_dirs_by_source_view: dict[str, Path] | None = None,
    channel: int,
):
    if flatfield_dirs_by_source_view:
        missing_source_view = [tile.path.name for tile in tiles if tile.source_view is None]
        if missing_source_view:
            raise ValueError(
                "--flatfield-dir-by-source-view requires every tile to have source_view; "
                f"missing for {missing_source_view[:5]}"
            )
        source_views = sorted({tile.source_view for tile in tiles if tile.source_view is not None})
        missing = [view for view in source_views if view not in flatfield_dirs_by_source_view]
        if missing:
            raise ValueError(
                "Missing --flatfield-dir-by-source-view entries for source_view(s): "
                f"{missing}"
            )

        inverse_by_view = {}
        for view in source_views:
            view_flatfield_dir = flatfield_dirs_by_source_view[view]
            inverse, source, pre_scale, scale_source = load_inverse_flatfield(
                view_flatfield_dir,
                channel,
                tiles[0].shape,
            )
            inverse_by_view[view] = inverse
            log(f"Fusing channel {channel} source_view={view} with BaSiC flatfield {source}")
            if scale_source is None:
                log(
                    f"BaSiC pre-scale for channel {channel} source_view={view}: "
                    f"{pre_scale:.9g} (no corrected-max JSON found)"
                )
            else:
                log(
                    f"BaSiC pre-scale for channel {channel} source_view={view}: "
                    f"{pre_scale:.9g} from {scale_source}"
                )
            log(
                "Normalized inverse flatfield stats: "
                f"source_view={view}, "
                f"shape={inverse.shape}, "
                f"min={float(inverse.min()):.6f}, "
                f"max={float(inverse.max()):.6f}"
            )
        return inverse_by_view

    inverse, source, pre_scale, scale_source = load_inverse_flatfield(flatfield_dir, channel, tiles[0].shape)
    log(f"Fusing channel {channel} with pooled BaSiC flatfield {source}")
    if scale_source is None:
        log(f"BaSiC pre-scale for channel {channel}: {pre_scale:.9g} (no corrected-max JSON found)")
    else:
        log(f"BaSiC pre-scale for channel {channel}: {pre_scale:.9g} from {scale_source}")
    log(
        "Normalized inverse flatfield stats: "
        f"shape={inverse.shape}, "
        f"min={float(inverse.min()):.6f}, "
        f"max={float(inverse.max()):.6f}"
    )
    return inverse


def _validate_basic_correction_bounds(
    flatfield_shape: tuple[int, int],
    y0: int,
    x0: int,
    y_size: int,
    x_size: int,
    *,
    message_prefix: str,
) -> None:
    if y0 < 0 or x0 < 0 or y0 + y_size > flatfield_shape[0] or x0 + x_size > flatfield_shape[1]:
        raise ValueError(
            f"{message_prefix} crop does not match zarr slice: "
            f"flatfield_shape={flatfield_shape}, slice={(y_size, x_size)}, origin={(y0, x0)}"
        )


def _apply_basic_correction_gpu(data: Any, correction_gpu: Any, reshape: list[int], data_dtype: np.dtype):
    import cupy as cp

    corrected = cp.asarray(data, dtype=cp.float32)
    if isinstance(data, cp.ndarray) and corrected is data:
        corrected = corrected.copy()
    corrected *= correction_gpu.reshape(reshape)
    if np.issubdtype(data_dtype, np.integer):
        cp.rint(corrected, out=corrected)
        cp.clip(corrected, 0, np.iinfo(data_dtype).max, out=corrected)
        return corrected.astype(data_dtype, copy=False)
    return corrected.astype(np.float32, copy=False)


def _copy_selected_attrs(selected: Any, sim: Any, extra_attr_keys: tuple[str, ...]) -> None:
    for key in extra_attr_keys:
        if key in sim.attrs:
            selected.attrs[key] = sim.attrs[key]


def _basic_slice_offsets(info: dict[str, Any], overlap_bb: dict[str, Any], dims: tuple[str, ...]) -> dict[str, int]:
    offsets = {}
    for dim in dims:
        if dim not in {"z", "y", "x"}:
            continue
        offsets[dim] = int(round((overlap_bb["origin"][dim] - info["origin"][dim]) / info["spacing"][dim]))
    return offsets


def _basic_slice_shape(overlap_bb: dict[str, Any], dims: tuple[str, ...]) -> dict[str, int]:
    return {dim: int(overlap_bb["shape"][dim]) for dim in dims if dim in {"z", "y", "x"}}


def _basic_cache_request_sim(
    cached: dict[str, Any],
    info: dict[str, Any],
    overlap_bb: dict[str, Any],
    si_utils: Any,
):
    from multiview_stitcher import param_utils

    dims = cached["dims"]
    offsets = _basic_slice_offsets(info, overlap_bb, dims)
    cached_offsets = cached.get("offsets", {})
    shape = _basic_slice_shape(overlap_bb, dims)
    indexer = []
    for dim in dims:
        if dim in offsets:
            start = offsets[dim] - int(cached_offsets.get(dim, 0))
            stop = start + shape[dim]
            axis_size = cached["data"].shape[dims.index(dim)]
            clipped_start = max(0, start)
            clipped_stop = min(axis_size, stop)
            if clipped_start >= clipped_stop:
                raise ValueError(
                    f"BaSiC tile cache entry does not cover requested {dim} slice: "
                    f"request={offsets[dim]}:{offsets[dim] + shape[dim]}, "
                    f"cache_offset={cached_offsets.get(dim, 0)}, "
                    f"cache_shape={cached['data'].shape}"
                )
            indexer.append(slice(clipped_start, clipped_stop))
        else:
            indexer.append(slice(None))
    data = cached["data"][tuple(indexer)]
    spatial_dims = [dim for dim in dims if dim in {"z", "y", "x"}]
    sim = si_utils.to_spatial_image(
        data,
        dims=dims,
        scale={dim: info["spacing"][dim] for dim in spatial_dims},
        translation={dim: overlap_bb["origin"][dim] for dim in spatial_dims},
        c_coords=info["c_coords"] if "c" in dims else None,
        t_coords=info["t_coords"] if "t" in dims else None,
    )
    si_utils.set_sim_affine(
        sim,
        param_utils.identity_transform(len(spatial_dims), t_coords=None),
        transform_key=si_utils.DEFAULT_TRANSFORM_KEY,
    )
    return sim


@profile
@contextmanager
def basic_corrected_zarr_reads(
    inverse_flatfields,
    *,
    dataset_info_key: str | None = None,
    dataset_attr_keys: tuple[str, ...] = (),
    cache_key_attr: str | None = None,
    tile_cache_size: int = 0,
    tile_cache_max_bytes: int | None = None,
    tile_cache_disk_dir: Path | None = None,
    tile_cache_z_chunk: int = 16,
    log_prefix: str = "BaSiC-correcting zarr slice",
    error_prefix: str = "BaSiC correction",
):
    import numpy as np
    import cupy as cp
    from multiview_stitcher import spatial_image_utils as si_utils

    default_dataset = "__default__"
    if isinstance(inverse_flatfields, dict):
        inverse_by_dataset = inverse_flatfields
    else:
        inverse_by_dataset = {default_dataset: inverse_flatfields}

    original_deserialize = si_utils.deserialize_zarr_backed_sim
    original_serialize = si_utils.serialize_zarr_backed_sim
    original_sim_sel_coords = si_utils.sim_sel_coords
    inverse_gpu = {
        dataset: cp.asarray(inverse, dtype=cp.float32)
        for dataset, inverse in inverse_by_dataset.items()
    }
    flatfield_fingerprints = {
        dataset: basic_array_fingerprint(np.asarray(inverse))
        for dataset, inverse in inverse_by_dataset.items()
    }
    state = {"corrected_slices": 0}
    tile_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
    cache_stats = {
        "hits": 0,
        "misses": 0,
        "disk_hits": 0,
        "evictions": 0,
        "loaded_bytes": 0,
        "current_bytes": 0,
    }
    cache_lock = threading.RLock()

    def log_basic_cache_summary(*, event: str, key: str) -> None:
        total_requests = cache_stats["hits"] + cache_stats["misses"]
        if total_requests <= 3 or total_requests % 100 == 0:
            log(
                "BaSiC tile-slab cache "
                f"{event}: requests={total_requests}, hits={cache_stats['hits']}, "
                f"misses={cache_stats['misses']}, disk_hits={cache_stats['disk_hits']}, "
                f"evictions={cache_stats['evictions']}, "
                f"entries={len(tile_cache)}/{tile_cache_size}, "
                f"max={format_bytes(tile_cache_max_bytes) if tile_cache_max_bytes is not None else 'none'}, "
                f"current={format_bytes(cache_stats['current_bytes'])}, "
                f"loaded_total={format_bytes(cache_stats['loaded_bytes'])}, "
                f"key={key}"
            )

    def dataset_serialize_zarr_backed_sim(sim):
        info = original_serialize(sim)
        if dataset_info_key is not None:
            info[dataset_info_key] = sim.attrs.get(dataset_info_key)
        if cache_key_attr is not None:
            info["_basic_cache_key"] = sim.attrs.get(cache_key_attr)
        info["_basic_cache_spatial_shape"] = {
            dim: int(sim.sizes[dim])
            for dim in sim.dims
            if dim in {"z", "y", "x"}
        }
        return info

    def zarr_safe_sim_sel_coords(sim, sel_dict):
        if not sel_dict:
            return sim

        selected = sim.sel(sel_dict)
        extra_attr_keys = dataset_attr_keys
        if cache_key_attr is not None:
            extra_attr_keys = dataset_attr_keys + (cache_key_attr,)
        _copy_selected_attrs(selected, sim, extra_attr_keys)
        selected.attrs["transforms"] = {}
        for transform_key, transform in sim.attrs["transforms"].items():
            for dim, values in sel_dict.items():
                if dim in transform.dims:
                    transform = transform.sel({dim: values})
            selected.attrs["transforms"][transform_key] = transform
        return selected

    def corrected_tile_cache_entry(
        info: dict[str, Any],
        dataset: str,
        overlap_bb: dict[str, Any],
    ) -> dict[str, Any]:
        base_cache_key = info.get("_basic_cache_key")
        if not base_cache_key:
            raise ValueError(
                "Tile-level BaSiC cache requires a stable cache key attr; "
                "set cache_key_attr and attach it to each fusion sim"
            )

        spatial_shape = info.get("_basic_cache_spatial_shape") or {}
        missing_dims = [dim for dim in ("z", "y", "x") if dim not in spatial_shape]
        if missing_dims:
            raise ValueError(f"Tile-level BaSiC cache missing spatial shape for {missing_dims}")

        y_size = int(spatial_shape["y"])
        x_size = int(spatial_shape["x"])
        _validate_basic_correction_bounds(
            tuple(inverse_by_dataset[dataset].shape),
            0,
            0,
            y_size,
            x_size,
            message_prefix=error_prefix,
        )

        z_total = int(spatial_shape["z"])
        request_z0 = int(round((overlap_bb["origin"]["z"] - info["origin"]["z"]) / info["spacing"]["z"]))
        request_z1 = request_z0 + int(overlap_bb["shape"]["z"])
        slab_depth = max(1, int(tile_cache_z_chunk))
        slab_z0 = max(0, (request_z0 // slab_depth) * slab_depth)
        slab_z1 = min(z_total, ((request_z1 + slab_depth - 1) // slab_depth) * slab_depth)
        if slab_z0 >= slab_z1:
            raise ValueError(
                f"Invalid BaSiC tile cache z slab for {base_cache_key}: "
                f"request_z={request_z0}:{request_z1}, tile_z={z_total}"
            )
        cache_key = (
            f"{base_cache_key}::basic{flatfield_fingerprints[dataset]}::"
            f"z{slab_z0}:{slab_z1}"
        )

        with cache_lock:
            cached = tile_cache.get(cache_key)
            if cached is not None:
                tile_cache.move_to_end(cache_key)
                cache_stats["hits"] += 1
                log_basic_cache_summary(event="hit", key=cache_key)
                return cached

            cache_stats["misses"] += 1
            cached = (
                None
                if tile_cache_disk_dir is None
                else read_basic_disk_cache_entry(tile_cache_disk_dir, cache_key)
            )
            if cached is not None:
                cache_stats["disk_hits"] += 1
            else:
                tile_bb = {
                    "origin": {
                        "z": info["origin"]["z"] + slab_z0 * info["spacing"]["z"],
                        "y": info["origin"]["y"],
                        "x": info["origin"]["x"],
                    },
                    "shape": {"z": slab_z1 - slab_z0, "y": y_size, "x": x_size},
                }
                slab = original_deserialize(
                    info,
                    reconstruct_slice=True,
                    overlap_bb=tile_bb,
                    sim_coord_dict=None,
                )
                slab_data = si_utils._get_backend_data(slab)
                data_dtype = np.dtype(slab.dtype)
                reshape = [1] * slab_data.ndim
                reshape[slab.get_axis_num("y")] = y_size
                reshape[slab.get_axis_num("x")] = x_size
                corrected = _apply_basic_correction_gpu(
                    slab_data,
                    inverse_gpu[dataset],
                    reshape,
                    data_dtype,
                )
                cached = {
                    "data": cp.asnumpy(corrected),
                    "dims": tuple(slab.dims),
                    "offsets": {"z": slab_z0, "y": 0, "x": 0},
                }
                if tile_cache_disk_dir is not None:
                    write_basic_disk_cache_entry(tile_cache_disk_dir, cache_key, cached)
            cached_bytes = int(cached["data"].nbytes)
            cached["bytes"] = cached_bytes
            tile_cache[cache_key] = cached
            tile_cache.move_to_end(cache_key)
            cache_stats["loaded_bytes"] += cached_bytes
            cache_stats["current_bytes"] += cached_bytes
            while (
                len(tile_cache) > tile_cache_size
                or (
                    tile_cache_max_bytes is not None
                    and len(tile_cache) > 1
                    and cache_stats["current_bytes"] > tile_cache_max_bytes
                )
            ):
                _evicted_key, evicted = tile_cache.popitem(last=False)
                cache_stats["evictions"] += 1
                cache_stats["current_bytes"] -= int(evicted.get("bytes", 0))
            log_basic_cache_summary(event="miss-loaded", key=cache_key)
            return cached

    @profile
    def corrected_deserialize_zarr_backed_sim(
        info,
        reconstruct_slice=False,
        overlap_bb=None,
        sim_coord_dict=None,
    ):
        dataset = default_dataset if dataset_info_key is None else info.get(dataset_info_key)
        if reconstruct_slice and tile_cache_size > 0 and dataset is not None:
            if overlap_bb is None:
                raise ValueError("overlap_bb is required for BaSiC-corrected zarr reads")
            if dataset not in inverse_by_dataset:
                raise ValueError(f"No BaSiC flatfield configured for dataset {dataset!r}")
            cached = corrected_tile_cache_entry(info, dataset, overlap_bb)
            return _basic_cache_request_sim(cached, info, overlap_bb, si_utils)

        sim = original_deserialize(
            info,
            reconstruct_slice=reconstruct_slice,
            overlap_bb=overlap_bb,
            sim_coord_dict=sim_coord_dict,
        )
        if not reconstruct_slice:
            return sim
        if overlap_bb is None:
            raise ValueError("overlap_bb is required for BaSiC-corrected zarr reads")
        if dataset is None:
            return sim
        if dataset not in inverse_by_dataset:
            raise ValueError(f"No BaSiC flatfield configured for dataset {dataset!r}")

        y0 = int(round((overlap_bb["origin"]["y"] - info["origin"]["y"]) / info["spacing"]["y"]))
        x0 = int(round((overlap_bb["origin"]["x"] - info["origin"]["x"]) / info["spacing"]["x"]))
        y_size = int(sim.sizes["y"])
        x_size = int(sim.sizes["x"])
        _validate_basic_correction_bounds(
            tuple(inverse_by_dataset[dataset].shape),
            y0,
            x0,
            y_size,
            x_size,
            message_prefix=error_prefix,
        )

        state["corrected_slices"] += 1
        if state["corrected_slices"] <= 3 or state["corrected_slices"] % 10000 == 0:
            dataset_message = "" if dataset == default_dataset else f"dataset={dataset}, "
            log(
                f"{log_prefix} "
                f"{state['corrected_slices']}: "
                f"{dataset_message}"
                f"z={overlap_bb['origin'].get('z', 0):.3f}, "
                f"y={y0}:{y0 + y_size}, x={x0}:{x0 + x_size}, "
                f"dims={sim.dims}, shape={tuple(sim.shape)}"
            )

        data = si_utils._get_backend_data(sim)
        data_dtype = np.dtype(sim.dtype)
        reshape = [1] * data.ndim
        reshape[sim.get_axis_num("y")] = y_size
        reshape[sim.get_axis_num("x")] = x_size

        correction_gpu = inverse_gpu[dataset][y0 : y0 + y_size, x0 : x0 + x_size]
        corrected = _apply_basic_correction_gpu(data, correction_gpu, reshape, data_dtype)
        return sim.copy(data=corrected)

    if dataset_info_key is not None or cache_key_attr is not None:
        si_utils.serialize_zarr_backed_sim = dataset_serialize_zarr_backed_sim
    si_utils.deserialize_zarr_backed_sim = corrected_deserialize_zarr_backed_sim
    si_utils.sim_sel_coords = zarr_safe_sim_sel_coords
    try:
        yield
    finally:
        si_utils.serialize_zarr_backed_sim = original_serialize
        si_utils.deserialize_zarr_backed_sim = original_deserialize
        si_utils.sim_sel_coords = original_sim_sel_coords


def close_stores(stores: list[Any]) -> None:
    for store in stores:
        close = getattr(store, "close", None)
        if close is not None:
            close()


def registration_source_channel(
    selected_channels: tuple[int, ...] | None,
    *,
    reg_channel_index: int,
    n_channels: int,
) -> int:
    if selected_channels is None:
        if reg_channel_index < 0 or reg_channel_index >= n_channels:
            raise ValueError(f"registration source channel index must be in [0, {n_channels - 1}]")
        return reg_channel_index

    if reg_channel_index < 0 or reg_channel_index >= len(selected_channels):
        raise ValueError(f"registration source channel index must be in [0, {len(selected_channels) - 1}]")
    return selected_channels[reg_channel_index]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "dims") and hasattr(value, "coords") and hasattr(value, "data"):
        return {
            "dims": list(value.dims),
            "coords": {str(name): json_safe(value.coords[name].values) for name in value.coords},
            "data": json_safe(value.data),
        }
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def _iter_edge_residual_values(edge_residuals: Any):
    if isinstance(edge_residuals, dict) and set(edge_residuals) == {"0"}:
        edge_residuals = edge_residuals["0"]
    if not isinstance(edge_residuals, dict):
        return
    for edge_key, residual in edge_residuals.items():
        if isinstance(residual, dict):
            for nested_key, nested_residual in residual.items():
                yield f"{edge_key}:{nested_key}", nested_residual
            continue
        yield str(edge_key), residual


def registration_residual_warning_payload(registration_result: Any) -> dict[str, Any] | None:
    if not isinstance(registration_result, dict):
        return None
    edge_residuals = (
        registration_result.get("groupwise_resolution", {})
        .get("metrics", {})
        .get("edge_residuals")
    )
    residual_records = []
    for edge_key, residual in _iter_edge_residual_values(edge_residuals):
        residual_array = np.asarray(residual, dtype=np.float64)
        residual_um = float(np.linalg.norm(residual_array)) if residual_array.ndim else float(residual_array)
        if np.isfinite(residual_um):
            residual_records.append({"edge": edge_key, "residual_um": residual_um})
    if not residual_records:
        return None

    worst = max(residual_records, key=lambda item: float(item["residual_um"]))
    max_residual_um = float(worst["residual_um"])
    if max_residual_um > 10.0:
        level = "huge_warning"
    elif max_residual_um > 5.0:
        level = "warning"
    else:
        level = "ok"
    return {
        "level": level,
        "warning_threshold_um": 5.0,
        "huge_warning_threshold_um": 10.0,
        "max_residual_um": max_residual_um,
        "max_residual_edge": worst["edge"],
        "edge_count": len(residual_records),
    }


def residual_warning_from_records(residual_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not residual_records:
        return None
    worst = max(residual_records, key=lambda item: float(item["residual_um"]))
    max_residual_um = float(worst["residual_um"])
    if max_residual_um > 10.0:
        level = "huge_warning"
    elif max_residual_um > 5.0:
        level = "warning"
    else:
        level = "ok"
    return {
        "level": level,
        "warning_threshold_um": 5.0,
        "huge_warning_threshold_um": 10.0,
        "max_residual_um": max_residual_um,
        "max_residual_edge": worst["edge"],
        "edge_count": len(residual_records),
    }


def robust_boundary_residual_warning_payload(
    robust_refinement: RobustBoundaryRefinementResult,
    spacing_um: dict[str, float],
) -> dict[str, Any] | None:
    spacing = np.asarray(
        [spacing_um["z"], spacing_um["y"], spacing_um["x"]],
        dtype=np.float64,
    )
    axis_to_index = {"z": 0, "y": 1, "x": 2}
    if robust_refinement.reference_geometry is not None:
        warning_axes = {"z", "y", "x"} - set(
            robust_refinement.reference_geometry.fixed_axes
        )
    else:
        warning_axes = {"z", "y", "x"}
    warning_indices = [
        axis_to_index[axis]
        for axis in ("z", "y", "x")
        if axis in warning_axes
    ]
    if not warning_indices:
        return None

    residual_records = []
    for constraint in robust_refinement.constraints:
        if not constraint.accepted or constraint.final_residual_zyx is None:
            continue
        residual_px = np.asarray(constraint.final_residual_zyx, dtype=np.float64)
        residual_um = float(np.linalg.norm((residual_px * spacing)[warning_indices]))
        if np.isfinite(residual_um):
            residual_records.append(
                {
                    "edge": f"{constraint.pair}",
                    "residual_um": residual_um,
                    "axes": [
                        axis for axis in ("z", "y", "x") if axis in warning_axes
                    ],
                }
            )
    warning = residual_warning_from_records(residual_records)
    if warning is not None:
        warning["axes"] = [
            axis for axis in ("z", "y", "x") if axis in warning_axes
        ]
    return warning


def warn_if_registration_residual_is_high(warning: dict[str, Any] | None) -> None:
    if warning is None or warning["level"] == "ok":
        return
    label = "HUGE WARNING" if warning["level"] == "huge_warning" else "WARNING"
    log(
        f"{label}: final global optimization residual is {warning['max_residual_um']:.3f} um "
        f"on edge {warning['max_residual_edge']} "
        f"(warning>{warning['warning_threshold_um']:.1f} um, "
        f"huge>{warning['huge_warning_threshold_um']:.1f} um)"
    )


def registration_metrics_payload(registration_result: Any) -> dict[str, Any]:
    if not isinstance(registration_result, dict):
        return {}

    pairwise = registration_result.get("pairwise_registration", {})
    graph = pairwise.get("graph")
    edges = []
    if graph is not None:
        for source, target, attrs in graph.edges(data=True):
            edges.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "source_tile": attrs.get("source_tile"),
                    "target_tile": attrs.get("target_tile"),
                    "quality": json_safe(attrs.get("quality")),
                    "attrs": json_safe(attrs),
                }
            )

    payload = {
        "pairwise_registration": {
            "edges": edges,
            "metrics": json_safe(pairwise.get("metrics", {})),
        },
        "groupwise_resolution": {
            "metrics": json_safe(registration_result.get("groupwise_resolution", {}).get("metrics", {})),
        },
    }
    warning = registration_residual_warning_payload(registration_result)
    if warning is not None:
        payload["global_optimization_residual_warning"] = warning
    return payload


def has_registration_overlap(
    msims: list[Any],
    pair: tuple[int, int],
    *,
    transform_key: str | None,
    reg_res_level: int | None,
    registration_binning: dict[str, int] | None,
    overlap_tolerance: Any,
) -> bool:
    from multiview_stitcher import msi_utils, mv_graph, spatial_image_utils

    sim_pair = []
    for tile_index in pair:
        scale_key = f"scale{reg_res_level}" if reg_res_level is not None else "scale0"
        if scale_key not in msi_utils.get_sorted_scale_keys(msims[tile_index]):
            raise ValueError(f"Registration scale {scale_key} does not exist for tile {tile_index}")
        sim_pair.append(msi_utils.get_sim_from_msim(msims[tile_index], scale=scale_key))

    if registration_binning is not None and max(registration_binning.values()) > 1:
        sim_pair = [
            sim.coarsen(registration_binning, boundary="trim").mean().astype(sim.dtype)
            for sim in sim_pair
        ]

    stack_props = [
        spatial_image_utils.get_stack_properties_from_sim(
            sim,
            transform_key=transform_key,
        )
        for sim in sim_pair
    ]
    if overlap_tolerance is not None:
        stack_props = [
            spatial_image_utils.extend_stack_props(
                props,
                extend_by=overlap_tolerance,
            )
            for props in stack_props
        ]
    _, intersection = mv_graph.get_overlap_between_pair_of_stack_props(
        stack_props1=stack_props[0],
        stack_props2=stack_props[1],
    )
    return intersection is not None


@contextmanager
def batched_pairwise_registration(*, dask_num_workers: int | None = None):
    from dask import compute
    from multiview_stitcher import msi_utils, registration

    original_compute_pairwise_registrations = registration.compute_pairwise_registrations

    def compute_pairwise_registrations_batched(
        msims,
        g_reg,
        n_parallel_pairwise_regs=None,
        **register_kwargs,
    ):
        g_reg_computed = g_reg.copy()
        edges = [tuple(sorted((edge[0], edge[1]))) for edge in g_reg.edges]
        overlap_kwargs = {
            "transform_key": register_kwargs.get("transform_key"),
            "reg_res_level": register_kwargs.get("reg_res_level"),
            "registration_binning": register_kwargs.get("registration_binning"),
            "overlap_tolerance": register_kwargs.get("overlap_tolerance"),
        }
        valid_edges = [
            edge
            for edge in edges
            if has_registration_overlap(msims, edge, **overlap_kwargs)
        ]
        skipped_edges = sorted(set(edges) - set(valid_edges))
        if skipped_edges:
            log(
                "Skipping explicit registration pairs without multiview-stitcher overlap "
                f"at the selected registration scale: {skipped_edges}"
            )
            for edge in skipped_edges:
                if g_reg_computed.has_edge(*edge):
                    g_reg_computed.remove_edge(*edge)
        if not valid_edges:
            raise ValueError("No explicit registration pairs overlap at the selected registration scale")
        edges = valid_edges
        if n_parallel_pairwise_regs is None:
            n_parallel_pairwise_regs = 1 if msi_utils.get_ndim(msims[0]) == 3 else len(edges)
        batch_size = len(edges) if n_parallel_pairwise_regs <= 0 else n_parallel_pairwise_regs
        batches = [
            edges[start : start + batch_size]
            for start in range(0, len(edges), batch_size)
        ]

        params = []
        for batch_index, batch_edges in enumerate(batches, start=1):
            log(
                "Building pairwise registration batch "
                f"{batch_index}/{len(batches)} with {len(batch_edges)} edge(s): {batch_edges}"
            )
            params_xds = [
                registration.register_pair_of_msims_over_time(
                    msims[pair[0]],
                    msims[pair[1]],
                    **register_kwargs,
                )
                for pair in batch_edges
            ]
            log(f"Computing pairwise registration batch {batch_index}/{len(batches)}")
            if dask_num_workers is None:
                params += compute(params_xds)[0]
            else:
                params += compute(params_xds, scheduler="threads", num_workers=dask_num_workers)[0]
            log(f"Finished pairwise registration batch {batch_index}/{len(batches)}")

        for index, pair in enumerate(edges):
            g_reg_computed.edges[pair]["transform"] = params[index]["transform"]
            g_reg_computed.edges[pair]["quality"] = params[index]["quality"]
            g_reg_computed.edges[pair]["bbox"] = params[index]["bbox"]

        return g_reg_computed

    registration.compute_pairwise_registrations = compute_pairwise_registrations_batched
    try:
        yield
    finally:
        registration.compute_pairwise_registrations = original_compute_pairwise_registrations


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def registration_metrics_res_level(msims: list[Any], preferred: int | None) -> int | None:
    if preferred is None:
        return None

    from multiview_stitcher import msi_utils

    scale_keys = msi_utils.get_sorted_scale_keys(msims[0])
    max_level = len(scale_keys) - 1
    if preferred > max_level:
        log(
            f"Registration metrics requested resolution level {preferred}, "
            f"but only {len(scale_keys)} scale(s) are available; using {max_level}"
        )
        return max_level
    return preferred


def selected_metric_sim(msim: Any, scale_key: str, metric_channel: str | None) -> Any:
    from multiview_stitcher import msi_utils, spatial_image_utils

    sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
    selection = {}
    if "t" in sim.dims:
        selection["t"] = sim.coords["t"].values[0]
    if "c" in sim.dims:
        selection["c"] = (
            sim.coords["c"].values[0] if metric_channel is None else metric_channel
        )
    if selection:
        sim = spatial_image_utils.sim_sel_coords(sim, selection)
    return sim


def cupy_normalized_cross_correlation(
    fixed_sim: Any,
    moving_sim: Any,
    halfspace_eqs: Any,
) -> float:
    import numpy as np
    from multiview_stitcher import mv_graph

    import cupy as cp

    fixed_data = (
        fixed_sim.data.compute() if hasattr(fixed_sim.data, "compute") else fixed_sim.data
    )
    moving_data = (
        moving_sim.data.compute() if hasattr(moving_sim.data, "compute") else moving_sim.data
    )
    fixed = cp.asarray(np.asarray(fixed_data, dtype=np.float32))
    moving = cp.asarray(np.asarray(moving_data, dtype=np.float32))
    halfspace_mask = cp.asarray(mv_graph.get_mask_from_halfspace(fixed_sim, halfspace_eqs))
    finite_mask = cp.isfinite(fixed) & cp.isfinite(moving) & halfspace_mask
    n_values = int(cp.count_nonzero(finite_mask).get())
    if n_values < 2:
        return float("nan")

    fixed_values = fixed[finite_mask]
    moving_values = moving[finite_mask]
    fixed_values = fixed_values - cp.mean(fixed_values)
    moving_values = moving_values - cp.mean(moving_values)
    denominator = cp.sqrt(
        cp.sum(fixed_values * fixed_values) * cp.sum(moving_values * moving_values)
    )
    if float(denominator.get()) == 0.0:
        return float("nan")
    return float((cp.sum(fixed_values * moving_values) / denominator).get())


def cupy_tile_pair_image_metrics_for_plot(
    msims: list[Any],
    *,
    base_transform_key: str,
    query_transform_keys: list[str],
    metric_channel: str | None,
    input_res_level: int | None,
    registration_pairs: list[tuple[int, int]] | None,
) -> dict[str, Any]:
    import numpy as np
    from multiview_stitcher import (
        metrics,
        mv_graph,
        spatial_image_utils,
        transformation,
    )

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(
        selected_metric_sim(msims[0], "scale0", metric_channel)
    )
    ndim = len(spatial_dims)
    scale_key = f"scale{input_res_level or 0}"
    sims_t0 = [
        selected_metric_sim(msim, scale_key, metric_channel)
        for msim in msims
    ]
    g_metrics = metrics._build_metrics_graph(
        msims,
        sims_t0,
        base_transform_key,
        query_transform_keys,
        max_tolerance=None,
        bidirectional=False,
    )
    if registration_pairs is not None:
        requested_pairs = {tuple(sorted(pair)) for pair in registration_pairs}
        for edge in list(g_metrics.edges()):
            if tuple(sorted(edge)) not in requested_pairs:
                g_metrics.remove_edge(*edge)

    log(
        "Computing GPU tile-pair NCC metrics for registration plots "
        f"(pairs={len(g_metrics.edges())}, scale={scale_key}, channel={metric_channel})"
    )
    result_pairs: dict[tuple[int, int], dict[str, dict[str, float]]] = {}
    bboxes: dict[tuple[int, int], Any] = {}
    weighted_values: dict[str, list[tuple[float, float]]] = {
        transform_key: [] for transform_key in query_transform_keys
    }

    for edge_index, (fixed_idx, moving_idx) in enumerate(g_metrics.edges(), start=1):
        edge_data = g_metrics.edges[(fixed_idx, moving_idx)]
        comparison_bbox = edge_data["comparison_bbox"]
        bboxes[(fixed_idx, moving_idx)] = comparison_bbox
        if comparison_bbox is None:
            result_pairs[(fixed_idx, moving_idx)] = {
                transform_key: {"ncc": float("nan")} for transform_key in query_transform_keys
            }
            log(
                f"GPU metrics edge {edge_index}/{len(g_metrics.edges())} "
                f"({fixed_idx}, {moving_idx}): empty comparison bbox"
            )
            continue

        sim_fixed = spatial_image_utils.ensure_dask_backed_dataarray(sims_t0[fixed_idx])
        sim_moving = spatial_image_utils.ensure_dask_backed_dataarray(sims_t0[moving_idx])
        spacing = spatial_image_utils.get_spacing_from_sim(sim_fixed)
        lower_intrinsic = comparison_bbox["lower"]
        upper_intrinsic = comparison_bbox["upper"]
        shape = {
            dim: max(
                1,
                int(
                    np.floor(
                        (upper_intrinsic[idim] - lower_intrinsic[idim])
                        / spacing[dim]
                        + 1
                    )
                ),
            )
            for idim, dim in enumerate(spatial_dims)
        }
        output_stack_properties = {
            "origin": {
                dim: float(lower_intrinsic[idim]) for idim, dim in enumerate(spatial_dims)
            },
            "spacing": {dim: float(spacing[dim]) for dim in spatial_dims},
            "shape": {dim: int(shape[dim]) for dim in spatial_dims},
        }
        fixed_spacing = spatial_image_utils.get_spacing_from_sim(
            sims_t0[fixed_idx],
            asarray=True,
        )
        intersection_halfspace = mv_graph.expand_halfspace(
            edge_data["intersection_halfspace"],
            distance=1e-3 * np.min(fixed_spacing),
        )
        sim_fixed_t = transformation.transform_sim(
            sim_fixed.astype(np.float32),
            p=np.eye(ndim + 1),
            output_stack_properties=output_stack_properties,
            mode="constant",
            cval=np.nan,
            order=1,
        )
        result_pairs[(fixed_idx, moving_idx)] = {}
        edge_nccs = []
        for transform_key in query_transform_keys:
            sim_moving_t = transformation.transform_sim(
                sim_moving.astype(np.float32),
                p=edge_data["transforms"][transform_key],
                output_stack_properties=output_stack_properties,
                mode="constant",
                cval=np.nan,
                order=1,
            )
            ncc = cupy_normalized_cross_correlation(
                sim_fixed_t,
                sim_moving_t,
                intersection_halfspace.halfspaces,
            )
            result_pairs[(fixed_idx, moving_idx)][transform_key] = {"ncc": ncc}
            if np.isfinite(ncc):
                weighted_values[transform_key].append((ncc, float(edge_data["vol"])))
            edge_nccs.append(f"{transform_key}:ncc={ncc:.4f}")
        log(
            f"GPU metrics edge {edge_index}/{len(g_metrics.edges())} "
            f"({fixed_idx}, {moving_idx}), shape={tuple(shape[dim] for dim in spatial_dims)}: "
            + ", ".join(edge_nccs)
        )

    summary = {}
    for transform_key, values in weighted_values.items():
        finite_values = [(value, weight) for value, weight in values if weight > 0]
        if finite_values:
            total_weight = sum(weight for _, weight in finite_values)
            weighted_mean = sum(value * weight for value, weight in finite_values) / total_weight
        else:
            weighted_mean = float("nan")
        summary[transform_key] = {"ncc": weighted_mean}

    return {"pairs": result_pairs, "bboxes": bboxes, "summary": summary}


def save_registration_metric_position_plots(
    msims: list[Any],
    output_dir: Path,
    *,
    reg_channel_label: str,
    reg_res_level: int | None,
    n_parallel_pairwise_regs: int | None,
    registration_pairs: list[tuple[int, int]] | None,
) -> None:
    del n_parallel_pairwise_regs

    from multiview_stitcher import vis_utils
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    query_transform_keys = [REGISTERED_TRANSFORM_KEY]
    metrics_res_level = registration_metrics_res_level(msims, reg_res_level)
    log(
        "Computing tile-pair image metrics for registration plots "
        f"(channel={reg_channel_label}, res_level={metrics_res_level if metrics_res_level is not None else 'auto'})"
    )
    with heartbeat("GPU tile-pair registration metric computation"):
        reg_metrics_result = cupy_tile_pair_image_metrics_for_plot(
            msims,
            base_transform_key=TRANSFORM_KEY,
            query_transform_keys=query_transform_keys,
            metric_channel=reg_channel_label,
            input_res_level=metrics_res_level,
            registration_pairs=registration_pairs,
        )
    log("Rendering registration metric positional plots")
    with heartbeat("registration metric plot rendering"):
        plots = vis_utils.plot_tile_pair_image_metrics(
            msims,
            reg_metrics_result,
            base_transform_key=TRANSFORM_KEY,
            query_transform_keys=query_transform_keys,
            show_plot_positions=True,
        )
    for transform_key, (fig, _) in plots.items():
        plot_path = output_dir / f"tile_pair_image_metrics_positions.{safe_filename(transform_key)}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        log(f"Wrote registration metric positional plot: {plot_path}")
    plt.close("all")


def save_registration_params(
    registration_result: Any,
    tiles: list[TileMetadata],
    output: Path,
    *,
    robust_refinement: RobustBoundaryRefinementResult | None = None,
    pre_stitch_tile_rotation: PreStitchTileRotation | None = None,
) -> None:
    if output.exists():
        raise FileExistsError(f"{output} already exists; move it before rerunning")

    if robust_refinement is not None:
        params = robust_refinement.params
        residual_warning = robust_boundary_residual_warning_payload(robust_refinement, tiles[0].spacing)
    else:
        params = registration_result["params"] if isinstance(registration_result, dict) else registration_result
        residual_warning = registration_residual_warning_payload(registration_result)
    warn_if_registration_residual_is_high(residual_warning)
    records = []
    for tile, param in zip(tiles, params):
        records.append(
            {
                "tile": tile.path.name,
                "source_view": tile.source_view,
                "stage_translation_um": tile.translation,
                "stage_scale_um": tile_stage_scale(tile),
                "registered_affine": {
                    "dims": list(param.dims),
                    "coords": {
                        name: param.coords[name].values.tolist()
                        for name in param.coords
                    },
                    "matrix": param.data.tolist(),
                },
            }
        )

    payload = {
        "input_dir": str(tiles[0].path.parent),
        "metadata_transform_key": TRANSFORM_KEY,
        "registered_transform_key": REGISTERED_TRANSFORM_KEY,
        "spacing_um": tiles[0].spacing,
        "metrics": registration_metrics_payload(registration_result),
        "tiles": records,
    }
    if residual_warning is not None:
        payload["metrics"]["final_registration_residual_warning"] = residual_warning
    if robust_refinement is not None:
        payload["robust_boundary_refinement"] = {
            "mode": "robust-boundary",
            "output_dir": str(robust_refinement.output_dir),
            "anchor_tile": robust_refinement.anchor_tile,
            "summary": robust_refinement.summary,
            "final_residual_warning": residual_warning,
            "corrections_zyx_px": robust_refinement.corrections_zyx,
        }
        if robust_refinement.reference_geometry is not None:
            payload["reference_geometry_constraint"] = {
                "mode": robust_refinement.reference_geometry.mode,
                "reference_registration_input": robust_refinement.reference_geometry.reference_input,
                "fixed_axes": list(robust_refinement.reference_geometry.fixed_axes),
                "shared_geometry_tracks": list(robust_refinement.reference_geometry.shared_geometry_tracks),
                "drift_from_reference_um": robust_refinement.reference_geometry.drift_from_reference_um,
                "constraint_counts_by_track": robust_refinement.reference_geometry.constraint_counts_by_track,
                "reference_prior_weights_zyx": robust_refinement.reference_geometry.reference_prior_weights_zyx,
                "residual_reject_axes": (
                    list(robust_refinement.reference_geometry.residual_reject_axes)
                    if robust_refinement.reference_geometry.residual_reject_axes is not None
                    else None
                ),
            }
    if isinstance(registration_result, dict) and "hierarchical_coarse_registration" in registration_result:
        payload["hierarchical_coarse_registration"] = registration_result["hierarchical_coarse_registration"]
    if pre_stitch_tile_rotation is not None:
        payload["pre_stitch_tile_rotation"] = pre_stitch_tile_rotation_payload(pre_stitch_tile_rotation)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def load_registration_params(path: Path, tiles: list[TileMetadata]):
    import xarray as xr

    payload = json.loads(path.read_text())
    records_by_tile = {record["tile"]: record for record in payload["tiles"]}
    params = []
    missing = [tile.path.name for tile in tiles if tile.path.name not in records_by_tile]
    if missing:
        raise ValueError(f"{path} is missing registration records for: {missing}")

    for tile in tiles:
        affine = records_by_tile[tile.path.name]["registered_affine"]
        params.append(
            xr.DataArray(
                affine["matrix"],
                dims=affine["dims"],
                coords=affine["coords"],
            )
        )
    return params


def spatial_affine_param(param):
    indexer = {
        dim: 0
        for dim in param.dims
        if dim not in {"x_in", "x_out"} and param.sizes[dim] == 1
    }
    if indexer:
        return param.isel(indexer, drop=True)
    return param


def output_chunksize_arg(value: tuple[int, int, int] | None) -> dict[str, int] | None:
    if value is None:
        return None
    z, y, x = value
    return {"z": z, "y": y, "x": x}


def fusion_output_spacing(base_spacing: dict[str, float], fusion_level: int) -> dict[str, float]:
    if fusion_level < 0:
        raise ValueError("--fusion-level must be non-negative")
    factor = 2**fusion_level
    return {dim: float(value) * factor for dim, value in base_spacing.items()}


def compression_level(value: float) -> float:
    return value / 100 if value > 1 else value


def zarr_v3_array_creation_kwargs(dims: tuple[str, ...]) -> dict[str, Any]:
    from zarr.codecs import BytesCodec, ZstdCodec

    return {
        "dimension_names": dims,
        "codecs": [BytesCodec(endian="little"), ZstdCodec(level=ZSTD_LEVEL)],
    }


@contextmanager
def single_scale_fusion_output():
    from multiview_stitcher import ngff_utils
    from multiview_stitcher import spatial_image_utils as si_utils

    original_write = ngff_utils.write_sim_to_ome_zarr

    def write_single_scale_ome_zarr(sim, *args, **kwargs):
        import numpy as np

        sdims = si_utils.get_spatial_dims_from_sim(sim)
        kwargs["downscale_factors_per_spatial_dim"] = {dim: 10**9 for dim in sdims}
        try:
            return original_write(sim, *args, **kwargs)
        except np.exceptions.AxisError as exc:
            log(
                "Skipping package OMERO channel-window metadata after single-scale "
                f"temporary fusion write: {exc}"
            )
            return sim

    ngff_utils.write_sim_to_ome_zarr = write_single_scale_ome_zarr
    try:
        yield
    finally:
        ngff_utils.write_sim_to_ome_zarr = original_write


@contextmanager
def spatial_chunks_for_package_pyramid():
    from multiview_stitcher import ngff_utils

    original_write = ngff_utils.write_sim_to_ome_zarr
    original_write_level = ngff_utils.write_and_return_downsampled_sim

    def write_ome_zarr_skip_stale_omero_axes(sim, *args, **kwargs):
        import numpy as np

        try:
            return original_write(sim, *args, **kwargs)
        except np.exceptions.AxisError as exc:
            output_zarr_url = kwargs.get("output_zarr_url") or args[0]
            repair_ome_metadata_axes(Path(output_zarr_url))
            log(f"Skipping package OMERO channel-window metadata after OME-Zarr write: {exc}")
            return sim

    def write_level_with_spatial_chunks(array, dims, output_zarr_array_url, chunksizes, *args, **kwargs):
        ndim = len(array.shape)
        if len(dims) > ndim:
            dims = tuple(dims[-ndim:])
        if len(chunksizes) > ndim:
            chunksizes = tuple(chunksizes[-ndim:])
        zarr_kwargs = kwargs.get("zarr_array_creation_kwargs")
        if zarr_kwargs is not None and len(zarr_kwargs.get("dimension_names", ())) != ndim:
            kwargs["zarr_array_creation_kwargs"] = {
                **zarr_kwargs,
                "dimension_names": tuple(dims),
            }
        return original_write_level(
            array,
            dims,
            output_zarr_array_url,
            chunksizes,
            *args,
            **kwargs,
        )

    ngff_utils.write_sim_to_ome_zarr = write_ome_zarr_skip_stale_omero_axes
    ngff_utils.write_and_return_downsampled_sim = write_level_with_spatial_chunks
    try:
        yield
    finally:
        ngff_utils.write_sim_to_ome_zarr = original_write
        ngff_utils.write_and_return_downsampled_sim = original_write_level


def jpegxr_plane_chunk_shape(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != len(chunks):
        raise ValueError(f"Shape/chunk dimensionality mismatch: shape={shape}, chunks={chunks}")
    return tuple(1 if axis < len(shape) - 2 else min(chunks[axis], shape[axis]) for axis in range(len(shape)))


def chunk_slices(shape: tuple[int, ...], chunks: tuple[int, ...]):
    ranges = [
        range(0, int(size), int(chunk_size))
        for size, chunk_size in zip(shape, chunks, strict=True)
    ]
    import itertools

    for starts in itertools.product(*ranges):
        yield tuple(
            slice(start, min(start + chunk_size, size))
            for start, size, chunk_size in zip(starts, shape, chunks, strict=True)
        )


def chunk_count(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    count = 1
    for size, chunk_size in zip(shape, chunks, strict=True):
        count *= (int(size) + int(chunk_size) - 1) // int(chunk_size)
    return count


def pyramid_relative_factors(shape: tuple[int, ...], dims: tuple[str, ...]) -> dict[str, int]:
    factors = {}
    for dim, size in zip(dims, shape, strict=True):
        if dim in {"z", "y", "x"} and int(size) // 2 > 100:
            factors[dim] = 2
        else:
            factors[dim] = 1
    return factors


def downsample_mean_preserve_dtype(array, *, dims: tuple[str, ...], factors: dict[str, int]):
    import dask.array as da
    import numpy as np

    def mean_dtype(block, **kwargs):
        return np.mean(block, **kwargs).astype(block.dtype)

    axes = {
        axis: factors[dim]
        for axis, dim in enumerate(dims)
        if factors[dim] > 1
    }
    return da.coarsen(mean_dtype, array, axes=axes, trim_excess=True)


def level_coordinate_transformations(
    base_transforms: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    abs_factors: dict[str, int],
) -> list[dict[str, Any]]:
    transforms = copy.deepcopy(base_transforms)
    for transform in transforms:
        if transform.get("type") != "scale":
            continue
        transform["scale"] = [
            float(value) * abs_factors.get(axis["name"], 1)
            for value, axis in zip(transform["scale"], axes, strict=True)
        ]
    return transforms


def write_jpegxr_level(
    array,
    destination: Path,
    *,
    dataset_path: str,
    dimension_names: tuple[str, ...],
    source_chunks: tuple[int, ...],
    jpegxr_level: float,
) -> None:
    import numpy as np
    import zarr

    shape = tuple(int(size) for size in array.shape)
    output_chunks = jpegxr_plane_chunk_shape(shape, source_chunks)
    destination_array_path = destination / dataset_path
    destination_array_path.parent.mkdir(parents=True, exist_ok=True)
    destination_array = zarr.open(
        str(destination_array_path),
        mode="w",
        shape=shape,
        chunks=output_chunks,
        dtype=array.dtype,
        zarr_format=3,
        dimension_names=dimension_names,
        codecs=[JpegxrZarrV3Codec(level=jpegxr_level)],
    )

    n_chunks = chunk_count(shape, output_chunks)
    log(
        f"JPEG-XR recompress level {dataset_path}: "
        f"shape={shape}, source_chunks={source_chunks}, output_chunks={output_chunks}, "
        f"chunks={n_chunks}"
    )
    for chunk_index, selection in enumerate(chunk_slices(shape, output_chunks), start=1):
        destination_array[selection] = np.asarray(array[selection])
        if chunk_index <= 3 or chunk_index % 1000 == 0 or chunk_index == n_chunks:
            log(
                f"JPEG-XR recompress level {dataset_path}: "
                f"{chunk_index}/{n_chunks} chunk(s)"
            )


def block_reduce_mean_gpu(block: np.ndarray, factors: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    import cupy as cp

    block_gpu = cp.asarray(block)
    reshape = []
    mean_axes = []
    reshaped_axis = 0
    for size, factor in zip(block_gpu.shape, factors, strict=True):
        if factor > 1:
            if int(size) % factor != 0:
                raise ValueError(f"Block shape {block_gpu.shape} is not divisible by factors {factors}")
            reshape.extend([int(size) // factor, factor])
            mean_axes.append(reshaped_axis + 1)
            reshaped_axis += 2
        else:
            reshape.append(int(size))
            reshaped_axis += 1
    reduced = block_gpu.reshape(tuple(reshape))
    if mean_axes:
        reduced = cp.mean(reduced, axis=tuple(mean_axes), dtype=cp.float32)
    return cp.asnumpy(reduced.astype(dtype, copy=False))


def write_zstd_downsampled_level(
    source_array: Any,
    destination: Path,
    *,
    dataset_path: str,
    dimension_names: tuple[str, ...],
    factors: dict[str, int],
) -> None:
    import zarr

    factor_tuple = tuple(int(factors[dim]) for dim in dimension_names)
    shape = tuple(
        int(size) // factor
        for size, factor in zip(source_array.shape, factor_tuple, strict=True)
    )
    output_chunks = tuple(
        min(int(chunk), int(size))
        for chunk, size in zip(source_array.chunks, shape, strict=True)
    )
    destination_array_path = destination / dataset_path
    destination_array_path.parent.mkdir(parents=True, exist_ok=True)
    destination_array = zarr.open(
        str(destination_array_path),
        mode="w",
        shape=shape,
        chunks=output_chunks,
        dtype=source_array.dtype,
        zarr_format=3,
        dimension_names=dimension_names,
        codecs=zarr_v3_array_creation_kwargs(dimension_names)["codecs"],
    )

    n_chunks = chunk_count(shape, output_chunks)
    log(
        f"Writing pyramid level {dataset_path} from completed Zarr level: "
        f"shape={shape}, factors={factors}, chunks={output_chunks}, chunks_total={n_chunks}"
    )
    for chunk_index, selection in enumerate(chunk_slices(shape, output_chunks), start=1):
        source_selection = tuple(
            slice(
                part.start * factor,
                part.stop * factor,
            )
            for part, factor in zip(selection, factor_tuple, strict=True)
        )
        destination_array[selection] = block_reduce_mean_gpu(
            np.asarray(source_array[source_selection]),
            factor_tuple,
            np.dtype(source_array.dtype),
        )
        if chunk_index <= 3 or chunk_index % 1000 == 0 or chunk_index == n_chunks:
            log(
                f"Writing pyramid level {dataset_path}: "
                f"{chunk_index}/{n_chunks} chunk(s)"
            )


def build_ome_zarr_pyramid_from_scale0(output: Path) -> None:
    import zarr

    group = zarr.open_group(str(output), mode="a", zarr_format=3)
    source_attrs = group.attrs.asdict()
    ome_attrs = source_attrs.get("ome") or {}
    multiscales = ome_attrs.get("multiscales") or []
    if not multiscales:
        raise ValueError(f"{output} does not contain OME multiscales metadata")

    datasets = multiscales[0].get("datasets") or []
    if not datasets:
        raise ValueError(f"{output} OME multiscales metadata does not list datasets")

    source_array = zarr.open_array(str(output / "0"), mode="r", zarr_format=3)
    dimension_names = tuple(source_array.metadata.dimension_names or ())
    if not dimension_names:
        dimension_names = tuple(f"dim_{axis}" for axis in range(len(source_array.shape)))
    current_shape = tuple(int(size) for size in source_array.shape)
    abs_factors = [{dim: 1 for dim in dimension_names}]
    level_factors = []

    while True:
        factors = pyramid_relative_factors(
            current_shape,
            dimension_names,
        )
        if not any(factor > 1 for factor in factors.values()):
            break
        level_factors.append(factors)
        abs_factors.append(
            {
                dim: abs_factors[-1][dim] * factors[dim]
                for dim in dimension_names
            }
        )
        current_shape = tuple(
            size // int(factors[dim])
            for size, dim in zip(current_shape, dimension_names, strict=True)
        )

    if not level_factors:
        log(f"Completed fusion output {output} does not need pyramid levels")
        return

    base_multiscale = copy.deepcopy(multiscales[0])
    axes = base_multiscale.get("axes") or [{"name": dim} for dim in dimension_names]
    base_transforms = datasets[0].get("coordinateTransformations") or []
    base_multiscale["datasets"] = [
        {
            "path": str(level_index),
            "coordinateTransformations": level_coordinate_transformations(
                base_transforms,
                axes,
                factors,
            ),
        }
        for level_index, factors in enumerate(abs_factors)
    ]
    final_ome_attrs = copy.deepcopy(ome_attrs)
    final_ome_attrs["multiscales"] = [base_multiscale]
    destination_attrs = copy.deepcopy(source_attrs)
    destination_attrs["ome"] = final_ome_attrs

    log(f"Building completed-fusion Zarr pyramid for {output}: levels={len(abs_factors)}")
    previous_array = source_array
    for level_index, factors in enumerate(level_factors, start=1):
        write_zstd_downsampled_level(
            previous_array,
            output,
            dataset_path=str(level_index),
            dimension_names=dimension_names,
            factors=factors,
        )
        previous_array = zarr.open_array(str(output / str(level_index)), mode="r", zarr_format=3)
    group.attrs.update(destination_attrs)
    repair_ome_metadata_axes(output)


def recompress_ome_zarr_to_jpegxr(
    source: Path,
    destination: Path,
    *,
    jpegxr_level: float,
) -> None:
    import zarr

    register_jpegxr_zarr_v3_codec()
    normalized_level = compression_level(jpegxr_level)
    source_group = zarr.open_group(str(source), mode="r", zarr_format=3)
    source_attrs = source_group.attrs.asdict()
    destination_group = zarr.open_group(str(destination), mode="w", zarr_format=3)

    ome_attrs = source_attrs.get("ome") or {}
    multiscales = ome_attrs.get("multiscales") or []
    if not multiscales:
        raise ValueError(f"{source} does not contain OME multiscales metadata")

    datasets = multiscales[0].get("datasets") or []
    if not datasets:
        raise ValueError(f"{source} OME multiscales metadata does not list datasets")

    import dask.array as da

    source_array = zarr.open_array(str(source / "0"), mode="r", zarr_format=3)
    dimension_names = tuple(source_array.metadata.dimension_names or ())
    if not dimension_names:
        dimension_names = tuple(f"dim_{axis}" for axis in range(len(source_array.shape)))
    source_chunks = tuple(int(size) for size in source_array.chunks)
    arrays = [da.from_zarr(str(source / "0"))]
    abs_factors = [{dim: 1 for dim in dimension_names}]

    while True:
        factors = pyramid_relative_factors(
            tuple(int(size) for size in arrays[-1].shape),
            dimension_names,
        )
        if not any(factor > 1 for factor in factors.values()):
            break
        arrays.append(
            downsample_mean_preserve_dtype(
                arrays[-1],
                dims=dimension_names,
                factors=factors,
            )
        )
        abs_factors.append(
            {
                dim: abs_factors[-1][dim] * factors[dim]
                for dim in dimension_names
            }
        )

    log(
        "Recompressing completed fusion pyramid to JPEG-XR "
        f"level={normalized_level:.3f}, levels={len(arrays)}"
    )

    base_multiscale = copy.deepcopy(multiscales[0])
    axes = base_multiscale.get("axes") or [{"name": dim} for dim in dimension_names]
    base_transforms = datasets[0].get("coordinateTransformations") or []
    base_multiscale["datasets"] = [
        {
            "path": str(level_index),
            "coordinateTransformations": level_coordinate_transformations(
                base_transforms,
                axes,
                factors,
            ),
        }
        for level_index, factors in enumerate(abs_factors)
    ]
    final_ome_attrs = copy.deepcopy(ome_attrs)
    final_ome_attrs["multiscales"] = [base_multiscale]
    destination_attrs = copy.deepcopy(source_attrs)
    destination_attrs["ome"] = final_ome_attrs
    destination_group.attrs.update(destination_attrs)

    for level_index, array in enumerate(arrays):
        write_jpegxr_level(
            array,
            destination,
            dataset_path=str(level_index),
            dimension_names=dimension_names,
            source_chunks=source_chunks,
            jpegxr_level=normalized_level,
        )


def ome_zarr_group_metadata_path(output: Path) -> Path:
    zarr_json_path = output / "zarr.json"
    if zarr_json_path.exists():
        return zarr_json_path
    return output / ".zattrs"


def ome_zarr_array_metadata_path(array_path: Path) -> Path:
    zarr_json_path = array_path / "zarr.json"
    if zarr_json_path.exists():
        return zarr_json_path
    return array_path / ".zarray"


def read_ome_zarr_group_metadata(output: Path) -> dict[str, Any]:
    metadata_path = ome_zarr_group_metadata_path(output)
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text() or "{}")
    if metadata_path.name == "zarr.json":
        return payload.get("attributes", {}).get("ome", {})
    return payload


def repair_ome_metadata_axes(output: Path) -> None:
    metadata_path = ome_zarr_group_metadata_path(output)
    if not metadata_path.exists() or metadata_path.name != "zarr.json":
        return

    payload = json.loads(metadata_path.read_text() or "{}")
    ome = payload.get("attributes", {}).get("ome") or {}
    multiscales = ome.get("multiscales") or []
    if not multiscales:
        return

    scale0_metadata_path = ome_zarr_array_metadata_path(output / "0")
    if not scale0_metadata_path.exists():
        return
    scale0 = json.loads(scale0_metadata_path.read_text() or "{}")
    dimension_names = scale0.get("dimension_names") or []
    if not dimension_names:
        return

    multiscale = multiscales[0]
    axes = multiscale.get("axes") or []
    if len(axes) == len(dimension_names):
        return

    names = set(dimension_names)
    axis_indices = [
        index
        for index, axis in enumerate(axes)
        if axis.get("name") in names
    ]
    if len(axis_indices) != len(dimension_names):
        return

    multiscale["axes"] = [axes[index] for index in axis_indices]
    for dataset in multiscale.get("datasets") or []:
        for transform in dataset.get("coordinateTransformations") or []:
            key = "scale" if "scale" in transform else "translation" if "translation" in transform else None
            if key is None or len(transform[key]) != len(axes):
                continue
            transform[key] = [transform[key][index] for index in axis_indices]

    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Repaired OME-Zarr axes metadata for {output}")


def ome_zarr_scale0_array_path(output: Path) -> Path:
    return ome_zarr_array_metadata_path(output / "0")


def ome_zarr_scale0_shape(output: Path) -> list[int]:
    return json.loads(ome_zarr_scale0_array_path(output).read_text())["shape"]


def ome_zarr_scale0_dimension_names(output: Path, ndim: int) -> list[str]:
    metadata = json.loads(ome_zarr_scale0_array_path(output).read_text())
    dimension_names = metadata.get("dimension_names") or []
    if dimension_names:
        return list(dimension_names)

    defaults = {
        2: ["y", "x"],
        3: ["z", "y", "x"],
        4: ["c", "z", "y", "x"],
        5: ["t", "c", "z", "y", "x"],
    }
    if ndim not in defaults:
        raise ValueError(f"Cannot infer dimension names for {output / '0'} with ndim={ndim}")
    return defaults[ndim]


def center_z_thumbnail_path(output: Path) -> Path:
    name = output.name
    if name.endswith(".ome.zarr"):
        name = f"{name.removesuffix('.ome.zarr')}.center-z.png"
    elif name.endswith(".zarr"):
        name = f"{name.removesuffix('.zarr')}.center-z.png"
    else:
        name = f"{name}.center-z.png"
    return output.with_name(name)


def scale_thumbnail_uint8(image) -> Any:
    import numpy as np

    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(np.asarray(image).shape, dtype=np.uint8)
    nonzero = finite[finite > 0]
    sample = nonzero if nonzero.size else finite
    low, high = np.percentile(sample, [1.0, 99.8])
    scaled = np.clip((image - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return np.asarray(np.rint(scaled * 255), dtype=np.uint8)


def write_center_z_thumbnail(output: Path, *, max_size: int = 2048) -> Path:
    import numpy as np
    from PIL import Image
    import zarr

    register_jpegxr_zarr_v3_codec()
    array = zarr.open(str(output / "0"), mode="r")
    dims = ome_zarr_scale0_dimension_names(output, len(array.shape))
    if "z" not in dims:
        raise ValueError(f"Cannot write center-z thumbnail for {output}: dimensions are {dims}")
    y_axis = dims.index("y")
    x_axis = dims.index("x")
    z_axis = dims.index("z")

    indexer = []
    for axis, size in enumerate(array.shape):
        if axis == z_axis:
            indexer.append(size // 2)
        elif axis in (y_axis, x_axis):
            indexer.append(slice(None))
        else:
            indexer.append(0)

    plane = np.asarray(array[tuple(indexer)])
    while plane.ndim > 2:
        plane = plane[0]
    thumbnail = Image.fromarray(scale_thumbnail_uint8(plane))
    thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    path = center_z_thumbnail_path(output)
    thumbnail.save(path)
    return path


def ome_zarr_scale0_chunk_status(output: Path) -> tuple[int, int]:
    metadata_path = ome_zarr_scale0_array_path(output)
    metadata = json.loads(metadata_path.read_text())
    expected = 1
    zarr_format = 3 if metadata_path.name == "zarr.json" else 2
    if metadata_path.name == "zarr.json":
        chunks = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    else:
        chunks = metadata["chunks"]
    for shape, chunksize in zip(metadata["shape"], chunks, strict=True):
        expected *= (int(shape) + int(chunksize) - 1) // int(chunksize)

    scale0 = output / "0"
    actual = 0
    for path in scale0.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {".zarray", ".zattrs", "zarr.json"}:
            continue
        relative_parts = path.relative_to(scale0).parts
        if zarr_format == 3:
            if len(relative_parts) > 1 and relative_parts[0] == "c" and all(
                part.isdigit() for part in relative_parts[1:]
            ):
                actual += 1
        elif all(part.isdigit() for part in relative_parts):
            actual += 1
    return actual, expected


def ome_zarr_has_multiscales_metadata(output: Path) -> bool:
    if (
        not ome_zarr_group_metadata_path(output).exists()
        or not ome_zarr_scale0_array_path(output).exists()
    ):
        return False
    payload = read_ome_zarr_group_metadata(output)
    multiscales = payload.get("multiscales") or []
    if not multiscales:
        return False
    axes = multiscales[0].get("axes") or []
    datasets = multiscales[0].get("datasets") or []
    if len(axes) != len(ome_zarr_scale0_shape(output)):
        return False
    return any(dataset.get("path") == "0" for dataset in datasets)


def raise_if_nonempty_output_dir(path: Path, label: str) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError(f"{label} path {path} already exists and is not a directory")
    if any(path.iterdir()):
        raise FileExistsError(f"{label} directory {path} is not empty; move it before rerunning")


def output_exists_error(output: Path) -> FileExistsError:
    if ome_zarr_has_multiscales_metadata(output):
        return FileExistsError(f"{output} already exists; move it before rerunning")
    if ome_zarr_scale0_array_path(output).exists():
        actual_chunks, expected_chunks = ome_zarr_scale0_chunk_status(output)
        return FileExistsError(
            f"{output} contains scale-0 zarr data without valid OME-Zarr metadata "
            f"({actual_chunks}/{expected_chunks} chunk files present). This may be a partial "
            "interrupted write; move it before rerunning."
        )
    return FileExistsError(
        f"{output} exists but does not contain OME-Zarr metadata or scale-0 data; "
        "move it before rerunning"
    )


def raise_if_output_exists(output: Path) -> None:
    if output.exists():
        raise output_exists_error(output)


def channels_to_fuse_for_tiles(
    tiles: list[TileMetadata],
    selected_channels: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if selected_channels is not None:
        n_channels = tile_channel_count(tiles[0])
        invalid = [index for index in selected_channels if index < 0 or index >= n_channels]
        if invalid:
            raise ValueError(f"Invalid channel indices for {n_channels} channels: {invalid}")
    return (
        tuple(range(tile_channel_count(tiles[0])))
        if selected_channels is None
        else selected_channels
    )


def should_run_registration(
    args: argparse.Namespace,
    registration_input: Path | None,
) -> bool:
    return args.register_only or registration_input is None


def should_refine_registration_input(
    args: argparse.Namespace,
    registration_input: Path | None,
) -> bool:
    return registration_input is not None and args.register_only


def should_run_coarse_registration(
    args: argparse.Namespace,
    registration_input: Path | None,
) -> bool:
    return should_run_registration(args, registration_input) and not should_refine_registration_input(
        args,
        registration_input,
    )


def resolve_pairwise_registration_jobs(args: argparse.Namespace) -> int:
    requested = getattr(args, "n_parallel_pairwise_regs", None)
    if requested is not None:
        return int(requested)
    if args.registration_pair_mode == "robust-boundary":
        return 0
    return 4


def resolve_dask_num_workers(args: argparse.Namespace) -> int | None:
    requested = getattr(args, "dask_num_workers", None)
    if requested is None:
        return None
    workers = int(requested)
    if workers <= 0:
        raise ValueError(f"--dask-num-workers must be positive, got {workers}")
    return workers


def resolve_coarse_reg_res_levels(args: argparse.Namespace, reg_res_level: int | None) -> tuple[int | None, ...]:
    requested = getattr(args, "coarse_reg_res_levels", None)
    if requested is None:
        return (reg_res_level,)
    levels = tuple(int(level) for level in requested)
    if not levels:
        raise ValueError("--coarse-reg-res-levels must contain at least one level")
    if any(level < 0 for level in levels):
        raise ValueError("--coarse-reg-res-levels values must be non-negative")
    if reg_res_level is None:
        raise ValueError("--coarse-reg-res-levels cannot be combined with --auto-reg-res-level")
    return levels


def preflight_track_run_outputs(
    args: argparse.Namespace,
    tiles: list[TileMetadata],
    track_configs: list[TrackRunConfig],
) -> None:
    if args.dry_run:
        return

    for config in track_configs:
        should_register = should_run_registration(args, config.registration_input)
        should_run_coarse = should_run_coarse_registration(args, config.registration_input)
        skip_registration_plots = getattr(args, "skip_registration_plots", False)
        if should_register and config.registration_output.exists():
            raise FileExistsError(f"{config.registration_output} already exists; move it before rerunning")
        if should_run_coarse and not skip_registration_plots:
            raise_if_nonempty_output_dir(config.registration_plots_dir, "Registration plots")
        if should_register and args.registration_pair_mode == "robust-boundary":
            raise_if_nonempty_output_dir(config.robust_boundary_qc_dir, "Robust boundary QC")

        if args.register_only:
            continue
        for channel in channels_to_fuse_for_tiles(tiles, config.selected_channels):
            raise_if_output_exists(channel_output_path(config.output, channel, separate_channels=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("20x-TL-561638"),
        help="Directory containing *.ome.tif tiles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="OME-Zarr output path. Defaults to INPUT_DIR/fused.ome.zarr. The script fails if this already exists.",
    )
    parser.add_argument(
        "--registration-output",
        type=Path,
        help="JSON output path for registration-only mode. Defaults to INPUT_DIR/registration.json.",
    )
    parser.add_argument(
        "--registration-input",
        type=Path,
        help="Use registration parameters previously written by --register-only.",
    )
    parser.add_argument(
        "--reference-registration-input",
        type=Path,
        help=(
            "Use a completed registration JSON as a physical geometry reference. "
            "Required when --reference-geometry-mode is penalized-xy."
        ),
    )
    parser.add_argument(
        "--reference-geometry-mode",
        choices=("none", "penalized-xy"),
        default="none",
        help=(
            "Constrain final registration geometry to --reference-registration-input. "
            "penalized-xy adds a soft y/x prior to keep geometry near the reference."
        ),
    )
    parser.add_argument(
        "--reference-xy-prior-weight",
        type=float,
        default=RobustBoundarySettings.reference_xy_prior_weight,
        help=(
            "Least-squares pseudo-observation weight for penalized-xy y/x corrections. "
            "Higher values keep y/x closer to --reference-registration-input."
        ),
    )
    parser.add_argument(
        "--shared-geometry-tracks",
        type=parse_track_slug_list,
        help=(
            "Comma-separated track slugs to solve as one shared geometry group, "
            "for example track1,track2. Requires --reference-geometry-mode penalized-xy "
            "and --register-only."
        ),
    )
    parser.add_argument(
        "--position-input",
        type=Path,
        help=(
            "JSON file listing OME-TIFF tile paths and metadata-derived translations "
            "to use instead of INPUT_DIR/*.ome.tif stage positions."
        ),
    )
    parser.add_argument(
        "--flatfield-dir",
        type=Path,
        help=(
            "Directory containing exported BaSiC *-chN-flatfield.tif files. "
            "Defaults to INPUT_DIR/basic_nz25_pooled."
        ),
    )
    parser.add_argument(
        "--flatfield-dir-by-source-view",
        type=parse_source_view_flatfield_dir,
        action="append",
        default=[],
        metavar="VIEW=DIR",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        help="Optional zero-based channel indices to stitch. Defaults to all channels.",
    )
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="Run registration and write registration parameters, but do not fuse.",
    )
    parser.add_argument(
        "--output-chunksize",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(32, 1024, 1024),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fusion-level",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-size",
        type=parse_batch_size,
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-jobs",
        type=int,
        default=16,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fusion-weight-mode",
        choices=("geometric", "content-dct", "content-preibisch", "content-preibisch-coarse"),
        default="content-preibisch-coarse",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--content-dct-size",
        type=int,
        default=32,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--content-dct-exponent",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--content-dct-otf-support-fraction",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--content-preibisch-sigma1",
        type=int,
        default=7,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--content-preibisch-sigma2",
        type=int,
        default=17,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--content-preibisch-coarse-stride",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(1, 8, 8),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--content-preibisch-softmax-exponent",
        type=float,
        default=2.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fusion-blend-width-voxels",
        type=float,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(64.0, 64.0, 64.0),
        help=(
            "Maximum border blending width in source voxels as Z Y X before conversion to physical units. "
            "Increase Y/X to soften residual flatfield or tile-gain seams."
        ),
    )
    parser.add_argument(
        "--fusion-blend-max-overlap-fraction",
        type=float,
        default=0.25,
        help=(
            "Maximum fraction of the smallest positive tile overlap used for border blending. "
            "Increase this to use more of the overlap ramp."
        ),
    )
    parser.add_argument(
        "--basic-cache-tiles",
        type=parse_nonnegative_int,
        default=128,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--basic-cache-max-gib",
        type=parse_nonnegative_float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--basic-cache-disk-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--basic-cache-z-chunk",
        type=parse_positive_int,
        default=32,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fusion-progress-log-seconds",
        type=parse_nonnegative_int,
        default=60,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Also write timestamped progress logs to this file. Useful when launchers buffer stdout.",
    )
    parser.set_defaults(
        registration_plots_dir=None,
        reg_channel_index=0,
        registration_binning=None,
        reg_res_level=4,
        coarse_reg_res_levels=None,
        auto_reg_res_level=False,
        n_parallel_pairwise_regs=None,
        dask_num_workers=32,
        registration_read_chunk_z=DEFAULT_REGISTRATION_READ_CHUNK_Z,
        registration_pair_mode="robust-boundary",
        gpu_pairwise_phase_correlation=False,
        robust_boundary_qc_dir=None,
        skip_registration_plots=True,
        fusion_compressor="zstd",
        jpegxr_level=DEFAULT_JPEGXR_LEVEL,
        profile_max_fusion_chunks=None,
        profile_skip_fusion_chunks=0,
        per_chunk_cupy_cleanup=False,
    )
    return parser.parse_args()


def parse_batch_size(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        batch_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'") from exc
    if batch_size <= 0:
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'")
    return batch_size


def parse_source_view_flatfield_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected source-view flatfield entry as VIEW=DIR, "
            f"got {value!r}"
        )
    view, path = value.split("=", 1)
    view = view.strip()
    if not view:
        raise argparse.ArgumentTypeError(f"Missing source-view name in {value!r}")
    if not path:
        raise argparse.ArgumentTypeError(f"Missing flatfield directory in {value!r}")
    return view, Path(path)


def parse_track_slug_list(value: str) -> tuple[str, ...]:
    tracks = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(tracks) < 2:
        raise argparse.ArgumentTypeError("expected at least two comma-separated track slugs")
    if len(set(tracks)) != len(tracks):
        raise argparse.ArgumentTypeError(f"duplicate track slug in {value!r}")
    return tracks


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def parse_nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return parsed


def parse_jpegxr_level(value: str) -> float:
    try:
        level = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("JPEG-XR level must be a number in [0, 100]") from exc
    if level < 0 or level > 100:
        raise argparse.ArgumentTypeError("JPEG-XR level must be in [0, 100]")
    return level


def ceil_div(numerator: int | float, denominator: int | float) -> int:
    numerator_int = int(numerator)
    denominator_int = int(denominator)
    return (numerator_int + denominator_int - 1) // denominator_int


def resolve_fusion_batch_size(
    requested_batch_size: int | str,
    output_stack_properties: dict[str, dict[str, int | float]],
    output_chunksize: dict[str, int],
) -> int:
    if requested_batch_size != "auto":
        return int(requested_batch_size)

    y_blocks = ceil_div(
        output_stack_properties["shape"]["y"],
        output_chunksize["y"],
    )
    x_blocks = ceil_div(
        output_stack_properties["shape"]["x"],
        output_chunksize["x"],
    )
    batch_size = y_blocks * x_blocks
    log(
        "Auto fusion batch size uses one full z-chunk plane "
        f"({y_blocks} y-blocks * {x_blocks} x-blocks = {batch_size}) "
        "to keep TIFF reads spatially local"
    )
    return batch_size


def log_tile_run_summary(tiles: list[TileMetadata]) -> None:
    source_views = Counter(tile.source_view or "__none__" for tile in tiles)
    shape_counts = Counter(tile_shape_zyx(tile) for tile in tiles)
    stage_scale_counts = Counter(
        tuple(round(tile_stage_scale(tile)[dim], 9) for dim in ("z", "y", "x"))
        for tile in tiles
    )
    translation_ranges = {
        dim: (
            min(tile.translation[dim] for tile in tiles),
            max(tile.translation[dim] for tile in tiles),
        )
        for dim in ("z", "y", "x")
    }
    log(
        "Tile metadata summary: "
        f"count={len(tiles)}, source_views={dict(source_views)}, "
        f"shapes_zyx={dict(shape_counts)}, spacing_um={tiles[0].spacing}, "
        f"stage_scale_patterns={dict(stage_scale_counts)}"
    )
    log(
        "Tile translation ranges um: "
        + ", ".join(
            f"{dim}={translation_ranges[dim][0]:.3f}..{translation_ranges[dim][1]:.3f}"
            for dim in ("z", "y", "x")
        )
    )
    log(f"Estimated output shape from metadata, before registration: {estimate_output_shape(tiles)}")


def log_transform_translation_summary(label: str, params: list[Any]) -> None:
    translations = [affine_translation_zyx(param) for param in params]
    ranges = {}
    for index, dim in enumerate(("z", "y", "x")):
        values = [translation[index] for translation in translations]
        ranges[dim] = (min(values), max(values))
    log(
        f"{label} transform translation ranges um: "
        + ", ".join(f"{dim}={ranges[dim][0]:.3f}..{ranges[dim][1]:.3f}" for dim in ("z", "y", "x"))
    )


@profile
def run_stitch_once(
    args: argparse.Namespace,
    *,
    tiles: list[TileMetadata],
    input_dir: Path,
    output: Path,
    registration_output: Path,
    registration_plots_dir: Path,
    robust_boundary_qc_dir: Path,
    flatfield_dir: Path,
    flatfield_dirs_by_source_view: dict[str, Path] | None,
    selected_channels: tuple[int, ...] | None,
    registration_input: Path | None,
    reference_registration_input: Path | None = None,
    source_label: str | None = None,
) -> int:
    log(f"Input directory: {input_dir}")
    log(f"Flatfield directory: {flatfield_dir}")
    if flatfield_dirs_by_source_view:
        log(f"Flatfield directories by source_view: {flatfield_dirs_by_source_view}")
    log(f"Registration output path: {registration_output}")
    log_tile_run_summary(tiles)

    channels_to_fuse = channels_to_fuse_for_tiles(tiles, selected_channels)
    log(f"Channels to fuse separately: {channels_to_fuse}")
    output_chunksize = output_chunksize_arg(args.output_chunksize)
    log(f"Requested fusion output chunksize: {output_chunksize}")
    output_spacing = fusion_output_spacing(tiles[0].spacing, args.fusion_level)
    log(f"Fusion output level: {args.fusion_level}; output spacing: {output_spacing}")

    registration_binning = None
    if args.registration_binning is not None:
        registration_binning = {
            "z": args.registration_binning[0],
            "y": args.registration_binning[1],
            "x": args.registration_binning[2],
        }
    reg_res_level = args.reg_res_level
    if args.auto_reg_res_level:
        reg_res_level = None
    elif reg_res_level is None and registration_binning is None:
        reg_res_level = 3
    coarse_reg_res_levels = resolve_coarse_reg_res_levels(args, reg_res_level)
    should_register = should_run_registration(args, registration_input)
    should_run_coarse = should_run_coarse_registration(args, registration_input)
    skip_registration_plots = getattr(args, "skip_registration_plots", False)
    if should_register and args.registration_pair_mode == "robust-boundary" and not args.dry_run:
        require_cuda_for_robust_boundary()
    if should_register and args.gpu_pairwise_phase_correlation and not args.dry_run:
        require_cuda_for_robust_boundary()

    n_parallel_pairwise_regs = resolve_pairwise_registration_jobs(args)
    dask_num_workers = resolve_dask_num_workers(args)
    registration_read_chunk_z = int(args.registration_read_chunk_z)
    if registration_read_chunk_z <= 0:
        raise ValueError(f"--registration-read-chunk-z must be positive, got {registration_read_chunk_z}")

    print_plan(
        tiles,
        output,
        registration_output,
        registration_plots_dir,
        robust_boundary_qc_dir,
        selected_channels,
        args.register_only,
        should_register,
        args.fusion_compressor,
        args.jpegxr_level,
        args.registration_pair_mode,
        tuple(args.registration_binning) if args.registration_binning is not None else None,
        reg_res_level,
        n_parallel_pairwise_regs,
        dask_num_workers,
        registration_read_chunk_z,
    )
    if should_run_coarse and len(coarse_reg_res_levels) > 1:
        log(f"Hierarchical coarse registration levels: {coarse_reg_res_levels}")
    if args.dry_run:
        return 0

    if should_register and registration_output.exists():
        raise FileExistsError(f"{registration_output} already exists; move it before rerunning")
    if should_run_coarse and not skip_registration_plots:
        raise_if_nonempty_output_dir(registration_plots_dir, "Registration plots")
    if should_register and args.registration_pair_mode == "robust-boundary":
        raise_if_nonempty_output_dir(robust_boundary_qc_dir, "Robust boundary QC")

    separate_channels = True
    channel_outputs = {
        channel: channel_output_path(output, channel, separate_channels=separate_channels)
        for channel in channels_to_fuse
    }
    for channel, channel_output in channel_outputs.items():
        log(f"Channel {channel} output path: {channel_output}")
        if not args.register_only:
            raise_if_output_exists(channel_output)
    from multiview_stitcher import fusion, misc_utils, registration
    from multiview_stitcher.fusion._core import process_output_chunksize, process_output_stack_properties
    from multiview_stitcher import spatial_image_utils as si_utils
    log("Imported multiview-stitcher fusion/registration modules")
    log(f"Fusion batch settings: n_batch={args.batch_size}, threaded_jobs={args.batch_jobs}")
    log(f"Final fusion compressor: {args.fusion_compressor}")
    if args.fusion_compressor == "jpegxr":
        log(
            "JPEG-XR output will be written after GPU fusion from temporary zstd chunks "
            f"with level={compression_level(args.jpegxr_level):.3f}"
        )

    transform_key = TRANSFORM_KEY
    params = None
    reference_params = None
    fixed_reference_axes: set[str] | None = None
    reference_prior_weights_zyx: tuple[float, float, float] | None = None
    residual_reject_axes: set[str] | None = None

    if registration_input is not None:
        log(f"Loading registration params from {registration_input.resolve()}")
        params = load_registration_params(registration_input.resolve(), tiles)
        transform_key = REGISTERED_TRANSFORM_KEY
        log(f"Loaded {len(params)} registration transforms")
        log_transform_translation_summary("Loaded registration", params)
    if args.reference_geometry_mode != "none":
        if reference_registration_input is None:
            raise ValueError("--reference-geometry-mode requires --reference-registration-input")
        reference_options = reference_geometry_solver_options(
            args.reference_geometry_mode,
            args.reference_xy_prior_weight,
        )
        log(f"Loading reference registration params from {reference_registration_input.resolve()}")
        reference_params = load_registration_params(reference_registration_input.resolve(), tiles)
        fixed_reference_axes = reference_options.fixed_axes
        reference_prior_weights_zyx = reference_options.reference_prior_weights_zyx
        residual_reject_axes = reference_options.residual_reject_axes
        if reference_prior_weights_zyx is not None:
            log(f"Reference xy prior weights z/y/x: {reference_prior_weights_zyx}")
        log(f"Loaded {len(reference_params)} reference registration transforms")
        log_transform_translation_summary("Reference registration", reference_params)

    if should_register:
        reg_source_channel = registration_source_channel(
            selected_channels,
            reg_channel_index=args.reg_channel_index,
            n_channels=tile_channel_count(tiles[0]),
        )
        log(f"Registration source channel: {reg_source_channel}")

    if should_refine_registration_input(args, registration_input):
        if params is None:
            raise RuntimeError("Loaded registration params were not initialized")
        refinement_start_params = (
            reference_params
            if reference_params is not None and args.reference_geometry_mode in {"full-xyz", "penalized-xy"}
            else params
        )
        if refinement_start_params is reference_params:
            log(f"Robust boundary refinement starts from reference geometry mode={args.reference_geometry_mode}")
        with heartbeat("robust boundary refinement"):
            robust_refinement = refine_registration_with_robust_boundaries(
                tiles,
                refinement_start_params,
                channel=reg_source_channel,
                output_dir=robust_boundary_qc_dir,
                settings=RobustBoundarySettings(),
                reference_params=reference_params,
                reference_input=reference_registration_input,
                fixed_reference_axes=fixed_reference_axes,
                reference_prior_weights_zyx=reference_prior_weights_zyx,
                residual_reject_axes=residual_reject_axes,
                reference_geometry_mode=args.reference_geometry_mode,
                source_label=source_label,
            )
        params = robust_refinement.params
        log_transform_translation_summary("Robust-boundary refined registration", params)
        save_registration_params(
            {"params": params},
            tiles,
            registration_output,
            robust_refinement=robust_refinement,
        )
        log(f"Wrote {registration_output}")
        return 0

    if should_run_coarse:
        registration_pairs = None
        pre_registration_pruning_method = "alternating_pattern"
        if args.registration_pair_mode in {"axis-aligned", "robust-boundary"}:
            registration_pairs = axis_aligned_registration_pairs(tiles)
            pre_registration_pruning_method = None
            log(f"Using {len(registration_pairs)} explicit axis-aligned registration pairs")
        elif args.registration_pair_mode == "spanning-tree":
            registration_pairs = spanning_tree_registration_pairs(tiles)
            pre_registration_pruning_method = None
            log(f"Using {len(registration_pairs)} explicit spanning-tree registration pairs")
        else:
            log("Using multiview-stitcher package pair graph and pruning")
        log("Building registration multiscale inputs")
        registration_source_level, registration_available_levels, registration_level_offset = registration_source_level_for_tiles(
            tiles,
            coarse_reg_res_levels,
            registration_binning,
        )
        log(
            "Registration TIFF source level: "
            f"level{registration_source_level}/available{registration_available_levels}; "
            f"registration level offset={registration_level_offset}"
        )
        reg_msims, reg_stores, reg_channel_label = build_registration_msims(
            tiles,
            reg_source_channel,
            read_chunk_z=registration_read_chunk_z,
            source_level=registration_source_level,
        )
        initial_transform_key = TRANSFORM_KEY
        pairwise_reg_func = pairwise_registration_function(use_gpu=args.gpu_pairwise_phase_correlation)
        log(
            "Pairwise registration function: "
            f"{pairwise_reg_func.__name__}; gpu={args.gpu_pairwise_phase_correlation}"
        )
        try:
            full_params = None
            registration_result = None
            current_transform_key = initial_transform_key
            with batched_pairwise_registration(dask_num_workers=dask_num_workers), dask_progress("registration.register"):
                for stage_index, stage_reg_res_level in enumerate(coarse_reg_res_levels):
                    if full_params is not None:
                        current_transform_key = f"{REGISTERED_TRANSFORM_KEY}_hierarchical_input_{stage_index}"
                        set_msims_affine_transform(reg_msims, full_params, current_transform_key)
                    stage_transform_key = f"{REGISTERED_TRANSFORM_KEY}_hierarchical_stage_{stage_index}"
                    if stage_index == len(coarse_reg_res_levels) - 1:
                        stage_transform_key = REGISTERED_TRANSFORM_KEY
                    effective_stage_reg_res_level = effective_registration_level(
                        stage_reg_res_level,
                        registration_level_offset,
                    )
                    log(
                        "Starting registration.register "
                        f"stage {stage_index + 1}/{len(coarse_reg_res_levels)} "
                        f"at reg_res_level={stage_reg_res_level} "
                        f"(effective={effective_stage_reg_res_level})"
                    )
                    with heartbeat(f"registration.register level {stage_reg_res_level}"):
                        registration_result = registration.register(
                            reg_msims,
                            reg_channel=reg_channel_label,
                            transform_key=current_transform_key,
                            new_transform_key=stage_transform_key,
                            registration_binning=registration_binning,
                            reg_res_level=effective_stage_reg_res_level,
                            n_parallel_pairwise_regs=n_parallel_pairwise_regs,
                            pairs=registration_pairs,
                            pairwise_reg_func=pairwise_reg_func,
                            pre_registration_pruning_method=pre_registration_pruning_method,
                            return_dict=True,
                        )
                    log(
                        "Finished registration.register "
                        f"stage {stage_index + 1}/{len(coarse_reg_res_levels)} "
                        f"at reg_res_level={stage_reg_res_level} "
                        f"(effective={effective_stage_reg_res_level})"
                    )
                    full_params = msim_full_transforms(reg_msims, stage_transform_key)
            if registration_result is None or full_params is None:
                raise RuntimeError("Registration did not produce parameters")
            params = full_params_relative_to_stage(reg_msims, full_params)
            log_transform_translation_summary("Coarse registration", params)
            registration_result = {
                **registration_result,
                "params": params,
                "hierarchical_coarse_registration": {
                    "reg_res_levels": list(coarse_reg_res_levels),
                },
            }
            transform_key = REGISTERED_TRANSFORM_KEY
            robust_refinement = None
            if args.registration_pair_mode == "robust-boundary":
                refinement_start_params = (
                    reference_params
                    if reference_params is not None and args.reference_geometry_mode in {"full-xyz", "penalized-xy"}
                    else params
                )
                if refinement_start_params is reference_params:
                    log(f"Robust boundary refinement starts from reference geometry mode={args.reference_geometry_mode}")
                with heartbeat("robust boundary refinement"):
                    robust_refinement = refine_registration_with_robust_boundaries(
                        tiles,
                        refinement_start_params,
                        channel=reg_source_channel,
                        output_dir=robust_boundary_qc_dir,
                        settings=RobustBoundarySettings(),
                        reference_params=reference_params,
                        reference_input=reference_registration_input,
                        fixed_reference_axes=fixed_reference_axes,
                        reference_prior_weights_zyx=reference_prior_weights_zyx,
                        residual_reject_axes=residual_reject_axes,
                        reference_geometry_mode=args.reference_geometry_mode,
                        source_label=source_label,
                )
                params = robust_refinement.params
                log_transform_translation_summary("Robust-boundary registration", params)
            save_registration_params(
                registration_result,
                tiles,
                registration_output,
                robust_refinement=robust_refinement,
            )
            log(f"Wrote {registration_output}")
            if skip_registration_plots:
                log("Skipped registration metric positional plots")
            else:
                save_registration_metric_position_plots(
                    reg_msims,
                    registration_plots_dir,
                    reg_channel_label=reg_channel_label,
                    reg_res_level=reg_res_level,
                    n_parallel_pairwise_regs=n_parallel_pairwise_regs,
                    registration_pairs=registration_pairs,
                )
        finally:
            close_stores(reg_stores)
            log("Closed registration TIFF stores")

        if args.register_only:
            return 0

    for channel in channels_to_fuse:
        channel_output = channel_outputs[channel]
        log(f"Output zarr: {channel_output}")
        log(f"Building zarr-backed fusion SpatialImages for channel {channel}")
        sims, stores, channel_label, source_level_counts, source_tiles = build_fusion_sims(
            tiles,
            channel,
            fusion_level=args.fusion_level,
        )
        zarr_backed_count = sum(si_utils.is_xarray_zarr_backed(sim) for sim in sims)
        source_level_summary = {
            f"level{source_level}/available{available_levels}": count
            for (source_level, available_levels), count in sorted(source_level_counts.items())
        }
        log(
            f"Built {len(sims)} fusion sims for {channel_label}; "
            f"zarr-backed={zarr_backed_count}/{len(sims)}, dims={sims[0].dims}, "
            f"shape={tuple(sims[0].shape)}"
        )
        log(f"Fusion TIFF source levels: {source_level_summary}")
        temp_workspace = None
        try:
            if params is not None:
                log(f"Applying {len(params)} registration transforms to channel {channel} sims")
                for sim, param in zip(sims, params, strict=True):
                    si_utils.set_sim_affine(
                        sim,
                        xaffine=spatial_affine_param(param),
                        transform_key=REGISTERED_TRANSFORM_KEY,
                        base_transform_key=TRANSFORM_KEY,
                    )
                log(f"Applied registration transforms to channel {channel} sims")
            output_stack_properties = process_output_stack_properties(
                sims,
                output_spacing=output_spacing,
                transform_key=transform_key,
            )
            log(f"Resolved output stack properties: {output_stack_properties}")
            log(f"Checking channel output path does not already exist: {channel_output}")
            raise_if_output_exists(channel_output)

            fusion_output = channel_output
            recompressed_output = None
            if args.fusion_compressor == "jpegxr":
                temp_workspace = tempfile.TemporaryDirectory(
                    prefix=f".{channel_output.name}.fusion-",
                    dir=channel_output.parent,
                )
                temp_root = Path(temp_workspace.name)
                fusion_output = temp_root / "fusion-zstd.ome.zarr"
                recompressed_output = temp_root / "fusion-jpegxr.ome.zarr"
                log(
                    "Channel "
                    f"{channel} will fuse to temporary zstd output {fusion_output} "
                    f"before JPEG-XR z-plane recompression to {channel_output}"
                )

            inverse_flatfield = load_fusion_inverse_flatfields(
                source_tiles,
                flatfield_dir=flatfield_dir,
                flatfield_dirs_by_source_view=flatfield_dirs_by_source_view or None,
                channel=channel,
            )
            if isinstance(inverse_flatfield, dict):
                log(f"Resolved source-view BaSiC corrections for channel {channel}")
            else:
                log(f"Resolved pooled BaSiC correction for channel {channel}")
            resolved_chunksize = process_output_chunksize(sims, output_chunksize)
            log(f"Resolved fusion output chunksize: {resolved_chunksize}")
            resolved_batch_size = resolve_fusion_batch_size(
                args.batch_size,
                output_stack_properties,
                resolved_chunksize,
            )
            weights_func, weights_func_kwargs = fusion_weight_config(args, resolved_chunksize)
            blending_widths = narrow_fusion_blending_widths(
                source_tiles,
                params,
                default_width_voxels_zyx=tuple(float(value) for value in args.fusion_blend_width_voxels),
                max_overlap_fraction=args.fusion_blend_max_overlap_fraction,
            )
            log(
                "Resolved fusion batch settings: "
                f"n_batch={resolved_batch_size}, threaded_jobs={args.batch_jobs}"
            )
            if weights_func_kwargs is None:
                log("Fusion weights: geometric border/valid-support weights only")
            else:
                log(f"Fusion weights: {args.fusion_weight_mode} quality weights {weights_func_kwargs}")
            log(
                "BaSiC cache settings: "
                f"tile_cache_size={args.basic_cache_tiles}, "
                f"tile_cache_max_gib={args.basic_cache_max_gib}, "
                f"tile_cache_disk_root={args.basic_cache_disk_dir or fusion_output.parent}, "
                f"tile_cache_z_chunk={args.basic_cache_z_chunk}"
            )
            batch_func = misc_utils.process_batch_using_joblib
            batch_func_kwargs = {
                "n_jobs": args.batch_jobs,
                "backend": "threading",
            }
            if args.profile_max_fusion_chunks is not None:
                batch_func = make_profile_limited_batch_func(
                    args.profile_max_fusion_chunks,
                    misc_utils.process_batch_using_joblib,
                    skip_chunks=args.profile_skip_fusion_chunks,
                )
                log(
                    "Profiling fusion is bounded to "
                    f"{args.profile_max_fusion_chunks} output chunk(s) "
                    f"after skipping {args.profile_skip_fusion_chunks}"
                )
            if args.per_chunk_cupy_cleanup:
                log("Fusion uses per-chunk CuPy cleanup")
            else:
                log("Fusion defers CuPy cleanup until the fusion context exits")
            log(f"Fusion blending widths (physical units): {blending_widths}")
            log(f"Entering multiview-stitcher fusion.fuse streaming write to {fusion_output}")
            if args.fusion_progress_log_seconds > 0:
                fusion_progress_context = filesystem_progress(
                    f"fusion output channel {channel}",
                    fusion_output,
                    every_seconds=args.fusion_progress_log_seconds,
                )
            else:
                fusion_progress_context = nullcontext()
                log("Filesystem progress logging for fusion output is disabled")
            fusion_output_context = single_scale_fusion_output()
            with (
                temporary_basic_disk_cache_dir(args.basic_cache_disk_dir, fusion_output) as basic_cache_disk_dir,
                basic_corrected_zarr_reads(
                    inverse_flatfield,
                    dataset_info_key="source_view" if isinstance(inverse_flatfield, dict) else None,
                    dataset_attr_keys=("source_view",) if isinstance(inverse_flatfield, dict) else (),
                    cache_key_attr="basic_tile_cache_key",
                    tile_cache_size=args.basic_cache_tiles,
                    tile_cache_max_bytes=(
                        None
                        if args.basic_cache_max_gib is None
                        else int(args.basic_cache_max_gib * 1024**3)
                    ),
                    tile_cache_disk_dir=basic_cache_disk_dir,
                    tile_cache_z_chunk=args.basic_cache_z_chunk,
                ),
                cupy_cleanup_context(args.per_chunk_cupy_cleanup),
                spatial_chunks_for_package_pyramid(),
                fusion_output_context,
                fusion_progress_context,
            ):
                with heartbeat(f"fusion.fuse channel {channel}"):
                    try:
                        fusion.fuse(
                            images=sims,
                            transform_key=transform_key,
                            output_spacing=output_spacing,
                            output_zarr_url=str(fusion_output),
                            output_chunksize=output_chunksize,
                            blending_widths=blending_widths,
                            weights_func=weights_func,
                            weights_func_kwargs=weights_func_kwargs,
                            zarr_options={
                                "ome_zarr": True,
                                "ngff_version": NGFF_VERSION,
                                "overwrite": False,
                                "zarr_array_creation_kwargs": zarr_v3_array_creation_kwargs(
                                    tuple(sims[0].dims)
                                ),
                            },
                            batch_options={
                                "n_batch": resolved_batch_size,
                                "batch_func": batch_func,
                                "batch_func_kwargs": batch_func_kwargs,
                            },
                            backend=FUSION_BACKEND,
                        )
                    except ProfileFusionStop:
                        log(
                            "Stopped bounded profiling fusion after "
                            f"{args.profile_max_fusion_chunks} output chunk(s)"
                        )
                        return 0
            log(f"Finished multiview-stitcher fusion.fuse scale0 write for channel {channel}")
            log(
                "Fusion scale0 output after package write: "
                f"{format_filesystem_progress(fusion_output, filesystem_progress_snapshot(fusion_output))}"
            )
            if args.fusion_compressor == "jpegxr":
                if recompressed_output is None:
                    raise RuntimeError("JPEG-XR recompression output was not initialized")
                with heartbeat(f"JPEG-XR z-plane recompression channel {channel}"):
                    recompress_ome_zarr_to_jpegxr(
                        fusion_output,
                        recompressed_output,
                        jpegxr_level=args.jpegxr_level,
                    )
                if channel_output.exists():
                    raise FileExistsError(f"{channel_output} appeared during JPEG-XR recompression")
                recompressed_output.rename(channel_output)
                log(f"Moved completed JPEG-XR output to {channel_output}")
            else:
                with heartbeat(f"completed Zarr pyramid channel {channel}"):
                    build_ome_zarr_pyramid_from_scale0(fusion_output)
                log(
                    "Fusion output after completed-Zarr pyramid write: "
                    f"{format_filesystem_progress(fusion_output, filesystem_progress_snapshot(fusion_output))}"
                )
        finally:
            if temp_workspace is not None:
                temp_workspace.cleanup()
            close_stores(stores)
            log(f"Closed TIFF stores for channel {channel}")

        thumbnail_path = write_center_z_thumbnail(channel_output)
        log(f"Wrote center-z thumbnail {thumbnail_path}")
        log(f"Wrote {channel_output} ({channel_label})")
    return 0


def validate_shared_reference_geometry_run(
    args: argparse.Namespace,
    configs: list[TrackRunConfig],
    reference_registration_input: Path | None,
) -> None:
    if not configs:
        return
    if args.reference_geometry_mode != "penalized-xy":
        raise ValueError("--shared-geometry-tracks requires --reference-geometry-mode penalized-xy")
    if reference_registration_input is None:
        raise ValueError("--shared-geometry-tracks requires --reference-registration-input")
    if args.reference_xy_prior_weight < 0.0:
        raise ValueError("--reference-xy-prior-weight must be non-negative")
    if not args.register_only:
        raise ValueError("--shared-geometry-tracks is only supported with --register-only")
    if any(config.registration_input is not None for config in configs):
        raise ValueError("--shared-geometry-tracks cannot be combined with --registration-input")


def run_shared_reference_geometry_registration(
    args: argparse.Namespace,
    *,
    tiles: list[TileMetadata],
    input_dir: Path,
    flatfield_dir: Path,
    configs: list[TrackRunConfig],
    reference_registration_input: Path,
) -> None:
    validate_shared_reference_geometry_run(args, configs, reference_registration_input)
    reference_options = reference_geometry_solver_options(
        args.reference_geometry_mode,
        args.reference_xy_prior_weight,
    )
    log(
        "Starting shared reference-constrained registration for tracks "
        f"{tuple(config.track.slug for config in configs)} "
        f"with mode={args.reference_geometry_mode}"
    )
    if reference_options.reference_prior_weights_zyx is not None:
        log(f"Reference xy prior weights z/y/x: {reference_options.reference_prior_weights_zyx}")
    for config in configs:
        log(
            f"Shared track {config.track.slug} ({config.track.track_id}): "
            f"channels={config.track.channels}, names={config.track.channel_names}"
        )
    if args.dry_run:
        return

    require_cuda_for_robust_boundary()
    log(f"Input directory: {input_dir}")
    log(f"Flatfield directory: {flatfield_dir}")
    log(f"Loading shared reference registration params from {reference_registration_input.resolve()}")
    reference_params = load_registration_params(reference_registration_input.resolve(), tiles)
    log_transform_translation_summary("Shared reference registration", reference_params)

    settings = RobustBoundarySettings()
    patch_specs = sample_boundary_patches(
        tiles,
        reference_params,
        axis_aligned_registration_pairs(tiles),
        settings,
    )
    log(
        "Shared robust boundary refinement sampled "
        f"{len(patch_specs)} combined patch(es)"
    )
    source_channels: list[int] = []
    for config in configs:
        reg_source_channel = registration_source_channel(
            config.selected_channels,
            reg_channel_index=args.reg_channel_index,
            n_channels=tile_channel_count(tiles[0]),
        )
        source_channels.append(reg_source_channel)
    combined_source_label = "+".join(config.track.slug for config in configs)
    source_channels_tuple = tuple(source_channels)
    log(
        "Building combined shared boundary constraints for "
        f"{combined_source_label} from channels {source_channels_tuple}"
    )
    combined_constraints = build_combined_boundary_constraints(
        tiles,
        source_channels_tuple,
        patch_specs,
        settings,
        source_label=combined_source_label,
    )

    corrections_zyx, combined_constraints, anchor_tile = solve_tile_corrections_with_residual_rejection(
        tiles,
        combined_constraints,
        settings,
        fixed_axes=reference_options.fixed_axes,
        reference_prior_weights_zyx=reference_options.reference_prior_weights_zyx,
        residual_reject_axes=reference_options.residual_reject_axes,
    )
    shared_params = apply_corrections_to_params(reference_params, corrections_zyx, tiles[0].spacing)
    if reference_options.fixed_axes:
        shared_params = apply_reference_fixed_axes(shared_params, reference_params, reference_options.fixed_axes)
    reference_geometry = reference_geometry_constraint(
        mode=args.reference_geometry_mode,
        reference_input=reference_registration_input,
        fixed_axes=reference_options.fixed_axes,
        params=shared_params,
        reference_params=reference_params,
        constraints=combined_constraints,
        shared_geometry_tracks=tuple(config.track.slug for config in configs),
        reference_prior_weights_zyx=reference_options.reference_prior_weights_zyx,
        residual_reject_axes=reference_options.residual_reject_axes,
    )
    log_transform_translation_summary("Shared reference-constrained registration", shared_params)

    for config in configs:
        summary = robust_summary(combined_constraints, corrections_zyx)
        reg_source_channel = registration_source_channel(
            config.selected_channels,
            reg_channel_index=args.reg_channel_index,
            n_channels=tile_channel_count(tiles[0]),
        )
        write_robust_boundary_qc(
            config.robust_boundary_qc_dir,
            tiles,
            shared_params,
            channel=reg_source_channel,
            constraints=combined_constraints,
            corrections_zyx=corrections_zyx,
            summary=summary,
            reference_geometry=reference_geometry,
            reference_params=reference_params,
        )
        robust_refinement = RobustBoundaryRefinementResult(
            params=shared_params,
            constraints=combined_constraints,
            corrections_zyx=corrections_zyx,
            anchor_tile=anchor_tile,
            output_dir=config.robust_boundary_qc_dir,
            summary=summary,
            reference_geometry=reference_geometry,
        )
        save_registration_params(
            {"params": reference_params},
            tiles,
            config.registration_output,
            robust_refinement=robust_refinement,
        )
        log(f"Wrote shared reference-constrained registration {config.registration_output}")


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    configure_writable_caches(root)
    configure_log_file(args.log_file)
    log("Starting 20x-TL multiview stitching script")
    if any(value <= 0.0 for value in args.fusion_blend_width_voxels):
        raise ValueError("--fusion-blend-width-voxels values must be positive")
    if not 0.0 < args.fusion_blend_max_overlap_fraction <= 1.0:
        raise ValueError("--fusion-blend-max-overlap-fraction must be in (0, 1]")

    input_dir = args.input_dir.resolve()
    output = (args.output or input_dir / "fused.ome.zarr").resolve()
    registration_output = (args.registration_output or input_dir / "registration.json").resolve()
    registration_input = args.registration_input.resolve() if args.registration_input is not None else None
    reference_registration_input = (
        args.reference_registration_input.resolve()
        if args.reference_registration_input is not None
        else None
    )
    if args.reference_geometry_mode != "none":
        if reference_registration_input is None:
            raise ValueError("--reference-geometry-mode requires --reference-registration-input")
    registration_plots_dir = (
        args.registration_plots_dir.resolve()
        if args.registration_plots_dir is not None
        else input_dir / "registration-plots"
    )
    robust_boundary_qc_dir = (
        args.robust_boundary_qc_dir.resolve()
        if args.robust_boundary_qc_dir is not None
        else input_dir / "robust-boundary-qc"
    )
    flatfield_dir = (
        args.flatfield_dir.resolve()
        if args.flatfield_dir is not None
        else input_dir / "basic_nz25_pooled"
    )
    flatfield_source_views = [view for view, _path in args.flatfield_dir_by_source_view]
    duplicate_flatfield_views = sorted(
        view
        for view, count in Counter(flatfield_source_views).items()
        if count > 1
    )
    if duplicate_flatfield_views:
        raise ValueError(
            "Duplicate --flatfield-dir-by-source-view entries for source_view(s): "
            f"{duplicate_flatfield_views}"
        )
    flatfield_dirs_by_source_view = {
        view: path.resolve()
        for view, path in args.flatfield_dir_by_source_view
    }
    position_input = args.position_input.resolve() if args.position_input is not None else None
    if position_input is not None:
        tiles = read_position_input_tiles(position_input)
        log(f"Read position-input metadata: {position_input}")
    elif registration_input is not None:
        tiles = read_registration_input_tiles(registration_input)
        log(f"Read tile metadata from registration-input: {registration_input}")
    else:
        tiles = read_tiles_metadata(input_dir)
        log("Read OME-TIFF metadata")

    selected_channels = tuple(args.channels) if args.channels is not None else None
    tracks = selected_track_metadata(tiles[0].tracks, selected_channels)
    if not tracks:
        raise ValueError(f"No detected tracks contain requested channels: {selected_channels}")

    split_by_track = len(tiles[0].tracks) > 1
    if split_by_track:
        log("Detected acquisition tracks; processing each track separately")
    else:
        log("Detected a single acquisition track")
    for track in tracks:
        log(
            f"Track {track.slug} ({track.track_id}): "
            f"channels={track.channels}, names={track.channel_names}"
        )

    track_configs = []
    for track in tracks:
        track_output = insert_track_suffix(output, track.slug) if split_by_track else output
        track_registration_output = (
            insert_track_suffix(registration_output, track.slug)
            if split_by_track
            else registration_output
        )
        track_registration_input = (
            insert_track_suffix(registration_input, track.slug)
            if registration_input is not None and split_by_track
            else registration_input
        )
        track_plots_dir = (
            track_registration_plots_dir(registration_plots_dir, track.slug)
            if split_by_track
            else registration_plots_dir
        )
        track_robust_boundary_qc_dir = (
            track_qc_dir(robust_boundary_qc_dir, track.slug)
            if split_by_track
            else robust_boundary_qc_dir
        )
        track_configs.append(
            TrackRunConfig(
                track=track,
                output=track_output,
                registration_output=track_registration_output,
                registration_input=track_registration_input,
                registration_plots_dir=track_plots_dir,
                robust_boundary_qc_dir=track_robust_boundary_qc_dir,
                selected_channels=track.channels,
            )
        )

    preflight_track_run_outputs(args, tiles, track_configs)

    shared_track_slugs = set(args.shared_geometry_tracks or ())
    if shared_track_slugs:
        known_track_slugs = {config.track.slug for config in track_configs}
        missing = sorted(shared_track_slugs - known_track_slugs)
        if missing:
            raise ValueError(f"--shared-geometry-tracks contains unknown selected track slug(s): {missing}")
        shared_configs = [config for config in track_configs if config.track.slug in shared_track_slugs]
        run_shared_reference_geometry_registration(
            args,
            tiles=tiles,
            input_dir=input_dir,
            flatfield_dir=flatfield_dir,
            configs=shared_configs,
            reference_registration_input=reference_registration_input,
        )

    for config in track_configs:
        if config.track.slug in shared_track_slugs:
            continue
        track = config.track
        log(f"Starting {track.slug} processing")
        run_stitch_once(
            args,
            tiles=tiles,
            input_dir=input_dir,
            output=config.output,
            registration_output=config.registration_output,
            registration_plots_dir=config.registration_plots_dir,
            robust_boundary_qc_dir=config.robust_boundary_qc_dir,
            flatfield_dir=flatfield_dir,
            flatfield_dirs_by_source_view=flatfield_dirs_by_source_view or None,
            selected_channels=config.selected_channels,
            registration_input=config.registration_input,
            reference_registration_input=reference_registration_input,
            source_label=track.slug,
        )
        log(f"Finished {track.slug} processing")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
