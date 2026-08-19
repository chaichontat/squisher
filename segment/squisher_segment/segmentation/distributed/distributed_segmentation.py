import csv
import datetime
import hashlib
import json
import logging
from contextlib import contextmanager
from json import JSONEncoder
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cellpose.io
import cupy as cp
import dask
import dask_jobqueue
import distributed
import imagecodecs
import numpy as np
import tifffile
import click
import zarr
from numpy.typing import NDArray

from squisher_segment.segment.normalize import sample_percentile
from squisher_segment.segmentation.distributed.cache_utils import (
    atomic_write_text,
    read_nonempty_cache,
    read_normalization_cache,
    write_nonempty_cache,
    write_normalization_cache,
)
from squisher_segment.segment.model_artifacts import (
    file_sha256,
    plan_path_for_device,
    runtime_source_sha256,
)
from squisher_segment.segmentation.distributed.gpu_cluster import cluster, myLocalCluster
from squisher_segment.segmentation.distributed.merge_utils import (
    block_faces,
    bounding_boxes_in_global_coordinates,
    get_block_crops,
    get_nblocks,
    global_segment_ids,
    merge_boxes_for_labels,
    remove_overlaps,
    stitch_labels,
    create_zarr_array,
    decode_block_global_labels,
    label_zarr_codecs,
)
from squisher_segment.segmentation.distributed.model_cache import CellposeModelPlugin, get_cached_model
from squisher_segment.segmentation.distributed.tiling import solve_internal_zyx_for_tiles

# Increase Dask timeouts to prevent "Event loop was unresponsive" warnings
# during long-running GPU operations (Cellpose inference can hold the GIL for seconds)
dask.config.set({
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "120s",
    "distributed.scheduler.worker-ttl": "10m",
    "distributed.admin.tick.limit": "10m",
})

# IMPORTANT: No Rich logging at module level - RichHandler uses ContextVar which
# cannot be pickled. This causes "cannot pickle '_contextvars.ContextVar'" errors
# when Dask serializes functions/objects that capture loggers with Rich handlers.
# CLI commands set up their own Rich logging independently (causes duplicate progress bars).
logger = logging.getLogger(__name__)


class NumpyEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@contextmanager
def progress_bar(total: int):
    def _advance(*args: Any, **kwargs: Any) -> None:
        return None

    yield _advance


def unsharp_all(img: NDArray[Any], crop: None = None, channel_axis: int = 3) -> NDArray[Any]:
    from cucim.skimage import filters as cucim_filters

    image = np.asarray(img)
    axis = channel_axis % image.ndim
    result = np.empty(image.shape, dtype=np.float32)
    pool = cp.get_default_memory_pool()
    for channel in range(image.shape[axis]):
        selection = [slice(None)] * image.ndim
        selection[axis] = channel
        selection_tuple = tuple(selection)
        image_gpu = cp.asarray(image[selection_tuple], dtype=cp.float32)
        result_gpu = cucim_filters.unsharp_mask(
            image_gpu, radius=3, preserve_range=True
        )
        result[selection_tuple] = cp.asnumpy(result_gpu)
        del image_gpu, result_gpu
        pool.free_all_blocks()
    return result


def _get_worker_logger() -> logging.Logger:
    """Get a logger safe for use in Dask workers.

    Returns a logger that uses only basic handlers (no Rich) to avoid
    ContextVar pickling issues when Dask serializes worker functions.
    """
    worker_logger = logging.getLogger(f"{__name__}.worker")
    if not worker_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\n%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        ))
        worker_logger.addHandler(handler)
        worker_logger.setLevel(logging.INFO)
        worker_logger.propagate = False  # Don't propagate to root (which may have Rich)
    return worker_logger


def _retire_worker_after_error(*, reason: str) -> None:
    """Best-effort retire the current Dask worker after a task failure.

    This prevents a worker that hit a fatal error (often GPU/CUDA state) from
    picking up the next tile. The task exception is still re-raised to the
    scheduler so remaining tiles can be rescheduled to healthy workers.
    """
    try:
        worker = distributed.get_worker()
    except ValueError:
        return

    setattr(worker, "_squisher_segment_fatal_error", True)

    loop = getattr(worker, "loop", None)
    if loop is None:
        try:
            worker.close(reason=reason)  # type: ignore[call-arg]
        except Exception as close_exc:
            _get_worker_logger().error(f"Failed to close worker after error: {close_exc!r}")
        return

    try:
        loop.add_callback(worker.close, reason=reason)  # type: ignore[misc]
    except Exception as close_exc:
        _get_worker_logger().error(f"Failed to schedule worker close after error: {close_exc!r}")

def _log_slurm_tile_summary(total_tiles: int, processed_tiles: int, elapsed_seconds: float) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return

    summary = f"Processed {processed_tiles}/{total_tiles} tiles in {elapsed_seconds:.1f}s"
    usage_suffix = ""

    if shutil.which("nvidia-smi") is None:
        logger.info(summary)
        return

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        logger.info(summary)
        return

    reader = csv.reader(result.stdout.splitlines())
    stats = []
    for row in reader:
        if len(row) < 4:
            continue
        idx, mem_used, mem_total, util = [field.strip() for field in row[:4]]
        stats.append(f"{idx}: {mem_used}/{mem_total} mem, util {util}")

    if stats:
        usage_suffix = f" | usage snapshot — {'; '.join(stats)}"

    logger.info(summary + usage_suffix)


def _apply_startup_stagger(stagger_seconds: float, workers_per_gpu: int) -> None:
    """Apply one-time stagger delay based on worker's sequential index.

    Workers are named "gpu-{dev}-w{k}". This function computes a linear index
    and delays the first task on each worker to avoid GPU memory contention.
    """
    import re

    worker = distributed.get_worker()

    # Store state on worker object (not module global) - persists across tasks
    if getattr(worker, "_stagger_done", False):
        return
    worker._stagger_done = True

    name = getattr(worker, "name", "")
    match = re.match(r"gpu-(\d+)-w(\d+)", name)
    if not match:
        return

    gpu_idx = int(match.group(1))
    worker_idx = int(match.group(2))
    linear_idx = gpu_idx * workers_per_gpu + worker_idx

    if linear_idx > 0:
        delay = linear_idx * stagger_seconds
        _get_worker_logger().info(f"Worker {name}: staggering start by {delay}s")
        time.sleep(delay)


def _save_intermediate_state(
    temp_dir: Path,
    faces_list: list,
    boxes_list: list,
    box_ids_list: list,
    non_empty_indices: list[tuple[int, ...]],
) -> None:
    """Save cellpose results for later stitching."""
    path = temp_dir / "intermediate_state.npz"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=temp_dir, prefix=".tmp-", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez(
                temporary_file,
                faces=np.array(faces_list, dtype=object),
                boxes=np.array(boxes_list, dtype=object),
                box_ids=np.array(box_ids_list, dtype=object),
                non_empty_indices=np.array(non_empty_indices),
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(temp_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_intermediate_state(temp_dir: Path) -> tuple[list, list, list, list[tuple[int, ...]]]:
    """Load saved cellpose results for stitching."""
    data = np.load(temp_dir / "intermediate_state.npz", allow_pickle=True)
    return (
        data["faces"].tolist(),
        data["boxes"].tolist(),
        data["box_ids"].tolist(),
        [tuple(idx) for idx in data["non_empty_indices"]],
    )


######################## Checkpoint/Resume Functions ###########################


def _trt_plan_identity(model_path: Path, device_names: set[str]) -> list[dict[str, Any]]:
    plans = []
    for device_name in sorted(device_names):
        plan_path = plan_path_for_device(model_path, device_name).resolve()
        if not plan_path.is_file():
            raise FileNotFoundError(
                f"TensorRT plan required. Expected plan at {plan_path} for CUDA device "
                f"'{device_name}'."
            )
        plan_stat = plan_path.stat()
        plans.append(
            {
                "device_name": device_name,
                "path": str(plan_path),
                "sha256": file_sha256(plan_path),
                "size": plan_stat.st_size,
                "mtime_ns": plan_stat.st_mtime_ns,
            }
        )
    return plans


def _input_provenance_identity(
    input_path: Path,
    *,
    expected_schema: dict[str, Any],
) -> list[dict[str, str]]:
    candidates = (input_path.with_suffix(".done"), input_path.parent / "manifest.json")
    provenance = []
    for path in candidates:
        if not path.is_file():
            continue
        if path.stat().st_size == 0:
            raise ValueError(f"Input provenance file {path} is empty.")
        if path.name == "manifest.json":
            try:
                manifest = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid input manifest {path}") from exc
            if not isinstance(manifest, dict):
                raise ValueError(f"Invalid input manifest {path}: expected a JSON object.")
            for field in ("shape", "chunks", "dtype"):
                if manifest.get(field) != expected_schema[field]:
                    raise ValueError(
                        f"Input manifest {path} does not match input {field}: "
                        f"{manifest.get(field)!r} != {expected_schema[field]!r}"
                    )
        provenance.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
        )
    if not provenance:
        raise RuntimeError(
            f"Input {input_path} has no trusted completion marker or sibling manifest.json; "
            "refusing to create resumable state for mutable data."
        )
    return provenance


def _runtime_artifact_identity(
    model_path: Path,
    input_provenance: list[dict[str, str]],
    *,
    max_devices: int | None = None,
) -> dict[str, Any]:
    """Bind resume state to the plans and source code that execute inference."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is not available. TensorRT requires a GPU.")

    device_count = torch.cuda.device_count()
    if max_devices is not None:
        if max_devices < 1:
            raise ValueError("n_workers must be at least 1 when LocalCUDACluster is enabled.")
        device_count = min(device_count, max_devices)

    return {
        "pipeline_revision": 1,
        "squisher_segment_version": version("squisher-segment"),
        "trt_plans": _trt_plan_identity(
            model_path,
            {torch.cuda.get_device_name(index) for index in range(device_count)},
        ),
        "source_sha256": runtime_source_sha256(),
        "input_provenance": input_provenance,
    }


def _normalize_for_comparison(obj: Any) -> Any:
    """Normalize objects for comparison (convert numpy arrays to lists)."""
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


def _identity_digest(identity: dict[str, Any]) -> str:
    normalized = _normalize_for_comparison(identity)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _preprocessing_identity(
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "callable": f"{function.__module__}.{function.__qualname__}",
            "kwargs": kwargs,
        }
        for function, kwargs in preprocessing_steps
    ]


def _block_selection_policy(assume_nonempty: bool) -> str:
    return "cache-or-all" if assume_nonempty else "cache-or-scan"


def _nonempty_cache_key(run_identity: dict[str, Any]) -> str:
    """Bind observed occupancy to inputs and tiling, independent of fallback policy."""
    cache_identity = _normalize_for_comparison(run_identity)
    cache_identity.pop("block_selection", None)
    return _identity_digest(cache_identity)


def _mask_identity(mask: NDArray[Any] | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    contiguous = np.ascontiguousarray(mask)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
    }


def _validate_runtime_artifacts(runtime_artifacts: dict[str, Any]) -> None:
    required_nonempty = ("trt_plans", "source_sha256", "input_provenance")
    missing = [name for name in required_nonempty if not runtime_artifacts.get(name)]
    if missing:
        raise ValueError(f"runtime_artifacts is missing required non-empty fields: {missing}")
    required_plan_fields = {"device_name", "path", "sha256", "size", "mtime_ns"}
    if not all(
        isinstance(plan, dict)
        and required_plan_fields <= plan.keys()
        and all(plan[field] for field in ("device_name", "path", "sha256"))
        and isinstance(plan["size"], int)
        and isinstance(plan["mtime_ns"], int)
        for plan in runtime_artifacts["trt_plans"]
    ):
        raise ValueError("runtime_artifacts contains an incomplete TensorRT plan identity.")
    if not isinstance(runtime_artifacts["source_sha256"], dict) or not all(
        isinstance(name, str) and isinstance(digest, str) and digest
        for name, digest in runtime_artifacts["source_sha256"].items()
    ):
        raise ValueError("runtime_artifacts source_sha256 must be a mapping.")
    if not all(
        isinstance(item, dict) and {"path", "sha256"} <= item.keys()
        for item in runtime_artifacts["input_provenance"]
    ):
        raise ValueError("runtime_artifacts contains incomplete input provenance.")


def _validate_run_identity_structure(run_identity: dict[str, Any]) -> dict[str, Any]:
    required = {
        "input",
        "model_sha256",
        "model_kwargs",
        "eval_kwargs",
        "blocksize",
        "preprocessing_steps",
        "runtime_artifacts",
    }
    missing = sorted(required - run_identity.keys())
    if run_identity.get("schema_version") != 3 or missing:
        raise ValueError(
            f"run_identity must be a complete schema-3 artifact-bound identity; missing {missing}."
        )
    if not isinstance(run_identity["input"], dict) or not run_identity["input"]:
        raise ValueError("run_identity input must be a non-empty mapping.")
    if not isinstance(run_identity["model_sha256"], str) or not run_identity["model_sha256"]:
        raise ValueError("run_identity model_sha256 must be non-empty.")
    if not isinstance(run_identity["model_kwargs"], dict) or not isinstance(
        run_identity["eval_kwargs"], dict
    ):
        raise ValueError("run_identity model_kwargs and eval_kwargs must be mappings.")
    runtime_artifacts = run_identity["runtime_artifacts"]
    if not isinstance(runtime_artifacts, dict):
        raise ValueError("run_identity runtime_artifacts must be a mapping.")
    _validate_runtime_artifacts(runtime_artifacts)
    return runtime_artifacts


def _build_run_identity(
    *,
    input_identity: dict[str, Any],
    channel_indices: tuple[int, ...],
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    blocksize: tuple[int, ...],
    overlap: int,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]],
    assume_nonempty: bool,
    mask: NDArray[Any] | None,
    runtime_artifacts: dict[str, Any],
) -> dict[str, Any]:
    _validate_runtime_artifacts(runtime_artifacts)
    model_path = Path(model_kwargs["pretrained_model"])
    return _normalize_for_comparison(
        {
            "schema_version": 3,
            "input": input_identity,
            "channel_indices": list(channel_indices),
            "model_sha256": file_sha256(model_path),
            "model_kwargs": model_kwargs,
            "eval_kwargs": eval_kwargs,
            "blocksize": list(blocksize),
            "overlap": overlap,
            "block_selection": _block_selection_policy(assume_nonempty),
            "mask": _mask_identity(mask),
            "preprocessing_steps": _preprocessing_identity(preprocessing_steps),
            "cellpose_version": version("cellpose"),
            "runtime_artifacts": runtime_artifacts,
        }
    )


def save_run_config(
    path: Path,
    run_identity: dict[str, Any],
) -> None:
    payload = {
        "run_identity": _normalize_for_comparison(run_identity),
        "created_at": datetime.datetime.now().isoformat(),
    }
    atomic_write_text(path, json.dumps(payload, indent=2, cls=NumpyEncoder))


def load_run_identity(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid run configuration at {path}") from exc
    if not isinstance(payload, dict) or "run_identity" not in payload:
        raise ValueError(f"Invalid run configuration at {path}")
    identity = payload["run_identity"]
    if not isinstance(identity, dict):
        raise ValueError(f"Invalid run identity at {path}")
    return identity


def validate_run_config(path: Path, run_identity: dict[str, Any]) -> None:
    saved = load_run_identity(path)
    current = _normalize_for_comparison(run_identity)
    if saved != current:
        raise ValueError(
            "Cannot resume: input, channels, model, or evaluation configuration changed. "
            "Use --overwrite to start a new run."
        )


def completion_marker_path(output_path: Path) -> Path:
    return output_path.with_suffix(".done")


def _zarr_schema(array: zarr.Array) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "dtype": str(array.dtype),
    }


def _open_blank_temp_zarr(
    path: Path,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> zarr.Array:
    """Adopt only the metadata-only store from a pre-config initialization stop."""
    array = zarr.open_array(path, mode="r+")
    expected = {"shape": list(shape), "chunks": list(chunks), "dtype": "uint32"}
    if _zarr_schema(array) != expected:
        raise RuntimeError(f"Temporary Zarr {path} has an unexpected schema; use --overwrite.")
    metadata_names = {"zarr.json", ".zarray", ".zattrs", ".zgroup"}
    if any(
        child.is_file() and child.name not in metadata_names
        for child in path.rglob("*")
    ):
        raise RuntimeError(
            f"Temporary Zarr {path} contains data without a run configuration; use --overwrite."
        )
    return array


def write_completion_marker(output_path: Path, run_identity: dict[str, Any]) -> None:
    marker = completion_marker_path(output_path)
    output = zarr.open_array(output_path, mode="r")
    atomic_write_text(
        marker,
        json.dumps(
            {
                "run_key": _identity_digest(run_identity),
                "run_identity": _normalize_for_comparison(run_identity),
                "output": _zarr_schema(output),
                "completed_at": datetime.datetime.now().isoformat(),
            },
            indent=2,
        ),
    )


def completed_run_matches(output_path: Path, run_identity: dict[str, Any]) -> bool:
    marker = completion_marker_path(output_path)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text())
    except json.JSONDecodeError as exc:
        if promoted_output_matches(output_path, run_identity):
            return False
        raise RuntimeError(f"Invalid completion marker {marker}; use --overwrite to replace it.") from exc
    if not isinstance(payload, dict):
        if promoted_output_matches(output_path, run_identity):
            return False
        raise RuntimeError(f"Invalid completion marker {marker}; use --overwrite to replace it.")
    if payload.get("run_key") != _identity_digest(run_identity):
        if promoted_output_matches(output_path, run_identity):
            return False
        raise FileExistsError(
            f"Completed output {output_path} belongs to a different run; use --overwrite to replace it."
        )
    if not output_path.exists():
        raise RuntimeError(f"Completion marker {marker} exists but output {output_path} is missing.")
    expected_schema = payload.get("output")
    actual_schema = _zarr_schema(zarr.open_array(output_path, mode="r"))
    if expected_schema != actual_schema:
        raise RuntimeError(
            f"Completed output {output_path} does not match the schema recorded in {marker}; "
            "use --overwrite to replace it."
        )
    return True


def promoted_output_matches(output_path: Path, run_identity: dict[str, Any]) -> bool:
    """Recognize a fully promoted output if shutdown preceded its done marker."""
    if not output_path.exists():
        return False
    output = zarr.open_array(output_path, mode="r")
    return (
        output.attrs.get("squisher_run_key") == _identity_digest(run_identity)
        and output.attrs.get("squisher_output_schema") == _zarr_schema(output)
    )


def load_checkpoint(path: Path) -> set[tuple[int, ...]]:
    """Load completed block indices from checkpoint file."""
    completed: set[tuple[int, ...]] = set()
    if not path.exists():
        return completed
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                completed.add(tuple(entry["index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning(f"Skipping corrupted checkpoint line {i}: {e}")
    return completed


def append_checkpoint(
    checkpoint_path: Path,
    block_index: tuple[int, ...],
    worker_name: str,
    duration_s: float,
    n_masks: int,
) -> None:
    """Append and flush one driver-owned checkpoint entry."""

    entry = {
        "index": list(block_index),
        "ts": datetime.datetime.now().isoformat(),
        "worker": worker_name,
        "duration_s": round(duration_s, 2),
        "n_masks": n_masks,
    }
    encoded = (json.dumps(entry) + "\n").encode()
    with open(checkpoint_path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())


def _gpu_probe() -> dict[str, Any]:
    """Executed on workers to report CUDA visibility and current device."""
    import distributed as _dist
    import torch  # type: ignore

    # Be defensive: during startup/shutdown (or in some LocalCluster thread modes)
    # `get_worker()` can raise. If we raise here, Dask may fail to serialize the
    # exception when Rich logging is enabled (ContextVar pickling), which surfaces
    # as a noisy "contextvars cannot be pickled" error.
    try:
        worker = getattr(_dist.get_worker(), "name", "unknown")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        torch_dev = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
        return {
            "worker": worker,
            "cuda_visible_devices": cuda_visible,
            "torch_current_device": torch_dev,
        }
    except Exception as exc:  # pragma: no cover - defensive against early startup failures
        return {
            "worker": "unknown",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_current_device": None,
            "error": repr(exc),
        }


######################## File format functions ################################
def numpy_array_to_zarr(write_path: Path | str, array: NDArray[Any], chunks: tuple[int, ...]) -> zarr.Array:
    """
    Store an in memory numpy array to disk as a chunked Zarr array

    Parameters
    ----------
    write_path : string
        Filepath where Zarr array will be created

    array : numpy.ndarray
        The already loaded in-memory numpy array to store as zarr

    chunks : tuple, must be array.ndim length
        How the array will be chunked in the Zarr array

    Returns
    -------
    zarr.core.Array
        A read+write reference to the zarr array on disk
    """

    zarr_array = create_zarr_array(
        write_path,
        shape=tuple(int(s) for s in array.shape),
        chunks=chunks,
        dtype=array.dtype,
        overwrite=True,
        codecs=label_zarr_codecs(array.dtype) if np.dtype(array.dtype) == np.dtype(np.uint32) else None,
    )
    zarr_array[...] = array
    return zarr_array


def wrap_folder_of_tiffs(
    filename_pattern: str,
    block_index_pattern: str = r"_(Z)(\d+)(Y)(\d+)(X)(\d+)",
) -> zarr.Array:
    """
    Wrap a folder of tiff files with a zarr array without duplicating data.
    Tiff files must all contain images with the same shape and data type.
    Tiff file names must contain a pattern indicating where individual files
    lie in the block grid.

    Distributed computing requires parallel access to small regions of your
    image from different processes. This is best accomplished with chunked
    file formats like Zarr and N5. This function can accommodate a folder of
    tiff files, but it is not equivalent to reformating your data as Zarr or N5.
    If your individual tiff files/tiles are huge, distributed performance will
    be poor or not work at all.

    It does not make sense to use this function if you have only one tiff file.
    That tiff file will become the only chunk in the zarr array, which means all
    workers will have to load the entire image to fetch their crop of data anyway.
    If you have a single tiff image, you should just reformat it with the
    numpy_array_to_zarr function. Single tiff files too large to fit into system
    memory are not be supported.

    Parameters
    ----------
    filename_pattern : string
        A glob pattern that will match all needed tif files

    block_index_pattern : regular expression string
        A regular expression pattern that indicates how to parse tiff filenames
        to determine where each tiff file lies in the overall block grid
        The default pattern assumes filenames like the following:
            {any_prefix}_Z000Y000X000{any_suffix}
            {any_prefix}_Z000Y000X001{any_suffix}
            ... and so on

    Returns
    -------
    zarr.core.Array
    """

    # define function to read individual files
    def imread(fname: str) -> NDArray[Any]:
        with open(fname, "rb") as fh:
            return imagecodecs.tiff_decode(fh.read(), index=None)

    # create zarr store, open it as zarr array and return
    store = tifffile.imread(
        filename_pattern,
        aszarr=True,
        imread=imread,
        pattern=block_index_pattern,
        axestiled={x: x for x in range(3)},
    )
    return zarr.open(store=store)


######################## Cluster related functions ############################


def format_slice(s: slice | tuple[slice, ...]) -> str:
    """Format a slice or tuple of slices as a human-readable string."""
    if isinstance(s, tuple):
        return ",".join(format_slice(item) for item in s)
    if not isinstance(s, slice):  # type: ignore
        return str(s)

    start, stop, step = s.start, s.stop, s.step
    parts = [
        "" if start in (None, 0) else str(start),
        "" if stop is None else str(stop),
        "" if step in (None, 1) else str(step),
    ]

    return ":".join(parts).rstrip(":")


def _resolve_channel_selection(
    input_zarr: zarr.Array,
    channels: str | None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    raw_names = input_zarr.attrs.get("key")
    if not isinstance(raw_names, (list, tuple)):
        raise ValueError("Input Zarr attribute 'key' must be a list of channel names.")
    names = tuple(str(name) for name in raw_names)
    if len(names) != input_zarr.shape[-1]:
        raise ValueError(
            f"Input Zarr has {input_zarr.shape[-1]} channels but attribute 'key' has {len(names)} names."
        )
    if len(set(names)) != len(names):
        raise ValueError("Input Zarr channel names in attribute 'key' must be unique.")

    if channels is None:
        return tuple(range(len(names))), names

    selected_names = tuple(name.strip() for name in channels.split(","))
    if any(not name for name in selected_names):
        raise ValueError("--channels must contain one or more non-empty channel names.")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("--channels cannot select the same channel more than once.")
    missing = [name for name in selected_names if name not in names]
    if missing:
        raise ValueError(f"Channel names {missing} not found in {list(names)}")
    return tuple(names.index(name) for name in selected_names), selected_names


def _read_input_crop(
    input_zarr: zarr.Array,
    crop: tuple[slice, ...],
    channel_indices: tuple[int, ...],
) -> NDArray[Any]:
    selection = crop[:-1] + (list(channel_indices),)
    return np.asarray(input_zarr.get_orthogonal_selection(selection))


def _sam_processing_blocksize(
    *,
    n_channels: int,
    diameter: int | float,
    target_nz: int | None,
    target_ny: int | None,
    target_nx: int | None,
) -> tuple[int, int, int, int]:
    """Resolve a 3-D core whose haloed inference crop meets the tile targets."""
    nz = 2 if target_nz is None else target_nz
    ny = 2 if target_ny is None else target_ny
    nx = 6 if target_nx is None else target_nx
    Lz_inference, Ly_inference, Lx_inference = solve_internal_zyx_for_tiles(
        nz,
        ny,
        nx,
        bsize=256,
        tile_overlap=0.1,
    )
    scale_back = float(diameter) / 30.0
    distributed_overlap = int(diameter * 2)
    return (
        int(Lz_inference * scale_back) - 2 * distributed_overlap,
        int(Ly_inference * scale_back) - 2 * distributed_overlap,
        int(Lx_inference * scale_back) - 2 * distributed_overlap,
        n_channels,
    )


def _build_cellpose_eval_kwargs(
    *,
    diameter: int | float,
    normalization: dict[str, Any],
    ortho_weights: list[float] | None,
) -> dict[str, Any]:
    eval_kwargs: dict[str, Any] = {
        "diameter": diameter,
        "batch_size": 1,
        "normalize": normalization,
        "flow_threshold": 0,
        "cellprob_threshold": 0,
        "anisotropy": 1.0,
        "resample": False,
        "flow3D_smooth": 1.5,
        "niter": 1000,
        "do_3D": True,
        "min_size": 500,
        "channel_axis": 3,
        "use_kde_clustering": True,
        "z_axis": 0,
    }
    if ortho_weights is not None:
        eval_kwargs["ortho_weights"] = ortho_weights
    return eval_kwargs


def _segmentation_block_crops(
    input_shape: tuple[int, ...],
    blocksize: tuple[int, ...],
    overlap: int,
    mask: NDArray[Any] | None,
) -> tuple[list[tuple[int, ...]], list[tuple[slice, ...]]]:
    if len(input_shape) != 4 or len(blocksize) != 4:
        raise ValueError("Distributed segmentation requires ZYXC input and block sizes.")
    if input_shape[-1] != blocksize[-1]:
        raise ValueError("The selected channel axis must fit in exactly one block.")
    overlap_by_axis = np.array((overlap, overlap, overlap, 0), dtype=int)
    return get_block_crops(input_shape, np.asarray(blocksize), overlap_by_axis, mask)


def _check_block_has_data(
    crop: tuple[slice, ...],
    zarr_array: zarr.Array,
    selected_channels: tuple[int, ...],
    threshold: int = 0,
) -> bool:
    """Return whether a planned crop contains input above the threshold."""
    data_slice = _read_input_crop(zarr_array, crop, selected_channels)
    return bool(data_slice.any() if threshold == 0 else (data_slice > threshold).any())


def _select_input_blocks(
    *,
    client: Any,
    block_crops: list[tuple[slice, ...]],
    input_zarr: zarr.Array,
    channel_indices: tuple[int, ...],
    blocksize: tuple[int, ...],
    run_key: str,
    path_nonempty: Path,
    assume_nonempty: bool,
) -> list[int]:
    """Select planned blocks according to the run's explicit selection policy."""
    idxs = read_nonempty_cache(path_nonempty, blocksize, run_key)
    if idxs is not None:
        logger.info(f"Loaded cached non-empty block indices ({len(idxs)} entries) from {path_nonempty}.")
        return idxs

    if assume_nonempty:
        # This selection is deterministic from the tiling plan, so the run config
        # and checkpoint are sufficient for resume; a nonempty cache is unnecessary.
        logger.info(
            "Non-empty cache miss; assuming dense input and selecting every planned block "
            "without an input scan."
        )
        return list(range(len(block_crops)))

    logger.info("Non-empty cache miss or invalidated; re-scanning input for non-zero blocks.")
    check_futures = client.map(
        _check_block_has_data,
        block_crops,
        zarr_array=input_zarr,
        selected_channels=channel_indices,
        threshold=1,
    )

    total_tiles = len(check_futures)
    logger.info(f"Checking non-zero blocks: 0/{total_tiles}")
    non_zero_results: list[bool] = [True] * total_tiles
    future_to_index = {fut: i for i, fut in enumerate(check_futures)}
    with progress_bar(total_tiles) as submit:
        [fut.add_done_callback(submit) for fut in check_futures]
        for fut in distributed.as_completed(check_futures):
            i = future_to_index.get(fut)
            if i is None:
                logger.error("Non-zero block check produced an unknown future; treating as non-empty")
                continue
            try:
                non_zero_results[i] = bool(fut.result())
            except Exception as exc:
                logger.error(
                    f"Non-zero block check failed for block_check[{i}]: {exc!r}; treating as non-empty"
                )
    logger.info(f"Checked non-zero blocks: {total_tiles}/{total_tiles}")

    idxs = [i for i, is_non_zero in enumerate(non_zero_results) if is_non_zero]
    path_nonempty.parent.mkdir(parents=True, exist_ok=True)
    write_nonempty_cache(path_nonempty, blocksize, run_key, idxs)
    logger.info(f"Persisted {len(idxs)} non-empty block indices to {path_nonempty}")
    return idxs


######################## the function to run on each block ####################


# ----------------------- The main function -----------------------------------#
def process_block(
    block_index: tuple[int, ...],
    crop: tuple[slice, ...],
    input_zarr: zarr.Array,
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    blocksize: tuple[int, ...],
    overlap: int,
    output_zarr: zarr.Array,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]] = [],
    worker_logs_directory: str | None = None,
    test_mode: bool = False,
    stagger_seconds: float = 0.0,
    workers_per_gpu: int = 4,
    channel_indices: tuple[int, ...] | None = None,
) -> (
    tuple[NDArray[np.uint32], list[tuple[slice, ...]], NDArray[np.uint32]]
    | dict[str, Any]
):
    """
    Preprocess and segment one block with eventual merger in mind.

    Steps: read/preprocess/segment → remove overlaps → remap to global IDs
    → write to output_zarr.

    Parameters
    ----------
    preprocessing_steps : list of tuples
        Must be in the format: [(func, {'arg1': val1, ...}), ...]
        Each function must have signature: def F(image, ..., crop=None)
        The crop kwarg is injected automatically.

    test_mode : bool
        When True, returns (segmentation, boxes, box_ids) without writing
        to disk. Useful for testing individual blocks before full runs.

    Returns
    -------
    If test_mode=False: completion metadata for the driver checkpoint writer
    If test_mode=True: (segmentation_array, boxes, box_ids)
    """
    try:
        worker = distributed.get_worker()
    except ValueError:
        worker = None

    if worker is not None and getattr(worker, "_squisher_segment_fatal_error", False):
        raise RuntimeError("Worker previously hit a fatal error; refusing further work")

    if worker is not None:
        _apply_startup_stagger(stagger_seconds, workers_per_gpu)

    wlog = _get_worker_logger()
    worker_name = getattr(worker, "name", "local") if worker is not None else "local"
    start_time = time.perf_counter()
    wlog.info(f"Worker {worker_name} RUNNING BLOCK: {block_index}\tREGION: [{format_slice(crop)}]")

    try:
        segmentation_3d = read_preprocess_and_segment(
            input_zarr,
            crop,
            channel_indices,
            preprocessing_steps,
            model_kwargs,
            eval_kwargs,
            worker_logs_directory,
        )
        wlog.info(f"Block {block_index}: {np.max(segmentation_3d)} masks found.")

        spatial_crop_slices = crop[:-1]
        spatial_blocksize = blocksize[:-1]

        segmentation_trimmed_3d, crop_trimmed_3d = remove_overlaps(
            segmentation_3d,
            spatial_crop_slices,
            overlap,
            spatial_blocksize,
        )
        crop_trimmed_3d = tuple(crop_trimmed_3d)

        nblocks_3d = get_nblocks(input_zarr.shape[:-1], spatial_blocksize)
        block_index_3d = block_index[:-1]

        segmentation_global_3d, _ = global_segment_ids(
            segmentation_trimmed_3d, block_index_3d, nblocks_3d
        )

        if test_mode:
            boxes = bounding_boxes_in_global_coordinates(segmentation_trimmed_3d, crop_trimmed_3d)
            final_unique_ids_3d = np.unique(segmentation_global_3d)
            box_ids_for_this_block = final_unique_ids_3d[final_unique_ids_3d != 0]
            return (segmentation_global_3d, boxes, box_ids_for_this_block)

        output_zarr[crop_trimmed_3d] = segmentation_global_3d
        return {
            "index": block_index,
            "worker": worker_name,
            "duration_s": time.perf_counter() - start_time,
            "n_masks": int(np.max(segmentation_3d)),
        }
    except Exception as exc:
        wlog.exception(
            f"Worker {worker_name} FAILED BLOCK: {block_index}\tREGION: [{format_slice(crop)}]\n{exc!r}"
        )
        if worker is not None:
            _retire_worker_after_error(reason=f"squisher_segment process_block failed: {exc!r}")
        raise


# ----------------------- component functions ---------------------------------#


def read_preprocess_and_segment(
    input_zarr: zarr.Array,
    crop: tuple[slice, ...],
    channel_indices: tuple[int, ...] | None,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]],
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    worker_logs_directory: str | None,
) -> NDArray[np.uint32]:
    """
    Read block, apply preprocessing pipeline, and run Cellpose segmentation.

    preprocessing_steps format: [(func, kwargs_dict), ...]
    Each func must accept (image, ..., crop=None). The crop kwarg is injected.
    """
    if preprocessing_steps is None:
        preprocessing_steps = []

    image = input_zarr[crop] if channel_indices is None else _read_input_crop(input_zarr, crop, channel_indices)
    for pp_step in preprocessing_steps:
        pp_step[1]["crop"] = crop
        image = pp_step[0](image, **pp_step[1])
    log_file = None
    if worker_logs_directory is not None:
        log_file = f"dask_worker_{distributed.get_worker().name}.log"
        log_file = pathlib.Path(worker_logs_directory).joinpath(log_file)
    cellpose.io.logger_setup(stdout_file_replacement=log_file)

    model = get_cached_model(model_kwargs)
    outputs = model.eval(image, **eval_kwargs)
    masks = outputs[0].astype(np.uint32)
    del outputs

    # Multiple worker processes cannot reuse another process's cached blocks.
    # Return inactive allocations to CUDA after each complete Cellpose block.
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    return masks


def _wait_for_futures_collect_errors(
    *,
    futures: list[distributed.Future],
    future_labels: dict[distributed.Future, str],
    stage: str,
    log: logging.Logger,
    checkpoint_path: Path,
) -> list[str]:
    failures: list[str] = []
    for fut in distributed.as_completed(futures):
        label = future_labels.get(fut, getattr(fut, "key", "<unknown>"))
        try:
            result = fut.result()
            if not isinstance(result, dict):
                raise TypeError(f"{label} returned invalid completion metadata")
            append_checkpoint(
                checkpoint_path,
                tuple(result["index"]),
                str(result["worker"]),
                float(result["duration_s"]),
                int(result["n_masks"]),
            )
        except Exception as exc:
            msg = f"{label}: {exc!r}"
            log.error(f"{stage} task failed: {msg}")
            failures.append(msg)
    return failures


######################## Distributed Cellpose #################################


# ----------------------- The main function -----------------------------------#
@cluster
def distributed_eval(
    input_zarr: zarr.Array,
    blocksize: tuple[int, ...],
    write_path: Path | str,
    mask: NDArray[Any] | None = None,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    eval_kwargs: dict[str, Any] | None = None,
    cluster: myLocalCluster | None = None,
    cluster_kwargs: dict[str, Any] | None = None,
    temporary_directory: Path | None = None,
    cellpose_only: bool = False,
    stagger_seconds: float = 0.0,
    channel_indices: tuple[int, ...] | None = None,
    run_identity: dict[str, Any] | None = None,
    assume_nonempty: bool = False,
    overwrite_output: bool = False,
) -> tuple[zarr.Array, list[tuple[slice, ...]]] | None:
    """
    Evaluate a cellpose model on overlapping blocks of a big image.
    Distributed over workstation or cluster resources with Dask.
    Optionally run preprocessing steps on the blocks before running cellpose.
    Optionally use a mask to ignore background regions in image.
    Either cluster or cluster_kwargs parameter must be set to a
    non-default value; please read these parameter descriptions below.
    If using cluster_kwargs, the workstation and Janelia LSF cluster cases
    are distinguished by the arguments present in the dictionary.

    PC/Mac/Linux workstations and the Janelia LSF cluster are supported;
    running on a different institute cluster will require implementing your
    own dask cluster class. Look at the JaneliaLSFCluster class in this
    module as an example, also look at the dask_jobqueue library. A PR with
    a solid start is the right way to get help running this on your own
    institute cluster.

    If running on a workstation, please read the docstring for the
    LocalCluster class defined in this module. That will tell you what to
    put in the cluster_kwargs dictionary. If using the Janelia cluster,
    please read the docstring for the janeliaLSFCluster class in this module.

    Parameters
    ----------
    input_zarr : zarr.core.Array
        A zarr.core.Array instance containing the image data you want to
        segment.

    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    write_path : string
        The location of a zarr file on disk where you'd like to write your results

    mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. lower) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. It is assumed that the domain of the input_zarr image data
        and the mask is the same in physical units, but they may be on
        different sampling/voxel grids.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image blocks before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    model_kwargs : dict (default: {})
        Arguments passed to PackedCellposeModel

    eval_kwargs : dict (default: {})
        Arguments passed to PackedCellposeModel.eval

    cluster : A dask cluster object (default: None)
        Only set if you have constructed your own static cluster. The default
        behavior is to construct a dask cluster for the duration of this function,
        then close it when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments used to parameterize your cluster.
        If you are running locally, see the docstring for the myLocalCluster
        class in this module. If you are running on the Janelia LSF cluster, see
        the docstring for the janeliaLSFCluster class in this module. If you are
        running on a different institute cluster, you may need to implement
        a dask cluster object that conforms to the requirements of your cluster.

    temporary_directory : string (default: None)
        Temporary files are created during segmentation. The temporary files
        will be in their own folder within the temporary_directory. The default
        is the current directory. Temporary files are removed if the function
        completes successfully.

    Returns
    -------
    Two values are returned:
    (1) A reference to the zarr array on disk containing the stitched cellpose
        segments for your entire image
    (2) Bounding boxes for every segment. This is a list of tuples of slices:
        [(slice(z1, z2), slice(y1, y2), slice(x1, x2)), ...]
        The list is sorted according to segment ID. That is the smallest segment
        ID is the first tuple in the list, the largest segment ID is the last
        tuple in the list.
    """
    overall_start = time.perf_counter()

    if preprocessing_steps is None:
        preprocessing_steps = []
    if model_kwargs is None:
        model_kwargs = {}
    if eval_kwargs is None:
        eval_kwargs = {}
    if cluster_kwargs is None:
        cluster_kwargs = {}
    if temporary_directory is None:
        temporary_directory = Path(write_path).parent / "cellpose_temp"
    temporary_directory = Path(temporary_directory)
    if input_zarr.ndim != 4:
        raise ValueError(f"Distributed segmentation requires ZYXC input; got shape {input_zarr.shape}.")
    if channel_indices is None:
        channel_indices = tuple(range(input_zarr.shape[-1]))
    if not channel_indices or len(set(channel_indices)) != len(channel_indices):
        raise ValueError("channel_indices must contain unique selected channels.")
    if min(channel_indices) < 0 or max(channel_indices) >= input_zarr.shape[-1]:
        raise ValueError(f"channel_indices {channel_indices} are invalid for shape {input_zarr.shape}.")
    selected_shape = input_zarr.shape[:-1] + (len(channel_indices),)
    if blocksize[-1] != len(channel_indices):
        raise ValueError("blocksize channel extent must equal the number of selected channels.")
    block_selection = _block_selection_policy(assume_nonempty)
    mask_identity = _mask_identity(mask)
    if run_identity is None:
        raise ValueError(
            "run_identity is required; build it with _build_run_identity so resume state "
            "is bound to model, TensorRT, input-provenance, and source artifacts."
        )
    if run_identity.get("block_selection") != block_selection:
        raise ValueError(
            "run_identity block_selection does not match assume_nonempty; "
            "build the identity with the same block-selection policy."
        )
    elif run_identity.get("mask") != mask_identity:
        raise ValueError(
            "run_identity mask identity does not match the supplied mask; "
            "build the identity from the same mask."
        )
    runtime_artifacts = _validate_run_identity_structure(run_identity)
    trt_plans = runtime_artifacts.get("trt_plans")
    if not isinstance(trt_plans, list):
        raise ValueError("run_identity runtime_artifacts must include TensorRT plans.")
    source_sha256 = runtime_artifacts["source_sha256"]

    run_config_path = temporary_directory / "run_config.json"
    is_resume = run_config_path.exists()
    blank_temp_zarr: zarr.Array | None = None
    if is_resume:
        validate_run_config(run_config_path, run_identity)

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    worker_logs_dirname = f"dask_worker_logs_{timestamp}"
    base_dir = temporary_directory.parent
    worker_logs_dir = base_dir / worker_logs_dirname
    worker_logs_dir.mkdir(parents=True, exist_ok=True)

    if "diameter" not in eval_kwargs:
        raise ValueError("Diameter must be set in eval_kwargs")

    overlap = int(eval_kwargs["diameter"] * 2)
    block_indices, block_crops = _segmentation_block_crops(
        selected_shape,
        blocksize,
        overlap,
        mask,
    )
    assert cluster is not None

    # GPU preflight probe to confirm worker pinning
    try:
        cluster.client.wait_for_workers(1, timeout=30)
    except Exception as e:
        logger.warning(f"GPU probe skipped: workers not ready ({e!r})")
    else:
        try:
            probe = cluster.client.run(_gpu_probe)
            if any("error" in info for info in probe.values()):
                logger.warning(f"GPU probe returned errors: {probe}")
            else:
                logger.info(f"GPU probe results: {probe}")
        except Exception as e:
            logger.warning(f"GPU probe failed: {e!r}")

    # Keep observed occupancy outside the policy-specific resumable run directory
    # so it survives successful cleanup and --overwrite mode switches.
    path_nonempty = temporary_directory.parent / "nonempty.json"
    run_key = _nonempty_cache_key(run_identity)
    idxs = _select_input_blocks(
        client=cluster.client,
        block_crops=block_crops,
        input_zarr=input_zarr,
        channel_indices=channel_indices,
        blocksize=blocksize,
        run_key=run_key,
        path_nonempty=path_nonempty,
        assume_nonempty=assume_nonempty,
    )

    final_block_indices, final_block_crops = (
        [block_indices[i] for i in idxs],
        [block_crops[i] for i in idxs],
    )
    total_non_empty_blocks = len(final_block_indices)
    del block_indices, block_crops

    logger.info(f"Selected {len(final_block_indices)} blocks for segmentation.")

    output_shape = input_zarr.shape[:-1]
    output_blocksize = blocksize[:-1]

    temporary_directory.mkdir(parents=True, exist_ok=True)
    assert temporary_directory.exists()
    temp_zarr_path = temporary_directory / "segmentation_unstitched.zarr"
    checkpoint_path = temporary_directory / "checkpoint.jsonl"

    if is_resume:
        if not temp_zarr_path.exists():
            raise RuntimeError(f"Cannot resume: temp_zarr missing at {temp_zarr_path}")
        completed_indices = load_checkpoint(checkpoint_path)
        logger.info(
            f"Resuming: {len(completed_indices)} of {len(final_block_indices)} blocks already completed"
        )
    else:
        if temp_zarr_path.exists():
            blank_temp_zarr = _open_blank_temp_zarr(
                temp_zarr_path,
                shape=output_shape,
                chunks=output_blocksize,
            )
        stale_paths = [
            path
            for path in (
                checkpoint_path,
                temporary_directory / "intermediate_state.npz",
            )
            if path.exists()
        ]
        if stale_paths:
            raise RuntimeError(
                "Cannot start a fresh run because temporary artifacts exist without a run "
                f"configuration: {stale_paths}. Use --overwrite to replace them."
            )
        completed_indices = set()

    # Filter to remaining blocks
    remaining_block_indices = []
    remaining_block_crops = []
    for idx, crop in zip(final_block_indices, final_block_crops):
        if tuple(idx) not in completed_indices:
            remaining_block_indices.append(idx)
            remaining_block_crops.append(crop)

    logger.info(
        f"Blocks to process: {len(remaining_block_indices)} (skipped {len(completed_indices)} already completed)"
    )

    if is_resume:
        temp_zarr = zarr.open(temp_zarr_path, mode="r+")
    elif blank_temp_zarr is not None:
        temp_zarr = blank_temp_zarr
    else:
        temp_zarr = create_zarr_array(
            temp_zarr_path,
            shape=output_shape,  # Use 3D shape
            chunks=output_blocksize,  # Use 3D chunks
            dtype=np.uint32,
            overwrite=True,
            codecs=label_zarr_codecs(np.uint32),
        )
        save_run_config(run_config_path, run_identity)
        logger.info(f"Fresh run - saved config to {run_config_path}")

    if not remaining_block_indices:
        logger.info("All blocks already completed, proceeding to merge")
    else:
        # Register plugin to cache model on workers (initialized once per worker)
        plugin = CellposeModelPlugin(
            model_kwargs,
            artifact_key=_identity_digest(run_identity),
            trt_plans=trt_plans,
            source_sha256=source_sha256,
        )
        cluster.client.register_plugin(plugin)

        workers_per_gpu = cluster_kwargs.get("workers_per_gpu", 4) if cluster_kwargs else 4
        futures = cluster.client.map(
            process_block,
            remaining_block_indices,
            remaining_block_crops,
            input_zarr=input_zarr,
            preprocessing_steps=preprocessing_steps,
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            blocksize=blocksize,
            overlap=overlap,
            output_zarr=temp_zarr,
            worker_logs_directory=str(worker_logs_dir),
            stagger_seconds=stagger_seconds,
            workers_per_gpu=workers_per_gpu,
            channel_indices=channel_indices,
        )

        with progress_bar(len(remaining_block_indices)) as submit:
            [fut.add_done_callback(submit) for fut in futures]
            future_labels = {
                fut: f"block={idx}"
                for fut, idx in zip(futures, remaining_block_indices, strict=True)
            }
            failures = _wait_for_futures_collect_errors(
                futures=futures,
                future_labels=future_labels,
                stage="Segmentation",
                log=logger,
                checkpoint_path=checkpoint_path,
            )

        if failures:
            preview = "\n".join(failures[:10])
            raise RuntimeError(
                f"Segmentation failed for {len(failures)} blocks; run can be resumed after fixing the issue.\n"
                f"First failures:\n{preview}"
            )

    logger.info("Computing faces and bounding boxes from temp_zarr...")
    results = []
    for block_crop in final_block_crops:
        spatial_crop = block_crop[:-1]
        spatial_blocksize = blocksize[:-1]
        trimmed_crop = []
        for slc, bs in zip(spatial_crop, spatial_blocksize):
            start = slc.start if slc.start == 0 else slc.start + overlap
            stop = min(start + bs, slc.stop)
            trimmed_crop.append(slice(start, stop))
        trimmed_crop = tuple(trimmed_crop)

        seg_block = temp_zarr[trimmed_crop]
        faces = block_faces(seg_block, shrink=True)
        local_labels, box_ids = decode_block_global_labels(seg_block)
        boxes = bounding_boxes_in_global_coordinates(local_labels, trimmed_crop)
        results.append((faces, boxes, box_ids))

    if isinstance(cluster, dask_jobqueue.core.JobQueueCluster):
        cluster.scale(0)

    # Filter to non-empty blocks only
    faces_list, boxes_list, box_ids_list, non_empty_indices = [], [], [], []
    for i, (faces, boxes, box_ids) in enumerate(results):
        if len(box_ids) > 0:
            faces_list.append(faces)
            boxes_list.append(boxes)
            box_ids_list.append(box_ids)
            non_empty_indices.append(final_block_indices[i])

    # Save intermediate state for potential separate stitching
    _save_intermediate_state(
        temporary_directory,
        faces_list,
        boxes_list,
        box_ids_list,
        non_empty_indices,
    )

    if cellpose_only:
        logger.info("Cellpose-only mode: skipping merge phase")
        logger.info(f"Intermediate results saved to: {temporary_directory}")
        _log_slurm_tile_summary(
            total_non_empty_blocks,
            len(remaining_block_indices),
            time.perf_counter() - overall_start,
        )
        return None

    # stitching step is cheap, we should release gpus and use small workers
    if isinstance(cluster, dask_jobqueue.core.JobQueueCluster):
        cluster.change_worker_attributes(
            min_workers=cluster.locals_store["min_workers"],
            max_workers=12,
            ncpus=1,
            memory="32GB",
            mem=int(32e9),
            queue=None,
            job_extra_directives=[],
        )
        cluster.scale(32)

    final_seg_zarr, merged_boxes = _stitch_precomputed(
        block_indices=non_empty_indices,
        faces_list=faces_list,
        boxes_list=boxes_list,
        box_ids_list=box_ids_list,
        temp_zarr=temp_zarr,
        temp_dir=temporary_directory,
        output_path=Path(write_path),
        run_identity=run_identity,
        overwrite_output=overwrite_output,
    )

    _log_slurm_tile_summary(
        total_non_empty_blocks,
        len(remaining_block_indices),
        time.perf_counter() - overall_start,
    )

    return final_seg_zarr, merged_boxes


def _staged_output_path(output_path: Path) -> Path:
    suffix = hashlib.sha256(str(output_path.resolve()).encode()).hexdigest()[:8]
    return output_path.with_name(f".stitch-{suffix}.zarr")


def _backup_output_path(output_path: Path, run_identity: dict[str, Any]) -> Path:
    output_key = hashlib.sha256(str(output_path.resolve()).encode()).hexdigest()[:8]
    run_key = _identity_digest(run_identity)[:8]
    return output_path.with_name(f".backup-{output_key}-{run_key}.zarr")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _restore_output_backup(output_path: Path, run_identity: dict[str, Any]) -> Path:
    """Restore the prior final if shutdown occurred between overwrite renames."""
    backup_path = _backup_output_path(output_path, run_identity)
    if backup_path.exists() and not output_path.exists():
        os.replace(backup_path, output_path)
        _fsync_directory(output_path.parent)
    return backup_path


def _remove_output_backup(backup_path: Path) -> None:
    if not backup_path.exists():
        return
    if backup_path.is_dir():
        shutil.rmtree(backup_path)
    else:
        backup_path.unlink()
    _fsync_directory(backup_path.parent)


def _remove_matching_temp(temp_dir: Path, run_identity: dict[str, Any]) -> None:
    run_config_path = temp_dir / "run_config.json"
    if not run_config_path.exists():
        return
    if load_run_identity(run_config_path) != _normalize_for_comparison(run_identity):
        logger.warning(f"Leaving unrelated temporary state at {temp_dir}")
        return
    shutil.rmtree(temp_dir)


def _stitch_precomputed(
    *,
    block_indices: list[tuple[int, ...]],
    faces_list: list,
    boxes_list: list,
    box_ids_list: list,
    temp_zarr: zarr.Array,
    temp_dir: Path,
    output_path: Path,
    run_identity: dict[str, Any],
    overwrite_output: bool,
) -> tuple[zarr.Array, list[tuple[slice, ...]]]:
    """Write a complete staged Zarr and atomically expose it at the final path."""
    if output_path.exists() and not overwrite_output:
        raise FileExistsError(f"Output {output_path} already exists; use --overwrite to replace it.")

    staged_path = _staged_output_path(output_path)
    _, new_labeling = stitch_labels(
        block_indices=block_indices,
        faces_list=faces_list,
        box_ids_list=box_ids_list,
        temp_zarr=temp_zarr,
        write_path=staged_path,
        lut_path=temp_dir / "new_labeling.npy",
        pre_shrunk=True,
    )
    staged_zarr = zarr.open_array(staged_path, mode="r+")
    merged_boxes = merge_boxes_for_labels(boxes_list, box_ids_list, new_labeling)
    if _zarr_schema(staged_zarr) != _zarr_schema(temp_zarr):
        raise RuntimeError(
            f"Staged output {staged_path} schema does not match temporary segmentation."
        )
    staged_zarr.attrs["squisher_run_key"] = _identity_digest(run_identity)
    staged_zarr.attrs["squisher_output_schema"] = _zarr_schema(staged_zarr)

    backup_path = _backup_output_path(output_path, run_identity)
    if backup_path.exists():
        raise RuntimeError(
            f"Cannot promote {staged_path}: unresolved output backup exists at {backup_path}."
        )
    if output_path.exists():
        os.replace(output_path, backup_path)
        _fsync_directory(output_path.parent)
    try:
        os.replace(staged_path, output_path)
    except OSError:
        if backup_path.exists() and not output_path.exists():
            os.replace(backup_path, output_path)
            _fsync_directory(output_path.parent)
        raise
    _fsync_directory(output_path.parent)
    return zarr.open_array(output_path, mode="r"), merged_boxes


def stitch_segmentation(
    temp_dir: Path,
    output_path: Path,
    *,
    overwrite_output: bool = False,
) -> tuple[zarr.Array, list[tuple[slice, ...]]]:
    """
    Run only the stitching/merging phase on pre-computed cellpose results.

    Parameters
    ----------
    temp_dir : Path
        Directory containing segmentation_unstitched.zarr and intermediate_state.npz
    output_path : Path
        Path for final stitched zarr output

    Returns
    -------
    tuple[zarr.Array, NDArray]
        Final segmentation zarr and merged bounding boxes
    """
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    temp_zarr = zarr.open(temp_dir / "segmentation_unstitched.zarr", mode="r")
    faces_list, boxes_list, box_ids_list, non_empty_indices = _load_intermediate_state(temp_dir)
    logger.info(f"Load intermediate state: {time.perf_counter() - t0:.2f}s")
    logger.info(f"Loaded {len(non_empty_indices)} non-empty blocks")

    run_identity = load_run_identity(temp_dir / "run_config.json")
    final_seg_zarr, merged_boxes = _stitch_precomputed(
        block_indices=non_empty_indices,
        faces_list=faces_list,
        boxes_list=boxes_list,
        box_ids_list=box_ids_list,
        temp_zarr=temp_zarr,
        temp_dir=temp_dir,
        output_path=output_path,
        run_identity=run_identity,
        overwrite_output=overwrite_output,
    )

    logger.info(f"Total stitch_segmentation: {time.perf_counter() - t_total_start:.2f}s")

    return final_seg_zarr, merged_boxes


def _run_single_input(
    *,
    input_path: Path,
    channels: str | None,
    overwrite: bool,
    config_path: Path | None,
    workers_per_gpu: int,
    threads_per_worker: int,
    use_localcuda: bool,
    n_workers: int | None,
    target_nz: int | None,
    target_ny: int | None,
    target_nx: int | None,
    assume_nonempty: bool,
    cellpose_only: bool,
    stagger_seconds: float,
) -> None:
    if input_path.suffix != ".zarr" or not input_path.exists():
        raise FileNotFoundError(f"Path {input_path} must be a '.zarr' store.")

    base_dir = input_path.parent

    cellpose_version = version("cellpose")
    if not (cellpose_version.startswith("4.") or "dev" in cellpose_version):
        raise RuntimeError("This script requires Cellpose version 4.x for SAM backend support.")

    resolved_config_path = config_path
    if resolved_config_path is None:
        resolved_config_path = base_dir.parent / "config.json"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Config file not found at {resolved_config_path}")

    config = json.loads(resolved_config_path.read_text())
    backend = config.get("backend", "sam").lower()
    if backend != "sam":
        raise ValueError("Distributed segmentation supports only backend='sam'.")
    zarr_output_path = base_dir / "output_segmentation-sam.zarr"
    temporary_directory = base_dir / "cellpose_temp"

    ortho_weights = config.get("ortho_weights", [3, 1.0, 1.0])
    diameter = config.get("diameter", 30)
    cellpose_model_kwargs = {
        "pretrained_model": config["pretrained_model"],
        "gpu": True,
    }

    local_cluster_kwargs = {
        "workers_per_gpu": workers_per_gpu,
        "threads_per_worker": threads_per_worker,
    }
    if use_localcuda and workers_per_gpu <= 1:
        local_cluster_kwargs.update(
            {
                "use_localcuda": True,
                "n_workers": n_workers,
            }
        )

    preprocessing_pipeline = [(unsharp_all, {})]

    input_zarr_array = zarr.open_array(input_path, mode="r")
    if input_zarr_array.ndim != 4:
        raise ValueError(f"Input Zarr must have ZYXC shape; got {input_zarr_array.shape}.")
    channel_indices, channel_names = _resolve_channel_selection(input_zarr_array, channels)
    source_channels = [index + 1 for index in channel_indices]
    logger.info(f"Using channels in requested order: {list(channel_names)}")
    input_schema = {
        "shape": list(input_zarr_array.shape),
        "chunks": list(input_zarr_array.chunks),
        "dtype": str(input_zarr_array.dtype),
    }
    input_provenance = _input_provenance_identity(
        input_path,
        expected_schema=input_schema,
    )
    input_identity = {
        "path": str(input_path.resolve()),
        **input_schema,
        "attrs": dict(input_zarr_array.attrs),
        "provenance": input_provenance,
    }
    input_key = _identity_digest(input_identity)

    processing_blocksize = _sam_processing_blocksize(
        n_channels=len(source_channels),
        diameter=diameter,
        target_nz=target_nz,
        target_ny=target_ny,
        target_nx=target_nx,
    )
    logger.info(
        "SAM backend: target tiles (nz=%s, ny=%s, nx=%s) → 3-D core blocksize %s",
        2 if target_nz is None else target_nz,
        2 if target_ny is None else target_ny,
        6 if target_nx is None else target_nx,
        processing_blocksize[:-1],
    )
    normalization_path = base_dir / "normalization.json"
    normalization_settings = {
        "implementation": "bounded-z-gpu-v2",
        "block_yx": [256, 1024],
        "spatial_samples": 30,
        "z_samples": 32,
        "z_selection": "seeded-stratified",
        "low": 1.0,
        "high": 99.9,
        "seed": 0,
        "unsharp_backend": "cucim",
        "unsharp_dimensionality": "plane-wise-2d",
        "unsharp_radius": 3.0,
    }
    normalization_key = _identity_digest(
        {"input_key": input_key, "settings": normalization_settings}
    )
    cached_normalization = read_normalization_cache(normalization_path, normalization_key)
    lowhigh_selected: NDArray[np.float64] | None = None

    if cached_normalization is not None:
        try:
            lowhigh_selected = np.array(
                [cached_normalization[str(ch)] for ch in source_channels],
                dtype=np.float64,
            )
        except KeyError:
            logger.info("Normalization cache missing requested channels; recomputing.")
    if lowhigh_selected is None:
        logger.info("Calculating normalization percentiles (cache miss).")
        perc, _ = sample_percentile(
            input_zarr_array,
            channels=source_channels,
            block=(256, 1024),
            n=30,
            low=1,
            high=99.9,
            seed=0,
            z_samples=32,
            unsharp_radius=3.0,
        )
        lowhigh_selected = np.asarray(perc, dtype=float)
        write_normalization_cache(
            normalization_path,
            normalization_key,
            {str(ch): lh.tolist() for ch, lh in zip(source_channels, lowhigh_selected)},
            settings=normalization_settings,
        )
        logger.info(f"Saved normalization thresholds to {normalization_path}")

    lowhigh_eval = np.asarray(lowhigh_selected, dtype=float)

    # Cellpose always produces 3 channels; pad missing ones with identity ranges
    if lowhigh_eval.shape[0] < 3:
        pad = np.repeat([[0.0, 1.0]], repeats=3 - lowhigh_eval.shape[0], axis=0)
        lowhigh_eval = np.concatenate([lowhigh_eval, pad], axis=0)

    normalization = {"lowhigh": lowhigh_eval.tolist()}
    logger.info(f"Normalization params (requested order): {normalization}")

    cellpose_eval_kwargs = _build_cellpose_eval_kwargs(
        diameter=diameter,
        normalization=normalization,
        ortho_weights=ortho_weights,
    )

    run_identity = _build_run_identity(
        input_identity=input_identity,
        channel_indices=channel_indices,
        model_kwargs=cellpose_model_kwargs,
        eval_kwargs=cellpose_eval_kwargs,
        blocksize=processing_blocksize,
        overlap=int(cellpose_eval_kwargs["diameter"] * 2),
        preprocessing_steps=preprocessing_pipeline,
        assume_nonempty=assume_nonempty,
        mask=None,
        runtime_artifacts=_runtime_artifact_identity(
            Path(cellpose_model_kwargs["pretrained_model"]),
            input_provenance,
            max_devices=(
                n_workers if use_localcuda and workers_per_gpu <= 1 else None
            ),
        ),
    )
    backup_path = _restore_output_backup(zarr_output_path, run_identity)
    if (
        backup_path.exists()
        and zarr_output_path.exists()
        and promoted_output_matches(zarr_output_path, run_identity)
    ):
        logger.info(f"Completing interrupted output promotion: {zarr_output_path}")
        write_completion_marker(zarr_output_path, run_identity)
        _remove_output_backup(backup_path)
        _remove_matching_temp(temporary_directory, run_identity)
        return
    if overwrite:
        run_config_path = temporary_directory / "run_config.json"
        resume_overwrite = False
        if run_config_path.exists():
            try:
                validate_run_config(run_config_path, run_identity)
            except ValueError as exc:
                logger.info(f"--overwrite: existing temporary state cannot resume ({exc})")
            else:
                resume_overwrite = True
        if temporary_directory.exists() and not resume_overwrite:
            logger.info(f"--overwrite: removing existing temp directory {temporary_directory}")
            shutil.rmtree(temporary_directory)
        elif resume_overwrite:
            logger.info("--overwrite: resuming matching temporary state")
    elif completed_run_matches(zarr_output_path, run_identity):
        logger.info(f"Completed output already matches this run: {zarr_output_path}")
        _remove_matching_temp(temporary_directory, run_identity)
        return
    elif promoted_output_matches(zarr_output_path, run_identity):
        logger.info(f"Recovering completion marker for promoted output: {zarr_output_path}")
        write_completion_marker(zarr_output_path, run_identity)
        _remove_matching_temp(temporary_directory, run_identity)
        return
    elif zarr_output_path.exists():
        raise FileExistsError(
            f"Output {zarr_output_path} exists without matching completion state; "
            "use --overwrite to replace it."
        )

    logger.info("Starting distributed Cellpose evaluation…")
    result = distributed_eval(
        input_zarr=input_zarr_array,
        blocksize=processing_blocksize,
        write_path=zarr_output_path,
        mask=None,
        preprocessing_steps=preprocessing_pipeline,
        model_kwargs=cellpose_model_kwargs,
        eval_kwargs=cellpose_eval_kwargs,
        cluster_kwargs=local_cluster_kwargs,
        temporary_directory=temporary_directory,
        cellpose_only=cellpose_only,
        stagger_seconds=stagger_seconds,
        channel_indices=channel_indices,
        run_identity=run_identity,
        assume_nonempty=assume_nonempty,
        overwrite_output=overwrite,
    )

    if cellpose_only:
        logger.info("Cellpose-only mode complete.")
        logger.info(f"Intermediate results saved to: {temporary_directory}")
        logger.info("Run 'stitch' command to complete the pipeline.")
        return

    final_segmentation_zarr, final_bounding_boxes = result
    write_completion_marker(zarr_output_path, run_identity)
    _remove_output_backup(_backup_output_path(zarr_output_path, run_identity))
    shutil.rmtree(temporary_directory)
    logger.info("Run Finished")
    logger.info(f"Final segmentation saved to: {zarr_output_path}")
    logger.info(
        f"Output Zarr shape: {final_segmentation_zarr.shape}, dtype: {final_segmentation_zarr.dtype}"
    )
    logger.info(f"Number of segmented objects found: {len(final_bounding_boxes)}")


@click.group()
def cli() -> None:
    """Distributed Cellpose segmentation utilities."""


@cli.command("run")
@click.argument("input_zarr", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--channels", default=None, type=str, help="Comma-separated list of channel names to use.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite existing segmentation.")
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Explicit path to config.json. Defaults to <input_zarr>/../config.json when omitted.",
)
@click.option(
    "--workers-per-gpu",
    default=4,
    show_default=True,
    type=int,
    help="Number of workers to spawn per GPU.",
)
@click.option(
    "--threads-per-worker",
    default=1,
    show_default=True,
    type=int,
    help="Threads per worker.",
)
@click.option(
    "--use-localcuda/--no-use-localcuda",
    default=False,
    show_default=True,
    help="If true and workers_per_gpu<=1, use dask-cuda LocalCUDACluster.",
)
@click.option("--n-workers", default=None, type=int, help="For LocalCUDACluster: number of workers.")
@click.option("--target-nz", default=None, type=int, help="Desired internal Cellpose nz tiles.")
@click.option("--target-ny", default=None, type=int, help="Desired internal Cellpose ny tiles.")
@click.option("--target-nx", default=None, type=int, help="Desired internal Cellpose nx tiles.")
@click.option(
    "--assume-nonempty/--scan-nonempty",
    default=False,
    show_default=True,
    help="On a nonempty-cache miss, process every planned block without scanning input.",
)
@click.option(
    "--cellpose-only/--no-cellpose-only",
    default=False,
    show_default=True,
    help="Stop after cellpose phase, save intermediate state for later stitching.",
)
@click.option(
    "--stagger-seconds",
    default=5.0,
    show_default=True,
    type=float,
    help="Seconds to stagger worker starts on the same GPU (0 to disable).",
)
def run(
    input_zarr: Path,
    channels: str | None,
    overwrite: bool,
    config_path: Path | None,
    workers_per_gpu: int,
    threads_per_worker: int,
    use_localcuda: bool,
    n_workers: int | None,
    target_nz: int | None,
    target_ny: int | None,
    target_nx: int | None,
    assume_nonempty: bool,
    cellpose_only: bool,
    stagger_seconds: float,
) -> None:
    """Run distributed Cellpose segmentation on one fused .zarr input."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("cellpose").setLevel(logging.WARNING)
    _run_single_input(
        input_path=input_zarr,
        channels=channels,
        overwrite=overwrite,
        config_path=config_path,
        workers_per_gpu=workers_per_gpu,
        threads_per_worker=threads_per_worker,
        use_localcuda=use_localcuda,
        n_workers=n_workers,
        target_nz=target_nz,
        target_ny=target_ny,
        target_nx=target_nx,
        assume_nonempty=assume_nonempty,
        cellpose_only=cellpose_only,
        stagger_seconds=stagger_seconds,
    )


def _run_stitch(temp_dir: Path, output_path: Path, *, cleanup: bool, overwrite: bool) -> None:
    """Own validation, recovery, promotion, and cleanup for both CLI frontends."""
    run_identity = load_run_identity(temp_dir / "run_config.json")
    backup_path = _restore_output_backup(output_path, run_identity)
    if (
        backup_path.exists()
        and output_path.exists()
        and promoted_output_matches(output_path, run_identity)
    ):
        logger.info(f"Completing interrupted output promotion: {output_path}")
        write_completion_marker(output_path, run_identity)
        _remove_output_backup(backup_path)
        if cleanup:
            shutil.rmtree(temp_dir)
        return
    if overwrite:
        logger.info("--overwrite: existing output remains visible until staged stitch completes")
    elif completed_run_matches(output_path, run_identity):
        logger.info(f"Completed output already matches this run: {output_path}")
        if cleanup:
            shutil.rmtree(temp_dir)
        return
    elif promoted_output_matches(output_path, run_identity):
        logger.info(f"Recovering completion marker for promoted output: {output_path}")
        write_completion_marker(output_path, run_identity)
        if cleanup:
            shutil.rmtree(temp_dir)
        return
    elif output_path.exists():
        raise FileExistsError(f"Output {output_path} already exists; use --overwrite to replace it.")

    if not (temp_dir / "segmentation_unstitched.zarr").exists():
        raise FileNotFoundError(f"No segmentation_unstitched.zarr found in {temp_dir}")
    if not (temp_dir / "intermediate_state.npz").exists():
        raise FileNotFoundError(f"No intermediate_state.npz found in {temp_dir}")
    logger.info(f"Stitching segmentation from {temp_dir}")
    final_zarr, final_boxes = stitch_segmentation(
        temp_dir, output_path, overwrite_output=overwrite
    )
    write_completion_marker(output_path, run_identity)
    _remove_output_backup(_backup_output_path(output_path, run_identity))

    if cleanup:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temp directory: {temp_dir}")

    logger.info("Stitching complete")
    logger.info(f"Final segmentation saved to: {output_path}")
    logger.info(f"Output Zarr shape: {final_zarr.shape}, dtype: {final_zarr.dtype}")
    logger.info(f"Number of segmented objects found: {len(final_boxes)}")


@cli.command("stitch")
@click.argument("temp_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--cleanup/--no-cleanup", default=True, show_default=True, help="Remove temp directory after successful stitching.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite existing output.")
def stitch(temp_dir: Path, output_path: Path, cleanup: bool, overwrite: bool) -> None:
    """Stitch pre-computed Cellpose results into a final segmentation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    _run_stitch(temp_dir, output_path, cleanup=cleanup, overwrite=overwrite)


if __name__ == "__main__":
    cli()
