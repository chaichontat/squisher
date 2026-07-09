import csv
import datetime
import json
import logging
from contextlib import contextmanager
from json import JSONEncoder
import os
import pathlib
import shutil
import subprocess
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

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

from fishtools.segment.normalize import sample_percentile
from fishtools.segmentation.distributed.cache_utils import (
    read_nonempty_cache,
    read_normalization_cache,
    write_nonempty_cache,
    write_normalization_cache,
)
from fishtools.segmentation.distributed.gpu_cluster import cluster, myLocalCluster
from fishtools.segmentation.distributed.merge_utils import (
    block_faces,
    bounding_boxes_in_global_coordinates,
    get_block_crops,
    get_nblocks,
    global_segment_ids,
    merge_boxes_for_labels,
    remove_overlaps,
    stitch_labels,
    create_zarr_array,
    label_zarr_codecs,
)
from fishtools.segmentation.distributed.model_cache import CellposeModelPlugin, get_cached_model
from fishtools.segmentation.distributed.tiling import solve_internal_xy_for_tiles

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

    img_gpu = cp.asarray(np.asarray(img), dtype=cp.float32)
    result_gpu = cucim_filters.unsharp_mask(
        img_gpu, radius=3, preserve_range=True, channel_axis=channel_axis
    )
    result = cp.asnumpy(result_gpu)
    del img_gpu, result_gpu
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

    setattr(worker, "_fishtools_fatal_error", True)

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
    np.savez(
        temp_dir / "intermediate_state.npz",
        faces=np.array(faces_list, dtype=object),
        boxes=np.array(boxes_list, dtype=object),
        box_ids=np.array(box_ids_list, dtype=object),
        non_empty_indices=np.array(non_empty_indices),
    )


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


def compute_model_md5(model_path: Path) -> str:
    """MD5 hash of model file. Matches `md5sum <file>` CLI output."""
    import hashlib

    h = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_run_config(
    path: Path,
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    blocksize: tuple[int, ...],
    input_shape: tuple[int, ...],
    overlap: int,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]],
) -> None:
    """Save run configuration for resume validation."""
    model_path = Path(model_kwargs["pretrained_model"])
    config = {
        "model_md5": compute_model_md5(model_path),
        "model_path": str(model_path),
        "model_kwargs": model_kwargs,
        "eval_kwargs": eval_kwargs,
        "blocksize": list(blocksize),
        "overlap": overlap,
        "input_shape": list(input_shape),
        "preprocessing_steps": [f[0].__name__ for f in preprocessing_steps],
        "created_at": datetime.datetime.now().isoformat(),
    }
    path.write_text(json.dumps(config, indent=2, cls=NumpyEncoder))


def _normalize_for_comparison(obj: Any) -> Any:
    """Normalize objects for comparison (convert numpy arrays to lists)."""
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


def validate_run_config(
    path: Path,
    model_kwargs: dict[str, Any],
    eval_kwargs: dict[str, Any],
    blocksize: tuple[int, ...],
    input_shape: tuple[int, ...],
    overlap: int,
    preprocessing_steps: list[tuple[Callable[..., NDArray[Any]], dict[str, Any]]],
) -> None:
    """Validate current config matches saved config. Raises on mismatch."""
    saved = json.loads(path.read_text())
    current_md5 = compute_model_md5(Path(model_kwargs["pretrained_model"]))

    # Normalize current values for comparison (numpy arrays -> lists)
    current_model_kwargs = _normalize_for_comparison(model_kwargs)
    current_eval_kwargs = _normalize_for_comparison(eval_kwargs)

    errors = []
    if saved["model_md5"] != current_md5:
        errors.append(f"model_md5: {saved['model_md5']} != {current_md5}")
    if saved["model_kwargs"] != current_model_kwargs:
        errors.append(f"model_kwargs differ: {saved['model_kwargs']} != {current_model_kwargs}")
    if saved["eval_kwargs"] != current_eval_kwargs:
        errors.append(f"eval_kwargs differ: {saved['eval_kwargs']} != {current_eval_kwargs}")
    if saved["blocksize"] != list(blocksize):
        errors.append(f"blocksize: {saved['blocksize']} != {list(blocksize)}")
    if saved["overlap"] != overlap:
        errors.append(f"overlap: {saved['overlap']} != {overlap}")
    if saved["input_shape"] != list(input_shape):
        errors.append(f"input_shape: {saved['input_shape']} != {list(input_shape)}")

    current_pp = [f[0].__name__ for f in preprocessing_steps]
    if saved["preprocessing_steps"] != current_pp:
        errors.append(f"preprocessing_steps: {saved['preprocessing_steps']} != {current_pp}")

    if errors:
        raise ValueError("Cannot resume - config mismatch:\n" + "\n".join(errors))

    logger.info(f"Config validated - model MD5 matches (verify: md5sum {model_kwargs['pretrained_model']})")


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
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping corrupted checkpoint line {i}: {e}")
    return completed


def append_checkpoint(
    checkpoint_path: Path,
    block_index: tuple[int, ...],
    worker_name: str,
    duration_s: float,
    n_masks: int,
) -> None:
    """Atomically append checkpoint entry."""

    entry = {
        "index": list(block_index),
        "ts": datetime.datetime.now().isoformat(),
        "worker": worker_name,
        "duration_s": round(duration_s, 2),
        "n_masks": n_masks,
    }
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


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
    checkpoint_path: Path | None = None,
    stagger_seconds: float = 0.0,
    workers_per_gpu: int = 4,
) -> (
    tuple[NDArray[np.uint32], list[tuple[slice, ...]], NDArray[np.uint32]]
    | tuple[list[NDArray[Any]], list[tuple[slice, ...]], NDArray[np.uint32]]
):
    """
    Preprocess and segment one block with eventual merger in mind.

    Steps: read/preprocess/segment → remove overlaps → compute bounding boxes
    → remap to global IDs → write to output_zarr → compute block faces.

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
    If test_mode=False: (faces, boxes, box_ids)
    If test_mode=True: (segmentation_array, boxes, box_ids)
    """
    import time

    try:
        worker = distributed.get_worker()
    except ValueError:
        worker = None

    if worker is not None and getattr(worker, "_fishtools_fatal_error", False):
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

        boxes = bounding_boxes_in_global_coordinates(segmentation_trimmed_3d, crop_trimmed_3d)

        nblocks_3d = get_nblocks(input_zarr.shape[:-1], spatial_blocksize)
        block_index_3d = block_index[:-1]

        segmentation_global_3d, remap = global_segment_ids(
            segmentation_trimmed_3d, block_index_3d, nblocks_3d
        )

        final_unique_ids_3d = np.unique(segmentation_global_3d)
        box_ids_for_this_block = final_unique_ids_3d[final_unique_ids_3d != 0]

        if test_mode:
            return (segmentation_global_3d, boxes, box_ids_for_this_block)

        output_zarr[crop_trimmed_3d] = segmentation_global_3d

        # Shrink labels on faces so expensive distance transforms happen on workers
        faces = block_faces(segmentation_global_3d, shrink=True)

        if checkpoint_path is not None:
            append_checkpoint(
                checkpoint_path,
                block_index,
                getattr(worker, "name", "local") if worker is not None else "local",
                time.perf_counter() - start_time,
                int(np.max(segmentation_global_3d)),
            )

        return faces, boxes, box_ids_for_this_block
    except Exception as exc:
        wlog.exception(
            f"Worker {worker_name} FAILED BLOCK: {block_index}\tREGION: [{format_slice(crop)}]\n{exc!r}"
        )
        if worker is not None:
            _retire_worker_after_error(reason=f"fishtools process_block failed: {exc!r}")
        raise


# ----------------------- component functions ---------------------------------#


def read_preprocess_and_segment(
    input_zarr: zarr.Array,
    crop: tuple[slice, ...],
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

    image = input_zarr[crop]
    for pp_step in preprocessing_steps:
        pp_step[1]["crop"] = crop
        image = pp_step[0](image, **pp_step[1])
    log_file = None
    if worker_logs_directory is not None:
        log_file = f"dask_worker_{distributed.get_worker().name}.log"
        log_file = pathlib.Path(worker_logs_directory).joinpath(log_file)
    cellpose.io.logger_setup(stdout_file_replacement=log_file)

    model = get_cached_model(model_kwargs)
    backend = model_kwargs.get("backend", "sam")
    filtered_eval_kwargs = dict(eval_kwargs)
    if backend == "unet":
        filtered_eval_kwargs.pop("ortho_weights", None)
    return model.eval(image, **filtered_eval_kwargs)[0].astype(np.uint32)


def _wait_for_futures_collect_errors(
    *,
    futures: list[distributed.Future],
    future_labels: dict[distributed.Future, str],
    stage: str,
    log: logging.Logger,
) -> list[str]:
    failures: list[str] = []
    for fut in distributed.as_completed(futures):
        label = future_labels.get(fut, getattr(fut, "key", "<unknown>"))
        try:
            fut.result()
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
        base_parent = Path(write_path).parent if isinstance(write_path, (str, Path)) else Path.cwd()
        temporary_directory = base_parent / "cellpose_temp"

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    worker_logs_dirname = f"dask_worker_logs_{timestamp}"
    base_dir = Path(temporary_directory).parent
    worker_logs_dir = base_dir / worker_logs_dirname
    worker_logs_dir.mkdir(parents=True, exist_ok=True)

    if "diameter" not in eval_kwargs.keys():
        raise ValueError("Diameter must be set in eval_kwargs")

    overlap = eval_kwargs["diameter"] * 2
    block_indices, block_crops = get_block_crops(input_zarr.shape, blocksize, overlap, mask)
    assert cluster is not None

    def check_block_has_data(crop: tuple[slice, ...], zarr_array: zarr.Array, threshold: int = 0) -> bool:
        """Check if a given crop in a Zarr array contains any data above threshold."""
        data_slice = zarr_array[crop]
        return data_slice.any() if threshold == 0 else (data_slice > threshold).any()

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

    offset = 0
    n = None

    path_nonempty = Path(write_path).parent / "nonempty.json"
    idxs = read_nonempty_cache(path_nonempty, blocksize)
    if idxs is not None:
        logger.info(f"Loaded cached non-empty block indices ({len(idxs)} entries) from {path_nonempty}.")
    else:
        logger.info("Non-empty cache miss or invalidated; re-scanning input for non-zero blocks.")
        check_futures = cluster.client.map(
            check_block_has_data,
            block_crops[offset : None if n is None else offset + n],
            zarr_array=input_zarr,
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

        idxs = [i for i, is_non_zero in enumerate(non_zero_results, offset) if is_non_zero]
        write_nonempty_cache(path_nonempty, blocksize, idxs)
        logger.info(f"Persisted {len(idxs)} non-empty block indices to {path_nonempty}")

    final_block_indices, final_block_crops = (
        [block_indices[i] for i in idxs],
        [block_crops[i] for i in idxs],
    )
    total_non_empty_blocks = len(final_block_indices)
    del block_indices, block_crops

    logger.info(f"Selected {len(final_block_indices)} blocks with non-zero input data.")

    output_shape = input_zarr.shape[:-1]
    output_blocksize = blocksize[:-1]

    Path(temporary_directory).mkdir(parents=True, exist_ok=True)
    assert temporary_directory.exists()
    temp_zarr_path = Path(temporary_directory) / "segmentation_unstitched.zarr"
    checkpoint_path = Path(temporary_directory) / "checkpoint.jsonl"
    run_config_path = Path(temporary_directory) / "run_config.json"

    # Detect resume vs fresh start
    is_resume = run_config_path.exists()

    if is_resume:
        if not temp_zarr_path.exists():
            raise RuntimeError(f"Cannot resume: temp_zarr missing at {temp_zarr_path}")
        validate_run_config(
            run_config_path,
            model_kwargs,
            eval_kwargs,
            blocksize,
            input_zarr.shape,
            overlap,
            preprocessing_steps,
        )
        completed_indices = load_checkpoint(checkpoint_path)
        logger.info(
            f"Resuming: {len(completed_indices)} of {len(final_block_indices)} blocks already completed"
        )
    else:
        save_run_config(
            run_config_path,
            model_kwargs,
            eval_kwargs,
            blocksize,
            input_zarr.shape,
            overlap,
            preprocessing_steps,
        )
        completed_indices = set()
        logger.info(f"Fresh run - saving config to {run_config_path}")

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
    else:
        temp_zarr = create_zarr_array(
            temp_zarr_path,
            shape=output_shape,  # Use 3D shape
            chunks=output_blocksize,  # Use 3D chunks
            dtype=np.uint32,
            overwrite=True,
            codecs=label_zarr_codecs(np.uint32),
        )

    if not remaining_block_indices:
        logger.info("All blocks already completed, proceeding to merge")
    else:
        # Register plugin to cache model on workers (initialized once per worker)
        plugin = CellposeModelPlugin(model_kwargs)
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
            checkpoint_path=checkpoint_path,
            stagger_seconds=stagger_seconds,
            workers_per_gpu=workers_per_gpu,
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
            )

        if failures:
            preview = "\n".join(failures[:10])
            raise RuntimeError(
                f"Segmentation failed for {len(failures)} blocks; run can be resumed after fixing the issue.\n"
                f"First failures:\n{preview}"
            )

    # Reconstruct faces/boxes from temp_zarr for ALL blocks (handles resume case)
    logger.info("Computing faces and bounding boxes from temp_zarr...")
    results = []
    for block_idx, block_crop in zip(final_block_indices, final_block_crops):
        # Get the trimmed crop (without overlap)
        spatial_crop = block_crop[:-1]  # ZYX slices
        spatial_blocksize = blocksize[:-1]
        trimmed_crop = []
        for axis, (slc, bs) in enumerate(zip(spatial_crop, spatial_blocksize)):
            start = slc.start if slc.start == 0 else slc.start + overlap
            stop = min(start + bs, slc.stop)
            trimmed_crop.append(slice(start, stop))
        trimmed_crop = tuple(trimmed_crop)

        seg_block = temp_zarr[trimmed_crop]
        faces = block_faces(seg_block, shrink=True)
        boxes = bounding_boxes_in_global_coordinates(seg_block, trimmed_crop)
        unique_ids = cp.asnumpy(cp.unique(cp.asarray(seg_block)))
        box_ids = unique_ids[unique_ids != 0]
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
        Path(temporary_directory),
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

    new_labeling_path = Path(temporary_directory) / "new_labeling.npy"
    final_seg_zarr, new_labeling = stitch_labels(
        block_indices=non_empty_indices,
        faces_list=faces_list,
        box_ids_list=box_ids_list,
        temp_zarr=temp_zarr,
        write_path=write_path,
        lut_path=new_labeling_path,
        pre_shrunk=True,
    )

    # Segmentation-specific: merge bounding boxes
    merged_boxes = merge_boxes_for_labels(boxes_list, box_ids_list, new_labeling)

    _log_slurm_tile_summary(
        total_non_empty_blocks,
        len(remaining_block_indices),
        time.perf_counter() - overall_start,
    )

    return final_seg_zarr, merged_boxes


def stitch_segmentation(
    temp_dir: Path,
    output_path: Path,
) -> tuple[zarr.Array, NDArray]:
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

    new_labeling_path = temp_dir / "new_labeling.npy"
    final_seg_zarr, new_labeling = stitch_labels(
        block_indices=non_empty_indices,
        faces_list=faces_list,
        box_ids_list=box_ids_list,
        temp_zarr=temp_zarr,
        write_path=output_path,
        lut_path=new_labeling_path,
        pre_shrunk=True,  # faces were pre-shrunk in block_faces(shrink=True)
    )

    # Segmentation-specific: merge bounding boxes
    merged_boxes = merge_boxes_for_labels(boxes_list, box_ids_list, new_labeling)

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
    target_ny: int | None,
    target_nx: int | None,
    cellpose_only: bool,
    stagger_seconds: float,
) -> None:
    if input_path.suffix != ".zarr" or not input_path.exists():
        raise FileNotFoundError(f"Path {input_path} must be a '.zarr' store.")

    base_dir = input_path.parent

    if not overwrite and (base_dir / "segmentation.done").exists():
        logger.warning(f"{base_dir}: segmentation already exists. Skipping.")
        return

    IS_CELLPOSE_SAM = version("cellpose").startswith("4.") or "dev" in version("cellpose")
    if not IS_CELLPOSE_SAM:
        raise RuntimeError("This script requires Cellpose version 4.x for SAM backend support.")

    resolved_config_path = config_path
    if resolved_config_path is None:
        resolved_config_path = base_dir.parent / "config.json"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Config file not found at {resolved_config_path}")

    config = json.loads(resolved_config_path.read_text())
    backend = config.get("backend", "sam").lower()
    if backend not in {"sam", "unet"}:
        raise ValueError("backend must be one of {'sam', 'unet'}.")
    using_sam_backend = backend == "sam"

    zarr_output_path = base_dir / f"output_segmentation-{backend}.zarr"
    temporary_directory = base_dir / "cellpose_temp"

    if overwrite and temporary_directory.exists():
        logger.info(f"--overwrite: removing existing temp directory {temporary_directory}")
        shutil.rmtree(temporary_directory)

    (base_dir / "segmentation.done").unlink(missing_ok=True)
    ortho_weights = config.get("ortho_weights", [3, 1.0, 1.0])
    diameter = config.get("diameter", 30)
    cellpose_model_kwargs = {
        "pretrained_model": config["pretrained_model"],
        "gpu": True,
        "backend": backend,
    }
    if using_sam_backend and IS_CELLPOSE_SAM:
        ...
    else:
        cellpose_model_kwargs["pretrained_model_ortho"] = config.get("pretrained_model_ortho", None)

    local_cluster_kwargs = {
        "workers_per_gpu": int(8 if backend == "unet" else workers_per_gpu),
        "threads_per_worker": int(threads_per_worker),
    }
    if use_localcuda and workers_per_gpu <= 1:
        local_cluster_kwargs.update(
            {
                "use_localcuda": True,
                "n_workers": n_workers,
            }
        )

    preprocessing_pipeline = [(unsharp_all, {})]

    foreground_mask = None
    input_zarr_array = zarr.open_array(input_path, mode="r")

    key = cast(str, input_zarr_array.attrs["key"])
    if channels is None:
        channels_list = list(range(1, input_zarr_array.shape[3] + 1))
        logger.info(f"No channels specified. Using all channels: {channels_list}")
    else:
        try:
            channels_list = [key.index(c) + 1 for c in channels.split(",")]
            logger.info(f"Using channels: {channels}")
            if len(channels_list) != len(key):
                logger.warning(
                    f"Selected channels {channels_list} do not match total channels {list(range(1, input_zarr_array.shape[3] + 1))}"
                )
        except ValueError:
            raise ValueError(f"Channel names {channels} not found in {key}")
    del channels

    if backend == "unet":
        processing_blocksize = (
            input_zarr_array.shape[0],
            224,
            224 * 4,
            len(channels_list),
        )
    else:
        ny_target = target_ny if target_ny is not None else 2
        nx_target = target_nx if target_nx is not None else 6
        Ly_internal, Lx_internal = solve_internal_xy_for_tiles(
            ny_target,
            nx_target,
            bsize=256,
            tile_overlap=0.1,
        )
        scale_back = float(diameter) / 30.0
        by = int(Ly_internal * scale_back)
        bx = int(Lx_internal * scale_back)
        processing_blocksize = (
            input_zarr_array.shape[0],
            by,
            bx,
            len(channels_list),
        )
        logger.info(
            f"SAM backend: target tiles (ny={ny_target}, nx={nx_target}) → internal size ({Ly_internal}x{Lx_internal}) → blocksize ({by}x{bx})"
        )
    normalization_path = base_dir / "normalization.json"
    normalization = read_normalization_cache(normalization_path)
    lowhigh_selected: NDArray[np.float64] | None = None

    if normalization is not None:
        try:
            lowhigh_selected = np.array([normalization[str(ch)] for ch in channels_list], dtype=np.float64)
        except KeyError:
            logger.info("Normalization cache missing requested channels; recomputing.")
    if lowhigh_selected is None:
        logger.info("Calculating normalization percentiles (cache miss).")
        perc, _ = sample_percentile(
            input_zarr_array,
            channels=channels_list,
            block=(256, 1024),
            n=30,
            low=1,
            high=99.9,
        )
        lowhigh_selected = np.asarray(perc, dtype=float)
        write_normalization_cache(
            normalization_path,
            {str(ch): lh.tolist() for ch, lh in zip(channels_list, lowhigh_selected)},
        )
        logger.info(f"Saved normalization thresholds to {normalization_path}")

    lowhigh_eval = np.asarray(lowhigh_selected, dtype=float)

    # Cellpose always produces 3 channels; pad missing ones with identity ranges
    if lowhigh_eval.shape[0] < 3:
        pad = np.repeat([[0.0, 1.0]], repeats=3 - lowhigh_eval.shape[0], axis=0)
        lowhigh_eval = np.concatenate([lowhigh_eval, pad], axis=0)

    normalization = {"lowhigh": lowhigh_eval.tolist()}
    logger.info(f"Normalization params (requested order): {normalization}")

    channels_for_cellpose = list(channels_list)
    if len(channels_for_cellpose) == 1:
        channels_for_cellpose = [channels_for_cellpose[0], channels_for_cellpose[0]]

    cellpose_eval_kwargs = {
        "diameter": config.get("diameter", 30),
        "batch_size": 16 if backend == "unet" else 1,
        "normalize": normalization,
        "flow_threshold": 0,
        "cellprob_threshold": 0,
        "anisotropy": 2.0,
        "resample": False,
        "flow3D_smooth": 1.5,
        "niter": 1000,
        "do_3D": True,
        "min_size": 500,
        "channel_axis": 3,
        "use_kde_clustering": True,
    }
    if using_sam_backend:
        cellpose_eval_kwargs["z_axis"] = 0
        if not IS_CELLPOSE_SAM:
            cellpose_eval_kwargs["channels"] = channels_for_cellpose
    else:
        cellpose_eval_kwargs["channels"] = channels_for_cellpose
    if ortho_weights is not None:
        cellpose_eval_kwargs["ortho_weights"] = ortho_weights

    logger.info("Starting distributed Cellpose evaluation…")
    result = distributed_eval(
        input_zarr=input_zarr_array,
        blocksize=processing_blocksize,
        write_path=zarr_output_path,
        mask=foreground_mask,
        preprocessing_steps=preprocessing_pipeline,
        model_kwargs=cellpose_model_kwargs,
        eval_kwargs=cellpose_eval_kwargs,
        cluster_kwargs=local_cluster_kwargs,
        temporary_directory=temporary_directory,
        cellpose_only=cellpose_only,
        stagger_seconds=stagger_seconds,
    )

    if cellpose_only:
        logger.info("Cellpose-only mode complete.")
        logger.info(f"Intermediate results saved to: {temporary_directory}")
        logger.info("Run 'stitch' command to complete the pipeline.")
        return

    final_segmentation_zarr, final_bounding_boxes = result
    (zarr_output_path.parent / "segmentation.done").touch()
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
@click.option("--target-ny", default=None, type=int, help="Desired internal Cellpose ny tiles.")
@click.option("--target-nx", default=None, type=int, help="Desired internal Cellpose nx tiles.")
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
    target_ny: int | None,
    target_nx: int | None,
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
        target_ny=target_ny,
        target_nx=target_nx,
        cellpose_only=cellpose_only,
        stagger_seconds=stagger_seconds,
    )


@cli.command("stitch")
@click.argument("temp_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--cleanup/--no-cleanup", default=True, show_default=True, help="Remove temp directory after successful stitching.")
def stitch(temp_dir: Path, output_path: Path, cleanup: bool) -> None:
    """
    Stitch pre-computed cellpose results into final segmentation.

    Use this after running with --cellpose-only to complete the pipeline.
    """
    # Set up Rich logging fresh for this CLI invocation
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    if not (temp_dir / "segmentation_unstitched.zarr").exists():
        raise FileNotFoundError(f"No segmentation_unstitched.zarr found in {temp_dir}")
    if not (temp_dir / "intermediate_state.npz").exists():
        raise FileNotFoundError(f"No intermediate_state.npz found in {temp_dir}")

    logger.info(f"Stitching segmentation from {temp_dir}")
    final_zarr, final_boxes = stitch_segmentation(temp_dir, output_path)

    if cleanup:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temp directory: {temp_dir}")

    (output_path.parent / "segmentation.done").touch()
    logger.info("Stitching complete")
    logger.info(f"Final segmentation saved to: {output_path}")
    logger.info(f"Output Zarr shape: {final_zarr.shape}, dtype: {final_zarr.dtype}")
    logger.info(f"Number of segmented objects found: {len(final_boxes)}")


if __name__ == "__main__":
    cli()
