"""
Cellpose model caching for distributed workers.

Provides a Dask WorkerPlugin that initializes and caches a CellPose model
once per worker, avoiding repeated model loading overhead.

Per-worker state pattern
------------------------
Dask workers serialize functions via pickle/cloudpickle, which captures module
references rather than module state. Module-level globals do NOT reliably
persist across task invocations on the same worker.

The correct pattern for per-worker state is to store attributes on the worker
object itself (via `distributed.get_worker()`), or use a WorkerPlugin with a
`setup()` method that runs once when the worker starts.

References:
    - https://distributed.dask.org/en/stable/plugins.html
    - https://distributed.dask.org/en/stable/serialization.html
"""

import logging
from pathlib import Path
from typing import Any

import distributed
from distributed import WorkerPlugin

from squisher_segment.segment.model_artifacts import (
    file_sha256,
    plan_path_for_device,
    runtime_source_sha256,
)


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


def _build_packed_cellpose_model(model_kwargs: dict[str, Any]):
    """Instantiate a packed Cellpose model from the required TensorRT plan."""
    import torch
    from cellpose.contrib.packed_infer import PackedCellposeModelTRT

    resolved_kwargs = dict(model_kwargs)
    backend = resolved_kwargs.pop("backend", "sam").lower()
    if backend != "sam":
        raise ValueError("Distributed segmentation supports only the SAM backend.")

    pretrained_model = resolved_kwargs.get("pretrained_model")
    if pretrained_model is None:
        raise ValueError("model_kwargs must include 'pretrained_model'.")

    pretrained_path = Path(pretrained_model)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. TensorRT requires CUDA.")
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices found. TensorRT requires a GPU.")

    device_name = torch.cuda.get_device_name(0)
    plan_candidate = plan_path_for_device(pretrained_path, device_name)

    if not plan_candidate.is_file():
        raise FileNotFoundError(
            f"TensorRT plan required. Expected plan at {plan_candidate} for the current GPU."
        )

    _get_worker_logger().info(
        f"Using TensorRT plan {plan_candidate.name} for CUDA device '{device_name}'"
    )
    resolved_kwargs["pretrained_model"] = str(plan_candidate)
    resolved_kwargs.setdefault("gpu", True)
    return PackedCellposeModelTRT(**resolved_kwargs)


def _validate_worker_plan(
    model_kwargs: dict[str, Any],
    trt_plans: list[dict[str, Any]],
) -> None:
    """Ensure this worker's selected device plan is represented in run identity."""
    import torch

    device_name = torch.cuda.get_device_name(0)
    plan_path = plan_path_for_device(Path(model_kwargs["pretrained_model"]), device_name).resolve()
    expected = next(
        (plan for plan in trt_plans if plan.get("device_name") == device_name),
        None,
    )
    plan_stat = plan_path.stat()
    if (
        expected is None
        or expected.get("path") != str(plan_path)
        or expected.get("size") != plan_stat.st_size
        or expected.get("mtime_ns") != plan_stat.st_mtime_ns
        or expected.get("sha256") != file_sha256(plan_path)
    ):
        raise RuntimeError(
            f"Worker device '{device_name}' selected unrecorded TensorRT plan {plan_path}."
        )


def _validate_worker_sources(expected: dict[str, str]) -> None:
    actual = runtime_source_sha256()
    if actual != expected:
        changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
        raise RuntimeError(f"Worker inference source differs from run identity: {changed}")


def get_cached_model(model_kwargs: dict[str, Any]):
    """Get model from worker attribute, building if necessary (fallback).

    Uses Dask's documented pattern of storing state on worker.my_attribute.
    Falls back to building the model if the CellposeModelPlugin wasn't registered.
    """
    worker = distributed.get_worker()

    if hasattr(worker, "cellpose_model") and getattr(
        worker, "cellpose_model_kwargs", None
    ) == model_kwargs:
        return worker.cellpose_model

    if hasattr(worker, "cellpose_model"):
        del worker.cellpose_model

    # Fallback: build and cache (handles case where plugin wasn't registered)
    _get_worker_logger().warning(f"Worker {worker.name}: Model not cached, building fresh")
    model = _build_packed_cellpose_model(model_kwargs)
    worker.cellpose_model = model
    worker.cellpose_model_kwargs = dict(model_kwargs)
    return model


class CellposeModelPlugin(WorkerPlugin):
    """Plugin to initialize and cache CellPose model once per worker.

    Uses the officially documented pattern of storing state on worker attributes
    (get_worker().my_attribute) rather than get_worker().data which is reserved
    for Dask's internal memory management.

    References:
    - https://distributed.dask.org/en/latest/plugins.html
    - https://stackoverflow.com/questions/58126830/
    """

    name = "cellpose-model-cache"

    def __init__(
        self,
        model_kwargs: dict[str, Any],
        artifact_key: str,
        trt_plans: list[dict[str, Any]],
        source_sha256: dict[str, str],
    ):
        self.model_kwargs = model_kwargs
        self.artifact_key = artifact_key
        self.trt_plans = trt_plans
        self.source_sha256 = source_sha256

    def setup(self, worker):
        """Initialize model when worker starts. Stores on worker.cellpose_model."""
        wlog = _get_worker_logger()
        _validate_worker_plan(self.model_kwargs, self.trt_plans)
        _validate_worker_sources(self.source_sha256)
        if (
            hasattr(worker, "cellpose_model")
            and getattr(worker, "cellpose_model_kwargs", None) == self.model_kwargs
            and getattr(worker, "cellpose_model_artifact_key", None) == self.artifact_key
        ):
            wlog.info(f"Worker {worker.name}: Model already initialized, skipping")
            return
        if hasattr(worker, "cellpose_model"):
            wlog.info(f"Worker {worker.name}: Replacing cached model for new run identity")
            del worker.cellpose_model

        wlog.info(f"Worker {worker.name}: Initializing CellPose SAM model")
        model = _build_packed_cellpose_model(self.model_kwargs)
        worker.cellpose_model = model
        worker.cellpose_model_kwargs = dict(self.model_kwargs)
        worker.cellpose_model_artifact_key = self.artifact_key
        wlog.info(f"Worker {worker.name}: Model cached on worker.cellpose_model")

    def teardown(self, worker):
        """Clean up model when worker shuts down."""
        if hasattr(worker, "cellpose_model"):
            del worker.cellpose_model
            _get_worker_logger().info(f"Worker {worker.name}: Cleared cellpose_model")
