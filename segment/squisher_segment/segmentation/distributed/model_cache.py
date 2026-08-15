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

from squisher_segment.segment.model_artifacts import plan_path_for_device


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
    from cellpose.contrib.packed_infer import (
        PackedCellposeModelTRT,
        PackedCellposeUNetModelTRT,
    )

    resolved_kwargs = dict(model_kwargs)
    backend = resolved_kwargs.pop("backend", "sam").lower()
    if backend not in {"sam", "unet"}:
        raise ValueError("backend must be either 'sam' or 'unet'.")

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
    trt_class = {
        "sam": PackedCellposeModelTRT,
        "unet": PackedCellposeUNetModelTRT,
    }[backend]
    return trt_class(**resolved_kwargs)


def get_cached_model(model_kwargs: dict[str, Any]):
    """Get model from worker attribute, building if necessary (fallback).

    Uses Dask's documented pattern of storing state on worker.my_attribute.
    Falls back to building the model if the CellposeModelPlugin wasn't registered.
    """
    worker = distributed.get_worker()

    if hasattr(worker, "cellpose_model"):
        return worker.cellpose_model

    # Fallback: build and cache (handles case where plugin wasn't registered)
    _get_worker_logger().warning(f"Worker {worker.name}: Model not cached, building fresh")
    model = _build_packed_cellpose_model(model_kwargs)
    worker.cellpose_model = model
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

    def __init__(self, model_kwargs: dict[str, Any]):
        self.model_kwargs = model_kwargs

    def setup(self, worker):
        """Initialize model when worker starts. Stores on worker.cellpose_model."""
        wlog = _get_worker_logger()
        if hasattr(worker, "cellpose_model"):
            wlog.info(f"Worker {worker.name}: Model already initialized, skipping")
            return

        backend = self.model_kwargs.get("backend", "sam")
        wlog.info(f"Worker {worker.name}: Initializing CellPose model (backend={backend})")
        model = _build_packed_cellpose_model(self.model_kwargs)
        worker.cellpose_model = model
        worker.cellpose_model_kwargs = self.model_kwargs
        wlog.info(f"Worker {worker.name}: Model cached on worker.cellpose_model")

    def teardown(self, worker):
        """Clean up model when worker shuts down."""
        if hasattr(worker, "cellpose_model"):
            del worker.cellpose_model
            _get_worker_logger().info(f"Worker {worker.name}: Cleared cellpose_model")
