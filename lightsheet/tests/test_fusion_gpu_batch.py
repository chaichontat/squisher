"""CUDA batches preserve one queue per device and worker correction context."""
from contextlib import contextmanager, nullcontext
import threading

import joblib
import numpy as np
import pytest
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


def test_weighted_devices_assign_one_serial_queue_per_gpu(monkeypatch):
    import cupy as cp

    queues = []
    active_device = {"value": None}

    @contextmanager
    def device_context(device):
        active_device["value"] = device
        yield
        active_device["value"] = None

    monkeypatch.setattr(legacy, "_CUDA_BATCH_DEVICE_OFFSET", 0)
    monkeypatch.setattr(cp.cuda, "Device", device_context)
    monkeypatch.setattr(joblib, "Parallel", lambda **kwargs: lambda tasks: [f(*a, **k) for f, a, k in tasks])

    def record(block, payload, device):
        assert payload == {"kind": "direct-fusion", "fusion_gpu_memory_max_bytes": 123}
        assert device == active_device["value"]
        queues.append((active_device["value"], block))

    monkeypatch.setattr(legacy, "mvs_fuse_chunk_payload", lambda _func: {"kind": "direct-fusion"})
    monkeypatch.setattr(legacy, "run_mvs_fuse_chunk_worker", record)
    legacy.process_batch_using_joblib_cuda_devices(
        lambda _block: (_ for _ in ()).throw(AssertionError("generic worker must not run")),
        list(range(7)),
        n_jobs=16,
        backend="threading",
        cuda_devices=(1, 0, 1, 0, 1, 0, 1),
        pipeline_staging_max_bytes=0,
        fusion_gpu_memory_max_bytes=123,
    )
    assert queues == [(1, 0), (1, 2), (1, 4), (1, 6), (0, 1), (0, 3), (0, 5)]


def test_threaded_queue_propagates_direct_worker_failure(monkeypatch):
    import cupy as cp

    @contextmanager
    def device_context(_device):
        yield

    monkeypatch.setattr(cp.cuda, "Device", device_context)
    monkeypatch.setattr(legacy, "mvs_fuse_chunk_payload", lambda _func: {"kind": "direct-fusion"})
    monkeypatch.setattr(
        legacy,
        "run_mvs_fuse_chunk_worker",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("direct worker failed")),
    )

    with pytest.raises(RuntimeError, match="direct worker failed"):
        legacy.process_batch_using_joblib_cuda_devices(
            lambda _block: None,
            [0],
            n_jobs=1,
            backend="threading",
            cuda_devices=(0,),
            pipeline_staging_max_bytes=0,
        )


def test_direct_pipeline_overlaps_prepare_gpu_and_ordered_write(monkeypatch):
    import cupy as cp

    prepared_second = threading.Event()
    computing_second = threading.Event()
    active_computes = 0
    max_active_computes = 0
    writes = []
    state_lock = threading.Lock()

    @contextmanager
    def device_context(_device):
        yield

    def plan(block_id, _payload):
        return {
            "block_id": block_id,
            "direct": True,
            "estimated_input_bytes": 10,
            "expected_shape": (1,),
            "output_dtype": np.dtype("uint8"),
        }

    def prepare(plan):
        if plan["block_id"] == 1:
            prepared_second.set()
        return {"plan": plan, "nbytes": 10}

    def compute(prepared, _device):
        nonlocal active_computes, max_active_computes
        block_id = prepared["plan"]["block_id"]
        if block_id == 0:
            assert prepared_second.wait(timeout=1), "next prepare did not overlap the GPU stage"
        with state_lock:
            active_computes += 1
            max_active_computes = max(max_active_computes, active_computes)
        if block_id == 1:
            computing_second.set()
        with state_lock:
            active_computes -= 1
        return {"block_id": block_id, "nbytes": 1}

    def write(result):
        if result["block_id"] == 0:
            assert computing_second.wait(timeout=1), "write did not overlap the next GPU stage"
        writes.append(result["block_id"])

    monkeypatch.setattr(cp.cuda, "Device", device_context)
    monkeypatch.setattr(joblib, "Parallel", lambda **kwargs: lambda tasks: [f(*a, **k) for f, a, k in tasks])
    monkeypatch.setattr(legacy, "mvs_fuse_chunk_payload", lambda _func: {"kind": "direct-fusion"})
    monkeypatch.setattr(legacy, "plan_mvs_fuse_chunk", plan)
    monkeypatch.setattr(legacy, "prepare_mvs_direct_fusion_block", prepare)
    monkeypatch.setattr(legacy, "compute_mvs_prepared_fusion_block", compute)
    monkeypatch.setattr(legacy, "write_mvs_fused_block", write)

    legacy.process_batch_using_joblib_cuda_devices(
        lambda _block: None,
        [0, 1],
        n_jobs=1,
        backend="threading",
        cuda_devices=(0,),
        pipeline_staging_max_bytes=100,
    )

    assert max_active_computes == 1
    assert writes == [0, 1]


def test_direct_pipeline_blocks_prepare_until_staging_bytes_are_released(monkeypatch):
    import cupy as cp

    prepared_second = threading.Event()

    @contextmanager
    def device_context(_device):
        yield

    def plan(block_id, _payload):
        return {
            "block_id": block_id,
            "direct": True,
            "estimated_input_bytes": 10,
            "expected_shape": (1,),
            "output_dtype": np.dtype("uint8"),
        }

    def prepare(plan):
        if plan["block_id"] == 1:
            prepared_second.set()
        return {"plan": plan, "nbytes": 10}

    def compute(prepared, _device):
        if prepared["plan"]["block_id"] == 0:
            assert not prepared_second.is_set()
        return {"block_id": prepared["plan"]["block_id"], "nbytes": 1}

    monkeypatch.setattr(cp.cuda, "Device", device_context)
    monkeypatch.setattr(joblib, "Parallel", lambda **kwargs: lambda tasks: [f(*a, **k) for f, a, k in tasks])
    monkeypatch.setattr(legacy, "mvs_fuse_chunk_payload", lambda _func: {})
    monkeypatch.setattr(legacy, "plan_mvs_fuse_chunk", plan)
    monkeypatch.setattr(legacy, "prepare_mvs_direct_fusion_block", prepare)
    monkeypatch.setattr(legacy, "compute_mvs_prepared_fusion_block", compute)
    monkeypatch.setattr(legacy, "write_mvs_fused_block", lambda _result: None)

    legacy.process_batch_using_joblib_cuda_devices(
        lambda _block: None,
        [0, 1],
        n_jobs=1,
        backend="threading",
        cuda_devices=(0,),
        pipeline_staging_max_bytes=11,
    )

    assert prepared_second.is_set()


def test_direct_pipeline_serializes_cross_device_writes_in_batch_order(monkeypatch):
    import cupy as cp

    computed_second = threading.Event()
    writes = []

    @contextmanager
    def device_context(_device):
        yield

    def plan(block_id, _payload):
        return {
            "block_id": block_id,
            "direct": True,
            "estimated_input_bytes": 1,
            "expected_shape": (1,),
            "output_dtype": np.dtype("uint8"),
        }

    def compute(prepared, _device):
        block_id = prepared["plan"]["block_id"]
        if block_id == 0:
            assert computed_second.wait(timeout=1)
        else:
            computed_second.set()
        return {"block_id": block_id, "nbytes": 1}

    monkeypatch.setattr(cp.cuda, "Device", device_context)
    monkeypatch.setattr(legacy, "mvs_fuse_chunk_payload", lambda _func: {})
    monkeypatch.setattr(legacy, "plan_mvs_fuse_chunk", plan)
    monkeypatch.setattr(
        legacy,
        "prepare_mvs_direct_fusion_block",
        lambda plan: {"plan": plan, "nbytes": 1},
    )
    monkeypatch.setattr(legacy, "compute_mvs_prepared_fusion_block", compute)
    monkeypatch.setattr(legacy, "write_mvs_fused_block", lambda result: writes.append(result["block_id"]))

    legacy.process_batch_using_joblib_cuda_devices(
        lambda _block: None,
        [0, 1],
        n_jobs=2,
        backend="threading",
        cuda_devices=(0, 1),
        pipeline_staging_max_bytes=10,
    )

    assert writes == [0, 1]


def test_pipeline_falls_back_when_all_device_queues_cannot_run_concurrently(monkeypatch):
    import cupy as cp

    completed = []

    @contextmanager
    def device_context(_device):
        yield

    monkeypatch.setattr(cp.cuda, "Device", device_context)
    monkeypatch.setattr(legacy, "mvs_fuse_chunk_payload", lambda _func: {})
    monkeypatch.setattr(
        legacy,
        "plan_mvs_fuse_chunk",
        lambda block_id, _payload: {"block_id": block_id, "direct": True},
    )
    monkeypatch.setattr(
        legacy,
        "run_mvs_fuse_chunk_worker",
        lambda block_id, _payload, device: completed.append((block_id, device)),
    )

    legacy.process_batch_using_joblib_cuda_devices(
        lambda _block: None,
        [0, 1, 2],
        n_jobs=1,
        backend="threading",
        cuda_devices=(0, 1),
        pipeline_staging_max_bytes=10,
    )

    assert sorted(completed) == [(0, 0), (1, 1), (2, 0)]


@pytest.mark.parametrize("failed_stage", ["prepare", "compute", "write"])
def test_direct_pipeline_propagates_stage_failures(monkeypatch, failed_stage):
    import cupy as cp

    @contextmanager
    def device_context(_device):
        yield

    def maybe_fail(stage):
        if failed_stage == stage:
            raise RuntimeError(f"{stage} failed")

    def prepare(plan):
        if plan["block_id"] == 1:
            maybe_fail("prepare")
        return {"plan": plan, "nbytes": 1}

    def compute(prepared, _device):
        maybe_fail("compute")
        return {"block_id": prepared["plan"]["block_id"], "nbytes": 1}

    monkeypatch.setattr(cp.cuda, "Device", device_context)
    monkeypatch.setattr(joblib, "Parallel", lambda **kwargs: lambda tasks: [f(*a, **k) for f, a, k in tasks])
    monkeypatch.setattr(legacy, "mvs_fuse_chunk_payload", lambda _func: {})
    monkeypatch.setattr(
        legacy,
        "plan_mvs_fuse_chunk",
        lambda block_id, _payload: {
            "block_id": block_id,
            "direct": True,
            "estimated_input_bytes": 1,
            "expected_shape": (1,),
            "output_dtype": np.dtype("uint8"),
        },
    )
    monkeypatch.setattr(legacy, "prepare_mvs_direct_fusion_block", prepare)
    monkeypatch.setattr(legacy, "compute_mvs_prepared_fusion_block", compute)
    monkeypatch.setattr(
        legacy,
        "write_mvs_fused_block",
        lambda _result: maybe_fail("write"),
    )

    with pytest.raises(RuntimeError, match=f"{failed_stage} failed"):
        legacy.process_batch_using_joblib_cuda_devices(
            lambda _block: None,
            [0, 1],
            n_jobs=1,
            backend="threading",
            cuda_devices=(0,),
            pipeline_staging_max_bytes=100,
        )


def test_worker_receives_correction_on_its_assigned_gpu(monkeypatch):
    import cupy as cp
    from multiview_stitcher import misc_utils

    active = {"device": None, "correction": False}
    config = {"inverse_flatfields": "test-field", "dataset_info_key": "source"}

    @contextmanager
    def device_context(device):
        active["device"] = device
        yield
        active["device"] = None

    @contextmanager
    def correction_context(**kwargs):
        assert active["device"] == 1
        assert kwargs == config
        active["correction"] = True
        yield
        active["correction"] = False

    def fuse_one(index, block, path, devices):
        assert active == {"device": 1, "correction": True}
        assert devices == (1,)

    monkeypatch.setattr(cp.cuda, "Device", device_context)
    monkeypatch.setattr(legacy, "load_mvs_fuse_chunk_payload", lambda path: {"worker_read_config": config})
    monkeypatch.setattr(legacy, "zarr_safe_fusion_selection", lambda **kwargs: nullcontext())
    monkeypatch.setattr(legacy, "basic_corrected_zarr_reads", correction_context)
    monkeypatch.setattr(legacy, "run_mvs_fuse_chunk_loky_worker", fuse_one)
    monkeypatch.setattr(misc_utils, "clear_cupy_memory", lambda: None)
    legacy.run_mvs_fuse_chunk_loky_worker_batch([0, 1], "payload", 1)
    assert active == {"device": None, "correction": False}
