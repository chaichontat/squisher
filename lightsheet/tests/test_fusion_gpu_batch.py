"""CUDA batches preserve one queue per device and worker correction context."""
from contextlib import contextmanager, nullcontext

import joblib
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
        assert payload == {"kind": "direct-fusion"}
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
