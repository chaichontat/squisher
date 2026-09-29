from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest

from squisher import gpu_retry


class FakeCudaOOM(RuntimeError):
    __module__ = "cupy.cuda.memory"


def _fake_cupy(cleanups: list[str]):
    class Device:
        def __init__(self, _device: int):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    pool = SimpleNamespace(free_all_blocks=lambda: cleanups.append("pool"))
    pinned = SimpleNamespace(free_all_blocks=lambda: cleanups.append("pinned"))
    return SimpleNamespace(
        cuda=SimpleNamespace(Device=Device, runtime=SimpleNamespace(getDevice=lambda: 0)),
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pinned,
    )


def test_cuda_oom_retries_after_cleanup(tmp_path, monkeypatch) -> None:
    cleanups: list[str] = []
    sleeps: list[float] = []
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(cleanups))
    monkeypatch.setenv("SQUISHER_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("SQUISHER_CUDA_OOM_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("SQUISHER_CUDA_OOM_INITIAL_DELAY_SECONDS", "0.25")
    monkeypatch.setattr(gpu_retry.time, "sleep", sleeps.append)
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FakeCudaOOM("CUDA out of memory")
        return "done"

    assert gpu_retry.run_with_cuda_oom_retry(operation, label="test", device=0) == "done"
    assert calls == 2
    assert cleanups == ["pool", "pinned"]
    assert sleeps == [0.25]


def test_cuda_retry_does_not_retry_non_oom(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy([]))
    monkeypatch.setenv("SQUISHER_GPU_LOCK_DIR", str(tmp_path))
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("invalid registration")

    with pytest.raises(RuntimeError, match="invalid registration"):
        gpu_retry.run_with_cuda_oom_retry(operation, label="test", device=0)
    assert calls == 1


def test_cuda_oom_raises_original_error_after_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy([]))
    monkeypatch.setenv("SQUISHER_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("SQUISHER_CUDA_OOM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("SQUISHER_CUDA_OOM_INITIAL_DELAY_SECONDS", "0.01")
    monkeypatch.setattr(gpu_retry.time, "sleep", lambda _seconds: None)

    with pytest.raises(FakeCudaOOM, match="CUDA out of memory"):
        gpu_retry.run_with_cuda_oom_retry(
            lambda: (_ for _ in ()).throw(FakeCudaOOM("CUDA out of memory")),
            label="test",
            device=0,
        )


def test_cuda_lease_serializes_same_device_threads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQUISHER_GPU_LOCK_DIR", str(tmp_path))
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with gpu_retry.cuda_device_lease(0):
            first_entered.set()
            release_first.wait(timeout=1)

    def second() -> None:
        first_entered.wait(timeout=1)
        with gpu_retry.cuda_device_lease(0):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=1)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_entered.is_set()
