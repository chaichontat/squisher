from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from squisher_deconv.scheduler import parse_devices, schedule_round_robin


def test_file_and_sample_jobs_are_assigned_round_robin_to_devices() -> None:
    schedule = schedule_round_robin(5, [0, 2])

    assert [(job.index, job.device) for job in schedule] == [(0, 0), (1, 2), (2, 0), (3, 2), (4, 0)]


def test_auto_devices_default_to_single_worker_for_non_gpu_flows() -> None:
    assert parse_devices("auto") == [0]


def test_gpu_auto_devices_uses_visible_cupy_device_count(monkeypatch) -> None:
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 3)),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    assert parse_devices("auto", gpu_auto=True) == [0, 1, 2]


def test_gpu_auto_devices_fails_when_no_cuda_devices_are_visible(monkeypatch) -> None:
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0)),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    with pytest.raises(ValueError, match="No CUDA devices are visible"):
        parse_devices("auto", gpu_auto=True)
