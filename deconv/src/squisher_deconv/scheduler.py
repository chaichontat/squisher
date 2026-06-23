from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    index: int
    device: int


def parse_devices(spec: str, *, gpu_auto: bool = False) -> list[int]:
    value = spec.strip().lower()
    if value == "auto":
        if not gpu_auto:
            return [0]
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            raise ValueError("No CUDA devices are visible for --devices auto.")
        return list(range(count))
    devices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not devices:
        raise ValueError(f"Invalid device specification: {spec!r}")
    return devices


def schedule_round_robin(job_count: int, devices: Sequence[int]) -> list[ScheduledJob]:
    if job_count < 0:
        raise ValueError(f"job_count must be non-negative, got {job_count}")
    if not devices:
        raise ValueError("At least one device is required.")
    return [ScheduledJob(index=index, device=int(devices[index % len(devices)])) for index in range(job_count)]
