"""Cross-process admission and bounded retry for host-to-host CUDA operations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import fcntl
from functools import wraps
import gc
import inspect
import os
from pathlib import Path
import re
import threading
import time
import traceback
from typing import ParamSpec, TypeVar

from loguru import logger


P = ParamSpec("P")
R = TypeVar("R")
DEFAULT_MAX_ATTEMPTS = 12
DEFAULT_INITIAL_DELAY_SECONDS = 5.0
DEFAULT_MAX_DELAY_SECONDS = 60.0
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_HELD = threading.local()


def _reset_after_fork() -> None:
    global _LOCKS, _LOCKS_GUARD, _HELD
    _LOCKS = {}
    _LOCKS_GUARD = threading.Lock()
    _HELD = threading.local()


os.register_at_fork(after_in_child=_reset_after_fork)


def _positive_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _device_key(device: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        identity = devices[device] if 0 <= device < len(devices) else f"local-{device}"
    else:
        identity = str(device)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", identity)


def _current_device() -> int:
    import cupy as cp

    runtime = getattr(getattr(cp, "cuda", None), "runtime", None)
    return 0 if runtime is None else int(runtime.getDevice())


@contextmanager
def cuda_device_lease(device: int | None = None) -> Iterator[None]:
    """Serialize host-to-host GPU transactions across threads and processes."""
    resolved_device = _current_device() if device is None else int(device)
    key = _device_key(resolved_device)
    held = getattr(_HELD, "keys", set())
    if key in held:
        yield
        return

    with _LOCKS_GUARD:
        thread_lock = _LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        lock_dir = Path(os.environ.get("SQUISHER_GPU_LOCK_DIR", f"/tmp/squisher-gpu-locks-{os.getuid()}"))
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"cuda-{key}.lock"
        with lock_path.open("a+b") as handle:
            started = time.monotonic()
            next_log = 60.0
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    elapsed = time.monotonic() - started
                    if elapsed >= next_log:
                        logger.info(
                            "Waiting for shared CUDA lease: device={} elapsed={:.0f}s lock={}",
                            key,
                            elapsed,
                            lock_path,
                        )
                        next_log += 60.0
                    time.sleep(1.0)
            _HELD.keys = held | {key}
            try:
                yield
            finally:
                _HELD.keys = held
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def is_cuda_oom(error: BaseException) -> bool:
    """Recognize CuPy and CUDA allocation failures without matching generic errors."""
    current: BaseException | None = error
    seen: set[int] = set()
    markers = (
        "cuda_error_out_of_memory",
        "cudaerrormemoryallocation",
        "cuda out of memory",
        "cupy.cuda.memory.outofmemoryerror",
    )
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        module = type(current).__module__.lower()
        name = type(current).__name__.lower()
        message = str(current).lower()
        if name == "outofmemoryerror" and (module.startswith("cupy") or "cuda" in message):
            return True
        if any(marker in message for marker in markers):
            return True
        current = current.__cause__ or current.__context__
    return False


def _clear_cuda_memory(device: int) -> None:
    try:
        import cupy as cp

        with cp.cuda.Device(device):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as cleanup_error:
        logger.warning("CUDA OOM cleanup failed on device {}: {}", device, cleanup_error)


def run_with_cuda_oom_retry(
    operation: Callable[[], R],
    *,
    label: str,
    device: int | None = None,
) -> R:
    """Run a reconstructible GPU transaction under a shared lease and retry OOMs."""
    resolved_device = _current_device() if device is None else int(device)
    max_attempts = _positive_int_env("SQUISHER_CUDA_OOM_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
    initial_delay = _positive_env("SQUISHER_CUDA_OOM_INITIAL_DELAY_SECONDS", DEFAULT_INITIAL_DELAY_SECONDS)
    max_delay = _positive_env("SQUISHER_CUDA_OOM_MAX_DELAY_SECONDS", DEFAULT_MAX_DELAY_SECONDS)
    for attempt in range(1, max_attempts + 1):
        try:
            with cuda_device_lease(resolved_device):
                return operation()
        except BaseException as error:
            if not is_cuda_oom(error) or attempt == max_attempts:
                raise
            if error.__traceback__ is not None:
                traceback.clear_frames(error.__traceback__)
                error.__traceback__ = None
            gc.collect()
            _clear_cuda_memory(resolved_device)
            delay = min(max_delay, initial_delay * 2 ** (attempt - 1))
            logger.warning(
                "CUDA OOM; retrying {}: device={} attempt={}/{} delay={:.1f}s",
                label,
                _device_key(resolved_device),
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("CUDA retry loop exhausted without returning or raising")


def cuda_oom_retry(
    label: str,
    *,
    device_parameter: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a host-to-host CUDA transaction."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            device = None
            if device_parameter is not None:
                bound = signature.bind_partial(*args, **kwargs)
                device = int(bound.arguments.get(device_parameter, 0))
            return run_with_cuda_oom_retry(
                lambda: function(*args, **kwargs),
                label=label,
                device=device,
            )

        return wrapped

    return decorate
