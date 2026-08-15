from __future__ import annotations

from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from threading import Event
from typing import Any

import cupy as cp
import numpy as np
from numpy.typing import NDArray


class TaskCancelledException(Exception):
    """Raised when extraction is cancelled."""


class ProgressReporter:
    def advance(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def print(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_CANCEL_EVENT = Event()


def get_cancel_event() -> Event:
    return _CANCEL_EVENT


def wrap_progress(advance: Callable[[], int | None]) -> ProgressReporter:
    class WrappedProgress(ProgressReporter):
        def advance(self, *_args: Any, **_kwargs: Any) -> None:
            advance()

    return WrappedProgress()


@contextmanager
def progress_reporter(_total: int) -> Generator[ProgressReporter, None, None]:
    yield ProgressReporter()


@contextmanager
def progress_bar_threadpool(
    _total: int,
    *,
    threads: int,
    stop_on_exception: bool = True,
    executor: ThreadPoolExecutor | None = None,
    debug: bool = False,
    step_stats: Any | None = None,
) -> Generator[Callable[..., Future[Any]], None, None]:
    del debug, step_stats
    owned_executor = executor is None
    pool = executor or ThreadPoolExecutor(max_workers=threads)
    futures: list[Future[Any]] = []

    def submit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        future = pool.submit(fn, *args, **kwargs)
        futures.append(future)
        return future

    try:
        yield submit
        for future in as_completed(futures):
            if stop_on_exception:
                future.result()
    finally:
        if owned_executor:
            pool.shutdown(wait=True, cancel_futures=False)


def unsharp_all(img: NDArray[Any], crop: None = None, channel_axis: int = 3) -> NDArray[Any]:
    del crop
    from cucim.skimage import filters as cucim_filters

    img_gpu = cp.asarray(np.asarray(img), dtype=cp.float32)
    result_gpu = cucim_filters.unsharp_mask(
        img_gpu,
        radius=3,
        preserve_range=True,
        channel_axis=channel_axis,
    )
    result = cp.asnumpy(result_gpu)
    del img_gpu, result_gpu
    return result
