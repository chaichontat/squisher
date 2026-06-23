from __future__ import annotations

import multiprocessing as mp
import queue
import signal
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from squisher_deconv.deconvolution import Deconvolver
from squisher_deconv.metadata import compression_tiff_tag, json_dumps_strict, provenance_payload
from squisher_deconv.planning import output_path_for, slab_windows
from squisher_deconv.scaling import ScalingParameters
from squisher_deconv.sink import write_streamed_ome_tiff
from squisher_deconv.source import TiffLogicalSource

try:
    MP_CONTEXT = mp.get_context("spawn")
except ValueError:  # pragma: no cover
    MP_CONTEXT = mp.get_context("forkserver")

WorkerStatus = Literal["ready", "ok", "error", "stopped"]


@dataclass(frozen=True, slots=True)
class WorkerMessage:
    worker_id: int
    status: WorkerStatus
    path: Path | None = None
    file_index: int | None = None
    device: int | None = None
    duration: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessRunConfig:
    out_dir: Path
    channels: int
    halo: int
    slab_depth: int
    output_mode: str
    psf_path: Path | None
    basic_paths: tuple[Path, ...]
    scaling_path: Path
    devices: tuple[int, ...]
    queue_depth: int
    overwrite: bool
    output_relative_root: Path | None


def run_process_gpu_streaming_deconv(
    paths: Sequence[Path],
    *,
    template_source: TiffLogicalSource,
    scaling: ScalingParameters,
    deconvolver_factory: Callable[[int], Deconvolver],
    config: ProcessRunConfig,
    stop_on_error: bool,
) -> list[str]:
    task_queues = [MP_CONTEXT.Queue(maxsize=config.queue_depth) for _ in config.devices]
    result_queue = MP_CONTEXT.Queue()
    processes = [
        MP_CONTEXT.Process(
            target=_worker_loop,
            kwargs={
                "worker_id": worker_id,
                "device": int(device),
                "task_queue": task_queues[worker_id],
                "result_queue": result_queue,
                "template_source": template_source,
                "scaling": scaling,
                "deconvolver_factory": deconvolver_factory,
                "config": config,
                "stop_on_error": stop_on_error,
            },
            daemon=False,
        )
        for worker_id, device in enumerate(config.devices)
    ]
    for process in processes:
        process.start()

    failures: list[str] = []
    next_index = 0
    stopped: set[int] = set()
    ready: set[int] = set()
    queued: dict[int, int] = {worker_id: 0 for worker_id in range(len(processes))}

    def assign(worker_id: int, count: int = 1) -> None:
        nonlocal next_index
        for _ in range(count):
            if next_index < len(paths):
                task_queues[worker_id].put((next_index, Path(paths[next_index])))
                queued[worker_id] += 1
                next_index += 1
            elif queued[worker_id] == 0:
                task_queues[worker_id].put(None)
                break

    try:
        while len(ready) + len(stopped) < len(processes):
            msg = _next_worker_message(result_queue, processes=processes, stopped=stopped)
            if msg.status == "ready":
                ready.add(msg.worker_id)
                continue
            if msg.status == "stopped":
                stopped.add(msg.worker_id)
                continue
            if msg.status == "error":
                failures.append(msg.error or f"Worker {msg.worker_id} failed during startup.")
                stopped.add(msg.worker_id)
                if stop_on_error:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    break
        for worker_id in sorted(ready):
            assign(worker_id, config.queue_depth)

        while len(stopped) < len(processes):
            msg = _next_worker_message(result_queue, processes=processes, stopped=stopped)
            if msg.status == "stopped":
                stopped.add(msg.worker_id)
                continue
            queued[msg.worker_id] = max(0, queued[msg.worker_id] - 1)
            if msg.status == "error":
                failures.append(msg.error or f"Worker {msg.worker_id} failed without traceback.")
                if msg.path is None:
                    stopped.add(msg.worker_id)
                if stop_on_error:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    break
                if msg.path is None:
                    continue
            else:
                _log(
                    f"file complete device={msg.device} file_index={msg.file_index} "
                    f"file={None if msg.path is None else msg.path.name} seconds={msg.duration:.2f}"
                )
            if not failures or not stop_on_error:
                assign(msg.worker_id)
    finally:
        for task_queue in task_queues:
            try:
                task_queue.put_nowait(None)
            except queue.Full:
                pass
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    return failures


def _worker_loop(
    *,
    worker_id: int,
    device: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    template_source: TiffLogicalSource,
    scaling: ScalingParameters,
    deconvolver_factory: Callable[[int], Deconvolver],
    config: ProcessRunConfig,
    stop_on_error: bool,
) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        deconvolver = deconvolver_factory(device)
        if config.output_mode == "u16" and not hasattr(deconvolver, "deconvolve_core_u16"):
            raise TypeError("Process GPU u16 worker requires deconvolve_core_u16.")
        _log(f"worker process start worker={worker_id} device={device}")
        result_queue.put(WorkerMessage(worker_id=worker_id, status="ready", device=device))
        while True:
            item = task_queue.get()
            if item is None:
                break
            file_index, path = item
            try:
                duration = _process_file(
                    worker_id=worker_id,
                    device=device,
                    file_index=int(file_index),
                    path=Path(path),
                    template_source=template_source,
                    scaling=scaling,
                    deconvolver=deconvolver,
                    config=config,
                )
            except Exception:
                result_queue.put(
                    WorkerMessage(
                        worker_id=worker_id,
                        status="error",
                        path=Path(path),
                        file_index=int(file_index),
                        device=device,
                        error=traceback.format_exc(),
                    )
                )
                if stop_on_error:
                    break
                continue
            result_queue.put(
                WorkerMessage(
                    worker_id=worker_id,
                    status="ok",
                    path=Path(path),
                    file_index=int(file_index),
                    device=device,
                    duration=duration,
                )
            )
    except Exception:
        result_queue.put(
            WorkerMessage(
                worker_id=worker_id,
                status="error",
                device=device,
                error=traceback.format_exc(),
            )
        )
    finally:
        result_queue.put(WorkerMessage(worker_id=worker_id, status="stopped", device=device))


def _process_file(
    *,
    worker_id: int,
    device: int,
    file_index: int,
    path: Path,
    template_source: TiffLogicalSource,
    scaling: ScalingParameters,
    deconvolver: Deconvolver,
    config: ProcessRunConfig,
) -> float:
    source = replace(template_source, path=path)
    slabs = slab_windows([source.path], z_counts=[source.z_count], slab_depth=config.slab_depth, halo=config.halo)
    output_path = output_path_for(config.out_dir, source.path, relative_root=config.output_relative_root)
    sidecar = output_path.with_suffix(".deconv.json")
    if sidecar.exists() and not config.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing sidecar {sidecar}; pass --overwrite to replace it.")

    provenance = provenance_payload(
        source.path,
        channels=config.channels,
        halo=config.halo,
        psf_path=config.psf_path,
        basic_paths=config.basic_paths,
        output_mode=config.output_mode,
        scaling_path=config.scaling_path,
        devices=config.devices,
        queue_depth=config.queue_depth,
    )
    sidecar_payload = {
        "source_metadata": asdict(source.metadata),
        "provenance": provenance,
        "chunking": {"slab_depth": config.slab_depth, "halo": config.halo},
        "output_mode": config.output_mode,
        "compression_tiff_tag": compression_tiff_tag(config.output_mode),
        "worker": {"worker_id": worker_id, "device": device},
    }
    sidecar_text = json_dumps_strict(
        sidecar_payload,
        context=f"Deconvolution sidecar for {source.path}",
        indent=2,
    )

    read_queue: queue.Queue[tuple[int, Any, np.ndarray] | object] = queue.Queue(maxsize=config.queue_depth)
    chunk_queue: queue.Queue[tuple[int, np.ndarray] | object] = queue.Queue(maxsize=config.queue_depth)
    sentinel = object()
    stop_event = threading.Event()
    errors: list[BaseException] = []
    t_file = time.perf_counter()

    _log(
        f"file start device={device} file_index={file_index} file={source.path.name} "
        f"slabs={len(slabs)} output={output_path}"
    )

    def reader() -> None:
        try:
            for slab_index, slab in enumerate(slabs):
                if stop_event.is_set():
                    break
                t_read = time.perf_counter()
                _log(
                    f"read start device={device} file={source.path.name} "
                    f"read_z=[{slab.read_start},{slab.read_stop})"
                )
                data = source.read_window(slab.read_start, slab.read_stop)
                _log(
                    f"read done device={device} file={source.path.name} "
                    f"read_z=[{slab.read_start},{slab.read_stop}) shape={data.shape} "
                    f"dtype={data.dtype} seconds={time.perf_counter() - t_read:.2f}"
                )
                _put(read_queue, (slab_index, slab, data), stop_event)
        except BaseException as exc:
            errors.append(exc)
            stop_event.set()
        finally:
            _put(read_queue, sentinel, stop_event)

    def compute() -> None:
        try:
            while True:
                try:
                    item = read_queue.get(timeout=0.1)
                except queue.Empty:
                    if stop_event.is_set() and errors:
                        break
                    continue
                if item is sentinel:
                    break
                slab_index, slab, read = item
                t_slab = time.perf_counter()
                _log(
                    f"slab received device={device} file={source.path.name} slab={slab_index}/{len(slabs)} "
                    f"read_z=[{slab.read_start},{slab.read_stop}) core_z=[{slab.core_start},{slab.core_stop}) "
                    f"shape={read.shape}"
                )
                t_deconv = time.perf_counter()
                _log(
                    f"deconv start device={device} file={source.path.name} slab={slab_index}/{len(slabs)} "
                    f"input_shape={read.shape}"
                )
                core_start = slab.core_start - slab.read_start
                core_stop = slab.core_stop - slab.read_start
                chunk = deconvolver.deconvolve_core_u16(
                    read,
                    core_start=core_start,
                    core_stop=core_stop,
                    scaling=scaling,
                )
                _validate_u16_core_chunk(chunk, source=source, core_planes=slab.core_stop - slab.core_start)
                _log(
                    f"deconv+quant done device={device} file={source.path.name} "
                    f"slab={slab_index}/{len(slabs)} output_shape={chunk.shape} "
                    f"seconds={time.perf_counter() - t_deconv:.2f}"
                )
                _log(
                    f"slab ready device={device} file={source.path.name} slab={slab_index}/{len(slabs)} "
                    f"chunk_shape={chunk.shape} seconds={time.perf_counter() - t_slab:.2f}"
                )
                _put(chunk_queue, (slab_index, chunk), stop_event)
        except BaseException as exc:
            errors.append(exc)
            stop_event.set()
        finally:
            _put(chunk_queue, sentinel, stop_event)

    def writer() -> None:
        def chunks():
            expected_index = 0
            while True:
                try:
                    item = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    if stop_event.is_set() and errors:
                        return
                    continue
                if item is sentinel:
                    return
                slab_index, chunk = item
                if slab_index != expected_index:
                    raise RuntimeError(
                        f"Out-of-order computed slab for {source.path.name}: got {slab_index}, "
                        f"expected {expected_index}."
                    )
                expected_index += 1
                yield chunk

        try:
            write_streamed_ome_tiff(
                output_path,
                source=source,
                core_plane_chunks=chunks(),
                output_mode=config.output_mode,
                provenance=provenance,
                overwrite=config.overwrite,
            )
            sidecar.write_text(sidecar_text)
        except BaseException as exc:
            errors.append(exc)
            stop_event.set()

    threads = [
        threading.Thread(target=reader, name=f"squisher-reader-{worker_id}", daemon=True),
        threading.Thread(target=compute, name=f"squisher-compute-{worker_id}", daemon=True),
        threading.Thread(target=writer, name=f"squisher-writer-{worker_id}", daemon=True),
    ]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(timeout=0.2)
        if errors:
            stop_event.set()
    if errors:
        raise errors[0]
    return time.perf_counter() - t_file


def _next_worker_message(
    result_queue: mp.Queue,
    *,
    processes: Sequence[mp.Process],
    stopped: set[int],
) -> WorkerMessage:
    while True:
        try:
            return result_queue.get(timeout=0.2)
        except queue.Empty:
            for worker_id, process in enumerate(processes):
                if worker_id in stopped:
                    continue
                if process.exitcode is not None:
                    return WorkerMessage(
                        worker_id=worker_id,
                        status="error",
                        error=f"Worker process {worker_id} exited with code {process.exitcode} without reporting status.",
                    )


def _put(q: queue.Queue[Any], item: Any, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.1)
            return
        except queue.Full:
            continue


def _validate_u16_core_chunk(chunk: np.ndarray, *, source: TiffLogicalSource, core_planes: int) -> None:
    expected_shape = (core_planes * source.channels, source.height, source.width)
    if chunk.dtype != np.uint16:
        raise TypeError(f"deconvolve_core_u16 must return uint16 data, got {chunk.dtype}.")
    if chunk.shape != expected_shape:
        raise ValueError(f"deconvolve_core_u16 returned shape {chunk.shape}, expected {expected_shape}.")


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[squisher-deconv {timestamp}] {message}", file=sys.stderr, flush=True)
