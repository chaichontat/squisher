from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np

from squisher_deconv.deconvolution import Deconvolver
from squisher_deconv.metadata import (
    compression_tiff_tag,
    dependency_versions,
    file_provenance_records,
    json_dumps_strict,
    provenance_payload,
)
from squisher_deconv.planning import (
    group_sample_windows,
    output_path_for,
    output_relative_root,
    output_sidecar_path,
    sample_planes_from_z_counts,
    slab_windows,
)
from squisher_deconv.process_workers import ProcessRunConfig, run_process_gpu_streaming_deconv
from squisher_deconv.scaling import (
    collate_scaling,
    load_scaling,
    quantize_global,
    save_float32_sample,
    validate_scaling_channels,
)
from squisher_deconv.scheduler import ScheduledJob, schedule_round_robin
from squisher_deconv.sink import write_streamed_ome_zarr
from squisher_deconv.source import TiffLogicalSource


class ProcessingError(RuntimeError):
    """Raised after one or more scheduled deconvolution jobs failed."""


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[squisher-deconv {timestamp}] {message}", file=sys.stderr, flush=True)


def sample_scale(
    inputs: Sequence[Path],
    *,
    out_dir: Path,
    planes: int,
    channels: int,
    halo: int,
    deconvolver: Deconvolver | None,
    psf_paths: Sequence[Path] | None,
    basic_paths: Sequence[Path] | None = None,
    deconvolver_factory: Callable[[int], Deconvolver] | None = None,
    seed: int,
    p_low: float,
    p_high: float,
    gamma: float,
    bins: int,
    devices: list[int],
    queue_depth: int,
    stop_on_error: bool,
    overwrite: bool = False,
) -> None:
    t_workflow = time.perf_counter()
    _validate_queue_depth(queue_depth)
    paths = [Path(path) for path in inputs]
    psfs = tuple(Path(path) for path in (psf_paths or ()))
    _log(
        "sample-scale start "
        f"inputs={len(paths)} out_dir={out_dir} planes={planes} channels={channels} halo={halo} "
        f"devices={devices} queue_depth={queue_depth} seed={seed} p_low={p_low} p_high={p_high} "
        f"gamma={gamma} bins={bins} overwrite={overwrite}"
    )
    for psf_path in psfs:
        _log(f"sample-scale psf={psf_path}")
    for basic_path in basic_paths or []:
        _log(f"sample-scale basic={basic_path}")
    t_sources = time.perf_counter()
    template_source = TiffLogicalSource.open(paths[0], channels=channels, metadata_mode="summary")
    _log(
        f"opened first source header in {time.perf_counter() - t_sources:.2f}s: "
        f"path={template_source.path} axes={template_source.axes} z={template_source.z_count} "
        f"channels={template_source.channels} yx=({template_source.height},{template_source.width}) "
        f"dtype={template_source.dtype}"
    )
    z_counts = [template_source.z_count] * len(paths)
    _log(f"using first source header for all {len(paths)} sample-scale input(s)")
    samples = sample_planes_from_z_counts(paths, z_counts=z_counts, planes=planes, seed=seed)
    windows = group_sample_windows(samples, z_counts=z_counts, halo=halo)
    schedule = schedule_round_robin(len(windows), devices)
    total_read_planes = sum(window.read_stop - window.read_start for window in windows)
    total_core_planes = sum(len(window.core_z) for window in windows)
    _log(
        "sample plan "
        f"total_logical_z={sum(z_counts)} sampled_core_planes={total_core_planes} "
        f"windows={len(windows)} total_read_z_planes={total_read_planes}"
    )
    for index, window in enumerate(windows):
        _log(
            f"plan window[{index:05d}] file={window.path} read_z=[{window.read_start},{window.read_stop}) "
            f"read_planes={window.read_stop - window.read_start} sampled_z={list(window.core_z)}"
        )
    _preflight_device_deconvolvers(
        devices,
        deconvolver=deconvolver,
        deconvolver_factory=deconvolver_factory,
        context="sample-scale",
    )
    sample_dir = out_dir / "float32-samples"
    _prepare_sample_scale_outputs(out_dir, sample_dir=sample_dir, overwrite=overwrite)
    sample_paths: list[Path] = []
    manifest_windows: list[dict[str, Any]] = []
    failures: list[str] = []

    def process_device_jobs(
        device: int, jobs: list[ScheduledJob]
    ) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
        device_deconvolver = _device_deconvolver(
            device,
            deconvolver=deconvolver,
            deconvolver_factory=deconvolver_factory,
        )
        local_samples: list[Path] = []
        local_manifest: list[dict[str, Any]] = []
        local_failures: list[str] = []

        def read_window_for_job(item: ScheduledJob) -> np.ndarray:
            window = windows[item.index]
            source = replace(template_source, path=window.path)
            t_read = time.perf_counter()
            _log(
                f"read start device={device} window[{item.index:05d}] file={source.path.name} "
                f"read_z=[{window.read_start},{window.read_stop}) sampled_z={list(window.core_z)}"
            )
            arr = source.read_window(window.read_start, window.read_stop)
            _log(
                f"read done device={device} window[{item.index:05d}] file={source.path.name} "
                f"shape={arr.shape} dtype={arr.dtype} seconds={time.perf_counter() - t_read:.2f}"
            )
            return arr

        for scheduled, slab, read_error in _prefetched_reads(
            jobs,
            queue_depth=queue_depth,
            read=read_window_for_job,
        ):
            window = windows[scheduled.index]
            source = replace(template_source, path=window.path)
            try:
                if read_error is not None:
                    raise read_error
                t_deconv = time.perf_counter()
                _log(
                    f"deconv start device={device} window[{scheduled.index:05d}] "
                    f"file={source.path.name} input_shape={slab.shape}"
                )
                deconvolved = device_deconvolver.deconvolve(slab)
                _log(
                    f"deconv done device={device} window[{scheduled.index:05d}] "
                    f"file={source.path.name} output_shape={deconvolved.shape} "
                    f"seconds={time.perf_counter() - t_deconv:.2f}"
                )
                core_indexes = [z - window.read_start for z in window.core_z]
                core = deconvolved[np.asarray(core_indexes, dtype=np.int64)]
                sample_path = sample_dir / f"{source.path.stem}-window{scheduled.index:05d}.tif"
                t_save = time.perf_counter()
                save_float32_sample(
                    sample_path,
                    core,
                    metadata={"axes": "ZYX", "source": str(source.path), "core_z": list(window.core_z)},
                )
                _log(
                    f"sample saved device={device} window[{scheduled.index:05d}] "
                    f"path={sample_path} core_shape={core.shape} seconds={time.perf_counter() - t_save:.2f}"
                )
                local_samples.append(sample_path)
                local_manifest.append(
                    {
                        "file": str(source.path),
                        "device": device,
                        "read_start": window.read_start,
                        "read_stop": window.read_stop,
                        "sampled_z": list(window.core_z),
                        "sample_path": str(sample_path),
                    }
                )
            except Exception as exc:
                if stop_on_error:
                    raise
                local_failures.append(
                    f"{source.path} window [{window.read_start}, {window.read_stop}) on device {device}: {exc}"
                )
        return local_samples, local_manifest, local_failures

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(process_device_jobs, device, jobs)
            for device, jobs in _jobs_by_device(schedule).items()
        ]
        for future in as_completed(futures):
            local_samples, local_manifest, local_failures = future.result()
            sample_paths.extend(local_samples)
            manifest_windows.extend(local_manifest)
            failures.extend(local_failures)

    if failures:
        joined = "\n".join(failures)
        raise ProcessingError(f"{len(failures)} sample-scale job(s) failed:\n{joined}")
    _log(f"sample jobs complete samples={len(sample_paths)} windows={len(manifest_windows)}")
    sample_paths.sort()
    manifest_windows.sort(
        key=lambda item: (item["file"], item["read_start"], item["read_stop"], item["sample_path"])
    )

    manifest = {
        "inputs": [str(path) for path in paths],
        "source_header_mode": "first_input",
        "source_header_template": str(template_source.path),
        "seed": int(seed),
        "channels": int(channels),
        "halo": int(halo),
        "psfs": file_provenance_records(psfs),
        "basic_profiles": file_provenance_records(basic_paths or []),
        "devices": [int(device) for device in devices],
        "queue_depth": int(queue_depth),
        "versions": dependency_versions(),
        "metadata": [
            {
                "file": str(path),
                "metadata_hash": template_source.metadata.metadata_hash,
                "raw_shape": list(template_source.metadata.raw_shape),
                "raw_dtype": template_source.metadata.raw_dtype,
            }
            for path in paths
        ],
        "windows": manifest_windows,
    }
    _log(f"collate scaling start sample_files={len(sample_paths)} out_dir={out_dir}")
    collate_scaling(
        sample_paths,
        channels=channels,
        out_dir=out_dir,
        p_low=p_low,
        p_high=p_high,
        gamma=gamma,
        bins=bins,
        manifest=manifest,
        overwrite=overwrite,
    )
    _log(f"sample-scale complete seconds={time.perf_counter() - t_workflow:.2f}")


def run_streaming_deconv(
    inputs: Sequence[Path],
    *,
    out_dir: Path,
    scaling_path: Path | None,
    channels: int,
    halo: int,
    slab_depth: int,
    output_mode: str,
    deconvolver: Deconvolver | None,
    psf_paths: Sequence[Path] | None,
    basic_paths: Sequence[Path] | None = None,
    deconvolver_factory: Callable[[int], Deconvolver] | None = None,
    devices: list[int],
    queue_depth: int,
    stop_on_error: bool,
    overwrite: bool = False,
) -> None:
    t_workflow = time.perf_counter()
    _validate_queue_depth(queue_depth)
    if output_mode not in {"u16", "float32"}:
        raise ValueError(f"Unsupported output_mode={output_mode!r}; expected 'u16' or 'float32'.")
    if output_mode == "u16" and scaling_path is None:
        raise ValueError("output_mode='u16' requires a scaling path.")
    paths = [Path(path) for path in inputs]
    psfs = tuple(Path(path) for path in (psf_paths or ()))
    relative_root = output_relative_root(paths)
    _log(
        "run start "
        f"inputs={len(paths)} out_dir={out_dir} channels={channels} halo={halo} slab_depth={slab_depth} "
        f"output_mode={output_mode} devices={devices} queue_depth={queue_depth} overwrite={overwrite}"
    )
    _log(f"run scaling={scaling_path}")
    for psf_path in psfs:
        _log(f"run psf={psf_path}")
    for basic_path in basic_paths or []:
        _log(f"run basic={basic_path}")
    t_sources = time.perf_counter()
    template_source = TiffLogicalSource.open(paths[0], channels=channels, metadata_mode="summary")
    _log(
        "using first source header for all "
        f"{len(paths)} run input(s): template={paths[0]} z_count={template_source.z_count} "
        f"shape=({template_source.z_count}, {channels}, {template_source.height}, {template_source.width}) "
        f"dtype={template_source.dtype} seconds={time.perf_counter() - t_sources:.2f}"
    )
    scaling = load_scaling(scaling_path) if output_mode == "u16" else None
    if scaling is not None:
        validate_scaling_channels(scaling, channels=channels, context=f"Scaling file {scaling_path}")
    if (
        output_mode == "u16"
        and deconvolver is None
        and deconvolver_factory is not None
        and getattr(deconvolver_factory, "process_safe", False)
    ):
        if scaling is None or scaling_path is None:
            raise RuntimeError("Process GPU u16 run requires loaded scaling parameters.")
        failures = run_process_gpu_streaming_deconv(
            paths,
            template_source=template_source,
            scaling=scaling,
            deconvolver_factory=deconvolver_factory,
            config=ProcessRunConfig(
                out_dir=out_dir,
                channels=channels,
                halo=halo,
                slab_depth=slab_depth,
                output_mode=output_mode,
                psf_paths=psfs,
                basic_paths=tuple(Path(path) for path in (basic_paths or ())),
                scaling_path=scaling_path,
                devices=tuple(int(device) for device in devices),
                queue_depth=queue_depth,
                overwrite=overwrite,
                output_relative_root=relative_root,
            ),
            stop_on_error=stop_on_error,
        )
        if failures:
            joined = "\n".join(failures)
            raise ProcessingError(f"{len(failures)} run job(s) failed:\n{joined}")
        _log(f"run complete seconds={time.perf_counter() - t_workflow:.2f}")
        return
    schedule = schedule_round_robin(len(paths), devices)
    _log(f"run schedule ready jobs={len(schedule)} devices={devices}")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    def process_device_jobs(device: int, jobs: list[ScheduledJob]) -> list[str]:
        device_deconvolver = _device_deconvolver(
            device,
            deconvolver=deconvolver,
            deconvolver_factory=deconvolver_factory,
        )
        if (
            output_mode == "u16"
            and deconvolver_factory is not None
            and not hasattr(device_deconvolver, "deconvolve_core_u16")
        ):
            raise TypeError(
                "Production u16 output requires a deconvolver with deconvolve_core_u16 so quantization remains "
                "on the GPU until final uint16 transfer."
            )
        local_failures: list[str] = []
        _log(f"worker start device={device} files={len(jobs)}")
        for scheduled in jobs:
            source = replace(template_source, path=paths[scheduled.index])
            slabs = slab_windows([source.path], z_counts=[source.z_count], slab_depth=slab_depth, halo=halo)
            output_path = output_path_for(out_dir, source.path, relative_root=relative_root)
            _log(
                f"file start device={device} file_index={scheduled.index} file={source.path.name} "
                f"slabs={len(slabs)} output={output_path}"
            )

            def core_iter():
                def read_slab(item):
                    t_read = time.perf_counter()
                    _log(
                        f"read start device={device} file={source.path.name} "
                        f"read_z=[{item.read_start},{item.read_stop})"
                    )
                    data = source.read_window(item.read_start, item.read_stop)
                    _log(
                        f"read done device={device} file={source.path.name} "
                        f"read_z=[{item.read_start},{item.read_stop}) shape={data.shape} "
                        f"dtype={data.dtype} seconds={time.perf_counter() - t_read:.2f}"
                    )
                    return data

                for slab_index, (slab, read, read_error) in enumerate(
                    _prefetched_reads(
                        slabs,
                        queue_depth=queue_depth,
                        read=read_slab,
                    )
                ):
                    t_slab = time.perf_counter()
                    _log(
                        f"slab received device={device} file={source.path.name} slab={slab_index}/{len(slabs)} "
                        f"read_z=[{slab.read_start},{slab.read_stop}) core_z=[{slab.core_start},{slab.core_stop}) "
                        f"shape={None if read is None else read.shape}"
                    )
                    if read_error is not None:
                        raise read_error
                    t_deconv = time.perf_counter()
                    _log(
                        f"deconv start device={device} file={source.path.name} slab={slab_index}/{len(slabs)} "
                        f"input_shape={read.shape}"
                    )
                    core_start = slab.core_start - slab.read_start
                    core_stop = slab.core_stop - slab.read_start
                    if output_mode == "u16" and hasattr(device_deconvolver, "deconvolve_core_u16"):
                        if scaling is None:
                            raise RuntimeError(
                                "u16 output reached quantization without loaded scaling parameters."
                            )
                        chunk = device_deconvolver.deconvolve_core_u16(
                            read,
                            core_start=core_start,
                            core_stop=core_stop,
                            scaling=scaling,
                        )
                        _validate_u16_core_chunk(
                            chunk,
                            source=source,
                            core_planes=slab.core_stop - slab.core_start,
                        )
                        _log(
                            f"deconv+quant done device={device} file={source.path.name} "
                            f"slab={slab_index}/{len(slabs)} output_shape={chunk.shape} "
                            f"seconds={time.perf_counter() - t_deconv:.2f}"
                        )
                    else:
                        deconvolved = device_deconvolver.deconvolve(read)
                        _log(
                            f"deconv done device={device} file={source.path.name} slab={slab_index}/{len(slabs)} "
                            f"output_shape={deconvolved.shape} seconds={time.perf_counter() - t_deconv:.2f}"
                        )
                        core = deconvolved[core_start:core_stop]
                        if output_mode == "u16":
                            if scaling is None:
                                raise RuntimeError(
                                    "u16 output reached quantization without loaded scaling parameters."
                                )
                            chunk = quantize_global(core, scaling)
                        else:
                            chunk = core.astype(np.float32, copy=False).reshape(
                                -1, source.height, source.width
                            )
                    _log(
                        f"slab ready device={device} file={source.path.name} slab={slab_index}/{len(slabs)} "
                        f"chunk_shape={chunk.shape} seconds={time.perf_counter() - t_slab:.2f}"
                    )
                    yield chunk

            provenance = provenance_payload(
                source.path,
                channels=channels,
                halo=halo,
                psf_paths=psfs,
                basic_paths=basic_paths,
                output_mode=output_mode,
                scaling_path=scaling_path,
                devices=devices,
                queue_depth=queue_depth,
            )
            sidecar_payload = {
                "source_metadata": asdict(source.metadata),
                "provenance": provenance,
                "chunking": {"slab_depth": slab_depth, "halo": halo},
                "output_mode": output_mode,
                "compression_tiff_tag": compression_tiff_tag(output_mode),
            }
            sidecar_text = json_dumps_strict(
                sidecar_payload,
                context=f"Deconvolution sidecar for {source.path}",
                indent=2,
            )
            try:
                sidecar = output_sidecar_path(output_path)
                if sidecar.exists() and not overwrite:
                    raise FileExistsError(
                        f"Refusing to overwrite existing sidecar {sidecar}; pass --overwrite to replace it."
                    )
                t_file = time.perf_counter()
                write_streamed_ome_zarr(
                    output_path,
                    source=source,
                    core_plane_chunks=core_iter(),
                    output_mode=output_mode,
                    provenance=provenance,
                    overwrite=overwrite,
                )
            except Exception as exc:
                if stop_on_error:
                    raise
                local_failures.append(f"{source.path} on device {device}: {exc}")
                continue
            sidecar.write_text(sidecar_text)
            _log(
                f"file complete device={device} file_index={scheduled.index} file={source.path.name} "
                f"seconds={time.perf_counter() - t_file:.2f}"
            )
        return local_failures

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(process_device_jobs, device, jobs)
            for device, jobs in _jobs_by_device(schedule).items()
        ]
        for future in as_completed(futures):
            failures.extend(future.result())

    if failures:
        joined = "\n".join(failures)
        raise ProcessingError(f"{len(failures)} run job(s) failed:\n{joined}")
    _log(f"run complete seconds={time.perf_counter() - t_workflow:.2f}")


def _jobs_by_device(schedule: Sequence[ScheduledJob]) -> dict[int, list[ScheduledJob]]:
    jobs: dict[int, list[ScheduledJob]] = {}
    for scheduled in schedule:
        jobs.setdefault(scheduled.device, []).append(scheduled)
    return jobs


def _device_deconvolver(
    device: int,
    *,
    deconvolver: Deconvolver | None,
    deconvolver_factory: Callable[[int], Deconvolver] | None,
) -> Deconvolver:
    if deconvolver_factory is not None:
        return deconvolver_factory(device)
    if deconvolver is None:
        raise ValueError("A deconvolver or deconvolver_factory is required.")
    return deconvolver


def _preflight_device_deconvolvers(
    devices: Sequence[int],
    *,
    deconvolver: Deconvolver | None,
    deconvolver_factory: Callable[[int], Deconvolver] | None,
    context: str,
) -> None:
    if deconvolver_factory is None:
        if deconvolver is None:
            raise ValueError("A deconvolver or deconvolver_factory is required.")
        return
    for device in devices:
        t_start = time.perf_counter()
        _log(f"{context} deconvolver preflight start device={device}")
        _device_deconvolver(
            int(device),
            deconvolver=deconvolver,
            deconvolver_factory=deconvolver_factory,
        )
        _log(
            f"{context} deconvolver preflight done device={device} "
            f"seconds={time.perf_counter() - t_start:.2f}"
        )


def _validate_queue_depth(queue_depth: int) -> None:
    if queue_depth < 1:
        raise ValueError(f"queue_depth must be at least 1, got {queue_depth}")


def _validate_u16_core_chunk(chunk: np.ndarray, *, source: TiffLogicalSource, core_planes: int) -> None:
    expected_shape = (core_planes * source.channels, source.height, source.width)
    if chunk.dtype != np.uint16:
        raise TypeError(f"deconvolve_core_u16 must return uint16 data, got {chunk.dtype}.")
    if chunk.shape != expected_shape:
        raise ValueError(f"deconvolve_core_u16 returned shape {chunk.shape}, expected {expected_shape}.")


def _prefetched_reads(
    items: Sequence[Any],
    *,
    queue_depth: int,
    read: Callable[[Any], np.ndarray],
) -> Iterator[tuple[Any, np.ndarray, Exception | None]]:
    iterator = iter(items)
    pending: deque[tuple[Any, Future[np.ndarray]]] = deque()

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending.append((item, executor.submit(read, item)))
        return True

    with ThreadPoolExecutor(max_workers=queue_depth) as executor:
        for _ in range(queue_depth):
            if not submit_next(executor):
                break
        while pending:
            item, future = pending.popleft()
            try:
                result = future.result()
                error = None
            except Exception as exc:
                result = np.empty((0,), dtype=np.float32)
                error = exc
            submit_next(executor)
            yield item, result, error


def _prepare_sample_scale_outputs(out_dir: Path, *, sample_dir: Path, overwrite: bool) -> None:
    artifact_paths = [
        out_dir / "scaling.json",
        out_dir / "scaling.txt",
        out_dir / "histogram.csv",
        out_dir / "sample-manifest.json",
        out_dir / "scaling-qc.png",
    ]
    existing = [path for path in artifact_paths if path.exists()]
    if sample_dir.exists():
        existing.append(sample_dir)
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing sample-scale output(s): {names}")
    if overwrite:
        for path in artifact_paths:
            path.unlink(missing_ok=True)
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
