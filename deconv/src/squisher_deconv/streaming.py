from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np
import zarr

from squisher.jpegxr_zarr import DEFAULT_JPEGXR_LEVEL

from squisher_deconv.deconvolution import Deconvolver
from squisher_deconv.metadata import (
    czi_dataset_metadata_payload,
    dependency_versions,
    file_stat_record,
    file_provenance_records,
    json_dumps_strict,
)
from squisher_deconv.planning import (
    group_sample_windows,
    output_path_for,
    output_relative_root,
    output_sidecar_path,
    sample_planes_from_z_counts,
)
from squisher_deconv.process_workers import (
    ProcessRunConfig,
    run_process_gpu_sample_scale,
    run_process_gpu_streaming_deconv,
)
from squisher_deconv.scaling import (
    collate_scaling,
    load_scaling,
    save_float32_sample,
    validate_scaling_channels,
)
from squisher_deconv.scheduler import ScheduledJob, schedule_round_robin
from squisher_deconv.source import TiffLogicalSource


class ProcessingError(RuntimeError):
    """Raised after one or more scheduled deconvolution jobs failed."""


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[squisher-deconv {timestamp}] {message}", file=sys.stderr, flush=True)


def _open_template_source(
    paths: Sequence[Path], *, channels: int, context: str, metadata_mode: str
) -> TiffLogicalSource:
    if not paths:
        raise ValueError(f"{context} requires at least one input")
    t_source = time.perf_counter()
    source = TiffLogicalSource.open(paths[0], channels=channels, metadata_mode=metadata_mode)
    _log(
        f"{context} opened template source header in {time.perf_counter() - t_source:.2f}s: "
        f"path={source.path} axes={source.axes} z={source.z_count} channels={source.channels} "
        f"yx=({source.height},{source.width}) dtype={source.dtype}"
    )
    return source


def _source_for_path(template_source: TiffLogicalSource, path: Path) -> TiffLogicalSource:
    metadata_mode = "summary" if template_source.metadata.tags.get("metadata_mode") == "summary" else "full"
    source = TiffLogicalSource.open(
        Path(path), channels=template_source.channels, metadata_mode=metadata_mode
    )
    if (source.height, source.width) != (template_source.height, template_source.width):
        raise ValueError(
            f"{source.path} has yx=({source.height},{source.width}), expected "
            f"({template_source.height},{template_source.width}) from template {template_source.path}."
        )
    if source.dtype != template_source.dtype:
        raise ValueError(f"{source.path} has dtype={source.dtype}, expected {template_source.dtype}.")
    return source


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
    iterations: int | None = None,
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
        f"iterations={iterations} "
        f"devices={devices} queue_depth={queue_depth} seed={seed} p_low={p_low} p_high={p_high} "
        f"gamma={gamma} bins={bins} overwrite={overwrite}"
    )
    for psf_path in psfs:
        _log(f"sample-scale psf={psf_path}")
    for basic_path in basic_paths or []:
        _log(f"sample-scale basic={basic_path}")
    template_source = _open_template_source(
        paths, channels=channels, context="sample-scale", metadata_mode="summary"
    )
    sources = [_source_for_path(template_source, path) for path in paths]
    z_counts = [source.z_count for source in sources]
    samples = sample_planes_from_z_counts(paths, z_counts=z_counts, planes=planes, seed=seed)
    windows = group_sample_windows(samples, z_counts=z_counts, halo=halo)
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
    sample_dir = out_dir / "float32-samples"
    _prepare_sample_scale_outputs(out_dir, sample_dir=sample_dir, overwrite=overwrite)
    sample_paths: list[Path] = []
    manifest_windows: list[dict[str, Any]] = []
    failures: list[str] = []

    if (
        deconvolver is None
        and deconvolver_factory is not None
        and getattr(deconvolver_factory, "process_safe", False)
    ):
        sample_paths, manifest_windows, failures = run_process_gpu_sample_scale(
            windows,
            paths=paths,
            template_source=template_source,
            sample_dir=sample_dir,
            deconvolver_factory=deconvolver_factory,
            devices=devices,
            queue_depth=queue_depth,
            stop_on_error=stop_on_error,
        )
    else:
        schedule = schedule_round_robin(len(windows), devices)
        device_deconvolvers = _preflight_device_deconvolvers(
            devices,
            deconvolver=deconvolver,
            deconvolver_factory=deconvolver_factory,
            context="sample-scale",
        )

        def process_device_jobs(
            device: int, jobs: list[ScheduledJob]
        ) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
            local_samples: list[Path] = []
            local_manifest: list[dict[str, Any]] = []
            local_failures: list[str] = []

            def read_window_for_job(item: ScheduledJob) -> np.ndarray:
                window = windows[item.index]
                source = _source_for_path(template_source, paths[window.file_index])
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
                source = _source_for_path(template_source, paths[window.file_index])
                try:
                    if read_error is not None:
                        raise read_error
                    t_deconv = time.perf_counter()
                    _log(
                        f"deconv start device={device} window[{scheduled.index:05d}] "
                        f"file={source.path.name} input_shape={slab.shape}"
                    )
                    deconvolved = device_deconvolvers[int(device)].deconvolve(slab)
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
        "source_header_mode": "per_input_summary",
        "seed": int(seed),
        "channels": int(channels),
        "halo": int(halo),
        "iterations": None if iterations is None else int(iterations),
        "psfs": file_provenance_records(psfs),
        "basic_profiles": file_provenance_records(basic_paths or []),
        "devices": [int(device) for device in devices],
        "queue_depth": int(queue_depth),
        "versions": dependency_versions(),
        "metadata": [
            {
                "file": str(source.path),
                "metadata_hash": source.metadata.metadata_hash,
                "raw_shape": list(source.metadata.raw_shape),
                "raw_dtype": source.metadata.raw_dtype,
            }
            for source in sources
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
    psf_paths: Sequence[Path] | None,
    basic_paths: Sequence[Path] | None = None,
    deconvolver_factory: Callable[[int], Deconvolver] | None = None,
    devices: list[int],
    queue_depth: int,
    stop_on_error: bool,
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    t_workflow = time.perf_counter()
    _validate_queue_depth(queue_depth)
    if scaling_path is None:
        raise ValueError("run requires a scaling path.")
    if deconvolver_factory is None or not getattr(deconvolver_factory, "process_safe", False):
        raise ValueError("run requires a process-safe deconvolver factory.")
    paths = [Path(path) for path in inputs]
    psfs = tuple(Path(path) for path in (psf_paths or ()))
    _log(
        "run start "
        f"inputs={len(paths)} out_dir={out_dir} channels={channels} halo={halo} slab_depth={slab_depth} "
        f"output_mode=u16 jpegxr_level={jpegxr_level} devices={devices} "
        f"queue_depth={queue_depth} overwrite={overwrite}"
    )
    _log(f"run scaling={scaling_path}")
    for psf_path in psfs:
        _log(f"run psf={psf_path}")
    for basic_path in basic_paths or []:
        _log(f"run basic={basic_path}")
    template_source = _open_template_source(paths, channels=channels, context="run", metadata_mode="full")
    iterations = getattr(deconvolver_factory, "iterations", None)
    relative_root = output_relative_root(paths)
    metadata_path = out_dir / "metadata.json"
    partial_metadata_path = metadata_path.with_name(f".{metadata_path.name}.partial")
    if metadata_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"Refusing to overwrite existing dataset metadata {metadata_path}; pass --overwrite to replace it."
        )
    if partial_metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing partial dataset metadata {partial_metadata_path}."
        )
    output_paths = [output_path_for(out_dir, path, relative_root=relative_root) for path in paths]
    if resume:
        expected_resume_identity = {
            "run_settings": {
                "channels": int(channels),
                "halo": int(halo),
                "slab_depth": int(slab_depth),
                "iterations": None if iterations is None else int(iterations),
                "output_mode": "u16",
                "jpegxr_level": float(jpegxr_level),
            },
            "psfs": file_provenance_records(psfs),
            "basic_profiles": file_provenance_records(basic_paths or []),
            "scaling": file_provenance_records([scaling_path])[0],
        }
        work_paths = _resume_pending_paths(
            paths,
            output_paths,
            expected_identity=expected_resume_identity,
        )
        _log(f"run resume complete={len(paths) - len(work_paths)} pending={len(work_paths)}")
    else:
        work_paths = paths
    metadata_text = (
        json_dumps_strict(
            czi_dataset_metadata_payload(paths, output_paths),
            context=f"Dataset metadata for {out_dir}",
            indent=2,
        )
        + "\n"
    )
    scaling = load_scaling(scaling_path)
    validate_scaling_channels(scaling, channels=channels, context=f"Scaling file {scaling_path}")
    if not work_paths:
        _write_dataset_metadata(metadata_path, metadata_text)
        _log(f"run complete seconds={time.perf_counter() - t_workflow:.2f}")
        return
    failures = run_process_gpu_streaming_deconv(
        work_paths,
        template_source=template_source,
        scaling=scaling,
        deconvolver_factory=deconvolver_factory,
        config=ProcessRunConfig(
            out_dir=out_dir,
            channels=channels,
            halo=halo,
            slab_depth=slab_depth,
            output_mode="u16",
            psf_paths=psfs,
            basic_paths=tuple(Path(path) for path in (basic_paths or ())),
            scaling_path=scaling_path,
            devices=tuple(int(device) for device in devices),
            queue_depth=queue_depth,
            overwrite=overwrite,
            output_relative_root=relative_root,
            iterations=None if iterations is None else int(iterations),
            jpegxr_level=jpegxr_level,
        ),
        stop_on_error=stop_on_error,
    )
    if failures:
        joined = "\n".join(failures)
        raise ProcessingError(f"{len(failures)} run job(s) failed:\n{joined}")
    _write_dataset_metadata(metadata_path, metadata_text)
    _log(f"run complete seconds={time.perf_counter() - t_workflow:.2f}")


def _write_dataset_metadata(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        partial.write_text(text)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _resume_pending_paths(
    paths: Sequence[Path],
    output_paths: Sequence[Path],
    *,
    expected_identity: dict[str, Any],
) -> list[Path]:
    pending: list[Path] = []
    for source, output in zip(paths, output_paths, strict=True):
        sidecar = output_sidecar_path(output)
        if not output.exists() and not sidecar.exists():
            pending.append(source)
            continue
        if not output.is_dir() or not sidecar.is_file():
            raise FileExistsError(
                f"Cannot resume {source}: expected complete output pair {output} and {sidecar}."
            )
        root = zarr.open_group(str(output), mode="r")
        if root.attrs.get("squisher_complete") is not True:
            raise ValueError(f"Cannot resume {source}: {output} is missing squisher_complete=true.")
        try:
            provenance = json.loads(sidecar.read_text())["provenance"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Cannot resume {source}: {sidecar} has invalid provenance.") from exc
        if provenance.get("source_file") != file_stat_record(source):
            raise ValueError(f"Cannot resume {source}: recorded source identity differs from the current file.")
        expected_settings = expected_identity["run_settings"]
        recorded_settings = provenance.get("run_settings")
        if not isinstance(recorded_settings, dict) or {
            key: recorded_settings.get(key) for key in expected_settings
        } != expected_settings:
            raise ValueError(f"Cannot resume {source}: recorded run settings differ from the current run.")
        for key in ("psfs", "basic_profiles", "scaling"):
            if provenance.get(key) != expected_identity[key]:
                raise ValueError(f"Cannot resume {source}: recorded {key} differ from the current run.")
        source_summary = TiffLogicalSource.open(
            source,
            channels=int(expected_settings["channels"]),
            metadata_mode="summary",
        )
        expected_shape = (
            source_summary.channels,
            source_summary.z_count,
            source_summary.height,
            source_summary.width,
        )
        if "0" not in root:
            raise ValueError(f"Cannot resume {source}: {output} is missing level-0 array '0'.")
        level0 = root["0"]
        if tuple(level0.shape) != expected_shape:
            raise ValueError(
                f"Cannot resume {source}: {output} level-0 shape {tuple(level0.shape)} "
                f"differs from expected {expected_shape}."
            )
        if np.dtype(level0.dtype) != np.dtype(np.uint16):
            raise ValueError(f"Cannot resume {source}: {output} level-0 dtype is {level0.dtype}, expected uint16.")
    return pending


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
) -> dict[int, Deconvolver]:
    if deconvolver_factory is None:
        if deconvolver is None:
            raise ValueError("A deconvolver or deconvolver_factory is required.")
        return {int(device): deconvolver for device in devices}
    device_deconvolvers: dict[int, Deconvolver] = {}
    for device in devices:
        device = int(device)
        t_start = time.perf_counter()
        _log(f"{context} deconvolver preflight start device={device}")
        device_deconvolvers[device] = _device_deconvolver(
            device,
            deconvolver=deconvolver,
            deconvolver_factory=deconvolver_factory,
        )
        _log(
            f"{context} deconvolver preflight done device={device} "
            f"seconds={time.perf_counter() - t_start:.2f}"
        )
    return device_deconvolvers


def _validate_queue_depth(queue_depth: int) -> None:
    if queue_depth < 1:
        raise ValueError(f"queue_depth must be at least 1, got {queue_depth}")


def _prefetched_reads(
    items: Sequence[Any],
    *,
    queue_depth: int,
    read: Callable[[Any], np.ndarray],
) -> Iterator[tuple[Any, np.ndarray, Exception | None]]:
    if queue_depth == 1:
        for item in items:
            try:
                yield item, read(item), None
            except Exception as exc:
                yield item, np.empty((0,), dtype=np.float32), exc
        return

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
