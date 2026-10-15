"""
GPU-aware Dask cluster utilities used by distributed_segmentation.

Provides a simple way to:
 - Start a SpecCluster with multiple workers per GPU (via CUDA_VISIBLE_DEVICES pinning)
 - Optionally start a LocalCUDACluster (one worker per GPU) when dask-cuda is available

Expose a context-managed cluster wrapper `myGPUCluster` with `.client` and
`.dashboard_link` to mirror the interface expected by distributed_segmentation.
"""

from __future__ import annotations

import functools
import getpass
import os
import subprocess
from typing import Any
from collections.abc import Callable

import dask
import distributed
from dask.distributed import Client

DEFAULT_WORKER_MEMORY = "64GB"
DEFAULT_CONFIG_FILENAME = "distributed_cellpose_dask_config.yaml"


# Construction is synchronous; restore the caller's configuration before returning.
# Nanny snapshots this configuration and passes it to its owned worker process.
CLUSTER_CONFIG = {
    "distributed.comm.timeouts.connect": "60s",
    "distributed.comm.timeouts.tcp": "120s",
    "distributed.scheduler.worker-ttl": "10m",
    "distributed.admin.tick.limit": "10m",
}


def _auto_device_tokens() -> list[str]:
    """Use nvidia-smi to build a sequential list of GPU indices."""
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        tokens: list[str] = []
        for idx, line in enumerate(out.splitlines()):
            if line.strip():
                tokens.append(str(idx))
        return tokens
    except Exception:
        return []


def _visible_devices() -> list[str]:
    """Respect CUDA_VISIBLE_DEVICES if provided, otherwise enumerate GPUs."""
    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        tokens = [token.strip() for token in env_devices.split(",") if token.strip()]
        if tokens:
            return tokens
    return _auto_device_tokens()


def _build_worker_spec(
    *,
    devices: list[str],
    workers_per_gpu: int,
    workers_per_device: tuple[int, ...] | None = None,
    threads_per_worker: int,
) -> dict[str, Any]:
    """Construct a SpecCluster worker spec with env pinning per GPU."""
    from distributed.nanny import Nanny

    worker_counts = workers_per_device or (workers_per_gpu,) * len(devices)
    if len(worker_counts) != len(devices):
        raise ValueError(
            f"workers_per_device has {len(worker_counts)} entries for {len(devices)} visible GPUs."
        )
    if any(count < 1 for count in worker_counts):
        raise ValueError("Every workers_per_device entry must be positive.")

    worker_spec: dict[str, Any] = {}
    for dev, (token, worker_count) in enumerate(zip(devices, worker_counts, strict=True)):
        for k in range(worker_count):
            name = f"gpu-{dev}-w{k}"
            worker_spec[name] = {
                "cls": Nanny,
                "options": {
                    "nthreads": threads_per_worker,
                    "memory_limit": DEFAULT_WORKER_MEMORY,
                    "env": {"CUDA_VISIBLE_DEVICES": token},
                },
            }
    return worker_spec


def _start_speccluster(
    *,
    workers_per_gpu: int,
    workers_per_device: tuple[int, ...] | None = None,
    threads_per_worker: int,
) -> tuple[Client, Any, str | None]:
    from distributed import Scheduler
    from distributed.deploy.spec import SpecCluster

    devices = _visible_devices()
    if not devices:
        raise RuntimeError("No GPUs detected; cannot start GPU SpecCluster")

    spec = _build_worker_spec(
        devices=devices,
        workers_per_gpu=int(workers_per_gpu),
        workers_per_device=workers_per_device,
        threads_per_worker=int(threads_per_worker),
    )
    with dask.config.set(CLUSTER_CONFIG):
        cluster = SpecCluster(workers=spec, scheduler={"cls": Scheduler, "options": {}})
        return _connect_cluster(cluster)


def _start_localcuda(*, n_workers: int | None, threads_per_worker: int) -> tuple[Client, Any, str | None]:
    try:
        from dask_cuda import LocalCUDACluster  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError("dask-cuda is required for LocalCUDACluster but is not installed") from e

    with dask.config.set(CLUSTER_CONFIG):
        cluster = LocalCUDACluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=DEFAULT_WORKER_MEMORY,
            silence_logs=False,
        )
        return _connect_cluster(cluster)


def _connect_cluster(cluster: Any) -> tuple[Client, Any, str | None]:
    try:
        client = Client(cluster, set_as_default=False)
    except BaseException:
        cluster.close(timeout=30)
        raise
    return client, cluster, getattr(cluster, "dashboard_link", None)


class myGPUCluster:
    """Context-managed GPU cluster exposing `.client` and `.dashboard_link`.

    Parameters
    ----------
    workers_per_gpu : int
        If >1, uses SpecCluster with CUDA_VISIBLE_DEVICES pinning and this many
        workers per GPU. If <=1 and `use_localcuda=True`, starts LocalCUDACluster.
    workers_per_device : tuple[int, ...] | None
        Optional worker count for each CUDA_VISIBLE_DEVICES entry, in order.
    threads_per_worker : int
        Dask threads per worker (default 1 for GPU-bound work).
    use_localcuda : bool
        If True and workers_per_gpu <= 1, use LocalCUDACluster (requires dask-cuda).
    n_workers : int | None
        When using LocalCUDACluster, number of workers (typically equals number of GPUs).
    """

    def __init__(
        self,
        *,
        workers_per_gpu: int = 2,
        workers_per_device: tuple[int, ...] | None = None,
        threads_per_worker: int = 1,
        use_localcuda: bool = False,
        n_workers: int | None = None,
        **_: Any,
    ) -> None:
        self._client: Client | None = None
        self._cluster: Any = None
        self._dashboard: str | None = None

        if workers_per_device is not None or (workers_per_gpu and workers_per_gpu > 1):
            self._client, self._cluster, self._dashboard = _start_speccluster(
                workers_per_gpu=workers_per_gpu,
                workers_per_device=workers_per_device,
                threads_per_worker=threads_per_worker,
            )
        elif use_localcuda:
            self._client, self._cluster, self._dashboard = _start_localcuda(
                n_workers=n_workers, threads_per_worker=threads_per_worker
            )
        else:
            # Fall back to SpecCluster with a single worker per GPU to avoid hard dependency on dask-cuda
            self._client, self._cluster, self._dashboard = _start_speccluster(
                workers_per_gpu=1, threads_per_worker=threads_per_worker
            )

        # Mirror attributes expected by distributed_segmentation
        self.client = self._client  # type: ignore[assignment]
        self.dashboard_link = self._dashboard

    def __enter__(self) -> myGPUCluster:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._client is not None:
                self._client.close(timeout=30)
        finally:
            try:
                if self._cluster is not None:
                    self._cluster.close(timeout=30)
            finally:
                self._client = None
                self._cluster = None
                self.client = None


# ----------------------- CPU LocalCluster ------------------------------------#
class myLocalCluster(distributed.LocalCluster):
    """
    This wrapper scopes configuration to cluster construction. Process workers
    inherit it through Nanny; the caller's Dask configuration is restored.

    For a list of full arguments (how to specify your worker resources) see:
    https://distributed.dask.org/en/latest/api.html#distributed.LocalCluster
    You need to know how many cpu cores and how much RAM your machine has.

    Most users will only need to specify:
    n_workers
    ncpus (number of physical cpu cores per worker)
    memory_limit (which is the limit per worker, should be a string like '16GB')
    threads_per_worker (for most workflows this should be 1)

    You can also modify any dask configuration option through the
    config argument.

    If your workstation has a GPU, one of the workers will have exclusive
    access to it by default. That worker will be much faster than the others.
    You may want to consider creating only one worker (which will have access
    to the GPU) and letting that worker process all blocks serially.
    """

    def __init__(
        self,
        ncpus: int,
        config: dict[str, Any] | None = None,
        config_name: str = DEFAULT_CONFIG_FILENAME,
        persist_config: bool = False,
        **kwargs: Any,
    ) -> None:
        if persist_config:
            raise ValueError("persist_config is unsupported; cluster configuration is scoped to startup")
        effective_config = {
            **CLUSTER_CONFIG,
            "temporary-directory": f"{os.getcwd()}/.{getpass.getuser()}_distributed_cellpose/",
            **(config or {}),
        }
        kwargs.setdefault("host", "")
        kwargs.setdefault("silence_logs", False)
        if kwargs.get("processes", True):
            environment_vars = {
                name: str(2 * ncpus)
                for name in (
                    "MKL_NUM_THREADS", "NUM_MKL_THREADS", "OPENBLAS_NUM_THREADS",
                    "OPENMP_NUM_THREADS", "OMP_NUM_THREADS",
                )
            }
            kwargs["env"] = {**environment_vars, **kwargs.get("env", {})}
        with dask.config.set(effective_config):
            super().__init__(**kwargs)
            self.client, _, _ = _connect_cluster(self)

        print("Cluster dashboard link: ", self.dashboard_link)

    def __enter__(self) -> myLocalCluster:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            self.client.close(timeout=30)
        finally:
            super().__exit__(exc_type, exc_value, traceback)


# ----------------------- decorator -------------------------------------------#
def cluster(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    This decorator ensures a function will run inside a cluster
    as a context manager. The decorated function, "func", must
    accept "cluster" and "cluster_kwargs" as parameters. If
    "cluster" is not None then the user has provided an existing
    cluster and we just run func. If "cluster" is None then
    "cluster_kwargs" are used to construct a new cluster, and
    the function is run inside that cluster context.
    """

    @functools.wraps(func)
    def create_or_pass_cluster(*args: Any, **kwargs: Any) -> Any:
        # TODO: this only checks if args are explicitly present in function call
        #       it does not check if they are set correctly in any way
        assert "cluster" in kwargs or "cluster_kwargs" in kwargs, (
            "Either cluster or cluster_kwargs must be defined"
        )
        if not "cluster" in kwargs:
            ck = dict(kwargs.get("cluster_kwargs", {}) or {})
            with myGPUCluster(**ck) as cluster:
                kwargs["cluster"] = cluster
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return create_or_pass_cluster
