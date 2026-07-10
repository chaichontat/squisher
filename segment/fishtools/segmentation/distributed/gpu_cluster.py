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
import pathlib
import subprocess
from pathlib import Path
from typing import Any
from collections.abc import Callable

import dask
import distributed
import yaml
from dask.distributed import Client

DEFAULT_WORKER_MEMORY = "64GB"
DEFAULT_CONFIG_FILENAME = "distributed_cellpose_dask_config.yaml"


# ----------------------- config stuff ----------------------------------------#
def _config_path(config_name: str) -> str:
    return str(pathlib.Path.home()) + "/.config/dask/" + config_name


def _modify_dask_config(
    config: dict[str, Any],
    config_name: str = DEFAULT_CONFIG_FILENAME,
) -> None:
    # Default: significantly quiet Dask/distributed logging so worker/scheduler
    # noise does not obscure application-level exceptions. Users can override
    # these levels by providing their own "logging" section in ``config``.
    logging_defaults: dict[str, Any] = {
        "logging": {
            "distributed": "error",
            "distributed.scheduler": "error",
            # Hide worker chatter entirely (including warnings) by raising
            # its threshold to CRITICAL.
            "distributed.worker": "critical",
            "distributed.nanny": "error",
            "distributed.worker_state_machine": "critical",
        }
    }
    merged_config: dict[str, Any] = {**logging_defaults, **config}
    dask.config.set(merged_config)
    cfg_path = Path(_config_path(config_name))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(dask.config.config, f, default_flow_style=False)


def _remove_config_file(
    config_name: str = DEFAULT_CONFIG_FILENAME,
) -> None:
    config_path = _config_path(config_name)
    if os.path.exists(config_path):
        os.remove(config_path)


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


def _gpu_count() -> int:
    return len(_visible_devices())


def _build_worker_spec(
    *, devices: list[str], workers_per_gpu: int, threads_per_worker: int
) -> dict[str, Any]:
    """Construct a SpecCluster worker spec with env pinning per GPU."""
    from distributed.nanny import Nanny

    worker_spec: dict[str, Any] = {}
    for dev, token in enumerate(devices):
        for k in range(workers_per_gpu):
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


def _start_speccluster(*, workers_per_gpu: int, threads_per_worker: int) -> tuple[Client, Any, str | None]:
    from distributed import Scheduler
    from distributed.deploy.spec import SpecCluster

    # Ensure dask config (including logging defaults) is written so that
    # scheduler and worker processes inherit the quieter logging setup.
    _modify_dask_config({}, DEFAULT_CONFIG_FILENAME)

    devices = _visible_devices()
    if not devices:
        raise RuntimeError("No GPUs detected; cannot start GPU SpecCluster")

    spec = _build_worker_spec(
        devices=devices,
        workers_per_gpu=int(workers_per_gpu),
        threads_per_worker=int(threads_per_worker),
    )
    cluster = SpecCluster(workers=spec, scheduler={"cls": Scheduler, "options": {}})
    client = Client(cluster)
    return client, cluster, getattr(cluster, "dashboard_link", None)


def _start_localcuda(*, n_workers: int | None, threads_per_worker: int) -> tuple[Client, Any, str | None]:
    try:
        from dask_cuda import LocalCUDACluster  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError("dask-cuda is required for LocalCUDACluster but is not installed") from e

    cluster = LocalCUDACluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=DEFAULT_WORKER_MEMORY,
    )
    client = Client(cluster)
    return client, cluster, getattr(cluster, "dashboard_link", None)


class myGPUCluster:
    """Context-managed GPU cluster exposing `.client` and `.dashboard_link`.

    Parameters
    ----------
    workers_per_gpu : int
        If >1, uses SpecCluster with CUDA_VISIBLE_DEVICES pinning and this many
        workers per GPU. If <=1 and `use_localcuda=True`, starts LocalCUDACluster.
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
        threads_per_worker: int = 1,
        use_localcuda: bool = False,
        n_workers: int | None = None,
        **_: Any,
    ) -> None:
        self._client: Client | None = None
        self._cluster: Any = None
        self._dashboard: str | None = None

        if workers_per_gpu and workers_per_gpu > 1:
            self._client, self._cluster, self._dashboard = _start_speccluster(
                workers_per_gpu=workers_per_gpu, threads_per_worker=threads_per_worker
            )
        else:
            if use_localcuda:
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
        import logging

        # Suppress noisy shutdown logs from distributed
        for name in [
            "distributed",
            "distributed.scheduler",
            "distributed.worker",
            "distributed.nanny",
            "distributed.batched",
            "distributed.core",
        ]:
            logging.getLogger(name).setLevel(logging.CRITICAL)

        try:
            if self._client is not None:
                try:
                    self._client.close(timeout=30)
                except TimeoutError as close_exc:
                    logging.getLogger(__name__).warning(
                        f"Timed out while closing Dask client (cluster shutdown); ignoring: {close_exc!r}"
                    )
            if self._cluster is not None:
                try:
                    self._cluster.close(timeout=30)
                except TimeoutError as close_exc:
                    logging.getLogger(__name__).warning(
                        f"Timed out while closing Dask cluster (cluster shutdown); ignoring: {close_exc!r}"
                    )
        finally:
            # Clean up transient dask config used for logging control.
            _remove_config_file(DEFAULT_CONFIG_FILENAME)
            self._client = None
            self._cluster = None


# ----------------------- CPU LocalCluster ------------------------------------#
class myLocalCluster(distributed.LocalCluster):
    """
    This is a thin wrapper extending dask.distributed.LocalCluster to set
    configs before the cluster or workers are initialized.

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
        # config
        self.config_name = config_name
        self.persist_config = persist_config
        scratch_dir = f"{os.getcwd()}/"
        scratch_dir += f".{getpass.getuser()}_distributed_cellpose/"
        config_defaults = {"temporary-directory": scratch_dir}
        if config is None:
            config = {}
        config = {**config_defaults, **config}
        _modify_dask_config(config, config_name)

        # construct
        if "host" not in kwargs:
            kwargs["host"] = ""
        super().__init__(**kwargs)
        self.client = distributed.Client(self)

        # set environment variables for workers (threading)
        environment_vars = {
            "MKL_NUM_THREADS": str(2 * ncpus),
            "NUM_MKL_THREADS": str(2 * ncpus),
            "OPENBLAS_NUM_THREADS": str(2 * ncpus),
            "OPENMP_NUM_THREADS": str(2 * ncpus),
            "OMP_NUM_THREADS": str(2 * ncpus),
        }

        def set_environment_vars():
            for k, v in environment_vars.items():
                os.environ[k] = v

        self.client.run(set_environment_vars)

        print("Cluster dashboard link: ", self.dashboard_link)

    def __enter__(self) -> myLocalCluster:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if not self.persist_config:
            _remove_config_file(self.config_name)
        import logging

        try:
            self.client.close(timeout=30)
        except TimeoutError as close_exc:
            logging.getLogger(__name__).warning(
                f"Timed out while closing Dask client (cluster shutdown); ignoring: {close_exc!r}"
            )

        try:
            super().__exit__(exc_type, exc_value, traceback)
        except TimeoutError as close_exc:
            logging.getLogger(__name__).warning(
                f"Timed out while closing Dask LocalCluster (cluster shutdown); ignoring: {close_exc!r}"
            )


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
