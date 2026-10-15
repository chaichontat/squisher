import copy
import logging
import os

import dask
import pytest

from squisher_segment.segmentation.distributed import gpu_cluster


@pytest.fixture
def process_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_file = tmp_path / ".config/dask/distributed_cellpose_dask_config.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("user-owned: original\n")
    before = copy.deepcopy(dask.config.config)
    levels = {name: logging.getLogger(name).level for name in (
        "distributed", "distributed.scheduler", "distributed.worker",
        "distributed.nanny", "distributed.batched", "distributed.core",
    )}
    environment = {name: os.environ.get(name) for name in (
        "MKL_NUM_THREADS", "NUM_MKL_THREADS", "OPENBLAS_NUM_THREADS",
        "OPENMP_NUM_THREADS", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
    )}
    yield config_file, before, levels, environment
    dask.config.config.clear()
    dask.config.config.update(before)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def assert_process_preserved(state):
    config_file, config, levels, environment = state
    assert config_file.read_text() == "user-owned: original\n"
    assert dask.config.config == config
    assert {name: logging.getLogger(name).level for name in levels} == levels
    assert {name: os.environ.get(name) for name in environment} == environment


def test_no_devices_does_not_mutate_process_or_user_config(monkeypatch, process_state):
    monkeypatch.setattr(gpu_cluster, "_visible_devices", lambda: [])
    with pytest.raises(RuntimeError, match="No GPUs"):
        gpu_cluster.myGPUCluster()
    assert_process_preserved(process_state)


@pytest.fixture
def fake_backend(monkeypatch):
    import distributed.deploy.spec

    clusters = []
    clients = []

    class FakeCluster:
        dashboard_link = "test-dashboard"

        def __init__(self, **kwargs):
            self.options = kwargs
            self.config = copy.deepcopy(dask.config.config)
            self.closed = False
            clusters.append(self)

        def close(self, timeout):
            self.closed = True

    class FakeClient:
        def __init__(self, cluster, *, set_as_default):
            assert set_as_default is False
            self.cluster = cluster
            self.closed = False
            clients.append(self)

        def close(self, timeout):
            self.closed = True

    monkeypatch.setattr(gpu_cluster, "_visible_devices", lambda: ["0"])
    monkeypatch.setattr(distributed.deploy.spec, "SpecCluster", FakeCluster)
    monkeypatch.setattr(gpu_cluster, "Client", FakeClient)
    return clusters, clients


def test_two_clusters_restore_config_and_close_independently(fake_backend, process_state):
    clusters, clients = fake_backend
    first = gpu_cluster.myGPUCluster()
    second = gpu_cluster.myGPUCluster()
    assert_process_preserved(process_state)
    for backend in clusters:
        assert backend.config["distributed"]["comm"]["timeouts"]["connect"] == "60s"
        assert backend.config["distributed"]["scheduler"]["worker-ttl"] == "10m"
    first.__exit__(None, None, None)
    assert clusters[0].closed and clients[0].closed
    assert not clusters[1].closed and not clients[1].closed
    assert_process_preserved(process_state)
    second.__exit__(None, None, None)
    assert clusters[1].closed and clients[1].closed
    assert_process_preserved(process_state)


def test_client_startup_failure_closes_cluster(fake_backend, monkeypatch, process_state):
    clusters, _ = fake_backend

    def fail_client(cluster, **kwargs):
        raise RuntimeError("client failed")

    monkeypatch.setattr(gpu_cluster, "Client", fail_client)
    with pytest.raises(RuntimeError, match="client failed"):
        gpu_cluster.myGPUCluster()
    assert clusters[0].closed
    assert_process_preserved(process_state)


def test_client_close_failure_still_closes_cluster(fake_backend, process_state):
    clusters, clients = fake_backend
    wrapper = gpu_cluster.myGPUCluster()

    def fail_close(timeout):
        raise TimeoutError("client close failed")

    clients[0].close = fail_close
    with pytest.raises(TimeoutError, match="client close failed"):
        wrapper.__exit__(None, None, None)
    assert clusters[0].closed
    assert wrapper.client is None
    assert_process_preserved(process_state)


def test_cpu_cluster_scopes_config_and_preserves_home(process_state, tmp_path):
    with gpu_cluster.myLocalCluster(
        ncpus=1, n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None, scheduler_kwargs={"dashboard_address": ":0"},
        config={"distributed.comm.timeouts.connect": "45s", "temporary-directory": str(tmp_path)},
    ) as cluster:
        assert len(cluster.client.scheduler_info()["workers"]) == 1
        assert cluster.client.submit(sum, [1, 2]).result() == 3
        assert_process_preserved(process_state)
    assert_process_preserved(process_state)


def test_two_cpu_clusters_close_independently(process_state, tmp_path):
    options = dict(ncpus=1, n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None, scheduler_kwargs={"dashboard_address": ":0"})
    first = gpu_cluster.myLocalCluster(
        **options, config={"temporary-directory": str(tmp_path / "first")},
    )
    with gpu_cluster.myLocalCluster(
        **options, config={"temporary-directory": str(tmp_path / "second")},
    ) as second:
        try:
            assert first.client.submit(sum, [1, 2]).result() == 3
            assert_process_preserved(process_state)
        finally:
            first.__exit__(None, None, None)
        assert second.client.submit(sum, [3, 4]).result() == 7
        assert_process_preserved(process_state)
    assert_process_preserved(process_state)
