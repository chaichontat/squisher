import json
import pickle
import warnings
from pathlib import Path

import dask
import numpy as np
import pytest
import zarr
from click.testing import CliRunner as ClickCliRunner
from dask.array.core import PerformanceWarning
from typer.testing import CliRunner as TyperCliRunner

from squisher_segment.cli import app
from squisher_segment.segmentation.distributed import cache_utils
from squisher_segment.segmentation.distributed import distributed_segmentation as segmentation
from squisher_segment.segmentation.distributed import gpu_cluster
from squisher_segment.segmentation.distributed import merge_utils


def _write_input(path: Path) -> tuple[zarr.Array, np.ndarray]:
    data = np.arange(2 * 4 * 5 * 3, dtype=np.uint16).reshape(2, 4, 5, 3)
    array = zarr.create_array(path, data=data, chunks=(1, 2, 3, 1))
    array.attrs["key"] = ["dna", "membrane", "far-red"]
    return array, data


def _runtime_artifacts(tag: str = "test") -> dict[str, object]:
    return {
        "trt_plans": [
            {
                "model_role": "xy",
                "device_name": "Test GPU",
                "path": f"/{tag}.plan",
                "sha256": tag,
                "size": 1,
                "mtime_ns": 1,
            }
        ],
        "source_sha256": {"pipeline": tag},
        "input_provenance": [{"path": "/manifest.json", "sha256": tag}],
    }


def _input_identity(path: Path, array: zarr.Array) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "dtype": str(array.dtype),
        "attrs": dict(array.attrs),
    }


def _valid_run_identity(tag: str = "test") -> dict[str, object]:
    return {
        "schema_version": 6,
        "input": {"path": f"/{tag}.zarr"},
        "channel_indices": [0],
        "model_sha256": {"xy": tag},
        "model_kwargs": {},
        "eval_kwargs": {},
        "blocksize": [1, 1, 1, 1],
        "overlap": 0,
        "nonempty_rule": {
            "channel": "561",
            "channel_index": 0,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
        "mask": None,
        "preprocessing_steps": [],
        "cellpose_version": "test",
        "runtime_artifacts": _runtime_artifacts(tag),
        "stitching": {"mode": "face", "iou_threshold": 0.25},
    }


def _write_ome_channel(path: Path, data: np.ndarray) -> None:
    root = zarr.create_group(path)
    array = root.create_array("0", data=data, chunks=(1, 2, 3))
    array.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    root.attrs.update(
        {
            "squisher_complete": True,
            "ome": {
                "multiscales": [
                    {
                        "axes": [{"name": axis} for axis in ("z", "y", "x")],
                        "datasets": [{"path": "0"}],
                    }
                ]
            },
        }
    )


def test_retire_worker_keeps_nanny_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduled: dict[str, object] = {}

    class FakeLoop:
        def add_callback(self, callback: object, **kwargs: object) -> None:
            scheduled["callback"] = callback
            scheduled["kwargs"] = kwargs

    class FakeWorker:
        loop = FakeLoop()

        def close(self, **kwargs: object) -> None:
            raise AssertionError("close must be scheduled on the worker loop")

    worker = FakeWorker()
    monkeypatch.setattr(segmentation.distributed, "get_worker", lambda: worker)

    segmentation._retire_worker_after_error(reason="CUDA OOM")

    assert scheduled["callback"] == worker.close
    assert scheduled["kwargs"] == {"nanny": False, "reason": "CUDA OOM"}
    assert worker._squisher_segment_fatal_error is True


def test_retire_worker_without_loop_keeps_nanny_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    close_kwargs: list[dict[str, object]] = []

    class FakeWorker:
        loop = None

        def close(self, **kwargs: object) -> None:
            close_kwargs.append(kwargs)

    worker = FakeWorker()
    monkeypatch.setattr(segmentation.distributed, "get_worker", lambda: worker)

    segmentation._retire_worker_after_error(reason="CUDA OOM")

    assert close_kwargs == [{"nanny": False, "reason": "CUDA OOM"}]
    assert worker._squisher_segment_fatal_error is True


def test_registered_ome_manifest_reads_roi_and_survives_pickle(tmp_path: Path) -> None:
    source_405 = tmp_path / "c405.ome.zarr"
    source_561 = tmp_path / "c561.ome.zarr"
    data_405 = np.arange(3 * 5 * 6, dtype=np.uint16).reshape(3, 5, 6)
    data_561 = data_405 + 1000
    _write_ome_channel(source_405, data_405)
    _write_ome_channel(source_561, data_561)

    manifest_path = tmp_path / "input.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "squisher_segment.registered_ome_input.v1",
                "channels": ["405", "561"],
                "sources": [
                    {
                        "channel": "405",
                        "path": str(source_405),
                        "root_metadata_sha256": segmentation.file_sha256(source_405 / "zarr.json"),
                    },
                    {
                        "channel": "561",
                        "path": str(source_561),
                        "root_metadata_sha256": segmentation.file_sha256(source_561 / "zarr.json"),
                    },
                ],
                "source_roi_zyx": [[1, 3], [1, 5], [2, 6]],
                "shape": [2, 4, 4, 2],
                "dtype": "uint16",
            }
        )
    )

    array, identity, provenance = segmentation._open_registered_ome_input(manifest_path)
    restored = pickle.loads(pickle.dumps(array))

    expected = np.stack(
        [data_405[1:3, 1:5, 2:6], data_561[1:3, 1:5, 2:6]],
        axis=-1,
    )
    np.testing.assert_array_equal(restored[:, :, :, :], expected)
    np.testing.assert_array_equal(
        restored.get_orthogonal_selection((slice(0, 1), slice(1, 3), slice(0, 2), [1, 0])),
        expected[0:1, 1:3, 0:2][:, :, :, [1, 0]],
    )
    assert array.shape == (2, 4, 4, 2)
    assert array.attrs["key"] == ["405", "561"]
    assert identity["kind"] == "registered-ome-zarr"
    assert {Path(item["path"]).name for item in provenance} == {
        "input.json",
        "zarr.json",
    }


def test_crop_zyxc_input_reads_exact_box_and_binds_identity(tmp_path: Path) -> None:
    array, data = _write_input(tmp_path / "input.zarr")
    identity = _input_identity(tmp_path / "input.zarr", array)

    cropped, cropped_identity = segmentation._crop_zyxc_input(
        array,
        identity,
        start_zyx=(0, 1, 1),
        box_size_zyx=(2, 2, 3),
    )

    np.testing.assert_array_equal(
        cropped.get_orthogonal_selection((slice(None), slice(None), slice(None), [2, 0])),
        data[:, 1:3, 1:4][:, :, :, [2, 0]],
    )
    assert cropped.shape == (2, 2, 3, 3)
    assert cropped_identity["start_zyx"] == [0, 1, 1]
    assert cropped_identity["box_size_zyx"] == [2, 2, 3]
    assert cropped_identity["source"] == identity


@pytest.mark.parametrize(
    ("start", "size", "message"),
    [
        ((0, 0, 0), None, "provided together"),
        ((-1, 0, 0), (1, 1, 1), "non-negative"),
        ((0, 0, 0), (0, 1, 1), "positive"),
        ((1, 3, 3), (2, 2, 3), "outside input shape"),
    ],
)
def test_crop_zyxc_input_rejects_invalid_boxes(
    tmp_path: Path,
    start: tuple[int, int, int] | None,
    size: tuple[int, int, int] | None,
    message: str,
) -> None:
    array, _ = _write_input(tmp_path / "input.zarr")

    with pytest.raises(ValueError, match=message):
        segmentation._crop_zyxc_input(
            array,
            _input_identity(tmp_path / "input.zarr", array),
            start_zyx=start,
            box_size_zyx=size,
        )


def test_selected_channels_are_read_once_in_requested_order(tmp_path: Path) -> None:
    array, data = _write_input(tmp_path / "input.zarr")
    channel_indices, names = segmentation._resolve_channel_selection(array, "far-red,dna")
    selected_shape = array.shape[:-1] + (len(channel_indices),)
    blocksize = (array.shape[0], 3, 3, len(channel_indices))

    block_indices, crops = segmentation._segmentation_block_crops(
        selected_shape,
        blocksize,
        overlap=1,
        mask=None,
    )

    assert channel_indices == (2, 0)
    assert names == ("far-red", "dna")
    assert {index[-1] for index in block_indices} == {0}
    assert {(crop[-1].start, crop[-1].stop) for crop in crops} == {(0, 2)}

    crop = (slice(0, 1), slice(0, 2), slice(1, 4), slice(0, 2))
    selected = segmentation._read_input_crop(array, crop, channel_indices)
    np.testing.assert_array_equal(selected, data[0:1, 0:2, 1:4][:, :, :, [2, 0]])


def test_cellpose_input_is_masked_from_raw_561(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.zeros((1, 2, 3, 3), dtype=np.uint16)
    data[..., 0] = 10
    data[..., 2] = 30
    data[0, 0, 1, 1] = 1000
    data[0, 1, 2, 1] = 1001
    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        data=data,
        chunks=data.shape,
    )
    reads: list[tuple[int, ...]] = []
    original_read = segmentation._read_input_crop

    def record_read(zarr_array, crop, channel_indices):
        reads.append(channel_indices)
        return original_read(zarr_array, crop, channel_indices)

    captured: dict[str, np.ndarray] = {}

    class Model:
        def eval(self, image: np.ndarray, **kwargs: object):
            captured["image"] = image.copy()
            return np.zeros(image.shape[:-1], dtype=np.uint32), [], None, None

    def add_five(image: np.ndarray, *, crop: object = None) -> np.ndarray:
        return image.astype(np.float32) + 5

    monkeypatch.setattr(segmentation, "_read_input_crop", record_read)
    monkeypatch.setattr(segmentation, "get_cached_model", lambda kwargs: Model())
    monkeypatch.setattr(segmentation.cellpose.io, "logger_setup", lambda **kwargs: None)
    monkeypatch.setattr(
        segmentation.cp,
        "get_default_memory_pool",
        lambda: type("Pool", (), {"free_all_blocks": lambda self: None})(),
    )

    segmentation.read_preprocess_and_segment(
        input_zarr,
        (slice(0, 1), slice(0, 2), slice(0, 3), slice(0, 2)),
        (2, 0),
        1,
        1000,
        [(add_five, {})],
        {},
        {"normalize": {"lowhigh": [[-2.0, 40.0], [-3.0, 20.0]]}},
        None,
    )

    expected = np.empty((1, 2, 3, 2), dtype=np.float32)
    expected[..., 0] = -2.0
    expected[..., 1] = -3.0
    expected[0, 1, 2] = (35.0, 15.0)
    assert reads == [(2, 0, 1)]
    np.testing.assert_array_equal(captured["image"], expected)


def test_sam_blocksize_targets_haloed_cellpose_crop() -> None:
    blocksize = segmentation._sam_processing_blocksize(
        input_spatial_shape=(1_000, 1_000, 3_000),
        n_channels=3,
        diameter=30,
        target_nz=2,
        target_ny=2,
        target_nx=6,
    )

    block_indices, crops = segmentation._segmentation_block_crops(
        (1_000, 1_000, 3_000, 3),
        blocksize,
        overlap=60,
        mask=None,
    )

    assert blocksize == (280, 280, 1_144, 3)
    assert (1, 1, 1, 0) in block_indices
    center_crop = crops[block_indices.index((1, 1, 1, 0))]
    assert center_crop == (
        slice(220, 620),
        slice(220, 620),
        slice(1_084, 2_348),
        slice(0, 3),
    )


def test_sam_target_nz_changes_only_z_block_extent() -> None:
    blocksize = segmentation._sam_processing_blocksize(
        input_spatial_shape=(1_000, 1_000, 3_000),
        n_channels=3,
        diameter=30,
        target_nz=1,
        target_ny=2,
        target_nx=6,
    )

    assert blocksize == (120, 280, 1_144, 3)


def test_sam_blocksize_avoids_tiny_subset_remainder() -> None:
    blocksize = segmentation._sam_processing_blocksize(
        input_spatial_shape=(2_470, 1_024, 1_024),
        n_channels=3,
        diameter=30,
        target_nz=1,
        target_ny=4,
        target_nx=3,
    )

    block_indices, crops = segmentation._segmentation_block_crops(
        (2_470, 1_024, 1_024, 3),
        blocksize,
        overlap=60,
        mask=None,
    )

    assert blocksize == (120, 512, 512, 3)
    assert len(block_indices) == 21 * 2 * 2
    assert max(crop[1].stop - crop[1].start for crop in crops) <= 832
    assert max(crop[2].stop - crop[2].start for crop in crops) <= 624


def test_sam_blocksize_keeps_full_volume_interior_geometry() -> None:
    blocksize = segmentation._sam_processing_blocksize(
        input_spatial_shape=(2_470, 10_657, 7_871),
        n_channels=3,
        diameter=30,
        target_nz=1,
        target_ny=4,
        target_nx=3,
    )

    assert blocksize == (120, 712, 504, 3)


def test_normalization_block_is_bounded_by_cropped_input() -> None:
    assert segmentation._normalization_block_yx((256, 512, 512)) == (256, 512)
    assert segmentation._normalization_block_yx((2_470, 10_657, 7_871)) == (256, 1024)


def test_normalization_settings_can_disable_unsharp() -> None:
    settings = segmentation._normalization_settings(
        block_yx=(256, 1024),
        nonempty_rule={"channel": "561", "threshold": 1000},
        unsharp=False,
    )

    assert settings["unsharp"] is False
    assert settings["unsharp_backend"] is None
    assert settings["unsharp_dimensionality"] is None
    assert settings["unsharp_radius"] is None


def test_cellpose_eval_uses_unit_anisotropy_and_masks_only() -> None:
    eval_kwargs = segmentation._build_cellpose_eval_kwargs(
        diameter=30,
        normalization={"lowhigh": [[0.0, 1.0]] * 3},
        ortho_weights=[3, 1.0, 1.0],
        flow3d_smooth=1.0,
    )

    assert eval_kwargs["anisotropy"] == 1.0
    assert eval_kwargs["flow3D_smooth"] == 1.0
    assert eval_kwargs["return_flows"] is False
    assert eval_kwargs["skip_empty_tiles"] is True


def test_cellpose_eval_can_disable_empty_tile_checks() -> None:
    eval_kwargs = segmentation._build_cellpose_eval_kwargs(
        diameter=30,
        normalization={"lowhigh": [[0.0, 1.0]] * 3},
        ortho_weights=[3, 1.0, 1.0],
        flow3d_smooth=1.0,
        skip_empty_tiles=False,
    )

    assert eval_kwargs["skip_empty_tiles"] is False


def test_channel_selection_rejects_ambiguous_metadata(tmp_path: Path) -> None:
    array, _ = _write_input(tmp_path / "input.zarr")
    array.attrs["key"] = ["dna", "dna", "far-red"]

    with pytest.raises(ValueError, match="must be unique"):
        segmentation._resolve_channel_selection(array, "dna")


def test_run_state_is_bound_to_exact_identity(tmp_path: Path) -> None:
    config_path = tmp_path / "run_config.json"
    output_path = tmp_path / "output_segmentation-sam.zarr"
    zarr.create_array(output_path, shape=(2, 3, 4), chunks=(1, 3, 4), dtype=np.uint32)
    identity = {"schema_version": 1, "input": {"path": "/data/a"}, "channel_indices": [0]}
    changed = {"schema_version": 1, "input": {"path": "/data/a"}, "channel_indices": [1]}

    segmentation.save_run_config(config_path, identity)
    segmentation.validate_run_config(config_path, identity)
    segmentation.write_completion_marker(output_path, identity)

    assert segmentation.completed_run_matches(output_path, identity)
    assert segmentation.completion_marker_path(output_path) == tmp_path / "output_segmentation-sam.done"
    with pytest.raises(ValueError, match="Cannot resume"):
        segmentation.validate_run_config(config_path, changed)
    with pytest.raises(FileExistsError, match="different run"):
        segmentation.completed_run_matches(output_path, changed)


def test_nonempty_rule_resolves_561_channel_and_threshold(tmp_path: Path) -> None:
    input_zarr, _ = _write_input(tmp_path / "input.zarr")
    input_zarr.attrs["key"] = ["405", "561", "638"]

    assert segmentation._resolve_nonempty_rule(input_zarr, 750) == {
        "channel": "561",
        "channel_index": 1,
        "threshold": 750,
        "min_fraction": 0.05,
    }


def test_block_requires_five_percent_foreground_voxels(tmp_path: Path) -> None:
    data = np.zeros((1, 1, 100, 3), dtype=np.uint16)
    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        data=data,
        chunks=(1, 1, 100, 3),
    )
    crop = (slice(0, 1), slice(0, 1), slice(0, 100), slice(0, 3))

    input_zarr[0, 0, :4, 1] = 1001
    assert not segmentation._check_block_has_data(crop, input_zarr, 1, 1000)

    input_zarr[0, 0, 4, 1] = 1001
    assert segmentation._check_block_has_data(crop, input_zarr, 1, 1000)


def test_default_scan_caches_blocks_with_561_above_1000(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ImmediateFuture:
        def __init__(self, value: bool) -> None:
            self.value = value

        def add_done_callback(self, callback) -> None:
            callback(self)

        def result(self) -> bool:
            return self.value

    class ImmediateClient:
        thresholds: list[int] = []
        channel_indices: list[int] = []

        def map(self, function, crops, **kwargs):
            self.thresholds.append(kwargs["threshold"])
            self.channel_indices.append(kwargs["channel_index"])
            return [
                ImmediateFuture(
                    function(
                        crop,
                        kwargs["zarr_array"],
                        kwargs["channel_index"],
                        kwargs["threshold"],
                    )
                )
                for crop in crops
            ]

    input_zarr = zarr.create_array(
        tmp_path / "scan-input.zarr",
        data=np.array(
            [
                [5000, 1000, 5000],
                [5000, 1001, 5000],
                [5000, 0, 5000],
            ],
            dtype=np.uint16,
        ).reshape(1, 1, 3, 3),
        chunks=(1, 1, 1, 3),
    )
    crops = [(slice(0, 1), slice(0, 1), slice(i, i + 1), slice(0, 3)) for i in range(3)]
    reads: list[tuple[slice, ...]] = []
    original_read = segmentation._read_input_crop

    def count_read(zarr_array, crop, channel_indices):
        reads.append(crop)
        return original_read(zarr_array, crop, channel_indices)

    monkeypatch.setattr(segmentation, "_read_input_crop", count_read)
    monkeypatch.setattr(segmentation.distributed, "as_completed", lambda futures: futures)
    client = ImmediateClient()
    cache_path = tmp_path / "nonempty.json"

    selected = segmentation._select_input_blocks(
        client=client,
        block_crops=crops,
        input_zarr=input_zarr,
        nonempty_channel_index=1,
        nonempty_threshold=1000,
        blocksize=(1, 1, 1, 3),
        run_key="sparse-run",
        path_nonempty=cache_path,
    )

    assert client.thresholds == [1000]
    assert client.channel_indices == [1]
    assert reads == crops
    assert selected == [1]
    assert cache_utils.read_nonempty_cache(cache_path, (1, 1, 1, 3), "sparse-run") == [1]

    class NoScanClient:
        def map(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("matching cache must not submit scan futures")

    def fail_input_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("matching cache must not read input")

    monkeypatch.setattr(segmentation, "_read_input_crop", fail_input_read)

    cached = segmentation._select_input_blocks(
        client=NoScanClient(),
        block_crops=crops,
        input_zarr=input_zarr,
        nonempty_channel_index=1,
        nonempty_threshold=1000,
        blocksize=(1, 1, 1, 3),
        run_key="sparse-run",
        path_nonempty=cache_path,
    )

    assert cached == [1]


def test_distributed_eval_scans_block_cores_without_inference_halos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SelectionChecked(RuntimeError):
        pass

    class Client:
        def wait_for_workers(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {}

    class Cluster:
        client = Client()

    data = np.zeros((1, 1, 6, 3), dtype=np.uint16)
    data[0, 0, 2, 1] = 1001
    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        data=data,
        chunks=(1, 1, 2, 3),
    )
    input_zarr.attrs["key"] = ["405", "561", "638"]
    model_path = tmp_path / "model"
    model_path.write_bytes(b"model")
    run_identity = segmentation._build_run_identity(
        input_identity=_input_identity(tmp_path / "input.zarr", input_zarr),
        channel_indices=(0,),
        model_kwargs={"pretrained_model": str(model_path)},
        eval_kwargs={"diameter": 1},
        blocksize=(1, 1, 2, 1),
        overlap=2,
        preprocessing_steps=[],
        nonempty_rule={
            "channel": "561",
            "channel_index": 1,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
        mask=None,
        runtime_artifacts=_runtime_artifacts(),
    )

    def assert_core_crops(**kwargs: object) -> list[int]:
        crops = kwargs["block_crops"]
        assert isinstance(crops, list)
        assert crops[0][2] == slice(0, 2)
        assert not segmentation._check_block_has_data(crops[0], input_zarr, 1, 1000)
        raise SelectionChecked

    monkeypatch.setattr(segmentation, "_select_input_blocks", assert_core_crops)

    with pytest.raises(SelectionChecked):
        segmentation.distributed_eval.__wrapped__(
            input_zarr=input_zarr,
            blocksize=(1, 1, 2, 1),
            write_path=tmp_path / "output.zarr",
            model_kwargs={"pretrained_model": str(model_path)},
            eval_kwargs={"diameter": 1},
            cluster=Cluster(),
            channel_indices=(0,),
            run_identity=run_identity,
        )


def test_nonempty_scan_fails_instead_of_scheduling_unknown_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingFuture:
        def add_done_callback(self, callback) -> None:
            callback(self)

        def result(self) -> bool:
            raise OSError("read failed")

    class FailingClient:
        def map(self, *args: object, **kwargs: object) -> list[FailingFuture]:
            return [FailingFuture()]

    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        shape=(1, 1, 1, 3),
        chunks=(1, 1, 1, 3),
        dtype=np.uint16,
    )
    monkeypatch.setattr(segmentation.distributed, "as_completed", lambda futures: futures)

    with pytest.raises(RuntimeError, match="Foreground scan failed for block 0"):
        segmentation._select_input_blocks(
            client=FailingClient(),
            block_crops=[(slice(0, 1),) * 3 + (slice(0, 3),)],
            input_zarr=input_zarr,
            nonempty_channel_index=1,
            nonempty_threshold=1000,
            blocksize=(1, 1, 1, 3),
            run_key="failed-run",
            path_nonempty=tmp_path / "nonempty.json",
        )


def test_nonempty_rule_changes_run_and_cache_identity(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.write_bytes(b"model")
    common = {
        "input_identity": {"path": "/data/input.zarr"},
        "channel_indices": (0,),
        "model_kwargs": {"pretrained_model": str(model_path)},
        "eval_kwargs": {"diameter": 30},
        "blocksize": (2, 4, 4, 1),
        "overlap": 60,
        "preprocessing_steps": [],
        "mask": None,
        "runtime_artifacts": _runtime_artifacts(),
    }

    threshold_1000 = segmentation._build_run_identity(
        **common,
        nonempty_rule={
            "channel": "561",
            "channel_index": 1,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
    )
    threshold_999 = segmentation._build_run_identity(
        **common,
        nonempty_rule={
            "channel": "561",
            "channel_index": 1,
            "threshold": 999,
            "min_fraction": 0.05,
        },
    )

    assert threshold_1000["schema_version"] == threshold_999["schema_version"] == 6
    assert segmentation._identity_digest(threshold_1000) != segmentation._identity_digest(threshold_999)
    assert segmentation._nonempty_cache_key(threshold_1000) != segmentation._nonempty_cache_key(threshold_999)


def test_nonempty_cache_ignores_inference_only_changes(tmp_path: Path) -> None:
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "model-b"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")
    common = {
        "input_identity": {"path": "/data/input.zarr"},
        "channel_indices": (0,),
        "blocksize": (2, 4, 4, 1),
        "overlap": 60,
        "nonempty_rule": {
            "channel": "561",
            "channel_index": 1,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
        "mask": None,
    }
    identity_a = segmentation._build_run_identity(
        **common,
        model_kwargs={"pretrained_model": str(model_a)},
        eval_kwargs={"diameter": 30, "flow_threshold": 0},
        preprocessing_steps=[],
        runtime_artifacts=_runtime_artifacts("a"),
    )
    identity_b = segmentation._build_run_identity(
        **common,
        model_kwargs={"pretrained_model": str(model_b)},
        eval_kwargs={"diameter": 30, "flow_threshold": 1},
        preprocessing_steps=[(np.asarray, {})],
        runtime_artifacts=_runtime_artifacts("b"),
    )

    assert segmentation._identity_digest(identity_a) != segmentation._identity_digest(identity_b)
    assert segmentation._nonempty_cache_key(identity_a) == segmentation._nonempty_cache_key(identity_b)


def test_stitch_options_change_run_but_not_nonempty_identity(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.write_bytes(b"model")
    common = {
        "input_identity": {"path": "/data/input.zarr"},
        "channel_indices": (0,),
        "model_kwargs": {"pretrained_model": str(model)},
        "eval_kwargs": {"diameter": 30},
        "blocksize": (2, 4, 4, 1),
        "overlap": 60,
        "preprocessing_steps": [],
        "nonempty_rule": {
            "channel": "561",
            "channel_index": 1,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
        "mask": None,
        "runtime_artifacts": _runtime_artifacts(),
    }
    face = segmentation._build_run_identity(**common, stitch_mode="face")
    overlap = segmentation._build_run_identity(
        **common, stitch_mode="overlap-iou", stitch_iou_threshold=0.4
    )

    assert segmentation._identity_digest(face) != segmentation._identity_digest(overlap)
    assert segmentation._nonempty_cache_key(face) == segmentation._nonempty_cache_key(overlap)


def test_mask_content_changes_run_and_nonempty_cache_identity(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.write_bytes(b"model")
    common = {
        "input_identity": {"path": "/data/input.zarr"},
        "channel_indices": (0,),
        "model_kwargs": {"pretrained_model": str(model_path)},
        "eval_kwargs": {"diameter": 30},
        "blocksize": (2, 4, 4, 1),
        "overlap": 60,
        "preprocessing_steps": [],
        "nonempty_rule": {
            "channel": "561",
            "channel_index": 1,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
        "runtime_artifacts": _runtime_artifacts(),
    }
    mask_a = np.zeros((2, 2, 2), dtype=np.uint8)
    mask_b = mask_a.copy()
    mask_b[0, 0, 0] = 1

    identity_a = segmentation._build_run_identity(**common, mask=mask_a)
    identity_b = segmentation._build_run_identity(**common, mask=mask_b)

    assert identity_a["mask"]["shape"] == [2, 2, 2]
    assert identity_a["mask"]["dtype"] == "uint8"
    assert identity_a["mask"]["sha256"] != identity_b["mask"]["sha256"]
    assert segmentation._identity_digest(identity_a) != segmentation._identity_digest(identity_b)
    assert segmentation._nonempty_cache_key(identity_a) != segmentation._nonempty_cache_key(identity_b)


def test_distributed_eval_rejects_conflicting_nonempty_rule(tmp_path: Path) -> None:
    input_zarr, _ = _write_input(tmp_path / "input.zarr")
    input_zarr.attrs["key"] = ["405", "561", "638"]
    model_path = tmp_path / "model"
    model_path.write_bytes(b"model")
    run_identity = segmentation._build_run_identity(
        input_identity=_input_identity(tmp_path / "input.zarr", input_zarr),
        channel_indices=(0,),
        model_kwargs={"pretrained_model": str(model_path)},
        eval_kwargs={"diameter": 1},
        blocksize=(2, 4, 4, 1),
        overlap=2,
        preprocessing_steps=[],
        nonempty_rule={
            "channel": "561",
            "channel_index": 1,
            "threshold": 999,
            "min_fraction": 0.05,
        },
        mask=None,
        runtime_artifacts=_runtime_artifacts(),
    )

    with pytest.raises(ValueError, match="nonempty_rule does not match"):
        segmentation.distributed_eval.__wrapped__(
            input_zarr=input_zarr,
            blocksize=(2, 4, 4, 1),
            write_path=tmp_path / "output.zarr",
            model_kwargs={},
            eval_kwargs={"diameter": 1},
            cluster=object(),
            channel_indices=(0,),
            run_identity=run_identity,
        )


def test_distributed_eval_rejects_identity_that_does_not_match_call(tmp_path: Path) -> None:
    input_zarr, _ = _write_input(tmp_path / "input.zarr")
    input_zarr.attrs["key"] = ["405", "561", "638"]
    identity_model = tmp_path / "identity-model"
    called_model = tmp_path / "called-model"
    identity_model.write_bytes(b"identity")
    called_model.write_bytes(b"called")
    run_identity = segmentation._build_run_identity(
        input_identity=_input_identity(tmp_path / "input.zarr", input_zarr),
        channel_indices=(0,),
        model_kwargs={"pretrained_model": str(identity_model)},
        eval_kwargs={"diameter": 1},
        blocksize=(2, 4, 4, 1),
        overlap=2,
        preprocessing_steps=[],
        nonempty_rule={
            "channel": "561",
            "channel_index": 1,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
        mask=None,
        runtime_artifacts=_runtime_artifacts(),
    )

    with pytest.raises(ValueError, match="run_identity does not match"):
        segmentation.distributed_eval.__wrapped__(
            input_zarr=input_zarr,
            blocksize=(2, 4, 4, 1),
            write_path=tmp_path / "output.zarr",
            model_kwargs={"pretrained_model": str(called_model)},
            eval_kwargs={"diameter": 1},
            cluster=object(),
            channel_indices=(0,),
            run_identity=run_identity,
        )


def test_distributed_eval_rejects_conflicting_mask_identity(tmp_path: Path) -> None:
    input_zarr, _ = _write_input(tmp_path / "input.zarr")

    with pytest.raises(ValueError, match="mask identity does not match"):
        segmentation.distributed_eval.__wrapped__(
            input_zarr=input_zarr,
            blocksize=(2, 4, 4, 1),
            write_path=tmp_path / "output.zarr",
            mask=np.ones((2, 2, 2), dtype=np.uint8),
            model_kwargs={},
            eval_kwargs={"diameter": 1},
            cluster=object(),
            channel_indices=(0,),
            run_identity={"mask": None},
        )


def test_typer_segment_run_rejects_assume_nonempty(tmp_path: Path) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()

    result = TyperCliRunner().invoke(
        app,
        ["segment", "run", str(input_path), "--assume-nonempty"],
    )

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_click_segment_run_rejects_assume_nonempty(tmp_path: Path) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()

    result = ClickCliRunner().invoke(
        segmentation.cli,
        ["run", str(input_path), "--assume-nonempty"],
    )

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_typer_segment_run_propagates_nonempty_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()
    captured: dict[str, object] = {}
    monkeypatch.setattr(segmentation, "_run_single_input", lambda **kwargs: captured.update(kwargs))

    result = TyperCliRunner().invoke(
        app,
        ["segment", "run", str(input_path), "--nonempty-threshold", "750"],
    )

    assert result.exit_code == 0, result.output
    assert captured["nonempty_threshold"] == 750


def test_typer_segment_run_propagates_overlap_iou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()
    captured: dict[str, object] = {}
    monkeypatch.setattr(segmentation, "_run_single_input", lambda **kwargs: captured.update(kwargs))

    result = TyperCliRunner().invoke(
        app,
        [
            "segment",
            "run",
            str(input_path),
            "--stitch-mode",
            "overlap-iou",
            "--stitch-iou-threshold",
            "0.4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["stitch_mode"] == "overlap-iou"
    assert captured["stitch_iou_threshold"] == 0.4


def test_click_segment_run_propagates_nonempty_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()
    captured: dict[str, object] = {}
    monkeypatch.setattr(segmentation, "_run_single_input", lambda **kwargs: captured.update(kwargs))

    result = ClickCliRunner().invoke(
        segmentation.cli,
        ["run", str(input_path), "--nonempty-threshold", "750"],
    )

    assert result.exit_code == 0, result.output
    assert captured["nonempty_threshold"] == 750


def test_typer_segment_run_propagates_per_device_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()
    captured: dict[str, object] = {}
    monkeypatch.setattr(segmentation, "_run_single_input", lambda **kwargs: captured.update(kwargs))

    result = TyperCliRunner().invoke(
        app,
        ["segment", "run", str(input_path), "--workers-per-device", "3,2"],
    )

    assert result.exit_code == 0, result.output
    assert captured["workers_per_device"] == (3, 2)


def test_worker_spec_supports_per_device_counts() -> None:
    spec = gpu_cluster._build_worker_spec(
        devices=["gpu-a", "gpu-b"],
        workers_per_gpu=4,
        workers_per_device=(3, 2),
        threads_per_worker=1,
    )

    assert set(spec) == {
        "gpu-0-w0",
        "gpu-0-w1",
        "gpu-0-w2",
        "gpu-1-w0",
        "gpu-1-w1",
    }
    assert spec["gpu-0-w2"]["options"]["env"] == {"CUDA_VISIBLE_DEVICES": "gpu-a"}
    assert spec["gpu-1-w1"]["options"]["env"] == {"CUDA_VISIBLE_DEVICES": "gpu-b"}


def test_gpu_cluster_routes_each_worker_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    routes: list[str] = []

    def start_speccluster(**kwargs: object) -> tuple[object, object, None]:
        routes.append("spec")
        return object(), object(), None

    def start_localcuda(**kwargs: object) -> tuple[object, object, None]:
        routes.append("localcuda")
        return object(), object(), None

    monkeypatch.setattr(gpu_cluster, "_start_speccluster", start_speccluster)
    monkeypatch.setattr(gpu_cluster, "_start_localcuda", start_localcuda)

    gpu_cluster.myGPUCluster(workers_per_device=(3, 2))
    gpu_cluster.myGPUCluster(workers_per_gpu=3)
    gpu_cluster.myGPUCluster(workers_per_gpu=1, use_localcuda=True)
    gpu_cluster.myGPUCluster(workers_per_gpu=1)

    assert routes == ["spec", "spec", "localcuda", "spec"]


def test_typer_segment_run_propagates_requested_box(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()
    captured: dict[str, object] = {}
    monkeypatch.setattr(segmentation, "_run_single_input", lambda **kwargs: captured.update(kwargs))

    result = TyperCliRunner().invoke(
        app,
        [
            "segment",
            "run",
            str(input_path),
            "--start-zyx",
            "1518,8538,3161",
            "--box-size-zyx",
            "256,512,512",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["start_zyx"] == (1518, 8538, 3161)
    assert captured["box_size_zyx"] == (256, 512, 512)


def test_click_segment_run_propagates_requested_box(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()
    captured: dict[str, object] = {}
    monkeypatch.setattr(segmentation, "_run_single_input", lambda **kwargs: captured.update(kwargs))

    result = ClickCliRunner().invoke(
        segmentation.cli,
        [
            "run",
            str(input_path),
            "--start-zyx",
            "1518,8538,3161",
            "--box-size-zyx",
            "256,512,512",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["start_zyx"] == (1518, 8538, 3161)
    assert captured["box_size_zyx"] == (256, 512, 512)


def test_typer_segment_stitch_propagates_transaction_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    output_path = tmp_path / "output.zarr"
    captured: dict[str, object] = {}

    def fake_run_stitch(
        received_temp: Path,
        received_output: Path,
        *,
        cleanup: bool,
        overwrite: bool,
    ) -> None:
        captured.update(
            temp=received_temp,
            output=received_output,
            cleanup=cleanup,
            overwrite=overwrite,
        )

    monkeypatch.setattr(segmentation, "_run_stitch", fake_run_stitch)

    result = TyperCliRunner().invoke(
        app,
        ["segment", "stitch", str(temp_dir), str(output_path), "--overwrite", "--no-cleanup"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "temp": temp_dir,
        "output": output_path,
        "cleanup": False,
        "overwrite": True,
    }


def test_nonempty_and_normalization_caches_require_matching_input(tmp_path: Path) -> None:
    nonempty_path = tmp_path / "nonempty.json"
    normalization_path = tmp_path / "normalization.json"
    blocksize = (2, 32, 32, 1)

    cache_utils.write_nonempty_cache(nonempty_path, blocksize, "run-a", [0, 2])
    settings = {"implementation": "bounded-z-gpu-v1", "z_samples": 32}
    cache_utils.write_normalization_cache(
        normalization_path,
        "input-a",
        {"1": [1.0, 9.0]},
        settings=settings,
    )

    assert cache_utils.read_nonempty_cache(nonempty_path, blocksize, "run-a") == [0, 2]
    assert cache_utils.read_nonempty_cache(nonempty_path, blocksize, "run-b") is None
    assert cache_utils.read_normalization_cache(normalization_path, "input-a") == {"1": [1.0, 9.0]}
    assert cache_utils.read_normalization_cache(normalization_path, "input-b") is None
    assert json.loads(normalization_path.read_text())["settings"] == settings


def test_stitching_empty_segmentation_writes_all_zero_output(tmp_path: Path) -> None:
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        shape=(2, 3, 4),
        chunks=(1, 3, 4),
        dtype=np.uint32,
        fill_value=0,
    )

    output, labeling = merge_utils.stitch_labels(
        block_indices=[],
        faces_list=[],
        box_ids_list=[],
        temp_zarr=temp,
        write_path=tmp_path / "output.zarr",
        lut_path=tmp_path / "labels.npy",
        pre_shrunk=True,
    )

    np.testing.assert_array_equal(labeling, np.array([0], dtype=np.uint32))
    np.testing.assert_array_equal(output[:], np.zeros(temp.shape, dtype=np.uint32))
    assert merge_utils.merge_boxes_for_labels([], [], labeling) == []


def test_trt_plan_content_is_part_of_run_identity(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.write_bytes(b"model")
    plan_path = segmentation.plan_path_for_device(model_path, "Test GPU")
    plan_path.write_bytes(b"plan-a")
    artifacts_a = _runtime_artifacts("a")
    artifacts_a["trt_plans"] = segmentation._trt_plan_identity({"xy": model_path}, {"Test GPU"})
    plan_path.write_bytes(b"plan-b")
    artifacts_b = _runtime_artifacts("a")
    artifacts_b["trt_plans"] = segmentation._trt_plan_identity({"xy": model_path}, {"Test GPU"})
    common = {
        "input_identity": {"path": "/data/input.zarr"},
        "channel_indices": (0,),
        "model_kwargs": {"pretrained_model": str(model_path)},
        "eval_kwargs": {"diameter": 30},
        "blocksize": (2, 4, 4, 1),
        "overlap": 60,
        "preprocessing_steps": [],
        "nonempty_rule": {
            "channel": "561",
            "channel_index": 1,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
        "mask": None,
    }

    identity_a = segmentation._build_run_identity(**common, runtime_artifacts=artifacts_a)
    identity_b = segmentation._build_run_identity(**common, runtime_artifacts=artifacts_b)

    assert artifacts_a != artifacts_b
    assert segmentation._identity_digest(identity_a) != segmentation._identity_digest(identity_b)


def test_ortho_checkpoint_and_plan_are_part_of_run_identity(tmp_path: Path) -> None:
    xy_model = tmp_path / "xy-model"
    ortho_model = tmp_path / "ortho-model"
    xy_model.write_bytes(b"xy")
    ortho_model.write_bytes(b"ortho-a")
    for model_path in (xy_model, ortho_model):
        segmentation.plan_path_for_device(model_path, "Test GPU").write_bytes(b"plan-a")
    model_kwargs = {
        "pretrained_model": str(xy_model),
        "pretrained_model_ortho": str(ortho_model),
    }
    common = {
        "input_identity": {"path": "/data/input.zarr"},
        "channel_indices": (0,),
        "model_kwargs": model_kwargs,
        "eval_kwargs": {"diameter": 30},
        "blocksize": (2, 4, 4, 1),
        "overlap": 60,
        "preprocessing_steps": [],
        "nonempty_rule": {
            "channel": "561",
            "channel_index": 1,
            "threshold": 1000,
            "min_fraction": 0.05,
        },
        "mask": None,
    }

    artifacts_a = _runtime_artifacts("a")
    artifacts_a["trt_plans"] = segmentation._trt_plan_identity(
        {"xy": xy_model, "ortho": ortho_model}, {"Test GPU"}
    )
    identity_a = segmentation._build_run_identity(**common, runtime_artifacts=artifacts_a)

    ortho_model.write_bytes(b"ortho-b")
    artifacts_b = _runtime_artifacts("a")
    artifacts_b["trt_plans"] = artifacts_a["trt_plans"]
    identity_b = segmentation._build_run_identity(**common, runtime_artifacts=artifacts_b)

    ortho_model.write_bytes(b"ortho-a")
    segmentation.plan_path_for_device(ortho_model, "Test GPU").write_bytes(b"plan-b")
    artifacts_c = _runtime_artifacts("a")
    artifacts_c["trt_plans"] = segmentation._trt_plan_identity(
        {"xy": xy_model, "ortho": ortho_model}, {"Test GPU"}
    )
    identity_c = segmentation._build_run_identity(**common, runtime_artifacts=artifacts_c)

    assert identity_a["model_sha256"].keys() == {"xy", "ortho"}
    assert {plan["model_role"] for plan in artifacts_a["trt_plans"]} == {"xy", "ortho"}
    assert segmentation._identity_digest(identity_a) != segmentation._identity_digest(identity_b)
    assert segmentation._identity_digest(identity_a) != segmentation._identity_digest(identity_c)


def test_runtime_artifacts_reject_duplicate_role_device_plans() -> None:
    artifacts = _runtime_artifacts()
    artifacts["trt_plans"].append(dict(artifacts["trt_plans"][0]))

    with pytest.raises(ValueError, match="duplicate model-role/device plans"):
        segmentation._validate_runtime_artifacts(artifacts)


def test_checkpoint_ignores_truncated_and_malformed_lines(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    checkpoint_path.write_text('{"index": [0, 1, 2]}\n{"worker": "missing-index"}\n{"index": [3, 4')

    assert segmentation.load_checkpoint(checkpoint_path) == {(0, 1, 2)}

    segmentation.append_checkpoint(checkpoint_path, (5, 6, 7), "gpu-0", 2.0, 3)

    assert segmentation.load_checkpoint(checkpoint_path) == {(0, 1, 2), (5, 6, 7)}


def test_driver_checkpoints_only_successful_futures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ImmediateFuture:
        def __init__(self, result: object = None, error: Exception | None = None) -> None:
            self._result = result
            self._error = error
            self.key = "test-future"

        def result(self) -> object:
            if self._error is not None:
                raise self._error
            return self._result

    success = ImmediateFuture({"index": (0, 0, 0, 0), "worker": "gpu-0", "duration_s": 1.25, "n_masks": 4})
    interrupted = ImmediateFuture(error=RuntimeError("worker stopped before completion"))
    monkeypatch.setattr(segmentation.distributed, "as_completed", lambda futures: futures)
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    failures = segmentation._wait_for_futures_collect_errors(
        futures=[success, interrupted],
        future_labels={success: "success", interrupted: "interrupted"},
        stage="Segmentation",
        log=segmentation.logger,
        checkpoint_path=checkpoint_path,
    )

    assert len(failures) == 1
    assert segmentation.load_checkpoint(checkpoint_path) == {(0, 0, 0, 0)}


def test_driver_waits_before_retrying_failed_block_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetriableFuture:
        key = "failed-block"
        attempt = 0

        def result(self) -> dict[str, object]:
            if self.attempt == 0:
                raise RuntimeError("worker stopped before completion")
            return {"index": (0, 0, 0, 0), "worker": "gpu-1", "duration_s": 2.0, "n_masks": 3}

    class Client:
        retried: list[object] = []

        def retry(self, futures: list[RetriableFuture]) -> None:
            self.retried.extend(futures)
            for future in futures:
                future.attempt += 1

    future = RetriableFuture()
    client = Client()
    sleeps: list[float] = []
    monkeypatch.setattr(segmentation.distributed, "as_completed", lambda futures: futures)
    monkeypatch.setattr(segmentation.time, "sleep", sleeps.append)
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    failures = segmentation._wait_for_futures_retry_once(
        client=client,
        futures=[future],
        future_labels={future: "block=(0, 0, 0, 0)"},
        stage="Segmentation",
        log=segmentation.logger,
        checkpoint_path=checkpoint_path,
    )

    assert sleeps == [30.0]
    assert client.retried == [future]
    assert failures == []
    assert segmentation.load_checkpoint(checkpoint_path) == {(0, 0, 0, 0)}


def test_driver_does_not_retry_checkpoint_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SuccessfulFuture:
        key = "successful-block"

        def result(self) -> dict[str, object]:
            return {"index": (0, 0, 0, 0), "worker": "gpu-0", "duration_s": 1.0, "n_masks": 2}

    class Client:
        def retry(self, futures: list[SuccessfulFuture]) -> None:
            raise AssertionError("a driver-side checkpoint failure must not retry the task")

    def fail_checkpoint(*args: object, **kwargs: object) -> None:
        raise OSError("checkpoint disk full")

    def fail_sleep(seconds: float) -> None:
        raise AssertionError("must not wait before a driver-side failure")

    monkeypatch.setattr(segmentation.distributed, "as_completed", lambda futures: futures)
    monkeypatch.setattr(segmentation, "append_checkpoint", fail_checkpoint)
    monkeypatch.setattr(segmentation.time, "sleep", fail_sleep)
    future = SuccessfulFuture()

    with pytest.raises(RuntimeError, match="completion handling failed.*checkpoint disk full"):
        segmentation._wait_for_futures_retry_once(
            client=Client(),
            futures=[future],
            future_labels={future: "block=(0, 0, 0, 0)"},
            stage="Segmentation",
            log=segmentation.logger,
            checkpoint_path=tmp_path / "checkpoint.jsonl",
        )


def test_stitch_failure_never_exposes_partial_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_zarr = zarr.create_array(
        tmp_path / "temp.zarr",
        data=np.zeros((2, 3, 4), dtype=np.uint32),
        chunks=(1, 3, 4),
    )
    output_path = tmp_path / "output.zarr"
    run_identity = {"schema_version": 3, "run": "test"}

    def interrupted_stitch(**kwargs: object) -> None:
        assert kwargs["chunk_coords"] == []
        write_path = Path(kwargs["write_path"])
        staged = zarr.create_array(
            write_path,
            shape=temp_zarr.shape,
            chunks=temp_zarr.chunks,
            dtype=np.uint32,
        )
        staged[0] = 1
        raise RuntimeError("stitch interrupted")

    original_stitch = segmentation.stitch_label_pairs
    monkeypatch.setattr(segmentation, "stitch_label_pairs", interrupted_stitch)
    with pytest.raises(RuntimeError, match="stitch interrupted"):
        segmentation._stitch_precomputed(
            label_pairs=[],
            boxes_list=[],
            box_ids_list=[],
            temp_zarr=temp_zarr,
            temp_dir=tmp_path,
            output_path=output_path,
            run_identity=run_identity,
            overwrite_output=False,
            non_empty_indices=[],
        )

    assert not output_path.exists()
    assert segmentation._staged_output_path(output_path).exists()

    monkeypatch.setattr(segmentation, "stitch_label_pairs", original_stitch)
    final, boxes = segmentation._stitch_precomputed(
        label_pairs=[],
        boxes_list=[],
        box_ids_list=[],
        temp_zarr=temp_zarr,
        temp_dir=tmp_path,
        output_path=output_path,
        run_identity=run_identity,
        overwrite_output=False,
        non_empty_indices=[],
    )

    np.testing.assert_array_equal(final[:], temp_zarr[:])
    assert boxes == []
    assert not segmentation._staged_output_path(output_path).exists()
    assert segmentation.promoted_output_matches(output_path, run_identity)


def test_completion_marker_rejects_changed_output_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "output.zarr"
    identity = {"schema_version": 3, "run": "test"}
    zarr.create_array(output_path, shape=(2, 3, 4), chunks=(1, 3, 4), dtype=np.uint32)
    segmentation.write_completion_marker(output_path, identity)
    zarr.create_array(
        output_path,
        shape=(1, 3, 4),
        chunks=(1, 3, 4),
        dtype=np.uint32,
        overwrite=True,
    )

    with pytest.raises(RuntimeError, match="does not match the schema"):
        segmentation.completed_run_matches(output_path, identity)


def test_process_block_returns_no_completion_when_output_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        shape=(2, 2, 2, 1),
        chunks=(2, 2, 2, 1),
        dtype=np.uint16,
    )

    class FailingOutput:
        def __setitem__(self, key: object, value: object) -> None:
            raise OSError("output write failed")

    monkeypatch.setattr(
        segmentation,
        "read_preprocess_and_segment",
        lambda *args, **kwargs: np.ones((2, 2, 2), dtype=np.uint32),
    )

    with pytest.raises(OSError, match="output write failed"):
        segmentation.process_block(
            block_index=(0, 0, 0, 0),
            crop=(slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 1)),
            input_zarr=input_zarr,
            model_kwargs={},
            eval_kwargs={},
            blocksize=(2, 2, 2, 1),
            overlap=0,
            output_zarr=FailingOutput(),
            channel_indices=(0,),
        )


def test_overwrite_promotion_failure_restores_prior_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_zarr = zarr.create_array(
        tmp_path / "temp.zarr",
        data=np.zeros((2, 3, 4), dtype=np.uint32),
        chunks=(1, 3, 4),
    )
    output_path = tmp_path / "output.zarr"
    prior = zarr.create_array(
        output_path,
        data=np.full((2, 3, 4), 7, dtype=np.uint32),
        chunks=(1, 3, 4),
    )
    prior.attrs["generation"] = "prior"
    run_identity = {"schema_version": 3, "run": "replacement"}
    staged_path = segmentation._staged_output_path(output_path)
    original_replace = segmentation.os.replace

    def fail_staged_promotion(source: object, destination: object) -> None:
        if Path(source) == staged_path and Path(destination) == output_path:
            raise OSError("promotion interrupted")
        original_replace(source, destination)

    monkeypatch.setattr(segmentation.os, "replace", fail_staged_promotion)

    with pytest.raises(OSError, match="promotion interrupted"):
        segmentation._stitch_precomputed(
            label_pairs=[],
            boxes_list=[],
            box_ids_list=[],
            temp_zarr=temp_zarr,
            temp_dir=tmp_path,
            output_path=output_path,
            run_identity=run_identity,
            overwrite_output=True,
            non_empty_indices=[],
        )

    restored = zarr.open_array(output_path, mode="r")
    np.testing.assert_array_equal(restored[:], np.full((2, 3, 4), 7, dtype=np.uint32))
    assert restored.attrs["generation"] == "prior"
    assert not segmentation._backup_output_path(output_path, run_identity).exists()


def test_stitch_recovers_promoted_output_before_stale_marker_check(tmp_path: Path) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "segmentation_unstitched.zarr").mkdir()
    (temp_dir / "intermediate_state.npz").touch()
    current_identity = _valid_run_identity("current")
    segmentation.save_run_config(temp_dir / "run_config.json", current_identity)

    output_path = tmp_path / "output.zarr"
    old_identity = {"schema_version": 3, "run": "old"}
    zarr.create_array(
        output_path,
        shape=(2, 3, 4),
        chunks=(1, 3, 4),
        dtype=np.uint32,
    )
    segmentation.write_completion_marker(output_path, old_identity)
    backup_path = segmentation._backup_output_path(output_path, current_identity)
    segmentation.os.replace(output_path, backup_path)
    promoted = zarr.create_array(
        output_path,
        shape=(2, 3, 4),
        chunks=(1, 3, 4),
        dtype=np.uint32,
    )
    promoted.attrs["squisher_run_key"] = segmentation._identity_digest(current_identity)
    promoted.attrs["squisher_output_schema"] = segmentation._zarr_schema(promoted)

    assert not segmentation.completed_run_matches(output_path, current_identity)
    segmentation._run_stitch(temp_dir, output_path, cleanup=False, overwrite=False)

    assert segmentation.completed_run_matches(output_path, current_identity)
    assert not backup_path.exists()


def test_matching_completed_stitch_cleans_interrupted_temp_state(tmp_path: Path) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    identity = _valid_run_identity("complete")
    segmentation.save_run_config(temp_dir / "run_config.json", identity)
    output_path = tmp_path / "output.zarr"
    zarr.create_array(output_path, shape=(2, 3, 4), chunks=(1, 3, 4), dtype=np.uint32)
    segmentation.write_completion_marker(output_path, identity)

    segmentation._run_stitch(temp_dir, output_path, cleanup=True, overwrite=False)

    assert not temp_dir.exists()


def test_blank_preconfig_temp_zarr_is_recoverable_but_written_store_is_not(
    tmp_path: Path,
) -> None:
    path = tmp_path / "temp.zarr"
    array = zarr.create_array(path, shape=(2, 3, 4), chunks=(1, 3, 4), dtype=np.uint32)

    recovered = segmentation._open_blank_temp_zarr(
        path,
        shape=(2, 3, 4),
        chunks=(1, 3, 4),
    )
    assert recovered.shape == (2, 3, 4)

    array[0] = 1
    with pytest.raises(RuntimeError, match="contains data"):
        segmentation._open_blank_temp_zarr(
            path,
            shape=(2, 3, 4),
            chunks=(1, 3, 4),
        )


def test_adopted_blank_temp_zarr_gets_run_config(tmp_path: Path) -> None:
    path = tmp_path / "temp.zarr"
    zarr.create_array(path, shape=(2, 3, 4), chunks=(1, 3, 4), dtype=np.uint32)
    config_path = tmp_path / "run_config.json"
    identity = {"schema_version": 4, "run": "adopted"}

    recovered = segmentation._initialize_temp_zarr(
        path=path,
        shape=(2, 3, 4),
        chunks=(1, 3, 4),
        run_config_path=config_path,
        run_identity=identity,
    )

    assert recovered.shape == (2, 3, 4)
    assert segmentation.load_run_identity(config_path) == identity


def test_run_stitch_replaces_output_transactionally_and_cleans_temp(tmp_path: Path) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    temp_zarr = zarr.create_array(
        temp_dir / "segmentation_unstitched.zarr",
        data=np.zeros((2, 3, 4), dtype=np.uint32),
        chunks=(1, 3, 4),
    )
    segmentation._save_intermediate_state(temp_dir, [], [], [], [])
    identity = _valid_run_identity("replacement")
    segmentation.save_run_config(temp_dir / "run_config.json", identity)
    output_path = tmp_path / "output.zarr"
    zarr.create_array(
        output_path,
        data=np.full(temp_zarr.shape, 9, dtype=np.uint32),
        chunks=temp_zarr.chunks,
    )

    segmentation._run_stitch(temp_dir, output_path, cleanup=True, overwrite=True)

    output = zarr.open_array(output_path, mode="r")
    np.testing.assert_array_equal(output[:], np.zeros(temp_zarr.shape, dtype=np.uint32))
    assert segmentation.completed_run_matches(output_path, identity)
    assert not segmentation._backup_output_path(output_path, identity).exists()
    assert not temp_dir.exists()


def test_run_stitch_rejects_stale_run_identity(tmp_path: Path) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    segmentation.save_run_config(
        temp_dir / "run_config.json",
        {"schema_version": 4, "run": "stale"},
    )

    with pytest.raises(ValueError, match="schema-6 artifact-bound identity"):
        segmentation._run_stitch(
            temp_dir,
            tmp_path / "output.zarr",
            cleanup=False,
            overwrite=False,
        )


def test_input_provenance_requires_and_hashes_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "input.zarr"
    input_path.mkdir()
    schema = {"shape": [2, 3, 4, 1], "chunks": [1, 3, 4, 1], "dtype": "uint16"}
    with pytest.raises(RuntimeError, match="no trusted completion marker"):
        segmentation._input_provenance_identity(input_path, expected_schema=schema)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"version": 1, "shape": [2, 3, 4, 1], "chunks": [1, 3, 4, 1], "dtype": "uint16"}'
    )
    first = segmentation._input_provenance_identity(input_path, expected_schema=schema)
    manifest_path.write_text(
        '{"version": 2, "shape": [2, 3, 4, 1], "chunks": [1, 3, 4, 1], "dtype": "uint16"}'
    )
    second = segmentation._input_provenance_identity(input_path, expected_schema=schema)

    assert first != second


def test_distributed_eval_requires_artifact_bound_identity(tmp_path: Path) -> None:
    input_zarr, _ = _write_input(tmp_path / "input.zarr")

    with pytest.raises(ValueError, match="run_identity is required"):
        segmentation.distributed_eval.__wrapped__(
            input_zarr=input_zarr,
            blocksize=(2, 4, 4, 1),
            write_path=tmp_path / "output.zarr",
            model_kwargs={},
            eval_kwargs={"diameter": 1},
            cluster=object(),
            channel_indices=(0,),
        )

    with pytest.raises(ValueError, match="complete schema-6"):
        segmentation.distributed_eval.__wrapped__(
            input_zarr=input_zarr,
            blocksize=(2, 4, 4, 1),
            write_path=tmp_path / "output.zarr",
            model_kwargs={},
            eval_kwargs={"diameter": 1},
            cluster=object(),
            channel_indices=(0,),
            run_identity={
                "schema_version": 3,
                "mask": None,
                "runtime_artifacts": {"trt_plans": []},
            },
        )


def test_relabel_write_preserves_large_zarr_chunks(tmp_path: Path) -> None:
    data = np.arange(12 * 64 * 64, dtype=np.uint32).reshape(12, 64, 64)
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        data=data,
        chunks=(6, 64, 64),
    )
    lut_path = tmp_path / "labels.npy"
    np.save(lut_path, np.arange(data.size, dtype=np.uint32))

    with dask.config.set({"array.chunk-size": "1KiB"}):
        merge_utils.relabel_and_write(
            temp,
            lut_path,
            tmp_path / "output.zarr",
        )

    output = zarr.open_array(tmp_path / "output.zarr", mode="r")
    np.testing.assert_array_equal(output[:], data)


def test_relabel_write_owns_each_ragged_zarr_chunk(tmp_path: Path) -> None:
    data = np.arange(13 * 100 * 100, dtype=np.uint32).reshape(13, 100, 100)
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        data=data,
        chunks=(6, 70, 50),
    )
    lut_path = tmp_path / "labels.npy"
    np.save(lut_path, np.arange(data.size, dtype=np.uint32))

    with (
        dask.config.set({"array.chunk-size": "1KiB"}),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", PerformanceWarning)
        merge_utils.relabel_and_write(
            temp,
            lut_path,
            tmp_path / "output.zarr",
        )

    output = zarr.open_array(tmp_path / "output.zarr", mode="r")
    np.testing.assert_array_equal(output[:], data)


def test_writer_rechunks_to_ragged_destination_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.zeros((13, 100, 100), dtype=np.uint32)
    source = dask.array.from_array(data, chunks=(5, 40, 40))
    output = zarr.create_array(
        tmp_path / "output.zarr",
        shape=data.shape,
        chunks=(6, 70, 50),
        dtype=data.dtype,
    )
    captured: dict[str, object] = {}

    def capture_store(
        array: dask.array.Array,
        target: zarr.Array,
        *,
        lock: bool,
        scheduler: str,
        num_workers: int,
    ) -> None:
        captured.update(
            chunks=array.chunks,
            target=target,
            lock=lock,
            scheduler=scheduler,
            num_workers=num_workers,
        )

    monkeypatch.setattr(dask.array.Array, "store", capture_store)

    merge_utils.write_dask_to_zarr(source, output)

    assert captured == {
        "chunks": ((6, 6, 1), (70, 30), (50, 50)),
        "target": output,
        "lock": False,
        "scheduler": "threads",
        "num_workers": merge_utils.FINALIZE_WORKERS,
    }


def test_sparse_global_labels_decode_to_bounded_block_labels() -> None:
    local = np.array([[[0, 1, 3]]], dtype=np.uint32)
    global_labels, _ = merge_utils.global_segment_ids(
        local,
        block_index=(0, 0, 1000),
        nblocks=np.array((1, 1, 1001)),
    )

    decoded, global_ids = merge_utils.decode_block_global_labels(global_labels)

    np.testing.assert_array_equal(decoded, local)
    np.testing.assert_array_equal(global_ids, global_labels[global_labels != 0])
    assert decoded.max() == 3
    assert global_labels.max() > 65_000_000


def test_global_label_bits_use_available_uint32_capacity() -> None:
    nblocks = np.array((9, 15, 12))
    label_bits = merge_utils.global_label_bits(nblocks)

    remap = merge_utils.global_segment_id_remap(
        75_414,
        block_index=(3, 5, 6),
        nblocks=nblocks,
        label_bits=label_bits,
    )
    decoded, global_ids = merge_utils.decode_block_global_labels(
        remap[[0, 1, 75_414]],
        label_bits=label_bits,
    )

    assert label_bits == 21
    np.testing.assert_array_equal(decoded, np.array([0, 1, 75_414], dtype=np.uint32))
    np.testing.assert_array_equal(global_ids, remap[[1, 75_414]])


def test_block_label_decode_rejects_mixed_block_tokens() -> None:
    mixed = np.array([1, (2 << merge_utils.GLOBAL_LABEL_BITS) | 1], dtype=np.uint32)

    with pytest.raises(ValueError, match="multiple block tokens"):
        merge_utils.decode_block_global_labels(mixed)
