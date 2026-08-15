from pathlib import Path

import numpy as np
import pytest
import zarr

from squisher_segment.segmentation.distributed import cache_utils
from squisher_segment.segmentation.distributed import distributed_segmentation as segmentation
from squisher_segment.segmentation.distributed import merge_utils


def _write_input(path: Path) -> tuple[zarr.Array, np.ndarray]:
    data = np.arange(2 * 4 * 5 * 3, dtype=np.uint16).reshape(2, 4, 5, 3)
    array = zarr.create_array(path, data=data, chunks=(1, 2, 3, 1))
    array.attrs["key"] = ["dna", "membrane", "far-red"]
    return array, data


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


def test_channel_selection_rejects_ambiguous_metadata(tmp_path: Path) -> None:
    array, _ = _write_input(tmp_path / "input.zarr")
    array.attrs["key"] = ["dna", "dna", "far-red"]

    with pytest.raises(ValueError, match="must be unique"):
        segmentation._resolve_channel_selection(array, "dna")


def test_run_state_is_bound_to_exact_identity(tmp_path: Path) -> None:
    config_path = tmp_path / "run_config.json"
    output_path = tmp_path / "output_segmentation-sam.zarr"
    output_path.mkdir()
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


def test_nonempty_and_normalization_caches_require_matching_input(tmp_path: Path) -> None:
    nonempty_path = tmp_path / "nonempty.json"
    normalization_path = tmp_path / "normalization.json"
    blocksize = (2, 32, 32, 1)

    cache_utils.write_nonempty_cache(nonempty_path, blocksize, "run-a", [0, 2])
    cache_utils.write_normalization_cache(normalization_path, "input-a", {"1": [1.0, 9.0]})

    assert cache_utils.read_nonempty_cache(nonempty_path, blocksize, "run-a") == [0, 2]
    assert cache_utils.read_nonempty_cache(nonempty_path, blocksize, "run-b") is None
    assert cache_utils.read_normalization_cache(normalization_path, "input-a") == {
        "1": [1.0, 9.0]
    }
    assert cache_utils.read_normalization_cache(normalization_path, "input-b") is None


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


def test_block_label_decode_rejects_mixed_block_tokens() -> None:
    mixed = np.array([1, (2 << merge_utils.GLOBAL_LABEL_BITS) | 1], dtype=np.uint32)

    with pytest.raises(ValueError, match="multiple block tokens"):
        merge_utils.decode_block_global_labels(mixed)
