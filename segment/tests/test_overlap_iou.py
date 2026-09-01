from pathlib import Path

import numpy as np
import pytest
import zarr

from squisher_segment.segmentation.distributed import merge_utils
from squisher_segment.segmentation.distributed import distributed_segmentation as segmentation
from squisher_segment.segmentation.distributed import overlap_stitch


def test_stitch_label_pairs_merges_ids_and_boxes(tmp_path) -> None:
    import zarr

    first = np.uint32(1)
    second = np.uint32((1 << merge_utils.GLOBAL_LABEL_BITS) | 1)
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        data=np.array([[[first, second]]], dtype=np.uint32),
        chunks=(1, 1, 1),
    )
    boxes = [
        [(slice(0, 1), slice(0, 1), slice(0, 1))],
        [(slice(0, 1), slice(0, 1), slice(1, 2))],
    ]
    box_ids = [np.array([first]), np.array([second])]

    output, mapping = merge_utils.stitch_label_pairs(
        label_pairs=[np.array([[first], [second]], dtype=np.uint32)],
        box_ids_list=box_ids,
        temp_zarr=temp,
        write_path=tmp_path / "output.zarr",
        mapping_path=tmp_path / "mapping.npy",
    )

    np.testing.assert_array_equal(output[:], np.ones((1, 1, 2), dtype=np.uint32))
    np.testing.assert_array_equal(mapping[0], np.array([first, second], dtype=np.uint32))
    assert merge_utils.merge_boxes_for_sparse_labels(boxes, box_ids, mapping) == [
        (slice(0, 1), slice(0, 1), slice(0, 2))
    ]


def test_stitch_label_pairs_supports_dynamic_label_bits(tmp_path) -> None:
    import zarr

    label_bits = 21
    encoded = np.uint32((7 << label_bits) | 75_414)
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        data=np.array([[[0, encoded]]], dtype=np.uint32),
        chunks=(1, 1, 2),
    )

    output, mapping = merge_utils.stitch_label_pairs(
        label_pairs=[],
        box_ids_list=[np.array([encoded], dtype=np.uint32)],
        temp_zarr=temp,
        write_path=tmp_path / "output.zarr",
        mapping_path=tmp_path / "mapping.npy",
        label_bits=label_bits,
    )

    np.testing.assert_array_equal(output[:], np.array([[[0, 1]]], dtype=np.uint32))
    np.testing.assert_array_equal(mapping[0], np.array([encoded], dtype=np.uint32))


def test_temp_block_metadata_preserves_global_ids(tmp_path, monkeypatch) -> None:
    import zarr

    label_bits = merge_utils.global_label_bits(np.array((5, 9, 5)))
    token = np.uint32(7 << label_bits)
    data = np.array([[[0, token | 1, token | 75_414]]], dtype=np.uint32)
    crop = (slice(4, 5), slice(8, 9), slice(12, 15))
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        shape=(5, 9, 15),
        chunks=(1, 1, 3),
        dtype=np.uint32,
    )
    temp[crop] = data

    def boxes(local: np.ndarray, received_crop: tuple[slice, ...]):
        np.testing.assert_array_equal(local, np.array([[[0, 1, 75_414]]], dtype=np.uint32))
        assert received_crop == crop
        return [(slice(4, 5), slice(8, 9), slice(13, 14))]

    monkeypatch.setattr(segmentation, "bounding_boxes_in_global_coordinates", boxes)

    block_boxes, ids = segmentation._block_metadata_from_temp(crop, temp_zarr=temp)

    assert block_boxes == [(slice(4, 5), slice(8, 9), slice(13, 14))]
    np.testing.assert_array_equal(ids, np.array([token | 1, token | 75_414], dtype=np.uint32))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_overlap_sidecars_stream_neighbor_pairs(tmp_path, axis: int) -> None:
    first_index = (0, 0, 0, 0)
    second_index = list(first_index)
    second_index[axis] = 1
    second_index = tuple(second_index)
    first_faces = []
    second_faces = []
    for face_axis in range(3):
        shape = [2, 2, 2]
        shape[face_axis] = 1
        first_faces.extend(np.zeros(shape, dtype=np.uint32) for _ in range(2))
        second_faces.extend(np.zeros(shape, dtype=np.uint32) for _ in range(2))
    first_faces[2 * axis + 1][:] = 5
    second_faces[2 * axis][:] = 9

    segmentation._save_overlap_faces(tmp_path, first_index, first_faces)
    segmentation._save_overlap_faces(tmp_path, second_index, second_faces)
    pairs = segmentation._overlap_pairs_from_sidecars(
        [first_index, second_index],
        tmp_path,
    )

    assert len(pairs) == 1
    np.testing.assert_array_equal(pairs[0], np.array([[5], [9]], dtype=np.uint32))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_overlap_sidecars_use_trimmed_face_contact(tmp_path, axis: int) -> None:
    first_index = (0, 0, 0, 0)
    second_index_list = [0, 0, 0, 0]
    second_index_list[axis] = 1
    second_index = tuple(second_index_list)
    first_faces = []
    second_faces = []
    for face_axis in range(3):
        shape = [2, 3, 4]
        shape[face_axis] = 1
        first_faces.extend(np.zeros(shape, dtype=np.uint32) for _ in range(2))
        second_faces.extend(np.zeros(shape, dtype=np.uint32) for _ in range(2))
    first_faces[2 * axis + 1].ravel()[:3] = 5
    second_faces[2 * axis].ravel()[[0, 3, 4]] = 9

    segmentation._save_overlap_faces(tmp_path, first_index, first_faces)
    segmentation._save_overlap_faces(tmp_path, second_index, second_faces)
    pairs = segmentation._overlap_pairs_from_sidecars(
        [first_index, second_index],
        tmp_path,
    )

    assert len(pairs) == 1
    np.testing.assert_array_equal(pairs[0], np.array([[5], [9]], dtype=np.uint32))


def test_process_block_persists_overlap_before_completion(tmp_path, monkeypatch) -> None:
    import zarr

    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        shape=(4, 4, 4, 1),
        chunks=(4, 4, 2, 1),
        dtype=np.uint16,
    )
    output_zarr = zarr.create_array(
        tmp_path / "output.zarr",
        shape=(4, 4, 4),
        chunks=(4, 4, 2),
        dtype=np.uint32,
    )
    monkeypatch.setattr(
        segmentation,
        "read_preprocess_and_segment",
        lambda *args, **kwargs: np.ones((4, 4, 3), dtype=np.uint32),
    )

    result = segmentation.process_block(
        block_index=(0, 0, 0, 0),
        crop=(slice(0, 4), slice(0, 4), slice(0, 3), slice(0, 1)),
        input_zarr=input_zarr,
        model_kwargs={},
        eval_kwargs={},
        blocksize=(4, 4, 2, 1),
        overlap=1,
        output_zarr=output_zarr,
        channel_indices=(0,),
        overlap_directory=str(tmp_path / "overlaps"),
    )

    assert result["index"] == (0, 0, 0, 0)
    assert result["n_masks"] == 1
    assert segmentation._overlap_sidecar_path(
        tmp_path / "overlaps", (0, 0, 0, 0)
    ).is_file()
    np.testing.assert_array_equal(output_zarr[:, :, :2], 1)


def test_intermediate_state_roundtrips_variable_pair_counts(tmp_path) -> None:
    pairs = [
        np.array([[1], [2]], dtype=np.uint32),
        np.array([[3, 4], [5, 6]], dtype=np.uint32),
    ]

    segmentation._save_intermediate_state(tmp_path, pairs, [], [], [])
    restored, _, _, _ = segmentation._load_intermediate_state(tmp_path)

    assert len(restored) == 2
    np.testing.assert_array_equal(restored[0], pairs[0])
    np.testing.assert_array_equal(restored[1], pairs[1])


def test_halo_only_label_can_bridge_owned_core_labels() -> None:
    mapping = merge_utils.determine_sparse_merge_relabeling(
        np.array([1, 3], dtype=np.uint32),
        [np.array([[1, 2], [2, 3]], dtype=np.uint32)],
    )

    np.testing.assert_array_equal(
        mapping,
        np.array([[1, 3], [1, 1]], dtype=np.uint32),
    )


def test_overlap_iou_requires_unique_reciprocal_best() -> None:
    first = np.array([[1, 1, 2, 2, 0]], dtype=np.uint32)
    second = np.array([[4, 5, 6, 6, 0]], dtype=np.uint32)

    pairs = overlap_stitch.match_overlap_iou(first, second, threshold=0.25)

    np.testing.assert_array_equal(pairs, np.array([[2], [6]], dtype=np.uint32))


def test_overlap_iou_keeps_exact_ties_separate() -> None:
    first = np.array([[1, 1]], dtype=np.uint32)
    second = np.array([[2, 3]], dtype=np.uint32)

    pairs = overlap_stitch.match_overlap_iou(first, second, threshold=0.25)

    assert pairs.shape == (2, 0)


def test_overlap_iou_threshold_is_inclusive() -> None:
    first = np.array([[1, 1]], dtype=np.uint32)
    second = np.array([[2, 0]], dtype=np.uint32)

    accepted = overlap_stitch.match_overlap_iou(first, second, threshold=0.5)
    rejected = overlap_stitch.match_overlap_iou(first, second, threshold=0.5001)

    np.testing.assert_array_equal(accepted, np.array([[1], [2]], dtype=np.uint32))
    assert rejected.shape == (2, 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_overlap_evidence_geometry_uses_shared_band_and_core(axis: int) -> None:
    shape = (7, 9, 11)
    blocksize = (4, 5, 6)
    indices, crops = segmentation._segmentation_block_crops(
        shape + (1,), blocksize + (1,), overlap=1, mask=None
    )
    selected = {tuple(index): crop[:-1] for index, crop in zip(indices, crops, strict=True)}
    first = (0, 0, 0)
    second_list = list(first)
    second_list[axis] = 1
    second = tuple(second_list)

    rows, by_block = overlap_stitch.plan_overlap_evidence(
        selected, shape=shape, blocksize=blocksize
    )

    row = next(row for row in rows[axis] if row["blocks"] == [first, second])
    global_slices = tuple(slice(*bounds) for bounds in row["slices"])
    assert global_slices[axis] == slice(blocksize[axis] - 1, blocksize[axis] + 1)
    for transverse_axis in set(range(3)) - {axis}:
        assert global_slices[transverse_axis] == slice(0, blocksize[transverse_axis])
    assert len(by_block[first]) >= 1
    assert len(by_block[second]) >= 1


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_overlap_evidence_write_preserves_global_geometry(tmp_path: Path, axis: int) -> None:
    shape = (7, 9, 11)
    blocksize = (4, 5, 6)
    indices, crops = segmentation._segmentation_block_crops(
        shape + (1,), blocksize + (1,), overlap=1, mask=None
    )
    all_crops = {tuple(index[:3]): crop[:-1] for index, crop in zip(indices, crops, strict=True)}
    first = (0, 0, 0)
    second_list = list(first)
    second_list[axis] = 1
    second = tuple(second_list)
    selected = {first: all_crops[first], second: all_crops[second]}
    rows, by_block = overlap_stitch.plan_overlap_evidence(
        selected, shape=shape, blocksize=blocksize
    )
    evidence_dir = tmp_path / f"axis-{axis}"
    overlap_stitch.initialize_overlap_evidence(
        evidence_dir, rows, run_key="geometry", resume=False
    )
    global_labels = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) + 1
    for block in (first, second):
        crop = selected[block]
        overlap_stitch.write_block_evidence(
            evidence_dir,
            block,
            by_block[block],
            global_labels[crop],
            crop,
            run_key="geometry",
        )

    row = rows[axis][0]
    bounds = tuple(slice(*values) for values in row["slices"])
    expected = np.moveaxis(global_labels[bounds], axis, 0)
    stored = zarr.open_array(evidence_dir / f"axis-{axis}.zarr", mode="r")
    valid = tuple(slice(0, size) for size in row["valid_shape"])
    np.testing.assert_array_equal(stored[(0, 0, *valid)], expected)
    np.testing.assert_array_equal(stored[(0, 1, *valid)], expected)


def test_overlap_evidence_marker_is_bound_to_run(tmp_path: Path) -> None:
    overlap_stitch.write_block_marker(tmp_path, (1, 2, 3), "run-a")

    assert overlap_stitch.block_marker_matches(tmp_path, (1, 2, 3), "run-a")
    assert not overlap_stitch.block_marker_matches(tmp_path, (1, 2, 3), "run-b")


def test_overlap_evidence_resume_rejects_identity_change(tmp_path: Path) -> None:
    rows: list[list[dict[str, object]]] = [[], [], []]
    evidence_dir = tmp_path / "evidence"
    overlap_stitch.initialize_overlap_evidence(evidence_dir, rows, run_key="run-a", resume=False)

    with pytest.raises(RuntimeError, match="run identity changed"):
        overlap_stitch.initialize_overlap_evidence(evidence_dir, rows, run_key="run-b", resume=True)


def test_overlap_evidence_resume_rejects_array_identity_change(tmp_path: Path) -> None:
    rows = [
        [
            {
                "blocks": [(0, 0, 0), (1, 0, 0)],
                "slices": [[1, 3], [0, 2], [0, 2]],
                "valid_shape": [2, 2, 2],
            }
        ],
        [],
        [],
    ]
    evidence_dir = tmp_path / "evidence"
    overlap_stitch.initialize_overlap_evidence(evidence_dir, rows, run_key="run-a", resume=False)
    array = zarr.open_array(evidence_dir / "axis-0.zarr", mode="r+")
    array.attrs["run_key"] = "wrong"

    with pytest.raises(RuntimeError, match="axis 0 identity changed"):
        overlap_stitch.initialize_overlap_evidence(
            evidence_dir, rows, run_key="run-a", resume=True
        )


def test_overlap_evidence_roundtrip_encodes_block_ids(tmp_path: Path) -> None:
    shape = (4, 2, 2)
    blocksize = (2, 2, 2)
    indices, crops = segmentation._segmentation_block_crops(
        shape + (1,), blocksize + (1,), overlap=1, mask=None
    )
    selected = {tuple(index[:3]): crop[:-1] for index, crop in zip(indices, crops, strict=True)}
    rows, by_block = overlap_stitch.plan_overlap_evidence(
        selected, shape=shape, blocksize=blocksize
    )
    evidence_dir = tmp_path / "evidence"
    overlap_stitch.initialize_overlap_evidence(
        evidence_dir, rows, run_key="test-run", resume=False
    )

    for index, crop in selected.items():
        labels = np.full(
            tuple(axis.stop - axis.start for axis in crop),
            3 if index[0] == 0 else 7,
            dtype=np.uint32,
        )
        overlap_stitch.write_block_evidence(
            evidence_dir,
            index,
            by_block[index],
            labels,
            crop,
            run_key="test-run",
        )

    label_bits = merge_utils.global_label_bits(np.array((2, 1, 1)))
    pairs = overlap_stitch.match_evidence(
        evidence_dir,
        threshold=0.25,
        nblocks=(2, 1, 1),
        label_bits=label_bits,
    )

    assert len(pairs) == 1
    np.testing.assert_array_equal(
        pairs[0],
        np.array([[3], [(1 << label_bits) | 7]], dtype=np.uint32),
    )


def test_core_empty_sidecar_can_supply_transient_links(tmp_path) -> None:
    indices = [(0, 0, i, 0) for i in range(3)]
    faces = []
    for _ in indices:
        block = []
        for axis in range(3):
            shape = [2, 2, 2]
            shape[axis] = 1
            block.extend(np.zeros(shape, dtype=np.uint32) for _ in range(2))
        faces.append(block)
    faces[0][5][:] = 1
    faces[1][4][:] = 2
    faces[1][5][:] = 2
    faces[2][4][:] = 3
    for index, block_faces in zip(indices, faces, strict=True):
        segmentation._save_overlap_faces(tmp_path, index, block_faces)

    pairs = segmentation._overlap_pairs_from_sidecars(
        indices,
        tmp_path,
    )
    mapping = merge_utils.determine_sparse_merge_relabeling(
        np.array([1, 3], dtype=np.uint32),
        pairs,
    )

    np.testing.assert_array_equal(mapping, np.array([[1, 3], [1, 1]], dtype=np.uint32))


def test_resume_retains_every_checkpointed_planned_block() -> None:
    planned = [
        (0, 2, 4, 0),
        (1, 2, 4, 0),
        (0, 2, 5, 0),
        (1, 2, 5, 0),
    ]
    checkpointed = {
        (0, 2, 4, 0),
        (1, 2, 4, 0),
        (0, 2, 5, 0),
        (9, 9, 9, 0),
    }

    assert segmentation._completed_blocks(planned, checkpointed) == {
        (0, 2, 4, 0),
        (1, 2, 4, 0),
        (0, 2, 5, 0),
    }


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_process_blocks_stitch_overlap_3d(tmp_path, monkeypatch, axis: int) -> None:
    import zarr

    spatial_shape = (8, 8, 8)
    spatial_blocksize = [8, 8, 8]
    spatial_blocksize[axis] = 4
    blocksize = tuple(spatial_blocksize) + (1,)
    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        shape=spatial_shape + (1,),
        chunks=blocksize,
        dtype=np.uint16,
    )
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        shape=spatial_shape,
        chunks=tuple(spatial_blocksize),
        dtype=np.uint32,
    )
    block_indices, crops = segmentation._segmentation_block_crops(
        input_zarr.shape,
        blocksize,
        overlap=1,
        mask=None,
    )
    object_box = tuple(slice(1, 7) for _ in range(3))

    def fake_segment(_input, crop, *_args, **_kwargs):
        spatial_crop = crop[:-1]
        labels = np.zeros(tuple(s.stop - s.start for s in spatial_crop), dtype=np.uint32)
        local_object = tuple(
            slice(
                max(0, obj.start - outer.start),
                min(outer.stop, obj.stop) - outer.start,
            )
            for obj, outer in zip(object_box, spatial_crop)
        )
        labels[local_object] = 1
        return labels

    monkeypatch.setattr(segmentation, "read_preprocess_and_segment", fake_segment)
    sidecars = tmp_path / "overlaps"
    for block_index, crop in zip(block_indices, crops):
        segmentation.process_block(
            block_index=block_index,
            crop=crop,
            input_zarr=input_zarr,
            model_kwargs={},
            eval_kwargs={},
            blocksize=blocksize,
            overlap=1,
            output_zarr=temp,
            channel_indices=(0,),
            overlap_directory=str(sidecars),
        )

    pairs = segmentation._overlap_pairs_from_sidecars(block_indices, sidecars)
    box_ids = []
    for block_index in block_indices:
        core = tuple(
            slice(i * size, min((i + 1) * size, extent))
            for i, size, extent in zip(block_index[:3], spatial_blocksize, spatial_shape)
        )
        labels = np.unique(temp[core])
        box_ids.append(labels[labels != 0])

    output, _ = merge_utils.stitch_label_pairs(
        label_pairs=pairs,
        box_ids_list=box_ids,
        temp_zarr=temp,
        write_path=tmp_path / "stitched.zarr",
        mapping_path=tmp_path / "mapping.npy",
    )
    expected = np.zeros(spatial_shape, dtype=np.uint32)
    expected[object_box] = 1
    np.testing.assert_array_equal(output[:], expected)
