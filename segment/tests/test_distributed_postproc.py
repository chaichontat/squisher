from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from click.testing import CliRunner as ClickCliRunner
from typer.testing import CliRunner

from squisher_segment.cli import app
from squisher_segment.segmentation.distributed import distributed_postproc as postproc
from squisher_segment.segmentation.distributed import merge_utils


def test_postproc_blocksize_defaults_to_input_chunks(tmp_path: Path) -> None:
    input_zarr = zarr.create_array(
        tmp_path / "segmentation.zarr",
        shape=(7, 9, 11),
        chunks=(3, 4, 5),
        dtype=np.uint32,
    )

    assert postproc._resolve_blocksize(input_zarr, None) == (3, 4, 5)
    assert postproc._resolve_blocksize(input_zarr, (4, 20, 2)) == (4, 9, 2)


@pytest.mark.parametrize("blocksize", [(0, 4, 5), (3, -1, 5)])
def test_postproc_blocksize_rejects_nonpositive_extents(
    tmp_path: Path,
    blocksize: tuple[int, int, int],
) -> None:
    input_zarr = zarr.create_array(
        tmp_path / "segmentation.zarr",
        shape=(7, 9, 11),
        chunks=(3, 4, 5),
        dtype=np.uint32,
    )

    with pytest.raises(ValueError, match="positive"):
        postproc._resolve_blocksize(input_zarr, blocksize)


def test_postproc_tiling_rejects_halo_at_least_as_large_as_core() -> None:
    with pytest.raises(ValueError, match="smaller than tiled core extents"):
        postproc._validate_tiling((7, 9, 11), (3, 4, 5), overlap=60)

    np.testing.assert_array_equal(
        postproc._validate_tiling((7, 9, 11), (3, 4, 5), overlap=1),
        [3, 3, 3],
    )
    np.testing.assert_array_equal(
        postproc._validate_tiling((7, 9, 11), (7, 9, 11), overlap=60),
        [1, 1, 1],
    )


def test_postproc_tiling_uses_available_uint32_capacity() -> None:
    nblocks = postproc._validate_tiling(
        (2470, 10657, 7871),
        (280, 712, 712),
        overlap=60,
    )

    np.testing.assert_array_equal(nblocks, [9, 15, 12])
    assert merge_utils.global_label_bits(nblocks) == 21


def test_postproc_tiling_rejects_unrepresentable_block_grid() -> None:
    with pytest.raises(ValueError, match="cannot fit in uint32"):
        postproc._validate_tiling((1, 1, 1 << 32), (1, 1, 1), overlap=0)


def test_postproc_gpu_smoothing_import_error_is_not_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        data=np.ones((1, 2, 2), dtype=np.uint32),
        chunks=(1, 2, 2),
    )
    output_zarr = zarr.create_array(
        tmp_path / "output.zarr",
        shape=input_zarr.shape,
        chunks=input_zarr.chunks,
        dtype=np.uint32,
    )
    monkeypatch.setattr(postproc.cp, "asarray", np.asarray)
    monkeypatch.setattr(postproc.cp, "asnumpy", np.asarray)
    monkeypatch.setattr(postproc.cp, "unique", np.unique)

    def fail_gpu_smoothing(*_args: object, **_kwargs: object) -> None:
        raise ImportError("gpu unavailable")

    monkeypatch.setattr(
        postproc,
        "gaussian_smooth_labels_cupy",
        fail_gpu_smoothing,
    )
    with pytest.raises(ImportError, match="gpu unavailable"):
        postproc.process_postproc_block(
            block_index=(0, 0, 0),
            crop=(slice(None), slice(None), slice(None)),
            input_zarr=input_zarr,
            output_zarr=output_zarr,
            blocksize=input_zarr.shape,
            overlap=0,
            nblocks=np.ones(3, dtype=int),
            postproc_kwargs={"sigma": 1.0},
        )


def test_zyx_overlap_trimming_covers_each_voxel_once() -> None:
    shape = (7, 9, 11)
    blocksize = (3, 4, 5)
    overlap = 1
    source = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    output = np.zeros_like(source)
    coverage = np.zeros(shape, dtype=np.uint8)

    _, crops = merge_utils.get_block_crops(
        shape,
        np.asarray(blocksize),
        overlap,
        mask=None,
    )
    for crop in crops:
        core, destination = merge_utils.remove_overlaps(
            source[crop],
            crop,
            overlap,
            blocksize,
        )
        destination_tuple = tuple(destination)
        output[destination_tuple] = core
        coverage[destination_tuple] += 1

    np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(coverage, np.ones(shape, dtype=np.uint8))


def test_z_face_labels_are_merged() -> None:
    nblocks = np.asarray((2, 1, 1))
    first = np.zeros((2, 8, 8), dtype=np.uint32)
    second = np.zeros_like(first)
    first[-1, 1:7, 1:7] = 1
    second[0, 1:7, 1:7] = 1
    second[1, 0, 0] = 2
    first_global, _ = merge_utils.global_segment_ids(first, (0, 0, 0), nblocks)
    second_global, _ = merge_utils.global_segment_ids(second, (1, 0, 0), nblocks)
    crossing_second = np.uint32((1 << merge_utils.GLOBAL_LABEL_BITS) | 1)
    isolated_second = np.uint32((1 << merge_utils.GLOBAL_LABEL_BITS) | 2)

    labeling = merge_utils.determine_merge_relabeling(
        [(0, 0, 0), (1, 0, 0)],
        [
            merge_utils.block_faces(first_global, shrink=True),
            merge_utils.block_faces(second_global, shrink=True),
        ],
        np.asarray([1, crossing_second, isolated_second], dtype=np.uint32),
        pre_shrunk=True,
    )

    assert labeling[1] == labeling[crossing_second]
    assert labeling[isolated_second] != labeling[1]


def test_boundary_pair_task_returns_only_unique_label_pairs() -> None:
    left = np.asarray([[[1, 1, 0], [2, 2, 0]]], dtype=np.uint32)
    right = np.asarray([[[7, 7, 0], [8, 8, 0]]], dtype=np.uint32)

    pairs = postproc._boundary_label_pairs(left, right)

    np.testing.assert_array_equal(
        pairs,
        np.asarray([[1, 2], [7, 8]], dtype=np.uint32),
    )


def test_boundary_contacts_keep_raw_overlap_for_eroded_fragment() -> None:
    left = np.zeros((1, 4, 4), dtype=np.uint32)
    right = np.zeros_like(left)
    left[:, 1:3, 1:3] = 1
    right[:, 1:3, 1:3] = 7

    robust_pairs, raw_pairs, raw_counts = postproc._boundary_label_contacts(left, right)

    assert robust_pairs.shape == (2, 0)
    np.testing.assert_array_equal(raw_pairs, np.asarray([[1], [7]], dtype=np.uint32))
    np.testing.assert_array_equal(raw_counts, np.asarray([4], dtype=np.uint64))


def test_sparse_merge_mapping_scales_with_used_labels() -> None:
    distant = np.uint32((50_000 << merge_utils.GLOBAL_LABEL_BITS) | 7)
    isolated = np.uint32((50_000 << merge_utils.GLOBAL_LABEL_BITS) | 8)

    mapping = merge_utils.determine_sparse_merge_relabeling(
        np.asarray([1, distant, isolated], dtype=np.uint32),
        [np.asarray([[1], [distant]], dtype=np.uint32)],
    )

    assert mapping.shape == (2, 3)
    assert mapping.nbytes == 2 * 3 * np.dtype(np.uint32).itemsize
    np.testing.assert_array_equal(mapping[0], [1, distant, isolated])
    assert mapping[1, 0] == mapping[1, 1]
    assert mapping[1, 2] != mapping[1, 0]


def test_final_label_volumes_sum_merged_core_counts() -> None:
    global_ids = np.asarray([30, 10, 20], dtype=np.uint32)
    core_counts = np.asarray([7, 2, 5], dtype=np.uint64)
    mapping = np.asarray(
        [[10, 20, 30], [1, 2, 1]],
        dtype=np.uint32,
    )

    volumes = postproc._aggregate_final_label_volumes(
        global_ids,
        core_counts,
        mapping,
    )

    assert volumes.dtype == np.uint64
    np.testing.assert_array_equal(volumes, [0, 9, 5])


def test_final_label_volumes_reject_incomplete_mapping() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        postproc._aggregate_final_label_volumes(
            np.asarray([10, 20], dtype=np.uint32),
            np.asarray([2, 5], dtype=np.uint64),
            np.asarray([[10], [1]], dtype=np.uint32),
        )


def test_small_boundary_recovery_chooses_strongest_large_neighbor() -> None:
    mapping = np.asarray(
        [[10, 20, 30, 40], [1, 2, 3, 4]],
        dtype=np.uint32,
    )
    final_volumes = np.asarray([0, 100, 1_000, 800, 200], dtype=np.uint64)
    raw_pairs = np.asarray(
        [[10, 10, 10, 40], [20, 20, 30, 30]],
        dtype=np.uint32,
    )
    contact_counts = np.asarray([3, 4, 6, 10], dtype=np.uint64)

    recovery_pairs = postproc._select_small_boundary_merges(
        mapping,
        final_volumes,
        raw_pairs,
        contact_counts,
        V_min=500,
    )

    np.testing.assert_array_equal(
        recovery_pairs,
        np.asarray([[10, 40], [20, 30]], dtype=np.uint32),
    )


def test_small_boundary_recovery_does_not_merge_large_components() -> None:
    mapping = np.asarray([[10, 20], [1, 2]], dtype=np.uint32)

    recovery_pairs = postproc._select_small_boundary_merges(
        mapping,
        np.asarray([0, 700, 800], dtype=np.uint64),
        np.asarray([[10], [20]], dtype=np.uint32),
        np.asarray([50], dtype=np.uint64),
        V_min=500,
    )

    assert recovery_pairs.shape == (2, 0)


def test_unrecoverable_small_final_labels_are_dropped_and_compacted() -> None:
    mapping = np.asarray(
        [[10, 20, 30], [1, 2, 3]],
        dtype=np.uint32,
    )
    final_volumes = np.asarray([0, 100, 800, 900], dtype=np.uint64)

    filtered = postproc._drop_small_final_labels(mapping, final_volumes, V_min=500)

    np.testing.assert_array_equal(
        filtered,
        np.asarray([[10, 20, 30], [0, 1, 2]], dtype=np.uint32),
    )


def test_sparse_relabel_write_maps_large_global_ids(tmp_path: Path) -> None:
    distant = np.uint32((50_000 << merge_utils.GLOBAL_LABEL_BITS) | 7)
    isolated = np.uint32((50_000 << merge_utils.GLOBAL_LABEL_BITS) | 8)
    data = np.asarray([[[0, 1, distant], [isolated, distant, 0]]], dtype=np.uint32)
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        data=data,
        chunks=data.shape,
    )
    mapping = np.asarray(
        [[1, distant, isolated], [1, 1, 2]],
        dtype=np.uint32,
    )
    mapping_path = tmp_path / "mapping.npy"
    np.save(mapping_path, mapping)

    merge_utils.sparse_relabel_and_write(
        temp,
        mapping_path,
        tmp_path / "output.zarr",
    )

    output = zarr.open_array(tmp_path / "output.zarr", mode="r")
    np.testing.assert_array_equal(
        output[:],
        np.asarray([[[0, 1, 1], [2, 1, 0]]], dtype=np.uint32),
    )


def test_sparse_relabel_write_can_drop_a_global_label(tmp_path: Path) -> None:
    data = np.asarray([[[0, 10, 20]]], dtype=np.uint32)
    temp = zarr.create_array(tmp_path / "temp.zarr", data=data, chunks=data.shape)
    mapping_path = tmp_path / "mapping.npy"
    np.save(
        mapping_path,
        np.asarray([[10, 20], [1, 0]], dtype=np.uint32),
    )

    merge_utils.sparse_relabel_and_write(temp, mapping_path, tmp_path / "output.zarr")

    output = zarr.open_array(tmp_path / "output.zarr", mode="r")
    np.testing.assert_array_equal(output[:], np.asarray([[[0, 1, 0]]], dtype=np.uint32))


def test_sparse_relabel_write_reads_only_selected_chunks(tmp_path: Path) -> None:
    data = np.asarray(
        [
            [[10, 0, 99, 99], [0, 10, 99, 99]],
            [[88, 88, 20, 0], [88, 88, 0, 20]],
        ],
        dtype=np.uint32,
    )
    temp = zarr.create_array(
        tmp_path / "temp.zarr",
        data=data,
        chunks=(1, 2, 2),
    )
    mapping_path = tmp_path / "mapping.npy"
    np.save(
        mapping_path,
        np.asarray([[10, 20], [1, 2]], dtype=np.uint32),
    )

    merge_utils.sparse_relabel_and_write(
        temp,
        mapping_path,
        tmp_path / "output.zarr",
        block_token_chunks=True,
        chunk_coords=[(0, 0, 0), (1, 0, 1)],
    )

    output = zarr.open_array(tmp_path / "output.zarr", mode="r")
    np.testing.assert_array_equal(
        output[:],
        np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 2, 0], [0, 0, 0, 2]],
            ],
            dtype=np.uint32,
        ),
    )


def test_face_pairing_does_not_infer_axis_from_extent_two() -> None:
    nblocks = np.asarray((1, 2, 1))
    first = np.zeros((2, 2, 8), dtype=np.uint32)
    second = np.zeros_like(first)
    first[0, -1, 2:6] = 1
    second[0, 0, 2:6] = 1
    first_global, _ = merge_utils.global_segment_ids(first, (0, 0, 0), nblocks)
    second_global, _ = merge_utils.global_segment_ids(second, (0, 1, 0), nblocks)
    crossing_second = np.uint32((1 << merge_utils.GLOBAL_LABEL_BITS) | 1)

    labeling = merge_utils.determine_merge_relabeling(
        [(0, 0, 0), (0, 1, 0)],
        [merge_utils.block_faces(first_global), merge_utils.block_faces(second_global)],
        np.asarray([1, crossing_second], dtype=np.uint32),
        pre_shrunk=True,
    )

    assert labeling[1] == labeling[crossing_second]


def test_postproc_cli_inherits_zyx_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "segmentation.zarr"
    zarr.create_array(
        input_path,
        shape=(100, 120, 140),
        chunks=(70, 80, 90),
        dtype=np.uint32,
    )
    captured: dict[str, Any] = {}

    def fake_distributed_postproc(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(postproc, "distributed_postproc", fake_distributed_postproc)

    result = CliRunner().invoke(
        app,
        ["postproc", "run", str(input_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured["blocksize"] is None
    assert captured["margin"] == 30
    assert captured["cluster_kwargs"] == {
        "workers_per_gpu": 1,
        "threads_per_worker": 1,
    }
    assert captured["overwrite"] is False


def test_postproc_clis_forward_explicit_zyx_blocksize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "segmentation.zarr"
    zarr.create_array(
        input_path,
        shape=(100, 120, 140),
        chunks=(70, 80, 90),
        dtype=np.uint32,
    )
    observed: list[tuple[int, int, int] | None] = []

    def fake_distributed_postproc(**kwargs: Any) -> None:
        observed.append(kwargs["blocksize"])

    monkeypatch.setattr(postproc, "distributed_postproc", fake_distributed_postproc)

    typer_result = CliRunner().invoke(
        app,
        ["postproc", "run", str(input_path), "--blocksize", "70", "80", "90"],
    )
    click_result = ClickCliRunner().invoke(
        postproc.cli,
        ["run", str(input_path), "--blocksize", "70", "80", "90"],
    )

    assert typer_result.exit_code == 0, typer_result.output
    assert click_result.exit_code == 0, click_result.output
    assert observed == [(70, 80, 90), (70, 80, 90)]


def test_postproc_owner_publishes_unique_workspace_and_returns_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.zarr"
    input_zarr = zarr.create_array(
        input_path,
        data=np.ones((2, 3, 4), dtype=np.uint32),
        chunks=(1, 3, 4),
    )
    workspaces: list[Path] = []

    def fake_run(**kwargs: Any) -> zarr.Array:
        staged_path = Path(kwargs["write_path"])
        workspace = Path(kwargs["temporary_directory"])
        workspaces.append(workspace)
        staged = zarr.create_array(
            staged_path,
            data=np.full((2, 3, 4), len(workspaces), dtype=np.uint32),
            chunks=(1, 3, 4),
        )
        np.save(staged_path / "volumes.npy", np.asarray([0, 24], dtype=np.uint64))
        return staged

    monkeypatch.setattr(postproc, "_run_distributed_postproc", fake_run)
    workspace_parent = tmp_path / "work"

    first = postproc.distributed_postproc.__wrapped__(
        input_zarr=input_zarr,
        write_path=tmp_path / "first.zarr",
        input_path=input_path,
        cluster=object(),
        temporary_directory=workspace_parent,
    )
    second = postproc.distributed_postproc.__wrapped__(
        input_zarr=input_zarr,
        write_path=tmp_path / "second.zarr",
        input_path=input_path,
        cluster=object(),
        temporary_directory=workspace_parent,
    )

    np.testing.assert_array_equal(first[:], np.ones((2, 3, 4), dtype=np.uint32))
    np.testing.assert_array_equal(second[:], np.full((2, 3, 4), 2, dtype=np.uint32))
    assert workspaces[0] != workspaces[1]
    assert all(not workspace.exists() for workspace in workspaces)
    assert (tmp_path / "first.zarr" / "volumes.npy").is_file()


def test_postproc_metadata_owns_transformed_artifact_identity(tmp_path: Path) -> None:
    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        shape=(2, 3, 4),
        chunks=(1, 3, 4),
        dtype=np.uint32,
    )
    input_zarr.attrs.update(
        {
            "key": ["labels"],
            "squisher_run_key": "raw-segmentation-run",
            "squisher_output_schema": {"shape": [2, 3, 4]},
        }
    )
    output_path = tmp_path / "output.zarr"
    zarr.create_array(
        output_path,
        shape=input_zarr.shape,
        chunks=input_zarr.chunks,
        dtype=np.uint32,
    )

    postproc._copy_zarr_metadata(
        input_zarr,
        output_path,
        postproc_params={"sigma": [1.0, 2.0, 2.0]},
    )

    output = zarr.open_array(output_path, mode="r")
    assert output.attrs["key"] == ["labels"]
    assert "squisher_run_key" not in output.attrs
    assert "squisher_output_schema" not in output.attrs
    assert output.attrs["squisher_postproc"]["source_run_key"] == "raw-segmentation-run"
    assert len(output.attrs["squisher_postproc_key"]) == 64


def test_postproc_metadata_records_dynamic_label_bits(tmp_path: Path) -> None:
    input_zarr = zarr.create_array(
        tmp_path / "input.zarr",
        shape=(1, 1, 1),
        chunks=(1, 1, 1),
        dtype=np.uint32,
    )
    output_path = tmp_path / "output.zarr"
    zarr.create_array(
        output_path,
        shape=input_zarr.shape,
        chunks=input_zarr.chunks,
        dtype=np.uint32,
    )

    postproc._copy_zarr_metadata(
        input_zarr,
        output_path,
        nblocks=(9, 15, 12),
        mapping_filename="label_mapping.npy",
    )

    mapping_metadata = zarr.open_array(output_path, mode="r").attrs["label_mapping"]
    assert mapping_metadata["label_bits"] == 21
    assert mapping_metadata["local_label_mask"] == (1 << 21) - 1


def test_postproc_rejects_source_as_destination(tmp_path: Path) -> None:
    input_path = tmp_path / "input.zarr"
    input_zarr = zarr.create_array(
        input_path,
        shape=(2, 3, 4),
        chunks=(1, 3, 4),
        dtype=np.uint32,
    )

    with pytest.raises(ValueError, match="must differ"):
        postproc.distributed_postproc.__wrapped__(
            input_zarr=input_zarr,
            write_path=input_path,
            input_path=input_path,
            cluster=object(),
            overwrite=True,
        )


def test_postproc_publish_restores_prior_output_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output.zarr"
    staged_path = tmp_path / "staged.zarr"
    zarr.create_array(
        output_path,
        data=np.full((1, 2, 3), 7, dtype=np.uint32),
        chunks=(1, 2, 3),
    )
    zarr.create_array(
        staged_path,
        data=np.full((1, 2, 3), 9, dtype=np.uint32),
        chunks=(1, 2, 3),
    )
    original_replace = postproc.os.replace

    def fail_staged_replace(source: Path | str, destination: Path | str) -> None:
        if Path(source) == staged_path and Path(destination) == output_path:
            raise OSError("publish failed")
        original_replace(source, destination)

    monkeypatch.setattr(postproc.os, "replace", fail_staged_replace)

    with pytest.raises(OSError, match="publish failed"):
        postproc._publish_postproc_output(staged_path, output_path, overwrite=True)

    restored = zarr.open_array(output_path, mode="r")
    np.testing.assert_array_equal(restored[:], np.full((1, 2, 3), 7, dtype=np.uint32))
