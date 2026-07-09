from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from multiview_stitcher import param_utils, spatial_image_utils as si_utils, transformation

from squisher_lightsheet.fusion import (
    canonical_fusion_base_output,
    channel_output_path,
    channel_output_paths,
    coarse_preibisch_content_weights,
    fuse_tiles,
    temporary_basic_disk_cache_dir,
)
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


def write_basic_sampling_manifest(
    path: Path, *, sort_intensity: bool = True, input_dirs: list[str] | None = None
) -> None:
    path.mkdir(parents=True)
    payload = {
        "sort_intensity": sort_intensity,
        "input_dirs": input_dirs if input_dirs is not None else [str(path.parent)],
    }
    (path / "basic-sampling.json").write_text(json.dumps(payload) + "\n")


def test_channel_output_paths_match_legacy_per_channel_names() -> None:
    assert channel_output_path(Path("/run/fused.ome.zarr"), 0) == Path("/run/fused.ch0.ome.zarr")
    assert channel_output_path(Path("/run/fused.zarr"), 1) == Path("/run/fused.ch1.zarr")
    assert canonical_fusion_base_output(Path("/run/final")) == Path("/run/final/fused.ome.zarr")
    assert channel_output_path(Path("/run/final"), 0) == Path("/run/final/fused.ch0.ome.zarr")
    assert channel_output_paths(Path("/run/fused.ome.zarr"), [0, 2]) == [
        Path("/run/fused.ch0.ome.zarr"),
        Path("/run/fused.ch2.ome.zarr"),
    ]


def test_fuse_uses_backend_supported_defaults(monkeypatch) -> None:
    captured = {}

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "ok"

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=Path("/run"),
        position_input=Path("/run/positions.json"),
        registration_input=Path("/run/registration.json"),
        output=Path("/run/fused.ome.zarr"),
        dry_run=True,
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    args = captured["args"]
    assert args[args.index("--fusion-weight-mode") + 1] == "content-preibisch-coarse"
    assert args[args.index("--fusion-level") + 1] == "0"
    assert args[args.index("--batch-size") + 1] == "4"
    assert args[args.index("--basic-cache-tiles") + 1] == "64"
    assert "--per-chunk-cupy-cleanup" not in args


def test_fuse_normalizes_canonical_output_dir(monkeypatch) -> None:
    captured = {}

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["args"] = args
        return script_name

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=Path("/run/input"),
        position_input=Path("/run/positions.json"),
        registration_input=Path("/run/registration.json"),
        output=Path("/run/final"),
    )

    args = captured["args"]
    assert args[args.index("--output") + 1] == "/run/final/fused.ome.zarr"


def test_fuse_passes_optional_basic_disk_cache(monkeypatch) -> None:
    captured = {}

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "ok"

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=Path("/run"),
        position_input=Path("/run/positions.json"),
        registration_input=Path("/run/registration.json"),
        output=Path("/run/fused.ome.zarr"),
        basic_cache_disk_dir=Path("/run/basic-slab-cache"),
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    args = captured["args"]
    assert args[args.index("--basic-cache-disk-dir") + 1] == "/run/basic-slab-cache"


def test_fuse_passes_fusion_level(monkeypatch) -> None:
    captured = {}

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["args"] = args
        return script_name

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=Path("/run"),
        position_input=Path("/run/positions.json"),
        registration_input=Path("/run/registration.json"),
        output=Path("/run/fused.ome.zarr"),
        fusion_level=4,
    )

    assert captured["args"][captured["args"].index("--fusion-level") + 1] == "4"


def test_fuse_passes_output_chunksize(monkeypatch) -> None:
    captured = {}

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["args"] = args
        return script_name

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=Path("/run"),
        position_input=Path("/run/positions.json"),
        registration_input=Path("/run/registration.json"),
        output=Path("/run/fused.ome.zarr"),
        output_chunksize_zyx=(12, 240, 240),
    )

    args = captured["args"]
    assert args[args.index("--output-chunksize") + 1 : args.index("--output-chunksize") + 4] == [
        "12",
        "240",
        "240",
    ]


def test_fuse_passes_output_grid_template(monkeypatch) -> None:
    captured = {}

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["args"] = args
        return script_name

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=Path("/run"),
        position_input=Path("/run/positions.json"),
        registration_input=Path("/run/registration.json"),
        output=Path("/run/fused.ome.zarr"),
        output_grid_template=Path("/run/reference.ch0.ome.zarr"),
    )

    args = captured["args"]
    assert args[args.index("--output-grid-template") + 1] == "/run/reference.ch0.ome.zarr"


def test_fuse_passes_source_view_flatfield_dirs(monkeypatch, tmp_path) -> None:
    captured = {}
    basic_left = tmp_path / "basic-left"
    basic_right = tmp_path / "basic-right"
    write_basic_sampling_manifest(basic_left)
    write_basic_sampling_manifest(basic_right)

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["script_name"] = script_name
        captured["args"] = args
        captured["dry_run"] = dry_run
        return "ok"

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=Path("/run"),
        position_input=Path("/run/positions.json"),
        registration_input=Path("/run/registration.json"),
        output=Path("/run/fused.ome.zarr"),
        flatfield_dirs_by_source_view={
            "L": basic_left,
            "R": basic_right,
        },
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    args = captured["args"]
    view_dir_args = [args[index + 1] for index, value in enumerate(args) if value == "--flatfield-dir-by-source-view"]
    assert view_dir_args == [f"L={basic_left}", f"R={basic_right}"]


def test_fuse_rejects_pooled_source_view_flatfield(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", lambda *_args, **_kwargs: "ok")
    pooled = tmp_path / "basic_nz25_pooled"
    write_basic_sampling_manifest(pooled)

    with pytest.raises(ValueError, match="pooled BaSiC"):
        fuse_tiles(
            input_dir=Path("/run"),
            position_input=Path("/run/positions.json"),
            registration_input=Path("/run/registration.json"),
            output=Path("/run/fused.ome.zarr"),
            flatfield_dirs_by_source_view={"L": pooled},
        )


def test_fuse_rejects_unsorted_source_view_flatfield(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", lambda *_args, **_kwargs: "ok")
    unsorted = tmp_path / "basic-left"
    write_basic_sampling_manifest(unsorted, sort_intensity=False)

    with pytest.raises(ValueError, match="sort_intensity=true"):
        fuse_tiles(
            input_dir=Path("/run"),
            position_input=Path("/run/positions.json"),
            registration_input=Path("/run/registration.json"),
            output=Path("/run/fused.ome.zarr"),
            flatfield_dirs_by_source_view={"L": unsorted},
        )


def test_fuse_rejects_multi_input_source_view_flatfield(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", lambda *_args, **_kwargs: "ok")
    pooled_manifest = tmp_path / "basic-left"
    write_basic_sampling_manifest(pooled_manifest, input_dirs=["/left", "/right"])

    with pytest.raises(ValueError, match="exactly one input_dir"):
        fuse_tiles(
            input_dir=Path("/run"),
            position_input=Path("/run/positions.json"),
            registration_input=Path("/run/registration.json"),
            output=Path("/run/fused.ome.zarr"),
            flatfield_dirs_by_source_view={"L": pooled_manifest},
        )


def test_temporary_basic_disk_cache_dir_uses_output_parent_and_cleans_up(tmp_path) -> None:
    output = tmp_path / "fused.ch0.ome.zarr"

    with temporary_basic_disk_cache_dir(None, output) as cache_dir:
        assert cache_dir.parent == tmp_path
        assert cache_dir.exists()
        (cache_dir / "sentinel").write_text("cache")

    assert not cache_dir.exists()


def test_bounded_memory_cache_enforces_entry_and_byte_limits() -> None:
    cache = legacy.BoundedMemoryCache(max_bytes=10, max_entries=2)

    cache.put("a", b"aaaa", nbytes=4)
    cache.put("b", b"bbbb", nbytes=4)
    assert cache.get("a") == b"aaaa"

    cache.put("c", b"cccc", nbytes=4)

    assert list(cache.data) == ["a", "c"]
    assert cache.total_bytes == 8
    assert cache.evictions == 1
    assert cache.hits == 1

    cache.put("d", b"dddddddd", nbytes=8)

    assert list(cache.data) == ["d"]
    assert cache.total_bytes == 8
    assert cache.evictions == 3


def test_bounded_memory_cache_supports_dask_cache_protocol() -> None:
    da = pytest.importorskip("dask.array")
    from dask.cache import Cache

    cache = legacy.BoundedMemoryCache(max_bytes=1024 * 1024)

    with Cache(cache):
        da.ones((8,), chunks=4).sum().compute(scheduler="single-threaded")

    assert cache.data


def test_mvs_affine_transform_is_pull_from_output_to_source() -> None:
    source = np.zeros((1, 1, 12), dtype=np.float32)
    source[0, 0, 3] = 1.0
    sim = si_utils.get_sim_from_array(
        source,
        dims=["z", "y", "x"],
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 10.0},
    ).isel(c=0, t=0)
    output_stack_properties = {
        "origin": {"z": 0.0, "y": 0.0, "x": 0.0},
        "spacing": {"z": 1.0, "y": 1.0, "x": 1.0},
        "shape": {"z": 1, "y": 1, "x": 40},
    }

    forward_fixed_from_source = np.eye(4, dtype=np.float64)
    forward_fixed_from_source[2, 3] = 17.0
    pull_source_from_fixed = np.linalg.inv(forward_fixed_from_source)

    forward_result = transformation.transform_sim(
        sim,
        p=param_utils.affine_to_xaffine(forward_fixed_from_source),
        output_stack_properties=output_stack_properties,
        order=0,
    )
    pull_result = transformation.transform_sim(
        sim,
        p=param_utils.affine_to_xaffine(pull_source_from_fixed),
        output_stack_properties=output_stack_properties,
        order=0,
    )

    forward_data = np.asarray(si_utils._get_backend_data(forward_result).compute())
    pull_data = np.asarray(si_utils._get_backend_data(pull_result).compute())

    assert not np.any(forward_data > 0.5)
    assert np.argwhere(pull_data[0, 0] > 0.5).ravel().tolist() == [30]


def test_legacy_fusion_flatfield_is_opt_in(monkeypatch, tmp_path) -> None:
    input_dir = tmp_path / "tiles"
    flatfield_dir = tmp_path / "basic"
    monkeypatch.setattr(sys, "argv", ["stitch", str(input_dir)])

    args = legacy.parse_args()

    assert args.flatfield_dir is None
    assert args.basic_cache_max_gib == legacy.DEFAULT_BASIC_CACHE_MAX_GIB

    monkeypatch.setattr(
        sys,
        "argv",
        ["stitch", str(input_dir), "--flatfield-dir", str(flatfield_dir)],
    )

    args = legacy.parse_args()

    assert args.flatfield_dir == flatfield_dir


def test_profile_fusion_skip_requires_max_batches() -> None:
    with pytest.raises(ValueError, match="--profile-skip-fusion-batches requires --profile-max-fusion-batches"):
        legacy.validate_profile_fusion_options(
            Namespace(
                profile_skip_fusion_batches=1,
                profile_max_fusion_batches=None,
            )
        )


def test_fusion_block_culling_matches_spatial_suffix_for_channel_prefixed_blocks() -> None:
    seen_batches = []

    def fake_batch_func(_fuse_chunk, batch, **_kwargs):
        seen_batches.append(list(batch))

    cull = legacy.culling_fusion_batch_func(fake_batch_func, {(0, 3, 1), (2, 4, 5)})
    cull(
        lambda _block_id: None,
        [
            (1, 0, 3, 1),
            (1, 0, 3, 2),
            (1, 2, 4, 5),
        ],
    )

    assert seen_batches == [[(1, 0, 3, 1), (1, 2, 4, 5)]]


def test_direct_fusion_view_candidate_plan_roundtrips_compact_json(tmp_path) -> None:
    path = tmp_path / "view-candidate-plan.json"
    candidate_map = {
        (0, 3, 1): [7, 2],
        (2, 4, 5): [9],
    }

    legacy.write_direct_fusion_view_candidate_plan(path, candidate_map, {"planned_blocks": 2})
    legacy._MVS_VIEW_CANDIDATE_PLANS.clear()

    assert legacy.load_direct_fusion_view_candidate_plan(str(path)) == candidate_map


def test_validate_written_scale0_rejects_shape_mismatch(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    output = tmp_path / "out.ome.zarr"
    zarr.open(
        str(output / "0"),
        mode="w",
        shape=(2, 3),
        chunks=(1, 3),
        dtype="uint8",
        zarr_format=3,
        dimension_names=("y", "x"),
    )

    class Sim:
        shape = (2, 4)
        dims = ("y", "x")

    with pytest.raises(ValueError, match="scale-0 shape mismatch"):
        legacy.validate_written_scale0(output, Sim())


def test_validate_written_scale0_accepts_squeezed_singleton_axes(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    output = tmp_path / "out.ome.zarr"
    zarr.open(
        str(output / "0"),
        mode="w",
        shape=(3, 4, 5),
        chunks=(1, 4, 5),
        dtype="uint8",
        zarr_format=3,
        dimension_names=("z", "y", "x"),
    )

    class Sim:
        shape = (1, 1, 3, 4, 5)
        dims = ("t", "c", "z", "y", "x")

    legacy.validate_written_scale0(output, Sim())


def test_cleanup_or_preserve_fusion_workspace_only_removes_completed(tmp_path) -> None:
    failed = tmp_path / ".fused.ch0.ome.zarr.fusion-failed"
    failed.mkdir()
    (failed / "sentinel").write_text("partial output")

    legacy.cleanup_or_preserve_fusion_workspace(failed, completed=False)

    assert failed.exists()
    assert (failed / "sentinel").exists()

    completed = tmp_path / ".fused.ch0.ome.zarr.fusion-completed"
    completed.mkdir()
    (completed / "sentinel").write_text("moved output")

    legacy.cleanup_or_preserve_fusion_workspace(completed, completed=True)

    assert not completed.exists()


def test_squeezed_scale0_metadata_repairs_pyramids_and_marks_complete(monkeypatch, tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    output = tmp_path / "out.ome.zarr"
    group = zarr.open_group(str(output), mode="w", zarr_format=3)
    group.attrs["ome"] = {
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 0.6, 0.3, 0.3]},
                            {"type": "translation", "translation": [0.0, 0.0, 10.0, 20.0, 30.0]},
                        ],
                    }
                ],
            }
        ]
    }
    zarr.open(
        str(output / "0"),
        mode="w",
        shape=(224, 224, 224),
        chunks=(112, 112, 112),
        dtype="uint16",
        zarr_format=3,
        dimension_names=("z", "y", "x"),
    )

    class Sim:
        shape = (1, 1, 224, 224, 224)
        dims = ("t", "c", "z", "y", "x")

    def fake_block_reduce_mean_gpu(block: np.ndarray, factors: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        reduced = block.reshape(
            block.shape[0] // factors[0],
            factors[0],
            block.shape[1] // factors[1],
            factors[1],
            block.shape[2] // factors[2],
            factors[2],
        ).mean(axis=(1, 3, 5))
        return reduced.astype(dtype)

    monkeypatch.setattr(legacy, "block_reduce_mean_gpu", fake_block_reduce_mean_gpu)

    legacy.validate_written_scale0(output, Sim())
    legacy.repair_ome_metadata_axes(output)
    legacy.build_ome_zarr_pyramid_from_scale0(output)
    legacy.mark_ome_zarr_complete(output)

    assert legacy.ome_zarr_is_complete(output)
    ome = legacy.read_ome_zarr_group_metadata(output)
    multiscale = ome["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["z", "y", "x"]
    assert [dataset["path"] for dataset in multiscale["datasets"]] == ["0", "1"]
    for dataset in multiscale["datasets"]:
        for transform in dataset["coordinateTransformations"]:
            values = transform.get("scale") or transform.get("translation")
            assert len(values) == 3


def test_ome_zarr_complete_requires_completion_marker(tmp_path) -> None:
    output = tmp_path / "out.ome.zarr"
    (output / "0").mkdir(parents=True)
    (output / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": {
                    "ome": {
                        "multiscales": [
                            {
                                "axes": [{"name": "y"}, {"name": "x"}],
                                "datasets": [{"path": "0"}],
                            }
                        ]
                    }
                }
            }
        )
        + "\n"
    )
    (output / "0" / "zarr.json").write_text(json.dumps({"shape": [4, 5]}) + "\n")

    assert legacy.ome_zarr_has_multiscales_metadata(output)
    assert not legacy.ome_zarr_is_complete(output)
    assert "no squisher_complete marker" in str(legacy.output_exists_error(output))

    payload = json.loads((output / "zarr.json").read_text())
    payload["attributes"]["squisher_complete"] = True
    (output / "zarr.json").write_text(json.dumps(payload) + "\n")
    assert legacy.ome_zarr_is_complete(output)


def test_downsampled_pyramid_writes_shard_aligned_blocks(monkeypatch, tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    source = zarr.open(
        str(tmp_path / "source.zarr"),
        mode="w",
        shape=(8, 12, 16),
        chunks=(4, 6, 8),
        dtype="uint16",
        zarr_format=3,
        dimension_names=("z", "y", "x"),
    )
    source[:] = np.arange(8 * 12 * 16, dtype=np.uint16).reshape(8, 12, 16)
    seen_block_shapes = []

    def fake_block_reduce_mean_gpu(block: np.ndarray, factors: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        seen_block_shapes.append(block.shape)
        reduced = block.reshape(
            block.shape[0] // factors[0],
            factors[0],
            block.shape[1] // factors[1],
            factors[1],
            block.shape[2] // factors[2],
            factors[2],
        ).mean(axis=(1, 3, 5))
        return reduced.astype(dtype)

    monkeypatch.setattr(legacy, "block_reduce_mean_gpu", fake_block_reduce_mean_gpu)

    legacy.write_zstd_downsampled_level(
        source,
        tmp_path / "pyramid.ome.zarr",
        dataset_path="1",
        dimension_names=("z", "y", "x"),
        factors={"z": 2, "y": 3, "x": 2},
    )

    destination = zarr.open(str(tmp_path / "pyramid.ome.zarr" / "1"), mode="r")
    assert destination.chunks == (2, 2, 4)
    assert destination.metadata.shards == (4, 4, 8)
    assert seen_block_shapes == [(8, 12, 16)]


def cached_metadata_payload(input_dir: Path, tile_names: list[str]) -> dict:
    return {
        "units": "micrometer",
        "input_dir": str(input_dir),
        "tiles": [
            {
                "tile": tile_name,
                "path": str(input_dir / tile_name),
                "shape": [5, 6, 7],
                "axes": "ZYX",
                "spacing_um": {"z": 1.0, "y": 0.5, "x": 0.5},
                "translation_um": {"z": index, "y": index + 1, "x": index + 2},
                "scale_um": {"z": 1.0, "y": 0.5, "x": 0.5},
                "channels": ["0"],
                "tracks": [
                    {
                        "slug": "track0",
                        "track_id": "all",
                        "channels": [0],
                        "channel_names": ["0"],
                    }
                ],
            }
            for index, tile_name in enumerate(tile_names)
        ],
    }


def test_position_input_uses_input_dir_cached_metadata_without_tiff_scan(monkeypatch, tmp_path) -> None:
    input_dir = tmp_path / "deconv"
    input_dir.mkdir()
    original_dir = tmp_path / "original"
    original_dir.mkdir()
    tile_names = ["tile0.ome.tif", "tile1.ome.tif"]
    for tile_name in tile_names:
        (input_dir / tile_name).touch()
        (original_dir / tile_name).touch()
    (input_dir / "sample.cached-metadata.registration-seed.json").write_text(
        json.dumps(cached_metadata_payload(input_dir, tile_names)) + "\n"
    )
    position_input = tmp_path / "positions.json"
    position_input.write_text(
        json.dumps(
            {
                "units": "micrometer",
                "tiles": [
                    {
                        "path": str(original_dir / tile_name),
                        "translation_um": {"z": index * 10, "y": index * 10 + 1, "x": index * 10 + 2},
                        "scale_um": {"z": 1.0, "y": 0.5, "x": 0.5},
                    }
                    for index, tile_name in enumerate(tile_names)
                ],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        legacy,
        "parse_ome_metadata",
        lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected TIFF scan: {path}")),
    )

    tiles = legacy.read_position_input_tiles(position_input, input_dir=input_dir)

    assert [tile.path for tile in tiles] == [input_dir / tile_name for tile_name in tile_names]
    assert [tile.translation["z"] for tile in tiles] == [0.0, 10.0]
    assert all(tile.shape == (5, 6, 7) for tile in tiles)


def test_position_input_reads_each_ome_zarr_shape(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    input_dir = tmp_path / "deconv"
    input_dir.mkdir()
    shapes = [(2, 4, 6, 7), (2, 3, 6, 7)]
    tile_paths = []
    for index, shape in enumerate(shapes):
        path = input_dir / f"tile{index}.ome.zarr"
        group = zarr.open_group(str(path), mode="w", zarr_format=3)
        group.create_array(
            "0",
            shape=shape,
            chunks=(1, 2, 6, 7),
            dtype="uint16",
            dimension_names=("c", "z", "y", "x"),
        )
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.5]},
                        ],
                    }
                ],
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
            }
        ]
        tile_paths.append(path)
    position_input = tmp_path / "positions.json"
    position_input.write_text(
        json.dumps(
            {
                "units": "micrometer",
                "tiles": [
                    {
                        "path": str(path),
                        "translation_um": {"z": 10.0 * index, "y": 0.0, "x": 0.0},
                        "scale_um": {
                            "z": -1.0 if index else 1.0,
                            "y": 0.5,
                            "x": -0.5 if index else 0.5,
                        },
                        "side": "R" if index else "L",
                    }
                    for index, path in enumerate(tile_paths)
                ],
            }
        )
        + "\n"
    )

    tiles = legacy.read_position_input_tiles(position_input, input_dir=input_dir)

    assert [tile.shape for tile in tiles] == shapes
    assert legacy.tile_flip_axes_zyx(tiles[1]) == (True, False, True)


def test_fusion_source_tile_uses_selected_ome_zarr_level_transform(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "tile.ome.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    for level, shape in enumerate(((2, 7, 9, 11), (2, 3, 4, 5))):
        array = group.create_array(str(level), shape=shape, chunks=shape, dtype="uint16")
        array.attrs["_ARRAY_DIMENSIONS"] = ["c", "z", "y", "x"]
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 2.0, 3.0, 4.0]},
                        {"type": "translation", "translation": [0.0, 10.0, 20.0, 30.0]},
                    ],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 5.0, 7.0, 9.0]},
                        {"type": "translation", "translation": [0.0, 11.5, 24.0, 35.0]},
                    ],
                },
            ],
        }
    ]
    tile = legacy.TileMetadata(
        path=path,
        shape=(2, 7, 9, 11),
        axes="CZYX",
        spacing={"z": 2.0, "y": 3.0, "x": 4.0},
        translation={"z": 100.0, "y": 200.0, "x": 300.0},
        channels=("0", "1"),
        tracks=(),
        stage_scale={"z": -2.0, "y": 3.0, "x": -4.0},
        source_view="R",
    )

    source_tile = legacy.fusion_tile_for_source_array(tile, (2, 3, 4, 5), source_level=1)

    assert source_tile.spacing == {"z": 5.0, "y": 7.0, "x": 9.0}
    assert source_tile.stage_scale == {"z": -5.0, "y": 7.0, "x": -9.0}
    assert source_tile.translation == {"z": 98.5, "y": 204.0, "x": 295.0}


def test_registration_input_uses_cached_metadata_without_tiff_scan(monkeypatch, tmp_path) -> None:
    input_dir = tmp_path / "deconv"
    input_dir.mkdir()
    tile_names = ["tile0.ome.tif", "tile1.ome.tif"]
    for tile_name in tile_names:
        (input_dir / tile_name).touch()
    (input_dir / "sample.cached-metadata.registration-seed.json").write_text(
        json.dumps(cached_metadata_payload(input_dir, tile_names)) + "\n"
    )
    registration_input = tmp_path / "registration.json"
    registration_input.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "tiles": [
                    {
                        "tile": tile_name,
                        "stage_translation_um": {"z": index * 10, "y": index * 10 + 1, "x": index * 10 + 2},
                        "stage_scale_um": {"z": 1.0, "y": 0.5, "x": 0.5},
                        "source_view": None,
                    }
                    for index, tile_name in enumerate(tile_names)
                ],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        legacy,
        "parse_ome_metadata",
        lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected TIFF scan: {path}")),
    )

    tiles = legacy.read_registration_input_tiles(registration_input)

    assert [tile.path for tile in tiles] == [input_dir / tile_name for tile_name in tile_names]
    assert [tile.translation["z"] for tile in tiles] == [0.0, 10.0]
    assert all(tile.channels == ("0",) for tile in tiles)


def test_registration_input_reads_ome_zarr_tile_metadata_from_registration_scale(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    input_dir = tmp_path / "rechunked"
    input_dir.mkdir()
    data = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    tile_names = ["tile0.ome.zarr", "tile1.ome.zarr"]
    for tile_name in tile_names:
        group = zarr.open_group(str(input_dir / tile_name), mode="w", zarr_format=3)
        array = group.create_array(
            "0",
            shape=data.shape,
            chunks=(2, 5, 6),
            dtype=data.dtype,
            dimension_names=("z", "y", "x"),
        )
        array[:] = data
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "datasets": [{"path": "0"}],
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
            }
        ]
    registration_input = tmp_path / "registration.json"
    registration_input.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "tiles": [
                    {
                        "tile": tile_name,
                        "stage_translation_um": {"z": index * 10, "y": index * 10 + 1, "x": index * 10 + 2},
                        "stage_scale_um": {"z": 0.6, "y": 0.287807, "x": 0.287807},
                        "source_view": None,
                    }
                    for index, tile_name in enumerate(tile_names)
                ],
            }
        )
        + "\n"
    )

    tiles = legacy.read_registration_input_tiles(registration_input)

    assert [tile.path for tile in tiles] == [input_dir / tile_name for tile_name in tile_names]
    assert all(tile.shape == data.shape for tile in tiles)
    assert all(tile.axes == "ZYX" for tile in tiles)
    assert all(tile.spacing == {"z": 0.6, "y": 0.287807, "x": 0.287807} for tile in tiles)
    assert [tile.translation["z"] for tile in tiles] == [0.0, 10.0]


def test_fusion_tile_reader_keeps_zarr_backed_optimized_path(tmp_path) -> None:
    tifffile = pytest.importorskip("tifffile")
    data = np.arange(4 * 3 * 2, dtype=np.uint16).reshape(4, 3, 2)
    path = tmp_path / "tile.ome.tif"
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"})
    tile = legacy.TileMetadata(
        path=path,
        shape=data.shape,
        axes="ZYX",
        spacing={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        channels=("0",),
        tracks=(
            legacy.TrackMetadata(
                slug="track0",
                track_id="all",
                channels=(0,),
                channel_names=("0",),
            ),
        ),
    )

    zarray, store, axes, shape = legacy.open_fusion_tile_array(tile, 0)
    try:
        assert axes == "ZYX"
        assert shape == data.shape
        np.testing.assert_array_equal(zarray[:], data)
    finally:
        legacy.close_stores([store])


def test_fusion_tile_reader_opens_ome_zarr_directly(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    data = np.arange(4 * 3 * 2, dtype=np.uint16).reshape(4, 3, 2)
    path = tmp_path / "tile.ome.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    array = group.create_array(
        "0",
        shape=data.shape,
        chunks=(2, 3, 2),
        dtype=data.dtype,
        dimension_names=("z", "y", "x"),
    )
    array[:] = data
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "datasets": [{"path": "0"}],
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }
    ]
    tile = legacy.parse_ome_metadata(path)

    zarray, store, axes, shape = legacy.open_fusion_tile_array(tile, 0)

    assert store is None
    assert axes == "ZYX"
    assert shape == data.shape
    np.testing.assert_array_equal(zarray[:], data)


def test_fusion_negative_stage_scale_uses_zarr_backed_orientation_affine(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    from multiview_stitcher.fusion import _core as fusion_core

    data = np.arange(2 * 4 * 3 * 5, dtype=np.uint16).reshape(2, 4, 3, 5)
    path = tmp_path / "tile.ome.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    array = group.create_array(
        "0",
        shape=data.shape,
        chunks=(1, 2, 3, 5),
        dtype=data.dtype,
        dimension_names=("c", "z", "y", "x"),
    )
    array[:] = data
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "datasets": [{"path": "0"}],
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }
    ]
    tile = legacy.TileMetadata(
        path=path,
        shape=data.shape,
        axes="CZYX",
        spacing={"z": 2.0, "y": 1.0, "x": 0.5},
        translation={"z": 100.0, "y": 20.0, "x": 50.0},
        channels=("0", "1"),
        tracks=(
            legacy.TrackMetadata(
                slug="track0",
                track_id="all",
                channels=(0, 1),
                channel_names=("0", "1"),
            ),
        ),
        stage_scale={"z": -2.0, "y": 1.0, "x": -0.5},
        source_view="R",
    )

    sims, stores, _label, _levels, source_tiles = legacy.build_fusion_sims([tile], 1)
    try:
        sim = sims[0]
        source_tile = source_tiles[0]
        assert si_utils.is_xarray_zarr_backed(sim)

        origin = legacy.tile_sim_translation(source_tile)
        spacing = legacy.tile_sim_scale(source_tile)
        orientation = si_utils.get_affine_from_sim(sim, transform_key=legacy.TRANSFORM_KEY)
        expected_orientation = np.eye(4)
        expected_orientation[0, 0] = -1.0
        expected_orientation[2, 2] = -1.0
        expected_orientation[0, 3] = 2 * origin["z"] + (data.shape[1] - 1) * spacing["z"]
        expected_orientation[2, 3] = 2 * origin["x"] + (data.shape[3] - 1) * spacing["x"]
        np.testing.assert_allclose(orientation, expected_orientation)

        output_stack_properties = legacy.reflected_fusion_output_stack_properties(
            sims,
            output_spacing=spacing,
            transform_key=legacy.TRANSFORM_KEY,
        )
        assert output_stack_properties == {
            "origin": origin,
            "spacing": spacing,
            "shape": {"z": data.shape[1], "y": data.shape[2], "x": data.shape[3]},
        }
        output_chunk_shape = {"z": 2, "y": 3, "x": 3}
        output_chunk_origin = {
            "z": origin["z"] + spacing["z"],
            "y": origin["y"],
            "x": origin["x"] + spacing["x"],
        }
        entry, _region = legacy.direct_zarr_block_entry(
            sims=[sim],
            transform_key=legacy.TRANSFORM_KEY,
            sim_coord_dict={},
            block_key=(0, 0, 0),
            output_stack_properties=output_stack_properties,
            output_chunksize=output_chunk_shape,
            output_chunk_shape=output_chunk_shape,
            output_chunk_origin=output_chunk_origin,
            overlap_in_pixels={"z": 0, "y": 0, "x": 0},
            interpolation_order=1,
        )
        fused = fusion_core._fuse_block_zarr_backed(
            np.asarray(entry, dtype=object),
            output_dtype=data.dtype,
            sim_coord_dict={},
            sdims=["z", "y", "x"],
            fusion_func=fusion_core.weighted_average_fusion,
            fusion_func_kwargs=None,
            weights_func=None,
            weights_func_kwargs=None,
            overlap_in_pixels={"z": 0, "y": 0, "x": 0},
            trim_overlap=True,
            interpolation_order=1,
            blending_widths=None,
            shrink_distance=0,
            backend=None,
        )

        np.testing.assert_array_equal(fused, data[1, ::-1, :, ::-1][1:3, :, 1:4])

        registered = np.eye(4)
        registered[1, 3] = 7.5
        si_utils.set_sim_affine(
            sim,
            xaffine=param_utils.affine_to_xaffine(registered),
            transform_key=legacy.REGISTERED_TRANSFORM_KEY,
            base_transform_key=legacy.TRANSFORM_KEY,
        )
        combined = si_utils.get_affine_from_sim(sim, transform_key=legacy.REGISTERED_TRANSFORM_KEY)
        np.testing.assert_allclose(combined, registered @ expected_orientation)
    finally:
        legacy.close_stores(stores)


def test_reflected_fusion_routes_mixed_views_through_candidate_worker(monkeypatch, tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    from multiview_stitcher import misc_utils
    from multiview_stitcher.fusion import _core as fusion_core

    level_shapes = ((3, 3, 4), (4, 3, 5))
    tiles = []
    for index, (side, level_shape) in enumerate(zip(("L", "R"), level_shapes, strict=True)):
        base_shape = tuple(size * 2 for size in level_shape)
        path = tmp_path / f"tile-{side}.ome.zarr"
        group = zarr.open_group(str(path), mode="w", zarr_format=2)
        for level, shape in enumerate((base_shape, level_shape)):
            array = group.create_array(str(level), shape=shape, chunks=shape, dtype="uint16")
            array.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
            array[:] = index + 1
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
                    },
                    {
                        "path": "1",
                        "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0, 2.0]}],
                    },
                ],
            }
        ]
        negative = side == "R"
        tiles.append(
            legacy.TileMetadata(
                path=path,
                shape=base_shape,
                axes="ZYX",
                spacing={"z": 1.0, "y": 1.0, "x": 1.0},
                translation={"z": 8.0 if negative else 0.0, "y": 0.0, "x": 18.0 if negative else 0.0},
                channels=("0",),
                tracks=(),
                stage_scale={"z": -1.0 if negative else 1.0, "y": 1.0, "x": -1.0 if negative else 1.0},
                source_view=side,
            )
        )

    sims, stores, _label, _levels, source_tiles = legacy.build_fusion_sims(tiles, 0, fusion_level=1)
    try:
        output_spacing = legacy.tile_sim_scale(source_tiles[0])
        output_stack_properties = legacy.reflected_fusion_output_stack_properties(
            sims,
            output_spacing=output_spacing,
            transform_key=legacy.TRANSFORM_KEY,
        )
        assert output_stack_properties == {
            "origin": {"z": 0.0, "y": 0.0, "x": 0.0},
            "spacing": {"z": 2.0, "y": 2.0, "x": 2.0},
            "shape": {"z": 4, "y": 3, "x": 9},
        }
        output_chunksize = {"z": 3, "y": 2, "x": 4}
        candidate_map, summary = legacy.direct_fusion_view_candidate_plan(
            sims,
            transform_key=legacy.TRANSFORM_KEY,
            output_stack_properties=output_stack_properties,
            output_chunksize=output_chunksize,
            weights_func=None,
            weights_func_kwargs=None,
            fusion_func=fusion_core.weighted_average_fusion,
            fusion_func_kwargs=None,
            interpolation_order=0,
        )
        assert summary["planned_blocks"] == len(candidate_map)
        assert any(indices == [0] for indices in candidate_map.values())
        assert any(indices == [1] for indices in candidate_map.values())

        plan_path = tmp_path / "candidate-plan.json"
        legacy.write_direct_fusion_view_candidate_plan(plan_path, candidate_map, summary)
        output_path = tmp_path / "fused-array.zarr"
        output_array = zarr.open_array(
            str(output_path),
            mode="w",
            shape=(4, 3, 9),
            chunks=(3, 2, 4),
            dtype="uint16",
            zarr_format=2,
        )
        calls = []

        def fake_fuse_block(chunk_params_block, **_kwargs):
            entry = np.asarray(chunk_params_block, dtype=object).item()
            shape = tuple(int(entry["output_bb"]["shape"][dim]) for dim in ("z", "y", "x"))
            calls.append((entry["block_key"], len(entry["views"]), shape))
            return np.full(shape, len(entry["views"]), dtype=np.uint16)

        class FakeDevice:
            def __init__(self, _device: int):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class FakeCupyArray:
            pass

        fake_cupy = type(
            "FakeCupy",
            (),
            {
                "cuda": type("FakeCuda", (), {"Device": FakeDevice}),
                "ndarray": FakeCupyArray,
                "asnumpy": staticmethod(np.asarray),
            },
        )
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
        monkeypatch.setattr(misc_utils, "clear_cupy_memory", lambda: None)
        monkeypatch.setattr(fusion_core, "_fuse_block_zarr_backed", fake_fuse_block)
        payload = {
            "output_zarr_url": str(output_path),
            "output_stack_properties": output_stack_properties,
            "ns_shape": {},
            "nsdims": (),
            "fuse_kwargs": {
                "images": sims,
                "transform_key": legacy.TRANSFORM_KEY,
                "fusion_func": fusion_core.weighted_average_fusion,
                "fusion_func_kwargs": None,
                "weights_func": None,
                "weights_func_kwargs": None,
                "interpolation_order": 0,
                "blending_widths": None,
                "backend": None,
            },
            "output_chunksize": output_chunksize,
            "view_candidate_plan_path": str(plan_path),
        }
        for block_id in sorted(candidate_map):
            legacy.run_mvs_fuse_chunk_loky_worker_once(block_id, payload, device=0)

        assert len(calls) == len(candidate_map)
        assert any(shape != (3, 2, 4) for _block, _views, shape in calls)
        assert np.count_nonzero(output_array[:]) > 0
    finally:
        legacy.close_stores(stores)


def test_zarr_safe_fusion_selection_avoids_deepcopying_tiff_store(tmp_path) -> None:
    tifffile = pytest.importorskip("tifffile")
    data = np.arange(4 * 3 * 2, dtype=np.uint16).reshape(4, 3, 2)
    path = tmp_path / "tile.ome.tif"
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"})
    tile = legacy.TileMetadata(
        path=path,
        shape=data.shape,
        axes="ZYX",
        spacing={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        channels=("0",),
        tracks=(
            legacy.TrackMetadata(
                slug="track0",
                track_id="all",
                channels=(0,),
                channel_names=("0",),
            ),
        ),
    )
    zarray, store, axes, shape = legacy.open_fusion_tile_array(tile, 0)
    try:
        source_tile = legacy.fusion_tile_for_source_array(tile, shape, source_level=0)
        sim = si_utils.get_sim_from_array(
            zarray,
            dims=["z", "y", "x"] if axes == "ZYX" else ["c", "z", "y", "x"],
            scale=legacy.tile_sim_scale(source_tile),
            translation=legacy.tile_sim_translation(source_tile),
            transform_key=legacy.TRANSFORM_KEY,
            c_coords=legacy.channel_labels_for_tiles([tile]),
        )
        selected = sim.isel(c=0, t=0)
        assert si_utils.is_xarray_zarr_backed(selected)

        with legacy.zarr_safe_fusion_selection():
            narrowed = si_utils.sim_sel_coords(selected, {})

        assert narrowed is selected
        assert si_utils.is_xarray_zarr_backed(narrowed)
    finally:
        legacy.close_stores([store])


def test_content_preibisch_coarse_softmax_sharpens_any_overlap(monkeypatch) -> None:
    calls = 0

    def fake_gaussian_filter(values, *, sigma, mode):
        nonlocal calls
        calls += 1
        return np.zeros_like(values) if calls == 1 else values

    monkeypatch.setattr("scipy.ndimage.gaussian_filter", fake_gaussian_filter)

    raw_weights = np.array([[0.25, 0.5, 1.0], [0.75, 0.5, np.nan]], dtype=np.float32)
    weights = coarse_preibisch_content_weights(
        np.sqrt(raw_weights),
        ~np.isnan(raw_weights),
        sigma_1=7,
        sigma_2=17,
        stride_zyx=(1,),
        softmax_exponent=2.0,
    )

    np.testing.assert_allclose(weights[:, 0], [0.1, 0.9], atol=1e-6)
    np.testing.assert_allclose(weights[:, 1], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(weights[:, 2], [1.0, 0.0], atol=1e-6)
