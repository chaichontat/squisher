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
from squisher_lightsheet import tiff_input


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
    assert channel_output_path(Path("/run/fused.ch0.ome.zarr"), 0) == Path("/run/fused.ch0.ome.zarr")
    assert legacy.channel_output_path(Path("/run/fused.ch0.ome.zarr"), 0, separate_channels=True) == Path(
        "/run/fused.ch0.ome.zarr"
    )


def test_channel_output_path_rejects_mismatched_qualified_channel() -> None:
    with pytest.raises(ValueError, match="already targets channel 1"):
        channel_output_path(Path("/run/fused.ch1.ome.zarr"), 0)


def test_create_fusion_temp_workspace_creates_missing_output_parent(tmp_path: Path) -> None:
    channel_output = tmp_path / "new-output" / "fused.ch1.ome.zarr"

    temp_root = legacy.create_fusion_temp_workspace(channel_output)

    assert temp_root.parent == channel_output.parent
    assert temp_root.is_dir()
    assert temp_root.name.startswith(".fused.ch1.ome.zarr.fusion-")


def test_fusion_resume_uses_only_matching_workspace(tmp_path: Path) -> None:
    channel_output = tmp_path / "fused.ch0.ome.zarr"
    matching = legacy.create_fusion_temp_workspace(channel_output)
    (legacy.fusion_output_from_temp_root(matching, channel_output) / "0").mkdir(parents=True)
    legacy.write_fusion_resume_plan(matching, {"plan": "matching"})
    newer_mismatch = legacy.create_fusion_temp_workspace(channel_output)
    (legacy.fusion_output_from_temp_root(newer_mismatch, channel_output) / "0").mkdir(parents=True)
    legacy.write_fusion_resume_plan(newer_mismatch, {"plan": "different"})

    actual = legacy.find_latest_fusion_temp_workspace(
        channel_output,
        expected_plan={"plan": "matching"},
    )

    assert actual == matching


def test_fusion_resume_rejects_workspace_without_plan(tmp_path: Path) -> None:
    channel_output = tmp_path / "fused.ch0.ome.zarr"
    workspace = legacy.create_fusion_temp_workspace(channel_output)
    (legacy.fusion_output_from_temp_root(workspace, channel_output) / "0").mkdir(parents=True)

    assert (
        legacy.find_latest_fusion_temp_workspace(
            channel_output,
            expected_plan={"plan": "matching"},
        )
        is None
    )


def test_fusion_resume_validates_outer_shards() -> None:
    array = Namespace(
        shape=(24, 960, 960),
        chunks=(1, 960, 960),
        dtype=np.dtype("uint16"),
        metadata=Namespace(shards=(12, 960, 960)),
    )

    legacy.validate_resumable_scale0_array(
        array,
        expected_shape=(24, 960, 960),
        expected_chunks=(12, 960, 960),
        expected_dtype=np.uint16,
        scale0_path=Path("scale0"),
    )


def test_fusion_resume_requires_completion_marker(tmp_path: Path) -> None:
    scale0 = tmp_path / "fused.ome.zarr" / "0"
    scale0.mkdir(parents=True)
    (scale0 / "zarr.json").write_text("{}")
    chunk = scale0 / "c" / "0" / "0" / "0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"partial")
    marker_dir = tmp_path / "completed-fusion-blocks"
    processed = []

    def fuse_chunk(block_id):
        processed.append(tuple(block_id))
        chunk.write_bytes(b"complete")

    resume = legacy.resume_fusion_batch_func(None, scale0_path=scale0, marker_dir=marker_dir)
    resume(fuse_chunk, [(0, 0, 0)])

    assert processed == [(0, 0, 0)]
    assert legacy.fusion_block_marker_path(marker_dir, (0, 0, 0)).is_file()

    resume(fuse_chunk, [(0, 0, 0)])
    assert processed == [(0, 0, 0)]


def test_fusion_resume_rejects_zarr_without_completion_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.ome.zarr"
    (source / "0").mkdir(parents=True)
    (source / "zarr.json").write_text("{}")
    (source / "0" / "zarr.json").write_text("{}")

    with pytest.raises(ValueError, match="missing squisher.complete.json"):
        legacy.source_tile_resume_identity(source, require_completion=True)


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
    assert args[args.index("--batch-size") + 1] == "1"
    assert args[args.index("--basic-cache-tiles") + 1] == "64"
    assert args[args.index("--jpegxr-level") + 1] == "0.7"
    assert args[args.index("--output-codec") + 1] == "jpegxr"
    assert "--per-chunk-cupy-cleanup" not in args
    assert "--resume-fusion" not in args


def test_fuse_passes_resume_fusion(monkeypatch) -> None:
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
        resume_fusion=True,
    )

    assert "--resume-fusion" in captured["args"]


def test_fuse_defaults_downsampled_materialization_to_zstd(monkeypatch, tmp_path) -> None:
    captured = {}
    position = tmp_path / "positions.json"
    position.write_text(json.dumps({"materialization_grid": {"level_factor_zyx": [4, 4, 4]}, "tiles": []}))

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["args"] = args
        return script_name

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=tmp_path / "tiles",
        position_input=position,
        registration_input=tmp_path / "registration.json",
        output=tmp_path / "preview.ome.zarr",
    )

    args = captured["args"]
    assert args[args.index("--output-codec") + 1] == "zstd"


def test_fuse_allows_downsampled_input_on_matching_level0_template(
    monkeypatch, tmp_path
) -> None:
    captured = {}
    position = tmp_path / "positions.json"
    position.write_text(json.dumps({"materialization_grid": {"level_factor_zyx": [4, 4, 4]}, "tiles": []}))

    def fake_run(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
        captured["args"] = args
        return script_name

    monkeypatch.setattr("squisher_lightsheet.fusion.run_legacy_script", fake_run)

    fuse_tiles(
        input_dir=tmp_path / "tiles",
        position_input=position,
        registration_input=tmp_path / "registration.json",
        output=tmp_path / "preview.ome.zarr",
        output_grid_template=tmp_path / "fixed-preview.ome.zarr",
        output_grid_template_level=0,
    )

    args = captured["args"]
    assert args[args.index("--output-codec") + 1] == "zstd"
    assert args[args.index("--output-grid-template-level") + 1] == "0"


@pytest.mark.parametrize("with_fusion_weights", [False, True])
def test_inplace_weighted_average_matches_mvs(with_fusion_weights: bool) -> None:
    from multiview_stitcher.fusion import _core as fusion_core

    transformed = np.asarray(
        [
            [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]],
            [[7.0, 8.0, np.nan], [10.0, np.nan, 12.0]],
        ],
        dtype=np.float32,
    )
    blending = np.asarray(
        [
            [[0.25, 0.0, 1.0], [0.0, 0.5, 0.0]],
            [[0.75, 1.0, 0.0], [1.0, 0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    fusion_weights = (
        np.asarray(
            [
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            ],
            dtype=np.float32,
        )
        if with_fusion_weights
        else None
    )
    transformed_input = transformed.copy()
    blending_input = blending.copy()
    fusion_weights_input = None if fusion_weights is None else fusion_weights.copy()
    expected = fusion_core.weighted_average_fusion(
        transformed.copy(),
        blending.copy(),
        None if fusion_weights is None else fusion_weights.copy(),
    )

    actual = legacy.inplace_weighted_average_fusion(
        transformed_input,
        blending_input,
        fusion_weights,
    )

    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(transformed_input, transformed, equal_nan=True)
    if fusion_weights is None:
        np.testing.assert_array_equal(blending_input, blending)
    else:
        assert not np.array_equal(blending_input, blending, equal_nan=True)
        np.testing.assert_array_equal(fusion_weights, fusion_weights_input)


def test_inplace_weighted_average_excludes_values_at_or_below_threshold() -> None:
    transformed = np.asarray(
        [
            [[25.0, 75.0, 50.0]],
            [[100.0, 40.0, np.nan]],
        ],
        dtype=np.float32,
    )
    blending = np.full(transformed.shape, 0.5, dtype=np.float32)

    actual = legacy.inplace_weighted_average_fusion(
        transformed,
        blending,
        intensity_threshold=50.0,
    )

    np.testing.assert_array_equal(actual, np.asarray([[100.0, 75.0, 0.0]], dtype=np.float32))


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


def test_fuse_passes_level0_output_chunksize_by_default(monkeypatch) -> None:
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
    )

    args = captured["args"]
    assert args[args.index("--output-chunksize") + 1 : args.index("--output-chunksize") + 4] == [
        "12",
        "960",
        "960",
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
        output_grid_template_level=2,
        output_codec="zstd",
        zstd_level=5,
    )

    args = captured["args"]
    assert args[args.index("--output-grid-template") + 1] == "/run/reference.ch0.ome.zarr"
    assert args[args.index("--output-grid-template-level") + 1] == "2"
    assert args[args.index("--output-codec") + 1] == "zstd"
    assert args[args.index("--zstd-level") + 1] == "5"


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
    view_dir_args = [
        args[index + 1] for index, value in enumerate(args) if value == "--flatfield-dir-by-source-view"
    ]
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
    assert args.batch_size == 1
    assert args.output_chunksize == (12, 960, 960)

    monkeypatch.setattr(
        sys,
        "argv",
        ["stitch", str(input_dir), "--flatfield-dir", str(flatfield_dir)],
    )

    args = legacy.parse_args()

    assert args.flatfield_dir == flatfield_dir


def test_legacy_fusion_accepts_hidden_intensity_threshold(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["stitch", str(tmp_path / "tiles"), "--fusion-intensity-threshold", "50"],
    )

    args = legacy.parse_args()

    assert args.fusion_intensity_threshold == 50.0


def test_legacy_fusion_accepts_hidden_skip_pyramid(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", ["stitch", str(tmp_path / "tiles"), "--skip-pyramid"])

    args = legacy.parse_args()

    assert args.skip_pyramid is True


def test_profile_fusion_skip_requires_max_batches() -> None:
    with pytest.raises(
        ValueError, match="--profile-skip-fusion-batches requires --profile-max-fusion-batches"
    ):
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

    def fake_block_reduce_mean_gpu(
        block: np.ndarray, factors: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
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
    level1 = zarr.open_array(str(output / "1"), mode="r")
    assert [codec["name"] for codec in level1.metadata.to_dict()["codecs"]] == ["bytes", "zstd"]


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

    def fake_block_reduce_mean_gpu(
        block: np.ndarray, factors: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
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

    legacy.write_downsampled_level(
        source,
        tmp_path / "pyramid.ome.zarr",
        dataset_path="1",
        dimension_names=("z", "y", "x"),
        factors={"z": 2, "y": 3, "x": 2},
        jpegxr_level=0.8,
        output_codec="jpegxr",
        zstd_level=3,
    )

    destination = zarr.open(str(tmp_path / "pyramid.ome.zarr" / "1"), mode="r")
    assert destination.chunks == (1, 2, 4)
    assert destination.metadata.shards == (4, 4, 8)
    inner_codecs = destination.metadata.to_dict()["codecs"][0]["configuration"]["codecs"]
    assert [codec["name"] for codec in inner_codecs] == ["squisher.jpegxr", "crc32c"]
    assert inner_codecs[0]["configuration"]["level"] == 0.8
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
        group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
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
            ],
        }
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


def test_position_input_maps_source_ome_tiff_name_to_deconvolved_ome_zarr(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    input_dir = tmp_path / "deconv"
    input_dir.mkdir()
    tile_path = input_dir / "sample.000.ome.zarr"
    group = zarr.open_group(str(tile_path), mode="w", zarr_format=3)
    group.create_array(
        "0",
        shape=(2, 3, 6, 7),
        chunks=(1, 1, 6, 7),
        dtype="uint16",
        dimension_names=("c", "z", "y", "x"),
    )
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": axis} for axis in ("c", "z", "y", "x")],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.6, 0.3, 0.3]}],
                    }
                ],
            }
        ],
    }
    position_input = tmp_path / "positions.json"
    position_input.write_text(
        json.dumps(
            {
                "units": "micrometer",
                "tiles": [
                    {
                        "path": "/original/sample.000.ome.tif",
                        "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                        "scale_um": {"z": 0.6, "y": 0.3, "x": 0.3},
                    }
                ],
            }
        )
        + "\n"
    )

    tiles = legacy.read_position_input_tiles(position_input, input_dir=input_dir)

    assert tiles[0].path == tile_path.resolve()
    assert tiles[0].shape == (2, 3, 6, 7)


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


def test_reopenable_tiff_store_is_lazy_and_picklable(tmp_path) -> None:
    from joblib.externals import cloudpickle

    tifffile = pytest.importorskip("tifffile")
    zarr = pytest.importorskip("zarr")
    data = np.arange(4 * 3 * 2, dtype=np.uint16).reshape(4, 3, 2)
    path = tmp_path / "tile.ome.tif"
    tifffile.imwrite(path, data, metadata={"axes": "ZYX"})

    store = cloudpickle.loads(cloudpickle.dumps(tiff_input.ReopenableTiffStore(path)))
    assert store._backend is None
    try:
        np.testing.assert_array_equal(zarr.open(store, mode="r")[:], data)
    finally:
        store.close()


def test_reopenable_tiff_store_lru_bounds_live_backends(tmp_path, monkeypatch) -> None:
    tifffile = pytest.importorskip("tifffile")
    zarr = pytest.importorskip("zarr")
    data = np.arange(4 * 3 * 2, dtype=np.uint16).reshape(4, 3, 2)
    monkeypatch.setattr(tiff_input, "REOPENABLE_TIFF_BACKEND_CACHE_SIZE", 2)
    tiff_input.clear_reopenable_tiff_backend_cache()
    stores = []
    arrays = []
    for index in range(3):
        path = tmp_path / f"tile-{index}.ome.tif"
        tifffile.imwrite(path, data + index, metadata={"axes": "ZYX"})
        store = tiff_input.ReopenableTiffStore(path)
        stores.append(store)
        arrays.append(zarr.open(store, mode="r"))

    try:
        np.testing.assert_array_equal(arrays[0][:], data)
        np.testing.assert_array_equal(arrays[1][:], data + 1)
        assert stores[0]._backend is not None
        assert stores[1]._backend is not None

        np.testing.assert_array_equal(arrays[2][:], data + 2)
        assert stores[0]._backend is None
        assert stores[1]._backend is not None
        assert stores[2]._backend is not None
        assert len(tiff_input._REOPENABLE_TIFF_BACKEND_CACHE) == 2

        np.testing.assert_array_equal(arrays[0][:], data)
        assert stores[0]._backend is not None
        assert stores[1]._backend is None
        assert stores[2]._backend is not None
    finally:
        legacy.close_stores(stores)
        tiff_input.clear_reopenable_tiff_backend_cache()


def test_worker_payload_is_written_once_for_stable_fusion_owner(tmp_path, monkeypatch) -> None:
    from joblib.externals import cloudpickle

    output = tmp_path / "fused.ome.zarr" / "0"
    output.mkdir(parents=True)
    owner = {"images": []}
    payload = {
        "output_zarr_url": str(output),
        "fuse_kwargs": owner,
    }
    calls = 0
    original_dump = cloudpickle.dump

    def record_dump(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_dump(*args, **kwargs)

    monkeypatch.setattr(cloudpickle, "dump", record_dump)
    legacy._MVS_FUSE_CHUNK_PAYLOAD_OWNERS.clear()

    first = legacy.write_mvs_fuse_chunk_payload_cache(payload)
    second = legacy.write_mvs_fuse_chunk_payload_cache(payload)

    assert first == second
    assert calls == 1


def test_worker_payload_excludes_parent_tiff_cache(tmp_path) -> None:
    output = tmp_path / "fused.ome.zarr" / "0"
    output.mkdir(parents=True)
    store = tiff_input.ReopenableTiffStore(tmp_path / "tile.tif")
    tiff_input._REOPENABLE_TIFF_BACKEND_CACHE[id(store)] = store
    payload = {
        "output_zarr_url": str(output),
        "fuse_kwargs": {"images": []},
    }
    legacy._MVS_FUSE_CHUNK_PAYLOAD_OWNERS.clear()

    legacy.write_mvs_fuse_chunk_payload_cache(payload)

    assert not tiff_input._REOPENABLE_TIFF_BACKEND_CACHE


def test_zcyx_tiff_routes_channels_for_registration_and_fusion(tmp_path) -> None:
    tifffile = pytest.importorskip("tifffile")
    from multiview_stitcher.fusion import _core as fusion_core

    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    path = tmp_path / "tile.tif"
    tifffile.imwrite(path, data, metadata={"axes": "ZCYX"})
    tile = legacy.TileMetadata(
        path=path,
        shape=data.shape,
        axes="ZCYX",
        spacing={"z": 0.6, "y": 0.108, "x": 0.108},
        translation={"z": 0.0, "y": 1.0, "x": 2.0},
        channels=("af", "edu"),
        tracks=(
            legacy.TrackMetadata(
                slug="track0",
                track_id="all",
                channels=(0, 1),
                channel_names=("af", "edu"),
            ),
        ),
    )

    registration, registration_store, source_tile = legacy.open_registration_tile_array(
        tile,
        1,
        read_chunk_z=2,
    )
    try:
        assert source_tile.axes == "ZCYX"
        assert registration.shape == (1, 3, 4, 5)
        np.testing.assert_array_equal(registration.compute()[0], data[:, 1])
    finally:
        legacy.close_stores([registration_store])

    sims, fusion_stores, label, _levels, source_tiles = legacy.build_fusion_sims(
        [tile], 1, assume_shared_tiff_schema=True
    )
    try:
        assert label == "edu"
        assert source_tiles[0].axes == "ZCYX"
        assert sims[0].dims == ("z", "y", "x")
        assert si_utils.is_xarray_zarr_backed(sims[0])
        assert fusion_stores[0]._backend is None
        np.testing.assert_array_equal(np.asarray(si_utils._get_backend_data(sims[0])), data[:, 1])

        stack_properties = si_utils.get_stack_properties_from_sim(sims[0])
        chunk_shape = {dim: int(size) for dim, size in stack_properties["shape"].items()}
        entry, _region = legacy.direct_zarr_block_entry(
            sims=sims,
            transform_key=legacy.TRANSFORM_KEY,
            sim_coord_dict={},
            block_key=(0, 0, 0),
            output_stack_properties=stack_properties,
            output_chunksize=chunk_shape,
            output_chunk_shape=chunk_shape,
            output_chunk_origin=stack_properties["origin"],
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
            interpolation_order=1,
            blending_widths=None,
            shrink_distance=0,
            trim_overlap=False,
            backend=None,
        )
        np.testing.assert_array_equal(fused, data[:, 1])
    finally:
        legacy.close_stores(fusion_stores)


def test_tiff_input_reuses_one_schema_and_reads_distinct_files(tmp_path, monkeypatch) -> None:
    tifffile = pytest.importorskip("tifffile")
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    paths = []
    for index in range(2):
        path = tmp_path / f"tile-{index}.tif"
        tifffile.imwrite(path, data + index, metadata={"axes": "ZCYX"})
        paths.append(path)

    calls = []
    original_imread = tifffile.imread

    def record_imread(*args, **kwargs):
        calls.append(Path(args[0]))
        return original_imread(*args, **kwargs)

    monkeypatch.setattr(tifffile, "imread", record_imread)
    handler = tiff_input.TiffInputHandler()
    arrays = []
    stores = []
    for path in paths:
        array, store = handler.open(path)
        arrays.append(array)
        stores.append(store)
    stores[0].release_backend()

    try:
        assert calls == [paths[0]]
        assert all(store._backend is None for store in stores)
        np.testing.assert_array_equal(arrays[0][:], data)
        np.testing.assert_array_equal(arrays[1][:], data + 1)
    finally:
        legacy.close_stores(stores)


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
            fusion_func=legacy.inplace_weighted_average_fusion,
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

        def fake_fuse_block(chunk_params_block, **kwargs):
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
                "fusion_func": legacy.inplace_weighted_average_fusion,
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


def test_loky_fusion_worker_registers_jpegxr_before_opening_output(
    monkeypatch,
) -> None:
    from multiview_stitcher import fusion as _fusion  # noqa: F401
    import zarr

    calls = []

    class StopAfterOpen(RuntimeError):
        pass

    class FakeDevice:
        def __init__(self, _device: int):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_open_array(*_args, **_kwargs):
        calls.append("open")
        raise StopAfterOpen

    monkeypatch.setattr(legacy, "register_jpegxr_codec", lambda: calls.append("register"))
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        type("FakeCupy", (), {"cuda": type("Cuda", (), {"Device": FakeDevice})}),
    )
    monkeypatch.setattr(zarr, "open_array", fake_open_array)

    with pytest.raises(StopAfterOpen):
        legacy.run_mvs_fuse_chunk_loky_worker_once((0, 0, 0), {"output_zarr_url": "fake.zarr"}, device=0)

    assert calls == ["register", "open"]


def test_fusion_main_registers_jpegxr_before_parsing_inputs(monkeypatch) -> None:
    calls = []

    class StopAfterParse(RuntimeError):
        pass

    def fake_parse_args():
        calls.append("parse")
        raise StopAfterParse

    monkeypatch.setattr(legacy, "register_jpegxr_codec", lambda: calls.append("register"))
    monkeypatch.setattr(legacy, "parse_args", fake_parse_args)

    with pytest.raises(StopAfterParse):
        legacy.main()

    assert calls == ["register", "parse"]


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
