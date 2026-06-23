from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from multiview_stitcher import spatial_image_utils as si_utils

from squisher_lightsheet.fusion import (
    channel_output_path,
    channel_output_paths,
    coarse_preibisch_content_weights,
    fuse_tiles,
    temporary_basic_disk_cache_dir,
)
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


def write_basic_sampling_manifest(path: Path, *, sort_intensity: bool = True, input_dirs: list[str] | None = None) -> None:
    path.mkdir(parents=True)
    payload = {
        "sort_intensity": sort_intensity,
        "input_dirs": input_dirs if input_dirs is not None else [str(path.parent)],
    }
    (path / "basic-sampling.json").write_text(
        json.dumps(payload) + "\n"
    )


def test_channel_output_paths_match_legacy_per_channel_names() -> None:
    assert channel_output_path(Path("/run/fused.ome.zarr"), 0) == Path("/run/fused.ch0.ome.zarr")
    assert channel_output_path(Path("/run/fused.zarr"), 1) == Path("/run/fused.ch1.zarr")
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
        args[index + 1]
        for index, value in enumerate(args)
        if value == "--flatfield-dir-by-source-view"
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


def test_legacy_fusion_flatfield_is_opt_in(monkeypatch, tmp_path) -> None:
    input_dir = tmp_path / "tiles"
    flatfield_dir = tmp_path / "basic"
    monkeypatch.setattr(sys, "argv", ["stitch", str(input_dir)])

    args = legacy.parse_args()

    assert args.flatfield_dir is None

    monkeypatch.setattr(
        sys,
        "argv",
        ["stitch", str(input_dir), "--flatfield-dir", str(flatfield_dir)],
    )

    args = legacy.parse_args()

    assert args.flatfield_dir == flatfield_dir


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


def test_downsampled_pyramid_chunks_map_to_source_chunks(monkeypatch, tmp_path) -> None:
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
    assert set(seen_block_shapes) == {(4, 6, 8)}


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
