from __future__ import annotations

from pathlib import Path

import numpy as np

from squisher_lightsheet.fusion import (
    channel_output_path,
    channel_output_paths,
    coarse_preibisch_content_weights,
    fuse_tiles,
    temporary_basic_disk_cache_dir,
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


def test_fuse_passes_source_view_flatfield_dirs(monkeypatch) -> None:
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
        flatfield_dirs_by_source_view={
            "L": Path("/run/basic-left"),
            "R": Path("/run/basic-right"),
        },
    )

    assert captured["script_name"] == "stitch_20x_tl_multiview.py"
    args = captured["args"]
    view_dir_args = [
        args[index + 1]
        for index, value in enumerate(args)
        if value == "--flatfield-dir-by-source-view"
    ]
    assert view_dir_args == ["L=/run/basic-left", "R=/run/basic-right"]


def test_temporary_basic_disk_cache_dir_uses_output_parent_and_cleans_up(tmp_path) -> None:
    output = tmp_path / "fused.ch0.ome.zarr"

    with temporary_basic_disk_cache_dir(None, output) as cache_dir:
        assert cache_dir.parent == tmp_path
        assert cache_dir.exists()
        (cache_dir / "sentinel").write_text("cache")

    assert not cache_dir.exists()


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
