from __future__ import annotations

from pathlib import Path

from squisher_lightsheet.fusion import fuse_tiles
from squisher_lightsheet.fusion import channel_output_path, channel_output_paths


def test_channel_output_paths_match_legacy_per_channel_names() -> None:
    assert channel_output_path(Path("/run/fused.ome.zarr"), 0) == Path("/run/fused.ch0.ome.zarr")
    assert channel_output_path(Path("/run/fused.zarr"), 1) == Path("/run/fused.ch1.zarr")
    assert channel_output_paths(Path("/run/fused.ome.zarr"), [0, 2]) == [
        Path("/run/fused.ch0.ome.zarr"),
        Path("/run/fused.ch2.ome.zarr"),
    ]


def test_fuse_uses_content_preibisch_coarse_defaults(monkeypatch) -> None:
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
    assert args[args.index("--content-preibisch-sigma1") + 1] == "7"
    assert args[args.index("--content-preibisch-sigma2") + 1] == "17"
    stride_index = args.index("--content-preibisch-coarse-stride")
    assert args[stride_index + 1 : stride_index + 4] == ["1", "8", "8"]
    assert args[args.index("--basic-cache-tiles") + 1] == "128"
