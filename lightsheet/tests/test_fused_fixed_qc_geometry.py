from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
import zarr

from squisher_lightsheet import fused_fixed_qc as qc


def _source(
    path: Path,
    *,
    moving: bool = False,
    factors: tuple[int, int, int] | None = None,
    level_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    units: str = "micrometer",
) -> zarr.Group:
    root = zarr.open_group(path, mode="w")
    dimensions = ("c", "z", "y", "x") if moving else ("z", "y", "x")
    factors = factors or ((1, 4, 4) if moving else (4, 4, 4))
    spatial_shape = (32, 96, 96)
    native_shape = (2, *spatial_shape) if moving else spatial_shape
    shape = tuple(size // factor for size, factor in zip(spatial_shape, factors, strict=True))
    if moving:
        shape = (2, *shape)
    root.create_array("0", shape=native_shape, chunks=(1,) * len(native_shape), dtype="uint16",
                      dimension_names=dimensions)
    root.create_array("2", shape=shape, chunks=shape, dtype="uint16", dimension_names=dimensions)
    scale = [0.7, 0.3, 0.3]
    origin = [19.0, -11.0, 23.0]
    datasets = []
    for name, factor, offset in (("0", (1, 1, 1), (0.0, 0.0, 0.0)), ("2", factors, level_offset)):
        level_scale = [spacing * stride for spacing, stride in zip(scale, factor, strict=True)]
        translation = [first + second for first, second in zip(origin, offset, strict=True)]
        if moving:
            level_scale = [1.0, *level_scale]
            translation = [0.0, *translation]
        datasets.append({"path": name, "coordinateTransformations": [
            {"type": "scale", "scale": level_scale},
            {"type": "translation", "translation": translation},
        ]})
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{
        "axes": [{"name": axis, "type": "channel"} if axis == "c" else
                 {"name": axis, "type": "space", "unit": units} for axis in dimensions],
        "datasets": datasets,
    }]}
    return root


@pytest.mark.parametrize("factors,offset,units", [
    ((2, 4, 4), (0.0, 0.0, 0.0), "micrometer"),
    ((4, 4, 4), (0.7, 0.0, 0.0), "micrometer"),
    ((4, 4, 4), (0.0, 0.0, 0.0), "nanometer"),
])
def test_scorer_rejects_unsupported_fixed_sampling(
    tmp_path: Path, factors, offset, units: str
) -> None:
    path = tmp_path / "fixed.zarr"
    _source(path, factors=factors, level_offset=offset, units=units)
    with pytest.raises(ValueError, match="sampling"):
        qc.WindowImageScorer(path, 100, "linear")


@pytest.mark.parametrize("factors,offset", [
    ((4, 4, 4), (0.0, 0.0, 0.0)),
    ((1, 4, 4), (0.0, 0.3, 0.0)),
])
def test_sampler_rejects_unsupported_moving_sampling(tmp_path: Path, factors, offset) -> None:
    path = tmp_path / "moving.zarr"
    _source(path, moving=True, factors=factors, level_offset=offset)
    with pytest.raises(ValueError, match="sampling"):
        qc._sample_moving_plane(
            row={"moving_path": str(path), "moving_channel": 0,
                 "moving_start_l0_zyx": [0, 0, 0], "fixed_start_zyx": [0, 0, 0]},
            matrix=np.eye(3), translation=np.zeros(3), fixed_z_l0=4,
            bounds=(0, 16, 0, 16), arrays={},
        )


@pytest.mark.parametrize("channel", [None, True, 1.5, -1, 2])
def test_sampler_requires_explicit_valid_acquisition_channel(tmp_path: Path, channel) -> None:
    path = tmp_path / "moving.zarr"
    _source(path, moving=True)
    row = {"moving_path": str(path), "moving_start_l0_zyx": [0, 0, 0], "fixed_start_zyx": [0, 0, 0]}
    if channel is not None:
        row["moving_channel"] = channel
    with pytest.raises(ValueError, match="channel"):
        qc._sample_moving_plane(
            row=row, matrix=np.eye(3), translation=np.zeros(3), fixed_z_l0=4,
            bounds=(0, 16, 0, 16), arrays={},
        )


def test_scorer_rejects_missing_physical_metadata(tmp_path: Path) -> None:
    path = tmp_path / "fixed.zarr"
    root = _source(path)
    del root.attrs["ome"]
    with pytest.raises(ValueError, match="sampling"):
        qc.WindowImageScorer(path, 100, "linear")


def test_scorer_rejects_array_axes_that_disagree_with_the_sampling_contract(tmp_path: Path) -> None:
    path = tmp_path / "fixed.zarr"
    root = _source(path)
    root["2"].attrs["_ARRAY_DIMENSIONS"] = ["y", "z", "x"]
    with pytest.raises(ValueError, match="sampling requires ZYX"):
        qc.WindowImageScorer(path, 100, "linear")


def test_sampler_rejects_changed_pyramid_channel_count(tmp_path: Path) -> None:
    path = tmp_path / "moving.zarr"
    root = _source(path, moving=True)
    root["2"].resize((1, *root["2"].shape[1:]))
    with pytest.raises(ValueError, match="channel count"):
        qc._sample_moving_plane(
            row={"moving_path": str(path), "moving_channel": 0,
                 "moving_start_l0_zyx": [0, 0, 0], "fixed_start_zyx": [0, 0, 0]},
            matrix=np.eye(3), translation=np.zeros(3), fixed_z_l0=4,
            bounds=(0, 16, 0, 16), arrays={},
        )


@pytest.mark.parametrize("surface", ["prefusion", "interpolated"])
def test_renderers_reject_the_same_unsupported_fixed_sampling(tmp_path: Path, surface: str) -> None:
    path = tmp_path / "fixed.zarr"
    fixed = _source(path, factors=(2, 4, 4))
    fixed["2"][:] = 200
    moving_path = tmp_path / "moving.zarr"
    _source(moving_path, moving=True)
    window = tmp_path / "window.json"
    attempt = "interpolated_method6_spatial_recovery"
    window.write_text(json.dumps({
        "moving_tile": "moving", "moving_path": str(moving_path), "moving_channel": 0,
        "moving_start_l0_zyx": [0, 0, 0], "fixed_start_zyx": [0, 0, 0],
        "selected_local_matrix_zyx": np.eye(3).tolist(), "selected_local_translation_zyx": [0, 0, 0],
        "selected_attempt": attempt,
        "interpolated_fallback_promotion": {"source_good_window_count": 2},
    }))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "fixed_fused": str(path), "window_shape_zyx": [128, 528, 528],
        "cache_config": {"native_method": "method6"},
        "windows": [{"level0_json": str(window), "selected_attempt": attempt}],
    }))
    with pytest.raises(ValueError, match="sampling"):
        if surface == "prefusion":
            qc.render_prefusion_qc(summaries=[("405", summary)], fixed_volume=path,
                                  fixed_threshold=100, output_dir=tmp_path / "qc")
        else:
            qc.render_interpolated_window_qc(summary_path=summary, output=tmp_path / "qc.png",
                                             moving_label="405", fixed_label="561")


def test_sampler_accepts_nonzero_independent_origins_and_nonzero_channel(tmp_path: Path) -> None:
    _source(tmp_path / "fixed.zarr")
    moving = _source(tmp_path / "moving.zarr", moving=True)
    metadata = dict(moving.attrs["ome"])
    for dataset in metadata["multiscales"][0]["datasets"]:
        dataset["coordinateTransformations"][1]["translation"][1:] = [-400.0, 800.0, 9.0]
    moving.attrs["ome"] = metadata
    moving["2"][0] = 7
    moving["2"][1] = 31
    qc.WindowImageScorer(tmp_path / "fixed.zarr", 100, "linear")
    sample, valid = qc._sample_moving_plane(
        row={"moving_path": str(tmp_path / "moving.zarr"), "moving_channel": 1,
             "moving_start_l0_zyx": [0, 0, 0], "fixed_start_zyx": [0, 0, 0]},
        matrix=np.eye(3), translation=np.zeros(3), fixed_z_l0=4,
        bounds=(0, 16, 0, 16), arrays={},
    )
    assert valid.all()
    assert np.array_equal(sample, np.full((16, 16), 31, dtype="float32"))
