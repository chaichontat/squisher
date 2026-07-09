from __future__ import annotations

import numpy as np
import pytest

from squisher_lightsheet import pyramid
from squisher_lightsheet._legacy import add_ome_zarr_pyramid


def test_chunk_slices_cover_shape_with_partial_edges() -> None:
    slices = list(pyramid.chunk_slices((5, 7), (2, 3)))

    assert len(slices) == pyramid.chunk_count((5, 7), (2, 3))
    assert slices[0] == (slice(0, 2), slice(0, 3))
    assert slices[-1] == (slice(4, 5), slice(6, 7))


def test_downsampled_chunks_scale_source_chunks_by_factor() -> None:
    assert pyramid.downsampled_chunks((4, 6, 8), (4, 4, 8), (2, 3, 2)) == (2, 2, 4)


def test_pyramid_shard_chunks_preserve_storage_chunks_with_shape_cap() -> None:
    assert pyramid.pyramid_shard_chunks((12, 960, 960), (3, 120, 80), (1, 60, 20)) == (3, 120, 80)
    assert pyramid.pyramid_shard_chunks((12, 960, 960), (32, 2000, 1200), (1, 60, 60)) == (12, 960, 960)
    assert pyramid.pyramid_shard_chunks((12, 960, 960), (198, 662, 549), (1, 60, 60)) == (12, 660, 540)


def test_standalone_pyramid_level_uses_shards_with_downsampled_inner_chunks(monkeypatch, tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "pyramid.ome.zarr"
    source = zarr.open(
        str(root / "0"),
        mode="w",
        shape=(8, 12, 16),
        chunks=(4, 6, 8),
        dtype="uint16",
        zarr_format=3,
        dimension_names=("z", "y", "x"),
    )
    source[:] = np.arange(8 * 12 * 16, dtype=np.uint16).reshape(8, 12, 16)
    monkeypatch.setattr(
        add_ome_zarr_pyramid,
        "reduced_chunk",
        lambda source, selection, factors: np.zeros(
            tuple(part.stop - part.start for part in selection),
            dtype=source.dtype,
        ),
    )

    add_ome_zarr_pyramid.write_pyramid_level(
        root,
        source_path="0",
        destination_path="1",
        dims=("z", "y", "x"),
        factors={"z": 2, "y": 3, "x": 2},
    )

    destination = zarr.open(str(root / "1"), mode="r")
    assert destination.chunks == (2, 2, 4)
    assert destination.metadata.shards == (4, 4, 8)


def test_pyramid_relative_factors_downsample_only_large_spatial_axes() -> None:
    assert pyramid.pyramid_relative_factors((1, 220, 202, 80), ("c", "z", "y", "x")) == {
        "c": 1,
        "z": 2,
        "y": 2,
        "x": 1,
    }


def test_level_coordinate_transformations_scale_spatial_axes_without_mutating_input() -> None:
    base = [
        {"type": "scale", "scale": [1.0, 2.0, 3.0]},
        {"type": "translation", "translation": [4.0, 5.0, 6.0]},
    ]
    axes = [{"name": "z"}, {"name": "y"}, {"name": "x"}]

    transformed = pyramid.level_coordinate_transformations(
        base,
        axes,
        {"z": 2, "y": 4, "x": 8},
    )

    assert transformed == [
        {"type": "scale", "scale": [2.0, 8.0, 24.0]},
        {"type": "translation", "translation": [4.0, 5.0, 6.0]},
    ]
    assert base[0]["scale"] == [1.0, 2.0, 3.0]
