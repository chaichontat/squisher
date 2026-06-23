from __future__ import annotations

from squisher_lightsheet import pyramid


def test_chunk_slices_cover_shape_with_partial_edges() -> None:
    slices = list(pyramid.chunk_slices((5, 7), (2, 3)))

    assert len(slices) == pyramid.chunk_count((5, 7), (2, 3))
    assert slices[0] == (slice(0, 2), slice(0, 3))
    assert slices[-1] == (slice(4, 5), slice(6, 7))


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
