from __future__ import annotations

import numpy as np
from pathlib import Path

from squisher_lightsheet.positions import TileInfo, compute_joined_tiles


def tile(name: str, side: str, translation: tuple[float, float, float], shape=(100, 100, 100)) -> TileInfo:
    return TileInfo(
        tile=name,
        side=side,
        path=Path(f"/tmp/{name}"),
        shape_zyx=shape,
        spacing_zyx=(1.0, 1.0, 1.0),
        translation_zyx=translation,
    )


def test_tltr_x_join_has_no_flips_and_aligns_centroids_in_yz() -> None:
    left = [tile("L.ome.tif", "L", (0.0, 0.0, 0.0))]
    right = [tile("R.ome.tif", "R", (10.0, 200.0, 0.0))]

    joined, diagnostics = compute_joined_tiles(left, right, mode="tltr_x_join_center_z_phase")

    right_joined = next(item for item in joined if item.info.side == "R")
    assert diagnostics["join_axis"] == "x"
    assert diagnostics["right_flip_axes"] == []
    assert diagnostics["centroid_alignment_axes"] == ["z", "y"]
    assert right_joined.scale_zyx == (1.0, 1.0, 1.0)
    np.testing.assert_allclose(
        diagnostics["right_joined_alignment_centroid_um"],
        diagnostics["left_alignment_centroid_um"],
    )
    assert diagnostics["x_overlap_um"] == 25.0


def test_lr_z_endview_flips_xz_and_aligns_centroids_in_xy() -> None:
    left = [tile("L.ome.tif", "L", (0.0, 0.0, 0.0))]
    right = [tile("R.ome.tif", "R", (0.0, 50.0, 25.0))]

    joined, diagnostics = compute_joined_tiles(left, right, mode="lr_z_endview_flip_xz")

    right_joined = next(item for item in joined if item.info.side == "R")
    assert diagnostics["join_axis"] == "z"
    assert diagnostics["right_flip_axes"] == ["z", "x"]
    assert diagnostics["centroid_alignment_axes"] == ["y", "x"]
    assert right_joined.scale_zyx == (-1.0, 1.0, -1.0)
    np.testing.assert_allclose(
        diagnostics["right_joined_alignment_centroid_um"],
        diagnostics["left_alignment_centroid_um"],
    )
    assert diagnostics["z_overlap_um"] == 25.0


def test_overlap_fraction_only_changes_join_axis_for_x_join() -> None:
    left = [tile("L.ome.tif", "L", (0.0, 0.0, 0.0))]
    right = [tile("R.ome.tif", "R", (10.0, 200.0, 0.0))]

    joined_10, _ = compute_joined_tiles(
        left,
        right,
        mode="tltr_x_join_center_z_phase",
        overlap_fraction=0.10,
    )
    joined_25, _ = compute_joined_tiles(
        left,
        right,
        mode="tltr_x_join_center_z_phase",
        overlap_fraction=0.25,
    )

    right_10 = next(item for item in joined_10 if item.info.side == "R")
    right_25 = next(item for item in joined_25 if item.info.side == "R")
    assert right_10.translation_zyx[:2] == right_25.translation_zyx[:2]
    assert right_10.translation_zyx[2] != right_25.translation_zyx[2]
