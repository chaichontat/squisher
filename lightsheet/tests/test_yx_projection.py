from dataclasses import replace
from pathlib import Path

import numpy as np
import tifffile

from squisher_lightsheet import seams
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.yx_projection import (
    _constraint_from_payload,
    _constraint_payload,
    _measure_constraint,
    build_yx_projections,
)


def _tile(path: Path, shape: tuple[int, ...], axes: str) -> legacy.TileMetadata:
    track = legacy.TrackMetadata(
        slug="track0",
        track_id="all",
        channels=(0, 1, 2, 3),
        channel_names=("af", "edu", "polyA", "reddot"),
    )
    return legacy.TileMetadata(
        path=path,
        shape=shape,
        axes=axes,
        spacing={"z": 0.6, "y": 0.108, "x": 0.108},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        channels=track.channel_names,
        tracks=(track,),
    )


def test_build_yx_projections_selects_reddot_and_maxes_z(tmp_path: Path) -> None:
    source = tmp_path / "reg-0001.tif"
    values = np.zeros((3, 4, 16, 12), dtype=np.uint16)
    values[0, 3] = 5
    values[1, 3, 4:8, 3:7] = 12
    values[2, 1] = 1000
    tifffile.imwrite(source, values, metadata={"axes": "ZCYX"})

    outputs = build_yx_projections(
        [_tile(source, values.shape, "ZCYX")],
        channel=3,
        output_dir=tmp_path / "mip",
        workers=1,
        jpegxr_level=1.0,
    )

    actual = tifffile.imread(outputs[0])
    expected = np.max(values[:, 3], axis=0)
    assert np.array_equal(actual, expected)
    with tifffile.TiffFile(outputs[0]) as tif:
        assert int(tif.pages[0].compression) == 22610


def test_constraint_payload_round_trips_slices() -> None:
    constraint = seams.BoundaryConstraint(
        fixed=0,
        moving=1,
        pair=(0, 1),
        axis="x",
        patch_index=0,
        shift_zyx=(0.0, -2.5, 3.0),
        weight=1.0,
        correlation_before=0.1,
        correlation_after=0.9,
        improvement=0.8,
        fixed_nonzero_fraction=1.0,
        moving_nonzero_fraction=1.0,
        fixed_std=1.0,
        moving_std=1.0,
        accepted=True,
        fixed_slices=(slice(0, 1), slice(2, 10), slice(3, 11)),
        moving_slices=(slice(0, 1), slice(4, 12), slice(5, 13)),
    )

    assert _constraint_from_payload(_constraint_payload(constraint)) == constraint


def test_measure_constraint_fixes_z_for_yx_projection(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    fixed = rng.integers(1, 2000, size=(256, 256), dtype=np.uint16)
    moving = np.roll(fixed, shift=(5, -7), axis=(0, 1))
    fixed_path = tmp_path / "fixed.tif"
    moving_path = tmp_path / "moving.tif"
    tifffile.imwrite(fixed_path, fixed)
    tifffile.imwrite(moving_path, moving)
    spec = seams.BoundaryPatchSpec(
        pair=(0, 1),
        axis="x",
        patch_index=0,
        fixed_slices=(slice(0, 1), slice(0, 256), slice(0, 256)),
        moving_slices=(slice(0, 1), slice(0, 256), slice(0, 256)),
        overlap_start_zyx=(0, 0, 0),
        overlap_shape_zyx=(1, 256, 256),
    )
    settings = replace(
        seams.RobustBoundarySettings(),
        min_center_z_p99=0.0,
        min_center_z_std=0.0,
        min_nonzero_fraction=0.0,
        min_content_voxels=256,
        min_gradient_component_ncc=-1.0,
        min_gradient_component_ncc_improvement=-1.0,
        min_improvement=0.0,
    )

    result = _measure_constraint(
        spec=spec,
        projections=[fixed_path, moving_path],
        settings=settings,
    )

    assert result.shift_zyx[0] == 0.0
    assert np.allclose(result.shift_zyx[1:], (-5.0, 7.0), atol=0.2)
