import importlib.util
from pathlib import Path

import numpy as np


def _load_trial():
    path = Path("/home/chaichontat/nvme/lightsheet/scripts/try_fused_fixed_method8.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    tile: str, translation_zyx: tuple[float, float, float], scale_zyx: tuple[float, float, float]
) -> dict[str, object]:
    return {
        "tile": tile,
        "path": f"/data/{tile}",
        "shape": [100, 100, 100],
        "axes": "ZYX",
        "translation_um": dict(zip("zyx", translation_zyx, strict=True)),
        "scale_um": dict(zip("zyx", scale_zyx, strict=True)),
    }


def test_metadata_fixed_start_places_flipped_moving_tile_by_oriented_origin() -> None:
    trial = _load_trial()
    fixed = _record("fixed.000.ome.zarr", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    # Index zero of a negative axis is its high bound; the oriented box starts at (10, 20, 30).
    moving = _record("moving.000.ome.zarr", (110.0, 20.0, 130.0), (-1.0, 1.0, -1.0))

    start = trial._metadata_fixed_start_level_zyx(
        fixed_record=fixed,
        moving_record=moving,
        moving_start_level_zyx=np.asarray([1, 2, 3]),
        fixed_factor_zyx=np.ones(3),
        moving_factor_zyx=np.ones(3),
    )

    np.testing.assert_array_equal(start, [11, 22, 33])


def test_metadata_fixed_start_uses_oriented_origin_of_flipped_fixed_tile() -> None:
    trial = _load_trial()
    fixed = _record("fixed.000.ome.zarr", (100.0, 0.0, 100.0), (-1.0, 1.0, -1.0))
    moving = _record("moving.000.ome.zarr", (10.0, 20.0, 30.0), (1.0, 1.0, 1.0))

    start = trial._metadata_fixed_start_level_zyx(
        fixed_record=fixed,
        moving_record=moving,
        moving_start_level_zyx=np.zeros(3, dtype=np.int64),
        fixed_factor_zyx=np.ones(3),
        moving_factor_zyx=np.ones(3),
    )

    np.testing.assert_array_equal(start, [10, 20, 30])


def test_recovery_window_displacement_uses_oriented_origin_for_flipped_moving_tile() -> None:
    from squisher_lightsheet import fused_fixed_recovery

    moving = _record("moving.000.ome.zarr", (110.0, 20.0, 130.0), (-1.0, 1.0, -1.0))
    row = {
        "moving_start_l0_zyx": [1, 2, 3],
        "moving_stop_l0_zyx": [6, 9, 12],
        "fixed_start_zyx": [4, 6, 8],
        "selected_local_translation_zyx": [0.5, -1.0, 2.0],
    }

    point, displacement = fused_fixed_recovery._window_physical_displacement(
        row,
        moving,
        fixed_scale_um_zyx=np.ones(3),
        fixed_translation_um_zyx=np.zeros(3),
    )

    np.testing.assert_allclose(point, [3.0, 5.0, 7.0])
    np.testing.assert_allclose(displacement, [6.5 - 13.0, 8.0 - 25.0, 14.0 - 37.0])
