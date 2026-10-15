import numpy as np
from squisher_lightsheet._backends import fused_fixed_trial as trial


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



def test_affine_placement_recovery_seed_reaches_native_crop_in_canonical_coordinates(tmp_path, monkeypatch) -> None:
    import json
    from types import SimpleNamespace
    import pytest
    seed = {"artifact_type": "lightsheet.fused_fixed_native_spatial_recovery_initializer.v1",
            "level_factor_zyx": [1, 1, 1], "local_matrix_zyx": [[1, .003, 0], [0, 1, .002], [0, 0, 1]],
            "local_translation_zyx": [2, -3, 4]}
    initializer = tmp_path / "seed.json"
    initializer.write_text(json.dumps(seed))
    root = {"0": SimpleNamespace(shape=(100, 100, 100))}
    monkeypatch.setattr(trial, "_fixed_fused_root", lambda _path: root)
    monkeypatch.setattr(trial, "_metadata_fixed_bounds", lambda **kwargs:
        (np.array([4, 5, 6]), np.array([9, 10, 11]), np.ones(3), np.ones(3), np.zeros(3)))
    resolved = []
    original_resolver = trial._resolve_starting_affine_matrix
    def resolve(configured, matrix, *, already_applied):
        result = original_resolver(configured, matrix, already_applied=already_applied)
        resolved.append(result[0])
        return result
    monkeypatch.setattr(trial, "_resolve_starting_affine_matrix", resolve)
    class NativeCropReached(Exception):
        pass

    def reached(*args):
        raise NativeCropReached()
    monkeypatch.setattr(trial, "_crop_padding_metadata", reached)
    args = SimpleNamespace(level2_result=initializer, fixed_fused=tmp_path / "fixed.ome.zarr",
                           moving_start_zyx=[0, 0, 0], shape_zyx=[5, 5, 5],
                           level0_initializer="window-interpolated",
                           starting_affine_matrix_zyx=tuple(np.eye(3).ravel()))
    record = _record("moving.000.ome.zarr", (0, 0, 0), (1, 1, 1))
    record["placement_affine_zyx_um"] = np.eye(4).tolist()
    with pytest.raises(NativeCropReached):
        trial._run_level0_from_level2(args, record, SimpleNamespace())
    np.testing.assert_allclose(resolved[0], seed["local_matrix_zyx"])


def test_parent_affine_reaches_child_fit_without_starting_matrix_override(tmp_path, monkeypatch) -> None:
    import json
    from types import SimpleNamespace
    import pytest

    matrix = np.array([[1.002, .003, 0], [-.001, 1, .002], [0, .001, .999]])
    parent = {"artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
              "selected_local_matrix_zyx": matrix.tolist(), "selected_local_translation_zyx": [2, -3, 4],
              "fixed_start_zyx": [4, 5, 6], "moving_start_l0_zyx": [0, 0, 0],
              "moving_shape_zyx": [528, 528, 528]}
    initializer = tmp_path / "parent.json"
    initializer.write_text(json.dumps(parent))
    monkeypatch.setattr(trial, "_fixed_fused_root", lambda _: {"0": SimpleNamespace(shape=(1000, 1000, 1000))})
    monkeypatch.setattr(trial, "_metadata_fixed_bounds", lambda **kwargs:
                        (np.array([84, 5, 6]), np.array([212, 533, 534]), np.ones(3), np.ones(3), np.zeros(3)))
    resolved = []
    resolver = trial._resolve_starting_affine_matrix

    def capture(configured, value, *, already_applied):
        result = resolver(configured, value, already_applied=already_applied)
        resolved.append(result[0])
        return result

    monkeypatch.setattr(trial, "_resolve_starting_affine_matrix", capture)

    class NativeCropReached(Exception):
        pass

    def stop(*args):
        raise NativeCropReached()

    monkeypatch.setattr(trial, "_crop_padding_metadata", stop)
    args = SimpleNamespace(level2_result=initializer, fixed_fused=tmp_path / "fixed.zarr",
                           moving_start_zyx=[80, 0, 0], shape_zyx=[128, 528, 528],
                           level0_initializer="window-parent", starting_affine_matrix_zyx=tuple(np.eye(3).ravel()))
    record = _record("moving.000.ome.zarr", (0, 0, 0), (1, 1, 1))
    with pytest.raises(NativeCropReached):
        trial._run_level0_from_level2(args, record, SimpleNamespace())
    np.testing.assert_allclose(resolved[0], matrix, atol=1e-7)
