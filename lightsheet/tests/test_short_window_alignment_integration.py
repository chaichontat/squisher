"""Numerical alignment contracts through source materialization and CPU fusion."""

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pytest
from multiview_stitcher import param_utils, spatial_image_utils as si_utils
from multiview_stitcher.fusion import _core as fusion_core

from squisher_lightsheet._backends.fused_fixed_sweep import _parent_window_path
from squisher_lightsheet._backends.fused_fixed_trial import (
    _global_index_affine_to_local_model,
    _local_model_to_global_index_affine,
)
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.local_affine import affine_from_local_shifts
from squisher_lightsheet.residual_correction import ResidualCorrection
from squisher_lightsheet.tile_quadrant_fusion import (
    _fused_fixed_registered_affine_um,
    _materialize_downsampled_channel_crop_ome_zarr,
)


@pytest.mark.parametrize("flipped", [False, True])
def test_tail_child_local_fit_materialization_and_cpu_fusion_pixels(tmp_path: Path, flipped: bool) -> None:
    import zarr

    source_shape = (20, 12, 14)
    child_shape = (8, 10, 12)
    child_start = np.array([12, 1, 2])
    parent_start = np.array([0, 1, 2])
    parent_fixed = np.array([30, 40, 50])
    child_fixed = np.array([42, 40, 50])
    parent_shape = np.array([20, 10, 12])
    parent_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.125], [0.0, 0.0, 1.0]])
    parent_translation = np.array([0.0, 1.0, -0.5])
    source = tmp_path / "source.ome.zarr"
    fixed = tmp_path / "fixed.ome.zarr"
    parent = tmp_path / "source.qy0_qx0.z00000.json"
    parent.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
                "status": "accepted",
                "moving_tile": source.name,
                "moving_path": str(source),
                "fixed_fused": str(fixed),
                "moving_channel": 0,
                "moving_start_l0_zyx": parent_start.tolist(),
                "moving_stop_l0_zyx": (parent_start + parent_shape).tolist(),
                "moving_shape_zyx": parent_shape.tolist(),
                "fixed_start_zyx": parent_fixed.tolist(),
                "selected_local_matrix_zyx": parent_matrix.tolist(),
                "selected_local_translation_zyx": parent_translation.tolist(),
                "local_matrix_zyx": np.eye(3).tolist(),
                "local_translation_zyx": [99.0, 99.0, 99.0],
            }
        )
    )
    selected = _parent_window_path(
        tmp_path,
        tile_name=source.name,
        quadrant="qy0_qx0",
        start_zyx=child_start.tolist(),
        shape_zyx=list(child_shape),
        fixed_fused=fixed,
        moving_path=source,
        moving_channel=0,
    )
    assert selected == parent
    row = json.loads(selected.read_text())
    global_matrix, global_offset = _local_model_to_global_index_affine(
        matrix=np.array(row["selected_local_matrix_zyx"]),
        translation=np.array(row["selected_local_translation_zyx"]),
        fixed_start_zyx=parent_fixed,
        moving_start_zyx=parent_start,
        crop_shape_zyx=parent_shape,
    )
    initial_matrix, initial_translation = _global_index_affine_to_local_model(
        matrix=global_matrix,
        offset=global_offset,
        fixed_start_zyx=child_fixed,
        moving_start_zyx=child_start,
        crop_shape_zyx=np.array(child_shape),
    )
    source_points = np.array(
        [[1.0, 2.0, 2.0], [6.0, 2.0, 2.0], [1.0, 7.0, 2.0], [1.0, 2.0, 9.0], [5.0, 6.0, 8.0]]
    )
    z, y, x = source_points.T
    truth = np.column_stack((z, y + 0.125 * x + 1.0, x + 0.25 * z + 0.5))
    center = (np.array(child_shape) - 1) / 2
    initial_targets = (source_points - center) @ initial_matrix.T + center + initial_translation
    # Parent coordinates give the same points independently of the centered child model.
    np.testing.assert_allclose(initial_targets[:, 0], z, atol=1e-6)
    np.testing.assert_allclose(initial_targets[:, 1], y + 0.125 * (x - 5.5) + 1.0, atol=1e-6)
    np.testing.assert_allclose(initial_targets[:, 2], x - 0.5, atol=1e-6)
    matrix, translation = affine_from_local_shifts(
        matrix_zyx=initial_matrix,
        translation_zyx=initial_translation,
        shape_zyx=child_shape,
        centers_zyx=truth,
        shifts_zyx=truth - initial_targets,
    )
    fitted_targets = (source_points - center) @ matrix.T + center + translation
    np.testing.assert_allclose(fitted_targets, truth, atol=1e-5)

    zz, yy, xx = np.indices(source_shape)
    raw = (1000 + 100 * zz + 10 * yy + xx).astype(np.uint16)
    root = zarr.open_group(source, mode="w", zarr_format=3)
    root.create_array("0", data=raw[None], chunks=(1, 4, 12, 14), dimension_names=("c", "z", "y", "x"))
    root.attrs["multiscales"] = [
        {
            "datasets": [{"path": "0"}],
            "axes": [{"name": a, "type": "channel" if a == "c" else "space"} for a in "czyx"],
        }
    ]
    moving_spacing = np.array([1.7, 0.8, 0.45])
    fixed_spacing = np.array([0.6, 0.3, 0.2])
    crop = tmp_path / "child.ome.zarr"
    correction = ResidualCorrection(
        coefficient=np.zeros((1, 5)),
        shape_zyx=source_shape,
        multiplier=2.0,
        fingerprint="constant-gain-two",
    )
    shape = _materialize_downsampled_channel_crop_ome_zarr(
        source_path=source,
        output_path=crop,
        source_record={
            "tile": source.name,
            "axes": "CZYX",
            "translation_um": {"z": 700.0, "y": -900.0, "x": 1100.0},
            "scale_um": dict(
                zip("zyx", moving_spacing * ([-1, 1, -1] if flipped else [1, 1, 1]), strict=True)
            ),
        },
        source_channel=0,
        start_zyx=child_start,
        stop_zyx=child_start + child_shape,
        level_factor_zyx=np.ones(3, dtype=int),
        spacing_zyx=moving_spacing,
        output_codec="zstd",
        residual_correction=correction,
    )
    assert shape == list(child_shape)
    oriented = raw[::-1, :, ::-1] if flipped else raw
    np.testing.assert_array_equal(zarr.open_group(crop, mode="r")["0"][:], 2 * oriented[12:20, 1:11, 2:14])
    placement, stage_origin = _fused_fixed_registered_affine_um(
        {
            "fixed_start_zyx": child_fixed.tolist(),
            "moving_shape_zyx": list(child_shape),
            "fused_scale_zyx": fixed_spacing.tolist(),
            "fused_translation_zyx": [-5.0, 7.0, 11.0],
            "selected_local_matrix_zyx": matrix.tolist(),
            "selected_local_translation_zyx": translation.tolist(),
        },
        moving_scale_zyx=moving_spacing,
    )
    expected_origin = np.array([-5.0, 7.0, 11.0]) + child_fixed * fixed_spacing
    np.testing.assert_allclose(stage_origin, expected_origin)
    placed_points = (stage_origin + source_points * moving_spacing) @ placement[:3, :3].T + placement[:3, 3]
    np.testing.assert_allclose(placed_points, expected_origin + truth * fixed_spacing, atol=1e-5)

    tile = legacy.TileMetadata(
        path=crop,
        shape=child_shape,
        axes="ZYX",
        spacing=dict(zip("zyx", moving_spacing, strict=True)),
        translation=dict(zip("zyx", stage_origin, strict=True)),
        channels=("0",),
        tracks=(legacy.TrackMetadata(slug="track0", track_id="all", channels=(0,), channel_names=("0",)),),
    )
    sims, stores, *_ = legacy.build_fusion_sims([tile], 0)
    try:
        sim = sims[0]
        si_utils.set_sim_affine(
            sim,
            xaffine=param_utils.affine_to_xaffine(placement),
            transform_key=legacy.REGISTERED_TRANSFORM_KEY,
            base_transform_key=legacy.TRANSFORM_KEY,
        )
        output_shape = dict(zip("zyx", [4, 4, 5], strict=True))
        output_origin = dict(zip("zyx", expected_origin + fixed_spacing * [2, 3, 4], strict=True))
        entry, _ = legacy.direct_zarr_block_entry(
            sims=[sim],
            transform_key=legacy.REGISTERED_TRANSFORM_KEY,
            sim_coord_dict={},
            block_key=(0, 0, 0),
            output_stack_properties={
                "origin": output_origin,
                "spacing": dict(zip("zyx", fixed_spacing, strict=True)),
                "shape": output_shape,
            },
            output_chunksize=output_shape,
            output_chunk_shape=output_shape,
            output_chunk_origin=output_origin,
            overlap_in_pixels=dict.fromkeys("zyx", 0),
            interpolation_order=1,
        )
        fused = fusion_core._fuse_block_zarr_backed(
            np.asarray(entry, dtype=object),
            output_dtype=raw.dtype,
            sim_coord_dict={},
            sdims=list("zyx"),
            fusion_func=fusion_core.weighted_average_fusion,
            fusion_func_kwargs=None,
            weights_func=None,
            weights_func_kwargs=None,
            overlap_in_pixels=dict.fromkeys("zyx", 0),
            interpolation_order=1,
            blending_widths=None,
            shrink_distance=0,
            trim_overlap=True,
            backend=None,
        )
        fz, fy, fx = np.indices((4, 4, 5), dtype=float) + np.array([2, 3, 4])[:, None, None, None]
        # Invert the known scalar truth equations, independently of the fitted matrix and placement helpers.
        ux = fx - 0.25 * fz - 0.5
        uy = fy - 0.125 * ux - 1.0
        raw_z = 19 - (12 + fz) if flipped else 12 + fz
        raw_x = 13 - (2 + ux) if flipped else 2 + ux
        expected_pixels = 2 * (1000 + 100 * raw_z + 10 * (1 + uy) + raw_x)
        np.testing.assert_allclose(fused, expected_pixels, atol=1.0)
    finally:
        legacy.close_stores(stores)


def test_short_window_maps_reduce_known_z_varying_landmark_residuals() -> None:
    # Z is unchanged; the known lateral displacement is curved over the parent domain.
    source = np.array(
        [
            [z, y, x]
            for z in [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
            for y, x in [(2.0, 2.0), (2.0, 8.0), (8.0, 2.0), (8.0, 8.0)]
        ]
    )
    truth = source.copy()
    truth[:, 2] += 0.04 * (source[:, 0] - 8.0) ** 2
    parent_matrix, parent_translation = affine_from_local_shifts(
        matrix_zyx=np.eye(3),
        translation_zyx=np.zeros(3),
        shape_zyx=(17, 12, 12),
        centers_zyx=truth,
        shifts_zyx=truth - source,
    )
    center = np.array([8.0, 5.5, 5.5])
    global_matrix, global_offset = _local_model_to_global_index_affine(
        matrix=parent_matrix,
        translation=parent_translation,
        fixed_start_zyx=np.zeros(3),
        moving_start_zyx=np.zeros(3),
        crop_shape_zyx=np.array([17, 12, 12]),
    )
    check_source = np.array([[z, 5.0, 5.0] for z in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]])
    check_truth = check_source.copy()
    check_truth[:, 2] += 0.04 * (check_source[:, 0] - 8.0) ** 2
    parent_prediction = (check_source - center) @ parent_matrix.T + center + parent_translation
    parent_errors = np.linalg.norm(parent_prediction - check_truth, axis=1)
    child_errors = []
    for start in [0, 4, 8]:
        child_center = np.array([4.0, 5.5, 5.5])
        initial_matrix, initial_translation = _global_index_affine_to_local_model(
            matrix=global_matrix,
            offset=global_offset,
            fixed_start_zyx=np.array([start, 0, 0]),
            moving_start_zyx=np.array([start, 0, 0]),
            crop_shape_zyx=np.array([9, 12, 12]),
        )
        train = (source[:, 0] > start) & (source[:, 0] < start + 8)
        local_source = source[train] - [start, 0, 0]
        local_truth = truth[train] - [start, 0, 0]
        initial_targets = (
            (local_source - child_center) @ initial_matrix.T + child_center + initial_translation
        )
        matrix, translation = affine_from_local_shifts(
            matrix_zyx=initial_matrix,
            translation_zyx=initial_translation,
            shape_zyx=(9, 12, 12),
            centers_zyx=local_truth,
            shifts_zyx=local_truth - initial_targets,
        )
        held_out = (check_source[:, 0] >= start + 2) & (check_source[:, 0] <= start + 6)
        prediction = (
            (check_source[held_out] - [start, 0, 0] - child_center) @ matrix.T
            + child_center
            + translation
            + [start, 0, 0]
        )
        child_errors.extend(np.linalg.norm(prediction - check_truth[held_out], axis=1))
    assert np.sqrt(np.mean(np.square(child_errors))) < 0.25
    assert np.sqrt(np.mean(np.square(parent_errors))) > 0.5


@pytest.mark.parametrize("cache_bytes", [0, 1024 * 1024])
@pytest.mark.parametrize("prepared", [False, True])
def test_fractional_source_crop_reads_cover_support_and_use_voxel_origin(
    tmp_path: Path, cache_bytes: int, prepared: bool
) -> None:
    import zarr

    data = np.arange(8 * 9 * 10, dtype=np.uint16).reshape(8, 9, 10)
    source = zarr.open_array(
        tmp_path / "crop_source.zarr",
        mode="w",
        shape=data.shape,
        chunks=(2, 3, 5),
        dtype=data.dtype,
        dimension_names=tuple("zyx"),
    )
    source[:] = data
    sim = si_utils.get_sim_from_array(
        source,
        dims=tuple("zyx"),
        scale={"z": 1.7, "y": 0.8, "x": 0.45},
        translation={"z": 20.2, "y": 19.0, "x": 21.0},
    )
    sim.attrs["fusion_source_cache_key"] = "fractional-source::ch0"
    request = {"origin": {"z": 21.9, "y": 19.1, "x": 21.5625}, "shape": {"z": 4, "y": 3, "x": 4}}
    with legacy.basic_corrected_zarr_reads(
        None, cache_key_attr="fusion_source_cache_key", source_cache_max_bytes=cache_bytes
    ):
        info = si_utils.serialize_zarr_backed_sim(sim)
        session = legacy._ACTIVE_DIRECT_SOURCE_READ_SESSION
        assert session is not None
        reads = (
            session.prepare({"views": [{"tile_info": info, "tile_overlap_bb": request}]}, {})
            if prepared
            else None
        )
        with session.consume(reads) if prepared else nullcontext():
            actual = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=request)
    np.testing.assert_array_equal(actual.data, data[1:5, 0:4, 1:6])
    np.testing.assert_allclose(si_utils.get_origin_from_sim(actual, asarray=True), [21.9, 19.0, 21.45])


def test_cached_source_crop_origin_tracks_actual_first_selected_voxel() -> None:
    cached = {
        "data": np.arange(4 * 5 * 6).reshape(4, 5, 6),
        "dims": tuple("zyx"),
        "offsets": {"z": 2, "y": 3, "x": 4},
    }
    info = {
        "origin": {"z": 20.0, "y": 30.0, "x": 40.0},
        "spacing": {"z": 2.0, "y": 3.0, "x": 4.0},
    }
    request = {"origin": {"z": 22.0, "y": 36.0, "x": 52.0}, "shape": {"z": 3, "y": 3, "x": 3}}
    actual = legacy._cached_slice_request_sim(
        cached, info, request, si_utils, message_prefix="test source cache"
    )
    np.testing.assert_array_equal(actual.data, cached["data"][:2, :2, :2])
    np.testing.assert_allclose(si_utils.get_origin_from_sim(actual, asarray=True), [24.0, 39.0, 56.0])
