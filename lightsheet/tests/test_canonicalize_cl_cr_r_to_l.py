from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import zarr


def load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "canonicalize_cl_cr_r_to_l.py"
    spec = importlib.util.spec_from_file_location("canonicalize_cl_cr_r_to_l", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clcr_workflow_commands_use_canonical_final_outputs() -> None:
    module = load_script_module()

    fusion = module.fusion_commands(
        dataset_dir=Path("/data/230Tnc-CLR-488514561638"),
        position_json=Path("/final/clcr_r_to_l.deconvCZYX.positions.json"),
        registration_json=Path("/final/registration.json"),
        output_root=Path("/final"),
        channels=["0", "1"],
    )
    movie = module.movie_commands(
        fusion_output_root=Path("/final"),
        position_json=Path("/final/clcr_r_to_l.deconvCZYX.positions.json"),
        channels=["0", "1"],
    )

    assert "--output /final --channel 0" in fusion[0]
    assert "--output /final --channel 1" in fusion[1]
    assert "/final/fused.ch0.ome.zarr" in movie[0]
    assert "/final/fused.ch1.ome.zarr" in movie[1]


def test_clcr_parser_does_not_default_to_sample_specific_fusion_root(monkeypatch) -> None:
    module = load_script_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "canonicalize_cl_cr_r_to_l.py",
            "--dataset-dir",
            "/data",
            "--left-position",
            "/left.json",
            "--right-position",
            "/right.json",
            "--manifest",
            "/manifest.json",
            "--phasecorr",
            "/phasecorr.json",
            "--method8-summary",
            "/method8.json",
            "--deconv-root",
            "/deconv",
        ],
    )

    args = module.parse_args()

    assert args.fusion_output_root is None


def test_record_centers_preserve_negative_axis_orientation() -> None:
    module = load_script_module()
    records = [
        {
            "tile": "sample-CR-000.ome.zarr",
            "translation_um": {"z": 8.0, "y": 0.0, "x": 12.0},
            "scale_um": {"z": -1.0, "y": 2.0, "x": -3.0},
        }
    ]

    center = module._record_centers(records, [1, 8, 4, 4])

    np.testing.assert_allclose(center, [4.0, 4.0, 6.0])


def test_zarr_level_shape_uses_ngff_dataset_path(tmp_path) -> None:
    module = load_script_module()
    path = tmp_path / "tile.ome.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.create_array("pixels", shape=(1, 2, 3, 4), chunks=(1, 1, 3, 4), dtype="uint16")
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{"datasets": [{"path": "pixels"}]}],
    }

    assert module._zarr_level_shape(path) == (1, 2, 3, 4)


def test_method8_shift_moves_flipped_cr_in_physical_fixed_direction(tmp_path) -> None:
    module = load_script_module()
    shape_czyx = [1, 8, 4, 4]
    left = [
        {
            "tile": "sample-CL-000.ome.zarr",
            "path": "/data/sample-CL-000.ome.zarr",
            "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
            "scale_um": {"z": 1.0, "y": 2.0, "x": 3.0},
        }
    ]
    right = [
        {
            "tile": "sample-CR-000.ome.zarr",
            "path": "/data/sample-CR-000.ome.zarr",
            "translation_um": {"z": 8.0, "y": 0.0, "x": 12.0},
            "scale_um": {"z": -1.0, "y": 2.0, "x": -3.0},
        }
    ]
    manifest = {
        "metadata_centers_um_zyx": {
            "CL": module._record_centers(left, shape_czyx).tolist(),
            "CR": module._record_centers(right, shape_czyx).tolist(),
        }
    }
    method8 = {
        "windows": [
            {
                "status": "accepted",
                "fixed_tile": "sample-CL-000.ome.zarr",
                "moving_tile": "sample-CR-000.ome.zarr",
                "local_translation_zyx": [1.0, 2.0, 3.0],
            }
        ]
    }

    payload = module.build_clspace_position_payload(
        left_records=left,
        right_records=right,
        manifest=manifest,
        phasecorr={
            "fixed": "CL",
            "moving": "CR",
            "phase_shift_to_apply_to_cr_yx_um": [0.0, 0.0],
            "phase_shift_to_apply_to_cr_yx_px": [0.0, 0.0],
        },
        method8_summary=method8,
        left_position=tmp_path / "cl.json",
        right_position=tmp_path / "cr.json",
        manifest_path=tmp_path / "manifest.json",
        phasecorr_path=tmp_path / "phase.json",
        method8_summary_path=tmp_path / "method8.json",
        shape_czyx=shape_czyx,
        channels=["0"],
        deconv_root=None,
        artifact_stem="sample",
    )

    cr = next(tile for tile in payload["tiles"] if tile["source_side"] == "CR")
    assert cr["translation_um"] == {"z": 9.0, "y": 4.0, "x": 21.0}
    assert payload["alignment"]["method8_median_shift_um_zyx"] == [1.0, 4.0, 9.0]


def test_registered_lr_manual_shift_is_kept_in_canonical_cr_placement(tmp_path) -> None:
    module = load_script_module()
    shape_czyx = [1, 8, 4, 4]
    left = [
        {
            "tile": "sample-CL-000.ome.zarr",
            "path": "/data/sample-CL-000.ome.zarr",
            "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
            "scale_um": {"z": 1.0, "y": 2.0, "x": 3.0},
        }
    ]
    metadata_right = {
        "tile": "sample-CR-000.ome.zarr",
        "path": "/data/sample-CR-000.ome.zarr",
        "translation_um": {"z": 8.0, "y": 0.0, "x": 12.0},
        "scale_um": {"z": -1.0, "y": 2.0, "x": -3.0},
    }
    manifest = {
        "metadata_centers_um_zyx": {
            "CL": module._record_centers(left, shape_czyx).tolist(),
            "CR": module._record_centers([metadata_right], shape_czyx).tolist(),
        }
    }
    # align-registered-lr bakes its total CR shift into the side positions it writes.
    right = [metadata_right | {"translation_um": {"z": 7.0, "y": 2.0, "x": 15.0}}]
    method8 = {
        "windows": [
            {
                "status": "accepted",
                "fixed_tile": "sample-CL-000.ome.zarr",
                "moving_tile": "sample-CR-000.ome.zarr",
                "local_translation_zyx": [0.0, 0.0, 0.0],
            }
        ]
    }

    payload = module.build_clspace_position_payload(
        left_records=left,
        right_records=right,
        manifest=manifest,
        phasecorr={
            "artifact_type": "squisher_lightsheet.registered_lr_phasecorr.v1",
            "fixed": "CL",
            "moving": "CR",
            "alignment_mode": "manual",
            "phase_shift_to_apply_to_cr_yx_um": [0.0, 0.0],
            "phase_shift_to_apply_to_cr_yx_px": [0.0, 0.0],
            "total_shift_to_apply_to_cr_zyx_um": [-1.0, 2.0, 3.0],
        },
        method8_summary=method8,
        left_position=tmp_path / "cl.json",
        right_position=tmp_path / "cr.json",
        manifest_path=tmp_path / "manifest.json",
        phasecorr_path=tmp_path / "phase.json",
        method8_summary_path=tmp_path / "method8.json",
        shape_czyx=shape_czyx,
        channels=["0"],
        deconv_root=None,
        artifact_stem="sample",
    )

    cr = next(tile for tile in payload["tiles"] if tile["source_side"] == "CR")
    assert cr["translation_um"] == {"z": 7.0, "y": 2.0, "x": 15.0}


def test_side_specific_tile_depths_keep_their_own_shapes(tmp_path) -> None:
    module = load_script_module()
    left = [
        {
            "tile": "sample-CL-000.ome.zarr",
            "path": "/data/sample-CL-000.ome.zarr",
            "shape": [2, 10, 4, 4],
            "axes": "CZYX",
            "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
            "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        }
    ]
    right = [
        {
            "tile": "sample-CR-000.ome.zarr",
            "path": "/data/sample-CR-000.ome.zarr",
            "shape": [2, 6, 4, 4],
            "axes": "CZYX",
            "translation_um": {"z": 6.0, "y": 0.0, "x": 4.0},
            "scale_um": {"z": -1.0, "y": 1.0, "x": -1.0},
        }
    ]
    default_shape = module._resolve_shape_czyx(None, left, right, tmp_path, ["0", "1"])
    payload = module.build_clspace_position_payload(
        left_records=left,
        right_records=right,
        manifest={
            "metadata_centers_um_zyx": {
                "CL": module._record_centers(left, default_shape).tolist(),
                "CR": module._record_centers(right, default_shape).tolist(),
            }
        },
        phasecorr={
            "fixed": "CL",
            "moving": "CR",
            "phase_shift_to_apply_to_cr_yx_um": [0.0, 0.0],
            "phase_shift_to_apply_to_cr_yx_px": [0.0, 0.0],
        },
        method8_summary={
            "windows": [
                {
                    "status": "accepted",
                    "fixed_tile": "sample-CL-000.ome.zarr",
                    "moving_tile": "sample-CR-000.ome.zarr",
                    "local_translation_zyx": [0.0, 0.0, 0.0],
                }
            ]
        },
        left_position=tmp_path / "cl.json",
        right_position=tmp_path / "cr.json",
        manifest_path=tmp_path / "manifest.json",
        phasecorr_path=tmp_path / "phase.json",
        method8_summary_path=tmp_path / "method8.json",
        shape_czyx=default_shape,
        channels=["0", "1"],
        deconv_root=None,
        artifact_stem="sample",
    )

    shapes = {tile["source_side"]: tile["shape"] for tile in payload["tiles"]}
    assert shapes == {"CL": [2, 10, 4, 4], "CR": [2, 6, 4, 4]}
    cr = next(tile for tile in payload["tiles"] if tile["source_side"] == "CR")
    assert cr["translation_um"] == {"z": 6.0, "y": 0.0, "x": 4.0}


def test_mixed_tile_depths_require_explicit_record_shapes(tmp_path) -> None:
    module = load_script_module()
    left = [{"tile": "sample-CL-000.ome.zarr", "shape": [2, 10, 4, 4], "axes": "CZYX"}]
    right = [
        {"tile": "sample-CR-000.ome.zarr", "shape": [2, 6, 4, 4], "axes": "CZYX"},
        {"tile": "sample-CR-001.ome.zarr"},
    ]

    with pytest.raises(ValueError, match="explicit"):
        module._resolve_shape_czyx(None, left, right, tmp_path, ["0", "1"])


def test_identity_registration_labels_views_like_the_position_input(tmp_path) -> None:
    module = load_script_module()
    record = {
        "tile": "sample-CR-000.ome.zarr",
        "side": "R",
        "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
        "scale_um": {"z": -1.0, "y": 1.0, "x": -1.0},
        "spacing_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        "shape": [1, 2, 2, 2],
        "axes": "CZYX",
        "channels": ["0"],
        "tracks": [],
    }

    registration = module.build_identity_registration_payload(
        {"tiles": [record]}, deconv_root=tmp_path, artifact_stem="sample"
    )

    # Fusion reads views from the position input's side; registration consumers must agree.
    assert registration["tiles"][0]["source_view"] == "R"
