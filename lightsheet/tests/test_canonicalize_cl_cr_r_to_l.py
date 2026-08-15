from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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
