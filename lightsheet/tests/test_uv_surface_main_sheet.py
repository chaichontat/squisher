from __future__ import annotations

import json

import numpy as np
import SimpleITK as sitk

from squisher_lightsheet.uv_surface import (
    _main_sheet_source_keep,
    build_bilateral_ventricular_uv_surfaces,
)


def test_main_sheet_source_keep_follows_upper_bridge_envelope() -> None:
    crossing = np.zeros((6, 7), dtype=bool)
    crossing[:4, 2] = True
    crossing[:3, 3] = True
    crossing[:2, 4] = True

    keep, diagnostics = _main_sheet_source_keep(crossing)

    expected = np.zeros_like(crossing)
    expected[:, :2] = True
    expected[4:, 2] = True
    expected[3:, 3] = True
    expected[2:, 4] = True
    np.testing.assert_array_equal(keep, expected)
    assert diagnostics == {
        "cut_bridge_y_index_range": [2, 4],
        "cut_bridge_upper_z_index_by_y": [3, 2, 1],
    }

    clipped, clipped_diagnostics = _main_sheet_source_keep(crossing, min_z_index=4)
    np.testing.assert_array_equal(clipped, expected & (np.indices(expected.shape)[0] >= 4))
    assert clipped_diagnostics["main_sheet_min_z_index"] == 4


def test_bilateral_main_sheet_trim_removes_inferior_bridge_wrap(tmp_path) -> None:
    mask = np.zeros((8, 8, 15), dtype=np.uint8)
    mask[:, :, 1:5] = 1
    mask[:, :, 10:14] = 1
    mask[:2, :, 1:14] = 1
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((0.001, 0.002, 0.003))
    image.SetOrigin((0.01, 0.02, 0.03))
    mask_path = tmp_path / "joined-hemispheres.nrrd"
    sitk.WriteImage(image, mask_path)

    output_dir = tmp_path / "bilateral"
    manifest_path = build_bilateral_ventricular_uv_surfaces(
        mask_path=mask_path,
        output_dir=output_dir,
        split_x_index=7,
        smoothing_um=0,
        keep_main_sheet=True,
        main_sheet_min_z_index=3,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["cut_bridge_y_index_range"] == [0, 7]
    assert manifest["cut_bridge_upper_z_index_by_y"] == [1] * 8
    assert manifest["main_sheet_min_z_index"] == 3
    for half in ("low-x", "high-x"):
        with np.load(output_dir / half / "surface.npz") as surface:
            assert surface["vertices_xyz_um"][:, 2].min() >= 39.0
