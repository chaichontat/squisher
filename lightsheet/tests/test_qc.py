from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import zarr

from squisher_lightsheet import qc
from squisher_lightsheet._legacy import render_lr_level4_registration_qc as legacy_qc
from squisher_lightsheet.qc import side_by_tile


def test_registration_qc_registers_jpegxr_before_parsing(monkeypatch) -> None:
    calls: list[str] = []

    class StopAfterParse(RuntimeError):
        pass

    def fake_parse_args(**_kwargs):
        calls.append("parse")
        raise StopAfterParse

    monkeypatch.setattr(legacy_qc, "register_jpegxr_codec", lambda: calls.append("register"), raising=False)
    monkeypatch.setattr(legacy_qc, "parse_args", fake_parse_args)

    with pytest.raises(StopAfterParse):
        legacy_qc.main()

    assert calls == ["register", "parse"]


def test_registration_qc_tile_labels_are_opt_in(tmp_path, monkeypatch) -> None:
    defaults = {
        "default_position_input": tmp_path / "positions.json",
        "default_registration_input": tmp_path / "registration.json",
        "default_output_dir": tmp_path / "output",
    }
    monkeypatch.setattr(sys, "argv", ["qc"])
    assert legacy_qc.parse_args(**defaults).draw_tile_labels is False

    monkeypatch.setattr(sys, "argv", ["qc", "--draw-tile-labels"])
    assert legacy_qc.parse_args(**defaults).draw_tile_labels is True


def test_full_affine_plane_label_uses_tile_index_only() -> None:
    label = legacy_qc.plane_label("234-405-5.019.ome.zarr", (10, 20), (40, 60))

    assert label.text == "019"
    assert label.yx == (30.0, 50.0)


def test_side_lookup_uses_resolved_paths_for_duplicate_basenames() -> None:
    payload = {
        "tiles": [
            {"tile": "tile001.ome.tif", "side": "L", "path": "/sample/TL/tile001.ome.tif"},
            {"tile": "tile001.ome.tif", "side": "R", "path": "/sample/TR/tile001.ome.tif"},
        ]
    }

    assert side_by_tile(payload) == {
        str(Path("/sample/TL/tile001.ome.tif").resolve()): "L",
        str(Path("/sample/TR/tile001.ome.tif").resolve()): "R",
    }


def test_place_global_projections_uses_max_compositing() -> None:
    projections = qc.empty_projection_canvases(np.asarray([3, 4, 5]))
    volume = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)

    qc.place_global_projections(
        projections,
        side="L",
        volume=volume,
        start_zyx=np.asarray([1, 1, 1]),
    )

    assert np.array_equal(projections["L"]["xy"][1:3, 1:4], volume.max(axis=0))
    assert np.array_equal(projections["L"]["xz"][1:3, 1:4], volume.max(axis=1))
    assert np.array_equal(projections["L"]["yz"][1:3, 1:3], volume.max(axis=2))


def test_write_contact_sheet_stacks_images_with_titles(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    output = tmp_path / "sheet.png"
    Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(first)
    Image.fromarray(np.zeros((5, 10, 3), dtype=np.uint8)).save(second)

    qc.write_contact_sheet(output, [("first", first), ("second", second)])

    sheet = Image.open(output)
    assert sheet.size == (20, 76)


def test_default_fused_tile_index_overlay_path_replaces_level0_token() -> None:
    path = Path("/run/Image_10.in-Image_14.fused.level0.ch1.ome.zarr")

    assert qc.default_fused_tile_index_overlay_path(path, 2) == Path(
        "/run/Image_10.in-Image_14.fused.level2.ch1.center-z.tile-index.png"
    )


def test_render_fused_tile_index_overlay_uses_ngff_level_coordinates(tmp_path) -> None:
    fused = tmp_path / "fused.ome.zarr"
    root = zarr.open_group(fused, mode="w")
    array = root.create_array("2", shape=(3, 32, 40), chunks=(1, 16, 20), dtype="uint16")
    array[:] = 0
    array[1, 4:20, 5:25] = 100
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
                "datasets": [
                    {
                        "path": "2",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [2.0, 4.0, 4.0]},
                            {"type": "translation", "translation": [10.0, 20.0, 30.0]},
                        ],
                    }
                ],
            }
        ],
    }
    registration = tmp_path / "registration.json"
    registration.write_text(
        """
{
  "tiles": [
    {
      "tile": "Image_10.000.ome.zarr",
      "shape": [10, 10, 10],
      "stage_translation_um": {"z": 7.0, "y": 35.0, "x": 49.0},
      "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
      "registered_affine": {"matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    },
    {
      "tile": "Image_10.001.ome.zarr",
      "shape": [10, 10, 10],
      "stage_translation_um": {"z": 7.0, "y": 75.0, "x": 89.0},
      "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
      "registered_affine": {"matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    }
  ]
}
""".strip()
        + "\n"
    )

    output = qc.render_fused_tile_index_overlay(
        fused_zarr=fused,
        registration_input=registration,
        level=2,
        z_index=1,
    )

    assert output == tmp_path / "fused.level2.center-z.tile-index.png"
    assert Image.open(output).size == (40, 32)
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["drawn_count"] == 2
    assert [record["label"] for record in summary["records"]] == ["0", "1"]
    np.testing.assert_allclose(summary["records"][0]["center_level_zyx"], [1.0, 5.0, 6.0])


def test_render_fused_tile_index_overlay_can_omit_text_labels(tmp_path) -> None:
    fused = tmp_path / "fused.ome.zarr"
    root = zarr.open_group(fused, mode="w")
    array = root.create_array("0", shape=(1, 24, 24), chunks=(1, 12, 12), dtype="uint16")
    array[:] = 100
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
                        ],
                    }
                ],
            }
        ],
    }
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "Image_10.123.ome.zarr",
                        "shape": [1, 2, 2],
                        "stage_translation_um": {"z": -0.5, "y": 11.0, "x": 11.0},
                        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                        "registered_affine": {"matrix": np.eye(4).tolist()},
                    }
                ]
            }
        )
    )

    output = qc.render_fused_tile_index_overlay(
        fused_zarr=fused,
        registration_input=registration,
        draw_labels=False,
        level=0,
    )

    pixels = np.asarray(Image.open(output))
    assert not np.any(np.all(pixels == (255, 255, 255), axis=-1))
    assert np.any(np.all(pixels == (255, 210, 0), axis=-1))
    assert json.loads(output.with_suffix(".json").read_text())["labels"] == "none"


def test_render_fused_tile_index_overlay_can_omit_tile_markers(tmp_path) -> None:
    fused = tmp_path / "fused.ome.zarr"
    root = zarr.open_group(fused, mode="w")
    array = root.create_array("0", shape=(1, 24, 24), chunks=(1, 12, 12), dtype="uint16")
    array[:] = 100
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
                        ],
                    }
                ],
            }
        ],
    }
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "Image_10.123.ome.zarr",
                        "shape": [1, 2, 2],
                        "stage_translation_um": {"z": -0.5, "y": 11.0, "x": 11.0},
                        "stage_scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                        "registered_affine": {"matrix": np.eye(4).tolist()},
                    }
                ]
            }
        )
    )

    output = qc.render_fused_tile_index_overlay(
        fused_zarr=fused,
        registration_input=registration,
        draw_labels=False,
        draw_markers=False,
        level=0,
    )

    pixels = np.asarray(Image.open(output))
    assert not np.any(np.all(pixels == (255, 210, 0), axis=-1))
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["labels"] == "none"
    assert summary["markers"] == "none"


def test_render_fused_xyz_overlay_qc_writes_native_contact_sheets(tmp_path) -> None:
    reference = tmp_path / "reference.ome.zarr"
    moving = tmp_path / "moving.ome.zarr"
    for path, value in ((reference, 100), (moving, 120)):
        root = zarr.open_group(path, mode="w")
        array = root.create_array("0", shape=(12, 18, 20), chunks=(4, 9, 10), dtype="uint16")
        array[:] = 0
        array[2:10, 4:14, 5:16] = value
    output_dir = tmp_path / "qc"

    output = qc.render_fused_xyz_overlay_qc(
        reference_zarr=reference,
        moving_zarr=moving,
        output_dir=output_dir,
        level=0,
        thumb_level=0,
        panel_size=8,
    )

    assert output == output_dir / "native_xyz_spanning_overlay_contact_sheet.png"
    assert Image.open(output).size == (24, 32)
    assert (output_dir / "native_xy_zspan_contentful_overlay.png").exists()
    payload = json.loads((output_dir / "native_xyz_spanning_overlay_contact_sheet.json").read_text())
    assert payload["artifact_type"] == "lightsheet.fused_xyz_overlay_qc.v1"
    assert payload["color"] == {"red": "reference", "green": "moving"}
    assert {panel["axis"] for panel in payload["panels"]} == {"xy", "xz", "yz"}


def test_render_live_fusion_preview_uses_latest_log_output_and_nonzero_planes(tmp_path) -> None:
    fused = tmp_path / "live.ome.zarr"
    root = zarr.open_group(fused, mode="w")
    array = root.create_array("0", shape=(6, 6, 8), chunks=(2, 3, 4), dtype="uint16")
    array[:] = 0
    array[1, 1:4, 2:6] = 100
    array[4, 2:5, 3:7] = 200
    stale = tmp_path / "stale.ome.zarr"
    log = tmp_path / "fusion.log"
    log.write_text(f"streaming write to {stale}\nstreaming write to {fused}\n")
    output = tmp_path / "preview.png"

    result = qc.render_live_fusion_preview(
        log_path=log,
        output=output,
        stride=1,
        z_start=0,
        z_step=1,
        max_panels=2,
    )

    assert result.source_zarr == fused.resolve()
    assert result.shape == (6, 6, 8)
    assert result.chunks == (2, 3, 4)
    assert result.sampled_plane_count == 2
    assert result.selected_z == (1, 4)
    assert result.selected_nonzero_pixels == (12, 12)
    assert Image.open(output).size == (16, 34)


def test_render_fused_xyz_overlay_qc_rejects_shape_mismatch(tmp_path) -> None:
    reference = tmp_path / "reference.ome.zarr"
    moving = tmp_path / "moving.ome.zarr"
    zarr.open_group(reference, mode="w").create_array("0", shape=(4, 6, 8), chunks=(2, 3, 4), dtype="uint16")
    zarr.open_group(moving, mode="w").create_array("0", shape=(5, 6, 8), chunks=(2, 3, 4), dtype="uint16")

    with pytest.raises(ValueError, match="shape mismatch"):
        qc.render_fused_xyz_overlay_qc(
            reference_zarr=reference,
            moving_zarr=moving,
            output_dir=tmp_path / "qc",
            level=0,
            thumb_level=0,
            panel_size=8,
        )
