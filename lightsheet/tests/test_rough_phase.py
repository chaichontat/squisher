from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from squisher_lightsheet import rough_phase
from squisher_lightsheet.rough_phase import estimate_shift_yx_px, write_overlay


def test_rough_phase_recovers_synthetic_translation_and_writes_overlay(tmp_path) -> None:
    y, x = np.mgrid[:96, :96]
    fixed = np.exp(-(((y - 48) ** 2 + (x - 44) ** 2) / 120.0)).astype(np.float32)
    moving = np.roll(np.roll(fixed, 3, axis=0), -5, axis=1)
    coverage = np.ones_like(fixed, dtype=bool)

    shift, details = estimate_shift_yx_px(
        {"L": fixed, "R": moving},
        {"L": coverage, "R": coverage},
        search_margin_px=0,
        upsample_factor=10,
    )
    overlay = tmp_path / "overlay.png"
    write_overlay(overlay, left=fixed, right=moving)

    np.testing.assert_allclose(shift, [-3, 5], atol=0.2)
    assert details["overlap_pixels"] == fixed.size
    assert details["crop_to_overlap"] is True
    assert overlay.exists()
    assert overlay.stat().st_size > 0


def test_estimate_shift_can_use_full_canvas_without_overlap_crop() -> None:
    y, x = np.mgrid[:96, :160]
    fixed = np.exp(-(((y - 48) ** 2 + (x - 80) ** 2) / 180.0)).astype(np.float32)
    moving = np.roll(np.roll(fixed, -4, axis=0), 7, axis=1)
    coverage = np.ones_like(fixed, dtype=bool)

    shift, details = rough_phase.legacy.estimate_shift_px(
        {"L": fixed, "R": moving},
        {"L": coverage, "R": coverage},
        axes=("y", "x"),
        phase_mask=None,
        crop_to_overlap=False,
        search_margin_px=0,
        upsample_factor=10,
    )

    np.testing.assert_allclose(shift, [4, -7], atol=0.2)
    assert details["crop_to_overlap"] is False
    assert details["crop_yx"] == [[0, 96], [0, 160]]


def test_estimate_shift_uses_masked_phase_correlation_for_seam_mask(monkeypatch) -> None:
    import skimage.registration

    fixed = np.zeros((32, 48), dtype=np.float32)
    moving = np.zeros_like(fixed)
    fixed[12:20, 18:26] = 1.0
    moving[13:21, 20:28] = 1.0
    coverage = np.ones_like(fixed, dtype=bool)
    phase_mask = np.zeros_like(fixed, dtype=bool)
    phase_mask[8:24, 10:34] = True
    observed = {}

    def fake_phase_cross_correlation(reference_image, moving_image, **kwargs):
        observed["reference_shape"] = reference_image.shape
        observed["moving_shape"] = moving_image.shape
        observed["reference_mask"] = kwargs.get("reference_mask")
        observed["moving_mask"] = kwargs.get("moving_mask")
        observed["overlap_ratio"] = kwargs.get("overlap_ratio")
        observed["upsample_factor"] = kwargs.get("upsample_factor")
        return np.array([-1.0, -2.0]), np.nan, np.nan

    monkeypatch.setattr(skimage.registration, "phase_cross_correlation", fake_phase_cross_correlation)

    shift, details = rough_phase.legacy.estimate_shift_px(
        {"L": fixed, "R": moving},
        {"L": coverage, "R": coverage},
        axes=("z", "x"),
        phase_mask=phase_mask,
        crop_to_overlap=True,
        search_margin_px=8,
        upsample_factor=10,
    )

    np.testing.assert_array_equal(shift, [-1.0, -2.0])
    assert observed["reference_shape"] == (16, 24)
    assert observed["moving_shape"] == (16, 24)
    assert observed["reference_mask"].shape == (16, 24)
    assert observed["moving_mask"].shape == (16, 24)
    assert observed["reference_mask"].all()
    assert observed["moving_mask"].all()
    assert observed["overlap_ratio"] == 0.05
    assert observed["upsample_factor"] is None
    assert details["phase_mode"] == "masked"
    assert details["masked_overlap_ratio"] == 0.05
    assert details["crop_zx"] == [[8, 24], [10, 34]]


def test_masked_phase_crop_uses_union_coverage_not_only_current_overlap(monkeypatch) -> None:
    import skimage.registration

    image = np.ones((40, 32), dtype=np.float32)
    coverage_l = np.zeros_like(image, dtype=bool)
    coverage_r = np.zeros_like(image, dtype=bool)
    coverage_l[18:36, 4:28] = True
    coverage_r[4:22, 4:28] = True
    phase_mask = np.zeros_like(image, dtype=bool)
    phase_mask[4:36, 4:28] = True
    observed = {}

    def fake_phase_cross_correlation(reference_image, moving_image, **kwargs):
        observed["reference_shape"] = reference_image.shape
        observed["reference_mask_sum"] = int(kwargs["reference_mask"].sum())
        observed["moving_mask_sum"] = int(kwargs["moving_mask"].sum())
        return np.array([0.0, 0.0]), np.nan, np.nan

    monkeypatch.setattr(skimage.registration, "phase_cross_correlation", fake_phase_cross_correlation)

    _shift, details = rough_phase.legacy.estimate_shift_px(
        {"L": image, "R": image},
        {"L": coverage_l, "R": coverage_r},
        axes=("z", "x"),
        phase_mask=phase_mask,
        crop_to_overlap=True,
        search_margin_px=8,
        upsample_factor=10,
    )

    assert details["crop_zx"] == [[4, 36], [4, 28]]
    assert details["overlap_pixels"] == 4 * 24
    assert observed["reference_shape"] == (32, 24)
    assert observed["reference_mask_sum"] == 18 * 24
    assert observed["moving_mask_sum"] == 18 * 24


def test_rough_phase_output_remains_a_position_artifact(monkeypatch, tmp_path) -> None:
    position_input = tmp_path / "input.positions.json"
    output_position = tmp_path / "output.positions.json"
    output_dir = tmp_path / "qc"
    position_input.write_text(json.dumps({"artifact_type": "lightsheet.position.v1", "tiles": []}))
    geometry = SimpleNamespace(
        level_factor=16,
        level_spacing_yx_um=np.array([2.0, 2.0]),
        level_spacing_zyx_um=np.array([10.0, 2.0, 2.0]),
        shape_yx=np.array([8, 8]),
        shape_zyx=np.array([4, 8, 8]),
    )

    monkeypatch.setattr(rough_phase.legacy, "load_tiles", lambda _payload: ["tile"])
    monkeypatch.setattr(rough_phase.legacy, "build_geometry", lambda _tiles, *, level: geometry)
    monkeypatch.setattr(
        rough_phase.legacy,
        "render_center_z_canvases",
        lambda _tiles, *, geometry, channel: (
            {"L": np.ones((8, 8), dtype=np.float32), "R": np.ones((8, 8), dtype=np.float32)},
            {"L": np.ones((8, 8), dtype=bool), "R": np.ones((8, 8), dtype=bool)},
            [],
        ),
    )
    monkeypatch.setattr(
        rough_phase.legacy,
        "estimate_shift_px",
        lambda _images, _coverage, *, axes, phase_mask, crop_to_overlap, search_margin_px, upsample_factor: (
            np.array([0.0, 0.0]),
            {"phase_axes": list(axes), "crop_to_overlap": crop_to_overlap},
        ),
    )
    monkeypatch.setattr(
        rough_phase.legacy,
        "shifted_payload",
        lambda payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        rough_phase.legacy,
        "write_overlay",
        lambda path, *, left, right: path.write_bytes(b"png"),
    )
    def fail_global_projection(*_args, **_kwargs):
        raise AssertionError("XY rough phase should not render XZ/YZ global projections")

    monkeypatch.setattr(rough_phase.legacy, "render_global_projection_canvases", fail_global_projection)

    rough_phase.rough_phase_align(
        position_input=position_input,
        output_position=output_position,
        output_dir=output_dir,
        z_slab_planes=1,
    )

    payload = json.loads(output_position.read_text())
    summary = json.loads((output_dir / "level4_xy_phase_alignment_ch0.json").read_text())
    assert payload["artifact_type"] == "lightsheet.position.v1"
    assert payload["derived_by"] == "lightsheet.rough_phase.v1"
    assert summary["phase_plane"] == "xy"
    assert summary["z_display_scale_for_xz_yz"] is None
    assert summary["phase_alignment"]["crop_to_overlap"] is True
    assert set(summary["corrected_projection_overlays"]) == {"xy"}
    assert summary["corrected_projection_contact_sheet"] is None


def test_rough_phase_uses_xz_for_z_join(monkeypatch, tmp_path) -> None:
    position_input = tmp_path / "input.positions.json"
    output_position = tmp_path / "output.positions.json"
    output_dir = tmp_path / "qc"
    position_input.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.position.v1",
                "diagnostics": {"join_axis": "z"},
                "tiles": [{"side": "R", "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0}}],
            }
        )
    )
    geometry = SimpleNamespace(
        level_factor=16,
        level_spacing_yx_um=np.array([2.0, 2.0]),
        level_spacing_zyx_um=np.array([10.0, 2.0, 4.0]),
        global_min_zyx_um=np.array([0.0, 0.0, 0.0]),
        shape_yx=np.array([8, 8]),
        shape_zyx=np.array([4, 8, 8]),
    )

    monkeypatch.setattr(rough_phase.legacy, "load_tiles", lambda _payload: ["tile"])
    monkeypatch.setattr(rough_phase.legacy, "build_geometry", lambda _tiles, *, level: geometry)
    monkeypatch.setattr(
        rough_phase.legacy,
        "render_xz_projection_canvases",
        lambda _tiles, *, geometry, channel, z_range_px: (
            {"L": np.ones((4, 8), dtype=np.float32), "R": np.ones((4, 8), dtype=np.float32)},
            {"L": np.ones((4, 8), dtype=bool), "R": np.ones((4, 8), dtype=bool)},
            [{"z_range_px": list(z_range_px)}],
        ),
    )
    monkeypatch.setattr(
        rough_phase.legacy,
        "seam_band_mask",
        lambda _tiles, *, geometry, axes, seam_fraction, overlap_um, overlap_fraction: (
            np.ones((4, 8), dtype=bool),
            {
                "seam_axis": "z",
                "seam_um": 20.0,
                "seam_fraction": seam_fraction,
                "seam_band_um_requested": 40.0,
                "seam_band_range_px": [0, 4],
            },
        ),
    )
    observed = {}

    def fake_estimate_shift(_images, _coverage, *, axes, phase_mask, crop_to_overlap, search_margin_px, upsample_factor):
        observed["phase_mask"] = phase_mask
        observed["image_shape"] = _images["L"].shape
        return np.array([1.5, -2.0]), {"phase_axes": list(axes), "crop_to_overlap": crop_to_overlap}

    monkeypatch.setattr(
        rough_phase.legacy,
        "estimate_shift_px",
        fake_estimate_shift,
    )
    monkeypatch.setattr(
        rough_phase.legacy,
        "render_global_projection_canvases",
        lambda _tiles, *, geometry, channel: (
            {
                "L": {
                    "xy": np.ones((8, 8), dtype=np.float32),
                    "xz": np.ones((4, 8), dtype=np.float32),
                    "yz": np.ones((4, 8), dtype=np.float32),
                },
                "R": {
                    "xy": np.ones((8, 8), dtype=np.float32),
                    "xz": np.ones((4, 8), dtype=np.float32),
                    "yz": np.ones((4, 8), dtype=np.float32),
                },
            },
            [],
        ),
    )
    monkeypatch.setattr(
        rough_phase.legacy,
        "write_global_projection_outputs",
        lambda output_dir, *, projections, level, channel, z_display_scale: (
            [
                ("XY", output_dir / "xy.png"),
                ("XZ", output_dir / "xz.png"),
                ("YZ", output_dir / "yz.png"),
            ],
            output_dir / "contact.png",
        ),
    )

    rough_phase.rough_phase_align(
        position_input=position_input,
        output_position=output_position,
        output_dir=output_dir,
        phase_downsample_zyx=(2, 1, 2),
    )

    payload = json.loads(output_position.read_text())
    summary = json.loads((output_dir / "level4_xz_phase_alignment_ch0.json").read_text())
    record = payload["tiles"][0]["translation_um"]
    assert summary["phase_plane"] == "xz"
    assert summary["phase_axes"] == ["z", "x"]
    assert summary["phase_alignment"]["crop_to_overlap"] is True
    assert summary["initial_tiles"] == [{"z_range_px": [0, 4]}]
    assert observed["phase_mask"] is not None
    assert observed["phase_mask"].shape == observed["image_shape"]
    assert set(summary["corrected_projection_overlays"]) == {"xz_seam"}
    assert summary["corrected_projection_contact_sheet"] is None
    assert record == {"z": 40.0, "y": 20.0, "x": 14.0}


def test_rough_phase_uses_zyx_slab_for_x_join_when_requested(monkeypatch, tmp_path) -> None:
    position_input = tmp_path / "input.positions.json"
    output_position = tmp_path / "output.positions.json"
    output_dir = tmp_path / "qc"
    position_input.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.position.v1",
                "diagnostics": {"join_axis": "x"},
                "tiles": [{"side": "R", "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0}}],
            }
        )
    )
    geometry = SimpleNamespace(
        level_factor=16,
        level_spacing_yx_um=np.array([2.0, 2.0]),
        level_spacing_zyx_um=np.array([10.0, 2.0, 4.0]),
        shape_yx=np.array([8, 8]),
        shape_zyx=np.array([6, 8, 8]),
    )
    volume = np.ones((4, 8, 8), dtype=np.float32)

    monkeypatch.setattr(rough_phase.legacy, "load_tiles", lambda _payload: ["tile"])
    monkeypatch.setattr(rough_phase.legacy, "build_geometry", lambda _tiles, *, level: geometry)
    monkeypatch.setattr(
        rough_phase.legacy,
        "render_center_z_slab_canvases",
        lambda _tiles, *, geometry, channel, slab_planes: (
            {"L": volume, "R": volume},
            {"L": np.ones(volume.shape, dtype=bool), "R": np.ones(volume.shape, dtype=bool)},
            [],
            {
                "slab_planes": slab_planes,
                "z_sampling": "native_center_z_slab",
                "native_z_spacing_um": 2.5,
            },
        ),
    )
    monkeypatch.setattr(
        rough_phase.legacy,
        "estimate_shift_px",
        lambda _images, _coverage, *, axes, phase_mask, crop_to_overlap, search_margin_px, upsample_factor: (
            np.array([1.0, -2.0, 3.0]),
            {"phase_axes": list(axes), "crop_to_overlap": crop_to_overlap},
        ),
    )

    rough_phase.rough_phase_align(
        position_input=position_input,
        output_position=output_position,
        output_dir=output_dir,
        z_slab_planes=4,
        crop_overlap=False,
        phase_downsample_zyx=(1, 1, 1),
    )

    payload = json.loads(output_position.read_text())
    summary = json.loads((output_dir / "level4_zyx_phase_alignment_ch0.json").read_text())
    record = payload["tiles"][0]["translation_um"]
    assert summary["phase_plane"] == "zyx"
    assert summary["phase_axes"] == ["z", "y", "x"]
    assert summary["phase_alignment"]["crop_to_overlap"] is False
    assert summary["phase_alignment"]["slab"] == {
        "slab_planes": 4,
        "z_sampling": "native_center_z_slab",
        "native_z_spacing_um": 2.5,
    }
    assert summary["phase_alignment"]["shift_spacing_zyx_um"] == [2.5, 2.0, 4.0]
    assert set(summary["corrected_projection_overlays"]) == {"xy_overlap_center"}
    assert summary["corrected_projection_contact_sheet"] is None
    assert summary["corrected_overlap_center_plane"]["overlap_plane_pixels"] == 30
    assert record == {"z": 12.5, "y": 16.0, "x": 42.0}


def test_seam_band_mask_uses_only_requested_fraction_next_to_z_seam() -> None:
    tile_shape = np.array([100, 10, 10])
    tiles = [
        rough_phase.legacy.TileRecord(
            tile="L",
            side="L",
            path=Path("L.tif"),
            translation_zyx_um=np.array([0.0, 0.0, 0.0]),
            scale_zyx_um=np.array([1.0, 1.0, 1.0]),
            shape_zyx=tile_shape,
            axes="ZYX",
        ),
        rough_phase.legacy.TileRecord(
            tile="R",
            side="R",
            path=Path("R.tif"),
            translation_zyx_um=np.array([-75.0, 0.0, 0.0]),
            scale_zyx_um=np.array([1.0, 1.0, 1.0]),
            shape_zyx=tile_shape,
            axes="ZYX",
        ),
    ]
    geometry = SimpleNamespace(
        global_min_zyx_um=np.array([-75.0, 0.0, 0.0]),
        level_spacing_zyx_um=np.array([1.0, 1.0, 1.0]),
        shape_zyx=np.array([175, 10, 10]),
    )

    mask, details = rough_phase.legacy.seam_band_mask(
        tiles,
        geometry=geometry,
        axes=("z", "x"),
        seam_fraction=0.10,
        overlap_um=25.0,
        overlap_fraction=0.25,
    )

    assert details["seam_band_range_um"] == [0.0, 10.0]
    assert details["seam_band_range_px"] == [75, 85]
    assert int(mask.sum()) == 10 * 10


def test_xz_phase_renderer_reads_only_requested_z_range(monkeypatch) -> None:
    tile = rough_phase.legacy.TileRecord(
        tile="L0",
        side="L",
        path=Path("L0.ome.zarr"),
        translation_zyx_um=np.array([0.0, 0.0, 0.0]),
        scale_zyx_um=np.array([1.0, 1.0, 1.0]),
        shape_zyx=np.array([100, 20, 28]),
        axes="ZYX",
    )
    geometry = SimpleNamespace(
        level_factor=4,
        level_spacing_zyx_um=np.array([4.0, 4.0, 4.0]),
        global_min_zyx_um=np.array([0.0, 0.0, 0.0]),
        shape_zyx=np.array([25, 5, 7]),
    )
    observed = {}

    def fake_sampled_tile_volume_z_range(
        _tile,
        *,
        channel,
        level_factor,
        sampled_z_start,
        sampled_z_stop,
    ):
        observed.update(
            channel=channel,
            level_factor=level_factor,
            sampled_z_range=[sampled_z_start, sampled_z_stop],
        )
        return np.ones((sampled_z_stop - sampled_z_start, 5, 7), dtype=np.float32)

    monkeypatch.setattr(rough_phase.legacy, "sampled_tile_volume_z_range", fake_sampled_tile_volume_z_range)

    images, coverage, rows = rough_phase.legacy.render_xz_projection_canvases(
        [tile],
        geometry=geometry,
        channel=0,
        z_range_px=(10, 18),
    )

    assert observed == {"channel": 0, "level_factor": 4, "sampled_z_range": [10, 18]}
    assert images["L"].shape == (8, 7)
    assert coverage["L"].shape == (8, 7)
    assert rows[0]["sampled_z_range"] == [10, 18]
    assert rows[0]["phase_canvas_start_zx"] == [0, 0]


def test_center_z_slab_reads_native_z_range_and_downsamples_yx(monkeypatch) -> None:
    tile = rough_phase.legacy.TileRecord(
        tile="L0",
        side="L",
        path=Path("L0.tif"),
        translation_zyx_um=np.array([0.0, 0.0, 0.0]),
        scale_zyx_um=np.array([1.0, 1.0, 1.0]),
        shape_zyx=np.array([20, 16, 16]),
        axes="ZYX",
    )
    geometry = SimpleNamespace(
        level_factor=4,
        level_spacing_zyx_um=np.array([4.0, 4.0, 4.0]),
        global_min_zyx_um=np.array([0.0, 0.0, 0.0]),
        global_max_zyx_um=np.array([20.0, 16.0, 16.0]),
        shape_zyx=np.array([5, 4, 4]),
    )
    observed = {}

    def fake_sampled_tile_center_z_slab(
        _tile,
        *,
        channel,
        source_z_start,
        source_z_stop,
        yx_level_factor,
    ):
        observed.update(
            channel=channel,
            source_z_range=[source_z_start, source_z_stop],
            yx_level_factor=yx_level_factor,
        )
        return np.ones((source_z_stop - source_z_start, 4, 4), dtype=np.float32)

    monkeypatch.setattr(rough_phase.legacy, "sampled_tile_center_z_slab", fake_sampled_tile_center_z_slab)

    images, coverage, rows, details = rough_phase.legacy.render_center_z_slab_canvases(
        [tile],
        geometry=geometry,
        channel=0,
        slab_planes=6,
    )

    assert observed == {"channel": 0, "source_z_range": [7, 13], "yx_level_factor": 4}
    assert details["z_sampling"] == "native_center_z_slab"
    assert details["slab_range_z_px"] == [7, 13]
    assert details["yx_downsample_factor"] == 4
    assert images["L"].shape == (6, 4, 4)
    assert coverage["L"].shape == (6, 4, 4)
    assert rows[0]["source_z_range"] == [7, 13]
