from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import squisher_lightsheet.method8_stitch_register as stitch_register
import squisher_lightsheet.stitch_cli as stitch_cli
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as stitch_legacy
from squisher_lightsheet.method8_stitch_register import constraints_from_method8_summary
from squisher_lightsheet.registration_workflow import RegistrationWorkflowOutputs
from squisher_lightsheet.stitch_cli import app


def _summary_row(
    *,
    status: str,
    method8_shift: list[float] | None,
    phase_shift: list[float],
    method8_grad: float | None,
    method8_corr: float | None,
    rejection_reason: str | None = None,
    corr_initial: float | None = 0.20,
    phase_shift_wrap_risk: bool = False,
    fixed_tile: str = "Image_14.000.ome.zarr",
    moving_tile: str = "Image_14.001.ome.zarr",
    seam_axis: str = "x",
) -> dict:
    return {
        "fixed_tile": fixed_tile,
        "moving_tile": moving_tile,
        "seam_axis": seam_axis,
        "status": status,
        "rejection_reason": rejection_reason,
        "fixed_slices_zyx": [[0, 10], [0, 10], [0, 10]],
        "moving_slices_zyx": [[0, 10], [0, 10], [0, 10]],
        "local_translation_zyx": method8_shift,
        "phase_shift_zyx": phase_shift,
        "phase_shift_wrap_risk": phase_shift_wrap_risk,
        "gradient_component_ncc_initial_mean": 0.20,
        "gradient_component_ncc_method8_mean": method8_grad,
        "gradient_component_ncc_phase_mean": 0.50,
        "corr_initial": corr_initial,
        "corr_method8": method8_corr,
        "corr_phase": 0.60,
        "fixed_threshold_mask": {
            "source": {"unmasked_fraction": 0.5},
            "fit": {"unmasked_fraction": 0.5},
        },
        "fixed_content": {"std": 1.0},
        "moving_content": {"std": 1.0},
    }


def test_lightsheet_stitch_exposes_register_subcommand() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "register" in result.stdout

    register_help = CliRunner().invoke(app, ["register", "--help"], env={"COLUMNS": "200"})
    assert register_help.exit_code == 0
    assert "--threshold" in register_help.stdout
    assert "--reviewed-dumb-tiff" not in register_help.stdout
    assert "--fixed-mask-threshold" not in register_help.stdout
    assert "--threshold-method" not in register_help.stdout
    assert "--unmasked" not in register_help.stdout
    assert "--level2-screen" not in register_help.stdout
    assert "--pair" not in register_help.stdout
    assert "--position-json" in register_help.stdout
    assert "required" in register_help.stdout.lower()
    assert "edges without" in register_help.stdout
    assert "[default: 0.1]" in register_help.stdout
    assert "--method8" in register_help.stdout
    assert "Phase correlation with shifted-crop" in register_help.stdout
    assert "recovery is the default" in register_help.stdout


def test_load_tiles_uses_dataset_names_from_position_contract(tmp_path, monkeypatch) -> None:
    zarr_dir = tmp_path / "zarr"
    zarr_dir.mkdir()
    tile_path = zarr_dir / "WGACtrl-405-L2.000.ome.zarr"
    tile_path.mkdir()
    position_json = tmp_path / "positions.json"
    position_json.write_text(
        json.dumps(
            {
                "units": "micrometer",
                "tiles": [
                    {
                        "tile": tile_path.name,
                        "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                        "scale_um": {"z": 0.6, "y": 0.3, "x": 0.3},
                    }
                ]
            }
        )
    )
    array = np.arange(2 * 10 * 20 * 30).reshape(2, 10, 20, 30)
    monkeypatch.setattr(stitch_register, "_load_array", lambda path: array)
    monkeypatch.setattr(stitch_register, "_load_axes", lambda path, array: "CZYX")

    tiles = stitch_register._load_tiles(position_json, zarr_dir, channel=1)

    assert list(tiles) == ["000"]
    assert tiles["000"].tile_name == tile_path.name
    assert tiles["000"].path == tile_path
    assert tiles["000"].shape_zyx.tolist() == [10, 20, 30]
    assert np.array_equal(
        stitch_register._read_tile_crop(tiles["000"], (slice(1, 3), slice(2, 5), slice(3, 7))),
        array[1, 1:3, 2:5, 3:7],
    )


def test_load_tiles_rejects_non_czyx_four_dimensional_input(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    zarr_dir = tmp_path / "zarr"
    zarr_dir.mkdir()
    tile_path = zarr_dir / "sample.000.ome.zarr"
    group = zarr.open_group(str(tile_path), mode="w", zarr_format=3)
    group.create_array(
        "pixels",
        shape=(3, 2, 4, 5),
        chunks=(1, 1, 4, 5),
        dtype="uint16",
        dimension_names=("z", "c", "y", "x"),
    )
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": axis} for axis in ("z", "c", "y", "x")],
                "datasets": [{"path": "pixels"}],
            }
        ],
    }
    position_json = tmp_path / "positions.json"
    position_json.write_text(
        json.dumps(
            {
                "units": "micrometer",
                "tiles": [
                    {
                        "tile": tile_path.name,
                        "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                        "scale_um": {"z": 0.6, "y": 0.3, "x": 0.3},
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="expected ZYX or CZYX"):
        stitch_register._load_tiles(position_json, zarr_dir)


def test_constraints_use_median_method8_shift_per_edge() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 2500.0, "method8": True},
        "rows": [
            _summary_row(
                status="accepted",
                method8_shift=[1.0, 2.0, 3.0],
                phase_shift=[10.0, 10.0, 10.0],
                method8_grad=0.55,
                method8_corr=0.65,
            ),
            _summary_row(
                status="accepted",
                method8_shift=[3.0, 6.0, 9.0],
                phase_shift=[10.0, 10.0, 10.0],
                method8_grad=0.55,
                method8_corr=0.65,
            ),
            _summary_row(
                status="accepted",
                method8_shift=[100.0, 100.0, 100.0],
                phase_shift=[10.0, 10.0, 10.0],
                method8_grad=0.55,
                method8_corr=0.65,
            ),
        ],
    }

    constraints, rejections, sources, weighting = constraints_from_method8_summary(
        summary, tile_index={"000": 0, "001": 1}
    )

    assert not rejections
    assert sources["method8"] == 1
    assert constraints[0].shift_zyx == (3.0, 6.0, 9.0)
    assert constraints[0].source_label == "level0_method8_mask2500_phase_gated_median_n3"
    assert weighting["seams"][0]["accepted_chunk_count"] == 3


def test_phase_fallback_is_downweighted_when_method8_edge_is_unusable() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 3000.0, "method8": True},
        "rows": [
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[4.0, 5.0, 6.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="low_corr_method8",
            ),
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[8.0, 9.0, 10.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="low_corr_method8",
            ),
        ],
    }

    constraints, rejections, sources, _weighting = constraints_from_method8_summary(
        summary,
        tile_index={"000": 0, "001": 1},
        phase_fallback_weight_scale=0.1,
    )

    assert rejections["low_corr_method8"] == 2
    assert sources["phase_fallback"] == 1
    assert constraints[0].shift_zyx == (6.0, 7.0, 8.0)
    assert constraints[0].weight == pytest.approx(0.035)
    assert constraints[0].source_label == "level0_phase_fallback_mask3000_phase_gated_median_n2"


def test_phase_only_constraints_are_full_weight_default() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 3000.0, "method8": False},
        "rows": [
            _summary_row(
                status="accepted",
                method8_shift=None,
                phase_shift=[4.0, 5.0, 6.0],
                method8_grad=None,
                method8_corr=None,
            ),
            _summary_row(
                status="accepted",
                method8_shift=None,
                phase_shift=[8.0, 9.0, 10.0],
                method8_grad=None,
                method8_corr=None,
            ),
        ],
    }

    constraints, rejections, sources, _weighting = constraints_from_method8_summary(
        summary,
        tile_index={"000": 0, "001": 1},
        phase_fallback_weight_scale=0.1,
    )

    assert not rejections
    assert sources["phase"] == 1
    assert constraints[0].shift_zyx == (6.0, 7.0, 8.0)
    assert constraints[0].weight == pytest.approx(0.35)
    assert constraints[0].source_label == "level0_phase_mask3000_phase_gated_median_n2"


def test_phase_constraints_reject_rows_with_sparse_threshold_masks() -> None:
    rows = [
        _summary_row(
            status="rejected",
            method8_shift=None,
            phase_shift=shift,
            method8_grad=None,
            method8_corr=None,
            rejection_reason="fixed_threshold_fit_mask_too_sparse",
        )
        for shift in ([4.0, 5.0, 6.0], [8.0, 9.0, 10.0])
    ]
    summary = {"settings": {"fixed_mask_threshold": 90.0, "method8": False}, "rows": rows}

    constraints, rejections, sources, _weighting = constraints_from_method8_summary(
        summary,
        tile_index={"000": 0, "001": 1},
    )

    assert constraints == []
    assert rejections["fixed_threshold_fit_mask_too_sparse"] == 2
    assert sources["missing_edges_after_fallback"] == 1


def test_phase_constraints_reject_duplicate_valid_attempts_for_one_chunk() -> None:
    direct = _summary_row(
        status="accepted",
        method8_shift=None,
        phase_shift=[4.0, 5.0, 6.0],
        method8_grad=None,
        method8_corr=None,
    )
    direct.update({"z_start": 0, "z_stop": 10, "measurement_mode": "level0_phase_correlation"})
    recovery = _summary_row(
        status="accepted",
        method8_shift=None,
        phase_shift=[4.5, 5.5, 6.5],
        method8_grad=None,
        method8_corr=None,
    )
    recovery.update(
        {"z_start": 0, "z_stop": 10, "measurement_mode": "phase_recovery_prior_shifted_crop"}
    )
    summary = {
        "settings": {"fixed_mask_threshold": 90.0, "method8": False},
        "rows": [direct, recovery],
    }

    with pytest.raises(ValueError, match="Multiple phase-valid attempts for one chunk"):
        constraints_from_method8_summary(summary, tile_index={"000": 0, "001": 1})


def test_sparse_threshold_rows_do_not_seed_phase_recovery_priors() -> None:
    row = _summary_row(
        status="rejected",
        method8_shift=None,
        phase_shift=[4.0, 5.0, 6.0],
        method8_grad=None,
        method8_corr=None,
        rejection_reason="fixed_threshold_fit_mask_too_sparse",
    )

    priors, covered_edges, diagnostics = stitch_register._axis_priors_from_phase_rows(
        [row],
        min_phase_grad=0.24,
        min_phase_corr=0.15,
        min_edges_per_axis=1,
    )

    assert priors == {}
    assert covered_edges == set()
    assert diagnostics == {}


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[1.0, 2.0, 3.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="low_corr_method8",
            ),
            ("source_accepted", "phase_accepted"),
        ),
        (
            _summary_row(
                status="accepted",
                method8_shift=[1.0, 2.0, 3.0],
                phase_shift=[100.0, 100.0, 100.0],
                method8_grad=0.5,
                method8_corr=0.5,
                phase_shift_wrap_risk=True,
            ),
            ("recoverable", "phase_shift_wrap_risk"),
        ),
        (
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[100.0, 100.0, 100.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="phase_shift_wrap_risk",
                phase_shift_wrap_risk=True,
            ),
            ("recoverable", "phase_shift_wrap_risk"),
        ),
        (
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[1.0, 2.0, 3.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="fixed_threshold_fit_mask_too_sparse",
            ),
            ("terminal", "fixed_threshold_fit_mask_too_sparse"),
        ),
        (
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[100.0, 100.0, 100.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="phase_shift_wrap_risk",
                corr_initial=None,
                phase_shift_wrap_risk=True,
            ),
            ("terminal", "initial_corr_not_finite"),
        ),
    ],
)
def test_phase_recovery_classification(row, expected) -> None:
    assert stitch_register._classify_phase_recovery(
        row,
        min_phase_grad=0.24,
        min_phase_corr=0.15,
    ) == expected


def test_phase_recovery_scheduler_skips_input_invalid_chunks(tmp_path, monkeypatch) -> None:
    fixed = stitch_register.TileInfo(
        tile_id="000",
        tile_name="sample.000.ome.zarr",
        path=tmp_path / "sample.000.ome.zarr",
        start_um_zyx=np.zeros(3),
        spacing_um_zyx=np.ones(3),
        shape_zyx=np.full(3, 4),
        channel=1,
    )
    moving = replace(fixed, tile_id="001", tile_name="sample.001.ome.zarr")
    sparse_row = {
        "fixed_tile": fixed.tile_name,
        "moving_tile": moving.tile_name,
        "z_start": 0,
        "z_stop": 4,
        "seam_axis": "x",
        "status": "rejected",
        "rejection_reason": "fixed_threshold_fit_mask_too_sparse",
        "phase_shift_zyx": None,
        "local_translation_zyx": None,
        "gradient_component_ncc_phase_mean": None,
    }
    monkeypatch.setattr(stitch_register, "_load_tiles", lambda *_args, **_kwargs: {"000": fixed, "001": moving})
    monkeypatch.setattr(stitch_register, "_all_adjacent_pairs", lambda _tiles: ["000-001"])
    monkeypatch.setattr(stitch_register, "_measure_z_chunk", lambda **_kwargs: sparse_row.copy())
    monkeypatch.setattr(
        stitch_register,
        "_axis_priors_from_phase_rows",
        lambda *_args, **_kwargs: ({"x": np.zeros(3, dtype=np.float32)}, set(), {}),
    )
    slices = (slice(0, 4), slice(0, 4), slice(0, 4))
    monkeypatch.setattr(
        stitch_register,
        "_crop_bounds_for_pair",
        lambda *_args, **_kwargs: ("x", slices, slices, np.full(3, 4)),
    )
    monkeypatch.setattr(
        stitch_register,
        "_measure_prior_shifted_phase_z_chunk",
        lambda **_kwargs: pytest.fail("known input-invalid chunks must not be scheduled for recovery"),
    )
    position_json = tmp_path / "positions.json"
    position_json.write_text('{"tiles": []}\n')
    zarr_dir = tmp_path / "zarr"
    zarr_dir.mkdir()
    output = tmp_path / "measurements.json"

    stitch_register.measure_method8_zcoverage(
        position_json=position_json,
        zarr_dir=zarr_dir,
        output=output,
        z_chunks=1,
        channel=1,
        fixed_mask_threshold=90.0,
        phase_recovery_min_prior_edges_per_axis=1,
    )

    payload = json.loads(output.read_text())
    assert payload["phase_recovery"]["recovery_attempted_row_count"] == 0
    assert payload["phase_recovery"]["skipped_terminal_rows"] == [
        {
            "pair": "000-001",
            "z_start": 0,
            "z_stop": 4,
            "reason": "fixed_threshold_fit_mask_too_sparse",
        }
    ]
    assert payload["phase_recovery"]["recovery_decision_counts"] == {
        "terminal:fixed_threshold_fit_mask_too_sparse": 1
    }


def test_phase_recovery_scheduler_attempts_only_recoverable_original_chunks(tmp_path, monkeypatch) -> None:
    fixed = stitch_register.TileInfo(
        tile_id="000",
        tile_name="sample.000.ome.zarr",
        path=tmp_path / "sample.000.ome.zarr",
        start_um_zyx=np.zeros(3),
        spacing_um_zyx=np.ones(3),
        shape_zyx=np.array([6, 4, 4]),
        channel=1,
    )
    moving = replace(fixed, tile_id="001", tile_name="sample.001.ome.zarr")

    def initial_row(*, z_start, z_stop, **kwargs):
        row = _summary_row(
            fixed_tile=fixed.tile_name,
            moving_tile=moving.tile_name,
            **kwargs,
        )
        row.update(
            {
                "z_start": z_start,
                "z_stop": z_stop,
                "measurement_mode": "level0_phase_correlation",
            }
        )
        return row

    initial_rows = iter(
        [
            initial_row(
                z_start=0,
                z_stop=2,
                status="accepted",
                method8_shift=None,
                phase_shift=[1.0, 2.0, 3.0],
                method8_grad=None,
                method8_corr=None,
            ),
            initial_row(
                z_start=2,
                z_stop=4,
                status="rejected",
                method8_shift=None,
                phase_shift=[4.0, 5.0, 6.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="fixed_threshold_fit_mask_too_sparse",
            ),
            initial_row(
                z_start=4,
                z_stop=6,
                status="rejected",
                method8_shift=None,
                phase_shift=[100.0, 100.0, 100.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="phase_shift_wrap_risk",
                phase_shift_wrap_risk=True,
            ),
        ]
    )
    monkeypatch.setattr(stitch_register, "_load_tiles", lambda *_args, **_kwargs: {"000": fixed, "001": moving})
    monkeypatch.setattr(stitch_register, "_all_adjacent_pairs", lambda _tiles: ["000-001"])
    monkeypatch.setattr(stitch_register, "_measure_z_chunk", lambda **_kwargs: next(initial_rows))
    monkeypatch.setattr(
        stitch_register,
        "_axis_priors_from_phase_rows",
        lambda *_args, **_kwargs: ({"x": np.zeros(3, dtype=np.float32)}, set(), {}),
    )
    slices = (slice(0, 2), slice(0, 4), slice(0, 4))
    monkeypatch.setattr(
        stitch_register,
        "_crop_bounds_for_pair",
        lambda *_args, **_kwargs: ("x", slices, slices, np.array([2, 4, 4])),
    )
    recovery_calls = []

    def recover(**kwargs):
        recovery_calls.append((kwargs["z_start"], kwargs["z_stop"]))
        row = initial_row(
            z_start=kwargs["z_start"],
            z_stop=kwargs["z_stop"],
            status="rejected",
            method8_shift=None,
            phase_shift=[1.0, 2.0, 3.0],
            method8_grad=None,
            method8_corr=None,
            rejection_reason="phase_recovery_no_method8",
        )
        row.update(
            {
                "measurement_mode": "phase_recovery_prior_shifted_crop",
                "moving_crop_offset_zyx": [0, 0, 0],
            }
        )
        return row

    monkeypatch.setattr(stitch_register, "_measure_prior_shifted_phase_z_chunk", recover)
    position_json = tmp_path / "positions.json"
    position_json.write_text('{"tiles": []}\n')
    zarr_dir = tmp_path / "zarr"
    zarr_dir.mkdir()
    output = tmp_path / "measurements.json"

    stitch_register.measure_method8_zcoverage(
        position_json=position_json,
        zarr_dir=zarr_dir,
        output=output,
        z_chunks=3,
        channel=1,
        fixed_mask_threshold=90.0,
        phase_recovery_min_prior_edges_per_axis=1,
    )

    payload = json.loads(output.read_text())
    assert recovery_calls == [(4, 6)]
    assert payload["phase_recovery"]["recovery_attempted_row_count"] == 1
    assert payload["phase_recovery"]["recovery_accepted_row_count"] == 1
    assert payload["phase_recovery"]["recovery_decision_counts"] == {
        "recoverable:phase_shift_wrap_risk": 1,
        "source_accepted:phase_accepted": 1,
        "terminal:fixed_threshold_fit_mask_too_sparse": 1,
    }


@pytest.mark.parametrize("measurement_mode", ["initial", "recovery"])
@pytest.mark.parametrize(
    ("content_reason", "expected_reason"),
    [
        (None, "fixed_threshold_fit_mask_too_sparse"),
        ("low_center_z_p99", "low_center_z_p99"),
    ],
)
def test_low_content_returns_before_phase_measurement(
    measurement_mode, content_reason, expected_reason, tmp_path, monkeypatch
) -> None:
    class FakeDevice:
        def use(self) -> None:
            return None

    fake_cupy = ModuleType("cupy")
    fake_cupy.bool_ = np.bool_
    fake_cupy.asarray = np.asarray
    fake_cupy.cuda = SimpleNamespace(Device=lambda _device: FakeDevice())
    fake_registration = ModuleType("cucim.skimage.registration")
    fake_registration.phase_cross_correlation = lambda *_args, **_kwargs: pytest.fail(
        "phase correlation must not run for a sparse fit mask"
    )
    fake_ndimage = ModuleType("cupyx.scipy.ndimage")
    fake_ndimage.shift = lambda *_args, **_kwargs: pytest.fail(
        "phase shift must not run for a sparse fit mask"
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cucim.skimage.registration", fake_registration)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", fake_ndimage)

    tile = stitch_register.TileInfo(
        tile_id="000",
        tile_name="sample.000.ome.zarr",
        path=tmp_path / "sample.000.ome.zarr",
        start_um_zyx=np.zeros(3),
        spacing_um_zyx=np.ones(3),
        shape_zyx=np.full(3, 4),
        channel=1,
    )
    moving = replace(tile, tile_id="001", tile_name="sample.001.ome.zarr")
    slices = (slice(0, 4), slice(0, 4), slice(0, 4))
    monkeypatch.setattr(
        stitch_register,
        "_crop_bounds_for_pair",
        lambda *_args, **_kwargs: ("x", slices, slices, np.full(3, 4)),
    )
    monkeypatch.setattr(stitch_register, "_fit_downsample_for_shape", lambda *_args: (1, 1, 1))
    monkeypatch.setattr(stitch_register, "_read_tile_crop", lambda *_args: np.zeros((4, 4, 4)))
    monkeypatch.setattr(
        stitch_register,
        "_robust_norm_and_content_stats_cupy",
        lambda array: (array, {"std": 0.0}),
    )
    monkeypatch.setattr(stitch_register, "_block_mean_downsample_zyx_cupy", lambda array, _factors: array)
    monkeypatch.setattr(stitch_register, "_block_any_downsample_zyx_cupy", lambda mask, _factors: mask)
    monkeypatch.setattr(
        stitch_register,
        "center_z_content_prefilter_reason",
        lambda *_args: (content_reason, {"p99": 100.0, "std": 20.0}, {"p99": 100.0, "std": 20.0}),
        raising=False,
    )
    monkeypatch.setattr(
        stitch_register,
        "_mask_stats",
        lambda _mask: {
            "voxel_count": 1,
            "total_voxels": 64,
            "unmasked_fraction": 1 / 64,
            "masked_fraction": 63 / 64,
        },
    )
    monkeypatch.setattr(
        stitch_register,
        "_corr_gpu",
        lambda *_args, **_kwargs: pytest.fail("correlation must not run for a sparse fit mask"),
    )

    common = {
        "fixed": tile,
        "moving": moving,
        "z_start": 0,
        "z_stop": 4,
        "device": 0,
        "fixed_mask_threshold": 90.0,
        "fixed_mask_min_voxels": 1,
        "fixed_mask_max_masked_fraction": 0.95,
    }
    if measurement_mode == "initial":
        row = stitch_register._measure_z_chunk(
            **common,
            method8=False,
            max_iterations=10,
            ftol=1e-4,
            min_corr=0.15,
            min_grad_ncc=0.24,
            min_phase_grad=0.24,
            min_phase_corr=0.15,
            native_lib_dir=tmp_path,
        )
    else:
        row = stitch_register._measure_prior_shifted_phase_z_chunk(
            **common,
            axis_prior_zyx=np.zeros(3),
        )

    assert row["status"] == "rejected"
    assert row["rejection_reason"] == expected_reason
    assert row["phase_shift_zyx"] is None
    assert row["corr_initial"] is None
    assert row["corr_phase"] is None


def test_phase_fallback_rejects_rows_with_missing_initial_correlation() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 3000.0, "method8": True},
        "rows": [
            _summary_row(
                status="rejected",
                method8_shift=None,
                phase_shift=[4.0, 5.0, 6.0],
                method8_grad=None,
                method8_corr=None,
                rejection_reason="low_corr_method8",
                corr_initial=None,
            )
        ],
    }

    constraints, rejections, sources, _weighting = constraints_from_method8_summary(
        summary,
        tile_index={"000": 0, "001": 1},
    )

    assert constraints == []
    assert rejections["low_corr_method8"] == 1
    assert rejections["initial_corr_not_finite"] == 1
    assert sources["missing_edges_after_fallback"] == 1


def test_phase_fallback_rejects_wrap_risk_rows() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 3000.0, "method8": True},
        "rows": [
            _summary_row(
                status="accepted",
                method8_shift=[1.0, 2.0, 3.0],
                phase_shift=[0.0, 0.0, 40.0],
                method8_grad=0.55,
                method8_corr=0.65,
                phase_shift_wrap_risk=True,
            )
        ],
    }

    constraints, rejections, sources, _weighting = constraints_from_method8_summary(
        summary,
        tile_index={"000": 0, "001": 1},
    )

    assert constraints == []
    assert rejections["phase_shift_wrap_risk"] == 1
    assert sources["missing_edges_after_fallback"] == 1


def test_inverse_variance_downweights_inconsistent_seam_and_preserves_median() -> None:
    rows = []
    for shift in ([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]):
        rows.append(
            _summary_row(
                status="accepted",
                method8_shift=shift,
                phase_shift=shift,
                method8_grad=0.55,
                method8_corr=0.65,
            )
        )
    for shift in ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]):
        rows.append(
            _summary_row(
                status="accepted",
                method8_shift=shift,
                phase_shift=shift,
                method8_grad=0.55,
                method8_corr=0.65,
                fixed_tile="Image_14.001.ome.zarr",
                moving_tile="Image_14.002.ome.zarr",
            )
        )
    summary = {"settings": {"fixed_mask_threshold": 2500.0, "method8": True}, "rows": rows}

    constraints, _rejections, _sources, weighting = constraints_from_method8_summary(
        summary, tile_index={"000": 0, "001": 1, "002": 2}
    )

    assert constraints[0].shift_zyx == pytest.approx((0.05, 0.05, 0.05))
    assert constraints[1].shift_zyx == pytest.approx((1.0, 1.0, 1.0))
    assert constraints[0].weight == pytest.approx(0.4)
    assert constraints[1].weight == pytest.approx(0.002)
    assert weighting["variance_floor_px2"] == 0.03
    assert weighting["seams"][0]["effective_total_variance_px2"] == pytest.approx(0.03)
    assert weighting["seams"][1]["normalized_precision"] == pytest.approx(0.005)


def test_equal_chunk_variances_preserve_gradient_quality_weights() -> None:
    rows = []
    for fixed, moving in (("000", "001"), ("001", "002")):
        for shift in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]):
            rows.append(
                _summary_row(
                    status="accepted",
                    method8_shift=shift,
                    phase_shift=shift,
                    method8_grad=0.55,
                    method8_corr=0.65,
                    fixed_tile=f"Image_14.{fixed}.ome.zarr",
                    moving_tile=f"Image_14.{moving}.ome.zarr",
                )
            )
    summary = {"settings": {"fixed_mask_threshold": 2500.0, "method8": True}, "rows": rows}

    constraints, _rejections, _sources, weighting = constraints_from_method8_summary(
        summary, tile_index={"000": 0, "001": 1, "002": 2}
    )

    assert [constraint.weight for constraint in constraints] == pytest.approx([0.4, 0.4])
    assert [seam["normalized_precision"] for seam in weighting["seams"]] == [1.0, 1.0]


def test_sparse_seam_uses_observed_ninetieth_percentile_variance_prior() -> None:
    rows = []
    for fixed, moving, shifts in (
        ("000", "001", ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])),
        ("001", "002", ([0.0, 0.0, 0.0], [3.0, 3.0, 3.0])),
        ("002", "003", ([1.0, 1.0, 1.0],)),
    ):
        for shift in shifts:
            rows.append(
                _summary_row(
                    status="accepted",
                    method8_shift=shift,
                    phase_shift=shift,
                    method8_grad=0.55,
                    method8_corr=0.65,
                    fixed_tile=f"Image_14.{fixed}.ome.zarr",
                    moving_tile=f"Image_14.{moving}.ome.zarr",
                )
            )
    summary = {"settings": {"fixed_mask_threshold": 2500.0, "method8": True}, "rows": rows}

    _constraints, _rejections, _sources, weighting = constraints_from_method8_summary(
        summary, tile_index={"000": 0, "001": 1, "002": 2, "003": 3}
    )

    # Total sample variances are 1.5 and 13.5 px²; their 90th percentile is 12.3 px².
    assert weighting["sparse_variance_prior_px2"] == pytest.approx(12.3)
    sparse = next(seam for seam in weighting["seams"] if seam["accepted_chunk_count"] == 1)
    assert sparse["variance_source"] == "observed_90th_percentile_prior"
    assert sparse["total_variance_px2"] is None
    assert sparse["effective_total_variance_px2"] == pytest.approx(12.3)


def test_inverse_variance_weighting_requires_one_multi_chunk_seam() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 2500.0, "method8": True},
        "rows": [
            _summary_row(
                status="accepted",
                method8_shift=[1.0, 2.0, 3.0],
                phase_shift=[1.0, 2.0, 3.0],
                method8_grad=0.55,
                method8_corr=0.65,
            )
        ],
    }

    with pytest.raises(ValueError, match="at least two accepted chunks"):
        constraints_from_method8_summary(summary, tile_index={"000": 0, "001": 1})


def test_mvs_edge_quality_uses_constraint_weight() -> None:
    summary = {
        "settings": {"fixed_mask_threshold": 2500.0, "method8": True},
        "rows": [
            _summary_row(
                status="accepted",
                method8_shift=[1.0, 2.0, 3.0],
                phase_shift=[1.0, 2.0, 3.0],
                method8_grad=0.95,
                method8_corr=0.95,
            ),
            _summary_row(
                status="accepted",
                method8_shift=[1.1, 2.1, 3.1],
                phase_shift=[1.1, 2.1, 3.1],
                method8_grad=0.95,
                method8_corr=0.95,
            ),
        ],
    }
    constraints, _rejections, _sources, _weighting = constraints_from_method8_summary(
        summary, tile_index={"000": 0, "001": 1}
    )

    constraint = replace(constraints[0], weight=0.2, gradient_component_ncc_after=0.95)

    assert stitch_legacy.seam_graph_edge_quality([constraint]) == pytest.approx(0.2)


def test_method8_reads_base_dataset_path_from_ngff_metadata(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "tile.ome.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    expected = root.create_array(
        "scale0",
        shape=(1, 2, 3, 4),
        chunks=(1, 1, 3, 4),
        dtype="uint16",
        dimension_names=("c", "z", "y", "x"),
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{"datasets": [{"path": "scale0"}]}],
    }

    actual = stitch_register._load_array(path)

    assert actual.path == expected.path


def test_register_uses_threshold_in_canonical_measurement_path(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_measure_method8_zcoverage(**kwargs):
        captured["method8_output"] = kwargs["output"]
        captured["threshold"] = kwargs["fixed_mask_threshold"]
        captured["method8"] = kwargs["method8"]
        return kwargs["output"]

    def fake_optimize_positions_from_method8_summary(**kwargs):
        captured["optimized_from"] = kwargs["method8_summary"]
        return stitch_register.Method8RegistrationOutputs(
            method8_summary=kwargs["method8_summary"],
            optimized_positions=tmp_path / "positions.json",
            diagnostics=tmp_path / "diagnostics.json",
            constraints_jsonl=tmp_path / "constraints.jsonl",
            tile_corrections=tmp_path / "corrections.json",
        )

    monkeypatch.setattr(stitch_register, "measure_method8_zcoverage", fake_measure_method8_zcoverage)
    monkeypatch.setattr(
        stitch_register,
        "optimize_positions_from_method8_summary",
        fake_optimize_positions_from_method8_summary,
    )
    output_dir = tmp_path / "out"
    stitch_register.register_level0_phase_recovery(
        position_json=tmp_path / "positions.json",
        zarr_dir=tmp_path / "zarr",
        output_dir=output_dir,
        fixed_mask_threshold=2500.0,
    )

    assert captured["threshold"] == 2500.0
    assert captured["method8"] is False
    assert captured["method8_output"] == output_dir / "registration.measurements.json"
    assert captured["optimized_from"] == captured["method8_output"]


def test_register_uses_existing_summary_in_requested_output_dir(tmp_path, monkeypatch) -> None:
    captured = {}
    summary = tmp_path / "summary.json"
    summary.write_text('{"settings": {"fixed_mask_threshold": 2500.0}, "rows": []}\n')

    def fake_optimize_positions_from_method8_summary(**kwargs):
        captured.update(kwargs)
        return stitch_register.Method8RegistrationOutputs(
            method8_summary=kwargs["method8_summary"],
            optimized_positions=kwargs["output_dir"] / "positions.json",
            diagnostics=kwargs["output_dir"] / "diagnostics.json",
            constraints_jsonl=kwargs["output_dir"] / "constraints.jsonl",
            tile_corrections=kwargs["output_dir"] / "corrections.json",
        )

    monkeypatch.setattr(
        stitch_register,
        "optimize_positions_from_method8_summary",
        fake_optimize_positions_from_method8_summary,
    )
    output_dir = tmp_path / "out"
    stitch_register.register_level0_phase_recovery(
        position_json=tmp_path / "positions.json",
        zarr_dir=tmp_path / "zarr",
        output_dir=output_dir,
        method8_summary=summary,
    )

    assert captured["method8_summary"] == summary
    assert captured["output_dir"] == output_dir


def test_cli_register_uses_threshold_as_human_review_gate(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_registration_workflow(**kwargs):
        captured.update(kwargs)
        return RegistrationWorkflowOutputs(
            threshold_record=tmp_path / "threshold.json",
            measurement_summary=tmp_path / "summary.json",
            optimized_positions=tmp_path / "positions.json",
            diagnostics=tmp_path / "diagnostics.json",
            constraints_jsonl=tmp_path / "constraints.jsonl",
            tile_corrections=tmp_path / "corrections.json",
            canonical_positions=tmp_path / "registration.positions.json",
            registration_json=tmp_path / "registration.json",
        )

    monkeypatch.setattr(stitch_cli, "run_registration_workflow", fake_run_registration_workflow)

    positions = tmp_path / "positions.json"
    positions.write_text("{}")
    zarr_dir = tmp_path / "zarr"
    zarr_dir.mkdir()
    result = CliRunner().invoke(
        app,
        [
            "register",
            "--position-json",
            str(positions),
            "--zarr-dir",
            str(zarr_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--threshold",
            "2500",
            "--allow-disconnected",
        ],
    )

    assert result.exit_code == 0
    assert captured["threshold"] == 2500.0
    assert "reviewed_dumb_tiff" not in captured
    assert captured["method8"] is False
    assert captured["allow_disconnected"] is True

    missing_gate = CliRunner().invoke(
        app,
        [
            "register",
            "--position-json",
            str(positions),
            "--zarr-dir",
            str(zarr_dir),
            "--output-dir",
            str(tmp_path / "missing-gate"),
        ],
    )
    assert missing_gate.exit_code == 2

    parameters = inspect.signature(stitch_cli.register).parameters
    assert "reviewed_dumb_tiff" not in parameters
    assert parameters["threshold"].default is inspect.Parameter.empty
    assert {"level2_screen", "threshold_method", "unmasked", "pair"}.isdisjoint(parameters)


def test_optimization_refuses_to_remove_stale_outputs(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    summary = tmp_path / "summary.json"
    summary.write_text('{"settings": {"fixed_mask_threshold": 2500.0}, "rows": []}\n')
    paths = stitch_register._optimized_output_paths(output_dir)
    for path in paths.values():
        path.write_text("stale")

    constraint = stitch_register.BoundaryConstraint(
        fixed=0,
        moving=1,
        pair=(0, 1),
        axis="x",
        patch_index=0,
        shift_zyx=(1.0, 2.0, 3.0),
        weight=1.0,
        correlation_before=0.1,
        correlation_after=0.2,
        improvement=0.1,
        fixed_nonzero_fraction=0.5,
        moving_nonzero_fraction=0.5,
        fixed_std=1.0,
        moving_std=1.0,
        accepted=True,
    )

    monkeypatch.setattr(
        stitch_register,
        "_load_position_tiles",
        lambda *_args, **_kwargs: (
            {},
            ["Image_14.000.ome.zarr", "Image_14.001.ome.zarr"],
            {"000": 0},
            np.ones(3),
            [],
        ),
    )
    monkeypatch.setattr(
        stitch_register,
        "constraints_from_method8_summary",
        lambda *_args, **_kwargs: ([constraint], {}, {"method8": 1}, {"seams": []}),
    )

    def fail_solver(*_args, **_kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(
        stitch_register.stitch_legacy, "solve_tile_corrections_with_multiview_stitcher", fail_solver
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        stitch_register.optimize_positions_from_method8_summary(
            method8_summary=summary,
            position_json=tmp_path / "positions.json",
            zarr_dir=tmp_path / "zarr",
            output_dir=output_dir,
        )

    assert all(path.read_text() == "stale" for path in paths.values())


def test_optimization_writes_inverse_variance_diagnostics(tmp_path, monkeypatch) -> None:
    position_json = tmp_path / "positions.json"
    zarr_dir = tmp_path / "zarr"
    output_dir = tmp_path / "out"
    zarr_dir.mkdir()
    position_payload = {
        "tiles": [
            {
                "tile": f"Image_14.{index:03d}.ome.zarr",
                "translation_um": {"z": 0.0, "y": 0.0, "x": float(index)},
            }
            for index in range(3)
        ]
    }
    position_json.write_text(json.dumps(position_payload) + "\n")
    rows = []
    for fixed, moving, shifts in (
        ("000", "001", ([0.0, 0.0, 0.0], [0.1, 0.1, 0.1])),
        ("001", "002", ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])),
    ):
        for shift in shifts:
            rows.append(
                _summary_row(
                    status="accepted",
                    method8_shift=shift,
                    phase_shift=shift,
                    method8_grad=0.55,
                    method8_corr=0.65,
                    fixed_tile=f"Image_14.{fixed}.ome.zarr",
                    moving_tile=f"Image_14.{moving}.ome.zarr",
                )
            )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.level0_phase_recovery_measurements.v1",
                "position_json": str(position_json.resolve()),
                "zarr_dir": str(zarr_dir.resolve()),
                "settings": {"fixed_mask_threshold": 2500.0, "method8": True, "channel": 0},
                "rows": rows,
            }
        )
        + "\n"
    )
    tile_names = [f"Image_14.{index:03d}.ome.zarr" for index in range(3)]
    monkeypatch.setattr(
        stitch_register,
        "_load_position_tiles",
        lambda *_args, **_kwargs: (
            position_payload,
            tile_names,
            {"000": 0, "001": 1, "002": 2},
            np.ones(3),
            [],
        ),
    )
    monkeypatch.setattr(
        stitch_register.stitch_legacy,
        "solve_tile_corrections_with_multiview_stitcher",
        lambda _tiles, constraints, _settings: ([(0.0, 0.0, 0.0)] * 3, constraints, 0),
    )
    monkeypatch.setattr(
        stitch_register.stitch_legacy,
        "anchor_connected_tiles",
        lambda *_args, **_kwargs: {0, 1, 2},
    )

    outputs = stitch_register.optimize_positions_from_method8_summary(
        method8_summary=summary,
        position_json=position_json,
        zarr_dir=zarr_dir,
        output_dir=output_dir,
    )

    diagnostics = json.loads(outputs.diagnostics.read_text())
    weighting = diagnostics["constraint_weighting"]
    assert weighting["formula"].startswith("gradient_quality * normalized_inverse")
    assert weighting["variance_floor_px2"] == 0.03
    assert [seam["final_weight"] for seam in weighting["seams"]] == pytest.approx([0.4, 0.002])


def test_pyproject_exposes_lightsheet_stitch_entrypoint() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()

    assert 'lightsheet-stitch = "squisher_lightsheet.stitch_cli:main"' in pyproject
