from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import zarr

from squisher_lightsheet import fused_fixed_qc as qc


def test_plane_support_bounds_preserves_fixed_528_geometry() -> None:
    bounds = qc.plane_support_bounds(
        np.zeros(3),
        np.eye(3),
        np.zeros(3),
        fixed_z_l0=264,
        canvas_shape=(200, 200),
    )

    assert bounds == (0, 132, 0, 132)


def test_recovery_qc_rejects_other_window_geometry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported window_shape_zyx"):
        qc._validate_summary_geometry(
            {"window_shape_zyx": [256, 528, 528]}, tmp_path / "summary.json"
        )


def test_interpolated_qc_handles_no_recovered_windows(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"window_shape_zyx": [528, 528, 528], "windows": []})
    )
    output = tmp_path / "empty.png"

    result = qc.render_interpolated_window_qc(
        summary_path=summary,
        output=output,
        moving_label="514",
        fixed_label="561",
        dataset_label="sample",
    )

    assert result == output.resolve()
    assert Image.open(output).size == (900, 80)


@pytest.mark.parametrize("native_method", ["method6", "method10-mattes"])
def test_interpolated_qc_selects_attempts_for_native_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_method: str
) -> None:
    prefix = f"interpolated_{native_method.replace('-', '_')}"
    attempts = [
        f"{prefix}_spatial_recovery",
        f"{prefix}_linear_mapping_recovery",
        "native_method10_mattes_from_recovery_initializer",
    ]
    windows = []
    for index, attempt in enumerate(attempts):
        path = tmp_path / f"window-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "moving_tile": f"tile-{index}.ome.zarr",
                    "moving_start_l0_zyx": [0, 0, 0],
                    "selected_attempt": attempt,
                    "interpolated_fallback_promotion": {"source_good_window_count": 2},
                    "linear_mapping_recovery": {"source_window_count": 3},
                }
            )
        )
        windows.append({"level0_json": str(path), "selected_attempt": attempt})
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "window_shape_zyx": [528, 528, 528],
                "cache_config": {"native_method": native_method},
                "fixed_fused": str(tmp_path / "fixed.ome.zarr"),
                "windows": windows,
            }
        )
    )
    monkeypatch.setattr(qc.zarr, "open_group", lambda *_args, **_kwargs: {"2": object()})
    rendered = []

    def render(row, _fixed, _moving_arrays):
        rendered.append(row["selected_attempt"])
        return Image.new("RGB", (184, 184)), 0

    monkeypatch.setattr(qc, "_render_interpolated_window", render)
    output = tmp_path / "contact-sheet.png"

    qc.render_interpolated_window_qc(
        summary_path=summary,
        output=output,
        moving_label="638",
        fixed_label="514",
        columns=2,
    )

    assert rendered == attempts[:2]
    assert Image.open(output).size == (2 * qc.CELL_WIDTH, qc.HEADER_HEIGHT + qc.CELL_HEIGHT)


def test_interpolated_qc_rejects_unsupported_native_method(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "window_shape_zyx": [528, 528, 528],
                "cache_config": {"native_method": "method8"},
                "windows": [{"level0_json": "window.json"}],
            }
        )
    )

    with pytest.raises(ValueError, match="Unsupported native method"):
        qc.render_interpolated_window_qc(
            summary_path=summary,
            output=tmp_path / "contact-sheet.png",
            moving_label="638",
            fixed_label="514",
        )


def test_prefusion_qc_writes_hole_metrics(tmp_path: Path, monkeypatch) -> None:
    fixed_volume = tmp_path / "fixed.ome.zarr"
    root = zarr.open_group(fixed_volume, mode="w")
    fixed = root.create_array("2", shape=(3, 6, 8), chunks=(1, 6, 8), dtype="uint16")
    fixed[:] = 100
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "window_shape_zyx": [528, 528, 528],
                "fixed_fused": str(fixed_volume),
                "windows": [],
            }
        )
    )
    monkeypatch.setattr(
        qc,
        "_render_plane",
        lambda _rows, _z, shape: (
            np.ones(shape, dtype=np.float32),
            np.ones(shape, dtype=np.uint16),
            0,
        ),
    )

    output = qc.render_prefusion_qc(
        summaries=[("514", summary)],
        fixed_volume=fixed_volume,
        fixed_threshold=50,
        output_dir=tmp_path / "qc",
    )

    assert output.is_file()
    report = json.loads((output.parent / "prefusion-hole-review.json").read_text())
    assert report["artifact_type"] == "lightsheet.fused_fixed_prefusion_qc.v1"
    assert report["channels"][0]["uncovered_tissue_percent"] == 0.0
    assert report["combined"] is None


def test_prefusion_qc_reports_union_coverage(tmp_path: Path, monkeypatch) -> None:
    fixed_volume = tmp_path / "fixed.ome.zarr"
    root = zarr.open_group(fixed_volume, mode="w")
    fixed = root.create_array("2", shape=(3, 6, 8), chunks=(1, 6, 8), dtype="uint16")
    fixed[:] = 100
    summaries = []
    for label_name in ("514-L", "514-R"):
        summary = tmp_path / f"{label_name}.json"
        summary.write_text(
            json.dumps(
                {
                    "window_shape_zyx": [528, 528, 528],
                    "fixed_fused": str(fixed_volume),
                    "windows": [],
                }
            )
        )
        summaries.append((label_name, summary))

    coverages = []
    for column_slice in (slice(0, 4), slice(4, 8)):
        coverage = np.zeros((6, 8), dtype=np.uint16)
        coverage[:, column_slice] = 1
        coverages.append(coverage)
    coverage_iterator = iter(coverages)
    monkeypatch.setattr(
        qc,
        "_render_plane",
        lambda _rows, _z, shape: (
            np.ones(shape, dtype=np.float32),
            next(coverage_iterator),
            1,
        ),
    )

    output = qc.render_prefusion_qc(
        summaries=summaries,
        fixed_volume=fixed_volume,
        fixed_threshold=50,
        output_dir=tmp_path / "qc",
    )

    report = json.loads((output.parent / "prefusion-hole-review.json").read_text())
    assert [row["uncovered_tissue_percent"] for row in report["channels"]] == [50.0, 50.0]
    assert report["combined"]["labels"] == ["514-L", "514-R"]
    assert report["combined"]["uncovered_tissue_percent"] == 0.0
    assert (output.parent / "prefusion-hole-map-combined.png").is_file()


def test_window_image_scorer_ranks_aligned_transform_above_shifted(tmp_path: Path) -> None:
    from scipy.ndimage import gaussian_filter

    size = qc.WINDOW // qc.MOVING_XY_FACTOR
    texture = gaussian_filter(np.random.default_rng(0).normal(size=(size, size)), 1.5)
    texture = (1000.0 + 4000.0 * (texture - texture.min()) / np.ptp(texture)).astype("uint16")
    fixed_root = zarr.open_group(tmp_path / "fixed.zarr", mode="w")
    fixed = fixed_root.create_array(
        "2", shape=(qc.WINDOW // qc.FIXED_LEVEL_FACTOR, size, size), chunks=(8, size, size), dtype="uint16"
    )
    fixed[:] = np.broadcast_to(texture, fixed.shape)
    moving_root = zarr.open_group(tmp_path / "moving.zarr", mode="w")
    moving = moving_root.create_array(
        "2", shape=(1, qc.WINDOW, size, size), chunks=(1, 32, size, size), dtype="uint16"
    )
    moving[:] = np.broadcast_to(texture, moving.shape)
    row = {
        "fixed_start_zyx": [0, 0, 0],
        "moving_start_l0_zyx": [0, 0, 0],
        "moving_path": str(tmp_path / "moving.zarr"),
        "moving_channel": 0,
    }
    identity = np.concatenate([np.eye(3).ravel(), [0.0, 0.0, 0.0]])
    shifted = np.concatenate([np.eye(3).ravel(), [0.0, 0.0, 4.0 * qc.MOVING_XY_FACTOR]])

    aligned_scores, shifted_scores = qc.WindowImageScorer(tmp_path / "fixed.zarr", 400.0, "linear").plane_scores(
        row, [identity, shifted]
    )

    assert aligned_scores == pytest.approx([1.0, 1.0, 1.0], abs=1e-3)
    assert all(shifted < aligned - 0.2 for aligned, shifted in zip(aligned_scores, shifted_scores, strict=True))


def test_window_image_scorer_returns_none_without_fixed_tissue(tmp_path: Path) -> None:
    size = qc.WINDOW // qc.MOVING_XY_FACTOR
    fixed_root = zarr.open_group(tmp_path / "fixed.zarr", mode="w")
    fixed_root.create_array(
        "2", shape=(qc.WINDOW // qc.FIXED_LEVEL_FACTOR, size, size), chunks=(8, size, size), dtype="uint16"
    )
    moving_root = zarr.open_group(tmp_path / "moving.zarr", mode="w")
    moving = moving_root.create_array(
        "2", shape=(1, qc.WINDOW, size, size), chunks=(1, 32, size, size), dtype="uint16"
    )
    moving[:] = 1000
    row = {
        "fixed_start_zyx": [0, 0, 0],
        "moving_start_l0_zyx": [0, 0, 0],
        "moving_path": str(tmp_path / "moving.zarr"),
        "moving_channel": 0,
    }
    identity = np.concatenate([np.eye(3).ravel(), [0.0, 0.0, 0.0]])

    scorer = qc.WindowImageScorer(tmp_path / "fixed.zarr", 400.0, "linear")

    assert scorer.plane_scores(row, [identity, identity]) == [[], []]
    assert scorer.residual_shifts(row, identity) == []


def test_window_image_scorer_residual_shift_recovers_known_offset(tmp_path: Path) -> None:
    from scipy.ndimage import gaussian_filter

    size = qc.WINDOW // qc.MOVING_XY_FACTOR
    texture = gaussian_filter(np.random.default_rng(1).normal(size=(size, size)), 1.5)
    texture = (1000.0 + 4000.0 * (texture - texture.min()) / np.ptp(texture)).astype("uint16")
    fixed = zarr.open_group(tmp_path / "fixed.zarr", mode="w").create_array(
        "2", shape=(qc.WINDOW // qc.FIXED_LEVEL_FACTOR, size, size), chunks=(8, size, size), dtype="uint16"
    )
    fixed[:] = np.broadcast_to(texture, fixed.shape)
    moving = zarr.open_group(tmp_path / "moving.zarr", mode="w").create_array(
        "2", shape=(1, qc.WINDOW, size, size), chunks=(1, 32, size, size), dtype="uint16"
    )
    moving[:] = np.broadcast_to(texture, moving.shape)
    row = {
        "fixed_start_zyx": [0, 0, 0],
        "moving_start_l0_zyx": [0, 0, 0],
        "moving_path": str(tmp_path / "moving.zarr"),
        "moving_channel": 0,
    }
    shifted = np.concatenate([np.eye(3).ravel(), [0.0, 0.0, 5.0 * qc.MOVING_XY_FACTOR]])
    scorer = qc.WindowImageScorer(tmp_path / "fixed.zarr", 400.0, "linear")

    shifts = scorer.residual_shifts(row, shifted)
    aligned, displaced = scorer.plane_scores(row, [np.concatenate([np.eye(3).ravel(), np.zeros(3)]), shifted])

    assert len(shifts) == 3
    assert all(shift.confident for shift in shifts)
    assert {shift.shift_yx for shift in shifts} == {(0.0, -5.0)}
    assert len(aligned) == len(displaced) == 3
    assert all(a > d for a, d in zip(aligned, displaced, strict=True))


def test_window_image_scorer_scores_in_the_fit_intensity_space(tmp_path: Path) -> None:
    from scipy.ndimage import gaussian_filter

    size = qc.WINDOW // qc.MOVING_XY_FACTOR
    texture = gaussian_filter(np.random.default_rng(2).normal(size=(size, size)), 1.5)
    fixed_values = np.exp(np.log(20.0) + np.log(150.0) * (texture - texture.min()) / np.ptp(texture))
    fixed = zarr.open_group(tmp_path / "fixed.zarr", mode="w").create_array(
        "2", shape=(qc.WINDOW // qc.FIXED_LEVEL_FACTOR, size, size), chunks=(8, size, size), dtype="uint16"
    )
    fixed[:] = np.broadcast_to(fixed_values.astype("uint16"), fixed.shape)
    moving = zarr.open_group(tmp_path / "moving.zarr", mode="w").create_array(
        "2", shape=(1, qc.WINDOW, size, size), chunks=(1, 32, size, size), dtype="uint16"
    )
    # Linear in log space, strongly nonlinear in raw intensity.
    moving[:] = np.broadcast_to(np.expm1(1.35 * np.log1p(fixed_values.astype("uint16"))).astype("uint16"), moving.shape)
    row = {
        "fixed_start_zyx": [0, 0, 0],
        "moving_start_l0_zyx": [0, 0, 0],
        "moving_path": str(tmp_path / "moving.zarr"),
        "moving_channel": 0,
    }
    identity = np.concatenate([np.eye(3).ravel(), np.zeros(3)])

    (log_scores,) = qc.WindowImageScorer(tmp_path / "fixed.zarr", 10.0, "log1p").plane_scores(row, [identity])
    (linear_scores,) = qc.WindowImageScorer(tmp_path / "fixed.zarr", 10.0, "linear").plane_scores(row, [identity])

    assert log_scores == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)
    assert all(log > linear + 0.003 for log, linear in zip(log_scores, linear_scores, strict=True))
    with pytest.raises(ValueError, match="intensity transform"):
        qc.WindowImageScorer(tmp_path / "fixed.zarr", 400.0, "identity")
