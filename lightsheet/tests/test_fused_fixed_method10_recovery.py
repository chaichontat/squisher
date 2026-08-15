from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import polar
from scipy.spatial.transform import Rotation
import zarr


def _load_recovery():
    path = Path(__file__).parents[1] / "scripts" / "recover_fused_fixed_method10_outliers.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trial():
    path = Path("/home/chaichontat/nvme/lightsheet/scripts/try_fused_fixed_method8.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recovery_row() -> dict[str, object]:
    return {
        "status": "accepted",
        "rejection_reason": None,
        "selected_attempt": "native_method10_mattes_from_phase_primed_initializer",
        "selected_local_matrix_zyx": np.eye(3).tolist(),
        "selected_local_translation_zyx": [0.0, 0.0, 0.0],
        "fixed_start_zyx": [0, 0, 0],
        "fixed_stop_zyx": [10, 10, 10],
        "moving_start_l0_zyx": [0, 0, 0],
        "moving_stop_l0_zyx": [10, 10, 10],
    }


def test_detects_and_interpolates_spatial_transform_outlier() -> None:
    recovery = _load_recovery()
    points = np.asarray([[z, y, x] for z in range(5) for y in range(4) for x in range(2)], dtype=np.float64)
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None, ...], len(points), axis=0)
    translations = np.column_stack(
        [
            0.2 * points[:, 0],
            -0.1 * points[:, 1],
            0.3 * points[:, 2] + 0.05 * points[:, 0],
        ]
    )
    transforms = np.concatenate([matrices.reshape(len(points), 9), translations], axis=1)
    outlier_index = 21
    transforms[outlier_index, 11] += 80.0

    outliers, _scores, _threshold = recovery.detect_outliers(
        points,
        transforms,
        np.asarray([352, 352, 352]),
        maximum_px=5.0,
    )

    assert np.flatnonzero(outliers).tolist() == [outlier_index]
    normalized, _origin, _spacing = recovery._normalized_points(points)
    good = np.flatnonzero(~outliers)
    predicted = recovery.predict_decomposed_affine_field(
        normalized[good],
        transforms[good],
        target_points=normalized[outlier_index : outlier_index + 1],
    )[0]
    assert np.allclose(predicted, np.concatenate([np.eye(3).ravel(), translations[outlier_index]]), atol=0.15)


def test_detects_affine_outlier_above_pixel_tolerance() -> None:
    recovery = _load_recovery()
    rng = np.random.default_rng(4)
    points = np.asarray([[z, y, 0] for z in range(5) for y in range(4)], dtype=np.float64)
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None, ...], 20, axis=0)
    for index in range(len(matrices)):
        perturbation = rng.normal(0.0, 2e-4, size=(3, 3))
        matrices[index] += (perturbation + perturbation.T) / 2.0
    outlier_index = 7
    matrices[outlier_index, 1, 1] += 0.015
    transforms = np.concatenate([matrices.reshape(len(matrices), 9), np.zeros((len(matrices), 3))], axis=1)

    outliers, scores, threshold = recovery.detect_outliers(
        points,
        transforms,
        np.asarray([1000, 1000, 1000]),
        maximum_px=5.0,
    )

    assert np.flatnonzero(outliers).tolist() == [outlier_index]
    assert scores[outlier_index] > threshold


def test_detects_more_than_twenty_outliers(monkeypatch) -> None:
    recovery = _load_recovery()
    points = np.column_stack([np.arange(30), np.zeros((30, 2))])
    transforms = np.repeat(np.r_[np.eye(3).ravel(), [0.0, 0.0, 0.0]][None, :], 30, axis=0)
    transforms[:25, 9] = np.arange(25, 0, -1) * 10.0
    monkeypatch.setattr(
        recovery,
        "predict_decomposed_affine_field",
        lambda _points, _transforms, *, target_points: np.repeat(
            np.r_[np.eye(3).ravel(), [0.0, 0.0, 0.0]][None, :], len(target_points), axis=0
        ),
    )

    outliers, _scores, _threshold = recovery.detect_outliers(
        points,
        transforms,
        np.asarray([10, 10, 10]),
        maximum_px=5.0,
    )

    assert np.flatnonzero(outliers).tolist() == list(range(25))


def test_linear_field_prediction_is_global_least_squares_model() -> None:
    recovery = _load_recovery()
    points = np.asarray([[z, 0, 0] for z in range(4)], dtype=np.float64)
    values = np.asarray([[0], [0], [0], [10]], dtype=np.float64)
    target = np.asarray([[1.5, 0, 0]], dtype=np.float64)

    predicted, model = recovery.fit_linear_field(points, values, target_points=target)

    expected = (
        np.r_[1.0, target[0, 0]]
        @ np.linalg.lstsq(
            np.column_stack([np.ones(len(points)), points[:, 0]]),
            values,
            rcond=None,
        )[0]
    )
    np.testing.assert_allclose(predicted[0], expected)
    assert model["modeled_spatial_axes"] == ["z"]


def test_decomposed_affine_mean_wraps_rotations_and_averages_stretch() -> None:
    recovery = _load_recovery()
    rotations = Rotation.from_euler("z", [179.0, -179.0], degrees=True).as_matrix()
    stretches = np.asarray(
        [
            [[1.0, 0.01, 0.0], [0.01, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.03, 0.0], [0.03, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )

    averaged = recovery.decomposed_affine_mean(rotations @ stretches)
    averaged_rotation, averaged_stretch = polar(averaged)

    angle = Rotation.from_matrix(averaged_rotation).magnitude()
    assert np.rad2deg(angle) == pytest.approx(180.0, abs=1e-6)
    np.testing.assert_allclose(averaged_stretch, np.mean(stretches, axis=0), atol=1e-12)


def test_json_safe_converts_nonfinite_legacy_values_to_null() -> None:
    recovery = _load_recovery()

    assert recovery._json_safe({"finite": 1.0, "values": [float("nan"), float("inf")]}) == {
        "finite": 1.0,
        "values": [None, None],
    }


def test_native_method_comes_from_required_cache_config() -> None:
    recovery = _load_recovery()

    assert recovery._native_method_from_summary(
        {"native_method": "method8", "cache_config": {"native_method": "method6"}}
    ) == "method6"
    with pytest.raises(KeyError, match="cache_config"):
        recovery._native_method_from_summary({"native_method": "method6"})


def test_empty_score_summary_is_json_safe() -> None:
    recovery = _load_recovery()

    assert recovery._score_summary(np.asarray([np.nan, np.nan])) == {"median": None, "maximum": None}


def test_two_source_outlier_threshold_remains_finite() -> None:
    recovery = _load_recovery()

    _outliers, _scores, threshold = recovery.detect_outliers(
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
        np.asarray([np.r_[np.eye(3).ravel(), [0, 0, 0]], np.r_[np.eye(3).ravel(), [1, 0, 0]]]),
        np.asarray([480, 480, 480]),
        maximum_px=5.0,
    )

    assert threshold == 5.0


def test_final_linear_filter_excludes_clustered_nonlinear_displacements() -> None:
    recovery = _load_recovery()
    points = np.asarray([[z, y, x] for z in range(7) for y in range(3) for x in range(2)], dtype=np.float64)
    displacements = np.column_stack(
        [
            2.0 + 0.1 * points[:, 0],
            -1.0 + 0.2 * points[:, 1],
            3.0 + 0.3 * points[:, 2] + 0.05 * points[:, 0],
        ]
    )
    corrupt = (points[:, 0] >= 5) & (points[:, 1] != 2)
    displacements[corrupt] += np.asarray([40.0, 80.0, 30.0])

    outliers, _scores, _threshold, median_residual = recovery.detect_linear_displacement_outliers(
        points,
        displacements,
        outlier_mad=4.0,
        minimum_outlier_um=3.0,
        loss_scale_um=2.0,
    )

    assert np.array_equal(outliers, corrupt)
    assert median_residual < 0.1


def test_final_linear_filter_ignores_roundoff_on_constant_axis() -> None:
    recovery = _load_recovery()
    points = np.asarray(
        [[z, y, 100.1] for z in range(6) for y in range(3)],
        dtype=np.float64,
    )
    assert np.std(points[:, 2]) > 0.0
    assert np.ptp(points[:, 2]) == 0.0
    displacements = np.column_stack(
        [
            2.0 + 0.1 * points[:, 0],
            -1.0 + 0.2 * points[:, 1],
            3.0 + 0.05 * points[:, 0],
        ]
    )

    outliers, _scores, _threshold, median_residual = recovery.detect_linear_displacement_outliers(
        points,
        displacements,
        outlier_mad=4.0,
        minimum_outlier_um=3.0,
        loss_scale_um=2.0,
    )

    assert not np.any(outliers)
    assert median_residual < 1e-10


def test_final_linear_filter_rejects_legacy_interpolated_recovery() -> None:
    recovery = _load_recovery()
    row = _recovery_row()
    row["selected_attempt"] = recovery._interpolated_recovery_attempt("method10-mattes")

    assert recovery._final_linear_input_exclusion(row) == {
        "reason": "unvalidated_interpolated_recovery",
        "score_um": None,
    }


def test_final_linear_filter_retains_native_registration() -> None:
    recovery = _load_recovery()

    assert recovery._final_linear_input_exclusion(_recovery_row()) is None


def test_linear_field_fit_smooths_translation_and_shear_together() -> None:
    recovery = _load_recovery()
    points = np.asarray([[z, y, x] for z in range(4) for y in range(3) for x in range(2)], dtype=np.float64)
    values = np.column_stack(
        [
            2.0 + 0.2 * points[:, 0],
            -1.0 + 0.1 * points[:, 1],
            3.0 - 0.3 * points[:, 2],
            0.001 + 0.0001 * points[:, 0],
            -0.002 + 0.0002 * points[:, 1],
            0.003 - 0.0003 * points[:, 2],
        ]
    )

    predicted, model = recovery.fit_linear_field(points, values)

    np.testing.assert_allclose(predicted, values, atol=1e-12)
    assert model["modeled_spatial_axes"] == ["z", "y", "x"]


def test_final_linear_smoothing_predicts_only_transform_outliers() -> None:
    recovery = _load_recovery()
    points = np.asarray([[z, y, x] for z in range(4) for y in range(3) for x in range(2)], dtype=np.float64)
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None, ...], len(points), axis=0)
    translations = np.column_stack(
        [
            0.2 * points[:, 0],
            -0.1 * points[:, 1],
            0.3 * points[:, 2] + 0.05 * points[:, 0],
        ]
    )
    transforms = np.concatenate([matrices.reshape(len(points), 9), translations], axis=1)
    values = np.column_stack(
        [
            2.0 + 0.2 * points[:, 0],
            -1.0 + 0.1 * points[:, 1],
            3.0 - 0.3 * points[:, 2],
            0.001 + 0.0001 * points[:, 0],
            -0.002 + 0.0002 * points[:, 1],
            0.003 - 0.0003 * points[:, 2],
        ]
    )
    outlier_index = 17
    transforms[outlier_index, 11] += 50.0
    corrupted_values = values.copy()
    corrupted_values[outlier_index, :3] += [20.0, 30.0, 40.0]

    outliers, predictions, _model, _scores, _threshold = recovery.outlier_only_linear_predictions(
        points=points,
        transforms=transforms,
        values=corrupted_values,
        shape_zyx=np.asarray([480, 480, 480]),
        maximum_outlier_px=5.0,
    )

    assert np.flatnonzero(outliers).tolist() == [outlier_index]
    np.testing.assert_array_equal(predictions[~outliers], corrupted_values[~outliers])
    np.testing.assert_allclose(predictions[outlier_index], values[outlier_index], atol=1e-12)


def test_selected_transform_update_keeps_all_representations_consistent() -> None:
    recovery = _load_recovery()
    row = _recovery_row()
    matrix = np.asarray([[1.0, 0.01, 0.0], [0.01, 1.0, -0.02], [0.0, -0.02, 1.0]])
    translation = np.asarray([1.0, -2.0, 3.0])

    recovery._set_selected_transform(
        row,
        matrix=matrix,
        translation=translation,
        attempt="linear_interpolated_method10_outlier_replacement",
    )

    expected_pull = recovery.output_to_input_from_model(matrix, translation, (10, 10, 10))
    expected_global = recovery._global_forward_from_local(row, matrix, translation)
    np.testing.assert_allclose(row["selected_local_matrix_zyx"], matrix)
    np.testing.assert_allclose(row["selected_local_translation_zyx"], translation)
    np.testing.assert_allclose(row["local_matrix_zyx"], matrix)
    np.testing.assert_allclose(row["local_translation_zyx"], translation)
    np.testing.assert_allclose(row["selected_fixed_fused_l0_to_moving_l0_pull_matrix_zyx"], expected_pull[0])
    np.testing.assert_allclose(row["selected_fixed_fused_l0_to_moving_l0_pull_offset_zyx"], expected_pull[1])
    np.testing.assert_allclose(row["selected_moving_l0_to_fixed_fused_l0_matrix_zyx"], expected_global[0])
    np.testing.assert_allclose(row["selected_moving_l0_to_fixed_fused_l0_offset_zyx"], expected_global[1])
    assert row["selected_corr_refined"] is None
    assert row["selected_gradient_component_ncc_mean"] is None
    assert row["selected_gradient_component_ncc_refined"] is None


@pytest.mark.parametrize(
    ("rerun_payload", "expected_reason"),
    [
        (None, "native_worker_output_missing"),
        ({"status": "rejected", "rejection_reason": "native_rejected"}, "native_rerun_rejected"),
        (
            {
                "status": "accepted",
                "rejection_reason": None,
                "selected_attempt": "native_method10_mattes_from_recovery_initializer",
                "selected_local_matrix_zyx": np.eye(3).tolist(),
                "selected_local_translation_zyx": [20.0, 0.0, 0.0],
            },
            "native_rerun_spatial_outlier",
        ),
    ],
    ids=["missing-output", "native-rejection", "spatial-outlier"],
)
def test_failed_refit_retains_accepted_original(tmp_path, rerun_payload, expected_reason) -> None:
    recovery = _load_recovery()
    original_path = tmp_path / "input" / "window.json"
    output_dir = tmp_path / "output"
    output_path = output_dir / "window_json" / original_path.name
    row = _recovery_row()
    if rerun_payload is not None:
        recovery._write_json(output_path, {**row, **rerun_payload})
    task = {
        "row": row,
        "original_path": original_path,
        "transform": np.r_[np.eye(3).ravel(), [0.0, 0.0, 0.0]],
        "provenance": {"maximum_refit_displacement_px": 1.0},
    }

    result = recovery._finalize_native_rerun(
        task,
        output_dir=output_dir,
        native_method="method10-mattes",
    )
    recovered = recovery._read_json(output_path)

    assert result["status"] == "accepted"
    assert recovered["status"] == "accepted"
    assert recovered["rejection_reason"] is None
    assert recovered["selected_attempt"] == row["selected_attempt"]
    assert recovered["outlier_recovery"]["native_rerun_selected"] is False
    assert recovered["outlier_recovery"]["retained_original_reason"] == expected_reason


def test_method6_recovery_accepts_its_native_recovery_attempt(tmp_path: Path) -> None:
    recovery = _load_recovery()
    original_path = tmp_path / "input" / "window.json"
    output_dir = tmp_path / "output"
    output_path = output_dir / "window_json" / original_path.name
    row = _recovery_row()
    recovery._write_json(
        output_path,
        {
            **row,
            "selected_attempt": "native_method6_from_recovery_initializer",
        },
    )
    task = {
        "row": row,
        "original_path": original_path,
        "transform": np.r_[np.eye(3).ravel(), [0.0, 0.0, 0.0]],
        "provenance": {"maximum_refit_displacement_px": 1.0},
    }

    result = recovery._finalize_native_rerun(
        task,
        output_dir=output_dir,
        native_method="method6",
    )

    assert result["status"] == "accepted"
    assert result["native_rerun_selected"] is True


def test_rerun_validation_uses_fixed_window_shape(tmp_path: Path) -> None:
    recovery = _load_recovery()
    original_path = tmp_path / "input" / "window.json"
    output_dir = tmp_path / "output"
    output_path = output_dir / "window_json" / original_path.name
    row = {**_recovery_row(), "moving_stop_l0_zyx": [100, 100, 100]}
    recovery._write_json(
        output_path,
        {
            **row,
            "selected_attempt": "native_method6_from_recovery_initializer",
            "selected_local_matrix_zyx": (np.eye(3) * 1.5).tolist(),
        },
    )

    result = recovery._finalize_native_rerun(
        {
            "row": row,
            "original_path": original_path,
            "transform": np.r_[np.eye(3).ravel(), [0.0, 0.0, 0.0]],
            "provenance": {"maximum_refit_displacement_px": 10.0},
        },
        output_dir=output_dir,
        native_method="method6",
    )

    assert result["status"] == "accepted"


def test_rerun_all_preserves_spatial_outlier_reason(tmp_path: Path, monkeypatch, capsys) -> None:
    recovery = _load_recovery()
    cache = {"native_method": "method6"}
    paths = []
    for index, translation_x in enumerate((0.0, 1.0, 2.0, 100.0)):
        path = tmp_path / f"window{index}.json"
        row = {
            **_recovery_row(),
            "moving_tile": "sample.001",
            "moving_start_l0_zyx": [index * 10, 0, 0],
            "moving_stop_l0_zyx": [index * 10 + 10, 10, 10],
            "selected_attempt": "native_method6_from_phase_primed_initializer",
            "selected_local_translation_zyx": [0.0, 0.0, translation_x],
            "cache_config": cache,
        }
        path.write_text(json.dumps(row))
        paths.append(path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "cache_config": cache,
                "windows": [{"tile": "sample.001", "level0_json": str(path)} for path in paths],
            }
        )
    )
    reasons = []
    original_provenance = recovery._recovery_provenance

    def capture_reason(*args, **kwargs):
        reasons.append(kwargs["reason"])
        return original_provenance(*args, **kwargs)

    monkeypatch.setattr(recovery, "_recovery_provenance", capture_reason)

    recovery.run(
        SimpleNamespace(
            exclude_nonlinear_only=False,
            smooth_retained_linear=False,
            summary=summary_path,
            tile_filter=None,
                rerun_all_eligible=True,
                maximum_outlier_px=5.0,
                maximum_refit_displacement_px=10.0,
                adjacency_json=None,
            dry_run=True,
        )
    )
    capsys.readouterr()

    assert reasons.count("spatial_transform_outlier") == 1


@pytest.mark.parametrize(
    "rerun_grad_ncc",
    [
        0.40,
        0.39,
        None,
    ],
)
def test_nonimproving_spatial_refit_retains_accepted_original(
    tmp_path: Path,
    rerun_grad_ncc: float | None,
) -> None:
    recovery = _load_recovery()
    original_path = tmp_path / "input" / "window.json"
    output_dir = tmp_path / "output"
    output_path = output_dir / "window_json" / original_path.name
    row = {**_recovery_row(), "selected_gradient_component_ncc_mean": 0.40}
    recovery._write_json(
        output_path,
        {
            **row,
            "selected_attempt": "native_method6_from_recovery_initializer",
            "selected_gradient_component_ncc_mean": rerun_grad_ncc,
        },
    )
    task = {
        "row": row,
        "original_path": original_path,
        "transform": np.r_[np.eye(3).ravel(), [0.0, 0.0, 0.0]],
        "provenance": {
            "reason": "spatial_transform_outlier",
            "maximum_refit_displacement_px": 5.0,
        },
    }

    result = recovery._finalize_native_rerun(
        task,
        output_dir=output_dir,
        native_method="method6",
    )
    recovered = recovery._read_json(output_path)

    assert result["status"] == "accepted"
    assert recovered["status"] == "accepted"
    assert recovered["rejection_reason"] is None
    assert recovered["selected_attempt"] == row["selected_attempt"]
    assert recovered["selected_gradient_component_ncc_mean"] == 0.40
    assert recovered["outlier_recovery"]["native_rerun_selected"] is False
    assert recovered["outlier_recovery"]["retained_original_reason"] == (
        "native_rerun_grad_ncc_not_improved"
    )
    assert recovered["outlier_recovery"]["original_gradient_component_ncc_mean"] == 0.40
    assert recovered["outlier_recovery"]["native_rerun_gradient_component_ncc_mean"] == rerun_grad_ncc


def test_final_linear_filter_retains_fit_when_support_is_insufficient(
    tmp_path: Path, monkeypatch
) -> None:
    recovery = _load_recovery()
    window_path = tmp_path / "window.json"
    moving_position_path = tmp_path / "positions.json"
    fixed_path = tmp_path / "fixed.zarr"
    output_dir = tmp_path / "output"
    row = {**_recovery_row(), "moving_tile": "sample.001"}
    recovery._write_json(window_path, row)
    recovery._write_json(
        moving_position_path,
        {
            "tiles": [
                {
                    "tile": "sample.001",
                    "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
                    "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
                }
            ]
        },
    )
    zarr.open_group(fixed_path, mode="w")
    monkeypatch.setattr(
        recovery.ngff,
        "scale_translation",
        lambda _group: (["z", "y", "x"], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], True, True),
    )
    summary_path = tmp_path / "summary.json"
    recovery._write_json(
        summary_path,
        {
            "moving_position": str(moving_position_path),
            "fixed_fused": str(fixed_path),
            "windows": [{"level0_json": str(window_path)}],
        },
    )

    result_path = recovery.filter_final_linear_outliers(
        SimpleNamespace(
            summary=summary_path,
            output_dir=output_dir,
            minimum_linear_samples=8,
            linear_loss_scale_um=2.0,
            outlier_mad=4.0,
            minimum_linear_outlier_um=3.0,
            maximum_linear_outlier_um=8.0,
            maximum_linear_median_residual_um=4.0,
        )
    )
    result = recovery._read_json(result_path)
    retained = recovery._read_json(Path(result["windows"][0]["level0_json"]))

    assert retained["status"] == "accepted"
    assert retained["rejection_reason"] is None
    assert result["final_linear_filter"]["excluded_count"] == 0
    assert result["final_linear_filter"]["tiles"][0]["status"] == (
        "skipped_insufficient_support"
    )


def test_adjacent_translation_initializer_collapses_duplicate_window_coordinates() -> None:
    recovery = _load_recovery()
    points = np.asarray([[0, 0, 0], [0, 0, 0], [2, 0, 0], [2, 0, 0]], dtype=np.float64)
    transforms = np.stack(
        [
            np.r_[np.eye(3).ravel(), [0, 0, 0]],
            np.r_[np.eye(3).ravel(), [2, 0, 0]],
            np.r_[np.eye(3).ravel(), [2, 0, 0]],
            np.r_[np.eye(3).ravel(), [4, 0, 0]],
        ]
    )

    result = recovery.adjacent_translation_initializer(
        points,
        transforms,
        np.asarray([1, 0, 0]),
    )

    np.testing.assert_allclose(result[:9].reshape(3, 3), np.eye(3))
    np.testing.assert_allclose(result[9:], [2, 0, 0])


def test_tile_adjacency_comes_from_registration_candidate_pairs(tmp_path: Path) -> None:
    recovery = _load_recovery()
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps({"pair_summaries": {"004-010": {}, "004-005": {}}}))

    adjacency = recovery._load_tile_adjacency(path)

    assert adjacency["004"] == {"005", "010"}
    assert adjacency["010"] == {"004"}


def test_recovery_uses_persistent_sweep_worker(monkeypatch) -> None:
    recovery = _load_recovery()
    parsed: list[str] = []
    parser = SimpleNamespace(parse_args=lambda arguments: parsed.extend(arguments) or arguments)
    monkeypatch.setattr(recovery, "_load_sweep_module", lambda: SimpleNamespace(build_parser=lambda: parser))
    cache = {
        "native_method": "method6",
        "fit_intensity_transform": "linear",
        "mattes_fixed_shear": False,
        "starting_affine_matrix_zyx": np.eye(3).ravel().tolist(),
        "fit_downsample_zyx": [1, 1, 1],
        "moving_channel": 1,
        "native_lib_dir": "/native",
        "mattes_bins": 50,
        "mattes_samples": 400000,
        "ftol": 1e-4,
        "max_iterations": 300,
        "phase_upsample_factor": 10,
        "min_corr": 0.15,
        "min_grad_ncc": 0.24,
        "fixed_mask_threshold": 500,
        "fixed_mask_level": 2,
        "fixed_mask_min_voxels": 256,
        "fixed_mask_max_masked_fraction": 0.95,
    }
    sweep_kwargs = {
        "summary": {
            "moving_position": "/moving.json",
            "moving_source_position": "/source.json",
            "fixed_fused": "/fixed.zarr",
            "core_shape_zyx": [320, 640, 640],
            "window_shape_zyx": [352, 704, 640],
        },
        "output_dir": Path("/output"),
        "initializer_dir": Path("/initializers"),
        "devices": (1,),
        "target_count": 711,
        "resume": True,
    }

    recovery._persistent_sweep_args(cache=cache, **sweep_kwargs)

    assert parsed[parsed.index("--level0-initializer") + 1] == "window-interpolated"
    assert parsed[parsed.index("--native-method") + 1] == "method6"
    assert parsed[parsed.index("--fit-intensity-transform") + 1] == "linear"
    assert "--starting-affine-matrix-zyx" not in parsed
    assert parsed[parsed.index("--workers") + 1] == "1"
    assert parsed[parsed.index("--max-tasks-per-worker") + 1] == "711"
    assert "--resume" in parsed
    assert "--no-resume" not in parsed

    for required_setting in ("fit_intensity_transform", "mattes_fixed_shear"):
        incomplete_cache = {key: value for key, value in cache.items() if key != required_setting}
        with pytest.raises(KeyError, match=required_setting):
            recovery._persistent_sweep_args(cache=incomplete_cache, **sweep_kwargs)


def test_persistent_sweep_worker_is_picklable_for_spawn() -> None:
    recovery = _load_recovery()

    sweep = recovery._load_sweep_module()

    pickle.dumps(sweep._init_cuda_worker)


def test_recovery_initializer_bypasses_phase_priming(monkeypatch) -> None:
    trial = _load_trial()
    monkeypatch.setattr(
        trial,
        "estimate_translation_gpu",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("phase correlation should not run")),
    )

    delta = trial._phase_priming_delta(
        SimpleNamespace(skip_phase_priming=True, phase_upsample_factor=10),
        object(),
        object(),
    )

    assert delta.tolist() == [0.0, 0.0, 0.0]
