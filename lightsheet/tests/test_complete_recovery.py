from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from squisher_lightsheet import complete_recovery as subject
from squisher_lightsheet.fused_fixed_qc import PlaneShift


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _native_fixture(
    tmp_path: Path, *, native_method: str = "method6"
) -> tuple[Path, Path]:
    moving = tmp_path / "moving.positions.json"
    fixed = tmp_path / "fixed.ome.zarr"
    fixed.mkdir()
    _write(fixed / ".zattrs", {"ome": {"multiscales": []}})
    _write(moving, {"tiles": []})
    row = tmp_path / "native" / "window_json" / "window-001.json"
    _write(
        row,
        {
            "moving_tile": "sample.001.ome.zarr",
            "status": "rejected",
            "rejection_reason": "fixed_mask_empty",
            "selected_attempt": None,
        },
    )
    summary = tmp_path / "native" / subject.SUMMARY_NAME
    _write(
        summary,
        {
            "moving_position": str(moving),
            "fixed_fused": str(fixed),
            "cache_config": {"native_method": native_method},
            "quality_settings": {"min_grad_ncc": 0.2, "fixed_mask_threshold": 400.0},
            "outlier_recovery": {
                "artifact_type": (
                    "lightsheet.fused_fixed_native_spatial_outlier_recovery.v1"
                )
            },
            "aggregate": {"window_count": 1},
            "windows": [
                {
                    "level0_json": str(row),
                    "status": "rejected",
                    "rejection_reason": "fixed_mask_empty",
                    "selected_attempt": None,
                }
            ],
        },
    )
    return summary, row


def _fake_policy_stages(native_summary: Path, workspace: Path, _settings):
    native = subject.read_json_object(native_summary)
    final_dir = workspace / "04-linear-recovery"
    window_dir = final_dir / "window_json"
    window_dir.mkdir(parents=True)
    source_row = Path(native["windows"][0]["level0_json"])
    final_row = window_dir / source_row.name
    _write(final_row, subject.read_json_object(source_row))
    native["windows"][0]["level0_json"] = str(final_row)
    final_summary = final_dir / subject.SUMMARY_NAME
    _write(final_summary, native)
    reports = []
    for index, stage in enumerate(subject.POLICY_ORDER, start=1):
        report = workspace / f"{index:02d}-{stage}" / "report.json"
        _write(report, {"stage": stage, "input_summary": str(final_summary)})
        reports.append((stage, report))
    return final_summary, reports


@pytest.mark.parametrize("native_method", ["method6", "method10-mattes"])
def test_complete_recovery_publishes_once_and_reuses_exact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_method: str
) -> None:
    native_summary, _row = _native_fixture(tmp_path, native_method=native_method)
    calls = []

    def fake_stages(*args):
        calls.append(args[0])
        return _fake_policy_stages(*args)

    monkeypatch.setattr(subject, "_run_policy_stages", fake_stages)
    output = tmp_path / "complete"
    first = subject.complete_recovery(native_summary=native_summary, output_dir=output)
    second = subject.complete_recovery(native_summary=native_summary, output_dir=output)

    assert first == second == output / subject.SUMMARY_NAME
    assert len(calls) == 1
    manifest = subject.read_json_object(output / "manifest.json")
    assert manifest["artifact_type"] == "lightsheet.complete_recovery_manifest.v1"
    assert set(manifest["report_sha256"]) == {
        f"{stage}.json" for stage in subject.POLICY_ORDER
    }
    assert subject.read_json_object(first)["complete_recovery"]["policy_order"] == list(
        subject.POLICY_ORDER
    )


def test_extra_or_corrupt_published_file_prevents_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_summary, _row = _native_fixture(tmp_path)
    monkeypatch.setattr(subject, "_run_policy_stages", _fake_policy_stages)
    output = tmp_path / "complete"
    subject.complete_recovery(native_summary=native_summary, output_dir=output)
    _write(output / "window_json" / "extra.json", {})

    with pytest.raises(ValueError, match="identity mismatch"):
        subject.complete_recovery(native_summary=native_summary, output_dir=output)


def test_complete_recovery_rejects_non_method6_checkpoint(tmp_path: Path) -> None:
    native_summary, _row = _native_fixture(tmp_path)
    payload = subject.read_json_object(native_summary)
    payload["cache_config"]["native_method"] = "method8"
    _write(native_summary, payload)

    with pytest.raises(ValueError, match="requires native Method 6 or Method 10 Mattes"):
        subject.complete_recovery(
            native_summary=native_summary,
            output_dir=tmp_path / "complete",
        )


class _FakeScorer:
    """Plane scores keyed by candidate x translation; fixed residual shifts."""

    def __init__(self, planes: dict[float, list[float]] | None = None, residual=()) -> None:
        self.planes = planes or {}
        self.residual = list(residual)
        self.compared: list[list[float]] = []
        self.checked: list[float] = []

    def plane_scores(self, _row, transforms):
        self.compared.append([float(transform[11]) for transform in transforms])
        return [self.planes[float(transform[11])] for transform in transforms]

    def residual_shifts(self, _row, transform):
        self.checked.append(float(transform[11]))
        return self.residual

    def aggregate_residual_shift(self, _row, _transform):
        return None


def _shift(y: float, x: float, *, peak: float = 0.6, second: float = 0.2) -> PlaneShift:
    return PlaneShift(shift_yx=(y, x), peak_ncc=peak, second_peak_ncc=second, at_search_edge=False)


def _spatial_row(*, reason: str | None, interpolated_x: float = 3.0, native_x: float = 7.0) -> dict:
    row = {
        **_transform_row(
            status="rejected", reason=subject.SPATIAL_TARGET_REASON, attempt=None, x=5.0
        ),
        "selected_corr_refined": 0.5,
        "selected_gradient_component_ncc_mean": 0.4,
        "outlier_recovery": {
            "interpolated_initializer_local_matrix_zyx": np.eye(3).tolist(),
            "interpolated_initializer_local_translation_zyx": [0.0, 0.0, interpolated_x],
            "source_good_window_json": ["a.json", "b.json"],
            "native_rerun_rejection_reason": reason,
        },
    }
    if reason is not None:
        row["method8_attempts"] = [
            _attempt("level2_initializer_only", reason="quality_gate_grad_ncc", grad=0.05, x=interpolated_x),
            _attempt("native_method6_from_recovery_initializer", reason=reason, grad=0.15, x=native_x),
        ]
    return row


@pytest.mark.parametrize("native_method", ["method6", "method10-mattes"])
def test_spatial_promotion_requires_recorded_support_and_clears_fit_metrics(
    native_method: str,
) -> None:
    scorer = _FakeScorer()

    promoted = subject._promote_spatial_fallback(
        _spatial_row(reason="low_gradient_ncc"),
        native_method=native_method,
        scorer=scorer,
        image_ncc_margin=0.02,
        maximum_unverified_rerun_departure_px=60.0,
    )

    assert promoted["status"] == "accepted"
    assert promoted["selected_attempt"] == (
        f"interpolated_{native_method.replace('-', '_')}_spatial_recovery"
    )
    assert promoted["selected_corr_refined"] is None
    assert promoted["selected_gradient_component_ncc_mean"] is None
    promotion = promoted["interpolated_fallback_promotion"]
    assert promotion["source_good_window_count"] == 2
    assert promotion["residual"]["verdict"] == "unverified"
    assert scorer.compared == []


def test_spatial_promotion_rejects_insufficient_support() -> None:
    row = _spatial_row(reason=None)
    row["outlier_recovery"]["source_good_window_json"] = ["only-one.json"]

    with pytest.raises(ValueError, match="Insufficient interpolation support"):
        subject._promote_spatial_fallback(
            row, native_method="method6", scorer=_FakeScorer(), image_ncc_margin=0.02, maximum_unverified_rerun_departure_px=60.0
        )


@pytest.mark.parametrize(
    ("reason", "expected_status"),
    [
        ("native_rerun_attempt_mismatch", "accepted"),
        ("native_rerun_grad_ncc_not_improved", "rejected"),
    ],
)
def test_spatial_promotion_recovers_only_native_attempt_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str, expected_status: str
) -> None:
    summary, row_path = _native_fixture(tmp_path, native_method="method10-mattes")
    monkeypatch.setattr(subject.recovery, "window_image_scorer", lambda _summary: _FakeScorer())
    row = {
        **_transform_row(
            status="rejected",
            reason=reason,
            attempt=None,
            x=5.0,
        ),
        "outlier_recovery": {
            "interpolated_initializer_local_matrix_zyx": np.eye(3).tolist(),
            "interpolated_initializer_local_translation_zyx": [0.0, 0.0, 5.0],
            "source_good_window_json": ["neighbor-a.json", "neighbor-b.json"],
            "native_rerun_status": "accepted",
            "native_rerun_rejection_reason": None,
        },
    }
    _write(row_path, row)
    payload = subject.read_json_object(summary)
    payload["windows"][0].update(
        rejection_reason=row["rejection_reason"], selected_attempt=None
    )
    _write(summary, payload)

    result = subject._promote_spatial_fallbacks(
        summary, tmp_path / "promoted", image_ncc_margin=0.02, maximum_unverified_rerun_departure_px=60.0
    )
    promoted_summary = subject.read_json_object(result)
    promoted = subject.read_json_object(
        Path(promoted_summary["windows"][0]["level0_json"])
    )

    assert promoted["status"] == expected_status
    if expected_status == "accepted":
        assert promoted["selected_attempt"] == "interpolated_method10_mattes_spatial_recovery"
    else:
        assert promoted["rejection_reason"] == reason
    assert promoted_summary["interpolated_fallback_promotion"]["promoted_count"] == (
        expected_status == "accepted"
    )


def _attempt(name: str, *, reason: str | None, grad: float, x: float) -> dict:
    return {
        "name": name,
        "status": "rejected" if reason else "accepted",
        "rejection_reason": reason,
        "corr_refined": 0.1,
        "gradient_component_ncc_mean": grad,
        "gradient_component_ncc_refined": {"z": grad, "y": grad, "x": grad},
        "local_matrix_zyx": np.eye(3).tolist(),
        "local_translation_zyx": [0.0, 0.0, x],
    }


@pytest.mark.parametrize(
    ("reason", "interpolated_planes", "native_planes", "selects_rerun"),
    [
        ("quality_gate_grad_ncc", [0.3, 0.4], [0.5, 0.6], True),
        ("quality_gate_corr", [0.3, 0.4], [0.5, 0.6], True),
        ("quality_gate_grad_ncc", [0.3, 0.4], [0.5, 0.35], False),
        ("quality_gate_grad_ncc", [0.1], [0.9], False),
        ("quality_gate_grad_ncc", [0.30, 0.40], [0.31, 0.41], False),
    ],
)
def test_spatial_promotion_keeps_rejected_rerun_only_when_every_plane_favors_it(
    reason: str,
    interpolated_planes: list[float],
    native_planes: list[float],
    selects_rerun: bool,
) -> None:
    scorer = _FakeScorer({3.0: interpolated_planes, 7.0: native_planes})

    promoted = subject._promote_spatial_fallback(
        _spatial_row(reason=reason), native_method="method6", scorer=scorer, image_ncc_margin=0.02,
        maximum_unverified_rerun_departure_px=60.0,
    )

    promotion = promoted["interpolated_fallback_promotion"]
    assert promoted["status"] == "accepted"
    assert scorer.compared == [[3.0, 7.0]]
    assert scorer.checked == [7.0 if selects_rerun else 3.0]
    if selects_rerun:
        assert promotion["selected_candidate"] == "native_rerun"
        assert promoted["selected_attempt"] == "native_method6_from_recovery_initializer"
        assert promoted["selected_local_translation_zyx"] == [0.0, 0.0, 7.0]
        assert promoted["selected_gradient_component_ncc_mean"] == 0.15
    else:
        assert promotion["selected_candidate"] == "interpolated"
        assert promoted["selected_attempt"] == "interpolated_method6_spatial_recovery"
        assert promoted["selected_local_translation_zyx"] == [0.0, 0.0, 3.0]
        assert promoted["selected_gradient_component_ncc_mean"] is None


def test_spatial_promotion_never_compares_affine_rejected_rerun() -> None:
    scorer = _FakeScorer()

    promoted = subject._promote_spatial_fallback(
        _spatial_row(reason="affine_scale_out_of_bounds"),
        native_method="method6",
        scorer=scorer,
        image_ncc_margin=0.02,
        maximum_unverified_rerun_departure_px=60.0,
    )

    assert scorer.compared == []
    assert promoted["selected_attempt"] == "interpolated_method6_spatial_recovery"


@pytest.mark.parametrize(
    ("residual", "status", "verdict"),
    [
        ([_shift(0.0, 4.0), _shift(0.0, 4.5)], "rejected", "contradicted"),
        ([_shift(0.0, 4.0)], "accepted", "unverified"),
        ([_shift(0.0, 4.0), _shift(0.0, -4.0)], "accepted", "unverified"),
        ([_shift(0.0, 1.0), _shift(1.0, 0.0)], "accepted", "verified"),
        ([_shift(0.0, 6.0, second=0.55), _shift(0.0, 6.0, second=0.55)], "accepted", "unverified"),
    ],
)
def test_spatial_promotion_withholds_image_contradicted_transform(
    residual: list[PlaneShift], status: str, verdict: str
) -> None:
    promoted = subject._promote_spatial_fallback(
        _spatial_row(reason="low_gradient_ncc"),
        native_method="method6",
        scorer=_FakeScorer(residual=residual),
        image_ncc_margin=0.02,
        maximum_unverified_rerun_departure_px=60.0,
    )

    assert promoted["status"] == status
    assert promoted["interpolated_fallback_promotion"]["residual"]["verdict"] == verdict
    if status == "rejected":
        assert promoted["rejection_reason"] == subject.CONTRADICTED_INTERPOLATION_REASON
        assert promoted["selected_attempt"] is None


def test_spatial_promotion_rejects_baseline_that_is_not_the_interpolation() -> None:
    row = _spatial_row(reason="quality_gate_grad_ncc")
    row["method8_attempts"][0]["local_translation_zyx"] = [0.0, 0.0, 9.0]

    with pytest.raises(ValueError, match="does not match the interpolated initializer"):
        subject._promote_spatial_fallback(
            row, native_method="method6", scorer=_FakeScorer(), image_ncc_margin=0.02, maximum_unverified_rerun_departure_px=60.0
        )


def _transform_row(
    *, status: str, reason: str | None, attempt: str | None, x: float
) -> dict:
    return {
        "moving_tile": "sample.001.ome.zarr",
        "moving_start_l0_zyx": [0, 0, int(x)],
        "fixed_start_zyx": [0, 0, int(x)],
        "fixed_stop_zyx": [10, 10, int(x) + 10],
        "selected_local_matrix_zyx": np.eye(3).tolist(),
        "selected_local_translation_zyx": [0.0, 0.0, x],
        "status": status,
        "rejection_reason": reason,
        "selected_attempt": attempt,
    }


@pytest.mark.parametrize("native_method", ["method6", "method10-mattes"])
@pytest.mark.parametrize(
    ("scores", "replaced"),
    [
        (([0.3, 0.3], [0.6, 0.6]), True),
        (([0.6, 0.6], [0.3, 0.3]), False),
        (([0.3], [0.6]), False),
        (([], []), False),
    ],
)
@pytest.mark.parametrize("local_3d", [False, True])
def test_linear_recovery_uses_only_native_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_method: str,
    scores: tuple[list[float], list[float]],
    replaced: bool,
    local_3d: bool,
) -> None:
    monkeypatch.setattr(
        subject.recovery,
        "window_image_scorer",
        lambda _summary: SimpleNamespace(plane_scores=lambda _row, _transforms: list(scores)),
    )
    rows = [
        _transform_row(
            status="accepted",
            reason=None,
            attempt=("local_3d_affine_refinement" if local_3d else
                     f"native_{native_method.replace('-', '_')}_from_phase_primed_initializer"),
            x=0.0,
        ) | {"cache_config": {"native_method": native_method, "local_3d_refinement": local_3d}},
        _transform_row(
            status="accepted",
            reason=None,
            attempt="level2_initializer_phase",
            x=10.0,
        ),
        _transform_row(
            status="accepted",
            reason=None,
            attempt=f"interpolated_{native_method.replace('-', '_')}_spatial_recovery",
            x=20.0,
        ),
        _transform_row(
            status="rejected",
            reason=subject.LINEAR_TARGET_REASON,
            attempt="linear_rejected",
            x=5.0,
        )
        | {"selected_local_translation_zyx": [0.0, 0.0, 9.0]},
        _transform_row(
            status="rejected",
            reason="linear_mapping_no_consensus",
            attempt="no_consensus",
            x=7.0,
        ),
    ]
    compact = []
    for index, row in enumerate(rows):
        path = tmp_path / "input" / f"window-{index}.json"
        _write(path, row)
        compact.append(
            {
                "level0_json": str(path),
                "status": row["status"],
                "rejection_reason": row["rejection_reason"],
                "selected_attempt": row["selected_attempt"],
            }
        )
    summary = tmp_path / "input" / subject.SUMMARY_NAME
    _write(
        summary,
        {
            "windows": compact,
            "aggregate": {"window_count": len(rows)},
            "cache_config": {"native_method": native_method},
        },
    )

    result = subject._recover_linear_mapping_rejections(
        summary, tmp_path / "output", image_ncc_margin=0.02
    )
    payload = subject.read_json_object(result)
    recovered = subject.read_json_object(Path(payload["windows"][3]["level0_json"]))
    untouched = subject.read_json_object(Path(payload["windows"][4]["level0_json"]))

    assert recovered["status"] == "accepted"
    assert recovered["linear_mapping_recovery"]["source_window_count"] == 2
    if replaced:
        assert recovered["selected_attempt"] == (
            f"interpolated_{native_method.replace('-', '_')}_linear_mapping_recovery"
        )
        assert recovered["selected_local_translation_zyx"] == pytest.approx([0.0, 0.0, 5.0])
    else:
        assert recovered["selected_attempt"] == "linear_rejected"
        assert recovered["selected_local_translation_zyx"] == [0.0, 0.0, 9.0]
        assert recovered["linear_mapping_recovery"]["selected_candidate"] == "fitted"
        assert payload["linear_mapping_recovery"]["fitted_retained_count"] == 1
    assert untouched["rejection_reason"] == "linear_mapping_no_consensus"


def test_method10_complete_recovery_runs_all_policy_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zarr

    native_summary, _row = _native_fixture(
        tmp_path, native_method="method10-mattes"
    )
    monkeypatch.setattr(zarr, "open_group", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        subject.recovery.ngff,
        "scale_translation",
        lambda _group: (
            ["z", "y", "x"],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            True,
            True,
        ),
    )

    output = tmp_path / "complete"
    final_summary = subject.complete_recovery(
        native_summary=native_summary, output_dir=output
    )

    payload = subject.read_json_object(final_summary)
    assert payload["cache_config"]["native_method"] == "method10-mattes"
    assert payload["complete_recovery"]["policy_order"] == list(
        subject.POLICY_ORDER
    )
    assert set((output / "reports").glob("*.json")) == {
        output / "reports" / f"{stage}.json"
        for stage in subject.POLICY_ORDER
    }


def test_spatial_promotion_accepts_manual_initializer_without_interpolation_support() -> None:
    row = _spatial_row(reason="low_gradient_ncc")
    row["outlier_recovery"].update(source_good_window_json=[], initializer_source="manual")

    promoted = subject._promote_spatial_fallback(
        row, native_method="method6", scorer=_FakeScorer(residual=[_shift(0.0, 0.5), _shift(0.5, 0.0)]), image_ncc_margin=0.02,
        maximum_unverified_rerun_departure_px=60.0,
    )

    assert promoted["status"] == "accepted"
    assert promoted["interpolated_fallback_promotion"]["initializer_source"] == "manual"
    assert promoted["interpolated_fallback_promotion"]["residual"]["verdict"] == "verified"


class _ResidualByX:
    """Residual shifts keyed by candidate x translation."""

    def __init__(self, residuals: dict[float, list[PlaneShift]]) -> None:
        self.residuals = residuals
        self.checked: list[float] = []

    def residual_shifts(self, _row, transform):
        self.checked.append(float(transform[11]))
        return self.residuals[float(transform[11])]


VERIFIED = [_shift(0.0, 0.0), _shift(0.0, 0.5)]
UNVERIFIED = [_shift(0.0, -3.0), _shift(0.0, -3.0)]
CONTRADICTED = [_shift(0.0, -6.0), _shift(0.0, -6.0)]


def _manual_row(tmp_path: Path, *, attempt: str, source: str = "manual") -> dict:
    initializer = tmp_path / "recovery_initializers" / "window-001.json"
    row = {
        **_transform_row(status="accepted", reason=None, attempt=attempt, x=7.0),
        "selected_corr_refined": 0.5,
        "selected_gradient_component_ncc_mean": 0.4,
        "outlier_recovery": {"initializer_source": "manual", "initializer_json": str(initializer)},
    }
    _write(
        initializer,
        {
            "moving_tile": row["moving_tile"],
            "moving_start_l0_zyx": row["moving_start_l0_zyx"],
            "local_matrix_zyx": np.eye(3).tolist(),
            "local_translation_zyx": [0.0, 0.0, 3.0],
            "recovery": {"initializer_source": source},
        },
    )
    return row


@pytest.mark.parametrize(
    "attempt",
    ["native_method6_from_recovery_initializer", "level2_initializer_phase_primed"],
)
def test_manual_seed_arbitration_publishes_verified_seed_over_unverified_fit(
    tmp_path: Path, attempt: str
) -> None:
    scorer = _ResidualByX({7.0: UNVERIFIED, 3.0: VERIFIED})

    row = subject._arbitrate_manual_seed(
        _manual_row(tmp_path, attempt=attempt), native_method="method6", scorer=scorer
    )

    assert row["status"] == "accepted"
    assert row["selected_local_translation_zyx"] == [0.0, 0.0, 3.0]
    assert row["selected_attempt"] == "interpolated_method6_spatial_recovery"
    assert row["selected_corr_refined"] is None
    arbitration = row["manual_seed_arbitration"]
    assert arbitration["selected_candidate"] == "manual_seed"
    assert arbitration["previous_selected_attempt"] == attempt
    assert arbitration["fitted_residual"]["verdict"] == "unverified"
    assert arbitration["seed_residual"]["verdict"] == "verified"


@pytest.mark.parametrize(
    ("fitted", "seed", "checked"),
    [
        (VERIFIED, VERIFIED, [7.0]),
        (CONTRADICTED, UNVERIFIED, [7.0, 3.0]),
        (UNVERIFIED, CONTRADICTED, [7.0, 3.0]),
    ],
)
def test_manual_seed_arbitration_keeps_fit_unless_only_the_seed_verifies(
    tmp_path: Path, fitted: list[PlaneShift], seed: list[PlaneShift], checked: list[float]
) -> None:
    scorer = _ResidualByX({7.0: fitted, 3.0: seed})
    original = _manual_row(tmp_path, attempt="native_method6_from_recovery_initializer")

    row = subject._arbitrate_manual_seed(original, native_method="method6", scorer=scorer)

    assert row["selected_local_translation_zyx"] == original["selected_local_translation_zyx"]
    assert row["selected_attempt"] == original["selected_attempt"]
    assert row["selected_corr_refined"] == 0.5
    assert row["manual_seed_arbitration"]["selected_candidate"] == "fitted"
    assert scorer.checked == checked


def test_manual_seed_arbitration_rejects_a_non_manual_initializer(tmp_path: Path) -> None:
    row = _manual_row(tmp_path, attempt="native_method6_from_recovery_initializer", source="interpolated")

    with pytest.raises(ValueError, match="is not the manual initializer"):
        subject._arbitrate_manual_seed(
            row, native_method="method6", scorer=_ResidualByX({7.0: UNVERIFIED, 3.0: VERIFIED})
        )


@pytest.mark.parametrize(("initializer_source", "arbitrated"), [("manual", True), (None, False)])
def test_manual_seed_stage_arbitrates_only_accepted_manual_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initializer_source: str | None, arbitrated: bool
) -> None:
    summary, row_path = _native_fixture(tmp_path)
    scorer = _ResidualByX({7.0: UNVERIFIED, 3.0: VERIFIED})
    monkeypatch.setattr(subject.recovery, "window_image_scorer", lambda _summary: scorer)
    row = _manual_row(tmp_path, attempt="native_method6_from_recovery_initializer")
    row["outlier_recovery"]["initializer_source"] = initializer_source
    _write(row_path, row)
    payload = subject.read_json_object(summary)
    payload["windows"][0].update(
        status="accepted", rejection_reason=None, selected_attempt=row["selected_attempt"]
    )
    _write(summary, payload)

    result = subject._arbitrate_manual_seeds(summary, tmp_path / "arbitrated")
    arbitrated_summary = subject.read_json_object(result)
    compact = arbitrated_summary["windows"][0]
    published = subject.read_json_object(Path(compact["level0_json"]))

    expected_x = 3.0 if arbitrated else 7.0
    assert published["selected_local_translation_zyx"] == [0.0, 0.0, expected_x]
    assert compact["selected_attempt"] == published["selected_attempt"]
    assert ("manual_seed_arbitration" in published) is arbitrated
    assert arbitrated_summary["manual_seed_arbitration"]["selected_candidates"] == (
        {"manual_seed": 1} if arbitrated else {}
    )


def test_manual_seed_arbitration_runs_on_the_linear_recovery_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def stage(name):
        def run(summary, output_dir, **_kwargs):
            calls.append((name, Path(summary).name))
            return Path(f"{name}.json")

        return run

    monkeypatch.setattr(subject.recovery, "filter_final_linear_outliers", lambda args: stage("filter")(args.summary, args.output_dir))
    monkeypatch.setattr(subject.recovery, "smooth_final_linear_mapping", lambda args: stage("smooth")(args.summary, args.output_dir))
    monkeypatch.setattr(subject, "_promote_spatial_fallbacks", stage("promote"))
    monkeypatch.setattr(subject, "_recover_linear_mapping_rejections", stage("linear"))
    monkeypatch.setattr(subject, "_arbitrate_manual_seeds", stage("manual"))

    final, reports = subject._run_policy_stages(
        Path("native.json"), tmp_path, subject.DEFAULT_COMPLETE_RECOVERY_SETTINGS
    )

    assert calls[-2:] == [("linear", "promote.json"), ("manual", "linear.json")]
    assert final == Path("manual.json")
    assert [stage for stage, _path in reports] == list(subject.POLICY_ORDER)


class _CandidateScorer:
    """Plane scores and residual shifts keyed by candidate x translation."""

    def __init__(
        self,
        planes: dict[float, list[float]],
        residuals: dict[float, list[PlaneShift]],
        aggregate: dict[float, PlaneShift] | None = None,
    ) -> None:
        self.planes = planes
        self.residuals = residuals
        self.aggregate = aggregate or {}
        self.checked: list[float] = []
        self.aggregated: list[float] = []

    def plane_scores(self, _row, transforms):
        return [self.planes[float(transform[11])] for transform in transforms]

    def residual_shifts(self, _row, transform):
        self.checked.append(float(transform[11]))
        return self.residuals[float(transform[11])]

    def aggregate_residual_shift(self, _row, transform):
        self.aggregated.append(float(transform[11]))
        return self.aggregate.get(float(transform[11]))


def _window_row(*, native_x: float, native_matrix: np.ndarray | None = None, manual: bool = False) -> dict:
    """Spatial-fallback row on a 528-voxel window with interpolated x=3 and a quality-gate-rejected rerun."""
    row = _spatial_row(reason="quality_gate_grad_ncc", native_x=native_x)
    row["fixed_stop_zyx"] = [528, 528, 528 + row["fixed_start_zyx"][2]]
    if native_matrix is not None:
        row["method8_attempts"][1]["local_matrix_zyx"] = native_matrix.tolist()
    if manual:
        row["outlier_recovery"].update(source_good_window_json=[], initializer_source="manual")
    return row


def _promote_window(row: dict, scorer: _CandidateScorer) -> dict:
    return subject._promote_spatial_fallback(
        row,
        native_method="method6",
        scorer=scorer,
        image_ncc_margin=0.02,
        maximum_unverified_rerun_departure_px=60.0,
    )


@pytest.mark.parametrize(
    ("native_x", "rerun_residual", "fallback_residual", "status", "selected", "checked"),
    [
        # An unverified rerun beyond the bound falls back to the interpolated placement.
        (200.0, [], VERIFIED, "accepted", "interpolated", [200.0, 3.0]),
        # The fallback still faces the residual check; its residual-shifted placement (x=-21) does not verify.
        (200.0, UNVERIFIED, CONTRADICTED, "rejected", "interpolated", [200.0, 3.0, -21.0]),
        # A verified rerun is kept at any distance.
        (200.0, VERIFIED, CONTRADICTED, "accepted", "native_rerun", [200.0]),
        # The bound is inclusive: a 60 px corner displacement keeps the rerun, whose residual-shifted
        # placement (x=51) does not verify either.
        (63.0, UNVERIFIED, VERIFIED, "accepted", "native_rerun", [63.0, 51.0]),
    ],
)
def test_spatial_promotion_withholds_unverified_rerun_far_from_interpolation(
    native_x: float,
    rerun_residual: list[PlaneShift],
    fallback_residual: list[PlaneShift],
    status: str,
    selected: str,
    checked: list[float],
) -> None:
    scorer = _CandidateScorer(
        {3.0: [0.01, 0.02], native_x: [0.05, 0.06]},
        {native_x: rerun_residual, 3.0: fallback_residual, -21.0: UNVERIFIED, 51.0: UNVERIFIED},
    )

    promoted = _promote_window(_window_row(native_x=native_x), scorer)

    promotion = promoted["interpolated_fallback_promotion"]
    withheld = selected == "interpolated"
    assert promoted["status"] == status
    assert promotion["selected_candidate"] == selected
    assert promotion["native_rerun_corner_departure_px"] == pytest.approx(native_x - 3.0)
    assert promotion["native_rerun_withheld_for_departure"] == withheld
    assert scorer.checked == checked
    if withheld:
        assert promotion["withheld_native_rerun_residual"]["verdict"] != "verified"
    if status == "accepted":
        expected_x = 3.0 if withheld else native_x
        assert promoted["selected_local_translation_zyx"] == [0.0, 0.0, expected_x]
    else:
        assert promoted["rejection_reason"] == subject.CONTRADICTED_INTERPOLATION_REASON


def test_spatial_promotion_bounds_rerun_rotation_without_translation_departure() -> None:
    angle = np.deg2rad(10.0)
    rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )
    scorer = _CandidateScorer({3.0: [0.01, 0.02], 3.5: [0.05, 0.06]}, {3.5: UNVERIFIED, 3.0: VERIFIED})

    promoted = _promote_window(_window_row(native_x=3.5, native_matrix=rotation), scorer)

    promotion = promoted["interpolated_fallback_promotion"]
    assert promotion["native_rerun_corner_departure_px"] > 60.0
    assert promotion["selected_candidate"] == "interpolated"
    assert promoted["selected_local_matrix_zyx"] == np.eye(3).tolist()


def test_spatial_promotion_falls_back_to_the_manual_seed() -> None:
    scorer = _CandidateScorer({3.0: [0.01, 0.02], 200.0: [0.05, 0.06]}, {200.0: UNVERIFIED, 3.0: VERIFIED})

    promoted = _promote_window(_window_row(native_x=200.0, manual=True), scorer)

    promotion = promoted["interpolated_fallback_promotion"]
    assert promotion["initializer_source"] == "manual"
    assert promotion["selected_candidate"] == "interpolated"
    assert promoted["selected_local_translation_zyx"] == [0.0, 0.0, 3.0]


def _recorded_fit_row(tmp_path: Path, *, original_matrix: np.ndarray | None = None) -> dict:
    """Spatial-fallback row with interpolated x=3, a rejected original fit at x=9 and a parent initializer at x=11."""
    row = _spatial_row(reason="local_3d_validation_residual")
    row["moving_shape_zyx"] = [10, 10, 10]
    source = tmp_path / "registration" / "window.json"
    _write(source, {
        **{key: row[key] for key in ("moving_tile", "fixed_start_zyx", "moving_start_l0_zyx", "moving_shape_zyx")},
        "initializer_conversion": {"source": "accepted_parent_window"},
        "initial_matrix_zyx": np.eye(3).tolist(),
        "initial_translation_zyx": [0.0, 0.0, 11.0],
    })
    row["outlier_recovery"].update(
        original_selected_local_matrix_zyx=(np.eye(3) if original_matrix is None else original_matrix).tolist(),
        original_selected_local_translation_zyx=[0.0, 0.0, 9.0],
        source_window_json=str(source),
    )
    return row


@pytest.mark.parametrize(
    ("residuals", "selected", "status"),
    [
        ({3.0: UNVERIFIED, 9.0: VERIFIED}, "original_fit", "accepted"),
        ({3.0: UNVERIFIED, 9.0: UNVERIFIED, 11.0: VERIFIED}, "parent_initializer", "accepted"),
        # A verified recorded fit also replaces a contradicted interpolation instead of leaving a hole.
        ({3.0: CONTRADICTED, 9.0: VERIFIED}, "original_fit", "accepted"),
        ({3.0: VERIFIED}, "interpolated", "accepted"),
        # The interpolated x=3 shifted by four times its -3 px level-2 residual sits at x=-9.
        ({3.0: UNVERIFIED, 9.0: UNVERIFIED, 11.0: CONTRADICTED, -9.0: UNVERIFIED}, "interpolated", "accepted"),
        ({3.0: CONTRADICTED, 9.0: UNVERIFIED, 11.0: UNVERIFIED, -21.0: UNVERIFIED}, "interpolated", "rejected"),
        # A contradicted interpolation shifted by its measured residual is promoted once that shift verifies.
        (
            {3.0: CONTRADICTED, 9.0: UNVERIFIED, 11.0: UNVERIFIED, -21.0: VERIFIED},
            "residual_corrected_interpolated",
            "accepted",
        ),
        # Unverified without a measured misalignment: no confident planes, or planes that disagree.
        ({3.0: []}, "interpolated", "accepted"),
        ({3.0: [_shift(0.0, 4.0), _shift(0.0, -4.0)]}, "interpolated", "accepted"),
    ],
)
def test_spatial_promotion_replaces_measured_misalignment_with_verified_recorded_fit(
    tmp_path: Path, residuals: dict[float, list[PlaneShift]], selected: str, status: str
) -> None:
    scorer = _CandidateScorer({}, residuals)

    promoted = _promote_window(_recorded_fit_row(tmp_path), scorer)

    promotion = promoted["interpolated_fallback_promotion"]
    assert scorer.checked == list(residuals)
    assert [check["candidate"] for check in promotion["recorded_fit_checks"]] == (
        ["original_fit", "parent_initializer", "residual_corrected_interpolated"][: len(residuals) - 1]
    )
    assert promoted["status"] == status
    assert promotion["selected_candidate"] == selected
    if status == "accepted":
        expected_x = {
            "original_fit": 9.0,
            "parent_initializer": 11.0,
            "residual_corrected_interpolated": -21.0,
            "interpolated": 3.0,
        }[selected]
        assert promoted["selected_local_translation_zyx"] == [0.0, 0.0, expected_x]
    if selected == "interpolated":
        assert promotion["replaced_selection_residual"] is None
    else:
        assert promoted["selected_attempt"] == f"residual_verified_{selected}"
        assert promotion["residual"]["verdict"] == "verified"
        assert promotion["replaced_selection_residual"]["verdict"] != "verified"


def test_spatial_promotion_skips_an_implausible_recorded_original_fit(tmp_path: Path) -> None:
    scorer = _CandidateScorer({}, {3.0: UNVERIFIED, 11.0: VERIFIED})

    promoted = _promote_window(_recorded_fit_row(tmp_path, original_matrix=np.diag([1.0, 1.0, 1.5])), scorer)

    assert scorer.checked == [3.0, 11.0]
    assert promoted["interpolated_fallback_promotion"]["selected_candidate"] == "parent_initializer"


def test_spatial_promotion_has_no_parent_candidate_when_the_original_fit_never_started(tmp_path: Path) -> None:
    row = _recorded_fit_row(tmp_path)
    source = Path(row["outlier_recovery"]["source_window_json"])
    original = json.loads(source.read_text())
    del original["initial_matrix_zyx"], original["initial_translation_zyx"]
    _write(source, original)
    scorer = _CandidateScorer({}, {3.0: UNVERIFIED, 9.0: UNVERIFIED, -9.0: UNVERIFIED})

    promoted = _promote_window(row, scorer)

    assert scorer.checked == [3.0, 9.0, -9.0]
    assert promoted["interpolated_fallback_promotion"]["selected_candidate"] == "interpolated"


def test_spatial_promotion_rejects_a_parent_initializer_from_another_window_frame(tmp_path: Path) -> None:
    row = _recorded_fit_row(tmp_path)
    source = Path(row["outlier_recovery"]["source_window_json"])
    _write(source, {**json.loads(source.read_text()), "moving_start_l0_zyx": [80, 0, 3]})

    with pytest.raises(ValueError, match="does not share the window frame"):
        _promote_window(row, _CandidateScorer({}, {3.0: UNVERIFIED, 9.0: UNVERIFIED}))


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), -1.0])
def test_complete_recovery_settings_reject_unusable_rerun_bound(bound: float) -> None:
    with pytest.raises(ValueError, match="maximum_unverified_rerun_departure_px"):
        subject.CompleteRecoverySettings(maximum_unverified_rerun_departure_px=bound)


@pytest.mark.parametrize("cache", [
    {}, {"native_method": "method6", "local_3d_refinement": False},
    {"native_method": "method10-mattes", "local_3d_refinement": True},
])
def test_unvalidated_local_3d_attempt_cannot_seed_recovery(cache: dict) -> None:
    row = {"status": "accepted", "selected_attempt": "local_3d_affine_refinement",
           "cache_config": cache}
    assert not subject._is_native_source(row, "method6")


def test_quality_rejected_local_refinement_uses_final_attempt_not_native_precursor() -> None:
    row = _spatial_row(reason='quality_gate_grad_ncc')
    row['cache_config'] = dict(native_method='method6', local_3d_refinement=True)
    row['method8_attempts'][1]['rejection_reason'] = None
    row['method8_attempts'][1]['status'] = 'accepted'
    row['method8_attempts'][1]['local_translation_zyx'] = [0, 0, 9]
    row['method8_attempts'].append(_attempt('local_3d_affine_refinement', reason='quality_gate_grad_ncc', grad=0.15, x=7))
    row['outlier_recovery']['native_rerun_gradient_component_ncc_mean'] = 0.15
    scorer = _FakeScorer({3.0: [0.3, 0.4], 7.0: [0.5, 0.6]})
    promoted = subject._promote_spatial_fallback(
        row, native_method='method6', scorer=scorer, image_ncc_margin=0.02,
        maximum_unverified_rerun_departure_px=60,
    )
    assert promoted['selected_attempt'] == 'local_3d_affine_refinement'
    assert promoted['selected_local_translation_zyx'] == [0, 0, 7]
    assert promoted['selected_gradient_component_ncc_mean'] == 0.15
    assert scorer.compared == [[3, 7]]


@pytest.mark.parametrize('reason', ['local_3d_validation_residual', 'insufficient_local_3d_support'])
def test_local_refinement_validation_failure_is_not_quality_only(reason: str) -> None:
    row = _spatial_row(reason=reason)
    row['cache_config'] = dict(native_method='method6', local_3d_refinement=True)
    row['method8_attempts'].append(_attempt('local_3d_affine_refinement', reason=reason, grad=0.15, x=7))
    assert subject._quality_gate_rejected_native_rerun(
        row, native_method='method6', interpolated=np.concatenate([np.eye(3).ravel(), [0, 0, 3]]),
    ) is None


def test_recorded_local_refinement_requires_one_final_attempt() -> None:
    row = _spatial_row(reason='quality_gate_grad_ncc')
    row['cache_config'] = dict(native_method='method6', local_3d_refinement=True)
    with pytest.raises(ValueError, match='Expected one final local'):
        subject._quality_gate_rejected_native_rerun(
            row, native_method='method6', interpolated=np.concatenate([np.eye(3).ravel(), [0, 0, 3]]),
        )


WEAK_PLANES = [_shift(0.0, -6.0, peak=0.15, second=0.1), _shift(0.0, 2.0, peak=0.15, second=0.1)]


@pytest.mark.parametrize(
    ("aggregate", "status", "selected", "aggregated"),
    [
        # Weak planes alone leave the interpolation unverified and accepted.
        ({}, "accepted", "interpolated", [3.0]),
        ({3.0: _shift(0.0, 0.0)}, "accepted", "interpolated", [3.0]),
        # A confident aggregate misalignment is corrected (x = 3 + 4 * -6) once the aggregate verifies the shift.
        ({3.0: _shift(0.0, -6.0), -21.0: _shift(0.0, 0.5)}, "accepted", "residual_corrected_interpolated", [3.0, 9.0, 11.0, -21.0]),
        # Without a verified candidate the measured misalignment is withheld.
        ({3.0: _shift(0.0, -6.0), -21.0: _shift(0.0, 5.0)}, "rejected", "interpolated", [3.0, 9.0, 11.0, -21.0]),
    ],
)
def test_spatial_promotion_measures_weak_planes_with_the_aggregate_residual(
    tmp_path: Path, aggregate: dict[float, PlaneShift], status: str, selected: str, aggregated: list[float]
) -> None:
    residuals = {x: WEAK_PLANES for x in (3.0, 9.0, 11.0, -21.0)}
    scorer = _CandidateScorer({}, residuals, aggregate)

    promoted = _promote_window(_recorded_fit_row(tmp_path), scorer)

    promotion = promoted["interpolated_fallback_promotion"]
    assert scorer.aggregated == aggregated
    assert promoted["status"] == status
    assert promotion["selected_candidate"] == selected
    assert promotion["residual"]["verdict"] == ("verified" if selected != "interpolated" else "unverified")
    if selected == "residual_corrected_interpolated":
        assert promoted["selected_local_translation_zyx"] == [0.0, 0.0, -21.0]
        assert promotion["residual"]["aggregate"]["verdict"] == "verified"
    if status == "rejected":
        assert promotion["aggregate_residual"]["verdict"] == "contradicted"


def test_spatial_promotion_skips_the_aggregate_when_a_plane_is_confident(tmp_path: Path) -> None:
    scorer = _CandidateScorer({}, {3.0: VERIFIED}, {3.0: _shift(0.0, -6.0)})

    promoted = _promote_window(_recorded_fit_row(tmp_path), scorer)

    assert scorer.aggregated == []
    assert promoted["interpolated_fallback_promotion"]["aggregate_residual"] is None
    assert promoted["status"] == "accepted"


SPREAD_PLANES = [_shift(0.0, -5.0), _shift(0.0, -8.0), _shift(0.0, -11.0)]


@pytest.mark.parametrize(
    ("corrected", "status", "selected"),
    [
        # Planes 3 px apart around a -8 px median: the shift to x = 3 + 4 * -8 is promoted once it verifies.
        (VERIFIED, "accepted", "residual_corrected_interpolated"),
        # Disagreeing planes never contradict, so an unverified correction leaves the interpolation accepted.
        (UNVERIFIED, "accepted", "interpolated"),
    ],
)
def test_spatial_promotion_corrects_a_depth_varying_misalignment(
    tmp_path: Path, corrected: list[PlaneShift], status: str, selected: str
) -> None:
    scorer = _CandidateScorer({}, {3.0: SPREAD_PLANES, 9.0: UNVERIFIED, 11.0: UNVERIFIED, -29.0: corrected})

    promoted = _promote_window(_recorded_fit_row(tmp_path), scorer)

    promotion = promoted["interpolated_fallback_promotion"]
    assert scorer.checked == [3.0, 9.0, 11.0, -29.0]
    assert promoted["status"] == status
    assert promotion["selected_candidate"] == selected
    expected_x = -29.0 if selected == "residual_corrected_interpolated" else 3.0
    assert promoted["selected_local_translation_zyx"] == [0.0, 0.0, expected_x]
