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
            row, native_method="method6", scorer=_FakeScorer(), image_ncc_margin=0.02
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
        summary, tmp_path / "promoted", image_ncc_margin=0.02
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
        _spatial_row(reason=reason), native_method="method6", scorer=scorer, image_ncc_margin=0.02
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
            row, native_method="method6", scorer=_FakeScorer(), image_ncc_margin=0.02
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
def test_linear_recovery_uses_only_native_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_method: str,
    scores: tuple[list[float], list[float]],
    replaced: bool,
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
            attempt=f"native_{native_method.replace('-', '_')}_from_phase_primed_initializer",
            x=0.0,
        ),
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
        row, native_method="method6", scorer=_FakeScorer(residual=[_shift(0.0, 0.5), _shift(0.5, 0.0)]), image_ncc_margin=0.02
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
