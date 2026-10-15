from pathlib import Path

import pytest

from squisher_segment.segment.model_artifacts import (
    invalidate_trt_plans, plan_path_for_device, validate_trt_plan, write_trt_plan_metadata,
)


def _build_artifacts(tmp_path: Path):
    checkpoint = tmp_path / "model"
    checkpoint.write_bytes(b"checkpoint")
    plan = plan_path_for_device(checkpoint, "Test GPU")
    plan.write_bytes(b"plan")
    metadata = write_trt_plan_metadata(checkpoint, "Test GPU", backend="sam", bsize=256,
                                       batch_size=1, opset=22, vram_mb=1024)
    return checkpoint, plan, metadata


def test_plan_is_bound_to_checkpoint_and_build(tmp_path):
    checkpoint, _, metadata = _build_artifacts(tmp_path)
    assert validate_trt_plan(checkpoint, "Test GPU") == metadata
    for override in ({"backend": "unet"}, {"bsize": 192}, {"batch_size": 2}):
        with pytest.raises(ValueError, match="incompatible"):
            validate_trt_plan(checkpoint, "Test GPU", **override)


@pytest.mark.parametrize("changed", ["checkpoint", "plan"])
def test_plan_rejects_artifact_replacement(tmp_path, changed):
    checkpoint, plan, _ = _build_artifacts(tmp_path)
    (checkpoint if changed == "checkpoint" else plan).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="stale"):
        validate_trt_plan(checkpoint, "Test GPU")


def test_retraining_invalidates_without_deleting_artifacts(tmp_path):
    checkpoint, plan, _ = _build_artifacts(tmp_path)
    invalidate_trt_plans(checkpoint)
    assert checkpoint.read_bytes() == b"checkpoint"
    assert plan.read_bytes() == b"plan"
    with pytest.raises(ValueError, match="stale"):
        validate_trt_plan(checkpoint, "Test GPU")


def test_plan_rejects_changed_runtime(tmp_path, monkeypatch):
    from squisher_segment.segment import model_artifacts
    checkpoint, _, _ = _build_artifacts(tmp_path)
    monkeypatch.setattr(model_artifacts, "_build_runtime_versions", lambda: {"torch": "changed"})
    with pytest.raises(ValueError, match="incompatible"):
        validate_trt_plan(checkpoint, "Test GPU")
