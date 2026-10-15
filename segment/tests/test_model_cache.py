from pathlib import Path
from types import SimpleNamespace

import cellpose.contrib.packed_infer as packed_infer
import pytest
import torch

from squisher_segment.segment.model_artifacts import (
    file_sha256,
    plan_path_for_device,
    runtime_implementation_identity,
    write_trt_plan_metadata,
)
from squisher_segment.segmentation.distributed import model_cache


def _verified_plan(model_path: Path, device_name: str, *, batch_size: int = 1) -> Path:
    model_path.write_bytes(b"checkpoint")
    plan_path = plan_path_for_device(model_path, device_name)
    plan_path.write_bytes(b"plan")
    write_trt_plan_metadata(model_path, device_name, backend="sam", bsize=256,
                            batch_size=batch_size, opset=22, vram_mb=1024)
    return plan_path


def test_runtime_identity_uses_revision_and_installed_cellpose_version(monkeypatch) -> None:
    from squisher_segment.segment import model_artifacts
    monkeypatch.setattr(model_artifacts, "version", lambda package: "test-cellpose" if package == "cellpose" else pytest.fail(package))
    monkeypatch.setattr(model_artifacts, "file_sha256", lambda *_: pytest.fail("must not hash sources"))
    assert runtime_implementation_identity() == {
        "segmentation_algorithm_revision": model_artifacts.SEGMENTATION_ALGORITHM_REVISION,
        "cellpose_version": "test-cellpose",
    }


@pytest.mark.parametrize("field", ["segmentation_algorithm_revision", "cellpose_version"])
def test_worker_rejects_runtime_implementation_mismatch(field) -> None:
    expected = runtime_implementation_identity()
    model_cache._validate_worker_implementation(expected)
    expected[field] = "changed"
    with pytest.raises(RuntimeError, match=field):
        model_cache._validate_worker_implementation(expected)


def test_model_cache_uses_device_specific_trt_plan(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "embryonicsheet"
    plan_path = plan_path_for_device(model_path, "NVIDIA RTX 6000 Ada")
    _verified_plan(model_path, "NVIDIA RTX 6000 Ada", batch_size=2)

    class FakeModel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA RTX 6000 Ada")
    monkeypatch.setattr(packed_infer, "PackedCellposeModelTRT", FakeModel)

    model = model_cache._build_packed_cellpose_model(
        {"backend": "sam", "pretrained_model": str(model_path), "batch_size": 2}
    )

    assert model.kwargs == {
        "pretrained_model": str(plan_path),
        "batch_size": 2,
        "gpu": True,
    }


def test_model_cache_uses_distinct_xy_and_ortho_plans(
    tmp_path: Path, monkeypatch
) -> None:
    xy_model = tmp_path / "xy-model"
    ortho_model = tmp_path / "ortho-model"
    xy_plan = plan_path_for_device(xy_model, "Test GPU")
    ortho_plan = plan_path_for_device(ortho_model, "Test GPU")
    _verified_plan(xy_model, "Test GPU")
    _verified_plan(ortho_model, "Test GPU")

    class FakeModel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
    monkeypatch.setattr(packed_infer, "PackedCellposeModelTRT", FakeModel)

    model = model_cache._build_packed_cellpose_model(
        {
            "pretrained_model": str(xy_model),
            "pretrained_model_ortho": str(ortho_model),
        }
    )

    assert model.kwargs == {
        "pretrained_model": str(xy_plan),
        "pretrained_model_ortho": str(ortho_plan),
        "gpu": True,
    }


def test_model_cache_requires_configured_ortho_plan(
    tmp_path: Path, monkeypatch
) -> None:
    xy_model = tmp_path / "xy-model"
    ortho_model = tmp_path / "ortho-model"
    _verified_plan(xy_model, "Test GPU")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")

    with pytest.raises(FileNotFoundError, match="required for ortho"):
        model_cache._build_packed_cellpose_model(
            {
                "pretrained_model": str(xy_model),
                "pretrained_model_ortho": str(ortho_model),
            }
        )


def test_model_cache_rejects_unet_backend() -> None:
    with pytest.raises(ValueError, match="only the SAM backend"):
        model_cache._build_packed_cellpose_model(
            {"backend": "unet", "pretrained_model": "/models/model"}
        )


def test_model_plugin_replaces_cache_when_artifact_identity_changes(monkeypatch) -> None:
    built: list[object] = []

    def fake_build(model_kwargs: dict[str, object]) -> object:
        model = object()
        built.append(model)
        return model

    monkeypatch.setattr(model_cache, "_build_packed_cellpose_model", fake_build)
    worker = SimpleNamespace(name="gpu-0")
    kwargs = {"backend": "sam", "pretrained_model": "/models/model"}

    monkeypatch.setattr(model_cache, "_validate_worker_plan", lambda *args: None)
    monkeypatch.setattr(model_cache, "_validate_worker_implementation", lambda *args: None)
    plans = [
        {
            "model_role": "xy",
            "device_name": "Test GPU",
            "path": "/plans/model.plan",
            "sha256": "abc",
        }
    ]
    sources = {"pipeline": "abc"}
    model_cache.CellposeModelPlugin(
        kwargs,
        artifact_key="plan-a",
        trt_plans=plans,
        implementation_identity=sources,
    ).setup(worker)
    first = worker.cellpose_model
    model_cache.CellposeModelPlugin(
        kwargs,
        artifact_key="plan-b",
        trt_plans=plans,
        implementation_identity=sources,
    ).setup(worker)

    assert len(built) == 2
    assert worker.cellpose_model is not first
    assert worker.cellpose_model_artifact_key == "plan-b"


def test_worker_rejects_plan_content_changed_after_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model"
    plan_path = plan_path_for_device(model_path, "Test GPU")
    _verified_plan(model_path, "Test GPU")
    plan_path.write_bytes(b"plan-a")
    write_trt_plan_metadata(model_path, "Test GPU", backend="sam", bsize=256,
                            batch_size=1, opset=22, vram_mb=1024)
    plan_stat = plan_path.stat()
    plans = [
        {
            "model_role": "xy",
            "device_name": "Test GPU",
            "path": str(plan_path.resolve()),
            "sha256": file_sha256(plan_path),
            "size": plan_stat.st_size,
            "mtime_ns": plan_stat.st_mtime_ns,
        }
    ]
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")

    model_cache._validate_worker_plan({"pretrained_model": str(model_path)}, plans)
    plan_path.write_bytes(b"plan-b")

    with pytest.raises(RuntimeError, match="unrecorded xy TensorRT plan"):
        model_cache._validate_worker_plan({"pretrained_model": str(model_path)}, plans)


def test_worker_rejects_changed_ortho_plan(tmp_path: Path, monkeypatch) -> None:
    xy_model = tmp_path / "xy-model"
    ortho_model = tmp_path / "ortho-model"
    plans = []
    for model_role, model_path in {"xy": xy_model, "ortho": ortho_model}.items():
        plan_path = plan_path_for_device(model_path, "Test GPU")
        _verified_plan(model_path, "Test GPU")
        plan_stat = plan_path.stat()
        plans.append(
            {
                "model_role": model_role,
                "device_name": "Test GPU",
                "path": str(plan_path.resolve()),
                "sha256": file_sha256(plan_path),
                "size": plan_stat.st_size,
                "mtime_ns": plan_stat.st_mtime_ns,
            }
        )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
    model_kwargs = {
        "pretrained_model": str(xy_model),
        "pretrained_model_ortho": str(ortho_model),
    }

    model_cache._validate_worker_plan(model_kwargs, plans)
    plan_path_for_device(ortho_model, "Test GPU").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="unrecorded ortho TensorRT plan"):
        model_cache._validate_worker_plan(model_kwargs, plans)


def test_model_cache_rejects_unverified_plan(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.write_bytes(b"new checkpoint")
    plan_path_for_device(model_path, "Test GPU").write_bytes(b"old plan")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test GPU")
    monkeypatch.setattr(packed_infer, "PackedCellposeModelTRT", lambda **_: object())
    with pytest.raises(ValueError, match="unverified"):
        model_cache._build_packed_cellpose_model({"pretrained_model": str(model_path)})
