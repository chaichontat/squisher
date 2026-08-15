from pathlib import Path

import cellpose.contrib.packed_infer as packed_infer
import torch

from squisher_segment.segment.model_artifacts import plan_path_for_device
from squisher_segment.segmentation.distributed import model_cache


def test_model_cache_uses_device_specific_trt_plan(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "embryonicsheet"
    plan_path = plan_path_for_device(model_path, "NVIDIA RTX 6000 Ada")
    plan_path.touch()

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
