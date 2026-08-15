from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from squisher_segment.cli import app
from squisher_segment.segment.model_artifacts import plan_path_for_device
from squisher_segment.segment.train import TrainConfig
import squisher_segment.segment.train as train_module


_BASE_CONFIG = {
    "base_model": None,
    "channels": (1, 2),
    "training_paths": ["sample"],
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "embryonicsheet"),
        ("bsize", 224),
        ("SGD", False),
        ("optimizer", "adamw"),
        ("use_te", False),
        ("te_fp8", False),
    ],
)
def test_train_config_rejects_removed_fields(field: str, value: object) -> None:
    config = {**_BASE_CONFIG, field: value}

    with pytest.raises(ValidationError) as exc_info:
        TrainConfig.model_validate(config)

    assert any(
        error["loc"] == (field,) and error["type"] == "extra_forbidden"
        for error in exc_info.value.errors()
    )


def test_run_train_rejects_invalid_test_folder(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "sample").mkdir()
    monkeypatch.setattr(
        train_module,
        "concat_output",
        lambda *_args, **_kwargs: (
            [np.zeros((4, 4), dtype=np.float32)],
            [np.zeros((4, 4), dtype=np.uint16)],
            [Path("sample/image.tif")],
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_train",
        lambda *_args, **_kwargs: pytest.fail("training should not start"),
    )

    with pytest.raises(ValueError, match="cannot contain '\\.\\.'"):
        train_module.run_train(
            "embryonicsheet",
            tmp_path,
            TrainConfig(**_BASE_CONFIG, test_folder="../outside"),
        )


def test_plan_path_for_device_sanitizes_device_name() -> None:
    model_path = Path("/models/embryonicsheet")

    assert plan_path_for_device(model_path, "NVIDIA RTX 6000 Ada") == Path(
        "/models/embryonicsheet-NVIDIA_RTX_6000_Ada.plan"
    )


def test_train_reads_config_and_writes_trained_config(tmp_path: Path, monkeypatch) -> None:
    models_path = tmp_path / "models"
    models_path.mkdir()
    config = TrainConfig(
        base_model=None,
        channels=(1, 2),
        training_paths=["sample"],
    )
    (models_path / "embryonicsheet.json").write_text(
        "// training config\n" + config.model_dump_json()
    )

    def fake_run_train(name: str, path: Path, train_config: TrainConfig) -> TrainConfig:
        assert name == "embryonicsheet"
        assert path == tmp_path
        assert train_config.skip_trt is True
        return train_config.model_copy(update={"train_losses": [0.3, 0.2], "model_md5": "abc123"})

    monkeypatch.setattr(train_module, "run_train", fake_run_train)

    result = CliRunner().invoke(
        app,
        ["train", str(tmp_path), "embryonicsheet", "--skip-trt"],
    )

    assert result.exit_code == 0, result.output
    trained = TrainConfig.model_validate_json(
        (models_path / "embryonicsheet.trained.json").read_text()
    )
    assert trained.train_losses == [0.3, 0.2]
    assert trained.model_md5 == "abc123"
    assert trained.skip_trt is True


def test_run_train_prepares_images_and_records_model(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(
        train_module,
        "_discover_training_dirs",
        lambda path, training_paths: [Path("sample")],
    )
    monkeypatch.setattr(
        train_module,
        "concat_output",
        lambda *_args, **_kwargs: (
            [
                np.zeros((1, 4, 4), dtype=np.float32),
                np.zeros((3, 4, 4), dtype=np.float32),
            ],
            [np.zeros((4, 4), dtype=np.uint16)] * 2,
            [Path("sample/single.tif"), Path("sample/three.tif")],
            None,
            None,
            None,
        ),
    )

    def fake_train(out, path: Path, name: str, train_config: TrainConfig):
        assert [image.shape for image in out[0]] == [(4, 4), (2, 4, 4)]
        model_path = path / "models" / name
        model_path.write_bytes(b"trained model")
        return model_path, [0.3, 0.2], None

    monkeypatch.setattr(train_module, "_train", fake_train)

    updated = train_module.run_train(
        "embryonicsheet",
        tmp_path,
        TrainConfig(
            base_model=None,
            channels=(1, 2),
            training_paths=["sample"],
        ),
    )

    assert updated.train_losses == [0.3, 0.2]
    assert updated.model_md5 is not None


def test_train_skips_trt_build_when_configured(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "models" / "embryonicsheet"
    model_path.parent.mkdir()
    model_path.write_bytes(b"trained model")

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_module.torch, "device", lambda name: name)
    monkeypatch.setattr(
        train_module,
        "CellposeModel",
        lambda **_: SimpleNamespace(net=SimpleNamespace(diam_mean=np.array(30.0))),
    )
    monkeypatch.setattr(
        train_module,
        "train_seg_transformer",
        lambda *_args, **_kwargs: (model_path, [0.3], None),
    )
    monkeypatch.setattr(
        train_module,
        "_cleanup_model_artifacts",
        lambda *_args, **_kwargs: pytest.fail("cleanup should not run"),
    )
    monkeypatch.setattr(
        train_module,
        "build_trt_engine",
        lambda *_args, **_kwargs: pytest.fail("TRT build should not run"),
    )

    returned_path, train_losses, test_losses = train_module._train(
        (
            [np.zeros((4, 4), dtype=np.float32)],
            [np.zeros((4, 4), dtype=np.uint16)],
            [Path("sample/image.tif")],
            None,
            None,
            None,
        ),
        tmp_path,
        "embryonicsheet",
        TrainConfig(
            base_model="cpsam",
            channels=(1, 2),
            training_paths=["sample"],
            skip_trt=True,
        ),
    )

    assert returned_path == model_path
    assert train_losses == [0.3]
    assert test_losses is None
