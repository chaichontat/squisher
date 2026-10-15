from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from squisher_segment.cli import app
from squisher_segment.segment.model_artifacts import plan_path_for_device
from squisher_segment.segment.train import TrainConfig
import squisher_segment.segment.train as train_module


_BASE_CONFIG = {
    "name": "embryonicsheet",
    "base_model": None,
    "channels": (1, 2),
    "training_paths": ["sample"],
}


def test_train_config_accepts_fishtools2_fields() -> None:
    config = TrainConfig.model_validate(
        {
            **_BASE_CONFIG,
            "bsize": 256,
            "SGD": False,
            "optimizer": None,
        }
    )

    assert config.name == "embryonicsheet"
    assert config.bsize == 256
    assert config.SGD is False
    assert config.optimizer is None


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
        name="embryonicsheet",
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
        assert [image.shape for image in out[0]] == [(4, 4), (3, 4, 4)]
        model_path = path / "models" / name
        model_path.write_bytes(b"trained model")
        return model_path, [0.3, 0.2], None

    monkeypatch.setattr(train_module, "_train", fake_train)

    updated = train_module.run_train(
        "embryonicsheet",
        tmp_path,
        TrainConfig(
            name="embryonicsheet",
            base_model=None,
            channels=(1, 2),
            training_paths=["sample"],
        ),
    )

    assert updated.train_losses == [0.3, 0.2]
    assert updated.model_md5 is not None


@pytest.mark.parametrize(("optimizer", "expected_sgd"), [(None, True), ("adamw", False)])
def test_train_passes_optimizer_controls(
    tmp_path: Path,
    monkeypatch,
    optimizer: str | None,
    expected_sgd: bool,
) -> None:
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
    seen_train_kwargs: dict[str, object] = {}
    seen_build_kwargs: dict[str, object] = {}

    def fake_train(*_args, **kwargs):
        seen_train_kwargs.update(kwargs)
        return model_path, [0.3], None

    monkeypatch.setattr(
        train_module,
        "train_seg_transformer",
        fake_train,
    )
    monkeypatch.setattr(
        train_module,
        "invalidate_trt_plans",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        train_module,
        "build_trt_engine",
        lambda **kwargs: seen_build_kwargs.update(kwargs),
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
            name="embryonicsheet",
            base_model="cpsam",
            channels=(1, 2),
            training_paths=["sample"],
            bsize=192,
            SGD=True,
            optimizer=optimizer,
        ),
    )

    assert returned_path == model_path
    assert train_losses == [0.3]
    assert test_losses is None
    assert seen_train_kwargs["bsize"] == 192
    assert seen_train_kwargs["SGD"] is expected_sgd
    assert seen_build_kwargs["bsize"] == 256


def test_sam_preparation_preserves_three_channel_order() -> None:
    image = np.broadcast_to(np.array([405, 561, 638])[:, None, None], (3, 4, 5)).copy()
    prepared, _ = train_module._prepare_2d_training_images([image])
    np.testing.assert_array_equal(prepared[0], image)


def test_sam_preparation_rejects_more_than_three_channels() -> None:
    with pytest.raises(ValueError, match="SAM.*1 to 3"):
        train_module._prepare_2d_training_images([np.zeros((4, 5, 6))])


@pytest.mark.parametrize(
    "configured,override,expected",
    [(True, [], True), (False, ["--packed"], True), (True, ["--no-packed"], False)],
)
def test_train_packed_override_preserves_config(tmp_path, monkeypatch, configured, override, expected):
    models = tmp_path / "models"
    models.mkdir()
    config = TrainConfig(**_BASE_CONFIG, packed=configured)
    (models / "embryonicsheet.json").write_text(config.model_dump_json())
    seen = []
    monkeypatch.setattr(train_module, "run_train", lambda name, path, config: seen.append(config.packed) or config)
    result = CliRunner().invoke(app, ["train", str(tmp_path), "embryonicsheet", *override])
    assert result.exit_code == 0, result.output
    assert seen == [expected]


def test_unet_preparation_preserves_channels_for_explicit_selection():
    image = np.zeros((4, 5, 6))
    prepared, _ = train_module._prepare_2d_training_images([image], backend="unet")
    assert prepared[0] is image


def test_skip_trt_training_invalidates_previous_plan(tmp_path, monkeypatch):
    from squisher_segment.segment.model_artifacts import validate_trt_plan, write_trt_plan_metadata
    checkpoint = tmp_path / "model"
    checkpoint.write_bytes(b"old checkpoint")
    plan_path = plan_path_for_device(checkpoint, "Test GPU")
    plan_path.write_bytes(b"old plan")
    write_trt_plan_metadata(checkpoint, "Test GPU", backend="sam", bsize=256,
                            batch_size=1, opset=22, vram_mb=1024)
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_module.torch, "device", lambda name: name)
    monkeypatch.setattr(train_module, "CellposeModel", lambda **_: SimpleNamespace(net=SimpleNamespace()))
    def fake_train(*args, **kwargs):
        checkpoint.write_bytes(b"replacement checkpoint")
        return checkpoint, [0.3], None
    monkeypatch.setattr(train_module, "train_seg_transformer", fake_train)
    monkeypatch.setattr(train_module, "build_trt_engine", lambda **_: pytest.fail("skip_trt must skip build"))
    train_module._train(([np.zeros((3, 4, 5))], [np.zeros((4, 5))], [], None, None, None),
                        tmp_path, "model", TrainConfig(**_BASE_CONFIG, skip_trt=True))
    assert plan_path.read_bytes() == b"old plan"
    with pytest.raises(ValueError, match="stale"):
        validate_trt_plan(checkpoint, "Test GPU")


@pytest.mark.parametrize("fail_build", [False, True])
def test_engine_build_exports_current_checkpoint_and_invalidates_on_failure(tmp_path, monkeypatch, fail_build):
    from squisher_segment.segment.model_artifacts import validate_trt_plan, write_trt_plan_metadata
    checkpoint = tmp_path / "model"
    checkpoint.write_bytes(b"checkpoint")
    plan = plan_path_for_device(checkpoint, "Test GPU")
    plan.write_bytes(b"old plan")
    write_trt_plan_metadata(checkpoint, "Test GPU", backend="sam", bsize=256,
                            batch_size=1, opset=22, vram_mb=1024)
    checkpoint.with_name("model.onnx").write_bytes(b"old ONNX")
    monkeypatch.setattr(train_module.torch.cuda, "get_device_name", lambda _: "Test GPU")
    monkeypatch.setattr(train_module.torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(total_memory=1024 * 1024 * 1024))
    seen = []
    def export(model_path, onnx_path, **kwargs):
        seen.append(kwargs)
        Path(onnx_path).write_bytes(b"current ONNX")
    def build(onnx_path, plan_path, **kwargs):
        assert Path(onnx_path).read_bytes() == b"current ONNX"
        if fail_build:
            raise RuntimeError("build failed")
        Path(plan_path).write_bytes(b"current plan")
    monkeypatch.setattr(train_module.trt_build, "export_onnx", export)
    monkeypatch.setattr(train_module.trt_build, "build_engine", build)
    if fail_build:
        with pytest.raises(RuntimeError, match="build failed"):
            train_module.build_trt_engine(model_path=checkpoint, device="cuda", bsize=256)
        with pytest.raises(ValueError, match="stale"):
            validate_trt_plan(checkpoint, "Test GPU")
    else:
        train_module.build_trt_engine(model_path=checkpoint, device="cuda", bsize=256)
        assert validate_trt_plan(checkpoint, "Test GPU")["status"] == "ready"
    assert seen == [{"batch_size": 1, "bsize": 256, "opset": 22, "backend": "sam"}]


@pytest.mark.parametrize("backend", ["sam", "unet"])
def test_build_trt_cli_uses_distributed_profile(tmp_path, monkeypatch, backend):
    checkpoint = tmp_path / "model"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    seen = []
    def fake_build(**kwargs):
        seen.append(kwargs)
        return plan_path_for_device(kwargs["model_path"], "Test GPU")
    monkeypatch.setattr(train_module, "build_trt_engine", fake_build)
    result = CliRunner().invoke(app, ["build-trt", str(checkpoint), "--backend", backend])
    assert result.exit_code == 0, result.output
    assert seen == [{"model_path": checkpoint, "device": train_module.torch.device("cuda"),
                     "backend": backend, "bsize": 256, "batch_size": 1}]
    assert "model-Test_GPU.plan" in result.output


@pytest.mark.parametrize("failure", ["missing", "no_cuda", "invalid_backend"])
def test_build_trt_cli_rejects_invalid_input(tmp_path, monkeypatch, failure):
    checkpoint = tmp_path / "model"
    if failure != "missing":
        checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: failure != "no_cuda")
    monkeypatch.setattr(train_module, "build_trt_engine", lambda **_: pytest.fail("invalid build must not run"))
    args = ["build-trt", str(checkpoint)]
    if failure == "invalid_backend":
        args += ["--backend", "other"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    if failure == "no_cuda":
        assert "CUDA GPU is required" in str(result.exception)
