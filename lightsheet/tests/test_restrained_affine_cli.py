import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from squisher_lightsheet.cli import app
from squisher_lightsheet.fused_fixed import validate_method6_summary
from squisher_lightsheet.local_affine import local_affine_cache_config


def _arguments(tmp_path: Path) -> list[str]:
    for name in ("fixed.json", "moving.json", "source.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "fixed.zarr").mkdir()
    return [
        "cross-register", "method6",
        "--fixed-position", str(tmp_path / "fixed.json"),
        "--moving-position", str(tmp_path / "moving.json"),
        "--moving-source-position", str(tmp_path / "source.json"),
        "--fixed-fused", str(tmp_path / "fixed.zarr"),
        "--output-dir", str(tmp_path / "out"),
        "--output-registration", str(tmp_path / "out.json"),
        "--fixed-mask-threshold", "1000",
        "--native-lib-dir", str(tmp_path),
        "--source-label", "405", "--target-label", "561",
        "--local-affine-estimator", "restrained",
    ]


def test_public_method6_forwards_restrained_estimator(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> tuple[Path, Path]:
        calls.append(kwargs)
        return tmp_path / "summary.json", tmp_path / "registration.json"

    monkeypatch.setattr("squisher_lightsheet.cli.run_fused_fixed_method6", run)
    result = CliRunner().invoke(app, [*_arguments(tmp_path), "--local-3d-refinement"])
    assert result.exit_code == 0, result.output
    assert calls[0]["local_affine_estimator"] == "restrained"
    assert calls[0]["local_3d_refinement"] is True


def test_restrained_estimator_requires_local_refinement(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> tuple[Path, Path]:
        calls.append(kwargs)
        return tmp_path / "summary.json", tmp_path / "registration.json"

    monkeypatch.setattr("squisher_lightsheet.cli.run_fused_fixed_method6", run)
    result = CliRunner().invoke(app, _arguments(tmp_path))
    assert result.exit_code == 2
    assert "requires" in result.output
    assert "--local-3d-refinement" in result.output
    assert not calls


@pytest.mark.parametrize("requested,cached", [("restrained", "least-squares"), ("least-squares", "restrained")])
def test_summary_rejects_different_local_estimators(tmp_path: Path, requested: str, cached: str) -> None:
    cache = {
        "native_method": "method6", "fit_intensity_transform": "log1p",
        "level0_initializer": "level2-method8", "moving_channel": 0,
        "local_3d_refinement": True,
        **local_affine_cache_config(cached),
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"cache_config": cache}))
    with pytest.raises(ValueError, match="local_affine_estimator"):
        validate_method6_summary(
            path, moving_position=tmp_path / "moving.json",
            moving_source_position=tmp_path / "source.json",
            fixed_fused=tmp_path / "fixed.zarr", moving_channel=0,
            fixed_mask_threshold=1000, local_3d_refinement=True,
            local_affine_estimator=requested,
        )


def test_summary_preserves_legacy_unregularized_contract(tmp_path: Path) -> None:
    cache = {
        "native_method": "method6", "fit_intensity_transform": "log1p",
        "level0_initializer": "level2-method8", "moving_channel": 0,
        "local_3d_refinement": True,
        "starting_affine_matrix_zyx": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "fixed_mask_threshold": 1000, "min_grad_ncc": 0.24,
        "moving_position": str(tmp_path / "moving.json"),
        "moving_source_position": str(tmp_path / "source.json"),
        "fixed_fused": str(tmp_path / "fixed.zarr"),
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"cache_config": cache}))
    result = validate_method6_summary(
        path, moving_position=tmp_path / "moving.json",
        moving_source_position=tmp_path / "source.json",
        fixed_fused=tmp_path / "fixed.zarr", moving_channel=0,
        fixed_mask_threshold=1000, local_3d_refinement=True,
        local_affine_estimator="least-squares",
    )
    assert result["cache_config"] == cache
