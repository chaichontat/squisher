from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).parents[1] / "scripts" / "prepare_fused_fixed_method10_linear_initializers.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linear_model_predicts_all_transform_parameters_from_zyx() -> None:
    module = _load()
    points = np.asarray([[z, y, x] for z in (0, 1, 2) for y in (0, 1) for x in (0, 1)], dtype=np.float64)
    coefficients = np.arange(4 * 12, dtype=np.float64).reshape(4, 12) / 100.0
    transforms = np.column_stack((np.ones(len(points)), points)) @ coefficients
    target = np.asarray([1.5, 0.25, 0.75])

    prediction, metadata = module.fit_linear_transform(points, transforms, target)

    assert metadata["design_rank"] == 4
    assert np.allclose(prediction, np.r_[1.0, target] @ coefficients)


def test_linear_model_uses_zero_slope_for_unobserved_spatial_axis() -> None:
    module = _load()
    points = np.asarray([[z, y, 0] for z in (0, 1, 2) for y in (0, 1)], dtype=np.float64)
    transforms = np.column_stack((np.ones(len(points)), points[:, :2])) @ np.arange(3 * 12).reshape(3, 12)

    prediction, metadata = module.fit_linear_transform(points, transforms, np.asarray([1.5, 0.25, 5.0]))

    assert metadata["modeled_spatial_axes"] == ["z", "y"]
    assert metadata["constant_spatial_axes"] == ["x"]
    assert np.allclose(prediction, np.r_[1.0, 1.5, 0.25] @ np.arange(3 * 12).reshape(3, 12))


def test_linear_model_drops_collinear_spatial_axis() -> None:
    module = _load()
    points = np.asarray([[value, value, 0] for value in (0, 1, 2, 3)], dtype=np.float64)
    transforms = np.column_stack((np.ones(len(points)), points[:, 0])) @ np.arange(2 * 12).reshape(2, 12)

    prediction, metadata = module.fit_linear_transform(points, transforms, np.asarray([1.5, 1.5, 0.0]))

    assert metadata["modeled_spatial_axes"] == ["z"]
    assert metadata["dependent_spatial_axes"] == ["y"]
    assert np.allclose(prediction, np.r_[1.0, 1.5] @ np.arange(2 * 12).reshape(2, 12))
