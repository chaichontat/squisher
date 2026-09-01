from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from squisher_lightsheet.channel_affine import estimate_translation_gpu


def _fake_gpu_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    observed: dict[str, object] = {}
    cupy = SimpleNamespace(
        float32=np.float32,
        bool_=np.bool_,
        asarray=np.asarray,
        asnumpy=np.asarray,
        mean=np.mean,
    )
    registration = ModuleType("cucim.skimage.registration")

    def fake_phase(reference: np.ndarray, moving: np.ndarray, **kwargs: object):
        observed.update(kwargs)
        return np.asarray([1.0, -2.0, 3.0]), 0.0, 0.0

    registration.phase_cross_correlation = fake_phase  # type: ignore[attr-defined]
    skimage = ModuleType("cucim.skimage")
    skimage.registration = registration  # type: ignore[attr-defined]
    cucim = ModuleType("cucim")
    cucim.skimage = skimage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cucim", cucim)
    monkeypatch.setitem(sys.modules, "cucim.skimage", skimage)
    monkeypatch.setitem(sys.modules, "cucim.skimage.registration", registration)
    return observed


def test_masked_phase_forwards_both_support_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _fake_gpu_modules(monkeypatch)
    fixed = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    moving = fixed.copy()
    fixed_mask = fixed > 3
    moving_mask = moving < 24

    shift = estimate_translation_gpu(
        fixed,
        moving,
        reference_mask=fixed_mask,
        moving_mask=moving_mask,
        overlap_ratio=0.1,
    )

    assert shift == (1.0, -2.0, 3.0)
    np.testing.assert_array_equal(observed["reference_mask"], fixed_mask)
    np.testing.assert_array_equal(observed["moving_mask"], moving_mask)
    assert observed["overlap_ratio"] == 0.1
    assert "upsample_factor" not in observed


def test_masked_phase_requires_both_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_gpu_modules(monkeypatch)

    with pytest.raises(ValueError, match="reference_mask and moving_mask must be provided together"):
        estimate_translation_gpu(
            np.zeros((3, 3, 3), dtype=np.float32),
            np.zeros((3, 3, 3), dtype=np.float32),
            reference_mask=np.ones((3, 3, 3), dtype=bool),
        )
