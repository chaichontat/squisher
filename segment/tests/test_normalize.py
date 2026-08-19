from __future__ import annotations

import numpy as np
import pytest

from squisher_segment.segment import normalize
from squisher_segment.segment.normalize import sample_percentiles


def test_sample_percentiles_accepts_block_equal_to_image() -> None:
    image = np.full((2, 4, 6, 1), 100, dtype=np.uint16)

    percentiles, samples = sample_percentiles(
        image,
        channels=[1],
        block=(4, 6),
        n=1,
        unsharp=False,
    )

    np.testing.assert_array_equal(percentiles, [[100, 100]])
    np.testing.assert_array_equal(samples, [[[100], [100]]])


def test_sample_percentiles_reads_bounded_z_samples() -> None:
    image = np.empty((20, 4, 6, 1), dtype=np.uint16)
    for z_index in range(image.shape[0]):
        image[z_index] = z_index + 10

    percentiles, samples = sample_percentiles(
        image,
        channels=[1],
        block=(4, 6),
        n=1,
        z_samples=3,
        unsharp=False,
    )

    sampled = image[[5, 10, 16]]
    expected = np.percentile(sampled, [1, 99], axis=(0, 1, 2))
    np.testing.assert_allclose(percentiles, expected.T)
    np.testing.assert_allclose(samples, expected[None])


def test_z_samples_are_seeded_and_stratified() -> None:
    np.testing.assert_array_equal(
        normalize._sample_z_indices(20, 3, seed=0),
        [5, 10, 16],
    )


def test_sample_percentiles_uses_gpu_unsharp(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((5, 4, 6, 1), 100, dtype=np.uint16)
    observed: list[tuple[tuple[int, ...], float]] = []

    def fake_gpu_unsharp(sampled: np.ndarray, *, radius: float) -> np.ndarray:
        observed.append((sampled.shape, radius))
        return sampled.astype(np.float32)

    monkeypatch.setattr(normalize, "_gpu_unsharp_planes", fake_gpu_unsharp)
    sample_percentiles(
        image,
        channels=[1],
        block=(4, 6),
        n=1,
        z_samples=3,
        unsharp_radius=2.5,
    )

    assert observed == [((3, 4, 6, 1), 2.5)]


def test_sample_percentiles_rejects_nonpositive_z_samples() -> None:
    image = np.full((2, 4, 6, 1), 100, dtype=np.uint16)

    with pytest.raises(ValueError, match="z_samples"):
        sample_percentiles(image, channels=[1], block=(4, 6), z_samples=0)
