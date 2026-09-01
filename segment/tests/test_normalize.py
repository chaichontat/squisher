from __future__ import annotations

import sys
import types

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


def test_sample_percentiles_reads_only_spatial_crops() -> None:
    image = np.full((5, 10, 12, 1), 100, dtype=np.uint16)
    reads: list[tuple[object, ...]] = []

    class TrackingArray:
        ndim = image.ndim
        shape = image.shape

        def __getitem__(self, key: tuple[object, ...]) -> np.ndarray:
            reads.append(key)
            return image[key]

    sample_percentiles(
        TrackingArray(),
        channels=[1],
        block=(4, 6),
        n=1,
        z_samples=3,
        unsharp=False,
    )

    assert len(reads) == 3
    assert all(key[1] != slice(None) and key[2] != slice(None) for key in reads)
    assert all(key[1].stop - key[1].start == 4 for key in reads)  # type: ignore[union-attr]
    assert all(key[2].stop - key[2].start == 6 for key in reads)  # type: ignore[union-attr]


def test_sample_percentiles_uses_foreground_voxels_in_sparse_crop() -> None:
    image = np.zeros((2, 4, 6, 3), dtype=np.uint16)
    image[:, 1:3, 2:4, 0] = 100
    image[:, 1:3, 2:4, 1] = 2_000
    image[:, 1:3, 2:4, 2] = 300
    foreground = image[..., 1] > 1_000

    percentiles, samples = sample_percentiles(
        image,
        channels=[1, 2, 3],
        block=(4, 6),
        n=1,
        unsharp=False,
        foreground_channel=2,
        foreground_threshold=1_000,
    )

    expected = np.percentile(image[foreground], [1, 99], axis=0)
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


def test_gpu_unsharp_uploads_one_plane_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    uploads: list[tuple[int, ...]] = []
    pool = types.SimpleNamespace(free_all_blocks=lambda: None)
    stream = types.SimpleNamespace(synchronize=lambda: None)
    cupy = types.ModuleType("cupy")
    cupy.float32 = np.float32  # type: ignore[attr-defined]
    cupy.asarray = lambda value, dtype: (  # type: ignore[attr-defined]
        uploads.append(np.shape(value)) or np.asarray(value, dtype=dtype)
    )
    cupy.asnumpy = np.asarray  # type: ignore[attr-defined]
    cupy.get_default_memory_pool = lambda: pool  # type: ignore[attr-defined]
    cupy.get_default_pinned_memory_pool = lambda: pool  # type: ignore[attr-defined]
    cupy.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        runtime=types.SimpleNamespace(getDeviceCount=lambda: 1),
        get_current_stream=lambda: stream,
    )
    filters = types.SimpleNamespace(
        unsharp_mask=lambda image, **_kwargs: image + np.float32(1)
    )
    cucim_skimage = types.ModuleType("cucim.skimage")
    cucim_skimage.filters = filters  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cucim.skimage", cucim_skimage)

    sampled = np.arange(3 * 4 * 6, dtype=np.uint16).reshape(3, 4, 6, 1)
    result = normalize._gpu_unsharp_planes(sampled, radius=2.5)

    assert uploads == [(4, 6, 1)] * 3
    np.testing.assert_array_equal(result, sampled.astype(np.float32) + 1)


def test_sample_percentiles_rejects_nonpositive_z_samples() -> None:
    image = np.full((2, 4, 6, 1), 100, dtype=np.uint16)

    with pytest.raises(ValueError, match="z_samples"):
        sample_percentiles(image, channels=[1], block=(4, 6), z_samples=0)
