from __future__ import annotations

import numpy as np
import tifffile

from squisher_deconv.planning import SamplePlane, group_sample_windows, uniform_sample_planes


def _write_stack(path, *, z: int, channels: int) -> None:
    data = np.arange(z * channels * 3 * 4, dtype=np.uint16).reshape(z * channels, 3, 4)
    tifffile.imwrite(path, data, photometric="minisblack")


def test_uniform_sampling_is_over_all_files_with_seed(tmp_path) -> None:
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    _write_stack(a, z=2, channels=2)
    _write_stack(b, z=4, channels=2)

    first = uniform_sample_planes([a, b], planes=4, channels=2, seed=10)
    second = uniform_sample_planes([a, b], planes=4, channels=2, seed=10)

    assert first == second
    assert {(sample.file_index, sample.true_z) for sample in first}.issubset(
        {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)}
    )
    assert len({(sample.file_index, sample.true_z) for sample in first}) == 4


def test_uniform_sampling_caps_requested_planes_at_total(tmp_path) -> None:
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    _write_stack(a, z=2, channels=2)
    _write_stack(b, z=4, channels=2)

    samples = uniform_sample_planes([a, b], planes=10, channels=2, seed=10)

    assert [(sample.file_index, sample.true_z) for sample in samples] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]


def test_group_sample_windows_preserves_requested_core_planes(tmp_path) -> None:
    path = tmp_path / "a.tif"
    samples = [
        SamplePlane(0, path, 1),
        SamplePlane(0, path, 2),
        SamplePlane(0, path, 8),
        SamplePlane(0, path, 9),
    ]

    windows = group_sample_windows(samples, z_counts=[12], halo=1)

    assert [(w.read_start, w.read_stop, w.core_z) for w in windows] == [
        (0, 4, (1, 2)),
        (7, 11, (8, 9)),
    ]
    assert sorted(z for window in windows for z in window.core_z) == [1, 2, 8, 9]
