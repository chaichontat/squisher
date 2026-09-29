from __future__ import annotations

import numpy as np
import pytest

from squisher_segment.segment import postproc3d


def test_gaussian_batches_preserve_order_and_memory_cap() -> None:
    slices = [
        (slice(5, 7), slice(10, 12), slice(10, 12)),
        (slice(5, 7), slice(20, 22), slice(20, 22)),
        None,
        (slice(5, 15), slice(30, 40), slice(30, 40)),
    ]
    batches = list(
        postproc3d._iter_gaussian_label_batches(
            slices,
            volume_shape=(20, 100, 100),
            margin=2,
            workspace_bytes=12 * 2 * 6**3,
        )
    )

    assert [[entry[0] for entry in entries] for entries, _ in batches] == [[1, 2], [4]]
    assert [shape for _, shape in batches] == [(6, 6, 6), (14, 14, 14)]


def test_gaussian_batch_offsets_preserve_physical_boundaries() -> None:
    batch_shape = (8, 8, 8)
    volume_shape = (20, 20, 20)

    assert postproc3d._gaussian_batch_offset(
        (slice(0, 5), slice(4, 9), slice(4, 9)),
        batch_shape,
        volume_shape,
    ) == (0, 0, 0)
    assert postproc3d._gaussian_batch_offset(
        (slice(15, 20), slice(11, 16), slice(11, 16)),
        batch_shape,
        volume_shape,
    ) == (3, 0, 0)


def test_gaussian_batches_reject_nonpositive_workspace() -> None:
    with pytest.raises(ValueError, match="workspace_bytes must be positive"):
        list(
            postproc3d._iter_gaussian_label_batches(
                [(slice(1, 2), slice(1, 2), slice(1, 2))],
                volume_shape=(3, 3, 3),
                margin=1,
                workspace_bytes=0,
            )
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("sigma", "bg_scale", "max_expansion"),
    [
        ((1.0, 2.0, 2.0), 1.0, 1),
        (1.5, 0.9, 0),
    ],
)
def test_batched_gaussian_matches_singleton_boundary_voting(
    sigma: float | tuple[float, float, float],
    bg_scale: float,
    max_expansion: int,
) -> None:
    masks = np.zeros((12, 48, 48), dtype=np.int32)
    masks[0:4, 0:8, 0:8] = 1
    masks[-4:, -8:, -8:] = 3
    masks[3:6, 12:17, 20:25] = 7
    masks[3:6, 24:29, 20:25] = 12

    singleton = postproc3d.gaussian_smooth_labels_cupy(
        masks,
        sigma=sigma,
        in_place=False,
        bg_scale=bg_scale,
        max_expansion=max_expansion,
        batch_workspace_bytes=1,
    )
    batched = postproc3d.gaussian_smooth_labels_cupy(
        masks,
        sigma=sigma,
        in_place=False,
        bg_scale=bg_scale,
        max_expansion=max_expansion,
    )

    np.testing.assert_array_equal(batched, singleton)
