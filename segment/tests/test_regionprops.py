from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr
from typer.testing import CliRunner

from squisher_segment.cli import app
from squisher_segment.segment import regionprops


def test_chunk_measurements_reduce_across_xyz_seams() -> None:
    volume = np.zeros((4, 6, 6), dtype=np.uint32)
    volume[1:4, 1:5, 2:5] = 7
    volume[0:2, 4:6, 0:2] = 9

    moments: list[pl.DataFrame] = []
    planes: list[pl.DataFrame] = []
    for z0 in (0, 2):
        for y0 in (0, 3):
            for x0 in (0, 3):
                measured, per_plane = regionprops.measure_block(
                    volume[z0 : z0 + 2, y0 : y0 + 3, x0 : x0 + 3],
                    offset_zyx=(z0, y0, x0),
                )
                moments.append(measured)
                planes.append(per_plane)

    cells = regionprops.reduce_measurements(moments=moments, planes=planes)
    row = cells.filter(pl.col("label") == 7).row(0, named=True)
    coordinates = np.argwhere(volume == 7)

    assert row["area"] == len(coordinates)
    assert row["centroid_z"] == np.mean(coordinates[:, 0])
    assert row["centroid_y"] == np.mean(coordinates[:, 1])
    assert row["centroid_x"] == np.mean(coordinates[:, 2])
    assert row["plane_z"] == 2
    assert row["plane_area"] == 12


def test_max_area_plane_tie_prefers_centroid_then_lower_z() -> None:
    volume = np.zeros((5, 3, 3), dtype=np.uint32)
    volume[1, :2, :2] = 4
    volume[3, :2, :2] = 4

    moments, planes = regionprops.measure_block(volume, offset_zyx=(0, 0, 0))
    cells = regionprops.reduce_measurements(moments=[moments], planes=[planes])

    assert cells.row(0, named=True)["plane_z"] == 1


@pytest.mark.parametrize(
    ("block", "offset", "message"),
    [
        (np.zeros((2, 2), dtype=np.uint32), (0, 0, 0), "3D"),
        (np.zeros((2, 2, 2), dtype=np.float32), (0, 0, 0), "integer"),
        (np.zeros((2, 2, 2), dtype=np.int32), (-1, 0, 0), "nonnegative"),
    ],
)
def test_measure_block_rejects_unsupported_inputs(
    block: np.ndarray,
    offset: tuple[int, int, int],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        regionprops.measure_block(block, offset_zyx=offset)


@pytest.mark.parametrize("workers", [1, 2])
def test_measure_zarr_writes_global_props_and_cleans_partials(
    tmp_path: Path,
    workers: int,
) -> None:
    labels_path = tmp_path / "labels.zarr"
    labels = zarr.create_array(
        labels_path,
        shape=(4, 6, 6),
        chunks=(2, 3, 3),
        dtype=np.uint32,
    )
    volume = np.zeros(labels.shape, dtype=np.uint32)
    volume[1:4, 1:5, 2:5] = 7
    volume[0:2, 4:6, 0:2] = 9
    labels[:] = volume
    output = tmp_path / "props.parquet"

    written = regionprops.measure_zarr(
        labels_path,
        output,
        workers=workers,
        offset_zyx=(10, 20, 30),
    )

    assert written == output
    cells = pl.read_parquet(output)
    assert cells["label"].to_list() == [7, 9]
    np.testing.assert_allclose(
        cells.filter(pl.col("label") == 7).select(
            "centroid_z", "centroid_y", "centroid_x"
        ).row(0),
        np.argwhere(volume == 7).mean(axis=0) + np.asarray((10, 20, 30)),
    )
    assert not (tmp_path / ".props-parts").exists()


def test_regionprops_cli_writes_default_output(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.zarr"
    labels = zarr.create_array(
        labels_path,
        shape=(2, 3, 3),
        chunks=(1, 3, 3),
        dtype=np.uint32,
    )
    labels[0, 1:, 1:] = 5

    result = CliRunner().invoke(
        app,
        ["regionprops", str(labels_path), "--workers", "1"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "props.parquet").is_file()
