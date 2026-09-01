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


def test_intensity_measurements_reduce_across_xyz_seams() -> None:
    labels = np.zeros((4, 6, 6), dtype=np.uint32)
    labels[1:4, 1:5, 2:5] = 7
    labels[0:2, 4:6, 0:2] = 9
    intensity = np.arange(labels.size, dtype=np.uint16).reshape(labels.shape)
    inverse = np.asarray(1000 - intensity, dtype=np.uint16)

    moments: list[pl.DataFrame] = []
    planes: list[pl.DataFrame] = []
    for z0 in (0, 2):
        for y0 in (0, 3):
            for x0 in (0, 3):
                measured, per_plane = regionprops.measure_block(
                    labels[z0 : z0 + 2, y0 : y0 + 3, x0 : x0 + 3],
                    offset_zyx=(z0, y0, x0),
                    intensity_image=np.stack(
                        (
                            intensity[z0 : z0 + 2, y0 : y0 + 3, x0 : x0 + 3],
                            inverse[z0 : z0 + 2, y0 : y0 + 3, x0 : x0 + 3],
                        ),
                        axis=-1,
                    ),
                    intensity_names=("edu", "brdu"),
                )
                moments.append(measured)
                planes.append(per_plane)

    cells = regionprops.reduce_measurements(moments=moments, planes=planes)
    row = cells.filter(pl.col("label") == 7).row(0, named=True)
    edu_values = intensity[labels == 7]
    brdu_values = inverse[labels == 7]

    assert row["intensity_edu_min"] == edu_values.min()
    assert row["intensity_edu_mean"] == edu_values.mean()
    assert row["intensity_edu_max"] == edu_values.max()
    assert row["intensity_brdu_min"] == brdu_values.min()
    assert row["intensity_brdu_mean"] == brdu_values.mean()
    assert row["intensity_brdu_max"] == brdu_values.max()


def test_intensity_nan_propagates_across_chunks() -> None:
    labels = np.ones((2, 1, 2), dtype=np.uint32)
    intensity = np.asarray([[[1.0, 2.0]], [[3.0, np.nan]]])
    parts = [
        regionprops.measure_block(
            labels[z : z + 1],
            offset_zyx=(z, 0, 0),
            intensity_image=intensity[z : z + 1],
            intensity_names=("signal",),
        )
        for z in range(2)
    ]

    cells = regionprops.reduce_measurements(
        moments=[part[0] for part in parts],
        planes=[part[1] for part in parts],
    )

    assert np.isnan(cells["intensity_signal_min"][0])
    assert np.isnan(cells["intensity_signal_mean"][0])
    assert np.isnan(cells["intensity_signal_max"][0])


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
    intensity_path = tmp_path / "intensity.zarr"
    intensity = zarr.create_array(
        intensity_path,
        shape=labels.shape,
        chunks=(1, 6, 2),
        dtype=np.uint16,
    )
    intensity_volume = np.arange(volume.size, dtype=np.uint16).reshape(volume.shape)
    intensity[:] = intensity_volume
    inverse_path = tmp_path / "inverse.zarr"
    inverse = zarr.create_array(
        inverse_path,
        shape=labels.shape,
        chunks=(4, 2, 6),
        dtype=np.uint16,
    )
    inverse_volume = np.asarray(1000 - intensity_volume, dtype=np.uint16)
    inverse[:] = inverse_volume
    output = tmp_path / "props.parquet"

    written = regionprops.measure_zarr(
        labels_path,
        output,
        intensity_paths={"edu": intensity_path, "brdu": inverse_path},
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
    row = cells.filter(pl.col("label") == 7).row(0, named=True)
    edu_values = intensity_volume[volume == 7]
    brdu_values = inverse_volume[volume == 7]
    assert row["intensity_edu_min"] == edu_values.min()
    assert row["intensity_edu_mean"] == edu_values.mean()
    assert row["intensity_edu_max"] == edu_values.max()
    assert row["intensity_brdu_min"] == brdu_values.min()
    assert row["intensity_brdu_mean"] == brdu_values.mean()
    assert row["intensity_brdu_max"] == brdu_values.max()
    assert not (tmp_path / ".props-parts").exists()


def test_measure_zarr_rejects_mismatched_intensity_shape(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.zarr"
    zarr.create_array(labels_path, shape=(2, 3, 3), chunks=(1, 3, 3), dtype=np.uint32)
    intensity_path = tmp_path / "intensity.zarr"
    zarr.create_array(intensity_path, shape=(2, 3, 2), chunks=(1, 3, 2), dtype=np.uint16)

    with pytest.raises(ValueError, match="identical shapes"):
        regionprops.measure_zarr(
            labels_path,
            tmp_path / "props.parquet",
            intensity_paths={"signal": intensity_path},
            workers=1,
        )


def test_measure_zarr_rejects_mixed_intensity_dtypes(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.zarr"
    zarr.create_array(labels_path, shape=(2, 3, 3), chunks=(1, 3, 3), dtype=np.uint32)
    first_path = tmp_path / "first.zarr"
    zarr.create_array(first_path, shape=(2, 3, 3), chunks=(1, 3, 3), dtype=np.uint16)
    second_path = tmp_path / "second.zarr"
    zarr.create_array(second_path, shape=(2, 3, 3), chunks=(1, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="identical dtypes"):
        regionprops.measure_zarr(
            labels_path,
            tmp_path / "props.parquet",
            intensity_paths={"first": first_path, "second": second_path},
            workers=1,
        )


def test_regionprops_cli_writes_default_output(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.zarr"
    labels = zarr.create_array(
        labels_path,
        shape=(2, 3, 3),
        chunks=(1, 3, 3),
        dtype=np.uint32,
    )
    labels[0, 1:, 1:] = 5
    intensity_path = tmp_path / "intensity.ome.zarr"
    intensity_root = zarr.create_group(intensity_path)
    intensity_root.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    intensity = intensity_root.create_array(
        "0",
        shape=labels.shape,
        chunks=labels.chunks,
        dtype=np.uint16,
    )
    intensity[0, 1:, 1:] = np.asarray([[2, 4], [6, 8]], dtype=np.uint16)

    result = CliRunner().invoke(
        app,
        [
            "regionprops",
            str(labels_path),
            "--intensity",
            f"edu={intensity_path}",
            "--intensity",
            f"brdu={intensity_path}",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    cells = pl.read_parquet(tmp_path / "props.parquet")
    assert cells.select(
        "intensity_edu_min",
        "intensity_edu_mean",
        "intensity_edu_max",
        "intensity_brdu_min",
        "intensity_brdu_mean",
        "intensity_brdu_max",
    ).row(0) == (2.0, 5.0, 8.0, 2.0, 5.0, 8.0)
