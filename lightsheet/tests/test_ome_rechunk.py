from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from zarr.codecs import BytesCodec, ZstdCodec

from squisher_lightsheet import mvs_seams
from squisher_lightsheet.ome_rechunk import rechunk_ome_tiffs
from squisher_lightsheet.ome_zarr_rechunk import rechunk_ome_zarr


def xy_block_mean(data: np.ndarray, factor: int) -> np.ndarray:
    usable_y = (data.shape[-2] // factor) * factor
    usable_x = (data.shape[-1] // factor) * factor
    trimmed = data[..., :usable_y, :usable_x]
    reduced = trimmed.reshape(*trimmed.shape[:-2], usable_y // factor, factor, usable_x // factor, factor).mean(
        axis=(-3, -1)
    )
    return np.rint(reduced).astype(data.dtype)


def test_rechunk_ome_tiff_writes_chunked_ome_zarr_tile(tmp_path) -> None:
    data = np.arange(5 * 9 * 10, dtype=np.uint16).reshape(5, 9, 10)
    source = tmp_path / "tile0.ome.tif"
    tifffile.imwrite(source, data, metadata={"axes": "ZYX"})

    summary = rechunk_ome_tiffs(
        inputs=[source],
        output_dir=tmp_path / "rechunked",
        chunk_shape_zyx=(2, 4, 4),
    )

    output = tmp_path / "rechunked" / "tile0.ome.zarr"
    array = zarr.open_array(str(output / "0"), mode="r")
    assert tuple(array.shape) == data.shape
    assert tuple(array.chunks) == (2, 4, 4)
    assert array.attrs["_ARRAY_DIMENSIONS"] == ["z", "y", "x"]
    np.testing.assert_array_equal(array[:], data)
    assert summary["outputs"][0]["chunks"] == [2, 4, 4]
    assert summary["outputs"][0]["pyramid_downsample_factors"] == [2, 4]
    assert summary["outputs"][0]["read_strategy"] == "z_slab_full_yx"

    level1 = zarr.open_array(str(output / "1"), mode="r")
    level2 = zarr.open_array(str(output / "2"), mode="r")
    assert tuple(level1.shape) == (5, 4, 5)
    assert tuple(level1.chunks) == (2, 4, 4)
    assert tuple(level2.shape) == (5, 2, 2)
    assert tuple(level2.chunks) == (2, 2, 2)
    np.testing.assert_array_equal(level1[:], xy_block_mean(data, 2))
    np.testing.assert_array_equal(level2[:], xy_block_mean(data, 4))

    root = zarr.open_group(str(output), mode="r")
    multiscale = root.attrs["multiscales"][0]
    assert [dataset["path"] for dataset in multiscale["datasets"]] == ["0", "1", "2"]
    assert [dataset["coordinateTransformations"][0]["scale"] for dataset in multiscale["datasets"]] == [
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0],
        [1.0, 4.0, 4.0],
    ]


def test_rechunk_ome_tiff_uses_deconv_sidecar_channels_for_flattened_zyx(tmp_path) -> None:
    data = np.arange(2 * 5 * 9 * 10, dtype=np.uint16).reshape(10, 9, 10)
    source = tmp_path / "tile0.ome.tif"
    tifffile.imwrite(source, data, metadata={"axes": "ZYX"})
    source.with_suffix(".deconv.json").write_text(
        json.dumps({"provenance": {"run_settings": {"channels": 2}}})
    )

    summary = rechunk_ome_tiffs(
        inputs=[source],
        output_dir=tmp_path / "rechunked",
        chunk_shape_zyx=(2, 4, 4),
    )

    output = tmp_path / "rechunked" / "tile0.ome.zarr"
    level0 = zarr.open_array(str(output / "0"), mode="r")
    level1 = zarr.open_array(str(output / "1"), mode="r")
    assert tuple(level0.shape) == (2, 5, 9, 10)
    assert tuple(level0.chunks) == (1, 2, 4, 4)
    assert level0.attrs["_ARRAY_DIMENSIONS"] == ["c", "z", "y", "x"]
    np.testing.assert_array_equal(level0[:], np.moveaxis(data.reshape(5, 2, 9, 10), 1, 0))
    assert tuple(level1.shape) == (2, 5, 4, 5)
    assert tuple(level1.chunks) == (1, 2, 4, 4)
    assert summary["outputs"][0]["axes"] == "CZYX"
    assert summary["outputs"][0]["shape"] == [2, 5, 9, 10]
    assert summary["outputs"][0]["chunks"] == [1, 2, 4, 4]

    root = zarr.open_group(str(output), mode="r")
    multiscale = root.attrs["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["c", "z", "y", "x"]
    assert [dataset["coordinateTransformations"][0]["scale"] for dataset in multiscale["datasets"]] == [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 1.0, 4.0, 4.0],
    ]


def test_rechunk_ome_tiffs_parallelizes_across_tiles_and_preserves_summary_order(tmp_path) -> None:
    first = tmp_path / "a.ome.tif"
    second = tmp_path / "b.ome.tif"
    tifffile.imwrite(first, np.ones((3, 9, 10), dtype=np.uint16), metadata={"axes": "ZYX"})
    tifffile.imwrite(second, np.full((3, 9, 10), 2, dtype=np.uint16), metadata={"axes": "ZYX"})

    summary = rechunk_ome_tiffs(
        inputs=[second, first],
        output_dir=tmp_path / "rechunked",
        chunk_shape_zyx=(1, 2, 2),
        workers=2,
    )

    assert summary["workers"] == 2
    assert summary["pyramid_downsample_factors"] == [2, 4]
    assert [record["source"] for record in summary["outputs"]] == [str(first), str(second)]
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "rechunked" / "a.ome.zarr" / "0"), mode="r")[:], 1)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "rechunked" / "b.ome.zarr" / "0"), mode="r")[:], 2)


def test_mvs_image_crop_reads_rechunked_ome_zarr(tmp_path) -> None:
    data = np.arange(2 * 5 * 9 * 10, dtype=np.uint16).reshape(2, 5, 9, 10)
    source = tmp_path / "tile0.ome.tif"
    tifffile.imwrite(source, data, metadata={"axes": "CZYX"})
    rechunk_ome_tiffs(
        inputs=[source],
        output_dir=tmp_path / "rechunked",
        chunk_shape_zyx=(2, 4, 4),
    )

    path = tmp_path / "rechunked" / "tile0.ome.zarr"
    axes, level, shape = mvs_seams._image_level_metadata(path, level=0)
    crop = mvs_seams._read_image_level_crop(
        path,
        level=level,
        axes=axes,
        channel=1,
        slices_zyx=(slice(1, 4), slice(2, 7), slice(3, 8)),
    )

    assert axes == "CZYX"
    assert shape == data.shape
    np.testing.assert_array_equal(crop, data[1, 1:4, 2:7, 3:8].astype(np.float32))


def _write_zarr_source(path: Path, *, levels: int = 4, complete: bool = True) -> list[np.ndarray]:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    shapes = [(4, 16, 20), (4, 8, 10), (2, 4, 5), (2, 2, 3)][:levels]
    transforms = [
        [{"type": "scale", "scale": [float(index + 1), float(2**index), float(2**index)]}]
        for index in range(levels)
    ]
    root.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": "fixture",
                        "axes": [
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {"path": str(index), "coordinateTransformations": transform}
                            for index, transform in enumerate(transforms)
                        ],
                    }
                ],
            },
            "fixture_metadata": {"preserve": True},
            "squisher_complete": complete,
        }
    )
    values = []
    for index, shape in enumerate(shapes):
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + index * 1000
        array = root.create_array(
            str(index),
            shape=shape,
            chunks=tuple(min(size, chunk) for size, chunk in zip(shape, (2, 4, 4), strict=True)),
            dtype=np.uint16,
            fill_value=7,
            attributes={"_ARRAY_DIMENSIONS": ["z", "y", "x"], "level_attr": index},
            dimension_names=("z", "y", "x"),
            serializer=BytesCodec(),
            compressors=[ZstdCodec(level=1)],
        )
        array[:] = data
        values.append(data)
    return values


def test_rechunk_ome_zarr_rebases_levels_and_writes_lossless_zstd(tmp_path: Path) -> None:
    source = tmp_path / "source.ome.zarr"
    expected = _write_zarr_source(source)
    destination = tmp_path / "preview.ome.zarr"

    summary = rechunk_ome_zarr(
        source=source,
        destination=destination,
        start_level=2,
        zstd_level=3,
        chunk_shape_zyx=(2, 4, 4),
        workers=2,
        codec_concurrency=2,
    )

    root = zarr.open_group(str(destination), mode="r")
    assert sorted(root.keys()) == ["0", "1"]
    assert root.attrs["squisher_complete"] is True
    assert root.attrs["fixture_metadata"] == {"preserve": True}
    assert root.attrs["squisher_rechunk"]["source_start_level"] == 2
    multiscale = root.attrs["ome"]["multiscales"][0]
    assert [dataset["path"] for dataset in multiscale["datasets"]] == ["0", "1"]
    assert [dataset["coordinateTransformations"] for dataset in multiscale["datasets"]] == [
        [{"type": "scale", "scale": [3.0, 4.0, 4.0]}],
        [{"type": "scale", "scale": [4.0, 8.0, 8.0]}],
    ]

    level0 = root["0"]
    level1 = root["1"]
    assert tuple(level0.metadata.dimension_names) == ("z", "y", "x")
    assert level0.attrs.asdict() == {"_ARRAY_DIMENSIONS": ["z", "y", "x"], "level_attr": 2}
    assert level0.fill_value == 7
    assert tuple(level0.chunks) == (2, 4, 4)
    assert tuple(level1.chunks) == (2, 2, 2)
    np.testing.assert_array_equal(level0[:], expected[2])
    np.testing.assert_array_equal(level1[:], expected[3])
    assert [codec.to_dict()["name"] for codec in level0.metadata.codecs] == ["bytes", "zstd"]
    assert level0.metadata.codecs[-1].to_dict()["configuration"]["level"] == 3
    assert summary["source_start_level"] == 2
    assert [level["source_path"] for level in summary["levels"]] == ["2", "3"]


def test_rechunk_ome_zarr_rejects_insufficient_levels(tmp_path: Path) -> None:
    source = tmp_path / "short.ome.zarr"
    _write_zarr_source(source, levels=2)

    with pytest.raises(ValueError, match="has 2 pyramid level"):
        rechunk_ome_zarr(source=source, destination=tmp_path / "out.ome.zarr")


def test_rechunk_ome_zarr_rejects_incomplete_source(tmp_path: Path) -> None:
    source = tmp_path / "incomplete.ome.zarr"
    _write_zarr_source(source, complete=False)

    with pytest.raises(ValueError, match="explicitly marked incomplete"):
        rechunk_ome_zarr(source=source, destination=tmp_path / "out.ome.zarr")


def test_rechunk_ome_zarr_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.ome.zarr"
    _write_zarr_source(source)
    destination = tmp_path / "existing.ome.zarr"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        rechunk_ome_zarr(source=source, destination=destination)
