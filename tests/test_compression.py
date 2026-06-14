from concurrent.futures import Future
from contextlib import contextmanager
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pylibCZIrw import czi
from tifffile import TiffFile, TiffWriter

from squisher.compression import (
    CZI_PROGRESS_INTERVAL,
    _block_mean_xy,
    _czi_output_dir,
    _czi_progress_steps,
    _czi_subblock_metadata,
    _progress_bar,
    _report_czi_plane_progress,
    _stitched_preview_path,
    _stitch_tile_preview_images,
    _start_progress_queue_listener,
    _verify_ome_tiff,
    _write_czi_tile,
    _write_czi_tile_process,
    compress_czi_to_ome_tiff,
    infer_stage_positions_from_pos,
    verify_czi_ome_tiff_outputs,
)


OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def _write_sample_czi(path: Path, data: np.ndarray) -> None:
    with czi.create_czi(str(path)) as writer:
        for channel_index in range(data.shape[0]):
            for z_index in range(data.shape[1]):
                assert writer.write(data[channel_index, z_index], plane={"C": channel_index, "Z": z_index}, scene=0)


def _write_multi_tile_czi(path: Path, tiles: list[np.ndarray]) -> None:
    with czi.create_czi(str(path)) as writer:
        for tile_index, data in enumerate(tiles):
            for channel_index in range(data.shape[0]):
                for z_index in range(data.shape[1]):
                    assert writer.write(
                        data[channel_index, z_index],
                        location=(tile_index * 10, tile_index * 20),
                        plane={"C": channel_index, "Z": z_index},
                        scene=0,
                    )


def _write_pos_file(path: Path, points: list[tuple[float, float, float]]) -> None:
    lines = [
        "Carl Zeiss LSM 510 - Position list file - Version = 1.000",
        "BEGIN PositionList Version = 10000",
        f"\tNumberPositions = {len(points)}",
    ]
    for index, (x, y, z) in enumerate(points, start=1):
        lines.extend(
            [
                f"\tBEGIN Position{index} Version = 10000",
                f"\t\tX = {x} um",
                f"\t\tY = {y} um",
                f"\t\tZ = {z} um",
                "\tEND",
            ]
        )
    lines.append("END")
    path.write_text("\n".join(lines))


def _minimal_ome_xml_bytes(*, size_x: int, size_y: int, tiff_pyramid: bool | None = None) -> bytes:
    annotation_ref = ""
    structured_annotations = ""
    if tiff_pyramid is not None:
        annotation_ref = '<AnnotationRef ID="Annotation:0"/>'
        structured_annotations = f"""
  <StructuredAnnotations>
    <MapAnnotation ID="Annotation:0" Namespace="squisher">
      <Value>
        <M K="squisher.tiff_pyramid">{json.dumps(tiff_pyramid)}</M>
      </Value>
    </MapAnnotation>
  </StructuredAnnotations>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    {annotation_ref}
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
            SizeX="{size_x}" SizeY="{size_y}" SizeZ="1" SizeC="1" SizeT="1">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"><LightPath/></Channel>
      <TiffData IFD="0" PlaneCount="1"/>
      <Plane TheC="0" TheZ="0" TheT="0"/>
    </Pixels>
  </Image>
  {structured_annotations}
</OME>
""".encode()


def test_block_mean_xy_downsamples_integer_plane_with_cpu_fallback(monkeypatch) -> None:
    plane = np.arange(4 * 4, dtype=np.uint16).reshape(4, 4)

    monkeypatch.setattr("squisher.compression._gpu_block_reduce_modules", lambda: None)
    reduced = _block_mean_xy(plane, factor=2)

    np.testing.assert_array_equal(reduced, np.array([[2, 4], [10, 12]], dtype=np.uint16))


def test_block_mean_xy_uses_gpu_result_when_available(monkeypatch) -> None:
    class FakeCupy:
        float32 = np.float32
        mean = np.mean

        @staticmethod
        def asarray(data, dtype):
            return np.asarray(data, dtype=dtype)

        @staticmethod
        def rint(data):
            return np.rint(data)

        @staticmethod
        def clip(data, lo, hi):
            return np.clip(data, lo, hi)

        @staticmethod
        def asnumpy(data):
            return np.asarray(data)

    monkeypatch.setattr(
        "squisher.compression._gpu_block_reduce_modules",
        lambda: (FakeCupy, lambda *_args, **_kwargs: np.full((2, 2), 7.0, dtype=np.float32)),
    )

    np.testing.assert_array_equal(_block_mean_xy(np.zeros((4, 4), dtype=np.uint16), factor=2), np.full((2, 2), 7))


def test_compress_czi_writes_tiled_ome_tiff_with_raw_metadata(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 3, 4, 5))
    _write_sample_czi(czi_path, data)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1)

    with TiffFile(tmp_path / "sample.ome.tif") as tif:
        assert tif.pages[0].compression == 22610
        assert tif.pages[0].is_tiled
        annotations = _map_annotation_values(tif.ome_metadata)
        assert annotations["squisher.source_format"] == "CZI"
        assert annotations["squisher.source_file"] == "sample.czi"
        assert annotations["squisher.compression"] == "JPEG-XR"
        assert annotations["squisher.compression_tiff_tag"] == "22610"
        assert annotations["squisher.compression_level_input"] == "90"
        assert annotations["squisher.compression_level_normalized"] == "0.9"
        assert annotations["squisher.tiff_tile_size"] == "16"
        assert "czi.global_metadata_xml" not in annotations
        assert "czi.subblock_metadata_xml" not in annotations
        shared_metadata = _shared_czi_metadata(tif.ome_metadata)
        assert shared_metadata.tag == "ImageDocument"
        raw_metadata = _raw_czi_metadata(tif.ome_metadata)
        assert raw_metadata.find("GlobalMetadata") is None
        assert raw_metadata.find("SubblockMetadata")[0].tag == "Subblocks"
        readback = tif.asarray()
        if tif.series[0].axes == "TCZYX":
            readback = readback[0]
        assert readback.shape == data.shape
        assert readback.dtype == data.dtype
        assert abs(float(readback.mean()) - float(data.mean())) < 10
    assert verify_czi_ome_tiff_outputs(czi_path, decode_samples=True)


def test_compress_czi_writes_center_z_thumbnail_by_default(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape((2, 3, 16, 16))
    _write_sample_czi(czi_path, data)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnail_size=8)

    png = tmp_path / "sample.center-z.png"
    assert png.exists()
    assert png.stat().st_size > 0


def test_compress_czi_writes_complete_ome_zarr(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    czi_path = tmp_path / "sample.czi"
    data = np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape((2, 3, 16, 16))
    _write_sample_czi(czi_path, data)

    assert compress_czi_to_ome_tiff(
        czi_path,
        output_format="ome-zarr",
        level=90,
        maxworkers=1,
        thumbnails=True,
    )

    out = tmp_path / "sample.ome.zarr"
    array = zarr.open(str(out / "0"), mode="r")
    assert array.shape == (1, 2, 3, 16, 16)
    assert array.dtype == data.dtype
    root = zarr.open_group(str(out), mode="r")
    assert root.attrs["multiscales"][0]["datasets"][0]["path"] == "0"
    assert root.attrs["squisher_complete"] is True
    assert (tmp_path / "sample.center-z.png").exists()


def test_compress_czi_writes_each_illumination_as_separate_ome_tiff(tmp_path: Path, monkeypatch) -> None:
    class Box:
        x = 0
        y = 0
        w = 16
        h = 16

    class FakeCziFile:
        meta = "<ImageDocument />"

        def __init__(self, path: Path) -> None:
            self.path = path

        def get_dims_shape(self):
            return [{"X": (0, 16), "Y": (0, 16), "Z": (0, 2), "C": (0, 1), "T": (0, 1), "I": (0, 2), "S": (0, 1)}]

        def get_all_mosaic_tile_bounding_boxes(self, **kwargs):
            assert kwargs == {"C": 0, "Z": 0, "T": 0, "I": 0}
            raise RuntimeError("not a mosaic")

        def get_tile_bounding_box(self, **kwargs):
            assert kwargs == {"C": 0, "Z": 0, "T": 0, "I": 0}
            return Box()

        def read_image(self, **kwargs):
            assert kwargs["I"] in {0, 1}
            assert kwargs["Z"] in {0, 1}
            value = kwargs["I"] * 1000 + kwargs["Z"]
            plane = np.full((1, 1, 1, 1, 1, 16, 16), value, dtype=np.uint16)
            return (plane, None)

        def read_subblock_metadata(self, unified_xml: bool, **kwargs):
            assert not unified_xml
            assert kwargs["S"] == 0
            assert kwargs["I"] in {0, 1}
            return [
                ({"S": 0, "I": kwargs["I"], "C": 0, "T": 0, "Z": z}, "<Subblock />")
                for z in range(2)
            ]

    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"fake")
    monkeypatch.setattr("aicspylibczi.CziFile", FakeCziFile)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)

    assert not (tmp_path / "sample.ome.tif").exists()
    for illumination in range(2):
        out = tmp_path / "sample" / f"sample.i{illumination:03d}.ome.tif"
        assert out.exists()
        with TiffFile(out) as tif:
            annotations = _map_annotation_values(tif.ome_metadata)
            assert annotations["czi.illumination_index"] == str(illumination)
            raw_metadata = _raw_czi_metadata(tif.ome_metadata)
            subblocks = raw_metadata.find("SubblockMetadata")[0]
            assert len(subblocks) == 2
            assert {subblock.attrib["I"] for subblock in subblocks} == {str(illumination)}

    assert verify_czi_ome_tiff_outputs(czi_path, decode_samples=True)

    import zarr

    assert compress_czi_to_ome_tiff(
        czi_path,
        output_format="ome-zarr",
        level=90,
        maxworkers=1,
        thumbnails=False,
    )
    for illumination in range(2):
        out = tmp_path / "sample" / f"sample.i{illumination:03d}.ome.zarr"
        assert out.exists()
        root = zarr.open_group(str(out), mode="r")
        assert root.attrs["squisher"]["czi_illumination_index"] == illumination
        assert root.attrs["czi"]["shared_metadata_xml"] == "<ImageDocument />"
        zarr_subblocks = ET.fromstring(root.attrs["czi"]["subblock_metadata_xml"])
        assert len(zarr_subblocks) == 2
        assert {subblock.attrib["I"] for subblock in zarr_subblocks} == {str(illumination)}
        assert zarr.open(str(out / "0"), mode="r").shape == (1, 1, 2, 16, 16)


def test_compress_czi_skips_ome_zarr_by_name_without_completion_marker(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    czi_path = tmp_path / "sample.czi"
    data = np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape((2, 3, 16, 16))
    _write_sample_czi(czi_path, data)

    assert compress_czi_to_ome_tiff(czi_path, output_format="ome-zarr", level=90, maxworkers=1, thumbnails=False)
    root = zarr.open_group(str(tmp_path / "sample.ome.zarr"), mode="a")
    del root.attrs["squisher_complete"]

    assert compress_czi_to_ome_tiff(czi_path, output_format="ome-zarr", level=90, thumbnails=False)


def test_compress_czi_skips_ome_zarr_by_name_without_czi_metadata(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    czi_path = tmp_path / "sample.czi"
    data = np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape((2, 3, 16, 16))
    _write_sample_czi(czi_path, data)

    assert compress_czi_to_ome_tiff(czi_path, output_format="ome-zarr", level=90, maxworkers=1, thumbnails=False)
    root = zarr.open_group(str(tmp_path / "sample.ome.zarr"), mode="a")
    del root.attrs["czi"]

    assert compress_czi_to_ome_tiff(czi_path, output_format="ome-zarr", level=90, thumbnails=False)


def test_compress_czi_rejects_tiny_ome_zarr_chunks(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16))
    _write_sample_czi(czi_path, data)

    with pytest.raises(ValueError, match="smaller than --min-zarr-chunk-pixels"):
        compress_czi_to_ome_tiff(
            czi_path,
            output_format="ome-zarr",
            level=90,
            zarr_chunks=(1, 1, 1, 16, 16),
            min_zarr_chunk_pixels=1024,
            thumbnails=False,
        )


def test_compress_czi_rejects_jpegxr_multi_z_chunks(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(2 * 16 * 16, dtype=np.uint16).reshape((1, 2, 16, 16))
    _write_sample_czi(czi_path, data)

    with pytest.raises(ValueError, match="JPEG-XR OME-Zarr output requires z chunk size 1"):
        compress_czi_to_ome_tiff(
            czi_path,
            output_format="ome-zarr",
            level=90,
            zarr_chunks=(1, 1, 2, 1024, 1024),
            thumbnails=False,
        )


def test_compress_czi_writes_each_tile_and_placement(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    tiles = [
        np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 3, 4, 5)),
        np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 3, 4, 5)) + 1000,
    ]
    _write_multi_tile_czi(czi_path, tiles)
    czi_path.with_name("sample_placement.json").write_text(
        json.dumps(
            {
                "version": 1,
                "placement": {
                    "origins": [
                        {"index_zyx": [0, 0, 0], "origin_zyx": [0.0, 11.5, 22.5]},
                        {"index_zyx": [0, 1, 1], "origin_zyx": [0.0, 33.5, 44.5]},
                    ]
                },
            }
        )
    )

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, tile_workers=2)

    with TiffFile(tmp_path / "sample" / "sample.000.ome.tif") as tif:
        assert 'PositionX="22.5"' in tif.ome_metadata
        assert 'PositionY="11.5"' in tif.ome_metadata
        assert json.loads(_map_annotation_values(tif.ome_metadata)["squisher.placement_json"])["version"] == 1
        first_shared_metadata = ET.tostring(_shared_czi_metadata(tif.ome_metadata), encoding="unicode")
    with TiffFile(tmp_path / "sample" / "sample.001.ome.tif") as tif:
        assert 'PositionX="44.5"' in tif.ome_metadata
        assert 'PositionY="33.5"' in tif.ome_metadata
        assert ET.tostring(_shared_czi_metadata(tif.ome_metadata), encoding="unicode") == first_shared_metadata
    assert (tmp_path / "sample" / "sample.stitched-preview.png").exists()


def test_stitch_tile_preview_images_places_tiles_by_czi_coordinates() -> None:
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    first[..., 0] = 255
    second = np.zeros((2, 3, 3), dtype=np.uint8)
    second[..., 1] = 255

    stitched = _stitch_tile_preview_images(
        [
            (
                {
                    "index": 0,
                    "scene": 0,
                    "x": -10,
                    "y": 20,
                    "width": 30,
                    "height": 20,
                    "position_x": 0.0,
                    "position_y": 0.0,
                },
                first,
            ),
            (
                {
                    "index": 1,
                    "scene": 0,
                    "x": 20,
                    "y": 40,
                    "width": 30,
                    "height": 20,
                    "position_x": 30.0,
                    "position_y": 20.0,
                },
                second,
            ),
        ]
    )

    assert stitched.shape == (4, 6, 3)
    np.testing.assert_array_equal(stitched[:2, :3], first)
    np.testing.assert_array_equal(stitched[2:4, 3:6], second)
    assert np.all(stitched[:2, 3:6] == 0)


def test_stitched_preview_path_includes_illumination_suffix(tmp_path: Path) -> None:
    assert (
        _stitched_preview_path(
            tmp_path / "sample.czi",
            illumination=2,
            illumination_count=3,
            output_dir=tmp_path / "sample",
        )
        == tmp_path / "sample" / "sample.i002.stitched-preview.png"
    )


def test_infer_stage_positions_from_pos_anchors_mosaic_index_zero(tmp_path: Path) -> None:
    pos_path = tmp_path / "sample.pos"
    _write_pos_file(pos_path, [(70.0, 140.0, 9.0), (20.0, 140.0, 9.0), (20.0, 100.0, 9.0), (70.0, 100.0, 9.0)])
    tiles = [
        {
            "index": 0,
            "scene": 0,
            "mosaic_index": 0,
            "x": 0,
            "y": 0,
            "width": 5,
            "height": 4,
            "position_x": 0.0,
            "position_y": 0.0,
        },
        {
            "index": 1,
            "scene": 0,
            "mosaic_index": 1,
            "x": 10,
            "y": 20,
            "width": 5,
            "height": 4,
            "position_x": 10.0,
            "position_y": 20.0,
        },
    ]

    inferred = infer_stage_positions_from_pos(pos_path, tiles)

    assert inferred[0]["stage_position_x_um"] == pytest.approx(45.0)
    assert inferred[0]["stage_position_y_um"] == pytest.approx(120.0)
    assert inferred[0]["stage_position_z_um"] == pytest.approx(9.0)
    assert inferred[1]["stage_position_x_um"] == pytest.approx(145.0)
    assert inferred[1]["stage_position_y_um"] == pytest.approx(320.0)
    assert inferred[1]["stage_position_scale_x_um_per_px"] == pytest.approx(10.0)
    assert inferred[1]["stage_position_scale_y_um_per_px"] == pytest.approx(10.0)


def test_infer_stage_positions_from_pos_requires_mosaic_index_zero(tmp_path: Path) -> None:
    pos_path = tmp_path / "sample.pos"
    _write_pos_file(pos_path, [(70.0, 140.0, 9.0), (20.0, 140.0, 9.0), (20.0, 100.0, 9.0), (70.0, 100.0, 9.0)])

    with pytest.raises(ValueError, match="do not contain M=0"):
        infer_stage_positions_from_pos(
            pos_path,
            [
                {
                    "index": 0,
                    "scene": 0,
                    "mosaic_index": 1,
                    "x": 0,
                    "y": 0,
                    "width": 5,
                    "height": 4,
                    "position_x": 0.0,
                    "position_y": 0.0,
                }
            ],
        )


def test_infer_stage_positions_from_pos_requires_first_hull(tmp_path: Path) -> None:
    pos_path = tmp_path / "sample.pos"
    _write_pos_file(pos_path, [(70.0, 140.0, 9.0), (20.0, 140.0, 9.0), (20.0, 100.0, 9.0)])

    with pytest.raises(ValueError, match="Expected at least four positions"):
        infer_stage_positions_from_pos(pos_path, [])


def test_compress_czi_writes_pos_inferred_stage_positions(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    pos_path = tmp_path / "sample.pos"
    tiles = [
        np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 3, 4, 5)),
        np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 3, 4, 5)) + 1000,
    ]
    _write_multi_tile_czi(czi_path, tiles)
    _write_pos_file(pos_path, [(70.0, 140.0, 9.0), (20.0, 140.0, 9.0), (20.0, 100.0, 9.0), (70.0, 100.0, 9.0)])

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, pos_path=pos_path)

    with TiffFile(tmp_path / "sample" / "sample.000.ome.tif") as tif:
        plane = _first_plane_attrs(tif.ome_metadata)
        annotations = _map_annotation_values(tif.ome_metadata)
        assert plane["PositionX"] == "45.0"
        assert plane["PositionY"] == "120.0"
        assert plane["PositionZ"] == "9.0"
        assert plane["PositionXUnit"] == "µm"
        assert plane["PositionYUnit"] == "µm"
        assert plane["PositionZUnit"] == "µm"
        assert annotations["squisher.stage_position_x_um"] == "45.0"
        assert annotations["squisher.stage_position_y_um"] == "120.0"
        assert annotations["squisher.stage_position_z_um"] == "9.0"
        assert annotations["squisher.stage_position_source"] == "zeiss-pos"
        assert annotations["squisher.stage_position_pos_file"] == str(pos_path)
        assert annotations["squisher.stage_position_anchor_mosaic_index"] == "0"
        assert annotations["squisher.stage_position_anchor_position_group"] == "1"
        assert annotations["squisher.stage_position_scale_x_um_per_px"] == "10.0"
        assert annotations["squisher.stage_position_scale_y_um_per_px"] == "10.0"
    with TiffFile(tmp_path / "sample" / "sample.001.ome.tif") as tif:
        plane = _first_plane_attrs(tif.ome_metadata)
        assert plane["PositionX"] == "145.0"
        assert plane["PositionY"] == "320.0"
        assert plane["PositionZ"] == "9.0"


def test_compress_czi_skips_existing_output_by_name(tmp_path: Path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16))
    _write_sample_czi(czi_path, data)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)

    def fail_write(*args, **kwargs):
        raise AssertionError("existing output name should be skipped")

    monkeypatch.setattr("squisher.compression._write_czi_tile", fail_write)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)


def test_compress_czi_skips_when_all_tile_output_names_exist(tmp_path: Path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    tiles = [
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)),
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)) + 1000,
    ]
    _write_multi_tile_czi(czi_path, tiles)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)

    def fail_write(*args, **kwargs):
        raise AssertionError("existing tile output names should be skipped")

    monkeypatch.setattr("squisher.compression._write_czi_tile", fail_write)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)


def test_compress_czi_refuses_partial_existing_tile_outputs(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    tiles = [
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)),
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)) + 1000,
    ]
    _write_multi_tile_czi(czi_path, tiles)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)
    (tmp_path / "sample" / "sample.001.ome.tif").unlink()

    with pytest.raises(FileExistsError, match="partial existing ome-tiff output set"):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)


def test_compress_czi_skips_malformed_existing_output_by_name(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(4 * 5, dtype=np.uint16).reshape((1, 1, 4, 5))
    _write_sample_czi(czi_path, data)
    (tmp_path / "sample.ome.tif").write_bytes(b"existing")

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)


def test_compress_czi_overwrites_existing_output_when_requested(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16))
    _write_sample_czi(czi_path, data)
    (tmp_path / "sample.ome.tif").write_bytes(b"existing")
    (tmp_path / "sample.center-z.png").write_bytes(b"old thumbnail")

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, overwrite=True)

    with TiffFile(tmp_path / "sample.ome.tif") as tif:
        assert tif.pages[0].compression == 22610
        annotations = _map_annotation_values(tif.ome_metadata)
        assert annotations["squisher.overwrite"] == "true"
    assert (tmp_path / "sample.center-z.png").read_bytes() != b"old thumbnail"


def test_compress_czi_keeps_existing_output_when_overwrite_fails(tmp_path: Path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16))
    _write_sample_czi(czi_path, data)
    out = tmp_path / "sample.ome.tif"
    out.write_bytes(b"existing")

    def fail_grayscale_plane(data):
        raise RuntimeError("write failed")

    monkeypatch.setattr("squisher.compression._as_grayscale_plane", fail_grayscale_plane)

    with pytest.raises(RuntimeError, match="write failed"):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, overwrite=True)

    assert out.read_bytes() == b"existing"


def test_verify_czi_decode_samples_can_enforce_source_diff_thresholds(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape((2, 3, 16, 16))
    _write_sample_czi(czi_path, data)

    assert compress_czi_to_ome_tiff(czi_path, level=65, tile_size=16, maxworkers=1, thumbnails=False)
    assert verify_czi_ome_tiff_outputs(czi_path, decode_samples=True)

    with pytest.raises(ValueError, match="MAE"):
        verify_czi_ome_tiff_outputs(czi_path, decode_samples=True, max_sample_mae=0.0)


def test_compress_czi_rejects_multi_scene_inputs(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    with czi.create_czi(str(czi_path)) as writer:
        assert writer.write(np.zeros((16, 16), dtype=np.uint16), plane={"C": 0, "Z": 0}, scene=0)
        assert writer.write(np.ones((16, 16), dtype=np.uint16), plane={"C": 0, "Z": 0}, scene=1)

    with pytest.raises(ValueError, match="Unsupported CZI dimensions.*S=0:2"):
        compress_czi_to_ome_tiff(czi_path, level=65, tile_size=16, maxworkers=1, thumbnails=False)


def test_verify_czi_detects_missing_tile_outputs(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    tiles = [
        np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 3, 4, 5)),
        np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape((2, 3, 4, 5)) + 1000,
    ]
    _write_multi_tile_czi(czi_path, tiles)

    with pytest.raises(ValueError, match="sample.000.ome.tif: missing"):
        verify_czi_ome_tiff_outputs(czi_path)


def test_verify_ome_tiff_checks_all_pages(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    out = tmp_path / "sample.ome.tif"
    with TiffWriter(out, bigtiff=True) as writer:
        writer.write(
            np.zeros((16, 16), dtype=np.uint16),
            tile=(16, 16),
            photometric="minisblack",
            compression=22610,
            compressionargs={"level": 0.9},
            metadata=None,
        )
        writer.write(np.zeros((16, 16), dtype=np.uint16), metadata=None)

    errors = _verify_ome_tiff(
        czi_path,
        out,
        tile={"index": 0, "scene": 0, "x": 0, "y": 0, "width": 16, "height": 16, "position_x": 0.0, "position_y": 0.0},
        plane_count=2,
    )

    assert any("page 1" in error and "compression" in error for error in errors)


def test_verify_ome_tiff_requires_declared_pyramid_subifds(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    out = tmp_path / "sample.ome.tif"
    with TiffWriter(out, bigtiff=True) as writer:
        writer.write(
            np.zeros((512, 512), dtype=np.uint16),
            description=_minimal_ome_xml_bytes(size_x=512, size_y=512, tiff_pyramid=True),
            tile=(256, 256),
            photometric="minisblack",
            compression=22610,
            compressionargs={"level": 0.9},
            metadata=None,
        )

    errors = _verify_ome_tiff(
        czi_path,
        out,
        tile={"index": 0, "scene": 0, "x": 0, "y": 0, "width": 512, "height": 512, "position_x": 0.0, "position_y": 0.0},
        plane_count=1,
    )

    assert any("page 0 expected 1 SubIFDs, found 0" in error for error in errors)


def test_compress_czi_uses_out_dir_as_parent_for_multi_tile_outputs(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    out_dir = tmp_path / "compressed"
    tiles = [
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)),
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)) + 1000,
    ]
    _write_multi_tile_czi(czi_path, tiles)

    assert compress_czi_to_ome_tiff(
        czi_path,
        level=90,
        out_dir=out_dir,
        tile_size=16,
        maxworkers=1,
        thumbnails=False,
    )

    assert (out_dir / "sample" / "sample.000.ome.tif").exists()
    assert (out_dir / "sample" / "sample.001.ome.tif").exists()
    assert verify_czi_ome_tiff_outputs(czi_path, out_dir=out_dir)


def test_compress_czi_promotes_multi_ome_tiff_temp_output_dir_after_success(
    tmp_path: Path, monkeypatch
) -> None:
    czi_path = tmp_path / "sample.czi"
    tiles = [
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)),
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)) + 1000,
    ]
    _write_multi_tile_czi(czi_path, tiles)
    final_dir = tmp_path / "sample"
    temp_dir = tmp_path / "sample-temp"

    def fake_write_czi_tile(_reader, _path, *, output_dir, tile, **_kwargs):
        assert output_dir == temp_dir
        assert not final_dir.exists()
        out = output_dir / f"sample.{tile['index']:03d}.ome.tif"
        out.write_bytes(b"complete")
        return out

    monkeypatch.setattr("squisher.compression._write_czi_tile", fake_write_czi_tile)

    assert compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)

    assert not temp_dir.exists()
    assert (final_dir / "sample.000.ome.tif").read_bytes() == b"complete"
    assert (final_dir / "sample.001.ome.tif").read_bytes() == b"complete"


def test_compress_czi_keeps_failed_multi_ome_tiff_output_dir_as_temp(
    tmp_path: Path, monkeypatch
) -> None:
    czi_path = tmp_path / "sample.czi"
    tiles = [
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)),
        np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16)) + 1000,
    ]
    _write_multi_tile_czi(czi_path, tiles)
    final_dir = tmp_path / "sample"
    temp_dir = tmp_path / "sample-temp"

    def fail_write_czi_tile(_reader, _path, *, output_dir, tile, **_kwargs):
        assert output_dir == temp_dir
        (output_dir / f"sample.{tile['index']:03d}.ome.tif").write_bytes(b"partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr("squisher.compression._write_czi_tile", fail_write_czi_tile)

    with pytest.raises(RuntimeError, match="write failed"):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1, thumbnails=False)

    assert not final_dir.exists()
    assert (temp_dir / "sample.000.ome.tif").read_bytes() == b"partial"


def test_czi_output_dir_omits_stem_folder_for_single_tile_outputs(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"

    assert _czi_output_dir(czi_path, out_dir=None, output_count=1) == tmp_path
    assert _czi_output_dir(czi_path, out_dir=tmp_path / "compressed", output_count=1) == tmp_path / "compressed"
    assert _czi_output_dir(czi_path, out_dir=None, output_count=2) == tmp_path / "sample"
    assert _czi_output_dir(czi_path, out_dir=tmp_path / "compressed", output_count=2) == (
        tmp_path / "compressed" / "sample"
    )


def test_czi_progress_steps_count_ceil_plane_blocks() -> None:
    assert _czi_progress_steps(1) == 1
    assert _czi_progress_steps(CZI_PROGRESS_INTERVAL) == 1
    assert _czi_progress_steps(CZI_PROGRESS_INTERVAL + 1) == 2
    assert _czi_progress_steps(CZI_PROGRESS_INTERVAL * 2) == 2


def test_report_czi_plane_progress_advances_every_interval_and_final_plane() -> None:
    advances = []
    plane_count = CZI_PROGRESS_INTERVAL * 2 + 25

    for plane_index in range(1, plane_count + 1):
        _report_czi_plane_progress(
            Path("sample.ome.tif"),
            plane_index=plane_index,
            plane_count=plane_count,
            t=0,
            c=0,
            z=plane_index - 1,
            progress_callback=lambda steps: advances.append(steps),
        )

    assert advances == [1, 1, 1]


def test_report_czi_plane_progress_can_advance_without_logging(monkeypatch) -> None:
    advances = []
    messages = []

    monkeypatch.setattr("squisher.compression.logger.info", lambda message: messages.append(message))

    _report_czi_plane_progress(
        Path("sample.ome.tif"),
        plane_index=1,
        plane_count=1,
        t=0,
        c=0,
        z=0,
        progress_callback=lambda steps: advances.append(steps),
        log_progress=False,
    )

    assert advances == [1]
    assert messages == []


def test_progress_bar_uses_shared_console(monkeypatch) -> None:
    shared_console = object()
    captured = {}

    class FakeProgress:
        def __init__(self, *_columns, console):
            captured["console"] = console
            self.advances = []

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def add_task(self, description, *, total):
            captured["description"] = description
            captured["total"] = total
            return 7

        def update(self, task_id, *, advance):
            captured.setdefault("updates", []).append((task_id, advance))

    monkeypatch.setattr("squisher.compression.get_shared_console", lambda: shared_console)
    monkeypatch.setattr("squisher.compression.Progress", FakeProgress)

    with _progress_bar(5) as advance:
        advance(2)

    assert captured["console"] is shared_console
    assert captured["description"] == "Compressing"
    assert captured["total"] == 5
    assert captured["updates"] == [(7, 2)]


def test_compress_czi_progress_total_counts_pending_tile_outputs(tmp_path: Path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"fake")
    totals = []

    class FakeCziFile:
        def __init__(self, path: Path) -> None:
            self.path = path

        def get_dims_shape(self):
            return [{"X": (0, 16), "Y": (0, 16), "Z": (0, CZI_PROGRESS_INTERVAL + 1), "C": (0, 1), "T": (0, 1)}]

    tiles = [
        {"index": 0, "scene": 0, "x": 0, "y": 0, "width": 16, "height": 16, "position_x": 0.0, "position_y": 0.0},
        {"index": 1, "scene": 0, "x": 16, "y": 0, "width": 16, "height": 16, "position_x": 16.0, "position_y": 0.0},
    ]

    @contextmanager
    def capture_progress(total: int):
        totals.append(total)
        yield lambda steps: None

    def fake_write_czi_tile(_reader, _path, *, output_dir, tile, **_kwargs):
        out = output_dir / f"sample.{tile['index']:03d}.ome.tif"
        out.write_bytes(b"complete")
        return out

    monkeypatch.setattr("aicspylibczi.CziFile", FakeCziFile)
    monkeypatch.setattr("squisher.compression._czi_tiles", lambda path, *, pos_path=None: tiles)
    monkeypatch.setattr("squisher.compression._progress_bar", capture_progress)
    monkeypatch.setattr("squisher.compression._write_czi_tile", fake_write_czi_tile)

    assert compress_czi_to_ome_tiff(czi_path, level=90, maxworkers=1, thumbnails=False)

    assert totals == [_czi_progress_steps(CZI_PROGRESS_INTERVAL + 1) * len(tiles)]


def test_write_czi_tile_reports_tiff_progress_from_real_writer_loop(tmp_path: Path, monkeypatch) -> None:
    advances = []

    class FakeTiffWriter:
        def __init__(self, path: Path, **_kwargs) -> None:
            self.path = path

        def __enter__(self):
            self.path.write_bytes(b"tiff")
            return self

        def __exit__(self, *_args) -> None:
            return None

        def write(self, *_args, **_kwargs) -> None:
            return None

    class FakeReader:
        meta = "<ImageDocument />"

        def read_image(self, **_kwargs):
            return (np.zeros((1, 1, 1, 1, 1, 16, 16), dtype=np.uint16), None)

        def read_subblock_metadata(self, unified_xml: bool, **_kwargs):
            assert not unified_xml
            return [({}, "<Subblock />")]

    monkeypatch.setattr("squisher.compression.TiffWriter", FakeTiffWriter)
    monkeypatch.setattr("squisher.compression._first_plane_ome_xml", lambda *args, **kwargs: "<OME />")

    out = _write_czi_tile(
        FakeReader(),
        tmp_path / "sample.czi",
        tile={"index": 0, "scene": 0, "x": 0, "y": 0, "width": 16, "height": 16, "position_x": 0.0, "position_y": 0.0},
        tile_count=1,
        illumination=0,
        illumination_count=1,
        output_dir=tmp_path,
        output_format="ome-tiff",
        level=90,
        tile_size=16,
        zarr_chunks=(1, 1, 1, 16, 16),
        zarr_compressor="jpegxr",
        maxworkers=1,
        dims={"X": (0, 16), "Y": (0, 16), "Z": (0, CZI_PROGRESS_INTERVAL + 1), "C": (0, 1), "T": (0, 1)},
        provenance={},
        progress_callback=lambda steps: advances.append(steps),
    )

    assert out == tmp_path / "sample.ome.tif"
    assert advances == [1, 1]


def test_write_czi_tile_adds_pyramid_subifds_by_default(tmp_path: Path, monkeypatch) -> None:
    class FakeReader:
        meta = "<ImageDocument />"

        def read_image(self, **_kwargs):
            data = np.arange(512 * 512, dtype=np.uint16).reshape(1, 1, 1, 1, 1, 512, 512)
            return (data, None)

        def read_subblock_metadata(self, unified_xml: bool, **_kwargs):
            assert not unified_xml
            return [({}, "<Subblock />")]

    monkeypatch.setattr(
        "squisher.compression._first_plane_ome_xml",
        lambda *args, **kwargs: _minimal_ome_xml_bytes(size_x=512, size_y=512),
    )

    out = _write_czi_tile(
        FakeReader(),
        tmp_path / "sample.czi",
        tile={
            "index": 0,
            "scene": 0,
            "x": 0,
            "y": 0,
            "width": 512,
            "height": 512,
            "position_x": 0.0,
            "position_y": 0.0,
        },
        tile_count=1,
        illumination=0,
        illumination_count=1,
        output_dir=tmp_path,
        output_format="ome-tiff",
        level=90,
        tile_size=256,
        zarr_chunks=(1, 1, 1, 512, 512),
        zarr_compressor="jpegxr",
        maxworkers=1,
        dims={"X": (0, 512), "Y": (0, 512), "Z": (0, 1), "C": (0, 1), "T": (0, 1)},
        provenance={},
        progress_callback=lambda _steps: None,
    )

    with TiffFile(out) as tif:
        assert tif.is_ome
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == 22610
        assert page.shape == (512, 512)
        assert len(page.subifds) == 1
        subifd = page.pages[0]
        assert subifd.is_subifd
        assert subifd.compression == 22610
        assert subifd.shape == (256, 256)


def test_write_czi_tile_can_disable_pyramid_subifds(tmp_path: Path, monkeypatch) -> None:
    class FakeReader:
        meta = "<ImageDocument />"

        def read_image(self, **_kwargs):
            return (np.zeros((1, 1, 1, 1, 1, 512, 512), dtype=np.uint16), None)

        def read_subblock_metadata(self, unified_xml: bool, **_kwargs):
            assert not unified_xml
            return [({}, "<Subblock />")]

    monkeypatch.setattr(
        "squisher.compression._first_plane_ome_xml",
        lambda *args, **kwargs: _minimal_ome_xml_bytes(size_x=512, size_y=512),
    )

    out = _write_czi_tile(
        FakeReader(),
        tmp_path / "sample.czi",
        tile={
            "index": 0,
            "scene": 0,
            "x": 0,
            "y": 0,
            "width": 512,
            "height": 512,
            "position_x": 0.0,
            "position_y": 0.0,
        },
        tile_count=1,
        illumination=0,
        illumination_count=1,
        output_dir=tmp_path,
        output_format="ome-tiff",
        level=90,
        tile_size=256,
        zarr_chunks=(1, 1, 1, 512, 512),
        zarr_compressor="jpegxr",
        maxworkers=1,
        dims={"X": (0, 512), "Y": (0, 512), "Z": (0, 1), "C": (0, 1), "T": (0, 1)},
        provenance={},
        progress_callback=lambda _steps: None,
        pyramid=False,
    )

    with TiffFile(out) as tif:
        assert len(tif.pages) == 1
        assert not tif.pages[0].subifds


def test_write_czi_tile_reports_zarr_progress_from_real_writer_loop(tmp_path: Path, monkeypatch) -> None:
    zarr = pytest.importorskip("zarr")
    advances = []

    class FakeArray:
        attrs = {}

        def __setitem__(self, _key, _value) -> None:
            return None

    class FakeRoot:
        def __init__(self, path: str) -> None:
            Path(path).mkdir()
            self.attrs = {}

        def create_array(self, *_args, **_kwargs):
            return FakeArray()

    class FakeReader:
        meta = "<ImageDocument />"

        def read_image(self, **_kwargs):
            return (np.zeros((1, 1, 1, 1, 1, 16, 16), dtype=np.uint16), None)

        def read_subblock_metadata(self, unified_xml: bool, **_kwargs):
            assert not unified_xml
            return [({}, "<Subblock />")]

    monkeypatch.setattr(zarr, "open_group", lambda path, **_kwargs: FakeRoot(path))
    monkeypatch.setattr("squisher.compression._ome_zarr_root_attrs", lambda *args, **kwargs: {})
    monkeypatch.setattr("squisher.compression._zarr_numcodecs_compressor", lambda *args, **kwargs: None)

    out = _write_czi_tile(
        FakeReader(),
        tmp_path / "sample.czi",
        tile={"index": 0, "scene": 0, "x": 0, "y": 0, "width": 16, "height": 16, "position_x": 0.0, "position_y": 0.0},
        tile_count=1,
        illumination=0,
        illumination_count=1,
        output_dir=tmp_path,
        output_format="ome-zarr",
        level=90,
        tile_size=16,
        zarr_chunks=(1, 1, 1, 16, 16),
        zarr_compressor="jpegxr",
        maxworkers=1,
        dims={"X": (0, 16), "Y": (0, 16), "Z": (0, CZI_PROGRESS_INTERVAL + 1), "C": (0, 1), "T": (0, 1)},
        provenance={},
        progress_callback=lambda steps: advances.append(steps),
    )

    assert out == tmp_path / "sample.ome.zarr"
    assert advances == [1, 1]


def test_write_czi_tile_process_forwards_real_tile_progress_and_logs(tmp_path: Path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"fake")
    queued_events = []

    class FakeCziFile:
        meta = "<ImageDocument />"

        def __init__(self, path: Path) -> None:
            self.path = path

        def get_dims_shape(self):
            return [{"X": (0, 16), "Y": (0, 16), "Z": (0, 1), "C": (0, 1), "T": (0, 1)}]

        def read_image(self, **_kwargs):
            return (np.zeros((1, 1, 1, 1, 1, 16, 16), dtype=np.uint16), None)

    class FakeTiffWriter:
        def __init__(self, path: Path, **_kwargs) -> None:
            self.path = path

        def __enter__(self):
            self.path.write_bytes(b"tiff")
            return self

        def __exit__(self, *_args) -> None:
            return None

        def write(self, *_args, **_kwargs) -> None:
            return None

    class FakeQueue:
        def put(self, value: object) -> None:
            queued_events.append(value)

    monkeypatch.setattr("aicspylibczi.CziFile", FakeCziFile)
    monkeypatch.setattr("squisher.compression.TiffWriter", FakeTiffWriter)
    monkeypatch.setattr("squisher.compression._first_plane_ome_xml", lambda *args, **kwargs: "<OME />")

    assert _write_czi_tile_process(
        czi_path,
        tile={"index": 0, "scene": 0, "x": 0, "y": 0, "width": 16, "height": 16, "position_x": 0.0, "position_y": 0.0},
        tile_count=1,
        illumination=0,
        illumination_count=1,
        output_dir=tmp_path,
        output_format="ome-tiff",
        level=90,
        tile_size=16,
        zarr_chunks=(1, 1, 1, 16, 16),
        zarr_compressor="jpegxr",
        maxworkers=1,
        provenance={},
        progress_queue=FakeQueue(),
    ) == tmp_path / "sample.ome.tif"
    assert ("progress", 1) in queued_events
    log_events = [payload for kind, payload in queued_events if kind == "log"]
    plane_log = next(event for event in log_events if "sample.ome.tif: wrote plane 1/1" in event)
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \| INFO\s+\| "
        r"squisher\.compression:\d+ - sample\.ome\.tif: wrote plane 1/1 \(T=0, C=0, Z=0\)\n",
        plane_log,
    )
    assert any("Writing tile 1/1" in event for event in log_events)
    assert any("Finished sample.ome.tif" in event for event in log_events)


def test_progress_queue_listener_prints_log_events_and_advances(monkeypatch) -> None:
    printed = []
    advances = []

    class FakeConsole:
        def print(self, message, *, end) -> None:
            printed.append((message, end))

    class FakeQueue:
        def __init__(self) -> None:
            self.events = [
                ("log", "2026-06-13 12:00:00.000 | INFO     | worker:1 - update\n"),
                ("progress", 2),
                ("progress", 1, "extra"),
                "legacy-progress",
                ("other", 1),
                None,
            ]

        def get(self):
            return self.events.pop(0)

    monkeypatch.setattr("squisher.compression.get_shared_console", lambda: FakeConsole())

    thread = _start_progress_queue_listener(FakeQueue(), lambda steps: advances.append(steps))
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert printed == [
        ("2026-06-13 12:00:00.000 | INFO     | worker:1 - update\n", ""),
        ("Unexpected progress queue event ('progress', 1, 'extra')", "\n"),
        ("Unexpected progress queue event 'legacy-progress'", "\n"),
        ("Unexpected progress queue event 'other'", "\n"),
    ]
    assert advances == [2]


def test_compress_czi_process_pool_failure_stops_progress_listener_and_keeps_temp_dir(
    tmp_path: Path, monkeypatch
) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"fake")
    temp_dir = tmp_path / "sample-temp"

    class FakeCziFile:
        def __init__(self, path: Path) -> None:
            self.path = path

        def get_dims_shape(self):
            return [{"X": (0, 16), "Y": (0, 16), "Z": (0, 1), "C": (0, 1), "T": (0, 1)}]

    class FailingPool:
        _processes = {}

        def __init__(self, *_args, **_kwargs) -> None:
            self.shutdown_calls = []

        def submit(self, *_args, **_kwargs):
            future = Future()
            future.set_exception(RuntimeError("worker failed"))
            return future

        def shutdown(self, *args, **kwargs) -> None:
            self.shutdown_calls.append((args, kwargs))

    @contextmanager
    def capture_progress(_total: int):
        yield lambda steps: None

    tiles = [
        {"index": 0, "scene": 0, "x": 0, "y": 0, "width": 16, "height": 16, "position_x": 0.0, "position_y": 0.0},
        {"index": 1, "scene": 0, "x": 16, "y": 0, "width": 16, "height": 16, "position_x": 16.0, "position_y": 0.0},
    ]

    monkeypatch.setattr("aicspylibczi.CziFile", FakeCziFile)
    monkeypatch.setattr("squisher.compression._czi_tiles", lambda path, *, pos_path=None: tiles)
    monkeypatch.setattr("squisher.compression._progress_bar", capture_progress)
    monkeypatch.setattr("squisher.compression.ProcessPoolExecutor", FailingPool)

    with pytest.raises(RuntimeError, match="worker failed"):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_workers=2, maxworkers=1, thumbnails=False)

    assert temp_dir.exists()
    assert not (tmp_path / "sample").exists()


def test_compress_czi_process_pool_keyboard_interrupt_terminates_workers(
    tmp_path: Path, monkeypatch
) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"fake")
    pools = []

    class FakeCziFile:
        def __init__(self, path: Path) -> None:
            self.path = path

        def get_dims_shape(self):
            return [{"X": (0, 16), "Y": (0, 16), "Z": (0, 1), "C": (0, 1), "T": (0, 1)}]

    class InterruptingPool:
        def __init__(self, *_args, **_kwargs) -> None:
            self.terminated = False
            pools.append(self)

        def submit(self, *_args, **_kwargs):
            future = Future()
            future.set_exception(KeyboardInterrupt())
            return future

        def terminate_workers(self) -> None:
            self.terminated = True

    class FakeQueue:
        def __init__(self) -> None:
            self.events = []

        def put(self, value) -> None:
            self.events.append(value)

    class FakeManager:
        def __init__(self) -> None:
            self.queue = FakeQueue()
            self.shutdown_called = False

        def Queue(self):
            return self.queue

        def shutdown(self) -> None:
            self.shutdown_called = True

    class FakeContext:
        def __init__(self) -> None:
            self.manager = FakeManager()

        def Manager(self):
            return self.manager

    class FakeThread:
        def __init__(self) -> None:
            self.joined = False

        def join(self) -> None:
            self.joined = True

    @contextmanager
    def capture_progress(_total: int):
        yield lambda steps: None

    fake_context = FakeContext()
    manager = fake_context.manager
    listener_thread = FakeThread()
    tiles = [
        {"index": 0, "scene": 0, "x": 0, "y": 0, "width": 16, "height": 16, "position_x": 0.0, "position_y": 0.0},
        {"index": 1, "scene": 0, "x": 16, "y": 0, "width": 16, "height": 16, "position_x": 16.0, "position_y": 0.0},
    ]

    monkeypatch.setattr("aicspylibczi.CziFile", FakeCziFile)
    monkeypatch.setattr("squisher.compression._czi_tiles", lambda path, *, pos_path=None: tiles)
    monkeypatch.setattr("squisher.compression._progress_bar", capture_progress)
    monkeypatch.setattr("squisher.compression.get_context", lambda _method: fake_context)
    monkeypatch.setattr("squisher.compression._start_progress_queue_listener", lambda queue, callback: listener_thread)
    monkeypatch.setattr("squisher.compression.ProcessPoolExecutor", InterruptingPool)

    with pytest.raises(KeyboardInterrupt):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_workers=2, maxworkers=1, thumbnails=False)

    assert pools[0].terminated
    assert manager.queue.events[-1] is None
    assert listener_thread.joined
    assert manager.shutdown_called


def test_compress_czi_process_pool_keyboard_interrupt_during_shutdown_terminates_workers(
    tmp_path: Path, monkeypatch
) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"fake")
    pools = []

    class FakeCziFile:
        def __init__(self, path: Path) -> None:
            self.path = path

        def get_dims_shape(self):
            return [{"X": (0, 16), "Y": (0, 16), "Z": (0, 1), "C": (0, 1), "T": (0, 1)}]

    class InterruptingShutdownPool:
        def __init__(self, *_args, **_kwargs) -> None:
            self.terminated = False
            pools.append(self)

        def submit(self, *_args, **_kwargs):
            future = Future()
            future.set_result(tmp_path / "sample.000.ome.tif")
            return future

        def shutdown(self, *, wait: bool) -> None:
            assert wait is True
            raise KeyboardInterrupt

        def terminate_workers(self) -> None:
            self.terminated = True

    class FakeQueue:
        def put(self, _value) -> None:
            return None

    class FakeManager:
        def Queue(self):
            return FakeQueue()

        def shutdown(self) -> None:
            return None

    class FakeContext:
        def Manager(self):
            return FakeManager()

    class FakeThread:
        def join(self) -> None:
            return None

    @contextmanager
    def capture_progress(_total: int):
        yield lambda steps: None

    tiles = [
        {"index": 0, "scene": 0, "x": 0, "y": 0, "width": 16, "height": 16, "position_x": 0.0, "position_y": 0.0},
        {"index": 1, "scene": 0, "x": 16, "y": 0, "width": 16, "height": 16, "position_x": 16.0, "position_y": 0.0},
    ]

    monkeypatch.setattr("aicspylibczi.CziFile", FakeCziFile)
    monkeypatch.setattr("squisher.compression._czi_tiles", lambda path, *, pos_path=None: tiles)
    monkeypatch.setattr("squisher.compression._progress_bar", capture_progress)
    monkeypatch.setattr("squisher.compression.get_context", lambda _method: FakeContext())
    monkeypatch.setattr("squisher.compression._start_progress_queue_listener", lambda queue, callback: FakeThread())
    monkeypatch.setattr("squisher.compression.ProcessPoolExecutor", InterruptingShutdownPool)

    with pytest.raises(KeyboardInterrupt):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_workers=2, maxworkers=1, thumbnails=False)

    assert pools[0].terminated


def test_empty_subblock_metadata_is_preserved_as_empty_marker() -> None:
    class Reader:
        def read_subblock_metadata(self, unified_xml: bool, **kwargs):
            return [({"M": 3, "C": 0, "Z": 0}, "")]

    metadata = _czi_subblock_metadata(Reader(), {"index": 0, "scene": 0, "mosaic_index": 3})
    subblock = metadata.find("Subblock")

    assert subblock is not None
    assert subblock.attrib["M"] == "3"
    assert subblock.attrib["MetadataEmpty"] == "true"


def _map_annotation_values(ome_metadata: str) -> dict[str, str]:
    root = ET.fromstring(ome_metadata)
    return {
        item.attrib["K"]: item.text or ""
        for item in root.findall(".//ome:MapAnnotation/ome:Value/ome:M", OME_NS)
    }


def _first_plane_attrs(ome_metadata: str) -> dict[str, str]:
    root = ET.fromstring(ome_metadata)
    plane = root.find(".//ome:Plane", OME_NS)
    assert plane is not None
    return plane.attrib


def _raw_czi_metadata(ome_metadata: str) -> ET.Element:
    root = ET.fromstring(ome_metadata)
    raw_annotations = [
        annotation
        for annotation in root.findall(".//ome:XMLAnnotation", OME_NS)
        if annotation.attrib.get("Namespace") == "squisher/czi/raw-metadata"
    ]
    assert len(raw_annotations) == 1
    assert root.find(f".//ome:AnnotationRef[@ID='{raw_annotations[0].attrib['ID']}']", OME_NS) is not None
    value = raw_annotations[0].find("ome:Value", OME_NS)
    assert value is not None
    assert len(value) == 1
    assert value[0].tag == "CZITileMetadata"
    return value[0]


def _shared_czi_metadata(ome_metadata: str) -> ET.Element:
    root = ET.fromstring(ome_metadata)
    shared_annotations = [
        annotation
        for annotation in root.findall(".//ome:XMLAnnotation", OME_NS)
        if annotation.attrib.get("Namespace") == "squisher/czi/shared-metadata"
    ]
    assert len(shared_annotations) == 1
    assert root.find(f".//ome:AnnotationRef[@ID='{shared_annotations[0].attrib['ID']}']", OME_NS) is not None
    value = shared_annotations[0].find("ome:Value", OME_NS)
    assert value is not None
    assert len(value) == 1
    assert value[0].tag == "CZISharedMetadata"
    assert len(value[0]) == 1
    return value[0][0]
