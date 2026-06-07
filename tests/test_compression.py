import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pylibCZIrw import czi
from tifffile import TiffFile, TiffWriter

from squisher.compression import (
    _czi_output_dir,
    _czi_subblock_metadata,
    _verify_ome_tiff,
    compress_czi_to_ome_tiff,
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
        assert annotations["czi.global_metadata_xml"].startswith("<ImageDocument")
        assert annotations["czi.subblock_metadata_xml"].startswith("<Subblocks")
        raw_metadata = _raw_czi_metadata(tif.ome_metadata)
        assert raw_metadata.find("GlobalMetadata")[0].tag == "ImageDocument"
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


def test_resume_rejects_ome_zarr_without_completion_marker(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    czi_path = tmp_path / "sample.czi"
    data = np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape((2, 3, 16, 16))
    _write_sample_czi(czi_path, data)

    assert compress_czi_to_ome_tiff(czi_path, output_format="ome-zarr", level=90, maxworkers=1, thumbnails=False)
    root = zarr.open_group(str(tmp_path / "sample.ome.zarr"), mode="a")
    del root.attrs["squisher_complete"]

    with pytest.raises(FileExistsError, match="incomplete ome-zarr"):
        compress_czi_to_ome_tiff(czi_path, output_format="ome-zarr", level=90, resume=True, thumbnails=False)


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
    with TiffFile(tmp_path / "sample" / "sample.001.ome.tif") as tif:
        assert 'PositionX="44.5"' in tif.ome_metadata
        assert 'PositionY="33.5"' in tif.ome_metadata


def test_compress_czi_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(4 * 5, dtype=np.uint16).reshape((1, 1, 4, 5))
    _write_sample_czi(czi_path, data)
    (tmp_path / "sample.ome.tif").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1)


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


def test_compress_czi_rejects_resume_with_overwrite(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(16 * 16, dtype=np.uint16).reshape((1, 1, 16, 16))
    _write_sample_czi(czi_path, data)

    with pytest.raises(ValueError, match="mutually exclusive"):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, resume=True, overwrite=True)


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


def test_czi_output_dir_omits_stem_folder_for_single_tile_outputs(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"

    assert _czi_output_dir(czi_path, out_dir=None, tile_count=1) == tmp_path
    assert _czi_output_dir(czi_path, out_dir=tmp_path / "compressed", tile_count=1) == tmp_path / "compressed"
    assert _czi_output_dir(czi_path, out_dir=None, tile_count=2) == tmp_path / "sample"
    assert _czi_output_dir(czi_path, out_dir=tmp_path / "compressed", tile_count=2) == (
        tmp_path / "compressed" / "sample"
    )


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


def _raw_czi_metadata(ome_metadata: str) -> ET.Element:
    root = ET.fromstring(ome_metadata)
    raw_annotations = [
        annotation
        for annotation in root.findall(".//ome:XMLAnnotation", OME_NS)
        if annotation.attrib.get("Namespace") == "fishtools/czi/raw-metadata"
    ]
    assert len(raw_annotations) == 1
    assert root.find(f".//ome:AnnotationRef[@ID='{raw_annotations[0].attrib['ID']}']", OME_NS) is not None
    value = raw_annotations[0].find("ome:Value", OME_NS)
    assert value is not None
    assert len(value) == 1
    assert value[0].tag == "CZIProvenanceMetadata"
    return value[0]
