import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pylibCZIrw import czi
from tifffile import TiffFile

from squisher.compression import compress_czi_to_ome_tiff


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

    with TiffFile(tmp_path / "sample.000.ome.tif") as tif:
        assert 'PositionX="22.5"' in tif.ome_metadata
        assert 'PositionY="11.5"' in tif.ome_metadata
        assert json.loads(_map_annotation_values(tif.ome_metadata)["squisher.placement_json"])["version"] == 1
    with TiffFile(tmp_path / "sample.001.ome.tif") as tif:
        assert 'PositionX="44.5"' in tif.ome_metadata
        assert 'PositionY="33.5"' in tif.ome_metadata


def test_compress_czi_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    czi_path = tmp_path / "sample.czi"
    data = np.arange(4 * 5, dtype=np.uint16).reshape((1, 1, 4, 5))
    _write_sample_czi(czi_path, data)
    (tmp_path / "sample.ome.tif").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        compress_czi_to_ome_tiff(czi_path, level=90, tile_size=16, maxworkers=1)


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
