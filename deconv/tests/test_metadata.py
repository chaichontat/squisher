from __future__ import annotations

import numpy as np
import pytest
import tifffile
import zarr

from squisher_deconv.metadata import SourceMetadata, czi_dataset_metadata_payload, json_dumps_strict
from squisher_deconv.sink import write_streamed_ome_zarr
from squisher_deconv.source import TiffLogicalSource


def test_json_dumps_strict_rejects_non_serializable_values() -> None:
    with pytest.raises(TypeError, match="Test payload must be JSON serializable"):
        json_dumps_strict({"bad": object()}, context="Test payload")


def test_czi_dataset_metadata_payload_collects_shared_metadata_and_tile_positions(tmp_path) -> None:
    shared = "<ImageDocument><Metadata><Information><Document><Name>sample</Name></Document></Information></Metadata></ImageDocument>"
    sources = []
    outputs = []
    for index, (x, y, z) in enumerate(((12.5, 23.5, 34.5), (45.5, 56.5, 67.5))):
        source = tmp_path / f"tile-{index}.ome.tif"
        output = tmp_path / "out" / f"tile-{index}.ome.zarr"
        sources.append(source)
        outputs.append(output)
        ome_xml = f"""<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0"><Pixels ID="Pixels:0">
    <Plane PositionX="{x}" PositionXUnit="&#181;m" PositionY="{y}" PositionYUnit="&#181;m"
           PositionZ="{z}" PositionZUnit="&#181;m"/>
  </Pixels></Image>
  <StructuredAnnotations>
    <MapAnnotation ID="Annotation:0"><Value>
      <M K="squisher.output_tile_index">{index}</M>
      <M K="czi.mosaic_index">{index + 10}</M>
    </Value></MapAnnotation>
    <XMLAnnotation ID="Annotation:1" Namespace="squisher/czi/shared-metadata"><Value>
      <CZISharedMetadata xmlns="">{shared}</CZISharedMetadata>
    </Value></XMLAnnotation>
  </StructuredAnnotations>
</OME>"""
        tifffile.imwrite(
            source,
            np.zeros((1, 2, 2), dtype=np.uint16),
            description=ome_xml,
            metadata=None,
            photometric="minisblack",
        )

    payload = czi_dataset_metadata_payload(sources, outputs)

    assert payload["czi_shared_metadata_xml"] == shared
    assert payload["positions"] == [
        {
            "path": str(outputs[0]),
            "source": str(sources[0]),
            "tile_index": 0,
            "mosaic_index": 10,
            "x": 12.5,
            "x_unit": "µm",
            "y": 23.5,
            "y_unit": "µm",
            "z": 34.5,
            "z_unit": "µm",
        },
        {
            "path": str(outputs[1]),
            "source": str(sources[1]),
            "tile_index": 1,
            "mosaic_index": 11,
            "x": 45.5,
            "x_unit": "µm",
            "y": 56.5,
            "y_unit": "µm",
            "z": 67.5,
            "z_unit": "µm",
        },
    ]


def test_czi_dataset_metadata_payload_rejects_mismatched_shared_metadata(tmp_path) -> None:
    sources = []
    for index in range(2):
        source = tmp_path / f"tile-{index}.ome.tif"
        sources.append(source)
        ome_xml = f"""<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0"><Pixels ID="Pixels:0"><Plane PositionX="{index}" PositionY="0"/></Pixels></Image>
  <StructuredAnnotations><XMLAnnotation Namespace="squisher/czi/shared-metadata"><Value>
    <CZISharedMetadata xmlns=""><ImageDocument><Metadata id="{index}"/></ImageDocument></CZISharedMetadata>
  </Value></XMLAnnotation></StructuredAnnotations>
</OME>"""
        tifffile.imwrite(
            source,
            np.zeros((1, 2, 2), dtype=np.uint16),
            description=ome_xml,
            metadata=None,
            photometric="minisblack",
        )

    with pytest.raises(ValueError, match="different shared CZI metadata"):
        czi_dataset_metadata_payload(sources, [tmp_path / "a", tmp_path / "b"])


def test_ome_zarr_provenance_serialization_failure_removes_partial_output(tmp_path) -> None:
    source = _source(tmp_path, channels=1, z_count=1, height=2, width=2)
    output = tmp_path / "out.ome.zarr"

    with pytest.raises(TypeError, match="OME-Zarr deconvolution provenance must be JSON serializable"):
        write_streamed_ome_zarr(
            output,
            source=source,
            core_plane_chunks=[np.zeros((1, 2, 2), dtype=np.uint16)],
            output_mode="u16",
            provenance={"bad": object()},
        )

    assert not output.exists()
    assert not (tmp_path / ".out.ome.zarr.partial").exists()


def test_streamed_ome_zarr_writes_czyx_array_with_metadata(tmp_path) -> None:
    source = _source(tmp_path, channels=2, z_count=2, height=5, width=6)
    output = tmp_path / "out.ome.zarr"
    payload = np.arange(2 * 2 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)

    write_streamed_ome_zarr(
        output,
        source=source,
        core_plane_chunks=[payload],
        output_mode="u16",
        provenance={"tool": "test"},
    )

    root = zarr.open_group(str(output), mode="r")
    array = root["0"]
    assert tuple(array.shape) == (2, 2, 5, 6)
    assert tuple(array.chunks) == (1, 2, 5, 6)
    assert np.dtype(array.dtype) == np.uint16
    assert array.attrs["_ARRAY_DIMENSIONS"] == ["c", "z", "y", "x"]
    assert root.attrs["squisher_complete"] is True
    assert root.attrs["multiscales"][0]["datasets"][0]["path"] == "0"
    assert root.attrs["squisher_deconv"]["provenance"] == {"tool": "test"}
    assert root.attrs["squisher_deconv"]["source_metadata_summary"]["raw_shape"] == [4, 5, 6]
    np.testing.assert_array_equal(np.moveaxis(array[:], 0, 1).reshape(4, 5, 6), payload)


def _source(tmp_path, *, channels: int, z_count: int, height: int, width: int) -> TiffLogicalSource:
    return TiffLogicalSource(
        path=tmp_path / "source.tif",
        channels=channels,
        z_count=z_count,
        height=height,
        width=width,
        dtype="uint16",
        metadata=SourceMetadata(
            shaped_metadata=[],
            imagej_metadata=None,
            ome_xml=None,
            tags={},
            raw_shape=(z_count * channels, height, width),
            raw_dtype="uint16",
            metadata_hash="hash",
        ),
    )
