from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from squisher_deconv.metadata import SourceMetadata, czi_dataset_metadata_payload, json_dumps_strict
from squisher_deconv.sink import write_streamed_ome_zarr, write_streamed_ome_zarr_with_sidecar
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
        source.write_bytes(b"placeholder")
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
    ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="source-image">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
            SizeX="6" SizeY="5" SizeZ="2" SizeC="2" SizeT="1"
            PhysicalSizeX="0.3" PhysicalSizeXUnit="µm"
            PhysicalSizeY="0.4" PhysicalSizeYUnit="µm"
            PhysicalSizeZ="0.6" PhysicalSizeZUnit="µm">
      <Channel ID="Channel:0:0" Name="WGA" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" Name="DAPI" SamplesPerPixel="1"/>
      <Plane TheC="0" TheZ="0" TheT="0" PositionX="12.5" PositionXUnit="µm"
             PositionY="23.5" PositionYUnit="µm" PositionZ="34.5" PositionZUnit="µm"/>
    </Pixels>
  </Image>
</OME>
"""
    source = _source(tmp_path, channels=2, z_count=2, height=5, width=6, ome_xml=ome_xml)
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
    assert tuple(array.chunks) == (1, 1, 5, 6)
    assert tuple(array.metadata.shards) == (1, 2, 5, 6)
    assert np.dtype(array.dtype) == np.uint16
    assert tuple(array.metadata.dimension_names) == ("c", "z", "y", "x")
    assert array.metadata.zarr_format == 3
    assert (output / "zarr.json").exists()
    assert not (output / ".zgroup").exists()
    assert root.attrs["squisher_complete"] is True
    ome = root.attrs["ome"]
    assert ome["version"] == "0.5"
    assert [dataset["path"] for dataset in ome["multiscales"][0]["datasets"]] == ["0", "1", "2"]
    assert [axis.get("unit") for axis in ome["multiscales"][0]["axes"]] == [
        None,
        "micrometer",
        "micrometer",
        "micrometer",
    ]
    assert ome["multiscales"][0]["datasets"][0]["coordinateTransformations"] == [
        {"type": "scale", "scale": [1.0, 0.6, 0.4, 0.3]},
        {"type": "translation", "translation": [0.0, 34.5, 23.5, 12.5]},
    ]
    assert ome["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]["scale"] == [
        1.0,
        0.6,
        0.8,
        0.6,
    ]
    assert ome["multiscales"][0]["datasets"][2]["coordinateTransformations"][0]["scale"] == [
        1.0,
        0.6,
        1.6,
        1.2,
    ]
    assert root.attrs["squisher_deconv"]["provenance"] == {"tool": "test"}
    assert root.attrs["squisher_deconv"]["source_metadata"]["ome_xml"] == ome_xml
    assert root.attrs["squisher_deconv"]["source_ome"]["channel_names"] == ["WGA", "DAPI"]
    assert root.attrs["squisher_deconv"]["source_metadata_summary"]["raw_shape"] == [4, 5, 6]
    assert root.attrs["squisher_deconv"]["storage"]["codec"] == {
        "name": "squisher.jpegxr",
        "level": 0.7,
        "checksum": "crc32c",
    }
    np.testing.assert_allclose(np.moveaxis(array[:], 0, 1).reshape(4, 5, 6), payload, atol=4)
    expected_level1 = np.rint(payload[:, :4, :].reshape(4, 2, 2, 3, 2).mean(axis=(2, 4))).astype(np.uint16)
    expected_level2 = np.rint(payload[:, :4, :4].reshape(4, 1, 4, 1, 4).mean(axis=(2, 4))).astype(np.uint16)
    # JPEG-XR block coding is least accurate for these intentionally tiny 2x3 test planes.
    np.testing.assert_allclose(np.moveaxis(root["1"][:], 0, 1).reshape(4, 2, 3), expected_level1, atol=16)
    np.testing.assert_allclose(np.moveaxis(root["2"][:], 0, 1).reshape(4, 1, 1), expected_level2, atol=4)


def test_streamed_ome_zarr_sidecar_failure_removes_final_output(tmp_path, monkeypatch) -> None:
    source = _source(tmp_path, channels=1, z_count=1, height=2, width=2)
    output = tmp_path / "out.ome.zarr"
    sidecar = tmp_path / "out.deconv.json"
    original_replace = Path.replace

    def fail_sidecar_replace(self: Path, target: Path) -> Path:
        if self.name == ".out.deconv.json.partial":
            raise OSError("forced sidecar finalization failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_sidecar_replace)

    with pytest.raises(OSError, match="forced sidecar finalization failure"):
        write_streamed_ome_zarr_with_sidecar(
            output,
            sidecar_path=sidecar,
            sidecar_text="{}\n",
            source=source,
            core_plane_chunks=[np.zeros((1, 2, 2), dtype=np.uint16)],
            output_mode="u16",
            provenance={"tool": "test"},
        )

    assert not output.exists()
    assert not sidecar.exists()
    assert not (tmp_path / ".out.deconv.json.partial").exists()


def test_streamed_ome_zarr_overwrite_failure_restores_existing_pair(tmp_path, monkeypatch) -> None:
    source = _source(tmp_path, channels=1, z_count=1, height=2, width=2)
    output = tmp_path / "out.ome.zarr"
    output.mkdir()
    (output / "old-data").write_text("keep\n")
    sidecar = tmp_path / "out.deconv.json"
    sidecar.write_text("old-sidecar\n")
    original_replace = Path.replace

    def fail_sidecar_replace(self: Path, target: Path) -> Path:
        if self.name == ".out.deconv.json.partial":
            raise OSError("forced sidecar finalization failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_sidecar_replace)

    with pytest.raises(OSError, match="forced sidecar finalization failure"):
        write_streamed_ome_zarr_with_sidecar(
            output,
            sidecar_path=sidecar,
            sidecar_text="new-sidecar\n",
            source=source,
            core_plane_chunks=[np.zeros((1, 2, 2), dtype=np.uint16)],
            output_mode="u16",
            provenance={"tool": "test"},
            overwrite=True,
        )

    assert (output / "old-data").read_text() == "keep\n"
    assert sidecar.read_text() == "old-sidecar\n"
    assert not (tmp_path / ".out.ome.zarr.backup").exists()
    assert not (tmp_path / ".out.deconv.json.backup").exists()


def test_streamed_ome_zarr_restores_existing_pair_before_fallible_cleanup(tmp_path, monkeypatch) -> None:
    source = _source(tmp_path, channels=1, z_count=1, height=2, width=2)
    output = tmp_path / "out.ome.zarr"
    output.mkdir()
    (output / "old-data").write_text("keep\n")
    sidecar = tmp_path / "out.deconv.json"
    sidecar.write_text("old-sidecar\n")
    original_replace = Path.replace

    def fail_sidecar_replace(self: Path, target: Path) -> Path:
        if self.name == ".out.deconv.json.partial":
            raise OSError("forced sidecar finalization failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_sidecar_replace)
    import squisher_deconv.sink as sink

    original_remove = sink._remove_path

    def fail_staged_cleanup(path: Path) -> None:
        if path.name == ".out.ome.zarr.staged":
            raise OSError("forced staged cleanup failure")
        original_remove(path)

    monkeypatch.setattr(sink, "_remove_path", fail_staged_cleanup)

    with pytest.raises(BaseExceptionGroup) as exc_info:
        write_streamed_ome_zarr_with_sidecar(
            output,
            sidecar_path=sidecar,
            sidecar_text="new-sidecar\n",
            source=source,
            core_plane_chunks=[np.zeros((1, 2, 2), dtype=np.uint16)],
            output_mode="u16",
            provenance={"tool": "test"},
            overwrite=True,
        )

    assert "forced sidecar finalization failure" in repr(exc_info.value.exceptions)
    assert "forced staged cleanup failure" in repr(exc_info.value.exceptions)
    assert (output / "old-data").read_text() == "keep\n"
    assert sidecar.read_text() == "old-sidecar\n"


def test_streamed_ome_zarr_overwrite_promotion_failure_restores_existing_output(
    tmp_path, monkeypatch
) -> None:
    source = _source(tmp_path, channels=1, z_count=1, height=2, width=2)
    output = tmp_path / "out.ome.zarr"
    output.mkdir()
    (output / "old-data").write_text("keep\n")
    original_rename = Path.rename

    def fail_partial_promotion(self: Path, target: Path) -> Path:
        if self.name == ".out.ome.zarr.partial":
            raise OSError("forced output promotion failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_partial_promotion)

    with pytest.raises(OSError, match="forced output promotion failure"):
        write_streamed_ome_zarr(
            output,
            source=source,
            core_plane_chunks=[np.zeros((1, 2, 2), dtype=np.uint16)],
            output_mode="u16",
            provenance={"tool": "test"},
            overwrite=True,
        )

    assert (output / "old-data").read_text() == "keep\n"
    assert not (tmp_path / ".out.ome.zarr.backup").exists()


def _source(
    tmp_path,
    *,
    channels: int,
    z_count: int,
    height: int,
    width: int,
    ome_xml: str | None = None,
) -> TiffLogicalSource:
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
            ome_xml=ome_xml,
            tags={},
            raw_shape=(z_count * channels, height, width),
            raw_dtype="uint16",
            metadata_hash="hash",
        ),
    )
