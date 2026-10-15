from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from squisher_lightsheet import hcr_manifest
from squisher_lightsheet.cli import app


def _write_tile(
    path: Path,
    *,
    index: int,
    emissions: tuple[float, ...],
    spacing_xy: float = 0.3,
    shared: bool = True,
    acquisition_name: str | None = None,
) -> None:
    """An OME-TIFF tile carrying the CZI annotations the converter embeds."""
    channels = len(emissions)
    data = np.zeros((channels, 2, 4, 4) if channels > 1 else (2, 4, 4), dtype=np.uint16)
    tifffile.imwrite(
        path,
        data,
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "CZYX" if channels > 1 else "ZYX",
            "PhysicalSizeX": spacing_xy,
            "PhysicalSizeY": spacing_xy,
            "PhysicalSizeZ": 1.8,
        },
    )
    czi_channels = "".join(
        f'<Channel Id="c{channel}" Name="Cam{channel}"><EmissionWavelength>{emission}</EmissionWavelength></Channel>'
        for channel, emission in enumerate(emissions)
    )
    shared_xml = (
        "<ImageDocument><Metadata><Information><Image><Dimensions><Channels>"
        f"{czi_channels}</Channels></Dimensions></Image></Information></Metadata></ImageDocument>"
    )
    if acquisition_name is not None:
        shared_xml = shared_xml.replace(
            "<Information>", f"<Information><Document><Name>{acquisition_name}</Name></Document>"
        )
    annotations = (
        "<StructuredAnnotations>"
        f'<MapAnnotation ID="Annotation:0"><Value><M K="squisher.output_tile_index">{index}</M>'
        f'<M K="czi.mosaic_index">{index}</M></Value></MapAnnotation>'
        + (
            '<XMLAnnotation ID="Annotation:1" Namespace="squisher/czi/shared-metadata"><Value>'
            f'<CZISharedMetadata xmlns="">{shared_xml}</CZISharedMetadata></Value></XMLAnnotation>'
            if shared
            else ""
        )
        + "</StructuredAnnotations>"
    )
    with tifffile.TiffFile(path) as tif:
        ome = tif.ome_metadata
    assert ome is not None
    plane = f'<Plane TheZ="0" TheC="0" TheT="0" PositionX="{100.0 * index}" PositionY="0.0"/>'
    ome = ome.replace("</Pixels>", f"{plane}</Pixels>", 1)
    tifffile.tiffcomment(path, ome.replace("</OME>", f"{annotations}</OME>"))


def _acquisition(root: Path, name: str, emissions: tuple[float, ...], *, tiles: int = 2, **kwargs) -> Path:
    directory = root / name
    directory.mkdir()
    for index in range(tiles):
        _write_tile(directory / f"{name}.{index:03d}.ome.tif", index=index, emissions=emissions, **kwargs)
    return directory


def _dataset(root: Path) -> list[hcr_manifest.AcquisitionInput]:
    """CL in one four-channel acquisition; CR split, as 229Ptprz1 was acquired."""
    return [
        hcr_manifest.AcquisitionInput("CL", _acquisition(root, "CL-main", (571.0, 648.0, 524.0, 498.0))),
        hcr_manifest.AcquisitionInput("CR", _acquisition(root, "CR-561638", (571.0, 648.0))),
        hcr_manifest.AcquisitionInput("CR", _acquisition(root, "CR-514", (524.0,), spacing_xy=0.29)),
    ]


def test_draft_labels_channels_from_embedded_emission_wavelengths(tmp_path: Path) -> None:
    manifest = hcr_manifest.draft_manifest(dataset="d", layout="CL/CR", acquisitions=_dataset(tmp_path))

    labels = {
        Path(item["directory"]).name: [c["label"] for c in item["channels"]]
        for item in manifest["acquisitions"]
    }
    assert labels == {"CL-main": ["561", "638", "514", "488"], "CR-561638": ["561", "638"], "CR-514": ["514"]}
    assert (manifest["fixed_side"], manifest["moving_side"], manifest["approved"]) == ("CL", "CR", False)
    split = manifest["acquisitions"][2]
    assert split["spacing_um"]["x"] == pytest.approx(0.29)
    assert [tile["mosaic_index"] for tile in split["tiles"]] == [0, 1]
    assert split["tiles"][1]["stage_position_um"] == {"y": 0.0, "x": 100.0}


@pytest.mark.parametrize(
    ("emissions", "kwargs", "message"),
    [
        ((560.0,), {}, "matches no known HCR label"),
        ((524.0,), {"shared": False}, "channel identity is unknown"),
    ],
)
def test_draft_rejects_channels_without_a_known_identity(
    tmp_path: Path, emissions: tuple[float, ...], kwargs: dict, message: str
) -> None:
    acquisitions = [hcr_manifest.AcquisitionInput("CL", _acquisition(tmp_path, "CL", emissions, **kwargs))]

    with pytest.raises(ValueError, match=message):
        hcr_manifest.draft_manifest(dataset="d", layout="CL/CR", acquisitions=acquisitions)


def test_draft_requires_one_561_and_unique_labels_per_side(tmp_path: Path) -> None:
    acquisitions = _dataset(tmp_path)
    duplicate = acquisitions + [
        hcr_manifest.AcquisitionInput("CR", _acquisition(tmp_path, "CR-514b", (524.0,)))
    ]
    with pytest.raises(ValueError, match=r"more than one acquisition for \['514'\]"):
        hcr_manifest.draft_manifest(dataset="d", layout="CL/CR", acquisitions=duplicate)
    with pytest.raises(ValueError, match="no 561 acquisition"):
        hcr_manifest.draft_manifest(
            dataset="d", layout="CL/CR", acquisitions=acquisitions[:1] + acquisitions[2:]
        )
    with pytest.raises(ValueError, match="not part of layout"):
        hcr_manifest.draft_manifest(dataset="d", layout="TL/TR", acquisitions=acquisitions)


def test_draft_rejects_a_tile_with_different_spacing(tmp_path: Path) -> None:
    directory = _acquisition(tmp_path, "CL", (571.0,))
    _write_tile(directory / "CL.002.ome.tif", index=2, emissions=(571.0,), spacing_xy=0.31)

    with pytest.raises(ValueError, match="differs in OME pixel size or spacing"):
        hcr_manifest.describe_acquisition("CL", directory)


def test_verify_fails_when_an_acquisition_changes(tmp_path: Path) -> None:
    acquisitions = _dataset(tmp_path)
    manifest = hcr_manifest.draft_manifest(dataset="d", layout="CL/CR", acquisitions=acquisitions)
    hcr_manifest.verify_manifest(manifest)

    _write_tile(
        acquisitions[2].directory / "CR-514.002.ome.tif", index=2, emissions=(524.0,), spacing_xy=0.29
    )

    with pytest.raises(ValueError, match="acquisitions no longer matches"):
        hcr_manifest.verify_manifest(manifest)


def test_cli_drafts_then_verifies(tmp_path: Path) -> None:
    acquisitions = _dataset(tmp_path)
    output = tmp_path / "manifest.json"
    arguments = ["hcr-manifest", "draft", "--dataset", "d", "--layout", "CL/CR", "--output", str(output)]
    for acquisition in acquisitions:
        arguments += ["--acquisition", f"{acquisition.side}={acquisition.directory}"]

    drafted = CliRunner().invoke(app, arguments)
    verified = CliRunner().invoke(app, ["hcr-manifest", "verify", str(output)])

    assert drafted.exit_code == 0, drafted.output
    assert verified.exit_code == 0, verified.output
    assert json.loads(output.read_text())["dataset"] == "d"


def test_manifest_retains_top_up_groups_and_verifies_exact_metadata(tmp_path: Path) -> None:
    acquisitions = _dataset(tmp_path)
    directory = acquisitions[0].directory
    for index in range(2):
        _write_tile(
            directory / f"CL-main.{index:03d}.ome.tif",
            index=index,
            emissions=(571.0, 648.0, 524.0, 498.0),
            acquisition_name=f"topup{index}",
        )
    manifest = hcr_manifest.draft_manifest(dataset="d", layout="CL/CR", acquisitions=acquisitions)
    groups = manifest["acquisitions"][0]["czi_metadata_groups"]
    assert len(manifest["acquisitions"][0]["tiles"]) == 2
    assert [group["tiles"] for group in groups] == [["CL-main.000.ome.tif"], ["CL-main.001.ome.tif"]]
    assert "topup1" in groups[1]["xml"]
    hcr_manifest.verify_manifest(manifest)

    tile = directory / "CL-main.001.ome.tif"
    before = tile.stat()
    with tifffile.TiffFile(tile) as tif:
        xml = tif.ome_metadata
    assert xml is not None
    tifffile.tiffcomment(tile, xml.replace("topup1", "topup2"))
    os.utime(tile, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert tile.stat().st_size == before.st_size
    with pytest.raises(ValueError, match="acquisitions no longer matches"):
        hcr_manifest.verify_manifest(manifest)
