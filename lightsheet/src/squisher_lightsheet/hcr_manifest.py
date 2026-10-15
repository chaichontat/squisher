"""Verified acquisition manifest for one HCR dataset (WORKFLOW_HCR.md).

The manifest names every acquisition the workflow may use, the side it belongs to, and the biological label of each
channel index. Labels come from the emission wavelengths in the CZI metadata the converter embedded in every tile,
never from directory or file names. Sides are assigned by the author when drafting. Verification re-reads every tile
and fails on any difference, so an approved manifest cannot drift from the files it describes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from squisher_deconv.metadata import czi_dataset_metadata_payload
import tifffile

from squisher_lightsheet._legacy.stitch_20x_tl_multiview import parse_ome_metadata

ARTIFACT_TYPE = "squisher_lightsheet.hcr_manifest.v1"
LAYOUT_SIDES = {"CL/CR": ("CL", "CR"), "TL/TR": ("TL", "TR")}
EMISSION_LABELS_NM = {415.0: "405", 498.0: "488", 524.0: "514", 571.0: "561", 648.0: "638"}
EMISSION_TOLERANCE_NM = 1.0
CHANNEL_PATH = "./Metadata/Information/Image/Dimensions/Channels/Channel"
PIXEL_KEYS = (
    "DimensionOrder",
    "SizeX",
    "SizeY",
    "SizeZ",
    "SizeC",
    "PhysicalSizeX",
    "PhysicalSizeY",
    "PhysicalSizeZ",
)


@dataclass(frozen=True)
class AcquisitionInput:
    side: str
    directory: Path


def parse_acquisition(value: str) -> AcquisitionInput:
    """Parse SIDE=DIRECTORY."""
    side, separator, directory = value.partition("=")
    if not separator or not side or not directory:
        raise ValueError(f"Expected SIDE=DIRECTORY, got {value!r}")
    return AcquisitionInput(side=side, directory=Path(directory))


def _ome_tiffs(directory: Path) -> list[Path]:
    tiles = sorted((*directory.glob("*.ome.tif"), *directory.glob("*.ome.tiff")))
    if not tiles:
        raise ValueError(f"No OME-TIFF tiles in {directory}")
    return tiles


def channel_labels(shared_metadata_xml: str) -> list[dict[str, Any]]:
    """Label each CZI channel, in channel-index order, from its emission wavelength."""
    channels = ElementTree.fromstring(shared_metadata_xml).findall(CHANNEL_PATH)
    if not channels:
        raise ValueError("Shared CZI metadata lists no image channels")
    labels = []
    for index, channel in enumerate(channels):
        text = channel.findtext("EmissionWavelength")
        if text is None:
            raise ValueError(f"CZI channel {index} ({channel.get('Name')}) has no emission wavelength")
        emission = float(text)
        matches = [
            label
            for nominal, label in EMISSION_LABELS_NM.items()
            if abs(emission - nominal) <= EMISSION_TOLERANCE_NM
        ]
        if len(matches) != 1:
            raise ValueError(f"CZI channel {index} emission {emission} nm matches no known HCR label")
        labels.append(
            {"index": index, "label": matches[0], "emission_nm": emission, "czi_name": channel.get("Name")}
        )
    return labels


def _pixels(tile: Path) -> dict[str, str]:
    """The OME Pixels attributes that fix a tile's shape and spacing, read from its header alone."""
    with tifffile.TiffFile(tile) as tif:
        ome = tif.ome_metadata
    if ome is None:
        raise ValueError(f"{tile} has no OME metadata")
    pixels = next(
        (
            element
            for element in ElementTree.fromstring(ome).iter()
            if element.tag.rsplit("}", 1)[-1] == "Pixels"
        ),
        None,
    )
    if pixels is None:
        raise ValueError(f"{tile} OME metadata has no Pixels element")
    return {key: pixels.attrib[key] for key in PIXEL_KEYS if key in pixels.attrib}


def _channel_count(axes: str, shape: Sequence[int]) -> int:
    if axes == "ZYX":
        return 1
    if axes in ("CZYX", "ZCYX"):
        return int(shape[axes.index("C")])
    raise ValueError(f"Unsupported OME axes {axes!r}")


def describe_acquisition(side: str, directory: Path) -> dict[str, Any]:
    """Record one acquisition: its tiles, channel labels, tracks, shape, spacing, and stage positions."""
    directory = directory.resolve()
    tiles = _ome_tiffs(directory)
    payload = czi_dataset_metadata_payload(tiles, tiles)
    groups = payload.get("czi_metadata_groups", [])
    shared = groups[0] if groups else payload["czi_shared_metadata_xml"]
    if shared is None:
        raise ValueError(f"{directory} tiles carry no shared CZI metadata, so channel identity is unknown")
    channels = channel_labels(shared)
    reference = parse_ome_metadata(tiles[0])
    if _channel_count(reference.axes, reference.shape) != len(channels):
        raise ValueError(
            f"{directory}: OME axes {reference.axes} shape {reference.shape} disagree with "
            f"{len(channels)} CZI channels"
        )
    labels = [channel["label"] for channel in channels]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{directory} has repeated channel labels {labels}")
    reference_pixels = _pixels(tiles[0])
    records = []
    for tile, position in zip(tiles, payload["positions"], strict=True):
        if _pixels(tile) != reference_pixels:
            raise ValueError(f"{tile} differs in OME pixel size or spacing from {tiles[0]}")
        stat = tile.stat()
        records.append(
            {
                "name": tile.name,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "output_tile_index": position.get("tile_index"),
                "mosaic_index": position.get("mosaic_index"),
                "stage_position_um": {axis: position[axis] for axis in "zyx" if axis in position} or None,
            }
        )
    described = {
        "side": side,
        "directory": str(directory),
        "channels": channels,
        "tracks": [list(track.channels) for track in reference.tracks],
        "axes": reference.axes,
        "shape": list(reference.shape),
        "spacing_um": {axis: float(reference.spacing[axis]) for axis in "zyx"},
        "tiles": records,
    }
    if groups:
        described["czi_metadata_groups"] = [
            {
                "xml": xml,
                "tiles": [
                    tile.name
                    for tile, position in zip(tiles, payload["positions"], strict=True)
                    if position["metadata_group_index"] == index
                ],
            }
            for index, xml in enumerate(groups)
        ]
    return described


def draft_manifest(*, dataset: str, layout: str, acquisitions: Sequence[AcquisitionInput]) -> dict[str, Any]:
    if layout not in LAYOUT_SIDES:
        raise ValueError(f"Unknown layout {layout!r}; expected one of {sorted(LAYOUT_SIDES)}")
    sides = LAYOUT_SIDES[layout]
    described = []
    for acquisition in acquisitions:
        if acquisition.side not in sides:
            raise ValueError(f"Side {acquisition.side!r} is not part of layout {layout}")
        described.append(describe_acquisition(acquisition.side, acquisition.directory))
    for side in sides:
        labels = [
            channel["label"] for item in described if item["side"] == side for channel in item["channels"]
        ]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise ValueError(f"Side {side} has more than one acquisition for {repeated}")
        if "561" not in labels:
            raise ValueError(f"Side {side} has no 561 acquisition, which defines its reference geometry")
    return {
        "artifact_type": ARTIFACT_TYPE,
        "dataset": dataset,
        "layout": layout,
        "fixed_side": sides[0],
        "moving_side": sides[1],
        "acquisitions": described,
        "approved": False,
    }


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    """Re-read every listed acquisition and fail on any difference from the manifest."""
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"Not a {ARTIFACT_TYPE} manifest")
    expected = draft_manifest(
        dataset=manifest["dataset"],
        layout=manifest["layout"],
        acquisitions=[
            AcquisitionInput(side=item["side"], directory=Path(item["directory"]))
            for item in manifest["acquisitions"]
        ],
    )
    for key in ("fixed_side", "moving_side", "acquisitions"):
        if manifest[key] != expected[key]:
            raise ValueError(f"Manifest {key} no longer matches the acquisition files")


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
