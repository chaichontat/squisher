from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree

import numpy as np
import tifffile


CZI_SHARED_METADATA_NAMESPACE = "squisher/czi/shared-metadata"


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    shaped_metadata: list[Any]
    imagej_metadata: dict[str, Any] | None
    ome_xml: str | None
    tags: dict[str, Any]
    raw_shape: tuple[int, ...]
    raw_dtype: str
    metadata_hash: str


def read_source_metadata(path: Path) -> SourceMetadata:
    with tifffile.TiffFile(path) as tif:
        shaped_metadata = _jsonable(tif.shaped_metadata or [])
        imagej_metadata = _jsonable(tif.imagej_metadata) if tif.imagej_metadata is not None else None
        ome_xml = tif.ome_metadata
        series = tif.series[0]
        tags = _selected_tags(tif.pages[0])
        raw_shape = tuple(int(v) for v in series.shape)
        raw_dtype = str(series.dtype)
    payload = {
        "shaped_metadata": shaped_metadata,
        "imagej_metadata": imagej_metadata,
        "ome_xml": ome_xml,
        "tags": tags,
        "raw_shape": raw_shape,
        "raw_dtype": raw_dtype,
    }
    encoded = json_dumps_strict(payload, context=f"Source metadata for {path}").encode("utf-8")
    return SourceMetadata(
        shaped_metadata=shaped_metadata,
        imagej_metadata=imagej_metadata,
        ome_xml=ome_xml,
        tags=tags,
        raw_shape=raw_shape,
        raw_dtype=raw_dtype,
        metadata_hash=hashlib.sha256(encoded).hexdigest(),
    )


def czi_dataset_metadata_payload(sources: Sequence[Path], outputs: Sequence[Path]) -> dict[str, Any]:
    """Collect the CZI metadata shared by all inputs and each tile's OME position."""
    if len(sources) != len(outputs):
        raise ValueError(
            f"Expected one output per source, got {len(sources)} sources and {len(outputs)} outputs."
        )

    shared_metadata_xml: str | None = None
    positions: list[dict[str, Any]] = []
    missing_shared_metadata: list[str] = []
    for source, output in zip(sources, outputs, strict=True):
        source = Path(source)
        with tifffile.TiffFile(source) as tif:
            ome_xml = tif.ome_metadata
        if ome_xml is None:
            missing_shared_metadata.append(str(source))
            positions.append({"path": str(output), "source": str(source)})
            continue
        try:
            ome = ElementTree.fromstring(ome_xml)
        except ElementTree.ParseError as exc:
            raise ValueError(f"Invalid OME-XML metadata in {source}: {exc}") from exc

        current_shared = _czi_shared_metadata_xml(ome, source=source)
        if current_shared is not None:
            if shared_metadata_xml is None:
                shared_metadata_xml = current_shared
            elif current_shared != shared_metadata_xml:
                raise ValueError(
                    f"{source} contains different shared CZI metadata from the first input tile."
                )
        else:
            missing_shared_metadata.append(str(source))
        positions.append(_ome_position_record(ome, source=source, output=Path(output)))

    if shared_metadata_xml is not None:
        if missing_shared_metadata:
            raise ValueError(
                f"Input tile(s) are missing the shared CZI metadata present on other tiles: {missing_shared_metadata}"
            )
        missing = [record["source"] for record in positions if "tile_index" not in record]
        if missing:
            raise ValueError(f"CZI input tile(s) are missing tile metadata: {missing}")
    return {"czi_shared_metadata_xml": shared_metadata_xml, "positions": positions}


def _czi_shared_metadata_xml(ome: ElementTree.Element, *, source: Path) -> str | None:
    annotations = [
        element
        for element in ome.iter()
        if _local_name(element.tag) == "XMLAnnotation"
        and element.attrib.get("Namespace") == CZI_SHARED_METADATA_NAMESPACE
    ]
    if not annotations:
        return None
    if len(annotations) != 1:
        raise ValueError(
            f"Expected one shared CZI metadata annotation in {source}, found {len(annotations)}."
        )
    value = next((element for element in annotations[0] if _local_name(element.tag) == "Value"), None)
    wrapper = None if value is None else next(iter(value), None)
    shared = None if wrapper is None else next(iter(wrapper), None)
    if (
        wrapper is None
        or _local_name(wrapper.tag) != "CZISharedMetadata"
        or shared is None
        or _local_name(shared.tag) != "ImageDocument"
    ):
        raise ValueError(f"Malformed shared CZI metadata annotation in {source}.")
    return ElementTree.tostring(shared, encoding="unicode")


def _ome_position_record(ome: ElementTree.Element, *, source: Path, output: Path) -> dict[str, Any]:
    map_values = {
        element.attrib["K"]: element.text or ""
        for element in ome.iter()
        if _local_name(element.tag) == "M" and "K" in element.attrib
    }
    record: dict[str, Any] = {"path": str(output), "source": str(source)}
    for metadata_key, record_key in (
        ("squisher.output_tile_index", "tile_index"),
        ("czi.mosaic_index", "mosaic_index"),
    ):
        if metadata_key in map_values:
            try:
                record[record_key] = int(map_values[metadata_key])
            except ValueError as exc:
                raise ValueError(f"Invalid {metadata_key} in {source}: {map_values[metadata_key]!r}") from exc

    plane = next((element for element in ome.iter() if _local_name(element.tag) == "Plane"), None)
    if plane is None:
        return record
    for axis in ("x", "y", "z"):
        value = plane.attrib.get(f"Position{axis.upper()}")
        if value is not None:
            try:
                record[axis] = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid Position{axis.upper()} in {source}: {value!r}") from exc
            unit = plane.attrib.get(f"Position{axis.upper()}Unit")
            if unit is not None:
                record[f"{axis}_unit"] = unit
    return record


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def provenance_payload(
    source: Path,
    *,
    channels: int,
    halo: int,
    slab_depth: int,
    iterations: int | None,
    psf_paths: Sequence[Path] | None,
    basic_paths: Sequence[Path] | None = None,
    output_mode: str,
    scaling_path: Path | None,
    devices: list[int],
    queue_depth: int,
    jpegxr_level: float,
) -> dict[str, Any]:
    return {
        "tool": "squisher-deconv",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_file": file_stat_record(source),
        "run_settings": {
            "channels": int(channels),
            "halo": int(halo),
            "slab_depth": int(slab_depth),
            "iterations": None if iterations is None else int(iterations),
            "output_mode": output_mode,
            "devices": [int(device) for device in devices],
            "queue_depth": int(queue_depth),
            "jpegxr_level": float(jpegxr_level),
        },
        "psfs": file_provenance_records(psf_paths or []),
        "basic_profiles": file_provenance_records(basic_paths or []),
        "scaling": None if scaling_path is None else file_provenance_records([scaling_path])[0],
        "versions": dependency_versions(),
    }


def file_stat_record(path: Path) -> dict[str, str | int]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def file_provenance_records(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": _file_sha256(path)} for path in paths]


def json_dumps_strict(payload: Any, *, context: str, indent: int | None = None) -> str:
    try:
        return json.dumps(payload, indent=indent, sort_keys=True)
    except TypeError as exc:
        raise TypeError(f"{context} must be JSON serializable: {exc}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    import tomllib

    packages = {
        "numpy": "numpy",
        "tifffile": "tifffile",
        "imagecodecs": "imagecodecs",
        "zarr": "zarr",
        "cupy": "cupy-cuda12x",
        "basicpy": "BaSiCPy",
        "squisher-deconv": "squisher-deconv",
    }
    versions: dict[str, str] = {}
    for label, distribution in packages.items():
        try:
            versions[label] = version(distribution)
        except PackageNotFoundError:
            versions[label] = "not-installed"
    if versions["squisher-deconv"] == "not-installed":
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject.exists():
            versions["squisher-deconv"] = str(tomllib.loads(pyproject.read_text())["project"]["version"])
    return versions


def _selected_tags(page: tifffile.TiffPage) -> dict[str, Any]:
    selected = {}
    for name in (
        "ImageDescription",
        "Software",
        "XResolution",
        "YResolution",
        "ResolutionUnit",
        "DateTime",
        "Artist",
        "Copyright",
    ):
        tag = page.tags.get(name)
        if tag is not None:
            selected[name] = _jsonable(tag.value)
    return selected


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    raise TypeError(f"Unsupported metadata value of type {type(value).__name__}")
