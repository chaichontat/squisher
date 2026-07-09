from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile


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


def provenance_payload(
    source: Path,
    *,
    channels: int,
    halo: int,
    psf_paths: Sequence[Path] | None,
    basic_paths: Sequence[Path] | None = None,
    output_mode: str,
    scaling_path: Path | None,
    devices: list[int],
    queue_depth: int,
) -> dict[str, Any]:
    return {
        "tool": "squisher-deconv",
        "source": str(source),
        "channels": int(channels),
        "halo": int(halo),
        "psfs": file_provenance_records(psf_paths or []),
        "basic_profiles": file_provenance_records(basic_paths or []),
        "output_mode": output_mode,
        "compression_tiff_tag": compression_tiff_tag(output_mode),
        "scaling_path": None if scaling_path is None else str(scaling_path),
        "devices": [int(device) for device in devices],
        "queue_depth": int(queue_depth),
        "versions": dependency_versions(),
    }


def compression_tiff_tag(output_mode: str) -> int | None:
    if output_mode == "u16":
        return 22610
    if output_mode == "float32":
        return None
    raise ValueError(f"Unsupported output_mode={output_mode!r}; expected 'u16' or 'float32'.")


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

    packages = ("numpy", "tifffile", "imagecodecs", "squisher-deconv")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
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
