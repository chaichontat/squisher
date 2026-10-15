from __future__ import annotations

from collections.abc import Callable, Mapping
import gzip
import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np

from squisher.jpegxr_zarr import register_jpegxr_codec
from squisher_lightsheet import ngff


DEFAULT_GZIP_LEVEL = 1
DEFAULT_SLAB_PLANES = 24


def export_nrrd(
    *,
    source: Path,
    output: Path,
    level: int = 0,
    channel: str | None = None,
    gzip_level: int = DEFAULT_GZIP_LEVEL,
    slab_planes: int = DEFAULT_SLAB_PLANES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stream one 3-D OME-Zarr pyramid level to a validated gzip NRRD."""
    import SimpleITK as sitk
    import zarr

    started = time.perf_counter()
    source = source.resolve()
    output = output.resolve()
    _validate_paths_and_options(
        source=source,
        output=output,
        level=level,
        gzip_level=gzip_level,
        slab_planes=slab_planes,
    )

    register_jpegxr_codec()
    root = zarr.open_group(str(source), mode="r")
    if root.attrs.get("squisher_complete") is False:
        raise ValueError(f"Source OME-Zarr is explicitly marked incomplete: {source}")

    records = ngff.multiscales(root)
    if len(records) != 1:
        raise ValueError(f"Expected exactly one OME-Zarr multiscale in {source}, found {len(records)}")
    dataset_path = ngff.level_path(root, level=level, context=source)
    array = root[dataset_path]
    if ngff.axes(root, array) != "ZYX":
        raise ValueError(f"Expected a 3-D ZYX OME-Zarr level, got shape {array.shape}")
    _require_micrometer_axes(records[0], source=source)

    _, scale, origin, has_scale, has_translation = ngff.scale_translation(
        root, dataset_index=ngff.dataset_paths(root).index(dataset_path)
    )
    if not has_scale or not has_translation:
        raise ValueError(f"OME-Zarr level {dataset_path} must define both scale and translation")
    scale_zyx_um = np.asarray(scale, dtype=np.float64)
    origin_zyx_um = np.asarray(origin, dtype=np.float64)
    if not np.isfinite(scale_zyx_um).all() or np.any(scale_zyx_um <= 0):
        raise ValueError(f"OME-Zarr scale must contain finite positive values, got {scale}")
    if not np.isfinite(origin_zyx_um).all():
        raise ValueError(f"OME-Zarr translation must contain finite values, got {origin}")

    source_dtype = np.dtype(array.dtype)
    if source_dtype.newbyteorder("=") != np.dtype("uint16"):
        raise ValueError(f"Expected uint16 OME-Zarr data, got {source_dtype}")
    disk_dtype = np.dtype("<u2")
    shape_zyx = tuple(int(size) for size in array.shape)
    header = _nrrd_header(
        shape_zyx=shape_zyx,
        scale_zyx_um=scale_zyx_um,
        origin_zyx_um=origin_zyx_um,
        source=source,
        level=level,
        channel=channel,
    )

    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    occupied = [path for path in (output, temporary) if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite existing output path(s): {occupied}")
    output.parent.mkdir(parents=True, exist_ok=True)

    decoded_hash = hashlib.sha256()
    created_temporary = False
    try:
        with temporary.open("xb") as raw:
            created_temporary = True
            raw.write(header)
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=gzip_level, mtime=0) as encoded:
                for start in range(0, shape_zyx[0], slab_planes):
                    stop = min(start + slab_planes, shape_zyx[0])
                    slab = np.ascontiguousarray(array[start:stop], dtype=disk_dtype)
                    payload = memoryview(slab).cast("B")
                    decoded_hash.update(payload)
                    encoded.write(payload)
                    if progress is not None:
                        progress(f"Exported Z planes {stop}/{shape_zyx[0]}")
        write_seconds = time.perf_counter() - started

        if progress is not None:
            progress("Validating NRRD payload and physical metadata")
        validation_started = time.perf_counter()
        restored_hash, restored_bytes = _validate_nrrd(
            temporary,
            sitk=sitk,
            expected_shape_zyx=shape_zyx,
            expected_scale_zyx_um=scale_zyx_um,
            expected_origin_zyx_um=origin_zyx_um,
        )
        expected_bytes = int(array.nbytes)
        if restored_bytes != expected_bytes:
            raise ValueError(f"NRRD decoded byte count mismatch: {restored_bytes} != {expected_bytes}")
        if restored_hash != decoded_hash.hexdigest():
            raise ValueError("NRRD decoded voxel hash does not match the OME-Zarr source")
        validation_seconds = time.perf_counter() - validation_started
        temporary.rename(output)
    except BaseException:
        if created_temporary:
            temporary.unlink(missing_ok=True)
        raise

    return {
        "artifact_type": "squisher_lightsheet.nrrd_export.v1",
        "source": str(source),
        "source_level": int(level),
        "source_dataset_path": dataset_path,
        "shape_zyx": list(shape_zyx),
        "dtype": "uint16",
        "scale_zyx_um": scale_zyx_um.tolist(),
        "origin_zyx_um": origin_zyx_um.tolist(),
        "channel": channel,
        "output": str(output),
        "nrrd_compression": f"gzip level {gzip_level}",
        "nrrd_coordinates": (
            "Acquisition XYZ mapped to LPS XYZ in mm; identity direction; "
            "no anatomical orientation inferred"
        ),
        "decoded_zyx_sha256": decoded_hash.hexdigest(),
        "all_voxels_equal": True,
        "slab_planes": int(slab_planes),
        "write_seconds": write_seconds,
        "validation_seconds": validation_seconds,
        "total_seconds": time.perf_counter() - started,
        "nrrd_bytes": output.stat().st_size,
    }


def _validate_paths_and_options(
    *, source: Path, output: Path, level: int, gzip_level: int, slab_planes: int
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source OME-Zarr does not exist or is not a directory: {source}")
    if output.suffix.lower() != ".nrrd":
        raise ValueError(f"Output must end in .nrrd, got {output}")
    if level < 0:
        raise ValueError(f"level must be non-negative, got {level}")
    if not 0 <= gzip_level <= 9:
        raise ValueError(f"gzip_level must be between 0 and 9, got {gzip_level}")
    if slab_planes < 1:
        raise ValueError(f"slab_planes must be positive, got {slab_planes}")


def _require_micrometer_axes(multiscale: Mapping[str, Any], *, source: Path) -> None:
    axes = multiscale.get("axes")
    if not isinstance(axes, list) or len(axes) != 3:
        raise ValueError(f"Expected three OME-Zarr axes in {source}")
    names = [axis.get("name") if isinstance(axis, Mapping) else None for axis in axes]
    units = [axis.get("unit") if isinstance(axis, Mapping) else None for axis in axes]
    if names != ["z", "y", "x"] or units != ["micrometer", "micrometer", "micrometer"]:
        raise ValueError(f"Expected micrometer ZYX axes in {source}, got names={names}, units={units}")


def _nrrd_header(
    *,
    shape_zyx: tuple[int, int, int],
    scale_zyx_um: np.ndarray,
    origin_zyx_um: np.ndarray,
    source: Path,
    level: int,
    channel: str | None,
) -> bytes:
    scale_xyz_mm = scale_zyx_um[::-1] / 1000
    origin_xyz_mm = origin_zyx_um[::-1] / 1000
    lines = [
        "NRRD0004",
        "# Complete NRRD file format specification at:",
        "# http://teem.sourceforge.net/nrrd/format.html",
        "type: unsigned short",
        "dimension: 3",
        "space: left-posterior-superior",
        f"sizes: {shape_zyx[2]} {shape_zyx[1]} {shape_zyx[0]}",
        "space directions: "
        f"({scale_xyz_mm[0]:.17g},0,0) "
        f"(0,{scale_xyz_mm[1]:.17g},0) "
        f"(0,0,{scale_xyz_mm[2]:.17g})",
        "kinds: domain domain domain",
        "endian: little",
        "encoding: gzip",
        "space origin: "
        f"({origin_xyz_mm[0]:.17g},{origin_xyz_mm[1]:.17g},{origin_xyz_mm[2]:.17g})",
        f"source:={_metadata_value(str(source))}",
        f"source_level:={level}",
    ]
    if channel is not None:
        lines.append(f"channel:={_metadata_value(channel)}")
    return ("\n".join(lines) + "\n\n").encode("ascii")


def _metadata_value(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("NRRD metadata values cannot contain newlines")
    return value


def _validate_nrrd(
    path: Path,
    *,
    sitk: Any,
    expected_shape_zyx: tuple[int, int, int],
    expected_scale_zyx_um: np.ndarray,
    expected_origin_zyx_um: np.ndarray,
) -> tuple[str, int]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetSize() != expected_shape_zyx[::-1]:
        raise ValueError(f"NRRD size mismatch: {reader.GetSize()} != {expected_shape_zyx[::-1]}")
    if not np.allclose(reader.GetSpacing(), expected_scale_zyx_um[::-1] / 1000, rtol=0, atol=1e-12):
        raise ValueError("NRRD spacing does not match the selected OME-Zarr level")
    if not np.allclose(reader.GetOrigin(), expected_origin_zyx_um[::-1] / 1000, rtol=0, atol=1e-12):
        raise ValueError("NRRD origin does not match the selected OME-Zarr level")
    if not np.allclose(
        np.asarray(reader.GetDirection()).reshape(3, 3), np.eye(3), rtol=0, atol=np.finfo(np.float64).eps
    ):
        raise ValueError("NRRD direction is not identity")

    digest = hashlib.sha256()
    decoded_bytes = 0
    with path.open("rb") as raw:
        while True:
            line = raw.readline()
            if not line:
                raise ValueError("NRRD header has no payload separator")
            if line in {b"\n", b"\r\n"}:
                break
        with gzip.GzipFile(fileobj=raw, mode="rb") as decoded:
            while block := decoded.read(8 * 1024 * 1024):
                digest.update(block)
                decoded_bytes += len(block)
    return digest.hexdigest(), decoded_bytes
