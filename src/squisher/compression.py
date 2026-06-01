from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path
from typing import Any, NotRequired, TypedDict
import xml.etree.ElementTree as ET

import numpy as np
import numpy.typing as npt
import tifffile
from tifffile import OmeXml, TiffFile, TiffWriter


JPEG_XR_KWARGS = {"photometric": "minisblack", "compression": 22610}
CZI_PROGRESS_INTERVAL = 100
OME_ORIGINAL_METADATA_NAMESPACE = "openmicroscopy.org/OriginalMetadata"
CZI_RAW_METADATA_NAMESPACE = "fishtools/czi/raw-metadata"


class CziTile(TypedDict):
    index: int
    scene: int
    mosaic_index: NotRequired[int]
    index_zyx: NotRequired[tuple[int, int, int]]
    x: int
    y: int
    width: int
    height: int
    position_x: float
    position_y: float


def compress_czi_to_ome_tiff(
    path: Path,
    *,
    level: float,
    tile_size: int = 512,
    maxworkers: int = 4,
    tile_workers: int = 1,
    resume: bool = False,
) -> bool:
    from aicspylibczi import CziFile

    if path.suffix.lower() != ".czi":
        raise ValueError(f"Expected a .czi input file, got {path}")
    if tile_size % 16 != 0:
        raise ValueError(f"TIFF tile size must be a multiple of 16, got {tile_size}")

    tiles = _czi_tiles(path)
    outputs = [_czi_tile_ome_tiff_path(path, tile, len(tiles)) for tile in tiles]
    existing = [out for out in outputs if out.exists()]
    if existing and not resume:
        raise FileExistsError(f"Refusing to overwrite existing OME-TIFF output(s): {existing}")

    reader = CziFile(path)
    dims = reader.get_dims_shape()[0]
    plane_count = (
        len(_czi_dim_indexes(dims, "T"))
        * len(_czi_dim_indexes(dims, "C"))
        * len(_czi_dim_indexes(dims, "Z"))
    )
    if resume:
        pending_tiles = []
        incomplete_existing = []
        for tile, out in zip(tiles, outputs, strict=True):
            if _is_complete_ome_tiff(out, plane_count=plane_count):
                continue
            pending_tiles.append(tile)
            if out.exists():
                incomplete_existing.append(out)
        if incomplete_existing:
            raise FileExistsError(f"Refusing to overwrite incomplete OME-TIFF output(s): {incomplete_existing}")
    else:
        pending_tiles = tiles

    if not pending_tiles:
        return True

    effective_tile_workers, effective_maxworkers = _effective_czi_workers(
        len(pending_tiles), tile_workers, maxworkers
    )
    if effective_tile_workers == 1:
        for tile in pending_tiles:
            _write_czi_tile_to_ome_tiff(
                reader,
                path,
                tile=tile,
                tile_count=len(tiles),
                level=level,
                tile_size=tile_size,
                maxworkers=effective_maxworkers,
                dims=dims,
            )
        return True

    del reader
    with ProcessPoolExecutor(effective_tile_workers, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                _write_czi_tile_to_ome_tiff_process,
                path,
                tile=tile,
                tile_count=len(tiles),
                level=level,
                tile_size=tile_size,
                maxworkers=effective_maxworkers,
            )
            for tile in pending_tiles
        ]
        for future in as_completed(futures):
            future.result()
    return True


def _write_czi_tile_to_ome_tiff(
    reader: Any,
    path: Path,
    *,
    tile: CziTile,
    tile_count: int,
    level: float,
    tile_size: int,
    maxworkers: int,
    dims: dict[str, tuple[int, int]],
) -> Path:
    t_indexes = _czi_dim_indexes(dims, "T")
    c_indexes = _czi_dim_indexes(dims, "C")
    z_indexes = _czi_dim_indexes(dims, "Z")
    out = _czi_tile_ome_tiff_path(path, tile, tile_count)
    plane_count = len(t_indexes) * len(c_indexes) * len(z_indexes)
    ome_shape = (len(t_indexes), len(c_indexes), len(z_indexes), tile["height"], tile["width"])

    print(
        f"Writing tile {tile['index'] + 1}/{tile_count} "
        f"at x={tile['x']} y={tile['y']} size={tile['width']}x{tile['height']} to {out.name}",
        flush=True,
    )

    plane_index = 0
    with TiffWriter(out, bigtiff=True, mode="x") as writer:
        for t in t_indexes:
            for c in c_indexes:
                for z in z_indexes:
                    plane = _as_grayscale_plane(reader.read_image(**_czi_read_kwargs(dims, tile, t, c, z))[0])
                    writer.write(
                        plane,
                        description=_first_plane_ome_xml(path, reader, tile, ome_shape, plane.dtype)
                        if plane_index == 0
                        else None,
                        metadata=None,
                        tile=(tile_size, tile_size),
                        **JPEG_XR_KWARGS,
                        compressionargs={"level": _compression_level(level)},
                        maxworkers=maxworkers,
                    )
                    plane_index += 1
                    if plane_index == 1 or plane_index % CZI_PROGRESS_INTERVAL == 0 or plane_index == plane_count:
                        print(
                            f"{out.name}: wrote plane {plane_index}/{plane_count} "
                            f"(T={t}, C={c}, Z={z})",
                            flush=True,
                        )
    print(f"Finished {out.name}", flush=True)
    return out


def _write_czi_tile_to_ome_tiff_process(
    path: Path,
    *,
    tile: CziTile,
    tile_count: int,
    level: float,
    tile_size: int,
    maxworkers: int,
) -> Path:
    from aicspylibczi import CziFile

    reader = CziFile(path)
    return _write_czi_tile_to_ome_tiff(
        reader,
        path,
        tile=tile,
        tile_count=tile_count,
        level=level,
        tile_size=tile_size,
        maxworkers=maxworkers,
        dims=reader.get_dims_shape()[0],
    )


def _first_plane_ome_xml(
    path: Path,
    reader: Any,
    tile: CziTile,
    shape: tuple[int, int, int, int, int],
    dtype: np.dtype[Any],
) -> bytes:
    subblock_metadata = _czi_subblock_metadata(reader, tile)
    return _czi_ome_xml(
        path,
        reader.meta,
        tile=tile,
        shape=shape,
        dtype=dtype,
        subblock_metadata=subblock_metadata,
    )


def _czi_read_kwargs(dims: dict[str, tuple[int, int]], tile: CziTile, t: int, c: int, z: int) -> dict[str, int]:
    read_kwargs = {}
    if "C" in dims:
        read_kwargs["C"] = c
    if "Z" in dims:
        read_kwargs["Z"] = z
    if "T" in dims:
        read_kwargs["T"] = t
    if "mosaic_index" in tile:
        read_kwargs["M"] = tile["mosaic_index"]
    elif "S" in dims:
        read_kwargs["S"] = tile["scene"]
    return read_kwargs


def _czi_ome_xml(
    path: Path,
    czi_metadata: Any,
    *,
    tile: CziTile,
    shape: tuple[int, int, int, int, int],
    dtype: np.dtype[Any],
    subblock_metadata: Any,
) -> bytes:
    plane_count = shape[0] * shape[1] * shape[2]
    metadata = _czi_ome_metadata(
        path,
        czi_metadata,
        tile=tile,
        plane_count=plane_count,
        subblock_metadata=subblock_metadata,
    )
    metadata.pop("axes", None)

    ome = OmeXml()
    ome.addimage(
        dtype=dtype,
        shape=shape,
        storedshape=(plane_count, 1, 1, shape[3], shape[4], 1),
        axes="TCZYX",
        **metadata,
    )
    _add_czi_raw_metadata_annotation(ome, czi_metadata, subblock_metadata)
    return ome.tostring().encode("utf-8")


def _czi_ome_metadata(
    path: Path,
    czi_metadata: Any,
    *,
    tile: CziTile,
    plane_count: int,
    subblock_metadata: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "axes": "TCZYX",
        "Name": f"{path.stem}:{tile['index']:03d}",
        "StructuredAnnotations": {
            "MapAnnotation": {
                "Namespace": OME_ORIGINAL_METADATA_NAMESPACE,
                "Value": _czi_original_metadata(path, czi_metadata, subblock_metadata, tile=tile),
            }
        },
        "Plane": {
            "PositionX": [float(tile["position_x"])] * plane_count,
            "PositionXUnit": ["reference frame"] * plane_count,
            "PositionY": [float(tile["position_y"])] * plane_count,
            "PositionYUnit": ["reference frame"] * plane_count,
        },
    }

    scale_x = _czi_scale_um(czi_metadata, "X")
    scale_y = _czi_scale_um(czi_metadata, "Y")
    scale_z = _czi_scale_um(czi_metadata, "Z")
    if scale_x is not None:
        metadata["PhysicalSizeX"] = scale_x
        metadata["PhysicalSizeXUnit"] = "µm"
        metadata["Plane"]["PositionX"] = [tile["position_x"] * scale_x] * plane_count
        metadata["Plane"]["PositionXUnit"] = ["µm"] * plane_count
    if scale_y is not None:
        metadata["PhysicalSizeY"] = scale_y
        metadata["PhysicalSizeYUnit"] = "µm"
        metadata["Plane"]["PositionY"] = [tile["position_y"] * scale_y] * plane_count
        metadata["Plane"]["PositionYUnit"] = ["µm"] * plane_count
    if scale_z is not None:
        metadata["PhysicalSizeZ"] = scale_z
        metadata["PhysicalSizeZUnit"] = "µm"
    return metadata


def _czi_original_metadata(path: Path, czi_metadata: Any, subblock_metadata: Any, *, tile: CziTile) -> dict[str, str]:
    values = {
        "squisher.source_file": path.name,
        "squisher.source_format": "CZI",
        "squisher.output_tile_index": str(tile["index"]),
        "czi.scene": str(tile["scene"]),
        "czi.tile_x": str(tile["x"]),
        "czi.tile_y": str(tile["y"]),
        "czi.tile_width": str(tile["width"]),
        "czi.tile_height": str(tile["height"]),
        "czi.global_metadata_xml": _metadata_text(czi_metadata),
        "czi.subblock_metadata_xml": _metadata_text(subblock_metadata),
    }
    if "mosaic_index" in tile:
        values["czi.mosaic_index"] = str(tile["mosaic_index"])
    if "index_zyx" in tile:
        values["squisher.tile_index_zyx"] = json.dumps(tile["index_zyx"])
    placement_path = path.with_name(f"{path.stem}_placement.json")
    if placement_path.exists():
        values["squisher.placement_json"] = placement_path.read_text()
    return values


def _add_czi_raw_metadata_annotation(ome: OmeXml, czi_metadata: Any, subblock_metadata: Any) -> None:
    annotation_id = f"Annotation:{len(ome.annotations)}"
    global_metadata = _xml_metadata_text(czi_metadata, "CZI global metadata")
    subblock_metadata_text = _xml_metadata_text(subblock_metadata, "CZI subblock metadata")
    ome.annotations.append(
        f'<XMLAnnotation ID="{annotation_id}" Namespace="{CZI_RAW_METADATA_NAMESPACE}">'
        "<Value>"
        '<CZIProvenanceMetadata xmlns="">'
        f"<GlobalMetadata>{global_metadata}</GlobalMetadata>"
        f"<SubblockMetadata>{subblock_metadata_text}</SubblockMetadata>"
        "</CZIProvenanceMetadata>"
        "</Value>"
        "</XMLAnnotation>"
    )
    ome.images[-1] = ome.images[-1].replace(
        "</Image>",
        f'<AnnotationRef ID="{annotation_id}"/></Image>',
        1,
    )


def _czi_tiles(path: Path) -> list[CziTile]:
    from aicspylibczi import CziFile

    czi_file = CziFile(path)
    dims = czi_file.get_dims_shape()[0]
    scene = _first_czi_dim(dims, "S")
    channel = _first_czi_dim(dims, "C")
    z = _first_czi_dim(dims, "Z")
    t = _first_czi_dim(dims, "T")
    placement_origins = _placement_origins(path)

    try:
        boxes = czi_file.get_all_mosaic_tile_bounding_boxes(C=channel, Z=z, T=t)
    except RuntimeError:
        boxes = {}

    if boxes:
        xs = sorted({int(box.x) for box in boxes.values()})
        ys = sorted({int(box.y) for box in boxes.values()})
        tiles = []
        for index, (info, box) in enumerate(sorted(boxes.items(), key=lambda item: (item[1].y, item[1].x))):
            index_zyx = (0, ys.index(int(box.y)), xs.index(int(box.x)))
            origin = placement_origins.get(index_zyx)
            tiles.append(
                CziTile(
                    index=index,
                    scene=scene,
                    mosaic_index=int(info.dimension_coordinates["M"]),
                    index_zyx=index_zyx,
                    y=int(box.y),
                    x=int(box.x),
                    height=int(box.h),
                    width=int(box.w),
                    position_y=float(origin[1]) if origin is not None else float(int(box.y) - min(ys)),
                    position_x=float(origin[2]) if origin is not None else float(int(box.x) - min(xs)),
                )
            )
        _validate_placement_origins(path, tiles, placement_origins)
        return tiles

    try:
        box = czi_file.get_tile_bounding_box(C=channel, Z=z, T=t)
        x, y, width, height = int(box.x), int(box.y), int(box.w), int(box.h)
    except RuntimeError:
        x = y = 0
        width = int(dims["X"][1] - dims["X"][0])
        height = int(dims["Y"][1] - dims["Y"][0])
    return [
        CziTile(
            index=0,
            scene=scene,
            y=y,
            x=x,
            height=height,
            width=width,
            position_y=0.0,
            position_x=0.0,
        )
    ]


def _validate_placement_origins(
    path: Path,
    tiles: list[CziTile],
    placement_origins: dict[tuple[int, int, int], tuple[float, float, float]],
) -> None:
    if not placement_origins:
        return
    tile_indices = {tile["index_zyx"] for tile in tiles}
    missing = sorted(tile_indices - set(placement_origins))
    extra = sorted(set(placement_origins) - tile_indices)
    if missing or extra:
        raise ValueError(
            f"Placement JSON for {path} does not match CZI tiles; missing origins: {missing}; extra origins: {extra}"
        )


def _placement_origins(path: Path) -> dict[tuple[int, int, int], tuple[float, float, float]]:
    placement_path = path.with_name(f"{path.stem}_placement.json")
    if not placement_path.exists():
        return {}
    payload = json.loads(placement_path.read_text())
    origins = {}
    for record in payload.get("placement", {}).get("origins", []):
        index = tuple(int(value) for value in record["index_zyx"])
        origin = tuple(float(value) for value in record["origin_zyx"])
        if len(index) != 3 or len(origin) != 3:
            raise ValueError(f"Invalid placement origin record in {placement_path}: {record}")
        origins[index] = origin
    return origins


def _as_grayscale_plane(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    data = np.squeeze(data)
    if data.ndim == 3 and data.shape[-1] == 1:
        return data[..., 0]
    if data.ndim == 2:
        return data
    raise ValueError(f"Expected a 2D grayscale CZI plane, found shape {data.shape}")


def _is_complete_ome_tiff(path: Path, *, plane_count: int) -> bool:
    try:
        with TiffFile(path) as tif:
            return len(tif.pages) == plane_count and tif.pages[0].compression == 22610 and tif.pages[0].is_tiled
    except (OSError, ValueError, KeyError, IndexError, tifffile.TiffFileError):
        return False


def _compression_level(level: float) -> float:
    return level / 100 if level > 1 else level


def _czi_dim_indexes(dims: dict[str, tuple[int, int]], dim: str) -> list[int]:
    start, end = dims.get(dim, (0, 1))
    return list(range(int(start), int(end)))


def _first_czi_dim(dims: dict[str, tuple[int, int]], dim: str) -> int:
    return int(dims.get(dim, (0, 1))[0])


def _czi_subblock_metadata(reader: Any, tile: CziTile) -> ET.Element:
    if "mosaic_index" in tile:
        return reader.read_subblock_metadata(unified_xml=True, M=tile["mosaic_index"])
    return reader.read_subblock_metadata(unified_xml=True, S=tile["scene"])


def _metadata_text(metadata: Any) -> str:
    if isinstance(metadata, ET.Element):
        return ET.tostring(metadata, encoding="unicode")
    if isinstance(metadata, str):
        return metadata
    return json.dumps(metadata, sort_keys=True)


def _xml_metadata_text(metadata: Any, name: str) -> str:
    if isinstance(metadata, ET.Element):
        return ET.tostring(metadata, encoding="unicode")
    if isinstance(metadata, str):
        ET.fromstring(metadata)
        return metadata
    raise TypeError(f"Expected XML metadata for {name}, found {type(metadata).__name__}")


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _czi_scale_um(metadata: Any, axis: str) -> float | None:
    if isinstance(metadata, ET.Element):
        root = metadata
    elif isinstance(metadata, str):
        root = ET.fromstring(metadata)
    else:
        raise TypeError(f"Expected XML CZI metadata, found {type(metadata).__name__}")

    for distance in root.iter():
        if _local_xml_name(distance.tag) != "Distance" or distance.attrib.get("Id") != axis:
            continue
        for child in distance.iter():
            if _local_xml_name(child.tag) == "Value" and child.text:
                value = float(child.text)
                return value * 1_000_000 if value > 0 else None
    return None


def _effective_czi_workers(tile_count: int, tile_workers: int, tiff_maxworkers: int) -> tuple[int, int]:
    cpu_count = os.cpu_count() or 1
    if tile_workers < 0:
        raise ValueError(f"CZI tile workers must be >= 0, got {tile_workers}")
    if tiff_maxworkers < 0:
        raise ValueError(f"TIFF maxworkers must be >= 0, got {tiff_maxworkers}")
    if tile_workers == 0 and tiff_maxworkers == 0:
        effective_tile_workers = min(tile_count, max(1, cpu_count // 8))
        return effective_tile_workers, max(1, min(4, cpu_count // effective_tile_workers))
    if tile_workers == 0:
        return max(1, min(tile_count, cpu_count // max(1, tiff_maxworkers))), tiff_maxworkers
    if tiff_maxworkers == 0:
        return min(tile_workers, tile_count), max(1, min(4, cpu_count // tile_workers))
    return min(tile_workers, tile_count), tiff_maxworkers


def _czi_tile_ome_tiff_path(path: Path, tile: CziTile, tile_count: int) -> Path:
    if tile_count == 1:
        return path.with_suffix(".ome.tif")
    return path.with_name(f"{path.stem}.{tile['index']:03d}.ome.tif")
