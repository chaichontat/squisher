#!/usr/bin/env python
"""Add OME-Zarr pyramid levels to an existing NGFF 0.5 / Zarr v3 output."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import cupy as cp
from cucim.skimage.measure import block_reduce
import numpy as np
from squisher_lightsheet.pyramid import (
    chunk_count,
    chunk_slices,
    level_coordinate_transformations,
    pyramid_relative_factors,
)
import zarr
from zarr.codecs import BytesCodec, ZstdCodec


ZSTD_LEVEL = 0


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def root_metadata_path(path: Path) -> Path:
    metadata_path = path / "zarr.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{path} does not look like a Zarr v3 group")
    return metadata_path


def read_root_payload(path: Path) -> dict[str, Any]:
    return json.loads(root_metadata_path(path).read_text())


def read_array_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "zarr.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{path} does not look like a Zarr v3 array")
    return json.loads(metadata_path.read_text())


def write_root_payload(path: Path, payload: dict[str, Any]) -> None:
    root_metadata_path(path).write_text(json.dumps(payload, indent=2) + "\n")


def output_shape(shape: tuple[int, ...], dims: tuple[str, ...], factors: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(size) // int(factors[dim]) for size, dim in zip(shape, dims, strict=True))


def reduced_chunk(source: zarr.Array, selection: tuple[slice, ...], factors: tuple[int, ...]) -> np.ndarray:
    source_selection = tuple(
        slice(dim_slice.start * factor, dim_slice.stop * factor)
        for dim_slice, factor in zip(selection, factors, strict=True)
    )
    source_data = cp.asarray(source[source_selection])
    reduced = block_reduce(source_data, block_size=factors, func=cp.mean)
    dtype = np.dtype(source.dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        reduced = cp.rint(reduced)
        reduced = cp.clip(reduced, info.min, info.max)
    return cp.asnumpy(reduced.astype(dtype, copy=False))


def write_pyramid_level(
    root: Path,
    *,
    source_path: str,
    destination_path: str,
    dims: tuple[str, ...],
    factors: dict[str, int],
) -> tuple[int, ...]:
    source = zarr.open_array(str(root / source_path), mode="r", zarr_format=3)
    factor_tuple = tuple(int(factors[dim]) for dim in dims)
    shape = output_shape(tuple(int(size) for size in source.shape), dims, factors)
    chunks = tuple(min(int(chunk), size) for chunk, size in zip(source.chunks, shape, strict=True))
    destination = zarr.open(
        str(root / destination_path),
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=source.dtype,
        zarr_format=3,
        dimension_names=dims,
        codecs=[BytesCodec(endian="little"), ZstdCodec(level=ZSTD_LEVEL)],
    )

    total_chunks = chunk_count(shape, chunks)
    started = time.monotonic()
    log(
        f"Writing level {destination_path}: source={source_path}, "
        f"shape={shape}, chunks={chunks}, factors={factor_tuple}, chunks_total={total_chunks}"
    )
    for chunk_index, selection in enumerate(chunk_slices(shape, chunks), start=1):
        destination[selection] = reduced_chunk(source, selection, factor_tuple)
        if chunk_index <= 3 or chunk_index % 100 == 0 or chunk_index == total_chunks:
            elapsed = time.monotonic() - started
            log(
                f"Level {destination_path}: {chunk_index}/{total_chunks} chunks "
                f"({100 * chunk_index / total_chunks:.1f}%), elapsed={elapsed:.0f}s"
            )
    return shape


def seed_root_metadata_from_template(root: Path, template: Path) -> None:
    if root_metadata_path(template).resolve() == (root / "zarr.json").resolve():
        raise ValueError("--template must be a different OME-Zarr root")
    if (root / "zarr.json").exists():
        return

    target_scale0 = read_array_metadata(root / "0")
    template_scale0 = read_array_metadata(template / "0")
    for key in ("shape", "dimension_names"):
        if target_scale0.get(key) != template_scale0.get(key):
            raise ValueError(
                f"Template level 0 {key} does not match target: "
                f"{template_scale0.get(key)} != {target_scale0.get(key)}"
            )

    payload = read_root_payload(template)
    payload = copy.deepcopy(payload)
    ome = payload.get("attributes", {}).get("ome") or {}
    multiscales = ome.get("multiscales") or []
    if not multiscales:
        raise ValueError(f"{template} does not contain OME multiscales metadata")
    multiscale = copy.deepcopy(multiscales[0])
    datasets = multiscale.get("datasets") or []
    if not datasets:
        raise ValueError(f"{template} OME multiscales metadata does not list datasets")
    multiscale["datasets"] = [copy.deepcopy(datasets[0])]
    ome = copy.deepcopy(ome)
    ome["multiscales"] = [multiscale]
    payload.setdefault("attributes", {})["ome"] = ome
    (root / "zarr.json").write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Seeded root OME metadata for {root} from {template}")


def add_pyramid(root: Path, *, template: Path | None = None) -> None:
    if template is not None:
        seed_root_metadata_from_template(root, template)
    payload = read_root_payload(root)
    ome = payload.get("attributes", {}).get("ome") or {}
    multiscales = ome.get("multiscales") or []
    if not multiscales:
        raise ValueError(f"{root} does not contain OME multiscales metadata")
    multiscale = copy.deepcopy(multiscales[0])
    datasets = multiscale.get("datasets") or []
    if not datasets:
        raise ValueError(f"{root} OME multiscales metadata does not list datasets")
    if datasets != datasets[:1]:
        raise ValueError(f"{root} already has pyramid datasets: {[item.get('path') for item in datasets]}")

    scale0 = zarr.open_array(str(root / "0"), mode="r", zarr_format=3)
    dims = tuple(scale0.metadata.dimension_names or ())
    if not dims:
        dims = tuple(axis["name"] for axis in multiscale["axes"])
    axes = multiscale.get("axes") or [{"name": dim} for dim in dims]
    base_transforms = datasets[0].get("coordinateTransformations") or []

    new_datasets = [copy.deepcopy(datasets[0])]
    abs_factors = {dim: 1 for dim in dims}
    source_path = "0"
    current_shape = tuple(int(size) for size in scale0.shape)
    level_index = 1
    while True:
        factors = pyramid_relative_factors(current_shape, dims)
        if not any(factor > 1 for factor in factors.values()):
            break
        for dim in dims:
            abs_factors[dim] *= factors[dim]
        destination_path = str(level_index)
        current_shape = write_pyramid_level(
            root,
            source_path=source_path,
            destination_path=destination_path,
            dims=dims,
            factors=factors,
        )
        new_datasets.append(
            {
                "path": destination_path,
                "coordinateTransformations": level_coordinate_transformations(
                    base_transforms,
                    axes,
                    abs_factors,
                ),
            }
        )
        source_path = destination_path
        level_index += 1

    multiscale["datasets"] = new_datasets
    ome = copy.deepcopy(ome)
    ome["multiscales"] = [multiscale]
    payload = copy.deepcopy(payload)
    payload.setdefault("attributes", {})["ome"] = ome
    write_root_payload(root, payload)
    log(f"Updated {root} with {len(new_datasets)} pyramid level(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=Path,
        help="OME-Zarr root whose level-0 multiscales metadata should seed targets that lack root metadata.",
    )
    parser.add_argument("ome_zarr", type=Path, nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = args.template.resolve() if args.template is not None else None
    for path in args.ome_zarr:
        root = path.resolve()
        log(f"Adding pyramid levels to {root}")
        add_pyramid(root, template=template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
