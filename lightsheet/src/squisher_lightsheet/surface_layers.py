"""Assign cortex cells to VZ, SVZ, or CP from reviewed boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from squisher_lightsheet.artifact_io import sha256_file
from squisher_lightsheet.surface_geometry import load_surface


def assign_surface_layers(*, coordinates_dir: Path, surface_path: Path, boundaries_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    coordinates_path = coordinates_dir / "cells.parquet"
    coordinates_metadata_path = coordinates_dir / "metadata.json"
    coordinates_metadata = json.loads(coordinates_metadata_path.read_text())
    if sha256_file(coordinates_path) != coordinates_metadata["output_sha256"]:
        raise ValueError("coordinate table does not match its metadata SHA-256")
    surface_hash = sha256_file(surface_path)
    if coordinates_metadata["surface_sha256"] != surface_hash:
        raise ValueError("coordinates and supplied surface identities differ")

    boundary_metadata_path = boundaries_path.parent / "metadata.json"
    boundary_metadata = json.loads(boundary_metadata_path.read_text())
    if sha256_file(boundaries_path) != boundary_metadata["outputs"]["boundaries.npz"]:
        raise ValueError("boundary artifact does not match its metadata SHA-256")
    if boundary_metadata["surface_sha256"] != surface_hash:
        raise ValueError("boundaries and supplied surface identities differ")
    surface = load_surface(surface_path)
    with np.load(boundaries_path) as boundaries:
        inner = np.asarray(boundaries["inner_um"], dtype=np.float64)
        outer = np.asarray(boundaries["outer_um"], dtype=np.float64)
        estimated_inner = np.asarray(boundaries["estimated_inner"], dtype=bool)
        estimated_outer = np.asarray(boundaries["estimated_outer"], dtype=bool)
        wall_support = np.asarray(boundaries["wall_support"], dtype=bool)
    expected_shape = (len(surface.vertices_world_xyz_um),)
    if any(values.shape != expected_shape for values in (inner, outer, estimated_inner, estimated_outer, wall_support)):
        raise ValueError("boundary vertex arrays do not match the supplied surface")

    stage = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(f"staging path already exists: {stage}")
    stage.mkdir(parents=True)
    output_path = stage / "cells.parquet"
    writer: pq.ParquetWriter | None = None
    counts = {"VZ": 0, "SVZ": 0, "CP": 0}
    boundary_defined_rows = 0
    started = time.perf_counter()
    try:
        columns = [
            "cell", "dataset", "roi", "label", "u", "v", "r_vz_um", "face_id",
            "barycentric_0", "barycentric_1", "barycentric_2", "on_surface_boundary",
            "on_tissue_side",
        ]
        for batch in pq.ParquetFile(coordinates_path).iter_batches(batch_size=100_000, columns=columns):
            face_index = batch.column(batch.schema.get_field_index("face_id")).to_numpy()
            faces = surface.faces[face_index]
            barycentric = np.column_stack(
                [batch.column(batch.schema.get_field_index(f"barycentric_{i}")).to_numpy() for i in range(3)]
            )
            inner_cutoff = np.einsum("ij,ij->i", inner[faces], barycentric)
            outer_cutoff = np.einsum("ij,ij->i", outer[faces], barycentric)
            face_supported = np.all(
                estimated_inner[faces] & estimated_outer[faces] & wall_support[faces], axis=1
            )
            radial = batch.column(batch.schema.get_field_index("r_vz_um")).to_numpy(zero_copy_only=False)
            tissue_side = batch.column(batch.schema.get_field_index("on_tissue_side")).to_numpy(
                zero_copy_only=False
            )
            boundary_defined = (
                face_supported
                & tissue_side
                & np.isfinite(radial)
                & np.isfinite(inner_cutoff)
                & np.isfinite(outer_cutoff)
                & (outer_cutoff >= inner_cutoff)
            )
            is_vz = boundary_defined & (radial <= inner_cutoff)
            is_svz = boundary_defined & ~is_vz & (radial <= outer_cutoff)
            layer_code = np.zeros(len(batch), dtype=np.int8)
            layer_code[is_vz] = 1
            layer_code[is_svz] = 2
            layer = pa.DictionaryArray.from_arrays(pa.array(layer_code), pa.array(["CP", "VZ", "SVZ"]))
            result = pa.table(
                {
                    "cell": batch.column(0),
                    "dataset": batch.column(1),
                    "roi": batch.column(2),
                    "label": batch.column(3),
                    "u": batch.column(4),
                    "v": batch.column(5),
                    "r_vz_um": batch.column(6),
                    "inner_boundary_um": pa.array(inner_cutoff, mask=~boundary_defined),
                    "outer_boundary_um": pa.array(outer_cutoff, mask=~boundary_defined),
                    "boundary_defined": boundary_defined,
                    "inside_vz_svz": is_vz | is_svz,
                    "layer": layer,
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, result.schema, compression="zstd")
            writer.write_table(result)
            boundary_defined_rows += int(np.count_nonzero(boundary_defined))
            counts["VZ"] += int(np.count_nonzero(is_vz))
            counts["SVZ"] += int(np.count_nonzero(is_svz))
            counts["CP"] += int(len(batch) - np.count_nonzero(is_vz) - np.count_nonzero(is_svz))
        if writer is None:
            raise ValueError("coordinate table contains no rows")
        writer.close()
        writer = None
        rows = pq.ParquetFile(output_path).metadata.num_rows
        if rows != coordinates_metadata["rows"] or sum(counts.values()) != rows:
            raise RuntimeError("layer output row count differs from its coordinates")
        metadata = {
            "artifact_type": "squisher_lightsheet.surface_layers.v1",
            "coordinates": str(coordinates_path.resolve()),
            "coordinates_sha256": sha256_file(coordinates_path),
            "coordinates_metadata_sha256": sha256_file(coordinates_metadata_path),
            "surface": str(surface_path.resolve()),
            "surface_sha256": surface_hash,
            "boundaries": str(boundaries_path.resolve()),
            "boundaries_sha256": sha256_file(boundaries_path),
            "boundaries_metadata_sha256": sha256_file(boundary_metadata_path),
            "rule": "VZ if r_vz <= inner; SVZ if inner < r_vz <= outer; otherwise CP",
            "inner": boundary_metadata["inner"],
            "outer": boundary_metadata["outer"],
            "validity": "finite non-rim tissue-side projection onto a face whose three vertices have both estimated boundaries",
            "rows": rows,
            "boundary_defined_rows": boundary_defined_rows,
            "counts": counts,
            "output": "cells.parquet",
            "output_sha256": sha256_file(output_path),
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.replace(output_dir)
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_dir / "metadata.json"

