from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from squisher.jpegxr_zarr import DEFAULT_JPEGXR_LEVEL
from squisher_lightsheet import seams
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.artifact_io import write_text_set_atomic
from squisher_lightsheet.channel_optimization import IDENTITY_AFFINE


ARTIFACT_TYPE = "lightsheet.yx_projection_registration.v1"


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _projection_metadata(tile: legacy.TileMetadata, channel: int) -> dict[str, Any]:
    return {
        "artifact_type": "lightsheet.yx_projection.v1",
        "source": _source_fingerprint(tile.path),
        "channel": int(channel),
        "operation": "maximum_z_projection",
    }


def _validate_projection(
    path: Path,
    *,
    tile: legacy.TileMetadata,
    channel: int,
    expected_shape_yx: tuple[int, int],
) -> None:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        description = json.loads(tif.pages[0].description)
        compression = int(tif.pages[0].compression)
        actual = (str(series.axes), tuple(int(value) for value in series.shape), np.dtype(series.dtype))
    expected = ("YX", expected_shape_yx, np.dtype(np.uint16))
    if actual != expected:
        raise ValueError(f"projection {path} metadata {actual} differs from {expected}")
    if compression != 22610:
        raise ValueError(f"projection {path} is not JPEG-XR compressed: {compression}")
    if description != _projection_metadata(tile, channel):
        raise ValueError(f"projection {path} belongs to different source data or settings")


def _write_projection(
    tile: legacy.TileMetadata,
    *,
    channel: int,
    output: Path,
    jpegxr_level: float,
) -> Path:
    import tifffile

    shape_zyx = legacy.tile_shape_zyx(tile)
    expected_shape_yx = (shape_zyx[1], shape_zyx[2])
    if output.exists():
        _validate_projection(
            output,
            tile=tile,
            channel=channel,
            expected_shape_yx=expected_shape_yx,
        )
        return output

    array, store, axes, shape = legacy.open_fusion_tile_array(tile, channel)
    try:
        projection = np.zeros(expected_shape_yx, dtype=np.uint16)
        z_count = shape_zyx[0]
        for z_index in range(z_count):
            if axes == "CZYX":
                plane = np.asarray(array[channel, z_index, :, :])
            elif axes == "ZCYX":
                plane = np.asarray(array[z_index, channel, :, :])
            elif axes == "ZYX":
                plane = np.asarray(array[z_index, :, :])
            else:
                raise ValueError(f"unsupported axes {axes!r} for {tile.path}")
            np.maximum(projection, plane, out=projection)
    finally:
        legacy.close_stores([store])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    tifffile.imwrite(
        temporary,
        projection,
        photometric="minisblack",
        compression=22610,
        compressionargs={"level": float(jpegxr_level)},
        description=json.dumps(_projection_metadata(tile, channel)),
        metadata=None,
    )
    if output.exists():
        raise FileExistsError(output)
    temporary.replace(output)
    _validate_projection(
        output,
        tile=tile,
        channel=channel,
        expected_shape_yx=expected_shape_yx,
    )
    return output


def build_yx_projections(
    tiles: list[legacy.TileMetadata],
    *,
    channel: int,
    output_dir: Path,
    workers: int = 4,
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Materialize resumable JPEG-XR Z-max planes without constructing a Dask graph."""
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{tile.path.stem}.mip.tif" for tile in tiles]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _write_projection,
                tile,
                channel=channel,
                output=output,
                jpegxr_level=jpegxr_level,
            ): index
            for index, (tile, output) in enumerate(zip(tiles, outputs, strict=True))
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if progress is not None and (completed == 1 or completed % 25 == 0 or completed == len(outputs)):
                progress(f"reddot projections {completed}/{len(outputs)}")
    return outputs


def _projection_tiles(
    tiles: list[legacy.TileMetadata], projections: list[Path]
) -> list[legacy.TileMetadata]:
    projected = []
    for tile, projection in zip(tiles, projections, strict=True):
        z_count, height, width = legacy.tile_shape_zyx(tile)
        del z_count
        projected.append(
            replace(
                tile,
                path=projection,
                shape=(1, height, width),
                axes="ZYX",
                channels=("reddot",),
            )
        )
    return projected


def _constraint_payload(constraint: seams.BoundaryConstraint) -> dict[str, Any]:
    payload = asdict(constraint)
    for key in ("fixed_slices", "moving_slices"):
        slices = getattr(constraint, key)
        payload[key] = (
            None
            if slices is None
            else [[int(value.start or 0), int(value.stop or 0)] for value in slices]
        )
    return payload


def _constraint_from_payload(payload: dict[str, Any]) -> seams.BoundaryConstraint:
    values = dict(payload)
    for key in ("fixed_slices", "moving_slices"):
        raw = values.get(key)
        values[key] = None if raw is None else tuple(slice(int(start), int(stop)) for start, stop in raw)
    for key in ("pair", "shift_zyx", "final_residual_zyx"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return seams.BoundaryConstraint(**values)


@lru_cache(maxsize=64)
def _read_projection(path: str) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(path), dtype=np.float32)


def _measure_constraint(
    *,
    spec: seams.BoundaryPatchSpec,
    projections: list[Path],
    settings: seams.RobustBoundarySettings,
) -> seams.BoundaryConstraint:
    fixed_index, moving_index = spec.pair
    fixed_plane = _read_projection(str(projections[fixed_index]))
    moving_plane = _read_projection(str(projections[moving_index]))
    fixed_patch = fixed_plane[spec.fixed_slices[1], spec.fixed_slices[2]][None, :, :]
    moving_patch = moving_plane[spec.moving_slices[1], spec.moving_slices[2]][None, :, :]
    fixed_stats = seams.center_z_content_stats(fixed_patch)
    moving_stats = seams.center_z_content_stats(moving_patch)
    reject_reason, _, _ = seams.center_z_content_prefilter_reason(fixed_patch, moving_patch, settings)
    if reject_reason is not None:
        return seams.boundary_constraint_from_prefilter_rejection(
            spec=spec,
            fixed_index=fixed_index,
            moving_index=moving_index,
            fixed_slices=spec.fixed_slices,
            moving_slices=spec.moving_slices,
            fixed_stats=fixed_stats,
            moving_stats=moving_stats,
            reject_reason=reject_reason,
            source_label="reddot_zmax",
        )

    (
        fixed_support,
        moving_support,
        (fixed_content, moving_content),
        corr_before,
        shift,
        peak,
        corr_after,
    ) = seams.evaluate_boundary_patch_gpu(fixed_patch, moving_patch, settings)
    shift = (0.0, float(shift[1]), float(shift[2]))
    gradient_before, gradient_after = seams.center_z_gradient_component_ncc_after_shift(
        fixed_patch,
        moving_patch,
        shift,
    )
    constraint = seams.boundary_constraint_from_evaluation(
        spec=spec,
        fixed_index=fixed_index,
        moving_index=moving_index,
        fixed_slices=spec.fixed_slices,
        moving_slices=spec.moving_slices,
        fixed_support=fixed_support,
        moving_support=moving_support,
        fixed_content=fixed_content,
        moving_content=moving_content,
        corr_before=corr_before,
        shift=shift,
        peak=peak,
        corr_after=corr_after,
        gradient_before=gradient_before,
        gradient_after=gradient_after,
        settings=settings,
        source_label="reddot_zmax",
    )
    return replace(
        constraint,
        fixed_center_z_p99=fixed_stats["p99"],
        moving_center_z_p99=moving_stats["p99"],
        fixed_center_z_std=fixed_stats["std"],
        moving_center_z_std=moving_stats["std"],
    )


def _write_identity_registration(
    *,
    optimized_position: dict[str, Any],
    tiles: list[legacy.TileMetadata],
    registration_output: Path,
    position_output: Path,
    provenance: dict[str, Any],
) -> None:
    canonical_position = deepcopy(optimized_position)
    canonical_position["registration_run"] = provenance
    registration_tiles = []
    for record, tile in zip(canonical_position["tiles"], tiles, strict=True):
        record["tile"] = tile.path.name
        record["path"] = str(tile.path)
        record["translation_um"] = dict(tile.translation)
        registration_tiles.append(
            {
                "tile": tile.path.name,
                "path": str(tile.path),
                "shape": list(tile.shape),
                "axes": tile.axes,
                "spacing_um": dict(tile.spacing),
                "channels": list(tile.channels),
                "tracks": [asdict(track) for track in tile.tracks],
                "source_view": tile.source_view,
                "stage_translation_um": dict(tile.translation),
                "stage_scale_um": legacy.tile_stage_scale(tile),
                "registered_affine": deepcopy(IDENTITY_AFFINE),
            }
        )
    registration = {
        "input_dir": str(tiles[0].path.parent),
        "metadata_transform_key": "stage_metadata",
        "registered_transform_key": "registered_affine",
        "spacing_um": dict(tiles[0].spacing),
        "metrics": {
            "artifact_type": ARTIFACT_TYPE,
            "registered_affine_note": "Identity affine; YX corrections are baked into stage translations.",
            "registration_run": provenance,
        },
        "tiles": registration_tiles,
    }
    write_text_set_atomic(
        {
            position_output: json.dumps(canonical_position, indent=2) + "\n",
            registration_output: json.dumps(registration, indent=2) + "\n",
        }
    )


def register_yx_projection(
    *,
    position_input: Path,
    input_dir: Path,
    output_dir: Path,
    position_output: Path,
    registration_output: Path,
    channel: int,
    projection_workers: int = 4,
    jpegxr_level: float = DEFAULT_JPEGXR_LEVEL,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Register TIFF tiles from direct CuPy phase correlation of reddot Z projections."""
    if registration_output.exists() and position_output.exists():
        payload = json.loads(registration_output.read_text())
        if payload.get("metrics", {}).get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError(f"{registration_output} is not a direct YX registration artifact")
        return registration_output
    if registration_output.exists() or position_output.exists():
        raise FileExistsError("registration and position outputs must either both exist or both be absent")

    tiles = legacy.read_position_input_tiles(position_input, input_dir=input_dir)
    if not tiles:
        raise ValueError(f"no tiles found in {position_input}")
    output_dir.mkdir(parents=True, exist_ok=True)
    projections = build_yx_projections(
        tiles,
        channel=channel,
        output_dir=output_dir / "mip",
        workers=projection_workers,
        jpegxr_level=jpegxr_level,
        progress=progress,
    )
    projected_tiles = _projection_tiles(tiles, projections)
    pairs = legacy.axis_aligned_registration_pairs(projected_tiles)
    height, width = legacy.tile_shape_zyx(projected_tiles[0])[1:]
    settings = replace(
        seams.RobustBoundarySettings(),
        patch_shape_zyx=(1, height, width),
        max_patches_per_edge=1,
        min_inlier_patches_per_edge=1,
        overlap_margin_zyx=(0, 128, 128),
        boundary_edge_workers=1,
    )
    specs = legacy.sample_boundary_patches(projected_tiles, None, pairs, settings)
    if len(specs) != len(pairs):
        raise ValueError(f"expected one YX projection patch per pair, got {len(specs)} for {len(pairs)}")

    run_record = {
        "artifact_type": ARTIFACT_TYPE,
        "position_input": str(position_input.resolve()),
        "position_input_sha256": hashlib.sha256(position_input.read_bytes()).hexdigest(),
        "input_dir": str(input_dir.resolve()),
        "channel": int(channel),
        "pair_count": len(pairs),
        "settings": asdict(settings),
    }
    run_record = json.loads(json.dumps(run_record))
    run_record_path = output_dir / "yx.run.json"
    if run_record_path.exists():
        if json.loads(run_record_path.read_text()) != run_record:
            raise ValueError(f"{run_record_path} belongs to different inputs or settings")
    else:
        temporary = run_record_path.with_name(f".{run_record_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(run_record, indent=2) + "\n")
        temporary.replace(run_record_path)

    edge_dir = output_dir / "edges"
    edge_dir.mkdir(parents=True, exist_ok=True)
    constraints = []
    for index, spec in enumerate(specs):
        checkpoint = edge_dir / f"{index:06d}.json"
        if checkpoint.exists():
            payload = json.loads(checkpoint.read_text())
            if tuple(payload["pair"]) != spec.pair:
                raise ValueError(f"checkpoint {checkpoint} belongs to pair {payload['pair']}, not {spec.pair}")
            constraint = _constraint_from_payload(payload)
        else:
            constraint = _measure_constraint(spec=spec, projections=projections, settings=settings)
            temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(_constraint_payload(constraint), indent=2) + "\n")
            temporary.replace(checkpoint)
        constraints.append(constraint)
        completed = index + 1
        if progress is not None and (completed == 1 or completed % 25 == 0 or completed == len(specs)):
            progress(f"direct YX edges {completed}/{len(specs)}")

    corrections, annotated, anchor_tile = legacy.solve_tile_corrections_with_residual_rejection(
        projected_tiles,
        constraints,
        settings,
        fixed_axes={"z"},
        residual_reject_axes={"y", "x"},
    )
    connected = legacy.anchor_connected_tiles(len(tiles), annotated, anchor_tile)
    if len(connected) != len(tiles):
        raise ValueError(f"YX registration graph is disconnected: {len(connected)}/{len(tiles)} tiles")
    if any(not math.isclose(float(correction[0]), 0.0, abs_tol=1e-12) for correction in corrections):
        raise RuntimeError("YX-only registration produced a nonzero Z correction")

    optimized = json.loads(position_input.read_text())
    for record, tile, correction in zip(optimized["tiles"], tiles, corrections, strict=True):
        translation = dict(tile.translation)
        translation["y"] += float(correction[1]) * float(tile.spacing["y"])
        translation["x"] += float(correction[2]) * float(tile.spacing["x"])
        record["translation_um"] = translation
    optimized["derived_by"] = "squisher_lightsheet.yx_projection.register_yx_projection"

    accepted = sum(item.accepted for item in annotated)
    provenance = {
        "artifact_type": ARTIFACT_TYPE,
        "position_input": str(position_input.resolve()),
        "position_input_sha256": hashlib.sha256(position_input.read_bytes()).hexdigest(),
        "input_dir": str(input_dir.resolve()),
        "channel": int(channel),
        "translation_axes": "yx",
        "projection": "maximum_z",
        "pair_count": len(pairs),
        "accepted_constraint_count": accepted,
        "connected_tile_count": len(connected),
        "tile_count": len(tiles),
        "anchor_tile": int(anchor_tile),
        "projection_codec": "jpegxr",
        "jpegxr_level": float(jpegxr_level),
        "settings": asdict(settings),
    }
    diagnostics = {
        **provenance,
        "corrections_zyx_px": [list(values) for values in corrections],
        "constraints": [_constraint_payload(item) for item in annotated],
    }
    diagnostics = json.loads(json.dumps(diagnostics))
    diagnostics_output = output_dir / "yx.diagnostics.json"
    if diagnostics_output.exists():
        existing = json.loads(diagnostics_output.read_text())
        if existing != diagnostics:
            raise FileExistsError(f"existing diagnostics differ: {diagnostics_output}")
    else:
        diagnostics_output.write_text(json.dumps(diagnostics, indent=2) + "\n")
    provenance["diagnostics"] = str(diagnostics_output.resolve())

    optimized_tiles = []
    for tile, record in zip(tiles, optimized["tiles"], strict=True):
        optimized_tiles.append(replace(tile, translation=dict(record["translation_um"])))
    _write_identity_registration(
        optimized_position=optimized,
        tiles=optimized_tiles,
        registration_output=registration_output,
        position_output=position_output,
        provenance=provenance,
    )
    return registration_output


__all__ = ["build_yx_projections", "register_yx_projection"]
