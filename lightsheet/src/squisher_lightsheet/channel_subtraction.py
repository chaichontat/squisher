"""Produce pre-fusion spillover-subtracted tiles.

Intended workflow for 514-minus-488 cleanup:

1. Register the physical 488/514/561/638 acquisition once at the track level,
   using the accepted 488 geometry as the target frame.
2. Measure the fine same-tile 514-to-488 offset at level 0 and compose that
   local shift onto the 488 rigid registration for the 514 channel.
3. Before BaSiC correction or multiview fusion, run this producer on the raw
   tiles: load target 514 and reference 488 from the same source tile, shift the
   reference on GPU into the target local coordinates, subtract the fitted
   spillover model, clip to non-negative values, crop the y/x border, and write
   single-channel corrected OME-Zarr tiles.
4. Emit a corrected position JSON, and when a registration JSON is supplied,
   emit an adapted registration JSON with the same registered transforms but the
   corrected tile paths and crop-adjusted stage transforms.
5. Build/use separate L/R intensity-sorted BaSiC profiles on these corrected
   cropped tiles, then fuse them as a normal single-channel acquisition. Fusion
   must not know about the reference channel or spillover model.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet import tile_input
from squisher_lightsheet.residual_correction import (
    ResidualCorrection,
    load_residual_corrections,
)
from squisher.gpu_retry import cuda_oom_retry
from squisher_lightsheet import ngff


DIMENSIONS = ("z", "y", "x")
DEFAULT_CHUNK_SHAPE_ZYX = (12, 480, 480)
DEFAULT_ZSTD_LEVEL = 3


@dataclass(frozen=True)
class SubtractedTileRecord:
    tile: str
    source_path: str
    output_path: str
    source_level: int
    source_shape_zyx: tuple[int, int, int]
    output_shape_zyx: tuple[int, int, int]
    translation_um: dict[str, float]
    scale_um: dict[str, float]
    side: str | None


@dataclass(frozen=True)
class ChannelSubtractionResult:
    position_output: Path
    summary_output: Path
    tile_count: int
    records: tuple[SubtractedTileRecord, ...]
    registration_output: Path | None = None


@dataclass(frozen=True)
class JoinedSubtractionResult:
    position_output: Path
    registration_output: Path
    summary_output: Path
    tile_count: int


@dataclass(frozen=True)
class RegisteredZModel:
    fixed_grid: Path
    fixed_origin_zyx_um: tuple[float, float, float]
    fixed_scale_zyx_um: tuple[float, float, float]
    z_origin: float
    z_range: tuple[float, float]
    alpha: tuple[float, float]
    beta: tuple[float, float]


def _crop_translation_um(
    tile: legacy.TileMetadata,
    *,
    crop_yx_px: int,
) -> dict[str, float]:
    stage_scale = legacy.tile_stage_scale(tile)
    return {
        "z": float(tile.translation["z"]),
        "y": float(tile.translation["y"] + crop_yx_px * stage_scale["y"]),
        "x": float(tile.translation["x"] + crop_yx_px * stage_scale["x"]),
    }


def _write_ome_zarr_czyx(
    output_path: Path,
    data: np.ndarray,
    *,
    translation_um: dict[str, float],
    scale_um: dict[str, float],
    channel_name: str,
    zstd_level: int,
    chunk_shape_zyx: tuple[int, int, int],
) -> None:
    """Write one CZYX scale with physical placement and a final completion marker."""
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (1, *(int(value) for value in data.shape))
    chunks = (1, *(min(size, chunk) for size, chunk in zip(data.shape, chunk_shape_zyx, strict=True)))
    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": output_path.name.removesuffix(".ome.zarr"),
                        "axes": [
                            {"name": "c", "type": "channel"},
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [
                                            1.0,
                                            abs(float(scale_um["z"])),
                                            abs(float(scale_um["y"])),
                                            abs(float(scale_um["x"])),
                                        ],
                                    },
                                    {
                                        "type": "translation",
                                        "translation": [
                                            0.0,
                                            float(translation_um["z"]),
                                            float(translation_um["y"]),
                                            float(translation_um["x"]),
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
            "omero": {"channels": [{"label": channel_name}]},
            "squisher_complete": False,
        }
    )
    output = root.create_array(
        "0",
        shape=shape,
        chunks=chunks,
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("c", "z", "y", "x"),
        serializer=BytesCodec(),
        compressors=[ZstdCodec(level=zstd_level)],
    )
    output[0] = data
    root.attrs["squisher_complete"] = True


@cuda_oom_retry("channel-subtraction slab")
def _subtract_slab_gpu(
    target_slab: np.ndarray,
    reference_halo: np.ndarray,
    *,
    halo_before: int,
    halo_after: int,
    reference_shift_zyx_px: tuple[float, float, float],
    alpha: float,
    beta: float,
    target_background: float,
    reference_background: float,
    crop_yx_px: int,
    output_dtype: np.dtype,
) -> np.ndarray:
    import cupy as cp
    from cupyx.scipy import ndimage as cndi

    target_gpu = cp.asarray(target_slab, dtype=cp.float32)
    reference_gpu = cp.asarray(reference_halo, dtype=cp.float32)
    axis_shift = tuple(-float(value) for value in reference_shift_zyx_px)
    if any(abs(value) > 1e-9 for value in axis_shift):
        reference_gpu = cndi.shift(
            reference_gpu,
            shift=axis_shift,
            order=1,
            mode="nearest",
            prefilter=False,
        )
    z_stop = reference_gpu.shape[0] - int(halo_after)
    reference_gpu = reference_gpu[int(halo_before) : z_stop]
    corrected = (
        (target_gpu - float(target_background))
        - float(alpha)
        * cp.maximum(
            reference_gpu - float(reference_background),
            0.0,
        )
        - float(beta)
    )
    cp.maximum(corrected, 0.0, out=corrected)
    if crop_yx_px > 0:
        corrected = corrected[:, crop_yx_px:-crop_yx_px, crop_yx_px:-crop_yx_px]
    if np.issubdtype(output_dtype, np.integer):
        cp.rint(corrected, out=corrected)
        cp.clip(corrected, 0, np.iinfo(output_dtype).max, out=corrected)
        corrected = corrected.astype(output_dtype, copy=False)
    else:
        corrected = corrected.astype(np.float32, copy=False)
    return cp.asnumpy(corrected)


def subtract_spillover_array_gpu(
    target: Any,
    reference: Any,
    *,
    reference_shift_zyx_px: tuple[float, float, float],
    alpha: float,
    beta: float,
    target_background: float,
    reference_background: float,
    crop_yx_px: int,
    z_chunk: int,
    output_dtype: np.dtype,
    target_correction: ResidualCorrection | None = None,
    reference_correction: ResidualCorrection | None = None,
) -> np.ndarray:
    source_shape = tuple(int(value) for value in target.shape)
    if len(source_shape) != 3:
        raise ValueError(f"Expected target ZYX data, got shape={source_shape}")
    if tuple(int(value) for value in reference.shape) != source_shape:
        raise ValueError(
            f"Target and reference shapes must match; got target={source_shape}, "
            f"reference={tuple(int(value) for value in reference.shape)}"
        )
    if crop_yx_px < 0:
        raise ValueError("crop_yx_px must be non-negative")
    if crop_yx_px * 2 >= min(source_shape[1], source_shape[2]):
        raise ValueError(f"crop_yx_px={crop_yx_px} removes the whole y/x tile from shape={source_shape}")
    if z_chunk <= 0:
        raise ValueError("z_chunk must be positive")

    output_shape = (
        source_shape[0],
        source_shape[1] - 2 * crop_yx_px,
        source_shape[2] - 2 * crop_yx_px,
    )
    output = np.empty(output_shape, dtype=output_dtype)
    z_halo = int(math.ceil(abs(float(reference_shift_zyx_px[0])))) + 1
    for z0 in range(0, source_shape[0], z_chunk):
        z1 = min(source_shape[0], z0 + z_chunk)
        halo_z0 = max(0, z0 - z_halo)
        halo_z1 = min(source_shape[0], z1 + z_halo)
        target_slab = np.asarray(target[z0:z1], dtype=output_dtype)
        reference_halo = np.asarray(reference[halo_z0:halo_z1], dtype=output_dtype)
        if target_correction is not None:
            target_slab = target_slab * target_correction.block(
                z_slice=slice(z0, z1),
                y_slice=slice(0, source_shape[1]),
                x_slice=slice(0, source_shape[2]),
            )
        if reference_correction is not None:
            reference_halo = reference_halo * reference_correction.block(
                z_slice=slice(halo_z0, halo_z1),
                y_slice=slice(0, source_shape[1]),
                x_slice=slice(0, source_shape[2]),
            )
        corrected = _subtract_slab_gpu(
            target_slab,
            reference_halo,
            halo_before=z0 - halo_z0,
            halo_after=halo_z1 - z1,
            reference_shift_zyx_px=reference_shift_zyx_px,
            alpha=alpha,
            beta=beta,
            target_background=target_background,
            reference_background=reference_background,
            crop_yx_px=crop_yx_px,
            output_dtype=output.dtype,
        )
        output[z0:z1] = corrected
    return output


def _output_tile_path(source_path: Path, output_tile_dir: Path) -> Path:
    identity = hashlib.sha256(str(source_path).encode()).hexdigest()[:8]
    return output_tile_dir / f"tile-{identity}.ome.zarr"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector_zyx(record: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict) and all(dim in value for dim in DIMENSIONS):
            result = np.asarray([value[dim] for dim in DIMENSIONS], dtype=np.float64)
            if result.shape == (3,) and np.all(np.isfinite(result)):
                return result
    raise ValueError(f"Tile {record.get('tile')!r} is missing finite z/y/x values for {keys}")


def _registration_affine(record: dict[str, Any]) -> np.ndarray:
    registered = record.get("registered_affine")
    matrix = registered.get("matrix") if isinstance(registered, dict) else registered
    affine = np.asarray(matrix, dtype=np.float64)
    while affine.ndim > 2:
        affine = affine[0]
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError(f"Tile {record.get('tile')!r} registered_affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"Tile {record.get('tile')!r} registered_affine must be homogeneous")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
        raise ValueError(f"Tile {record.get('tile')!r} registered_affine is singular")
    return affine


def _translation_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = values
    return matrix


def _scale_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.diag(values)
    return matrix


def _paired_pull_affine(
    *,
    target_record: dict[str, Any],
    reference_record: dict[str, Any],
    crop_yx_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map cropped target ZYX indices to reference indices and registered microns."""
    target_stage = _vector_zyx(target_record, "stage_translation_um", "translation_um")
    target_scale = _vector_zyx(target_record, "stage_scale_um", "scale_um", "spacing_um")
    reference_stage = _vector_zyx(reference_record, "stage_translation_um", "translation_um")
    reference_scale = _vector_zyx(reference_record, "stage_scale_um", "scale_um", "spacing_um")
    if np.any(target_scale <= 0) or np.any(reference_scale <= 0):
        raise ValueError("Paired subtraction requires positive target/reference stage scales")
    crop = np.asarray([0.0, float(crop_yx_px), float(crop_yx_px)], dtype=np.float64)
    output_to_target_stage = (
        _translation_matrix(target_stage) @ _scale_matrix(target_scale) @ _translation_matrix(crop)
    )
    output_to_registered = _registration_affine(target_record) @ output_to_target_stage
    registered_to_reference = (
        np.linalg.inv(_scale_matrix(reference_scale))
        @ _translation_matrix(-reference_stage)
        @ np.linalg.inv(_registration_affine(reference_record))
    )
    return registered_to_reference @ output_to_registered, output_to_registered


def _indexed_records(path: Path, *, side: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    rows = payload.get("tiles") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty tiles list")
    selected: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"{path} tile {index} must be an object")
        record = dict(raw)
        record_side = record.get("source_view", record.get("side"))
        if record_side != side:
            continue
        tile = record.get("tile")
        if not isinstance(tile, str) or not tile:
            raise ValueError(f"{path} tile {index} has no canonical tile identity")
        if tile in selected:
            raise ValueError(f"{path} contains duplicate tile identity {tile!r}")
        selected[tile] = record
    if not selected:
        raise ValueError(f"{path} contains no tiles for side {side!r}")
    return payload, selected


def _validate_position_registration_pair(
    *,
    position_path: Path,
    registration_path: Path,
    side: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _position_payload, positions = _indexed_records(position_path, side=side)
    registration_payload, registrations = _indexed_records(registration_path, side=side)
    if positions.keys() != registrations.keys():
        raise ValueError(
            f"Position/registration canonical tile IDs differ for side {side}: "
            f"position_only={sorted(positions.keys() - registrations.keys())}, "
            f"registration_only={sorted(registrations.keys() - positions.keys())}"
        )
    for tile, position in positions.items():
        registration = registrations[tile]
        position_path_value = Path(str(position.get("path", ""))).resolve()
        registration_path_value = Path(str(registration.get("path", ""))).resolve()
        if position_path_value != registration_path_value:
            raise ValueError(f"Position/registration source path differs for tile {tile!r}")
        position_translation = _vector_zyx(position, "translation_um")
        registration_translation = _vector_zyx(registration, "stage_translation_um")
        if not np.allclose(position_translation, registration_translation, rtol=1e-9, atol=1e-6):
            raise ValueError(f"Position/registration stage translation differs for tile {tile!r}")
    return registration_payload, registrations


def _load_registered_z_model(
    path: Path,
    *,
    side: str,
    target_registration: Path,
    reference_registration: Path,
) -> tuple[dict[str, Any], RegisteredZModel]:
    payload = json.loads(path.read_text())
    if payload.get("artifact_type") != "230Ptp.514-minus-488.constrained-registered-z-profile.v1":
        raise ValueError(f"Unsupported subtraction model schema: {path}")
    expected_hashes = {
        "target_registration_sha256": _sha256_file(target_registration),
        "reference_registration_sha256": _sha256_file(reference_registration),
    }
    for key, actual in expected_hashes.items():
        if payload.get(key) != actual:
            raise ValueError(f"Subtraction model {key} does not match {actual}")
    profiles = payload.get("sides")
    profile = profiles.get(side) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict) or profile.get("model") != "registered_fixed_z_linear_clamped":
        raise ValueError(f"Subtraction model has no registered-Z profile for side {side!r}")
    fixed_grid = Path(str(payload.get("fixed_fused", ""))).resolve()
    if not fixed_grid.is_dir():
        raise FileNotFoundError(f"Subtraction model fixed grid does not exist: {fixed_grid}")
    root = __import__("zarr").open_group(str(fixed_grid), mode="r")
    array = ngff.level_array(root, level=0, context=fixed_grid)
    axes = ngff.axes(root, array)
    names, scale, origin, has_scale, has_origin = ngff.scale_translation(root, dataset_index=0)
    if axes != "ZYX" or [name.lower() for name in names] != list("zyx") or not has_scale or not has_origin:
        raise ValueError(
            f"Subtraction model fixed grid must define a ZYX scale and translation: {fixed_grid}"
        )

    def pair(key: str) -> tuple[float, float]:
        values = np.asarray(profile.get(key), dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Subtraction model {side}.{key} must contain two finite values")
        return float(values[0]), float(values[1])

    z_range = pair("fixed_z_fit_range")
    z_origin = float(profile.get("fixed_z_origin"))
    if not np.isfinite(z_origin) or z_range[1] < z_range[0]:
        raise ValueError(f"Subtraction model {side} has an invalid registered-Z range")
    model = RegisteredZModel(
        fixed_grid=fixed_grid,
        fixed_origin_zyx_um=tuple(float(value) for value in origin),
        fixed_scale_zyx_um=tuple(float(value) for value in scale),
        z_origin=z_origin,
        z_range=z_range,
        alpha=pair("alpha_intercept_slope"),
        beta=pair("beta_intercept_slope"),
    )
    return payload, model


def _paired_tile_ids(
    target_records: dict[str, dict[str, Any]],
    reference_records: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Return target-order intersection plus explicit unmatched source IDs."""
    target_ids = list(target_records)
    reference_ids = set(reference_records)
    paired = [tile_id for tile_id in target_ids if tile_id in reference_ids]
    target_only = [tile_id for tile_id in target_ids if tile_id not in reference_ids]
    reference_only = [tile_id for tile_id in reference_records if tile_id not in target_records]
    if not paired:
        raise ValueError("Target/reference registrations have no paired canonical tile IDs")
    return paired, target_only, reference_only


def _canonical_pair_records(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index wavelength-specific source records by shared side/tile identity."""
    canonical: dict[str, dict[str, Any]] = {}
    for tile, record in records.items():
        source_tile = str(record.get("source_tile") or tile)
        parts = source_tile.split(".")
        side = record.get("source_view", record.get("side"))
        key = f"{side}.{parts[-3]}" if side in {"L", "R"} and len(parts) >= 3 and parts[-3].isdigit() else tile
        if key in canonical:
            raise ValueError(f"Duplicate canonical paired tile identity {key!r}")
        canonical[key] = record
    return canonical


def _spatial_shape(record: dict[str, Any]) -> tuple[int, int, int]:
    axes = str(record.get("axes", "")).upper()
    shape = tuple(int(value) for value in record.get("shape", []))
    if len(axes) != len(shape) or not set("ZYX") <= set(axes):
        raise ValueError(f"Tile {record.get('tile')!r} has invalid axes/shape: {axes!r}/{shape}")
    return tuple(shape[axes.index(dim)] for dim in "ZYX")


def _open_level0_channel(record: dict[str, Any], *, channel: int) -> tuple[Any, tuple[int, int, int]]:
    import zarr

    path = Path(str(record["path"])).resolve()
    root = zarr.open_group(str(path), mode="r")
    array = ngff.level_array(root, level=0, context=path)
    axes = ngff.axes(root, array)
    if axes != "CZYX":
        raise ValueError(f"Paired subtraction requires CZYX OME-Zarr sources, got {axes!r}: {path}")
    if not 0 <= channel < int(array.shape[0]):
        raise ValueError(f"Channel {channel} is outside {path} shape {array.shape}")
    shape = tuple(int(value) for value in array.shape[-3:])
    if _spatial_shape(record) != shape:
        raise ValueError(
            f"Registration shape does not match source {path}: {_spatial_shape(record)} != {shape}"
        )
    return array, shape


def _reference_bounds(
    pull: np.ndarray,
    *,
    output_start_zyx: tuple[int, int, int],
    output_shape_zyx: tuple[int, int, int],
    reference_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray(output_start_zyx) + np.asarray(output_shape_zyx) - 1
    corners = np.asarray(
        [[z, y, x, 1.0] for z in (output_start_zyx[0], ends[0]) for y in (0, ends[1]) for x in (0, ends[2])],
        dtype=np.float64,
    )
    coordinates = (pull @ corners.T).T[:, :3]
    lo = np.maximum(0, np.floor(coordinates.min(axis=0)).astype(int) - 2)
    hi = np.minimum(reference_shape_zyx, np.ceil(coordinates.max(axis=0)).astype(int) + 3)
    return lo, hi


@cuda_oom_retry("paired registered channel-subtraction slab")
def _subtract_registered_slab_gpu(
    target_slab: np.ndarray,
    reference_crop: np.ndarray,
    *,
    pull: np.ndarray,
    reference_lo_zyx: np.ndarray,
    output_z0: int,
    output_to_registered: np.ndarray,
    model: RegisteredZModel,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, float | int]]:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform

    target = cp.asarray(target_slab, dtype=cp.float32)
    reference = cp.asarray(reference_crop, dtype=cp.float32)
    slab_origin = np.asarray([output_z0, 0.0, 0.0], dtype=np.float64)
    offset = pull[:3, :3] @ slab_origin + pull[:3, 3] - reference_lo_zyx
    sampled_reference = gpu_affine_transform(
        reference,
        cp.asarray(pull[:3, :3], dtype=cp.float32),
        cp.asarray(offset, dtype=cp.float32),
        output_shape=target.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    support = gpu_affine_transform(
        cp.ones(reference.shape, dtype=cp.uint8),
        cp.asarray(pull[:3, :3], dtype=cp.float32),
        cp.asarray(offset, dtype=cp.float32),
        output_shape=target.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(cp.bool_)

    registered_z_row = output_to_registered[0]
    z = cp.arange(output_z0, output_z0 + target.shape[0], dtype=cp.float32)[:, None, None]
    y = cp.arange(target.shape[1], dtype=cp.float32)[None, :, None]
    x = cp.arange(target.shape[2], dtype=cp.float32)[None, None, :]
    registered_z_um = (
        float(registered_z_row[0]) * z
        + float(registered_z_row[1]) * y
        + float(registered_z_row[2]) * x
        + float(registered_z_row[3])
    )
    fixed_z = (registered_z_um - model.fixed_origin_zyx_um[0]) / model.fixed_scale_zyx_um[0]
    fixed_z = cp.clip(fixed_z, model.z_range[0], model.z_range[1]) - model.z_origin
    alpha = model.alpha[0] + model.alpha[1] * fixed_z
    beta = model.beta[0] + model.beta[1] * fixed_z
    signed = target - alpha * sampled_reference - beta
    signed = cp.where(support, signed, target)
    nonpositive = int(cp.asnumpy(cp.count_nonzero(support & (signed <= 0))))
    supported = int(cp.asnumpy(cp.count_nonzero(support)))
    signed_min = float(cp.asnumpy(cp.min(signed)))
    signed_max = float(cp.asnumpy(cp.max(signed)))
    cp.maximum(signed, 0.0, out=signed)
    if np.issubdtype(output_dtype, np.integer):
        cp.rint(signed, out=signed)
        cp.clip(signed, 0, np.iinfo(output_dtype).max, out=signed)
    output = cp.asnumpy(signed.astype(output_dtype, copy=False))
    return output, {
        "supported_voxels": supported,
        "nonpositive_voxels_before_clip": nonpositive,
        "signed_min": signed_min,
        "signed_max": signed_max,
    }


def _initialize_paired_output(
    path: Path,
    *,
    shape_zyx: tuple[int, int, int],
    dtype: np.dtype,
    translation_um: dict[str, float],
    scale_um: dict[str, float],
    channel_name: str,
    zstd_level: int,
    chunk_shape_zyx: tuple[int, int, int],
) -> tuple[Any, Any]:
    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": path.name.removesuffix(".ome.zarr"),
                        "axes": [
                            {"name": "c", "type": "channel"},
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, *[abs(float(scale_um[d])) for d in DIMENSIONS]],
                                    },
                                    {
                                        "type": "translation",
                                        "translation": [0.0, *[float(translation_um[d]) for d in DIMENSIONS]],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
            "omero": {"channels": [{"label": channel_name}]},
            "squisher_complete": False,
        }
    )
    chunks = (1, *(min(size, chunk) for size, chunk in zip(shape_zyx, chunk_shape_zyx, strict=True)))
    array = root.create_array(
        "0",
        shape=(1, *shape_zyx),
        chunks=chunks,
        dtype=dtype,
        fill_value=0,
        dimension_names=("c", "z", "y", "x"),
        serializer=BytesCodec(),
        compressors=[ZstdCodec(level=zstd_level)],
    )
    return root, array


def subtract_paired_channel_tiles(
    *,
    position_input: Path,
    registration_input: Path,
    reference_position_input: Path,
    reference_registration_input: Path,
    subtraction_model: Path,
    output_dir: Path,
    side: str,
    target_residual_correction: Path,
    reference_residual_correction: Path,
    target_channel: int = 0,
    reference_channel: int = 0,
    crop_yx_px: int = 20,
    z_chunk: int = 24,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    chunk_shape_zyx: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE_ZYX,
    limit_tiles: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> ChannelSubtractionResult:
    """Stream paired registered sources into cropped target-grid subtraction tiles.

    The correction models cover each complete side source set, while production
    subtraction runs on the exact target/reference tile intersection. This
    permits a disconnected or rejected tile on one acquisition to remain
    explicitly unpaired without inventing a reference transform.
    """
    if side not in {"L", "R"}:
        raise ValueError("side must be L or R")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite paired subtraction output: {output_dir}")
    if crop_yx_px < 0 or z_chunk < 1 or not 1 <= zstd_level <= 22:
        raise ValueError("Invalid crop, z_chunk, or zstd_level")
    if len(chunk_shape_zyx) != 3 or any(value <= 0 for value in chunk_shape_zyx):
        raise ValueError("chunk_shape_zyx must contain three positive values")
    if limit_tiles is not None and limit_tiles <= 0:
        raise ValueError("limit_tiles must be positive when provided")

    target_payload, target_records = _validate_position_registration_pair(
        position_path=position_input.resolve(),
        registration_path=registration_input.resolve(),
        side=side,
    )
    _reference_payload, reference_records = _validate_position_registration_pair(
        position_path=reference_position_input.resolve(),
        registration_path=reference_registration_input.resolve(),
        side=side,
    )
    target_records = _canonical_pair_records(target_records)
    reference_records = _canonical_pair_records(reference_records)
    tile_ids, unpaired_target, unpaired_reference = _paired_tile_ids(target_records, reference_records)
    model_payload, model = _load_registered_z_model(
        subtraction_model.resolve(),
        side=side,
        target_registration=registration_input.resolve(),
        reference_registration=reference_registration_input.resolve(),
    )
    target_source_ids = list(target_records)
    reference_source_ids = list(reference_records)
    target_sources = [Path(str(target_records[tile]["path"])).resolve() for tile in target_source_ids]
    reference_sources = [Path(str(reference_records[tile]["path"])).resolve() for tile in reference_source_ids]
    target_shapes = [_spatial_shape(target_records[tile]) for tile in target_source_ids]
    reference_shapes = [_spatial_shape(reference_records[tile]) for tile in reference_source_ids]
    if limit_tiles is not None:
        tile_ids = tile_ids[:limit_tiles]
    target_corrections = load_residual_corrections(
        target_residual_correction.resolve(),
        sources=target_sources,
        shapes_zyx=target_shapes,
        channel=target_channel,
    )
    reference_corrections = load_residual_corrections(
        reference_residual_correction.resolve(),
        sources=reference_sources,
        shapes_zyx=reference_shapes,
        channel=reference_channel,
    )

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    output_tile_dir = stage / "tiles"
    output_tile_dir.mkdir()
    output_records: list[dict[str, Any]] = []
    summary_records: list[SubtractedTileRecord] = []
    tile_summaries: list[dict[str, Any]] = []
    try:
        for tile_index, tile_id in enumerate(tile_ids):
            target_record = target_records[tile_id]
            reference_record = reference_records[tile_id]
            target_array, target_shape = _open_level0_channel(target_record, channel=target_channel)
            reference_array, reference_shape = _open_level0_channel(
                reference_record, channel=reference_channel
            )
            if crop_yx_px * 2 >= min(target_shape[1:]):
                raise ValueError(f"crop_yx_px={crop_yx_px} removes target tile {tile_id!r}")
            output_shape = (
                target_shape[0],
                target_shape[1] - 2 * crop_yx_px,
                target_shape[2] - 2 * crop_yx_px,
            )
            pull, output_to_registered = _paired_pull_affine(
                target_record=target_record,
                reference_record=reference_record,
                crop_yx_px=crop_yx_px,
            )
            target_source = Path(str(target_record["path"])).resolve()
            reference_source = Path(str(reference_record["path"])).resolve()
            output_path = _output_tile_path(target_source, output_tile_dir)
            published_output_path = output_dir / "tiles" / output_path.name
            translation_um = {
                dim: float(value)
                for dim, value in zip(
                    DIMENSIONS,
                    _vector_zyx(target_record, "stage_translation_um")
                    + _vector_zyx(target_record, "stage_scale_um") * np.asarray([0, crop_yx_px, crop_yx_px]),
                    strict=True,
                )
            }
            scale_values = _vector_zyx(target_record, "stage_scale_um")
            scale_um = {dim: float(value) for dim, value in zip(DIMENSIONS, scale_values, strict=True)}
            channels = target_record.get("channels")
            channel_name = (
                str(channels[target_channel]) if isinstance(channels, list) else str(target_channel)
            )
            root, output = _initialize_paired_output(
                output_path,
                shape_zyx=output_shape,
                dtype=np.dtype(target_array.dtype),
                translation_um=translation_um,
                scale_um=scale_um,
                channel_name=channel_name,
                zstd_level=zstd_level,
                chunk_shape_zyx=chunk_shape_zyx,
            )
            totals = {"supported_voxels": 0, "nonpositive_voxels_before_clip": 0}
            signed_min = math.inf
            signed_max = -math.inf
            for z0 in range(0, output_shape[0], z_chunk):
                z1 = min(output_shape[0], z0 + z_chunk)
                target_slice = (
                    slice(z0, z1),
                    slice(crop_yx_px, target_shape[1] - crop_yx_px),
                    slice(crop_yx_px, target_shape[2] - crop_yx_px),
                )
                target_slab = np.asarray(target_array[(target_channel, *target_slice)], dtype=np.float32)
                target_slab *= target_corrections[str(target_source)].block(
                    z_slice=target_slice[0], y_slice=target_slice[1], x_slice=target_slice[2]
                )
                reference_lo, reference_hi = _reference_bounds(
                    pull,
                    output_start_zyx=(z0, 0, 0),
                    output_shape_zyx=(z1 - z0, output_shape[1], output_shape[2]),
                    reference_shape_zyx=reference_shape,
                )
                if np.any(reference_hi <= reference_lo):
                    reference_crop = np.zeros((1, 1, 1), dtype=np.float32)
                    reference_lo = np.zeros(3, dtype=int)
                else:
                    reference_slices = tuple(
                        slice(int(lo), int(hi)) for lo, hi in zip(reference_lo, reference_hi, strict=True)
                    )
                    reference_crop = np.asarray(
                        reference_array[(reference_channel, *reference_slices)], dtype=np.float32
                    )
                    reference_crop *= reference_corrections[str(reference_source)].block(
                        z_slice=reference_slices[0], y_slice=reference_slices[1], x_slice=reference_slices[2]
                    )
                corrected, stats = _subtract_registered_slab_gpu(
                    target_slab,
                    reference_crop,
                    pull=pull,
                    reference_lo_zyx=reference_lo,
                    output_z0=z0,
                    output_to_registered=output_to_registered,
                    model=model,
                    output_dtype=np.dtype(target_array.dtype),
                )
                output[(0, slice(z0, z1), slice(None), slice(None))] = corrected
                totals["supported_voxels"] += int(stats["supported_voxels"])
                totals["nonpositive_voxels_before_clip"] += int(stats["nonpositive_voxels_before_clip"])
                signed_min = min(signed_min, float(stats["signed_min"]))
                signed_max = max(signed_max, float(stats["signed_max"]))
                if progress is not None:
                    progress(
                        f"Paired subtraction {side} tile {tile_index + 1}/{len(tile_ids)} {tile_id} z={z0}:{z1}"
                    )
            root.attrs["squisher_complete"] = True
            output_record = {
                "tile": output_path.name,
                "source_tile": tile_id,
                "source_path": str(target_source),
                "reference_path": str(reference_source),
                "side": side,
                "path": str(published_output_path),
                "axes": "CZYX",
                "shape": [1, *output_shape],
                "channels": [channel_name],
                "translation_um": translation_um,
                "scale_um": scale_um,
            }
            output_records.append(output_record)
            summary_records.append(
                SubtractedTileRecord(
                    tile=tile_id,
                    source_path=str(target_source),
                    output_path=str(published_output_path),
                    source_level=0,
                    source_shape_zyx=target_shape,
                    output_shape_zyx=output_shape,
                    translation_um=translation_um,
                    scale_um=scale_um,
                    side=side,
                )
            )
            supported = totals["supported_voxels"]
            tile_summaries.append(
                {
                    "tile": tile_id,
                    "target_source": str(target_source),
                    "reference_source": str(reference_source),
                    "output_path": str(published_output_path),
                    "supported_voxels": supported,
                    "nonpositive_voxels_before_clip": totals["nonpositive_voxels_before_clip"],
                    "nonpositive_fraction_before_clip": (
                        totals["nonpositive_voxels_before_clip"] / supported if supported else None
                    ),
                    "signed_min": signed_min,
                    "signed_max": signed_max,
                    "status": "complete",
                }
            )

        position_output = stage / f"{side}.positions.json"
        registration_output = stage / f"{side}.registration.json"
        position_payload = {
            "artifact_type": "lightsheet.position.v1",
            "units": "micrometer",
            "source_position_input": str(position_input.resolve()),
            "tiles": output_records,
        }
        position_output.write_text(json.dumps(position_payload, indent=2) + "\n")
        adapted_registration = dict(target_payload)
        adapted_registration["input_dir"] = str(output_dir / "tiles")
        adapted_registration["derived_from"] = {
            "target_registration": str(registration_input.resolve()),
            "reference_registration": str(reference_registration_input.resolve()),
            "subtraction_model": str(subtraction_model.resolve()),
        }
        adapted_tiles = []
        outputs_by_source_tile = {record["source_tile"]: record for record in output_records}
        for tile_id in tile_ids:
            source_record = target_records[tile_id]
            output_record = outputs_by_source_tile[tile_id]
            adapted = dict(source_record)
            adapted.update(
                {
                    "tile": output_record["tile"],
                    "path": output_record["path"],
                    "source_tile": tile_id,
                    "source_path": output_record["source_path"],
                    "reference_path": output_record["reference_path"],
                    "source_view": side,
                    "stage_translation_um": output_record["translation_um"],
                    "stage_scale_um": output_record["scale_um"],
                    "axes": output_record["axes"],
                    "shape": output_record["shape"],
                    "channels": output_record["channels"],
                }
            )
            adapted_tiles.append(adapted)
        adapted_registration["tiles"] = adapted_tiles
        registration_output.write_text(json.dumps(adapted_registration, indent=2) + "\n")
        summary_output = stage / f"{side}.summary.json"
        summary_payload = {
            "artifact_type": "squisher_lightsheet.paired_channel_subtraction.v1",
            "status": "complete",
            "side": side,
            "target_position": str(position_input.resolve()),
            "target_registration": str(registration_input.resolve()),
            "target_registration_sha256": _sha256_file(registration_input.resolve()),
            "reference_position": str(reference_position_input.resolve()),
            "reference_registration": str(reference_registration_input.resolve()),
            "reference_registration_sha256": _sha256_file(reference_registration_input.resolve()),
            "target_residual_correction": str(target_residual_correction.resolve()),
            "target_residual_correction_sha256": _sha256_file(target_residual_correction.resolve()),
            "reference_residual_correction": str(reference_residual_correction.resolve()),
            "reference_residual_correction_sha256": _sha256_file(reference_residual_correction.resolve()),
            "subtraction_model": str(subtraction_model.resolve()),
            "subtraction_model_sha256": _sha256_file(subtraction_model.resolve()),
            "fixed_grid": str(model.fixed_grid),
            "paired_tile_count": len(tile_ids),
            "unpaired_target_tiles": unpaired_target,
            "unpaired_reference_tiles": unpaired_reference,
            "crop_yx_px": crop_yx_px,
            "z_chunk": z_chunk,
            "chunk_shape_zyx": list(chunk_shape_zyx),
            "tile_count": len(tile_summaries),
            "tiles": tile_summaries,
            "model_profile": model_payload["sides"][side],
        }
        summary_output.write_text(json.dumps(summary_payload, indent=2) + "\n")
        final_dir = output_dir
        stage.replace(final_dir)
        return ChannelSubtractionResult(
            position_output=final_dir / position_output.name,
            summary_output=final_dir / summary_output.name,
            tile_count=len(summary_records),
            records=tuple(summary_records),
            registration_output=final_dir / registration_output.name,
        )
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=False)
        raise


def join_paired_subtraction_sides(
    *,
    side_output_dirs: list[Path],
    output_root: Path,
) -> JoinedSubtractionResult:
    """Join complete disjoint paired-subtraction sides without changing geometry."""
    output_root = output_root.resolve()
    resolved_sides = [path.resolve() for path in side_output_dirs]
    if len(resolved_sides) < 2 or len(set(resolved_sides)) != len(resolved_sides):
        raise ValueError("At least two unique side output directories are required")
    position_output = output_root / "subtracted.positions.json"
    registration_output = output_root / "subtracted.registration.json"
    summary_output = output_root / "channel_subtraction_summary.json"
    tile_output = output_root / "tiles"
    existing = [path for path in (position_output, registration_output, summary_output, tile_output) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite joined subtraction outputs: {existing}")

    sides: dict[str, dict[str, Any]] = {}
    canonical_ids: set[str] = set()
    output_names: set[str] = set()
    joined_positions: list[dict[str, Any]] = []
    joined_registrations: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    registration_template: dict[str, Any] | None = None
    stage = Path(tempfile.mkdtemp(prefix=".subtraction-join-", dir=output_root))
    stage_tiles = stage / "tiles"
    stage_tiles.mkdir()
    try:
        for side_dir in resolved_sides:
            summaries = sorted(side_dir.glob("*.summary.json"))
            positions = sorted(side_dir.glob("*.positions.json"))
            registrations = sorted(side_dir.glob("*.registration.json"))
            if len(summaries) != 1 or len(positions) != 1 or len(registrations) != 1:
                raise ValueError(
                    f"{side_dir} must contain one summary, position, and registration JSON"
                )
            summary = json.loads(summaries[0].read_text())
            side = summary.get("side")
            if side not in {"L", "R"} or side in sides or summary.get("status") != "complete":
                raise ValueError(f"Invalid or duplicate completed side summary: {summaries[0]}")
            position = json.loads(positions[0].read_text())
            registration = json.loads(registrations[0].read_text())
            position_rows = position.get("tiles")
            registration_rows = registration.get("tiles")
            if not isinstance(position_rows, list) or not isinstance(registration_rows, list):
                raise ValueError(f"Side {side} position/registration must contain tile lists")
            if len(position_rows) != len(registration_rows) or len(position_rows) != summary.get("tile_count"):
                raise ValueError(f"Side {side} tile counts disagree")
            registration_by_tile = {str(row.get("tile")): row for row in registration_rows}
            if len(registration_by_tile) != len(registration_rows):
                raise ValueError(f"Side {side} registration contains duplicate output tile names")
            for position_row in position_rows:
                output_name = str(position_row.get("tile"))
                registration_row = registration_by_tile.get(output_name)
                if registration_row is None:
                    raise ValueError(f"Side {side} position tile {output_name!r} has no registration")
                canonical_id = str(position_row.get("source_tile"))
                if not canonical_id or canonical_id == "None" or canonical_id in canonical_ids:
                    raise ValueError(f"Duplicate or invalid canonical tile identity {canonical_id!r}")
                if output_name in output_names:
                    raise ValueError(f"Duplicate derived output tile name {output_name!r}")
                if position_row.get("side") != side or registration_row.get("source_view") != side:
                    raise ValueError(f"Derived tile {canonical_id!r} side metadata disagrees")
                source_path = Path(str(position_row.get("path", ""))).resolve()
                if source_path != Path(str(registration_row.get("path", ""))).resolve():
                    raise ValueError(f"Derived tile {canonical_id!r} position/registration path differs")
                import zarr

                root = zarr.open_group(str(source_path), mode="r")
                if root.attrs.get("squisher_complete") is not True:
                    raise ValueError(f"Derived tile is incomplete: {source_path}")
                linked_path = output_root / "tiles" / output_name
                (stage_tiles / output_name).symlink_to(source_path, target_is_directory=True)
                joined_position = dict(position_row)
                joined_position["path"] = str(linked_path)
                joined_registration = dict(registration_row)
                joined_registration["path"] = str(linked_path)
                joined_positions.append(joined_position)
                joined_registrations.append(joined_registration)
                canonical_ids.add(canonical_id)
                output_names.add(output_name)
            sides[side] = summary
            provenance.append(
                {
                    "side": side,
                    "output_dir": str(side_dir),
                    "summary": str(summaries[0]),
                    "summary_sha256": _sha256_file(summaries[0]),
                    "position": str(positions[0]),
                    "position_sha256": _sha256_file(positions[0]),
                    "registration": str(registrations[0]),
                    "registration_sha256": _sha256_file(registrations[0]),
                    "tile_count": len(position_rows),
                }
            )
            if registration_template is None:
                registration_template = registration

        if set(sides) != {"L", "R"}:
            raise ValueError(f"Joined subtraction requires exactly L and R, got {sorted(sides)}")
        if registration_template is None:
            raise RuntimeError("No registration template was loaded")
        position_payload = {
            "artifact_type": "lightsheet.position.v1",
            "units": "micrometer",
            "derived_from": provenance,
            "tiles": joined_positions,
        }
        registration_payload = dict(registration_template)
        registration_payload.update(
            {
                "artifact_type": "squisher_lightsheet.joined_paired_channel_subtraction.v1",
                "input_dir": str(tile_output),
                "derived_from": provenance,
                "tiles": joined_registrations,
            }
        )
        summary_payload = {
            "artifact_type": "squisher_lightsheet.joined_paired_channel_subtraction_summary.v1",
            "status": "complete",
            "tile_count": len(joined_positions),
            "side_counts": {side: int(sides[side]["tile_count"]) for side in ("L", "R")},
            "canonical_tile_count": len(canonical_ids),
            "position_output": str(position_output),
            "registration_output": str(registration_output),
            "tile_dir": str(tile_output),
            "sides": provenance,
        }
        (stage / position_output.name).write_text(json.dumps(position_payload, indent=2) + "\n")
        (stage / registration_output.name).write_text(json.dumps(registration_payload, indent=2) + "\n")
        (stage / summary_output.name).write_text(json.dumps(summary_payload, indent=2) + "\n")
        stage_tiles.replace(tile_output)
        (stage / position_output.name).replace(position_output)
        (stage / registration_output.name).replace(registration_output)
        (stage / summary_output.name).replace(summary_output)
        stage.rmdir()
        return JoinedSubtractionResult(
            position_output=position_output,
            registration_output=registration_output,
            summary_output=summary_output,
            tile_count=len(joined_positions),
        )
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=False)
        raise


def _adapt_registration_payload(
    *,
    registration_input: Path,
    output_registration: Path,
    output_tile_dir: Path,
    output_records: list[dict[str, Any]],
) -> Path:
    payload = json.loads(registration_input.read_text())
    registration_records = payload.get("tiles")
    if not isinstance(registration_records, list) or not registration_records:
        raise ValueError(f"{registration_input} must contain a non-empty tiles list")

    outputs_by_source_name = {Path(str(record["source_path"])).name: record for record in output_records}
    outputs_by_source_tile = {str(record["source_tile"]): record for record in output_records}
    adapted_tiles: list[dict[str, Any]] = []
    for record in registration_records:
        if not isinstance(record, dict):
            raise ValueError(f"{registration_input} tile records must be objects")
        raw_tile = str(record.get("tile") or Path(str(record.get("path", ""))).name)
        output_record = outputs_by_source_tile.get(raw_tile) or outputs_by_source_name.get(raw_tile)
        if output_record is None:
            continue
        adapted = dict(record)
        adapted["tile"] = output_record["tile"]
        adapted["path"] = output_record["path"]
        adapted["source_tile"] = output_record["source_tile"]
        adapted["source_path"] = output_record["source_path"]
        adapted["source_view"] = output_record.get("side")
        adapted["stage_translation_um"] = output_record["translation_um"]
        adapted["stage_scale_um"] = output_record["scale_um"]
        adapted["axes"] = output_record["axes"]
        adapted["shape"] = output_record["shape"]
        adapted["channels"] = output_record["channels"]
        adapted_tiles.append(adapted)

    if len(adapted_tiles) != len(output_records):
        raise ValueError(
            f"{registration_input} matched {len(adapted_tiles)} of {len(output_records)} corrected tiles"
        )

    adapted_payload = dict(payload)
    adapted_payload["input_dir"] = str(output_tile_dir)
    adapted_payload["derived_from"] = {
        "registration_input": str(registration_input.resolve()),
        "previous": payload.get("derived_from"),
        "operation": "channel_subtraction_tile_path_adaptation",
    }
    adapted_payload["tiles"] = adapted_tiles
    output_registration.parent.mkdir(parents=True, exist_ok=True)
    output_registration.write_text(json.dumps(adapted_payload, indent=2) + "\n")
    return output_registration


def subtract_channel_tiles(
    *,
    position_input: Path,
    output_dir: Path,
    output_position: Path | None = None,
    registration_input: Path | None = None,
    output_registration: Path | None = None,
    target_channel: int,
    reference_channel: int,
    source_level: int,
    reference_shift_zyx_px: tuple[float, float, float],
    alpha: float,
    beta: float,
    target_background: float,
    reference_background: float,
    target_residual_correction: Path | None = None,
    reference_residual_correction: Path | None = None,
    crop_yx_px: int = 20,
    z_chunk: int = 64,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    chunk_shape_zyx: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE_ZYX,
    overwrite: bool = False,
    limit_tiles: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> ChannelSubtractionResult:
    if target_channel == reference_channel:
        raise ValueError("target_channel and reference_channel must differ")
    if source_level < 0:
        raise ValueError("source_level must be non-negative")
    if limit_tiles is not None and limit_tiles <= 0:
        raise ValueError("limit_tiles must be positive when provided")
    if not 1 <= zstd_level <= 22:
        raise ValueError("zstd_level must be between 1 and 22")
    if len(chunk_shape_zyx) != 3 or any(value <= 0 for value in chunk_shape_zyx):
        raise ValueError("chunk_shape_zyx must contain three positive values")

    output_dir = output_dir.resolve()
    output_tile_dir = output_dir / "tiles"
    output_position = (output_position or output_dir / "subtracted.positions.json").resolve()
    if registration_input is not None:
        output_registration = (output_registration or output_dir / "subtracted.registration.json").resolve()
    summary_output = output_dir / "channel_subtraction_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tile_dir.mkdir(parents=True, exist_ok=True)

    input_payload = json.loads(position_input.read_text())
    records = input_payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{position_input} must contain a non-empty tiles list")
    tiles = legacy.read_position_input_tiles(position_input.resolve())
    selected = list(zip(tiles, records, strict=True))
    if limit_tiles is not None:
        selected = selected[:limit_tiles]

    if (target_residual_correction is None) != (reference_residual_correction is None):
        raise ValueError("Target and reference residual corrections must be supplied together")
    target_corrections: dict[str, ResidualCorrection] = {}
    reference_corrections: dict[str, ResidualCorrection] = {}
    if target_residual_correction is not None and reference_residual_correction is not None:
        correction_sources = [tile.path for tile, _ in selected]
        correction_shapes = [tile_input.spatial_shape_zyx(tile.shape, tile.axes) for tile, _ in selected]
        target_corrections = load_residual_corrections(
            target_residual_correction,
            sources=correction_sources,
            shapes_zyx=correction_shapes,
            channel=target_channel,
        )
        reference_corrections = load_residual_corrections(
            reference_residual_correction,
            sources=correction_sources,
            shapes_zyx=correction_shapes,
            channel=reference_channel,
        )

    output_records: list[dict[str, Any]] = []
    summary_records: list[SubtractedTileRecord] = []
    for tile_index, (tile, source_record) in enumerate(selected):
        array, store = legacy.open_tile_array(tile, source_level=source_level)
        try:
            source_shape = tuple(int(value) for value in array.shape)
            source_tile = legacy.fusion_tile_for_source_array(tile, source_shape, source_level=source_level)
            source_shape_zyx = tile_input.spatial_shape_zyx(source_shape, source_tile.axes)
            target = tile_input.channel_view(array, source_tile.axes, target_channel, path=tile.path)
            reference = tile_input.channel_view(array, source_tile.axes, reference_channel, path=tile.path)
            resolved_tile_path = tile.path.resolve()
            output_path = _output_tile_path(resolved_tile_path, output_tile_dir)
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")
            translation_um = _crop_translation_um(source_tile, crop_yx_px=crop_yx_px)
            scale_um = {dim: float(legacy.tile_stage_scale(source_tile)[dim]) for dim in DIMENSIONS}
            if progress is not None:
                progress(
                    f"Subtracting tile {tile_index + 1}/{len(selected)} "
                    f"{tile.path.name} ch{target_channel}-alpha*ch{reference_channel} "
                    f"source_level={source_level} shape_zyx={source_shape_zyx}"
                )
            corrected = subtract_spillover_array_gpu(
                target,
                reference,
                reference_shift_zyx_px=reference_shift_zyx_px,
                alpha=alpha,
                beta=beta,
                target_background=target_background,
                reference_background=reference_background,
                crop_yx_px=crop_yx_px,
                z_chunk=z_chunk,
                output_dtype=np.dtype(target.dtype),
                target_correction=target_corrections.get(str(resolved_tile_path)),
                reference_correction=reference_corrections.get(str(resolved_tile_path)),
            )
            _write_ome_zarr_czyx(
                output_path,
                corrected,
                translation_um=translation_um,
                scale_um=scale_um,
                channel_name=source_tile.channels[target_channel],
                zstd_level=zstd_level,
                chunk_shape_zyx=chunk_shape_zyx,
            )
            source_tile_name = str(source_record.get("tile") or tile.path.name)
            side = source_record.get("side") if isinstance(source_record, dict) else None
            output_shape_zyx = tuple(int(value) for value in corrected.shape)
            output_record = {
                "tile": output_path.name,
                "source_tile": source_tile_name,
                "source_path": str(tile.path),
                "side": side,
                "path": str(output_path),
                "axes": "CZYX",
                "shape": [1, *output_shape_zyx],
                "channels": [source_tile.channels[target_channel]],
                "translation_um": translation_um,
                "scale_um": scale_um,
            }
            output_records.append(output_record)
            summary_records.append(
                SubtractedTileRecord(
                    tile=source_tile_name,
                    source_path=str(tile.path),
                    output_path=str(output_path),
                    source_level=source_level,
                    source_shape_zyx=source_shape_zyx,
                    output_shape_zyx=output_shape_zyx,
                    translation_um=translation_um,
                    scale_um=scale_um,
                    side=side if isinstance(side, str) else None,
                )
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()

    output_payload = {
        "artifact_type": "lightsheet.position.v1",
        "units": "micrometer",
        "source_position_input": str(position_input.resolve()),
        "subtraction": {
            "target_channel": int(target_channel),
            "reference_channel": int(reference_channel),
            "source_level": int(source_level),
            "reference_shift_zyx_px": [float(value) for value in reference_shift_zyx_px],
            "alpha": float(alpha),
            "beta": float(beta),
            "target_background": float(target_background),
            "reference_background": float(reference_background),
            "target_residual_correction": (
                None if target_residual_correction is None else str(target_residual_correction.resolve())
            ),
            "reference_residual_correction": (
                None
                if reference_residual_correction is None
                else str(reference_residual_correction.resolve())
            ),
            "crop_yx_px": int(crop_yx_px),
            "z_chunk": int(z_chunk),
            "zstd_level": int(zstd_level),
            "chunk_shape_zyx": [int(value) for value in chunk_shape_zyx],
        },
        "tiles": output_records,
    }
    output_position.parent.mkdir(parents=True, exist_ok=True)
    output_position.write_text(json.dumps(output_payload, indent=2) + "\n")
    written_registration = None
    if registration_input is not None:
        if output_registration is None:
            raise ValueError("output_registration was not resolved")
        written_registration = _adapt_registration_payload(
            registration_input=registration_input.resolve(),
            output_registration=output_registration,
            output_tile_dir=output_tile_dir,
            output_records=output_records,
        )

    summary_payload = {
        "position_output": str(output_position),
        "registration_output": None if written_registration is None else str(written_registration),
        "tile_count": len(summary_records),
        "tiles": [
            {
                "tile": record.tile,
                "source_path": record.source_path,
                "output_path": record.output_path,
                "source_level": record.source_level,
                "source_shape_zyx": list(record.source_shape_zyx),
                "output_shape_zyx": list(record.output_shape_zyx),
                "translation_um": record.translation_um,
                "scale_um": record.scale_um,
                "side": record.side,
            }
            for record in summary_records
        ],
    }
    summary_output.write_text(json.dumps(summary_payload, indent=2) + "\n")
    return ChannelSubtractionResult(
        position_output=output_position,
        summary_output=summary_output,
        tile_count=len(summary_records),
        records=tuple(summary_records),
        registration_output=written_registration,
    )
