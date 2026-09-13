"""Fit post-deconvolution overlap corrections and apply them at fusion reads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import numpy as np
import tifffile
import zarr
from loguru import logger
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import threshold_otsu

from squisher_deconv.residual_field import residual_block, validate_coefficients
from squisher_deconv.tile_gains import write_tile_gains
from squisher_lightsheet.ngff import level_array, scale_translation
from squisher_lightsheet.post_basic import (
    _Grid,
    _SampledPlane,
    _SampledTile,
    _artifact,
    _boundary_metrics,
    _corrected_tiles,
    _fixed_grid,
    _mosaic,
    _plan_record,
    _pull_grid,
    _registration_records,
    _sha256,
    _validate_z_models,
    _write_qc,
    fit_residual_model,
)
from squisher_lightsheet.post_basic_qc import write_post_basic_qc


DEFAULT_Z_PERCENTILES = tuple(index * 2.5 for index in range(1, 40))


@dataclass(frozen=True)
class ResidualCorrection:
    """Compact source correction evaluated only for requested original-coordinate slabs."""

    coefficient: np.ndarray
    shape_zyx: tuple[int, int, int]
    multiplier: float
    fingerprint: str

    def block(self, *, z_slice: slice, y_slice: slice, x_slice: slice) -> np.ndarray:
        return (
            residual_block(
                self.coefficient,
                z_slice=z_slice,
                shape_zyx=self.shape_zyx,
                y_slice=y_slice,
                x_slice=x_slice,
            )
            * self.multiplier
        )


def _source_identity(path: Path) -> str:
    candidates = (
        path / "zarr.json",
        path / ".zattrs",
        path / "0/zarr.json",
        path / "0/.zarray",
        path / "squisher.complete.json",
    )
    selected = [candidate for candidate in candidates if candidate.is_file()]
    if not any(candidate.parent == path for candidate in selected) or not any(
        candidate.parent == path / "0" for candidate in selected
    ):
        raise ValueError(f"Residual fitting requires root and level-0 Zarr metadata: {path}")
    digest = hashlib.sha256()
    for candidate in selected:
        digest.update(str(candidate.relative_to(path)).encode())
        digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _sample_tiles(
    records: list[dict],
    grid: _Grid,
    channel: int,
    level: int,
    workers: int,
) -> list[_SampledTile]:
    """Pull registered locations using the source's actual NGFF level transforms."""

    def sample(record: dict) -> _SampledTile | None:
        path = Path(record["path"]).resolve()
        group = zarr.open_group(path, mode="r")
        native = level_array(group, level=0)
        array = level_array(group, level=level)
        axes, scale0, origin0, _, _ = scale_translation(group, dataset_index=0)
        axes_n, scale_n, origin_n, _, _ = scale_translation(group, dataset_index=level)
        if axes != ["c", "z", "y", "x"] or axes_n != axes:
            raise ValueError(f"Residual fitting requires CZYX deconvolved sources: {path}")
        if not 0 <= channel < native.shape[0]:
            raise ValueError(f"Channel {channel} is outside {path}")
        if record.get("materialized_source_start_zyx") is not None:
            raise ValueError("Residual fitting requires original deconvolved tiles, not materialized crops")
        shape = tuple(native.shape[-3:])
        if record.get("axes") != "CZYX" or tuple(record["shape"][-3:]) != shape:
            raise ValueError(f"Registration shape/axes do not match source: {path}")
        plan = _plan_record(record=record, source=path, source_shape=shape, grid=grid)
        if plan is None:
            return None
        camera = _pull_grid(plan, grid)
        coordinates = (
            camera * np.asarray(scale0[1:])[:, None, None]
            + (np.asarray(origin0[1:]) - origin_n[1:])[:, None, None]
        ) / np.asarray(scale_n[1:])[:, None, None]
        valid = np.all(
            (camera >= plan.source_start[:, None, None]) & (camera <= (plan.source_stop - 1)[:, None, None]),
            axis=0,
        ) & np.all(
            (coordinates >= 0) & (coordinates <= (np.asarray(array.shape[-3:]) - 1)[:, None, None]),
            axis=0,
        )
        if not np.any(valid):
            return None
        lo = np.maximum(0, np.floor(coordinates[:, valid].min(axis=1)).astype(int))
        hi = np.minimum(array.shape[-3:], np.ceil(coordinates[:, valid].max(axis=1)).astype(int) + 1)
        crop = np.asarray(
            array[(channel,) + tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))], dtype=np.float32
        )
        data = map_coordinates(
            crop, coordinates - lo[:, None, None], order=1, prefilter=False, mode="constant"
        )
        data[~valid] = 0
        yx = np.moveaxis(camera[1:] / (np.asarray(shape[1:]) - 1)[:, None, None], 0, -1).astype(np.float32)
        camera_z = (camera[0] / max(shape[0] - 1, 1)).astype(np.float32)
        return _SampledTile(path, plan.lo, plan.hi, data, yx, valid, camera_z)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [tile for tile in pool.map(sample, records) if tile is not None]


def _overlap_samples(tiles: list[_SampledTile], cutoff: float) -> list[dict]:
    """Use all registered overlap support, including CL/CR interior overlaps."""
    smooth = []
    for tile in tiles:
        support = gaussian_filter(tile.valid.astype(np.float32), 1)
        values = gaussian_filter(tile.data, 1)
        smooth.append(np.divide(values, support, out=np.zeros_like(values), where=support > 0))
    rows = []
    for i, a in enumerate(tiles):
        for j in range(i + 1, len(tiles)):
            b = tiles[j]
            lo, hi = np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi)
            if np.any(hi - lo < 8):
                continue
            sa = tuple(
                slice(int(low - origin), int(high - origin), 4)
                for low, high, origin in zip(lo, hi, a.lo, strict=True)
            )
            sb = tuple(
                slice(int(low - origin), int(high - origin), 4)
                for low, high, origin in zip(lo, hi, b.lo, strict=True)
            )
            av, bv = smooth[i][sa], smooth[j][sb]
            mask = (
                a.valid[sa]
                & b.valid[sb]
                & (a.data[sa] > cutoff)
                & (b.data[sb] > cutoff)
                & (av > 0)
                & (bv > 0)
            )
            if np.count_nonzero(mask) < 12:
                continue
            rows.append(
                {
                    "first": i,
                    "second": j,
                    "pair_id": f"{a.source}:{b.source}",
                    "ya": a.yx[sa][mask],
                    "yb": b.yx[sb][mask],
                    "za": None if a.camera_z is None else a.camera_z[sa][mask],
                    "zb": None if b.camera_z is None else b.camera_z[sb][mask],
                    "target": np.log(bv[mask] / av[mask]),
                }
            )
    return rows


def fit_residual_correction(
    *,
    registration: Path,
    fixed_fused: Path,
    output_dir: Path,
    channel: int = 0,
    source_level: int = 2,
    stride: int = 4,
    xy_degree: int = 1,
    z_degree: int = 1,
    field_penalty: float | None = None,
    source_field_penalty: float | None = None,
    z_percentiles: Sequence[float] = DEFAULT_Z_PERCENTILES,
    workers: int = 8,
    seed: int = 0,
) -> Path:
    """Fit, validate, and atomically publish one post-deconvolution correction."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite residual correction output: {output_dir}")
    if channel < 0 or source_level < 0 or stride < 1 or workers < 1 or xy_degree not in (1, 2):
        raise ValueError("Invalid residual fitting channel, source level, stride, workers, or XY degree")
    if not 0 <= z_degree <= 3:
        raise ValueError("Z degree must be between 0 and 3")
    if field_penalty is not None and (not np.isfinite(field_penalty) or field_penalty <= 0):
        raise ValueError("Field penalty must be positive and finite")
    if source_field_penalty is not None and (
        not np.isfinite(source_field_penalty) or source_field_penalty <= 0
    ):
        raise ValueError("Source-field penalty must be positive and finite")
    if len(z_percentiles) < z_degree + 2 or any(
        not np.isfinite(percentile) or not 0 <= percentile <= 100 for percentile in z_percentiles
    ):
        raise ValueError("Z percentiles must be finite in [0, 100] and include at least Z degree + 2 planes")

    registration = registration.resolve()
    fixed_fused = fixed_fused.resolve()
    output_dir = output_dir.resolve()
    records = _registration_records(registration)
    sources = [Path(record["path"]).resolve() for record in records]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Residual fitting requires unique registered source paths")
    identities = {str(source): _source_identity(source) for source in sources}
    source_index = {source: index for index, source in enumerate(sources)}
    base = _fixed_grid(fixed_fused, fixed_z=None, stride=stride)
    sorted_percentiles = sorted(z_percentiles)
    sampled_z = [int(np.floor((base.shape[0] - 1) * percentile / 100)) for percentile in sorted_percentiles]
    if len(set(sampled_z)) != len(sampled_z):
        raise ValueError("Z percentiles must select distinct planes")
    representative_z = min(sampled_z, key=lambda z: (abs(z - base.z), z))

    rows: list[dict[str, object]] = []
    planes: list[_SampledPlane] = []
    for z in sampled_z:
        grid = replace(base, z=z)
        tiles = _sample_tiles(records, grid, channel, source_level, workers)
        if len(tiles) < 2:
            raise ValueError(f"Residual fitting sampled fewer than two sources at fixed Z {z}")
        before, owner = _mosaic(grid.output_shape, tiles)
        positive = before[before > 0]
        if positive.size < 2:
            raise ValueError(f"Insufficient source signal at fixed Z {z}")
        cutoff = float(np.expm1(threshold_otsu(np.log1p(positive))))
        plane_rows = _overlap_samples(tiles, cutoff)
        if not plane_rows:
            raise ValueError(f"No eligible source overlaps at fixed Z {z}")
        for row in plane_rows:
            row["first"] = source_index[tiles[int(row["first"])].source]
            row["second"] = source_index[tiles[int(row["second"])].source]
            row["z"] = z
        rows.extend(plane_rows)
        planes.append(
            _SampledPlane(
                grid=grid,
                tiles=tiles,
                before=before,
                owner=owner,
                cutoff=cutoff,
                sampling={
                    "records_intersecting": len(tiles),
                    "sampled_sources": [str(tile.source) for tile in tiles],
                },
            )
        )
        logger.info("Residual correction sampled Z {}: {} overlap pairs", z, len(plane_rows))

    coefficient, gains, evaluation = fit_residual_model(
        rows=rows,
        sources=sources,
        seed=seed,
        fit_tile_gains=True,
        fit_source_fields=True,
        z_degree=z_degree,
        xy_degree=xy_degree,
        field_penalty=field_penalty,
        source_field_penalty=source_field_penalty,
    )
    coefficient_2d = coefficient.reshape(z_degree + 1, 5)
    source_fields = evaluation["source_fields"]
    if [Path(row["source"]) for row in source_fields] != sources:
        raise RuntimeError("Fitted source fields do not align with registered sources")
    source_coefficients = np.asarray([row["coefficient"] for row in source_fields], dtype=np.float64).reshape(
        len(sources), z_degree + 1, 5
    )
    support_by_source = {str(row["source"]): int(row["training_pairs"]) for row in evaluation["tile_gains"]}
    combined_coefficients = source_coefficients + coefficient_2d
    log_field_bounds = np.asarray(
        [
            sum(
                float(np.abs(source_coefficient[degree]).sum()) / (1 + degree**2)
                for degree in range(z_degree + 1)
            )
            for source_coefficient in combined_coefficients
        ]
    )
    global_scale = 1 / max(1.0, float(np.max(gains * np.exp(log_field_bounds))))
    correction = {
        "schema_version": 3,
        "kind": "squisher.residual-correction",
        "channel": channel,
        "coordinate": "normalized-original-raw-zyx",
        "basis": "cosine-xy2-z-scaled",
        "coefficient": coefficient_2d.tolist(),
        "global_scale": global_scale,
        "sources": [
            {
                "path": str(source),
                "source_sha256": identities[str(source)],
                "shape_zyx": [int(value) for value in record["shape"][-3:]],
                "gain": float(gain),
                "coefficient": source_coefficient.tolist(),
                "training_pairs": support_by_source[str(source)],
            }
            for source, record, gain, source_coefficient in zip(
                sources, records, gains, source_coefficients, strict=True
            )
        ],
        "fit": {
            "registration": str(registration),
            "registration_sha256": _sha256(registration),
            "fixed_fused": str(fixed_fused),
            "source_level": source_level,
            "stride": stride,
            "xy_degree": xy_degree,
            "z_degree": z_degree,
            "field_penalty": field_penalty,
            "source_field_penalty": source_field_penalty,
            "seed": seed,
            "planes": [
                {
                    "z": plane.grid.z,
                    "cutoff": plane.cutoff,
                    "pairs": sum(row["z"] == plane.grid.z for row in rows),
                }
                for plane in planes
            ],
            "evaluation": evaluation,
            "final_training": "all eligible source pairs",
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".fit-residual-", dir=output_dir.parent))
    try:
        correction_path = stage / "correction.json"
        correction_path.write_text(json.dumps(correction, indent=2) + "\n")
        preview = next(plane for plane in planes if plane.grid.z == representative_z)
        gain_by_source = dict(zip(sources, gains, strict=True))
        corrected = _corrected_tiles(
            preview.tiles,
            coefficient,
            np.asarray([gain_by_source[tile.source] for tile in preview.tiles]) * global_scale,
            np.asarray([source_coefficients[source_index[tile.source]] for tile in preview.tiles]).reshape(
                len(preview.tiles), -1
            ),
        )
        after, owner = _mosaic(preview.grid.output_shape, corrected)
        if not np.array_equal(preview.owner, owner):
            raise RuntimeError("Residual QC owner map changed during correction")
        artifacts = _write_qc(
            stage,
            root=stage,
            channel=channel,
            before=preview.before,
            after=after,
            spacing_yx=np.abs(base.scale[1:]) * stride,
            labels=("Deconvolved", "Residual corrected"),
        )
        owner_path = stage / f"owner-ch{channel}.tif"
        tifffile.imwrite(owner_path, owner, compression="zstd")
        artifacts["owner"] = _artifact(owner_path, stage)
        source_shape = tuple(int(value) for value in records[0]["shape"][-3:])
        mask = residual_block(
            coefficient_2d,
            z_slice=slice(source_shape[0] // 2, source_shape[0] // 2 + 1),
            shape_zyx=source_shape,
        )[0]
        mask_path = stage / f"mask-ch{channel}.tif"
        tifffile.imwrite(mask_path, mask, compression="zstd")
        artifacts["mask"] = _artifact(mask_path, stage)

        gain_matrix = np.ones((len(sources), channel + 1), dtype=np.float32)
        gain_matrix[:, channel] = gains
        gains_path = write_tile_gains(
            stage / "tile-gains.json",
            sources=sources,
            gains=gain_matrix,
        )
        result = dict(evaluation)
        result.update(
            {
                "label": f"channel {channel}",
                "channel": channel,
                "registration": str(registration),
                "registration_sha256": _sha256(registration),
                "cutoff": preview.cutoff,
                "coefficient": coefficient.tolist(),
                "z_degree": z_degree,
                "field_range": [
                    float(np.exp(-log_field_bounds.max())),
                    float(np.exp(log_field_bounds.max())),
                ],
                "global_scale": global_scale,
                "mask_raw_z_fraction": 0.5,
                "tile_gains_applied": True,
                "sampling": preview.sampling,
                "excluded_sources": [],
                "excluded_source_gain_policy": "not applicable",
                "artifacts": artifacts,
                "boundaries": _boundary_metrics(preview.before, after, owner, preview.cutoff),
                "planes": [],
            }
        )
        result["z_validation"] = _validate_z_models(
            rows,
            sources=sources,
            reference_z=representative_z,
            seed=seed,
            fit_tile_gains=True,
            fit_source_fields=True,
            z_degree=z_degree,
            xy_degree=xy_degree,
            field_penalty=field_penalty,
            source_field_penalty=source_field_penalty,
        )
        for plane in planes:
            if plane is preview:
                plane_artifacts = artifacts
            else:
                plane_dir = stage / "planes" / f"z{plane.grid.z}"
                plane_dir.mkdir(parents=True)
                plane_corrected = _corrected_tiles(
                    plane.tiles,
                    coefficient,
                    np.asarray([gain_by_source[tile.source] for tile in plane.tiles]) * global_scale,
                    np.asarray(
                        [source_coefficients[source_index[tile.source]] for tile in plane.tiles]
                    ).reshape(len(plane.tiles), -1),
                )
                plane_after, _ = _mosaic(plane.grid.output_shape, plane_corrected)
                plane_artifacts = _write_qc(
                    plane_dir,
                    root=stage,
                    channel=channel,
                    before=plane.before,
                    after=plane_after,
                    spacing_yx=np.abs(base.scale[1:]) * stride,
                    labels=("Deconvolved", "Residual corrected"),
                )
            result["planes"].append(
                {
                    "z": plane.grid.z,
                    "z_um": float(base.origin[0] + base.scale[0] * plane.grid.z),
                    "cutoff": plane.cutoff,
                    "sampling": plane.sampling,
                    "artifacts": plane_artifacts,
                }
            )

        input_snapshot = {
            "fixed_grid": _source_identity(fixed_fused),
            "registration": {"path": str(registration), "sha256": _sha256(registration)},
            "sources": identities,
        }
        manifest = {
            "schema_version": 3,
            "artifact_type": "squisher_lightsheet.residual_calibration.v3",
            "status": "complete",
            "input_stage": "deconvolved",
            "fixed_fused": str(fixed_fused),
            "fixed_z": representative_z,
            "sampled_z": sampled_z,
            "z_percentiles": sorted_percentiles,
            "z_degree": z_degree,
            "xy_degree": xy_degree,
            "field_penalty": field_penalty,
            "source_field_penalty": source_field_penalty,
            "fixed_z_um": float(base.origin[0] + base.scale[0] * representative_z),
            "fixed_spacing_um": base.scale.tolist(),
            "fixed_origin_um": base.origin.tolist(),
            "stride": stride,
            "output_shape_yx": list(base.output_shape),
            "seed": seed,
            "inputs": input_snapshot,
            "channel_results": {str(channel): result},
            "tile_gains": _artifact(gains_path, stage),
            "correction": _artifact(correction_path, stage),
        }
        write_post_basic_qc(
            stage,
            manifest=manifest,
        )
        manifest["qc"] = {
            "artifacts": {path.name: _artifact(path, stage) for path in sorted((stage / "qc").iterdir())},
        }
        if identities != {str(source): _source_identity(source) for source in sources}:
            raise RuntimeError("Residual fitting sources changed while the workflow was running")
        if _sha256(registration) != input_snapshot["registration"]["sha256"]:
            raise RuntimeError("Residual fitting registration changed while the workflow was running")
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        stage.replace(output_dir)
        logger.info("Residual correction output complete: {}", output_dir)
        return output_dir / "manifest.json"
    except BaseException:
        shutil.rmtree(stage, ignore_errors=False)
        raise


def load_residual_corrections(
    path: Path,
    *,
    sources: Sequence[Path],
    shapes_zyx: Sequence[tuple[int, int, int]],
    channel: int,
) -> dict[str, ResidualCorrection]:
    """Require exact source identity and return compact, lazily evaluated corrections."""
    payload = json.loads(path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 3
        or payload.get("kind") != "squisher.residual-correction"
    ):
        raise ValueError(f"Invalid residual correction schema: {path}")
    if (
        payload.get("channel") != channel
        or payload.get("coordinate") != "normalized-original-raw-zyx"
        or payload.get("basis") != "cosine-xy2-z-scaled"
    ):
        raise ValueError(f"Residual correction channel or coordinate mismatch: {path}")
    coefficient = validate_coefficients(payload["coefficient"])
    scale = payload["global_scale"]
    if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Residual correction global scale must be positive and finite")
    rows = payload["sources"]
    source_rows: dict[str, dict[str, object]] = {}
    for row in rows:
        source = Path(row["path"])
        resolved_source = str(source.resolve())
        if not source.is_absolute() or resolved_source in source_rows:
            raise ValueError("Residual correction sources must be unique absolute paths")
        gain = row["gain"]
        if type(gain) not in (int, float) or not np.isfinite(gain) or gain <= 0:
            raise ValueError(f"Invalid residual gain for {source}")
        source_coefficient = validate_coefficients(row.get("coefficient"))
        if source_coefficient.shape != coefficient.shape:
            raise ValueError(
                f"Residual source coefficient shape must match the shared coefficient shape: {source}"
            )
        row["coefficient"] = source_coefficient
        source_rows[resolved_source] = row
    expected = [str(p.resolve()) for p in sources]
    if (
        len(set(expected)) != len(expected)
        or set(source_rows) != set(expected)
        or len(shapes_zyx) != len(expected)
    ):
        raise ValueError("Residual correction must cover the exact fusion source set")
    result = {}
    for source, shape in zip(expected, shapes_zyx, strict=True):
        row = source_rows[source]
        if _source_identity(Path(source)) != row["source_sha256"]:
            raise ValueError(f"Residual correction source metadata changed: {source}")
        recorded_shape = tuple(int(value) for value in row["shape_zyx"])
        if recorded_shape != shape or len(shape) != 3 or min(shape[1:]) < 2 or shape[0] < 1:
            raise ValueError(
                f"Residual correction source shape changed: {source}; "
                f"recorded={recorded_shape}, current={shape}"
            )
        model_identity = {
            "shared_coefficient": coefficient.tolist(),
            "source_coefficient": row["coefficient"].tolist(),
            "shape_zyx": shape,
            "gain": row["gain"],
            "global_scale": scale,
        }
        fingerprint = hashlib.sha256(
            json.dumps(model_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        result[source] = ResidualCorrection(
            coefficient=coefficient + row["coefficient"],
            shape_zyx=shape,
            multiplier=float(row["gain"] * scale),
            fingerprint=fingerprint,
        )
    return result
