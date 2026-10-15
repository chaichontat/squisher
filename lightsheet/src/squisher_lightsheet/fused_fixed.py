from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any, Literal, Sequence

import numpy as np

from squisher_lightsheet.channel_affine import write_global_channel_affine_registration
from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR
from squisher_lightsheet.local_affine import local_affine_cache_config
from squisher_lightsheet._backends import backend_executable


DEFAULT_SWEEP_RUNNER = Path(__file__).parent / "_backends" / "fused_fixed_sweep.py"
SUMMARY_NAME = "fused_fixed_method8_summary.json"
SPATIAL_RECOVERY_ATTEMPT = "interpolated_method6_spatial_recovery"
LINEAR_MAPPING_RECOVERY_ATTEMPT = "interpolated_method6_linear_mapping_recovery"
DIMENSIONS = ("z", "y", "x")
FUSED_COORDINATE_ADAPTATION_VERSION = "v2_common_fixed_frame"
DEFAULT_STARTING_AFFINE_MATRIX_ZYX = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
FitIntensityTransform = Literal["linear", "log1p"]


def normalize_starting_affine_matrix_zyx(values: Sequence[float]) -> tuple[float, ...]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (9,):
        raise ValueError("Starting affine matrix must contain 9 row-major values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Starting affine matrix must contain only finite values")
    if np.linalg.matrix_rank(matrix.reshape(3, 3)) != 3:
        raise ValueError("Starting affine matrix must be invertible")
    return tuple(float(value) for value in matrix)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _records_by_tile(payload: dict[str, Any], *, source: Path) -> dict[str, dict[str, Any]]:
    records = payload.get("tiles")
    if not isinstance(records, list):
        raise ValueError(f"{source} has no tiles list")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("tile"), str):
            raise ValueError(f"{source} contains a tile record without a tile name")
        tile = record["tile"]
        if tile in indexed:
            raise ValueError(f"{source} contains duplicate tile {tile!r}")
        indexed[tile] = record
    return indexed


def _canonical_tile_key(record: dict[str, Any]) -> tuple[str, str] | None:
    view = record.get("source_view", record.get("side"))
    tile = record.get("tile")
    if not isinstance(view, str) or not view or not isinstance(tile, str):
        return None
    parts = tile.split(".")
    if len(parts) < 3 or not parts[-3].isdigit():
        return None
    return view, parts[-3]


def _vector(record: dict[str, Any], field: str, *, source: Path) -> np.ndarray:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{source} tile {record.get('tile')!r} has no {field}")
    try:
        vector = np.asarray([value[dim] for dim in DIMENSIONS], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} tile {record.get('tile')!r} has invalid {field}"
        ) from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{source} tile {record.get('tile')!r} has non-finite {field}")
    return vector


def _shape_zyx(record: dict[str, Any], *, source: Path) -> np.ndarray:
    shape = record.get("shape")
    axes = str(record.get("axes", ""))
    if not isinstance(shape, list):
        raise ValueError(f"{source} tile {record.get('tile')!r} has no shape")
    if axes == "CZYX" and len(shape) == 4:
        values = shape[1:]
    elif axes == "ZYX" and len(shape) == 3:
        values = shape
    else:
        raise ValueError(
            f"{source} tile {record.get('tile')!r} has unsupported axes/shape "
            f"{axes!r}/{shape!r}"
        )
    result = np.asarray(values, dtype=np.float64)
    if np.any(result <= 0) or not np.all(np.isfinite(result)):
        raise ValueError(f"{source} tile {record.get('tile')!r} has invalid shape")
    return result


def _registered_affine(record: dict[str, Any], *, source: Path) -> np.ndarray:
    value = record.get("registered_affine")
    matrix = None if not isinstance(value, dict) else value.get("matrix")
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{source} tile {record.get('tile')!r} has no finite 4x4 registered_affine"
        )
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(
            f"{source} tile {record.get('tile')!r} has a non-affine homogeneous row"
        )
    return result


def _fused_proxy_origin(
    position_record: dict[str, Any],
    affine: np.ndarray,
    *,
    source: Path,
) -> np.ndarray:
    """Center an axis-aligned tile proxy on its registered position."""
    origin = _vector(position_record, "translation_um", source=source)
    scale = _vector(position_record, "scale_um", source=source)
    center = _shape_zyx(position_record, source=source) * scale / 2.0
    registered_center = affine @ np.r_[origin + center, 1.0]
    return registered_center[:3] - center


def _registration_from_fusion_provenance(fixed_fused: Path) -> Path | None:
    manifest_path = fixed_fused / "provenance" / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json_object(manifest_path)
    inputs = manifest.get("inputs")
    registration_value = inputs.get("registration_input") if isinstance(inputs, dict) else None
    if isinstance(registration_value, str) and registration_value:
        registration = Path(registration_value)
        if registration.is_file():
            return registration.resolve()

    candidates: list[Path] = []
    for artifact in manifest.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        roles = artifact.get("roles")
        has_registration_role = artifact.get("role") == "registration_input" or (
            isinstance(roles, list) and "registration_input" in roles
        )
        if not has_registration_role:
            continue
        source = artifact.get("source_path")
        if isinstance(registration_value, str) and registration_value and source != registration_value:
            continue
        if isinstance(source, str) and Path(source).is_file():
            candidates.append(Path(source).resolve())
            continue
        bundled = artifact.get("bundled_path")
        if isinstance(bundled, str) and (fixed_fused / bundled).is_file():
            candidates.append((fixed_fused / bundled).resolve())

    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{fixed_fused} provenance has multiple registration_input artifacts: {candidates}"
        )
    if isinstance(registration_value, str) and registration_value:
        raise ValueError(
            f"{fixed_fused} provenance names registration input {registration_value}, "
            "but it is unavailable"
        )
    return None


def _output_grid_template_from_fusion_provenance(fixed_fused: Path) -> Path | None:
    manifest_path = fixed_fused / "provenance" / "manifest.json"
    if not manifest_path.is_file():
        return None
    inputs = _read_json_object(manifest_path).get("inputs")
    template = inputs.get("output_grid_template") if isinstance(inputs, dict) else None
    if template is None:
        return None
    return Path(template["path"])


def _prepare_fused_coordinate_positions(
    *,
    fixed_position: Path,
    moving_position: Path,
    fixed_fused: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Match Method 6 position inputs to the coordinate frame of the fixed fusion."""
    registration_path = _registration_from_fusion_provenance(fixed_fused)
    if registration_path is None:
        return fixed_position, moving_position

    fixed_payload = _read_json_object(fixed_position)
    moving_payload = _read_json_object(moving_position)
    registration_payload = _read_json_object(registration_path)
    fixed_records = _records_by_tile(fixed_payload, source=fixed_position)
    moving_records = _records_by_tile(moving_payload, source=moving_position)
    registration_records = _records_by_tile(registration_payload, source=registration_path)
    missing = sorted(set(fixed_records) - set(registration_records))
    if missing:
        # Materialized-window fusions name windows, not source tiles; a fusion written onto an
        # output grid template shares that template's coordinate frame.
        template = _output_grid_template_from_fusion_provenance(fixed_fused)
        if template is not None:
            return _prepare_fused_coordinate_positions(
                fixed_position=fixed_position,
                moving_position=moving_position,
                fixed_fused=template,
                output_dir=output_dir,
            )
        raise ValueError(
            f"Fixed-fusion registration {registration_path} is missing fixed tiles: {missing}"
        )

    identity = np.eye(4, dtype=np.float64)
    nonidentity_tiles = [
        tile
        for tile in fixed_records
        if not np.allclose(
            _registered_affine(registration_records[tile], source=registration_path),
            identity,
            atol=1e-8,
            rtol=0.0,
        )
    ]
    if not nonidentity_tiles:
        return fixed_position, moving_position

    stage_matches = []
    fused_matches = []
    for tile, fixed_record in fixed_records.items():
        registration_record = registration_records[tile]
        affine = _registered_affine(registration_record, source=registration_path)
        position_origin = _vector(fixed_record, "translation_um", source=fixed_position)
        stage_origin = _vector(
            registration_record, "stage_translation_um", source=registration_path
        )
        fused_origin = _fused_proxy_origin(fixed_record, affine, source=fixed_position)
        stage_matches.append(np.allclose(position_origin, stage_origin, atol=1e-5, rtol=0.0))
        fused_matches.append(np.allclose(position_origin, fused_origin, atol=1e-5, rtol=0.0))

    if all(fused_matches):
        return fixed_position, moving_position
    if not all(stage_matches):
        raise ValueError(
            "Fixed positions are neither registration-stage coordinates nor fused-coordinate "
            f"proxies for {registration_path}; refusing ambiguous Method 6 placement"
        )

    digest = hashlib.sha256()
    digest.update(FUSED_COORDINATE_ADAPTATION_VERSION.encode())
    for path in (fixed_position, moving_position, registration_path):
        digest.update(path.resolve().as_posix().encode())
        digest.update(path.read_bytes())
    adapted_dir = output_dir / "fused-coordinate-inputs" / digest.hexdigest()[:16]
    adapted_fixed = adapted_dir / "fixed.positions.json"
    adapted_moving = adapted_dir / "moving.positions.json"
    if adapted_fixed.is_file() and adapted_moving.is_file():
        return adapted_fixed.resolve(), adapted_moving.resolve()

    fixed_affines = {
        tile: _registered_affine(registration_records[tile], source=registration_path)
        for tile in fixed_records
    }
    moving_tiles_match_fixed = set(moving_records).issubset(fixed_records)
    if moving_tiles_match_fixed:
        moving_affines = {tile: fixed_affines[tile] for tile in moving_records}
        moving_affine_source = "matching_fixed_tile_registered_affine"
    else:
        fixed_by_canonical = {
            key: fixed_affines[tile]
            for tile, record in fixed_records.items()
            if (key := _canonical_tile_key(record)) is not None
        }
        moving_canonical = {
            tile: _canonical_tile_key(record) for tile, record in moving_records.items()
        }
        if moving_canonical and all(
            key is not None and key in fixed_by_canonical for key in moving_canonical.values()
        ):
            moving_affines = {
                tile: fixed_by_canonical[key]
                for tile, key in moving_canonical.items()
                if key is not None
            }
            moving_affine_source = "matching_canonical_fixed_tile_registered_affine"
        else:
            fixed_frame_affine = next(iter(fixed_affines.values()))
            if not all(
                np.allclose(affine, fixed_frame_affine, atol=1e-8, rtol=0.0)
                for affine in fixed_affines.values()
            ):
                raise ValueError(
                    "Fixed-fusion registration has tile-varying affines, so no single fused-frame "
                    "transform can be applied to disjoint moving tiles"
                )
            moving_affines = {tile: fixed_frame_affine for tile in moving_records}
            moving_affine_source = "common_fixed_frame_registered_affine"

    def adapt(
        payload: dict[str, Any],
        records: dict[str, dict[str, Any]],
        affines: dict[str, np.ndarray],
        source: Path,
        *,
        affine_source: str,
    ) -> None:
        for tile, record in records.items():
            origin = _fused_proxy_origin(record, affines[tile], source=source)
            record["translation_um"] = {
                dim: float(value) for dim, value in zip(DIMENSIONS, origin, strict=True)
            }
        payload["fused_coordinate_adaptation"] = {
            "version": FUSED_COORDINATE_ADAPTATION_VERSION,
            "method": "registered_affine_centered_axis_aligned_proxy",
            "affine_source": affine_source,
            "fixed_fused": str(fixed_fused.resolve()),
            "registration_input": str(registration_path),
        }

    adapt(
        fixed_payload,
        fixed_records,
        fixed_affines,
        fixed_position,
        affine_source="matching_fixed_tile_registered_affine",
    )
    adapt(
        moving_payload,
        moving_records,
        moving_affines,
        moving_position,
        affine_source=moving_affine_source,
    )
    adapted_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in ((adapted_fixed, fixed_payload), (adapted_moving, moving_payload)):
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
    return adapted_fixed.resolve(), adapted_moving.resolve()


def _resolved_path(value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Method 6 summary has no {key!r} path")
    return Path(value).resolve()


def validate_method6_summary(
    summary_path: Path,
    *,
    moving_position: Path,
    moving_source_position: Path,
    fixed_fused: Path,
    moving_channel: int,
    fixed_mask_threshold: float,
    fit_intensity_transform: FitIntensityTransform = "log1p",
    starting_affine_matrix_zyx: Sequence[float] = DEFAULT_STARTING_AFFINE_MATRIX_ZYX,
    min_grad_ncc: float = 0.24,
    local_3d_refinement: bool = False,
    local_affine_estimator: Literal["least-squares", "restrained"] = "least-squares",
    level0_initializer: str = "level2-method8",
    split_parent_windows: bool = False,
) -> dict[str, Any]:
    """Reject cached output unless it matches the canonical Method 6 contract."""
    payload = json.loads(summary_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{summary_path} does not contain a JSON object")
    cache = payload.get("cache_config")
    if not isinstance(cache, dict):
        raise ValueError(f"{summary_path} has no cache_config object")

    expected_scalars = {
        "native_method": "method6",
        "fit_intensity_transform": fit_intensity_transform,
        "level0_initializer": level0_initializer,
        "moving_channel": moving_channel,
    }
    for key, expected in expected_scalars.items():
        actual = cache.get(key)
        if actual != expected:
            raise ValueError(
                f"Method 6 summary contract mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
    if cache.get("split_parent_windows", False) is not split_parent_windows:
        raise ValueError("Method 6 summary contract mismatch for split_parent_windows")
    if cache.get("local_3d_refinement", False) is not local_3d_refinement:
        raise ValueError("Method 6 summary contract mismatch for local_3d_refinement")
    expected_policy = local_affine_cache_config(local_affine_estimator)
    if cache.get("local_affine_estimator", "least-squares") != local_affine_estimator:
        raise ValueError("Method 6 summary contract mismatch for local_affine_estimator")
    for key, expected in expected_policy.items():
        if cache.get(key) != expected:
            raise ValueError(f"Method 6 summary contract mismatch for {key}")
    expected_matrix = normalize_starting_affine_matrix_zyx(starting_affine_matrix_zyx)
    try:
        actual_matrix = normalize_starting_affine_matrix_zyx(cache["starting_affine_matrix_zyx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Method 6 summary has no valid starting_affine_matrix_zyx"
        ) from exc
    if actual_matrix != expected_matrix:
        raise ValueError(
            "Method 6 summary contract mismatch for starting_affine_matrix_zyx: "
            f"expected {expected_matrix!r}, got {actual_matrix!r}"
        )
    actual_threshold = cache.get("fixed_mask_threshold")
    if actual_threshold is None or float(actual_threshold) != float(fixed_mask_threshold):
        raise ValueError(
            "Method 6 summary contract mismatch for fixed_mask_threshold: "
            f"expected {fixed_mask_threshold}, got {actual_threshold!r}"
        )
    actual_min_grad_ncc = cache.get("min_grad_ncc")
    if actual_min_grad_ncc is None or float(actual_min_grad_ncc) != float(min_grad_ncc):
        raise ValueError(
            "Method 6 summary contract mismatch for min_grad_ncc: "
            f"expected {min_grad_ncc}, got {actual_min_grad_ncc!r}"
        )

    expected_paths = {
        "moving_position": moving_position,
        "moving_source_position": moving_source_position,
        "fixed_fused": fixed_fused,
    }
    for key, expected in expected_paths.items():
        actual = _resolved_path(cache.get(key), key=key)
        if actual != expected.resolve():
            raise ValueError(
                f"Method 6 summary contract mismatch for {key}: "
                f"expected {expected.resolve()}, got {actual}"
            )
    return payload


def run_fused_fixed_method6(
    *,
    fixed_position: Path,
    moving_position: Path,
    moving_source_position: Path,
    fixed_fused: Path,
    output_dir: Path,
    output_registration: Path,
    moving_channel: int,
    fixed_mask_threshold: float,
    source_label: str,
    target_label: str,
    sweep_runner: Path = DEFAULT_SWEEP_RUNNER,
    window_filter_json: Path | None = None,
    parent_window_dir: Path | None = None,
    parent_reference_fused: Path | None = None,
    split_parent_windows: bool = False,
    core_shape_zyx: tuple[int, int, int] = (480, 480, 480),
    window_shape_zyx: tuple[int, int, int] = (528, 528, 528),
    fit_downsample_zyx: tuple[int, int, int] = (1, 1, 1),
    native_lib_dir: Path = DEFAULT_LIB_DIR,
    ftol: float = 1e-4,
    max_iterations: int = 300,
    local_3d_refinement: bool = False,
    local_affine_estimator: Literal["least-squares", "restrained"] = "least-squares",
    phase_upsample_factor: int = 10,
    min_corr: float = 0.15,
    min_grad_ncc: float = 0.24,
    fit_intensity_transform: FitIntensityTransform = "log1p",
    starting_affine_matrix_zyx: Sequence[float] = DEFAULT_STARTING_AFFINE_MATRIX_ZYX,
    fixed_mask_level: int = 2,
    fixed_mask_min_voxels: int = 256,
    fixed_mask_max_masked_fraction: float = 0.95,
    workers: int = 1,
    max_tasks_per_worker: int = 10,
    devices: tuple[int, ...] = (0,),
    max_windows: int | None = None,
    resume: bool = True,
) -> tuple[Path, Path]:
    """Run the Prod2-style fused-fixed Method 6 fit and emit a canonical registration."""
    if split_parent_windows and parent_window_dir is None:
        raise ValueError("split_parent_windows requires parent_window_dir")
    if parent_reference_fused is not None and parent_window_dir is None:
        raise ValueError("parent_reference_fused requires parent_window_dir")
    local_affine_cache_config(local_affine_estimator)
    if local_affine_estimator != "least-squares" and not local_3d_refinement:
        raise ValueError("restrained estimator requires local_3d_refinement")
    if fit_intensity_transform not in {"linear", "log1p"}:
        raise ValueError(
            "fit_intensity_transform must be either 'linear' or 'log1p', got "
            f"{fit_intensity_transform!r}"
        )
    normalized_starting_affine = normalize_starting_affine_matrix_zyx(
        starting_affine_matrix_zyx
    )
    effective_fixed_position, effective_moving_position = _prepare_fused_coordinate_positions(
        fixed_position=fixed_position,
        moving_position=moving_position,
        fixed_fused=fixed_fused,
        output_dir=output_dir,
    )
    if sweep_runner.resolve() == DEFAULT_SWEEP_RUNNER.resolve():
        command = [str(backend_executable("lightsheet-fused-fixed-sweep"))]
    else:
        command = [sys.executable, "-u", str(sweep_runner.resolve())]
    command.extend([
        "--fixed-position",
        str(effective_fixed_position.resolve()),
        "--moving-position",
        str(effective_moving_position.resolve()),
        "--moving-source-position",
        str(moving_source_position.resolve()),
        "--fixed-fused",
        str(fixed_fused.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--core-shape-zyx",
        ",".join(map(str, core_shape_zyx)),
        "--window-shape-zyx",
        ",".join(map(str, window_shape_zyx)),
        "--fit-downsample-zyx",
        ",".join(map(str, fit_downsample_zyx)),
        "--moving-channel",
        str(moving_channel),
        "--native-lib-dir",
        str(native_lib_dir.resolve()),
        "--native-method",
        "method6",
        "--starting-affine-matrix-zyx",
        ",".join(map(str, normalized_starting_affine)),
        "--fit-intensity-transform",
        fit_intensity_transform,
        "--ftol",
        str(ftol),
        "--max-iterations",
        str(max_iterations),
        "--phase-upsample-factor",
        str(phase_upsample_factor),
        "--min-corr",
        str(min_corr),
        "--min-grad-ncc",
        str(min_grad_ncc),
        "--fixed-mask-threshold",
        str(fixed_mask_threshold),
        "--fixed-mask-level",
        str(fixed_mask_level),
        "--fixed-mask-min-voxels",
        str(fixed_mask_min_voxels),
        "--fixed-mask-max-masked-fraction",
        str(fixed_mask_max_masked_fraction),
        "--workers",
        str(workers),
        "--max-tasks-per-worker",
        str(max_tasks_per_worker),
        "--devices",
        ",".join(map(str, devices)),
        "--level0-initializer",
        "window-parent" if parent_window_dir is not None else "level2-method8",
        "--resume" if resume else "--no-resume",
    ])
    if parent_window_dir is not None:
        command.extend(["--parent-window-dir", str(parent_window_dir.resolve())])
    if parent_reference_fused is not None:
        command.extend(["--parent-reference-fused", str(parent_reference_fused.resolve())])
    if split_parent_windows:
        command.append("--split-parent-windows")
    if local_3d_refinement:
        command.extend(["--local-3d-refinement", "--local-affine-estimator", local_affine_estimator])
    if window_filter_json is not None:
        command.extend(["--window-filter-json", str(window_filter_json.resolve())])
    if max_windows is not None:
        command.extend(["--max-windows", str(max_windows)])

    subprocess.run(command, check=True)
    summary_path = output_dir / SUMMARY_NAME
    if not summary_path.is_file():
        raise RuntimeError(f"Method 6 sweep completed without writing {summary_path}")
    validate_method6_summary(
        summary_path,
        moving_position=effective_moving_position,
        moving_source_position=moving_source_position,
        fixed_fused=fixed_fused,
        moving_channel=moving_channel,
        fixed_mask_threshold=fixed_mask_threshold,
        fit_intensity_transform=fit_intensity_transform,
        starting_affine_matrix_zyx=normalized_starting_affine,
        min_grad_ncc=min_grad_ncc,
        local_3d_refinement=local_3d_refinement,
        local_affine_estimator=local_affine_estimator,
        level0_initializer="window-parent" if parent_window_dir is not None else "level2-method8",
        split_parent_windows=split_parent_windows,
    )
    registration = write_global_channel_affine_registration(
        window_dir=output_dir / "window_json",
        reference_registration_input=moving_source_position,
        output_registration=output_registration,
        expected_moving_channel=moving_channel,
        expected_fixed_fused=fixed_fused,
        source_label=source_label,
        target_label=target_label,
    )
    return summary_path.resolve(), registration
