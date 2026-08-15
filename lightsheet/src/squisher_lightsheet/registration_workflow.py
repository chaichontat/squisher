from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from skimage import filters

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as stitch_legacy
from squisher_lightsheet.artifact_io import (
    registration_input_fingerprint,
    sha256_file,
    write_text_set_atomic,
)
from squisher_lightsheet.channel_optimization import IDENTITY_AFFINE
from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR
from squisher_lightsheet.ngff import axes as ngff_axes
from squisher_lightsheet.ngff import level_array
from squisher_lightsheet.overlap_screen import screen_level2_overlaps


ThresholdMethod = Literal["minimum", "li"]
DIMENSIONS = ("z", "y", "x")


@dataclass(frozen=True)
class RegistrationThreshold:
    threshold: float
    method: str
    tile_count: int
    sampled_value_count: int
    output: Path


@dataclass(frozen=True)
class RegistrationWorkflowOutputs:
    threshold_record: Path
    measurement_summary: Path
    optimized_positions: Path
    diagnostics: Path
    constraints_jsonl: Path
    tile_corrections: Path
    canonical_positions: Path
    registration_json: Path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _require_absent(*paths: Path) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing registration output(s): {existing}")


def _require_matching_path(
    payload: dict[str, Any], *, key: str, expected: Path, artifact: Path
) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
        label = "position file" if key == "position_json" else key.replace("_", " ")
        raise ValueError(f"{artifact} belongs to a different {label}: {value}")


def _require_matching_settings(
    payload: dict[str, Any], *, expected: dict[str, Any], artifact: Path
) -> None:
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"{artifact} is missing settings")
    for key, expected_value in expected.items():
        if key not in settings or settings[key] != expected_value:
            raise ValueError(
                f"{artifact} setting {key}={settings.get(key)!r} differs from requested {expected_value!r}"
            )


def _validate_measurement_summary(
    payload: dict[str, Any],
    *,
    artifact: Path,
    position_json: Path,
    zarr_dir: Path,
    expected_settings: dict[str, Any],
) -> None:
    if payload.get("artifact_type") != "lightsheet.level0_phase_recovery_measurements.v1":
        raise ValueError(f"{artifact} is not a level-0 phase recovery measurement summary")
    _require_matching_path(payload, key="position_json", expected=position_json, artifact=artifact)
    _require_matching_path(payload, key="zarr_dir", expected=zarr_dir, artifact=artifact)
    if payload.get("input_fingerprint") != registration_input_fingerprint(position_json, zarr_dir):
        raise ValueError(f"{artifact} input fingerprint differs from the current registration inputs")
    _require_matching_settings(payload, expected=expected_settings, artifact=artifact)


def _validate_optimization_diagnostics(
    payload: dict[str, Any],
    *,
    artifact: Path,
    method8_summary: Path,
    position_json: Path,
    zarr_dir: Path,
    expected_settings: dict[str, Any],
    expected_outputs: dict[str, Path],
    allow_disconnected: bool = False,
) -> None:
    if payload.get("artifact_type") != "lightsheet.level0_phase_recovery_tile_optimization.v1":
        raise ValueError(f"{artifact} is not a level-0 phase recovery optimization record")
    _require_matching_path(payload, key="method8_summary", expected=method8_summary, artifact=artifact)
    if payload.get("method8_summary_sha256") != sha256_file(method8_summary):
        raise ValueError(f"{artifact} measurement summary content differs from the optimized input")
    _require_matching_path(payload, key="position_json", expected=position_json, artifact=artifact)
    _require_matching_settings(
        payload,
        expected={**expected_settings, "zarr_dir": str(zarr_dir.resolve())},
        artifact=artifact,
    )
    for key in ("tile_count", "connected_tile_count"):
        if not isinstance(payload.get(key), int):
            raise ValueError(f"{artifact} is missing integer {key}")
    if not allow_disconnected and payload["connected_tile_count"] != payload["tile_count"]:
        raise ValueError(
            f"registration graph is not fully connected: "
            f"{payload['connected_tile_count']}/{payload['tile_count']} tiles"
        )
    recorded_outputs = payload.get("outputs")
    if not isinstance(recorded_outputs, dict):
        raise ValueError(f"{artifact} is missing outputs")
    for key, expected_path in expected_outputs.items():
        value = recorded_outputs.get(key)
        if not isinstance(value, str) or Path(value).resolve() != expected_path.resolve():
            raise ValueError(f"{artifact} output {key}={value!r} differs from {expected_path.resolve()}")


def _record_vector(record: dict[str, Any], key: str) -> np.ndarray:
    values = record.get(key)
    if not isinstance(values, dict) or any(dimension not in values for dimension in DIMENSIONS):
        raise ValueError(f"tile {record.get('tile')!r} is missing {key} z/y/x")
    return np.asarray([float(values[dimension]) for dimension in DIMENSIONS], dtype=np.float64)


def _tile_path(record: dict[str, Any], zarr_dir: Path) -> Path:
    raw_tile = record.get("tile")
    if not isinstance(raw_tile, str) or not raw_tile:
        raise ValueError("position tile record is missing tile")
    name = Path(raw_tile).name
    for suffix in (".ome.tif", ".ome.tiff"):
        if name.endswith(suffix):
            name = f"{name[: -len(suffix)]}.ome.zarr"
            break
    path = zarr_dir / name
    if not path.is_dir():
        raise FileNotFoundError(f"missing tile OME-Zarr for {raw_tile}: {path}")
    return path


def _level_zero(path: Path) -> tuple[Any, str]:
    import zarr

    group = zarr.open_group(str(path), mode="r")
    array = level_array(group, context=path)
    return array, ngff_axes(group, array)


def _threshold(values: np.ndarray, method: ThresholdMethod) -> tuple[float, dict[str, Any]]:
    float_values = values.astype(np.float32, copy=False)
    if method == "li":
        threshold = float(filters.threshold_li(float_values))
        return threshold, {"saturated_tail_guard_applied": False}
    if method != "minimum":
        raise ValueError(f"unsupported threshold method {method!r}; expected 'minimum' or 'li'")

    raw_threshold = float(filters.threshold_minimum(float_values))
    raw_above_fraction = float(np.count_nonzero(values >= raw_threshold) / values.size)
    trim_max = float(np.percentile(values, 99.99))
    if raw_above_fraction < 0.01 and raw_threshold > trim_max:
        trimmed = values[values <= trim_max]
        threshold = float(filters.threshold_minimum(trimmed.astype(np.float32, copy=False)))
        return threshold, {
            "saturated_tail_guard_applied": True,
            "raw_threshold": raw_threshold,
            "raw_above_threshold_fraction": raw_above_fraction,
            "trim_percentile": 99.99,
            "trim_max": trim_max,
        }
    return raw_threshold, {
        "saturated_tail_guard_applied": False,
        "raw_threshold": raw_threshold,
        "raw_above_threshold_fraction": raw_above_fraction,
    }


def derive_registration_threshold(
    *,
    position_json: Path,
    zarr_dir: Path,
    output: Path,
    channel: int = 0,
    method: ThresholdMethod = "minimum",
    sample_step_yx: int = 4,
) -> RegistrationThreshold:
    """Derive a fixed registration mask from sampled level-0 center-Z planes.

    Sampling is bounded to one plane per tile and does not synthesize a pyramid
    by striding Z. XY subsampling only controls the threshold sample size.
    """
    if output.exists():
        raise FileExistsError(output)
    if sample_step_yx < 1:
        raise ValueError(f"sample_step_yx must be positive, got {sample_step_yx}")
    payload = json.loads(position_json.read_text())
    records = payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{position_json} must contain a non-empty tiles list")

    prepared = []
    z_min = np.inf
    z_max = -np.inf
    common_axes: str | None = None
    for record in records:
        path = _tile_path(record, zarr_dir)
        array, axes = _level_zero(path)
        if common_axes is None:
            common_axes = axes
        elif axes != common_axes:
            raise ValueError(f"mixed level-0 axes are unsupported: {common_axes} and {axes} in {path}")
        if axes == "CZYX":
            if not 0 <= int(channel) < int(array.shape[0]):
                raise ValueError(f"channel {channel} is outside CZYX shape {array.shape} in {path}")
            z_size = int(array.shape[1])
        elif axes == "ZYX":
            if int(channel) != 0:
                raise ValueError(f"channel {channel} requested from ZYX array in {path}")
            z_size = int(array.shape[0])
        else:
            raise ValueError(f"expected ZYX or CZYX level 0 in {path}, found axes {axes!r}")
        translation = _record_vector(record, "translation_um")
        scale = _record_vector(record, "scale_um")
        if scale[0] == 0:
            raise ValueError(f"tile {record.get('tile')!r} has zero Z scale")
        z_stop = translation[0] + z_size * scale[0]
        z_min = min(z_min, translation[0], z_stop)
        z_max = max(z_max, translation[0], z_stop)
        prepared.append((record, path, array, axes, translation, scale, z_size))

    center_z_um = float((z_min + z_max) / 2.0)
    samples = []
    sampled_tiles = []
    for record, path, array, axes, translation, scale, z_size in prepared:
        local_z = int(round((center_z_um - translation[0]) / scale[0]))
        if not 0 <= local_z < z_size:
            continue
        selection = (
            (int(channel), local_z, slice(None, None, sample_step_yx), slice(None, None, sample_step_yx))
            if axes == "CZYX"
            else (local_z, slice(None, None, sample_step_yx), slice(None, None, sample_step_yx))
        )
        plane = np.asarray(array[selection])
        finite = plane[np.isfinite(plane)]
        if finite.size:
            samples.append(finite.reshape(-1))
            sampled_tiles.append({"tile": record["tile"], "path": str(path), "local_z": local_z})
    if not samples:
        raise ValueError(f"no tiles intersect the global center Z={center_z_um} um")

    values = np.concatenate(samples)
    try:
        threshold, details = _threshold(values, method)
    except RuntimeError as exc:
        raise ValueError(f"{method} threshold failed for {values.size} sampled values") from exc
    method_name = f"skimage.filters.threshold_{method}"
    record = {
        "schema_version": 1,
        "artifact_type": "lightsheet.registration_threshold.v1",
        "method": method_name,
        "threshold": threshold,
        "source_level": 0,
        "axes": common_axes,
        "channel": int(channel),
        "sample_step_yx": int(sample_step_yx),
        "center_z_um": center_z_um,
        "position_json": str(position_json.resolve()),
        "zarr_dir": str(zarr_dir.resolve()),
        "tile_count": len(sampled_tiles),
        "sampled_value_count": int(values.size),
        "above_threshold_fraction": float(np.count_nonzero(values >= threshold) / values.size),
        "percentiles": {
            str(percentile): float(np.percentile(values, percentile))
            for percentile in (0, 1, 5, 50, 95, 99, 99.8, 99.99, 100)
        },
        "details": details,
        "tiles": sampled_tiles,
    }
    _write_json_atomic(output, record)
    return RegistrationThreshold(
        threshold=threshold,
        method=method_name,
        tile_count=len(sampled_tiles),
        sampled_value_count=int(values.size),
        output=output,
    )


def write_canonical_registration(
    *,
    optimized_position: Path,
    zarr_dir: Path,
    position_output: Path,
    registration_output: Path,
    measurement_summary: Path,
    diagnostics: Path,
    threshold_record: Path,
    allow_disconnected: bool = False,
) -> None:
    """Write fusion-ready identity affines around optimized stage translations."""
    _require_absent(position_output, registration_output)
    for path in (optimized_position, measurement_summary, diagnostics, threshold_record):
        if not path.is_file():
            raise FileNotFoundError(path)

    position_payload = json.loads(optimized_position.read_text())
    summary_payload = json.loads(measurement_summary.read_text())
    diagnostics_payload = json.loads(diagnostics.read_text())
    threshold_payload = json.loads(threshold_record.read_text())
    position_records = position_payload.get("tiles")
    if not isinstance(position_records, list):
        raise ValueError(f"{optimized_position} is missing tiles")
    for key in ("tile_count", "connected_tile_count"):
        if not isinstance(diagnostics_payload.get(key), int):
            raise ValueError(f"{diagnostics} is missing integer {key}")
    tile_count = diagnostics_payload["tile_count"]
    connected_count = diagnostics_payload["connected_tile_count"]
    if tile_count != len(position_records):
        raise ValueError(
            f"{diagnostics} tile_count={tile_count} differs from {optimized_position} tiles={len(position_records)}"
        )
    if not allow_disconnected and connected_count != tile_count:
        raise ValueError(f"registration graph is not fully connected: {connected_count}/{tile_count} tiles")

    method8 = bool(summary_payload.get("settings", {}).get("method8", False))
    provenance = {
        "method": (
            "level-0 phase correlation with axis-prior shifted-crop recovery and gated Method8"
            if method8
            else "level-0 phase correlation with axis-prior shifted-crop recovery"
        ),
        "measurement_summary": str(measurement_summary.resolve()),
        "optimized_position": str(optimized_position.resolve()),
        "diagnostics": str(diagnostics.resolve()),
        "threshold_record": str(threshold_record.resolve()),
        "settings": summary_payload.get("settings", {}),
        "threshold": threshold_payload,
        "connectivity": {
            "allow_disconnected": bool(allow_disconnected),
            "tile_count": int(tile_count),
            "connected_tile_count": int(connected_count),
        },
    }
    canonical_position = deepcopy(position_payload)
    canonical_position["registration_run"] = provenance

    tiles = stitch_legacy.read_position_input_tiles(optimized_position, input_dir=zarr_dir)
    canonical_position["input_dir"] = str(zarr_dir.resolve())
    canonical_position["tiles"] = []
    registration_tiles = []
    for source_record, tile in zip(position_records, tiles, strict=True):
        canonical_record = deepcopy(source_record)
        canonical_record.update(
            {
                "tile": tile.path.name,
                "path": str(tile.path),
                "shape": list(tile.shape),
                "axes": tile.axes,
                "channels": list(tile.channels),
                "tracks": [asdict(track) for track in tile.tracks],
                "translation_um": tile.translation,
                "scale_um": stitch_legacy.tile_stage_scale(tile),
            }
        )
        canonical_position["tiles"].append(canonical_record)
        record = {
            "tile": tile.path.name,
            "path": str(tile.path),
            "shape": list(tile.shape),
            "axes": tile.axes,
            "spacing_um": tile.spacing,
            "channels": list(tile.channels),
            "tracks": [asdict(track) for track in tile.tracks],
            "stage_translation_um": tile.translation,
            "stage_scale_um": stitch_legacy.tile_stage_scale(tile),
            "registered_affine": deepcopy(IDENTITY_AFFINE),
        }
        if tile.source_view is not None:
            record["source_view"] = tile.source_view
        registration_tiles.append(record)

    registration = {
        "input_dir": str(zarr_dir.resolve()),
        "metadata_transform_key": "stage_metadata",
        "registered_transform_key": "registered_affine",
        "spacing_um": tiles[0].spacing,
        "metrics": {
            "artifact_type": "lightsheet.level0_phase_recovery_identity_registration.v1",
            "registered_affine_note": (
                "Identity affine; optimized placement is baked into each tile's stage_translation_um."
            ),
            "registration_run": provenance,
            "optimization": diagnostics_payload,
        },
        "tiles": registration_tiles,
    }
    write_text_set_atomic(
        {
            position_output: json.dumps(canonical_position, indent=2) + "\n",
            registration_output: json.dumps(registration, indent=2) + "\n",
        }
    )


def _write_human_threshold_record(
    *,
    output: Path,
    threshold: float,
    channel: int,
    position_json: Path,
    zarr_dir: Path,
) -> None:
    _write_json_atomic(
        output,
        {
            "schema_version": 1,
            "artifact_type": "lightsheet.registration_threshold.v1",
            "method": "human_reviewed_threshold",
            "threshold": float(threshold),
            "source_level": 0,
            "channel": int(channel),
            "position_json": str(position_json.resolve()),
            "zarr_dir": str(zarr_dir.resolve()),
        },
    )


def run_registration_workflow(
    *,
    position_json: Path,
    zarr_dir: Path,
    output_dir: Path,
    threshold: float,
    channel: int = 0,
    method8_summary: Path | None = None,
    z_chunks: int = 6,
    device: int = 0,
    method8: bool = False,
    max_iterations: int = 300,
    ftol: float = 1e-4,
    min_corr: float = 0.15,
    min_grad_ncc: float = 0.24,
    phase_recovery_min_prior_edges_per_axis: int = 3,
    max_grad_regression: float = 0.02,
    max_corr_regression: float = 0.01,
    phase_fallback: bool = True,
    min_phase_grad: float = 0.24,
    min_phase_corr: float = 0.15,
    phase_fallback_weight_scale: float = 0.1,
    allow_disconnected: bool = False,
    native_lib_dir: Path = DEFAULT_LIB_DIR,
    progress: Callable[[str], None] | None = None,
) -> RegistrationWorkflowOutputs:
    """Run the human-gated level-2-screened registration workflow."""
    from squisher_lightsheet.method8_stitch_register import register_level0_phase_recovery

    if isinstance(threshold, bool) or not np.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError(f"threshold must be a finite non-negative number, got {threshold!r}")
    if not position_json.is_file():
        raise FileNotFoundError(position_json)
    if not zarr_dir.is_dir():
        raise FileNotFoundError(zarr_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    level2_screen = output_dir / "level2-screen.json"
    threshold_record = output_dir / "registration.threshold.json"
    if threshold_record.exists():
        threshold_payload = json.loads(threshold_record.read_text())
        if threshold_payload.get("artifact_type") != "lightsheet.registration_threshold.v1":
            raise ValueError(f"{threshold_record} is not a registration threshold record")
        if "threshold" not in threshold_payload:
            raise ValueError(f"{threshold_record} is missing threshold")
        if threshold_payload.get("method") != "human_reviewed_threshold":
            raise ValueError(
                f"threshold method {threshold_payload.get('method')!r} is not human_reviewed_threshold"
            )
        expected_threshold_settings = {
            "threshold": float(threshold),
            "channel": int(channel),
            "position_json": str(position_json.resolve()),
            "zarr_dir": str(zarr_dir.resolve()),
        }
        for key, expected in expected_threshold_settings.items():
            if threshold_payload.get(key) != expected:
                raise ValueError(
                    f"threshold record {key}={threshold_payload.get(key)!r} differs from {expected!r}"
                )
    else:
        _write_human_threshold_record(
            output=threshold_record,
            threshold=float(threshold),
            channel=channel,
            position_json=position_json,
            zarr_dir=zarr_dir,
        )

    if not level2_screen.exists():
        screen_level2_overlaps(
            position_json=position_json,
            zarr_dir=zarr_dir,
            output=level2_screen,
            threshold=float(threshold),
            level=2,
            channel=channel,
            z_chunks=z_chunks,
            min_foreground_pixels=256,
            min_foreground_fraction=0.05,
            progress=progress,
        )

    measurement_settings = {
        "pairs": [],
        "all_adjacent": True,
        "z_chunks": int(z_chunks),
        "device": int(device),
        "method8": bool(method8),
        "channel": int(channel),
        "max_iterations": int(max_iterations),
        "ftol": float(ftol),
        "min_corr": float(min_corr),
        "min_grad_ncc": float(min_grad_ncc),
        "fixed_mask_min_voxels": 256,
        "fixed_mask_max_masked_fraction": 0.95,
        "phase_recovery_shifted_crop": True,
        "phase_recovery_min_prior_edges_per_axis": int(phase_recovery_min_prior_edges_per_axis),
        "phase_recovery_min_phase_grad": float(min_phase_grad),
        "phase_recovery_min_phase_corr": float(min_phase_corr),
        "native_lib_dir": str(native_lib_dir.resolve()),
        "level2_screen": str(level2_screen.resolve()),
        "level2_screen_sha256": sha256_file(level2_screen),
    }
    optimization_settings = {
        "max_grad_regression": float(max_grad_regression),
        "max_corr_regression": float(max_corr_regression),
        "phase_fallback": bool(phase_fallback),
        "min_phase_grad": float(min_phase_grad),
        "min_phase_corr": float(min_phase_corr),
        "phase_fallback_weight_scale": float(phase_fallback_weight_scale),
    }
    default_summary = output_dir / "registration.measurements.json"
    if method8_summary is None and default_summary.is_file():
        method8_summary = default_summary
    canonical_positions = output_dir / "registration.positions.json"
    registration_json = output_dir / "registration.json"
    _require_absent(canonical_positions, registration_json)

    summary_threshold: float | None = None
    if method8_summary is not None:
        summary_payload = json.loads(method8_summary.read_text())
        _validate_measurement_summary(
            summary_payload,
            artifact=method8_summary,
            position_json=position_json,
            zarr_dir=zarr_dir,
            expected_settings=measurement_settings,
        )
        summary_threshold = summary_payload.get("settings", {}).get("fixed_mask_threshold")
        if summary_threshold != float(threshold):
            raise ValueError(
                f"explicit threshold {threshold} differs from summary threshold {summary_threshold}"
            )

    measurement_settings["fixed_mask_threshold"] = float(threshold)
    if method8_summary is not None:
        _validate_measurement_summary(
            json.loads(method8_summary.read_text()),
            artifact=method8_summary,
            position_json=position_json,
            zarr_dir=zarr_dir,
            expected_settings=measurement_settings,
        )

    optimized_positions = output_dir / "registration.optimized.positions.json"
    optimization_diagnostics = output_dir / "registration.optimization.diagnostics.json"
    constraints_jsonl = output_dir / "registration.constraints.jsonl"
    tile_corrections = output_dir / "registration.tile-corrections.json"
    if method8_summary is not None and all(
        path.is_file()
        for path in (
            optimized_positions,
            optimization_diagnostics,
            constraints_jsonl,
            tile_corrections,
        )
    ):
        from squisher_lightsheet.method8_stitch_register import Method8RegistrationOutputs

        _validate_optimization_diagnostics(
            json.loads(optimization_diagnostics.read_text()),
            artifact=optimization_diagnostics,
            method8_summary=method8_summary,
            position_json=position_json,
            zarr_dir=zarr_dir,
            expected_settings=optimization_settings,
            expected_outputs={
                "positions": optimized_positions,
                "constraints_jsonl": constraints_jsonl,
                "corrections": tile_corrections,
            },
            allow_disconnected=allow_disconnected,
        )

        outputs = Method8RegistrationOutputs(
            method8_summary=method8_summary,
            optimized_positions=optimized_positions,
            diagnostics=optimization_diagnostics,
            constraints_jsonl=constraints_jsonl,
            tile_corrections=tile_corrections,
        )
    else:
        outputs = register_level0_phase_recovery(
            position_json=position_json,
            zarr_dir=zarr_dir,
            output_dir=output_dir,
            level2_screen=level2_screen,
            method8_summary=method8_summary,
            pairs=None,
            all_adjacent=True,
            z_chunks=z_chunks,
            device=device,
            method8=method8,
            channel=channel,
            max_iterations=max_iterations,
            ftol=ftol,
            min_corr=min_corr,
            min_grad_ncc=min_grad_ncc,
            fixed_mask_threshold=float(threshold),
            fixed_mask_min_voxels=256,
            fixed_mask_max_masked_fraction=0.95,
            phase_recovery_min_prior_edges_per_axis=phase_recovery_min_prior_edges_per_axis,
            max_grad_regression=max_grad_regression,
            max_corr_regression=max_corr_regression,
            phase_fallback=phase_fallback,
            min_phase_grad=min_phase_grad,
            min_phase_corr=min_phase_corr,
            phase_fallback_weight_scale=phase_fallback_weight_scale,
            native_lib_dir=native_lib_dir,
            progress=progress,
        )

    _validate_measurement_summary(
        json.loads(outputs.method8_summary.read_text()),
        artifact=outputs.method8_summary,
        position_json=position_json,
        zarr_dir=zarr_dir,
        expected_settings=measurement_settings,
    )
    _validate_optimization_diagnostics(
        json.loads(outputs.diagnostics.read_text()),
        artifact=outputs.diagnostics,
        method8_summary=outputs.method8_summary,
        position_json=position_json,
        zarr_dir=zarr_dir,
        expected_settings=optimization_settings,
        expected_outputs={
            "positions": outputs.optimized_positions,
            "constraints_jsonl": outputs.constraints_jsonl,
            "corrections": outputs.tile_corrections,
        },
        allow_disconnected=allow_disconnected,
    )
    write_canonical_registration(
        optimized_position=outputs.optimized_positions,
        zarr_dir=zarr_dir,
        position_output=canonical_positions,
        registration_output=registration_json,
        measurement_summary=outputs.method8_summary,
        diagnostics=outputs.diagnostics,
        threshold_record=threshold_record,
        allow_disconnected=allow_disconnected,
    )
    return RegistrationWorkflowOutputs(
        threshold_record=threshold_record,
        measurement_summary=outputs.method8_summary,
        optimized_positions=outputs.optimized_positions,
        diagnostics=outputs.diagnostics,
        constraints_jsonl=outputs.constraints_jsonl,
        tile_corrections=outputs.tile_corrections,
        canonical_positions=canonical_positions,
        registration_json=registration_json,
    )
