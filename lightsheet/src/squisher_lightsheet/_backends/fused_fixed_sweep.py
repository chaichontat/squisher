#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from squisher_lightsheet.local_affine import local_affine_cache_config
from scipy.linalg import polar
from scipy.spatial.transform import Rotation


LEVEL0_REFINABLE_LEVEL2_REJECTIONS = frozenset(("quality_gate_corr", "quality_gate_grad_ncc"))
DEFAULT_OUTPUT_DIR = Path(
    "/home/chaichontat/nvme/lightsheet/working/eduseg/Image_14/"
    "image10-ch0-to-image14-fused-fixed-method8-c528-overlap10"
)
DEFAULT_LEVEL2_PHASEPRIMED_DIR = Path(
    "/home/chaichontat/nvme/lightsheet/working/eduseg/Image_14/"
    "image10-ch0-to-image14-ch0-level2-actual-tile-dense-z6x96-fullxy-phaseprimed-method8"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=True) + "\n")
    tmp.replace(path)


def _parse_zyx(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 3 or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(f"expected positive z,y,x tuple, got {text!r}")
    return values


def _parse_devices(text: str) -> tuple[int, ...]:
    devices = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not devices:
        raise argparse.ArgumentTypeError("expected at least one CUDA device")
    return devices


def _safe_stem(value: str) -> str:
    return value.removesuffix(".ome.zarr").replace("/", "_")


def _tile_number(tile_name: str) -> str:
    return tile_name.split(".")[-3]


def _axis_starts(size: int, *, core: int, window: int) -> list[int]:
    if window > size:
        raise ValueError(f"window={window} exceeds axis size={size}")
    starts = list(range(0, size - window + 1, core))
    tail = size - window
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return starts


def _windows(shape_zyx: tuple[int, int, int], *, core_shape_zyx: tuple[int, int, int], window_shape_zyx: tuple[int, int, int]) -> list[dict[str, Any]]:
    z_starts = _axis_starts(shape_zyx[0], core=core_shape_zyx[0], window=window_shape_zyx[0])
    y_starts = _axis_starts(shape_zyx[1], core=core_shape_zyx[1], window=window_shape_zyx[1])
    x_starts = _axis_starts(shape_zyx[2], core=core_shape_zyx[2], window=window_shape_zyx[2])
    output: list[dict[str, Any]] = []
    for qy, y in enumerate(y_starts):
        for qx, x in enumerate(x_starts):
            for z in z_starts:
                output.append(
                    {
                        "quadrant": f"qy{qy}_qx{qx}",
                        "start_zyx": [int(z), int(y), int(x)],
                        "shape_zyx": [int(value) for value in window_shape_zyx],
                    }
                )
    return output


def _load_tiles(position: Path, tile_filter: set[str] | None) -> list[dict[str, Any]]:
    payload = json.loads(position.read_text())
    records = []
    for record in payload["tiles"]:
        tile_no = _tile_number(str(record["tile"]))
        if tile_filter is None or tile_no in tile_filter:
            records.append(record)
    return records


def _record_vector_zyx(record: dict[str, Any], key: str) -> np.ndarray:
    source = record[key]
    return np.asarray([float(source[axis]) for axis in ("z", "y", "x")], dtype=np.float64)


def _record_center_um(record: dict[str, Any]) -> np.ndarray:
    translation_key = "stage_translation_um" if "stage_translation_um" in record else "translation_um"
    scale_key = "stage_scale_um" if "stage_scale_um" in record else "scale_um"
    translation = _record_vector_zyx(record, translation_key)
    if record.get("shape") is None:
        return translation
    scale = _record_vector_zyx(record, scale_key)
    axes = str(record["axes"]).upper()
    shape = np.asarray([int(record["shape"][axes.index(axis)]) for axis in "ZYX"], dtype=np.float64)
    return translation + scale * (shape - 1.0) / 2.0


def _nearest_level2_phaseprimed_paths(
    *,
    level2_dir: Path,
    records: list[dict[str, Any]],
    path_for_tile: Any,
) -> dict[str, dict[str, Any]]:
    records_by_tile = {_tile_number(str(record["tile"])): record for record in records}
    accepted: list[tuple[str, Path, np.ndarray]] = []
    for tile_no, record in records_by_tile.items():
        path = path_for_tile(level2_dir, str(record["tile"]))
        if not path.exists():
            continue
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if row.get("status") == "accepted":
            accepted.append((tile_no, path, _record_center_um(record)))
    if not accepted:
        raise ValueError(f"{level2_dir} has no accepted level2 phase-primed tile initializers")

    accepted_centers = np.stack([item[2] for item in accepted], axis=0)
    output: dict[str, dict[str, Any]] = {}
    for tile_no, record in records_by_tile.items():
        own_path = path_for_tile(level2_dir, str(record["tile"]))
        own_accepted = False
        if own_path.exists():
            try:
                own_row = json.loads(own_path.read_text())
            except json.JSONDecodeError:
                own_row = {}
            own_accepted = own_row.get("status") == "accepted"
        if own_accepted:
            output[tile_no] = {
                "path": own_path,
                "source_tile": tile_no,
                "source_tile_name": str(record["tile"]),
                "distance_um_yx": 0.0,
                "fallback": False,
            }
            continue
        center = _record_center_um(record)
        distances = np.linalg.norm(accepted_centers[:, 1:3] - center[1:3], axis=1)
        nearest_index = int(np.argmin(distances))
        source_tile, source_path, _source_center = accepted[nearest_index]
        output[tile_no] = {
            "path": source_path,
            "source_tile": source_tile,
            "source_tile_name": str(records_by_tile[source_tile]["tile"]),
            "distance_um_yx": float(distances[nearest_index]),
            "fallback": True,
        }
    return output


def _window_identity_from_virtual_row(row: dict[str, Any]) -> tuple[str, str, int]:
    tile_name = str(row.get("source_tile") or row.get("moving_tile") or "")
    if not tile_name:
        raise ValueError(f"window filter row is missing source_tile/moving_tile: {row}")
    virtual_name = str(row.get("tile", ""))
    parts = virtual_name.removesuffix(".ome.zarr").split(".")
    quadrant = str(row.get("quadrant") or "")
    z_start = row.get("z_start")
    if not quadrant:
        for part in parts:
            if part.startswith("qy") and "_qx" in part:
                quadrant = part
                break
    if z_start is None:
        for part in parts:
            if part.startswith("z"):
                z_start = int(part[1:])
                break
    if z_start is None and isinstance(row.get("source_start_zyx"), list):
        z_start = int(row["source_start_zyx"][0])
    if not quadrant or z_start is None:
        raise ValueError(f"could not parse window identity from filter row: {row}")
    return tile_name, quadrant, int(z_start)


def _load_window_filter(args: argparse.Namespace) -> set[tuple[str, str, int]] | None:
    if args.window_filter_json is None:
        return None
    payload = json.loads(Path(args.window_filter_json).read_text())
    rows = payload.get("tiles") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{args.window_filter_json} must contain a list or an object with a tiles list")
    transform_source = args.window_filter_transform_source
    selected: set[tuple[str, str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if transform_source is not None and row.get("transform_source") != transform_source:
            continue
        selected.add(_window_identity_from_virtual_row(row))
    if not selected:
        raise ValueError(f"{args.window_filter_json} selected no windows")
    return selected


def _task_paths(output_dir: Path, tile_name: str, quadrant: str, start_zyx: list[int]) -> tuple[Path, Path]:
    stem = f"{_safe_stem(tile_name)}.{quadrant}.z{int(start_zyx[0]):05d}"
    return output_dir / "level2_json" / f"{stem}.json", output_dir / "window_json" / f"{stem}.json"


def _level2_phaseprimed_tile_path(level2_dir: Path, tile_name: str) -> Path:
    return level2_dir / "tile_json" / f"{_safe_stem(tile_name)}.level2_dense_z_phaseprimed_method8.json"


def _level2_mattes_tile_path(level2_dir: Path, tile_name: str) -> Path:
    return level2_dir / "tile_json" / f"{_safe_stem(tile_name)}.level2_method10_mattes.json"


def _window_initializer_path(initializer_dir: Path, level0_path: Path) -> Path:
    return initializer_dir / level0_path.name


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_window_initializer(path: Path, *, tile_name: str, start_zyx: list[int]) -> None:
    try:
        row = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid per-window initializer JSON {path}: {exc}") from exc
    accepted_artifacts = {
        "lightsheet.fused_fixed_spatial_initializer.v1",
        "lightsheet.fused_fixed_native_spatial_recovery_initializer.v1",
    }
    if row.get("artifact_type") not in accepted_artifacts:
        raise ValueError(
            f"unusable per-window initializer {path}: artifact_type={row.get('artifact_type')!r}; "
            f"expected one of {sorted(accepted_artifacts)!r}"
        )
    required = ("level_factor_zyx", "local_matrix_zyx", "local_translation_zyx")
    missing = [key for key in required if key not in row]
    if row.get("status") != "accepted" or missing:
        raise ValueError(
            f"unusable per-window initializer {path}: status={row.get('status')!r} missing={missing}"
        )
    factor = np.asarray(row["level_factor_zyx"], dtype=np.float64)
    matrix = np.asarray(row["local_matrix_zyx"], dtype=np.float64)
    translation = np.asarray(row["local_translation_zyx"], dtype=np.float64)
    if factor.shape != (3,) or matrix.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(
            f"invalid per-window initializer transform shapes for {path}: "
            f"factor={factor.shape} matrix={matrix.shape} translation={translation.shape}"
        )
    if np.any(factor <= 0) or not all(np.all(np.isfinite(values)) for values in (factor, matrix, translation)):
        raise ValueError(f"non-finite or non-positive per-window initializer transform in {path}")
    if row.get("moving_tile") != tile_name or row.get("moving_start_l0_zyx") != start_zyx:
        raise ValueError(
            f"per-window initializer identity mismatch for {path}: "
            f"expected tile={tile_name!r} start={start_zyx}, "
            f"got tile={row.get('moving_tile')!r} start={row.get('moving_start_l0_zyx')!r}"
        )


def _accepted_parent_windows(
    directory: Path, *, tile_name: str, pattern: str,
    fixed_fused: Path, moving_path: Path, moving_channel: int,
    parent_reference_fused: Path | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    reference = fixed_fused if parent_reference_fused is None else parent_reference_fused
    if parent_reference_fused is not None:
        from squisher.ome_geometry import ome_spatial_grid
        from squisher_lightsheet.ngff import require_same_multiscale_grid
        from squisher_lightsheet._backends.fused_fixed_trial import _fixed_fused_root

        roots = [_fixed_fused_root(path) for path in (reference, fixed_fused)]
        if any(root.attrs.get("squisher_complete") is not True for root in roots):
            raise ValueError("Parent reference and target grid sources must be complete")
        grids = [ome_spatial_grid(dict(root.attrs), "0") for root in roots]
        if grids[0] is None or grids[0] != grids[1] or grids[0]["axes"] != ["z", "y", "x"]:
            raise ValueError("Parent reference and target must have the same physical grid")
        require_same_multiscale_grid(reference, fixed_fused, tolerance_um=0.0)
    parents = []
    for path in sorted(directory.glob(pattern)):
        row = json.loads(path.read_text())
        if row.get("status") != "accepted" or row.get("moving_tile") != tile_name:
            continue
        if row.get("artifact_type") != "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1":
            raise ValueError(f"unsupported accepted parent artifact {path}")
        if Path(row["fixed_fused"]).resolve() != reference.resolve() or Path(row["moving_path"]).resolve() != moving_path.resolve() or row["moving_channel"] != moving_channel:
            raise ValueError(f"parent {path} source/reference/channel differs from child inputs")
        parents.append((path, row))
    return parents


def _parent_window_path(
    directory: Path, *, tile_name: str, quadrant: str, start_zyx: list[int], shape_zyx: list[int],
    fixed_fused: Path, moving_path: Path, moving_channel: int,
    parent_reference_fused: Path | None = None,
) -> Path:
    start = np.asarray(start_zyx, dtype=np.int64)
    stop = start + np.asarray(shape_zyx, dtype=np.int64)
    matches = []
    for path, row in _accepted_parent_windows(
        directory, tile_name=tile_name, pattern=f"{_safe_stem(tile_name)}.{quadrant}.z*.json",
        fixed_fused=fixed_fused, moving_path=moving_path, moving_channel=moving_channel,
        parent_reference_fused=parent_reference_fused,
    ):
        parent_start = np.asarray(row["moving_start_l0_zyx"], dtype=np.int64)
        parent_stop = np.asarray(row["moving_stop_l0_zyx"], dtype=np.int64)
        if np.all(start >= parent_start) and np.all(stop <= parent_stop):
            if np.array_equal(start, parent_start) and np.array_equal(stop, parent_stop):
                raise ValueError(f"child domain must be smaller than parent {path}")
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"child {tile_name} {start.tolist()}:{stop.tolist()} requires exactly one accepted parent; found {matches}")
    return matches[0]


def _parent_windows(
    directory: Path, *, tile_name: str, tile_shape_zyx: tuple[int, int, int],
    core_shape_zyx: tuple[int, int, int], window_shape_zyx: tuple[int, int, int],
    fixed_fused: Path, moving_path: Path, moving_channel: int,
    parent_reference_fused: Path | None = None,
) -> list[dict[str, Any]]:
    """Subdivide accepted parent domains axially, retaining their exact XY bounds."""
    if not 0 < core_shape_zyx[0] <= window_shape_zyx[0]:
        raise ValueError("parent subdivision requires positive axial core <= window to preserve coverage")
    windows = []
    identities = set()
    for path, row in _accepted_parent_windows(
        directory, tile_name=tile_name, pattern=f"{_safe_stem(tile_name)}.qy*_qx*.z*.json",
        fixed_fused=fixed_fused, moving_path=moving_path, moving_channel=moving_channel,
        parent_reference_fused=parent_reference_fused,
    ):
        start = np.asarray(row["moving_start_l0_zyx"], dtype=np.int64)
        stop = np.asarray(row["moving_stop_l0_zyx"], dtype=np.int64)
        if start.shape != (3,) or stop.shape != (3,) or np.any(start < 0) or np.any(stop <= start) or np.any(stop > tile_shape_zyx):
            raise ValueError(f"parent {path} is outside native source bounds {tile_shape_zyx}")
        shape = stop - start
        if tuple(shape[1:]) != tuple(window_shape_zyx[1:]):
            raise ValueError(f"parent subdivision requires unchanged XY bounds: {path}")
        if window_shape_zyx[0] >= shape[0]:
            raise ValueError(f"child domain must be axially smaller than parent {path}")
        _, quadrant, parent_z = _window_identity_from_virtual_row({
            "moving_tile": tile_name, "tile": path.stem + ".ome.zarr",
        })
        if parent_z != start[0]:
            raise ValueError(f"parent filename/source start mismatch: {path}")
        for relative_z in _axis_starts(int(shape[0]), core=core_shape_zyx[0], window=window_shape_zyx[0]):
            child_start = [int(start[0]) + relative_z, int(start[1]), int(start[2])]
            identity = (quadrant, child_start[0])
            if identity in identities:
                raise ValueError(f"duplicate child identity for {tile_name} {identity}; overlapping parent grids cannot share an output")
            identities.add(identity)
            windows.append({
                "quadrant": quadrant, "start_zyx": child_start,
                "shape_zyx": list(window_shape_zyx), "parent_path": path,
            })
    return windows


def _initializer_windows(
    directory: Path, *, tile_name: str, tile_shape_zyx: tuple[int, int, int],
    core_shape_zyx: tuple[int, int, int], window_shape_zyx: tuple[int, int, int],
) -> list[dict[str, Any]]:
    """Windows named by per-window initializers, on the regular XY quadrant grid.

    Split-parent children start at parent-relative axial offsets, so their z starts need
    not lie on the regular axial grid.
    """
    y_starts = _axis_starts(tile_shape_zyx[1], core=core_shape_zyx[1], window=window_shape_zyx[1])
    x_starts = _axis_starts(tile_shape_zyx[2], core=core_shape_zyx[2], window=window_shape_zyx[2])
    windows = []
    for path in sorted(directory.glob(f"{_safe_stem(tile_name)}.qy*_qx*.z*.json")):
        _, quadrant, z_start = _window_identity_from_virtual_row({
            "moving_tile": tile_name, "tile": path.stem + ".ome.zarr",
        })
        qy, qx = (int(part[2:]) for part in quadrant.split("_"))
        start = json.loads(path.read_text()).get("moving_start_l0_zyx")
        if (qy >= len(y_starts) or qx >= len(x_starts)
                or start != [z_start, y_starts[qy], x_starts[qx]]
                or not 0 <= z_start <= tile_shape_zyx[0] - window_shape_zyx[0]):
            raise ValueError(f"per-window initializer {path} lies outside the window grid of {tile_name}: start={start}")
        windows.append({"quadrant": quadrant, "start_zyx": start, "shape_zyx": [int(v) for v in window_shape_zyx]})
    return windows


def _build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if getattr(args, "parent_reference_fused", None) is not None and args.parent_window_dir is None:
        raise ValueError("--parent-reference-fused requires --parent-window-dir")
    split_parents = bool(getattr(args, "split_parent_windows", False))
    if split_parents and (args.level0_initializer != "window-parent" or args.parent_window_dir is None):
        raise ValueError("--split-parent-windows requires --parent-window-dir and window-parent initialization")
    tile_filter = None
    if args.tile_filter:
        tile_filter = {part.strip().zfill(3) for part in args.tile_filter.split(",") if part.strip()}
    window_filter = _load_window_filter(args)
    from squisher_lightsheet._backends import fused_fixed_trial as trial
    tasks: list[dict[str, Any]] = []
    records = _load_tiles(Path(args.moving_position), tile_filter)
    source_records = {
        _tile_number(str(record["tile"])): record
        for record in _load_tiles(Path(args.moving_source_position), tile_filter)
    }
    nearest_level2_paths = None
    if args.level0_initializer in {"level2-phaseprimed-tile", "level2-mattes-tile"}:
        all_records = _load_tiles(Path(args.moving_position), None)
        path_for_tile = (
            _level2_mattes_tile_path
            if args.level0_initializer == "level2-mattes-tile"
            else _level2_phaseprimed_tile_path
        )
        nearest_level2_paths = _nearest_level2_phaseprimed_paths(
            level2_dir=Path(args.level2_phaseprimed_dir),
            records=all_records,
            path_for_tile=path_for_tile,
        )
    for record in records:
        tile_no = _tile_number(str(record["tile"]))
        if tile_no not in source_records:
            raise ValueError(f"{args.moving_source_position} has no source record for tile {tile_no}")
        moving_tile = trial.tile_record_from_position_record(
            trial.source_record_with_geometry(source_records[tile_no], record)
        )
        tile_shape = tuple(int(value) for value in moving_tile.shape_zyx)
        windows = (
            _parent_windows(
                Path(args.parent_window_dir), tile_name=str(record["tile"]), tile_shape_zyx=tile_shape,
                core_shape_zyx=args.core_shape_zyx, window_shape_zyx=args.window_shape_zyx,
                fixed_fused=Path(args.fixed_fused), moving_path=moving_tile.path,
                moving_channel=int(args.moving_channel),
                parent_reference_fused=getattr(args, "parent_reference_fused", None),
            )
            if split_parents else
            _initializer_windows(
                Path(args.window_initializer_dir), tile_name=str(record["tile"]), tile_shape_zyx=tile_shape,
                core_shape_zyx=args.core_shape_zyx, window_shape_zyx=args.window_shape_zyx,
            )
            if args.level0_initializer == "window-interpolated" else
            _windows(tile_shape, core_shape_zyx=args.core_shape_zyx, window_shape_zyx=args.window_shape_zyx)
        )
        for window in windows:
            if window_filter is not None and (
                str(record["tile"]),
                str(window["quadrant"]),
                int(window["start_zyx"][0]),
            ) not in window_filter:
                continue
            level2_path, level0_path = _task_paths(args.output_dir, str(record["tile"]), window["quadrant"], window["start_zyx"])
            if args.level0_initializer in {"level2-phaseprimed-tile", "level2-mattes-tile"}:
                assert nearest_level2_paths is not None
                level2_info = nearest_level2_paths[tile_no]
                level2_path = Path(level2_info["path"])
            elif args.level0_initializer == "window-parent":
                level2_info = None
                level2_path = window["parent_path"] if split_parents else _parent_window_path(
                    Path(args.parent_window_dir), tile_name=str(record["tile"]),
                    quadrant=window["quadrant"], start_zyx=window["start_zyx"],
                    shape_zyx=window["shape_zyx"], fixed_fused=Path(args.fixed_fused),
                    moving_path=moving_tile.path, moving_channel=int(args.moving_channel),
                    parent_reference_fused=getattr(args, "parent_reference_fused", None),
                )
            elif args.level0_initializer == "window-interpolated":
                level2_info = None
                level2_path = _window_initializer_path(Path(args.window_initializer_dir), level0_path)
                _validate_window_initializer(
                    level2_path,
                    tile_name=str(record["tile"]),
                    start_zyx=window["start_zyx"],
                )
            else:
                level2_info = None
            expected_initializer_source = level2_path
            if isinstance(level2_info, dict) and bool(level2_info.get("fallback")):
                expected_initializer_source = (
                    level0_path.parent.parent
                    / "level2_json"
                    / f"{level0_path.stem}.nearest_level2_phaseprimed.json"
                )
            if args.resume and not args.level2_only and level0_path.exists():
                try:
                    row = json.loads(level0_path.read_text())
                except json.JSONDecodeError:
                    row = {}
                valid_artifact_types = {
                    "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
                    "lightsheet.image10_to_fused_image14_method8_trial.v1",
                }
                if row.get("artifact_type") in valid_artifact_types and _is_resumable_level0_row(
                    row,
                    args,
                    expected_initializer_source=expected_initializer_source,
                ):
                    continue
            tasks.append(
                {
                    "tile": tile_no,
                    "tile_name": str(record["tile"]),
                    "quadrant": window["quadrant"],
                    "start_zyx": window["start_zyx"],
                    "shape_zyx": window["shape_zyx"],
                    "level2_path": str(level2_path),
                    "level2_fallback": level2_info,
                    "level0_path": str(level0_path),
                }
            )
    if args.max_windows is not None:
        tasks = tasks[: int(args.max_windows)]
    return tasks


def _namespace_for_task(args: dict[str, Any], task: dict[str, Any], *, mode: str, output: Path, level2_result: Path | None, device: int) -> argparse.Namespace:
    return argparse.Namespace(
        fixed_position=Path(args["fixed_position"]),
        moving_position=Path(args["moving_position"]),
        moving_source_position=Path(args["moving_source_position"]),
        fixed_fused=Path(args["fixed_fused"]),
        fixed_level=2 if mode == "level2-metadata" else 0,
        moving_level=2,
        mode=mode,
        level2_result=level2_result,
        tile=task["tile"],
        moving_channel=int(args["moving_channel"]),
        source_cache_max_gib=float(args.get("source_cache_max_gib", 0.0)),
        level0_initializer=str(args["level0_initializer"]),
        level2_phaseprimed_dir=str(Path(args["level2_phaseprimed_dir"]).resolve()),
        window_initializer_dir=None
        if args.get("window_initializer_dir") is None
        else Path(args["window_initializer_dir"]),
        parent_window_dir=None if args.get("parent_window_dir") is None else Path(args["parent_window_dir"]),
        parent_reference_fused=None if args.get("parent_reference_fused") is None else Path(args["parent_reference_fused"]),
        split_parent_windows=bool(args.get("split_parent_windows", False)),
        initializer_sha256=None if level2_result is None else _file_sha256(level2_result),
        moving_start_zyx=tuple(int(value) for value in task["start_zyx"]),
        shape_zyx=tuple(int(value) for value in task["shape_zyx"]),
        fit_downsample_zyx=tuple(int(value) for value in args["fit_downsample_zyx"]),
        native_lib_dir=Path(args["native_lib_dir"]),
        native_method=str(args["native_method"]),
        starting_affine_matrix_zyx=args.get("starting_affine_matrix_zyx"),
        fit_intensity_transform=str(args["fit_intensity_transform"]),
        mattes_bins=int(args["mattes_bins"]),
        mattes_samples=int(args["mattes_samples"]),
        mattes_fixed_shear=bool(args["mattes_fixed_shear"]),
        ftol=float(args["ftol"]),
        max_iterations=int(args["max_iterations"]),
        local_3d_refinement=bool(args["local_3d_refinement"]),
        local_affine_estimator=str(args["local_affine_estimator"]),
        phase_upsample_factor=int(args["phase_upsample_factor"]),
        min_corr=float(args["min_corr"]),
        min_grad_ncc=float(args["min_grad_ncc"]),
        fixed_mask_threshold=args["fixed_mask_threshold"],
        fixed_mask_level=int(args["fixed_mask_level"]),
        fixed_mask_min_voxels=int(args["fixed_mask_min_voxels"]),
        fixed_mask_max_masked_fraction=float(args["fixed_mask_max_masked_fraction"]),
        device=int(device),
        output=output,
    )


def _cache_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    from squisher_lightsheet._backends import fused_fixed_trial as trial
    return {
        "version": trial._cache_version(args),
        **trial._parent_reference_cache_fields(args),
        "level0_initializer": str(args.level0_initializer),
        "level2_phaseprimed_dir": str(Path(args.level2_phaseprimed_dir).resolve()),
        "window_initializer_dir": None
        if getattr(args, "window_initializer_dir", None) is None
        else str(Path(args.window_initializer_dir).resolve()),
        **({"split_parent_windows": True} if getattr(args, "split_parent_windows", False) else {}),
        **({"parent_window_dir": str(Path(args.parent_window_dir).resolve())}
           if getattr(args, "parent_window_dir", None) is not None else {}),
        "fixed_fused": str(Path(args.fixed_fused).resolve()),
        "moving_position": str(Path(args.moving_position).resolve()),
        "moving_source_position": str(Path(args.moving_source_position).resolve()),
        "moving_channel": int(args.moving_channel),
        "native_method": str(args.native_method),
        "starting_affine_matrix_zyx": None
        if args.starting_affine_matrix_zyx is None
        else [float(value) for value in args.starting_affine_matrix_zyx],
        "fit_intensity_transform": str(args.fit_intensity_transform),
        "mattes_bins": int(args.mattes_bins),
        "mattes_samples": int(args.mattes_samples),
        "mattes_fixed_shear": bool(args.mattes_fixed_shear),
        "fit_downsample_zyx": [int(value) for value in args.fit_downsample_zyx],
        "native_lib_dir": str(Path(args.native_lib_dir).resolve()),
        "ftol": float(args.ftol),
        "max_iterations": int(args.max_iterations),
        **({"local_3d_refinement": True} if args.local_3d_refinement else {}),
        **(local_affine_cache_config(args.local_affine_estimator) if args.local_3d_refinement else {}),
        "phase_upsample_factor": int(args.phase_upsample_factor),
        "min_corr": float(args.min_corr),
        "min_grad_ncc": float(args.min_grad_ncc),
        "fixed_mask_threshold": None if args.fixed_mask_threshold is None else float(args.fixed_mask_threshold),
        "fixed_mask_level": int(args.fixed_mask_level),
        "fixed_mask_min_voxels": int(args.fixed_mask_min_voxels),
        "fixed_mask_max_masked_fraction": float(args.fixed_mask_max_masked_fraction),
    }


def _is_resumable_level0_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    expected_initializer_source: Path | None = None,
) -> bool:
    cache = row.get("cache_config")
    if not isinstance(cache, dict):
        return False
    current = _cache_config_from_args(args)
    if cache.get("version") != current["version"]:
        return False
    cached_initializer_sha256 = cache.get("initializer_sha256")
    cache_without_initializer = {
        key: value for key, value in cache.items() if key != "initializer_sha256"
    }
    if cache_without_initializer != current:
        return False
    if str(args.level0_initializer) not in {
        "level2-phaseprimed-tile",
        "level2-mattes-tile",
        "window-interpolated",
        "window-parent",
    }:
        return True
    initializer_source = row.get("initializer_source")
    if not isinstance(initializer_source, str):
        return False
    source = Path(initializer_source).resolve()
    if expected_initializer_source is not None:
        return (
            source == expected_initializer_source.resolve()
            and source.is_file()
            and isinstance(cached_initializer_sha256, str)
            and cached_initializer_sha256 == _file_sha256(source)
        )
    if str(args.level0_initializer) == "window-parent":
        return source.is_relative_to(Path(args.parent_window_dir).resolve())
    if str(args.level0_initializer) == "window-interpolated":
        return source.is_relative_to(Path(args.window_initializer_dir).resolve())
    return source.is_relative_to(Path(args.level2_phaseprimed_dir).resolve()) or source.is_relative_to(
        (Path(args.output_dir) / "level2_json").resolve()
    )


def _worker_counts_by_device(total_workers: int, devices: tuple[int, ...]) -> dict[int, int]:
    if total_workers < 1:
        raise ValueError("total_workers must be >= 1")
    counts = {device: 0 for device in devices}
    for index in range(total_workers):
        counts[devices[index % len(devices)]] += 1
    return {device: count for device, count in counts.items() if count > 0}


def _assign_tile_affinity_devices(
    tasks: list[dict[str, Any]], devices: tuple[int, ...]
) -> None:
    """Keep a moving tile on one device so its worker-local host cache is reused."""
    task_counts = {device: 0 for device in devices}
    device_by_tile: dict[str, int] = {}
    for task in tasks:
        tile = str(task["tile"])
        if tile not in device_by_tile:
            device_by_tile[tile] = min(devices, key=lambda device: task_counts[device])
        device = device_by_tile[tile]
        task["device"] = device
        task_counts[device] += 1


def _task_waves(
    tasks_by_device: dict[int, list[dict[str, Any]]],
    worker_counts: dict[int, int],
    *,
    max_tasks_per_worker: int,
) -> list[dict[int, list[dict[str, Any]]]]:
    """Partition work so each executor generation has a bounded task count."""
    offsets = {device: 0 for device in tasks_by_device}
    waves: list[dict[int, list[dict[str, Any]]]] = []
    while True:
        wave: dict[int, list[dict[str, Any]]] = {}
        for device, device_tasks in tasks_by_device.items():
            start = offsets[device]
            stop = min(start + worker_counts[device] * max_tasks_per_worker, len(device_tasks))
            if start < stop:
                wave[device] = device_tasks[start:stop]
                offsets[device] = stop
        if not wave:
            return waves
        waves.append(wave)


def _init_cuda_worker(device: int) -> None:
    device_index = int(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_devices = [value.strip() for value in visible.split(",") if value.strip()]
        if not 0 <= device_index < len(visible_devices):
            raise ValueError(
                f"CUDA device {device_index} is outside the inherited "
                f"CUDA_VISIBLE_DEVICES mask with {len(visible_devices)} device(s)"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[device_index]
        return
    if device_index < 0:
        raise ValueError(f"CUDA device must be non-negative, got {device_index}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)


def _run_task(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    args, task = payload
    device = int(task["device"])
    from squisher_lightsheet._backends import fused_fixed_trial as trial
    started = time.perf_counter()
    level2_path = Path(task["level2_path"])
    level0_path = Path(task["level0_path"])
    try:
        if args.get("level0_initializer") == "placement-preseed":
            trial.run(
                _namespace_for_task(
                    args,
                    task,
                    mode="level0-preseed",
                    output=level0_path,
                    level2_result=None,
                    device=0,
                )
            )
            row = json.loads(level0_path.read_text())
            fixed_mask = row.get("fixed_threshold_mask")
            return {
                "status": str(row.get("status", "done")),
                "rejection_reason": row.get("rejection_reason"),
                "tile": task["tile_name"],
                "quadrant": task["quadrant"],
                "z": int(task["start_zyx"][0]),
                "start_zyx": task["start_zyx"],
                "device": device,
                "level2_json": None,
                "level0_json": str(level0_path),
                "selected_attempt": "placement_preseed_native_method8",
                "selected_corr": row.get("corr_refined"),
                "selected_grad_ncc": row.get("gradient_component_ncc_mean"),
                "fixed_masked_fraction": None
                if not isinstance(fixed_mask, dict)
                else fixed_mask.get("fit_masked_fraction", fixed_mask.get("masked_fraction")),
                "seconds": None if not isinstance(row.get("timing_seconds"), dict) else row["timing_seconds"].get("total"),
            }
        if args.get("level0_initializer") in {
            "level2-phaseprimed-tile",
            "level2-mattes-tile",
            "window-interpolated",
            "window-parent",
        }:
            if not level2_path.exists():
                row = {
                    "schema_version": 1,
                    "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
                    "status": "rejected",
                    "rejection_reason": "missing_level2_phaseprimed_tile_initializer",
                    "cache_config": args["cache_config"],
                    "fixed_fused": args["fixed_fused"],
                    "fixed_source": "Image_14 fused OME-Zarr",
                    "moving_tile": task["tile_name"],
                    "moving_channel": int(args["moving_channel"]),
                    "moving_start_l0_zyx": task["start_zyx"],
                    "moving_stop_l0_zyx": [
                        int(start) + int(size) for start, size in zip(task["start_zyx"], task["shape_zyx"], strict=True)
                    ],
                    "quadrant": task["quadrant"],
                    "initializer_source": str(level2_path),
                    "timing_seconds": {"total": time.perf_counter() - started},
                }
                _write_json(level0_path, row)
                return {
                    "status": "rejected",
                    "rejection_reason": row.get("rejection_reason"),
                    "tile": task["tile_name"],
                    "quadrant": task["quadrant"],
                    "z": int(task["start_zyx"][0]),
                    "device": device,
                    "level2_json": str(level2_path),
                    "level0_json": str(level0_path),
                    "selected_attempt": None,
                    "selected_corr": None,
                    "selected_grad_ncc": None,
                    "fixed_masked_fraction": None,
                    "seconds": time.perf_counter() - started,
                }
            level2_row = json.loads(level2_path.read_text())
            level2_fallback = task.get("level2_fallback")
            effective_level2_path = level2_path
            if isinstance(level2_fallback, dict) and bool(level2_fallback.get("fallback")):
                level2_row = dict(level2_row)
                level2_row["nearest_level2_phaseprimed_tile_fallback"] = {
                    "status": "nearest_accepted_level2_phaseprimed_tile",
                    "requested_tile": task["tile_name"],
                    "source_tile": level2_fallback.get("source_tile_name"),
                    "source_tile_number": level2_fallback.get("source_tile"),
                    "distance_um_yx": level2_fallback.get("distance_um_yx"),
                    "source_json": str(level2_path.resolve()),
                    "method": "nearest accepted level2 phase-primed tile by stage tile-center distance in YX micrometers",
                }
                effective_level2_path = (
                    Path(task["level0_path"]).parent.parent
                    / "level2_json"
                    / f"{Path(task['level0_path']).stem}.nearest_level2_phaseprimed.json"
                )
                _write_json(effective_level2_path, level2_row)
            if args.get("level0_initializer") == "window-parent":
                required_initializer_keys = ("selected_local_matrix_zyx", "selected_local_translation_zyx", "fixed_start_zyx", "moving_start_l0_zyx", "moving_shape_zyx")
            elif args.get("level0_initializer") in {"level2-mattes-tile", "window-interpolated"}:
                required_initializer_keys = ("level_factor_zyx", "local_matrix_zyx", "local_translation_zyx")
            else:
                required_initializer_keys = (
                    "fixed_phaseprimed_start_level_zyx",
                    "moving_start_level_zyx",
                    "fixed_level_factor_zyx",
                    "moving_level_factor_zyx",
                    "local_matrix_zyx",
                    "local_translation_zyx",
                )
            missing_initializer_keys = [key for key in required_initializer_keys if key not in level2_row]
            if level2_row.get("status") != "accepted" or missing_initializer_keys:
                row = {
                    "schema_version": 1,
                    "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
                    "status": "rejected",
                    "rejection_reason": "unusable_level2_phaseprimed_tile_initializer"
                    if missing_initializer_keys
                    else level2_row.get("rejection_reason"),
                    "missing_initializer_keys": missing_initializer_keys,
                    "cache_config": args["cache_config"],
                    "fixed_fused": args["fixed_fused"],
                    "fixed_source": "Image_14 fused OME-Zarr",
                    "moving_tile": task["tile_name"],
                    "moving_channel": int(args["moving_channel"]),
                    "moving_start_l0_zyx": task["start_zyx"],
                    "moving_stop_l0_zyx": [
                        int(start) + int(size) for start, size in zip(task["start_zyx"], task["shape_zyx"], strict=True)
                    ],
                    "quadrant": task["quadrant"],
                    "initializer_source": str(effective_level2_path.resolve()),
                    "level2_status": level2_row.get("status"),
                    "level2_rejection_reason": level2_row.get("rejection_reason"),
                    "level2_fallback": None if not isinstance(level2_fallback, dict) else level2_fallback,
                    "timing_seconds": {"total": time.perf_counter() - started},
                }
                _write_json(level0_path, row)
                return {
                    "status": "rejected",
                    "rejection_reason": row.get("rejection_reason"),
                    "tile": task["tile_name"],
                    "quadrant": task["quadrant"],
                    "z": int(task["start_zyx"][0]),
                    "device": device,
                    "level2_json": str(effective_level2_path),
                    "level0_json": str(level0_path),
                    "selected_attempt": None,
                    "selected_corr": None,
                    "selected_grad_ncc": None,
                    "fixed_masked_fraction": None,
                    "seconds": time.perf_counter() - started,
                }
            trial.run(
                _namespace_for_task(
                    args,
                    task,
                    mode="level0-from-level2",
                    output=level0_path,
                    level2_result=effective_level2_path,
                    device=0,
                )
            )
            row = json.loads(level0_path.read_text())
            fixed_mask = row.get("fixed_threshold_mask")
            return {
                "status": str(row.get("status", "done")),
                "rejection_reason": row.get("rejection_reason"),
                "tile": task["tile_name"],
                "quadrant": task["quadrant"],
                "z": int(task["start_zyx"][0]),
                "start_zyx": task["start_zyx"],
                "device": device,
                "level2_json": str(effective_level2_path),
                "level0_json": str(level0_path),
                "selected_attempt": row.get("selected_attempt"),
                "selected_corr": row.get("selected_corr_refined"),
                "selected_grad_ncc": row.get("selected_gradient_component_ncc_mean"),
                "fixed_masked_fraction": None
                if not isinstance(fixed_mask, dict)
                else fixed_mask.get("fit_masked_fraction", fixed_mask.get("masked_fraction")),
                "seconds": None if not isinstance(row.get("timing_seconds"), dict) else row["timing_seconds"].get("total"),
                **_refinement_table_values(row),
            }
        level2_current = False
        if level2_path.exists():
            try:
                level2_row = json.loads(level2_path.read_text())
            except json.JSONDecodeError:
                level2_row = {}
            level2_artifact_current = (
                level2_row.get("artifact_type") == "lightsheet.image10_to_fused_image14_method8_level2_metadata_trial.v1"
            )
            level2_current = level2_artifact_current and (
                bool(args.get("reuse_existing_level2"))
                or level2_row.get("cache_config") == args["cache_config"]
            )
        if not level2_current:
            trial.run(_namespace_for_task(args, task, mode="level2-metadata", output=level2_path, level2_result=None, device=0))
        level2_row = json.loads(level2_path.read_text())
        if bool(args.get("level2_only")):
            fixed_mask = level2_row.get("fixed_threshold_mask")
            return {
                "status": str(level2_row.get("status", "done")),
                "rejection_reason": level2_row.get("rejection_reason"),
                "tile": task["tile_name"],
                "quadrant": task["quadrant"],
                "z": int(task["start_zyx"][0]),
                "start_zyx": task["start_zyx"],
                "device": device,
                "level2_json": str(level2_path),
                "level0_json": None,
                "selected_attempt": level2_row.get("native_initializer"),
                "selected_corr": level2_row.get("corr_refined"),
                "selected_grad_ncc": level2_row.get("gradient_component_ncc_mean"),
                "fixed_masked_fraction": None
                if not isinstance(fixed_mask, dict)
                else fixed_mask.get("masked_fraction"),
                "seconds": None
                if not isinstance(level2_row.get("timing_seconds"), dict)
                else level2_row["timing_seconds"].get("total"),
            }
        if level2_row.get("status") == "rejected":
            required_initializer_keys = ("local_matrix_zyx", "local_translation_zyx", "level_factor_zyx")
            missing_initializer_keys = [key for key in required_initializer_keys if key not in level2_row]
            level2_refine_rejection = (
                None
                if level2_row.get("rejection_reason") in LEVEL0_REFINABLE_LEVEL2_REJECTIONS
                else str(level2_row.get("rejection_reason"))
            )
            level2_affine_rejection = None
            if not missing_initializer_keys:
                try:
                    level2_matrix = np.asarray(level2_row["local_matrix_zyx"], dtype=np.float64)
                except (TypeError, ValueError):
                    level2_affine_rejection = "invalid_level2_affine"
                else:
                    level2_affine_rejection = trial._affine_quality_rejection(
                        trial._affine_quality(level2_matrix)
                    )
            if level2_refine_rejection is not None or level2_affine_rejection is not None or missing_initializer_keys:
                row = {
                    "schema_version": 1,
                    "artifact_type": "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
                    "status": "rejected",
                    "rejection_reason": (
                        level2_refine_rejection
                        or level2_affine_rejection
                        or "rejected_level2_missing_initializer"
                    ),
                    "missing_initializer_keys": missing_initializer_keys,
                    "cache_config": args["cache_config"],
                    "fixed_fused": args["fixed_fused"],
                    "fixed_source": "Image_14 fused OME-Zarr",
                    "moving_tile": task["tile_name"],
                    "moving_channel": int(args["moving_channel"]),
                    "moving_start_l0_zyx": task["start_zyx"],
                    "moving_stop_l0_zyx": [int(start) + int(size) for start, size in zip(task["start_zyx"], task["shape_zyx"], strict=True)],
                    "quadrant": task["quadrant"],
                    "initializer_source": str(level2_path.resolve()),
                    "level2_status": "rejected",
                    "fixed_start_zyx": level2_row.get("fixed_start_zyx"),
                    "fixed_stop_zyx": level2_row.get("fixed_stop_zyx"),
                    "fixed_shape_zyx": level2_row.get("fixed_shape_zyx"),
                    "fixed_level_shape_zyx": level2_row.get("fixed_level_shape_zyx"),
                    "fixed_threshold_mask": level2_row.get("fixed_threshold_mask"),
                    "quality_mask": level2_row.get("quality_mask"),
                    "timing_seconds": {"total": time.perf_counter() - started},
                }
                _write_json(level0_path, row)
                return {
                    "status": "rejected",
                    "rejection_reason": row.get("rejection_reason"),
                    "tile": task["tile_name"],
                    "quadrant": task["quadrant"],
                    "z": int(task["start_zyx"][0]),
                    "device": device,
                    "level2_json": str(level2_path),
                    "level0_json": str(level0_path),
                    "selected_attempt": None,
                    "selected_corr": None,
                    "selected_grad_ncc": None,
                    "fixed_masked_fraction": None
                    if not isinstance(row.get("fixed_threshold_mask"), dict)
                    else row["fixed_threshold_mask"].get("masked_fraction"),
                    "seconds": time.perf_counter() - started,
                }
        trial.run(
            _namespace_for_task(
                args,
                task,
                mode="level0-from-level2",
                output=level0_path,
                level2_result=level2_path,
                device=0,
            )
        )
        row = json.loads(level0_path.read_text())
        status = str(row.get("status", "done"))
        return {
            "status": status,
            "rejection_reason": row.get("rejection_reason"),
            "tile": task["tile_name"],
            "quadrant": task["quadrant"],
            "z": int(task["start_zyx"][0]),
            "device": device,
            "level2_json": str(level2_path),
            "level0_json": str(level0_path),
            "selected_attempt": row.get("selected_attempt"),
            "selected_corr": row.get("selected_corr_refined"),
            "selected_grad_ncc": row.get("selected_gradient_component_ncc_mean"),
            "fixed_masked_fraction": None
            if not isinstance(row.get("fixed_threshold_mask"), dict)
            else row["fixed_threshold_mask"].get("fit_masked_fraction"),
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        error = {
            "status": "error",
            "tile": task["tile_name"],
            "quadrant": task["quadrant"],
            "z": int(task["start_zyx"][0]),
            "device": device,
            "error": f"{type(exc).__name__}: {exc}",
            "level2_json": str(level2_path),
            "level0_json": str(level0_path),
            "seconds": time.perf_counter() - started,
        }
        _write_json(level0_path, error)
        return error


def _refinement_table_values(row: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "refined_translation_zyx": None,
        "refined_rotation_deg_zyx": None,
        "refined_stretch_shear_zy_zx_yx": None,
    }
    attempts = row.get("method8_attempts")
    if not isinstance(attempts, list):
        return empty
    native = next(
        (
            attempt
            for attempt in attempts
            if isinstance(attempt, dict)
            and str(attempt.get("name", "")).startswith("native_")
            and "_from_" in str(attempt["name"])
        ),
        None,
    )
    if native is None:
        return empty
    matrix = np.asarray(native.get("local_matrix_zyx"), dtype=np.float64)
    translation = np.asarray(native.get("local_translation_zyx"), dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or translation.shape != (3,)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(translation))
    ):
        return empty
    values = {**empty, "refined_translation_zyx": translation.tolist()}
    if np.linalg.det(matrix) <= 0:
        return values
    rotation, stretch = polar(matrix)
    return {
        **values,
        "refined_rotation_deg_zyx": np.rad2deg(
            Rotation.from_matrix(rotation).as_rotvec()
        ).tolist(),
        "refined_stretch_shear_zy_zx_yx": [
            float(stretch[0, 1]),
            float(stretch[0, 2]),
            float(stretch[1, 2]),
        ],
    }


def _table_value(value: Any) -> str:
    return "null" if value is None else f"{float(value):.8g}"


def _refinement_progress(row: dict[str, Any]) -> str:
    translation = row.get("refined_translation_zyx") or (None, None, None)
    rotation = row.get("refined_rotation_deg_zyx") or (None, None, None)
    shear = row.get("refined_stretch_shear_zy_zx_yx") or (None, None, None)
    start = row.get("start_zyx") or (row.get("z"), None, None)
    return (
        f"| {_tile_number(str(row['tile']))} | {start[0]} | {start[1]} | {start[2]} "
        f"| {row.get('status')} | {row.get('selected_attempt')} "
        f"| {_table_value(translation[0])} | {_table_value(translation[1])} | {_table_value(translation[2])} "
        f"| {_table_value(rotation[0])} | {_table_value(rotation[1])} | {_table_value(rotation[2])} "
        f"| {_table_value(shear[0])} | {_table_value(shear[1])} | {_table_value(shear[2])} |"
    )


def _progress(row: dict[str, Any], *, native_method: str) -> str:
    prefix = f"fused-fixed-{native_method}"
    if row.get("status") == "error":
        return (
            f"{prefix} {row['tile']} {row['quadrant']} z={row['z']} "
            f"status=error device={row['device']} reason={row.get('error')}"
        )
    corr = row.get("selected_corr")
    grad = row.get("selected_grad_ncc")
    masked = row.get("fixed_masked_fraction")
    reason = row.get("rejection_reason")
    return (
        f"{prefix} {row['tile']} {row['quadrant']} z={row['z']} status={row.get('status')} "
        f"device={row['device']} selected={row.get('selected_attempt')} "
        f"corr={'nan' if corr is None else f'{float(corr):.4f}'} "
        f"grad_ncc={'nan' if grad is None else f'{float(grad):.4f}'} "
        f"masked={'nan' if masked is None else f'{100.0 * float(masked):.4f}%'} "
        f"reason={reason} "
        f"seconds={float(row.get('seconds', 0.0)):.2f}"
    )


def _summary_row_from_level0_path(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if payload.get("artifact_type") not in {
        "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1",
        "lightsheet.image10_to_fused_image14_method8_trial.v1",
    }:
        return None
    parts = path.stem.split(".")
    quadrant = payload.get("quadrant")
    if quadrant is None and len(parts) >= 3:
        quadrant = parts[-2]
    z_value = None
    if len(parts) >= 1 and parts[-1].startswith("z"):
        try:
            z_value = int(parts[-1][1:])
        except ValueError:
            z_value = None
    fixed_mask = payload.get("fixed_threshold_mask")
    return {
        "status": payload.get("status", "done"),
        "rejection_reason": payload.get("rejection_reason"),
        "tile": payload.get("moving_tile", ".".join(parts[:3])),
        "quadrant": quadrant,
        "z": z_value,
        "start_zyx": payload.get("moving_start_l0_zyx"),
        "device": None,
        "level2_json": payload.get("initializer_source"),
        "level0_json": str(path),
        "selected_attempt": payload.get("selected_attempt", "placement_preseed_native_method8"),
        "selected_corr": payload.get("selected_corr_refined", payload.get("corr_refined")),
        "selected_grad_ncc": payload.get("selected_gradient_component_ncc_mean", payload.get("gradient_component_ncc_mean")),
        "fixed_masked_fraction": None if not isinstance(fixed_mask, dict) else fixed_mask.get("fit_masked_fraction", fixed_mask.get("masked_fraction")),
        "seconds": None if not isinstance(payload.get("timing_seconds"), dict) else payload["timing_seconds"].get("total"),
        **_refinement_table_values(payload),
    }


def _cached_summary_rows(output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "window_json").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        expected_initializer_source = (
            _window_initializer_path(Path(args.window_initializer_dir), path)
            if str(args.level0_initializer) == "window-interpolated"
            else None
        )
        if (
            payload.get("artifact_type") == "lightsheet.image10_to_fused_image14_method8_level0_from_level2_trial.v1"
            and _is_resumable_level0_row(
                payload,
                args,
                expected_initializer_source=expected_initializer_source,
            )
        ):
            row = _summary_row_from_level0_path(path)
            if row is not None:
                rows.append(row)
    return rows


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    from squisher_lightsheet._backends import fused_fixed_trial as trial

    accepted = [row for row in rows if row.get("status") == "accepted"]
    grad = np.asarray([row["selected_grad_ncc"] for row in accepted if row.get("selected_grad_ncc") is not None], dtype=np.float64)
    corr = np.asarray([row["selected_corr"] for row in accepted if row.get("selected_corr") is not None], dtype=np.float64)
    summary = {
        "artifact_type": "lightsheet.image10_to_fused_image14_method8_sweep_summary.v1",
        **trial._parent_reference_cache_fields(args),
        **({"split_parent_windows": True} if getattr(args, "split_parent_windows", False) else {}),
        **({"parent_window_dir": str(Path(args.parent_window_dir).resolve())}
           if getattr(args, "parent_window_dir", None) is not None else {}),
        "fixed_fused": str(Path(args.fixed_fused).resolve()),
        "moving_position": str(Path(args.moving_position).resolve()),
        "moving_source_position": str(Path(args.moving_source_position).resolve()),
        "level2_phaseprimed_dir": str(Path(args.level2_phaseprimed_dir).resolve()),
        "window_initializer_dir": None
        if getattr(args, "window_initializer_dir", None) is None
        else str(Path(args.window_initializer_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "level0_initializer": str(args.level0_initializer),
        "core_shape_zyx": [int(value) for value in args.core_shape_zyx],
        "window_shape_zyx": [int(value) for value in args.window_shape_zyx],
        "overlap_policy": "core_shape_zyx steps with window_shape_zyx context",
        "moving_channel": int(args.moving_channel),
        "native_method": str(args.native_method),
        "starting_affine_matrix_zyx": None
        if args.starting_affine_matrix_zyx is None
        else [float(value) for value in args.starting_affine_matrix_zyx],
        "cache_config": _cache_config_from_args(args),
        "quality_settings": {
            "min_corr": float(args.min_corr),
            "min_grad_ncc": float(args.min_grad_ncc),
            "fixed_mask_threshold": None if args.fixed_mask_threshold is None else float(args.fixed_mask_threshold),
            "fixed_mask_level": int(args.fixed_mask_level),
            "fixed_mask_min_voxels": int(args.fixed_mask_min_voxels),
            "fixed_mask_max_masked_fraction": float(args.fixed_mask_max_masked_fraction),
        },
        "workers": int(args.workers),
        "devices": list(_parse_devices(args.devices)),
        "aggregate": {
            "window_count": len(rows),
            "accepted_window_count": len(accepted),
            "rejected_window_count": sum(1 for row in rows if row.get("status") == "rejected"),
            "error_window_count": sum(1 for row in rows if row.get("status") == "error"),
            "selected_corr_median": None if corr.size == 0 else float(np.median(corr)),
            "selected_grad_ncc_median": None if grad.size == 0 else float(np.median(grad)),
        },
        "windows": rows,
    }
    path = output_dir / "fused_fixed_method8_summary.json"
    _write_json(path, summary)
    return path


def run(args: argparse.Namespace) -> Path:
    if args.local_affine_estimator != "least-squares" and not args.local_3d_refinement:
        raise ValueError("restrained estimator requires --local-3d-refinement")
    from squisher_lightsheet._backends import fused_fixed_trial as trial
    trial._apply_native_method_defaults(args)
    missing = [name for name in ("CUDA_PATH", "LD_LIBRARY_PATH") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required native/GPU env vars: {', '.join(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks(args)
    devices = _parse_devices(args.devices)
    worker_counts = _worker_counts_by_device(int(args.workers), devices)
    active_devices = tuple(worker_counts)
    if str(args.level0_initializer) == "window-interpolated":
        _assign_tile_affinity_devices(tasks, active_devices)
    else:
        for index, task in enumerate(tasks):
            task["device"] = active_devices[index % len(active_devices)]
    args_payload = {
        "fixed_position": str(args.fixed_position),
        "moving_position": str(args.moving_position),
        "moving_source_position": str(args.moving_source_position),
        "fixed_fused": str(args.fixed_fused),
        "level2_phaseprimed_dir": str(args.level2_phaseprimed_dir),
        "window_initializer_dir": None
        if getattr(args, "window_initializer_dir", None) is None
        else str(args.window_initializer_dir),
        "parent_window_dir": None if getattr(args, "parent_window_dir", None) is None else str(args.parent_window_dir),
        "parent_reference_fused": None if getattr(args, "parent_reference_fused", None) is None else str(args.parent_reference_fused),
        "split_parent_windows": bool(getattr(args, "split_parent_windows", False)),
        "cache_config": _cache_config_from_args(args),
        "level0_initializer": str(args.level0_initializer),
        "local_3d_refinement": bool(args.local_3d_refinement),
        "local_affine_estimator": str(args.local_affine_estimator),
        "moving_channel": int(args.moving_channel),
        "source_cache_max_gib": float(getattr(args, "source_cache_max_gib", 0.0)),
        "native_method": str(args.native_method),
        "starting_affine_matrix_zyx": None
        if args.starting_affine_matrix_zyx is None
        else [float(value) for value in args.starting_affine_matrix_zyx],
        "fit_intensity_transform": str(args.fit_intensity_transform),
        "mattes_bins": int(args.mattes_bins),
        "mattes_samples": int(args.mattes_samples),
        "mattes_fixed_shear": bool(args.mattes_fixed_shear),
        "fit_downsample_zyx": list(args.fit_downsample_zyx),
        "native_lib_dir": str(args.native_lib_dir),
        "ftol": float(args.ftol),
        "max_iterations": int(args.max_iterations),
        "phase_upsample_factor": int(args.phase_upsample_factor),
        "min_corr": float(args.min_corr),
        "min_grad_ncc": float(args.min_grad_ncc),
        "fixed_mask_threshold": None if args.fixed_mask_threshold is None else float(args.fixed_mask_threshold),
        "fixed_mask_level": int(args.fixed_mask_level),
        "fixed_mask_min_voxels": int(args.fixed_mask_min_voxels),
        "fixed_mask_max_masked_fraction": float(args.fixed_mask_max_masked_fraction),
        "reuse_existing_level2": bool(getattr(args, "reuse_existing_level2", False)),
        "level2_only": bool(getattr(args, "level2_only", False)),
    }
    rows: list[dict[str, Any]] = (
        _cached_summary_rows(args.output_dir, args)
        if args.resume and not bool(getattr(args, "level2_only", False))
        else []
    )
    print(
        f"fused-fixed-{args.native_method} start windows={len(tasks)} workers={args.workers} "
        f"devices={list(active_devices)} "
        f"core={args.core_shape_zyx} window={args.window_shape_zyx}",
        flush=True,
    )
    refinement_mode = str(args.level0_initializer) == "window-interpolated"
    if refinement_mode:
        print(
            "Transform space: local level-0 moving tile -> fixed fused center model; "
            "translation is in fixed level-0 ZYX pixels; rotation is the polar SO(3) rotation vector in degrees; "
            "shear is from the symmetric polar stretch tensor.\n",
            flush=True,
        )
        print(
            "| tile | start_z | start_y | start_x | status | selected_attempt | "
            "translation_z_px | translation_y_px | translation_x_px | "
            "rotation_z_deg | rotation_y_deg | rotation_x_deg | "
            "stretch_shear_zy | stretch_shear_zx | stretch_shear_yx |",
            flush=True,
        )
        print(
            "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            flush=True,
        )
    if int(args.workers) == 1:
        _init_cuda_worker(active_devices[0])
        for task in tasks:
            row = _run_task((args_payload, task))
            rows.append(row)
            print(
                _refinement_progress(row)
                if refinement_mode
                else _progress(row, native_method=str(args.native_method)),
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        tasks_by_device: dict[int, list[dict[str, Any]]] = {device: [] for device in active_devices}
        for task in tasks:
            tasks_by_device[int(task["device"])].append(task)
        waves = _task_waves(
            tasks_by_device,
            worker_counts,
            max_tasks_per_worker=int(args.max_tasks_per_worker),
        )
        for wave_index, wave in enumerate(waves, start=1):
            print(
                f"fused-fixed-{args.native_method} worker-wave {wave_index}/{len(waves)} "
                f"windows={sum(len(device_tasks) for device_tasks in wave.values())}",
                flush=True,
            )
            with ExitStack() as stack:
                executors = {
                    device: stack.enter_context(
                        ProcessPoolExecutor(
                            max_workers=worker_counts[device],
                            mp_context=context,
                            initializer=_init_cuda_worker,
                            initargs=(device,),
                        )
                    )
                    for device in wave
                }
                futures = [
                    executors[device].submit(_run_task, (args_payload, task))
                    for device, device_tasks in wave.items()
                    for task in device_tasks
                ]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    print(
                        _refinement_progress(row)
                        if refinement_mode
                        else _progress(row, native_method=str(args.native_method)),
                        flush=True,
                    )
    if int(args.workers) == 1 and float(getattr(args, "source_cache_max_gib", 0.0)) > 0:
        source_cache = trial._moving_source_cache(float(args.source_cache_max_gib))
        assert source_cache is not None
        cache = source_cache.cache
        print(
            f"moving-source-cache hits={cache.hits} misses={cache.misses} "
            f"evictions={cache.evictions} retained_bytes={cache.total_bytes} "
            f"loaded_bytes={cache.loaded_bytes}",
            flush=True,
        )
    return _write_summary(args.output_dir, rows, args).resolve()


def build_parser() -> argparse.ArgumentParser:
    from squisher_lightsheet._backends import fused_fixed_trial as trial
    parser = argparse.ArgumentParser(description="Sweep Image10 chunks against fused Image14 using metadata-seeded level2 then level0 method8/phase candidates.")
    parser.add_argument("--fixed-position", type=Path, default=trial.DEFAULT_FIXED_POSITION)
    parser.add_argument("--moving-position", type=Path, default=trial.DEFAULT_MOVING_POSITION)
    parser.add_argument("--moving-source-position", type=Path, default=trial.DEFAULT_MOVING_POSITION)
    parser.add_argument("--fixed-fused", type=Path, default=trial.DEFAULT_FIXED_FUSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--core-shape-zyx", type=_parse_zyx, default=(480, 480, 480))
    parser.add_argument("--window-shape-zyx", type=_parse_zyx, default=(528, 528, 528))
    parser.add_argument("--fit-downsample-zyx", type=_parse_zyx, default=(1, 1, 1))
    parser.add_argument("--moving-channel", type=int, default=0)
    parser.add_argument("--source-cache-max-gib", type=float, default=0.0)
    parser.add_argument("--native-lib-dir", type=Path, default=trial.DEFAULT_LIB_DIR)
    parser.add_argument(
        "--native-method",
        choices=("method6", "method8", "method10-mattes", "method11-mattes"),
        default="method8",
    )
    parser.add_argument(
        "--starting-affine-matrix-zyx",
        type=trial._parse_starting_affine_matrix_zyx,
        default=None,
        help=(
            "Optional 3x3 row-major moving-to-fixed matrix in the local centered ZYX model. "
            "Defaults to the accepted slab-pivot trial median and replaces only the starting linear block; "
            "placement/phase continues to own translation."
        ),
    )
    parser.add_argument(
        "--fit-intensity-transform",
        choices=("linear", "log1p"),
        default="linear",
    )
    parser.add_argument("--mattes-bins", type=int, default=50)
    parser.add_argument("--mattes-samples", type=int, default=100000)
    parser.add_argument("--mattes-fixed-shear", action="store_true")
    parser.add_argument("--ftol", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--local-3d-refinement", action="store_true")
    parser.add_argument("--local-affine-estimator", choices=("least-squares", "restrained"), default="least-squares")
    parser.add_argument("--phase-upsample-factor", type=int, default=10)
    parser.add_argument("--min-corr", type=float, default=0.15)
    parser.add_argument("--min-grad-ncc", type=float, default=0.24)
    parser.add_argument("--fixed-mask-threshold", type=float, default=3000.0)
    parser.add_argument("--fixed-mask-level", type=int, default=2)
    parser.add_argument("--fixed-mask-min-voxels", type=int, default=256)
    parser.add_argument("--fixed-mask-max-masked-fraction", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tasks-per-worker", type=int, default=25)
    parser.add_argument(
        "--devices",
        default="0",
        help=(
            "Comma-separated local CUDA indices when CUDA_VISIBLE_DEVICES is set; "
            "physical indices otherwise."
        ),
    )
    parser.add_argument("--tile-filter", default=None)
    parser.add_argument("--window-filter-json", type=Path, default=None)
    parser.add_argument("--window-filter-transform-source", default=None)
    parser.add_argument(
        "--reuse-existing-level2",
        action="store_true",
        help="Reuse existing metadata-level-2 artifacts even when level-0 cache settings changed.",
    )
    parser.add_argument(
        "--level2-only",
        action="store_true",
        help="Generate coarse per-window candidates without reading or fitting level 0.",
    )
    parser.add_argument(
        "--level0-initializer",
        choices=(
            "level2-method8",
            "placement-preseed",
            "level2-phaseprimed-tile",
            "level2-mattes-tile",
            "window-interpolated",
            "window-parent",
        ),
        default="level2-method8",
    )
    parser.add_argument("--level2-phaseprimed-dir", type=Path, default=DEFAULT_LEVEL2_PHASEPRIMED_DIR)
    parser.add_argument("--window-initializer-dir", type=Path, default=None)
    parser.add_argument("--parent-window-dir", type=Path, default=None)
    parser.add_argument("--parent-reference-fused", type=Path, default=None)
    parser.add_argument("--split-parent-windows", action="store_true", help="Subdivide each accepted parent axially instead of using the native tile grid.")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main() -> None:
    from squisher_lightsheet._backends import fused_fixed_trial as trial
    args = build_parser().parse_args()
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    if int(args.max_tasks_per_worker) < 1:
        raise SystemExit("--max-tasks-per-worker must be >= 1")
    if not np.isfinite(args.source_cache_max_gib) or args.source_cache_max_gib < 0:
        raise SystemExit("--source-cache-max-gib must be finite and nonnegative")
    if args.level0_initializer == "window-parent" and args.parent_window_dir is None:
        raise SystemExit("--parent-window-dir is required for --level0-initializer window-parent")
    if args.level0_initializer == "window-interpolated" and args.window_initializer_dir is None:
        raise SystemExit("--window-initializer-dir is required for --level0-initializer window-interpolated")
    if args.mattes_fixed_shear and args.native_method not in trial.MATTES_NATIVE_METHODS:
        raise SystemExit("--mattes-fixed-shear requires --native-method method10-mattes or method11-mattes")
    trial._resolve_starting_affine_matrix(args.starting_affine_matrix_zyx, np.eye(3, dtype=np.float32))
    summary = run(args)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
