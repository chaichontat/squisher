from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


def _stat_fingerprint(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zarr_store_name(tile_name: str) -> str:
    name = Path(tile_name).name
    for suffix in (".ome.tif", ".ome.tiff"):
        if name.endswith(suffix):
            return f"{name[: -len(suffix)]}.ome.zarr"
    return name


def registration_input_fingerprint(position_json: Path, zarr_dir: Path) -> dict[str, Any]:
    """Fingerprint registration placement plus each level-0 store boundary."""
    position_bytes = position_json.read_bytes()
    payload = json.loads(position_bytes)
    tiles = []
    for record in payload.get("tiles", []):
        tile_path = zarr_dir / _zarr_store_name(str(record["tile"]))
        level0_path = tile_path / "0"
        tiles.append(
            {
                "tile": str(record["tile"]),
                "store": _stat_fingerprint(tile_path),
                "store_metadata": _stat_fingerprint(tile_path / "zarr.json"),
                "level0": _stat_fingerprint(level0_path),
                "level0_metadata": _stat_fingerprint(level0_path / "zarr.json"),
            }
        )
    return {
        "position_json": _stat_fingerprint(position_json),
        "position_json_sha256": hashlib.sha256(position_bytes).hexdigest(),
        "tiles": tiles,
    }


def write_text_set_atomic(files: Mapping[Path, str]) -> None:
    """Publish an absent set of text artifacts, rolling back partial promotion."""
    staged = {path: path.with_name(f".{path.name}.tmp") for path in files}
    occupied = [path for path in (*files, *staged.values()) if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite existing artifact(s): {occupied}")
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)

    promoted: list[Path] = []
    try:
        for path, text in files.items():
            staged[path].write_text(text)
        for path in files:
            staged[path].replace(path)
            promoted.append(path)
    except BaseException as error:
        cleanup_errors: list[BaseException] = []
        for path in reversed(promoted):
            try:
                path.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        for path in staged.values():
            try:
                path.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise BaseExceptionGroup("artifact publication and rollback failed", [error, *cleanup_errors])
        raise
