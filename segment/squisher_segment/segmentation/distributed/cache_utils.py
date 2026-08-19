from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def atomic_write_text(path: Path, text: str) -> None:
    """Durably replace a small metadata file without exposing partial JSON."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".tmp-",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_nonempty_cache(
    path: Path,
    blocksize: tuple[int, ...],
    run_key: str,
) -> list[int] | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text())
        if tuple(parsed.get("block_size", ())) != tuple(blocksize):
            return None
        if parsed.get("run_key") != run_key:
            return None
        return [int(i) for i in parsed["idxs"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def write_nonempty_cache(
    path: Path,
    blocksize: tuple[int, ...],
    run_key: str,
    idxs: list[int],
) -> None:
    payload = {"idxs": idxs, "block_size": list(blocksize), "run_key": run_key}
    atomic_write_text(path, json.dumps(payload, indent=2))


def read_normalization_cache(path: Path, input_key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text())
        if parsed.get("input_key") != input_key:
            return None
        channels = parsed["channels"]
        return channels if isinstance(channels, dict) else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def write_normalization_cache(
    path: Path,
    input_key: str,
    normalization: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
) -> None:
    payload = {"input_key": input_key, "channels": normalization}
    if settings is not None:
        payload["settings"] = settings
    atomic_write_text(path, json.dumps(payload, indent=2, cls=NumpyEncoder))
