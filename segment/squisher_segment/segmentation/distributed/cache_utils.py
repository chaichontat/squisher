from __future__ import annotations

import json
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
    path.write_text(json.dumps(payload, indent=2))


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
) -> None:
    payload = {"input_key": input_key, "channels": normalization}
    path.write_text(json.dumps(payload, indent=2, cls=NumpyEncoder))
