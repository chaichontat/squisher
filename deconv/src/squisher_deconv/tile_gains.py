"""Validated per-source, per-channel multipliers applied after BaSiC correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def validate_channel_gains(values: Sequence[float], *, channels: int) -> np.ndarray:
    if len(values) != channels or any(isinstance(v, (bool, str)) for v in values):
        raise ValueError(f"Tile gains must contain exactly {channels} numeric channel values.")
    gains = np.asarray(values, dtype=np.float32)
    if gains.shape != (channels,) or not np.all(np.isfinite(gains)) or np.any(gains <= 0):
        raise ValueError("Tile gains must be finite and strictly positive.")
    return gains


def load_tile_gains(path: Path, *, inputs: Sequence[Path], channels: int) -> dict[str, tuple[float, ...]]:
    """Require exact source coverage so a missing tile cannot silently use gain 1."""
    payload = json.loads(path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("channels") != channels
    ):
        raise ValueError(f"Invalid tile-gain schema or channel count in {path}.")
    rows = payload.get("tiles")
    if not isinstance(rows, list):
        raise ValueError(f"Tile-gain manifest {path} requires a tiles list.")
    result = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("source"), str)
            or not isinstance(row.get("gains"), list)
        ):
            raise ValueError(f"Each tile in {path} requires source and gains.")
        source = Path(row["source"])
        if not source.is_absolute():
            raise ValueError(f"Tile-gain source must be absolute: {source}")
        key = str(source.resolve())
        if key in result:
            raise ValueError(f"Duplicate tile-gain source: {key}")
        result[key] = tuple(float(v) for v in validate_channel_gains(row["gains"], channels=channels))
    expected = {str(Path(p).resolve()) for p in inputs}
    if set(result) != expected:
        raise ValueError(
            f"Tile-gain sources differ from inputs: missing={sorted(expected - set(result))}, unexpected={sorted(set(result) - expected)}"
        )
    return result


def write_tile_gains(path: Path, *, sources: Sequence[Path], gains: np.ndarray) -> Path:
    """Write the strict source-keyed schema consumed by deconvolution."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite tile-gain manifest: {path}")
    resolved = [Path(source).resolve() for source in sources]
    if len({str(source) for source in resolved}) != len(resolved):
        raise ValueError("Tile-gain sources must resolve to unique paths.")
    values = np.asarray(gains)
    if values.ndim != 2 or values.shape[0] != len(resolved):
        raise ValueError(
            f"Tile gains must have shape (sources, channels); got {values.shape} for {len(resolved)} sources."
        )
    rows = [
        {
            "source": str(source),
            "gains": [float(value) for value in validate_channel_gains(row, channels=values.shape[1])],
        }
        for source, row in zip(resolved, values, strict=True)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "channels": int(values.shape[1]), "tiles": rows},
            indent=2,
        )
        + "\n"
    )
    load_tile_gains(path, inputs=resolved, channels=values.shape[1])
    return path.resolve()
