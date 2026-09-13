"""Composition of fitted multiplicative fields into serialized BaSiC profiles."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Protocol, cast

import numpy as np

from squisher_deconv.residual_field import validate_coefficients


class _BasicProfile(Protocol):
    flatfield: object
    darkfield: object


@dataclass(frozen=True)
class BasicProfileArrays:
    flatfield: np.ndarray
    darkfield: np.ndarray | None
    residual_coefficient: np.ndarray | None = None


def _load_payload(source: Path) -> tuple[dict[object, object], _BasicProfile]:
    payload = pickle.loads(source.read_bytes())
    if not isinstance(payload, dict) or "basic" not in payload:
        raise ValueError(f"{source} must contain a pickled mapping with a 'basic' profile")
    basic = cast(_BasicProfile, payload["basic"])
    if not hasattr(basic, "flatfield") or not hasattr(basic, "darkfield"):
        raise ValueError(f"{source} basic profile must contain flatfield and darkfield")
    return payload, basic


def _profile_arrays(source: Path, basic: _BasicProfile) -> BasicProfileArrays:
    flatfield = np.asarray(basic.flatfield, dtype=np.float32)
    darkfield = None if basic.darkfield is None else np.asarray(basic.darkfield, dtype=np.float32)
    if flatfield.ndim != 2 or not np.all(np.isfinite(flatfield)) or np.any(flatfield <= 0):
        raise ValueError(f"{source} flatfield must be a positive finite 2D array")
    if darkfield is not None and (darkfield.shape != flatfield.shape or not np.all(np.isfinite(darkfield))):
        raise ValueError(f"{source} darkfield must be finite and match the flatfield shape")
    return BasicProfileArrays(flatfield=flatfield, darkfield=darkfield)


def load_basic_profile_arrays(source: Path) -> BasicProfileArrays:
    """Load and validate the arrays used by the deconvolution BaSiC profile."""
    payload, basic = _load_payload(source)
    arrays = _profile_arrays(source, basic)
    model = payload.get("residual_field")
    if model is None:
        return arrays
    if (
        not isinstance(model, dict)
        or model.get("coordinate") != "normalized-original-raw-zyx"
        or model.get("basis") != "cosine-xy2-z-scaled"
    ):
        raise ValueError(f"{source} has an unsupported residual field coordinate or basis")
    return BasicProfileArrays(
        arrays.flatfield, arrays.darkfield, validate_coefficients(model.get("coefficient"))
    )


def compose_basic_profile(
    source: Path,
    output: Path,
    *,
    mask: np.ndarray,
    provenance: Mapping[str, object],
) -> BasicProfileArrays:
    """Divide a BaSiC flatfield by a positive residual mask and preserve its payload."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite BaSiC profile: {output}")
    payload, basic = _load_payload(source)
    arrays = _profile_arrays(source, basic)
    flatfield, darkfield = arrays.flatfield, arrays.darkfield
    residual = np.asarray(mask, dtype=np.float32)
    if residual.shape != flatfield.shape or not np.all(np.isfinite(residual)) or np.any(residual <= 0):
        raise ValueError(
            f"Residual mask must be positive, finite, and have shape {flatfield.shape}; got {residual.shape}"
        )

    composed = flatfield / residual
    basic.flatfield = composed
    payload["residual_correction"] = dict(provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pickle.dumps(payload))
    return BasicProfileArrays(flatfield=composed, darkfield=darkfield)


def compose_z_profile(
    source: Path, output: Path, *, coefficient: np.ndarray, provenance: Mapping[str, object]
) -> BasicProfileArrays:
    """Persist a smooth 3D residual separately from the original 2D BaSiC arrays."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite BaSiC profile: {output}")
    payload, basic = _load_payload(source)
    if "residual_field" in payload:
        raise ValueError(f"{source} already contains a Z-dependent residual field")
    arrays = _profile_arrays(source, basic)
    coefficient = validate_coefficients(coefficient)
    payload["residual_field"] = {
        "coordinate": "normalized-original-raw-zyx",
        "basis": "cosine-xy2-z-scaled",
        "coefficient": coefficient.tolist(),
    }
    payload["residual_correction"] = dict(provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pickle.dumps(payload))
    return BasicProfileArrays(arrays.flatfield, arrays.darkfield, coefficient)
