from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from squisher_lightsheet.artifact_io import sha256_file
from squisher_lightsheet.channel_affine import RegistrationTransformContract


JOINED_ARTIFACT_TYPE = "squisher_lightsheet.joined_channel_affine_registration.v1"
SIDE_ARTIFACT_TYPE = "squisher_lightsheet.global_channel_affine_registration.v1"


def _read_registration(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    records = payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} must contain a nonempty tiles list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{label} tiles must be JSON objects")
    return payload


def _records_by_tile(payload: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in payload["tiles"]:
        tile = record.get("tile")
        if not isinstance(tile, str) or not tile:
            raise ValueError(f"{label} contains a tile without a nonempty string tile identity")
        if tile in records:
            raise ValueError(f"{label} contains duplicate tile {tile}")
        records[tile] = record
    return records


def _validate_registered_affine(record: dict[str, Any], *, label: str) -> None:
    try:
        matrix = np.asarray(record["registered_affine"]["matrix"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has no numeric registered_affine.matrix") from error
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} registered_affine.matrix must be a finite 4x4 matrix")


def _source_geometry(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "registered_affine"}


def write_joined_channel_affine_registration(
    *,
    reference_registration_input: Path,
    side_registration_inputs: Sequence[Path],
    output_registration: Path,
    source_label: str,
    target_label: str,
) -> Path:
    """Join disjoint full-tile channel-affine registrations in reference tile order.

    Side artifacts may supply only ``registered_affine``. Every source identity and
    geometry field comes from, and must agree exactly with, the reference registration.
    """
    reference_registration_input = reference_registration_input.expanduser().resolve()
    side_registration_inputs = [path.expanduser().resolve() for path in side_registration_inputs]
    output_registration = output_registration.expanduser().resolve()
    if not side_registration_inputs:
        raise ValueError("At least one side registration is required")
    if len(set(side_registration_inputs)) != len(side_registration_inputs):
        raise ValueError("Side registration paths must be unique")
    if output_registration in {reference_registration_input, *side_registration_inputs}:
        raise ValueError("output_registration must differ from all registration inputs")
    if not source_label.strip() or not target_label.strip():
        raise ValueError("source_label and target_label must be nonempty")

    reference = _read_registration(reference_registration_input, label="reference registration")
    reference_by_tile = _records_by_tile(reference, label="reference registration")
    for tile, record in reference_by_tile.items():
        _validate_registered_affine(record, label=f"Reference tile {tile}")

    replacements: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    reference_tiles = set(reference_by_tile)
    for path in side_registration_inputs:
        side = _read_registration(path, label=f"side registration {path}")
        if side.get("artifact_type") != SIDE_ARTIFACT_TYPE:
            raise ValueError(
                f"Side registration {path} has artifact_type={side.get('artifact_type')!r}; "
                f"expected {SIDE_ARTIFACT_TYPE!r}"
            )
        side_by_tile = _records_by_tile(side, label=f"side registration {path}")
        unknown = sorted(set(side_by_tile) - reference_tiles)
        if unknown:
            raise ValueError(f"Side registration {path} has unknown tiles: {', '.join(unknown)}")
        overlap = sorted(set(side_by_tile) & set(replacements))
        if overlap:
            raise ValueError(f"Side registrations have overlapping tile {overlap[0]}")
        for tile, side_record in side_by_tile.items():
            if _source_geometry(side_record) != _source_geometry(reference_by_tile[tile]):
                raise ValueError(f"Side registration {path} tile {tile} source geometry differs")
            _validate_registered_affine(side_record, label=f"Side registration {path} tile {tile}")
            replacements[tile] = side_record["registered_affine"]
        provenance.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "artifact_type": side["artifact_type"],
                "adaptation_method": side.get("adaptation_method"),
                "tile_count": len(side_by_tile),
                "tiles": list(side_by_tile),
                "transform_contract": side.get("transform_contract"),
                "global_channel_affine": side.get("diagnostics", {}).get(
                    "global_channel_affine"
                ),
            }
        )

    missing = [tile for tile in reference_by_tile if tile not in replacements]
    if missing:
        raise ValueError(f"Side registrations are missing tiles: {', '.join(missing)}")

    joined = copy.deepcopy(reference)
    for record in joined["tiles"]:
        record["registered_affine"] = copy.deepcopy(replacements[str(record["tile"])])
    joined["artifact_type"] = JOINED_ARTIFACT_TYPE
    joined["adapted_from"] = str(reference_registration_input)
    joined["adaptation_method"] = "join_disjoint_channel_affine_registrations"
    joined["transform_contract"] = RegistrationTransformContract(
        registered_affine_semantics="moving_tile_stage_um_to_reference_registered_um",
        source_space=f"{source_label.strip()}_stage_um",
        target_space=f"{target_label.strip()}_registered_um",
        composition_order=(
            "reference_registered_affine",
            "reference_stage_translation_um",
            "moving_to_reference_channel_affine_um",
            "inverse_moving_stage_translation_um",
        ),
        registered_affine_contains_full_channel_affine=True,
        stage_translation_source="reference_registration_input",
    ).model_dump(mode="json", by_alias=True)
    joined.setdefault("diagnostics", {})["registration_join"] = {
        "reference_registration": {
            "path": str(reference_registration_input),
            "sha256": sha256_file(reference_registration_input),
        },
        "side_registrations": provenance,
        "tile_count": len(reference_by_tile),
        "tile_order_source": "reference_registration_input",
        "source_geometry_source": "reference_registration_input",
        "coverage": "exact_disjoint_partition",
    }

    output_registration.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_registration.with_name(f".{output_registration.name}.tmp")
    temporary.write_text(json.dumps(joined, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_registration)
    return output_registration
