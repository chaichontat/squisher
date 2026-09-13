from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from squisher_lightsheet.registration_join import write_joined_channel_affine_registration


def _affine(translation_x: float) -> dict[str, object]:
    matrix = np.eye(4)
    matrix[2, 3] = translation_x
    return {
        "dims": ["x_in", "x_out"],
        "coords": {
            "x_in": ["z", "y", "x", "1"],
            "x_out": ["z", "y", "x", "1"],
        },
        "matrix": matrix.tolist(),
    }


def _tile(name: str, *, translation_x: float = 0.0) -> dict[str, object]:
    return {
        "tile": name,
        "path": f"/data/{name}",
        "shape": [4, 21, 31, 41],
        "scale": {"z": 1.8, "y": 0.3, "x": 0.3},
        "stage_translation_um": {"z": 0.0, "y": 10.0, "x": 20.0},
        "registered_affine": _affine(translation_x),
    }


def _write_registration(path: Path, tiles: list[dict[str, object]], *, side: bool) -> None:
    payload: dict[str, object] = {
        "input_dir": "/data",
        "registered_transform_key": "registered_affine",
        "metadata_transform_key": "stage_translation_um",
        "spacing_um": {"z": 1.8, "y": 0.3, "x": 0.3},
        "tiles": tiles,
    }
    if side:
        payload.update(
            {
                "artifact_type": "squisher_lightsheet.global_channel_affine_registration.v1",
                "adaptation_method": "componentwise_median_full_tile_centered_channel_affine",
                "transform_contract": {
                    "schema": "squisher_lightsheet.registration_transform_contract.v1",
                    "registered_affine_semantics": (
                        "moving_tile_stage_um_to_reference_registered_um"
                    ),
                    "source_space": "514-side_stage_um",
                    "target_space": "561_registered_um",
                    "composition_order": [
                        "reference_registered_affine",
                        "reference_stage_translation_um",
                        "moving_to_reference_channel_affine_um",
                        "inverse_moving_stage_translation_um",
                    ],
                    "registered_affine_contains_full_channel_affine": True,
                    "stage_translation_source": "reference_registration_input",
                },
                "diagnostics": {"global_channel_affine": {"accepted_window_count": 3}},
            }
        )
    path.write_text(json.dumps(payload) + "\n")


def test_join_registration_uses_reference_order_and_side_affines(tmp_path: Path) -> None:
    reference_tiles = [_tile("CL.001"), _tile("CR.000"), _tile("CL.002")]
    reference = tmp_path / "reference.json"
    _write_registration(reference, reference_tiles, side=False)
    cl = tmp_path / "cl.json"
    cr = tmp_path / "cr.json"
    cl_tiles = [copy.deepcopy(reference_tiles[2]), copy.deepcopy(reference_tiles[0])]
    cl_tiles[0]["registered_affine"] = _affine(12.0)
    cl_tiles[1]["registered_affine"] = _affine(11.0)
    cr_tiles = [copy.deepcopy(reference_tiles[1])]
    cr_tiles[0]["registered_affine"] = _affine(21.0)
    _write_registration(cl, cl_tiles, side=True)
    _write_registration(cr, cr_tiles, side=True)
    output = tmp_path / "joined.json"

    result = write_joined_channel_affine_registration(
        reference_registration_input=reference,
        side_registration_inputs=[cl, cr],
        output_registration=output,
        source_label="514",
        target_label="561",
    )

    assert result == output.resolve()
    joined = json.loads(output.read_text())
    assert [tile["tile"] for tile in joined["tiles"]] == ["CL.001", "CR.000", "CL.002"]
    assert [tile["registered_affine"]["matrix"][2][3] for tile in joined["tiles"]] == [
        11.0,
        21.0,
        12.0,
    ]
    for reference_tile, joined_tile in zip(reference_tiles, joined["tiles"], strict=True):
        assert {key: value for key, value in joined_tile.items() if key != "registered_affine"} == {
            key: value for key, value in reference_tile.items() if key != "registered_affine"
        }
    assert joined["artifact_type"] == (
        "squisher_lightsheet.joined_channel_affine_registration.v1"
    )
    assert joined["transform_contract"]["source_space"] == "514_stage_um"
    provenance = joined["diagnostics"]["registration_join"]
    assert provenance["reference_registration"]["path"] == str(reference.resolve())
    assert len(provenance["reference_registration"]["sha256"]) == 64
    assert [item["path"] for item in provenance["side_registrations"]] == [
        str(cl.resolve()),
        str(cr.resolve()),
    ]
    assert provenance["tile_count"] == 3


@pytest.mark.parametrize(
    ("side_tile_names", "message"),
    [
        ((["CL.001"], ["CL.001", "CR.000", "CL.002"]), "overlapping tile CL.001"),
        ((["CL.001"], ["CR.000"]), "missing tiles: CL.002"),
        ((["CL.001", "extra"], ["CR.000", "CL.002"]), "unknown tiles: extra"),
    ],
)
def test_join_registration_rejects_non_disjoint_or_inexact_coverage(
    tmp_path: Path,
    side_tile_names: tuple[list[str], list[str]],
    message: str,
) -> None:
    reference_tiles = [_tile("CL.001"), _tile("CR.000"), _tile("CL.002")]
    reference_by_tile = {str(tile["tile"]): tile for tile in reference_tiles}
    reference = tmp_path / "reference.json"
    _write_registration(reference, reference_tiles, side=False)
    sides = []
    for index, names in enumerate(side_tile_names):
        path = tmp_path / f"side{index}.json"
        tiles = [copy.deepcopy(reference_by_tile.get(name, _tile(name))) for name in names]
        _write_registration(path, tiles, side=True)
        sides.append(path)

    with pytest.raises(ValueError, match=message):
        write_joined_channel_affine_registration(
            reference_registration_input=reference,
            side_registration_inputs=sides,
            output_registration=tmp_path / "joined.json",
            source_label="514",
            target_label="561",
        )

def test_join_registration_rejects_changed_source_geometry(tmp_path: Path) -> None:
    reference_tile = _tile("CL.001")
    reference = tmp_path / "reference.json"
    _write_registration(reference, [reference_tile], side=False)
    side = tmp_path / "side.json"
    changed = copy.deepcopy(reference_tile)
    changed["path"] = "/wrong/source.ome.zarr"
    changed["registered_affine"] = _affine(5.0)
    _write_registration(side, [changed], side=True)

    with pytest.raises(ValueError, match="source geometry differs"):
        write_joined_channel_affine_registration(
            reference_registration_input=reference,
            side_registration_inputs=[side],
            output_registration=tmp_path / "joined.json",
            source_label="514",
            target_label="561",
        )
