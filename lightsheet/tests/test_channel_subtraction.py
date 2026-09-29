from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from squisher_lightsheet import channel_subtraction
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


class DummyStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ConstantCorrection:
    def __init__(self, value: float) -> None:
        self.value = value

    def block(self, *, z_slice, y_slice, x_slice):
        return np.full(
            (
                z_slice.stop - z_slice.start,
                y_slice.stop - y_slice.start,
                x_slice.stop - x_slice.start,
            ),
            self.value,
            dtype=np.float32,
        )


def _registration_record(
    *,
    tile: str = "tile.ome.zarr",
    stage: tuple[float, float, float] = (10.0, 20.0, 30.0),
    scale: tuple[float, float, float] = (2.0, 0.5, 0.25),
    affine: np.ndarray | None = None,
) -> dict[str, object]:
    matrix = np.eye(4) if affine is None else affine
    return {
        "tile": tile,
        "stage_translation_um": dict(zip("zyx", stage, strict=True)),
        "stage_scale_um": dict(zip("zyx", scale, strict=True)),
        "registered_affine": {"matrix": matrix.tolist()},
    }


def test_paired_pull_maps_cropped_target_indices_to_reference_indices() -> None:
    target = _registration_record()
    reference = _registration_record()

    pull, output_to_registered = channel_subtraction._paired_pull_affine(
        target_record=target,
        reference_record=reference,
        crop_yx_px=20,
    )

    np.testing.assert_allclose(pull[:3, :3], np.eye(3))
    np.testing.assert_allclose(pull[:3, 3], [0.0, 20.0, 20.0])
    output_index = np.asarray([3.0, 4.0, 5.0, 1.0])
    np.testing.assert_allclose(
        output_to_registered @ output_index,
        [16.0, 32.0, 36.25, 1.0],
    )


def test_paired_pull_uses_inverse_reference_affine() -> None:
    target = _registration_record(stage=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
    reference_affine = np.eye(4)
    reference_affine[2, 3] = 7.0
    reference = _registration_record(
        stage=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        affine=reference_affine,
    )

    pull, _output_to_registered = channel_subtraction._paired_pull_affine(
        target_record=target,
        reference_record=reference,
        crop_yx_px=0,
    )

    np.testing.assert_allclose(pull[:3, :3], np.eye(3))
    np.testing.assert_allclose(pull[:3, 3], [0.0, 0.0, -7.0])


def test_position_registration_pair_rejects_different_canonical_ids(tmp_path: Path) -> None:
    position = tmp_path / "input.positions.json"
    registration = tmp_path / "input.registration.json"
    position.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "target.ome.zarr",
                        "side": "L",
                        "path": str(tmp_path / "source.ome.zarr"),
                        "translation_um": {"z": 0, "y": 0, "x": 0},
                    }
                ]
            }
        )
    )
    registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {
                        "tile": "other.ome.zarr",
                        "source_view": "L",
                        "path": str(tmp_path / "source.ome.zarr"),
                        "stage_translation_um": {"z": 0, "y": 0, "x": 0},
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="canonical tile IDs differ"):
        channel_subtraction._validate_position_registration_pair(
            position_path=position,
            registration_path=registration,
            side="L",
        )


def test_paired_tile_ids_use_intersection_and_report_unmatched_sources() -> None:
    paired, target_only, reference_only = channel_subtraction._paired_tile_ids(
        {"L.001": {}, "L.009": {}, "L.002": {}},
        {"L.002": {}, "L.001": {}},
    )

    assert paired == ["L.001", "L.002"]
    assert target_only == ["L.009"]
    assert reference_only == []


def test_canonical_pair_records_match_wavelength_specific_tile_names() -> None:
    records = {
        "230-Ptprz1-TL-514.009.ome.zarr": {
            "source_view": "L",
            "path": "/tmp/514.009.ome.zarr",
        }
    }

    assert list(channel_subtraction._canonical_pair_records(records)) == ["L.009"]


def test_join_paired_subtraction_sides_publishes_exact_disjoint_union(tmp_path: Path) -> None:
    output_root = tmp_path / "subtraction"
    output_root.mkdir()
    side_dirs = []
    for side, canonical in (("L", "canonical-left"), ("R", "canonical-right")):
        side_dir = output_root / side
        tile_dir = side_dir / "tiles"
        tile_dir.mkdir(parents=True)
        tile = tile_dir / f"{side}.ome.zarr"
        channel_subtraction._write_ome_zarr_czyx(
            tile,
            np.ones((2, 3, 4), dtype=np.uint16),
            translation_um={"z": 0.0, "y": 0.0, "x": 0.0},
            scale_um={"z": 1.0, "y": 1.0, "x": 1.0},
            channel_name="514-minus-488",
            zstd_level=3,
            chunk_shape_zyx=(2, 3, 4),
        )
        position_record = {
            "tile": tile.name,
            "source_tile": canonical,
            "side": side,
            "path": str(tile),
            "axes": "CZYX",
            "shape": [1, 2, 3, 4],
            "channels": ["514-minus-488"],
            "translation_um": {"z": 0.0, "y": 0.0, "x": 0.0},
            "scale_um": {"z": 1.0, "y": 1.0, "x": 1.0},
        }
        registration_record = {
            **position_record,
            "source_view": side,
            "stage_translation_um": position_record["translation_um"],
            "stage_scale_um": position_record["scale_um"],
            "registered_affine": {"matrix": np.eye(4).tolist()},
        }
        (side_dir / f"{side}.positions.json").write_text(
            json.dumps({"tiles": [position_record]}) + "\n"
        )
        (side_dir / f"{side}.registration.json").write_text(
            json.dumps({"input_dir": str(tile_dir), "tiles": [registration_record]}) + "\n"
        )
        (side_dir / f"{side}.summary.json").write_text(
            json.dumps({"status": "complete", "side": side, "tile_count": 1}) + "\n"
        )
        side_dirs.append(side_dir)

    result = channel_subtraction.join_paired_subtraction_sides(
        side_output_dirs=side_dirs,
        output_root=output_root,
    )

    assert result.tile_count == 2
    positions = json.loads(result.position_output.read_text())
    registrations = json.loads(result.registration_output.read_text())
    summary = json.loads(result.summary_output.read_text())
    assert summary["status"] == "complete"
    assert summary["side_counts"] == {"L": 1, "R": 1}
    assert {row["source_tile"] for row in positions["tiles"]} == {
        "canonical-left",
        "canonical-right",
    }
    assert len(registrations["tiles"]) == 2
    for row in positions["tiles"]:
        path = Path(row["path"])
        assert path.is_symlink()
        assert zarr.open_group(str(path), mode="r").attrs["squisher_complete"] is True


def test_subtraction_applies_target_and_reference_corrections_before_subtracting(monkeypatch) -> None:
    target = np.full((3, 4, 5), 20, dtype=np.uint16)
    reference = np.full((3, 4, 5), 4, dtype=np.uint16)

    def fake_subtract(target_slab, reference_halo, **kwargs):
        z_stop = reference_halo.shape[0] - kwargs["halo_after"]
        return target_slab - reference_halo[kwargs["halo_before"] : z_stop]

    monkeypatch.setattr(channel_subtraction, "_subtract_slab_gpu", fake_subtract)
    actual = channel_subtraction.subtract_spillover_array_gpu(
        target,
        reference,
        reference_shift_zyx_px=(0.0, 0.0, 0.0),
        alpha=1.0,
        beta=0.0,
        target_background=0.0,
        reference_background=0.0,
        crop_yx_px=0,
        z_chunk=2,
        output_dtype=np.dtype(np.float32),
        target_correction=ConstantCorrection(2.0),
        reference_correction=ConstantCorrection(3.0),
    )

    np.testing.assert_array_equal(actual, 28.0)


def test_corrected_zarr_preserves_single_channel_spatial_metadata(tmp_path: Path) -> None:
    output = tmp_path / "corrected.ome.zarr"
    data = np.ones((3, 4, 5), dtype=np.uint16)

    channel_subtraction._write_ome_zarr_czyx(
        output,
        data,
        translation_um={"z": 10.0, "y": 20.0, "x": 30.0},
        scale_um={"z": 1.0, "y": 0.5, "x": 0.25},
        channel_name="514",
        zstd_level=3,
        chunk_shape_zyx=(2, 3, 4),
    )

    root = zarr.open_group(str(output), mode="r")
    level0 = root["0"]
    assert root.attrs["squisher_complete"] is True
    assert root.attrs["omero"]["channels"] == [{"label": "514"}]
    assert level0.shape == (1, 3, 4, 5)
    assert level0.chunks == (1, 2, 3, 4)
    assert level0.metadata.dimension_names == ("c", "z", "y", "x")
    np.testing.assert_array_equal(level0[0], data)
    parsed = legacy.parse_ome_metadata(output)
    assert parsed.axes == "CZYX"
    assert parsed.translation == {"z": 10.0, "y": 20.0, "x": 30.0}
    assert parsed.spacing == {"z": 1.0, "y": 0.5, "x": 0.25}


@pytest.mark.parametrize("axes", ["CZYX", "ZCYX"])
def test_subtract_channel_tiles_writes_cropped_parseable_position_file(monkeypatch, tmp_path, axes) -> None:
    source_tile = tmp_path / "tile.ome.tif"
    plane_count = 4 * 3
    tifffile.imwrite(
        source_tile,
        np.zeros((4, 3, 6, 6), dtype=np.uint16),
        ome=True,
        metadata={
            "axes": "CZYX",
            "PhysicalSizeX": 0.25,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeZ": 1.0,
            "Plane": {
                "PositionX": [30.0] * plane_count,
                "PositionY": [20.0] * plane_count,
                "PositionZ": [10.0 + (index % 3) for index in range(plane_count)],
            },
        },
    )
    position_input = tmp_path / "input.positions.json"
    position_input.write_text(
        json.dumps(
            {
                "artifact_type": "lightsheet.position.v1",
                "units": "micrometer",
                "tiles": [
                    {
                        "tile": source_tile.name,
                        "side": "L",
                        "path": str(source_tile),
                        "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                        "scale_um": {"z": 1.0, "y": 0.5, "x": 0.25},
                    }
                ],
            }
        )
        + "\n"
    )
    registration_input = tmp_path / "input.registration.json"
    registration_input.write_text(
        json.dumps(
            {
                "input_dir": str(tmp_path),
                "tiles": [
                    {
                        "tile": source_tile.name,
                        "source_view": "L",
                        "stage_translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                        "stage_scale_um": {"z": 1.0, "y": 0.5, "x": 0.25},
                        "registered_affine": {"matrix": [[1, 0, 0, 0]]},
                    }
                ],
            }
        )
        + "\n"
    )
    data = np.zeros((4, 3, 6, 6), dtype=np.uint16)
    data[2] = 100
    data[3] = 20
    if axes == "ZCYX":
        data = data.transpose(1, 0, 2, 3)
    tile = legacy.TileMetadata(
        path=source_tile,
        shape=data.shape,
        axes=axes,
        spacing={"z": 1.0, "y": 0.5, "x": 0.25},
        translation={"z": 10.0, "y": 20.0, "x": 30.0},
        channels=("0", "1", "2", "3"),
        tracks=(
            legacy.TrackMetadata(
                slug="track0", track_id="all", channels=(0, 1, 2, 3), channel_names=("0", "1", "2", "3")
            ),
        ),
    )
    store = DummyStore()

    monkeypatch.setattr(channel_subtraction.legacy, "read_position_input_tiles", lambda _path: [tile])
    monkeypatch.setattr(
        channel_subtraction.legacy, "open_tile_array", lambda *_args, **_kwargs: (data, store)
    )

    def fake_subtract(target, reference, **kwargs):
        corrected = (
            target.astype(np.float32)
            - kwargs["target_background"]
            - kwargs["alpha"] * np.maximum(reference.astype(np.float32) - kwargs["reference_background"], 0)
            - kwargs["beta"]
        )
        corrected = corrected[:, 1:-1, 1:-1]
        return np.rint(np.maximum(corrected, 0)).astype(np.uint16)

    monkeypatch.setattr(channel_subtraction, "subtract_spillover_array_gpu", fake_subtract)

    result = channel_subtraction.subtract_channel_tiles(
        position_input=position_input,
        output_dir=tmp_path / "subtracted",
        registration_input=registration_input,
        target_channel=2,
        reference_channel=3,
        source_level=0,
        reference_shift_zyx_px=(0.0, 0.0, 0.0),
        alpha=2.0,
        beta=-1.0,
        target_background=10.0,
        reference_background=5.0,
        crop_yx_px=1,
    )

    payload = json.loads(result.position_output.read_text())
    assert payload["tiles"][0]["side"] == "L"
    assert payload["tiles"][0]["translation_um"] == {"z": 10.0, "y": 20.5, "x": 30.25}
    assert payload["tiles"][0]["scale_um"] == {"z": 1.0, "y": 0.5, "x": 0.25}
    assert payload["tiles"][0]["axes"] == "CZYX"
    assert payload["tiles"][0]["shape"] == [1, 3, 4, 4]
    output_tile = Path(payload["tiles"][0]["path"])
    assert output_tile.exists()
    assert store.closed is True

    root = zarr.open_group(str(output_tile), mode="r")
    output = np.asarray(root["0"][0])
    assert output.shape == (3, 4, 4)
    assert np.all(output == 61)
    assert output_tile.name.endswith(".ome.zarr")
    assert root.attrs["squisher_complete"] is True

    parsed = legacy.parse_ome_metadata(output_tile)
    assert parsed.axes == "CZYX"
    assert parsed.translation == {"z": 10.0, "y": 20.5, "x": 30.25}

    assert result.registration_output is not None
    registration_payload = json.loads(result.registration_output.read_text())
    assert registration_payload["input_dir"].endswith("/subtracted/tiles")
    assert registration_payload["tiles"][0]["tile"] == output_tile.name
    assert registration_payload["tiles"][0]["path"] == str(output_tile)
    assert registration_payload["tiles"][0]["axes"] == "CZYX"
    assert registration_payload["tiles"][0]["shape"] == [1, 3, 4, 4]
    assert registration_payload["tiles"][0]["source_tile"] == source_tile.name
    assert registration_payload["tiles"][0]["stage_translation_um"] == {"z": 10.0, "y": 20.5, "x": 30.25}
    assert registration_payload["tiles"][0]["registered_affine"] == {"matrix": [[1, 0, 0, 0]]}
