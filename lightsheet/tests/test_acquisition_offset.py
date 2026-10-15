from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
import zarr

from squisher_lightsheet.acquisition_offset import run_acquisition_offset, tile_number

SCALE_ZYX = (1.807, 0.3, 0.3)
OFFSET_ZYX = (30, 8, -12)


def _volume(seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(size=(300, 240, 240)).astype(np.float32)
    smooth = ndi.gaussian_filter(noise, (2.0, 3.0, 3.0))
    return np.clip(1000.0 + 4000.0 * smooth / smooth.std(), 0, 60000).astype(np.uint16)


def _write_tile(path: Path, level0: np.ndarray) -> None:
    """Three-level pyramid that downsamples XY only, like the deconvolved light-sheet tiles."""
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    axes = ["c", "z", "y", "x"][-level0.ndim :]
    datasets = []
    for level, factor in enumerate((1, 2, 4)):
        view = level0[..., ::factor, ::factor]
        group.create_array(str(level), data=np.ascontiguousarray(view), chunks=(*view.shape[:-2], 48, 48))
        scale = [1.0] * (level0.ndim - 3) + [SCALE_ZYX[0], SCALE_ZYX[1] * factor, SCALE_ZYX[2] * factor]
        datasets.append({"path": str(level), "coordinateTransformations": [{"type": "scale", "scale": scale}]})
    group.attrs["multiscales"] = [
        {"version": "0.4", "axes": [{"name": axis} for axis in axes], "datasets": datasets}
    ]


def _record(name: str, shape: tuple[int, ...], x_um: float, yx_ratio: float = 1.0) -> dict:
    scale = (SCALE_ZYX[0], SCALE_ZYX[1] * yx_ratio, SCALE_ZYX[2] * yx_ratio)
    return {
        "tile": name,
        "translation_um": {"z": 0.0, "y": 0.0, "x": x_um},
        "scale_um": dict(zip("zyx", scale, strict=True)),
        "shape": list(shape),
    }


def _dataset(
    tmp_path: Path, *, yx_ratio: float = 1.0, moving_z_um: float = SCALE_ZYX[0]
) -> tuple[Path, Path, Path, Path]:
    """Four tiles whose reference carries the shared structure in channel 2; moving tiles start 30 planes deeper.

    yx_ratio is the moving acquisition's XY pixel spacing relative to the reference's: moving pixel p images the
    reference position offset + yx_ratio * p.
    """
    moving_dir, reference_dir = tmp_path / "moving", tmp_path / "reference"
    moving_records, reference_records = [], []
    for index in range(4):
        base = _volume(index)
        reference = base[20:260, 20:212, 20:212]
        dz, dy, dx = OFFSET_ZYX
        moving = ndi.affine_transform(
            base,
            np.diag([1.0, yx_ratio, yx_ratio]),
            offset=(20 + dz, 20 + dy, 20 + dx),
            output_shape=(200, 192, 192),
            order=1,
        ).astype(np.uint16)
        decoy = _volume(100 + index)[:240, :192, :192]
        channels = np.stack([decoy, decoy, reference])
        _write_tile(reference_dir / f"S-CL-488514561638.{index:03d}.ome.zarr", channels)
        _write_tile(moving_dir / f"S-CL-405.{index:03d}.ome.zarr", moving)
        reference_records.append(_record(f"S-CL-488514561638.{index:03d}.ome.zarr", channels.shape, 50.0 * index))
        moving_record = _record(f"S-CL-405.{index:03d}.ome.tif", moving.shape, 50.0 * index, yx_ratio)
        moving_record["scale_um"]["z"] = moving_z_um
        moving_records.append(moving_record)
    moving_position = tmp_path / "moving.positions.json"
    reference_position = tmp_path / "reference.positions.json"
    moving_position.write_text(json.dumps({"source": "placed 405", "tiles": moving_records}))
    reference_position.write_text(json.dumps({"tiles": reference_records}))
    return moving_position, reference_position, moving_dir, reference_dir


def test_acquisition_offset_recovers_depth_lag_in_level0_planes_and_shifts_positions(tmp_path: Path) -> None:
    moving_position, reference_position, moving_dir, reference_dir = _dataset(tmp_path)
    output_position = tmp_path / "out/shifted.positions.json"

    summary = run_acquisition_offset(
        moving_position=moving_position,
        reference_position=reference_position,
        moving_tile_dir=moving_dir,
        reference_tile_dir=reference_dir,
        output_dir=tmp_path / "out",
        output_position=output_position,
        reference_channel=2,
        tile_count=3,
    )

    expected = np.asarray(OFFSET_ZYX) * np.asarray(SCALE_ZYX)
    np.testing.assert_allclose(summary["median_correction_zyx_um"], expected, atol=0.31)
    assert [row["fine_lag_planes"] for row in summary["tiles"]] == [30, 30, 30]
    # The tile left out of the median validates the shift at three depths.
    measured = {row["tile"] for row in summary["tiles"]}
    validation = summary["validation"]
    assert validation["passed"] is True
    assert len(validation["held_out_tiles"]) == 1 and validation["held_out_tiles"][0] not in measured
    assert len(validation["checks"]) == 3
    assert all(check["peak_margin"] > 0 for check in validation["checks"])
    shifted = json.loads(output_position.read_text())
    assert shifted["derived_by"] == "lightsheet cross-register acquisition-offset"
    np.testing.assert_allclose(
        [shifted["tiles"][1]["translation_um"][axis] for axis in "zyx"],
        np.asarray([0.0, 0.0, 50.0]) + np.asarray(summary["median_correction_zyx_um"]),
    )


def test_acquisition_offset_measures_moving_tiles_sampled_at_a_different_xy_spacing(tmp_path: Path) -> None:
    # The depth spacings differ as 230Ptprz1's TL and TR record them, well under a plane across the stack.
    moving_position, reference_position, moving_dir, reference_dir = _dataset(
        tmp_path, yx_ratio=1.02, moving_z_um=1.806543463
    )

    summary = run_acquisition_offset(
        moving_position=moving_position,
        reference_position=reference_position,
        moving_tile_dir=moving_dir,
        reference_tile_dir=reference_dir,
        output_dir=tmp_path / "out",
        output_position=tmp_path / "out/shifted.positions.json",
        reference_channel=2,
        tile_count=3,
    )

    # Moving pixel 0 images reference pixel OFFSET_ZYX, so the correction is that offset at the reference spacing.
    expected = np.asarray(OFFSET_ZYX) * np.asarray(SCALE_ZYX)
    np.testing.assert_allclose(summary["median_correction_zyx_um"], expected, atol=0.31)
    assert summary["validation"]["passed"] is True


def test_acquisition_offset_rejects_different_depth_spacings(tmp_path: Path) -> None:
    moving_position, reference_position, moving_dir, reference_dir = _dataset(tmp_path)
    payload = json.loads(moving_position.read_text())
    for record in payload["tiles"]:
        record["scale_um"]["z"] = 2.0
    moving_position.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="drift apart by a plane or more"):
        run_acquisition_offset(
            moving_position=moving_position,
            reference_position=reference_position,
            moving_tile_dir=moving_dir,
            reference_tile_dir=reference_dir,
            output_dir=tmp_path / "out",
            output_position=tmp_path / "out/shifted.positions.json",
            reference_channel=2,
            tile_count=3,
        )


def test_acquisition_offset_rejects_tiles_absent_from_one_acquisition(tmp_path: Path) -> None:
    moving_position, reference_position, moving_dir, reference_dir = _dataset(tmp_path)

    with pytest.raises(ValueError, match=r"Tiles \['007'\] are not in both"):
        run_acquisition_offset(
            moving_position=moving_position,
            reference_position=reference_position,
            moving_tile_dir=moving_dir,
            reference_tile_dir=reference_dir,
            output_dir=tmp_path / "out",
            output_position=tmp_path / "out/shifted.positions.json",
            reference_channel=2,
            tiles=["001", "007"],
        )


def test_acquisition_offset_refuses_to_publish_positions_that_fail_the_pair_check(tmp_path: Path) -> None:
    moving_position, reference_position, moving_dir, reference_dir = _dataset(tmp_path)
    output_position = tmp_path / "out/shifted.positions.json"

    # Channel 0 is a decoy with no shared structure.
    summary = run_acquisition_offset(
        moving_position=moving_position,
        reference_position=reference_position,
        moving_tile_dir=moving_dir,
        reference_tile_dir=reference_dir,
        output_dir=tmp_path / "out",
        output_position=output_position,
        reference_channel=0,
        tile_count=3,
    )

    assert summary["validation"]["passed"] is False
    assert json.loads((tmp_path / "out" / "acquisition-offset.json").read_text())["validation"]["passed"] is False
    assert not output_position.exists()


def test_acquisition_offset_requires_a_tile_left_to_validate(tmp_path: Path) -> None:
    moving_position, reference_position, moving_dir, reference_dir = _dataset(tmp_path)

    with pytest.raises(ValueError, match="No shared tile is left to validate against"):
        run_acquisition_offset(
            moving_position=moving_position,
            reference_position=reference_position,
            moving_tile_dir=moving_dir,
            reference_tile_dir=reference_dir,
            output_dir=tmp_path / "out",
            output_position=tmp_path / "out/shifted.positions.json",
            reference_channel=2,
            tile_count=4,
        )


def test_cross_register_acquisition_offset_cli_measures_the_requested_tiles(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from squisher_lightsheet.cli import app

    moving_position, reference_position, moving_dir, reference_dir = _dataset(tmp_path)
    output_position = tmp_path / "out/shifted.positions.json"

    result = CliRunner().invoke(
        app,
        [
            "cross-register", "acquisition-offset",
            "--moving-position", str(moving_position),
            "--reference-position", str(reference_position),
            "--moving-tile-dir", str(moving_dir),
            "--reference-tile-dir", str(reference_dir),
            "--output-dir", str(tmp_path / "out"),
            "--output-position", str(output_position),
            "--reference-channel", "2",
            "--tile", "002",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "out/acquisition-offset.json").read_text())
    assert [row["tile"] for row in summary["tiles"]] == ["002"]
    assert output_position.is_file()


def test_tile_number_reads_the_numeric_suffix_of_tiff_and_zarr_names() -> None:
    assert tile_number("229Tnc-CL-405.018.ome.tif") == "018"
    assert tile_number("229Tnc-CL-488514561638.018.ome.zarr") == "018"
    with pytest.raises(ValueError, match="numeric"):
        tile_number("tile.ome.zarr")


@pytest.mark.parametrize("shift", [[float("nan"), 0., 0.], [0., 0.], [0., 0., 0., 0.]])
def test_acquisition_translation_rejects_invalid_measurement(shift: list[float], tmp_path: Path) -> None:
    from squisher_lightsheet.acquisition_offset import shifted_acquisition_positions
    with pytest.raises(ValueError, match="three finite ZYX"):
        shifted_acquisition_positions({"tiles": []}, shift, summary_path=tmp_path / "report.json")
