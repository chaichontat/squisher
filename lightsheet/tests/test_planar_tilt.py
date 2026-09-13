import numpy as np
import pytest
import json
from pathlib import Path

from squisher_lightsheet.planar_tilt import _minor_angle, _validate_window
from squisher_lightsheet.planar_tilt import write_planar_tilt_fit


def _write_source(path: Path, data: np.ndarray, translation=(0.0, 0.0, 0.0)) -> None:
    import zarr

    root = zarr.open_group(str(path), mode="w")
    root.create_array("0", data=data, chunks=data.shape, dimension_names=["z", "y", "x"])
    root.attrs["ome"] = {
        "multiscales": [
            {
                "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0] * 3},
                            {"type": "translation", "translation": list(translation)},
                        ],
                    }
                ],
            }
        ]
    }


def _write_mask(path: Path, direction=None) -> None:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.ones((24, 24, 24), dtype=np.uint8))
    image.SetSpacing((0.001, 0.001, 0.001))
    if direction is not None:
        image.SetDirection(direction)
    sitk.WriteImage(image, str(path))


def test_public_fit_writes_deterministic_result(tmp_path: Path) -> None:
    mask, source, output = tmp_path / "mask.nrrd", tmp_path / "source.zarr", tmp_path / "fit.json"
    _write_mask(mask)
    z, y, x = np.indices((24, 24, 24), dtype=float)
    data = np.exp(-((z - 0.3 * x - 0.2 * y - 8) ** 2) / 2).astype(np.float32)
    _write_source(source, data)
    write_planar_tilt_fit(
        mask_path=mask,
        source_paths=[source],
        windows=[(0, 1)],
        output_path=output,
        level=0,
        z_depth_um=12,
        xy_step_um=2,
    )
    first = output.read_bytes()
    payload = json.loads(first)
    write_planar_tilt_fit(
        mask_path=mask,
        source_paths=[source],
        windows=[(0, 1)],
        output_path=output,
        level=0,
        z_depth_um=12,
        xy_step_um=2,
    )
    assert output.read_bytes() == first
    assert payload["fitted_z_planes"] == 12
    assert payload["fitted_xy_stride_yx"] == [2, 2]
    sampled = data[6:18, ::2, ::2]
    zz, yy, xx = np.meshgrid(np.arange(6, 18), np.arange(0, 24, 2), np.arange(0, 24, 2), indexing="ij")
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    weights = sampled.ravel()
    center = np.average(coordinates, axis=0, weights=weights)
    centered = coordinates - center
    expected_covariance = np.einsum("ni,n,nj->ij", centered, weights, centered) / weights.sum()
    assert payload["coronal_lateral_tilt_deg"] == pytest.approx(_minor_angle(expected_covariance, (0, 2)))
    assert payload["coronal_pitch_deg"] == pytest.approx(_minor_angle(expected_covariance, (1, 2)))
    assert np.isclose(np.linalg.norm(payload["normal_xyz"]), 1)


def test_public_fit_rejects_mask_and_source_contracts(tmp_path: Path) -> None:
    mask, first, second = tmp_path / "mask.nrrd", tmp_path / "a.zarr", tmp_path / "b.zarr"
    _write_mask(mask, (0, 1, 0, 1, 0, 0, 0, 0, 1))
    with pytest.raises(ValueError, match="identity direction"):
        write_planar_tilt_fit(
            mask_path=mask,
            source_paths=[first],
            windows=[(0, 1)],
            output_path=tmp_path / "out",
            level=0,
            z_depth_um=1,
            xy_step_um=1,
        )
    _write_mask(mask)
    data = np.ones((24, 24, 24), dtype=np.float32)
    _write_source(first, data)
    _write_source(second, data, translation=(1, 0, 0))
    with pytest.raises(ValueError, match="does not match"):
        write_planar_tilt_fit(
            mask_path=mask,
            source_paths=[first, second],
            windows=[(0, 1), (0, 1)],
            output_path=tmp_path / "out",
            level=0,
            z_depth_um=4,
            xy_step_um=2,
        )


def test_minor_axis_angles_are_independent() -> None:
    def covariance_at(angle_deg: float) -> np.ndarray:
        angle = np.deg2rad(angle_deg)
        rotation = np.asarray([[np.sin(angle), -np.cos(angle)], [np.cos(angle), np.sin(angle)]])
        return rotation @ np.diag([1.0, 4.0]) @ rotation.T

    covariance = np.zeros((3, 3))
    covariance[np.ix_((0, 2), (0, 2))] = covariance_at(17.0)
    assert _minor_angle(covariance, (0, 2)) == pytest.approx(17.0)
    covariance[np.ix_((1, 2), (1, 2))] = covariance_at(-11.0)
    assert _minor_angle(covariance, (1, 2)) == pytest.approx(-11.0)


def test_validate_window_requires_finite_ordered_pair() -> None:
    assert _validate_window((1, 4), 0) == (1.0, 4.0)
    with pytest.raises(ValueError, match="lower < upper"):
        _validate_window((4, 1), 0)
    with pytest.raises(ValueError, match="finite"):
        _validate_window((0, np.inf), 0)
