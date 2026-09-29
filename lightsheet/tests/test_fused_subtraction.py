from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.codecs import BytesCodec, ZstdCodec

from squisher_lightsheet.fused_subtraction import (
    fit_fused_subtraction,
    fit_quantile_line,
    subtract_fused_volumes,
)


def _write_fused(path: Path, levels: list[np.ndarray], *, translation: float = 0.0, complete: bool = True) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "axes": [{"name": axis, "type": "space", "unit": "micrometer"} for axis in "zyx"],
                        "datasets": [
                            {
                                "path": str(level),
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [2.0**level] * 3},
                                    {"type": "translation", "translation": [translation, 0.0, 0.0]},
                                ],
                            }
                            for level in range(len(levels))
                        ],
                    }
                ],
            },
            "squisher_complete": complete,
            "squisher_fusion": {"channel": 0},
        }
    )
    for level, data in enumerate(levels):
        array = root.create_array(
            str(level),
            shape=data.shape,
            chunks=(2, 8, 8),
            dtype=data.dtype,
            dimension_names=("z", "y", "x"),
            serializer=BytesCodec(),
            compressors=[ZstdCodec(level=1)],
        )
        array[:] = data


def _pair(rng: np.random.Generator, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    reference = rng.integers(100, 1000, size=shape).astype(np.uint16)
    target = np.rint(0.5 * reference + 30 + rng.exponential(40.0, size=shape)).astype(np.uint16)
    return target, reference


def test_fit_quantile_line_recovers_lower_envelope() -> None:
    rng = np.random.default_rng(1)
    reference = rng.uniform(100, 2000, size=4000)
    target = 0.8 * reference + 20 + rng.exponential(50.0, size=reference.size)

    alpha, beta = fit_quantile_line(reference, target, quantile=0.05)

    assert alpha == pytest.approx(0.8, abs=0.01)
    assert beta == pytest.approx(20 + 50.0 * -np.log(0.95), abs=4)
    assert np.mean(target < alpha * reference + beta) == pytest.approx(0.05, abs=0.005)


def test_fit_and_subtract_stream_every_level_of_the_shared_grid(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    target0, reference0 = _pair(rng, (8, 32, 32))
    target1, reference1 = _pair(rng, (4, 16, 16))
    reference0[:, :4] = 0
    _write_fused(tmp_path / "514.ome.zarr", [target0, target1])
    _write_fused(tmp_path / "488.ome.zarr", [reference0, reference1])

    model = fit_fused_subtraction(
        target=tmp_path / "514.ome.zarr",
        reference=tmp_path / "488.ome.zarr",
        output_dir=tmp_path / "model",
        planes=6,
        stride_yx=1,
    )
    summary = subtract_fused_volumes(model=Path(model["model_path"]), output=tmp_path / "sub.ome.zarr", workers=2)

    assert model["alpha"] == pytest.approx(0.5, abs=0.02)
    assert Path(model["qc_image"]).is_file()
    root = zarr.open_group(str(tmp_path / "sub.ome.zarr"), mode="r")
    assert root.attrs["squisher_complete"] is True
    assert "squisher_fusion" not in root.attrs
    assert root.attrs["ome"] == zarr.open_group(str(tmp_path / "514.ome.zarr"), mode="r").attrs["ome"]
    for level, (target, reference) in enumerate(((target0, reference0), (target1, reference1))):
        expected = np.clip(
            np.rint(target.astype(np.float32) - np.float32(model["alpha"]) * reference - np.float32(model["beta"])),
            0,
            None,
        )
        expected[reference == 0] = 0
        np.testing.assert_array_equal(root[str(level)][:], expected.astype(np.uint16))
    assert summary["levels"][0]["target_only_voxels"] == 8 * 4 * 32
    assert not np.any(root["0"][:, :4])


def test_fit_rejects_volumes_on_different_grids(tmp_path: Path) -> None:
    target, reference = _pair(np.random.default_rng(3), (4, 16, 16))
    _write_fused(tmp_path / "514.ome.zarr", [target])
    _write_fused(tmp_path / "488.ome.zarr", [reference], translation=1.0)

    with pytest.raises(ValueError, match="multiscale grid"):
        fit_fused_subtraction(
            target=tmp_path / "514.ome.zarr", reference=tmp_path / "488.ome.zarr", output_dir=tmp_path / "model"
        )


def test_subtract_rejects_incomplete_input(tmp_path: Path) -> None:
    target, reference = _pair(np.random.default_rng(4), (4, 16, 16))
    _write_fused(tmp_path / "514.ome.zarr", [target])
    _write_fused(tmp_path / "488.ome.zarr", [reference])
    model = fit_fused_subtraction(
        target=tmp_path / "514.ome.zarr", reference=tmp_path / "488.ome.zarr", output_dir=tmp_path / "model", stride_yx=1
    )
    _write_fused(tmp_path / "488.ome.zarr", [reference], complete=False)

    with pytest.raises(ValueError, match="not marked complete"):
        subtract_fused_volumes(model=Path(model["model_path"]), output=tmp_path / "sub.ome.zarr")
    assert json.loads(Path(model["model_path"]).read_text())["sample_count"] > 0
