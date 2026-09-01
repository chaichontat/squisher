from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from squisher_segment.cli import app
from squisher_segment.segment import n4


def _downsample_mean_numpy(block: np.ndarray, *, factors: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    reshape: list[int] = []
    axes = []
    for axis, (size, factor) in enumerate(zip(block.shape, factors, strict=True)):
        reshape.extend((size // factor, factor))
        axes.append(2 * axis + 1)
    reduced = block.reshape(reshape).mean(axis=tuple(axes), dtype=np.float32)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        reduced = np.clip(np.rint(reduced), info.min, info.max)
    return reduced.astype(dtype)


def _write_ome_zarr(
    path: Path, *, complete: bool = True, with_provenance: bool = False
) -> tuple[zarr.Group, np.ndarray]:
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": "/",
                        "axes": [
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [0.6, 0.3, 0.3]},
                                    {"type": "translation", "translation": [1.0, 2.0, 3.0]},
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.2, 0.6, 0.6]},
                                    {"type": "translation", "translation": [1.0, 2.0, 3.0]},
                                ],
                            },
                        ],
                    }
                ],
                "omero": {"channels": [{"label": "membrane"}]},
            },
            "acquisition": {"sample": "example"},
            "squisher_complete": complete,
        }
    )
    if with_provenance:
        provenance = path / "provenance"
        provenance.mkdir()
        manifest = {
            "schema_version": 1,
            "artifacts": [{"bundled_path": "provenance/input.json"}],
        }
        (provenance / "manifest.json").write_text(json.dumps(manifest))
        (provenance / "input.json").write_text('{"source": "lightsheet"}')
        root.attrs["squisher_fusion"] = {"manifest": "provenance/manifest.json"}

    y, x = np.mgrid[:32, :32]
    z_bias = np.linspace(0.8, 1.2, 16, dtype=np.float32)[:, None, None]
    x_bias = np.linspace(0.85, 1.15, 32, dtype=np.float32)[None, None, :]
    smooth_field = z_bias * x_bias
    base = np.stack([100 + 20 * z + 2 * y + x for z in range(16)]).astype(np.float32)
    data = np.rint(base * smooth_field).astype(np.uint16)
    data[:, :2, :2] = 0

    level0 = root.create_array(
        "0",
        data=data,
        chunks=(2, 8, 8),
        shards=(4, 16, 16),
        dimension_names=("z", "y", "x"),
    )
    level0.attrs["level_note"] = "copy me"
    root.create_array(
        "1",
        data=np.full((8, 16, 16), np.iinfo(np.uint16).max, dtype=np.uint16),
        chunks=(2, 8, 8),
        shards=(4, 16, 16),
        dimension_names=("z", "y", "x"),
    )
    return root, np.broadcast_to(smooth_field, data.shape).copy()


def _write_completed_n4_output(path: Path, marker: str) -> None:
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "squisher_complete": True,
            "squisher_n4": {"field_path": n4.N4_FIELD_FILENAME},
        }
    )
    (path / n4.N4_FIELD_FILENAME).write_bytes(b"field")
    (path / marker).write_text(marker)


@pytest.mark.gpu
def test_run_n4_preserves_ngff_metadata_and_rebuilds_pyramid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cupy as cp

    source_path = tmp_path / "fused.ch2.ome.zarr"
    source, field = _write_ome_zarr(source_path, with_provenance=True)
    source_attrs = copy.deepcopy(source.attrs.asdict())
    source_level_metadata = {path: copy.deepcopy(source[path].metadata.to_dict()) for path in ("0", "1")}

    monkeypatch.setattr(n4, "_estimate_n4_field", lambda *_args, **_kwargs: cp.asarray(field))
    output_path = tmp_path / "corrected.ome.zarr"

    result = n4.run_n4_ome_zarr(
        source_path,
        output_path,
        n4.N4Config(field_level=0, shrink=2, unsharp=False),
    )

    assert result == output_path
    output = zarr.open_group(output_path, mode="r")
    for key, value in source_attrs.items():
        assert output.attrs[key] == value
    assert output.attrs["squisher_complete"] is True
    provenance = output.attrs["squisher_n4"]
    assert provenance["schema_version"] == 1
    assert provenance["source"] == str(source_path.resolve())
    assert provenance["field_dataset"] == "0"
    assert provenance["field_shape"] == [16, 32, 32]
    assert provenance["field_path"] == "n4-field.npy"
    assert provenance["quantization"]["reference_dataset"] == "0"
    assert provenance["field_sha256"] == n4._file_sha256(output_path / "n4-field.npy")
    assert np.load(output_path / "n4-field.npy").shape == (16, 32, 32)
    assert (output_path / "provenance/manifest.json").read_text() == (
        source_path / "provenance/manifest.json"
    ).read_text()
    assert (output_path / "provenance/input.json").read_text() == (
        source_path / "provenance/input.json"
    ).read_text()

    for path in ("0", "1"):
        source_meta = source_level_metadata[path]
        output_meta = output[path].metadata.to_dict()
        assert output_meta == source_meta
    assert output["0"].attrs["level_note"] == "copy me"

    source_level0 = np.asarray(source["0"], dtype=np.float32)
    corrected = source_level0 / field
    quantization = provenance["quantization"]
    expected_level0 = np.rint(
        np.clip(
            (corrected - quantization["lower"]) * quantization["scale"],
            0,
            np.iinfo(np.uint16).max - 1,
        )
    ).astype(np.uint16)
    expected_level0[corrected > quantization["upper"]] = np.iinfo(np.uint16).max
    expected_level0[source_level0 == 0] = 0
    np.testing.assert_allclose(np.asarray(output["0"]), expected_level0, rtol=0, atol=1)

    expected_level1 = _downsample_mean_numpy(
        np.asarray(output["0"]),
        factors=(2, 2, 2),
        dtype=np.dtype(np.uint16),
    )
    np.testing.assert_array_equal(np.asarray(output["1"]), expected_level1)
    assert not np.all(np.asarray(output["1"]) == np.iinfo(np.uint16).max)


def test_run_n4_rejects_incomplete_source(tmp_path: Path) -> None:
    source_path = tmp_path / "fused.ch0.ome.zarr"
    _write_ome_zarr(source_path, complete=False)

    with pytest.raises(ValueError, match="not marked complete"):
        n4.run_n4_ome_zarr(
            source_path,
            tmp_path / "output.ome.zarr",
            n4.N4Config(unsharp=False),
        )


def test_n4_cli_uses_compact_channel_output_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "long-sample-name.fused.ch2.ome.zarr"
    source_path.mkdir()
    captured: dict[str, object] = {}

    def fake_run(source: Path, output: Path, config: n4.N4Config, *, overwrite: bool) -> Path:
        captured.update(source=source, output=output, config=config, overwrite=overwrite)
        return output

    monkeypatch.setattr(n4, "run_n4_ome_zarr", fake_run)
    result = CliRunner().invoke(
        app,
        ["n4", str(source_path), "--field-level", "1", "--no-unsharp"],
    )

    assert result.exit_code == 0, result.output
    assert captured["source"] == source_path
    assert captured["output"] == tmp_path / "fused-n4.ch2.ome.zarr"
    assert captured["config"] == n4.N4Config(field_level=1, unsharp=False)
    assert captured["overwrite"] is False


def test_default_spline_spacing_matches_lightsheet_voxels() -> None:
    config = n4.N4Config()
    geometry = n4.ImageGeometry((0.6, 0.3, 0.3), (0.0, 0.0, 0.0))

    assert config.spline_lowres_px_zyx == (24.0, 48.0, 48.0)
    assert n4._spline_spacing_zyx(config, geometry) == pytest.approx((57.6, 57.6, 57.6))


@pytest.mark.gpu
def test_field_level_does_not_change_physical_spline_spacing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    source_path = tmp_path / "fused.ch0.ome.zarr"
    _write_ome_zarr(source_path)
    calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    def fake_estimator(image: object, **kwargs: object) -> object:
        calls.append((kwargs["spacing_zyx"], kwargs["spline_spacing_zyx"]))
        return cp.ones(cp.asarray(image).shape, dtype=cp.float32)

    monkeypatch.setattr(n4, "_estimate_n4_field", fake_estimator)
    for field_level in (0, 1):
        n4.run_n4_ome_zarr(
            source_path,
            tmp_path / f"n4-{field_level}.ome.zarr",
            n4.N4Config(field_level=field_level),
        )

    assert calls[0][0] == pytest.approx((0.6, 0.3, 0.3))
    assert calls[1][0] == pytest.approx((1.2, 0.6, 0.6))
    assert calls[0][1] == pytest.approx((57.6, 57.6, 57.6))
    assert calls[1][1] == pytest.approx((57.6, 57.6, 57.6))


def test_n4_cli_rejects_named_threshold(tmp_path: Path) -> None:
    source_path = tmp_path / "fused.ch0.ome.zarr"
    source_path.mkdir()

    result = CliRunner().invoke(app, ["n4", str(source_path), "--threshold", "sauvola"])

    assert result.exit_code == 2
    assert "threshold must be numeric" in result.output


@pytest.mark.gpu
def test_estimate_n4_field_is_positive_and_normalized() -> None:
    import cupy as cp

    z, y, x = np.mgrid[-1:1:16j, -1:1:32j, -1:1:32j]
    mask = (x * x + y * y) < 0.8
    bias = (0.8 + 0.2 * (z + 1.0) / 2.0) * (0.7 + 0.3 * (x + 1.0) / 2.0)
    image = np.where(mask, 1_000.0 * bias, 0.0).astype(np.float32)

    field = n4._estimate_n4_field(
        image,
        spacing_zyx=(2.0, 1.0, 1.0),
        spline_spacing_zyx=(16.0, 8.0, 8.0),
        shrink=2,
        iterations=(10, 5),
        threshold=None,
    )

    field = cp.asnumpy(field)
    assert field.shape == image.shape
    assert field.dtype == np.float32
    assert np.all(np.isfinite(field))
    assert np.all(field > 0)
    assert np.median(field[mask]) == pytest.approx(1.0, abs=1e-3)
    assert np.all(field[~mask] == 1.0)
    foreground_z = np.median(field, axis=(1, 2))
    assert np.ptp(foreground_z) > 0.01
    before = np.std(np.median(image[:, mask[0]], axis=1))
    after = np.std(np.median((image / field)[:, mask[0]], axis=1))
    assert after < before


@pytest.mark.gpu
def test_estimate_n4_field_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="ZYX volume"):
        n4._estimate_n4_field(
            np.ones((8, 8), dtype=np.float32),
            spacing_zyx=(1.0, 1.0, 1.0),
            spline_spacing_zyx=(8.0, 8.0, 8.0),
            shrink=2,
            iterations=(2,),
            threshold=None,
        )


@pytest.mark.gpu
def test_estimate_n4_field_rejects_axis_collapsed_by_shrink() -> None:
    with pytest.raises(ValueError, match="at least two voxels per axis"):
        n4._estimate_n4_field(
            np.ones((7, 16, 16), dtype=np.float32),
            spacing_zyx=(1.0, 1.0, 1.0),
            spline_spacing_zyx=(8.0, 8.0, 8.0),
            shrink=4,
            iterations=(2,),
            threshold=None,
        )


@pytest.mark.gpu
def test_threshold_mask_uses_gpu_numeric_cutoff() -> None:
    import cupy as cp

    volume_gpu = cp.asarray([[[0.0, 1.0, 2.0, cp.nan]]], dtype=cp.float32)

    np.testing.assert_array_equal(
        cp.asnumpy(n4._threshold_mask(volume_gpu, 1.0)),
        np.asarray([[[False, False, True, False]]]),
    )


@pytest.mark.parametrize("value", ["otsu", "yen", "triangle"])
def test_parse_threshold_rejects_named_methods(value: str) -> None:
    with pytest.raises(ValueError, match="threshold must be numeric"):
        n4._parse_threshold(value)


def test_validate_source_rejects_inconsistent_pyramid_scale(tmp_path: Path) -> None:
    source_path = tmp_path / "fused.ch0.ome.zarr"
    source, _ = _write_ome_zarr(source_path)
    attrs = source.attrs.asdict()
    attrs["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]["scale"] = [
        5.0,
        1.0,
        1.0,
    ]
    source.attrs.update(attrs)

    with pytest.raises(ValueError, match="scale does not match"):
        n4._validate_source(source, source_path, n4.N4Config(unsharp=False))


def test_dataset_geometry_composes_multiscale_transform(tmp_path: Path) -> None:
    source_path = tmp_path / "fused.ch0.ome.zarr"
    source, _ = _write_ome_zarr(source_path)
    attrs = source.attrs.asdict()
    attrs["ome"]["multiscales"][0]["coordinateTransformations"] = [
        {"type": "scale", "scale": [2.0, 3.0, 4.0]},
        {"type": "translation", "translation": [5.0, 6.0, 7.0]},
    ]
    source.attrs.update(attrs)

    geometry = n4._dataset_geometry(source, source_path, "0")

    assert geometry.scale_zyx == pytest.approx((1.2, 0.9, 1.2))
    assert geometry.translation_zyx == pytest.approx((7.0, 12.0, 19.0))


def test_validate_source_rejects_non_micrometer_axes(tmp_path: Path) -> None:
    source_path = tmp_path / "fused.ch0.ome.zarr"
    source, _ = _write_ome_zarr(source_path)
    attrs = source.attrs.asdict()
    attrs["ome"]["multiscales"][0]["axes"][0]["unit"] = "millimeter"
    source.attrs.update(attrs)

    with pytest.raises(ValueError, match="micrometer units"):
        n4._validate_source(source, source_path, n4.N4Config())


def test_source_sidecars_reject_unreferenced_auxiliary_nodes(tmp_path: Path) -> None:
    source_path = tmp_path / "fused.ch0.ome.zarr"
    source, _ = _write_ome_zarr(source_path)
    (source_path / "labels").mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(ValueError, match="unsupported auxiliary store nodes"):
        n4._copy_source_sidecars(
            source_path,
            destination,
            dataset_paths=["0", "1"],
            root_attrs=source.attrs.asdict(),
        )


def test_publish_never_clobbers_concurrent_destination(tmp_path: Path) -> None:
    output = tmp_path / "output.ome.zarr"
    staged = tmp_path / "staged.tmp"
    staged.mkdir()
    (staged / "new").write_text("new")
    initial_identity = n4._path_identity(output)
    output.mkdir()
    (output / "owner").write_text("concurrent")

    with pytest.raises(FileExistsError, match="created while N4 was running"):
        n4._publish_staged(staged, output, overwrite=False, initial_identity=initial_identity)

    assert (output / "owner").read_text() == "concurrent"


def test_publish_overwrite_rejects_changed_destination(tmp_path: Path) -> None:
    output = tmp_path / "output.ome.zarr"
    output.mkdir()
    initial_identity = n4._path_identity(output)
    output.rmdir()
    output.mkdir()
    staged = tmp_path / "staged.tmp"
    staged.mkdir()

    with pytest.raises(RuntimeError, match="changed while N4 was running"):
        n4._publish_staged(staged, output, overwrite=True, initial_identity=initial_identity)


def test_publish_overwrite_exchanges_complete_directories(tmp_path: Path) -> None:
    output = tmp_path / "output.ome.zarr"
    _write_completed_n4_output(output, "old")
    initial_identity = n4._path_identity(output)
    staged = tmp_path / "staged.tmp"
    staged.mkdir()
    (staged / "new").write_text("new")

    n4._publish_staged(staged, output, overwrite=True, initial_identity=initial_identity)

    assert (output / "new").read_text() == "new"
    assert not staged.exists()


def test_publish_rolls_back_replaced_overwrite_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output.ome.zarr"
    replacement = tmp_path / "replacement.ome.zarr"
    displaced = tmp_path / "displaced.ome.zarr"
    staged = tmp_path / "staged.tmp"
    _write_completed_n4_output(output, "original")
    _write_completed_n4_output(replacement, "replacement")
    staged.mkdir()
    (staged / "new").write_text("new")
    initial_identity = n4._path_identity(output)
    validate = n4._validate_overwrite_target

    def replace_after_validation(path: Path) -> None:
        validate(path)
        path.rename(displaced)
        replacement.rename(path)

    monkeypatch.setattr(n4, "_validate_overwrite_target", replace_after_validation)

    with pytest.raises(RuntimeError, match="replaced while N4 was publishing"):
        n4._publish_staged(staged, output, overwrite=True, initial_identity=initial_identity)

    assert (output / "replacement").read_text() == "replacement"
    assert (staged / "new").read_text() == "new"
    assert (displaced / "original").read_text() == "original"


def test_publish_overwrite_rejects_foreign_directory(tmp_path: Path) -> None:
    output = tmp_path / "output.ome.zarr"
    output.mkdir()
    staged = tmp_path / "staged.tmp"
    staged.mkdir()

    with pytest.raises(ValueError, match="non-N4 output"):
        n4._publish_staged(
            staged,
            output,
            overwrite=True,
            initial_identity=n4._path_identity(output),
        )

    assert output.exists()
    assert staged.exists()


@pytest.mark.gpu
def test_quantization_expands_degenerate_range() -> None:
    import cupy as cp

    sample = cp.full(32, 10.0, dtype=cp.float32)
    params = n4._quantization_params_from_sample(sample, population=sample.size)
    assert params.upper > params.lower


@pytest.mark.gpu
def test_quantization_uses_exact_full_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    source = zarr.open_array(
        tmp_path / "source.zarr",
        mode="w",
        shape=(1, 500, 500),
        chunks=(1, 100, 100),
        dtype=np.uint16,
    )
    selection = (slice(0, 1), slice(0, 100), slice(0, 100))
    source[:] = 0
    source[selection] = 1
    monkeypatch.setattr(n4, "_quantization_windows", lambda *_args: [selection])
    monkeypatch.setattr(
        n4,
        "_correct_selection_gpu",
        lambda *_args, **_kwargs: (
            cp.linspace(1, 100, 10_000, dtype=cp.float32).reshape(1, 100, 100),
            cp.ones((1, 100, 100), dtype=cp.bool_),
        ),
    )

    params = n4._level0_quantization_params(
        source,
        cp.ones((1, 1, 1), dtype=cp.float32),
        n4.ImageGeometry((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        n4.ImageGeometry((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        n4.N4Config(),
    )

    assert params.population_count == 10_000
    assert params.upper_percentile == n4.QUANT_FALLBACK_UPPER_PERCENTILE


def test_quantization_windows_are_disjoint() -> None:
    windows = n4._quantization_windows((4, 100, 100), (4, 100, 100))

    total = sum(
        math.prod(part.stop - part.start for part in selection)
        for selection in windows
    )
    assert total <= 4 * 100 * 100
    for index, left in enumerate(windows):
        for right in windows[index + 1 :]:
            assert any(
                left_axis.stop <= right_axis.start or right_axis.stop <= left_axis.start
                for left_axis, right_axis in zip(left, right, strict=True)
            )


@pytest.mark.gpu
def test_quantization_samples_corrected_level0_not_field_level(tmp_path: Path) -> None:
    import cupy as cp

    source_path = tmp_path / "fused.ch0.ome.zarr"
    source, _ = _write_ome_zarr(source_path)
    data = np.full(source["0"].shape, 100, dtype=np.uint16)
    data[::2, ::2, ::2] = 1_000
    source["0"][:] = data
    geometry = n4._dataset_geometry(source, source_path, "0")
    field_gpu = cp.ones(data.shape, dtype=cp.float32)

    params = n4._level0_quantization_params(
        source["0"],
        field_gpu,
        geometry,
        geometry,
        n4.N4Config(field_level=0, shrink=2, unsharp=False),
    )

    assert params.upper > 900


@pytest.mark.gpu
def test_gpu_correction_and_pyramid_reduction() -> None:
    import cupy as cp

    n4._ensure_gpu()
    block = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4) + 10
    field = np.linspace(0.5, 1.5, block.size, dtype=np.float32).reshape(block.shape)

    corrected_gpu, _ = n4._correct_gpu(block, cp.asarray(field), unsharp=False)
    corrected = cp.asnumpy(corrected_gpu)
    np.testing.assert_allclose(corrected, block.astype(np.float32) / field)

    reduced = n4._downsample_mean(block, factors=(1, 2, 2), dtype=np.dtype(np.uint16))
    expected = _downsample_mean_numpy(block, factors=(1, 2, 2), dtype=np.dtype(np.uint16))
    np.testing.assert_array_equal(reduced, expected)

    z, y, x = np.mgrid[:3, :3, :3]
    lowres_field = (2 * z + 3 * y + 5 * x).astype(np.float32)
    upsampled = cp.asnumpy(
        n4._resample_field_gpu(
            cp.asarray(lowres_field),
            n4.ImageGeometry((4.0, 2.0, 1.0), (10.0, 20.0, 30.0)),
            n4.ImageGeometry((2.0, 1.0, 0.5), (12.0, 21.0, 30.5)),
            (slice(0, 2), slice(0, 2), slice(0, 2)),
        )
    )
    target_z, target_y, target_x = np.mgrid[:2, :2, :2]
    expected_field = (
        2 * (0.5 + 0.5 * target_z) + 3 * (0.5 + 0.5 * target_y) + 5 * (0.5 + 0.5 * target_x)
    ).astype(np.float32)
    np.testing.assert_allclose(upsampled, expected_field)
