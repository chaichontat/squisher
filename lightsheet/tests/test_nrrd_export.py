from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from typer.testing import CliRunner
import zarr
from zarr.codecs import BytesCodec, ZstdCodec

import squisher_lightsheet.cli as cli_module
from squisher_lightsheet.cli import app
from squisher_lightsheet.nrrd_export import export_nrrd


def _write_source(
    path: Path, *, complete: bool = True, dtype: np.dtype | type[np.generic] = np.uint16
) -> list[np.ndarray]:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": "fixture",
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
                                    {"type": "translation", "translation": [-1.2, 2.4, -3.6]},
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.2, 0.6, 0.6]},
                                    {"type": "translation", "translation": [-1.2, 2.4, -3.6]},
                                ],
                            },
                        ],
                    }
                ],
            },
            "squisher_complete": complete,
        }
    )
    values = []
    for level, shape in enumerate(((5, 8, 10), (3, 4, 5))):
        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape) + level * 1000
        array = root.create_array(
            str(level),
            shape=shape,
            chunks=(1, 4, 5),
            dtype=dtype,
            attributes={"_ARRAY_DIMENSIONS": ["z", "y", "x"]},
            dimension_names=("z", "y", "x"),
            serializer=BytesCodec(),
            compressors=[ZstdCodec(level=1)],
        )
        array[:] = data
        values.append(data)
    return values


def test_export_nrrd_streams_selected_level_with_exact_voxels_and_transform(tmp_path: Path) -> None:
    source = tmp_path / "source.ome.zarr"
    expected = _write_source(source)[1]
    output = tmp_path / "561.nrrd"
    messages = []

    summary = export_nrrd(
        source=source,
        output=output,
        level=1,
        channel="561",
        gzip_level=1,
        slab_planes=2,
        progress=messages.append,
    )

    image = sitk.ReadImage(output)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), expected)
    np.testing.assert_allclose(image.GetSpacing(), (0.0006, 0.0006, 0.0012), rtol=0, atol=1e-12)
    np.testing.assert_allclose(image.GetOrigin(), (-0.0036, 0.0024, -0.0012), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(image.GetDirection()).reshape(3, 3), np.eye(3))
    assert image.GetMetaData("channel") == "561"
    assert image.GetMetaData("source_level") == "1"
    assert summary["shape_zyx"] == [3, 4, 5]
    assert summary["all_voxels_equal"] is True
    assert summary["nrrd_compression"] == "gzip level 1"
    assert messages[-1] == "Validating NRRD payload and physical metadata"
    with output.open("rb") as handle:
        while handle.readline() not in {b"\n", b"\r\n"}:
            pass
        with gzip.GzipFile(fileobj=handle) as decoded:
            assert decoded.read() == expected.tobytes()


def test_export_nrrd_rejects_incomplete_source_and_existing_output(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.ome.zarr"
    _write_source(incomplete, complete=False)
    with pytest.raises(ValueError, match="explicitly marked incomplete"):
        export_nrrd(source=incomplete, output=tmp_path / "incomplete.nrrd")

    source = tmp_path / "source.ome.zarr"
    _write_source(source)
    output = tmp_path / "existing.nrrd"
    output.write_bytes(b"owned")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_nrrd(source=source, output=output)
    assert output.read_bytes() == b"owned"


def test_export_nrrd_rejects_non_uint16_source(tmp_path: Path) -> None:
    source = tmp_path / "float.ome.zarr"
    _write_source(source, dtype=np.float32)

    with pytest.raises(ValueError, match="Expected uint16"):
        export_nrrd(source=source, output=tmp_path / "float.nrrd")


def test_export_nrrd_cli_forwards_contract(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.ome.zarr"
    source.mkdir()
    output = tmp_path / "561.nrrd"
    summary_output = tmp_path / "561.json"
    captured = {}

    def fake_export(**kwargs):
        captured.update(kwargs)
        return {"output": str(output), "all_voxels_equal": True}

    monkeypatch.setattr(cli_module, "export_nrrd", fake_export)
    result = CliRunner().invoke(
        app,
        [
            "export-nrrd",
            str(source),
            str(output),
            "--level",
            "3",
            "--channel",
            "561",
            "--gzip-level",
            "2",
            "--slab-planes",
            "12",
            "--summary-output",
            str(summary_output),
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == str(summary_output.resolve())
    assert summary_output.read_text().endswith("\n")
    assert captured == {
        "source": source,
        "output": output,
        "level": 3,
        "channel": "561",
        "gzip_level": 2,
        "slab_planes": 12,
        "progress": cli_module._log_progress,
    }
