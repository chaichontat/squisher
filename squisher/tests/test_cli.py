from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import squisher
from squisher import DEFAULT_MIN_ZARR_CHUNK_PIXELS, DEFAULT_ZARR_CHUNKS_TCZYX
from squisher import app


def test_cli_exposes_compress_subcommand() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "compress" in result.stdout
    assert "compare" in result.stdout
    assert "pyramid" in result.stdout
    assert "video" in result.stdout
    assert "verify" in result.stdout


def test_pyramid_cli_uses_fixed_two_subifd_levels() -> None:
    result = CliRunner().invoke(app, ["pyramid", "--help"])

    assert result.exit_code == 0
    assert "--levels" not in result.stdout
    assert "--min-size" not in result.stdout


def test_readme_documents_pyramid_subcommand() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    assert "uv run squisher pyramid" in readme
    assert "python -m squisher.tiff_pyramid" not in readme


def test_compress_uses_expected_defaults(tmp_path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"not checked by fake compressor")
    captured = {}

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return True

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(app, ["compress", str(czi_path), "--output-format", "ome-zarr"])

    assert result.exit_code == 0
    assert captured["level"] == 0.7
    assert captured["tile_size"] is None
    assert captured["maxworkers"] == 8
    assert captured["tile_workers"] == 8
    assert captured["zarr_chunks"] == DEFAULT_ZARR_CHUNKS_TCZYX
    assert captured["min_zarr_chunk_pixels"] == DEFAULT_MIN_ZARR_CHUNK_PIXELS
    assert captured["pos_path"] is None
    assert captured["pyramid"] is True
    assert czi_path.exists()


def test_compress_forwards_options(tmp_path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    pos_path = tmp_path / "sample.pos"
    czi_path.write_bytes(b"not checked by fake compressor")
    pos_path.write_text("not checked by fake compressor")
    captured = {}

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return True

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(
        app,
        [
            "compress",
            str(czi_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--output-format",
            "ome-zarr",
            "--level",
            "90",
            "--tile-size",
            "512",
            "--zarr-compressor",
            "jpegxr",
            "--zarr-chunk-z",
            "1",
            "--zarr-chunk-y",
            "1024",
            "--zarr-chunk-x",
            "1024",
            "--min-zarr-chunk-pixels",
            "1048576",
            "--tiff-maxworkers",
            "1",
            "--czi-tile-workers",
            "2",
            "--no-thumbnails",
            "--thumbnail-size",
            "256",
            "--pos",
            str(pos_path),
            "--no-pyramid",
        ],
    )

    assert result.exit_code == 0
    assert captured["path"] == czi_path
    assert captured["out_dir"] == tmp_path / "out"
    assert captured["output_format"] == "ome-zarr"
    assert captured["level"] == 90.0
    assert captured["tile_size"] == 512
    assert captured["zarr_chunks"] == (1, 1, 1, 1024, 1024)
    assert captured["min_zarr_chunk_pixels"] == 1048576
    assert captured["zarr_compressor"] == "jpegxr"
    assert captured["maxworkers"] == 1
    assert captured["tile_workers"] == 2
    assert captured["overwrite"] is False
    assert captured["thumbnails"] is False
    assert captured["thumbnail_size"] == 256
    assert captured["pos_path"] == pos_path
    assert captured["pyramid"] is False


def test_compress_accepts_multiple_czi_paths(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.czi"
    second = tmp_path / "second.czi"
    first.write_bytes(b"not checked by fake compressor")
    second.write_bytes(b"not checked by fake compressor")
    pos_path = tmp_path / "sample.pos"
    pos_path.write_text("not checked by fake compressor")
    captured = []

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        captured.append((path, kwargs["pos_path"]))
        return True

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(
        app, ["compress", str(first), str(second), "--no-thumbnails", "--pos", str(pos_path)]
    )

    assert result.exit_code == 0
    assert captured == [(first, pos_path), (second, pos_path)]


def test_compress_expands_literal_glob_patterns(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.czi"
    second = tmp_path / "second.czi"
    first.write_bytes(b"not checked by fake compressor")
    second.write_bytes(b"not checked by fake compressor")
    (tmp_path / "ignored.txt").write_text("not a czi")
    captured = []

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        captured.append(path)
        return True

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(app, ["compress", str(tmp_path / "*.czi")])

    assert result.exit_code == 0
    assert captured == [first, second]


def test_compress_preserves_literal_glob_metacharacter_filenames(tmp_path, monkeypatch) -> None:
    czi_path = tmp_path / "sample[1].czi"
    czi_path.write_bytes(b"not checked by fake compressor")
    captured = []

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        captured.append(path)
        return True

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(app, ["compress", str(czi_path)])

    assert result.exit_code == 0
    assert captured == [czi_path]


def test_compress_rejects_duplicate_stems_with_shared_out_dir(tmp_path, monkeypatch) -> None:
    first = tmp_path / "a" / "sample.czi"
    second = tmp_path / "b" / "sample.czi"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"not checked by fake compressor")
    second.write_bytes(b"not checked by fake compressor")

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        raise AssertionError("duplicate stems should fail before compression")

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(
        app, ["compress", str(first), str(second), "--out-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 2


def test_validate_unique_output_stems_reports_shared_out_dir_collision() -> None:
    with pytest.raises(typer.BadParameter, match="would collide under --out-dir"):
        squisher._validate_unique_output_stems([Path("/a/sample.czi"), Path("/b/sample.czi")])


def test_compress_delete_removes_source_after_success(tmp_path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"not checked by fake compressor")

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        assert path == czi_path
        return True

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(app, ["compress", str(czi_path), "--delete"])

    assert result.exit_code == 0
    assert not czi_path.exists()


def test_compress_delete_waits_for_full_batch_success(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.czi"
    second = tmp_path / "second.czi"
    first.write_bytes(b"not checked by fake compressor")
    second.write_bytes(b"not checked by fake compressor")

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        if path == second:
            raise RuntimeError("compression failed")
        return True

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(app, ["compress", str(first), str(second), "--delete"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == f"Compression failed for {second}"
    assert isinstance(result.exception.__cause__, RuntimeError)
    assert str(result.exception.__cause__) == "compression failed"
    assert first.exists()
    assert second.exists()


def test_compress_delete_keeps_source_after_failure(tmp_path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"not checked by fake compressor")

    def fake_compress_czi_to_ome_tiff(path, **kwargs):
        assert path == czi_path
        raise RuntimeError("compression failed")

    monkeypatch.setattr("squisher.compress_czi_to_ome_tiff", fake_compress_czi_to_ome_tiff)

    result = CliRunner().invoke(app, ["compress", str(czi_path), "--delete"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == f"Compression failed for {czi_path}"
    assert isinstance(result.exception.__cause__, RuntimeError)
    assert str(result.exception.__cause__) == "compression failed"
    assert czi_path.exists()


def test_compress_rejects_resume_option(tmp_path) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"not checked by fake compressor")

    result = CliRunner().invoke(app, ["compress", str(czi_path), "--resume"])

    assert result.exit_code == 2
    assert "No such option" in result.output


def test_verify_forwards_options(tmp_path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"not checked by fake verifier")
    captured = {}

    def fake_verify_czi_ome_tiff_outputs(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return True

    monkeypatch.setattr("squisher.verify_czi_ome_tiff_outputs", fake_verify_czi_ome_tiff_outputs)

    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(czi_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--decode-samples",
            "--max-sample-mae",
            "3.5",
            "--max-sample-max-abs",
            "25",
        ],
    )

    assert result.exit_code == 0
    assert captured["path"] == czi_path
    assert captured["out_dir"] == tmp_path / "out"
    assert captured["decode_samples"] is True
    assert captured["max_sample_mae"] == 3.5
    assert captured["max_sample_max_abs"] == 25.0


def test_verify_accepts_multiple_czi_paths(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.czi"
    second = tmp_path / "second.czi"
    first.write_bytes(b"not checked by fake verifier")
    second.write_bytes(b"not checked by fake verifier")
    captured = []

    def fake_verify_czi_ome_tiff_outputs(path, **kwargs):
        captured.append((path, kwargs))
        return True

    monkeypatch.setattr("squisher.verify_czi_ome_tiff_outputs", fake_verify_czi_ome_tiff_outputs)

    result = CliRunner().invoke(app, ["verify", str(first), str(second), "--decode-samples"])

    assert result.exit_code == 0
    assert captured == [
        (
            first,
            {
                "out_dir": None,
                "decode_samples": True,
                "max_sample_mae": None,
                "max_sample_max_abs": None,
            },
        ),
        (
            second,
            {
                "out_dir": None,
                "decode_samples": True,
                "max_sample_mae": None,
                "max_sample_max_abs": None,
            },
        ),
    ]


def test_verify_expands_literal_glob_patterns(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.czi"
    second = tmp_path / "second.czi"
    first.write_bytes(b"not checked by fake verifier")
    second.write_bytes(b"not checked by fake verifier")
    (tmp_path / "ignored.txt").write_text("not a czi")
    captured = []

    def fake_verify_czi_ome_tiff_outputs(path, **kwargs):
        captured.append(path)
        return True

    monkeypatch.setattr("squisher.verify_czi_ome_tiff_outputs", fake_verify_czi_ome_tiff_outputs)

    result = CliRunner().invoke(app, ["verify", str(tmp_path / "*.czi")])

    assert result.exit_code == 0
    assert captured == [first, second]


def test_compare_forwards_options(tmp_path, monkeypatch) -> None:
    czi_path = tmp_path / "sample.czi"
    czi_path.write_bytes(b"not checked by fake comparer")
    captured = {}

    def fake_compare_czi_compression(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return tmp_path / "out"

    monkeypatch.setattr("squisher.compare_czi_compression", fake_compare_czi_compression)

    result = CliRunner().invoke(
        app,
        [
            "compare",
            str(czi_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--count",
            "3",
            "--crop-size",
            "128",
            "--min-level",
            "0.65",
            "--max-level",
            "0.90",
            "--level-step",
            "0.05",
            "--seed",
            "123",
            "--tile-size",
            "128",
            "--discard-encoded-tiffs",
            "--t",
            "2",
            "--channel",
            "0",
            "--z",
            "5",
            "--max-attempts",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert captured["path"] == czi_path
    assert captured["out_dir"] == tmp_path / "out"
    assert captured["count"] == 3
    assert captured["crop_size"] == 128
    assert captured["min_level"] == 0.65
    assert captured["max_level"] == 0.90
    assert captured["level_step"] == 0.05
    assert captured["seed"] == 123
    assert captured["tile_size"] == 128
    assert captured["keep_encoded_tiffs"] is False
    assert captured["t"] == 2
    assert captured["channel"] == 0
    assert captured["z"] == 5
    assert captured["max_attempts"] == 10
