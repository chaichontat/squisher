from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import squisher.tiff_pyramid as tiff_pyramid
import tifffile


def load_module():
    return tiff_pyramid


def test_default_output_path_uses_new_sibling_folder() -> None:
    module = load_module()

    assert module.default_output_path(Path("/data/sample.ome.tif")) == Path(
        "/data/sample.pyramid/sample.ome.tif"
    )
    assert module.default_output_path(Path("/data/sample.ome.tiff")) == Path(
        "/data/sample.pyramid/sample.ome.tiff"
    )


def test_write_pyramid_adds_subifds_with_source_compression(tmp_path: Path) -> None:
    write_pyramid = load_module().write_pyramid
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "pyramid.ome.tif"
    data = np.arange(3 * 33 * 35, dtype=np.uint16).reshape(3, 33, 35)
    tifffile.imwrite(
        source,
        data,
        ome=True,
        metadata={"axes": "ZYX"},
        compression="zlib",
        predictor=True,
        resolution=((10, 1), (20, 1)),
        resolutionunit="CENTIMETER",
        tile=(16, 16),
    )

    write_pyramid(
        source,
        output,
        factor=2,
        overwrite=False,
        gpu_batch_size=2,
        tiff_maxworkers=2,
    )

    with tifffile.TiffFile(source) as source_tif, tifffile.TiffFile(output) as tif:
        assert tif.is_ome
        assert tif.is_bigtiff
        assert tif.series[0].shape == data.shape
        assert tif.series[0].axes == "ZYX"
        assert len(tif.pages) == data.shape[0]

        for plane_index, page in enumerate(tif.series[0].pages):
            source_page = source_tif.series[0].pages[plane_index]
            assert page.compression == tifffile.COMPRESSION.ADOBE_DEFLATE
            assert page.databytecounts == source_page.databytecounts
            np.testing.assert_array_equal(page.asarray(), data[plane_index])
            assert len(page.subifds) == 2
            for level_index, offset in enumerate(page.subifds):
                tif.filehandle.seek(offset)
                subifd = tifffile.TiffPage(tif, (plane_index, level_index), keyframe=tif.pages[0])
                assert subifd.is_subifd
                assert subifd.compression == tifffile.COMPRESSION.ADOBE_DEFLATE
                assert subifd.shape == ((17, 18), (9, 9))[level_index]
                assert subifd.tags["XResolution"].value == ((5, 1), (5, 2))[level_index]
                assert subifd.tags["YResolution"].value == ((10, 1), (5, 1))[level_index]


def test_write_pyramid_skips_existing_pyramid_output(tmp_path: Path) -> None:
    write_pyramid = load_module().write_pyramid
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "pyramid.ome.tif"
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    tifffile.imwrite(source, data, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))

    write_pyramid(source, output, factor=2, overwrite=False)
    write_pyramid(source, output, factor=2, overwrite=False)

    with tifffile.TiffFile(output) as tif:
        assert len(tif.series[0].pages[0].subifds) == 2


def test_write_pyramid_skips_pyramidal_source(tmp_path: Path) -> None:
    write_pyramid = load_module().write_pyramid
    source = tmp_path / "source.ome.tif"
    pyramidal_source = tmp_path / "pyramid.ome.tif"
    output = tmp_path / "again.ome.tif"
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    tifffile.imwrite(source, data, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))

    write_pyramid(source, pyramidal_source, factor=2, overwrite=False)
    write_pyramid(pyramidal_source, output, factor=2, overwrite=False)

    assert not output.exists()


def test_write_pyramid_overwrites_source_via_temp_output(tmp_path: Path) -> None:
    write_pyramid = load_module().write_pyramid
    source = tmp_path / "source.ome.tif"
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    tifffile.imwrite(source, data, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))

    write_pyramid(source, source, factor=2, overwrite=True)

    with tifffile.TiffFile(source) as tif:
        assert tif.is_ome
        assert len(tif.series[0].pages[0].subifds) == 2


def test_write_pyramid_reduces_full_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "pyramid.ome.tif"
    data = np.arange(3 * 32 * 32, dtype=np.uint16).reshape(3, 32, 32)
    tifffile.imwrite(source, data, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))
    original_pyramid_reductions = module.pyramid_reductions
    reduced_shapes = []

    def record_pyramid_reductions(*args, **kwargs):
        reduced_shapes.append(args[0].shape)
        return original_pyramid_reductions(*args, **kwargs)

    monkeypatch.setattr(module, "pyramid_reductions", record_pyramid_reductions)

    module.write_pyramid(source, output, factor=2, overwrite=False, gpu_batch_size=2)

    assert reduced_shapes == [(2, 32, 32), (1, 32, 32)]


def test_write_pyramid_removes_overwrite_temp_after_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    source = tmp_path / "source.ome.tif"
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    tifffile.imwrite(source, data, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))

    def fail_subifd_write(*_args, **_kwargs):
        raise RuntimeError("stop after temp creation")

    monkeypatch.setattr(module, "write_subifd_pyramid", fail_subifd_write)

    with pytest.raises(RuntimeError, match="stop after temp creation"):
        module.write_pyramid(
            source,
            source,
            factor=2,
            overwrite=True,
            temp_token="cleanup-token",
        )

    assert list(tmp_path.glob(".source.ome.tif.cleanup-token.*.tmp")) == []
    with tifffile.TiffFile(source) as tif:
        assert tif.series[0].shape == data.shape


def test_cleanup_temp_outputs_removes_only_current_run_overwrite_temps(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "output.ome.tif"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    current_temp = tmp_path / ".source.ome.tif.current.abc.tmp"
    other_run_temp = tmp_path / ".source.ome.tif.other.abc.tmp"
    non_overwrite_temp = tmp_path / ".output.ome.tif.current.abc.tmp"
    for path in (current_temp, other_run_temp, non_overwrite_temp):
        path.write_bytes(b"tmp")

    module.cleanup_temp_outputs([(source, source), (source, output)], "current")

    assert not current_temp.exists()
    assert other_run_temp.exists()
    assert non_overwrite_temp.exists()


def test_directory_input_writes_each_tiff_to_output_dir(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    source_dir.mkdir()
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    first = source_dir / "first.ome.tif"
    second = source_dir / "second.ome.tiff"
    ignored = source_dir / "notes.txt"
    tifffile.imwrite(first, data, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))
    tifffile.imwrite(second, data + 1, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))
    ignored.write_text("not a tiff")

    jobs = module.pyramid_jobs(source_dir, output=None, output_dir=output_dir, overwrite=False)

    assert jobs == [
        (first.resolve(), output_dir.resolve() / "first.ome.tif"),
        (second.resolve(), output_dir.resolve() / "second.ome.tiff"),
    ]
    for source, output in jobs:
        module.write_pyramid(source, output, factor=2, overwrite=False)

    for output_name in ("first.ome.tif", "second.ome.tiff"):
        with tifffile.TiffFile(output_dir / output_name) as tif:
            assert tif.is_ome
            assert len(tif.series[0].pages[0].subifds) == 2


def test_jpegxr_subifds_use_hardcoded_compression_level() -> None:
    module = load_module()
    kwargs = module.subifd_write_kwargs(
        {
            "compression": tifffile.COMPRESSION.JPEGXR_NDPI,
            "description": b"ome",
            "subifds": 1,
        },
        scale=2,
    )

    assert kwargs["compressionargs"] == {"level": 0.65}
    assert kwargs["description"] is None
    assert kwargs["subifds"] is None
    assert kwargs["subfiletype"] == 1


def test_terminate_process_pool_terminates_alive_workers() -> None:
    module = load_module()

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.joined = False

        def is_alive(self) -> bool:
            return not self.killed

        def terminate(self) -> None:
            self.terminated = True

        def join(self, timeout: int) -> None:
            assert timeout == 1
            self.joined = True

        def kill(self) -> None:
            self.killed = True

    class FakeExecutor:
        def __init__(self) -> None:
            self.process = FakeProcess()
            self._processes = {1: self.process}
            self.shutdown_kwargs = None

        def shutdown(self, **kwargs) -> None:
            self.shutdown_kwargs = kwargs

    executor = FakeExecutor()

    module.terminate_process_pool(executor)

    assert executor.process.terminated
    assert executor.process.joined
    assert executor.process.killed
    assert executor.shutdown_kwargs == {"wait": False, "cancel_futures": True}


def test_directory_input_rejects_single_output_file(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()

    with pytest.raises(ValueError, match="--output can only be used with a single input file"):
        module.pyramid_jobs(source_dir, output=tmp_path / "one.ome.tif", output_dir=None, overwrite=False)


def test_directory_input_skips_empty_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = load_module()
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()

    assert module.pyramid_jobs(source_dir, output=None, output_dir=None, overwrite=False) == []

    captured = capsys.readouterr()
    assert f"Skipping {source_dir.resolve()}: no TIFF files found" in captured.out


def test_main_exits_cleanly_when_all_inputs_are_empty_folders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_module()
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(
            inputs=[source_dir],
            output=None,
            output_dir=None,
            factor=2,
            overwrite=False,
            gpu_batch_size=module.DEFAULT_GPU_BATCH_SIZE,
            tiff_maxworkers=None,
            file_workers=8,
        ),
    )

    assert module.main() == 0

    captured = capsys.readouterr()
    assert "No TIFF files to process" in captured.out


def test_directory_input_defaults_to_one_sibling_output_folder(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    first = source_dir / "first.ome.tif"
    second = source_dir / "second.ome.tif"
    first.write_bytes(b"")
    second.write_bytes(b"")

    jobs = module.pyramid_jobs(source_dir, output=None, output_dir=None, overwrite=False)

    assert jobs == [
        (first.resolve(), tmp_path / "inputs.pyramid" / "first.ome.tif"),
        (second.resolve(), tmp_path / "inputs.pyramid" / "second.ome.tif"),
    ]


def test_directory_input_with_overwrite_targets_sources(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    first = source_dir / "first.ome.tif"
    second = source_dir / "second.ome.tif"
    first.write_bytes(b"")
    second.write_bytes(b"")

    jobs = module.pyramid_jobs(source_dir, output=None, output_dir=None, overwrite=True)

    assert jobs == [
        (first.resolve(), first.resolve()),
        (second.resolve(), second.resolve()),
    ]


def test_multiple_inputs_expand_files_and_folders(tmp_path: Path) -> None:
    module = load_module()
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    folder_file = source_dir / "folder.ome.tif"
    direct_file = tmp_path / "direct.ome.tif"
    folder_file.write_bytes(b"")
    direct_file.write_bytes(b"")

    jobs = module.expand_pyramid_jobs([direct_file, source_dir], output=None, output_dir=None, overwrite=True)

    assert jobs == [
        (direct_file.resolve(), direct_file.resolve()),
        (folder_file.resolve(), folder_file.resolve()),
    ]


def test_multiple_inputs_reject_single_output_file(tmp_path: Path) -> None:
    module = load_module()

    with pytest.raises(ValueError, match="--output can only be used with a single input path"):
        module.expand_pyramid_jobs(
            [tmp_path / "a.ome.tif", tmp_path / "b.ome.tif"],
            output=tmp_path / "out.ome.tif",
            output_dir=None,
            overwrite=False,
        )


def test_multiple_inputs_reject_duplicate_outputs(tmp_path: Path) -> None:
    module = load_module()
    first = tmp_path / "a" / "same.ome.tif"
    second = tmp_path / "b" / "same.ome.tif"
    output_dir = tmp_path / "outputs"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"")
    second.write_bytes(b"")

    with pytest.raises(ValueError, match="Multiple inputs resolve to the same output path"):
        module.expand_pyramid_jobs([first, second], output=None, output_dir=output_dir, overwrite=False)


def test_cli_processes_directory_with_file_workers(tmp_path: Path) -> None:
    source_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    source_dir.mkdir()
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    tifffile.imwrite(
        source_dir / "first.ome.tif",
        data,
        ome=True,
        metadata={"axes": "ZYX"},
        compression="zlib",
        tile=(16, 16),
    )
    tifffile.imwrite(
        source_dir / "second.ome.tif",
        data + 1,
        ome=True,
        metadata={"axes": "ZYX"},
        compression="zlib",
        tile=(16, 16),
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "squisher.tiff_pyramid",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--file-workers",
            "2",
        ],
        check=True,
    )

    for output_name in ("first.ome.tif", "second.ome.tif"):
        with tifffile.TiffFile(output_dir / output_name) as tif:
            assert tif.is_ome
            assert len(tif.series[0].pages[0].subifds) == 2


def test_cli_processes_multiple_explicit_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    first = tmp_path / "first.ome.tif"
    second = tmp_path / "second.ome.tif"
    tifffile.imwrite(first, data, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))
    tifffile.imwrite(second, data + 1, ome=True, metadata={"axes": "ZYX"}, compression="zlib", tile=(16, 16))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "squisher.tiff_pyramid",
            str(first),
            str(second),
            "--output-dir",
            str(output_dir),
            "--file-workers",
            "2",
        ],
        check=True,
    )

    for output_name in ("first.ome.tif", "second.ome.tif"):
        with tifffile.TiffFile(output_dir / output_name) as tif:
            assert tif.is_ome
            assert len(tif.series[0].pages[0].subifds) == 2
