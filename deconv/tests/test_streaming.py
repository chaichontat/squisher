from __future__ import annotations

import hashlib
import json
import threading
import time

import numpy as np
import pytest
import tifffile
import zarr

from squisher_deconv.deconvolution import IdentityDeconvolver
from squisher_deconv.process_workers import (
    ProcessRunConfig,
    _process_file,
    _process_file_with_slab_recovery,
    run_process_gpu_streaming_deconv,
)
from squisher_deconv.scaling import ScalingParameters, save_float32_sample, write_scaling_artifacts
from squisher_deconv.source import TiffLogicalSource
from squisher_deconv.streaming import ProcessingError, run_streaming_deconv, sample_scale


def _read_deconv_ome_zarr(path):
    root = zarr.open_group(str(path), mode="r")
    array = root["0"]
    czyx = array[:]
    flattened = np.moveaxis(czyx, 0, 1).reshape(-1, array.shape[-2], array.shape[-1])
    return flattened, root, array


class FailingDeconvolver:
    halo = 0

    def deconvolve(self, volume: np.ndarray) -> np.ndarray:
        raise RuntimeError(f"forced failure for {volume.shape}")


class RecordingDeconvolver:
    halo = 0

    def __init__(self, device: int, calls: list[int]) -> None:
        self._device = device
        self._calls = calls

    def deconvolve(self, volume: np.ndarray) -> np.ndarray:
        self._calls.append(self._device)
        return volume.astype(np.float32, copy=True)

    def deconvolve_core_u16(
        self,
        volume: np.ndarray,
        *,
        core_start: int,
        core_stop: int,
        scaling: ScalingParameters,
    ) -> np.ndarray:
        self._calls.append(self._device)
        core = volume[core_start:core_stop]
        return core.reshape(-1, core.shape[-2], core.shape[-1]).astype(np.uint16, copy=False)


class RecordingU16CoreDeconvolver:
    halo = 0

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int, int, tuple[float, ...], tuple[float, ...]]] = []

    def deconvolve(self, volume: np.ndarray) -> np.ndarray:
        raise AssertionError("u16 run should use deconvolve_core_u16 when the backend exposes it")

    def deconvolve_core_u16(
        self,
        volume: np.ndarray,
        *,
        core_start: int,
        core_stop: int,
        scaling: ScalingParameters,
    ) -> np.ndarray:
        self.calls.append(
            (
                tuple(volume.shape),
                core_start,
                core_stop,
                tuple(float(x) for x in scaling.offset),
                tuple(float(x) for x in scaling.scale),
            )
        )
        core = volume[core_start:core_stop]
        return core.reshape(-1, core.shape[-2], core.shape[-1]).astype(np.uint16, copy=False)


class OutOfMemoryError(RuntimeError):
    pass


OutOfMemoryError.__module__ = "cupy.cuda.memory"


class SlabLimitedU16Deconvolver:
    def __init__(self, *, max_read_z: int) -> None:
        self.max_read_z = max_read_z
        self.read_depths: list[int] = []
        self.release_calls = 0

    def deconvolve_core_u16(
        self,
        volume: np.ndarray,
        *,
        core_start: int,
        core_stop: int,
        scaling: ScalingParameters,
    ) -> np.ndarray:
        self.read_depths.append(int(volume.shape[0]))
        if volume.shape[0] > self.max_read_z:
            raise OutOfMemoryError(f"forced OOM for z={volume.shape[0]}")
        core = volume[core_start:core_stop]
        return core.reshape(-1, core.shape[-2], core.shape[-1]).astype(np.uint16, copy=False)

    def release_memory(self) -> None:
        self.release_calls += 1


class BadU16CoreDtypeDeconvolver(RecordingU16CoreDeconvolver):
    def deconvolve_core_u16(
        self,
        volume: np.ndarray,
        *,
        core_start: int,
        core_stop: int,
        scaling: ScalingParameters,
    ) -> np.ndarray:
        core = volume[core_start:core_stop]
        return core.reshape(-1, core.shape[-2], core.shape[-1]).astype(np.float32, copy=False)


class BadU16CoreShapeDeconvolver(RecordingU16CoreDeconvolver):
    def deconvolve_core_u16(
        self,
        volume: np.ndarray,
        *,
        core_start: int,
        core_stop: int,
        scaling: ScalingParameters,
    ) -> np.ndarray:
        return np.zeros((1, volume.shape[-2], volume.shape[-1] + 1), dtype=np.uint16)


class ProcessSafeRecordingFactory:
    process_safe = True

    def __call__(self, device: int) -> RecordingU16CoreDeconvolver:
        return RecordingU16CoreDeconvolver()


class ProcessSafeSampleFactory:
    process_safe = True

    def __call__(self, device: int) -> IdentityDeconvolver:
        return IdentityDeconvolver()


class RaisingProcessFactory:
    process_safe = True

    def __call__(self, device: int) -> RecordingU16CoreDeconvolver:
        raise RuntimeError(f"bootstrap failed on device {device}")


def _write_identity_scaling(path) -> None:
    params = ScalingParameters(
        offset=np.array([0, 0], dtype=np.float32),
        scale=np.array([1, 1], dtype=np.float32),
        p_low=0,
        p_high=1,
        gamma=1,
        i_max=65535,
    )
    write_scaling_artifacts(
        path,
        params,
        histograms=[(np.array([1]), np.array([0, 1])), (np.array([1]), np.array([0, 1]))],
        manifest={"seed": 1},
        sample_paths=[],
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sample_scale_writes_float32_samples_and_global_artifacts(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    psf = tmp_path / "psf.tif"
    basic = tmp_path / "basic-c0.pkl"
    basic.write_bytes(b"basic profile")
    payload = np.arange(4 * 2 * 3 * 4, dtype=np.uint16).reshape(8, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX", "source": "test"}, photometric="minisblack")
    tifffile.imwrite(psf, np.ones((3, 3, 3), dtype=np.float32), photometric="minisblack")

    sample_scale(
        [src],
        out_dir=tmp_path / "scale",
        planes=4,
        channels=2,
        halo=0,
        deconvolver=IdentityDeconvolver(),
        iterations=7,
        psf_paths=[psf],
        basic_paths=[basic],
        seed=1,
        p_low=0,
        p_high=1,
        gamma=1,
        bins=4,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    sample_paths = sorted((tmp_path / "scale" / "float32-samples").glob("*.tif"))
    assert sample_paths
    with tifffile.TiffFile(sample_paths[0]) as tif:
        assert tif.shaped_metadata[0]["axes"] == "ZYX"
    assert (tmp_path / "scale" / "scaling.json").exists()
    assert (tmp_path / "scale" / "sample-manifest.json").exists()
    manifest = json.loads((tmp_path / "scale" / "sample-manifest.json").read_text())
    assert manifest["channels"] == 2
    assert manifest["iterations"] == 7
    assert manifest["psfs"] == [{"path": str(psf), "sha256": _sha256(psf)}]
    assert manifest["basic_profiles"] == [{"path": str(basic), "sha256": _sha256(basic)}]
    assert "numpy" in manifest["versions"]
    assert manifest["windows"]
    assert manifest["windows"] == sorted(
        manifest["windows"],
        key=lambda item: (item["file"], item["read_start"], item["read_stop"], item["sample_path"]),
    )


def test_sample_scale_uses_summary_metadata(tmp_path, monkeypatch) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    def fail_full_metadata(path):
        raise AssertionError(f"sample-scale must not read full metadata for {path}")

    monkeypatch.setattr("squisher_deconv.source.read_source_metadata", fail_full_metadata)

    sample_scale(
        [src],
        out_dir=tmp_path / "scale",
        planes=1,
        channels=2,
        halo=0,
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        seed=1,
        p_low=0,
        p_high=1,
        gamma=1,
        bins=4,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    manifest = json.loads((tmp_path / "scale" / "sample-manifest.json").read_text())
    assert manifest["metadata"][0]["raw_shape"] == [4, 3, 4]
    assert (tmp_path / "scale" / "scaling.json").exists()


def test_sample_scale_uses_per_input_summary_z_counts(tmp_path, monkeypatch) -> None:
    inputs = []
    for index in range(3):
        src = tmp_path / f"tile{index}.tif"
        payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4) + index
        tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
        inputs.append(src)

    original_open = TiffLogicalSource.open
    calls: list[tuple[object, int, str]] = []

    def tracked_open(path, *, channels: int, metadata_mode: str = "full"):
        calls.append((path, channels, metadata_mode))
        return original_open(path, channels=channels, metadata_mode=metadata_mode)

    monkeypatch.setattr(TiffLogicalSource, "open", staticmethod(tracked_open))

    sample_scale(
        inputs,
        out_dir=tmp_path / "scale",
        planes=2,
        channels=2,
        halo=0,
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        seed=1,
        p_low=0,
        p_high=1,
        gamma=1,
        bins=4,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    assert calls[0] == (inputs[0], 2, "summary")
    assert all(call[2] == "summary" for call in calls)
    assert {call[0] for call in calls} >= set(inputs)
    manifest = json.loads((tmp_path / "scale" / "sample-manifest.json").read_text())
    assert manifest["source_header_mode"] == "per_input_summary"
    assert [row["file"] for row in manifest["metadata"]] == [str(path) for path in inputs]


def test_process_safe_sample_scale_reports_worker_bootstrap_failure(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    with pytest.raises(ProcessingError, match="bootstrap failed on device"):
        sample_scale(
            [src],
            out_dir=tmp_path / "scale",
            planes=1,
            channels=2,
            halo=0,
            deconvolver=None,
            deconvolver_factory=RaisingProcessFactory(),
            psf_paths=None,
            seed=1,
            p_low=0,
            p_high=1,
            gamma=1,
            bins=4,
            devices=[0, 1],
            queue_depth=1,
            stop_on_error=True,
        )

    assert not (tmp_path / "scale").exists()


def test_process_safe_sample_scale_uses_process_gpu_runner(tmp_path, monkeypatch) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    calls: list[dict[str, object]] = []

    def fake_process_runner(windows, **kwargs):
        calls.append({"windows": windows, **kwargs})
        sample_dir = kwargs["sample_dir"]
        template_source = kwargs["template_source"]
        sample_paths = []
        manifest_windows = []
        for index, window in enumerate(windows):
            sample_path = sample_dir / f"{window.path.stem}-window{index:05d}.tif"
            shape = (
                len(window.core_z),
                template_source.channels,
                template_source.height,
                template_source.width,
            )
            data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index
            save_float32_sample(
                sample_path,
                data,
                metadata={"axes": "ZYX", "source": str(window.path), "core_z": list(window.core_z)},
            )
            sample_paths.append(sample_path)
            manifest_windows.append(
                {
                    "file": str(window.path),
                    "device": 0,
                    "read_start": window.read_start,
                    "read_stop": window.read_stop,
                    "sampled_z": list(window.core_z),
                    "sample_path": str(sample_path),
                }
            )
        return sample_paths, manifest_windows, []

    monkeypatch.setattr("squisher_deconv.streaming.run_process_gpu_sample_scale", fake_process_runner)

    sample_scale(
        [src],
        out_dir=tmp_path / "scale",
        planes=1,
        channels=2,
        halo=0,
        deconvolver=None,
        deconvolver_factory=ProcessSafeSampleFactory(),
        psf_paths=None,
        seed=1,
        p_low=0,
        p_high=1,
        gamma=1,
        bins=4,
        devices=[0, 1],
        queue_depth=2,
        stop_on_error=True,
    )

    assert len(calls) == 1
    assert calls[0]["paths"] == [src]
    assert calls[0]["devices"] == [0, 1]
    assert calls[0]["queue_depth"] == 2
    assert calls[0]["template_source"].path == src
    assert (tmp_path / "scale" / "scaling.json").exists()


def test_process_safe_sample_scale_spawn_writes_samples(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(3 * 2 * 3 * 4, dtype=np.uint16).reshape(6, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    sample_scale(
        [src],
        out_dir=tmp_path / "scale",
        planes=2,
        channels=2,
        halo=0,
        deconvolver=None,
        deconvolver_factory=ProcessSafeSampleFactory(),
        psf_paths=None,
        seed=1,
        p_low=0,
        p_high=1,
        gamma=1,
        bins=4,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    assert sorted((tmp_path / "scale" / "float32-samples").glob("*.tif"))
    assert (tmp_path / "scale" / "scaling.json").exists()


def test_process_safe_sample_scale_requires_device(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    with pytest.raises(ValueError, match="At least one device"):
        sample_scale(
            [src],
            out_dir=tmp_path / "scale",
            planes=1,
            channels=2,
            halo=0,
            deconvolver=None,
            deconvolver_factory=ProcessSafeSampleFactory(),
            psf_paths=None,
            seed=1,
            p_low=0,
            p_high=1,
            gamma=1,
            bins=4,
            devices=[],
            queue_depth=1,
            stop_on_error=True,
        )


def test_process_safe_run_requires_device(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)

    with pytest.raises(ValueError, match="At least one device"):
        run_streaming_deconv(
            [src],
            out_dir=tmp_path / "out",
            scaling_path=scaling_dir / "scaling.json",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            deconvolver=None,
            deconvolver_factory=ProcessSafeRecordingFactory(),
            psf_paths=None,
            devices=[],
            queue_depth=1,
            stop_on_error=True,
        )


def test_run_uses_first_input_header_template(tmp_path, monkeypatch) -> None:
    inputs = []
    for index in range(3):
        src = tmp_path / f"tile{index}.tif"
        payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4) + index
        tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
        inputs.append(src)

    original_open = TiffLogicalSource.open
    calls: list[tuple[object, int, str]] = []

    def tracked_open(path, *, channels: int, metadata_mode: str = "full"):
        calls.append((path, channels, metadata_mode))
        return original_open(path, channels=channels, metadata_mode=metadata_mode)

    monkeypatch.setattr(TiffLogicalSource, "open", staticmethod(tracked_open))

    run_streaming_deconv(
        inputs,
        out_dir=tmp_path / "out",
        scaling_path=None,
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="float32",
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    assert calls[0] == (inputs[0], 2, "full")
    assert all(call[2] == "full" for call in calls)
    assert {call[0] for call in calls} >= set(inputs)
    for src in inputs:
        assert (tmp_path / "out" / f"{src.stem}.ome.zarr").exists()
    assert json.loads((tmp_path / "out" / "metadata.json").read_text()) == {
        "czi_shared_metadata_xml": None,
        "positions": [
            {
                "path": str(tmp_path / "out" / f"{src.stem}.ome.zarr"),
                "source": str(src),
            }
            for src in inputs
        ],
    }


def test_run_resume_preserves_complete_tiles_and_writes_full_dataset_metadata(tmp_path) -> None:
    inputs = []
    for index in range(2):
        src = tmp_path / f"tile{index}.tif"
        payload = np.full((4, 3, 4), index + 1, dtype=np.uint16)
        tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
        inputs.append(src)
    out_dir = tmp_path / "out"
    run_streaming_deconv(
        inputs[:1],
        out_dir=out_dir,
        scaling_path=None,
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="float32",
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )
    first_output = out_dir / "tile0.ome.zarr"
    zarr.open_group(str(first_output), mode="a").attrs["resume_sentinel"] = True
    (out_dir / "metadata.json").unlink()

    run_streaming_deconv(
        inputs,
        out_dir=out_dir,
        scaling_path=None,
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="float32",
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
        resume=True,
    )

    assert zarr.open_group(str(first_output), mode="r").attrs["resume_sentinel"] is True
    assert (out_dir / "tile1.ome.zarr").is_dir()
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert [item["source"] for item in metadata["positions"]] == [str(path) for path in inputs]


def test_run_uses_per_input_z_count(tmp_path) -> None:
    src_a = tmp_path / "tile-a.tif"
    src_b = tmp_path / "tile-b.tif"
    payload_a = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    payload_b = np.arange(3 * 2 * 3 * 4, dtype=np.uint16).reshape(6, 3, 4) + 100
    tifffile.imwrite(src_a, payload_a, metadata={"axes": "ZYX"}, photometric="minisblack")
    tifffile.imwrite(src_b, payload_b, metadata={"axes": "ZYX"}, photometric="minisblack")

    run_streaming_deconv(
        [src_a, src_b],
        out_dir=tmp_path / "out",
        scaling_path=None,
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="float32",
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    actual_a, _root_a, array_a = _read_deconv_ome_zarr(tmp_path / "out" / "tile-a.ome.zarr")
    actual_b, _root_b, array_b = _read_deconv_ome_zarr(tmp_path / "out" / "tile-b.ome.zarr")
    assert tuple(array_a.shape) == (2, 2, 3, 4)
    assert tuple(array_b.shape) == (2, 3, 3, 4)
    np.testing.assert_allclose(actual_a, payload_a)
    np.testing.assert_allclose(actual_b, payload_b)


def test_streamed_u16_deconv_matches_eager_identity(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    basic = tmp_path / "basic-c0.pkl"
    basic.write_bytes(b"basic profile")
    payload = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX", "source": "test"}, photometric="minisblack")

    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)

    run_streaming_deconv(
        [src],
        out_dir=tmp_path / "out",
        scaling_path=scaling_dir / "scaling.json",
        channels=2,
        halo=0,
        slab_depth=2,
        output_mode="u16",
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        basic_paths=[basic],
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    out = tmp_path / "out" / "tile.ome.zarr"
    actual, root, array = _read_deconv_ome_zarr(out)
    assert tuple(array.shape) == (2, 3, 4, 5)
    assert tuple(array.chunks) == (1, 3, 4, 5)
    assert tuple(array.metadata.dimension_names) == ("c", "z", "y", "x")
    assert array.metadata.zarr_format == 3
    assert root.attrs["squisher_complete"] is True
    assert root.attrs["squisher_deconv"]["provenance"]["basic_profiles"] == [
        {"path": str(basic), "sha256": _sha256(basic)}
    ]
    assert root.attrs["squisher_deconv"]["source_metadata_summary"]["raw_shape"] == [6, 4, 5]
    assert root.attrs["squisher_deconv"]["source_metadata_summary"]["raw_dtype"] == "uint16"
    assert np.allclose(actual, payload, atol=5)

    sidecar = json.loads((tmp_path / "out" / "tile.deconv.json").read_text())
    assert sidecar["provenance"]["basic_profiles"] == [{"path": str(basic), "sha256": _sha256(basic)}]
    assert sidecar["provenance"]["run_settings"] == {
        "channels": 2,
        "halo": 0,
        "iterations": None,
        "output_mode": "u16",
        "devices": [0],
        "queue_depth": 1,
        "slab_depth": 2,
    }
    assert sidecar["provenance"]["scaling"] == {
        "path": str(scaling_dir / "scaling.json"),
        "sha256": _sha256(scaling_dir / "scaling.json"),
    }


def test_run_preserves_parent_dirs_for_duplicate_basenames(tmp_path) -> None:
    roi_a = tmp_path / "roi-a"
    roi_b = tmp_path / "roi-b"
    roi_a.mkdir()
    roi_b.mkdir()
    src_a = roi_a / "tile.tif"
    src_b = roi_b / "tile.tif"
    payload_a = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    payload_b = (payload_a + 200).astype(np.uint16)
    tifffile.imwrite(src_a, payload_a, metadata={"axes": "ZYX"}, photometric="minisblack")
    tifffile.imwrite(src_b, payload_b, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)

    run_streaming_deconv(
        [src_a, src_b],
        out_dir=tmp_path / "out",
        scaling_path=scaling_dir / "scaling.json",
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="u16",
        deconvolver=IdentityDeconvolver(),
        deconvolver_factory=None,
        psf_paths=None,
        devices=[0],
        queue_depth=2,
        stop_on_error=True,
    )

    assert np.allclose(
        _read_deconv_ome_zarr(tmp_path / "out" / "roi-a" / "tile.ome.zarr")[0], payload_a, atol=5
    )
    assert np.allclose(
        _read_deconv_ome_zarr(tmp_path / "out" / "roi-b" / "tile.ome.zarr")[0], payload_b, atol=5
    )
    assert not (tmp_path / "out" / "tile.ome.zarr").exists()


def test_run_preserves_enough_parent_dirs_for_repeated_parent_names(tmp_path) -> None:
    roi_a = tmp_path / "top-a" / "roi"
    roi_b = tmp_path / "top-b" / "roi"
    roi_a.mkdir(parents=True)
    roi_b.mkdir(parents=True)
    src_a = roi_a / "tile.tif"
    src_b = roi_b / "tile.tif"
    payload_a = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    payload_b = (payload_a + 300).astype(np.uint16)
    tifffile.imwrite(src_a, payload_a, metadata={"axes": "ZYX"}, photometric="minisblack")
    tifffile.imwrite(src_b, payload_b, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)

    run_streaming_deconv(
        [src_a, src_b],
        out_dir=tmp_path / "out",
        scaling_path=scaling_dir / "scaling.json",
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="u16",
        deconvolver=IdentityDeconvolver(),
        deconvolver_factory=None,
        psf_paths=None,
        devices=[0],
        queue_depth=2,
        stop_on_error=True,
    )

    assert np.allclose(
        _read_deconv_ome_zarr(tmp_path / "out" / "top-a" / "roi" / "tile.ome.zarr")[0],
        payload_a,
        atol=5,
    )
    assert np.allclose(
        _read_deconv_ome_zarr(tmp_path / "out" / "top-b" / "roi" / "tile.ome.zarr")[0],
        payload_b,
        atol=5,
    )


def test_streamed_u16_uses_backend_core_quantization_path(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    deconvolver = RecordingU16CoreDeconvolver()

    run_streaming_deconv(
        [src],
        out_dir=tmp_path / "out",
        scaling_path=scaling_dir / "scaling.json",
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="u16",
        deconvolver=deconvolver,
        psf_paths=None,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    assert [call[1:3] for call in deconvolver.calls] == [(0, 1), (0, 1), (0, 1)]
    assert {call[3] for call in deconvolver.calls} == {(0.0, 0.0)}
    assert {call[4] for call in deconvolver.calls} == {(1.0, 1.0)}
    assert np.allclose(_read_deconv_ome_zarr(tmp_path / "out" / "tile.ome.zarr")[0], payload, atol=5)


def test_factory_u16_deconvolver_must_expose_core_quantization_path(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)

    with pytest.raises(TypeError, match="deconvolve_core_u16"):
        run_streaming_deconv(
            [src],
            out_dir=tmp_path / "out",
            scaling_path=scaling_dir / "scaling.json",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            deconvolver=None,
            deconvolver_factory=lambda _device: IdentityDeconvolver(),
            psf_paths=None,
            devices=[0],
            queue_depth=1,
            stop_on_error=True,
        )

    assert not (tmp_path / "out" / "tile.ome.zarr").exists()


def test_process_safe_factory_uses_process_gpu_runner(tmp_path, monkeypatch) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    calls: list[dict[str, object]] = []

    def fake_process_runner(paths, **kwargs):
        calls.append({"paths": paths, **kwargs})
        return []

    monkeypatch.setattr("squisher_deconv.streaming.run_process_gpu_streaming_deconv", fake_process_runner)

    run_streaming_deconv(
        [src],
        out_dir=tmp_path / "out",
        scaling_path=scaling_dir / "scaling.json",
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="u16",
        deconvolver=None,
        deconvolver_factory=ProcessSafeRecordingFactory(),
        psf_paths=None,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    assert len(calls) == 1
    assert calls[0]["paths"] == [src]
    assert calls[0]["stop_on_error"] is True
    assert calls[0]["template_source"].path == src


def test_process_worker_file_pipeline_writes_streamed_u16_output(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    duration = _process_file(
        worker_id=0,
        device=0,
        file_index=0,
        path=src,
        template_source=TiffLogicalSource.open(src, channels=2, metadata_mode="summary"),
        scaling=ScalingParameters(
            offset=np.array([0, 0], dtype=np.float32),
            scale=np.array([1, 1], dtype=np.float32),
            p_low=0,
            p_high=1,
            gamma=1,
            i_max=65535,
        ),
        deconvolver=RecordingU16CoreDeconvolver(),
        config=ProcessRunConfig(
            out_dir=tmp_path / "out",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            psf_paths=None,
            basic_paths=(),
            scaling_path=scaling_dir / "scaling.json",
            devices=(0,),
            queue_depth=2,
            overwrite=False,
            output_relative_root=None,
        ),
    )

    assert duration >= 0
    out = tmp_path / "out" / "tile.ome.zarr"
    actual, _root, _array = _read_deconv_ome_zarr(out)
    assert np.allclose(actual, payload, atol=5)
    sidecar = json.loads((tmp_path / "out" / "tile.deconv.json").read_text())
    assert sidecar["worker"] == {"worker_id": 0, "device": 0}


def test_process_worker_halves_slab_depth_and_retries_file_after_cupy_oom(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(8 * 3 * 4, dtype=np.uint16).reshape(8, 3, 4)
    tifffile.imwrite(src, payload, ome=True, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    deconvolver = SlabLimitedU16Deconvolver(max_read_z=2)

    duration = _process_file_with_slab_recovery(
        worker_id=0,
        device=0,
        file_index=0,
        path=src,
        template_source=TiffLogicalSource.open(src, channels=1, metadata_mode="summary"),
        scaling=ScalingParameters(
            offset=np.array([0], dtype=np.float32),
            scale=np.array([1], dtype=np.float32),
            p_low=0,
            p_high=1,
            gamma=1,
            i_max=65535,
        ),
        deconvolver=deconvolver,
        config=ProcessRunConfig(
            out_dir=tmp_path / "out",
            channels=1,
            halo=0,
            slab_depth=4,
            output_mode="u16",
            psf_paths=(),
            basic_paths=(),
            scaling_path=scaling_dir / "scaling.json",
            devices=(0,),
            queue_depth=1,
            overwrite=False,
            output_relative_root=None,
        ),
    )

    assert duration >= 0
    assert deconvolver.read_depths == [4, 2, 2, 2, 2]
    assert deconvolver.release_calls == 1
    actual, _root, _array = _read_deconv_ome_zarr(tmp_path / "out" / "tile.ome.zarr")
    assert np.array_equal(actual, payload)
    sidecar = json.loads((tmp_path / "out" / "tile.deconv.json").read_text())
    assert sidecar["chunking"]["slab_depth"] == 2


def test_process_worker_preserves_parent_dirs_for_duplicate_basenames(tmp_path) -> None:
    roi_a = tmp_path / "roi-a"
    roi_b = tmp_path / "roi-b"
    roi_a.mkdir()
    roi_b.mkdir()
    src_a = roi_a / "tile.tif"
    src_b = roi_b / "tile.tif"
    payload_a = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    payload_b = (payload_a + 100).astype(np.uint16)
    tifffile.imwrite(src_a, payload_a, metadata={"axes": "ZYX"}, photometric="minisblack")
    tifffile.imwrite(src_b, payload_b, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    failures = run_process_gpu_streaming_deconv(
        [src_a, src_b],
        template_source=TiffLogicalSource.open(src_a, channels=2, metadata_mode="summary"),
        scaling=ScalingParameters(
            offset=np.array([0, 0], dtype=np.float32),
            scale=np.array([1, 1], dtype=np.float32),
            p_low=0,
            p_high=1,
            gamma=1,
            i_max=65535,
        ),
        deconvolver_factory=ProcessSafeRecordingFactory(),
        config=ProcessRunConfig(
            out_dir=tmp_path / "out",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            psf_paths=None,
            basic_paths=(),
            scaling_path=scaling_dir / "scaling.json",
            devices=(0,),
            queue_depth=2,
            overwrite=False,
            output_relative_root=tmp_path,
        ),
        stop_on_error=True,
    )

    assert failures == []
    assert np.allclose(
        _read_deconv_ome_zarr(tmp_path / "out" / "roi-a" / "tile.ome.zarr")[0], payload_a, atol=5
    )
    assert np.allclose(
        _read_deconv_ome_zarr(tmp_path / "out" / "roi-b" / "tile.ome.zarr")[0], payload_b, atol=5
    )


def test_process_gpu_runner_spawn_writes_output(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    scaling = ScalingParameters(
        offset=np.array([0, 0], dtype=np.float32),
        scale=np.array([1, 1], dtype=np.float32),
        p_low=0,
        p_high=1,
        gamma=1,
        i_max=65535,
    )

    failures = run_process_gpu_streaming_deconv(
        [src],
        template_source=TiffLogicalSource.open(src, channels=2, metadata_mode="summary"),
        scaling=scaling,
        deconvolver_factory=ProcessSafeRecordingFactory(),
        config=ProcessRunConfig(
            out_dir=tmp_path / "out",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            psf_paths=None,
            basic_paths=(),
            scaling_path=scaling_dir / "scaling.json",
            devices=(0,),
            queue_depth=2,
            overwrite=False,
            output_relative_root=None,
        ),
        stop_on_error=True,
    )

    assert failures == []
    assert np.allclose(_read_deconv_ome_zarr(tmp_path / "out" / "tile.ome.zarr")[0], payload, atol=5)


def test_process_gpu_runner_reports_worker_bootstrap_failure_without_hanging(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    failures = run_process_gpu_streaming_deconv(
        [src],
        template_source=TiffLogicalSource.open(src, channels=2, metadata_mode="summary"),
        scaling=ScalingParameters(
            offset=np.array([0, 0], dtype=np.float32),
            scale=np.array([1, 1], dtype=np.float32),
            p_low=0,
            p_high=1,
            gamma=1,
            i_max=65535,
        ),
        deconvolver_factory=RaisingProcessFactory(),
        config=ProcessRunConfig(
            out_dir=tmp_path / "out",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            psf_paths=None,
            basic_paths=(),
            scaling_path=scaling_dir / "scaling.json",
            devices=(0,),
            queue_depth=1,
            overwrite=False,
            output_relative_root=None,
        ),
        stop_on_error=False,
    )

    assert len(failures) == 1
    assert "bootstrap failed on device 0" in failures[0]
    assert not (tmp_path / "out" / "tile.ome.zarr").exists()


def test_streamed_u16_rejects_non_uint16_core_backend_output(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)

    with pytest.raises(TypeError, match="must return uint16"):
        run_streaming_deconv(
            [src],
            out_dir=tmp_path / "out",
            scaling_path=scaling_dir / "scaling.json",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            deconvolver=BadU16CoreDtypeDeconvolver(),
            psf_paths=None,
            devices=[0],
            queue_depth=1,
            stop_on_error=True,
        )

    assert not (tmp_path / "out" / "tile.ome.zarr").exists()


def test_streamed_u16_rejects_wrong_shape_core_backend_output(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)

    with pytest.raises(ValueError, match="returned shape"):
        run_streaming_deconv(
            [src],
            out_dir=tmp_path / "out",
            scaling_path=scaling_dir / "scaling.json",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            deconvolver=BadU16CoreShapeDeconvolver(),
            psf_paths=None,
            devices=[0],
            queue_depth=1,
            stop_on_error=True,
        )

    assert not (tmp_path / "out" / "tile.ome.zarr").exists()


def test_streamed_float32_deconv_does_not_require_scaling(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    run_streaming_deconv(
        [src],
        out_dir=tmp_path / "out",
        scaling_path=None,
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="float32",
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )

    actual, _root, array = _read_deconv_ome_zarr(tmp_path / "out" / "tile.ome.zarr")
    assert array.dtype == np.float32
    assert actual.dtype == np.float32
    assert np.allclose(actual, payload.astype(np.float32))
    sidecar = json.loads((tmp_path / "out" / "tile.deconv.json").read_text())
    assert "compression_tiff_tag" not in sidecar
    assert "compression_tiff_tag" not in sidecar["provenance"]


def test_run_prefetches_reads_up_to_queue_depth(tmp_path, monkeypatch) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(6 * 2 * 3 * 4, dtype=np.uint16).reshape(12, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    lock = threading.Lock()
    active = 0
    max_active = 0
    original_read_window = TiffLogicalSource.read_window

    def tracked_read_window(self, start: int, stop: int) -> np.ndarray:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        try:
            return original_read_window(self, start, stop)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(TiffLogicalSource, "read_window", tracked_read_window)

    run_streaming_deconv(
        [src],
        out_dir=tmp_path / "out",
        scaling_path=None,
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="float32",
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        devices=[0],
        queue_depth=2,
        stop_on_error=True,
    )

    assert max_active == 2
    actual = _read_deconv_ome_zarr(tmp_path / "out" / "tile.ome.zarr")[0]
    assert np.allclose(actual, payload.astype(np.float32))


def test_sample_scale_prefetches_reads_up_to_queue_depth(tmp_path, monkeypatch) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(6 * 2 * 3 * 4, dtype=np.uint16).reshape(12, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    lock = threading.Lock()
    active = 0
    max_active = 0
    original_read_window = TiffLogicalSource.read_window

    def tracked_read_window(self, start: int, stop: int) -> np.ndarray:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        try:
            return original_read_window(self, start, stop)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(TiffLogicalSource, "read_window", tracked_read_window)

    sample_scale(
        [src],
        out_dir=tmp_path / "scale",
        planes=4,
        channels=2,
        halo=0,
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        seed=1,
        p_low=0,
        p_high=1,
        gamma=1,
        bins=4,
        devices=[0],
        queue_depth=2,
        stop_on_error=True,
    )

    assert max_active == 2


def test_queue_depth_must_be_positive(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    with pytest.raises(ValueError, match="queue_depth must be at least 1"):
        run_streaming_deconv(
            [src],
            out_dir=tmp_path / "out",
            scaling_path=None,
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="float32",
            deconvolver=IdentityDeconvolver(),
            psf_paths=None,
            devices=[0],
            queue_depth=0,
            stop_on_error=True,
        )


def test_run_refuses_existing_output_without_overwrite(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "tile.ome.zarr"
    existing.mkdir(parents=True)
    (existing / "sentinel").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing output"):
        run_streaming_deconv(
            [src],
            out_dir=out_dir,
            scaling_path=scaling_dir / "scaling.json",
            channels=2,
            halo=0,
            slab_depth=2,
            output_mode="u16",
            deconvolver=IdentityDeconvolver(),
            psf_paths=None,
            devices=[0],
            queue_depth=1,
            stop_on_error=True,
        )
    assert (existing / "sentinel").read_bytes() == b"existing"


def test_run_rejects_scaling_channel_mismatch_before_output(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    params = ScalingParameters(
        offset=np.array([0], dtype=np.float32),
        scale=np.array([1], dtype=np.float32),
        p_low=0,
        p_high=1,
        gamma=1,
        i_max=65535,
    )
    write_scaling_artifacts(
        scaling_dir,
        params,
        histograms=[(np.array([1]), np.array([0, 1]))],
        manifest={"seed": 1},
        sample_paths=[],
    )

    with pytest.raises(ValueError, match="has 1 channel\\(s\\), but run was configured with channels=2"):
        run_streaming_deconv(
            [src],
            out_dir=tmp_path / "out",
            scaling_path=scaling_dir / "scaling.json",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            deconvolver=IdentityDeconvolver(),
            psf_paths=None,
            devices=[0],
            queue_depth=1,
            stop_on_error=True,
        )

    assert not (tmp_path / "out").exists()


def test_run_overwrites_existing_output_when_requested(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "tile.ome.zarr").mkdir(parents=True)

    run_streaming_deconv(
        [src],
        out_dir=out_dir,
        scaling_path=scaling_dir / "scaling.json",
        channels=2,
        halo=0,
        slab_depth=2,
        output_mode="u16",
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
        overwrite=True,
    )

    assert (out_dir / "tile.ome.zarr" / "zarr.json").exists()


def test_sample_scale_refuses_existing_artifacts_without_overwrite(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    out_dir = tmp_path / "scale"
    out_dir.mkdir()
    (out_dir / "scaling.json").write_text("{}")

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing sample-scale output"):
        sample_scale(
            [src],
            out_dir=out_dir,
            planes=1,
            channels=2,
            halo=0,
            deconvolver=IdentityDeconvolver(),
            psf_paths=None,
            seed=1,
            p_low=0,
            p_high=1,
            gamma=1,
            bins=4,
            devices=[0],
            queue_depth=1,
            stop_on_error=True,
        )


def test_sample_scale_overwrite_removes_stale_artifacts(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    out_dir = tmp_path / "scale"
    sample_dir = out_dir / "float32-samples"
    sample_dir.mkdir(parents=True)
    stale = sample_dir / "stale.tif"
    stale.write_bytes(b"stale")
    (out_dir / "scaling.json").write_text("{}")

    sample_scale(
        [src],
        out_dir=out_dir,
        planes=1,
        channels=2,
        halo=0,
        deconvolver=IdentityDeconvolver(),
        psf_paths=None,
        seed=1,
        p_low=0,
        p_high=1,
        gamma=1,
        bins=4,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
        overwrite=True,
    )

    assert not stale.exists()
    assert (out_dir / "scaling.json").exists()


def test_run_uses_factory_deconvolver_for_scheduled_devices(tmp_path) -> None:
    inputs = []
    for file_index in range(2):
        src = tmp_path / f"tile{file_index}.tif"
        payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
        tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
        inputs.append(src)
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)
    calls: list[int] = []

    run_streaming_deconv(
        inputs,
        out_dir=tmp_path / "out",
        scaling_path=scaling_dir / "scaling.json",
        channels=2,
        halo=0,
        slab_depth=2,
        output_mode="u16",
        deconvolver=IdentityDeconvolver(),
        deconvolver_factory=lambda device: RecordingDeconvolver(device, calls),
        psf_paths=None,
        devices=[0, 1],
        queue_depth=1,
        stop_on_error=True,
    )

    assert sorted(calls) == [0, 1]


def test_keep_going_reports_failed_run_jobs(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
    scaling_dir = tmp_path / "scale"
    _write_identity_scaling(scaling_dir)

    with pytest.raises(ProcessingError, match="1 run job\\(s\\) failed"):
        run_streaming_deconv(
            [src],
            out_dir=tmp_path / "out",
            scaling_path=scaling_dir / "scaling.json",
            channels=2,
            halo=0,
            slab_depth=1,
            output_mode="u16",
            deconvolver=FailingDeconvolver(),
            psf_paths=None,
            devices=[0],
            queue_depth=1,
            stop_on_error=False,
        )
    assert not (tmp_path / "out" / "tile.ome.zarr").exists()
    assert not (tmp_path / "out" / ".tile.ome.zarr.partial").exists()


def test_keep_going_reports_failed_sample_jobs(tmp_path) -> None:
    src = tmp_path / "tile.tif"
    payload = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    with pytest.raises(ProcessingError, match="1 sample-scale job\\(s\\) failed"):
        sample_scale(
            [src],
            out_dir=tmp_path / "scale",
            planes=1,
            channels=2,
            halo=0,
            deconvolver=FailingDeconvolver(),
            psf_paths=None,
            seed=1,
            p_low=0,
            p_high=1,
            gamma=1,
            bins=4,
            devices=[0],
            queue_depth=1,
            stop_on_error=False,
        )
