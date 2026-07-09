from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from squisher_deconv.cli import app


class _FakeCp:
    float32 = np.float32

    @staticmethod
    def asarray(array, dtype=None):
        return np.asarray(array, dtype=dtype)


def _has_cupy_gpu() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _cupy_gpu_count() -> int:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _write_basic(path, *, darkfield, flatfield) -> None:
    basic = SimpleNamespace(
        darkfield=np.asarray(darkfield, dtype=np.float32),
        flatfield=np.asarray(flatfield, dtype=np.float32),
    )
    path.write_bytes(pickle.dumps({"basic": basic}))


def test_basic_profile_loader_rejects_non_2d_profiles(tmp_path) -> None:
    from squisher_deconv.gpu import _load_basic_profiles

    path = tmp_path / "basic.pkl"
    _write_basic(
        path,
        darkfield=np.zeros((2, 2, 2), dtype=np.float32),
        flatfield=np.ones((2, 2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="darkfield.*must be 2-D"):
        _load_basic_profiles([path], cp=_FakeCp)


def test_cupy_jit_cuda_path_uses_conda_target(tmp_path, monkeypatch) -> None:
    import squisher_deconv.gpu as gpu

    target = tmp_path / "targets" / "x86_64-linux"
    (target / "include").mkdir(parents=True)
    (target / "lib").mkdir()
    (target / "include" / "cuda.h").write_text("")
    (target / "lib" / "libnvrtc.so").write_text("")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(gpu.sys, "prefix", str(tmp_path))

    gpu.ensure_cuda_path_for_cupy_jit()

    assert gpu.os.environ["CUDA_PATH"] == str(target)


def test_cupy_jit_cuda_path_preserves_explicit_env(monkeypatch) -> None:
    import squisher_deconv.gpu as gpu

    monkeypatch.setenv("CUDA_PATH", "/explicit/cuda")

    gpu.ensure_cuda_path_for_cupy_jit()

    assert gpu.os.environ["CUDA_PATH"] == "/explicit/cuda"


@pytest.mark.parametrize(
    ("darkfield", "flatfield", "message"),
    [
        (np.array([[np.nan]], dtype=np.float32), np.ones((1, 1), dtype=np.float32), "darkfield.*finite"),
        (np.zeros((1, 1), dtype=np.float32), np.array([[np.inf]], dtype=np.float32), "flatfield.*finite"),
        (
            np.zeros((1, 1), dtype=np.float32),
            np.array([[0]], dtype=np.float32),
            "flatfield.*strictly positive",
        ),
    ],
)
def test_basic_profile_loader_rejects_invalid_values(tmp_path, darkfield, flatfield, message) -> None:
    from squisher_deconv.gpu import _load_basic_profiles

    path = tmp_path / "basic.pkl"
    _write_basic(path, darkfield=darkfield, flatfield=flatfield)

    with pytest.raises(ValueError, match=message):
        _load_basic_profiles([path], cp=_FakeCp)


@pytest.mark.skipif(not _has_cupy_gpu(), reason="CuPy GPU is not available")
def test_cupy_basic_richardson_lucy_deconvolver_smoke(tmp_path) -> None:
    from squisher_deconv.gpu import CupyBasicRichardsonLucyDeconvolver

    basic_paths = []
    for channel in range(2):
        basic = SimpleNamespace(
            darkfield=np.zeros((5, 5), dtype=np.float32),
            flatfield=np.ones((5, 5), dtype=np.float32),
        )
        path = tmp_path / f"basic-c{channel}.pkl"
        path.write_bytes(pickle.dumps({"basic": basic}))
        basic_paths.append(path)

    psf = np.zeros((5, 5, 5), dtype=np.float32)
    psf[2, 2, 2] = 1.0
    psf_path = tmp_path / "psf.tif"
    tifffile.imwrite(psf_path, psf)

    deconvolver = CupyBasicRichardsonLucyDeconvolver(
        basic_paths=basic_paths,
        psf_paths=(psf_path, psf_path),
        device=0,
    )
    volume = np.arange(3 * 2 * 5 * 5, dtype=np.uint16).reshape(3, 2, 5, 5)

    out = deconvolver.deconvolve(volume)

    assert out.shape == volume.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert deconvolver.halo == 8


@pytest.mark.skipif(not _has_cupy_gpu(), reason="CuPy GPU is not available")
def test_cupy_engine_applies_basic_before_deconvolution(tmp_path, monkeypatch) -> None:
    import cupy as cp
    import squisher_deconv.gpu as gpu
    from squisher_deconv.gpu import CupyBasicRichardsonLucyDeconvolver

    basic = SimpleNamespace(
        darkfield=np.full((3, 3), 10, dtype=np.float32),
        flatfield=np.full((3, 3), 2, dtype=np.float32),
    )
    basic_path = tmp_path / "basic.pkl"
    basic_path.write_bytes(pickle.dumps({"basic": basic}))

    psf = np.zeros((3, 3, 3), dtype=np.float32)
    psf[1, 1, 1] = 1.0
    psf_path = tmp_path / "psf.tif"
    tifffile.imwrite(psf_path, psf, photometric="minisblack")

    captured: list[np.ndarray] = []

    def capture_deconvolution_input(img, projectors, *, iterations, cp):
        assert iterations == 1
        captured.append(cp.asnumpy(img))
        return img

    monkeypatch.setattr(gpu, "_deconvolve_lucyrichardson_guo", capture_deconvolution_input)
    deconvolver = CupyBasicRichardsonLucyDeconvolver(
        basic_paths=[basic_path],
        psf_paths=(psf_path,),
        device=0,
    )
    volume = np.full((2, 1, 3, 3), 14, dtype=np.uint16)

    out = deconvolver.deconvolve(volume)

    assert np.allclose(captured[0], 2.0)
    assert np.allclose(out, 2.0)
    cp.get_default_memory_pool().free_all_blocks()


@pytest.mark.skipif(not _has_cupy_gpu(), reason="CuPy GPU is not available")
def test_guo_backward_projector_keeps_dc_response() -> None:
    import cupy as cp
    from squisher_deconv.gpu import _calculate_projectors_3d

    psf = cp.zeros((11, 9, 9), dtype=cp.float32)
    psf[5, 4, 4] = 1.0

    _forward, backward = _calculate_projectors_3d(psf, sigma_g=1.7, a=0.02, b=0.02, n=10, cp=cp)

    assert 0.9 < float(backward.sum()) < 1.0
    cp.get_default_memory_pool().free_all_blocks()


@pytest.mark.skipif(not _has_cupy_gpu(), reason="CuPy GPU is not available")
def test_gpu_cli_sample_scale_and_run_smoke(tmp_path) -> None:
    runner = CliRunner()
    src = tmp_path / "tile.tif"
    psf_path = tmp_path / "psf.tif"
    payload = np.arange(2 * 2 * 5 * 5, dtype=np.uint16).reshape(4, 5, 5)
    tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")

    psf = np.zeros((5, 5, 5), dtype=np.float32)
    psf[2, 2, 2] = 1.0
    tifffile.imwrite(psf_path, psf, photometric="minisblack")

    basic_args: list[str] = []
    for channel in range(2):
        basic = SimpleNamespace(
            darkfield=np.zeros((5, 5), dtype=np.float32),
            flatfield=np.ones((5, 5), dtype=np.float32),
        )
        path = tmp_path / f"basic-c{channel}.pkl"
        path.write_bytes(pickle.dumps({"basic": basic}))
        basic_args.extend(["--basic", str(path)])

    sample = runner.invoke(
        app,
        [
            "sample-scale",
            str(src),
            "--out-dir",
            str(tmp_path / "scale"),
            "--planes",
            "1",
            "--channels",
            "2",
            "--psf",
            str(psf_path),
            "--psf",
            str(psf_path),
            *basic_args,
            "--devices",
            "0",
            "--halo",
            "0",
            "--p-low",
            "0",
            "--p-high",
            "1",
            "--bins",
            "4",
        ],
    )
    assert sample.exit_code == 0, sample.output

    run = runner.invoke(
        app,
        [
            "run",
            str(src),
            "--out-dir",
            str(tmp_path / "out"),
            "--scaling",
            str(tmp_path / "scale" / "scaling.json"),
            "--channels",
            "2",
            "--psf",
            str(psf_path),
            "--psf",
            str(psf_path),
            *basic_args,
            "--devices",
            "0",
            "--halo",
            "0",
            "--slab-depth",
            "1",
        ],
    )
    assert run.exit_code == 0, run.output
    with tifffile.TiffFile(tmp_path / "out" / "tile.tif") as tif:
        assert tif.is_bigtiff
        assert len(tif.pages) == 4
        assert all(page.compression == 22610 for page in tif.pages)


@pytest.mark.skipif(_cupy_gpu_count() < 2, reason="At least two CuPy GPUs are required")
def test_gpu_sample_scale_projectors_are_device_local(tmp_path) -> None:
    runner = CliRunner()
    inputs = []
    for index in range(2):
        src = tmp_path / f"tile{index}.tif"
        payload = np.arange(2 * 5 * 5, dtype=np.uint16).reshape(2, 5, 5) + index
        tifffile.imwrite(src, payload, metadata={"axes": "ZYX"}, photometric="minisblack")
        inputs.append(src)

    psf = np.zeros((3, 3, 3), dtype=np.float32)
    psf[1, 1, 1] = 1.0
    psf_path = tmp_path / "psf.tif"
    tifffile.imwrite(psf_path, psf, photometric="minisblack")

    basic_path = tmp_path / "basic.pkl"
    _write_basic(
        basic_path,
        darkfield=np.zeros((5, 5), dtype=np.float32),
        flatfield=np.ones((5, 5), dtype=np.float32),
    )

    result = runner.invoke(
        app,
        [
            "sample-scale",
            *(str(path) for path in inputs),
            "--out-dir",
            str(tmp_path / "scale"),
            "--planes",
            "4",
            "--channels",
            "1",
            "--psf",
            str(psf_path),
            "--basic",
            str(basic_path),
            "--devices",
            "0,1",
            "--halo",
            "0",
            "--p-low",
            "0",
            "--p-high",
            "1",
            "--bins",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "scale" / "float32-samples").glob("*.tif"))) == 2
