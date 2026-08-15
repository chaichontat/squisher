from __future__ import annotations

import fcntl
import functools
import hashlib
import os
from dataclasses import dataclass
from os import fsync
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import tifffile

from squisher_deconv.deconvolution import Deconvolver
from squisher_deconv.scaling import ScalingParameters

EPS = np.float32(1e-9)
I_MAX = np.float32(2**16 - 1)


class CupyBasicRichardsonLucyDeconvolver(Deconvolver):
    """Optional CuPy BaSiC correction followed by Guo LR deconvolution."""

    def __init__(
        self,
        *,
        basic_paths: Sequence[Path],
        psf_paths: Sequence[Path],
        device: int,
        iterations: int = 1,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1.")
        ensure_cuda_path_for_cupy_jit()
        import cupy as cp

        cp.cuda.Device(device).use()
        self._memory_pool, self._pinned_memory_pool = _initialize_cupy_allocator(cp)
        self._cp = cp
        self._device = int(device)
        self._iterations = int(iterations)
        if basic_paths:
            self._darkfield, flatfield = _load_basic_profiles(basic_paths, cp=cp)
            self._inv_flatfield = 1.0 / flatfield
            self._basic_kernel = cp.ElementwiseKernel(
                "float32 x, float32 df, float32 inv_ff",
                "float32 y",
                "float t = (x - df) * inv_ff; y = t > 0.0f ? t : 0.0f;",
                name="squisher_deconv_basic_correct_clip",
            )
        else:
            self._darkfield = None
            self._inv_flatfield = None
            self._basic_kernel = None
        if not psf_paths:
            raise ValueError("At least one PSF path is required.")
        self._projectors = tuple(
            _projectors(
                psf_path,
                cp=cp,
                device=self._device,
            )
            for psf_path in psf_paths
        )

    @property
    def halo(self) -> int:
        return max(
            int((forward.shape[0] - 1) + (backward.shape[0] - 1)) for forward, backward in self._projectors
        )

    def deconvolve(self, volume: np.ndarray) -> np.ndarray:
        result = self._deconvolve_gpu(volume)
        return self._cp.asnumpy(result).astype(np.float32, copy=False)

    def deconvolve_core_u16(
        self,
        volume: np.ndarray,
        *,
        core_start: int,
        core_stop: int,
        scaling: ScalingParameters,
    ) -> np.ndarray:
        if volume.ndim != 4:
            raise ValueError(f"Expected (Z, C, Y, X) volume, got {volume.shape}")
        if not 0 <= core_start < core_stop <= volume.shape[0]:
            raise ValueError(f"Invalid core slice [{core_start}, {core_stop}) for input shape {volume.shape}")
        result = self._deconvolve_gpu(volume)
        return _quantize_global_gpu(result[core_start:core_stop], scaling, cp=self._cp)

    def release_memory(self) -> None:
        """Release unused CuPy blocks before retrying with a smaller slab."""
        self._cp.cuda.Device(self._device).use()
        self._memory_pool.free_all_blocks()
        self._pinned_memory_pool.free_all_blocks()

    def _deconvolve_gpu(self, volume: np.ndarray) -> Any:
        if volume.ndim != 4:
            raise ValueError(f"Expected (Z, C, Y, X) volume, got {volume.shape}")
        cp = self._cp
        cp.cuda.Device(self._device).use()
        x = cp.asarray(volume, dtype=cp.float32)
        if x.shape[1] != len(self._projectors):
            raise ValueError(
                f"Input has {x.shape[1]} channel(s), but {len(self._projectors)} PSF(s) were loaded."
            )
        if self._basic_kernel is not None:
            if self._darkfield is None or self._inv_flatfield is None:
                raise RuntimeError("BaSiC correction kernel was initialized without correction profiles.")
            if x.shape[1] != self._darkfield.shape[1]:
                raise ValueError(
                    f"Input has {x.shape[1]} channel(s), but "
                    f"{self._darkfield.shape[1]} BaSiC profile(s) were loaded."
                )
            self._basic_kernel(x, self._darkfield, self._inv_flatfield, x)
        out = cp.empty_like(x, dtype=cp.float32)
        for channel, projectors in enumerate(self._projectors):
            out[:, channel : channel + 1] = _deconvolve_lucyrichardson_guo(
                x[:, channel : channel + 1],
                projectors,
                iterations=self._iterations,
                cp=cp,
            )
        return out


@dataclass(frozen=True, slots=True)
class CupyDeconvolverFactory:
    basic_paths: tuple[Path, ...]
    psf_paths: tuple[Path, ...]
    iterations: int = 1
    process_safe: bool = True

    def __call__(self, device: int) -> CupyBasicRichardsonLucyDeconvolver:
        ensure_cuda_path_for_cupy_jit()
        return CupyBasicRichardsonLucyDeconvolver(
            basic_paths=self.basic_paths,
            psf_paths=self.psf_paths,
            device=device,
            iterations=self.iterations,
        )


def ensure_cuda_path_for_cupy_jit() -> None:
    """Set CUDA_PATH from the active conda env when CuPy needs JIT compilation."""
    if os.environ.get("CUDA_PATH"):
        return
    cuda_path = _conda_cuda_target(Path(sys.prefix))
    if cuda_path is not None:
        os.environ["CUDA_PATH"] = str(cuda_path)


def _conda_cuda_target(prefix: Path) -> Path | None:
    target = prefix / "targets" / "x86_64-linux"
    if not (target / "include" / "cuda.h").exists():
        return None
    for lib_dir in (target / "lib", target / "lib64"):
        if any(lib_dir.glob("libnvrtc.so*")):
            return target
    return None


def _initialize_cupy_allocator(cp: Any) -> tuple[Any, Any]:
    cuda = cp.cuda
    pool = cuda.MemoryPool()
    cuda.set_allocator(pool.malloc)
    pinned_pool = cuda.PinnedMemoryPool()
    cuda.set_pinned_memory_allocator(pinned_pool.malloc)
    return pool, pinned_pool


def _load_basic_profiles(paths: Sequence[Path], *, cp: Any) -> tuple[Any, Any]:
    darks: list[np.ndarray] = []
    flats: list[np.ndarray] = []
    reference_shape: tuple[int, int] | None = None

    for pkl_path in paths:
        loaded = pickle.loads(Path(pkl_path).read_bytes())
        basic = loaded.get("basic") if isinstance(loaded, dict) else loaded
        if basic is None:
            raise ValueError(f"BaSiC profile {pkl_path} does not contain a 'basic' payload.")

        dark = np.asarray(basic.darkfield, dtype=np.float32)
        flat = np.asarray(basic.flatfield, dtype=np.float32)
        _validate_basic_profile(pkl_path, dark=dark, flat=flat)
        if dark.shape != flat.shape:
            raise ValueError(f"Dark/flat shape mismatch for {pkl_path}: {dark.shape} vs {flat.shape}.")
        if reference_shape is None:
            reference_shape = dark.shape
        elif dark.shape != reference_shape:
            raise ValueError("Inconsistent BaSiC profile geometry; all profiles must match.")
        darks.append(dark)
        flats.append(flat)

    if not darks:
        raise ValueError("At least one BaSiC profile is required for the GPU production engine.")

    darkfield = np.stack(darks, axis=0)[np.newaxis, ...].astype(np.float32, copy=False)
    flatfield = np.stack(flats, axis=0)[np.newaxis, ...].astype(np.float32, copy=False)
    return cp.asarray(darkfield, dtype=cp.float32), cp.asarray(flatfield, dtype=cp.float32)


def _validate_basic_profile(path: Path, *, dark: np.ndarray, flat: np.ndarray) -> None:
    if dark.ndim != 2:
        raise ValueError(f"BaSiC darkfield for {path} must be 2-D, got shape {dark.shape}.")
    if flat.ndim != 2:
        raise ValueError(f"BaSiC flatfield for {path} must be 2-D, got shape {flat.shape}.")
    if not np.isfinite(dark).all():
        raise ValueError(f"BaSiC darkfield for {path} must contain only finite values.")
    if not np.isfinite(flat).all():
        raise ValueError(f"BaSiC flatfield for {path} must contain only finite values.")
    if not (flat > 0).all():
        raise ValueError(f"BaSiC flatfield for {path} must be strictly positive.")


@functools.cache
def _projectors(
    psf_path: Path,
    *,
    cp: Any,
    device: int,
) -> tuple[Any, Any]:
    cp.cuda.Device(device).use()
    psf_path = Path(psf_path)
    cache_path = _projector_cache_path(psf_path)
    if not cache_path.exists():
        lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
        with open(lock_path, "w") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            if not cache_path.exists():
                _make_projectors(
                    psf_path,
                    output=cache_path,
                    cp=cp,
                )
    return tuple(x.astype(cp.float32) for x in cp.load(cache_path))


def _projector_cache_path(psf_path: Path) -> Path:
    digest = hashlib.blake2b(psf_path.read_bytes(), digest_size=8).hexdigest()
    return psf_path.with_name(f"{psf_path.stem}.{digest}.fishtools-exact-projectors.npy")


def _make_projectors(
    path: Path,
    *,
    output: Path,
    cp: Any,
) -> None:
    gen = tifffile.imread(path)
    if gen.ndim != 3:
        raise ValueError(f"Expected 3-D PSF at {path}, got shape {gen.shape}")
    psf = gen[::-1].astype(np.float32, copy=False)
    psf_sum = float(psf.sum())
    if psf_sum <= 0:
        raise ValueError(f"PSF {path} has non-positive sum.")
    psf /= np.float32(psf_sum)
    projectors = [
        cp.asnumpy(projector)[:, np.newaxis, ...]
        for projector in _calculate_projectors_3d(cp.asarray(psf), sigma_g=1.7, a=0.02, b=0.02, n=10, cp=cp)
    ]
    _atomic_save_projectors(output, projectors)


def _atomic_save_projectors(path: Path, projectors: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as tmp_fp:
        np.save(tmp_fp, projectors)
        tmp_fp.flush()
        fsync(tmp_fp.fileno())
        tmp_path = Path(tmp_fp.name)
    try:
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _calculate_projectors_3d(
    pf: Any, *, sigma_g: float, a: float, b: float, n: int, cp: Any
) -> tuple[Any, Any]:
    pf_fft = cp.fft.fftn(pf)
    kc = 1.0 / (0.5 * 2.355 * sigma_g)
    kz = cp.fft.fftfreq(pf_fft.shape[0])
    kw = cp.fft.fftfreq(pf_fft.shape[1])
    kk = cp.sqrt((cp.asarray(cp.meshgrid(kz, kw, kw, indexing="ij")) ** 2).sum(axis=0))
    wiener = pf_fft / (cp.abs(pf_fft) ** 2 + a)
    eps = cp.sqrt(1.0 / (b**2) - 1)
    butterworth = 1.0 / cp.sqrt(1.0 + eps**2 * (kk / kc) ** (2 * n))
    backward = cp.real(cp.fft.ifftn(wiener * butterworth))
    return pf, backward


def _deconvolve_lucyrichardson_guo(
    img: Any, projectors: tuple[Any, Any], *, iterations: int = 1, cp: Any
) -> Any:
    from cupyx.scipy.ndimage import convolve as cconvolve

    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if img.dtype not in [cp.float32, cp.float16]:
        raise ValueError("Image must be float32 or float16.")
    forward_projector, backward_projector = projectors
    observed = cp.clip(img, EPS, None)
    estimate = observed.copy()
    for _ in range(iterations):
        filtered_estimate = cconvolve(estimate, forward_projector, mode="reflect").clip(EPS, I_MAX)
        ratio = observed / filtered_estimate
        estimate *= cconvolve(ratio, backward_projector, mode="reflect")
    return estimate


def _quantize_global_gpu(data: Any, params: ScalingParameters, *, cp: Any) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected (Z, C, Y, X) data, got {data.shape}")
    if data.shape[1] != params.offset.shape[0]:
        raise ValueError(f"Data has {data.shape[1]} channel(s), scaling has {params.offset.shape[0]}")
    offset = cp.asarray(params.offset, dtype=cp.float32).reshape(1, -1, 1, 1)
    scale = cp.asarray(params.scale, dtype=cp.float32).reshape(1, -1, 1, 1)
    arr = scale * (data - offset)
    arr = arr.clip(0.0, float(params.i_max))
    arr = cp.rint(arr).astype(cp.uint16, copy=False)
    return arr.reshape(-1, data.shape[-2], data.shape[-1]).get()
