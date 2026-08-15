from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.ctypeslib import ndpointer


DEFAULT_LIB_DIR = Path("/home/chaichontat/microImageLib/bin/linux")
ZYX_TO_XYZ = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
MATTES_OPTIMIZE_TRANSLATION_AND_SHEAR = 0
MATTES_OPTIMIZE_TRANSLATION_FIXED_SHEAR = 1


@dataclass(frozen=True)
class NativeReg3DResult:
    registered_zyx: np.ndarray
    matrix_zyx: np.ndarray
    offset_zyx: np.ndarray
    matrix_xyz_3x4: np.ndarray
    records: np.ndarray
    return_code: int


@dataclass(frozen=True)
class NativeReg3DDeviceResult:
    registered_zyx: Any
    matrix_zyx: np.ndarray
    offset_zyx: np.ndarray
    matrix_xyz_3x4: np.ndarray
    records: np.ndarray
    return_code: int


def zyx_to_xyz_3x4(matrix_zyx: np.ndarray, translation_zyx: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix_zyx, dtype=np.float32)
    translation = np.asarray(translation_zyx, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix_zyx must have shape (3, 3), got {matrix.shape}")
    if translation.shape != (3,):
        raise ValueError(f"translation_zyx must have shape (3,), got {translation.shape}")
    output = np.zeros((3, 4), dtype=np.float32)
    output[:, :3] = ZYX_TO_XYZ @ matrix @ ZYX_TO_XYZ
    output[:, 3] = ZYX_TO_XYZ @ translation
    return output



def register_method6_device(
    fixed_zyx: Any,
    moving_zyx: Any,
    *,
    fixed_mask_zyx: Any | None = None,
    lib_dir: Path = DEFAULT_LIB_DIR,
    ftol: float = 1e-4,
    max_iterations: int = 300,
    device: int = 0,
    initial_matrix_xyz_3x4: np.ndarray | None = None,
    aff_pivot_zyx: np.ndarray | None = None,
) -> NativeReg3DDeviceResult:
    """Run Method 6 with an optional zyx voxel pivot for both optimization stages."""
    return _register_ncc_affine_device(
        fixed_zyx,
        moving_zyx,
        method=6,
        fixed_mask_zyx=fixed_mask_zyx,
        lib_dir=lib_dir,
        ftol=ftol,
        max_iterations=max_iterations,
        device=device,
        initial_matrix_xyz_3x4=initial_matrix_xyz_3x4,
        aff_pivot_zyx=aff_pivot_zyx,
    )


def register_method8_device(
    fixed_zyx: Any,
    moving_zyx: Any,
    *,
    fixed_mask_zyx: Any | None = None,
    lib_dir: Path = DEFAULT_LIB_DIR,
    ftol: float = 1e-4,
    max_iterations: int = 300,
    device: int = 0,
    initial_matrix_xyz_3x4: np.ndarray | None = None,
    method8_zero_z_shear: bool = False,
) -> NativeReg3DDeviceResult:
    return _register_ncc_affine_device(
        fixed_zyx,
        moving_zyx,
        method=8,
        fixed_mask_zyx=fixed_mask_zyx,
        lib_dir=lib_dir,
        ftol=ftol,
        max_iterations=max_iterations,
        device=device,
        initial_matrix_xyz_3x4=initial_matrix_xyz_3x4,
        method8_zero_z_shear=method8_zero_z_shear,
    )


def _register_ncc_affine_device(
    fixed_zyx: Any,
    moving_zyx: Any,
    *,
    method: int,
    fixed_mask_zyx: Any | None,
    lib_dir: Path,
    ftol: float,
    max_iterations: int,
    device: int,
    initial_matrix_xyz_3x4: np.ndarray | None,
    method8_zero_z_shear: bool = False,
    aff_pivot_zyx: np.ndarray | None = None,
) -> NativeReg3DDeviceResult:
    import cupy as cp

    if method not in (6, 8):
        raise ValueError(f"NCC affine device method must be 6 or 8, got {method}")
    if method != 8 and method8_zero_z_shear:
        raise ValueError("method8_zero_z_shear is only valid for Method 8")
    if method != 6 and aff_pivot_zyx is not None:
        raise ValueError("aff_pivot_zyx is only valid for Method 6")
    fixed = _require_cupy_float32_zyx(fixed_zyx, name="fixed_zyx", device=device)
    moving = _require_cupy_float32_zyx(moving_zyx, name="moving_zyx", device=device)
    if fixed.shape != moving.shape:
        raise ValueError(f"Method {method} requires same-shaped crops, got {fixed.shape} and {moving.shape}")
    fixed_mask = (
        None
        if fixed_mask_zyx is None
        else _require_cupy_mask_float32_zyx(fixed_mask_zyx, name="fixed_mask_zyx", device=device)
    )
    if fixed_mask is not None and fixed_mask.shape != fixed.shape:
        raise ValueError(f"fixed_mask_zyx must match fixed_zyx shape, got {fixed_mask.shape} and {fixed.shape}")

    registered = cp.empty_like(fixed)
    tmx = (
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
        if initial_matrix_xyz_3x4 is None
        else np.asarray(initial_matrix_xyz_3x4, dtype=np.float32).reshape(12).copy()
    )
    z, y, x = fixed.shape
    size_xyz = np.array([x, y, z], dtype=np.uint32)
    pivot_xyz = None
    if aff_pivot_zyx is not None:
        pivot_zyx = np.asarray(aff_pivot_zyx, dtype=np.float32)
        if pivot_zyx.shape != (3,):
            raise ValueError(f"aff_pivot_zyx must have shape (3,), got {pivot_zyx.shape}")
        if not np.all(np.isfinite(pivot_zyx)):
            raise ValueError(f"aff_pivot_zyx must be finite, got {pivot_zyx.tolist()}")
        pivot_xyz = np.ascontiguousarray(pivot_zyx[::-1])
    records = np.zeros(11, dtype=np.float32)
    lib = _load_libapi(lib_dir)
    use_mask = fixed_mask is not None
    function_name = f"reg_3dgpu_method{method}{'_mask' if use_mask else ''}_device"
    if not hasattr(lib, function_name):
        raise RuntimeError(f"{Path(lib_dir) / 'libapi.so'} does not export {function_name}")
    if method == 8:
        _set_method8_zero_z_shear(
            lib,
            enabled=method8_zero_z_shear,
            lib_path=Path(lib_dir) / "libapi.so",
        )
    common_args = (
        ctypes.c_void_p(int(registered.data.ptr)),
        tmx,
        ctypes.c_void_p(int(fixed.data.ptr)),
        ctypes.c_void_p(int(moving.data.ptr)),
    )
    if use_mask:
        args = [
            *common_args,
            ctypes.c_void_p(int(fixed_mask.data.ptr)),
            size_xyz,
            size_xyz,
            1,
            ctypes.c_float(ftol),
            int(max_iterations),
            0,
            int(device),
        ]
    else:
        args = [
            *common_args,
            size_xyz,
            size_xyz,
            1,
            ctypes.c_float(ftol),
            int(max_iterations),
            0,
            int(device),
        ]
    if method == 6:
        args.append(
            None
            if pivot_xyz is None
            else pivot_xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
    return_code = getattr(lib, function_name)(*args, records)
    cp.cuda.Device(device).synchronize()
    matrix_zyx, offset_zyx = _tmx_xyz_to_zyx(tmx)
    return NativeReg3DDeviceResult(
        registered_zyx=registered,
        matrix_zyx=matrix_zyx,
        offset_zyx=offset_zyx,
        matrix_xyz_3x4=tmx.reshape(3, 4).copy(),
        records=records.copy(),
        return_code=int(return_code),
    )


def register_method10_mattes_device(
    fixed_zyx: Any,
    moving_zyx: Any,
    *,
    fixed_mask_zyx: Any | None = None,
    lib_dir: Path = DEFAULT_LIB_DIR,
    ftol: float = 1e-4,
    max_iterations: int = 300,
    device: int = 0,
    histogram_bins: int = 50,
    sample_count: int = 100_000,
    initial_matrix_xyz_3x4: np.ndarray | None = None,
    translation_only: bool = False,
) -> NativeReg3DDeviceResult:
    """Run Mattes registration, optionally preserving the initial symmetric shear."""
    return _register_mattes_device(
        fixed_zyx,
        moving_zyx,
        method=10,
        fixed_mask_zyx=fixed_mask_zyx,
        lib_dir=lib_dir,
        ftol=ftol,
        max_iterations=max_iterations,
        device=device,
        histogram_bins=histogram_bins,
        sample_count=sample_count,
        initial_matrix_xyz_3x4=initial_matrix_xyz_3x4,
        translation_only=translation_only,
    )


def register_method11_mattes_device(
    fixed_zyx: Any,
    moving_zyx: Any,
    *,
    fixed_mask_zyx: Any | None = None,
    lib_dir: Path = DEFAULT_LIB_DIR,
    ftol: float = 1e-4,
    max_iterations: int = 300,
    device: int = 0,
    histogram_bins: int = 50,
    sample_count: int = 100_000,
    initial_matrix_xyz_3x4: np.ndarray | None = None,
    translation_only: bool = False,
) -> NativeReg3DDeviceResult:
    """Run Method 11 Mattes registration with GPU-native sample preparation."""
    return _register_mattes_device(
        fixed_zyx,
        moving_zyx,
        method=11,
        fixed_mask_zyx=fixed_mask_zyx,
        lib_dir=lib_dir,
        ftol=ftol,
        max_iterations=max_iterations,
        device=device,
        histogram_bins=histogram_bins,
        sample_count=sample_count,
        initial_matrix_xyz_3x4=initial_matrix_xyz_3x4,
        translation_only=translation_only,
    )


def _register_mattes_device(
    fixed_zyx: Any,
    moving_zyx: Any,
    *,
    method: int,
    fixed_mask_zyx: Any | None,
    lib_dir: Path,
    ftol: float,
    max_iterations: int,
    device: int,
    histogram_bins: int,
    sample_count: int,
    initial_matrix_xyz_3x4: np.ndarray | None,
    translation_only: bool,
) -> NativeReg3DDeviceResult:
    import cupy as cp

    if method not in (10, 11):
        raise ValueError(f"Mattes method must be 10 or 11, got {method}")
    if histogram_bins < 6:
        raise ValueError("histogram_bins must be >= 6")
    if sample_count < histogram_bins:
        raise ValueError("sample_count must be >= histogram_bins")
    if translation_only and initial_matrix_xyz_3x4 is None:
        raise ValueError("translation_only requires initial_matrix_xyz_3x4")
    fixed = _require_cupy_float32_zyx(fixed_zyx, name="fixed_zyx", device=device)
    moving = _require_cupy_float32_zyx(moving_zyx, name="moving_zyx", device=device)
    if fixed.shape != moving.shape:
        raise ValueError(f"Method {method} requires same-shaped crops, got {fixed.shape} and {moving.shape}")
    fixed_mask = (
        None
        if fixed_mask_zyx is None
        else _require_cupy_mask_float32_zyx(fixed_mask_zyx, name="fixed_mask_zyx", device=device)
    )
    if fixed_mask is not None and fixed_mask.shape != fixed.shape:
        raise ValueError(f"fixed_mask_zyx must match fixed_zyx shape, got {fixed_mask.shape} and {fixed.shape}")

    registered = cp.empty_like(fixed)
    has_initial_matrix = initial_matrix_xyz_3x4 is not None
    tmx = (
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
        if not has_initial_matrix
        else np.asarray(initial_matrix_xyz_3x4, dtype=np.float32).reshape(12).copy()
    )
    z, y, x = fixed.shape
    size_xyz = np.array([x, y, z], dtype=np.uint32)
    records = np.zeros(11, dtype=np.float32)
    lib = _load_libapi(lib_dir)
    use_mask = fixed_mask is not None
    function_name = f"reg_3dgpu_method{method}_mattes{'_mask' if use_mask else ''}_device"
    if not hasattr(lib, function_name):
        raise RuntimeError(f"{Path(lib_dir) / 'libapi.so'} does not export {function_name}")
    common_args = (
        ctypes.c_void_p(int(registered.data.ptr)),
        tmx,
        ctypes.c_void_p(int(fixed.data.ptr)),
        ctypes.c_void_p(int(moving.data.ptr)),
    )
    trailing_args = (
        size_xyz,
        size_xyz,
        int(has_initial_matrix),
        ctypes.c_float(ftol),
        int(max_iterations),
        0,
        int(device),
        int(histogram_bins),
        int(sample_count),
        MATTES_OPTIMIZE_TRANSLATION_FIXED_SHEAR
        if translation_only
        else MATTES_OPTIMIZE_TRANSLATION_AND_SHEAR,
        records,
    )
    if use_mask:
        return_code = getattr(lib, function_name)(
            *common_args,
            ctypes.c_void_p(int(fixed_mask.data.ptr)),
            *trailing_args,
        )
    else:
        return_code = getattr(lib, function_name)(*common_args, *trailing_args)
    cp.cuda.Device(device).synchronize()
    matrix_zyx, offset_zyx = _tmx_xyz_to_zyx(tmx)
    return NativeReg3DDeviceResult(
        registered_zyx=registered,
        matrix_zyx=matrix_zyx,
        offset_zyx=offset_zyx,
        matrix_xyz_3x4=tmx.reshape(3, 4).copy(),
        records=records.copy(),
        return_code=int(return_code),
    )


def _require_cupy_float32_zyx(array: Any, *, name: str, device: int) -> Any:
    import cupy as cp

    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy ndarray for device registration")
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D zyx array, got ndim={array.ndim}")
    if array.dtype != cp.float32:
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}")
    if array.device.id != int(device):
        raise ValueError(f"{name} is on CUDA device {array.device.id}, but device={device}")
    return cp.ascontiguousarray(array)


def _require_cupy_mask_float32_zyx(array: Any, *, name: str, device: int) -> Any:
    import cupy as cp

    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy ndarray for masked device registration")
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D zyx array, got ndim={array.ndim}")
    if array.device.id != int(device):
        raise ValueError(f"{name} is on CUDA device {array.device.id}, but device={device}")
    return cp.ascontiguousarray(array, dtype=cp.float32)


def _run_reg_3dgpu(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    *,
    aff_method: int,
    lib_dir: Path,
    ftol: float,
    max_iterations: int,
    device: int,
    tmx_only: bool,
    initial_matrix_xyz_3x4: np.ndarray | None = None,
    aff_pivot_xyz: np.ndarray | None = None,
    aff_pivot_zyx: np.ndarray | None = None,
    default_center_pivot: bool = False,
    weighted_ncc: bool = False,
    weighted_ncc_min_overlap: float = 0.80,
    weighted_ncc_floor: float = 1e-3,
) -> NativeReg3DResult:
    if fixed_zyx.ndim != 3:
        raise ValueError(f"reg_3dgpu requires 3D zyx arrays, got ndim={fixed_zyx.ndim}")
    if moving_zyx.ndim != 3:
        raise ValueError(f"reg_3dgpu requires 3D zyx arrays, got ndim={moving_zyx.ndim}")
    if fixed_zyx.shape != moving_zyx.shape:
        raise ValueError(f"reg_3dgpu requires same-shaped crops, got {fixed_zyx.shape} and {moving_zyx.shape}")

    fixed = np.ascontiguousarray(fixed_zyx, dtype=np.float32).ravel()
    moving = np.ascontiguousarray(moving_zyx, dtype=np.float32).ravel()
    registered = np.zeros_like(fixed)
    tmx = (
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
        if initial_matrix_xyz_3x4 is None
        else np.asarray(initial_matrix_xyz_3x4, dtype=np.float32).reshape(12).copy()
    )
    z, y, x = fixed_zyx.shape
    size_xyz = np.array([x, y, z], dtype=np.uint32)
    records = np.zeros(11, dtype=np.float32)
    pivot = _pivot_xyz(
        aff_pivot_xyz=aff_pivot_xyz,
        aff_pivot_zyx=aff_pivot_zyx,
        shape_zyx=np.asarray([z, y, x], dtype=np.float32),
        tmx_only=tmx_only,
        default_center_pivot=default_center_pivot,
    )
    lib = _load_libapi(lib_dir)
    _set_weighted_ncc(
        lib,
        enabled=weighted_ncc,
        min_overlap=weighted_ncc_min_overlap,
        floor=weighted_ncc_floor,
        lib_path=lib_dir / "libapi.so",
    )
    if tmx_only and hasattr(lib, "reg_3dgpu_tmxonly"):
        return_code = lib.reg_3dgpu_tmxonly(
            registered,
            tmx,
            fixed,
            moving,
            size_xyz,
            size_xyz,
            int(aff_method),
            1,
            ctypes.c_float(ftol),
            int(max_iterations),
            0,
            int(device),
            records,
        )
    elif pivot is not None:
        if not hasattr(lib, "reg_3dgpu_pivot"):
            raise RuntimeError(f"{lib_dir / 'libapi.so'} does not export reg_3dgpu_pivot")
        return_code = lib.reg_3dgpu_pivot(
            registered,
            tmx,
            fixed,
            moving,
            size_xyz,
            size_xyz,
            int(aff_method),
            1,
            ctypes.c_float(ftol),
            int(max_iterations),
            0,
            int(device),
            pivot.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            records,
        )
    else:
        return_code = lib.reg_3dgpu(
            registered,
            tmx,
            fixed,
            moving,
            size_xyz,
            size_xyz,
            int(aff_method),
            1,
            ctypes.c_float(ftol),
            int(max_iterations),
            0,
            int(device),
            records,
        )
    matrix_zyx, offset_zyx = _tmx_xyz_to_zyx(tmx)
    return NativeReg3DResult(
        registered_zyx=registered.reshape(fixed_zyx.shape),
        matrix_zyx=matrix_zyx,
        offset_zyx=offset_zyx,
        matrix_xyz_3x4=tmx.reshape(3, 4).copy(),
        records=records.copy(),
        return_code=int(return_code),
    )


def _set_weighted_ncc(
    lib: ctypes.CDLL,
    *,
    enabled: bool,
    min_overlap: float,
    floor: float,
    lib_path: Path,
) -> None:
    if not hasattr(lib, "set_reg_weighted_ncc"):
        if enabled:
            raise RuntimeError(f"{lib_path} does not export set_reg_weighted_ncc")
        return
    lib.set_reg_weighted_ncc(int(bool(enabled)), ctypes.c_float(float(min_overlap)), ctypes.c_float(float(floor)))


def _set_method8_zero_z_shear(
    lib: ctypes.CDLL,
    *,
    enabled: bool,
    lib_path: Path,
) -> None:
    if not hasattr(lib, "set_method8_xy_shear_only"):
        raise RuntimeError(f"{lib_path} does not export set_method8_xy_shear_only")
    lib.set_method8_xy_shear_only(int(bool(enabled)))


def _pivot_xyz(
    *,
    aff_pivot_xyz: np.ndarray | None,
    aff_pivot_zyx: np.ndarray | None,
    shape_zyx: np.ndarray,
    tmx_only: bool,
    default_center_pivot: bool,
) -> np.ndarray | None:
    if aff_pivot_xyz is not None and aff_pivot_zyx is not None:
        raise ValueError("Provide only one of aff_pivot_xyz or aff_pivot_zyx")
    if (aff_pivot_xyz is not None or aff_pivot_zyx is not None) and tmx_only:
        raise ValueError("aff_pivot_xyz/aff_pivot_zyx is only supported by reg_3dgpu_pivot")
    if aff_pivot_xyz is not None:
        pivot = np.asarray(aff_pivot_xyz, dtype=np.float32)
    elif aff_pivot_zyx is not None:
        pivot = np.asarray(aff_pivot_zyx, dtype=np.float32)[::-1]
    elif default_center_pivot:
        pivot = ((shape_zyx - 1.0) / 2.0)[::-1]
    else:
        return None
    if pivot.shape != (3,):
        raise ValueError(f"Affine pivot must have shape (3,), got {pivot.shape}")
    if not np.all(np.isfinite(pivot)):
        raise ValueError(f"Affine pivot must be finite, got {pivot.tolist()}")
    return np.ascontiguousarray(pivot, dtype=np.float32)


def _load_libapi(lib_dir: Path) -> ctypes.CDLL:
    lib_dir = Path(lib_dir)
    dependency_dirs = [
        lib_dir,
        Path(sys.prefix) / "lib",
        Path(sys.prefix) / "targets" / "x86_64-linux" / "lib",
        *sorted((Path(sys.prefix) / "lib").glob("python*/site-packages/nvidia/*/lib")),
        *(Path(path) for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path),
    ]
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_root = Path(cuda_path)
        dependency_dirs.extend(
            [
                cuda_root / "lib",
                cuda_root / "targets" / "x86_64-linux" / "lib",
                *sorted((cuda_root / "lib").glob("python*/site-packages/nvidia/*/lib")),
            ]
        )
    for candidates in (
        ("libtiff.so.5", "libtiff.so.6"),
        ("libcudart.so.10.0", "libcudart.so.12"),
        ("libcufft.so.10.0", "libcufft.so.11"),
        ("libfftw3f.so.3",),
    ):
        for directory in dependency_dirs:
            for name in candidates:
                path = directory / name
                if not path.exists():
                    continue
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                break
            else:
                continue
            break
    lib = ctypes.CDLL(str(lib_dir / "libapi.so"))
    lib.reg_3dgpu.restype = ctypes.c_int
    reg_3dgpu_argtypes = [
        ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ndpointer(np.uint32, flags="C_CONTIGUOUS"),
        ndpointer(np.uint32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ndpointer(np.float32, flags="C_CONTIGUOUS"),
    ]
    lib.reg_3dgpu.argtypes = reg_3dgpu_argtypes
    if hasattr(lib, "reg_3dgpu_pivot"):
        lib.reg_3dgpu_pivot.restype = ctypes.c_int
        lib.reg_3dgpu_pivot.argtypes = [
            *reg_3dgpu_argtypes[:-1],
            ctypes.POINTER(ctypes.c_float),
            reg_3dgpu_argtypes[-1],
        ]
    try:
        lib.reg_3dgpu_tmxonly.restype = ctypes.c_int
        lib.reg_3dgpu_tmxonly.argtypes = reg_3dgpu_argtypes
    except AttributeError:
        pass
    if hasattr(lib, "set_reg_weighted_ncc"):
        lib.set_reg_weighted_ncc.restype = None
        lib.set_reg_weighted_ncc.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
    lib.set_method8_xy_shear_only.restype = None
    lib.set_method8_xy_shear_only.argtypes = [ctypes.c_int]
    method8_device_argtypes = [
        ctypes.c_void_p,
        ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ndpointer(np.uint32, flags="C_CONTIGUOUS"),
        ndpointer(np.uint32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ndpointer(np.float32, flags="C_CONTIGUOUS"),
    ]
    for method in (6, 8):
        device_argtypes = (
            [
                *method8_device_argtypes[:-1],
                ctypes.POINTER(ctypes.c_float),
                method8_device_argtypes[-1],
            ]
            if method == 6
            else method8_device_argtypes
        )
        function_name = f"reg_3dgpu_method{method}_device"
        if hasattr(lib, function_name):
            function = getattr(lib, function_name)
            function.restype = ctypes.c_int
            function.argtypes = device_argtypes
        mask_function_name = f"reg_3dgpu_method{method}_mask_device"
        if hasattr(lib, mask_function_name):
            mask_function = getattr(lib, mask_function_name)
            mask_function.restype = ctypes.c_int
            mask_function.argtypes = [
                *device_argtypes[:4],
                ctypes.c_void_p,
                *device_argtypes[4:],
            ]
    method10_device_argtypes = [
        *method8_device_argtypes[:-1],
        ctypes.c_uint,
        ctypes.c_ulonglong,
        ctypes.c_int,
        method8_device_argtypes[-1],
    ]
    for method in (10, 11):
        function_name = f"reg_3dgpu_method{method}_mattes_device"
        if hasattr(lib, function_name):
            function = getattr(lib, function_name)
            function.restype = ctypes.c_int
            function.argtypes = method10_device_argtypes
        mask_function_name = f"reg_3dgpu_method{method}_mattes_mask_device"
        if hasattr(lib, mask_function_name):
            mask_function = getattr(lib, mask_function_name)
            mask_function.restype = ctypes.c_int
            mask_function.argtypes = [
                *method10_device_argtypes[:4],
                ctypes.c_void_p,
                *method10_device_argtypes[4:],
            ]
    return lib


def _tmx_xyz_to_zyx(tmx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    affine_xyz = np.asarray(tmx, dtype=np.float32).reshape(3, 4)
    matrix_xyz = affine_xyz[:, :3]
    offset_xyz = affine_xyz[:, 3]
    return ZYX_TO_XYZ @ matrix_xyz @ ZYX_TO_XYZ, ZYX_TO_XYZ @ offset_xyz
