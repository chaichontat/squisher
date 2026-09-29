"""Fit and stream a linear spillover subtraction between two completed same-grid fused volumes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from squisher.jpegxr_zarr import register_jpegxr_codec
from squisher_lightsheet import ngff

MODEL_ARTIFACT_TYPE = "squisher_lightsheet.fused_subtraction_model.v1"
DEFAULT_QUANTILE = 0.05
# Scales intensities near unity for the LP solver; the fitted intercept is returned in input units.
_FIT_INTENSITY_SCALE = 1000.0


def _open_complete(path: Path) -> Any:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    if root.attrs.get("squisher_complete") is not True:
        raise ValueError(f"Fused volume is not marked complete: {path}")
    return root


def _grid(root: Any, path: Path) -> list[tuple[str, tuple[int, ...], list[float], list[float]]]:
    levels = []
    for index, level_path in enumerate(ngff.dataset_paths(root)):
        array = root[level_path]
        if ngff.axes(root, array) != "ZYX":
            raise ValueError(f"{path}/{level_path} must be ZYX, got {ngff.axes(root, array)}")
        _names, scale, origin, _has_scale, _has_origin = ngff.scale_translation(root, dataset_index=index)
        levels.append((level_path, tuple(int(size) for size in array.shape), scale, origin))
    return levels


def _open_same_grid(target: Path, reference: Path) -> tuple[Any, Any, list[str]]:
    target_root = _open_complete(target)
    reference_root = _open_complete(reference)
    target_grid = _grid(target_root, target)
    reference_grid = _grid(reference_root, reference)
    same = len(target_grid) == len(reference_grid) and all(
        a[:2] == b[:2] and np.allclose(a[2], b[2]) and np.allclose(a[3], b[3])
        for a, b in zip(target_grid, reference_grid, strict=False)
    )
    if not same:
        raise ValueError(f"Target {target} and reference {reference} do not share one multiscale grid")
    paths = [level[0] for level in target_grid]
    for level_path in paths:
        if target_root[level_path].dtype != reference_root[level_path].dtype:
            raise ValueError(f"Target and reference level {level_path} dtypes differ")
    return target_root, reference_root, paths


def fit_quantile_line(reference: np.ndarray, target: np.ndarray, *, quantile: float) -> tuple[float, float]:
    """Return (alpha, beta) minimizing the pinball loss of target ~ alpha * reference + beta."""

    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile}")
    count = len(target)
    if count < 2 or len(reference) != count:
        raise ValueError("Quantile fit needs at least two paired samples")
    design = sparse.csr_array(np.column_stack([reference / _FIT_INTENSITY_SCALE, np.ones(count)]))
    identity = sparse.identity(count, format="csr")
    result = linprog(
        np.concatenate([[0.0, 0.0], np.full(count, quantile), np.full(count, 1.0 - quantile)]),
        A_eq=sparse.hstack([design, identity, -identity], format="csr"),
        b_eq=target / _FIT_INTENSITY_SCALE,
        bounds=[(None, None), (None, None)] + [(0.0, None)] * (2 * count),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Quantile fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1] * _FIT_INTENSITY_SCALE)


def fit_fused_subtraction(
    *,
    target: Path,
    reference: Path,
    output_dir: Path,
    quantile: float = DEFAULT_QUANTILE,
    planes: int = 16,
    stride_yx: int = 32,
    max_samples: int = 50_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit target ~ alpha * reference + beta on the lower envelope of level-0 voxels both volumes support."""

    register_jpegxr_codec()
    target = target.resolve()
    reference = reference.resolve()
    target_root, reference_root, paths = _open_same_grid(target, reference)
    target_array = target_root[paths[0]]
    reference_array = reference_root[paths[0]]
    _names, scale, origin, _has_scale, _has_origin = ngff.scale_translation(target_root, dataset_index=0)
    shape = target_array.shape
    z_indices = np.unique(np.rint(np.linspace(0, shape[0] - 1, planes + 2)[1:-1]).astype(int))

    target_samples, reference_samples, sample_planes = [], [], []
    for z in z_indices:
        target_plane = np.asarray(target_array[int(z), ::stride_yx, ::stride_yx], dtype=np.float64).ravel()
        reference_plane = np.asarray(reference_array[int(z), ::stride_yx, ::stride_yx], dtype=np.float64).ravel()
        supported = (target_plane > 0) & (reference_plane > 0)
        target_samples.append(target_plane[supported])
        reference_samples.append(reference_plane[supported])
        sample_planes.append(np.full(int(supported.sum()), int(z)))
    target_values = np.concatenate(target_samples)
    reference_values = np.concatenate(reference_samples)
    plane_values = np.concatenate(sample_planes)
    if len(target_values) < 2:
        raise ValueError(f"Target {target} and reference {reference} share no supported sampled voxels")
    # Bright structure outside the spillover relation stays out of the fit.
    keep = (target_values < np.percentile(target_values, 99.5)) & (
        reference_values < np.percentile(reference_values, 99.5)
    )
    target_values, reference_values, plane_values = target_values[keep], reference_values[keep], plane_values[keep]
    if len(target_values) > max_samples:
        selected = np.sort(np.random.default_rng(seed).choice(len(target_values), max_samples, replace=False))
        target_values, reference_values, plane_values = (
            target_values[selected],
            reference_values[selected],
            plane_values[selected],
        )

    alpha, beta = fit_quantile_line(reference_values, target_values, quantile=quantile)
    residual = target_values - alpha * reference_values - beta
    per_plane = [
        {
            "z_index": int(z),
            "z_um": float(origin[0] + scale[0] * z),
            "samples": int((plane_values == z).sum()),
            "residual_quantile": float(np.quantile(residual[plane_values == z], quantile)),
            "residual_median": float(np.median(residual[plane_values == z])),
        }
        for z in z_indices
        if np.any(plane_values == z)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "fused_subtraction_model.json"
    samples_path = output_dir / "fused_subtraction_samples.npz"
    qc_path = output_dir / "fused_subtraction_fit_qc.png"
    np.savez_compressed(samples_path, target=target_values, reference=reference_values, z_index=plane_values)
    model = {
        "artifact_type": MODEL_ARTIFACT_TYPE,
        "target": str(target),
        "reference": str(reference),
        "model": "target - alpha * reference - beta, clipped at 0, zero where either volume is unsupported",
        "quantile": quantile,
        "alpha": alpha,
        "beta": beta,
        "sampling": {
            "level": paths[0],
            "z_indices": z_indices.tolist(),
            "stride_yx": stride_yx,
            "max_samples": max_samples,
            "seed": seed,
            "upper_trim_percentile": 99.5,
        },
        "sample_count": len(target_values),
        "nonpositive_fraction_before_clip": float(np.mean(residual <= 0)),
        "residual_median": float(np.median(residual)),
        "residual_reference_correlation": float(np.corrcoef(residual, reference_values)[0, 1]),
        "per_plane": per_plane,
        "samples": str(samples_path),
        "qc_image": str(qc_path),
    }
    _plot_fit(target_values, reference_values, residual, per_plane, model=model, output=qc_path)
    model_path.write_text(json.dumps(model, indent=2) + "\n")
    return model | {"model_path": str(model_path)}


def _plot_fit(
    target: np.ndarray,
    reference: np.ndarray,
    residual: np.ndarray,
    per_plane: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha, beta, quantile = model["alpha"], model["beta"], model["quantile"]
    reference_limits = np.percentile(reference, [0.5, 99.5])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax = axes[0]
    ax.hexbin(reference, target, gridsize=90, bins="log", mincnt=1, cmap="viridis")
    line = np.linspace(*reference_limits, 200)
    ax.plot(line, alpha * line + beta, color="tab:red", linewidth=2)
    ax.set(
        xlim=reference_limits,
        ylim=np.percentile(target, [0.5, 99.5]),
        title=f"q={quantile:g} fit: target = {alpha:.3f} × reference + {beta:.1f}",
        xlabel="fused reference intensity",
        ylabel="fused target intensity",
    )
    ax = axes[1]
    ax.hexbin(reference, residual, gridsize=90, bins="log", mincnt=1, cmap="viridis")
    edges = np.quantile(reference, np.linspace(0, 1, 21))
    bins = [(reference >= low) & (reference <= high) for low, high in zip(edges[:-1], edges[1:], strict=True)]
    bins = [selected for selected in bins if np.any(selected)]
    centers = [float(np.median(reference[selected])) for selected in bins]
    ax.plot(centers, [np.quantile(residual[s], quantile) for s in bins], color="tab:red", label=f"binned q{quantile:g}")
    ax.plot(centers, [np.median(residual[s]) for s in bins], color="white", label="binned median")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set(
        xlim=reference_limits,
        ylim=np.percentile(residual, [0.5, 99.5]),
        title="subtracted intensity before clipping",
        xlabel="fused reference intensity",
        ylabel="target - alpha × reference - beta",
    )
    ax.legend(loc="upper left")
    ax = axes[2]
    z_um = [row["z_um"] for row in per_plane]
    ax.plot(z_um, [row["residual_quantile"] for row in per_plane], "o-", color="tab:red", label=f"q{quantile:g}")
    ax.plot(z_um, [row["residual_median"] for row in per_plane], "o-", color="tab:gray", label="median")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set(title="subtracted intensity by sampled plane", xlabel="z (µm)", ylabel="subtracted intensity")
    ax.legend(loc="upper left")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _chunk_selections(shape: tuple[int, ...], chunks: tuple[int, ...]) -> Iterator[tuple[slice, ...]]:
    counts = tuple(math.ceil(size / chunk) for size, chunk in zip(shape, chunks, strict=True))
    for flat_index in range(math.prod(counts)):
        coordinates = np.unravel_index(flat_index, counts)
        yield tuple(
            slice(coordinate * chunk, min(size, (coordinate + 1) * chunk))
            for coordinate, size, chunk in zip(coordinates, shape, chunks, strict=True)
        )


def subtract_block(target: np.ndarray, reference: np.ndarray, *, alpha: float, beta: float) -> tuple[np.ndarray, dict[str, int]]:
    """Return clip(target - alpha * reference - beta, 0), zero where either input is unsupported."""

    supported = (target > 0) & (reference > 0)
    signed = target.astype(np.float32) - np.float32(alpha) * reference.astype(np.float32) - np.float32(beta)
    signed[~supported] = 0
    counts = {
        "supported_voxels": int(supported.sum()),
        "nonpositive_supported_voxels": int((supported & (signed <= 0)).sum()),
        "target_only_voxels": int(((target > 0) & (reference == 0)).sum()),
    }
    np.rint(signed, out=signed)
    np.clip(signed, 0, np.iinfo(target.dtype).max, out=signed)
    return signed.astype(target.dtype), counts


def subtract_fused_volumes(
    *,
    model: Path,
    output: Path,
    zstd_level: int = 3,
    workers: int = 8,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stream the fitted subtraction over every pyramid level of the shared target/reference grid."""

    import zarr
    from zarr.codecs import BytesCodec, ZstdCodec

    payload = json.loads(model.read_text())
    if payload.get("artifact_type") != MODEL_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported fused subtraction model: {model}")
    register_jpegxr_codec()
    target_root, reference_root, paths = _open_same_grid(Path(payload["target"]), Path(payload["reference"]))
    alpha, beta = float(payload["alpha"]), float(payload["beta"])
    output = output.resolve()
    if not output.name.endswith(".zarr"):
        raise ValueError(f"Output must be a .zarr directory, got {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    existing = [path for path in (output, temporary) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output path(s): {', '.join(map(str, existing))}")
    for path in existing:
        shutil.rmtree(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    target_attrs = target_root.attrs.asdict()
    provenance = {
        "model": str(model.resolve()),
        "target": payload["target"],
        "reference": payload["reference"],
        "alpha": alpha,
        "beta": beta,
        "codec": "zstd",
        "zstd_level": zstd_level,
    }
    levels = []
    try:
        output_root = zarr.open_group(str(temporary), mode="w", zarr_format=3)
        output_root.attrs.update(
            {key: target_attrs[key] for key in ("ome", "multiscales") if key in target_attrs}
            | {"squisher_complete": False, "squisher_fused_subtraction": provenance}
        )
        for level_path in paths:
            target_array = target_root[level_path]
            reference_array = reference_root[level_path]
            shape = tuple(int(size) for size in target_array.shape)
            chunks = tuple(int(size) for size in target_array.chunks)
            output_array = output_root.create_array(
                level_path,
                shape=shape,
                chunks=chunks,
                dtype=target_array.dtype,
                fill_value=0,
                dimension_names=("z", "y", "x"),
                serializer=BytesCodec(),
                compressors=[ZstdCodec(level=zstd_level)],
            )
            if progress is not None:
                progress(f"Subtracting level {level_path}: shape={shape}, chunks={chunks}")

            def run(selection: tuple[slice, ...]) -> dict[str, int]:
                block, counts = subtract_block(
                    np.asarray(target_array[selection]),
                    np.asarray(reference_array[selection]),
                    alpha=alpha,
                    beta=beta,
                )
                if np.any(block):
                    output_array[selection] = block
                return counts

            totals = {"supported_voxels": 0, "nonpositive_supported_voxels": 0, "target_only_voxels": 0}
            pending: set[Future[dict[str, int]]] = set()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for selection in _chunk_selections(shape, chunks):
                    pending.add(executor.submit(run, selection))
                    if len(pending) < workers * 2:
                        continue
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        totals = {key: totals[key] + value for key, value in future.result().items()}
                for future in pending:
                    totals = {key: totals[key] + value for key, value in future.result().items()}
            levels.append({"path": level_path, "shape": list(shape), "chunks": list(chunks)} | totals)
        output_root.attrs["squisher_complete"] = True
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "artifact_type": "squisher_lightsheet.fused_subtraction.v1",
        "output": str(output),
        **provenance,
        "levels": levels,
    }


__all__ = [
    "DEFAULT_QUANTILE",
    "MODEL_ARTIFACT_TYPE",
    "fit_fused_subtraction",
    "fit_quantile_line",
    "subtract_block",
    "subtract_fused_volumes",
]
