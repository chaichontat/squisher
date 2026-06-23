from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import tifffile

from squisher_deconv.metadata import json_dumps_strict

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class ScalingParameters:
    offset: np.ndarray
    scale: np.ndarray
    p_low: float
    p_high: float
    gamma: float
    i_max: int


def save_float32_sample(path: Path, data: np.ndarray, *, metadata: dict[str, Any]) -> None:
    if data.dtype != np.float32:
        raise ValueError(f"Sample output must be float32, got {data.dtype}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing float32 sample {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        data.reshape(-1, *data.shape[-2:]),
        dtype=np.float32,
        metadata=metadata,
        photometric="minisblack",
    )


def collate_scaling(
    sample_paths: Sequence[Path],
    *,
    channels: int,
    out_dir: Path,
    p_low: float,
    p_high: float,
    gamma: float,
    bins: int,
    i_max: int = 65535,
    manifest: dict[str, Any],
    overwrite: bool = False,
) -> ScalingParameters:
    if not sample_paths:
        raise ValueError("At least one float32 sample is required to compute scaling.")
    per_channel = [[] for _ in range(channels)]
    for path in sample_paths:
        arr = tifffile.imread(path).astype(np.float32, copy=False)
        if arr.ndim != 3:
            raise ValueError(f"Expected flattened sample stack at {path}, got shape {arr.shape}")
        if arr.shape[0] % channels:
            raise ValueError(f"{path} has {arr.shape[0]} plane(s), not divisible by channels={channels}")
        arr = arr.reshape(-1, channels, *arr.shape[-2:])
        for channel in range(channels):
            per_channel[channel].append(arr[:, channel].reshape(-1))

    offsets = np.zeros(channels, dtype=np.float32)
    scales = np.zeros(channels, dtype=np.float32)
    histograms: list[tuple[np.ndarray, np.ndarray]] = []
    for channel, parts in enumerate(per_channel):
        values = np.concatenate(parts)
        if not np.isfinite(values).all():
            raise ValueError(f"Channel {channel} sample values must contain only finite values.")
        low = float(np.quantile(values, p_low))
        high = float(np.quantile(values, p_high))
        dynamic_range = (high - low) * gamma
        if dynamic_range <= 0:
            raise ValueError(f"Channel {channel} has non-positive scaling range: low={low}, high={high}")
        offsets[channel] = low
        scales[channel] = np.float32(i_max / dynamic_range)
        counts, edges = np.histogram(values, bins=bins)
        histograms.append((counts, edges))

    params = ScalingParameters(
        offset=offsets,
        scale=scales,
        p_low=float(p_low),
        p_high=float(p_high),
        gamma=float(gamma),
        i_max=int(i_max),
    )
    write_scaling_artifacts(
        out_dir,
        params,
        histograms=histograms,
        manifest=manifest,
        sample_paths=sample_paths,
        overwrite=overwrite,
    )
    return params


def write_scaling_artifacts(
    out_dir: Path,
    params: ScalingParameters,
    *,
    histograms: Sequence[tuple[np.ndarray, np.ndarray]],
    manifest: dict[str, Any],
    sample_paths: Sequence[Path],
    overwrite: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = [
        out_dir / "scaling.json",
        out_dir / "scaling.txt",
        out_dir / "histogram.csv",
        out_dir / "sample-manifest.json",
        out_dir / "scaling-qc.png",
    ]
    existing = [path for path in artifact_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing scaling artifact(s): {names}")
    scaling_json = {
        "offset": params.offset.tolist(),
        "scale": params.scale.tolist(),
        "p_low": params.p_low,
        "p_high": params.p_high,
        "gamma": params.gamma,
        "i_max": params.i_max,
        "sample_paths": [str(path) for path in sample_paths],
    }
    scaling_json_text = json_dumps_strict(scaling_json, context="Scaling parameters", indent=2)
    manifest_out = dict(manifest)
    manifest_out["scaling"] = scaling_json
    manifest_json_text = json_dumps_strict(manifest_out, context="Sample scaling manifest", indent=2)

    (out_dir / "scaling.json").write_text(scaling_json_text)
    np.savetxt(out_dir / "scaling.txt", np.vstack([params.offset, params.scale]))

    with open(out_dir / "histogram.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["channel", "bin_left", "bin_right", "count"])
        for channel, (counts, edges) in enumerate(histograms):
            for idx, count in enumerate(counts):
                writer.writerow([channel, f"{edges[idx]:.8g}", f"{edges[idx + 1]:.8g}", int(count)])

    (out_dir / "sample-manifest.json").write_text(manifest_json_text)
    _write_qc_plot(out_dir / "scaling-qc.png", histograms=histograms, params=params)


def load_scaling(path: Path) -> ScalingParameters:
    import json

    payload = json.loads(path.read_text())
    params = ScalingParameters(
        offset=np.asarray(payload["offset"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
        p_low=float(payload["p_low"]),
        p_high=float(payload["p_high"]),
        gamma=float(payload["gamma"]),
        i_max=int(payload["i_max"]),
    )
    validate_scaling_parameters(params, context=f"Scaling parameters in {path}")
    return params


def validate_scaling_parameters(params: ScalingParameters, *, context: str) -> None:
    if params.offset.ndim != 1:
        raise ValueError(f"{context} offset must be 1-D, got shape {params.offset.shape}")
    if params.scale.ndim != 1:
        raise ValueError(f"{context} scale must be 1-D, got shape {params.scale.shape}")
    if params.offset.shape != params.scale.shape:
        raise ValueError(
            f"{context} offset/scale channel count mismatch: "
            f"offset has {params.offset.shape[0]}, scale has {params.scale.shape[0]}"
        )
    if not np.isfinite(params.offset).all():
        raise ValueError(f"{context} offset must contain only finite values.")
    if not np.isfinite(params.scale).all():
        raise ValueError(f"{context} scale must contain only finite values.")
    if not (params.scale > 0).all():
        raise ValueError(f"{context} scale must be strictly positive.")
    numeric_fields = np.asarray([params.p_low, params.p_high, params.gamma, params.i_max], dtype=np.float64)
    if not np.isfinite(numeric_fields).all():
        raise ValueError(f"{context} scalar fields must contain only finite values.")
    if not 0 <= params.p_low <= 1:
        raise ValueError(f"{context} p_low must be in [0, 1], got {params.p_low}.")
    if not 0 <= params.p_high <= 1:
        raise ValueError(f"{context} p_high must be in [0, 1], got {params.p_high}.")
    if params.p_low >= params.p_high:
        raise ValueError(f"{context} p_low must be less than p_high, got {params.p_low} >= {params.p_high}.")
    if params.gamma <= 0:
        raise ValueError(f"{context} gamma must be strictly positive, got {params.gamma}.")
    if params.i_max <= 0:
        raise ValueError(f"{context} i_max must be strictly positive, got {params.i_max}.")


def validate_scaling_channels(params: ScalingParameters, *, channels: int, context: str) -> None:
    validate_scaling_parameters(params, context=context)
    scaling_channels = int(params.offset.shape[0])
    if scaling_channels != channels:
        raise ValueError(f"{context} has {scaling_channels} channel(s), but run was configured with channels={channels}")


def quantize_global(data: np.ndarray, params: ScalingParameters) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expected (Z, C, Y, X) data, got {data.shape}")
    if data.shape[1] != params.offset.shape[0]:
        raise ValueError(f"Data has {data.shape[1]} channel(s), scaling has {params.offset.shape[0]}")
    offset = params.offset[None, :, None, None]
    scale = params.scale[None, :, None, None]
    out = (data.astype(np.float32, copy=False) - offset) * scale
    np.clip(out, 0.0, float(params.i_max), out=out)
    return np.rint(out).astype(np.uint16, copy=False).reshape(-1, *data.shape[-2:])


def _write_qc_plot(path: Path, *, histograms: Sequence[tuple[np.ndarray, np.ndarray]], params: ScalingParameters) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for channel, (counts, edges) in enumerate(histograms):
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.plot(centers, counts, label=f"C{channel}")
        ax.axvline(float(params.offset[channel]), linestyle="--", linewidth=0.8)
    ax.set_xlabel("float32 intensity")
    ax.set_ylabel("sample count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
