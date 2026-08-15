from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import typer

from lightsheet_psf import __version__
from lightsheet_psf.beads import detect_beads as detect_beads_in_stack
from lightsheet_psf.beads import write_bead_qc_png
from lightsheet_psf.centroid_qc import choose_spots, render_sheet
from lightsheet_psf.io import load_image_zyx
from lightsheet_psf.median import add_quality_flags, build_medians, write_median_qc_png
from lightsheet_psf.radial import (
    central_fwhm,
    peak_normalize,
    radial_average_xy,
    shear_volume_x,
    sum_normalize,
    z_fwhm_along_xz_shear,
)
from lightsheet_psf.reporting import dependency_versions, file_sha256
from lightsheet_psf.shear import crop_shear_distribution, shear_for_volume


class ShearMode(str, Enum):
    sum_y = "sum_y"
    central_y = "central_y"
    max_y = "max_y"


class MedianProfile(str, Enum):
    june_std = "june-std"
    june_571_zstrict = "june-571-zstrict"


@dataclass(frozen=True)
class MedianDefaults:
    crop_shape_text: str = "21,21,21"
    size_mad_mult: float = 2.0
    z_asym_mad_mult: float = 1.5
    z_near_ratio_floor: float = 0.35
    min_z_fwhm_px: float | None = None
    min_z_support_span_px: float | None = None
    min_z_pre_tail_fraction: float | None = None
    max_central_z_peak_offset_px: float | None = None


MEDIAN_DEFAULTS = {
    MedianProfile.june_std: MedianDefaults(),
    MedianProfile.june_571_zstrict: MedianDefaults(
        min_z_fwhm_px=5.0,
        min_z_support_span_px=9.0,
        min_z_pre_tail_fraction=0.05,
        max_central_z_peak_offset_px=2.0,
    ),
}


app = typer.Typer(no_args_is_help=True)


def parse_spacing_zyx_um(value: str) -> tuple[float, float, float]:
    parts = tuple(float(part) for part in value.split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise typer.BadParameter("expected Z,Y,X as three positive comma-separated floats")
    return parts


def parse_shape_zyx(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise typer.BadParameter("expected Z,Y,X as three positive comma-separated integers")
    return parts


def parse_odd_shape_zyx(value: str) -> tuple[int, int, int]:
    parts = parse_shape_zyx(value)
    if any(part % 2 != 1 for part in parts):
        raise typer.BadParameter("expected Z,Y,X as three odd comma-separated integers")
    return parts


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def read_good_quality_count(path: Path) -> int:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "good_quality" not in reader.fieldnames:
            raise ValueError(f"{path} is missing a good_quality column.")
        return sum(row["good_quality"].strip().lower() in {"1", "true", "yes"} for row in reader)


def provenance(
    input_paths: dict[str, Path], output_paths: dict[str, Path] | None = None
) -> dict[str, object]:
    result: dict[str, object] = {
        "package": {"name": "lightsheet-psf", "version": __version__},
        "dependency_versions": dependency_versions(("numpy", "scipy", "tifffile", "matplotlib", "typer")),
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "input_sha256": {name: file_sha256(path) for name, path in input_paths.items()},
    }
    if output_paths is not None:
        result["output_paths"] = {name: str(path) for name, path in output_paths.items()}
        result["output_sha256"] = {
            name: file_sha256(path)
            for name, path in output_paths.items()
            if path.exists() and name != "report_json"
        }
    return result


def spacing_report(spacing_zyx_um: tuple[float, float, float]) -> dict[str, float]:
    return {
        "dxy_um": spacing_zyx_um[2],
        "dy_um": spacing_zyx_um[1],
        "dx_um": spacing_zyx_um[2],
        "dz_um": spacing_zyx_um[0],
    }


@app.command("detect-beads")
def detect_beads(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    qc: Annotated[Path, typer.Option("--qc")],
    channel: Annotated[int | None, typer.Option("--channel", min=0)] = None,
    z_start: Annotated[int, typer.Option("--z-start", min=0)] = 0,
    fwhm: Annotated[float, typer.Option("--fwhm", min=0.0)] = 1.5,
    threshold_sigma: Annotated[float, typer.Option("--threshold-sigma", min=0.0)] = 5.0,
    brightest: Annotated[int | None, typer.Option("--brightest", min=1)] = None,
    xy_radius: Annotated[int, typer.Option("--xy-radius", min=0)] = 2,
    z_radius: Annotated[int, typer.Option("--z-radius", min=0)] = 2,
) -> None:
    stack = load_image_zyx(image, channel=channel, z_start=z_start)
    beads = detect_beads_in_stack(
        stack,
        fwhm=fwhm,
        threshold_sigma=threshold_sigma,
        brightest=brightest,
        xy_radius=xy_radius,
        z_radius=z_radius,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    qc.parent.mkdir(parents=True, exist_ok=True)
    beads.to_csv(output, index=False)
    write_bead_qc_png(stack, beads, qc)
    typer.echo(f"stack_shape_zyx={tuple(int(v) for v in stack.shape)}")
    typer.echo(f"z_start={z_start}")
    typer.echo(f"beads={len(beads)}")
    typer.echo(f"csv={output}")
    typer.echo(f"qc={qc}")


@app.command("make-median")
def make_median(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    centers: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    prefix: Annotated[Path, typer.Option("--prefix")],
    channel: Annotated[int | None, typer.Option("--channel", min=0)] = None,
    z_start: Annotated[int, typer.Option("--z-start", min=0)] = 0,
    profile: Annotated[MedianProfile, typer.Option("--profile")] = MedianProfile.june_std,
    crop_shape_text: Annotated[str | None, typer.Option("--crop-shape", metavar="Z,Y,X")] = None,
    min_xy_distance: Annotated[float, typer.Option("--min-xy-distance", min=0.0)] = 12.0,
    size_mad_mult: Annotated[float | None, typer.Option("--size-mad-mult", min=0.0)] = None,
    z_asym_mad_mult: Annotated[float | None, typer.Option("--z-asym-mad-mult", min=0.0)] = None,
    z_near_ratio_floor: Annotated[float | None, typer.Option("--z-near-ratio-floor", min=0.0)] = None,
    require_full_crop: Annotated[bool, typer.Option("--require-full-crop/--allow-padded-crop")] = False,
    min_z_fwhm_px: Annotated[float | None, typer.Option("--min-z-fwhm-px", min=0.0)] = None,
    max_z_fwhm_px: Annotated[float | None, typer.Option("--max-z-fwhm-px", min=0.0)] = None,
    min_z_support_span_px: Annotated[float | None, typer.Option("--min-z-support-span-px", min=0.0)] = None,
    max_z_support_span_px: Annotated[float | None, typer.Option("--max-z-support-span-px", min=0.0)] = None,
    min_z_pre_tail_fraction: Annotated[
        float | None, typer.Option("--min-z-pre-tail-fraction", min=0.0)
    ] = None,
    max_central_z_peak_offset_px: Annotated[
        float | None, typer.Option("--max-central-z-peak-offset-px", min=0.0)
    ] = None,
) -> None:
    defaults = MEDIAN_DEFAULTS[profile]
    crop_shape = parse_odd_shape_zyx(crop_shape_text or defaults.crop_shape_text)
    resolved_size_mad_mult = defaults.size_mad_mult if size_mad_mult is None else size_mad_mult
    resolved_z_asym_mad_mult = defaults.z_asym_mad_mult if z_asym_mad_mult is None else z_asym_mad_mult
    resolved_z_near_ratio_floor = (
        defaults.z_near_ratio_floor if z_near_ratio_floor is None else z_near_ratio_floor
    )
    resolved_min_z_fwhm_px = defaults.min_z_fwhm_px if min_z_fwhm_px is None else min_z_fwhm_px
    resolved_min_z_support_span_px = (
        defaults.min_z_support_span_px if min_z_support_span_px is None else min_z_support_span_px
    )
    resolved_min_z_pre_tail_fraction = (
        defaults.min_z_pre_tail_fraction
        if min_z_pre_tail_fraction is None
        else min_z_pre_tail_fraction
    )
    resolved_max_central_z_peak_offset_px = (
        defaults.max_central_z_peak_offset_px
        if max_central_z_peak_offset_px is None
        else max_central_z_peak_offset_px
    )
    stack = load_image_zyx(image, channel=channel, z_start=z_start)
    centers_df = pd.read_csv(centers)
    quality = add_quality_flags(centers_df, stack.shape, crop_shape, min_xy_distance)
    crops, raw_median, norm_median = build_medians(
        stack,
        quality,
        crop_shape,
        size_mad_mult=resolved_size_mad_mult,
        z_asym_mad_mult=resolved_z_asym_mad_mult,
        z_near_ratio_floor=resolved_z_near_ratio_floor,
        require_full_crop=require_full_crop,
        min_z_fwhm_px=resolved_min_z_fwhm_px,
        max_z_fwhm_px=max_z_fwhm_px,
        min_z_support_span_px=resolved_min_z_support_span_px,
        max_z_support_span_px=max_z_support_span_px,
        min_z_pre_tail_fraction=resolved_min_z_pre_tail_fraction,
        max_central_z_peak_offset_px=resolved_max_central_z_peak_offset_px,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    quality_path = prefix.with_name(f"{prefix.name}_quality.csv")
    raw_tif = prefix.with_name(f"{prefix.name}_raw.tif")
    norm_tif = prefix.with_name(f"{prefix.name}_peaknorm.tif")
    crops_npy = prefix.with_name(f"{prefix.name}_crops_peaknorm.npy")
    qc_png = prefix.with_name(f"{prefix.name}_qc.png")
    quality.to_csv(quality_path, index=False)
    tifffile.imwrite(raw_tif, raw_median, metadata={"axes": "ZYX"})
    tifffile.imwrite(norm_tif, norm_median, metadata={"axes": "ZYX"})
    crop_peaks = np.nanmax(crops, axis=(1, 2, 3), keepdims=True)
    np.save(crops_npy, crops / np.maximum(crop_peaks, 1))
    write_median_qc_png(quality, raw_median, norm_median, qc_png)
    typer.echo(f"stack_shape_zyx={tuple(int(v) for v in stack.shape)}")
    typer.echo(f"z_start={z_start}")
    typer.echo(f"profile={profile.value}")
    typer.echo(f"crop_shape_zyx={crop_shape}")
    typer.echo(f"candidate_beads={len(quality)}")
    typer.echo(f"good_quality_beads={int(quality['good_quality'].sum())}")
    typer.echo(f"size_consistent_beads={int(quality['size_consistent'].sum())}")
    typer.echo(f"z_asym_ok_beads={int(quality['z_asym_ok'].sum())}")
    typer.echo(f"used_crops={crops.shape[0]}")
    typer.echo(f"quality_csv={quality_path}")
    typer.echo(f"raw_median_tif={raw_tif}")
    typer.echo(f"peaknorm_median_tif={norm_tif}")
    typer.echo(f"peaknorm_crops_npy={crops_npy}")
    typer.echo(f"qc_png={qc_png}")


@app.command("render-centroid-qc")
def render_centroid_qc(
    image: Annotated[Path, typer.Option("--image", exists=True, dir_okay=False, readable=True)],
    quality_csv: Annotated[Path, typer.Option("--quality-csv", exists=True, dir_okay=False, readable=True)],
    output_png: Annotated[Path, typer.Option("--output-png")],
    output_csv: Annotated[Path, typer.Option("--output-csv")],
    channel: Annotated[int | None, typer.Option("--channel", min=0)] = None,
    z_start: Annotated[int, typer.Option("--z-start", min=0)] = 0,
    crop_shape_text: Annotated[str, typer.Option("--crop-shape", metavar="Z,Y,X")] = "21,21,21",
    require_full_crop: Annotated[bool, typer.Option("--require-full-crop/--allow-edge-crop")] = False,
) -> None:
    crop_shape = parse_odd_shape_zyx(crop_shape_text)
    df = pd.read_csv(quality_csv)
    if require_full_crop:
        df = df[df["full_crop"]].copy()
    stack = load_image_zyx(image, channel=channel, z_start=z_start)
    selected = choose_spots(df, crop_shape, stack.shape)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)
    render_sheet(stack, selected, crop_shape, output_png)
    typer.echo(f"selected_spots={len(selected)}")
    typer.echo(f"z_start={z_start}")
    typer.echo(f"output_png={output_png}")
    typer.echo(f"output_csv={output_csv}")


@app.command("weighted-average")
def weighted_average(
    inputs: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    report: Annotated[Path, typer.Option("--report")],
    weights_text: Annotated[str, typer.Option("--weights", metavar="W1,W2,...")],
) -> None:
    """Combine co-registered PSFs after normalizing each input to unit mass."""
    if len(inputs) < 2:
        raise typer.BadParameter("weighted-average requires at least two input PSFs")
    try:
        weights = np.asarray([float(value) for value in weights_text.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise typer.BadParameter("--weights must be comma-separated numbers") from exc
    if len(weights) != len(inputs):
        raise typer.BadParameter(
            f"--weights has {len(weights)} values but {len(inputs)} input PSFs were provided"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0) or float(weights.sum()) <= 0:
        raise typer.BadParameter("--weights must be finite, non-negative, and have a positive sum")

    volumes = [np.asarray(tifffile.imread(path), dtype=np.float64) for path in inputs]
    if any(volume.ndim != 3 for volume in volumes):
        raise typer.BadParameter("all input PSFs must be 3D ZYX TIFFs")
    shapes = {volume.shape for volume in volumes}
    if len(shapes) != 1:
        raise typer.BadParameter(f"all input PSFs must have the same shape, got {sorted(shapes)}")

    normalized_volumes = []
    for path, volume in zip(inputs, volumes, strict=True):
        if not np.isfinite(volume).all():
            raise typer.BadParameter(f"input PSF contains non-finite values: {path}")
        if np.any(volume < 0):
            raise typer.BadParameter(f"input PSF contains negative values: {path}")
        mass = float(volume.sum())
        if mass <= 0:
            raise typer.BadParameter(f"input PSF must have positive mass: {path}")
        normalized_volumes.append(volume / mass)

    normalized_weights = weights / float(weights.sum())
    combined = np.tensordot(
        normalized_weights, np.stack(normalized_volumes, axis=0), axes=(0, 0)
    )
    combined /= float(combined.sum())
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, combined.astype(np.float32), metadata={"axes": "ZYX"})

    input_paths = {f"psf_{index}": path for index, path in enumerate(inputs)}
    output_paths = {"weighted_tif": output, "report_json": report}
    payload: dict[str, object] = {
        **provenance(input_paths, output_paths),
        "command": "weighted-average",
        "input_psfs": [str(path) for path in inputs],
        "raw_weights": weights.tolist(),
        "normalized_weights": normalized_weights.tolist(),
        "shape_zyx": list(combined.shape),
        "output_sum": float(combined.sum()),
        "output_tif": str(output),
        "report_json": str(report),
    }
    write_json(report, payload)
    typer.echo(json.dumps(payload, indent=2))


def write_tif_bundle(out_dir: Path, name: str, volume: np.ndarray, raw_peak: float) -> dict[str, Path]:
    peak = peak_normalize(np.clip(volume, 0, None)).astype(np.float32)
    sumnorm = sum_normalize(peak).astype(np.float32)
    rawscale = (peak * raw_peak).astype(np.float32)
    paths = {
        "peaknorm": out_dir / f"{name}_peaknorm.tif",
        "sumnorm": out_dir / f"{name}_sumnorm.tif",
        "rawscale": out_dir / f"{name}_rawscale.tif",
    }
    for path, image in (
        (paths["peaknorm"], peak),
        (paths["sumnorm"], sumnorm),
        (paths["rawscale"], rawscale),
    ):
        tifffile.imwrite(path, image, metadata={"axes": "ZYX"})
    return paths


def flatten_tif_bundle_paths(prefix: str, paths: dict[str, Path]) -> dict[str, Path]:
    return {f"{prefix}_{name}": path for name, path in paths.items()}


def plot_shear(
    median: np.ndarray,
    median_metrics: dict[str, object],
    crop_slopes_um: np.ndarray | None,
    out_png: Path,
) -> None:
    z = np.arange(median.shape[0], dtype=np.float64)
    x_centers = np.array(
        [np.nan if v is None else float(v) for v in median_metrics["x_centers_px"]],
        dtype=np.float64,
    )
    slope = float(median_metrics["slope_x_px_per_z_plane"])
    intercept = float(median_metrics["intercept_x_px"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180, constrained_layout=True)
    axes[0].imshow(np.nanmax(median, axis=1), cmap="magma", origin="upper", aspect="auto")
    axes[0].plot(x_centers, z, "co", markersize=3, label="x centroid")
    axes[0].plot(intercept + slope * z, z, "w-", linewidth=1.2, label="fit")
    axes[0].invert_yaxis()
    axes[0].set_title("Median XZ max projection")
    axes[0].set_xlabel("x px")
    axes[0].set_ylabel("z plane")
    axes[0].legend(fontsize=7, loc="lower right")

    if crop_slopes_um is not None:
        finite_slopes = crop_slopes_um[np.isfinite(crop_slopes_um)]
        axes[1].hist(finite_slopes, bins=40, color="0.25")
    axes[1].axvline(float(median_metrics["slope_x_um_per_z_um"]), color="tab:red", linewidth=1.5)
    axes[1].set_title("Crop shear distribution")
    axes[1].set_xlabel("dx/dz (um/um)")
    axes[1].set_ylabel("count")

    fig.suptitle(
        f"Median shear = {float(median_metrics['slope_x_um_per_z_um']):+.3f} um/um, "
        f"{float(median_metrics['angle_deg_from_z_axis']):+.2f} deg",
        fontsize=10,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def write_radial_qc(
    source: np.ndarray,
    deskewed: np.ndarray,
    radial_deskewed: np.ndarray,
    radial_reskewed: np.ndarray,
    out_png: Path,
) -> None:
    volumes = [
        ("source", peak_normalize(source)),
        ("deskewed", peak_normalize(deskewed)),
        ("radial deskewed", peak_normalize(radial_deskewed)),
        ("radial reskewed", peak_normalize(radial_reskewed)),
    ]
    zc, yc, xc = (size // 2 for size in source.shape)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=180, constrained_layout=True)
    for col, (title, vol) in enumerate(volumes):
        axes[0, col].imshow(vol[zc], cmap="magma", vmin=0, vmax=1)
        axes[0, col].set_title(f"{title} XY")
        axes[1, col].imshow(vol[:, yc, :], cmap="magma", vmin=0, vmax=1, aspect="auto")
        axes[1, col].set_title(f"{title} XZ")
        axes[2, col].imshow(vol[:, :, xc], cmap="magma", vmin=0, vmax=1, aspect="auto")
        axes[2, col].set_title(f"{title} YZ")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_png)
    plt.close(fig)


def write_profile_qc(
    source: np.ndarray,
    deskewed: np.ndarray,
    radial_deskewed: np.ndarray,
    radial_reskewed: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    out_png: Path,
) -> None:
    volumes = {
        "source": peak_normalize(source),
        "deskewed": peak_normalize(deskewed),
        "radial deskewed": peak_normalize(radial_deskewed),
        "radial reskewed": peak_normalize(radial_reskewed),
    }
    zc, yc, xc = (size // 2 for size in source.shape)
    coords = {
        "x": (np.arange(source.shape[2]) - xc) * spacing_zyx_um[2],
        "y": (np.arange(source.shape[1]) - yc) * spacing_zyx_um[1],
        "z": (np.arange(source.shape[0]) - zc) * spacing_zyx_um[0],
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=180, constrained_layout=True)
    for ax, axis in zip(axes, ("x", "y", "z")):
        for label, volume in volumes.items():
            if axis == "x":
                profile = volume[zc, yc, :]
            elif axis == "y":
                profile = volume[zc, :, xc]
            else:
                profile = volume[:, yc, xc]
            ax.plot(coords[axis], profile, "o-", label=label, markersize=3)
        ax.axhline(0.5, color="0.6", linewidth=1, linestyle="--")
        ax.set_title(f"{axis.upper()} profile")
        ax.set_xlabel("um")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("peak-normalized intensity")
    axes[0].legend(fontsize=7)
    fig.savefig(out_png)
    plt.close(fig)


@app.command("measure-shear")
def measure_shear(
    median_psf: Annotated[Path, typer.Option("--median-psf", exists=True, dir_okay=False, readable=True)],
    output_json: Annotated[Path, typer.Option("--output-json")],
    output_png: Annotated[Path, typer.Option("--output-png")],
    spacing_zyx_um_text: Annotated[str, typer.Option("--spacing-zyx-um", metavar="Z,Y,X")],
    crops_npy: Annotated[
        Path | None, typer.Option("--crops-npy", exists=True, dir_okay=False, readable=True)
    ] = None,
    quality_csv: Annotated[
        Path | None, typer.Option("--quality-csv", exists=True, dir_okay=False, readable=True)
    ] = None,
    z_fraction_threshold: Annotated[float, typer.Option("--z-fraction-threshold", min=0.0, max=1.0)] = 0.20,
    mode: Annotated[ShearMode, typer.Option("--mode")] = ShearMode.sum_y,
) -> None:
    spacing_zyx_um = parse_spacing_zyx_um(spacing_zyx_um_text)
    median = np.asarray(tifffile.imread(median_psf), dtype=np.float64)
    metrics = shear_for_volume(
        median,
        spacing_zyx_um=spacing_zyx_um,
        z_fraction_threshold=z_fraction_threshold,
        mode=mode.value,
    )

    input_paths = {"median_psf": median_psf}
    crop_distribution = None
    crop_slopes_arr = None
    if crops_npy is not None:
        crops = np.asarray(np.load(crops_npy), dtype=np.float64)
        input_paths["crops_npy"] = crops_npy
        crop_distribution = crop_shear_distribution(
            crops,
            spacing_zyx_um=spacing_zyx_um,
            z_fraction_threshold=z_fraction_threshold,
            mode=mode.value,
        )
        crop_slopes_arr = np.asarray(
            [
                float(
                    shear_for_volume(
                        crop,
                        spacing_zyx_um=spacing_zyx_um,
                        z_fraction_threshold=z_fraction_threshold,
                        mode=mode.value,
                    )["slope_x_um_per_z_um"]
                )
                for crop in crops
            ],
            dtype=np.float64,
        )

    good_quality_count = None
    if quality_csv is not None:
        input_paths["quality_csv"] = quality_csv
        good_quality_count = read_good_quality_count(quality_csv)

    plot_shear(median, metrics, crop_slopes_arr, output_png)
    output_paths = {"report_json": output_json, "qc_png": output_png}

    report: dict[str, object] = {
        **provenance(input_paths, output_paths),
        "median_psf": str(median_psf),
        "crops_npy": None if crops_npy is None else str(crops_npy),
        "quality_csv": None if quality_csv is None else str(quality_csv),
        **spacing_report(spacing_zyx_um),
        "mode": mode.value,
        "z_fraction_threshold": z_fraction_threshold,
        "good_quality_count": good_quality_count,
        "command": "measure-shear",
        "command_args": {
            "spacing_zyx_um": list(spacing_zyx_um),
            "z_fraction_threshold": z_fraction_threshold,
            "mode": mode.value,
        },
        "shape_zyx": list(median.shape),
        "median_psf_shear": metrics,
        "accepted_crop_shear_distribution": crop_distribution,
        "outputs": {
            "report_json": str(output_json),
            "qc_png": str(output_png),
        },
    }
    write_json(output_json, report)
    typer.echo(json.dumps(report, indent=2, allow_nan=True))


@app.command("radialize")
def radialize(
    peak_psf: Annotated[Path, typer.Option("--peak-psf", exists=True, dir_okay=False, readable=True)],
    raw_psf: Annotated[Path, typer.Option("--raw-psf", exists=True, dir_okay=False, readable=True)],
    out_dir: Annotated[Path, typer.Option("--out-dir", file_okay=False)],
    spacing_zyx_um_text: Annotated[str, typer.Option("--spacing-zyx-um", metavar="Z,Y,X")],
    shear_slope_px_per_z: Annotated[float | None, typer.Option("--shear-slope-px-per-z")] = None,
    shear_mode: Annotated[ShearMode, typer.Option("--shear-mode")] = ShearMode.central_y,
    z_fraction_threshold: Annotated[float, typer.Option("--z-fraction-threshold", min=0.0, max=1.0)] = 0.20,
) -> None:
    spacing_zyx_um = parse_spacing_zyx_um(spacing_zyx_um_text)
    source = np.asarray(tifffile.imread(peak_psf), dtype=np.float64)
    raw = np.asarray(tifffile.imread(raw_psf), dtype=np.float64)
    if source.ndim != 3:
        raise typer.BadParameter(f"--peak-psf must be a 3D ZYX TIFF, got shape {source.shape}")
    if raw.ndim != 3:
        raise typer.BadParameter(f"--raw-psf must be a 3D ZYX TIFF, got shape {raw.shape}")
    if not np.any(np.isfinite(raw)):
        raise typer.BadParameter("--raw-psf must have a positive finite peak")
    raw_peak = float(np.nanmax(raw))
    if not np.isfinite(raw_peak) or raw_peak <= 0:
        raise typer.BadParameter("--raw-psf must have a positive finite peak")

    out_dir.mkdir(parents=True, exist_ok=True)
    if shear_slope_px_per_z is None:
        shear_metrics = shear_for_volume(
            source,
            spacing_zyx_um=spacing_zyx_um,
            z_fraction_threshold=z_fraction_threshold,
            mode=shear_mode.value,
        )
        resolved_shear_slope_px_per_z = float(shear_metrics["slope_x_px_per_z_plane"])
    else:
        shear_metrics = None
        resolved_shear_slope_px_per_z = shear_slope_px_per_z

    deskewed = shear_volume_x(source, resolved_shear_slope_px_per_z, inverse=False)
    radial_deskewed = radial_average_xy(deskewed)
    radial_reskewed = shear_volume_x(radial_deskewed, resolved_shear_slope_px_per_z, inverse=True)

    deskewed_paths = write_tif_bundle(out_dir, "deskewed_source", deskewed, raw_peak)
    radial_deskewed_paths = write_tif_bundle(out_dir, "radial_symmetric_deskewed", radial_deskewed, raw_peak)
    radial_reskewed_paths = write_tif_bundle(out_dir, "radial_symmetric_reskewed", radial_reskewed, raw_peak)
    qc_png = out_dir / "deskew_radial_reskew_qc.png"
    profiles_png = out_dir / "deskew_radial_reskew_profiles.png"
    report_json = out_dir / "deskew_radial_reskew_report.json"
    write_radial_qc(source, deskewed, radial_deskewed, radial_reskewed, qc_png)
    write_profile_qc(source, deskewed, radial_deskewed, radial_reskewed, spacing_zyx_um, profiles_png)

    nested_outputs = {
        "deskewed": {name: str(path) for name, path in deskewed_paths.items()},
        "radial_deskewed": {name: str(path) for name, path in radial_deskewed_paths.items()},
        "radial_reskewed": {name: str(path) for name, path in radial_reskewed_paths.items()},
        "qc_png": str(qc_png),
        "profiles_png": str(profiles_png),
        "report_json": str(report_json),
    }
    output_paths = {
        **flatten_tif_bundle_paths("deskewed", deskewed_paths),
        **flatten_tif_bundle_paths("radial_deskewed", radial_deskewed_paths),
        **flatten_tif_bundle_paths("radial_reskewed", radial_reskewed_paths),
        "qc_png": qc_png,
        "profiles_png": profiles_png,
        "report_json": report_json,
    }
    input_paths = {"peak_psf": peak_psf, "raw_psf": raw_psf}
    report: dict[str, object] = {
        **provenance(input_paths, output_paths),
        "source_peak_psf": str(peak_psf),
        "source_raw_psf": str(raw_psf),
        **spacing_report(spacing_zyx_um),
        "shear_mode": shear_mode.value,
        "z_fraction_threshold": z_fraction_threshold,
        "command": "radialize",
        "command_args": {
            "spacing_zyx_um": list(spacing_zyx_um),
            "shear_slope_px_per_z": shear_slope_px_per_z,
            "shear_mode": shear_mode.value,
            "z_fraction_threshold": z_fraction_threshold,
        },
        "shape_zyx": list(source.shape),
        "input_shear_metrics": shear_metrics,
        "shear_slope_x_px_per_z_plane": resolved_shear_slope_px_per_z,
        "shear_slope_x_um_per_z_um": resolved_shear_slope_px_per_z * spacing_zyx_um[2] / spacing_zyx_um[0],
        "deskew_shift_rule": "x_shift_px = -slope_x_px_per_z_plane * (z - z_center)",
        "reskew_shift_rule": "x_shift_px = +slope_x_px_per_z_plane * (z - z_center)",
        "radial_profile": "exact-radius XY average with monotonic radial falloff",
        "fwhm_um": {
            "source": central_fwhm(peak_normalize(source), spacing_zyx_um=spacing_zyx_um),
            "deskewed": central_fwhm(peak_normalize(deskewed), spacing_zyx_um=spacing_zyx_um),
            "radial_deskewed": central_fwhm(peak_normalize(radial_deskewed), spacing_zyx_um=spacing_zyx_um),
            "radial_reskewed": central_fwhm(peak_normalize(radial_reskewed), spacing_zyx_um=spacing_zyx_um),
        },
        "z_fwhm_along_xz_shear_um": {
            "source": z_fwhm_along_xz_shear(
                peak_normalize(source),
                resolved_shear_slope_px_per_z,
                spacing_z_um=spacing_zyx_um[0],
            ),
            "deskewed": z_fwhm_along_xz_shear(peak_normalize(deskewed), 0.0, spacing_z_um=spacing_zyx_um[0]),
            "radial_deskewed": z_fwhm_along_xz_shear(
                peak_normalize(radial_deskewed),
                0.0,
                spacing_z_um=spacing_zyx_um[0],
            ),
            "radial_reskewed": z_fwhm_along_xz_shear(
                peak_normalize(radial_reskewed),
                resolved_shear_slope_px_per_z,
                spacing_z_um=spacing_zyx_um[0],
            ),
        },
        "sum_normalized_sums": {
            "deskewed": float(np.sum(tifffile.imread(deskewed_paths["sumnorm"]))),
            "radial_deskewed": float(np.sum(tifffile.imread(radial_deskewed_paths["sumnorm"]))),
            "radial_reskewed": float(np.sum(tifffile.imread(radial_reskewed_paths["sumnorm"]))),
        },
        "outputs": nested_outputs,
    }
    write_json(report_json, report)
    typer.echo(json.dumps(report, indent=2, allow_nan=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
