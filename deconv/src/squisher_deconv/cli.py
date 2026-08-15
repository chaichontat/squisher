from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer

from squisher_deconv.basic import fit_basic_profiles
from squisher_deconv.deconvolution import infer_psf_halo_many
from squisher_deconv.gpu import CupyDeconvolverFactory
from squisher_deconv.qc import DEFAULT_Z_PLANES, render_before_after_qc
from squisher_deconv.scheduler import parse_devices
from squisher_deconv.streaming import run_streaming_deconv, sample_scale as sample_scale_workflow

app = typer.Typer(no_args_is_help=True)


@app.command("basic")
def basic(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Level-0 OME-TIFF tile inputs.",
        ),
    ],
    out_dir: Annotated[Path, typer.Option("--out-dir", file_okay=False)],
    label: Annotated[str, typer.Option("--label", help="Output profile basename.")],
    channels: Annotated[int, typer.Option("--channels", min=1)],
    samples: Annotated[int, typer.Option("--samples", min=1)] = 500,
    cache_samples_per_channel: Annotated[
        int | None,
        typer.Option("--cache-samples-per-channel", min=1),
    ] = None,
    samples_per_tile: Annotated[
        int,
        typer.Option(
            "--samples-per-tile",
            min=1,
            help="Target planes per sampled TIFF; higher values index fewer source files.",
        ),
    ] = 25,
    blank_slice_sample_stride: Annotated[
        int,
        typer.Option("--blank-slice-sample-stride", min=1),
    ] = 16,
    blank_slice_min_relative_signal: Annotated[
        float,
        typer.Option("--blank-slice-min-relative-signal", min=0.0),
    ] = 0.10,
    blank_slice_min_nonzero_fraction: Annotated[
        float,
        typer.Option("--blank-slice-min-nonzero-fraction", min=0.0, max=1.0),
    ] = 1e-4,
    exclude_blank_slices: Annotated[
        bool,
        typer.Option("--exclude-blank-slices/--include-blank-slices"),
    ] = True,
    exclude_edge_slices: Annotated[
        bool,
        typer.Option("--exclude-edge-slices/--include-edge-slices"),
    ] = True,
    edge_slice_min_profile_jump: Annotated[
        float,
        typer.Option("--edge-slice-min-profile-jump", min=0.0),
    ] = 0.05,
    edge_slice_min_band_delta: Annotated[
        float,
        typer.Option("--edge-slice-min-band-delta", min=0.0),
    ] = 0.35,
    smoothness_flatfield: Annotated[
        float,
        typer.Option("--smoothness-flatfield", min=0.0),
    ] = 1.8,
    fitting_mode: Annotated[str, typer.Option("--fitting-mode")] = "approximate",
    working_size: Annotated[int, typer.Option("--working-size", min=1)] = 128,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    seed: Annotated[int, typer.Option("--seed")] = 20260709,
) -> None:
    """Fit a joint-channel autotuned BaSiC profile with darkfield correction."""
    outputs = fit_basic_profiles(
        inputs=inputs,
        out_dir=out_dir,
        label=label,
        channels=channels,
        samples=samples,
        cache_samples_per_channel=math.ceil(samples / channels)
        if cache_samples_per_channel is None
        else cache_samples_per_channel,
        samples_per_tile=samples_per_tile,
        blank_slice_sample_stride=blank_slice_sample_stride,
        blank_slice_min_relative_signal=blank_slice_min_relative_signal,
        blank_slice_min_nonzero_fraction=blank_slice_min_nonzero_fraction,
        exclude_blank_slices=exclude_blank_slices,
        exclude_edge_slices=exclude_edge_slices,
        edge_slice_min_profile_jump=edge_slice_min_profile_jump,
        edge_slice_min_band_delta=edge_slice_min_band_delta,
        smoothness_flatfield=smoothness_flatfield,
        fitting_mode=fitting_mode,
        working_size=working_size,
        device=device,
        seed=seed,
        progress=typer.echo,
    )
    typer.echo(str(outputs.manifest))
    for profile_path in outputs.profile_paths:
        typer.echo(str(profile_path))


@app.command("sample-scale")
def sample_scale(
    inputs: Annotated[list[Path], typer.Argument(help="Flattened TIFF inputs.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", file_okay=False)],
    channels: Annotated[int, typer.Option("--channels", min=1)],
    psf: Annotated[
        list[Path],
        typer.Option(
            "--psf", exists=True, dir_okay=False, readable=True, help="PSF TIFF path; repeat per channel."
        ),
    ],
    planes: Annotated[int, typer.Option("--planes", min=1)] = 200,
    basic: Annotated[
        list[Path] | None,
        typer.Option(
            "--basic",
            exists=True,
            dir_okay=False,
            readable=True,
            help="BaSiC pickle path; repeat per channel.",
        ),
    ] = None,
    iterations: Annotated[int, typer.Option("--iter", min=1, help="Richardson-Lucy iterations.")] = 1,
    halo: Annotated[int | None, typer.Option("--halo", min=0)] = None,
    seed: Annotated[int, typer.Option("--seed")] = 20260622,
    p_low: Annotated[float, typer.Option("--p-low", min=0.0, max=1.0)] = 0.001,
    p_high: Annotated[float, typer.Option("--p-high", min=0.0, max=1.0)] = 0.99999,
    gamma: Annotated[float, typer.Option("--gamma", min=0.0)] = 1.05,
    bins: Annotated[int, typer.Option("--bins", min=2)] = 8192,
    devices: Annotated[str, typer.Option("--devices")] = "auto",
    queue_depth: Annotated[int, typer.Option("--queue-depth", min=1)] = 2,
    stop_on_error: Annotated[bool, typer.Option("--stop-on-error/--keep-going")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    sample_scale_workflow(
        inputs,
        out_dir=out_dir,
        planes=planes,
        channels=channels,
        halo=infer_psf_halo_many(psf) if halo is None else halo,
        deconvolver=None,
        deconvolver_factory=_build_deconvolver_factory(
            basic=basic,
            channels=channels,
            psfs=psf,
            iterations=iterations,
        ),
        iterations=iterations,
        psf_paths=psf,
        basic_paths=basic,
        seed=seed,
        p_low=p_low,
        p_high=p_high,
        gamma=gamma,
        bins=bins,
        devices=parse_devices(devices, gpu_auto=True),
        queue_depth=queue_depth,
        stop_on_error=stop_on_error,
        overwrite=overwrite,
    )


@app.command("run")
def run(
    inputs: Annotated[list[Path], typer.Argument(help="Flattened TIFF inputs.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", file_okay=False)],
    channels: Annotated[int, typer.Option("--channels", min=1)],
    psf: Annotated[
        list[Path],
        typer.Option(
            "--psf", exists=True, dir_okay=False, readable=True, help="PSF TIFF path; repeat per channel."
        ),
    ],
    scaling: Annotated[
        Path | None, typer.Option("--scaling", exists=True, dir_okay=False, readable=True)
    ] = None,
    basic: Annotated[
        list[Path] | None,
        typer.Option(
            "--basic",
            exists=True,
            dir_okay=False,
            readable=True,
            help="BaSiC pickle path; repeat per channel.",
        ),
    ] = None,
    iterations: Annotated[int, typer.Option("--iter", min=1, help="Richardson-Lucy iterations.")] = 1,
    output_mode: Annotated[str, typer.Option("--output-mode")] = "u16",
    halo: Annotated[int | None, typer.Option("--halo", min=0)] = None,
    slab_depth: Annotated[int, typer.Option("--slab-depth", min=1)] = 16,
    devices: Annotated[str, typer.Option("--devices")] = "auto",
    queue_depth: Annotated[int, typer.Option("--queue-depth", min=1)] = 2,
    stop_on_error: Annotated[bool, typer.Option("--stop-on-error/--keep-going")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
) -> None:
    if output_mode == "u16" and scaling is None:
        raise typer.BadParameter("--output-mode u16 requires --scaling.")
    run_streaming_deconv(
        inputs,
        out_dir=out_dir,
        scaling_path=scaling,
        channels=channels,
        halo=infer_psf_halo_many(psf) if halo is None else halo,
        slab_depth=slab_depth,
        output_mode=output_mode,
        deconvolver=None,
        deconvolver_factory=_build_deconvolver_factory(
            basic=basic,
            channels=channels,
            psfs=psf,
            iterations=iterations,
        ),
        psf_paths=psf,
        basic_paths=basic,
        devices=parse_devices(devices, gpu_auto=True),
        queue_depth=queue_depth,
        stop_on_error=stop_on_error,
        overwrite=overwrite,
        resume=resume,
    )


@app.command("qc")
def qc(
    raw_dir: Annotated[Path, typer.Option("--raw-dir", exists=True, file_okay=False, readable=True)],
    deconv_dir: Annotated[Path | None, typer.Option("--deconv-dir", file_okay=False, readable=True)] = None,
    qc_dir: Annotated[Path | None, typer.Option("--qc-dir", file_okay=False)] = None,
    image_prefix: Annotated[str | None, typer.Option("--image-prefix")] = None,
    z_planes: Annotated[
        list[int],
        typer.Option("--z-plane", min=0, help="Z plane to render; repeat for multiple planes."),
    ] = list(DEFAULT_Z_PLANES),
    tile_count: Annotated[int, typer.Option("--tile-count", min=1)] = 5,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
) -> None:
    """Render raw/deconvolved before-after QC panels for completed decon tiles."""
    manifest = render_before_after_qc(
        raw_dir=raw_dir,
        deconv_dir=deconv_dir,
        qc_dir=qc_dir,
        image_prefix=image_prefix,
        z_planes=z_planes,
        tile_count=tile_count,
        channel=channel,
    )
    output_dir = qc_dir or raw_dir / "squisher-deconv-run-u16-qc"
    typer.echo(str(output_dir / "manifest.json"))
    for item in manifest:
        typer.echo(str(item["png"]))


def _build_deconvolver_factory(
    *,
    basic: list[Path] | None,
    channels: int,
    psfs: list[Path],
    iterations: int,
) -> CupyDeconvolverFactory:
    if len(psfs) != channels:
        raise typer.BadParameter(f"Expected exactly {channels} --psf path(s), got {len(psfs)}.")
    if basic is None or len(basic) != channels:
        raise typer.BadParameter(
            f"Expected exactly {channels} --basic profile path(s), got {0 if basic is None else len(basic)}."
        )
    return CupyDeconvolverFactory(
        basic_paths=tuple(Path(path) for path in basic),
        psf_paths=tuple(Path(path) for path in psfs),
        iterations=iterations,
    )


def main() -> None:
    app()
