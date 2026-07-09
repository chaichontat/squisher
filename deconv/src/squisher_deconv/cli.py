from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from squisher_deconv.deconvolution import infer_psf_halo_many
from squisher_deconv.gpu import CupyDeconvolverFactory
from squisher_deconv.qc import DEFAULT_Z_PLANES, render_before_after_qc
from squisher_deconv.scheduler import parse_devices
from squisher_deconv.streaming import run_streaming_deconv, sample_scale as sample_scale_workflow

app = typer.Typer(no_args_is_help=True)


@app.command("sample-scale")
def sample_scale(
    inputs: Annotated[list[Path], typer.Argument(help="Flattened TIFF inputs.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", file_okay=False)],
    planes: Annotated[int, typer.Option("--planes", min=1)],
    channels: Annotated[int, typer.Option("--channels", min=1)],
    psf: Annotated[
        list[Path],
        typer.Option(
            "--psf", exists=True, dir_okay=False, readable=True, help="PSF TIFF path; repeat per channel."
        ),
    ],
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
