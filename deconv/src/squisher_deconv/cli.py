from __future__ import annotations

from pathlib import Path
from typing import Annotated, Callable

import typer

from squisher_deconv.deconvolution import Deconvolver, ScipyRichardsonLucyDeconvolver, infer_psf_halo
from squisher_deconv.scheduler import parse_devices
from squisher_deconv.streaming import run_streaming_deconv, sample_scale as sample_scale_workflow

app = typer.Typer(no_args_is_help=True)


@app.command("sample-scale")
def sample_scale(
    inputs: Annotated[list[Path], typer.Argument(help="Flattened TIFF inputs.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", file_okay=False)],
    planes: Annotated[int, typer.Option("--planes", min=1)],
    channels: Annotated[int, typer.Option("--channels", min=1)],
    psf: Annotated[Path, typer.Option("--psf", exists=True, dir_okay=False, readable=True)],
    basic: Annotated[
        list[Path] | None,
        typer.Option("--basic", exists=True, dir_okay=False, readable=True, help="BaSiC pickle path; repeat per channel."),
    ] = None,
    engine: Annotated[str, typer.Option("--engine", help="Production default is gpu; scipy is for debugging.")] = "gpu",
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
    deconvolver_factory = _build_deconvolver_factory(
        engine=engine,
        basic=basic,
        channels=channels,
        psf=psf,
    )
    parsed_devices = parse_devices(devices, gpu_auto=engine == "gpu")
    sample_scale_workflow(
        inputs,
        out_dir=out_dir,
        planes=planes,
        channels=channels,
        halo=infer_psf_halo(psf) if halo is None else halo,
        deconvolver=None,
        deconvolver_factory=deconvolver_factory,
        psf_path=psf,
        basic_paths=basic,
        seed=seed,
        p_low=p_low,
        p_high=p_high,
        gamma=gamma,
        bins=bins,
        devices=parsed_devices,
        queue_depth=queue_depth,
        stop_on_error=stop_on_error,
        overwrite=overwrite,
    )


@app.command("run")
def run(
    inputs: Annotated[list[Path], typer.Argument(help="Flattened TIFF inputs.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", file_okay=False)],
    channels: Annotated[int, typer.Option("--channels", min=1)],
    psf: Annotated[Path, typer.Option("--psf", exists=True, dir_okay=False, readable=True)],
    scaling: Annotated[Path | None, typer.Option("--scaling", exists=True, dir_okay=False, readable=True)] = None,
    basic: Annotated[
        list[Path] | None,
        typer.Option("--basic", exists=True, dir_okay=False, readable=True, help="BaSiC pickle path; repeat per channel."),
    ] = None,
    engine: Annotated[str, typer.Option("--engine", help="Production default is gpu; scipy is for debugging.")] = "gpu",
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
    deconvolver_factory = _build_deconvolver_factory(
        engine=engine,
        basic=basic,
        channels=channels,
        psf=psf,
    )
    parsed_devices = parse_devices(devices, gpu_auto=engine == "gpu")
    run_streaming_deconv(
        inputs,
        out_dir=out_dir,
        scaling_path=scaling,
        channels=channels,
        halo=infer_psf_halo(psf) if halo is None else halo,
        slab_depth=slab_depth,
        output_mode=output_mode,
        deconvolver=None,
        deconvolver_factory=deconvolver_factory,
        psf_path=psf,
        basic_paths=basic,
        devices=parsed_devices,
        queue_depth=queue_depth,
        stop_on_error=stop_on_error,
        overwrite=overwrite,
    )


def _build_deconvolver_factory(
    *,
    engine: str,
    basic: list[Path] | None,
    channels: int,
    psf: Path,
) -> Callable[[int], Deconvolver]:
    if engine == "gpu":
        if basic is None or len(basic) != channels:
            raise typer.BadParameter(
                f"--engine gpu requires exactly {channels} --basic profile path(s), got {0 if basic is None else len(basic)}."
            )
        from squisher_deconv.gpu import CupyDeconvolverFactory

        return CupyDeconvolverFactory(
            basic_paths=tuple(Path(path) for path in basic),
            psf_path=psf,
        )
    if engine == "scipy":
        return lambda _device: ScipyRichardsonLucyDeconvolver.from_psf(psf)
    raise typer.BadParameter(f"Unsupported --engine {engine!r}; expected 'gpu' or 'scipy'.")


def main() -> None:
    app()
