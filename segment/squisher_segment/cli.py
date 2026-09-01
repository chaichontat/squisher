from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from squisher_segment.segment.extract import run_extract


app = typer.Typer(no_args_is_help=True)
segment_app = typer.Typer(no_args_is_help=True)
postproc_app = typer.Typer(no_args_is_help=True)


@app.command("train")
def train(
    path: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Training-data root.")],
    name: Annotated[str, typer.Argument(help="Model config stem under PATH/models.")],
    packed: Annotated[
        bool,
        typer.Option("--packed/--no-packed", help="Enable packed-stripe training."),
    ] = False,
    skip_trt: Annotated[
        bool,
        typer.Option("--skip-trt", help="Skip TensorRT engine generation after training."),
    ] = False,
) -> None:
    """Train Cellpose from ``PATH/models/NAME.json``."""
    from squisher_segment.segment.train import TrainConfig, run_train

    models_path = path / "models"
    if not models_path.is_dir():
        raise typer.BadParameter(f"Models path {models_path} does not exist.", param_hint="path")

    config_path = models_path / f"{name}.json"
    if not config_path.is_file():
        raise typer.BadParameter(f"Config file {config_path} does not exist.", param_hint="name")

    config_text = config_path.read_text()
    train_config = TrainConfig.model_validate_json(
        "\n".join(line for line in config_text.splitlines() if not line.lstrip().startswith("//"))
        + ("\n" if config_text.endswith("\n") else "")
    )
    train_config = train_config.model_copy(
        update={
            "packed": packed,
            "skip_trt": train_config.skip_trt or skip_trt,
        }
    )

    updated = run_train(name, path, train_config)
    (models_path / f"{name}.trained.json").write_text(updated.model_dump_json(indent=2))


@app.command("n4")
def n4_correct(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            help="Channel-separated Lightsheet OME-Zarr input.",
        ),
    ],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Corrected OME-Zarr output.")] = None,
    field_level: Annotated[
        int | None,
        typer.Option(min=0, help="Pyramid level used to estimate the 3D field; defaults to coarsest."),
    ] = None,
    shrink: Annotated[int, typer.Option(min=1, help="N4 fitting shrink factor.")] = 4,
    spline_lowres_px: Annotated[
        tuple[float, float, float],
        typer.Option(
            min=0.000001,
            help="B-spline spacing in shrink-downsampled level-0 Z Y X pixels.",
        ),
    ] = (24.0, 48.0, 48.0),
    threshold: Annotated[
        str | None,
        typer.Option(help="Numeric foreground threshold; defaults to values greater than zero."),
    ] = None,
    unsharp: Annotated[
        bool,
        typer.Option("--unsharp/--no-unsharp", help="Apply optional radius-3 unsharp masking."),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Replace an existing completed output."),
    ] = False,
) -> None:
    """Apply 3D N4 correction to a channel-separated Lightsheet OME-Zarr."""
    from squisher_segment.segment import n4

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    try:
        parsed_threshold = n4._parse_threshold(threshold)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--threshold") from exc
    resolved_output = output or n4._default_n4_output_path(input_path)
    config = n4.N4Config(
        field_level=field_level,
        shrink=shrink,
        spline_lowres_px_zyx=spline_lowres_px,
        threshold=parsed_threshold,
        unsharp=unsharp,
    )
    written = n4.run_n4_ome_zarr(
        input_path,
        resolved_output,
        config,
        overwrite=overwrite,
    )
    typer.echo(written)


@segment_app.command("run")
def segment_run(
    input_zarr: Annotated[
        Path,
        typer.Argument(exists=True, help="Input ZYXC Zarr or registered-source JSON manifest."),
    ],
    channels: Annotated[str | None, typer.Option(help="Comma-separated channel names to segment.")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite", help="Overwrite existing segmentation.")] = False,
    config_path: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Config JSON path.")] = None,
    workers_per_gpu: Annotated[int, typer.Option(help="Workers to spawn per GPU.")] = 4,
    threads_per_worker: Annotated[int, typer.Option(help="Threads per worker.")] = 1,
    use_localcuda: Annotated[bool, typer.Option("--use-localcuda/--no-use-localcuda", help="Use dask-cuda LocalCUDACluster when workers_per_gpu<=1.")] = False,
    n_workers: Annotated[int | None, typer.Option(help="LocalCUDACluster worker count.")] = None,
    target_nz: Annotated[int | None, typer.Option(help="Desired internal Cellpose nz tiles.")] = None,
    target_ny: Annotated[int | None, typer.Option(help="Desired internal Cellpose ny tiles.")] = None,
    target_nx: Annotated[int | None, typer.Option(help="Desired internal Cellpose nx tiles.")] = None,
    nonempty_threshold: Annotated[
        int,
        typer.Option(
            min=0,
            help="561 threshold for block scheduling and Cellpose input masking; values must be strictly greater.",
        ),
    ] = 1000,
    cellpose_only: Annotated[bool, typer.Option("--cellpose-only/--no-cellpose-only", help="Stop after Cellpose inference.")] = False,
    stagger_seconds: Annotated[float, typer.Option(help="Seconds to stagger worker starts on one GPU.")] = 5.0,
) -> None:
    from squisher_segment.segmentation.distributed import distributed_segmentation as segment_mod

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("cellpose").setLevel(logging.WARNING)
    segment_mod._run_single_input(
        input_path=input_zarr,
        channels=channels,
        overwrite=overwrite,
        config_path=config_path,
        workers_per_gpu=workers_per_gpu,
        threads_per_worker=threads_per_worker,
        use_localcuda=use_localcuda,
        n_workers=n_workers,
        target_nz=target_nz,
        target_ny=target_ny,
        target_nx=target_nx,
        nonempty_threshold=nonempty_threshold,
        cellpose_only=cellpose_only,
        stagger_seconds=stagger_seconds,
    )


@segment_app.command("stitch")
def segment_stitch(
    temp_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Cellpose temp directory.")],
    output_path: Annotated[Path, typer.Argument(help="Output segmentation .zarr path.")],
    cleanup: Annotated[bool, typer.Option("--cleanup/--no-cleanup", help="Remove temp directory after stitching.")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite", help="Overwrite existing output.")] = False,
) -> None:
    from squisher_segment.segmentation.distributed import distributed_segmentation as segment_mod

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    segment_mod._run_stitch(temp_dir, output_path, cleanup=cleanup, overwrite=overwrite)


@postproc_app.command("run")
def postproc_run(
    input_zarr_path: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Input segmentation .zarr.")],
    output_path: Annotated[Path | None, typer.Option(help="Output postprocessed .zarr path.")] = None,
    blocksize: Annotated[
        tuple[int, int, int] | None,
        typer.Option(help="Optional Z Y X core block size; defaults to input chunks."),
    ] = None,
    sigma: Annotated[str, typer.Option(help="Gaussian smoothing sigma; scalar or 'z,y,x'.")] = "1,2,2",
    v_min: Annotated[int, typer.Option("--v-min", help="Minimum volume for small cell donation.")] = 500,
    margin: Annotated[int, typer.Option(help="Margin parameter; overlap is 2*margin.")] = 30,
    workers_per_gpu: Annotated[int, typer.Option(help="Workers per GPU.")] = 1,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite", help="Overwrite existing output.")] = False,
) -> None:
    import zarr
    from squisher_segment.segmentation.distributed import distributed_postproc as postproc_mod

    input_zarr = zarr.open(input_zarr_path, mode="r")
    resolved_output_path = output_path
    if resolved_output_path is None:
        sigma_str = sigma.replace(",", "-").replace(" ", "")
        resolved_output_path = input_zarr_path.parent / f"{input_zarr_path.stem}_postproc_s{sigma_str}_v{v_min}.zarr"
    postproc_mod.distributed_postproc(
        input_zarr=input_zarr,
        write_path=resolved_output_path,
        blocksize=blocksize,
        margin=margin,
        sigma=postproc_mod._parse_sigma_option(sigma),
        V_min=v_min,
        input_path=input_zarr_path,
        cluster_kwargs={"workers_per_gpu": workers_per_gpu, "threads_per_worker": 1},
        overwrite=overwrite,
    )


@app.command("extract")
def extract(
    input_path: Annotated[Path, typer.Argument(exists=True, help="Input registered TIFF or fused .zarr volume.")],
    mode: Annotated[str, typer.Option(help="Extraction mode: z, ortho, or maxproj.")] = "z",
    out: Annotated[Path | None, typer.Option(help="Output directory.")] = None,
    dz: Annotated[int, typer.Option(help="Use every Nth Z plane for z/maxproj modes.")] = 1,
    n: Annotated[int, typer.Option(help="Number of sampled tiles/slices for ortho and Zarr modes.")] = 50,
    z_crops_per_file: Annotated[int, typer.Option(help="Random XY crops per Z plane for TIFF z mode.")] = 1,
    anisotropy: Annotated[int, typer.Option(help="Z/YX anisotropy for ortho mode.")] = 6,
    ortho_depth: Annotated[
        int | None,
        typer.Option(help="Exact native Z-plane depth for content-sampled Zarr ortho crops."),
    ] = None,
    channels: Annotated[str | None, typer.Option(help="Comma-separated channel indices or names.")] = None,
    crop: Annotated[int, typer.Option(help="Pixels to crop from spatial borders before sampling.")] = 0,
    threads: Annotated[int, typer.Option(help="Worker threads for extraction.")] = 8,
    upscale: Annotated[float | None, typer.Option(help="Spatial output upscale factor.")] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed for sampled candidates.")] = None,
    label: Annotated[str | None, typer.Option(help="Filename prefix; defaults to input stem.")] = None,
    masks: Annotated[Path | None, typer.Option(exists=True, help="Optional matching mask TIFF/Zarr.")] = None,
    enrich_boundaries: Annotated[Path | None, typer.Option(exists=True, help="Optional mask used to bias candidate selection.")] = None,
    aux_channel_stack: Annotated[
        Path | None,
        typer.Option(
            "--aux-channel-stack",
            exists=True,
            help="Optional coordinate-matched OME-TIFF/Zarr stack whose channels are appended to z/ortho outputs.",
        ),
    ] = None,
) -> None:
    run_extract(
        input_path,
        mode=mode,
        out=out,
        dz=dz,
        n=n,
        z_crops_per_file=z_crops_per_file,
        anisotropy=anisotropy,
        ortho_depth=ortho_depth,
        channels=channels,
        crop=crop,
        threads=threads,
        upscale=upscale,
        seed=seed,
        label=label,
        masks=masks,
        enrich_boundaries=enrich_boundaries,
        aux_channel_stack=aux_channel_stack,
    )


@app.command("regionprops")
def regionprops_command(
    labels: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Input 3-D label Zarr."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output Parquet; defaults beside labels."),
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Parallel label-chunk workers.")] = 2,
    offset_zyx: Annotated[
        tuple[int, int, int],
        typer.Option(min=0, help="Global Z Y X offset added to centroids and plane Z."),
    ] = (0, 0, 0),
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse matching completed chunk partials."),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Replace an existing output atomically."),
    ] = False,
) -> None:
    """Measure chunk-parallel region properties without loading the full label volume."""
    from squisher_segment.segment.regionprops import measure_zarr

    resolved_output = output or labels.parent / "props.parquet"
    written = measure_zarr(
        labels,
        resolved_output,
        workers=workers,
        offset_zyx=offset_zyx,
        resume=resume,
        overwrite=overwrite,
    )
    typer.echo(written)


app.add_typer(segment_app, name="segment")
app.add_typer(postproc_app, name="postproc")


def main() -> None:
    app()
