from pathlib import Path
from typing import Annotated

import typer

from squisher.compression import compress_czi_to_ome_tiff, verify_czi_ome_tiff_outputs


app = typer.Typer(no_args_is_help=True)


@app.callback()
def callback() -> None:
    pass


@app.command()
def compress(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    level: Annotated[float, typer.Option("--level", min=0.0, max=100.0)] = 90,
    tile_size: Annotated[int, typer.Option("--tile-size")] = 512,
    tiff_maxworkers: Annotated[int, typer.Option("--tiff-maxworkers")] = 4,
    czi_tile_workers: Annotated[int, typer.Option("--czi-tile-workers")] = 1,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = False,
) -> None:
    compress_czi_to_ome_tiff(
        path,
        level=level,
        tile_size=tile_size,
        maxworkers=tiff_maxworkers,
        tile_workers=czi_tile_workers,
        resume=resume,
    )


@app.command()
def verify(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    decode_samples: Annotated[bool, typer.Option("--decode-samples/--no-decode-samples")] = False,
) -> None:
    verify_czi_ome_tiff_outputs(path, decode_samples=decode_samples)


def main() -> None:
    app()
