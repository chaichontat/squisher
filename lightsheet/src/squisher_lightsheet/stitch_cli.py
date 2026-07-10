from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from squisher_lightsheet.method8_stitch_register import (
    DEFAULT_FIXED_MASK_THRESHOLD,
    DEFAULT_IMAGE14_POSITION_JSON,
    DEFAULT_IMAGE14_ZARR_DIR,
    register_image14_method8,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def stitch() -> None:
    """Lightsheet stitching workflows."""


@app.command("register")
def register(
    position_json: Annotated[
        Path, typer.Option("--position-json", help="Image_14 metadata position JSON.")
    ] = DEFAULT_IMAGE14_POSITION_JSON,
    zarr_dir: Annotated[
        Path, typer.Option("--zarr-dir", help="Directory containing Image_14 tile OME-Zarrs.")
    ] = DEFAULT_IMAGE14_ZARR_DIR,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory for Method8 summary and optimized position outputs; defaults to a threshold-specific tmp path.",
        ),
    ] = None,
    method8_summary: Annotated[
        Path | None,
        typer.Option(
            "--method8-summary", help="Existing Method8 summary to optimize from; skips Method8 measurement."
        ),
    ] = None,
    method8_output: Annotated[
        Path | None,
        typer.Option(
            "--method8-output", help="Output path for the Method8 z-coverage summary when measuring."
        ),
    ] = None,
    pair: Annotated[
        list[str] | None,
        typer.Option(
            "--pair",
            help="Specific fixed-moving tile pair like 061-062. Repeatable; default is all adjacent.",
        ),
    ] = None,
    z_chunks: Annotated[int, typer.Option("--z-chunks", min=1, help="Number of z-only chunks per seam.")] = 6,
    device: Annotated[int, typer.Option("--device", min=0, help="CUDA device for Method8.")] = 0,
    max_iterations: Annotated[int, typer.Option("--max-iterations", min=1)] = 300,
    ftol: Annotated[float, typer.Option("--ftol")] = 1e-4,
    min_corr: Annotated[
        float, typer.Option("--min-corr", help="Minimum accepted Method8 correlation.")
    ] = 0.15,
    min_grad_ncc: Annotated[
        float,
        typer.Option("--min-grad-ncc", help="Minimum accepted Method8 gradient-component NCC."),
    ] = 0.24,
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Fixed-tile raw intensity threshold used as the Method8 mask.",
        ),
    ] = DEFAULT_FIXED_MASK_THRESHOLD,
    fixed_mask_min_voxels: Annotated[int, typer.Option("--fixed-mask-min-voxels", min=1)] = 256,
    fixed_mask_max_masked_fraction: Annotated[
        float,
        typer.Option("--fixed-mask-max-masked-fraction", min=0.0, max=1.0),
    ] = 0.95,
    max_grad_regression: Annotated[
        float,
        typer.Option(
            "--max-grad-regression", help="Reject Method8 chunks that regress this far behind phase NCC."
        ),
    ] = 0.02,
    max_corr_regression: Annotated[
        float,
        typer.Option(
            "--max-corr-regression",
            help="Reject Method8 chunks that regress this far behind phase correlation.",
        ),
    ] = 0.01,
    phase_fallback: Annotated[
        bool,
        typer.Option(
            "--phase-fallback/--no-phase-fallback",
            help="Use phase correlation for edges without gated Method8 chunks.",
        ),
    ] = True,
    min_phase_grad: Annotated[float, typer.Option("--min-phase-grad")] = 0.24,
    min_phase_corr: Annotated[float, typer.Option("--min-phase-corr")] = 0.15,
    phase_fallback_weight_scale: Annotated[
        float,
        typer.Option(
            "--phase-fallback-weight-scale",
            help="Downweight phase fallback constraints before MVS optimization.",
        ),
    ] = 0.1,
) -> None:
    pairs = tuple(pair or ())
    outputs = register_image14_method8(
        position_json=position_json,
        zarr_dir=zarr_dir,
        output_dir=output_dir,
        method8_summary=method8_summary,
        method8_output=method8_output,
        pairs=pairs or None,
        all_adjacent=not pairs,
        z_chunks=z_chunks,
        device=device,
        max_iterations=max_iterations,
        ftol=ftol,
        min_corr=min_corr,
        min_grad_ncc=min_grad_ncc,
        fixed_mask_threshold=threshold,
        fixed_mask_min_voxels=fixed_mask_min_voxels,
        fixed_mask_max_masked_fraction=fixed_mask_max_masked_fraction,
        max_grad_regression=max_grad_regression,
        max_corr_regression=max_corr_regression,
        phase_fallback=phase_fallback,
        min_phase_grad=min_phase_grad,
        min_phase_corr=min_phase_corr,
        phase_fallback_weight_scale=phase_fallback_weight_scale,
        progress=typer.echo,
    )
    typer.echo(
        json.dumps(
            {
                "method8_summary": str(outputs.method8_summary),
                "optimized_positions": str(outputs.optimized_positions),
                "diagnostics": str(outputs.diagnostics),
                "constraints_jsonl": str(outputs.constraints_jsonl),
                "tile_corrections": str(outputs.tile_corrections),
            },
            indent=2,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
