from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR
from squisher_lightsheet.registration_workflow import run_registration_workflow

app = typer.Typer(no_args_is_help=True)


@app.callback()
def stitch() -> None:
    """Lightsheet stitching workflows."""
    from squisher.jpegxr_zarr import register_jpegxr_codec

    register_jpegxr_codec()


@app.command("screen-overlaps")
def screen_overlaps(
    position_json: Annotated[
        Path,
        typer.Option("--position-json", exists=True, dir_okay=False, help="Tile metadata position JSON."),
    ],
    zarr_dir: Annotated[
        Path,
        typer.Option(
            "--zarr-dir", exists=True, file_okay=False, help="Directory containing tile OME-Zarrs."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output level-2 overlap-screen manifest."),
    ],
    threshold: Annotated[
        float,
        typer.Option("--threshold", min=0.0, help="Human-reviewed foreground threshold."),
    ],
    level: Annotated[int, typer.Option("--level", min=1)] = 2,
    channel: Annotated[int, typer.Option("--channel", min=0)] = 0,
    z_chunks: Annotated[int, typer.Option("--z-chunks", min=1)] = 6,
    min_foreground_pixels: Annotated[
        int, typer.Option("--min-foreground-pixels", min=1)
    ] = 256,
    min_foreground_fraction: Annotated[
        float, typer.Option("--min-foreground-fraction", min=0.0, max=1.0)
    ] = 0.05,
) -> None:
    """Screen every adjacent pair and registration z chunk before level-0 reads."""
    from squisher_lightsheet.overlap_screen import screen_level2_overlaps

    result = screen_level2_overlaps(
        position_json=position_json,
        zarr_dir=zarr_dir,
        output=output,
        threshold=threshold,
        level=level,
        channel=channel,
        z_chunks=z_chunks,
        min_foreground_pixels=min_foreground_pixels,
        min_foreground_fraction=min_foreground_fraction,
        progress=typer.echo,
    )
    typer.echo(str(result))


@app.command("register")
def register(
    position_json: Annotated[
        Path,
        typer.Option("--position-json", exists=True, dir_okay=False, help="Tile metadata position JSON."),
    ],
    zarr_dir: Annotated[
        Path,
        typer.Option(
            "--zarr-dir", exists=True, file_okay=False, help="Directory containing the tile OME-Zarrs."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for measurements, optimized positions, registration.json, and provenance.",
        ),
    ],
    threshold: Annotated[
        float,
        typer.Option("--threshold", min=0.0, help="Exact threshold selected from the reviewed TIFF."),
    ],
    method8_summary: Annotated[
        Path | None,
        typer.Option(
            "--method8-summary",
            help="Existing registration measurement summary to optimize from; skips measurement.",
        ),
    ] = None,
    z_chunks: Annotated[int, typer.Option("--z-chunks", min=1, help="Number of z-only chunks per seam.")] = 6,
    device: Annotated[int, typer.Option("--device", min=0, help="CUDA device for Method8.")] = 0,
    method8: Annotated[
        bool,
        typer.Option(
            "--method8",
            help=(
                "Opt into native Method8 refinement after phase correlation. "
                "Phase correlation with shifted-crop recovery is the default."
            ),
        ),
    ] = False,
    native_lib_dir: Annotated[
        Path,
        typer.Option(
            "--native-lib-dir",
            exists=True,
            file_okay=False,
            help="Directory containing the native Method8 libapi.so.",
        ),
    ] = DEFAULT_LIB_DIR,
    channel: Annotated[
        int,
        typer.Option("--channel", min=0, help="Channel index for CZYX level-0 inputs."),
    ] = 0,
    max_iterations: Annotated[int, typer.Option("--max-iterations", min=1)] = 300,
    ftol: Annotated[float, typer.Option("--ftol")] = 1e-4,
    min_corr: Annotated[
        float, typer.Option("--min-corr", help="Minimum accepted Method8 correlation.")
    ] = 0.15,
    min_grad_ncc: Annotated[
        float,
        typer.Option("--min-grad-ncc", help="Minimum accepted Method8 gradient-component NCC."),
    ] = 0.24,
    phase_recovery_min_prior_edges_per_axis: Annotated[
        int,
        typer.Option(
            "--phase-recovery-min-prior-edges-per-axis",
            min=1,
            help="Minimum first-pass accepted edges needed to form an x or y recovery prior.",
        ),
    ] = 3,
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
    allow_disconnected: Annotated[
        bool,
        typer.Option(
            "--allow-disconnected",
            help=(
                "Write canonical registration outputs even when the constraint graph is disconnected; "
                "connectivity remains recorded in provenance."
            ),
        ),
    ] = False,
) -> None:
    """Run human-reviewed thresholding, level-2 screening, and level-0 registration."""
    outputs = run_registration_workflow(
        position_json=position_json,
        zarr_dir=zarr_dir,
        output_dir=output_dir,
        threshold=threshold,
        method8_summary=method8_summary,
        z_chunks=z_chunks,
        device=device,
        method8=method8,
        native_lib_dir=native_lib_dir,
        channel=channel,
        max_iterations=max_iterations,
        ftol=ftol,
        min_corr=min_corr,
        min_grad_ncc=min_grad_ncc,
        phase_recovery_min_prior_edges_per_axis=phase_recovery_min_prior_edges_per_axis,
        max_grad_regression=max_grad_regression,
        max_corr_regression=max_corr_regression,
        phase_fallback=phase_fallback,
        min_phase_grad=min_phase_grad,
        min_phase_corr=min_phase_corr,
        phase_fallback_weight_scale=phase_fallback_weight_scale,
        allow_disconnected=allow_disconnected,
        progress=typer.echo,
    )
    typer.echo(
        json.dumps(
            {
                "threshold_record": str(outputs.threshold_record),
                "measurement_summary": str(outputs.measurement_summary),
                "optimized_positions": str(outputs.optimized_positions),
                "diagnostics": str(outputs.diagnostics),
                "constraints_jsonl": str(outputs.constraints_jsonl),
                "tile_corrections": str(outputs.tile_corrections),
                "canonical_positions": str(outputs.canonical_positions),
                "registration_json": str(outputs.registration_json),
            },
            indent=2,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
