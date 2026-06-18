# Lean v1: `squisher-lightsheet` Stitching Package

## Summary

Create a sibling monorepo package for the accepted basic stitching workflow only. Keep the existing `squisher` compression package unchanged. Do not migrate channel alignment or experimental diagnostics in v1.

## Package Layout

```text
~/squisher/
  pyproject.toml
  squisher/
  lightsheet/
    pyproject.toml
    src/squisher_lightsheet/
    tests/
```

`lightsheet/pyproject.toml` exposes:

```toml
[project]
name = "squisher-lightsheet"

[project.scripts]
lightsheet = "squisher_lightsheet.cli:main"
```

The root workspace includes:

```toml
[tool.uv.workspace]
members = ["squisher", "lightsheet"]
```

## CLI

Implement only:

```bash
lightsheet position
lightsheet rough-phase
lightsheet register
lightsheet fuse
lightsheet pyramid
lightsheet qc
lightsheet run-tltr
```

`run-tltr` orchestrates:

```text
position -> rough-phase -> register -> qc
```

with optional `--fuse` and `--pyramid`.

## Implementation

Port accepted code into owner modules:

- `positions.py`: metadata tile positions, centroid alignment, x/z join geometry.
- `rough_phase.py`: center-z dumb fusion, phase correlation, rough overlay.
- `registration.py`: multiview-stitcher registration only.
- `fusion.py`: content-aware fusion.
- `pyramid.py`: OME-Zarr pyramid generation.
- `qc.py`: registration overlays.
- `workflow.py`: `run-tltr`.
- `cli.py`: Typer command wrappers.

Use one `workflow_summary.json` per run directory. Add minimal `schema_version` and `artifact_type` to downstream-consumed JSONs.

## Defaults

Named modes:

- `tltr_x_join_center_z_phase`
  - join axis `x`
  - no flips
  - overlap `0.25`
  - centroid alignment in `yz`

- `lr_z_endview_flip_xz`
  - join axis `z`
  - flip `x` and `z`
  - centroid alignment in `xy`

Fusion defaults:

```text
content-preibisch-coarse
sigma1 = 7
sigma2 = 17
stride = 1,8,8
cache_tiles = 128
```

## Tests

Add focused tests for:

- CLI help and argument parsing.
- x-join has no flips and aligns centroids in `yz`.
- z-join flips x/z and aligns centroids in `xy`.
- overlap only affects the join axis.
- `rough-phase` synthetic translation recovery and overlay output.
- `run-tltr --dry-run` output paths.
- existing `squisher` tests still pass.

## Out Of Scope

Do not migrate channel alignment, Mattes MI refinement, raw tile experiments, masked NCC, or one-off diagnostics in v1.
