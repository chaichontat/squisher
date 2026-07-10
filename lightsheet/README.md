# squisher-lightsheet

Lightsheet stitching workflow commands for metadata positioning, rough center-z phase alignment,
registration, fusion, pyramid generation, QC, and 405-to-488 alignment orchestration.

```bash
uv run --package squisher-lightsheet lightsheet --help
```

Reusable seam registration primitives live in `squisher_lightsheet.seams`. They cover
overlap-plus-margin seam sampling, masked phase correlation, robust-boundary settings,
and boundary constraint records used by both legacy robust-boundary registration and
405-to-488 overlap recovery workflows.

The 405-to-488 workflow wrappers are exposed under:

```bash
lightsheet align-405-to-488 --help
```

## Cross-Channel Native Registration

### Native CUDA DLL Setup

Native method-8 registration uses the `microImageLib` CUDA DLL through
`squisher_lightsheet.native_reg3dgpu`. The current Linux DLL location is:

```text
/home/chaichontat/microImageLib/bin/linux/libapi.so
```

Run native registration, spotcheck rendering, and fusion from the `multi`
environment with CUDA paths exported before the Python process starts:

```bash
source /home/chaichontat/miniforge3/etc/profile.d/conda.sh
conda activate multi
export PYTHONPATH=/home/chaichontat/squisher/lightsheet/src:/home/chaichontat/squisher/squisher/src
export CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux
export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
```

`CUDA_PATH` lets CuPy find CUDA headers for JIT kernels. `LD_LIBRARY_PATH`
lets the native DLL loader resolve CUDA runtime libraries such as
`libcudart.so.12` and `libcufft.so.11`. Missing exports commonly show up as
`Failed to auto-detect CUDA root directory` or
`OSError: libcudart.so.12: cannot open shared object file`, even if Python can
import CuPy.

Before a long run, verify both CUDA and the DLL path:

```bash
test -f /home/chaichontat/microImageLib/bin/linux/libapi.so
python - <<'PY'
import cupy as cp
from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR

print("cuda devices:", cp.cuda.runtime.getDeviceCount())
print("native lib dir:", DEFAULT_LIB_DIR)
PY
```

For within-acquisition stitching with the packaged method-8 CLI, use the same
activated environment:

```bash
CUDA_VISIBLE_DEVICES=0 lightsheet-stitch register \
  --threshold 3000 \
  --z-chunks 6
```

`lightsheet-stitch register` uses the default native library directory above.
For cross-channel Method8 runs, pass the native library directory to the staged
command when using a non-default DLL location:

```bash
lightsheet cross-register-method8 method8 \
  --native-lib-dir /home/chaichontat/microImageLib/bin/linux \
  ...
```

For long native runs, prefer an activated shell or the environment executable
directly. Avoid captured `conda run` wrappers because they can hide progress
and make monitoring unreliable.

The Image_10-to-Image_14 quadrant Method8 workflow is exposed as staged CLI
commands. Use the artifact naming convention in
`Cross-Channel Quadrant Method8 Registration` below when choosing `--output-dir`
and explicit output paths.

For ambiguous local seam fixes, render explicit candidate correction grids instead of
editing position files by hand:

```bash
lightsheet align-405-to-488 render-candidate-grid \
  --position-input 405.to488.optimized.positions.json \
  --registration-input 405-to488.optimized.registration.json \
  --candidate-json top_left_candidates.json \
  --output-dir top_left_basin_grid \
  --channel 0 \
  --level 4 \
  --render-jobs 2
```

`top_left_candidates.json` is a Cartesian-product spec. Correction values are
`z,y,x` pixel deltas at the same native pixel scale as the position file:

```json
{
  "tiles": [
    {
      "tile": "230Tnc-CL-405.001.ome.tif",
      "label": "L001",
      "candidates": {
        "smallY": [0.2, 9.15, -0.15],
        "horiz": [2.2, 22.95, -89.95]
      }
    },
    {
      "tile": "230Tnc-CL-405.005.ome.tif",
      "label": "L005",
      "candidates": {
        "zero": [0.0, 0.0, 0.0],
        "minusY58_plusX22": [5.2, -57.9, 22.3]
      }
    }
  ]
}
```

The helper writes one variant folder per combination, plus
`candidate_grid_summary.json`, `candidate_grid_sheet.png`, and
`candidate_grid_boundaries_sheet.png`.

## Standard MVS Seam Refinement

Use `mvs-refine-level0` to refine a coarse standard MVS registration with
level-0 seam phase correlation and then solve one globally consistent
translation correction field. This is a seam optimization step: accepted seam
measurements become constraints, low-weight coarse fallback constraints keep
the graph connected when level-0 measurements are rejected, and the output JSON
stores optimized per-tile corrections under `.metrics.level0_refinement`.

Run long jobs from the `multi` environment's executable directly, with local
source on `PYTHONPATH`; do not wrap them in captured `conda run`.

```bash
PYTHONPATH=/home/chaichontat/squisher/lightsheet/src:/home/chaichontat/squisher/squisher/src \
CUDA_VISIBLE_DEVICES=1 \
/home/chaichontat/miniforge3/envs/multi/bin/lightsheet mvs-refine-level0 \
  --registration-input INPUT_REGISTRATION.json \
  --output-registration OUTPUT_REGISTRATION.json \
  --channel 0 \
  --patch-shape-zyx 10,480,480 \
  --patches-per-edge 12 \
  --retry-patches-per-edge 50 \
  --min-quality 0 \
  --all-edges \
  --workers 1 \
  --fallback-refinement-level 1
```

Phase-correlation shifts at the exact search boundary are invalid, not
successful large shifts. With the default `--max-phase-shift-zyx 3,64,64`,
results such as `y=-64` or `x=64` are saturated search-bound hits and must be
rejected before the global seam solve.

Recorded Image_10/Image_14 seam-refinement artifacts:

- Logs:
  `/working/test/00_run_logs/15_standard_zarr_level1_pyramid_prodready_refine_saturated_bound_reject_tmux.log`,
  `/working/test/00_run_logs/13_image10_level0_refinement_level1_fallback_saturated_bound_reject.log`,
  `/working/test/00_run_logs/14_image14_level0_refinement_level1_fallback_saturated_bound_reject.log`
- Image_10 output:
  `/working/test/01_image10_standard_mvs_level2_refine/04_level0_refinement_level1_fallback_saturated_bound_reject.registration.json`
- Image_10 result:
  `accepted_constraints=192/192`, `connected_tiles=107/107`,
  `p95_abs_residual_zyx=(3.76,14.01,11.71)`,
  `clamp_hits_zyx=[2,0,0]`, with 70 low-weight level-2 fallback constraints.
- Image_14 output:
  `/working/test/02_image14_standard_mvs_level2_refine/04_level0_refinement_level1_fallback_saturated_bound_reject.registration.json`
- Image_14 result:
  `accepted_constraints=187/192`, `connected_tiles=107/107`,
  `p95_abs_residual_zyx=(2.51,19.63,17.54)`,
  `clamp_hits_zyx=[1,2,1]`, with 34 low-weight level-2 fallback constraints.

## Cross-Channel Quadrant Method8 Registration

Use the staged `lightsheet cross-register-method8` CLI for Image_10-to-Image_14
or other fixed/moving cross-acquisition registration. The fixed acquisition
defines the target coordinate frame. The moving acquisition is transformed into
that frame while preserving its own pixel data through materialized quadrant/z
crops.

The supported stages are:

1. `coarse`: run level-2 tile phase alignment to produce the moving-to-fixed
   initialization position JSON.
2. `method8`: split each fixed tile into four 10%-overlap xy quadrants and
   480-plane z chunks, run native Method8 with phase-correlation priming, and
   write one window JSON per accepted/rejected candidate.
3. `materialize`: cut simple Image10-local quadrant/z crops, skip only windows
   marked unusable by Method8, and write materialized position/registration
   JSONs. Method8 transforms are not applied to crop bytes; they are stored in
   each materialized chunk's `registered_affine` for fusion.
4. `manifest`: summarize the expected output paths for a run directory.

### Artifact Naming

Names must encode direction, channels, method, geometry, and target space. Do
not use relative state words such as `current`, `corrected`, `new`, `old`,
`latest`, or `final`.

Use this run-id pattern for the registration/materialization directory:

```text
{moving}.ch{moving_ref}_to_{fixed}.ch{fixed_ref}.{workflow}.{geometry}.{mask}.{initializer}
```

For the Image_10 channel-0 to Image_14 channel-0 quadrant Method8 run with
480-plane cores, 10% overlap, 528-plane windows, threshold 3000, and phase
preseeding, name the run directory after the fused product:

```text
Image10-ch1-fused
```

Use matching short names for the Image14-native fused grid:

```text
Image14-ch0-fused/
Image14-ch0-fused/Image14-ch0-fused.ch0.ome.zarr
Image14-ch0-fused/Image14-ch0-fused.identity-registration.json
Image14-ch0-fused/Image14-ch0-fused.positions.json
Image14-ch0-fused/qc/Image14-ch0-fused.tile-index.png
```

Within the Image10 fused run directory, use these artifact names:

```text
Image10-ch1-fused.manifest.json
Image10-ch1-fused.coarse.positions.json
Image10-ch1-fused.reference.positions.json
Image10-ch1-fused.method8-summary.json
coarse.ch0/
method8-windows/
materialized.ch1/
materialized.ch1/chunks.positions.json
materialized.ch1/chunks.registration.json
materialized.ch1/chunks.summary.json
materialized.ch1/tiles/
Image10-ch1-fused.ome.zarr
Image10-ch1-fused.ch1.ome.zarr
qc/Image10-ch1-fused.tile-index.png
qc/Image10-ch1-fused.tile-seams.contact-sheet.png
qc/Image10-ch1-fused.tile-seams.contact-sheet.json
```

The `--output` path passed to `lightsheet fuse` is the channel-neutral base
ending in `.level0.ome.zarr`; the fuser writes the channel-specific output with
`.level0.ch{channel}.ome.zarr`. Keep both names in the convention because they
appear in commands and logs at different points.

Keep stage names stable:

- `coarse-aligned` means phase-correlation initialization only.
- `Image10-ch1-fused.method8-summary.json` summarizes native Method8 window
  registration.
- `method8-windows/` contains one accepted/rejected Method8 candidate record per
  quadrant/z window.
- `materialized.ch{channel}/tiles/` contains simple source-local quadrant/z
  crops. Method8 is not baked into the crop bytes.
- `materialized.ch{channel}/chunks.registration.json` is the fuser registration contract;
  each chunk carries the Method8 transform as `registered_affine`.
- `Image10-ch1-fused...` is the fused moving-channel mosaic in Image14 space.
  Do not call materialized inputs fused, and do not call fused outputs
  materialized tiles.

### Example De Novo Run

Define the run id once and use explicit paths for artifacts whose default names
are intentionally shorter:

```bash
RUN_ID=Image10-ch1-fused
RUN_DIR="$PWD/$RUN_ID"
MAT_DIR="$RUN_DIR/materialized.ch1"
FUSED_PREFIX=Image10-ch1-fused
IMAGE14_FUSED_DIR="$PWD/squisher-deconv-run-u16/Image14-ch0-fused"

lightsheet cross-register-method8 coarse \
  --fixed-position Image_14.zarr.reference.positions.json \
  --output-dir "$RUN_DIR" \
  --output-position "$RUN_DIR/Image10-ch1-fused.coarse.positions.json" \
  --fixed-token Image_14 \
  --moving-token Image_10 \
  --fixed-channel 0 \
  --moving-channel 0

lightsheet cross-register-method8 method8 \
  --fixed-position Image_14.zarr.reference.positions.json \
  --coarse-moving-position "$RUN_DIR/Image10-ch1-fused.coarse.positions.json" \
  --output-dir "$RUN_DIR" \
  --fixed-mask-threshold 3000 \
  --core-shape-zyx 480,480,480 \
  --window-shape-zyx 528,528,528 \
  --workers 4 \
  --devices 0,0,1,1

lightsheet cross-register-method8 materialize \
  --window-json-dir "$RUN_DIR/method8-windows" \
  --coarse-moving-position "$RUN_DIR/Image10-ch1-fused.coarse.positions.json" \
  --fixed-registration "$IMAGE14_FUSED_DIR/Image14-ch0-fused.identity-registration.json" \
  --output-dir "$RUN_DIR" \
  --materialized-output-dir "$MAT_DIR" \
  --fusion-channel 1 \
  --channel-source-shift-px-zyx 2.299999952316284,1.2999999523162842,1.100000023841858
```

With the current CLI defaults, materialization writes:

```text
$MAT_DIR/chunks.positions.json
$MAT_DIR/chunks.registration.json
$MAT_DIR/chunks.summary.json
$MAT_DIR/tiles/
```

Fuse the materialized tiles with the standard fuser. Use the Image14 fused grid
as `--output-grid-template` so the Image10 channel lands in Image14 space:

```bash
lightsheet fuse \
  "$MAT_DIR/tiles" \
  --position-input "$MAT_DIR/chunks.positions.json" \
  --registration-input "$MAT_DIR/chunks.registration.json" \
  --output "$RUN_DIR/$FUSED_PREFIX.ome.zarr" \
  --fusion-weight-mode content-preibisch-coarse \
  --fusion-level 0 \
  --batch-size 110 \
  --output-chunksize-zyx 12,960,960 \
  --output-grid-template "$IMAGE14_FUSED_DIR/Image14-ch0-fused.ch0.ome.zarr" \
  --channel 1
```

For Image_10 channel 1, pass the measured source-local ch1-to-ch0 shift to
`materialize`. The registration keeps the channel-0 Method8 linear block and
folds the source offset into the translation as `M_ch1 = M_ch0 @ T(c)`.

Render seam and tile-index QC from the fused output with the standard QC
commands, for example:

```bash
lightsheet fused-tile-index-qc \
  "$RUN_DIR/$FUSED_PREFIX.ch1.ome.zarr" \
  --registration-input "$MAT_DIR/chunks.registration.json" \
  --level 2 \
  --output "$RUN_DIR/qc/$FUSED_PREFIX.tile-index.png"
```

## Fusion OME-Zarr chunking

Do not blindly copy per-tile source chunks to a fused mosaic. Tile-local chunks
such as `12,240,240` can create hundreds of thousands of output Zarr chunks and
make the writer CPU/file-count bound. Preserve the z slab when that matches the
source access pattern, but choose XY chunks for the fused output grid and the
writer batch size.

Before a full run, benchmark one bounded fusion batch with
`--profile-max-fusion-batches 1` and compare block count, wall time, read/write
bytes, and output file count. For Image_14 level-0 fusion, `12,240,240`
produced `433048` output blocks and was projected to take more than a day,
while `12,960,960` produced `29260` blocks and a representative batch finished
in about 10 seconds.

These commands require explicit input paths and do not hardcode local dataset locations.
