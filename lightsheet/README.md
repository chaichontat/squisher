# squisher-lightsheet

Lightsheet stitching workflow commands for metadata positioning, rough center-z phase alignment,
registration, fusion, pyramid generation, QC, and 405-to-488 alignment orchestration.

See [WORKFLOW.md](WORKFLOW.md) for the current registration workflow,
coordinate contracts, QC gates, and canonical final output layout.

```bash
uv run --package squisher-lightsheet lightsheet --help
```

## WGACtrl-405-L2 Run Record (2026-07-10)

The commands below record the processing sequence launched for
`WGACtrl-405-L2`. Paths, the position-building helper, and tuning values are
specific to this acquisition. The final registration used unmasked phase
correlation with axis-prior shifted-crop recovery; method 8 was not enabled.

### Stage the Input and Configure the Environment

```bash
set -o pipefail
RUN_ROOT=/working/eduseg/20260709
IMAGE_DIR="$RUN_ROOT/WGACtrl-405-L2"

cd "$RUN_ROOT"
rsync -trv --info=progress2 --info=name0 \
  /mnt/hive/csriwor1/20260709-wgaref/WGACtrl-405-L2 .

source /home/chaichontat/miniforge3/etc/profile.d/conda.sh
conda activate multi
export PYTHONPATH=/home/chaichontat/squisher/deconv/src:/home/chaichontat/squisher/lightsheet/src:/home/chaichontat/squisher/squisher/src
export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$IMAGE_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$IMAGE_DIR/.cache"

mkdir -p "$MPLCONFIGDIR" "$RUN_ROOT/run-logs"
```

### Estimate BaSiC Illumination, Scale Intensities, and Deconvolve

```bash
BASIC_DIR="$IMAGE_DIR/basic"
BASIC_LABEL=WGACtrl_405_L2_z500_edgeReject_noContent_basicpy2_gpu_darkfield_sortIntensity_autotune_n500total
BASIC="$BASIC_DIR/$BASIC_LABEL-ch0.pkl"
SCALE_DIR="$IMAGE_DIR/squisher-deconv-scale-z100"
DECONV_DIR="$IMAGE_DIR/squisher-deconv-run-u16"
PSF=/home/chaichontat/nvme/lightsheet/20260606/psf_Image_57_571nm_zstrict/radial_symmetric_reskewed_rawscale_zcenter2_xy9_sumnorm.tif
inputs=("$IMAGE_DIR"/*.ome.tif)

python -u /home/chaichontat/nvme/lightsheet/scripts/fit_basic_ome_tiff_tiles.py \
  "$IMAGE_DIR" \
  --output-dir "$BASIC_DIR" \
  --label "$BASIC_LABEL" \
  --channels 0 \
  --z-total 500 \
  --autotune \
  --get-darkfield \
  --sort-intensity \
  --exclude-blank-slices \
  --exclude-edge-slices \
  --device cuda

python -u -m squisher_deconv sample-scale "${inputs[@]}" \
  --out-dir "$SCALE_DIR" \
  --planes 100 \
  --channels 1 \
  --psf "$PSF" \
  --basic "$BASIC" \
  --devices 0,1 \
  --queue-depth 2

python -u -m squisher_deconv run "${inputs[@]}" \
  --out-dir "$DECONV_DIR" \
  --channels 1 \
  --psf "$PSF" \
  --basic "$BASIC" \
  --scaling "$SCALE_DIR/scaling.json" \
  --output-mode u16 \
  --slab-depth 240 \
  --devices 0,1 \
  --queue-depth 2
```

### Build Positions and Register the Tiles

```bash
POSITIONS="$IMAGE_DIR/WGACtrl-405-L2.metadata.positions.json"
REG_DIR="$IMAGE_DIR/registration-level0-recovery"

# This acquisition-local helper must remain with the run record; it is not
# installed by squisher-lightsheet.
python -u "$RUN_ROOT/.codex/build_wgactrl_405_l2_positions.py" \
  "$DECONV_DIR" \
  "$POSITIONS" \
  --expected-tiles 90

CUDA_VISIBLE_DEVICES=0 lightsheet-stitch register \
  --position-json "$POSITIONS" \
  --zarr-dir "$DECONV_DIR" \
  --output-dir "$REG_DIR" \
  --threshold MANUALLY_SELECTED_VALUE \
  --channel 0 \
  --device 0 \
  --z-chunks 6 \
  2>&1 | tee -a "$RUN_ROOT/run-logs/codex-lightsheet-wgactrl-405-l2-register-recovery.log"
```

The canonical optimized inputs for fusion are
`$REG_DIR/registration.positions.json` and `$REG_DIR/registration.json`.

### Validate and Run Fusion

The dry run validates the registered inputs and reports the output geometry
without writing blocks. The second command performs the fusion.

```bash
CUDA_VISIBLE_DEVICES=0 lightsheet fuse \
  "$DECONV_DIR" \
  --position-input "$REG_DIR/registration.positions.json" \
  --registration-input "$REG_DIR/registration.json" \
  --output "$IMAGE_DIR" \
  --channel 0 \
  --fusion-weight-mode content-preibisch-coarse \
  --fusion-level 0 \
  --batch-size 1 \
  --output-chunksize-zyx 12,960,960 \
  --dry-run

CUDA_VISIBLE_DEVICES=0 lightsheet fuse \
  "$DECONV_DIR" \
  --position-input "$REG_DIR/registration.positions.json" \
  --registration-input "$REG_DIR/registration.json" \
  --output "$IMAGE_DIR" \
  --channel 0 \
  --fusion-weight-mode content-preibisch-coarse \
  --fusion-level 0 \
  --batch-size 1 \
  --output-chunksize-zyx 12,960,960 \
  2>&1 | tee -a "$RUN_ROOT/run-logs/codex-lightsheet-wgactrl-405-l2-fusion.log"
```

As of 2026-07-10, fusion is still running in the tmux session
`codex-lightsheet-wgactrl-405-l2-fusion`. A successful run promotes its hidden
staging output to `$IMAGE_DIR/fused.ch0.ome.zarr`. Inspect the active session
and log with:

```bash
tmux capture-pane -pt codex-lightsheet-wgactrl-405-l2-fusion -S -80
tail -n 80 "$RUN_ROOT/run-logs/codex-lightsheet-wgactrl-405-l2-fusion.log"
```

The completed registration log is
`$RUN_ROOT/run-logs/codex-lightsheet-wgactrl-405-l2-register-recovery.log`.

Reusable seam registration primitives live in `squisher_lightsheet.seams`. They cover
overlap-plus-margin seam sampling, masked phase correlation, robust-boundary settings,
and boundary constraint records used by both legacy robust-boundary registration and
405-to-488 overlap recovery workflows.

The 405-to-488 workflow wrappers are exposed under:

```bash
lightsheet align-405-to-488 --help
```

Render BaSiC/raw center-z dumb-stitch QC directly from source OME physical
metadata with:

```bash
lightsheet ome-metadata-dumb-stitch \
  --input-dir L=/path/to/sample-L-561638 \
  --input-dir R=/path/to/sample-R-561638 \
  --basic-dir /path/to/sample-LR-561638/basic \
  --output-dir /path/to/sample-LR-561638/basic/dumb-stitch-qc-ome-metadata \
  --channels 0,1
```

The command literally pastes tiles into a per-view mosaic from OME-TIFF plane
positions or OME-Zarr multiscales transforms. It does not draw labels or tile
outlines unless those options are explicitly enabled.

For side-internal registration, render the deconvolved OME-Zarr center plane as
a tiled, native-intensity `uint16` OME-TIFF before choosing the foreground
threshold. Omit `--center-z-index` to use the center plane:

```bash
lightsheet ome-metadata-dumb-stitch \
  --input-dir R=DECONVOLVED_TILES \
  --output-dir REGISTRATION_REVIEW \
  --channels 0 \
  --level 0 \
  --write-tiff \
  --output-prefix reviewed-center-z
```

Give `REGISTRATION_REVIEW/reviewed-center-z-R_raw_ch0_omeMetadata_noBlend.ome.tif`
to the human reviewer and stop. The PNG is only supplementary. After the
reviewer supplies a threshold, pass that exact threshold to `lightsheet-stitch
register`; do not chain registration before manual selection. The registration
command does not consume or record the review TIFF.

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

For within-acquisition stitching, the packaged command accepts only the
human-gated workflow. The required exact threshold is the review confirmation.
The command records it, screens every adjacent pair and planned Z chunk at
level 2, and reads level 0 only for accepted units before phase recovery and
global translation optimization:

```bash
CUDA_VISIBLE_DEVICES=0 lightsheet-stitch register \
  --position-json /path/to/sample.positions.json \
  --zarr-dir /path/to/deconvolved-tiles \
  --output-dir /path/to/registration-run \
  --threshold MANUALLY_SELECTED_VALUE \
  --channel 0 \
  --z-chunks 6
```

The output directory contains `level2-screen.json`, `registration.threshold.json`,
`registration.measurements.json`, `registration.optimized.positions.json`,
`registration.positions.json`, `registration.json`, constraints, corrections,
and optimization diagnostics. Automatic thresholds, unmasked registration,
caller-supplied screens, and partial pair lists are not register modes. The
command refuses stale provenance and existing final artifacts.

Phase correlation plus shifted-crop phase recovery is the default and does not
run Method8. Pass `--method8` only when native Method8 refinement is explicitly
required; in that mode, phase correlation remains the initializer and fallback.
`--native-lib-dir` only affects the opt-in Method8 mode.
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

## Cross-Channel Fused-Fixed Registration

The current workflow registers a tiled moving channel to an existing fused
fixed-channel OME-Zarr. The fixed mosaic defines the target coordinate frame.
Local registration estimates an affine for each moving-tile window; fusion
applies those affines without resampling the materialized crop bytes first.

Use this order:

1. Coarsely place the moving tiles in the fixed frame.
2. Fit phase-primed native Method10 affines on a non-overlapping core grid. A
   fixed-image threshold is acquisition-specific. For the L-514638
   moving-channel-0 to fused-channel-1 run, use `--fixed-mask-threshold 50`;
   the mask is evaluated on fixed fused level 2. Mark threshold-empty or overly
   masked windows as rejected and exclude them from recovery.
3. Detect implausible or missing fits from per-tile leave-one-out corner-displacement
   residuals. Rerun eligible windows from a leave-one-out initializer. Average
   the affine linear blocks by polar decomposition: use a generalized SO(3)
   rotation mean and an arithmetic mean of the symmetric stretch tensors;
   spatially interpolate translation only.
   A tile without enough local support may use only registration-graph-adjacent
   good tiles.
4. Apply the physical linear-field filter. Reject windows and whole tiles with
   insufficient support or no field consensus. Later stages must not restore
   these rows.
5. Fit the retained field again and replace only robust affine outliers. Keep
   every other accepted measured matrix and translation exactly as fitted.
6. Materialize accepted windows from the corrected summary with overlap. The
   standard geometry uses `480,480,480` level-0 source cores,
   `528,528,528` level-0 source windows, a 480-voxel step, and `4,4,4`
   reduction. Regular outputs have shape `132,132,132` and overlap their
   neighbors by 12 materialized pixels; tail-aligned windows may overlap more.
7. Fuse the materialized windows on the fixed mosaic's selected pyramid grid,
   then validate the OME-Zarr completion marker, multiscales, shapes, chunks,
   codecs, and an overlay-free intensity QC image.

The overlap in step 6 is required. Materializing only the 480-voxel level-0
source cores gives the blending weights no shared pixels at core boundaries
and produces a checker-stripe pattern. The materializer maps each accepted
core to the 528-voxel source window at the corresponding grid index, shifts its
stage origin, and retains its selected affine in `registered_affine`.

### Recovery, Filtering, and Outlier-Only Replacement

The commands below recover and finalize an existing fused-fixed native-method
summary; coarse placement and the initial sweep are prerequisites. Recovery
uses the native method and intensity transform recorded by that summary, so a
Method6 input is rerun with Method6 rather than Method10.
Accepted fits seed recovery only when their leave-one-out transform differs
from the remaining fits by no more than the configured corner-displacement
tolerance. Outliers become recovery targets and cannot seed their own tile or
an adjacent tile. A refit of an accepted spatial outlier is retained only when
its gradient NCC exceeds the original fit and it remains within the
refit-displacement tolerance of the consensus initializer. Detection defaults
to 5 px; the separate refit tolerance defaults to 10 px.
The historical filename remains `fused_fixed_method8_summary.json`. Each
command writes a new directory and refuses to overwrite an existing result.

```bash
RECOVERY_SCRIPT=lightsheet/scripts/recover_fused_fixed_method10_outliers.py
RAW_SUMMARY=/path/to/native-run/fused_fixed_method8_summary.json
MOVING_TILE_ADJACENCY=/path/to/moving-registration/registration.measurements.json
RECOVERED_DIR=/path/to/native-spatial-recovery
FILTERED_DIR=/path/to/native-linear-filter
CORRECTED_DIR=/path/to/native-final

python "$RECOVERY_SCRIPT" \
  --summary "$RAW_SUMMARY" \
  --output-dir "$RECOVERED_DIR" \
  --adjacency-json "$MOVING_TILE_ADJACENCY" \
  --devices 0,1

python "$RECOVERY_SCRIPT" \
  --summary "$RECOVERED_DIR/fused_fixed_method8_summary.json" \
  --output-dir "$FILTERED_DIR" \
  --exclude-nonlinear-only

python "$RECOVERY_SCRIPT" \
  --summary "$FILTERED_DIR/fused_fixed_method8_summary.json" \
  --output-dir "$CORRECTED_DIR" \
  --smooth-retained-linear
```

Despite its historical flag name, `--smooth-retained-linear` does not smooth
the entire retained field. It detects leave-one-out corner-displacement outliers and
changes only flagged rows. Their translation comes from the inlier spatial
field; their complete affine linear block comes from the same polar-decomposed
SO(3)/stretch mean used by recovery. The output report records replaced and
exactly preserved counts.

### Overlapping Materialization, Fusion, and QC

Pass the filtered, outlier-replaced summary through `--source-summary`; it
contains the final accepted rows and selected transforms. A manifest generated
before those stages contains stale acceptance and affine data.

```bash
MOVING_POSITIONS=/path/to/moving.positions.json
FIXED_FUSED=/path/to/fixed/fused.ch1.ome.zarr
MAT_DIR="$CORRECTED_DIR/materialized-level2-overlap528"
FUSION_DIR=/path/to/cross-registered-fusion

lightsheet fused-fixed-materialize-overlap \
  --moving-position "$MOVING_POSITIONS" \
  --source-summary "$CORRECTED_DIR/fused_fixed_method8_summary.json" \
  --output-dir "$MAT_DIR" \
  --source-channel 0 \
  --core-shape-zyx 480,480,480 \
  --window-shape-zyx 528,528,528 \
  --level-factor-zyx 4,4,4 \
  --zstd-level 3 \
  --workers 4

lightsheet fuse \
  "$MAT_DIR/materialized_tiles" \
  --position-input "$MAT_DIR/fused_fixed_materialized_chunks.positions.json" \
  --registration-input "$MAT_DIR/fused_fixed_materialized_chunks.registration.json" \
  --output "$FUSION_DIR/fused.ome.zarr" \
  --fusion-weight-mode content-preibisch-coarse \
  --fusion-level 0 \
  --batch-size 1 \
  --output-codec zstd \
  --zstd-level 3 \
  --output-chunksize-zyx 12,960,960 \
  --output-grid-template "$FIXED_FUSED" \
  --output-grid-template-level 2 \
  --channel 0

lightsheet fused-tile-index-qc \
  "$FUSION_DIR/fused.ch0.ome.zarr" \
  --registration-input "$MAT_DIR/fused_fixed_materialized_chunks.registration.json" \
  --output "$FUSION_DIR/fused.ch0.level2.center-z.intensity.png" \
  --level 2 \
  --no-labels \
  --no-markers
```

Fusion writes structured lineage into the completed OME-Zarr. The compact
root `squisher_fusion` attribute points to `provenance/manifest.json`; the
bundle includes the position and registration inputs, native and recovered
window fits, outlier-only replacement decisions, overlapping materialization
manifests, BaSiC settings, deconvolution settings, requested and resolved
fusion options, and the actual pyramid layout. Large image stores, PSFs, and
profile pickles remain external references. Missing historical JSON is listed
under `coverage.unresolved` and changes the provenance status to `partial`.

The materializer writes accepted rows only. Isolated black rectangles in QC
can therefore represent rejected or missing windows; an alternating stripe at
the core-grid cadence indicates that non-overlapping materialization was used.

### Baseline Quadrant Method8 CLI

The staged `lightsheet cross-register-method8` CLI remains available for
Image_10-to-Image_14 and other de novo Method8 runs. Its `coarse`, `method8`,
`materialize`, and `manifest` commands produce baseline artifacts. Use the
fused-fixed flow above when recovery, physical filtering, and overlap-aware
finalization are required.

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

The `--output` path passed to `lightsheet fuse` is channel-neutral. The fuser
inserts `.ch{channel}` before `.ome.zarr`; for example, `fused.ome.zarr` writes
`fused.ch1.ome.zarr` for `--channel 1`. When `--output` names a directory, the
default channel-neutral base is `fused.ome.zarr` inside that directory.

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

### Baseline De Novo Method8 Run

The example below is a separate de novo Method8 path. Its summary and direct
materialization artifacts are not inputs to fused-fixed Method10 recovery.

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
  --output "$RUN_DIR" \
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
