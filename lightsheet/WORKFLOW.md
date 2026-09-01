# Lightsheet Registration Workflow

This document records the registration workflow used for the current Tnc
lightsheet datasets. The README stays focused on CLI reference; this file
describes the order of operations, the coordinate contracts, and the checks that
decide whether a run is usable.

## Hard Contracts

- Use metadata to identify tracks, channels, source views, pixel sizes, and tile
  positions. Do not infer wavelength order from folder names.
- For `230Tnc-CL/CR-488514561638`, the Zeiss track metadata is the source of
  truth: `track0` channels `(0, 1)` are the 561/638 pair, `track1` channel `2`
  is 514, and `track2` channel `3` is 488.
- For 514/561/638 within one physical 488/514/561/638 acquisition, solve one
  tile geometry per source view and apply it to the channels in that
  acquisition. Per-channel registrations are diagnostics unless a separate
  physical acquisition justifies separate geometry.
- For separately imaged 405 data, preserve the 405 acquisition's own tile
  geometry. The 488 acquisition defines the target frame, but 488 transforms
  must not be copied onto 405 tiles.
- For dense 405-to-fused-channel Method 6 refinement, rotate about the full
  moving tile's XY center and the current moving slab's Z center. The default
  linear initializer is the median polar rotation and stretch from the
  accepted tile-061 and tile-081 slab-pivot trials, recorded as
  `DEFAULT_STARTING_AFFINE_MATRIX_ZYX` in the fused-fixed trial script.
- When previews or phase-correlation initialization need lower resolution data,
  read the TIFF subIFD or OME-Zarr pyramid level. Do not fake pyramid data by
  striding the base-resolution array.
- Fusion outputs must use the zarr-backed direct OME-Zarr fuser path.
- `lightsheet-stitch register` defaults to level-0 phase correlation followed
  by axis-prior shifted-crop phase recovery. Native Method8 is opt-in with
  `--method8`; it must not run in the default side-internal workflow.
- `lightsheet-stitch register` accepts only a validated, human-reviewed tiled
  `uint16` YX OME-TIFF and its explicit threshold. It generates and consumes a
  complete level-2 pair-by-Z-chunk screen before any level-0 crop read; it has
  no automatic-threshold, unmasked, caller-supplied-screen, or partial-pair mode.
- Final deliverables use this folder contract:

```text
<sample-output>/
  fused.ch0.ome.zarr
  fused.ch1.ome.zarr
  registration.json
  basic
  README
```

## Runtime Setup

Run long GPU work from the `multi` environment's executable or an activated
shell. Avoid captured `conda run` wrappers for long jobs because they can buffer
stdout and hide progress.

```bash
source /home/chaichontat/miniforge3/etc/profile.d/conda.sh
conda activate multi
export PYTHONPATH=/home/chaichontat/squisher/lightsheet/src:/home/chaichontat/squisher/squisher/src
export CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux
export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
```

For native Method8, verify CUDA and the microImageLib shared library before a
long run:

```bash
test -f /home/chaichontat/microImageLib/bin/linux/libapi.so
python - <<'PY'
import cupy as cp
from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR

print("cuda devices:", cp.cuda.runtime.getDeviceCount())
print("native lib dir:", DEFAULT_LIB_DIR)
PY
```

## BaSiC And Deconvolution

1. Build the BaSiC profile before registration or fusion QC.
2. Prefer the pooled profile only when QC shows it improves seam continuity.
   For the accepted 230 488/514/561/638 sample, the pooled profile across both
   channels and both sides looked best.
3. Cache selected BaSiC slices so repeated model fits do not resample the
   dataset.
4. Deconvolve after BaSiC. Side-specific CL and CR deconvolution can be run
   independently, but downstream registration should treat side placement as a
   separate step.
5. Render a no-blend dumb stitch after BaSiC and again after deconvolution.
   Use global display normalization for visual comparison, and prefer threshold
   methods such as Li or minimum over hard-coded constants.

The dumb stitch is a QC image, not a registration model. It should place tiles
from metadata or the current position JSON and literally paste image data
without overlap averaging.

For source OME-TIFF or OME-Zarr tiles, render the center-z metadata QC with the
packaged CLI. This reads OME physical tile positions directly and draws no tile
labels or outlines by default:

```bash
lightsheet ome-metadata-dumb-stitch \
  --input-dir L=/path/to/sample-L-561638 \
  --input-dir R=/path/to/sample-R-561638 \
  --basic-dir /path/to/sample-LR-561638/basic \
  --output-dir /path/to/sample-LR-561638/basic/dumb-stitch-qc-ome-metadata \
  --channels 0,1
```

## Side-Internal Registration

Side-internal registration solves tile-to-tile geometry within one source view,
for example CL alone or CR alone.

Start by rendering the metadata-positioned center-Z plane as a tiled `uint16`
YX OME-TIFF containing native deconvolved intensities without normalization,
rescaling, clipping, or display stretching. Do not launch registration in the
same command or shell chain: give the full TIFF path to a human, inspect it,
choose the foreground threshold in source intensity units, and only then pass
the TIFF and exact value to registration.

```bash
lightsheet ome-metadata-dumb-stitch \
  --input-dir R=DECONVOLVED_TILES \
  --output-dir REGISTRATION_REVIEW \
  --channels 0 \
  --level 0 \
  --write-tiff \
  --output-prefix reviewed-center-z
```

Omitting `--center-z-index` selects the center Z plane. Review the generated
`reviewed-center-z-R_raw_ch0_omeMetadata_noBlend.ome.tif`; the PNG is only a
supplement. The manifest records the display range and per-tile Z index. Pause
here for manual threshold selection.

1. Render the native-intensity center-Z metadata dumb-stitch TIFF and select the threshold.
2. Generate metadata-driven positions.
3. Run `register`; it screens all adjacent pair-by-Z-chunk units at level 2
   with the selected threshold before reading accepted units at level 0.
4. Report both the original phase-correlation score and the recovery score.
5. Optimize positions from the accepted constraints.
6. Render no-blend dumb-stitch QC from the optimized positions.

The standard packaged call for steps 2–5 is:

```bash
lightsheet-stitch register \
  --position-json POSITIONS.json \
  --zarr-dir DECONVOLVED_TILES \
  --output-dir REGISTRATION_RUN \
  --channel 0 \
  --threshold MANUALLY_SELECTED_VALUE \
  --z-chunks 6
```

This call is phase-only by default: it performs the initial level-0 phase
correlation, reruns failed or ambiguous edges with axis-prior shifted crops,
and optimizes positions from the accepted phase constraints. Use `--method8`
only for an explicitly requested native refinement experiment.

By default, registration refuses to emit fusion-ready canonical artifacts when
the accepted constraint graph is disconnected. Use `--allow-disconnected` only
after explicitly accepting that condition. The canonical provenance retains
both the total and connected tile counts; the option does not infer missing
constraints or claim that disconnected tiles were registered.

It emits both `registration.positions.json` and the identity-affine
`registration.json` expected by fusion; optimized placement is stored in each
tile's stage translation.

The recovery sequence is part of the main workflow. For every adjacent edge,
first measure the normal phase-correlation chunks. Then estimate horizontal and
vertical median shifts separately from accepted edges and rerun phase
correlation on prior-shifted full face-adjacent crops. The recovery crop must
use tile data from at least one full face-adjacent side and clip to tile bounds;
do not reduce the matcher to a thin overlap sliver. The recovery run should
emit an aligned table with three significant figures showing:

```text
edge  axis  phase_corr_before  phase_shift_before_zyx  phase_corr_after  recovered_shift_zyx  decision
```

Rejected saturated boundary shifts are diagnostics. They should not enter the
position optimizer.

## CL/CR Right-To-Left Merge

The CL/CR merge places CR into CL space after the two source views already have
side-internal geometry.

1. Render a level-2 z-median dumb stitch for CL and CR. Use tile positions based
   on the Li/side-internal geometry.
2. Phase-correlate the CL-fixed and CR-moving z-median mosaics. This estimates
   the coarse CR-to-CL shift.
3. At level 0, sample overlapping CL/CR regions implied by that shift. Use the
   same exhaustive chunking pattern as the 405 mapping work, and apply a mask
   filter derived from image content.
4. Run native Method8 on the accepted overlap chunks.
5. Use the median accepted Method8 local translation as the fine global CR
   correction. Add it to the z-median phase-correlation shift.
6. Do not fit or apply a single full affine to the CL/CR merge unless QC proves
   translation is insufficient. The accepted 230 CL/CR merge used the median
   translation from local Method8 fits, not a global affine.
7. Canonicalize the fused output with:

```bash
python /home/chaichontat/squisher/lightsheet/scripts/canonicalize_cl_cr_r_to_l.py \
  --dataset-dir DATASET_DIR \
  --left-position CL_POSITIONS.json \
  --right-position CR_POSITIONS.json \
  --manifest CLCR_ZMEDIAN_PHASE_MANIFEST.json \
  --phasecorr CLCR_PHASECORR.json \
  --method8-summary METHOD8_SUMMARY.json \
  --deconv-root DECONV_ROOT \
  --output-dir SAMPLE_OUTPUT \
  --render-qc
```

The canonicalization script writes `registration.json`, keeps a position JSON as
an internal fuser input, and prints fusion/movie commands that produce
`fused.ch0.ome.zarr` and `fused.ch1.ome.zarr`.

## 405-To-488 Mapping

The 405 acquisition is registered into the 488 coordinate frame without losing
405 seam continuity.

1. Register the 488-containing acquisition in its own frame.
2. Build a 405 TL/TR dumb stitch, then phase-correlate it against the 488 TLR
   reference to get the global 405-to-488 initialization.
3. Use that initialization to sample level-3 405-to-488 patches and measure
   per-tile anchors.
4. Refine accepted anchors at level 0.
5. Recover missing tiles from same-channel 405-to-405 overlap phase
   correlation.
6. Solve 405 tile positions in the 488 coordinate frame with robust weights.
7. Run robust-boundary 405 seam refinement after the cross-channel solve.
8. QC both questions separately:
   - Cross-channel QC: does 405 land on 488?
   - Seam QC: are neighboring 405 tiles continuous?

Good 405 seam residuals do not prove good 405-to-488 alignment. Always inspect
overall 405 red / 488 green overlays when judging the mapping.

## Fusion

Run fusion only after registration QC is acceptable. The CLI accepts either an
explicit zarr base path or the canonical output directory:

```bash
lightsheet fuse \
  INPUT_TILES \
  --position-input POSITIONS.json \
  --registration-input registration.json \
  --output SAMPLE_OUTPUT \
  --fusion-weight-mode content-preibisch-coarse \
  --fusion-level 0 \
  --batch-size 1 \
  --output-chunksize-zyx 8,1024,1024 \
  --channel 0
```

Production workflows write two separate fusion artifacts from the same accepted
transforms:

1. A level-2 preview encoded with Zstd for Napari review. It may use a
   level-2 materialization (`--level-factor-zyx 4,4,4`) and a level-2 fixed-grid
   template.
2. A level-0 production fusion encoded with JPEG-XR. Rematerialize from the
   corrected, deconvolved native tiles with `--level-factor-zyx 1,1,1`, then
   fuse onto the fixed level-0 grid. Never reuse the level-2 materialized pixels
   for this output.

Leave `--output-codec auto` at its default. The materializer and fuser resolve
Zstd for downsampled preview output and JPEG-XR for native level 0. An explicit
codec remains available for diagnostics, but is not part of the production
workflow. Keep the preview and production outputs in separate directories; a
completed preview is not a resumable level-0 fusion.

For fused-fixed cross-registration, the two materializations are:

```bash
# Napari preview input: level 2, Zstd
lightsheet fused-fixed-materialize-overlap \
  --source-summary FINAL_SUMMARY.json \
  --moving-position MOVING.positions.json \
  --output-dir MATERIALIZED_LEVEL2 \
  --level-factor-zyx 4,4,4

# Production input: level 0, JPEG-XR
lightsheet fused-fixed-materialize-overlap \
  --source-summary FINAL_SUMMARY.json \
  --moving-position MOVING.positions.json \
  --output-dir MATERIALIZED_LEVEL0 \
  --level-factor-zyx 1,1,1
```

Both materializations derive geometry from the same final registration summary.
The level-0 fusion guard rejects downsampled materialization when the requested
fixed-grid template is level 0.

When `--output SAMPLE_OUTPUT` has no `.zarr` suffix, the fuser normalizes it to
`SAMPLE_OUTPUT/fused.ome.zarr`. Separate channel outputs then land at
`SAMPLE_OUTPUT/fused.ch0.ome.zarr`, `SAMPLE_OUTPUT/fused.ch1.ome.zarr`, and so
on.

Use `--output-grid-template` when a moving acquisition must land on an existing
fixed fused grid.

Each successful fusion records a compact `squisher_fusion` root index and a
self-contained structured lineage bundle at `provenance/manifest.json` before
setting `squisher_complete`. The bundle follows registration recovery and
materialization JSON references, records BaSiC and deconvolution settings once
per settings cohort, and records the requested and resolved fusion settings
plus actual OME-Zarr level metadata. Large binary and image artifacts are
referenced by path and filesystem metadata; fusion does not compute new
checksums for provenance.

Position manifests may use negative spatial scales to orient a source view.
Fusion keeps those inputs Zarr-backed and represents each negative axis as a
source orientation affine before composing the registered transform. For the
sagittal L/R mode, retain the R-side negative Z/X scales in the joined position
manifest; do not pre-flip or materialize the source tiles.

## QC Checklist

- Render a center-z or z-median dumb stitch before trusting any optimizer.
- Overlay tile numbers when diagnosing bad seams or leftmost/rightmost tile
  failures.
- Compare contact sheets across z, especially near the z range used for phase
  correlation.
- For Method8 and phase recovery, inspect raw fitted values before reducing to
  a median.
- Split horizontal and vertical priors when recovering phase-correlation
  failures.
- Check whether failures are position dependent. If local Method8 shifts vary
  by tile position, report the magnitude before deciding whether a global
  translation, rigid transform, or affine is justified.
- For cross-channel alignment, use overlap correlation after applying the same
  BaSiC-treated center-z data to each model. Do not compare different
  preprocessing pipelines.

## Final Cleanup

After the fused OME-Zarr outputs, `registration.json`, `basic`, and `README`
are in the canonical output folder, remove unused caches, deconvolved
intermediates, and exploratory fit folders. Keep only artifacts needed to
reproduce the final registration and fusion, plus QC images referenced by the
README.
