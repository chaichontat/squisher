# Light-sheet workflow for 561/638/514 plus 405 datasets

Working SOP for tiled `561/638/514` acquisitions with a separate `405` series.
Dataset notebooks hold paths and results; this file holds the reusable procedure.

## Master runner

Run or resume the workflow by giving the packaged runner one representative
OME-TIFF from either acquisition:

```bash
/home/chaichontat/squisher/run-edu-workflow.sh /path/to/one.ome.tif
```

The runner identifies the three-channel acquisition and the unique sibling 405
acquisition from their OME metadata, then records the resolved inputs in
`workflow.json`. By default, outputs go beside the acquisition directories in a
`<shared-dataset-prefix>-edu-workflow` directory. Pass `--output-root PATH` only
when that location is unsuitable.

The command is resumable and must be rerun unchanged after each required human
review. It exits with status 3 and writes the next identity-bound gate file:

1. `native-thresholds.json` after rendering the four native-intensity TIFFs;
2. `fused-561-threshold.json` after rendering the fused-561 threshold TIFF;
3. `prefusion-approval.json` after recovery and pre-fusion hole QC;
4. `segmentation-approval.json` after final four-channel alignment QC.

Fill the requested positive native thresholds or change only `approved` to
`true`, then rerun the same command. The fused-561 gate inherits the accepted
native 561 threshold; its TIFF confirms the mask boundary but does not request
a second threshold value. Completed stages are validated and reused; 514 and
405 materialized windows are deleted only after their corresponding fusion is
complete. Every external command has a persistent log under `run-logs/`.

## Coordinate model

The 561 fusion defines the common physical frame and output grid.

Channel geometry follows the acquisition:

| Channel | Acquisition relationship                                            | Geometry contract                                                                                                                                                                              |
|---------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 561     | Reference channel                                                   | Estimate one internally consistent tile graph and fuse it first.                                                                                                                               |
| 638     | Co-acquired with 561 in the same scan track                         | Reuse the 561 tile positions. Refit one global 638-to-561 affine against the completed 561 fusion, initialized by the hash-pinned accepted prod2 physical affine. Do not independently optimize a 638 tile graph. |
| 514     | Stored with 561/638 but acquired through a different track          | Preserve its own same-channel tile geometry. Cross-register that acquisition to fused 561, then run the complete fused-fixed recovery before materialization.                             |
| 405     | Separate acquisition                                                | Preserve its own same-channel seam geometry, then register that acquisition into the fused 561 frame.                                                                                          |

Only 561 and 638 may share a tile registration, and 638 must still receive the
accepted 638-to-561 affine offset. Every other channel or track requires its
own tile registration followed by cross-registration.

A global channel affine is one transform in physical ZYX coordinates. Its
per-tile `registered_affine` matrices may differ after stage conjugation; those
matrices still represent one shared physical transform. An independently
optimized set of per-tile translations is a different model and must not be
substituted for it.

## Core rules

- Read axes, channels, tracks, spacing, positions, and orientation from OME
  metadata.
- Preserve native ZYX spacing and translations through the whole pipeline.
- Select thresholds from native-intensity OME-TIFFs, not display-scaled PNGs.
- Use the accepted native 561 threshold unchanged for the fused-561 mask.
- Keep BaSiC out of placement estimation and apply correction during pixel reads.
- Let held-out validation select both fit-residual field penalties.
- Treat registration as a durable input to fusion.
- Require every moving fusion to match the 561 pyramid grid exactly.
- Trust a store only when `squisher_complete=true` and provenance is complete.
- Write PNG/PDF/SVG and JSON/CSV QC; do not make an HTML index.

## Input check

Record the finalized tile count, axes, shape, dtype, spacing, TIFF layout,
compression, channel/track mapping, PSFs, total bytes, and available GPU UUIDs.
Reject partial transfers, duplicate tiles, inconsistent geometry, or missing Z
placement.

## Runtime setup

Run long commands in `tmux` from the `multi` environment with a persistent log.
Pin GPUs by UUID and verify the CUDA PID after launch. Validate `sys.executable`;
enable Bash nounset only after `conda activate`.

## 1. BaSiC, scaling, and deconvolution

Fit one dataset-specific BaSiC profile per source channel, sample corrected
scaling, then deconvolve. Do not copy residual fields or tile gains from another
specimen; reuse a PSF only when the optics match.

The prod3 baseline used 500 BaSiC samples, 100 scaling planes, one
Richardson-Lucy iteration, 100-plane multichannel slabs, 50-plane 405 slabs,
and three-level `uint16` JPEG-XR 0.7 OME-Zarr output. The reference launchers are
`prod3/run-prod3-multi.sh` and `prod3/run-prod3-405.sh`.

Require a one-to-one input/output tile mapping, correct axes and geometry,
decodable pyramid levels, the requested codec, and a completion marker.

## 2. Native dumb-stitch review and thresholds

Render native-intensity, no-blend center-Z OME-TIFFs after deconvolution. The
TIFF preserves values used by registration; the PNG is only a viewing aid.

Pause here. A human records the native threshold used by each same-channel
registration. Thresholds are dataset-specific; prod3 recorded `561=700`,
`638=700`, `514=200`, and `405=2000`, but these values are examples rather than
defaults. The 638 value is retained as native QC provenance; production 638
geometry comes from a global refit against fused 561, initialized by the
accepted prod2 affine calibration. The accepted native 561 threshold is also
the fused-561 mask threshold reviewed in step 5. The prod2 affine is a
reference, not the final dataset-specific transform.

## 3. Reference tile geometry and common grid

Create metadata position JSONs, then register 561, 514, and 405 within their
native acquisitions. The 638 channel shares the 561 acquisition geometry.
Preserve the 514 and 405 same-channel registrations as the source geometry for
later materialization; neither acquisition enters the 561 coordinate system
until cross-registration in step 7.

Derive a metadata-only sampling grid from each native acquisition registration
for fit-residual fitting. Separately derive the final common output grid from the
accepted 561 registration's physical tile extents. Validate and record every
grid's level-0 shape, spacing, and origin. The 561 grid becomes the fusion
template, but it does not place 514 or 405 into 561 coordinates before their
cross-registration.

The registration must contain a complete level-2 pair-by-Z-chunk screen and
must report accepted, recovered, rejected, and disconnected constraints.
`--allow-disconnected` is an explicit acceptance of incomplete graph
connectivity, not a recovery method. Inspect the disconnected tiles and the
optimized no-blend stitch before using the result.

Do not optimize a same-channel 638 tile graph for production. The physical
model calls for the 561 tile positions followed by one global channel-affine
refit in step 6.

## 4. Run fit-residual correction for every channel

After deconvolution and establishment of each channel's native acquisition
geometry, fit residual intensity correction for 561, 638, 514, and 405. This
stage is mandatory and precedes all cross-registration and fusion. Use each
channel's original full CZYX sources, native acquisition registration, and
metadata-only sampling grid. The fitted correction remains in original source
coordinates and is applied during later pixel reads; fit-residual does not place
514 or 405 into the 561 frame.

The residual fitter rejects materialized crops because they do not preserve
raw-source coordinates. Do not defer this stage until window recovery and do
not pass a fused-fixed recovery summary where a source-tile registration is
required.

Leave both `--field-penalty` and `--source-field-penalty` unset so they are
selected independently on held-out source pairs. Review held-out-plane
performance separately, together with the fitted fields, seam comparisons,
held-out errors, and support coverage. The prod3 settings were source level 2,
stride 4, XY/Z degree 1, eight workers, and a recorded seed.

## 5. Fuse the 561 reference

Fuse 561 first with its accepted registration and residual correction. Use
`content-preibisch-coarse`, automatic batching, no source-tile cache,
`(24,960,960)` output chunks, and JPEG-XR 0.7. The package-owned coarse
Preibisch defaults are sigma 1/2 of `7/17`, stride ZYX `(1,8,8)`, and softmax
exponent `2.0`; record those values and the fusion producer hash with the run.
The public CLI currently selects the mode but does not expose those four
parameters, so do not imply that they were supplied as command-line options.

```bash
CUDA_VISIBLE_DEVICES=<GPU_UUID> lightsheet fuse <SOURCE_TILES> \
  --position-input <POSITIONS> \
  --registration-input <REGISTRATION> \
  --output <VERSIONED_FUSION_DIR> \
  --channel <SOURCE_CHANNEL> \
  --residual-correction <FIT_RESIDUAL_CORRECTION> \
  --fusion-level 0 \
  --fusion-weight-mode content-preibisch-coarse \
  --batch-size auto \
  --source-cache-max-gib 64 \
  --output-chunksize-zyx 24,960,960 \
  --output-grid-template <561_GRID_OR_COMPLETED_FUSION> \
  --output-grid-template-level 0 \
  --output-codec jpegxr \
  --jpegxr-level 0.7 \
  --resume-fusion
```

Record the completed store's shape, spacing, translation, chunks, codec, and
provenance. Every later fusion uses it as the level-0 grid template.

Before any fused-fixed registration, render level 2 from this exact completed
fusion as an unscaled native-intensity OME-TIFF and persist the accepted native
561 threshold as the fused-mask threshold with that artifact:

```bash
lightsheet threshold-review \
  --fixed-fused <FUSED_561> \
  --output <FUSED_561_THRESHOLD_REVIEW>.ome.tif \
  --level 2
```

The fused threshold is not selected independently: use the same numeric value
accepted for native 561. Review the fused TIFF as QC for the resulting mask
boundary and record approval against the exact completed fusion identity.

## 6. Refit the 638 affine from the accepted reference and fuse

Use the accepted prod2 physical 638-to-561 calibration as the initializer for a
dataset-specific global refit against the completed fused 561. Bind the
initializer to the reviewed calibration artifact's exact hash and transform
direction. Pass its physical ZYX 3x3 linear block to Method 6; the packaged
level-2 phase initializer continues to own translation.

```bash
test "$(sha256sum <PROD2_638_TO_561_CALIBRATION> | cut -d' ' -f1)" = \
  <PROD2_CALIBRATION_SHA256>

CUDA_VISIBLE_DEVICES=<GPU_UUID> lightsheet cross-register method6 \
  --fixed-position <561_POSITIONS> \
  --moving-position <561_POSITIONS> \
  --moving-source-position <561_REGISTRATION> \
  --fixed-fused <FUSED_561> \
  --output-dir <638_REFIT_DIR> \
  --output-registration <638_REFIT_DIR>/registration.json \
  --moving-channel 1 \
  --fit-intensity-transform log1p \
  --starting-affine-matrix-zyx <PROD2_638_LINEAR_MATRIX_ZYX> \
  --min-corr 0.15 \
  --min-grad-ncc 0.24 \
  --fixed-mask-threshold <REVIEWED_FUSED_561_THRESHOLD> \
  --source-label 638 \
  --target-label 561 \
  --workers 1 --devices 0 --resume
```

Require the reference calibration to declare the 638-to-561 direction, source
channel 1, and a finite invertible physical affine. The completed refit summary
must record the exact reference-derived starting matrix and the exact fused-561
identity. Its final registration must cover every 561 reference tile and derive
one finite invertible physical affine from supported windows. No independently
optimized per-tile 638 positions may enter this stage.

Fuse channel 1 directly from the deconvolved multichannel tiles using the 561
positions, the refitted global registration, the 638 fit-residual correction
from step 4, and the same coarse-Preibisch/output settings as 561. Never use a
separately optimized 638 position file for this step.

## 7. Register and recover 514 and 405

Both channels run the full cross-registration and recovery sequence:

1. Start from the channel's accepted native same-channel positions and
   registration. Estimate a global phase/affine initialization into fused 561
   space; this is the first point at which 514 or 405 uses 561 coordinates.
2. Render native overlays across XY, XZ, and YZ.
3. Run fused-fixed Method 6 windows at level 2 and level 0. A good global
   initializer does not skip this stage.
4. Rerun eligible rejected original windows from leave-one-out or spatial-
   consensus initializers. Graph-adjacent good tiles may supply transform
   samples when local support is insufficient; recovery does not enlarge the
   fit into a full adjacent-face patch. Keep the interpolated initializer
   matrix; do not replace it with a CLI default.
5. Apply the physical linear-field filter, retained-affine-outlier replacement,
   spatial fallback promotion, and final linear-mapping recovery in that order.
   A fitted transform flagged by retained-affine-outlier replacement or by the
   linear-field filter is replaced by the field prediction only when the image
   favors it. Both are scored by zero-shift high-passed NCC against fused 561 on
   shared level-2 planes, and the prediction must win by
   `--replacement-image-ncc-margin` (0.02). Otherwise the fitted transform is
   kept. Spatial fallback promotion keeps a native rerun rejected only by the
   correlation or gradient-NCC gate when its gradient NCC beats the interpolated
   initializer's by `--native-rerun-grad-ncc-margin` (0.05), or when it failed
   only correlation while meeting the gradient-NCC minimum.
6. Atomically publish one immutable final summary whose window rows and
   aggregate counts agree exactly, together with its stage reports and identity
   manifest.

Recovery and complete-recovery are mandatory even when the global initializer
appears accurate. Promote an interpolated fallback only when image evidence
supports it. Clear signal rejected by the fitter is a recovery bug or
quality-gate problem; it is not evidence that the tissue is absent.

Curated window initializations live in one dataset master file,
`<DATASET_ROOT>/manual-initializations.json`. Use it when a tissue window's
automatic starting point is wrong and a seed has been read from the image. Seed
tissue windows only; background windows are not worth rescuing. The master file
has the following structure:

```json
{
  "artifact_type": "lightsheet.manual_window_initializations.v1",
  "channels": {
    "514": {
      "<window JSON name>": {
        "moving_tile": "...", "moving_start_l0_zyx": [...],
        "local_matrix_zyx": [[...]], "local_translation_zyx": [...],
        "seed_origin": "manual | measured", "evidence": "where the seed came from"
      }
    }
  }
}
```

Transforms are absolute level-0 window-local models, in the same coordinates as
`recover-native` initializers, not deltas. `recover-native` refuses an entry
whose tile or level-0 moving start no longer matches the window, and it refuses any entry that names a missing or mask-rejected window.
When the file exists, the EDU workflow passes it to `recover-native` together with
the channel label. `run-edu-xreg-recovery.sh` does the same when
`MANUAL_INITIALIZATIONS` is set.

`recover-native` then reruns Method 6 from each curated seed. Complete-recovery
applies its usual image checks: a curated seed that the image contradicts is
still withheld. Method 6 (or the retained original fit when the rerun is not
selected) can leave a verified seed. If the residual check does not verify the
accepted fit of a curated window but does verify its seed, complete-recovery
publishes the seed. This arbitration is its last stage, after linear-mapping
recovery, and the row records it under `manual_seed_arbitration`. The stage identity does not cover the file's contents, so after
editing it, remove the `recover-native-<label>` and `complete-recovery-<label>`
stage markers and outputs, and the pre-fusion QC markers, before rerunning.

Produce the Method 6 summary with an explicit coordinate handoff. `global-phase`
moves the channel's native positions into the 561 frame; Method 6 uses those
cross-registered positions for placement while reading pixels and adjacency
from the unchanged native source registration:

`global-phase` scores each candidate XY shift by normalized cross-correlation
over the pixels both center-Z canvases cover, after subtracting each canvas's
coverage-weighted local mean (`--highpass-sigma-px`, in canvas pixels). Whole-
canvas phase correlation is not used: 405 shares the multichannel acquisition's
X stage lattice, so mosaic outlines and per-tile brightness steps coincide at
zero X shift and outscored tissue in prod3-2 (x = 0 instead of about +46 um).

Use the highest pyramid level present in both fixed and moving sources for the
rough fit. This was level 4 in prod2 and level 2 in prod3. The channel-specific
fit contract reproduced in both datasets is:

| Channel | Rough transform | Method 6 transform | Starting matrix | Minimum gradient NCC |
|---|---|---|---|---:|
| 514 | `log1p` | `log1p` | `STARTING_MATRIX_514` | 0.24 |
| 405 | `identity` | `linear` | `STARTING_MATRIX_405` | 0.20 |

```bash
STARTING_MATRIX_514=0.9999090828,-0.0083501824,-0.0029757332,0.0141554887,0.9999284579,-0.0096124636,0.0026881526,0.0099598448,0.9999495898
STARTING_MATRIX_405=1.0000085830688477,0.0029159076511859894,-0.0002120283606927842,0.0029159076511859894,1.0000085830688477,0.00018651569553185254,-0.0002120283606927842,0.00018651569553185254,1.0000001192092896
```

```bash
lightsheet cross-register global-phase \
  --fixed-position <561_POSITIONS> \
  --moving-position <CHANNEL_NATIVE_POSITIONS> \
  --fixed-tile-dir <561_DECONVOLVED_DIR> \
  --moving-tile-dir <CHANNEL_DECONVOLVED_DIR> \
  --output-dir <CHANNEL_GLOBAL_PHASE_DIR> \
  --output-position <CHANNEL_GLOBAL_PHASE_DIR>/global-phase.positions.json \
  --fixed-channel 0 \
  --moving-channel <CHANNEL_INDEX> \
  --level <HIGHEST_COMMON_PYRAMID_LEVEL> \
  --z-slab-planes 32 \
  --fixed-intensity-transform log1p \
  --moving-intensity-transform <CHANNEL_ROUGH_INTENSITY_TRANSFORM> \
  --highpass-sigma-px 8

CUDA_VISIBLE_DEVICES=<GPU_UUID> lightsheet cross-register method6 \
  --fixed-position <561_POSITIONS> \
  --moving-position <CHANNEL_GLOBAL_PHASE_DIR>/global-phase.positions.json \
  --moving-source-position <CHANNEL_NATIVE_REGISTRATION> \
  --fixed-fused <FUSED_561> \
  --output-dir <METHOD6_DIR> \
  --output-registration <METHOD6_DIR>/registration.json \
  --moving-channel <CHANNEL_INDEX> \
  --fit-intensity-transform <CHANNEL_METHOD6_INTENSITY_TRANSFORM> \
  --starting-affine-matrix-zyx <CHANNEL_STARTING_AFFINE_MATRIX_ZYX> \
  --min-corr 0.15 \
  --min-grad-ncc <CHANNEL_MIN_GRAD_NCC> \
  --fixed-mask-threshold <REVIEWED_FUSED_561_THRESHOLD> \
  --source-label <CHANNEL> \
  --target-label 561 \
  --workers 1 --devices 0 --resume
```

The recovery input below is
`<METHOD6_DIR>/fused_fixed_method8_summary.json`. Do not substitute the native
positions for the global-phase output or the global-phase output for
`--moving-source-position`; those files own different coordinate contracts.

For a measured local nuclear residual, `method6 --local-3d-refinement`
adds an equally weighted affine fit to a geometry-fixed grid of 3D patch
correspondences and reserves opposite-parity patch centers for internal consistency checks.
The original native, correlation, gradient and affine gates still apply.
Every previously confident validation patch must remain confident and have
residuals within one capture voxel per axis. The capture grid uses native Z
and twice native XY sampling. A rejected local fit stays rejected. The
criterion has a distinct cache identity, and native diagnostics separately
record evaluation-limit exhaustion.

For a restrained candidate, add `--local-affine-estimator restrained` to
`--local-3d-refinement`. The fixed policy applies a dimensionless 0.01 SD
Gaussian prior to the physical linear shift field, leaves translation free,
and uses robust Huber residuals with radius 2 after scaling by capture spacing.
That residual scale is a working convention, not a calibrated uncertainty.
Correspondences use actual fixed-mask support centroids; at least six
noncoplanar fitting points and normalized design condition at most 100 are
required. Unsupported windows remain unresolved, and solver failures fail
explicitly. The complete policy is recorded in the cache and diagnostics.
`least-squares` remains the original prototype comparator and default.

Parity patches can share image content and establish internal consistency.
Evaluate a frozen configuration on separate nuclear regions, same-channel
overlap regions and blend boundaries across representative datasets before
adopting it as a general method. Use a fixed physical window geometry for
that comparison; manual explicit-domain mixtures remain a rescue facility.

When the displacement bends within a fitting window, use smaller recorded
domains with `--core-shape-zyx` and `--window-shape-zyx`, together with
`--parent-window-dir <ACCEPTED_NATIVE_WINDOW_JSON_DIR>`. Each child must lie
inside exactly one accepted parent from the same source, reference and
channel. Its initializer preserves the parent's global index mapping and
rebases it into the child's crop coordinates. A crop-centred translation
cannot be copied between domains. Review exact-input masks anew before each
run and record the chosen threshold.

For full axial refinement over the accepted production domains, add
`--split-parent-windows`. It generates child starts relative to each accepted
parent, including nonaligned tail parents, and records the explicit parent
initializer. Each parent's XY bounds must match the requested window; the
axial core cannot exceed the child window. The child union preserves the
parent's source domain. Duplicate child output identities fail explicitly.
This option requires `--parent-window-dir`; the default native tile grid and
its requirement for exactly one containing parent remain unchanged. Tiles
without accepted parents do not contribute tasks in this mode.

Validate the resulting individual contributions against 561, same-channel
neighbors and child-overlap boundaries before materialization. The final
window manifest must replace superseded parents and preserve transformed
tissue coverage. The CLI's per-channel median registration is a candidate
artifact; it does not replace this per-window fusion contract.

For a bounded validation of accepted local refinements, assemble explicit
source domains with `lightsheet cross-register assemble-bounded-domains
--plan <DOMAIN_PLAN> --output-dir <DOMAIN_DIR>`. The plan names each published
parent, the accepted transform, exact source start/stop coordinates, quadrant
provenance, and either `validated_refinement` or
`retained_published_mapping`. Only axial subdivisions with unchanged XY
bounds are supported. The assembly preserves every parent's source-domain
coverage and excludes retained superseded mappings from the protected fixed
field. Retaining a published mapping does not certify a rejected refinement.

Materialize the resulting `fused_fixed_explicit_domain_summary.json` with
`lightsheet fused-fixed-materialize-overlap --use-explicit-domains
--source-summary <SUMMARY>`, using the native registration positions and
source registration. This mode samples each declared crop exactly; it does
not expand domains back to uniform windows. For the validation fusion, pass
the actual fixed image to `lightsheet fuse --output-grid-template <FIXED>
--output-grid-bounds-zyx z0,y0,x0,z1,y1,x1`. Bounds are half open template
indices; the output retains the template spacing and records the cropped
physical origin. Check the actual blend, coverage and both transition
collars before publishing a source replacement. Explicit-domain assembly
alone establishes geometry and scope, not final blended image quality.

Use the packaged recovery owners:

```bash
lightsheet cross-register recover-native \
  --summary <METHOD6_SUMMARY> \
  --output-dir <NATIVE_RECOVERY_DIR> \
  --adjacency-json <MOVING_REGISTRATION_MEASUREMENTS> \
  --devices <CUDA_DEVICE_INDICES>

lightsheet cross-register complete-recovery \
  --native-summary <NATIVE_RECOVERY_DIR>/fused_fixed_method8_summary.json \
  --output-dir <COMPLETE_RECOVERY_DIR>
```

With `CUDA_VISIBLE_DEVICES` set to one or more UUIDs, `--devices` names local
indices within that mask; without a mask, it names physical GPU indices.

`complete-recovery` accepts a completed native Method 6 or Method 10 Mattes checkpoint. It
keeps intermediate stages temporary and reuses a completed output only when
the input identity and all published hashes still match. Existing prod3 v4
outputs created by the earlier dataset-local implementation are immutable
historical artifacts: validate their stored manifests and hashes rather than
recomputing their identity with current producer code.

Before materialization, render a no-blend middle-Z stitch, a coverage map, and
contact sheets for interpolated windows. Measure uncovered pixels inside the
reviewed 561 tissue mask and report the uncovered fraction and largest hole.
Approval is bound to the exact recovery identities.

The coverage numbers do not show alignment. Near-zero uncovered tissue can
coexist with recovered windows that are 5–10 µm off. Open the interpolated-window
contact sheets and look at the windows directly. Doubled nuclei or edges at zero
shift that merge at a small shift mean the window is misaligned. Separate
windows that contain tissue from background, haze, or compression-artifact
windows: only tissue windows affect the fused result. Each cell stretches its
own contrast, so a red- or green-dominant cell is not by itself evidence of
misalignment. When a metric and the image disagree, render the fixed and moving
planes separately next to the overlay before deciding.

```bash
lightsheet cross-register prefusion-qc \
  --summary <CHANNEL>=<COMPLETE_RECOVERY_DIR>/fused_fixed_method8_summary.json \
  --fixed-volume <FUSED_561> \
  --fixed-threshold <REVIEWED_FUSED_561_THRESHOLD> \
  --output-dir <PREFUSION_QC_DIR>

lightsheet cross-register interpolated-window-qc \
  --summary <COMPLETE_RECOVERY_DIR>/fused_fixed_method8_summary.json \
  --moving-label <CHANNEL> \
  --fixed-label 561 \
  --output <PREFUSION_QC_DIR>/interpolated-<CHANNEL>.png
```

After approval, materialize accepted windows at level 0 from the final recovery
summary and original native source geometry:

```bash
lightsheet fused-fixed-materialize-overlap \
  --moving-position <CHANNEL_NATIVE_POSITIONS> \
  --moving-source-registration <CHANNEL_NATIVE_REGISTRATION> \
  --moving-source-dir <CHANNEL_DECONVOLVED_DIR> \
  --source-summary <COMPLETE_RECOVERY_DIR>/fused_fixed_method8_summary.json \
  --output-dir <MATERIALIZED_LEVEL0_DIR> \
  --source-channel <CHANNEL_INDEX> \
  --residual-correction <CHANNEL_FIT_RESIDUAL_CORRECTION> \
  --core-shape-zyx 480,480,480 \
  --window-shape-zyx 528,528,528 \
  --level-factor-zyx 1,1,1 \
  --output-codec zstd \
  --zstd-level 3 \
  --workers 4 --resume
```

`1,1,1` is also the packaged default; spelling it out here documents the
production level-0 invariant. Require one materialized tile per accepted
window, then fuse onto the 561 grid with the coarse-Preibisch settings from
step 5, adding `--fusion-gpu-memory-max-gib 18`: materialized windows overlap
densely, and unsplit output chunks have needed up to 35 GB of GPU memory. The
budget splits only the GPU computation in x/y and is not resume state. Remove only that channel's exact materialization after its completed
fusion passes grid, dtype, pyramid, chunk, codec, and completion validation.

## 8. Final QC

After each fusion, render its same-grid middle-Z PNG. After all channels are
complete, render them together and request the combined contact sheet
explicitly. Also render native red/green overlays against 561 across XY, XZ,
and YZ. Keep the JSON sampling record with the images.

```bash
lightsheet fused-middle-z-qc \
  --channel 561=<FUSED_561> \
  --channel 638=<FUSED_638> \
  --channel 514=<FUSED_514> \
  --channel 405=<FUSED_405> \
  --fixed-volume <FUSED_561> \
  --output-dir <QC_DIR> \
  --level 2 \
  --contact-sheet-name combined-middle-z.png
```

Use the packaged overlay renderer:

```bash
lightsheet fused-xyz-overlay-qc \
  --reference-zarr <FUSED_561> \
  --moving-zarr <FUSED_CHANNEL> \
  --output-dir <QC_DIR> \
  --level 0 --thumb-level 2 --panel-size 512
```

A constant red/green offset suggests a global-transform error. An offset that
changes across panels suggests the wrong model, direction, or local field. Good
seams do not prove cross-channel alignment, and different labels need not be
yellow; judge shared structures and tissue boundaries.

## 9. Cellpose segmentation and paired export

Start segmentation only after the 405, 561, and 638 fusions are complete on the
same grid and their alignment QC is accepted. A refusion creates new source
identities even when its geometry is unchanged; rerun segmentation and export
when the canonical fusion branch changes rather than attributing old labels to
the new pixels. Write a registered-source JSON
manifest whose sources are the exact fused stores, including each root metadata
hash, channel order `405,561,638`, and a source ROI covering the whole fused
brain. The label volume must use the same ZYX shape as that ROI. Keep 514 out of
the Cellpose input unless a separately reviewed model contract adds it.

Cellpose production runs MUST cover the whole brain represented by the fused
405/561/638 stores; do not treat a representative XY subset as a completed
run. A smaller full-Z ROI is allowed only for QC or profiling, and its labels
must not be used as the production result. The prod2/prod3 baseline uses
`embryonicsheet` for both XY and orthogonal inference,
diameter 30, flow smoothing 1.0, and orthogonal weights `(1,1,1)`. The persisted
run identity must also record batch size 1, flow threshold 0, cell-probability
threshold 0, anisotropy 1.0, 1000 iterations, minimum size 500, 60-voxel block
overlap, unsharp preprocessing, and overlap-IoU stitching at 0.25:

```json
{
  "backend": "sam",
  "diameter": 30,
  "flow3D_smooth": 1.0,
  "ortho_weights": [1.0, 1.0, 1.0],
  "pretrained_model": "/working/cellpose-training/models/embryonicsheet",
  "pretrained_model_ortho": "/working/cellpose-training/models/embryonicsheet"
}
```

Build or validate a TensorRT plan for each physical GPU model being used; a plan
from another GPU model is not interchangeable. Pin GPUs by UUID. The accepted
prod3 allocation uses the RTX 5090 and RTX 4090 with one worker on each device.
Choose target tiles and block sizes that cover the full-brain manifest without
silently narrowing its source ROI.

Apply the nonempty test to raw 561 and inherit its value from the accepted native
561 threshold in `native-thresholds.json`; do not choose a separate segmentation
threshold. A core is scheduled only when at least 5% of its raw 561 voxels are
strictly greater than that value. The accepted prod3 value is 700; prod2 used
1000 because native thresholds are dataset-specific:

```bash
CUDA_VISIBLE_DEVICES=<RTX_5090_UUID>,<RTX_4090_UUID> \
squisher-segment segment run <REGISTERED_FULL_BRAIN_JSON> \
  --channels 405,561,638 \
  --config <SEGMENT_CONFIG_JSON> \
  --workers-per-device 1,1 \
  --threads-per-worker 1 \
  --target-nz <TARGET_NZ> --target-ny <TARGET_NY> --target-nx <TARGET_NX> \
  --nonempty-threshold <ACCEPTED_NATIVE_561_THRESHOLD> \
  --no-cellpose-only \
  --stagger-seconds 0
```

Require `output_segmentation-sam.done`. Verify the persisted source ROI, model
hashes, inference parameters, TensorRT device plans, and raw label shape/dtype;
the raw labels must be a ZYX `uint32` array. A matching model pathname is not
sufficient when its checkpoint hash differs. Schedule postprocessing directly
after segmentation in the same tmux-owned `set -e` script so it starts only
after segmentation exits successfully, then postprocess once:

```bash
CUDA_VISIBLE_DEVICES=<RTX_5090_UUID>,<RTX_4090_UUID> \
squisher-segment postproc run \
  <RUN_DIR>/output_segmentation-sam.zarr \
  --output-path <RUN_DIR>/postproc.zarr \
  --sigma 1,2,2 --v-min 500 --margin 30 --workers-per-gpu 1
```

Before launch, verify that the runtime can import both Numba and the packaged
postprocessor and that its NumPy version satisfies the package constraint. Do
not rerun completed Cellpose inference to repair a postprocessor environment
failure. Resume from the raw labels and require `postproc.zarr` to match their
shape and `uint32` dtype.

Use only completed postprocessed labels as export masks; never export raw
`output_segmentation-sam.zarr` labels. Reproduce the paired whole-brain export
with deterministic sampling:

```bash
squisher-segment extract <REGISTERED_FULL_BRAIN_JSON> \
  --mode z --out <RUN_DIR>/extract-z512-full \
  --n 50 --dz 1 --upscale 1 \
  --channels 405,561,638 --threads 8 --seed 42 --label <DATASET_LABEL> \
  --masks <RUN_DIR>/postproc.zarr

squisher-segment extract <REGISTERED_FULL_BRAIN_JSON> \
  --mode ortho --out <RUN_DIR>/extract-o512-full \
  --n 50 --anisotropy 1 --ortho-depth 512 --upscale 1 \
  --channels 405,561,638 --threads 8 --seed 42 --label <DATASET_LABEL> \
  --masks <RUN_DIR>/postproc.zarr
```

For whole-brain registered-source inputs, read sampled export regions directly
rather than materializing a full staging array unless repeated downstream reads
justify the extra storage and I/O. Never clear an existing export directory during a rerun;
preserve user-authored `*_seg.npy` annotations and overwrite only deterministic
generated TIFF pairs.

An export is complete only when every image TIFF has a same-basename
`_masks.tif`, every image is channel-first `uint16` with channel metadata
`405,561,638`, every mask is `uint32`, and each image/mask spatial shape agrees.
Record the registered manifest, exact postprocessed mask path, parameters,
pair counts, and output directories in a machine-readable completion marker.

## Ventricular Eomes contour review

Build the ventricular chart from the reviewed cortex mask, then derive the
canonical closest-surface 561 profiles and ordered Eomes boundaries. Each
command writes a fresh artifact directory; do not mix files from separate runs.

```bash
lightsheet uv-surface \
  --mask <CORTEX_MASK_NRRD> \
  --output-dir <SPATIAL_DIR>/vz-surface

lightsheet eomes-profiles \
  --surface <SPATIAL_DIR>/vz-surface/surface.npz \
  --chart <SPATIAL_DIR>/vz-surface/chart.npz \
  --mask <CORTEX_MASK_NRRD> \
  --source-zarr <ACCEPTED_561_FUSION> \
  --projection-domain ventricular-wall \
  --output-dir <SPATIAL_DIR>/eomes-profiles

lightsheet eomes-boundaries \
  --profiles <SPATIAL_DIR>/eomes-profiles/profiles.npz \
  --surface <SPATIAL_DIR>/vz-surface/surface.npz \
  --min-crossing-samples 5 \
  --max-recovery-distance-um 20 \
  --output-dir <SPATIAL_DIR>/eomes-boundaries

lightsheet eomes-boundary-qc \
  --surface <SPATIAL_DIR>/vz-surface/surface.npz \
  --chart <SPATIAL_DIR>/vz-surface/chart.npz \
  --boundaries <SPATIAL_DIR>/eomes-boundaries/boundaries.npz \
  --mask <CORTEX_MASK_NRRD> \
  --source-zarr <ACCEPTED_561_FUSION> \
  --output-dir <SPATIAL_DIR>/eomes-boundary-qc \
  --grid-stride 1 \
  --intrinsic-z-index <Z0> \
  --intrinsic-z-index <Z1> \
  --intrinsic-z-index <Z2>
```

The numerical defaults reproduce the prod2 estimator: level 2, sampling stride `2,4,4`,
20 µm mesh-geodesic pooling, 4.8 µm depth centers, rising 0.60/30 µm inner
fit, and falling 0.75/80 µm ordered outer fit. Review the direct and recovered
contours in the PNG, PDF, or SVG before projecting cells or assigning layers.
The explicit ventricular-wall projection domain excludes end-cap faces before
nearest-surface assignment; downstream boundary fitting and QC inherit the
same hash-bound vertex mask.
Boundary recovery uses crossings supported by at least five pooled samples and
is reported only within 20 µm mesh-geodesically of such evidence. QC also
invalidates grid points adjacent to discontinuous closest-wall branch switches.
Consequently, weak terminal regions may remain gapped or unassigned; do not
join those gaps or interpret them as measured layer boundaries.
Those downstream stages remain deferred until this QC gate passes.
Keep each profile artifact with its sibling `metadata.json`; boundary fitting
verifies that it belongs to the supplied surface before solving.

## Completion checklist

A channel is complete only when all applicable checks pass:

- [ ] Inputs, channel mapping, spacing, and PSFs recorded.
- [ ] Deconvolved tiles complete and decodable.
- [ ] Native TIFF reviewed and thresholds recorded.
- [ ] Reference geometry and residual correction accepted.
- [ ] Fused-561 threshold QC approved and the native 561 mask value persisted.
- [ ] Channel transform matches the acquisition model.
- [ ] Recovery windows and coverage holes reviewed where applicable.
- [ ] Fusion complete, provenance present, and every level matches 561.
- [ ] Middle-Z and XYZ overlay QC reviewed.
- [ ] Downstream paths use only the accepted fusion.
- [ ] Cellpose source manifest is hash-bound to accepted 405/561/638 fusions.
- [ ] Raw and postprocessed label identity, shape, dtype, and parameters pass.
- [ ] Paired Z and orthogonal exports use postprocessed masks and validate.
- [ ] Ventricular/Eomes direct and recovered contour QC is reviewed before layer assignment.

## Cleanup policy

- Production orchestration removes its exact materialized-window directory
  automatically only after the corresponding final fusion passes completion,
  pyramid, and grid validation. `lightsheet fuse` itself does not delete its
  inputs.
- Keep registrations, final windows, correction models, provenance, approvals,
  logs, and referenced QC.
- Delete superseded fusions only after checking live readers. Preserve incomplete
  resumable fusions.
- Delete raw TIFFs only with explicit authorization after one-to-one validation
  of their deconvolved replacements.

## Scheduling and performance

- Give each long stage one tmux session and persistent log.
- Validate each producer before its consumer starts.
- Pin GPUs by UUID and verify the CUDA PID.
- Avoid concurrent I/O-heavy stages on the same storage pool.
- Use automatic batching unless measured memory pressure requires a limit.
- Monitor progress markers, output growth, GPU ownership, RSS, and I/O.

## Evidence and implementation references

- `/working/eduseg/prod3/README.md` and
  `/working/eduseg/20260808-prod2/README.md`: dataset evidence.
- `/home/chaichontat/squisher/lightsheet/WORKFLOW.md`: package contracts.
- Accepted prod2 638-to-561 initializer:
  `/working/eduseg/20260808-prod2/234-3-514561638/reg-638-to-561/registration.json`,
  SHA-256 `ee7820a860b1d247f8116d86fe98206cbcf19ee93957b35e88706179e1b5e4f3`.
- `/working/eduseg/prod3/run-prod3-postnative-v4.sh` and
  `/working/eduseg/prod3/run-prod3-after-v4-approval.sh`: recovery, review, and
  fusion pattern.

When an implementation and this SOP disagree, stop before producing a new
fusion. Resolve the coordinate contract first, update the owning code, and then
record the change here and in the dataset notebook.


### Assign cells after Eomes boundary review

Use `lightsheet eomes-layers --coordinates-dir COORDINATES --surface SURFACE.npz
--boundaries BOUNDARIES/boundaries.npz --output-dir LAYERS` after reviewing the
direct and recovered QC. The coordinate and boundary artifacts must identify the
same surface. VZ is `r_vz <= inner`, SVZ is `inner < r_vz <= outer`, and the
remainder is CP. Unsupported or non-tissue-side projections have
`boundary_defined=false`; the CP remainder does not imply supported boundaries.
The layer metadata records the actual inner/outer fractions and smooth lengths
from the boundary producer, rather than assuming its default 60/75 settings.
