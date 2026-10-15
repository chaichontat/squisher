# HCR Lightsheet Workflow

The post-deconvolution intensity-correction stage is named `fit-residual`.
Legacy artifact folders and implementation identifiers may retain older names.

All HCR image data use approximately `1.8 um` Z spacing (`1.807 um` in current
acquisitions). For a `0.3 um` Z-sampled PSF, use the centered `zstride6_xy9`
kernel before deconvolution; never use the stride-2 `zcenter2_xy9` kernel.

HCR samples skip 405: leave the separate 405 (and DAPI) acquisitions out of the
manifest, and do not deconvolve, register, or fuse them. `lightsheet
hcr-workflow` refuses a manifest that lists 405. The "Separate 405
acquisition" section records how 229Tnc and 230Tnc were registered before this
decision.

## Reference-frame contract

For multichannel HCR acquisitions, channel 561 defines both the shared-track
tile geometry and the final biological reference frame. Do not substitute a
stronger channel's independently optimized tile graph for the 561 geometry.

For the `488514561638` Zeiss acquisition used by 230Tnc and 229Tnc, use the
parsed track metadata rather than the wavelength order in the filename:

- track 0 contains channel 0 = 561 and channel 1 = 638;
- track 1 contains channel 2 = 514;
- track 2 contains channel 3 = 488.

Only 561 and 638 may share a tile registration, and 638 must still receive the
accepted 638-to-561 affine offset. Every other channel or track requires its
own tile registration followed by cross-registration.

Follow this production contract:

Use the zarr-backed direct OME-Zarr fuser for every production fusion. Use an
in-memory, TIFF, or non-zarr backend only when the user requests it.

1. Solve the shared track-0 tile geometry from channel 0 / 561. Reuse that tile
   geometry for both 561 and 638.
2. Define channel 0 / 561 as the fixed coordinate frame.
3. After accepting native geometry, run and review a channel-specific
   fit-residual correction for every side and channel before cross-channel
   registration or production fusion. Use the shared 561 geometry to sample
   561 and 638; use each channel's native geometry for 514 and 488. Apply each
   fit-residual correction exactly once, either when reading deconvolved tiles
   for direct fusion or when materializing registered windows. Never reapply it
   when fusing materialized stores.
   Review actual native tile boundaries as well as held-out scores. Residual
   validation aggregates absolute pixel errors within each source-pair plane;
   signed bright/dark discrepancies must not cancel. A curved edge profile may
   require `--xy-degree 2`, but improved overlap scores alone do not validate
   intensity inside tiles. Inspect native previews for repeated camera-shaped
   modulation and, where independently illuminated same-channel coverage exists,
   compare held-out spatial blocks with only a common scale before accepting a
   more flexible field. Reject a field that worsens that independent comparison.
   For a mismatch demonstrated in raw native camera edges, `fit-residual-edges`
   can refine the accepted native correction with four bounded quadratic edge
   ramps, keeping its interior field and source gains fixed. Choose the width
   from native overlap profiles, then review all sampled depths and independent
   same-channel intensity with a scale learned from unchanged interiors.
4. Fuse the two 561 datasets separately for cross-dataset registration. These
   dataset-local mosaics are registration inputs, not the canonical final grid.
5. Select the cross-dataset placement mode from the acquisition layout; CL/CR
   and TL/TR are different geometries and are not interchangeable:
   - For CL/CR, keep CL fixed and CR moving and use
     `lr_z_endview_flip_xz`. Join along Z and preserve CR's negative Z/X
     orientation.
   - For TL/TR, keep TL fixed and TR moving and use
     `tltr_x_join_center_z_phase`. Join along X and do not apply the CL/CR Z/X
     flips.
   Phase-correlate the actual 561 overlap in the selected geometry, then run
   Method8 in spatial-overlap mode (`--pair-mode spatial-overlap`) and add the
   median accepted local translation as the fine correction to the
   phase-correlation shift. Pass `--preseed-matrix-zyx 1,0,0,0,1,0,0,0,1`
   explicitly: the generic Method8 command defaults to a different dataset's
   calibration tilt, which is not an HCR side-placement initializer.
   Do not fit a global cross-dataset affine unless
   reviewed QC demonstrates that translation is inadequate. For a reviewed TL/TR
   Method8 affine, set `reviewed_global_affine` in `placement-coarse.json` to
   `{"path": "<method8>/global_affine.json", "sha256": "<reviewed artifact SHA-256>"}`.
   The publisher retains the coarse source stage placement and applies the full
   physical affine to TR through `registered_affine`, replacing the median fine
   translation. Review the resampled XY/XZ planes in `placement/final/qc-affine`;
   final approval binds both positions and registration. 638 composes its
   calibration onto that registration. 514/488 measure their native offsets in
   coarse coordinates, then compose the reviewed world placement into the
   fused-fixed initializer before refinement and materialization.
6. After accepting the layout-appropriate cross-dataset transform, fuse the
   combined 561 volume.
   This completed multiscale volume is the sole canonical grid template for
   every final channel. Use the accepted native 561 threshold unchanged as the
   fused-561 mask threshold. Render an unscaled native-intensity threshold TIFF
   from the completed fusion and review the resulting mask boundary as QC, but
   do not select a second threshold value. Bind that approval and threshold to
   the exact completed fusion identity before fused-fixed registration.
7. For 638, reuse the matching 561 position and placement record for every
   same-source tile in the accepted combined placement. Apply one
   accepted physical 638-to-561 affine, stage-conjugated into every 561 tile
   registration. Do not invert or refit that affine, and do not compose an
   independently optimized 638 tile graph; a 638-only registration is
   diagnostic QC only.
8. Preserve independent native 514 geometry for each source dataset. After the
   combined 561 frame is accepted, estimate each dataset's 514-to-561 transform,
   materialize the accepted registered windows, and fit the cross-dataset
   intensity correction. Label the single materialized channel with its semantic
   label (`--output-channel-label 514` or `488`); preserve each acquisition's
   source index as provenance. Completed pixels may be reused when refreshing
   only this label with `--resume`. Fuse all 514 datasets together on the canonical 561
   grid, using `crop-sharpness-seam` edge selection between datasets.
   Registered fusion on an explicit output-grid template preserves each source's
   native spacing and resamples onto the template; source spacings need not match.
   Registration without a fixed output grid retains its equal-spacing checks.
9. Preserve independent native 488 geometry for each source dataset. Register
   each 488 dataset to the corresponding region of the completed fused 514,
   compose the accepted 488-to-514-to-561 transform, and materialize the
   accepted registered windows. Fit the cross-dataset intensity correction,
   then fuse all 488 datasets together on the same canonical grid using
   `crop-sharpness-seam` edge selection. Do not create side-only 488 fusions.
10. Fit the subtraction model from the completed, same-grid fused 514 and 488
    volumes, then stream `fused 514 - fused 488`. Do not subtract native tiles,
    create subtraction tiles, or fuse tilewise subtraction outputs.
    `lightsheet fit-fused-subtraction <FUSED_514> <FUSED_488> --output-dir <MODEL_DIR>`
    fits `514 = alpha * 488 + beta` at the 5% lower envelope of level-0 voxels
    both volumes cover and writes a fit QC image; review it, then run
    `lightsheet subtract-fused <MODEL_DIR>/fused_subtraction_model.json <OUTPUT>.ome.zarr`.
11. Validate the fused 514, fused 488, and fused subtraction volumes against
    the canonical 561 multiscale grid and render same-grid QC.
12. Export the completed fused 561 volume to NRRD. The HCR default is channel
    561 at pyramid level 3; pass both explicitly because the generic CLI
    defaults to level 0 and no channel label:

    ```bash
    lightsheet export-nrrd \
      <FINAL_FUSED_561_OME_ZARR> \
      <OUTPUT_DIR>/561.level3.nrrd \
      --level 3 \
      --channel 561 \
      --summary-output <OUTPUT_DIR>/561.level3.json
    ```

    Accept the export only after its streamed readback validates the voxel hash,
    byte count, shape, spacing, and origin.

For a same-acquisition 514/488 rough phase candidate rejected by the required
correlation-improvement check, the author may review `rough/before.png` and
`rough/after.png` and retain the existing rough placement explicitly through
`rough/no-shift-approval.json` (artifact type
`squisher_lightsheet.hcr_rough_no_shift_approval.v1`). Approval binds the exact
moving/reference inputs, recorded phase attempt and log, and both QC images.
The workflow publishes unchanged native positions with that provenance and
then performs the normal Method6 refinement, recovery, and visual screening.
Without this approval, the phase improvement check remains required.

A separately acquired series whose acquisition-offset candidate fails held-out
validation can likewise be used only as a rough initializer after explicit
visual review. Select the measured reference in `<side>-offset-reference.json`
and approve `<rough>/offset-<reference>/rough-initializer-approval.json`
(artifact type `squisher_lightsheet.hcr_offset_initializer_approval.v1`). Approval
binds native inputs, the exact failed measurement, and the held-out contact
sheet plus render record in `<rough>/offset-review`. Its measured median shifts
native tile positions once; the failed validation remains recorded. Method6,
recovery, and both-side visual screening remain required before materialization.

Review registered and recovered windows by looking at their contact sheets, not
only at summary metrics. Coverage, accepted counts, and gate scores can all pass
while individual windows are several micrometres off. Look for doubled nuclei or
edges at zero shift that merge at a small shift. Separate tissue windows from
background or artifact windows, since only the former affect fusion. Remember
that per-cell contrast stretching makes channel dominance unreliable on its own.
When a metric and the image disagree, render the fixed and moving planes
separately next to the overlay before deciding.

Use `content-preibisch-coarse` within each dataset. The fixed order for 514/488
subtraction is: combined 514 edge fusion, 488-to-514 registration, combined 488
edge fusion, fused-volume subtraction fit, then fused-volume subtraction. Use
`crop-sharpness-seam` only where datasets are combined, never within a dataset
or for subtraction. Finish and accept the combined 561 frame before placing
514 or 488; never apply one dataset's cross-channel solution to both datasets.

## Separate 405 acquisition

HCR samples now skip 405 (see the top of this document); this section is the
record of the 229Tnc and 230Tnc 405 registrations.

A 405 series acquired apart from the `488514561638` acquisition (229Tnc has
`cl-405` and `cr-405`) is its own dataset with its own tile geometry. It reaches
the 561 frame through fused 488, or through fused 514 when the dataset has no
fused 488 (230Tnc); `WORKFLOW_EDU.md` step 7 registers 405 to fused 561, which
does not work for HCR data. Do not register it to 561 here.

- **Reference.** 405 is a dense nuclear stain and HCR 561 is sparse, so they
  share almost no structure: against fused 561, Method 6 gradient NCC was
  0.001 to 0.013 (gate 0.20) and every one of the 704 CR windows was rejected.
  Fused 488 (or 514) shares nuclear structure with 405 (plane-level
  cross-correlation peaks of 0.4 to 0.6 against a median of 0.02). Use fused
  488 as the fixed volume, fixed-mask threshold read from its histogram (700 for
  229Tnc: the valley between the background and tissue lobes), so the transform
  composes 405 to 488 to 514 to 561 like the 488 registration. Without a fused
  488, use fused 514 and its reviewed fixed-mask threshold (230Tnc: 200).
- **Native geometry.** Run `lightsheet-stitch register` on each side's 405
  tiles (channel 0) with a side-specific threshold from the native no-blend
  review TIFF; the CL and CR 405 background floors can differ by an order of
  magnitude (229Tnc: CL 1000, CR 150). Positions come from the deconvolved
  stores' OME-Zarr translations when the raw TIFFs are gone
  (`clr/make-405-positions.py`). Run `fit-residual` on the 405 registration.
- **Global placement from matching tiles, not global phase.** The 405 series
  is a separate acquisition with its own Z stack: in 229Tnc it was acquired
  hours apart from the multichannel series, with a different scene stage Z and
  fewer planes (CL 991 vs 1024, CR 838 vs 920). The OME plane positions carry
  no Z, so tile Z translations are all zero, and the 405 stacks start at a
  different depth than the multichannel stacks (229Tnc: 172 um on CL, 128 um on
  CR), beyond the 55 to 100 um search of `cross-register global-phase`. The 405
  then does not cover the multichannel Z range wherever its stack ends sooner
  (229Tnc: the last 235 um of the CL stacks). That
  command also fails on 405 even at matched depth (`--z-slab-center-um`): at
  1.2 um pixels over a 32-plane MIP the dense nuclei blur out, its peak NCC was
  0.02 to 0.03, its XY shift pointed the wrong way, and its orthogonal ZX and ZY
  Z residuals disagreed by 16 to 59 um. Because both acquisitions share one
  stage lattice, measure the offset directly instead: for about nine central
  tiles, register highpassed 405 planes to the same tile's reference-channel
  planes (488, or 514) over depth lags and XY shifts, scanning every lag at
  level 2 before refining at level 1 because the offset is unknown in advance
  (230Tnc `full-reg/clr/xreg-405/measure-offset.py`), convert each
  tile's offset to the world correction of its placed position, and shift the
  placed 405 positions by the median. The tile pyramids downsample XY only, so
  a depth lag counts level-0 planes; scale only the XY shift by the level
  factor. For 229Tnc the per-tile corrections had a MAD of at most 3 um per
  axis (CL -172.0, -40.0, -56.8 um; CR -349.2, +38.5, +16.6 um ZYX). The
  229Tnc run scaled the lag by 2 as well, misplacing CL by 171.6 um and CR by
  128.3 um in Z; Method 6 level-2 phase priming recovered both, and the fused 405
  and 488 match at zero shift in level-2 XZ and XY sections. Before Method 6,
  validate the shifted positions on tiles held out of the median and spread
  away from the measured ones (three by default), at three depths each: a
  depth-lag scan around the predicted planes must peak at zero with the
  position-implied XY shift (230Tnc tile 012: both sides peak at zero, XY within
  0.5 level-1 pixels). `cross-register acquisition-offset` does this and
  publishes positions only when every held-out check passes. Moving planes are
  resampled onto the reference XY spacing first (229Ptprz1 CR 514 and 488
  record 0.300061 um against 0.300247 um); recorded Z spacings are taken as
  the nominal 1.8 um step and may differ only by less than a plane across the
  stack. When several reference channels could serve (CR 514 against the CR
  561/638 track), measure each and choose by its held-out validation and peak
  margins. Use the shifted positions as the Method 6 moving
  positions in place of a global-phase output, recording why beside them.
- **Fits and fusion.** Then run the standard sequence from step 9
  (Method 6 with `--fit-intensity-transform linear` and minimum gradient NCC
  0.20, native and complete recovery, pre-fusion QC, materialization, join,
  cross-dataset match, `crop-sharpness-seam` fusion on the canonical grid).
  Start Method 6 from the EDU 405 initializer; for a dataset whose combined-frame
  axes flip Z and X (CR under `lr_z_endview_flip_xz`), flip the sign of its
  ZY, YX, and their transposed terms (229Tnc CR); a translation-only join
  (230Tnc) uses it unchanged. Give Method 6 a CuPy pool of at least 14 GiB:
  at 10 GiB its masked structural translation runs out of memory. Run
  `recover-native` under a capped resume-retry loop, because it can exit
  silently mid-run (230Tnc CR: four times); each rerun resumes finished windows.
  Keep the 405 deconvolved stores until its fusion is accepted.
- **Internal holes.** As in `WORKFLOW_EDU.md`, place tissue windows that
  automatic recovery leaves uncovered inside the tissue from curated seeds:
  list them in a `manual-initializations.json` channel keyed by dataset
  (`405-CL`), pass it to `recover-native` with `--manual-initializations` and
  `--manual-initializations-channel`, then rerun `complete-recovery` and
  pre-fusion QC. `recover-native` rejects entries that name windows absent from
  its summary, so keep one channel per dataset. `lightsheet cross-register
  propose-seeds --summary <complete-recovery summary> --manual-initializations
  <master> --manual-initializations-channel 405-CL` writes the seeds. A window
  rejected `spatial_fallback_contradicted_by_image` is seeded with the exact
  placement selected by complete-recovery (interpolation or a quality-gate-rejected
  native rerun) shifted by the residual measured for that candidate
  (230Tnc: 3 to 10 level-2 px, consistent across planes). Every other rejected
  tissue window (fixed masked fraction below 0.6) is seeded from its best
  image-derived native-recovery attempt: the phase-primed level-2 initializer
  or the native refit from the recovery initializer, skipping drifted attempts
  (any translation above 60 level-0 voxels) or out-of-bounds affine scales and
  requiring gradient NCC of at least 0.10. Seeds of windows already accepted
  are kept; windows without such evidence stay unseeded. Complete-recovery
  still checks every seed against the image. Mask-rejected windows cannot be
  seeded. For 229Tnc, two CL windows (tile 018) were seeded. For 230Tnc, 11 of
  16 attempt-based CL seeds were accepted, and all 10 residual seeds (5 per
  side) were accepted.
- **Screen the registration before materialization.** Materialization and
  fusion of one 405 take about 10 GPU hours, and neither can be updated for a
  few changed windows, so gate them on the registration. In each side's
  complete-recovery summary, every window rejected for a reason outside the
  fixed-mask rejections is a window the fusion leaves uncovered unless
  neighbouring windows cover it. Seed these windows and rerun recovery until
  none remain, or until the author accepts each remaining one. Accepted
  windows can be misplaced too. Complete-recovery's spatial fallback promotes a
  quality-gate-rejected native rerun over the interpolated placement when it
  scores higher on every plane, which can happen at near-zero NCC. A rerun that
  the residual check does not verify is therefore kept only when no window
  corner moves more than `--maximum-unverified-rerun-departure-px` (default
  60 level-0 px, a provisional value; about 18 um in XY and 108 um in Z at
  230Tnc spacing) from the interpolated placement. Each promotion records the
  corner displacement and, when the rerun is withheld, the rerun's residual.
  On 230Tnc, verified promotions displaced corners up to 57 px, and four
  unverified ones 71 to 464 px. One of these (CL 006 qy3_qx2 z480, 158 px) was
  fused before the bound existed and left a gap of 22 um beside its neighbour.
  Still list accepted windows whose final transform departs from the rest of
  their tile's field: recovery's outlier rule (leave-one-out decomposed-affine
  prediction, maximum corner displacement) at the same 60 px bound flags
  exactly the three 230Tnc windows promoted 173 to 467 px from their
  neighbours, where recovery's own 5 px tolerance flags 319 of 1,508. List also
  linear-mapping recoveries that kept their own fit with
  fewer than two image-comparison planes (230Tnc: 8 of 22), since that stage
  has no residual check. Then pass both datasets' summaries together to
  `cross-register prefusion-qc` (combined coverage) at planes through every
  repaired or remaining window and at several planes spanning the tissue.
  Read the hole maps, which mark reference pixels above the fixed threshold
  that no window covers: red inside the brain is a defect. Red along thin
  tissue outside the brain comes from windows rejected because most of their
  reference lies below threshold, and is expected. The uncovered percentage
  alone does not separate the two. On 230Tnc, the
  combined map at level-2 plane 60 showed an interior hole from one
  contradicted window while reporting only 0.66% uncovered. It was fused
  anyway, and the fusion had to be redone.
- **Pre-fusion QC depth.** The CL and CR 405 stacks cover different world Z
  ranges, and the default middle plane of the fixed volume can lie outside one
  dataset's range, which reports it 100% uncovered. Pass
  `cross-register prefusion-qc --fixed-z-index` with a level-2 plane inside
  that dataset's range (229Tnc CR: 98).
- **Z coverage.** The fused 405 is empty wherever neither 405 dataset reaches
  the canonical grid; for 229Tnc that is world Z above about 1620 um, the last
  31 of 442 level-2 planes. This is the extent of the acquisition, not a fusion
  failure.

`clr/run-xreg.sh 405 place|cl|cr` and `clr/measure-405-offset.py` in the dataset
folder implement this for 229Tnc. For 230Tnc, `full-reg/clr/xreg-405` holds the
whole sequence: `prepare.py`, `measure-offset.py`, `fit-residual.sh`,
`run-xreg.sh` with `run-xreg-retry.sh`, and `run-fuse.sh` with
`run-fuse-chain.sh`; seeds come from `cross-register propose-seeds`.

## Reviewed native-only 561 intensity correction

When additional between-side 561 intensity matching worsens held-out agreement,
retain its rejected `canonical/match-561` artifacts for review. After explicit
author approval, `canonical/intensity-561-native-only-approval.json` may approve
native-only 561 corrections. Its artifact type is
`squisher_lightsheet.hcr_native_only_561_approval.v1`; `approved` must be true,
and `identity` binds `registration`, `rejected_match`, `rejected_field`,
`native_L`, and `native_R` SHA-256 values. The packaged workflow composes the
accepted native corrections once in `canonical/native-corrections-561`, and
omits the rejected registered intensity field from 561 fusion. The approval
is refused after any bound input changes. Other channels retain their intensity
matching and validation requirements.

## Packaged workflow

`lightsheet hcr-manifest draft` records each acquisition's side, channel labels
(from the embedded CZI emission wavelengths), tracks, spacing, and tiles;
`lightsheet hcr-workflow <manifest>` then runs this document for CL/CR and TL/TR
datasets and stops at each decision with a review file to fill, resuming from the first
incomplete stage when rerun. The decisions, in order: the manifest itself,
native registration thresholds, native geometry and fit-residual corrections,
the coarse placement (CL/CR: `align-registered-lr` on the dataset-local 561
mosaics; TL/TR: a centre-Z slab `rough-phase` of the two sides' registered
tiles; either may be replaced by a manual moving-side shift), the final placement, the
fused 561 mask threshold (one accepted native 561 value), any choice among
reference channels for a separate acquisition's offset, each label's
registration screen, the fused 514 and 488 thresholds, the subtraction fit,
final QC, and cleanup. Each review file is bound to the artifacts it reviewed
and is refused once they change. Recovery repeats in rounds while measured
seeds reduce the rejected tissue windows; adding entries to the dataset's
`manual-initializations.json` starts another round. Manual entries added or changed
while a recovery round runs are also evaluated, even when the seed proposer
preserves them in its output and the current round made no count improvement.

## Cleanup after final validation

Materialized windows can go before the whole dataset: as soon
as the fusion that consumed them is complete, matches the canonical 561 grid,
and passes QC, delete that channel's `materialized-<side>/materialized_tiles`.
Keep the `fused_fixed_materialized_chunks.*.json` records in each side and in
`materialized-joined`; they are the fusion's position and registration inputs.
The packaged workflow instead retires them with everything else in one
approved cleanup after final QC and the NRRD export, keeping each deleted
store's Zarr metadata and a record of every deletion; a rerun after cleanup
only verifies the retained finals against that record.

Clean up everything else only after every required final store and NRRD export
is complete, the stores match the canonical 561 grid, and QC is accepted. The
following are then deletable:

- deconvolved stores and their joined symlink views when no rerun is planned;
- dataset-local registration mosaics after accepting the cross-dataset
  transform and completing the downstream fusion;
- sampling and runtime caches, including `.cache`, `cache`, `.mpl`,
  `__pycache__`, `basic/sample-cache`, and `scale/float32-samples`;
- abandoned temporary fusion workspaces and superseded failed, tilewise
  subtraction, side-only 488, registration, materialization, or fusion outputs.

Delete raw source OME-TIFF acquisitions only after verifying an independent
archive and obtaining explicit user approval. Preserve final fused volumes,
the final subtraction, accepted correction models, registrations, positions,
acceptance and provenance manifests, QC, and logs.

In short: **561 owns the shared-track tile geometry, reference frame,
layout-specific cross-dataset alignment, and final fusion grid. CL/CR uses
`lr_z_endview_flip_xz`; TL/TR uses `tltr_x_join_center_z_phase`. Production 638
reuses the 561 tile graph and adds only the accepted 638-to-561 physical
affine. Production 488 reaches that frame through the accepted
488-to-514-to-561 transform chain.**


## Mosaic top-up metadata

The acquisition manifest retains every tile from compatible mosaic and later
single-tile CZI acquisitions in the same converted folder. Multiple exact XML
records and their tile membership are bound in `czi_metadata_groups`; manifest
verification rechecks those records as well as channel identity, geometry and
file identity. The deconvolution metadata collector validates common imaging
settings and preserves per-acquisition XML and original tile positions. Do not
discard top-up tiles or replace their raw metadata with the main mosaic XML.


For interrupted mosaics whose continuation tiles have acquisition-local OME
positions, review measured same-channel overlaps before placing them. Save
`acquisitions/<name>/position-overrides.json` with artifact type
`lightsheet.position_overrides.v1`, `approved: true`, and a `tiles` mapping
keyed by exact TIFF filename. Each record contains `source_identity`
(`size_bytes`, `mtime_ns`), `original_translation_um_zyx`, and reviewed absolute
`translation_um_zyx` in micrometres. The workflow passes this to
`lightsheet single-position --position-overrides`; stale sources, unknown tiles,
invalid coordinates and unreviewed overrides fail. Position-stage fingerprints
bind the override file; raw metadata and deconvolved pixels remain unchanged.

For reviewed continuation overlaps wider than the automatic adjacency selector's
25% limit, the same approved override may list `additional_registration_pairs`
as pairs of exact TIFF filenames. These links propagate into the position file
and are used by both overlap screening and measurement. The reader verifies
the review-file identity and approval and rejects missing tiles, self-links,
incompatible spacings or non-overlapping links. Automatic adjacency criteria
remain unchanged. HCR registration uses the normal acquisition-edge retention
policy and refuses interior disconnections instead of silently excluding tiles.
