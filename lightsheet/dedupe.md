# Dedupe Plan

This note records duplicated implementation segments found during the 230Tnc
registration/fusion cleanup. The goal is to make reusable workflow owners clear
without changing runtime behavior.

## Priority 1: Phase-Correlation Primitives

Owner: `src/squisher_lightsheet/seams.py`

Status: implemented. The legacy stitcher keeps compatibility shims, but the
GPU DFT refinement implementation now lives in `seams.py`.

Duplicated segments:

- `upsampled_dft_gpu`
  - `src/squisher_lightsheet/seams.py`
  - `src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py`
- `refined_phase_shifts_gpu`
  - `src/squisher_lightsheet/seams.py`
  - `src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py`

Plan:

- Keep the legacy function names only as compatibility shims.
- Delegate implementation to `seams.py`.
- Keep all GPU phase-correlation math in one owner.

Validation:

- `tests/test_seams.py`
- `tests/test_tile_phase.py`
- static checks on the legacy stitch script.

## Priority 2: OME-Zarr Pyramid Utilities

Owner: new or expanded `src/squisher_lightsheet/pyramid.py`

Status: partially implemented. Shared ceiling division, chunk
iteration/count, and pyramid metadata scale helpers now live in `pyramid.py`;
both legacy pyramid writers delegate those helpers to the package owner. Full
scale-level write and metadata-update consolidation remains future work.

Duplicated segments:

- `chunk_slices`
- `chunk_count`
- `pyramid_relative_factors`
- `level_coordinate_transformations`
- scale-level metadata update logic

Locations:

- `src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py`
- `src/squisher_lightsheet/_legacy/add_ome_zarr_pyramid.py`

Plan:

- Move shared chunk iteration, downsampling, Zarr v3 codec construction, and
  multiscales metadata updates into `pyramid.py`.
- Keep legacy scripts as thin callers until the CLI is fully canonicalized.
- Preserve the stitcher-specific behavior that repairs stale OME axis metadata
  after multiview-stitcher writes.

Validation:

- `tests/test_fusion.py`
- add or reuse a small pyramid metadata test before replacing both call sites.

## Priority 3: QC Rendering Helpers

Owner: `src/squisher_lightsheet/qc.py`

Status: implemented for shared rendering primitives. The legacy rough-phase
and LR registration QC scripts delegate scaling, overlays, contact sheets, max
placement, and projection-canvas helpers to `qc.py`. Label-specific rendering
remains local to the LR QC script. The LR QC script also now shares one
internal geometry builder for normal and full-affine bounds.

Duplicated segments:

- `scale_u8`
- `write_contact_sheet`
- `empty_projection_canvases`
- `place_global_projections`

Locations:

- `src/squisher_lightsheet/_legacy/rough_align_tltr_center_z_phase.py`
- `src/squisher_lightsheet/_legacy/render_lr_level4_registration_qc.py`

Plan:

- Move image scaling, overlay, contact sheet, and projection-canvas helpers to
  `qc.py`.
- Preserve existing labels and color mapping.
- Avoid changing QC image filenames or layout.

Validation:

- `tests/test_rough_phase.py`
- `tests/test_registration.py`
- spot check representative QC PNG dimensions.

## Priority 4: Phase Metrics

Owner: new `src/squisher_lightsheet/phase_metrics.py` or existing `seams.py`
if the metric is seam-specific.

Status: implemented for cross-workflow phase metrics. Robust percentile
normalization and masked Pearson correlation now live in `phase_metrics.py`;
`tile_phase.py`, `track_z.py`, and the rough phase legacy script keep
compatibility wrappers.

Duplicated segments:

- robust percentile normalization
- masked correlation helpers

Locations:

- `src/squisher_lightsheet/tile_phase.py`
- `src/squisher_lightsheet/track_z.py`
- `src/squisher_lightsheet/_legacy/rough_align_tltr_center_z_phase.py`

Plan:

- Extract only exact, behavior-equivalent metrics.
- Do not force together workflows whose thresholds or support masks differ.
- Keep gradient-NCC seam metrics in `seams.py` unless they become truly
  cross-workflow.

Validation:

- `tests/test_tile_phase.py`
- `tests/test_rough_phase.py`
- track-z tests once present.

## Low Priority

Argument parser helpers such as `parse_nonnegative_int` and
`parse_nonnegative_float` have similar structure but are not worth extracting
unless CLI parser construction is being reorganized anyway.

Status: mostly cleaned. Same-file numeric parser duplication in the legacy
stitcher was collapsed behind one helper, tile-phase cache loaders now share
one loader, and source-view flatfield parsing now shares
`parse_source_view_path_entry`. The Typer CLI and argparse legacy script keep
thin local wrappers so they preserve their original exception types. Common
TIFF helpers now live in `tiff.py`, including series level counting,
ZYX spatial-shape extraction, level downsample factors, and source-level
selection.
