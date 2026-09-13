# Post-deconvolution residual correction

This document explains how `lightsheet fit-residual` calibrates deconvolved
tiles and how fusion applies the result. The same mechanism handles a single
acquisition side and a joined acquisition with multiple sides. Side count
changes the registration and source set; it does not select a different fitting
algorithm.

`lightsheet post-basic` is an exact alias for `lightsheet fit-residual`. Both
names accept the same post-deconvolution inputs and run the same code. The
workflow does not compose raw BaSiC profiles or apply another BaSiC correction
during fusion.

## System overview

```text
deconvolved source tiles ─┐
accepted registration ────┼─> sample registered overlaps across Z
fixed fused grid ──────────┘                 │
                                             v
                              fit and evaluate residual model
                                             │
                                             v
                          correction.json + QC + manifest
                                             │
                                             v
source-block read during level-0 fusion ─> correct ─> interpolate ─> blend
```

The fixed fused image supplies the output shape, physical origin, spacing, and
sampled Z planes. Its pixel values are not used to estimate the correction.
All intensity measurements come from the registered deconvolved source tiles.
This means an uncorrected preview can serve as the fitting grid as long as it
has the exact geometry intended for the final fusion.

The registration supplies the source paths and the transform from each source's
original pixel coordinates into the fixed physical frame. The fitter inverts
that mapping at each sampled fixed-grid point, reads the requested OME-Zarr
pyramid level, and retains the original source Z/Y/X coordinates for the model.

## What the model represents

The fitted multiplier has three parts:

```text
corrected(z, y, x, source)
    = deconvolved(z, y, x, source)
    × shared_field(z, y, x)
    × source_field(z, y, x, source)
    × source_gain(source)
    × global_scale
```

`field` is a smooth camera-coordinate correction shared by every source in the
fit. It uses five cosine modes over normalized original camera Y/X. Their
coefficients may vary over normalized original camera Z. With coefficient
matrix `c`, the log field is

```text
log field(z, y, x)
    = sum_d cos(pi × d × z) / (1 + d²)
      × sum_m c[d, m] × XY_mode_m(y, x)
```

where normalized coordinates run from zero to one over each source's full
native Z/Y/X shape. `--xy-degree 1` activates the first-order X and Y modes;
`--xy-degree 2` also activates the second-order and X/Y interaction modes.
`--z-degree` controls the number of depth modes from 0 through 3.

Each source also receives its own smooth field over the same camera-coordinate
basis and one constant positive gain. The source field represents spatial
variation unique to that acquisition tile, while the gain represents its
constant offset. Source-field regularization is selected on held-out source
pairs by default; `--source-field-penalty` fixes it explicitly.

`global_scale` is common to every source. It conservatively scales the largest
possible combined multiplier to at most one, preserving relative calibration
while avoiding amplification when corrected values are written back to the
source integer dtype.

## How fitting works

1. The command reads the accepted records from the registration and verifies
   that they point to unique, original deconvolved `CZYX` OME-Zarr sources.
   Materialized crops are rejected because their local coordinates do not
   preserve the original camera-coordinate contract.
2. Requested Z percentiles are converted to level-0 fixed-grid indices with
   `floor((Z_count - 1) × percentile / 100)`. The default is 39 planes at
   2.5-percentile intervals.
3. At each fixed Z, the registration transform maps the sampled grid into each
   source. Reads use `--source-level`, including that level's NGFF scale and
   translation, while model coordinates remain normalized level-0 source
   coordinates.
4. A foreground cutoff is estimated independently for each plane with Otsu's
   method on `log1p` positive intensity. Measurements below the cutoff or
   outside either source are excluded.
5. Every eligible overlapping source pair contributes measurements, including
   overlaps between sides that do not appear as a last-writer boundary in the
   preview mosaic. Each target is the local log intensity ratio
   `log(source_b / source_a)`.
6. The field is fit with pseudo-Huber loss and L2 regularization. The CLI uses
   the requested `--field-penalty`, or selects it on held-out pairs when omitted. The shared fitting
   function can instead select among `0.003`, `0.03`, and `0.3` when called
   programmatically with no fixed penalty.
7. Source gains are fit from the median residual of each pair after accounting
   for the shared field. The deployment fit requires overlap support for every
   source and one connected overlap graph.
8. Regularized source-specific fields are fit from the remaining pixel-level
   residual and selected using the same pair-disjoint validation boundary.

The fit/evaluation split is by source pair, not by pixel. All observations of
the same pair stay in the same fold at every sampled depth. This prevents the
same physical overlap from appearing in both training and test data. The
workflow reports held-out pair metrics, then refits the deployment coefficients
and gains on every eligible pair.

The command also leaves out each sampled Z plane in turn. It compares the
depth-aware model trained on the remaining planes with a two-dimensional model
and a single-plane model. These results are written to `qc/z_validation.csv`
and shown in each channel QC figure.

## Single-side workflow

A single-side run uses a position and registration containing only that side's
deconvolved sources. The fixed grid is normally an uncorrected preview made
from the same accepted geometry.

```bash
lightsheet fit-residual \
  --fixed-fused SIDE_PREVIEW/fused.ch0.ome.zarr \
  --registration SIDE_REGISTRATION/registration.json \
  --output-dir SIDE_CALIBRATION \
  --channel 0 \
  --source-level 2 \
  --stride 4 \
  --xy-degree 1 \
  --z-degree 1 \
  --field-penalty 0.03 \
  --source-field-penalty 0.03 \
  --workers 8 \
  --seed 20260907

lightsheet fuse SIDE_DECONVOLVED_TILES \
  --position-input SIDE_POSITIONS.json \
  --registration-input SIDE_REGISTRATION/registration.json \
  --output SIDE_OUTPUT \
  --channel 0 \
  --fusion-level 0 \
  --residual-correction SIDE_CALIBRATION/correction.json
```

For fused-fixed cross-channel workflows, apply the same correction while
materializing registered overlap windows:

```bash
lightsheet fused-fixed-materialize-overlap \
  --moving-position MOVING.positions.json \
  --source-summary FINAL_SUMMARY.json \
  --output-dir MATERIALIZED \
  --source-channel 0 \
  --residual-correction SIDE_CALIBRATION/correction.json \
  --level-factor-zyx 1,1,1 \
  --output-codec zstd
```

The materializer validates the correction against the complete original
moving source set and applies it to native level-0 blocks before resampling or
downsampling. Its identity is part of materialization provenance and resume
validation, so outputs created with another correction cannot be reused.

The correction must contain exactly the sources selected by fusion. If CL and
CR are fused separately, each side therefore needs its own fit and its own
correction artifact.

## Multiple-side workflow

A joined run first places every side in one physical coordinate frame. For
CL/CR data, this means completing side-internal registration and the CR-to-CL
merge before fitting. The joined registration must retain the original
deconvolved source paths and any orientation scales needed by fusion.

Fit one correction against the complete joined source set:

```bash
lightsheet fit-residual \
  --fixed-fused JOINED_PREVIEW/fused.ch0.ome.zarr \
  --registration JOINED_REGISTRATION/registration.json \
  --output-dir JOINED_CALIBRATION \
  --channel 0 \
  --source-level 2 \
  --stride 4 \
  --xy-degree 1 \
  --z-degree 1 \
  --field-penalty 0.03 \
  --source-field-penalty 0.03 \
  --workers 8 \
  --seed 20260907

lightsheet fuse ALL_DECONVOLVED_TILES \
  --position-input JOINED_POSITIONS.json \
  --registration-input JOINED_REGISTRATION/registration.json \
  --output JOINED_OUTPUT \
  --channel 0 \
  --fusion-level 0 \
  --residual-correction JOINED_CALIBRATION/correction.json
```

The joined fit uses side-internal and cross-side overlaps wherever the
registered sources intersect. Side labels are retained in registration and
fusion provenance, but the residual solver indexes sources by their resolved
paths. It fits one gain for every tile across all sides and requires the full
source graph to be connected.

Two separately fitted side corrections cannot be combined in one joined
fusion. Fusion accepts one correction artifact and verifies that it covers the
exact source set. If the final product contains all sides, fit all sides
together. If the products remain separate, fit each product separately.

| Final product | Fit source set | Fixed grid | Correction artifacts |
| --- | --- | --- | --- |
| One side | That side's original tiles | That side's final geometry | One per channel |
| Separate side outputs | Each side independently | Each side's final geometry | One per side and channel |
| Joined multi-side output | All original tiles from all sides | Joined final geometry | One per channel |

The same rule applies to more than two sides: the artifact boundary is the
complete set of original sources entering one fusion, not the acquisition-side
label.

## Fusion-time application

Fusion loads `correction.json` before processing output blocks and rejects it
unless all of the following match:

- schema and coordinate convention;
- requested channel;
- exact resolved source-path set;
- every source's native Z/Y/X shape;
- hashes of root and level-0 source metadata.

Residual correction is supported only with `--fusion-level 0` and cannot be
combined with `--flatfield-dir-by-source-view` or another fusion-time BaSiC
correction. The input tiles must already contain the accepted upstream
correction and deconvolution.

For each source read, fusion evaluates only the requested original-coordinate
Z/Y/X slab. The compact coefficient matrix is never expanded into a complete
source volume. The corrected read is used consistently by direct fusion reads,
the corrected-slab cache, GPU workers, and crop-sharpness scoring.

The correction fingerprint is part of corrected-slab cache keys, crop-score
provenance, and the fusion resume plan. Changing coefficients, gains, source
shape, or global scale therefore invalidates results that depend on the old
model.

Correction and blending have different responsibilities. Residual calibration
changes each source's intensity before interpolation. The fusion weight mode
then decides which registered source contributes to each output voxel. The
default `crop-sharpness-seam` mode selects contiguous sharper overlap regions
and feathers their interfaces over the requested physical seam width.

## Outputs and acceptance

`fit-residual` refuses to overwrite an existing output directory. It builds a
staged result and publishes the directory atomically only after the correction,
manifest, sampled-plane images, and numerical and rendered QC have all been
written successfully.

The main artifacts are:

- `correction.json`: compact deployment model and exact source identities;
- `manifest.json`: inputs, sampled planes, parameters, metrics, and hashes;
- `tile-gains.json`: source gains for the fitted channel;
- `qc/source_fields.csv`: per-source field ranges across camera depth;
- `qc/channel_N.{png,pdf,svg}`: visual and numerical review figures;
- `qc/z_validation.csv`: held-out-depth comparisons;
- `planes/zN/`: shared-scale before/after images for additional sampled depths.

Review low, middle, and high Z planes, the applied multiplier field, tile
interiors, and tile boundaries together. Accept the artifact only when the
visual result and held-out seam measurements agree. The detailed review rules
are in [POST_BASIC.md](POST_BASIC.md).

For multiple channels, run the complete fit once per channel. A correction
artifact records one channel and fusion rejects it for any other channel.
