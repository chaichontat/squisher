# squisher-deconv

`squisher-deconv` streams flattened TIFF stacks as logical `(Z, C, Y, X)` volumes,
computes global scaling from uniformly sampled deconvolved planes, and writes
chunked CZYX OME-Zarr outputs directly without a full float32 staging pass.

Commands:

```bash
squisher-deconv basic INPUT... --out-dir BASIC_DIR --label SAMPLE --channels 2 --device cuda
squisher-deconv sample-scale INPUT... --out-dir DIR --planes N --channels 2 --psf PSF-c0.tif --psf PSF-c1.tif --iter 1
squisher-deconv run INPUT... --out-dir DIR --channels 2 --psf PSF-c0.tif --psf PSF-c1.tif --scaling DIR/scaling.json --iter 1
```

`basic` samples level-0 OME-TIFF planes with blank- and edge-slice rejection, then fits one
joint-channel BaSiCPy model with autotuning, darkfield estimation, and intensity sorting. It writes
one run-compatible pickle per channel, flatfield/darkfield TIFFs, a reusable sample cache, and a JSON
manifest containing the input identity, sampling decisions, and complete fit settings. Existing final
outputs are never overwritten. The workspace overrides BaSiCPy 2.0's stale SciPy upper bound so it
uses the workspace's SciPy version.

In the `multi` conda environment before installation, run the module directly:

```bash
PYTHONPATH=/home/chaichontat/squisher/deconv/src python -m squisher_deconv basic ...
PYTHONPATH=/home/chaichontat/squisher/deconv/src python -m squisher_deconv sample-scale ...
PYTHONPATH=/home/chaichontat/squisher/deconv/src python -m squisher_deconv run ...
```

### Per-tile channel gains

Both `sample-scale` and `run` accept `--tile-gains gains.json`. The manifest must
cover exactly the supplied source files, using absolute paths and one positive,
finite multiplier per channel in the same order as `--basic` and `--psf`:

```json
{
  "schema_version": 1,
  "channels": 3,
  "tiles": [
    {"source": "/data/tile.ome.tif", "gains": [1.0, 1.12, 1.0]}
  ]
}
```

The engine applies gains after BaSiC subtraction/division and nonnegative
clipping, before Richardson–Lucy deconvolution. A gain of 1 preserves the
channel. Missing, duplicate, extra, nonpositive, or nonfinite entries are errors;
relative paths are rejected. Paths are resolved before matching.

Pass the same manifest to scaling estimation and the production run, and create
fresh scaling samples when changing gains or BaSiC profiles. Sample manifests
and output provenance record the gain-file identity. Resume rejects outputs made
with a different manifest. Registration and fusion do not apply these gains
again: they consume the already corrected deconvolved tiles.

### Z-dependent fit-residual fields

Profiles produced by `lightsheet fit-residual --z-degree 1` contain a smooth
residual field in normalized original raw Z/Y/X coordinates alongside the
original 2D BaSiC arrays. Pass these pickles through the existing `--basic`
arguments to both `sample-scale` and `run`. The GPU engine evaluates the field
using the slab's original raw Z start and the full source Z count, then applies
it after BaSiC correction and before tile gains and deconvolution. Changing the
profile requires fresh scaling samples, as with any BaSiC change.


### Metadata from mosaic top-up acquisitions

A converted input folder may contain tiles from a mosaic and later single-tile
acquisitions. Deconvolution retains all exact CZI XML records in
`metadata.json`: one common record uses `czi_shared_metadata_xml`; multiple
compatible records use `czi_metadata_groups`, with `metadata_group_index` on
each tile position and a null shared record. Grouped inputs must agree in OME
pixel geometry and CZI imaging settings, including channel order, tracks,
exposure, illumination and calibration. Acquisition identity, time, stage
placement, mosaic grid and display settings may differ; Airyscan transforms
are ignored for comparison only when their corresponding mode is `Off`.
Original XML is retained without modification. Incompatible imaging settings
or missing metadata fail before processing.
