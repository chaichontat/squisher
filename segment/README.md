# squisher-segment

`squisher-segment` extracts training images, trains Cellpose models, runs
distributed segmentation, stitches intermediate labels, and postprocesses 3D
segmentations.

## Install

Install the package in editable mode in the `multi` environment from the
repository root:

```bash
conda activate multi
python -m pip install -e ./segment
squisher-segment --help
```

The editable install keeps the `squisher-segment` command pointed at this
checkout. Use `squisher-segment COMMAND --help` for command-specific options.

## Commands

```text
squisher-segment extract INPUT
squisher-segment n4 INPUT_OME_ZARR
squisher-segment train TRAINING_ROOT MODEL_NAME
squisher-segment segment run INPUT_ZARR
squisher-segment segment stitch TEMP_DIR OUTPUT_ZARR
squisher-segment postproc run INPUT_ZARR
squisher-segment regionprops LABELS_ZARR
```

- `extract` samples registered TIFF or fused Zarr data for model training.
- `n4` corrects one channel-separated Lightsheet OME-Zarr and rebuilds its pyramid.
- `train` reads `TRAINING_ROOT/models/MODEL_NAME.json`, trains the model, and
  writes `MODEL_NAME.trained.json` beside the input configuration.
- `segment run` performs distributed Cellpose inference. Add `--cellpose-only`
  to retain the unstitched output for a separate stitching step.
- `segment stitch` converts a Cellpose temporary directory into the final Zarr
  segmentation.
- `postproc run` smooths and filters an existing 3D segmentation.
- `regionprops` measures labels chunk-by-chunk and writes one Parquet row per
  globally consistent label.

Training, segmentation, and postprocessing require CUDA-capable workers.

## N4 correction

`n4` accepts a completed Zarr v3 Lightsheet output with one `ZYX uint16`
channel and NGFF multiscale metadata. Run it from the `multi` Conda environment:

```bash
squisher-segment n4 fused.ch0.ome.zarr
```

The default output is the compact sibling `fused-n4.ch0.ome.zarr`; use
`--output` to choose another path. A 3D correction field is estimated from the
coarsest pyramid dataset by default; use `--field-level` to select another
level. Its NGFF scale and translation map the field into level-0
storage-aligned GPU blocks by trilinear interpolation. B-spline spacing is
derived from level-0 physical scale, so `--field-level` does not change the
field's physical smoothness. The default low-resolution ZYX spacing is
`24 48 48`; with shrink 4 and `0.6 0.3 0.3` µm voxels, this is 57.6 µm on
every axis. Pyramid levels are
rebuilt by mean downsampling the preceding corrected level into the source
level's exact shape, chunks, shards, codecs, and dimension names.

Root NGFF metadata and per-array attributes are copied from the source without
reconstruction. A referenced `squisher_fusion` provenance bundle is copied and
checksummed; other auxiliary Zarr nodes are rejected rather than silently
dropped. The destination adds a `squisher_n4` provenance attribute and a
checksummed `n4-field.npy` field artifact. Quantization percentiles come from
deterministic corrected level-0 windows, independent of `--field-level`.

The 3D field stays resident on the GPU while thresholds, normalization,
level-0 interpolation and correction, optional sharpening, quantization, and
pyramid reduction run without a CPU fallback. The SimpleITK 3D N4 field fit is
the sole CPU exception. Output data and metadata are synced before
`squisher_complete` is set, and Linux atomic rename semantics prevent a newly
created or replaced destination from being overwritten. `--overwrite` accepts
only a completed prior N4 output; concurrent mutation inside that output is not
supported.

`--threshold` accepts a numeric foreground threshold; its default is values
greater than zero. Pyramid shapes, physical scales, and translations must
describe the same integer mean-downsampling transitions.

`segment run --channels` selects and orders the named input channels before
inference. Resume state, normalization data, nonempty-block caches, and the
output-specific `.done` manifest are bound to the input metadata, selected
channels, model checksum, and evaluation settings. Input Zarr stores must not
be modified in place while a run is resumable; use `--overwrite` after changing
input chunk contents. `segment stitch` also requires `--overwrite` before it
will replace an existing output.

Every run first reuses a matching nonempty-block cache. On a cache miss, it
scans only the input channel named `561` and sends a planned block to Cellpose
only when that crop contains a value strictly greater than
`--nonempty-threshold` (default: 1000). Within selected inference crops, raw
561 values at or below the same threshold mask every model channel to its
normalization baseline before Cellpose. The channel and threshold are part of
the run and cache identities, and scan-produced
nonempty caches persist beside the temporary run directory. Direct API masks
are identified by shape, dtype, and content hash so cached block positions
cannot cross masks. This run-identity schema change requires older resumable
runs to be restarted with `--overwrite`.

`segment run` also accepts a `squisher_segment.registered_ome_input.v1` JSON
manifest in place of a materialized ZYXC array. Its `sources` entries name registered,
channel-separated ZYX OME-Zarrs, and `source_roi_zyx` defines the shared crop.
Workers read the requested level-0 crops directly and stack only those crops in
memory. Source shapes and dtypes must match, and every source must be marked
complete; no intermediate multichannel Zarr is written.

Distributed segmentation uses the SAM TensorRT backend. `--target-nz`,
`--target-ny`, and `--target-nx` control the
number of internal 256-pixel Cellpose tiles along each spatial axis. XY uses
`(ny, nx)`, XZ uses `(nz, nx)`, and YZ uses `(nz, ny)`. The resolved ZYX core
shape is part of the run identity and also determines temporary label chunks.

## Parallel region properties

Run region measurements directly on a completed 3-D integer label Zarr:

```bash
squisher-segment regionprops postproc.zarr \
  --output props.parquet \
  --workers 4
```

Each process reads one on-disk Zarr chunk at a time. Chunk results contain
additive voxel counts and coordinate moments plus per-Z areas; Polars reduces
labels that cross Z, Y, or X chunk boundaries without loading the label volume
as one array. The output columns are `label`, `area`, `centroid_z`,
`centroid_y`, `centroid_x`, `plane_z`, and `plane_area`. The representative
plane maximizes area; ties prefer the plane nearest the Z centroid and then the
lower Z index.

Incomplete runs retain atomic partial Parquet files in a hidden sibling
directory and resume only when the label path, shape, chunks, dtype, artifact
key, metadata hash, and `--offset-zyx` match. At most `--workers` decoded label
chunks and worker results are active concurrently. Resume assumes label chunks
remain immutable; it does not hash the complete label volume. Successful runs
atomically publish the output and remove their partial directory.

## Training configuration

Create `TRAINING_ROOT/models/MODEL_NAME.json` before running `train`. A minimal
configuration is:

```json
{
  "base_model": null,
  "backend": "sam",
  "channels": [1, 2],
  "training_paths": ["sample"],
  "n_epochs": 200,
  "batch_size": 16,
  "skip_trt": false
}
```

`backend` accepts `sam` or `unet`. When `base_model` is `null`, training uses
`cpsam` for the SAM backend and `cyto3` for the UNet backend.

`training_paths` and the optional `test_folder` are resolved relative to
`TRAINING_ROOT`. They may name files or directories, but they cannot contain
`..` path components. Invalid paths stop training rather than silently falling
back to a random train/test split. The optional `include` and `exclude` arrays
contain regular expressions applied to discovered image paths.

Configuration is strict: unknown keys are errors. The removed keys `name`,
`bsize`, `SGD`, `optimizer`, `use_te`, and `te_fp8` are not accepted. Use the
`--packed` and `--skip-trt` command options to override those two runtime modes.

Unless TensorRT generation is skipped, training creates an ONNX model and a
device-specific TensorRT plan named
`MODEL_NAME-SANITIZED_CUDA_DEVICE_NAME.plan`. Inference selects the plan for the
current CUDA device, so a plan built for one GPU model should not be reused on
another GPU model.

## Development checks

From the repository root with `multi` active:

```bash
python -m pytest -q segment/tests
ruff check --no-cache segment/squisher_segment segment/tests
```
