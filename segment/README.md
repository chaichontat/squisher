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

## Distributed segmentation

`segment run` accepts either a materialized ZYXC Zarr or a
`squisher_segment.registered_ome_input.v1` manifest. A manifest names completed,
channel-separated ZYX OME-Zarrs and a shared `source_roi_zyx`; workers stack only
the requested crop, avoiding a full intermediate multichannel volume.

`--channels` fixes model channel order. The same order is used for bounded,
cached normalization, background masking, and training exports. On a nonempty
cache miss, a core runs only when at least 5% of raw channel `561` is strictly
greater than `--nonempty-threshold`. Inside selected cores, voxels at or below
that threshold are set to each channel's normalization baseline before
Cellpose. All-zero internal XY, ZX, and ZY tiles are skipped as a second,
finer-grained empty-data check.

The SAM backend uses `pretrained_model` for XY and, when configured,
`pretrained_model_ortho` for ZX and ZY. Each checkpoint needs a compatible,
device-specific TensorRT plan for every participating GPU type; a missing
orthogonal artifact never falls back to the XY model.

### Reference profile and rationale

These settings are measured production choices, not universal defaults:

```bash
squisher-segment segment run registered.json \
  --channels 405,561,638 \
  --config config.json \
  --workers-per-device 2,2 \
  --threads-per-worker 1 \
  --target-nz 2 \
  --target-ny 4 \
  --target-nx 4 \
  --nonempty-threshold 1000 \
  --no-cellpose-only \
  --stagger-seconds 0
```

The matching reference model configuration is:

```json
{
  "backend": "sam",
  "diameter": 30,
  "flow3D_smooth": 1.0,
  "ortho_weights": [1.0, 1.0, 1.0],
  "pretrained_model": "/models/embryonicsheet",
  "pretrained_model_ortho": "/models/embryonicsheet"
}
```

| Choice | Reason |
| --- | --- |
| `diameter=30`; overlap `2 * diameter` | Diameter sets Cellpose scale. A 60-pixel halo gives boundary cells context before predictions are trimmed to disjoint cores. |
| `flow3D_smooth=1.0`; weights `[1,1,1]` | Light smoothing reduces directional discontinuities, while equal weights avoid privileging XY over ZX or ZY. Both alter masks and are identity-bound. |
| Target tiles `(2,4,4)` | Resolves to a `(280,712,712)` ZYX core and `(400,832,832)` interior crop with halo. This bounds the working set without shrinking Z depth solely to avoid OOM. |
| `--workers-per-device 2,2` | Three synchronized RTX 5090 workers entered `follow_flows` with 109M, 149M, and 156M active pixels and exhausted 31.36 GiB on a required 596 MiB update. Two workers per GPU bound that phase peak. Worker count is a memory limit, not a throughput target. |
| `--threads-per-worker 1` | GPU work dominates; one CPU thread per process avoids host oversubscription across workers. |
| `--stagger-seconds 0` | With measured-safe worker counts, staggering adds idle time without reducing the later synchronized dynamics peak. |
| Resume each block independently | A block owns a disjoint trimmed core and persists its overlap sidecar before checkpointing. Every checkpointed planned block is retained after interruption; remaining blocks are fungible and submitted together. |
| Derive temporary ID bits from the block grid | A fixed 16-bit local field failed at 75,414 labels. The 1,620-block reference grid needs 11 block bits and leaves 21 local bits. Encoding, decoding, and sparse stitching share this split; final labels remain dense `uint32`. |
| Bind state to inputs, models, plans, source, and evaluation | Mixing incompatible blocks can corrupt a segmentation. Resume fails closed; override a source fingerprint only after proving block results are unchanged. |
| `--no-cellpose-only` | Run stitching after inference. Use `--cellpose-only` only when a separately managed stitch step is intentional. |

### Stitching modes

Face contact remains the default and preserves the established behavior:

```bash
squisher-segment segment run registered.json --stitch-mode face
```

`--stitch-mode overlap-iou` is an alternate identity rule for labels that
disagree at a block boundary. Before inference, the run allocates three sparse,
pair-centric Zarr arrays. Workers store their local `uint32` labels over the
full shared halo, limited transversely to the owned core. Pair-side chunks have
one writer, use Zstd level 1 without shuffle, and stay at or below 32 MiB.
After all blocks finish, matching streams those chunks and merges only unique
reciprocal-best pairs whose shared-band IoU is at least
`--stitch-iou-threshold` (default `0.25`). Exact ties remain separate. The
overlap evidence changes identity only; the disjoint core foreground written by
Cellpose is not filled, erased, or otherwise changed.

Each completed block publishes an evidence marker before its driver checkpoint.
On resume, checkpointed blocks require matching metadata, run identity, and
markers; uncheckpointed retries overwrite only their exclusively owned
pair-sides. The stitch mode and threshold are part of `run_config.json`, so a
later standalone `segment stitch` uses the same inference-time decision.

The current full-volume planning estimate is 298.18 GiB of logical evidence
before compression. On one cell-rich `120x712x712` band, no-shuffle Zstd level 1
stored 6.62% of logical bytes versus 17.72% with bitshuffle and projects to
19.7 GiB for that estimate. This is a bounded representative benchmark, not a
whole-run guarantee. Treat 25 GiB stored evidence, 512 MiB incremental worker
RSS, 10% inference overhead, and 30 minutes for matching as acceptance limits.
The matcher logs logical bytes, candidate and accepted counts, elapsed time, and
peak chunk scratch memory for each run.

Re-profile `follow_flows` foreground counts and peak device memory before
increasing concurrency. Changing input contents, channel order, models, plans,
normalization, foreground policy, or evaluation settings invalidates resume
state. Input stores must remain immutable while temporary state is reusable.

### Iterative training loop

Use bounded ROI runs before a retrained model is applied to a full volume:

1. Choose a cell-rich ROI and run the same 3D inference configuration intended
   for production.
2. Run postprocessing with recorded parameters, then export both Z and
   orthogonal training stacks with paired `_masks.tif` files.
3. Export the same number and order of image channels that Cellpose received.
   Dropping a channel makes the cleanup set inconsistent with the model even
   when the masks remain readable.
4. Correct both views, retrain, and move to a new ROI so successive iterations
   expose errors in different tissue regions.
5. Record model checksums, inference and postprocessing settings,
   ROI coordinates, and export paths beside the run. Start the full run only
   when successive ROIs no longer reveal a systematic failure mode.

Cellpose and postprocessed outputs are label images, not probability maps. Keep
resumable temporary state until stitching succeeds, then postprocess stitched
labels and export cleanup images from that result.

## Parallel region properties

Run region measurements directly on a completed 3-D integer label Zarr:

```bash
squisher-segment regionprops postproc.zarr \
  --output props.parquet \
  --intensity edu=fused-514.ome.zarr \
  --intensity brdu=fused-638.ome.zarr \
  --workers 4
```

Repeat `--intensity NAME=PATH` for each coordinate-matched 3-D Zarr or OME-Zarr;
OME-Zarr inputs use their level-0 dataset. Every intensity source must match the
label shape and use the same dtype. Each process reads one label chunk and the
same bounds from every intensity source, then measures the multichannel block
in the existing regionprops pass. Polars reduces labels that cross Z, Y, or X
chunk boundaries without loading the full label volume. The output columns are
`label`, `area`, `centroid_z`, `centroid_y`, `centroid_x`, `plane_z`, and
`plane_area`, followed by
`intensity_NAME_min`, `intensity_NAME_mean`, and `intensity_NAME_max` for each
source. The representative plane maximizes area; ties prefer the plane nearest
the Z centroid and then the lower Z index.

Incomplete runs retain atomic partial Parquet files in a hidden sibling
directory and resume only when the label path, shape, chunks, dtype, artifact
key, metadata hash, and `--offset-zyx` match. When supplied, the intensity path,
resolved array path, shape, chunks, dtype, artifact key, and metadata hash must
also match. At most `--workers` decoded label and intensity chunks and worker
results are active concurrently. Resume assumes source chunks remain immutable;
it does not hash the complete volumes. Successful runs atomically publish the
output and remove their partial directory.

## Training configuration

Create `TRAINING_ROOT/models/MODEL_NAME.json` before running `train`. A minimal
configuration is:

```json
{
  "name": "MODEL_NAME",
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

Configuration is strict: unknown keys are errors. The supported `fishtools2`
fields `name`, `bsize`, `SGD`, and `optimizer` are accepted. The `bsize` and
`SGD` values are passed to Cellpose; `optimizer: "adamw"` selects AdamW by
disabling SGD. TensorRT plans retain the 256-pixel input profile required by
distributed inference. Use the `--packed` and `--skip-trt` command options to
override those two runtime modes.

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
