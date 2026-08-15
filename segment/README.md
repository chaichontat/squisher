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
squisher-segment train TRAINING_ROOT MODEL_NAME
squisher-segment segment run INPUT_ZARR
squisher-segment segment stitch TEMP_DIR OUTPUT_ZARR
squisher-segment postproc run INPUT_ZARR
```

- `extract` samples registered TIFF or fused Zarr data for model training.
- `train` reads `TRAINING_ROOT/models/MODEL_NAME.json`, trains the model, and
  writes `MODEL_NAME.trained.json` beside the input configuration.
- `segment run` performs distributed Cellpose inference. Add `--cellpose-only`
  to retain the unstitched output for a separate stitching step.
- `segment stitch` converts a Cellpose temporary directory into the final Zarr
  segmentation.
- `postproc run` smooths and filters an existing 3D segmentation.

Training, segmentation, and postprocessing require CUDA-capable workers.

`segment run --channels` selects and orders the named input channels before
inference. Resume state, normalization data, nonempty-block caches, and the
output-specific `.done` manifest are bound to the input metadata, selected
channels, model checksum, and evaluation settings. Input Zarr stores must not
be modified in place while a run is resumable; use `--overwrite` after changing
input chunk contents. `segment stitch` also requires `--overwrite` before it
will replace an existing output.

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
