# squisher-deconv

`squisher-deconv` streams flattened TIFF stacks as logical `(Z, C, Y, X)` volumes,
computes global scaling from uniformly sampled deconvolved planes, and writes
JPEG-XR OME-TIFF outputs directly without a full float32 staging pass.

Commands:

```bash
squisher-deconv sample-scale INPUT... --out-dir DIR --planes N --channels C --psf PSF.tif
squisher-deconv run INPUT... --out-dir DIR --scaling DIR/scaling.json --output-mode u16
```

In the `multi` conda environment before installation, run the module directly:

```bash
PYTHONPATH=/home/chaichontat/squisher/deconv/src python -m squisher_deconv sample-scale ...
PYTHONPATH=/home/chaichontat/squisher/deconv/src python -m squisher_deconv run ...
```
