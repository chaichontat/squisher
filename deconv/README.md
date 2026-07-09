# squisher-deconv

`squisher-deconv` streams flattened TIFF stacks as logical `(Z, C, Y, X)` volumes,
computes global scaling from uniformly sampled deconvolved planes, and writes
chunked CZYX OME-Zarr outputs directly without a full float32 staging pass.

Commands:

```bash
squisher-deconv sample-scale INPUT... --out-dir DIR --planes N --channels 2 --psf PSF-c0.tif --psf PSF-c1.tif --iter 1
squisher-deconv run INPUT... --out-dir DIR --channels 2 --psf PSF-c0.tif --psf PSF-c1.tif --scaling DIR/scaling.json --output-mode u16 --iter 1
```

In the `multi` conda environment before installation, run the module directly:

```bash
PYTHONPATH=/home/chaichontat/squisher/deconv/src python -m squisher_deconv sample-scale ...
PYTHONPATH=/home/chaichontat/squisher/deconv/src python -m squisher_deconv run ...
```
