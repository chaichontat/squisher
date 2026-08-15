# squisher-deconv

`squisher-deconv` streams flattened TIFF stacks as logical `(Z, C, Y, X)` volumes,
computes global scaling from uniformly sampled deconvolved planes, and writes
chunked CZYX OME-Zarr outputs directly without a full float32 staging pass.

Commands:

```bash
squisher-deconv basic INPUT... --out-dir BASIC_DIR --label SAMPLE --channels 2 --device cuda
squisher-deconv sample-scale INPUT... --out-dir DIR --planes N --channels 2 --psf PSF-c0.tif --psf PSF-c1.tif --iter 1
squisher-deconv run INPUT... --out-dir DIR --channels 2 --psf PSF-c0.tif --psf PSF-c1.tif --scaling DIR/scaling.json --output-mode u16 --iter 1
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
