# lightsheet-psf

Empirical lightsheet PSF utilities for bead-derived PSF measurement, axial shear
measurement, and radialized PSF cleanup.

The package intentionally has no dataset defaults. Every command takes explicit
input paths, output paths, channel selection when needed, and physical voxel
spacing for reports that emit micron-scale measurements.

## Intended Workflow

1. Detect bead candidates from the acquisition.
2. Build a median PSF from isolated, unsaturated bead crops.
3. Render centroid QC crops and inspect rejected/accepted examples.
4. Measure XZ shear from the median and accepted crop distribution.
5. Deskew, radialize in XY, and reskew the PSF.
6. Inspect the final radialization QC and report before using the PSF.

Run commands from the monorepo root:

```bash
cd ~/squisher
```

## 1. Detect Beads

Use the wavelength/channel that matches the PSF you want. Multi-channel CZI or
OME-TIFF inputs require `--channel`; single-channel inputs do not.

```bash
uv run --package lightsheet-psf lightsheet-psf detect-beads \
  /path/to/Image.czi \
  --channel 0 \
  --fwhm 1.5 \
  --output /path/to/out/Image_bead_centers_fwhm1p5.csv \
  --qc /path/to/out/Image_bead_centers_fwhm1p5_qc.png
```

The QC PNG is a max-intensity XY projection with detected bead positions.

## 2. Build Median PSF

Start with the historical quality filter:

```bash
uv run --package lightsheet-psf lightsheet-psf make-median \
  /path/to/Image.czi \
  /path/to/out/Image_bead_centers_fwhm1p5.csv \
  --channel 0 \
  --crop-shape 21,21,21 \
  --size-mad-mult 2.0 \
  --z-asym-mad-mult 1.5 \
  --z-near-ratio-floor 0.35 \
  --prefix /path/to/out/Image_fwhm1p5_good_median_21x21x21
```

If centroid QC shows padded crops, discontinuities, or clear axial streaks, add
explicit artifact filters. Prefer `--require-full-crop` first. Use
`--max-z-support-span-px` only after inspecting QC; tighter values remove
streaks but can bias the final Z FWHM downward.

```bash
uv run --package lightsheet-psf lightsheet-psf make-median \
  /path/to/Image.czi \
  /path/to/out/Image_bead_centers_fwhm1p5.csv \
  --channel 0 \
  --crop-shape 21,21,21 \
  --size-mad-mult 2.0 \
  --z-asym-mad-mult 1.5 \
  --z-near-ratio-floor 0.35 \
  --require-full-crop \
  --max-z-support-span-px 16 \
  --prefix /path/to/out/Image_fwhm1p5_good_median_21x21x21_fullcrop_maxzspan16
```

This writes:

- `<prefix>_quality.csv`
- `<prefix>_raw.tif`
- `<prefix>_peaknorm.tif`
- `<prefix>_crops_peaknorm.npy`
- `<prefix>_qc.png`

## 3. Render Centroid QC

Render representative accepted crops before trusting the median. The sheet
includes XY, ZY, and ZX max projections with the fitted centroid overlay.

```bash
uv run --package lightsheet-psf lightsheet-psf render-centroid-qc \
  --image /path/to/Image.czi \
  --channel 0 \
  --quality-csv /path/to/out/Image_fwhm1p5_good_median_21x21x21_fullcrop_maxzspan16_quality.csv \
  --crop-shape 21,21,21 \
  --require-full-crop \
  --output-png /path/to/out/Image_centroid_qc_spots.png \
  --output-csv /path/to/out/Image_centroid_qc_spots.csv
```

Reject a filtering setup if accepted examples still contain obvious
discontinuities or long streaks. Also reject a setup if the filter is so tight
that it selects only unusually compact axial profiles.

## 4. Measure Shear

Voxel spacing is `Z,Y,X` in microns. Use acquisition metadata, not filenames.

```bash
uv run --package lightsheet-psf lightsheet-psf measure-shear \
  --median-psf /path/to/out/Image_fwhm1p5_good_median_21x21x21_fullcrop_maxzspan16_peaknorm.tif \
  --crops-npy /path/to/out/Image_fwhm1p5_good_median_21x21x21_fullcrop_maxzspan16_crops_peaknorm.npy \
  --quality-csv /path/to/out/Image_fwhm1p5_good_median_21x21x21_fullcrop_maxzspan16_quality.csv \
  --spacing-zyx-um 0.2,0.2878,0.2878 \
  --mode central_y \
  --output-json /path/to/out/Image_shear.json \
  --output-png /path/to/out/Image_shear.png
```

The JSON records package versions, input hashes, spacing, median shear, and the
accepted-crop shear distribution.

## 5. Radialize

Radialization deskews the source PSF by the measured XZ slope, enforces an exact
XY radial average, then reskews into the original coordinate frame.

```bash
uv run --package lightsheet-psf lightsheet-psf radialize \
  --peak-psf /path/to/out/Image_fwhm1p5_good_median_21x21x21_fullcrop_maxzspan16_peaknorm.tif \
  --raw-psf /path/to/out/Image_fwhm1p5_good_median_21x21x21_fullcrop_maxzspan16_raw.tif \
  --out-dir /path/to/out/radialized \
  --spacing-zyx-um 0.2,0.2878,0.2878 \
  --shear-mode central_y
```

The final deconvolution PSF is usually:

```text
radialized/radial_symmetric_reskewed_sumnorm.tif
```

The radialize command also writes peak-normalized and raw-scale variants, QC
PNGs, and `deskew_radial_reskew_report.json`.

## Quality Notes

- Use `--require-full-crop` to exclude padded crops with discontinuities.
- Use axial support-span filters to remove obvious streaks, but compare Z FWHM
  across several thresholds before choosing a final PSF.
- Do not compare FWHM across acquisitions unless voxel spacing and channel are
  confirmed.
- If a two-channel CZI is used, inspect metadata and pass the correct
  `--channel`.
- Treat surprisingly small Z FWHM as a possible filter-selection artifact until
  centroid QC and the accepted-crop distribution support it.

## Command Reference

```bash
uv run --package lightsheet-psf lightsheet-psf detect-beads --help
uv run --package lightsheet-psf lightsheet-psf make-median --help
uv run --package lightsheet-psf lightsheet-psf render-centroid-qc --help
uv run --package lightsheet-psf lightsheet-psf measure-shear --help
uv run --package lightsheet-psf lightsheet-psf radialize --help
```
