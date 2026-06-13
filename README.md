# squisher

Lossy CZI to tiled OME-TIFF or OME-Zarr compression with imagecodecs-backed codecs.

`squisher` is meant for compressed microscopy archives that can replace the original CZI as the record copy when lossy output is acceptable. It writes OME-TIFF with JPEG-XR TIFF tag `22610`, or OME-Zarr with imagecodecs-backed JPEG-XR/JPEG-XL chunks. Every OME-TIFF tile stores the full shared CZI global metadata once, in a linked `squisher/czi/shared-metadata` OME `XMLAnnotation`; tile-specific CZI subblock metadata is stored once, in a linked `squisher/czi/raw-metadata` OME `XMLAnnotation`. OME-Zarr output stores the same shared and tile-specific CZI XML metadata in root attributes.
Each output also records squisher provenance and compression settings in OME `MapAnnotation`
entries, including the JPEG-XR tag, normalized quality level, tile size, worker settings,
output directory, and a JSON settings blob.

The default `--level 0.7` is lossy. Use `--level 1` or `--level 100` when you need
lossless JPEG-XR behavior, then verify with strict sample thresholds.

## Install

```bash
git clone https://github.com/chaichontat/squisher.git
cd squisher
uv sync --locked --all-groups
```

## Usage

Compress one CZI:

```bash
uv run squisher compress sample.czi
```

By default, compression also writes one center-z PNG thumbnail beside each OME-TIFF:

```text
sample.ome.tif -> sample.center-z.png
```

Use explicit tiling and workers:

```bash
uv run squisher compress sample.czi \
  --level 0.7 \
  --out-dir compressed \
  --tile-size 512 \
  --thumbnail-size 512 \
  --tiff-maxworkers 4 \
  --czi-tile-workers 8
```

Write OME-Zarr instead of OME-TIFF:

```bash
uv run squisher compress sample.czi \
  --output-format ome-zarr \
  --zarr-compressor jpegxr \
  --zarr-chunk-z 1 \
  --zarr-chunk-y 4096 \
  --zarr-chunk-x 4096
```

OME-Zarr output enforces a minimum requested spatial chunk size so a run cannot
accidentally create millions of tiny chunk files. The default is
`--min-zarr-chunk-pixels 16777216`. JPEG-XR chunks are restricted to
`--zarr-chunk-z 1`; use `--zarr-compressor jpegxl` if you need multi-Z chunks.

Disable thumbnail output when only the archival OME-TIFFs are needed:

```bash
uv run squisher compress sample.czi --no-thumbnails
```

When all expected output names already exist, compression skips them automatically. Partial
existing output sets are rejected; run `verify` to check output integrity:

```bash
uv run squisher compress sample.czi
```

Overwrite existing or incomplete outputs explicitly:

```bash
uv run squisher compress sample.czi --overwrite
```

Verify before treating the OME-TIFFs as the record copy:

```bash
uv run squisher verify sample.czi --decode-samples
```

`--decode-samples` decodes the first, middle, and last page of each output tile, reads the
matching planes from the source CZI, and logs `max_abs`, `mae`, and `rmse` differences.
For stricter checks, add thresholds:

```bash
uv run squisher verify sample.czi \
  --decode-samples \
  --max-sample-mae 20 \
  --max-sample-max-abs 128
```

Compare crop-level compression quality across a JPEG-XR level sweep:

```bash
uv run squisher compare sample.czi \
  --count 6 \
  --crop-size 256 \
  --min-level 0.65 \
  --max-level 0.90 \
  --level-step 0.05
```

`compare` writes labeled two-row PNG figures plus `manifest.json` and `size_metrics.csv`.
Each figure shows raw CZI crops on the top row, aligned compressed-minus-raw diffs on the
bottom row, and per-level size/error metrics in the column titles.
When a CZI produces more than one output, output is grouped under `<czi-stem>/` by default.
If `--out-dir` is provided, it is treated as the parent directory and the same `<czi-stem>/`
folder is created there.

Output naming:

```text
sample.czi          -> sample.ome.tif
multi_tile.czi      -> multi_tile/multi_tile.000.ome.tif, multi_tile/multi_tile.001.ome.tif, ...
multi_illum.czi     -> multi_illum/multi_illum.i000.ome.tif, multi_illum/multi_illum.i001.ome.tif, ...
tile_illum.czi      -> tile_illum/tile_illum.000.i000.ome.tif, tile_illum/tile_illum.000.i001.ome.tif, ...
sample.czi          -> sample.ome.zarr                    # with --output-format ome-zarr
multi_tile.czi      -> multi_tile/multi_tile.000.ome.zarr, multi_tile/multi_tile.001.ome.zarr, ...
```

For any CZI that produces multiple outputs, `--out-dir compressed` writes into
`compressed/<czi-stem>/`. A CZI that produces exactly one output writes directly under
`compressed/`.

If `<stem>_placement.json` is present, tile origins are used for OME `PositionX` and `PositionY`.
Use `--pos BL.pos` to infer actual Zeiss stage positions from a position-list file. The
first four positions are treated as the M=0 tile field-of-view, and all mosaic tile
positions are inferred from their CZI bbox offsets relative to M=0. Inferred stage
`PositionX`, `PositionY`, and `PositionZ` are written in micrometers and also recorded
in OME `MapAnnotation`.

Supported CZI layout:

- Arbitrary `C`, `Z`, and `T` planes.
- Mosaic `M` tiles written as separate OME-TIFF files.
- Arbitrary `I` illuminations written as separate OME-TIFF or OME-Zarr outputs.
- Exactly one scene (`S`) and only singleton `R`, `H`, `V`, and `B` dimensions.

Inputs outside that layout are rejected before writing output.

## Full Dataset Notes

For large tiled CZI files, run from the output directory with a symlink to the source CZI so the
stem-named output folder is written beside the link:

```bash
ln -s /data/sample.czi sample.czi
uv run squisher compress sample.czi \
  --level 0.7 \
  --tile-size 512 \
  --czi-tile-workers 8 \
  --tiff-maxworkers 4
uv run squisher verify sample.czi --decode-samples
```

Future large runs should try more tile-level parallelism first, for example `--czi-tile-workers 16 --tiff-maxworkers 4`, while watching memory and disk throughput.

## Development

```bash
uv lock --check
uv run ruff check --output-format=concise src tests
uv run pytest -q
```

Build the standalone Windows executable from Windows:

```powershell
pwsh ./scripts/build-windows-exe.ps1
```

The executable is written to `dist/squisher.exe`. GitHub Actions also publishes it as the
`squisher-windows-exe` artifact.

## License

MIT
