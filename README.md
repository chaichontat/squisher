# squisher

Lossy CZI to tiled OME-TIFF compression with JPEG-XR TIFF tag `22610`.

`squisher` is meant for compressed microscopy archives that can replace the original CZI as the record copy. It writes OME-TIFF, preserves parsed OME metadata, and stores raw CZI global and per-tile subblock metadata in a linked OME `XMLAnnotation`.

## Install

```bash
git clone https://github.com/chaichontat/squisher.git
cd squisher
uv sync --locked --all-groups
```

## Usage

Compress one CZI:

```bash
uv run squisher compress sample.czi --level 90
```

Use explicit tiling and workers:

```bash
uv run squisher compress sample.czi \
  --level 90 \
  --tile-size 512 \
  --tiff-maxworkers 4 \
  --czi-tile-workers 8
```

Resume without rewriting complete outputs:

```bash
uv run squisher compress sample.czi --resume
```

Verify before treating the OME-TIFFs as the record copy:

```bash
uv run squisher verify sample.czi --decode-samples
```

Output naming:

```text
sample.czi          -> sample.ome.tif
multi_tile.czi      -> multi_tile.000.ome.tif, multi_tile.001.ome.tif, ...
```

If `<stem>_placement.json` is present, tile origins are used for OME `PositionX` and `PositionY`.

## Full Dataset Notes

For large tiled CZI files, run from the output directory with a symlink to the source CZI so outputs are written beside the link:

```bash
ln -s /data/sample.czi sample.czi
uv run squisher compress sample.czi \
  --level 90 \
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

## License

MIT
