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
  --czi-tile-workers 1
```

Resume without rewriting complete outputs:

```bash
uv run squisher compress sample.czi --resume
```

Output naming:

```text
sample.czi          -> sample.ome.tif
multi_tile.czi      -> multi_tile.000.ome.tif, multi_tile.001.ome.tif, ...
```

If `<stem>_placement.json` is present, tile origins are used for OME `PositionX` and `PositionY`.

## Development

```bash
uv lock --check
uv run ruff check --output-format=concise src tests
uv run pytest -q
```

## License

MIT
