# squisher

`squisher` converts Zeiss CZI files into lossy, tiled OME-TIFF files using TIFF compression tag `22610` (JPEG-XR). It is built for archiving compressed microscopy data as a version of record while preserving the CZI metadata needed for provenance.

## What It Writes

For a single-image CZI:

```text
sample.czi -> sample.ome.tif
```

For a tiled CZI, each native CZI tile is written separately:

```text
sample.czi -> sample.000.ome.tif
sample.czi -> sample.001.ome.tif
```

The output files are tiled BigTIFF OME-TIFFs. Each file includes:

- JPEG-XR compression tag `22610`
- OME `Pixels`, `Plane`, and physical size metadata
- OME plane positions from native CZI tile positions
- optional plane positions from `<stem>_placement.json`
- OME `MapAnnotation` provenance fields
- raw CZI global metadata and per-tile subblock metadata in a linked OME `XMLAnnotation`

## Install

This repo uses `uv` for dependency management.

```bash
git clone https://github.com/chaichontat/squisher.git
cd squisher
uv sync --locked --all-groups
```

## Usage

Compress a CZI at quality level 90:

```bash
uv run squisher compress sample.czi --level 90
```

Useful options:

```bash
uv run squisher compress sample.czi \
  --level 90 \
  --tile-size 512 \
  --tiff-maxworkers 4 \
  --czi-tile-workers 1
```

Resume a previous completed run without rewriting complete outputs:

```bash
uv run squisher compress sample.czi --resume
```

Existing outputs are not overwritten unless `--resume` is used, and incomplete existing outputs still cause the command to fail.

## Placement JSON

If a file named `<stem>_placement.json` exists next to the CZI, `squisher` uses it for OME plane positions. The expected shape is:

```json
{
  "version": 1,
  "placement": {
    "origins": [
      {"index_zyx": [0, 0, 0], "origin_zyx": [0.0, 11.5, 22.5]}
    ]
  }
}
```

`index_zyx` must match the native CZI tile grid. `origin_zyx` is stored as OME `PositionY` and `PositionX`.

## Development

Run the test suite and lint checks with `uv`:

```bash
uv lock --check
uv sync --locked --all-groups
uv run ruff check --output-format=concise src tests
uv run pytest -q
```

CI runs the same checks on every push and pull request.

## License

MIT
