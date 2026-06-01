# squisher

Lossy CZI to tiled OME-TIFF compression.

```bash
uv sync
uv run squisher compress sample.czi --level 90 --tile-size 512
```

For multi-tile CZI files, each native tile is written as a separate OME-TIFF:

```text
sample.000.ome.tif
sample.001.ome.tif
```

Each output stores:

- JPEG-XR TIFF compression tag `22610`
- OME `Plane` positions from native CZI tile positions or `<stem>_placement.json`
- OME `MapAnnotation` provenance fields
- raw CZI global and per-tile subblock metadata in a linked OME `XMLAnnotation`
