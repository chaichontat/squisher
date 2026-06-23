# squisher-lightsheet

Lightsheet stitching workflow commands for metadata positioning, rough center-z phase alignment,
registration, fusion, pyramid generation, QC, and 405-to-488 alignment orchestration.

```bash
uv run --package squisher-lightsheet lightsheet --help
```

Reusable seam registration primitives live in `squisher_lightsheet.seams`. They cover
overlap-plus-margin seam sampling, masked phase correlation, robust-boundary settings,
and boundary constraint records used by both legacy robust-boundary registration and
405-to-488 overlap recovery workflows.

The 405-to-488 workflow wrappers are exposed under:

```bash
lightsheet align-405-to-488 --help
```

For ambiguous local seam fixes, render explicit candidate correction grids instead of
editing position files by hand:

```bash
lightsheet align-405-to-488 render-candidate-grid \
  --position-input 405.to488.optimized.positions.json \
  --registration-input 405-to488.optimized.registration.json \
  --candidate-json top_left_candidates.json \
  --output-dir top_left_basin_grid \
  --channel 0 \
  --level 4 \
  --render-jobs 2
```

`top_left_candidates.json` is a Cartesian-product spec. Correction values are
`z,y,x` pixel deltas at the same native pixel scale as the position file:

```json
{
  "tiles": [
    {
      "tile": "230Tnc-CL-405.001.ome.tif",
      "label": "L001",
      "candidates": {
        "smallY": [0.2, 9.15, -0.15],
        "horiz": [2.2, 22.95, -89.95]
      }
    },
    {
      "tile": "230Tnc-CL-405.005.ome.tif",
      "label": "L005",
      "candidates": {
        "zero": [0.0, 0.0, 0.0],
        "minusY58_plusX22": [5.2, -57.9, 22.3]
      }
    }
  ]
}
```

The helper writes one variant folder per combination, plus
`candidate_grid_summary.json`, `candidate_grid_sheet.png`, and
`candidate_grid_boundaries_sheet.png`.

These commands require explicit input paths and do not hardcode local dataset locations.
