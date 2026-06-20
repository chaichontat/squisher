# Lightsheet Cleanup Plan

This plan focuses on code cleanup in `~/squisher/lightsheet`. It intentionally omits generated-cache cleanup.

## 1. Extract Shared Image Metrics

Owner: `src/squisher_lightsheet/tile_phase.py` and `src/squisher_lightsheet/track_z.py`

Both modules implement the same normalization and masked-correlation behavior:

- `tile_phase.normalize_volume_for_phase`
- `tile_phase.corrcoef_on_mask`
- `track_z.robust_normalize`
- `track_z.corrcoef_masked`

Create a small shared module, likely `src/squisher_lightsheet/image_metrics.py`, with:

- robust percentile normalization for positive finite image data
- masked Pearson correlation with the existing minimum-voxel and zero-variance behavior

Keep the first pass behavior-preserving. Move call sites one at a time and run the existing `test_tile_phase.py` and `test_track_z.py` coverage after each move.

## 2. Split Tile-Phase Orchestration

Owner: `src/squisher_lightsheet/tile_phase.py`

`align_tiles_to_reference` currently owns validation, cache loading, per-tile record mutation, direct measurement, cached-attempt rescore, failure recording, fallback inference, output writing, and optional registration adaptation.

Extract the repeated mechanics before changing algorithm behavior:

- a helper that applies an accepted shift to a position record
- a helper that builds accepted measurement rows
- a helper that builds failed-attempt rows
- a helper that writes cache state after row changes
- a small per-tile result object for accepted, failed, and fallback outcomes

The desired invariant is that each tile path has exactly one final outcome row, and only accepted rows mutate the output position payload.

## 3. Type Tile-Phase Measurement Rows

Owner: `src/squisher_lightsheet/tile_phase.py`

The tile-phase cache and summary use plain dictionaries with repeated status strings:

- `direct_accepted`
- `direct_failed`
- `fallback_accepted`

Introduce explicit types for the row contract. A narrow option is:

- `MeasurementStatus = Literal["direct_accepted", "direct_failed", "fallback_accepted"]`
- `TypedDict` definitions for accepted rows and failed rows

This should make `adapt_registration_from_reference` and cache loading reject invalid rows through one owner instead of checking ad hoc dictionary shapes at each call site.

## 4. Tighten the Legacy Boundary

Owner: `src/squisher_lightsheet/_legacy/` plus package modules that import it

The package still tests and imports `_legacy` directly. Examples include:

- `tests/test_tile_phase.py`
- `tests/test_legacy_reference_geometry.py`
- `tests/test_fusion.py`

Promote the tested legacy contracts into package-owned modules before shrinking `_legacy`:

- geometry and tile-record helpers used by rough phase, tile phase, and track-z diagnostics
- phase-correlation helpers used by tile phase and registration tests
- reference-geometry solver contracts covered by `test_legacy_reference_geometry.py`

Keep `_legacy` as the compatibility layer for copied scripts until the package modules own the tested behavior. Do not delete legacy code until each promoted contract has package-level tests.

## 5. Standardize Legacy Command Assembly

Owner: `src/squisher_lightsheet/legacy_runner.py`, `registration.py`, `fusion.py`, `qc.py`, and `pyramid.py`

Several wrapper modules build command arguments by hand. Add small helpers in `legacy_runner.py` for common argument patterns:

- append `--flag value` only when the value is not `None`
- append boolean flags using the existing positive and negative flag names
- append list-valued options such as channel lists

Keep the helper intentionally small. It should reduce repeated argument-building code without hiding the command shape from each wrapper.

## Validation

Use the existing focused tests as guardrails:

- `uv run --package squisher-lightsheet pytest lightsheet/tests/test_tile_phase.py`
- `uv run --package squisher-lightsheet pytest lightsheet/tests/test_track_z.py`
- `uv run --package squisher-lightsheet pytest lightsheet/tests/test_registration.py lightsheet/tests/test_fusion.py lightsheet/tests/test_workflow.py`

For each cleanup step, keep the diff small and run only the tests that cover the touched owner. Run the full lightsheet test suite after the legacy-boundary cleanup.
