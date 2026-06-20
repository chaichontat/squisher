2# 405-to-488 Tile-Phase Alignment and Fusion Weighting Review Packet

## Problem

We are aligning a 405 nm lightsheet tile mosaic to an already-registered 488 nm reference mosaic for the same sample.

- Dataset root: `/home/chaichontat/nvme/lightsheet/20260613`
- Working output root: `/working/230Tnc`
- Reference channel/token: `488514561638`, channel index `3`
- Moving channel/token: `405`, channel index `0`
- Reference registration source: `/working/230Tnc/230Tnc-CLR-488514561638-registration/registration.track0.json`
- New 405 phase-adjusted positions: `/working/230Tnc/230Tnc-CLR-405.to488.tilePhase.l0patch.positions.json`
- New adapted 405 registration: `/working/230Tnc/230Tnc-CLR-405.to488.l0patch.registration.json`
- Measurement output directory: `/working/230Tnc/230Tnc-CLR-405-to488-tilePhase-l0patch32x512x512`

The intent is to place each 405 tile into the canonical 488 tile geometry without re-solving a new tile mosaic from 405-only overlap constraints. The current implementation estimates a per-tile 405-to-488 translation by phase-correlating each 405 tile against the same-index 488 tile, writes a new 405 position file, then adapts the 488 registration JSON by copying the optimized 488 `registered_affine` matrices exactly while replacing tile identity/path/stage placement with the phase-adjusted 405 position records.

Separately, fusion uses `content-preibisch-coarse` weighting. We recently added a softmax-like exponent to sharpen content weights in overlap regions, while preserving multiview-stitcher's geometric border blending.

## Desired Physical Behavior

- Pair each 488 tile with the corresponding 405 tile by replacing `488514561638` with `405` in path and tile name.
- Estimate one rigid z/y/x translation per 405 tile relative to its 488 counterpart.
- Use high-content level-0 patches for the final measurement, not low-resolution whole-tile estimates alone.
- Require at least two mutually consistent patch shifts per tile.
- Fail if a tile cannot be directly measured, except for an explicitly marked adjacent-tile inference fallback.
- Preserve the canonical 488 optimized affine matrices in the adapted 405 registration.
- Let the later fusion step handle intensity seams and multi-overlap blending; do not expect the dumb preview PNG to be seam-free.

## Current Run Summary

Command used:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/chaichontat/squisher/lightsheet/src \
CONDA_NO_PLUGINS=true conda run --no-capture-output -n multi \
python -u -m squisher_lightsheet.cli tile-phase-align \
  --reference-position /home/chaichontat/nvme/lightsheet/20260613/230Tnc-CLR-488514561638.overlap25.roughPhase.zSlab32.fullCanvas.positions.json \
  --output-position /working/230Tnc/230Tnc-CLR-405.to488.tilePhase.l0patch.positions.json \
  --output-dir /working/230Tnc/230Tnc-CLR-405-to488-tilePhase-l0patch32x512x512 \
  --reference-channel 3 \
  --reference-token 488514561638 \
  --moving-token 405 \
  --level 0 \
  --patch-shape-zyx 32,512,512 \
  --min-inliers 2 \
  --max-candidate-patches 24 \
  --coarse-level 4 \
  --reference-registration-input /working/230Tnc/230Tnc-CLR-488514561638-registration/registration.track0.json \
  --output-registration /working/230Tnc/230Tnc-CLR-405.to488.l0patch.registration.json
```

Runtime results from `tile_phase_alignment.json`:

| metric | value |
| --- | ---: |
| tile measurements | 50 |
| position records | 50 |
| adapted registration tiles | 50 |
| adjacent fallback count | 0 |
| min / median / max inliers | 2 / 2 / 2 |
| corr_after min / median / max | 0.047 / 0.713 / 0.932 |
| max adapted affine diff vs 488 registration | 0.0 |

The main weak tile is:

| tile | shift_px_zyx | corr_after | n_inliers |
| --- | --- | ---: | ---: |
| `230Tnc-CL-405.005.ome.tif` | `[-64.5, -167.0, -308.5]` | 0.047 | 2 |

Level-2 dumb placement preview:

- `/working/230Tnc/230Tnc-CLR-405-to488-tilePhase-l0patch32x512x512/dumb-stitch-preview-level2/level2_registered_centerZ_placement_ch0.png`
- Size: `3560 x 2329`, 8-bit grayscale
- This preview averages overlapping center-z pixels and does not run full fusion.

## Review Questions

Please evaluate both algorithmic validity and implementation correctness:

1. Is same-index 405-to-488 per-tile phase correlation an appropriate model, or should the cross-channel alignment be solved as a global field/graph instead?
2. Is the coarse level-4 seed plus level-0 residual patch phase-correlation composition correct?
3. Are the inlier thresholds reasonable: z <= 3 px, y/x <= 12 px at level 0?
4. Is requiring exactly at least two consistent patch measurements enough, or should the acceptance criteria also require a minimum peak/correlation/improvement?
5. Is early stopping after the first consistent inlier cluster statistically safe, given candidates are ranked by coarse high-content score?
6. Are there edge cases where filtering out coarse-shifted moving patches before ranking biases patch selection?
7. Is copying every 488 `registered_affine` exactly into the 405 registration, while replacing `stage_translation_um`, a correct representation of the intended geometry?
8. Should the low-confidence tile `230Tnc-CL-405.005.ome.tif` be accepted, flagged, remeasured with more candidates, or inferred from neighbors?
9. Is the adjacent-tile fallback conceptually sound if direct same-index 405-to-488 phase correlation fails?
10. For fusion, is applying `w_i ** exponent / sum(w ** exponent)` to content weights before geometric blending a reasonable way to reduce ghosting in overlaps?
11. Is exponent `2.0` a defensible default, or should this be tuned/validated quantitatively?
12. Does the current measurement cache key include enough inputs to prevent stale reuse?

## Algorithm Summary

For each 488 reference tile:

1. Resolve the 405 tile by token replacement in the path and tile name.
2. Read a coarse whole-tile 3D scout volume using TIFF SubIFDs where possible.
3. Phase-correlate 488 coarse scout against 405 coarse scout with CuPy FFT to estimate a coarse z/y/x shift.
4. Scale that coarse shift back to level-0 pixels.
5. Generate level-0 fixed-patch candidates from high-content regions in the coarse 488 scout.
6. Reject candidate patches whose coarse-shifted 405 moving patch would be out of bounds.
7. Read only level-0 `32x512x512` fixed and moving patches.
8. Phase-correlate each fixed patch against the coarse-shifted moving patch to estimate residual shift.
9. Compose total shift as `coarse_shift_l0_px + residual_shift_px`.
10. Cluster total patch shifts using per-axis thresholds and take the median of the selected inlier cluster.
11. Stop early once at least `min_inliers` measured patches form a consistent cluster.
12. Apply the final tile shift in microns to the 405 tile's stage `translation_um`.
13. Write a summary JSON and incremental cache after each tile.
14. Adapt the 488 registration JSON by copying 488 affine transforms and replacing tile names/paths/stage transforms with 405 records.

## Core Code Under Review

### Token Pairing

```python
def corresponding_moving_path(reference_path: Path, *, reference_token: str, moving_token: str) -> Path:
    text = str(reference_path)
    if reference_token not in text:
        raise ValueError(f"{reference_path} does not contain reference token {reference_token!r}")
    path = Path(text.replace(reference_token, moving_token))
    if not path.exists():
        raise FileNotFoundError(f"Corresponding moving tile does not exist: {path}")
    return path


def make_moving_tile_name(reference_tile: str, *, reference_token: str, moving_token: str) -> str:
    if reference_token not in reference_tile:
        raise ValueError(f"Tile name {reference_tile!r} does not contain reference token {reference_token!r}")
    return reference_tile.replace(reference_token, moving_token)
```

### Normalization and GPU Phase Correlation

```python
def normalize_volume_for_phase(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    positive = finite[finite > 0]
    out = np.zeros(volume.shape, dtype=np.float32)
    if positive.size == 0:
        return out
    low, high = np.percentile(positive, [1.0, 99.5])
    clipped = np.clip(volume, low, high)
    valid = np.isfinite(clipped)
    centered = clipped - float(np.median(clipped[valid]))
    denom = max(float(np.percentile(np.abs(centered[valid]), 95.0)), 1.0)
    out[valid] = centered[valid] / denom
    return out


def estimate_patch_shift_zyx_px(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import ndimage

    if fixed.shape != moving.shape:
        raise ValueError(f"Patch phase correlation requires matching shapes, got {fixed.shape} and {moving.shape}")
    fixed_norm = normalize_volume_for_phase(fixed)
    moving_norm = normalize_volume_for_phase(moving)
    try:
        shift, peak = stitch_legacy.phase_correlation_shift_gpu(fixed_norm, moving_norm)
    except Exception as exc:
        raise RuntimeError("CuPy patch phase correlation failed; patch mode requires GPU FFT support") from exc
    shift_array = np.asarray(shift, dtype=np.float64)
    shifted = ndimage.shift(moving_norm, shift=shift_array, order=1, mode="constant", cval=0.0, prefilter=False)
    finite = np.isfinite(fixed_norm) & np.isfinite(moving_norm)
    shifted_finite = np.isfinite(fixed_norm) & np.isfinite(shifted)
    return shift_array, {
        "shape_zyx": [int(value) for value in fixed_norm.shape],
        "peak": float(peak),
        "corr_before": corrcoef_on_mask(fixed_norm, moving_norm, finite),
        "corr_after": corrcoef_on_mask(fixed_norm, shifted, shifted_finite),
    }
```

### Candidate Patch Selection

```python
def candidate_patch_slices(
    scout_volume: np.ndarray,
    *,
    tile_shape_zyx: np.ndarray,
    patch_shape_zyx: tuple[int, int, int],
    scout_scale_zyx: np.ndarray,
    max_candidates: int,
    moving_shape_zyx: np.ndarray,
    shift_zyx_px: np.ndarray,
) -> list[dict[str, Any]]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if any(size > int(tile_size) for size, tile_size in zip(patch_shape_zyx, tile_shape_zyx, strict=True)):
        raise ValueError(
            f"Patch shape {list(patch_shape_zyx)} is larger than tile shape {tile_shape_zyx.tolist()}"
        )

    starts_by_axis = []
    for patch_size, tile_size in zip(patch_shape_zyx, tile_shape_zyx, strict=True):
        tile_size = int(tile_size)
        if tile_size == patch_size:
            starts_by_axis.append([0])
            continue
        step = max(1, patch_size // 2)
        starts = list(range(0, tile_size - patch_size + 1, step))
        if starts[-1] != tile_size - patch_size:
            starts.append(tile_size - patch_size)
        starts_by_axis.append(starts)

    ranked = []
    scout_norm = normalize_volume_for_phase(scout_volume)
    for z0 in starts_by_axis[0]:
        for y0 in starts_by_axis[1]:
            for x0 in starts_by_axis[2]:
                fixed_slices = (slice(z0, z0 + patch_shape_zyx[0]), slice(y0, y0 + patch_shape_zyx[1]), slice(x0, x0 + patch_shape_zyx[2]))
                moving_slices = shifted_slices_zyx(fixed_slices, shift_zyx_px=shift_zyx_px)
                if not slices_within_shape(moving_slices, moving_shape_zyx):
                    continue
                scout_slices = tuple(
                    slice(
                        max(0, int(np.floor(slc.start / scout_scale_zyx[axis]))),
                        min(int(scout_volume.shape[axis]), int(np.ceil(slc.stop / scout_scale_zyx[axis]))),
                    )
                    for axis, slc in enumerate(fixed_slices)
                )
                scout_patch = scout_norm[scout_slices]
                positive_fraction = float(np.count_nonzero(scout_patch > 0) / scout_patch.size) if scout_patch.size else 0.0
                score = float(np.std(scout_patch)) * max(positive_fraction, 1e-6)
                ranked.append(
                    {
                        "fixed_slices": fixed_slices,
                        "scout_slices": scout_slices,
                        "content_score": score,
                        "positive_fraction": positive_fraction,
                    }
                )
    ranked.sort(key=lambda item: item["content_score"], reverse=True)
    return ranked[:max_candidates]
```

### Inlier Selection

```python
PATCH_INLIER_THRESHOLDS_ZYX = np.asarray([3.0, 12.0, 12.0], dtype=np.float64)


def select_inlier_patch_measurements(
    total_shifts: np.ndarray,
    *,
    thresholds_zyx: np.ndarray = PATCH_INLIER_THRESHOLDS_ZYX,
    min_inliers: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if total_shifts.ndim != 2 or total_shifts.shape[1] != 3:
        raise ValueError(f"Expected n x 3 total shifts, got {total_shifts.shape}")
    if total_shifts.shape[0] == 0:
        raise ValueError("No accepted patch shifts are available for inlier selection")
    neighbor_masks = np.all(np.abs(total_shifts[:, None, :] - total_shifts[None, :, :]) <= thresholds_zyx, axis=2)
    counts = neighbor_masks.sum(axis=1)
    best_count = int(counts.max())
    best_indices = np.flatnonzero(counts == best_count)
    if best_indices.size > 1:
        medians = np.asarray([np.median(total_shifts[neighbor_masks[index]], axis=0) for index in best_indices])
        distances = np.asarray(
            [np.median(np.linalg.norm(total_shifts[neighbor_masks[index]] - median, axis=1)) for index, median in zip(best_indices, medians, strict=True)]
        )
        best_index = int(best_indices[int(np.argmin(distances))])
    else:
        best_index = int(best_indices[0])
    inliers = neighbor_masks[best_index]
    if int(np.count_nonzero(inliers)) < min_inliers:
        raise ValueError(f"Only {int(np.count_nonzero(inliers))} inlier patch shifts found; require {min_inliers}")
    return inliers, np.median(total_shifts[inliers], axis=0)
```

### Patch-Mode Tile Shift Measurement

```python
def measure_patch_tile_shift(
    *,
    reference_tile: rough_legacy.TileRecord,
    moving_tile: rough_legacy.TileRecord,
    reference_channel: int,
    patch_shape_zyx: tuple[int, int, int],
    coarse_level: int,
    upsample_factor: int,
    max_candidate_patches: int,
    min_inliers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fixed_coarse, fixed_coarse_scale_zyx, fixed_source_level, fixed_available_levels = sampled_tile_volume_from_subifd(
        reference_tile,
        channel=reference_channel,
        requested_level=coarse_level,
    )
    moving_coarse, moving_coarse_scale_zyx, moving_source_level, moving_available_levels = sampled_tile_volume_from_subifd(
        moving_tile,
        channel=0,
        requested_level=coarse_level,
    )
    if not np.array_equal(fixed_coarse_scale_zyx, moving_coarse_scale_zyx):
        raise ValueError(
            "Fixed and moving coarse scout scales differ: "
            f"{fixed_coarse_scale_zyx.tolist()} vs {moving_coarse_scale_zyx.tolist()}"
        )
    coarse_shift_coarse_px, coarse_details = estimate_tile_shift_zyx_px_gpu(fixed_coarse, moving_coarse)
    coarse_shift_l0_px = coarse_shift_coarse_px * fixed_coarse_scale_zyx
    candidates = candidate_patch_slices(
        fixed_coarse,
        tile_shape_zyx=reference_tile.shape_zyx,
        patch_shape_zyx=patch_shape_zyx,
        scout_scale_zyx=fixed_coarse_scale_zyx,
        max_candidates=max_candidate_patches,
        moving_shape_zyx=moving_tile.shape_zyx,
        shift_zyx_px=coarse_shift_l0_px,
    )
    patch_rows = []
    accepted_shift_rows = []
    accepted_patch_indices = []
    final_shift_px: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    early_stop_after_patch: int | None = None
    for patch_index, candidate in enumerate(candidates):
        fixed_slices = candidate["fixed_slices"]
        moving_slices = shifted_slices_zyx(fixed_slices, shift_zyx_px=coarse_shift_l0_px)
        row = {
            "patch_index": int(patch_index),
            "fixed_slices_zyx": slices_to_json(fixed_slices),
            "moving_slices_zyx": slices_to_json(moving_slices),
            "coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
            "content_score": float(candidate["content_score"]),
            "positive_fraction": float(candidate["positive_fraction"]),
        }
        if not slices_within_shape(moving_slices, moving_tile.shape_zyx):
            row.update(status="rejected", reason="moving_patch_out_of_bounds")
            patch_rows.append(row)
            continue
        fixed_patch = read_tile_patch(reference_tile, channel=reference_channel, slices_zyx=fixed_slices)
        moving_patch = read_tile_patch(moving_tile, channel=0, slices_zyx=moving_slices)
        if fixed_patch.shape != patch_shape_zyx or moving_patch.shape != patch_shape_zyx:
            row.update(
                status="rejected",
                reason="patch_shape_mismatch",
                fixed_shape_zyx=[int(value) for value in fixed_patch.shape],
                moving_shape_zyx=[int(value) for value in moving_patch.shape],
            )
            patch_rows.append(row)
            continue
        residual_shift_px, details = estimate_patch_shift_zyx_px(fixed_patch, moving_patch)
        total_shift_px = coarse_shift_l0_px + residual_shift_px
        row.update(
            status="accepted",
            reason="measured",
            residual_shift_px_zyx=[float(value) for value in residual_shift_px],
            total_shift_px_zyx=[float(value) for value in total_shift_px],
            peak=details["peak"],
            corr_before=details["corr_before"],
            corr_after=details["corr_after"],
            fixed_stats={
                "min": float(np.nanmin(fixed_patch)),
                "max": float(np.nanmax(fixed_patch)),
                "mean": float(np.nanmean(fixed_patch)),
                "std": float(np.nanstd(fixed_patch)),
            },
            moving_stats={
                "min": float(np.nanmin(moving_patch)),
                "max": float(np.nanmax(moving_patch)),
                "mean": float(np.nanmean(moving_patch)),
                "std": float(np.nanstd(moving_patch)),
            },
        )
        accepted_patch_indices.append(patch_index)
        accepted_shift_rows.append(total_shift_px)
        patch_rows.append(row)
        if len(accepted_shift_rows) >= min_inliers:
            try:
                inlier_mask, final_shift_px = select_inlier_patch_measurements(
                    np.vstack(accepted_shift_rows).astype(np.float64),
                    min_inliers=min_inliers,
                )
            except ValueError:
                final_shift_px = None
                inlier_mask = None
            else:
                early_stop_after_patch = patch_index
                break

    if len(accepted_shift_rows) < min_inliers:
        raise ValueError(
            f"{reference_tile.tile} produced {len(accepted_shift_rows)} accepted patch shifts; require {min_inliers}"
        )
    if final_shift_px is None or inlier_mask is None:
        total_shifts = np.vstack(accepted_shift_rows).astype(np.float64)
        inlier_mask, final_shift_px = select_inlier_patch_measurements(total_shifts, min_inliers=min_inliers)

    return final_shift_px, {
        "mode": "l0_patch_phase",
        "patch_shape_zyx": [int(value) for value in patch_shape_zyx],
        "coarse_level": int(coarse_level),
        "coarse_scale_zyx": [float(value) for value in fixed_coarse_scale_zyx],
        "coarse_shift_level_px_zyx": [float(value) for value in coarse_shift_coarse_px],
        "coarse_shift_px_zyx": [float(value) for value in coarse_shift_l0_px],
        "coarse_corr_before": coarse_details["corr_before"],
        "coarse_corr_after": coarse_details["corr_after"],
        "n_candidates": len(candidates),
        "n_measured": len(accepted_shift_rows),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "early_stop_after_patch": early_stop_after_patch,
        "patches": patch_rows,
    }
```

The actual implementation also records skipped candidates after early stop and marks accepted patch rows as inliers/outliers before returning. Those bookkeeping lines were omitted here only to keep the review packet shorter.

### Incremental Cache

```python
def tile_phase_cache_key(
    *,
    reference_position: Path,
    reference_channel: int,
    reference_token: str,
    moving_token: str,
    level: int,
    upsample_factor: int,
    patch_shape_zyx: tuple[int, int, int] | None,
    min_inliers: int,
    max_candidate_patches: int,
    coarse_level: int,
) -> dict[str, Any]:
    return {
        "cache_version": "tile_phase_patch_inbounds_v1",
        "reference_position": str(reference_position.resolve()),
        "reference_channel": int(reference_channel),
        "reference_token": reference_token,
        "moving_token": moving_token,
        "level": int(level),
        "upsample_factor": int(upsample_factor),
        "patch_shape_zyx": None if patch_shape_zyx is None else [int(value) for value in patch_shape_zyx],
        "min_inliers": int(min_inliers),
        "max_candidate_patches": int(max_candidate_patches),
        "coarse_level": int(coarse_level),
    }


def write_tile_phase_cache(cache_path: Path, cache_key: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "lightsheet.tile_phase_measurement_cache.v1",
        "cache_key": cache_key,
        "measurements": rows,
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(cache_path)
```

### Position Update and Registration Adaptation

```python
def apply_shift_row_to_position_record(record: dict[str, Any], row: dict[str, Any], *, moving_path: Path) -> np.ndarray:
    record["tile"] = row["tile"]
    record["path"] = str(moving_path)
    shift_um = np.asarray(row["shift_um_zyx"], dtype=np.float64)
    for axis, value in zip(DIMENSIONS, shift_um, strict=True):
        record["translation_um"][axis] = float(record["translation_um"][axis] + value)
    return shift_um


def adapt_registration_from_reference(
    *,
    reference_registration_input: Path,
    output_registration: Path,
    adapted_position_payload: dict[str, Any],
    reference_token: str,
    moving_token: str,
    adapted_to_position: Path,
    tile_phase_summary: dict[str, Any],
) -> Path:
    reference_registration = json.loads(reference_registration_input.read_text())
    adapted = json.loads(json.dumps(reference_registration))
    position_by_tile = position_records_by_tile(adapted_position_payload)
    adapted_tiles = []
    missing = []
    for record in adapted["tiles"]:
        moving_tile = make_moving_tile_name(record["tile"], reference_token=reference_token, moving_token=moving_token)
        position_record = position_by_tile.get(moving_tile)
        if position_record is None:
            missing.append(moving_tile)
            continue
        adapted_record = json.loads(json.dumps(record))
        adapted_record["tile"] = moving_tile
        adapted_record["stage_translation_um"] = position_record["translation_um"]
        adapted_record["stage_scale_um"] = position_record["scale_um"]
        if "path" in adapted_record or position_record.get("path") is not None:
            adapted_record["path"] = position_record["path"]
        adapted_tiles.append(adapted_record)
    if missing:
        raise ValueError(f"Adapted position file is missing tiles required by registration: {missing}")
    adapted["tiles"] = adapted_tiles
    if adapted_tiles:
        first_path = Path(position_by_tile[adapted_tiles[0]["tile"]]["path"])
        adapted["input_dir"] = str(first_path.parent)
    adapted["adapted_from"] = str(reference_registration_input.resolve())
    adapted["adapted_to_position"] = str(adapted_to_position.resolve())
    adapted["adaptation_method"] = "copy_registered_affine_from_reference_replace_405_stage_from_tile_phase"
    adapted["tile_phase_summary"] = {
        "output_position": tile_phase_summary["output_position"],
        "summary_path": tile_phase_summary.get("summary_path"),
        "tile_count": len(tile_phase_summary["measurements"]),
        "measurements": [
            {
                "tile": item["tile"],
                "reference_tile": item["reference_tile"],
                "final_shift_px_zyx": item["shift_px_zyx"],
                "final_shift_um_zyx": item["shift_um_zyx"],
                "n_inliers": item.get("n_inliers"),
            }
            for item in tile_phase_summary["measurements"]
        ],
    }
    output_registration.parent.mkdir(parents=True, exist_ok=True)
    output_registration.write_text(json.dumps(adapted, indent=2) + "\n")
    return output_registration.resolve()
```

### Top-Level Tile-Phase Command Interface

```python
@app.command("tile-phase-align")
def tile_phase_align(
    reference_position: Annotated[Path, typer.Option("--reference-position", exists=True, dir_okay=False, readable=True)],
    output_position: Annotated[Path, typer.Option("--output-position")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    reference_channel: Annotated[int, typer.Option("--reference-channel", min=0)] = 3,
    reference_token: Annotated[str, typer.Option("--reference-token")] = "488514561638",
    moving_token: Annotated[str, typer.Option("--moving-token")] = "405",
    level: Annotated[int, typer.Option("--level", min=0)] = DEFAULT_LEVEL,
    upsample_factor: Annotated[int, typer.Option("--upsample-factor", min=1)] = 10,
    patch_shape_zyx: Annotated[str | None, typer.Option("--patch-shape-zyx")] = None,
    min_inliers: Annotated[int, typer.Option("--min-inliers", min=1)] = 2,
    max_candidate_patches: Annotated[int, typer.Option("--max-candidate-patches", min=1)] = 24,
    coarse_level: Annotated[int, typer.Option("--coarse-level", min=0)] = DEFAULT_LEVEL,
    reference_registration_input: Annotated[
        Path | None,
        typer.Option("--reference-registration-input", exists=True, dir_okay=False, readable=True),
    ] = None,
    output_registration: Annotated[Path | None, typer.Option("--output-registration")] = None,
) -> None:
    path = align_tiles_to_reference(
        reference_position=reference_position,
        output_position=output_position,
        output_dir=output_dir,
        reference_channel=reference_channel,
        reference_token=reference_token,
        moving_token=moving_token,
        level=level,
        upsample_factor=upsample_factor,
        patch_shape_zyx=None if patch_shape_zyx is None else parse_shape_zyx(patch_shape_zyx),
        min_inliers=min_inliers,
        max_candidate_patches=max_candidate_patches,
        coarse_level=coarse_level,
        reference_registration_input=reference_registration_input,
        output_registration=output_registration,
    )
    typer.echo(path)
```

### Adjacent-Tile Fallback Concept

If direct same-index 405-to-488 patch phase fails, the implementation can infer the failed 405 tile's shift by phase-correlating it against already-successful neighboring 405 tiles in physical overlap coordinates, then composing that neighbor's known 405-to-488 shift with the measured 405-vs-405 overlap residual. This fallback is explicitly marked with `"fallback": true` in the summary. It was not used in the completed run summarized above.

The fallback acceptance currently uses `min_inliers=1` at the call site:

```python
shift_px, details = infer_shift_from_adjacent_tiles(
    failed_tile=moving_tile,
    successful_tiles=successful_moving_tiles,
    patch_shape_zyx=patch_shape_zyx,
    min_inliers=1,
)
```

Please review whether that is too permissive, especially for low-content tiles.

## Fusion Weighting Under Review

Full fusion calls `multiview_stitcher.fusion.fuse`. Its `weighted_average_fusion` multiplies normalized geometric `blending_weights` by optional `fusion_weights`, then renormalizes. The custom content-preibisch-coarse function below supplies `fusion_weights`.

The recent change is `softmax_exponent`. With exponent `2.0`, normalized content weights are sharpened as:

`w_i = w_i^2 / sum_j(w_j^2)`

`1.0` disables sharpening. Border/geometric blending is still applied afterward by multiview-stitcher.

```python
def normalize_or_uniform_valid(weights, valid, xp):
    totals = xp.sum(weights, axis=0, keepdims=True)
    zero_quality = (totals <= 0.0) & xp.any(valid, axis=0, keepdims=True)
    weights = xp.where(zero_quality & valid, 1.0, weights)
    totals = xp.sum(weights, axis=0, keepdims=True)
    return xp.where(totals > 0.0, weights / totals, 0.0)


def coarse_preibisch_content_weights(
    transformed_views,
    blending_weights,
    *,
    sigma_1: int,
    sigma_2: int,
    stride_zyx: tuple[int, int, int],
    softmax_exponent: float = 2.0,
):
    try:
        import cupy as cp
    except ImportError:  # pragma: no cover - depends on runtime environment
        cp = None

    if cp is not None and isinstance(transformed_views, cp.ndarray):
        import cupyx.scipy.ndimage as cpx_ndimage

        xp = cp
        gaussian_filter = cpx_ndimage.gaussian_filter
    else:
        import scipy.ndimage as scipy_ndimage

        xp = np
        gaussian_filter = scipy_ndimage.gaussian_filter

    spatial_ndim = transformed_views.ndim - 1
    strides = tuple(int(value) for value in stride_zyx[-spatial_ndim:])
    if any(value <= 0 for value in strides):
        raise ValueError("content Preibisch coarse strides must be positive")

    ds_slices = (slice(None),) + tuple(slice(None, None, stride) for stride in strides)
    views_ds = transformed_views[ds_slices].astype(xp.float32, copy=False)
    blending_ds = blending_weights[ds_slices]
    valid_ds = (blending_ds > 1e-7) & ~xp.isnan(views_ds)
    filled = xp.where(valid_ds, views_ds, 0.0)

    sigma1_ds = (0.0,) + tuple(max(0.5, float(sigma_1) / stride) for stride in strides)
    sigma2_ds = (0.0,) + tuple(max(0.5, float(sigma_2) / stride) for stride in strides)
    low_pass = gaussian_filter(filled, sigma=sigma1_ds, mode="reflect")
    quality = gaussian_filter(xp.where(valid_ds, (filled - low_pass) ** 2, 0.0), sigma=sigma2_ds, mode="reflect")
    quality = xp.where(valid_ds, quality, 0.0)
    quality = normalize_or_uniform_valid(quality, valid_ds, xp)

    weights = quality
    for axis, stride in enumerate(strides, start=1):
        if stride > 1:
            weights = xp.repeat(weights, stride, axis=axis)
    crop = (slice(None),) + tuple(slice(0, size) for size in transformed_views.shape[1:])
    weights = weights[crop]

    valid_full = (blending_weights > 1e-7) & ~xp.isnan(transformed_views)
    weights = xp.where(valid_full, weights, 0.0)
    weights = normalize_or_uniform_valid(weights, valid_full, xp)
    if softmax_exponent > 1.0:
        eps = xp.asarray(1e-12, dtype=weights.dtype)
        weights = xp.where(valid_full, xp.power(xp.maximum(weights, eps), float(softmax_exponent)), 0.0)
        weights = normalize_or_uniform_valid(weights, valid_full, xp)
    return weights.astype(xp.float32, copy=False)
```

Wrapper default:

```python
def fuse_tiles(
    *,
    input_dir: Path,
    position_input: Path,
    registration_input: Path,
    output: Path,
    channels: list[int] | None = None,
    fusion_weight_mode: str = "content-preibisch-coarse",
    content_preibisch_sigma1: int = 7,
    content_preibisch_sigma2: int = 17,
    content_preibisch_coarse_stride: tuple[int, int, int] = (1, 8, 8),
    content_preibisch_softmax_exponent: float = 2.0,
    basic_cache_tiles: int = 128,
    dry_run: bool = False,
) -> str:
    args = [
        str(input_dir),
        "--position-input", str(position_input),
        "--registration-input", str(registration_input),
        "--output", str(output),
        "--fusion-weight-mode", fusion_weight_mode,
        "--content-preibisch-sigma1", str(content_preibisch_sigma1),
        "--content-preibisch-sigma2", str(content_preibisch_sigma2),
        "--content-preibisch-coarse-stride", *(str(value) for value in content_preibisch_coarse_stride),
        "--content-preibisch-softmax-exponent", str(content_preibisch_softmax_exponent),
        "--basic-cache-tiles", str(basic_cache_tiles),
    ]
```

## Tests Currently Covering This Path

Representative tests:

```python
def test_patch_shift_uses_gpu_phase_helper_for_synthetic_translation(monkeypatch) -> None:
    from skimage.registration import phase_cross_correlation

    z, y, x = np.mgrid[:32, :96, :96]
    fixed = np.exp(-(((z - 16) ** 2) / 14.0 + ((y - 48) ** 2 + (x - 48) ** 2) / 220.0)).astype(np.float32)
    moving = np.roll(fixed, shift=3, axis=0)
    moving = np.roll(moving, shift=-7, axis=1)
    moving = np.roll(moving, shift=9, axis=2)

    def fake_gpu_phase(fixed_norm: np.ndarray, moving_norm: np.ndarray) -> tuple[tuple[float, float, float], float]:
        shift, _error, _phase = phase_cross_correlation(fixed_norm, moving_norm, upsample_factor=1)
        return tuple(float(value) for value in shift), 1.0

    monkeypatch.setattr(tile_phase_module.stitch_legacy, "phase_correlation_shift_gpu", fake_gpu_phase)

    shift, details = estimate_patch_shift_zyx_px(fixed, moving)

    np.testing.assert_allclose(shift, [-3, 7, -9], atol=0.25)
    assert details["corr_after"] > details["corr_before"]
```

```python
def test_patch_mode_composes_coarse_seed_and_residual(monkeypatch) -> None:
    # coarse shift at scale 4 is [-1, 2, -3]
    # residual shift is [1, -2, 3]
    # expected total level-0 shift is [-3, 6, -9]
    shift, details = measure_patch_tile_shift(
        reference_tile=reference,
        moving_tile=moving,
        reference_channel=3,
        patch_shape_zyx=(16, 32, 32),
        coarse_level=2,
        upsample_factor=10,
        max_candidate_patches=2,
        min_inliers=2,
    )

    np.testing.assert_allclose(shift, [-3.0, 6.0, -9.0])
    assert details["n_inliers"] == 2
    assert details["n_measured"] == 2
    assert details["early_stop_after_patch"] == 1
```

```python
def test_inlier_selection_returns_cluster_median_and_rejects_outlier() -> None:
    shifts = np.asarray(
        [
            [10.0, -20.0, 4.0],
            [11.0, -18.0, 6.0],
            [9.5, -21.0, 5.0],
            [40.0, 50.0, -90.0],
        ]
    )

    inliers, median = select_inlier_patch_measurements(shifts, min_inliers=2)

    assert inliers.tolist() == [True, True, True, False]
    np.testing.assert_allclose(median, [10.0, -20.0, 5.0])
```

```python
def test_candidate_patch_slices_filters_shifted_moving_out_of_bounds() -> None:
    candidates = candidate_patch_slices(
        scout,
        tile_shape_zyx=np.asarray([64, 128, 128]),
        patch_shape_zyx=(32, 64, 64),
        scout_scale_zyx=np.asarray([16.0, 16.0, 16.0]),
        max_candidates=24,
        moving_shape_zyx=np.asarray([64, 128, 128]),
        shift_zyx_px=np.asarray([-32.0, -64.0, -64.0]),
    )

    assert candidates
    for candidate in candidates:
        moving_slices = tile_phase_module.shifted_slices_zyx(
            candidate["fixed_slices"],
            shift_zyx_px=np.asarray([-32.0, -64.0, -64.0]),
        )
        assert tile_phase_module.slices_within_shape(moving_slices, np.asarray([64, 128, 128]))
```

```python
def test_adapt_registration_copies_affine_and_replaces_405_stage(tmp_path) -> None:
    output = json.loads(output_registration.read_text())
    assert output["input_dir"] == str(tmp_path / "230Tnc-CL-405")
    assert output["adapted_from"] == str(reference_registration.resolve())
    assert output["adapted_to_position"] == str((tmp_path / "positions.405.json").resolve())
    assert output["adaptation_method"] == "copy_registered_affine_from_reference_replace_405_stage_from_tile_phase"
    assert output["tiles"][0]["tile"] == "230Tnc-CL-405.000.ome.tif"
    assert output["tiles"][0]["path"] == str(tmp_path / "230Tnc-CL-405" / "230Tnc-CL-405.000.ome.tif")
    assert output["tiles"][0]["stage_translation_um"] == {"z": 10.0, "y": 20.0, "x": 30.0}
    assert output["tiles"][0]["stage_scale_um"] == {"z": 1.0, "y": 2.0, "x": 3.0}
    assert output["tiles"][0]["registered_affine"] == affine
```

```python
def test_content_preibisch_coarse_softmax_sharpens_any_overlap(monkeypatch) -> None:
    raw_weights = np.array([[0.25, 0.5, 1.0], [0.75, 0.5, np.nan]], dtype=np.float32)
    weights = coarse_preibisch_content_weights(
        np.sqrt(raw_weights),
        ~np.isnan(raw_weights),
        sigma_1=7,
        sigma_2=17,
        stride_zyx=(1,),
        softmax_exponent=2.0,
    )

    np.testing.assert_allclose(weights[:, 0], [0.1, 0.9], atol=1e-6)
    np.testing.assert_allclose(weights[:, 1], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(weights[:, 2], [1.0, 0.0], atol=1e-6)
```

## Checks Already Run

Tile-phase path:

```bash
CONDA_NO_PLUGINS=true PYTHONPYCACHEPREFIX=/tmp/lightsheet-pycache \
conda run -n multi python -m py_compile \
  src/squisher_lightsheet/tile_phase.py \
  src/squisher_lightsheet/cli.py \
  tests/test_tile_phase.py

CONDA_NO_PLUGINS=true RUFF_CACHE_DIR=/tmp/ruff-cache \
conda run -n multi ruff check \
  src/squisher_lightsheet/tile_phase.py \
  src/squisher_lightsheet/cli.py \
  tests/test_tile_phase.py

CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
conda run -n multi pytest -p no:cacheprovider tests/test_tile_phase.py
```

Fusion softmax path:

```bash
CONDA_NO_PLUGINS=true PYTHONPYCACHEPREFIX=/tmp/lightsheet-pycache \
conda run -n multi python -m py_compile \
  src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py \
  src/squisher_lightsheet/fusion.py \
  src/squisher_lightsheet/cli.py \
  tests/test_fusion.py

CONDA_NO_PLUGINS=true RUFF_CACHE_DIR=/tmp/ruff-cache \
conda run -n multi ruff check \
  src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py \
  src/squisher_lightsheet/fusion.py \
  src/squisher_lightsheet/cli.py \
  tests/test_fusion.py

CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
conda run -n multi pytest -p no:cacheprovider tests/test_fusion.py
```

Current focused test results:

- `tests/test_tile_phase.py`: 8 passed
- `tests/test_fusion.py`: 5 passed

## Known Limitations and Concerns

- One tile has very low post-shift correlation despite two inliers: `230Tnc-CL-405.005.ome.tif`.
- Patch acceptance currently relies primarily on shift-cluster consistency, not a hard minimum `peak`, `corr_after`, or `corr_after - corr_before`.
- Early stopping can stop after only two consistent high-ranked patches.
- Cache invalidation does not include source tile file mtimes, source code version, or GPU phase-correlation implementation version.
- The adjacent fallback uses `min_inliers=1` from the top-level failure recovery call.
- The dumb stitch preview averages overlaps and is not expected to match final fusion behavior.
- The softmax-like fusion exponent has unit-test coverage but has not yet been validated by running a full fusion output and comparing seam/ghosting metrics.

## Files Most Relevant for Review

- `/home/chaichontat/squisher/lightsheet/src/squisher_lightsheet/tile_phase.py`
- `/home/chaichontat/squisher/lightsheet/src/squisher_lightsheet/cli.py`
- `/home/chaichontat/squisher/lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py`
- `/home/chaichontat/squisher/lightsheet/src/squisher_lightsheet/fusion.py`
- `/home/chaichontat/squisher/lightsheet/tests/test_tile_phase.py`
- `/home/chaichontat/squisher/lightsheet/tests/test_fusion.py`

