# Reference-Prior Tile Registration Review Packet

## Problem

We are stitching large lightsheet OME-TIFF tile mosaics across acquisition tracks. For the current sample:

- Dataset root: `/home/chaichontat/nvme/lightsheet/20260613`
- Working output root: `/working/230Tnc`
- Channels/tracks of interest: `488`, `514`, `561`, `638`
- Track `488` is the reference geometry after its own registration/global optimization.
- Tracks `561` and `638` are acquired simultaneously and should use identical tile geometry.
- Other tracks can have small local z differences because z motion/stage behavior differs between imaging rounds.

We need to register non-reference tracks to the 488 tile geometry while minimizing lateral xy drift from the 488 registration. The desired physical behavior is:

- Use 488's optimized tile y/x positions as the preferred geometry.
- Allow z corrections because local z offsets may differ between imaging tracks.
- For `561` and `638`, solve one shared geometry so both channels remain exactly aligned.
- Do not blindly force full xyz equality to 488, because z can differ.
- Do not let noisy phase correlations in dissimilar channels produce large xy deviations unless strongly supported.

The current candidate solution is a reference-prior global optimization:

- `fixed-xy`: hard pins y/x to 488 and solves z only.
- `full-xyz`: starts from 488 geometry but freely solves z/y/x.
- `penalized-xy`: starts from 488 geometry, solves z/y/x, but adds a soft pseudo-observation prior on each tile's y/x correction toward zero. For this mode, residual rejection is based only on z so the soft y/x prior is not contradicted by hard y/x residual filtering.

## Constraints

- Tile positions are represented as affine translations in physical microns.
- Phase correlation estimates are represented as per-edge patch shifts in pixels, `shift_zyx`.
- Solver corrections are in pixels and converted to microns by tile spacing before being added to affine translations.
- The robust-boundary registration operates on many sampled overlap patches between neighboring tiles.
- Only accepted boundary constraints enter the final global solve.
- For shared `561`/`638` geometry, constraints from both tracks are pooled and one set of tile corrections is solved.
- The current implementation uses bounded weighted least squares with Huber-style IRLS.
- The anchor tile has correction zero; all other corrections are relative to that anchor.
- Max correction clamp defaults are `(16, 96, 96)` px for z/y/x.
- Final residual rejection thresholds default to `(4, 8, 8)` px for z/y/x.
- For `penalized-xy`, final residual rejection is z-only.

## Current Cross-Validation Summary

The cross-validation sweep used 931 accepted boundary constraints from the corrected penalized run. It split constraints into 5 folds, stratified by `(track, pair, axis)`, solved on training constraints, and scored held-out residuals plus drift from 488.

Selected results:

| mode | prior weight | held-out weighted RMS um | held-out xy p95 um | xy drift p95 from 488 um |
| --- | ---: | ---: | ---: | ---: |
| fixed-xy | n/a | 4.669 | 7.123 | 0.000 |
| full-xyz | 0 | 3.753 | 6.789 | 5.368 |
| penalized-xy | 3e-5 | 3.741 | 6.767 | 5.221 |
| penalized-xy | 0.001 | 3.827 | 6.313 | 4.040 |
| penalized-xy | 0.01 | 3.860 | 6.132 | 1.806 |
| penalized-xy | 0.1 | 4.478 | 6.708 | 0.343 |

Interpretation so far:

- The best held-out RMS is near `3e-5`, but that permits about 5.2 um xy p95 drift from 488.
- `0.01` seems like a practical conservative value: held-out RMS is close to the low-prior optimum, while xy drift p95 is about 1.8 um.
- `fixed-xy` has zero xy drift but worse held-out RMS.

## Review Questions

Please evaluate:

1. Is the current least-squares formulation statistically sound for this physical problem?
2. Is the pseudo-observation prior on per-tile y/x corrections the right way to preserve 488 geometry?
3. Is z-only residual rejection in `penalized-xy` the right choice, or should y/x residuals be handled differently?
4. Is the prior-weight scale meaningful and transferable across datasets, given constraint weights and patch sampling?
5. Should the prior be applied to absolute per-tile corrections, pairwise edge corrections, a smooth deformation field, or some hybrid?
6. Does anchoring one tile at zero introduce an avoidable bias in the prior or drift metrics?
7. Is the cross-validation design unbiased enough, or is using constraints accepted by a previous penalized run circular?
8. Should simultaneous tracks such as `561` and `638` share one geometry exactly, as implemented here?

## Core Data Structures

```python
@dataclass(frozen=True)
class RobustBoundarySettings:
    patch_shape_zyx: tuple[int, int, int] = (64, 512, 512)
    max_patches_per_edge: int = 12
    min_nonzero_fraction: float = 0.05
    min_std: float = 1e-3
    content_mask_percentile: float = 60.0
    min_content_fraction: float = 0.002
    min_content_voxels: int = 4096
    min_correlation: float = 0.60
    min_improvement: float = 0.02
    min_stable_correlation: float = 0.75
    max_stable_shift_zyx: tuple[float, float, float] = (1.0, 2.0, 2.0)
    max_correction_zyx: tuple[float, float, float] = (16.0, 96.0, 96.0)
    max_final_residual_zyx: tuple[float, float, float] = (4.0, 8.0, 8.0)
    huber_delta: float = 4.0
    irls_iterations: int = 5
    reference_xy_prior_weight: float = 0.01


@dataclass(frozen=True)
class BoundaryConstraint:
    fixed: int
    moving: int
    pair: tuple[int, int]
    axis: str
    patch_index: int
    shift_zyx: tuple[float, float, float]
    weight: float
    correlation_before: float
    correlation_after: float
    improvement: float
    fixed_nonzero_fraction: float
    moving_nonzero_fraction: float
    fixed_std: float
    moving_std: float
    accepted: bool
    fixed_content_fraction: float = 0.0
    moving_content_fraction: float = 0.0
    reject_reason: str | None = None
    final_residual_zyx: tuple[float, float, float] | None = None
    fixed_slices: tuple[slice, slice, slice] | None = None
    moving_slices: tuple[slice, slice, slice] | None = None
    source_label: str | None = None


@dataclass(frozen=True)
class ReferenceGeometryConstraint:
    mode: str
    reference_input: str
    fixed_axes: tuple[str, ...]
    shared_geometry_tracks: tuple[str, ...] = ()
    drift_from_reference_um: dict[str, Any] | None = None
    constraint_counts_by_track: dict[str, Any] | None = None
    reference_prior_weights_zyx: tuple[float, float, float] | None = None
    residual_reject_axes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ReferenceGeometrySolverOptions:
    fixed_axes: set[str]
    reference_prior_weights_zyx: tuple[float, float, float] | None
    residual_reject_axes: set[str] | None
```

## Solver

This is the global correction solve. For every accepted boundary constraint and each non-fixed dimension, the equation is:

`correction[moving] - correction[fixed] ~= measured_shift`

For `penalized-xy`, additional pseudo-observation rows are added:

`correction[tile] ~= 0`

for y and x only, with weight `reference_xy_prior_weight`.

```python
def solve_tile_corrections_zyx(
    n_tiles: int,
    constraints: list[BoundaryConstraint],
    settings: RobustBoundarySettings,
    anchor_tile: int,
    *,
    fixed_axes: set[str] | None = None,
    reference_prior_weights_zyx: tuple[float, float, float] | None = None,
) -> list[tuple[float, float, float]]:
    import numpy as np
    from scipy.optimize import lsq_linear

    fixed_dim_indices = {{"z": 0, "y": 1, "x": 2}[axis] for axis in (fixed_axes or set())}
    connected_tiles = anchor_connected_tiles(n_tiles, constraints, anchor_tile)
    accepted = [
        constraint
        for constraint in constraints
        if constraint.accepted
        and constraint.fixed in connected_tiles
        and constraint.moving in connected_tiles
    ]
    corrections = np.zeros((n_tiles, 3), dtype=float)
    if not accepted:
        return [tuple(float(value) for value in row) for row in corrections]

    variable_tiles = [index for index in sorted(connected_tiles) if index != anchor_tile]
    if not variable_tiles:
        return [tuple(float(value) for value in row) for row in corrections]
    variable_index = {tile: index for index, tile in enumerate(variable_tiles)}
    clamp = np.asarray(settings.max_correction_zyx, dtype=float)
    prior_weights = np.asarray(reference_prior_weights_zyx or (0.0, 0.0, 0.0), dtype=float)

    for dim in range(3):
        if dim in fixed_dim_indices:
            continue
        rows = []
        targets = []
        weights = []
        is_prior = []
        for constraint in accepted:
            row = np.zeros(len(variable_tiles), dtype=float)
            if constraint.moving != anchor_tile:
                row[variable_index[constraint.moving]] += 1.0
            if constraint.fixed != anchor_tile:
                row[variable_index[constraint.fixed]] -= 1.0
            rows.append(row)
            targets.append(constraint.shift_zyx[dim])
            weights.append(max(constraint.weight, 1e-6))
            is_prior.append(False)
        if prior_weights[dim] > 0.0:
            for tile in variable_tiles:
                row = np.zeros(len(variable_tiles), dtype=float)
                row[variable_index[tile]] = 1.0
                rows.append(row)
                targets.append(0.0)
                weights.append(float(prior_weights[dim]))
                is_prior.append(True)

        a = np.vstack(rows)
        b = np.asarray(targets, dtype=float)
        base_w = np.asarray(weights, dtype=float)
        prior_mask = np.asarray(is_prior, dtype=bool)
        w = base_w.copy()
        solution = np.zeros(len(variable_tiles), dtype=float)
        for _ in range(settings.irls_iterations):
            sqrt_w = np.sqrt(w)
            aw = a * sqrt_w[:, None]
            bw = b * sqrt_w
            result = lsq_linear(
                aw,
                bw,
                bounds=(-clamp[dim], clamp[dim]),
                lsmr_tol="auto",
            )
            solution = np.asarray(result.x, dtype=float)
            residual = a @ solution - b
            scale = np.maximum(1.0, np.abs(residual) / settings.huber_delta)
            scale[prior_mask] = 1.0
            w = base_w / scale
        for tile, index in variable_index.items():
            corrections[tile, dim] = solution[index]

    corrections[anchor_tile, :] = 0.0
    return [tuple(float(value) for value in row) for row in corrections]
```

## Residual Rejection

The implementation iterates solve -> annotate final residuals -> reject high residual constraints -> re-anchor if needed, until accepted/rejected constraints and anchor stabilize.

For `penalized-xy`, `residual_reject_axes={"z"}`. This is intentional: y/x can remain inconsistent with individual measured shifts because the soft prior pulls y/x corrections toward the 488 geometry.

```python
def annotate_final_residuals(
    constraints: list[BoundaryConstraint],
    corrections_zyx: list[tuple[float, float, float]],
    settings: RobustBoundarySettings,
    connected_tiles: set[int] | None = None,
    *,
    reject_axes: set[str] | None = None,
) -> list[BoundaryConstraint]:
    max_residual = settings.max_final_residual_zyx
    reject_dim_indices = {{"z": 0, "y": 1, "x": 2}[axis] for axis in (reject_axes or {"z", "y", "x"})}
    updated = []
    for constraint in constraints:
        residual = tuple(
            corrections_zyx[constraint.moving][index]
            - corrections_zyx[constraint.fixed][index]
            - constraint.shift_zyx[index]
            for index in range(3)
        )
        disconnected = (
            connected_tiles is not None
            and constraint.accepted
            and (
                constraint.fixed not in connected_tiles
                or constraint.moving not in connected_tiles
            )
        )
        reject = disconnected or (
            constraint.accepted
            and any(abs(residual[index]) > max_residual[index] for index in reject_dim_indices)
        )
        if disconnected:
            reject_reason = "disconnected_from_anchor"
        elif reject:
            reject_reason = "high_final_residual"
        else:
            reject_reason = constraint.reject_reason
        updated.append(
            replace(
                constraint,
                accepted=constraint.accepted and not reject,
                reject_reason=reject_reason,
                final_residual_zyx=residual,
                weight=0.0 if reject else constraint.weight,
            )
        )
    return updated


def solve_tile_corrections_with_residual_rejection(
    tiles: list[TileMetadata],
    constraints: list[BoundaryConstraint],
    settings: RobustBoundarySettings,
    *,
    fixed_axes: set[str] | None = None,
    reference_prior_weights_zyx: tuple[float, float, float] | None = None,
    residual_reject_axes: set[str] | None = None,
) -> tuple[list[tuple[float, float, float]], list[BoundaryConstraint], int]:
    n_tiles = len(tiles)
    current = constraints
    anchor_tile = choose_anchor_tile(tiles, current)
    for _ in range(len(constraints) + n_tiles + 1):
        connected_tiles = anchor_connected_tiles(n_tiles, current, anchor_tile)
        corrections_zyx = solve_tile_corrections_zyx(
            n_tiles,
            current,
            settings,
            anchor_tile,
            fixed_axes=fixed_axes,
            reference_prior_weights_zyx=reference_prior_weights_zyx,
        )
        updated = annotate_final_residuals(
            current,
            corrections_zyx,
            settings,
            connected_tiles,
            reject_axes=residual_reject_axes
            if residual_reject_axes is not None
            else {"z", "y", "x"} - (fixed_axes or set()),
        )
        next_anchor_tile = choose_anchor_tile(tiles, updated)
        if all(
            before.accepted == after.accepted
            for before, after in zip(current, updated, strict=True)
        ) and next_anchor_tile == anchor_tile:
            return corrections_zyx, updated, anchor_tile
        current = updated
        anchor_tile = next_anchor_tile

    raise RuntimeError("Robust boundary residual rejection did not converge")
```

## Applying Corrections and Reference Constraints

Corrections are solved in pixels, then converted to microns with tile spacing and added to the affine translations. In `fixed-xy`, y/x are overwritten from the reference after applying corrections.

```python
def apply_corrections_to_params(
    params: list[Any],
    corrections_zyx: list[tuple[float, float, float]],
    spacing: dict[str, float],
) -> list[Any]:
    corrected = []
    for param, correction in zip(params, corrections_zyx, strict=True):
        updated = param.copy(deep=True)
        for index, dim in enumerate(("z", "y", "x")):
            updated.data[(0,) * (updated.data.ndim - 2) + (index, 3)] += correction[index] * spacing[dim]
        corrected.append(updated)
    return corrected


def apply_reference_fixed_axes(
    params: list[Any],
    reference_params: list[Any],
    fixed_axes: set[str],
) -> list[Any]:
    axis_to_index = {"z": 0, "y": 1, "x": 2}
    fixed_indices = {axis_to_index[axis] for axis in fixed_axes}
    constrained = []
    for param, reference in zip(params, reference_params, strict=True):
        translation = list(affine_translation_zyx(param))
        reference_translation = affine_translation_zyx(reference)
        for index in fixed_indices:
            translation[index] = reference_translation[index]
        constrained.append(set_affine_translation_um(param, tuple(translation)))
    return constrained
```

## Reference Geometry Mode Mapping

This helper is the single source of truth for how CLI mode maps to solver behavior.

```python
def reference_geometry_solver_options(
    mode: str,
    reference_xy_prior_weight: float,
) -> ReferenceGeometrySolverOptions:
    if reference_xy_prior_weight < 0.0:
        raise ValueError("--reference-xy-prior-weight must be non-negative")
    if mode not in {"fixed-xy", "full-xyz", "penalized-xy"}:
        raise ValueError(f"Unsupported reference geometry mode: {mode}")
    if mode == "fixed-xy":
        return ReferenceGeometrySolverOptions(
            fixed_axes={"y", "x"},
            reference_prior_weights_zyx=None,
            residual_reject_axes=None,
        )
    if mode == "penalized-xy":
        return ReferenceGeometrySolverOptions(
            fixed_axes=set(),
            reference_prior_weights_zyx=(0.0, float(reference_xy_prior_weight), float(reference_xy_prior_weight)),
            residual_reject_axes={"z"},
        )
    return ReferenceGeometrySolverOptions(
        fixed_axes=set(),
        reference_prior_weights_zyx=None,
        residual_reject_axes=None,
    )
```

## Single-Track Reference Geometry Run Path

The single-track robust-boundary refinement starts from reference geometry for `full-xyz` and `penalized-xy`; for `fixed-xy`, it can start from loaded params and then copy y/x from the reference after correction.

```python
if args.reference_geometry_mode != "none":
    if reference_registration_input is None:
        raise ValueError("--reference-geometry-mode requires --reference-registration-input")
    reference_options = reference_geometry_solver_options(
        args.reference_geometry_mode,
        args.reference_xy_prior_weight,
    )
    log(f"Loading reference registration params from {reference_registration_input.resolve()}")
    reference_params = load_registration_params(reference_registration_input.resolve(), tiles)
    fixed_reference_axes = reference_options.fixed_axes
    reference_prior_weights_zyx = reference_options.reference_prior_weights_zyx
    residual_reject_axes = reference_options.residual_reject_axes
    if reference_prior_weights_zyx is not None:
        log(f"Reference xy prior weights z/y/x: {reference_prior_weights_zyx}")
    log(f"Loaded {len(reference_params)} reference registration transforms")
    log_transform_translation_summary("Reference registration", reference_params)

if should_refine_registration_input(args, registration_input):
    if params is None:
        raise RuntimeError("Loaded registration params were not initialized")
    refinement_start_params = (
        reference_params
        if reference_params is not None and args.reference_geometry_mode in {"full-xyz", "penalized-xy"}
        else params
    )
    if refinement_start_params is reference_params:
        log(f"Robust boundary refinement starts from reference geometry mode={args.reference_geometry_mode}")
    with heartbeat("robust boundary refinement"):
        robust_refinement = refine_registration_with_robust_boundaries(
            tiles,
            refinement_start_params,
            channel=reg_source_channel,
            output_dir=robust_boundary_qc_dir,
            settings=RobustBoundarySettings(),
            reference_params=reference_params,
            reference_input=reference_registration_input,
            fixed_reference_axes=fixed_reference_axes,
            reference_prior_weights_zyx=reference_prior_weights_zyx,
            residual_reject_axes=residual_reject_axes,
            reference_geometry_mode=args.reference_geometry_mode,
            source_label=source_label,
        )
```

## Robust Boundary Refinement Function

```python
def refine_registration_with_robust_boundaries(
    tiles: list[TileMetadata],
    params: list[Any],
    *,
    channel: int,
    output_dir: Path,
    settings: RobustBoundarySettings,
    reference_params: list[Any] | None = None,
    reference_input: Path | None = None,
    fixed_reference_axes: set[str] | None = None,
    reference_prior_weights_zyx: tuple[float, float, float] | None = None,
    residual_reject_axes: set[str] | None = None,
    reference_geometry_mode: str | None = None,
    source_label: str | None = None,
    shared_geometry_tracks: tuple[str, ...] = (),
) -> RobustBoundaryRefinementResult:
    require_cuda_for_robust_boundary()
    pairs = axis_aligned_registration_pairs(tiles)
    patch_specs = sample_boundary_patches(tiles, params, pairs, settings)
    log(
        "Robust boundary refinement sampled "
        f"{len(patch_specs)} patch(es) from {len(pairs)} axis-aligned edge(s)"
    )
    constraints = build_boundary_constraints(tiles, channel, patch_specs, settings)
    if source_label is not None:
        constraints = [replace(constraint, source_label=source_label) for constraint in constraints]
    corrections_zyx, constraints, anchor_tile = solve_tile_corrections_with_residual_rejection(
        tiles,
        constraints,
        settings,
        fixed_axes=fixed_reference_axes,
        reference_prior_weights_zyx=reference_prior_weights_zyx,
        residual_reject_axes=residual_reject_axes,
    )
    corrected_params = apply_corrections_to_params(params, corrections_zyx, tiles[0].spacing)
    reference_geometry = None
    if reference_params is not None:
        if fixed_reference_axes:
            corrected_params = apply_reference_fixed_axes(corrected_params, reference_params, fixed_reference_axes)
        if reference_input is None:
            raise ValueError("reference_input is required when reference_params are provided")
        reference_geometry = reference_geometry_constraint(
            mode=(
                reference_geometry_mode
                or ("fixed-xy" if fixed_reference_axes == {"y", "x"} else "fixed-axes")
            ),
            reference_input=reference_input,
            fixed_axes=fixed_reference_axes or set(),
            params=corrected_params,
            reference_params=reference_params,
            constraints=constraints,
            shared_geometry_tracks=shared_geometry_tracks,
            reference_prior_weights_zyx=reference_prior_weights_zyx,
            residual_reject_axes=residual_reject_axes,
        )
```

## Shared Geometry Run Path for Simultaneous Tracks

This is used for `561` and `638`. It samples patches once from the reference geometry, builds constraints for each grouped track/channel, pools all constraints, solves one correction field, and writes the same shared parameters for each track.

```python
def run_shared_reference_geometry_registration(
    args: argparse.Namespace,
    *,
    tiles: list[TileMetadata],
    input_dir: Path,
    flatfield_dir: Path,
    configs: list[TrackRunConfig],
    reference_registration_input: Path,
) -> None:
    validate_shared_reference_geometry_run(args, configs, reference_registration_input)
    reference_options = reference_geometry_solver_options(
        args.reference_geometry_mode,
        args.reference_xy_prior_weight,
    )
    log(
        "Starting shared reference-constrained registration for tracks "
        f"{tuple(config.track.slug for config in configs)} "
        f"with mode={args.reference_geometry_mode}"
    )
    if reference_options.reference_prior_weights_zyx is not None:
        log(f"Reference xy prior weights z/y/x: {reference_options.reference_prior_weights_zyx}")
    for config in configs:
        log(
            f"Shared track {config.track.slug} ({config.track.track_id}): "
            f"channels={config.track.channels}, names={config.track.channel_names}"
        )
    if args.dry_run:
        return

    require_cuda_for_robust_boundary()
    log(f"Input directory: {input_dir}")
    log(f"Flatfield directory: {flatfield_dir}")
    log(f"Loading shared reference registration params from {reference_registration_input.resolve()}")
    reference_params = load_registration_params(reference_registration_input.resolve(), tiles)
    log_transform_translation_summary("Shared reference registration", reference_params)

    settings = RobustBoundarySettings()
    combined_constraints: list[BoundaryConstraint] = []
    patch_specs = sample_boundary_patches(
        tiles,
        reference_params,
        axis_aligned_registration_pairs(tiles),
        settings,
    )
    log(
        "Shared robust boundary refinement sampled "
        f"{len(patch_specs)} patch(es) per grouped track"
    )
    for config in configs:
        reg_source_channel = registration_source_channel(
            config.selected_channels,
            reg_channel_index=args.reg_channel_index,
            n_channels=tile_channel_count(tiles[0]),
        )
        log(f"Building shared boundary constraints for {config.track.slug} from channel {reg_source_channel}")
        constraints = build_boundary_constraints(tiles, reg_source_channel, patch_specs, settings)
        combined_constraints.extend(
            replace(constraint, source_label=config.track.slug)
            for constraint in constraints
        )

    corrections_zyx, combined_constraints, anchor_tile = solve_tile_corrections_with_residual_rejection(
        tiles,
        combined_constraints,
        settings,
        fixed_axes=reference_options.fixed_axes,
        reference_prior_weights_zyx=reference_options.reference_prior_weights_zyx,
        residual_reject_axes=reference_options.residual_reject_axes,
    )
    shared_params = apply_corrections_to_params(reference_params, corrections_zyx, tiles[0].spacing)
    if reference_options.fixed_axes:
        shared_params = apply_reference_fixed_axes(shared_params, reference_params, reference_options.fixed_axes)
    reference_geometry = reference_geometry_constraint(
        mode=args.reference_geometry_mode,
        reference_input=reference_registration_input,
        fixed_axes=reference_options.fixed_axes,
        params=shared_params,
        reference_params=reference_params,
        constraints=combined_constraints,
        shared_geometry_tracks=tuple(config.track.slug for config in configs),
        reference_prior_weights_zyx=reference_options.reference_prior_weights_zyx,
        residual_reject_axes=reference_options.residual_reject_axes,
    )
    log_transform_translation_summary("Shared reference-constrained registration", shared_params)

    for config in configs:
        track_constraints = [
            constraint
            for constraint in combined_constraints
            if constraint.source_label == config.track.slug
        ]
        summary = robust_summary(track_constraints, corrections_zyx)
        reg_source_channel = registration_source_channel(
            config.selected_channels,
            reg_channel_index=args.reg_channel_index,
            n_channels=tile_channel_count(tiles[0]),
        )
        write_robust_boundary_qc(
            config.robust_boundary_qc_dir,
            tiles,
            shared_params,
            channel=reg_source_channel,
            constraints=track_constraints,
            corrections_zyx=corrections_zyx,
            summary=summary,
            reference_geometry=reference_geometry,
            reference_params=reference_params,
        )
        robust_refinement = RobustBoundaryRefinementResult(
            params=shared_params,
            constraints=track_constraints,
            corrections_zyx=corrections_zyx,
            anchor_tile=anchor_tile,
            output_dir=config.robust_boundary_qc_dir,
            summary=summary,
            reference_geometry=reference_geometry,
        )
        save_registration_params(
            {"params": reference_params},
            tiles,
            config.registration_output,
            robust_refinement=robust_refinement,
        )
```

## Residual Warning Payload

For reference-constrained runs, warning residuals are computed only over axes relevant to rejection. Thus `fixed-xy` excludes fixed axes, and `penalized-xy` uses z only because its `residual_reject_axes` is `{"z"}`.

```python
def robust_boundary_residual_warning_payload(
    robust_refinement: RobustBoundaryRefinementResult,
    spacing_um: dict[str, float],
) -> dict[str, Any] | None:
    spacing = np.asarray([spacing_um["z"], spacing_um["y"], spacing_um["x"]], dtype=np.float64)
    axis_to_index = {"z": 0, "y": 1, "x": 2}
    if robust_refinement.reference_geometry is not None:
        if robust_refinement.reference_geometry.residual_reject_axes is not None:
            warning_axes = set(robust_refinement.reference_geometry.residual_reject_axes)
        else:
            warning_axes = {"z", "y", "x"} - set(robust_refinement.reference_geometry.fixed_axes)
    else:
        warning_axes = {"z", "y", "x"}
    warning_indices = [axis_to_index[axis] for axis in ("z", "y", "x") if axis in warning_axes]
    if not warning_indices:
        return None
    residual_records = []
    for constraint in robust_refinement.constraints:
        if not constraint.accepted or constraint.final_residual_zyx is None:
            continue
        residual_px = np.asarray(constraint.final_residual_zyx, dtype=np.float64)
        residual_um = float(np.linalg.norm((residual_px * spacing)[warning_indices]))
        if np.isfinite(residual_um):
            residual_records.append(
                {
                    "edge": f"{constraint.pair}",
                    "residual_um": residual_um,
                    "axes": [axis for axis in ("z", "y", "x") if axis in warning_axes],
                }
            )
    warning = residual_warning_from_records(residual_records)
    if warning is not None:
        warning["axes"] = [axis for axis in ("z", "y", "x") if axis in warning_axes]
    return warning
```

## Public Wrapper API

The newer wrapper forwards reference-prior options to the legacy script.

```python
def register_tiles(
    *,
    run_dir: Path,
    position_input: Path,
    registration_output: Path,
    level: int = 4,
    registration_pair_mode: str = "robust-boundary",
    robust_boundary_qc_dir: Path | None = None,
    registration_plots_dir: Path | None = None,
    skip_registration_plots: bool = True,
    dask_num_workers: int | None = None,
    pairwise_jobs: int | None = None,
    reference_registration_input: Path | None = None,
    reference_geometry_mode: str = "none",
    reference_xy_prior_weight: float | None = None,
    shared_geometry_tracks: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> str:
    if not dry_run and shared_geometry_tracks is None:
        ensure_single_track_position_input(position_input)

    args = [
        str(run_dir),
        "--position-input",
        str(position_input),
        "--register",
        "--register-only",
        "--registration-output",
        str(registration_output),
        "--registration-pair-mode",
        registration_pair_mode,
        "--reg-res-level",
        str(level),
    ]
    if skip_registration_plots:
        args.append("--skip-registration-plots")
    else:
        args.append("--no-skip-registration-plots")
    if registration_plots_dir is not None:
        args.extend(["--registration-plots-dir", str(registration_plots_dir)])
    if robust_boundary_qc_dir is not None:
        args.extend(["--robust-boundary-qc-dir", str(robust_boundary_qc_dir)])
    if dask_num_workers is not None:
        args.extend(["--dask-num-workers", str(dask_num_workers)])
    if pairwise_jobs is not None:
        args.extend(["--n-parallel-pairwise-regs", str(pairwise_jobs)])
    if reference_registration_input is not None:
        args.extend(["--reference-registration-input", str(reference_registration_input)])
    if reference_geometry_mode != "none":
        args.extend(["--reference-geometry-mode", reference_geometry_mode])
    if reference_xy_prior_weight is not None:
        args.extend(["--reference-xy-prior-weight", str(reference_xy_prior_weight)])
    if shared_geometry_tracks is not None:
        args.extend(["--shared-geometry-tracks", ",".join(shared_geometry_tracks)])
    if dry_run:
        args.append("--dry-run")
    return run_legacy_script("stitch_20x_tl_multiview.py", args, dry_run=dry_run)
```

## Cross-Validation Script Excerpt

This script reuses cached accepted boundary measurements and does not reread TIFF image data. It is included because the choice of `reference_xy_prior_weight=0.01` was based on this sweep.

```python
DEFAULT_PRIOR_WEIGHTS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def split_folds(
    constraints: list[stitch.BoundaryConstraint],
    *,
    n_folds: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    groups: dict[tuple[str, tuple[int, int], str], list[int]] = defaultdict(list)
    for index, constraint in enumerate(constraints):
        groups[(constraint.source_label or "", constraint.pair, constraint.axis)].append(index)

    folds = [-1 for _ in constraints]
    for indices in groups.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for offset, index in enumerate(shuffled):
            folds[index] = offset % n_folds

    if any(fold < 0 for fold in folds):
        raise RuntimeError("Internal error: not all constraints were assigned a fold")
    return folds


def residuals_px(
    corrections_zyx: list[tuple[float, float, float]],
    constraints: list[stitch.BoundaryConstraint],
) -> np.ndarray:
    corrections = np.asarray(corrections_zyx, dtype=float)
    residuals = []
    for constraint in constraints:
        residuals.append(
            corrections[constraint.moving] - corrections[constraint.fixed] - np.asarray(constraint.shift_zyx, dtype=float)
        )
    return np.asarray(residuals, dtype=float)


def residual_metrics(
    residual_um: np.ndarray,
    weights: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    abs_residual = np.abs(residual_um)
    norm_zyx = np.linalg.norm(residual_um, axis=1)
    norm_xy = np.linalg.norm(residual_um[:, 1:3], axis=1)
    return {
        f"{prefix}_n": float(residual_um.shape[0]),
        f"{prefix}_zyx_weighted_rms_um": weighted_rms(norm_zyx, weights),
        f"{prefix}_zyx_median_um": float(np.median(norm_zyx)),
        f"{prefix}_zyx_p95_um": float(np.percentile(norm_zyx, 95)),
        f"{prefix}_z_weighted_rms_um": weighted_rms(residual_um[:, 0], weights),
        f"{prefix}_z_median_abs_um": float(np.median(abs_residual[:, 0])),
        f"{prefix}_z_p95_abs_um": float(np.percentile(abs_residual[:, 0], 95)),
        f"{prefix}_xy_weighted_rms_um": weighted_rms(norm_xy, weights),
        f"{prefix}_xy_median_um": float(np.median(norm_xy)),
        f"{prefix}_xy_p95_um": float(np.percentile(norm_xy, 95)),
        f"{prefix}_y_p95_abs_um": float(np.percentile(abs_residual[:, 1], 95)),
        f"{prefix}_x_p95_abs_um": float(np.percentile(abs_residual[:, 2], 95)),
    }
```

## Tests Covering the Contract

```python
def test_fixed_xy_solver_solves_z_only_and_keeps_xy_corrections_zero() -> None:
    settings = stitch.RobustBoundarySettings(
        max_correction_zyx=(16, 96, 96),
        max_final_residual_zyx=(4.0, 8.0, 8.0),
    )
    constraints = [
        constraint(0, 1, (3.0, 40.0, -25.0), weight=10.0),
        constraint(1, 2, (2.0, -30.0, 12.0), weight=10.0),
    ]

    corrections, filtered, anchor_tile = stitch.solve_tile_corrections_with_residual_rejection(
        [tile(0), tile(1), tile(2)],
        constraints,
        settings,
        fixed_axes={"y", "x"},
    )

    assert anchor_tile == 1
    assert all(item.accepted for item in filtered)
    assert [correction[1:] for correction in corrections] == [(0.0, 0.0)] * 3
    assert corrections[1][0] == 0.0
    assert corrections[0][0] == pytest.approx(-3.0)
    assert corrections[2][0] == pytest.approx(2.0)
    assert filtered[0].final_residual_zyx == pytest.approx((0.0, -40.0, 25.0))


def test_full_xyz_solver_solves_lateral_corrections_when_axes_are_not_fixed() -> None:
    settings = stitch.RobustBoundarySettings(
        max_correction_zyx=(16, 96, 96),
        max_final_residual_zyx=(4.0, 8.0, 8.0),
    )
    constraints = [
        constraint(0, 1, (3.0, 40.0, -25.0), weight=10.0),
        constraint(1, 2, (2.0, -30.0, 12.0), weight=10.0),
    ]

    corrections, filtered, anchor_tile = stitch.solve_tile_corrections_with_residual_rejection(
        [tile(0), tile(1), tile(2)],
        constraints,
        settings,
        fixed_axes=None,
    )

    assert anchor_tile == 1
    assert all(item.accepted for item in filtered)
    assert corrections[1] == pytest.approx((0.0, 0.0, 0.0))
    assert corrections[0] == pytest.approx((-3.0, -40.0, 25.0))
    assert corrections[2] == pytest.approx((2.0, -30.0, 12.0))
    assert filtered[0].final_residual_zyx == pytest.approx((0.0, 0.0, 0.0))


def test_reference_prior_penalizes_lateral_corrections_without_fixing_them() -> None:
    settings = stitch.RobustBoundarySettings(
        max_correction_zyx=(16, 96, 96),
        irls_iterations=1,
    )
    constraints = [constraint(0, 1, (0.0, 40.0, -20.0), weight=0.001)]

    corrections, filtered, anchor_tile = stitch.solve_tile_corrections_with_residual_rejection(
        [tile(0), tile(1)],
        constraints,
        settings,
        reference_prior_weights_zyx=(0.0, 0.01, 0.01),
        residual_reject_axes={"z"},
    )

    assert anchor_tile == 0
    assert all(item.accepted for item in filtered)
    assert corrections[0] == pytest.approx((0.0, 0.0, 0.0))
    assert corrections[1][0] == pytest.approx(0.0)
    assert corrections[1][1] == pytest.approx(40.0 / 11.0)
    assert corrections[1][2] == pytest.approx(-20.0 / 11.0)
    assert abs(filtered[0].final_residual_zyx[1]) > settings.max_final_residual_zyx[1]


def test_reference_geometry_solver_options_define_reference_prior_contract() -> None:
    fixed_xy = stitch.reference_geometry_solver_options("fixed-xy", 0.01)
    full_xyz = stitch.reference_geometry_solver_options("full-xyz", 0.01)
    penalized_xy = stitch.reference_geometry_solver_options("penalized-xy", 0.01)

    assert fixed_xy.fixed_axes == {"y", "x"}
    assert fixed_xy.reference_prior_weights_zyx is None
    assert fixed_xy.residual_reject_axes is None

    assert full_xyz.fixed_axes == set()
    assert full_xyz.reference_prior_weights_zyx is None
    assert full_xyz.residual_reject_axes is None

    assert penalized_xy.fixed_axes == set()
    assert penalized_xy.reference_prior_weights_zyx == (0.0, 0.01, 0.01)
    assert penalized_xy.residual_reject_axes == {"z"}
```

## Validation Already Run

The current code passed:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex-pycache \
PYTHONPATH=/home/chaichontat/squisher/lightsheet/src \
CONDA_NO_PLUGINS=true conda run -n multi python -m py_compile \
  /home/chaichontat/squisher/lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py \
  /home/chaichontat/squisher/lightsheet/src/squisher_lightsheet/registration.py \
  /home/chaichontat/squisher/lightsheet/src/squisher_lightsheet/cli.py

cd /home/chaichontat/squisher && \
RUFF_CACHE_DIR=/tmp/ruff-cache \
PYTHONPYCACHEPREFIX=/tmp/codex-pycache \
PYTHONPATH=/home/chaichontat/squisher/lightsheet/src \
CONDA_NO_PLUGINS=true conda run -n multi ruff check \
  lightsheet/src/squisher_lightsheet/_legacy/stitch_20x_tl_multiview.py \
  lightsheet/src/squisher_lightsheet/registration.py \
  lightsheet/src/squisher_lightsheet/cli.py \
  lightsheet/tests/test_legacy_reference_geometry.py \
  lightsheet/tests/test_registration.py \
  --output-format=concise

cd /home/chaichontat/squisher && \
PYTHONPYCACHEPREFIX=/tmp/codex-pycache \
PYTHONPATH=/home/chaichontat/squisher/lightsheet/src \
CONDA_NO_PLUGINS=true conda run -n multi pytest \
  lightsheet/tests/test_legacy_reference_geometry.py \
  lightsheet/tests/test_registration.py -q
```

Result: `9 passed`.

