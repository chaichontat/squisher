# Cross-Channel Rigid Transform Investigation

Date: 2026-06-26

This note summarizes an ongoing investigation into per-tile cross-channel
lightsheet registration for large OME-Zarr tiles. It is intended for an
external reviewer who cannot see the source repository or the prior
conversation.

## Context

We are aligning `Image_10` channel 0 to `Image_14` on a per-tile basis. This is
cross-channel lightsheet registration, not within-channel tile stitching. Each
native level-0 tile is roughly `3000` z planes by `960 x 960` pixels in y/x.
The working test case has been tile `064`, using level-2 initialization
followed by level-0 refinement on a `200 x 480 x 480` crop. Data are in `zyx`
order.

The current workflow:

1. Read fixed and moving tile volumes at level 2 using 200 sampled z planes.
2. Estimate an initial translation by phase correlation.
3. Fit a transform at level 2.
4. Pick a level-0 crop with content.
5. Map the level-2 transform into the level-0 crop frame.
6. Fit a local level-0 transform on the crop.
7. Convert the local crop transform back into full level-0 tile coordinates.
8. Persist the result as a homogeneous affine in microns and compose it into
   the reference registration.

The current default fit family is rigid. A full 12-DOF affine fit was tested,
but most of its benefit appears to come from the rotation component. Native
CUDA DLL method 8 is now a translation plus unit-diagonal shear model: all
diagonal linear entries are fixed at `1`, and all six off-diagonal shear terms
may be fitted.

## Main Concern

The optimizer fits rotation in the coordinate system of the selected crop. The
code converts that crop-local fit back to full-tile coordinates correctly under
the assumed convention. However, the fitted translation still depends on where
the crop was sampled, because small rotations and translations are coupled by
the pivot location.

The practical question is whether we should persist:

- a per-crop local transform,
- a per-tile transform about the full tile center, or
- a shared/global channel rotation about an explicit physical pivot.

The current implementation implicitly uses center-coordinate model parameters
and later converts them to an origin-based homogeneous matrix.

## Current Evidence

### Tile 064: Affine Versus Rigid

Using the same `200 x 480 x 480` level-0 crop:

| Fit | Corr | Gradient NCC mean | Angle |
| --- | ---: | ---: | ---: |
| rigid-only | `0.825144` | `0.754328` | `0.354 deg` |
| affine-12DOF | `0.826498` | `0.758651` | `0.403 deg` |

The affine-12DOF fit improves metrics only slightly over rigid. Its determinant
was near 1 (`1.000696`), and the singular values were near 1. The result looks
mostly rotational rather than meaningful scale/shear.

### Method 8: Native Shear-Only Fit

The native CUDA `libapi.so` supports method 8. The current method 8 definition
is a unit-diagonal shear model, not a rotation model: it fits translation plus
six off-diagonal shear coefficients, including xy-plane shear. Older method 8
definitions were tested on tile 064 after a library update on 2026-06-26; those
historical results are kept below only as prior evidence.

Single-crop comparison:

| Fit | Corr | Gradient NCC mean |
| --- | ---: | ---: |
| affine-12DOF baseline | `0.826498` | `0.758651` |
| method 8, identity-linear init | `0.823651` | `0.750289` |
| method 8, level-2 local-linear init | `0.814061` | `0.718821` |

Historical best method-8 local matrix in `zyx` order from the older axial-tilt
implementation:

```text
[1.000000   0.002056   0.000572]
[0.000000   0.999983  -0.005777]
[0.000000   0.005777   0.999983]
```

That historical matrix combined axial z-as-a-function-of-y/x terms with a
small y/x plane rotation-like block. Current method 8 instead constrains only
the diagonal to `[1, 1, 1]` in zyx order and allows every off-diagonal shear
entry, including z-to-lateral, lateral-to-z, and xy shear.

However, the z-slab sweep shows that method 8 has not solved translation
stability yet. Using eight z slabs with the same `200 x 480 x 480` y/x crop and
identity-linear initialization:

```text
native corr mean/std:          0.771 / 0.085
local translation zyx range:   [2.29, 0.47, 1.40] px
full translation zyx range:    [2.19, 18.56, 16.57] px
```

High-content z chunks aligned well:

| Fixed center z | Method 8 corr | Gradient NCC mean |
| ---: | ---: | ---: |
| `2567.5` | `0.840482` | `0.696712` |
| `2905.5` | `0.823651` | `0.750289` |
| `3060.5` | `0.920243` | `0.673110` |

But the converted full translation drifts strongly with z:

```text
full y: 181.1 -> 199.6 px
full x: -75.1 -> -91.7 px
```

Important implementation caveat: replaying the returned method-8 matrix/offset
through this repo's current CuPy center-model transform gives much worse
correlation than the native registered output. This indicates that the DLL's
returned offset convention is not the same as the repo's center-coordinate
translation convention, or at least not directly reusable with the current
`local_model_to_full()` conversion.

This is now a central review question: method 8 may be a useful constrained
affine model, but only after the returned transform convention and centered
volume reference point are decoded exactly.

### Rotation Converted to Shear-Like Approximations

We projected the affine-12DOF fit to its closest rotation, then tried
approximating that rotation with no-rotation linear models.

On tile 064:

| Transform | Corr | Gradient NCC mean |
| --- | ---: | ---: |
| fitted affine-12DOF local | `0.826498` | `0.758651` |
| projected pure rotation | `0.823255` | `0.752822` |
| nearest no-rotation SPD/shear-stretch | `0.795701` | `0.672484` |
| symmetric part of rotation | `0.795704` | `0.672492` |
| first-order small-angle off-diagonal approximation | `0.823249` | `0.752828` |

The first-order approximation works because it preserves the antisymmetric
rotation terms to first order. A true no-rotation/shear-stretch approximation
drops NCC substantially.

### Rigid Fits Across Z Slabs

We ran rigid level-0 fits on tile 064 across eight different `200 x 480 x 480`
z slabs, keeping the same y/x crop start and initialization logic.

Summary:

- mean rotation angle: `0.436 deg`
- angle standard deviation: `0.021 deg`
- angle range: `0.406` to `0.468 deg`
- mean rotation vector in `zyx` radians:
  `[-0.00570, 0.00261, 0.00424]`
- rotation-vector std:
  `[0.00019, 0.00060, 0.00058]`

Higher-content z slabs had better NCC and similar rotations:

| Fixed center z | Refined corr | Gradient NCC mean | Angle |
| ---: | ---: | ---: | ---: |
| `2567.5` | `0.846877` | `0.711325` | `0.442 deg` |
| `2905.5` | `0.825983` | `0.756147` | `0.443 deg` |
| `3060.5` | `0.920587` | `0.673094` | `0.422 deg` |

The rotation appears reasonably stable across z. The full-tile translation does
not. For example, the full y translation moved from about `188.6 px` in low z
to about `185 px` in high z. This is consistent with rotation-pivot coupling:
if the rotation is real but the pivot is implicit, the origin-frame translation
depends on which crop was used to estimate the local model.

## Current Hypothesis

The useful cross-channel signal is a small geometric tilt/rotation or a
shear-like first-order approximation to it. Full rigid supports true rotation;
method 8 supports only unit-diagonal shear. In both cases, translation remains
crop-dependent unless the pivot/reference point and transform convention are
explicit. Treating each crop-derived origin-frame translation as globally
meaningful is likely wrong.

Two candidate models:

1. **Tile-center pivot model**: fit/average a rigid rotation per tile, expressed
   about the full tile center. This is closest to current code and is simple.
2. **Shared physical pivot model**: estimate a common channel rotation or
   lateral z-shear in stage/registered coordinates, with translations measured
   relative to that pivot. This may be more scientifically appropriate if the
   effect is an optical/channel transform rather than tile-specific
   deformation.
3. **Unit-diagonal shear model**: model all cross-axis terms as off-diagonal
   shear coefficients plus translation at a clear reference point. This matches
   the observed z-dependent lateral translation drift without adding explicit
   rotation, but it requires an exact mapping from DLL output to repo
   coordinates.

The reviewer should evaluate which model is appropriate for channel-to-channel
lightsheet alignment and whether the z-slab evidence is sufficient to support a
shared rotation or lateral z-shear model.

## Relevant Code

The following snippets are the relevant implementation pieces. They are copied
from `lightsheet/src/squisher_lightsheet/channel_affine.py`.

### Local Crop Transform to Full Tile Transform

The model convention is center-based: `matrix` and `translation` map moving to
fixed in a coordinate system centered on the volume being fit.

```python
def local_model_to_full(
    *,
    local_matrix: np.ndarray,
    local_translation: np.ndarray,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    full_shape_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(local_matrix, dtype=np.float64)
    full_center = (np.asarray(full_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    crop_center = (np.asarray(crop_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    translation = (
        np.asarray(local_translation, dtype=np.float64)
        - matrix @ (np.asarray(moving_start_zyx, dtype=np.float64) + crop_center - full_center)
        - full_center
        + np.asarray(fixed_start_zyx, dtype=np.float64)
        + crop_center
    )
    return matrix.astype(np.float32), translation.astype(np.float32)
```

The inverse mapping used to initialize a crop from a full-tile model:

```python
def full_model_to_local(
    *,
    full_matrix: np.ndarray,
    full_translation: np.ndarray,
    fixed_start_zyx: np.ndarray,
    moving_start_zyx: np.ndarray,
    full_shape_zyx: np.ndarray,
    crop_shape_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(full_matrix, dtype=np.float64)
    full_center = (np.asarray(full_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    crop_center = (np.asarray(crop_shape_zyx, dtype=np.float64) - 1.0) / 2.0
    translation = (
        matrix @ (np.asarray(moving_start_zyx, dtype=np.float64) + crop_center - full_center)
        + full_center
        + np.asarray(full_translation, dtype=np.float64)
        - np.asarray(fixed_start_zyx, dtype=np.float64)
        - crop_center
    )
    return matrix.astype(np.float32), translation.astype(np.float32)
```

### Center-Model Parameters to Origin-Based Homogeneous Matrix

This converts the center-coordinate full-tile model to a homogeneous affine in
microns:

```python
def center_model_to_homogeneous_um(
    *,
    matrix_px: np.ndarray,
    translation_px: np.ndarray,
    shape_zyx: np.ndarray,
    fixed_scale_um_zyx: np.ndarray,
    moving_scale_um_zyx: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(matrix_px, dtype=np.float64)
    translation = np.asarray(translation_px, dtype=np.float64)
    center = (np.asarray(shape_zyx, dtype=np.float64) - 1.0) / 2.0
    fixed_scale = np.diag(np.abs(np.asarray(fixed_scale_um_zyx, dtype=np.float64)))
    moving_scale = np.diag(np.abs(np.asarray(moving_scale_um_zyx, dtype=np.float64)))
    origin_translation_px = center + translation - matrix @ center
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :3] = fixed_scale @ matrix @ np.linalg.inv(moving_scale)
    homogeneous[:3, 3] = fixed_scale @ origin_translation_px
    return homogeneous
```

The inverse:

```python
def homogeneous_um_to_center_model(
    *,
    homogeneous_um: np.ndarray,
    shape_zyx: np.ndarray,
    fixed_scale_um_zyx: np.ndarray,
    moving_scale_um_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.asarray(homogeneous_um, dtype=np.float64)
    if homogeneous.shape != (4, 4):
        raise ValueError(f"Expected 4x4 homogeneous affine, got {homogeneous.shape}")
    fixed_scale = np.diag(np.abs(np.asarray(fixed_scale_um_zyx, dtype=np.float64)))
    moving_scale = np.diag(np.abs(np.asarray(moving_scale_um_zyx, dtype=np.float64)))
    matrix = np.linalg.inv(fixed_scale) @ homogeneous[:3, :3] @ moving_scale
    center = (np.asarray(shape_zyx, dtype=np.float64) - 1.0) / 2.0
    origin_translation_px = np.linalg.inv(fixed_scale) @ homogeneous[:3, 3]
    translation = origin_translation_px - center + matrix @ center
    return matrix.astype(np.float32), translation.astype(np.float32)
```

### Composing the Channel Affine Into an Existing Registration

The channel affine is tile-local. It is conjugated through the stage placement
before being composed with the reference registered affine:

```python
def compose_registration_affine(
    *,
    reference_affine: dict[str, Any],
    channel_affine_um: np.ndarray,
    stage_translation_um_zyx: np.ndarray | None = None,
    moving_stage_translation_um_zyx: np.ndarray | None = None,
) -> dict[str, Any]:
    output = json.loads(json.dumps(reference_affine))
    matrix = np.asarray(output["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"registered_affine.matrix must be 4x4, got {matrix.shape}")
    fixed_stage = (
        np.zeros(3, dtype=np.float64)
        if stage_translation_um_zyx is None
        else np.asarray(stage_translation_um_zyx, dtype=np.float64)
    )
    moving_stage = (
        fixed_stage
        if moving_stage_translation_um_zyx is None
        else np.asarray(moving_stage_translation_um_zyx, dtype=np.float64)
    )
    output["matrix"] = (
        matrix
        @ _translation_matrix_zyx_um(fixed_stage)
        @ np.asarray(channel_affine_um, dtype=np.float64)
        @ np.linalg.inv(_translation_matrix_zyx_um(moving_stage))
    ).tolist()
    return output
```

### Core Per-Tile Fit Flow

This is the core part of the per-tile fit. Details such as IO and diagnostics
are omitted here only where they do not affect transform semantics.

```python
def _measure_tile_affine(
    *,
    reference_tile: rough_legacy.TileRecord,
    moving_tile: rough_legacy.TileRecord,
    reference_channel: int,
    moving_channel: int,
    init_level: int,
    init_z_samples: int,
    refine_crop_shape_zyx: tuple[int, int, int],
    max_iterations: int,
    contact_sheet_path: Path | None,
    fit_mode: AffineFitMode = "rigid",
    prior_channel_affine_um: np.ndarray | None = None,
    accepted_inliers_before_tile: int = 0,
    tile_order_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fixed_init_raw, init_factor, init_source_level, fixed_available_levels, fixed_z_l0 = sampled_tile_volume_from_subifd(
        reference_tile,
        channel=reference_channel,
        requested_level=init_level,
        z_samples=init_z_samples,
    )
    moving_init_raw, moving_init_factor, moving_source_level, moving_available_levels, _moving_z_l0 = (
        sampled_tile_volume_from_subifd(
            moving_tile,
            channel=moving_channel,
            requested_level=init_level,
            z_samples=init_z_samples,
        )
    )
    fixed_init = _robust_norm(fixed_init_raw)
    moving_init = _robust_norm(moving_init_raw)

    if prior_channel_affine_um is None:
        phase_shift = np.asarray(estimate_translation_gpu(fixed_init, moving_init), dtype=np.float32)
        initial_matrix = np.eye(3, dtype=np.float32)
        initial_translation = phase_shift
    else:
        prior_level0_matrix, prior_level0_translation = homogeneous_um_to_center_model(
            homogeneous_um=prior_channel_affine_um,
            shape_zyx=reference_tile.shape_zyx,
            fixed_scale_um_zyx=reference_tile.scale_zyx_um,
            moving_scale_um_zyx=moving_tile.scale_zyx_um,
        )
        prior_sampled_matrix, prior_sampled_translation = level0_model_to_sampled(
            level0_matrix=prior_level0_matrix,
            level0_translation=prior_level0_translation,
            fixed_sampled_factor_zyx=init_factor,
            moving_sampled_factor_zyx=moving_init_factor,
        )
        moving_prior_registered = transform_volume_gpu(moving_init, prior_sampled_matrix, prior_sampled_translation)
        phase_shift = np.asarray(estimate_translation_gpu(fixed_init, moving_prior_registered), dtype=np.float32)
        initial_matrix = prior_sampled_matrix
        initial_translation = (prior_sampled_translation + phase_shift).astype(np.float32)

    if fit_mode == "rigid":
        init_stage_modes = ("translation", "rigid")
        refine_stage_modes = ("rigid",)
    else:
        init_stage_modes = ("translation", "rigid", "scale-9dof", "affine-12dof")
        refine_stage_modes = ("affine-12dof",)

    init_matrix, init_translation, init_corr = fit_affine_gpu(
        fixed_init,
        moving_init,
        initial_matrix=initial_matrix,
        initial_translation=initial_translation,
        max_iterations=max_iterations,
        stage_modes=init_stage_modes,
    )
    level0_matrix, level0_translation = model_to_level0(
        model_matrix=init_matrix,
        model_translation=init_translation,
        fixed_sampled_factor_zyx=init_factor,
        moving_sampled_factor_zyx=moving_init_factor,
    )
    moving_init_registered = transform_volume_gpu(moving_init, init_matrix, init_translation)
    fixed_start_l0, crop_selection = select_content_fixed_crop_start_l0(
        fixed_sampled=fixed_init,
        moving_registered_sampled=moving_init_registered,
        sampled_factor_zyx=init_factor,
        sampled_z_l0=fixed_z_l0,
        tile_shape_zyx=reference_tile.shape_zyx,
        crop_shape_zyx=refine_crop_shape_zyx,
    )
    moving_start_l0 = moving_crop_start_for_fixed_crop(
        fixed_start_zyx=fixed_start_l0,
        crop_shape_zyx=refine_crop_shape_zyx,
        full_matrix=level0_matrix,
        full_translation=level0_translation,
        fixed_shape_zyx=reference_tile.shape_zyx,
        moving_shape_zyx=moving_tile.shape_zyx,
    )
    local_init_matrix, local_init_translation = full_model_to_local(
        full_matrix=level0_matrix,
        full_translation=level0_translation,
        fixed_start_zyx=fixed_start_l0,
        moving_start_zyx=moving_start_l0,
        full_shape_zyx=reference_tile.shape_zyx,
        crop_shape_zyx=np.asarray(refine_crop_shape_zyx, dtype=np.int64),
    )

    fixed_refine_raw, fixed_crop_slices = _read_level0_crop(
        reference_tile,
        channel=reference_channel,
        start_zyx=fixed_start_l0,
        crop_shape_zyx=refine_crop_shape_zyx,
    )
    moving_refine_raw, moving_crop_slices = _read_level0_crop(
        moving_tile,
        channel=moving_channel,
        start_zyx=moving_start_l0,
        crop_shape_zyx=refine_crop_shape_zyx,
    )
    fixed_refine = _robust_norm(fixed_refine_raw)
    moving_refine = _robust_norm(moving_refine_raw)
    local_refined_matrix, local_refined_translation, refined_corr = fit_affine_gpu(
        fixed_refine,
        moving_refine,
        initial_matrix=local_init_matrix,
        initial_translation=local_init_translation,
        max_iterations=max_iterations,
        stage_modes=refine_stage_modes,
    )
    refined_matrix, refined_translation = local_model_to_full(
        local_matrix=local_refined_matrix,
        local_translation=local_refined_translation,
        fixed_start_zyx=fixed_start_l0,
        moving_start_zyx=moving_start_l0,
        full_shape_zyx=reference_tile.shape_zyx,
        crop_shape_zyx=np.asarray(refine_crop_shape_zyx, dtype=np.int64),
    )
    channel_affine_um = center_model_to_homogeneous_um(
        matrix_px=refined_matrix,
        translation_px=refined_translation,
        shape_zyx=reference_tile.shape_zyx,
        fixed_scale_um_zyx=reference_tile.scale_zyx_um,
        moving_scale_um_zyx=moving_tile.scale_zyx_um,
    )
```

## Diagnostic Scripts Used

### Method 8 Z-Slab Translation Probe

This script reruns method 8 on multiple z slabs while holding y/x fixed. It
also checks whether the returned native matrix/offset can be replayed by the
repo's current center-model transform convention.

```python
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/chaichontat/nvme/lightsheet/231-Ptprz1")
from native_reg3dgpu import _run_reg_3dgpu

from squisher_lightsheet.channel_affine import (
    _corr,
    _read_level0_crop,
    _robust_norm,
    full_model_to_local,
    gradient_component_ncc_3d_gpu,
    local_model_to_full,
    moving_crop_start_for_fixed_crop,
    transform_volume_gpu,
)
from squisher_lightsheet.tile_phase import make_moving_tile_record, tile_record_from_position_record

P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)


def zyx_to_xyz_3x4(matrix_zyx, offset_zyx):
    out = np.zeros((3, 4), dtype=np.float32)
    out[:, :3] = P @ np.asarray(matrix_zyx, dtype=np.float32) @ P
    out[:, 3] = P @ np.asarray(offset_zyx, dtype=np.float32)
    return out


summary = json.loads(Path("/tmp/tile064-affine-cli/tile_affine_alignment.json").read_text())
row = summary["measurements"][0]
ref_tile = tile_record_from_position_record(json.loads(Path("/tmp/tile064.reference.positions.json").read_text())["tiles"][0])
mov_tile = make_moving_tile_record(ref_tile, Path(row["moving_path"]))
crop = np.asarray(row["refine_crop_shape_zyx"], dtype=np.int64)
prior_fixed_start = np.array([lo for lo, _ in row["fixed_refine_crop_slices_zyx"]], dtype=np.int64)
yx_start = prior_fixed_start[1:3]
full_init_matrix = np.asarray(row["initial_level0_moving_to_fixed_matrix_zyx"], dtype=np.float32)
full_init_translation = np.asarray(row["initial_level0_moving_to_fixed_translation_px_zyx"], dtype=np.float32)
shape = np.asarray(ref_tile.shape_zyx, dtype=np.int64)

z_starts = np.unique(np.rint(np.linspace(0, int(shape[0] - crop[0]), 7)).astype(np.int64))
z_starts = np.unique(np.r_[z_starts, prior_fixed_start[0]]).astype(np.int64)

for z_start in z_starts:
    fixed_start = np.asarray([int(z_start), int(yx_start[0]), int(yx_start[1])], dtype=np.int64)
    moving_start = moving_crop_start_for_fixed_crop(
        fixed_start_zyx=fixed_start,
        crop_shape_zyx=tuple(int(v) for v in crop),
        full_matrix=full_init_matrix,
        full_translation=full_init_translation,
        fixed_shape_zyx=ref_tile.shape_zyx,
        moving_shape_zyx=mov_tile.shape_zyx,
    )
    _local_init_matrix, local_init_translation = full_model_to_local(
        full_matrix=full_init_matrix,
        full_translation=full_init_translation,
        fixed_start_zyx=fixed_start,
        moving_start_zyx=moving_start,
        full_shape_zyx=ref_tile.shape_zyx,
        crop_shape_zyx=crop,
    )
    fixed = _robust_norm(
        _read_level0_crop(ref_tile, channel=summary["reference_channel"], start_zyx=fixed_start, crop_shape_zyx=tuple(crop))[0]
    )
    moving = _robust_norm(
        _read_level0_crop(mov_tile, channel=summary["moving_channel"], start_zyx=moving_start, crop_shape_zyx=tuple(crop))[0]
    )

    result = _run_reg_3dgpu(
        fixed,
        moving,
        aff_method=8,
        lib_dir=Path("/home/chaichontat/microImageLib/bin/linux"),
        ftol=1e-4,
        max_iterations=300,
        device=0,
        tmx_only=False,
        initial_matrix_xyz_3x4=zyx_to_xyz_3x4(np.eye(3, dtype=np.float32), local_init_translation),
    )
    local_matrix = result.matrix_zyx.astype(np.float32)
    local_translation = result.offset_zyx.astype(np.float32)
    full_matrix, full_translation = local_model_to_full(
        local_matrix=local_matrix,
        local_translation=local_translation,
        fixed_start_zyx=fixed_start,
        moving_start_zyx=moving_start,
        full_shape_zyx=ref_tile.shape_zyx,
        crop_shape_zyx=crop,
    )
    native_registered = result.registered_zyx.astype(np.float32, copy=False)
    replay_registered = transform_volume_gpu(moving, local_matrix, local_translation)
    print(
        fixed_start.tolist(),
        "native_corr",
        _corr(fixed, native_registered),
        "native_grad",
        gradient_component_ncc_3d_gpu(fixed, native_registered)["mean"],
        "replay_corr",
        _corr(fixed, replay_registered),
        "local_translation",
        local_translation.tolist(),
        "full_translation",
        full_translation.tolist(),
        "matrix",
        local_matrix.tolist(),
        "records",
        result.records.tolist(),
    )
```

### Z-Slab Rigid Stability Probe

This script reruns rigid fitting on multiple z slabs while holding y/x fixed.

```python
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from squisher_lightsheet.channel_affine import (
    _corr,
    _read_level0_crop,
    _robust_norm,
    fit_affine_gpu,
    full_model_to_local,
    gradient_component_ncc_3d_gpu,
    local_model_to_full,
    moving_crop_start_for_fixed_crop,
    transform_volume_gpu,
)
from squisher_lightsheet.tile_phase import make_moving_tile_record, tile_record_from_position_record

summary = json.loads(Path("/tmp/tile064-affine-cli/tile_affine_alignment.json").read_text())
row = summary["measurements"][0]
ref_tile = tile_record_from_position_record(json.loads(Path("/tmp/tile064.reference.positions.json").read_text())["tiles"][0])
mov_tile = make_moving_tile_record(ref_tile, Path(row["moving_path"]))

crop = np.asarray(row["refine_crop_shape_zyx"], dtype=np.int64)
prior_fixed_start = np.array([lo for lo, _ in row["fixed_refine_crop_slices_zyx"]], dtype=np.int64)
yx_start = prior_fixed_start[1:3]
full_init_matrix = np.asarray(row["initial_level0_moving_to_fixed_matrix_zyx"], dtype=np.float32)
full_init_translation = np.asarray(row["initial_level0_moving_to_fixed_translation_px_zyx"], dtype=np.float32)
shape = np.asarray(ref_tile.shape_zyx, dtype=np.int64)

z_starts = np.unique(np.rint(np.linspace(0, int(shape[0] - crop[0]), 7)).astype(np.int64))
z_starts = np.unique(np.r_[z_starts, prior_fixed_start[0]]).astype(np.int64)

for z_start in z_starts:
    fixed_start = np.asarray([int(z_start), int(yx_start[0]), int(yx_start[1])], dtype=np.int64)
    moving_start = moving_crop_start_for_fixed_crop(
        fixed_start_zyx=fixed_start,
        crop_shape_zyx=tuple(int(v) for v in crop),
        full_matrix=full_init_matrix,
        full_translation=full_init_translation,
        fixed_shape_zyx=ref_tile.shape_zyx,
        moving_shape_zyx=mov_tile.shape_zyx,
    )
    local_init_matrix, local_init_translation = full_model_to_local(
        full_matrix=full_init_matrix,
        full_translation=full_init_translation,
        fixed_start_zyx=fixed_start,
        moving_start_zyx=moving_start,
        full_shape_zyx=ref_tile.shape_zyx,
        crop_shape_zyx=crop,
    )
    fixed = _robust_norm(
        _read_level0_crop(ref_tile, channel=summary["reference_channel"], start_zyx=fixed_start, crop_shape_zyx=tuple(crop))[0]
    )
    moving = _robust_norm(
        _read_level0_crop(mov_tile, channel=summary["moving_channel"], start_zyx=moving_start, crop_shape_zyx=tuple(crop))[0]
    )
    refined_matrix, refined_translation, objective_corr = fit_affine_gpu(
        fixed,
        moving,
        initial_matrix=local_init_matrix,
        initial_translation=local_init_translation,
        max_iterations=20,
        stage_modes=("rigid",),
    )
    refined_moved = transform_volume_gpu(moving, refined_matrix, refined_translation)
    full_matrix, full_translation = local_model_to_full(
        local_matrix=refined_matrix,
        local_translation=refined_translation,
        fixed_start_zyx=fixed_start,
        moving_start_zyx=moving_start,
        full_shape_zyx=ref_tile.shape_zyx,
        crop_shape_zyx=crop,
    )
    rotvec = Rotation.from_matrix(np.asarray(full_matrix, dtype=np.float64)).as_rotvec()
    print(
        fixed_start.tolist(),
        "corr",
        _corr(fixed, refined_moved),
        "grad",
        gradient_component_ncc_3d_gpu(fixed, refined_moved)["mean"],
        "rotvec",
        rotvec.tolist(),
        "angle_deg",
        float(np.linalg.norm(rotvec) * 180 / np.pi),
        "full_translation",
        full_translation.tolist(),
    )
```

## Questions for Review

1. What final transform family best matches cross-channel lightsheet geometry:
   full rigid, restricted lateral z-shear, method-8-style constrained affine,
   or a shared physical model across tiles?
2. Should the final model have a full-tile pivot, a per-tile physical pivot, or
   a shared global/channel pivot in stage or registered coordinates?
3. Given the z-slab stability above, should translation be refit after fixing a
   shared rotation/tilt, rather than derived from each crop-local fit?
4. What is the correct interpretation of the native method-8 returned
   matrix/offset? It aligns well natively, but replay through the repo's
   center-model transform gives much worse correlation.
5. Are the `zyx` center-coordinate conventions above appropriate, or should the
   model be rewritten around explicit physical coordinates from the start?
6. Is there a better way to combine multiple z-slab measurements than averaging
   rotations/tilts and separately estimating translation at a declared pivot?

## Current Recommended Direction

Do not commit to the current crop-derived translation convention. The next
model should make the pivot explicit and should first decode the method-8
native transform convention. A pragmatic next step is to fit multiple
high-content z slabs per tile, estimate a shared rotation/tilt component, then
refit or solve translation at a declared full-tile or global physical pivot.
The output registration should record the pivot convention explicitly so
downstream fusion does not depend on which crop was selected during fitting.
