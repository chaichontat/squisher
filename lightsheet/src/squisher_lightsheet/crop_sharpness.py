"""Measure focus on registered native crops and reuse pair preferences in fusion."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import itertools
import json
from pathlib import Path

import cupy as cp
import numpy as np
from loguru import logger
from multiview_stitcher import mv_graph, spatial_image_utils as si_utils, transformation
from multiview_stitcher.misc_utils import requires_overlap
from multiview_stitcher.weights import nan_gaussian_filter
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

from squisher_lightsheet.artifact_io import write_text_set_atomic
from squisher_lightsheet.seam_fusion import feather_ownership


def native_crop_scores(views, sigma: tuple[float, ...]) -> list[float]:
    """Compare high-pass energy on the same native pixels, normalized for gain."""
    xp = cp if isinstance(views, cp.ndarray) else np
    common = xp.isfinite(views).all(axis=0)
    if not bool(common.any()):
        raise ValueError("Sharpness crops have no common native support")
    scores = []
    for view in views:
        masked = xp.where(common, view, xp.nan)
        reference = masked.ravel()[common.ravel().argmax()]
        centered = masked - reference
        high_pass = centered - nan_gaussian_filter(centered, sigma=sigma, mode="nearest")
        mean = float(xp.nanmean(masked))
        scores.append(float(xp.nanmean(high_pass**2)) / mean**2 if mean > 0 else 0.0)
    return scores


def overlap_crop_boxes(first: dict, second: dict, spacing: np.ndarray) -> list[dict]:
    """Use convex overlap geometry to place three contiguous native crop boxes."""
    equations = np.concatenate(
        [
            mv_graph.get_halfspace_equations_from_stack_props(first),
            mv_graph.get_halfspace_equations_from_stack_props(second),
        ]
    )
    normals, offsets = equations[:, :3], equations[:, 3]
    result = linprog(
        [0, 0, 0, -1],
        A_ub=np.column_stack([normals, np.linalg.norm(normals, axis=1)]),
        b_ub=-offsets,
        bounds=[(None, None)] * 3 + [(0, None)],
        method="highs",
    )
    if result.status == 2 or (result.success and result.x[3] <= 0):
        return []
    if not result.success:
        raise ValueError(f"Could not locate overlap crops: {result.message}")
    vertices = HalfspaceIntersection(equations, result.x[:3]).intersections
    middle = vertices.mean(axis=0)
    long_axis = int(np.argmax(np.ptp(vertices, axis=0) / spacing))
    centers = [
        (middle + vertices[vertices[:, long_axis].argmin()]) / 2,
        middle,
        (middle + vertices[vertices[:, long_axis].argmax()]) / 2,
    ]
    requested_half = np.array([15, 95, 95])
    projected_half = np.abs(normals) @ (requested_half * spacing)
    boxes = []
    for center in centers:
        ratio = min(1.0, float(np.min(-(normals @ center + offsets) / projected_half)))
        half = np.floor(requested_half * max(0, ratio)).astype(int)
        boxes.append(
            {
                "origin": dict(zip("zyx", (center - half * spacing).tolist(), strict=True)),
                "spacing": dict(zip("zyx", spacing.tolist(), strict=True)),
                "shape": dict(zip("zyx", (2 * half + 1).tolist(), strict=True)),
            }
        )
    return boxes


def fit_crop_preferences(
    *,
    sims,
    source_paths,
    transform_key,
    output_spacing,
    output: Path,
    read_config: dict | None,
    sigma: tuple[float, ...],
) -> list[list[int]]:
    """Score each physical source overlap once and persist its native crop provenance."""
    from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy

    props = [si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key) for sim in sims]
    identity = legacy.json_safe(
        {
            "metric": "native-normalized-highpass-v1",
            "sigma": sigma,
            "sources": [
                legacy.source_tile_resume_identity(Path(p), require_completion=False) for p in source_paths
            ],
            "transformed_properties": props,
            "output_spacing": output_spacing,
            "corrections": None
            if read_config is None
            else {
                str(key): legacy.basic_correction_fingerprint(value)
                for key, value in read_config["inverse_flatfields"].items()
            },
        }
    )
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if output.exists():
        recorded = json.loads(output.read_text())
        if recorded["identity_sha256"] != fingerprint and not legacy.fusion_resume_plans_match(
            recorded.get("identity"), identity
        ):
            raise ValueError(f"Sharpness crop provenance changed: {output}")
        return recorded["pair_preferences"]
    matrix = np.zeros((len(sims), len(sims)), dtype=np.int8)
    records = []
    spacing = np.array([output_spacing[d] for d in "zyx"])
    corners = [mv_graph.get_vertices_from_stack_props(p) for p in props]
    bounds = [(v.min(axis=0), v.max(axis=0)) for v in corners]
    context = nullcontext() if read_config is None else legacy.basic_corrected_zarr_reads(**read_config)
    with cp.cuda.Device(0), context:
        for i, j in itertools.combinations(range(len(sims)), 2):
            if np.any(np.minimum(bounds[i][1], bounds[j][1]) <= np.maximum(bounds[i][0], bounds[j][0])):
                continue
            boxes = overlap_crop_boxes(props[i], props[j], spacing)
            if not boxes:
                continue
            crop_records = []
            for box in boxes:
                pair_sims = [sims[i], sims[j]]
                selection = {
                    d: pair_sims[0].coords[d].values[0]
                    for d in si_utils.get_nonspatial_dims_from_sim(pair_sims[0])
                }
                entry, _ = legacy.direct_zarr_block_entry(
                    sims=pair_sims,
                    transform_key=transform_key,
                    sim_coord_dict=selection,
                    block_key=(0, 0, 0),
                    output_stack_properties=box,
                    output_chunksize=box["shape"],
                    output_chunk_shape=box["shape"],
                    output_chunk_origin=box["origin"],
                    overlap_in_pixels=dict.fromkeys("zyx", 0),
                    interpolation_order=1,
                )
                if len(entry["views"]) != 2:
                    raise ValueError(
                        f"Native crop does not intersect both sources: {source_paths[i]}, {source_paths[j]}"
                    )
                views = []
                for view in entry["views"]:
                    section = si_utils.deserialize_zarr_backed_sim(
                        view["tile_info"],
                        reconstruct_slice=True,
                        overlap_bb=view["tile_overlap_bb"],
                        sim_coord_dict=selection,
                    )
                    section = section.copy(
                        data=cp.asarray(si_utils._get_backend_data(section), dtype=cp.float32)
                    )
                    pulled = transformation.transform_sim(
                        section,
                        np.linalg.inv(view["sparam"]),
                        output_stack_properties=box,
                        order=1,
                        cval=np.nan,
                    )
                    views.append(si_utils._get_backend_data(pulled))
                scores = native_crop_scores(cp.stack(views), sigma)
                crop_records.append({"box": box, "scores": scores})
            scores = np.mean([item["scores"] for item in crop_records], axis=0)
            preference = int(np.sign(scores[0] - scores[1]))
            matrix[i, j], matrix[j, i] = preference, -preference
            records.append({"first": i, "second": j, "scores": scores.tolist(), "crops": crop_records})
            logger.info(
                "Native sharpness crops: pair {}/{} ({}, {}), scores={}",
                len(records),
                len(sims),
                i,
                j,
                scores.tolist(),
            )
            cp.get_default_memory_pool().free_all_blocks()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_set_atomic(
        {
            output: json.dumps(
                {
                    "identity": identity,
                    "identity_sha256": fingerprint,
                    "pair_preferences": matrix.tolist(),
                    "pairs": records,
                },
                indent=2,
            )
            + "\n"
        }
    )
    return matrix.tolist()


@requires_overlap(
    lambda kwargs: dict(
        zip(("z", "y", "x")[-len(kwargs["feather_radius"]) :], kwargs["feather_radius"], strict=True)
    )
)
def crop_sharpness_fusion(
    transformed_views,
    blending_weights,
    *,
    pair_preferences,
    source_indices,
    feather_radius: tuple[int, ...],
    intensity_threshold: float | None = None,
):
    """Use crop pair preferences among valid sources; feather only native interfaces.

    Pairwise win-minus-loss voting resolves multiple source overlaps, with source
    order breaking ties. Decisions depend on source support, never block boundaries.
    """
    xp = cp if isinstance(transformed_views, cp.ndarray) else np
    valid = (blending_weights > 1e-7) & xp.isfinite(transformed_views)
    if intensity_threshold is not None:
        valid &= transformed_views > intensity_threshold
    if len(source_indices) != len(valid):
        raise ValueError("Crop preferences must identify every transformed source")
    preferences = np.asarray(pair_preferences)[np.ix_(source_indices, source_indices)]
    best = xp.full(valid.shape[1:], -len(valid) - 1, dtype=xp.int16)
    ownership = xp.full(valid.shape[1:], -1, dtype=xp.int32)
    for i in range(len(valid)):
        votes = xp.zeros(valid.shape[1:], dtype=xp.int16)
        for j, preference in enumerate(preferences[i]):
            if preference > 0:
                votes += valid[j]
            elif preference < 0:
                votes -= valid[j]
        better = valid[i] & (votes > best)
        xp.copyto(best, votes, where=better)
        xp.copyto(ownership, i, where=better)
    weights = feather_ownership(ownership, valid, feather_radius)
    return (xp.where(valid, transformed_views, 0) * weights).sum(axis=0).astype(transformed_views.dtype)
