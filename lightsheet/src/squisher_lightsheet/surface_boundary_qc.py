"""Render direct and recovered surface-boundary fields on matched intrinsic-Z sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates
import SimpleITK as sitk

from squisher_lightsheet import ngff
from squisher_lightsheet.artifact_io import atomic_output_directory, sha256_file
from squisher_lightsheet.surface_geometry import (
    ImageGeometry,
    SurfaceProjector,
    load_surface,
)


FloatArray = npt.NDArray[np.float64]
INNER_COLOR = "#00B8D9"
OUTER_COLOR = "#FF4F9A"


def _face_normals(vertices_xyz_um: FloatArray, faces: npt.NDArray[np.int64]) -> FloatArray:
    corners = vertices_xyz_um[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("surface contains a degenerate face")
    return normals / lengths[:, None]


def evaluate_boundary_fields(
    points_world_xyz_um: npt.ArrayLike,
    *,
    vertices_world_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    vertex_uv: npt.ArrayLike,
    cutoffs_um: dict[str, npt.ArrayLike],
    eligible_vertices: dict[str, npt.ArrayLike],
    inside_cortex: npt.ArrayLike | None = None,
    projector: SurfaceProjector | None = None,
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, int]]:
    """Evaluate ``r_vz - cutoff`` using exact projection and barycentric cutoffs.

    A point is valid only on the tissue side of an oriented, non-rim closest
    face whose three vertices are eligible for the selected boundary field.
    """
    points = np.asarray(points_world_xyz_um, dtype=np.float64)
    vertices = np.asarray(vertices_world_xyz_um, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(vertex_uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world_xyz_um must be finite with shape (n_points, 3)")
    if not cutoffs_um or set(cutoffs_um) != set(eligible_vertices):
        raise ValueError("cutoffs_um and eligible_vertices must have the same nonempty keys")

    if projector is None:
        projector = SurfaceProjector(vertices, triangles, uv)
    elif not (
        np.array_equal(projector.vertices, vertices)
        and np.array_equal(projector.faces, triangles)
        and np.array_equal(projector.vertex_uv, uv)
    ):
        raise ValueError("projector geometry does not match the supplied surface and chart")
    projection = projector.project(points)
    selected_faces = triangles[projection.face_index]
    normals = _face_normals(vertices, triangles)[projection.face_index]
    on_tissue_side = np.einsum("ij,ij->i", points - projection.closest_xyz_um, normals) <= 0
    if inside_cortex is None:
        cortex = np.ones(len(points), dtype=bool)
    else:
        cortex = np.asarray(inside_cortex)
        if cortex.shape != (len(points),) or cortex.dtype != np.bool_:
            raise ValueError("inside_cortex must be a boolean array with one value per point")
    common_valid = ~projection.on_surface_boundary & on_tissue_side & cortex
    fields: dict[str, npt.NDArray[np.float64]] = {}
    valid_counts: dict[str, int] = {}
    for name in cutoffs_um:
        cutoff = np.asarray(cutoffs_um[name], dtype=np.float64)
        eligible = np.asarray(eligible_vertices[name])
        if cutoff.shape != (len(vertices),):
            raise ValueError(f"cutoffs_um[{name!r}] must have one value per vertex")
        if eligible.shape != (len(vertices),) or eligible.dtype != np.bool_:
            raise ValueError(f"eligible_vertices[{name!r}] must be a boolean vertex array")
        face_valid = np.all(eligible[selected_faces], axis=1)
        interpolated = np.einsum("ni,ni->n", projection.barycentric, cutoff[selected_faces])
        valid = common_valid & face_valid & np.isfinite(interpolated)
        field = projection.r_vz_um - interpolated
        field[~valid] = np.nan
        fields[name] = field
        valid_counts[name] = int(np.count_nonzero(valid))
    counts = {
        "points": len(points),
        "rim": int(np.count_nonzero(projection.on_surface_boundary)),
        "tissue_side": int(np.count_nonzero(on_tissue_side)),
        "inside_cortex": int(np.count_nonzero(cortex)),
        **{f"valid_{name}": count for name, count in valid_counts.items()},
    }
    return fields, counts


def _mask_geometry(
    path: Path,
) -> tuple[ImageGeometry, tuple[int, int, int], npt.NDArray[np.bool_]]:
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError("reference mask must be three-dimensional")
    geometry = ImageGeometry.from_sitk(image)
    values = sitk.GetArrayFromImage(image)
    unique = np.unique(values)
    if not np.all(np.isin(unique, [0, 1])) or not np.any(values == 1):
        raise ValueError("reference mask must be binary and contain foreground")
    return (
        geometry,
        tuple(int(value) for value in image.GetSize()[::-1]),
        np.asarray(values, dtype=bool),
    )


def _source_level(source_zarr: Path, level: int):
    import zarr

    root = zarr.open_group(str(source_zarr), mode="r")
    if root.attrs.get("squisher_complete") is not True:
        raise ValueError(f"source OME-Zarr is not marked complete: {source_zarr}")
    dataset_path = ngff.level_path(root, level=level, context=source_zarr)
    array = root[dataset_path]
    axes = list(ngff.axes(root, array).lower())
    if axes != ["z", "y", "x"] or array.ndim != 3 or array.dtype != np.dtype(np.uint16):
        raise ValueError(
            f"expected a channel-separated 3-D uint16 ZYX source, "
            f"found axes {axes}, shape {array.shape}, dtype {array.dtype}"
        )
    dataset_index = ngff.dataset_paths(root).index(dataset_path)
    transform_axes, scale, origin, has_scale, has_origin = ngff.scale_translation(
        root, dataset_index=dataset_index
    )
    if transform_axes != axes or not has_scale or not has_origin:
        raise ValueError("source level requires explicit ZYX scale and translation")
    multiscale_axes = ngff.multiscales(root)[0]["axes"]
    units = [axis.get("unit") if isinstance(axis, dict) else None for axis in multiscale_axes]
    if units != ["micrometer"] * 3:
        raise ValueError(f"expected micrometer ZYX axes, found units {units}")
    scale_zyx = np.asarray(scale, dtype=np.float64)
    origin_zyx = np.asarray(origin, dtype=np.float64)
    if not np.isfinite(scale_zyx).all() or np.any(scale_zyx <= 0):
        raise ValueError("source scale must be finite and positive")
    if not np.isfinite(origin_zyx).all():
        raise ValueError("source translation must be finite")
    return array, dataset_path, scale_zyx, origin_zyx


def _sample_world_points(
    source: Any,
    points_world_xyz_um: FloatArray,
    scale_zyx_um: FloatArray,
    origin_zyx_um: FloatArray,
) -> npt.NDArray[np.float32]:
    index_zyx = (points_world_xyz_um[:, ::-1] - origin_zyx_um) / scale_zyx_um
    result = np.zeros(len(index_zyx), dtype=np.float32)
    # Row batches keep oblique planes from materializing their full 3-D bounding box.
    for start in range(0, len(index_zyx), 65_536):
        stop = min(start + 65_536, len(index_zyx))
        coordinates = index_zyx[start:stop]
        lower = np.floor(coordinates.min(axis=0)).astype(np.int64) - 1
        upper = np.ceil(coordinates.max(axis=0)).astype(np.int64) + 2
        clipped_lower = np.maximum(lower, 0)
        clipped_upper = np.minimum(upper, np.asarray(source.shape, dtype=np.int64))
        if np.any(clipped_lower >= clipped_upper):
            continue
        block = np.asarray(
            source[
                tuple(slice(int(lo), int(hi)) for lo, hi in zip(clipped_lower, clipped_upper, strict=True))
            ]
        )
        local = coordinates - clipped_lower
        result[start:stop] = map_coordinates(
            block,
            local.T,
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
            output=np.float32,
        )
    return result


def _resolve_sections(
    shape_zyx: tuple[int, int, int],
    spacing_z_um: float,
    *,
    intrinsic_z_indices: npt.ArrayLike | None,
    intrinsic_z_positions_um: npt.ArrayLike | None,
) -> list[dict[str, float | int]]:
    if (intrinsic_z_indices is None) == (intrinsic_z_positions_um is None):
        raise ValueError("provide exactly one of intrinsic_z_indices or intrinsic_z_positions_um")
    records: list[dict[str, float | int]] = []
    if intrinsic_z_indices is not None:
        requested = np.asarray(intrinsic_z_indices)
        if requested.ndim != 1 or not len(requested) or not np.issubdtype(requested.dtype, np.integer):
            raise ValueError("intrinsic_z_indices must be a nonempty one-dimensional integer array")
        indices = requested.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= shape_zyx[0]):
            raise ValueError(f"intrinsic Z index outside [0, {shape_zyx[0]})")
        for index in indices:
            records.append(
                {
                    "requested_index_z": int(index),
                    "requested_position_z_um": float(index * spacing_z_um),
                    "actual_index_z": int(index),
                    "actual_position_z_um": float(index * spacing_z_um),
                }
            )
    else:
        positions = np.asarray(intrinsic_z_positions_um, dtype=np.float64)
        if positions.ndim != 1 or not len(positions) or not np.isfinite(positions).all():
            raise ValueError("intrinsic_z_positions_um must be a nonempty finite 1-D array")
        indices = np.floor(positions / spacing_z_um + 0.5).astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= shape_zyx[0]):
            raise ValueError("requested intrinsic Z position has no matching reference-mask plane")
        for requested, index in zip(positions, indices, strict=True):
            records.append(
                {
                    "requested_position_z_um": float(requested),
                    "actual_index_z": int(index),
                    "actual_position_z_um": float(index * spacing_z_um),
                }
            )
    actual = [record["actual_index_z"] for record in records]
    if len(set(actual)) != len(actual):
        raise ValueError("requested sections resolve to duplicate intrinsic Z planes")
    return records


def _has_zero_contour(field: npt.ArrayLike) -> bool:
    finite = np.asarray(field)[np.isfinite(field)]
    return bool(len(finite) and finite.min() <= 0 <= finite.max())


def build_surface_boundary_qc(
    *,
    surface_path: Path,
    chart_path: Path,
    boundaries_path: Path,
    mask_path: Path,
    source_zarr: Path,
    output_dir: Path,
    source_level: int = 2,
    intrinsic_z_indices: npt.ArrayLike | None = None,
    intrinsic_z_positions_um: npt.ArrayLike | None = None,
    grid_stride: int = 2,
    grayscale_quantiles: tuple[float, float] = (0.01, 0.998),
) -> Path:
    """Publish matched direct/recovered implicit-boundary section QC atomically."""
    surface_path = Path(surface_path)
    chart_path = Path(chart_path)
    boundaries_path = Path(boundaries_path)
    mask_path = Path(mask_path)
    source_zarr = Path(source_zarr)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if source_level < 0:
        raise ValueError("source_level must be non-negative")
    if not isinstance(grid_stride, int) or isinstance(grid_stride, bool) or grid_stride < 1:
        raise ValueError("grid_stride must be a positive integer")
    quantiles = np.asarray(grayscale_quantiles, dtype=np.float64)
    if quantiles.shape != (2,) or not (0 <= quantiles[0] < quantiles[1] <= 1):
        raise ValueError("grayscale_quantiles must be increasing values in [0, 1]")

    geometry, mask_shape, cortex_mask = _mask_geometry(mask_path)
    sections = _resolve_sections(
        mask_shape,
        geometry.spacing_xyz_um[2],
        intrinsic_z_indices=intrinsic_z_indices,
        intrinsic_z_positions_um=intrinsic_z_positions_um,
    )
    mesh = load_surface(surface_path)
    vertices = mesh.vertices_world_xyz_um
    faces = mesh.faces
    for name, actual, expected in (
        ("original_image_origin_xyz_um", mesh.source_geometry.origin_xyz_um, geometry.origin_xyz_um),
        (
            "original_image_spacing_xyz_um",
            mesh.source_geometry.spacing_xyz_um,
            geometry.spacing_xyz_um,
        ),
        ("original_image_direction", mesh.source_geometry.direction, geometry.direction),
    ):
        if not np.allclose(actual, expected, atol=1e-9, rtol=1e-9):
            raise ValueError(f"surface {name} does not match the reference mask geometry")
    surface_hash = sha256_file(surface_path)
    with np.load(chart_path) as chart:
        vertex_uv = np.asarray(chart["vertex_uv"], dtype=np.float64)
        if "mesh_sha256" in chart.files and str(chart["mesh_sha256"]) != surface_hash:
            raise ValueError("chart mesh_sha256 does not match the supplied surface")
    boundary_metadata_path = boundaries_path.parent / "metadata.json"
    if not boundary_metadata_path.is_file():
        raise ValueError(f"boundary metadata is missing: {boundary_metadata_path}")
    boundary_metadata = json.loads(boundary_metadata_path.read_text())
    if not isinstance(boundary_metadata, dict):
        raise ValueError("boundary metadata must contain a JSON object")
    if boundary_metadata.get("artifact_type") != "squisher_lightsheet.surface_boundaries.v1":
        raise ValueError("boundary metadata has an unsupported artifact_type")
    if boundary_metadata.get("surface_sha256") != surface_hash:
        raise ValueError("boundary metadata surface_sha256 does not match the supplied surface")
    boundary_hash = sha256_file(boundaries_path)
    boundary_outputs = boundary_metadata.get("outputs")
    if not isinstance(boundary_outputs, dict) or boundary_outputs.get("boundaries.npz") != boundary_hash:
        raise ValueError("boundary metadata does not bind the supplied boundaries.npz")
    with np.load(boundaries_path) as boundaries:
        required = {
            "raw_inner_um",
            "raw_outer_um",
            "inner_um",
            "outer_um",
            "direct_supported_inner",
            "direct_supported_outer",
            "wall_support",
            "estimated_inner",
            "estimated_outer",
        }
        missing = required.difference(boundaries.files)
        if missing:
            raise ValueError(f"boundaries artifact is missing arrays: {sorted(missing)}")
        boundary_arrays = {name: np.asarray(boundaries[name]) for name in required}
    for name, values in boundary_arrays.items():
        if values.shape != (len(vertices),):
            raise ValueError(f"boundary array {name} must have shape ({len(vertices)},)")
    for name in required - {"raw_inner_um", "raw_outer_um", "inner_um", "outer_um"}:
        if boundary_arrays[name].dtype != np.bool_:
            raise ValueError(f"boundary array {name} must be boolean")
    if np.any(
        boundary_arrays["estimated_inner"]
        & boundary_arrays["estimated_outer"]
        & (boundary_arrays["outer_um"] < boundary_arrays["inner_um"])
    ):
        raise ValueError("estimated outer boundary must be greater than or equal to inner")

    source, dataset_path, source_scale, source_origin = _source_level(source_zarr, source_level)
    _, ny, nx = mask_shape
    x_centers = np.arange(nx, dtype=np.float64) * geometry.spacing_xyz_um[0]
    y_centers = np.arange(ny, dtype=np.float64) * geometry.spacing_xyz_um[1]
    grid_x_full, grid_y_full = np.meshgrid(x_centers, y_centers)
    section_images: list[npt.NDArray[np.float32]] = []
    world_grids: list[FloatArray] = []
    for record in sections:
        local = np.column_stack(
            (
                grid_x_full.ravel(),
                grid_y_full.ravel(),
                np.full(grid_x_full.size, record["actual_position_z_um"]),
            )
        )
        world = geometry.intrinsic_to_world(local)
        world_grids.append(world)
        section_images.append(
            _sample_world_points(source, world, source_scale, source_origin).reshape(ny, nx)
        )
    pooled = np.concatenate([image[np.isfinite(image) & (image > 0)][::8] for image in section_images])
    if not len(pooled):
        raise ValueError("selected source sections contain no positive finite intensity")
    grayscale_low, grayscale_high = np.quantile(pooled, quantiles)
    if not grayscale_low < grayscale_high:
        raise ValueError("pooled source sections do not have a usable grayscale range")

    sample_x = np.unique(np.append(np.arange(0, nx, grid_stride), nx - 1))
    sample_y = np.unique(np.append(np.arange(0, ny, grid_stride), ny - 1))
    sampled_grid_x, sampled_grid_y = np.meshgrid(x_centers[sample_x], y_centers[sample_y])
    direct_eligibility = {
        "inner": boundary_arrays["direct_supported_inner"] & np.isfinite(boundary_arrays["raw_inner_um"]),
        "outer": boundary_arrays["direct_supported_outer"] & np.isfinite(boundary_arrays["raw_outer_um"]),
    }
    recovered_eligibility = {
        "inner": boundary_arrays["wall_support"]
        & boundary_arrays["estimated_inner"]
        & np.isfinite(boundary_arrays["inner_um"]),
        "outer": boundary_arrays["wall_support"]
        & boundary_arrays["estimated_outer"]
        & np.isfinite(boundary_arrays["outer_um"]),
    }
    modes = {
        "direct": (
            {"inner": boundary_arrays["raw_inner_um"], "outer": boundary_arrays["raw_outer_um"]},
            direct_eligibility,
        ),
        "recovered": (
            {"inner": boundary_arrays["inner_um"], "outer": boundary_arrays["outer_um"]},
            recovered_eligibility,
        ),
    }
    combined_cutoffs = {
        f"{mode}_{name}": values for mode, (cutoffs, _) in modes.items() for name, values in cutoffs.items()
    }
    combined_eligibility = {
        f"{mode}_{name}": values
        for mode, (_, eligibility) in modes.items()
        for name, values in eligibility.items()
    }
    projector = SurfaceProjector(vertices, faces, vertex_uv)
    extent = (
        -geometry.spacing_xyz_um[0] / 2,
        (nx - 0.5) * geometry.spacing_xyz_um[0],
        (ny - 0.5) * geometry.spacing_xyz_um[1],
        -geometry.spacing_xyz_um[1] / 2,
    )
    figure, axes = plt.subplots(
        len(sections), 2, figsize=(11, 4.8 * len(sections)), squeeze=False, constrained_layout=True
    )
    try:
        plane_results = []
        for row, (record, image, full_world) in enumerate(
            zip(sections, section_images, world_grids, strict=True)
        ):
            sampled_world = full_world.reshape(ny, nx, 3)[np.ix_(sample_y, sample_x)].reshape(-1, 3)
            sampled_inside_cortex = cortex_mask[int(record["actual_index_z"])][
                np.ix_(sample_y, sample_x)
            ].reshape(-1)
            fields, counts = evaluate_boundary_fields(
                sampled_world,
                vertices_world_xyz_um=vertices,
                faces=faces,
                vertex_uv=vertex_uv,
                cutoffs_um=combined_cutoffs,
                eligible_vertices=combined_eligibility,
                inside_cortex=sampled_inside_cortex,
                projector=projector,
            )
            mode_results: dict[str, object] = {}
            for column, mode in enumerate(modes):
                axis = axes[row, column]
                axis.imshow(
                    image,
                    cmap="gray",
                    vmin=grayscale_low,
                    vmax=grayscale_high,
                    extent=extent,
                    interpolation="nearest",
                )
                contour_present: dict[str, bool] = {}
                for name, color in (("inner", INNER_COLOR), ("outer", OUTER_COLOR)):
                    field = fields[f"{mode}_{name}"].reshape(sampled_grid_x.shape)
                    contour_present[name] = _has_zero_contour(field)
                    if contour_present[name]:
                        axis.contour(
                            sampled_grid_x,
                            sampled_grid_y,
                            field,
                            levels=[0],
                            colors=[color],
                            linewidths=1.1,
                        )
                axis.set(
                    title=(
                        f"{mode.capitalize()} fields · intrinsic z={record['actual_position_z_um']:.1f} µm"
                    ),
                    xlabel="intrinsic X (µm)",
                    ylabel="intrinsic Y (µm)",
                )
                axis.set_aspect("equal")
                mode_results[mode] = {
                    "counts": {
                        "points": counts["points"],
                        "rim": counts["rim"],
                        "tissue_side": counts["tissue_side"],
                        "inside_cortex": counts["inside_cortex"],
                        "valid_inner": counts[f"valid_{mode}_inner"],
                        "valid_outer": counts[f"valid_{mode}_outer"],
                    },
                    "contour_present": contour_present,
                }
            plane_results.append({**record, "modes": mode_results})
        figure.legend(
            handles=[
                Line2D([0], [0], color=INNER_COLOR, lw=1.5, label="inner boundary"),
                Line2D([0], [0], color=OUTER_COLOR, lw=1.5, label="outer boundary"),
            ],
            loc="upper center",
            ncol=2,
            frameon=False,
        )
    except BaseException:
        plt.close(figure)
        raise

    with atomic_output_directory(output_dir) as stage:
        try:
            outputs: dict[str, str] = {}
            for suffix, dpi in (("png", 220), ("pdf", None), ("svg", None)):
                path = stage / f"boundary-qc.{suffix}"
                figure.savefig(path, dpi=dpi)
                outputs[path.name] = sha256_file(path)
        finally:
            plt.close(figure)
        metadata = {
            "artifact_type": "squisher_lightsheet.surface_boundary_qc.v1",
            "surface": str(surface_path.resolve()),
            "surface_sha256": surface_hash,
            "chart": str(chart_path.resolve()),
            "chart_sha256": sha256_file(chart_path),
            "boundaries": str(boundaries_path.resolve()),
            "boundaries_sha256": boundary_hash,
            "boundary_metadata": str(boundary_metadata_path.resolve()),
            "boundary_metadata_sha256": sha256_file(boundary_metadata_path),
            "reference_mask": str(mask_path.resolve()),
            "reference_mask_sha256": sha256_file(mask_path),
            "reference_geometry": {
                "shape_zyx": list(mask_shape),
                "origin_world_xyz_um": geometry.origin_xyz_um.tolist(),
                "spacing_intrinsic_xyz_um": geometry.spacing_xyz_um.tolist(),
                "direction_intrinsic_to_world": geometry.direction.tolist(),
                "direction_determinant": geometry.determinant,
            },
            "source_zarr": str(source_zarr.resolve()),
            "source_level": source_level,
            "source_dataset": dataset_path,
            "source_shape_zyx": list(source.shape),
            "source_scale_zyx_um": source_scale.tolist(),
            "source_origin_zyx_um": source_origin.tolist(),
            "section_axis": "reference-mask intrinsic Z",
            "section_sampling": "one trilinearly sampled world-physical plane; no slab projection",
            "field_definition": "exact closest-surface r_vz minus barycentric vertex cutoff",
            "validity": {
                "common": "inside cortex mask with tissue-side, non-rim closest-face projection",
                "direct": "finite raw per-edge observations with per-edge direct support",
                "recovered": "finite estimated per-edge fields on wall support",
            },
            "grid_stride_reference_pixels": grid_stride,
            "image_extent_intrinsic_xy_um_pixel_edges": list(extent),
            "grayscale": {
                "pool": "positive finite pixels from every selected section, stride 8",
                "quantiles": quantiles.tolist(),
                "limits": [float(grayscale_low), float(grayscale_high)],
            },
            "sections": plane_results,
            "outputs": outputs,
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return (output_dir / "metadata.json").resolve()
