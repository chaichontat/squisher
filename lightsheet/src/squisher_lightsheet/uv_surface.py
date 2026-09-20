from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Literal

import igl
import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, zoom
import SimpleITK as sitk

from squisher_lightsheet.artifact_io import atomic_output_directory, sha256_file
from squisher_lightsheet.surface_geometry import (
    ImageGeometry,
    validate_manifold_disk,
)


DEFAULT_SMOOTHING_UM = 19.2
ARTIFACT_TYPE = "squisher_lightsheet.ventricular_uv_surface.v1"

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def cubic_resample(values: npt.ArrayLike, factor: int = 2) -> tuple[FloatArray, FloatArray]:
    """Cubic-resample on an endpoint-aligned grid and return source-index scale."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("values must be a 3-D array")
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    output_shape = tuple(factor * (size - 1) + 1 for size in array.shape)
    sampled = zoom(
        array,
        tuple(out / size for out, size in zip(output_shape, array.shape, strict=True)),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
        grid_mode=False,
    )
    index_scale = (np.asarray(array.shape, dtype=np.float64) - 1) / (
        np.asarray(sampled.shape, dtype=np.float64) - 1
    )
    return sampled, index_scale


def positive_x_graph(
    sampled: npt.ArrayLike,
    index_scale_zyx: npt.ArrayLike,
    spacing_zyx_um: npt.ArrayLike,
    origin_zyx_um: npt.ArrayLike,
    *,
    level: float = 0.0,
) -> tuple[FloatArray, IntArray, dict[str, int]]:
    """Mesh the outermost +X level crossing on every nonempty ZY ray."""
    field = np.asarray(sampled, dtype=np.float64)
    inside = field >= level
    runs = _binary_ray_counts(inside)
    valid = runs > 0
    z_index, y_index = np.where(valid)
    if len(z_index) == 0:
        raise ValueError("smoothed mask has no nonempty X rays")
    x0 = (inside.shape[2] - 1 - np.argmax(inside[:, :, ::-1], axis=2))[z_index, y_index]
    if np.any(x0 + 1 >= inside.shape[2]):
        raise ValueError("the +X isosurface touches the resampled array boundary")
    value0 = field[z_index, y_index, x0]
    value1 = field[z_index, y_index, x0 + 1]
    x_crossing = x0 + (level - value0) / (value1 - value0)

    index_zyx = np.column_stack((z_index, y_index, x_crossing))
    physical_zyx = index_zyx * np.asarray(index_scale_zyx) * np.asarray(spacing_zyx_um) + np.asarray(
        origin_zyx_um
    )
    vertices = physical_zyx[:, ::-1]

    faces = _graph_faces(valid)
    if not len(faces):
        raise ValueError("the +X surface contains no triangles")
    return (
        np.asarray(vertices, dtype=np.float64),
        faces,
        {
            "refined_rays": int(runs.size),
            "nonempty_refined_rays": int(valid.sum()),
            "multi_interval_refined_rays": int(np.count_nonzero(runs > 1)),
        },
    )


def _graph_faces(valid: npt.NDArray[np.bool_]) -> IntArray:
    vertex_id = np.full(valid.shape, -1, dtype=np.int64)
    vertex_id[valid] = np.arange(np.count_nonzero(valid))
    faces: list[list[int]] = []
    for z in range(valid.shape[0] - 1):
        for y in range(valid.shape[1] - 1):
            corners = [
                vertex_id[z, y],
                vertex_id[z, y + 1],
                vertex_id[z + 1, y + 1],
                vertex_id[z + 1, y],
            ]
            present = [vertex >= 0 for vertex in corners]
            present_count = sum(present)
            if present_count == 4:
                faces.extend(([corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]))
            elif present_count == 3:
                faces.append([corner for corner, keep in zip(corners, present, strict=True) if keep])
    return np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def _main_sheet_source_keep(
    cut_crossing: npt.ArrayLike,
    *,
    min_z_index: int | None = None,
) -> tuple[npt.NDArray[np.bool_], dict[str, object]]:
    """Keep the superior sheet around a contiguous inferior cut bridge."""
    crossing = np.asarray(cut_crossing, dtype=bool)
    if crossing.ndim != 2:
        raise ValueError("cut_crossing must be a 2-D ZY ray grid")
    crossing_columns = np.flatnonzero(crossing.any(axis=0))
    if not len(crossing_columns):
        raise ValueError("main-sheet trimming requires rays crossing the split")
    first_y = int(crossing_columns[0])
    last_y = int(crossing_columns[-1])
    if not crossing[:, first_y : last_y + 1].any(axis=0).all():
        raise ValueError("main-sheet trimming requires a contiguous cut bridge along Y")

    z_index, y_index = np.indices(crossing.shape)
    upper_z = np.max(np.where(crossing, z_index, -1), axis=0)
    keep = (y_index < first_y) | ((y_index <= last_y) & (z_index > upper_z[y_index]))
    diagnostics: dict[str, object] = {
        "cut_bridge_y_index_range": [first_y, last_y],
        "cut_bridge_upper_z_index_by_y": upper_z[first_y : last_y + 1].tolist(),
    }
    if min_z_index is not None:
        if not 0 <= min_z_index < crossing.shape[0]:
            raise ValueError(f"main_sheet_min_z_index must be in [0, {crossing.shape[0] - 1}]")
        keep &= z_index >= min_z_index
        diagnostics["main_sheet_min_z_index"] = min_z_index
    return keep, diagnostics


def _refined_ray_exclusion(
    source_excluded: npt.ArrayLike, refined_shape: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    source = np.asarray(source_excluded, dtype=bool)
    expected_shape = tuple(2 * (size - 1) + 1 for size in source.shape)
    if refined_shape != expected_shape:
        raise ValueError(f"refined ray grid must have shape {expected_shape}, got {refined_shape}")
    excluded = np.zeros(refined_shape, dtype=bool)
    excluded[::2, ::2] = source
    return binary_dilation(excluded, structure=np.ones((3, 3), dtype=bool))


def medial_x_graph(
    sampled: npt.ArrayLike,
    index_scale_zyx: npt.ArrayLike,
    spacing_zyx_um: npt.ArrayLike,
    origin_zyx_um: npt.ArrayLike,
    *,
    split_x_index: int,
    half: Literal["low-x", "high-x"],
    excluded_rays: npt.ArrayLike,
    level: float = 0.0,
) -> tuple[FloatArray, IntArray, dict[str, int]]:
    """Mesh one medial X face while omitting rays exposed only by the split."""
    field = np.asarray(sampled, dtype=np.float64)
    index_scale = np.asarray(index_scale_zyx, dtype=np.float64)
    excluded = np.asarray(excluded_rays, dtype=bool)
    if excluded.shape != field.shape[:2]:
        raise ValueError("excluded_rays must match the sampled ZY ray grid")
    cut_sample_x = (split_x_index - 0.5) / index_scale[2]
    edge_x = np.arange(field.shape[2] - 1)
    inside = field >= level
    if half == "low-x":
        candidates = inside[:, :, :-1] & ~inside[:, :, 1:] & (edge_x < cut_sample_x)
        valid = candidates.any(axis=2) & ~excluded
        crossing_index = field.shape[2] - 2 - np.argmax(candidates[:, :, ::-1], axis=2)
    elif half == "high-x":
        candidates = ~inside[:, :, :-1] & inside[:, :, 1:] & (edge_x >= cut_sample_x)
        valid = candidates.any(axis=2) & ~excluded
        crossing_index = np.argmax(candidates, axis=2)
    else:
        raise ValueError(f"unknown half: {half}")
    z_index, y_index = np.where(valid)
    if len(z_index) == 0:
        raise ValueError(f"{half} has no medial X crossings after excluding the cut cap")
    x0 = crossing_index[z_index, y_index]
    value0 = field[z_index, y_index, x0]
    value1 = field[z_index, y_index, x0 + 1]
    x_crossing = x0 + (level - value0) / (value1 - value0)
    index_zyx = np.column_stack((z_index, y_index, x_crossing))
    vertices = (index_zyx * index_scale * np.asarray(spacing_zyx_um) + np.asarray(origin_zyx_um))[:, ::-1]
    faces = _graph_faces(valid)
    if not len(faces):
        raise ValueError(f"{half} medial surface contains no triangles")
    if half == "high-x":
        faces = faces[:, [0, 2, 1]]
    return (
        vertices,
        faces,
        {
            "candidate_refined_rays": int(np.count_nonzero(candidates.any(axis=2))),
            "excluded_refined_rays": int(np.count_nonzero(excluded)),
            "retained_refined_rays": int(np.count_nonzero(valid)),
        },
    )


def _write_ply(path: Path, vertices: FloatArray, faces: IntArray) -> None:
    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(f"element face {len(faces)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        np.savetxt(stream, vertices, fmt="%.6f %.6f %.6f")
        np.savetxt(stream, faces, fmt="3 %d %d %d")


def _validate_surface(
    vertices: FloatArray,
    faces: IntArray,
    label: str,
    *,
    anchor_vertices: FloatArray | None = None,
) -> tuple[dict[str, int], FloatArray]:
    try:
        return validate_manifold_disk(vertices, faces, anchor_vertices_xyz_um=anchor_vertices)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error


def _keep_largest_face_component(
    vertices: FloatArray, faces: IntArray
) -> tuple[FloatArray, IntArray, IntArray, dict[str, int]]:
    component_count, face_component = igl.facet_components(faces)
    counts = np.bincount(face_component, minlength=component_count)
    retained_component = int(np.argmax(counts))
    retained_faces = faces[face_component == retained_component]
    retained_vertex_index = np.unique(retained_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[retained_vertex_index] = np.arange(len(retained_vertex_index))
    return (
        vertices[retained_vertex_index],
        remap[retained_faces],
        retained_vertex_index,
        {
            "source_face_components": int(component_count),
            "removed_vertices": int(len(vertices) - len(retained_vertex_index)),
            "removed_faces": int(len(faces) - len(retained_faces)),
        },
    )


def _exclude_long_edge_faces(
    vertices: FloatArray,
    faces: IntArray,
    max_face_edge_um: float | None,
) -> tuple[IntArray, dict[str, float | int | None]]:
    if max_face_edge_um is None:
        return faces, {"max_face_edge_um": None, "excluded_long_edge_faces": 0}
    if not np.isfinite(max_face_edge_um) or max_face_edge_um <= 0:
        raise ValueError("max_face_edge_um must be finite and positive")

    triangles = vertices[faces]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
        ),
        axis=1,
    )
    retained = faces[edge_lengths.max(axis=1) <= max_face_edge_um]
    if not len(retained):
        raise ValueError("max_face_edge_um excludes every surface face")
    return retained, {
        "max_face_edge_um": float(max_face_edge_um),
        "excluded_long_edge_faces": int(len(faces) - len(retained)),
    }


def _write_surface_bundle(
    output_dir: Path,
    vertices_world: FloatArray,
    vertices_intrinsic: FloatArray,
    faces: IntArray,
    vertex_uv: FloatArray,
    geometry: ImageGeometry,
    source_geometry: ImageGeometry,
    cavity_direction_world: FloatArray,
) -> dict[str, str]:
    surface_path = output_dir / "surface.npz"
    chart_path = output_dir / "chart.npz"
    np.savez_compressed(
        surface_path,
        vertices_xyz_um=vertices_world,
        vertices_world_xyz_um=vertices_world,
        vertices_intrinsic_xyz_um=vertices_intrinsic,
        faces=faces,
        image_origin_xyz_um=geometry.origin_xyz_um,
        image_spacing_xyz_um=geometry.spacing_xyz_um,
        image_direction=geometry.direction,
        original_image_origin_xyz_um=source_geometry.origin_xyz_um,
        original_image_spacing_xyz_um=source_geometry.spacing_xyz_um,
        original_image_direction=source_geometry.direction,
        cavity_direction_world_xyz=cavity_direction_world,
    )
    _write_ply(output_dir / "surface.ply", vertices_world, faces)
    surface_hash = sha256_file(surface_path)
    np.savez_compressed(
        chart_path,
        vertex_uv=vertex_uv,
        mesh_sha256=np.asarray(surface_hash),
        uv_units=np.asarray("dimensionless"),
        uv_orientation=np.asarray("geometric, anchored in intrinsic XYZ; no anatomical orientation"),
    )
    return {
        "surface.npz": surface_hash,
        "surface.ply": sha256_file(output_dir / "surface.ply"),
        "chart.npz": sha256_file(chart_path),
    }


def _binary_ray_counts(mask: npt.ArrayLike) -> npt.NDArray[np.int64]:
    values = np.asarray(mask, dtype=bool)
    return np.count_nonzero(
        np.diff(np.pad(values.astype(np.int8), ((0, 0), (0, 0), (1, 1))), axis=2) == 1,
        axis=2,
    )


def _stride_mask(
    mask: npt.NDArray,
    geometry: ImageGeometry,
    stride_zyx: tuple[int, int, int],
) -> tuple[npt.NDArray, ImageGeometry, npt.NDArray[np.int64]]:
    stride = np.asarray(stride_zyx)
    if stride.shape != (3,) or not np.issubdtype(stride.dtype, np.integer) or np.any(stride < 1):
        raise ValueError("stride_zyx must contain three positive integers")
    stride = stride.astype(np.int64, copy=False)
    return (
        mask[:: stride[0], :: stride[1], :: stride[2]],
        geometry.strided(stride),
        stride,
    )


def _orient_faces(
    vertices_world: FloatArray,
    faces: IntArray,
    cavity_direction_world: FloatArray,
) -> IntArray:
    triangles = vertices_world[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    signed = face_normals @ cavity_direction_world
    if np.all(signed < 0):
        return np.ascontiguousarray(faces[:, [0, 2, 1]])
    if not np.all(signed > 0):
        raise ValueError("surface faces do not consistently face the cavity direction")
    return faces


def build_ventricular_uv_surface(
    *,
    mask_path: Path,
    output_dir: Path,
    smoothing_um: float = DEFAULT_SMOOTHING_UM,
    stride_zyx: tuple[int, int, int] = (1, 1, 1),
    allow_multi_interval_rays: bool = False,
) -> Path:
    """Build and atomically publish a +X ventricular mesh and its UV chart."""
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)
    if not np.isfinite(smoothing_um) or smoothing_um < 0:
        raise ValueError("smoothing_um must be finite and nonnegative")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    image = sitk.ReadImage(mask_path)
    if image.GetDimension() != 3:
        raise ValueError("mask must be a 3-D image")
    original_mask = sitk.GetArrayFromImage(image)
    if not np.array_equal(np.unique(original_mask), np.array([0, 1], dtype=original_mask.dtype)):
        raise ValueError("mask must contain both binary values 0 and 1")
    source_geometry = ImageGeometry.from_sitk(image)
    mask, geometry, stride = _stride_mask(original_mask, source_geometry, stride_zyx)
    if len(np.unique(mask)) != 2:
        raise ValueError("striding must retain both binary mask values 0 and 1")
    original_runs = _binary_ray_counts(original_mask)
    strided_runs = _binary_ray_counts(mask)
    if not allow_multi_interval_rays and (np.any(original_runs > 1) or np.any(strided_runs > 1)):
        raise ValueError(
            "mask is not X-monotone; +X visibility does not define one face "
            "(set allow_multi_interval_rays=True to select the outermost crossing)"
        )
    spacing_zyx_um = geometry.spacing_xyz_um[::-1]

    signed_distance = distance_transform_edt(mask, sampling=spacing_zyx_um)
    signed_distance -= distance_transform_edt(~mask.astype(bool), sampling=spacing_zyx_um)
    if smoothing_um > 0:
        signed_distance = gaussian_filter(
            signed_distance, sigma=smoothing_um / spacing_zyx_um, mode="nearest"
        )
    sampled, index_scale = cubic_resample(signed_distance)
    vertices_intrinsic, faces, ray_diagnostics = positive_x_graph(
        sampled, index_scale, spacing_zyx_um, np.zeros(3)
    )
    vertices_world = geometry.intrinsic_to_world(vertices_intrinsic)
    cavity_direction_world = geometry.direction[:, 0]
    faces = _orient_faces(vertices_world, faces, cavity_direction_world)
    topology, vertex_uv = _validate_surface(
        vertices_world, faces, "candidate", anchor_vertices=vertices_intrinsic
    )

    with atomic_output_directory(output_dir) as stage:
        outputs = _write_surface_bundle(
            stage,
            vertices_world,
            vertices_intrinsic,
            faces,
            vertex_uv,
            geometry,
            source_geometry,
            cavity_direction_world,
        )
        metadata = {
            "artifact_type": ARTIFACT_TYPE,
            "mask": str(mask_path.resolve()),
            "mask_sha256": sha256_file(mask_path),
            "selection": "outermost +X signed-distance zero crossing on each nonempty ZY ray",
            "coordinate_frame": "world physical XYZ",
            "intrinsic_coordinate_frame": "index XYZ times spacing; origin excluded",
            "mask_shape_zyx": list(original_mask.shape),
            "strided_mask_shape_zyx": list(mask.shape),
            "stride_zyx": stride.tolist(),
            "mask_spacing_zyx_um": source_geometry.spacing_xyz_um[::-1].tolist(),
            "strided_mask_spacing_zyx_um": spacing_zyx_um.tolist(),
            "mask_origin_xyz_um": geometry.origin_xyz_um.tolist(),
            "mask_direction": geometry.direction.tolist(),
            "direction_determinant": geometry.determinant,
            "cavity_direction_world_xyz": cavity_direction_world.tolist(),
            "signed_distance_smoothing_um": smoothing_um,
            "resampling": "scipy cubic spline, order 3, endpoint-aligned half grid",
            "surface_chart": "Tutte embedding with convex length-spaced boundary and uniform positive weights",
            "uv_units": "dimensionless",
            "uv_orientation": "geometric, anchored in intrinsic XYZ; no anatomical orientation",
            "allow_multi_interval_rays": allow_multi_interval_rays,
            "original_binary_x_rays": int(original_runs.size),
            "original_binary_nonempty_x_rays": int(np.count_nonzero(original_runs)),
            "original_binary_multi_interval_x_rays": int(np.count_nonzero(original_runs > 1)),
            "strided_binary_x_rays": int(strided_runs.size),
            "strided_binary_nonempty_x_rays": int(np.count_nonzero(strided_runs)),
            "strided_binary_multi_interval_x_rays": int(np.count_nonzero(strided_runs > 1)),
            "binary_nonempty_x_rays": int(np.count_nonzero(strided_runs)),
            "binary_multi_interval_x_rays": int(np.count_nonzero(strided_runs > 1)),
            **ray_diagnostics,
            "topology": topology,
            "library_versions": {
                name: importlib.metadata.version(name) for name in ("libigl", "numpy", "scipy", "SimpleITK")
            },
            "outputs": outputs,
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return (output_dir / "metadata.json").resolve()


def build_bilateral_ventricular_uv_surfaces(
    *,
    mask_path: Path,
    output_dir: Path,
    split_x_index: int,
    smoothing_um: float = DEFAULT_SMOOTHING_UM,
    keep_main_sheet: bool = False,
    main_sheet_min_z_index: int | None = None,
    max_face_edge_um: float | None = None,
) -> Path:
    """Build medial UV surfaces for two hemispheres separated along array X."""
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)
    if not np.isfinite(smoothing_um) or smoothing_um < 0:
        raise ValueError("smoothing_um must be finite and nonnegative")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    image = sitk.ReadImage(mask_path)
    if image.GetDimension() != 3:
        raise ValueError("mask must be a 3-D image")
    mask = sitk.GetArrayFromImage(image)
    if not np.array_equal(np.unique(mask), np.array([0, 1], dtype=mask.dtype)):
        raise ValueError("mask must contain both binary values 0 and 1")
    if not 1 <= split_x_index < mask.shape[2]:
        raise ValueError(f"split_x_index must be in [1, {mask.shape[2] - 1}]")
    if main_sheet_min_z_index is not None and not keep_main_sheet:
        raise ValueError("main_sheet_min_z_index requires keep_main_sheet=True")
    if max_face_edge_um is not None and (not np.isfinite(max_face_edge_um) or max_face_edge_um <= 0):
        raise ValueError("max_face_edge_um must be finite and positive")

    geometry = ImageGeometry.from_sitk(image)
    spacing_zyx_um = geometry.spacing_xyz_um[::-1]
    signed_distance = distance_transform_edt(mask, sampling=spacing_zyx_um)
    signed_distance -= distance_transform_edt(~mask.astype(bool), sampling=spacing_zyx_um)
    if smoothing_um > 0:
        signed_distance = gaussian_filter(
            signed_distance, sigma=smoothing_um / spacing_zyx_um, mode="nearest"
        )
    sampled, index_scale = cubic_resample(signed_distance)

    cut_crossing = mask[:, :, split_x_index - 1].astype(bool) & mask[:, :, split_x_index].astype(bool)
    cut_cap_excluded = _refined_ray_exclusion(cut_crossing, sampled.shape[:2])
    main_sheet_diagnostics: dict[str, object] = {"main_sheet_policy": None}
    if keep_main_sheet:
        source_keep, bridge_diagnostics = _main_sheet_source_keep(
            cut_crossing, min_z_index=main_sheet_min_z_index
        )
        main_sheet_excluded = _refined_ray_exclusion(~source_keep, sampled.shape[:2])
        excluded_rays = cut_cap_excluded | main_sheet_excluded
        main_sheet_diagnostics = {
            "main_sheet_policy": (
                "keep Y before the cut bridge and Z above its per-Y upper envelope; "
                "exclude Y after the bridge"
            ),
            **bridge_diagnostics,
            "excluded_main_sheet_refined_rays": int(np.count_nonzero(main_sheet_excluded)),
        }
    else:
        excluded_rays = cut_cap_excluded

    halves: dict[str, dict[str, object]] = {}
    half_arrays: dict[str, tuple[FloatArray, FloatArray, IntArray, FloatArray, FloatArray]] = {}
    world_x_direction = geometry.direction[0, 0]
    if world_x_direction > 1e-10:
        hemisphere_by_half = {"low-x": "right", "high-x": "left"}
    elif world_x_direction < -1e-10:
        hemisphere_by_half = {"low-x": "left", "high-x": "right"}
    else:
        hemisphere_by_half = {"low-x": "undetermined", "high-x": "undetermined"}
    for half in ("low-x", "high-x"):
        vertices_intrinsic, faces, diagnostics = medial_x_graph(
            sampled,
            index_scale,
            spacing_zyx_um,
            np.zeros(3),
            split_x_index=split_x_index,
            half=half,
            excluded_rays=excluded_rays,
        )
        vertices_world = geometry.intrinsic_to_world(vertices_intrinsic)
        cavity_direction_world = geometry.direction[:, 0] * (1 if half == "low-x" else -1)
        faces = _orient_faces(vertices_world, faces, cavity_direction_world)
        faces, edge_diagnostics = _exclude_long_edge_faces(vertices_world, faces, max_face_edge_um)
        retained_world, faces, retained_vertex_index, component_diagnostics = _keep_largest_face_component(
            vertices_world, faces
        )
        retained_intrinsic = vertices_intrinsic[retained_vertex_index]
        topology, vertex_uv = _validate_surface(
            retained_world,
            faces,
            f"{half} candidate",
            anchor_vertices=retained_intrinsic,
        )
        half_arrays[half] = (
            retained_world,
            retained_intrinsic,
            faces,
            vertex_uv,
            cavity_direction_world,
        )
        halves[half] = {
            "anatomical_hemisphere_lps": hemisphere_by_half[half],
            "medial_face": "+X" if half == "low-x" else "-X",
            **diagnostics,
            **edge_diagnostics,
            **component_diagnostics,
            "topology": topology,
        }

    with atomic_output_directory(output_dir) as stage:
        for half, (
            vertices_world,
            vertices_intrinsic,
            faces,
            vertex_uv,
            cavity_direction,
        ) in half_arrays.items():
            (stage / half).mkdir()
            halves[half]["outputs"] = _write_surface_bundle(
                stage / half,
                vertices_world,
                vertices_intrinsic,
                faces,
                vertex_uv,
                geometry,
                geometry,
                cavity_direction,
            )
        metadata = {
            "artifact_type": "squisher_lightsheet.bilateral_ventricular_uv_surfaces.v1",
            "mask": str(mask_path.resolve()),
            "mask_sha256": sha256_file(mask_path),
            "split_axis": "array X (axis 2 in ZYX)",
            "split_x_index": split_x_index,
            "split_definition": f"low-x: X < {split_x_index}; high-x: X >= {split_x_index}",
            "split_x_intrinsic_um": float((split_x_index - 0.5) * geometry.spacing_xyz_um[0]),
            "split_plane_world_xyz_um": geometry.intrinsic_to_world(
                np.asarray([(split_x_index - 0.5) * geometry.spacing_xyz_um[0], 0.0, 0.0])
            ).tolist(),
            "cut_cap_policy": "exclude refined ZY rays neighboring original rays occupied on both sides of the split",
            "original_cut_crossing_rays": int(np.count_nonzero(cut_crossing)),
            "excluded_cut_cap_refined_rays": int(np.count_nonzero(cut_cap_excluded)),
            **main_sheet_diagnostics,
            "signed_distance_smoothing_um": smoothing_um,
            "mask_shape_zyx": list(mask.shape),
            "mask_spacing_zyx_um": spacing_zyx_um.tolist(),
            "mask_origin_xyz_um": geometry.origin_xyz_um.tolist(),
            "mask_direction": geometry.direction.tolist(),
            "direction_determinant": geometry.determinant,
            "surface_chart": "Tutte embedding with convex length-spaced boundary and uniform positive weights",
            "uv_units": "dimensionless",
            "uv_orientation": "geometric, anchored in intrinsic XYZ; no anatomical orientation",
            "halves": halves,
        }
        (stage / "bilateral.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return (output_dir / "bilateral.json").resolve()
