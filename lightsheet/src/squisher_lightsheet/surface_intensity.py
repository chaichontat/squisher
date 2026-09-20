from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates
import SimpleITK as sitk

from squisher_lightsheet import ngff
from squisher_lightsheet.artifact_io import atomic_output_directory, sha256_file
from squisher_lightsheet.surface_geometry import ImageGeometry, load_surface


FloatArray = npt.NDArray[np.float64]


def first_inside_interval(inside: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Keep the first contiguous true interval along axis 1 of each ray."""
    values = np.asarray(inside, dtype=bool)
    if values.ndim != 2:
        raise ValueError("inside must be a 2-D ray-by-depth array")
    seen_inside = np.maximum.accumulate(values, axis=1)
    exited = np.maximum.accumulate(seen_inside & ~values, axis=1)
    return values & ~exited


def vertex_normals(vertices_xyz_um: npt.ArrayLike, faces: npt.ArrayLike) -> FloatArray:
    """Return unit area-weighted normals preserving the mesh face orientation."""
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    face_normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("surface contains vertices with undefined normals")
    return normals / lengths[:, None]


def _sample_array(
    values_zyx: npt.ArrayLike,
    points_world_xyz_um: FloatArray,
    geometry: ImageGeometry,
    *,
    cval: float,
) -> npt.NDArray[np.float32]:
    coordinates = geometry.world_to_index_zyx(points_world_xyz_um)
    return map_coordinates(
        np.asarray(values_zyx),
        np.moveaxis(coordinates, -1, 0),
        order=1,
        mode="constant",
        cval=cval,
        prefilter=False,
        output=np.float32,
    )


def project_max_intensity(
    *,
    intensity_zyx: npt.ArrayLike,
    intensity_geometry: ImageGeometry,
    mask_zyx: npt.ArrayLike,
    mask_geometry: ImageGeometry,
    vertices_xyz_um: npt.ArrayLike,
    normals_xyz: npt.ArrayLike,
    depth_um: npt.ArrayLike,
    batch_size: int,
) -> tuple[FloatArray, npt.NDArray[np.uint32], FloatArray, npt.NDArray[np.bool_]]:
    """Max-project each vertex along its negative normal through the first mask interval."""
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    normals = np.asarray(normals_xyz, dtype=np.float64)
    depth = np.asarray(depth_um, dtype=np.float64)
    if vertices.shape != normals.shape or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_xyz_um and normals_xyz must have matching Nx3 shapes")
    if depth.ndim != 1 or not len(depth) or depth[0] != 0 or np.any(np.diff(depth) <= 0):
        raise ValueError("depth_um must be strictly increasing and start at zero")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    projected = np.full(len(vertices), np.nan, dtype=np.float64)
    valid_count = np.zeros(len(vertices), dtype=np.uint32)
    last_depth = np.full(len(vertices), np.nan, dtype=np.float64)
    reaches_cap = np.zeros(len(vertices), dtype=bool)

    for start in range(0, len(vertices), batch_size):
        stop = min(start + batch_size, len(vertices))
        points = vertices[start:stop, None, :] - normals[start:stop, None, :] * depth[None, :, None]
        inside = (
            _sample_array(
                mask_zyx,
                points,
                mask_geometry,
                cval=0.0,
            )
            >= 0.5
        )
        tissue = first_inside_interval(inside)
        intensity = _sample_array(
            intensity_zyx,
            points,
            intensity_geometry,
            cval=np.nan,
        )
        valid = tissue & np.isfinite(intensity)
        count = valid.sum(axis=1)
        maximum = np.max(np.where(valid, intensity, -np.inf), axis=1)
        maximum[count == 0] = np.nan
        final_index = np.max(np.where(valid, np.arange(len(depth)), -1), axis=1)

        projected[start:stop] = maximum
        valid_count[start:stop] = count
        has_sample = final_index >= 0
        last_depth[start:stop][has_sample] = depth[final_index[has_sample]]
        reaches_cap[start:stop] = tissue[:, -1]

    return projected, valid_count, last_depth, reaches_cap


def rasterize_uv(
    vertex_uv: npt.ArrayLike,
    faces: npt.ArrayLike,
    vertex_values: npt.ArrayLike,
    size: int,
) -> npt.NDArray[np.float32]:
    """Linearly interpolate vertex values onto a square raster of the disk chart."""
    if size < 2:
        raise ValueError("UV raster size must be at least 2")
    uv = np.asarray(vertex_uv, dtype=np.float64)
    values = np.asarray(vertex_values, dtype=np.float64)
    if uv.shape != (len(values), 2):
        raise ValueError("vertex_uv must have one 2-D coordinate per vertex value")
    axis = np.linspace(-1.0, 1.0, size)
    grid_u, grid_v = np.meshgrid(axis, axis)
    interpolation = mtri.LinearTriInterpolator(
        mtri.Triangulation(uv[:, 0], uv[:, 1], np.asarray(faces, dtype=np.int64)),
        values,
    )
    return np.ma.filled(interpolation(grid_u, grid_v), np.nan).astype(np.float32)


def _image_geometry(source_zarr: Path, level: int):
    import zarr

    root = zarr.open_group(str(source_zarr), mode="r")
    if root.attrs.get("squisher_complete") is not True:
        raise ValueError(f"source OME-Zarr is not marked complete: {source_zarr}")
    dataset_path = ngff.level_path(root, level=level, context=source_zarr)
    array = root[dataset_path]
    axes = list(ngff.axes(root, array).lower())
    if axes != ["z", "y", "x"]:
        raise ValueError(f"expected a channel-separated ZYX source, found axes {axes}")
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
    if array.ndim != 3 or array.dtype != np.dtype(np.uint16):
        raise ValueError(f"expected a 3-D uint16 source array, found {array.shape} {array.dtype}")
    geometry = ImageGeometry(
        np.asarray(origin, dtype=np.float64)[::-1],
        np.asarray(scale, dtype=np.float64)[::-1],
        np.eye(3),
    )
    return array, dataset_path, geometry


def _write_preview(path: Path, maps: dict[str, npt.NDArray[np.float32]], values: list[FloatArray]) -> None:
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    if not len(finite):
        raise ValueError("surface projection produced no finite intensities")
    lower, upper = np.quantile(finite, [0.01, 0.995])
    if not lower < upper:
        lower, upper = float(finite.min()), float(finite.max())
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
    try:
        for axis, half, title in zip(axes, ("low-x", "high-x"), ("Right", "Left"), strict=True):
            image = axis.imshow(
                maps[half],
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap="magma",
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
            )
            axis.set(title=title, xlabel="surface u", ylabel="surface v")
            axis.set_aspect("equal")
        figure.colorbar(image, ax=axes, label="Maximum intensity (native uint16 units)", shrink=0.82)
        figure.suptitle("Orthogonal tissue-side maximum projection")
        figure.savefig(path, dpi=220)
    finally:
        plt.close(figure)


def build_surface_intensity_map(
    *,
    surface_root: Path,
    mask_path: Path,
    source_zarr: Path,
    output_dir: Path,
    level: int,
    max_depth_um: float,
    depth_step_um: float,
    uv_size: int,
    batch_size: int,
) -> Path:
    """Project one aligned intensity volume through tissue onto two UV surface charts."""
    surface_root = Path(surface_root)
    mask_path = Path(mask_path)
    source_zarr = Path(source_zarr)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if not np.isfinite(max_depth_um) or max_depth_um <= 0:
        raise ValueError("max_depth_um must be finite and positive")
    if not np.isfinite(depth_step_um) or depth_step_um <= 0:
        raise ValueError("depth_step_um must be finite and positive")
    depth = np.arange(int(np.floor(max_depth_um / depth_step_um)) + 1) * depth_step_um

    image = sitk.ReadImage(mask_path)
    if image.GetDimension() != 3:
        raise ValueError("mask must be three-dimensional")
    mask = sitk.GetArrayFromImage(image)
    if not np.array_equal(np.unique(mask), np.array([0, 1], dtype=mask.dtype)):
        raise ValueError("mask must contain both binary values 0 and 1")
    mask_geometry = ImageGeometry.from_sitk(image)

    source_array, dataset_path, source_geometry = _image_geometry(source_zarr, level)
    intensity = np.asarray(source_array[:])
    surface_manifest_path = surface_root / "bilateral.json"
    surface_manifest = json.loads(surface_manifest_path.read_text())
    if surface_manifest.get("artifact_type") != "squisher_lightsheet.bilateral_ventricular_uv_surfaces.v1":
        raise ValueError(f"surface root has an unsupported manifest: {surface_manifest_path}")

    half_results: dict[str, dict[str, object]] = {}
    output_arrays: dict[str, dict[str, npt.NDArray]] = {}
    maps: dict[str, npt.NDArray[np.float32]] = {}
    projected_values: list[FloatArray] = []
    for half in ("low-x", "high-x"):
        surface_path = surface_root / half / "surface.npz"
        chart_path = surface_root / half / "chart.npz"
        surface = load_surface(surface_path)
        vertices = surface.vertices_world_xyz_um
        faces = surface.faces
        with np.load(chart_path) as chart:
            uv = chart["vertex_uv"]
            if str(chart["mesh_sha256"]) != sha256_file(surface_path):
                raise ValueError(f"{half} chart does not belong to its surface mesh")
        normals = vertex_normals(vertices, faces)
        if np.any(normals @ surface.cavity_direction_world_xyz <= 0):
            raise ValueError(f"{half} vertex normals do not face the recorded medial direction")
        projected, valid_count, last_depth, reaches_cap = project_max_intensity(
            intensity_zyx=intensity,
            intensity_geometry=source_geometry,
            mask_zyx=mask,
            mask_geometry=mask_geometry,
            vertices_xyz_um=vertices,
            normals_xyz=normals,
            depth_um=depth,
            batch_size=batch_size,
        )
        if np.any(reaches_cap):
            raise ValueError(
                f"{half} has {np.count_nonzero(reaches_cap)} tissue rays reaching "
                f"the {depth[-1]:g} um sampling cap"
            )
        uv_map = rasterize_uv(uv, faces, projected, uv_size)
        maps[half] = uv_map
        projected_values.append(projected)
        output_arrays[half] = {
            "vertex_intensity_max": projected.astype(np.float32),
            "valid_sample_count": valid_count,
            "last_inside_depth_um": last_depth.astype(np.float32),
            "uv_intensity_max": uv_map,
            "uv_extent": np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
        }
        finite = np.isfinite(projected)
        half_results[half] = {
            "vertices": len(vertices),
            "finite_vertex_values": int(np.count_nonzero(finite)),
            "missing_vertex_values": int(np.count_nonzero(~finite)),
            "valid_samples": int(valid_count.sum()),
            "last_inside_depth_um": {
                "median": float(np.nanmedian(last_depth)),
                "p95": float(np.nanquantile(last_depth, 0.95)),
                "maximum": float(np.nanmax(last_depth)),
            },
        }

    with atomic_output_directory(output_dir) as stage:
        for half, arrays in output_arrays.items():
            (stage / half).mkdir()
            np.savez_compressed(stage / half / "intensity-map.npz", **arrays)
        _write_preview(stage / "intensity-map.png", maps, projected_values)
        for half in half_results:
            half_results[half]["output"] = {
                "intensity-map.npz": sha256_file(stage / half / "intensity-map.npz")
            }
        metadata = {
            "artifact_type": "squisher_lightsheet.surface_intensity_map.v1",
            "surface_root": str(surface_root.resolve()),
            "surface_manifest_sha256": sha256_file(surface_manifest_path),
            "source_zarr": str(source_zarr.resolve()),
            "source_level": level,
            "source_dataset": dataset_path,
            "source_shape_zyx": list(int(value) for value in source_array.shape),
            "source_scale_zyx_um": source_geometry.spacing_xyz_um[::-1].tolist(),
            "source_origin_zyx_um": source_geometry.origin_xyz_um[::-1].tolist(),
            "source_dtype": str(source_array.dtype),
            "source_root_metadata_sha256": sha256_file(source_zarr / "zarr.json"),
            "source_array_metadata_sha256": sha256_file(source_zarr / dataset_path / "zarr.json"),
            "mask": str(mask_path.resolve()),
            "mask_sha256": sha256_file(mask_path),
            "sampling_direction": "negative oriented vertex normal (tissue side)",
            "mask_interval": "first contiguous trilinear mask>=0.5 interval on each ray",
            "reducer": "maximum of trilinearly sampled native intensities",
            "depth_step_um": depth_step_um,
            "maximum_sampled_depth_um": float(depth[-1]),
            "uv_raster_shape": [uv_size, uv_size],
            "uv_interpolation": "piecewise linear on the saved triangle chart",
            "halves": half_results,
            "outputs": {"intensity-map.png": sha256_file(stage / "intensity-map.png")},
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return (output_dir / "metadata.json").resolve()
