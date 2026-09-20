from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import igl
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import spsolve

from squisher_lightsheet.artifact_io import atomic_output_directory, sha256_file


ARTIFACT_TYPE = "squisher_lightsheet.surface_boundaries.v1"
DEFAULT_INNER_FRACTION = 0.60
DEFAULT_INNER_SMOOTH_UM = 30.0
DEFAULT_OUTER_FRACTION = 0.75
DEFAULT_OUTER_SMOOTH_UM = 80.0

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def peak_fraction_crossing(
    depth_um: npt.ArrayLike,
    signal: npt.ArrayLike,
    valid_count: npt.ArrayLike,
    *,
    fraction: float,
    side: Literal["rising", "falling"],
) -> float:
    """Interpolate the first observed fraction crossing away from the raw peak."""
    depth = np.asarray(depth_um, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    counts = np.asarray(valid_count)
    if depth.ndim != 1 or values.shape != depth.shape or counts.shape != depth.shape:
        raise ValueError("depth_um, signal, and valid_count must be aligned 1-D arrays")
    if len(depth) < 2 or not np.isfinite(depth).all() or np.any(np.diff(depth) <= 0):
        raise ValueError("depth_um must be finite and strictly increasing")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be finite in (0, 1]")
    if side not in ("rising", "falling"):
        raise ValueError("side must be rising or falling")

    valid = counts > 0
    if not valid.any():
        return float("nan")
    if not np.isfinite(values[valid]).all():
        raise ValueError("observed intensities must be finite")
    peak = int(np.argmax(np.where(valid, values, -np.inf)))
    if values[peak] <= 0:
        return float("nan")
    if fraction == 1:
        neighbor = peak - 1 if side == "rising" else peak + 1
        if neighbor < 0 or neighbor >= len(values) or not valid[neighbor]:
            return float("nan")
        return float(depth[peak])

    threshold = values[peak] * fraction
    step = -1 if side == "rising" else 1
    stop = -1 if side == "rising" else len(values)
    for index in range(peak + step, stop, step):
        if not valid[index]:
            return float("nan")
        if values[index] <= threshold:
            toward_peak = index - step
            return float(
                np.interp(
                    threshold,
                    [values[index], values[toward_peak]],
                    [depth[index], depth[toward_peak]],
                )
            )
    return float("nan")


def measure_peak_boundaries(
    depth_um: npt.ArrayLike,
    intensity: npt.ArrayLike,
    valid_count: npt.ArrayLike,
    *,
    inner_fraction: float = DEFAULT_INNER_FRACTION,
    outer_fraction: float = DEFAULT_OUTER_FRACTION,
) -> tuple[FloatArray, FloatArray, BoolArray, BoolArray, BoolArray]:
    """Measure rising inner and falling outer Eomes crossings for every profile."""
    values = np.asarray(intensity, dtype=np.float64)
    counts = np.asarray(valid_count)
    if values.ndim != 2 or counts.shape != values.shape:
        raise ValueError("intensity and valid_count must be aligned 2-D arrays")
    depth = np.asarray(depth_um, dtype=np.float64)
    if values.shape[1:] != depth.shape:
        raise ValueError("profile columns must align with depth_um")

    raw_inner = np.asarray(
        [
            peak_fraction_crossing(depth, signal, row_counts, fraction=inner_fraction, side="rising")
            for signal, row_counts in zip(values, counts, strict=True)
        ],
        dtype=np.float64,
    )
    raw_outer = np.asarray(
        [
            peak_fraction_crossing(depth, signal, row_counts, fraction=outer_fraction, side="falling")
            for signal, row_counts in zip(values, counts, strict=True)
        ],
        dtype=np.float64,
    )
    direct_inner = np.isfinite(raw_inner)
    direct_outer = np.isfinite(raw_outer)
    return raw_inner, raw_outer, direct_inner, direct_outer, direct_inner & direct_outer


def _mesh_operators(
    vertices_xyz_um: npt.ArrayLike, faces: npt.ArrayLike
) -> tuple[FloatArray, sparse.csr_matrix]:
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices_xyz_um must be finite with shape (n_vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("faces must be a nonempty array with shape (n_faces, 3)")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("faces contain an out-of-range vertex index")
    corners = vertices[triangles]
    twice_area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    if np.any(twice_area <= 0) or not np.isfinite(twice_area).all():
        raise ValueError("mesh faces must be finite and nondegenerate")
    mass = np.asarray(
        igl.massmatrix(vertices, triangles, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal(),
        dtype=np.float64,
    )
    stiffness = (-igl.cotmatrix(vertices, triangles)).tocsr()
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("mesh vertices must have finite positive lumped mass")
    return mass, stiffness


def _log_fit_system(
    vertices_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    raw_depth_um: npt.ArrayLike,
    direct_supported: npt.ArrayLike,
    *,
    smooth_um: float,
) -> tuple[sparse.csc_matrix, FloatArray]:
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    raw = np.asarray(raw_depth_um, dtype=np.float64)
    supported = np.asarray(direct_supported, dtype=bool)
    if raw.shape != (len(vertices),) or supported.shape != raw.shape:
        raise ValueError("raw depths and direct support must align with mesh vertices")
    if not np.isfinite(smooth_um) or smooth_um <= 0:
        raise ValueError("smooth_um must be finite and positive")
    if np.any(~np.isfinite(raw[supported])):
        raise ValueError("directly supported raw depths must be finite")
    if np.any(raw[supported] <= 0):
        raise ValueError("directly supported raw depths must be positive before log fitting")

    mass, stiffness = _mesh_operators(vertices, faces)
    component_count, component = sparse.csgraph.connected_components(stiffness, directed=False)
    missing = [label for label in range(component_count) if not np.any(supported[component == label])]
    if missing:
        raise ValueError("each mesh component requires a directly supported anchor for this boundary solve")
    fidelity_weight = mass * supported
    target = np.zeros(len(vertices), dtype=np.float64)
    target[supported] = np.log(raw[supported])
    bending = stiffness @ sparse.diags(1.0 / mass) @ stiffness
    system = (sparse.diags(fidelity_weight) + smooth_um**4 * bending).tocsc()
    rhs = fidelity_weight * target
    return system, rhs


def fit_log_boundary(
    vertices_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    raw_depth_um: npt.ArrayLike,
    direct_supported: npt.ArrayLike,
    *,
    smooth_um: float,
) -> FloatArray:
    """Fit a positive full-mesh log-depth field from all direct observations."""
    system, rhs = _log_fit_system(
        vertices_xyz_um,
        faces,
        raw_depth_um,
        direct_supported,
        smooth_um=smooth_um,
    )
    solution = np.asarray(spsolve(system, rhs), dtype=np.float64)
    fitted = np.exp(solution)
    if fitted.shape != rhs.shape or not np.isfinite(fitted).all() or np.any(fitted <= 0):
        raise RuntimeError("unconstrained log-depth fit did not produce finite positive values")
    return fitted


def fit_ordered_outer_boundary(
    vertices_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    raw_outer_um: npt.ArrayLike,
    direct_supported_outer: npt.ArrayLike,
    inner_um: npt.ArrayLike,
    *,
    constraint_mask: npt.ArrayLike,
    smooth_um: float,
) -> FloatArray:
    """Fit outer log depth on the full mesh with ordering on selected vertices."""
    import osqp

    system, rhs = _log_fit_system(
        vertices_xyz_um,
        faces,
        raw_outer_um,
        direct_supported_outer,
        smooth_um=smooth_um,
    )
    inner = np.asarray(inner_um, dtype=np.float64)
    if inner.shape != rhs.shape or not np.isfinite(inner).all() or np.any(inner <= 0):
        raise ValueError("inner_um must be finite, positive, and align with mesh vertices")
    constrained = np.asarray(constraint_mask, dtype=bool)
    if constrained.shape != inner.shape:
        raise ValueError("constraint_mask must align with mesh vertices")
    if not constrained.any():
        raise ValueError("constraint_mask must select at least one mesh vertex")
    lower = np.full(len(inner), -np.inf, dtype=np.float64)
    lower[constrained] = np.log(inner[constrained])
    unconstrained = np.asarray(spsolve(system, rhs), dtype=np.float64)
    scale = float(np.median(system.diagonal()))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("ordered boundary system has an invalid diagonal scale")
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.triu(system / scale, format="csc"),
        q=-rhs / scale,
        A=sparse.eye(len(inner), format="csc"),
        l=lower,
        u=np.full(len(inner), np.inf),
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=100_000,
        polishing=True,
        verbose=False,
    )
    solver.warm_start(x=unconstrained)
    result = solver.solve(raise_error=True)
    if result.info.status.lower() != "solved":
        raise RuntimeError(f"ordered outer boundary fit failed: {result.info.status}")
    solution = np.asarray(result.x, dtype=np.float64)
    violation = lower[constrained] - solution[constrained]
    if np.any(violation > 1e-8):
        raise RuntimeError("ordered outer fit exceeds numerical feasibility tolerance")
    solution[constrained] = np.maximum(solution[constrained], lower[constrained])
    active = constrained & (np.abs(solution - lower) <= 1e-8)
    fitted = np.exp(solution)
    fitted[constrained] = np.maximum(fitted[constrained], inner[constrained])
    fitted[active] = inner[active]
    if not np.isfinite(fitted).all() or np.any(fitted[constrained] < inner[constrained]):
        raise RuntimeError("ordered outer fit did not produce finite ordered values")
    return fitted


def wall_support(
    vertices_intrinsic_xyz_um: npt.ArrayLike,
    *,
    cavity_x_sign: Literal[-1, 1] = 1,
) -> tuple[BoolArray, FloatArray]:
    """Retain the intrinsic concave wall between the two X turnarounds on each Z row."""
    vertices = np.asarray(vertices_intrinsic_xyz_um, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices_intrinsic_xyz_um must be finite with shape (n_vertices, 3)")
    if cavity_x_sign not in (-1, 1):
        raise ValueError("cavity_x_sign must be -1 or 1")
    keep = np.zeros(len(vertices), dtype=bool)
    bounds: list[list[float]] = []
    for z in np.unique(vertices[:, 2]):
        row = np.flatnonzero(vertices[:, 2] == z)
        row = row[np.argsort(vertices[row, 1])]
        if np.any(np.diff(vertices[row, 1]) <= 0):
            raise ValueError("intrinsic surface must have unique increasing Y coordinates per Z row")
        x = cavity_x_sign * vertices[row, 0]
        valley = int(np.argmin(x))
        start = int(np.argmax(x[: valley + 1]))
        stop = valley + int(np.argmax(x[valley:]))
        keep[row[start : stop + 1]] = True
        bounds.append([float(z), float(vertices[row[start], 1]), float(vertices[row[stop], 1])])
    return keep, np.asarray(bounds, dtype=np.float64).reshape(-1, 3)


def build_surface_boundaries(
    *,
    profiles_path: Path,
    surface_path: Path,
    output_dir: Path,
    inner_fraction: float = DEFAULT_INNER_FRACTION,
    inner_smooth_um: float = DEFAULT_INNER_SMOOTH_UM,
    outer_fraction: float = DEFAULT_OUTER_FRACTION,
    outer_smooth_um: float = DEFAULT_OUTER_SMOOTH_UM,
) -> Path:
    """Measure and fit ordered Eomes boundaries into a fresh atomic artifact directory."""
    profiles_path = Path(profiles_path)
    surface_path = Path(surface_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    with np.load(surface_path) as surface:
        required = {"vertices_xyz_um", "vertices_intrinsic_xyz_um", "faces"}
        missing = required.difference(surface.files)
        if missing:
            raise ValueError(f"surface artifact is missing arrays: {sorted(missing)}")
        vertices = np.asarray(surface["vertices_xyz_um"], dtype=np.float64)
        intrinsic_vertices = np.asarray(surface["vertices_intrinsic_xyz_um"], dtype=np.float64)
        faces = np.asarray(surface["faces"], dtype=np.int64)
        orientation_fields = {"image_direction", "cavity_direction_world_xyz"}
        present_orientation_fields = orientation_fields.intersection(surface.files)
        if present_orientation_fields and present_orientation_fields != orientation_fields:
            raise ValueError("surface orientation requires image_direction and cavity_direction_world_xyz")
        if present_orientation_fields:
            image_direction = np.asarray(surface["image_direction"], dtype=np.float64)
            cavity_direction = np.asarray(surface["cavity_direction_world_xyz"], dtype=np.float64)
            intrinsic_cavity = image_direction.T @ cavity_direction
            if not np.allclose(np.abs(intrinsic_cavity), [1.0, 0.0, 0.0], atol=1e-10, rtol=1e-10):
                raise ValueError("surface cavity direction must align with intrinsic X")
            cavity_x_sign: Literal[-1, 1] = 1 if intrinsic_cavity[0] > 0 else -1
        else:
            cavity_x_sign = 1
    if intrinsic_vertices.shape != vertices.shape:
        raise ValueError("intrinsic and world surface vertices must have matching shapes")

    surface_hash = sha256_file(surface_path)
    profiles_metadata_path = profiles_path.parent / "metadata.json"
    if not profiles_metadata_path.is_file():
        raise ValueError(f"profile metadata is missing: {profiles_metadata_path}")
    profiles_metadata = json.loads(profiles_metadata_path.read_text())
    if not isinstance(profiles_metadata, dict):
        raise ValueError("profile metadata must contain a JSON object")
    if profiles_metadata.get("artifact_type") != "squisher_lightsheet.surface_profiles.v1":
        raise ValueError("profile metadata has an unsupported artifact_type")
    if profiles_metadata.get("surface_sha256") != surface_hash:
        raise ValueError("profile metadata surface_sha256 does not match the supplied surface")
    profile_hash = sha256_file(profiles_path)
    if profiles_metadata.get("output_sha256") != profile_hash:
        raise ValueError("profile metadata does not bind the supplied profiles.npz")

    with np.load(profiles_path) as profiles:
        required = {"depth_um", "intensity", "valid_count", "vertex_index"}
        missing = required.difference(profiles.files)
        if missing:
            raise ValueError(f"profile artifact is missing arrays: {sorted(missing)}")
        depth = np.asarray(profiles["depth_um"], dtype=np.float64)
        intensity = np.asarray(profiles["intensity"], dtype=np.float64)
        counts = np.asarray(profiles["valid_count"])
        vertex_index = np.asarray(profiles["vertex_index"], dtype=np.int64)
    if vertex_index.shape != (len(vertices),) or not np.array_equal(
        np.sort(vertex_index), np.arange(len(vertices), dtype=np.int64)
    ):
        raise ValueError("profiles must contain every surface vertex exactly once")
    if intensity.shape[0] != len(vertex_index):
        raise ValueError("profile rows must align with vertex_index")

    row_inner, row_outer, row_direct_inner, row_direct_outer, _ = measure_peak_boundaries(
        depth,
        intensity,
        counts,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
    )
    vertex_order = np.argsort(vertex_index)
    raw_inner = row_inner[vertex_order]
    raw_outer = row_outer[vertex_order]
    direct_inner = row_direct_inner[vertex_order]
    direct_outer = row_direct_outer[vertex_order]
    direct_joint = direct_inner & direct_outer

    inner = fit_log_boundary(vertices, faces, raw_inner, direct_inner, smooth_um=inner_smooth_um)
    keep, bounds = wall_support(intrinsic_vertices, cavity_x_sign=cavity_x_sign)
    outer = fit_ordered_outer_boundary(
        vertices,
        faces,
        raw_outer,
        direct_outer,
        inner,
        constraint_mask=keep,
        smooth_um=outer_smooth_um,
    )
    inner[~keep] = np.nan
    outer[~keep] = np.nan
    estimated_inner = np.isfinite(inner)
    estimated_outer = np.isfinite(outer)
    recovered_inner = estimated_inner & ~direct_inner
    recovered_outer = estimated_outer & ~direct_outer

    with atomic_output_directory(output_dir) as stage:
        boundaries_path = stage / "boundaries.npz"
        np.savez_compressed(
            boundaries_path,
            raw_inner_um=raw_inner,
            raw_outer_um=raw_outer,
            inner_um=inner,
            outer_um=outer,
            direct_supported_inner=direct_inner,
            direct_supported_outer=direct_outer,
            direct_supported_joint=direct_joint,
            wall_support=keep,
            estimated_inner=estimated_inner,
            estimated_outer=estimated_outer,
            recovered_missing_measurement_inner=recovered_inner,
            recovered_missing_measurement_outer=recovered_outer,
        )
        metadata = {
            "artifact_type": ARTIFACT_TYPE,
            "profiles": str(profiles_path.resolve()),
            "profiles_sha256": profile_hash,
            "profiles_metadata": str(profiles_metadata_path.resolve()),
            "profiles_metadata_sha256": sha256_file(profiles_metadata_path),
            "surface": str(surface_path.resolve()),
            "surface_sha256": surface_hash,
            "inner": {
                "peak_side": "rising",
                "peak_fraction": inner_fraction,
                "smooth_um": inner_smooth_um,
                "direct_supported_vertices": int(direct_inner.sum()),
                "estimated_vertices": int(estimated_inner.sum()),
                "recovered_missing_measurement_vertices": int(recovered_inner.sum()),
            },
            "outer": {
                "peak_side": "falling",
                "peak_fraction": outer_fraction,
                "smooth_um": outer_smooth_um,
                "direct_supported_vertices": int(direct_outer.sum()),
                "estimated_vertices": int(estimated_outer.sum()),
                "recovered_missing_measurement_vertices": int(recovered_outer.sum()),
            },
            "direct_supported_joint_vertices": int(direct_joint.sum()),
            "fit": "full-mesh log-depth biharmonic FEM; direct observations supply fidelity",
            "ordering": (
                "outer log depth constrained >= fitted inner log depth on wall_support with OSQP; "
                "cap vertices unconstrained"
            ),
            "equal_inner_outer_vertices": int(
                np.count_nonzero(estimated_inner & estimated_outer & (outer == inner))
            ),
            "vertices": len(vertices),
            "wall_support_vertices": int(keep.sum()),
            "trimmed_cap_vertices": int((~keep).sum()),
            "cap_trimming_coordinates": "vertices_intrinsic_xyz_um",
            "cavity_intrinsic_x_sign": cavity_x_sign,
            "terminal_bounds_intrinsic_z_ymin_ymax_um": bounds.tolist(),
            "raw_measurement_policy": "retained unchanged, including outside wall support",
            "outputs": {"boundaries.npz": sha256_file(boundaries_path)},
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return (output_dir / "metadata.json").resolve()
