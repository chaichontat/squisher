from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import igl
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from squisher_lightsheet.artifact_io import atomic_output_directory, sha256_file
from squisher_lightsheet.surface_geometry import ventricular_wall_support as wall_support


ARTIFACT_TYPE = "squisher_lightsheet.surface_boundaries.v1"
DEFAULT_INNER_FRACTION = 0.60
DEFAULT_INNER_SMOOTH_UM = 30.0
DEFAULT_OUTER_FRACTION = 0.75
DEFAULT_OUTER_SMOOTH_UM = 80.0
DEFAULT_MIN_CROSSING_SAMPLES = 5
DEFAULT_MAX_RECOVERY_DISTANCE_UM = 20.0

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def _peak_fraction_measurement(
    depth_um: npt.ArrayLike,
    signal: npt.ArrayLike,
    valid_count: npt.ArrayLike,
    *,
    fraction: float,
    side: Literal["rising", "falling"],
) -> tuple[float, int]:
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
        return float("nan"), 0
    if not np.isfinite(values[valid]).all():
        raise ValueError("observed intensities must be finite")
    peak = int(np.argmax(np.where(valid, values, -np.inf)))
    if values[peak] <= 0:
        return float("nan"), 0
    if fraction == 1:
        neighbor = peak - 1 if side == "rising" else peak + 1
        if neighbor < 0 or neighbor >= len(values) or not valid[neighbor]:
            return float("nan"), 0
        return float(depth[peak]), int(min(counts[peak], counts[neighbor]))

    threshold = values[peak] * fraction
    step = -1 if side == "rising" else 1
    stop = -1 if side == "rising" else len(values)
    for index in range(peak + step, stop, step):
        if not valid[index]:
            return float("nan"), 0
        if values[index] <= threshold:
            toward_peak = index - step
            crossing = float(
                np.interp(threshold, [values[index], values[toward_peak]], [depth[index], depth[toward_peak]])
            )
            support_count = int(min(counts[peak], counts[index], counts[toward_peak]))
            return crossing, support_count
    return float("nan"), 0


def peak_fraction_crossing(
    depth_um: npt.ArrayLike,
    signal: npt.ArrayLike,
    valid_count: npt.ArrayLike,
    *,
    fraction: float,
    side: Literal["rising", "falling"],
) -> float:
    """Interpolate the first observed fraction crossing away from the raw peak."""
    return _peak_fraction_measurement(depth_um, signal, valid_count, fraction=fraction, side=side)[0]


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

    raw_inner, raw_outer, _, _ = _measure_peak_boundaries_with_quality(
        depth,
        values,
        counts,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
    )
    # Log-domain fitting requires strictly positive depths. A crossing can land
    # exactly on the zero-depth sample, so retain that raw measurement while
    # treating it as unsupported for the fit and recovering it from neighbors.
    direct_inner = np.isfinite(raw_inner) & (raw_inner > 0)
    direct_outer = np.isfinite(raw_outer) & (raw_outer > 0)
    return raw_inner, raw_outer, direct_inner, direct_outer, direct_inner & direct_outer


def _measure_peak_boundaries_with_quality(
    depth_um: npt.ArrayLike,
    intensity: npt.ArrayLike,
    valid_count: npt.ArrayLike,
    *,
    inner_fraction: float,
    outer_fraction: float,
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    depth = np.asarray(depth_um, dtype=np.float64)
    values = np.asarray(intensity, dtype=np.float64)
    counts = np.asarray(valid_count)
    inner = [
        _peak_fraction_measurement(depth, signal, row_counts, fraction=inner_fraction, side="rising")
        for signal, row_counts in zip(values, counts, strict=True)
    ]
    outer = [
        _peak_fraction_measurement(depth, signal, row_counts, fraction=outer_fraction, side="falling")
        for signal, row_counts in zip(values, counts, strict=True)
    ]
    return (
        np.asarray([value for value, _ in inner], dtype=np.float64),
        np.asarray([value for value, _ in outer], dtype=np.float64),
        np.asarray([count for _, count in inner], dtype=np.uint32),
        np.asarray([count for _, count in outer], dtype=np.uint32),
    )


def evidence_recovery_support(
    vertices_xyz_um: npt.ArrayLike,
    faces: npt.ArrayLike,
    reliable: npt.ArrayLike,
    domain: npt.ArrayLike,
    *,
    max_distance_um: float,
) -> tuple[BoolArray, FloatArray]:
    """Bound recovery by mesh-geodesic distance from reliable measurements."""
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    anchors = np.asarray(reliable, dtype=bool)
    allowed = np.asarray(domain, dtype=bool)
    if anchors.shape != (len(vertices),) or allowed.shape != anchors.shape:
        raise ValueError("reliable and domain masks must align with mesh vertices")
    if not np.isfinite(max_distance_um) or max_distance_um <= 0:
        raise ValueError("max_distance_um must be finite and positive")
    domain_faces = triangles[np.all(allowed[triangles], axis=1)]
    if not len(domain_faces) or not np.any(anchors & allowed):
        raise ValueError("recovery domain requires complete faces and a reliable anchor")
    edges = np.sort(
        np.concatenate((domain_faces[:, [0, 1]], domain_faces[:, [1, 2]], domain_faces[:, [2, 0]])),
        axis=1,
    )
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = sparse.csr_matrix(
        (
            np.concatenate((lengths, lengths)),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(len(vertices), len(vertices)),
    )
    distance = np.asarray(
        dijkstra(
            graph,
            directed=False,
            indices=np.flatnonzero(anchors & allowed),
            min_only=True,
            limit=max_distance_um,
        ),
        dtype=np.float64,
    )
    return allowed & (distance <= max_distance_um), distance


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


def build_surface_boundaries(
    *,
    profiles_path: Path,
    surface_path: Path,
    output_dir: Path,
    inner_fraction: float = DEFAULT_INNER_FRACTION,
    inner_smooth_um: float = DEFAULT_INNER_SMOOTH_UM,
    outer_fraction: float = DEFAULT_OUTER_FRACTION,
    outer_smooth_um: float = DEFAULT_OUTER_SMOOTH_UM,
    min_crossing_samples: int = DEFAULT_MIN_CROSSING_SAMPLES,
    max_recovery_distance_um: float = DEFAULT_MAX_RECOVERY_DISTANCE_UM,
    trim_caps: bool = True,
) -> Path:
    """Measure and fit ordered Eomes boundaries into a fresh atomic artifact directory."""
    profiles_path = Path(profiles_path)
    surface_path = Path(surface_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if (
        not isinstance(min_crossing_samples, int)
        or isinstance(min_crossing_samples, bool)
        or min_crossing_samples < 1
    ):
        raise ValueError("min_crossing_samples must be a positive integer")
    if not np.isfinite(max_recovery_distance_um) or max_recovery_distance_um <= 0:
        raise ValueError("max_recovery_distance_um must be finite and positive")

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
        projection_domain = (
            np.asarray(profiles["projection_domain_vertices"])
            if "projection_domain_vertices" in profiles.files
            else np.ones(len(vertices), dtype=bool)
        )
    if vertex_index.shape != (len(vertices),) or not np.array_equal(
        np.sort(vertex_index), np.arange(len(vertices), dtype=np.int64)
    ):
        raise ValueError("profiles must contain every surface vertex exactly once")
    if intensity.shape[0] != len(vertex_index):
        raise ValueError("profile rows must align with vertex_index")
    if projection_domain.shape != (len(vertices),) or projection_domain.dtype != np.bool_:
        raise ValueError("profile projection_domain_vertices must be boolean and align with the surface")
    projection_metadata = profiles_metadata.get("projection_domain")
    projection_digest = hashlib.sha256(projection_domain.astype(np.uint8).tobytes()).hexdigest()
    if projection_metadata is not None and (
        not isinstance(projection_metadata, dict)
        or projection_metadata.get("vertex_mask_sha256") != projection_digest
    ):
        raise ValueError("profile projection-domain metadata does not match its vertex mask")

    row_inner, row_outer, row_inner_count, row_outer_count = _measure_peak_boundaries_with_quality(
        depth,
        intensity,
        counts,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
    )
    vertex_order = np.argsort(vertex_index)
    raw_inner = row_inner[vertex_order]
    raw_outer = row_outer[vertex_order]
    direct_inner = np.isfinite(raw_inner) & (raw_inner > 0)
    direct_outer = np.isfinite(raw_outer) & (raw_outer > 0)
    crossing_count_inner = row_inner_count[vertex_order]
    crossing_count_outer = row_outer_count[vertex_order]
    direct_inner &= projection_domain
    direct_outer &= projection_domain
    direct_joint = direct_inner & direct_outer
    fit_supported_inner = direct_inner & (crossing_count_inner >= min_crossing_samples)
    fit_supported_outer = direct_outer & (crossing_count_outer >= min_crossing_samples)

    inner = fit_log_boundary(vertices, faces, raw_inner, fit_supported_inner, smooth_um=inner_smooth_um)
    if trim_caps:
        if isinstance(projection_metadata, dict) and projection_metadata.get("name") == "ventricular-wall":
            keep = projection_domain.copy()
            bounds = np.empty((0, 3), dtype=np.float64)
        else:
            cap_support, bounds = wall_support(intrinsic_vertices, cavity_x_sign=cavity_x_sign)
            keep = projection_domain & cap_support
    else:
        keep = projection_domain.copy()
        bounds = np.empty((0, 3), dtype=np.float64)
    outer = fit_ordered_outer_boundary(
        vertices,
        faces,
        raw_outer,
        fit_supported_outer,
        inner,
        constraint_mask=keep,
        smooth_um=outer_smooth_um,
    )
    estimated_inner, distance_inner = evidence_recovery_support(
        vertices,
        faces,
        fit_supported_inner,
        keep,
        max_distance_um=max_recovery_distance_um,
    )
    estimated_outer, distance_outer = evidence_recovery_support(
        vertices,
        faces,
        fit_supported_outer,
        keep,
        max_distance_um=max_recovery_distance_um,
    )
    inner[~estimated_inner] = np.nan
    outer[~estimated_outer] = np.nan
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
            fit_supported_inner=fit_supported_inner,
            fit_supported_outer=fit_supported_outer,
            crossing_support_count_inner=crossing_count_inner,
            crossing_support_count_outer=crossing_count_outer,
            recovery_distance_inner_um=distance_inner,
            recovery_distance_outer_um=distance_outer,
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
                "fit_supported_vertices": int(fit_supported_inner.sum()),
                "estimated_vertices": int(estimated_inner.sum()),
                "recovered_missing_measurement_vertices": int(recovered_inner.sum()),
            },
            "outer": {
                "peak_side": "falling",
                "peak_fraction": outer_fraction,
                "smooth_um": outer_smooth_um,
                "direct_supported_vertices": int(direct_outer.sum()),
                "fit_supported_vertices": int(fit_supported_outer.sum()),
                "estimated_vertices": int(estimated_outer.sum()),
                "recovered_missing_measurement_vertices": int(recovered_outer.sum()),
            },
            "direct_supported_joint_vertices": int(direct_joint.sum()),
            "fit": "full-mesh log-depth biharmonic FEM; direct observations supply fidelity",
            "evidence_quality": {
                "minimum_crossing_support_samples": min_crossing_samples,
                "maximum_recovery_geodesic_distance_um": max_recovery_distance_um,
                "crossing_support": "minimum valid_count at peak and interpolated crossing bracket",
            },
            "ordering": (
                "outer log depth constrained >= fitted inner log depth on wall_support with OSQP; "
                "cap vertices unconstrained"
                if trim_caps
                else "outer log depth constrained >= fitted inner log depth on every vertex with OSQP"
            ),
            "equal_inner_outer_vertices": int(
                np.count_nonzero(estimated_inner & estimated_outer & (outer == inner))
            ),
            "vertices": len(vertices),
            "wall_support_vertices": int(keep.sum()),
            "trimmed_cap_vertices": int((~keep).sum()),
            "cap_trimming": "intrinsic-x-turnarounds" if trim_caps else "none",
            "projection_domain": projection_metadata
            or {
                "name": "full",
                "rule": "legacy full-surface profile artifact",
                "vertex_mask_sha256": projection_digest,
                "vertices": int(projection_domain.sum()),
            },
            "cap_trimming_coordinates": "vertices_intrinsic_xyz_um" if trim_caps else None,
            "cavity_intrinsic_x_sign": cavity_x_sign,
            "terminal_bounds_intrinsic_z_ymin_ymax_um": bounds.tolist(),
            "raw_measurement_policy": "retained unchanged, including outside wall support",
            "outputs": {"boundaries.npz": sha256_file(boundaries_path)},
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return (output_dir / "metadata.json").resolve()
