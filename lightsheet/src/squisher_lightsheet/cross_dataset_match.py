"""Match two already-corrected datasets with one view-level intensity factor."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from squisher_deconv.residual_field import validate_coefficients
from squisher_lightsheet.post_basic import (
    _artifact,
    _corrected_tiles,
    _fixed_grid,
    _registration_records,
    _sha256,
)
from squisher_lightsheet.residual_correction import (
    _sample_tiles,
    _source_identity,
    load_residual_corrections,
)


DEFAULT_Z_PERCENTILES = (20.0, 35.0, 50.0, 65.0, 80.0)


def _view(record: Mapping[str, object]) -> str:
    value = record.get("source_view", record.get("side"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Registration tile {record.get('tile')!r} has no source_view or side")
    return value.strip()


def _source(record: Mapping[str, object]) -> Path:
    value = record.get("path")
    if not isinstance(value, str):
        raise ValueError(f"Registration tile {record.get('tile')!r} has no source path")
    return Path(value).resolve()


def _pad_coefficient(coefficient: np.ndarray, rows: int) -> np.ndarray:
    result = np.zeros((rows, 5), dtype=np.float64)
    result[: len(coefficient)] = coefficient
    return result


def _identity_payload(
    records: Sequence[Mapping[str, object]], *, channel: int
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "kind": "squisher.residual-correction",
        "channel": channel,
        "coordinate": "normalized-original-raw-zyx",
        "basis": "cosine-xy2-z-scaled",
        "coefficient": [[0.0] * 5],
        "global_scale": 1.0,
        "sources": [
            {
                "path": str(_source(record)),
                "source_sha256": _source_identity(_source(record)),
                "shape_zyx": [int(value) for value in record["shape"][-3:]],
                "gain": 1.0,
                "coefficient": [[0.0] * 5],
                "training_pairs": 0,
            }
            for record in records
        ],
    }


def compose_corrections(
    *,
    records: Sequence[Mapping[str, object]],
    correction_payloads: Mapping[str, Mapping[str, object]],
    channel: int,
    moving_view: str,
    moving_factor: float,
) -> dict[str, object]:
    """Compose fixed per-view corrections and one moving-view factor into schema v3."""
    if not np.isfinite(moving_factor) or moving_factor <= 0:
        raise ValueError("Moving-view factor must be positive and finite")
    record_view = {str(_source(record)): _view(record) for record in records}
    if len(record_view) != len(records):
        raise ValueError("Registration sources must be unique")
    if set(record_view.values()) != set(correction_payloads):
        raise ValueError("Correction views must exactly match registration source views")

    rows_by_source: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {}
    coefficient_rows = 1
    for view, payload in correction_payloads.items():
        if (
            payload.get("schema_version") != 3
            or payload.get("kind") != "squisher.residual-correction"
            or payload.get("channel") != channel
            or payload.get("coordinate") != "normalized-original-raw-zyx"
            or payload.get("basis") != "cosine-xy2-z-scaled"
        ):
            raise ValueError(f"Invalid base residual correction for view {view!r}")
        shared = validate_coefficients(payload.get("coefficient"))
        coefficient_rows = max(coefficient_rows, len(shared))
        scale = payload.get("global_scale")
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid base correction global scale for view {view!r}")
        source_rows = payload.get("sources")
        if not isinstance(source_rows, list):
            raise ValueError(f"Invalid base correction sources for view {view!r}")
        for row in source_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                raise ValueError(f"Invalid base correction source row for view {view!r}")
            source = str(Path(row["path"]).resolve())
            if source in rows_by_source:
                raise ValueError(f"Base corrections contain duplicate source {source}")
            source_coefficient = validate_coefficients(row.get("coefficient"))
            if source_coefficient.shape != shared.shape:
                raise ValueError(
                    f"Base source coefficient shape does not match its shared coefficient: {source}"
                )
            rows_by_source[source] = (payload, row)

    if set(rows_by_source) != set(record_view):
        raise ValueError("Base corrections must cover the exact registered source set")
    if moving_view not in correction_payloads:
        raise ValueError(f"Moving view {moving_view!r} is not present in the registration")

    output_rows = []
    for record in records:
        source = str(_source(record))
        view = record_view[source]
        payload, row = rows_by_source[source]
        shared = _pad_coefficient(validate_coefficients(payload["coefficient"]), coefficient_rows)
        source_coefficient = _pad_coefficient(
            validate_coefficients(row.get("coefficient")), coefficient_rows
        )
        row_gain = row.get("gain")
        if type(row_gain) not in (int, float) or not np.isfinite(row_gain) or row_gain <= 0:
            raise ValueError(f"Invalid base correction source gain: {source}")
        gain = float(payload["global_scale"]) * float(row_gain)
        if view == moving_view:
            gain *= moving_factor
        output_rows.append(
            {
                "path": source,
                "source_sha256": row["source_sha256"],
                "shape_zyx": row["shape_zyx"],
                "gain": gain,
                "coefficient": (shared + source_coefficient).tolist(),
                "training_pairs": row.get("training_pairs", 0),
                "source_view": view,
            }
        )

    maximum = 1.0
    for row in output_rows:
        coefficient = np.asarray(row["coefficient"], dtype=np.float64)
        log_field_bound = sum(
            float(np.abs(coefficient[degree]).sum()) / (1 + degree**2)
            for degree in range(len(coefficient))
        )
        upper_bound = float(row["gain"]) * float(np.exp(log_field_bound))
        if not np.isfinite(upper_bound):
            raise ValueError("Composed correction has a non-finite multiplier bound")
        maximum = max(maximum, upper_bound)
    common_scale = 1.0 / maximum

    return {
        "schema_version": 3,
        "kind": "squisher.residual-correction",
        "channel": channel,
        "coordinate": "normalized-original-raw-zyx",
        "basis": "cosine-xy2-z-scaled",
        "coefficient": np.zeros((coefficient_rows, 5), dtype=np.float64).tolist(),
        "global_scale": common_scale,
        "sources": output_rows,
        "fit": {
            "method": "cross-dataset-match",
            "moving_view": moving_view,
            "moving_factor": moving_factor,
            "common_saturation_scale": common_scale,
            "fixed_per_dataset_corrections": True,
            "per_tile_gains_fitted": False,
            "spatial_fields_fitted": False,
            "cross_validation": False,
        },
    }


def _apply_base_corrections(tiles, corrections):
    corrected = []
    for tile in tiles:
        correction = corrections[str(tile.source.resolve())]
        corrected.extend(
            _corrected_tiles(
                [tile],
                correction.coefficient.reshape(-1),
                np.asarray([correction.multiplier]),
            )
        )
    return corrected


def _cross_view_measurements(
    tiles,
    *,
    view_by_source: Mapping[str, str],
    cutoff_by_source: Mapping[str, float],
    z: int,
):
    smooth = []
    for tile in tiles:
        support = gaussian_filter(tile.valid.astype(np.float32), 1)
        values = gaussian_filter(tile.data, 1)
        smooth.append(np.divide(values, support, out=np.zeros_like(values), where=support > 0))
    rows = []
    for first, a in enumerate(tiles):
        for second in range(first + 1, len(tiles)):
            b = tiles[second]
            a_source = str(a.source.resolve())
            b_source = str(b.source.resolve())
            a_view = view_by_source[a_source]
            b_view = view_by_source[b_source]
            if a_view == b_view:
                continue
            lo, hi = np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi)
            if np.any(hi - lo < 8):
                continue
            sa = tuple(
                slice(int(low - origin), int(high - origin), 4)
                for low, high, origin in zip(lo, hi, a.lo, strict=True)
            )
            sb = tuple(
                slice(int(low - origin), int(high - origin), 4)
                for low, high, origin in zip(lo, hi, b.lo, strict=True)
            )
            av, bv = smooth[first][sa], smooth[second][sb]
            mask = (
                a.valid[sa]
                & b.valid[sb]
                & (a.data[sa] > cutoff_by_source[a_source])
                & (b.data[sb] > cutoff_by_source[b_source])
                & (av > 0)
                & (bv > 0)
            )
            pixels = int(np.count_nonzero(mask))
            if pixels < 12:
                continue
            rows.append(
                {
                    "z": z,
                    "first_source": str(a.source.resolve()),
                    "second_source": str(b.source.resolve()),
                    "first_view": a_view,
                    "second_view": b_view,
                    "pixels": pixels,
                    "log_ratio_second_over_first": float(np.median(np.log(bv[mask] / av[mask]))),
                }
            )
    return rows


def _write_qc(output: Path, rows: Sequence[Mapping[str, object]], *, moving_view: str, factor: float):
    before = []
    z_values = []
    for row in rows:
        ratio = float(row["log_ratio_second_over_first"])
        if row["second_view"] != moving_view:
            ratio = -ratio
        before.append(ratio / np.log(2))
        z_values.append(int(row["z"]))
    before_array = np.asarray(before)
    after_array = before_array + np.log2(factor)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].scatter(z_values, before_array, s=18, alpha=0.7, label="before")
    axes[0].scatter(z_values, after_array, s=18, alpha=0.7, label="after")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Fused-grid Z", ylabel=f"log2({moving_view} / reference)")
    axes[0].legend()
    axes[1].boxplot([before_array, after_array], tick_labels=["before", "after"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel(f"log2({moving_view} / reference)")
    fig.suptitle(f"Cross-dataset match: {moving_view} × {factor:.6g}")
    artifacts = {}
    for suffix in ("png", "pdf", "svg"):
        path = output / f"cross-dataset-match-qc.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        artifacts[suffix] = _artifact(path, output)
    plt.close(fig)
    return artifacts, before_array, after_array


def cross_dataset_match(
    *,
    registration: Path,
    fixed_fused: Path,
    output_dir: Path,
    corrections_by_view: Mapping[str, Path],
    reference_view: str,
    moving_view: str,
    channel: int = 0,
    source_level: int = 2,
    stride: int = 4,
    z_percentiles: Sequence[float] = DEFAULT_Z_PERCENTILES,
    workers: int = 8,
) -> Path:
    """Fit one cross-view factor and atomically publish a composed correction."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite cross-dataset match output: {output_dir}")
    if reference_view == moving_view:
        raise ValueError("Reference and moving views must differ")
    selected_views = {reference_view, moving_view}
    if not set(corrections_by_view) <= selected_views:
        raise ValueError("Base corrections may only name the selected reference and moving views")
    if channel < 0 or source_level < 0 or stride < 1 or workers < 1:
        raise ValueError("Invalid channel, source level, stride, or workers")
    if not z_percentiles or any(
        not np.isfinite(percentile) or not 0 <= percentile <= 100 for percentile in z_percentiles
    ):
        raise ValueError("Z percentiles must be finite and within [0, 100]")

    registration = registration.resolve()
    fixed_fused = fixed_fused.resolve()
    output_dir = output_dir.resolve()
    records = list(_registration_records(registration))
    views = [_view(record) for record in records]
    if set(views) != {reference_view, moving_view}:
        raise ValueError("Registration must contain exactly the selected reference and moving views")
    view_by_source = {str(_source(record)): _view(record) for record in records}
    if len(view_by_source) != len(records):
        raise ValueError("Registration sources must be unique")

    payloads = {}
    corrections = {}
    correction_modes = {}
    for view in (reference_view, moving_view):
        selected = [record for record in records if _view(record) == view]
        path = corrections_by_view.get(view)
        if path is None:
            payload = _identity_payload(selected, channel=channel)
            correction_modes[view] = "identity"
        else:
            path = path.resolve()
            payload = json.loads(path.read_text())
            correction_modes[view] = "supplied"
        payloads[view] = payload
        sources = [_source(record) for record in selected]
        shapes = [tuple(int(value) for value in record["shape"][-3:]) for record in selected]
        if path is None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".correction.json") as handle:
                json.dump(payload, handle)
                handle.flush()
                loaded = load_residual_corrections(
                    Path(handle.name), sources=sources, shapes_zyx=shapes, channel=channel
                )
        else:
            loaded = load_residual_corrections(
                path, sources=sources, shapes_zyx=shapes, channel=channel
            )
        corrections.update(loaded)

    base = _fixed_grid(fixed_fused, fixed_z=None, stride=stride)
    sampled_z = sorted(
        {int(np.floor((base.shape[0] - 1) * percentile / 100)) for percentile in z_percentiles}
    )
    if len(sampled_z) != len(z_percentiles):
        raise ValueError("Z percentiles must select distinct planes")
    measurements = []
    for z in sampled_z:
        grid = replace(base, z=z)
        tiles = _apply_base_corrections(
            _sample_tiles(records, grid, channel, source_level, workers), corrections
        )
        if len(tiles) < 2:
            raise ValueError(f"Cross-dataset match sampled fewer than two sources at fixed Z {z}")
        cutoff_by_source = {}
        for tile in tiles:
            positive = tile.data[tile.valid & (tile.data > 0)]
            if positive.size < 2:
                raise ValueError(f"Insufficient source signal in {tile.source} at fixed Z {z}")
            cutoff_by_source[str(tile.source.resolve())] = float(
                np.expm1(threshold_otsu(np.log1p(positive)))
            )
        measurements.extend(
            _cross_view_measurements(
                tiles,
                view_by_source=view_by_source,
                cutoff_by_source=cutoff_by_source,
                z=z,
            )
        )
    if not measurements:
        raise ValueError("No eligible cross-view overlaps were found")

    moving_over_reference = []
    for row in measurements:
        ratio = float(row["log_ratio_second_over_first"])
        if row["second_view"] != moving_view:
            ratio = -ratio
        moving_over_reference.append(ratio)
    log_factor = -float(np.median(moving_over_reference))
    factor = float(np.exp(log_factor))
    correction = compose_corrections(
        records=records,
        correction_payloads=payloads,
        channel=channel,
        moving_view=moving_view,
        moving_factor=factor,
    )
    correction["fit"].update(
        {
            "registration": str(registration),
            "registration_sha256": _sha256(registration),
            "fixed_fused": str(fixed_fused),
            "reference_view": reference_view,
            "z_percentiles": list(z_percentiles),
            "sampled_z": sampled_z,
            "estimator": "median of cross-view pair/depth median log ratios",
            "base_correction_modes": correction_modes,
        }
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cross-dataset-match-", dir=output_dir.parent))
    try:
        correction_path = stage / "correction.json"
        correction_path.write_text(json.dumps(correction, indent=2) + "\n")
        qc_artifacts, before, after = _write_qc(
            stage, measurements, moving_view=moving_view, factor=factor
        )
        for row, before_value, after_value in zip(measurements, before, after, strict=True):
            row["before_log2_moving_over_reference"] = float(before_value)
            row["after_log2_moving_over_reference"] = float(after_value)
        match = {
            "schema_version": 1,
            "kind": "squisher.cross-dataset-match",
            "reference_view": reference_view,
            "moving_view": moving_view,
            "moving_factor": factor,
            "pair_depth_measurements": measurements,
            "summary": {
                "pairs": len(measurements),
                "before_median_abs_log2": float(np.median(np.abs(before))),
                "after_median_abs_log2": float(np.median(np.abs(after))),
                "before_p95_abs_log2": float(np.percentile(np.abs(before), 95)),
                "after_p95_abs_log2": float(np.percentile(np.abs(after), 95)),
            },
        }
        match_path = stage / "match.json"
        match_path.write_text(json.dumps(match, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "artifact_type": "squisher_lightsheet.cross_dataset_match.v1",
            "status": "complete",
            "registration": {"path": str(registration), "sha256": _sha256(registration)},
            "fixed_fused": str(fixed_fused),
            "channel": channel,
            "base_corrections": {
                view: (
                    {
                        "mode": "supplied",
                        "path": str(corrections_by_view[view].resolve()),
                        "sha256": _sha256(corrections_by_view[view].resolve()),
                    }
                    if view in corrections_by_view
                    else {"mode": "identity", "sources": views.count(view)}
                )
                for view in (reference_view, moving_view)
            },
            "correction": _artifact(correction_path, stage),
            "match": _artifact(match_path, stage),
            "qc": qc_artifacts,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        stage.replace(output_dir)
        return output_dir / "manifest.json"
    except BaseException:
        shutil.rmtree(stage, ignore_errors=False)
        raise
