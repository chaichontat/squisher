"""Consolidated channel QC from the post-BaSiC sampler's saved artifacts."""

from __future__ import annotations

import csv
from collections.abc import Mapping
import json
from pathlib import Path

from loguru import logger
from matplotlib import colormaps, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from squisher_deconv.residual_field import cosine_basis

import numpy as np
import tifffile


def write_post_basic_qc(
    output: Path,
    *,
    manifest: dict,
    flatfields: Mapping[int, np.ndarray] | None = None,
) -> None:
    """Write one figure per channel before the caller publishes the staged run.

    Ownership indices refer to sampling.sampled_sources, never filename order.
    Camera fields are summarized independently from the source-specific gains.
    Identity-only fields use a finite display range of at least 1 percent.
    """
    with rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
            "savefig.bbox": None,
        }
    ):
        _write_report(output, manifest=manifest, flatfields=flatfields)


def _write_report(
    output: Path,
    *,
    manifest: dict,
    flatfields: Mapping[int, np.ndarray] | None,
) -> None:
    qc_dir = output / "qc"
    qc_dir.mkdir()
    residual_only = manifest.get("input_stage") == "deconvolved"
    report_title = "Residual calibration QC" if residual_only else "Post-BaSiC QC"
    comparison_titles = (
        ["(a) Deconvolved", "(b) Residual corrected"]
        if residual_only
        else ["(a) BaSiC only", "(b) BaSiC + post-BaSiC"]
    )
    field_titles = (
        ["(d) Input field · identity", "(e) Shared residual · raw Z 50%"]
        if residual_only
        else ["(d) BaSiC flatfield · divisor", "(e) Post-BaSiC · raw Z 50%"]
    )
    correction_stage_names = (
        ("Input", "Residual", "Applied") if residual_only else ("BaSiC", "Post-BaSiC", "Combined")
    )
    if flatfields is None and not residual_only:
        raise ValueError("Post-BaSiC QC requires the fitted BaSiC flatfields")
    gains = json.loads((output / "tile-gains.json").read_text())["tiles"]
    spacing_y, spacing_x = np.asarray(manifest["fixed_spacing_um"][1:]) * manifest["stride"]
    origin_y, origin_x = manifest["fixed_origin_um"][1:]
    mask_limits = [
        0.01
        if flatfields is None
        else max(0.01, max(float(np.abs(field - 1).max()) for field in flatfields.values())),
        max(
            0.01,
            max(
                float(np.abs(tifffile.imread(output / result["artifacts"]["mask"]["path"]) - 1).max())
                for result in manifest["channel_results"].values()
            ),
        ),
    ]
    summary = []
    z_rows = []
    figure_info = {}
    tile_rows = []
    source_field_rows = []
    for channel, result in manifest["channel_results"].items():
        paths = [
            output / result["artifacts"][key]["path"] for key in ["basic_tiff", "corrected_tiff", "mask"]
        ]
        before, after, post = [tifffile.imread(path) for path in paths]
        basic = np.ones_like(post, dtype=np.float32) if flatfields is None else flatfields[int(channel)]
        owner = tifffile.imread(output / result["artifacts"]["owner"]["path"])
        sampled_sources = result["sampling"]["sampled_sources"]
        if result.get("source_fields_applied"):
            source_fields = {row["source"]: np.asarray(row["coefficient"]) for row in result["source_fields"]}
            coefficient_count = len(np.asarray(result["coefficient"]))
            if not set(sampled_sources) <= set(source_fields) or any(
                coefficient.shape != (coefficient_count,) for coefficient in source_fields.values()
            ):
                raise ValueError(f"Channel {channel} source fields do not match sampled sources")
            fraction = np.linspace(0, 1, 64)
            yy, xx = np.meshgrid(fraction, fraction, indexing="ij")
            yx = np.column_stack([yy.ravel(), xx.ravel()])
            source_z_degree = coefficient_count // 5 - 1
            for source, source_coefficient in source_fields.items():
                for z in (0.0, 0.5, 1.0) if source_z_degree else (0.5,):
                    values = np.exp(
                        cosine_basis(
                            yx,
                            z=np.full(len(yx), z),
                            z_degree=source_z_degree,
                        )
                        @ source_coefficient
                    )
                    percentiles = np.percentile(values, [0, 5, 50, 95, 100])
                    source_field_rows.append(
                        {
                            "channel": channel,
                            "source": source,
                            "raw_z_fraction": z,
                            "minimum": percentiles[0],
                            "p05": percentiles[1],
                            "median": percentiles[2],
                            "p95": percentiles[3],
                            "maximum": percentiles[4],
                        }
                    )
        if owner.shape != before.shape or np.any(owner < -1) or np.any(owner >= len(sampled_sources)):
            raise ValueError(f"Channel {channel} QC owner map does not match sampled sources")
        if not all(np.isfinite(a).all() for a in [before, after, post, basic]):
            raise ValueError(f"Channel {channel} QC arrays contain nonfinite values")
        if np.any(basic <= 0) or np.any(post <= 0) or post.shape != basic.shape:
            raise ValueError(f"Channel {channel} QC fields must be positive and have matching shapes")
        fitted = result["tile_gains_applied"]
        width = 11.3 if fitted else 9.1
        multi_z = "z_validation" in result
        height = max(6.0, len(gains) * 0.14) if fitted else 6.0
        fig = Figure(figsize=(width, height + 2 if multi_z else height))
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(
            3 if multi_z else 2,
            4 if fitted else 3,
            left=0.045 if fitted else 0.065,
            right=0.965,
            bottom=0.09,
            top=0.915,
            hspace=0.28,
            wspace=0.32,
            width_ratios=[1, 1, 1, 0.66] if fitted else [1, 1, 1],
            height_ratios=[1, 1, 0.65] if multi_z else [1, 1],
        )
        axes = np.array([[fig.add_subplot(gs[row, col]) for col in range(3)] for row in range(2)])
        fig.suptitle(f"{result['label']} · channel {channel}", fontsize=12, y=0.985)
        lo, hi = result["artifacts"]["display_range"]["values"]
        for ax, data, title in zip(axes[0, :2], [before, after], comparison_titles):
            ax.imshow(
                data,
                cmap="gray",
                vmin=lo,
                vmax=hi,
                interpolation="nearest",
                aspect=abs(spacing_y / spacing_x),
            )
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_axis_off()
        physical_width = before.shape[1] * abs(spacing_x)
        target = physical_width * 0.2
        power = 10 ** np.floor(np.log10(target))
        bar_um = float(max(value for value in (1, 2, 5, 10) if value <= target / power) * power)
        bar = bar_um / abs(spacing_x)
        height, image_width = after.shape
        margin = 0.05 * min(height, image_width)
        axes[0, 1].add_patch(
            Rectangle(
                (image_width - margin - bar, height - margin - max(1, height * 0.004)),
                bar,
                max(1, height * 0.004),
                color="white",
                linewidth=0,
            )
        )
        for ax, data, title, limit in zip(
            axes[1, :2],
            [basic, post],
            [
                field_titles[0],
                field_titles[1]
                if result.get("z_degree", 0)
                else field_titles[1].replace(" · raw Z 50%", " · multiplier"),
            ],
            mask_limits,
        ):
            shown = ax.imshow(
                data, cmap="RdBu_r", norm=TwoSlopeNorm(1, 1 - limit, 1 + limit), interpolation="nearest"
            )
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_xticks([0, (data.shape[1] - 1) // 2, data.shape[1] - 1])
            ax.set_yticks([0, (data.shape[0] - 1) // 2, data.shape[0] - 1])
            ax.set_xlabel("Camera x (px)")
            ax.set_ylabel("Camera y (px)")
            # Inset colorbars preserve the same square panel size as the images.
            cax = ax.inset_axes([1.035, 0, 0.045, 1])
            fig.colorbar(shown, cax=cax)
        axes[1, 1].set_ylabel("")
        axes[1, 1].set_yticks([])
        channel_summary = []
        for name, factor in zip(
            correction_stage_names,
            (1.0 / basic, post, post / basic),
            strict=True,
        ):
            values = 100 * (factor.astype(np.float64) - 1)
            percentiles = np.percentile(values, [0, 5, 50, 95, 100])
            row = dict(
                channel=channel,
                label=result["label"],
                stage=name,
                minimum=percentiles[0],
                p05=percentiles[1],
                median=percentiles[2],
                p95=percentiles[3],
                maximum=percentiles[4],
            )
            summary.append(row)
            channel_summary.append(row)
        ax = axes[1, 2]
        ax.set_title("(f) Field correction magnitude", loc="left", fontsize=9)
        ax.axvline(0, color="0.7", linewidth=0.8)
        for y, row, color in zip(range(3), channel_summary, ["#0072B2", "#E69F00", "#009E73"]):
            ax.plot([row["p05"], row["p95"]], [y, y], color=color, linewidth=2)
            ax.plot(row["median"], y, "o", color=color)
        ax.set_yticks([])
        for y, row in enumerate(channel_summary):
            ax.text(0.04, y - 0.23, row["stage"], transform=ax.get_yaxis_transform(), fontsize=8)
        ax.set_ylim(2.6, -0.6)
        extent = max(
            1.0,
            max(abs(row[key]) for row in channel_summary for key in ["p05", "p95"]) * 1.08,
        )
        ax.set_xlim(-extent, extent)
        ax.set_xlabel("Intensity change (%)")
        ax.set_box_aspect(1)

        excluded = set(result["excluded_sources"])
        unsupported = set(result.get("tile_gain_unestimated_sources", []))
        rows = []
        for index, tile_gain in enumerate(gains):
            source = tile_gain["source"]
            if source in excluded:
                status = "excluded"
            elif source in unsupported:
                status = "identity: no training-pair support"
            elif fitted:
                status = "fitted"
            else:
                status = "identity: not fitted"
            rows.append(
                dict(
                    channel=channel,
                    tile=f"{index:03d}",
                    source=source,
                    gain=tile_gain["gains"][int(channel)],
                    change_percent=100 * (tile_gain["gains"][int(channel)] - 1),
                    status=status,
                )
            )
        tile_rows.extend(rows)
        ax = axes[0, 2]
        limit = max(1.0, max(abs(row["change_percent"]) for row in rows))
        norm = TwoSlopeNorm(0, -limit, limit) if fitted else None
        cmap = colormaps["RdBu_r"]
        source_rows = {row["source"]: row for row in rows}
        sampled_gains = np.asarray([source_rows[source]["change_percent"] for source in sampled_sources])
        gain_image = np.ma.array(sampled_gains[np.maximum(owner, 0)], mask=owner < 0)
        height, image_width = owner.shape
        x0, y0 = origin_x / 1000, origin_y / 1000
        extent_xy = [x0, x0 + image_width * spacing_x / 1000, y0 + height * spacing_y / 1000, y0]
        ax.imshow(
            gain_image,
            cmap=cmap if fitted else "Greys",
            norm=norm,
            vmin=None if fitted else -1,
            vmax=None if fitted else 1,
            extent=extent_xy,
            interpolation="nearest",
        )
        # Use the sampler's visible ownership regions, including affine/cropped support.
        for index, source in enumerate(sampled_sources):
            yy, xx = np.nonzero(owner == index)
            if not len(yy):
                continue
            label = source_rows[source]["tile"] + ("*" if source in unsupported else "")
            ax.text(
                x0 + (float(xx.mean()) + 0.5) * spacing_x / 1000,
                y0 + (float(yy.mean()) + 0.5) * spacing_y / 1000,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3),
            )
        ax.set_xlabel("Registered x (mm)")
        ax.set_ylabel("Registered y (mm)")
        ax.set_title("(c) Tile gains" if fitted else "(c) Tile gains · all 1×", loc="left", fontsize=9)
        if fitted:
            cax = ax.inset_axes([1.035, 0, 0.045, 1])
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            ax = fig.add_subplot(gs[:, 3])
            for index, row in enumerate(rows):
                if row["status"] == "excluded":
                    ax.plot(0, index, "x", color="0.55")
                else:
                    ax.barh(
                        index,
                        row["change_percent"],
                        color=cmap(norm(row["change_percent"])),
                    )
                    if row["status"] != "fitted":
                        ax.plot(0, index, "o", markerfacecolor="white", markeredgecolor="0.3")
            ax.axvline(0, color="0.65", linewidth=0.6)
            ax.set_yticks(range(len(rows)), [row["tile"] for row in rows])
            ax.invert_yaxis()
            ax.yaxis.tick_right()
            ax.set_xlim(-limit * 1.15, limit * 1.15)
            ax.set_xlabel("Gain change (%)")
            ax.set_title("(g) Per-tile change", loc="left", fontsize=9)
        if multi_z:
            percentile_by_z = dict(zip(manifest["sampled_z"], manifest["z_percentiles"], strict=True))
            ax = fig.add_subplot(gs[2, :3])
            evaluations = result["z_validation"]
            percentiles = [percentile_by_z[row["heldout_z"]] for row in evaluations]
            model_series = [
                ("before", "Deconvolved" if residual_only else "BaSiC only", "0.55"),
                ("single_plane", "Single training plane", "#E69F00"),
                (
                    "shared",
                    "Shared + source fields"
                    if result.get("source_fields_applied")
                    else ("Regularized 3D" if result.get("z_degree", 0) else "Shared across training planes"),
                    "#0072B2",
                ),
            ]
            if result.get("z_degree", 0):
                model_series.append(("shared_2d", "Shared 2D", "#009E73"))
            for name, label, color in model_series:
                ax.plot(
                    percentiles,
                    [row[name]["median"] for row in evaluations],
                    marker="o",
                    color=color,
                    label=label,
                )
            ax.set_xlabel("Held-out Z percentile")
            ax.set_ylabel("Seam discrepancy (log2)", labelpad=0)
            ax.set_xticks(percentiles if len(percentiles) <= 10 else np.linspace(0, 100, 11))
            ax.set_ylim(bottom=0)
            ax.set_title(
                "(h) Held-out-plane generalization" if fitted else "(g) Held-out-plane generalization",
                loc="left",
                fontsize=9,
            )
            ax.legend(loc="upper right", frameon=True, fontsize=7)
            for evaluation in evaluations:
                row = {
                    "channel": channel,
                    "label": result["label"],
                    "heldout_z": evaluation["heldout_z"],
                    "percentile": percentile_by_z[evaluation["heldout_z"]],
                    "training_z": json.dumps(evaluation["training_z"]),
                    "single_plane_z": evaluation["single_plane_z"],
                    "pairs": evaluation["before"]["pairs"],
                }
                for name, _label, _color in model_series:
                    for metric in ("median", "p90", "pixel_abs_median"):
                        row[f"{name}_{metric}"] = evaluation[name][metric]
                z_rows.append(row)
        fig.text(
            0.5,
            0.012,
            f"Fields: median and 5th–95th percentiles · Scale bar: {bar_um:g} µm"
            + (
                " · Tile gains: color = % change; × excluded; * identity"
                if fitted
                else " · Per-tile gains: unity"
            ),
            ha="center",
            fontsize=8,
        )
        stem = f"channel_{channel}"
        for extension in ("png", "pdf", "svg"):
            fig.savefig(qc_dir / f"{stem}.{extension}", dpi=300, facecolor="white")
        figure_info[channel] = {
            "label": result["label"],
            "scale_bar_um": bar_um,
            "mask_half_ranges": mask_limits,
            "camera_shape_yx": list(basic.shape),
            "display_range": [lo, hi],
            "tile_gains_fitted": fitted,
            "source_fields_fitted": bool(result.get("source_fields_applied")),
            "paths": {ext: f"{stem}.{ext}" for ext in ("png", "pdf", "svg")},
        }
        fig.clear()
        diagnostic = Figure(figsize=(10, 3.5), layout="constrained")
        FigureCanvasAgg(diagnostic)
        diagnostic_axes = diagnostic.subplots(1, 3)
        valid = (owner >= 0) & (before > 0)
        ratio = np.divide(after, before, out=np.full_like(after, np.nan), where=valid)
        limit = max(0.01, float(np.max(np.abs(ratio[valid] - 1))))
        shown = diagnostic_axes[0].imshow(ratio, cmap="RdBu_r", norm=TwoSlopeNorm(1, 1 - limit, 1 + limit))
        diagnostic.colorbar(shown, ax=diagnostic_axes[0], shrink=0.75, label="Applied multiplier")
        diagnostic_axes[0].set_title("After / before · fixed Z", fontsize=9)
        diagnostic_axes[0].set_axis_off()
        fraction = np.linspace(0, 1, 512)
        coefficient = np.asarray(result["coefficient"])
        z_degree = len(coefficient) // 5 - 1
        for axis, name in enumerate(("Y", "X"), start=1):
            yx = np.full((len(fraction), 2), 0.5)
            yx[:, axis - 1] = fraction
            for z in (0.0, 0.5, 1.0) if z_degree else (0.5,):
                field = np.exp(cosine_basis(yx, z=np.full(len(fraction), z), z_degree=z_degree) @ coefficient)
                diagnostic_axes[axis].plot(fraction, field, label=f"Raw Z {100 * z:g}%")
            diagnostic_axes[axis].axhline(1, color="0.6", linewidth=0.6)
            diagnostic_axes[axis].set_xlabel(f"Camera {name} fraction")
            diagnostic_axes[axis].set_ylabel("Residual multiplier")
            diagnostic_axes[axis].set_title("Other camera axis at 50%", fontsize=9)
        diagnostic_axes[-1].legend(fontsize=7)
        for extension in ("png", "pdf", "svg"):
            diagnostic.savefig(qc_dir / f"field_{channel}.{extension}", dpi=300, facecolor="white")
        diagnostic.clear()
        logger.info("{} rendered channel {}", report_title, channel)

    for filename, rows in [("correction_magnitude.csv", summary), ("tile_gains.csv", tile_rows)]:
        with (qc_dir / filename).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if source_field_rows:
        with (qc_dir / "source_fields.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(source_field_rows[0]))
            writer.writeheader()
            writer.writerows(source_field_rows)

    if z_rows:
        with (qc_dir / "z_validation.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(z_rows[0]))
            writer.writeheader()
            writer.writerows(z_rows)

    if residual_only:
        caption = f"""# Residual calibration QC

Before is deconvolved input; after applies the shared depth-aware field,
source-specific depth-aware fields, and saved source gains. Each image pair shares the recorded intensity range. The
plane is fixed-grid Z={manifest["fixed_z"]} ({manifest["fixed_z_um"]:.3f} µm).
Scale bars are labeled in each figure's footer.

The common scale S, shared field M, source field T, and source gain G multiply
intensity as corrected = raw × S × M × T × G. The residual mask and field
summaries show the shared field at raw Z 50%; source_fields.csv summarizes each
source field. Fusion evaluates both fields at each requested source pixel's
original raw Z/Y/X coordinates.

Field charts show 100 × (factor − 1). Points are medians; lines span the
5th–95th percentiles over all camera pixels. CSVs include extrema.

The tile map shows the sampler's visible source ownership at the selected plane.
Tile labels are source indices mapped to deconvolved sources in tile_gains.csv.
Color and bars show 100 × (gain − 1).
"""
    else:
        caption = f"""# Post-BaSiC QC

Before is BaSiC-only; after includes the residual field and saved tile gains.
Each image pair shares the recorded intensity range. The plane is fixed-grid
Z={manifest["fixed_z"]} ({manifest["fixed_z_um"]:.3f} µm). Scale bars are labeled
in each figure's footer.

BaSiC flatfield F divides intensity; residual field M multiplies it. The applied
correction is max((raw − darkfield) × M/F, 0) × tile gain. Mask scales are centered
on identity and shared across channels within each field type.
{"The residual mask and field summaries show raw Z 50%; image correction evaluates the field at each source pixel’s original raw Z." if manifest.get("z_degree", 0) else ""}

Field charts show 100 × (factor − 1), using 1/F, M, and M/F. Points are medians;
lines span the 5th–95th percentiles over all camera pixels. CSVs include extrema.

The tile map shows the sampler's visible source ownership at the selected plane.
Tile labels are source indices mapped to raw filenames in tile_gains.csv.
Color and bars show 100 × (gain − 1). An asterisk or open circle marks unsupported
identity gains; crosses mark excluded sources. Channels without fitted tile
gains use unity. Unit fields and gains have a minimum display half-range of 1%.
"""
    if z_rows:
        caption += """

The shared field and source gains are fitted across all requested Z planes.
Generalization curves hold out one complete plane at a time. The shared model
uses the other requested planes; the single-plane model uses the training plane
closest to the report's representative Z. Penalty selection uses training planes
only, with all measurements of a source pair assigned to the same split.
The curves report median absolute log2 seam discrepancy on the excluded plane.
Before/after images at every sampled Z are linked under each channel.
"""
    (qc_dir / "README.md").write_text(caption)
    (qc_dir / "qc_manifest.json").write_text(
        json.dumps(
            {
                "inputs": manifest["inputs"],
                "channels": figure_info,
                "dpi": 300,
                "field_percentiles": [0, 5, 50, 95, 100],
                "fixed_spacing_um": manifest["fixed_spacing_um"],
                "fixed_origin_um": manifest["fixed_origin_um"],
                "stride": manifest["stride"],
            },
            indent=2,
        )
        + "\n"
    )
