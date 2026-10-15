"""Per-acquisition HCR stages (WORKFLOW_HCR.md steps 1-4): preprocessing, native tile geometry with reviewed
fit-residual corrections, and the dataset-local 561 mosaics that cross-dataset placement registers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from squisher_lightsheet._legacy.stitch_20x_tl_multiview import (
    file_resume_identity,
    source_tile_resume_identity,
)
from squisher_lightsheet.artifact_io import registration_input_fingerprint
from squisher_lightsheet.edu_workflow import _write_metadata_grid as write_metadata_grid
from squisher_lightsheet.workflow_runner import (
    StageRunner,
    sha256_file,
)

PSF_DIR = Path("/home/chaichontat/nvme/lightsheet/20260606")
# WORKFLOW_HCR.md: HCR data are sampled at about 1.8 um in Z, so use the centred stride-6 kernels. Only 571 nm and
# 648 nm kernels exist; 405, 488, and 514 use the 571 nm kernel, as 229Tnc and 230Tnc did.
PSF_571 = PSF_DIR / "psf_Image_57_571nm_zstrict/radial_symmetric_reskewed_rawscale_zstride6_xy9_sumnorm.tif"
PSF_648 = PSF_DIR / "psf_Image_57_648nm_std/radial_symmetric_reskewed_rawscale_zstride6_xy9_sumnorm.tif"
# About 100 channel-planes per deconvolution slab, at most 50 planes, as in the EDU production baseline.
SLAB_CHANNEL_PLANES = 100
MAXIMUM_SLAB_DEPTH = 50


@dataclass(frozen=True)
class HcrPaths:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "run-logs"

    @property
    def stages(self) -> Path:
        return self.root / ".stages"

    def acquisition(self, name: str) -> Path:
        return self.root / "acquisitions" / name

    def deconv(self, name: str) -> Path:
        return self.acquisition(name) / "deconvolved"

    def native_review(self, name: str) -> Path:
        return self.acquisition(name) / "native-review"

    def positions(self, name: str) -> Path:
        return self.acquisition(name) / "metadata.positions.json"

    def registration_dir(self, name: str, label: str) -> Path:
        return self.acquisition(name) / f"registration-{label}"

    def registration(self, name: str, label: str) -> Path:
        return self.registration_dir(name, label) / "registration.json"

    def registered_positions(self, name: str, label: str) -> Path:
        return self.registration_dir(name, label) / "registration.positions.json"

    def grid(self, name: str, label: str) -> Path:
        return self.acquisition(name) / f"grid-{label}.ome.zarr"

    def residual(self, name: str, label: str) -> Path:
        return self.acquisition(name) / f"fit-residual-{label}"

    def side_fusion_dir(self, side: str, label: str) -> Path:
        return self.root / "sides" / side / f"fusion-{label}"

    def side_fused(self, side: str, label: str, channel: int) -> Path:
        return self.side_fusion_dir(side, label) / f"fused.ch{channel}.ome.zarr"

    def channel(self, label: str) -> Path:
        return self.root / "channels" / label

    @property
    def manual_initializations(self) -> Path:
        """The dataset master of curated and measured window seeds, one channel per label and side (``514-CL``)."""
        return self.root / "manual-initializations.json"


@dataclass(frozen=True)
class ChannelSource:
    """Where one side's channel lives, and which tile registration places it."""

    side: str
    label: str
    acquisition: str
    index: int
    registration_label: str


def channel_sources(manifest: dict[str, Any]) -> list[ChannelSource]:
    """638 reuses the 561 tile registration of its own acquisition and track; every other label registers itself."""
    sources = []
    for acquisition in manifest["acquisitions"]:
        name = Path(acquisition["directory"]).name
        indices = {channel["label"]: channel["index"] for channel in acquisition["channels"]}
        for label, index in indices.items():
            registration_label = label
            if label == "638":
                track = next(track for track in acquisition["tracks"] if index in track)
                if indices.get("561") not in track:
                    raise ValueError(
                        f"{name}: 638 does not share a track with 561, so it has no shared geometry"
                    )
                registration_label = "561"
            sources.append(ChannelSource(acquisition["side"], label, name, index, registration_label))
    return sources


def _psf(label: str) -> Path:
    return PSF_648 if label == "638" else PSF_571


def require_complete_fusion(path: Path) -> None:
    metadata = json.loads((path / "zarr.json").read_text())
    if metadata.get("attributes", {}).get("squisher_complete") is not True:
        raise ValueError(f"Fusion is not complete: {path}")


def _require_complete_deconvolution(path: Path, expected_count: int) -> None:
    stores = sorted(path.glob("*.ome.zarr"))
    if len(stores) != expected_count:
        raise ValueError(f"Expected {expected_count} deconvolved stores in {path}, found {len(stores)}")
    for store in stores:
        metadata = json.loads((store / "zarr.json").read_text())
        if metadata.get("attributes", {}).get("squisher_complete") is not True:
            raise ValueError(f"Incomplete deconvolved store: {store}")


def preprocess_acquisition(
    runner: StageRunner,
    paths: HcrPaths,
    acquisition: dict[str, Any],
    *,
    all_gpu_env: dict[str, str],
    primary_gpu_env: dict[str, str],
) -> None:
    """BaSiC, corrected scaling samples, and deconvolution of one acquisition (WORKFLOW_EDU.md step 1)."""
    name = Path(acquisition["directory"]).name
    tiffs = [Path(acquisition["directory"]) / tile["name"] for tile in acquisition["tiles"]]
    labels = [channel["label"] for channel in acquisition["channels"]]
    channels = len(labels)
    base = paths.acquisition(name)
    basic = base / "basic"
    scale = base / "scale"
    deconv = paths.deconv(name)
    profiles = [basic / f"{name}-ch{channel}.pkl" for channel in range(channels)]
    psfs = [_psf(label) for label in labels]
    tiff_identity = [source_tile_resume_identity(path, require_completion=False) for path in tiffs]
    runner.command(
        f"{name}-basic",
        [
            "squisher-deconv",
            "basic",
            *tiffs,
            "--out-dir",
            basic,
            "--label",
            name,
            "--channels",
            str(channels),
            "--samples",
            "500",
            "--samples-per-tile",
            "2",
            "--working-size",
            "128",
        ],
        required=profiles,
        input_identity=lambda: tiff_identity,
        env=primary_gpu_env,
    )
    common: list[str | Path] = ["--channels", str(channels)]
    for psf in psfs:
        common += ["--psf", psf]
    for profile in profiles:
        common += ["--basic", profile]
    common += ["--iter", "1", "--devices", "auto", "--queue-depth", "2"]
    runner.command(
        f"{name}-scale",
        ["squisher-deconv", "sample-scale", *tiffs, "--out-dir", scale, *common, "--planes", "100"],
        required=[scale / "scaling.json"],
        input_identity=lambda: {
            "tiffs": tiff_identity,
            "profiles": [file_resume_identity(path) for path in profiles],
            "psfs": [file_resume_identity(path) for path in psfs],
        },
        env=all_gpu_env,
    )
    runner.command(
        f"{name}-deconvolution",
        [
            "squisher-deconv",
            "run",
            *tiffs,
            "--out-dir",
            deconv,
            *common,
            "--scaling",
            scale / "scaling.json",
            "--slab-depth",
            str(min(MAXIMUM_SLAB_DEPTH, SLAB_CHANNEL_PLANES // channels)),
            "--jpegxr-level",
            "0.7",
            "--resume",
        ],
        required=[deconv],
        env=all_gpu_env,
        validate=lambda: _require_complete_deconvolution(deconv, len(tiffs)),
        input_identity=lambda: {
            "tiffs": tiff_identity,
            "profiles": [file_resume_identity(path) for path in profiles],
            "psfs": [file_resume_identity(path) for path in psfs],
            "scaling": file_resume_identity(scale / "scaling.json"),
        },
    )


def registration_threshold_keys(manifest: dict[str, Any]) -> list[str]:
    """One native threshold per independently registered tile graph: 561 (shared with 638) and each other label."""
    return [f"{source.side}/{source.label}" for source in channel_sources(manifest) if source.label != "638"]


def register_native_geometry(
    runner: StageRunner,
    paths: HcrPaths,
    manifest: dict[str, Any],
    thresholds: dict[str, float],
    *,
    primary_gpu_env: dict[str, str],
) -> list[ChannelSource]:
    """Metadata positions, tile registration, grid template, and fit-residual for every side and channel."""
    sources = channel_sources(manifest)
    for acquisition in manifest["acquisitions"]:
        name = Path(acquisition["directory"]).name
        tiffs = [Path(acquisition["directory"]) / tile["name"] for tile in acquisition["tiles"]]
        overrides = paths.acquisition(name) / "position-overrides.json"
        override_args = ["--position-overrides", overrides] if overrides.is_file() else []
        runner.command(
            f"{name}-positions",
            [
                "lightsheet",
                "single-position",
                "--input-dir",
                acquisition["directory"],
                "--output",
                paths.positions(name),
                "--side",
                "L" if acquisition["side"] == manifest["fixed_side"] else "R",
                *override_args,
            ],
            required=[paths.positions(name)],
            input_identity=lambda tiffs=tiffs, overrides=overrides: {
                "tiles": [source_tile_resume_identity(path, require_completion=False) for path in tiffs],
                "position_overrides": file_resume_identity(overrides) if overrides.is_file() else None,
            },
        )
    for source in sources:
        if source.registration_label != source.label:
            continue
        name, label = source.acquisition, source.label
        runner.command(
            f"{name}-register-{label}",
            [
                "lightsheet-stitch",
                "register",
                "--position-json",
                paths.positions(name),
                "--zarr-dir",
                paths.deconv(name),
                "--output-dir",
                paths.registration_dir(name, label),
                "--channel",
                str(source.index),
                "--threshold",
                str(thresholds[f"{source.side}/{label}"]),
                "--level",
                "0",
                "--device",
                "0",
            ],
            required=[paths.registration(name, label), paths.registered_positions(name, label)],
            input_identity=lambda name=name: registration_input_fingerprint(
                paths.positions(name), paths.deconv(name)
            ),
            env=primary_gpu_env,
        )
    for source in sources:
        name, label = source.acquisition, source.label
        registration = paths.registration(name, source.registration_label)
        grid = paths.grid(name, label)
        runner.action(
            f"{name}-grid-{label}",
            {"registration_sha256": sha256_file(registration), "label": label},
            lambda grid=grid, registration=registration, name=name: write_metadata_grid(
                registration, paths.deconv(name), grid
            ),
            required=[grid / "zarr.json", grid / "0/zarr.json"],
        )
        residual = paths.residual(name, label)
        runner.command(
            f"{name}-fit-residual-{label}",
            [
                "lightsheet",
                "fit-residual",
                "--fixed-fused",
                grid,
                "--registration",
                registration,
                "--output-dir",
                residual,
                "--channel",
                str(source.index),
                "--source-level",
                "2",
                "--stride",
                "4",
                "--xy-degree",
                "1",
                "--z-degree",
                "1",
                "--workers",
                "8",
                "--seed",
                "20260907",
            ],
            required=[residual / "correction.json", residual / "manifest.json"],
            input_identity=lambda registration=registration, grid=grid, name=name: {
                "registration": file_resume_identity(registration),
                "tiles": [
                    source_tile_resume_identity(path, require_completion=True)
                    for path in sorted(paths.deconv(name).glob("*.ome.zarr"))
                ],
                "grid": source_tile_resume_identity(grid, require_completion=False),
            },
        )
    return sources


def fuse_side_561(
    runner: StageRunner, paths: HcrPaths, source: ChannelSource, *, primary_gpu_env: dict[str, str]
) -> Path:
    """The dataset-local 561 mosaic of one side: a cross-dataset registration input, not the canonical grid."""
    name = source.acquisition
    registration = paths.registration(name, "561")
    positions = paths.registered_positions(name, "561")
    correction = paths.residual(name, "561") / "correction.json"
    grid = paths.grid(name, "561")
    fused = paths.side_fused(source.side, "561", source.index)
    runner.command(
        f"{source.side}-fuse-561",
        [
            "lightsheet",
            "fuse",
            paths.deconv(name),
            "--position-input",
            positions,
            "--registration-input",
            registration,
            "--output",
            paths.side_fusion_dir(source.side, "561"),
            "--channel",
            str(source.index),
            "--fusion-level",
            "0",
            "--fusion-weight-mode",
            "content-preibisch-coarse",
            "--batch-size",
            "auto",
            "--source-cache-max-gib",
            "0",
            "--output-chunksize-zyx",
            "24,960,960",
            "--output-grid-template",
            grid,
            "--output-grid-template-level",
            "0",
            "--output-codec",
            "jpegxr",
            "--jpegxr-level",
            "0.7",
            "--residual-correction",
            correction,
            "--resume-fusion",
        ],
        required=[fused / "zarr.json"],
        env=primary_gpu_env,
        validate=lambda: require_complete_fusion(fused),
        input_identity=lambda: {
            "tiles": [
                source_tile_resume_identity(path, require_completion=True)
                for path in sorted(paths.deconv(name).glob("*.ome.zarr"))
            ],
            "positions": file_resume_identity(positions),
            "registration": file_resume_identity(registration),
            "correction": file_resume_identity(correction),
            "grid": source_tile_resume_identity(grid, require_completion=False),
        },
    )
    return fused
