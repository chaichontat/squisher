"""Resumable orchestration for the EDU 561/638/514 plus 405 workflow.

The workflow deliberately stops at the three human decisions required by
WORKFLOW_EDU.md. Rerunning the same command after editing the emitted gate
file resumes from the first incomplete stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable, Sequence

import numpy as np

from squisher_lightsheet._legacy.stitch_20x_tl_multiview import (
    TileMetadata,
    parse_ome_metadata,
)


EXPECTED_PYTHON = Path("/home/chaichontat/miniforge3/envs/multi/bin/python")
DEFAULT_PSF_571 = Path(
    "/home/chaichontat/nvme/lightsheet/20260606/"
    "psf_Image_57_571nm_zstrict/radial_symmetric_reskewed_rawscale_zcenter2_xy9_sumnorm.tif"
)
DEFAULT_PSF_638 = Path(
    "/home/chaichontat/nvme/lightsheet/20260606/"
    "psf_Image_57_648nm_std/radial_symmetric_reskewed_rawscale_zcenter2_xy9_sumnorm.tif"
)
DEFAULT_638_REFERENCE = Path(
    "/working/eduseg/20260808-prod2/234-3-514561638/"
    "reg-638-to-561/registration.json"
)
DEFAULT_638_REFERENCE_SHA256 = (
    "ee7820a860b1d247f8116d86fe98206cbcf19ee93957b35e88706179e1b5e4f3"
)
STARTING_MATRIX_514 = (
    "0.9999090828,-0.0083501824,-0.0029757332,"
    "0.0141554887,0.9999284579,-0.0096124636,"
    "0.0026881526,0.0099598448,0.9999495898"
)
STARTING_MATRIX_405 = (
    "1.0000085830688477,0.0029159076511859894,-0.0002120283606927842,"
    "0.0029159076511859894,1.0000085830688477,0.00018651569553185254,"
    "-0.0002120283606927842,0.00018651569553185254,1.0000001192092896"
)
DEFAULT_CELLPOSE_MODEL = Path("/working/cellpose-training/models/embryonicsheet")
# Materialized cross-registration windows overlap densely: one 24x960x960 output chunk of prod3-2 514 needed
# 35 GB of GPU memory unsplit. The budget splits the computation in x/y without changing the chunk's sources.
MATERIALIZED_FUSION_GPU_MEMORY_MAX_GIB = "18"


class ReviewRequired(RuntimeError):
    """Raised after publishing the next human-review gate."""

    def __init__(self, message: str, path: Path) -> None:
        super().__init__(f"{message}: {path}")


@dataclass(frozen=True)
class EduInputs:
    multi_dir: Path
    dapi_dir: Path
    multi_tiffs: tuple[Path, ...]
    dapi_tiffs: tuple[Path, ...]


@dataclass(frozen=True)
class EduPaths:
    root: Path
    multi: Path
    dapi: Path

    @property
    def logs(self) -> Path:
        return self.root / "run-logs"

    @property
    def stages(self) -> Path:
        return self.root / ".stages"

    def acquisition(self, label: str) -> Path:
        return self.dapi if label == "405" else self.multi

    def deconv(self, label: str) -> Path:
        return self.acquisition(label) / "deconvolved"

    def positions(self, label: str) -> Path:
        return self.acquisition(label) / "metadata.positions.json"

    def registration_dir(self, label: str) -> Path:
        return self.acquisition(label) / f"registration-{label}"

    def registration(self, label: str) -> Path:
        return self.registration_dir(label) / "registration.json"

    def registered_positions(self, label: str) -> Path:
        return self.registration_dir(label) / "registration.positions.json"

    def grid(self, label: str) -> Path:
        return self.acquisition(label) / f"grid-{label}.ome.zarr"

    def residual(self, label: str) -> Path:
        return self.acquisition(label) / f"fit-residual-{label}"

    def fusion_dir(self, label: str) -> Path:
        return self.multi / f"fusion-{label}"

    def fused(self, label: str) -> Path:
        channel = 1 if label == "638" else 0
        return self.fusion_dir(label) / f"fused.ch{channel}.ome.zarr"

    @property
    def manual_initializations(self) -> Path:
        """Dataset master file of curated cross-registration window initializations."""
        return self.root / "manual-initializations.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manual_initialization_args(paths: EduPaths, label: str) -> list[str | Path]:
    """Pass the dataset master file when it exists."""
    master = paths.manual_initializations
    if not master.is_file():
        return []
    return [
        "--manual-initializations",
        master,
        "--manual-initializations-channel",
        label,
    ]


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _ome_tiffs(directory: Path) -> tuple[Path, ...]:
    paths = tuple(sorted((*directory.glob("*.ome.tif"), *directory.glob("*.ome.tiff"))))
    partials = tuple(sorted(directory.glob(".*.ome.tif.*"))) + tuple(
        sorted(directory.glob(".*.ome.tiff.*"))
    )
    if partials:
        raise ValueError(f"Input transfer is incomplete; partial OME-TIFFs remain in {directory}")
    return paths


def _channel_count(metadata: TileMetadata) -> int:
    if metadata.axes == "CZYX":
        return int(metadata.shape[0])
    if metadata.axes == "ZCYX":
        return int(metadata.shape[1])
    if metadata.axes == "ZYX":
        return 1
    raise ValueError(f"Unsupported OME axes {metadata.axes!r} in {metadata.path}")


def _looks_like_405(directory: Path, metadata: TileMetadata) -> bool:
    text = " ".join((directory.name, *metadata.channels)).lower()
    return "405" in text or "dapi" in text


def _dataset_key(directory: Path) -> str:
    name = directory.name.lower()
    name = re.sub(r"(?:405|488|514|561|638)+", "", name)
    name = re.sub(r"[-_. ]+\d+$", "", name)
    return re.sub(r"[-_. ]+", "-", name).strip("-")


def discover_inputs(source: Path) -> EduInputs:
    """Find the three-channel acquisition and its unique sibling 405 series."""
    source = source.expanduser().resolve()
    if not source.is_file() or not source.name.lower().endswith((".ome.tif", ".ome.tiff")):
        raise ValueError(f"Expected one existing OME-TIFF path, got {source}")

    source_metadata = parse_ome_metadata(source)
    source_count = _channel_count(source_metadata)
    if source_count not in {1, 3}:
        raise ValueError(
            f"EDU input must identify the three-channel or 405 acquisition; "
            f"{source} has {source_count} channels"
        )

    candidates: list[tuple[Path, tuple[Path, ...], TileMetadata]] = []
    source_key = _dataset_key(source.parent)
    for directory in sorted(path for path in source.parent.parent.iterdir() if path.is_dir()):
        if _dataset_key(directory) != source_key:
            continue
        tiffs = _ome_tiffs(directory)
        if not tiffs:
            continue
        try:
            metadata = source_metadata if directory == source.parent else parse_ome_metadata(tiffs[0])
        except (OSError, ValueError):
            continue
        candidates.append((directory.resolve(), tiffs, metadata))

    multi = [candidate for candidate in candidates if _channel_count(candidate[2]) == 3]
    dapi = [
        candidate
        for candidate in candidates
        if _channel_count(candidate[2]) == 1 and _looks_like_405(candidate[0], candidate[2])
    ]
    if source_count == 3:
        multi = [candidate for candidate in multi if candidate[0] == source.parent]
    else:
        dapi = [candidate for candidate in dapi if candidate[0] == source.parent]
    if len(multi) != 1 or len(dapi) != 1:
        raise ValueError(
            "Could not identify exactly one three-channel acquisition and one sibling 405 "
            f"acquisition under {source.parent.parent}; found multi={len(multi)}, 405={len(dapi)}"
        )

    multi_dir, multi_tiffs, multi_metadata = multi[0]
    dapi_dir, dapi_tiffs, _ = dapi[0]
    track_channels = {channel for track in multi_metadata.tracks for channel in track.channels}
    if track_channels != {0, 1, 2}:
        raise ValueError(f"Three-channel track metadata does not cover channels 0,1,2: {multi_metadata.tracks}")
    return EduInputs(
        multi_dir=multi_dir,
        dapi_dir=dapi_dir,
        multi_tiffs=multi_tiffs,
        dapi_tiffs=dapi_tiffs,
    )


def _gpu_uuids() -> tuple[str, ...]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        uuid, memory = (piece.strip() for piece in line.split(",", 1))
        rows.append((int(memory), uuid))
    if not rows:
        raise RuntimeError("No CUDA GPUs reported by nvidia-smi")
    return tuple(uuid for _memory, uuid in sorted(rows, reverse=True))


def _require_multi_environment() -> None:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"EDU workflow must run in the multi environment: expected {EXPECTED_PYTHON}, "
            f"got {sys.executable}"
        )


def _require_complete_fusion(path: Path) -> None:
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


class StageRunner:
    def __init__(self, paths: EduPaths, *, dry_run: bool) -> None:
        self.paths = paths
        self.dry_run = dry_run
        paths.logs.mkdir(parents=True, exist_ok=True)
        paths.stages.mkdir(parents=True, exist_ok=True)

    def _marker(self, name: str) -> Path:
        return self.paths.stages / f"{name}.json"

    def command(
        self,
        name: str,
        argv: Sequence[str | Path],
        *,
        required: Sequence[Path],
        env: dict[str, str] | None = None,
        validate: Callable[[], None] | None = None,
    ) -> None:
        command = [str(value) for value in argv]
        identity = _json_sha256({"command": command, "environment": env or {}})
        marker = self._marker(name)
        if marker.is_file():
            payload = json.loads(marker.read_text())
            if payload.get("identity") != identity:
                raise ValueError(f"Stage {name} command changed; refusing to reuse {marker}")
            if all(path.exists() for path in required):
                if validate is not None:
                    validate()
                print(f"[edu] reuse {name}", flush=True)
                return
        print("[edu] " + " ".join(command), flush=True)
        if self.dry_run:
            return
        run_env = os.environ.copy()
        run_env.update(env or {})
        run_env.setdefault("PYTHONUNBUFFERED", "1")
        log_path = self.paths.logs / f"{name}.log"
        with log_path.open("a", buffering=1) as log:
            log.write(f"\n{datetime.now().astimezone().isoformat()} {' '.join(command)}\n")
            process = subprocess.Popen(
                command,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Stage {name} completed without required outputs: {missing}")
        if validate is not None:
            validate()
        _atomic_json(
            marker,
            {
                "artifact_type": "squisher_lightsheet.edu_stage.v1",
                "name": name,
                "identity": identity,
                "command": command,
                "required_outputs": [str(path.resolve()) for path in required],
                "completed_at": datetime.now().astimezone().isoformat(),
            },
        )

    def action(
        self,
        name: str,
        identity_payload: object,
        action: Callable[[], None],
        *,
        required: Sequence[Path],
        validate: Callable[[], None] | None = None,
    ) -> None:
        identity = _json_sha256(identity_payload)
        marker = self._marker(name)
        if marker.is_file() and json.loads(marker.read_text()).get("identity") == identity:
            if all(path.exists() for path in required):
                if validate is not None:
                    validate()
                print(f"[edu] reuse {name}", flush=True)
                return
        print(f"[edu] {name}", flush=True)
        if self.dry_run:
            return
        action()
        if validate is not None:
            validate()
        if not all(path.exists() for path in required):
            raise RuntimeError(f"Action {name} did not create all required outputs")
        _atomic_json(
            marker,
            {
                "artifact_type": "squisher_lightsheet.edu_stage.v1",
                "name": name,
                "identity": identity,
                "completed_at": datetime.now().astimezone().isoformat(),
            },
        )


def _write_metadata_grid(registration: Path, tile_dir: Path, output: Path) -> None:
    payload = json.loads(registration.read_text())
    records = payload.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Registration has no tiles: {registration}")
    spacing = np.asarray([payload["spacing_um"][axis] for axis in "zyx"], dtype=np.float64)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid registration spacing in {registration}: {spacing}")

    starts: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    for record in records:
        source = tile_dir / Path(record["path"]).name
        if source.resolve() != Path(record["path"]).resolve():
            raise ValueError(f"Registration does not own current source {source}: {record['path']}")
        shape = np.asarray(record["shape"][-3:], dtype=np.float64)
        scale = np.asarray([record["stage_scale_um"][axis] for axis in "zyx"])
        translation = np.asarray([record["stage_translation_um"][axis] for axis in "zyx"])
        affine = np.asarray(record["registered_affine"]["matrix"], dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError(f"Invalid registered affine for {record['path']}")
        corners = []
        for z in (0.0, shape[0]):
            for y in (0.0, shape[1]):
                for x in (0.0, shape[2]):
                    physical = np.append(translation + scale * (z, y, x), 1.0)
                    corners.append((affine @ physical)[:3])
        transformed = np.asarray(corners)
        starts.append(transformed.min(axis=0))
        stops.append(transformed.max(axis=0))
    origin = np.min(starts, axis=0)
    shape = np.ceil((np.max(stops, axis=0) - origin) / spacing).astype(np.int64)
    if np.any(shape <= 0):
        raise ValueError(f"Invalid grid shape derived from {registration}: {shape}")

    root = {
        "attributes": {
            "metadata_only_template": True,
            "source_registration": str(registration.resolve()),
            "source_registration_sha256": _sha256(registration),
            "ome": {
                "multiscales": [
                    {
                        "axes": [
                            {"name": axis, "type": "space", "unit": "micrometer"}
                            for axis in "zyx"
                        ],
                        "datasets": [
                            {
                                "coordinateTransformations": [
                                    {"scale": spacing.tolist(), "type": "scale"},
                                    {"translation": origin.tolist(), "type": "translation"},
                                ],
                                "path": "0",
                            }
                        ],
                        "name": "/",
                    }
                ],
                "version": "0.5",
            },
        },
        "zarr_format": 3,
        "node_type": "group",
    }
    level = {
        "shape": shape.tolist(),
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [24, 960, 960]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [1, 960, 960],
                    "codecs": [
                        {"name": "squisher.jpegxr", "configuration": {"level": 0.7}},
                        {"name": "crc32c"},
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
        ],
        "attributes": {},
        "dimension_names": list("zyx"),
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    if output.exists():
        if (
            json.loads((output / "zarr.json").read_text()) != root
            or json.loads((output / "0/zarr.json").read_text()) != level
        ):
            raise ValueError(f"Existing metadata grid differs from current registration: {output}")
        return
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    (temporary / "0").mkdir(parents=True)
    _atomic_json(temporary / "zarr.json", root)
    _atomic_json(temporary / "0/zarr.json", level)
    temporary.replace(output)


def _gate_numbers(path: Path, keys: Sequence[str], *, message: str) -> dict[str, float]:
    template = {
        "artifact_type": "squisher_lightsheet.edu_thresholds.v1",
        "thresholds": {key: None for key in keys},
    }
    if not path.exists():
        _atomic_json(path, template)
        raise ReviewRequired(message, path)
    payload = json.loads(path.read_text())
    values = payload.get("thresholds")
    if not isinstance(values, dict) or set(values) != set(keys):
        raise ReviewRequired(message, path)
    if any(not isinstance(values[key], (int, float)) or values[key] <= 0 for key in keys):
        raise ReviewRequired(message, path)
    return {key: float(values[key]) for key in keys}


def _gate_approval(path: Path, identity: dict[str, str], *, message: str) -> None:
    expected = {
        "artifact_type": "squisher_lightsheet.edu_approval.v1",
        "approved": False,
        "identity": identity,
    }
    if not path.exists():
        _atomic_json(path, expected)
        raise ReviewRequired(message, path)
    payload = json.loads(path.read_text())
    if payload != expected | {"approved": True}:
        if payload.get("identity") != identity:
            raise ValueError(f"Approval identity no longer matches reviewed artifacts: {path}")
        raise ReviewRequired(message, path)


def _reference_638_matrix(path: Path) -> str:
    if _sha256(path) != DEFAULT_638_REFERENCE_SHA256:
        raise ValueError(f"Pinned prod2 638 calibration hash mismatch: {path}")
    payload = json.loads(path.read_text())
    contract = payload.get("transform_contract", {})
    if (
        payload.get("artifact_type")
        != "squisher_lightsheet.global_channel_affine_registration.v1"
        or contract.get("source_space") != "638_stage_um"
        or contract.get("target_space") != "561_registered_um"
    ):
        raise ValueError(f"Pinned calibration has the wrong transform direction: {path}")
    matrix = np.asarray(
        payload["diagnostics"]["global_channel_affine"]["channel_affine_um_zyx_homogeneous"],
        dtype=np.float64,
    )
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)) or np.linalg.det(matrix[:3, :3]) == 0:
        raise ValueError(f"Pinned calibration is not a finite invertible affine: {path}")
    return ",".join(str(value) for value in matrix[:3, :3].ravel())


def _common_pyramid_level(*directories: Path) -> int:
    maxima = []
    for directory in directories:
        source = next(iter(sorted(directory.glob("*.ome.zarr"))), None)
        if source is None:
            raise ValueError(f"No OME-Zarr tiles in {directory}")
        metadata = json.loads((source / "zarr.json").read_text())
        datasets = metadata.get("attributes", {}).get("ome", {}).get("multiscales", [{}])[0].get(
            "datasets", []
        )
        maxima.append(len(datasets) - 1)
    return min(maxima)


def _preprocess(
    runner: StageRunner,
    paths: EduPaths,
    *,
    label: str,
    tiffs: Sequence[Path],
    channels: int,
    psfs: Sequence[Path],
    slab_depth: int,
    all_gpu_env: dict[str, str],
    primary_gpu_env: dict[str, str],
) -> None:
    base = paths.acquisition(label)
    basic = base / "basic"
    scale = base / "scale"
    deconv = paths.deconv(label)
    profile_label = "multi" if channels == 3 else "405"
    profiles = [basic / f"{profile_label}-ch{channel}.pkl" for channel in range(channels)]
    runner.command(
        f"{label}-basic",
        [
            "squisher-deconv",
            "basic",
            *tiffs,
            "--out-dir",
            basic,
            "--label",
            profile_label,
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
        env=primary_gpu_env,
    )
    common: list[str | Path] = ["--channels", str(channels)]
    for psf in psfs:
        common += ["--psf", psf]
    for profile in profiles:
        common += ["--basic", profile]
    common += ["--iter", "1", "--devices", "auto", "--queue-depth", "2"]
    runner.command(
        f"{label}-scale",
        [
            "squisher-deconv",
            "sample-scale",
            *tiffs,
            "--out-dir",
            scale,
            *common,
            "--planes",
            "100",
        ],
        required=[scale / "scaling.json"],
        env=all_gpu_env,
    )
    runner.command(
        f"{label}-deconvolution",
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
            str(slab_depth),
            "--jpegxr-level",
            "0.7",
            "--resume",
        ],
        required=[deconv],
        env=all_gpu_env,
        validate=lambda: _require_complete_deconvolution(deconv, len(tiffs)),
    )


def _fuse_source(
    runner: StageRunner,
    paths: EduPaths,
    *,
    label: str,
    source_dir: Path,
    position: Path,
    registration: Path,
    channel: int,
    correction: Path | None,
    primary_gpu_env: dict[str, str],
    gpu_memory_max_gib: str | None = None,
) -> None:
    argv: list[str | Path] = [
        "lightsheet",
        "fuse",
        source_dir,
        "--position-input",
        position,
        "--registration-input",
        registration,
        "--output",
        paths.fusion_dir(label),
        "--channel",
        str(channel),
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
        paths.grid("561") if label == "561" else paths.fused("561"),
        "--output-grid-template-level",
        "0",
        "--output-codec",
        "jpegxr",
        "--jpegxr-level",
        "0.7",
        "--resume-fusion",
    ]
    if correction is not None:
        argv += ["--residual-correction", correction]
    if gpu_memory_max_gib is not None:
        argv += ["--fusion-gpu-memory-max-gib", gpu_memory_max_gib]
    runner.command(
        f"fuse-{label}",
        argv,
        required=[paths.fused(label) / "zarr.json"],
        env=primary_gpu_env,
        validate=lambda: _require_complete_fusion(paths.fused(label)),
    )


def _run_cross_channel_recovery(
    runner: StageRunner,
    paths: EduPaths,
    *,
    label: str,
    channel: int,
    rough_transform: str,
    fit_transform: str,
    starting_matrix: str,
    minimum_gradient_ncc: float,
    fixed_threshold: float,
    common_level: int,
    primary_gpu_env: dict[str, str],
) -> Path:
    rough = paths.multi / f"xreg-{label}-global-phase"
    method6 = paths.multi / f"xreg-{label}-method6"
    native = paths.multi / f"xreg-{label}-native-recovery"
    complete = paths.multi / f"xreg-{label}-complete-recovery"
    runner.command(
        f"global-phase-{label}",
        [
            "lightsheet",
            "cross-register",
            "global-phase",
            "--fixed-position",
            paths.registered_positions("561"),
            "--moving-position",
            paths.registered_positions(label),
            "--fixed-tile-dir",
            paths.deconv("561"),
            "--moving-tile-dir",
            paths.deconv(label),
            "--output-dir",
            rough,
            "--output-position",
            rough / "global-phase.positions.json",
            "--fixed-channel",
            "0",
            "--moving-channel",
            str(channel),
            "--level",
            str(common_level),
            "--z-slab-planes",
            "32",
            "--fixed-intensity-transform",
            "log1p",
            "--moving-intensity-transform",
            rough_transform,
            "--highpass-sigma-px",
            "8",
        ],
        required=[rough / "global-phase.positions.json", rough / "global-phase.summary.json"],
        env=primary_gpu_env,
    )
    runner.command(
        f"method6-{label}",
        [
            "lightsheet",
            "cross-register",
            "method6",
            "--fixed-position",
            paths.registered_positions("561"),
            "--moving-position",
            rough / "global-phase.positions.json",
            "--moving-source-position",
            paths.registration(label),
            "--fixed-fused",
            paths.fused("561"),
            "--output-dir",
            method6,
            "--output-registration",
            method6 / "registration.json",
            "--moving-channel",
            str(channel),
            "--fit-intensity-transform",
            fit_transform,
            "--starting-affine-matrix-zyx",
            starting_matrix,
            "--min-corr",
            "0.15",
            "--min-grad-ncc",
            str(minimum_gradient_ncc),
            "--fixed-mask-threshold",
            str(fixed_threshold),
            "--source-label",
            label,
            "--target-label",
            "561",
            "--workers",
            "1",
            "--devices",
            "0",
            "--resume",
        ],
        required=[method6 / "fused_fixed_method8_summary.json"],
        env=primary_gpu_env,
    )
    runner.command(
        f"recover-native-{label}",
        [
            "lightsheet",
            "cross-register",
            "recover-native",
            "--summary",
            method6 / "fused_fixed_method8_summary.json",
            "--output-dir",
            native,
            "--adjacency-json",
            paths.registration_dir(label) / "registration.measurements.json",
            "--devices",
            "0",
            "--resume",
            *_manual_initialization_args(paths, label),
        ],
        required=[native / "fused_fixed_method8_summary.json"],
        env=primary_gpu_env,
    )
    runner.command(
        f"complete-recovery-{label}",
        [
            "lightsheet",
            "cross-register",
            "complete-recovery",
            "--native-summary",
            native / "fused_fixed_method8_summary.json",
            "--output-dir",
            complete,
        ],
        required=[complete / "fused_fixed_method8_summary.json", complete / "manifest.json"],
    )
    return complete / "fused_fixed_method8_summary.json"


def _run_segmentation(
    runner: StageRunner,
    paths: EduPaths,
    *,
    fusion_identity: dict[str, str],
    primary_gpu_env: dict[str, str],
) -> Path:
    import zarr

    shape = tuple(
        int(value) for value in zarr.open_group(paths.fused("561"), mode="r")["0"].shape
    )
    if shape[1] < 2048 or shape[2] < 2048:
        raise ValueError(f"Reference fusion is smaller than the 2048x2048 Cellpose subset: {shape}")
    roi = [[0, shape[0]]]
    for size in shape[1:]:
        start = (size - 2048) // 2
        roi.append([start, start + 2048])

    cellpose = paths.root / "cellpose-subset"
    source_manifest = cellpose / "registered-sources.json"
    config = cellpose / "config.json"
    source_payload = {
        "artifact_type": "squisher_segment.registered_ome_input.v1",
        "channels": ["405", "561", "638"],
        "sources": [
            {
                "channel": label,
                "path": str(paths.fused(label)),
                "root_metadata_sha256": fusion_identity[label],
            }
            for label in ("405", "561", "638")
        ],
        "source_roi_zyx": roi,
        "shape": [shape[0], 2048, 2048, 3],
        "dtype": "uint16",
    }
    config_payload = {
        "backend": "sam",
        "diameter": 30,
        "flow3D_smooth": 1.0,
        "ortho_weights": [1.0, 1.0, 1.0],
        "pretrained_model": str(DEFAULT_CELLPOSE_MODEL),
        "pretrained_model_ortho": str(DEFAULT_CELLPOSE_MODEL),
    }

    runtime_report = cellpose / "runtime-plan.json"

    def prepare_runtime() -> None:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = primary_gpu_env["CUDA_VISIBLE_DEVICES"]
        try:
            import numba
            import torch

            from squisher_segment.segment.model_artifacts import plan_path_for_device
            from squisher_segment.segment.train import build_trt_engine
            from squisher_segment.segmentation.distributed import distributed_postproc

            del distributed_postproc
            device = torch.device("cuda:0")
            device_name = torch.cuda.get_device_name(device)
            plan = plan_path_for_device(DEFAULT_CELLPOSE_MODEL, device_name)
            if not plan.is_file() or plan.stat().st_mtime_ns <= DEFAULT_CELLPOSE_MODEL.stat().st_mtime_ns:
                plan = build_trt_engine(
                    model_path=DEFAULT_CELLPOSE_MODEL,
                    bsize=256,
                    device=device,
                    batch_size=1,
                    backend="sam",
                )
            if not plan.is_file() or plan.stat().st_mtime_ns <= DEFAULT_CELLPOSE_MODEL.stat().st_mtime_ns:
                raise RuntimeError(f"TensorRT plan is missing or stale after build: {plan}")
            _atomic_json(
                runtime_report,
                {
                    "artifact_type": "squisher_lightsheet.edu_cellpose_runtime.v1",
                    "gpu_uuid": primary_gpu_env["CUDA_VISIBLE_DEVICES"],
                    "device_name": device_name,
                    "model": str(DEFAULT_CELLPOSE_MODEL.resolve()),
                    "model_sha256": _sha256(DEFAULT_CELLPOSE_MODEL),
                    "plan": str(plan.resolve()),
                    "plan_sha256": _sha256(plan),
                    "numba_version": numba.__version__,
                },
            )
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    model_stat = DEFAULT_CELLPOSE_MODEL.stat()

    def validate_runtime() -> None:
        payload = json.loads(runtime_report.read_text())
        if (
            payload.get("gpu_uuid") != primary_gpu_env["CUDA_VISIBLE_DEVICES"]
            or not Path(payload["plan"]).is_file()
        ):
            raise ValueError(f"Cellpose runtime identity is no longer valid: {runtime_report}")

    runner.action(
        "cellpose-runtime",
        {
            "model": str(DEFAULT_CELLPOSE_MODEL.resolve()),
            "model_size": model_stat.st_size,
            "model_mtime_ns": model_stat.st_mtime_ns,
            "gpu_uuid": primary_gpu_env["CUDA_VISIBLE_DEVICES"],
        },
        prepare_runtime,
        required=[runtime_report],
        validate=validate_runtime,
    )

    def write_inputs() -> None:
        _atomic_json(source_manifest, source_payload)
        _atomic_json(config, config_payload)

    runner.action(
        "cellpose-inputs",
        {"sources": source_payload, "config": config_payload},
        write_inputs,
        required=[source_manifest, config],
    )
    runner.command(
        "cellpose-segmentation",
        [
            "squisher-segment",
            "segment",
            "run",
            source_manifest,
            "--channels",
            "405,561,638",
            "--config",
            config,
            "--workers-per-device",
            "2",
            "--threads-per-worker",
            "1",
            "--target-nz",
            "1",
            "--target-ny",
            "4",
            "--target-nx",
            "3",
            "--nonempty-threshold",
            "1000",
            "--no-cellpose-only",
            "--stagger-seconds",
            "0",
        ],
        required=[cellpose / "output_segmentation-sam.done", cellpose / "output_segmentation-sam.zarr"],
        env=primary_gpu_env,
    )
    runner.command(
        "cellpose-postprocess",
        [
            "squisher-segment",
            "postproc",
            "run",
            cellpose / "output_segmentation-sam.zarr",
            "--output-path",
            cellpose / "postproc.zarr",
            "--sigma",
            "1,2,2",
            "--v-min",
            "500",
            "--margin",
            "30",
            "--workers-per-gpu",
            "1",
        ],
        required=[cellpose / "postproc.zarr"],
        env=primary_gpu_env,
    )
    expected_label_shape = tuple(source_payload["shape"][:3])
    for output in (cellpose / "output_segmentation-sam.zarr", cellpose / "postproc.zarr"):
        labels = zarr.open_array(output, mode="r")
        if tuple(labels.shape) != expected_label_shape or str(labels.dtype) != "uint32":
            raise ValueError(f"Invalid Cellpose labels {output}: {labels.shape} {labels.dtype}")
    for mode, output in (
        ("z", cellpose / "extract-z512-full"),
        ("ortho", cellpose / "extract-o512-full"),
    ):
        argv: list[str | Path] = [
            "squisher-segment",
            "extract",
            source_manifest,
            "--mode",
            mode,
            "--out",
            output,
            "--n",
            "50",
            "--upscale",
            "1",
            "--channels",
            "405,561,638",
            "--threads",
            "8",
            "--seed",
            "42",
            "--label",
            paths.root.name,
            "--masks",
            cellpose / "postproc.zarr",
        ]
        if mode == "z":
            argv += ["--dz", "1"]
        else:
            argv += ["--anisotropy", "1", "--ortho-depth", "512"]
        runner.command(f"cellpose-export-{mode}", argv, required=[output])
    export_marker = cellpose / "export.complete.json"

    def validate_exports() -> None:
        counts = {}
        for directory in (cellpose / "extract-z512-full", cellpose / "extract-o512-full"):
            images = sorted(
                path for path in directory.glob("*.tif") if not path.name.endswith("_masks.tif")
            )
            masks = {path for path in directory.glob("*_masks.tif")}
            expected_masks = {
                image.with_name(f"{image.stem}_masks.tif") for image in images
            }
            if len(images) != 50 or masks != expected_masks:
                raise ValueError(
                    f"Incomplete paired export in {directory}: "
                    f"images={len(images)}, masks={len(masks)}"
                )
            counts[directory.name] = len(images)
        if export_marker.is_file():
            payload = json.loads(export_marker.read_text())
            if payload.get("counts") != counts:
                raise ValueError(f"Cellpose export marker does not match outputs: {export_marker}")

    runner.action(
        "cellpose-export-complete",
        {"source_identity": fusion_identity, "seed": 42, "count": 50},
        lambda: _atomic_json(
            export_marker,
            {
                "artifact_type": "squisher_lightsheet.edu_cellpose_export.v1",
                "input": str(source_manifest.resolve()),
                "masks": str((cellpose / "postproc.zarr").resolve()),
                "channels": ["405", "561", "638"],
                "seed": 42,
                "counts": {"extract-z512-full": 50, "extract-o512-full": 50},
            },
        ),
        required=[export_marker],
        validate=validate_exports,
    )
    return cellpose


def run_edu_workflow(
    source: Path,
    *,
    output_root: Path | None = None,
    dry_run: bool = False,
    allow_disconnected_561: bool = False,
    allow_disconnected_514: bool = False,
    allow_disconnected_405: bool = False,
) -> Path:
    """Run or resume WORKFLOW_EDU.md from one representative OME-TIFF."""
    _require_multi_environment()

    inputs = discover_inputs(source)
    if output_root is None:
        prefix = os.path.commonprefix((inputs.multi_dir.name, inputs.dapi_dir.name)).rstrip(
            "-_. "
        )
        output_root = inputs.multi_dir.parent / f"{prefix or inputs.multi_dir.name}-edu-workflow"
    root = output_root.expanduser().resolve()
    paths = EduPaths(root=root, multi=root / "multi", dapi=root / "405")
    paths.multi.mkdir(parents=True, exist_ok=True)
    paths.dapi.mkdir(parents=True, exist_ok=True)
    gpu_uuids = _gpu_uuids()
    manifest = {
        "artifact_type": "squisher_lightsheet.edu_workflow.v1",
        "multi_input_dir": str(inputs.multi_dir),
        "dapi_input_dir": str(inputs.dapi_dir),
        "multi_tile_count": len(inputs.multi_tiffs),
        "dapi_tile_count": len(inputs.dapi_tiffs),
        "channel_mapping": {"561": 0, "638": 1, "514": 2, "405": 0},
        "output_root": str(paths.root),
        "gpu_uuids": list(gpu_uuids),
        "workflow_document": "/home/chaichontat/squisher/WORKFLOW_EDU.md",
    }
    manifest_path = root / "workflow.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Existing workflow manifest identifies different inputs: {manifest_path}")
    _atomic_json(manifest_path, manifest)
    runner = StageRunner(paths, dry_run=dry_run)
    all_gpu_env = {"CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids)}
    primary_gpu_env = {"CUDA_VISIBLE_DEVICES": gpu_uuids[0]}

    _preprocess(
        runner,
        paths,
        label="multi",
        tiffs=inputs.multi_tiffs,
        channels=3,
        psfs=(DEFAULT_PSF_571, DEFAULT_PSF_638, DEFAULT_PSF_571),
        slab_depth=100,
        all_gpu_env=all_gpu_env,
        primary_gpu_env=primary_gpu_env,
    )
    _preprocess(
        runner,
        paths,
        label="405",
        tiffs=inputs.dapi_tiffs,
        channels=1,
        psfs=(DEFAULT_PSF_571,),
        slab_depth=50,
        all_gpu_env=all_gpu_env,
        primary_gpu_env=primary_gpu_env,
    )

    for label, channels in (("multi", "0,1,2"), ("405", "0")):
        review = paths.acquisition(label) / "native-review"
        runner.command(
            f"{label}-native-review",
            [
                "lightsheet",
                "ome-metadata-dumb-stitch",
                "--input-dir",
                f"R={paths.deconv(label)}",
                "--output-dir",
                review,
                "--channels",
                channels,
                "--level",
                "0",
                "--output-prefix",
                label,
                "--write-tiff",
            ],
            required=[review / f"{label}_omeMetadata_noBlend_noLines_manifest.json"],
        )
    thresholds = _gate_numbers(
        root / "native-thresholds.json",
        ("561", "638", "514", "405"),
        message="Review the native-intensity TIFFs and fill all thresholds",
    )

    for acquisition in ("multi", "405"):
        runner.command(
            f"{acquisition}-positions",
            [
                "lightsheet",
                "single-position",
                "--input-dir",
                inputs.dapi_dir if acquisition == "405" else inputs.multi_dir,
                "--output",
                paths.positions(acquisition),
                "--side",
                "L",
            ],
            required=[paths.positions(acquisition)],
        )
    for label, channel in (("561", 0), ("514", 2), ("405", 0)):
        register_command = [
                "lightsheet-stitch",
                "register",
                "--position-json",
                paths.positions(label),
                "--zarr-dir",
                paths.deconv(label),
                "--output-dir",
                paths.registration_dir(label),
                "--threshold",
                str(thresholds[label]),
                "--channel",
                str(channel),
                "--device",
                "0",
                "--z-chunks",
                "6",
        ]
        if label == "561" and allow_disconnected_561:
            register_command.append("--allow-disconnected")
        if label == "514" and allow_disconnected_514:
            register_command.append("--allow-disconnected")
        if label == "405" and allow_disconnected_405:
            register_command.append("--allow-disconnected")
        runner.command(
            f"register-{label}",
            register_command,
            required=[paths.registration(label), paths.registered_positions(label)],
            env=primary_gpu_env,
        )

    for label, registration_label, channel in (
        ("561", "561", 0),
        ("638", "561", 1),
        ("514", "514", 2),
        ("405", "405", 0),
    ):
        grid = paths.grid(label)
        registration = paths.registration(registration_label)
        runner.action(
            f"grid-{label}",
            {"registration_sha256": _sha256(registration), "label": label},
            lambda grid=grid, registration=registration, label=label: _write_metadata_grid(
                registration, paths.deconv(label), grid
            ),
            required=[grid / "zarr.json", grid / "0/zarr.json"],
        )
        residual = paths.residual(label)
        runner.command(
            f"fit-residual-{label}",
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
                str(channel),
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
        )

    _fuse_source(
        runner,
        paths,
        label="561",
        source_dir=paths.deconv("561"),
        position=paths.registered_positions("561"),
        registration=paths.registration("561"),
        channel=0,
        correction=paths.residual("561") / "correction.json",
        primary_gpu_env=primary_gpu_env,
    )
    threshold_review = root / "fused-561-threshold-review.ome.tif"
    runner.command(
        "fused-561-threshold-review",
        [
            "lightsheet",
            "threshold-review",
            "--fixed-fused",
            paths.fused("561"),
            "--output",
            threshold_review,
            "--level",
            "2",
        ],
        required=[threshold_review],
    )
    fused_threshold = _gate_numbers(
        root / "fused-561-threshold.json",
        ("561",),
        message="Review the fused-561 native TIFF and fill its mask threshold",
    )["561"]

    refit_638 = paths.multi / "xreg-638"
    runner.command(
        "refit-638",
        [
            "lightsheet",
            "cross-register",
            "method6",
            "--fixed-position",
            paths.registered_positions("561"),
            "--moving-position",
            paths.registered_positions("561"),
            "--moving-source-position",
            paths.registration("561"),
            "--fixed-fused",
            paths.fused("561"),
            "--output-dir",
            refit_638,
            "--output-registration",
            refit_638 / "registration.json",
            "--moving-channel",
            "1",
            "--fit-intensity-transform",
            "log1p",
            "--starting-affine-matrix-zyx",
            _reference_638_matrix(DEFAULT_638_REFERENCE),
            "--min-corr",
            "0.15",
            "--min-grad-ncc",
            "0.24",
            "--fixed-mask-threshold",
            str(fused_threshold),
            "--source-label",
            "638",
            "--target-label",
            "561",
            "--workers",
            "1",
            "--devices",
            "0",
            "--resume",
        ],
        required=[refit_638 / "registration.json", refit_638 / "fused_fixed_method8_summary.json"],
        env=primary_gpu_env,
    )
    _fuse_source(
        runner,
        paths,
        label="638",
        source_dir=paths.deconv("638"),
        position=paths.registered_positions("561"),
        registration=refit_638 / "registration.json",
        channel=1,
        correction=paths.residual("638") / "correction.json",
        primary_gpu_env=primary_gpu_env,
    )

    common_level = _common_pyramid_level(paths.deconv("561"), paths.deconv("405"))
    recoveries = {
        "514": _run_cross_channel_recovery(
            runner,
            paths,
            label="514",
            channel=2,
            rough_transform="log1p",
            fit_transform="log1p",
            starting_matrix=STARTING_MATRIX_514,
            minimum_gradient_ncc=0.24,
            fixed_threshold=fused_threshold,
            common_level=common_level,
            primary_gpu_env=primary_gpu_env,
        ),
        "405": _run_cross_channel_recovery(
            runner,
            paths,
            label="405",
            channel=0,
            rough_transform="identity",
            fit_transform="linear",
            starting_matrix=STARTING_MATRIX_405,
            minimum_gradient_ncc=0.20,
            fixed_threshold=fused_threshold,
            common_level=common_level,
            primary_gpu_env=primary_gpu_env,
        ),
    }

    prefusion_qc = paths.multi / "qc-prefusion"
    runner.command(
        "prefusion-qc",
        [
            "lightsheet",
            "cross-register",
            "prefusion-qc",
            "--summary",
            f"514={recoveries['514']}",
            "--summary",
            f"405={recoveries['405']}",
            "--fixed-volume",
            paths.fused("561"),
            "--fixed-threshold",
            str(fused_threshold),
            "--output-dir",
            prefusion_qc,
        ],
        required=[prefusion_qc],
    )
    for label in ("514", "405"):
        runner.command(
            f"interpolated-window-qc-{label}",
            [
                "lightsheet",
                "cross-register",
                "interpolated-window-qc",
                "--summary",
                recoveries[label],
                "--moving-label",
                label,
                "--fixed-label",
                "561",
                "--dataset-label",
                root.name,
                "--output",
                prefusion_qc / f"interpolated-{label}.png",
            ],
            required=[prefusion_qc / f"interpolated-{label}.png"],
        )
    _gate_approval(
        root / "prefusion-approval.json",
        {label: _sha256(summary) for label, summary in recoveries.items()},
        message="Review pre-fusion holes and interpolated windows, then set approved=true",
    )

    for label, channel in (("514", 2), ("405", 0)):
        materialized = paths.multi / f"materialized-{label}"
        try:
            _require_complete_fusion(paths.fused(label))
        except (FileNotFoundError, ValueError):
            runner.command(
                f"materialize-{label}",
                [
                    "lightsheet",
                    "fused-fixed-materialize-overlap",
                    "--moving-position",
                    paths.registered_positions(label),
                    "--moving-source-registration",
                    paths.registration(label),
                    "--moving-source-dir",
                    paths.deconv(label),
                    "--source-summary",
                    recoveries[label],
                    "--output-dir",
                    materialized,
                    "--source-channel",
                    str(channel),
                    "--residual-correction",
                    paths.residual(label) / "correction.json",
                    "--core-shape-zyx",
                    "480,480,480",
                    "--window-shape-zyx",
                    "528,528,528",
                    "--level-factor-zyx",
                    "1,1,1",
                    "--output-codec",
                    "zstd",
                    "--zstd-level",
                    "3",
                    "--workers",
                    "4",
                    "--resume",
                ],
                required=[
                    materialized / "materialized_tiles",
                    materialized / "fused_fixed_materialized_chunks.positions.json",
                    materialized / "fused_fixed_materialized_chunks.registration.json",
                    materialized / "fused_fixed_materialized_chunks.summary.json",
                ],
                env=primary_gpu_env,
            )
            _fuse_source(
                runner,
                paths,
                label=label,
                source_dir=materialized / "materialized_tiles",
                position=materialized / "fused_fixed_materialized_chunks.positions.json",
                registration=materialized / "fused_fixed_materialized_chunks.registration.json",
                channel=0,
                correction=None,
                primary_gpu_env=primary_gpu_env,
                gpu_memory_max_gib=MATERIALIZED_FUSION_GPU_MEMORY_MAX_GIB,
            )
        else:
            print(f"[edu] reuse complete fuse-{label}", flush=True)
        if materialized.exists() and not dry_run:
            _require_complete_fusion(paths.fused(label))
            shutil.rmtree(materialized)
            print(f"[edu] removed successful intermediate {materialized}", flush=True)

    final_qc = paths.multi / "qc-final"
    middle_args: list[str | Path] = ["lightsheet", "fused-middle-z-qc"]
    for label in ("561", "638", "514", "405"):
        middle_args += ["--channel", f"{label}={paths.fused(label)}"]
    middle_args += [
        "--fixed-volume",
        paths.fused("561"),
        "--output-dir",
        final_qc,
        "--level",
        "2",
        "--contact-sheet-name",
        "combined-middle-z.png",
    ]
    runner.command(
        "final-middle-z-qc",
        middle_args,
        required=[final_qc / "combined-middle-z.png"],
    )
    for label in ("638", "514", "405"):
        overlay = final_qc / f"561-vs-{label}"
        runner.command(
            f"final-overlay-{label}",
            [
                "lightsheet",
                "fused-xyz-overlay-qc",
                "--reference-zarr",
                paths.fused("561"),
                "--moving-zarr",
                paths.fused(label),
                "--output-dir",
                overlay,
                "--level",
                "0",
                "--thumb-level",
                "2",
                "--panel-size",
                "512",
            ],
            required=[overlay],
        )

    fusion_identity = {
        label: _sha256(paths.fused(label) / "zarr.json") for label in ("405", "561", "638")
    }
    _gate_approval(
        root / "segmentation-approval.json",
        fusion_identity,
        message="Review final alignment QC, then set approved=true to start Cellpose",
    )
    cellpose = _run_segmentation(
        runner,
        paths,
        fusion_identity=fusion_identity,
        primary_gpu_env=primary_gpu_env,
    )
    completion = root / "complete.json"
    runner.action(
        "complete",
        {"workflow": manifest, "fusions": fusion_identity},
        lambda: _atomic_json(
            completion,
            {
                "artifact_type": "squisher_lightsheet.edu_workflow_complete.v1",
                "workflow": str(manifest_path),
                "fusions": fusion_identity,
                "cellpose": str(cellpose),
                "completed_at": datetime.now().astimezone().isoformat(),
            },
        ),
        required=[completion],
    )
    return completion
