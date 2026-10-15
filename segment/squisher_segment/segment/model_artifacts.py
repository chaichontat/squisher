import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from tempfile import NamedTemporaryFile
import os
import re
import shlex
from pathlib import Path
from typing import Mapping


_PLAN_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def configured_model_paths(model_kwargs: Mapping[str, object]) -> dict[str, Path]:
    """Return configured checkpoint paths keyed by their inference role."""
    primary = model_kwargs.get("pretrained_model")
    if not isinstance(primary, (str, os.PathLike)) or not os.fspath(primary):
        raise ValueError("model_kwargs must include a non-empty 'pretrained_model'.")

    paths = {"xy": Path(primary)}
    ortho = model_kwargs.get("pretrained_model_ortho")
    if ortho is None:
        return paths
    if not isinstance(ortho, (str, os.PathLike)) or not os.fspath(ortho):
        raise ValueError("model_kwargs must include a non-empty 'pretrained_model_ortho'.")
    paths["ortho"] = Path(ortho)
    return paths


def plan_path_for_device(model_path: Path, device_name: str) -> Path:
    safe_device_name = _PLAN_SANITIZE_PATTERN.sub("_", device_name).strip("_") or "cuda"
    return model_path.with_name(f"{model_path.name}-{safe_device_name}.plan")


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local model/runtime artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Increment when inference semantics change incompatibly with existing run outputs.
SEGMENTATION_ALGORITHM_REVISION = "2"


def runtime_implementation_identity() -> dict[str, str]:
    """Identify maintained inference semantics and the installed Cellpose release."""
    return {
        "segmentation_algorithm_revision": SEGMENTATION_ALGORITHM_REVISION,
        "cellpose_version": version("cellpose"),
    }


def plan_metadata_path(plan_path: Path) -> Path:
    return plan_path.with_name(f"{plan_path.name}.json")


def _build_runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("cellpose", "torch", "tensorrt"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    import torch
    versions["cuda"] = torch.version.cuda
    return versions


def _write_plan_metadata(plan_path: Path, metadata: Mapping[str, object]) -> None:
    destination = plan_metadata_path(plan_path)
    with NamedTemporaryFile(mode="w", dir=destination.parent, prefix=destination.name, delete=False) as stream:
        json.dump(metadata, stream, sort_keys=True, indent=2)
        temporary = Path(stream.name)
    os.replace(temporary, destination)


def invalidate_trt_plans(model_path: Path, *, device_name: str | None = None) -> None:
    """Mark derivatives invalid without removing existing user artifacts."""
    plans = (
        [plan_path_for_device(model_path, device_name)]
        if device_name is not None
        else sorted(model_path.parent.glob(f"{model_path.name}-*.plan"))
    )
    for plan_path in plans:
        _write_plan_metadata(plan_path, {"schema_version": 1, "status": "invalid"})


def write_trt_plan_metadata(
    model_path: Path,
    device_name: str,
    *,
    backend: str,
    bsize: int,
    batch_size: int,
    opset: int,
    vram_mb: int,
) -> dict[str, object]:
    """Publish the generating checkpoint and profile after a successful build."""
    plan_path = plan_path_for_device(model_path, device_name)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "status": "ready",
        "checkpoint_sha256": file_sha256(model_path),
        "plan_sha256": file_sha256(plan_path),
        "device_name": device_name,
        "build": {"backend": backend, "bsize": bsize, "batch_size": batch_size,
                  "opset": opset, "vram_mb": vram_mb, "dtype": "bfloat16"},
        "runtime_versions": _build_runtime_versions(),
    }
    _write_plan_metadata(plan_path, metadata)
    return metadata


def validate_trt_plan(
    model_path: Path,
    device_name: str,
    *,
    backend: str = "sam",
    bsize: int = 256,
    batch_size: int = 1,
) -> dict[str, object]:
    """Fail closed for unbound, stale, corrupted, or incompatible derivatives."""
    plan_path = plan_path_for_device(model_path, device_name)
    sidecar = plan_metadata_path(plan_path)
    rebuild_command = f"squisher-segment build-trt {shlex.quote(str(model_path))} --backend {shlex.quote(backend)}"
    if not sidecar.is_file():
        raise ValueError(f"TensorRT plan is unverified: {plan_path}; rebuild with `{rebuild_command}`.")
    metadata = json.loads(sidecar.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid TensorRT plan metadata: {sidecar}")
    build = metadata.get("build")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("status") != "ready"
        or metadata.get("device_name") != device_name
        or metadata.get("checkpoint_sha256") != file_sha256(model_path)
        or metadata.get("plan_sha256") != file_sha256(plan_path)
        or metadata.get("runtime_versions") != _build_runtime_versions()
        or not isinstance(build, dict)
        or build.get("backend") != backend
        or build.get("bsize") != bsize
        or build.get("batch_size") != batch_size
        or build.get("dtype") != "bfloat16"
        or not isinstance(build.get("opset"), int)
        or not isinstance(build.get("vram_mb"), int)
    ):
        raise ValueError(f"TensorRT plan is stale or incompatible: {plan_path}; rebuild with `{rebuild_command}`.")
    return metadata
