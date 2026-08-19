import hashlib
import re
from pathlib import Path


_PLAN_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


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


def runtime_source_sha256() -> dict[str, str]:
    """Fingerprint source modules that own distributed Cellpose inference."""
    import cellpose.contrib.cellposetrt
    import cellpose.contrib.pack_utils
    import cellpose.contrib.packed_infer
    import cellpose.core
    import cellpose.dynamics
    import cellpose.models
    import cellpose.transforms
    import cellpose.utils

    package_root = Path(__file__).parents[1]
    distributed_root = package_root / "segmentation" / "distributed"
    sources = {
        "distributed_segmentation": distributed_root / "distributed_segmentation.py",
        "model_cache": distributed_root / "model_cache.py",
        "merge_utils": distributed_root / "merge_utils.py",
        "model_artifacts": Path(__file__),
        "normalize": package_root / "segment" / "normalize.py",
        "cellpose_core": Path(cellpose.core.__file__),
        "cellpose_dynamics": Path(cellpose.dynamics.__file__),
        "cellpose_models": Path(cellpose.models.__file__),
        "cellpose_packed_infer": Path(cellpose.contrib.packed_infer.__file__),
        "cellpose_pack_utils": Path(cellpose.contrib.pack_utils.__file__),
        "cellpose_trt": Path(cellpose.contrib.cellposetrt.__file__),
        "cellpose_transforms": Path(cellpose.transforms.__file__),
        "cellpose_utils": Path(cellpose.utils.__file__),
    }
    return {name: file_sha256(path.resolve()) for name, path in sources.items()}
