import re
from pathlib import Path


_PLAN_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def plan_path_for_device(model_path: Path, device_name: str) -> Path:
    safe_device_name = _PLAN_SANITIZE_PATTERN.sub("_", device_name).strip("_") or "cuda"
    return model_path.with_name(f"{model_path.name}-{safe_device_name}.plan")
