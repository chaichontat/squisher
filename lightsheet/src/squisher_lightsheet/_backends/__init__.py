"""Packaged scientific backends for fused-fixed registration and rendering."""

from pathlib import Path
import sys


def backend_executable(name: str, *, python: Path | None = None) -> Path:
    """Select the installed backend in the interpreter's environment."""
    executable = (python or Path(sys.executable)).absolute().parent / name
    if not executable.is_file():
        raise FileNotFoundError(
            f"Missing packaged backend {executable}; install squisher-lightsheet in this environment"
        )
    return executable
