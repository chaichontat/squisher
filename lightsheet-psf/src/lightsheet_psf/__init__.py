from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lightsheet-psf")
except PackageNotFoundError:  # pragma: no cover - editable source tree before install
    __version__ = "0+unknown"

__all__ = ["__version__"]
