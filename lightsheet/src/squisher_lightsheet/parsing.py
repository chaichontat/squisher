from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def parse_source_view_path_entry(
    value: str,
    *,
    error_factory: Callable[[str], Exception],
) -> tuple[str, Path]:
    if "=" not in value:
        raise error_factory(f"Expected source-view flatfield entry as VIEW=DIR, got {value!r}")
    view, path = value.split("=", 1)
    view = view.strip()
    if not view:
        raise error_factory(f"Missing source-view name in {value!r}")
    if not path:
        raise error_factory(f"Missing flatfield directory in {value!r}")
    return view, Path(path)
