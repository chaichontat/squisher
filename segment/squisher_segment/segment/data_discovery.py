from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Pattern, Sequence

_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _compile_patterns(patterns: Sequence[str]) -> list[Pattern[str]]:
    compiled: list[Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern {pattern!r}: {exc}") from exc
    return compiled


def _matches_any(patterns: Sequence[Pattern[str]], value: str) -> bool:
    return any(regex.search(value) for regex in patterns)


def _iter_image_dirs(start: Path) -> Iterable[Path]:
    if start.is_file():
        yield start.parent
        return

    if not start.exists():
        return

    discovered_dirs: set[Path] = set()
    for file_path in start.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in _IMAGE_EXTENSIONS or file_path.name.endswith("_seg.npy"):
            discovered_dirs.add(file_path.parent)

    if not discovered_dirs:
        if start.is_dir():
            yield start
        return

    for directory in sorted(discovered_dirs):
        yield directory


def _discover_training_dirs(root: Path, training_paths: Sequence[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for entry in training_paths:
        entry_path = Path(entry)
        if ".." in entry_path.parts:
            raise ValueError(f"Training path {entry} cannot contain '..'.")

        candidate = root / entry_path
        if not candidate.exists():
            raise FileNotFoundError(f"Training path {candidate} does not exist.")

        for directory in _iter_image_dirs(candidate):
            try:
                relative = directory.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Discovered training directory {directory} is outside the provided training root {root}."
                ) from exc

            if ".." in relative.parts:
                raise ValueError(
                    f"Discovered training directory {directory} escapes the training root {root}. "
                    "Remove '..' segments from training_paths."
                )

            if relative in seen:
                continue

            seen.add(relative)
            discovered.append(relative)

    if not discovered:
        raise ValueError(
            "No training directories discovered under the configured training paths. "
            f"Checked entries={list(training_paths)}"
        )

    return discovered

