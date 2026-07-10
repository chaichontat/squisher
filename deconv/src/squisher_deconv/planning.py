from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile


@dataclass(frozen=True, slots=True, order=True)
class SamplePlane:
    file_index: int
    path: Path
    true_z: int


@dataclass(frozen=True, slots=True)
class SampleWindow:
    file_index: int
    path: Path
    read_start: int
    read_stop: int
    core_z: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SlabWindow:
    file_index: int
    path: Path
    core_start: int
    core_stop: int
    read_start: int
    read_stop: int


def output_relative_root(paths: Sequence[Path]) -> Path | None:
    names = [Path(path).name for path in paths]
    if len(set(names)) == len(names):
        return None
    parents = [str(Path(path).parent) or "." for path in paths]
    return Path(os.path.commonpath(parents))


def output_path_for(out_dir: Path, source_path: Path, *, relative_root: Path | None) -> Path:
    source_path = Path(source_path)
    output_name = _ome_zarr_name(source_path)
    if relative_root is not None:
        relative_path = source_path.relative_to(relative_root)
        return out_dir / relative_path.parent / output_name
    return out_dir / output_name


def output_sidecar_path(output_path: Path) -> Path:
    name = output_path.name
    if not name.endswith(".ome.zarr"):
        raise ValueError(f"Expected an OME-Zarr output path, got {output_path}")
    return output_path.with_name(f"{name.removesuffix('.ome.zarr')}.deconv.json")


def _ome_zarr_name(source_path: Path) -> str:
    name = source_path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.endswith(suffix):
            return f"{name.removesuffix(suffix)}.ome.zarr"
    return f"{source_path.stem}.ome.zarr"


def logical_z_count(path: Path, *, channels: int) -> int:
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
    if len(shape) < 3:
        raise ValueError(f"{path} must have at least flattened planes plus Y/X, got shape {shape}")
    plane_count = int(np.prod(shape[:-2]))
    if plane_count % channels:
        raise ValueError(
            f"{path} has {plane_count} plane(s), not divisible by channels={channels}"
        )
    return plane_count // channels


def uniform_sample_planes(
    paths: Sequence[Path],
    *,
    planes: int,
    channels: int,
    seed: int,
) -> list[SamplePlane]:
    populations = [logical_z_count(path, channels=channels) for path in paths]
    return sample_planes_from_z_counts(paths, z_counts=populations, planes=planes, seed=seed)


def sample_planes_from_z_counts(
    paths: Sequence[Path],
    *,
    z_counts: Sequence[int],
    planes: int,
    seed: int,
) -> list[SamplePlane]:
    populations = [int(count) for count in z_counts]
    total = sum(populations)
    if planes > total:
        raise ValueError(f"Requested {planes} sampled plane(s), but only {total} logical plane(s) exist.")
    offsets = np.cumsum([0, *populations])
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, size=planes, replace=False))
    out: list[SamplePlane] = []
    for flat in selected:
        file_index = int(np.searchsorted(offsets, flat, side="right") - 1)
        true_z = int(flat - offsets[file_index])
        out.append(SamplePlane(file_index=file_index, path=Path(paths[file_index]), true_z=true_z))
    return out


def group_sample_windows(samples: Sequence[SamplePlane], *, z_counts: Sequence[int], halo: int) -> list[SampleWindow]:
    by_file: dict[int, list[SamplePlane]] = {}
    for sample in samples:
        by_file.setdefault(sample.file_index, []).append(sample)

    windows: list[SampleWindow] = []
    for file_index in sorted(by_file):
        file_samples = sorted(by_file[file_index], key=lambda item: item.true_z)
        z_count = int(z_counts[file_index])
        current_start: int | None = None
        current_stop: int | None = None
        current_core: list[int] = []
        current_path = file_samples[0].path
        for sample in file_samples:
            read_start = max(0, sample.true_z - halo)
            read_stop = min(z_count, sample.true_z + halo + 1)
            if current_start is None:
                current_start = read_start
                current_stop = read_stop
                current_core = [sample.true_z]
                continue
            assert current_stop is not None
            if read_start <= current_stop:
                current_stop = max(current_stop, read_stop)
                current_core.append(sample.true_z)
            else:
                windows.append(
                    SampleWindow(
                        file_index=file_index,
                        path=current_path,
                        read_start=current_start,
                        read_stop=current_stop,
                        core_z=tuple(current_core),
                    )
                )
                current_start = read_start
                current_stop = read_stop
                current_core = [sample.true_z]
        if current_start is not None and current_stop is not None:
            windows.append(
                SampleWindow(
                    file_index=file_index,
                    path=current_path,
                    read_start=current_start,
                    read_stop=current_stop,
                    core_z=tuple(current_core),
                )
            )
    return windows


def slab_windows(paths: Sequence[Path], *, z_counts: Sequence[int], slab_depth: int, halo: int) -> list[SlabWindow]:
    windows: list[SlabWindow] = []
    for file_index, path in enumerate(paths):
        z_count = int(z_counts[file_index])
        for core_start in range(0, z_count, slab_depth):
            core_stop = min(z_count, core_start + slab_depth)
            windows.append(
                SlabWindow(
                    file_index=file_index,
                    path=Path(path),
                    core_start=core_start,
                    core_stop=core_stop,
                    read_start=max(0, core_start - halo),
                    read_stop=min(z_count, core_stop + halo),
                )
            )
    return windows
