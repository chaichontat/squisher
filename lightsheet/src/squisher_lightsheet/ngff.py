from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def multiscales(group: Any) -> list[dict[str, Any]]:
    """Return NGFF multiscales metadata from either the 0.4 or 0.5 location."""
    attrs = group.attrs.asdict() if hasattr(group.attrs, "asdict") else dict(group.attrs)
    ome = attrs.get("ome")
    records = ome.get("multiscales") if isinstance(ome, Mapping) else attrs.get("multiscales")
    if records is None:
        return []
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("OME-Zarr multiscales metadata must be a list of objects")
    return records


def dataset_paths(group: Any) -> list[str]:
    records = multiscales(group)
    if records:
        datasets = records[0].get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("OME-Zarr multiscales[0] must contain a non-empty datasets list")
        paths = []
        for index, dataset in enumerate(datasets):
            path = dataset.get("path") if isinstance(dataset, dict) else None
            if not isinstance(path, str) or not path:
                raise ValueError(f"OME-Zarr multiscales dataset {index} is missing a path")
            paths.append(path)
        return paths

    return sorted(
        (str(key) for key in group.keys()),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )


def level_path(group: Any, *, level: int, context: Path | str) -> str:
    source_level = int(level)
    if source_level < 0:
        raise ValueError(f"Requested negative OME-Zarr level {level} for {context}")
    paths = dataset_paths(group)
    if source_level < len(paths):
        return paths[source_level]
    legacy_path = str(source_level)
    if legacy_path in paths:
        return legacy_path
    raise ValueError(f"{context} has {len(paths)} OME-Zarr level(s); requested level {level}")


def level_array(group: Any, *, level: int = 0, context: Path | str = "OME-Zarr") -> Any:
    if hasattr(group, "shape"):
        if int(level) != 0:
            raise ValueError(f"Array-backed {context} has only level 0; requested level {level}")
        return group
    return group[level_path(group, level=level, context=context)]


def open_level_array(path: Path, *, level: int = 0) -> Any:
    import zarr

    root = zarr.open(str(path), mode="r")
    return level_array(root, level=level, context=path)


def axes(group: Any, array: Any) -> str:
    dimensions = array.attrs.get("_ARRAY_DIMENSIONS")
    if dimensions is None:
        dimensions = getattr(getattr(array, "metadata", None), "dimension_names", None)
    if dimensions is None:
        records = multiscales(group)
        raw_axes = records[0].get("axes") if records else None
        if isinstance(raw_axes, list):
            dimensions = [axis.get("name") if isinstance(axis, dict) else axis for axis in raw_axes]
    if dimensions is None:
        dimensions = {3: ("z", "y", "x"), 4: ("c", "z", "y", "x")}.get(array.ndim)
    if (
        not dimensions
        or len(dimensions) != array.ndim
        or not all(isinstance(dimension, str) and dimension for dimension in dimensions)
    ):
        raise ValueError(f"Cannot determine OME-Zarr axes for shape {array.shape}")
    return "".join(dimension.upper() for dimension in dimensions)


def scale_translation(
    group: Any, *, dataset_index: int = 0
) -> tuple[list[str], list[float], list[float], bool, bool]:
    """Compose dataset and multiscale transforms in NGFF application order."""
    records = multiscales(group)
    if not records:
        raise ValueError("OME-Zarr root is missing multiscales metadata")
    record = records[0]
    raw_axes = record.get("axes")
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ValueError("OME-Zarr multiscales metadata is missing axes")
    names = [axis.get("name") if isinstance(axis, Mapping) else axis for axis in raw_axes]
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("OME-Zarr axes must have non-empty names")

    datasets = record.get("datasets")
    if not isinstance(datasets, list) or not 0 <= dataset_index < len(datasets):
        raise ValueError(f"OME-Zarr multiscales metadata has no dataset {dataset_index}")
    dataset = datasets[dataset_index]
    if not isinstance(dataset, Mapping):
        raise ValueError(f"OME-Zarr dataset {dataset_index} must be an object")

    scale = [1.0] * len(names)
    translation = [0.0] * len(names)
    has_scale = False
    has_translation = False
    for owner, transforms in (("dataset", dataset.get("coordinateTransformations", [])),
                              ("multiscale", record.get("coordinateTransformations", []))):
        if not isinstance(transforms, list):
            raise ValueError(f"OME-Zarr {owner} coordinateTransformations must be a list")
        for transform in transforms:
            if not isinstance(transform, Mapping):
                raise ValueError(f"OME-Zarr {owner} transform must be an object")
            transform_type = transform.get("type")
            if transform_type not in {"scale", "translation"}:
                raise ValueError(f"unsupported OME-Zarr {owner} transform type {transform_type!r}")
            values = transform.get(transform_type)
            if not isinstance(values, list) or len(values) != len(names):
                raise ValueError(f"OME-Zarr {owner} {transform_type} must match the axes length")
            numeric = [float(value) for value in values]
            if transform_type == "scale":
                has_scale = True
                translation = [offset * factor for offset, factor in zip(translation, numeric, strict=True)]
                scale = [current * factor for current, factor in zip(scale, numeric, strict=True)]
            else:
                has_translation = True
                translation = [current + offset for current, offset in zip(translation, numeric, strict=True)]
    return list(names), scale, translation, has_scale, has_translation


def require_same_multiscale_grid(store: Path, template: Path, *, tolerance_um: float = 1e-6) -> None:
    """Fail unless ``store`` lies on ``template``'s grid: the same spatial axes, shapes, scales, and origins per level."""
    import numpy as np
    import zarr

    groups = {path: zarr.open_group(str(path), mode="r") for path in (store, template)}
    levels = {path: dataset_paths(group) for path, group in groups.items()}
    if len(levels[store]) != len(levels[template]):
        raise ValueError(f"{store} has {len(levels[store])} pyramid levels; {template} has {len(levels[template])}")
    for index, (path, template_path) in enumerate(zip(levels[store], levels[template], strict=True)):
        names, scale, origin, *_ = scale_translation(groups[store], dataset_index=index)
        template_names, template_scale, template_origin, *_ = scale_translation(groups[template], dataset_index=index)
        shape = tuple(groups[store][path].shape[-3:])
        template_shape = tuple(groups[template][template_path].shape[-3:])
        if (
            names[-3:] != template_names[-3:]
            or shape != template_shape
            or not np.allclose(scale[-3:], template_scale[-3:], rtol=0.0, atol=tolerance_um)
            or not np.allclose(origin[-3:], template_origin[-3:], rtol=0.0, atol=tolerance_um)
        ):
            raise ValueError(
                f"{store} level {index} ({names[-3:]}, shape {shape}, scale {scale[-3:]}, origin {origin[-3:]}) is "
                f"not on {template}'s grid ({template_names[-3:]}, {template_shape}, {template_scale[-3:]}, "
                f"{template_origin[-3:]})"
            )
