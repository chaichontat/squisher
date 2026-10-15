"""CPU-only physical grid descriptor for NGFF array consumers."""

from collections.abc import Mapping
import math


def ome_spatial_grid(attrs: Mapping[str, object], dataset: str) -> dict[str, object] | None:
    """Compose dataset then image transforms; return None when no grid is declared.

    Partial or unsupported physical metadata fails closed. An omitted translation
    is the NGFF zero origin; a physical scale and spatial units must be explicit.
    """
    ome = attrs.get("ome")
    multiscales = ome.get("multiscales") if isinstance(ome, dict) else attrs.get("multiscales")
    if multiscales is None:
        return None
    if not isinstance(multiscales, list) or len(multiscales) != 1 or not isinstance(multiscales[0], dict):
        raise ValueError("OME physical grid requires exactly one multiscale image.")
    image = multiscales[0]
    datasets = image.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("OME physical grid has no datasets.")
    selected = [entry for entry in datasets if isinstance(entry, dict) and entry.get("path") == dataset]
    if len(selected) != 1:
        raise ValueError(f"OME physical grid has no unique dataset {dataset!r}.")
    axes = image.get("axes")
    local = selected[0].get("coordinateTransformations")
    global_transforms = image.get("coordinateTransformations")
    if axes is None and local is None and global_transforms is None:
        return None
    if not isinstance(axes, list) or not axes:
        raise ValueError("OME physical grid requires declared axes.")
    spatial = []
    names = []
    units = []
    all_names = []
    for index, axis in enumerate(axes):
        if not isinstance(axis, dict) or not isinstance(axis.get("name"), str):
            raise ValueError("OME physical grid requires named, typed axes.")
        name = axis["name"]
        all_names.append(name)
        if name in ("z", "y", "x"):
            unit = axis.get("unit")
            if axis.get("type") != "space" or not isinstance(unit, str) or not unit:
                raise ValueError("OME physical grid requires spatial axis types and units.")
            spatial.append(index)
            names.append(name)
            units.append(unit)
        elif name not in ("t", "c") or axis.get("type") not in ("time", "channel"):
            raise ValueError(f"Unsupported OME physical grid axis {name!r}.")
    if len(set(all_names)) != len(all_names) or names not in (["z", "y", "x"], ["y", "x"]):
        raise ValueError(f"Unsupported OME spatial grid axes {names!r}.")
    scale = [1.0] * len(axes)
    translation = [0.0] * len(axes)
    has_scale = False
    for transforms in (local, global_transforms):
        if transforms is None:
            continue
        if not isinstance(transforms, list) or not transforms:
            raise ValueError("OME physical grid has invalid coordinate transforms.")
        for transform in transforms:
            if not isinstance(transform, dict) or transform.get("type") not in ("scale", "translation"):
                raise ValueError("Unsupported OME physical grid coordinate transform.")
            kind = transform["type"]
            values = transform.get(kind)
            if (
                not isinstance(values, list)
                or len(values) != len(axes)
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in values
                )
            ):
                raise ValueError(f"OME physical grid has invalid {kind} transform.")
            if kind == "scale":
                if any(value <= 0 for value in values):
                    raise ValueError("OME physical grid scales must be positive.")
                has_scale = True
                scale = [first * second for first, second in zip(scale, values, strict=True)]
                translation = [first * second for first, second in zip(translation, values, strict=True)]
            else:
                translation = [first + second for first, second in zip(translation, values, strict=True)]
    if not has_scale or not all(math.isfinite(value) for value in (*scale, *translation)):
        raise ValueError("OME physical grid requires a finite explicit scale transform.")
    return {
        "axes": names,
        "units": units,
        "scale": [scale[index] for index in spatial],
        "translation": [translation[index] for index in spatial],
    }
