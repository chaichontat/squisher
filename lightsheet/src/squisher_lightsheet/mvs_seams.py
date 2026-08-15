from __future__ import annotations

import json
import ast
import hashlib
import math
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from squisher_lightsheet import ngff


DEFAULT_MVS_SEAM_QUALITY_THRESHOLD = 0.25
DIMENSIONS = ("z", "y", "x")
MIN_SHIFTED_VALID_SUPPORT_COVERAGE = 0.25
DEFAULT_LEVEL0_PATCHES_PER_EDGE = 12
DEFAULT_LEVEL0_RETRY_PATCHES_PER_EDGE = 50
DEFAULT_LEVEL0_MAX_DISCONNECTED_ISLAND_SIZE = 3
DEFAULT_LEVEL0_FALLBACK_LEVEL2_WEIGHT_SCALE = 0.05
LEVEL2_FALLBACK_SOURCE_LABEL = "mvs_pairwise_level2_fallback"
Level0CandidateMode = Literal["scout", "seam-span"]


@dataclass(frozen=True)
class Level0PatchCandidate:
    patch_index: int
    center_px_zyx: tuple[float, float, float]
    start_px_zyx: tuple[int, int, int]
    shape_zyx: tuple[int, int, int]
    scout_score: float


def normalize_mvs_edge(edge: Any) -> tuple[int, int]:
    if isinstance(edge, str):
        edge = ast.literal_eval(edge)
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        return tuple(sorted((int(edge[0]), int(edge[1]))))
    raise ValueError(f"Invalid MVS edge identifier: {edge!r}")


def mvs_data_scalar(value: Any) -> float:
    if isinstance(value, dict) and "data" in value:
        data = value["data"]
        while isinstance(data, list):
            if not data:
                return float("nan")
            data = data[0]
        return float(data)
    return float(value)


def mvs_transform_matrix(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        data = np.asarray(value["data"], dtype=float)
        if data.ndim == 3:
            data = data[0]
        return data
    return np.asarray(value, dtype=float)


def _zyx(record: dict[str, Any], key: str) -> np.ndarray:
    values = record[key]
    return np.asarray([values[dim] for dim in DIMENSIONS], dtype=np.float64)


def tile_image_path(registration_payload: dict[str, Any], tile_record: dict[str, Any]) -> Path:
    tile_path = Path(str(tile_record.get("path", tile_record["tile"])))
    if tile_path.is_absolute() and tile_path.exists():
        return tile_path

    input_dir = Path(str(registration_payload["input_dir"]))
    candidate = input_dir / tile_path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve image path for {tile_record['tile']} from input_dir={input_dir}; "
        "provide a valid path field or a tile path relative to input_dir"
    )


def _tiff_level_metadata(path: Path, *, level: int) -> tuple[str, int, tuple[int, ...]]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        source_level = int(level)
        if source_level < 0 or source_level >= len(series.levels):
            raise ValueError(f"{path} has {len(series.levels)} TIFF pyramid level(s); requested level {level}")
        page_series = series.levels[source_level]
        return str(page_series.axes), source_level, tuple(int(value) for value in page_series.shape)


def _zarr_multiscales(root: Any) -> list[dict[str, Any]]:
    return ngff.multiscales(root)


def _zarr_level_path(root: Any, *, level: int, path: Path) -> str:
    return ngff.level_path(root, level=level, context=path)


def _zarr_level_metadata(path: Path, *, level: int) -> tuple[str, int, tuple[int, ...]]:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    source_level = int(level)
    array = root[_zarr_level_path(root, level=source_level, path=path)]
    return ngff.axes(root, array), source_level, tuple(int(value) for value in array.shape)


def _image_level_metadata(path: Path, *, level: int) -> tuple[str, int, tuple[int, ...]]:
    if path.name.endswith(".zarr"):
        return _zarr_level_metadata(path, level=level)
    return _tiff_level_metadata(path, level=level)


def _spatial_shape_zyx(*, axes: str, shape: tuple[int, ...]) -> np.ndarray:
    if axes == "CZYX":
        return np.asarray(shape[1:4], dtype=np.int64)
    if axes == "ZYX":
        return np.asarray(shape, dtype=np.int64)
    raise ValueError(f"Expected CZYX or ZYX axes, got {axes!r}")


def _read_tiff_level_crop(
    path: Path,
    *,
    level: int,
    axes: str,
    channel: int,
    slices_zyx: tuple[slice, slice, slice],
) -> np.ndarray:
    import tifffile
    import zarr

    store = tifffile.imread(path, aszarr=True, level=level)
    try:
        zarray = zarr.open(store, mode="r")
        if hasattr(zarray, "keys") and "0" in zarray:
            zarray = zarray["0"]
        if axes == "CZYX":
            crop = zarray[(channel, *slices_zyx)]
        elif axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} requested for single-channel tile {path}")
            crop = zarray[slices_zyx]
        else:
            raise ValueError(f"Expected CZYX or ZYX axes, got {axes!r} for {path}")
        return np.asarray(crop, dtype=np.float32)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _read_zarr_level_crop(
    path: Path,
    *,
    level: int,
    axes: str,
    channel: int,
    slices_zyx: tuple[slice, slice, slice],
) -> np.ndarray:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    array = root[_zarr_level_path(root, level=level, path=path)]
    if axes == "CZYX":
        crop = array[(channel, *slices_zyx)]
    elif axes == "ZYX":
        if channel != 0:
            raise ValueError(f"Channel {channel} requested for single-channel tile {path}")
        crop = array[slices_zyx]
    else:
        raise ValueError(f"Expected CZYX or ZYX axes, got {axes!r} for {path}")
    return np.asarray(crop, dtype=np.float32)


def _read_image_level_crop(
    path: Path,
    *,
    level: int,
    axes: str,
    channel: int,
    slices_zyx: tuple[slice, slice, slice],
) -> np.ndarray:
    if path.name.endswith(".zarr"):
        return _read_zarr_level_crop(path, level=level, axes=axes, channel=channel, slices_zyx=slices_zyx)
    return _read_tiff_level_crop(path, level=level, axes=axes, channel=channel, slices_zyx=slices_zyx)


@dataclass
class _ImageLevelAccessor:
    axes: str
    level: int
    shape: tuple[int, ...]
    array: Any
    store: Any | None = None

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()


class _ImageLevelReaderCache:
    def __init__(self) -> None:
        self._accessors: dict[tuple[Path, int], _ImageLevelAccessor] = {}
        self._lock = Lock()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        seen: set[int] = set()
        for accessor in self._accessors.values():
            if id(accessor) in seen:
                continue
            seen.add(id(accessor))
            accessor.close()
        self._accessors.clear()

    def metadata(self, path: Path, *, level: int) -> tuple[str, int, tuple[int, ...]]:
        accessor = self._accessor(path, level=level)
        return accessor.axes, accessor.level, accessor.shape

    def read_crop(
        self,
        path: Path,
        *,
        level: int,
        axes: str,
        channel: int,
        slices_zyx: tuple[slice, slice, slice],
    ) -> np.ndarray:
        accessor = self._accessor(path, level=level)
        if accessor.axes != axes:
            raise ValueError(f"Cached image axes changed for {path} level {level}: {accessor.axes!r} != {axes!r}")
        if axes == "CZYX":
            crop = accessor.array[(channel, *slices_zyx)]
        elif axes == "ZYX":
            if channel != 0:
                raise ValueError(f"Channel {channel} requested for single-channel tile {path}")
            crop = accessor.array[slices_zyx]
        else:
            raise ValueError(f"Expected CZYX or ZYX axes, got {axes!r} for {path}")
        return np.asarray(crop, dtype=np.float32)

    def _accessor(self, path: Path, *, level: int) -> _ImageLevelAccessor:
        key = (path.resolve(), int(level))
        with self._lock:
            if key in self._accessors:
                return self._accessors[key]
            accessor = _open_image_level_accessor(path, level=int(level))
            self._accessors[key] = accessor
            self._accessors.setdefault((path.resolve(), int(accessor.level)), accessor)
            return accessor


def _open_image_level_accessor(path: Path, *, level: int) -> _ImageLevelAccessor:
    if path.name.endswith(".zarr"):
        import zarr

        axes, source_level, shape = _zarr_level_metadata(path, level=level)
        root = zarr.open_group(str(path), mode="r")
        array = root[_zarr_level_path(root, level=source_level, path=path)]
        return _ImageLevelAccessor(axes=axes, level=source_level, shape=shape, array=array)

    import tifffile
    import zarr

    axes, source_level, shape = _tiff_level_metadata(path, level=level)
    # tifffile resolves this level through the TIFF pyramid representation
    # exposed as series levels. Keep the store open across crops.
    store = tifffile.imread(path, aszarr=True, level=source_level)
    array = zarr.open(store, mode="r")
    if hasattr(array, "keys") and "0" in array:
        array = array["0"]
    return _ImageLevelAccessor(axes=axes, level=source_level, shape=shape, array=array, store=store)


def _cached_image_level_metadata(
    path: Path,
    *,
    level: int,
    image_reader_cache: _ImageLevelReaderCache | None,
) -> tuple[str, int, tuple[int, ...]]:
    if image_reader_cache is None:
        return _image_level_metadata(path, level=level)
    return image_reader_cache.metadata(path, level=level)


def _cached_read_image_level_crop(
    path: Path,
    *,
    level: int,
    axes: str,
    channel: int,
    slices_zyx: tuple[slice, slice, slice],
    image_reader_cache: _ImageLevelReaderCache | None,
) -> np.ndarray:
    if image_reader_cache is None:
        return _read_image_level_crop(path, level=level, axes=axes, channel=channel, slices_zyx=slices_zyx)
    return image_reader_cache.read_crop(path, level=level, axes=axes, channel=channel, slices_zyx=slices_zyx)


def _level_shape_zyx(
    native_shape_zyx: tuple[int, int, int],
    native_shift_scale_zyx: np.ndarray,
) -> tuple[int, int, int]:
    return tuple(
        max(1, int(math.ceil(float(size) / float(scale))))
        for size, scale in zip(native_shape_zyx, native_shift_scale_zyx, strict=True)
    )


def _registration_hash_from_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _phase_margin_zyx(max_phase_shift_zyx: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(math.ceil(float(value))) for value in max_phase_shift_zyx)


def _default_contact_sheet_dir(output_registration: Path) -> Path:
    return output_registration.with_name(f"{output_registration.stem}.level0-contact-sheet")


def _scale_u8(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(sample, [1.0, 99.8])
    scaled = np.clip((values - low) / max(float(high - low), 1.0), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _shift_volume_cpu(volume: np.ndarray, shift_zyx: tuple[float, float, float]) -> np.ndarray:
    from scipy import ndimage as scipy_ndimage

    return scipy_ndimage.shift(
        volume,
        shift=shift_zyx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)


def _valid_support_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def _phase_crop_slices(
    *,
    fixed_patch: np.ndarray,
    moving_patch: np.ndarray,
    max_phase_shift_zyx: tuple[float, float, float],
) -> tuple[tuple[slice, slice, slice], tuple[int, int, int], tuple[int, int, int], float] | None:
    support = _valid_support_mask(fixed_patch) & _valid_support_mask(moving_patch)
    if not np.any(support):
        return None
    coords = np.where(support)
    starts = []
    stops = []
    for axis, values in enumerate(coords):
        margin = int(math.ceil(float(max_phase_shift_zyx[axis])))
        start = max(0, int(values.min()) - margin)
        stop = min(fixed_patch.shape[axis], int(values.max()) + margin + 1)
        if stop <= start:
            return None
        starts.append(start)
        stops.append(stop)
    slices = tuple(slice(starts[axis], stops[axis]) for axis in range(3))
    shape = tuple(stops[axis] - starts[axis] for axis in range(3))
    overlap_fraction = float(np.count_nonzero(support[slices]) / np.prod(shape))
    return slices, tuple(starts), shape, overlap_fraction


def _shifted_valid_support_metrics(
    *,
    fixed_patch: np.ndarray,
    moving_patch: np.ndarray,
    shift_zyx: tuple[float, float, float],
) -> dict[str, float]:
    shifted_moving = _shift_volume_cpu(_valid_support_mask(moving_patch).astype(np.float32), shift_zyx) > 0.5
    fixed_support = _valid_support_mask(fixed_patch)
    moving_support = _valid_support_mask(moving_patch)
    shifted_common = fixed_support & shifted_moving
    common_count = int(np.count_nonzero(shifted_common))
    fixed_count = int(np.count_nonzero(fixed_support))
    moving_count = int(np.count_nonzero(moving_support))
    return {
        "shifted_valid_overlap_fraction": float(common_count / fixed_patch.size),
        "shifted_fixed_valid_coverage": float(common_count / fixed_count) if fixed_count else 0.0,
        "shifted_moving_valid_coverage": float(common_count / moving_count) if moving_count else 0.0,
    }


def _local_phase_overlay_image(source_patch: np.ndarray, target_patch: np.ndarray, shift_zyx: tuple[float, float, float]) -> Any:
    from PIL import Image

    shifted_target = _shift_volume_cpu(target_patch, shift_zyx)
    shifted_target_support = _shift_volume_cpu(_valid_support_mask(target_patch).astype(np.float32), shift_zyx) > 0.5
    shifted_common_support = _valid_support_mask(source_patch) & shifted_target_support
    if np.any(shifted_common_support):
        y_indices, x_indices = np.where(np.max(shifted_common_support, axis=0))
        y_slice = slice(int(y_indices.min()), int(y_indices.max()) + 1)
        x_slice = slice(int(x_indices.min()), int(x_indices.max()) + 1)
        display_source = np.where(shifted_common_support, source_patch, 0.0)[:, y_slice, x_slice]
        display_target = np.where(shifted_common_support, shifted_target, 0.0)[:, y_slice, x_slice]
    else:
        display_source = source_patch
        display_target = shifted_target
    source_mip = np.max(display_source, axis=0)
    target_mip = np.max(display_target, axis=0)
    rgb = np.zeros((*source_mip.shape, 3), dtype=np.uint8)
    rgb[..., 0] = _scale_u8(target_mip)
    rgb[..., 1] = _scale_u8(source_mip)
    return Image.fromarray(rgb, mode="RGB")


def _draw_level0_contact_panel(
    image: Any,
    *,
    title: str,
    subtitle: str,
    thumb_size: int,
) -> Any:
    from PIL import Image, ImageDraw

    image = image.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
    panel = Image.new("RGB", (thumb_size, thumb_size + 42), "white")
    panel.paste(image, (0, 0))
    draw = ImageDraw.Draw(panel)
    draw.text((4, thumb_size + 4), title, fill=(0, 0, 0))
    draw.text((4, thumb_size + 22), subtitle, fill=(0, 0, 0))
    return panel


def _contact_sheet_edges(
    measured_edges: list[dict[str, Any]],
    max_panels: int,
    *,
    edge_statuses: tuple[str, ...] = ("accepted",),
) -> list[dict[str, Any]]:
    selected = [edge for edge in measured_edges if edge["status"] in edge_statuses]
    selected.sort(key=lambda edge: (int(edge["pair"][0]), int(edge["pair"][1])))
    if len(selected) <= max_panels:
        return selected
    indices = np.linspace(0, len(selected) - 1, max_panels)
    return [selected[int(round(index))] for index in indices]


def _contact_sheet_patch(edge: dict[str, Any]) -> dict[str, Any]:
    inliers = [patch for patch in edge["patches"] if patch.get("inlier")]
    if inliers:
        return max(inliers, key=lambda patch: float(patch["scout_score"]))
    accepted = [patch for patch in edge["patches"] if patch["accepted"]]
    return max(accepted or edge["patches"], key=lambda patch: float(patch["scout_score"]))


def _candidate_from_patch_record(patch: dict[str, Any]) -> Level0PatchCandidate:
    return Level0PatchCandidate(
        patch_index=int(patch["patch_index"]),
        center_px_zyx=tuple(float(value) for value in patch["center_px_zyx"]),
        start_px_zyx=tuple(int(value) for value in patch["start_px_zyx"]),
        shape_zyx=tuple(int(value) for value in patch["shape_zyx"]),
        scout_score=float(patch["scout_score"]),
    )


def render_mvs_level0_refinement_contact_sheet(
    *,
    diagnostics: dict[str, Any],
    sample_registration_payload: dict[str, Any],
    output_dir: Path,
    channel: int,
    max_panels: int = 128,
    columns: int = 6,
    thumb_size: int = 256,
    edge_statuses: tuple[str, ...] = ("accepted",),
) -> dict[str, Any]:
    """Render measured local shifts in the same seed frame used for phase correlation."""
    if max_panels < 1:
        raise ValueError("max_panels must be >= 1")
    if columns < 1:
        raise ValueError("columns must be >= 1")
    if thumb_size < 32:
        raise ValueError("thumb_size must be >= 32")

    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    spacing_um_zyx = _zyx(sample_registration_payload, "spacing_um")
    tiles = sample_registration_payload["tiles"]
    panels = []
    panel_records = []
    for edge in _contact_sheet_edges(diagnostics["measured_edges"], max_panels, edge_statuses=edge_statuses):
        patch = _contact_sheet_patch(edge)
        candidate = _candidate_from_patch_record(patch)
        refinement_level = int(patch.get("refinement_level", edge.get("refinement_level", 0)))
        native_shift_scale_zyx = np.asarray(
            patch.get("native_shift_scale_zyx", edge.get("native_shift_scale_zyx", [1.0, 1.0, 1.0])),
            dtype=np.float64,
        )
        sample_shape_zyx = _level_shape_zyx(candidate.shape_zyx, native_shift_scale_zyx)
        center_level_px_zyx = np.asarray(candidate.center_px_zyx, dtype=np.float64) / native_shift_scale_zyx
        sample_start_level_px_zyx = tuple(
            int(round(float(center_level_px_zyx[index]) - float(sample_shape_zyx[index]) / 2.0))
            for index in range(3)
        )
        phase_input_origin = patch.get("phase_input_origin_px_zyx")
        if phase_input_origin is None:
            phase_margin = (0, 0, 0)
        else:
            phase_margin = tuple(
                int(sample_start_level_px_zyx[index] - int(phase_input_origin[index]))
                for index in range(3)
            )
            if any(value < 0 for value in phase_margin):
                raise ValueError(
                    f"Patch {edge['pair']}:{candidate.patch_index} has phase_input_origin_px_zyx after candidate start"
                )
        source, target = [int(value) for value in edge["pair"]]
        source_patch = _sample_candidate_patch(
            registration_payload=sample_registration_payload,
            tile_record=tiles[source],
            candidate=candidate,
            margin_zyx=phase_margin,
            spacing_um_zyx=spacing_um_zyx,
            channel=channel,
            level=refinement_level,
            native_shift_scale_zyx=native_shift_scale_zyx,
        )
        target_patch = _sample_candidate_patch(
            registration_payload=sample_registration_payload,
            tile_record=tiles[target],
            candidate=candidate,
            margin_zyx=phase_margin,
            spacing_um_zyx=spacing_um_zyx,
            channel=channel,
            level=refinement_level,
            native_shift_scale_zyx=native_shift_scale_zyx,
        )
        shift = tuple(float(value) for value in patch["shift_px_zyx"])
        crop_start = tuple(int(value) for value in patch.get("phase_crop_start_offset_zyx", [0, 0, 0]))
        crop_shape = tuple(
            int(value)
            for value in patch.get(
                "phase_crop_shape_zyx",
                tuple(candidate.shape_zyx[index] + 2 * phase_margin[index] for index in range(3)),
            )
        )
        crop_slices = tuple(
            slice(crop_start[index], crop_start[index] + crop_shape[index])
            for index in range(3)
        )
        source_patch = source_patch[crop_slices]
        target_patch = target_patch[crop_slices]
        image = _local_phase_overlay_image(source_patch, target_patch, shift)
        title = f"{source}->{target} q={edge['quality']:.3f} p{patch['patch_index']}"
        overlap = patch.get("shifted_valid_overlap_fraction")
        reason = edge.get("reject_reason") or patch.get("reject_reason") or edge["status"]
        subtitle = (
            f"meas=({shift[0]:.1f},{shift[1]:.1f},{shift[2]:.1f}) "
            f"ov={0.0 if overlap is None else float(overlap):.2f} {reason}"
        )
        panels.append(_draw_level0_contact_panel(image, title=title, subtitle=subtitle, thumb_size=thumb_size))
        panel_records.append(
            {
                "pair": [source, target],
                "source_tile": edge["source_tile"],
                "target_tile": edge["target_tile"],
                "patch_index": int(patch["patch_index"]),
                "patch_shift_px_zyx": [float(value) for value in shift],
                "patch_shift_native_px_zyx": patch.get("shift_native_px_zyx"),
                "render_shift_px_zyx": [float(value) for value in shift],
                "edge_shift_px_zyx": edge["edge_shift_px_zyx"],
                "refinement_level": refinement_level,
                "native_shift_scale_zyx": [float(value) for value in native_shift_scale_zyx],
                "edge_status": edge["status"],
                "edge_reject_reason": edge.get("reject_reason"),
                "patch_reject_reason": patch.get("reject_reason"),
                "quality": float(edge["quality"]),
                "phase_crop_start_offset_zyx": [int(value) for value in crop_start],
                "phase_input_origin_px_zyx": patch.get("phase_input_origin_px_zyx"),
                "phase_crop_shape_zyx": [int(value) for value in crop_shape],
                "seed_valid_overlap_fraction": patch.get("seed_valid_overlap_fraction"),
                "shifted_valid_overlap_fraction": patch.get("shifted_valid_overlap_fraction"),
                "shifted_fixed_valid_coverage": patch.get("shifted_fixed_valid_coverage"),
                "shifted_moving_valid_coverage": patch.get("shifted_moving_valid_coverage"),
                "patch_read_strategy": diagnostics["settings"].get("patch_read_strategy"),
                "sample_registration": diagnostics["registration_input"],
            }
        )

    if not panels:
        raise ValueError(f"No level-0 seam patches with statuses {edge_statuses!r} are available for the contact sheet")
    rows = int(math.ceil(len(panels) / columns))
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + 42)), "white")
    for index, panel in enumerate(panels):
        x = (index % columns) * thumb_size
        y = (index // columns) * (thumb_size + 42)
        sheet.paste(panel, (x, y))

    sheet_path = output_dir / "level0_refinement_local_phase_contact_sheet.png"
    index_path = output_dir / "level0_refinement_local_phase_contact_sheet.json"
    payload = {
        "artifact_type": "lightsheet.mvs_level0_refinement_contact_sheet.v1",
        "contact_sheet": str(sheet_path),
        "index": str(index_path),
        "channel": int(channel),
        "panel_count": len(panels),
        "max_panels": int(max_panels),
        "edge_statuses": list(edge_statuses),
        "columns": int(columns),
        "thumb_size": int(thumb_size),
        "render": "seed registration coordinates with local phase-correlation shift applied directly; source green, target red",
        "sample_registration": diagnostics["registration_input"],
        "panels": panel_records,
    }
    sheet.save(sheet_path)
    index_path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
    return payload


def _stage_translation_um(tile_record: dict[str, Any]) -> np.ndarray:
    if "stage_translation_um" in tile_record:
        return _zyx(tile_record, "stage_translation_um")
    if "translation_um" in tile_record:
        return _zyx(tile_record, "translation_um")
    return np.zeros(3, dtype=np.float64)


def _registered_affine(tile_record: dict[str, Any]) -> np.ndarray:
    return np.asarray(tile_record["registered_affine"]["matrix"], dtype=np.float64)


def _edge_bbox_center_um(edge: dict[str, Any]) -> np.ndarray | None:
    bbox = edge.get("attrs", {}).get("bbox")
    if not isinstance(bbox, dict) or "data" not in bbox:
        return None
    data = np.asarray(bbox["data"], dtype=np.float64)
    if data.ndim == 3:
        data = data[0]
    if data.shape != (2, 3):
        return None
    return np.mean(data, axis=0)


def _edge_bbox_um(edge: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    bbox = edge.get("attrs", {}).get("bbox")
    if not isinstance(bbox, dict) or "data" not in bbox:
        return None
    data = np.asarray(bbox["data"], dtype=np.float64)
    if data.ndim == 3:
        data = data[0]
    if data.shape != (2, 3):
        return None
    return np.minimum(data[0], data[1]), np.maximum(data[0], data[1])


def _content_score(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if sample.size == 0:
        return 0.0
    return float(np.std(sample) * max(0.0, np.percentile(sample, 99.0) - np.percentile(sample, 50.0)))


def _candidate_centers_from_bbox(
    *,
    edge: dict[str, Any],
    spacing_um_zyx: np.ndarray,
    patch_shape_zyx: tuple[int, int, int],
    patches_per_edge: int,
) -> list[np.ndarray]:
    bbox = _edge_bbox_um(edge)
    if bbox is None:
        return []
    lo_um, hi_um = bbox
    lo_px = lo_um / spacing_um_zyx
    hi_px = hi_um / spacing_um_zyx
    center_px = (lo_px + hi_px) / 2.0
    usable_lo = lo_px + np.asarray(patch_shape_zyx, dtype=np.float64) / 2.0
    usable_hi = hi_px - np.asarray(patch_shape_zyx, dtype=np.float64) / 2.0
    center_z = float(np.clip(center_px[0], usable_lo[0], usable_hi[0])) if usable_hi[0] >= usable_lo[0] else float(center_px[0])

    count = max(1, int(patches_per_edge))
    y_count = max(1, int(math.floor(math.sqrt(count))))
    x_count = max(1, int(math.ceil(count / y_count)))
    if usable_hi[1] < usable_lo[1]:
        y_values = [float(center_px[1])]
    else:
        y_values = [float(value) for value in np.linspace(usable_lo[1], usable_hi[1], y_count)]
    if usable_hi[2] < usable_lo[2]:
        x_values = [float(center_px[2])]
    else:
        x_values = [float(value) for value in np.linspace(usable_lo[2], usable_hi[2], x_count)]
    centers = []
    for y in y_values:
        for x in x_values:
            centers.append(np.asarray([center_z, y, x], dtype=np.float64))
            if len(centers) >= count:
                return centers
    return centers


def _candidate_start(
    center_px_zyx: np.ndarray,
    patch_shape_zyx: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(
        int(round(float(center_px_zyx[index]) - float(patch_shape_zyx[index]) / 2.0))
        for index in range(3)
    )


def _edge_bbox_px(edge: dict[str, Any], spacing_um_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    bbox = _edge_bbox_um(edge)
    if bbox is None:
        return None
    lo_um, hi_um = bbox
    return lo_um / spacing_um_zyx, hi_um / spacing_um_zyx


def _seam_face_orientation(edge: dict[str, Any], spacing_um_zyx: np.ndarray) -> tuple[int, int, Literal["xz", "yz"], str, list[float]] | None:
    bbox = _edge_bbox_px(edge, spacing_um_zyx)
    if bbox is None:
        return None
    lo_px, hi_px = bbox
    extents_px = np.maximum(hi_px - lo_px, 0.0)
    if extents_px[1] <= extents_px[2]:
        return 1, 2, "xz", "normal_y", [float(value) for value in extents_px]
    return 2, 1, "yz", "normal_x", [float(value) for value in extents_px]


def _face_content_score_map(source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
    from scipy import ndimage as scipy_ndimage

    source = np.asarray(source_face, dtype=np.float32)
    target = np.asarray(target_face, dtype=np.float32)
    source_mask = _valid_support_mask(source)
    target_mask = _valid_support_mask(target)
    source_values = np.where(source_mask, source, 0.0)
    target_values = np.where(target_mask, target, 0.0)

    window = (7, 7)

    def local_std(values: np.ndarray) -> np.ndarray:
        mean = scipy_ndimage.uniform_filter(values, size=window, mode="nearest")
        mean_sq = scipy_ndimage.uniform_filter(values * values, size=window, mode="nearest")
        return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    shared_fraction = scipy_ndimage.uniform_filter(
        (source_mask & target_mask).astype(np.float32),
        size=window,
        mode="nearest",
    )
    return np.minimum(local_std(source_values), local_std(target_values)) * shared_fraction


def _select_level0_retry_face_candidates(
    *,
    registration_payload: dict[str, Any],
    edge: dict[str, Any],
    tiles: list[dict[str, Any]],
    spacing_um_zyx: np.ndarray,
    channel: int,
    patch_shape_zyx: tuple[int, int, int],
    patches_per_edge: int,
    image_reader_cache: _ImageLevelReaderCache | None = None,
) -> tuple[list[Level0PatchCandidate], dict[str, Any]]:
    bbox = _edge_bbox_px(edge, spacing_um_zyx)
    orientation = _seam_face_orientation(edge, spacing_um_zyx)
    if bbox is None or orientation is None:
        return [], {"mode": "seam_face_content", "orientation": None, "plane": None}

    lo_px, hi_px = bbox
    center_px = (lo_px + hi_px) / 2.0
    normal_axis, along_axis, plane, orientation_name, extents_px = orientation
    usable_lo = lo_px + np.asarray(patch_shape_zyx, dtype=np.float64) / 2.0
    usable_hi = hi_px - np.asarray(patch_shape_zyx, dtype=np.float64) / 2.0

    def axis_values(axis: int, sample_count: int) -> np.ndarray:
        if usable_hi[axis] < usable_lo[axis]:
            return np.asarray([center_px[axis]], dtype=np.float64)
        return np.linspace(usable_lo[axis], usable_hi[axis], max(1, int(sample_count)), dtype=np.float64)

    count = max(1, int(patches_per_edge))
    z_values = axis_values(0, min(256, max(64, count * 4)))
    along_values = axis_values(along_axis, min(512, max(128, count * 4)))

    source = int(edge["source"])
    target = int(edge["target"])
    source_face = _sample_registered_plane_grid(
        registration_payload=registration_payload,
        tile_record=tiles[source],
        fixed_axis=normal_axis,
        fixed_value_px=float(center_px[normal_axis]),
        axis0=0,
        axis0_values_px=z_values,
        axis1=along_axis,
        axis1_values_px=along_values,
        spacing_um_zyx=spacing_um_zyx,
        level=2,
        channel=channel,
        image_reader_cache=image_reader_cache,
    )
    target_face = _sample_registered_plane_grid(
        registration_payload=registration_payload,
        tile_record=tiles[target],
        fixed_axis=normal_axis,
        fixed_value_px=float(center_px[normal_axis]),
        axis0=0,
        axis0_values_px=z_values,
        axis1=along_axis,
        axis1_values_px=along_values,
        spacing_um_zyx=spacing_um_zyx,
        level=2,
        channel=channel,
        image_reader_cache=image_reader_cache,
    )
    score_map = _face_content_score_map(source_face, target_face)
    z_indices, along_indices = np.indices(score_map.shape)
    order = np.lexsort((along_indices.ravel(), z_indices.ravel(), -score_map.ravel()))

    candidates: list[Level0PatchCandidate] = []
    selected_centers: list[np.ndarray] = []
    for flat_index in order:
        score = float(score_map.ravel()[int(flat_index)])
        if score <= 0.0:
            break
        z_index = int(z_indices.ravel()[int(flat_index)])
        along_index = int(along_indices.ravel()[int(flat_index)])
        candidate_center = center_px.copy()
        candidate_center[0] = float(z_values[z_index])
        candidate_center[normal_axis] = float(center_px[normal_axis])
        candidate_center[along_axis] = float(along_values[along_index])
        if any(
            abs(candidate_center[0] - selected[0]) < patch_shape_zyx[0] / 2.0
            and abs(candidate_center[along_axis] - selected[along_axis]) < patch_shape_zyx[along_axis] / 2.0
            for selected in selected_centers
        ):
            continue
        selected_centers.append(candidate_center)
        candidates.append(
            Level0PatchCandidate(
                patch_index=len(candidates),
                center_px_zyx=tuple(float(value) for value in candidate_center),
                start_px_zyx=_candidate_start(candidate_center, patch_shape_zyx),
                shape_zyx=patch_shape_zyx,
                scout_score=score,
            )
        )
        if len(candidates) >= count:
            break

    return candidates, {
        "mode": "seam_face_content",
        "orientation": orientation_name,
        "plane": plane,
        "bbox_extent_px_zyx": extents_px,
        "selected_candidate_count": len(candidates),
        "unique_z_count": len({round(candidate.center_px_zyx[0], 3) for candidate in candidates}),
        "unique_along_count": len({round(candidate.center_px_zyx[along_axis], 3) for candidate in candidates}),
    }


def _select_level0_candidates(
    *,
    registration_payload: dict[str, Any],
    edge: dict[str, Any],
    tiles: list[dict[str, Any]],
    spacing_um_zyx: np.ndarray,
    channel: int,
    patch_shape_zyx: tuple[int, int, int],
    patches_per_edge: int,
    image_reader_cache: _ImageLevelReaderCache | None = None,
) -> list[Level0PatchCandidate]:
    source = int(edge["source"])
    target = int(edge["target"])
    candidates = []
    scout_shape_yx = (min(128, patch_shape_zyx[1]), min(128, patch_shape_zyx[2]))
    for patch_index, center_px in enumerate(
        _candidate_centers_from_bbox(
            edge=edge,
            spacing_um_zyx=spacing_um_zyx,
            patch_shape_zyx=patch_shape_zyx,
            patches_per_edge=max(int(patches_per_edge) * 2, int(patches_per_edge)),
        )
    ):
        center_um = center_px * spacing_um_zyx
        source_scout = _sample_registered_center_patch(
            registration_payload=registration_payload,
            tile_record=tiles[source],
            center_zyx_um=center_um,
            spacing_um_zyx=spacing_um_zyx,
            level=2,
            channel=channel,
            patch_shape_yx=scout_shape_yx,
            image_reader_cache=image_reader_cache,
        )
        target_scout = _sample_registered_center_patch(
            registration_payload=registration_payload,
            tile_record=tiles[target],
            center_zyx_um=center_um,
            spacing_um_zyx=spacing_um_zyx,
            level=2,
            channel=channel,
            patch_shape_yx=scout_shape_yx,
            image_reader_cache=image_reader_cache,
        )
        score = min(_content_score(source_scout), _content_score(target_scout))
        candidates.append(
            Level0PatchCandidate(
                patch_index=patch_index,
                center_px_zyx=tuple(float(value) for value in center_px),
                start_px_zyx=_candidate_start(center_px, patch_shape_zyx),
                shape_zyx=patch_shape_zyx,
                scout_score=score,
            )
        )
    candidates.sort(key=lambda candidate: candidate.scout_score, reverse=True)
    return candidates[: max(1, int(patches_per_edge))]


def _select_level0_seam_span_candidate(
    *,
    edge: dict[str, Any],
    spacing_um_zyx: np.ndarray,
    patch_shape_zyx: tuple[int, int, int],
) -> list[Level0PatchCandidate]:
    bbox = _edge_bbox_px(edge, spacing_um_zyx)
    if bbox is None:
        return []
    lo_px, hi_px = bbox
    center_px = (lo_px + hi_px) / 2.0
    extent_px = np.maximum(hi_px - lo_px, 0.0)
    shape_zyx = (
        int(patch_shape_zyx[0]),
        max(1, int(math.ceil(float(extent_px[1])))),
        max(1, int(math.ceil(float(extent_px[2])))),
    )
    return [
        Level0PatchCandidate(
            patch_index=0,
            center_px_zyx=tuple(float(value) for value in center_px),
            start_px_zyx=_candidate_start(center_px, shape_zyx),
            shape_zyx=shape_zyx,
            scout_score=float(shape_zyx[1] * shape_zyx[2]),
        )
    ]


def _sample_registered_center_patch(
    *,
    registration_payload: dict[str, Any],
    tile_record: dict[str, Any],
    center_zyx_um: np.ndarray,
    spacing_um_zyx: np.ndarray,
    level: int,
    channel: int,
    patch_shape_yx: tuple[int, int],
    image_reader_cache: _ImageLevelReaderCache | None = None,
) -> np.ndarray:
    height, width = patch_shape_yx
    path = tile_image_path(registration_payload, tile_record)
    _axes, source_level, _source_shape = _cached_image_level_metadata(
        path,
        level=level,
        image_reader_cache=image_reader_cache,
    )
    level_spacing = np.array(spacing_um_zyx, dtype=np.float64, copy=True)
    level_spacing[1:] *= 2**int(source_level)
    y_values_px = center_zyx_um[1] / spacing_um_zyx[1] + (
        np.arange(height, dtype=np.float64) - height // 2
    ) * level_spacing[1] / spacing_um_zyx[1]
    x_values_px = center_zyx_um[2] / spacing_um_zyx[2] + (
        np.arange(width, dtype=np.float64) - width // 2
    ) * level_spacing[2] / spacing_um_zyx[2]
    return _sample_registered_plane_grid(
        registration_payload=registration_payload,
        tile_record=tile_record,
        fixed_axis=0,
        fixed_value_px=float(center_zyx_um[0] / spacing_um_zyx[0]),
        axis0=1,
        axis0_values_px=y_values_px,
        axis1=2,
        axis1_values_px=x_values_px,
        spacing_um_zyx=spacing_um_zyx,
        level=level,
        channel=channel,
        image_reader_cache=image_reader_cache,
    )


def _sample_registered_plane_grid(
    *,
    registration_payload: dict[str, Any],
    tile_record: dict[str, Any],
    fixed_axis: int,
    fixed_value_px: float,
    axis0: int,
    axis0_values_px: np.ndarray,
    axis1: int,
    axis1_values_px: np.ndarray,
    spacing_um_zyx: np.ndarray,
    level: int,
    channel: int,
    image_reader_cache: _ImageLevelReaderCache | None = None,
) -> np.ndarray:
    from scipy import ndimage as scipy_ndimage

    path = tile_image_path(registration_payload, tile_record)
    axes, source_level, source_shape = _cached_image_level_metadata(
        path,
        level=level,
        image_reader_cache=image_reader_cache,
    )
    shape_zyx = _spatial_shape_zyx(axes=axes, shape=source_shape)
    level_spacing = np.array(spacing_um_zyx, dtype=np.float64, copy=True)
    level_spacing[1:] *= 2**int(source_level)

    axis0_values_px = np.asarray(axis0_values_px, dtype=np.float64)
    axis1_values_px = np.asarray(axis1_values_px, dtype=np.float64)
    axis0_grid, axis1_grid = np.meshgrid(axis0_values_px, axis1_values_px, indexing="ij")
    registered_px = np.empty((axis0_grid.size, 3), dtype=np.float64)
    registered_px[:, fixed_axis] = float(fixed_value_px)
    registered_px[:, axis0] = axis0_grid.ravel()
    registered_px[:, axis1] = axis1_grid.ravel()
    homogeneous = np.column_stack(
        (
            registered_px[:, 0] * spacing_um_zyx[0],
            registered_px[:, 1] * spacing_um_zyx[1],
            registered_px[:, 2] * spacing_um_zyx[2],
            np.ones(registered_px.shape[0], dtype=np.float64),
        )
    ).T

    local_input_um = (np.linalg.inv(_registered_affine(tile_record)) @ homogeneous)[:3].T
    local_um = local_input_um - _stage_translation_um(tile_record)
    coords = np.empty_like(local_um)
    coords[:, 0] = local_um[:, 0] / spacing_um_zyx[0]
    coords[:, 1] = local_um[:, 1] / level_spacing[1]
    coords[:, 2] = local_um[:, 2] / level_spacing[2]
    if not np.all(np.isfinite(coords)):
        return np.zeros((axis0_values_px.size, axis1_values_px.size), dtype=np.float32)

    lo = np.floor(np.nanmin(coords, axis=0)).astype(int) - 2
    hi = np.ceil(np.nanmax(coords, axis=0)).astype(int) + 3
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape_zyx)
    if np.any(hi <= lo):
        return np.zeros((axis0_values_px.size, axis1_values_px.size), dtype=np.float32)

    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lo, hi, strict=True))
    source = _cached_read_image_level_crop(
        path,
        level=source_level,
        axes=axes,
        channel=channel,
        slices_zyx=slices,
        image_reader_cache=image_reader_cache,
    )
    crop_coords = coords - lo[None, :]
    inside = np.all((coords >= 0.0) & (coords <= (shape_zyx - 1)[None, :]), axis=1)
    sampled = scipy_ndimage.map_coordinates(
        source,
        [crop_coords[:, 0], crop_coords[:, 1], crop_coords[:, 2]],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    sampled[~inside] = 0.0
    return sampled.reshape(axis0_values_px.size, axis1_values_px.size).astype(np.float32, copy=False)


def _sample_registered_patch(
    *,
    registration_payload: dict[str, Any],
    tile_record: dict[str, Any],
    center_zyx_um: np.ndarray,
    spacing_um_zyx: np.ndarray,
    level: int,
    channel: int,
    patch_shape_zyx: tuple[int, int, int],
    image_reader_cache: _ImageLevelReaderCache | None = None,
) -> np.ndarray:
    from scipy import ndimage as scipy_ndimage

    path = tile_image_path(registration_payload, tile_record)
    axes, source_level, source_shape = _cached_image_level_metadata(
        path,
        level=level,
        image_reader_cache=image_reader_cache,
    )
    shape_zyx = _spatial_shape_zyx(axes=axes, shape=source_shape)
    level_spacing = np.array(spacing_um_zyx, dtype=np.float64, copy=True)
    level_spacing[1:] *= 2**int(source_level)
    depth, height, width = patch_shape_zyx
    z_um = center_zyx_um[0] + (np.arange(depth, dtype=np.float64) - depth // 2) * level_spacing[0]
    y_um = center_zyx_um[1] + (np.arange(height, dtype=np.float64) - height // 2) * level_spacing[1]
    x_um = center_zyx_um[2] + (np.arange(width, dtype=np.float64) - width // 2) * level_spacing[2]
    zz, yy, xx = np.meshgrid(z_um, y_um, x_um, indexing="ij")
    homogeneous = np.stack([zz, yy, xx, np.ones_like(zz)], axis=0).reshape(4, -1)

    local_input_um = (np.linalg.inv(_registered_affine(tile_record)) @ homogeneous)[:3].T
    local_um = local_input_um - _stage_translation_um(tile_record)
    coords = np.empty_like(local_um)
    coords[:, 0] = local_um[:, 0] / spacing_um_zyx[0]
    coords[:, 1] = local_um[:, 1] / level_spacing[1]
    coords[:, 2] = local_um[:, 2] / level_spacing[2]
    if not np.all(np.isfinite(coords)):
        return np.zeros((depth, height, width), dtype=np.float32)

    lo = np.floor(np.min(coords, axis=0)).astype(int) - 2
    hi = np.ceil(np.max(coords, axis=0)).astype(int) + 3
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape_zyx)
    if np.any(hi <= lo):
        return np.zeros((depth, height, width), dtype=np.float32)

    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lo, hi, strict=True))
    source = _cached_read_image_level_crop(
        path,
        level=source_level,
        axes=axes,
        channel=channel,
        slices_zyx=slices,
        image_reader_cache=image_reader_cache,
    )
    crop_coords = coords - lo[None, :]
    inside = np.all((coords >= 0.0) & (coords <= (shape_zyx - 1)[None, :]), axis=1)
    sampled = scipy_ndimage.map_coordinates(
        source,
        [crop_coords[:, 0], crop_coords[:, 1], crop_coords[:, 2]],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    sampled[~inside] = 0.0
    return sampled.reshape(depth, height, width).astype(np.float32, copy=False)


def _sample_candidate_patch(
    *,
    registration_payload: dict[str, Any],
    tile_record: dict[str, Any],
    candidate: Level0PatchCandidate,
    margin_zyx: tuple[int, int, int],
    spacing_um_zyx: np.ndarray,
    channel: int,
    level: int = 0,
    native_shift_scale_zyx: np.ndarray | None = None,
    image_reader_cache: _ImageLevelReaderCache | None = None,
) -> np.ndarray:
    base_shape_zyx = (
        tuple(int(value) for value in candidate.shape_zyx)
        if native_shift_scale_zyx is None
        else _level_shape_zyx(candidate.shape_zyx, native_shift_scale_zyx)
    )
    expanded_shape = tuple(
        int(base_shape_zyx[index] + 2 * int(margin_zyx[index]))
        for index in range(3)
    )
    return _sample_registered_patch(
        registration_payload=registration_payload,
        tile_record=tile_record,
        center_zyx_um=np.asarray(candidate.center_px_zyx, dtype=np.float64) * spacing_um_zyx,
        spacing_um_zyx=spacing_um_zyx,
        level=level,
        channel=channel,
        patch_shape_zyx=expanded_shape,
        image_reader_cache=image_reader_cache,
    )


def _native_shift_scale_zyx_for_edge_level(
    *,
    registration_payload: dict[str, Any],
    tiles: list[dict[str, Any]],
    source: int,
    target: int,
    level: int,
    image_reader_cache: _ImageLevelReaderCache | None = None,
) -> np.ndarray:
    if int(level) == 0:
        return np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    source_path = tile_image_path(registration_payload, tiles[source])
    target_path = tile_image_path(registration_payload, tiles[target])
    source_axes0, _source_level0, source_shape0 = _cached_image_level_metadata(
        source_path,
        level=0,
        image_reader_cache=image_reader_cache,
    )
    target_axes0, _target_level0, target_shape0 = _cached_image_level_metadata(
        target_path,
        level=0,
        image_reader_cache=image_reader_cache,
    )
    source_axes, source_level, source_shape = _cached_image_level_metadata(
        source_path,
        level=level,
        image_reader_cache=image_reader_cache,
    )
    target_axes, target_level, target_shape = _cached_image_level_metadata(
        target_path,
        level=level,
        image_reader_cache=image_reader_cache,
    )
    if source_level != target_level:
        raise ValueError(
            "MVS refinement requires matching source levels for each edge; "
            f"got source level {source_level} for {source_path} and target level {target_level} for {target_path}"
        )
    if source_axes != source_axes0 or target_axes != target_axes0:
        raise ValueError(
            "MVS refinement requires stable axes across pyramid levels; "
            f"got {source_axes0!r}->{source_axes!r} for {source_path} and "
            f"{target_axes0!r}->{target_axes!r} for {target_path}"
        )
    source_scale = _spatial_shape_zyx(axes=source_axes0, shape=source_shape0).astype(np.float64) / _spatial_shape_zyx(
        axes=source_axes,
        shape=source_shape,
    ).astype(np.float64)
    target_scale = _spatial_shape_zyx(axes=target_axes0, shape=target_shape0).astype(np.float64) / _spatial_shape_zyx(
        axes=target_axes,
        shape=target_shape,
    ).astype(np.float64)
    if not np.all(np.isfinite(source_scale)) or not np.all(source_scale > 0.0):
        raise ValueError(f"Invalid pyramid scale for {source_path} level {source_level}: {source_scale.tolist()}")
    if not np.allclose(source_scale, target_scale, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "MVS refinement requires matching source/target pyramid scales; "
            f"got {source_scale.tolist()} for {source_path} and {target_scale.tolist()} for {target_path}"
        )
    return source_scale


def _phase_refine_shift(
    fixed_patch: np.ndarray,
    moving_patch: np.ndarray,
    *,
    phase_highpass_sigma_zyx: tuple[float, float, float] | None = None,
    phase_upsample_factor: int = 10,
    measure_gradient: bool = True,
) -> tuple[tuple[float, float, float], float, float, float]:
    import cupy as cp

    from squisher_lightsheet.seams import (
        RobustBoundarySettings,
        content_mask_gpu_array,
        mask_fraction_gpu_array,
        phase_correlation_shift_gpu_arrays,
    )

    settings = RobustBoundarySettings(
        patch_shape_zyx=tuple(int(value) for value in fixed_patch.shape),
        min_content_voxels=1024,
        min_content_fraction=0.001,
    )
    fixed_gpu = cp.asarray(np.asarray(fixed_patch, dtype=np.float32))
    moving_gpu = cp.asarray(np.asarray(moving_patch, dtype=np.float32))
    fixed_mask = content_mask_gpu_array(fixed_gpu, settings)
    moving_mask = content_mask_gpu_array(moving_gpu, settings)
    fixed_content = mask_fraction_gpu_array(fixed_mask)
    moving_content = mask_fraction_gpu_array(moving_mask)
    fixed_phase_gpu = fixed_gpu
    moving_phase_gpu = moving_gpu
    if phase_highpass_sigma_zyx is not None and any(float(value) > 0.0 for value in phase_highpass_sigma_zyx):
        import cupyx.scipy.ndimage as cpx_ndimage

        sigma = tuple(float(value) for value in phase_highpass_sigma_zyx)
        fixed_phase_gpu = fixed_gpu - cpx_ndimage.gaussian_filter(fixed_gpu, sigma=sigma)
        moving_phase_gpu = moving_gpu - cpx_ndimage.gaussian_filter(moving_gpu, sigma=sigma)
    shift, peak = phase_correlation_shift_gpu_arrays(
        fixed_phase_gpu,
        moving_phase_gpu,
        fixed_mask,
        moving_mask,
        min_mask_voxels=settings.min_content_voxels,
        upsample_factor=phase_upsample_factor,
    )
    if measure_gradient:
        from squisher_lightsheet.seams import center_z_gradient_component_ncc_after_shift

        _gradient_before, gradient_after = center_z_gradient_component_ncc_after_shift(fixed_patch, moving_patch, shift)
    else:
        gradient_after = float("nan")
    return shift, float(peak), float(gradient_after), float(min(fixed_content, moving_content))


def _is_near_wraparound(
    shift_zyx: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
) -> bool:
    return any(abs(float(shift)) >= max(0.0, float(size) / 2.0 - 1.0) for shift, size in zip(shift_zyx, shape_zyx, strict=True))


def _measure_level0_candidate(
    *,
    fixed_patch: np.ndarray,
    moving_patch: np.ndarray,
    candidate: Level0PatchCandidate,
    patch_origin_px_zyx: tuple[int, int, int] | None = None,
    max_phase_shift_zyx: tuple[float, float, float],
    phase_highpass_sigma_zyx: tuple[float, float, float] | None,
    phase_upsample_factor: int,
) -> dict[str, Any]:
    patch_origin = patch_origin_px_zyx or candidate.start_px_zyx
    crop = _phase_crop_slices(
        fixed_patch=fixed_patch,
        moving_patch=moving_patch,
        max_phase_shift_zyx=max_phase_shift_zyx,
    )
    if crop is None:
        return {
            "patch_index": int(candidate.patch_index),
            "center_px_zyx": [float(value) for value in candidate.center_px_zyx],
            "start_px_zyx": [int(value) for value in candidate.start_px_zyx],
            "shape_zyx": [int(value) for value in candidate.shape_zyx],
            "scout_score": float(candidate.scout_score),
            "shift_px_zyx": [0.0, 0.0, 0.0],
            "peak": float("nan"),
            "gradient_component_ncc_before": None,
            "gradient_component_ncc_after": None,
            "min_content_fraction": 0.0,
            "phase_input_origin_px_zyx": [int(value) for value in patch_origin],
            "phase_crop_start_offset_zyx": [0, 0, 0],
            "phase_crop_start_px_zyx": [int(value) for value in patch_origin],
            "phase_crop_shape_zyx": [0, 0, 0],
            "seed_valid_overlap_fraction": 0.0,
            "shifted_valid_overlap_fraction": 0.0,
            "shifted_fixed_valid_coverage": 0.0,
            "shifted_moving_valid_coverage": 0.0,
            "accepted": False,
            "reject_reason": "no_seed_valid_overlap",
        }
    crop_slices, crop_start, crop_shape, seed_overlap = crop
    fixed_phase_patch = fixed_patch[crop_slices]
    moving_phase_patch = moving_patch[crop_slices]
    shift, peak, _gradient_after, content = _phase_refine_shift(
        fixed_phase_patch,
        moving_phase_patch,
        phase_highpass_sigma_zyx=phase_highpass_sigma_zyx,
        phase_upsample_factor=phase_upsample_factor,
        measure_gradient=False,
    )
    support_metrics = _shifted_valid_support_metrics(
        fixed_patch=fixed_phase_patch,
        moving_patch=moving_phase_patch,
        shift_zyx=shift,
    )
    shift_array = np.asarray(shift, dtype=np.float64)
    reject_reason = None
    if content < 0.001:
        reject_reason = "low_content"
    elif min(
        support_metrics["shifted_fixed_valid_coverage"],
        support_metrics["shifted_moving_valid_coverage"],
    ) < MIN_SHIFTED_VALID_SUPPORT_COVERAGE:
        reject_reason = "low_shifted_valid_overlap"
    elif _is_near_wraparound(shift, crop_shape):
        reject_reason = "phase_shift_near_wraparound"
    elif np.any(np.abs(shift_array) >= np.asarray(max_phase_shift_zyx, dtype=np.float64)):
        reject_reason = "phase_shift_out_of_bounds"
    return {
        "patch_index": int(candidate.patch_index),
        "center_px_zyx": [float(value) for value in candidate.center_px_zyx],
        "start_px_zyx": [int(value) for value in candidate.start_px_zyx],
        "shape_zyx": [int(value) for value in candidate.shape_zyx],
        "scout_score": float(candidate.scout_score),
        "shift_px_zyx": [float(value) for value in shift],
        "peak": float(peak),
        "gradient_component_ncc_before": None,
        "gradient_component_ncc_after": None,
        "min_content_fraction": float(content),
        "phase_input_origin_px_zyx": [int(value) for value in patch_origin],
        "phase_crop_start_offset_zyx": [int(value) for value in crop_start],
        "phase_crop_start_px_zyx": [
            int(patch_origin[index] + crop_start[index])
            for index in range(3)
        ],
        "phase_crop_shape_zyx": [int(value) for value in crop_shape],
        "seed_valid_overlap_fraction": float(seed_overlap),
        **support_metrics,
        "accepted": reject_reason is None,
        "reject_reason": reject_reason,
    }


def score_mvs_edges_with_gradient_ncc(
    registration_payload: dict[str, Any],
    *,
    level: int = 2,
    channel: int = 0,
    patch_shape_yx: tuple[int, int] = (256, 256),
    phase_patch_shape_zyx: tuple[int, int, int] = (32, 256, 256),
    min_gradient_ncc: float = 0.15,
    max_phase_shift_native_zyx: tuple[float, float, float] = (16.0, 96.0, 96.0),
    phase_refine_bad_gradients: bool = False,
    used_edges_only: bool = True,
    max_cached_tiles: int = 2,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from squisher_lightsheet.seams import center_z_gradient_component_ncc

    payload = deepcopy(registration_payload)
    tiles = payload["tiles"]
    spacing_um_zyx = np.asarray(
        [
            payload["spacing_um"]["z"],
            payload["spacing_um"]["y"],
            payload["spacing_um"]["x"],
        ],
        dtype=np.float64,
    )
    used_edges = mvs_used_edge_set(payload)

    scored = []
    skipped = []
    for edge in payload["metrics"]["pairwise_registration"]["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        pair = normalize_mvs_edge((source, target))
        if used_edges_only and used_edges and pair not in used_edges:
            continue
        center_um = _edge_bbox_center_um(edge)
        if center_um is None:
            skipped.append({"pair": list(pair), "reason": "missing_bbox"})
            continue
        source_patch = _sample_registered_center_patch(
            registration_payload=payload,
            tile_record=tiles[source],
            center_zyx_um=center_um,
            spacing_um_zyx=spacing_um_zyx,
            level=level,
            channel=channel,
            patch_shape_yx=patch_shape_yx,
        )
        target_patch = _sample_registered_center_patch(
            registration_payload=payload,
            tile_record=tiles[target],
            center_zyx_um=center_um,
            spacing_um_zyx=spacing_um_zyx,
            level=level,
            channel=channel,
            patch_shape_yx=patch_shape_yx,
        )
        score = center_z_gradient_component_ncc(source_patch, target_patch)
        attrs = edge.setdefault("attrs", {})
        attrs["gradient_component_ncc_before_phase"] = None if not np.isfinite(score) else float(score)
        attrs["gradient_component_ncc_source"] = "registered_center_z_patch"
        phase_shift = None
        phase_peak = None
        phase_content = None
        phase_refined = False
        phase_reject_reason = None
        if phase_refine_bad_gradients and (not np.isfinite(score) or float(score) < float(min_gradient_ncc)):
            original_score = score
            fixed_volume = _sample_registered_patch(
                registration_payload=payload,
                tile_record=tiles[source],
                center_zyx_um=center_um,
                spacing_um_zyx=spacing_um_zyx,
                level=level,
                channel=channel,
                patch_shape_zyx=phase_patch_shape_zyx,
            )
            moving_volume = _sample_registered_patch(
                registration_payload=payload,
                tile_record=tiles[target],
                center_zyx_um=center_um,
                spacing_um_zyx=spacing_um_zyx,
                level=level,
                channel=channel,
                patch_shape_zyx=phase_patch_shape_zyx,
            )
            phase_shift, phase_peak, refined_score, phase_content = _phase_refine_shift(fixed_volume, moving_volume)
            phase_refined = True
            if np.isfinite(refined_score):
                phase_shift_native = np.asarray(phase_shift, dtype=np.float64)
                phase_shift_native[1:] *= 2**int(level)
                attrs["phase_refined_shift_level_px_zyx"] = [float(value) for value in phase_shift]
                attrs["phase_refined_shift_native_px_zyx"] = [float(value) for value in phase_shift_native]
                if np.any(np.abs(phase_shift_native) > np.asarray(max_phase_shift_native_zyx, dtype=np.float64)):
                    phase_reject_reason = "phase_shift_out_of_bounds"
                    score = original_score
                else:
                    score = float(refined_score)
                    base_target_delta_px = -mvs_transform_matrix(attrs["transform"])[:3, 3] / spacing_um_zyx
                    attrs["target_correction_delta_px_zyx"] = [
                        float(value) for value in (base_target_delta_px + phase_shift_native)
                    ]
        attrs["gradient_component_ncc_after"] = None if not np.isfinite(score) else float(score)
        attrs["phase_refined_bad_gradient"] = bool(phase_refined)
        attrs["phase_refined_peak"] = phase_peak
        attrs["phase_refined_min_content_fraction"] = phase_content
        attrs["phase_refined_reject_reason"] = phase_reject_reason
        scored.append(
            {
                "source": source,
                "target": target,
                "pair": list(pair),
                "gradient_component_ncc_before_phase": attrs["gradient_component_ncc_before_phase"],
                "gradient_component_ncc_after": None if not np.isfinite(score) else float(score),
                "phase_refined_bad_gradient": bool(phase_refined),
                "phase_refined_shift_level_px_zyx": None if phase_shift is None else [float(value) for value in phase_shift],
                "phase_refined_peak": phase_peak,
                "phase_refined_min_content_fraction": phase_content,
                "phase_refined_reject_reason": phase_reject_reason,
                "mvs_quality": mvs_data_scalar(edge.get("quality", edge.get("attrs", {}).get("quality", float("nan")))),
            }
        )

    summary = {
        "artifact_type": "lightsheet.mvs_seam_gradient_ncc_scoring.v1",
        "level": int(level),
        "channel": int(channel),
        "patch_shape_yx": [int(patch_shape_yx[0]), int(patch_shape_yx[1])],
        "phase_patch_shape_zyx": [int(value) for value in phase_patch_shape_zyx],
        "min_gradient_ncc": float(min_gradient_ncc),
        "max_phase_shift_native_zyx": [float(value) for value in max_phase_shift_native_zyx],
        "phase_refine_bad_gradients": bool(phase_refine_bad_gradients),
        "used_edges_only": bool(used_edges_only),
        "max_cached_tiles": int(max_cached_tiles),
        "scored_edge_count": len(scored),
        "skipped_edge_count": len(skipped),
        "scored_edges": scored,
        "skipped_edges": skipped,
    }
    payload.setdefault("metrics", {})["gradient_ncc_edge_scoring"] = summary
    return payload, summary


def mvs_edge_score(edge: dict[str, Any]) -> tuple[float, str]:
    """Return raw MVS pairwise quality for translation-only seam stitching."""

    return mvs_data_scalar(edge.get("quality", edge.get("attrs", {}).get("quality", float("nan")))), "mvs_quality"


def mvs_used_edge_set(registration_payload: dict[str, Any]) -> set[tuple[int, int]]:
    used_edges = (
        registration_payload.get("metrics", {})
        .get("groupwise_resolution", {})
        .get("metrics", {})
        .get("used_edges", {})
    )
    if isinstance(used_edges, dict):
        used_edges = used_edges.get("0") or used_edges.get(0) or []
    return {normalize_mvs_edge(edge) for edge in used_edges}


def mvs_edge_residuals(registration_payload: dict[str, Any]) -> dict[tuple[int, int], float]:
    residuals = (
        registration_payload.get("metrics", {})
        .get("groupwise_resolution", {})
        .get("metrics", {})
        .get("edge_residuals", {})
    )
    if isinstance(residuals, dict) and ("0" in residuals or 0 in residuals):
        residuals = residuals.get("0") or residuals.get(0) or {}
    if not isinstance(residuals, dict):
        return {}
    return {normalize_mvs_edge(edge): float(value) for edge, value in residuals.items()}


def mvs_measured_edge_records(registration_payload: dict[str, Any]) -> list[dict[str, Any]]:
    edges = (
        registration_payload.get("metrics", {})
        .get("pairwise_registration", {})
        .get("edges", [])
    )
    records = []
    residuals = mvs_edge_residuals(registration_payload)
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        pair = normalize_mvs_edge((source, target))
        quality = mvs_data_scalar(edge.get("quality", edge.get("attrs", {}).get("quality", float("nan"))))
        score, score_source = mvs_edge_score(edge)
        records.append(
            {
                "source": source,
                "target": target,
                "pair": list(pair),
                "quality": quality,
                "score": score,
                "score_source": score_source,
                "residual_um": residuals.get(pair),
            }
        )
    return records


def mvs_used_edge_audit(registration_payload: dict[str, Any]) -> dict[str, Any]:
    measured_records = mvs_measured_edge_records(registration_payload)
    measured = {tuple(record["pair"]) for record in measured_records}
    used = mvs_used_edge_set(registration_payload)
    dropped = measured - used if used else set()
    residuals = mvs_edge_residuals(registration_payload)
    record_by_pair = {tuple(record["pair"]): record for record in measured_records}

    def edge_summary(pair: tuple[int, int]) -> dict[str, Any]:
        record = dict(record_by_pair.get(pair, {"pair": list(pair)}))
        record["pair"] = list(pair)
        record["used"] = pair in used if used else None
        record["residual_um"] = residuals.get(pair)
        return record

    measured_residual_pairs = [pair for pair in measured if pair in residuals]
    used_residual_pairs = [pair for pair in used if pair in residuals]
    max_measured_pair = max(measured_residual_pairs, key=lambda pair: residuals[pair], default=None)
    max_used_pair = max(used_residual_pairs, key=lambda pair: residuals[pair], default=None)

    return {
        "measured_edge_count": len(measured),
        "used_edge_count": len(used),
        "dropped_edge_count": len(dropped),
        "used_edges_present": bool(used),
        "dropped_edges": [edge_summary(pair) for pair in sorted(dropped)],
        "max_measured_residual_edge": None if max_measured_pair is None else edge_summary(max_measured_pair),
        "max_used_residual_edge": None if max_used_pair is None else edge_summary(max_used_pair),
        "measured_edges": [edge_summary(pair) for pair in sorted(measured)],
        "used_edges": [edge_summary(pair) for pair in sorted(used)],
    }


def mvs_pairwise_edges(
    registration_payload: dict[str, Any],
    *,
    min_quality: float = DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    used_edges_only: bool = True,
) -> list[dict[str, Any]]:
    edges = (
        registration_payload.get("metrics", {})
        .get("pairwise_registration", {})
        .get("edges", [])
    )
    used_edges = mvs_used_edge_set(registration_payload)
    filtered = []
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        pair = tuple(sorted((source, target)))
        used_by_groupwise = pair in used_edges if used_edges else True
        if used_edges_only and used_edges and not used_by_groupwise:
            continue
        quality = mvs_data_scalar(edge.get("quality", edge.get("attrs", {}).get("quality", float("nan"))))
        score, score_source = mvs_edge_score(edge)
        if not np.isfinite(score) or score < min_quality:
            continue
        matrix = mvs_transform_matrix(edge["attrs"]["transform"])
        if matrix.shape[0] < 3 or matrix.shape[1] < 4:
            raise ValueError(f"MVS edge {pair} has invalid transform shape {matrix.shape}")
        filtered.append(
            {
                "source": source,
                "target": target,
                "pair": [source, target],
                "quality": quality,
                "score": score,
                "score_source": score_source,
                "translation_um_zyx": matrix[:3, 3].astype(float),
                "attrs": edge.get("attrs", {}),
                "used_by_groupwise": bool(used_by_groupwise),
            }
        )
    return filtered


def mvs_seam_constraints(
    registration_payload: dict[str, Any],
    *,
    tile_names: list[str],
    spacing_um_zyx: np.ndarray,
    min_quality: float = DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    used_edges_only: bool = True,
    unused_edge_weight_scale: float = 1.0,
) -> list[dict[str, Any]]:
    if unused_edge_weight_scale < 0.0:
        raise ValueError("unused_edge_weight_scale must be non-negative")
    registration_tiles = [str(record["tile"]) for record in registration_payload["tiles"]]
    tile_index = {tile: index for index, tile in enumerate(tile_names)}
    constraints = []
    for edge in mvs_pairwise_edges(
        registration_payload,
        min_quality=min_quality,
        used_edges_only=used_edges_only,
    ):
        fixed_tile = registration_tiles[edge["source"]]
        moving_tile = registration_tiles[edge["target"]]
        if fixed_tile not in tile_index or moving_tile not in tile_index:
            continue
        attrs = edge.get("attrs", {})
        if attrs.get("target_correction_delta_px_zyx") is not None:
            target_delta_px = np.asarray(attrs["target_correction_delta_px_zyx"], dtype=float)
            delta_source = "target_correction_delta_px_zyx"
        else:
            target_delta_px = -np.asarray(edge["translation_um_zyx"], dtype=float) / spacing_um_zyx
            delta_source = "mvs_pairwise_transform"
        quality = float(edge["score"])
        weight_scale = 1.0 if edge["used_by_groupwise"] else float(unused_edge_weight_scale)
        constraints.append(
            {
                "fixed": fixed_tile,
                "moving": moving_tile,
                "fixed_index": tile_index[fixed_tile],
                "moving_index": tile_index[moving_tile],
                "pair": edge["pair"],
                "axis": "mvs_pairwise",
                "patch_index": -1,
                "target_correction_delta_px": target_delta_px,
                "target_correction_delta_source": delta_source,
                "corr_after": quality,
                "corr_before": None,
                "weight": max(quality - 0.15, 1e-3) * weight_scale,
                "source": "mvs_pairwise_registration",
                "score_source": edge["score_source"],
                "mvs_quality": float(edge["quality"]),
                "mvs_translation_um_zyx": edge["translation_um_zyx"].tolist(),
                "phase_refined_bad_gradient": attrs.get("phase_refined_bad_gradient"),
                "phase_refined_shift_native_px_zyx": attrs.get("phase_refined_shift_native_px_zyx"),
                "mvs_used_by_groupwise": bool(edge["used_by_groupwise"]),
                "unused_edge_weight_scale": weight_scale,
            }
        )
    if not constraints:
        raise ValueError("No MVS seam constraints passed the quality/tile filters")
    return constraints


def _inlier_shift_cluster(
    accepted_patches: list[dict[str, Any]],
    *,
    min_inliers: int,
    max_final_residual_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    from squisher_lightsheet.tile_phase import select_inlier_patch_measurements

    shifts = np.asarray([patch["shift_px_zyx"] for patch in accepted_patches], dtype=np.float64)
    thresholds = np.maximum(np.asarray(max_final_residual_zyx, dtype=np.float64), np.asarray([1.0, 2.0, 2.0]))
    return select_inlier_patch_measurements(shifts, thresholds_zyx=thresholds, min_inliers=min_inliers)


def _has_spatial_diversity(patches: list[dict[str, Any]]) -> bool:
    if len(patches) <= 1:
        return True
    centers = np.asarray([patch["center_px_zyx"] for patch in patches], dtype=np.float64)
    return bool(np.any(np.ptp(centers[:, 1:3], axis=0) > 0.0))


def _edge_constraint_from_measurement(
    *,
    edge: dict[str, Any],
    edge_shift_px_zyx: np.ndarray,
    weight: float,
    axis: str = "mvs_level0",
    source_label: str = "mvs_level0_refinement",
) -> Any:
    from squisher_lightsheet.seams import BoundaryConstraint

    source = int(edge["source"])
    target = int(edge["target"])
    return BoundaryConstraint(
        fixed=source,
        moving=target,
        pair=(source, target),
        axis=axis,
        patch_index=-1,
        shift_zyx=tuple(float(value) for value in edge_shift_px_zyx),
        weight=float(max(weight, 1e-6)),
        correlation_before=float("nan"),
        correlation_after=float("nan"),
        improvement=0.0,
        fixed_nonzero_fraction=0.0,
        moving_nonzero_fraction=0.0,
        fixed_std=0.0,
        moving_std=0.0,
        accepted=True,
        source_label=source_label,
    )


def _edge_constraint_from_level2_fallback(
    *,
    edge: dict[str, Any],
    spacing_um_zyx: np.ndarray,
    weight_scale: float,
) -> Any:
    if weight_scale < 0.0:
        raise ValueError("weight_scale must be non-negative")
    shift_px_zyx = -np.asarray(edge["translation_um_zyx"], dtype=np.float64) / spacing_um_zyx
    return _edge_constraint_from_measurement(
        edge=edge,
        edge_shift_px_zyx=shift_px_zyx,
        weight=max(float(edge["score"]), 1e-3) * float(weight_scale),
        axis="mvs_level2_fallback",
        source_label=LEVEL2_FALLBACK_SOURCE_LABEL,
    )


def _synthetic_solver_tiles(*, n_tiles: int, spacing_um_zyx: np.ndarray) -> list[Any]:
    from squisher_lightsheet.registration import TileMetadata, TrackMetadata

    spacing = {dim: float(value) for dim, value in zip(DIMENSIONS, spacing_um_zyx, strict=True)}
    track = TrackMetadata(slug="mvs_level0", track_id="mvs_level0", channels=(0,), channel_names=("0",))
    return [
        TileMetadata(
            path=Path(f"tile_{index:05d}.zarr"),
            shape=(1, 1, 1),
            axes="ZYX",
            spacing=spacing,
            translation={dim: 0.0 for dim in DIMENSIONS},
            channels=("0",),
            tracks=(track,),
        )
        for index in range(n_tiles)
    ]


def _solve_level0_edge_corrections(
    *,
    n_tiles: int,
    constraints: list[Any],
    spacing_um_zyx: np.ndarray,
    max_correction_zyx: tuple[float, float, float],
    max_final_residual_zyx: tuple[float, float, float],
    max_disconnected_island_size: int = DEFAULT_LEVEL0_MAX_DISCONNECTED_ISLAND_SIZE,
) -> tuple[np.ndarray, list[Any], int, dict[str, Any]]:
    from squisher_lightsheet.registration import solve_tile_corrections_with_residual_rejection
    from squisher_lightsheet.seams import RobustBoundarySettings

    if max_disconnected_island_size < 0:
        raise ValueError("max_disconnected_island_size must be >= 0")
    if not constraints:
        raise ValueError("No accepted level-0 seam constraints are available for solving")
    settings = RobustBoundarySettings(
        max_correction_zyx=max_correction_zyx,
        max_final_residual_zyx=max_final_residual_zyx,
        min_inlier_patches_per_edge=1,
    )
    solver_tiles = _synthetic_solver_tiles(n_tiles=n_tiles, spacing_um_zyx=spacing_um_zyx)
    corrections, annotated, anchor_tile = solve_tile_corrections_with_residual_rejection(
        solver_tiles,
        constraints,
        settings,
        fixed_axes=set(),
    )
    protected_indices = {
        index
        for index, constraint in enumerate(constraints)
        if constraint.source_label == LEVEL2_FALLBACK_SOURCE_LABEL
    }
    if protected_indices:
        corrections, annotated = _solve_with_protected_level2_fallbacks(
            n_tiles=n_tiles,
            solver_tiles=solver_tiles,
            constraints=constraints,
            initial_constraints=annotated,
            anchor_tile=int(anchor_tile),
            settings=settings,
            max_final_residual_zyx=max_final_residual_zyx,
            protected_indices=protected_indices,
        )

    connectivity = _accepted_constraint_connectivity(
        n_tiles=n_tiles,
        constraints=annotated,
        anchor_tile=int(anchor_tile),
    )
    largest_island_size = int(connectivity["largest_disconnected_island_size"])
    if largest_island_size > max_disconnected_island_size:
        raise RuntimeError(
            "Level-0 optimizer produced a disconnected island that is too large after residual rejection: "
            f"largest disconnected island has {largest_island_size} tile(s), "
            f"allowed {max_disconnected_island_size}; disconnected island sizes="
            f"{connectivity['disconnected_island_sizes']}, anchor_tile={anchor_tile}"
        )
    return np.asarray(corrections, dtype=np.float64), annotated, int(anchor_tile), connectivity


def _solve_with_protected_level2_fallbacks(
    *,
    n_tiles: int,
    solver_tiles: list[Any],
    constraints: list[Any],
    initial_constraints: list[Any],
    anchor_tile: int,
    settings: Any,
    max_final_residual_zyx: tuple[float, float, float],
    protected_indices: set[int],
) -> tuple[list[tuple[float, float, float]], list[Any]]:
    from squisher_lightsheet.registration import solve_tile_corrections_zyx

    active_indices = {
        index
        for index, constraint in enumerate(initial_constraints)
        if index in protected_indices or constraint.reject_reason != "high_final_residual"
    }
    max_residual = np.asarray(max_final_residual_zyx, dtype=np.float64)
    for _ in range(len(constraints) + n_tiles + 1):
        active_constraints = [constraints[index] for index in sorted(active_indices)]
        corrections_raw = solve_tile_corrections_zyx(
            n_tiles,
            active_constraints,
            settings,
            anchor_tile,
        )
        corrections = np.asarray(corrections_raw, dtype=np.float64)
        connected_tiles = set(
            _accepted_constraint_connectivity(
                n_tiles=n_tiles,
                constraints=active_constraints,
                anchor_tile=anchor_tile,
            )["anchor_component_tiles"]
        )
        next_active_indices = set(protected_indices)
        next_constraints = []
        for index, constraint in enumerate(constraints):
            residual = tuple(
                float(
                    corrections[int(constraint.moving), dim]
                    - corrections[int(constraint.fixed), dim]
                    - constraint.shift_zyx[dim]
                )
                for dim in range(3)
            )
            if index in protected_indices:
                next_constraints.append(
                    replace(
                        constraint,
                        accepted=True,
                        reject_reason=None,
                        final_residual_zyx=residual,
                    )
                )
                continue
            disconnected = (
                int(constraint.fixed) not in connected_tiles
                or int(constraint.moving) not in connected_tiles
            )
            high_residual = any(abs(residual[dim]) > max_residual[dim] for dim in range(3))
            accepted = not disconnected and not high_residual
            if accepted:
                next_active_indices.add(index)
            next_constraints.append(
                replace(
                    constraint,
                    accepted=accepted,
                    reject_reason=(
                        "disconnected_from_anchor"
                        if disconnected
                        else "high_final_residual"
                        if high_residual
                        else None
                    ),
                    final_residual_zyx=residual,
                )
            )
        if next_active_indices == active_indices:
            return [tuple(float(value) for value in row) for row in corrections], next_constraints
        active_indices = next_active_indices
    raise RuntimeError("Protected level-2 fallback correction solve did not converge")


def _accepted_constraint_connectivity(
    *,
    n_tiles: int,
    constraints: list[Any],
    anchor_tile: int,
) -> dict[str, Any]:
    neighbors: list[set[int]] = [set() for _ in range(n_tiles)]
    for constraint in constraints:
        if not constraint.accepted and constraint.reject_reason != "disconnected_from_anchor":
            continue
        fixed = int(constraint.fixed)
        moving = int(constraint.moving)
        neighbors[fixed].add(moving)
        neighbors[moving].add(fixed)

    components: list[list[int]] = []
    seen: set[int] = set()
    for tile in range(n_tiles):
        if tile in seen:
            continue
        component = []
        queue = deque([tile])
        seen.add(tile)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda component: (-len(component), component[0]))

    anchor_component = next(
        (component for component in components if int(anchor_tile) in component),
        [int(anchor_tile)],
    )
    disconnected_tiles = sorted(set(range(n_tiles)) - set(anchor_component))
    disconnected_components = [component for component in components if int(anchor_tile) not in component]
    disconnected_island_sizes = [len(component) for component in disconnected_components]
    return {
        "connected_tile_count": len(anchor_component),
        "disconnected_tile_count": len(disconnected_tiles),
        "component_sizes": [len(component) for component in components],
        "disconnected_island_sizes": disconnected_island_sizes,
        "largest_disconnected_island_size": max(disconnected_island_sizes, default=0),
        "anchor_component_tiles": anchor_component,
        "disconnected_tiles": disconnected_tiles,
    }


def _apply_registered_affine_corrections(
    *,
    registration_payload: dict[str, Any],
    corrections_px: np.ndarray,
    spacing_um_zyx: np.ndarray,
) -> None:
    for index, tile_record in enumerate(registration_payload["tiles"]):
        if "registered_affine" not in tile_record or "matrix" not in tile_record["registered_affine"]:
            raise ValueError(f"Tile {tile_record.get('tile', index)!r} is missing registered_affine.matrix")
        matrix = np.asarray(tile_record["registered_affine"]["matrix"], dtype=np.float64).copy()
        if matrix.shape[0] < 3 or matrix.shape[1] < 4:
            raise ValueError(f"Tile {tile_record.get('tile', index)!r} has invalid registered_affine shape {matrix.shape}")
        matrix[:3, 3] += corrections_px[index] * spacing_um_zyx
        tile_record["registered_affine"]["matrix"] = matrix.tolist()


def _constraint_diagnostics(constraints: list[Any], tile_names: list[str]) -> list[dict[str, Any]]:
    records = []
    for constraint in constraints:
        records.append(
            {
                "fixed": tile_names[int(constraint.fixed)],
                "moving": tile_names[int(constraint.moving)],
                "pair": [int(constraint.fixed), int(constraint.moving)],
                "axis": constraint.axis,
                "source_label": constraint.source_label,
                "shift_px_zyx": [float(value) for value in constraint.shift_zyx],
                "weight": float(constraint.weight),
                "accepted": bool(constraint.accepted),
                "reject_reason": constraint.reject_reason,
                "final_residual_px_zyx": None
                if constraint.final_residual_zyx is None
                else [float(value) for value in constraint.final_residual_zyx],
            }
        )
    return records


def _zyx_abs_stats(values: list[tuple[float, float, float]]) -> dict[str, list[float]] | None:
    if not values:
        return None
    residuals = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "median_abs_px_zyx": [float(value) for value in np.median(residuals, axis=0)],
        "p95_abs_px_zyx": [float(value) for value in np.percentile(residuals, 95.0, axis=0)],
        "max_abs_px_zyx": [float(value) for value in np.max(residuals, axis=0)],
        "mean_abs_px_zyx": [float(value) for value in np.mean(residuals, axis=0)],
    }


def _constraint_source_summary(constraints: list[Any]) -> dict[str, dict[str, Any]]:
    by_axis = sorted({str(constraint.axis) for constraint in constraints})
    summary = {}
    for axis in by_axis:
        axis_constraints = [constraint for constraint in constraints if str(constraint.axis) == axis]
        accepted = [constraint for constraint in axis_constraints if constraint.accepted]
        residuals = [
            tuple(float(value) for value in constraint.final_residual_zyx)
            for constraint in accepted
            if constraint.final_residual_zyx is not None
        ]
        reject_reasons = Counter(
            str(constraint.reject_reason)
            for constraint in axis_constraints
            if not constraint.accepted and constraint.reject_reason is not None
        )
        summary[axis] = {
            "constraint_count": len(axis_constraints),
            "accepted_count": len(accepted),
            "rejected_count": len(axis_constraints) - len(accepted),
            "reject_reasons": dict(sorted(reject_reasons.items())),
            "residual_stats": _zyx_abs_stats(residuals),
        }
    return summary


def _correction_summary(
    *,
    corrections_px: np.ndarray,
    spacing_um_zyx: np.ndarray,
    max_correction_zyx: tuple[float, float, float],
    tile_names: list[str],
) -> dict[str, Any]:
    corrections_px = np.asarray(corrections_px, dtype=np.float64)
    corrections_um = corrections_px * np.asarray(spacing_um_zyx, dtype=np.float64)
    limits = np.asarray(max_correction_zyx, dtype=np.float64)
    clamp_hits = np.isclose(np.abs(corrections_px), limits, rtol=0.0, atol=1e-6)

    def ranges(values: np.ndarray) -> dict[str, list[float]]:
        return {
            "min_zyx": [float(value) for value in np.min(values, axis=0)],
            "max_zyx": [float(value) for value in np.max(values, axis=0)],
            "median_zyx": [float(value) for value in np.median(values, axis=0)],
            "mean_zyx": [float(value) for value in np.mean(values, axis=0)],
        }

    return {
        "px": ranges(corrections_px),
        "um": ranges(corrections_um),
        "clamp_hit_count_zyx": [int(value) for value in np.sum(clamp_hits, axis=0)],
        "clamp_hit_tiles_zyx": [
            [tile_names[int(index)] for index in np.where(clamp_hits[:, axis])[0]]
            for axis in range(3)
        ],
    }


def _optimization_summary(
    *,
    constraints: list[Any],
    corrections_px: np.ndarray,
    spacing_um_zyx: np.ndarray,
    max_correction_zyx: tuple[float, float, float],
    solver_connectivity: dict[str, Any],
    tile_names: list[str],
) -> dict[str, Any]:
    accepted = [constraint for constraint in constraints if constraint.accepted]
    rejected = [constraint for constraint in constraints if not constraint.accepted]
    residuals = [
        tuple(float(value) for value in constraint.final_residual_zyx)
        for constraint in accepted
        if constraint.final_residual_zyx is not None
    ]
    reject_reasons = Counter(
        str(constraint.reject_reason)
        for constraint in rejected
        if constraint.reject_reason is not None
    )
    worst = sorted(
        [
            constraint
            for constraint in accepted
            if constraint.final_residual_zyx is not None
        ],
        key=lambda constraint: max(abs(float(value)) for value in constraint.final_residual_zyx),
        reverse=True,
    )[:10]
    return {
        "constraint_count": len(constraints),
        "accepted_constraint_count": len(accepted),
        "rejected_constraint_count": len(rejected),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "constraint_counts_by_axis": _constraint_source_summary(constraints),
        "residual_stats": _zyx_abs_stats(residuals),
        "worst_accepted_residuals": [
            {
                "pair": [int(constraint.fixed), int(constraint.moving)],
                "fixed": tile_names[int(constraint.fixed)],
                "moving": tile_names[int(constraint.moving)],
                "axis": constraint.axis,
                "source_label": constraint.source_label,
                "shift_px_zyx": [float(value) for value in constraint.shift_zyx],
                "final_residual_px_zyx": [float(value) for value in constraint.final_residual_zyx],
                "max_abs_residual_px": float(max(abs(float(value)) for value in constraint.final_residual_zyx)),
                "weight": float(constraint.weight),
            }
            for constraint in worst
        ],
        "correction_summary": _correction_summary(
            corrections_px=corrections_px,
            spacing_um_zyx=spacing_um_zyx,
            max_correction_zyx=max_correction_zyx,
            tile_names=tile_names,
        ),
        "connectivity": {
            "connected_tile_count": int(solver_connectivity["connected_tile_count"]),
            "disconnected_tile_count": int(solver_connectivity["disconnected_tile_count"]),
            "component_sizes": [int(value) for value in solver_connectivity["component_sizes"]],
            "largest_disconnected_island_size": int(solver_connectivity["largest_disconnected_island_size"]),
        },
    }


def refine_mvs_registration_level0(
    *,
    registration_input: Path,
    output_registration: Path,
    channel: int = 0,
    patch_shape_zyx: tuple[int, int, int] = (12, 320, 320),
    candidate_mode: Level0CandidateMode = "scout",
    patches_per_edge: int = DEFAULT_LEVEL0_PATCHES_PER_EDGE,
    retry_patches_per_edge: int = DEFAULT_LEVEL0_RETRY_PATCHES_PER_EDGE,
    min_inliers: int = 3,
    max_phase_shift_zyx: tuple[float, float, float] = (3.0, 64.0, 64.0),
    phase_highpass_sigma_zyx: tuple[float, float, float] | None = (0.0, 10.0, 10.0),
    phase_upsample_factor: int = 10,
    max_correction_zyx: tuple[float, float, float] = (4.0, 64.0, 64.0),
    max_final_residual_zyx: tuple[float, float, float] = (2.0, 8.0, 8.0),
    min_quality: float = DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    used_edges_only: bool = True,
    max_edges: int | None = None,
    max_disconnected_island_size: int = DEFAULT_LEVEL0_MAX_DISCONNECTED_ISLAND_SIZE,
    fallback_refinement_levels: tuple[int, ...] = (),
    fallback_level2_weight_scale: float = DEFAULT_LEVEL0_FALLBACK_LEVEL2_WEIGHT_SCALE,
    workers: int = 1,
    render_contact_sheet: bool = True,
    contact_sheet_output_dir: Path | None = None,
    contact_sheet_max_panels: int = 128,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if any(value < 1 for value in patch_shape_zyx):
        raise ValueError(f"patch_shape_zyx must be positive, got {patch_shape_zyx}")
    if candidate_mode not in ("scout", "seam-span"):
        raise ValueError(f"candidate_mode must be 'scout' or 'seam-span', got {candidate_mode!r}")
    if patches_per_edge < 1:
        raise ValueError("patches_per_edge must be >= 1")
    if retry_patches_per_edge < patches_per_edge:
        raise ValueError("retry_patches_per_edge must be >= patches_per_edge")
    if min_inliers < 1:
        raise ValueError("min_inliers must be >= 1")
    if phase_highpass_sigma_zyx is not None and any(value < 0.0 for value in phase_highpass_sigma_zyx):
        raise ValueError(f"phase_highpass_sigma_zyx must be non-negative, got {phase_highpass_sigma_zyx}")
    if phase_upsample_factor < 1:
        raise ValueError("phase_upsample_factor must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if max_disconnected_island_size < 0:
        raise ValueError("max_disconnected_island_size must be >= 0")
    fallback_refinement_levels = tuple(dict.fromkeys(int(level) for level in fallback_refinement_levels))
    if any(level < 1 for level in fallback_refinement_levels):
        raise ValueError(f"fallback_refinement_levels must be >= 1, got {fallback_refinement_levels}")
    if fallback_level2_weight_scale < 0.0:
        raise ValueError("fallback_level2_weight_scale must be non-negative")
    if render_contact_sheet and contact_sheet_max_panels < 1:
        raise ValueError("contact_sheet_max_panels must be >= 1")
    payload = load_mvs_registration(registration_input)
    refined = deepcopy(payload)
    output_registration = output_registration.resolve()
    registration_hash = _registration_hash_from_path(registration_input)
    spacing_um_zyx = _zyx(refined, "spacing_um")
    tiles = refined["tiles"]
    tile_names = [str(tile["tile"]) for tile in tiles]
    selected_edges = mvs_pairwise_edges(refined, min_quality=min_quality, used_edges_only=used_edges_only)
    level2_fallback_edges = (
        mvs_pairwise_edges(refined, min_quality=min_quality, used_edges_only=False)
        if fallback_level2_weight_scale > 0.0
        else []
    )
    if max_edges is not None:
        selected_edges = selected_edges[: int(max_edges)]
    if not selected_edges:
        raise ValueError("No MVS edges passed the level-0 refinement filters")

    accepted_constraints = []
    measured_edges = []
    rejected_edges = []
    accepted_level0_pairs: set[tuple[int, int]] = set()
    image_reader_cache = _ImageLevelReaderCache()
    try:
        for edge in selected_edges:
            source = int(edge["source"])
            target = int(edge["target"])
            edge_key = f"{source}_{target}"
            if progress is not None:
                progress(f"Measuring MVS level-0 seam edge {edge_key}")
            if candidate_mode == "seam-span":
                candidates = _select_level0_seam_span_candidate(
                    edge=edge,
                    spacing_um_zyx=spacing_um_zyx,
                    patch_shape_zyx=patch_shape_zyx,
                )
            else:
                candidates = _select_level0_candidates(
                    registration_payload=refined,
                    edge=edge,
                    tiles=tiles,
                    spacing_um_zyx=spacing_um_zyx,
                    channel=channel,
                    patch_shape_zyx=patch_shape_zyx,
                    patches_per_edge=patches_per_edge,
                    image_reader_cache=image_reader_cache,
                )
            if not candidates:
                rejected_edges.append({"pair": [source, target], "reason": "missing_bbox_or_candidates"})
                continue

            def measure_edge_candidates(
                edge_candidates: list[Level0PatchCandidate],
                *,
                refinement_level: int = 0,
                early_stop: bool = False,
            ) -> tuple[
                list[dict[str, Any]],
                list[dict[str, Any]],
                str,
                str | None,
                set[int],
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray,
            ]:
                native_shift_scale_zyx = _native_shift_scale_zyx_for_edge_level(
                    registration_payload=refined,
                    tiles=tiles,
                    source=source,
                    target=target,
                    level=refinement_level,
                    image_reader_cache=image_reader_cache,
                )
                level_max_phase_shift_zyx = tuple(
                    float(value)
                    for value in np.asarray(max_phase_shift_zyx, dtype=np.float64) / native_shift_scale_zyx
                )
                level_max_final_residual_zyx = tuple(
                    float(value)
                    for value in np.asarray(max_final_residual_zyx, dtype=np.float64) / native_shift_scale_zyx
                )
                level_phase_highpass_sigma_zyx = (
                    None
                    if phase_highpass_sigma_zyx is None
                    else tuple(
                        float(value)
                        for value in np.asarray(phase_highpass_sigma_zyx, dtype=np.float64) / native_shift_scale_zyx
                    )
                )
                level_phase_margin = _phase_margin_zyx(level_max_phase_shift_zyx)

                def read_patch_pair(
                    candidate: Level0PatchCandidate,
                ) -> tuple[Level0PatchCandidate, tuple[int, int, int], np.ndarray, np.ndarray]:
                    sample_shape_zyx = _level_shape_zyx(candidate.shape_zyx, native_shift_scale_zyx)
                    center_level_px_zyx = np.asarray(candidate.center_px_zyx, dtype=np.float64) / native_shift_scale_zyx
                    patch_origin = tuple(
                        int(round(float(center_level_px_zyx[index]) - float(sample_shape_zyx[index]) / 2.0))
                        - int(level_phase_margin[index])
                        for index in range(3)
                    )
                    return (
                        candidate,
                        patch_origin,
                        _sample_candidate_patch(
                            registration_payload=refined,
                            tile_record=tiles[source],
                            candidate=candidate,
                            margin_zyx=level_phase_margin,
                            spacing_um_zyx=spacing_um_zyx,
                            channel=channel,
                            level=refinement_level,
                            native_shift_scale_zyx=native_shift_scale_zyx,
                            image_reader_cache=image_reader_cache,
                        ),
                        _sample_candidate_patch(
                            registration_payload=refined,
                            tile_record=tiles[target],
                            candidate=candidate,
                            margin_zyx=level_phase_margin,
                            spacing_um_zyx=spacing_um_zyx,
                            channel=channel,
                            level=refinement_level,
                            native_shift_scale_zyx=native_shift_scale_zyx,
                            image_reader_cache=image_reader_cache,
                        ),
                    )

                def summarize_measurements(
                    measurements: list[dict[str, Any]],
                ) -> tuple[list[dict[str, Any]], str, str | None, set[int], np.ndarray | None, np.ndarray | None]:
                    accepted = [patch for patch in measurements if patch["accepted"]]
                    status = "accepted"
                    reject_reason = None
                    inlier_indices: set[int] = set()
                    edge_shift_level = None
                    edge_shift_native = None
                    if len(accepted) < min_inliers:
                        status = "rejected"
                        reject_reason = "low_inlier_count"
                    else:
                        try:
                            inlier_mask, edge_shift = _inlier_shift_cluster(
                                accepted,
                                min_inliers=min_inliers,
                                max_final_residual_zyx=level_max_final_residual_zyx,
                            )
                        except ValueError as exc:
                            status = "rejected"
                            reject_reason = str(exc)
                            inlier_mask = np.zeros(len(accepted), dtype=bool)
                            edge_shift = None
                        inlier_patches = [
                            patch
                            for patch, is_inlier in zip(accepted, inlier_mask, strict=True)
                            if bool(is_inlier)
                        ]
                        inlier_indices = {int(patch["patch_index"]) for patch in inlier_patches}
                        if edge_shift is not None:
                            edge_shift_level = np.asarray(edge_shift, dtype=np.float64)
                            edge_shift_native = edge_shift_level * native_shift_scale_zyx
                        if status == "accepted" and not _has_spatial_diversity(inlier_patches):
                            status = "rejected"
                            reject_reason = "low_spatial_diversity"
                            edge_shift_level = None
                            edge_shift_native = None
                    return accepted, status, reject_reason, inlier_indices, edge_shift_level, edge_shift_native

                measurements = []
                batch_size = max(1, workers) if early_stop else max(1, len(edge_candidates))
                for start in range(0, len(edge_candidates), batch_size):
                    batch = edge_candidates[start : start + batch_size]
                    if workers == 1:
                        patch_pairs = [read_patch_pair(candidate) for candidate in batch]
                    else:
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                            patch_pairs = list(executor.map(read_patch_pair, batch))
                    measurements.extend(
                        _measure_level0_candidate(
                            fixed_patch=fixed_patch,
                            moving_patch=moving_patch,
                            candidate=candidate,
                            patch_origin_px_zyx=patch_origin,
                            max_phase_shift_zyx=level_max_phase_shift_zyx,
                            phase_highpass_sigma_zyx=level_phase_highpass_sigma_zyx,
                            phase_upsample_factor=phase_upsample_factor,
                        )
                        for candidate, patch_origin, fixed_patch, moving_patch in patch_pairs
                    )
                    accepted, status, reject_reason, inlier_indices, edge_shift_level, edge_shift_native = summarize_measurements(
                        measurements
                    )
                    if early_stop and status == "accepted":
                        break
                accepted, status, reject_reason, inlier_indices, edge_shift_level, edge_shift_native = summarize_measurements(
                    measurements
                )
                for patch in measurements:
                    patch["refinement_level"] = int(refinement_level)
                    patch["native_shift_scale_zyx"] = [float(value) for value in native_shift_scale_zyx]
                    patch["shift_native_px_zyx"] = [
                        float(value)
                        for value in np.asarray(patch["shift_px_zyx"], dtype=np.float64) * native_shift_scale_zyx
                    ]
                    patch["inlier"] = int(patch["patch_index"]) in inlier_indices
                return (
                    measurements,
                    accepted,
                    status,
                    reject_reason,
                    inlier_indices,
                    edge_shift_level,
                    edge_shift_native,
                    native_shift_scale_zyx,
                )

            patch_measurements, accepted_patches, edge_status, edge_reject_reason, inlier_indices, edge_shift_level, edge_shift, native_shift_scale_zyx = (
                measure_edge_candidates(candidates)
            )
            edge_refinement_level = 0
            retried_with_patch_count = None
            retry_candidate_info: dict[str, Any] | None = None
            fallback_refinement_attempts: list[dict[str, Any]] = []
            if candidate_mode == "scout" and (
                edge_reject_reason == "low_inlier_count"
                or (
                    isinstance(edge_reject_reason, str)
                    and "mutually compatible patch shifts found" in edge_reject_reason
                )
            ):
                if retry_patches_per_edge > patches_per_edge:
                    if progress is not None:
                        progress(f"Retrying MVS level-0 seam edge {edge_key} with {retry_patches_per_edge} patches")
                    retry_candidates, retry_candidate_info = _select_level0_retry_face_candidates(
                        registration_payload=refined,
                        edge=edge,
                        tiles=tiles,
                        spacing_um_zyx=spacing_um_zyx,
                        channel=channel,
                        patch_shape_zyx=patch_shape_zyx,
                        patches_per_edge=retry_patches_per_edge,
                        image_reader_cache=image_reader_cache,
                    )
                    patch_measurements, accepted_patches, edge_status, edge_reject_reason, inlier_indices, edge_shift_level, edge_shift, native_shift_scale_zyx = (
                        measure_edge_candidates(retry_candidates, early_stop=True)
                    )
                    retried_with_patch_count = len(patch_measurements)
                    retry_candidate_info["measured_candidate_count"] = len(patch_measurements)
                    retry_candidate_info["early_stopped"] = len(patch_measurements) < len(retry_candidates)

            if edge_status != "accepted" or edge_shift is None:
                for fallback_refinement_level in fallback_refinement_levels:
                    if progress is not None:
                        progress(f"Measuring MVS level-{fallback_refinement_level} fallback seam edge {edge_key}")
                    if candidate_mode == "scout":
                        fallback_candidates, fallback_candidate_info = _select_level0_retry_face_candidates(
                            registration_payload=refined,
                            edge=edge,
                            tiles=tiles,
                            spacing_um_zyx=spacing_um_zyx,
                            channel=channel,
                            patch_shape_zyx=patch_shape_zyx,
                            patches_per_edge=retry_patches_per_edge,
                            image_reader_cache=image_reader_cache,
                        )
                        fallback_patch_measurements, fallback_accepted_patches, fallback_status, fallback_reject_reason, fallback_inlier_indices, fallback_edge_shift_level, fallback_edge_shift, fallback_native_shift_scale_zyx = measure_edge_candidates(
                            fallback_candidates,
                            refinement_level=fallback_refinement_level,
                            early_stop=True,
                        )
                        fallback_candidate_info["measured_candidate_count"] = len(fallback_patch_measurements)
                        fallback_candidate_info["early_stopped"] = len(fallback_patch_measurements) < len(
                            fallback_candidates
                        )
                    else:
                        fallback_candidates = _select_level0_seam_span_candidate(
                            edge=edge,
                            spacing_um_zyx=spacing_um_zyx,
                            patch_shape_zyx=patch_shape_zyx,
                        )
                        fallback_candidate_info = None
                        fallback_patch_measurements, fallback_accepted_patches, fallback_status, fallback_reject_reason, fallback_inlier_indices, fallback_edge_shift_level, fallback_edge_shift, fallback_native_shift_scale_zyx = measure_edge_candidates(
                            fallback_candidates,
                            refinement_level=fallback_refinement_level,
                        )
                    fallback_refinement_attempts.append(
                        {
                            "refinement_level": int(fallback_refinement_level),
                            "status": fallback_status,
                            "reject_reason": fallback_reject_reason,
                            "accepted_patch_count": len(fallback_accepted_patches),
                            "measured_patch_count": len(fallback_patch_measurements),
                            "inlier_patch_count": len(fallback_inlier_indices),
                            "candidate_info": fallback_candidate_info,
                            "edge_shift_level_px_zyx": None
                            if fallback_edge_shift_level is None
                            else [float(value) for value in fallback_edge_shift_level],
                            "edge_shift_native_px_zyx": None
                            if fallback_edge_shift is None
                            else [float(value) for value in fallback_edge_shift],
                            "native_shift_scale_zyx": [float(value) for value in fallback_native_shift_scale_zyx],
                        }
                    )
                    if progress is not None:
                        shift_text = (
                            "none"
                            if fallback_edge_shift is None
                            else f"({fallback_edge_shift[0]:.2f},{fallback_edge_shift[1]:.2f},{fallback_edge_shift[2]:.2f})"
                        )
                        reason_text = "" if fallback_reject_reason is None else f" reason={fallback_reject_reason}"
                        progress(
                            f"Result MVS level-{fallback_refinement_level} fallback seam edge {edge_key} "
                            f"status={fallback_status} shift_native_zyx={shift_text} "
                            f"accepted_patches={len(fallback_accepted_patches)}/{len(fallback_patch_measurements)} "
                            f"inliers={len(fallback_inlier_indices)}{reason_text}"
                        )
                    if fallback_status == "accepted" and fallback_edge_shift is not None:
                        patch_measurements = fallback_patch_measurements
                        accepted_patches = fallback_accepted_patches
                        edge_status = fallback_status
                        edge_reject_reason = fallback_reject_reason
                        inlier_indices = fallback_inlier_indices
                        edge_shift_level = fallback_edge_shift_level
                        edge_shift = fallback_edge_shift
                        native_shift_scale_zyx = fallback_native_shift_scale_zyx
                        edge_refinement_level = int(fallback_refinement_level)
                        break

            edge_record = {
                "pair": [source, target],
                "source_tile": tile_names[source],
                "target_tile": tile_names[target],
                "quality": float(edge["quality"]),
                "patches": patch_measurements,
                "accepted_patch_count": len(accepted_patches),
                "inlier_patch_count": len(inlier_indices),
                "retried_with_patch_count": retried_with_patch_count,
                "retry_candidate_info": retry_candidate_info,
                "edge_shift_px_zyx": None if edge_shift is None else [float(value) for value in edge_shift],
                "edge_shift_level_px_zyx": None if edge_shift_level is None else [float(value) for value in edge_shift_level],
                "refinement_level": int(edge_refinement_level),
                "native_shift_scale_zyx": [float(value) for value in native_shift_scale_zyx],
                "fallback_refinement_attempts": fallback_refinement_attempts,
                "status": edge_status,
                "reject_reason": edge_reject_reason,
            }
            if progress is not None:
                shift_text = (
                    "none"
                    if edge_shift is None
                    else f"({edge_shift[0]:.2f},{edge_shift[1]:.2f},{edge_shift[2]:.2f})"
                )
                retry_text = "" if retried_with_patch_count is None else f" retried_patches={retried_with_patch_count}"
                reason_text = "" if edge_reject_reason is None else f" reason={edge_reject_reason}"
                progress(
                    f"Result MVS level-0 seam edge {edge_key} status={edge_status} "
                    f"shift_zyx={shift_text} accepted_patches={len(accepted_patches)}/{len(patch_measurements)} "
                    f"inliers={len(inlier_indices)}{retry_text}{reason_text}"
                )
            measured_edges.append(edge_record)
            if edge_status != "accepted" or edge_shift is None:
                rejected_edges.append({"pair": [source, target], "reason": edge_reject_reason})
                continue
            accepted_constraints.append(
                _edge_constraint_from_measurement(
                    edge=edge,
                    edge_shift_px_zyx=edge_shift,
                    weight=max(float(edge["score"]), 1e-3),
                    axis="mvs_level0" if edge_refinement_level == 0 else f"mvs_level{edge_refinement_level}",
                    source_label="mvs_level0_refinement"
                    if edge_refinement_level == 0
                    else f"mvs_level{edge_refinement_level}_fallback_refinement",
                )
            )
            accepted_level0_pairs.add(normalize_mvs_edge((source, target)))

        fallback_constraints = []
        if fallback_level2_weight_scale > 0.0:
            fallback_by_pair = {}
            for edge in level2_fallback_edges:
                pair = normalize_mvs_edge((edge["source"], edge["target"]))
                if pair not in accepted_level0_pairs:
                    fallback_by_pair[pair] = edge
            fallback_constraints = [
                _edge_constraint_from_level2_fallback(
                    edge=edge,
                    spacing_um_zyx=spacing_um_zyx,
                    weight_scale=fallback_level2_weight_scale,
                )
                for _pair, edge in sorted(fallback_by_pair.items())
            ]
        if fallback_constraints and progress is not None:
            progress(
                "Adding "
                f"{len(fallback_constraints)} low-weight level-2 fallback constraint(s) "
                "for non-level-0-accepted MVS seam(s)"
            )

        corrections_px, annotated_constraints, anchor_tile, solver_connectivity = _solve_level0_edge_corrections(
            n_tiles=len(tiles),
            constraints=[*accepted_constraints, *fallback_constraints],
            spacing_um_zyx=spacing_um_zyx,
            max_correction_zyx=max_correction_zyx,
            max_final_residual_zyx=max_final_residual_zyx,
            max_disconnected_island_size=max_disconnected_island_size,
        )
        _apply_registered_affine_corrections(
            registration_payload=refined,
            corrections_px=corrections_px,
            spacing_um_zyx=spacing_um_zyx,
        )
        optimization_summary = _optimization_summary(
            constraints=annotated_constraints,
            corrections_px=corrections_px,
            spacing_um_zyx=spacing_um_zyx,
            max_correction_zyx=max_correction_zyx,
            solver_connectivity=solver_connectivity,
            tile_names=tile_names,
        )
        if progress is not None:
            residual_stats = optimization_summary["residual_stats"] or {}
            p95 = residual_stats.get("p95_abs_px_zyx")
            p95_text = "none" if p95 is None else f"({p95[0]:.2f},{p95[1]:.2f},{p95[2]:.2f})"
            progress(
                "Level-0 optimization summary: "
                f"accepted_constraints={optimization_summary['accepted_constraint_count']}/"
                f"{optimization_summary['constraint_count']} "
                f"connected_tiles={optimization_summary['connectivity']['connected_tile_count']}/{len(tile_names)} "
                f"p95_abs_residual_zyx={p95_text} "
                f"clamp_hits_zyx={optimization_summary['correction_summary']['clamp_hit_count_zyx']}"
            )
        diagnostics = {
            "artifact_type": "lightsheet.mvs_level0_refinement.v1",
            "registration_input": str(registration_input.resolve()),
            "output_registration": str(output_registration),
            "settings": {
                "channel": int(channel),
                "patch_shape_zyx": [int(value) for value in patch_shape_zyx],
                "candidate_mode": candidate_mode,
                "patches_per_edge": int(patches_per_edge),
                "retry_patches_per_edge": int(retry_patches_per_edge),
                "min_inliers": int(min_inliers),
                "max_phase_shift_zyx": [float(value) for value in max_phase_shift_zyx],
                "patch_read_strategy": "direct_registered_source_patch",
                "phase_crop": "seed_shared_valid_support_expanded_by_max_phase_shift",
                "min_shifted_valid_support_coverage": MIN_SHIFTED_VALID_SUPPORT_COVERAGE,
                "phase_highpass_sigma_zyx": None
                if phase_highpass_sigma_zyx is None
                else [float(value) for value in phase_highpass_sigma_zyx],
                "phase_upsample_factor": int(phase_upsample_factor),
                "max_correction_zyx": [float(value) for value in max_correction_zyx],
                "max_final_residual_zyx": [float(value) for value in max_final_residual_zyx],
                "min_quality": float(min_quality),
                "used_edges_only": bool(used_edges_only),
                "max_edges": max_edges,
                "max_disconnected_island_size": int(max_disconnected_island_size),
                "fallback_refinement_levels": [int(level) for level in fallback_refinement_levels],
                "fallback_level2_weight_scale": float(fallback_level2_weight_scale),
                "workers": int(workers),
            },
            "registration_input_hash": registration_hash,
            "selected_edge_count": len(selected_edges),
            "measured_edge_count": len(measured_edges),
            "accepted_edge_count": sum(1 for edge in measured_edges if edge["status"] == "accepted"),
            "accepted_edge_count_by_refinement_level": {
                str(level): sum(
                    1
                    for edge in measured_edges
                    if edge["status"] == "accepted" and int(edge.get("refinement_level", 0)) == int(level)
                )
                for level in sorted({int(edge.get("refinement_level", 0)) for edge in measured_edges})
            },
            "fallback_refinement_rescued_edge_count": sum(
                1
                for edge in measured_edges
                if edge["status"] == "accepted" and int(edge.get("refinement_level", 0)) > 0
            ),
            "fallback_level2_constraint_count": len(fallback_constraints),
            "rejected_edges": rejected_edges,
            "measured_edges": measured_edges,
            "anchor_tile": tile_names[anchor_tile],
            "optimization_summary": optimization_summary,
            "solver_connectivity": {
                "connected_tile_count": int(solver_connectivity["connected_tile_count"]),
                "disconnected_tile_count": int(solver_connectivity["disconnected_tile_count"]),
                "component_sizes": [int(value) for value in solver_connectivity["component_sizes"]],
                "disconnected_island_sizes": [
                    int(value)
                    for value in solver_connectivity["disconnected_island_sizes"]
                ],
                "largest_disconnected_island_size": int(solver_connectivity["largest_disconnected_island_size"]),
                "anchor_component_tiles": [
                    tile_names[int(index)]
                    for index in solver_connectivity["anchor_component_tiles"]
                ],
                "disconnected_tiles": [
                    tile_names[int(index)]
                    for index in solver_connectivity["disconnected_tiles"]
                ],
            },
            "corrections": [
                {
                    "tile": tile,
                    "correction_px_zyx": [float(value) for value in corrections_px[index]],
                    "correction_um_zyx": [float(value) for value in corrections_px[index] * spacing_um_zyx],
                }
                for index, tile in enumerate(tile_names)
            ],
            "constraints": _constraint_diagnostics(annotated_constraints, tile_names),
        }
        refined.setdefault("metrics", {})["level0_refinement"] = diagnostics
        if render_contact_sheet:
            if progress is not None:
                progress("Rendering MVS level-0 local phase contact sheet")
            diagnostics["contact_sheet"] = render_mvs_level0_refinement_contact_sheet(
                diagnostics=diagnostics,
                sample_registration_payload=payload,
                output_dir=(
                    contact_sheet_output_dir.resolve()
                    if contact_sheet_output_dir is not None
                    else _default_contact_sheet_dir(output_registration)
                ),
                channel=channel,
                max_panels=contact_sheet_max_panels,
            )
        output_registration.parent.mkdir(parents=True, exist_ok=True)
        output_registration.write_text(json.dumps(_json_ready(refined), indent=2) + "\n")
        return refined, diagnostics
    finally:
        image_reader_cache.close()


def load_mvs_registration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "metrics" not in payload or "pairwise_registration" not in payload.get("metrics", {}):
        raise ValueError(f"{path} is not an MVS registration JSON with pairwise metrics")
    return payload


def recover_anchor_shifts_from_mvs_seams(
    *,
    direct_anchor_shift_um_by_tile: dict[str, np.ndarray],
    mvs_registration: dict[str, Any],
    spacing_um_zyx: np.ndarray,
    min_quality: float = DEFAULT_MVS_SEAM_QUALITY_THRESHOLD,
    used_edges_only: bool = True,
    unused_edge_weight_scale: float = DEFAULT_LEVEL0_FALLBACK_LEVEL2_WEIGHT_SCALE,
    max_residual_px_zyx: tuple[float, float, float] = (np.inf, np.inf, np.inf),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tile_names = [str(record["tile"]) for record in mvs_registration["tiles"]]
    direct_tiles = set(direct_anchor_shift_um_by_tile)
    unknown_tiles = [tile for tile in tile_names if tile not in direct_tiles]
    spacing = np.asarray(spacing_um_zyx, dtype=float)
    direct_px = {tile: shift_um / spacing for tile, shift_um in direct_anchor_shift_um_by_tile.items()}
    edge_constraints = mvs_seam_constraints(
        mvs_registration,
        tile_names=tile_names,
        spacing_um_zyx=spacing,
        min_quality=min_quality,
        used_edges_only=used_edges_only,
        unused_edge_weight_scale=unused_edge_weight_scale,
    )

    adjacency: dict[str, set[str]] = defaultdict(set)
    for constraint in edge_constraints:
        adjacency[constraint["fixed"]].add(constraint["moving"])
        adjacency[constraint["moving"]].add(constraint["fixed"])

    anchored_component_tiles = set(direct_tiles)
    queue = deque(direct_tiles)
    while queue:
        tile = queue.popleft()
        for neighbor in adjacency.get(tile, set()):
            if neighbor not in anchored_component_tiles:
                anchored_component_tiles.add(neighbor)
                queue.append(neighbor)

    solved_unknowns = [tile for tile in unknown_tiles if tile in anchored_component_tiles]
    solved_unknown_index = {tile: index for index, tile in enumerate(solved_unknowns)}
    active_constraints = [
        constraint
        for constraint in edge_constraints
        if constraint["fixed"] in anchored_component_tiles and constraint["moving"] in anchored_component_tiles
    ]

    def tile_shift(tile: str, values: np.ndarray) -> np.ndarray:
        if tile in direct_px:
            return direct_px[tile]
        index = solved_unknown_index[tile]
        return values.reshape(len(solved_unknowns), 3)[index]

    def residual_vector(flat: np.ndarray) -> np.ndarray:
        residuals = []
        for constraint in active_constraints:
            quality = np.sqrt(float(constraint["weight"]))
            residuals.append(
                quality
                * (
                    tile_shift(constraint["moving"], flat)
                    - tile_shift(constraint["fixed"], flat)
                    - constraint["target_correction_delta_px"]
                )
            )
        return np.concatenate(residuals) if residuals else np.zeros(0, dtype=float)

    if solved_unknowns:
        result = least_squares(
            residual_vector,
            np.zeros(len(solved_unknowns) * 3, dtype=float),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=500,
        )
        solved = result.x.reshape(len(solved_unknowns), 3)
    else:
        result = None
        solved = np.zeros((0, 3), dtype=float)

    recovered = {
        tile: solved[index] * spacing
        for tile, index in solved_unknown_index.items()
    }
    max_residual = np.asarray(max_residual_px_zyx, dtype=float)
    residual_records = []
    incident_inlier_count: dict[str, int] = {tile: 0 for tile in solved_unknowns}
    for constraint in active_constraints:
        fixed = constraint["fixed"]
        moving = constraint["moving"]
        residual = (
            tile_shift(moving, solved.reshape(-1))
            - tile_shift(fixed, solved.reshape(-1))
            - constraint["target_correction_delta_px"]
        )
        inlier = bool(np.all(np.abs(residual) <= max_residual))
        for tile in (fixed, moving):
            if tile in incident_inlier_count and inlier:
                incident_inlier_count[tile] += 1
        residual_records.append(
            {
                "fixed": fixed,
                "moving": moving,
                "pair": constraint["pair"],
                "quality": constraint["corr_after"],
                "target_delta_px_zyx": constraint["target_correction_delta_px"].tolist(),
                "residual_px_zyx": residual.tolist(),
                "residual_abs_within_bound_zyx": inlier,
            }
        )

    recovered = {
        tile: shift
        for tile, shift in recovered.items()
        if incident_inlier_count.get(tile, 0) > 0
    }
    diagnostics = {
        "mvs_pairwise_edge_count": len(edge_constraints),
        "active_edge_count": len(active_constraints),
        "direct_anchor_count": len(direct_tiles),
        "unknown_tile_count": len(unknown_tiles),
        "recovered_tile_count": len(recovered),
        "unrecovered_tiles": [tile for tile in unknown_tiles if tile not in recovered],
        "min_quality": min_quality,
        "used_edges_only": used_edges_only,
        "unused_edge_weight_scale": unused_edge_weight_scale,
        "optimizer": None
        if result is None
        else {
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
        },
        "residuals": residual_records,
        "incident_inlier_count_by_tile": incident_inlier_count,
    }
    return recovered, diagnostics
