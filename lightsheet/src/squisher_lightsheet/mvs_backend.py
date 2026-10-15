"""Checked capabilities and patch lifetime for the isolated MVS fusion process.

MVS internals remain version-sensitive. Check every production patch target before
installing any patches; retain the numerical implementations in the fusion engine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
import inspect
import sys
from types import ModuleType


def require_callable(owner: object, name: str, capability: str) -> Callable[..., object]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise RuntimeError(f"Unsupported MVS backend: required capability {capability} is unavailable")
    return value


def fusion_zarr_module() -> ModuleType:
    from multiview_stitcher import fusion

    fuse = require_callable(fusion, "fuse", "fusion.fuse")
    core = sys.modules.get(getattr(fuse, "__module__", ""))
    zarr_module = getattr(core, "zarr", None)
    require_callable(zarr_module, "create", "fusion core zarr.create")
    return zarr_module


@dataclass(frozen=True)
class MVSBackend:
    fuse: Callable[..., object]
    fuse_block: Callable[..., object]
    weighted_average_fusion: Callable[..., object]

    @classmethod
    def load(cls) -> MVSBackend:
        from multiview_stitcher import fusion, misc_utils, ngff_utils, spatial_image_utils, weights

        fuse = require_callable(fusion, "fuse", "fusion.fuse")
        core = sys.modules.get(getattr(fuse, "__module__", ""))
        require_callable(getattr(core, "zarr", None), "create", "fusion core zarr.create")
        block = require_callable(core, "_fuse_block_zarr_backed", "fusion core _fuse_block_zarr_backed")
        # Verify the production request shapes before any patch or output write.
        fuse_arguments = (
            "images",
            "transform_key",
            "fusion_func",
            "fusion_func_kwargs",
            "weights_func",
            "weights_func_kwargs",
            "output_stack_properties",
            "output_chunksize",
            "blending_widths",
            "output_zarr_url",
            "zarr_options",
            "batch_options",
            "backend",
        )
        try:
            inspect.signature(fuse).bind(**dict.fromkeys(fuse_arguments))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Unsupported MVS backend: incompatible fusion.fuse signature") from exc
        block_arguments = (
            "output_dtype",
            "sim_coord_dict",
            "sdims",
            "fusion_func",
            "fusion_func_kwargs",
            "weights_func",
            "weights_func_kwargs",
            "overlap_in_pixels",
            "trim_overlap",
            "interpolation_order",
            "blending_widths",
            "shrink_distance",
            "backend",
        )
        try:
            inspect.signature(block).bind(object(), **dict.fromkeys(block_arguments))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Unsupported MVS backend: incompatible _fuse_block_zarr_backed signature"
            ) from exc
        for owner, names in (
            (
                spatial_image_utils,
                ("sim_sel_coords", "serialize_zarr_backed_sim", "deserialize_zarr_backed_sim"),
            ),
            (ngff_utils, ("write_sim_to_ome_zarr", "write_and_return_downsampled_sim")),
            (misc_utils, ("clear_cupy_memory",)),
            (weights, ("normalize_weights",)),
        ):
            for name in names:
                require_callable(owner, name, f"{owner.__name__}.{name}")
        weighted_average = require_callable(
            core, "weighted_average_fusion", "fusion core weighted_average_fusion"
        )
        return cls(fuse, block, weighted_average)

    @contextmanager
    def patches(
        self,
        *,
        source_selection: AbstractContextManager[object],
        source_reads: AbstractContextManager[object],
        profiling: AbstractContextManager[object],
        cleanup: AbstractContextManager[object],
        normalization: AbstractContextManager[object],
        spatial_chunks: AbstractContextManager[object],
        single_scale: AbstractContextManager[object],
        resume: AbstractContextManager[object],
    ) -> Iterator[None]:
        """Own patch ordering and unwind partial entry or interrupted fusion."""
        with ExitStack() as stack:
            for context in (
                source_selection,
                source_reads,
                profiling,
                cleanup,
                normalization,
                spatial_chunks,
                single_scale,
                resume,
            ):
                stack.enter_context(context)
            yield
