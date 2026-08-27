from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import threading
from typing import Any

from zarr.abc.store import Store


REOPENABLE_TIFF_BACKEND_CACHE_SIZE = 32
_REOPENABLE_TIFF_BACKEND_CACHE: OrderedDict[int, ReopenableTiffStore] = OrderedDict()


class ReopenableTiffStore(Store):
    """Expose a TIFF as a lazy Zarr store that workers can serialize."""

    def __init__(self, path: Path | str, level: int = 0) -> None:
        super().__init__(read_only=True)
        self.path = Path(path)
        self.level = level
        self._backend = None
        self._backend_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "level": self.level}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["path"], state["level"])

    def _backend_store(self):
        backend = self._backend
        if backend is None:
            with self._backend_lock:
                backend = self._backend
                if backend is None:
                    import tifffile

                    backend = tifffile.imread(self.path, aszarr=True, level=self.level)
                    self._backend = backend
        touch_reopenable_tiff_backend(self)
        return backend

    def with_read_only(self, read_only: bool = False):
        if not read_only:
            raise ValueError("TIFF sources are read-only")
        return type(self)(self.path, self.level)

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, ReopenableTiffStore) and self.path == value.path and self.level == value.level
        )

    async def get(self, key, prototype, byte_range=None):
        return await self._backend_store().get(key, prototype, byte_range)

    async def get_partial_values(self, prototype, key_ranges):
        return await self._backend_store().get_partial_values(prototype, key_ranges)

    async def exists(self, key: str) -> bool:
        return await self._backend_store().exists(key)

    @property
    def supports_writes(self) -> bool:
        return False

    async def set(self, key, value) -> None:
        raise ValueError("TIFF sources are read-only")

    @property
    def supports_deletes(self) -> bool:
        return False

    async def delete(self, key: str) -> None:
        raise ValueError("TIFF sources are read-only")

    @property
    def supports_listing(self) -> bool:
        return self._backend_store().supports_listing

    def list(self):
        return self._backend_store().list()

    def list_prefix(self, prefix: str):
        return self._backend_store().list_prefix(prefix)

    def list_dir(self, prefix: str):
        return self._backend_store().list_dir(prefix)

    def _release_backend_only(self) -> None:
        with self._backend_lock:
            backend = self._backend
            self._backend = None
            if backend is not None:
                backend.close()

    def release_backend(self) -> None:
        """Close the current TIFF handle while leaving this lazy store reusable."""
        _REOPENABLE_TIFF_BACKEND_CACHE.pop(id(self), None)
        self._release_backend_only()

    def close(self) -> None:
        self.release_backend()
        super().close()


def touch_reopenable_tiff_backend(store: ReopenableTiffStore) -> None:
    """Keep recently used TIFF handles open within a serial worker process."""
    _REOPENABLE_TIFF_BACKEND_CACHE.pop(id(store), None)
    _REOPENABLE_TIFF_BACKEND_CACHE[id(store)] = store
    if len(_REOPENABLE_TIFF_BACKEND_CACHE) <= REOPENABLE_TIFF_BACKEND_CACHE_SIZE:
        return
    _, evicted = _REOPENABLE_TIFF_BACKEND_CACHE.popitem(last=False)
    evicted._release_backend_only()


def clear_reopenable_tiff_backend_cache() -> None:
    """Close all worker-local cached TIFF handles."""
    stores = list(_REOPENABLE_TIFF_BACKEND_CACHE.values())
    _REOPENABLE_TIFF_BACKEND_CACHE.clear()
    for store in stores:
        store._release_backend_only()


class TiffInputHandler:
    """Open same-schema TIFFs lazily, reading one header per pyramid level."""

    def __init__(self) -> None:
        self._schema_by_level: dict[int, dict[str, Any]] = {}

    def open(self, path: Path | str, *, level: int = 0):
        from zarr.core.array import Array
        from zarr.storage import StorePath
        import zarr

        store = ReopenableTiffStore(path, level)
        schema = self._schema_by_level.get(level)
        if schema is not None:
            return Array.from_dict(StorePath(store), schema), store

        array = zarr.open(store, mode="r")
        self._schema_by_level[level] = array.metadata.to_dict()
        return array, store
