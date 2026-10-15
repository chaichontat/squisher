"""Checked MVS integration must fail before unsafe output and restore patches."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from multiview_stitcher import fusion, misc_utils, weights
from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy


@pytest.mark.parametrize("missing", ["module", "zarr", "create"])
def test_resume_requires_expected_mvs_create_capability(monkeypatch, tmp_path: Path, missing):
    def unsupported_fuse():
        pass

    unsupported_fuse.__module__ = "unsupported_mvs_core"
    monkeypatch.setattr(fusion, "fuse", unsupported_fuse)
    if missing != "module":
        core = SimpleNamespace() if missing == "zarr" else SimpleNamespace(zarr=SimpleNamespace())
        monkeypatch.setitem(sys.modules, unsupported_fuse.__module__, core)
    with pytest.raises(RuntimeError, match="MVS.*zarr.create"):
        with legacy.resumable_mvs_zarr_create(scale0_path=tmp_path / "0", enabled=True):
            pytest.fail("Unsupported resume backend must fail before fusion")


def test_cleanup_failure_still_restores_mvs_functions(monkeypatch):
    def failed_cleanup():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(misc_utils, "clear_cupy_memory", failed_cleanup)
    original_weights_cleanup = getattr(weights, "clear_cupy_memory", None)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        with legacy.cupy_cleanup_context(False):
            assert misc_utils.clear_cupy_memory() is False
    assert misc_utils.clear_cupy_memory is failed_cleanup
    assert getattr(weights, "clear_cupy_memory", None) is original_weights_cleanup


def test_installed_mvs_capabilities_are_supported():
    from squisher_lightsheet.mvs_backend import MVSBackend

    backend = MVSBackend.load()
    assert backend.fuse is fusion.fuse


@pytest.mark.parametrize("capability", ["normalize_weights", "sim_sel_coords", "_fuse_block_zarr_backed"])
def test_adapter_rejects_missing_required_capability(monkeypatch, capability):
    from multiview_stitcher import spatial_image_utils
    from multiview_stitcher.fusion import _core
    from squisher_lightsheet.mvs_backend import MVSBackend

    owner = {
        "normalize_weights": weights,
        "sim_sel_coords": spatial_image_utils,
        "_fuse_block_zarr_backed": _core,
    }[capability]
    monkeypatch.delattr(owner, capability)
    with pytest.raises(RuntimeError, match=capability):
        MVSBackend.load()


def test_adapter_rejects_incompatible_private_block_signature(monkeypatch):
    from multiview_stitcher.fusion import _core
    from squisher_lightsheet.mvs_backend import MVSBackend

    monkeypatch.setattr(_core, "_fuse_block_zarr_backed", lambda block: block)
    with pytest.raises(RuntimeError, match="incompatible _fuse_block_zarr_backed signature"):
        MVSBackend.load()


@pytest.mark.parametrize("interruption", ["entry", "fusion"])
def test_production_patch_stack_unwinds_interruption(interruption):
    from contextlib import contextmanager
    from squisher_lightsheet.mvs_backend import MVSBackend

    events = []
    names = (
        "source_selection",
        "source_reads",
        "profiling",
        "cleanup",
        "normalization",
        "spatial_chunks",
        "single_scale",
        "resume",
    )

    @contextmanager
    def patch(name):
        if name == "single_scale" and interruption == "entry":
            raise RuntimeError("entry interrupted")
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    backend = MVSBackend.load()
    with pytest.raises(RuntimeError, match="interrupted"):
        with backend.patches(**{name: patch(name) for name in names}):
            raise RuntimeError("fusion interrupted")
    entered = names[:6] if interruption == "entry" else names
    assert events == [("enter", name) for name in entered] + [("exit", name) for name in reversed(entered)]


def test_resume_opens_existing_scale0_without_recreation(tmp_path):
    import numpy as np
    import zarr
    from squisher_lightsheet.mvs_backend import fusion_zarr_module

    path = tmp_path / "0"
    existing = zarr.create_array(str(path), shape=(2, 3, 4), chunks=(1, 3, 4), dtype="uint16")
    existing[:] = 27
    backend_zarr = fusion_zarr_module()
    original_create = backend_zarr.create
    with pytest.raises(RuntimeError, match="interrupted"):
        with legacy.resumable_mvs_zarr_create(scale0_path=path, enabled=True):
            resumed = backend_zarr.create(shape=(2, 3, 4), chunks=(1, 3, 4), dtype="uint16", store=str(path))
            np.testing.assert_array_equal(resumed[:], np.full((2, 3, 4), 27, dtype="uint16"))
            raise RuntimeError("interrupted")
    assert backend_zarr.create is original_create
    np.testing.assert_array_equal(zarr.open_array(str(path), mode="r")[:], existing[:])


def test_source_summary_failure_still_restores_mvs_reads(monkeypatch):
    from multiview_stitcher import spatial_image_utils

    names = ("serialize_zarr_backed_sim", "deserialize_zarr_backed_sim", "sim_sel_coords")
    originals = {name: getattr(spatial_image_utils, name) for name in names}

    def failed_log(_message):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(legacy, "log", failed_log)
    with pytest.raises(RuntimeError, match="summary failed"):
        with legacy.basic_corrected_zarr_reads(None, cache_key_attr="fusion_source_cache_key"):
            assert legacy._ACTIVE_DIRECT_SOURCE_READ_SESSION is not None
    assert legacy._ACTIVE_DIRECT_SOURCE_READ_SESSION is None
    for name in names:
        assert getattr(spatial_image_utils, name) is originals[name]


def test_real_production_patches_restore_after_interrupted_fusion(monkeypatch, tmp_path):
    from multiview_stitcher import ngff_utils, spatial_image_utils
    from squisher_lightsheet.mvs_backend import MVSBackend, fusion_zarr_module

    monkeypatch.setattr(misc_utils, "clear_cupy_memory", lambda: None)
    targets = (
        (spatial_image_utils, "sim_sel_coords"),
        (spatial_image_utils, "serialize_zarr_backed_sim"),
        (spatial_image_utils, "deserialize_zarr_backed_sim"),
        (weights, "normalize_weights"),
        (misc_utils, "clear_cupy_memory"),
        (ngff_utils, "write_sim_to_ome_zarr"),
        (ngff_utils, "write_and_return_downsampled_sim"),
        (fusion_zarr_module(), "create"),
    )
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    with pytest.raises(RuntimeError, match="fusion interrupted"):
        with MVSBackend.load().patches(
            source_selection=legacy.zarr_safe_fusion_selection(),
            source_reads=legacy.basic_corrected_zarr_reads(None, cache_key_attr="fusion_source_cache_key"),
            profiling=legacy.profile_zarr_slice_materialization(enabled=True),
            cleanup=legacy.cupy_cleanup_context(False),
            normalization=legacy.inplace_mvs_normalize_weights_context(),
            spatial_chunks=legacy.spatial_chunks_for_package_pyramid(),
            single_scale=legacy.single_scale_fusion_output(),
            resume=legacy.resumable_mvs_zarr_create(scale0_path=tmp_path / "0", enabled=True),
        ):
            raise RuntimeError("fusion interrupted")
    assert legacy._ACTIVE_DIRECT_SOURCE_READ_SESSION is None
    for owner, name, original in originals:
        assert getattr(owner, name) is original


def test_native_registration_uses_packaged_adapter(monkeypatch):
    import numpy as np
    from squisher_lightsheet import native_reg3dgpu

    calls = []
    original_path = list(sys.path)

    def register(fixed, moving, **kwargs):
        calls.append((fixed, moving, kwargs))
        return SimpleNamespace(
            return_code=0,
            matrix_zyx=np.eye(3),
            offset_zyx=np.array([1, 2, 3]),
            records=np.array([0, 0, 0, 0.9]),
        )

    monkeypatch.setattr(native_reg3dgpu, "_run_reg_3dgpu", register)
    data = SimpleNamespace(data=np.ones((2, 3, 4), dtype=np.float32))
    result = legacy.native_rigid_pairwise_registration.__wrapped__(data, data)
    assert len(calls) == 1
    assert calls[0][2]["lib_dir"] == legacy.NATIVE_REG3D_LIB_DIR
    np.testing.assert_array_equal(result["affine_matrix"][:3, 3], [1, 2, 3])
    assert result["quality"] == 0.9
    assert sys.path == original_path
