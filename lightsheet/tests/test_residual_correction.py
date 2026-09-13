import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr
from multiview_stitcher import spatial_image_utils as si_utils

from squisher_lightsheet._legacy import stitch_20x_tl_multiview as legacy
from squisher_lightsheet.residual_correction import (
    ResidualCorrection,
    fit_residual_correction,
    load_residual_corrections,
)
from squisher_lightsheet.fusion import fuse_tiles


def write_source(path, data, axes, origin, *, pyramid=False):
    group = zarr.open_group(path, mode="w-")
    group.create_array("0", data=data, chunks=data.shape, dimension_names=list(axes))
    datasets = [
        {
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0] * len(axes)},
                {"type": "translation", "translation": origin},
            ],
        }
    ]
    if pyramid:
        level1 = data[..., ::2, ::2]
        group.create_array("1", data=level1, chunks=level1.shape, dimension_names=list(axes))
        scale = [1.0] * len(axes)
        scale[-2:] = [2.0, 2.0]
        datasets.append(
            {
                "path": "1",
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": origin},
                ],
            }
        )
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": a, "type": "channel" if a == "c" else "space"} for a in axes],
                "datasets": datasets,
            }
        ],
    }


@pytest.fixture
def fitted_correction(tmp_path):
    yy, xx = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    scene = 1000 + 200 * np.sin(yy / 8) + 100 * np.cos(xx / 6)
    scene += 700 * np.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / 2000)
    gains = [0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.85, 1.15, 1.0]
    zz = np.linspace(0, 1, 8)[:, None, None]
    camera_y = np.linspace(0, 1, 64)[None, :, None]
    depth_field = np.exp(0.3 * np.cos(np.pi * zz) * np.cos(np.pi * camera_y))
    records, sources = [], []
    for i, gain in enumerate(gains):
        y, x = (i // 3) * 32, (i % 3) * 32
        data = (scene[y : y + 64, x : x + 64][None, None] * depth_field[None] * gain).astype(np.float32)
        path = tmp_path / f"tile-{i}.ome.zarr"
        write_source(path, data, "czyx", [0, 0, y, x], pyramid=True)
        sources.append(path)
        records.append(
            {
                "path": str(path),
                "tile": path.name,
                "axes": "CZYX",
                "shape": list(data.shape),
                "translation_um": {"z": 0, "y": y, "x": x},
                "scale_um": dict(z=1.0, y=1.0, x=1.0),
            }
        )
    fixed = tmp_path / "fixed.ome.zarr"
    write_source(fixed, np.zeros((8, 128, 128), dtype=np.uint16), "zyx", [0, 0, 0])
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"tiles": records}))
    output = tmp_path / "residual"
    manifest = fit_residual_correction(
        registration=registration,
        fixed_fused=fixed,
        output_dir=output,
        source_level=1,
        stride=1,
        z_percentiles=(25, 50, 75),
    )
    assert manifest == output / "manifest.json"
    return output / "correction.json", sources, np.asarray(gains)


def test_joint_gain_fit_improves_unseen_overlaps_and_refits_every_source(fitted_correction):
    path, sources, acquisition_gains = fitted_correction
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 3
    assert all(np.asarray(row["coefficient"]).shape == (2, 5) for row in payload["sources"])
    assert (path.parent / "qc/source_fields.csv").is_file()
    evaluation = payload["fit"]["evaluation"]
    assert evaluation["source_fields_applied"] is True
    assert evaluation["heldout_with_tile_gain"]["median"] < evaluation["heldout_before"]["median"] * 0.6
    assert all(row["training_pairs"] > 0 for row in payload["sources"])
    fields = load_residual_corrections(
        path,
        sources=sources,
        shapes_zyx=[(8, 64, 64)] * 9,
        channel=0,
    )
    reversed_fields = load_residual_corrections(
        path,
        sources=list(reversed(sources)),
        shapes_zyx=[(8, 64, 64)] * 9,
        channel=0,
    )
    assert (
        reversed_fields[str(sources[0].resolve())].fingerprint
        == fields[str(sources[0].resolve())].fingerprint
    )
    blocks = [
        fields[str(source.resolve())].block(
            z_slice=slice(0, 8),
            y_slice=slice(0, 64),
            x_slice=slice(0, 64),
        )
        for source in sources
    ]
    expected_first = ResidualCorrection(
        coefficient=np.asarray(payload["coefficient"]) + np.asarray(payload["sources"][0]["coefficient"]),
        shape_zyx=(8, 64, 64),
        multiplier=payload["global_scale"] * payload["sources"][0]["gain"],
        fingerprint=fields[str(sources[0].resolve())].fingerprint,
    ).block(z_slice=slice(0, 8), y_slice=slice(0, 64), x_slice=slice(0, 64))
    np.testing.assert_allclose(blocks[0], expected_first)
    corrected_gains = np.array([field.mean() for field in blocks]) * acquisition_gains
    assert corrected_gains.max() / corrected_gains.min() < 1.2
    assert all(np.isfinite(field).all() and field.min() > 0 and field.max() <= 1 for field in blocks)
    assert not np.allclose(blocks[0][0], blocks[0][-1])


def test_correction_rejects_missing_sources_and_wrong_channel(fitted_correction):
    path, sources, _ = fitted_correction
    with pytest.raises(ValueError, match="exact fusion source set"):
        load_residual_corrections(
            path,
            sources=sources[:-1],
            shapes_zyx=[(8, 64, 64)] * 8,
            channel=0,
        )
    with pytest.raises(ValueError, match="channel or coordinate"):
        load_residual_corrections(
            path,
            sources=sources,
            shapes_zyx=[(8, 64, 64)] * 9,
            channel=1,
        )


def test_correction_rejects_mismatched_source_field_shape(fitted_correction):
    path, sources, _ = fitted_correction
    payload = json.loads(path.read_text())
    payload["sources"][0]["coefficient"] = payload["sources"][0]["coefficient"][:1]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="coefficient shape"):
        load_residual_corrections(
            path,
            sources=sources,
            shapes_zyx=[(8, 64, 64)] * 9,
            channel=0,
        )


def test_correction_rejects_changed_source_metadata(fitted_correction):
    path, sources, _ = fitted_correction
    zarr.open_group(sources[0], mode="r+").attrs["changed"] = True
    with pytest.raises(ValueError, match="source metadata changed"):
        load_residual_corrections(
            path,
            sources=sources,
            shapes_zyx=[(8, 64, 64)] * 9,
            channel=0,
        )


def test_fusion_evaluates_only_requested_original_coordinate_block():
    coefficient = np.zeros((2, 5))
    coefficient[0, 0] = 0.2
    coefficient[1, 2] = -0.4
    correction = ResidualCorrection(
        coefficient=coefficient,
        shape_zyx=(9, 11, 13),
        multiplier=0.7,
        fingerprint="model-identity",
    )

    actual = legacy._residual_correction_for_region(
        correction,
        z0=3,
        z_size=4,
        y0=2,
        x0=5,
        y_size=6,
        x_size=7,
        message_prefix="Residual correction",
    )

    expected = correction.block(
        z_slice=slice(3, 7),
        y_slice=slice(2, 8),
        x_slice=slice(5, 12),
    )
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (4, 6, 7)
    assert legacy.basic_correction_fingerprint(correction) == "model-identity"
    with pytest.raises(ValueError, match="does not match zarr slice"):
        legacy._residual_correction_for_region(
            correction,
            z0=8,
            z_size=2,
            y0=0,
            x0=0,
            y_size=11,
            x_size=13,
            message_prefix="Residual correction",
        )


def test_fusion_read_hook_applies_depth_aware_block(monkeypatch):
    coefficient = np.zeros((2, 5))
    coefficient[0, 0] = 0.2
    coefficient[1, 2] = -0.4
    correction = ResidualCorrection(
        coefficient=coefficient,
        shape_zyx=(9, 11, 13),
        multiplier=0.7,
        fingerprint="model-identity",
    )
    sim = xr.DataArray(np.ones((2, 3, 4), dtype=np.float32), dims=("z", "y", "x"))
    monkeypatch.setattr(si_utils, "deserialize_zarr_backed_sim", lambda *args, **kwargs: sim)
    fake_cupy = SimpleNamespace(
        asarray=np.asarray,
        asnumpy=np.asarray,
        float32=np.float32,
        ndarray=np.ndarray,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    info = {
        "origin": {"z": 0.0, "y": 0.0, "x": 0.0},
        "spacing": {"z": 1.0, "y": 1.0, "x": 1.0},
        "source": "tile",
    }
    overlap = {
        "origin": {"z": 2.0, "y": 3.0, "x": 4.0},
        "shape": {"z": 2, "y": 3, "x": 4},
    }

    with legacy.basic_corrected_zarr_reads(
        {"tile": correction},
        dataset_info_key="source",
        dataset_attr_keys=("source",),
        error_prefix="Residual correction",
    ):
        actual = si_utils.deserialize_zarr_backed_sim(
            info,
            reconstruct_slice=True,
            overlap_bb=overlap,
        )

    expected = correction.block(
        z_slice=slice(2, 4),
        y_slice=slice(3, 6),
        x_slice=slice(4, 8),
    )
    np.testing.assert_allclose(actual.data, expected)


def test_fusion_source_cache_reuses_canonical_raw_chunks_across_requests(monkeypatch, tmp_path):
    source = zarr.open_array(
        tmp_path / "source.zarr",
        mode="w",
        shape=(1, 6, 6, 6),
        chunks=(1, 2, 2, 2),
        dtype="u2",
        dimension_names=("c", "z", "y", "x"),
    )
    data = np.arange(6**3, dtype=np.uint16).reshape(1, 6, 6, 6)
    source[:] = data
    sim = si_utils.get_sim_from_array(
        source,
        dims=("c", "z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
    ).isel(c=0, drop=True)
    sim.attrs["fusion_source_cache_key"] = "tile-a::ch0"

    original_deserialize = si_utils.deserialize_zarr_backed_sim
    calls = []

    def counting_deserialize(*args, **kwargs):
        if kwargs.get("reconstruct_slice"):
            calls.append(kwargs["overlap_bb"])
        return original_deserialize(*args, **kwargs)

    fake_cupy = SimpleNamespace(
        asarray=np.asarray,
        float32=np.float32,
        ndarray=np.ndarray,
        rint=np.rint,
        clip=np.clip,
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0)),
    )
    monkeypatch.setattr(si_utils, "deserialize_zarr_backed_sim", counting_deserialize)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    correction = np.full((6, 6), 2.0, dtype=np.float32)

    with legacy.basic_corrected_zarr_reads(
        correction,
        cache_key_attr="fusion_source_cache_key",
        source_cache_max_bytes=1024 * 1024,
    ):
        info = si_utils.serialize_zarr_backed_sim(sim)
        first_bb = {
            "origin": {"z": 0.0, "y": 0.0, "x": 0.0},
            "shape": {"z": 3, "y": 3, "x": 3},
        }
        second_bb = {
            "origin": {"z": 2.0, "y": 2.0, "x": 2.0},
            "shape": {"z": 3, "y": 3, "x": 3},
        }
        first = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=first_bb)
        second = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=second_bb)
        repeated = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=first_bb)

    assert len(calls) == 15
    np.testing.assert_array_equal(first.data, data[0, :3, :3, :3] * 2)
    np.testing.assert_array_equal(second.data, data[0, 2:5, 2:5, 2:5] * 2)
    np.testing.assert_array_equal(repeated.data, first.data)


def test_fusion_source_cache_also_applies_without_correction(monkeypatch, tmp_path):
    source = zarr.open_array(
        tmp_path / "source.zarr",
        mode="w",
        shape=(4, 4, 4),
        chunks=(2, 2, 2),
        dtype="u2",
        dimension_names=("z", "y", "x"),
    )
    data = np.arange(4**3, dtype=np.uint16).reshape(4, 4, 4)
    source[:] = data
    sim = si_utils.get_sim_from_array(
        source,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
    )
    sim.attrs["fusion_source_cache_key"] = "tile-a::ch0"
    original_deserialize = si_utils.deserialize_zarr_backed_sim
    calls = []

    def counting_deserialize(*args, **kwargs):
        if kwargs.get("reconstruct_slice"):
            calls.append(kwargs["overlap_bb"])
        return original_deserialize(*args, **kwargs)

    monkeypatch.setattr(si_utils, "deserialize_zarr_backed_sim", counting_deserialize)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    request = {
        "origin": {"z": 0.0, "y": 0.0, "x": 0.0},
        "shape": {"z": 3, "y": 3, "x": 3},
    }
    with legacy.basic_corrected_zarr_reads(
        None,
        cache_key_attr="fusion_source_cache_key",
        source_cache_max_bytes=1024,
    ):
        info = si_utils.serialize_zarr_backed_sim(sim)
        first = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=request)
        repeated = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=request)

    assert len(calls) == 8
    np.testing.assert_array_equal(first.data, data[:3, :3, :3])
    np.testing.assert_array_equal(repeated.data, first.data)


def test_fusion_source_cache_reads_outer_shards_not_inner_chunks(monkeypatch, tmp_path):
    from zarr.codecs import ShardingCodec

    source = zarr.open_array(
        tmp_path / "source.zarr",
        mode="w",
        shape=(1, 8, 8, 8),
        chunks=(1, 4, 4, 4),
        codecs=[ShardingCodec(chunk_shape=(1, 1, 2, 2))],
        dtype="u2",
        dimension_names=("c", "z", "y", "x"),
    )
    data = np.arange(8**3, dtype=np.uint16).reshape(1, 8, 8, 8)
    source[:] = data
    assert source.chunks == (1, 1, 2, 2)
    assert source.shards == (1, 4, 4, 4)
    sim = si_utils.get_sim_from_array(
        source,
        dims=("c", "z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
    ).isel(c=0, drop=True)
    sim.attrs["fusion_source_cache_key"] = "tile-a::ch0"
    original_deserialize = si_utils.deserialize_zarr_backed_sim
    calls = []

    def counting_deserialize(*args, **kwargs):
        if kwargs.get("reconstruct_slice"):
            calls.append(kwargs["overlap_bb"])
        return original_deserialize(*args, **kwargs)

    monkeypatch.setattr(si_utils, "deserialize_zarr_backed_sim", counting_deserialize)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    request = {
        "origin": {"z": 0.0, "y": 0.0, "x": 0.0},
        "shape": {"z": 3, "y": 3, "x": 3},
    }
    with legacy.basic_corrected_zarr_reads(
        None,
        cache_key_attr="fusion_source_cache_key",
        source_cache_max_bytes=1024,
    ):
        info = si_utils.serialize_zarr_backed_sim(sim)
        actual = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=request)

    assert len(calls) == 1
    assert calls[0]["shape"] == {"z": 4, "y": 4, "x": 4}
    np.testing.assert_array_equal(actual.data, data[0, :3, :3, :3])


def test_fusion_source_cache_uses_exact_read_when_chunk_exceeds_budget(monkeypatch, tmp_path):
    source = zarr.open_array(
        tmp_path / "source.zarr",
        mode="w",
        shape=(8, 8, 8),
        chunks=(8, 8, 8),
        dtype="u2",
        dimension_names=("z", "y", "x"),
    )
    data = np.arange(8**3, dtype=np.uint16).reshape(8, 8, 8)
    source[:] = data
    sim = si_utils.get_sim_from_array(
        source,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
    )
    sim.attrs["fusion_source_cache_key"] = "tile-a::ch0"
    original_deserialize = si_utils.deserialize_zarr_backed_sim
    requests = []

    def counting_deserialize(*args, **kwargs):
        if kwargs.get("reconstruct_slice"):
            requests.append(kwargs["overlap_bb"])
        return original_deserialize(*args, **kwargs)

    monkeypatch.setattr(si_utils, "deserialize_zarr_backed_sim", counting_deserialize)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    request = {
        "origin": {"z": 2.0, "y": 3.0, "x": 4.0},
        "shape": {"z": 2, "y": 2, "x": 2},
    }
    with legacy.basic_corrected_zarr_reads(
        None,
        cache_key_attr="fusion_source_cache_key",
        source_cache_max_bytes=64,
    ):
        info = si_utils.serialize_zarr_backed_sim(sim)
        actual = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=request)

    assert requests == [request]
    np.testing.assert_array_equal(actual.data, data[2:4, 3:5, 4:6])


@pytest.mark.parametrize("edge_dim", ["z", "y", "x"])
def test_fusion_source_cache_preserves_reader_behavior_for_upper_edge_halo(
    monkeypatch, tmp_path, edge_dim
):
    source = zarr.open_array(
        tmp_path / "source.zarr",
        mode="w",
        shape=(4, 4, 4),
        chunks=(2, 2, 2),
        dtype="u2",
        dimension_names=("z", "y", "x"),
    )
    data = np.arange(4**3, dtype=np.uint16).reshape(4, 4, 4)
    source[:] = data
    sim = si_utils.get_sim_from_array(
        source,
        dims=("z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
    )
    sim.attrs["fusion_source_cache_key"] = "tile-a::ch0"
    original_deserialize = si_utils.deserialize_zarr_backed_sim
    requests = []

    def counting_deserialize(*args, **kwargs):
        if kwargs.get("reconstruct_slice"):
            requests.append(kwargs["overlap_bb"])
        return original_deserialize(*args, **kwargs)

    monkeypatch.setattr(si_utils, "deserialize_zarr_backed_sim", counting_deserialize)
    fake_cupy = SimpleNamespace(
        asarray=np.asarray,
        float32=np.float32,
        ndarray=np.ndarray,
        rint=np.rint,
        clip=np.clip,
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0)),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    origin = {"z": 0.0, "y": 0.0, "x": 0.0}
    origin[edge_dim] = 3.0
    request = {
        "origin": origin,
        "shape": {"z": 2, "y": 2, "x": 2},
    }
    with legacy.basic_corrected_zarr_reads(
        np.full((4, 4), 2.0, dtype=np.float32),
        cache_key_attr="fusion_source_cache_key",
        source_cache_max_bytes=1024,
    ):
        info = si_utils.serialize_zarr_backed_sim(sim)
        expected = original_deserialize(info, reconstruct_slice=True, overlap_bb=request)
        actual = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=request)
        repeated = si_utils.deserialize_zarr_backed_sim(info, reconstruct_slice=True, overlap_bb=request)

    assert len(requests) == 1
    assert requests[0] != request
    assert actual.dims == expected.dims
    assert dict(actual.sizes) == dict(expected.sizes)
    for dim in actual.dims:
        np.testing.assert_array_equal(actual.coords[dim], expected.coords[dim])
    np.testing.assert_array_equal(actual.data, expected.data * 2)
    np.testing.assert_array_equal(repeated.data, actual.data)


def test_source_chunk_cache_deduplicates_concurrent_loads():
    cache = legacy.SourceChunkCache(max_bytes=1024)
    loader_started = threading.Event()
    release_loader = threading.Event()
    calls = 0
    results = []
    errors = []

    def loader():
        nonlocal calls
        calls += 1
        loader_started.set()
        assert release_loader.wait(timeout=5)
        return {"data": np.arange(8, dtype=np.uint16)}

    def read():
        try:
            results.append(cache.get_or_load("same", expected_bytes=16, loader=loader))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=read)
    second = threading.Thread(target=read)
    first.start()
    assert loader_started.wait(timeout=5)
    second.start()
    release_loader.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert calls == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_fusion_passes_residual_correction_to_owner_and_rejects_basic(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "squisher_lightsheet.fusion.run_legacy_script", lambda name, args, **kwargs: calls.append(args)
    )
    kwargs = dict(
        input_dir=tmp_path,
        position_input=tmp_path / "positions.json",
        registration_input=tmp_path / "registration.json",
        output=tmp_path / "out",
        residual_correction=tmp_path / "correction.json",
        dry_run=True,
    )
    fuse_tiles(**kwargs)
    assert calls[0][calls[0].index("--residual-correction") + 1] == str(tmp_path / "correction.json")
    with pytest.raises(ValueError, match="do not also apply BaSiC"):
        fuse_tiles(**kwargs, flatfield_dirs_by_source_view={"CL": tmp_path})
