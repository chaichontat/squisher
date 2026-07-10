from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

import fishtools.segment.extract_core as extract_core
from fishtools.segment.extract_core import (
    ContentEstimationVolume,
    LazyTiffArray,
    OrthoSliceRequest,
    RandomContentCrop,
    SingleChannelMaskArray,
    ZarrBackedTiffVolume,
    _fill_requested_ortho_strips,
    _open_aux_channel_volumes,
    _open_volume,
    _contentful_z_candidates_around,
    _sample_random_content_z_crops,
    _write_random_content_z_crops,
    _z_candidates_around,
    _z_crop_slice_around,
    normalize_numeric_options,
)
from fishtools.segment.extract_helpers import _expand_positions_with_context


def _write_tiff(path: Path, data: np.ndarray, axes: str) -> None:
    tifffile.imwrite(path, data, metadata={"axes": axes}, photometric="minisblack")


def test_lazy_tiff_normalizes_zcyx_and_czyx(tmp_path: Path) -> None:
    zcyx = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    zcyx_path = tmp_path / "zcyx.tif"
    _write_tiff(zcyx_path, zcyx, "ZCYX")
    vol, _ = _open_volume(zcyx_path)

    assert vol.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(vol[1, :, :, 2], zcyx[1, 2])

    czyx = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    czyx_path = tmp_path / "czyx.tif"
    _write_tiff(czyx_path, czyx, "CZYX")
    vol, _ = _open_volume(czyx_path)

    assert vol.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(vol[1, :, :, 2], czyx[2, 1])


def test_lazy_tiff_rejects_channel_only_series(tmp_path: Path) -> None:
    cyx_path = tmp_path / "cyx.tif"
    _write_tiff(cyx_path, np.zeros((3, 4, 5), dtype=np.uint16), "CYX")

    with pytest.raises(ValueError, match="requires a Z axis"):
        _open_volume(cyx_path)


def test_zarr_backed_tiff_volume_normalizes_czyx_with_windowed_keys(tmp_path: Path) -> None:
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    path = tmp_path / "aux.ome.tif"
    _write_tiff(path, data, "CZYX")
    vol = ZarrBackedTiffVolume(path)

    assert vol.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(vol[1, :, :, :], np.moveaxis(data[:, 1], 0, -1))
    np.testing.assert_array_equal(vol[:, 2, 1:4, 2], data[2, :, 2, 1:4])


def test_lazy_tiff_mask_preserves_spatial_singletons(tmp_path: Path) -> None:
    mask_data = np.arange(2 * 1 * 5, dtype=np.uint16).reshape(2, 1, 5)
    mask_path = tmp_path / "mask.tif"
    _write_tiff(mask_path, mask_data, "ZYX")
    mask = LazyTiffArray(mask_path, kind="mask")

    assert mask.shape == (2, 1, 5)
    np.testing.assert_array_equal(mask[1, :, :], mask_data[1])


def test_single_channel_mask_array_exposes_zyx() -> None:
    data = np.arange(2 * 1 * 4 * 5, dtype=np.uint16).reshape(2, 1, 4, 5)
    mask = SingleChannelMaskArray(data)

    assert mask.shape == (2, 4, 5)
    np.testing.assert_array_equal(mask[1, :2, :3], data[1, 0, :2, :3])


def test_open_volume_accepts_single_channel_ome_zarr_zyx(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    path = tmp_path / "tile.ome.zarr"
    root = zarr.open_group(str(path), mode="w")
    level0 = root.create_array("0", data=data, chunks=(1, 3, 4))
    level0.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "datasets": [{"path": "0"}],
            "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
        }
    ]

    vol, names = _open_volume(path)

    assert names is None
    assert vol.shape == (2, 3, 4, 1)
    np.testing.assert_array_equal(vol[1, 1:, :2, 0], data[1, 1:, :2])
    np.testing.assert_array_equal(vol[0, :2, :3, :], data[0, :2, :3, None])


def test_open_volume_accepts_ome_v05_nested_multiscales_without_array_dimensions(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    path = tmp_path / "tile.ome.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_array("0", data=data, chunks=(1, 3, 4))
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "datasets": [{"path": "0"}],
                "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
            }
        ],
    }

    vol, names = _open_volume(path)

    assert names is None
    assert vol.shape == (2, 3, 4, 1)
    np.testing.assert_array_equal(vol[1, :, :, 0], data[1])


def test_content_estimation_accepts_ome_v05_coarsest_zyx_without_array_dimensions(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "tile.ome.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_array("0", shape=(4, 8, 8), chunks=(1, 4, 4), dtype="uint16")
    root.create_array("1", shape=(2, 4, 4), chunks=(1, 4, 4), dtype="uint16")
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "datasets": [{"path": "0"}, {"path": "1"}],
                "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
            }
        ],
    }
    level0, _names = _open_volume(path)

    estimate = extract_core._content_estimation_volume(path, level0)

    assert estimate.vol.shape == (2, 4, 4, 1)
    assert estimate.scale_zyx == (2.0, 2.0, 2.0)


def test_ortho_zarr_reads_only_requested_slab_windows() -> None:
    class RecordingVolume:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data
            self.shape = data.shape
            self.ndim = data.ndim
            self.chunks = (2, 2, 2, 1)
            self.keys = []

        def __getitem__(self, key):
            self.keys.append(key)
            return self.data[key]

    data = np.arange(2 * 4 * 5 * 2, dtype=np.uint16).reshape(2, 4, 5, 2)
    vol = RecordingVolume(data)
    requests = [
        OrthoSliceRequest(
            axis="y",
            position=2,
            perpendicular_slice=slice(1, 4),
            out_file=Path("y.tif"),
            axes="CZX",
        ),
        OrthoSliceRequest(
            axis="x",
            position=3,
            perpendicular_slice=slice(0, 2),
            out_file=Path("x.tif"),
            axes="CZY",
        ),
    ]

    caches = _fill_requested_ortho_strips(
        vol=vol,
        mask_vol=None,
        other_vol=None,
        requests=requests,
        selected_indices=[1],
    )

    assert vol.keys == [
        (slice(0, 2), 2, slice(1, 4), 1),
        (slice(0, 2), slice(0, 2), 3, 1),
    ]
    np.testing.assert_array_equal(caches[0].channel_strips[0], data[:, 2, 1:4, 1])
    np.testing.assert_array_equal(caches[1].channel_strips[0], data[:, 0:2, 3, 1])


def test_ortho_appends_auxiliary_channels_from_matching_stack() -> None:
    primary = np.arange(2 * 4 * 5 * 1, dtype=np.uint16).reshape(2, 4, 5, 1)
    aux = np.zeros((2, 4, 5, 2), dtype=np.uint16)
    aux[..., 0] = 100
    aux[..., 1] = 200
    request = OrthoSliceRequest(
        axis="y",
        position=2,
        perpendicular_slice=slice(1, 4),
        out_file=Path("aux.tif"),
        axes="CZX",
    )

    caches = _fill_requested_ortho_strips(
        vol=primary,
        mask_vol=None,
        other_vol=None,
        requests=[request],
        selected_indices=[0],
        aux_channel_vols=[aux],
    )

    assert len(caches[0].channel_strips) == 3
    np.testing.assert_array_equal(caches[0].channel_strips[0], primary[:, 2, 1:4, 0])
    np.testing.assert_array_equal(caches[0].channel_strips[1], aux[:, 2, 1:4, 0])
    np.testing.assert_array_equal(caches[0].channel_strips[2], aux[:, 2, 1:4, 1])


def test_auxiliary_channel_stack_must_match_primary_spatial_shape(tmp_path: Path) -> None:
    primary = np.zeros((2, 4, 5, 1), dtype=np.uint16)
    aux = np.zeros((3, 4, 5), dtype=np.uint16)
    aux_path = tmp_path / "aux.ome.tif"
    _write_tiff(aux_path, aux, "ZYX")

    with pytest.raises(ValueError, match="does not match primary volume shape"):
        _open_aux_channel_volumes(aux_path, primary, label="test")


def test_ortho_zarr_respects_z_crop_window() -> None:
    data = np.arange(4 * 4 * 5 * 1, dtype=np.uint16).reshape(4, 4, 5, 1)
    request = OrthoSliceRequest(
        axis="y",
        position=2,
        perpendicular_slice=slice(1, 4),
        out_file=Path("crop.tif"),
        axes="CZX",
        z_slice=slice(1, 3),
    )

    caches = _fill_requested_ortho_strips(
        vol=data,
        mask_vol=None,
        other_vol=None,
        requests=[request],
        selected_indices=[0],
    )

    assert caches[0].channel_strips[0].shape == (2, 3)
    np.testing.assert_array_equal(caches[0].channel_strips[0], data[1:3, 2, 1:4, 0])


def test_content_ortho_context_expands_25_adjacent_slabs_each_direction_step_two() -> None:
    positions = _expand_positions_with_context(
        [100],
        crop=0,
        axis_len=300,
        step=2,
        context_pairs=25,
    )

    assert len(positions) == 51
    assert positions[0] == 100
    assert set(positions) == {100 + 2 * offset for offset in range(-25, 26)}


def test_content_ortho_z_crop_uses_25_adjacent_planes_each_direction() -> None:
    assert _z_crop_slice_around(100, z_len=300) == slice(75, 126)


def test_content_ortho_z_crop_clips_at_volume_bounds() -> None:
    assert _z_crop_slice_around(10, z_len=300) == slice(0, 36)
    assert _z_crop_slice_around(290, z_len=300) == slice(265, 300)


def test_z_candidates_around_uses_25_adjacent_planes_each_direction() -> None:
    assert _z_candidates_around(100, z_len=300, dz=1) == list(range(75, 126))


def test_z_candidates_around_respects_dz_from_sampled_center() -> None:
    assert _z_candidates_around(100, z_len=300, dz=2) == list(range(76, 125, 2))


def test_zarr_upscale_respects_explicit_value() -> None:
    assert (
        normalize_numeric_options(
            mode="ortho",
            dz=1,
            anisotropy=1,
            upscale=1.0,
            use_zarr=True,
            has_max_from=False,
            ortho_anisotropy_default=6,
        )
        == 1.0
    )
    assert (
        normalize_numeric_options(
            mode="ortho",
            dz=1,
            anisotropy=1,
            upscale=None,
            use_zarr=True,
            has_max_from=False,
            ortho_anisotropy_default=6,
        )
        == 2.0
    )


def test_random_content_z_crop_writes_512_tile(tmp_path: Path) -> None:
    data = np.ones((2, 512, 512, 1), dtype=np.uint16)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _write_random_content_z_crops(
        file=tmp_path / "tile.ome.zarr",
        roi="tile",
        vol=data,
        mask_vol=None,
        selected_indices=[0],
        out_names=["0"],
        out_dir=out_dir,
        channels="0",
        upscale=1.0,
        content_crops=[RandomContentCrop(z_index=0, y0=0, x0=0)],
    )

    outputs = sorted(out_dir.glob("*.tif"))
    assert len(outputs) == 1
    with tifffile.TiffFile(outputs[0]) as tif:
        assert tif.asarray().shape == (1, 512, 512)
        assert json.loads(tif.pages[0].description)["axes"] == "CYX"


def test_random_content_z_crop_appends_auxiliary_channels(tmp_path: Path) -> None:
    data = np.ones((1, 512, 512, 1), dtype=np.uint16)
    aux = np.zeros((1, 512, 512, 2), dtype=np.uint16)
    aux[..., 0] = 11
    aux[..., 1] = 22
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _write_random_content_z_crops(
        file=tmp_path / "tile.ome.zarr",
        roi="tile",
        vol=data,
        mask_vol=None,
        selected_indices=[0],
        out_names=["0", "aux0", "aux1"],
        out_dir=out_dir,
        channels="0",
        upscale=1.0,
        content_crops=[RandomContentCrop(z_index=0, y0=0, x0=0)],
        aux_channel_vols=[aux],
    )

    outputs = sorted(out_dir.glob("*.tif"))
    assert len(outputs) == 1
    with tifffile.TiffFile(outputs[0]) as tif:
        assert tif.series[0].shape == (3, 512, 512)
        out = tif.asarray()
    assert np.all(out[0] == 1)
    assert np.all(out[1] == 11)
    assert np.all(out[2] == 22)


def test_random_content_sampling_can_use_coarsest_level_estimator() -> None:
    level0 = np.zeros((2, 512, 512, 1), dtype=np.uint16)
    coarsest = np.ones((2, 128, 128, 1), dtype=np.uint16)

    crops = _sample_random_content_z_crops(
        vol=level0,
        content_estimation=ContentEstimationVolume(vol=coarsest, scale_zyx=(1.0, 4.0, 4.0)),
        crop=0,
        count=1,
        rng=np.random.default_rng(1),
        label="test",
    )

    assert len(crops) == 1
    assert crops[0].y0 == 0
    assert crops[0].x0 == 0


def test_zarr_z_extraction_default_samples_xyz_and_limits_z_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVolume:
        shape = (300, 1024, 1024, 1)
        ndim = 4

        def __getitem__(self, key: object) -> np.ndarray:
            return np.ones((512, 512, 1), dtype=np.uint16)

    vol = FakeVolume()
    jobs: list[extract_core.TileJob] = []

    monkeypatch.setattr(extract_core, "_open_volume", lambda _file: (vol, None))
    monkeypatch.setattr(
        extract_core,
        "_content_estimation_volume",
        lambda _file, level0_vol: ContentEstimationVolume(vol=level0_vol, scale_zyx=(1.0, 1.0, 1.0)),
    )
    monkeypatch.setattr(
        extract_core,
        "_sample_random_content_z_crops",
        lambda **_kwargs: [RandomContentCrop(z_index=100, y0=64, x0=128)],
    )

    def record_job(*, job: extract_core.TileJob, **_kwargs) -> None:
        jobs.append(job)

    monkeypatch.setattr(extract_core, "_extract_tiles_from_zarr", record_job)

    extract_core._execute_zarr_z_extraction(
        label="tile",
        files=[tmp_path / "Image_14.064.ome.zarr"],
        config=extract_core.ExtractionConfig(
            mode="z",
            channels=None,
            crop=0,
            dz=1,
            n=1,
            anisotropy=6,
            upscale=1.0,
            seed=1,
            threads=1,
        ),
        out_dir=tmp_path / "out",
        max_from_path=None,
        explicit_mask_path=None,
        enrich_boundaries=None,
    )

    assert len(jobs) == 1
    assert jobs[0].tile_origins == [(64, 128)]
    assert jobs[0].z_candidates == list(range(75, 126))


def test_contentful_z_candidates_replace_empty_planes() -> None:
    vol = np.ones((130, 512, 512, 1), dtype=np.uint16)
    vol[75] = 0

    candidates = _contentful_z_candidates_around(
        vol=vol,
        center=100,
        y0=0,
        x0=0,
        z_len=vol.shape[0],
        dz=1,
    )

    assert 75 not in candidates
    assert len(candidates) == len(_z_candidates_around(100, z_len=vol.shape[0], dz=1))
    assert min(candidates) < 75 or max(candidates) > 125


def test_zarr_z_tile_extraction_rejects_empty_z_candidates(tmp_path: Path) -> None:
    class FakeVolume:
        shape = (300, 1024, 1024, 1)
        ndim = 4

    job = extract_core.TileJob(
        file=tmp_path / "Image_14.064.ome.zarr",
        vol=FakeVolume(),
        channel_names=None,
        mask_vol=None,
        mask_path=None,
        tile_origins=[(0, 0)],
        z_candidates=[],
    )

    with pytest.raises(ValueError, match="No Z indices"):
        extract_core._extract_tiles_from_zarr(
            job=job,
            roi="tile",
            out_dir=tmp_path / "out",
            channels=None,
            dz=1,
            upscale=1.0,
            max_from_path=None,
            progress=None,
        )


def test_zarr_z_tile_extraction_appends_aux_channel(tmp_path: Path) -> None:
    primary = np.ones((1, 512, 512, 1), dtype=np.uint16)
    aux = np.full((1, 512, 512, 1), 17, dtype=np.uint16)
    job = extract_core.TileJob(
        file=tmp_path / "Image_14.064.ome.zarr",
        vol=primary,
        channel_names=["Image_14"],
        mask_vol=None,
        mask_path=None,
        tile_origins=[(0, 0)],
        z_candidates=[0],
        aux_channel_vols=[aux],
        aux_channel_names=["Image_10"],
    )

    extract_core._extract_tiles_from_zarr(
        job=job,
        roi="tile",
        out_dir=tmp_path,
        channels="0",
        dz=1,
        upscale=1.0,
        max_from_path=None,
        progress=None,
    )

    outputs = sorted(tmp_path.glob("*.tif"))
    assert len(outputs) == 1
    with tifffile.TiffFile(outputs[0]) as tif:
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (2, 512, 512)
        metadata = tif.shaped_metadata[0]
        assert metadata["channel_names"] == ["0", "Image_10"]
        out = tif.asarray()
    assert np.all(out[0] == 1)
    assert np.all(out[1] == 17)


def test_zarr_maxproj_tile_extraction_rejects_empty_z_candidates(tmp_path: Path) -> None:
    class FakeVolume:
        shape = (300, 1024, 1024, 1)
        ndim = 4

    job = extract_core.TileJob(
        file=tmp_path / "Image_14.064.ome.zarr",
        vol=FakeVolume(),
        channel_names=None,
        mask_vol=None,
        mask_path=None,
        tile_origins=[(0, 0)],
        z_candidates=[],
    )

    with pytest.raises(ValueError, match="No Z indices"):
        extract_core._extract_maxproj_tiles_from_zarr(
            job=job,
            roi="tile",
            out_dir=tmp_path / "out",
            channels=None,
            dz=1,
            upscale=1.0,
            max_from_path=None,
            progress=None,
        )


def test_ortho_reads_only_selected_channel_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = np.arange(2 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 4, 5, 6)
    path = tmp_path / "zcyx.tif"
    _write_tiff(path, data, "ZCYX")
    vol, _ = _open_volume(path)
    read_pages: list[int] = []
    original_read_page = vol._read_page

    def track_read_page(page_index: int) -> np.ndarray:
        read_pages.append(page_index)
        return original_read_page(page_index)

    monkeypatch.setattr(vol, "_read_page", track_read_page)
    requests = [
        OrthoSliceRequest(
            axis="y",
            position=2,
            perpendicular_slice=slice(1, 5),
            out_file=tmp_path / "out.tif",
            axes="CZX",
        )
    ]

    caches = _fill_requested_ortho_strips(
        vol=vol,
        mask_vol=None,
        other_vol=None,
        requests=requests,
        selected_indices=[2],
    )

    assert read_pages == [2, 6]
    np.testing.assert_array_equal(caches[0].channel_strips[0], data[:, 2, 2, 1:5])
