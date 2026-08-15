from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import tifffile

import squisher_deconv.basic as basic_module
from squisher_deconv.basic import fit_basic_profiles
from squisher_deconv.source import TiffLogicalSource


class FakeBasic:
    model_fields = {"device": object()}

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.smoothness_darkfield = 0.2
        self.baseline = np.ones(1, dtype=np.float32)
        self.autotune_calls = 0
        self.fit_calls = 0

    def autotune(self, images, *, is_timelapse, skip_shape_warning) -> None:
        assert images.ndim == 3
        assert is_timelapse is False
        assert skip_shape_warning is True
        self.autotune_calls += 1
        self.smoothness_flatfield = 5.0
        self.smoothness_darkfield = 0.5

    def fit(self, images, *, skip_shape_warning) -> None:
        assert skip_shape_warning is True
        self.fit_calls += 1
        shape = images.shape[1:]
        self.flatfield = np.ones(shape, dtype=np.float32)
        self.darkfield = np.full(shape, 3.0, dtype=np.float32)
        self.baseline = np.ones(images.shape[0], dtype=np.float32)


def _write_tile(path: Path, *, offset: int) -> None:
    data = np.empty((1, 6, 16, 16), dtype=np.uint16)
    for z in range(data.shape[1]):
        plane = np.full((16, 16), 100 + offset + z, dtype=np.uint16)
        plane[4:12, 4:12] += 500
        data[0, z] = plane
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "CZYX"}, photometric="minisblack")


def test_tiff_logical_source_maps_channel_z_to_page(tmp_path) -> None:
    path = tmp_path / "tile.ome.tif"
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "CZYX"}, photometric="minisblack")
    source = TiffLogicalSource.open(path, channels=2, metadata_mode="summary")

    assert source.page_key(channel=0, z=2) == 2
    assert source.page_key(channel=1, z=2) == 5


def test_basic_plane_reads_do_not_build_tiff_series(tmp_path) -> None:
    path = tmp_path / "tile.ome.tif"
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "CZYX"}, photometric="minisblack")
    source = TiffLogicalSource.open(path, channels=2, metadata_mode="summary")

    class Page:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def asarray(self) -> np.ndarray:
            return self.value

    class Series:
        def __getitem__(self, key: int) -> None:
            pytest.fail(f"TIFF series must not be built to read logical page {key}")

    class Tiff:
        pages = [Page(data[channel, z]) for channel in range(2) for z in range(3)]
        series = Series()

    plane = basic_module._read_plane(Tiff(), source, channel=1, z=2)

    assert np.array_equal(plane, data[1, 2])


def test_basic_plane_reader_reuses_tiff_index_across_channels(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tile.ome.tif"
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "CZYX"}, photometric="minisblack")
    source = TiffLogicalSource.open(path, channels=2, metadata_mode="summary")
    real_tiff_file = tifffile.TiffFile
    opened: list[Path] = []

    def recording_tiff_file(input_path: Path) -> tifffile.TiffFile:
        opened.append(Path(input_path))
        return real_tiff_file(input_path)

    monkeypatch.setattr(basic_module.tifffile, "TiffFile", recording_tiff_file)

    with basic_module._TiffPlaneReader() as reader:
        ch0 = reader.read(source, channel=0, z=2)
        ch1 = reader.read(source, channel=1, z=2)

    assert opened == [path]
    assert np.array_equal(ch0, data[0, 2])
    assert np.array_equal(ch1, data[1, 2])


def test_basic_source_discovery_inspects_only_representative_input(tmp_path, monkeypatch) -> None:
    inputs = [tmp_path / f"tile{index}.ome.tif" for index in range(3)]
    for index, path in enumerate(inputs):
        _write_tile(path, offset=index * 10)
    opened: list[Path] = []
    real_open = TiffLogicalSource.open

    def recording_open(path: Path, *, channels: int, metadata_mode: str) -> TiffLogicalSource:
        opened.append(path)
        return real_open(path, channels=channels, metadata_mode=metadata_mode)

    monkeypatch.setattr(TiffLogicalSource, "open", staticmethod(recording_open))

    sources = basic_module._inspect_sources(inputs, channels=1, progress=lambda _message: None)

    assert opened == [inputs[0].resolve()]
    assert [source.path for source in sources] == [path.resolve() for path in inputs]


def test_basic_sampling_balances_planes_over_a_bounded_tile_subset(tmp_path) -> None:
    path = tmp_path / "representative.ome.tif"
    _write_tile(path, offset=0)
    representative = TiffLogicalSource.open(path, channels=1, metadata_mode="summary")
    sources = [
        basic_module.replace(representative, path=tmp_path / f"tile{index:03d}.ome.tif", z_count=100)
        for index in range(90)
    ]

    sampled_sources = basic_module._sample_sources(sources, target=250, samples_per_tile=25)
    candidates = basic_module._candidate_order(sampled_sources, seed=7)
    first_batch = candidates[:250]

    assert len(sampled_sources) == 10
    assert sampled_sources[0] == sources[0]
    assert sampled_sources[-1] == sources[-1]
    assert {source.path for source, _z in first_batch} == {source.path for source in sampled_sources}
    assert all(sum(source == candidate for candidate, _z in first_batch) == 25 for source in sampled_sources)
    assert all(len({z for candidate, z in first_batch if candidate == source}) == 25 for source in sampled_sources)


def test_fit_basic_profiles_writes_run_compatible_joint_profiles(tmp_path) -> None:
    inputs = [tmp_path / "tile0.ome.tif", tmp_path / "tile1.ome.tif"]
    for index, path in enumerate(inputs):
        _write_tile(path, offset=index * 10)
    progress: list[str] = []

    outputs = fit_basic_profiles(
        inputs=inputs,
        out_dir=tmp_path / "basic",
        label="sample",
        channels=1,
        samples=4,
        cache_samples_per_channel=5,
        blank_slice_sample_stride=2,
        exclude_edge_slices=False,
        device="cpu",
        basic_factory=FakeBasic,
        progress=progress.append,
    )

    assert progress[0].startswith("basic start inputs=2 channels=1 samples=4")
    assert progress[1] == f"basic inspect representative=1/2 path={inputs[0].resolve()}"
    assert progress[2].startswith("basic inspect complete representative=")
    assert progress[-1].startswith("basic complete manifest=")
    assert outputs.profile_paths == (tmp_path / "basic" / "sample-ch0.pkl",)
    assert outputs.png_paths == (tmp_path / "basic" / "sample-ch0.png",)
    assert outputs.png_paths[0].is_file()
    payload = pickle.loads(outputs.profile_paths[0].read_bytes())
    assert payload["shared_profile"] is True
    assert payload["training_channels"] == ["ch0"]
    assert payload["basic"].autotune_calls == 1
    assert payload["basic"].fit_calls == 1
    assert payload["basic"].flatfield.shape == (16, 16)
    manifest = json.loads(outputs.manifest.read_text())
    assert manifest["basic_settings"]["autotune"] is True
    assert manifest["basic_settings"]["get_darkfield"] is True
    assert manifest["basic_settings"]["sort_intensity"] is True
    assert manifest["per_channel_samples"] == {"ch0": 4}
    assert manifest["sampling"]["selected_per_channel"] == {"ch0": 5}
    assert manifest["outputs"]["pngs"] == [str(outputs.png_paths[0].resolve())]


def test_fit_basic_profiles_refuses_existing_final_outputs(tmp_path) -> None:
    out_dir = tmp_path / "basic"
    out_dir.mkdir()
    existing = out_dir / "sample-ch0.pkl"
    existing.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="sample-ch0.pkl"):
        fit_basic_profiles(
            inputs=[tmp_path / "missing.ome.tif"],
            out_dir=out_dir,
            label="sample",
            channels=1,
            samples=1,
            cache_samples_per_channel=1,
            basic_factory=FakeBasic,
        )

    assert existing.read_bytes() == b"keep"
