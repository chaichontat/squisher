import json

import numpy as np
import pytest

from squisher_deconv.tile_gains import load_tile_gains, write_tile_gains


def test_write_tile_gains_emits_loader_validated_exact_source_rows(tmp_path):
    sources = [tmp_path / "a.tif", tmp_path / "b.tif"]
    for source in sources:
        source.touch()
    path = tmp_path / "gains.json"

    write_tile_gains(path, sources=sources, gains=np.asarray([[1.0, 1.2], [0.8, 1.0]]))

    loaded = load_tile_gains(path, inputs=list(reversed(sources)), channels=2)
    np.testing.assert_allclose(loaded[str(sources[0].resolve())], [1.0, 1.2])
    np.testing.assert_allclose(loaded[str(sources[1].resolve())], [0.8, 1.0])


def test_manifest_resolves_sources_and_preserves_channel_order(tmp_path):
    source = tmp_path / "tile.tif"
    source.touch()
    path = tmp_path / "gains.json"
    path.write_text(
        json.dumps(dict(schema_version=1, channels=3, tiles=[dict(source=str(source), gains=[1, 1.2, 0.8])]))
    )
    values = load_tile_gains(path, inputs=[source], channels=3)
    np.testing.assert_allclose(values[str(source.resolve())], [1, 1.2, 0.8])


@pytest.mark.parametrize("gains", [[1], [1, 0], [1, -1], [1, float("nan")], [True, 1]])
def test_manifest_rejects_invalid_channel_gains(tmp_path, gains):
    source = tmp_path / "tile.tif"
    source.touch()
    path = tmp_path / "gains.json"
    path.write_text(
        json.dumps(dict(schema_version=1, channels=2, tiles=[dict(source=str(source), gains=gains)]))
    )
    with pytest.raises(ValueError):
        load_tile_gains(path, inputs=[source], channels=2)


def test_manifest_rejects_missing_duplicate_and_unexpected_sources(tmp_path):
    source = tmp_path / "tile.tif"
    source.touch()
    row = dict(source=str(source), gains=[1])
    path = tmp_path / "gains.json"
    for rows in [[], [row, row], [row, dict(source=str(tmp_path / "extra.tif"), gains=[1])]]:
        path.write_text(json.dumps(dict(schema_version=1, channels=1, tiles=rows)))
        with pytest.raises(ValueError):
            load_tile_gains(path, inputs=[source], channels=1)


class GainSampleFactory:
    process_safe = True

    def __call__(self, device):
        from squisher_deconv.deconvolution import IdentityDeconvolver

        return IdentityDeconvolver()


@pytest.mark.parametrize("process", [False, True])
def test_scaling_samples_receive_each_sources_channel_gains(tmp_path, process):
    import tifffile
    from squisher_deconv.deconvolution import IdentityDeconvolver
    from squisher_deconv.streaming import sample_scale

    sources = [tmp_path / f"tile{i}.tif" for i in range(2)]
    for source in sources:
        tifffile.imwrite(
            source, np.full((4, 4, 5), 100, np.uint16), metadata={"axes": "ZYX"}, photometric="minisblack"
        )
    gains = [[1, 2], [3, 1]]
    path = tmp_path / "gains.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version=1,
                channels=2,
                tiles=[dict(source=str(s), gains=g) for s, g in zip(sources, gains)],
            )
        )
    )
    output = tmp_path / "scale"
    sample_scale(
        sources,
        out_dir=output,
        planes=4,
        channels=2,
        halo=0,
        deconvolver=None if process else IdentityDeconvolver(),
        deconvolver_factory=GainSampleFactory() if process else None,
        psf_paths=None,
        tile_gains_path=path,
        seed=1,
        p_low=0,
        p_high=1,
        gamma=1,
        bins=8,
        devices=[0],
        queue_depth=1,
        stop_on_error=True,
    )
    for i, expected in enumerate(gains):
        files = list((output / "float32-samples").glob(f"tile{i}-*.tif"))
        assert files
        for p in files:
            a = tifffile.imread(p)
            assert a.shape == (4, 4, 5)
            np.testing.assert_array_equal(a[0::2], expected[0] * 100)
            np.testing.assert_array_equal(a[1::2], expected[1] * 100)
    manifest = json.loads((output / "sample-manifest.json").read_text())
    assert manifest["tile_gains"]["path"] == str(path)


class GainCore:
    def deconvolve_core_u16(self, volume, *, core_start, core_stop, scaling, channel_gains, raw_z_start=None, raw_z_size=None):
        result = volume[core_start:core_stop] * np.array(channel_gains)[None, :, None, None]
        return result.reshape(-1, *result.shape[-2:]).astype(np.uint16)


def test_streamed_output_records_gains_and_resume_rejects_changed_manifest(tmp_path):
    import tifffile
    import zarr
    from squisher_deconv.process_workers import ProcessRunConfig, _process_file
    from squisher_deconv.scaling import ScalingParameters
    from squisher_deconv.source import TiffLogicalSource
    from squisher_deconv.streaming import _resume_pending_paths

    source = tmp_path / "tile.tif"
    tifffile.imwrite(
        source, np.full((4, 4, 5), 100, np.uint16), metadata={"axes": "ZYX"}, photometric="minisblack"
    )
    gain_path = tmp_path / "gains.json"
    gain_path.write_text(
        json.dumps(dict(schema_version=1, channels=2, tiles=[dict(source=str(source), gains=[1, 2])]))
    )
    scaling_path = tmp_path / "scaling.json"
    scaling_path.write_text("{}")
    config = ProcessRunConfig(
        out_dir=tmp_path / "out",
        channels=2,
        halo=0,
        slab_depth=1,
        output_mode="u16",
        psf_paths=(),
        basic_paths=(),
        scaling_path=scaling_path,
        devices=(0,),
        queue_depth=1,
        overwrite=False,
        output_relative_root=None,
        jpegxr_level=1,
        tile_gains=load_tile_gains(gain_path, inputs=[source], channels=2),
        tile_gains_path=gain_path,
    )
    _process_file(
        worker_id=0,
        device=0,
        file_index=0,
        path=source,
        template_source=TiffLogicalSource.open(source, channels=2, metadata_mode="summary"),
        scaling=ScalingParameters(
            offset=np.zeros(2), scale=np.ones(2), p_low=0, p_high=1, gamma=1, i_max=65535
        ),
        deconvolver=GainCore(),
        config=config,
    )
    out = tmp_path / "out/tile.ome.zarr"
    a = zarr.open_group(out, mode="r")["0"][:]
    np.testing.assert_array_equal(a[0], 100)
    np.testing.assert_array_equal(a[1], 200)
    provenance = json.loads((tmp_path / "out/tile.deconv.json").read_text())["provenance"]
    assert provenance["tile_gains"]["path"] == str(gain_path)
    expected = {k: provenance[k] for k in ["run_settings", "psfs", "basic_profiles", "scaling", "tile_gains"]}
    expected["tile_gains"] = dict(path=str(gain_path), sha256="changed")
    with pytest.raises(ValueError, match="tile_gains"):
        _resume_pending_paths([source], [out], expected_identity=expected)
