from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

import squisher_lightsheet.ome_metadata_dumb_stitch as dumb_stitch
from squisher_lightsheet.ome_metadata_dumb_stitch import (
    BasicProfile,
    TileMetadata,
    canvas_for_tiles,
    orient_plane_yx,
    read_tile_metadata,
)


def test_read_tile_metadata_accepts_ngff_v05_nested_multiscales(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "tile.ome.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.create_array(
        "0",
        shape=(2, 3, 4, 5),
        chunks=(1, 1, 4, 5),
        dtype="uint16",
        dimension_names=("c", "z", "y", "x"),
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": axis} for axis in ("c", "z", "y", "x")],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 2.0, 3.0, 4.0]},
                            {"type": "translation", "translation": [0.0, 10.0, 20.0, 30.0]},
                        ],
                    }
                ],
            }
        ],
    }

    metadata = read_tile_metadata(path)

    assert metadata.axes == "CZYX"
    assert metadata.shape == (2, 3, 4, 5)
    assert metadata.spacing_um_zyx == (2.0, 3.0, 4.0)
    assert metadata.translation_um_zyx == (10.0, 20.0, 30.0)


def test_read_level_two_metadata_and_center_plane_from_ome_zarr(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "tile.ome.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    shapes = ((1, 8, 16, 20), (1, 4, 8, 10), (1, 2, 4, 5))
    scales = ((1.0, 1.0, 0.5, 0.5), (1.0, 2.0, 1.0, 1.0), (1.0, 4.0, 2.0, 2.0))
    datasets = []
    for level, (shape, scale) in enumerate(zip(shapes, scales, strict=True)):
        array = root.create_array(
            str(level),
            shape=shape,
            chunks=(1, 1, shape[2], shape[3]),
            dtype="uint16",
            dimension_names=("c", "z", "y", "x"),
        )
        array[:] = level
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": list(scale)},
                    {"type": "translation", "translation": [0.0, 10.0, 20.0, 30.0]},
                ],
            }
        )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": axis} for axis in ("c", "z", "y", "x")],
                "datasets": datasets,
            }
        ],
    }

    metadata = read_tile_metadata(path, level=2)
    planes = dumb_stitch._read_planes(metadata, channels=(0,), z_index=1, level=2)

    assert metadata.shape == (1, 2, 4, 5)
    assert metadata.spacing_um_zyx == (4.0, 2.0, 2.0)
    assert metadata.translation_um_zyx == (10.0, 20.0, 30.0)
    np.testing.assert_array_equal(planes[0], np.full((4, 5), 2, dtype=np.float32))


def test_render_registers_jpegxr_before_opening_tiles(tmp_path, monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        dumb_stitch,
        "register_jpegxr_codec",
        lambda: events.append("registered"),
        raising=False,
    )

    def render_view(**_kwargs):
        assert events == ["registered"]
        return ({"outputs": {}}, [])

    monkeypatch.setattr(dumb_stitch, "_render_view", render_view)
    monkeypatch.setattr(dumb_stitch, "_write_contact_sheet", lambda *_args: None)

    dumb_stitch.render_ome_metadata_dumb_stitch(
        input_dirs_by_view={"R": tmp_path},
        output_dir=tmp_path / "output",
        channels=(0,),
    )


def test_read_tile_metadata_composes_dataset_and_multiscale_transforms(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "tile.ome.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.create_array(
        "pixels",
        shape=(1, 3, 4, 5),
        chunks=(1, 1, 4, 5),
        dtype="uint16",
        dimension_names=("c", "z", "y", "x"),
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": axis} for axis in ("c", "z", "y", "x")],
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 10.0, 10.0, 10.0]},
                    {"type": "translation", "translation": [0.0, 100.0, 200.0, 300.0]},
                ],
                "datasets": [
                    {
                        "path": "pixels",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 2.0, 3.0, 4.0]},
                            {"type": "translation", "translation": [0.0, 1.0, 2.0, 3.0]},
                        ],
                    }
                ],
            }
        ],
    }

    metadata = read_tile_metadata(path)

    assert metadata.spacing_um_zyx == (20.0, 30.0, 40.0)
    assert metadata.translation_um_zyx == (110.0, 220.0, 330.0)


def test_ome_tiff_metadata_does_not_build_tiff_series(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tile.ome.tif"
    ome_metadata = """<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image ID="Image:0"><Pixels DimensionOrder="XYZCT" Type="uint16"
        SizeX="5" SizeY="4" SizeZ="3" SizeC="2" SizeT="1"
        PhysicalSizeX="0.25" PhysicalSizeY="0.5" PhysicalSizeZ="1.0">
        <Plane TheC="0" TheZ="0" TheT="0" PositionX="30" PositionY="20" PositionZ="10"/>
      </Pixels></Image>
    </OME>"""

    class FakeTiffFile:
        def __init__(self, _path) -> None:
            self.ome_metadata = ome_metadata

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        @property
        def series(self):
            raise AssertionError("metadata discovery must not construct the TIFF series")

    monkeypatch.setattr(dumb_stitch, "TiffFile", FakeTiffFile)

    metadata = dumb_stitch._ome_tiff_metadata(path)

    assert metadata.axes == "CZYX"
    assert metadata.shape == (2, 3, 4, 5)
    assert metadata.spacing_um_zyx == (1.0, 0.5, 0.25)
    assert metadata.translation_um_zyx == (10.0, 20.0, 30.0)


def test_read_planes_opens_tiff_once_without_building_series(tmp_path, monkeypatch) -> None:
    opened = 0
    pages = [np.full((2, 2), index, dtype=np.uint16) for index in range(6)]

    class FakePage:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data

        def asarray(self) -> np.ndarray:
            return self.data

    class FakeTiffFile:
        def __init__(self, _path) -> None:
            nonlocal opened
            opened += 1
            self.pages = [FakePage(page) for page in pages]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        @property
        def series(self):
            raise AssertionError("plane reads must not construct the TIFF series")

    tile = TileMetadata(
        path=tmp_path / "tile.ome.tif",
        name="tile.ome.tif",
        axes="CZYX",
        shape=(2, 3, 2, 2),
        spacing_um_zyx=(1.0, 1.0, 1.0),
        translation_um_zyx=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(dumb_stitch, "TiffFile", FakeTiffFile)

    planes = dumb_stitch._read_planes(tile, channels=(0, 1), z_index=1)

    assert opened == 1
    np.testing.assert_array_equal(planes[0], np.full((2, 2), 1, dtype=np.float32))
    np.testing.assert_array_equal(planes[1], np.full((2, 2), 4, dtype=np.float32))


def test_basic_correction_precedes_signed_axis_orientation(tmp_path, monkeypatch) -> None:
    plane = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    tile = TileMetadata(
        path=tmp_path / "tile.ome.zarr",
        name="tile.ome.zarr",
        axes="ZYX",
        shape=(1, 2, 3),
        spacing_um_zyx=(1.0, -1.0, -1.0),
        translation_um_zyx=(0.0, 2.0, 3.0),
    )
    profile = BasicProfile(
        flatfield=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        darkfield=None,
        flatfield_path=Path("flatfield.tif"),
        darkfield_path=None,
    )
    rendered: list[np.ndarray] = []

    monkeypatch.setattr(dumb_stitch, "tile_paths_from_dir", lambda _path: [tile.path])
    monkeypatch.setattr(dumb_stitch, "read_tile_metadata", lambda _path, *, level: tile)
    monkeypatch.setattr(dumb_stitch, "_read_planes", lambda *_args, **_kwargs: {0: plane})
    monkeypatch.setattr(dumb_stitch, "load_basic_profile", lambda *_args, **_kwargs: profile)

    def capture_stretch(images):
        rendered.append(np.asarray(images[0]))
        return [np.zeros_like(images[0], dtype=np.uint8)], (0.0, 1.0)

    monkeypatch.setattr(dumb_stitch, "stretch_uint8", capture_stretch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    dumb_stitch._render_view(
        view="test",
        input_dir=tmp_path,
        output_dir=output_dir,
        channels=(0,),
        basic_dir=tmp_path,
        level=0,
        center_z_index=0,
        draw_tile_labels=False,
        draw_tile_outlines=False,
        write_tiff=False,
        output_prefix="test",
        progress=None,
    )

    np.testing.assert_array_equal(rendered[1], np.ones((2, 3), dtype=np.float32))


def test_render_view_writes_native_intensity_ome_tiff(tmp_path, monkeypatch) -> None:
    plane = np.array([[0, 17, 1024], [4096, 32768, 65535]], dtype=np.float32)
    tile = TileMetadata(
        path=tmp_path / "tile.ome.zarr",
        name="tile.ome.zarr",
        axes="ZYX",
        shape=(1, 2, 3),
        spacing_um_zyx=(1.0, 2.0, 3.0),
        translation_um_zyx=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(dumb_stitch, "tile_paths_from_dir", lambda _path: [tile.path])
    monkeypatch.setattr(dumb_stitch, "read_tile_metadata", lambda _path, *, level: tile)
    monkeypatch.setattr(dumb_stitch, "_read_planes", lambda *_args, **_kwargs: {0: plane})

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    summary, _ = dumb_stitch._render_view(
        view="test",
        input_dir=tmp_path,
        output_dir=output_dir,
        channels=(0,),
        basic_dir=None,
        level=0,
        center_z_index=0,
        draw_tile_labels=False,
        draw_tile_outlines=False,
        write_tiff=True,
        output_prefix="test",
        progress=None,
    )

    output = Path(summary["outputs"]["raw_ch0_tiff"])
    np.testing.assert_array_equal(tifffile.imread(output), plane.astype(np.uint16))
    with tifffile.TiffFile(output) as tif:
        assert tif.series[0].axes == "YX"
        assert tif.series[0].dtype == np.dtype("uint16")
        assert tif.pages[0].is_tiled


def test_render_view_refuses_lossy_raw_uint16_tiff(tmp_path, monkeypatch) -> None:
    plane = np.array([[0.5, 17.0], [1024.0, 65536.0]], dtype=np.float32)
    tile = TileMetadata(
        path=tmp_path / "tile.ome.zarr",
        name="tile.ome.zarr",
        axes="ZYX",
        shape=(1, 2, 2),
        spacing_um_zyx=(1.0, 2.0, 3.0),
        translation_um_zyx=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(dumb_stitch, "tile_paths_from_dir", lambda _path: [tile.path])
    monkeypatch.setattr(dumb_stitch, "read_tile_metadata", lambda _path, *, level: tile)
    monkeypatch.setattr(dumb_stitch, "_read_planes", lambda *_args, **_kwargs: {0: plane})

    with pytest.raises(ValueError, match="cannot be represented exactly as uint16"):
        dumb_stitch._render_view(
            view="test",
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
            channels=(0,),
            basic_dir=None,
            level=0,
            center_z_index=0,
            draw_tile_labels=False,
            draw_tile_outlines=False,
            write_tiff=True,
            output_prefix="test",
            progress=None,
        )


def test_negative_yx_scales_use_positive_canvas_and_flip_pixels() -> None:
    tile = TileMetadata(
        path=Path("tile.ome.zarr"),
        name="tile.ome.zarr",
        axes="ZYX",
        shape=(2, 3, 4),
        spacing_um_zyx=(1.0, -2.0, -3.0),
        translation_um_zyx=(0.0, 6.0, 12.0),
    )

    shape, pixel_um, bounds_min, bounds_max = canvas_for_tiles([tile])
    oriented = orient_plane_yx(np.arange(12).reshape(3, 4), tile.spacing_um_zyx[1:])

    assert shape == (3, 4)
    assert pixel_um == (2.0, 3.0)
    assert bounds_min == (0.0, 0.0)
    assert bounds_max == (6.0, 12.0)
    np.testing.assert_array_equal(oriented, np.arange(12).reshape(3, 4)[::-1, ::-1])
