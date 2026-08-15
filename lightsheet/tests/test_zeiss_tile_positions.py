import json
from pathlib import Path

import pytest

from squisher_lightsheet.zeiss_positions import ZeissPosition, write_zeiss_positions
from squisher_lightsheet.zeiss_tile_positions import (
    OmeTilePosition,
    create_zeiss_tile_position_file,
    derive_tile_positions_from_pos,
    read_ome_tile_positions,
)


def ome_xml(*, output_index: int, x: float) -> str:
    return f"""<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
        SizeX="10" SizeY="20" SizeZ="3" SizeC="1" SizeT="1"
        PhysicalSizeX="1" PhysicalSizeY="1" PhysicalSizeZ="2">
        <Plane TheC="0" TheZ="0" TheT="0" PositionX="{x}" PositionY="0"/>
      </Pixels></Image>
      <StructuredAnnotations><MapAnnotation ID="Annotation:0"><Value>
        <M K="squisher.output_tile_index">{output_index}</M>
        <M K="squisher.tile_count">2</M>
        <M K="czi.mosaic_index">{output_index + 10}</M>
        <M K="czi.tile_x">{output_index * 8}</M>
        <M K="czi.tile_y">0</M>
        <M K="czi.tile_width">10</M>
        <M K="czi.tile_height">20</M>
      </Value></MapAnnotation></StructuredAnnotations>
    </OME>"""


def tile(path: Path, *, index: int, x: float) -> OmeTilePosition:
    return OmeTilePosition(
        path=path,
        size_bytes=100,
        mtime_ns=200,
        output_index=index,
        mosaic_index=index + 10,
        expected_tile_count=2,
        shape_zyx=(3, 20, 10),
        spacing_zyx=(2.0, 1.0, 1.0),
        translation_zyx=(0.0, 0.0, x),
        czi_box_xywh=(index * 8, 0, 10, 20),
    )


def test_read_ome_tile_positions_reuses_unchanged_cache(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    paths = [input_dir / f"tile.{index:03d}.ome.tif" for index in range(2)]
    for path in paths:
        path.write_bytes(b"tiff")
    opened = []

    class FakeTiffFile:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)
            opened.append(self.path)
            index = paths.index(self.path)
            self.ome_metadata = ome_xml(output_index=index, x=index * 8.0)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    import tifffile

    monkeypatch.setattr(tifffile, "TiffFile", FakeTiffFile)
    cache_path = tmp_path / "ome-cache.json"

    first = read_ome_tile_positions(input_dir, cache_path)
    second = read_ome_tile_positions(input_dir, cache_path)
    paths[1].write_bytes(b"changed")
    third = read_ome_tile_positions(input_dir, cache_path)

    assert first == second
    assert third[1].size_bytes == len(b"changed")
    assert opened == [paths[0], paths[1], paths[1]]


def test_read_ome_tile_positions_caches_partial_conversion(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    path = input_dir / "tile.000.ome.tif"
    path.write_bytes(b"tiff")

    class FakeTiffFile:
        ome_metadata = ome_xml(output_index=0, x=0.0)

        def __init__(self, _path: Path) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    import tifffile

    monkeypatch.setattr(tifffile, "TiffFile", FakeTiffFile)
    cache_path = tmp_path / "ome-cache.json"

    with pytest.raises(ValueError, match="Expected 2 OME tiles, found 1"):
        read_ome_tile_positions(input_dir, cache_path)

    assert json.loads(cache_path.read_text())["tiles"][0]["output_index"] == 0


def test_derive_tile_positions_from_pos_builds_clipped_meander_grid() -> None:
    def rectangle(x: float, y: float) -> list[ZeissPosition]:
        return [
            ZeissPosition(x=x, y=y + 10, z=7),
            ZeissPosition(x=x + 10, y=y + 10, z=7),
            ZeissPosition(x=x + 10, y=y, z=7),
            ZeissPosition(x=x, y=y, z=7),
        ]

    positions = [
        *rectangle(0, 0),
        *rectangle(20, 0),
        *rectangle(0, 10),
        *rectangle(20, 10),
    ]

    derived, diagnostics = derive_tile_positions_from_pos(
        positions,
        overlap_fraction=0,
        min_hull_overlap_fraction=0,
    )

    assert [item.translation_zyx for item in derived] == [
        (7, 10, 20),
        (7, 10, 10),
        (7, 10, 0),
        (7, 0, 20),
        (7, 0, 10),
        (7, 0, 0),
    ]
    assert [item.mosaic_index for item in derived] == [0, 1, 2, 5, 4, 3]
    assert diagnostics["grid_shape_yx"] == [2, 3]
    assert diagnostics["row_tile_counts_bottom_to_top"] == [3, 3]


def test_create_zeiss_tile_position_file_writes_lightsheet_positions(
    tmp_path: Path,
) -> None:
    pos_path = tmp_path / "input.pos"
    write_zeiss_positions(
        pos_path,
        [
            ZeissPosition(x=110.0, y=220.0, z=7.0),
            ZeissPosition(x=100.0, y=220.0, z=7.0),
            ZeissPosition(x=100.0, y=200.0, z=7.0),
            ZeissPosition(x=110.0, y=200.0, z=7.0),
        ],
    )
    output = tmp_path / "positions.json"

    result = create_zeiss_tile_position_file(
        pos_input=pos_path,
        output=output,
        side="sample",
    )

    payload = json.loads(output.read_text())
    assert result == output.resolve()
    assert payload["artifact_type"] == "squisher_lightsheet.zeiss_tile_grid.v1"
    assert payload["tiles"][0]["output_index"] == 0
    assert payload["tiles"][0]["mosaic_index"] == 0
    assert payload["tiles"][0]["translation_um"] == {"z": 7.0, "y": 200.0, "x": 100.0}
    assert payload["tiles"][0]["side"] == "sample"
