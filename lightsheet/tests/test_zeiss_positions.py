import json
from pathlib import Path

import pytest

from squisher_lightsheet.zeiss_positions import (
    ZeissPosition,
    offset_positions,
    plot_xy_positions,
    read_zeiss_positions,
    write_zeiss_positions,
)


ZEISS_POSITIONS = """Carl Zeiss LSM 510 - Position list file - Version = 1.000
BEGIN PositionList Version = 10000
\tNumberPositions = 2
\tBEGIN Position1 Version = 10000
\t\tX = -2012.460 µm
\t\tY = 20430.143 µm
\t\tZ = 10.260 µm
\tEND
\tBEGIN Position2 Version = 10000
\t\tX = -2626.448 µm
\t\tY = 19509.161 µm
\t\tZ = 11.500 µm
\tEND
END
"""


def test_read_zeiss_positions_returns_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "input.pos"
    path.write_text(ZEISS_POSITIONS)

    assert read_zeiss_positions(path) == [
        ZeissPosition(x=-2012.460, y=20430.143, z=10.260),
        ZeissPosition(x=-2626.448, y=19509.161, z=11.500),
    ]


def test_offset_positions_translates_every_dimension() -> None:
    original = [ZeissPosition(x=1.0, y=2.0, z=3.0)]

    shifted = offset_positions(original, x=10.0, y=-20.0, z=0.5)

    assert shifted == [ZeissPosition(x=11.0, y=-18.0, z=3.5)]
    assert original == [ZeissPosition(x=1.0, y=2.0, z=3.0)]


def test_write_zeiss_positions_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "output.pos"
    positions = [
        ZeissPosition(x=-1.25, y=2.5, z=3.0),
        ZeissPosition(x=4.125, y=-5.5, z=6.75),
    ]

    write_zeiss_positions(path, positions)

    assert read_zeiss_positions(path) == positions
    text = path.read_text()
    assert "NumberPositions = 2" in text
    assert "BEGIN Position2 Version = 10000" in text


def test_read_zeiss_positions_rejects_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pos"
    path.write_text(ZEISS_POSITIONS.replace("NumberPositions = 2", "NumberPositions = 3"))

    with pytest.raises(ValueError, match="declares 3 positions but contains 2"):
        read_zeiss_positions(path)


def test_plot_xy_positions_closes_each_rectangle(tmp_path: Path, monkeypatch) -> None:
    from matplotlib.axes import Axes
    from matplotlib.colors import to_rgba

    input_path = tmp_path / "input.pos"
    output_path = tmp_path / "positions.png"
    tile_positions_path = tmp_path / "tiles.json"
    positions = [
        ZeissPosition(x=1.0, y=4.0, z=0.0),
        ZeissPosition(x=3.0, y=4.0, z=0.0),
        ZeissPosition(x=3.0, y=2.0, z=0.0),
        ZeissPosition(x=1.0, y=2.0, z=0.0),
    ]
    write_zeiss_positions(input_path, positions)
    tile_positions_path.write_text(
        json.dumps(
            {
                "diagnostics": {"tile_footprint_yx_um": [20.0, 10.0]},
                "tiles": [
                    {
                        "mosaic_index": 17,
                        "translation_um": {"z": 0.0, "y": 198.0, "x": 102.0},
                    }
                ],
            }
        )
    )
    plotted_lines = []
    inverted_x_axes = []
    added_patches = []
    added_text = []
    original_plot = Axes.plot
    original_invert_xaxis = Axes.invert_xaxis
    original_add_patch = Axes.add_patch
    original_text = Axes.text

    def record_plot(self, x_values, y_values, *args, **kwargs):
        plotted_lines.append((list(x_values), list(y_values)))
        return original_plot(self, x_values, y_values, *args, **kwargs)

    def record_invert_xaxis(self):
        inverted_x_axes.append(self)
        return original_invert_xaxis(self)

    def record_add_patch(self, patch):
        added_patches.append(patch)
        return original_add_patch(self, patch)

    def record_text(self, x, y, text, *args, **kwargs):
        added_text.append((x, y, text, kwargs))
        return original_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)
    monkeypatch.setattr(Axes, "invert_xaxis", record_invert_xaxis)
    monkeypatch.setattr(Axes, "add_patch", record_add_patch)
    monkeypatch.setattr(Axes, "text", record_text)

    result = plot_xy_positions(
        input_path,
        output_path,
        title="Stage positions",
        tile_positions=tile_positions_path,
    )

    assert result == output_path.resolve()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plotted_lines == [([1.0, 3.0, 3.0, 1.0, 1.0], [4.0, 4.0, 2.0, 2.0, 4.0])]
    assert len(inverted_x_axes) == 1
    assert len(added_patches) == 1
    assert added_patches[0].get_bbox().bounds == (102.0, 198.0, 10.0, 20.0)
    assert added_patches[0].get_edgecolor() == to_rgba("dimgray")
    assert added_text[0] == (
        107.0,
        208.0,
        "17",
        {"color": "lightgray", "fontsize": 5, "ha": "center", "va": "center", "zorder": 1},
    )


def test_plot_xy_positions_rejects_incomplete_rectangle(tmp_path: Path) -> None:
    input_path = tmp_path / "input.pos"
    write_zeiss_positions(
        input_path,
        [
            ZeissPosition(x=1.0, y=1.0, z=0.0),
            ZeissPosition(x=2.0, y=1.0, z=0.0),
        ],
    )

    with pytest.raises(ValueError, match="groups of four"):
        plot_xy_positions(input_path, tmp_path / "positions.png", title="Stage positions")
