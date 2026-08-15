from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


_POSITION_COUNT = re.compile(r"NumberPositions\s*=\s*(\d+)")
_POSITION_START = re.compile(r"BEGIN Position(\d+) Version\s*=\s*\d+")
_COORDINATE_LINE = re.compile(
    r"([XYZ])\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*µm"
)


@dataclass(frozen=True, slots=True)
class ZeissPosition:
    """A stage position in micrometers."""

    x: float
    y: float
    z: float


def read_zeiss_positions(path: Path) -> list[ZeissPosition]:
    """Read a Zeiss LSM 510 ``.pos`` position list."""
    lines = path.read_text(encoding="utf-8").splitlines()
    count_matches = [
        match for line in lines if (match := _POSITION_COUNT.fullmatch(line.strip()))
    ]
    if len(count_matches) != 1:
        raise ValueError(f"{path} must contain exactly one NumberPositions field")
    declared_count = int(count_matches[0].group(1))

    positions: list[ZeissPosition] = []
    position_numbers: list[int] = []
    line_index = 0
    while line_index < len(lines):
        position_start = _POSITION_START.fullmatch(lines[line_index].strip())
        if position_start is None:
            line_index += 1
            continue

        position_number = int(position_start.group(1))
        coordinates: dict[str, float] = {}
        line_index += 1
        while line_index < len(lines) and lines[line_index].strip() != "END":
            line = lines[line_index].strip()
            coordinate = _COORDINATE_LINE.fullmatch(line)
            if coordinate is None:
                raise ValueError(
                    f"{path}:{line_index + 1}: invalid field in Position{position_number}: {line!r}"
                )
            axis = coordinate.group(1)
            if axis in coordinates:
                raise ValueError(f"{path}:{line_index + 1}: duplicate {axis} coordinate")
            coordinates[axis] = float(coordinate.group(2))
            line_index += 1

        if line_index == len(lines):
            raise ValueError(f"{path}: Position{position_number} has no END marker")
        missing = {"X", "Y", "Z"} - coordinates.keys()
        if missing:
            raise ValueError(
                f"{path}: Position{position_number} is missing {', '.join(sorted(missing))}"
            )
        position_numbers.append(position_number)
        positions.append(ZeissPosition(x=coordinates["X"], y=coordinates["Y"], z=coordinates["Z"]))
        line_index += 1

    if position_numbers != list(range(1, len(positions) + 1)):
        raise ValueError(
            f"{path}: position numbers must be consecutive starting at 1; found {position_numbers}"
        )
    if declared_count != len(positions):
        raise ValueError(
            f"{path} declares {declared_count} positions but contains {len(positions)}"
        )
    return positions


def offset_positions(
    positions: Iterable[ZeissPosition],
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> list[ZeissPosition]:
    """Return positions translated by the same XYZ offsets in micrometers."""
    return [
        ZeissPosition(x=position.x + x, y=position.y + y, z=position.z + z)
        for position in positions
    ]


def plot_xy_positions(
    input_path: Path,
    output_path: Path,
    *,
    title: str,
    tile_positions: Path | None = None,
) -> Path:
    """Plot each consecutive group of four stage positions as an XY rectangle."""
    positions = read_zeiss_positions(input_path)
    if not positions:
        raise ValueError(f"{input_path} contains no positions to plot")
    if len(positions) % 4:
        raise ValueError(f"{input_path} must contain positions in groups of four")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if tile_positions is not None:
        tile_payload = json.loads(tile_positions.read_text())
        footprint_y, footprint_x = tile_payload["diagnostics"]["tile_footprint_yx_um"]
        for tile in tile_payload["tiles"]:
            translation = tile["translation_um"]
            axes.add_patch(
                Rectangle(
                    (translation["x"], translation["y"]),
                    footprint_x,
                    footprint_y,
                    fill=False,
                    edgecolor="dimgray",
                    linewidth=0.8,
                    zorder=1,
                )
            )
            axes.text(
                translation["x"] + footprint_x / 2,
                translation["y"] + footprint_y / 2,
                str(tile["mosaic_index"]),
                color="lightgray",
                fontsize=5,
                ha="center",
                va="center",
                zorder=1,
            )
    for start in range(0, len(positions), 4):
        rectangle = positions[start : start + 4]
        if (
            len({position.x for position in rectangle}) != 2
            or len({position.y for position in rectangle}) != 2
            or len({(position.x, position.y) for position in rectangle}) != 4
        ):
            raise ValueError(f"{input_path}: positions {start + 1}-{start + 4} do not form a rectangle")
        closed_rectangle = [*rectangle, rectangle[0]]
        axes.plot(
            [position.x for position in closed_rectangle],
            [position.y for position in closed_rectangle],
            color="#1f77b4",
            marker="o",
            markersize=4,
            zorder=2,
        )
        axes.text(
            sum(position.x for position in rectangle) / 4,
            sum(position.y for position in rectangle) / 4,
            str(start // 4 + 1),
            fontsize=7,
            ha="center",
            va="center",
            zorder=3,
        )
    axes.set(title=title, xlabel="X (µm)", ylabel="Y (µm)")
    axes.set_aspect("equal", adjustable="box")
    axes.invert_xaxis()
    axes.grid(alpha=0.25)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_zeiss_positions(path: Path, positions: Iterable[ZeissPosition]) -> None:
    """Write positions as a Zeiss LSM 510 ``.pos`` position list."""
    position_list = list(positions)
    lines = [
        "Carl Zeiss LSM 510 - Position list file - Version = 1.000",
        "BEGIN PositionList Version = 10000",
        f"\tNumberPositions = {len(position_list)}",
    ]
    for index, position in enumerate(position_list, start=1):
        lines.extend(
            [
                f"\tBEGIN Position{index} Version = 10000",
                f"\t\tX = {position.x:.3f} µm",
                f"\t\tY = {position.y:.3f} µm",
                f"\t\tZ = {position.z:.3f} µm",
                "\tEND",
            ]
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
