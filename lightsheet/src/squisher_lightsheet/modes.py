from __future__ import annotations

from typing import Literal


ModeName = Literal["tltr_x_join_center_z_phase", "lr_z_endview_flip_xz"]
DEFAULT_OVERLAP_FRACTION = 0.25

POSITION_MODES = {
    "tltr_x_join_center_z_phase": {
        "join_axis": "x",
        "right_flip_axes": (),
        "overlap_fraction": DEFAULT_OVERLAP_FRACTION,
    },
    "lr_z_endview_flip_xz": {
        "join_axis": "z",
        "right_flip_axes": ("z", "x"),
        "overlap_fraction": DEFAULT_OVERLAP_FRACTION,
    },
}
