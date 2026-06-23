from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import typer

from squisher_lightsheet._legacy.stitch_20x_tl_multiview import (
    parse_source_view_flatfield_dir as parse_legacy_source_view_flatfield_dir,
)
from squisher_lightsheet.cli import _parse_source_view_flatfield_dir as parse_cli_source_view_flatfield_dir
from squisher_lightsheet.parsing import parse_source_view_path_entry


def test_parse_source_view_path_entry_accepts_view_path_pair() -> None:
    assert parse_source_view_path_entry(
        "L=/tmp/basic",
        error_factory=ValueError,
    ) == ("L", Path("/tmp/basic"))


def test_parse_source_view_path_entry_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected source-view flatfield entry"):
        parse_source_view_path_entry("L", error_factory=ValueError)


def test_source_view_flatfield_wrappers_preserve_exception_types() -> None:
    with pytest.raises(typer.BadParameter):
        parse_cli_source_view_flatfield_dir("=/tmp/basic")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_legacy_source_view_flatfield_dir("=/tmp/basic")
