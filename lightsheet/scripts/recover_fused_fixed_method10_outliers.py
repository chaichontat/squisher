#!/usr/bin/env python
"""Compatibility entry point for fused-fixed spatial recovery."""

from __future__ import annotations

from squisher_lightsheet import fused_fixed_recovery as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
