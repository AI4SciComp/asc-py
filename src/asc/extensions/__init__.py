# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit backend extensions outside the portable Array API core."""

from __future__ import annotations

import importlib
import typing

if typing.TYPE_CHECKING:
    from asc.extensions import autodiff, compilation, indexing, random


def __getattr__(name: str) -> object:
    """Load one public compatibility module on first attribute access."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


__all__ = ["autodiff", "compilation", "indexing", "random"]
