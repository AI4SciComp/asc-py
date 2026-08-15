# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Lazy optional-backend adapter loading."""

from __future__ import annotations

import importlib
import types

from asc import errors
from asc import typing as asc_typing


def load_backend(backend: asc_typing.BackendName) -> types.ModuleType:
    """Import one adapter only after its capability is requested."""
    if backend == "array_api_strict":
        raise errors.UnsupportedCapabilityError(
            "extension: array-api-strict has no backend extension adapter"
        )
    try:
        return importlib.import_module(f"asc.backends.{backend}")
    except ModuleNotFoundError as exception:
        if exception.name in {
            backend,
            "torch" if backend == "torch" else backend,
        }:
            raise errors.BackendUnavailableError(
                f"extension: backend {backend!r} is not installed; "
                f"install the asc-py[{backend}] extra"
            ) from exception
        raise
