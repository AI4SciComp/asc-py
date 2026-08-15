# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared index-width selection for portable integer arithmetic."""

from __future__ import annotations

from asc import errors
from asc import typing as asc_typing
from asc.backends.capabilities import backend_info


def safe_index_dtype(
    xp: asc_typing.ArrayNamespace,
    backend: asc_typing.BackendName,
    maximum: int,
    operation: str,
) -> object:
    """Return the widest active signed index dtype that fits a range."""
    dtype_name = "int64" if "int64" in backend_info(backend).dtypes else "int32"
    dtype = getattr(xp, dtype_name)
    if maximum > int(xp.iinfo(dtype).max):
        raise errors.CapabilityNotSupportedError(
            f"{operation}: index range exceeds the active backend integer "
            "capability"
        )
    return dtype


__all__: list[str] = []
