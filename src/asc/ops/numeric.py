# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Numerical metadata for active native dtypes."""

from __future__ import annotations

from asc import _array_api_compat, errors
from asc.core import namespace as namespace_module


def _finfo(array_or_dtype: object) -> object:
    if not isinstance(
        array_or_dtype, type
    ) and _array_api_compat.compat.is_array_api_obj(array_or_dtype):
        xp = namespace_module.array_namespace(array_or_dtype)
        dtype = array_or_dtype.dtype
    else:
        raise errors.DTypeError(
            "numeric metadata: pass a native floating or complex array"
        )
    try:
        return xp.finfo(dtype)
    except (AttributeError, TypeError, ValueError) as exception:
        raise errors.DTypeError(
            "numeric metadata: dtype must be floating or complex"
        ) from exception


def eps(array: object) -> float:
    """Return machine epsilon for a native array's dtype."""
    return float(_finfo(array).eps)


def tiny(array: object) -> float:
    """Return the smallest positive normal for a native array's dtype."""
    return float(_finfo(array).smallest_normal)


def finite_range(array: object) -> tuple[float, float]:
    """Return the finite minimum and maximum of an active dtype."""
    info = _finfo(array)
    return float(info.min), float(info.max)


__all__ = ["eps", "finite_range", "tiny"]
