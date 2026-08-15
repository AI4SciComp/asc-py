# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Functional indexed and scatter update operations."""

from __future__ import annotations

import typing

from asc import errors
from asc import typing as asc_typing
from asc.core import _dtype
from asc.core import namespace as namespace_module
from asc.core._indexing import safe_index_dtype
from asc.extensions import _dispatch


class _UpdateAdapter(typing.Protocol):
    """Backend-specific functional update primitive."""

    def index_update(
        self,
        array: asc_typing.NativeArray,
        indices: asc_typing.NativeArray,
        values: asc_typing.NativeArray,
        *,
        axis: int,
        update_shape: asc_typing.Shape,
        reduction: str,
    ) -> asc_typing.NativeArray:
        """Apply one native functional update."""
        ...  # pylint: disable=unnecessary-ellipsis


def _update(
    array: asc_typing.NativeArray,
    indices: asc_typing.NativeArray,
    values: asc_typing.NativeArray,
    *,
    axis: int,
    reduction: str,
) -> asc_typing.NativeArray:
    operation = f"index_{reduction}"
    xp = namespace_module.array_namespace(array, indices, values)
    backend = namespace_module.identify_backend(xp)
    if backend == "array_api_strict":
        raise errors.CapabilityNotSupportedError(
            f"{operation}: array-api-strict has no functional update extension"
        )
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise errors.IndexUpdateError(f"{operation}: axis must be an integer")
    shape = array.shape
    if not shape:
        raise errors.IndexUpdateError(
            f"{operation}: destination must have rank at least one"
        )
    normalized_axis = axis + len(shape) if axis < 0 else axis
    if normalized_axis < 0 or normalized_axis >= len(shape):
        raise errors.IndexUpdateError(f"{operation}: axis is out of bounds")
    if len(indices.shape) != 1:
        raise errors.IndexUpdateError(
            f"{operation}: indices must be one-dimensional"
        )
    if not xp.isdtype(indices.dtype, "signed integer"):
        raise errors.IndexUpdateError(
            f"{operation}: indices must have a signed integer dtype"
        )
    numeric = (
        "signed integer",
        "unsigned integer",
        "real floating",
        "complex floating",
    )
    if reduction != "set" and (
        not xp.isdtype(array.dtype, numeric)
        or not xp.isdtype(values.dtype, numeric)
    ):
        raise errors.DTypeError(
            f"{operation}: arithmetic updates require numeric arrays"
        )
    if reduction in {"min", "max"} and (
        xp.isdtype(array.dtype, "complex floating")
        or xp.isdtype(values.dtype, "complex floating")
    ):
        raise errors.DTypeError(
            f"{operation}: ordered updates do not support complex arrays"
        )
    update_shape = (
        *shape[:normalized_axis],
        indices.shape[0],
        *shape[normalized_axis + 1 :],
    )
    try:
        index_dtype = safe_index_dtype(
            xp,
            backend,
            shape[normalized_axis] - 1,
            operation,
        )
        dispatch_indices = xp.astype(indices, index_dtype, copy=True)
        result_dtype = _dtype.extension_result_type(
            xp, array, values, operation=operation
        )
        destination = xp.astype(array, result_dtype, copy=True)
        source = xp.astype(values, result_dtype, copy=True)
        source = xp.broadcast_to(source, update_shape)
    except (RuntimeError, TypeError, ValueError) as exception:
        raise errors.DTypeError(
            f"{operation}: values must broadcast and promote with destination"
        ) from exception
    adapter = typing.cast(_UpdateAdapter, _dispatch.load_backend(backend))
    try:
        result = adapter.index_update(
            destination,
            dispatch_indices,
            source,
            axis=normalized_axis,
            update_shape=update_shape,
            reduction=reduction,
        )
    except errors.AscError:
        raise
    except (IndexError, RuntimeError, TypeError, ValueError) as exception:
        raise errors.IndexUpdateError(
            f"{operation}: backend rejected indices, shape, or values"
        ) from exception
    if result.shape != shape or result.dtype != result_dtype:
        raise errors.IndexUpdateError(
            f"{operation}: backend changed result shape or promoted dtype"
        )
    return result


def index_set(
    array: object, indices: object, values: object, *, axis: int = 0
) -> object:
    """Functionally set indexed slices; duplicate indices are rejected."""
    return _update(array, indices, values, axis=axis, reduction="set")


def index_add(
    array: object, indices: object, values: object, *, axis: int = 0
) -> object:
    """Functionally add values into indexed slices."""
    return _update(array, indices, values, axis=axis, reduction="add")


def index_multiply(
    array: object, indices: object, values: object, *, axis: int = 0
) -> object:
    """Functionally multiply values into indexed slices."""
    return _update(array, indices, values, axis=axis, reduction="multiply")


def index_min(
    array: object, indices: object, values: object, *, axis: int = 0
) -> object:
    """Functionally reduce indexed slices by minimum."""
    return _update(array, indices, values, axis=axis, reduction="min")


def index_max(
    array: object, indices: object, values: object, *, axis: int = 0
) -> object:
    """Functionally reduce indexed slices by maximum."""
    return _update(array, indices, values, axis=axis, reduction="max")


scatter_set = index_set
scatter_add = index_add
scatter_multiply = index_multiply
scatter_min = index_min
scatter_max = index_max

__all__ = [
    "index_add",
    "index_max",
    "index_min",
    "index_multiply",
    "index_set",
    "scatter_add",
    "scatter_max",
    "scatter_min",
    "scatter_multiply",
    "scatter_set",
]
