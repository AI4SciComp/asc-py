# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Portable operations not supplied directly by the Array API standard."""

from __future__ import annotations

import importlib
import typing

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.core import namespace as namespace_module
from asc.core._indexing import safe_index_dtype
from asc.core._scalar import require_representable_scalar
from asc.extensions import _dispatch
from asc.ops.activations import (
    elu,
    gelu,
    leaky_relu,
    relu,
    selu,
    sigmoid,
    silu,
    softplus,
    softsign,
    tanhshrink,
)
from asc.ops.comparison import allclose, assert_allclose, isclose
from asc.ops.numeric import eps, finite_range, tiny
from asc.ops.signal import convolve1d, moving_mean


def diag(array: object, *, offset: int = 0) -> object:
    """Construct a matrix from a one-dimensional native array."""
    xp = namespace_module.array_namespace(array)
    if len(array.shape) != 1:
        raise ValueError("diag: array must be one-dimensional")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("diag: offset must be an integer")
    size = array.shape[0]
    dimension = size + abs(offset)
    eye = xp.eye(
        dimension,
        dimension,
        k=offset,
        dtype=array.dtype,
        device=_array_api_compat.compat.device(array),
    )
    padding = xp.zeros(
        (abs(offset),),
        dtype=array.dtype,
        device=_array_api_compat.compat.device(array),
    )
    row_values = (
        xp.concat((array, padding))
        if offset >= 0
        else xp.concat((padding, array))
    )
    values = xp.expand_dims(row_values, axis=1)
    return xp.where(eye != xp.zeros_like(eye), values, xp.zeros_like(eye))


def flatten(array: object) -> object:
    """Flatten to one dimension and always return new logical storage."""
    xp = namespace_module.array_namespace(array)
    return xp.reshape(array, (-1,), copy=True)


def ravel(array: object, *, copy: bool | None = None) -> object:
    """Flatten with an explicit Array API copy policy."""
    xp = namespace_module.array_namespace(array)
    return xp.reshape(array, (-1,), copy=copy)


def _pad_pairs(pad_width: object, ndim: int) -> tuple[tuple[int, int], ...]:
    if isinstance(pad_width, bool):
        raise ValueError("pad: pad_width must contain non-negative integers")
    if isinstance(pad_width, int):
        if pad_width < 0:
            raise ValueError(
                "pad: pad_width must contain non-negative integers"
            )
        pairs: tuple[object, ...] = ((pad_width, pad_width),) * ndim
    elif isinstance(pad_width, tuple):
        pairs = typing.cast(tuple[object, ...], pad_width)
    else:
        raise ValueError("pad: pad_width must be an integer or tuple of pairs")
    if len(pairs) != ndim:
        raise ValueError("pad: pad_width must provide one pair per array axis")
    normalized: list[tuple[int, int]] = []
    for pair in pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in pair
            )
        ):
            raise ValueError(
                "pad: each width must be a non-negative integer pair"
            )
        normalized.append(typing.cast(tuple[int, int], pair))
    return tuple(normalized)


def _padding_indices(
    xp: object,
    *,
    size: int,
    before: int,
    after: int,
    mode: str,
    device: object,
) -> object:
    raw = xp.arange(-before, size + after, dtype=xp.int32, device=device)
    if mode == "wrap":
        return xp.remainder(raw, size)
    if mode == "edge":
        return xp.clip(raw, 0, size - 1)
    if mode == "reflect":
        if size == 1:
            return xp.zeros_like(raw)
        period = 2 * (size - 1)
        phase = xp.remainder(raw, period)
        return xp.where(phase < size, phase, period - phase)
    period = 2 * size
    phase = xp.remainder(raw, period)
    return xp.where(phase < size, phase, period - 1 - phase)


def pad(
    array: object,
    pad_width: int | tuple[tuple[int, int], ...],
    *,
    mode: typing.Literal[
        "constant", "edge", "reflect", "symmetric", "wrap"
    ] = "constant",
    constant_values: object = False,
) -> object:
    """Pad an array with one of five portable, graph-safe modes."""
    xp = namespace_module.array_namespace(array)
    if mode not in {"constant", "edge", "reflect", "symmetric", "wrap"}:
        raise errors.CapabilityNotSupportedError(
            f"pad: mode {mode!r} is unsupported; use constant, edge, reflect, "
            "symmetric, or wrap"
        )
    if _array_api_compat.compat.is_array_api_obj(constant_values):
        raise errors.MixedBackendError(
            "pad: constant_values must be a Python scalar, not a native array"
        )
    result = array
    pairs = _pad_pairs(pad_width, len(array.shape))
    device = _array_api_compat.compat.device(array)
    if mode == "constant":
        require_representable_scalar(xp, array.dtype, constant_values, "pad")
    for axis, (before, after) in enumerate(pairs):
        if before == 0 and after == 0:
            continue
        shape = result.shape
        size = shape[axis]
        if mode != "constant":
            if size == 0:
                raise ValueError(f"pad: mode {mode!r} cannot pad an empty axis")
            indices = _padding_indices(
                xp,
                size=size,
                before=before,
                after=after,
                mode=mode,
                device=device,
            )
            result = xp.take(result, indices, axis=axis)
            continue
        pieces: list[object] = []
        for width in (before,):
            if width:
                block_shape = (*shape[:axis], width, *shape[axis + 1 :])
                pieces.append(
                    xp.full(
                        block_shape,
                        constant_values,
                        dtype=result.dtype,
                        device=device,
                    )
                )
        pieces.append(result)
        if after:
            block_shape = (*shape[:axis], after, *shape[axis + 1 :])
            pieces.append(
                xp.full(
                    block_shape,
                    constant_values,
                    dtype=result.dtype,
                    device=device,
                )
            )
        result = xp.concat(tuple(pieces), axis=axis)
    return result


def ravel_multi_index(
    coordinates: typing.Sequence[object],
    shape: tuple[int, ...],
) -> object:
    """Convert C-order coordinates to flat indices."""
    if len(coordinates) != len(shape) or not coordinates:
        raise ValueError(
            "ravel_multi_index: coordinates and shape ranks must match"
        )
    xp = namespace_module.array_namespace(*coordinates)
    if any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0
        for extent in shape
    ):
        raise ValueError(
            "ravel_multi_index: shape extents must be positive integers"
        )
    if any(
        not xp.isdtype(coordinate.dtype, "signed integer")
        for coordinate in coordinates
    ):
        raise errors.DTypeError(
            "ravel_multi_index: coordinates must have signed integer dtypes"
        )
    try:
        broadcast = xp.broadcast_arrays(*coordinates)
    except (RuntimeError, TypeError, ValueError) as exception:
        raise ValueError(
            "ravel_multi_index: coordinate shapes must broadcast"
        ) from exception
    backend = namespace_module.identify_backend(xp)
    size = 1
    for extent in shape:
        size *= extent
    index_dtype = safe_index_dtype(xp, backend, size - 1, "ravel_multi_index")
    broadcast = tuple(
        xp.astype(coordinate, index_dtype, copy=True)
        for coordinate in broadcast
    )
    for coordinate, extent in zip(broadcast, shape, strict=True):
        invalid = xp.any((coordinate < 0) | (coordinate >= extent))
        _check_index_bounds(invalid, backend, "ravel_multi_index")
    result = xp.zeros_like(broadcast[0], dtype=index_dtype)
    stride = 1
    for coordinate, extent in zip(
        reversed(broadcast), reversed(shape), strict=True
    ):
        result = result + coordinate * stride
        stride *= extent
    return result


def unravel_index(
    indices: object, shape: tuple[int, ...]
) -> tuple[object, ...]:
    """Convert flat indices to C-order coordinates."""
    xp = namespace_module.array_namespace(indices)
    if not shape or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0
        for extent in shape
    ):
        raise ValueError(
            "unravel_index: shape must contain positive integer extents"
        )
    if not xp.isdtype(indices.dtype, "signed integer"):
        raise errors.DTypeError(
            "unravel_index: indices must have a signed integer dtype"
        )
    size = 1
    for extent in shape:
        size *= extent
    backend = namespace_module.identify_backend(xp)
    index_dtype = safe_index_dtype(xp, backend, size - 1, "unravel_index")
    remaining = xp.astype(indices, index_dtype, copy=True)
    invalid = xp.any((remaining < 0) | (remaining >= size))
    _check_index_bounds(invalid, backend, "unravel_index")
    coordinates: list[object] = []
    for extent in reversed(shape):
        coordinates.append(xp.remainder(remaining, extent))
        remaining = xp.floor_divide(remaining, extent)
    return tuple(reversed(coordinates))


def _check_index_bounds(
    invalid: object,
    backend: asc_typing.BackendName,
    operation: str,
) -> None:
    """Keep dynamic JAX validation inside checkify tracing."""
    if backend == "jax" and "Tracer" in type(invalid).__name__:
        checkify = importlib.import_module("jax.experimental.checkify")
        checkify.check(~invalid, f"asc {operation} index out of bounds")
        return
    if backend == "torch":
        adapter = _dispatch.load_backend("torch")
        adapter.check_index_bounds(invalid, operation)
        return
    if bool(invalid):
        raise IndexError(f"{operation}: index is out of bounds")


__all__ = [
    "allclose",
    "assert_allclose",
    "convolve1d",
    "diag",
    "elu",
    "eps",
    "finite_range",
    "flatten",
    "gelu",
    "isclose",
    "leaky_relu",
    "moving_mean",
    "pad",
    "ravel",
    "ravel_multi_index",
    "relu",
    "selu",
    "sigmoid",
    "silu",
    "softplus",
    "softsign",
    "tanhshrink",
    "tiny",
    "unravel_index",
]
