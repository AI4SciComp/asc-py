# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe portable one-dimensional signal helpers."""

from __future__ import annotations

import typing

from asc import _array_api_compat, errors
from asc.core import _dtype
from asc.core import namespace as namespace_module


def _normalize_axis(axis: int, ndim: int, operation: str) -> int:
    """Validate and normalize one positional axis."""
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise ValueError(f"{operation}: axis must be a non-Boolean integer")
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"{operation}: axis is out of bounds")
    return normalized


def convolve1d(
    array: object,
    kernel: object,
    *,
    axis: int = -1,
    mode: typing.Literal["valid", "same", "full"] = "valid",
) -> object:
    """Convolve along one axis using only native Array API operations."""
    xp = namespace_module.array_namespace(array, kernel)
    if len(kernel.shape) != 1:
        raise ValueError("convolve1d: kernel must be one-dimensional")
    ndim = len(array.shape)
    normalized_axis = _normalize_axis(axis, ndim, "convolve1d")
    input_size = array.shape[normalized_axis]
    if input_size < 1:
        raise ValueError("convolve1d: input signal must not be empty")
    kernel_size = kernel.shape[0]
    if kernel_size < 1:
        raise ValueError("convolve1d: kernel must not be empty")
    if mode not in {"valid", "same", "full"}:
        raise ValueError("convolve1d: mode must be 'valid', 'same', or 'full'")
    result_dtype = _dtype.extension_result_type(
        xp, array, kernel, operation="convolve1d"
    )
    values = xp.astype(array, result_dtype, copy=True)
    weights = xp.astype(kernel, result_dtype, copy=True)
    values = xp.moveaxis(values, normalized_axis, -1)
    if mode == "valid" and input_size < kernel_size:
        flipped_values = xp.flip(values, axis=-1)
        output_size = kernel_size - input_size + 1
        outputs = [
            xp.sum(
                flipped_values * weights[index : index + input_size],
                axis=-1,
                dtype=result_dtype,
            )
            for index in range(output_size)
        ]
        result = xp.stack(tuple(outputs), axis=-1)
        return xp.moveaxis(result, -1, normalized_axis)
    if mode == "full":
        widths = ((0, 0),) * (ndim - 1) + ((kernel_size - 1, kernel_size - 1),)
    elif mode == "same":
        output_size = max(input_size, kernel_size)
        total = output_size + kernel_size - 1 - input_size
        widths = ((0, 0),) * (ndim - 1) + ((total - total // 2, total // 2),)
    else:
        widths = ((0, 0),) * ndim
    if mode != "valid":
        from asc.ops import pad  # Local import avoids an initialization cycle.

        values = pad(values, widths, mode="constant")
    output_size = values.shape[-1] - kernel_size + 1
    flipped = xp.flip(weights, axis=0)
    outputs = [
        xp.sum(
            values[..., index : index + kernel_size] * flipped,
            axis=-1,
            dtype=result_dtype,
        )
        for index in range(output_size)
    ]
    if not outputs:
        shape = (*values.shape[:-1], 0)
        result = xp.empty(
            shape,
            dtype=result_dtype,
            device=_array_api_compat.compat.device(array),
        )
    else:
        result = xp.stack(tuple(outputs), axis=-1)
    return xp.moveaxis(result, -1, normalized_axis)


def moving_mean(array: object, window: int, *, axis: int = -1) -> object:
    """Return a valid moving arithmetic mean along one axis."""
    xp = namespace_module.array_namespace(array)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("moving_mean: window must be a positive integer")
    if not xp.isdtype(array.dtype, ("real floating", "complex floating")):
        raise errors.DTypeError(
            "moving_mean: array must have a floating or complex dtype"
        )
    ndim = len(array.shape)
    normalized_axis = _normalize_axis(axis, ndim, "moving_mean")
    if window > array.shape[normalized_axis]:
        raise ValueError(
            "moving_mean: window must not exceed the selected axis length"
        )
    calculation_dtype = (
        xp.float32 if int(xp.finfo(array.dtype).bits) < 32 else array.dtype
    )
    calculation = xp.astype(array, calculation_dtype, copy=True)
    kernel = xp.full(
        (window,),
        1.0 / window,
        dtype=calculation_dtype,
        device=_array_api_compat.compat.device(array),
    )
    result = convolve1d(calculation, kernel, axis=normalized_axis, mode="valid")
    return xp.astype(result, array.dtype, copy=True)


__all__ = ["convolve1d", "moving_mean"]
