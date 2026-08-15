# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Small, pure operations that exercise the portability contract."""

from __future__ import annotations

import warnings
from typing import cast

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.core import namespace as namespace_module
from asc.core._scalar import require_representable_scalar
from asc.core.device import is_cpu_device
from asc.extensions import _dispatch


def _validated_shape(shape: object) -> asc_typing.Shape:
    if not isinstance(shape, tuple):
        raise errors.ContextError("create_full: shape must be a tuple")
    raw_shape = cast(tuple[object, ...], shape)
    for extent in raw_shape:
        if (
            isinstance(extent, bool)
            or not isinstance(extent, int)
            or extent < 0
        ):
            raise errors.ContextError(
                "create_full: shape extents must be non-negative integers"
            )
    return cast(asc_typing.Shape, raw_shape)


def _validated_reduction_controls(
    axis: object, keepdims: object
) -> tuple[asc_typing.Axis, bool]:
    """Validate portable reduction controls before selecting a backend."""
    if not isinstance(keepdims, bool):
        raise TypeError("sum_of_squares: keepdims must be a Boolean")
    if axis is None:
        return None, keepdims
    if isinstance(axis, int) and not isinstance(axis, bool):
        return axis, keepdims
    if isinstance(axis, tuple) and all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in axis
    ):
        return cast(tuple[int, ...], axis), keepdims
    raise TypeError(
        "sum_of_squares: axis must be an integer, tuple of integers, or None"
    )


def _normalized_reduction_axis(
    axis: asc_typing.Axis, ndim: int
) -> asc_typing.Axis:
    """Normalize valid axes and reject duplicates before native reduction."""
    if axis is None:
        return None
    axes = (axis,) if isinstance(axis, int) else axis
    normalized = tuple(entry + ndim if entry < 0 else entry for entry in axes)
    if any(entry < 0 or entry >= ndim for entry in normalized):
        raise ValueError("sum_of_squares: axis is out of bounds")
    if len(set(normalized)) != len(normalized):
        raise ValueError("sum_of_squares: axis entries must be unique")
    if isinstance(axis, int):
        return normalized[0]
    return normalized


def create_full(
    shape: asc_typing.Shape,
    fill_value: object,
    *,
    context: asc_typing.CreationContextLike,
) -> asc_typing.NativeArray:
    """Create a native array using an explicit immutable context.

    Args:
        shape: Tuple of non-negative extents.
        fill_value: Value used for every element.
        context: Explicit namespace, backend, dtype, and CPU device.

    Returns:
        A native array from the context namespace.

    Raises:
        ContextError: If the shape or context result violates the contract.
    """
    validated_shape = _validated_shape(shape)
    namespace_module.validate_namespace_revision(context.namespace)
    observed_backend = namespace_module.identify_backend(context.namespace)
    if observed_backend != context.backend:
        raise errors.ContextError(
            "create_full: context backend does not match its namespace; "
            f"expected {context.backend!r}, observed {observed_backend!r}"
        )
    if not is_cpu_device(context.device):
        raise errors.ContextError("create_full: only CPU devices are supported")
    if _array_api_compat.compat.is_array_api_obj(fill_value):
        raise errors.ContextError(
            "create_full: array-valued fill values are not supported"
        )

    adapter = None
    if observed_backend != "array_api_strict":
        adapter = _dispatch.load_backend(observed_backend)
        if context.dtype is not None:
            try:
                adapter.validate_dtype(context.dtype)
            except errors.AscError as exception:
                raise errors.ContextError(
                    "create_full: backend rejected the requested dtype, "
                    "device, shape, or fill value"
                ) from exception
    try:
        require_representable_scalar(
            context.namespace, context.dtype, fill_value, "create_full"
        )
    except errors.DTypeError as exception:
        raise errors.ContextError(
            "create_full: fill value is not representable in the requested "
            "dtype"
        ) from exception

    resolved_device = context.device
    if adapter is not None:
        resolver = getattr(adapter, "resolve_device", None)
        if resolver is not None:
            try:
                resolved_device = resolver(context.device)
            except errors.AscError as exception:
                raise errors.ContextError(
                    "create_full: context device cannot be represented by "
                    f"backend {observed_backend!r}"
                ) from exception
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = cast(
                asc_typing.NativeArray,
                context.namespace.full(
                    validated_shape,
                    fill_value,
                    dtype=context.dtype,
                    device=resolved_device,
                ),
            )
    except errors.AscError as exception:
        raise errors.ContextError(
            "create_full: backend rejected the requested dtype, device, "
            "shape, or fill value"
        ) from exception
    except (
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        Warning,
    ) as exception:
        raise errors.ContextError(
            "create_full: backend rejected the requested dtype, device, "
            "shape, or fill value"
        ) from exception
    try:
        result_namespace = namespace_module.array_namespace(result)
        result_backend = namespace_module.identify_backend(result_namespace)
    except errors.AscError as exception:
        raise errors.ContextError(
            "create_full: result is not a supported dense numeric CPU array"
        ) from exception
    if result_backend != context.backend:
        raise errors.ContextError(
            "create_full: namespace returned an array from another backend"
        )
    if (
        context.dtype is not None
        and getattr(result, "dtype", None) != context.dtype
    ):
        raise errors.ContextError(
            "create_full: backend did not preserve the requested dtype"
        )
    if not is_cpu_device(_array_api_compat.compat.device(result)):
        raise errors.ContextError(
            "create_full: backend returned a non-CPU array"
        )
    return result


def sum_of_squares(
    array: asc_typing.ArrayT,
    *,
    axis: asc_typing.Axis = None,
    keepdims: bool = False,
) -> asc_typing.ArrayT:
    """Sum squared elements of a real floating native array.

    Args:
        array: Native NumPy, PyTorch, JAX, or array-api-strict array.
        axis: Axis or axes to reduce, or ``None`` for every axis.
        keepdims: Whether reduced axes remain with size one.

    Returns:
        A native, zero-dimensional array for a complete reduction.

    Raises:
        DTypeError: If ``array`` is not real floating point.
        NamespaceError: If ``array`` is not a supported native array.
        TypeError: If ``axis`` or ``keepdims`` has an invalid type.
    """
    axis, keepdims = _validated_reduction_controls(axis, keepdims)
    namespace = namespace_module.array_namespace(array)
    axis = _normalized_reduction_axis(axis, len(array.shape))
    dtype = getattr(array, "dtype", None)
    if dtype is None or not namespace.isdtype(dtype, "real floating"):
        raise errors.DTypeError(
            "sum_of_squares: expected a real floating-point array"
        )
    squared = namespace.square(array)
    return cast(
        asc_typing.ArrayT,
        namespace.sum(squared, axis=axis, keepdims=keepdims),
    )
