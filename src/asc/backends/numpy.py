# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Lazy NumPy extension adapter."""

from __future__ import annotations

import dataclasses
import math
import typing

import array_api_compat.numpy as array_api_namespace
import numpy

from asc import errors
from asc import typing as asc_typing
from asc.backends import _namespace, _state
from asc.core import _dtype


class _NumpyNamespace:
    """Forward the frozen namespace after validating created dtypes."""

    __name__ = array_api_namespace.__name__
    __array_api_version__ = array_api_namespace.__array_api_version__

    def __getattr__(self, name: str) -> object:
        return _namespace.checked_attribute(
            array_api_namespace,
            name,
            "numpy",
            validate_dtype,
            resolve_device,
        )


_NAMESPACE = _NumpyNamespace()


def namespace() -> object:
    """Return the frozen compatibility namespace without allocation."""
    return _NAMESPACE


def einsum(subscripts: str, *operands: object) -> object:
    """Evaluate Einstein summation through the private native extension."""
    return numpy.einsum(subscripts, *operands)


def kron(first: object, second: object) -> object:
    """Evaluate a Kronecker product through the private native extension."""
    return numpy.kron(first, second)


def resolve_device(device: object | None) -> object | None:
    """Normalize the only NumPy device accepted by the standard surface."""
    if device is None or (type(device) is str and device == "cpu"):
        return None
    raise errors.DeviceError("numpy backend: only the CPU device is supported")


def validate_dtype(dtype: object) -> None:
    """Reject dtypes outside the frozen NumPy release surface."""
    try:
        native_dtype = numpy.dtype(dtype)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.DTypeError(
            "numpy backend: dtype is outside the supported release surface"
        ) from exception
    if (
        dtype is not native_dtype and dtype is not native_dtype.type
    ) or native_dtype.metadata is not None:
        raise errors.DTypeError(
            "numpy backend: dtype is outside the supported release surface"
        )
    _dtype.require_supported_dtype("numpy", native_dtype, "numpy backend")


def to_cpu(value: object) -> object:
    """Return an already-host-resident NumPy array."""
    return value


def create_key(seed: int, *, device: object | None = None) -> object:
    """Create replayable immutable NumPy random state."""
    del device
    return _state.CounterKey("numpy", seed)


def owns_key(key: object) -> bool:
    """Return whether ``key`` is NumPy state."""
    return isinstance(key, _state.CounterKey) and key.backend == "numpy"


def _generator(key: object) -> tuple[numpy.random.Generator, _state.CounterKey]:
    if not owns_key(key):
        raise errors.RandomStateError("numpy random: incompatible state")
    native_key = typing.cast(_state.CounterKey, key)
    if not 0 <= native_key.counter < 2**32:
        raise errors.RandomStateError(
            "numpy random: counter must fit in an unsigned 32-bit integer"
        )
    if native_key.counter == 2**32 - 1:
        raise errors.RandomStateError(
            "numpy random: state is exhausted and cannot be advanced"
        )
    sequence = numpy.random.SeedSequence([native_key.seed, native_key.counter])
    return numpy.random.default_rng(sequence), native_key


def _advanced(key: _state.CounterKey) -> _state.CounterKey:
    if key.counter >= 2**32 - 1:
        raise errors.RandomStateError(
            "numpy random: state is exhausted and cannot be advanced"
        )
    return dataclasses.replace(key, counter=key.counter + 1)


def _random_dtype(
    dtype: object | None,
    default: object,
    family: object,
    operation: str,
) -> numpy.dtype[typing.Any]:
    """Return one canonical release dtype on the random error surface."""
    requested = default if dtype is None else dtype
    try:
        validate_dtype(requested)
    except errors.DTypeError as exception:
        raise errors.RandomStateError(
            f"numpy {operation}: requested dtype is outside the supported "
            "release surface"
        ) from exception
    native_dtype = numpy.dtype(requested)
    if not numpy.issubdtype(native_dtype, family):
        requirement = (
            "a real floating dtype"
            if family is numpy.floating
            else "a signed integer dtype"
        )
        raise errors.RandomStateError(
            f"numpy {operation}: dtype must be {requirement}"
        )
    return native_dtype


def uniform(
    shape: asc_typing.Shape,
    *,
    key: object,
    low: float,
    high: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample NumPy without process-global random state."""
    generator, native_key = _generator(key)
    native_dtype = _random_dtype(
        dtype, numpy.float64, numpy.floating, "uniform"
    )
    information = numpy.finfo(native_dtype)
    minimum = float(information.min)  # pylint: disable=no-member
    maximum = float(information.max)  # pylint: disable=no-member
    if low < minimum or high > maximum or high - low > maximum:
        raise errors.RandomStateError(
            "numpy random: interval is not representable in the requested dtype"
        )
    low_value = numpy.asarray(low, dtype=native_dtype)
    high_value = numpy.asarray(high, dtype=native_dtype)
    if float(low_value) != low or float(high_value) != high:
        raise errors.RandomStateError(
            "numpy random: interval endpoints must be exactly representable "
            "in the requested dtype"
        )
    sample = generator.uniform(low, high, size=shape)
    result = numpy.asarray(sample, dtype=native_dtype)
    negative_infinity = numpy.asarray(-numpy.inf, dtype=native_dtype)
    upper = numpy.nextafter(high_value, negative_infinity)
    result = numpy.clip(result, low_value, upper)
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def normal(
    shape: asc_typing.Shape,
    *,
    key: object,
    mean: float,
    std: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a NumPy normal distribution from explicit state."""
    generator, native_key = _generator(key)
    native_dtype = _random_dtype(dtype, numpy.float64, numpy.floating, "normal")
    with numpy.errstate(over="ignore"):
        result = numpy.asarray(
            generator.normal(mean, std, shape), dtype=native_dtype
        )
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def truncated_normal(
    shape: asc_typing.Shape,
    *,
    key: object,
    mean: float,
    std: float,
    lower: float,
    upper: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a NumPy truncated normal with bounded rejection work."""
    generator, native_key = _generator(key)
    native_dtype = _random_dtype(
        dtype, numpy.float64, numpy.floating, "truncated_normal"
    )
    standardized_lower = (lower - mean) / std
    standardized_upper = (upper - mean) / std
    if not (
        math.isfinite(standardized_lower)
        and math.isfinite(standardized_upper)
        and standardized_lower < standardized_upper
    ):
        raise errors.RandomStateError(
            "numpy truncated_normal: standardized bounds must be finite and "
            "ordered"
        )
    count = math.prod(shape)
    standard = numpy.empty(count, dtype=numpy.float64)
    remaining = numpy.arange(count)
    reflected = standardized_upper <= 0.0
    if reflected:
        proposal_lower = -standardized_upper
        proposal_upper = -standardized_lower
    else:
        proposal_lower = standardized_lower
        proposal_upper = standardized_upper
    for _ in range(1024):
        if remaining.size == 0:
            break
        size = remaining.size
        if proposal_lower >= 0.0:
            alpha = 0.5 * (proposal_lower + math.hypot(proposal_lower, 2.0))
            mass = -math.expm1(-alpha * (proposal_upper - proposal_lower))
            proposal = (
                proposal_lower
                - numpy.log1p(-generator.random(size) * mass) / alpha
            )
            accepted = generator.random(size) <= numpy.exp(
                -0.5 * numpy.square(proposal - alpha)
            )
        elif proposal_upper - proposal_lower <= 2.0:
            proposal = generator.uniform(
                proposal_lower, proposal_upper, size=size
            )
            accepted = generator.random(size) <= numpy.exp(
                -0.5 * numpy.square(proposal)
            )
        else:
            proposal = generator.normal(size=size)
            accepted = (proposal >= proposal_lower) & (
                proposal <= proposal_upper
            )
        selected = -proposal[accepted] if reflected else proposal[accepted]
        standard[remaining[accepted]] = selected
        remaining = remaining[~accepted]
    if remaining.size:
        raise errors.RandomStateError(
            "numpy truncated_normal: sampling did not converge"
        )
    result = numpy.asarray(
        mean + std * standard.reshape(shape), dtype=native_dtype
    )
    lower_value = numpy.asarray(lower, dtype=native_dtype)
    upper_value = numpy.asarray(upper, dtype=native_dtype)
    result = numpy.clip(result, lower_value, upper_value)
    return (
        typing.cast(asc_typing.NativeArray, result),
        _advanced(native_key),
    )


def randint(
    shape: asc_typing.Shape,
    *,
    key: object,
    low: int,
    high: int,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample signed NumPy integers from explicit state."""
    generator, native_key = _generator(key)
    native_dtype = _random_dtype(
        dtype, numpy.int64, numpy.signedinteger, "randint"
    )
    result = generator.integers(low, high, shape, dtype=native_dtype)
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def bernoulli(
    shape: asc_typing.Shape,
    *,
    key: object,
    probability: float,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample Boolean NumPy Bernoulli values from explicit state."""
    generator, native_key = _generator(key)
    result = generator.random(shape) < probability
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def gamma(
    shape: asc_typing.Shape,
    *,
    key: object,
    concentration: float,
    scale: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a NumPy gamma distribution from explicit state."""
    generator, native_key = _generator(key)
    native_dtype = _random_dtype(dtype, numpy.float64, numpy.floating, "gamma")
    with numpy.errstate(over="ignore"):
        result = numpy.asarray(
            generator.gamma(concentration, scale, shape), dtype=native_dtype
        )
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def exponential(
    shape: asc_typing.Shape,
    *,
    key: object,
    scale: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a NumPy exponential distribution from explicit state."""
    generator, native_key = _generator(key)
    native_dtype = _random_dtype(
        dtype, numpy.float64, numpy.floating, "exponential"
    )
    with numpy.errstate(over="ignore"):
        result = numpy.asarray(
            generator.exponential(scale, shape), dtype=native_dtype
        )
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def choice(
    population: object,
    shape: asc_typing.Shape,
    *,
    key: object,
    replace: bool,
    probabilities: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample from a NumPy population using explicit state."""
    generator, native_key = _generator(key)
    result = generator.choice(
        population, size=shape, replace=replace, p=probabilities
    )
    return typing.cast(
        asc_typing.NativeArray, numpy.asarray(result)
    ), _advanced(native_key)


def permutation(
    value: object, *, key: object
) -> tuple[asc_typing.NativeArray, object]:
    """Return a functional NumPy permutation."""
    generator, native_key = _generator(key)
    result = generator.permutation(value)
    return typing.cast(
        asc_typing.NativeArray, numpy.asarray(result)
    ), _advanced(native_key)


def split_key(key: object, count: int) -> tuple[object, ...]:
    """Derive deterministic independent NumPy child states."""
    _, native_key = _generator(key)
    sequence = numpy.random.SeedSequence([native_key.seed, native_key.counter])
    return tuple(
        _state.CounterKey("numpy", int(child.generate_state(1)[0]))
        for child in sequence.spawn(count)
    )


def index_add(
    array: asc_typing.NativeArray,
    indices: asc_typing.NativeArray,
    values: asc_typing.NativeArray,
    *,
    axis: int,
    update_shape: asc_typing.Shape,
) -> asc_typing.NativeArray:
    """Apply NumPy ``add.at`` to a private copy."""
    native_array = typing.cast(
        numpy.ndarray[tuple[int, ...], numpy.dtype], array
    )
    native_indices = typing.cast(
        numpy.ndarray[tuple[int, ...], numpy.dtype], indices
    )
    native_values = typing.cast(
        numpy.ndarray[tuple[int, ...], numpy.dtype], values
    )
    if native_indices.dtype.kind != "i":
        raise errors.IndexContractError(
            "index_add: NumPy indices must have a signed integer dtype"
        )
    if numpy.any(native_indices < 0) or numpy.any(
        native_indices >= native_array.shape[axis]
    ):
        raise errors.IndexContractError("index_add: index is out of bounds")
    broadcast_values = numpy.broadcast_to(native_values, update_shape)
    moved = numpy.moveaxis(numpy.array(native_array, copy=True), axis, 0)
    moved_values = numpy.moveaxis(broadcast_values, axis, 0)
    numpy.add.at(moved, native_indices, moved_values)
    result = numpy.moveaxis(moved, 0, axis)
    return typing.cast(asc_typing.NativeArray, result)


def index_update(
    array: asc_typing.NativeArray,
    indices: asc_typing.NativeArray,
    values: asc_typing.NativeArray,
    *,
    axis: int,
    update_shape: asc_typing.Shape,
    reduction: str,
) -> asc_typing.NativeArray:
    """Apply one functional indexed update to a private NumPy copy."""
    native_array = typing.cast(numpy.ndarray, array)
    native_indices = typing.cast(numpy.ndarray, indices)
    native_values = typing.cast(numpy.ndarray, values)
    if native_indices.dtype.kind != "i":
        raise errors.IndexUpdateError(
            "index update: NumPy indices must be signed integers"
        )
    if numpy.any(native_indices < 0) or numpy.any(
        native_indices >= native_array.shape[axis]
    ):
        raise errors.IndexUpdateError("index update: index is out of bounds")
    if (
        reduction == "set"
        and numpy.unique(native_indices).size != native_indices.size
    ):
        raise errors.DuplicateIndexError(
            "index_set: duplicate indices have no deterministic set policy"
        )
    broadcast_values = numpy.broadcast_to(native_values, update_shape)
    moved = numpy.moveaxis(numpy.array(native_array, copy=True), axis, 0)
    moved_values = numpy.moveaxis(broadcast_values, axis, 0)
    if reduction == "set":
        moved[native_indices] = moved_values
    else:
        operation = {
            "add": numpy.add,
            "multiply": numpy.multiply,
            "min": numpy.minimum,
            "max": numpy.maximum,
        }[reduction]
        operation.at(moved, native_indices, moved_values)
    return typing.cast(asc_typing.NativeArray, numpy.moveaxis(moved, 0, axis))
