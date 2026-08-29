# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, replayable backend-native random generation."""

from __future__ import annotations

import math
import typing
import warnings

from asc import _array_api_compat, config, errors
from asc import typing as asc_typing
from asc.backends import _state
from asc.core import namespace as namespace_module
from asc.core.device import is_cpu_device
from asc.extensions import _dispatch

_RANDOM_FAILURES = (RuntimeError, TypeError, ValueError, Warning)
_KEY_FAILURES = (OverflowError, *_RANDOM_FAILURES)


class _RandomAdapter(typing.Protocol):
    """Random operations implemented by a lazy backend adapter."""

    def create_key(self, seed: int, *, device: object | None = None) -> object:
        """Create explicit backend state."""
        ...  # pylint: disable=unnecessary-ellipsis

    def owns_key(self, key: object) -> bool:
        """Return whether this adapter owns a state value."""
        ...  # pylint: disable=unnecessary-ellipsis

    def uniform(
        self,
        shape: asc_typing.Shape,
        *,
        key: object,
        low: float,
        high: float,
        dtype: object | None,
    ) -> tuple[asc_typing.NativeArray, object]:
        """Sample an array and return advanced state."""
        ...  # pylint: disable=unnecessary-ellipsis


def _adapter(backend: asc_typing.BackendName) -> _RandomAdapter:
    module = _dispatch.load_backend(backend)
    return typing.cast(_RandomAdapter, module)


def _validated_shape(shape: object) -> asc_typing.Shape:
    if not isinstance(shape, tuple):
        raise errors.RandomStateError("random.uniform: shape must be a tuple")
    raw_shape = typing.cast(tuple[object, ...], shape)
    if any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
        for extent in raw_shape
    ):
        raise errors.RandomStateError(
            "random.uniform: shape extents must be non-negative integers"
        )
    return typing.cast(asc_typing.Shape, raw_shape)


def _validated_seed(seed: object) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**32
    ):
        raise errors.RandomStateError(
            "random.create_key: seed must be a non-negative 32-bit integer"
        )
    return seed


def create_key(seed: int, *, context: config.CreationContext) -> object:
    """Create explicit random state for a selected backend.

    Args:
        seed: Non-negative Python integer seed.
        context: Explicit backend selection.

    Returns:
        Backend-specific immutable state or a native JAX key.

    Raises:
        RandomStateError: If the seed is invalid.
        UnsupportedCapabilityError: If the backend has no random extension.
    """
    validated_seed = _validated_seed(seed)
    if context.backend == "array_api_strict":
        raise errors.UnsupportedCapabilityError(
            "random.create_key: array-api-strict does not provide random state"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            key = _adapter(context.backend).create_key(
                validated_seed,
                device=context.device,
            )
    except _KEY_FAILURES as exception:
        raise errors.RandomStateError(
            "random.create_key: backend rejected the seed or CPU device"
        ) from exception
    if context.backend == "jax":
        key_backend = _key_backend(key)
        if key_backend != context.backend:
            raise errors.RandomStateError(
                "random.create_key: backend returned incompatible state"
            )
    return key


def _key_backend(key: object) -> asc_typing.BackendName:
    if isinstance(key, _state.CounterKey):
        return key.backend
    try:
        namespace = _array_api_compat.compat.array_namespace(
            key,
            api_version=namespace_module.ARRAY_API_VERSION,
        )
        namespace_module.validate_namespace_revision(namespace)
        device = _array_api_compat.compat.device(key)
    except (AttributeError, TypeError, ValueError) as exception:
        raise errors.RandomStateError(
            "random.uniform: key is not recognized backend state"
        ) from exception
    backend = namespace_module.identify_backend(namespace)
    if backend != "jax" or not is_cpu_device(device):
        raise errors.RandomStateError(
            "random.uniform: native array keys must belong to JAX on CPU"
        )
    return backend


def uniform(
    shape: asc_typing.Shape,
    *,
    key: object,
    low: float = 0.0,
    high: float = 1.0,
    dtype: object | None = None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a uniform native array and return advanced random state.

    Args:
        shape: Tuple of non-negative extents.
        key: State produced by :func:`create_key` or the previous call.
        low: Inclusive lower distribution bound.
        high: Exclusive upper distribution bound.
        dtype: Optional backend-native floating dtype.

    Returns:
        ``(sample, advanced_key)`` in the key's backend.

    Raises:
        RandomStateError: If shape, bounds, or key are invalid.
    """
    validated_shape = _validated_shape(shape)
    try:
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, (int, float))
            or not isinstance(high, (int, float))
        ):
            raise TypeError
        normalized_low = float(low)
        normalized_high = float(high)
    except (OverflowError, TypeError, ValueError) as exception:
        raise errors.RandomStateError(
            "random.uniform: low and high must be finite real values with "
            "low strictly less than high"
        ) from exception
    if (
        not math.isfinite(normalized_low)
        or not math.isfinite(normalized_high)
        or normalized_low >= normalized_high
    ):
        raise errors.RandomStateError(
            "random.uniform: low and high must be finite real values with "
            "low strictly less than high"
        )
    backend = _key_backend(key)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sample, advanced_key = _adapter(backend).uniform(
                validated_shape,
                key=key,
                low=normalized_low,
                high=normalized_high,
                dtype=dtype,
            )
    except _RANDOM_FAILURES as exception:
        raise errors.RandomStateError(
            "random.uniform: backend rejected the key, dtype, or shape"
        ) from exception
    sample_namespace = namespace_module.array_namespace(sample)
    if namespace_module.identify_backend(sample_namespace) != backend:
        raise errors.RandomStateError(
            "random.uniform: backend returned an incompatible sample"
        )
    if dtype is not None and sample.dtype != dtype:
        raise errors.RandomStateError(
            "random.uniform: backend did not preserve the requested dtype"
        )
    if _key_backend(advanced_key) != backend:
        raise errors.RandomStateError(
            "random.uniform: backend returned incompatible state"
        )
    return sample, advanced_key


__all__ = ["create_key", "uniform"]
