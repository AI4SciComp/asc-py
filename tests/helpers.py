# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Backend factories used by parametrized semantic tests."""

from __future__ import annotations

import os
import typing

import array_api_compat.numpy as numpy_namespace
import array_api_strict
import numpy as numpy_library

from asc import backend as select_backend
from asc import typing as asc_typing

PROFILE: typing.Final[str] = os.environ.get("ASC_TEST_PROFILE", "all")
_PROFILE_BACKENDS: typing.Final[
    dict[str, tuple[asc_typing.BackendName, ...]]
] = {
    "base": ("numpy",),
    "torch": ("numpy", "torch"),
    "jax": ("jax", "numpy"),
    "all": ("jax", "numpy", "torch"),
}
if PROFILE not in _PROFILE_BACKENDS:
    raise RuntimeError(f"unsupported ASC_TEST_PROFILE {PROFILE!r}")
NATIVE_BACKENDS: typing.Final[tuple[asc_typing.BackendName, ...]] = (
    _PROFILE_BACKENDS[PROFILE]
)
BACKENDS: typing.Final[tuple[asc_typing.BackendName, ...]] = (
    "array_api_strict",
    *NATIVE_BACKENDS,
)

array_api_strict.set_array_api_strict_flags(api_version="2024.12")


class TestNamespace(asc_typing.ArrayNamespace, typing.Protocol):
    """Creation surface needed only by backend-parametrized tests."""

    bool: object
    float32: object
    float64: object
    int8: object
    int16: object
    int32: object
    uint8: object

    def asarray(
        self, value: object, *, dtype: object | None = None
    ) -> asc_typing.NativeArray:
        """Create a native array from test data."""
        ...  # pylint: disable=unnecessary-ellipsis

    def empty(
        self,
        shape: asc_typing.Shape,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> asc_typing.NativeArray:
        """Create an uninitialized native test array."""
        ...  # pylint: disable=unnecessary-ellipsis


def namespace(backend: asc_typing.BackendName) -> TestNamespace:
    """Return the native or compatibility namespace for a test backend."""
    if backend == "array_api_strict":
        selected = array_api_strict
    elif backend == "jax":
        import jax.numpy

        selected = jax.numpy
    elif backend == "numpy":
        selected = numpy_namespace
    else:
        selected = select_backend("torch").xp
    return typing.cast(TestNamespace, selected)


def has_backend(backend: str) -> bool:
    """Return whether a backend belongs to the active test profile."""
    return backend in NATIVE_BACKENDS


def float_array(
    backend: asc_typing.BackendName, value: object
) -> asc_typing.NativeArray:
    """Create a float32 array in one backend."""
    selected = namespace(backend)
    return selected.asarray(value, dtype=selected.float32)


def int_array(
    backend: asc_typing.BackendName, value: object
) -> asc_typing.NativeArray:
    """Create an int32 array in one backend."""
    selected = namespace(backend)
    return selected.asarray(value, dtype=selected.int32)


def as_numpy(
    value: object,
) -> numpy_library.ndarray[
    tuple[int, ...], numpy_library.dtype[numpy_library.float64]
]:
    """Copy a CPU result to NumPy only for test assertions."""
    return numpy_library.asarray(value, dtype=numpy_library.float64)
